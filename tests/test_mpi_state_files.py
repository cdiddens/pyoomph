#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  @author Maxim de Wildt <m.dewildt@utwente.nl>
#
#  @section LICENSE
#
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC
#  Copyright (C) 2021-2026  Christian Diddens, Duarte Rocha & Maxim de Wildt
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#  The main author may be contacted at c.diddens@utwente.nl
#
# ========================================================================

# State files of distributed problems (dev_docs/distributed_state_files.md).
#
# A state file used to be rank-local in every respect - the refinement pattern is numbered within the
# local mesh, the nodal data is written in the order the local elements happen to be walked in - so
# save_state/load_state simply refused on a distributed problem, which also took out --runmode continue
# and the dedicated plotting subprocess under MPI. The file now addresses elements and nodes
# structurally, and the tests below are what that claim means:
#
#   - a serial round trip, which catches format bugs without any MPI at all;
#   - serial writes read by 2/3/4 processes and the other way round. This is the actual promise: the
#     file does not mention the partition, so anything can read anything;
#   - byte-identical files from serial, np=3 and np=4 for the same data. The strongest form of the
#     same statement, and it also pins the record ORDER down, not just the content;
#   - the same with adaptive refinement, loaded into a mesh that never adapted, so the refinement tree
#     really has to be replayed rather than being there already. A uniformly refined mesh would pass
#     even with a broken signature;
#   - a load must not silently do nothing: the worker overwrites every nodal value with -12345 first.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_state_file_worker.py")


def _mpi_reason():
    if shutil.which("mpirun") is None:
        return "mpirun not found"
    try:
        from pyoomph.generic.mpi import has_mpi, have_pymetis
        if not has_mpi():
            return "pyoomph was built without MPI"
        if not have_pymetis():
            return "PyMetis is not installed, so the mesh cannot be partitioned"
    except Exception as e:
        return "MPI unavailable: " + str(e)
    return None


_SKIP_REASON = _mpi_reason()
pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]

# The fingerprint is a sum over all nodes, so it is reproduced up to the order the ranks are summed in
_FINGERPRINT_RTOL = 1e-10


def _run(nproc, tmpdir, mode, fname, resave_to=None, adapt=False, levels=2, adapt_rounds=2,
         refine_to_level=0, distribute=True, timeout=900):
    """Run the worker, in-process when nproc is 1, under mpirun otherwise."""
    if nproc == 1:
        sys.path.insert(0, _HERE)
        try:
            import mpi_state_file_worker
            return [mpi_state_file_worker.run_case(mode, fname, resave_to=resave_to, adapt=adapt,
                                                   levels=levels, adapt_rounds=adapt_rounds,
                                                   refine_to_level=refine_to_level,
                                                   outdir=os.path.join(str(tmpdir), "serial"))]
        finally:
            sys.path.remove(_HERE)
    cmd = ["mpirun", "-n", str(nproc), sys.executable, _WORKER, mode, fname,
           "--outdir", os.path.join(str(tmpdir), "out" + str(nproc)),
           "--levels", str(levels), "--adapt-rounds", str(adapt_rounds),
           "--refine-to-level", str(refine_to_level)]
    if resave_to is not None:
        cmd += ["--resave-to", resave_to]
    if adapt:
        cmd += ["--adapt"]
    if distribute:
        cmd += ["--distribute"]
    # Importing pyoomph calls MPI_Init, so THIS pytest process is already a (singleton) MPI job and
    # owns an Open MPI session directory under TMPDIR. A nested mpirun collides with it. Give the child
    # its own TMPDIR.
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        # Bounded on purpose: writing is collective, and a rank that does not reach it waits forever.
        # That has to surface as a FAILURE, not as a suite that never finishes.
        raise AssertionError(
            "mpirun did not finish within %d s -- suspect a deadlock in the state file handling "
            "(nproc=%d mode=%s adapt=%s).\n--- stdout tail ---\n%s"
            % (timeout, nproc, mode, adapt, (e.stdout or "")[-3000:]))
    per_rank = []
    for line in proc.stdout.splitlines():
        if line.startswith("PYOOMPH_MPI_RESULT "):
            per_rank.append(json.loads(line[len("PYOOMPH_MPI_RESULT "):]))
    if not per_rank:
        raise AssertionError(
            "no results from mpirun (exit %d)\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s"
            % (proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
    assert len(per_rank) == nproc, "reported from %d of %d ranks" % (len(per_rank), nproc)
    for r in per_rank:
        assert "error" not in r, "failed on rank %d: %s\n%s" % (r["rank"], r["error"], r.get("traceback", ""))
    return sorted(per_rank, key=lambda r: r["rank"])


def _assert_same_state(got, expected, what):
    a, b = got["fingerprint"], expected["fingerprint"]
    assert a[2] == b[2], "%s: %d nodal entries, expected %d" % (what, a[2], b[2])
    assert a[0] == pytest.approx(b[0], rel=_FINGERPRINT_RTOL), "%s: values landed on different nodes" % what
    assert a[1] == pytest.approx(b[1], rel=_FINGERPRINT_RTOL), "%s: values differ" % what


def test_serial_round_trip(tmp_path):
    fname = str(tmp_path / "serial.dump")
    saved = _run(1, tmp_path, "save", fname)[0]
    loaded = _run(1, tmp_path, "load", fname)[0]
    _assert_same_state(loaded, saved, "serial round trip")


@pytest.mark.parametrize("nproc", [2, 3, 4])
def test_serial_file_read_by_distributed_run(tmp_path, nproc):
    fname = str(tmp_path / "serial.dump")
    saved = _run(1, tmp_path, "save", fname)[0]
    per_rank = _run(nproc, tmp_path, "load", fname)
    assert per_rank[0]["distributed"], "the mesh was not distributed"
    _assert_same_state(per_rank[0], saved, "serial file on %d ranks" % nproc)


@pytest.mark.parametrize("nproc", [2, 3])
def test_distributed_file_read_serially(tmp_path, nproc):
    fname = str(tmp_path / ("dist%d.dump" % nproc))
    per_rank = _run(nproc, tmp_path, "save", fname)
    assert per_rank[0]["distributed"]
    loaded = _run(1, tmp_path, "load", fname)[0]
    _assert_same_state(loaded, per_rank[0], "file from %d ranks read serially" % nproc)


def test_file_is_independent_of_the_number_of_processes(tmp_path):
    # Written from identical data (all three loaded the same file), so any difference is the format
    # leaking the partition - which is the one thing it must not do.
    source = str(tmp_path / "source.dump")
    _run(1, tmp_path, "save", source)
    written = {}
    for nproc in (1, 3, 4):
        out = str(tmp_path / ("resaved%d.dump" % nproc))
        _run(nproc, tmp_path, "load", source, resave_to=out)
        with open(out, "rb") as f:
            written[nproc] = f.read()
    assert written[3] == written[1], "the file written on 3 ranks differs from the serial one"
    assert written[4] == written[1], "the file written on 4 ranks differs from the serial one"


def test_adaptive_refinement_is_replayed(tmp_path):
    # Saved from an adaptively refined mesh, loaded into runs that never adapted: the refinement tree
    # has to be rebuilt from the file, serially and distributed.
    fname = str(tmp_path / "adaptive.dump")
    saved = _run(1, tmp_path, "save", fname, adapt=True)[0]
    for nproc in (1, 3):
        loaded = _run(nproc, tmp_path, "load", fname)[0]
        assert loaded["nnode_before"] < saved["fingerprint"][2], "the loading run was already refined"
        _assert_same_state(loaded, saved, "adaptive state on %d rank(s)" % nproc)


def test_distributed_adaptive_round_trip(tmp_path):
    fname = str(tmp_path / "adaptive_dist.dump")
    per_rank = _run(3, tmp_path, "save", fname, adapt=True)
    loaded = _run(1, tmp_path, "load", fname)[0]
    _assert_same_state(loaded, per_rank[0], "adaptive state written on 3 ranks")


def test_uniformly_prerefined_distributed_mesh_is_addressable(tmp_path):
    """A mesh that is uniformly refined BEFORE it is distributed, which re-roots the forest.

    This is the configuration the tutorial scripts failed in and the round trips above do not reach.
    ``RefineToLevel`` raises the mesh's ``_initial_uniform_refinement_level``, and that refinement
    happens while the mesh is still whole - so ``Problem::distribute()`` partitions LEAVES of the
    refinement trees, and re-roots the forest at whatever each rank kept. Those new roots never went
    through ``assign_global_base_element_indices()``, which could only run before the distribution,
    so their ``global_base_index`` is -1.

    Two things used to go wrong at once, and both are asserted here:

      * the refinement signatures were keyed by that live root's number, so every re-rooted tree
        collapsed onto the sentinel -1 and the writer refused the file ("two processes describe
        different refinement trees for root element -1"). min_signature_root pins that directly, and
        measured on the pre-fix tree it is -1 on all four ranks with a single root each - exactly what
        droplet_spread_marangoni_and_gravity.py reported;
      * the refusal was raised on rank 0 only, from inside a section whose remaining collectives the
        other ranks had already committed to - so instead of failing the run HUNG, at 100% CPU on
        every rank, with a half-written state file. The timeout in _run is what makes that a failure
        rather than a suite that never returns.

    Note that refining AFTER the distribution is not enough: the trees keep the roots they were
    distributed with, and that is why the adaptive round trips above pass on the pre-fix tree too.

    See dev_docs/distributed_tutorial_failures.md.
    """
    fname = str(tmp_path / "prerefined.dump")
    per_rank = _run(4, tmp_path, "save", fname, adapt=True, levels=4, adapt_rounds=2, refine_to_level=2)
    assert per_rank[0]["distributed"], "the mesh was not distributed"
    for r in per_rank:
        assert r["min_signature_root"] >= 0, (
            "rank %d reports refinement signatures for root element %d: a tree whose root has no "
            "base-element number, so its signature is merged with every other such tree"
            % (r["rank"], r["min_signature_root"]))
    # ... and the file it wrote is a real one: read it back without the partition that wrote it.
    loaded = _run(1, tmp_path, "load", fname, levels=4, refine_to_level=2)[0]
    _assert_same_state(loaded, per_rank[0], "pre-refined state written on 4 ranks")
