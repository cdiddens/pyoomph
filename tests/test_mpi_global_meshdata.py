#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
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

# Globally merged mesh data of a distributed mesh (get_cached_mesh_data(..., global_mesh=True), see
# pyoomph/meshes/meshdatamerge.py).
#
# The reference is always the SERIAL run of the same worker: merging is only correct if the result is
# the mesh you would have had without --distribute. What each assertion is for:
#
#   - node count. Too many means nodes on the partition interface were not identified with each other
#     (the merged mesh would fall apart along the partition boundaries), too few means distinct nodes
#     were merged. Neither shows up in a plot of a smooth field, which is why it is checked first.
#   - the coordinate digest. Exact, because coordinates are copied node positions, not solver output.
#   - the element digest, over each element's set of node coordinates. This is what checks that the
#     connectivity still points at the right nodes after the renumbering.
#   - field statistics, which are what would catch values being attached to the wrong nodes.
#   - interface line segments. A purely topological quantity: per-rank data yields one segment per
#     partition, the merged data must yield the single line the serial run sees.
#   - the non-root ranks getting None with --distribute, and NOT getting None without it (where every
#     rank holds the whole mesh anyway).
#   - a repeated request, which must be served from the cache without leaving the other ranks waiting
#     in a gather that nobody joins.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_global_meshdata_worker.py")


def _mpi_reason():
    """None if a distributed run is possible here, else the reason to skip."""
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

# Field values pass through the linear solver, whose factorisation order depends on the partition, so
# they agree to round-off rather than exactly. Everything structural is compared exactly instead.
_FIELD_RTOL = 1e-9


def _run_mpi(nproc, tmpdir, distribute=True, discontinuous=False, twice=False, size=8, timeout=900):
    cmd = ["mpirun", "-n", str(nproc), sys.executable, _WORKER,
           "--outdir", os.path.join(str(tmpdir), "out"), "--size", str(size)]
    if discontinuous:
        cmd += ["--discontinuous"]
    if twice:
        cmd += ["--twice"]
    if distribute:
        cmd += ["--distribute"]
    # Importing pyoomph calls MPI_Init, so THIS pytest process is already a (singleton) MPI job and
    # owns an Open MPI session directory under TMPDIR. A nested mpirun collides with it and dies with
    # exit code 1 and no diagnostics. Give the child its own TMPDIR.
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        # Bounded on purpose: the merge is collective, and a rank that does not reach it (or reaches a
        # different one) waits forever. That has to surface as a FAILURE, not as a suite that hangs.
        raise AssertionError(
            "mpirun did not finish within %d s -- suspect a deadlock in the mesh data merge "
            "(nproc=%d distribute=%s discontinuous=%s).\n--- stdout tail ---\n%s"
            % (timeout, nproc, distribute, discontinuous, (e.stdout or "")[-3000:]))
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


def _serial_reference(tmpdir, discontinuous=False, size=8):
    sys.path.insert(0, _HERE)
    try:
        import mpi_global_meshdata_worker
        return mpi_global_meshdata_worker.run_case(N=size, discontinuous=discontinuous,
                                                   outdir=os.path.join(str(tmpdir), "serial"))
    finally:
        sys.path.remove(_HERE)


def _assert_matches_serial(merged, serial, what):
    for mesh_name, ref in serial["meshes"].items():
        got = merged["meshes"][mesh_name]
        assert got is not None, "%s: no merged data for %s" % (what, mesh_name)
        assert got["nnode"] == ref["nnode"], "%s: %s has %d nodes, serial has %d" % (
            what, mesh_name, got["nnode"], ref["nnode"])
        assert got["nelem"] == ref["nelem"], "%s: %s has %d elements, serial has %d" % (
            what, mesh_name, got["nelem"], ref["nelem"])
        assert got["elem_types"] == ref["elem_types"], "%s: %s element types differ" % (what, mesh_name)
        assert got["coord_digest"] == ref["coord_digest"], "%s: %s node coordinates differ" % (what, mesh_name)
        assert got["elem_digest"] == ref["elem_digest"], "%s: %s element connectivity differs" % (what, mesh_name)
        assert got.get("segment_lengths") == ref.get("segment_lengths"), (
            "%s: %s interface line segments differ: %s vs %s" % (
                what, mesh_name, got.get("segment_lengths"), ref.get("segment_lengths")))
        for key, value in ref.items():
            if key.endswith(("_sum", "_sqsum", "_max")):
                assert got[key] == pytest.approx(value, rel=_FIELD_RTOL), (
                    "%s: %s %s is %r, serial has %r" % (what, mesh_name, key, got[key], value))


@pytest.mark.parametrize("nproc", [2, 3, 4])
def test_merged_data_equals_serial(tmp_path, nproc):
    serial = _serial_reference(tmp_path)
    per_rank = _run_mpi(nproc, tmp_path)
    assert per_rank[0]["distributed"], "the mesh was not distributed, so nothing was actually merged"
    _assert_matches_serial(per_rank[0], serial, "np=%d" % nproc)
    for r in per_rank[1:]:
        assert all(v is None for v in r["meshes"].values()), (
            "rank %d received merged data; it is only assembled on rank 0" % r["rank"])


def test_merged_data_equals_serial_discontinuous(tmp_path):
    serial = _serial_reference(tmp_path, discontinuous=True)
    per_rank = _run_mpi(3, tmp_path, discontinuous=True)
    _assert_matches_serial(per_rank[0], serial, "discontinuous np=3")


def test_without_distribute_every_rank_has_the_whole_mesh(tmp_path):
    # Without --distribute each rank holds the complete mesh, so global data is local data: no
    # communication at all, and no rank is left without it.
    serial = _serial_reference(tmp_path)
    per_rank = _run_mpi(2, tmp_path, distribute=False)
    assert not per_rank[0]["distributed"]
    for r in per_rank:
        _assert_matches_serial(r, serial, "np=2 without --distribute, rank %d" % r["rank"])


def test_repeated_request_is_cached_and_does_not_hang(tmp_path):
    serial = _serial_reference(tmp_path)
    per_rank = _run_mpi(3, tmp_path, twice=True)
    _assert_matches_serial(per_rank[0], serial, "repeated request")
