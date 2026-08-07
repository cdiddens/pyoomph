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

# Tracer particles on a distributed mesh (dev_docs/tracers.md).
#
# A particle is owned by the process on which its element is not a halo. It is allowed to advect
# THROUGH the halo layer - whose nodal positions and dof values are synchronised copies - and only at
# the end of the step is ownership reconsidered and the particle handed to the process that owns the
# element it ended in. Doing it that way is what avoids ever having to place a particle in the
# time-interpolated configuration on a process that did not integrate it.
#
# The field here is Poiseuille flow written as an analytic expression, so the exact answer is known
# to round-off and the assertions can be made at 1e-11 rather than at some tolerance a lost or
# duplicated particle could hide behind. What each case is for:
#
#   - the analytic answer at np = 2, 3, 4, plus the proof that particles really did migrate (a
#     serial implementation that never migrated anything would pass the first assertion alone);
#   - the global count, agreed on by every process, so that a duplicate or a loss is caught;
#   - the gathered view being identical on every process and equal to the serial answer, which is
#     the actual promise of the partition-independent gather;
#   - every particle seeded in one corner, so that some process holds none throughout. Any
#     collective entered only by the processes that happen to hold particles deadlocks there;
#   - a state file written at one process count and read at another.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_tracer_worker.py")


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

# Machine-exact case, so this is round-off over ~55 steps, not a modelling tolerance.
_ATOL = 1e-11


def _run(nproc, tmpdir, mode, statefile=None, timeout=1200):
    """Run the worker, in-process when nproc is 1, under mpirun otherwise."""
    if nproc == 1:
        sys.path.insert(0, _HERE)
        try:
            import mpi_tracer_worker
            return [mpi_tracer_worker.run_case(mode, os.path.join(str(tmpdir), "serial"),
                                               statefile=statefile)]
        finally:
            sys.path.remove(_HERE)
    cmd = ["mpirun", "-n", str(nproc), sys.executable, _WORKER, mode,
           "--outdir", os.path.join(str(tmpdir), "out" + str(nproc)), "--distribute"]
    if statefile is not None:
        cmd += ["--statefile", statefile]
    # Importing pyoomph calls MPI_Init, so this pytest process is already a singleton MPI job with
    # its own Open MPI session directory. A nested mpirun collides with it unless given its own.
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        # Bounded on purpose. Every tracer collective is entered by all processes including the ones
        # holding no particles; if that ever stops being true the symptom is a hang, and a hang has
        # to surface as a failure rather than as a suite that never finishes.
        raise AssertionError(
            "mpirun did not finish within %d s - suspect a deadlock in the tracer collectives "
            "(nproc=%d mode=%s).\n--- stdout tail ---\n%s"
            % (timeout, nproc, mode, (e.stdout or "")[-3000:]))
    per_rank = []
    for line in proc.stdout.splitlines():
        if line.startswith("PYOOMPH_MPI_RESULT "):
            per_rank.append(json.loads(line[len("PYOOMPH_MPI_RESULT "):]))
    if not per_rank:
        raise AssertionError(
            "no results from mpirun (exit %d)\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s"
            % (proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
    assert len(per_rank) == nproc, "reported from %d of %d processes" % (len(per_rank), nproc)
    return per_rank


def _assert_all_ranks_agree(per_rank):
    """The gathered view is supposed to be identical everywhere, not merely consistent."""
    first = per_rank[0]
    for r in per_rank[1:]:
        assert r["ids"] == first["ids"], "processes disagree about which particles exist"
        assert r["nglobal"] == first["nglobal"]
        for a, b in zip(r["positions"], first["positions"]):
            for va, vb in zip(a, b):
                assert abs(va - vb) < 1e-14, "processes disagree about a particle position"


@pytest.mark.parametrize("nproc", [2, 3, 4])
def test_distributed_tracers_reproduce_the_analytic_answer(nproc, tmp_path):
    per_rank = _run(nproc, tmp_path, "run")
    _assert_all_ranks_agree(per_rank)
    first = per_rank[0]
    assert first["nglobal"] == first["nstart"] == 5, "a particle was lost or duplicated"
    assert first["analytic_error"] < _ATOL, "analytic error %g" % first["analytic_error"]
    assert sum(r["nlocal"] for r in per_rank) == first["nglobal"]
    assert any(r["nlocal_changed"] for r in per_rank), \
        "no particle ever changed process, so migration was never exercised"


@pytest.mark.parametrize("nproc", [2, 4])
def test_distributed_result_matches_the_serial_one(nproc, tmp_path):
    serial = _run(1, tmp_path, "run")[0]
    dist = _run(nproc, tmp_path, "run")[0]
    assert dist["ids"] == serial["ids"], "the particle identities depend on the partitioning"
    assert len(dist["positions"]) == len(serial["positions"])
    for a, b in zip(dist["positions"], serial["positions"]):
        for va, vb in zip(a, b):
            assert abs(va - vb) < _ATOL
    for a, b in zip(dist["payloads"], serial["payloads"]):
        assert abs(a - b) < _ATOL


@pytest.mark.parametrize("nproc", [2, 4])
def test_a_process_with_no_particles_does_not_hang(nproc, tmp_path):
    """Every particle starts in one corner, so at least one process holds none from beginning to
    end. It still has to enter every collective."""
    per_rank = _run(nproc, tmp_path, "corner")
    _assert_all_ranks_agree(per_rank)
    assert min(r["nlocal"] for r in per_rank) == 0, \
        "the test is vacuous unless some process really holds no particles"
    assert per_rank[0]["analytic_error"] < _ATOL


@pytest.mark.parametrize("nproc", [1, 2, 4])
def test_periodic_reinjection_crosses_processes(nproc, tmp_path):
    """The case the halo exchange cannot do. A particle leaving through the outlet has to reappear
    at the inlet, which under a partitioning that knows nothing about the periodicity belongs to
    an entirely different process - not a halo of the one it left from, so exchange_migrants()
    would never route it there. It goes through the collective reinjection round instead, and the
    answer has to be the same as the serial one, to round-off."""
    per_rank = _run(nproc, tmp_path, "periodic")
    _assert_all_ranks_agree(per_rank)
    first = per_rank[0]
    assert sum(r["nwrapped"] + r["nreinjected"] for r in per_rank) >= 1, \
        "no particle ever reached the outlet, so nothing was re-injected and this is vacuous"
    assert first["nglobal"] == first["nstart"] == 5, \
        "%d of %d particles survived the wrap" % (first["nglobal"], first["nstart"])
    assert first["analytic_error"] < _ATOL, "analytic error %g" % first["analytic_error"]
    assert sum(r["nlocal"] for r in per_rank) == first["nglobal"]
    reinjected = sum(r["nreinjected"] for r in per_rank)
    if nproc == 1:
        assert reinjected == 0, "a serial run has nothing to re-inject across processes"
        assert first["nwrapped"] >= 1
    else:
        # The whole point of the collective round. If this ever fires, the partitioning happened to
        # put both ends of the domain on one process and the test is only checking the local path.
        assert reinjected >= 1, \
            "no particle was re-injected across processes, so the collective round was not exercised"


@pytest.mark.parametrize("write_nproc,read_nproc", [(3, 1), (1, 4), (2, 4)])
def test_state_file_is_partition_independent(write_nproc, read_nproc, tmp_path):
    """The file holds the whole particle set sorted by identity, so it says nothing about how the
    mesh was split and can be written at one process count and read at another."""
    statefile = str(tmp_path / "tracers.dump")
    written_ranks = _run(write_nproc, tmp_path, "save", statefile=statefile)
    read_ranks = _run(read_nproc, tmp_path, "load", statefile=statefile)
    written, read = written_ranks[0], read_ranks[0]
    assert read["nglobal"] == written["nglobal"]
    assert read["ids"] == written["ids"]
    assert read["analytic_error"] < _ATOL

    # The rolling position history the trail plots read has to come back too, and the gather that
    # writes it has to be partition-independent like everything else in the file. Merged over the
    # ranks by identity, since only the owner of a particle holds its history.
    def merged(ranks):
        out = {}
        for r in ranks:
            out.update(r["history_at_state"])
        return out

    wh, rh = merged(written_ranks), merged(read_ranks)
    assert sorted(wh.keys()) == sorted(rh.keys())
    assert min(len(v) for v in wh.values()) >= 6, "no history was recorded - the check is vacuous"
    for k in wh:
        assert rh[k] == wh[k], "the history of particle " + k + " did not survive the state file"
