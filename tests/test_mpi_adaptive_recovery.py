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

# Recovering from a failed re-solve after spatial adaptation, under MPI.
#
# There are two genuinely different cases and both are covered here:
#
#   * WITHOUT --distribute, every rank holds the whole problem and does identical work. A snapshot is
#     therefore per-rank private, and save_state's "only one rank writes" guard -- which exists so
#     that N ranks do not all write the same FILE -- must not apply, or the ranks other than 0 would
#     hold an empty snapshot and could not restore.
#   * WITH --distribute, save_state merges the mesh into one partition-independent stream that only
#     rank 0 ends up holding, while load_state needs every rank to read the whole thing. So the
#     buffer has to be broadcast.
#
# What both share is the reason this is delicate at all: restoring a state is COLLECTIVE, so the
# handler must only run when every rank is in it. The failure is injected from rank 0 ONLY, which is
# the partition-dependent case -- Problem::consume_newton_abort_request() Allreduces the request
# before throwing, so one rank's decision becomes every rank's failure. If that agreement ever
# breaks, these tests hang rather than fail, which is why _run() has a timeout that reports as an
# assertion.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_adaptive_recovery_worker.py")


def _mpi_reason():
    if shutil.which("mpirun") is None:
        return "mpirun not found"
    try:
        from pyoomph.generic.mpi import has_mpi
        if not has_mpi():
            return "pyoomph was built without MPI"
    except Exception as e:  # noqa: BLE001
        return "MPI unavailable: " + str(e)
    try:
        from petsc4py import PETSc  # type:ignore
        if not PETSc.Sys.hasExternalPackage("mumps"):
            return "PETSc has no MUMPS support (no distributed-capable direct solver)"
    except Exception:  # noqa: BLE001
        return "petsc4py not available (no distributed-capable direct solver)"
    return None


_SKIP_REASON = _mpi_reason()
pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]


def _run(nproc, tmpdir, mode, distribute, strategy="accept_unadapted", timeout=900):
    # No --oversubscribe, and nproc is kept to 4 at most: the problems here are small and the point
    # is the control flow, not throughput.
    cmd = ["mpirun", "-n", str(nproc), sys.executable, _WORKER,
           "--outdir", str(tmpdir), "--mode", mode, "--strategy", strategy]
    if distribute:
        cmd += ["--distribute"]
    # Importing pyoomph calls MPI_Init, so this pytest process already owns an Open MPI session
    # directory under TMPDIR; a nested mpirun collides with it and dies with no diagnostics.
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        raise AssertionError(
            "mpirun did not finish within %d s -- suspect a deadlock: a recovery agreed by only "
            "some ranks leaves the others waiting in the next collective (mode=%s, nproc=%d, "
            "distribute=%s).\n--- stdout tail ---\n%s"
            % (timeout, mode, nproc, distribute, (e.stdout or "")[-3000:]))
    per_rank = [json.loads(l[len("PYOOMPH_MPI_RESULT "):]) for l in proc.stdout.splitlines()
                if l.startswith("PYOOMPH_MPI_RESULT ")]
    if not per_rank:
        raise AssertionError(
            "no results from mpirun (exit %d -- a negative value is the killing signal)\n"
            "--- stdout tail ---\n%s\n--- stderr tail ---\n%s"
            % (proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
    assert len(per_rank) == nproc, "reported from %d of %d ranks" % (len(per_rank), nproc)
    for r in per_rank:
        assert "error" not in r, "failed on rank %d: %s\n%s" % (
            r["rank"], r["error"], r.get("traceback", ""))
    assert proc.returncode == 0, \
        "mpirun exited %d after every rank reported -- suspect a crash during teardown" % proc.returncode
    # Without this, the "distributed" half of the matrix would pass while silently exercising the
    # replicated path -- which is a different code path in save_state, save_mesh_state and
    # _snapshot_state, i.e. exactly the ones under test.
    for r in per_rank:
        assert r["distributed"] == distribute, \
            "rank %d reports distributed=%s, expected %s" % (r["rank"], r["distributed"], distribute)
    return per_rank


@pytest.mark.parametrize("distribute", [False, True], ids=["replicated", "distributed"])
@pytest.mark.parametrize("nproc", [2, 4])
def test_rollback_survives_and_every_rank_agrees(tmp_path, nproc, distribute):
    """The core promise, under MPI: the run does not die, and the ranks stay in step afterwards."""
    per_rank = _run(nproc, tmp_path, "recover", distribute)
    ref = per_rank[0]
    for r in per_rank:
        where = "rank %d/%d (distribute=%s)" % (r["rank"], nproc, distribute)
        assert r["total_failures"] == 1, \
            "%s: the sabotage fired %d times, expected once" % (where, r["total_failures"])
        assert r["finite"], "%s: the recovered state has non-finite dofs" % where
        assert r["ndof"] == r["ndof_before_sabotaged_adapt"], \
            "%s: not rolled back (ndof=%d, pre-adapt was %d)" % (
                where, r["ndof"], r["ndof_before_sabotaged_adapt"])
        assert r["resolve_ok"], "%s: the problem was not usable after the recovery" % where
        # Agreement is the whole point: a recovery that ran on some ranks only would show up as a
        # different mesh size or a different solution here (if it did not simply hang first).
        assert r["ndof"] == ref["ndof"], \
            "%s: ndof=%d but rank 0 has %d -- the ranks diverged" % (where, r["ndof"], ref["ndof"])
        assert r["dofsum"] == pytest.approx(ref["dofsum"], rel=1e-12), \
            "%s: the solution differs from rank 0's" % where


@pytest.mark.parametrize("distribute", [False, True], ids=["replicated", "distributed"])
@pytest.mark.parametrize("nproc", [2, 4])
def test_snapshot_roundtrip(tmp_path, nproc, distribute):
    """The snapshot itself: every rank must hold a usable one, and get its own state back.

    Without the stream exemption to save_state's redundant-writer guard, the replicated case fails
    here with an empty buffer on every rank but 0; without the broadcast, the distributed case does.
    """
    per_rank = _run(nproc, tmp_path, "snapshot", distribute)
    ref = per_rank[0]
    for r in per_rank:
        where = "rank %d/%d (distribute=%s)" % (r["rank"], nproc, distribute)
        assert r["snapshot_bytes"] > 0, "%s: the snapshot is empty" % where
        assert r["moved_ndof"] != r["before_ndof"], \
            "%s: the problem did not move, so the restore proves nothing" % where
        assert r["restored_ndof"] == r["before_ndof"], \
            "%s: restored to ndof=%d, expected %d" % (where, r["restored_ndof"], r["before_ndof"])
        assert r["restored_exact"], "%s: the restored solution is not the one that was saved" % where
        assert r["resolve_ok"], "%s: the problem was not usable after the restore" % where
        assert r["snapshot_bytes"] == ref["snapshot_bytes"], \
            "%s: snapshot is %d bytes but rank 0's is %d" % (
                where, r["snapshot_bytes"], ref["snapshot_bytes"])
