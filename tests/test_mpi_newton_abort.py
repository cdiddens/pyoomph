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

# Two things that only go wrong on a distributed problem.
#
# 1. get_current_dofs()/set_current_dofs(). oomph-lib's Problem::get_dofs(DoubleVector&) built a
#    vector holding nrow_local() doubles and then wrote ndof() -- the GLOBAL count -- entries into
#    it, reading Dof_pt (also local) with the same index. set_dofs(const DoubleVector&) did the
#    mirror image. Both therefore ran off the end of two buffers on every call and corrupted the
#    heap. Silently: the abort typically came much later, in an unrelated allocation, so the reported
#    symptom was a "malloc(): invalid size" at interpreter teardown with no connection to the cause.
#    oomph-lib guards the HISTORY overloads with a PARANOID "not designed for distributed problems"
#    throw; these two had no such guard. Fixed in the vendored copy (//FOR PYOOMPH).
#
# 2. Abandoning a Newton solve. An equation can reject a step from
#    before_newton_convergence_check() -- the real user is topological_changes.py, rejecting a step
#    that would make two interfaces overlap. Under time-adaptive stepping oomph-lib catches the
#    resulting NewtonSolverError, halves dt and retries, which is the entire point of the mechanism.
#    Two ways that used to fail under MPI: the rejection is normally only visible on the ranks
#    holding the offending part of the mesh, so the decision has to be agreed (the old dof-flooding
#    implementation instead called two COLLECTIVE routines from a subset of ranks, and deadlocked);
#    and that same implementation went through exactly the two corrupting routines above.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_newton_abort_worker.py")


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


def _run(nproc, tmpdir, mode, reject_at=-1.0, timeout=900):
    cmd = ["mpirun", "-n", str(nproc)]
    if os.environ.get("PYOOMPH_MPI_OVERSUBSCRIBE", "1") == "1":
        cmd += ["--oversubscribe"]
    cmd += [sys.executable, _WORKER, "--outdir", str(tmpdir), "--mode", mode,
            "--reject-at", str(reject_at), "--distribute"]
    # Importing pyoomph calls MPI_Init, so this pytest process already owns an Open MPI session
    # directory under TMPDIR; a nested mpirun collides with it and dies with no diagnostics.
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        # Bounded on purpose: a rejection agreed by only some ranks hangs rather than returning, and
        # that has to surface as a failure rather than a suite that never finishes.
        raise AssertionError(
            "mpirun did not finish within %d s -- suspect a deadlock (mode=%s, nproc=%d)."
            "\n--- stdout tail ---\n%s" % (timeout, mode, nproc, (e.stdout or "")[-3000:]))
    per_rank = [json.loads(l[len("PYOOMPH_MPI_RESULT "):]) for l in proc.stdout.splitlines()
                if l.startswith("PYOOMPH_MPI_RESULT ")]
    if not per_rank:
        # A heap-corruption abort kills the rank before it can report, so "no output" is the normal
        # way this family of bugs shows up here. Surface the exit status, which names the signal.
        raise AssertionError(
            "no results from mpirun (exit %d -- a negative value is the killing signal)\n"
            "--- stdout tail ---\n%s\n--- stderr tail ---\n%s"
            % (proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
    assert len(per_rank) == nproc, "reported from %d of %d ranks" % (len(per_rank), nproc)
    for r in per_rank:
        assert "error" not in r, "failed on rank %d: %s\n%s" % (
            r["rank"], r["error"], r.get("traceback", ""))
    # Exit status is checked last: the results are more informative when both are wrong.
    assert proc.returncode == 0, \
        "mpirun exited %d after every rank reported -- suspect a crash during teardown" % proc.returncode
    return per_rank


@pytest.mark.parametrize("nproc", [2, 4])
def test_distributed_dof_get_set_roundtrip(tmp_path, nproc):
    per_rank = _run(nproc, tmp_path, "dofs")
    ref = per_rank[0]
    for r in per_rank:
        where = "rank %d/%d" % (r["rank"], nproc)
        assert r["finite"], "%s: get_current_dofs returned non-finite values" % where
        assert r["n"] == r["ndof"], \
            "%s: got %d dofs for a problem with ndof=%d -- the gather is not global" % (
                where, r["n"], r["ndof"])
        # Every rank must see the same global vector, in the same order.
        assert r["checksum"] == pytest.approx(ref["checksum"], rel=1e-12), \
            "%s: dof vector differs from rank 0 (checksum %.17g vs %.17g)" % (
                where, r["checksum"], ref["checksum"])
        assert r["roundtrip_exact"], "%s: set_current_dofs(get_current_dofs()) changed the dofs" % where
        assert r["ke_shift"] == 0.0, "%s: the round-trip moved the solution by %.3e" % (where, r["ke_shift"])
        assert r["set_exact"], "%s: set_current_dofs did not take effect exactly" % where


@pytest.mark.parametrize("nproc", [2, 4])
def test_rejected_step_reduces_the_timestep_and_the_run_continues(tmp_path, nproc):
    """Rank 0 alone rejects one step; every rank must abandon it, halve dt, and finish the run."""
    rejected = _run(nproc, tmp_path / "rej", "transient", reject_at=0.05)
    for r in rejected:
        where = "rank %d/%d" % (r["rank"], nproc)
        # Reaching the end time is the whole point: the solve was abandoned, not the run.
        assert r["final_time"] == pytest.approx(0.5, abs=1e-5), \
            "%s: run stopped at t=%.6g instead of finishing at 0.5" % (where, r["final_time"])
        assert r["ke"] == pytest.approx(rejected[0]["ke"], rel=1e-10), \
            "%s: ranks disagree about the solution (ke %.17g vs %.17g)" % (
                where, r["ke"], rejected[0]["ke"])


def test_transient_run_without_rejection_is_unaffected(tmp_path):
    """Control: the machinery must not perturb a run in which nothing is ever rejected."""
    per_rank = _run(2, tmp_path, "transient", reject_at=-1.0)
    for r in per_rank:
        assert r["final_time"] == pytest.approx(0.5, abs=1e-9)
        assert r["ke"] == pytest.approx(per_rank[0]["ke"], rel=1e-10)
