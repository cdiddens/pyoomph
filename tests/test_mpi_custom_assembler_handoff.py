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

# The use_custom_residual_jacobian handoff under MPI: dev_docs/mpi_augmented_systems.md B2.
#
# A Python assembler returns the WHOLE system indexed 0..ndof-1. The caller has already built the
# DoubleVector and CRDoubleMatrix on the LINEAR SOLVER's distribution, which under mpirun is a row
# block -- and, without --distribute, not even the same partition of those rows as the dof
# distribution: oomph's SuperLUSolver::solve(Problem*) imposes a uniform split, while the dof
# distribution is whatever distribute() produced. On this problem at np=2 that is 60/61 against
# 55/66. Copying ndof entries into a vector holding nrow_local() of them writes past the end.
#
# What makes this test worth its runtime: against the build immediately before the fix it does not
# merely report a wrong number, it dies with
#
#     [rank 1] [1]PETSC ERROR: Caught signal number 11 SEGV: Segmentation Violation,
#              probably memory access out of range
#
# at np=2, and passes after. That is the whole justification for fixing the handoff ahead of the
# pipeline it serves.
#
# Deliberately NOT routed through Problem.set_custom_assembler, which still refuses nproc>1
# (_require_single_rank) because the multi-assembly below it throws under MPI. The guard sits on
# set_custom_assembler, not on use_custom_residual_jacobian, and get_custom_residuals_jacobian is
# documented as overridable directly -- which is what lets the handoff be tested on its own.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_custom_handoff_worker.py")


def _mpi_reason():
    """None if a distributed run is possible here, else the reason to skip."""
    if shutil.which("mpirun") is None:
        return "mpirun not found"
    try:
        from pyoomph.generic.mpi import has_mpi
        if not has_mpi():
            return "pyoomph was built without MPI"
    except Exception as e:
        return "MPI unavailable: " + str(e)
    try:
        from petsc4py import PETSc  # type:ignore
        if not PETSc.Sys.hasExternalPackage("mumps"):
            return "PETSc has no MUMPS support (no distributed-capable direct solver)"
    except Exception:
        return "petsc4py not available"
    return None


_SKIP_REASON = _mpi_reason()
pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]

# The two paths assemble the SAME system -- the custom one hands back what the ordinary assembly
# produced -- so they converge through identical Newton steps and the observable agrees to round-off,
# not to engineering accuracy. Measured spread across serial / np=2 / np=4 / --distribute is ~3e-16
# relative. 1e-12 leaves room for a different summation order without hiding a misplaced row: a
# residual or Jacobian row copied to the wrong place moves this by percent, or crashes.
_OBS_RTOL = 1e-12


def _run(nproc, tmpdir, distribute=False, timeout=900, declare_rows="none", expect_failure=False):
    """Launch the worker (under mpirun when nproc>1) and return the per-rank result dicts."""
    cmd = []
    if nproc > 1:
        cmd += ["mpirun", "-n", str(nproc)]
    cmd += [sys.executable, _WORKER, "--outdir", str(tmpdir), "--declare-rows", declare_rows]
    if distribute:
        cmd += ["--distribute"]
    # Importing pyoomph calls MPI_Init, so THIS pytest process is already a singleton MPI job owning
    # an Open MPI session directory under TMPDIR; a nested mpirun collides with it and dies with exit
    # code 1 and no diagnostics. Give the child its own.
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        # Bounded on purpose: a handoff that leaves the ranks disagreeing about how many rows they
        # own can hang in a collective rather than returning, and that has to be a FAILURE.
        raise AssertionError(
            "mpirun did not finish within %d s -- suspect a deadlock (nproc=%d distribute=%s)."
            "\n--- stdout tail ---\n%s" % (timeout, nproc, distribute, (e.stdout or "")[-3000:]))
    per_rank = []
    for line in proc.stdout.splitlines():
        if line.startswith("PYOOMPH_MPI_RESULT "):
            per_rank.append(json.loads(line[len("PYOOMPH_MPI_RESULT "):]))
    assert len(per_rank) == nproc, (
        "reported from %d of %d ranks (exit %d)\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s"
        % (len(per_rank), nproc, proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
    if not expect_failure:
        for r in per_rank:
            assert r.get("ok"), "rank %s failed: %s\n%s" % (r.get("rank"), r.get("error"), r.get("traceback", ""))
    return per_rank


@pytest.mark.parametrize("nproc,distribute", [(1, False), (2, False), (4, False), (2, True), (3, True)])
def test_custom_handoff_matches_ordinary_assembly(tmp_path, nproc, distribute):
    """The custom path must converge to what the ordinary assembly converges to, on every rank."""
    per_rank = _run(nproc, tmp_path, distribute=distribute)
    for r in per_rank:
        assert r["usqr"] == pytest.approx(r["usqr_ordinary"], rel=_OBS_RTOL), (
            "rank %d: the custom handoff converged somewhere else than the ordinary assembly" % r["rank"])
        # Several Newton steps, so a corrupted residual has somewhere to show rather than being
        # hidden by a single linear solve.
        assert r["n_handoffs"] >= 3, "only %d handoffs -- the solve was too short to be a test" % r["n_handoffs"]
    # Every rank must agree: they all read the same converged state through a partition-independent
    # integral, so there is nothing left here to differ.
    ref = per_rank[0]["usqr"]
    for r in per_rank[1:]:
        assert r["usqr"] == pytest.approx(ref, rel=_OBS_RTOL), "ranks disagree about the converged state"


@pytest.mark.parametrize("nproc,distribute", [(2, False), (4, False), (2, True)])
def test_solver_really_handed_over_a_row_block(tmp_path, nproc, distribute):
    """Guard against the previous test passing for the wrong reason.

    If the solver asked for the whole system on every rank there would be nothing to slice, and a
    broken handoff would pass. Under mpirun the blocks must be proper subsets that tile [0,n).
    """
    per_rank = _run(nproc, tmp_path, distribute=distribute)
    blocks = []
    for r in per_rank:
        assert r["solver_row_blocks"], "rank %d saw no distributed solve at all" % r["rank"]
        for n, nrow_local, first_row in r["solver_row_blocks"]:
            assert nrow_local < n, (
                "rank %d was handed all %d rows, so this run proves nothing about slicing" % (r["rank"], n))
            blocks.append((first_row, nrow_local, n))
    n_global = blocks[0][2]
    covered = sorted((f, nl) for f, nl, _ in set(blocks))
    expect = 0
    for first_row, nrow_local in covered:
        assert first_row == expect, "the solver's row blocks do not tile [0,%d): %s" % (n_global, covered)
        expect += nrow_local
    assert expect == n_global, "the row blocks cover %d of %d rows" % (expect, n_global)


def test_fresh_matrix_survives_the_custom_path(tmp_path):
    """Problem.assemble_jacobian() hands in a FRESH CRDoubleMatrix, not one oomph distributed.

    build(ncol,...) takes its row count from the distribution, so an unbuilt one used to make it
    write a full row_start array into a zero-row matrix. Serial is enough to cover it -- the bug is
    the unbuilt distribution, not the rank count -- and it is the case that segfaulted whenever a
    Python assembler was installed.
    """
    r = _run(1, tmp_path)[0]
    assert r["fresh_matrix_ok"], "assemble_jacobian() failed on the custom path: %s" % r.get("fresh_error")
    assert r["fresh_shape"] == [r["ndof"], r["ndof"]]
    assert r["fresh_res_len"] == r["ndof"]
    assert r["fresh_nnz"] > r["ndof"], "a Jacobian with no off-diagonal entries is not this problem's"


# --- the declared row block (CustomResJacInfo.set_row_distribution) -------------------------------
#
# An assembler may say which rows of the global system it built, instead of returning all of them.
# That is the shape a distributed tracker produces, and the contract the linear-algebra backend will
# use. What is testable before the backend exists is the contract itself: declaring the whole system
# must be indistinguishable from not declaring, and declaring anything else than what the solver
# asked for must be refused rather than reconciled.


def test_declaring_the_whole_system_changes_nothing(tmp_path):
    """first_row=0, nrow_local=n is what a serial assembler produces either way."""
    plain = _run(1, tmp_path / "plain")[0]
    declared = _run(1, tmp_path / "declared", declare_rows="global")[0]
    assert declared["usqr"] == pytest.approx(plain["usqr"], rel=_OBS_RTOL)
    assert declared["usqr"] == pytest.approx(declared["usqr_ordinary"], rel=_OBS_RTOL)


def test_a_block_that_is_not_the_solvers_is_refused(tmp_path):
    """Serial: one row short of the whole system, which no distribution ever asks for."""
    r = _run(1, tmp_path, declare_rows="wrong")[0]
    assert r["wrong_block_refused"], "a short row block was accepted"
    assert "disagree about the row distribution" in r["refusal"], r["refusal"]
    # The message has to name both blocks, or it cannot be acted on.
    assert "built rows [0,120) of 121" in r["refusal"], r["refusal"]
    assert "holds rows [0,121) of 121" in r["refusal"], r["refusal"]


def test_declaring_the_whole_system_is_refused_under_mpirun(tmp_path):
    """And this is the point of the check: under mpirun the solver asks for a BLOCK.

    An assembler that declares the whole system there has not produced what the solver wants, and
    silently copying the overlap would give a plausible wrong answer that differs per rank. Each
    rank's message names the block it was actually asked for, which is how a tracker learns what to
    build.
    """
    per_rank = _run(2, tmp_path, declare_rows="global", expect_failure=True)
    assert all(not r["ok"] for r in per_rank), "declaring the global system was accepted under mpirun"
    seen = set()
    for r in per_rank:
        msg = str(r.get("error", ""))
        assert "disagree about the row distribution" in msg, msg
        assert "built rows [0,121) of 121" in msg, msg
        # "holds rows [a,b)" differs per rank -- that is the uniform split the solver imposed.
        start = msg.index("holds rows ")
        seen.add(msg[start:start + 30])
    assert len(seen) == 2, "both ranks reported the same block, so the message is not per-rank: %s" % seen
