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

# Eigenvalue problems under MPI, through SLEPc.
#
# pytest runs serially, so each test launches tests/mpi_eigen_worker.py under mpirun and compares
# against a serial run of the same worker computed in-process.
#
# There are two distinct MPI situations and they exercise different code, so both are covered:
#
#   1. plain `mpirun` with NO --distribute. oomph-lib assembles in parallel and then redistributes
#      the eigenproblem matrices back to a globally replicated form, so every rank holds the whole
#      thing. The eigensolver imposes its own contiguous row split and each rank contributes only
#      its slice, so the solve is parallel anyway. This used to die outright ("Row too large: row
#      255 max 254"), because the solver declared an n-row local block on each of nproc ranks.
#   2. `--distribute`. Each rank holds a row block with global column indices, and the split comes
#      from oomph's dof distribution instead.
#
# Either way the EPS lives on COMM_WORLD and the eigenvectors are gathered back to full length
# afterwards, so the two paths differ only in where the row split comes from and whether the rows
# had to be sliced out of a bigger matrix.
#
# What each assertion is for:
#
#   - eigenvalues vs the SERIAL run. A globally consistent but wrong answer -- every rank agreeing
#     on the spectrum of a matrix that is not the problem's -- passes cross-rank agreement and fails
#     only here.
#   - agreement BETWEEN RANKS. SLEPc computes the eigenvalues collectively so they can hardly
#     differ, but "evect_len" and "evect0_absum" can: a rank that skipped the gather reports its own
#     row block, which is shorter and sums to less.
#   - "eigfunc_usqr", the integral of the squared eigenfunction over the mesh, vs the serial run.
#     This is the only assertion that constrains WHERE on the mesh the eigenvector's entries end up.
#     It is what caught set_current_dofs() not synchronising halo nodes: eigenvalues, vector length
#     and vector norm were all correct, and the eigenfunction was still 13% short.
#
# Everything compared across serial and distributed is numbering-independent, because distribute()
# renumbers the dofs: eigenvalues and mesh integrals qualify, a dof vector does not.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_eigen_worker.py")


def _mpi_reason():
    """None if a distributed eigen run is possible here, else the reason to skip."""
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
    try:
        import slepc4py  # type:ignore  # noqa: F401
    except Exception:
        return "slepc4py not available"
    return None


_SKIP_REASON = _mpi_reason()
pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]

# Tolerances. Serial and distributed solve the same discrete eigenproblem and differ only in how the
# rows are spread over ranks, so the answers should agree far better than engineering accuracy. They
# do not agree to round-off, though: shift-and-invert Krylov-Schur stops at its own tolerance, and a
# different factorisation ordering moves the converged eigenvalue within it. The spread actually
# observed across serial / np=2 / np=3 is up to ~6e-9 on eigenvalues of order 10-40, so 1e-7 leaves
# room without being loose enough to hide a genuinely different matrix -- a wrong row block or a
# misapplied axis constraint moves these by percent, not by 1e-9.
_EVAL_TOL = 1e-7
# The eigenfunction integral, by contrast, IS reproduced to round-off: it comes out of the same dof
# values placed on the same mesh. Measured agreement is ~1e-15 relative, so this stays tight.
_OBS_RTOL = 1e-8
# Between ranks of ONE run there is nothing left to differ -- the same collective produced them.
_RANK_RTOL = 1e-12


def _run_mpi(nproc, tmpdir, distribute, mode="eigen", eigensolver="slepc", size=8, neigen=3,
             azimuthal_m=None, problem="diffusion", timeout=900):
    """Launch the worker under mpirun and return the list of per-rank result dicts."""
    cmd = ["mpirun", "-n", str(nproc)]
    if os.environ.get("PYOOMPH_MPI_OVERSUBSCRIBE", "1") == "1":
        # CI machines routinely have fewer slots than we ask for; without this OpenMPI refuses to start.
        cmd += ["--oversubscribe"]
    cmd += [sys.executable, _WORKER, "--outdir", str(tmpdir), "--mode", mode,
            "--eigensolver", eigensolver, "--size", str(size), "--neigen", str(neigen),
            "--problem", problem,
            "--azimuthal-m", str(-1 if azimuthal_m is None else azimuthal_m)]
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
        # Bounded on purpose: ranks that disagree about whether the eigenproblem is complex, or about
        # how many rows they own, deadlock in a collective rather than returning. That has to surface
        # as a FAILURE, not as a suite that never finishes.
        raise AssertionError(
            "mpirun did not finish within %d s -- suspect a deadlock (nproc=%d distribute=%s mode=%s)."
            "\n--- stdout tail ---\n%s" % (timeout, nproc, distribute, mode, (e.stdout or "")[-3000:]))
    per_rank = []
    for line in proc.stdout.splitlines():
        if line.startswith("PYOOMPH_MPI_RESULT "):
            per_rank.append(json.loads(line[len("PYOOMPH_MPI_RESULT "):]))
    if not per_rank:
        raise AssertionError(
            "no results from mpirun (exit %d)\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s"
            % (proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
    assert len(per_rank) == nproc, "reported from %d of %d ranks" % (len(per_rank), nproc)
    return per_rank


def _check_no_errors(per_rank):
    for r in per_rank:
        assert "error" not in r, "failed on rank %d: %s\n%s" % (
            r["rank"], r["error"], r.get("traceback", ""))


def _serial_reference(tmpdir, size=8, neigen=3, azimuthal_m=None, problem="diffusion",
                      eigensolver="slepc"):
    sys.path.insert(0, _HERE)
    try:
        import mpi_eigen_worker
        return mpi_eigen_worker.solve_case(N=size, neigen=neigen, azimuthal_m=azimuthal_m,
                                           problem=problem, eigensolver=eigensolver,
                                           outdir=os.path.join(str(tmpdir), "serial"))
    finally:
        sys.path.remove(_HERE)


def _assert_solve_was_split(what, per_rank):
    """The eigensolve must have been genuinely divided over the ranks.

    Worth asserting separately because it is invisible in the answer: a solve that quietly ran
    redundantly on every rank returns exactly the same eigenvalues and eigenvectors, just without any
    of the benefit. The row blocks must tile [0, ndof) exactly -- contiguous, non-overlapping, no gaps
    -- which also rules out the slicing and the matrix construction disagreeing about the split.
    """
    ndof = per_rank[0]["ndof"]
    for r in per_rank:
        assert r["eigen_parallel"], "%s: rank %d solved on its own, not in parallel" % (what, r["rank"])
        assert 0 < r["eigen_nrow_local"] < ndof, \
            "%s: rank %d owns %d of %d rows -- not a split" % (what, r["rank"], r["eigen_nrow_local"], ndof)
    blocks = sorted((r["eigen_first_row"], r["eigen_nrow_local"]) for r in per_rank)
    expected = 0
    for first, nloc in blocks:
        assert first == expected, \
            "%s: row blocks do not tile: expected next block at %d, got %d" % (what, expected, first)
        expected = first + nloc
    assert expected == ndof, "%s: row blocks cover %d of %d rows" % (what, expected, ndof)


def _assert_ranks_agree(per_rank):
    """Every rank must report the same eigenpairs, at full global length."""
    ref = per_rank[0]
    for r in per_rank[1:]:
        assert r["nconv"] == ref["nconv"], \
            "rank %d converged %d eigenpairs, rank 0 got %d" % (r["rank"], r["nconv"], ref["nconv"])
        assert r["evect_len"] == ref["evect_len"], \
            "rank %d reports eigenvectors of length %d, rank 0 of %d -- a rank kept its local block" \
            % (r["rank"], r["evect_len"], ref["evect_len"])
        for i in range(ref["nconv"]):
            assert r["evals_re"][i] == pytest.approx(ref["evals_re"][i], rel=_RANK_RTOL, abs=1e-12), \
                "rank %d eigenvalue %d: %.17g vs %.17g" % (r["rank"], i, r["evals_re"][i], ref["evals_re"][i])
        assert r["evect0_absum"] == pytest.approx(ref["evect0_absum"], rel=_RANK_RTOL), \
            "rank %d sum|v0| = %.17g vs %.17g" % (r["rank"], r["evect0_absum"], ref["evect0_absum"])


def _assert_matches_serial(what, per_rank, serial):
    got = per_rank[0]
    assert got["ndof"] == serial["ndof"], "%s: ndof %d vs %d" % (what, got["ndof"], serial["ndof"])
    assert got["evect_len"] == serial["ndof"], \
        "%s: eigenvectors are %d long but the problem has %d dofs -- they were not gathered" \
        % (what, got["evect_len"], serial["ndof"])
    assert got["nconv"] == serial["nconv"], \
        "%s: %d eigenvalues vs %d serially" % (what, got["nconv"], serial["nconv"])
    for i in range(serial["nconv"]):
        assert got["evals_re"][i] == pytest.approx(serial["evals_re"][i], abs=_EVAL_TOL), \
            "%s: eigenvalue %d = %.17g vs %.17g serially" % (what, i, got["evals_re"][i], serial["evals_re"][i])
        assert got["evals_im"][i] == pytest.approx(serial["evals_im"][i], abs=_EVAL_TOL), \
            "%s: Im(eigenvalue %d) = %.17g vs %.17g serially" % (what, i, got["evals_im"][i], serial["evals_im"][i])
    # The one assertion that constrains where the eigenvector lands ON THE MESH.
    keys = [k for k in serial if k.startswith("eigfunc_")]
    assert keys, "no eigenfunction observables -- the eigenvector's placement would go unchecked"
    for k in keys:
        assert got[k] == pytest.approx(serial[k], rel=_OBS_RTOL), \
            "%s: %s = %.17g vs %.17g serially" % (what, k, got[k], serial[k])


@pytest.mark.parametrize("nproc", [2, 3])
def test_replicated_matches_serial(tmp_path, nproc):
    """mpirun without --distribute: one parallel eigenproblem over a replicated matrix.

    oomph assembles in parallel and then replicates, so each rank holds the whole J and M; the
    eigensolver imposes its own contiguous row split and each rank contributes only its slice. The
    solve is therefore just as parallel as the --distribute one -- what is not saved is matrix
    memory. Row splitting that disagreed with the slicing would show up as a wrong spectrum here,
    not as a crash, since every block is individually a valid set of rows.
    """
    serial = _serial_reference(tmp_path)
    per_rank = _run_mpi(nproc, tmp_path, distribute=False)
    _check_no_errors(per_rank)
    assert not per_rank[0]["distributed"], "the mesh should NOT be distributed in this case"
    _assert_solve_was_split("replicated np=%d" % nproc, per_rank)
    _assert_ranks_agree(per_rank)
    _assert_matches_serial("replicated np=%d" % nproc, per_rank, serial)


@pytest.mark.parametrize("nproc", [2, 3])
def test_distributed_matches_serial(tmp_path, nproc):
    """mpirun with --distribute: one parallel eigenproblem over row-partitioned matrices."""
    serial = _serial_reference(tmp_path)
    per_rank = _run_mpi(nproc, tmp_path, distribute=True)
    _check_no_errors(per_rank)
    assert per_rank[0]["distributed"], "the run was not actually distributed"
    _assert_solve_was_split("distributed np=%d" % nproc, per_rank)
    _assert_ranks_agree(per_rank)
    _assert_matches_serial("distributed np=%d" % nproc, per_rank, serial)


def test_distributed_azimuthal_matches_serial(tmp_path):
    """Azimuthal stability on a scalar field, i.e. the m != 0 machinery on its own.

    Deliberately the SIMPLE azimuthal case: expanding a scalar diffusion equation as u(r,z)e^{i m phi}
    produces a purely real operator (the m^2/r^2 term is real), so this exercises the azimuthal setup
    and the distributed solve without the complex assembly. The asserted complex_assembly is False
    records that -- an early version of this test claimed to cover the complex branch and did not.
    test_distributed_axisymmetric_flow is where that branch is actually taken.
    """
    serial = _serial_reference(tmp_path, azimuthal_m=1, problem="azimuthal")
    assert not serial["complex_assembly"], \
        "this problem is expected to stay real; if it no longer does, the comment above is stale"
    per_rank = _run_mpi(2, tmp_path, distribute=True, azimuthal_m=1, problem="azimuthal")
    _check_no_errors(per_rank)
    assert per_rank[0]["distributed"]
    _assert_solve_was_split("distributed azimuthal m=1", per_rank)
    _assert_ranks_agree(per_rank)
    _assert_matches_serial("distributed azimuthal m=1", per_rank, serial)


def test_distributed_axisymmetric_flow(tmp_path):
    """The two branches that only an azimuthal VECTOR field reaches: complex assembly and manipulators.

    Expanding a velocity field as e^{i m phi} couples the radial and azimuthal components through
    factors of i, so J and M are genuinely complex here -- and whether they are is decided from a
    nonzero count of the imaginary matrices, which is a PER-RANK quantity. Ranks that answered it
    differently would issue different collectives and hang, so this test is as much about not
    deadlocking as about the numbers.

    At the same time the AxisymmetryBC imposes the axis conditions by rewriting rows of J and M.
    Serially that is scipy row surgery on the assembled global matrix; distributed, no rank holds
    one, so it becomes MatZeroRows on the PETSc matrices restricted to the rows this rank owns.
    Getting that restriction wrong -- too many rows or too few -- moves the eigenvalues, which the
    comparison against the serial run detects.

    slepc_mumps rather than slepc: with a pressure field the shifted matrix has empty diagonal
    entries and PETSc's own LU refuses those ("Matrix is missing diagonal entry"). That is a
    pre-existing property of the serial path too, nothing to do with MPI.
    """
    serial = _serial_reference(tmp_path, size=6, neigen=2, azimuthal_m=1, problem="axiflow",
                               eigensolver="slepc_mumps")
    assert serial["complex_assembly"], \
        "the serial run did not produce a complex eigenproblem -- that branch went untested"
    assert serial["zeromap_size"] > 0, \
        "no rows were constrained even serially -- the manipulator branch went untested"
    per_rank = _run_mpi(2, tmp_path, distribute=True, azimuthal_m=1, problem="axiflow",
                        eigensolver="slepc_mumps", size=6, neigen=2)
    _check_no_errors(per_rank)
    assert per_rank[0]["distributed"]
    # Every rank must have reached the SAME verdict; disagreeing is what deadlocks.
    assert all(r["complex_assembly"] for r in per_rank), \
        "ranks disagree on whether the eigenproblem is complex: %s" % [r["complex_assembly"] for r in per_rank]
    # Per-rank, because the axis lies in one partition: the ranks owning none of it legitimately
    # constrain nothing. What must not happen is NO rank constraining anything.
    assert sum(r["zeromap_size"] for r in per_rank) > 0, \
        "no rank applied the axis constraint -- the distributed manipulator path never ran"
    _assert_solve_was_split("distributed axisymmetric flow m=1", per_rank)
    _assert_ranks_agree(per_rank)
    _assert_matches_serial("distributed axisymmetric flow m=1", per_rank, serial)


def test_scipy_eigensolver_refuses_distributed(tmp_path):
    """A backend that cannot see a partitioned matrix must say so, not answer wrongly.

    scipy/ARPACK only ever sees one process' rows. Without the distributed_possible() check it would
    solve each rank's row block as if it were the whole eigenproblem and return numbers that look
    entirely reasonable.
    """
    per_rank = _run_mpi(2, tmp_path, distribute=True, mode="guard", eigensolver="scipy")
    _check_no_errors(per_rank)
    for r in per_rank:
        assert r["distributed"], "rank %d was not distributed" % r["rank"]
        assert r["raised"], "rank %d silently solved a distributed eigenproblem with scipy" % r["rank"]
        assert "distribute" in r["message"], \
            "rank %d raised, but the message does not mention distribution: %s" % (r["rank"], r["message"])
