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

# Bifurcation tracking under MPI, for the four augmented assembly handlers of src/bifurcation.cpp:
# MyFoldHandler, MyHopfHandler, MyPitchForkHandler and AzimuthalSymmetryBreakingHandler.
#
# There are two distinct MPI situations and they exercise different code, so both are covered:
#
#   1. plain `mpirun` with NO --distribute. Every rank holds the whole mesh and the whole dof
#      vector; only the element loop is split. The handlers then push all of their extra unknowns
#      on every rank and build a NON-distributed augmented dof distribution -- which is what commit
#      27a7c23 had to fix, because a distributed one made rank 1 apply the eigenvector block's
#      Newton increment to the base dofs. This case is here to keep that fixed.
#   2. `--distribute`. Each rank owns a row block of the base dofs. The handlers then store their
#      eigenvector blocks as DoubleVectorWithHaloEntries, keep the scalar unknowns (parameter,
#      Omega, Sigma) on rank 0 alone, and translate their naive [base | k*Ndof+m] numbering into a
#      per-rank interleaved one -- modelled on upstream oomph-lib's distributed PitchForkHandler.
#
# What each assertion is for:
#
#   - the critical parameter (and, for Hopf, omega) vs the SERIAL run. A globally consistent but
#     wrong augmented system -- every rank agreeing on a bifurcation that is not the problem's --
#     passes cross-rank agreement and fails only here.
#   - agreement BETWEEN RANKS. The parameter and omega live on rank 0's dof vector alone when
#     distributed, so a missing synchronise() broadcast shows up here as ranks reporting different
#     values, and "evect_len" catches a rank that returned its own row block instead of gathering.
#   - "eigfunc_usqr", the integral of the squared tracked eigenfunction over the mesh, vs the serial
#     run. This is the only assertion that constrains WHERE on the mesh the eigenvector's entries
#     end up: a broken eqn_number translation or a missing halo synchronise leaves the critical
#     parameter and the eigenvector norm correct and moves this.
#   - the dof vector is restored on deactivation, to its LOCAL length when distributed.
#
# Everything compared across serial and distributed is numbering-independent, because distribute()
# renumbers the dofs: the critical parameter, omega and mesh integrals qualify, a dof vector does not.

import json
import os
import shutil
import subprocess
import sys

import numpy
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_bifurcation_worker.py")


def _mpi_reason():
    """None if a distributed tracking run is possible here, else the reason to skip."""
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

# Tolerances. Serial and distributed solve the same augmented system and differ only in how the
# rows are spread over ranks, so the answers should agree far better than engineering accuracy. They
# do not agree to round-off, though: the Newton solve stops at its own tolerance and a different
# MUMPS ordering moves the converged point within it. The spread actually observed across serial /
# np=2 / np=3 is up to ~4e-10 relative on the critical parameter, so 1e-7 leaves room without being
# loose enough to hide a genuinely different augmented system -- a wrong Count, a wrong element
# count or a misplaced eigenvector block moves these by percent, not by 1e-10.
_PARAM_RTOL = 1e-7
# The eigenfunction integral, by contrast, IS reproduced to near round-off: it comes from the same
# dof values placed on the same mesh. Measured agreement is ~1e-12 relative or better.
_OBS_RTOL = 1e-8
# Between ranks of ONE run there is nothing left to differ -- every rank reads the same converged
# augmented state, and the scalars are broadcast from rank 0.
_RANK_RTOL = 1e-12
# The base state's eigenvalues, compared both across serial/distributed and against the same state
# with the tracker removed. Looser than _RANK_RTOL because shift-invert Krylov-Schur stops at its own
# tolerance and a different MUMPS ordering moves the result within it; the same 6e-9-ish spread the
# plain MPI eigenvalue suite records. A base-vs-augmented layout mix-up does not move an eigenvalue
# by 1e-7 -- it solves a different problem.
_EIG_RTOL = 1e-7
_EIG_ATOL = 1e-7


def _run_mpi(nproc, tmpdir, distribute, case="fold", size=8, timeout=1200,
             eigenvector_scaling="unit", eigen_during_tracking=False):
    """Launch the worker under mpirun and return the list of per-rank result dicts."""
    # No --oversubscribe: an oversubscribed run makes the timings meaningless and, on this project's
    # machines, is explicitly not wanted. nproc stays small enough to fit real cores.
    cmd = ["mpirun", "-n", str(nproc), sys.executable, _WORKER,
           "--outdir", str(tmpdir), "--case", case, "--size", str(size),
           "--eigenvector-scaling", eigenvector_scaling]
    if eigen_during_tracking:
        cmd += ["--eigen-during-tracking"]
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
        # Bounded on purpose: ranks that disagree about how many augmented rows they own deadlock in
        # a collective rather than returning. That has to surface as a FAILURE, not as a suite that
        # never finishes.
        raise AssertionError(
            "mpirun did not finish within %d s -- suspect a deadlock (nproc=%d distribute=%s case=%s)."
            "\n--- stdout tail ---\n%s" % (timeout, nproc, distribute, case, (e.stdout or "")[-3000:]))
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


def _serial_reference(tmpdir, case="fold", size=8, **kwargs):
    sys.path.insert(0, _HERE)
    try:
        import mpi_bifurcation_worker
        return mpi_bifurcation_worker._CASES[case](
            N=size, outdir=os.path.join(str(tmpdir), "serial_" + case), **kwargs)
    finally:
        sys.path.remove(_HERE)


def _assert_ranks_agree(per_rank):
    """Every rank must report the same converged bifurcation, at full global eigenvector length."""
    ref = per_rank[0]
    for r in per_rank[1:]:
        assert r["ndof"] == ref["ndof"], "rank %d sees %d augmented dofs, rank 0 sees %d" % (
            r["rank"], r["ndof"], ref["ndof"])
        # Length, not just value: a rank that skipped the gather reports only its own row block.
        assert r["evect_len"] == ref["evect_len"], \
            "rank %d returned an eigenvector of length %d, rank 0 of %d" % (
                r["rank"], r["evect_len"], ref["evect_len"])
        for key in ("param", "omega", "evect_norm", "evect_absmax", "eigfunc_usqr"):
            if key not in ref:
                continue
            a, b = ref[key], r[key]
            assert abs(a - b) <= _RANK_RTOL * max(1.0, abs(a)), \
                "ranks disagree on %s: rank 0 %r vs rank %d %r" % (key, a, r["rank"], b)


def _assert_matches_serial(what, serial, per_rank, distributed):
    r = per_rank[0]
    assert r["distributed"] is distributed, \
        "%s: expected distributed=%s, got %s" % (what, distributed, r["distributed"])
    assert r["evect_len"] == serial["evect_len"], \
        "%s: eigenvector length %d, serial %d" % (what, r["evect_len"], serial["evect_len"])
    for key in ("param", "omega"):
        if key not in serial:
            continue
        a, b = serial[key], r[key]
        if key == "omega":
            # A Hopf frequency comes as the conjugate pair +-i*omega and which of the two the
            # eigensolver hands back is arbitrary -- a plain replicated mpirun flips it against
            # serial just as readily as a distributed one does. The magnitude is the physics.
            a, b = abs(a), abs(b)
        assert abs(a - b) <= _PARAM_RTOL * max(1.0, abs(a)), \
            "%s: %s %r, serial %r" % (what, key, b, a)
    # The placement of the eigenfunction on the mesh, which nothing above constrains.
    a, b = serial["eigfunc_usqr"], r["eigfunc_usqr"]
    assert abs(a - b) <= _OBS_RTOL * abs(a), \
        "%s: eigfunc_usqr %r, serial %r (relative %g)" % (what, b, a, abs(a - b) / abs(a))
    # Deactivation has to put the dof vector back: globally to the base count, locally to this
    # rank's share of it. Getting this wrong leaves the problem unusable after tracking.
    assert r["ndof_after_deactivate"] == serial["ndof_after_deactivate"], \
        "%s: %d dofs after deactivating, serial %d" % (
            what, r["ndof_after_deactivate"], serial["ndof_after_deactivate"])
    total_local = sum(x["nrow_local_after_deactivate"] for x in per_rank)
    if distributed:
        assert total_local == r["ndof_after_deactivate"], \
            "%s: local dof counts after deactivating sum to %d, not %d" % (
                what, total_local, r["ndof_after_deactivate"])
    else:
        for x in per_rank:
            assert x["nrow_local_after_deactivate"] == r["ndof_after_deactivate"], \
                "%s: rank %d kept %d local dofs, expected the full %d (replicated run)" % (
                    what, x["rank"], x["nrow_local_after_deactivate"], r["ndof_after_deactivate"])


# fold      -- MyFoldHandler, [u | param | Y], real eigenvector, parameter is the only scalar.
# hopf      -- MyHopfHandler, [u | Phi | Psi | param | Omega]: two eigenvector blocks and two
#              scalars, so it is the case that catches a scalar left un-broadcast (omega).
# pitchfork -- MyPitchForkHandler, [u | param | Y | Sigma]: a scalar on EITHER side of the
#              eigenvector block in the naive numbering, which the translation table has to place.
# azimuthal -- AzimuthalSymmetryBreakingHandler, [u | Re | Im | param | Omega] plus the axis dofs
#              it forces to zero by global equation number.
# fold_interface -- the fold case on a problem that ALSO has an interface mesh, so a handler sets up
#              its dof halo scheme over more than one mesh. Same fold, so it must give the same
#              numbers. It is NOT a regression test for the tree-less halo element of INFO_oomph-lib,
#              29th August 2026 (three tutorials died with SIGSEGV in
#              Tree::stick_leaves_into_vector(this=0) on activation): that one is a BULK element in
#              the halo layer, and this interface mesh has no root halo elements at all. Covering it
#              is open -- dev_docs/mpi_augmented_systems.md section 6b.
_CASES = ["fold", "hopf", "pitchfork", "azimuthal", "fold_interface"]


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("nproc", [2, 3])
def test_distributed_tracking_matches_serial(tmp_path, case, nproc):
    serial = _serial_reference(tmp_path, case=case)
    per_rank = _run_mpi(nproc, tmp_path, distribute=True, case=case)
    _check_no_errors(per_rank)
    _assert_ranks_agree(per_rank)
    _assert_matches_serial("%s distributed np=%d" % (case, nproc), serial, per_rank, True)


@pytest.mark.parametrize("case", _CASES)
def test_replicated_mpirun_tracking_matches_serial(tmp_path, case):
    """Plain mpirun, no --distribute: the path commit 27a7c23 fixed. Kept green on purpose."""
    serial = _serial_reference(tmp_path, case=case)
    per_rank = _run_mpi(2, tmp_path, distribute=False, case=case)
    _check_no_errors(per_rank)
    _assert_ranks_agree(per_rank)
    _assert_matches_serial("%s replicated np=2" % case, serial, per_rank, False)


def test_replicated_mpirun_fold_without_eigenvector_guess(tmp_path):
    """Fold tracking with no eigenproblem solved first: MyFoldHandler's no-guess constructor.

    It builds its own guess by solving against d(residual)/d(parameter), and that vector came back
    filled on rank 0 only, because Problem::get_residuals() assembles a REPLICATED target vector
    entirely onto rank 0 (see INFO_oomph-lib, 9th August 2026). The guess therefore pointed
    elsewhere under mpirun than serially and the augmented Newton solve converged onto a different
    fold -- fully converged on both sides, so only a comparison against the serial answer catches it.
    Replicated only: the constructor refuses --distribute and demands an explicit guess there.
    """
    serial = _serial_reference(tmp_path, case="fold_noguess")
    # The guess is only a guess: it must find the same fold as the eigenvector-guided run.
    serial_guided = _serial_reference(tmp_path, case="fold")
    assert abs(serial["param"] - serial_guided["param"]) <= _PARAM_RTOL * abs(serial_guided["param"])
    per_rank = _run_mpi(2, tmp_path, distribute=False, case="fold_noguess")
    _check_no_errors(per_rank)
    _assert_ranks_agree(per_rank)
    _assert_matches_serial("fold without guess, replicated np=2", serial, per_rank, False)


def _assert_eigen_during_tracking(what, serial, per_rank, distributed):
    """The base state's eigenproblem, solved with the tracker still installed.

    Three things have to hold, and they fail in different ways:

      - the spectrum matches the SAME state with the tracker removed (the A/B inside each run). Both
        assemble the identical element contributions -- oomph's get_eigenproblem_matrices installs
        its own EigenProblemHandler either way -- so a difference can only come from the row layout,
        which is what Problem::BaseDofDistributionScope restores.
      - it matches the SERIAL spectrum. Eigenvalues are computed collectively, so every rank
        necessarily agrees with its peers whether or not it was handed the right rows; only the
        comparison against serial constrains WHICH eigenproblem was solved.
      - the row blocks tile [0, base_ndof) exactly. This is the one assertion that says the
        eigensolver was handed the BASE distribution rather than the augmented one, and that the
        blocks are the partitioned ones rather than nproc copies of the whole thing.
    """
    for r in per_rank:
        tracked = numpy.array(r["track_eig_re"]) + 1j * numpy.array(r["track_eig_im"])
        plain = numpy.array(r["plain_eig_re"]) + 1j * numpy.array(r["plain_eig_im"])
        assert len(tracked) == len(plain) and len(tracked) > 0, (what, r["rank"], tracked, plain)
        assert numpy.allclose(tracked, plain, rtol=_EIG_RTOL, atol=_EIG_ATOL), \
            "%s rank %d: tracked %r vs untracked %r" % (what, r["rank"], tracked, plain)
        ref = numpy.array(serial["track_eig_re"]) + 1j * numpy.array(serial["track_eig_im"])
        assert numpy.allclose(tracked, ref, rtol=_EIG_RTOL, atol=_EIG_ATOL), \
            "%s rank %d: %r, serial %r" % (what, r["rank"], tracked, ref)
        # Not the augmented count, which is what res["ndof"] is while tracking.
        assert r["eig_nrow"] == serial["eig_nrow"] < r["ndof"], (what, r["rank"], r["eig_nrow"], r["ndof"])
        assert r["eig_row_distributed"] is distributed, (what, r["rank"])

    blocks = sorted((r["eig_first_row"], r["eig_nrow_local"]) for r in per_rank)
    if distributed:
        at = 0
        for first, n in blocks:
            assert first == at, "%s: eigen row blocks do not tile [0, %d): %r" % (
                what, serial["eig_nrow"], blocks)
            at += n
        assert at == serial["eig_nrow"], "%s: eigen row blocks cover %d of %d rows: %r" % (
            what, at, serial["eig_nrow"], blocks)
    else:
        # Replicated: oomph redistributes the assembled matrices back to a globally replicated form,
        # so every rank holds all of them and the eigensolver imposes its own split further down.
        for first, n in blocks:
            assert (first, n) == (0, serial["eig_nrow"]), "%s: %r" % (what, blocks)


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("nproc", [2, 3])
def test_eigen_during_tracking_distributed(tmp_path, case, nproc):
    """Solve the base state's eigenproblem while the tracker is installed, under --distribute.

    For "azimuthal" the eigensolve is taken at m=0 while an m=1 bifurcation is tracked, i.e. the
    mode of the eigenproblem differs from the tracked one -- see the worker.
    """
    serial = _serial_reference(tmp_path, case=case, with_eigen=True)
    per_rank = _run_mpi(nproc, tmp_path, distribute=True, case=case, eigen_during_tracking=True)
    _check_no_errors(per_rank)
    _assert_ranks_agree(per_rank)
    _assert_matches_serial("%s eigen-while-tracking distributed np=%d" % (case, nproc),
                           serial, per_rank, True)
    _assert_eigen_during_tracking("%s eigen-while-tracking distributed np=%d" % (case, nproc),
                                  serial, per_rank, True)


@pytest.mark.parametrize("case", _CASES)
def test_eigen_during_tracking_replicated(tmp_path, case):
    """The same, under a plain mpirun: the replicated augmented distribution is rebuilt in place
    rather than pointer-swapped, so it is a genuinely different path through the scope."""
    serial = _serial_reference(tmp_path, case=case, with_eigen=True)
    per_rank = _run_mpi(2, tmp_path, distribute=False, case=case, eigen_during_tracking=True)
    _check_no_errors(per_rank)
    _assert_ranks_agree(per_rank)
    _assert_matches_serial("%s eigen-while-tracking replicated np=2" % case, serial, per_rank, False)
    _assert_eigen_during_tracking("%s eigen-while-tracking replicated np=2" % case,
                                  serial, per_rank, False)


def test_maxabs_eigenvector_scaling_distributed(tmp_path):
    """eigenvector_scaling="auto" must locate the same fold, distributed, with O(1) unknowns.

    The two reductions it adds -- an MPI_Allreduce(MAX) for the scale and an MPI_Allreduce(SUM) for
    the constraint's new right-hand side -- are the whole distributed content of the option, and a
    rank-local max or a rank-local dot product would give every rank a different constraint. That
    shows up as ranks disagreeing, and as a critical parameter that no longer matches the "unit" run.
    """
    serial_unit = _serial_reference(tmp_path, case="fold")
    serial_auto = _serial_reference(tmp_path, case="fold", eigenvector_scaling="auto")
    # Same bifurcation, differently scaled eigenvector: that is the entire intended difference.
    assert abs(serial_auto["param"] - serial_unit["param"]) <= _PARAM_RTOL * abs(serial_unit["param"])
    # The guess is scaled to max|y| = 1 exactly and the converged eigenvector stays near it, so this
    # is the O(1) claim itself rather than a ratio: what "unit" gives instead is 1/sqrt(ndof)-ish and
    # keeps shrinking with the mesh, which is the whole reason the option exists.
    assert 0.5 < serial_auto["evect_absmax"] < 2.0, \
        "auto did not rescale the eigenvector to order one: absmax %r" % serial_auto["evect_absmax"]
    assert serial_auto["evect_absmax"] > 3 * serial_unit["evect_absmax"], \
        "auto did not rescale the eigenvector: absmax %r vs unit %r" % (
            serial_auto["evect_absmax"], serial_unit["evect_absmax"])

    per_rank = _run_mpi(2, tmp_path, distribute=True, case="fold", eigenvector_scaling="auto")
    _check_no_errors(per_rank)
    _assert_ranks_agree(per_rank)
    _assert_matches_serial("fold auto-scaled distributed np=2", serial_auto, per_rank, True)
