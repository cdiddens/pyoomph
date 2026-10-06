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

# The PYTHON trackers of pyoomph/generic/bifurcation_tools.py under MPI: FoldTracker,
# PitchForkTracker, HopfTracker, both eigenbranch trackers and NormalModeBifurcationTracker, one
# case per family as they are ported.
#
# BOTH branches of NormalModeBifurcationTracker are covered: the stationary one on a Turing mode and
# the oscillatory (has_imag) one on an advected mode. What is NOT covered is
# CriticalWavenumberTracker, which supports_mpi() still refuses.
#
# HopfTracker's LEFT-eigenvector branch is not covered, and is refused under mpirun on purpose: its
# bordered system needs J^T and M^T as blocks, and transposing a row-distributed matrix is an
# off-processor exchange that lands on PETSc's own ownership range rather than the base layout
# block() requires. supports_mpi() returns False for it; the eigensolver route is what MPI uses for
# the Hopf adjoint anyway.
#
# Everything the tracker stands on is tested on its own: the handoff
# (test_mpi_custom_assembler_handoff), the dof layout (test_mpi_augmentation_layout), the assembly
# (test_mpi_multiassembly), the backend (test_distributed_la, test_mpi_distributed_la) and a bordered
# solve through all of it (test_mpi_bordered_solve). What is left for here is the tracker.
#
# The problem is the one tests/mpi_bifurcation_worker.py uses for the C++ MyFoldHandler -- 2D Bratu,
# whose branch turns at a genuine limit point -- so the two routes can be compared against each
# other. Two independent implementations of the same augmented system agreeing is a stronger
# statement than either agreeing with itself across rank counts.
#
# --distribute works too, and the reason it did not at first is worth knowing: under --distribute the
# augmented SCALARS (here the bifurcation parameter) live on rank 0 ALONE, so a Newton update wrote
# the new value into rank 0's Dof_pt and no other rank learnt it. Every rank then assembled a
# slightly different augmented system and the Newton converged LINEARLY -- a factor of about three
# per step -- rather than failing. The C++ trackers broadcast their scalars from
# AssemblyHandler::synchronise(); the Python route had no handler to override, and now installs a
# sync-only one. test_the_newton_converges_quadratically below is what would catch a regression,
# because the critical parameter alone does not: with a high enough iteration cap the broken version
# reached the same answer eventually.

import json
import os
import shutil
import subprocess
import sys

import numpy
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_python_tracker_worker.py")


def _mpi_reason():
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
        return "slepc4py not available (the eigenvector guess needs an eigensolve)"
    return None


_SKIP_REASON = _mpi_reason()
pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]

# Serial, np=2 and np=3 all converge to lam = 6.808263809409476 -- every digit -- and the
# eigenfunction integral agrees to ~1e-18, because the replicated regime assembles the same global
# system and solves it the same way. The C++ route lands on the same parameter to 16 digits too.
# 1e-9 is far looser than any of that and still far tighter than a real defect: a misplaced border or
# a wrong constraint row moves a critical parameter by percent.
_PARAM_RTOL = 1e-9
_OBS_RTOL = 1e-8
# The critical-wavenumber family needs a TIGHTER NEWTON TOLERANCE, not a looser comparison, and
# finding that out was the whole of the work on it. At pyoomph's default 1e-8 the serial and MPI runs
# disagreed by 3.95e-7 in the critical parameter -- 44 times worse than any other family -- and the
# first explanation (the class divides by a finite-difference step of ~1e-6, so round-off is
# amplified a millionfold) was WRONG: the gap was exactly 3.954e-07 at k_fd_step = 1e-6, 1e-5 AND
# 1e-4, i.e. completely independent of the step.
#
# What it actually was: the runs stopped at different points inside the same tolerance ball. Serial's
# last Newton step overshot to a residual of 6.9e-14 while np=2 stopped at 8.5e-9, just under the
# 1e-8 default -- and a tangency condition is a DOUBLE root, so the system is ill-conditioned and a
# residual of 8.5e-9 is worth 4e-7 in the parameter. Asking for 1e-11 instead makes serial and np=2
# agree to 5e-12 on the oscillatory case and 7e-11 on the stationary one, both inside the ordinary
# _PARAM_RTOL, so this family gets no special tolerance at all.
#
# The stationary case then reproduces the closed form to 13 digits, which is what _ANALYTIC below
# asserts -- a far stronger statement than cross-regime agreement, and the reason it is worth
# spending the extra Newton steps.
_FD_NEWTON_TOL = 1e-11


def _run(nproc, tmpdir, distribute=False, cxx=False, nonlinear=False, N=8, case="fold",
         expect_failure=False, timeout=900, newton_tol=None):
    outdir = os.path.join(str(tmpdir), "out")
    os.makedirs(outdir, exist_ok=True)
    cmd = []
    if nproc > 1:
        cmd += ["mpirun", "-n", str(nproc)]
    cmd += [sys.executable, _WORKER, "--outdir", outdir, "--N", str(N), "--case", case]
    if distribute:
        cmd += ["--distribute"]
    if cxx:
        cmd += ["--cxx"]
    if nonlinear:
        cmd += ["--nonlinear-constraint"]
    # The codim-2 family is solved harder by default, because at pyoomph's 1e-8 the runs stop at
    # different points inside the same tolerance ball -- see _FD_NEWTON_TOL.
    tol = newton_tol if newton_tol is not None else (_FD_NEWTON_TOL if case in _FD_CASES else None)
    if tol is not None:
        cmd += ["--newton-tol", repr(float(tol))]
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        raise AssertionError(
            "mpirun did not finish within %d s -- suspect a deadlock (nproc=%d distribute=%s cxx=%s)."
            "\n--- stdout tail ---\n%s" % (timeout, nproc, distribute, cxx, (e.stdout or "")[-3000:]))
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
    return sorted(per_rank, key=lambda r: r["rank"])


_GROUPS = {"critical_wavenumber": [False, False, False, True, True],
           "critical_wavenumber_osc": [False] * 5 + [True] * 4,
           "fold": [False, False, True],
           "pitchfork": [False, False, True, True],
           "hopf": [False, False, False, True, True],
           # An eigenbranch tracker drives no parameter: the eigenvalue itself is the extra scalar,
           # so the layouts coincide with the fold's and the Hopf's.
           "eigenbranch_real": [False, False, True],
           "eigenbranch_complex": [False, False, False, True, True],
           # A Turing mode is STATIONARY, so the normal-mode tracker takes its non-has_imag branch:
           # one eigenvector block and the parameter, no omega. That is the branch whose header
           # comment records four bugs that could not surface until a problem reached it, so it is
           # worth having under MPI as well.
           "normal_mode": [False, False, True],
           # The OSCILLATORY normal mode takes the has_imag branch: five groups, like a Hopf. What
           # makes the mode oscillatory is advection ALONG the mode direction -- one derivative in z
           # contributes i*k, where a diffusive term pairs exp(+ikz) with exp(-ikz) and gives k^2.
           # A purely diffusive problem has no imaginary contribution to assemble at all, which is
           # why the stationary case above cannot reach this branch.
           "normal_mode_osc": [False, False, False, True, True]}
# The augmented size: a fold adds [V | p], a pitchfork [V | p | slack], a Hopf [Vr | Vi | p | omega].
_AUG = {"fold": lambda n: 2 * n + 1, "pitchfork": lambda n: 2 * n + 2, "hopf": lambda n: 3 * n + 2,
        "eigenbranch_real": lambda n: 2 * n + 1, "eigenbranch_complex": lambda n: 3 * n + 2,
        "normal_mode": lambda n: 2 * n + 1, "normal_mode_osc": lambda n: 3 * n + 2,
        # Codim-2: the parameter AND the wavenumber are unknowns, and each eigenvector block is
        # joined by its k-derivative dV/dk. Stationary: [V, W] + (parameter, k). Oscillatory:
        # [Vr, Vi, Wr, Wi] + (parameter, omega, k, mu).
        "critical_wavenumber": lambda n: 3 * n + 2, "critical_wavenumber_osc": lambda n: 5 * n + 4}
_CASES = ["fold", "pitchfork", "hopf", "eigenbranch_real", "eigenbranch_complex",
          "normal_mode", "normal_mode_osc", "critical_wavenumber", "critical_wavenumber_osc"]

# The codim-2 family: solved to _FD_NEWTON_TOL rather than the default, for the reason given there.
_FD_CASES = {"critical_wavenumber", "critical_wavenumber_osc"}

# The uniform Turing system's critical point in closed form: B = (1 + A/sqrt(d))^2 at
# k = sqrt(A)/d^(1/4), with the worker's A=2, d=8. Not an invariant of the discretisation -- it is
# the exact answer for the uniform state, which the worker's LineMesh represents exactly -- so it can
# be asserted to solver accuracy rather than to mesh accuracy.
_ANALYTIC = {"critical_wavenumber": {"critical": (1 + 2.0 / numpy.sqrt(8.0)) ** 2,
                                     "critical_k": numpy.sqrt(2.0) / 8.0 ** 0.25}}


def _param_rtol(case):
    return _PARAM_RTOL


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("nproc,distribute", [(1, False), (2, False), (3, False), (2, True), (3, True)])
def test_the_python_tracker_finds_the_bifurcation(tmp_path, nproc, distribute, case):
    per_rank = _run(nproc, tmp_path, distribute=distribute, case=case)
    for r in per_rank:
        assert r["supports_mpi"] is True
        # The group layout the bordered system is laid out from: a fold adds [V | parameter], a
        # pitchfork also a slack unknown for the symmetry constraint.
        assert r["groups"] == _GROUPS[case]
        assert r["ndof_aug"] == _AUG[case](r["ndof_base"])
        # The augmentation must be gone again afterwards.
        assert r["ndof_after"] == r["ndof_base"]
        # The eigenvector comes back globally replicated at full length, which every consumer of
        # get_last_eigenvectors() assumes (dev_docs/mpi_eigenproblems.md).
        assert r["evect_len"] == r["ndof_base"]
    # Every rank reads the same converged augmented state.
    for r in per_rank[1:]:
        assert r["critical"] == pytest.approx(per_rank[0]["critical"], rel=1e-12)
        assert r["eigfunc_usqr"] == pytest.approx(per_rank[0]["eigfunc_usqr"], rel=1e-12)
    if case in _ANALYTIC:
        # The strongest assertion in this file: the uniform Turing system's critical point is known
        # in closed form and the mesh represents the uniform state exactly, so this is solver
        # accuracy, not discretisation accuracy. Measured at 13 digits under MPI. A tangency row that
        # was merely plausible would not land here.
        for key, exact in _ANALYTIC[case].items():
            assert per_rank[0][key] == pytest.approx(exact, rel=1e-8), (
                "%s came out %.17g, the closed form is %.17g" % (key, per_rank[0][key], exact))


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("nproc,distribute", [(2, False), (3, False), (2, True), (3, True)])
def test_mpirun_agrees_with_serial(tmp_path, nproc, distribute, case):
    serial = _run(1, tmp_path / "serial", case=case)[0]
    got = _run(nproc, tmp_path / ("np%d%s" % (nproc, "d" if distribute else "")),
               distribute=distribute, case=case)[0]
    assert got["critical"] == pytest.approx(serial["critical"], rel=_param_rtol(case)), (
        "np=%d%s found the bifurcation at %.17g, serial at %.17g"
        % (nproc, " --distribute" if distribute else "", got["critical"], serial["critical"]))
    if "critical_k" in serial:
        # The second unknown of the codim-2 system, and the one this family exists for: a wrong
        # tangency row moves k while leaving the critical parameter plausible.
        assert got["critical_k"] == pytest.approx(serial["critical_k"], rel=_param_rtol(case)), (
            "np=%d%s found k=%.17g, serial %.17g"
            % (nproc, " --distribute" if distribute else "", got["critical_k"], serial["critical_k"]))
        assert got["has_imag"] is serial["has_imag"], \
            "the branch taken differs from serial: has_imag %r vs %r" % (
                got["has_imag"], serial["has_imag"])
        if serial["has_imag"]:
            # Absolute value, as for the complex eigenbranch: either member of the conjugate pair is
            # the same branch.
            assert abs(got["critical_omega"]) == pytest.approx(abs(serial["critical_omega"]),
                                                               rel=_param_rtol(case))
    # The mesh integral of the squared eigenfunction: the one assertion that constrains WHERE on the
    # mesh the eigenvector's entries ended up, which a wrong translation would move while leaving the
    # critical parameter alone.
    assert got["eigfunc_usqr"] == pytest.approx(serial["eigfunc_usqr"], rel=_OBS_RTOL)
    if "tracked_omega" in serial:
        # ABSOLUTE value: a complex eigenbranch may converge onto either member of the conjugate
        # pair, and it does -- +0.968 serially against -0.968 under --distribute. Both are the same
        # branch, the same invariance as an eigenvector's sign.
        assert abs(got["tracked_omega"]) == pytest.approx(abs(serial["tracked_omega"]), rel=_OBS_RTOL)


# The C++ handlers cover fold, pitchfork, hopf and the azimuthal case; eigenbranch tracking has no
# C++ counterpart to compare against (activate_bifurcation_tracking has no such mode).
@pytest.mark.parametrize("case", ["fold", "pitchfork", "hopf"])
@pytest.mark.parametrize("nproc", [1, 2])
def test_the_python_route_agrees_with_the_cxx_handler(tmp_path, nproc, case):
    """Two independent implementations of the same augmented system, on the same problem.

    The C++ MyFoldHandler has been correct under MPI for a while, so this is the strongest check
    available on the Python one -- stronger than it agreeing with itself across rank counts.
    """
    py = _run(nproc, tmp_path / "py", case=case)[0]
    cxx = _run(nproc, tmp_path / "cxx", cxx=True, case=case)[0]
    assert py["ndof_aug"] == cxx["ndof_aug"], "the two routes build different-sized augmented systems"
    assert py["critical"] == pytest.approx(cxx["critical"], rel=_PARAM_RTOL), (
        "the Python tracker found the fold at %.17g, the C++ handler at %.17g"
        % (py["critical"], cxx["critical"]))
    assert py["eigfunc_usqr"] == pytest.approx(cxx["eigfunc_usqr"], rel=_OBS_RTOL)


def test_the_nonlinear_length_constraint_also_works_under_mpirun(tmp_path):
    """<V,V> instead of <V,V0>: the normalisation row then depends on V and has to be re-replicated
    on every assembly, which is a different code path through the border row."""
    serial = _run(1, tmp_path / "serial", nonlinear=True)[0]
    got = _run(2, tmp_path / "np2", nonlinear=True)[0]
    assert got["critical"] == pytest.approx(serial["critical"], rel=_PARAM_RTOL)
    # The same fold, whichever constraint is used to pin the eigenvector's length.
    plain = _run(1, tmp_path / "plain")[0]
    assert serial["critical"] == pytest.approx(plain["critical"], rel=1e-6)


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("nproc,distribute", [(1, False), (2, False), (2, True), (3, True)])
def test_the_newton_converges_quadratically(tmp_path, nproc, distribute, case):
    """The RATE, not just the answer -- which is the only thing that catches a stale scalar.

    When the augmented scalars were not broadcast from rank 0, every rank held its own value of the
    bifurcation parameter, assembled a slightly different system, and the Newton converged linearly:
    a factor of about three per step instead of squaring the residual. It still reached the right
    answer given enough iterations, so a test on the critical parameter alone passed. Serial,
    replicated and --distribute all take four steps -- 0.155, 8.1e-3, 9.1e-5, 6.3e-9 -- and a
    quadratic rate is what this asserts.
    """
    steps = _run(nproc, tmp_path, distribute=distribute, case=case)[0]["newton_residuals"]
    if case in _FD_CASES:
        # The codim-2 family is NOT quadratic early on, and that is intrinsic rather than an MPI
        # defect: its tangency rows need d2J/dk dsigma, which has no analytic form, so the Jacobian
        # it solves with is a finite-difference approximation and the rate is quadratic only once the
        # step is small enough for the FD error to stop dominating. Measured serially, which is the
        # proof it is not about distribution: 1.8e-2, 1.28e-2, 4.3e-3, 6.3e-4, 1.9e-5, 8.5e-9 --
        # factors of 0.71 and 0.34 at the start and squaring at the end. So what is asserted is the
        # TERMINAL rate plus a bound on the step count, which is still enough to catch the defect
        # this test exists for: a stale rank-0-only scalar crawls at a constant factor all the way
        # down and never reaches the terminal regime.
        assert steps, "no residual history"
        assert len(steps) <= 12, "the codim-2 solve took %d Newton steps: %s" % (len(steps), steps)
        assert steps[-1] < _FD_NEWTON_TOL, "the solve did not converge: %s" % steps
        # A FACTOR, not an exponent. Squaring stops being the right bound once the residual reaches
        # the round-off floor: the measured tail is 1.09e-5 -> 1.28e-9 -> 1.20e-13, and 1.28e-9
        # squared is 1.6e-18, below anything the arithmetic can reach. What separates this from the
        # defect is the SIZE of the reduction -- about 1e-4 per step here against the ~0.3 of a
        # constant-factor crawl -- so two orders of magnitude is both comfortably satisfied and
        # nowhere near a linear rate.
        assert steps[-1] < 0.01 * steps[-2], (
            "the terminal step went %.3e -> %.3e (factor %.3g); a constant factor all the way down "
            "is the signature of a state the ranks disagree about: %s"
            % (steps[-2], steps[-1], steps[-1] / steps[-2], steps))
        return
    if steps and steps[0] < 1e-10:
        # Nothing to converge: an eigenbranch tracker is handed the eigenvalue the eigensolve just
        # found, so its augmented residual can start at round-off already. There is no rate to judge;
        # what matters is that it STAYS there rather than being pushed off by a bad step.
        assert max(steps) < 1e-8, "a solve that started converged did not stay converged: %s" % steps
        return
    # A fold from this guess takes four steps, a pitchfork two; what is being excluded is a long
    # crawl, not a particular count.
    assert 2 <= len(steps) <= 7, (
        "the tracked solve took %d Newton steps, which is not quadratic convergence: %s"
        % (len(steps), steps))
    assert steps[-1] < 1e-7, "the solve did not actually converge: %s" % steps
    # Each step should roughly square the previous residual. The first entry is the state the solve
    # started from, so the rate is judged from the second onwards, and the slack is generous: the
    # point is to separate squaring from a constant factor, not to pin the constant.
    for k in range(2, len(steps)):
        prev, cur = steps[k - 1], steps[k]
        assert cur < prev * 0.1, (
            "step %d only reduced the residual from %.3e to %.3e (factor %.2f); a constant factor "
            "like that is the signature of a state the ranks disagree about: %s"
            % (k, prev, cur, cur / prev, steps))
    # And the last step must be a big one, which a linear crawl never is.
    assert steps[-1] < steps[-2] * 0.1, (
        "the final step only gained a factor %.2f: %s" % (steps[-1] / steps[-2], steps))
