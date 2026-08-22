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

# Solving the BASE state's eigenproblem while a bifurcation tracker from src/bifurcation.cpp is
# installed. That is what makes a codim-2 point visible along a bifurcation locus -- a second
# eigenvalue reaching zero (cusp), a pair crossing (fold-Hopf, Bogdanov-Takens), or the axisymmetric
# base state folding underneath an m=1 locus -- and it used to be refused outright.
#
# Why it is only bookkeeping: oomph-lib's Problem::get_eigenproblem_matrices installs its OWN
# EigenProblemHandler for the duration of the assembly, and that handler's ndof/eqn_number delegate
# straight to the element. So the elemental assembly is the base one either way, and the base state
# still sits in the node data, which no tracker moves. The one thing that was augmented is the row
# layout the matrices are built on (and Problem::ndof(), which oomph's own PARANOID block compares
# against), which Problem::BaseDofDistributionScope now puts back for the duration.
#
# WHAT EACH ASSERTION IS FOR:
#
#   - the spectrum vs the SAME state with the tracker removed. This is the assertion that matters.
#     Both runs assemble the identical element contributions, so anything that differs can only have
#     come from the row layout -- which is precisely what was wrong before.
#   - an eigenvalue at 0 (fold, pitchfork) or a pair at +-i*omega (Hopf). Independent of the A/B: it
#     says the base state really is sitting on the bifurcation the tracker converged it onto, which
#     no dof-layout accident can fake.
#   - the row block the eigensolver was handed (eig_nrow). Serially this is the base dof count, NOT
#     the augmented one -- the single number that says the scope was in effect at all.
#   - for the azimuthal case, the m=0 spectrum while an m=1 bifurcation is tracked, plus that the
#     tracked m and the tracked state both survive it. The mode of the eigensolve is deliberately
#     independent of the tracked mode, and the tracker reads the very same global "azimuthal_m"
#     parameter when it assembles its own eigen rows.
#
# Only ONE Problem is constructed per process (a second one segfaults in the JIT loader, see
# tests/test_multiple_problems.py), so each case runs the shared worker in its own subprocess. That
# worker is tests/mpi_bifurcation_worker.py, the same one tests/test_mpi_bifurcation_tracking.py
# drives under mpirun -- so the serial numbers asserted here and the distributed ones asserted there
# come from the same code.

import json
import os
import subprocess
import sys

import numpy
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_bifurcation_worker.py")


def _skip_reason():
    try:
        from petsc4py import PETSc  # type:ignore
        if not PETSc.Sys.hasExternalPackage("mumps"):
            return "PETSc has no MUMPS support"
    except Exception:
        return "petsc4py not available (PYTHONPATH must carry a complex PETSc build)"
    try:
        import slepc4py  # type:ignore  # noqa: F401
    except Exception:
        return "slepc4py not available"
    return None


_SKIP = _skip_reason()
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))

# The two spectra are NOT taken at the same state, which the tolerance has to allow for.
#
# _reaction_case re-converges the augmented system between them ("Re-converging has to land back on
# the same critical parameter"), and that second solve moves the critical parameter a little: 1.73e-8
# on the pitchfork at N=8. An eigenvalue that is zero at the bifurcation is linear in the distance to
# it, so the tracked zero mode comes out at exactly that shift - measured, -1.7299e-08 against a
# parameter step of +1.7298e-08, a ratio of 1.0000. The other three eigenvalues, being O(30), are
# unmoved at seven digits and are governed by the relative tolerance instead.
#
# So the floor here is the tracker's own convergence, not the eigensolver's. It used to sit just under
# 1e-8 (1.4e-9) and passed by a factor of seven, until the Gauss<2,3> knot fix (4ed8580) changed where
# the first solve lands and pushed it to 1.7e-8. What the test is actually for is unaffected: a
# base-vs-augmented layout mix-up does not move eigenvalues by 1e-8, it produces a different problem,
# and _assert_base_layout() checks the layout directly anyway.
_EIG_ATOL = 1e-6
_EIG_RTOL = 1e-8


def _run(tmp_path, case, size=8, timeout=1800):
    """Run one worker case in its own process and return its result dict."""
    out = subprocess.run([sys.executable, _WORKER, "--outdir", str(tmp_path), "--case", case,
                          "--size", str(size), "--eigen-during-tracking"],
                         cwd=str(tmp_path), capture_output=True, text=True, timeout=timeout)
    assert out.returncode == 0, out.stdout[-4000:] + out.stderr[-4000:]
    lines = [l for l in out.stdout.splitlines() if l.startswith("PYOOMPH_MPI_RESULT ")]
    assert len(lines) == 1, out.stdout[-4000:] + out.stderr[-4000:]
    res = json.loads(lines[0][len("PYOOMPH_MPI_RESULT "):])
    assert "error" not in res, res.get("traceback", res.get("error"))
    return res


def _spectra(res):
    tracked = numpy.array(res["track_eig_re"]) + 1j * numpy.array(res["track_eig_im"])
    plain = numpy.array(res["plain_eig_re"]) + 1j * numpy.array(res["plain_eig_im"])
    return tracked, plain


def _assert_matches_untracked(res):
    tracked, plain = _spectra(res)
    assert len(tracked) == len(plain) and len(tracked) > 0, (tracked, plain)
    assert numpy.allclose(tracked, plain, rtol=_EIG_RTOL, atol=_EIG_ATOL), (tracked, plain)


def _assert_base_layout(res):
    """The eigenproblem was assembled on the BASE rows, not the augmented ones.

    While tracking, res["ndof"] is the augmented count (2*Ndof+1 for a fold, 3*Ndof+2 for a Hopf),
    and it is deliberately compared against here rather than against a hard-coded number: if the
    scope had not been in effect, eig_nrow would BE res["ndof"].
    """
    assert res["eig_nrow"] == res["ndof_after_deactivate"], res
    assert res["eig_nrow"] < res["ndof"], res
    # Serially there is one row block and it is the whole thing.
    assert res["eig_row_distributed"] is False
    assert (res["eig_first_row"], res["eig_nrow_local"]) == (0, res["eig_nrow"]), res


def test_fold(tmp_path):
    """Bratu fold: lambda = 0 is an exact eigenvalue of the tracked state."""
    res = _run(tmp_path, "fold")
    _assert_base_layout(res)
    _assert_matches_untracked(res)
    tracked, _ = _spectra(res)
    # The fold's own zero. It is not exactly zero because the tracking Newton stops at its own
    # tolerance, and it is the eigenvalue nearest to zero by a wide margin (the next one is O(30)).
    assert numpy.amin(numpy.absolute(tracked)) < 1e-5, tracked
    assert numpy.isclose(res["param"], 6.8082638, rtol=1e-6), res["param"]


def test_hopf(tmp_path):
    """Brusselator Hopf: the tracked eigenvalue is a pair at +-i*omega, not at zero."""
    res = _run(tmp_path, "hopf", size=20)
    _assert_base_layout(res)
    _assert_matches_untracked(res)
    tracked, _ = _spectra(res)
    omega = res["omega"]
    # A complex pair on the imaginary axis at the tracked frequency. The worker reports |Im|, so the
    # conjugate partner folds onto the same entry; what is asserted is that SOME eigenvalue sits at
    # Re=0, Im=+-omega, which is the definition of the Hopf the tracker converged onto.
    on_axis = numpy.absolute(tracked - 1j * abs(omega))
    assert numpy.amin(on_axis) < 1e-5, (tracked, omega)
    # B = 1 + A^2 = 2 with A = 1, up to the diffusive correction of a finite domain.
    assert numpy.isclose(res["param"], 2.0, rtol=1e-4), res["param"]


def test_pitchfork(tmp_path):
    """Reaction-diffusion pitchfork at lam = 2*pi^2: also a zero eigenvalue."""
    res = _run(tmp_path, "pitchfork")
    _assert_base_layout(res)
    _assert_matches_untracked(res)
    tracked, _ = _spectra(res)
    assert numpy.amin(numpy.absolute(tracked)) < 1e-5, tracked
    assert numpy.isclose(res["param"], 2 * numpy.pi ** 2, rtol=1e-3), res["param"]


def test_azimuthal_m0_spectrum_while_tracking_m1(tmp_path):
    """The axisymmetric spectrum, taken while an m=1 bifurcation is being tracked.

    The interesting case, and the one the mode policy exists for: the eigensolve's azimuthal mode is
    independent of the tracked one. It is available here because azimuthal tracking has ALREADY
    released the strong axis conditions, so nothing has to be renumbered; the m=0 axis conditions
    come back as a matrix manipulator instead.
    """
    res = _run(tmp_path, "azimuthal")
    _assert_base_layout(res)
    _assert_matches_untracked(res)
    assert numpy.isclose(res["param"], 24.552372532331, rtol=1e-8), res["param"]
    # The tracker reads the same global "azimuthal_m" parameter that the eigensolve just retuned to
    # 0 and back. If it were left at 0, every subsequent tracking assembly would silently be the
    # axisymmetric one.
    assert res["m_after_eigen"] == 1.0, res["m_after_eigen"]
    # And the augmented system itself is unharmed: re-converging lands on the same critical value.
    assert numpy.isclose(res["param_after_eigen"], res["param"], rtol=1e-10), res
    tracked, _ = _spectra(res)
    # The m=0 spectrum is NOT the m=1 one: the tracked mode's zero eigenvalue must not appear here,
    # which is what says the manipulator and the mode switch actually took effect rather than the
    # eigensolve quietly staying at m=1.
    assert numpy.amin(numpy.absolute(tracked)) > 1e-3, tracked


def test_refusals(tmp_path):
    """The two ways an eigensolve while tracking must be turned away, and what a refusal leaves.

    Both are on the same problem, an axisymmetric VECTOR field with an AxisymmetryBC -- a scalar
    field needs no axis condition at m=0 at all, so it has no mode-dependent Dirichlet machinery and
    nothing to refuse.
    """
    res = _run(tmp_path, "eigen_refusals", size=6)

    # The default shift of solve_eigenproblem is 0, and it is the one value that cannot work: the
    # tracker has put lambda = 0 exactly on the spectrum.
    assert res["zero_shift_refused"], res
    assert "NON-ZERO shift" in res["zero_shift_message"], res["zero_shift_message"]
    assert res["none_shift_refused"], res

    # m != 0 while tracking a fold: releasing the axis conditions would renumber.
    assert res["m1_refused"], res
    assert "renumbered" in res["m1_message"], res["m1_message"]
    # The refusal has to undo the flag flip that _before_eigen_solve already performed while working
    # out its answer. Without the snapshot this is False and the problem is left describing boundary
    # conditions its equation numbering does not have -- silently, until the next solve.
    assert res["dirichlet_flags_restored"], res

    # m = 0 still works after both refusals, on the base row layout...
    assert res["eig_nrow"] == res["ndof_after_deactivate"] < res["ndof"], res
    assert len(res["m0_eig_re"]) == 3, res
    # ...and the tracker itself is unharmed: it still converges.
    assert numpy.isfinite(res["param_after_refusals"]), res
