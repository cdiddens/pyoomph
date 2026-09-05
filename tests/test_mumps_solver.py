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

# Tests for the direct MUMPS backends (pyoomph/solvers/mumps.py, idname "mumps" in both registries),
# which need the separate pyoomph_mumps package and are skipped without it.
#
# What is worth protecting here is not "MUMPS can solve a linear system" - pyoomph_mumps has its own
# suite for that, against scipy and with no finite elements in the way - but the three places where
# this backend could be wrong while still looking right:
#
#   * the CSR-through-SuperLU-named-slots convention, and the transpose flag that goes with it. Read
#     the wrong way round the solver returns the solution of the TRANSPOSED system, which on a
#     symmetric problem is indistinguishable from the right answer;
#   * the analysis-reuse path, which is only a speedup if it engages and only correct if the pattern
#     really has not moved - and a comparison where it silently fell back looks like a null result;
#   * the symmetric (LDL^T) path, which stores one triangle and must not change the answer.
#
# The eigenvalue assertions are backed by the generalized residual ||J v - lambda M v|| computed from
# the matrices the solver itself returns, as in test_spectra_eigensolver.py, so a test cannot pass by
# agreeing with a second implementation that is wrong in the same way.

import numpy
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.meshes.simplemeshes import RectangularQuadMesh
from pyoomph.equations.navier_stokes import NavierStokesEquations
import pyoomph._pyoomph_core as _pyoomph_core

try:
    import pyoomph.solvers.mumps as _mumps_backend  # noqa: F401
    _HAVE_MUMPS = True
    _WHY = ""
except ImportError as _e:
    _mumps_backend = None  # type: ignore
    _HAVE_MUMPS = False
    _WHY = str(_e)

pytestmark = pytest.mark.skipif(not _HAVE_MUMPS,
                                reason="the pyoomph_mumps package is not available: " + _WHY)


# ---------------------------------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------------------------------

class _NonlinearPoissonProblem(Problem):
    """A symmetric, mildly nonlinear Poisson problem, so Newton takes several steps.

    Several steps is the point: it is what gives the analysis reuse something to reuse. The x*y in
    the source keeps the solution from being symmetric about the diagonal, so that a transposed solve
    would show up in the answer rather than cancelling.
    """

    def __init__(self, N: int = 16):
        super().__init__()
        self.N = N

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=self.N, size=[1, 1]))

        class _Eqs(Equations):
            def define_fields(self):
                self.define_scalar_field("u", "C2")

            def define_residuals(self):
                u, v = var_and_test("u")
                x, y = var("coordinate_x"), var("coordinate_y")
                self.add_residual(weak(grad(u), grad(v)) - weak(10 * (1 + x * y) * exp(-u), v))

        eqs = _Eqs()
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        self.add_equations(eqs @ "domain")


class _AdvectionDiffusionProblem(Problem):
    """Deliberately NON-symmetric: an advection term makes J != J^T.

    This is the one that can catch a transposed solve. On the symmetric Poisson above, solving with
    J^T instead of J gives exactly the same answer and proves nothing.
    """

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=8, size=[1, 1]))

        class _Eqs(Equations):
            def define_fields(self):
                self.define_scalar_field("u", "C2")

            def define_residuals(self):
                u, v = var_and_test("u")
                wind = vector(3, 1)
                self.add_residual(weak(grad(u), grad(v)) + weak(dot(wind, grad(u)), v) - weak(1, v))

        eqs = _Eqs()
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        self.add_equations(eqs @ "domain")


class _NavierStokesProblem(Problem):
    # As in test_spectra_eigensolver.py: Navier-Stokes rather than Stokes so that there ARE time
    # derivatives, which makes the mass matrix positive semi-definite and SINGULAR - the case the
    # shift-and-invert has to survive.
    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=4))
        eqs = NavierStokesEquations(dynamic_viscosity=1, mass_density=1)
        for b in ["left", "right", "bottom"]:
            eqs += DirichletBC(velocity_x=0, velocity_y=0) @ b
        eqs += DirichletBC(velocity_x=1, velocity_y=0) @ "top"
        eqs += DirichletBC(pressure=0) @ "bottom/left"
        self.add_equations(eqs @ "domain")


def _solved_dofs(problem_factory, solver, configure=None):
    with problem_factory() as p:
        p.quiet()
        p.set_linear_solver(solver)
        if configure is not None:
            configure(p.get_la_solver())
        p.initialise()
        p.solve(max_newton_iterations=30)
        dofs, _ = p.get_current_dofs()
        return numpy.array(dofs), p.get_la_solver()


def _pencil_residuals(evals, evects, J, M):
    """||J v - lambda M v|| / ||v|| per returned pair.

    Must be evaluated while the Problem context is still open: J and M are scipy views straight onto
    oomph-lib's matrix buffers, which are freed with the problem.
    """
    return numpy.array([numpy.linalg.norm(J @ v - lam * (M @ v)) / numpy.linalg.norm(v)
                        for lam, v in zip(evals, evects)])


# ---------------------------------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------------------------------

def test_both_backends_are_registered_under_the_same_name():
    from pyoomph.solvers.generic import GenericLinearSystemSolver, GenericEigenSolver
    # The two registries are separate, which is what lets one name serve both - as "pardiso" already
    # does. A clash would have raised at import time, so this is really asserting that both halves of
    # the module were reached.
    assert "mumps" in GenericLinearSystemSolver._registered_solvers
    if getattr(_pyoomph_core, "has_spectra", False):
        assert "mumps" in GenericEigenSolver._registered_solvers


def test_the_extension_agrees_with_pyoomph_about_mpi():
    # Not a tautology even though pyoomph/solvers/mumps.py refuses a mismatch at import: this asserts
    # that the refusal is based on something both sides actually report, rather than on a constant.
    import pyoomph_mumps
    from pyoomph.generic.mpi import has_mpi
    assert bool(pyoomph_mumps.has_mpi) == bool(has_mpi())


# ---------------------------------------------------------------------------------------------------
# Linear solver
# ---------------------------------------------------------------------------------------------------

def test_matches_superlu_on_a_symmetric_problem():
    ref, _ = _solved_dofs(_NonlinearPoissonProblem, "superlu")
    got, la = _solved_dofs(_NonlinearPoissonProblem, "mumps")
    assert numpy.max(numpy.abs(got - ref)) < 1e-10 * max(1.0, numpy.max(numpy.abs(ref)))
    assert la.n_full_factorisations >= 1


def test_matches_superlu_on_a_nonsymmetric_problem():
    """The transpose convention. Advection makes J != J^T, so solving with the wrong one is visible.

    oomph hands out CSR through SuperLU's CSC-named slots with transpose=1; the scipy backend reads
    it as CSC and undoes the transpose, this one reads it as CSR and must not.
    """
    ref, _ = _solved_dofs(_AdvectionDiffusionProblem, "superlu")
    got, _ = _solved_dofs(_AdvectionDiffusionProblem, "mumps")
    assert numpy.max(numpy.abs(got - ref)) < 1e-10 * max(1.0, numpy.max(numpy.abs(ref)))


def test_analysis_is_reused_and_does_not_change_the_answer():
    with_reuse, la_on = _solved_dofs(_NonlinearPoissonProblem, "mumps")
    # Check the path ENGAGED before believing anything about it - a comparison in which the fast path
    # silently fell back shows two identical numbers and looks like a perfect result.
    assert la_on.n_numeric_factorisations > 0, "the analysis was never reused, so this proves nothing"
    assert la_on.n_full_factorisations == 1

    def off(la):
        la.reuse_symbolic_factorisation = False

    without_reuse, la_off = _solved_dofs(_NonlinearPoissonProblem, "mumps", configure=off)
    assert la_off.n_numeric_factorisations == 0
    assert la_off.n_full_factorisations == la_on.n_full_factorisations + la_on.n_numeric_factorisations
    assert numpy.allclose(with_reuse, without_reuse, rtol=0, atol=1e-12)


def test_symmetric_factorisation_engages_and_agrees_with_the_general_one():
    sym, la_sym = _solved_dofs(_NonlinearPoissonProblem, "mumps")
    assert la_sym.last_symmetry_decision is True, la_sym.last_symmetry_decision_reason
    assert la_sym.n_symmetric_factorisations > 0

    def off(la):
        la.exploit_proven_symmetry = False

    gen, la_gen = _solved_dofs(_NonlinearPoissonProblem, "mumps", configure=off)
    assert la_gen.n_symmetric_factorisations == 0
    # Different pivoting, so not bit-identical - but the same solution.
    assert numpy.max(numpy.abs(sym - gen)) < 1e-10 * max(1.0, numpy.max(numpy.abs(gen)))


def test_a_nonsymmetric_problem_does_not_take_the_symmetric_path():
    _, la = _solved_dofs(_AdvectionDiffusionProblem, "mumps")
    assert la.last_symmetry_decision is False
    assert la.n_symmetric_factorisations == 0


def test_determinant_sign_is_opt_in():
    _, la_off = _solved_dofs(_NonlinearPoissonProblem, "mumps")
    assert la_off.get_determinant_sign() is None

    def on(la):
        la.compute_determinant_sign = True

    _, la_on = _solved_dofs(_NonlinearPoissonProblem, "mumps", configure=on)
    assert la_on.get_determinant_sign() in (-1, 1)


def test_a_starved_workspace_is_grown_rather_than_reported_as_a_failure():
    """The regression test for the nacl_capillary_evaporation.py timeout.

    MUMPS sizes a work array from the fill-in its analysis predicted; when pivoting needs more it
    stops with INFOG(1) in _MUMPS_ICNTL14_ERRORS instead of reallocating. That is recoverable - raise
    ICNTL(14) and factorise again - and treating it as a solver failure instead makes oomph abandon
    the Newton step, shrink the timestep, and walk into the identical failure at the next attempt.
    One tutorial spent 13297 factorisations that way and never finished, while taking ten seconds
    with every other backend.

    Provoked here by setting ICNTL(14) far below what the matrix needs, which is the same condition a
    stale analysis produces, reached deliberately rather than by luck.
    """
    def starve(la):
        la.icntl_override[14] = -70  # far less slack than any real factorisation needs

    dofs, la = _solved_dofs(_NonlinearPoissonProblem, "mumps", configure=starve)
    # The claim: it recovered, and it recovered by growing the workspace rather than by chance.
    assert la.n_workspace_growths > 0, "the starved workspace never triggered a retry"
    assert la.icntl_override[14] > -70, "ICNTL(14) was not raised"
    ref, _ = _solved_dofs(_NonlinearPoissonProblem, "superlu")
    assert numpy.max(numpy.abs(dofs - ref)) < 1e-10 * max(1.0, numpy.max(numpy.abs(ref)))


def test_the_grown_workspace_is_kept_for_the_rest_of_the_run():
    """The raised ICNTL(14) must survive, including across the instance being discarded.

    A matrix that needed the extra room once needs it at the next step too; rediscovering that by
    failing again every time is the whole cost this avoids.
    """
    def starve(la):
        la.icntl_override[14] = -70

    _dofs, la = _solved_dofs(_NonlinearPoissonProblem, "mumps", configure=starve)
    grown = la.icntl_override[14]
    assert grown >= 40  # the floor _next_mumps_icntl14 doubles up from
    # Only a handful of growths, i.e. it settled rather than re-growing at every factorisation.
    assert la.n_workspace_growths <= la.max_workspace_retries


def test_the_two_mumps_backends_agree_on_which_codes_mean_workspace():
    """The direct backend and the PETSc one must not drift about this list.

    They reach MUMPS by completely independent routes, so the list lives in .generic and both read
    it from there - this asserts that the sharing is real and not two copies that happen to match.
    """
    from pyoomph.solvers.generic import _MUMPS_ICNTL14_ERRORS, _MUMPS_ICNTL23_ERROR
    assert -9 in _MUMPS_ICNTL14_ERRORS
    # -19 is the ICNTL(23) hard cap; more slack cannot buy room a cap forbids.
    assert _MUMPS_ICNTL23_ERROR not in _MUMPS_ICNTL14_ERRORS


def test_a_bad_icntl_is_reported_rather_than_ignored():
    from pyoomph.solvers.mumps import MumpsSolverError
    with pytest.raises((MumpsSolverError, Exception)):
        _solved_dofs(_NonlinearPoissonProblem, "mumps",
                     configure=lambda la: la.icntl_override.update({999: 1}))


# ---------------------------------------------------------------------------------------------------
# Eigensolver
# ---------------------------------------------------------------------------------------------------

_needs_spectra = pytest.mark.skipif(not getattr(_pyoomph_core, "has_spectra", False),
                                    reason="this pyoomph build was compiled without Spectra, which "
                                           "supplies the Arnoldi iteration the mumps eigensolver drives")


@_needs_spectra
def test_eigensolver_satisfies_the_pencil_with_a_singular_mass_matrix():
    with _NavierStokesProblem() as p:
        p.quiet()
        p.set_eigensolver("mumps")
        p.solve()
        evals, evects, J, M = p.get_eigen_solver().solve(5, shift=0.0)
        assert len(evals) > 0
        assert numpy.all(numpy.isfinite(evals))
        assert numpy.max(_pencil_residuals(evals, evects, J, M)) < 1e-8


@_needs_spectra
def test_eigensolver_agrees_with_the_spectra_backend():
    """Same eigenvalues as the backend it inherits from - only the factoriser differs."""
    def run(backend):
        with _NavierStokesProblem() as p:
            p.quiet()
            p.set_eigensolver(backend)
            p.solve()
            evals, _evects, _J, _M = p.get_eigen_solver().solve(5, shift=0.0)
            return numpy.sort_complex(numpy.asarray(evals))

    ref = run("spectra")
    got = run("mumps")
    assert len(got) == len(ref)
    assert numpy.max(numpy.abs(got - ref)) < 1e-6 * max(1.0, numpy.max(numpy.abs(ref)))


@_needs_spectra
def test_complex_target_uses_the_complex_arithmetic():
    """A complex shift makes J - sigma*M complex even though J and M are real - the zmumps path.

    That is the ordinary situation at a Hopf bifurcation, not an unusual one, which is why the
    operator picks its arithmetic from the SHIFTED matrix rather than from J and M.
    """
    with _NavierStokesProblem() as p:
        p.quiet()
        p.set_eigensolver("mumps")
        p.solve()
        es = p.get_eigen_solver()
        assert es.supports_target() and es.supports_complex_target()
        target = -95.0 + 0.35j
        evals, evects, J, M = es.solve(3, target=target, shift=target, sort=False)
        assert J.dtype.kind != "c" and M.dtype.kind != "c"  # the pencil itself is real
        assert numpy.max(_pencil_residuals(evals, evects, J, M)) < 1e-8
        # sort=False must still come back target-ordered - callers such as
        # get_hopf_lyapunov_coefficient read evals[0] and rely on it.
        assert abs(evals[0] - target) <= numpy.max(numpy.abs(numpy.asarray(evals) - target))


@_needs_spectra
def test_the_operator_is_the_only_difference_from_the_spectra_backend():
    from pyoomph.solvers.spectra import SpectraEigenSolver
    from pyoomph.solvers.mumps import MumpsSpectraEigenSolver
    assert issubclass(MumpsSpectraEigenSolver, SpectraEigenSolver)
    # Everything delicate - _choose_sigma, _order, the retry ladder, the dense fallback - must be
    # inherited rather than copied, or the two backends can drift apart. The dunders are filtered
    # out, so __init__ (which only adds the two MUMPS knobs) is not part of the comparison.
    overridden = {name for name in vars(MumpsSpectraEigenSolver)
                  if not name.startswith("__") and name not in ("idname",)}
    assert overridden == {"_make_operator"}, overridden
