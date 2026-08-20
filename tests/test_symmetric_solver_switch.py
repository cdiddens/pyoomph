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

# The whole-matrix symmetry verdict (Problem::get_proven_matrix_symmetry) and the solver switch built
# on it (exploit_proven_symmetry, ON by default).
#
# The contract being protected is one-sided, like the per-block flags it reduces
# (test_jacobian_block_flags.py): True is a PROOF the solvers act on without checking - Pardiso then
# factorises via mtype -2, the scipy eigensolver runs eigsh - so a wrong True is silently wrong
# linear algebra. False only ever costs the general factorisation. Hence:
#
#   - every True verdict here is backed by an A/B against the general path (same solution/eigenvalues),
#   - the runtime gates are checked in BOTH directions: a bifurcation tracker or a matrix manipulator
#     must force False the moment it is installed, and the verdict must come BACK when it is removed -
#     the flip is where the caching bugs live (an mtype -2 symbolic factorisation refreshed with an
#     augmented matrix, or vice versa), which is why the fold-tracking test below checks the Pardiso
#     factorisation counters across the whole cycle.

import numpy
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.meshes.simplemeshes import LineMesh, RectangularQuadMesh
from pyoomph.equations.ALE import LaplaceSmoothedMesh
from pyoomph.equations.navier_stokes import NavierStokesEquations, StokesEquations
from pyoomph.solvers.generic import EigenMatrixSetDofsToZero


# ---------------------------------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------------------------------

class _PoissonProblem(Problem):
    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=4))

        class _Eqs(Equations):
            def define_fields(self):
                self.define_scalar_field("u", "C2")

            def define_residuals(self):
                u, v = var_and_test("u")
                self.add_residual(weak(grad(u), grad(v)) - weak(1, v))

        eqs = _Eqs()
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        self.add_equations(eqs @ "domain")


class _TransientDiffusionProblem(Problem):
    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=4))

        class _Eqs(Equations):
            def define_fields(self):
                self.define_scalar_field("u", "C2")

            def define_residuals(self):
                u, v = var_and_test("u")
                self.add_residual(weak(partial_t(u), v) + weak(grad(u), grad(v)) - weak(1, v))

        eqs = _Eqs()
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        self.add_equations(eqs @ "domain")


class _CavityProblem(Problem):
    def __init__(self, navier=False):
        super().__init__()
        self.navier = navier

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=4))
        if self.navier:
            eqs = NavierStokesEquations(dynamic_viscosity=1, mass_density=1)
        else:
            eqs = StokesEquations(dynamic_viscosity=1)
        for b in ["left", "right", "bottom"]:
            eqs += DirichletBC(velocity_x=0, velocity_y=0) @ b
        eqs += DirichletBC(velocity_x=1, velocity_y=0) @ "top"
        eqs += DirichletBC(pressure=0) @ "bottom/left"
        self.add_equations(eqs @ "domain")


class _MovingMeshProblem(Problem):
    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=4))

        class _Eqs(Equations):
            def define_fields(self):
                self.define_scalar_field("u", "C2")

            def define_residuals(self):
                u, v = var_and_test("u")
                self.add_residual(weak(grad(u), grad(v)) - weak(1, v))

        eqs = _Eqs() + LaplaceSmoothedMesh()
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0, mesh_x=True, mesh_y=True) @ b
        self.add_equations(eqs @ "domain")


class _WeakBCSaddlePointProblem(Problem):
    """Poisson whose top boundary condition is imposed weakly, by a Lagrange-multiplier FIELD.

    [[A, B^T], [B, 0]]: symmetric, and every multiplier row has a ZERO diagonal. That is what makes
    MKL perturb pivots under mtype -2 (13 of them here), which is the situation the symmetric path has
    to survive - see test_symmetric_saddle_point_reaches_machine_zero.
    """

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=6))

        class _Bulk(Equations):
            def define_fields(self):
                self.define_scalar_field("u", "C2")

            def define_residuals(self):
                u, v = var_and_test("u")
                self.add_residual(weak(grad(u), grad(v)) - weak(1, v))

        class _WeakBC(InterfaceEquations):
            required_parent_type = Equations

            def define_fields(self):
                self.define_scalar_field("lambda_u", "C2")

            def define_residuals(self):
                lam, lamtest = var_and_test("lambda_u")
                u, utest = var_and_test("u")
                # Both halves written with the same weak(), so the two off-diagonal blocks really are
                # each other's transpose and the whole matrix is symmetric.
                self.add_residual(weak(lam, utest) + weak(u, lamtest))

        eqs = _Bulk()
        # Strongly pinned on the bottom only. Pinning the sides as well would put a Dirichlet
        # condition on the two top CORNER nodes, whose multipliers then constrain nothing and are
        # free to take any value at all -- a genuinely singular pair of rows, which is a different
        # (and much less interesting) reason for Pardiso to perturb pivots.
        eqs += DirichletBC(u=0) @ "bottom"
        eqs += _WeakBC() @ "top"
        self.add_equations(eqs @ "domain")


class _BratuProblem(Problem):
    """u'' + lambda*exp(u) = 0: symmetric Jacobian AND a genuine fold (at lambda ~ 3.5138), so the
    same problem exercises the verdict, the tracker gate and the flip back - end to end."""

    def define_problem(self):
        self.lam = self.define_global_parameter(lam=1.0)
        self.add_mesh(LineMesh(N=20))

        class _Eqs(Equations):
            def define_fields(self):
                self.define_scalar_field("u", "C2")

            def define_residuals(self2):
                u, v = var_and_test("u")
                # partial_t vanishes on the steady branch, so it moves no fold - it is only here to
                # give the eigenproblem (which supplies the tracker's eigenvector guess) a mass matrix.
                self2.add_residual(weak(partial_t(u), v) + weak(grad(u), grad(v)) - weak(self.lam * exp(u), v))

        eqs = _Eqs()
        for b in ["left", "right"]:
            eqs += DirichletBC(u=0) @ b
        self.add_equations(eqs @ "domain")


# ---------------------------------------------------------------------------------------------------
# The verdict itself
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("factory, jac_sym", [
    (lambda: _PoissonProblem(), True),
    (lambda: _TransientDiffusionProblem(), True),
    # Stokes is False, not True: the u-u block is proven symmetric, but pyoomph's velocity-pressure
    # pair is ANTIsymmetric (see test_the_velocity_pressure_pair_is_antisymmetric), so the assembled
    # matrix as a whole is genuinely not symmetric with this sign convention.
    (lambda: _CavityProblem(navier=False), False),
    (lambda: _CavityProblem(navier=True), False),
    (lambda: _MovingMeshProblem(), False),
], ids=["poisson", "transient_diffusion", "stokes", "navier_stokes", "moving_mesh"])
def test_whole_matrix_verdict(factory, jac_sym):
    with factory() as p:
        p.quiet()
        p.initialise()
        got_jac, _got_mass = p._get_proven_matrix_symmetry("")
        assert got_jac == jac_sym


def test_mass_matrix_verdict_on_transient_diffusion():
    with _TransientDiffusionProblem() as p:
        p.quiet()
        p.initialise()
        assert p._get_proven_matrix_symmetry("") == (True, True)


# ---------------------------------------------------------------------------------------------------
# The Pardiso switch, its A/B, and the tracker gate with the mtype flip
# ---------------------------------------------------------------------------------------------------

def _pardiso_or_skip(p):
    s = p.get_la_solver()
    if type(s).__name__ != "PardisoSolver":
        pytest.skip("MKL Pardiso is not the default solver on this machine")
    return s


def test_pardiso_engages_and_matches_the_general_path():
    with _PoissonProblem() as p:
        p.quiet()
        p.initialise()
        s = _pardiso_or_skip(p)
        p.solve()
        assert s.last_symmetry_decision is True
        assert s._current_pardiso is not None and s._current_pardiso.mtype == -2
        assert s.n_symmetric_factorisations > 0
        dofs_sym = numpy.array(p.get_current_dofs()[0])

        # Same problem, symmetric path off: the answer must be the same to roundoff. Perturb first,
        # or the second solve converges in zero iterations and factorises nothing.
        s.exploit_proven_symmetry = False
        p.set_current_dofs(dofs_sym + 0.01)
        p.solve()
        assert s.last_symmetry_decision is False
        assert s._current_pardiso.mtype == 11
        assert numpy.max(numpy.abs(numpy.array(p.get_current_dofs()[0]) - dofs_sym)) < 1e-9


def test_fold_tracking_gates_the_verdict_and_flips_the_mtype():
    """The correctness trap this feature must not fall into: the fold-augmented system is structurally
    non-symmetric even though the Bratu Jacobian is symmetric. So the verdict must drop to False the
    moment the tracker is installed, Pardiso must refuse to reuse the mtype -2 factorisation for it,
    and everything must come back - with a FRESH symbolic factorisation - on deactivation."""
    with _BratuProblem() as p:
        p.quiet()
        p.setup_for_stability_analysis(analytic_hessian=True)
        p.set_eigensolver("scipy")
        p.initialise()
        s = _pardiso_or_skip(p)

        p.lam.value = 3.0
        p.solve()
        assert p._get_proven_matrix_symmetry("") == (True, True)
        assert s._current_pardiso.mtype == -2

        p.solve_eigenproblem(1)
        p.activate_bifurcation_tracking("lam", bifurcation_type="fold", eigenvector=0)
        assert p._get_proven_matrix_symmetry("") == (False, False)
        p.solve()
        # The literature fold of the 1D Bratu problem - the tracking converged on the augmented,
        # NON-symmetric system, so the general factorisation must have been used throughout.
        assert abs(p.lam.value - 3.513830719) < 1e-4
        assert s.last_symmetry_decision is False
        assert s._current_pardiso.mtype == 11

        p.deactivate_bifurcation_tracking()
        assert p._get_proven_matrix_symmetry("") == (True, True)
        n_full_before = s.n_full_factorisations
        # Perturb, or this solve converges in zero iterations off the tracked state and factorises
        # nothing at all - the flip back would go untested.
        p.set_current_dofs(numpy.array(p.get_current_dofs()[0]) + 1e-3)
        p.solve()
        assert s._current_pardiso.mtype == -2
        # The flip back must NOT have been served from any reuse tier: an mtype 11 symbolic
        # factorisation refreshed as -2 would be silently wrong, so it has to be a full rebuild.
        assert s.n_full_factorisations > n_full_before


# ---------------------------------------------------------------------------------------------------
# The eigen switch
# ---------------------------------------------------------------------------------------------------

def _eigen_ab(p, neval=5):
    es = p.get_eigen_solver()
    es.exploit_proven_symmetry = True
    ev_on, _ = p.solve_eigenproblem(neval)
    dec_on = es.last_symmetry_decision
    es.exploit_proven_symmetry = False
    ev_off, _ = p.solve_eigenproblem(neval)
    return numpy.sort_complex(ev_on), dec_on, numpy.sort_complex(ev_off)


def test_scipy_eigsh_engages_and_matches_eigs():
    with _TransientDiffusionProblem() as p:
        p.quiet()
        p.set_eigensolver("scipy")
        p.initialise()
        p.solve()
        ev_on, dec_on, ev_off = _eigen_ab(p)
        assert dec_on is True
        scale = max(1.0, float(numpy.max(numpy.abs(ev_off))))
        assert numpy.max(numpy.abs(ev_on - ev_off)) < 1e-6 * scale


def test_matrix_manipulators_force_the_general_driver():
    # Checked on the helper directly: solve_eigenproblem() manages the manipulator list itself
    # (setup_forced_zero_dof_list_for_eigenproblems flushes it and installs the forced-zero-dof
    # ones), so a manipulator added here would be cleared before the solve ever sees it. The gate
    # fires on whatever is in the list at decision time - which is exactly that framework-installed
    # set, e.g. the axis dofs of an azimuthal eigenproblem.
    with _TransientDiffusionProblem() as p:
        p.quiet()
        p.set_eigensolver("scipy")
        p.initialise()
        es = p.get_eigen_solver()
        assert es._use_symmetric_eigensolver_now() is True
        es.add_matrix_manipulator(EigenMatrixSetDofsToZero(p))
        assert es._use_symmetric_eigensolver_now() is False
        assert "manipulator" in es.last_symmetry_decision_reason
        es.clear_matrix_manipulators()
        assert es._use_symmetric_eigensolver_now() is True


def test_slepc_ghep_matches_gnhep():
    pytest.importorskip("slepc4py", reason="slepc4py not available (PYTHONPATH must carry a PETSc build)")
    with _TransientDiffusionProblem() as p:
        p.quiet()
        p.set_eigensolver("slepc")
        p.initialise()
        p.solve()
        ev_on, dec_on, ev_off = _eigen_ab(p)
        assert dec_on is True
        scale = max(1.0, float(numpy.max(numpy.abs(ev_off))))
        assert numpy.max(numpy.abs(ev_on - ev_off)) < 1e-6 * scale


# ---------------------------------------------------------------------------------------------------
# Symmetric but INDEFINITE mass matrix
# ---------------------------------------------------------------------------------------------------

class _PendulumProblem(Problem):
    # phi'=psi, psi'=-sin(phi): both J and M are symmetric, but partial_t couples the two DIFFERENT
    # fields, so M=[[0,1],[1,0]] - symmetric with eigenvalues +-1. Nothing here is a Gram matrix.
    def define_problem(self):
        class _Eqs(ODEEquations):
            def define_fields(self):
                self.define_ode_variable("phi", "psi")

            def define_residuals(self):
                phi, phi_test = var_and_test("phi")
                psi, psi_test = var_and_test("psi")
                self.add_residual((partial_t(psi) + sin(phi)) * phi_test)
                self.add_residual((partial_t(phi) - psi) * psi_test)

        self.add_equations(_Eqs() @ "pendulum")


@pytest.mark.parametrize("solver", ["scipy", "slepc"])
def test_indefinite_mass_matrix_falls_back_to_the_general_driver(solver):
    # The symmetric drivers use M as an inner product and need it positive semi-definite on top of
    # symmetric. Engaging them here made SLEPc abort outright ("The inner product is not well
    # defined: indefinite matrix"), so the screen must veto the symmetric path and the eigenvalues
    # must come out as the exact +-i of the linearised pendulum.
    if solver == "slepc":
        pytest.importorskip("slepc4py", reason="slepc4py not available (PYTHONPATH must carry a PETSc build)")
    with _PendulumProblem() as p:
        p.quiet()
        p.set_eigensolver(solver)
        ode = p.get_ode("pendulum")
        ode.set_value(phi=0.01, psi=0)
        p.solve()
        es = p.get_eigen_solver()
        es.exploit_proven_symmetry = True
        evs, _ = p.solve_eigenproblem(2)
        assert es.last_symmetry_decision is False
        assert "positive semi-definite" in es.last_symmetry_decision_reason
        assert numpy.max(numpy.abs(numpy.sort_complex(evs) - numpy.array([-1j, 1j]))) < 1e-8


# ---------------------------------------------------------------------------------------------------
# Perturbed pivots on the symmetric path
# ---------------------------------------------------------------------------------------------------

def test_symmetric_saddle_point_reaches_machine_zero():
    """MKL grants a general mtype up to two iterative-refinement steps by itself whenever it perturbed
    a pivot, and grants a symmetric one NONE. Left alone, mtype -2 therefore returns this Lagrange-
    multiplier system with a backward error of ~1e-5 and the Newton step stalls seven digits short - it
    reaches 5.0e-09 instead of 2.6e-16 (measured by disabling _needs_explicit_refinement), and 1e-10 on
    the coupled interfaces of test_adaptive_interface_coupling.py, which is how this was found.
    pardisoSolver asks MKL for those two steps itself."""
    with _WeakBCSaddlePointProblem() as p:
        p.quiet()
        p.initialise()
        s = _pardiso_or_skip(p)
        p.solve()
        ps = s._current_pardiso
        assert s.last_symmetry_decision is True and ps.mtype == -2
        # The premise: with no perturbed pivot there is nothing to refine and the test would pass
        # whatever the refinement does.
        assert ps.iparm[13] > 0, "no pivot was perturbed, so the refinement under test never applied"
        assert ps.iparm[6] > 0, "MKL was not asked to refine the symmetric solve"
        conv = p.get_last_residual_convergence()
        assert conv[1] / conv[0] < 1e-12, \
            "one Newton step only reduced the residual by %.2e -- the linear solve is inaccurate, not " \
            "the Jacobian" % (conv[1] / conv[0])
