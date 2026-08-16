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

# Code generation for the azimuthal (m!=0) contributions on a MOVING mesh. Three defects lived here,
# all of which only appear when the expansion, the moving mesh and one further ingredient meet:
#
#  1. A cached subexpression derivative is a scalar C variable, so it must hold the pure partial
#     derivative. Differentiating by a coordinate field also produced the moving-mesh dpsi/dX terms,
#     which are l_shape-indexed arrays, and the generated code did not compile:
#         domain.c: error: 'l_shape' undeclared
#     Needs subexpr() + azimuthal + moving mesh + a second spatial derivative inside the subexpression.
#
#  2./3. absolute() and signum() did not tell GiNaC that they are real (and, for absolute(),
#     non-negative). power::real_part() only leaves a fractional power alone when its basis reports
#     info_flags::nonnegative, so tau = 1/sqrt(sum of squares) was rewritten into polar form,
#     |X|^p*(cos+I*sin)(p*atan2(Im,Re)), and the element failed to LOAD with
#         undefined symbol: imag_part
#     signum() gets there through d|x| = signum(x) dx.
#
# Both tests are cheap: tiny meshes, and no eigensolver is required.

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.ALE import LaplaceSmoothedMesh
from pyoomph.meshes.simplemeshes import RectangularQuadMesh


def _axisym_mesh():
    # away from r=0 on purpose, so nothing here depends on the axis treatment
    return RectangularQuadMesh(N=2, size=[1, 1], lower_left=[1, 0])


def test_azimuthal_subexpression_with_second_derivatives_on_moving_mesh():
    """subexpr() + azimuthal + moving mesh + second derivatives: must compile AND be correct.

    The augmented Jacobian of the azimuthal bifurcation tracker is checked against a central
    difference of the augmented residual: its eigenvector rows are built from the m!=0 Jacobian,
    which is what the cached subexpression derivatives feed, so a wrong cached partial shows up
    there. Noise floor ~1e-7.
    """
    import numpy

    class Eq(Equations):
        def define_fields(self):
            self.define_scalar_field("u", "C2")

        def define_residuals(self):
            u, v = var_and_test("u")
            A = self.get_current_code_generator().get_problem().get_global_parameter("A")
            # cached, solution dependent, and containing a second spatial derivative
            fac = subexpression(1 + dot(grad(u), grad(u)) + div(grad(u)))
            self.add_residual(weak(fac * grad(u), grad(v)) + weak(A * fac * u, v)
                              + weak(partial_t(u), v))

    class P(Problem):
        def define_problem(self):
            self.A = self.define_global_parameter(A=1.0)
            self.set_coordinate_system("axisymmetric")
            self.add_mesh(_axisym_mesh())
            eqs = Eq() + LaplaceSmoothedMesh()
            eqs += DirichletBC(u=0) @ "bottom" + DirichletBC(u=1) @ "top"
            for b in ("left", "right", "top", "bottom"):
                eqs += DirichletBC(mesh_x=True, mesh_y=True) @ b
            self.add_equations(eqs @ "domain")

    with P() as p:
        p.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
        p.initialise()          # this alone caught defect 1: the code did not compile
        nbase = p.ndof()

        rng = numpy.random.default_rng(3)
        x0 = numpy.array(p.get_current_dofs()[0]) + 0.02 * rng.standard_normal(nbase)
        p.set_current_dofs(x0)

        V = rng.standard_normal(nbase) + 1j * rng.standard_normal(nbase)
        V /= numpy.linalg.norm(V)
        p.activate_bifurcation_tracking("A", bifurcation_type="azimuthal", azimuthal_mode=1,
                                        eigenvector=V, omega=0.3)
        naug = p.ndof()
        xaug = numpy.array(p.get_current_dofs()[0])
        J = p.assemble_jacobian(with_residual=False)

        d = numpy.zeros(naug)
        d[:nbase] = rng.standard_normal(nbase)
        d /= numpy.linalg.norm(d)
        eps = 1e-6
        p.set_current_dofs(xaug + eps * d)
        rp = numpy.array(p.get_residuals())
        p.set_current_dofs(xaug - eps * d)
        rm = numpy.array(p.get_residuals())
        p.set_current_dofs(xaug)

        fd = (rp - rm) / (2 * eps)
        ana = J @ d
        lo, hi = nbase, min(3 * nbase, naug)      # the eigenvector rows
        rel = numpy.max(numpy.abs(ana[lo:hi] - fd[lo:hi])) / max(numpy.max(numpy.abs(fd[lo:hi])), 1e-30)
        assert rel < 1e-5, "azimuthal Jacobian disagrees with a finite difference: rel=%.3e" % rel


def test_stabilized_navier_stokes_with_azimuthal_stability_builds():
    """tau is 1/sqrt(sum of squares) built from subexpr(), which is what defects 2 and 3 hit.

    Assembling the Jacobian (not just initialise()) makes sure the generated element also *loads*,
    which is where "undefined symbol: imag_part" struck.
    """
    from pyoomph.equations.stabilized_ns import StabilizedNavierStokes

    class P(Problem):
        def define_problem(self):
            self.set_coordinate_system("axisymmetric")
            self.add_mesh(_axisym_mesh())
            eqs = StabilizedNavierStokes(space="C1C1", stabilization="SUPGPSPGLSIC",
                                         viscous_form="stress", dynamic_viscosity=1, mass_density=1)
            eqs += DirichletBC(velocity_x=0, velocity_y=0) @ "bottom"
            eqs += DirichletBC(velocity_x=0, velocity_y=0) @ "top"
            eqs += LaplaceSmoothedMesh()          # the moving mesh is part of the trigger
            for b in ("left", "right", "top", "bottom"):
                eqs += DirichletBC(mesh_x=True, mesh_y=True) @ b
            self.add_equations(eqs @ "domain")

    with P() as p:
        p.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
        p.initialise()
        J = p.assemble_jacobian(with_residual=False)
        assert J.nnz > 0


def test_element_size_expansion_defaults_to_frozen():
    """The element size must not follow the mesh in the mode expansion by default.

    Measured on an oscillating free drop against Lamb's analytic eigenvalue (the frequency depends
    only on the polar wavenumber l, so m=0/1/2 share a ground truth): with the element size expanded,
    the l=2 damping rate came out 664% too large at 2439 dofs (2549% one refinement coarser), because
    the perturbation of tau injects a spurious dissipation that only decays as the stabilization
    itself vanishes with h. Frozen is within 6% and converging. See
    pyoomph_runs/Bugs/AzimuthalTracking/oscillating_drop.py.
    """
    from pyoomph import _pyoomph

    class P(Problem):
        def define_problem(self):
            self.set_coordinate_system("axisymmetric")
            self.add_mesh(_axisym_mesh())
            self.add_equations(Equations() @ "domain")

    with P() as p:
        p.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
        assert _pyoomph.get_expand_element_size_in_expansion_modes() is False
        p.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False,
                                       expand_element_size=True)
        assert _pyoomph.get_expand_element_size_in_expansion_modes() is True
        # leave the global in its default state for whatever runs next
        p.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)


def test_augmented_azimuthal_jacobian_is_exact_with_a_complex_mass_matrix():
    """Every block of the azimuthal tracker's augmented Jacobian, against a full finite difference.

    The eq_V_im residual carried `+ Omega*M_imag*Vi` where the eigenproblem (and pyoomph's own
    Python tracker, and its own get_dresiduals_dparameter) use `- Omega*M_imag*Vi`. It was invisible
    for every unstabilized formulation, whose m!=0 mass matrix has no imaginary part; a residual-based
    stabilization gives it one, and then the tracker solved a slightly wrong system - the rising
    bubble's critical Bond number came out 0.85% off and Newton converged only linearly.

    The check finite-differences EVERY augmented column, so it also covers the Vr/Vi blocks that a
    single directional probe misses.
    """
    import numpy
    from pyoomph.equations.stabilized_ns import StabilizedNavierStokes

    class P(Problem):
        def define_problem(self):
            self.A = self.define_global_parameter(A=1.0)
            self.set_coordinate_system("axisymmetric")
            self.add_mesh(_axisym_mesh())
            eqs = StabilizedNavierStokes(space="C1C1", stabilization="SUPGPSPGLSIC",
                                         viscous_form="stress", dynamic_viscosity=self.A,
                                         mass_density=1)
            eqs += DirichletBC(velocity_x=0, velocity_y=0) @ "bottom"
            eqs += LaplaceSmoothedMesh()
            eqs += DirichletBC(mesh_x=True, mesh_y=True) @ "bottom"
            self.add_equations(eqs @ "domain")

    with P() as p:
        p.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=True)
        p.initialise()
        n = p.ndof()
        rng = numpy.random.default_rng(11)
        p.set_current_dofs(numpy.array(p.get_current_dofs()[0]) + 0.05 * rng.standard_normal(n))
        V = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        V /= numpy.linalg.norm(V)
        p.activate_bifurcation_tracking("A", bifurcation_type="azimuthal", azimuthal_mode=1,
                                        eigenvector=V, omega=0.3)
        naug = p.ndof()
        x0 = numpy.array(p.get_current_dofs()[0])
        J = numpy.asarray(p.assemble_jacobian(with_residual=False).todense())

        eps = 1e-6
        worst, worst_at = 0.0, ""
        lab = lambda i: ("base" if i < n else "Vr" if i < 2 * n else "Vi" if i < 3 * n
                         else ("param" if i == 3 * n else "omega"))
        for j in range(naug):
            x = x0.copy(); x[j] += eps
            p.set_current_dofs(x); rp = numpy.array(p.get_residuals())
            x = x0.copy(); x[j] -= eps
            p.set_current_dofs(x); rm = numpy.array(p.get_residuals())
            col_fd = (rp - rm) / (2 * eps)
            scale = max(numpy.max(numpy.abs(col_fd)), numpy.max(numpy.abs(J[:, j])), 1e-30)
            rel = numpy.max(numpy.abs(J[:, j] - col_fd)) / scale
            if rel > worst:
                worst, worst_at = rel, "%s column %d" % (lab(j), j)
        p.set_current_dofs(x0)
        assert worst < 1e-5, "augmented Jacobian disagrees with a finite difference: %.3e at %s" % (worst, worst_at)
