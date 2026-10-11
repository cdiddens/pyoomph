#  @file
#  @author Christian Diddens
#  @author Duarte Rocha
#  @author Maxim de Wildt
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
from pyoomph.expressions.units import meter, second
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


def test_deeply_nested_dimensional_subexpressions_build_under_azimuthal_stability():
    """A dimensional subexpr() chain of nesting depth 4 and fan-out 2, i.e. 16 markers as a tree.

    Under azimuthal_stability=True every marker is split into a real and an imaginary one, and the
    unit analysis in add_residual used to re-traverse the whole nesting once per level. Measured on
    this construction, add_residual per contribution (PYOOMPH_TIME_ADD_RESIDUAL=1):

        depth   before    after
          4     0.68 s    0.011 s
          5     2.74 s    0.061 s
          6    30.7  s    0.088 s

    No wall-clock bound is asserted here, because it would not measure this: what is left of
    initialise() for this construction is the *code writer*, which has the same DAG-as-tree problem
    and still grows by an order of magnitude per level (7.6 s / 76 s / 887 s at depths 4 / 5 / 6).
    See dev_docs/subexpression_unit_analysis_stall.md.

    What is checked is that the element builds, loads and assembles: a placeholder symbol left
    behind by the masking, or a unit left inside a marker, does not survive code generation.
    """
    import numpy

    depth = 4

    class Eq(Equations):
        def define_fields(self):
            self.define_scalar_field("u", "C2", scale=meter, testscale=1 / meter)

        def define_residuals(self):
            u, v = var_and_test("u")
            e = subexpression(u + 1 * meter)
            for _ in range(depth):
                e = subexpression(e * e / (1 * meter) + 0.5 * e * exp(-u / (1 * meter)))
            self.add_residual(weak(e, v) + weak(grad(u), grad(v)) * meter * meter)

    class P(Problem):
        def define_problem(self):
            self.set_coordinate_system("axisymmetric")
            self.set_scaling(spatial=meter, temporal=second)
            self.add_mesh(RectangularQuadMesh(N=2, size=[1 * meter, 1 * meter],
                                              lower_left=[1 * meter, 0]))
            self.add_equations((Eq() + DirichletBC(u=0) @ "bottom") @ "domain")

    with P() as p:
        p.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
        p.initialise()
        n = p.ndof()
        rng = numpy.random.default_rng(5)
        x0 = numpy.array(p.get_current_dofs()[0]) + 0.01 * rng.standard_normal(n)
        p.set_current_dofs(x0)
        r, J = p.assemble_jacobian(with_residual=True)
        r = numpy.array(r)
        assert numpy.all(numpy.isfinite(r))
        assert numpy.all(numpy.isfinite(numpy.asarray(J.todense())))

        d = rng.standard_normal(n)
        d /= numpy.linalg.norm(d)
        eps = 1e-7
        p.set_current_dofs(x0 + eps * d)
        rp = numpy.array(p.get_residuals())
        p.set_current_dofs(x0 - eps * d)
        rm = numpy.array(p.get_residuals())
        p.set_current_dofs(x0)
        fd = (rp - rm) / (2 * eps)
        ana = J @ d
        rel = numpy.max(numpy.abs(ana - fd)) / max(numpy.max(numpy.abs(fd)), 1e-30)
        assert rel < 1e-5, "Jacobian disagrees with a finite difference: rel=%.3e" % rel


# The azimuthal real/imaginary split (SubExpressionsToRealAndImag, src/expressions.cpp) memoises its
# results in a GiNaC::exmap. GiNaC hashes and compares numbers by VALUE, not by representation
# (numeric::calchash: "3 and 3.0 share the same hashvalue"), so an exact -2 and an inexact -2.0 were
# one and the same key there - and ex::compare() unifies two ex's it finds equal by rebinding one's
# pointer to the other's, so merely *looking up* the exact -2 that is the exponent of a power
# rewrote that power to X^(-2.0) in place. That is not cosmetic: power::real_part() uses its
# integer-binomial branch only for exponent.info(integer), so an inexact exponent sends it to the
# polar form |X|^c*cos(c*atan2(b,a)+...), and a dimensional basis inside an atan2 makes the unit
# analysis of a residual fail with "The added residual contribution is not dimensionless". Five
# parameter sets of an evaporating-droplet stability run died there.
#
# Both halves are pinned below: numbers must survive the split with their exactness intact, and
# GiNaC's real_part()/imag_part() must use the binomial branch for an inexact whole-number exponent
# too (citools/patches/ginac-inexact-whole-number-exponent.patch).

def _split(expr):
    from pyoomph import _pyoomph_core
    return _pyoomph_core.GiNaC_split_subexpressions_in_real_and_imaginary_parts(expr)


def _symbol(name):
    from pyoomph import _pyoomph_core
    return _pyoomph_core.GiNaC_new_symbol(name)


def test_azimuthal_split_keeps_exact_and_inexact_numbers_apart():
    """An exact integer exponent must stay exact when a float of the same value is also present."""
    x, y = _symbol("x"), _symbol("y")

    # -2.0 (a material constant, say) and the exponent -2 meet in one expression.
    got = str(_split(-2.0 * x + (x + y) ** (-2)))
    assert "**(-2.0)" not in got and "^(-2.0)" not in got, \
        "the exact exponent -2 came back inexact: " + got
    # ...and not the other way round either: the float must not be exactified.
    assert "(2.0)" in got, "the inexact coefficient -2.0 came back exact: " + got

    # Same in a product, where GiNaC stores the exponent as a mul coefficient instead.
    got = str(_split((-2.0 * x) * (x + y) ** (-2)))
    assert "**(-2.0)" not in got and "^(-2.0)" not in got, got
    assert "(2.0)" in got, got


def test_real_part_of_an_inexact_whole_number_power_stays_polynomial():
    """power::real_part() must not fall through to atan2 just because the exponent reads 2.0."""
    x, y = _symbol("x"), _symbol("y")
    I = imaginary_i()

    exact = str(_split(subexpression(x ** (-2) + I * y)))
    inexact = str(_split(subexpression(x ** (-2.0) + I * y)))
    assert "atan2" not in inexact, "an inexact exponent produced a polar real part: " + inexact
    assert inexact == exact, "%s\n!=\n%s" % (inexact, exact)
