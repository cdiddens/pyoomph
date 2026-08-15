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

# Optional residual-based stabilization of the scalar transport equations, i.e. the "stabilization"
# argument of AdvectionDiffusionEquations, CompositionAdvectionDiffusionEquations and the temperature
# equations.
#
# The three things that actually have to hold, and that the tests below check directly:
#
#   1. CONSISTENCY. SUPG/GLS/ASGS are proportional to the strong residual, so on a solution the space
#      represents exactly they must return the unstabilized answer. The manufactured solution here is
#      chosen to lie in the space AND to satisfy the PDE strongly, so R == 0 pointwise. The
#      measurement floor is ~1e-8 for a squared-error observable, as the sibling flow module records.
#   2. THE DEFAULT IS OFF. With stab_factor=0 the residual must be *bitwise* identical, not merely
#      close: that is what catches a term that is added unconditionally or a prefactor that does not
#      reach every contribution. The removed pyoomph.equations.SUPG carried exactly such a bug in a
#      dead line for years.
#   3. NOTHING LEAKS INTO THE INTERFACE PHYSICS. Splitting the residual by dof type shows which rows
#      a switch actually touches. Marangoni, the kinematic BC, mass transfer, latent heat and the
#      velocity connection must not move when only the composition or temperature is stabilized.
#
# Each Problem gets its OWN output directory. Several Problems in one directory share the JIT cache,
# and variants that differ only in constructor flags then silently reuse the first one's compiled
# code -- which makes every stabilization variant look identical to the unstabilized one.

import itertools

import numpy
import pytest
from typing import Any, cast

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.expressions import FiniteElementSpaceEnum
from pyoomph.expressions.units import *
from pyoomph.equations.advection_diffusion import AdvectionDiffusionEquations
from pyoomph.equations.stabilization import ScalarTransportStabilization
from pyoomph.meshes.simplemeshes import LineMesh, RectangularQuadMesh

_run_counter = itertools.count()


def _fresh(problem):
    """Own output directory per Problem, see the note at the top."""
    problem.set_output_directory("run%d" % next(_run_counter))
    problem.set_c_compiler("system")
    return problem


# =================================================================================================
#  Plain advection-diffusion
# =================================================================================================

_D = 0.05
_WIND = vector(1, 0.3)
# Representable in the respective space on a rectangular (affine) quad mesh: Q1 has x, y and x*y,
# Q2 additionally x^2, y^2 and x^2*y^2.
_EXACT = {"C1": 1 + var("coordinate_x") + 2 * var("coordinate_y") + 3 * var("coordinate_x") * var("coordinate_y"),
          "C2": 1 + var("coordinate_x") + 2 * var("coordinate_y") + 3 * var("coordinate_x") * var("coordinate_y")
                + var("coordinate_x") ** 2 + var("coordinate_y") ** 2
                + var("coordinate_x") ** 2 * var("coordinate_y") ** 2}


class _ManufacturedAdvDiff(Problem):
    """Steady advection-diffusion whose exact solution lies in the finite element space."""

    def __init__(self, stabilization=None, space="C1", advection_by_parts=False, N=(8, 4)):
        super().__init__()
        self.stabilization = stabilization
        self.space = space
        self.advection_by_parts = advection_by_parts
        self.N = list(N)

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=self.N, size=[2, 1]))
        ce = _EXACT[self.space]
        # The source is built from the same symbolic expression, so ce solves the PDE strongly
        eqs = AdvectionDiffusionEquations("c", diffusivity=_D, space=cast(FiniteElementSpaceEnum, self.space), wind=_WIND,
                                          consider_scaling=False,
                                          source=dot(_WIND, grad(ce)) - _D * div(grad(ce)),
                                          advection_by_parts=self.advection_by_parts,
                                          stabilization=self.stabilization)
        adeq = eqs
        for b in ["left", "right", "top", "bottom"]:
            eqs = eqs + DirichletBC(c=ce) @ b
        eqs = eqs + IntegralObservables(err=(var("c") - ce) ** 2, V=1,
                                        R2=adeq.strong_residual("c") ** 2,
                                        A2=dot(_WIND, grad(var("c"))) ** 2)
        self.add_equations(eqs @ "domain")


def _solve_advdiff(**kwargs):
    with _fresh(_ManufacturedAdvDiff(**kwargs)) as p:
        p.solve()
        o = p.get_mesh("domain").evaluate_all_observables()
        return numpy.array(p.get_current_dofs()[0]), o


@pytest.mark.parametrize("space", ["C1", "C2"])
@pytest.mark.parametrize("advection_by_parts", [False, True, "skew"])
@pytest.mark.parametrize("stabilization", ["SUPG", "GLS", "ASGS"])
def test_consistency_on_a_representable_solution(space, advection_by_parts, stabilization):
    """Every consistent variant must return the unstabilized solution when the strong residual is
    zero. This is the whole point of a residual-based stabilization: it changes the conditioning,
    not the answer."""
    ref, oref = _solve_advdiff(space=space, advection_by_parts=advection_by_parts)
    got, o = _solve_advdiff(space=space, advection_by_parts=advection_by_parts,
                            stabilization=stabilization)
    # The *reference* solution really does have a zero strong residual, otherwise the check below
    # proves nothing. Deliberately not checked on the stabilized solution too: its R is a consequence
    # of the amplification measured below, so requiring it to be small would be circular.
    assert (float(oref["R2"]) / float(oref["A2"])) ** 0.5 < 1e-7
    # Not roundoff: R is zero only to the accuracy at which it can be *evaluated*, and on C2 it
    # contains second derivatives, whose cancellation bottoms out near 1e-7 relative. Measured
    # relative dof differences on C2: SUPG and GLS <= 4e-9, ASGS up to 5e-7 -- the adjoint sign makes
    # ASGS anti-dissipative in the diffusive part, so it amplifies that floor by ~100x. That is the
    # same fragility the flow module records for ASGS. An inconsistent stabilization would be off by
    # ~1e-2 here, so even the loose bound is a real test.
    tol = 1e-5 if stabilization == "ASGS" else 1e-7
    scale = numpy.max(numpy.abs(ref))
    assert numpy.max(numpy.abs(got - ref)) < tol * scale, \
        "%s on %s (advection_by_parts=%s) moved a solution it must reproduce" % (
            stabilization, space, advection_by_parts)
    # ... and it must not have made the solution worse either. "err" is the *squared* L2 error, so
    # the floor is the square of the bound above; a squared error at roundoff is as likely to
    # integrate to a small negative number as to zero, hence the abs().
    assert float(o["err"]).real <= max(4 * abs(float(oref["err"]).real), (tol * scale) ** 2)


@pytest.mark.parametrize("terms", ["SUPG", "GLS", "ASGS", "SUPG+DC"])
def test_stab_factor_zero_is_bitwise_identical(terms):
    """stab_factor=0 (and dc_factor=0) must switch every single contribution off. Bitwise, because a
    term that is added unconditionally would still show up as a tiny difference."""
    ref, _ = _solve_advdiff(space="C2")
    cfg = ScalarTransportStabilization(terms, stab_factor=0, dc_factor=0)
    got, _ = _solve_advdiff(space="C2", stabilization=cfg)
    assert numpy.array_equal(got, ref)


def test_supg_suppresses_oscillations_at_high_peclet():
    """The reason the feature exists. A boundary layer thinner than one element makes the Galerkin
    solution oscillate; SUPG must not."""

    class P(Problem):
        def __init__(self, stabilization):
            super().__init__()
            self.stabilization = stabilization

        def define_problem(self):
            self.add_mesh(LineMesh(N=20, size=1))
            eqs = AdvectionDiffusionEquations("c", diffusivity=1e-3, space="C1",
                                              wind=vector(1), consider_scaling=False,
                                              stabilization=self.stabilization)
            eqs += DirichletBC(c=0) @ "left"
            eqs += DirichletBC(c=1) @ "right"
            self.add_equations(eqs @ "domain")

    def overshoot(stabilization):
        with _fresh(P(stabilization)) as p:
            p.solve()
            c = numpy.array(p.get_current_dofs()[0])
            # The exact solution is monotone in [0,1]; anything outside is a numerical oscillation
            return max(numpy.max(c) - 1.0, -numpy.min(c), 0.0)

    galerkin = overshoot(None)
    supg = overshoot("SUPG")
    assert galerkin > 0.1, "the unstabilized reference does not oscillate, so the test proves nothing"
    assert supg < 1e-3, "SUPG left an overshoot of %g" % supg


@pytest.mark.parametrize("advection_by_parts,expect_conservative", [(False, False), (True, True)])
def test_conservative_and_convective_residuals_differ_by_the_divergence_of_the_wind(
        advection_by_parts, expect_conservative):
    """The strong residual must mirror what is *assembled*. With a non-solenoidal wind the
    conservative form div(a c) and the convective form a.grad(c) differ by exactly c*div(a), and
    'auto' has to pick the one that matches advection_by_parts.

    Both forms are evaluated on the *same* solution, since two solves would compare two states."""
    x = var("coordinate_x")
    wind = vector(x, 0)          # div(wind) = 1

    class P(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=[4, 2], size=[1, 1]))
            eqs = AdvectionDiffusionEquations("c", diffusivity=_D, space="C1", wind=wind,
                                              consider_scaling=False,
                                              advection_by_parts=advection_by_parts,
                                              stabilization="SUPG")
            eqs.stab_cfg.conservative_residual = False
            convective = eqs.strong_residual("c")
            eqs.stab_cfg.conservative_residual = True
            conservative = eqs.strong_residual("c")
            eqs.stab_cfg.conservative_residual = "auto"
            auto = eqs.strong_residual("c")
            out = eqs
            for b in ["left", "right", "top", "bottom"]:
                out = out + DirichletBC(c=1 + var("coordinate_x") + var("coordinate_y")) @ b
            out = out + IntegralObservables(diff=conservative - convective, c=var("c"),
                                            auto_minus_expected=auto - (conservative if expect_conservative
                                                                        else convective))
            self.add_equations(out @ "domain")

    with _fresh(P()) as p:
        p.solve()
        o = p.get_mesh("domain").evaluate_all_observables()
    # int(div(a c) - a.grad c) = int(c div(a)) = int(c), since div(a) = 1 here
    assert abs(float(o["diff"]) - float(o["c"])) < 1e-10 * abs(float(o["c"]))
    assert abs(float(o["auto_minus_expected"])) < 1e-10 * abs(float(o["c"]))


def test_stabilization_in_dimensional_units():
    """tau mixes 1/dt, |a|/h and D/h^2, and the regularizations add constants under square roots.
    Getting a scale factor wrong there is invisible in a nondimensional problem and rejected outright
    in a dimensional one, so this compiles the same problem in mm/ms."""

    class P(Problem):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg

        def define_problem(self):
            self.set_scaling(spatial=1 * milli * meter, temporal=1 * milli * second,
                             velocity=1 * milli * meter / second, c=1 * mol / meter ** 3)
            self.add_mesh(RectangularQuadMesh(N=[4, 2], size=[2 * milli * meter, 1 * milli * meter]))
            eqs = AdvectionDiffusionEquations("c", diffusivity=1e-9 * meter ** 2 / second, space="C1",
                                              wind=vector(1, 0) * milli * meter / second,
                                              velocity_name_for_scaling="velocity",
                                              stabilization=self.cfg)
            eqs += DirichletBC(c=0 * mol / meter ** 3) @ "left"
            eqs += DirichletBC(c=1 * mol / meter ** 3) @ "right"
            self.add_equations(eqs @ "domain")

    for cfg in [ScalarTransportStabilization("SUPG"),
                ScalarTransportStabilization("GLS", tau_formula="codina"),
                ScalarTransportStabilization("SUPG+DC", dc_form="crosswind"),
                ScalarTransportStabilization("SUPG+DC", dc_form="isotropic")]:
        with _fresh(P(cfg)) as p:
            p.initialise()   # compiling is the test; a unit error raises here


@pytest.mark.parametrize("natural_bc_correction,expect_zero", [(False, True), (True, False)])
def test_stabilization_flux_hook_is_zero_by_default(natural_bc_correction, expect_zero):
    """The boundary footprint is opt-in. Off, get_stabilization_flux must be exactly zero, so that
    every flux boundary condition keeps imposing what it imposed before."""
    seen = []

    class _Probe(AdvectionDiffusionEquations):
        def define_residuals(self):
            super().define_residuals()
            # inside the code generator's scope, which the hook needs (it asks the mesh whether the
            # coordinates are degrees of freedom)
            seen.append(is_zero(self.get_stabilization_flux("c", var("normal"))))

    class P(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=[2, 2]))
            cfg = ScalarTransportStabilization("SUPG+DC", natural_bc_correction=natural_bc_correction)
            eqs = _Probe("c", diffusivity=_D, space="C1", wind=_WIND,
                         consider_scaling=False, stabilization=cfg)
            eqs = eqs + DirichletBC(c=0) @ "left"
            self.add_equations(eqs @ "domain")

    with _fresh(P()) as p:
        p.initialise()
    assert seen and all(s == expect_zero for s in seen)


# =================================================================================================
#  Multi-component flow: the interface physics must not notice
# =================================================================================================

def _rows_that_moved(reference, candidate, rtol=1e-11):
    r0, (idx0, names0) = reference
    r1, (idx1, names1) = candidate
    assert list(names0) == list(names1) and numpy.array_equal(idx0, idx1), "dof layout changed"
    moved = []
    for i, name in enumerate(names0):
        sel = idx0 == i
        if not numpy.any(sel):
            continue
        scale = max(numpy.max(numpy.abs(r0[sel])), 1e-300)
        if numpy.max(numpy.abs(r1[sel] - r0[sel])) > rtol * scale:
            moved.append(name)
    return set(moved)


def _evaporating_droplet_residual(**stabilization_kwargs):
    from pyoomph.equations.multi_component import (CompositionFlowEquations,
                                                   CompositionDiffusionEquations,
                                                   MultiComponentNavierStokesInterface)
    from pyoomph.materials import Mixture, get_pure_liquid, get_pure_gas
    import pyoomph.materials.default_materials  # noqa: F401  (registers the library)

    class P(Problem):
        def define_problem(self):
            L, T = 1 * milli * meter, 20 * celsius
            liq = Mixture(get_pure_liquid("water") + 0.5 * get_pure_liquid("ethanol"))
            gas = Mixture(get_pure_gas("air") + 20 * percent * get_pure_gas("ethanol")
                          + 40 * percent * get_pure_gas("water"),
                          quantity="relative_humidity", temperature=T)
            self.set_scaling(spatial=L, temporal=1 * second)
            liq.set_reference_scaling_to_problem(self, temperature=T)
            self.set_scaling(pressure=10 * pascal, velocity=1e-4 * meter / second)
            self.define_named_var(temperature=T, absolute_pressure=1 * atm)
            self.add_mesh(RectangularQuadMesh(size=[2 * L, 0.5 * L], N=[6, 2],
                                              name=lambda x, y: "gas" if x > 1 else "liquid"))
            x, y = var(["coordinate_x", "coordinate_y"])
            leqs = CompositionFlowEquations(liq, compo_space="C2", isothermal=False,
                                            initial_temperature=T,
                                            gravity=vector(0, -9.81) * meter / second ** 2,
                                            **stabilization_kwargs)
            # A non-uniform state, so that the strong residual is genuinely nonzero
            leqs += InitialCondition(massfrac_ethanol=0.5 + 0.15 * x / L,
                                     temperature=T + 3 * kelvin * y / L,
                                     velocity_x=1e-3 * meter / second * y / L)
            leqs += DirichletBC(velocity_y=0) @ "bottom"
            leqs += DirichletBC(velocity_y=0) @ "top"
            leqs += MultiComponentNavierStokesInterface(cast(Any, liq | gas)) @ "gas_liquid"
            geqs = CompositionDiffusionEquations(gas)
            geqs += DirichletBC(**{"massfrac_" + c: True for c in gas.required_adv_diff_fields}) @ "right"
            self.add_equations(leqs @ "liquid" + geqs @ "gas")

    with _fresh(P()) as p:
        p.initialise()
        p.set_initial_condition()
        # The residual at a *fixed* dof vector: solving first would compare two different states
        return numpy.array(p.get_residuals()), p.get_dof_description()


@pytest.mark.parametrize("switch,expected_rows", [
    ("compo_stabilization", {"massfrac_"}),
    ("thermal_stabilization", {"temperature"}),
    ("ns_stabilization", {"velocity", "pressure"}),
])
def test_stabilization_touches_only_its_own_rows(switch, expected_rows):
    """The requirement this feature has to meet: switching a stabilization on must not perturb the
    Marangoni stress, the kinematic boundary condition, mass transfer, latent heat, the velocity
    connection or the opposite phase. Those all test against velocity / mesh / interface fields,
    which no stabilization term is ever written against."""
    setting = "SUPG+PSPG" if switch == "ns_stabilization" else "SUPG"
    reference = _evaporating_droplet_residual()
    candidate = _evaporating_droplet_residual(**{switch: setting})
    moved = _rows_that_moved(reference, candidate)
    assert moved, "the switch did nothing at all, so the test proves nothing"
    unexpected = [m for m in moved if not any(tag in m for tag in expected_rows)]
    assert not unexpected, "%s=%s also moved %s" % (switch, setting, sorted(unexpected))


def _bilinear_distort(mesh):
    """A globally bilinear node map: elements stay quads but stop being rectangles."""
    for n in mesh.nodes():
        a, b = n.x(0), n.x(1)
        n.set_x(0, a + 0.30 * a * b)
        n.set_x(1, b - 0.20 * a * b)


@pytest.mark.parametrize("split,space,distort,exact", [
    ("alternate_left", "C1", False, True),   # affine simplex: second derivatives vanish identically
    ("alternate_left", "C2", False, False),  # quadratic on a simplex: they do not
    (False, "C1", False, True),              # Q1 on a rectangle: d_xx = d_yy = 0, so the Laplacian is too
    (False, "C1", True, False),              # ... but not once the quad stops being a rectangle
])
def test_dropping_the_scalar_second_derivative(split, space, distort, exact):
    """``include_diffusion_in_residual=False`` is advertised as free on linear simplices and as an
    approximation otherwise. Both halves are load-bearing, and the quad rows are why the advertised
    rule is about *simplices* and not merely about C1: an undistorted Q1 quad happens to qualify,
    but any distortion -- i.e. every moving mesh -- ends that."""
    class P(Problem):
        def __init__(self, keep):
            super().__init__()
            self.keep = keep

        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=8, size=[1, 1], split_in_tris=split))
            eqs = AdvectionDiffusionEquations(
                "c", diffusivity=0.01, space=cast(FiniteElementSpaceEnum, space),
                consider_scaling=False, wind=_WIND,
                stabilization=ScalarTransportStabilization(
                    "SUPG", include_diffusion_in_residual=self.keep))
            for b in ["left", "right", "top", "bottom"]:
                eqs = eqs + DirichletBC(c=var("coordinate_x") ** 2 + var("coordinate_y")) @ b
            self.add_equations(eqs @ "domain")

    res = []
    for keep in (True, False):
        with _fresh(P(keep)) as p:
            p.initialise()
            if distort:
                _bilinear_distort(p.get_mesh("domain"))
            p.set_initial_condition()
            res.append(numpy.array(p.get_residuals()))
    rel = numpy.max(numpy.abs(res[0] - res[1])) / max(numpy.max(numpy.abs(res[0])), 1e-300)
    if exact:
        assert rel < 1e-14, "dropping the term is supposed to be free here, but moved by %g" % rel
    else:
        assert rel > 1e-3, "dropping the term is supposed to matter here, but moved only %g" % rel


@pytest.mark.parametrize("split,viscous_form,exact", [
    ("alternate_left", "stress", True),   # affine simplex: free whatever the form
    (False, "laplace", True),             # Q1 rectangle, only div(grad u): free
    (False, "stress", False),             # ... but the stress form adds grad(div u), whose MIXED
])                                        # derivative a bilinear map does not kill
def test_dropping_the_viscous_second_derivative(split, viscous_form, exact):
    """The momentum counterpart, and the trap in it: ``include_viscous_in_residual=False`` is free on
    a C1 quad mesh only in the Laplace form. In the default stress form the same mesh gives a ~40%
    different stabilization, because grad(div(u)) survives a bilinear map."""
    from pyoomph.equations.stabilized_ns import StabilizedNavierStokes

    class P(Problem):
        def __init__(self, keep):
            super().__init__()
            self.keep = keep

        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=8, size=[1, 1], split_in_tris=split))
            eqs = StabilizedNavierStokes(
                space="C1C1", stabilization="SUPG+PSPG",
                viscous_form=cast(Any, viscous_form),
                dynamic_viscosity=0.01, mass_density=1,
                include_viscous_in_residual=self.keep)
            eqs += DirichletBC(velocity_x=0, velocity_y=0) @ "bottom"
            eqs += DirichletBC(velocity_x=1, velocity_y=0) @ "top"
            eqs += DirichletBC(velocity_x=0, velocity_y=0) @ "left"
            eqs += DirichletBC(velocity_x=0, velocity_y=0) @ "right"
            eqs += DirichletBC(pressure=0) @ "bottom/left"
            self.add_equations(eqs @ "domain")

    res = []
    for keep in (True, False):
        with _fresh(P(keep)) as p:
            p.initialise()
            p.set_initial_condition()
            res.append(numpy.array(p.get_residuals()))
    rel = numpy.max(numpy.abs(res[0] - res[1])) / max(numpy.max(numpy.abs(res[0])), 1e-300)
    if exact:
        assert rel < 1e-14, "dropping the term is supposed to be free here, but moved by %g" % rel
    else:
        assert rel > 1e-3, "dropping the term is supposed to matter here, but moved only %g" % rel


def test_div_of_a_gradient_combines_with_numbers():
    """div() used to be registered without a return type, so GiNaC inferred one from its argument and
    a *held* div(grad(c)) inherited grad's non-commutativity. Adding a number to it then raised
    "sum of non-commutative objects has non-zero numeric term", which made the scalar Laplacian
    unusable inside any nonlinear expression -- exactly what a sqrt(R^2+eps^2) needs."""
    built = []

    class _Probe(Equations):
        def define_fields(self):
            self.define_scalar_field("c", "C2")
            self.define_vector_field("u", "C2")

        def define_residuals(self):
            c, v = var_and_test("c")
            u = var("u")
            for expr in [lambda: div(grad(c)) + 1,
                         lambda: div(grad(c)) ** 2 + 1,
                         lambda: square_root(div(grad(c)) ** 2 + 1e-20),
                         lambda: div((1 + c ** 2) * grad(c)) + 1,
                         lambda: div(u) + 1]:
                built.append(expr())
            # the tensor divergence still returns a vector and must stay usable as one
            self.add_residual(weak(div(grad(u)), testfunction("u")))
            self.add_residual(weak(grad(c), grad(v)))

    class P(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=[2, 2]))
            self.add_equations(_Probe() @ "domain")

    with _fresh(P()) as p:
        p.initialise()
    assert len(built) == 5


def test_gcl_composition_honours_dt_factor():
    """The GCL branch assembles the derivative of the whole integral and used to do so with a plain
    add_dweak_dt(), i.e. without dt_factor, while both sibling branches applied it. compo_dt_factor
    was therefore silently ignored under GCL=True alone."""
    from pyoomph.equations.multi_component import CompositionAdvectionDiffusionEquations
    from pyoomph.materials import Mixture, get_pure_liquid
    import pyoomph.materials.default_materials  # noqa: F401

    liq = Mixture(get_pure_liquid("water") + 0.5 * get_pure_liquid("ethanol"))
    eqs = CompositionAdvectionDiffusionEquations(liq, GCL=True, integrate_advection_by_parts=True,
                                                 dt_factor=2)
    # The strong residual is the cheapest place to see it: it mirrors what is assembled, so if
    # dt_factor reaches the residual it reaches the Galerkin term too.
    fn = eqs.fieldnames[0]
    eqs.dt_factor = 1
    one = eqs.strong_residual(fn)
    eqs.dt_factor = 2
    two = eqs.strong_residual(fn)
    assert not is_zero(two - one), "dt_factor does not reach the GCL strong residual"


@pytest.mark.parametrize("terms", ["SUPG", "GLS", "SUPG+DC"])
def test_stabilization_in_a_flow_free_domain(terms):
    """A purely diffusive domain need not define a velocity scale at all. SUPG and DC have nothing to
    act on there and must switch themselves off; GLS/ASGS still apply, and must not reach for
    scale_factor("velocity") on the way -- which fails at code generation, not at construction."""
    from pyoomph.equations.multi_component import CompositionDiffusionEquations
    from pyoomph.materials import Mixture, get_pure_gas
    import pyoomph.materials.default_materials  # noqa: F401

    class P(Problem):
        def define_problem(self):
            T = 20 * celsius
            gas = Mixture(get_pure_gas("air") + 1 * percent * get_pure_gas("water"),
                          quantity="relative_humidity", temperature=T)
            self.set_scaling(spatial=1 * milli * meter, temporal=1 * second,
                             mass_density=1 * kilogram / meter ** 3, temperature=1 * kelvin,
                             thermal_conductivity=0.026 * watt / (meter * kelvin),
                             rho_cp=1000 * joule / (meter ** 3 * kelvin))
            self.define_named_var(temperature=T, absolute_pressure=1 * atm)
            self.add_mesh(RectangularQuadMesh(N=[3, 3], size=[1 * milli * meter, 1 * milli * meter]))
            eqs = CompositionDiffusionEquations(gas, isothermal=False, initial_temperature=T,
                                                compo_stabilization=terms, thermal_stabilization=terms)
            eqs += DirichletBC(massfrac_water=0.01) @ "left"
            self.add_equations(eqs @ "domain")

    with _fresh(P()) as p:
        p.initialise()


def test_composition_flow_stabilization_defaults_to_the_plain_flow_equations():
    """The default must be the plain class, not StabilizedNavierStokes(stabilization='none'):
    equivalent residuals are not enough, the generated code has to stay the same too."""
    from pyoomph.equations.multi_component import CompositionFlowEquations
    from pyoomph.equations.navier_stokes import NavierStokesEquations
    from pyoomph.equations.stabilized_ns import StabilizedNavierStokes
    from pyoomph.materials import Mixture, get_pure_liquid
    import pyoomph.materials.default_materials  # noqa: F401

    liq = Mixture(get_pure_liquid("water") + 0.5 * get_pure_liquid("ethanol"))
    plain = CompositionFlowEquations(liq).get_equation_of_type(NavierStokesEquations)
    assert type(plain) is NavierStokesEquations
    stabilized = CompositionFlowEquations(liq, ns_stabilization="PSPG", ns_mode="C1C1")
    assert isinstance(stabilized.get_equation_of_type(NavierStokesEquations), StabilizedNavierStokes)
    # ... and an equal-order pair without PSPG still builds -- mode="C1" was reachable before this
    # work and stays reachable -- but says out loud that it is inf-sup unstable
    equal_order = CompositionFlowEquations(liq, ns_mode="C1C1")
    assert type(equal_order.get_equation_of_type(NavierStokesEquations)) is NavierStokesEquations
