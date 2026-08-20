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

# pyoomph.equations.electrohydrodynamics: coupling an electric field into the flow.
#
# The one thing that must be right, and that nothing else in the module can rescue:
#
#   THE TWO ROUTES ARE THE SAME PDE AND NOT THE SAME WEAK FORM. For a constant permittivity
#   div(sigma_M) == rho_e*E holds EXACTLY, because E = -grad(phi) is a gradient and the two
#   non-Gauss terms cancel identically. What separates
#
#       weak(sigma_M, grad(v))          [MaxwellStressEquations]
#       -weak(rho_e*E, v)               [ElectricBodyForceEquations]
#
#   is therefore ONLY the surface integral <n.sigma_M, v> on the boundary. Hence:
#
#     * With the velocity pinned on the whole boundary that integral is eliminated, and the two
#       routes must give the same velocity AND the same pressure, to solver tolerance.
#       (Note there is no eps*|E|^2/2 pressure offset here: the offset story belongs to a
#       formulation that keeps only part of the Maxwell stress, not to these two.)
#     * On a boundary that carries a traction -- a do-nothing outlet, and above all a FREE
#       SURFACE -- the integral survives, so the two routes describe genuinely different physics
#       and the body-force route is missing the entire electric traction. Adding
#       MaxwellStressInterface must restore the agreement exactly.
#
#   Both halves are checked below. If either fails, one of the routes has a sign or a factor
#   wrong, and any free-surface result computed with either is meaningless.
#
# The manufactured potential below lies in Q2, so phi_h is the exact solution and
# -div(eps grad phi_h) equals the prescribed rho_e POINTWISE. Without that the two routes would
# only agree up to the finite element residual and the test would have a floor for the wrong reason.
#
# Each Problem gets its OWN output directory, see tests/test_electrostatics.py.

import itertools

import numpy
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.expressions.units import *
from pyoomph.expressions.phys_consts import *
from pyoomph.equations.navier_stokes import StokesEquations
from pyoomph.equations.electrostatics import ElectricPotentialEquations, ElectrodeBC
from pyoomph.equations.electrohydrodynamics import *
from pyoomph.meshes.simplemeshes import RectangularQuadMesh

_run_counter = itertools.count()


def _fresh(problem):
    problem.set_output_directory("run%d" % next(_run_counter))
    problem.set_c_compiler("system")
    return problem


_EPS = 2.5


def _phi_exact():
    x, y = var(["coordinate_x", "coordinate_y"])
    return x * (1 - x) * y * (1 - y) * 30


class _DrivenBox(Problem):
    """Closed unit box, no-slip everywhere, driven by a prescribed charge cloud in a dielectric."""

    def __init__(self, route):
        super().__init__()
        self.route = route

    def define_problem(self):
        phi_e = _phi_exact()
        rho_e = -div(_EPS * grad(phi_e))
        self.add_mesh(RectangularQuadMesh(N=6))
        eqs = StokesEquations(dynamic_viscosity=1).with_pressure_fixation(nondim_p_value=0.0)
        # This test compares the two explicit routes against each other, so the automatic
        # coupling has to be off; test_automatic_coupling_matches_the_explicit_one below is what
        # checks that the automatic one agrees with them.
        eqs += ElectricPotentialEquations(permittivity=_EPS, charge_density=rho_e,
                                          permittivity_scale=1,
                                          add_maxwell_stress_to_momentum=False)
        if self.route == "stress":
            eqs += MaxwellStressEquations()
        elif self.route == "body_force":
            eqs += ElectricBodyForceEquations()
        else:
            raise ValueError(self.route)
        for b in ["left", "right", "top", "bottom"]:
            eqs += (DirichletBC(velocity_x=0, velocity_y=0) + ElectrodeBC(phi_e)) @ b
        E = var("electric_field")
        eqs += IntegralObservables(
            _area=1,
            u2=subexpression(dot(var("velocity"), var("velocity"))),
            ux=var("velocity")[0], uy=var("velocity")[1],
            ux_x=var("velocity")[0] * var("coordinate_x"),
            uy_y=var("velocity")[1] * var("coordinate_y"),
            p=var("pressure"),
            p_shift=var("pressure") + _EPS * dot(E, E) / 2,
            p_shift2=subexpression((var("pressure") + _EPS * dot(E, E) / 2) ** 2))
        self.add_equations(eqs @ "domain")


def _run(route):
    with _fresh(_DrivenBox(route)) as p:
        p.solve()
        o = p.get_mesh("domain").evaluate_all_observables()
        dofs, _ = p.get_current_dofs()
    return {k: float(v) for k, v in o.items()}, numpy.array(dofs)


def test_stress_and_body_force_agree_when_velocity_is_pinned():
    a, dofs_a = _run("stress")
    b, dofs_b = _run("body_force")
    assert a["u2"] > 1e-6, "the test would be vacuous if nothing moved"
    # Neither route adds a dof, so the two dof vectors are element-for-element comparable.
    assert dofs_a.shape == dofs_b.shape
    assert numpy.max(numpy.abs(dofs_a - dofs_b)) < 1e-8 * max(1.0, numpy.max(numpy.abs(dofs_a)))
    for k in ("u2", "ux", "uy", "ux_x", "uy_y", "p"):
        assert a[k] == pytest.approx(b[k], abs=1e-9 + 1e-8 * abs(a[k])), k


def test_body_force_route_needs_the_charge_to_do_anything():
    # A pure dielectric with no free charge and constant permittivity exerts no body force at all,
    # so the flow must be identically zero -- while the stress route still produces a nonzero
    # (curl-free) Maxwell stress that the pressure absorbs.
    class _NoCharge(_DrivenBox):
        def define_problem(self):
            phi_e = _phi_exact()
            self.add_mesh(RectangularQuadMesh(N=4))
            eqs = StokesEquations(dynamic_viscosity=1).with_pressure_fixation(nondim_p_value=0.0)
            eqs += ElectricPotentialEquations(permittivity=_EPS, permittivity_scale=1,
                                              add_maxwell_stress_to_momentum=False)
            eqs += MaxwellStressEquations() if self.route == "stress" else ElectricBodyForceEquations()
            for b in ["left", "right", "top", "bottom"]:
                eqs += (DirichletBC(velocity_x=0, velocity_y=0) + ElectrodeBC(phi_e)) @ b
            eqs += IntegralObservables(_area=1,
                                       u2=subexpression(dot(var("velocity"), var("velocity"))))
            self.add_equations(eqs @ "domain")

    for route in ("stress", "body_force"):
        with _fresh(_NoCharge(route)) as p:
            p.solve()
            o = p.get_mesh("domain").evaluate_all_observables()
        assert float(o["u2"]) == pytest.approx(0.0, abs=1e-18), route


def test_double_counting_is_refused():
    class _Twice(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=3))
            eqs = StokesEquations(dynamic_viscosity=1,
                                  extra_stress=maxwell_stress_tensor(_EPS, -grad(var("phi"))))
            eqs += ElectricPotentialEquations(permittivity=_EPS, permittivity_scale=1,
                                              add_maxwell_stress_to_momentum=False)
            eqs += MaxwellStressEquations()
            for b in ["left", "right", "top", "bottom"]:
                eqs += (DirichletBC(velocity_x=0, velocity_y=0) + ElectrodeBC(0)) @ b
            self.add_equations(eqs @ "domain")

    with pytest.raises(RuntimeError, match="twice"):
        with _fresh(_Twice()) as p:
            p.initialise()


def test_capacitor_plate_traction():
    # The classic result: two plates at a potential difference V attract with eps*V^2/(2h^2) per
    # unit area. Here that is read off the Maxwell stress as n.sigma_M.n on the plate.
    h, V, eps_r = 10 * micro * meter, 3 * volt, 2.5

    class _Cap(Problem):
        def define_problem(self):
            self.set_scaling(spatial=h)
            self.set_scaling(phi=V, permittivity=epsilon_0, electric_field=V / h)
            self.add_mesh(RectangularQuadMesh(N=4, size=h))
            eqs = ElectricPotentialEquations(relative_permittivity=eps_r)
            eqs += ElectrodeBC(0) @ "bottom"
            eqs += ElectrodeBC(V) @ "top"
            n = var("normal")
            sM = maxwell_stress_tensor(eps_r * epsilon_0, var("electric_field"))
            eqs += (IntegralObservables(_len=1, tn=dot(n, matproduct(sM, n)))
                    + IntegralObservableOutput("plate")) @ "top"
            self.add_equations(eqs @ "domain")

    with _fresh(_Cap()) as p:
        p.solve()
        o = p.get_mesh("domain/top").evaluate_all_observables()
    expected = eps_r * epsilon_0 * V ** 2 / (2 * h ** 2)
    # 1e-7 rather than machine precision: the potential is solved for, so this inherits the Newton
    # tolerance rather than being an algebraic identity.
    assert float((o["tn"] / o["_len"]) / pascal) == pytest.approx(float(expected / pascal), rel=1e-7)


def test_maxwell_stress_interface_restores_the_traction():
    """The half of the equivalence that a free surface actually depends on.

    On a do-nothing boundary the surface integral <n.sigma_M, v> is NOT eliminated, so the stress
    route and the body-force route describe different physics there. They must therefore disagree --
    and adding MaxwellStressInterface to the body-force route must make them agree again, exactly.

    This is the direct analogue of what happens on a free surface, where getting it wrong is silent:
    the simulation converges and the interface shape is simply wrong.
    """

    class _OpenBox(Problem):
        def __init__(self, route, with_interface=False):
            super().__init__()
            self.route, self.with_interface = route, with_interface

        def define_problem(self):
            phi_e = _phi_exact()
            rho_e = -div(_EPS * grad(phi_e))
            self.add_mesh(RectangularQuadMesh(N=6))
            eqs = StokesEquations(dynamic_viscosity=1)
            eqs += ElectricPotentialEquations(permittivity=_EPS, charge_density=rho_e,
                                              permittivity_scale=1,
                                              add_maxwell_stress_to_momentum=False)
            eqs += MaxwellStressEquations() if self.route == "stress" else ElectricBodyForceEquations()
            for b in ["left", "right", "bottom"]:
                eqs += DirichletBC(velocity_x=0, velocity_y=0) @ b
            eqs += ElectrodeBC(phi_e) @ "top"          # "top" is do-nothing for the flow
            for b in ["left", "right", "bottom"]:
                eqs += ElectrodeBC(phi_e) @ b
            if self.with_interface:
                # The parent's own Maxwell stress is NOT in its momentum row (body-force route), so
                # it is this equation that has to put the traction on the boundary.
                eqs += MaxwellStressInterface(mode="parent_only") @ "top"
            eqs += IntegralObservables(_area=1,
                                       u2=subexpression(dot(var("velocity"), var("velocity"))),
                                       uy_top=var("velocity")[1])
            self.add_equations(eqs @ "domain")

    def run(route, with_interface=False):
        with _fresh(_OpenBox(route, with_interface)) as p:
            p.solve()
            o = p.get_mesh("domain").evaluate_all_observables()
            dofs, _ = p.get_current_dofs()
        return {k: float(v) for k, v in o.items()}, numpy.array(dofs)

    stress, d_stress = run("stress")
    plain, d_plain = run("body_force")
    fixed, d_fixed = run("body_force", with_interface=True)

    assert stress["u2"] > 1e-6, "the test would be vacuous if nothing moved"
    # Without the interface term the two routes are genuinely different physics on this boundary.
    rel_gap = numpy.max(numpy.abs(d_stress - d_plain)) / numpy.max(numpy.abs(d_stress))
    assert rel_gap > 1e-3, "expected a visible difference on a traction boundary, got %g" % rel_gap
    # With it, they agree to solver tolerance.
    # 1e-7, not machine precision: both sides inherit the Newton tolerance. The gap it has to
    # close is five orders larger, so the statement is not weakened by the loose threshold.
    rel_fixed = numpy.max(numpy.abs(d_stress - d_fixed)) / numpy.max(numpy.abs(d_stress))
    assert rel_fixed < 1e-7, "MaxwellStressInterface did not restore the traction (%g)" % rel_fixed


def test_automatic_coupling_matches_the_explicit_one():
    """ElectricPotentialEquations applies the Maxwell stress to a co-located flow by default.

    That default is what makes an EHD problem a two-line affair, so it has to be exactly the stress
    route and not an approximation of it. Compared dof by dof against the explicit
    MaxwellStressEquations, which the tests above have already tied to the body-force route.
    """

    class _Auto(_DrivenBox):
        def define_problem(self):
            phi_e = _phi_exact()
            rho_e = -div(_EPS * grad(phi_e))
            self.add_mesh(RectangularQuadMesh(N=6))
            eqs = StokesEquations(dynamic_viscosity=1).with_pressure_fixation(nondim_p_value=0.0)
            # No MaxwellStressEquations, no bulkforce: just the field equations next to the flow.
            eqs += ElectricPotentialEquations(permittivity=_EPS, charge_density=rho_e,
                                              permittivity_scale=1)
            for b in ["left", "right", "top", "bottom"]:
                eqs += (DirichletBC(velocity_x=0, velocity_y=0) + ElectrodeBC(phi_e)) @ b
            eqs += IntegralObservables(_area=1,
                                       u2=subexpression(dot(var("velocity"), var("velocity"))))
            self.add_equations(eqs @ "domain")

    explicit, d_explicit = _run("stress")
    with _fresh(_Auto("stress")) as p:
        p.solve()
        auto_u2 = float(p.get_mesh("domain").evaluate_all_observables()["u2"])
        d_auto = numpy.array(p.get_current_dofs()[0])

    assert auto_u2 > 1e-6, "the test would be vacuous if nothing moved"
    assert d_auto.shape == d_explicit.shape
    assert numpy.max(numpy.abs(d_auto - d_explicit)) < 1e-10 * numpy.max(numpy.abs(d_explicit))


def test_automatic_coupling_is_silent_without_a_flow():
    # A pure electrostatics problem must be untouched by the default: no flow on the domain means
    # nothing to couple to, and asking for a velocity test function there would be an error.
    class _NoFlow(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=3))
            eqs = ElectricPotentialEquations(permittivity=_EPS, permittivity_scale=1)
            for b in ["left", "right", "top", "bottom"]:
                eqs += ElectrodeBC(_phi_exact()) @ b
            self.add_equations(eqs @ "domain")

    with _fresh(_NoFlow()) as p:
        p.solve()


@pytest.mark.parametrize("explicit", ["stress", "body_force"])
def test_automatic_and_explicit_together_are_refused(explicit):
    class _Both(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=3))
            eqs = StokesEquations(dynamic_viscosity=1).with_pressure_fixation(nondim_p_value=0.0)
            eqs += ElectricPotentialEquations(permittivity=_EPS, permittivity_scale=1)  # default: on
            eqs += MaxwellStressEquations() if explicit == "stress" else ElectricBodyForceEquations()
            for b in ["left", "right", "top", "bottom"]:
                eqs += (DirichletBC(velocity_x=0, velocity_y=0) + ElectrodeBC(0)) @ b
            self.add_equations(eqs @ "domain")

    with pytest.raises(RuntimeError, match="twice"):
        with _fresh(_Both()) as p:
            p.initialise()


def test_maxwell_stress_assembles_in_axisymmetry():
    """A smoke test, and labelled as one: it establishes that the coordinate system survives, not
    that the answer is right.

    ``maxwell_stress_tensor`` builds ``dyadic(E,E)`` and ``identity_matrix()``, both of which go
    through the coordinate system, and in axisymmetry the tensor is 3x3 over a 2D mesh. There is no
    analytic reference behind this. The test that would validate the axisymmetric electric traction
    properly is Taylor's leaky-dielectric drop deformation; see dev_docs/electrohydrodynamics.md
    section 10.3.
    """

    class _Axi(Problem):
        def define_problem(self):
            self.set_coordinate_system(axisymmetric)
            r, z = var(["coordinate_x", "coordinate_y"])
            phi_e = 10 * r * (1 - r) * z * (1 - z)
            self.add_mesh(RectangularQuadMesh(N=4))
            eqs = StokesEquations(dynamic_viscosity=1).with_pressure_fixation(nondim_p_value=0.0)
            eqs += ElectricPotentialEquations(permittivity=_EPS, permittivity_scale=1,
                                              charge_density=-div(_EPS * grad(phi_e)))
            for b in ["left", "right", "top", "bottom"]:
                eqs += (DirichletBC(velocity_x=0, velocity_y=0) + ElectrodeBC(phi_e)) @ b
            eqs += IntegralObservables(
                u2=subexpression(dot(var("velocity"), var("velocity"))),
                sM_rr=maxwell_stress_tensor(_EPS, var("electric_field"))[0, 0])
            self.add_equations(eqs @ "domain")

    with _fresh(_Axi()) as p:
        p.solve()
        o = p.get_mesh("domain").evaluate_all_observables()
    assert float(o["u2"]) > 1e-8, "the electric force should drive a flow"
    assert numpy.isfinite(float(o["sM_rr"]))
