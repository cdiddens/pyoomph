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
from pyoomph.equations.electrostatics import (ElectricPotentialEquations, ElectrodeBC,
                                             set_electrostatic_scaling)
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


# =================================================================================================
# ElectroosmoticSlip -- the thin-double-layer route.
#
# The Debye layer is 1-100 nm against a millimetric device, so resolving it with Poisson-Nernst-
# Planck is a 1D validation exercise, not a way to compute a real channel. The production route is
# to collapse the layer into a boundary condition,
#
#     u_t = -eps*zeta*E_t/mu     (Helmholtz-Smoluchowski),
#
# and solve an electroneutral bulk. Everything below rests on the one exact solution that route
# has: a straight channel with open ends, a uniform axial field and a uniform zeta develops a
# PLUG FLOW. Not "approximately uniform" -- the constant velocity, p = 0 and a vanishing multiplier
# satisfy the discrete residual exactly, so the nodal spread is a solver-tolerance quantity and any
# real defect (a wrong sign, a missing permittivity, a tangential projection that is not one) shows
# up either as a shifted plateau or as a profile that is no longer flat.
#
# Two things this pins down that nothing else in the module does:
#
#   * THE SIGN. With a negative zeta -- glass, silica, most oxides in water -- and a field pointing
#     in +x, the counter-ion cloud is positive and is dragged with the field, so the liquid moves
#     in +x. u = -eps*zeta*E/mu is then positive. Getting this backwards reverses every
#     electroosmotic pump ever written with this class, and no other test would notice.
#   * THAT IT IS THE *TANGENTIAL* FIELD. test_slip_ignores_the_normal_field imposes an oblique
#     uniform field, whose normal component at the wall is large and must not appear in the slip.
#
# WHY THE QUAD TOLERANCES ARE 1e-6 AND THE TRIANGLE ONES 1e-11. They are measuring the same thing;
# the difference is not the physics. The 3x3 Gauss rule used by 2D QUADRILATERAL elements has a typo
# in its knot table -- src/thirdparty/oomph-lib/include/integral.cc, Gauss<2,3>::Knot, where five
# entries read 0.774596662941483 instead of 0.774596669241483. The rule is therefore asymmetric, and
# the integral of a mid-side shape function derivative over an element comes out as 7e-9 instead of
# 0. On a C2 field that is a fixed ~1e-9 defect, MESH-INDEPENDENT (it is a defect of the rule, not a
# discretisation error), and it is why the plug is flat only to about 1e-8 on a quad mesh. Split the
# same mesh into triangles -- a different quadrature -- and the very same problem is flat to 1e-15.
# So each geometry is asserted at its own floor, and the triangle case is the one that says the plug
# flow is EXACT. See also the 1D Gauss<1,3> table, which is correct, hence the 1D tests elsewhere in
# this suite reaching 1e-11.


class _EOFChannel(Problem):
    """Straight 2D channel, uniform field, uniform zeta, both ends open (do-nothing).

    The exact solution is the plug ``u = (-eps*zeta*E_x/mu, 0)`` with ``p = 0``: a uniform velocity
    has no strain, so the do-nothing ends force ``p = 0``, and the wall multipliers then vanish too.
    """

    def __init__(self, *, zeta=-0.7, E0=1.3, field_y=0.0, eps=2.0, mu=1.5, L=2.0, H=1.0,
                 N=(8, 4), wall_velocity=0, pass_permittivity=True, maxwell_stress=False,
                 impose_no_penetration=True, split_in_tris=False):
        super().__init__()
        self.split_in_tris = split_in_tris
        self.zeta, self.E0, self.field_y = zeta, E0, field_y
        self.eps, self.mu, self.L, self.H, self.N = eps, mu, L, H, N
        self.wall_velocity = wall_velocity
        self.pass_permittivity = pass_permittivity
        self.maxwell_stress = maxwell_stress
        self.impose_no_penetration = impose_no_penetration

    def expected_velocity(self):
        return -self.eps * self.zeta * self.E0 / self.mu

    def define_problem(self):
        x, y = var(["coordinate_x", "coordinate_y"])
        phi_e = -(self.E0 * x + self.field_y * y)   # E = -grad(phi) = (E0, field_y), uniform
        self.add_mesh(RectangularQuadMesh(N=list(self.N), size=[self.L, self.H],
                                          split_in_tris=self.split_in_tris))
        eqs = StokesEquations(dynamic_viscosity=self.mu)
        eqs += ElectricPotentialEquations(permittivity=self.eps, permittivity_scale=1,
                                          add_maxwell_stress_to_momentum=self.maxwell_stress)
        # phi_e is linear, hence exactly representable in C2 and exactly harmonic. With a purely
        # axial field the walls can stay insulating (natural); an oblique one needs Dirichlet data
        # on all four sides, otherwise the normal component would be the thing that is wrong.
        for b in (["left", "right"] if self.field_y == 0 else ["left", "right", "top", "bottom"]):
            eqs += ElectrodeBC(phi_e) @ b

        def slip():
            return ElectroosmoticSlip(
                zeta_potential=self.zeta,
                permittivity=self.eps if self.pass_permittivity else None,
                wall_velocity=self.wall_velocity,
                impose_no_penetration=self.impose_no_penetration)

        eqs += slip() @ "top"
        eqs += slip() @ "bottom"
        if not self.impose_no_penetration:
            # Without no-penetration the walls leave u_y free, and a uniform u_y is then a genuine
            # nullspace of the whole problem (zero strain, zero traction). Close it at the ends.
            eqs += DirichletBC(velocity_y=0) @ "left"
            eqs += DirichletBC(velocity_y=0) @ "right"
        u = var("velocity")
        eqs += IntegralObservables(_area=1, ux=u[0], uy=u[1], p=var("pressure"),
                                   ux_err=subexpression((u[0] - self.expected_velocity()) ** 2))
        self.add_equations(eqs @ "domain")


def _nodal_range(mesh, name):
    idx = mesh.get_nodal_field_indices()[name]
    vals = [n.value(idx) for n in mesh.nodes()]
    return min(vals), max(vals)


def _run_eof(**kw):
    prob = _EOFChannel(**kw)
    with _fresh(prob) as p:
        p.solve()
        mesh = p.get_mesh("domain")
        o = {k: float(v) for k, v in mesh.evaluate_all_observables().items()}
        # IntegralObservables really integrates: divide by the area to get the mean, which is what
        # the plug value is compared against.
        area = o.pop("_area")
        o = {k: v / area for k, v in o.items()}
        o["ux_min"], o["ux_max"] = _nodal_range(mesh, "velocity_x")
        o["uy_min"], o["uy_max"] = _nodal_range(mesh, "velocity_y")
    return prob, o


# Relative floors, see the note at the top of this section: quadrilaterals inherit the ~1e-9
# absolute defect of the 2D 3x3 Gauss knot table, triangles do not.
_MESHES = [pytest.param(False, 1e-6, id="quads"), pytest.param("left", 1e-11, id="tris")]


@pytest.mark.parametrize("tris,tol", _MESHES)
def test_helmholtz_smoluchowski_plug_flow(tris, tol):
    """T17: the slip velocity fills the whole channel.

    On triangles this is exact -- the constant velocity, ``p = 0`` and vanishing wall multipliers
    satisfy the discrete residual identically, so the assertion at 1e-11 is a statement about the
    formulation and not about a convergence rate.
    """
    prob, o = _run_eof(split_in_tris=tris)
    u_hs = prob.expected_velocity()
    assert u_hs > 0, "a negative zeta with a field along +x drives the liquid along +x"
    assert o["ux_max"] - o["ux_min"] < tol * u_hs, "plug flow, so the profile must be flat"
    assert o["ux"] == pytest.approx(u_hs, rel=tol)
    assert max(abs(o["uy_max"]), abs(o["uy_min"])) < tol * u_hs
    # p == 0 is not decoration: a uniform velocity has no viscous traction, so the open ends can
    # only be traction-free if the pressure vanishes. A nonzero p here would mean the slip is
    # fighting something.
    assert abs(o["p"]) < tol * u_hs * prob.mu


@pytest.mark.parametrize("zeta,E0", [(-0.7, 1.3), (-0.7, -1.3), (0.7, 1.3), (0.7, -1.3)])
def test_slip_direction_follows_zeta_times_field(zeta, E0):
    """The plug reverses with either sign, and only with the product of the two."""
    prob, o = _run_eof(zeta=zeta, E0=E0, split_in_tris="left")
    assert o["ux"] == pytest.approx(prob.expected_velocity(), rel=1e-11)
    assert numpy.sign(o["ux"]) == -numpy.sign(zeta * E0)


@pytest.mark.parametrize("tris,tol", _MESHES)
def test_slip_ignores_the_normal_field(tris, tol):
    """Only E_t drives the slip -- the wall-normal component must be projected out.

    ``helmholtz_smoluchowski_velocity`` reads ``grad(var("phi"))`` *on the interface*, where grad is
    the surface gradient, and the residual projects tangentially on top of that. Here the field is
    oblique with a normal component more than twice the axial one, so a formulation that let the
    normal part through would either move the plateau or fight the no-penetration constraint.
    """
    prob, o = _run_eof(field_y=3.0, split_in_tris=tris)
    u_hs = prob.expected_velocity()
    assert o["ux_max"] - o["ux_min"] < tol * u_hs
    assert o["ux"] == pytest.approx(u_hs, rel=tol)
    assert max(abs(o["uy_max"]), abs(o["uy_min"])) < tol * u_hs


def test_permittivity_is_taken_from_the_potential_equations():
    """Omitting ``permittivity=`` must find it on the parent domain, not silently fall back to 1."""
    _, given = _run_eof(pass_permittivity=True, split_in_tris="left")
    prob, found = _run_eof(pass_permittivity=False, split_in_tris="left")
    assert found["ux"] == pytest.approx(given["ux"], rel=1e-12)
    assert found["ux"] == pytest.approx(prob.expected_velocity(), rel=1e-11)


def test_wall_velocity_adds_to_the_slip():
    """The slip is measured relative to the wall, so a moving wall shifts the whole plug."""
    v_wall = 0.4
    prob, o = _run_eof(wall_velocity=vector(v_wall, 0), split_in_tris="left")
    assert o["ux"] == pytest.approx(prob.expected_velocity() + v_wall, rel=1e-11)
    assert o["ux_max"] - o["ux_min"] < 1e-11 * abs(o["ux"])


def test_no_penetration_can_be_switched_off():
    """Without it the normal direction is natural, which the plug already satisfies.

    The walls then leave ``u_y`` free and a uniform ``u_y`` would be a genuine nullspace of the whole
    problem, so the ends pin it; the point of the test is that the tangential constraint alone still
    produces the same plug.
    """
    prob, o = _run_eof(impose_no_penetration=False, split_in_tris="left")
    assert o["ux"] == pytest.approx(prob.expected_velocity(), rel=1e-11)
    assert max(abs(o["uy_max"]), abs(o["uy_min"])) < 1e-11 * abs(o["ux"])


def test_maxwell_stress_leaves_the_plug_flow_alone():
    """A uniform field makes sigma_M constant, so it is a pure boundary term.

    ``weak(sigma_M, grad(v))`` with a constant tensor reduces to the surface integral, whose axial
    part at the open ends is ``eps*E0^2/2``; the pressure absorbs exactly that and the velocity is
    untouched. Worth asserting because the automatic coupling is on by DEFAULT, so this is what a
    user who writes ``ElectricPotentialEquations`` next to a Stokes actually gets -- an
    electroosmotic channel must not need to know about the Maxwell stress to come out right.
    """
    prob, off = _run_eof(maxwell_stress=False, split_in_tris="left")
    _, on = _run_eof(maxwell_stress=True, split_in_tris="left")
    assert on["ux"] == pytest.approx(off["ux"], rel=1e-10)
    assert on["ux_max"] - on["ux_min"] < 1e-10 * abs(on["ux"])
    assert on["p"] - off["p"] == pytest.approx(prob.eps * prob.E0 ** 2 / 2, rel=1e-10)


class _WaterEOFChannel(Problem):
    """The same channel in SI units, with water, a real zeta and a real field strength.

    Everything above runs with unit scales, where a factor of ``epsilon_0`` cannot go missing. This
    one carries the whole nondimensionalisation -- separate scales for length, velocity, pressure,
    potential and permittivity -- and still has to land on the textbook number, so it is what says
    the slip survives the scaling machinery rather than only the algebra.
    """

    zeta = -50 * milli * volt          # silica in water, near-neutral pH
    E0 = 10 * kilo * volt / meter
    mu = 1 * milli * pascal * second
    eps_r = 78.3                       # water at 25 C
    H = 100 * micro * meter
    L = 400 * micro * meter

    def expected_velocity(self):
        return -self.eps_r * epsilon_0 * self.zeta * self.E0 / self.mu

    def define_problem(self):
        U = 1 * milli * meter / second
        self.set_scaling(spatial=self.H, velocity=U, temporal=self.H / U)
        self.set_scaling(pressure=self.mu * U / self.H)
        set_electrostatic_scaling(self, potential=self.E0 * self.L, permittivity=epsilon_0)
        x = var("coordinate_x")
        self.add_mesh(RectangularQuadMesh(N=[8, 4], size=[self.L, self.H], split_in_tris="left"))
        eqs = StokesEquations(dynamic_viscosity=self.mu)
        eqs += ElectricPotentialEquations(relative_permittivity=self.eps_r,
                                          add_maxwell_stress_to_momentum=False)
        for b in ["left", "right"]:
            eqs += ElectrodeBC(-self.E0 * x) @ b
        for b in ["top", "bottom"]:
            eqs += ElectroosmoticSlip(zeta_potential=self.zeta) @ b
        eqs += IntegralObservables(_area=1, ux=var("velocity")[0])
        self.add_equations(eqs @ "domain")


def test_water_channel_hits_the_textbook_slip_velocity():
    prob = _WaterEOFChannel()
    with _fresh(prob) as p:
        p.solve()
        o = p.get_mesh("domain").evaluate_all_observables()
        u_mean = float(o["ux"] / o["_area"] / (milli * meter / second))
    # 78.3*eps_0*50 mV*10 kV/m / 1 mPa s = 0.347 mm/s, the usual few-hundred-micron-per-second of
    # an electroosmotic pump. Hard-coded as well as computed, so that a change in epsilon_0 or in
    # the unit handling cannot move both sides together.
    assert u_mean == pytest.approx(float(prob.expected_velocity() / (milli * meter / second)),
                                   rel=1e-10)
    assert u_mean == pytest.approx(0.34664, rel=1e-4)


def test_slip_without_a_zeta_potential_is_refused():
    class _NoZeta(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=4))
            eqs = StokesEquations(dynamic_viscosity=1).with_pressure_fixation(nondim_p_value=0.0)
            eqs += ElectricPotentialEquations(permittivity=1, permittivity_scale=1,
                                              add_maxwell_stress_to_momentum=False)
            eqs += ElectrodeBC(0) @ "left"
            eqs += ElectroosmoticSlip() @ "bottom"
            self.add_equations(eqs @ "domain")

    with pytest.raises(ValueError, match="zeta_potential"):
        with _fresh(_NoZeta()) as p:
            p.initialise()


def test_slip_without_a_findable_permittivity_is_refused():
    """A bare Stokes domain has nothing to read eps from, and guessing 1 would be silently wrong."""

    class _NoPermittivity(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=4))
            eqs = StokesEquations(dynamic_viscosity=1).with_pressure_fixation(nondim_p_value=0.0)
            eqs += ElectroosmoticSlip(zeta_potential=0.1) @ "bottom"
            self.add_equations(eqs @ "domain")

    with pytest.raises(RuntimeError, match="permittivity"):
        with _fresh(_NoPermittivity()) as p:
            p.initialise()
