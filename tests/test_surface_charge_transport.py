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

"""
Surface charge on a deforming, evaporating interface, and the ions that feed it.

The question this file answers: a droplet evaporates, so its surface shrinks - does the free charge
sitting on that surface stay put? It must. Evaporation removes solvent, not charge, so the density
has to rise exactly as fast as the area falls and the total must not move at all.

The testbed prescribes the interface motion instead of solving for it, exactly as
``test_surfactant_transport.py`` does: the mesh velocity and the fluid velocity are given affine
fields, so the dilatation, the tangential slip and the normal slip (= mass transfer) are all known
and the only error measured is the transport scheme's.

  1. THE CONSERVATIVE FORM CONSERVES THE CHARGE. To the Newton tolerance, not to the order of the
     time stepping. The legacy form drifts at O(dt^p) - including under a purely tangential mesh
     slide, where the interface does not even change shape.
  2. THE ENDS OF AN OPEN INTERFACE DO NOTHING UNLESS ASKED. Omitting the contour term of the
     conservative form is the zero-end-flux condition; SurfaceChargeEndFlux imposes a nonzero one.
  3. AD-/DESORPTION MOVES THE RIGHT AMOUNT IN THE RIGHT DIRECTION. The per-ion form pins the whole
     sign chain at once: what the surface gains, the bulk loses, and the total charge is constant.
  4. THE ION TRANSPORT CONSERVES TOO, once GCL is on. A drying film keeps its dissolved ions, and
     without GCL it loses exactly the thinning ratio - 39% here - at any time step.
  5. SO DO THE OTHER TRANSPORT EQUATIONS. AdvectionDiffusionEquations and the temperature had no
     conservative ALE branch at all.

Each Problem gets its own output directory: several Problems in one directory share the JIT cache,
and variants that differ only in constructor flags then silently reuse the first one's compiled code.
"""

import math
import tempfile

import pytest

from typing import Literal

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.expressions.units import *
from pyoomph.expressions.phys_consts import *
from pyoomph.equations.ALE import PrescribedMovingMesh
from pyoomph.equations.generic import ElementSpace, IntegralObservables
from pyoomph.equations.electrostatics import (ElectricPotentialEquations, SurfaceChargeConservation,
                                              SurfaceChargeEndFlux, NernstPlanckEquations, IonSpec,
                                              set_electrostatic_scaling)
from pyoomph.meshes.simplemeshes import CircularMesh, LineMesh


# ---------------------------------------------------------------- the prescribed-motion testbed

class _PrescribedInterface(Problem):
    """A sphere (axisymmetric) or a circle (Cartesian) whose motion is dictated, not solved for.

    ``a_m`` dilates the mesh, ``a_f`` the fluid: their difference is a normal slip, i.e. exactly what
    evaporation does. ``om_m``/``om_f`` rotate them, so their difference is a tangential slip -- the
    case a Laplace- or pseudo-elastically smoothed mesh is in permanently.

    The bulk carries ``ElectricPotentialEquations`` because that is the required parent type, but it
    is deliberately inert: no conductivity, ``bulk_currents=0``, ``surface_conductivity=0``. The
    potential is therefore decoupled from the charge and pinned on the interface to remove its
    nullspace, so that any change in the total charge is the transport scheme's doing and nothing
    else's.
    """

    def __init__(self, *, form: Literal["conservative", "legacy"] = "conservative",
                 a_m=0.0, a_f=0.0, om_m=0.0, om_f=0.0, K_s=0.0, end_flux=None,
                 axisymmetric=True, nref=2):
        super().__init__()
        self.form: Literal["conservative", "legacy"] = form
        self.a_m, self.a_f, self.om_m, self.om_f = a_m, a_f, om_m, om_f
        self.K_s, self.axisymmetric, self.nref = K_s, axisymmetric, nref
        self.end_flux: ExpressionNumOrNone = end_flux

    def define_problem(self):
        x, y = var("coordinate_x"), var("coordinate_y")
        if self.axisymmetric:
            self.set_coordinate_system("axisymmetric")
            self += CircularMesh(radius=1, segments=["NE", "SE"], outer_interface="ifc",
                                 straight_interface_name={"center_to_north": "axis",
                                                          "center_to_south": "axis"})
            # No rotation in axisymmetry: an azimuthal velocity is out of the plane.
            w = vector(self.a_m * x, self.a_m * y)
            u = vector(self.a_f * x, self.a_f * y)
        else:
            self.set_coordinate_system(cartesian)
            self += CircularMesh(radius=1, outer_interface="ifc")
            w = vector(self.a_m * x - self.om_m * y, self.a_m * y + self.om_m * x)
            u = vector(self.a_f * x - self.om_f * y, self.a_f * y + self.om_f * x)

        eqs = PrescribedMovingMesh(w) + ElementSpace("C2")
        eqs += ElectricPotentialEquations(relative_permittivity=1)
        # Written in Lagrangian coordinates, so the initial profile does not depend on where the
        # mesh has moved to by the time it is evaluated.
        X, Y = var("lagrangian_x"), var("lagrangian_y")
        q0 = 1 + rational_num(1, 2) * (Y if self.axisymmetric else X * X - Y * Y)
        self.surf = SurfaceChargeConservation(name="qs", form=self.form, surface_diffusivity=self.K_s,
                                              fluid_velocity=u, interface_velocity=w,
                                              bulk_currents=0, initial_charge=q0)
        ieqs = self.surf
        ieqs += DirichletBC(phi=0)
        ieqs += IntegralObservables(charge=var("qs"), area=1)
        if self.end_flux is not None:
            ieqs += SurfaceChargeEndFlux(self.end_flux) @ "axis"
        eqs += ieqs @ "ifc"
        if self.axisymmetric:
            eqs += DirichletBC(mesh_x=0) @ "axis"
        self += eqs @ "domain"


def _evolve(problem, nsteps=20, T=1.0):
    """Run the testbed and return the initial and final total charge and interface area."""
    problem.set_output_directory(tempfile.mkdtemp(prefix="surfcharge_"))
    problem.quiet()
    problem.newton_solver_tolerance = 1e-11
    problem.initialise()
    for _ in range(problem.nref):
        problem.refine_uniformly()
    # refine_uniformly places new boundary nodes on the piecewise-quadratic arc rather than on the
    # exact circle, and further refinement does not reduce that. Snap them, so the geometry is a
    # converging discretization of a sphere and the discrete normal keeps improving with h.
    ifc = problem.get_mesh("domain/ifc")
    for nd in ifc.nodes():
        r = math.hypot(nd.x(0), nd.x(1))
        for i in range(2):
            nd.set_x(i, nd.x(i) / r)
    problem.get_mesh("domain").set_lagrangian_nodal_coordinates()
    problem.set_initial_condition(ic_name="")
    obs = ifc.evaluate_all_observables()
    Q0, A0 = float(obs["charge"]), float(obs["area"])
    for _ in range(nsteps):
        problem.solve(timestep=T / nsteps, do_not_set_IC=True)
    obs = ifc.evaluate_all_observables()
    return dict(Q0=Q0, Q=float(obs["charge"]), A0=A0, A=float(obs["area"]))


def _drift(**kwargs):
    nsteps = kwargs.pop("nsteps", 20)
    r = _evolve(_PrescribedInterface(**kwargs), nsteps=nsteps)
    return r["Q"] / r["Q0"] - 1


# ---------------------------------------------------------------- 1. conservation under evaporation

@pytest.mark.parametrize("case", [
    pytest.param(dict(a_m=-0.3, a_f=-0.3), id="dilatation"),
    pytest.param(dict(a_m=-0.3, a_f=0.0), id="evaporation"),
    pytest.param(dict(a_m=0.0, a_f=0.0, om_m=0.6, om_f=1.0, axisymmetric=False), id="tangential_slip"),
    pytest.param(dict(a_m=-0.2, a_f=0.05, om_m=-0.4, om_f=0.8, axisymmetric=False), id="everything"),
])
def test_the_conservative_form_conserves_the_surface_charge(case):
    # The transient is the derivative of the whole integral and the advection is a flux, so the
    # constant test function makes the residual a telescoping difference of the total charge. That is
    # a property of the discrete system, not of the time step: the floor here is the Newton
    # tolerance, not the discretization.
    assert abs(_drift(form="conservative", **case)) < 1e-10


def test_evaporation_shrinks_the_area_and_raises_the_density():
    # The physical statement the whole file is about: the interface really does shrink, the charge
    # density really does rise, and the product does not move.
    r = _evolve(_PrescribedInterface(form="conservative", a_m=-0.3, a_f=0.0), nsteps=20)
    assert r["A"] / r["A0"] == pytest.approx(math.exp(-0.6), rel=1e-3)
    assert abs(r["Q"] / r["Q0"] - 1) < 1e-10
    # mean density = Q/A, so it must have risen by exactly the area ratio
    assert (r["Q"] / r["A"]) / (r["Q0"] / r["A0"]) == pytest.approx(math.exp(0.6), rel=1e-3)


def test_the_legacy_drift_under_pure_dilatation_is_second_order_in_the_time_step():
    # With u == w the mesh and the liquid move together, so the only thing the legacy form gets wrong
    # is that the discrete rate of change of the surface metric is not the discrete div_s(w). That is
    # a pure time-stepping error: halving dt divides it by four, and no mesh refinement removes it.
    case = dict(form="legacy", a_m=-0.3, a_f=-0.3)
    coarse = abs(_drift(nsteps=20, **case))
    fine = abs(_drift(nsteps=40, **case))
    assert coarse > 1e-4
    assert coarse / fine == pytest.approx(4.0, rel=0.2)


def test_under_evaporation_the_legacy_error_does_not_converge_in_the_time_step_at_all():
    """The legacy form is worse than second order here, and this is the sharpest reason to change it.

    Under a normal slip the legacy ``div(q*u_s)`` differentiates the element normal, which the
    conservative form never touches. That leaves a *spatial* conservation error the time step cannot
    reach: refining dt drives the drift to a nonzero limit -- through a sign change, so a two-point
    convergence study lands on a ratio of 4 by coincidence and reports second order. Measured at
    nref=2: 6.8e-4, -1.7e-4, -3.8e-4, -4.4e-4 for 20/40/80/160 steps. The limit is second order in h
    (-1.9e-3, -4.4e-4, -9.8e-5 at nref=1/2/3), so it is a discretization of the normal, not a defect
    of the time integration.
    """
    case = dict(form="legacy", a_m=-0.3, a_f=0.0)
    tail = [abs(_drift(nsteps=n, **case)) for n in (80, 160)]
    assert tail[0] > 1e-4 and tail[1] > 1e-4
    assert tail[1] / tail[0] > 0.8          # it is not shrinking with dt
    coarse_h = abs(_drift(nsteps=160, nref=1, **case))
    fine_h = abs(_drift(nsteps=160, nref=2, **case))
    assert coarse_h / fine_h == pytest.approx(4.0, rel=0.3)   # but it is second order in h
    # the conservative form has neither error
    assert abs(_drift(form="conservative", nsteps=160, a_m=-0.3, a_f=0.0)) < 1e-10


def test_tangential_mesh_sliding_alone_breaks_the_legacy_form():
    # The sharpest case: the interface is a circle at every instant and its length does not change,
    # so there is no dilatation to get wrong. Only the nodes slide along it - which is what every
    # smoothed moving mesh does permanently - and that alone is enough.
    case = dict(a_m=0.0, a_f=0.0, om_m=0.6, om_f=1.0, axisymmetric=False)
    assert abs(_drift(form="conservative", **case)) < 1e-10
    assert abs(_drift(form="legacy", **case)) > 1e-4


def test_surface_diffusion_does_not_break_the_conservation():
    # Written against grad(q_test), so the constant test function annihilates it. Same argument as
    # for the surfactant stabilizations.
    assert abs(_drift(form="conservative", a_m=-0.3, a_f=0.0, K_s=0.05)) < 1e-10


# ---------------------------------------------------------------- 2. the ends of an open interface

def test_the_ends_lose_nothing_unless_asked():
    # In axisymmetry the interface is an arc from pole to pole, so it has two ends. Omitting the
    # contour term of the by-parts form is exactly the zero-total-flux condition.
    assert abs(_drift(form="conservative", a_m=-0.3, a_f=0.0)) < 1e-10


def test_an_end_flux_on_the_symmetry_axis_is_inert():
    # The measure of a point domain in axisymmetry carries 2*pi*r, so an end point sitting on the
    # axis has zero ring circumference and cannot remove anything, however large the flux.
    r = _evolve(_PrescribedInterface(form="conservative", a_m=-0.3, a_f=0.0, end_flux=0.25))
    assert abs(r["Q"] / r["Q0"] - 1) < 1e-10


class _CartesianEndFluxInterface(_PrescribedInterface):
    """Same testbed, but the end flux uses a Cartesian point measure so that it is not silenced by
    the 2*pi*r the axisymmetric point domain carries."""

    def define_problem(self):
        x, y = var("coordinate_x"), var("coordinate_y")
        self.set_coordinate_system("axisymmetric")
        self += CircularMesh(radius=1, segments=["NE", "SE"], outer_interface="ifc",
                             straight_interface_name={"center_to_north": "axis",
                                                      "center_to_south": "axis"})
        w = vector(self.a_m * x, self.a_m * y)
        u = vector(self.a_f * x, self.a_f * y)
        eqs = PrescribedMovingMesh(w) + ElementSpace("C2")
        eqs += ElectricPotentialEquations(relative_permittivity=1)
        self.surf = SurfaceChargeConservation(name="qs", form=self.form, fluid_velocity=u,
                                              interface_velocity=w, bulk_currents=0, initial_charge=1)
        ieqs = self.surf
        ieqs += DirichletBC(phi=0)
        ieqs += IntegralObservables(charge=var("qs"), area=1)
        assert self.end_flux is not None
        ieqs += SurfaceChargeEndFlux(self.end_flux, coordsys=cartesian) @ "axis"
        eqs += ieqs @ "ifc"
        eqs += DirichletBC(mesh_x=0) @ "axis"
        self += eqs @ "domain"


def test_an_end_flux_removes_exactly_what_it_says():
    # A static arc with two end points, each carrying a Cartesian point measure of 1, so over T=1 at
    # a flux of 0.25 exactly 2*0.25 has to leave. Positive means leaving.
    r = _evolve(_CartesianEndFluxInterface(form="conservative", end_flux=0.25), nsteps=10)
    assert r["Q"] - r["Q0"] == pytest.approx(-2 * 0.25, abs=1e-9)


# ---------------------------------------------------------------- 3. ad-/desorption

class _AdsorbingWall(Problem):
    """A 1d electrolyte film whose right boundary adsorbs charge.

    Fully dimensional, so that the ``z*F*R`` conversion between a molar rate and a charge flux is
    actually checked rather than hidden behind scales of 1 - the same reason
    ``test_electric_field_matches_gradient`` runs with a potential scale that is not 1. The right
    boundary of a 1d Cartesian mesh is a point of measure 1, so a rate per unit area is also an
    absolute rate.

    Migration is switched off (``mobility=0``), which decouples the potential: the only thing that
    changes the dissolved amount is what the interface takes out of it.
    """

    _c0 = 1 * mol / meter ** 3
    _D = 1e-9 * meter ** 2 / second
    _L = 1 * milli * meter

    def __init__(self, *, rate, per_ion: bool, valence=1, N=20, bulk_coupling="auto"):
        super().__init__()
        self.rate, self.per_ion, self.valence = rate, per_ion, valence
        self.N, self.bulk_coupling = N, bulk_coupling

    def define_problem(self):
        self.set_scaling(spatial=self._L, temporal=1 * second)
        set_electrostatic_scaling(self, potential=25 * milli * volt, ion_concentration=self._c0,
                                  surface_charge_density=1 * milli * coulomb / meter ** 2)
        self += LineMesh(N=self.N, size=self._L)
        eqs = ElectricPotentialEquations(relative_permittivity=1)
        eqs += NernstPlanckEquations([IonSpec("X", self.valence, diffusivity=self._D, mobility=0,
                                              bulk_concentration=self._c0)],
                                     wind=0, temperature=293 * kelvin)
        eqs += DirichletBC(phi=0) @ "left"
        ads = {"X": self.rate} if self.per_ion else self.rate
        ifeqs = SurfaceChargeConservation(name="qs", adsorption=ads, bulk_currents=0,
                                          bulk_coupling=self.bulk_coupling, initial_charge=0)
        ifeqs += DirichletBC(phi=0)
        ifeqs += IntegralObservables(qs=var("qs"), _pt=1)
        eqs += ifeqs @ "right"
        eqs += IntegralObservables(moles=var("c_X"), volume=1)
        self += eqs @ "domain"


def _adsorb(problem, nsteps=10, T=1 * second):
    problem.set_output_directory(tempfile.mkdtemp(prefix="surfcharge_ads_"))
    problem.quiet()
    problem.newton_solver_tolerance = 1e-11
    problem.initialise()
    problem.set_initial_condition(ic_name="")

    def read():
        b = problem.get_mesh("domain").evaluate_all_observables()
        i = problem.get_mesh("domain/right").evaluate_all_observables()
        # 1d: the bulk integral is per unit cross-sectional area, and so is the boundary point,
        # whose Cartesian measure is 1. Both therefore come back per m^2.
        return float(b["moles"] / (mol / meter ** 2)), float(i["qs"] / (coulomb / meter ** 2))

    moles0, Q0 = read()
    for _ in range(nsteps):
        problem.solve(timestep=T / nsteps, do_not_set_IC=True)
    moles, Q = read()
    return dict(moles0=moles0, moles=moles, Q0=Q0, Q=Q)


_RATE = 2e-8 * mol / (meter ** 2 * second)      # a molar ad-/desorption rate
_QRATE = 1e-4 * coulomb / (meter ** 2 * second)  # a net charge flux


@pytest.mark.parametrize("sign", [1, -1], ids=["adsorb", "desorb"])
def test_a_charge_flux_deposits_exactly_what_it_says(sign):
    # The simple form: a net charge flux in C/(m^2 s), positive towards the interface. Over 1 s on a
    # point of measure 1 the surface must gain exactly that, and a negative rate must desorb.
    r = _adsorb(_AdsorbingWall(rate=sign * _QRATE, per_ion=False))
    expect = sign * float(_QRATE * (1 * second) / (coulomb / meter ** 2))
    assert r["Q"] - r["Q0"] == pytest.approx(expect, rel=1e-9)


@pytest.mark.parametrize("valence", [1, -1, 2])
def test_the_per_ion_form_moves_charge_and_ions_consistently(valence):
    """The one test that pins the whole sign chain at once.

    A molar rate R adsorbing on the interface must (a) add z*F*R to the surface charge, (b) remove
    exactly R moles from the bulk, and therefore (c) leave the total charge, bulk plus surface,
    unchanged. Getting the bulk sign backwards passes (a) and fails (b) and (c) - which is exactly
    the failure mode the wrong `IonFluxBC` docstring would have produced.
    """
    r = _adsorb(_AdsorbingWall(rate=_RATE, per_ion=True, valence=valence))
    F = float(faraday_constant / (coulomb / mol))
    moles = float(_RATE * (1 * second) / (mol / meter ** 2))
    assert r["Q"] - r["Q0"] == pytest.approx(valence * F * moles, rel=1e-9)
    assert r["moles"] - r["moles0"] == pytest.approx(-moles, rel=1e-7)
    total0 = valence * F * r["moles0"] + r["Q0"]
    total = valence * F * r["moles"] + r["Q"]
    assert (total - total0) / abs(total0) == pytest.approx(0.0, abs=1e-9)


def test_bulk_coupling_can_be_switched_off():
    # Without the coupling the surface still gains the charge, but nothing is taken out of the bulk:
    # the charge appears from nowhere. Legitimate when the reservoir is not modelled, so it is an
    # option rather than an error - but it must be the explicit choice.
    r = _adsorb(_AdsorbingWall(rate=_RATE, per_ion=True, bulk_coupling=False))
    F = float(faraday_constant / (coulomb / mol))
    moles = float(_RATE * (1 * second) / (mol / meter ** 2))
    assert r["Q"] - r["Q0"] == pytest.approx(F * moles, rel=1e-9)
    assert r["moles"] - r["moles0"] == pytest.approx(0.0, abs=1e-12)


def test_an_unknown_ion_is_refused():
    with pytest.raises(Exception, match="does not transport|no NernstPlanckEquations"):
        _adsorb(_AdsorbingWall(rate={"Y": _RATE}, per_ion=False))


# ---------------------------------------------------------------- 4. the ions in a drying film

class _DryingFilm(Problem):
    """A 1d film whose right end recedes, with the liquid at rest.

    The lightweight version of ``test_salt_transport.py::_EvaporatingFilm``: the mesh motion is
    prescribed rather than solved for, so no flow, no materials and no mass-transfer model are
    involved and the only thing under test is the ALE form of the ion transport. The left end is a
    wall and the liquid does not move at all -- the interface simply sweeps past it -- so the answer
    is exact: the ions cannot go anywhere and ``int c dV`` must not change.
    """

    def __init__(self, *, GCL="auto", N=20, rate=0.5, T=1 * second):
        super().__init__()
        self.GCL, self.N, self.rate, self.T = GCL, N, rate, T

    _c0 = 1 * mol / meter ** 3
    _D = 1e-9 * meter ** 2 / second
    _L = 1 * milli * meter

    def define_problem(self):
        self.set_scaling(spatial=self._L, temporal=self.T)
        set_electrostatic_scaling(self, potential=25 * milli * volt, ion_concentration=self._c0)
        self += LineMesh(N=self.N, size=self._L)
        # A mesh velocity proportional to x: the left end is held and every node recedes at a rate
        # proportional to its distance from it, so the film thins exponentially, to exp(-rate*T).
        w = vector(-self.rate * var("coordinate_x") / self.T)
        eqs = PrescribedMovingMesh(w)
        eqs += ElectricPotentialEquations(relative_permittivity=1)
        eqs += NernstPlanckEquations([IonSpec("X", 1, diffusivity=self._D, mobility=0,
                                              bulk_concentration=self._c0)],
                                     wind=0, temperature=293 * kelvin, GCL=self.GCL)
        eqs += DirichletBC(phi=0) @ "left"
        eqs += DirichletBC(mesh_x=0) @ "left"
        eqs += IntegralObservables(moles=var("c_X"), length=1)
        self += eqs @ "domain"


def _dry(problem, nsteps=20):
    problem.set_output_directory(tempfile.mkdtemp(prefix="dryingfilm_"))
    problem.quiet()
    # Not 1e-11, and not 1e-10 at 80 steps: the decoupled Poisson block carries eps_0 and Newton
    # stalls there. 1e-9 converges at every step count, and the conservation floor is 6e-14 anyway -
    # unlike the surface-charge testbed this system is linear once the mesh motion is prescribed, so
    # the floor is machine precision rather than the Newton tolerance.
    problem.newton_solver_tolerance = 1e-9
    problem.initialise()
    problem.set_initial_condition(ic_name="")
    obs = problem.get_mesh("domain").evaluate_all_observables()
    N0, L0 = float(obs["moles"] / (mol / meter ** 2)), float(obs["length"] / meter)
    for _ in range(nsteps):
        problem.solve(timestep=problem.T / nsteps, do_not_set_IC=True)
    obs = problem.get_mesh("domain").evaluate_all_observables()
    return dict(N0=N0, N=float(obs["moles"] / (mol / meter ** 2)),
                L0=L0, L=float(obs["length"] / meter))


@pytest.mark.parametrize("nsteps", [20, 80])
def test_a_drying_film_keeps_its_ions_under_gcl(nsteps):
    # The conservative ALE form's natural boundary condition is zero flux *through the moving
    # boundary*, which is exactly what a non-volatile ion at a receding surface needs -- so that
    # interface needs no term at all and nothing is lost, at any step size.
    r = _dry(_DryingFilm(GCL=True), nsteps=nsteps)
    assert abs(r["N"] / r["N0"] - 1) < 1e-10
    # and the film really did thin, so the ions really did concentrate
    # exp(-0.5): the mesh velocity is proportional to x, so the thinning is exponential. The 4e-4
    # gap at 20 steps is the time integration of the prescribed mesh motion, not of the transport.
    assert r["L"] / r["L0"] == pytest.approx(math.exp(-0.5), rel=1e-3)


def test_without_gcl_the_receding_surface_sweeps_the_ions_out():
    """And it is not a small error, nor one that a smaller time step touches.

    The non-conservative form's natural condition is zero *diffusive* flux, so the retreating
    interface simply carries the ions away with it: the concentration never changes and the dissolved
    amount falls by exactly the thinning ratio. Here that is 39.3% of the ions, at 20 steps and
    unchanged at 160 (-3.9324e-01, -3.9341e-01, -3.9346e-01, -3.9347e-01). ``salt_transport.md``
    section 3 has the interface term that repairs it for the non-conservative branch; under GCL none
    is needed.
    """
    r = _dry(_DryingFilm(GCL=False), nsteps=20)
    # the amount fell by exactly the thinning ratio, i.e. c did not change at all
    assert r["N"] / r["N0"] == pytest.approx(r["L"] / r["L0"], rel=2e-3)
    # and refining dt does not help, because this is not a time-stepping error
    fine = _dry(_DryingFilm(GCL=False), nsteps=80)
    assert abs(fine["N"] / fine["N0"] - 1) > 0.39


def test_auto_switches_gcl_on_when_the_mesh_moves():
    # 'auto' is the default and must behave like GCL=True here: the mesh moves, so the conservative
    # form is the point. If auto ever stopped resolving, this test would drift like the one above.
    assert abs(_dry(_DryingFilm(), nsteps=20)["N"] / _dry(_DryingFilm(), nsteps=20)["N0"] - 1) < 1e-10


# ---------------------------------------------------------------- 5. the same cure elsewhere

class _ShrinkingSlab(Problem):
    """The drying film again, but for a plain transported scalar and for the temperature.

    Same statement, same mechanism, different equation class: the liquid is at rest, the right end
    recedes, and under GCL the conserved quantity - the amount, or the enthalpy - must not move.
    """

    def __init__(self, *, what: Literal["scalar", "temperature"], GCL, N=20, rate=0.5,
                 T=1 * second):
        super().__init__()
        self.what, self.GCL, self.N, self.rate, self.T = what, GCL, N, rate, T

    _L = 1 * milli * meter

    def define_problem(self):
        from pyoomph.equations.advection_diffusion import AdvectionDiffusionEquations
        from pyoomph.equations.multi_component import TemperatureConductionEquation
        from pyoomph.materials import get_pure_liquid
        import pyoomph.materials.default_materials  # noqa: F401  (registers "water")
        self.set_scaling(spatial=self._L, temporal=self.T)
        self += LineMesh(N=self.N, size=self._L)
        w = vector(-self.rate * var("coordinate_x") / self.T)
        eqs = PrescribedMovingMesh(w)
        if self.what == "scalar":
            D = 1e-9 * meter ** 2 / second
            self.set_scaling(c=1 * mol / meter ** 3, velocity=self._L / self.T)
            eqs += AdvectionDiffusionEquations("c", diffusivity=D, wind=0, GCL=self.GCL)
            eqs += InitialCondition(c=1 * mol / meter ** 3)
            eqs += IntegralObservables(amount=var("c"), length=1)
        else:
            water = get_pure_liquid("water")
            Tref = 300 * kelvin
            water.set_reference_scaling_to_problem(self, temperature=Tref)
            self.set_scaling(spatial=self._L, temporal=self.T, temperature=Tref)
            # All three properties constant, for two separate reasons. rho and cp: with rho(T)*cp(T)
            # the GCL form differentiates the enthalpy density rather than multiplying rho*cp onto
            # d_t T, which is a different model, so a conservation test would be comparing two
            # different equations. k: conserving the enthalpy of a domain shrinking to 0.61 of its
            # size means T rises to 495 K, which is outside water's conductivity correlation and
            # diverges Newton at step 5 - a property of the material data, not of the scheme.
            heat = TemperatureConductionEquation(water, GCL=self.GCL,
                                                 rho_override=1000 * kilogram / meter ** 3,
                                                 cp_override=4180 * joule / (kilogram * kelvin),
                                                 lambda_override=0.6 * watt / (meter * kelvin))
            eqs += heat
            eqs += InitialCondition(temperature=Tref)
            rho, cp, _ = heat.get_rho_cp_k()
            eqs += IntegralObservables(amount=rho * cp * var("temperature"), length=1)
        self += eqs @ "domain"


def _shrink(problem, nsteps=20):
    problem.set_output_directory(tempfile.mkdtemp(prefix="shrinkslab_"))
    problem.quiet()
    problem.newton_solver_tolerance = 1e-9
    problem.initialise()
    problem.set_initial_condition(ic_name="")
    a0 = problem.get_mesh("domain").evaluate_all_observables()["amount"]
    for _ in range(nsteps):
        problem.solve(timestep=problem.T / nsteps, do_not_set_IC=True)
    a = problem.get_mesh("domain").evaluate_all_observables()["amount"]
    return float(a / a0) - 1


@pytest.mark.parametrize("what", ["scalar", "temperature"])
def test_the_gcl_form_conserves_for_the_other_transport_equations(what):
    # AdvectionDiffusionEquations and TemperatureConductionEquation had no conservative ALE branch at
    # all, so a passive scalar and the enthalpy of a shrinking domain both drifted. For the
    # temperature the conserved quantity is rho*cp*T, which is why the integral is written that way.
    assert abs(_shrink(_ShrinkingSlab(what=what, GCL=True), nsteps=20)) < 1e-10
    assert abs(_shrink(_ShrinkingSlab(what=what, GCL=True), nsteps=80)) < 1e-10


@pytest.mark.parametrize("what", ["scalar", "temperature"])
def test_without_gcl_the_other_transport_equations_lose_it_too(what):
    # And they lose exactly the shrink ratio, for the same reason as the ions: the receding boundary
    # carries the quantity out and the density never changes.
    drift = _shrink(_ShrinkingSlab(what=what, GCL=False), nsteps=20)
    assert drift == pytest.approx(math.exp(-0.5) - 1, rel=5e-3)


def test_the_refactor_generates_the_same_code_when_gcl_is_off():
    """`AdvectionDiffusionEquations.define_residuals` was restructured to share a transient/advection
    helper between the two `fluid_props` branches. With `GCL=False` the generated C must be exactly
    what it was, for every `advection_by_parts` setting -- checked by md5 rather than asserted,
    because "it still passes the tests" is a much weaker statement than "it is the same code".

    Two hashes per setting, not one, because a byte hash of the file is not portable and the Windows
    wheel job of 29th August 2026 proved it: the file it generates is 337 lines and 24499 bytes, as
    here, and identical character for character once the DIGITS are stripped - only the last places
    of the six to twelve float literals differ, as platform arithmetic will. So the claim is split in
    two, and both halves are asserted everywhere:

      * `expected_structure` - the file with every digit removed. This is the whole of the code's
        shape: its statements, its identifiers, its order.
      * `expected_numbers` - the float literals, each rounded to twelve significant digits. That
        absorbs a difference in the last place (~1e-16 relative) and catches any real change of a
        coefficient, which is the only thing the digits could otherwise be hiding.

    `expected` keeps the exact bytes taken from the pre-refactor tree on Linux, for provenance and
    for the check that the three settings really do differ from each other. It is reported when it
    disagrees, but not asserted: with the structure and twelve significant digits both matching,
    what is left is round-off.
    """
    import glob
    import hashlib
    import os
    from pyoomph.equations.advection_diffusion import AdvectionDiffusionEquations

    import re

    expected = {False: "8086c09b7b2d5919e7c23817f927d90e",
                True: "23d23029c5c77112a0af2bc6b639fbd4",
                "skew": "33801fb787dad330235c7c7423552eb6"}
    expected_structure = {False: "3e3d3f2a93ed3c28e9a18a974fe7877e",
                          True: "019fe472baad87d904c75319ae5eb1da",
                          "skew": "11793398bc6b362339d3216fdfa2b3de"}
    expected_numbers = {False: "50c1e894df31cec9c82fe20772d27ac3",
                        True: "0e151f5ac962f4520492e98bb3dbd5d9",
                        "skew": "6b6ddc7e35a47d2af9b9b0d2ec2fc3da"}
    # The whole file with every float literal reprinted at twelve significant digits, and everything
    # else - integers included - left alone. This is the check that actually decides; the two above
    # only sharpen the message when it fails. It is needed because they have a hole between them: a
    # differing INTEGER (an index, or C1 against C2) survives digit-stripping and never reaches a
    # float hash, so both would pass while the code really had changed.
    expected_canonical = {False: "912d0c19574fc63a3f87f64da2ada5f6",
                          True: "7d97610368acdb4d749c08abc3b2473d",
                          "skew": "243dd9e705865c098547da77a1423fc6"}
    _FLOAT = re.compile(rb"[-+]?\d+\.\d*(?:[eE][-+]?\d+)?|[-+]?\d+[eE][-+]?\d+")

    class _P(Problem):
        def __init__(self, byparts):
            super().__init__()
            self.byparts = byparts

        def define_problem(self):
            self += LineMesh(N=6)
            eqs = AdvectionDiffusionEquations("c", diffusivity=1, wind=vector(1),
                                              advection_by_parts=self.byparts)
            eqs += DirichletBC(c=0) @ "left"
            self += eqs @ "domain"

    for byparts, md5 in expected.items():
        d = tempfile.mkdtemp(prefix="advdiff_md5_")
        p = _P(byparts)
        p.set_output_directory(d)
        p.quiet()
        p.set_c_compiler("system")
        p.initialise()
        raw = open(os.path.join(d, "_ccode", "domain.c"), "rb").read()
        # Normalise the newlines first: the file is written through a text-mode std::ofstream
        # (src/nanobind/problem.cpp), so on Windows every newline reaches the disk as \r\n.
        text = raw.replace(b"\r\n", b"\n")

        structure = hashlib.md5(re.sub(rb"[0-9]", b"", text)).hexdigest()
        assert structure == expected_structure[byparts], (
            "advection_by_parts=%r changed the SHAPE of the generated code (%d lines, %d bytes)"
            % (byparts, text.count(b"\n"), len(text)))

        floats = [float(m.group(0)) for m in _FLOAT.finditer(text)]
        numbers = hashlib.md5("|".join("%.12g" % v for v in floats).encode()).hexdigest()
        canonical = hashlib.md5(
            _FLOAT.sub(lambda m: ("%.12g" % float(m.group(0))).encode(), text)).hexdigest()
        assert canonical == expected_canonical[byparts], (
            "advection_by_parts=%r changed the generated code (%d lines, %d bytes, %d float "
            "literals; structure %s, numbers %s)"
            % (byparts, text.count(b"\n"), len(text), len(floats),
               "matches" if structure == expected_structure[byparts] else "differs",
               "match" if numbers == expected_numbers[byparts] else "differ"))

        got = hashlib.md5(text).hexdigest()
        if got != expected[byparts]:
            # Not a failure: the shape and twelve significant digits of every number already agree,
            # so this is the last place of a float literal and nothing else.
            print("note: the exact bytes differ from the Linux reference for advection_by_parts=%r "
                  "(%s vs %s), while the structure and the numbers to 12 significant digits match "
                  "- platform arithmetic, not a code change" % (byparts, got, expected[byparts]))
    # sanity: the three settings really do differ from each other, so the check above is not vacuous
    assert len(set(expected.values())) == 3
    assert glob.glob(os.path.join(d, "_ccode", "*.c"))


def test_the_default_field_name_and_scale_do_not_collide():
    """`charge_scale` may not default to the field's own name.

    `define_scalar_field(name, scale=scale_factor(name))` is self-referential and dies at code
    generation with "Cannot expand the expression any further" -- and only once something has
    actually registered that scale, which `set_electrostatic_scaling` does. So the class used to be
    unusable with both of its own defaults, and every test and example dodged it by renaming the
    field. The scale is now called "surface_charge" and the field "surface_charge_density".
    """
    from pyoomph.equations.electrostatics import (set_electrostatic_scaling,
                                                  ElectricPotentialConnection, ElectrodeBC)

    class _P(Problem):
        def __init__(self, name, charge_scale=None):
            super().__init__()
            self.name_, self.charge_scale = name, charge_scale

        def define_problem(self):
            L = 1 * milli * meter
            self.set_scaling(spatial=L, temporal=1 * second)
            set_electrostatic_scaling(self, potential=1 * volt,
                                      surface_charge_density=1e-9 * coulomb / meter ** 2)
            self += LineMesh(N=8, size=L, name=lambda x: "A" if x < 0.5 else "B")
            a = ElectricPotentialEquations(relative_permittivity=80)
            kw = {} if self.charge_scale is None else {"charge_scale": self.charge_scale}
            ifc = SurfaceChargeConservation(name=self.name_, bulk_currents=0,
                                            advection_velocity=0, **kw)
            ifc += ElectricPotentialConnection(surface_charge_density=self.name_)
            a += ifc @ "A_B"
            b = ElectricPotentialEquations(relative_permittivity=1)
            b += ElectrodeBC(0) @ "right"
            self += a @ "A"
            self += b @ "B"

    # the default name, with the default scale: the case that used to fail
    for name in ("surface_charge_density", "qs"):
        p = _P(name)
        p.set_output_directory(tempfile.mkdtemp(prefix="scname_"))
        p.quiet()
        p.initialise()
        p.solve(timestep=0.1 * second)
    # set_electrostatic_scaling registers both names, so either can be asked for explicitly
    for scale in ("surface_charge", "surface_charge_density"):
        p = _P("qs", charge_scale=scale)
        p.set_output_directory(tempfile.mkdtemp(prefix="scname_"))
        p.quiet()
        p.initialise()
        p.solve(timestep=0.1 * second)
