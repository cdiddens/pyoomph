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

"""
Surfactant transport along a deforming interface.

The testbed prescribes the interface motion instead of solving for it: the mesh velocity and the
fluid velocity are given affine fields, so the dilatation, the tangential slip and the normal slip
(= mass transfer) are all known exactly and the only error measured is the transport scheme's. Since
the prescribed field is affine and identical for every node, the *discrete* interface is at all times
an exact affine image of the initial one, so exact discrete references exist.

Four things have to hold, and each has a section below.

  1. AN INSOLUBLE SURFACTANT IS CONSERVED. The amount on a closed interface cannot change, and the
     conservative form keeps it to the Newton tolerance -- not to the order of the time stepping.
     The legacy form drifts at O(dt^p), which no mesh refinement removes.
  2. THE LEGACY FORM IS STILL EXACTLY THE OLD ONE, so that a published run can be reproduced.
  3. THE ENDS OF THE INTERFACE DO NOTHING UNLESS ASKED. Omitting the contour term of the conservative
     form is the zero-total-flux condition; SurfactantEndFlux imposes a nonzero one, and at an end
     point on the symmetry axis it is inert because the point measure carries 2*pi*r.
  4. THE VARIANTS DO NOT COST THE CONSERVATION. Neither the log variable nor any stabilization can
     break it, because both are written so that the constant test function annihilates them.

Each Problem gets its own output directory: several Problems in one directory share the JIT cache,
and variants that differ only in constructor flags then silently reuse the first one's compiled code.
"""

import math
import tempfile

import pytest

from typing import Literal

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.ALE import PrescribedMovingMesh
from pyoomph.equations.generic import ElementSpace, IntegralObservables
from pyoomph.equations.surfactants import SurfactantEndFlux, SurfactantTransportEquations
from pyoomph.meshes.simplemeshes import CircularMesh


# ---------------------------------------------------------------- the prescribed-motion testbed

class _PrescribedInterface(Problem):
    """A sphere (axisymmetric) or a circle (Cartesian) whose motion is dictated, not solved for.

    ``a_m`` dilates the mesh, ``a_f`` the fluid: their difference is a normal slip, i.e. exactly what
    evaporation does. ``om_m``/``om_f`` rotate them, so their difference is a tangential slip -- the
    case a Laplace- or pseudo-elastically smoothed mesh is in permanently.
    """

    def __init__(self, *, form: Literal["conservative", "legacy", "strong", "dg_upwind"] = "conservative",
                 variable: Literal["direct", "log"] = "direct", stabilization=None, space=None,
                 a_m=0.0, a_f=0.0, om_m=0.0, om_f=0.0, D=0.0, axisymmetric=True, nref=2):
        super().__init__()
        self.form, self.variable, self.stabilization = form, variable, stabilization
        self.a_m, self.a_f, self.om_m, self.om_f = a_m, a_f, om_m, om_f
        self.D, self.axisymmetric, self.nref, self.space = D, axisymmetric, nref, space
        self.surf = None

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
        X, Y = var("lagrangian_x"), var("lagrangian_y")
        G0 = 1 + rational_num(1, 2) * (Y if self.axisymmetric else X * X - Y * Y)
        self.surf = SurfactantTransportEquations({"S": G0}, diffusivity=self.D, form=self.form,
                                                 variable=self.variable, stabilization=self.stabilization,
                                                 space=self.space,
                                                 fluid_velocity=u, interface_velocity=w, log_reference=1)
        ieqs = self.surf
        ieqs += IntegralObservables(amount=self.surf.concentration("S"), area=1)
        eqs += ieqs @ "ifc"
        if self.axisymmetric:
            eqs += DirichletBC(mesh_x=0) @ "axis"
        self += eqs @ "domain"


def _evolve(problem, nsteps=20, T=1.0):
    """Run the testbed and return the initial and final surfactant amount and the interface area."""
    problem.set_output_directory(tempfile.mkdtemp(prefix="surfactant_"))
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
    N0, A0 = float(obs["amount"]), float(obs["area"])
    for _ in range(nsteps):
        problem.solve(timestep=T / nsteps, do_not_set_IC=True)
    obs = ifc.evaluate_all_observables()
    return dict(N0=N0, N=float(obs["amount"]), A0=A0, A=float(obs["area"]))


def _drift(**kwargs):
    nsteps = kwargs.pop("nsteps", 20)
    r = _evolve(_PrescribedInterface(**kwargs), nsteps=nsteps)
    return r["N"] / r["N0"] - 1


# ---------------------------------------------------------------- 1. conservation

@pytest.mark.parametrize("case", [
    pytest.param(dict(a_m=-0.3, a_f=-0.3), id="dilatation"),
    pytest.param(dict(a_m=-0.3, a_f=0.0), id="mass_transfer"),
    pytest.param(dict(a_m=0.0, a_f=0.0, om_m=0.6, om_f=1.0, axisymmetric=False), id="tangential_slip"),
    pytest.param(dict(a_m=-0.2, a_f=0.05, om_m=-0.4, om_f=0.8, axisymmetric=False), id="everything"),
])
def test_the_conservative_form_conserves_the_surfactant_exactly(case):
    # The transient term is the derivative of the whole integral and the advection is a flux, so the
    # constant test function makes the residual a telescoping difference of the discrete amount. That
    # is a property of the discrete system, not of the time step: the floor here is the Newton
    # tolerance, not the discretization.
    assert abs(_drift(form="conservative", **case)) < 1e-10


def test_tangential_mesh_sliding_alone_breaks_the_legacy_form():
    # The sharpest case: the interface is a circle at every instant and its length does not change,
    # so there is no dilatation to get wrong. Only the nodes slide along it - which is what every
    # smoothed moving mesh does permanently - and that alone costs the legacy form 7e-4.
    case = dict(a_m=0.0, a_f=0.0, om_m=0.6, om_f=1.0, axisymmetric=False)
    assert abs(_drift(form="conservative", **case)) < 1e-10
    assert abs(_drift(form="legacy", **case)) > 1e-4


def test_the_legacy_drift_is_second_order_in_the_time_step():
    # Halving dt divides the drift by four: it is a time-stepping error, so no mesh refinement and no
    # amount of spatial resolution removes it. This is the reason the default changed.
    case = dict(form="legacy", a_m=-0.3, a_f=0.0)
    coarse = abs(_drift(nsteps=20, **case))
    fine = abs(_drift(nsteps=40, **case))
    assert coarse / fine == pytest.approx(4.0, rel=0.35)


def test_uniform_dilatation_is_exact_in_the_conservative_form():
    # With u == w there is no slip at all, so the discrete statement reduces to "Gamma times the
    # local metric is constant", which holds nodewise. The legacy form is merely second order here.
    r = _evolve(_PrescribedInterface(form="conservative", a_m=-0.3, a_f=-0.3), nsteps=20)
    assert abs(r["N"] / r["N0"] - 1) < 1e-10
    # the area really did change, so this is not a trivially satisfied test
    assert r["A"] / r["A0"] == pytest.approx(math.exp(-0.6), rel=1e-3)


# ---------------------------------------------------------------- 2. the legacy form is the old one

def test_the_legacy_form_reproduces_the_documented_legacy_numbers():
    # Pinned against the pre-refactor implementation, whose generated C code for the interface
    # element is byte-identical to what form="legacy" produces on this problem.
    assert _drift(form="legacy", a_m=-0.3, a_f=0.0, nsteps=20) == pytest.approx(1.1343e-03, rel=1e-3)
    assert _drift(form="legacy", a_m=-0.3, a_f=0.0, nsteps=40) == pytest.approx(2.8460e-04, rel=1e-3)


def test_the_legacy_form_still_defines_its_projected_advection_velocity():
    # contact_angle.py gates its contact-line constraint on this, so it has to stay observable.
    legacy = SurfactantTransportEquations("S", form="legacy")
    conservative = SurfactantTransportEquations("S", form="conservative")
    assert legacy.uses_projected_advection_velocity(legacy)
    assert not conservative.uses_projected_advection_velocity(conservative)


# ---------------------------------------------------------------- 3. the ends of the interface

class _EndFluxProblem(Problem):
    """A quarter circle with an imposed surfactant flux at one of its two end points."""

    def __init__(self, flux, *, axisymmetric=False, at_axis=False, nref=2):
        super().__init__()
        self.flux, self.axisymmetric, self.at_axis, self.nref = flux, axisymmetric, at_axis, nref

    def define_problem(self):
        self.set_coordinate_system("axisymmetric" if self.axisymmetric else cartesian)
        self += CircularMesh(radius=1, segments=["NE"], outer_interface="ifc",
                             straight_interface_name={"center_to_north": "axis",
                                                      "center_to_east": "equator"})
        eqs = PrescribedMovingMesh(vector(0, 0)) + ElementSpace("C2")
        surf = SurfactantTransportEquations({"S": 1}, diffusivity=0.01)
        ieqs = surf + IntegralObservables(amount=surf.concentration("S"), area=1)
        ieqs += SurfactantEndFlux(S=self.flux) @ ("axis" if self.at_axis else "equator")
        eqs += ieqs @ "ifc"
        self += eqs @ "domain"


def _end_flux(flux, **kw):
    p = _EndFluxProblem(flux, **kw)
    p.set_output_directory(tempfile.mkdtemp(prefix="surfactant_end_"))
    p.quiet()
    p.newton_solver_tolerance = 1e-11
    p.initialise()
    for _ in range(p.nref):
        p.refine_uniformly()
    p.set_initial_condition(ic_name="")
    ifc = p.get_mesh("domain/ifc")
    N0 = float(ifc.evaluate_all_observables()["amount"])
    for _ in range(10):
        p.solve(timestep=0.1, do_not_set_IC=True)
    return float(ifc.evaluate_all_observables()["amount"]) - N0


def test_an_interface_end_loses_nothing_unless_asked():
    # Adding no contour term *is* the zero-total-flux condition. Diffusion is on here, so this also
    # pins that the natural condition of the diffusive flux is the same one.
    assert abs(_end_flux(0.0)) < 1e-10


def test_an_imposed_end_flux_removes_exactly_what_it_says():
    assert _end_flux(0.25) == pytest.approx(-0.25, abs=1e-9)


def test_an_end_flux_on_the_symmetry_axis_is_inert():
    # In axisymmetry a point domain carries 2*pi*r, so an end point at r=0 sits on a ring of zero
    # circumference and cannot carry any flux at all. Nothing special is needed to make that happen.
    assert abs(_end_flux(0.25, axisymmetric=True, at_axis=True)) < 1e-10
    # ... whereas the same flux at the equator is the rate times the circumference there
    assert _end_flux(0.25, axisymmetric=True, at_axis=False) == pytest.approx(-0.25 * 2 * math.pi, rel=1e-6)


# ---------------------------------------------------------------- 4. the variants keep it

@pytest.mark.parametrize("variant", [
    pytest.param(dict(variable="log"), id="log_variable"),
    pytest.param(dict(stabilization="artificial"), id="artificial_diffusion"),
    pytest.param(dict(stabilization="limited"), id="limited_diffusion"),
    pytest.param(dict(variable="log", stabilization="artificial"), id="log_and_stabilized"),
])
def test_nothing_added_on_top_can_break_the_conservation(variant):
    # Every stabilization is written against grad(v) and therefore vanishes for the constant test
    # function; the log variable changes what Gamma depends on but not the structure of the two terms
    # the constant test function sees. So both are free, conservation-wise - which is the whole
    # argument for reaching for them rather than for a different transport form.
    assert abs(_drift(form="conservative", a_m=-0.3, a_f=0.0, **variant)) < 1e-10


# ---------------------------------------------------------------- 5. the upwind DG form

# These were skipped for a while: building an interface "_internal_facets_" skeleton used to segfault
# at teardown, because BulkElementBase::free_element_info read the loop bounds back out of the JIT
# code, which a nanobind-owned Python object can outlive. The element caches them now, so a destructor
# no longer depends on another object's lifetime -- see dev_docs/surfactant_transport.md section 10.

# A piecewise-constant space with an upwind numerical flux is the one formulation here that is
# genuinely bound-preserving: implicit in time it is an M-matrix system, so Gamma cannot go negative
# at all rather than merely less often. It lives on the interior facets between neighbouring
# interface elements -- a facet of a curve is a point, and the normal there is the in-surface
# conormal, so the flux is expressible without any reference to the interface's own normal.


_APEX_ZC, _APEX_WD = 0.5, 0.1


class _ApexAdvection(Problem):
    """A static unit sphere with a tangential flow converging on the north pole.

    u = c*r*(-z, r) is tangential to any sphere about the origin, converges on the pole and stalls at
    both of them, so a tanh front in Gamma is swept onto the axis and compressed there, where the ring
    area 2*pi*r goes to zero. This is where a continuous Galerkin scheme goes negative.
    """

    def __init__(self, *, form="conservative", space=None, c=1.0, nref=3):
        super().__init__()
        self.form, self.space, self.c, self.nref = form, space, c, nref

    def define_problem(self):
        self.set_coordinate_system("axisymmetric")
        self += CircularMesh(radius=1, segments=["NE", "SE"], outer_interface="ifc",
                             straight_interface_name={"center_to_north": "axis",
                                                      "center_to_south": "axis"})
        r, z = var("coordinate_x"), var("coordinate_y")
        u = self.c * r * vector(-z, r)
        eqs = PrescribedMovingMesh(vector(0, 0)) + ElementSpace("C2")
        Z = var("lagrangian_y")
        G0 = rational_num(1, 2) * (1 + tanh((_APEX_ZC - Z) / _APEX_WD))
        surf = SurfactantTransportEquations({"S": G0}, diffusivity=0, form=self.form,
                                            space=self.space, fluid_velocity=u,
                                            interface_velocity=vector(0, 0))
        G = surf.concentration("S")
        # the integral of the negative part: exactly zero iff Gamma >= 0 almost everywhere, and it
        # works for any space, unlike a nodal minimum
        eqs += (surf + IntegralObservables(amount=G, negative=-minimum(G, 0))) @ "ifc"
        eqs += DirichletBC(mesh_x=0) @ "axis"
        self += eqs @ "domain"
        self.surf = surf


def _apex(*, form, space=None, T=2.0, nsteps=240):
    p = _ApexAdvection(form=form, space=space)
    p.set_output_directory(tempfile.mkdtemp(prefix="surfactant_apex_"))
    p.quiet()
    p.newton_solver_tolerance = 1e-11
    p.initialise()
    for _ in range(p.nref):
        p.refine_uniformly()
    ifc = p.get_mesh("domain/ifc")
    for nd in ifc.nodes():
        rr = math.hypot(nd.x(0), nd.x(1))
        for i in range(2):
            nd.set_x(i, nd.x(i) / rr)
    p.get_mesh("domain").set_lagrangian_nodal_coordinates()
    p.set_initial_condition(ic_name="")
    N0 = float(ifc.evaluate_all_observables()["amount"])
    worst = 0.0
    for _ in range(nsteps):
        p.solve(timestep=T / nsteps, do_not_set_IC=True)
        worst = max(worst, float(ifc.evaluate_all_observables()["negative"]))
    return dict(N0=N0, N=float(ifc.evaluate_all_observables()["amount"]), worst_negative=worst)


@pytest.mark.parametrize("case", [
    pytest.param(dict(a_m=-0.3, a_f=-0.3), id="dilatation"),
    pytest.param(dict(a_m=-0.3, a_f=0.0), id="mass_transfer"),
    pytest.param(dict(a_m=0.0, a_f=0.0, om_m=0.6, om_f=1.0, axisymmetric=False), id="tangential_slip"),
    pytest.param(dict(a_m=-0.2, a_f=0.05, om_m=-0.4, om_f=0.8, axisymmetric=False), id="everything"),
])
def test_the_upwind_dg_form_conserves_the_surfactant_exactly(case):
    # The numerical flux is single-valued, so summing the element residuals against the constant test
    # function cancels it pairwise; together with the derivative of the whole integral for the metric
    # that leaves the same telescoping difference as the continuous conservative form.
    assert abs(_drift(form="dg_upwind", **case)) < 1e-10


def test_an_interface_mesh_has_interior_facets():
    # Nothing else in the tree puts facet terms on an interface, so this pins the enumeration itself:
    # a closed curve of N elements has N facets, and the two ends of an open one are excluded because
    # exterior boundary facets are not part of the skeleton.
    closed = _PrescribedInterface(form="dg_upwind", axisymmetric=False)
    _evolve(closed, nsteps=1)
    ifc = closed.get_mesh("domain/ifc")
    skel = closed.get_mesh("domain/ifc/_internal_facets_")
    assert skel.nelement() == ifc.nelement()

    openarc = _PrescribedInterface(form="dg_upwind", axisymmetric=True)   # pole to pole, two free ends
    _evolve(openarc, nsteps=1)
    ifc = openarc.get_mesh("domain/ifc")
    skel = openarc.get_mesh("domain/ifc/_internal_facets_")
    assert skel.nelement() == ifc.nelement() - 1


def test_the_upwind_dg_form_keeps_gamma_non_negative():
    # The apex problem: a front swept onto the symmetry axis, where the ring area 2*pi*r goes to zero.
    # Every continuous form undershoots badly here; this one cannot.
    cg = _apex(form="conservative")
    dg = _apex(form="dg_upwind")
    assert cg["worst_negative"] > 1e-2        # the continuous form really does go negative
    assert dg["worst_negative"] == 0.0        # and this one never does, at any step
    assert abs(dg["N"] / dg["N0"] - 1) < 1e-10


def test_a_dg_surfactant_needs_a_discontinuous_space():
    with pytest.raises(ValueError, match="discontinuous space"):
        SurfactantTransportEquations("S", form="dg_upwind", space="C2")


def test_dg_on_a_three_dimensional_interface_is_refused():
    # pyoomph can only enumerate interior facets of a one-dimensional interface. A 2d interface (a
    # surface in 3d) throws deep inside the mesh layer; this turns it into a message about surfactants.
    surf = SurfactantTransportEquations("S", form="dg_upwind")

    class _FakeHost:
        def get_element_dimension(self):
            return 2

    with pytest.raises(RuntimeError, match="one-dimensional interface"):
        surf._check_dg_is_available(_FakeHost())   # type:ignore[arg-type]


class _StaticDiffusion(Problem):
    """A static circle, no flow at all, and a mode-2 initial profile.

    Pure diffusion has an exact invariant: the variance about the mean decays as exp(-2*D*k^2*t) for
    mode k, while the amount is untouched. That is what pins the diffusion operator without needing a
    pointwise solution, and it works for any space.
    """

    def __init__(self, *, form="conservative", space=None, D=0.05, nref=2):
        super().__init__()
        self.form, self.space, self.D, self.nref = form, space, D, nref

    def define_problem(self):
        self.set_coordinate_system(cartesian)
        self += CircularMesh(radius=1, outer_interface="ifc")
        eqs = PrescribedMovingMesh(vector(0, 0)) + ElementSpace("C2")
        X, Y = var("lagrangian_x"), var("lagrangian_y")
        G0 = 1 + rational_num(1, 2) * (X * X - Y * Y)          # cos(2*theta) on the unit circle
        surf = SurfactantTransportEquations({"S": G0}, diffusivity=self.D, form=self.form,
                                            space=self.space, fluid_velocity=vector(0, 0),
                                            interface_velocity=vector(0, 0))
        G = surf.concentration("S")
        eqs += (surf + IntegralObservables(amount=G, sq=G ** 2, length=1)) @ "ifc"
        self += eqs @ "domain"


def _variance_decay(*, form, space=None, D=0.05, T=0.5, nsteps=50):
    p = _StaticDiffusion(form=form, space=space, D=D)
    p.set_output_directory(tempfile.mkdtemp(prefix="surfactant_diff_"))
    p.quiet()
    p.newton_solver_tolerance = 1e-11
    p.initialise()
    for _ in range(p.nref):
        p.refine_uniformly()
    ifc = p.get_mesh("domain/ifc")
    for nd in ifc.nodes():
        r = math.hypot(nd.x(0), nd.x(1))
        for i in range(2):
            nd.set_x(i, nd.x(i) / r)
    p.get_mesh("domain").set_lagrangian_nodal_coordinates()
    p.set_initial_condition(ic_name="")

    def variance():
        o = ifc.evaluate_all_observables()
        N, S, L = float(o["amount"]), float(o["sq"]), float(o["length"])
        return N, S - N * N / L

    N0, v0 = variance()
    for _ in range(nsteps):
        p.solve(timestep=T / nsteps, do_not_set_IC=True)
    N, v = variance()
    return dict(ratio=v / v0, drift=N / N0 - 1)


@pytest.mark.parametrize("form,space,tol", [
    pytest.param("conservative", None, 1e-6, id="continuous"),
    pytest.param("dg_upwind", "D0", 5e-3, id="dg_D0"),
    pytest.param("dg_upwind", "D1", 5e-3, id="dg_D1"),
])
def test_surface_diffusion_decays_the_variance_at_the_right_rate(form, space, tol):
    # On a discontinuous space the element term couples nothing across a facet, and at order 0 it is
    # identically zero - a diffusivity used to be accepted and silently ignored. The interior penalty
    # is what carries the diffusion there, and at order 0 it *is* the two-point flux rather than a
    # stabilization, hence first-order accurate rather than merely present.
    exact = math.exp(-2 * 0.05 * 2 ** 2 * 0.5)
    r = _variance_decay(form=form, space=space)
    assert r["ratio"] == pytest.approx(exact, abs=tol)
    assert abs(r["drift"]) < 1e-10      # diffusion is a flux, so it moves nothing in total


def test_a_dg_surfactant_accepts_the_DL_space():
    # DL was refused for a while because an InitialCondition on it kept only the constant mode; that
    # was a bug in Mesh::setup_initial_conditions, now fixed, so the space is usable again. It still
    # needs a limiter to be bounded, which is why D0 remains the default.
    SurfactantTransportEquations("S", form="dg_upwind", space="DL")


# ---------------------------------------------------------------- 6. scale invariance

# Every other test in this file is nondimensional, with every scale left at 1. That makes them blind
# to a whole class of error: a test scale in pyoomph gains a factor 1/spatial per domain level, so a
# term whose test function belongs to another domain -- the end-point flux, the DG facet flux -- can
# carry a spurious power of the spatial scale and still look perfect at scale 1. These tests solve the
# *same physical problem* nondimensionalised three different ways and require the same answer.
#
# add_interior_facet_residual makes this worse by not running the unit check at all, so a facet term
# has nothing but a test like this standing behind it.

from pyoomph.expressions.units import meter, milli, micro, mol, nano, second

_R0 = 1 * milli * meter
_G0 = 1 * micro * mol / meter ** 2
_T = 1 * second


class _ScaledEndFlux(Problem):
    """A quarter circle of fixed physical size, nondimensionalised by a variable spatial scale."""

    def __init__(self, Lscale, flux, nref=2):
        super().__init__()
        self.Lscale, self.flux, self.nref = Lscale, flux, nref

    def define_problem(self):
        self.set_coordinate_system(cartesian)
        self += CircularMesh(radius=_R0, segments=["NE"], outer_interface="ifc",
                             straight_interface_name={"center_to_north": "axis",
                                                      "center_to_east": "equator"})
        self.set_scaling(spatial=self.Lscale, temporal=_T, surface_concentration=_G0)
        eqs = PrescribedMovingMesh(vector(0, 0)) + ElementSpace("C2")
        surf = SurfactantTransportEquations({"S": _G0}, diffusivity=0)
        ieqs = surf + IntegralObservables(amount=surf.concentration("S"))
        ieqs += SurfactantEndFlux(S=self.flux) @ "equator"
        eqs += ieqs @ "ifc"
        self += eqs @ "domain"


@pytest.mark.parametrize("Lscale", [
    pytest.param(_R0, id="spatial_is_the_radius"),
    pytest.param(1 * meter, id="spatial_is_a_metre"),
    pytest.param(10 * milli * meter, id="spatial_is_a_centimetre"),
])
def test_an_imposed_end_flux_does_not_depend_on_the_spatial_scale(Lscale):
    # The flux is per unit length of the end point even here, where that "length" is a point with a
    # dimensionless measure of 1: the interface test function supplies the length the measure does
    # not. Pass a rate in mol/s instead and a dimensional problem rejects it on units.
    q = 0.25 * nano * mol / (meter * second)
    p = _ScaledEndFlux(Lscale, q)
    p.set_output_directory(tempfile.mkdtemp(prefix="surfactant_scale_"))
    p.quiet()
    p.newton_solver_tolerance = 1e-11
    p.initialise()
    for _ in range(p.nref):
        p.refine_uniformly()
    p.set_initial_condition(ic_name="")
    ifc = p.get_mesh("domain/ifc")
    unit = nano * mol / meter          # an interface is a curve in 2d, so the amount is per depth
    N0 = float(ifc.evaluate_all_observables()["amount"] / unit)
    for _ in range(10):
        p.solve(timestep=_T / 10, do_not_set_IC=True)
    N = float(ifc.evaluate_all_observables()["amount"] / unit)
    assert N - N0 == pytest.approx(-0.25, abs=1e-9)
