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

"""Worker for the *physics* tests of AxisymmetricReconnection -- one Problem per process, since a
second one segfaults in the JIT loader.

The kinematic suites (tests/axisymm_reconnection_worker.py) prescribe the interface and ask what the
detection, the surgery plan and the rebuilt geometry do with it. Here the interface is instead moved
by surface tension through :py:class:`~pyoomph.equations.navier_stokes.NavierStokesFreeSurface` on a
mesh that follows it, i.e. by the flow the feature is meant to be used with. What that adds over the
kinematic tests is everything the event has to survive rather than produce:

* the transfer has real velocity and pressure fields to carry, not a projected ramp;
* the time stepper's history has to remain usable across the remesh, or the next step diverges;
* the volume is no longer a property of a prescribed profile but the outcome of the free-surface
  kinematic condition, so it drifts on *every* step and the surgery's share of that drift has to be
  separated out. That is what ``PYOOMPH_STEP`` records exist for: one per accepted step, carrying the
  volume and whether a remesh happened in it.

Everything is nondimensionalised with the unperturbed radius R0, the density rho and the surface
tension sigma, so lengths are in R0, times in the inertio-capillary time sqrt(rho*R0^3/sigma) and the
viscosity is the Ohnesorge number.
"""

import argparse
import json
import sys
import traceback

import numpy

from pyoomph import Problem, DirichletBC, InitialCondition, var
from pyoomph.equations.ALE import (ConnectMeshAtInterface, HyperelasticSmoothedMesh,
                                   LaplaceSmoothedMesh)
from pyoomph.equations.generic import (AxisymmetryBC, ExtremumObservables, IntegralObservables,
                                       RemeshWhen, RemeshingOptions)
from pyoomph.equations.navier_stokes import (ConnectVelocityAtInterface, NavierStokesEquations,
                                             NavierStokesFreeSurface)
from pyoomph.equations.topological_changes import (AxisymmetricReconnection,
                                                   TopologicalChangesGmshTemplate)
from pyoomph.meshes.zeta import (AssignZetaCoordinatesByArclength,
                                 AssignZetaCoordinatesByEulerianCoordinate)
from pyoomph.output.meshio import MeshFileOutput


def emit(_kind, **kw):
    print("PYOOMPH_" + _kind + " " + json.dumps(kw, sort_keys=True), flush=True)


# --------------------------------------------------------------------------------------
# The mesh templates
# --------------------------------------------------------------------------------------

class _FreeSurfaceMeshBase(TopologicalChangesGmshTemplate):
    """Shared part of the two geometries: the mesh size fields and the remeshing branch.

    The remeshing branch is written once and serves a reconnection and a plain quality remesh alike -
    it never asks which of the two it is in, because ``get_reconnected_boundaries`` fills in the same
    structure either way. It is also the only branch that has to cope with a *variable number of
    fragments*: after the pinch the loop below simply runs twice.
    """

    #: Boundary name of the free surface and of this domain's share of the symmetry axis.
    interface_name = "interface"
    axis_name = "axisymm"

    def _size_fields(self, interface_curves, axis_curves):
        """Resolve the local radius, on the interface and on the axis facing it.

        Verbatim the construction of the Rayleigh-Plateau and beads-on-string tutorials. It is what
        makes a neck of radius r be crossed by a couple of elements however small r gets - a mesh
        uniform on the scale of the drop has none at all there, and then the pinch is a numerical
        accident rather than a detected event.
        """
        pr = self.get_problem()
        at_interface = self.add_mesh_size_field("MathEval", F="x/" + str(pr.elements_per_radius))
        restr_interface = self.add_mesh_size_field("Restrict", InField=at_interface,
                                                   CurvesList=list(interface_curves))
        to_interface = self.add_mesh_size_field("Distance", CurvesList=list(interface_curves),
                                                Sampling=400)
        at_axis = self.add_mesh_size_field(
            "MathEval", F="F" + str(to_interface) + "/" + str(0.75 * pr.elements_per_radius))
        restr_axis = self.add_mesh_size_field("Restrict", InField=at_axis, CurvesList=list(axis_curves))
        self.set_mesh_size_background_field(
            self.add_mesh_size_field("Min", FieldsList=[restr_interface, restr_axis]))

    #: Number of times define_geometry has rebuilt this template. A step that raised it contained a
    #: remesh, which is what separates the surgery's volume jump from the ordinary drift.
    _remesh_counter = 0
    last_report: dict = {}

    def _note_remesh(self, rb):
        self.last_report = _report_of(rb)
        self._remesh_counter += 1
        emit("REMESH_REPORT", **self.last_report)

    def _common_setup(self):
        pr = self.get_problem()
        self.mesh_mode = "tris"
        self.set_gmsh_parameter("Mesh.MeshSizeMin", pr.hmin)
        self.set_gmsh_parameter("Mesh.MeshSizeMax", pr.hmax)
        self.set_gmsh_parameter("Mesh.MeshSizeFromCurvature", 4 * pr.elements_per_radius)
        # Single-threaded meshing, so that the run is reproducible. Gmsh parallelises its 2d meshing,
        # and the mesh it returns then differs from run to run in the last bits; on a transient that
        # ends in a capillary singularity those bits separate within a few dozen steps, and two runs
        # of the same script take visibly different paths through the pinch. A test that is allowed to
        # do that is not a test.
        self.set_gmsh_parameter("General.NumThreads", 1)


class JetMesh(_FreeSurfaceMeshBase):
    """One full Rayleigh-Plateau wavelength of a liquid column.

    ``bottom`` (z=0) and ``top`` (z=L) are symmetry planes of the mode, so they stay put and the
    interface meets them at right angles. After the pinch each of the two fragments still owns one of
    them, i.e. every chain has one ``"fixed"`` end and one fresh ``"axis"`` cap.
    """

    def define_geometry(self):
        self._common_setup()
        pr = self.get_problem()
        L = pr.L
        if self.is_first_time():
            zs = numpy.linspace(0.0, L, pr.interface_spline_points)
            pts = [self.point(float(pr.initial_radius(z)), float(z)) for z in zs]
            iface = self.spline(pts, name=self.interface_name)
            p_bot, p_top = self.point(0.0, 0.0), self.point(0.0, L)
            axis = self.create_lines(pts[0], "bottom", p_bot, self.axis_name, p_top, "top", pts[-1])[1]
            self.plane_surface("bottom", self.axis_name, "top", self.interface_name, name="liquid")
            self._size_fields([iface], [axis])
            return

        rb = self.get_reconnected_boundaries("liquid/" + self.interface_name,
                                             "liquid/" + self.axis_name)
        self._note_remesh(rb)
        ifaces = [self.spline_from_chain(ch, self.interface_name) for ch in rb.interface_chains]
        axes = self.lines_from_axis_segments(rb.axis_segments, self.axis_name)
        # The two ends of the domain are horizontal walls between the axis and the interface. Which
        # chain end they belong to follows from the end type: a "fixed" end is by definition one that
        # is not on the axis, and in this geometry the only such ends are the two symmetry planes.
        for ch in rb.interface_chains:
            for end, kind in ((0, ch.end_types[0]), (-1, ch.end_types[1])):
                if kind != "fixed":
                    continue
                x, y = ch.points[end]
                self.line(self.point(0.0, float(y)), self.point(float(x), float(y)),
                          name="bottom" if float(y) < 0.5 * L else "top")
        self.plane_surface("bottom", self.axis_name, "top", self.interface_name, name="liquid")
        self._size_fields(ifaces, axes)


class DropletsMesh(_FreeSurfaceMeshBase):
    """Free droplets on the axis: every chain runs from one axial tip to the other, no walls at all.

    Before the merge this is two disjoint bulk fragments in one domain, which
    ``plane_surface`` builds from two curve loops in one call; afterwards it is one.
    """

    def define_geometry(self):
        self._common_setup()
        pr = self.get_problem()
        if self.is_first_time():
            ifaces, axes = [], []
            for zc, R in pr.droplets:
                pts = _sphere_points(zc, R, pr.interface_spline_points)
                gpts = [self.point(float(x), float(y)) for x, y in pts]
                ifaces.append(self.spline(gpts, name=self.interface_name))
                axes.append(self.line(self.point(0.0, float(zc - R)), self.point(0.0, float(zc + R)),
                                      name=self.axis_name))
            self.plane_surface(self.axis_name, self.interface_name, name="liquid")
            self._size_fields(ifaces, axes)
            return

        rb = self.get_reconnected_boundaries("liquid/" + self.interface_name,
                                             "liquid/" + self.axis_name)
        self._note_remesh(rb)
        ifaces = [self.spline_from_chain(ch, self.interface_name) for ch in rb.interface_chains]
        axes = self.lines_from_axis_segments(rb.axis_segments, self.axis_name)
        self.plane_surface(self.axis_name, self.interface_name, name="liquid")
        self._size_fields(ifaces, axes)


class TwoPhaseDropletsMesh(_FreeSurfaceMeshBase):
    """Droplets on the axis inside a gas box, both domains rebuilt from this one template.

    The gas owns the rest of the axis (``gas_axis``) and the box walls (``outer``); the interface is a
    boundary of both domains, so the merge has to leave the two meshes conforming along it. The gas
    side is the one that changes topology hardest: the coalescence takes a stretch of axis away from
    it, and its remaining nodes there have no counterpart on the old gas boundary.
    """

    interface_name = "interface"
    axis_name = "axis"

    def define_geometry(self):
        self._common_setup()
        pr = self.get_problem()
        Rbox, Zbox = pr.gas_box

        if self.is_first_time():
            ifaces, axes, liquid_spans = [], [], []
            for zc, R in pr.droplets:
                pts = _sphere_points(zc, R, pr.interface_spline_points)
                ifaces.append(self.spline([self.point(float(x), float(y)) for x, y in pts],
                                          name=self.interface_name))
                axes.append(self.line(self.point(0.0, float(zc - R)), self.point(0.0, float(zc + R)),
                                      name=self.axis_name))
                liquid_spans.append((float(zc - R), float(zc + R)))
            gas_axis = _complement_spans(liquid_spans, -Zbox, Zbox)
        else:
            rb = self.get_reconnected_boundaries("liquid/" + self.interface_name,
                                                 "liquid/" + self.axis_name, "gas/gas_axis")
            self._note_remesh(rb)
            ifaces = [self.spline_from_chain(ch, self.interface_name) for ch in rb.interface_chains]
            axes = self.lines_from_axis_segments(rb.axis_segments, self.axis_name)
            # Everything the liquid does not cover. After the merge the gap between the two droplets
            # is gone from this list without the caller having done anything about it.
            gas_axis = [((0.0, float(a[1])), (0.0, float(b[1]))) for a, b in rb.opposite_axis_segments]

        gas_axis_curves = self.lines_from_axis_segments(gas_axis, "gas_axis")
        self.plane_surface(self.axis_name, self.interface_name, name="liquid")
        # The corners on the axis are written here as literals and arrive from the mesh in
        # lines_from_axis_segments; they only become one Gmsh point because
        # TopologicalChangesGmshTemplate.point() merges near-coincident points.
        corners = [self.point(0.0, -Zbox), self.point(Rbox, -Zbox), self.point(Rbox, Zbox),
                   self.point(0.0, Zbox)]
        for a, b in zip(corners, corners[1:]):
            self.line(a, b, name="outer")
        self.plane_surface(self.interface_name, "gas_axis", "outer", name="gas")
        self._size_fields(ifaces, axes + gas_axis_curves)


def _complement_spans(spans, lo, hi):
    """The pieces of ``[lo, hi]`` that ``spans`` does not cover, as point pairs on the axis."""
    out, cur = [], lo
    for a, b in sorted(spans):
        if a > cur:
            out.append(((0.0, cur), (0.0, a)))
        cur = max(cur, b)
    if cur < hi:
        out.append(((0.0, cur), (0.0, hi)))
    return out


def _report_of(rb):
    return {"has_plan": rb.has_plan,
            "events": [(e.kind, float(e.z_center)) for e in rb.events],
            "n_chains": len(rb.interface_chains),
            "fragment_volumes": [float(v) for v in rb.fragment_volumes],
            "axis_spans": [(float(a[1]), float(b[1])) for a, b in rb.axis_segments]}


# --------------------------------------------------------------------------------------
# Prescribed initial half-sections
# --------------------------------------------------------------------------------------

def _tip_clustered(n):
    """Samples in [-1,1] bunched at the ends; an axial tip has r ~ sqrt(z_tip-z) and a uniform
    sample puts its first point off the tip at a radius of order sqrt(h)."""
    s = numpy.linspace(-1.0, 1.0, n)
    return numpy.sign(s) * (1.0 - (1.0 - numpy.abs(s)) ** 2)


def _sphere_points(zc, R, n):
    t = _tip_clustered(n)
    return numpy.stack([R * numpy.sqrt(numpy.maximum(0.0, 1.0 - t * t)), zc + R * t], axis=1)


# --------------------------------------------------------------------------------------
# The problems
# --------------------------------------------------------------------------------------

class _FreeSurfaceProblemBase(Problem):
    """The parts every scenario shares: nondimensionalisation, mesh motion, the free surface and the
    bookkeeping the tests read."""

    def __init__(self):
        super().__init__()
        self.Oh = 0.1                     # Ohnesorge number, i.e. the nondimensional viscosity
        self.elements_per_radius = 2.5
        self.hmin = 0.04
        self.hmax = 0.35
        self.interface_spline_points = 61
        self.rmin = None                  # pinch-off threshold, in R0
        self.distmin = None               # coalescence threshold, in R0
        self.mesh_motion = "hyperelastic"
        self.reconnection = None
        self.mesh_template = None
        self._steps = []

    def _bulk_equations(self):
        eqs = NavierStokesEquations(mass_density=1, dynamic_viscosity=self.Oh)
        eqs += HyperelasticSmoothedMesh() if self.mesh_motion == "hyperelastic" else LaplaceSmoothedMesh()
        eqs += RemeshWhen(RemeshingOptions())
        eqs += IntegralObservables(volume=1)
        eqs += MeshFileOutput()
        return eqs

    def _interface_equations(self):
        self.reconnection = AxisymmetricReconnection(rmin=self.rmin, distmin=self.distmin)
        eqs = NavierStokesFreeSurface(surface_tension=1)
        eqs += self.reconnection
        eqs += ExtremumObservables(r=var("mesh_x"))
        # An arclength chart of the interface is what a user of the free-surface equations would put
        # there anyway; the handler takes it over for the one remesh that carries an event and hands
        # it back afterwards, which is exactly the interaction worth having under test.
        eqs += AssignZetaCoordinatesByArclength(sort_along_axis="y+")
        return eqs

    # -- observables the tests read ----------------------------------------------------------------

    def liquid_volume(self):
        return float(self.get_mesh("liquid").evaluate_observable("volume"))

    def minimum_radius(self):
        return float(self.get_mesh("liquid/" + self._interface_boundary())
                     .evaluate_minimum("r", dimensional=False, as_float=True))

    def _interface_boundary(self):
        return self.mesh_template.interface_name

    def fragment_count(self):
        return len(self.interface_segments())

    def min_nodal_radius(self):
        """The most negative (or least positive) nodal radius on the interface, with its z.

        Separate from the ``ExtremumObservables`` minimum, which is evaluated inside the elements:
        a node at r<0 is a mesh that has crossed the axis, while a mere sampling point at r<0 is a
        curved element edge bulging past it.
        """
        best = None
        for seg in self.interface_segments():
            for x, y in seg:
                if best is None or x < best[0]:
                    best = (x, y)
        return best if best is not None else (0.0, 0.0)

    def tip_gap(self):
        """Smallest axial gap between two consecutive fragments, or 0 with fewer than two."""
        segs = self.interface_segments()
        if len(segs) < 2:
            return 0.0
        zs = [(min(p[1] for p in s), max(p[1] for p in s)) for s in segs]
        zs.sort()
        return min(hi[0] - lo[1] for lo, hi in zip(zs, zs[1:]))

    def interface_segments(self):
        from pyoomph.meshes.meshdatacache import MeshDataCache
        from pyoomph.meshes.ordering import sort_line_segments
        mesh = self.get_mesh("liquid/" + self._interface_boundary())
        data = MeshDataCache(tesselate_tri=False, nondimensional=True).get_data(mesh)
        segs, _ = data.get_interface_line_segments()
        pts = data.get_coordinates()
        segs = sort_line_segments(pts, segs, sort_along_axis="y+", whom="interface_segments")
        return [[(float(pts[0, i]), float(pts[1, i])) for i in s] for s in segs]

    def worst_element(self, domain="liquid"):
        """The most degenerate signed triangle area of a domain, scaled by the largest one.

        Ordering-agnostic: whatever the first three nodes of an element are, a mesh without inverted
        or collapsed elements gives them all the same sign.
        """
        mesh = self.get_mesh(domain)
        areas = []
        for e in mesh.elements():
            p = [(e.node_pt(i).x(0), e.node_pt(i).x(1)) for i in range(3)]
            areas.append(0.5 * ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                                - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1])))
        scale = max(abs(a) for a in areas)
        return (min(a / scale for a in areas), sum(1 for a in areas if a < 0), len(areas))


class RayleighPlateauProblem(_FreeSurfaceProblemBase):
    """One wavelength of a perturbed liquid column, seeded at the fastest-growing mode.

    ``k*R0 = 0.697`` is the inviscid maximum of the Rayleigh dispersion relation, so the domain is
    one wavelength ``L = 2*pi/k`` long, with the perturbation ``r = 1 + a*cos(k z)``: bulges at both
    symmetry planes and the neck in the middle, i.e. as far from the two fixed ends as the geometry
    allows. (The tutorial ``docs/source/tutorial/ale/rayleigh_plateau.py`` uses half of this, which
    puts the neck *on* the top boundary - a pinch there is a different problem and the surgery
    refuses it.)
    """

    def __init__(self):
        super().__init__()
        self.k = 0.697
        self.L = 2 * numpy.pi / self.k
        self.amplitude = 0.5
        self.rmin = 0.08

    def initial_radius(self, z):
        return 1.0 + self.amplitude * numpy.cos(self.k * z)

    def define_problem(self):
        self.set_coordinate_system("axisymmetric")
        self.mesh_template = JetMesh()
        self.add_mesh(self.mesh_template)
        eqs = self._bulk_equations()
        eqs += AxisymmetryBC() @ "axisymm"
        eqs += DirichletBC(mesh_y=True, velocity_y=0) @ ["top", "bottom"]
        eqs += self._interface_equations() @ "interface"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("y") @ "axisymm"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("x") @ "top"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("x") @ "bottom"
        self.add_equations(eqs @ "liquid")


class CoalescenceProblem(_FreeSurfaceProblemBase):
    """Two equal droplets flying towards each other on the axis until they touch.

    There is no ambient phase, so the approach is ballistic - each droplet is given a uniform initial
    velocity and translates rigidly until surface tension takes over at the merge. That is on purpose:
    what is under test is the coalescence event and the continuation past it, and a driving mechanism
    with a force balance of its own would only add something else that can fail.
    """

    def __init__(self):
        super().__init__()
        self.radius = 0.8
        self.gap = 0.5
        self.approach_speed = 0.25
        #: Step size while the droplets approach. The approach itself is a rigid translation and any
        #: step resolves it; what needs resolving is the start, where the velocity field is switched
        #: on impulsively.
        self.approach_dt = 0.25
        self.distmin = 0.15
        self.elements_per_radius = 2.5
        self.hmin = 0.06
        self.hmax = 0.25

    @property
    def droplets(self):
        zc = self.radius + 0.5 * self.gap
        return [(-zc, self.radius), (zc, self.radius)]

    def define_problem(self):
        self.set_coordinate_system("axisymmetric")
        self.mesh_template = DropletsMesh()
        self.add_mesh(self.mesh_template)
        eqs = self._bulk_equations()
        # Towards z=0 from both sides. The sign follows the Lagrangian coordinate rather than the
        # Eulerian one, so it keeps meaning the same thing once the mesh has moved.
        eqs += InitialCondition(velocity_y=-self.approach_speed * _sign_expr())
        eqs += AxisymmetryBC() @ "axisymm"
        eqs += self._interface_equations() @ "interface"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("y") @ "axisymm"
        self.add_equations(eqs @ "liquid")


class TwoPhaseCoalescenceProblem(CoalescenceProblem):
    """The coalescence of :py:class:`CoalescenceProblem`, with a gas box around it.

    Coalescence rather than a pinch, for two reasons. It is by far the cheaper of the two - the merge
    is reached in a handful of steps, where a capillary pinch needs the neck to collapse through a
    decade - and a dumbbell short enough to be affordable with a gas domain around it is *Rayleigh
    stable*: at ``neck = 0.25``, ``half length = 1.5`` the neck stopped thinning at 0.22 and sat there
    for 400 steps, which is a correct answer to a badly posed question. What the two-phase case is
    here to check is the mesh side of an event - two domains from one template, conforming across the
    interface, and the axis changing hands - and a coalescence exercises exactly that: the gas *loses*
    the stretch of axis between the two droplets.
    """

    def __init__(self):
        super().__init__()
        self.gas_box = (1.2, 2.4)
        # Coarser than the single-phase case: with a gas domain around it the same resolution puts the
        # post-merge problem just under 20000 dofs, and nothing here needs that.
        self.hmin = 0.07
        self.elements_per_radius = 2.2
        # The gas starts at rest while the liquid starts moving, so the first step is an impulsive
        # start that the connected velocity has to absorb. At the single-phase step size it cost 2e-3
        # of the liquid volume in that one step - a property of this initial condition, not of the
        # surgery, but one that would sit in the middle of the volume budget the test reads.
        self.approach_dt = 0.0625
        self.viscosity_ratio = 0.05      # gas viscosity / liquid viscosity
        self.density_ratio = 0.02

    def define_problem(self):
        self.set_coordinate_system("axisymmetric")
        self.mesh_template = TwoPhaseDropletsMesh()
        self.add_mesh(self.mesh_template)

        liq = self._bulk_equations()
        liq += InitialCondition(velocity_y=-self.approach_speed * _sign_expr())
        liq += AxisymmetryBC() @ "axis"
        liq += self._interface_equations() @ "interface"
        # Without these two the gas mesh is not attached to the interface at all: the two domains of a
        # Gmsh template share the curve, not the nodes, so the liquid's kinematic condition moves the
        # liquid side and the gas side floats. It showed up as the two interfaces drifting apart by
        # 0.07 R0 within a few steps, and then as a Gmsh loop that could not be closed at the next
        # remesh, because the gas axis no longer met the liquid tip.
        liq += ConnectMeshAtInterface() @ "interface"
        liq += ConnectVelocityAtInterface() @ "interface"
        liq += AssignZetaCoordinatesByEulerianCoordinate("y") @ "axis"
        self.add_equations(liq @ "liquid")

        gas = NavierStokesEquations(mass_density=self.density_ratio,
                                    dynamic_viscosity=self.viscosity_ratio * self.Oh)
        gas += HyperelasticSmoothedMesh()
        gas += RemeshWhen(RemeshingOptions())
        gas += IntegralObservables(volume=1)
        gas += MeshFileOutput()
        gas += AxisymmetryBC() @ "gas_axis"
        # The box keeps its shape but is open: the natural zero-traction condition lets the gas flow
        # through it and supplies the pressure datum. A closed box would have to keep its gas volume
        # exactly, i.e. would turn the liquid's O(1e-5) volume drift into an inconsistency.
        gas += DirichletBC(mesh_x=True, mesh_y=True) @ "outer"
        gas += AssignZetaCoordinatesByEulerianCoordinate("y") @ "gas_axis"
        self.add_equations(gas @ "gas")


def _sign_expr():
    from pyoomph.expressions import signum
    return signum(var("lagrangian_y"))


# --------------------------------------------------------------------------------------
# Running and reporting
# --------------------------------------------------------------------------------------

def setup(problem, outdir):
    problem.set_output_directory(outdir)
    problem.quiet()
    problem.initialise()


class StepRecorder:
    """One ``PYOOMPH_STEP`` record per accepted step, with the volume and whether a remesh happened.

    Separating the surgery's volume jump from the ordinary per-step drift of the kinematic boundary
    condition is the whole point of recording it per step rather than only at the two ends: both are
    a fraction of a percent and only their attribution says whether the feature conserves volume.
    """

    def __init__(self, problem, extra_domains=()):
        self.problem = problem
        self.extra_domains = list(extra_domains)
        self.rows = []
        self.remeshes = 0

    def _volumes(self):
        out = {"volume": self.problem.liquid_volume()}
        for d in self.extra_domains:
            out["volume_" + d] = float(self.problem.get_mesh(d).evaluate_observable("volume"))
        return out

    def record(self, tag=""):
        n = self.problem.mesh_template._remesh_counter
        row = {"t": float(self.problem.get_current_time(dimensional=False)),
               "n_fragments": self.problem.fragment_count(),
               "r_min": self.problem.minimum_radius(),
               "gap": self.problem.tip_gap(),
               "min_nodal_r": self.problem.min_nodal_radius(),
               "remeshed": bool(n != self.remeshes), "tag": tag,
               # Only meaningful on a step that remeshed: whether that remesh carried a surgery plan
               # or was an ordinary quality remesh. The three kinds of step lose volume for three
               # different reasons and the test has to keep them apart.
               "event_remesh": bool(n != self.remeshes
                                    and self.problem.mesh_template.last_report.get("has_plan"))}
        row.update(self._volumes())
        worst, n_neg, n_el = self.problem.worst_element()
        row["min_scaled_area"] = worst
        row["n_negative"] = n_neg
        row["n_elements"] = n_el
        for d in self.extra_domains:
            worst, n_neg, n_el = self.problem.worst_element(d)
            row["min_scaled_area_" + d] = worst
            row["n_negative_" + d] = n_neg
            row["n_elements_" + d] = n_el
        self.remeshes = n
        self.rows.append(row)
        emit("STEP", **row)
        return row


def _dof_count(problem):
    return int(problem.ndof())


# --------------------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------------------

def case_rayleigh_plateau(outdir, args):
    p = RayleighPlateauProblem()
    p.Oh = args.Oh
    if args.amplitude is not None:
        p.amplitude = args.amplitude
    if args.hmin is not None:
        p.hmin = args.hmin
    setup(p, outdir)
    p.DTSF_max_increase_factor = 1.25
    p.DTSF_min_decrease_factor = 0.75
    rec = StepRecorder(p)
    emit("SETUP", ndof=_dof_count(p), L=p.L, k=p.k, rmin=p.rmin, hmin=p.hmin, Oh=p.Oh)
    rec.record("initial")

    pinched_at = None
    fragments = 1
    t = 0.0
    for step in range(args.max_steps):
        r = p.minimum_radius()
        # dt proportional to the neck radius: the inertial collapse is r ~ (t0-t)^(2/3), so a step
        # that is a fixed fraction of r never jumps over the event.
        dt = min(args.max_dt, max(args.min_dt, args.dt_factor * r))
        t += dt
        # The step size is chosen by the rule above; temporal adaptivity is left on only for the
        # rejection it can fall back on. (Its estimator reports 1e-12 on every step of this problem,
        # so it never picks a step size here.) Without it the post-event phase dies immediately rather
        # than occasionally - see the restart below, which is the other half of that.
        p.run(t, outstep=False, maxstep=dt, temporal_error=1, do_not_set_IC=True)
        row = rec.record()
        # Every event, not only the first: a satellite can pinch again, and the restart below is owed
        # to each of them. Without that, a later event continues on BDF2 through an inherited history
        # and the run dies there - which is what it did while this was `pinched_at is None`.
        if row["n_fragments"] != fragments:
            fragments = row["n_fragments"]
            if pinched_at is None:
                pinched_at = len(rec.rows) - 1
                emit("EVENT", step=len(rec.rows) - 1, t=row["t"], n_fragments=row["n_fragments"])
            if args.restart_mode != "none":
                # Restart the time stepper at the event. A node the surgery created has no history:
                # the transfer gives it whatever the old mesh had at that place, which for a fresh
                # cap is the middle of a neck that was collapsing at the largest velocity anywhere in
                # the domain, and BDF2 extrapolates through that on the next step.
                #
                # The two ways of not doing that are measured against each other in section 8.4 of
                # dev_docs/axisymmetric_topological_changes.md. One step with BDF1 weights is the
                # accurate one - the history slots are shifted at the start of every step, so the
                # inherited history only ever reaches the scheme as its second level, which BDF1
                # ignores - and it is about 270 times more accurate at the same step size. The
                # impulsive flattening is the ROBUST one, and it is what this scenario needs: its
                # post-event phase is a cap retraction that the mesh only just resolves, and with the
                # BDF1 restart a cap node is carried across the symmetry axis a dozen steps later
                # (20 post-event steps with `impulsive`, a folded mesh with `bdf1`). The extra
                # dissipation that makes it inaccurate is exactly what keeps that from happening.
                if args.restart_mode == "impulsive":
                    p.assign_initial_values_impulsive()
                else:
                    p.timestepper.set_num_unsteady_steps_done(0)
            if args.no_post_remesh:
                # Not the default, and the flag exists because the answer was worth measuring: taking
                # the continuation on the surgery's own mesh, without further quality remeshes, fails
                # in 5 runs out of 5. The two fresh caps retract fast enough to degenerate the mesh
                # within a handful of steps, so remeshing after a pinch is not optional - which also
                # says that any post-event fragility belongs to the remeshing path and not to the
                # surgery having left a bad mesh behind.
                p.do_call_remeshing_when_necessary = False
        if pinched_at is not None and len(rec.rows) - 1 - pinched_at >= args.post_steps:
            break
    emit("DONE", pinched=pinched_at is not None, n_steps=len(rec.rows),
         post_steps=0 if pinched_at is None else len(rec.rows) - 1 - pinched_at,
         ndof=_dof_count(p))


def _run_coalescence(p, outdir, args, extra_domains=()):
    """Approach at ``max_dt``, merge, then continue at the (much smaller) ``post_dt``.

    The two phases need different step sizes and there is no point pretending otherwise: before the
    merge the flow is a rigid translation and any step resolves it, while the merged shape is a
    peanut whose bridge opens at the capillary-inertial rate of its own radius. Left at ``max_dt``,
    the post-merge steps alone drifted the volume by 1e-2 - not a property of the surgery but of a
    step ten times too long for what follows it.
    """
    setup(p, outdir)
    rec = StepRecorder(p, extra_domains=extra_domains)
    emit("SETUP", ndof=_dof_count(p), distmin=p.distmin, hmin=p.hmin, Oh=p.Oh,
         gap=p.gap, speed=p.approach_speed)
    rec.record("initial")
    if extra_domains:
        emit("CONFORMING", step=0, **_conformity(p))

    merged_at = None
    t = 0.0
    dt_pre = min(args.max_dt, p.approach_dt)
    for _step in range(args.max_steps):
        dt = dt_pre if merged_at is None else args.post_dt
        t += dt
        p.run(t, outstep=False, maxstep=dt, temporal_error=1, do_not_set_IC=True)
        row = rec.record()
        if extra_domains:
            emit("CONFORMING", step=len(rec.rows) - 1, **_conformity(p))
        if merged_at is None and row["n_fragments"] == 1:
            merged_at = len(rec.rows) - 1
            emit("EVENT", step=merged_at, t=row["t"], n_fragments=1)
        if merged_at is not None and len(rec.rows) - 1 - merged_at >= args.post_steps:
            break
    emit("DONE", merged=merged_at is not None, n_steps=len(rec.rows),
         post_steps=0 if merged_at is None else len(rec.rows) - 1 - merged_at,
         ndof=_dof_count(p))


def case_coalescence(outdir, args):
    p = CoalescenceProblem()
    p.Oh = args.Oh
    if args.hmin is not None:
        p.hmin = args.hmin
    _run_coalescence(p, outdir, args)


def case_twophase(outdir, args):
    p = TwoPhaseCoalescenceProblem()
    p.Oh = args.Oh
    if args.hmin is not None:
        p.hmin = args.hmin
    _run_coalescence(p, outdir, args, extra_domains=["gas"])


def _conformity(problem):
    """How far the gas side of the interface is from the liquid side.

    The two domains of one Gmsh template share the *curve*, not the nodes: each gets its own copy of
    the boundary and a ``ConnectMeshAtInterface`` constrains the two copies to coincide. So the
    statement worth checking is not identity but that the copies still describe the same curve - the
    same number of nodes, in the same places.

    The pairing is by rank along the interface (sorted by ``(y, x)``), not by nearest neighbour: the
    two lists have no index correspondence, but they do have the same order along the curve, and a
    nearest-neighbour pairing would happily pass a mesh that had lost half of its nodes.
    """
    out = {}
    pts = {}
    for dom in ("liquid", "gas"):
        mesh = problem.get_mesh(dom)
        bi = mesh.get_boundary_index("interface")
        nodes = [mesh.boundary_node_pt(bi, i) for i in range(mesh.nboundary_node(bi))]
        pts[dom] = sorted((n.x(1), n.x(0)) for n in nodes)
        out["n_" + dom] = len(nodes)
    if out["n_liquid"] != out["n_gas"]:
        out["max_mismatch"] = float("inf")
        return out
    out["max_mismatch"] = max(
        (numpy.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(pts["liquid"], pts["gas"])),
        default=0.0)
    return out


CASES = {"rayleigh_plateau": case_rayleigh_plateau,
         "coalescence": case_coalescence,
         "twophase": case_twophase}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--Oh", type=float, default=0.1)
    parser.add_argument("--amplitude", type=float, default=None)
    parser.add_argument("--hmin", type=float, default=None)
    parser.add_argument("--dt-factor", dest="dt_factor", type=float, default=0.15)
    parser.add_argument("--max-dt", dest="max_dt", type=float, default=0.25)
    parser.add_argument("--post-dt", dest="post_dt", type=float, default=0.02)
    # Only reached after the pinch, where the neck radius is no longer a length scale: the two fresh
    # caps are the fastest thing on the mesh and retract at the capillary-inertial rate of their own
    # radius, which is h_min. 0.05*h_min^(3/2) is a twentieth of that time. At 2e-3, a quarter of it,
    # the caps moved far enough per step to force a quality remesh on EVERY step, and one run in six
    # then diverged a few remeshes later - a plain remesh of a violently retracting tip, nothing to do
    # with the surgery, but a flaky test all the same.
    parser.add_argument("--min-dt", dest="min_dt", type=float, default=5e-4)
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=400)
    parser.add_argument("--post-steps", dest="post_steps", type=int, default=20)
    parser.add_argument("--restart-mode", dest="restart_mode", default="impulsive",
                        choices=["bdf1", "impulsive", "none"],
                        help="how to restart the time stepper at every event: one step with BDF1 "
                             "weights, an impulsive flattening of the whole history, or nothing at "
                             "all (which diverges immediately). See case_rayleigh_plateau.")
    parser.add_argument("--no-post-remesh", dest="no_post_remesh", action="store_true",
                        help="switch quality remeshing off after a pinch (it then fails within a few "
                             "steps; see case_rayleigh_plateau)")
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    try:
        CASES[args.case](args.outdir, args)
    except BaseException as e:  # noqa: BLE001
        print("PYOOMPH_RAISED %s: %s" % (type(e).__name__, " | ".join(str(e).splitlines())), flush=True)
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
