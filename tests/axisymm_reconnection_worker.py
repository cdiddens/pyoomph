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

"""Worker for the AxisymmetricReconnection tests -- one Problem per process, since a second one
segfaults in the JIT loader.

The scenarios are deliberately kinematic. What is under test is the detection, the surgery plan and
the geometry that ``define_geometry`` rebuilds from it; a real free-surface flow would add a Newton
solve whose outcome the assertions would then depend on, and would have to be *tuned* until it
pinches, which is exactly the kind of test that stops being about the code under test.

So: the interface is a prescribed profile. Where a case needs the interface to *move* (the overlap
guard, and the mesh-velocity gate that refuses to coalesce two tips flying apart), the bulk carries a
PrescribedMovingMesh with

    u_mesh = (0, -Vz*signum(lagrangian_y))

with a global parameter ``Vz``, so that ``Vz > 0`` moves two stacked blobs together and ``Vz < 0``
moves them apart, rigidly. The mesh VELOCITY rather than the position, because a position pinned by
a DirichletBC carries the prescribed value in all of its history slots and therefore has a mesh
velocity of exactly zero -- which is the very quantity the coalescence gate reads.
"""

import argparse
import json
import sys
import traceback

import numpy

from pyoomph import Problem, Equations, DirichletBC, ElementSpace, InterfaceEquations, var
from pyoomph.expressions import cos, grad, partial_t, signum, scale_factor, vector, var_and_test, weak
from pyoomph.expressions.units import meter
from pyoomph.equations.ALE import PrescribedMovingMesh
from pyoomph.equations.generic import ProjectExpression
from pyoomph.equations.topological_changes import (AxisymmetricReconnection,
                                                   TopologicalChangesGmshTemplate,
                                                   TopologicalChangesTQMeshTemplate)
from pyoomph.meshes.axisymm_topology import revolved_volume


# --------------------------------------------------------------------------------------
# Prescribed half-sections (nondimensional, x = r, y = z), bottom axis tip -> top axis tip
# --------------------------------------------------------------------------------------

def _tip_clustered(n):
    """Parameter samples in [-1,1] that bunch up at the two ends.

    An axial tip of a smooth axisymmetric body behaves like r ~ sqrt(z_tip - z), so a uniform sample
    puts its first point off the tip at a radius of order sqrt(h) -- a jump of 0.16 for h = 0.0125.
    That is not a resolved tip and it is not what the mesh will look like either.
    """
    s = numpy.linspace(-1.0, 1.0, n)
    return numpy.sign(s) * (1.0 - (1.0 - numpy.abs(s)) ** 2)


def dumbbell(neck, height=1.0, bulge=1.0, n=161):
    """r(t) = sqrt(1-t^2)*(neck + bulge*t^2), z = height*t: one waist of radius ``neck`` at z=0."""
    t = _tip_clustered(n)
    r = numpy.sqrt(numpy.maximum(0.0, 1.0 - t * t)) * (neck + bulge * t * t)
    return numpy.stack([r, height * t], axis=1)


def spheroid(radius, half_height, z_center, n=101):
    t = _tip_clustered(n)
    r = radius * numpy.sqrt(numpy.maximum(0.0, 1.0 - t * t))
    return numpy.stack([r, z_center + half_height * t], axis=1)


def two_blobs(gap, radius=0.35, half_height=0.35, n=101):
    """Two stacked spheroids whose facing tips are ``gap`` apart, symmetric about z = 0."""
    zc = half_height + 0.5 * gap
    return [spheroid(radius, half_height, -zc, n), spheroid(radius, half_height, zc, n)]


def halfsection_volume(points):
    """Revolved volume of an interface polyline that starts and ends on the axis."""
    return revolved_volume(numpy.asarray(points, dtype=float))


# --------------------------------------------------------------------------------------
# The mesh template
# --------------------------------------------------------------------------------------

class _BlobMeshBody:
    """Axisymmetric fluid fragments on the symmetry axis, optionally inside a gas box.

    Boundaries: ``interface`` (free surface), ``axis`` (the liquid's share of r=0), and, with a gas
    box, ``gas_axis`` (the gas's share of r=0) and ``outer`` (the box walls).

    The remeshing branch is the point of the exercise: it is entered both for a reconnection and for
    a plain quality remesh, and it does not know which -- ``get_reconnected_boundaries`` fills in the
    same structure either way.

    The body is written once and given to one class per meshing backend below: the geometry calls the
    surgery goes through are spelled the same way by gmsh and by TQMesh, which is the whole claim of
    TopologicalChangesTemplate, and running the same scenarios on both is how that claim is checked.
    Not a base class, because nanobind refuses a MeshTemplate with two bases.
    """

    def __init__(self, profiles, resolution, unit=1, gas_box=None, size_via_callable=False):
        self.profiles = [numpy.asarray(p, dtype=float) for p in profiles]
        self.resolution = resolution
        self.unit = unit          # the spatial unit the coordinates are expressed in
        self.gas_box = gas_box    # (Rbox, Zbox) or None
        #: Take spline_from_chain's callable branch instead of its default one. The callable hands
        #: back the suggested size unchanged, so the mesh is identical either way and the two paths
        #: are directly comparable.
        self.size_via_callable = size_via_callable
        #: Filled on every remesh, so the test can see what define_geometry was handed.
        self.last_report = {}

    def define_geometry(self):
        self.mesh_mode = "tris"
        self.default_resolution = self.resolution
        U = self.unit
        if self.is_first_time():
            chains = [[(float(x) * U, float(y) * U) for x, y in p] for p in self.profiles]
            axis = [((0.0 * U, float(p[0, 1]) * U), (0.0 * U, float(p[-1, 1]) * U))
                    for p in self.profiles]
            opposite = None
        else:
            rb = self.get_reconnected_boundaries(
                "liquid/interface", "liquid/axis",
                "gas/gas_axis" if self.gas_box is not None else None)
            self.last_report = {
                "has_plan": rb.has_plan,
                "events": [(e.kind, float(e.z_center)) for e in rb.events],
                "n_chains": len(rb.interface_chains),
                "fragment_volumes": [float(v / U ** 3) for v in rb.fragment_volumes],
                "axis_spans": [(float(a[1] / U), float(b[1] / U)) for a, b in rb.axis_segments],
                "opposite_axis_spans": [(float(a[1] / U), float(b[1] / U))
                                        for a, b in rb.opposite_axis_segments],
            }
            size = (lambda _x, _y, suggested: suggested) if self.size_via_callable else None
            for chain in rb.interface_chains:
                self.spline_from_chain(chain, "interface", size=size)
            self.lines_from_axis_segments(rb.axis_segments, "axis")
            # Whatever axis is left over goes to the opposite phase. Without a gas box that list is
            # empty and this is a no-op, which is the point: one code path for both.
            self.lines_from_axis_segments(rb.opposite_axis_segments, "gas_axis")
            self.plane_surface("interface", "axis", name="liquid")
            if self.gas_box is not None:
                self._gas_box_walls()
                self.plane_surface("interface", "gas_axis", "outer", name="gas")
            return

        # ---- first time: the same geometry, straight from the prescribed profiles ----
        for pts in chains:
            self.spline([self.point(x, y) for x, y in pts], name="interface")
        self.lines_from_axis_segments(axis, "axis")
        self.plane_surface("interface", "axis", name="liquid")
        if self.gas_box is not None:
            _, Zbox = self.gas_box
            zlo = float(self.profiles[0][0, 1])
            zhi = float(self.profiles[-1][-1, 1])
            gaps = [((0.0 * U, -Zbox * U), (0.0 * U, zlo * U)), ((0.0 * U, zhi * U), (0.0 * U, Zbox * U))]
            for k in range(len(self.profiles) - 1):
                gaps.append(((0.0 * U, float(self.profiles[k][-1, 1]) * U),
                             (0.0 * U, float(self.profiles[k + 1][0, 1]) * U)))
            self.lines_from_axis_segments(gaps, "gas_axis")
            self._gas_box_walls()
            self.plane_surface("interface", "gas_axis", "outer", name="gas")

    def _gas_box_walls(self):
        Rbox, Zbox = self.gas_box
        U = self.unit
        # The two corners on the axis are shared with the "gas_axis" lines. They are written here as
        # literals and there as numbers that travelled through the mesh, so they only meet because
        # TopologicalChangesGmshTemplate.point() merges near-coincident points.
        p = [self.point(0.0 * U, -Zbox * U), self.point(Rbox * U, -Zbox * U),
             self.point(Rbox * U, Zbox * U), self.point(0.0 * U, Zbox * U)]
        for a, b in zip(p, p[1:]):
            self.line(a, b, name="outer")


class BlobMesh(TopologicalChangesGmshTemplate):
    __doc__ = _BlobMeshBody.__doc__

    def __init__(self, *args, **kwargs):
        super().__init__()
        _BlobMeshBody.__init__(self, *args, **kwargs)

    define_geometry = _BlobMeshBody.define_geometry
    _gas_box_walls = _BlobMeshBody._gas_box_walls


class BlobMeshTQMesh(TopologicalChangesTQMeshTemplate):
    __doc__ = _BlobMeshBody.__doc__

    def __init__(self, *args, **kwargs):
        super().__init__()
        _BlobMeshBody.__init__(self, *args, **kwargs)

    define_geometry = _BlobMeshBody.define_geometry
    _gas_box_walls = _BlobMeshBody._gas_box_walls


#: Which backend the scenarios are built on; set from --backend in main().
BLOB_MESH_CLASS = BlobMesh


# --------------------------------------------------------------------------------------
# The problem
# --------------------------------------------------------------------------------------

class SurfaceField(InterfaceEquations):
    """A field that exists on the interface only, projected onto the axial coordinate.

    ``f = z`` is the useful profile here: the pinch happens at the waist, so a value that ended up on
    the wrong side of it announces itself by its SIGN, and along a coalescence bridge the two old tip
    values bracket everything the bridge may legitimately have.
    """

    def __init__(self, name="f"):
        super().__init__()
        self.fieldname = name

    def define_fields(self):
        self.define_scalar_field(self.fieldname, "C2")

    def define_residuals(self):
        f, ft = var_and_test(self.fieldname)
        self.add_residual(weak(f - var("coordinate_y") / scale_factor("spatial"), ft))


class TransportedField(Equations):
    """A diffusing scalar on the moving mesh: something with a genuine TIME HISTORY.

    ProjectExpression would not do - a field with no time derivative in its residual is never given
    history slots, so all of its past levels stay at zero and there is nothing for the transfer to
    get right or wrong.
    """

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, ut = var_and_test("u")
        self.add_weak(partial_t(u, ALE=True), ut).add_weak(0.05 * grad(u), grad(ut))
        # Curved, so that diffusion actually changes it: a profile that merely translates with the
        # mesh has du/dt = 0 at every node and, again, nothing to measure.
        self.set_initial_condition("u", 1 + cos(2 * var("coordinate_y") / scale_factor("spatial")))


class ReconnectionProblem(Problem):
    def __init__(self, profiles, *, resolution=0.06, rmin=None, distmin=None, scale=None,
                 moving_mesh=False, check_motion=False, gas_box=None, volume_conservation=True,
                 overlap_reject_factor=0.2, size_via_callable=False, surface_field=False,
                 handle_zeta=True, user_zeta=False, with_field=False):
        super().__init__()
        #: Carry a projected bulk field alongside the moving mesh, so that the VALUE history can be
        #: inspected as well as the position history.
        self.with_field = with_field
        self.surface_field = surface_field
        self.handle_zeta = handle_zeta
        #: Attach a user's own AssignZetaCoordinatesByArclength next to the handler. Its chart cannot
        #: describe the event -- it renormalises per segment, so the lower fragment's [0,1] covers the
        #: WHOLE old interface, waist included -- which is exactly what makes it a test of who wins.
        self.user_zeta = user_zeta
        self.profiles = profiles
        self.resolution = resolution
        self.rmin = rmin
        self.distmin = distmin
        self.scale = scale
        self.moving_mesh = moving_mesh
        self.check_motion = check_motion
        self.gas_box = gas_box
        self.volume_conservation = volume_conservation
        self.overlap_reject_factor = overlap_reject_factor
        self.size_via_callable = size_via_callable
        self.reconnection = None
        self.mesh_template = None

    def define_problem(self):
        self.set_coordinate_system("axisymmetric")
        unit = 1
        if self.scale is not None:
            self.set_scaling(spatial=self.scale * meter)
            unit = meter
        self.mesh_template = BLOB_MESH_CLASS(self.profiles, self.resolution, unit=unit, gas_box=self.gas_box,
                                            size_via_callable=self.size_via_callable)
        self.add_mesh(self.mesh_template)

        self.reconnection = AxisymmetricReconnection(
            rmin=None if self.rmin is None else self.rmin * unit,
            distmin=None if self.distmin is None else self.distmin * unit,
            check_mesh_motion_direction=self.check_motion,
            overlap_reject_factor=self.overlap_reject_factor,
            volume_conservation=self.volume_conservation,
            handle_zeta=self.handle_zeta)

        if self.moving_mesh:
            # The mesh VELOCITY is prescribed, not the position. Pinning a position with a
            # DirichletBC would have been the obvious way to drive the blobs, but a pinned nodal
            # position carries the prescribed value in all of its history slots, so partial_t of it
            # is exactly zero -- which is precisely the quantity the velocity gate reads. Verified
            # on a translating unit square: every free node reported the right mesh velocity and the
            # Dirichlet-pinned row reported 0.
            Vz = self.get_global_parameter("Vz").get_symbol()
            # Vz > 0 brings two blobs stacked about z = 0 together, Vz < 0 pulls them apart.
            umesh = vector(0, -Vz * signum(var("lagrangian_y")))
            for dom in (["liquid", "gas"] if self.gas_box is not None else ["liquid"]):
                eqs = ElementSpace("C2") + PrescribedMovingMesh(umesh)
                if self.with_field:
                    eqs += TransportedField()
                eqs += Equations() @ ("axis" if dom == "liquid" else "gas_axis")
                if dom == "gas":
                    eqs += Equations() @ "outer"
                    eqs += Equations() @ "interface"
                else:
                    eqs += self.reconnection @ "interface"
                self.add_equations(eqs @ dom)
        else:
            # No mesh motion at all: something still has to carry a dof, and a projected field is
            # also what makes the mesh-to-mesh transfer after the remesh do some work.
            for dom in (["liquid", "gas"] if self.gas_box is not None else ["liquid"]):
                eqs = ElementSpace("C2") + ProjectExpression(u=1 + var("coordinate_y") / scale_factor("spatial"))
                eqs += Equations() @ ("axis" if dom == "liquid" else "gas_axis")
                if dom == "gas":
                    eqs += Equations() @ "outer"
                    # A separate name from the liquid's, so that reading one back cannot accidentally
                    # be reading the other: the two sides share the nodes, not the dof.
                    gas_iface = Equations() if not self.surface_field else SurfaceField("fgas")
                    eqs += gas_iface @ "interface"
                else:
                    iface = self.reconnection
                    if self.surface_field:
                        iface = iface + SurfaceField("f")
                    if self.user_zeta:
                        from pyoomph.meshes.zeta import AssignZetaCoordinatesByArclength
                        iface = iface + AssignZetaCoordinatesByArclength(sort_along_axis="y+")
                    eqs += iface @ "interface"
                self.add_equations(eqs @ dom)


# --------------------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------------------

def emit(_kind, **kw):
    print("PYOOMPH_" + _kind + " " + json.dumps(kw, sort_keys=True), flush=True)


def interface_state(problem, domain="liquid", interface="interface", axis="axis"):
    """The current interface polylines and the axis spans, nondimensional, ordered by z."""
    from pyoomph.meshes.meshdatacache import MeshDataCache
    from pyoomph.meshes.ordering import sort_line_segments
    cache = MeshDataCache(tesselate_tri=False, nondimensional=True)
    out = {}
    for name, key in ((interface, "interface"), (axis, "axis")):
        mesh = problem.get_mesh(domain + "/" + name)
        data = cache.get_data(mesh)
        segs, _ = data.get_interface_line_segments()
        pts = data.get_coordinates()
        segs = sort_line_segments(pts, segs, sort_along_axis="y+", whom="interface_state")
        out[key] = [[(float(pts[0, i]), float(pts[1, i])) for i in s] for s in segs]
    return out


def element_orientations(problem, domain):
    """Signed areas of the first three nodes of every element.

    Ordering-agnostic as an inversion test: whatever those three nodes are, a mesh without inverted
    or collapsed elements gives them all the same sign and none of them near zero.
    """
    mesh = problem.get_mesh(domain)
    areas = []
    for e in mesh.elements():
        p = [(e.node_pt(i).x(0), e.node_pt(i).x(1)) for i in range(3)]
        areas.append(0.5 * ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                            - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1])))
    return areas


def report_mesh(problem, tag, domain="liquid", interface="interface", axis="axis"):
    st = interface_state(problem, domain, interface, axis)
    vols = [halfsection_volume(seg) for seg in st["interface"]]
    areas = element_orientations(problem, domain)
    scaled = [a / max(abs(x) for x in areas) for a in areas]
    emit("MESH", tag=tag, domain=domain,
         n_interface=len(st["interface"]),
         n_axis=len(st["axis"]),
         interface_ends=[[list(seg[0]), list(seg[-1])] for seg in st["interface"]],
         axis_spans=[[seg[0][1], seg[-1][1]] for seg in st["axis"]],
         volumes=vols,
         n_elements=len(areas),
         n_negative=sum(1 for a in scaled if a < 0),
         n_positive=sum(1 for a in scaled if a > 0),
         min_abs_scaled_area=min(abs(a) for a in scaled))
    return st, vols


def interface_field_state(problem, domain, interface, fieldname):
    """Per interface segment, the ordered ``(x, y, f)`` of its nodes, nondimensional and by ascending z."""
    from pyoomph.meshes.meshdatacache import MeshDataCache
    from pyoomph.meshes.ordering import sort_line_segments
    mesh = problem.get_mesh(domain + "/" + interface)
    data = MeshDataCache(tesselate_tri=False, nondimensional=True).get_data(mesh)
    segs, _ = data.get_interface_line_segments()
    pts = data.get_coordinates()
    vals = data.get_data(fieldname)
    segs = sort_line_segments(pts, segs, sort_along_axis="y+", whom="interface_field_state")
    out = []
    for s in segs:
        seg = [(float(pts[0, i]), float(pts[1, i]), float(vals[i])) for i in s]
        if seg[0][1] > seg[-1][1]:
            seg = seg[::-1]
        out.append(seg)
    return out


def bulk_field_state(problem, domain):
    """``(x, y, u)`` of every node of a bulk domain, with u the projected 1 + z."""
    mesh = problem.get_mesh(domain)
    return [(n.x(0), n.x(1), n.value(0)) for n in mesh.nodes()]


def report_fields(problem, tag, domain="liquid", interface="interface", fieldname="f"):
    segs = interface_field_state(problem, domain, interface, fieldname)
    bulk = bulk_field_state(problem, domain)
    emit("FIELD", tag=tag, domain=domain, field=fieldname, segments=segs)
    emit("BULK", tag=tag, domain=domain,
         worst=max(abs(u - (1.0 + y)) for _x, y, u in bulk),
         worst_at=max(((abs(u - (1.0 + y)), x, y, u) for x, y, u in bulk))[1:],
         umin=min(u for _x, _y, u in bulk), umax=max(u for _x, _y, u in bulk),
         n_nonfinite=sum(1 for _x, _y, u in bulk if not numpy.isfinite(u)),
         n_nodes=len(bulk))


def watch_interpolator_config():
    """Emit what each interpolator was configured with, just before it runs.

    The only way to see from outside whether the reconnection handler took a boundary's zeta over -
    which is the difference between an event remesh and a plain quality one.
    """
    from pyoomph.meshes import interpolator as interp_mod
    original = interp_mod.InternalInterpolator.interpolate

    def wrapped(self):
        emit("INTERP", domain=self.new.get_name(),
             zeta_overridden=sorted(self.zeta_overridden_boundaries),
             bulk_locate=sorted(self.bulk_locate_boundaries),
             zeta_iface_only=sorted(self.zeta_for_interface_fields_only),
             max_dist=sorted(self.boundary_max_distances.items()))
        return original(self)

    interp_mod.InternalInterpolator.interpolate = wrapped


def setup(problem, outdir):
    problem.set_output_directory(outdir)
    problem.quiet()
    problem.initialise()


# --------------------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------------------

def case_detect_pinch(outdir, args):
    """A neck above / below rmin. Run with a spatial scaling, so that the dimensional rmin has to be
    nondimensionalised correctly for the answer to come out right at all."""
    neck = args.neck
    p = ReconnectionProblem([dumbbell(neck)], rmin=0.12, distmin=None, scale=0.5,
                            resolution=0.06, moving_mesh=False, check_motion=False)
    setup(p, outdir)
    p.do_call_remeshing_when_necessary = False   # detection only; do not rebuild the mesh here
    p.solve()
    tmpl = p.mesh_template
    emit("DETECT", neck=neck, has_plan=tmpl.has_pending_surgery_plan("liquid/interface"),
         queued=tmpl in p._domains_to_remesh)
    plan = p.reconnection._last_plan
    if plan is not None and tmpl.has_pending_surgery_plan("liquid/interface"):
        emit("PLAN", kinds=[e.kind for e in plan.events],
             z=[float(e.z_center) for e in plan.events],
             before=[float(v) for v in plan.fragment_volumes_before],
             after=[float(v) for v in plan.fragment_volumes_after])


def case_overlap_guard(outdir, args):
    """Armed by a gap of 2*distmin, then asked to take a step that would close it right through."""
    distmin = 0.05
    p = ReconnectionProblem(two_blobs(2.0 * distmin), rmin=None, distmin=distmin,
                            resolution=0.06, moving_mesh=True, check_motion=True)
    setup(p, outdir)
    p.do_call_remeshing_when_necessary = False

    def gap():
        st = interface_state(p, "liquid")
        if len(st["interface"]) < 2:
            return 0.0
        return st["interface"][1][0][1] - st["interface"][0][-1][1]

    p.solve(timestep=1.0)
    emit("ARMED", armed=bool(p.reconnection._armed), gap=gap(),
         limit=p.reconnection.overlap_reject_factor * distmin)

    # A modest step: the gap stays well above the limit, so the guard must let it through.
    p.get_global_parameter("Vz").value = 0.02
    try:
        p.solve(timestep=1.0)
        emit("HARMLESS", rejected=False, gap=gap())
    except Exception as e:  # noqa: BLE001
        emit("HARMLESS", rejected=True, exception=type(e).__name__, gap=gap())

    # And one that would drive the two tips straight through each other must be refused.
    p.get_global_parameter("Vz").value = 0.2
    try:
        p.solve(timestep=1.0)
        emit("GUARD", rejected=False, gap=gap())
    except Exception as e:  # noqa: BLE001 -- what matters is that the solve was abandoned
        emit("GUARD", rejected=True, exception=type(e).__name__)


def case_separating_tips(outdir, args):
    """Two tips within distmin of each other, but flying apart.

    With ``check_mesh_motion_direction`` the coalescence must not be planned; without it, the very
    same configuration must be, or the test would prove nothing about the gate.
    """
    distmin = 0.05
    gap = 0.5 * distmin
    p = ReconnectionProblem(two_blobs(gap), rmin=None, distmin=distmin, resolution=0.06,
                            moving_mesh=True, check_motion=args.check_motion,
                            overlap_reject_factor=None)
    setup(p, outdir)
    p.do_call_remeshing_when_necessary = False
    # Sz < 0 pulls the two blobs apart; a small step over a small dt gives a clear separation
    # velocity while leaving the gap comfortably below distmin.
    p.get_global_parameter("Vz").value = -0.2
    p.solve(timestep=0.01)
    st = interface_state(p, "liquid")
    gap_now = st["interface"][1][0][1] - st["interface"][0][-1][1]
    emit("SEPARATING", check_motion=args.check_motion, gap=gap_now, distmin=distmin,
         has_plan=p.mesh_template.has_pending_surgery_plan("liquid/interface"))


def case_pinch_remesh(outdir, args):
    """A plain quality remesh through the same define_geometry, then the pinch-off itself."""
    p = ReconnectionProblem([dumbbell(0.04)], rmin=0.12, distmin=None,
                            resolution=args.resolution, moving_mesh=False, check_motion=False,
                            volume_conservation=not args.no_volume_conservation)
    setup(p, outdir)
    st0, v0 = report_mesh(p, "initial")

    # (5) The no-event path: force a remesh with nothing pending. The same code must reproduce the
    # boundary it was given.
    p.force_remesh({p.mesh_template})
    emit("REMESH_REPORT", which="quality", **p.mesh_template.last_report)
    st1, v1 = report_mesh(p, "after_quality_remesh")

    # (2) Now the pinch-off.
    p.solve()
    plan = p.reconnection._last_plan
    # The plan's own books: with volume_conservation the fresh cap points are moved along the normal
    # until each fragment hits its share of the parent volume exactly, so these must balance. Without
    # it they must not -- that is the only measurement that isolates the correction from the O(h^2)
    # error of turning the plan's polyline into a spline and meshing it.
    emit("PLAN_BALANCE", conservation=not args.no_volume_conservation,
         before=[float(v) for v in plan.fragment_volumes_before],
         after=[float(v) for v in plan.fragment_volumes_after],
         resolution=args.resolution)
    emit("REMESH_REPORT", which="pinch", **p.mesh_template.last_report)
    st2, v2 = report_mesh(p, "after_pinch")
    p.solve()   # one more step on the new mesh: it must be solvable, and must not re-detect
    emit("RESOLVED", has_plan=p.mesh_template.has_pending_surgery_plan("liquid/interface"))


def case_coalescence_remesh(outdir, args):
    distmin = 0.05
    p = ReconnectionProblem(two_blobs(0.4 * distmin), rmin=None, distmin=distmin,
                            resolution=args.resolution, moving_mesh=False, check_motion=False,
                            overlap_reject_factor=None, size_via_callable=True)
    setup(p, outdir)
    report_mesh(p, "initial")
    p.solve()
    emit("REMESH_REPORT", which="coalescence", **p.mesh_template.last_report)
    report_mesh(p, "after_coalescence")
    p.solve()
    emit("RESOLVED", has_plan=p.mesh_template.has_pending_surgery_plan("liquid/interface"))


def case_twophase_remesh(outdir, args):
    p = ReconnectionProblem([dumbbell(0.04)], rmin=0.12, distmin=None, resolution=0.08,
                            moving_mesh=False, check_motion=False, gas_box=(0.6, 1.3))
    setup(p, outdir)
    report_mesh(p, "initial", "liquid")
    report_mesh(p, "initial_gas", "gas", "interface", "gas_axis")
    p.solve()
    emit("REMESH_REPORT", which="pinch", **p.mesh_template.last_report)
    report_mesh(p, "after_pinch", "liquid")
    report_mesh(p, "after_pinch_gas", "gas", "interface", "gas_axis")

    # Conformity: every node of the liquid side of the interface must coincide with a node of the
    # gas side. They are separate meshes, so this is a real statement about the shared spline.
    def nodes_of(mesh):
        # InterfaceMesh.nodes() is empty here (the interface carries no dofs of its own), so take
        # the nodes off the elements instead.
        seen = {}
        for e in mesh.elements():
            for i in range(e.nnode()):
                n = e.node_pt(i)
                seen[(round(n.x(0), 14), round(n.x(1), 14))] = (n.x(0), n.x(1))
        return numpy.array(list(seen.values()))

    lpts, gpts = nodes_of(p.get_mesh("liquid/interface")), nodes_of(p.get_mesh("gas/interface"))
    worst = 0.0
    for x, y in lpts:
        worst = max(worst, float(numpy.min(numpy.hypot(gpts[:, 0] - x, gpts[:, 1] - y))))
    emit("CONFORMITY", n_liquid=len(lpts), n_gas=len(gpts), worst_distance=worst)
    p.solve()


# --------------------------------------------------------------------------------------
# Field transfer across an event (stage 6)
# --------------------------------------------------------------------------------------

def case_transfer_quality(outdir, args):
    """The handler attached, but nothing to detect: a plain quality remesh must be untouched by it.

    A neck well above ``rmin``, so the same equation that carries the surgery is present and silent.
    """
    watch_interpolator_config()
    p = ReconnectionProblem([dumbbell(0.3)], rmin=0.12, distmin=None, resolution=args.resolution,
                            moving_mesh=False, check_motion=False, surface_field=True)
    setup(p, outdir)
    p.do_call_remeshing_when_necessary = False
    p.solve()
    report_fields(p, "initial")
    p.force_remesh({p.mesh_template})
    emit("REMESH_REPORT", which="quality", **p.mesh_template.last_report)
    report_fields(p, "after_quality_remesh")
    p.solve()
    emit("RESOLVED", has_plan=p.mesh_template.has_pending_surgery_plan("liquid/interface"))


def case_transfer_pinch(outdir, args):
    """An interface field f = z and a bulk field u = 1 + z, carried through a pinch-off.

    The solve is kept away from the remesh (``do_call_remeshing_when_necessary = False``) so that the
    field state before the event can be reported: the plan is parked by the solve and built by the
    explicit force_remesh that follows. A second, ordinary remesh afterwards, because a one-off zeta
    chart that is not taken back again makes exactly that one fail.
    """
    watch_interpolator_config()
    p = ReconnectionProblem([dumbbell(0.04)], rmin=0.12, distmin=None, resolution=args.resolution,
                            moving_mesh=False, check_motion=False, surface_field=True,
                            handle_zeta=not args.no_handle_zeta)
    setup(p, outdir)
    p.do_call_remeshing_when_necessary = False
    p.solve()
    report_fields(p, "initial")
    p.force_remesh({p.mesh_template})
    emit("REMESH_REPORT", which="pinch", **p.mesh_template.last_report)
    report_fields(p, "after_pinch")
    p.solve()
    p.force_remesh({p.mesh_template})
    emit("REMESH_REPORT", which="after_event_quality", **p.mesh_template.last_report)
    report_fields(p, "after_second_remesh")
    emit("RESOLVED", has_plan=p.mesh_template.has_pending_surgery_plan("liquid/interface"))


def case_transfer_user_zeta(outdir, args):
    """The same pinch with the user's own zeta assigner attached to the same interface.

    Two things have to hold. The handler's chart must win -- the assigner's renormalised arclength
    would map the lower fragment onto the whole old interface, waist included -- and the assigner
    must go on owning the boundary coordinate afterwards, so that the ordinary remesh that follows
    still finds one on both meshes.
    """
    watch_interpolator_config()
    p = ReconnectionProblem([dumbbell(0.04)], rmin=0.12, distmin=None, resolution=args.resolution,
                            moving_mesh=False, check_motion=False, surface_field=True,
                            handle_zeta=not args.no_handle_zeta, user_zeta=True)
    setup(p, outdir)
    p.do_call_remeshing_when_necessary = False
    p.solve()
    report_fields(p, "initial")
    p.force_remesh({p.mesh_template})
    emit("REMESH_REPORT", which="pinch", **p.mesh_template.last_report)
    report_fields(p, "after_pinch")
    p.solve()
    p.force_remesh({p.mesh_template})
    emit("REMESH_REPORT", which="after_event_quality", **p.mesh_template.last_report)
    report_fields(p, "after_second_remesh")
    emit("RESOLVED", has_plan=p.mesh_template.has_pending_surgery_plan("liquid/interface"))


def case_transfer_coalescence(outdir, args):
    """The same fields across a coalescence: the bridge is fresh interface between two old tips."""
    watch_interpolator_config()
    distmin = 0.05
    p = ReconnectionProblem(two_blobs(0.4 * distmin), rmin=None, distmin=distmin,
                            resolution=args.resolution, moving_mesh=False, check_motion=False,
                            overlap_reject_factor=None, surface_field=True,
                            handle_zeta=not args.no_handle_zeta)
    setup(p, outdir)
    p.do_call_remeshing_when_necessary = False
    p.solve()
    report_fields(p, "initial")
    p.force_remesh({p.mesh_template})
    emit("REMESH_REPORT", which="coalescence", **p.mesh_template.last_report)
    report_fields(p, "after_coalescence")
    p.solve()
    emit("RESOLVED", has_plan=p.mesh_template.has_pending_surgery_plan("liquid/interface"))


def case_transfer_twophase(outdir, args):
    """Both sides of the interface carry their own surface field through a pinch.

    The gas is the interesting side: the pinch hands it a stretch of axis that used to be liquid, so
    its fresh nodes have no counterpart anywhere on the old gas boundary.
    """
    watch_interpolator_config()
    p = ReconnectionProblem([dumbbell(0.04)], rmin=0.12, distmin=None, resolution=0.08,
                            moving_mesh=False, check_motion=False, gas_box=(0.6, 1.3),
                            surface_field=True, handle_zeta=not args.no_handle_zeta)
    setup(p, outdir)
    p.do_call_remeshing_when_necessary = False
    p.solve()
    report_fields(p, "initial")
    report_fields(p, "initial_gas", "gas", "interface", "fgas")
    p.force_remesh({p.mesh_template})
    emit("REMESH_REPORT", which="pinch", **p.mesh_template.last_report)
    report_fields(p, "after_pinch")
    report_fields(p, "after_pinch_gas", "gas", "interface", "fgas")

    # The freshly opened stretch of the gas axis: it was liquid a moment ago, so nothing on the old
    # gas mesh describes it. What matters is that its nodes come out finite and inside the range the
    # old gas field had, not that they reproduce 1 + z there.
    gas = p.get_mesh("gas")
    bi = gas.get_boundary_index("gas_axis")
    axis_nodes = [gas.boundary_node_pt(bi, i) for i in range(gas.nboundary_node(bi))]
    gap = [(n.x(1), n.value(0)) for n in axis_nodes if abs(n.x(1)) < 0.21]
    emit("GAS_GAP", n=len(gap), values=gap,
         n_nonfinite=sum(1 for _y, u in gap if not numpy.isfinite(u)))
    p.solve()


def case_fresh_node_history(outdir, args):
    """What history the ordinary transfer gives the nodes the surgery created.

    Two blobs driven together by a prescribed mesh velocity until they coalesce. A coalescence is the
    sharper of the two events for this: the bridge the surgery builds is genuinely OUTSIDE the old
    liquid, so the nodes on it have no old material point of their own at all, and what the ordinary
    transfer gives them is whichever old point the locator fell back on - together with its motion.
    Measured right after the transfer and before any further solve, so what is reported is what the
    transfer wrote: per node, how far the position and the value history are from the current state,
    split into the nodes within reach of the surgery's fresh points and all the others.
    """
    distmin = 0.12
    p = ReconnectionProblem(two_blobs(0.8 * distmin), rmin=None, distmin=distmin,
                            resolution=args.resolution, moving_mesh=True, check_motion=False,
                            overlap_reject_factor=None, with_field=True)
    setup(p, outdir)
    p.get_global_parameter("Vz").value = 0.05
    for _ in range(20):
        p.solve(timestep=0.1)
        if p.mesh_template.last_report.get("n_chains", 2) < 2:
            break
    else:
        emit("HISTORY", coalesced=False)
        return
    # "Within reach" is a ball of a couple of distmin around each fresh point of the plan, which is
    # about the size of the region the surgery rebuilt: the plan names the fresh interface POINTS,
    # while the mesh generator puts nodes of its own inside the bridge that are named nowhere.
    reach = 2.0 * p.reconnection._distmin_nd
    plan = p.reconnection._last_plan
    fresh = numpy.concatenate([numpy.asarray(nc.points)[numpy.asarray(nc.origin) < 0]
                               for nc in plan.new_chains])
    mesh = p.get_mesh("liquid")
    inside = {"dx": 0.0, "du": 0.0, "n": 0}
    outside = {"dx": 0.0, "du": 0.0, "n": 0}
    for n in mesh.nodes():
        x = numpy.array([n.x(0), n.x(1)])
        b = inside if numpy.min(numpy.linalg.norm(fresh - x, axis=1)) <= reach else outside
        b["n"] += 1
        b["dx"] = max(b["dx"], max(abs(n.x_at_t(t, i) - n.x(i)) for t in (1, 2) for i in (0, 1)))
        b["du"] = max(b["du"], max(abs(n.value_at_t(t, 0) - n.value(0)) for t in (1, 2)))
    emit("HISTORY", coalesced=True, reach=float(reach),
         n_fresh_points=int(len(fresh)), inside=inside, outside=outside)


def case_history_bdf1(outdir, args):
    """Whether ONE BDF1 step after the event makes the transferred history irrelevant.

    It should, and for a reason that has nothing to do with the surgery: oomph shifts the history
    slots at the start of every step (``BDF::shift_time_values``: value(2) := value(1),
    value(1) := value(0)), so the first post-event step reads the transferred level-0 state as its
    level 1 and the transferred level 1 as its level 2, and the second step has shifted both of the
    transferred history levels out again. BDF1 weights ignore level 2. So a first step taken with
    BDF1 weights uses nothing the surgery could have got wrong, and everything after it is built on
    post-event states alone.

    The case runs two steps past the event and digests the result, so that two runs differing ONLY in
    the transferred history can be compared: with ``--degrade-bdf1`` they must agree exactly, without
    it they must not. ``--flatten-history`` is what makes the two runs differ - it overwrites the
    history levels the transfer just wrote, and only those, leaving the current state (level 0)
    untouched. A test instrument, not a strategy: the point is that the answer does not depend on it.
    """
    distmin = 0.12
    p = ReconnectionProblem(two_blobs(0.8 * distmin), rmin=None, distmin=distmin,
                            resolution=args.resolution, moving_mesh=True, check_motion=False,
                            overlap_reject_factor=None, with_field=True)
    setup(p, outdir)
    p.get_global_parameter("Vz").value = 0.05
    for _ in range(20):
        p.solve(timestep=0.1)
        if p.mesh_template.last_report.get("n_chains", 2) < 2:
            break
    else:
        emit("DIGEST", coalesced=False)
        return
    p.do_call_remeshing_when_necessary = False   # keep the mesh fixed, so only the history varies
    if args.flatten_history:
        for n in p.get_mesh("liquid").nodes():
            for t in range(1, 3):
                for i in range(2):
                    n.set_x_at_t(t, i, n.x(i))
                for vi in range(n.nvalue()):
                    n.set_value_at_t(t, vi, n.value(vi))
    if args.degrade_bdf1:
        p.timestepper.set_num_unsteady_steps_done(0)
    for _ in range(2):
        p.solve(timestep=0.1)
    ns = list(p.get_mesh("liquid").nodes())
    emit("DIGEST", coalesced=True, flatten_history=bool(args.flatten_history),
         degrade_bdf1=bool(args.degrade_bdf1),
         n_nodes=len(ns),
         u_sum=sum(n.value(0) for n in ns), u_min=min(n.value(0) for n in ns),
         u_max=max(n.value(0) for n in ns),
         x_sum=sum(n.x(0) + n.x(1) for n in ns))


CASES = {"detect_pinch": case_detect_pinch, "overlap_guard": case_overlap_guard,
         "separating_tips": case_separating_tips, "pinch_remesh": case_pinch_remesh,
         "coalescence_remesh": case_coalescence_remesh, "twophase_remesh": case_twophase_remesh,
         "transfer_quality": case_transfer_quality, "transfer_pinch": case_transfer_pinch,
         "transfer_user_zeta": case_transfer_user_zeta,
         "transfer_coalescence": case_transfer_coalescence,
         "transfer_twophase": case_transfer_twophase,
         "fresh_node_history": case_fresh_node_history,
         "history_bdf1": case_history_bdf1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--neck", type=float, default=0.04)
    parser.add_argument("--resolution", type=float, default=0.06)
    parser.add_argument("--check-motion", dest="check_motion", action="store_true")
    parser.add_argument("--no-volume-conservation", action="store_true")
    parser.add_argument("--no-handle-zeta", action="store_true")
    parser.add_argument("--flatten-history", dest="flatten_history", action="store_true")
    parser.add_argument("--degrade-bdf1", dest="degrade_bdf1", action="store_true")
    parser.add_argument("--backend", choices=["gmsh", "tqmesh"], default="gmsh")
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    global BLOB_MESH_CLASS
    BLOB_MESH_CLASS = BlobMesh if args.backend == "gmsh" else BlobMeshTQMesh
    try:
        CASES[args.case](args.outdir, args)
    except BaseException as e:  # noqa: BLE001
        print("PYOOMPH_RAISED %s: %s" % (type(e).__name__, " | ".join(str(e).splitlines())), flush=True)
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
