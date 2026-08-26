from __future__ import annotations
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

"""Topological changes of an axisymmetric free surface: pinch-off and coalescence.

The two pieces here work as a pair.

* :py:class:`AxisymmetricReconnection` is an :py:class:`~pyoomph.generic.codegen.InterfaceEquations`
  attached to the free interface. After each successful Newton solve it hands the current interface
  polylines to :py:func:`pyoomph.meshes.axisymm_topology.detect_and_plan`, which decides -
  morphologically, on the mirrored cross section - whether a neck has thinned below ``rmin`` or two
  fragments have approached to within ``distmin``, and returns a
  :py:class:`~pyoomph.meshes.axisymm_topology.SurgeryPlan` describing the interface *after* the
  event. The plan is parked on the mesh template and a remesh is requested.

* :py:class:`TopologicalChangesGmshTemplate` is the mesh template that consumes the plan. The
  surgery is delivered through the ordinary remeshing-by-recreation path: pyoomph calls
  ``define_geometry`` again, and there :py:meth:`~TopologicalChangesGmshTemplate.get_reconnected_boundaries`
  returns the interface chains and axis spans that the new geometry has to be built from - the
  reconnected ones if an event was planned, the current ones if this is a plain quality remesh. The
  user code that turns them into splines and lines is the same in both cases.

This replaces the previous ``AxisymmetricPinchoffAndCoalescence``, which edited a
:py:class:`~pyoomph.meshes.remesher.Remesher2d`'s point and line entries in place. The reasons for
the clean break, rather than a port:

* the detection was a spline fit of ``r(arclength)`` per interface segment, so it could only see a
  waist *within one* segment and never the merging of two, and it depended on the knot placement of
  the fit;
* the surgery inserted two hand-placed points per new tip and left the volume to chance;
* it needed a ``Remesher2d``, i.e. a geometry reconstructed from the deformed mesh, which is the one
  remeshing path that cannot use the user's own ``define_geometry``.

Coordinate convention throughout: ``x`` is the radial coordinate with the symmetry axis at ``x = 0``,
``y`` is the axial coordinate.
"""

from dataclasses import dataclass, field

import numpy

from ..expressions.generic import Expression, ExpressionNumOrNone, ExpressionOrNum, var, partial_t
from ..generic.codegen import InterfaceEquations, Equations, FiniteElementCodeGenerator
from ..generic.mpi import (get_mpi_any, get_mpi_max, get_mpi_nproc, get_mpi_rank, get_mpi_sum,
                           get_mpi_world_comm, mpi_share_root_failure)
from ..meshes.axisymm_topology import (InterfaceChain, ReconnectionEvent, SurgeryPlan,
                                       detect_and_plan, revolved_volume)
from ..meshes.gmsh import GmshTemplate
from ..meshes.mesh import AnySpatialMesh, InterfaceMesh, MeshFromTemplate2d, MeshFromTemplateBase, Node, Element, AnyMesh
from ..meshes.meshdatacache import MeshDataCache, MeshDataCacheEntry
from ..typings import *

if TYPE_CHECKING:
    from ..meshes.ordering import SortAlongAxis
    from pygmsh.common.line import Line  # type:ignore
    from pygmsh.common.point import Point  # type:ignore
    from pygmsh.common.spline import Spline  # type:ignore
    from ..generic.problem import Problem
    from ..generic.codegen import EquationTree
    from ..meshes.interpolator import BaseMeshToMeshInterpolator


# --------------------------------------------------------------------------------------
# What define_geometry gets to see
# --------------------------------------------------------------------------------------

@dataclass
class ReconnectedChain:
    """One interface polyline of the geometry that is about to be built.

    ``points`` carry the spatial scaling unless ``nondimensional=True`` was requested, so they can be
    fed straight to :py:meth:`~pyoomph.meshes.gmsh.GmshTemplate.point`. ``suggested_sizes`` are
    *always* plain nondimensional floats, because that is what Gmsh's ``size`` argument is.
    """
    points: list[tuple[ExpressionOrNum, ExpressionOrNum]]
    suggested_sizes: list[float]
    end_types: tuple[str, str] = ("axis", "axis")
    #: Old interface chart of each point, or ``None`` for a plain remesh. Handed out for user code;
    #: the transfer itself reads the plan's ``NewChain.zeta`` directly, in
    #: :py:meth:`AxisymmetricReconnection._before_mesh_to_mesh_interpolation`.
    zeta: NPFloatArray | None = None
    #: ``>= 0``: index of the old interface point this point *is*; ``-1``: freshly created by the
    #: surgery. ``None`` for a plain remesh, where every point is an old one.
    origin: NPIntArray | None = None
    #: Whether :py:attr:`points` are plain numbers rather than dimensional expressions.
    nondimensional: bool = False


@dataclass
class ReconnectedBoundaries:
    """The result of :py:meth:`TopologicalChangesGmshTemplate.get_reconnected_boundaries`."""
    #: The free interface, one entry per connected fluid fragment, ordered by ascending ``y``.
    interface_chains: list[ReconnectedChain] = field(default_factory=list)
    #: Axis pieces that belong to the domain the interface bounds, as ``((x0,y0),(x1,y1))`` pairs.
    axis_segments: list[tuple[tuple[ExpressionOrNum, ExpressionOrNum], tuple[ExpressionOrNum, ExpressionOrNum]]] = field(default_factory=list)
    #: Axis pieces that belong to the *other* side of the interface, i.e. the gaps between fragments
    #: plus whatever the opposite phase covered before. Empty unless ``opposite_axis_name`` was given.
    opposite_axis_segments: list[tuple[tuple[ExpressionOrNum, ExpressionOrNum], tuple[ExpressionOrNum, ExpressionOrNum]]] = field(default_factory=list)
    #: The reconnection events behind this geometry; empty for a plain quality remesh.
    events: list[ReconnectionEvent] = field(default_factory=list)
    #: Target volume of each fragment, in the same units as ``points``. Meant for verification.
    fragment_volumes: list[float | ExpressionOrNum] = field(default_factory=list)
    #: Whether a surgery plan was consumed, i.e. whether this is a reconnection or a quality remesh.
    has_plan: bool = False
    #: Whether ``points``/``fragment_volumes`` are nondimensional.
    nondimensional: bool = False


# --------------------------------------------------------------------------------------
# The mesh template
# --------------------------------------------------------------------------------------

class TopologicalChangesGmshTemplate(GmshTemplate):
    """A :py:class:`~pyoomph.meshes.gmsh.GmshTemplate` that can rebuild itself after a pinch-off or
    a coalescence.

    Write ``define_geometry`` so that the remeshing branch takes its interface and axis from
    :py:meth:`get_reconnected_boundaries` instead of from
    :py:meth:`~pyoomph.meshes.mesh.MeshedMeshTemplate.get_boundary_coordinates`::

        class MyMesh(TopologicalChangesGmshTemplate):
            def define_geometry(self):
                if self.is_first_time():
                    ...                                  # the initial geometry
                else:
                    rb = self.get_reconnected_boundaries("liquid/interface", "liquid/axis")
                    for k, chain in enumerate(rb.interface_chains):
                        self.spline_from_chain(chain, "interface")
                    self.lines_from_axis_segments(rb.axis_segments, "axis")
                    ...                                  # plane_surface per fragment

    That branch is entered for a reconnection *and* for an ordinary quality remesh - there is only
    one code path, and :py:meth:`get_reconnected_boundaries` fills in the same structure either way.
    """

    def __init__(self, loaded_from_mesh_file: str | None = None):
        super().__init__(loaded_from_mesh_file)
        # interface mesh name (e.g. "liquid/interface") -> (plan, extra info of the equation)
        self._pending_surgery_plans: dict[str, tuple[SurgeryPlan, dict[str, Any]]] = {}
        #: Two points closer than this *nondimensional* distance are the same Gmsh point. See
        #: :py:meth:`point`. Nondimensional geometries are O(1) by construction, so an absolute
        #: value works here and keeps the lookup independent of the order the points arrive in.
        self.point_snap_tolerance: float = 1e-9
        # Quantised coordinates -> point, so that the tolerant lookup in point() stays O(1). A plain
        # scan over _pointhash would be O(N^2) over an interface of a few thousand points.
        self._snap_buckets: dict[tuple[int, int, int], list["Point"]] = {}
        # Whether the last get_reconnected_boundaries() returned nondimensional coordinates; the
        # convenience builders need to know, since GmshTemplate.point() divides by the spatial scale.
        self._recon_nondimensional: bool = False

    # -- pending plans ---------------------------------------------------------------------------

    def _set_pending_surgery_plan(self, interface_name: str, plan: SurgeryPlan, extra: dict[str, Any]) -> None:
        self._pending_surgery_plans[interface_name] = (plan, extra)

    def _clear_pending_surgery_plan(self, interface_name: str) -> None:
        self._pending_surgery_plans.pop(interface_name, None)

    def has_pending_surgery_plan(self, interface_name: str | None = None) -> bool:
        """Whether a reconnection is waiting to be built into the next mesh."""
        if interface_name is None:
            return len(self._pending_surgery_plans) > 0
        return interface_name in self._pending_surgery_plans

    # -- point deduplication ---------------------------------------------------------------------

    def point(self, x: ExpressionOrNum, y: ExpressionOrNum = 0.0, z: ExpressionOrNum = 0.0, size: ExpressionNumOrNone = None, *, name: str | None = None, consider_spatial_scale: bool | None = None) -> "Point":
        """As :py:meth:`~pyoomph.meshes.gmsh.GmshTemplate.point`, but merges *nearly* coincident points.

        The base class keys its point cache on the exact ``(x,y,z)`` doubles, which is enough as long
        as every point comes out of the same arithmetic. Here they do not: the interface tip of a
        surgered chain is computed by :py:mod:`pyoomph.meshes.axisymm_topology` and then dimensionalised
        and divided by the spatial scale again on the way in, while the very same corner may be
        written out by hand in ``define_geometry`` as a literal. Two doubles that differ in the last
        bit become two Gmsh points, and a curve loop through both of them is not closed - Gmsh then
        fails with a message about a non-manifold or unbounded surface, far from the cause. Hence a
        tolerant lookup, at a tolerance far below any mesh size and far above the round-off.
        """
        if self._geom is None:
            # Outside define_geometry the base class raises; let it produce that message.
            return super().point(x, y, z, size, name=name, consider_spatial_scale=consider_spatial_scale)
        if consider_spatial_scale is None:
            consider_spatial_scale = self.consider_spatial_scale
        nd: list[float] = []
        for c in (x, y, z):
            if consider_spatial_scale:
                c = c / self.get_problem().get_scaling("spatial")
            if isinstance(c, Expression):
                c = c.float_value()
            nd.append(float(c))
        key = (nd[0], nd[1], nd[2])
        existing = self._pointhash.get(key)
        if existing is not None:
            return existing
        q = self.point_snap_tolerance
        bucket: tuple[int, int, int] | None = None
        if q > 0.0:
            bx, by, bz = (int(numpy.floor(v / q)) for v in nd)
            bucket = (bx, by, bz)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for p in self._snap_buckets.get((bx + dx, by + dy, bz + dz), ()):
                            if abs(p.x[0] - nd[0]) <= q and abs(p.x[1] - nd[1]) <= q \
                                    and abs(p.x[2] - nd[2]) <= q:
                                return p
        res = super().point(nd[0], nd[1], nd[2], size, name=name, consider_spatial_scale=False)
        if bucket is not None:
            self._snap_buckets.setdefault(bucket, []).append(res)
        return res

    def _reset(self):
        super()._reset()
        self._snap_buckets = {}

    # -- the boundaries of the next mesh ---------------------------------------------------------

    def get_reconnected_boundaries(self, interface_name: str, axis_name: str, opposite_axis_name: str | None = None, nondimensional: bool = False) -> ReconnectedBoundaries:
        """The interface and axis of the mesh that is about to be created.

        May only be called from ``define_geometry`` while remeshing. If an
        :py:class:`AxisymmetricReconnection` has planned a pinch-off or coalescence, the returned
        chains and axis spans describe the *reconnected* geometry; otherwise they describe the
        current one, so that the same ``define_geometry`` branch also serves a plain quality remesh.

        Args:
            interface_name: The free interface, as ``"domain/boundary"``, e.g. ``"liquid/interface"``.
                This is also the key the surgery plan was filed under.
            axis_name: The symmetry axis on the side of ``interface_name``, as ``"domain/boundary"``.
            opposite_axis_name: The symmetry axis of the opposite phase, if there is one. Only then
                is :py:attr:`~ReconnectedBoundaries.opposite_axis_segments` filled.
            nondimensional: Return plain numbers instead of coordinates carrying the spatial scale.
        """
        self._assert_within_define_geometry("get_reconnected_boundaries()")
        if not self.is_remeshing():
            raise RuntimeError("get_reconnected_boundaries() describes the mesh that REPLACES the "
                               "current one, so it only means something while remeshing. Guard it "
                               "with 'if self.is_first_time(): ... else: ...'.")

        # Unconditional, and in the same order on every rank: get_boundary_coordinates() is
        # collective on a distributed mesh (see MeshedMeshTemplate).
        old_interface = self.get_boundary_coordinates(interface_name, sort_along_axis="y+", nondimensional=True)
        old_axis = self.get_boundary_coordinates(axis_name, sort_along_axis="y+", nondimensional=True)
        old_opposite_axis = None
        if opposite_axis_name is not None:
            old_opposite_axis = self.get_boundary_coordinates(opposite_axis_name, sort_along_axis="y+", nondimensional=True)

        SS: ExpressionOrNum = 1 if nondimensional else self.get_problem().get_scaling("spatial")
        self._recon_nondimensional = nondimensional
        entry = self._pending_surgery_plans.get(interface_name)
        res = ReconnectedBoundaries(has_plan=entry is not None, nondimensional=nondimensional)

        if entry is not None:
            plan = entry[0]
            for nc in plan.new_chains:
                res.interface_chains.append(ReconnectedChain(
                    points=[(float(p[0]) * SS, float(p[1]) * SS) for p in nc.points],
                    suggested_sizes=[float(s) for s in nc.sizes],
                    end_types=tuple(nc.end_types), zeta=nc.zeta, origin=nc.origin,  # type:ignore[arg-type]
                    nondimensional=nondimensional))
            inside = [(float(a), float(b)) for a, b in numpy.asarray(plan.axis_spans_inside).reshape(-1, 2)]
            res.events = list(plan.events)
            res.fragment_volumes = [float(v) * SS ** 3 for v in plan.fragment_volumes_after]
        else:
            size_factor = self._interface_element_size_factor(interface_name)
            for seg in old_interface:
                pts = numpy.array([[float(x), float(y)] for x, y in seg], dtype=float)
                res.interface_chains.append(ReconnectedChain(
                    points=[(float(p[0]) * SS, float(p[1]) * SS) for p in pts],
                    suggested_sizes=[float(s) for s in size_factor * _polyline_spacing(pts)],
                    end_types=_end_types_of(pts, _interface_extent(old_interface), _AXIS_TOL_FALLBACK),
                    nondimensional=nondimensional))
            # The axis segments are straight, so their two extreme points carry all the information.
            inside = _merge_spans([(float(seg[0][1]), float(seg[-1][1])) for seg in old_axis])
            res.fragment_volumes = [float(revolved_volume(_closed_half_section(
                numpy.array([[float(x), float(y)] for x, y in seg], dtype=float)))) * SS ** 3
                for seg in old_interface]

        res.axis_segments = [((0.0 * SS, a * SS), (0.0 * SS, b * SS)) for a, b in inside]

        if old_opposite_axis is not None:
            # What the opposite phase covers afterwards is everything the axis was covered by before
            # - by either phase - minus what this phase covers now. Building it as a complement
            # rather than reading the old opposite axis is what makes the freshly opened gap of a
            # pinch-off appear on the opposite side without the caller having to special-case it.
            covered = _merge_spans([(float(seg[0][1]), float(seg[-1][1])) for seg in old_opposite_axis]
                                   + [(float(seg[0][1]), float(seg[-1][1])) for seg in old_axis])
            res.opposite_axis_segments = [((0.0 * SS, a * SS), (0.0 * SS, b * SS))
                                          for a, b in _subtract_spans(covered, inside)]
        return res

    def _interface_element_size_factor(self, interface_name: str) -> float:
        """How many polyline points one interface element spans - 2 for C2, 1 for C1.

        ``get_boundary_coordinates`` walks the interface *nodes*, so on a second-order interface the
        spacing between consecutive points is half an element. Without this the suggested sizes halve
        at every remesh and the mesh grows geometrically.

        Collective on a distributed mesh, and it has to be: the answer is read off this rank's first
        interface element, and a rank whose partition holds none of that interface has no element to
        ask. It would then report 1 while the others report 2 - and since only rank 0's ``.msh`` file
        becomes the mesh, a rank 0 without interface elements would halve the mesh size of the whole
        problem at every remesh. Reached by every rank because the branch it sits in is decided by
        the pending plan, which stage 7 makes unanimous.
        """
        local = 0
        try:
            mesh = self.get_problem().get_mesh(interface_name)
            if mesh.nelement() > 0:
                local = max(1, mesh.element_pt(0).nnode() - 1)
        except Exception:
            local = 0
        if get_mpi_nproc() > 1 and bool(self.get_problem().is_distributed()):
            local = int(get_mpi_max(local))
        return float(max(1, local))

    # -- convenience builders --------------------------------------------------------------------

    def spline_from_chain(self, chain: ReconnectedChain, name: str, size: float | Callable[[ExpressionOrNum, ExpressionOrNum, float], float] | None = None) -> "Spline | Line":
        """Turn one :py:class:`ReconnectedChain` into a named spline (a line if it has two points).

        Args:
            chain: One entry of :py:attr:`ReconnectedBoundaries.interface_chains`.
            name: Boundary name of the created curve.
            size: ``None`` uses the chain's own :py:attr:`~ReconnectedChain.suggested_sizes`, a float
                a uniform size, and a callable is asked ``(x, y, suggested) -> size`` per point.
        """
        pts: list["Point"] = []
        for (x, y), suggested in zip(chain.points, chain.suggested_sizes):
            if size is None:
                sz: float | None = suggested
            elif callable(size):
                sz = float(size(x, y, suggested))
            else:
                sz = float(size)
            pts.append(self.point(x, y, size=sz, consider_spatial_scale=not chain.nondimensional))
        if len(pts) < 2:
            raise RuntimeError("An interface chain of " + str(len(pts)) + " point(s) cannot become a curve")
        if len(pts) == 2:
            # A two-point spline would be handed to CurvedEntityCatmullRomSpline, which needs more
            # than its two endpoints to mean anything; a straight line is what it would describe.
            res = self.line(pts[0], pts[1], name=name)
            assert res is not None
            return res
        return self.spline(pts, name=name)

    def lines_from_axis_segments(self, segments: Sequence[tuple[tuple[ExpressionOrNum, ExpressionOrNum], tuple[ExpressionOrNum, ExpressionOrNum]]], name: str, size: float | None = None) -> list["Line"]:
        """Turn axis spans into named two-point lines. An empty list of segments is not an error:
        after a coalescence a phase can lose its share of the axis entirely."""
        res: list["Line"] = []
        css = not self._recon_nondimensional
        for p0, p1 in segments:
            a = self.point(p0[0], p0[1], size=size, consider_spatial_scale=css)
            b = self.point(p1[0], p1[1], size=size, consider_spatial_scale=css)
            if a is b:
                raise RuntimeError("An axis segment of zero length was requested for boundary '" + name + "'")
            ln = self.line(a, b, name=name)
            assert ln is not None
            res.append(ln)
        return res


# --------------------------------------------------------------------------------------
# The interface equations
# --------------------------------------------------------------------------------------

#: Fallback relative on-axis tolerance where no :py:class:`AxisymmetricReconnection` supplies one.
_AXIS_TOL_FALLBACK = 1e-7


class AxisymmetricReconnection(InterfaceEquations):
    """Detects axisymmetric pinch-off and coalescence and requests the corresponding remesh.

    Attach it to the free interface of an axisymmetric problem whose bulk mesh comes from a
    :py:class:`TopologicalChangesGmshTemplate`::

        eqs += AxisymmetricReconnection(rmin=5*micro*meter, distmin=5*micro*meter) @ "interface"

    Both thresholds are *physical* lengths (they may carry units) and either may be ``None`` to
    switch off that kind of event:

    * a neck whose minimal interface radius drops below ``rmin`` pinches off;
    * two fragments whose axial tip-to-tip gap drops below ``distmin`` coalesce.

    The event itself is not applied here. It is handed to the mesh template as a
    :py:class:`~pyoomph.meshes.axisymm_topology.SurgeryPlan` and built into the next mesh by
    ``define_geometry``, i.e. by the ordinary remeshing-by-recreation path.
    """

    def __init__(self, rmin: ExpressionNumOrNone = None, distmin: ExpressionNumOrNone = None, *,
                 volume_conservation: bool = True, volume_tolerance: float = 1e-9,
                 cap_window_factor: float = 6.0, buffer_resolution: int = 8,
                 cap_spacing_factor: float = 0.7,
                 check_mesh_motion_direction: bool = True, overlap_reject_factor: float | None = 0.2,
                 coalescence_arm_factor: float = 3.0, axis_tolerance: float = 1e-7,
                 allow_fragment_removal: bool = True, segment_jump_offset: float = 1.0,
                 handle_zeta: bool = True) -> None:
        super().__init__()
        #: Minimal interface radius below which a neck pinches off. ``None`` disables pinch-off.
        self.rmin = rmin
        #: Axial tip-to-tip gap below which two fragments coalesce. ``None`` disables coalescence.
        self.distmin = distmin
        self.volume_conservation = volume_conservation
        self.volume_tolerance = volume_tolerance
        self.cap_window_factor = cap_window_factor
        self.buffer_resolution = buffer_resolution
        self.cap_spacing_factor = cap_spacing_factor
        #: Refuse to coalesce a gap whose two tips are moving apart.
        self.check_mesh_motion_direction = check_mesh_motion_direction
        #: Reject a Newton step that brings two tips closer than this multiple of ``distmin``, so
        #: that an adaptive time stepper cuts ``dt`` instead of letting them pass through each
        #: other. ``None`` switches the guard off.
        self.overlap_reject_factor = overlap_reject_factor
        #: The guard above is only armed once some gap is below this multiple of ``distmin``.
        self.coalescence_arm_factor = coalescence_arm_factor
        #: A point counts as sitting on the axis if ``|x|`` is below this times the size of the interface.
        self.axis_tolerance = axis_tolerance
        self.allow_fragment_removal = allow_fragment_removal
        self.segment_jump_offset = segment_jump_offset
        #: Carry the interface fields across the event by writing the plan's zeta chart onto both the
        #: old and the new interface (see :py:meth:`_before_mesh_to_mesh_interpolation`). Switch off
        #: to leave the transfer to whatever the geometry alone can match.
        self.handle_zeta = handle_zeta

        self._datacache = MeshDataCache(tesselate_tri=False, nondimensional=True)
        self._rmin_nd: float | None = None
        self._distmin_nd: float | None = None
        self._nondim_done: bool = False
        # Whether the last after_newton_solve saw a gap close enough for the overlap guard to be
        # worth its (cheap, but per-Newton-step) cost.
        self._armed: bool = False
        self._warned_about_distmin: bool = False
        self._last_plan: SurgeryPlan | None = None
        # (template, interface name) of the plan that is waiting to be built, so that
        # after_remeshing can clear it without having to resolve the tree again.
        self._pending_on: tuple["TopologicalChangesGmshTemplate", str] | None = None
        # (bulk mesh, boundary index) pairs on the NEW meshes whose boundary-coordinate flag this
        # handler raised, so that after_remeshing can put it back. See _install_zeta_chart.
        self._zeta_flags_to_restore: list[tuple[Any, int]] = []
        # The NEW interface meshes this handler has claimed the zeta chart of for the duration of one
        # remesh, likewise released by after_remeshing.
        self._zeta_override_meshes: list[InterfaceMesh] = []

    # -- local functions -------------------------------------------------------------------------

    def define_additional_functions(self):
        if self.check_mesh_motion_direction:
            # ALE=False is essential and is what the legacy handler got wrong. With the default
            # ALE="auto", partial_t(var("mesh_y")) on a moving mesh is
            #     d(mesh_y)/dt - u_mesh . grad(mesh_y) = u_y - u_y = 0,
            # identically, so the old gate compared 0 > 0 and never fired: it silently blocked every
            # pinch-off (which required v_r < 0) and never suppressed a coalescence. partial_t(...,
            # ALE=False) is the raw nodal time derivative, i.e. the mesh velocity - the same thing
            # mesh_velocity() is a shorthand for.
            #
            # Both components, although only the axial one is read: the radial one is what a user
            # looking at the interface output would expect next to it, and it costs nothing.
            self.add_local_function("_topo_mesh_v_r", partial_t(var("mesh_x"), ALE=False))
            self.add_local_function("_topo_mesh_v_z", partial_t(var("mesh_y"), ALE=False))
        return super().define_additional_functions()

    # -- parameters ------------------------------------------------------------------------------

    def ensure_nondimensional_distance_parameters(self, problem: "Problem") -> None:
        """Convert ``rmin``/``distmin`` to the problem's spatial scale, once."""
        if self._nondim_done:
            return
        SS = problem.get_scaling("spatial")
        self._rmin_nd = None if self.rmin is None else float(self.rmin / SS)
        self._distmin_nd = None if self.distmin is None else float(self.distmin / SS)
        self._nondim_done = True
        if self._rmin_nd is not None and self._distmin_nd is not None and self._distmin_nd > self._rmin_nd \
                and not self._warned_about_distmin:
            self._warned_about_distmin = True
            if get_mpi_rank() == 0:
                print("WARNING: AxisymmetricReconnection has distmin > rmin. A neck thinner than distmin "
                  "is bridged by the coalescence criterion before the pinch-off criterion can open "
                  "it, so pinch-off will not be reported there. Choose distmin < rmin.")

    # -- helpers ---------------------------------------------------------------------------------

    def get_boundary_line_segments(self, name: str, sort_along_axis: "SortAlongAxis | None" = "y+") -> list[NPFloatArray]:
        """The nondimensional polylines of a boundary of the *parent* domain, ordered and oriented.

        The mesh-based replacement of the old helper of the same name, which read the point entries
        of a :py:class:`~pyoomph.meshes.remesher.Remesher2d`. There is no such remesher on the
        recreation path, and the mesh is the more direct source anyway.

        Local: on a distributed mesh this describes *this rank's* partition of the boundary, not the
        whole of it. The detection does not go through here - it merges, see
        :py:meth:`_needs_merged_interface`; use
        :py:meth:`~pyoomph.meshes.mesh.MeshedMeshTemplate.get_boundary_coordinates` if the whole
        boundary is what is wanted.
        """
        from ..meshes.ordering import sort_line_segments
        domain = self.get_current_code_generator().get_full_name().split("/")[0]
        mesh = self.get_current_code_generator().get_problem().get_mesh(domain + "/" + name)
        data = self._datacache.get_data(mesh)  # type:ignore[arg-type]
        assert data is not None  # a local (non-global) cache always returns data
        segs, _ = data.get_interface_line_segments()
        pts = data.get_coordinates()
        segs = sort_line_segments(pts, segs, sort_along_axis=sort_along_axis, whom="get_boundary_line_segments()")
        return [numpy.array([[pts[0, i], pts[1, i]] for i in seg], dtype=float) for seg in segs]

    def get_interface_and_axisymm_name(self) -> tuple[str, str]:
        """The short names of this interface and of the symmetry axis of the same domain.

        The axis is found the way the legacy handler found it - it is the boundary all of whose
        points sit at ``r = 0`` - but from the bulk mesh rather than from a remesher's line entries,
        and with a tolerance relative to the size of the domain instead of a hard-coded ``1e-7``.

        Collective on a distributed problem: see :py:meth:`_interface_and_axis_name`.
        """
        return self._interface_and_axis_name(self.get_current_code_generator())

    def get_opposite_and_opposite_axisymm_name(self) -> tuple[str | None, str | None]:
        """The same, but for the domain on the other side of this interface (``None`` if there is none)."""
        opp = self.get_current_code_generator()._get_opposite_interface()
        if opp is None:
            return None, None
        assert isinstance(opp, FiniteElementCodeGenerator)
        return self._interface_and_axis_name(opp)

    def _interface_and_axis_name(self, cg: FiniteElementCodeGenerator) -> tuple[str, str]:
        """Which boundary of ``cg``'s domain is the symmetry axis, decided over the whole mesh.

        Collective on a distributed problem, and it has to be. Both halves of the test are about the
        boundary as a whole: the extent it is measured against, and "every node of it sits at r=0".
        A partition holding no node of the axis answers "not a candidate" and one holding only the
        stretch of some other boundary that happens to run along r=0 answers "candidate" - so the
        ranks would name different boundaries (or none), and the axis name travels into the transfer
        configuration of the remesh. The reductions are unconditional, so every rank runs the same
        number of them and a disagreement becomes an error on all of them rather than a hang.

        ``get_boundary_names()`` is ordered by boundary index, which is assigned per mesh from the
        template, i.e. identically everywhere.
        """
        splt = cg.get_full_name().split("/")
        bulk = cg.get_problem().get_mesh(splt[0])
        assert isinstance(bulk, MeshFromTemplateBase)
        collective = get_mpi_nproc() > 1 and bool(cg.get_problem().is_distributed())
        extent = 0.0
        for n in bulk.nodes():
            extent = max(extent, abs(n.x(0)), abs(n.x(1)))
        if collective:
            extent = float(get_mpi_max(extent))
        tol = max(self.axis_tolerance * extent, 1e-300)
        candidates: list[str] = []
        for bname in bulk.get_boundary_names():
            if not bname or bname == splt[-1]:
                continue
            bi = bulk.get_boundary_index(bname)
            nodes = [bulk.boundary_node_pt(bi, i) for i in range(bulk.nboundary_node(bi))]
            non_axis, total = sum(1 for n in nodes if abs(n.x(0)) >= tol), len(nodes)
            if collective:
                # A ratio test, so the halo copies a node may have on several ranks cannot change the
                # verdict - a node is on the axis or it is not, wherever it is counted.
                non_axis, total = int(get_mpi_sum(non_axis)), int(get_mpi_sum(total))
            if total > 0 and non_axis == 0:
                candidates.append(bname)
        if len(candidates) == 0:
            raise RuntimeError("Cannot find the axis of symmetry of domain '" + splt[0] + "': no "
                               "boundary of it has all its nodes at r=0")
        if len(candidates) > 1:
            raise RuntimeError("Cannot identify the axis of symmetry of domain '" + splt[0] +
                               "'. It could be any of " + ", ".join(sorted(candidates)))
        return splt[-1], candidates[0]

    # -- detection -------------------------------------------------------------------------------

    def _validate_and_get_template(self) -> tuple[InterfaceMesh, "TopologicalChangesGmshTemplate"]:
        mesh = self.get_my_domain()._mesh  # type:ignore[attr-defined]
        if not isinstance(mesh, InterfaceMesh):
            raise RuntimeError("Please attach AxisymmetricReconnection to an interface, not to a bulk domain")
        if mesh.get_element_dimension() != 1:
            raise RuntimeError("AxisymmetricReconnection only works on an InterfaceMesh of elemental dimension 1")
        bulk = mesh.get_bulk_mesh()
        if not isinstance(bulk, MeshFromTemplateBase):
            raise RuntimeError("AxisymmetricReconnection only works when attached to a 2d bulk mesh "
                               "generated from a mesh template")
        template = bulk._templatemesh  # type:ignore[attr-defined]
        if not isinstance(template, TopologicalChangesGmshTemplate):
            raise RuntimeError(
                "The bulk mesh of an AxisymmetricReconnection must come from a "
                "TopologicalChangesGmshTemplate, not from a " + type(template).__name__ + ". The "
                "surgery is delivered by re-running define_geometry, which needs that class's "
                "get_reconnected_boundaries() to describe the reconnected geometry.")
        return mesh, template

    def _needs_merged_interface(self, mesh: InterfaceMesh) -> bool:
        """Whether the detection has to run on the globally merged interface rather than this rank's.

        The morphology is a statement about the WHOLE interface - where its thinnest waist is, which
        fragments face each other across which gap - so a rank's partition of it is not a piece of
        the answer but a different, truncated geometry, cut wherever the partition happens to run.

        Collective, and answered identically on every rank, exactly as in
        :py:meth:`pyoomph.meshes.zeta.AssignZetaCoordinatesBase._needs_merged_interface`: the gate
        decides whether this rank enters the merge, so it is agreed rather than trusted locally.
        Serial and ``mpirun`` without ``--distribute`` answer False without any communication at all,
        and take the local path unchanged.
        """
        problem = self.get_current_code_generator().get_problem()
        if problem is None or not problem.is_distributed() or get_mpi_nproc() <= 1:
            return False
        from ..meshes.meshdatamerge import needs_merging
        return get_mpi_any(needs_merging(mesh))

    def _merged_interface_data(self, mesh: InterfaceMesh) -> "MeshDataCacheEntry | None":
        """The whole interface, merged onto rank 0 (``None`` elsewhere). Collective.

        The problem-wide cache is flushed first because the merge reads each rank's LOCAL entry out
        of it, and nothing invalidates that between Newton steps - ``actions_after_newton_solve``
        clears it only after the equations' hooks have run. A stale entry would describe the
        interface as it was before the step, which is precisely what both callers must not see.
        """
        problem = self.get_current_code_generator().get_problem()
        problem.invalidate_cached_mesh_data()
        return problem.get_cached_mesh_data(mesh, nondimensional=True, tesselate_tri=False,
                                            global_mesh=True)

    def _chains_and_tips(self, mesh: InterfaceMesh, with_velocities: bool = True) -> tuple[list[InterfaceChain], list[dict[str, float]]]:
        """Interface polylines for :py:func:`detect_and_plan`, plus the axial tip data for the gate,
        read from this rank's own mesh data."""
        data = self._datacache.get_data(mesh)
        assert data is not None  # only a global-mesh cache returns None, and this one is local
        return self._chains_and_tips_from_data(data, with_velocities)

    def _chains_and_tips_from_data(self, data: "MeshDataCacheEntry", with_velocities: bool = True) -> tuple[list[InterfaceChain], list[dict[str, float]]]:
        """The same, from a given mesh-data entry - the local one, or the merged one on rank 0.

        Touches no node, so it serves both: the merged entry's point indices address the nodes of the
        whole interface and no rank's own node numbering.

        ``with_velocities=False`` skips evaluating the mesh velocity, which the overlap guard does not
        need and would otherwise pay for on every Newton step."""
        segments, ninter = data.get_interface_line_segments()
        coords = data.get_coordinates()
        if len(segments) == 0:
            return [], []
        # The polylines run over the interface *nodes*, so on a C2 interface consecutive points are
        # half an element apart. The size a mesh generator wants is the element size, hence the
        # factor - without it every remesh halves the mesh size and the mesh grows without bound.
        size_factor = float(ninter + 1)
        allidx = [i for seg in segments for i in seg]
        extent = float(numpy.hypot(numpy.ptp(coords[0, allidx]), numpy.ptp(coords[1, allidx])))
        tol = max(self.axis_tolerance * extent, 1e-300)

        vz: NPFloatArray | None = None
        if self.check_mesh_motion_direction and with_velocities:
            vz = data.get_data("_topo_mesh_v_z")
            if vz is None:
                raise RuntimeError("AxisymmetricReconnection: the local function '_topo_mesh_v_z' is "
                                   "not available although check_mesh_motion_direction is on")

        chains: list[InterfaceChain] = []
        tips: list[dict[str, float]] = []
        for seg in segments:
            idx = list(seg)
            if coords[1, idx[0]] > coords[1, idx[-1]]:
                idx = list(reversed(idx))
            pts = numpy.array([[coords[0, i], coords[1, i]] for i in idx], dtype=float)
            ends = ("axis" if abs(pts[0, 0]) < tol else "fixed",
                    "axis" if abs(pts[-1, 0]) < tol else "fixed")
            chains.append(InterfaceChain(points=pts, sizes=size_factor * _polyline_spacing(pts),
                                         end_types=ends))
            tips.append({"z_lo": float(pts[0, 1]), "z_hi": float(pts[-1, 1]),
                         "axis_lo": 1.0 if ends[0] == "axis" else 0.0,
                         "axis_hi": 1.0 if ends[1] == "axis" else 0.0,
                         "vz_lo": float(vz[idx[0]]) if vz is not None else 0.0,
                         "vz_hi": float(vz[idx[-1]]) if vz is not None else 0.0})
        order = sorted(range(len(chains)), key=lambda k: tips[k]["z_lo"])
        return [chains[k] for k in order], [tips[k] for k in order]

    @staticmethod
    def _gaps(tips: list[dict[str, float]]) -> list[tuple[float, float, bool]]:
        """``(gap, z_center, separating)`` for each consecutive pair of axial tips."""
        res: list[tuple[float, float, bool]] = []
        for lo, hi in zip(tips, tips[1:]):
            if lo["axis_hi"] < 0.5 or hi["axis_lo"] < 0.5:
                continue  # not two free axial tips facing each other
            res.append((hi["z_lo"] - lo["z_hi"], 0.5 * (hi["z_lo"] + lo["z_hi"]),
                        hi["vz_lo"] - lo["vz_hi"] > 0.0))
        return res

    def _detect_from_data(self, data: "MeshDataCacheEntry") -> dict[str, Any]:
        """Run the whole detection on one mesh-data entry and return the broadcastable result.

        Everything in the returned dictionary is numpy or builtins - a
        :py:class:`~pyoomph.meshes.axisymm_topology.SurgeryPlan` is plain arrays and dataclasses -
        so on a distributed mesh this runs on rank 0 alone and the result is broadcast verbatim.
        Nothing here touches a node or a mesh, which is what makes that possible.
        """
        none_found: dict[str, Any] = {"armed": False, "plan": None, "table": None,
                                      "distmin_nd": self._distmin_nd}
        chains, tips = self._chains_and_tips_from_data(data)
        if not chains:
            return none_found
        gaps = self._gaps(tips)

        distmin_for_call = self._distmin_nd
        armed = False
        if self._distmin_nd is not None:
            armed = any(g < self.coalescence_arm_factor * self._distmin_nd for g, _, _ in gaps)
            if self.check_mesh_motion_direction:
                closing = [g for g in gaps if g[0] < self._distmin_nd]
                if closing and all(sep for _, _, sep in closing):
                    # Every candidate gap is opening, so there is nothing to detect: skip the
                    # coalescence criterion outright rather than let the closing operation bridge it.
                    distmin_for_call = None
        nothing: dict[str, Any] = {"armed": armed, "plan": None, "table": None,
                                   "distmin_nd": distmin_for_call}
        if self._rmin_nd is None and distmin_for_call is None:
            return nothing

        plan = detect_and_plan(chains, self._rmin_nd, distmin_for_call,
                               buffer_resolution=self.buffer_resolution,
                               cap_window_factor=self.cap_window_factor,
                               cap_spacing_factor=self.cap_spacing_factor,
                               volume_conservation=self.volume_conservation,
                               volume_tolerance=self.volume_tolerance,
                               allow_fragment_removal=self.allow_fragment_removal,
                               segment_jump_offset=self.segment_jump_offset)
        if plan is None:
            return nothing

        if self.check_mesh_motion_direction and gaps:
            # Conservative on purpose: a plan is a single consistent description of the WHOLE new
            # interface - fragments, spliced points, volume targets - so a coalescence that should
            # not have happened cannot be edited out of it. Dropping the plan costs at most a delay
            # of one step for any other event it contained, since the next solve re-detects it.
            for ev in plan.events:
                if ev.kind != "coalescence":
                    continue
                g, _, sep = min(gaps, key=lambda t: abs(t[1] - ev.z_center))
                if sep:
                    print("Discarding a reconnection plan: its coalescence at z=" + repr(ev.z_center) +
                          " joins two tips that are moving apart.")
                    return nothing

        # detect_and_plan() normalised the chains in place and filled their zeta, and those points
        # ARE the nondimensional positions of the current interface nodes - the chains were built
        # from this very mesh (or, distributed, from the merged copy of it, which carries the very
        # same coordinates). So this is the old chart, addressable by position, and it is what
        # _before_mesh_to_mesh_interpolation writes back onto the old interface. Plain numpy, so that
        # it can be broadcast along with the plan.
        table = numpy.column_stack([
            numpy.vstack([c.points for c in chains]),
            numpy.concatenate([numpy.asarray(c.zeta, dtype=float) for c in chains])])
        return {"armed": armed, "plan": plan, "table": table, "distmin_nd": distmin_for_call}

    def after_newton_solve(self) -> None:
        mesh, template = self._validate_and_get_template()
        problem = self.get_current_code_generator().get_problem()
        self.ensure_nondimensional_distance_parameters(problem)
        iname = self.get_current_code_generator().get_full_name()
        # A plan that was never built (a remesh suppressed during arclength continuation, say) refers
        # to interface coordinates that have since moved on, so it must not survive this solve.
        template._clear_pending_surgery_plan(iname)
        self._pending_on = None
        # Disarmed before any early return below. The flag gates a COLLECTIVE guard on every Newton
        # step (see before_newton_convergence_check), so a rank that kept a stale True while the
        # others went False would walk into a broadcast alone and hang the run. It used to survive a
        # solve that found no interface at all, which is that same asymmetry serially.
        self._armed = False
        if self._rmin_nd is None and self._distmin_nd is None:
            return

        self._datacache.clear()
        if not self._needs_merged_interface(mesh):
            local = self._datacache.get_data(mesh)
            assert local is not None  # only a global-mesh cache returns None, and this one is local
            result = self._detect_from_data(local)
        else:
            # Distributed: the morphology has to be read off the whole interface, so merge it
            # (collective, result on rank 0), detect there, and hand the plan to everybody. The plan
            # rather than the merged entry, both because it is the smaller payload and because every
            # rank must end up with the IDENTICAL plan - define_geometry then rebuilds the same
            # geometry everywhere without any further communication.
            comm = get_mpi_world_comm()
            assert comm is not None  # needs_merging implies more than one process
            data = self._merged_interface_data(mesh)
            payload: dict[str, Any] | None = None
            error: BaseException | None = None
            if get_mpi_rank() == 0:
                # Everything from here to the broadcast happens on rank 0 alone, so it has to end for
                # all ranks or for none: detect_and_plan refuses several topologies by raising.
                try:
                    assert data is not None
                    payload = self._detect_from_data(data)
                except BaseException as e:  # noqa: BLE001
                    error = e
            payload = comm.bcast(payload, root=0)
            mpi_share_root_failure(error, context="detecting a topological change on the merged "
                                                  "interface '" + iname + "'")
            assert payload is not None
            result = payload

        self._armed = bool(result["armed"])
        plan = result["plan"]
        if plan is None:
            return
        distmin_for_call = result["distmin_nd"]

        self._last_plan = plan
        # Only informational: the user names the boundaries in get_reconnected_boundaries() anyway,
        # so a geometry whose axis cannot be identified automatically must not fail the detection.
        # Reached by every rank, since the plan they got is the same one - and the lookups are
        # collective on a distributed mesh, so the names come out the same everywhere too.
        try:
            _, axis_name = self.get_interface_and_axisymm_name()
        except RuntimeError:
            axis_name = None
        try:
            _, opp_axis_name = self.get_opposite_and_opposite_axisymm_name()
        except RuntimeError:
            opp_axis_name = None
        try:
            opp_cg = self.get_current_code_generator()._get_opposite_interface()
            opp_interface_name = None if opp_cg is None else opp_cg.get_full_name()
        except RuntimeError:
            opp_interface_name = None
        self._pending_on = (template, iname)
        template._set_pending_surgery_plan(iname, plan, {
            "spatial_scale": problem.get_scaling("spatial"),
            "interface_name": iname, "axis_name": axis_name, "opposite_axis_name": opp_axis_name,
            "opposite_interface_name": opp_interface_name,
            "old_zeta_table": result["table"],
            "rmin_nd": self._rmin_nd, "distmin_nd": distmin_for_call})

        if get_mpi_rank() == 0:
            for ev in plan.events:
                print("Topological change: " + ev.kind + " at nondimensional z=" + repr(ev.z_center))
            print("  fragment volumes (nondimensional): " + repr([float(v) for v in plan.fragment_volumes_before]) +
                  " -> " + repr([float(v) for v in plan.fragment_volumes_after]) +
                  (" (removed " + repr(float(plan.volume_lost_by_removal)) + ")" if plan.volume_lost_by_removal > 0 else ""))

        # This is the template of the parent (bulk) domain: _validate_and_get_template() took it off
        # mesh.get_bulk_mesh(), which is the same mesh the parent code generator points at.
        problem._domains_to_remesh.add(template)

    # -- the overlap guard -----------------------------------------------------------------------

    def _overlap_verdict(self, data: "MeshDataCacheEntry") -> tuple[float, float] | None:
        """The offending ``(gap, z_center)``, or ``None`` if the step is acceptable."""
        _chains, tips = self._chains_and_tips_from_data(data, with_velocities=False)
        limit = self.overlap_reject_factor * self._distmin_nd  # type:ignore[operator]
        for gap, zc, _sep in self._gaps(tips):
            if gap < limit:
                return float(gap), float(zc)
        return None

    def before_newton_convergence_check(self, eqtree: "EquationTree") -> bool:
        """Reject a step that would push two approaching tips through each other.

        Only worth doing once something is actually close (see ``coalescence_arm_factor``): it runs
        per Newton step, so it recomputes the tip positions from the mesh data and does no morphology.

        On a distributed mesh the verdict is a statement about the whole interface - the two tips
        approaching each other are routinely on different ranks - so the armed case merges, decides on
        rank 0 and broadcasts. Every rank therefore returns the same answer *and* enters the same
        collectives. ``_armed`` is what keeps that off the per-Newton-step path away from an event,
        and it is itself broadcast (see :py:meth:`after_newton_solve`), so the ranks cannot disagree
        about whether to enter the merge.
        """
        if self._distmin_nd is None or self.overlap_reject_factor is None or not self._armed:
            return super().before_newton_convergence_check(eqtree)
        mesh = eqtree.get_mesh()
        assert isinstance(mesh, InterfaceMesh)
        self._datacache.clear()  # the positions changed with the Newton step
        if not self._needs_merged_interface(mesh):
            local = self._datacache.get_data(mesh)
            assert local is not None
            offender = self._overlap_verdict(local)
        else:
            comm = get_mpi_world_comm()
            assert comm is not None
            data = self._merged_interface_data(mesh)
            payload: tuple[bool, tuple[float, float] | None] | None = None
            error: BaseException | None = None
            if get_mpi_rank() == 0:
                try:
                    assert data is not None
                    payload = (True, self._overlap_verdict(data))
                except BaseException as e:  # noqa: BLE001
                    error = e
            payload = comm.bcast(payload, root=0)
            mpi_share_root_failure(error, context="checking the interface tip gaps of '"
                                                  + mesh.get_full_name() + "' on the merged interface")
            assert payload is not None
            offender = payload[1]
        if offender is not None:
            if get_mpi_rank() == 0:
                print("Rejecting this step: two interface tips would come within " + repr(offender[0]) +
                      " of each other near z=" + repr(offender[1]) + ", below " +
                      repr(self.overlap_reject_factor * self._distmin_nd) + ".")
            return False
        return super().before_newton_convergence_check(eqtree)

    # -- after the remesh ------------------------------------------------------------------------

    def after_remeshing(self, eqtree: "EquationTree"):
        if self._pending_on is not None:
            self._pending_on[0]._clear_pending_surgery_plan(self._pending_on[1])
            self._pending_on = None
        # The transfer is over, so the one-off chart of _install_zeta_chart has done its job.
        for bulk, bind in self._zeta_flags_to_restore:
            bulk.boundary_coordinate_bool(bind, False)
        self._zeta_flags_to_restore = []
        for imesh in self._zeta_override_meshes:
            imesh._zeta_chart_overridden = False
        self._zeta_override_meshes = []
        # The interface it was armed on no longer exists; the next solve arms it again if it must.
        self._armed = False
        self._datacache.clear()
        return super().after_remeshing(eqtree)

    # -- carrying the interface fields across the event -------------------------------------------

    def _install_zeta_chart(self, old_iface: InterfaceMesh, new_iface: InterfaceMesh,
                            table: NPFloatArray, chains: list[tuple[NPFloatArray, NPFloatArray]]) -> None:
        """Write the surgery's chart onto one interface, old side and new side.

        The two sides are matched differently on purpose. The OLD interface nodes *are* the points the
        detection ran on, so their zeta is looked up exactly. The NEW ones are not: they were produced
        by the mesh generator from a spline through the plan's polyline, so they sag off it by
        O(h^2 * curvature) and only a projection can place them.
        """
        from ..meshes.zeta import (_check_zeta_is_invertible, assign_zetas_by_polyline_projection,
                                   assign_zetas_from_position_table)
        old_bulk = old_iface.get_bulk_mesh()
        assert old_bulk is not None
        # Whether a permanent chart on this interface already exists, i.e. whether the user has a zeta
        # assigner on it: such an assigner charted the OLD mesh long before this remesh, so the answer
        # does not depend on which hook of this remesh ran first. It decides whether the flag raised
        # on the new mesh below has to be taken back again - see below.
        charted_by_someone_else = old_bulk.is_boundary_coordinate_defined(
            old_bulk.get_boundary_index(old_iface.get_name()))
        for mesh, side in ((old_iface, "old"), (new_iface, "new")):
            bulk = mesh.get_bulk_mesh()
            assert bulk is not None
            bind = bulk.get_boundary_index(mesh.get_name())
            if side == "old":
                assign_zetas_from_position_table(mesh, bind, table)
            else:
                assign_zetas_by_polyline_projection(mesh, bind, chains)
                # A user's zeta assigner would otherwise re-chart this mesh from the geometry in
                # after_mapping_on_macro_elements, which runs after this hook and before the transfer
                # - and only on the NEW mesh, so old and new would be read through different
                # parameterisations. Released again in after_remeshing.
                mesh._zeta_chart_overridden = True
                self._zeta_override_meshes.append(mesh)
                if not charted_by_someone_else:
                    # A ONE-OFF chart: the next, ordinary remesh produces none, and the transfer
                    # refuses to run when the old mesh claims a chart the new one does not have
                    # ("Boundary coordinate along ... is defined on the old, but not the new mesh").
                    # So the flag goes back down once this transfer is over.
                    self._zeta_flags_to_restore.append((bulk, bind))
            bulk.boundary_coordinate_bool(bind)
            # The chart is open on every fragment. A period left over from an earlier assignment
            # would make locate_zeta wrap a fresh cap back onto the far end of the interface.
            bulk.set_boundary_zeta_period(bind, 0.0)
            mesh.update_zeta_in_buffer()
            _check_zeta_is_invertible(mesh, bind, "AxisymmetricReconnection (" + side + " interface '"
                                      + mesh.get_full_name() + "')")

    @staticmethod
    def _configure_axis_transfer(interpolator: "BaseMeshToMeshInterpolator", iface_name: str,
                                 axis_name: str | None, extra: dict[str, Any]) -> None:
        """Tell one domain's interpolator how to treat its symmetry axis across the event."""
        if axis_name is None:
            return
        reach = 2.0 * max(extra.get("rmin_nd") or 0.0, extra.get("distmin_nd") or 0.0)
        if reach > 0.0:
            # Nondimensional: boundary_max_dist is compared against distances between nodal
            # positions, which pyoomph stores nondimensionally (src/mesh.cpp, sqrt(mindist) >
            # boundary_max_dist). Both key orders, since the codim-2 pass accepts either.
            interpolator.boundary_max_distances[iface_name + "/" + axis_name] = reach
            interpolator.boundary_max_distances[axis_name + "/" + iface_name] = reach
        # A pinch-off opens a gap in this axis and a coalescence closes one, so the fresh axis nodes
        # have no counterpart on the old axis at all - the old boundary either did not reach there or
        # ran through material that now belongs to the other phase. They do lie in the old BULK of
        # the correct phase, which is the question worth asking. Note this gives up whatever
        # interface-only dofs the axis carries; symmetry boundaries normally carry none.
        interpolator.bulk_locate_boundaries.add(axis_name)

    def _before_mesh_to_mesh_interpolation(self, eqtree: "EquationTree", interpolator: "BaseMeshToMeshInterpolator"):
        """Continue the interface chart across the topological change.

        The old and the new interface are not the same curve near the event, so nothing derived from
        the geometry alone can map one onto the other there: an arclength chart restarts at a fresh
        cap, and a projection would happily pair a point just below a pinch with the interface just
        above it. The plan, on the other hand, knows which new points *are* old ones and where the
        fresh ones came from, and it expresses that as a zeta chart. Writing that chart onto both
        sides turns the transfer of the interface fields into the ordinary zeta-based one.

        Nothing happens on a plain quality remesh - there is no plan then, and the existing machinery
        is right.
        """
        if not self.handle_zeta or self._pending_on is None:
            return super()._before_mesh_to_mesh_interpolation(eqtree, interpolator)
        template, iname = self._pending_on
        entry = template._pending_surgery_plans.get(iname)
        if entry is None:
            return super()._before_mesh_to_mesh_interpolation(eqtree, interpolator)
        plan, extra = entry
        table = extra.get("old_zeta_table")
        if table is None:
            return super()._before_mesh_to_mesh_interpolation(eqtree, interpolator)
        chains = [(nc.points, nc.zeta) for nc in plan.new_chains]

        new_iface = eqtree.get_mesh()
        assert isinstance(new_iface, InterfaceMesh)
        old_iface = interpolator.old.get_mesh(new_iface.get_name())  # type:ignore[union-attr]
        assert isinstance(old_iface, InterfaceMesh)
        self._install_zeta_chart(old_iface, new_iface, table, chains)
        interpolator.zeta_overridden_boundaries.add(new_iface.get_full_name())
        interpolator.zeta_for_interface_fields_only.add(new_iface.get_name())
        self._configure_axis_transfer(interpolator, new_iface.get_name(), extra.get("axis_name"), extra)

        # The other phase, if there is one. The interface is a boundary of BOTH domains, so the gas
        # side of it has its own interface mesh, its own boundary index and its own zeta - and it is
        # transferred by the gas domain's interpolator, which this equation is never dispatched with
        # (one interpolator per domain, the hook called on each domain's own eqtree). remesh_group is
        # how it is reached.
        opp_name = extra.get("opposite_interface_name")
        if opp_name:
            opp_domain, opp_iface_name = opp_name.split("/")[0], opp_name.split("/")[-1]
            opp_interp = interpolator.remesh_group.get(opp_domain)
            if opp_interp is None:
                if get_mpi_rank() == 0:
                    print("WARNING: AxisymmetricReconnection cannot reach the interpolator of the "
                          "opposite domain '" + opp_domain + "', so its side of the interface is "
                          "transferred without the reconnected zeta chart.")
            else:
                opp_new = opp_interp.new.get_mesh(opp_iface_name, return_None_if_not_found=True)  # type:ignore[union-attr]
                opp_old = opp_interp.old.get_mesh(opp_iface_name, return_None_if_not_found=True)  # type:ignore[union-attr]
                if isinstance(opp_new, InterfaceMesh) and isinstance(opp_old, InterfaceMesh):
                    self._install_zeta_chart(opp_old, opp_new, table, chains)
                    opp_interp.zeta_overridden_boundaries.add(opp_new.get_full_name())
                    opp_interp.zeta_for_interface_fields_only.add(opp_iface_name)
                self._configure_axis_transfer(opp_interp, opp_iface_name,
                                              extra.get("opposite_axis_name"), extra)
        return super()._before_mesh_to_mesh_interpolation(eqtree, interpolator)


# --------------------------------------------------------------------------------------
# Small geometric helpers
# --------------------------------------------------------------------------------------

def _polyline_spacing(pts: NPFloatArray) -> NPFloatArray:
    """Mean distance of each polyline point to its neighbours - a robust local element size proxy."""
    if len(pts) < 2:
        return numpy.ones(len(pts), dtype=float)
    d = numpy.linalg.norm(numpy.diff(pts, axis=0), axis=1)
    out = numpy.empty(len(pts), dtype=float)
    out[0] = d[0]
    out[-1] = d[-1]
    if len(pts) > 2:
        out[1:-1] = 0.5 * (d[:-1] + d[1:])
    return out


def _interface_extent(segments: Sequence[Sequence[tuple[float, float]]]) -> float:
    xs = [float(p[0]) for seg in segments for p in seg]
    ys = [float(p[1]) for seg in segments for p in seg]
    if not xs:
        return 1.0
    return float(numpy.hypot(max(xs) - min(xs), max(ys) - min(ys)))


def _end_types_of(pts: NPFloatArray, extent: float, rel_tol: float) -> tuple[str, str]:
    tol = max(rel_tol * extent, 1e-300)
    return ("axis" if abs(pts[0, 0]) < tol else "fixed",
            "axis" if abs(pts[-1, 0]) < tol else "fixed")


def _closed_half_section(pts: NPFloatArray) -> NPFloatArray:
    """Close an interface polyline onto the axis, so that its revolved volume is defined."""
    extra: list[list[float]] = []
    if abs(pts[-1, 0]) > 0.0:
        extra.append([0.0, float(pts[-1, 1])])
    if abs(pts[0, 0]) > 0.0:
        extra.append([0.0, float(pts[0, 1])])
    if not extra:
        return pts
    return numpy.vstack([pts, numpy.array(extra, dtype=float)])


def _merge_spans(spans: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in sorted((min(a, b), max(a, b)) for a, b in spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _subtract_spans(spans: Sequence[tuple[float, float]], holes: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    merged_holes = _merge_spans(holes)
    for a, b in spans:
        cur = a
        for ha, hb in merged_holes:
            if hb <= cur or ha >= b:
                continue
            if ha > cur:
                out.append((cur, ha))
            cur = max(cur, hb)
        if cur < b:
            out.append((cur, b))
    return out


# --------------------------------------------------------------------------------------
# Marking disjunct domains by an integer D0 field
# Can be used e.g. in integral expressions or similar
# --------------------------------------------------------------------------------------

class DisjunctDomainMarker(Equations):
    def __init__(self,name:str,direction:Literal["up","down"]="up") -> None:
        super().__init__()
        self.name=name
        self.direction:Literal["up","down"]=direction # Direction of increasing marker

    def define_fields(self) -> None:
        self.define_scalar_field(self.name,"D0")

    def define_residuals(self) -> None:
        self.set_Dirichlet_condition(self.name,True) # Do not solve for it. Will be set by hand

    def _update_marker(self,mesh:AnySpatialMesh):
        if mesh.nelement()==0:
            return
        marker_index=mesh.element_pt(0).get_jit_code().get_discontinuous_field_index(self.name)
        # Reset all markers
        unhandled_nodes:set[Node]=set()
        unhandled_elems:set[Element]=set()
        nodes2elem:dict[Node,list[Element]]={}
        # Create the look-up tables for unhandles nodes and node->elements map
        for e in mesh.elements():
            e.internal_data_pt(marker_index).set_value(0,-1)
            unhandled_elems.add(e)
            for ni in range(e.nnode()):
                n=e.node_pt(ni)
                unhandled_nodes.add(n)
                if n not in nodes2elem.keys():
                    nodes2elem[n]=[]
                nodes2elem[n].append(e)

        # Start over numbering the droplets
        domain_index=0
        self._max_droplet_index=0
        while len(unhandled_nodes)>0: # We still have nodes which do not belong to any domain
            # Find the node with maximum or minimum y
            ym=1e20*(-1 if self.direction=="down" else 1)
            startnode=None
            for n in unhandled_nodes:
                if (n.x(1)<ym if self.direction=="up" else n.x(1)>ym):
                    startnode=n
                    ym=n.x(1)
            if startnode is None:
                break

            # Flood-fill like algorithm
            checknodes:set[Node]=set([startnode]) # seed the start node
            while len(checknodes)>0:
                nn=checknodes.pop() # get one node out of the bucket
                if nn in unhandled_nodes: # only check further if the node was not handled before
                    unhandled_nodes.remove(nn)
                    for e in nodes2elem[nn]: # go over all elements the node is part of
                        if e in unhandled_elems:
                            e.internal_data_pt(marker_index).set_value(0,domain_index) # mark the element
                            self._max_droplet_index=domain_index
                            unhandled_elems.remove(e)
                            for ni in range(e.nnode()):
                                n=e.node_pt(ni)
                                if n in unhandled_nodes:
                                    checknodes.add(n)
            domain_index+=1


    def on_apply_boundary_conditions(self, mesh: "AnyMesh"):
        assert isinstance(mesh,MeshFromTemplate2d)
        self._update_marker(mesh)
        return super().on_apply_boundary_conditions(mesh)


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
