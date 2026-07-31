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

import math

import numpy

from ..typings import *

from .. import _pyoomph_core as _pyoomph
from ..expressions.generic import Expression, ExpressionNumOrNone, ExpressionOrNum
from .mesh import MeshedMeshTemplate

if TYPE_CHECKING:
    from ..generic.problem import Problem


class TQMeshPoint:
    """
    A point of a :py:class:`TQMeshTemplate` geometry. Created by :py:meth:`TQMeshTemplate.point` - do not construct
    directly, since points are deduplicated by their coordinates.

    The coordinates are stored nondimensionally, i.e. divided by the spatial scale of the problem.
    """
    __slots__ = ("x", "y", "size", "size_range", "name")

    def __init__(self, x: float, y: float, size: float | None, size_range: float | None, name: str | None):
        self.x = x
        self.y = y
        #: Desired element size at this point (nondimensional), or ``None`` to leave it to the size function
        self.size = size
        #: Range over which :py:attr:`size` blends into the surrounding element size, or ``None`` to let TQMesh pick it
        self.size_range = size_range
        self.name = name

    @property
    def coordinate(self) -> tuple[float, float]:
        """The nondimensional (x,y) coordinate."""
        return (self.x, self.y)

    def __repr__(self) -> str:
        return "TQMeshPoint({}, {})".format(self.x, self.y)


class TQMeshCurve:
    """
    A named part of a domain boundary, i.e. a polygonal chain through a sequence of :py:class:`TQMeshPoint`.
    Created by :py:meth:`TQMeshTemplate.line`, :py:meth:`TQMeshTemplate.circle_arc` and friends.

    TQMesh only knows straight boundary edges, so curved boundaries are approximated by such a chain when they are
    created. TQMesh will subdivide each of these edges further to match the local element size, but it will never
    move its end points - so the chain is what determines how well a curved boundary is resolved.

    A curved boundary additionally carries the exact curve as a macro element (:py:attr:`curved_entity`), which is
    what nodes created by later spatial refinement are placed on - they then end up on the true curve rather than on
    the chords of the chain.
    """

    def __init__(self, points: list[TQMeshPoint], name: str | None,
                 curved_entity: "_pyoomph.MeshTemplateCurvedEntityBase | None" = None):
        if len(points) < 2:
            raise RuntimeError("A curve requires at least two points")
        self.points = points
        #: Name of the boundary this curve belongs to. All curves sharing a name form one boundary of the mesh.
        self.name = name
        #: The exact geometry this chain approximates, or ``None`` for a straight line
        self.curved_entity = curved_entity

    def segments(self) -> list[tuple[TQMeshPoint, TQMeshPoint]]:
        """The individual straight segments of this curve."""
        return [(self.points[i], self.points[i + 1]) for i in range(len(self.points) - 1)]

    def __repr__(self) -> str:
        return "TQMeshCurve({}, {} points)".format(self.name, len(self.points))


class _TQMeshLoop:
    # A closed cycle of segments, assembled from the curves passed to plane_surface(). "points" lists the corners in
    # cycle order (without repeating the first one at the end) and "names" the boundary name of the segment starting
    # at the corresponding corner - which is exactly the (coordinates, colors) layout TQMesh expects.
    def __init__(self, points: list[TQMeshPoint], names: list[str]):
        self.points = points
        self.names = names

    def signed_area(self) -> float:
        area = 0.0
        for i, p in enumerate(self.points):
            q = self.points[(i + 1) % len(self.points)]
            area += p.x * q.y - q.x * p.y
        return 0.5 * area

    def reverse(self) -> "_TQMeshLoop":
        # Reversing the cycle direction: the corners come in the opposite order, and each segment is now traversed
        # backwards, i.e. the name that belonged to the segment leaving a corner now belongs to the one entering it.
        pts = list(reversed(self.points))
        names = [self.names[(len(self.points) - 2 - i) % len(self.points)] for i in range(len(self.points))]
        return _TQMeshLoop(pts, names)

    def oriented(self, counter_clockwise: bool) -> "_TQMeshLoop":
        if (self.signed_area() > 0) == counter_clockwise:
            return self
        return self.reverse()

    def contains(self, x: float, y: float) -> bool:
        # Standard ray casting. Only ever called with a point of another loop, i.e. never with one exactly on this
        # loop's boundary, where the answer would be arbitrary.
        inside = False
        n = len(self.points)
        for i in range(n):
            p, q = self.points[i], self.points[(i + 1) % n]
            if (p.y > y) != (q.y > y):
                xs = p.x + (y - p.y) * (q.x - p.x) / (q.y - p.y)
                if x < xs:
                    inside = not inside
        return inside


class _TQMeshCurvedLookup:
    # Finds the curve a generated boundary edge stems from, so that the edge can be given that curve's macro
    # element. TQMesh subdivides the chain segments it is handed, so every boundary edge of the mesh lies within one
    # such segment and is found by the segment closest to its midpoint. Only boundaries that actually have a curved
    # part are looked up, but then all of their curves take part - otherwise a straight piece of the same boundary
    # would be attributed to the closest curved one.
    def __init__(self, curves: list[TQMeshCurve]):
        starts, ends = [], []
        self.entities: list[Any] = []
        for curve in curves:
            for p1, p2 in curve.segments():
                starts.append((p1.x, p1.y))
                ends.append((p2.x, p2.y))
                self.entities.append(curve.curved_entity)
        self.start = numpy.array(starts)
        self.direction = numpy.array(ends) - self.start
        self.length_sqr = numpy.maximum(numpy.sum(self.direction ** 2, axis=1), 1e-300)

    def entity_at(self, x: float, y: float) -> Any:
        offset = numpy.array([x, y]) - self.start
        t = numpy.clip(numpy.sum(offset * self.direction, axis=1) / self.length_sqr, 0.0, 1.0)
        distance = numpy.sum((offset - t[:, None] * self.direction) ** 2, axis=1)
        return self.entities[int(numpy.argmin(distance))]


class _TQMeshSurface:
    # One plane_surface() call: an exterior loop, its holes, and the settings to mesh it with
    def __init__(self, name: str, exterior: _TQMeshLoop, holes: list[_TQMeshLoop], resolution: float | None):
        self.name = name
        self.exterior = exterior
        self.holes = holes
        self.resolution = resolution


class _TQMeshQuadLayer:
    def __init__(self, boundary: str, domain: str | None, n_layers: int, first_height: float, growth_rate: float,
                 start: tuple[float, float] | None, end: tuple[float, float] | None):
        self.boundary = boundary
        self.domain = domain
        self.n_layers = n_layers
        self.first_height = first_height
        self.growth_rate = growth_rate
        self.start = start
        self.end = end


class TQMeshTemplate(MeshedMeshTemplate):
    """
    A two-dimensional mesh created by TQMesh (https://github.com/FloSewn/TQMesh), an advancing front generator for
    triangular and quadrilateral meshes. Specify the geometry in an overridden
    :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.define_geometry` method with :py:meth:`point`, :py:meth:`line`,
    :py:meth:`create_lines`, :py:meth:`circle_arc` and :py:meth:`plane_surface`, in the same way as for a
    :py:class:`~pyoomph.meshes.gmsh.GmshTemplate`.

    Multiple domains are made by calling :py:meth:`plane_surface` several times. Whenever two domains share a
    boundary - i.e. some of their curves carry the same name - TQMesh will discretize that interface only once and
    let the second domain conform to it, so the resulting meshes match node by node. The domains are meshed in the
    order they are created, i.e. the first one to use an interface sets its resolution.

    Coordinates are given dimensionally, i.e. in the spatial unit of the problem (see
    :py:meth:`~pyoomph.generic.problem.Problem.set_scaling`), whereas all element sizes are nondimensional, exactly
    as for the :py:class:`~pyoomph.meshes.gmsh.GmshTemplate`.

    TQMesh only knows straight boundary edges, so a curved boundary is handed to it as a polygonal chain, whose
    points :py:meth:`circle_arc` and :py:meth:`spline` distribute along the curve by the local element size. The
    exact curve is kept as a macro element, so that nodes created by spatial refinement end up on it rather than on
    the chords.

    As for any :py:class:`~pyoomph.meshes.mesh.MeshedMeshTemplate`, remeshing recreates the geometry by calling
    :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.define_geometry` again, where
    :py:meth:`~pyoomph.meshes.mesh.MeshedMeshTemplate.is_remeshing` and
    :py:meth:`~pyoomph.meshes.mesh.MeshedMeshTemplate.get_boundary_coordinates` are available.
    """

    def __init__(self):
        super().__init__()
        if not getattr(_pyoomph, "has_tqmesh", False):
            raise RuntimeError("This build of pyoomph was made without TQMesh (cmake option PYOOMPH_HAS_TQMESH=OFF), so TQMeshTemplate cannot be used. Use a GmshTemplate instead or rebuild pyoomph with TQMesh.")
        #: The default element size as a nondimensional length scale, used wherever no other size is prescribed
        self.default_resolution: float | None = None
        #: All element sizes, including :py:attr:`default_resolution`, are multiplied by this factor. Useful to
        #: refine or coarsen an entire mesh at once.
        self.mesh_size_factor: float = 1.0
        #: ``"tris"`` gives a purely triangular mesh, ``"quads"`` merges suitable triangle pairs into quadrilaterals
        #: and ``"only_quads"`` splits everything into quadrilaterals (which quadruples the element count, so the
        #: element sizes are doubled beforehand, as in the :py:class:`~pyoomph.meshes.gmsh.GmshTemplate`).
        self.mesh_mode: Literal["tris", "quads", "only_quads"] = "tris"
        #: Whether coordinates passed to :py:meth:`point` are divided by the spatial scale of the problem
        self.consider_spatial_scale: bool = True
        #: Number of smoothing iterations applied to each domain after the elements have been generated
        self.smoothing_iterations: int = 2
        #: Strategy used for the smoothing
        self.smoothing_kind: Literal["mixed", "laplace", "torsion"] = "mixed"
        #: Extent of TQMesh's internal quadtree, which must enclose the entire geometry. When ``None``, it is
        #: calculated from the geometry itself, which is usually what you want.
        self.quadtree_scale: float | None = None
        #: Factor by which the automatically determined :py:attr:`quadtree_scale` exceeds the geometry extent
        self.quadtree_scale_factor: float = 0.5 * (1.0 + 5.0 ** 0.5)
        #: If set, a callback ``f(x,y)`` returning the desired nondimensional element size at the nondimensional
        #: position (x,y), or ``None`` there to fall back to the default. Alternatively, override :py:meth:`mesh_size`.
        self.mesh_size_callback: Callable[[float, float], float | None] | None = None
        #: Whether to raise an error if TQMesh could not fill a domain entirely. Meshing can fail without saying so,
        #: hence this is checked by default.
        self.check_completeness: bool = True
        #: If set, each generated domain is additionally written to ``<name>.vtu`` in the output directory, which is
        #: helpful to see what TQMesh actually made of a geometry.
        self.write_vtu_output: bool = False
        self._reset_geometry()

    ###########################################################################
    # Geometry definition
    ###########################################################################

    def _reset_geometry(self):
        self._points: list[TQMeshPoint] = []
        self._pointhash: dict[tuple[float, float], TQMeshPoint] = {}
        self._curves: list[TQMeshCurve] = []
        self._named_curves: dict[str, list[TQMeshCurve]] = {}
        self._named_points: dict[str, TQMeshPoint] = {}
        self._surfaces: list[_TQMeshSurface] = []
        self._quad_layers: list[_TQMeshQuadLayer] = []
        self._boundary_colors: dict[str, int] = {}
        self._curved_lookup_cache: dict[str, _TQMeshCurvedLookup] | None = None

    def _reset(self):
        super()._reset()
        self._reset_geometry()

    def _nondim_coordinate(self, v: ExpressionOrNum, consider_spatial_scale: bool) -> float:
        if consider_spatial_scale:
            v = v / self.get_problem().get_scaling("spatial")
        if isinstance(v, Expression):
            return v.float_value()
        return float(v)

    def _nondim_resolution(self, size: ExpressionNumOrNone) -> float | None:
        # Same convention as GmshTemplate.point(): sizes are nondimensional, a negative one is taken relative to
        # default_resolution, and everything is scaled by mesh_size_factor.
        if size is None:
            size = self.default_resolution
        if size is None:
            return None
        if not isinstance(size, (int, float)):
            try:
                size = float(size)  # type:ignore
            except Exception:
                raise RuntimeError("A mesh resolution (i.e. a size argument) is expected to be nondimensional, i.e. a float, not " + str(size))
        size = float(size)
        if size < 0:
            if self.default_resolution is None:
                raise RuntimeError("A negative mesh resolution is given relative to self.default_resolution, but self.default_resolution is not set")
            size = -size * self.default_resolution
        size *= self.mesh_size_factor
        if self.mesh_mode == "only_quads":
            # each element will be split into four quads later on
            size *= 2
        return size

    def point(self, x: ExpressionOrNum, y: ExpressionOrNum = 0.0, size: ExpressionNumOrNone = None, *,
              size_range: ExpressionNumOrNone = None, name: str | None = None,
              consider_spatial_scale: bool | None = None) -> TQMeshPoint:
        """
        Add a point to the geometry. Coordinates are given in the spatial unit of the problem, i.e. in meter if the
        problem has ``set_scaling(spatial=...)`` set. Points with identical coordinates are only created once, so it
        is fine to describe adjacent curves by repeating the same coordinates.

        Args:
            x: The x-coordinate of the point.
            y: The y-coordinate of the point.
            size: Desired element size near this point, as a nondimensional length. Defaults to
                :py:attr:`default_resolution`; a negative value is taken relative to it.
            size_range: Distance over which the local ``size`` blends into the surrounding element size. Defaults to
                the length of the adjacent boundary edges.
            name: Name of the point, so that it can be found again with :py:meth:`get_point`.
            consider_spatial_scale: Whether to divide the coordinates by the spatial scale. Defaults to
                :py:attr:`consider_spatial_scale`.

        Returns:
            The created point, to be used for :py:meth:`line`, :py:meth:`circle_arc` and friends.
        """
        if consider_spatial_scale is None:
            consider_spatial_scale = self.consider_spatial_scale
        xf = self._nondim_coordinate(x, consider_spatial_scale)
        yf = self._nondim_coordinate(y, consider_spatial_scale)
        existing = self._pointhash.get((xf, yf))
        if existing is not None:
            # A repeated coordinate may still carry a size that the first occurrence did not
            if size is not None:
                existing.size = self._nondim_resolution(size)
            if size_range is not None:
                existing.size_range = self._nondim_resolution(size_range)
            if name is not None:
                self._named_points[name] = existing
            return existing
        pt = TQMeshPoint(xf, yf, self._nondim_resolution(size), self._nondim_resolution(size_range), name)
        self._points.append(pt)
        self._pointhash[(xf, yf)] = pt
        if name is not None:
            self._named_points[name] = pt
        return pt

    def get_point(self, name: str) -> TQMeshPoint:
        """Returns a point previously created with a ``name``."""
        if name not in self._named_points:
            raise RuntimeError("There is no point named '" + name + "' in this geometry")
        return self._named_points[name]

    def _as_point(self, p: TQMeshPoint | Sequence[ExpressionOrNum]) -> TQMeshPoint:
        if isinstance(p, TQMeshPoint):
            return p
        if isinstance(p, (list, tuple)):
            return self.point(*p)
        raise RuntimeError("Expected a point or a coordinate tuple, got " + str(p))

    def _add_curve(self, points: list[TQMeshPoint], name: str | None,
                   curved_entity: "_pyoomph.MeshTemplateCurvedEntityBase | None" = None) -> TQMeshCurve:
        curve = TQMeshCurve(points, name, curved_entity)
        self._curves.append(curve)
        if name is not None:
            self._named_curves.setdefault(name, []).append(curve)
        return curve

    def line(self, *args: TQMeshPoint | Sequence[ExpressionOrNum], name: str | None = None) -> TQMeshCurve:
        """
        Add a straight line - or, when more than two points are given, a polygonal chain - to the geometry. When a
        name is given, the corresponding part of the mesh boundary can be addressed by it, e.g. to set boundary
        conditions there.

        Args:
            *args: The points of the line, either as :py:class:`TQMeshPoint` or as coordinate tuples.
            name: Name of the boundary this line belongs to.

        Returns:
            The created curve, to be passed to :py:meth:`plane_surface`.
        """
        if len(args) < 2:
            raise RuntimeError("A line requires at least two points")
        return self._add_curve([self._as_point(a) for a in args], name)

    def create_lines(self, *args: TQMeshPoint | Sequence[ExpressionOrNum] | str) -> list[TQMeshCurve]:
        """
        Create several named lines at once, alternating points and boundary names, e.g.

        .. code-block:: python

            lines=self.create_lines((0,0),"bottom",(1,0),"right",(1,1),"top",(0,1),"left")
            self.plane_surface(*lines,name="box")

        If the arguments end with a name rather than a point, the loop is closed back to the first point.

        Args:
            *args: Points and names in the order p1, <name>, p2, <name>, p3, ...

        Returns:
            The created curves.
        """
        closed_loop = len(args) % 2 == 0
        res: list[TQMeshCurve] = []
        for i in range(len(args) // 2):
            pstart, name = args[2 * i], args[2 * i + 1]
            pend = args[0] if (closed_loop and 2 * i + 2 == len(args)) else args[2 * i + 2]
            if isinstance(pstart, str) or isinstance(pend, str) or not isinstance(name, str):
                raise RuntimeError("create_lines needs arguments like p1, <name>, p2, <name>, p3, ...")
            res.append(self.line(pstart, pend, name=name if name != "" else None))
        return res

    def circle_arc(self, start: TQMeshPoint | Sequence[ExpressionOrNum], end: TQMeshPoint | Sequence[ExpressionOrNum],
                   *, center: TQMeshPoint | Sequence[ExpressionOrNum] | None = None,
                   through_point: TQMeshPoint | Sequence[ExpressionOrNum] | None = None, name: str | None = None,
                   n_segments: int | None = None, size: ExpressionNumOrNone = None,
                   with_macro_element: bool = True) -> TQMeshCurve:
        """
        Add a circular arc from ``start`` to ``end``, either around a ``center`` or through a third point on the same
        circle. Since TQMesh has no curved boundaries, the arc is approximated by a chain of straight segments, whose
        points are placed along the arc according to the local element size (see :py:meth:`mesh_size`) unless an
        explicit ``n_segments`` asks for an equidistant subdivision. The exact circle is kept as a macro element, so
        that spatial refinement places new nodes on the circle rather than on the chords.

        Args:
            start: Start point of the arc.
            end: End point of the arc.
            center: Center of the circle.
            through_point: Alternatively, a third point on the circle.
            name: Name of the boundary this arc belongs to.
            n_segments: Number of equidistant segments to approximate the arc with. By default, the points are
                distributed by the local element size instead.
            size: Element size along the arc, overriding :py:attr:`default_resolution` and :py:meth:`mesh_size` here.
            with_macro_element: Whether to keep the circle as a macro element for spatial refinement.

        Returns:
            The created curve, to be passed to :py:meth:`plane_surface`.
        """
        p_start, p_end = self._as_point(start), self._as_point(end)
        if center is not None:
            if through_point is not None:
                raise RuntimeError("Pass either a center or a through_point to circle_arc, not both")
            p_center = self._as_point(center)
            cx, cy = p_center.x, p_center.y
        elif through_point is not None:
            p_through = self._as_point(through_point)
            cx, cy = self._circle_center(p_start, p_end, p_through)
        else:
            raise RuntimeError("circle_arc requires either a center or a through_point")

        r_start = math.hypot(p_start.x - cx, p_start.y - cy)
        r_end = math.hypot(p_end.x - cx, p_end.y - cy)
        if abs(r_start - r_end) > 1e-8 * max(r_start, r_end, 1e-30):
            raise RuntimeError("The start and end point of a circle_arc do not have the same distance to the center ({} vs {})".format(r_start, r_end))
        radius = 0.5 * (r_start + r_end)

        a_start = math.atan2(p_start.y - cy, p_start.x - cx)
        a_end = math.atan2(p_end.y - cy, p_end.x - cx)
        # Always take the shorter way around, as gmsh's circle arcs do
        delta = a_end - a_start
        while delta <= -math.pi:
            delta += 2 * math.pi
        while delta > math.pi:
            delta -= 2 * math.pi

        def position(t: float) -> tuple[float, float]:
            angle = a_start + delta * t
            return (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

        if n_segments is not None:
            if n_segments < 1:
                raise RuntimeError("A circle_arc requires at least one segment")
            interior = [position(i / n_segments) for i in range(1, n_segments)]
        else:
            interior = self._distribute_along_curve(position, size, "circle_arc")

        points = [p_start] + [self.point(x, y, size=size, consider_spatial_scale=False) for x, y in interior] + [p_end]
        entity = None
        if with_macro_element:
            # The parameters are the center and the two end points, despite what the argument names of
            # CurvedEntityCircleArc suggest - it charts the circle by the polar angle around the center.
            entity = _pyoomph.CurvedEntityCircleArc([cx, cy, 0.0], [p_start.x, p_start.y, 0.0], [p_end.x, p_end.y, 0.0])
        return self._add_curve(points, name, entity)

    @staticmethod
    def _circle_center(p1: TQMeshPoint, p2: TQMeshPoint, p3: TQMeshPoint) -> tuple[float, float]:
        ax, ay, bx, by, cx, cy = p1.x, p1.y, p2.x, p2.y, p3.x, p3.y
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-30:
            raise RuntimeError("The three points of a circle_arc are collinear, so they do not define a circle")
        a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
        return ((a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d,
                (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d)

    def spline(self, points: Sequence[TQMeshPoint | Sequence[ExpressionOrNum]], *, name: str | None = None,
               resample: bool = False, size: ExpressionNumOrNone = None,
               with_macro_element: bool = True) -> TQMeshCurve:
        """
        Add a boundary following the given sequence of points, interpolated by a Catmull-Rom spline. It is the
        counterpart of the gmsh method of the same name and is typically used to rebuild a deformed interface upon
        remeshing, where the points come from
        :py:meth:`~pyoomph.meshes.mesh.MeshedMeshTemplate.get_boundary_coordinates` and are already spaced by the
        element size.

        By default, the given points are used as the polygonal chain handed to TQMesh, i.e. they all end up in the
        mesh. With ``resample=True``, they are treated as control points of the spline instead and the chain is
        distributed along its arclength according to the local element size (see :py:meth:`mesh_size`) - the boundary
        still follows the given points, but only the first and the last of them are guaranteed to be mesh vertices.
        Either way, the spline itself is kept as a macro element, so that spatial refinement places new nodes on it
        rather than on the chords of the chain.

        Args:
            points: The points defining the curve.
            name: Name of the boundary this curve belongs to.
            resample: Whether to distribute the chain points by element size instead of using the given points.
            size: Element size along the curve, overriding :py:attr:`default_resolution` and :py:meth:`mesh_size`
                here. Only used with ``resample=True``.
            with_macro_element: Whether to keep the spline as a macro element for spatial refinement.

        Returns:
            The created curve, to be passed to :py:meth:`plane_surface`.
        """
        if len(points) < 2:
            raise RuntimeError("A spline requires at least two points")
        control = [self._as_point(p) for p in points]
        entity = None
        if with_macro_element:
            if len(control) < 3:
                # A Catmull-Rom spline through two points is the straight line between them anyway
                with_macro_element = False
            else:
                entity = _pyoomph.CurvedEntityCatmullRomSpline([[p.x, p.y, 0.0] for p in control])

        if not resample:
            return self._add_curve(control, name, entity)

        if len(control) < 3:
            chain = control
        else:
            evaluate = self._catmull_rom_evaluator(control)
            interior = self._distribute_along_curve(evaluate, size, "spline")
            chain = [control[0]] + [self.point(x, y, size=size, consider_spatial_scale=False) for x, y in interior] + [control[-1]]
        return self._add_curve(chain, name, entity)

    @staticmethod
    def _catmull_rom_evaluator(control: list[TQMeshPoint]) -> Callable[[float], tuple[float, float]]:
        # The very same spline that CurvedEntityCatmullRomSpline (src/meshtemplate.cpp) evaluates, so that the chain
        # points distributed along it lie exactly on the macro element: the control point list is extended by a
        # mirrored phantom point at each end (which makes the spline interpolate the first and the last point), and
        # the parameter runs from 0 to len(control)-1, one unit per interval.
        pts = [(2 * control[0].x - control[1].x, 2 * control[0].y - control[1].y)]
        pts += [(p.x, p.y) for p in control]
        pts += [(2 * control[-1].x - control[-2].x, 2 * control[-1].y - control[-2].y)]
        n_intervals = len(control) - 1

        def evaluate(t: float) -> tuple[float, float]:
            u = min(max(t, 0.0), 1.0) * n_intervals
            offset = min(int(u), n_intervals - 1)
            u -= offset
            u2, u3 = u * u, u * u * u
            s0, s1 = -0.5 * u3 + u2 - 0.5 * u, 1.5 * u3 - 2.5 * u2 + 1.0
            s2, s3 = -1.5 * u3 + 2.0 * u2 + 0.5 * u, 0.5 * u3 - 0.5 * u2
            return (s0 * pts[offset][0] + s1 * pts[offset + 1][0] + s2 * pts[offset + 2][0] + s3 * pts[offset + 3][0],
                    s0 * pts[offset][1] + s1 * pts[offset + 1][1] + s2 * pts[offset + 2][1] + s3 * pts[offset + 3][1])

        return evaluate

    def _distribute_along_curve(self, position: Callable[[float], tuple[float, float]], size: ExpressionNumOrNone,
                                what: str) -> list[tuple[float, float]]:
        """
        Places points along the curve ``position(t)``, t running from 0 to 1, such that the distance between
        neighbours matches the local element size there. Returns the interior points only, i.e. neither of the two
        end points, which the caller keeps exactly.
        """
        # Sample the curve densely enough that the arclength and the size are resolved, then integrate ds/h along
        # it: the result is the number of elements the curve deserves, and equal increments of that integral are
        # exactly the point positions we are looking for.
        n_samples = 500
        ts = [i / (n_samples - 1.0) for i in range(n_samples)]
        pos = [position(t) for t in ts]
        fixed = self._nondim_resolution(size) if size is not None else None
        integral = [0.0]
        for i in range(1, n_samples):
            ds = math.hypot(pos[i][0] - pos[i - 1][0], pos[i][1] - pos[i - 1][1])
            h1 = fixed if fixed is not None else self._element_size_at(*pos[i - 1], what=what)
            h2 = fixed if fixed is not None else self._element_size_at(*pos[i], what=what)
            integral.append(integral[-1] + 2.0 * ds / (h1 + h2))
        total = integral[-1]
        n_elements = max(1, int(round(total)))
        if n_elements == 1:
            return []

        res: list[tuple[float, float]] = []
        j = 1
        for k in range(1, n_elements):
            target = total * k / n_elements
            while j < n_samples - 1 and integral[j] < target:
                j += 1
            # linear interpolation of the parameter between the two samples bracketing the target
            span = integral[j] - integral[j - 1]
            frac = (target - integral[j - 1]) / span if span > 0 else 0.0
            res.append(position(ts[j - 1] + frac * (ts[j] - ts[j - 1])))
        return res

    def _element_size_at(self, x: float, y: float, what: str) -> float:
        # The domain a curve belongs to is only known once plane_surface() has been called, and an interface curve
        # belongs to two of them, so the size hook is asked without a domain name while distributing boundary points.
        size = self.mesh_size(x, y, None)
        if size is None:
            size = self._nondim_resolution(None)
        if size is None or size <= 0:
            raise RuntimeError("To distribute the points of a " + what + " by element size, either set self.default_resolution, pass a size, return one from mesh_size() or prescribe the number of segments explicitly")
        return size

    def plane_surface(self, *args: TQMeshCurve | str, name: str, resolution: ExpressionNumOrNone = None) -> None:
        """
        Create a domain from the given curves, which must enclose it completely. Pass the curves either directly or
        by their boundary names. The curves are sorted into closed loops automatically: the outer loop becomes the
        domain, any loop inside it a hole, and a loop inside a hole another (disjoint) part of the same domain.

        Call this once per domain. Curves shared by two domains, i.e. those passed to two calls, become the interface
        between them; TQMesh discretizes such an interface when the first of the two domains is meshed and lets the
        second one conform to it.

        Args:
            *args: The curves enclosing the domain, in any order, or the names of their boundaries.
            name: Name of the domain, used to add equations to it later on.
            resolution: Element size within this domain, as a nondimensional length. Defaults to
                :py:attr:`default_resolution`.
        """
        curves: list[TQMeshCurve] = []
        for a in args:
            if isinstance(a, str):
                if a not in self._named_curves:
                    raise RuntimeError("There is no curve named '" + a + "' in this geometry")
                curves.extend(self._named_curves[a])
            elif isinstance(a, TQMeshCurve):
                curves.append(a)
            else:
                raise RuntimeError("plane_surface expects curves or boundary names, got " + str(a))
        if self.has_domain(name):
            raise RuntimeError("A domain named '" + name + "' was already created. Every plane_surface needs its own name.")

        loops = self._assemble_loops(curves, name)
        for exterior, holes in self._classify_loops(loops, name):
            self._surfaces.append(_TQMeshSurface(name, exterior.oriented(True), [h.oriented(False) for h in holes],
                                                 self._nondim_resolution(resolution)))

    def quad_layer(self, boundary: str, n_layers: int = 3, first_height: ExpressionNumOrNone = None,
                   growth_rate: float = 1.5, *, domain: str | None = None,
                   start: TQMeshPoint | Sequence[ExpressionOrNum] | None = None,
                   end: TQMeshPoint | Sequence[ExpressionOrNum] | None = None) -> None:
        """
        Cover the given boundary with structured layers of quadrilateral elements before the rest of the domain is
        filled - typically used to resolve a boundary layer along a wall or a free surface.

        Args:
            boundary: Name of the boundary to attach the layers to.
            n_layers: Number of layers.
            first_height: Height of the first layer as a nondimensional length. Defaults to a tenth of the element
                size at the boundary.
            growth_rate: Factor by which each layer is thicker than the previous one.
            domain: Name of the domain to generate the layers in. Only required if the boundary is an interface, i.e.
                belongs to more than one domain.
            start: Where along the boundary the layers begin. Defaults to the start of the boundary.
            end: Where they end. Defaults to the end of the boundary. Passing the same position as ``start`` covers
                the entire closed boundary it lies on.
        """
        height = self._nondim_resolution(first_height)
        if height is None:
            raise RuntimeError("The first_height of a quad layer must be given, or self.default_resolution must be set")
        if first_height is None:
            height *= 0.1
        self._quad_layers.append(_TQMeshQuadLayer(
            boundary, domain, n_layers, height, growth_rate,
            self._as_point(start).coordinate if start is not None else None,
            self._as_point(end).coordinate if end is not None else None))

    def mesh_size(self, x: float, y: float, domain: str | None) -> float | None:
        """
        The desired element size at the nondimensional position (x,y) of the given domain, or ``None`` to use the
        domain's own resolution. Override this (or set :py:attr:`mesh_size_callback`) to grade the mesh; note that
        this is called very often during meshing, so it should be cheap.

        Args:
            x: Nondimensional x-coordinate.
            y: Nondimensional y-coordinate.
            domain: Name of the domain currently being meshed. ``None`` while the points of a resampled
                :py:meth:`circle_arc` or :py:meth:`spline` are distributed, which happens before the curve is
                assigned to a domain - and an interface curve belongs to two of them anyway.

        Returns:
            The nondimensional element size, or ``None``.
        """
        if self.mesh_size_callback is not None:
            return self.mesh_size_callback(x, y)
        return None

    ###########################################################################
    # Loop assembly
    ###########################################################################

    def _assemble_loops(self, curves: list[TQMeshCurve], domain_name: str) -> list[_TQMeshLoop]:
        # Every segment of the given curves must end up in exactly one closed loop, i.e. each point must be shared by
        # exactly two segments. Anything else is a geometry that does not enclose a domain, which TQMesh would only
        # notice by failing to fill it.
        segments: list[tuple[TQMeshPoint, TQMeshPoint, str]] = []
        for curve in curves:
            if curve.name is None:
                raise RuntimeError("The curves of the domain '" + domain_name + "' must all be named, since their name identifies the corresponding mesh boundary")
            for p1, p2 in curve.segments():
                if p1 is p2:
                    raise RuntimeError("The boundary '" + curve.name + "' of the domain '" + domain_name + "' has a segment of zero length at " + str(p1))
                segments.append((p1, p2, curve.name))

        adjacency: dict[int, list[int]] = {}
        for i, (p1, p2, _) in enumerate(segments):
            adjacency.setdefault(id(p1), []).append(i)
            adjacency.setdefault(id(p2), []).append(i)
        for pt in self._points:
            attached = adjacency.get(id(pt), [])
            if len(attached) not in (0, 2):
                raise RuntimeError("The point " + str(pt) + " is used by " + str(len(attached)) + " boundary segments of the domain '" + domain_name + "', but a closed boundary requires exactly two. Check that the curves passed to plane_surface enclose the domain and do not overlap.")

        loops: list[_TQMeshLoop] = []
        used: set[int] = set()
        for start_index in range(len(segments)):
            if start_index in used:
                continue
            used.add(start_index)
            p_first, p_current, name = segments[start_index]
            points, names = [p_first], [name]
            while p_current is not p_first:
                for i in adjacency[id(p_current)]:
                    if i in used:
                        continue
                    p1, p2, name = segments[i]
                    used.add(i)
                    points.append(p_current)
                    names.append(name)
                    p_current = p2 if p1 is p_current else p1
                    break
                else:
                    raise RuntimeError("The boundary of the domain '" + domain_name + "' is not closed: it ends at " + str(p_current))
            loops.append(_TQMeshLoop(points, names))
        if not loops:
            raise RuntimeError("The domain '" + domain_name + "' has no boundary at all")
        return loops

    def _classify_loops(self, loops: list[_TQMeshLoop], domain_name: str) -> list[tuple[_TQMeshLoop, list[_TQMeshLoop]]]:
        # Which loop is the domain and which one is a hole follows from the nesting: a loop inside an odd number of
        # others is a hole of the innermost of them, everything else bounds (a part of) the domain.
        parents: list[int | None] = []
        for i, loop in enumerate(loops):
            best: int | None = None
            for j, other in enumerate(loops):
                if i == j or not other.contains(loop.points[0].x, loop.points[0].y):
                    continue
                if best is None or abs(other.signed_area()) < abs(loops[best].signed_area()):
                    best = j
            parents.append(best)

        def depth(i: int) -> int:
            d, p = 0, parents[i]
            while p is not None:
                d, p = d + 1, parents[p]
            return d

        result: list[tuple[_TQMeshLoop, list[_TQMeshLoop]]] = []
        for i, loop in enumerate(loops):
            if depth(i) % 2 == 0:
                holes = [loops[j] for j in range(len(loops)) if parents[j] == i and depth(j) % 2 == 1]
                result.append((loop, holes))
        if not result:
            raise RuntimeError("The domain '" + domain_name + "' consists of holes only")
        return result

    ###########################################################################
    # Mesh generation
    ###########################################################################

    def _do_define_geometry(self, problem: "Problem", filename_trunk: str | None = None):
        self._set_problem(problem)
        if self._geometry_defined:
            return
        # Runs define_geometry() (with is_remeshing() and get_boundary_coordinates() available) and marks the
        # geometry as defined; the actual meshing happens afterwards, since it requires the finished geometry.
        super()._do_define_geometry(problem, filename_trunk)
        self._generate()
        if self.auto_find_opposite_interface_connections:
            # Repeated here (the base class already did it before there were any elements) exactly as the
            # GmshTemplate does it after loading its mesh file: only now are there domains to connect.
            self._find_opposite_interface_connections()

    def _color_of(self, boundary_name: str) -> int:
        # TQMesh identifies boundaries by integer colors, pyoomph by names
        if boundary_name not in self._boundary_colors:
            self._boundary_colors[boundary_name] = len(self._boundary_colors) + 1
        return self._boundary_colors[boundary_name]

    def _determine_quadtree_scale(self) -> float:
        if self.quadtree_scale is not None:
            return self.quadtree_scale
        # TQMesh's quadtree spans [-scale/2, scale/2] around the origin, so the scale must cover twice the largest
        # coordinate - and a bit more, since the mesh vertices may end up slightly outside the boundary polygon.
        # The default factor is irrational on purpose (see _generate): the quadtree cell boundaries lie at
        # scale/2**k, which a coordinate given as a simple fraction of the geometry extent can then never hit.
        extent = 0.0
        for p in self._points:
            extent = max(extent, abs(p.x), abs(p.y))
        if extent <= 0.0:
            raise RuntimeError("The geometry has no extent at all - did define_geometry create any points?")
        return 2 * extent * self.quadtree_scale_factor

    def _size_function(self, surface: _TQMeshSurface) -> Callable[[float, float], float] | float:
        base = surface.resolution if surface.resolution is not None else self._nondim_resolution(None)
        name = surface.name

        def evaluate(x: float, y: float) -> float:
            size = self.mesh_size(x, y, name)
            if size is None:
                size = base
            if size is None or size <= 0:
                raise RuntimeError("No element size is defined for the domain '" + name + "' at (" + str(x) + ", " + str(y) + "). Set self.default_resolution, pass a resolution to plane_surface or return a size from mesh_size().")
            return size

        # A constant size lets TQMesh evaluate it without calling back into python for every single query
        if base is not None and self.mesh_size_callback is None and type(self).mesh_size is TQMeshTemplate.mesh_size:
            return base
        return evaluate

    def _generate(self):
        if not self._surfaces:
            raise RuntimeError("No domain was created - call plane_surface() in define_geometry()")
        scale = self._determine_quadtree_scale()
        # TQMesh sorts its entities into a quadtree whose cells are tested inclusively on both sides, so an entity
        # landing exactly on a cell boundary can be inserted into one cell and looked up in another - after which
        # meshing fails with "Failed to remove item from QuadTree". Whether that happens depends only on the scale,
        # since the cell boundaries sit at scale/2**k: retrying with a slightly detuned scale gets around it, and
        # the scale chosen by _determine_quadtree_scale() avoids it in the first place.
        for attempt, detune in enumerate([1.0, 1.037, 1.0813]):
            try:
                return self._generate_with_quadtree_scale(scale * detune)
            except RuntimeError as error:
                if "QuadTree" not in str(error) or attempt == 2:
                    raise
                if not self.get_problem().is_quiet():
                    print("Retrying the mesh generation with a slightly different TQMesh quadtree scale, since an entity ended up exactly on a quadtree cell boundary")

    def _generate_with_quadtree_scale(self, quadtree_scale: float):
        generator = _pyoomph.TQMeshGenerator()

        color_names: dict[int, str] = {}
        meshes: list[tuple[_TQMeshSurface, Any]] = []
        for index, surface in enumerate(self._surfaces):
            domain = _pyoomph.TQMeshDomain(self._size_function(surface), quadtree_scale=quadtree_scale)
            self._add_loop(domain, surface.exterior, color_names, exterior=True)
            for hole in surface.holes:
                self._add_loop(domain, hole, color_names, exterior=False)
            mesh = generator.new_mesh(domain, mesh_id=index, element_color=index)
            meshes.append((surface, mesh))

            layers = self._quad_layers_of(surface)
            for layer in layers:
                start, end = self._quad_layer_positions(layer, surface)
                if not generator.quad_layer(mesh, n_layers=layer.n_layers, first_height=layer.first_height,
                                            growth_rate=layer.growth_rate, start=start, end=end):
                    raise RuntimeError("TQMesh could not generate the quad layer at the boundary '" + layer.boundary + "' of the domain '" + surface.name + "'")

            if not generator.triangulate(mesh):
                raise RuntimeError("TQMesh could not fill the domain '" + surface.name + "' with elements. Its boundary may be self-intersecting, or the element size may be too coarse to resolve it.")

            def smooth(single_shape: bool):
                if self.smoothing_iterations <= 0:
                    return
                kind = self.smoothing_kind
                if not single_shape and kind != "laplace":
                    # A mesh holding triangles and quadrilaterals at once cannot be smoothed by TQMesh's
                    # torsion-based strategies (which the default "mixed" one uses as well): they update elements
                    # that were discarded while the quads were formed and trip over their stale quadtree entries.
                    # Upstream only ever smooths meshes of a single element shape, so it never runs into this. The
                    # laplacian strategy is unaffected and is used instead.
                    kind = "laplace"
                generator.smooth(mesh, iterations=self.smoothing_iterations, kind=kind,
                                 quad_layer_smoothing=bool(layers))

            if self.mesh_mode == "only_quads":
                generator.tri2quad(mesh)
                generator.quad_refine(mesh)
                smooth(True)
            elif self.mesh_mode == "quads":
                # Smoothed before the merge, where the mesh still consists of triangles alone (bar the quad layers).
                # Merging triangles into quads moves no vertex, so this gives the same mesh as smoothing afterwards.
                smooth(not layers)
                generator.tri2quad(mesh)
            else:
                smooth(not layers)
            if self.check_completeness and not mesh.check_completeness():
                raise RuntimeError("TQMesh did not fill the domain '" + surface.name + "' completely. Set self.check_completeness=False to use the mesh anyway.")
            if self.write_vtu_output:
                import os
                # _fntrunk is set anew for every remeshing round, so the rounds do not overwrite each other
                trunk = (self._fntrunk + "_" if self._fntrunk else "") + surface.name
                generator.write(mesh, os.path.join(self.get_problem().get_output_directory(), trunk), "vtu")

        for surface, mesh in meshes:
            self._transfer_mesh(surface, mesh, color_names)

    def _add_loop(self, domain: Any, loop: _TQMeshLoop, color_names: dict[int, str], exterior: bool):
        coords = [(p.x, p.y) for p in loop.points]
        colors = []
        for name in loop.names:
            color = self._color_of(name)
            color_names[color] = name
            colors.append(color)
        # (size, range) per vertex, where zeros mean "no local refinement here"
        properties = [(p.size if p.size is not None else 0.0, p.size_range if p.size_range is not None else 0.0)
                      for p in loop.points]
        if not any(s > 0 for s, _ in properties):
            properties = None
        if exterior:
            domain.add_exterior_boundary(coords, colors, properties)
        else:
            domain.add_interior_boundary(coords, colors, properties)

    @staticmethod
    def _has_boundary(surface: _TQMeshSurface, name: str) -> bool:
        return any(name in loop.names for loop in [surface.exterior] + surface.holes)

    def _quad_layers_of(self, surface: _TQMeshSurface) -> list[_TQMeshQuadLayer]:
        res: list[_TQMeshQuadLayer] = []
        for layer in self._quad_layers:
            owners = sorted(set(s.name for s in self._surfaces if self._has_boundary(s, layer.boundary)))
            if not owners:
                raise RuntimeError("No domain has a boundary named '" + layer.boundary + "' to attach a quad layer to")
            if layer.domain is None and len(owners) > 1:
                raise RuntimeError("The boundary '" + layer.boundary + "' is shared by the domains " + ", ".join(owners) + ", so the domain to generate the quad layer in must be given explicitly")
            if (layer.domain if layer.domain is not None else owners[0]) != surface.name:
                continue
            if not self._has_boundary(surface, layer.boundary):
                raise RuntimeError("The domain '" + surface.name + "' has no boundary named '" + layer.boundary + "' to attach a quad layer to")
            res.append(layer)
        return res

    def _quad_layer_positions(self, layer: _TQMeshQuadLayer, surface: _TQMeshSurface) -> tuple[tuple[float, float], tuple[float, float]]:
        if layer.start is not None and layer.end is not None:
            return layer.start, layer.end
        # By default, the layer covers the whole named boundary, i.e. it runs from the first corner of its first
        # segment to the last corner of its last one - which coincide if the boundary is closed on its own.
        for loop in [surface.exterior] + surface.holes:
            indices = [i for i, name in enumerate(loop.names) if name == layer.boundary]
            if not indices:
                continue
            first, last = indices[0], indices[-1]
            # A boundary wrapping around the start of the loop would be split into two runs here
            if len(indices) > 1 and indices[-1] - indices[0] + 1 != len(indices):
                runs = [i for i in indices if (i - 1) % len(loop.names) not in indices]
                if len(runs) == 1:
                    first = runs[0]
                    last = (first - 1) % len(loop.names)
            start = layer.start if layer.start is not None else loop.points[first].coordinate
            end = layer.end if layer.end is not None else loop.points[(last + 1) % len(loop.points)].coordinate
            return start, end
        raise RuntimeError("The domain '" + surface.name + "' has no boundary named '" + layer.boundary + "'")

    def _curved_lookups(self) -> dict[str, _TQMeshCurvedLookup]:
        # Built once per generation and only for those boundaries that have a curved part at all
        if self._curved_lookup_cache is None:
            names = set(c.name for c in self._curves if c.curved_entity is not None and c.name is not None)
            self._curved_lookup_cache = {name: _TQMeshCurvedLookup([c for c in self._curves if c.name == name])
                                         for name in names}
        return self._curved_lookup_cache

    def _transfer_mesh(self, surface: _TQMeshSurface, mesh: Any, color_names: dict[int, str]):
        if self.has_domain(surface.name):
            collection = self.get_domain(surface.name)
        else:
            collection = self.new_domain(surface.name)
        # add_node_unique() identifies nodes by their position, which is what stitches the domains together: TQMesh
        # gives conforming interfaces, i.e. the vertices of a shared boundary have bit-identical coordinates in both
        # meshes, so the second domain reuses the nodes the first one created.
        vertices = mesh.vertices()
        nodes = [self.add_node_unique(float(x), float(y)) for x, y in vertices]
        for tri in mesh.triangles():
            collection.add_tri_2d_C1(nodes[tri[0]], nodes[tri[1]], nodes[tri[2]])
        for quad in mesh.quads():
            # TQMesh numbers the corners cyclically, oomph-lib expects them in tensor product order
            collection.add_quad_2d_C1(nodes[quad[0]], nodes[quad[1]], nodes[quad[3]], nodes[quad[2]])
        collection.set_nodal_dimension(2)
        collection.set_lagrangian_dimension(2)
        curved = self._curved_lookups()
        for edge in mesh.boundary_edges():
            name = color_names.get(int(edge[2]))
            if name is None:
                continue
            n1, n2 = nodes[edge[0]], nodes[edge[1]]
            entity = None
            if name in curved:
                mid = 0.5 * (vertices[edge[0]] + vertices[edge[1]])
                entity = curved[name].entity_at(float(mid[0]), float(mid[1]))
            self.add_facet_to_boundary(name, [n1, n2], [n1, n2], entity)
