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

"""Shared ``sort_along_axis``/``start_near_point`` treatment.

Points and line segments extracted from a mesh come in whatever order the mesh happens to store
them, which is fine for a plot of markers but not for anything that reads the result as a curve:
writing a text file, parameterising an interface by arclength, or handing boundary coordinates back
to a mesh generator. All of those want the same two knobs - order along a Cartesian direction, or
order by distance from a given point - so they are implemented once here.

Only the segment *end points* decide orientation and order; the nodes inside a segment keep the
order the mesh connectivity gives them. That is what makes the sorting survive an overhanging or
otherwise arbitrarily curved interface, where no Cartesian direction is monotonic along the curve.
"""

import numpy

from ..expressions.generic import ExpressionOrNum
from ..typings import *

#: Cartesian direction to order along, e.g. ``"x+"`` for increasing x, ``"y-"`` for decreasing y.
SortAlongAxis: TypeAlias = Literal["x+", "x-", "y+", "y-", "z+", "z-"]

_AXIS_INDEX_AND_SIGN: dict[str, tuple[int, int]] = {"x+": (0, 1), "x-": (0, -1), "y+": (1, 1), "y-": (1, -1), "z+": (2, 1), "z-": (2, -1)}


def check_sorting_arguments(sort_along_axis: SortAlongAxis | None, start_near_point: Sequence[ExpressionOrNum] | ExpressionOrNum | None, require_one: bool = False, whom: str = "") -> None:
    """Reject a combination of ``sort_along_axis`` and ``start_near_point`` that cannot be honoured.

    Args:
        sort_along_axis: The axis argument to check, may be None.
        start_near_point: The point argument to check, may be None.
        require_one: If set, exactly one of the two must be given, instead of at most one.
        whom: Name of the caller, prepended to the error message.
    """
    prefix = whom + ": " if whom else ""
    if sort_along_axis is not None and start_near_point is not None:
        raise RuntimeError(prefix + "Please set either sort_along_axis or start_near_point, not both - they are two ways of saying the same thing and would contradict each other")
    if require_one and sort_along_axis is None and start_near_point is None:
        raise RuntimeError(prefix + "Please set either sort_along_axis or start_near_point to identify the direction of the parameterization")
    if sort_along_axis is not None and sort_along_axis not in _AXIS_INDEX_AND_SIGN:
        raise RuntimeError(prefix + "Unknown sort_along_axis '" + str(sort_along_axis) + "', must be one of " + ", ".join(sorted(_AXIS_INDEX_AND_SIGN.keys())))


def nondimensionalise_point(start_near_point: Sequence[ExpressionOrNum] | ExpressionOrNum, numcoords: int, spatial_unit: ExpressionOrNum = 1) -> NPFloatArray:
    """Bring a user-given ``start_near_point`` into the nondimensional coordinates the mesh data uses.

    Args:
        start_near_point: The point, either a scalar (for a 1d coordinate space) or a sequence. Entries may carry units.
        numcoords: Number of Cartesian coordinate directions the mesh data has.
        spatial_unit: Spatial scale the mesh data is nondimensionalised with, or 1 if it is dimensional.
    """
    if not isinstance(start_near_point, (list, tuple, numpy.ndarray)):
        start_near_point = [cast(ExpressionOrNum, start_near_point)]
    if len(start_near_point) != numcoords:
        raise RuntimeError("start_near_point has " + str(len(start_near_point)) + " entries, but the mesh has " + str(numcoords) + " coordinate direction(s)")
    try:
        return numpy.array([float(c / spatial_unit) for c in start_near_point], dtype=numpy.float64)  # type:ignore
    except Exception as e:
        raise RuntimeError("Cannot express start_near_point=" + str(list(start_near_point)) + " in nondimensional coordinates by dividing by the spatial scale " + str(spatial_unit) + ". Entries with units must match that scale, entries without units are taken as already nondimensional") from e


def sort_line_segments(pts: NPFloatArray, segs: Sequence[Sequence[int]], sort_along_axis: SortAlongAxis | None = None, start_near_point: Sequence[ExpressionOrNum] | ExpressionOrNum | None = None, spatial_unit: ExpressionOrNum = 1, whom: str = "") -> list[list[int]]:
    """Orient and order line segments, e.g. the ones from ``MeshDataCacheEntry.get_interface_line_segments``.

    Each segment is reversed if it runs against the requested direction, and the segments are then
    sorted among each other. Both decisions are made on the segment end points only, so a segment
    that curves back on itself is not torn apart. If neither argument is given, the input order is
    kept.

    Args:
        pts: Nondimensional coordinates, shape ``(numcoords, numpoints)``.
        segs: Segments as lists of indices into the point columns.
        sort_along_axis: Order along a Cartesian direction, e.g. ``"x+"``.
        start_near_point: Order by distance from this point, closest first.
        spatial_unit: Spatial scale ``pts`` is nondimensionalised with, applied to ``start_near_point``.
        whom: Name of the caller, prepended to error messages.

    Returns:
        The reoriented segments in their new order. The input is not modified.
    """
    check_sorting_arguments(sort_along_axis, start_near_point, whom=whom)
    res = [list(s) for s in segs]
    if sort_along_axis is not None:
        index, sign = _AXIS_INDEX_AND_SIGN[sort_along_axis]
        if index >= pts.shape[0]:
            raise RuntimeError((whom + ": " if whom else "") + "Cannot sort along '" + sort_along_axis + "': the mesh only has " + str(pts.shape[0]) + " coordinate direction(s)")
        for i, seg in enumerate(res):
            if sign * (pts[index, seg[-1]] - pts[index, seg[0]]) < 0:
                res[i] = list(reversed(seg))
        res = sorted(res, key=lambda s: sign * pts[index, s[0]])
    elif start_near_point is not None:
        stp = nondimensionalise_point(start_near_point, pts.shape[0], spatial_unit)

        def dist2(ptind: int) -> float:
            return float(numpy.sum((pts[:, ptind] - stp) ** 2))

        for i, seg in enumerate(res):
            # Closest end first - the old duplicated versions of this had the comparison the wrong
            # way round and started each segment at the end farthest from the point, while still
            # sorting the segments themselves nearest-first.
            if dist2(seg[-1]) < dist2(seg[0]):
                res[i] = list(reversed(seg))
        res = sorted(res, key=lambda s: dist2(s[0]))
    return res


def sort_point_indices(pts: NPFloatArray, indices: Sequence[int] | None = None, sort_along_axis: SortAlongAxis | None = None, start_near_point: Sequence[ExpressionOrNum] | ExpressionOrNum | None = None, spatial_unit: ExpressionOrNum = 1, whom: str = "") -> list[int]:
    """Order isolated points, i.e. the 0d case of :py:func:`sort_line_segments`.

    Args:
        pts: Nondimensional coordinates, shape ``(numcoords, numpoints)``.
        indices: Point indices to order. Defaults to all columns of ``pts``.
        sort_along_axis: Order along a Cartesian direction, e.g. ``"x+"``.
        start_near_point: Order by distance from this point, closest first.
        spatial_unit: Spatial scale ``pts`` is nondimensionalised with, applied to ``start_near_point``.
        whom: Name of the caller, prepended to error messages.

    Returns:
        The point indices in their new order.
    """
    if indices is None:
        indices = range(pts.shape[1])
    # A single point is a degenerate segment: it cannot be reversed, so only the sorting acts.
    return [s[0] for s in sort_line_segments(pts, [[i] for i in indices], sort_along_axis=sort_along_axis, start_near_point=start_near_point, spatial_unit=spatial_unit, whom=whom)]
