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

"""Pure geometry for axisymmetric topology changes (pinch-off and coalescence).

This module is deliberately free of any pyoomph import: it only needs numpy, scipy's
``brentq`` and the optional dependency shapely.  That keeps it unit-testable without the
FEM stack and makes every dataclass here trivially picklable (numpy arrays and builtins
only), which is what an MPI broadcast of a computed :class:`SurgeryPlan` will need.

Coordinate convention (nondimensional throughout): ``x`` is the radial coordinate ``r``,
with the axis of symmetry at ``x = 0`` and all input points at ``x >= 0``; ``y`` is the
axial coordinate ``z``.

The detection is morphological.  The interface polylines are turned into a closed
half-section, mirrored about the axis into a full 2D cross section ``P``.  A neck of
minimal radius ``r_w`` then appears in ``P`` as a strip of *full* width ``2 r_w`` and an
axial gap of length ``d`` between two fragments appears as a gap of length ``d``.  See
:func:`_open` / :func:`_close` for the derivation of the structuring-element radii that
make the detection thresholds come out at exactly ``rmin_nd`` and ``distmin_nd``.
"""

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
from scipy.optimize import brentq

__all__ = [
    "InterfaceChain",
    "ReconnectionEvent",
    "NewChain",
    "SurgeryPlan",
    "detect_and_plan",
    "revolved_volume",
]


# --------------------------------------------------------------------------------------
# Optional dependency
# --------------------------------------------------------------------------------------

_SHAPELY: Optional[SimpleNamespace] = None


def _require_shapely() -> SimpleNamespace:
    """Import shapely lazily and return the handful of names this module uses."""
    global _SHAPELY
    if _SHAPELY is None:
        try:
            from shapely.geometry import LineString, Point, Polygon, box
            from shapely.geometry.polygon import orient
            from shapely.ops import split, substring, unary_union
        except ImportError as e:
            raise RuntimeError(
                "pyoomph.meshes.axisymm_topology requires the optional dependency "
                "'shapely' (>=2.0). Install via pip install shapely"
            ) from e
        _SHAPELY = SimpleNamespace(
            LineString=LineString, Point=Point, Polygon=Polygon, box=box,
            orient=orient, split=split, substring=substring, unary_union=unary_union,
        )
    return _SHAPELY


# --------------------------------------------------------------------------------------
# Data structures (picklable: numpy arrays and builtins only)
# --------------------------------------------------------------------------------------

@dataclass
class InterfaceChain:
    """One connected old interface polyline, traversed from its lower to its upper end.

    ``end_types`` gives the nature of the two ends: ``"axis"`` for an end sitting on the
    symmetry axis (``x == 0``) and ``"fixed"`` for an end sitting on some other boundary,
    e.g. a wall.  ``zeta`` is the old arclength chart; pass ``None`` to have
    :func:`detect_and_plan` fill it in.
    """
    points: np.ndarray
    sizes: np.ndarray
    end_types: Tuple[str, str]
    zeta: Optional[np.ndarray] = None


@dataclass
class ReconnectionEvent:
    kind: str                       # "pinch" | "coalescence" | "removal"
    z_center: float                 # waist / gap-center axial position
    zeta_info: Dict[str, float] = field(default_factory=dict)
    parents: List[int] = field(default_factory=list)
    children: List[int] = field(default_factory=list)


@dataclass
class NewChain:
    points: np.ndarray              # (M,2)
    sizes: np.ndarray               # (M,)
    zeta: np.ndarray                # (M,) strictly increasing
    origin: np.ndarray              # (M,) int, >=0 index into the concatenated old points
    end_types: Tuple[str, str] = ("axis", "axis")


@dataclass
class SurgeryPlan:
    events: List[ReconnectionEvent]
    new_chains: List[NewChain]
    axis_spans_inside: np.ndarray   # (K,2)
    axis_spans_outside: np.ndarray  # (L,2)
    fragment_volumes_before: List[float]
    fragment_volumes_after: List[float]
    volume_lost_by_removal: float


# --------------------------------------------------------------------------------------
# Volume of a revolved closed half-section polyline
# --------------------------------------------------------------------------------------

def revolved_volume(closed_halfsection_points: Any) -> float:
    """Exact volume of the body obtained by revolving a closed half-section polyline.

    The polyline is closed implicitly (last point connects back to the first).  Each
    linear segment ``(r_i,z_i) -> (r_j,z_j)`` revolved about ``r = 0`` contributes the
    frustum term ``pi * (r_i^2 + r_i r_j + r_j^2)/3 * (z_j - z_i)``; the signed sum over a
    closed loop is the enclosed volume, so the result is exact for a polygonal section and
    independent of orientation once the absolute value is taken.  Segments lying on the
    axis contribute nothing.
    """
    p = np.asarray(closed_halfsection_points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 3:
        return 0.0
    r0 = p[:, 0]
    z0 = p[:, 1]
    r1 = np.roll(r0, -1)
    z1 = np.roll(z0, -1)
    return float(abs(np.sum((r0 * r0 + r0 * r1 + r1 * r1) / 3.0 * (z1 - z0)) * math.pi))


def _closed_section(points: np.ndarray, end_types: Sequence[str]) -> np.ndarray:
    """Close a half-section polyline down onto the axis at its ``"fixed"`` ends."""
    extra: List[Tuple[float, float]] = []
    if end_types[1] == "fixed":
        extra.append((0.0, float(points[-1, 1])))
    if end_types[0] == "fixed":
        extra.append((0.0, float(points[0, 1])))
    if not extra:
        return points
    return np.vstack([points, np.array(extra, dtype=float)])


def _poly_volume(poly: Any) -> float:
    """Revolved volume of a half-plane (x>=0) polygon, from its exterior ring."""
    return revolved_volume(np.asarray(poly.exterior.coords)[:-1])


# --------------------------------------------------------------------------------------
# Morphology
# --------------------------------------------------------------------------------------
#
# Derivation of the structuring-element radii.
#
# *Opening* (erosion followed by dilation, `P.buffer(-e).buffer(+e)`) deletes exactly
# those parts of the fluid that cannot contain an open disc of radius `e`.  Because the
# half-section has been mirrored about the axis, a neck whose minimal interface radius is
# `r_w` is a full-width strip of half-width `r_w`, whose medial axis is the symmetry axis
# itself.  Erosion by `e` therefore empties the neck (and hence disconnects the fragment)
# precisely when `r_w < e`.  So `eps_p = rmin_nd`, NOT `rmin_nd/2`: the mirroring already
# supplies the factor of two.  With that choice the pinch threshold is exactly
# "minimal interface radius below rmin_nd".
#
# *Closing* (dilation followed by erosion, `Q.buffer(+e).buffer(-e)`) fills exactly those
# parts of the complement that cannot contain an open disc of radius `e`.  Two fragments
# separated by an axial gap of length `d` each grow by `e` under the dilation, so they
# touch when `2 e >= d`.  So `eps_c = distmin_nd/2` and the coalescence threshold is
# exactly "tip-to-tip gap below distmin_nd".
#
# The three-step composites `+e,-2e,+e` and `-e,+2e,-e` were rejected on purpose.
# `P.buffer(+e).buffer(-2e).buffer(+e)` is not an opening: regrouping it as
# `((P (+e) (-e)) (-e) (+e))` shows it is the opening OF THE CLOSING, i.e. it silently
# bridges every gap shorter than `2 eps_p` as well.  That couples the two thresholds and
# makes the combined neck+gap case ambiguous (a pinch would immediately be re-bridged by
# its own detector).  The plain two-step forms above are the textbook opening/closing,
# they commute with the intended thresholds, and they are what the unit tests pin down.
#
# Two genuine caveats remain, and both are diagnosed rather than papered over.
#
# (i) It is the *erosion* that carries the topology; the dilation which completes the
#     opening re-glues two eroded components whenever they end up closer than `2 eps_p`,
#     i.e. whenever the waist is axially shorter than the structuring element.  A physical
#     pinch-off neck is long and slender, so this only shows up when rmin_nd is coarse
#     compared with the axial extent of the waist.  The component counts of the erosion and
#     of the opening are compared and a mismatch raises rather than silently reporting no
#     event.  With that guarded, the pinch threshold is exactly "minimum interface radius
#     below rmin_nd".
#
# (ii) If the axial gap that the opening carves out at a waist is itself shorter than
#      `distmin_nd`, the subsequent closing would immediately re-merge the two pinch
#      children.  That is a contradictory parameter choice (rmin_nd too large relative to
#      distmin_nd), and it raises too.

def _open(poly: Any, eps: float, quad_segs: int) -> Any:
    return poly.buffer(-eps, quad_segs=quad_segs, join_style="round").buffer(
        eps, quad_segs=quad_segs, join_style="round")


def _close(poly: Any, eps: float, quad_segs: int) -> Any:
    return poly.buffer(eps, quad_segs=quad_segs, join_style="round").buffer(
        -eps, quad_segs=quad_segs, join_style="round")


def _polygons(geom: Any) -> List[Any]:
    """Flatten a shapely geometry into its (nonempty) polygon components."""
    sh = _require_shapely()
    out: List[Any] = []
    if geom.is_empty:
        return out
    if geom.geom_type == "Polygon":
        out.append(sh.orient(geom))
    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        for g in geom.geoms:
            if g.geom_type == "Polygon" and not g.is_empty and g.area > 0.0:
                out.append(sh.orient(g))
    return out


def _match(children: List[Any], parents: List[Any]) -> List[int]:
    """For each child polygon, the index of the parent it overlaps most (-1 if none)."""
    res: List[int] = []
    for c in children:
        best, best_a = -1, 0.0
        for i, p in enumerate(parents):
            try:
                a = c.intersection(p).area
            except Exception:
                a = 0.0
            if a > best_a:
                best, best_a = i, a
        res.append(best)
    return res


# --------------------------------------------------------------------------------------
# Half-plane extraction
# --------------------------------------------------------------------------------------

def _halfplane(poly: Any, clip: Any, snap: float) -> Any:
    sh = _require_shapely()
    h = poly.intersection(clip)
    polys = _polygons(h)
    if not polys:
        return None
    # Numerically the mirror symmetry is only approximate, so take the dominant part.
    polys.sort(key=lambda p: p.area, reverse=True)
    p = polys[0]
    c = np.asarray(p.exterior.coords)
    c[np.abs(c[:, 0]) < snap, 0] = 0.0
    c[c[:, 0] < 0.0, 0] = 0.0
    return sh.orient(sh.Polygon(c))


def _interface_curve(hpoly: Any, snap: float, eps: float) -> np.ndarray:
    """The single interface run (x>0) of a half-plane fragment, ordered ascending z.

    The returned curve starts and ends exactly on the axis: the axis vertices adjacent to
    the run are included with ``x`` forced to zero.
    """
    ring = np.asarray(hpoly.exterior.coords)[:-1]
    n = len(ring)
    on = ring[:, 0] <= snap
    if on.all():
        raise RuntimeError("axisymmetric topology: degenerate fragment with no interface")
    if not on.any():
        raise RuntimeError(
            "axisymmetric topology: a fragment does not touch the symmetry axis "
            "(toroidal / detached topology is not supported)")
    # cyclic maximal runs of x > snap
    start = int(np.argmax(on))  # first on-axis index -> runs do not wrap from here
    runs: List[List[int]] = []
    cur: List[int] = []
    for k in range(n):
        i = (start + k) % n
        if on[i]:
            if cur:
                runs.append(cur)
                cur = []
        else:
            cur.append(i)
    if cur:
        runs.append(cur)
    if not runs:
        raise RuntimeError("axisymmetric topology: degenerate fragment with no interface")
    runs.sort(key=lambda r: float(np.max(ring[r, 0])), reverse=True)
    main = runs[0]
    for r in runs[1:]:
        if float(np.max(ring[r, 0])) > 0.1 * eps:
            raise RuntimeError(
                "axisymmetric topology: fragment touches the symmetry axis in more than "
                "one interval (unsupported topology)")
    i0, i1 = main[0], main[-1]
    pre = (i0 - 1) % n
    post = (i1 + 1) % n
    idx = [pre] + main + [post]
    c = ring[idx].copy()
    c[0, 0] = 0.0
    c[-1, 0] = 0.0
    if c[0, 1] > c[-1, 1]:
        c = c[::-1].copy()
    return c


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------

def detect_and_plan(chains: List[InterfaceChain],
                    rmin_nd: Optional[float],
                    distmin_nd: Optional[float],
                    *,
                    buffer_resolution: int = 8,
                    cap_window_factor: float = 6.0,
                    cap_spacing_factor: float = 0.7,
                    volume_conservation: bool = True,
                    volume_tolerance: float = 1e-9,
                    allow_fragment_removal: bool = True,
                    segment_jump_offset: float = 1.0) -> Optional[SurgeryPlan]:
    """Detect axisymmetric pinch-off / coalescence and plan the interface surgery.

    Returns ``None`` when no topological change is detected, in which case the input is
    left untouched.  Otherwise a :class:`SurgeryPlan` is returned whose ``new_chains``
    reproduce the old points bit-identically everywhere outside the event windows.

    .. note::
       The input is *normalized in place*: each chain's ``points``/``sizes``/``zeta`` are
       reversed if the chain was given from its upper to its lower end, and the ``chains``
       list itself is sorted by lower-end ``z``.  This is what makes ``NewChain.origin``
       well defined -- it indexes ``numpy.concatenate([c.points for c in chains])`` as the
       list stands *after* the call.
    """
    sh = _require_shapely()
    if not chains:
        return None
    if rmin_nd is None and distmin_nd is None:
        return None

    # ---- 1. normalize input and build the old zeta chart -------------------------------
    for ch in chains:
        ch.points = np.asarray(ch.points, dtype=float)
        ch.sizes = np.asarray(ch.sizes, dtype=float)
        if ch.points[0, 1] > ch.points[-1, 1]:
            ch.points = ch.points[::-1].copy()
            ch.sizes = ch.sizes[::-1].copy()
            ch.end_types = (ch.end_types[1], ch.end_types[0])
            if ch.zeta is not None:
                ch.zeta = np.asarray(ch.zeta, dtype=float)[::-1].copy()
    chains.sort(key=lambda c: float(c.points[0, 1]))

    allp = np.vstack([c.points for c in chains])
    extent = float(max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1]), 1e-30))
    snap = 1e-9 * extent
    for ch in chains:
        ch.points[np.abs(ch.points[:, 0]) < snap, 0] = 0.0

    zoff = 0.0
    for ch in chains:
        if ch.zeta is None:
            seg = np.linalg.norm(np.diff(ch.points, axis=0), axis=1)
            ch.zeta = zoff + np.concatenate([[0.0], np.cumsum(seg)])
        else:
            ch.zeta = np.asarray(ch.zeta, dtype=float)
        zoff = float(ch.zeta[-1]) + segment_jump_offset

    offs = np.cumsum([0] + [len(c.points) for c in chains])
    old_pts = np.vstack([c.points for c in chains])
    old_sizes = np.concatenate([c.sizes for c in chains])
    old_zeta = np.concatenate([cast(np.ndarray, c.zeta) for c in chains])  # zeta filled by the loop above
    old_chain_of = np.concatenate([np.full(len(c.points), k, dtype=int)
                                   for k, c in enumerate(chains)])
    fixed_ends: List[Tuple[int, np.ndarray]] = []
    for k, ch in enumerate(chains):
        if ch.end_types[0] == "fixed":
            fixed_ends.append((int(offs[k]), ch.points[0]))
        if ch.end_types[1] == "fixed":
            fixed_ends.append((int(offs[k + 1]) - 1, ch.points[-1]))
    fixed_gidx = set(g for g, _ in fixed_ends)

    # ---- 2. polygon construction -------------------------------------------------------
    rings = [sh.Polygon(_ring_coords(ch)) for ch in chains]
    for r in rings:
        if not r.is_valid:
            raise RuntimeError("axisymmetric topology: an input chain produces a "
                               "self-intersecting cross section")
    P_geom = sh.unary_union(rings)
    P = _polygons(P_geom)
    if not P:
        return None
    for p in P:
        if len(p.interiors) > 0:
            raise RuntimeError("axisymmetric topology: the input cross section already "
                               "encloses a hole (entrapped opposite phase); unsupported")

    eps_p = float(rmin_nd) if rmin_nd is not None else 0.0
    eps_c = 0.5 * float(distmin_nd) if distmin_nd is not None else 0.0
    eps_ref = max(eps_p, eps_c)
    qs = int(buffer_resolution)

    # ---- 3. morphology -----------------------------------------------------------------
    P_union = sh.unary_union(P)
    if eps_p > 0.0:
        eroded = [e for e in _polygons(P_union.buffer(-eps_p, quad_segs=qs,
                                                      join_style="round"))
                  if e.area > 0.1 * eps_p * eps_p]
        Q = _polygons(_open(P_union, eps_p, qs))
        # The erosion is what carries the topology; the dilation that follows it can glue
        # two eroded components back together if they are less than 2*eps_p apart, i.e. if
        # the neck is axially shorter than the structuring element. A physical pinch-off
        # neck is slender and long, so this only happens when rmin_nd is coarse relative to
        # the axial extent of the waist -- say so instead of silently reporting no event.
        seen_e: Dict[int, int] = {}
        for ei, pi in enumerate(_match(eroded, Q)):
            if pi < 0:
                continue
            if pi in seen_e:
                b = eroded[ei].bounds
                raise RuntimeError(
                    "axisymmetric topology: a neck near z={:g} is thinner than "
                    "rmin_nd={} but axially shorter than 2*rmin_nd, so the opening "
                    "re-bridges it; reduce rmin_nd below half the axial extent of the "
                    "waist".format(0.5 * (b[1] + b[3]), rmin_nd))
            seen_e[pi] = ei
    else:
        Q = list(P)
    q2p = _match(Q, P)
    if any(i < 0 for i in q2p):
        raise RuntimeError("axisymmetric topology: internal error, an opened fragment "
                           "has no parent")
    children_of: List[List[int]] = [[] for _ in P]
    for qi, pi in enumerate(q2p):
        children_of[pi].append(qi)

    removed = [pi for pi, cs in enumerate(children_of) if not cs]
    volume_lost = 0.0

    clip = sh.box(0.0, float(np.min(allp[:, 1])) - 10.0 * extent - 1.0,
                  float(np.max(allp[:, 0])) + 10.0 * extent + 1.0,
                  float(np.max(allp[:, 1])) + 10.0 * extent + 1.0)
    P_half = [_halfplane(p, clip, snap) for p in P]
    vol_before = [_poly_volume(h) if h is not None else 0.0 for h in P_half]

    if removed:
        if not allow_fragment_removal:
            raise RuntimeError(
                "axisymmetric topology: fragment(s) {} would vanish entirely under the "
                "rmin_nd criterion, but allow_fragment_removal=False".format(removed))
        volume_lost = float(sum(vol_before[pi] for pi in removed))

    if eps_c > 0.0 and Q:
        R = _polygons(_close(sh.unary_union(Q), eps_c, qs))
    else:
        R = list(Q)
    if not R:
        raise RuntimeError("axisymmetric topology: all fluid vanished; refusing to plan")
    q2r = _match(Q, R)
    merged_of: List[List[int]] = [[] for _ in R]
    for qi, ri in enumerate(q2r):
        if ri < 0:
            raise RuntimeError("axisymmetric topology: internal error, a fragment was "
                               "lost by the closing step")
        merged_of[ri].append(qi)

    for ri, qq in enumerate(merged_of):
        seen: Dict[int, int] = {}
        for qi in qq:
            pi = q2p[qi]
            if pi in seen:
                raise RuntimeError(
                    "axisymmetric topology: distmin_nd={} re-bridges the gap that "
                    "rmin_nd={} just opened at a waist (fragment {}). Choose distmin_nd "
                    "smaller than the axial extent of the pinched neck.".format(
                        distmin_nd, rmin_nd, pi))
            seen[pi] = qi

    for r in R:
        if len(r.interiors) > 0:
            raise RuntimeError("axisymmetric topology: the reconnection would enclose a "
                               "hole (entrapped opposite phase); unsupported topology")

    n_pinch = sum(1 for cs in children_of if len(cs) > 1)
    n_merge = sum(1 for qq in merged_of if len(qq) > 1)
    if n_pinch == 0 and n_merge == 0 and not removed:
        return None

    # ---- 4. half-plane extraction ------------------------------------------------------
    R_half = []
    for r in R:
        h = _halfplane(r, clip, snap)
        if h is None:
            raise RuntimeError("axisymmetric topology: a fragment vanished when clipped "
                               "to the half plane x>=0")
        R_half.append(h)
    curves = [_interface_curve(h, max(snap, 1e-12 * extent), max(eps_ref, snap))
              for h in R_half]
    order = np.argsort([float(c[0, 1]) for c in curves])
    R = [R[i] for i in order]
    R_half = [R_half[i] for i in order]
    curves = [curves[i] for i in order]
    remap = {int(o): i for i, o in enumerate(order)}
    merged_of = [merged_of[int(o)] for o in order]
    q2r = [remap[i] for i in q2r]

    # ---- 5. change region and fixed-end safety ----------------------------------------
    blobs = _change_blobs(sh.unary_union(P), sh.unary_union(R), eps_p, eps_c, eps_ref)
    for gidx, pt in fixed_ends:
        # The whole synthetic wall closure (and its mirror) is what must stay clear of an
        # event, not just the contact point: a waist a few eps above a wall still lands on
        # the closure even though it is far from the contact line itself.
        p = sh.LineString([(-float(pt[0]), float(pt[1])), (float(pt[0]), float(pt[1]))])
        for blob, be in blobs:
            if blob.distance(p) < 4.0 * be:
                raise RuntimeError(
                    "axisymmetric topology: a reconnection event occurs within 4*eps of "
                    "the fixed (non-axis) chain end at r={:g}, z={:g}; surgery at a wall "
                    "contact line is not supported".format(float(pt[0]), float(pt[1])))

    # ---- 6. events ---------------------------------------------------------------------
    events: List[ReconnectionEvent] = []
    waists: List[Tuple[float, float]] = []   # (z_center, zeta_waist)
    for pi, cs in enumerate(children_of):
        if len(cs) < 2:
            continue
        kids = sorted(cs, key=lambda qi: Q[qi].bounds[1])
        for a, b in zip(kids[:-1], kids[1:]):
            zlo = float(Q[a].bounds[3])
            zhi = float(Q[b].bounds[1])
            zc = 0.5 * (zlo + zhi)
            zw = _waist_zeta(old_pts, old_zeta, zlo, zhi, eps_p)
            waists.append((zc, zw))
            events.append(ReconnectionEvent(
                kind="pinch", z_center=zc, zeta_info={"zeta_waist": zw},
                parents=[pi], children=sorted({q2r[a], q2r[b]})))
    for ri, qq in enumerate(merged_of):
        if len(qq) < 2:
            continue
        kids = sorted(qq, key=lambda qi: Q[qi].bounds[1])
        for a, b in zip(kids[:-1], kids[1:]):
            zc = 0.5 * (float(Q[a].bounds[3]) + float(Q[b].bounds[1]))
            zl = _tip_zeta(chains, float(Q[a].bounds[3]))
            zu = _tip_zeta(chains, float(Q[b].bounds[1]))
            events.append(ReconnectionEvent(
                kind="coalescence", z_center=zc,
                zeta_info={"zeta_lower_tip": zl, "zeta_upper_tip": zu},
                parents=sorted({q2p[a], q2p[b]}), children=[ri]))
    for pi in removed:
        b = P[pi].bounds
        events.append(ReconnectionEvent(kind="removal", z_center=0.5 * (b[1] + b[3]),
                                        zeta_info={}, parents=[pi], children=[]))

    # ---- 7. which old points survive, and on which new fragment? ----------------------
    #
    # An old point survives unless it sits inside an event window.  Ownership is decided
    # by the *nearest* new fragment rather than by an on-the-curve test, because an
    # opening also cosmetically rounds convex corners (wall contact lines) and over-sharp
    # axial tips with radius eps -- displacements of up to ~0.4*eps that must not cost
    # those points their identity.  Survivors are ordered by their old global index, which
    # is exactly the traversal order: every fragment is a union of contiguous runs of old
    # chains taken in ascending chain and point order.
    lines = [sh.LineString(c) for c in curves]
    owner = np.full(len(old_pts), -1, dtype=int)
    tol_keep = max(0.6 * eps_ref, 100.0 * snap)
    # The exclusion radius is deliberately the same generous cap_window_factor*eps used to decide
    # whether a curve end is a fresh tip, even though it costs the identity of every old point within
    # 6*eps of the event: the volume correction has only the fresh points to work with, and on a
    # narrower window it either drives the cap through the symmetry axis or closes the pinch gap it
    # has just opened (both pinned by tests here).  What the wide window does NOT excuse is a zeta
    # chart that stops following the old arclength across it - see _splice_fragment.
    for g in range(len(old_pts)):
        pt = sh.Point(float(old_pts[g, 0]), float(old_pts[g, 1]))
        if any(blob.distance(pt) <= cap_window_factor * be for blob, be in blobs):
            continue
        d = [float(ln.distance(pt)) for ln in lines]
        j = int(np.argmin(d))
        if d[j] <= tol_keep:
            owner[g] = j

    # The corner where an interface meets a wall is convex, so the opening rounds it with
    # radius eps and displaces the contact point itself by eps*(1/sin(theta/2)-1) -- 0.41
    # eps at a right angle, more at an acute one, i.e. past any sane ownership tolerance.
    # Since no event is allowed within 4*eps of a wall closure (checked above), the whole
    # 4*eps zone there is simply declared surviving and inherited from the first point
    # beyond it; that keeps the wall contact point exactly where it was.
    for k, ch in enumerate(chains):
        arc = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(ch.points, axis=0), axis=1))])
        for side in (0, 1):
            if ch.end_types[side] != "fixed":
                continue
            d_end = arc if side == 0 else (arc[-1] - arc)
            zone = [int(offs[k]) + i for i in np.where(d_end <= 4.0 * eps_ref)[0]]
            probe = range(len(ch.points)) if side == 0 else range(len(ch.points) - 1, -1, -1)
            j = -1
            for i in probe:
                g = int(offs[k]) + i
                if g in zone:
                    continue
                if owner[g] >= 0:
                    j = int(owner[g])
                    break
            if j < 0:
                continue
            for g in zone:
                pt = sh.Point(float(old_pts[g, 0]), float(old_pts[g, 1]))
                if owner[g] < 0 and not any(blob.distance(pt) <= cap_window_factor * be
                                            for blob, be in blobs):
                    owner[g] = j

    new_chains: List[NewChain] = []
    for ri in range(len(R)):
        new_chains.append(_splice_fragment(
            sh, curves[ri], lines[ri], np.where(owner == ri)[0], old_pts, old_sizes,
            old_zeta, old_chain_of, chains, fixed_gidx, blobs, waists, eps_ref,
            cap_window_factor, cap_spacing_factor, extent))

    # ---- 8. volume targets and local correction ---------------------------------------
    tgt_q = _q_targets(sh, P, P_half, Q, children_of, q2p, vol_before, waists)
    targets = [float(sum(tgt_q[qi] for qi in merged_of[ri])) for ri in range(len(R))]
    if volume_conservation:
        for ri, nc in enumerate(new_chains):
            if not np.any(nc.origin < 0):
                continue
            _correct_volume(sh, nc, targets[ri], eps_ref, volume_tolerance)
    for nc in new_chains:
        if np.any(nc.points[:, 0] < -1e-12 * extent):
            raise RuntimeError("axisymmetric topology: the corrected interface crosses "
                               "the symmetry axis")
        nc.points[nc.points[:, 0] < 0.0, 0] = 0.0
        if not sh.LineString(nc.points).is_simple:
            raise RuntimeError("axisymmetric topology: the corrected interface is "
                               "self-intersecting")
        if np.any(np.diff(nc.zeta) <= 0.0):
            raise RuntimeError("axisymmetric topology: the new zeta chart is not "
                               "strictly monotone")

    vol_after = [revolved_volume(_closed_section(nc.points, nc.end_types))
                 for nc in new_chains]

    inside = np.array([[float(nc.points[0, 1]), float(nc.points[-1, 1])]
                       for nc in new_chains], dtype=float).reshape(-1, 2)
    z0 = float(min(float(c.points[0, 1]) for c in chains))
    z1 = float(max(float(c.points[-1, 1]) for c in chains))
    outside = _complement(inside, z0, z1)

    return SurgeryPlan(events=events, new_chains=new_chains,
                       axis_spans_inside=inside, axis_spans_outside=outside,
                       fragment_volumes_before=[float(v) for v in vol_before],
                       fragment_volumes_after=[float(v) for v in vol_after],
                       volume_lost_by_removal=float(volume_lost))


# --------------------------------------------------------------------------------------
# Helpers used by detect_and_plan
# --------------------------------------------------------------------------------------

def _ring_coords(ch: InterfaceChain) -> List[Tuple[float, float]]:
    """Closed full-plane ring: the chain, its synthetic wall closures, and its mirror."""
    p = ch.points
    seq: List[Tuple[float, float]] = []
    if ch.end_types[0] == "fixed":
        seq.append((0.0, float(p[0, 1])))
    seq.extend((float(a), float(b)) for a, b in p)
    if ch.end_types[1] == "fixed":
        seq.append((0.0, float(p[-1, 1])))
    seq.extend((-float(a), float(b)) for a, b in p[::-1] if a > 0.0)
    return seq


def _change_blobs(P: Any, R: Any, eps_p: float, eps_c: float,
                  eps_ref: float) -> List[Tuple[Any, float]]:
    """Components of ``P xor R`` that are genuine events rather than buffer slivers.

    Two filters are needed.

    *Slivers.*  Opening/closing are the identity away from necks and gaps only in exact
    arithmetic; shapely approximates the buffer arcs by polygons, so the boundary wanders
    by about ``eps*(1-cos(pi/(4*quad_segs)))`` everywhere and the symmetric difference is
    full of hair-thin slivers.  A real event blob has a width of order ``eps``, a sliver a
    width of order ``0.005 eps``, so the width proxy ``area/perimeter`` separates them.

    *Cosmetic reshaping.*  An opening also genuinely rounds every convex corner of the
    section with radius ``eps`` -- most notably the 90-degree corner where the interface
    meets a wall, which produces a blob of area ``(1-pi/4) eps^2`` that has nothing to do
    with any reconnection.  In this mirrored half-section every *topological* change
    necessarily happens on the symmetry axis (a neck is a full-width strip whose medial
    axis is the axis; two fragments merge along the axis), so only blobs that straddle
    ``x = 0`` are event blobs.  Filtering on that is what keeps a wall contact line from
    being reported as an event.
    """
    out: List[Tuple[Any, float]] = []
    for blob in _polygons(P.symmetric_difference(R)):
        inside_P = P.contains(blob.representative_point())
        e = (eps_p if inside_P else eps_c) or eps_ref
        if e <= 0.0:
            continue
        if blob.area < 0.02 * e * e:
            continue
        if blob.area / max(blob.length, 1e-300) < 0.02 * e:
            continue
        b = blob.bounds
        if not (b[0] < -0.05 * e and b[2] > 0.05 * e):
            continue
        out.append((blob, e))
    return out


def _waist_zeta(old_pts, old_zeta, zlo: float, zhi: float, eps: float) -> float:
    """Old zeta of the narrowest old point inside the removed neck band."""
    m = (old_pts[:, 1] >= zlo - eps) & (old_pts[:, 1] <= zhi + eps)
    if not np.any(m):
        m = np.abs(old_pts[:, 1] - 0.5 * (zlo + zhi)) < np.inf
    idx = np.where(m)[0]
    return float(old_zeta[idx[int(np.argmin(old_pts[idx, 0]))]])


def _tip_zeta(chains, z: float) -> float:
    """Old zeta of the chain end closest (in z) to the given axial position."""
    best, bd = 0.0, float("inf")
    for k, ch in enumerate(chains):
        for j, zz in ((len(ch.points) - 1, float(ch.points[-1, 1])),
                      (0, float(ch.points[0, 1]))):
            d = abs(zz - z)
            if d < bd:
                bd, best = d, float(ch.zeta[j])
    return best


def _complement(inside: np.ndarray, z0: float, z1: float) -> np.ndarray:
    if inside.size == 0:
        return np.zeros((0, 2), dtype=float)
    spans = sorted((min(a, b), max(a, b)) for a, b in inside)
    out: List[Tuple[float, float]] = []
    cur = z0
    for a, b in spans:
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < z1:
        out.append((cur, z1))
    return np.array(out, dtype=float).reshape(-1, 2)


# --------------------------------------------------------------------------------------
# Splicing
# --------------------------------------------------------------------------------------

def _splice_fragment(sh, curve: np.ndarray, line: Any, owned: np.ndarray, old_pts,
                     old_sizes, old_zeta, old_chain_of, chains, fixed_gidx, blobs, waists,
                     eps_ref: float, cap_window_factor: float, cap_spacing_factor: float,
                     extent: float) -> NewChain:
    if len(owned) == 0:
        raise RuntimeError("axisymmetric topology: a new fragment retains no old "
                           "interface point; the event windows are too wide")
    survivors: List[Tuple[float, int]] = [
        (float(line.project(sh.Point(float(old_pts[g, 0]), float(old_pts[g, 1])))), int(g))
        for g in owned]

    L = float(line.length)
    pts: List[np.ndarray] = []
    szs: List[float] = []
    zts: List[float] = []
    orig: List[int] = []
    ends = ["axis", "axis"]

    def _h(size: float) -> float:
        e = eps_ref if eps_ref > 0.0 else size
        return max(min(cap_spacing_factor * size, 0.7 * e), 1e-6 * extent)

    # --- lower end ---
    d0, g0 = survivors[0]
    if g0 in fixed_gidx:
        ends[0] = "fixed"
    elif (_near_event(sh, curve[0], blobs, cap_window_factor)
          or float(old_pts[g0, 0]) > 0.6 * eps_ref) and \
            d0 > 0.5 * _h(float(old_sizes[g0])):
        cap = _cap_window(sh, line, 0.0, d0, _h(float(old_sizes[g0])), eps_ref,
                          at_start=True)
        zw = _nearest_waist(waists, float(cap[0, 1]), float(old_zeta[g0]),
                            float(old_zeta[g0]) - 1.0)
        zf = float(old_zeta[g0])
        s = _cap_param(old_pts, old_zeta, old_chain_of, g0, cap, zw, zf)  # 0 at tip ... 1 at splice
        for k in range(len(cap) - 1):
            pts.append(cap[k])
            szs.append(float(old_sizes[g0]))
            # u = 1 - s is the fraction towards the tip; the (1 - MARGIN*u) keeps the chart short
            # of the waist by a hair, which is all it has to be now that s is the true old-chart
            # fraction (see _cap_param) rather than a uniform ramp: the previous 5% margin, spent
            # over the whole window, shifted every resampled point of it by up to 4% of the range.
            u = 1.0 - s[k]
            zts.append(zw + (1.0 - u * (1.0 - _CAP_WAIST_MARGIN * u)) * (zf - zw))
            orig.append(-1)

    # --- survivors and interior windows ---
    for i, (d, g) in enumerate(survivors):
        pts.append(old_pts[g].copy())
        szs.append(float(old_sizes[g]))
        zts.append(float(old_zeta[g]))
        orig.append(int(g))
        if i + 1 >= len(survivors):
            continue
        dn, gn = survivors[i + 1]
        if gn == g + 1 and old_chain_of[gn] == old_chain_of[g]:
            continue
        h = _h(0.5 * (float(old_sizes[g]) + float(old_sizes[gn])))
        full = _resample(sh, line, d, dn, h)
        t_full = _norm_arclen(full)     # 0 at the lower splice, 1 at the upper splice
        br, t = full[1:-1], t_full[1:-1]
        if len(br) == 0:
            continue
        ka, kb = int(old_chain_of[g]), int(old_chain_of[gn])
        zl = float(chains[ka].zeta[-1])
        zu = float(chains[kb].zeta[0])
        za, zb = float(old_zeta[g]), float(old_zeta[gn])
        bridges_two_chains = za < zl < zu < zb
        if not bridges_two_chains:
            zl = za + 0.4 * (zb - za)
            zu = za + 0.6 * (zb - za)
        # A window that stays within ONE old chain is a *resampling* of a stretch of old interface
        # that is still there, so its chart is the old arclength of that stretch - see _cap_param for
        # what a uniform ramp costs instead.  A window that bridges two old chains has no such
        # stretch to follow: it is fresh interface between two former tips, and the ramp with the
        # inter-chain offset in the middle is the point.
        win_z = None
        if not bridges_two_chains and gn > g + 1:
            sp = (_project_old_zeta(old_pts[g:gn + 1], old_zeta[g:gn + 1], br) - za) / (zb - za)
            sp = np.maximum.accumulate(np.clip(sp, 0.0, 1.0))
            sp = (1.0 - 1e-6) * sp + 1e-6 * t    # strictly increasing, as the chart must be
            win_z = za + (zb - za) * (1e-6 + (1.0 - 2e-6) * (sp - sp[0]) / max(sp[-1] - sp[0], 1e-300))
        for k in range(len(br)):
            pts.append(br[k])
            szs.append(float(old_sizes[g]) + (float(old_sizes[gn]) - float(old_sizes[g])) * t[k])
            if win_z is not None:
                zts.append(float(win_z[k]))
            elif t[k] <= 0.5:
                zts.append(za + (t[k] / 0.5) * (zl - za))
            else:
                zts.append(zu + ((t[k] - 0.5) / 0.5) * (zb - zu))
            orig.append(-1)

    # --- upper end ---
    d1, g1 = survivors[-1]
    if g1 in fixed_gidx:
        ends[1] = "fixed"
    elif (_near_event(sh, curve[-1], blobs, cap_window_factor)
          or float(old_pts[g1, 0]) > 0.6 * eps_ref) and \
            L - d1 > 0.5 * _h(float(old_sizes[g1])):
        cap = _cap_window(sh, line, d1, L, _h(float(old_sizes[g1])), eps_ref,
                          at_start=False)
        zf = float(old_zeta[g1])
        zw = _nearest_waist(waists, float(cap[-1, 1]), zf, zf + 1.0)
        t = _cap_param(old_pts, old_zeta, old_chain_of, g1, cap, zf, zw)  # 0 at splice ... 1 at tip
        for k in range(1, len(cap)):
            pts.append(cap[k])
            szs.append(float(old_sizes[g1]))
            # (1 - MARGIN*t) keeps the chart short of the waist by a hair, which is all it has to be
            # now that t is the true old-chart fraction (see _cap_param) rather than a uniform ramp:
            # the previous 0.95*t, applied over the whole window, shifted every resampled point of it
            # by up to 5% of the range.
            zts.append(zf + t[k] * (1.0 - _CAP_WAIST_MARGIN * t[k]) * (zw - zf))
            orig.append(-1)

    return NewChain(points=np.array(pts, dtype=float), sizes=np.array(szs, dtype=float),
                    zeta=np.array(zts, dtype=float), origin=np.array(orig, dtype=int),
                    end_types=(ends[0], ends[1]))


#: How far short of the waist a fresh cap's chart has to stop, as a fraction of the chart range it
#: spans.  Only a guard: an old-mesh lookup at exactly the waist zeta must not be able to land on the
#: far side of it.  See _cap_param for why this is not a place to be generous.
_CAP_WAIST_MARGIN = 1e-4


def _project_old_zeta(old_p: np.ndarray, old_z: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Old chart of arbitrary points, by projecting them onto the old polyline ``old_p``."""
    e = np.diff(old_p, axis=0)                                   # (S,2)
    l2 = np.maximum(np.einsum("ij,ij->i", e, e), 1e-300)
    d = pts[:, None, :] - old_p[None, :-1, :]                    # (N,S,2)
    u = np.clip(np.einsum("nsj,sj->ns", d, e) / l2[None, :], 0.0, 1.0)
    off = d - u[:, :, None] * e[None, :, :]
    j = np.argmin(np.einsum("nsj,nsj->ns", off, off), axis=1)
    rows = np.arange(len(pts))
    return old_z[j] + u[rows, j] * (old_z[j + 1] - old_z[j])


def _cap_param(old_pts: np.ndarray, old_zeta: np.ndarray, old_chain_of: np.ndarray, g: int,
               cap: np.ndarray, z_at_0: float, z_at_1: float) -> np.ndarray:
    """Where each point of a cap window sits between its two chart ends, 0 at ``z_at_0``, 1 at ``z_at_1``.

    The obvious answer, normalized arclength along the cap, is wrong, and by more than it looks.  The
    cap window is as wide as the survivor exclusion, i.e. ``cap_window_factor*eps`` -- deliberately
    so, because the volume correction has only the fresh points to work with -- and on a dumbbell
    that is most of the fragment.  Nearly all of those points therefore lie exactly on interface the
    surgery never touched, and giving them a chart that ramps uniformly along the new curve shifts
    them against the old one: measured at h = 0.06, by 0.05 in arclength, about one element, all the
    way down to the far tip.  A surface field transferred through that chart comes out shifted by
    the same amount.

    So the parameter is measured in the OLD chart instead: each cap point is projected back onto the
    stretch of old interface between the splice and the waist and reports where it landed.  For the
    points that are a resampling of unchanged interface that IS their old zeta; the genuinely fresh
    ones near the tip land on whatever old interface is closest to them, which is the neck material
    the tip is made of.  Deliberately NOT rescaled to reach 1 at the tip - that would stretch the
    whole window by however far short of the waist the tip's own projection falls.
    """
    lo, hi = (z_at_0, z_at_1) if z_at_0 < z_at_1 else (z_at_1, z_at_0)
    m = (old_chain_of == old_chain_of[g]) & (old_zeta >= lo) & (old_zeta <= hi)
    idx = np.where(m)[0]
    a = _norm_arclen(cap)
    if len(idx) < 2 or z_at_1 == z_at_0:
        return a   # nothing of the old interface under this cap; it is fresh all the way
    s = (_project_old_zeta(old_pts[idx], old_zeta[idx], cap) - z_at_0) / (z_at_1 - z_at_0)
    s = np.maximum.accumulate(np.clip(s, 0.0, 1.0))
    # Strictly increasing, which the plan's chart has to be: a trace of the (strictly increasing)
    # arclength parameter breaks the ties the saturation above produces, without moving anything.
    return (1.0 - 1e-6) * s + 1e-6 * a


def _near_event(sh, pt: np.ndarray, blobs, factor: float) -> bool:
    """Is this curve end an axial tip that an event actually created?

    Only then does the end deserve a fresh cap.  A tip whose old points merely got nudged
    by the cosmetic rounding of the opening keeps its old geometry instead.
    """
    p = sh.Point(float(pt[0]), float(pt[1]))
    return any(blob.distance(p) <= factor * be for blob, be in blobs)


def _nearest_waist(waists, z_tip: float, zeta_splice: float, fallback: float) -> float:
    if not waists:
        return fallback
    zc, zw = min(waists, key=lambda w: abs(w[0] - z_tip))
    if (zw - zeta_splice) * (fallback - zeta_splice) <= 0.0:
        return fallback
    return zw


def _norm_arclen(p: np.ndarray) -> np.ndarray:
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    return d / max(d[-1], 1e-300)


def _resample(sh, line, da: float, db: float, h: float) -> np.ndarray:
    sub = sh.substring(line, da, db)
    if sub.geom_type != "LineString":
        sub = sh.LineString([line.interpolate(da), line.interpolate(db)])
    L = float(sub.length)
    n = max(1, int(round(L / max(h, 1e-300))))
    return np.array([[q.x, q.y] for q in (sub.interpolate(k * L / n) for k in range(n + 1))])


def _cap_window(sh, line, da: float, db: float, h: float, eps: float,
                at_start: bool) -> np.ndarray:
    """Resample an end window and replace its near-tip part by a perpendicular sqrt cap.

    Returned points always run splice -> tip for ``at_start=False`` and tip -> splice for
    ``at_start=True``; the tip has ``x == 0`` exactly.
    """
    c = _resample(sh, line, da, db, h)
    if at_start:
        c = c[::-1].copy()             # work splice -> tip
    tip_z = float(c[-1, 1])
    if eps <= 0.0 or len(c) < 3:
        c[-1, 0] = 0.0
        return c[::-1].copy() if at_start else c

    # The clipped buffer arc meets x=0 at a finite angle because it is polygonal. Near a
    # smooth axial tip the true section obeys x^2 = c1*(y_tip-y) (a paraboloid in u=x^2),
    # so fit u=x^2 linearly against y on the *outer* part of the tip zone -- the inner part
    # is exactly where the polygonal artefact sits -- and extrapolate to u=0 for the apex.
    dz = np.abs(c[:, 1] - tip_z)
    zone = np.where(dz <= 2.0 * eps)[0]
    zone = zone[zone > 0]
    if len(zone) < 3:
        zone = np.arange(max(1, len(c) - 4), len(c))
    fit = [j for j in zone if dz[j] >= 0.5 * eps and c[j, 0] > 0.0]
    if len(fit) < 2:
        fit = [j for j in zone if c[j, 0] > 0.0]
    if len(fit) < 2:
        c[-1, 0] = 0.0
        return c[::-1].copy() if at_start else c
    y = c[fit, 1]
    u = c[fit, 0] ** 2
    c1, c0 = np.polyfit(y, u, 1)
    if abs(c1) < 1e-300:
        c[-1, 0] = 0.0
        return c[::-1].copy() if at_start else c
    y_tip = -c0 / c1
    sgn = 1.0 if tip_z > float(c[0, 1]) else -1.0
    if not (sgn * (y_tip - float(y.min() if sgn > 0 else y.max())) > 0.0) or \
            abs(y_tip - tip_z) > 2.0 * eps:
        y_tip = tip_z
        c0 = -c1 * y_tip

    j0 = int(zone[0])
    x0 = float(c[j0, 0])
    if x0 <= 0.0:
        c[-1, 0] = 0.0
        return c[::-1].copy() if at_start else c
    # Uniform radial spacing over the bulk of the cap, then a short geometric ramp so that
    # the final segment is within a degree of perpendicular: the angle to the axis of the
    # last segment is atan(|c1|/x_last), so x_last <= |c1|/tan(89deg) is what it takes.
    n = max(2, int(round(x0 / h)))
    xs = list(np.linspace(x0, 0.0, n + 1)[1:-1])
    x_last = min(0.5 * h, abs(c1) / math.tan(math.radians(89.0)))
    x = xs[-1] if xs else x0
    while x > x_last * 1.0001 and len(xs) < n + 12:
        x *= 0.35
        xs.append(x)
    tail = [(0.0, y_tip)]
    for xv in xs:
        tail.insert(-1, (xv, (xv * xv - c0) / c1))
    out = np.vstack([c[:j0 + 1], np.array(tail, dtype=float)])
    return out[::-1].copy() if at_start else out


# --------------------------------------------------------------------------------------
# Volume targets and local correction
# --------------------------------------------------------------------------------------

def _q_targets(sh, P, P_half, Q, children_of, q2p, vol_before, waists) -> List[float]:
    tgt = [0.0] * len(Q)
    for pi, cs in enumerate(children_of):
        if len(cs) == 1:
            tgt[cs[0]] = float(vol_before[pi])
            continue
        if not cs:
            continue
        kids = sorted(cs, key=lambda qi: Q[qi].bounds[1])
        zsplit = []
        for a, b in zip(kids[:-1], kids[1:]):
            zsplit.append(0.5 * (float(Q[a].bounds[3]) + float(Q[b].bounds[1])))
        parts = [P_half[pi]]
        b = P_half[pi].bounds
        for zc in zsplit:
            cut = sh.LineString([(b[0] - 1.0, zc), (b[2] + 1.0, zc)])
            nxt = []
            for pp in parts:
                try:
                    nxt.extend(_polygons(sh.split(pp, cut)))
                except Exception:
                    nxt.append(pp)
            parts = nxt
        parts.sort(key=lambda pp: pp.bounds[1])
        if len(parts) == len(kids):
            for qi, pp in zip(kids, parts):
                tgt[qi] = _poly_volume(pp)
        else:
            # Fall back on an overlap match so the targets still sum to the parent volume.
            tot = float(vol_before[pi])
            areas = [Q[qi].area for qi in kids]
            s = sum(areas) or 1.0
            for qi, a in zip(kids, areas):
                tgt[qi] = tot * a / s
    return tgt


def _normals(pts: np.ndarray, poly: Any, sh) -> np.ndarray:
    t = np.zeros_like(pts)
    t[1:-1] = pts[2:] - pts[:-2]
    t[0] = pts[1] - pts[0]
    t[-1] = pts[-1] - pts[-2]
    ln = np.linalg.norm(t, axis=1)
    ln[ln < 1e-300] = 1.0
    t /= ln[:, None]
    n = np.stack([t[:, 1], -t[:, 0]], axis=1)
    k = len(pts) // 2
    step = 1e-6 * max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1e-30)
    if poly.contains(sh.Point(pts[k, 0] + n[k, 0] * step, pts[k, 1] + n[k, 1] * step)):
        n = -n
    # An endpoint sitting on the axis must keep x == 0 exactly, so give it the exactly
    # axial outward normal instead of the one-sided finite difference, which carries a
    # small radial component from the last (very short) cap segment.
    if abs(pts[0, 0]) < 1e-14:
        n[0] = (0.0, -1.0)
    if abs(pts[-1, 0]) < 1e-14:
        n[-1] = (0.0, 1.0)
    return n


def _correct_volume(sh, nc: NewChain, target: float, eps: float, tol: float) -> None:
    """Move only the fresh points of one fragment until its revolved volume hits target."""
    fresh = nc.origin < 0
    if not np.any(fresh) or target <= 0.0:
        return
    poly = sh.Polygon(_closed_section(nc.points, nc.end_types))
    if not poly.is_valid:
        poly = poly.buffer(0)
    normals = _normals(nc.points, poly, sh)

    # One weight per fresh window: caps use a smoothstep (C1 at the splice, full weight at
    # the tip, whose normal is axial so the tip slides along the axis and keeps x=0 and the
    # 90-degree contact); interior bridges use sin^2, which vanishes at both splices.
    w = np.zeros(len(nc.points))
    for a, b in _runs(fresh):
        seg = nc.points[max(a - 1, 0):min(b + 2, len(nc.points))]
        s = _norm_arclen(seg)
        lo = 1 if a > 0 else 0
        loc = s[lo:lo + (b - a + 1)]
        if a == 0:                       # cap at the lower end: tip is at t=0
            t = 1.0 - loc / max(loc[-1], 1e-300)
            w[a:b + 1] = t * t * (3.0 - 2.0 * t)
        elif b == len(nc.points) - 1:    # cap at the upper end: tip is at t=1
            t = (loc - loc[0]) / max(1.0 - loc[0], 1e-300)
            w[a:b + 1] = t * t * (3.0 - 2.0 * t)
        else:
            t = loc
            w[a:b + 1] = np.sin(math.pi * t) ** 2

    base = nc.points.copy()
    disp = (w[:, None] * normals)

    def vol(A: float) -> float:
        return revolved_volume(_closed_section(base + A * disp, nc.end_types)) - target

    span = 2.0 * (eps if eps > 0.0 else 1.0)
    for _ in range(2):
        fa, fb = vol(-span), vol(span)
        if fa * fb <= 0.0:
            A = cast(float, brentq(vol, -span, span, xtol=1e-15 * max(span, 1.0),
                                   rtol=8.9e-16, maxiter=200))  # type:ignore[arg-type] # scipy stubs over-restrict rtol and pretend full_output
            res = abs(vol(A))
            if res > tol * target:
                raise RuntimeError(
                    "axisymmetric topology: the volume correction stalled at a relative "
                    "residual of {:g}, above volume_tolerance={:g}".format(
                        res / target, tol))
            nc.points = base + A * disp
            nc.points[np.abs(nc.points[:, 0]) < 1e-15, 0] = 0.0
            return
        span *= 2.0
    raise RuntimeError(
        "axisymmetric topology: cannot conserve the volume of a reconnected fragment; "
        "the required normal offset exceeds +-{:g}".format(span))


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    a = None
    for i, m in enumerate(mask):
        if m and a is None:
            a = i
        elif not m and a is not None:
            out.append((a, i - 1))
            a = None
    if a is not None:
        out.append((a, len(mask) - 1))
    return out
