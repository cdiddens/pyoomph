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

# MixedBoxMesh3D: the box [-0.5,0.5]^3 as an N x N x N grid of cells, each cell filled by one element
# family (brick / tetrahedra / wedges / pyramids) according to a named layout. This is the 3D counterpart of
# box_cases.MixedBoxMesh and the domain the 3D campaign runs on -- the mixed meshes already in
# tests/test_mixed_3d.py are deliberately minimal 2-3 element toys built to isolate a single cross-shape
# facet, which is the wrong tool for Stokes / ALE / Neumann on a refined domain.
#
# WHICH LAYOUTS ARE GEOMETRICALLY POSSIBLE
#
# Two adjacent cells conform only if they present the same trace on their shared cube face. What each
# family presents (with the conventions used below: Kuhn tets, wedges extruded along z, pyramids with the
# apex at the cell centre):
#
#   family    x-faces   y-faces   z-faces
#   brick     quad      quad      quad
#   pyramid   quad      quad      quad
#   wedge     quad      quad      2 triangles
#   tet       2 tris    2 tris    2 triangles
#
# Hence:
#   * brick<->brick, brick<->pyramid, pyramid<->pyramid  -- any direction
#   * wedge<->wedge, tet<->tet                           -- any direction
#   * brick<->wedge, pyramid<->wedge                     -- x or y only (the wedge's quad sides)
#   * tet<->wedge                                        -- z only (both give triangles)
#   * tet<->brick, tet<->pyramid                         -- IMPOSSIBLE in any direction: a tet has no quad
#                                                           face, so no shared facet can match
#
# The triangle traces must also split along the same diagonal. They do: the Kuhn split cuts every cube face
# from the corner whose two in-plane coordinates are both minimal to the one where both are maximal, which
# is translation-invariant (so tet<->tet matches), and the wedge split along the (x,y) diagonal produces the
# same diagonal on its z-faces (so tet<->wedge matches).
#
# The tet<->brick impossibility is real geometry, not a limitation of the engine, and it is why the classic
# hex-to-tet transition needs an intermediate cell. The "trans" cell below is exactly that: one cube filled
# with ONE pyramid whose base is the cube's bottom quad face (matching a brick below) plus 10 tets filling
# the remaining five pyramidal wedges (each split into 2 tets, so all five remaining faces present matching
# triangle traces to a tet neighbour). A cell of this kind therefore has a quad bottom and triangulated
# sides/top, which is what lets brick and tet coexist in one mesh -- with pyramids AND tets inside the same
# cube.

from pyoomph.meshes.mesh import MeshTemplate

# Cube corners in TENSOR order (the order add_brick_3d_C1 expects): index = 4*bz + 2*by + bx.
_T000, _T100, _T010, _T110, _T001, _T101, _T011, _T111 = range(8)

# Kuhn (Freudenthal) split of a cube into 6 tetrahedra sharing the main diagonal 000-111, in tensor indices.
# These are wound the familiar det(p1-p0, p2-p0, p3-p0) > 0 way, which is the OPPOSITE handedness from the
# one oomph's TElement<3,N> face-normal table expects (its local frame is based at node 3). add_tetra_3d_C1
# repairs that on the way in; see dev_docs/mesh_construction.md 6 for what it cost before it did.
_KUHN = [(0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)]

# The cube's 6 faces as CYCLICALLY wound quads in tensor indices, oriented so that (quad, cell centre) is a
# positive-volume pyramid. Taken from the validated PyramidCubeMesh in tests/test_pyramid_refinement.py,
# translated from its cyclic corner array into tensor indices.
_CUBE_FACES_CYCLIC = [
    (_T000, _T100, _T110, _T010),  # z = lo
    (_T001, _T011, _T111, _T101),  # z = hi
    (_T000, _T001, _T101, _T100),  # y = lo
    (_T100, _T101, _T111, _T110),  # x = hi
    (_T110, _T111, _T011, _T010),  # y = hi
    (_T010, _T011, _T001, _T000),  # x = lo
]

_WALLS = {"left": (0, -0.5), "right": (0, 0.5), "front": (1, -0.5), "back": (1, 0.5),
          "bottom": (2, -0.5), "top": (2, 0.5)}

FAMILIES = ("hex", "tet", "wedge", "pyr", "trans")


def _layout_hex(i, j, k, N):
    return "hex"


def _layout_tet(i, j, k, N):
    return "tet"


def _layout_wedge(i, j, k, N):
    return "wedge"


def _layout_pyr(i, j, k, N):
    return "pyr"


def _layout_hex_pyr(i, j, k, N):
    # Checkerboard: both families present quads on every face, so any interface is legal.
    return "hex" if (i + j + k) % 2 == 0 else "pyr"


def _layout_hex_wedge(i, j, k, N):
    # Split along x: brick<->wedge is only legal across an x (or y) plane.
    return "hex" if i < N // 2 else "wedge"


def _layout_pyr_wedge(i, j, k, N):
    return "pyr" if i < N // 2 else "wedge"


def _layout_tet_wedge(i, j, k, N):
    # Split along z: tet<->wedge is only legal across a z plane (both present triangles there).
    return "tet" if k < N // 2 else "wedge"


def _layout_hex_pyr_wedge(i, j, k, N):
    # Wedge occupies the upper x half, so every wedge interface is an x plane; brick and pyramid split by z
    # inside the lower x half, which is legal in any direction.
    if i >= N // 2:
        return "wedge"
    return "hex" if k < N // 2 else "pyr"


def _layout_hex_tet(i, j, k, N):
    # Brick and tet cannot touch, so the z plane between them is bridged by transition cells: brick below,
    # one layer of transition cells (pyramid base down, tets above), tet above.
    if k < N // 2:
        return "hex"
    if k == N // 2:
        return "trans"
    return "tet"


def _layout_all_four(i, j, k, N):
    # All four families in one mesh. Lower x half: the brick / transition / tet stack of _layout_hex_tet,
    # which already contains pyramids inside its transition cells. Upper x half: wedges, which meet the
    # brick, transition and tet columns only across x planes... except that a wedge presents a QUAD on an x
    # face while tet and trans present triangles there, so the wedge column is restricted to the brick
    # levels and the levels above it get tets (tet<->tet across x is fine).
    if i >= N // 2:
        return "wedge" if k < N // 2 else "tet"
    if k < N // 2:
        return "hex"
    if k == N // 2:
        return "trans"
    return "tet"


LAYOUTS = {
    "hex": _layout_hex,
    "tet": _layout_tet,
    "wedge": _layout_wedge,
    "pyr": _layout_pyr,
    "hex_pyr": _layout_hex_pyr,
    "hex_wedge": _layout_hex_wedge,
    "pyr_wedge": _layout_pyr_wedge,
    "tet_wedge": _layout_tet_wedge,
    "hex_pyr_wedge": _layout_hex_pyr_wedge,
    "hex_tet": _layout_hex_tet,
    "all_four": _layout_all_four,
}

# Layouts that genuinely mix families (i.e. exclude the four pure ones), for parametrising tests.
MIXED_LAYOUTS = ["hex_pyr", "hex_wedge", "pyr_wedge", "tet_wedge", "hex_pyr_wedge", "hex_tet", "all_four"]
PURE_LAYOUTS = ["hex", "tet", "wedge", "pyr"]
ALL_LAYOUTS = PURE_LAYOUTS + MIXED_LAYOUTS


class MixedBoxMesh3D(MeshTemplate):
    """[-0.5,0.5]^3 as N^3 cells, each filled by one element family per the named layout (see LAYOUTS)."""

    def __init__(self, kind="hex", N=2, name="domain"):
        super().__init__()
        if kind not in LAYOUTS:
            raise ValueError("unknown 3d layout %r; known: %s" % (kind, sorted(LAYOUTS)))
        self.kind, self.N, self.dname = kind, N, name

    def define_geometry(self):
        N, h = self.N, 1.0 / self.N
        dom = self.new_domain(self.dname)
        layout = LAYOUTS[self.kind]
        nodes, coords, faces = {}, {}, []

        def node(x, y, z):
            key = (round(x, 9), round(y, 9), round(z, 9))
            if key not in nodes:
                nodes[key] = self.add_node_unique(x, y, z)
                coords[nodes[key]] = key
            return nodes[key]

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    x0, y0, z0 = -0.5 + i * h, -0.5 + j * h, -0.5 + k * h
                    # tensor-order cube corners
                    c = [node(x0 + bx * h, y0 + by * h, z0 + bz * h)
                         for bz in (0, 1) for by in (0, 1) for bx in (0, 1)]
                    ctr = (x0 + h / 2, y0 + h / 2, z0 + h / 2)
                    self._fill_cell(dom, layout(i, j, k, N), c, ctr, node, faces, coords)

        # A face lies on a wall iff all its nodes share that wall's coordinate. The domain is a box and all
        # element faces are planar, so this cannot mistake an interior face for a boundary one.
        for f in faces:
            for name, (axis, val) in _WALLS.items():
                if all(abs(coords[n][axis] - val) < 1e-9 for n in f):
                    self.add_facet_to_boundary(name, list(f))
                    break

    def _fill_cell(self, dom, family, c, ctr, node, faces, coords):
        if family == "hex":
            dom.add_brick_3d_C1(*c)
            faces.extend([tuple(c[q] for q in f) for f in _CUBE_FACES_CYCLIC])
        elif family == "tet":
            for t in _KUHN:
                nn = [c[q] for q in t]
                dom.add_tetra_3d_C1(*nn)
                # All four faces; the boundary test below keeps only the ones on a wall.
                faces.extend([(nn[1], nn[2], nn[3]), (nn[0], nn[2], nn[3]),
                              (nn[0], nn[1], nn[3]), (nn[0], nn[1], nn[2])])
        elif family == "wedge":
            # Two prisms extruded along z, the cube's (x,y) square cut along the 00-11 diagonal -- the same
            # diagonal the Kuhn split leaves on a z face, which is what makes tet<->wedge conform.
            lo = [c[_T000], c[_T100], c[_T110], c[_T010]]
            hi = [c[_T001], c[_T101], c[_T111], c[_T011]]
            for w in ([lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]],
                      [lo[0], lo[2], lo[3], hi[0], hi[2], hi[3]]):
                dom.add_wedge_3d_C1(*w)
                # 2 triangular caps + 3 quad sides, in wedge-local indices.
                for fl in ((0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (0, 3, 5, 2), (1, 2, 5, 4)):
                    faces.append(tuple(w[q] for q in fl))
        elif family == "pyr":
            apex = node(*ctr)
            for f in _CUBE_FACES_CYCLIC:
                q = [c[i] for i in f]
                dom.add_pyramid_3d_C1(q[0], q[1], q[2], q[3], apex)
                faces.append(tuple(q))
                faces.extend([(q[0], q[1], apex), (q[1], q[2], apex),
                              (q[2], q[3], apex), (q[3], q[0], apex)])
        elif family == "trans":
            self._fill_transition(dom, c, ctr, node, faces, coords)
        else:
            raise ValueError("unknown family " + str(family))

    def _fill_transition(self, dom, c, ctr, node, faces, coords):
        """Brick-to-tet transition cell: the bottom (z=lo) face stays a QUAD, carried by a single pyramid
        with the cell centre as apex, so a brick can sit below it. The other five cube faces are split into
        two triangles each so that tets can sit beside and above -- each of the remaining five pyramids is
        cut into 2 tets. The split must use the SAME diagonal the Kuhn cells use, i.e. from the corner whose
        two in-plane coordinates are both minimal to the one where both are maximal; _diag_first() rotates
        each cyclic quad so that its first node is that minimal corner, which makes (q0,q1,q2)+(q0,q2,q3)
        cut along exactly that diagonal."""
        apex = node(*ctr)
        for idx, f in enumerate(_CUBE_FACES_CYCLIC):
            q = [c[i] for i in f]
            if idx == 0:  # z = lo: keep the quad, so a brick neighbour below conforms
                dom.add_pyramid_3d_C1(q[0], q[1], q[2], q[3], apex)
                faces.append(tuple(q))
                faces.extend([(q[0], q[1], apex), (q[1], q[2], apex),
                              (q[2], q[3], apex), (q[3], q[0], apex)])
                continue
            q = self._diag_first(q, coords)
            for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
                dom.add_tetra_3d_C1(t[0], t[1], t[2], apex)
                faces.append(t)
                faces.extend([(t[0], t[1], apex), (t[1], t[2], apex), (t[0], t[2], apex)])

    @staticmethod
    def _diag_first(q, coords):
        """Rotate a cyclically wound quad so that node 0 is the corner minimal in both in-plane directions.
        Splitting as (q0,q1,q2)+(q0,q2,q3) then cuts along the min-min -> max-max diagonal, matching Kuhn."""
        pos = [coords[n] for n in q]
        # The face is planar: the constant axis is the one taking a single value across the four corners.
        const_axis = next(d for d in range(3) if len({p[d] for p in pos}) == 1)
        plane = [d for d in range(3) if d != const_axis]
        lo = min(range(4), key=lambda t: (pos[t][plane[0]], pos[t][plane[1]]))
        return q[lo:] + q[:lo]
