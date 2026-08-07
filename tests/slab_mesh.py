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

# SlabTemplate: a one-element-thick slab, N x N x 1 cells over [0,1]^2 x [0,t], whose TOP AND BOTTOM
# faces carry the SAME boundary name.
#
# That is the configuration nodal boundary membership goes wrong on, made big enough to distribute.
# Every element then has two of its faces on "wall", so all eight of its nodes are on "wall", and the
# four vertical edges -- which lie on "side", or in the interior -- have both ends on "wall" without
# lying on it. Refinement gives each of those edges a midpoint, the "boundaries shared by all my
# generating nodes" rule marks it on "wall", and every such node seeds four more at the next level.
#
# The single tetrahedron of tests/test_curved_boundaries.py (_SphericalTetTemplate with 2 or 4 curved
# faces) is the same defect in its smallest form, but one element cannot be distributed, which is why
# this exists alongside it.

from pyoomph.meshes.mesh import MeshTemplate

# Kuhn (Freudenthal) split of a cube into 6 tets sharing the main diagonal, in tensor-corner indices
# (index = 4*bz + 2*by + bx), same convention and same split as tests/box_mesh_3d.py.
_KUHN = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
# The six cube faces as node quadruples, again in tensor-corner indices.
_CUBE_FACES = {"bottom": (0, 1, 3, 2), "top": (4, 5, 7, 6), "left": (0, 2, 6, 4),
               "right": (1, 3, 7, 5), "front": (0, 1, 5, 4), "back": (2, 3, 7, 6)}


class SlabTemplate(MeshTemplate):
    """N x N x 1 slab of bricks or tets; "wall" is top AND bottom, "side" is the four sides."""

    def __init__(self, N=2, family="hex", thickness=0.25, name="domain"):
        super().__init__()
        self.N, self.family, self.t, self.dname = N, family, thickness, name

    def define_geometry(self):
        N, h, t = self.N, 1.0 / self.N, self.t
        dom = self.new_domain(self.dname)
        nodes = {}

        def node(x, y, z):
            key = (round(x, 9), round(y, 9), round(z, 9))
            if key not in nodes:
                nodes[key] = self.add_node_unique(x, y, z)
            return nodes[key]

        for i in range(N):
            for j in range(N):
                x0, y0 = i * h, j * h
                c = [node(x0 + bx * h, y0 + by * h, bz * t)
                     for bz in (0, 1) for by in (0, 1) for bx in (0, 1)]
                if self.family == "hex":
                    dom.add_brick_3d_C1(*c)
                elif self.family == "tet":
                    for q in _KUHN:
                        dom.add_tetra_3d_C1(*[c[k] for k in q])
                else:
                    raise ValueError("unknown family %r" % self.family)

                # Both z-faces are "wall"; the outer x/y faces are "side". A tet cell presents two
                # triangles per cube face rather than a quad, so the facets are declared per shape.
                for fname, quad in _CUBE_FACES.items():
                    if fname in ("bottom", "top"):
                        bname = "wall"
                    elif fname == "left" and i == 0:
                        bname = "side"
                    elif fname == "right" and i == N - 1:
                        bname = "side"
                    elif fname == "front" and j == 0:
                        bname = "side"
                    elif fname == "back" and j == N - 1:
                        bname = "side"
                    else:
                        continue  # interior face
                    q = [c[k] for k in quad]
                    if self.family == "hex":
                        self.add_facet_to_boundary(bname, q)
                    else:
                        # The Kuhn split cuts every cube face along the diagonal between the corner
                        # with both in-plane coordinates minimal and the one where both are maximal,
                        # i.e. between quad entries 0 and 2 in the cyclic order used above.
                        self.add_facet_to_boundary(bname, [q[0], q[1], q[2]])
                        self.add_facet_to_boundary(bname, [q[0], q[2], q[3]])
