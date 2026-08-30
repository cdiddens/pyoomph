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

#  Mixed quad+tri adaptive meshes (branch mixed_adapt).
#
#  A single 2D domain containing BOTH quadrilateral and triangular elements, refined via the shared
#  QuadTree forest. The quad<->tri INTERFACE is handled purely topologically (no geometry/locate_zeta):
#    - node-sharing: a quad and a tri that both refine to the same level share the coincident interface
#      node (BulkElementBase::mixed_quad_shared_node + RefineableTElement<2>::node_at_root_coordinate),
#      mirroring how oomph shares pure-quad nodes (get_node_at_local_coordinate on the neighbour leaf);
#    - hanging: a fine element hangs its interface nodes on the coarser cross-shape neighbour LEAF
#      (mixed_hang_edge_node: shared-root-edge corner-node coordinate blend + interpolating_basis),
#      via the overridden quad_hang_helper (quad side) and TriEdgeNeighbour::cross_shape_root (tri side);
#    - 2:1 balancing across the interface keeps every quad<->tri jump single-level.
#  Poisson is linear, so max|residual| ~ 0 after one Newton step certifies the interface Jacobian.

import numpy as np
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.meshes.gmsh import GmshTemplate
from pyoomph.meshes.mesh import MeshTemplate
from pyoomph.equations.poisson import PoissonEquation

_BND = ["left", "right", "top", "bottom"]


class MixedRectMesh(MeshTemplate):
    # [0,2]x[0,1] as N x N cells: left half (x<1) quads, right half (x>=1) each split into 2 tris.
    # The shared vertical interface at x=1 is the quad<->tri boundary.
    def __init__(self, N=4):
        super().__init__()
        self.N = N

    def define_geometry(self):
        N = self.N
        dom = self.new_domain("domain")
        idx = {}

        def node(i, j):
            if (i, j) not in idx:
                idx[(i, j)] = self.add_node_unique(2.0 * i / N, 1.0 * j / N)
            return idx[(i, j)]

        for i in range(N):
            for j in range(N):
                a, b, c, d = node(i, j), node(i + 1, j), node(i, j + 1), node(i + 1, j + 1)
                if i < N // 2:
                    dom.add_quad_2d_C1(a, b, c, d)
                else:
                    dom.add_tri_2d_C1(a, b, d)
                    dom.add_tri_2d_C1(a, d, c)
        for j in range(N):
            self.add_facet_to_boundary("left", [node(0, j), node(0, j + 1)])
            self.add_facet_to_boundary("right", [node(N, j), node(N, j + 1)])
        for i in range(N):
            self.add_facet_to_boundary("bottom", [node(i, 0), node(i + 1, 0)])
            self.add_facet_to_boundary("top", [node(i, N), node(i + 1, N)])


class _MixedPoisson(Problem):
    def __init__(self, N=4, source=1, use_ee=False):
        super().__init__()
        self._N = N
        self._source = source
        self._use_ee = use_ee

    def define_problem(self):
        self += MixedRectMesh(N=self._N)
        eqs = PoissonEquation(source=self._source) + DirichletBC(u=0) @ _BND
        if self._use_ee:
            eqs += SpatialErrorEstimator(u=1)
        self += eqs @ "domain"


def _max_abs_residual(problem):
    return float(np.max(np.abs(np.asarray(problem.get_residuals()))))


def _interface_health(m):
    # Returns (n_duplicate_coincident, n_torn) for nodes on the interface x=1. A "torn" node is one that
    # belongs to only one shape's elements and is not hanging (an unshared, unconstrained interface node).
    from collections import defaultdict
    shapes = defaultdict(set)
    handle = {}
    for ie in range(m.nelement()):
        el = m.element_pt(ie)
        nn = el.nnode()
        for k in range(nn):
            nd = el.node_pt(k)
            if abs(nd.x(0) - 1.0) < 1e-9:
                shapes[id(nd)].add("q" if nn == 9 else "t")
                handle[id(nd)] = nd
    # coincident duplicates: two DISTINCT node objects at the same interface position
    seen = {}
    dup = 0
    for nid, nd in handle.items():
        key = round(nd.x(1), 9)
        if key in seen and seen[key] != nid:
            dup += 1
        seen.setdefault(key, nid)
    torn = sum(1 for nid, sh in shapes.items() if len(sh) == 1 and not handle[nid].is_hanging())
    return dup, torn


def test_mixed_mesh_non_adaptive():
    # The mixed mesh builds and solves; linear residual -> machine zero.
    with _MixedPoisson(N=4) as problem:
        problem.solve()
        assert _max_abs_residual(problem) < 1e-9


def test_mixed_uniform_refine_node_sharing():
    # Uniform refinement: both sides refine to the same level; the interface nodes must be SHARED (no
    # coincident duplicates), certifying topological cross-shape node-sharing.
    with _MixedPoisson(N=4) as problem:
        problem.max_refinement_level = 2
        problem.initialise()
        m = problem.get_mesh("domain")
        m.refine_uniformly()
        problem.solve()
        assert _max_abs_residual(problem) < 1e-9
        dup, torn = _interface_health(m)
        assert dup == 0
        assert torn == 0


@pytest.mark.parametrize("refine", ["quad", "tri"])
def test_mixed_single_level_interface_hanging(refine):
    # Force a single-level 2:1 quad<->tri interface by refining only the elements adjacent to it on ONE
    # side. The fine side must hang on the coarse side (no torn nodes) and the linear residual -> 0.
    with _MixedPoisson(N=4) as problem:
        problem.max_refinement_level = 3
        problem.initialise()
        m = problem.get_mesh("domain")
        want_nnode = 9 if refine == "quad" else 6
        lo, hi = (0.5, 1.0) if refine == "quad" else (1.0, 1.5)
        sel = []
        for ie in range(m.nelement()):
            el = m.element_pt(ie)
            if el.nnode() != want_nnode:
                continue
            xc = sum(el.node_pt(k).x(0) for k in range(el.nnode())) / el.nnode()
            if lo < xc < hi:
                sel.append(ie)
        m.refine_base_mesh([sel])
        problem.solve()
        assert _max_abs_residual(problem) < 1e-9
        dup, torn = _interface_health(m)
        assert dup == 0
        assert torn == 0


def test_mixed_error_based_adaptivity_correct():
    # Genuine Z2 error-driven adaptivity with a localized source straddling the interface (multi-cycle).
    # Machine-zero residual + no torn interface nodes, AND the solution matches a uniform-fine reference
    # (certifying that the cross-shape node-sharing/hanging give the correct -- not merely tear-free -- field).
    x, y = var("coordinate")[0], var("coordinate")[1]
    # A moderately localized source straddling the interface -- forces asymmetric refinement across it.
    src = 30 * exp(-20 * ((x - 1.0) ** 2 + (y - 0.5) ** 2))

    def probe(m, px, py):
        best, bd = None, 1e9
        for nd in m.nodes():
            d = abs(nd.x(0) - px) + abs(nd.x(1) - py)
            if d < bd:
                bd, best = d, nd.value(0)
        return best

    with _MixedPoisson(N=8, source=src) as ref:  # well-resolved uniform reference
        ref.initialise()
        mref = ref.get_mesh("domain")
        mref.refine_uniformly()
        mref.refine_uniformly()
        ref.solve()
        u_ref = probe(mref, 0.5, 0.5)  # quad region, smooth -> both meshes resolve it

    with _MixedPoisson(N=8, source=src, use_ee=True) as problem:
        problem.max_refinement_level = 4
        problem.solve(spatial_adapt=3)
        m = problem.get_mesh("domain")
        assert _max_abs_residual(problem) < 1e-9
        dup, torn = _interface_health(m)
        assert dup == 0
        assert torn == 0
        # No interface tear -> the adaptive solution agrees with the reference in the smooth quad region.
        assert abs(probe(m, 0.5, 0.5) - u_ref) < 5e-3


# ---------------------------------------------------------------------------------------------
# The gmsh boundary-layer case: a quad band along a curved line, triangles everywhere else.
#
# The hand-built MixedRectMesh above never caught this, because its quad and triangle roots meet in
# only one relative orientation. A gmsh mesh produces all of them, and one of them is enough:
# oomph's RefineableQElement<2>::node_created_by_neighbour maps a son node's fractional position
# into the edge neighbour with the QUAD BOX map (s_lo/s_hi/translate_s) and then asks that
# neighbour get_node_at_local_coordinate. Across a quad<->tri interface the neighbour is a triangle,
# for which that coordinate means nothing -- but a triangle's node coordinates live in [0,1]^2 and a
# quad's in [-1,1]^2, so the lookup can MATCH a triangle node by accident and hand back a node from
# a completely different edge. The quad son adopts it, and the adopted node -- a real node of the
# coarse triangle -- is then dragged onto the quad's edge by the hanging-node position
# interpolation, folding every triangle that owns it. It is fixed in the vendored
# refineable_quad_element.cc by skipping a non-quad neighbour there (see INFO_oomph-lib); the
# cross-shape case is handled topologically by BulkElementBase::mixed_quad_shared_node.
#
# The oracle is deliberately about the MESH, not the solution: refinement must not move a node that
# already existed, must not leave two nodes at the same place, and must not bend a mid-side node
# off its own edge. Before the fix this mesh gave a 0.13 node displacement, 5-20 coincident pairs
# and 49-180 folded elements, and uniform refinement took the process down outright.

class _CurvedBLMesh(GmshTemplate):
    # Triangles with a quad boundary layer (Quads=1) along a wavy spline -- the mesh_mode="tris" +
    # BoundaryLayer combination users get from gmsh, and the smallest one that reproduces.
    def define_geometry(self):
        self.default_resolution = 0.2
        self.mesh_mode = "tris"
        n = 40
        pts = [self.point(k / n, 0.6 + 0.25 * np.sin(2 * np.pi * k / n) * np.exp(-2.0 * k / n), size=-2)
               for k in range(n + 1)]
        curve = self.spline(pts, name="top")
        p00, p10 = self.point(0, 0), self.point(1, 0)
        self.create_lines(pts[-1], "right", p10, "bottom", p00, "left", pts[0])
        self.plane_surface("top", "right", "bottom", "left", name="domain")
        self.set_mesh_size_boundary_layer_field(
            self.add_mesh_size_field("BoundaryLayer", CurvesList=[curve], Quads=1, Thickness=0.1))
        self.set_gmsh_parameter("Mesh.MeshSizeFromCurvature", 2)


class _CurvedBLPoisson(Problem):
    def define_problem(self):
        self += _CurvedBLMesh()
        x, y = var("coordinate_x"), var("coordinate_y")
        eqs = PoissonEquation(source=40 * exp(-30 * ((x - 0.35) ** 2 + (y - 0.45) ** 2)))
        eqs += DirichletBC(u=0) @ ["top", "right", "bottom", "left"]
        eqs += SpatialErrorEstimator(u=1)
        self += eqs @ "domain"


def _mesh_damage(m, before):
    # (largest displacement of a node that already existed, coincident node pairs, folded elements).
    nodes = [m.node_pt(i) for i in range(m.nnode())]  # keep the wrappers alive: id() is the identity
    moved = max([np.hypot(n.x(0) - before[id(n)][0], n.x(1) - before[id(n)][1])
                 for n in nodes if id(n) in before] + [0.0])
    seen = {}
    dup = 0
    for n in nodes:
        key = (round(n.x(0), 10), round(n.x(1), 10))
        dup += key in seen
        seen[key] = True
    folded = 0
    for ie in range(m.nelement()):
        el = m.element_pt(ie)
        nn = el.nnode()
        p = [(el.node_pt(k).x(0), el.node_pt(k).x(1)) for k in range(nn)]
        # (corner, corner, mid-side) triples; quad9 nodes are in lexicographic order
        edges = ((0, 1, 3), (1, 2, 4), (2, 0, 5)) if nn == 6 else \
                (((0, 2, 1), (2, 8, 5), (8, 6, 7), (6, 0, 3)) if nn == 9 else ())
        for a, b, mid in edges:
            L = np.hypot(p[b][0] - p[a][0], p[b][1] - p[a][1])
            off = np.hypot(p[mid][0] - 0.5 * (p[a][0] + p[b][0]), p[mid][1] - 0.5 * (p[a][1] + p[b][1]))
            # 0.3 of the chord is far beyond any curvature this geometry produces; a mid-side node
            # that far off its own edge means the element is folded.
            folded += L > 0 and off / L > 0.3
    return moved, dup, folded


@pytest.mark.parametrize("how", ["uniform", "error_driven"])
def test_mixed_gmsh_boundary_layer_refinement_keeps_mesh_intact(how):
    with _CurvedBLPoisson() as problem:
        problem.max_refinement_level = 2
        problem.initial_adaption_steps = 0
        problem.initialise()
        m = problem.get_mesh("domain")
        keep = [m.node_pt(i) for i in range(m.nnode())]
        before = {id(n): (n.x(0), n.x(1)) for n in keep}
        assert any(m.element_pt(ie).nnode() == 9 for ie in range(m.nelement())), \
            "no quads in the boundary layer -- the mixed interface is not being exercised"
        if how == "uniform":
            m.refine_uniformly()
            problem.solve()
        else:
            problem.solve(spatial_adapt=2)
        moved, dup, folded = _mesh_damage(problem.get_mesh("domain"), before)
        assert moved < 1e-12   # was 0.13
        assert dup == 0        # was 5-20
        assert folded == 0     # was 49-180
        assert _max_abs_residual(problem) < 1e-9
