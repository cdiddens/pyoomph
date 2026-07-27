#  Uniform (1->8) refinement of wedge (triangular-prism) meshes (branch mixed_adapt).
#
#  A wedge refines shape-closed: its triangular cross-section splits 1->4 (three corner sub-triangles +
#  the inverted middle one, as for a 2d triangle) and its extrusion direction splits 1->2, giving 8 wedge
#  sons. New father-boundary nodes are shared via a father-node-keyed registry (keyed on the POSITIVE-weight
#  father shape nodes, so C2 edge 1/4- and 3/4-points do not collide). Oracle: element count x8 per level,
#  ZERO duplicate (torn) nodes, and -- since the problem is linear -- the residual reaches machine zero in
#  one Newton step iff shape/integration/node-sharing are all correct.

import numpy as np
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.meshes.mesh import MeshTemplate
from pyoomph.generic.codegen import Equations


class WedgeCubeMesh(MeshTemplate):
    # Unit cube [0,1]^3 as N^3 cells, each split into 2 triangular prisms (wedges) along the (x,y) diagonal.
    def __init__(self, N=1):
        super().__init__()
        self.N = N

    def define_geometry(self):
        N = self.N
        dom = self.new_domain("domain")
        nd = {}

        def node(x, y, z):
            k = (x, y, z)
            if k not in nd:
                nd[k] = self.add_node_unique(x / N, y / N, z / N)
            return nd[k]

        for ix in range(N):
            for iy in range(N):
                for iz in range(N):
                    n = lambda a, b, c: node(a, b, c)
                    c = [n(ix, iy, iz), n(ix + 1, iy, iz), n(ix + 1, iy + 1, iz), n(ix, iy + 1, iz)]
                    t = [n(ix, iy, iz + 1), n(ix + 1, iy, iz + 1), n(ix + 1, iy + 1, iz + 1), n(ix, iy + 1, iz + 1)]
                    dom.add_wedge_3d_C1(c[0], c[1], c[2], t[0], t[1], t[2])
                    dom.add_wedge_3d_C1(c[0], c[2], c[3], t[0], t[2], t[3])


class _HelmholtzLike(Equations):
    # grad(u).grad(v) + (u-1)*v : non-singular with natural BCs (no boundary facets needed), linear,
    # unique solution -> a single Newton step must reach machine zero iff the element assembles correctly.
    def __init__(self, order="C1"):
        super().__init__()
        self._order = order

    def define_fields(self):
        self.define_scalar_field("u", self._order)

    def define_residuals(self):
        u = var("u")
        v = testfunction("u")
        self.add_residual(weak(grad(u), grad(v)) + weak(u - 1, v))


def _max_abs_residual(problem):
    r = problem.get_last_residual_convergence()
    return r[-1] if r else 1.0


def _count_coincident_nodes(mesh):
    seen = set()
    dup = 0
    for n in mesh.nodes():
        key = (round(n.x(0), 8), round(n.x(1), 8), round(n.x(2), 8))
        if key in seen:
            dup += 1
        seen.add(key)
    return dup


@pytest.mark.parametrize("order", ["C1", "C2"])
@pytest.mark.parametrize("level", [1, 2])
def test_uniform_wedge_refinement(order, level):
    class _P(Problem):
        def define_problem(self):
            self += WedgeCubeMesh(N=1)  # 2 wedges
            self += _HelmholtzLike(order) @ "domain"
            self += RefineToLevel(level) @ "domain"

    with _P() as p:
        p.max_refinement_level = level
        p.solve()
        m = p.get_mesh("domain")
        assert m.nelement() == 2 * 8 ** level, f"expected {2 * 8 ** level} wedges, got {m.nelement()}"
        assert _count_coincident_nodes(m) == 0, "duplicate (torn) nodes after uniform wedge refinement"
        assert _max_abs_residual(p) < 1e-9
