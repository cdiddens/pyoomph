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
        coord = {}

        def node(x, y, z):
            k = (x, y, z)
            if k not in nd:
                nd[k] = self.add_node_unique(x / N, y / N, z / N)
                coord[nd[k]] = (x, y, z)
            return nd[k]

        wedges = []
        for ix in range(N):
            for iy in range(N):
                for iz in range(N):
                    n = lambda a, b, c: node(a, b, c)
                    c = [n(ix, iy, iz), n(ix + 1, iy, iz), n(ix + 1, iy + 1, iz), n(ix, iy + 1, iz)]
                    t = [n(ix, iy, iz + 1), n(ix + 1, iy, iz + 1), n(ix + 1, iy + 1, iz + 1), n(ix, iy + 1, iz + 1)]
                    w1 = [c[0], c[1], c[2], t[0], t[1], t[2]]
                    w2 = [c[0], c[2], c[3], t[0], t[2], t[3]]
                    dom.add_wedge_3d_C1(*w1)
                    dom.add_wedge_3d_C1(*w2)
                    wedges += [w1, w2]
        # Boundary facets: a wedge face (local nodes) is on a cube-face boundary iff all its nodes lie in
        # that plane. Faces: 2 triangular caps {0,1,2},{3,4,5} + 3 quad sides.
        bounds = {"left": (0, 0), "right": (0, N), "bottom": (1, 0), "top": (1, N), "back": (2, 0), "front": (2, N)}
        facelists = [[0, 1, 2], [3, 4, 5], [0, 1, 4, 3], [0, 3, 5, 2], [1, 2, 5, 4]]
        for w in wedges:
            for fl in facelists:
                fn = [w[i] for i in fl]
                for bname, (ax, val) in bounds.items():
                    if all(coord[x][ax] == val for x in fn):
                        self.add_facet_to_boundary(bname, list(fn))


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


from pyoomph.equations.poisson import PoissonEquation


def test_nonuniform_wedge_2to1_hanging():
    # Non-uniform (2:1) C1 wedge refinement: refine only the cells with Eulerian x<0.5, creating a 2:1
    # hanging interface at x=0.5 (enforce_refinement_balance keeps it single-level). The exact solution of
    # -laplace(u)=0 with Dirichlet u=x+2y+3z is that linear field itself, which the C1 space represents
    # exactly -- so it is reproduced to machine precision at every node IFF the hanging nodes on the 2:1
    # interface are correctly constrained (a free/mis-constrained hanging node would deviate). This is a
    # strict correctness oracle for the hanging (unlike a bare linear-residual check, which is auto-zero).
    x = var("coordinate")[0]
    y = var("coordinate")[1]
    z = var("coordinate")[2]
    uex = x + 2 * y + 3 * z

    class _P(Problem):
        def define_problem(self):
            self += WedgeCubeMesh(N=2)
            eqs = PoissonEquation(source=0, space="C1")
            eqs += DirichletBC(u=uex) @ ["left", "right", "top", "bottom", "front", "back"]
            eqs += RefineAccordingToElement(level_func=lambda e: 1 if e.get_Eulerian_midpoint()[0] < 0.5 else 0)
            self += eqs @ "domain"

    with _P() as p:
        p.max_refinement_level = 2
        p.solve()
        m = p.get_mesh("domain")
        assert m.nelement() > 16, "refinement did not happen"
        assert _count_coincident_nodes(m) == 0, "duplicate (torn) nodes at the 2:1 wedge interface"
        err = 0.0
        for n in m.nodes():
            err = max(err, abs(n.value(0) - (n.x(0) + 2 * n.x(1) + 3 * n.x(2))))
        assert err < 1e-10, f"linear field not reproduced across the 2:1 wedge interface (max err {err:.2e})"


def test_nonuniform_wedge_2to1_hanging_C2():
    # C2 (quadratic) analogue of the 2:1 test. The exact solution of -laplace(u)=-12 with Dirichlet
    # u = x^2+2y^2+3z^2 is that quadratic field, which the C2 wedge space (quadratic-triangle x
    # quadratic-line = 18 nodes) represents exactly on an affine mesh -- so it must be reproduced to machine
    # precision at every node IFF (a) the uniform C2 build shares the cross-section quadratic nodes correctly
    # (the share key must carry weights, not just the positive-node set, or distinct interior tri points
    # collide and the quadratic space collapses) and (b) the C2 quarter-point hanging nodes on the 2:1
    # interface are all constrained. This caught an over-merging share-key bug invisible to the residual and
    # coincident-node checks (over-merged nodes are not torn -- they are simply missing).
    x = var("coordinate")[0]
    y = var("coordinate")[1]
    z = var("coordinate")[2]
    uex = x * x + 2 * y * y + 3 * z * z

    class _P(Problem):
        def define_problem(self):
            self += WedgeCubeMesh(N=2)
            eqs = PoissonEquation(source=-12, space="C2")  # -laplace(u) = -(2+4+6) = -12
            eqs += DirichletBC(u=uex) @ ["left", "right", "top", "bottom", "front", "back"]
            eqs += RefineAccordingToElement(level_func=lambda e: 1 if e.get_Eulerian_midpoint()[0] < 0.5 else 0)
            self += eqs @ "domain"

    with _P() as p:
        p.max_refinement_level = 2
        p.solve()
        m = p.get_mesh("domain")
        assert m.nelement() > 16, "refinement did not happen"
        assert _count_coincident_nodes(m) == 0, "duplicate (torn) nodes at the 2:1 C2 wedge interface"
        err = 0.0
        for n in m.nodes():
            err = max(err, abs(n.value(0) - (n.x(0) ** 2 + 2 * n.x(1) ** 2 + 3 * n.x(2) ** 2)))
        assert err < 1e-10, f"quadratic field not reproduced across the 2:1 C2 wedge interface (max err {err:.2e})"


def test_uniform_wedge_C2_reproduces_quadratic():
    # Uniform (conforming, no hanging) C2 refinement must reproduce a global quadratic exactly. The bare
    # linear-residual oracle above cannot see the cross-section share-key over-merge (it silently drops DOFs
    # without tearing the mesh); this manufactured-quadratic check does.
    x = var("coordinate")[0]
    y = var("coordinate")[1]
    z = var("coordinate")[2]
    uex = x * x + 2 * y * y + 3 * z * z

    class _P(Problem):
        def define_problem(self):
            self += WedgeCubeMesh(N=2)
            eqs = PoissonEquation(source=-12, space="C2")
            eqs += DirichletBC(u=uex) @ ["left", "right", "top", "bottom", "front", "back"]
            eqs += RefineToLevel(1)
            self += eqs @ "domain"

    with _P() as p:
        p.max_refinement_level = 2
        p.solve()
        m = p.get_mesh("domain")
        assert m.nelement() == 2 * 8 * 8, "uniform refinement of 16 base wedges did not produce 128 wedges"
        err = 0.0
        for n in m.nodes():
            err = max(err, abs(n.value(0) - (n.x(0) ** 2 + 2 * n.x(1) ** 2 + 3 * n.x(2) ** 2)))
        assert err < 1e-10, f"uniform C2 wedge mesh does not reproduce a quadratic (max err {err:.2e})"
