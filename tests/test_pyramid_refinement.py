#  Refinement of pyramid meshes (branch mixed_adapt).
#
#  A pyramid is NOT shape-closed under refinement: its "red" split produces MIXED offspring -- 6 sub-pyramids
#  + 4 tetrahedra (10 children). This exercises the heterogeneous-son path (PyramidMixedRefinementPattern +
#  the cross-shape son builder BulkElementBase::build_as_pyramid_son). New father-boundary nodes are shared
#  topologically via a father-node-keyed registry (the same one the tet build uses in a pyramid forest), so a
#  sub-pyramid and an adjacent tet-of-pyramid sharing a face key on the same shared father vertex pointers and
#  reuse one node -- no tearing, MPI-safe. Oracles: exact mixed element counts, ZERO duplicate (torn) nodes at
#  the pyramid<->tet interfaces, and a manufactured linear solution reproduced to machine precision with
#  Dirichlet BCs (which needs face_index_in_father for boundary-tag propagation AND the conical Gauss-Jacobi
#  pyramid quadrature for a patch-exact stiffness at the pyramid<->tet interfaces).

import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.meshes.mesh import MeshTemplate
from pyoomph.generic.codegen import Equations


class PyramidCubeMesh(MeshTemplate):
    # Unit cube as N^3 cells; each cube is split into 6 pyramids (one per cube face, common apex at the cube
    # centre). The 6-pyramid decomposition of a hex is the natural all-pyramid test mesh.
    def __init__(self, N=1):
        super().__init__()
        self.N = N

    def define_geometry(self):
        N = self.N
        dom = self.new_domain("domain")
        nd = {}
        coord = {}

        def node(x, y, z):
            k = (round(x, 9), round(y, 9), round(z, 9))
            if k not in nd:
                nd[k] = self.add_node_unique(x, y, z)
                coord[nd[k]] = (x, y, z)
            return nd[k]

        bases = []  # each pyramid's quad base (its 4 nodes), for boundary-facet detection
        for ix in range(N):
            for iy in range(N):
                for iz in range(N):
                    x0, y0, z0 = ix / N, iy / N, iz / N
                    h = 1.0 / N
                    c = [node(x0, y0, z0), node(x0 + h, y0, z0), node(x0 + h, y0 + h, z0), node(x0, y0 + h, z0),
                         node(x0, y0, z0 + h), node(x0 + h, y0, z0 + h), node(x0 + h, y0 + h, z0 + h), node(x0, y0 + h, z0 + h)]
                    ctr = node(x0 + h / 2, y0 + h / 2, z0 + h / 2)
                    # 6 cube faces as quad bases (wound so the apex=centre gives a positive-volume pyramid).
                    faces = [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]]
                    for f in faces:
                        q = [c[f[0]], c[f[1]], c[f[2]], c[f[3]]]
                        dom.add_pyramid_3d_C1(q[0], q[1], q[2], q[3], ctr)
                        bases.append(q)
        # A pyramid base (a cube face) is a boundary facet iff all four of its nodes lie in one cube-face plane.
        # This exercises face_index_in_father: the boundary tag must propagate to the refined sub-faces so
        # DirichletBC pins the refined boundary nodes.
        bounds = {"left": (0, 0.0), "right": (0, 1.0), "front": (1, 0.0), "back": (1, 1.0), "bottom": (2, 0.0), "top": (2, 1.0)}
        for q in bases:
            for bname, (ax, val) in bounds.items():
                if all(abs(coord[n][ax] - val) < 1e-9 for n in q):
                    self.add_facet_to_boundary(bname, list(q))


class _HelmholtzLike(Equations):
    # grad(u).grad(v) + (u-1)*v : non-singular with natural BCs (no boundary facets needed), linear, unique
    # solution -> a single Newton step must reach machine zero iff every element assembles correctly.
    def __init__(self, order="C1"):
        super().__init__()
        self._order = order

    def define_fields(self):
        self.define_scalar_field("u", self._order)

    def define_residuals(self):
        u = var("u")
        v = testfunction("u")
        self.add_residual(weak(grad(u), grad(v)) + weak(u - 1, v))


def _count_coincident_nodes(mesh):
    seen = set()
    dup = 0
    for n in mesh.nodes():
        key = (round(n.x(0), 8), round(n.x(1), 8), round(n.x(2), 8))
        if key in seen:
            dup += 1
        seen.add(key)
    return dup


# Exact mixed element counts. Level 0: 6 pyramids. A pyramid -> 6 pyr + 4 tet; a tet -> 8 tet. So with
# p pyramids and t tets, one refinement gives (6 p) pyramids and (4 p + 8 t) tets.
#   level 0: (p,t) = (6, 0)   -> 6
#   level 1: (36, 24)         -> 60
#   level 2: (216, 336)       -> 552
_EXPECTED_NELEM = {1: 60, 2: 552}


@pytest.mark.parametrize("level", [1, 2])
def test_uniform_pyramid_refinement(level):
    class _P(Problem):
        def define_problem(self):
            self += PyramidCubeMesh(N=1)  # 6 pyramids
            self += _HelmholtzLike("C1") @ "domain"

    with _P() as p:
        p.max_refinement_level = level
        p.initialise()
        for _ in range(level):
            p.refine_uniformly()
        p.solve()
        m = p.get_mesh("domain")
        assert m.nelement() == _EXPECTED_NELEM[level], f"expected {_EXPECTED_NELEM[level]} elements, got {m.nelement()}"
        # Zero torn nodes -- in particular none at the pyramid<->tet interfaces that appear from level 2 on.
        assert _count_coincident_nodes(m) == 0, "duplicate (torn) nodes after uniform pyramid refinement"
        r = p.get_last_residual_convergence()
        assert r and r[-1] < 1e-9, f"residual not machine-zero ({r[-1] if r else None})"


def test_pyramid_red_split_tiles_exactly():
    # The 6-pyramid + 4-tet red split must TILE the parent exactly: the children's volumes sum to the
    # parent's, with no gaps/overlaps. Checked on the unit cube (total volume 1).
    import numpy as np

    class _P(Problem):
        def define_problem(self):
            self += PyramidCubeMesh(N=1)
            self += _HelmholtzLike("C1") @ "domain"

    with _P() as p:
        p.max_refinement_level = 1
        p.initialise()
        p.refine_uniformly()
        m = p.get_mesh("domain")
        tot = 0.0
        for ie in range(m.nelement()):
            el = m.element_pt(ie)
            pts = np.array([[el.node_pt(i).x(0), el.node_pt(i).x(1), el.node_pt(i).x(2)] for i in range(el.nnode())])
            if el.nnode() == 5:  # pyramid: split base into 2 triangles
                base, apex = pts[:4], pts[4]
                for tri in [(0, 1, 2), (0, 2, 3)]:
                    a, b, c = base[tri[0]], base[tri[1]], base[tri[2]]
                    tot += abs(np.dot(np.cross(b - a, c - a), apex - a)) / 6
            else:  # tet
                a, b, c, d = pts
                tot += abs(np.dot(np.cross(b - a, c - a), d - a)) / 6
        assert abs(tot - 1.0) < 1e-10, f"pyramid red split does not tile exactly (total volume {tot})"


from pyoomph.equations.poisson import PoissonEquation


@pytest.mark.parametrize("level", [0, 1, 2])
def test_pyramid_dirichlet_manufactured_linear(level):
    # Strict correctness oracle for pyramid refinement WITH boundaries. The exact solution of -laplace(u)=0
    # with Dirichlet u=x+2y+3z is that harmonic linear field, exactly representable in the (isoparametric) C1
    # space -- so it must be reproduced to machine precision at every node IFF (a) face_index_in_father
    # propagates the boundary tags to the refined sub-faces (so DirichletBC pins the right nodes) and (b) the
    # pyramid element integrates its stiffness patch-exactly. (b) needs the CONICAL Gauss-Jacobi quadrature:
    # plain tensor Gauss samples the rational shape functions OUTSIDE the shrinking [0,1-s2] cross-section and
    # is not patch-exact, which -- while it cancels in a pure-pyramid mesh -- fails at the pyramid<->tet
    # interfaces the red split creates, giving an O(1e-2) error that this test would catch.
    x, y, z = var("coordinate")[0], var("coordinate")[1], var("coordinate")[2]
    uex = x + 2 * y + 3 * z

    class _P(Problem):
        def define_problem(self):
            self += PyramidCubeMesh(N=1)
            eqs = PoissonEquation(source=0, space="C1")
            eqs += DirichletBC(u=uex) @ ["left", "right", "front", "back", "bottom", "top"]
            self += eqs @ "domain"

    with _P() as p:
        p.max_refinement_level = level
        p.initialise()
        for _ in range(level):
            p.refine_uniformly()
        p.solve()
        m = p.get_mesh("domain")
        err = 0.0
        for n in m.nodes():
            err = max(err, abs(n.value(0) - (n.x(0) + 2 * n.x(1) + 3 * n.x(2))))
        assert err < 1e-10, f"pyramid mesh does not reproduce the linear field at level {level} (max err {err:.2e})"
