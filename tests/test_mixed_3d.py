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

# Branch mixed_adapt: genuinely MIXED 3d meshes (>=2 of {brick, tet, wedge, pyramid} in one domain), refined
# via the shared OcTree forest. All four families key their new interface nodes into one weight-augmented
# shared-node registry (RefineablePyramidElement::Mixed_forest_active), so elements of different shapes sharing
# a face create one shared node instead of a torn pair. A tet meets a wedge/pyramid only on a TRIANGLE (a tet
# has no quad face); a wedge meets a pyramid on a triangle (cap<->tri face) or a QUAD (side<->base); a brick
# (only quad faces) meets a pyramid base or a wedge side on a QUAD. Both sides' shapes restrict to the standard
# triangle/quad trace on the shared face, so they produce matching keys. In a mixed mesh a brick refines
# through the registry (build_as_brick_son) instead of oomph-lib's native octree build. Non-uniform 2:1
# hanging among the registry families is installed by the unified post_adapt_setup_hanging_nodes pass.
#
# Coverage: for tet+wedge, wedge+pyramid, the three-way mix, and hex+pyramid / hex+wedge -- UNIFORM refinement,
# NON-UNIFORM 2:1 cross-shape hanging (including across a hex face, both refine directions), and MULTI-LEVEL
# (>1) non-uniform refinement (balancing + the cross-round position snapshot), C1 and C2, with strict
# manufactured (linear + quadratic) oracles at every node.

import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.meshes.mesh import MeshTemplate
from pyoomph.equations.poisson import PoissonEquation


class TetWedgeMesh(MeshTemplate):
    # A wedge (triangular prism: tri {(0,0,0),(1,0,0),(0,1,0)} extruded to z=1) capped by a tet on its top
    # triangular face (apex (0,0,2)). The wedge's top cap == the tet's base -> the single mixed tri interface.
    def define_geometry(self):
        dom = self.new_domain("domain")
        nd, co = {}, {}

        def n(a, b, c):
            k = (round(a, 9), round(b, 9), round(c, 9))
            if k not in nd:
                nd[k] = self.add_node_unique(a, b, c)
                co[nd[k]] = (a, b, c)
            return nd[k]

        wb = [n(0, 0, 0), n(1, 0, 0), n(0, 1, 0)]
        wt = [n(0, 0, 1), n(1, 0, 1), n(0, 1, 1)]
        dom.add_wedge_3d_C1(wb[0], wb[1], wb[2], wt[0], wt[1], wt[2])
        apex = n(0, 0, 2)
        dom.add_tetra_3d_C1(wt[0], wt[1], wt[2], apex)
        shared = frozenset(wt)  # the interior mixed interface (not a boundary)
        wfaces = [[wb[0], wb[1], wb[2]], [wt[0], wt[1], wt[2]],
                  [wb[0], wb[1], wt[1], wt[0]], [wb[0], wt[0], wt[2], wb[2]], [wb[1], wb[2], wt[2], wt[1]]]
        tfaces = [[wt[1], wt[2], apex], [wt[0], wt[2], apex], [wt[0], wt[1], apex], [wt[0], wt[1], wt[2]]]
        for f in wfaces + tfaces:
            if frozenset(f) != shared:
                self.add_facet_to_boundary("outer", list(f))


class WedgePyrMesh(MeshTemplate):
    # A wedge (tri {(0,0,0),(1,0,0),(0,1,0)} extruded to z=1) + a pyramid whose base is the wedge's y=0 quad
    # side {(0,0,0),(1,0,0),(1,0,1),(0,0,1)} (apex (0.5,-1,0.5)). This exercises the wedge<->pyramid QUAD
    # interface (a tet has no quad face, so tet+wedge cannot). With three_way=True a tet also caps the wedge's
    # top triangle (apex (0,0,2)) -> all three families and both a quad and a triangular mixed interface.
    def __init__(self, three_way=False):
        super().__init__()
        self.three_way = three_way

    def define_geometry(self):
        dom = self.new_domain("domain")
        nd = {}

        def n(a, b, c):
            k = (round(a, 9), round(b, 9), round(c, 9))
            if k not in nd:
                nd[k] = self.add_node_unique(a, b, c)
            return nd[k]

        wb = [n(0, 0, 0), n(1, 0, 0), n(0, 1, 0)]
        wt = [n(0, 0, 1), n(1, 0, 1), n(0, 1, 1)]
        dom.add_wedge_3d_C1(wb[0], wb[1], wb[2], wt[0], wt[1], wt[2])
        pa = n(0.5, -1, 0.5)
        dom.add_pyramid_3d_C1(wb[0], wb[1], wt[1], wt[0], pa)  # base == wedge y=0 quad side
        interior = [frozenset([wb[0], wb[1], wt[1], wt[0]])]  # the wedge<->pyramid quad interface
        faces = [[wb[0], wb[1], wb[2]], [wt[0], wt[1], wt[2]],  # wedge tri caps
                 [wb[0], wb[1], wt[1], wt[0]], [wb[0], wt[0], wt[2], wb[2]], [wb[1], wb[2], wt[2], wt[1]],  # wedge quads
                 [wb[0], wb[1], pa], [wb[1], wt[1], pa], [wt[1], wt[0], pa], [wt[0], wb[0], pa]]  # pyramid tris
        if self.three_way:
            ta = n(0, 0, 2)
            dom.add_tetra_3d_C1(wt[0], wt[1], wt[2], ta)
            interior.append(frozenset([wt[0], wt[1], wt[2]]))  # the wedge<->tet tri interface
            faces += [[wt[1], wt[2], ta], [wt[0], wt[2], ta], [wt[0], wt[1], ta]]
        for f in faces:
            if frozenset(f) not in interior:
                self.add_facet_to_boundary("outer", list(f))


class HexMixMesh(MeshTemplate):
    # A brick/hex on [0,1]^3 sharing its z=1 QUAD face with a pyramid (base == that face, apex (0.5,0.5,2)) or
    # a wedge (its s1=0 quad side == that face). A hex has only quad faces, so it can only meet a pyramid base
    # or a wedge side (never a tet). In a mixed mesh the brick refines through the shared registry
    # (build_as_brick_son), so hex and pyramid/wedge sub-nodes on the shared quad face coincide, not tear.
    def __init__(self, other="pyr"):
        super().__init__()
        self.other = other

    def define_geometry(self):
        dom = self.new_domain("domain")
        nd = {}

        def n(a, b, c):
            k = (round(a, 9), round(b, 9), round(c, 9))
            if k not in nd:
                nd[k] = self.add_node_unique(a, b, c)
            return nd[k]

        h = [n(0, 0, 0), n(1, 0, 0), n(0, 1, 0), n(1, 1, 0), n(0, 0, 1), n(1, 0, 1), n(0, 1, 1), n(1, 1, 1)]
        dom.add_brick_3d_C1(*h)  # oomph brick order: 000,100,010,110,001,101,011,111
        top = [h[4], h[5], h[7], h[6]]  # the z=1 quad face
        interior = [frozenset(top)]
        faces = [[h[0], h[1], h[3], h[2]], [h[4], h[5], h[7], h[6]], [h[0], h[1], h[5], h[4]],
                 [h[2], h[3], h[7], h[6]], [h[0], h[2], h[6], h[4]], [h[1], h[3], h[7], h[5]]]  # hex faces
        if self.other == "pyr":
            pa = n(0.5, 0.5, 2)
            dom.add_pyramid_3d_C1(top[0], top[1], top[2], top[3], pa)  # base == hex top face
            faces += [[top[0], top[1], pa], [top[1], top[2], pa], [top[2], top[3], pa], [top[3], top[0], pa]]
        else:  # wedge whose quad side {h4,h5,h7,h6} == hex top face; ridge nodes above
            b2, t2 = n(0.5, 0.5, 2), n(0.5, 1.5, 2)
            dom.add_wedge_3d_C1(h[4], h[5], b2, h[6], h[7], t2)
            faces += [[h[4], h[5], b2], [h[6], h[7], t2], [h[4], b2, t2, h[6]], [h[5], b2, t2, h[7]]]
        for f in faces:
            if frozenset(f) not in interior:
                self.add_facet_to_boundary("outer", list(f))


def _dup(m):
    seen = set()
    d = 0
    for nn in m.nodes():
        k = (round(nn.x(0), 7), round(nn.x(1), 7), round(nn.x(2), 7))
        if k in seen:
            d += 1
        seen.add(k)
    return d


# (order, field): a linear field is exact for both spaces; the quadratic is the strict oracle for C2 (it is
# NOT reproduced by a torn/mis-shared mesh, unlike a linear field or a residual-only check).
@pytest.mark.parametrize("order,field", [("C1", "linear"), ("C2", "linear"), ("C2", "quadratic")])
@pytest.mark.parametrize("level", [1, 2])
def test_mixed_tet_wedge_uniform_manufactured(order, field, level):
    # Strict correctness oracle for UNIFORM mixed tet+wedge refinement: the manufactured harmonic-adjacent
    # field (linear, or the quadratic u=x^2+2y^2+3z^2 with source -12) must be reproduced at every node. This
    # fails unless the tet and wedge SHARE the nodes on their common triangular face (the cross-shape
    # weight-augmented registry) -- a torn interface leaves the finer nodes free and the field deviates.
    x, y, z = var("coordinate")[0], var("coordinate")[1], var("coordinate")[2]
    uex = (x + 2 * y + 3 * z) if field == "linear" else (x * x + 2 * y * y + 3 * z * z)
    src = 0 if field == "linear" else -12

    class _P(Problem):
        def define_problem(self):
            self += TetWedgeMesh()
            eqs = PoissonEquation(source=src, space=order) + DirichletBC(u=uex) @ "outer"
            self += eqs @ "domain"

    with _P() as p:
        p.max_refinement_level = level
        p.initialise()
        for _ in range(level):
            p.refine_uniformly()
        p.solve()
        m = p.get_mesh("domain")
        assert _dup(m) == 0, "duplicate (torn) nodes at the mixed tet<->wedge interface"
        if field == "linear":
            err = max(abs(nn.value(0) - (nn.x(0) + 2 * nn.x(1) + 3 * nn.x(2))) for nn in m.nodes())
        else:
            err = max(abs(nn.value(0) - (nn.x(0) ** 2 + 2 * nn.x(1) ** 2 + 3 * nn.x(2) ** 2)) for nn in m.nodes())
        assert err < 1e-10, f"mixed tet+wedge ({order},{field}) not reproduced at level {level} (max err {err:.2e})"


# (refine, field): refine only the wedge (z near 0.5) or only the tet (z near 1.3) -> a 2:1 hang exactly on
# the shared triangular interface, tested for both the linear and the stricter quadratic C2 field.
@pytest.mark.parametrize("refine,zc", [("wedge", 0.5), ("tet", 1.3)])
@pytest.mark.parametrize("field", ["linear", "quadratic"])
def test_mixed_tet_wedge_cross_shape_hanging(refine, zc, field):
    # Non-uniform (2:1) CROSS-SHAPE hanging across the mixed tet<->wedge interface. Refining only one side
    # makes the finer side's extra face nodes hang on the coarser cross-shape neighbour's C2 interpolation
    # (the unified post_adapt_setup_hanging_nodes pass). A free/mis-constrained hang node would deviate; the
    # manufactured field (linear, or the strict quadratic u=x^2+2y^2+3z^2, source -12) must be reproduced to
    # machine precision at every node. Which side is refined is chosen by a midpoint-z band.
    x, y, z = var("coordinate")[0], var("coordinate")[1], var("coordinate")[2]
    uex = (x + 2 * y + 3 * z) if field == "linear" else (x * x + 2 * y * y + 3 * z * z)
    src = 0 if field == "linear" else -12

    class _P(Problem):
        def define_problem(self):
            self += TetWedgeMesh()
            eqs = PoissonEquation(source=src, space="C2") + DirichletBC(u=uex) @ "outer"
            eqs += RefineAccordingToElement(level_func=lambda e: 1 if abs(e.get_Eulerian_midpoint()[2] - zc) < 0.4 else 0)
            self += eqs @ "domain"

    with _P() as p:
        p.max_refinement_level = 1
        p.solve()
        m = p.get_mesh("domain")
        assert sum(1 for nn in m.nodes() if nn.is_hanging()) > 0, "no hanging nodes on the mixed 2:1 interface"
        assert _dup(m) == 0, "duplicate (torn) nodes at the mixed tet<->wedge interface"
        if field == "linear":
            err = max(abs(nn.value(0) - (nn.x(0) + 2 * nn.x(1) + 3 * nn.x(2))) for nn in m.nodes())
        else:
            err = max(abs(nn.value(0) - (nn.x(0) ** 2 + 2 * nn.x(1) ** 2 + 3 * nn.x(2) ** 2)) for nn in m.nodes())
        assert err < 1e-10, f"mixed tet+wedge 2:1 hanging (refine {refine}, {field}) not reproduced (max err {err:.2e})"


def _err(m, field):
    if field == "linear":
        return max(abs(nn.value(0) - (nn.x(0) + 2 * nn.x(1) + 3 * nn.x(2))) for nn in m.nodes())
    return max(abs(nn.value(0) - (nn.x(0) ** 2 + 2 * nn.x(1) ** 2 + 3 * nn.x(2) ** 2)) for nn in m.nodes())


# The wedge<->pyramid QUAD interface and the full three-way mix, to confirm the shared-registry + unified
# hanging pass are general (not tet+wedge-specific). (three_way, order, field): a linear field is exact for
# both spaces; the quadratic is the strict C2 oracle.
@pytest.mark.parametrize("three_way", [False, True])
@pytest.mark.parametrize("order,field", [("C1", "linear"), ("C2", "linear"), ("C2", "quadratic")])
@pytest.mark.parametrize("level", [1, 2])
def test_mixed_wedge_pyramid_uniform_manufactured(three_way, order, field, level):
    # UNIFORM refinement of a wedge+pyramid mesh (and, with a tet, the three-way mesh). The wedge and pyramid
    # meet on a QUAD face (wedge side == pyramid base); both families' shapes restrict to the standard quad
    # trace there, so they share those nodes. The manufactured field must be reproduced to machine precision.
    src = 0 if field == "linear" else -12
    x, y, z = var("coordinate")[0], var("coordinate")[1], var("coordinate")[2]
    uex = (x + 2 * y + 3 * z) if field == "linear" else (x * x + 2 * y * y + 3 * z * z)

    class _P(Problem):
        def define_problem(self):
            self += WedgePyrMesh(three_way=three_way)
            self += (PoissonEquation(source=src, space=order) + DirichletBC(u=uex) @ "outer") @ "domain"

    with _P() as p:
        p.max_refinement_level = level
        p.initialise()
        for _ in range(level):
            p.refine_uniformly()
        p.solve()
        m = p.get_mesh("domain")
        assert _dup(m) == 0, "duplicate (torn) nodes at the mixed wedge<->pyramid interface"
        assert _err(m, field) < 1e-10, f"mixed wedge+pyr (3way={three_way},{order},{field}) not reproduced at L{level}"


# Non-uniform 2:1 hanging across mixed interfaces. (label, three_way, refine-band): refining one region hangs
# the finer nodes on the coarser cross-shape neighbour. The three-way refine-wedge case hangs on BOTH a quad
# (wedge<->pyramid) and a triangular (wedge<->tet) interface at once.
@pytest.mark.parametrize("label,three_way,which", [
    ("wp_refine_pyr", False, "pyr"),
    ("wp_refine_wedge", False, "wedge"),
    ("3way_refine_wedge", True, "wedge"),
    ("3way_refine_pyr", True, "pyr"),
])
@pytest.mark.parametrize("field", ["linear", "quadratic"])
def test_mixed_wedge_pyramid_cross_shape_hanging(label, three_way, which, field):
    src = 0 if field == "linear" else -12
    x, y, z = var("coordinate")[0], var("coordinate")[1], var("coordinate")[2]
    uex = (x + 2 * y + 3 * z) if field == "linear" else (x * x + 2 * y * y + 3 * z * z)
    # refine the pyramid (midpoint y<0) or the wedge (midpoint y>0.05, and z in (0,1) so a capping tet stays coarse)
    if which == "pyr":
        lf = lambda e: 1 if e.get_Eulerian_midpoint()[1] < 0 else 0
    else:
        lf = lambda e: 1 if e.get_Eulerian_midpoint()[1] > 0.05 and 0.0 < e.get_Eulerian_midpoint()[2] < 1.0 else 0

    class _P(Problem):
        def define_problem(self):
            eqs = PoissonEquation(source=src, space="C2") + DirichletBC(u=uex) @ "outer"
            eqs += RefineAccordingToElement(level_func=lf)
            self += WedgePyrMesh(three_way=three_way)
            self += eqs @ "domain"

    with _P() as p:
        p.max_refinement_level = 1
        p.solve()
        m = p.get_mesh("domain")
        assert sum(1 for nn in m.nodes() if nn.is_hanging()) > 0, f"no hanging nodes ({label})"
        assert _dup(m) == 0, f"duplicate (torn) nodes at the mixed interface ({label})"
        assert _err(m, field) < 1e-10, f"mixed 2:1 hanging {label} ({field}) not reproduced (max err {_err(m, field):.2e})"


# Bricks in a mixed mesh: a hex sharing a QUAD face with a pyramid (base) or a wedge (side). The hex refines
# through the shared registry (build_as_brick_son) only when the forest is mixed; pure-brick meshes keep
# oomph-lib's native octree build. (other, order, field): linear is exact for both spaces, quadratic is the
# strict C2 oracle. UNIFORM (conforming) refinement here; 2:1 cross-shape hanging across a hex face is a
# separate step.
@pytest.mark.parametrize("other", ["pyr", "wedge"])
@pytest.mark.parametrize("order,field", [("C1", "linear"), ("C2", "linear"), ("C2", "quadratic")])
@pytest.mark.parametrize("level", [1, 2])
def test_mixed_hex_uniform_manufactured(other, order, field, level):
    src = 0 if field == "linear" else -12
    x, y, z = var("coordinate")[0], var("coordinate")[1], var("coordinate")[2]
    uex = (x + 2 * y + 3 * z) if field == "linear" else (x * x + 2 * y * y + 3 * z * z)

    class _P(Problem):
        def define_problem(self):
            self += HexMixMesh(other=other)
            self += (PoissonEquation(source=src, space=order) + DirichletBC(u=uex) @ "outer") @ "domain"

    with _P() as p:
        p.max_refinement_level = level
        p.initialise()
        for _ in range(level):
            p.refine_uniformly()
        p.solve()
        m = p.get_mesh("domain")
        assert _dup(m) == 0, f"duplicate (torn) nodes at the hex<->{other} interface"
        assert _err(m, field) < 1e-10, f"mixed hex+{other} ({order},{field}) not reproduced at L{level}"


# Non-uniform 2:1 cross-shape hanging ACROSS a hex face. Refining only the pyramid/wedge (band z>1) or only
# the hex (band z<1) makes the finer side's extra nodes on the shared quad face hang on the coarser
# cross-shape neighbour's interpolation -- installed by the mesh-level post_adapt pass, which now includes
# bricks (BRICK_FACES). Both the linear and the strict C2 quadratic field must be reproduced to machine
# precision. (other, which, field).
@pytest.mark.parametrize("other", ["pyr", "wedge"])
@pytest.mark.parametrize("which", ["refine_other", "refine_hex"])
@pytest.mark.parametrize("field", ["linear", "quadratic"])
def test_mixed_hex_cross_shape_hanging(other, which, field):
    src = 0 if field == "linear" else -12
    x, y, z = var("coordinate")[0], var("coordinate")[1], var("coordinate")[2]
    uex = (x + 2 * y + 3 * z) if field == "linear" else (x * x + 2 * y * y + 3 * z * z)
    # the hex occupies z in [0,1], the pyramid/wedge z in [1,2]; refine the band on one side of z=1
    if which == "refine_hex":
        lf = lambda e: 1 if e.get_Eulerian_midpoint()[2] < 1.0 else 0
    else:
        lf = lambda e: 1 if e.get_Eulerian_midpoint()[2] > 1.0 else 0

    class _P(Problem):
        def define_problem(self):
            eqs = PoissonEquation(source=src, space="C2") + DirichletBC(u=uex) @ "outer"
            eqs += RefineAccordingToElement(level_func=lf)
            self += HexMixMesh(other=other)
            self += eqs @ "domain"

    with _P() as p:
        p.max_refinement_level = 1
        p.solve()
        m = p.get_mesh("domain")
        assert sum(1 for nn in m.nodes() if nn.is_hanging()) > 0, f"no hanging nodes (hex+{other} {which})"
        assert _dup(m) == 0, f"duplicate (torn) nodes at the hex<->{other} 2:1 interface"
        assert _err(m, field) < 1e-10, f"hex+{other} 2:1 hanging ({which},{field}) not reproduced (err {_err(m, field):.2e})"


# MULTI-LEVEL (>1) non-uniform refinement across a mixed interface: refine ONE side to level 2 while the other
# starts coarse, so enforce_refinement_balance must refine the coarse side to level 1 (keeping every jump 2:1)
# and the cross-round position snapshot must let a level-2 element reuse a node a level-1 neighbour built in
# the previous round. Both are exercised here: without balancing a hex+pyramid mesh jumps 0->2 (fails); without
# the cross-round reuse the registry families tear (dup>0). Small meshes + a one-sided band keep it cheap.
@pytest.mark.parametrize("label,mesh,lf", [
    ("tet_wedge", lambda: TetWedgeMesh(), lambda e: 2 if e.get_Eulerian_midpoint()[2] > 1.0 else 0),
    ("wedge_pyr", lambda: WedgePyrMesh(), lambda e: 2 if e.get_Eulerian_midpoint()[1] < 0 else 0),
    ("three_way", lambda: WedgePyrMesh(three_way=True), lambda e: 2 if e.get_Eulerian_midpoint()[1] < 0 else 0),
    ("hex_pyr", lambda: HexMixMesh(other="pyr"), lambda e: 2 if e.get_Eulerian_midpoint()[2] > 1.0 else 0),
    ("hex_wedge", lambda: HexMixMesh(other="wedge"), lambda e: 2 if e.get_Eulerian_midpoint()[2] > 1.0 else 0),
])
def test_mixed_multilevel_quadratic(label, mesh, lf):
    x, y, z = var("coordinate")[0], var("coordinate")[1], var("coordinate")[2]

    class _P(Problem):
        def define_problem(self):
            eqs = PoissonEquation(source=-12, space="C2") + DirichletBC(u=x * x + 2 * y * y + 3 * z * z) @ "outer"
            eqs += RefineAccordingToElement(level_func=lf)
            self += mesh()
            self += eqs @ "domain"

    with _P() as p:
        p.max_refinement_level = 2
        p.solve(spatial_adapt=3)
        m = p.get_mesh("domain")
        lvls = sorted({m.element_pt(ie).refinement_level() for ie in range(m.nelement())})
        assert 2 in lvls, f"{label}: level-2 refinement did not happen (levels {lvls})"
        assert max(lvls) - min(lvls) <= 1, f"{label}: unbalanced jump >2:1 (levels {lvls})"  # balancing worked
        assert _dup(m) == 0, f"{label}: torn nodes at the multi-level interface (cross-round sharing failed)"
        assert _err(m, "quadratic") < 1e-10, f"{label}: quadratic not reproduced at multi-level (err {_err(m, 'quadratic'):.2e})"


# NODE IDENTIFICATION MUST BE PURELY TOPOLOGICAL -- the mixed-3d statement of the invariant tested for
# triangles and tetrahedra in test_triangle_refinement.py / test_tet_refinement.py. These families share
# nodes by the (father node, rounded father-shape weight) key rather than by a tree walk, and the
# cross-round half of that -- reusing a node an earlier round built -- used to be done by matching node
# POSITIONS. That is only as good as the positions are, and a hanging node's position is a cache of its
# masters which goes stale whenever something writes the dof vector from outside the Newton solver (see
# dev_docs/adaptive_refinement.md §7). It is now the same key, snapshotted from the live nodes.
#
# Displacing every hanging node before each adapt is exactly what that stale cache looks like, and is
# harmless because the values are recomputed at the next assembly. The refinement pattern is keyed on the
# element's LAGRANGIAN midpoint, which the displacement does not touch, so what gets refined cannot change
# and any difference in the resulting topology is a sharing decision that consulted a position.
# axis/sign/thresholds follow the same band each layout is refined in by test_mixed_multilevel_quadratic
# (z for the tet/hex families, y for the wedge+pyramid ones), so the band really straddles the mixed
# interface and hanging nodes actually appear -- the test asserts that it is not vacuous.
@pytest.mark.parametrize("label,mesh,axis,sign,thresholds", [
    ("tet_wedge", lambda: TetWedgeMesh(), 2, +1, (0.6, 1.4, 0.4)),
    ("wedge_pyr", lambda: WedgePyrMesh(), 1, -1, (-0.3, 0.3, -0.1)),
    ("three_way", lambda: WedgePyrMesh(three_way=True), 1, -1, (-0.3, 0.3, -0.1)),
    ("hex_pyr", lambda: HexMixMesh(other="pyr"), 2, +1, (0.6, 1.4, 0.4)),
    ("hex_wedge", lambda: HexMixMesh(other="wedge"), 2, +1, (0.6, 1.4, 0.4)),
])
def test_mixed_node_sharing_ignores_node_positions(label, mesh, axis, sign, thresholds):
    x, y, z = var("coordinate")[0], var("coordinate")[1], var("coordinate")[2]

    class _P(Problem):
        def __init__(self):
            super().__init__()
            self.threshold = thresholds[0]

        def define_problem(self):
            eqs = PoissonEquation(source=-12, space="C2") + DirichletBC(u=x * x + 2 * y * y + 3 * z * z) @ "outer"
            eqs += RefineAccordingToElement(
                level_func=lambda e: 2 if sign * (e.get_Lagrangian_midpoint()[axis] - self.threshold) > 0 else 0,
                prevent_unrefinement=False)
            self += mesh()
            self += eqs @ "domain"

    def run(displace):
        history, ndisplaced = [], 0
        with _P() as p:
            p.max_refinement_level = 2
            p.solve(spatial_adapt=3)
            m = p.get_mesh("domain")
            for threshold in thresholds:
                p.threshold = threshold
                if displace:
                    for n in m.nodes():
                        if n.is_hanging():
                            ndisplaced += 1
                            n.set_x(0, n.x(0) + 0.083)  # >> any element size at these levels
                            n.set_x(1, n.x(1) - 0.071)
                            n.set_x(2, n.x(2) + 0.067)
                p.adapt()
                p.solve()
                history.append((m.nnode(), m.nelement()))
        return history, ndisplaced

    clean, _ = run(False)
    stale, ndisplaced = run(True)
    assert ndisplaced > 0, f"{label}: no hanging nodes were displaced -- the test would be vacuous"
    assert clean == stale, (
        f"{label}: adapting with stale hanging-node positions changed the mesh topology, so node "
        f"identification still depends on positions: clean={clean} stale={stale}")
