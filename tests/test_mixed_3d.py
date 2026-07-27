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

# Branch mixed_adapt, slice A: genuinely MIXED 3d meshes (>=2 of {tet, wedge, pyramid} in one domain),
# refined via the shared OcTree forest. The registry-based families (no bricks) all key their new interface
# nodes into one weight-augmented shared-node registry (RefineablePyramidElement::Mixed_forest_active), so a
# tet and an adjacent wedge sharing a TRIANGULAR face create one shared node instead of a torn pair. A tet
# can only meet a wedge on the wedge's triangular cap (a tet has no quad faces), and both refine that triangle
# 1->4 with matching edge-midpoints, so uniform refinement stays conforming.
#
# This file covers UNIFORM (conforming) mixed refinement. Non-uniform (2:1 cross-shape hanging) across a
# mixed interface is a separate step.

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
