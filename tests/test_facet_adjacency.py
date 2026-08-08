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

# Tests the shape/split-scheme-neutral facet-adjacency primitive used by the
# generic mixed-mesh refinement engine (see dev_docs/adaptive_refinement.md).
# On a freshly generated (conforming) mesh, every facet must be incident on
# exactly 1 (boundary) or 2 (interior) element faces regardless of element shape.

import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.meshes.simplemeshes import RectangularQuadMesh, CuboidBrickMesh
from pyoomph.equations.poisson import PoissonEquation


def _summary_for(meshobj, boundary_names):
    class _P(Problem):
        def define_problem(self):
            self += meshobj
            self += (PoissonEquation(source=1) + DirichletBC(u=0) @ boundary_names) @ "domain"

    with _P() as problem:
        problem.initialise()
        n_facets, n_bnd, n_int, max_inc = problem.get_mesh("domain").facet_adjacency_summary()
        return int(n_facets), int(n_bnd), int(n_int), int(max_inc)


def _assert_manifold(summary):
    n_facets, n_bnd, n_int, max_inc = summary
    # Conforming mesh: no non-manifold / hanging facets.
    assert max_inc == 2, f"non-manifold facet (max incidence {max_inc})"
    # Every facet is either boundary (1) or interior (2); nothing left over.
    assert n_bnd + n_int == n_facets
    assert n_bnd > 0 and n_int > 0


BOUNDS_2D = ["left", "right", "top", "bottom"]
BOUNDS_3D = ["left", "right", "top", "bottom", "front", "back"]


def test_facet_adjacency_quads():
    # N x N quad mesh has 2*N*(N+1) edges, 4*N on the boundary.
    N = 3
    summary = _summary_for(RectangularQuadMesh(name="domain", N=N), BOUNDS_2D)
    _assert_manifold(summary)
    n_facets, n_bnd, n_int, _ = summary
    assert n_facets == 2 * N * (N + 1)
    assert n_bnd == 4 * N


@pytest.mark.parametrize("split", ["left", "right", "crossed"])
def test_facet_adjacency_triangles(split):
    # Triangulating the quads keeps the mesh conforming and manifold; the boundary
    # edge count is unchanged (the split only adds interior diagonals).
    N = 3
    summary = _summary_for(RectangularQuadMesh(name="domain", N=N, split_in_tris=split), BOUNDS_2D)
    _assert_manifold(summary)
    assert summary[1] == 4 * N  # boundary edges unchanged by interior triangulation


def test_facet_adjacency_bricks():
    # N x N x N brick mesh has 3*N^2*(N+1) faces, 6*N^2 on the boundary.
    N = 2
    summary = _summary_for(CuboidBrickMesh(N=N), BOUNDS_3D)
    _assert_manifold(summary)
    n_facets, n_bnd, _, _ = summary
    assert n_facets == 3 * N * N * (N + 1)
    assert n_bnd == 6 * N * N
