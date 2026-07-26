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

# Boundary-element identification (Boundary_element_pt / Face_index_at_boundary) via the per-element
# face boundary tags: seeded once from the MeshTemplate's facet records and inherited by the sons at
# every split (BulkElementBase::dynamic_split -> face_index_in_father), replacing the per-shape
# reconstruction from nodal boundary membership.
#
# The reconstruction from nodal membership cannot distinguish a genuine boundary face from an
# INTERIOR face all of whose vertices happen to lie on one and the same boundary. Both shapes have
# such a configuration:
#   * triangle: two edges on boundary b -> the third (interior) edge also has both ends on b,
#   * quad: opposite walls sharing a boundary name -> all four corners on b, so all four faces match.
# For triangles the old code masked this with an "edge seen exactly once" filter, which silently
# breaks as soon as the neighbour across that edge is refined to a different level -- i.e. exactly on
# the non-uniformly adapted meshes this test drives.

import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.meshes.simplemeshes import RectangularQuadMesh
from pyoomph.equations.poisson import PoissonEquation

_TOL = 1e-9

# All four sides of the unit square merged into ONE boundary, so every corner element has two of its
# faces on the same boundary.
_ONE_WALL = {"left": "wall", "right": "wall", "top": "wall", "bottom": "wall"}


class _Wall(Problem):
    def __init__(self, N=1, split="left", size=1.0, boundary_names=None):
        super().__init__()
        self._N, self._split, self._size = N, split, size
        self._bnames = _ONE_WALL if boundary_names is None else boundary_names

    def define_problem(self):
        self += RectangularQuadMesh(name="domain", N=self._N, size=self._size,
                                    split_in_tris=self._split, boundary_names=self._bnames)
        eqs = PoissonEquation(source=1, space="C1") + DirichletBC(u=0) @ "wall"
        self += eqs @ "domain"


def _centroid(el):
    n = el.nnode()
    return (sum(el.node_pt(k).x(0) for k in range(n)) / n,
            sum(el.node_pt(k).x(1) for k in range(n)) / n)


def _face_ends(el, face_index):
    """The two end nodes of a 2d element face, as (x,y) pairs."""
    n0 = el.boundary_node_pt(face_index, 0)
    n1 = el.boundary_node_pt(face_index, el.nnode_1d() - 1)
    return (n0.x(0), n0.x(1)), (n1.x(0), n1.x(1))


def _face_key(p0, p1):
    # Faces are identified geometrically (the Python element wrappers have no stable identity across
    # calls). A boundary face belongs to exactly one element, so the rounded, unordered pair of end
    # points is a unique key for the comparison below.
    return frozenset((round(p0[0], 9), round(p0[1], 9)) for p0 in (p0, p1))


def _check_boundary_faces(mesh, bname, on_boundary):
    """Return (n_registered, false_positives, missing).

    A face is a genuine boundary face iff both its end points AND its midpoint satisfy the geometric
    predicate `on_boundary` (the midpoint test is what rejects an interior face spanning a corner).
    Cross-checks the registered set against every face of every element, so it catches both
    over-detection (interior faces registered) and under-detection (boundary faces missed).
    """
    b = mesh.get_boundary_index(bname)
    registered, false_positives = set(), []
    for e in range(mesh.nboundary_element(b)):
        el = mesh.boundary_element_pt(b, e)
        fi = mesh.face_index_at_boundary(b, e)
        p0, p1 = _face_ends(el, fi)
        mid = (0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1]))
        if not on_boundary(*mid):
            false_positives.append((p0, p1))
        registered.add(_face_key(p0, p1))

    missing = []
    for e in range(mesh.nelement()):
        el = mesh.element_pt(e)
        for fi in ((0, 1, 2) if el.nnode() in (3, 6, 7) else (-2, -1, 1, 2)):
            p0, p1 = _face_ends(el, fi)
            mid = (0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1]))
            if on_boundary(*p0) and on_boundary(*p1) and on_boundary(*mid):
                if _face_key(p0, p1) not in registered:
                    missing.append((p0, p1))
    return mesh.nboundary_element(b), false_positives, missing


def _unit_square_wall(x, y):
    return abs(x) < _TOL or abs(x - 1) < _TOL or abs(y) < _TOL or abs(y - 1) < _TOL


def test_triangle_third_edge_not_a_boundary_under_nonuniform_refinement():
    # Unit square as two triangles, whole perimeter = one boundary "wall". Both triangles therefore
    # have two edges on "wall", so their shared diagonal has both ends on "wall" while being
    # interior. Refining only ONE side of that diagonal makes the coarse element's diagonal a face
    # that no same-level neighbour shares -- which is what defeated the old edge-count filter (it
    # registered the whole diagonal, and after further refinement every sub-diagonal too).
    with _Wall(N=1, split="left") as pr:
        pr.max_refinement_level = 5
        pr.initialise()
        m = pr.get_mesh("domain")

        n, fp, missing = _check_boundary_faces(m, "wall", _unit_square_wall)
        assert (n, fp, missing) == (4, [], [])

        # Perimeter faces after refining only the elements above the diagonal: the untouched
        # triangle keeps its 2 coarse edges, the refined side doubles its 2 edges per level.
        for level, expected in enumerate((6, 10, 18), start=1):
            sel = [e for e in range(m.nelement())
                   if _centroid(m.element_pt(e))[1] > _centroid(m.element_pt(e))[0]]
            m.refine_selected_elements(sel)
            n, fp, missing = _check_boundary_faces(m, "wall", _unit_square_wall)
            assert fp == [], f"level {level}: interior faces registered as boundary: {fp}"
            assert missing == [], f"level {level}: boundary faces not registered: {missing}"
            assert n == expected, f"level {level}: {n} boundary faces, expected {expected}"


def test_quad_channel_with_shared_wall_name():
    # The quad analogue: a one-element-wide channel whose left and right walls share a boundary name
    # puts all four corners of every quad on "wall", so the nodal reconstruction also registers the
    # two horizontal (interior or differently-named) faces of each element -- 16 instead of 8.
    class _Channel(Problem):
        def define_problem(self):
            self += RectangularQuadMesh(name="domain", N=[1, 4], size=[0.25, 1],
                                        boundary_names={"left": "wall", "right": "wall"})
            self += (PoissonEquation(source=1, space="C1") + DirichletBC(u=0) @ "wall") @ "domain"

    def on_wall(x, y):
        return abs(x) < _TOL or abs(x - 0.25) < _TOL

    with _Channel() as pr:
        pr.max_refinement_level = 3
        pr.initialise()
        m = pr.get_mesh("domain")
        n, fp, missing = _check_boundary_faces(m, "wall", on_wall)
        assert fp == [], f"interior/other-boundary faces registered as 'wall': {fp}"
        assert missing == [], f"wall faces not registered: {missing}"
        assert n == 8


@pytest.mark.parametrize("split", ["left", "right", "crossed"])
def test_error_driven_adaptivity_keeps_boundary_faces_exact(split):
    # Genuine Z2 error-driven, non-uniform adaptation of a triangular mesh whose whole perimeter is
    # a single boundary, with the source peaked in the interior so the corner elements stay coarse
    # while their interior neighbours refine.
    class _Adaptive(Problem):
        def define_problem(self):
            self += RectangularQuadMesh(name="domain", N=2, split_in_tris=split,
                                        boundary_names=_ONE_WALL)
            x, y = var("coordinate")[0], var("coordinate")[1]
            eqs = PoissonEquation(source=200 * exp(-80 * ((x - 0.5) ** 2 + (y - 0.5) ** 2)), space="C1")
            eqs += DirichletBC(u=0) @ "wall"
            eqs += SpatialErrorEstimator(u=1)
            self += eqs @ "domain"

    with _Adaptive() as pr:
        pr.max_refinement_level = 4
        pr.solve(spatial_adapt=4)
        m = pr.get_mesh("domain")
        n, fp, missing = _check_boundary_faces(m, "wall", _unit_square_wall)
        assert fp == [], f"interior faces registered as boundary: {fp}"
        assert missing == [], f"boundary faces not registered: {missing}"
        assert n > 8  # the boundary did get refined somewhere


def test_named_boundaries_are_unaffected_by_the_tag_route():
    # Sanity check on the ordinary case (four separately named sides, quads and triangles): the
    # per-side boundary element counts must be exactly what uniform refinement predicts.
    for split, per_side in ((False, 8), ("left", 8)):
        class _Plain(Problem):
            def define_problem(self):
                self += RectangularQuadMesh(name="domain", N=2, split_in_tris=split)
                eqs = PoissonEquation(source=1, space="C1")
                eqs += DirichletBC(u=0) @ ["left", "right", "top", "bottom"]
                self += eqs @ "domain"

        with _Plain() as pr:
            pr.max_refinement_level = 3
            pr += RefineToLevel(2) @ "domain"
            pr.initialise()
            m = pr.get_mesh("domain")
            for side in ("left", "right", "top", "bottom"):
                b = m.get_boundary_index(side)
                assert m.nboundary_element(b) == per_side, (split, side, m.nboundary_element(b))
