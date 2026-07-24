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

# Phase 2 (branch mixed_adapt): tree-based refinement of pure triangular meshes.
#
# Phase 2a -- UNIFORM refinement: a linear (C1) triangle refines 1->4 via the QuadTree
# hierarchy, with mid-edge nodes shared geometrically (father-edge corner-node registry) so
# the refined mesh stays conforming (manifold: every facet incident on 1 or 2 element faces).
# We check element counts (x4 per uniform level), manifoldness, and Poisson-integral convergence.
#
# Phase 2b -- NON-UNIFORM (hanging-node) refinement: a node lying in the interior of a coarser
# neighbour's edge is constrained by linear interpolation of that edge's two end nodes. Because
# the Poisson problem is linear, a single Newton step drives the residual to machine zero *iff*
# the hanging-node constraints (and hence the assembled Jacobian) are correct -- the same oracle
# used by test_constrained_adaptivity.py.

import numpy as np

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.meshes.simplemeshes import RectangularQuadMesh
from pyoomph.equations.poisson import PoissonEquation


# Reference: integral of u over the unit square for -laplace(u)=1, u=0 on the boundary.
_INT_U_REF = 0.035144


class _TriPoisson(Problem):
    def define_problem(self):
        # Linear (C1) triangles (3-node) so the shared-node registry applies.
        self += RectangularQuadMesh(name="domain", N=2, split_in_tris="left")
        eqs = PoissonEquation(source=1, space="C1") + DirichletBC(u=0) @ ["left", "right", "top", "bottom"]
        eqs += IntegralObservables(intu=var("u"))
        self += eqs @ "domain"


def _is_manifold(summary):
    n_facets, n_bnd, n_int, max_inc = summary
    return max_inc == 2 and n_bnd + n_int == n_facets


def _max_abs_residual(problem):
    return float(np.max(np.abs(np.asarray(problem.get_residuals()))))


def test_uniform_triangle_refinement_conforming_and_converges():
    with _TriPoisson() as problem:
        problem.max_refinement_level = 4
        # Force UNIFORM refinement to level 2 (RefineToLevel drives every element to the same
        # level -> conforming, no hanging nodes).
        problem += RefineToLevel(2) @ "domain"
        problem.initialise()
        m = problem.get_mesh("domain")

        # Base mesh: 2x2 quads split into 8 triangles; uniform 1->4 twice -> 8*4*4 = 128.
        assert m.nelement() == 128
        assert _is_manifold(list(m.facet_adjacency_summary()))

        problem.solve()
        intu = float(m.evaluate_observable("intu"))
        # Converges from below to the known value; loose tolerance for a level-2 mesh.
        assert 0.032 < intu < _INT_U_REF
        assert abs(intu - _INT_U_REF) < 0.003


def test_uniform_triangle_refinement_level_counts():
    # Uniform refinement to levels 0/1/2 (each via RefineToLevel in a fresh problem, so no
    # error-based non-uniform adaption is triggered). Element counts x4 per level, always
    # conforming, and the Poisson integral converges monotonically from below.
    counts = {0: 8, 1: 32, 2: 128}
    prev_intu = None
    for level in (0, 1, 2):
        with _TriPoisson() as problem:
            problem.max_refinement_level = 4
            if level > 0:
                problem += RefineToLevel(level) @ "domain"
            problem.initialise()
            m = problem.get_mesh("domain")
            assert m.nelement() == counts[level]
            assert _is_manifold(list(m.facet_adjacency_summary()))
            problem.solve()
            intu = float(m.evaluate_observable("intu"))
            if prev_intu is not None:
                assert intu > prev_intu  # monotone convergence from below
            prev_intu = intu


class _TriPoissonPlain(Problem):
    # Same as _TriPoisson but without the observable (kept minimal for the residual oracle).
    def __init__(self, split="left", N=4):
        super().__init__()
        self._split, self._N = split, N

    def define_problem(self):
        self += RectangularQuadMesh(name="domain", N=self._N, split_in_tris=self._split)
        eqs = PoissonEquation(source=1, space="C1") + DirichletBC(u=0) @ ["left", "right", "top", "bottom"]
        self += eqs @ "domain"


import pytest


@pytest.mark.parametrize("split", ["left", "right", "crossed"])
@pytest.mark.parametrize("boundary", ["left", "top"])
def test_nonuniform_triangle_refinement_residual_oracle(split, boundary):
    # Refine more strongly near one boundary than the interior -> hanging nodes at the interface.
    # Linear problem: residual ~0 after the Newton step certifies the hanging-node Jacobian.
    with _TriPoissonPlain(split=split) as problem:
        problem.max_refinement_level = 3
        problem += RefineToLevel(1) @ "domain"
        problem += RefineToLevel(3) @ ("domain/" + boundary)
        problem.solve()
        assert _max_abs_residual(problem) < 1e-9


class _TriPoissonAdaptive(Problem):
    def define_problem(self):
        self += RectangularQuadMesh(name="domain", N=6, split_in_tris="crossed")
        x = var("coordinate")[0]
        y = var("coordinate")[1]
        eqs = PoissonEquation(source=100 * exp(-50 * ((x - 0.2) ** 2 + (y - 0.2) ** 2)), space="C1")
        eqs += DirichletBC(u=0) @ ["left", "right", "top", "bottom"]
        eqs += SpatialErrorEstimator(u=1)
        self += eqs @ "domain"


def test_error_based_triangle_adaptivity_residual_oracle():
    # Genuine error-driven (Z2) adaptive refinement of a triangular mesh around a localized
    # source. The refined mesh is non-uniform (hanging nodes); the linear residual must still
    # reach machine zero.
    with _TriPoissonAdaptive() as problem:
        problem.max_refinement_level = 4
        problem.solve(spatial_adapt=3)
        assert _max_abs_residual(problem) < 1e-9
        assert problem.get_mesh("domain").nelement() > 1000  # adaption actually happened
