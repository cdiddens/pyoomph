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

# Phase 4 (branch mixed_adapt): tree-based refinement of pure tetrahedral meshes.
#
# A linear (C1) tetrahedron refines 1->8 reusing the OcTree hierarchy (8 sons), with node-sharing
# done GEOMETRICALLY (a new edge-midpoint node is keyed by the father node pair it bisects) rather
# than via oomph's brick coordinate descent. Hanging nodes lie in the interior of a coarser tet's
# edge (constrained linearly) or, for >1-level jumps, its face (constrained barycentrically), set
# by a mesh-level pass after refinement. Since the Poisson problem is linear, a single Newton step
# drives the residual to machine zero iff the hanging-node Jacobian is correct.
#
# Correctness is validated for uniform, single-level (2:1-balanced) and error-driven adaptive
# refinement -- the meshes adaptivity actually produces. (Abrupt >1-level RefineToLevel jumps are
# a known-incomplete edge case and are not asserted here.)

import numpy as np
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.meshes.mesh import MeshTemplate
from pyoomph.equations.poisson import PoissonEquation


class TetCubeMesh(MeshTemplate):
    # Unit cube [0,1]^3 as N^3 cells, each split into 6 tetrahedra (Kuhn/Freudenthal, diagonal
    # c000-c111). Boundary triangle facets are added for tet faces lying on a domain face.
    _TETS_LOCAL = [(0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)]

    def __init__(self, N=1, space="C1"):
        super().__init__()
        self.N = N
        self.space = space

    def define_geometry(self):
        N = self.N
        dom = self.new_domain("domain")
        coord = {}

        def node(gx, gy, gz):
            idx = self.add_node_unique(gx / N, gy / N, gz / N)
            coord[idx] = (gx, gy, gz)
            return idx

        cubes = []
        for ix in range(N):
            for iy in range(N):
                for iz in range(N):
                    c = {}
                    for bx in (0, 1):
                        for by in (0, 1):
                            for bz in (0, 1):
                                c[4 * bz + 2 * by + bx] = node(ix + bx, iy + by, iz + bz)
                    cubes.append(c)
                    for tl in self._TETS_LOCAL:
                        nn = [c[i] for i in tl]
                        dom.add_tetra_3d_C1(nn[0], nn[1], nn[2], nn[3])

        bounds = {"left": (0, 0), "right": (0, N), "bottom": (1, 0),
                  "top": (1, N), "back": (2, 0), "front": (2, N)}
        for c in cubes:
            for tl in self._TETS_LOCAL:
                nn = [c[i] for i in tl]
                faces = [(nn[1], nn[2], nn[3]), (nn[0], nn[2], nn[3]), (nn[0], nn[1], nn[3]), (nn[0], nn[1], nn[2])]
                for f in faces:
                    for bname, (ax, val) in bounds.items():
                        if all(coord[n][ax] == val for n in f):
                            self.add_facet_to_boundary(bname, list(f))


_ALL_BOUNDS = ["left", "right", "top", "bottom", "front", "back"]


def _max_abs_residual(problem):
    return float(np.max(np.abs(np.asarray(problem.get_residuals()))))


def _is_manifold(summary):
    n_facets, n_bnd, n_int, max_inc = summary
    return max_inc == 2 and n_bnd + n_int == n_facets


class _TetPoisson(Problem):
    def __init__(self, N=2, source=None, space="C1"):
        super().__init__()
        self._N = N
        self._source = source if source is not None else 1
        self._space = space

    def define_problem(self):
        self += TetCubeMesh(N=self._N, space=self._space)
        eqs = PoissonEquation(source=self._source, space=self._space) + DirichletBC(u=0) @ _ALL_BOUNDS
        eqs += SpatialErrorEstimator(u=1)
        self += eqs @ "domain"


def test_uniform_tet_refinement_conforming():
    # A tet refines 1->8; the mesh must stay conforming (manifold) and element count x8 per level.
    with _TetPoisson(N=1) as problem:
        problem.max_refinement_level = 3
        problem.initialise()
        m = problem.get_mesh("domain")
        assert m.nelement() == 6  # one cube -> 6 tets
        assert _is_manifold(list(m.facet_adjacency_summary()))
        for expected in (48, 384):
            m.refine_uniformly()
            assert m.nelement() == expected
            assert _is_manifold(list(m.facet_adjacency_summary()))


# Single-level (2:1) refinement combos. C1 is cheap so it sweeps boundaries/levels; C2 (10-node,
# expensive at level 3) gets a lighter but still level-3-reaching set. (deeper level-3 C2 solves are
# large, so we keep the count small.)
@pytest.mark.parametrize("space,boundary,lo,hi", [
    ("C1", "left", 1, 2), ("C1", "top", 2, 3), ("C1", "front", 1, 2), ("C1", "left", 2, 3),
    ("C2", "left", 1, 2), ("C2", "top", 2, 3),
])
def test_single_level_tet_hanging_residual_oracle(space, boundary, lo, hi):
    # Single-level (2:1) refinement near one boundary. Linear residual -> machine zero certifies the
    # hanging-node Jacobian, for both linear (C1) and quadratic (C2) tetrahedra.
    with _TetPoisson(N=3, space=space) as problem:
        problem.max_refinement_level = 4
        problem += RefineToLevel(lo) @ "domain"
        problem += RefineToLevel(hi) @ ("domain/" + boundary)
        problem.solve()
        assert _max_abs_residual(problem) < 1e-9


@pytest.mark.parametrize("space", ["C1", "C2"])
def test_abrupt_multilevel_tet_hanging_residual_oracle(space):
    # A deliberately non-2:1 abrupt >1-level jump (level 1 domain, level 3 near one boundary). The
    # 2:1 balancing pass + hang-chain flattening must still drive the linear residual to machine
    # zero (small N=1 mesh so the balancing-induced refinement stays cheap, esp. for C2).
    with _TetPoisson(N=1, space=space) as problem:
        problem.max_refinement_level = 4
        problem += RefineToLevel(1) @ "domain"
        problem += RefineToLevel(3) @ "domain/left"
        problem.solve()
        assert _max_abs_residual(problem) < 1e-9


@pytest.mark.parametrize("space", ["C1", "C2"])
@pytest.mark.parametrize("nadapt", [1, 2])
def test_error_based_tet_adaptivity_residual_oracle(space, nadapt):
    # Genuine Z2 error-driven adaptive refinement around a localized source. The refined mesh is
    # non-uniform (hanging nodes) but 2:1-balanced; the linear residual must reach machine zero.
    x, y, z = var("coordinate")[0], var("coordinate")[1], var("coordinate")[2]
    src = 100 * exp(-40 * ((x - 0.3) ** 2 + (y - 0.3) ** 2 + (z - 0.3) ** 2))
    with _TetPoisson(N=3, source=src, space=space) as problem:
        problem.max_refinement_level = 4
        problem.solve(spatial_adapt=nadapt)
        assert _max_abs_residual(problem) < 1e-9
        assert problem.get_mesh("domain").nelement() > 162  # more than the 27*6 base tets: adaption happened


from pyoomph.equations.navier_stokes import StokesEquations


class _CavityStokes3D(Problem):
    # Lid-driven cavity Stokes flow with 3D Taylor-Hood (C2 velocity + C1 pressure) tetrahedra. On a
    # C2-coordinate tet mesh the C1 pressure owns a SEPARATE hang slot (hangindex >= 0) and must hang
    # linearly on the coarse edge corners -- even at the coarse edge mid-node, whose velocity is a
    # real dof but whose pressure must hang. Without that separate pressure hang the Jacobian is
    # wrong and the (linear) Stokes Newton step does not converge in one step.
    def __init__(self, N=2, error_adapt=False):
        super().__init__()
        self._N = N
        self._error_adapt = error_adapt

    def define_problem(self):
        self += TetCubeMesh(N=self._N)
        stokes = StokesEquations(mode="TH", dynamic_viscosity=1)
        eqs = stokes
        eqs += DirichletBC(velocity_x=1, velocity_y=0, velocity_z=0) @ "top"
        eqs += DirichletBC(velocity_x=0, velocity_y=0, velocity_z=0) @ ["left", "right", "bottom", "front", "back"]
        eqs += stokes.create_pressure_fixation(value=0)
        if self._error_adapt:
            eqs += SpatialErrorEstimator(velocity=1)
        self += eqs @ "domain"


def test_taylor_hood_tet_cavity_single_level():
    # Single-level (2:1) non-conforming refinement near the lid of a 3D Taylor-Hood tet mesh. Stokes
    # is linear, so max|residual| ~ 0 after solve certifies the mixed C2-velocity / C1-pressure
    # hanging-node Jacobian (the C1 pressure hangs linearly on the coarse edge corners).
    with _CavityStokes3D(N=2) as problem:
        problem.max_refinement_level = 2
        problem += RefineToLevel(1) @ "domain"
        problem += RefineToLevel(2) @ "domain/top"
        problem.solve()
        assert _max_abs_residual(problem) < 1e-9
        assert problem.get_mesh("domain").nelement() > 384  # uniform level 1 is 384; the lid added more


def test_taylor_hood_tet_cavity_error_adaptivity():
    # Genuine Z2 error-driven adaptivity (the 2:1-balanced non-uniform meshes adaptivity produces).
    # The mixed C2/C1 hanging Jacobian must keep the linear Stokes residual at machine zero.
    with _CavityStokes3D(N=2, error_adapt=True) as problem:
        problem.max_refinement_level = 2
        problem.solve(spatial_adapt=1)
        assert _max_abs_residual(problem) < 1e-9
        assert problem.get_mesh("domain").nelement() > 384
