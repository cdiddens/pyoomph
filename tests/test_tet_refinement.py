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
from pyoomph.equations.additional import RefineAccordingToElement  # not in "from pyoomph import *"
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


@pytest.mark.parametrize("space", ["C1", "C1TB", "C2", "C2TB"])
def test_uniform_tet_refinement_conforming(space):
    # A tet refines 1->8; the mesh must stay conforming (manifold) and element count x8 per level.
    #
    # Every enrichment of the tet is swept here because refining a C1TB one used to SEGFAULT, and did
    # so on the very first refinement of any tet mesh: BulkElementTetra3dC1TB::local_coordinate_of_node
    # wrote s[0..2] without resizing s, and FiniteElement::get_node_at_local_coordinate hands it a
    # default-constructed (empty) Vector to size itself. Every other shape in the codebase -- the 2d
    # C1TB triangle, the wedges, the pyramids, oomph's own TBubbleEnrichedElementShape -- resizes
    # first; this one had been missed, so no test ever refined a C1TB tet. Note that a regression here
    # takes the interpreter down rather than failing this assertion.
    with _TetPoisson(N=1, space=space) as problem:
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


@pytest.mark.parametrize("space", ["C1", "C1TB", "C2", "C2TB"])
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


class _CavityStokes3DCR(Problem):
    # Lid-driven cavity Stokes with 3D Crouzeix-Raviart tets: C2TB bubble-enriched velocity (10 C2
    # nodes + 4 face-centroid + 1 volume-centroid bubble) and DL discontinuous pressure. Refinement
    # must build & SHARE the 4 face-bubble nodes across face-adjacent tets (continuous velocity),
    # keep the volume bubble interior, and -- crucially, unlike C2/Taylor-Hood -- hang the C2TB
    # face-interior fine nodes (sub-face bubbles + inner-edge-mids) that appear on a 2:1 coarse face,
    # using the enriched triangle face interpolation over the coarse face's 7 nodes. CR is 2-Newton-
    # step and more poorly conditioned than TH (residual floor ~1e-11 even without refinement), so
    # the tolerance is looser (~1e-7).
    def __init__(self, N=2, error_adapt=False):
        super().__init__()
        self._N = N
        self._error_adapt = error_adapt

    def define_problem(self):
        self += TetCubeMesh(N=self._N)
        stokes = StokesEquations(mode="CR", dynamic_viscosity=1)
        eqs = stokes
        eqs += DirichletBC(velocity_x=1, velocity_y=0, velocity_z=0) @ "top"
        eqs += DirichletBC(velocity_x=0, velocity_y=0, velocity_z=0) @ ["left", "right", "bottom", "front", "back"]
        eqs += stokes.create_pressure_fixation(value=0)
        if self._error_adapt:
            eqs += SpatialErrorEstimator(velocity=1)
        self += eqs @ "domain"


def test_crouzeix_raviart_tet_cavity_uniform():
    # Uniform (conforming) refinement of a 3D CR mesh: no hanging, so this isolates the bubble-node
    # build -- creating the volume bubble fresh and sharing the 4 face bubbles across adjacent tets.
    with _CavityStokes3DCR(N=1) as problem:
        problem.max_refinement_level = 2
        problem += RefineToLevel(1) @ "domain"
        problem.solve()
        assert _max_abs_residual(problem) < 1e-7
        assert problem.get_mesh("domain").nelement() == 48  # 6 tets x8


def test_crouzeix_raviart_tet_cavity_single_level():
    # Single-level (2:1) non-conforming CR refinement: exercises the enriched C2TB face-interior
    # hanging (sub-face bubbles + inner-edge-mids hang on the coarse face's 7 nodes).
    with _CavityStokes3DCR(N=2) as problem:
        problem.max_refinement_level = 2
        problem += RefineToLevel(1) @ "domain"
        problem += RefineToLevel(2) @ "domain/top"
        problem.solve()
        assert _max_abs_residual(problem) < 1e-7
        assert problem.get_mesh("domain").nelement() > 384


def test_crouzeix_raviart_tet_cavity_error_adaptivity():
    # Genuine Z2 error-driven adaptivity (the 2:1-balanced meshes adaptivity produces) for 3D CR.
    with _CavityStokes3DCR(N=2, error_adapt=True) as problem:
        problem.max_refinement_level = 2
        problem.solve(spatial_adapt=1)
        assert _max_abs_residual(problem) < 1e-7
        assert problem.get_mesh("domain").nelement() > 384


def test_taylor_hood_tet_unrefinement_residual_oracle():
    # 3D Taylor-Hood unrefinement (coarsening). Z2 adaptivity that refines AND unrefines yields a
    # non-conforming mesh reached partly by coarsening; the mixed C2/C1 hanging Jacobian must stay
    # exact. Oracle: after the adaptive solve, zero all dofs and solve once more -- a linear problem
    # reaches machine zero in ONE Newton step iff the Jacobian on the refined+unrefined mesh is exact.
    with _CavityStokes3D(N=2, error_adapt=True) as problem:
        problem.max_refinement_level = 2
        problem.min_refinement_level = 0
        problem.solve(spatial_adapt=2)
        nel = problem.get_mesh("domain").nelement()
        problem.set_current_dofs(np.zeros(problem.ndof()))
        problem.solve()
        assert _max_abs_residual(problem) < 1e-9
        assert nel > 384


# CROSS-ROUND node-sharing (tetrahedra): the 3d analogue of the triangle mesh-tear bug. Refining to
# level 2 takes two rounds; a son built in round 2 that coincides with a node round 1 already built --
# notably one built by a FINER neighbour's son -- must be REUSED, not duplicated. The per-round
# Shared_edge_node_registry (cleared each round) cannot dedupe cross-round, so without the start-of-
# round position-snapshot fallback the shared node is duplicated (a torn mesh: e.g. 15 coincident-but-
# distinct nodes here). This uses a FIXED mesh (Poisson) so the check is purely topological.
class _TetReadaptShare(Problem):
    def __init__(self, N=3):
        super().__init__()
        self._N = N

    def define_problem(self):
        self += TetCubeMesh(N=self._N)
        eqs = PoissonEquation(source=1, space="C1")
        eqs += DirichletBC(u=0) @ ["bottom", "top", "left", "right", "front", "back"]
        eqs += RefineAccordingToElement(level_func=lambda e: 2 if e.get_Eulerian_midpoint()[2] > 0.45 else 0)
        self += eqs @ "domain"


def test_tet_readapt_no_torn_nodes():
    with _TetReadaptShare(N=3) as p:
        p.max_refinement_level = 2
        p.initial_adaption_steps = 0
        p.solve(spatial_adapt=2)
        m = p.get_mesh("domain")
        seen = {}
        dup = 0
        for n in m.nodes():
            key = (round(n.x(0), 8), round(n.x(1), 8), round(n.x(2), 8))
            if key in seen:
                dup += 1
            seen[key] = n
        assert dup == 0, f"{dup} duplicate (torn) tet nodes across the level-1/level-2 interface"
        assert m.nelement() > 384


def _tet_dup(m):
    seen = set()
    d = 0
    for n in m.nodes():
        k = (round(n.x(0), 8), round(n.x(1), 8), round(n.x(2), 8))
        if k in seen:
            d += 1
        seen.add(k)
    return d


@pytest.mark.parametrize("mode", ["uniform1", "uniform2", "nonuniform"])
def test_tet_c2_manufactured_quadratic(mode):
    # STRICT correctness oracle for C2 (quadratic-field) tetrahedra. The other C2 tet tests here use a
    # residual oracle (source=const, homogeneous Dirichlet), which certifies the hanging Jacobian but -- as
    # the C2 pyramid work showed -- can read machine-zero even on a torn mesh (a duplicated node just adds an
    # independent dof the linear solve still zeroes). This test instead checks the manufactured harmonic-
    # adjacent QUADRATIC u=x^2+2y^2+3z^2 (source -laplace(u)=-12) is reproduced at every node, which fails
    # unless the refined C2 mesh is truly conforming. It confirms the tet edge-node registry key
    # (father_edge_node_key: the father node PAIR a son node bisects) stays unique for C2 -- the father's own
    # edge-mid nodes disambiguate the 1/4 and 3/4 points of an edge, so tets need no weight-augmented key (the
    # pyramid, whose son vertices include non-node tri-face-centres, does).
    x, y, z = var("coordinate")[0], var("coordinate")[1], var("coordinate")[2]
    uex = x * x + 2 * y * y + 3 * z * z

    def _central(e):
        mx = e.get_Eulerian_midpoint()
        return 1 if ((mx[0] - 0.5) ** 2 + (mx[1] - 0.5) ** 2 + (mx[2] - 0.5) ** 2) ** 0.5 < 0.25 else 0

    N = 1 if mode.startswith("uniform") else 3

    class _P(Problem):
        def define_problem(self):
            self += TetCubeMesh(N=N, space="C2")
            eqs = PoissonEquation(source=-12, space="C2") + DirichletBC(u=uex) @ _ALL_BOUNDS
            if mode == "nonuniform":
                eqs += RefineAccordingToElement(level_func=_central)
            self += eqs @ "domain"

    with _P() as p:
        p.max_refinement_level = 1 if mode != "uniform2" else 2
        if mode == "nonuniform":
            p.solve()
        else:
            p.initialise()
            for _ in range(2 if mode == "uniform2" else 1):
                p.refine_uniformly()
            p.solve()
        m = p.get_mesh("domain")
        assert _tet_dup(m) == 0, "duplicate (torn) nodes after C2 tet refinement"
        if mode == "nonuniform":
            assert sum(1 for n in m.nodes() if n.is_hanging()) > 0, "no hanging nodes on the 2:1 interface"
        err = max(abs(n.value(0) - (n.x(0) ** 2 + 2 * n.x(1) ** 2 + 3 * n.x(2) ** 2)) for n in m.nodes())
        assert err < 1e-10, f"C2 tet mesh ({mode}) does not reproduce the quadratic field (max err {err:.2e})"


from pyoomph.equations.ALE import LaplaceSmoothedMesh


class _MovingTetIdentity(Problem):
    # 3d analogue of test_triangle_refinement.py::_MovingMeshIdentity. A moving (LaplaceSmoothed) tet mesh
    # whose boundary positions are all pinned -> the undeformed (identity) mesh is the exact solution, so
    # the residual must be ~machine zero without solving IFF the POSITION hanging installed during
    # refinement is correct. This exercises the refinement path that runs at problem setup
    # (refine_selected_elements / custom_adapt), which BYPASSES the mesh-level post_adapt pass -- the tet
    # analogue of the 2d fix. Before the per-element tet-hang hooks, that path installed no hanging and the
    # residual was O(1); it must now be machine zero.
    def __init__(self, where="domain"):
        super().__init__()
        self._where = where

    def define_problem(self):
        self += TetCubeMesh(N=2)
        eqs = LaplaceSmoothedMesh()
        eqs += PoissonEquation(source=0, space="C2")  # fixes the coordinate space to C2; u=0 is exact
        eqs += DirichletBC(u=0) @ _ALL_BOUNDS
        eqs += DirichletBC(mesh_x=True, mesh_y=True, mesh_z=True) @ _ALL_BOUNDS  # pin boundary positions
        if self._where == "top":
            eqs += RefineToLevel(2) @ "top"  # refine only the top band -> 2:1 hanging interface
        self += eqs @ "domain"
        if self._where == "domain":
            self += RefineToLevel(2) @ "domain"  # uniform (conforming) refinement


@pytest.mark.parametrize("where", ["domain", "top"])
def test_moving_tet_refinement_identity_residual(where):
    with _MovingTetIdentity(where=where) as problem:
        problem.max_refinement_level = 3
        problem.initialise()
        m = problem.get_mesh("domain")
        assert m.nelement() > 48  # refinement actually happened (base = 2^3 * 6 = 48 tets)
        # No solve: the mesh has not moved, so the identity is the exact state; the residual is machine zero
        # only if the position (and field) hanging installed during refinement is exact.
        assert _max_abs_residual(problem) < 1e-10


# NODE IDENTIFICATION MUST BE PURELY TOPOLOGICAL -- the 3d statement of the invariant tested for
# triangles in test_triangle_refinement.py::test_node_sharing_ignores_node_positions. A hanging node's
# stored position is only a CACHE of its masters, so anything writing the dof vector from outside the
# Newton solver leaves it stale; a sharing decision that consults positions then duplicates nodes and
# tears the mesh. Displacing every hanging node before each adapt is exactly what that stale cache
# looks like (and is harmless, since the values are recomputed at the next assembly), so the resulting
# mesh topology must be unchanged. The refinement pattern is keyed on the element's LAGRANGIAN
# midpoint, which the displacement does not touch, so what gets refined cannot change.
class _TetPositionIndependence(Problem):
    def __init__(self, N=3):
        super().__init__()
        self._N = N
        self.threshold = 0.45

    def define_problem(self):
        self += TetCubeMesh(N=self._N)
        eqs = PoissonEquation(source=1, space="C1")
        eqs += DirichletBC(u=0) @ ["bottom", "top", "left", "right", "front", "back"]
        eqs += RefineAccordingToElement(
            level_func=lambda e: 2 if e.get_Lagrangian_midpoint()[2] > self.threshold else 0,
            prevent_unrefinement=False)
        self += eqs @ "domain"


def _tet_readapt_topology(displace_hanging):
    history = []
    ndisplaced = 0
    with _TetPositionIndependence(N=3) as p:
        p.max_refinement_level = 2
        p.initial_adaption_steps = 0
        p.solve(spatial_adapt=2)
        m = p.get_mesh("domain")
        for threshold in (0.3, 0.7, 0.25):
            p.threshold = threshold
            if displace_hanging:
                for n in m.nodes():
                    if n.is_hanging():
                        ndisplaced += 1
                        n.set_x(0, n.x(0) + 0.031)  # >> any element size at these levels
                        n.set_x(1, n.x(1) - 0.027)
                        n.set_x(2, n.x(2) + 0.019)
            p.adapt()
            p.solve()
            history.append((m.nnode(), m.nelement()))
    return history, ndisplaced


def test_tet_node_sharing_ignores_node_positions():
    clean, _ = _tet_readapt_topology(displace_hanging=False)
    stale, ndisplaced = _tet_readapt_topology(displace_hanging=True)
    assert ndisplaced > 0, "no hanging nodes were displaced -- the test would be vacuous"
    assert clean == stale, (
        "adapting with stale hanging-node positions changed the mesh topology, so node "
        f"identification still depends on positions: clean={clean} stale={stale}")
