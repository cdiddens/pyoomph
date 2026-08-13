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
        # This system is right at the edge of what Pardiso's static pivoting copes with: the tet-CR
        # saddle point on an error-adapted mesh. It used to land on the good side by luck, and the
        # tetrahedron-winding repair in add_tetra_3d_C1 (which renumbers the nodes, and nothing else)
        # moved it to the other side - Pardiso then reports backward errors of order 1e+2 and the
        # Newton solver gives up. Nothing is wrong with the Jacobian, which is what this test is
        # about: umfpack and superlu both reach 6.7e-14 on exactly the same 2911-element mesh, and so
        # does Pardiso once it is allowed to pivot harder.
        problem.get_la_solver().repair_bad_solves = True
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


# ----------------------------------------------------------------------------------------------
# Tetrahedron handedness
# ----------------------------------------------------------------------------------------------
#
# oomph-lib's TElement<3,NNODE_1D> bases its local frame at node 3, with the s0/s1/s2 axes pointing
# at nodes 0/1/2 -- so the winding that decides the face-normal direction is
# det(p0-p3, p1-p3, p2-p3), not the det(p1-p0, p2-p0, p3-p0) that most mesh generators (and the
# _TETS_LOCAL table above, before this was found) use. Getting it wrong is invisible almost
# everywhere: the integration measure uses |J|, so volumes and every mass/stiffness matrix stay
# right, and only the outward normal flips. That silently reverses every Neumann/Robin flux and
# makes interior-penalty DG inconsistent -- for years the symptom was read as "tets need a bigger
# DG_alpha to be coercive" (see tests/test_internal_facet_fields.py), which they do not.
#
# add_tetra_3d_C1/C2 therefore accept either winding and repair a left-handed one. The oracle here
# is the divergence theorem, which needs no knowledge of the convention at all: over a closed
# surface, int(x.n) dA = 3*Volume if and only if n points outwards.

_TET_PTS = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
_TET_FACES = [("f0", (1, 2, 3)), ("f1", (0, 2, 3)), ("f2", (0, 1, 3)), ("f3", (0, 1, 2))]


class _OneTet(MeshTemplate):
    """A single tetrahedron, with all four faces on named boundaries.

    `order` permutes the four vertices before they are handed to add_tetra_3d_C1, which is how the
    two windings are produced from the same geometry."""

    def __init__(self, order=(0, 1, 2, 3)):
        super().__init__()
        self.order = order

    def define_geometry(self):
        dom = self.new_domain("domain")
        n = [self.add_node_unique(*_TET_PTS[i]) for i in self.order]
        dom.add_tetra_3d_C1(*n)
        for name, f in _TET_FACES:
            self.add_facet_to_boundary(name, [n[i] for i in f])


class _NormalFlux(Equations):
    def define_residuals(self):
        x = vector(var("coordinate_x"), var("coordinate_y"), var("coordinate_z"))
        self.add_integral_function("xn", dot(x, var("normal")) * self.get_dx())


class _Volume(Equations):
    def define_fields(self):
        self.define_scalar_field("u", "C1")

    def define_residuals(self):
        u, ut = var_and_test("u")
        self.add_residual(weak(u - 1, ut))
        self.add_integral_function("vol", 1 * self.get_dx())


class _OneTetProblem(Problem):
    def __init__(self, order):
        super().__init__()
        self.order = order

    def define_problem(self):
        self += _OneTet(self.order)
        eqs = _Volume()
        for name, _f in _TET_FACES:
            eqs += _NormalFlux() @ name
        self += eqs @ "domain"
        self.max_refinement_level = 0


def _closed_surface_flux(problem, boundaries):
    return sum(float(problem.get_mesh("domain/" + b).evaluate_all_observables()["xn"])
               for b in boundaries)


@pytest.mark.parametrize("order", [(0, 1, 2, 3), (0, 2, 1, 3)])
def test_tetrahedron_face_normals_point_outwards_for_either_winding(tmp_path, monkeypatch, order):
    monkeypatch.chdir(tmp_path)
    with _OneTetProblem(order) as p:
        p.quiet()
        p.initialise()
        vol = float(p.get_mesh("domain").evaluate_all_observables()["vol"])
        flux = _closed_surface_flux(p, [n for n, _f in _TET_FACES])
    assert vol == pytest.approx(1.0 / 6.0)
    assert flux == pytest.approx(3 * vol), (
        "int(x.n) dA = %.6g over the closed surface of a tetrahedron given in vertex order %s, "
        "but 3*Volume = %.6g -- the face normals point inwards" % (flux, order, 3 * vol))


def _tet_handedness(mesh):
    """(#right-handed, #left-handed) in oomph's sense, over the tetrahedra of a mesh."""
    right = left = 0
    for i in range(mesh.nelement()):
        e = mesh.element_pt(i)
        if e.nnode() not in (4, 10):
            continue
        x = np.array([[e.node_pt(j).x(k) for k in range(3)] for j in range(4)])
        det = np.linalg.det(np.vstack([x[0] - x[3], x[1] - x[3], x[2] - x[3]]))
        if det < 0:
            left += 1
        else:
            right += 1
    return right, left


def test_tet_cube_mesh_is_wound_right_handed(tmp_path, monkeypatch):
    """The mesh helper above states its tets in the other convention; the repair has to catch it.

    Without this, everything in this file still passes -- refinement, hanging nodes and the Poisson
    solves are all blind to the winding -- which is exactly why it went unnoticed."""
    monkeypatch.chdir(tmp_path)

    class _P(Problem):
        def define_problem(self):
            self += TetCubeMesh(N=2)
            self += (PoissonEquation(source=1) + DirichletBC(u=0) @ _ALL_BOUNDS) @ "domain"
            self.max_refinement_level = 0

    with _P() as p:
        p.quiet()
        p.initialise()
        right, left = _tet_handedness(p.get_mesh("domain"))
    assert right + left == 6 * 8, "expected 48 tetrahedra, got %d" % (right + left)
    assert left == 0, "%d of %d tetrahedra are left-handed, so their face normals point inwards" % (
        left, right + left)


# The 10-node case has a second thing to get right: the repair swaps two vertices, and the six
# mid-side nodes have to travel with them. TElementShape<3,3> places the mid-node of edge (i,j) at
# index (0,1)->4 (0,2)->5 (0,3)->6 (1,2)->7 (2,3)->8 (1,3)->9, so a 1<->2 vertex swap exchanges 4<->5
# and 8<->9 while 6 and 7 stay put. A wrong entry there puts a mid-node on the wrong edge, which
# curves the element rather than merely reorienting it - so the volume moves too, and both assertions
# below are needed to tell the two failures apart.
_TET_EDGES_C2 = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (1, 3)]
_TET_FACES_C2 = [("f0", (1, 2, 3), (7, 8, 9)), ("f1", (0, 2, 3), (5, 8, 6)),
                 ("f2", (0, 1, 3), (4, 9, 6)), ("f3", (0, 1, 2), (4, 7, 5))]


class _OneTetC2(MeshTemplate):
    def __init__(self, order=(0, 1, 2, 3)):
        super().__init__()
        self.order = order

    def define_geometry(self):
        dom = self.new_domain("domain")
        v = [np.array(_TET_PTS[i], dtype=float) for i in self.order]
        n = [self.add_node_unique(*p) for p in v]
        n += [self.add_node_unique(*(0.5 * (v[a] + v[b]))) for a, b in _TET_EDGES_C2]
        dom.add_tetra_3d_C2(n)
        for name, verts, mids in _TET_FACES_C2:
            face = [n[i] for i in verts] + [n[i] for i in mids]
            self.add_facet_to_boundary(name, face, [n[i] for i in verts])


class _OneTetC2Problem(Problem):
    def __init__(self, order):
        super().__init__()
        self.order = order

    def define_problem(self):
        self += _OneTetC2(self.order)
        eqs = _Volume()
        for name, _v, _m in _TET_FACES_C2:
            eqs += _NormalFlux() @ name
        self += eqs @ "domain"
        self.max_refinement_level = 0


@pytest.mark.parametrize("order", [(0, 1, 2, 3), (0, 2, 1, 3)])
def test_quadratic_tetrahedron_face_normals_point_outwards_for_either_winding(tmp_path, monkeypatch,
                                                                             order):
    monkeypatch.chdir(tmp_path)
    with _OneTetC2Problem(order) as p:
        p.quiet()
        p.initialise()
        vol = float(p.get_mesh("domain").evaluate_all_observables()["vol"])
        flux = _closed_surface_flux(p, [n for n, _v, _m in _TET_FACES_C2])
    assert vol == pytest.approx(1.0 / 6.0), (
        "volume %.6g instead of 1/6 for vertex order %s -- a mid-side node landed on the wrong edge"
        % (vol, order))
    assert flux == pytest.approx(3 * vol), (
        "int(x.n) dA = %.6g over the closed surface of a 10-node tetrahedron given in vertex order "
        "%s, but 3*Volume = %.6g -- the face normals point inwards" % (flux, order, 3 * vol))
