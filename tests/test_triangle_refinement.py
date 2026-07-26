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
    # space="C1" -> linear (3-node) triangles; "C2" -> quadratic (6-node) triangles.
    def __init__(self, split="left", N=4, space="C1"):
        super().__init__()
        self._split, self._N, self._space = split, N, space

    def define_problem(self):
        self += RectangularQuadMesh(name="domain", N=self._N, split_in_tris=self._split)
        eqs = PoissonEquation(source=1, space=self._space) + DirichletBC(u=0) @ ["left", "right", "top", "bottom"]
        self += eqs @ "domain"


import pytest


@pytest.mark.parametrize("space", ["C1", "C2"])
@pytest.mark.parametrize("split", ["left", "right", "crossed"])
@pytest.mark.parametrize("boundary", ["left", "top"])
def test_nonuniform_triangle_refinement_residual_oracle(space, split, boundary):
    # Refine more strongly near one boundary than the interior -> hanging nodes at the interface.
    # Linear problem: residual ~0 after the Newton step certifies the hanging-node Jacobian
    # (linear weights for C1 edges, quadratic weights for C2 edges).
    with _TriPoissonPlain(split=split, space=space) as problem:
        problem.max_refinement_level = 3
        problem += RefineToLevel(1) @ "domain"
        problem += RefineToLevel(3) @ ("domain/" + boundary)
        problem.solve()
        assert _max_abs_residual(problem) < 1e-9


@pytest.mark.parametrize("space", ["C1", "C2"])
def test_deep_multilevel_triangle_hanging_residual_oracle(space):
    # A large refinement-level jump (1 -> 4) creates hanging nodes whose masters are themselves
    # hanging (non-2:1); the assembly-time flattening must still yield the correct Jacobian.
    with _TriPoissonPlain(split="left", space=space) as problem:
        problem.max_refinement_level = 5
        problem += RefineToLevel(1) @ "domain"
        problem += RefineToLevel(4) @ "domain/left"
        problem.solve()
        assert _max_abs_residual(problem) < 1e-9


class _TriPoissonAdaptive(Problem):
    def __init__(self, space="C1"):
        super().__init__()
        self._space = space

    def define_problem(self):
        self += RectangularQuadMesh(name="domain", N=6, split_in_tris="crossed")
        x = var("coordinate")[0]
        y = var("coordinate")[1]
        eqs = PoissonEquation(source=100 * exp(-50 * ((x - 0.2) ** 2 + (y - 0.2) ** 2)), space=self._space)
        eqs += DirichletBC(u=0) @ ["left", "right", "top", "bottom"]
        eqs += SpatialErrorEstimator(u=1)
        self += eqs @ "domain"


@pytest.mark.parametrize("space", ["C1", "C2"])
def test_error_based_triangle_adaptivity_residual_oracle(space):
    # Genuine error-driven (Z2) adaptive refinement of a triangular mesh around a localized
    # source. The refined mesh is non-uniform (hanging nodes); the linear residual must still
    # reach machine zero.
    with _TriPoissonAdaptive(space=space) as problem:
        problem.max_refinement_level = 4
        problem.solve(spatial_adapt=3)
        assert _max_abs_residual(problem) < 1e-9


def _tesselated_triangles(mesh):
    # (points, triangles) of the tesselate_tri=True numpy export used by MeshFileOutput/pyvista.
    nodal, eleminds = mesh.to_numpy(True, False, 0, False)[:2]
    pts = np.asarray(nodal)[:, :2]
    tris = []
    for row in np.asarray(eleminds):
        valid = [int(i) for i in row if i >= 0]
        if len(valid) == 3:
            tris.append(valid)
    return pts, np.asarray(tris, dtype=int)


def _count_tjunctions(pts, tris):
    # A conforming triangulation has no vertex in the interior of another triangle's edge. Each such
    # hanging vertex on a coarser element's edge that is NOT introduced as a sub-triangle vertex is a
    # visible T-junction crack in the plotted output.
    verts = np.unique(tris.reshape(-1))
    tj, seen = 0, set()
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            A, B = pts[a], pts[b]
            AB = B - A
            L2 = AB @ AB
            if L2 < 1e-20:
                continue
            for p in verts:
                if p == a or p == b:
                    continue
                lam = ((pts[p] - A) @ AB) / L2
                if lam <= 1e-7 or lam >= 1 - 1e-7:
                    continue
                perp = pts[p] - A - lam * AB
                if perp @ perp < 1e-14 * L2:
                    key = (min(a, b), max(a, b), int(p))
                    if key not in seen:
                        seen.add(key)
                        tj += 1
    return tj


def _tess_area(pts, tris):
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    return float(0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])).sum())


@pytest.mark.parametrize("space", ["C1", "C2", "C1TB", "C2TB"])
def test_tesselate_tri_output_no_tjunctions(space):
    # MeshFileOutput(tesselate_tri=True) must split every element (including coarser ones on a
    # hanging/T-junction edge, and under pure-tri MULTI-level hanging where an edge carries several
    # collinear subdivision points) so the plotted triangulation is crack-free and covers the domain
    # exactly. The oracle: zero T-junctions and total sub-triangle area == the unit square.
    with _TriPoissonAdaptive(space=space) as problem:
        problem.max_refinement_level = 4
        problem.solve(spatial_adapt=3)
        pts, tris = _tesselated_triangles(problem.get_mesh("domain"))
        assert len(tris) > 0
        assert _count_tjunctions(pts, tris) == 0, "T-junction cracks in tesselated tri output"
        assert abs(_tess_area(pts, tris) - 1.0) < 1e-9, "tesselated output does not cover the domain exactly"


from pyoomph.equations.ALE import LaplaceSmoothedMesh


@pytest.mark.parametrize("space", ["C2", "C2TB"])
def test_tesselate_tri_curved_geometry_valid(space):
    # Regression for the droplet-evaporation crash: on CURVED (C2/C2TB-geometry) elements the finer-
    # neighbour hanging nodes sit on the BOWED physical edge, off the straight corner chord. A straight-
    # line edge test would reject them, so the coarser element keeps a straight chord that OVERLAPS the
    # finer neighbour's curve-following triangles -- an invalid self-overlapping triangulation that
    # matplotlib's TrapezoidMapTriFinder rejects (as the plotter builds). tess_locate_on_edge places the
    # nodes on the quadratic edge instead. Oracle: consistent triangle orientation (no folds) AND a valid
    # matplotlib triangulation. (Straight-triangle tesselation of VERY strongly curved elements can still
    # fold; this uses a realistic, droplet-scale curvature.)
    mtri = pytest.importorskip("matplotlib.tri")

    class _Curved(Problem):
        def define_problem(self):
            self += RectangularQuadMesh(name="domain", N=6, split_in_tris="crossed")
            X = var("lagrangian")
            eqs = LaplaceSmoothedMesh() + ElementSpace(space)
            eqs += DirichletBC(mesh_x=True, mesh_y=True) @ "bottom"
            eqs += DirichletBC(mesh_x=True) @ ["left", "right"]
            eqs += DirichletBC(mesh_y=1 + 0.05 * sin(pi * X[0])) @ "top"  # smoothly bow the top -> curved C2 geometry
            # multi-level (0/2/3) refinement -> curved elements carrying several hanging nodes per edge
            eqs += RefineAccordingToElement(level_func=lambda e: 3 if e.get_Eulerian_midpoint()[1] > 0.55 else (2 if e.get_Eulerian_midpoint()[0] > 0.5 else 0))
            self += eqs @ "domain"

    with _Curved() as problem:
        problem.max_refinement_level = 3
        problem.solve()
        pts, tris = _tesselated_triangles(problem.get_mesh("domain"))
        assert len(tris) > 0
        a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
        area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
        assert (area > 0).all() or (area < 0).all(), "folded (inverted) sub-triangles in curved tesselation"
        span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1e-30)
        tr = mtri.Triangulation(pts[:, 0] / span, pts[:, 1] / span, tris)
        tr.get_trifinder()  # raises "Triangulation is invalid" on any self-overlap


from pyoomph.equations.navier_stokes import StokesEquations


class _CavityStokes(Problem):
    # Lid-driven cavity Stokes flow with Taylor-Hood (C2 velocity + C1 pressure) triangles. This
    # exercises MIXED continuous spaces on the same mesh: on a C2-coordinate triangle mesh the C1
    # pressure owns a SEPARATE hang slot (hangindex >= 0) and must hang linearly on the coarse edge
    # corners -- even at the coarse edge mid-node, whose velocity is a real dof but whose pressure
    # must hang. If that separate pressure hang is missing, the Jacobian is wrong and Newton no
    # longer converges in one step for the (linear) Stokes problem.
    def __init__(self, split="left"):
        super().__init__()
        self._split = split

    def define_problem(self):
        self += RectangularQuadMesh(name="domain", N=4, split_in_tris=self._split)
        stokes = StokesEquations(mode="TH", dynamic_viscosity=1)
        eqs = stokes
        eqs += DirichletBC(velocity_x=1, velocity_y=0) @ "top"
        eqs += DirichletBC(velocity_x=0, velocity_y=0) @ ["left", "right", "bottom"]
        eqs += stokes.create_pressure_fixation(value=0)
        self += eqs @ "domain"


@pytest.mark.parametrize("split", ["left", "right", "crossed"])
def test_taylor_hood_triangle_cavity_residual_oracle(split):
    # Non-uniform refinement (interior level 1, lid level 3) of a Taylor-Hood triangle mesh. Stokes
    # is linear, so max|residual| ~ 0 after solve certifies that the mixed C2-velocity / C1-pressure
    # hanging-node Jacobian is exact.
    with _CavityStokes(split=split) as problem:
        problem.max_refinement_level = 3
        problem += RefineToLevel(1) @ "domain"
        problem += RefineToLevel(3) @ "domain/top"
        problem.solve()
        assert _max_abs_residual(problem) < 1e-9
        # Non-uniform refinement happened (base is 32 tris for left/right, more for crossed).
        assert problem.get_mesh("domain").nelement() > 150


class _CavityStokesCR(Problem):
    # Lid-driven cavity Stokes with Crouzeix-Raviart (C2TB bubble-enriched velocity + DL
    # discontinuous pressure) triangles. Refinement must build the son bubble (centroid) node and
    # allocate the son's internal (DL) pressure data. The pressure is element-internal (never hangs);
    # the velocity hangs quadratically on edges just like C2 (the bubble vanishes on edges).
    def __init__(self, split="left", N=4):
        super().__init__()
        self._split = split
        self._N = N

    def define_problem(self):
        self += RectangularQuadMesh(name="domain", N=self._N, split_in_tris=self._split)
        stokes = StokesEquations(mode="CR", dynamic_viscosity=1)
        eqs = stokes
        eqs += DirichletBC(velocity_x=1, velocity_y=0) @ "top"
        eqs += DirichletBC(velocity_x=0, velocity_y=0) @ ["left", "right", "bottom"]
        eqs += stokes.create_pressure_fixation(value=0)
        eqs += SpatialErrorEstimator(velocity=1)
        self += eqs @ "domain"


@pytest.mark.parametrize("split", ["left", "right", "crossed"])
def test_crouzeix_raviart_triangle_cavity_single_level(split):
    # Single-level (2:1) non-conforming refinement of a CR mesh. Linear Stokes -> the residual after
    # solve certifies the bubble-enriched velocity hanging Jacobian + the son DL pressure allocation.
    # CR is more poorly conditioned than TH, so the machine-zero tolerance is looser (~1e-7).
    with _CavityStokesCR(split=split) as problem:
        problem.max_refinement_level = 3
        problem += RefineToLevel(1) @ "domain"
        problem += RefineToLevel(2) @ "domain/top"
        problem.solve()
        assert _max_abs_residual(problem) < 1e-7
        assert problem.get_mesh("domain").nelement() > 150


@pytest.mark.parametrize("split", ["left", "crossed"])
def test_crouzeix_raviart_triangle_error_adaptivity(split):
    # Genuine Z2 error-driven adaptivity (the mesh adaptivity actually produces: 2:1-balanced,
    # non-uniform). CR bubble refinement must keep the linear Stokes residual near machine zero.
    with _CavityStokesCR(split=split) as problem:
        problem.max_refinement_level = 3
        problem.solve(spatial_adapt=2)
        assert _max_abs_residual(problem) < 1e-7
        assert problem.get_mesh("domain").nelement() > 150


def _reset_dofs_to_zero(problem):
    problem.set_current_dofs(np.zeros(problem.ndof()))


# UNREFINEMENT (coarsening). Z2 error-driven adaptivity that both refines AND unrefines produces a
# non-conforming mesh reached partly by coarsening. On such a mesh a mixed C2/C1 (Taylor-Hood) or
# C2TB/DL (Crouzeix-Raviart) field can develop hanging-on-hanging chains in the C1-pressure value
# slot (and former-corner nodes left as stale-pressure edge-mids). The oracle: after the adaptive
# solve, zero all dofs and solve once more -- a linear problem reaches machine zero in ONE Newton
# step from any start iff the hanging Jacobian on the (refined+unrefined) mesh is exact.
@pytest.mark.parametrize("mode,tol", [("TH", 1e-9), ("CR", 1e-6)])
def test_stokes_triangle_unrefinement_residual_oracle(mode, tol):
    class _Cav(Problem):
        def define_problem(self):
            self += RectangularQuadMesh(name="domain", N=4, split_in_tris="left")
            st = StokesEquations(mode=mode, dynamic_viscosity=1)
            eqs = st + DirichletBC(velocity_x=1, velocity_y=0) @ "top"
            eqs += DirichletBC(velocity_x=0, velocity_y=0) @ ["left", "right", "bottom"]
            eqs += st.create_pressure_fixation(value=0)
            eqs += SpatialErrorEstimator(velocity=1)
            self += eqs @ "domain"

    with _Cav() as problem:
        problem.max_refinement_level = 3
        problem.min_refinement_level = 0  # allow coarsening below the initial level
        problem.solve(spatial_adapt=3)    # several cycles -> refines near the lid, unrefines interior
        nel = problem.get_mesh("domain").nelement()
        _reset_dofs_to_zero(problem)
        problem.solve()
        assert _max_abs_residual(problem) < tol
        assert nel > 150  # genuine (non-uniform) adaptation happened


from pyoomph.equations.ALE import LaplaceSmoothedMesh


# MOVING MESH (LaplaceSmoothedMesh): refined SolidNodes must inherit their Lagrangian (reference)
# coordinates, interpolated from the father element. Historic bug: RefineableTElement<2>/<3>::build
# set every new node's Eulerian position x but never its Lagrangian xi, so refined tri/tet nodes
# kept xi=0 while x sat at the correct geometric midpoint. Mesh-smoothing/solid residuals are written
# in terms of the deformation x - xi, so on the IDENTITY mesh (which has not moved) the residual was
# grossly nonzero -- O(10) even for CONFORMING uniform refinement with zero hanging nodes. With all
# boundary positions pinned, identity is the exact solution, so the residual right after refinement
# (no solve) must be ~machine zero for ANY refinement pattern and any element shape.
class _MovingMeshIdentity(Problem):
    def __init__(self, split="left", refine_where="domain"):
        super().__init__()
        self._split, self._where = split, refine_where

    def define_problem(self):
        self += RectangularQuadMesh(name="domain", N=4, split_in_tris=self._split)
        eqs = LaplaceSmoothedMesh()
        eqs += PoissonEquation(source=0, space="C2")  # fixes the coordinate space to C2; u=0 is exact
        eqs += DirichletBC(u=0) @ ["left", "right", "top", "bottom"]
        # Pin ALL boundary positions -> the undeformed (identity) mesh is the exact solution.
        eqs += DirichletBC(mesh_x=True, mesh_y=True) @ ["left", "right", "top", "bottom"]
        if self._where == "top":
            eqs += RefineToLevel(2) @ "top"  # refine only the top band -> 2:1 hanging interface
        self += eqs @ "domain"
        if self._where == "domain":
            self += RefineToLevel(2) @ "domain"  # uniform (conforming) refinement of the whole mesh


@pytest.mark.parametrize("split", [False, "left", "crossed"])  # False -> quads (regression baseline)
@pytest.mark.parametrize("where", ["domain", "top"])  # "domain" -> conforming uniform, "top" -> 2:1 hanging
def test_moving_mesh_refinement_identity_residual(split, where):
    base = {False: 16, "left": 32, "crossed": 64}[split]
    with _MovingMeshIdentity(split=split, refine_where=where) as problem:
        problem.max_refinement_level = 3
        problem.initialise()
        m = problem.get_mesh("domain")
        assert m.nelement() > base  # refinement actually happened
        # No solve: the mesh has not moved, so identity is exactly the current state. Pre-fix this
        # residual was O(10) for tris (any refinement); it must now be ~machine zero.
        assert _max_abs_residual(problem) < 1e-10


# TRANSIENT RE-ADAPTATION node-sharing (moving mesh): a LaplaceSmoothedMesh whose refinement level
# follows a spatial threshold. As the mesh moves, elements repeatedly cross the threshold, forcing
# refine AND unrefine each step. A son node built in one round that coincides with a node an EARLIER
# round already built -- notably one created by a FINER neighbour's son (which the >=-sized tree
# neighbour search cannot reach) -- must be REUSED, not duplicated. Otherwise the shared vertex splits
# into two coincident nodes and the moving mesh tears apart at the refine/coarsen interface. This is a
# regression for RefineableTElement<2>::build's node-sharing (node_created_by_neighbour + the
# start-of-round position snapshot fallback).
class _MovingReadaptTear(Problem):
    def __init__(self, N=4):
        super().__init__()
        self._N = N

    def define_problem(self):
        self += RectangularQuadMesh(split_in_tris="left", N=self._N)
        eqs = LaplaceSmoothedMesh() + ElementSpace("C1")
        eqs += DirichletBC(mesh_x=True, mesh_y=0) @ "bottom"
        eqs += NeumannBC(mesh_y=-var("time")) @ "top"
        eqs += RefineAccordingToElement(level_func=lambda e: 2 if e.get_Eulerian_midpoint()[1] > 0.5 else 0)
        self += eqs @ "domain"


def _count_coincident_nodes(mesh):
    seen = {}
    dup = 0
    for n in mesh.nodes():
        key = (round(n.x(0), 8), round(n.x(1), 8))
        if key in seen:
            dup += 1
        seen[key] = n
    return dup


def test_moving_mesh_readapt_no_torn_nodes():
    with _MovingReadaptTear(N=4) as p:
        p.max_refinement_level = 2
        p.initial_adaption_steps = 0
        p.solve()  # t=0
        assert _count_coincident_nodes(p.get_mesh("domain")) == 0
        # Step through time with re-adaptation; the tear appeared once coarsening kicked in (t~0.5).
        for _ in range(8):
            p.solve(timestep=0.1, spatial_adapt=2)
            assert _count_coincident_nodes(p.get_mesh("domain")) == 0, "duplicate (torn) shared nodes after re-adaptation"
