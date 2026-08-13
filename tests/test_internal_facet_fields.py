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

# Fields (unknowns) living on the interior-facet skeleton, i.e. on the reserved "_internal_facets_"
# subdomain of a bulk mesh:
#
#     eqs += MyFacetEquations() @ "_internal_facets_"     # define_scalar_field("lam", "DL") inside
#
# Before this, the skeleton could only carry residuals (the DG jump/penalty terms added through
# add_interior_facet_residual); a facet unknown -- an HDG trace, a facet Lagrange multiplier -- had
# no way of being declared. Only DISCONTINUOUS facet spaces are supported: those live in the facet
# element's own internal oomph::Data, whereas a continuous one would need shared per-node dofs, which
# only exist on BoundaryNodes (interior nodes are plain pyoomph::Node, so the old code path
# segfaulted). Two of the tests below pin exactly that error message down.
#
# The assertions are chosen so that a silent failure cannot pass:
#
#   * A LINEAR bulk field is represented exactly, so its trace on a facet is representable exactly by
#     DL/D1/D2 as well. The L2 projection of it onto the skeleton must therefore be machine-exact --
#     any wrongly wired shape function, node ordering or Data slot shows up immediately, where a
#     smooth field would only give "small".
#   * D0 cannot represent a linear trace, but its projection still reproduces the facet MEAN exactly:
#     the signed error integrates to zero while the squared error does not.
#   * dof COUNTING against facet_adjacency_summary(): the facet fields must be numbered once per
#     facet, not once per facet element side, and the never-assembled opposite dummy element must
#     contribute nothing at all.
#   * A hybridised (mortar) Poisson problem, where the skeleton multiplier is what glues a fully
#     discontinuous bulk field together. With a linear manufactured solution the answer is exact, so
#     the multiplier is genuinely load bearing rather than a passive observer.
#
# Spatial adaptivity has its own section at the end of this file (DL/D0 are carried across, the nodal
# Dx spaces are not). What stays out of scope - continuous facet spaces, MPI, non-conforming 3d and
# non-uniformly adapted triangles - must fail with an explanatory error rather than a crash, which is
# what the "rejected configurations" tests keep honest.

import math
import os

import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.meshes.gmsh import GmshTemplate
from pyoomph.meshes.interpolator import InternalInterpolator, ProjectionInternalInterpolator
from pyoomph.meshes.remesher import RemesherViaRecreation
from pyoomph.meshes.simplemeshes import RectangularQuadMesh, LineMesh


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------

def _mesh_for(kind, N):
    if kind == "quad":
        return RectangularQuadMesh(name="domain", N=N)
    if kind == "tri":
        return RectangularQuadMesh(name="domain", N=N, split_in_tris="crossed")
    if kind == "line":
        return LineMesh(name="domain", N=N)
    raise ValueError(kind)


def _linear_expr(dim):
    """A field an isoparametric space reproduces exactly, so every error below is transfer error."""
    f = 1 + 2 * var("coordinate_x")
    if dim > 1:
        f = f + 3 * var("coordinate_y")
    return f


class _LinearBulk(Equations):
    """u := the exact interpolant of a linear function (an algebraic definition, so no BCs needed)."""

    def __init__(self, space="C2"):
        super().__init__()
        self.space = space

    def define_fields(self):
        self.define_scalar_field("u", self.space)

    def define_residuals(self):
        u, ut = var_and_test("u")
        self.add_residual(weak(u - _linear_expr(self.get_nodal_dimension()), ut))


class _FacetTrace(Equations):
    """L2 projection of the bulk trace onto a facet field: weak(p - avg(u), ptest)."""

    def __init__(self, space="DL", name="p"):
        super().__init__()
        self.space = space
        self.name = name

    def define_fields(self):
        self.define_scalar_field(self.name, self.space)

    def define_residuals(self):
        p, ptest = var_and_test(self.name)
        u = avg(var("u"))
        self.add_residual(weak(p - u, ptest))
        self.add_integral_function("err2", (p - u) ** 2 * self.get_dx())
        self.add_integral_function("err1", (p - u) * self.get_dx())
        self.add_integral_function("meas", 1 * self.get_dx())


class _TraceProblem(Problem):
    def __init__(self, kind="quad", N=3, space="DL", extra=None):
        super().__init__()
        self.kind, self.N, self.space, self.extra = kind, N, space, extra

    def define_problem(self):
        self += _mesh_for(self.kind, self.N)
        eqs = _LinearBulk()
        if self.space is not None:
            feqs = _FacetTrace(space=self.space)
            if self.extra is not None:
                feqs += self.extra
            eqs += feqs @ "_internal_facets_"
        self += eqs @ "domain"
        # Nodal Dx facet spaces cannot be snapshotted, so rebuild_after_adapt refuses to carry them
        # through an adaptation. Nothing in this group needs adaptivity, and the initial adaption
        # would otherwise trip that guard for the Dx parametrisations.
        self.max_refinement_level = 0


@pytest.fixture(autouse=True)
def _quiet_and_local(tmp_path, monkeypatch):
    """Run each problem in its own tmp dir: every Problem writes an output directory next to the
    script, and several of these tests would otherwise reuse (and read back) each other's dumps."""
    monkeypatch.chdir(tmp_path)


def _observables(p, dom="domain/_internal_facets_"):
    return {k: float(v) for k, v in p.get_mesh(dom).evaluate_all_observables().items()}


# ----------------------------------------------------------------------------------------------
# trace projection
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind,space", [
    ("quad", "DL"), ("quad", "D1"), ("quad", "D2"),
    ("tri", "DL"), ("tri", "D1"), ("tri", "D2"),
    ("line", "DL"), ("line", "D0"),
])
def test_trace_projection_is_exact(kind, space):
    """A linear trace lies in every one of these facet spaces, so the projection must be exact.
    (On a 1D bulk a facet is a single point, where D0 is exact as well.)"""
    with _TraceProblem(kind=kind, space=space) as p:
        p.quiet()
        p.initialise()
        p.solve()
        obs = _observables(p)
    assert obs["meas"] > 0
    # both integrands vanish identically, so what is left is cancellation round-off
    assert abs(obs["err2"]) < 1e-12, obs
    assert abs(obs["err1"]) < 1e-11, obs


def test_d0_projection_reproduces_the_facet_mean():
    """D0 cannot follow a linear trace, but the Galerkin projection still matches its mean exactly:
    the signed error vanishes while the squared one does not (which is what makes this test sharp -
    a facet field stuck at zero would satisfy neither)."""
    with _TraceProblem(kind="quad", space="D0") as p:
        p.quiet()
        p.initialise()
        p.solve()
        obs = _observables(p)
    assert abs(obs["err1"]) < 1e-11, obs
    assert obs["err2"] > 1e-3, obs


# ----------------------------------------------------------------------------------------------
# dof accounting
# ----------------------------------------------------------------------------------------------

# Number of values a facet element of the given space owns, for a 1d facet (2d bulk) and for a 0d
# facet (1d bulk). DL carries one constant plus one gradient mode per facet direction.
_MODES_2D = {"D0": 1, "DL": 2, "D1": 2, "D2": 3}
_MODES_1D = {"D0": 1, "DL": 1, "D1": 1, "D2": 1}


@pytest.mark.parametrize("kind,space", [
    ("quad", "D0"), ("quad", "DL"), ("quad", "D1"), ("quad", "D2"),
    ("tri", "D0"), ("tri", "DL"), ("tri", "D1"), ("tri", "D2"),
    ("line", "D0"), ("line", "DL"),
])
def test_ndof_counts_one_set_of_facet_dofs_per_interior_facet(kind, space):
    """The opposite "dummy" facet element allocates the same internal Data as the real one but is
    never part of a mesh; if it were numbered (or left unpinned and picked up anyway) the count would
    come out too high."""
    with _TraceProblem(kind=kind, space=None) as p:
        p.quiet()
        p.initialise()
        ndof_ref = p.ndof()
    with _TraceProblem(kind=kind, space=space) as p:
        p.quiet()
        p.initialise()
        ndof = p.ndof()
        _n_facets, _n_bnd, n_int, _max_inc = p.get_mesh("domain").facet_adjacency_summary()
    modes = (_MODES_1D if kind == "line" else _MODES_2D)[space]
    assert ndof - ndof_ref == int(n_int) * modes


# ----------------------------------------------------------------------------------------------
# a facet unknown that actually carries the physics
# ----------------------------------------------------------------------------------------------

class _BrokenPoisson(Equations):
    """-u'' = f with a FULLY discontinuous u: without the facet multiplier below this is a pile of
    unconnected element-local problems."""

    def __init__(self, source, exact):
        super().__init__()
        self.source, self.exact = source, exact

    def define_fields(self):
        self.define_scalar_field("u", "D1")

    def define_residuals(self):
        u, ut = var_and_test("u")
        self.add_residual(weak(grad(u), grad(ut)) - weak(self.source, ut))
        self.add_integral_function("uerr2", (u - self.exact) ** 2 * self.get_dx())


class _MortarGlue(Equations):
    """The hybridised (mortar) coupling: lam is the interior-facet Lagrange multiplier enforcing
    jump(u)=0, and it enters the bulk equations as the interelement flux."""

    def define_fields(self):
        self.define_scalar_field("lam", "D0")

    def define_residuals(self):
        lam, lamtest = var_and_test("lam")
        u, ut = var("u"), testfunction("u")
        self.add_residual(weak(lam, jump(ut)) + weak(jump(u), lamtest))
        self.add_integral_function("jump2", jump(u) ** 2 * self.get_dx())


class _MortarPoisson1d(Problem):
    """1D on purpose: there the multiplier count matches the number of independent continuity
    constraints exactly. In 2D the same formulation is rank deficient, because the facets meeting at a
    vertex each re-impose continuity there - a property of mortar methods, not of the facet fields."""

    def __init__(self, N, source, exact):
        super().__init__()
        self.N, self.source, self.exact = N, source, exact

    def define_problem(self):
        self += LineMesh(name="domain", N=self.N)
        eqs = _BrokenPoisson(self.source, self.exact)
        eqs += DirichletBC(u=self.exact) @ "left"
        eqs += DirichletBC(u=self.exact) @ "right"
        eqs += _MortarGlue() @ "_internal_facets_"
        self += eqs @ "domain"
        self.max_refinement_level = 0


def _solve_mortar(N, source, exact):
    with _MortarPoisson1d(N, source, exact) as p:
        p.quiet()
        p.initialise()
        p.solve()
        return _observables(p), _observables(p, "domain")


@pytest.mark.parametrize("N", [4, 7])
def test_mortar_poisson_is_exact_for_a_linear_solution(N):
    """The multiplier glues the discontinuous D1 field back together, so the discrete problem is the
    continuous P1 Galerkin one; a linear manufactured solution lies in that space and is reproduced
    exactly, at any resolution. If lam were not assembled (or were read off the never-numbered
    opposite dummy) u would fall apart into element-local pieces and the jump would not vanish."""
    x = var("coordinate_x")
    facet_obs, bulk_obs = _solve_mortar(N, 0, 1 + 2 * x)
    assert abs(facet_obs["jump2"]) < 1e-14, facet_obs
    assert abs(bulk_obs["uerr2"]) < 1e-16, bulk_obs


def test_mortar_poisson_converges_at_second_order():
    """Manufactured solution sin(pi x): the glued scheme is continuous P1, so the L2 error must fall
    by 4 per refinement - i.e. the squared error reported here by 16."""
    x = var("coordinate_x")
    exact, source = sin(pi * x), pi ** 2 * sin(pi * x)
    errs = []
    for N in (8, 16, 32):
        facet_obs, bulk_obs = _solve_mortar(N, source, exact)
        assert abs(facet_obs["jump2"]) < 1e-14, facet_obs
        errs.append(bulk_obs["uerr2"])
    for coarse, fine in zip(errs[:-1], errs[1:]):
        assert 12 < coarse / fine < 20, errs


# ----------------------------------------------------------------------------------------------
# a facet unknown that writes back into a continuous bulk field
# ----------------------------------------------------------------------------------------------

class _FacetFeedback(Equations):
    """lam is the (exact, since C1 traces are linear on a straight facet) trace of u, and it is fed
    back into the bulk equation. At the solution the feedback term vanishes identically, so u is the
    plain C1 Poisson solution - but only if lam really is the trace: a lam stuck at zero, or read from
    the opposite dummy, turns the term into a facet mass term and moves u."""

    def define_fields(self):
        self.define_scalar_field("lam", "DL")

    def define_residuals(self):
        lam, lamtest = var_and_test("lam")
        u, ut = avg(var("u")), avg(testfunction("u"))
        self.add_residual(weak(lam - u, lamtest) + 10 * weak(lam - u, ut))
        self.add_integral_function("lerr2", (lam - u) ** 2 * self.get_dx())


class _C1Poisson(Equations):
    def define_fields(self):
        self.define_scalar_field("u", "C1")

    def define_residuals(self):
        u, ut = var_and_test("u")
        self.add_residual(weak(grad(u), grad(ut)))
        self.add_integral_function("uerr2", (u - _linear_expr(2)) ** 2 * self.get_dx())


class _FeedbackPoisson(Problem):
    def __init__(self, N=3):
        super().__init__()
        self.N = N

    def define_problem(self):
        self += RectangularQuadMesh(name="domain", N=self.N)
        eqs = _C1Poisson()
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=_linear_expr(2)) @ b
        eqs += _FacetFeedback() @ "_internal_facets_"
        self += eqs @ "domain"
        self.max_refinement_level = 0


def test_facet_field_feeding_back_into_a_continuous_bulk_field():
    """The other coupling direction, in 2D: a residual assembled on the skeleton that writes into the
    bulk field's equations. (This is also what uncovered the null BoundaryNode dereference in
    InterfaceElementBase::setup_additional_dof_constraints, which fires as soon as the bulk carries a
    C1 field.)"""
    with _FeedbackPoisson(N=3) as p:
        p.quiet()
        p.initialise()
        p.solve()
        facet_obs = _observables(p)
        bulk_obs = _observables(p, "domain")
    assert abs(facet_obs["lerr2"]) < 1e-14, facet_obs
    # squared L2 error, so 1e-12 here is an L2 error of 1e-6 - the direct solver's own residual on
    # this saddle-point-flavoured system, not a discretisation error (the linear solution is exact)
    assert abs(bulk_obs["uerr2"]) < 1e-12, bulk_obs


# ----------------------------------------------------------------------------------------------
# skeleton field alongside the pre-existing DG jump residuals
# ----------------------------------------------------------------------------------------------

class _FacetAverage(Equations):
    def define_fields(self):
        self.define_scalar_field("lam", "DL")

    def define_residuals(self):
        lam, lamtest = var_and_test("lam")
        self.add_residual(weak(lam - avg(var("u")), lamtest))
        self.add_integral_function("err2", (lam - avg(var("u"))) ** 2 * self.get_dx())
        self.add_integral_function("jump2", jump(var("u")) ** 2 * self.get_dx())


class _SIPWithFacetField(Problem):
    def define_problem(self):
        self += RectangularQuadMesh(name="domain", N=4)
        eqs = PoissonEquation(source=1, space="D2")
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        eqs += _FacetAverage() @ "_internal_facets_"
        self += eqs @ "domain"
        self.max_refinement_level = 0


def test_skeleton_field_coexists_with_DG_jump_residuals():
    """The user-supplied "_internal_facets_" child must suppress the auto-created DummyEquations and
    still receive the bulk equations' interior-facet residuals. This is also the case that exercises
    the dummy opposite element's equation maps while it owns pinned facet Data of its own."""
    with _SIPWithFacetField() as p:
        p.quiet()
        p.initialise()
        p.solve()
        obs = _observables(p)
    assert obs["jump2"] < 1e-6, obs           # SIP keeps the D2 solution nearly continuous
    assert obs["err2"] < 1e-4, obs            # lam follows the (smooth) average trace


# ----------------------------------------------------------------------------------------------
# initial conditions, Dirichlet pinning, output
# ----------------------------------------------------------------------------------------------

def test_initial_condition_on_a_skeleton_field():
    with _TraceProblem(kind="quad", space="DL", extra=InitialCondition(p=7)) as p:
        p.quiet()
        p.initialise()
        obs = _observables(p)
        # (p-avg(u))^2 integrated over the skeleton, with p==7 everywhere
        assert obs["err2"] > 1, obs
        p.solve()
        assert abs(_observables(p)["err2"]) < 1e-12


def test_dirichlet_condition_pins_a_skeleton_field_completely():
    """A DirichletBC on a DL field has to pin the gradient modes as well, not only the constant one -
    otherwise the residual is free to pick a slope and the condition is not imposed at all."""
    with _TraceProblem(kind="quad", space="DL", extra=None) as p:
        p.quiet()
        p.initialise()
        ndof_free = p.ndof()
        _n_facets, _n_bnd, n_int, _m = p.get_mesh("domain").facet_adjacency_summary()
    with _TraceProblem(kind="quad", space="DL", extra=DirichletBC(p=0)) as p:
        p.quiet()
        p.initialise()
        ndof_pinned = p.ndof()
        p.solve()
        obs = _observables(p)
    assert ndof_free - ndof_pinned == int(n_int) * _MODES_2D["DL"]
    # p is identically zero, so the projection error is the norm of the trace itself
    assert obs["err2"] > 1


def test_mesh_file_output_of_the_skeleton_domain(tmp_path):
    with _TraceProblem(kind="quad", space="DL", extra=MeshFileOutput()) as p:
        p.quiet()
        p.initialise()
        p.solve()
        p.output()
        # MeshFileOutput writes <outdir>/<domain-with-__-separators>/*.vtu plus a .pvd next to it
        skeldir = os.path.join(p.get_output_directory(), "domain___internal_facets_")
        assert os.path.isdir(skeldir), sorted(os.listdir(p.get_output_directory()))
        written = [f for f in os.listdir(skeldir) if f.endswith(".vtu")]
    assert written


# ----------------------------------------------------------------------------------------------
# rejected configurations
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("space", ["C1", "C2"])
def test_continuous_skeleton_field_is_rejected(space):
    """Interior nodes are not BoundaryNodes, so a shared per-node facet dof cannot be allocated at
    all - this used to segfault in InterfaceElementBase::add_interface_dofs."""
    with pytest.raises(NotImplementedError, match="Continuous fields on the interior-facet skeleton"):
        with _TraceProblem(kind="quad", space=space) as p:
            p.quiet()
            p.initialise()


class _OppositeAccess(Equations):
    def __init__(self, at_facet):
        super().__init__()
        self.at_facet = at_facet

    def define_fields(self):
        self.define_scalar_field("p", "DL")

    def define_residuals(self):
        p, ptest = var_and_test("p")
        other = jump(var("p"), at_facet=True) if self.at_facet else var("p", domain="|-")
        self.add_residual(weak(p - other, ptest))


class _OppositeProblem(Problem):
    def __init__(self, at_facet):
        super().__init__()
        self.at_facet = at_facet

    def define_problem(self):
        self += RectangularQuadMesh(name="domain", N=3)
        eqs = _LinearBulk()
        eqs += _OppositeAccess(self.at_facet) @ "_internal_facets_"
        self += eqs @ "domain"
        self.max_refinement_level = 0


@pytest.mark.parametrize("at_facet", [False, True])
def test_opposite_side_access_of_a_skeleton_field_is_rejected(at_facet):
    """A facet field is single-valued on its facet; its "other side" lives on the dummy element that
    is never numbered, so reading it used to yield a silent zero."""
    with pytest.raises(RuntimeError, match="single-valued on each facet"):
        with _OppositeProblem(at_facet) as p:
            p.quiet()
            p.initialise()


class _AdaptiveTrace(_TraceProblem):
    def define_problem(self):
        super().define_problem()
        self.max_refinement_level = 2


def _adapt_and_resolve(space):
    """Adapt once, then solve, reporting how many facets the transfer could not fill."""
    with _AdaptiveTrace(kind="quad", space=space) as p:
        p.quiet()
        p.initialise()
        p.solve()
        p.refine_uniformly()
        unrestored = len(p.get_mesh("domain/_internal_facets_").get_discontinuous_unrestored_elements())
        p.solve()
        return unrestored, _observables(p)


@pytest.mark.parametrize("space", ["D1", "D2"])
def test_nodal_dg_skeleton_field_is_carried_through_an_adaptation(space):
    """The nodal discontinuous spaces go through the adaptation exactly like DL now.

    They used to be refused, on the grounds that their values sit in per-node slots of the element's
    internal Data and there was "no get_interpolated_fields_Dx() to build the snapshot point cloud
    from". The sampler was always there - BulkElementBase::get_DG_fields_at_s(), which the bulk
    father->son transfer uses - so snapshot/refit now carries every discontinuous space, each fitted in
    its own basis (see dev_docs/internal_facet_fields.md).

    The oracle is DL, which took this path all along: a nodal space must reach the same facets (same
    unrestored count on the same mesh) and must be exact again after the solve, since a linear trace
    lies in every one of these spaces."""
    ref_unrestored, _ref = _adapt_and_resolve("DL")
    unrestored, obs = _adapt_and_resolve(space)
    assert unrestored == ref_unrestored, (
        "%s left %d facets unfilled where DL leaves %d on the same adaptation" % (
            space, unrestored, ref_unrestored))
    assert abs(obs["err2"]) < 1e-12, "%s: trace error %.3e after adapting and solving" % (space, obs["err2"])
    assert abs(obs["err1"]) < 1e-11, obs


# ----------------------------------------------------------------------------------------------
# 3d bulk meshes
# ----------------------------------------------------------------------------------------------
#
# Interior-facet enumeration in 3d (TemplatedMeshBase3d::fill_internal_facet_buffers) is built on the
# shape-neutral build_facet_adjacency() primitive, so bricks, tetrahedra, wedges, pyramids and mixed
# meshes all go through one code path. What is genuinely new per element family is the opposite-side
# matching of the FACE elements: a triangular face (tet, wedge cap, pyramid side) already had its 6
# vertex permutations, a quadrilateral one (brick face, wedge side, pyramid base) had none and threw
# "Implement". test_3d_opposite_side_matching_is_exact below is the sharp test of that: it compares
# the two sides' coordinates and a continuous field's trace point by point, which can only agree if
# every facet pair AND its local-coordinate map is right.
#
# Conforming meshes only - a non-conformingly refined mesh must say so rather than silently drop the
# 2:1 facets (where the coarse face and the four fine faces have different vertex sets and would all
# look like boundary facets).

import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyoomph.meshes.simplemeshes import CuboidBrickMesh
from box_mesh_3d import MixedBoxMesh3D

# Element families of the box mesh, plus one genuinely mixed layout. "hex" is covered by
# CuboidBrickMesh separately (that is the brick mesh users actually build).
_LAYOUTS_3D = ["tet", "wedge", "pyr", "all_four"]
# Every layout box_mesh_3d.py can build, for the SIP tests below: those are the ones that depend on
# the facet normal, which is a property of each element FAMILY, so a family pairing that only occurs
# in one mixed layout would otherwise never be assembled anywhere.
from box_mesh_3d import ALL_LAYOUTS as _LAYOUTS_3D_ALL  # noqa: E402
_BOUNDS_3D = ["left", "right", "top", "bottom", "front", "back"]


def _mesh_3d(kind, N=2):
    if kind == "brick":
        return CuboidBrickMesh(N=N)
    return MixedBoxMesh3D(kind=kind, N=N, name="domain")


class _Linear3dBulk(Equations):
    def __init__(self, space="C1"):
        super().__init__()
        self.space = space

    def define_fields(self):
        self.define_scalar_field("u", self.space)

    def define_residuals(self):
        u, ut = var_and_test("u")
        f = 1 + 2 * var("coordinate_x") + 3 * var("coordinate_y") - 0.5 * var("coordinate_z")
        self.add_residual(weak(u - f, ut))


class _Trace3d(Problem):
    def __init__(self, kind, space, bulk_space="C1", N=2):
        super().__init__()
        self.kind, self.space, self.bulk_space, self.N = kind, space, bulk_space, N

    def define_problem(self):
        self += _mesh_3d(self.kind, self.N)
        eqs = _Linear3dBulk(self.bulk_space)
        if self.space is not None:
            eqs += _FacetTrace(space=self.space) @ "_internal_facets_"
        self += eqs @ "domain"
        self.max_refinement_level = 0


@pytest.mark.parametrize("kind", ["brick"] + _LAYOUTS_3D)
@pytest.mark.parametrize("space", ["DL", "D0"])
def test_3d_trace_projection_is_exact(kind, space):
    """The linear bulk field's trace is in DL on every facet, so the projection is machine-exact; for
    D0 only the facet MEAN is, which is what err1 checks."""
    with _Trace3d(kind, space) as p:
        p.quiet()
        p.initialise()
        p.solve()
        obs = _observables(p)
    assert obs["meas"] > 0
    assert abs(obs["err1"]) < 1e-11, obs
    if space == "DL":
        assert abs(obs["err2"]) < 1e-12, obs


# DL on a 2d facet carries one constant plus two in-facet gradient modes, on triangular and
# quadrilateral facets alike, so a mixed skeleton still has a single mode count.
_MODES_3D = {"D0": 1, "DL": 3}


@pytest.mark.parametrize("kind", ["brick"] + _LAYOUTS_3D)
@pytest.mark.parametrize("space", ["DL", "D0"])
def test_3d_ndof_counts_one_set_of_facet_dofs_per_interior_facet(kind, space):
    """Cross-checks the enumerator against the independent facet count of facet_adjacency_summary():
    every interior facet must contribute exactly one set of dofs, and the never-numbered opposite
    dummy none at all."""
    with _Trace3d(kind, None) as p:
        p.quiet()
        p.initialise()
        ndof_ref = p.ndof()
    with _Trace3d(kind, space) as p:
        p.quiet()
        p.initialise()
        ndof = p.ndof()
        n_facets, n_bnd, n_int, max_inc = p.get_mesh("domain").facet_adjacency_summary()
    assert int(max_inc) == 2 and int(n_bnd) + int(n_int) == int(n_facets)
    assert ndof - ndof_ref == int(n_int) * _MODES_3D[space]


def test_3d_brick_facet_count_matches_the_closed_form():
    """N x N x N bricks: 3*N^2*(N+1) faces, 6*N^2 of them on the boundary."""
    N = 2
    with _Trace3d("brick", "DL", N=N) as p:
        p.quiet()
        p.initialise()
        n_facets, n_bnd, n_int, _m = p.get_mesh("domain").facet_adjacency_summary()
    assert int(n_facets) == 3 * N * N * (N + 1)
    assert int(n_bnd) == 6 * N * N
    assert int(n_int) == 3 * N * N * (N + 1) - 6 * N * N


class _OppositeSideCheck(Equations):
    """Declares no field of its own: it only measures how well the two sides of each facet agree."""

    def define_residuals(self):
        self.add_integral_function("jump2", jump(var("u")) ** 2 * self.get_dx())
        for c in ("x", "y", "z"):
            n = "coordinate_" + c
            self.add_integral_function("d" + c, (var(n, domain="|-") - var(n)) ** 2 * self.get_dx())
        self.add_integral_function("meas", 1 * self.get_dx())


class _OppositeSide3d(Problem):
    def __init__(self, kind, bulk_space):
        super().__init__()
        self.kind, self.bulk_space = kind, bulk_space

    def define_problem(self):
        self += _mesh_3d(self.kind)
        self += (_Linear3dBulk(self.bulk_space) + _OppositeSideCheck() @ "_internal_facets_") @ "domain"
        self.max_refinement_level = 0


@pytest.mark.parametrize("kind", ["brick"] + _LAYOUTS_3D)
@pytest.mark.parametrize("bulk_space", ["C1", "C2"])
def test_3d_opposite_side_matching_is_exact(kind, bulk_space):
    """Evaluating the opposite side at the mapped local coordinate must land on the SAME physical
    point: the coordinate residuals below are zero only if the facet pairing, the face indices and
    local_coordinate_in_opposite_side all agree. A wrong quad symmetry (the brick/wedge/pyramid case,
    which had no implementation at all before) shows up here as an O(1) mismatch, whereas a plain DG
    residual would only converge a bit worse."""
    with _OppositeSide3d(kind, bulk_space) as p:
        p.quiet()
        p.initialise()
        p.solve()
        obs = _observables(p)
    assert obs["meas"] > 0
    for k in ("dx", "dy", "dz"):
        assert abs(obs[k]) < 1e-14, obs
    assert abs(obs["jump2"]) < 1e-14, obs


class _JumpMeasure(Equations):
    def define_residuals(self):
        self.add_integral_function("jump2", jump(var("u")) ** 2 * self.get_dx())


class _SIP3d(Problem):
    """Symmetric interior penalty Poisson in 3d: the pre-existing DG machinery, which could not run in
    3d at all before because the facets could not be enumerated."""

    def __init__(self, kind, space, alpha=1, N=2):
        super().__init__()
        self.kind, self.space, self.alpha, self.N = kind, space, alpha, N

    def define_problem(self):
        self += _mesh_3d(self.kind, self.N)
        if self.space.startswith("D"):
            eqs = PoissonEquation(source=1, space=self.space, DG_alpha=self.alpha)
            eqs += _JumpMeasure() @ "_internal_facets_"
        else:
            eqs = PoissonEquation(source=1, space=self.space)
        for b in _BOUNDS_3D:
            eqs += DirichletBC(u=0) @ b
        eqs += IntegralObservables(uint=var("u"))
        self += eqs @ "domain"
        self.max_refinement_level = 0


@pytest.mark.parametrize("kind", ["brick"] + _LAYOUTS_3D_ALL)
def test_3d_sip_poisson_reproduces_the_continuous_solution(kind):
    """The DG solve must land on the continuous answer, not merely converge to something: an interior
    facet paired with the wrong neighbour still gives a symmetric positive system and a plausible
    Newton history.

    This used to run on bricks and tets only, and the tets needed DG_alpha=20 where bricks needed 1.
    That was read as a property of the SIP formulation ("tets are not coercive at alpha=1"). It was
    not: the tetrahedra of every hand-built mesh in tests/ were wound the other way round, so their
    face normals pointed inwards and the scheme was inconsistent - the penalty was merely papering
    over it, which is why the error looked like it decayed with alpha. add_tetra_3d_C1 repairs the
    winding now (see tests/test_tet_refinement.py), and alpha=1 is enough for every family."""
    with _SIP3d(kind, "C2") as p:
        p.quiet()
        p.initialise()
        p.solve()
        ref = _observables(p, "domain")["uint"]
    with _SIP3d(kind, "D2", alpha=1) as p:
        p.quiet()
        p.initialise()
        p.solve()
        jump2 = _observables(p)["jump2"]
        uint = _observables(p, "domain")["uint"]
    assert ref > 0
    assert jump2 < 1e-3 * ref, (jump2, ref)
    assert abs(uint - ref) < 0.2 * ref, (uint, ref)


class _LinearSIP3d(Problem):
    """SIP Poisson whose exact solution is linear, hence inside the "D1" space and reproduced exactly.

    Sharper than the comparison above, which is limited by the discretisation error of a solution
    neither space represents. Consistency is what is at stake: the flux term does NOT vanish at the
    exact solution, so it holds only if every facet normal, measure and pairing is right. The
    inward-normal bug moved `uerr2` from 1e-17 to 1e-1 on every tet-bearing layout."""

    def __init__(self, kind, N=2):
        super().__init__()
        self.kind, self.N = kind, N

    def define_problem(self):
        self += _mesh_3d(self.kind, self.N)
        exact = 1 + 2 * var("coordinate_x") + 3 * var("coordinate_y") + 4 * var("coordinate_z")
        eqs = PoissonEquation(source=0, space="D1", DG_alpha=1)
        for b in _BOUNDS_3D:
            eqs += DirichletBC(u=exact) @ b
        eqs += IntegralObservables(uerr2=(var("u") - exact) ** 2)
        eqs += _JumpMeasure() @ "_internal_facets_"
        self += eqs @ "domain"
        self.max_refinement_level = 0


@pytest.mark.parametrize("kind", ["brick"] + _LAYOUTS_3D_ALL)
def test_3d_sip_poisson_is_exact_for_a_linear_solution(kind):
    with _LinearSIP3d(kind) as p:
        p.quiet()
        p.initialise()
        p.solve()
        uerr2 = _observables(p, "domain")["uerr2"]
        jump2 = _observables(p)["jump2"]
    assert abs(uerr2) < 1e-13, "%s: |u - u_exact|^2 = %.3e, so the 3d SIP scheme is inconsistent" % (
        kind, uerr2)
    assert abs(jump2) < 1e-13, "%s: the solution is not continuous (jump^2 = %.3e)" % (kind, jump2)


class _RefinedSIP3d(_SIP3d):
    def __init__(self, uniform=0, partial=0):
        super().__init__("brick", "D2")
        self.uniform, self.partial = uniform, partial

    def define_problem(self):
        super().define_problem()
        if self.uniform:
            self += RefineToLevel(self.uniform) @ "domain"
        if self.partial:
            self += RefineToLevel(self.partial) @ "domain/top"
        self.max_refinement_level = 3


def test_3d_uniformly_refined_mesh_still_has_interior_facets():
    """Uniform refinement leaves the mesh conforming, so it must go through rather than trip the
    non-conforming guard - the guard keys on differing refinement levels, not on "was adapted"."""
    with _RefinedSIP3d(uniform=1) as p:
        p.quiet()
        p.initialise()
        p.solve()
        obs = _observables(p)
        _nf, _nb, n_int, max_inc = p.get_mesh("domain").facet_adjacency_summary()
    assert int(max_inc) == 2
    assert int(n_int) == 3 * 4 * 4 * (4 + 1) - 6 * 4 * 4  # 4^3 bricks after one uniform refinement
    assert obs["jump2"] < 1e-6, obs


@pytest.mark.parametrize("levels", [(0, 1), (1, 2)])
def test_3d_non_conforming_mesh_is_rejected_cleanly(levels):
    """2:1 facets have no matching vertex set on the coarse side, so the adjacency map would report
    all five faces involved as boundary facets and quietly build a skeleton with holes in it. The 2d
    version has a quadtree branch for this; 3d must say that it has not."""
    with pytest.raises(RuntimeError, match="conforming"):
        with _RefinedSIP3d(uniform=levels[0], partial=levels[1]) as p:
            p.quiet()
            p.initialise()


# ----------------------------------------------------------------------------------------------
# spatial adaptivity
# ----------------------------------------------------------------------------------------------
#
# The skeleton is never adapted incrementally: clear_before_adapt() deletes it and
# rebuild_after_adapt() regenerates it from the refined bulk mesh. What has to survive that is the
# element-owned DL/D0 data, which snapshot_discontinuous_data() samples on a point lattice before the
# deletion and restore_discontinuous_data() least-squares refits afterwards.
#
# Facets created INSIDE a refined bulk element are the interesting case: nothing in the old skeleton
# sits there, so they cannot be transferred at all. They keep zero (plus a one-time warning) unless
# the equations define a recovery expression through Equations.set_facet_recovery(), which is the HDG
# answer - the trace of a new facet is determined by the bulk solution, not by the old skeleton.
#
# Oracles used below:
#   * the bulk field is LINEAR, so its trace is exactly representable in DL and a correct transfer is
#     machine-exact rather than merely small;
#   * an extra D0 field "old" with residual weak(old-1) acts as a MASK: it is 1 wherever the restore
#     found samples and stays at its allocated 0 on the new facets, so integrating against it
#     separates "did the surviving facets survive exactly" from "were the new facets filled".
#     It is only a valid mask BEFORE the next solve, which is where every assertion using it sits.


class _MaskedFacetTrace(Equations):
    """DL trace of the bulk field plus a D0 mask marking the facets the restore could fill."""

    def __init__(self, space="DL", recovery=False):
        super().__init__()
        self.space, self.recovery = space, recovery

    def define_fields(self):
        self.define_scalar_field("p", self.space)
        self.define_scalar_field("old", "D0")

    def define_residuals(self):
        p, ptest = var_and_test("p")
        old, oldtest = var_and_test("old")
        u = avg(var("u"))
        self.add_residual(weak(p - u, ptest) + weak(old - 1, oldtest))
        if self.recovery:
            self.set_facet_recovery("p", u)
            self.set_facet_recovery("old", 1)
        self.add_integral_function("err2_old", old * (p - u) ** 2 * self.get_dx())
        self.add_integral_function("err2", (p - u) ** 2 * self.get_dx())
        # u is continuous and exactly linear, so its jump is exactly zero on every facet - including
        # a 2:1 one, where reading the coarse side goes through local_coordinate_in_opposite_side on
        # a shared opposite dummy.
        self.add_integral_function("jump2", jump(var("u")) ** 2 * self.get_dx())
        self.add_integral_function("meas_old", old * self.get_dx())
        self.add_integral_function("meas_new", (1 - old) * self.get_dx())


class _AdaptProblem(Problem):
    def __init__(self, kind="quad", N=3, space="DL", recovery=False):
        super().__init__()
        self.kind, self.N, self.space, self.recovery = kind, N, space, recovery

    def define_problem(self):
        self += _mesh_for(self.kind, self.N)
        eqs = _LinearBulk() + SpatialErrorEstimator(u=1)
        eqs += _MaskedFacetTrace(space=self.space, recovery=self.recovery) @ "_internal_facets_"
        self += eqs @ "domain"
        self.max_refinement_level = 2


def _refine_elements(p, indices, meshname="domain"):
    """Refine exactly the given bulk elements, at problem level.

    p.refine_uniformly() cannot produce a 2:1 configuration, and the mesh-level
    refine_selected_elements() alone would leave the interface meshes and the equation numbering
    behind - this is the same sequence Problem does around its own uniform refinement.
    """
    p.actions_before_adapt()
    p.get_mesh(meshname).refine_selected_elements(list(indices))
    p.relink_external_data()
    p.actions_after_adapt()
    p.reapply_boundary_conditions()
    p.assign_eqn_numbers()


def _skeleton_state(p):
    """Per-element D0 values of the skeleton, keyed by the facet's midpoint."""
    im = p.get_mesh("domain/_internal_facets_")
    out = {}
    for e in im.elements():
        mid = tuple(round(sum(e.node_pt(i).x(d) for i in range(e.nnode())) / e.nnode(), 10)
                    for d in range(e.node_pt(0).ndim()))
        out[mid] = [e.internal_data_pt(i).value(0) for i in range(e.ninternal_data())]
    return out


def test_linear_trace_survives_a_uniform_refinement():
    """Surviving facets must be exact immediately after the adaptation, i.e. WITHOUT a solve; the new
    ones must be visibly there and left at zero, and one solve must fix them."""
    with _AdaptProblem(kind="quad", N=3, space="DL") as p:
        p.quiet()
        p.initialise()
        p.solve()
        assert abs(_observables(p)["err2"]) < 1e-12
        p.refine_uniformly()
        obs = _observables(p)
        unrestored = list(p.get_mesh("domain/_internal_facets_").get_discontinuous_unrestored_elements())
        p.solve()
        after = _observables(p)
    assert obs["meas_old"] > 0 and obs["meas_new"] > 0, obs   # both populations exist
    # the mask counts exactly the elements the restore reported as unfilled
    assert obs["meas_new"] == pytest.approx(sum(1 for _ in unrestored) * 0.5 / 3, rel=1e-9), (obs, len(unrestored))
    assert abs(obs["err2_old"]) < 1e-14, obs                  # transfer is exact where it applies
    assert obs["err2"] > 1e-2, obs                            # ... and the new facets really are at 0
    assert abs(after["err2"]) < 1e-12, after                  # one solve repairs them


def test_new_facets_are_reported_and_warned_about(capfd):
    with _AdaptProblem(kind="quad", N=3, space="DL") as p:
        p.quiet()
        p.initialise()
        p.solve()
        capfd.readouterr()
        p.refine_uniformly()
        im = p.get_mesh("domain/_internal_facets_")
        unrestored = set(im.get_discontinuous_unrestored_elements())
        # "old" is the last internal Data (DL fields come before D0 ones)
        zeroed = {i for i, e in enumerate(im.elements())
                  if e.internal_data_pt(e.ninternal_data() - 1).value(0) == 0.0}
        out = capfd.readouterr().out
    assert unrestored, "no new facet was reported although the mesh was refined"
    assert unrestored == zeroed, (sorted(unrestored), sorted(zeroed))
    assert "set_facet_recovery" in out, out[-2000:]


def test_facet_recovery_fills_the_new_facets_exactly():
    """With a recovery expression the new facets are right straight away: no unrestored elements, no
    warning, and the projection error is machine zero before any solve."""
    with _AdaptProblem(kind="quad", N=3, space="DL", recovery=True) as p:
        p.quiet()
        p.initialise()
        p.solve()
        p.refine_uniformly()
        obs = _observables(p)
        unrestored = list(p.get_mesh("domain/_internal_facets_").get_discontinuous_unrestored_elements())
    assert unrestored == [], unrestored
    assert obs["meas_new"] == pytest.approx(0.0, abs=1e-12), obs   # the mask was recovered too
    assert abs(obs["err2"]) < 1e-14, obs


def test_two_to_one_refinement_keeps_the_skeleton_consistent():
    """One refined element next to unrefined neighbours: its four boundary facets are 2:1, i.e. two
    fine facet elements share ONE coarse opposite dummy (opposite_already_at_index). The bulk trace
    seen through avg() must stay right there, and the equation numbering must not pick up the dummy's
    pinned data."""
    with _AdaptProblem(kind="quad", N=4, space="DL") as p:
        p.quiet()
        p.initialise()
        p.solve()
        ndof_before = p.ndof()
        nel_before = p.get_mesh("domain/_internal_facets_").nelement()
        _refine_elements(p, [5])                     # an interior element of the 4x4 mesh
        im = p.get_mesh("domain/_internal_facets_")
        obs = _observables(p)
        unrestored = list(im.get_discontinuous_unrestored_elements())
        nel_after = im.nelement()
        ndof_after = p.ndof()
        p.solve()
        after = _observables(p)
    # 4 of the coarse element's facets split in two, 4 new ones appear inside it
    assert nel_after == nel_before + 4 + 4, (nel_before, nel_after)
    assert len(unrestored) == 4, unrestored
    assert abs(obs["err2_old"]) < 1e-14, obs        # 2:1 facets included: avg() still reads the bulk right
    assert abs(obs["jump2"]) < 1e-14, obs           # ... and so does the opposite side of a 2:1 facet (round-off only)
    assert obs["meas_new"] == pytest.approx(4 * 0.125, rel=1e-9), obs
    # DL(2) + D0(1) values per facet element, and the never-numbered opposite dummies contribute none
    assert ndof_after - ndof_before == (nel_after - nel_before) * 3 + 8
    assert abs(after["err2"]) < 1e-12, after


def test_residual_and_jacobian_assemble_after_an_adaptation():
    """The opposite dummies are re-pinned by generate_interface_elements on every rebuild; if they
    were not, their unclassified internal Data would surface here rather than at ndof()."""
    import numpy
    with _AdaptProblem(kind="quad", N=3, space="DL", recovery=True) as p:
        p.quiet()
        p.initialise()
        p.solve()
        _refine_elements(p, [0, 1])
        res = numpy.asarray(p.get_residuals())
        J = p.assemble_jacobian(with_residual=False).tocsr()
        p.solve()
        obs = _observables(p)
    assert res.shape[0] == J.shape[0] == J.shape[1]
    assert numpy.all(numpy.isfinite(res)) and numpy.all(numpy.isfinite(J.data))
    assert J.nnz > 0
    assert abs(obs["err2"]) < 1e-12, obs


def test_history_of_a_facet_field_survives_an_adaptation():
    """A field its own residual pins down algebraically recovers at the next solve even with no
    transfer at all. One carrying a time derivative does not: its history levels are what the next
    step is computed from, so those are what this checks."""
    class _FacetRate(Equations):
        def define_fields(self):
            self.define_scalar_field("r", "D0")

        def define_residuals(self):
            r, rtest = var_and_test("r")
            self.add_residual(weak(partial_t(r) - 1, rtest))

    class _Prob(Problem):
        def define_problem(self):
            self += RectangularQuadMesh(name="domain", N=3)
            eqs = _LinearBulk() + SpatialErrorEstimator(u=1)
            eqs += _FacetRate() @ "_internal_facets_"
            self += eqs @ "domain"
            self.max_refinement_level = 2

    def history(mesh):
        return [[e.internal_data_pt(0).value_at_t(t, 0) for e in mesh.elements()] for t in range(3)]

    with _Prob() as p:
        p.quiet()
        p.max_refinement_level = 2
        p.run(0.3, outstep=False, startstep=0.1, temporal_error=None, do_not_set_IC=False)
        before = history(p.get_mesh("domain/_internal_facets_"))
        p.refine_uniformly()
        im = p.get_mesh("domain/_internal_facets_")
        after = history(im)
        new = set(im.get_discontinuous_unrestored_elements())
        survivors = [i for i in range(im.nelement()) if i not in new]

    # r = t, so consecutive levels differ by the timestep - otherwise "history survived" would be
    # vacuous (all-zero levels would pass a same-before-and-after check just as well).
    # abs=1e-4 for the first gap because run() stretches the last step to land on the end time.
    assert before[0][0] - before[1][0] == pytest.approx(0.1, abs=1e-4), before
    assert before[1][0] - before[2][0] == pytest.approx(0.1, abs=1e-9), before
    assert survivors and new
    for t in range(3):
        assert all(after[t][i] == pytest.approx(before[t][0], abs=1e-12) for i in survivors), \
            (t, before[t][0], [after[t][i] for i in survivors])


def test_refine_then_unrefine_restores_the_original_values():
    """The coarsening direction, which is where the fit is genuinely doing work: a coarse facet is
    refitted from the samples of the two fine facets that covered it.

    The trap this pins down is the facets that DISAPPEAR - the ones created inside the refined
    element end ON the surrounding facets, so their samples used to be projected onto them and mixed
    into the fit. A constant field came back at ~5/6 of its value, a solved-for one at a third.
    """
    with _AdaptProblem(kind="quad", N=3, space="D0") as p:
        p.quiet()
        p.initialise()
        p.solve()
        before = _skeleton_state(p)
        p.refine_uniformly()
        p.solve()                       # the new facets get their correct values here
        p.unrefine_uniformly()
        after = _skeleton_state(p)
    assert len(before) == len(after) > 0
    assert set(before) == set(after), (sorted(before)[:3], sorted(after)[:3])
    worst = max(abs(a - b) for k in before for a, b in zip(after[k], before[k]))
    # not machine zero: the coarse value is a least-squares mean over points the locator projected
    # back onto the coarsened facet. A fit polluted by the vanishing interior facets lands at ~0.5.
    assert worst < 1e-9, worst


def test_triangle_skeleton_adapts_uniformly():
    """Uniform refinement keeps the mesh conforming, so the node-based branch of
    TemplatedMeshBase2d::fill_internal_facet_buffers applies and triangles work like quads.
    (DL only: a D0 trace cannot represent the linear field, so err2 would not be an oracle there -
    the D0 path is covered by the refine/unrefine test above.)"""
    with _AdaptProblem(kind="tri", N=2, space="DL") as p:
        p.quiet()
        p.initialise()
        p.solve()
        p.refine_uniformly()
        obs = _observables(p)
        p.solve()
        after = _observables(p)
    assert obs["meas_old"] > 0 and obs["meas_new"] > 0, obs
    assert abs(obs["err2_old"]) < 1e-14, obs
    assert abs(after["err2"]) < 1e-12, after


def test_triangle_skeleton_rejects_a_non_uniform_adaptation():
    """With hanging nodes the 2d enumerator switches to the quadtree neighbour walk, which is
    quad-only. That has to be an error, not a skeleton with holes in it."""
    with pytest.raises(RuntimeError, match="Mixed meshes here"):
        with _AdaptProblem(kind="tri", N=2, space="DL") as p:
            p.quiet()
            p.initialise()
            p.solve()
            _refine_elements(p, [3])


# ----------------------------------------------------------------------------------------------
# remeshing
# ----------------------------------------------------------------------------------------------
#
# A remesh replaces the mesh wholesale, so - unlike an adaptation - nothing stays where it was and
# there is no old facet to inherit from. What makes the transfer possible at all is that the OLD mesh
# is still alive while the new one is filled: every new facet PULLS the trace it needs at its own
# sample points, instead of the old skeleton pushing a snapshot onto whatever comes out (which is all
# an adaptation can do, and which leaves a refined skeleton mostly empty - each old sample lands on
# exactly one new facet).
#
# The facet soup is disambiguated topologically, not by distance: a new facet is located in the OLD
# BULK mesh and may only take values from old facets of the bulk element(s) it runs through, widening
# by one ring of face neighbours if that finds nothing. Without that, "the nearest old facet" is
# routinely one on the far side of a bulk element, carrying a different trace altogether.
#
# Oracles, in increasing order of severity:
#   * an IDENTICAL remesh (gmsh gets the same input twice) must reproduce the old state exactly - the
#     new facets coincide with the old ones, so every sample sits on its source and nothing is
#     interpolated. This is the sharp test; it fails on any wiring, ordering or history mistake.
#   * a genuinely different mesh cannot be exact: the values are old traces evaluated a distance away,
#     so the error is O(distance * gradient). The assertions bound it against h*|grad u| and check
#     that one solve removes it entirely.
#   * the "old" D0 mask of _MaskedFacetTrace marks the facets the transfer could fill, exactly as in
#     the adaptivity section above.


class _RemeshTri(GmshTemplate):
    """A triangle domain, meshed by gmsh so that it can actually be remeshed.

    ``res_after`` makes the remesh produce a genuinely DIFFERENT mesh: RemesherViaRecreation calls
    define_geometry() again, and is_remeshing() is True the second time. Left at None, gmsh is handed
    the same input twice and returns the same mesh, which is the case the transfer must get exact.
    """

    def __init__(self, res=0.3, res_after=None, mode="tris"):
        super().__init__()
        self.res, self.res_after, self.mode = res, res_after, mode

    def define_geometry(self):
        self.default_resolution = self.res_after if (self.res_after is not None and self.is_remeshing()) else self.res
        self.mesh_mode = self.mode
        p00, p10, p01 = self.point(0, 0), self.point(1, 0), self.point(0, 1)
        self.line(p00, p10, name="bottom")
        self.line(p10, p01, name="diag")
        self.line(p01, p00, name="axis")
        self.plane_surface("bottom", "diag", "axis", name="domain")


class _RemeshProblem(Problem):
    def __init__(self, space="DL", res=0.3, res_after=None, recovery=False, mode="tris", adaptive=False):
        super().__init__()
        self.space, self.res, self.res_after, self.recovery = space, res, res_after, recovery
        self.mode, self.adaptive = mode, adaptive

    def define_problem(self):
        m = _RemeshTri(self.res, self.res_after, self.mode)
        # Explicitly, rather than relying on the auto-remesher: force_remesh() skips a template whose
        # define_geometry does not react to remeshing, which is exactly the res_after=None case here.
        m.remesher = RemesherViaRecreation(m)
        self += m
        eqs = _LinearBulk()
        if self.adaptive:
            eqs += SpatialErrorEstimator(u=1)
        eqs += _MaskedFacetTrace(space=self.space, recovery=self.recovery) @ "_internal_facets_"
        self += eqs @ "domain"
        self.max_refinement_level = 2 if self.adaptive else 0
        self.initial_adaption_steps = 0


# |grad u| of the linear bulk field, i.e. the scale every transfer error is measured against.
_GRAD_U = math.sqrt(2 ** 2 + 3 ** 2)


def _rms(obs):
    """Root-mean-square trace error over the whole skeleton."""
    return math.sqrt(max(obs["err2"], 0.0) / obs["meas_old"] if obs["meas_old"] > 0 else 0.0)


def _skeleton_history(p, nt=3):
    """Per-element internal-data values at every time level, keyed by the facet's midpoint."""
    im = p.get_mesh("domain/_internal_facets_")
    out = {}
    for e in im.elements():
        mid = tuple(round(sum(e.node_pt(i).x(d) for i in range(e.nnode())) / e.nnode(), 10)
                    for d in range(e.node_pt(0).ndim()))
        out[mid] = [[e.internal_data_pt(i).value_at_t(t, j)
                     for i in range(e.ninternal_data())
                     for j in range(e.internal_data_pt(i).nvalue())]
                    for t in range(nt)]
    return out


def _unrestored(p):
    return list(p.get_mesh("domain/_internal_facets_").get_discontinuous_unrestored_elements())


@pytest.mark.parametrize("space", ["DL", "D0"])
def test_an_identical_remesh_reproduces_the_skeleton_exactly(space):
    """The sharp one. Nothing is interpolated here - every sample point of every new facet sits on the
    old facet it came from - so the whole state has to come back to round-off, without a solve."""
    with _RemeshProblem(space=space) as p:
        p.quiet()
        p.initialise()
        p.solve()
        before = _observables(p)
        state_before = _skeleton_history(p, nt=1)
        p.force_remesh(interpolator=InternalInterpolator)
        after = _observables(p)
        state_after = _skeleton_history(p, nt=1)
        unrestored = _unrestored(p)
    assert before["meas_old"] > 0
    assert unrestored == [], unrestored
    assert after["meas_new"] == pytest.approx(0.0, abs=1e-12), after   # the D0 mask came over as 1
    assert set(state_before) == set(state_after), (len(state_before), len(state_after))
    worst = max(abs(a - b)
                for k in state_before
                for a, b in zip(state_after[k][0], state_before[k][0]))
    assert worst < 1e-12, worst
    assert after["err2"] == pytest.approx(before["err2"], abs=1e-12), (before, after)


def test_a_coarser_remesh_keeps_the_trace_within_the_interpolation_error():
    """Every new facet still finds old traces around it, so nothing is left at zero; the values are
    old traces read off a distance away, which is a first-order error in that distance."""
    with _RemeshProblem(space="DL", res=0.3, res_after=0.6) as p:
        p.quiet()
        p.initialise()
        p.solve()
        n_before = p.get_mesh("domain/_internal_facets_").nelement()
        p.force_remesh(interpolator=InternalInterpolator)
        obs = _observables(p)
        unrestored = _unrestored(p)
        n_after = p.get_mesh("domain/_internal_facets_").nelement()
        p.solve()
        after = _observables(p)
    assert n_after < n_before, (n_before, n_after)     # the remesh really did coarsen
    assert unrestored == [], unrestored
    assert obs["meas_new"] == pytest.approx(0.0, abs=1e-12), obs
    # Well below one element's worth of the field's variation, and far below the ~3.5 an untransferred
    # (zeroed) skeleton would give.
    assert _rms(obs) < 0.1 * 0.6 * _GRAD_U, _rms(obs)
    assert abs(after["err2"]) < 1e-12, after           # one solve makes it exact again


def test_a_finer_remesh_still_reaches_almost_every_new_facet():
    """The direction that a snapshot-and-push transfer cannot do: most new facets lie in the INTERIOR
    of an old bulk element, where the old skeleton has no points to push. Pulling from the old facets
    around them covers all but a handful."""
    with _RemeshProblem(space="DL", res=0.3, res_after=0.15) as p:
        p.quiet()
        p.initialise()
        p.solve()
        n_before = p.get_mesh("domain/_internal_facets_").nelement()
        p.force_remesh(interpolator=InternalInterpolator)
        n_after, unrestored = p.get_mesh("domain/_internal_facets_").nelement(), _unrestored(p)
        obs = _observables(p)
        p.solve()
        after = _observables(p)
    assert n_after > 2 * n_before, (n_before, n_after)
    assert len(unrestored) < 0.1 * n_after, (len(unrestored), n_after)
    # err2 counts the zeroed facets too, so the trace error is measured on the filled ones only.
    assert math.sqrt(obs["err2_old"] / obs["meas_old"]) < 0.1 * 0.3 * _GRAD_U, obs
    assert abs(after["err2"]) < 1e-12, after


def test_the_history_of_a_facet_field_survives_a_remesh():
    """A field its own residual determines algebraically is repaired by the next solve even with no
    transfer at all; its HISTORY is not, and that is what the next time step is computed from."""

    class _FacetRate(Equations):
        def define_fields(self):
            self.define_scalar_field("r", "DL")

        def define_residuals(self):
            r, rtest = var_and_test("r")
            self.add_residual(weak(partial_t(r) - (1 + var("coordinate_x")), rtest))

    class _Prob(Problem):
        def define_problem(self):
            m = _RemeshTri()
            m.remesher = RemesherViaRecreation(m)
            self += m
            self += (_LinearBulk() + _FacetRate() @ "_internal_facets_") @ "domain"
            self.max_refinement_level = 0

    with _Prob() as p:
        p.quiet()
        p.run(0.3, outstep=False, startstep=0.1, temporal_error=None, do_not_set_IC=False)
        before = _skeleton_history(p)
        p.force_remesh(interpolator=InternalInterpolator)
        after = _skeleton_history(p)
        unrestored = _unrestored(p)
    # r = t*(1+x): the levels differ from each other AND from facet to facet, so neither a zeroed
    # history nor a per-facet mix-up could pass this.
    assert unrestored == [], unrestored
    assert set(before) == set(after), (len(before), len(after))
    assert len(set(round(v[0][0], 8) for v in before.values())) > 3, "r is not varying over the facets"
    for k in before:
        for t in range(3):
            assert after[k][t] == pytest.approx(before[k][t], abs=1e-12), (k, t, before[k], after[k])


def test_the_recovery_expression_fills_what_a_finer_remesh_could_not():
    """Same remesh as above, but with set_facet_recovery: the facets the old skeleton cannot reach are
    filled from the bulk instead of being left at zero, so nothing is reported unrestored and the D0
    mask comes out at 1 everywhere."""
    common = dict(space="DL", res=0.3, res_after=0.15)
    with _RemeshProblem(recovery=False, **common) as p:
        p.quiet()
        p.initialise()
        p.solve()
        p.force_remesh(interpolator=InternalInterpolator)
        plain = _observables(p)
        n_plain = len(_unrestored(p))
    with _RemeshProblem(recovery=True, **common) as p:
        p.quiet()
        p.initialise()
        p.solve()
        p.force_remesh(interpolator=InternalInterpolator)
        rec = _observables(p)
        unrestored = _unrestored(p)
    assert n_plain > 0, "nothing was left over, so this test would be vacuous"
    assert unrestored == [], unrestored
    assert rec["meas_new"] == pytest.approx(0.0, abs=1e-12), rec
    assert plain["meas_new"] > 0, plain
    assert rec["err2"] < plain["err2"], (rec, plain)


def test_facets_a_remesh_could_not_fill_are_reported_and_warned_about(capfd):
    with _RemeshProblem(space="DL", res=0.3, res_after=0.15) as p:
        p.quiet()
        p.initialise()
        p.solve()
        capfd.readouterr()
        p.force_remesh(interpolator=InternalInterpolator)
        im = p.get_mesh("domain/_internal_facets_")
        unrestored = set(im.get_discontinuous_unrestored_elements())
        # "old" is the last internal Data (DL fields come before D0 ones)
        zeroed = {i for i, e in enumerate(im.elements())
                  if e.internal_data_pt(e.ninternal_data() - 1).value(0) == 0.0}
        out = capfd.readouterr().out
    assert unrestored, "no new facet was reported although the mesh was refined"
    assert unrestored == zeroed, (sorted(unrestored), sorted(zeroed))
    assert "set_facet_recovery" in out, out[-2000:]
    assert "from the previous mesh" in out, out[-2000:]   # not the adaptation wording


def test_a_nodal_dx_facet_field_is_rejected_on_a_remesh():
    """Dx values sit in per-node slots with no get_interpolated_fields_Dx() to read them, so neither
    transfer path can carry them.

    It has to be refused BEFORE the remesh starts, not by the rebuild in the middle of it: throwing
    from there leaves half the meshes replaced and the problem unusable - which showed up as the NEXT
    problem in the same process segfaulting, not as a failure here."""
    with pytest.raises(RuntimeError, match="Cannot remesh"):
        with _RemeshProblem(space="D1") as p:
            p.quiet()
            p.initialise()
            p.solve()
            p.force_remesh(interpolator=InternalInterpolator)


def test_a_remesh_followed_by_an_adaptation_composes():
    """The two paths in sequence: the adaptation snapshots and refits the skeleton the remesh has just
    transferred. After a solve the transferred state is exact again, so the adaptation's own oracle
    applies unchanged - surviving facets machine-exact, new ones reported."""
    with _RemeshProblem(space="DL", res=0.3, res_after=0.45, adaptive=True) as p:
        p.quiet()
        p.initialise()
        p.solve()
        p.force_remesh(interpolator=InternalInterpolator)
        p.solve()
        assert abs(_observables(p)["err2"]) < 1e-12
        p.refine_uniformly()
        obs = _observables(p)
        p.solve()
        after = _observables(p)
    assert obs["meas_old"] > 0 and obs["meas_new"] > 0, obs   # both populations exist
    assert abs(obs["err2_old"]) < 1e-13, obs                  # the facets that survived are exact
    assert abs(after["err2"]) < 1e-12, after


def test_the_projection_interpolator_transfers_the_skeleton_too():
    """ProjectionInternalInterpolator does not walk the skeleton itself; it seeds from the nodal
    interpolator and then re-runs the facet transfer after its solve. The re-run matters: only the
    meshes put into projection mode assemble the projection residual, so during that solve the facet
    unknowns are assembled with their PHYSICAL equations and would otherwise be left wherever those
    dragged them."""
    with _RemeshProblem(space="DL") as p:
        p.quiet()
        p.initialise()
        p.solve()
        before = _observables(p)
        state_before = _skeleton_history(p, nt=1)
        p.force_remesh(interpolator=ProjectionInternalInterpolator)
        after = _observables(p)
        state_after = _skeleton_history(p, nt=1)
        unrestored = _unrestored(p)
    assert unrestored == [], unrestored
    assert set(state_before) == set(state_after)
    worst = max(abs(a - b)
                for k in state_before
                for a, b in zip(state_after[k][0], state_before[k][0]))
    assert worst < 1e-10, worst
    assert after["err2"] == pytest.approx(before["err2"], abs=1e-10), (before, after)
