#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  @author Maxim de Wildt <m.dewildt@utwente.nl>
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

# Serial half of the coupled-interface adaptivity campaign.
#
# Two domains sharing an interface are adapted INDIVIDUALLY by oomph-lib, so a refinement criterion
# stated for one of them leaves the other with no reason to follow -- and the opposite-element matcher,
# which pairs interface elements by exact vertex-position sets, then has nothing to pair up. Every case
# here drives refinement asymmetrically on purpose. See dev_docs/interface_refinement_coupling.md.
#
# Problem definitions live in two_domain_cases.py and are shared verbatim with the MPI harness
# (test_mpi_interface_coupling.py), so the serial and distributed campaigns cannot drift apart.
#
# Oracles, in increasing order of strength:
#   * the run does not die in connect_interface_elements_by_kdtree -- necessary, and it is what fails
#     today, but it says nothing about whether the answer is right;
#   * Problem.check_interface_conformity() == 0 -- the invariant stated directly rather than inferred
#     from the absence of a crash;
#   * max|residual| ~ 0 and one Newton step removing the whole residual (these problems are linear);
#   * u == y at every node -- the exact solution of the coupled Poisson cases, representable in every
#     discretisation here. A mis-paired opposite element leaves the two sides coupled to the WRONG
#     neighbour, which this catches even when the residual has happily converged.
#
# All four are statements about FACETS or about the solution, and a mesh can be badly graded while every
# one of them is perfect -- which is how the vertex-only blind spot of dev_docs section 14.9 survived
# this matrix. two_domain_cases.max_vertex_level_jump is the fifth oracle, and the only one that looks
# at an element the conformity machinery cannot see.
#
# Negative-tested: with PYOOMPH_DISABLE_INTERFACE_CONFORMITY=1, 80 of the 112 (kind, eq, levels)
# combinations fail. The 32 that survive are exactly the ones that cannot detect anything -- the
# non-adaptive (0,0) baseline and the "interface" criterion, which is symmetric by construction (see
# TwoDomainProblem._add_refinement_criterion). test_interface_criterion_is_symmetric pins that down so
# the distinction stays honest rather than becoming an unexamined pass.

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import two_domain_cases

# Part of the validation campaign, so the wheel builds deselect it with -m "not campaign" (see
# conftest.py). Not marked "slow": it stays in the fast local run.
pytestmark = pytest.mark.campaign

_KINDS = two_domain_cases.MESH_KINDS
_LEVELS = two_domain_cases.LEVELS

# Machine-zero bound for a converged linear solve; ALE carries the mesh-position dofs too.
_RES_TOL = {"ale": 1e-12}
_RES_TOL_DEFAULT = 1e-9

# One Newton step must reduce the residual by at least this factor: these problems are linear, so an
# exact analytic Jacobian reaches machine zero in a single step. Same oracle and same calibration as
# test_adaptive_2d_campaign.py.
_NEWTON_REDUCTION = 1e-10

# How far u may stray from the exact solution y. NOT machine zero, and deliberately so: the
# Lagrange-multiplier formulation of ConnectFieldsAtInterface does not reproduce a linear field exactly
# even on a single unrefined mesh (~3e-10 on 81 dofs for the C2 variants), so this bound measures the
# COUPLING, not the refinement. A torn interface misses it by orders of magnitude.
_U_TOL = 1e-8


def _solve(kind, eq, levels, tmp_path):
    cid = two_domain_cases.case_id(kind, eq, levels)
    return two_domain_cases.solve_case(kind, eq, levels, outdir=str(tmp_path / cid))


def _assert_case_is_sound(res, eq, cid):
    assert res["nonconforming"] == 0, \
        "%s: %d boundary facet(s) of the coupled interface have no counterpart on the opposite side" \
        % (cid, res["nonconforming"])
    tol = _RES_TOL.get(eq, _RES_TOL_DEFAULT)
    assert res["maxres"] < tol, "%s: max|residual| = %.3e" % (cid, res["maxres"])
    conv = res["newton_conv"]
    assert len(conv) >= 2, "%s: no Newton iteration was performed (history %r)" % (cid, conv)
    if eq == "ale":
        # The ALE cases end on a converged solve, so their history starts at machine zero and carries no
        # information: the reduction oracle needs a solve that starts from a real residual, and the only
        # way to arrange one here would be to disturb the mesh POSITIONS, which is not a reset but a
        # different problem. maxres plus the conformity check carry these cases; the hanging-node
        # Jacobian on moving meshes is covered by the ALE cases of test_adaptive_2d_campaign.py.
        assert res["newton_steps"] <= 3, "%s: Newton took %d steps (history %r)" % (cid, res["newton_steps"], conv)
        return
    assert conv[0] > 1e-6, \
        "%s: initial residual %.3e is too small for the reduction test to mean anything -- solve_case " \
        "is meant to wipe the field before the final solve" % (cid, conv[0])
    reduction = conv[1] / conv[0]
    assert reduction < _NEWTON_REDUCTION, \
        "%s: one Newton step only reduced the residual by %.2e (%.3e -> %.3e) -- for a linear problem " \
        "that means the analytic Jacobian does not match the residual" % (cid, reduction, conv[0], conv[1])
    if "maxuerr" in res:
        assert res["maxuerr"] < _U_TOL, \
            "%s: u deviates from the exact solution y by %.3e -- the two sides of the interface are " \
            "coupled, but to the wrong elements" % (cid, res["maxuerr"])


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("levels", _LEVELS)
@pytest.mark.parametrize("eq", ["connect1", "connect2", "connect12"])
def test_connected_fields_across_asymmetrically_refined_interface(eq, kind, levels, tmp_path):
    # ConnectFieldsAtInterface across an interface whose two sides are told to refine differently.
    # connect12 additionally gives the two domains different spaces (C2 below, C1 above), so the
    # coupling space itself has to be negotiated and a hanging node on one side meets a
    # differently-interpolated node on the other.
    cid = two_domain_cases.case_id(kind, eq, levels)
    _assert_case_is_sound(_solve(kind, eq, levels, tmp_path), eq, cid)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("levels", _LEVELS)
def test_connected_moving_mesh_across_asymmetrically_refined_interface(kind, levels, tmp_path):
    # ConnectMeshAtInterface: the interface GEOMETRY is itself an unknown, so the facet positions the
    # conformity machinery keys on are solution-dependent rather than fixed by the template.
    cid = two_domain_cases.case_id(kind, "ale", levels)
    _assert_case_is_sound(_solve(kind, "ale", levels, tmp_path), "ale", cid)


@pytest.mark.parametrize("kind", _KINDS)
def test_refinement_actually_reached_the_opposite_domain(kind, tmp_path):
    # The conformity assertions above are satisfiable by doing nothing at all -- if neither domain ever
    # refined, both sides trivially agree. This is what rules that out: refining ONLY the lower domain
    # has to grow the problem well beyond the unrefined baseline, and the upper domain has to have
    # followed rather than merely not crashed.
    base = _solve(kind, "connect1", (0, 0, "level"), tmp_path)
    driven = _solve(kind, "connect1", (0, 3, "level"), tmp_path)
    assert driven["ndof"] > 4 * base["ndof"], \
        "%s: refinement did not take effect (ndof %d -> %d)" % (kind, base["ndof"], driven["ndof"])
    # The upper domain carries no refinement criterion of its own, so any growth in its own observable
    # count comes from having been dragged along by the interface. Its area is unchanged by refinement,
    # so compare what does change: the interface can only conform if "upper" refined too, and
    # nonconforming == 0 with a 3-level jump on the other side is only reachable that way.
    assert driven["nonconforming"] == 0


@pytest.mark.parametrize("kind", _KINDS)
def test_interface_criterion_is_symmetric(kind, tmp_path):
    # A criterion stated ON the interface reaches BOTH adjacent bulk elements through
    # InterfaceMesh._override_bulk_errors_where_necessary, so it is symmetric without any help from the
    # conformity machinery -- which is why it is one of the cases that still passes when that machinery
    # is switched off. Asserting the symmetry directly keeps that a checked property rather than an
    # unexamined pass: the two domains must end up with the SAME number of interface facets.
    res = _solve(kind, "connect1", (1, 2, "interface"), tmp_path)
    assert res["nonconforming"] == 0
    assert res["maxuerr"] < _U_TOL


@pytest.mark.parametrize("kind", _KINDS)
def test_adapt_selection_is_reconciled_before_acting(kind, tmp_path):
    # The two sides must agree BEFORE either of them acts, not be patched up afterwards.
    #
    # Both routes end at a conforming mesh, so nothing above this test can tell them apart -- but they
    # are not equally good. Repairing after the fact means an element that has just been merged away is
    # refined again, and its sons are then re-interpolated from the merged father: the answer is still
    # correct, the fine-scale solution is not still there. It also makes adapt non-idempotent, so
    # successive adapts can oscillate.
    #
    # pyoomph therefore adapts coupled meshes in two stages -- decide for all of them, reconcile the
    # decisions across every coupled interface, then act (Problem._adapt_with_interfacial_errors). This
    # asserts that the reconciliation is what does the work: the post-adapt repair must have nothing
    # left to do.
    #
    # The "estimator" criterion is the one that discriminates, and it is also the realistic one: no
    # explicit level anywhere, just a Z2 estimate on a field that is only sharp in the lower domain, so
    # the two domains genuinely disagree DURING adaptation rather than at set-up. Measured with
    # PYOOMPH_DISABLE_ADAPT_RECONCILIATION=1 it needs 5-7 after-the-fact repairs; with reconciliation,
    # zero. The criteria that reach their target level at initialise never disagree during adapt and so
    # need no repair either way -- they cannot detect this and are not used here.
    res = _solve(kind, "connect1", (1, 2, "estimator"), tmp_path)
    assert res["nonconforming"] == 0
    assert res["repairs_during_adapt"] == 0, \
        "%s: the post-adapt repair had to refine %d element(s) back -- the two sides acted before " \
        "agreeing, so that patch lost its fine-scale solution to a merge/re-refine round trip" \
        % (kind, res["repairs_during_adapt"])


def test_lower_max_refinement_level_governs_the_interface(tmp_path):
    # Coupled domains with DIFFERENT max_refinement_level. The shallower cap has to win AT THE
    # INTERFACE -- the two sides cannot be matched otherwise -- while leaving each domain free to
    # refine to its own cap away from it. That is not a repair, it happens in the decision: a facet
    # whose partner cannot follow (because it sits at its own max_refinement_level) is not selected for
    # refinement in the first place, see IFACET_CAN_REFINE in harmonise_adapt_selection.
    prob = two_domain_cases.TwoDomainProblem(kind="quad", eq="connect1", levels=(0, 3, "callback"))
    with prob as p:
        p.set_output_directory(str(tmp_path / "caps"))
        p.max_refinement_level = 3
        p.initial_adaption_steps = 0   # nothing refines until the caps below are in place
        p.initialise()
        p.get_mesh("lower").max_refinement_level = 3
        p.get_mesh("upper").max_refinement_level = 1
        for _ in range(5):
            p.solve(spatial_adapt=1)

        def levels_on_boundary(mesh, bname):
            bi = mesh.get_boundary_names().index(bname)
            return {mesh.boundary_element_pt(bi, i).refinement_level()
                    for i in range(mesh.nboundary_element(bi))}

        lower, upper = p.get_mesh("lower"), p.get_mesh("upper")
        assert p.check_interface_conformity(throw_on_mismatch=False, when="capped") == 0
        assert max(levels_on_boundary(lower, "interface")) <= 1, \
            "the deep side refined past its partner's cap at the interface"
        assert max(levels_on_boundary(upper, "interface")) <= 1
        # ...but the cap only binds AT the interface: the deep domain still reaches its own level
        # elsewhere, which is the whole point of not simply clamping both domains to the minimum.
        assert max(e.refinement_level() for e in lower.elements()) == 3, \
            "the deep side was held back away from the interface too"


def test_unsatisfiable_cap_is_diagnosed_not_left_to_the_matcher(tmp_path):
    # The case the above cannot rescue: RefineToLevel drives one domain through refine_uniformly,
    # which does NOT respect max_refinement_level, so the other side can end up needing a refinement
    # its own cap forbids. Nothing downstream can repair that -- and left alone it surfaces as
    # "Cannot locate opposite node at x=(...)" from the opposite-element matcher, which says nothing
    # about why. It must be diagnosed where the reason is still known.
    class Capped(two_domain_cases.TwoDomainProblem):
        def actions_before_adapt(self):
            m = self._meshdict.get("upper")
            if m is not None and hasattr(m, "max_refinement_level"):
                m.max_refinement_level = 1
            super().actions_before_adapt()

    prob = Capped(kind="quad", eq="connect1", levels=(0, 3, "level"))
    with pytest.raises(RuntimeError, match="Cannot make the two sides of a coupled interface match"):
        with prob as p:
            p.set_output_directory(str(tmp_path / "unsat"))
            p.max_refinement_level = 3
            p.initialise()


@pytest.mark.parametrize("kind", two_domain_cases.FOUR_DOMAIN_KINDS)
def test_four_domains_meeting_at_a_cross_point(kind, tmp_path):
    # Four domains, A|B over C|D. Two things here that the two-domain cases cannot reach.
    #
    # The coupling graph is a CYCLE, not a chain: A-B-D-C-A. D shares no interface with A -- they touch
    # only at the cross point -- so a refinement demand raised in A reaches D only by travelling all the
    # way round. And the cross point is four distinct nodes tied pairwise by four Lagrange multipliers,
    # of which only three are independent: a genuine over-constraint at a single point.
    res = two_domain_cases.solve_case(kind, None, None, outdir=str(tmp_path / kind))
    _assert_case_is_sound(res, "connect1", kind)
    if kind == "four_corner":
        # Raised in A alone, at the cross point; every other domain has to follow it round the cycle.
        assert res["maxlevel"]["D"] == 3, \
            "refinement did not reach the diagonally opposite domain (levels %r)" % (res["maxlevel"],)
    elif kind == "four_away":
        # ...but only where it must. A is refined far from every interface, so nothing may propagate.
        # This is what separates "the neighbours follow where they have to" from "the neighbours follow
        # always"; the other two cases cannot tell those apart.
        assert res["maxlevel"]["A"] == 3
        assert all(res["maxlevel"][d] == 0 for d in "BCD"), \
            "refinement away from the interfaces still dragged the neighbours along (levels %r)" \
            % (res["maxlevel"],)


def test_conformity_check_reports_a_torn_interface(tmp_path, monkeypatch):
    # The checker is the oracle every other test in this file leans on, so it must be able to FAIL.
    # Switch the enforcement off, refine one domain only, and require the check to say so -- and to say
    # so by raising when asked to.
    monkeypatch.setenv("PYOOMPH_DISABLE_INTERFACE_CONFORMITY", "1")
    prob = two_domain_cases.TwoDomainProblem(kind="quad", eq="connect1", levels=(1, 2, "level"))
    with prob as p:
        p.set_output_directory(str(tmp_path / "torn"))
        p.max_refinement_level = 3
        # initialise() applies the per-domain uniform refinement, which is already asymmetric. It ends
        # in the opposite-element matcher, which is exactly what a torn interface breaks.
        with pytest.raises(RuntimeError, match="Cannot locate opposite"):
            p.initialise()
        assert p.check_interface_conformity(throw_on_mismatch=False, when="torn") > 0
        with pytest.raises(RuntimeError, match="Interface conformity violated"):
            p.check_interface_conformity(throw_on_mismatch=True, when="torn")


@pytest.mark.parametrize("kind", _KINDS)
def test_vertex_connected_elements_follow_the_forced_interface(kind, tmp_path):
    # Conformity is a statement about FACETS -- and an element can share a single VERTEX with a coupled
    # interface while carrying no facet on it at all. Nothing that enforces conformity can see such an
    # element: it is not a boundary element, and it contributes no key to either side's facet set. So
    # when the OPPOSITE domain is what forces the refinement, its facet-carrying neighbours follow it
    # and it does not, and the level jump across that shared vertex grows without bound.
    #
    # Every tri kind here has such triangles at the interface -- one per cell for the two-way splits,
    # two for tri_crossed -- so driving "lower" to level 3 uniformly used to leave the "upper" side
    # with level-3 facet elements sitting against corner neighbours at levels 0 through 2. "quad" has
    # none of them (every quad above the interface meets it with a full edge) and is carried along to
    # show the test measures the geometry rather than the driving.
    res = _solve(kind, "connect1", (0, 3, "level"), tmp_path)
    assert res["nonconforming"] == 0
    assert res["vertex_jump"] <= 1, \
        "%s: an element touching the coupled interface at a vertex only is %d levels coarser than the " \
        "interface there -- the facet-based conformity machinery cannot see it, so the vertex closure " \
        "has to" % (kind, res["vertex_jump"])


@pytest.mark.parametrize("kind", ["tri_left", "tri_crossed", "mixed"])
def test_vertex_balance_closure_is_what_closes_it(kind, tmp_path, monkeypatch):
    # The negative test for the one above: with the closure switched off the same case must FAIL, or
    # that assertion is measuring nothing. Only the kinds that actually have vertex-only elements at the
    # interface can discriminate -- "quad" cannot, which is why it is not parametrised here.
    monkeypatch.setenv("PYOOMPH_DISABLE_INTERFACE_VERTEX_BALANCE", "1")
    res = _solve(kind, "connect1", (0, 3, "level"), tmp_path)
    # Facet conformity is untouched by the closure and must still hold -- the two are separate
    # properties, and this is what says so.
    assert res["nonconforming"] == 0
    assert res["vertex_jump"] > 1, \
        "%s: no vertex-only element lags behind the interface even with the closure disabled, so " \
        "test_vertex_connected_elements_follow_the_forced_interface cannot be measuring it" % kind


# --- The mixed-element-space matrix (dev_docs/interface_refinement_coupling.md section 14) ------------
#
# The tests above run the two domains in the same space, or in the one hard-coded C2/C1 pair of
# "connect12". These sweep the space axis itself, and they pin BOTH outcomes: what works today, and what
# does not, so that a fix has something to flip.

_FLAT_PAIRS = [("C1", "C2"), ("C2", "C1"), ("C1", "C2TB"), ("C2TB", "C1"),
               ("C1TB", "C2"), ("C1TB", "C2TB"), ("C2TB", "C1TB")]


@pytest.mark.parametrize("lo,up", _FLAT_PAIRS)
def test_mixed_spaces_conform_on_a_flat_interface(lo, up, tmp_path):
    # On a flat interface a C1 domain and a C2 one place a refinement node in exactly the same spot: the
    # C2 side promotes its midside node, which sits at the chord midpoint, and the C1 side creates one
    # there. max_vertex_gap states that directly instead of inferring it from the absence of a crash.
    res = _solve("tri_left", "connect:%s/%s" % (lo, up), (1, 2, "estimator"), tmp_path)
    assert res["nonconforming"] == 0
    assert res["max_vertex_gap"] == 0.0, \
        "%s/%s: the two sides' interface vertices are %.3e apart" % (lo, up, res["max_vertex_gap"])


@pytest.mark.parametrize("lo,up", [("C1", "C1"), ("C2", "C2"), ("C1TB", "C1TB"), ("C2", "C2TB")])
def test_same_order_spaces_survive_a_curved_moving_interface(lo, up, tmp_path):
    # The "move" family prescribes the interface as a CURVE on both sides -- the geometry a free surface
    # produces. Two domains of the same order agree about it, whether or not they carry a bubble.
    res = _solve("quad" if lo in ("C1", "C2") else "tri_left",
                 "move:%s/%s" % (lo, up), (1, 2, "estimator"), tmp_path)
    assert res["nonconforming"] == 0
    assert res["max_vertex_gap"] == 0.0


@pytest.mark.parametrize("lo,up", [("C2", "C1"), ("C1", "C2")])
def test_mixed_order_on_a_curved_moving_interface(lo, up, tmp_path):
    # The case the topological identity exists for. On a curved interface the C2 side promotes its
    # midside node -- which sits ON the curve -- to a vertex, while the C1 side creates its new vertex at
    # the chord midpoint. The two sides refine the SAME facet and disagree about where its midpoint is,
    # so every quantised-Eulerian key in the machinery used to report "the coarser side could not be
    # refined any further". Matching on pyoomph::Node::interface_topological_id does not care where the
    # node ended up. Its negative twin is below.
    res = _solve("quad", "move:%s/%s" % (lo, up), (1, 2, "estimator"), tmp_path)
    assert res["nonconforming"] == 0


@pytest.mark.parametrize("lo,up", [("C2", "C1"), ("C1", "C2")])
def test_mixed_order_needs_the_topological_identity(lo, up, tmp_path, monkeypatch):
    # ... and with the identity switched off it must fail again, or the test above is measuring nothing.
    monkeypatch.setenv("PYOOMPH_DISABLE_TOPOLOGICAL_INTERFACE_KEYS", "1")
    with pytest.raises(RuntimeError):
        _solve("quad", "move:%s/%s" % (lo, up), (1, 2, "estimator"), tmp_path)


@pytest.mark.parametrize("spaces", ["C1,C1,C1,C2", "C2,C1,C1,C2", "C1,C1,C2,C2"])
def test_mixed_spaces_at_a_four_domain_junction_are_refused(spaces, tmp_path):
    # At a junction ConnectFieldsAtInterface gives both of a domain's interfaces the same multiplier
    # NAME, so they share one slot on the cross-point node and the multiplier enforces the SUM of the two
    # coupling conditions. That is exact when every domain there carries the same space (the remaining
    # multipliers still force each condition individually -- rank 13 of 13 shared against 13 of 14 with
    # distinct names) and WRONG as soon as one of them does not: measured max|u-y| 2.6e-2 against
    # 1.1e-16, full rank, Newton converging, nonconforming == 0. Nothing downstream can see it, so it is
    # refused rather than left to be discovered. See dev_docs/interface_refinement_coupling.md 14.4.
    with pytest.raises(RuntimeError, match="share ONE multiplier slot"):
        _solve("four_corner:" + spaces, "connect1", (1, 2, "level"), tmp_path)


@pytest.mark.parametrize("kind", two_domain_cases.FOUR_DOMAIN_KINDS)
def test_homogeneous_four_domain_junctions_are_not_refused(kind, tmp_path):
    # The negative twin of the test above: the refusal must be triggered by the SPACES at the junction,
    # not by the junction itself, or every four-domain case in the suite would stop running.
    res = _solve(kind, "connect1", (1, 2, "level"), tmp_path)
    assert res["maxuerr"] < _U_TOL


@pytest.mark.parametrize("kind", ["tri_left", "tri_crossed", "curved_tri"])
def test_c2_against_c1tb(kind, tmp_path):
    # This SEGFAULTED, on any refinement at all. get_interface_field_connection_space walked a total
    # order in which C1TB sat below C2 and returned C1TB -- but a C2 triangle has no bubble node, its
    # C1TB row of Nodal_Space_Index_To_Element_Index_Map is empty, and
    # interpolate_newly_constructed_additional_dof indexed off the end of it. C1TB and C2 are
    # incomparable; their meet is C1. See dev_docs/interface_refinement_coupling.md section 14.5.
    res = _solve(kind, "connect:C2/C1TB", (1, 2, "estimator"), tmp_path)
    assert res["nonconforming"] == 0
