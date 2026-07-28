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
