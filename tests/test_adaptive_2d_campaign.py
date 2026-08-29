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

# Serial half of the 2D adaptive-mesh campaign on branch mixed_adapt: the physics users actually run --
# mixed C1/C2 spaces (Taylor-Hood Stokes), the C1-space field constraints, Neumann fluxes and ALE moving
# meshes -- exercised on adaptive QUAD, TRIANGLE and genuinely MIXED quad+tri meshes, at a non-adaptive, a
# uniform and a TWO-LEVEL non-uniform (2:1 hanging) refinement state.
#
# The problem definitions live in box_cases.py and are shared verbatim with the MPI harness
# (test_mpi_adaptivity.py), so the serial and distributed campaigns cannot drift apart.
#
# Oracles used here, in increasing order of strength:
#   * max|residual| ~ 0 after the solve -- necessary, but it only says the solve converged.
#   * one Newton step removes essentially the whole residual -- these problems are linear, so an exact
#     analytic Jacobian takes the residual from O(1e-2) to machine zero in a single step. This is the real
#     test of the Jacobian: a hanging or constrained dof that was pinned instead of being given a
#     registered hang makes the Jacobian inconsistent with the residual, and the reduction collapses.
#     Expressed as a RATIO rather than an iteration count, which would be tolerance-dependent (see the
#     note in box_cases.solve_case).
#   * cross-discretisation agreement of a global integral -- a torn or mis-hung interface changes the FIELD,
#     which the two checks above would not notice.
#   * exact discrete identities (the dof counts of the C1 constraint; the Green duality int(v)==int(u^2)).

import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import box_cases

# Part of the validation campaign, so the wheel builds deselect it with -m "not campaign" (see
# conftest.py). Deliberately NOT marked "slow": at ~30 s it stays in the fast local run.
pytestmark = pytest.mark.campaign

_KINDS = box_cases.MESH_KINDS
_LEVELS = box_cases.LEVELS

# Machine-zero bound for a converged linear solve. Loosened only for Crouzeix-Raviart, which is markedly
# worse conditioned than Taylor-Hood, and for ALE, whose residual carries the mesh-position dofs too.
_RES_TOL = {"stokes_cr": 1e-7, "ale": 1e-12}
_RES_TOL_DEFAULT = 1e-9

# One Newton step must reduce the residual by at least this factor. Measured worst case over every
# non-Crouzeix-Raviart case in this file (8 equation systems x 4 meshes x 3 refinement states) is 1.5e-13,
# so 1e-10 leaves three orders of headroom while sitting eight or more orders below anything an
# inconsistent Jacobian produces -- that degrades to roughly linear convergence, i.e. a ratio near 1.
_NEWTON_REDUCTION = 1e-10

# Crouzeix-Raviart is exempt from the reduction test and only has to converge within a few iterations.
# Its saddle-point system on refined triangle meshes is ill-conditioned enough that the direct linear solve
# is itself inaccurate: one step takes the residual from 8.4e-04 to 1.5e-05 and the second one to ~1e-10.
# This is not a fault of the hanging-node Jacobian -- it reproduces identically with the pre-existing
# (pre-mixed_adapt) pressure fixation, and the equivalent TH cases on the same meshes reduce by 1e-14.
# The conditioning interacts with the SOLVER, though, and calling it a property of the discretisation
# alone was too generous: MKL Pardiso pivots statically and returns these systems with backward errors
# of order 1e0 (Newton then fails outright, which is what solve_case's repair_bad_solves addresses),
# while a dynamically pivoting solver reaches 1e-14 -- see _use_pivoting_solver in
# tests/test_triangle_refinement.py.
_NO_REDUCTION_TEST = {"stokes_cr"}
_MAX_NEWTON_STEPS = 3


def _solve(kind, eq, levels, tmp_path):
    cid = box_cases.case_id(kind, eq, levels)
    return box_cases.solve_case(kind, eq, levels, outdir=str(tmp_path / cid))


def _assert_linear_solve(res, eq, cid):
    tol = _RES_TOL.get(eq, _RES_TOL_DEFAULT)
    assert res["maxres"] < tol, "%s: max|residual| = %.3e" % (cid, res["maxres"])
    conv = res["newton_conv"]
    assert len(conv) >= 2, "%s: no Newton iteration was performed (history %r)" % (cid, conv)
    assert conv[0] > 1e-6, "%s: initial residual %.3e is too small for the reduction test to mean " \
                           "anything" % (cid, conv[0])
    if eq in _NO_REDUCTION_TEST:
        assert res["newton_steps"] <= _MAX_NEWTON_STEPS, \
            "%s: Newton took %d steps (history %r)" % (cid, res["newton_steps"], conv)
        return
    reduction = conv[1] / conv[0]
    assert reduction < _NEWTON_REDUCTION, \
        "%s: one Newton step only reduced the residual by %.2e (%.3e -> %.3e) -- for a linear problem " \
        "that means the analytic Jacobian does not match the residual (a hanging/constrained dof pinned " \
        "instead of hung?)" % (cid, reduction, conv[0], conv[1])


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("levels", _LEVELS)
@pytest.mark.parametrize("eq", ["stokes_th", "stokes_cr"])
def test_stokes_box_bulkforce(eq, kind, levels, tmp_path):
    # Stokes in [-0.5,0.5]^2 driven by f = (-y, x), i.e. MIXED continuous spaces on one adaptive mesh:
    # Taylor-Hood is C2 velocity + C1 pressure (the C1 pressure owns a separate hang slot and must hang
    # linearly on the coarse edge corners, including at a coarse edge mid-node whose velocity is a real
    # dof); Crouzeix-Raviart is bubble-enriched velocity + element-internal discontinuous pressure.
    cid = box_cases.case_id(kind, eq, levels)
    _assert_linear_solve(_solve(kind, eq, levels, tmp_path), eq, cid)


def test_stokes_box_agrees_across_discretisations(tmp_path):
    # The strong check for the MIXED mesh: the same physical problem solved on pure quads, on two different
    # triangle splits and on a mixed quad+tri mesh must give the same global angular momentum. Refinement
    # must make them AGREE BETTER, which is what rules out a tear at the quad<->tri interface -- a torn or
    # mis-hung interface leaves free nodes and shifts the integral instead of converging it.
    def spread(levels):
        vals = [_solve(k, "stokes_th", levels, tmp_path)["obs_intcurl"] for k in _KINDS]
        return (max(vals) - min(vals)) / abs(sum(vals) / len(vals)), vals

    coarse, _ = spread((0, 0))
    fine, vals = spread((1, 3))
    assert fine < 1e-2, "discretisations disagree by %.2e on the refined mesh: %r" % (fine, vals)
    assert fine < coarse / 5, \
        "refinement did not bring the discretisations together (%.2e -> %.2e); suspect an interface tear" \
        % (coarse, fine)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("levels", _LEVELS)
@pytest.mark.parametrize("eq", ["poisson1", "poisson2", "mixed12"])
def test_poisson_and_mixed_c1c2(eq, kind, levels, tmp_path):
    # Baselines: pure C1, pure C2, and u on C2 driving v on C1 on the same adaptive mesh (the two spaces
    # hang independently, each with its own hang slot).
    cid = box_cases.case_id(kind, eq, levels)
    _assert_linear_solve(_solve(kind, eq, levels, tmp_path), eq, cid)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("levels", _LEVELS)
def test_constrain_field_to_c1_space(kind, levels, tmp_path):
    # ConstrainFieldsToC1Space / UnconstrainFieldsFromC1Space on a C2 field coupled to a C1 field.
    # Three independent oracles, so a constraint that is silently inert (or silently wrong) cannot pass:
    #   1. it converges in one Newton step -> the constrained dofs got a registered hang, not a pin;
    #   2. it actually REMOVES dofs, and unconstraining "top" puts some back;
    #   3. once u is restricted to the C1 space it lives in the same discrete space as v, so the Green
    #      identity int(v) == int(u^2) becomes EXACT. It is not exact for the unconstrained baseline, so
    #      this genuinely distinguishes a correct restriction from an approximate one.
    base = _solve(kind, "mixed12", levels, tmp_path)
    constrained = _solve(kind, "constrain12", levels, tmp_path)
    unconstrained = _solve(kind, "unconstrain12", levels, tmp_path)
    for eq, res in (("mixed12", base), ("constrain12", constrained), ("unconstrain12", unconstrained)):
        _assert_linear_solve(res, eq, box_cases.case_id(kind, eq, levels))

    assert constrained["ndof"] < unconstrained["ndof"] < base["ndof"], \
        "expected ndof(constrained) < ndof(unconstrained on top) < ndof(baseline), got %d / %d / %d" % (
            constrained["ndof"], unconstrained["ndof"], base["ndof"])

    scale = abs(constrained["obs_intu2"])
    assert abs(constrained["obs_intv"] - constrained["obs_intu2"]) < 1e-10 * scale, \
        "Green identity int(v)==int(u^2) broken under the C1 constraint: %.16g vs %.16g" % (
            constrained["obs_intv"], constrained["obs_intu2"])
    # ... and it is a real discriminator, not a triviality: without the constraint it does NOT hold.
    assert abs(base["obs_intv"] - base["obs_intu2"]) > 1e-6 * abs(base["obs_intu2"])


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("levels", _LEVELS)
def test_neumann_bcs(kind, levels, tmp_path):
    # Neumann fluxes on an adaptive/mixed mesh: a constant flux on "right" (which mixes refined and
    # unrefined face parents) and a spatially varying one on "top" (whose parents are all refined, since
    # "top" carries the refinement band). The face elements must integrate over the correct sub-facets of
    # hanging-node parents of BOTH shapes.
    cid = box_cases.case_id(kind, "neumann", levels)
    _assert_linear_solve(_solve(kind, "neumann", levels, tmp_path), "neumann", cid)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("levels", _LEVELS)
def test_ale_constrain_positions_to_c1(kind, levels, tmp_path):
    # ConstrainPositionsToC1Space / UnconstrainPositionsFromC1Space on the moving mesh: the position
    # analogue of test_constrain_field_to_c1_space. At a 2:1 T-junction the constrained position
    # redistributes onto its C1 corners, which -- because c1_constraint_corners are written by whichever
    # element sees the node as a NON-vertex -- can be vertices of a NEIGHBOURING element that oomph-lib
    # never registered as position-hang masters here. That is what used to abort.
    base = _solve(kind, "ale", levels, tmp_path)
    constrained = _solve(kind, "ale_posc1", levels, tmp_path)
    unconstrained = _solve(kind, "ale_posc1_unc", levels, tmp_path)
    for eq, res in (("ale_posc1", constrained), ("ale_posc1_unc", unconstrained)):
        _assert_linear_solve(res, "ale", box_cases.case_id(kind, eq, levels))
    assert constrained["ndof"] < unconstrained["ndof"] < base["ndof"], \
        "expected ndof(constrained) < ndof(unconstrained on top) < ndof(baseline), got %d / %d / %d" % (
            constrained["ndof"], unconstrained["ndof"], base["ndof"])
    # the prescribed outflow must survive the position constraint
    assert abs(constrained["obs_intuy"] - box_cases.J_EVAP) < 1e-6


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("levels", [(0, 0), (1, 1), (1, 3)])
def test_ale_moving_mesh(kind, levels, tmp_path):
    # ALE: Stokes on a Laplace-smoothed mesh with a free top surface and a prescribed outflow standing in
    # for evaporation. The nodal POSITIONS are unknowns, so the hanging-node machinery has to handle the
    # position dofs as well as the fields.
    cid = box_cases.case_id(kind, "ale", levels)
    res = _solve(kind, "ale", levels, tmp_path)
    _assert_linear_solve(res, "ale", cid)
    # The prescribed outflow is reproduced by the integrated flux through the free surface (the top edge
    # has unit length), which certifies that the moving-mesh solve did not distort the boundary condition.
    assert abs(res["obs_intuy"] - box_cases.J_EVAP) < 1e-6, \
        "%s: mean outflow %.10g, expected %.10g" % (cid, res["obs_intuy"], box_cases.J_EVAP)
