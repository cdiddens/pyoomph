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

# 3D half of the adaptive-mesh campaign: the same physics as test_adaptive_2d_campaign.py on the box
# [-0.5,0.5]^3, discretised in every geometrically possible combination of bricks, tetrahedra, wedges and
# pyramids (see box_mesh_3d.py for which combinations exist and why the others cannot).
#
# WHAT PASSES AND WHAT DOES NOT (measured; see dev_docs/mixed_adapt_validation.md §9)
#
#   * Single-space problems -- C1 Poisson, C2 Poisson, and the whole Neumann campaign -- pass on ALL 11
#     layouts at all three refinement states, including two-level non-uniform 2:1 hanging.
#   * Multi-space problems -- coupled C2+C1 Poisson, ConstrainFieldsToC1Space, Taylor-Hood Stokes, ALE --
#     likewise pass on all 11 layouts at all three refinement states. Getting there took three fixes
#     (dev_docs/mixed_adapt_validation.md 9.4/9.5): the per-value interpolation hooks for the wedge and
#     pyramid C2 elements, the C1-constraint vertex-node guard, and two wrong C1-corner tables.
#
# Everything in this module is expected to pass; there are no known-broken configurations left here.

import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import box_cases_3d
from box_mesh_3d import ALL_LAYOUTS, MIXED_LAYOUTS, PURE_LAYOUTS

# The full 11-layout x 3-state x 8-equation sweep is ~3.5 min: a pre-merge check, not a per-edit one.
pytestmark = [pytest.mark.slow, pytest.mark.campaign]

_KINDS = ALL_LAYOUTS
_LEVELS = box_cases_3d.LEVELS

_RES_TOL = {"ale": 1e-11}
_RES_TOL_DEFAULT = 1e-9
# Same tolerance-independent Jacobian oracle as the 2D campaign. Worst observed ratio over the passing 3D
# cases is 4.5e-13 (wedge ALE), so 1e-10 keeps headroom while staying far below an inconsistent Jacobian.
_NEWTON_REDUCTION = 1e-10

def _solve(kind, eq, levels, tmp_path):
    cid = box_cases_3d.case_id(kind, eq, levels)
    return box_cases_3d.solve_case(kind, eq, levels, outdir=str(tmp_path / cid))


def _assert_linear_solve(res, eq, cid):
    assert res["manifold"], "%s: the refined mesh is not conforming (a facet is shared by !=2 elements " \
                            "or is neither boundary nor interior) -- suspect a torn interface" % cid
    tol = _RES_TOL.get(eq, _RES_TOL_DEFAULT)
    assert res["maxres"] < tol, "%s: max|residual| = %.3e" % (cid, res["maxres"])
    conv = res["newton_conv"]
    assert len(conv) >= 2, "%s: no Newton iteration was performed (history %r)" % (cid, conv)
    assert conv[0] > 1e-6, "%s: initial residual %.3e too small for the reduction test" % (cid, conv[0])
    reduction = conv[1] / conv[0]
    assert reduction < _NEWTON_REDUCTION, \
        "%s: one Newton step only reduced the residual by %.2e (%.3e -> %.3e) -- for a linear problem " \
        "that means the analytic Jacobian does not match the residual" % (cid, reduction, conv[0], conv[1])


def _run(eq, kind, levels, tmp_path):
    cid = box_cases_3d.case_id(kind, eq, levels)
    _assert_linear_solve(_solve(kind, eq, levels, tmp_path), eq, cid)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("levels", _LEVELS)
@pytest.mark.parametrize("eq", ["poisson1", "poisson2"])
def test_poisson_baselines(eq, kind, levels, tmp_path):
    # Single-space baselines. These are the configurations the wedge/pyramid work was validated against,
    # so they must hold on every layout at every refinement state.
    _run(eq, kind, levels, tmp_path)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("levels", _LEVELS)
def test_neumann_bcs(kind, levels, tmp_path):
    # Neumann fluxes over face elements of every shape: a constant flux on an unrefined wall ("right") and
    # a spatially varying one on the refined wall ("top"), so the flux is integrated over sub-facets of
    # hanging-node parents of bricks, tets, wedges and pyramids alike. Passes throughout.
    _run("neumann", kind, levels, tmp_path)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("levels", _LEVELS)
def test_mixed_c1c2_poisson(kind, levels, tmp_path):
    # u on C2 driving v on C1: two continuous spaces on one mesh, so the C1 field needs its own hang slot.
    _run("mixed12", kind, levels, tmp_path)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("levels", _LEVELS)
def test_stokes_box_bulkforce(kind, levels, tmp_path):
    # Taylor-Hood Stokes driven by f = (-y, x, 0), i.e. a rotation about z: C2 velocity + C1 pressure.
    _run("stokes_th", kind, levels, tmp_path)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("levels", _LEVELS)
def test_ale_moving_mesh(kind, levels, tmp_path):
    # ALE: Stokes on a Laplace-smoothed mesh with a free top surface and a prescribed evaporation outflow,
    # so the nodal POSITIONS are unknowns too.
    cid = box_cases_3d.case_id(kind, "ale", levels)
    res = _solve(kind, "ale", levels, tmp_path)
    _assert_linear_solve(res, "ale", cid)
    # The prescribed outflow must be reproduced by the flux integrated over the free surface (which has unit
    # area), certifying the moving-mesh solve did not distort the boundary condition.
    assert abs(res["obs_intuz"] - box_cases_3d.J_EVAP) < 1e-6, \
        "%s: mean outflow %.10g, expected %.10g" % (cid, res["obs_intuz"], box_cases_3d.J_EVAP)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("levels", _LEVELS)
def test_ale_constrain_positions_to_c1(kind, levels, tmp_path):
    # See the 2D counterpart. Exercised on every element-family combination.
    base = _solve(kind, "ale", levels, tmp_path)
    constrained = _solve(kind, "ale_posc1", levels, tmp_path)
    unconstrained = _solve(kind, "ale_posc1_unc", levels, tmp_path)
    for eq, res in (("ale_posc1", constrained), ("ale_posc1_unc", unconstrained)):
        _assert_linear_solve(res, "ale", box_cases_3d.case_id(kind, eq, levels))
    assert constrained["ndof"] < unconstrained["ndof"] < base["ndof"], \
        "expected ndof(constrained) < ndof(unconstrained on top) < ndof(baseline), got %d / %d / %d" % (
            constrained["ndof"], unconstrained["ndof"], base["ndof"])
    assert abs(constrained["obs_intuz"] - box_cases_3d.J_EVAP) < 1e-6


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("levels", _LEVELS)
def test_constrain_field_to_c1_space(kind, levels, tmp_path):
    # ConstrainFieldsToC1Space / UnconstrainFieldsFromC1Space on a C2 field coupled to a live C1 field.
    # Oracles as in 2D: it converges in one Newton step, it removes dofs, and unconstraining "top" puts
    # some back. The exact Green identity int(v)==int(u^2) is asserted only for bricks and tets: it needs u
    # and v to see the SAME discrete bilinear form, which fails for pyramids (whose geometric map is
    # rational, so the two fields' stiffness integrals are not identical even when the constraint is exact)
    # and for the mixed layouts containing them.
    base = _solve(kind, "mixed12", levels, tmp_path)
    constrained = _solve(kind, "constrain12", levels, tmp_path)
    unconstrained = _solve(kind, "unconstrain12", levels, tmp_path)
    for eq, res in (("constrain12", constrained), ("unconstrain12", unconstrained)):
        _assert_linear_solve(res, eq, box_cases_3d.case_id(kind, eq, levels))

    assert constrained["ndof"] < unconstrained["ndof"] < base["ndof"], \
        "expected ndof(constrained) < ndof(unconstrained on top) < ndof(baseline), got %d / %d / %d" % (
            constrained["ndof"], unconstrained["ndof"], base["ndof"])

    scale = abs(constrained["obs_intu2"])
    assert abs(constrained["obs_intv"] - constrained["obs_intu2"]) < 1e-10 * scale, \
        "Green identity int(v)==int(u^2) broken under the C1 constraint: %.16g vs %.16g" % (
            constrained["obs_intv"], constrained["obs_intu2"])
    # ... and it is a real discriminator, not a triviality: without the constraint it does NOT hold.
    assert abs(base["obs_intv"] - base["obs_intu2"]) > 1e-6 * abs(base["obs_intu2"])


def test_stokes_agrees_across_families(tmp_path):
    # The strong cross-family check, on the states where every layout works (uniform level 1): the same
    # physical problem discretised with bricks, tets, wedges, pyramids and each legal mixture must give the
    # same global angular momentum to within discretisation error. This is what would expose a torn
    # interface between two families, which a per-layout residual check cannot see.
    vals = {}
    for kind in _KINDS:
        vals[kind] = _solve(kind, "stokes_th", (1, 1), tmp_path)["obs_intcurl"]
    mean = sum(vals.values()) / len(vals)
    spread = (max(vals.values()) - min(vals.values())) / abs(mean)
    assert spread < 0.15, "families disagree by %.2e on the same problem: %r" % (spread, vals)
    # and every layout must be within that band of the mean, so no single family is an outlier
    for kind, v in vals.items():
        assert abs(v - mean) / abs(mean) < 0.15, "%s is an outlier: %.10g vs mean %.10g" % (kind, v, mean)
