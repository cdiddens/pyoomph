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
#   * Multi-space problems -- coupled C2+C1 Poisson, Taylor-Hood Stokes, ALE -- pass on all 11 layouts when
#     the mesh is non-adaptive or UNIFORMLY refined, and pass under NON-UNIFORM refinement on bricks and
#     tetrahedra, but FAIL under non-uniform refinement on wedges, pyramids and every mixed layout. A C1
#     field on a C2-geometry mesh needs its own hang slot; that per-value-index hang is installed for
#     bricks and tets but not for the wedge/pyramid/registry families.
#     The two ConstrainFieldsToC1Space variants behave exactly like the other multi-space cases: they used
#     to fail on every family, including bricks, because of a separate defect in the constraint's own
#     vertex-node guard (src/elements.cpp) -- that is fixed, and what remains is the same family
#     restriction as above.
#
# The failing configurations are marked xfail(strict=True) rather than skipped or dropped, so they stay
# visible, cannot silently rot, and will fail the suite the moment they start working (which is the signal
# to delete the marker).

import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import box_cases_3d
from box_mesh_3d import ALL_LAYOUTS, MIXED_LAYOUTS, PURE_LAYOUTS

# The full 11-layout x 3-state x 8-equation sweep is ~3.5 min: a pre-merge check, not a per-edit one.
pytestmark = pytest.mark.slow

_KINDS = ALL_LAYOUTS
_LEVELS = box_cases_3d.LEVELS

_RES_TOL = {"ale": 1e-11}
_RES_TOL_DEFAULT = 1e-9
# Same tolerance-independent Jacobian oracle as the 2D campaign. Worst observed ratio over the passing 3D
# cases is 4.5e-13 (wedge ALE), so 1e-10 keeps headroom while staying far below an inconsistent Jacobian.
_NEWTON_REDUCTION = 1e-10

# The families whose C1 hang slot is installed correctly under non-uniform refinement (see the header).
# The two ConstrainFieldsToC1Space variants belong here as well: they also carry a live C1 field alongside
# the C2 one, so once the separate defect in the constraint's own vertex-node guard was fixed they inherited
# exactly the same family restriction as the other multi-space cases.
_MULTISPACE_OK = {"hex", "tet"}
_MULTISPACE_EQS = {"mixed12", "constrain12", "unconstrain12", "stokes_th", "ale"}


def _expected_broken(eq, kind, levels):
    """Reason string if this configuration is a known failure, else None."""
    non_uniform = levels[0] != levels[1]
    if eq in _MULTISPACE_EQS and non_uniform and kind not in _MULTISPACE_OK:
        return ("two coexisting continuous spaces (C1 field on C2 geometry) under non-uniform refinement: "
                "the separate C1 hang slot is not installed for the wedge/pyramid/registry families")
    return None


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


def _run(eq, kind, levels, tmp_path, request):
    reason = _expected_broken(eq, kind, levels)
    if reason is not None:
        request.node.add_marker(pytest.mark.xfail(strict=True, reason=reason))
    cid = box_cases_3d.case_id(kind, eq, levels)
    _assert_linear_solve(_solve(kind, eq, levels, tmp_path), eq, cid)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("levels", _LEVELS)
@pytest.mark.parametrize("eq", ["poisson1", "poisson2"])
def test_poisson_baselines(eq, kind, levels, tmp_path, request):
    # Single-space baselines. These are the configurations the wedge/pyramid work was validated against,
    # so they must hold on every layout at every refinement state.
    _run(eq, kind, levels, tmp_path, request)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("levels", _LEVELS)
def test_neumann_bcs(kind, levels, tmp_path, request):
    # Neumann fluxes over face elements of every shape: a constant flux on an unrefined wall ("right") and
    # a spatially varying one on the refined wall ("top"), so the flux is integrated over sub-facets of
    # hanging-node parents of bricks, tets, wedges and pyramids alike. Passes throughout.
    _run("neumann", kind, levels, tmp_path, request)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("levels", _LEVELS)
def test_mixed_c1c2_poisson(kind, levels, tmp_path, request):
    # u on C2 driving v on C1: two continuous spaces on one mesh, so the C1 field needs its own hang slot.
    _run("mixed12", kind, levels, tmp_path, request)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("levels", _LEVELS)
def test_stokes_box_bulkforce(kind, levels, tmp_path, request):
    # Taylor-Hood Stokes driven by f = (-y, x, 0), i.e. a rotation about z: C2 velocity + C1 pressure.
    _run("stokes_th", kind, levels, tmp_path, request)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("levels", _LEVELS)
def test_ale_moving_mesh(kind, levels, tmp_path, request):
    # ALE: Stokes on a Laplace-smoothed mesh with a free top surface and a prescribed evaporation outflow,
    # so the nodal POSITIONS are unknowns too.
    reason = _expected_broken("ale", kind, levels)
    if reason is not None:
        request.node.add_marker(pytest.mark.xfail(strict=True, reason=reason))
    cid = box_cases_3d.case_id(kind, "ale", levels)
    res = _solve(kind, "ale", levels, tmp_path)
    _assert_linear_solve(res, "ale", cid)
    # The prescribed outflow must be reproduced by the flux integrated over the free surface (which has unit
    # area), certifying the moving-mesh solve did not distort the boundary condition.
    assert abs(res["obs_intuz"] - box_cases_3d.J_EVAP) < 1e-6, \
        "%s: mean outflow %.10g, expected %.10g" % (cid, res["obs_intuz"], box_cases_3d.J_EVAP)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("levels", _LEVELS)
def test_constrain_field_to_c1_space(kind, levels, tmp_path, request):
    # ConstrainFieldsToC1Space / UnconstrainFieldsFromC1Space on a C2 field coupled to a live C1 field.
    # Oracles as in 2D: it converges in one Newton step, it removes dofs, and unconstraining "top" puts
    # some back. The exact Green identity int(v)==int(u^2) is asserted only for bricks and tets: it needs u
    # and v to see the SAME discrete bilinear form, which fails for pyramids (whose geometric map is
    # rational, so the two fields' stiffness integrals are not identical even when the constraint is exact)
    # and for the mixed layouts containing them.
    reason = _expected_broken("constrain12", kind, levels)
    if reason is not None:
        request.node.add_marker(pytest.mark.xfail(strict=True, reason=reason))
    base = _solve(kind, "mixed12", levels, tmp_path)
    constrained = _solve(kind, "constrain12", levels, tmp_path)
    unconstrained = _solve(kind, "unconstrain12", levels, tmp_path)
    for eq, res in (("constrain12", constrained), ("unconstrain12", unconstrained)):
        _assert_linear_solve(res, eq, box_cases_3d.case_id(kind, eq, levels))

    assert constrained["ndof"] < unconstrained["ndof"] < base["ndof"], \
        "expected ndof(constrained) < ndof(unconstrained on top) < ndof(baseline), got %d / %d / %d" % (
            constrained["ndof"], unconstrained["ndof"], base["ndof"])

    if kind in ("hex", "tet"):
        scale = abs(constrained["obs_intu2"])
        assert abs(constrained["obs_intv"] - constrained["obs_intu2"]) < 1e-10 * scale, \
            "Green identity int(v)==int(u^2) broken under the C1 constraint: %.16g vs %.16g" % (
                constrained["obs_intv"], constrained["obs_intu2"])
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
