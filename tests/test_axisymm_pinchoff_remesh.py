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

"""Pinch-off delivered end to end, through GmshTemplate.define_geometry.

A dumbbell whose waist is already below ``rmin`` is detected after the first solve, and the mesh is
rebuilt from the surgery plan by the ordinary remeshing-by-recreation path. Checked afterwards:

* two interface segments, both starting and ending on the axis, and the axis boundary covering only
  the two fragments;
* no inverted or collapsed elements, and the mesh still solves;
* the volume. Two separate statements, because they fail for different reasons:

  - *inside the plan*, the fragment volumes must add up to the parent volume to round-off. That is
    what the normal-offset correction of the fresh cap points buys, and switching
    ``volume_conservation`` off must destroy it;
  - *end to end*, the volume of the interface polyline the new mesh actually carries drifts from the
    plan's target, because ``define_geometry`` turns the plan's points into a Catmull-Rom spline and
    the spline bulges outside the chords. That is an O(h^2) geometric error of the recreation path,
    not of the surgery. Measured on this dumbbell (relative drift per fragment):

        h = 0.08 -> 3.59e-3   (C = drift/h^2 = 0.56)
        h = 0.06 -> 1.76e-3   (C = 0.49)
        h = 0.04 -> 8.61e-4   (C = 0.54)
        h = 0.03 -> 4.72e-4   (C = 0.53)

    so the bound below uses C = 1.5, i.e. about three times the measured constant.

The same ``define_geometry`` branch is also exercised with nothing pending, i.e. as a plain quality
remesh: it must reproduce the boundary it was handed.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("shapely")

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "axisymm_reconnection_worker.py")

pytestmark = pytest.mark.skipif(shutil.which("gmsh") is None, reason="gmsh not found")

#: Relative volume drift allowed between the plan's target and what the rebuilt mesh carries, as a
#: multiple of h^2. Three times the constant measured above; see the module docstring.
VOLUME_DRIFT_CONSTANT = 1.5


def run_case(case, tmp_path, *extra):
    outdir = str(tmp_path / case)
    proc = subprocess.run([sys.executable, _WORKER, "--case", case, "--outdir", outdir] + list(extra),
                          cwd=_HERE, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, \
        "the %s worker failed:\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s" % (
            case, proc.stdout[-4000:], proc.stderr[-4000:])
    return proc


def records(proc, kind):
    return [json.loads(line.split(" ", 1)[1]) for line in proc.stdout.splitlines()
            if line.startswith("PYOOMPH_" + kind + " ")]


def mesh_state(proc, tag):
    got = [r for r in records(proc, "MESH") if r["tag"] == tag]
    assert len(got) == 1, "expected one PYOOMPH_MESH record tagged %r, got %d" % (tag, len(got))
    return got[0]


def remesh_report(proc, which):
    got = [r for r in records(proc, "REMESH_REPORT") if r["which"] == which]
    assert len(got) == 1, "expected one PYOOMPH_REMESH_REPORT %r, got %d" % (which, len(got))
    return got[0]


@pytest.fixture(scope="module")
def coarse(tmp_path_factory):
    return run_case("pinch_remesh", tmp_path_factory.mktemp("h006"), "--resolution", "0.06")


@pytest.fixture(scope="module")
def fine(tmp_path_factory):
    return run_case("pinch_remesh", tmp_path_factory.mktemp("h003"), "--resolution", "0.03")


@pytest.fixture(scope="module")
def unconserved(tmp_path_factory):
    return run_case("pinch_remesh", tmp_path_factory.mktemp("novc"), "--resolution", "0.06",
                    "--no-volume-conservation")


# --------------------------------------------------------------------------------------
# The no-event path through the same define_geometry
# --------------------------------------------------------------------------------------

def test_a_quality_remesh_goes_through_the_same_code_and_changes_nothing(coarse):
    rep = remesh_report(coarse, "quality")
    assert not rep["has_plan"] and rep["events"] == [], \
        "a forced remesh with nothing pending was handed a surgery plan: %s" % (rep,)
    assert rep["n_chains"] == 1 and rep["axis_spans"] == [[-1.0, 1.0]], \
        "get_reconnected_boundaries did not reproduce the current boundary: %s" % (rep,)

    before, after = mesh_state(coarse, "initial"), mesh_state(coarse, "after_quality_remesh")
    assert after["n_interface"] == before["n_interface"] == 1
    assert after["axis_spans"] == before["axis_spans"]
    drift = abs(after["volumes"][0] - before["volumes"][0]) / before["volumes"][0]
    assert drift < 1e-6, \
        "a remesh with no topological change moved the volume by %.3g relative" % drift


# --------------------------------------------------------------------------------------
# The pinch itself
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["coarse", "fine"])
def test_the_pinch_produces_two_fragments_on_the_axis(request, which):
    proc = request.getfixturevalue(which)
    rep = remesh_report(proc, "pinch")
    assert rep["has_plan"] and [e[0] for e in rep["events"]] == ["pinch"]
    st = mesh_state(proc, "after_pinch")
    assert st["n_interface"] == 2, \
        "expected two interface segments after the pinch, got %d" % st["n_interface"]
    assert st["n_axis"] == 2, \
        "the axis boundary should now consist of the two fragments' spans, got %d segment(s)" \
        % st["n_axis"]
    for ends in st["interface_ends"]:
        for x, _y in ends:
            assert abs(x) < 1e-9, \
                "an interface segment does not end on the axis (r=%g)" % x
    # The axis covers the fragments and nothing else: its spans are the interface segments' extents.
    for span, ends in zip(st["axis_spans"], st["interface_ends"]):
        assert abs(span[0] - ends[0][1]) < 1e-12 and abs(span[1] - ends[1][1]) < 1e-12, \
            "the axis span %s does not match its fragment %s" % (span, ends)


@pytest.mark.parametrize("which", ["coarse", "fine"])
def test_the_new_mesh_has_no_inverted_elements(request, which):
    proc = request.getfixturevalue(which)
    st = mesh_state(proc, "after_pinch")
    assert st["n_negative"] == 0 or st["n_positive"] == 0, \
        "the rebuilt mesh mixes orientations: %d negative and %d positive elements" % (
            st["n_negative"], st["n_positive"])
    assert st["min_abs_scaled_area"] > 1e-3, \
        "the rebuilt mesh contains a collapsed element (smallest |area| is %.3g of the largest)" \
        % st["min_abs_scaled_area"]


def test_the_new_mesh_still_solves_and_does_not_re_detect(coarse):
    """The worker solves once more after the remesh; reaching this record means it converged."""
    got = records(coarse, "RESOLVED")
    assert got, "the solve on the rebuilt mesh did not complete"
    assert not got[0]["has_plan"], \
        "the fragments the pinch just created were immediately taken for another event"


# --------------------------------------------------------------------------------------
# Volume
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["coarse", "fine"])
def test_the_plan_conserves_the_volume_exactly(request, which):
    proc = request.getfixturevalue(which)
    bal = records(proc, "PLAN_BALANCE")[0]
    assert bal["conservation"]
    rel = abs(sum(bal["after"]) - sum(bal["before"])) / sum(bal["before"])
    assert rel < 1e-12, "the surgery plan lost %.3g of the volume relatively" % rel


def test_switching_the_correction_off_is_measurably_worse(unconserved, coarse):
    """Proves the correction is live end to end rather than an accident of the geometry."""
    off = records(unconserved, "PLAN_BALANCE")[0]
    on = records(coarse, "PLAN_BALANCE")[0]
    assert not off["conservation"]
    rel_off = abs(sum(off["after"]) - sum(off["before"])) / sum(off["before"])
    rel_on = abs(sum(on["after"]) - sum(on["before"])) / sum(on["before"])
    assert rel_off > 1e-3, \
        "volume_conservation=False left a defect of only %.3g, so the correction was never " \
        "doing anything here" % rel_off
    assert rel_off > 1e6 * max(rel_on, 1e-16), \
        "the corrected (%.3g) and uncorrected (%.3g) defects are not distinguishable" % (
            rel_on, rel_off)


@pytest.mark.parametrize("which,h", [("coarse", 0.06), ("fine", 0.03)])
def test_the_rebuilt_mesh_carries_the_planned_volume(request, which, h):
    proc = request.getfixturevalue(which)
    rep = remesh_report(proc, "pinch")
    st = mesh_state(proc, "after_pinch")
    limit = max(1e-6, VOLUME_DRIFT_CONSTANT * h * h)
    for k, (target, got) in enumerate(zip(rep["fragment_volumes"], st["volumes"])):
        rel = abs(got - target) / target
        assert rel < limit, (
            "fragment %d of the rebuilt mesh has a relative volume drift of %.3g against the "
            "planned %.12g, above the %.3g allowed at h=%g" % (k, rel, target, limit, h))


def test_the_volume_drift_shrinks_with_the_mesh(coarse, fine):
    """The drift is the spline-versus-chords error of the recreation, so it must converge."""
    def worst(proc):
        rep = remesh_report(proc, "pinch")
        st = mesh_state(proc, "after_pinch")
        return max(abs(g - t) / t for t, g in zip(rep["fragment_volumes"], st["volumes"]))
    dc, df = worst(coarse), worst(fine)
    assert df < 0.5 * dc, \
        "halving h only reduced the volume drift from %.3g to %.3g; that is not second order" % (
            dc, df)
