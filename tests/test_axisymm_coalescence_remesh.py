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

"""Coalescence delivered end to end: two stacked spheroids whose facing tips are 0.4*distmin apart
become one fragment spanning both, with the merged volume of its parents.

Same tolerance reasoning as tests/test_axisymm_pinchoff_remesh.py: the plan is exact to round-off,
and what the rebuilt mesh carries drifts from it by the O(h^2) spline-versus-chords error of the
recreation path. The bridge that replaces the two tips is a low-curvature region, so the drift here
(5.8e-5 relative at h=0.06, i.e. C = 0.016) is far below the bound the pinch needs.
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
_H = 0.06

pytestmark = pytest.mark.skipif(shutil.which("gmsh") is None, reason="gmsh not found")


def records(proc, kind):
    return [json.loads(line.split(" ", 1)[1]) for line in proc.stdout.splitlines()
            if line.startswith("PYOOMPH_" + kind + " ")]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    outdir = str(tmp_path_factory.mktemp("coalesce") / "run")
    proc = subprocess.run([sys.executable, _WORKER, "--case", "coalescence_remesh",
                           "--resolution", str(_H), "--outdir", outdir],
                          cwd=_HERE, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, \
        "the coalescence worker failed:\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s" % (
            proc.stdout[-4000:], proc.stderr[-4000:])
    return proc


def state(proc, tag):
    got = [r for r in records(proc, "MESH") if r["tag"] == tag]
    assert len(got) == 1, "expected one PYOOMPH_MESH record tagged %r, got %d" % (tag, len(got))
    return got[0]


def test_the_starting_point_really_is_two_fragments(run):
    st = state(run, "initial")
    assert st["n_interface"] == 2 and st["n_axis"] == 2, \
        "the case did not start from two separate blobs: %s" % (st,)


def test_the_two_fragments_become_one(run):
    rep = records(run, "REMESH_REPORT")[0]
    assert rep["has_plan"] and [e[0] for e in rep["events"]] == ["coalescence"], \
        "expected exactly one coalescence event, got %s" % (rep["events"],)
    st = state(run, "after_coalescence")
    assert st["n_interface"] == 1, \
        "expected one interface segment after the merge, got %d" % st["n_interface"]
    assert st["n_axis"] == 1, \
        "the two axis spans should have become one, got %d" % st["n_axis"]
    ends = st["interface_ends"][0]
    for x, _y in ends:
        assert abs(x) < 1e-9, "the merged interface does not end on the axis (r=%g)" % x
    before = state(run, "initial")
    assert abs(ends[0][1] - before["interface_ends"][0][0][1]) < 1e-9, \
        "the merged fragment does not start at the lower parent's outer tip"
    assert abs(ends[1][1] - before["interface_ends"][1][1][1]) < 1e-9, \
        "the merged fragment does not end at the upper parent's outer tip"


def test_the_merged_volume_is_the_sum_of_the_parents(run):
    before = state(run, "initial")
    rep = records(run, "REMESH_REPORT")[0]
    parents = sum(before["volumes"])
    assert len(rep["fragment_volumes"]) == 1
    planned = rep["fragment_volumes"][0]
    assert abs(planned - parents) < 1e-12 * parents, \
        "the plan's merged volume %.12g is not the sum of the parents %.12g" % (planned, parents)

    after = state(run, "after_coalescence")
    rel = abs(after["volumes"][0] - planned) / planned
    assert rel < max(1e-6, 1.5 * _H * _H), \
        "the rebuilt mesh drifted by %.3g relative from the planned merged volume" % rel


def test_the_merged_mesh_is_sound(run):
    st = state(run, "after_coalescence")
    assert st["n_negative"] == 0 or st["n_positive"] == 0, \
        "the rebuilt mesh mixes orientations: %d negative, %d positive" % (
            st["n_negative"], st["n_positive"])
    assert st["min_abs_scaled_area"] > 1e-3, \
        "the rebuilt mesh has a collapsed element (%.3g of the largest)" % st["min_abs_scaled_area"]
    got = records(run, "RESOLVED")
    assert got and not got[0]["has_plan"], \
        "the merged fragment was immediately taken for another event, or the follow-up solve failed"
