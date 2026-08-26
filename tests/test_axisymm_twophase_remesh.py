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

"""A pinch-off in a two-phase geometry: a liquid column on the axis inside a gas box, both built by
one template as two ``plane_surface`` calls sharing the interface spline.

What is specific to two phases, and is therefore what this file is about:

* the gap the pinch opens on the axis becomes part of the *gas* boundary. The template asks for it
  with ``get_reconnected_boundaries(..., opposite_axis_name="gas/gas_axis")``, which builds the
  opposite side as the complement of the liquid's new axis spans within everything the axis was
  covered by before - so the caller does not have to special-case the freshly opened gap, nor the
  stretches of axis between the box walls and the column that were the gas's all along;
* both domains are rebuilt, and the interface stays conforming. The two domains have separate nodes
  on it, so that is a real statement: it holds only because the liquid and the gas reference the
  same named spline, which in turn holds only because the axis corner points written as literals in
  the wall loop and the ones coming out of the plan land on the same Gmsh point.
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
_ZBOX = 1.3

pytestmark = pytest.mark.skipif(shutil.which("gmsh") is None, reason="gmsh not found")


def records(proc, kind):
    return [json.loads(line.split(" ", 1)[1]) for line in proc.stdout.splitlines()
            if line.startswith("PYOOMPH_" + kind + " ")]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    outdir = str(tmp_path_factory.mktemp("twophase") / "run")
    proc = subprocess.run([sys.executable, _WORKER, "--case", "twophase_remesh", "--outdir", outdir],
                          cwd=_HERE, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, \
        "the two-phase worker failed:\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s" % (
            proc.stdout[-4000:], proc.stderr[-4000:])
    return proc


def state(proc, tag):
    got = [r for r in records(proc, "MESH") if r["tag"] == tag]
    assert len(got) == 1, "expected one PYOOMPH_MESH record tagged %r, got %d" % (tag, len(got))
    return got[0]


def test_both_domains_were_rebuilt_around_the_pinch(run):
    rep = records(run, "REMESH_REPORT")[0]
    assert rep["has_plan"] and [e[0] for e in rep["events"]] == ["pinch"]
    liq = state(run, "after_pinch")
    gas = state(run, "after_pinch_gas")
    assert liq["n_interface"] == 2, "the liquid did not split: %d segment(s)" % liq["n_interface"]
    assert gas["n_interface"] == 2, \
        "the gas side of the interface still has %d segment(s); the two domains disagree about " \
        "the topology" % gas["n_interface"]
    for st, dom in ((liq, "liquid"), (gas, "gas")):
        assert st["n_negative"] == 0 or st["n_positive"] == 0, \
            "the rebuilt %s mesh mixes orientations" % dom
        assert st["min_abs_scaled_area"] > 1e-3, \
            "the rebuilt %s mesh has a collapsed element" % dom


def test_the_gas_takes_over_the_gap_the_pinch_opened(run):
    """Three gas axis pieces afterwards: below the column, the new gap, above the column."""
    before = state(run, "initial_gas")
    assert before["n_axis"] == 2, \
        "the gas started with %d axis segment(s), expected the two outside the column" \
        % before["n_axis"]

    rep = records(run, "REMESH_REPORT")[0]
    liquid_spans = [tuple(s) for s in rep["axis_spans"]]
    gas_spans = [tuple(s) for s in rep["opposite_axis_spans"]]
    assert len(liquid_spans) == 2 and len(gas_spans) == 3, \
        "expected 2 liquid and 3 gas axis spans, got %s and %s" % (liquid_spans, gas_spans)
    # The gap between the two fragments must be one of them, and the outer two must reach the walls.
    gap = (liquid_spans[0][1], liquid_spans[1][0])
    assert any(abs(a - gap[0]) < 1e-12 and abs(b - gap[1]) < 1e-12 for a, b in gas_spans), \
        "the freshly opened gap %s is not among the gas axis spans %s" % (gap, gas_spans)
    lo = min(a for a, _ in gas_spans)
    hi = max(b for _, b in gas_spans)
    assert abs(lo + _ZBOX) < 1e-9 and abs(hi - _ZBOX) < 1e-9, \
        "the gas axis no longer reaches the box walls: [%g, %g]" % (lo, hi)

    after = state(run, "after_pinch_gas")
    assert after["n_axis"] == 3, \
        "the rebuilt gas mesh has %d axis segment(s), not the 3 that were asked for" % after["n_axis"]
    flat_got = [v for s in after["axis_spans"] for v in s]
    flat_want = [v for s in gas_spans for v in s]
    assert flat_got == pytest.approx(flat_want, abs=1e-9), \
        "the gas axis of the rebuilt mesh does not match what define_geometry was given: %s vs %s" \
        % (after["axis_spans"], gas_spans)


def test_the_interface_stays_conforming(run):
    got = records(run, "CONFORMITY")
    assert got, "the worker did not report the interface conformity"
    got = got[0]
    assert got["n_liquid"] > 0 and got["n_gas"] > 0, \
        "one side of the interface has no nodes at all (%d liquid, %d gas), so nothing was compared" \
        % (got["n_liquid"], got["n_gas"])
    assert got["n_liquid"] == got["n_gas"], \
        "the two sides of the interface carry a different number of nodes: %d vs %d" % (
            got["n_liquid"], got["n_gas"])
    assert got["worst_distance"] < 1e-9, \
        "an interface node of the liquid is %.3g away from the nearest gas node; the two domains " \
        "were meshed on different curves" % got["worst_distance"]
