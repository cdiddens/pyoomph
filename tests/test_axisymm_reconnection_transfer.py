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

"""Carrying the fields across a pinch-off / coalescence, i.e. the zeta continuation.

The old and the new interface are not the same curve near the event, so nothing derived from the
geometry alone maps one onto the other there. :py:class:`~pyoomph.equations.topological_changes.AxisymmetricReconnection`
therefore writes the surgery plan's chart onto both sides in ``_before_mesh_to_mesh_interpolation``,
and the transfer of the interface fields becomes the ordinary zeta-based one.

What each of these is about:

* the chart governs the INTERFACE-ONLY dofs and nothing else. A fresh cap is charted onto the old
  interface near the waist, which is where its surface field comes from but not where a bulk field
  sampled at the cap does, so the bulk fields and the position history stay on the geometric pass;
* a value must never cross the waist. With ``f = z`` on the interface, that is a statement about the
  SIGN: after the pinch the lower fragment may carry no positive value and the upper no negative one;
* away from the event windows, the chart is the old arclength, so ``f = z`` has to come back exactly
  (to the interpolation error of a C2 field on a curve that moved by O(h^2));
* the bridge of a coalescence is fresh interface between two old tips: its values must be monotone
  along it and bracketed by the two surviving values that flank it;
* the chart is a ONE-OFF. It is taken back afterwards, or the next ordinary remesh dies with
  "Boundary coordinate along interface is defined on the old, but not the new mesh";
* with nothing to detect, none of this may happen at all: the handler must leave a plain quality
  remesh exactly as it was.

The zeta invertibility check runs inside the handler on both the old and the new interface, so a
chart that folded back on itself would fail the worker rather than any assertion here.
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

#: Beyond this |z| the pinch of ``dumbbell(0.04)`` has changed nothing, so f = z must be reproduced.
AWAY_FROM_CAPS = 0.45
#: ... to this. Measured 2.2e-3 at h = 0.06, i.e. 4% of an element, and it is not an interpolation
#: error: the new interface is the morphologically OPENED old one, which differs from it everywhere
#: by the buffer approximation and by the volume correction. A chart that had stopped following the
#: old arclength would be off by a whole element instead (0.05 was measured while getting here).
AWAY_TOLERANCE = 5e-3


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


def field_state(proc, tag):
    got = [r for r in records(proc, "FIELD") if r["tag"] == tag]
    assert len(got) == 1, "expected one PYOOMPH_FIELD record tagged %r, got %d" % (tag, len(got))
    return got[0]["segments"]


def bulk_state(proc, tag):
    got = [r for r in records(proc, "BULK") if r["tag"] == tag]
    assert len(got) == 1, "expected one PYOOMPH_BULK record tagged %r, got %d" % (tag, len(got))
    return got[0]


@pytest.fixture(scope="module")
def quality(tmp_path_factory):
    return run_case("transfer_quality", tmp_path_factory.mktemp("tq"))


@pytest.fixture(scope="module")
def pinch(tmp_path_factory):
    return run_case("transfer_pinch", tmp_path_factory.mktemp("tp"))


@pytest.fixture(scope="module")
def user_zeta(tmp_path_factory):
    return run_case("transfer_user_zeta", tmp_path_factory.mktemp("uz"))


@pytest.fixture(scope="module")
def coalescence(tmp_path_factory):
    return run_case("transfer_coalescence", tmp_path_factory.mktemp("tc"))


@pytest.fixture(scope="module")
def twophase(tmp_path_factory):
    return run_case("transfer_twophase", tmp_path_factory.mktemp("t2"))


# --------------------------------------------------------------------------------------
# A plain quality remesh must not notice the handler at all
# --------------------------------------------------------------------------------------

def test_a_quality_remesh_is_left_alone_by_the_handler(quality):
    interp = records(quality, "INTERP")
    assert interp, "no interpolator ran"
    for rec in interp:
        assert rec["zeta_overridden"] == [] and rec["bulk_locate"] == [] \
            and rec["zeta_iface_only"] == [] and rec["max_dist"] == [], \
            "the handler configured the transfer although there was no event: %s" % (rec,)


def test_the_interface_field_still_transfers_without_the_handler_doing_anything(quality):
    segs = field_state(quality, "after_quality_remesh")
    assert len(segs) == 1
    worst = max(abs(f - y) for _x, y, f in segs[0])
    assert worst < AWAY_TOLERANCE, \
        "the interface field lost %.3g in a plain quality remesh" % worst


# --------------------------------------------------------------------------------------
# The pinch
# --------------------------------------------------------------------------------------

def test_the_handler_configures_both_the_chart_and_the_axis(pinch):
    rec = records(pinch, "INTERP")[0]
    assert rec["zeta_overridden"] == ["liquid/interface"], \
        "the chart was not registered as taken over: %s" % (rec,)
    assert rec["zeta_iface_only"] == ["interface"], \
        "the chart was not restricted to the interface-only dofs: %s" % (rec,)
    assert rec["bulk_locate"] == ["axis"], \
        "the axis was not put on the bulk-locate path: %s" % (rec,)
    # 2 * max(rmin_nd, distmin_nd), nondimensional: boundary_max_dist is compared against distances
    # between nodal positions, which pyoomph stores nondimensionally.
    assert dict(rec["max_dist"]) == {"axis/interface": 0.24, "interface/axis": 0.24}, \
        "the codimension-2 reach around the fresh tip is wrong: %s" % (rec,)


def test_no_interface_value_crosses_the_waist(pinch):
    segs = field_state(pinch, "after_pinch")
    assert len(segs) == 2, "expected two fragments, got %d" % len(segs)
    lower, upper = segs
    worst_lo = max(f for _x, _y, f in lower)
    worst_hi = min(f for _x, _y, f in upper)
    assert worst_lo <= 1e-6, \
        "the lower fragment carries f = %.6g, which can only have come from above the waist" % worst_lo
    assert worst_hi >= -1e-6, \
        "the upper fragment carries f = %.6g, which can only have come from below the waist" % worst_hi


def test_the_chart_reproduces_the_old_field_away_from_the_caps(pinch):
    for k, seg in enumerate(field_state(pinch, "after_pinch")):
        far = [abs(f - y) for _x, y, f in seg if abs(y) > AWAY_FROM_CAPS]
        assert far, "fragment %d has no node away from the event window" % k
        assert max(far) < AWAY_TOLERANCE, \
            "fragment %d: the chart is off by %.3g where the surgery changed nothing" % (k, max(far))


def test_the_cap_takes_its_values_from_its_own_side_of_the_waist(pinch):
    """The fresh cap is charted onto the old interface between the splice and the waist, so its
    values interpolate between the value at the splice and the value at the waist - and not, as a
    geometric match would have it, whatever happens to be closest across the gap."""
    # The window's own edge is only bounded to within the transfer's accuracy, hence the slack.
    edge = AWAY_FROM_CAPS + AWAY_TOLERANCE
    lower, upper = field_state(pinch, "after_pinch")
    cap = [f for _x, y, f in lower if y > -AWAY_FROM_CAPS]
    assert len(cap) >= 3, "the lower cap window has only %d nodes" % len(cap)
    assert max(cap) < 0.0 and min(cap) > -edge, \
        "the lower cap values %s are not inside the old range between the splice and the waist" % (cap,)
    cap = [f for _x, y, f in upper if y < AWAY_FROM_CAPS]
    assert min(cap) > 0.0 and max(cap) < edge, \
        "the upper cap values %s are not inside the old range between the splice and the waist" % (cap,)


def test_the_bulk_field_stays_continuous_across_the_pinch(pinch):
    before, after = bulk_state(pinch, "initial"), bulk_state(pinch, "after_pinch")
    assert after["n_nonfinite"] == 0
    assert after["worst"] < 0.05, \
        "a bulk node is %.3g off 1 + z after the pinch (at %s); the chart has leaked into the bulk " \
        "fields, which it must not" % (after["worst"], after["worst_at"])
    assert before["worst"] < 1e-10
    assert after["umin"] >= before["umin"] - 1e-9 and after["umax"] <= before["umax"] + 1e-9, \
        "the transfer produced bulk values outside the range the old field had"


def test_an_ordinary_remesh_after_the_event_still_works(pinch):
    """The chart the handler installs is a one-off and has to be taken back again: a mesh left
    claiming a boundary coordinate that nothing re-establishes makes the NEXT remesh raise."""
    rep = [r for r in records(pinch, "REMESH_REPORT") if r["which"] == "after_event_quality"]
    assert rep, "the ordinary remesh after the event never happened"
    assert not rep[0]["has_plan"] and rep[0]["events"] == []
    assert records(pinch, "RESOLVED"), "the run did not get past the second remesh"
    after = bulk_state(pinch, "after_second_remesh")
    assert after["n_nonfinite"] == 0 and after["worst"] < 0.05


# --------------------------------------------------------------------------------------
# Coexisting with a user's own zeta assigner
# --------------------------------------------------------------------------------------

def test_the_handlers_chart_wins_over_a_user_zeta_assigner(user_zeta, pinch):
    """An AssignZetaCoordinatesByArclength on the same interface renormalises per segment, so its
    chart maps the lower fragment's [0,1] onto the WHOLE old interface, waist included. It must not
    get a say on the event step - and it re-charts the interface in
    ``after_mapping_on_macro_elements``, which runs after the interpolation hooks and only on the
    NEW mesh, so "the handler wrote its chart last" is not enough by itself. Measured while getting
    here: with the assigner winning that race, the fragments came out at f in -1 .. -0.37 and
    0.54 .. 1.0 and were off by 0.15 and 0.34 where the surgery had changed nothing."""
    with_user = field_state(user_zeta, "after_pinch")
    without = field_state(pinch, "after_pinch")
    assert len(with_user) == 2
    for k, (a, b) in enumerate(zip(with_user, without)):
        assert max(abs(f - y) for _x, y, f in a) == pytest.approx(
            max(abs(f - y) for _x, y, f in b), abs=1e-9), \
            "fragment %d was transferred differently just because a zeta assigner is attached" % k
    assert max(f for _x, _y, f in with_user[0]) <= 1e-6
    assert min(f for _x, _y, f in with_user[1]) >= -1e-6


def test_the_user_assigner_gets_its_boundary_back_afterwards(user_zeta):
    """Its chart has to be the one standing again once the transfer is over, or the ordinary remesh
    that follows finds a boundary coordinate on the old mesh and none on the new one."""
    rep = [r for r in records(user_zeta, "REMESH_REPORT") if r["which"] == "after_event_quality"]
    assert rep and not rep[0]["has_plan"]
    segs = field_state(user_zeta, "after_second_remesh")
    assert len(segs) == 2
    for k, seg in enumerate(segs):
        far = [abs(f - y) for _x, y, f in seg if abs(y) > AWAY_FROM_CAPS]
        assert max(far) < AWAY_TOLERANCE, \
            "fragment %d lost the field in the ordinary remesh after the event (worst %.3g)" % (
                k, max(far))


# --------------------------------------------------------------------------------------
# The coalescence bridge
# --------------------------------------------------------------------------------------

def test_the_two_fragments_became_one(coalescence):
    segs = field_state(coalescence, "after_coalescence")
    assert len(segs) == 1, "expected one fragment after the coalescence, got %d" % len(segs)
    assert len(field_state(coalescence, "initial")) == 2


def test_the_bridge_interpolates_monotonically_between_the_two_old_tips(coalescence):
    seg = field_state(coalescence, "after_coalescence")[0]
    fs = [f for _x, _y, f in seg]
    steps = [b - a for a, b in zip(fs, fs[1:])]
    assert min(steps) > 0.0, \
        "the transferred field is not monotone along the merged fragment (worst step %.3g)" % min(steps)

    # The bridge is the stretch the surgery created, i.e. where f is no longer the node's own z.
    bridge = [i for i, (_x, y, f) in enumerate(seg) if abs(f - y) > 1e-9]
    assert bridge, "nothing on the merged interface came from anywhere but its own position"
    lo, hi = min(bridge), max(bridge)
    assert lo > 0 and hi < len(seg) - 1, "the bridge reaches the ends of the fragment"
    below, above = seg[lo - 1][2], seg[hi + 1][2]
    for i in bridge:
        assert below <= seg[i][2] <= above, \
            "bridge node %d carries f = %.6g, outside the surviving values %.6g .. %.6g that flank " \
            "the bridge" % (i, seg[i][2], below, above)


# --------------------------------------------------------------------------------------
# Two phases
# --------------------------------------------------------------------------------------

def test_both_sides_of_the_interface_get_the_chart(twophase):
    by_domain = {r["domain"]: r for r in records(twophase, "INTERP")}
    assert set(by_domain) == {"liquid", "gas"}, "expected one interpolator per phase, got %s" % (
        sorted(by_domain),)
    assert by_domain["liquid"]["zeta_overridden"] == ["liquid/interface"]
    assert by_domain["gas"]["zeta_overridden"] == ["gas/interface"], \
        "the handler sits on the liquid interface only and did not reach the gas domain's " \
        "interpolator: %s" % (by_domain["gas"],)
    assert by_domain["gas"]["bulk_locate"] == ["gas_axis"]
    assert dict(by_domain["gas"]["max_dist"]) == {"gas_axis/interface": 0.24,
                                                  "interface/gas_axis": 0.24}


def test_the_gas_side_surface_field_transfers_like_the_liquid_one(twophase):
    liq = field_state(twophase, "after_pinch")
    gas = field_state(twophase, "after_pinch_gas")
    assert len(gas) == len(liq) == 2
    for k, (lseg, gseg) in enumerate(zip(liq, gas)):
        far = [abs(f - y) for _x, y, f in gseg if abs(y) > AWAY_FROM_CAPS]
        assert far and max(far) < AWAY_TOLERANCE, \
            "the gas side of fragment %d lost the field away from the caps (worst %.3g)" % (
                k, max(far) if far else float("nan"))
        assert max(f for _x, _y, f in gseg) * max(f for _x, _y, f in lseg) > 0, \
            "the gas side of fragment %d took its values from the other side of the waist" % k


def test_the_freshly_opened_gas_axis_gets_sane_values(twophase):
    """The pinch hands the gas a stretch of axis that used to be liquid. Nothing on the old gas mesh
    describes it, so those nodes go down the bulk-locate path and, where even that finds nothing, the
    nearest-node fallback. What must hold is that they come out finite and inside the range the old
    gas field had - not that they reproduce 1 + z there."""
    rec = records(twophase, "GAS_GAP")[0]
    assert rec["n"] >= 3, "the pinch opened no gas axis to speak of (%d nodes)" % rec["n"]
    assert rec["n_nonfinite"] == 0, "a fresh gas axis node came out non-finite"
    before = bulk_state(twophase, "initial_gas")
    for y, u in rec["values"]:
        assert before["umin"] - 1e-9 <= u <= before["umax"] + 1e-9, \
            "the fresh gas axis node at z = %.6g got u = %.6g, outside the range %.6g .. %.6g the " \
            "old gas field had" % (y, u, before["umin"], before["umax"])
