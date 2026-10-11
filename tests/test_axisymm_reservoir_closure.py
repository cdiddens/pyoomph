#  @file
#  @author Christian Diddens
#  @author Duarte Rocha
#  @author Maxim de Wildt
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

"""InterfaceChain.reservoir: what is behind a wall contact line.

A ``"fixed"`` chain end is closed synthetically, and the plain closure runs straight across at the
contact height. That is right for a drop cut by a symmetry plane, where there is nothing behind the
contact, and wrong for a nozzle meniscus, where the liquid continues up the nozzle: the closure then
cuts the reservoir off and what the morphology sees is the sliver in front of the interface.

Two failure modes follow, both of them from a perfectly ordinary meniscus and neither of them
anything to do with a topological change:

* a meniscus **dimpled** near the axis leaves a sliver thinner than ``2 rmin``, which the erosion
  either erases entirely or - worse - hollows into a ring, whose mirrored cross section does not
  touch the axis and cannot be represented at all;
* a meniscus that **crosses** the contact height is cut by its own closure, i.e. the cross section
  self-intersects.

``reservoir`` says how deep the liquid goes behind the contact line and along which direction, and
the closure then follows the wall instead. These are direct calls of ``detect_and_plan``, no mesh
and no Problem: it is the plan that is under test.
"""

import numpy
import pytest

pytest.importorskip("shapely")

from pyoomph.meshes.axisymm_topology import (InterfaceChain, InterfaceStateNotPlannable,
                                             _closed_section, _closure_linestring, _halfplane,
                                             _interface_curve, _open, _polygons, _require_shapely,
                                             _ring_coords, detect_and_plan, revolved_volume)

#: Nozzle radius, wall contact at (1, 0), liquid at y > 0 behind it.
_RES = 3.0


def _chain(points, end_types, reservoir=(None, None)):
    pts = numpy.asarray(points, dtype=float)
    spacing = numpy.concatenate([[0.0], numpy.linalg.norm(numpy.diff(pts, axis=0), axis=1)])
    spacing[0] = spacing[1]
    return InterfaceChain(points=pts, sizes=spacing, end_types=end_types, reservoir=reservoir)


def _meniscus(axis_offset, hump=0.34, n=109):
    """A meniscus from the wall contact (1,0) to the axis, humped by ``hump`` in between.

    ``axis_offset`` is where it meets the axis: positive is retracted into the nozzle, negative is
    protruding out of it. Either way the hump means the interface goes *above* the contact height,
    which is what the plain closure cannot survive.
    """
    r = numpy.linspace(1.0, 0.0, n)
    y = hump * numpy.sin(numpy.pi * r) + axis_offset * (1.0 - r)
    return numpy.stack([r, y], axis=1)


def _ligament_with_neck(neck=0.03):
    """A drop on a long slender neck, hanging out of the nozzle and still attached at (1,0).

    The waist is deliberately several times ``rmin`` long: a shorter one is refused as not yet
    separable (see WaistNotYetSeparable), which is a different thing from what is under test here.
    """
    y = numpy.concatenate([numpy.linspace(-4.0, -2.0, 140)[:-1], numpy.linspace(-2.0, 0.0, 120)])
    sphere = numpy.sqrt(numpy.maximum(0.0, 0.64 - (y + 3.2) ** 2))    # drop, tip at y = -4
    r = numpy.where(y <= -2.7, sphere, numpy.maximum(sphere, neck))
    s = numpy.clip((y + 2.0) / 2.0, 0.0, 1.0)
    r = numpy.where(y > -2.0, neck + (1.0 - neck) * numpy.sqrt(s), r)  # out to the nozzle rim
    return numpy.stack([r, y], axis=1)


# ------------------------------------------------------------------------------------------------
# 1. the two failure modes of the plain closure, and that the reservoir removes them
# ------------------------------------------------------------------------------------------------

def test_a_humped_meniscus_needs_the_reservoir():
    # A meniscus that reaches below its own contact height is cut by the plain closure. (The other
    # mode - a dimple thin enough to be eroded away - no longer gets this far: the opening leaves a
    # band around every closure alone, see detect_and_plan.)
    plain = _chain(_meniscus(-0.05), ("fixed", "axis"))
    try:
        plan = detect_and_plan([plain], 0.025, 0.02)
    except RuntimeError:
        plan = "raised"
    assert plan is not None, "the plain closure was expected to fail on a humped meniscus"


@pytest.mark.parametrize("axis_offset", [0.02, -0.05, 0.4])
def test_the_reservoir_leaves_a_quiet_meniscus_alone(axis_offset):
    ch = _chain(_meniscus(axis_offset), ("fixed", "axis"), reservoir=(_RES, None))
    assert detect_and_plan([ch], 0.025, 0.02) is None


def test_the_reservoir_volume_is_the_liquid_behind_the_interface():
    # A flat meniscus closed against a reservoir of depth 3 encloses the nozzle itself, pi*R^2*3 -
    # not the sliver of zero volume the flat closure would give. This is the section every volume
    # target in the plan is measured against, so it is worth pinning directly.
    pts = _meniscus(0.0, hump=0.0)
    assert revolved_volume(_closed_section(pts, ("fixed", "axis"), (_RES, None))) \
        == pytest.approx(numpy.pi * _RES, rel=1e-12)
    assert revolved_volume(_closed_section(pts, ("fixed", "axis"))) == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------------------------------------
# 2. a real pinch below the nozzle, with the reservoir in place
# ------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pinch():
    ch = _chain(_ligament_with_neck(), ("axis", "fixed"), reservoir=(None, _RES))
    plan = detect_and_plan([ch], 0.08, None)
    assert plan is not None
    return plan


def test_the_neck_below_the_nozzle_still_pinches(pinch):
    assert [e.kind for e in pinch.events] == ["pinch"]
    assert len(pinch.new_chains) == 2


def test_the_wall_contact_point_survives_the_surgery(pinch):
    anchored = [nc for nc in pinch.new_chains if "fixed" in nc.end_types]
    assert len(anchored) == 1
    nc = anchored[0]
    end = nc.points[0] if nc.end_types[0] == "fixed" else nc.points[-1]
    assert end == pytest.approx([1.0, 0.0], abs=1e-9)


def test_the_reservoir_is_inherited_and_not_turned_into_interface(pinch):
    anchored = [nc for nc in pinch.new_chains if "fixed" in nc.end_types][0]
    side = 0 if anchored.end_types[0] == "fixed" else 1
    assert anchored.reservoir[side] == pytest.approx(_RES)
    # The closure runs at x = 1 up to y = 3; if it had been left in the run, the new interface would
    # reach up there. It stops at the rim instead.
    assert anchored.points[:, 1].max() < 0.5
    free = [nc for nc in pinch.new_chains if "fixed" not in nc.end_types][0]
    assert free.reservoir == (None, None)


def test_the_axis_span_of_the_anchored_fragment_reaches_up_the_nozzle(pinch):
    # The span has to run to where the CLOSURE meets the axis, not to the contact line: a mesh
    # rebuilt from spans that stop at the rim leaves the nozzle above the meniscus outside every
    # span, and the curve loop of the liquid then cannot be closed at all.
    spans = numpy.asarray(pinch.axis_spans_inside).reshape(-1, 2)
    assert float(spans.max()) == pytest.approx(_RES, abs=1e-9)
    lo, hi = spans[numpy.argmax(spans[:, 1])]
    assert hi == pytest.approx(_RES, abs=1e-9) and lo < -2.0    # the fresh cap of the anchored side


def test_the_freed_drop_keeps_its_volume(pinch):
    free = [nc for nc in pinch.new_chains if "fixed" not in nc.end_types][0]
    # The sphere of radius 0.8 it was cut from, to within what the cap resampling can hold.
    assert revolved_volume(free.points) == pytest.approx(4.0 / 3.0 * numpy.pi * 0.8 ** 3, rel=0.05)


# ------------------------------------------------------------------------------------------------
# 3. the band the opening leaves around a closure covers the L of a reservoir end too
# ------------------------------------------------------------------------------------------------

def _gas_finger(n=137, rwall=1.0, tip=3.0):
    """A meniscus pinned at the rim (1,0) and retracted to an axial tip, with a film on the wall.

    This is the printhead state measured on 2026-10-03: gas drawn into the bore leaves liquid
    between the interface and the wall, and that film is thinner than ``2*rmin`` at both of its
    ends - at the rim, where it goes to zero by construction, and again further in. Without a band
    around the closure the opening takes the middle of it out as a free-standing annulus, whose
    mirrored cross section does not touch the axis and cannot be represented.
    """
    z = numpy.linspace(0.0, tip, n)
    r = rwall - 0.17 * numpy.sin(numpy.pi * numpy.clip(z / (0.78 * tip), 0, 1)) ** 0.6
    cap = z > 0.87 * tip
    r[cap] = r[cap] * numpy.sqrt(numpy.clip(1.0 - ((z[cap] - 0.87 * tip) / (0.13 * tip)) ** 2, 0, 1))
    r[0] = rwall
    r[-1] = 0.0
    return numpy.stack([r, z], axis=1)


def test_a_wall_film_is_not_cut_into_a_ring():
    ch = _chain(_gas_finger(), ("fixed", "axis"), reservoir=(14.375, None))
    assert detect_and_plan([ch], 0.05, 0.02) is None


def test_the_film_would_be_cut_into_a_ring_without_the_band():
    # The premise of the test above: without the protect band the very same state opens into three
    # components - the body, the ring, and the ring's mirror twin - so the band is what is doing
    # the work and not some other property of the state.
    sh = _require_shapely()
    ch = _chain(_gas_finger(), ("fixed", "axis"), reservoir=(14.375, None))
    unprotected = _polygons(_open(sh.unary_union([sh.Polygon(_ring_coords(ch))]), 0.05, 8))
    assert len(unprotected) == 3
    off_axis = [q for q in unprotected if q.bounds[0] > 1e-9 or q.bounds[2] < -1e-9]
    assert len(off_axis) == 2, "the ring and its mirror twin, neither of them crossing the axis"


def test_the_band_follows_the_closure_the_volume_is_measured_against():
    # The whole bug was two spellings of "where is this closure" drifting apart, so pin that the
    # one the band is built from is the one _ring_coords draws and _closed_section measures.
    sh = _require_shapely()
    for reservoir in (None, _RES):
        pts = _meniscus(0.0, hump=0.0)
        ends = ("fixed", "axis")
        ch = _chain(pts, ends, reservoir=(reservoir, None))
        closure = _closure_linestring(sh, ch.points[0], ch.reservoir[0])
        ring = sh.Polygon(_ring_coords(ch))
        assert closure.within(ring.buffer(1e-9)), "the closure must lie on the section it closes"
        assert float(closure.bounds[3]) == pytest.approx(
            float(pts[0, 1]) + (reservoir or 0.0), abs=1e-12)


def test_an_off_axis_fragment_is_refused_as_toroidal_and_is_deferrable():
    # If a ring ever does reach the half-plane extraction by some other route, it must say what is
    # actually wrong with it - it does not touch the axis - rather than "a fragment vanished", and
    # it must be the deferrable type, because a transient can pass through such a state.
    sh = _require_shapely()
    ring = sh.box(0.4, 0.0, 0.9, 1.0)        # an off-axis fragment of the half plane
    with pytest.raises(InterfaceStateNotPlannable, match="does not touch the symmetry axis"):
        _interface_curve(ring, 1e-9, 0.05)


def test_the_mirror_twin_of_an_off_axis_fragment_clips_to_nothing():
    # ... and that is how the half-plane extraction recognises it: the twin lies wholly at x < 0,
    # so its upper x bound is at or below the snap tolerance. This is the predicate the
    # "vanished"/"toroidal" branch turns on.
    sh = _require_shapely()
    clip = sh.box(0.0, -10.0, 10.0, 10.0)
    twin = sh.box(-0.9, 0.0, -0.4, 1.0)
    assert _halfplane(twin, clip, 1e-9) is None
    assert float(twin.bounds[2]) <= 1e-9
