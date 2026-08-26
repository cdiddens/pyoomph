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

"""Unit tests for pyoomph.meshes.axisymm_topology.

Deliberately free of any pyoomph Problem, mesh or gmsh: the module under test is pure
geometry, so these run in a fraction of a second and need only numpy, scipy and shapely.
"""

import math
import pickle

import numpy as np
import pytest

pytest.importorskip("shapely")


def _load_module():
    """Import the module under test, also before the package has been reinstalled.

    A brand-new file inside the pyoomph package is not visible through the editable
    install until the next ./build_for_develop.sh, so fall back on loading it straight
    from the source tree.
    """
    try:
        import pyoomph.meshes.axisymm_topology as m
        return m
    except Exception:
        import importlib.util
        import sys
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent / "pyoomph" / "meshes" / \
            "axisymm_topology.py"
        spec = importlib.util.spec_from_file_location("_axisymm_topology_standalone", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod


axt = _load_module()
InterfaceChain = axt.InterfaceChain
detect_and_plan = axt.detect_and_plan
revolved_volume = axt.revolved_volume


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _chain(points, size=0.05, ends=("axis", "axis")):
    p = np.asarray(points, dtype=float)
    return InterfaceChain(p, np.full(len(p), float(size)), ends, None)


def _sphere(zc, R, n=241):
    th = np.linspace(0.0, math.pi, n)
    p = np.stack([R * np.sin(th), zc - R * np.cos(th)], axis=1)
    p[0, 0] = 0.0
    p[-1, 0] = 0.0
    return p


def _cosine_jet(r0, a, L, n, npts):
    """A Rayleigh-Plateau-like thread held between two walls at z=0 and z=L.

    The profile has its maxima at the two walls and exactly ``n`` interior minima of
    radius ``r0-a``, so with ``rmin_nd`` between ``r0-a`` and the next larger radius the
    expected outcome is unambiguous: ``n`` pinch events and ``n+1`` fragments.
    """
    z = np.linspace(0.0, L, npts)
    r = r0 + a * np.cos(2.0 * math.pi * n * z / L)
    return np.stack([r, z], axis=1)


def _band_volume(points, za, zb):
    """Revolved volume of the part of a z-monotone half-section between two z-planes.

    Independent of the module under test apart from the (separately tested) quadrature.
    """
    p = np.asarray(points, dtype=float)
    z, r = p[:, 1], p[:, 0]
    keep = (z >= za) & (z <= zb)
    seq = []
    if za > z[0]:
        seq.append((float(np.interp(za, z, r)), za))
    seq.extend((float(rr), float(zz)) for rr, zz in zip(r[keep], z[keep]))
    if zb < z[-1]:
        seq.append((float(np.interp(zb, z, r)), zb))
    seq.append((0.0, float(min(zb, z[-1]))))
    seq.append((0.0, float(max(za, z[0]))))
    return revolved_volume(np.array(seq, dtype=float))


def _old_concat(chains):
    return np.vstack([c.points for c in chains])


def _check_identity_and_zeta(plan, chains):
    old = _old_concat(chains)
    for nc in plan.new_chains:
        keep = nc.origin >= 0
        assert np.array_equal(nc.points[keep], old[nc.origin[keep]])
        assert np.all(np.diff(nc.zeta) > 0.0)
        assert len(nc.points) == len(nc.sizes) == len(nc.zeta) == len(nc.origin)
        assert np.all(nc.points[:, 0] >= 0.0)


def _tip_angles(plan):
    """Angle (deg) between the axis and the last segment at every fresh axial tip."""
    out = []
    for nc in plan.new_chains:
        for i, j in ((0, 1), (-1, -2)):
            if abs(float(nc.points[i, 0])) > 1e-12 or nc.origin[i] >= 0:
                continue
            d = nc.points[j] - nc.points[i]
            out.append(math.degrees(math.atan2(abs(d[0]), abs(d[1]))))
    return out


# --------------------------------------------------------------------------------------
# revolved_volume
# --------------------------------------------------------------------------------------

def test_revolved_volume_sphere():
    R = 1.3
    V = revolved_volume(_sphere(0.0, R, n=4001))
    assert abs(V - 4.0 / 3.0 * math.pi * R ** 3) < 1e-3


def test_revolved_volume_cylinder_and_orientation():
    cyl = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]])
    assert abs(revolved_volume(cyl) - math.pi * 4.0 * 3.0) < 1e-12
    assert abs(revolved_volume(cyl[::-1]) - math.pi * 4.0 * 3.0) < 1e-12


# --------------------------------------------------------------------------------------
# pinch-off
# --------------------------------------------------------------------------------------

def test_cosine_jet_pinches_once_per_neck():
    n = 2
    L, r0, a = 4.0, 0.5, 0.44          # necks of radius 0.06 at z = 1 and z = 3
    rmin = 0.08
    pts = _cosine_jet(r0, a, L, n, 1601)
    chains = [_chain(pts, ends=("fixed", "fixed"))]
    plan = detect_and_plan(chains, rmin, None, volume_tolerance=1e-9)
    assert plan is not None

    kinds = [e.kind for e in plan.events]
    assert kinds == ["pinch"] * n
    assert len(plan.new_chains) == n + 1
    for e, zc in zip(plan.events, (1.0, 3.0)):
        assert abs(e.z_center - zc) < 0.05
        assert "zeta_waist" in e.zeta_info

    _check_identity_and_zeta(plan, chains)

    # The plan must split the parent volume exactly, band by band.
    tot = plan.fragment_volumes_before[0]
    assert abs(sum(plan.fragment_volumes_after) - tot) <= 1e-9 * tot
    cuts = [0.0] + [e.z_center for e in plan.events] + [L]
    for i, V in enumerate(plan.fragment_volumes_after):
        assert abs(V - _band_volume(pts, cuts[i], cuts[i + 1])) <= 1e-9 * tot

    # the two end fragments keep their wall, the middle one is capped on both sides
    assert plan.new_chains[0].end_types == ("fixed", "axis")
    assert plan.new_chains[1].end_types == ("axis", "axis")
    assert plan.new_chains[2].end_types == ("axis", "fixed")


def test_liquid_bridge_between_two_walls():
    pts = _cosine_jet(0.5, 0.44, 2.0, 1, 801)
    chains = [_chain(pts, ends=("fixed", "fixed"))]
    plan = detect_and_plan(chains, 0.08, None)
    assert plan is not None
    assert [e.kind for e in plan.events] == ["pinch"]
    assert len(plan.new_chains) == 2
    lo, hi = plan.new_chains
    assert lo.end_types == ("fixed", "axis")
    assert hi.end_types == ("axis", "fixed")
    # each fragment keeps its own wall point untouched
    assert np.array_equal(lo.points[0], pts[0])
    assert np.array_equal(hi.points[-1], pts[-1])
    assert abs(lo.points[-1, 0]) == 0.0 and abs(hi.points[0, 0]) == 0.0
    tot = plan.fragment_volumes_before[0]
    assert abs(sum(plan.fragment_volumes_after) - tot) <= 1e-9 * tot
    _check_identity_and_zeta(plan, chains)


def test_pinch_tips_meet_the_axis_perpendicularly():
    pts = _cosine_jet(0.5, 0.44, 4.0, 2, 1601)
    plan = detect_and_plan([_chain(pts, ends=("fixed", "fixed"))], 0.08, None)
    angles = _tip_angles(plan)
    assert len(angles) == 4                     # two caps per waist
    for ang in angles:
        assert abs(ang - 90.0) < 2.0


# --------------------------------------------------------------------------------------
# coalescence
# --------------------------------------------------------------------------------------

def test_two_caps_coalesce():
    gap = 0.05
    a = _sphere(0.0, 1.0)
    b = _sphere(2.0 + gap, 1.0)
    chains = [_chain(a), _chain(b)]
    plan = detect_and_plan(chains, None, 0.1)   # distmin = 0.1 > gap
    assert plan is not None
    assert [e.kind for e in plan.events] == ["coalescence"]
    assert len(plan.new_chains) == 1
    ev = plan.events[0]
    assert abs(ev.z_center - (1.0 + 0.5 * gap)) < 0.1
    assert ev.zeta_info["zeta_lower_tip"] < ev.zeta_info["zeta_upper_tip"]

    want = sum(plan.fragment_volumes_before)
    assert abs(plan.fragment_volumes_after[0] - want) <= 1e-9 * want
    _check_identity_and_zeta(plan, chains)

    # one contiguous fluid span on the axis, nothing outside it
    assert plan.axis_spans_inside.shape == (1, 2)
    assert plan.axis_spans_outside.shape == (0, 2)


def test_coalescence_is_not_triggered_above_distmin():
    a = _sphere(0.0, 1.0)
    b = _sphere(2.0 + 0.2, 1.0)
    assert detect_and_plan([_chain(a), _chain(b)], None, 0.1) is None


# --------------------------------------------------------------------------------------
# combined
# --------------------------------------------------------------------------------------

def _peanut(n=801):
    """A dumbbell with a 0.05-radius waist at z=0 and smooth axial tips at z=+-2."""
    z = np.linspace(-2.0, 2.0, n)
    s = z / 2.0
    r = np.sqrt(np.clip(1.0 - s * s, 0.0, None)) * (0.05 + 0.95 * s * s)
    p = np.stack([r, z], axis=1)
    p[0, 0] = 0.0
    p[-1, 0] = 0.0
    return p


def test_neck_and_gap_in_one_call_do_not_interfere():
    peanut = _peanut()
    upper = _sphere(3.05, 1.0)          # bottom at z = 2.05, i.e. a gap of 0.05
    chains = [_chain(peanut), _chain(upper)]
    plan = detect_and_plan(chains, 0.08, 0.1)
    assert plan is not None

    kinds = sorted(e.kind for e in plan.events)
    assert kinds == ["coalescence", "pinch"]
    # lower half of the peanut on its own; upper half fused with the sphere
    assert len(plan.new_chains) == 2
    pinch = [e for e in plan.events if e.kind == "pinch"][0]
    coal = [e for e in plan.events if e.kind == "coalescence"][0]
    assert abs(pinch.z_center) < 0.1
    assert abs(coal.z_center - 2.025) < 0.1
    assert coal.parents == [0, 1]

    tot = sum(plan.fragment_volumes_before)
    assert abs(sum(plan.fragment_volumes_after) - tot) <= 1e-9 * tot
    # the pinch children of the peanut are symmetric, so the lower one is exactly half
    assert abs(plan.fragment_volumes_after[0]
               - 0.5 * plan.fragment_volumes_before[0]) <= 1e-9 * tot
    _check_identity_and_zeta(plan, chains)
    for ang in _tip_angles(plan):
        assert abs(ang - 90.0) < 2.0


# --------------------------------------------------------------------------------------
# no event
# --------------------------------------------------------------------------------------

def test_no_event_returns_none_and_leaves_input_alone():
    a, b = _sphere(0.0, 1.0), _sphere(4.0, 1.0)
    chains = [_chain(a), _chain(b)]
    before = [c.points.copy() for c in chains]
    assert detect_and_plan(chains, 0.5, 0.1) is None
    for c, p in zip(chains, before):
        assert np.array_equal(c.points, p)


def test_no_criteria_returns_none():
    assert detect_and_plan([_chain(_sphere(0.0, 1.0))], None, None) is None
    assert detect_and_plan([], 0.1, 0.1) is None


# --------------------------------------------------------------------------------------
# bookkeeping
# --------------------------------------------------------------------------------------

def test_zeta_chart_and_axis_spans():
    pts = _cosine_jet(0.5, 0.44, 4.0, 2, 1601)
    chains = [_chain(pts, ends=("fixed", "fixed"))]
    plan = detect_and_plan(chains, 0.08, None)

    # the old chart is filled in and is the plain cumulative arclength
    seg = np.linalg.norm(np.diff(chains[0].points, axis=0), axis=1)
    assert np.allclose(chains[0].zeta, np.concatenate([[0.0], np.cumsum(seg)]))

    # every fresh cap zeta stays strictly inside the old chart range of its own side of
    # the waist, so an old-mesh lookup can never land across the waist
    zw = [e.zeta_info["zeta_waist"] for e in plan.events]
    lo, mid, hi = plan.new_chains
    assert np.all(lo.zeta[lo.origin < 0] < zw[0])
    assert np.all(hi.zeta[hi.origin < 0] > zw[1])
    fresh_mid = mid.zeta[mid.origin < 0]
    assert np.all((fresh_mid > zw[0]) | (fresh_mid < zw[1]))

    # the axis is covered by the three fragments and the two pinched gaps, nothing else
    ins, outs = plan.axis_spans_inside, plan.axis_spans_outside
    assert ins.shape == (3, 2) and outs.shape == (2, 2)
    covered = sorted([tuple(s) for s in ins] + [tuple(s) for s in outs])
    assert abs(covered[0][0] - 0.0) < 1e-12
    assert abs(covered[-1][1] - 4.0) < 1e-12
    for (a0, a1), (b0, b1) in zip(covered[:-1], covered[1:]):
        assert abs(a1 - b0) < 1e-12


def test_plan_is_picklable():
    pts = _cosine_jet(0.5, 0.44, 2.0, 1, 801)
    plan = detect_and_plan([_chain(pts, ends=("fixed", "fixed"))], 0.08, None)
    clone = pickle.loads(pickle.dumps(plan))
    assert len(clone.new_chains) == len(plan.new_chains)
    assert np.array_equal(clone.new_chains[0].points, plan.new_chains[0].points)
    assert clone.events[0].kind == "pinch"


def test_reversed_input_chain_is_normalized():
    pts = _cosine_jet(0.5, 0.44, 2.0, 1, 801)
    chains = [_chain(pts[::-1].copy(), ends=("fixed", "fixed"))]
    plan = detect_and_plan(chains, 0.08, None)
    assert plan is not None
    assert chains[0].points[0, 1] < chains[0].points[-1, 1]
    _check_identity_and_zeta(plan, chains)


# --------------------------------------------------------------------------------------
# fragment removal
# --------------------------------------------------------------------------------------

def _satellite_case():
    big = _sphere(0.0, 1.0)
    tiny = _sphere(5.0, 0.03, n=61)
    return [_chain(big), _chain(tiny, size=0.01)]


def test_fragment_removal_records_the_lost_volume():
    chains = _satellite_case()
    plan = detect_and_plan(chains, 0.08, None, allow_fragment_removal=True)
    assert plan is not None
    assert [e.kind for e in plan.events] == ["removal"]
    assert plan.events[0].parents == [1]
    want = revolved_volume(chains[1].points)
    assert abs(plan.volume_lost_by_removal - want) < 1e-12
    assert len(plan.new_chains) == 1
    # the surviving droplet is untouched
    assert np.array_equal(plan.new_chains[0].points, chains[0].points)


def test_fragment_removal_can_be_refused():
    with pytest.raises(RuntimeError, match="allow_fragment_removal"):
        detect_and_plan(_satellite_case(), 0.08, None, allow_fragment_removal=False)


# --------------------------------------------------------------------------------------
# unsupported topologies / guards
# --------------------------------------------------------------------------------------

def _polyline(corners, per_edge=40):
    c = np.asarray(corners, dtype=float)
    parts = [np.linspace(c[i], c[i + 1], per_edge)[:-1] for i in range(len(c) - 1)]
    return np.vstack(parts + [c[-1:]])


def test_entrapment_of_the_opposite_phase_raises():
    bowl = _polyline([[0, 0], [1, 0], [1, 1], [0.8, 1], [0.8, 0.2], [0, 0.2]])
    lid = _polyline([[0, 1.05], [1, 1.05], [1, 1.15], [0, 1.15]])
    with pytest.raises(RuntimeError, match="hole"):
        detect_and_plan([_chain(bowl), _chain(lid)], None, 0.1)


def test_event_close_to_a_fixed_end_raises():
    # a flat lens welded to a wall at z=0, with a droplet hovering 0.05 above its tip
    z = np.linspace(0.0, 0.1, 201)
    r = np.sqrt(np.clip(1.0 - (z / 0.1) ** 2, 0.0, None))
    r[-1] = 0.0
    lens = _chain(np.stack([r, z], axis=1), ends=("fixed", "axis"))
    drop = _chain(_sphere(0.65, 0.5, n=201))
    with pytest.raises(RuntimeError, match="4\\*eps"):
        detect_and_plan([lens, drop], None, 0.1)


def test_axially_short_neck_is_reported_not_silently_missed():
    z = np.linspace(0.0, 2.0, 1601)
    r = 0.5 - 0.45 * np.exp(-((z - 0.3) / 0.08) ** 2)
    ch = _chain(np.stack([r, z], axis=1), ends=("fixed", "fixed"))
    with pytest.raises(RuntimeError, match="axially shorter"):
        detect_and_plan([ch], 0.08, None)


def test_pinch_gap_shorter_than_distmin_raises():
    # the opening carves a ~0.08-long gap at each waist; asking to bridge anything below
    # 0.5 would immediately undo the pinch, which is a contradictory parameter choice
    pts = _cosine_jet(0.5, 0.44, 2.0, 1, 801)
    with pytest.raises(RuntimeError, match="re-bridges"):
        detect_and_plan([_chain(pts, ends=("fixed", "fixed"))], 0.08, 0.5)


def test_missing_shapely_message(monkeypatch):
    import builtins
    monkeypatch.setattr(axt, "_SHAPELY", None)
    real_import = builtins.__import__

    def fake(name, *args, **kwargs):
        if name.startswith("shapely"):
            raise ImportError("no shapely")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(RuntimeError, match="pip install shapely"):
        axt._require_shapely()
