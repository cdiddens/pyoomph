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

"""Two droplets merging, in a real Navier-Stokes free-surface solve.

The counterpart of ``tests/test_rayleigh_plateau_pinchoff.py``: there the interface loses a fragment,
here it gains one, and the two halves of the surgery are different enough to be worth both. A
coalescence has to build a *bridge* out of nothing between two old axial tips, and the zeta chart has
to send the two halves of that bridge back to the two tips they came from rather than to one of them.

Two equal droplets approach ballistically on the axis - each is given a uniform initial velocity, and
there is no ambient phase to slow them down. Deliberately so: what is under test is the event and the
continuation past it, and a driving mechanism with a force balance of its own would only add something
else that can fail. Until they touch, the flow is a rigid translation, so the volume up to the merge is
also a clean measurement of the free-surface kinematic condition on its own.

Asserted: one fragment afterwards, the merged volume equal to the sum of the parents to the tolerance
the surgery actually achieves, no inverted elements, and twenty accepted steps past the merge. The
volume is measured the way the pinch-off module measures it - per step, split by what happened in that
step - but with its own bounds; see the constants below for why they are looser.

The second half of the module is the **two-phase** variant: the same merge with a gas box around it,
both domains rebuilt from the same ``TopologicalChangesGmshTemplate``. Coalescence rather than a pinch
there too, and deliberately: a dumbbell short enough to be affordable with a gas domain around it is
*Rayleigh stable* (at neck 0.25, half length 1.5 the neck stopped thinning at 0.22 and sat there for
400 steps), while the merge is reached in a handful of steps and gives the gas the harder job anyway -
it *loses* the stretch of axis between the two droplets.
"""

import os
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("shapely")

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "axisymm_physics_worker.py")

pytestmark = [pytest.mark.skipif(shutil.which("gmsh") is None, reason="gmsh not found"),
              pytest.mark.slow]

from test_rayleigh_plateau_pinchoff import records, one, volume_changes, worst  # noqa: E402

#: Bounds on the relative volume change of one step, by kind. Looser than the pinch module's, and for
#: a reason worth stating: a coalescence bridge is by far the most strongly curved piece of interface
#: either scenario ever carries, and the O(h^2 * curvature) sag of the spline the mesh generator draws
#: through the plan's points is correspondingly larger there. Measured (relative):
#:
#:                        single phase   two phase
#:   worst ordinary step     -6.3e-5      -1.4e-4  (the two-phase start is impulsive; see the worker)
#:   worst quality remesh    -8.8e-5      -3.7e-5
#:   THE EVENT STEP          -4.0e-4      -1.3e-5
#:   whole run               -8.2e-4      -4.0e-4
#:
#: i.e. still an order of magnitude below the 0.5 % the plan asked for, but not the 1e-5 of the
#: pinch-off, and the difference is geometry rather than a defect.
MAX_ORDINARY_STEP = 5e-4
MAX_QUALITY_REMESH = 5e-4
MAX_EVENT_STEP = 2e-3
MAX_TOTAL_DRIFT = 4e-3


def run_case(case, tmp_path, *extra):
    outdir = str(tmp_path / case)
    # Single-threaded, so that the run is reproducible. The transient ends in a capillary singularity,
    # where two trajectories that differ in the last bits separate within a few dozen steps; with the
    # BLAS and Gmsh free to thread, two runs of the same worker took visibly different paths through
    # the event and the test passed or failed accordingly. With this (and General.NumThreads=1 in the
    # template) successive runs are bit-identical.
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, _WORKER, "--case", case, "--outdir", outdir] + list(extra),
                          cwd=_HERE, capture_output=True, text=True, timeout=3600, env=env)
    assert proc.returncode == 0, \
        "the %s worker failed:\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s" % (
            case, proc.stdout[-6000:], proc.stderr[-4000:])
    return proc


@pytest.fixture(scope="module")
def merge(tmp_path_factory):
    return run_case("coalescence", tmp_path_factory.mktemp("coal"))


def test_the_droplets_merge_and_the_run_continues(merge):
    done = one(merge, "DONE")
    assert done["merged"], "the two droplets never merged within the step budget"
    assert done["post_steps"] >= 20, \
        "only %d steps were taken after the merge" % done["post_steps"]


def test_two_fragments_before_and_one_after(merge):
    steps = records(merge, "STEP")
    event = one(merge, "EVENT")
    before = [s for s in steps if s["t"] < event["t"]]
    after = [s for s in steps if s["t"] >= event["t"]]
    assert before and all(s["n_fragments"] == 2 for s in before)
    assert all(s["n_fragments"] == 1 for s in after)


def test_the_gap_closed_rather_than_the_tips_passing_through(merge):
    """The event must happen while the two tips still face each other across a gap.

    Two tips that pass through one another produce a self-intersecting cross section, which the
    detection refuses - so this is really a statement about the time step being small enough for the
    threshold to be seen at all.
    """
    steps = records(merge, "STEP")
    event = one(merge, "EVENT")
    setup = one(merge, "SETUP")
    last_before = [s for s in steps if s["t"] < event["t"]][-1]
    assert 0.0 < last_before["gap"] < 2.0 * setup["distmin"]


def test_merged_volume_is_the_sum_of_the_parents(merge):
    steps = records(merge, "STEP")
    event = one(merge, "EVENT")
    before = [s for s in steps if s["t"] < event["t"]][-1]
    at = [s for s in steps if s["t"] == event["t"]][0]
    rel = (at["volume"] - before["volume"]) / before["volume"]
    assert abs(rel) < MAX_EVENT_STEP, \
        "the merged drop carries %+.3e of the two parents' volume more than it should" % rel


def test_volume_budget(merge):
    steps = records(merge, "STEP")
    ch = volume_changes(steps)
    ordinary, t_o = worst(ch, "ordinary")
    quality, t_q = worst(ch, "quality")
    event, t_e = worst(ch, "event")
    total = (steps[-1]["volume"] - steps[0]["volume"]) / steps[0]["volume"]
    detail = " (ordinary %+.2e at t=%s, quality %+.2e at t=%s, event %+.2e at t=%s, total %+.2e)" % (
        ordinary, t_o, quality, t_q, event, t_e, total)
    assert abs(ordinary) < MAX_ORDINARY_STEP, "an ordinary step lost volume" + detail
    assert abs(quality) < MAX_QUALITY_REMESH, "a quality remesh lost volume" + detail
    assert abs(event) < MAX_EVENT_STEP, "the surgery lost volume" + detail
    assert abs(total) < MAX_TOTAL_DRIFT, "the run lost volume" + detail


def test_no_inverted_or_collapsed_elements(merge):
    steps = records(merge, "STEP")
    assert all(s["n_negative"] == 0 for s in steps), "an element turned inside out"
    assert min(s["min_scaled_area"] for s in steps) > 1e-3


def test_the_bridge_leaves_one_chain_and_one_axis_span(merge):
    plans = [r for r in records(merge, "REMESH_REPORT") if r["has_plan"]]
    assert len(plans) == 1, "expected exactly one remesh with a surgery plan, got %d" % len(plans)
    plan = plans[0]
    assert [kind for kind, _z in plan["events"]] == ["coalescence"]
    assert plan["n_chains"] == 1
    # The two droplets' shares of the axis plus the gap between them become one span.
    assert len(plan["axis_spans"]) == 1


# --------------------------------------------------------------------------------------
# Two phase: the same merge with a gas domain sharing the interface
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def twophase(tmp_path_factory):
    return run_case("twophase", tmp_path_factory.mktemp("tp"))


def test_twophase_merge_happens(twophase):
    done = one(twophase, "DONE")
    assert done["merged"], "the two droplets never merged inside the gas box"
    assert done["post_steps"] >= 20


def test_twophase_meshes_stay_conforming(twophase):
    """The two domains of one Gmsh template share the *curve*, not the nodes.

    Each gets its own copy of the boundary and a ``ConnectMeshAtInterface`` holds the copies together;
    the surgery has to hand both copies the same new curve. Checked on every step, not only at the
    event, because the way this fails is a slow drift that only becomes visible at the *next* remesh.
    """
    rows = records(twophase, "CONFORMING")
    assert len(rows) >= 20
    for r in rows:
        assert r["n_liquid"] == r["n_gas"], "the two sides carry different node counts: %r" % (r,)
    assert max(r["max_mismatch"] for r in rows) == 0.0


def test_twophase_gas_gets_the_axis_between_the_droplets(twophase):
    """The coalescence takes the gap between the droplets away from the gas.

    The complement construction in ``get_reconnected_boundaries`` is what makes that happen without
    ``define_geometry`` special-casing it: what the opposite phase covers is everything the axis was
    covered by, minus what this phase covers now.
    """
    plans = [r for r in records(twophase, "REMESH_REPORT") if r["has_plan"]]
    assert len(plans) == 1
    assert [kind for kind, _z in plans[0]["events"]] == ["coalescence"]
    assert len(plans[0]["axis_spans"]) == 1


def test_twophase_volume_budget(twophase):
    steps = records(twophase, "STEP")
    ch = volume_changes(steps)
    event, _ = worst(ch, "event")
    total = (steps[-1]["volume"] - steps[0]["volume"]) / steps[0]["volume"]
    assert abs(event) < MAX_EVENT_STEP, "the two-phase surgery lost %+.3e in the event step" % event
    assert abs(total) < MAX_TOTAL_DRIFT, "the two-phase run lost %+.3e of the liquid" % total


def test_twophase_gas_stays_healthy(twophase):
    """The gas has to follow the interface through the event as much as the liquid does."""
    steps = records(twophase, "STEP")
    assert all(s["n_negative_gas"] == 0 for s in steps), "a gas element turned inside out"
    assert min(s["min_scaled_area_gas"] for s in steps) > 1e-3
    assert all(s["volume_gas"] > 0 for s in steps)


def test_twophase_box_volume_is_a_geometric_identity(twophase):
    """Liquid + gas is the box, whose walls do not move. A cheap but decisive cross-check that the two
    meshes tile the same region - the surgery could conserve the liquid volume perfectly and still
    leave a sliver of the box belonging to neither domain, or to both."""
    steps = records(twophase, "STEP")
    total = [s["volume"] + s["volume_gas"] for s in steps]
    drift = (total[-1] - total[0]) / total[0]
    assert abs(drift) < 1e-12, "the closed box changed volume by %+.3e" % drift
