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

"""Detection side of AxisymmetricReconnection: what it notices, and what it refuses to notice.

Three things, none of which needs the mesh to actually be rebuilt:

* a neck thinner than ``rmin`` produces a surgery plan and queues the mesh template for remeshing,
  a neck thicker than ``rmin`` produces neither. Both are run with a spatial *scaling* set and a
  dimensional ``rmin``, so a wrong nondimensionalisation flips the answer;
* the overlap guard rejects a Newton step that would drive two approaching axial tips through each
  other, so that an adaptive time stepper cuts dt instead. It is the canonical user of
  ``before_newton_convergence_check`` - see tests/test_newton_abort.py for the mechanism;
* the mesh-velocity gate refuses to coalesce two tips that are moving apart, even though their gap
  is below ``distmin``. The same configuration with ``check_mesh_motion_direction=False`` must plan
  the coalescence, or this would prove nothing.

One Problem per process (a second one segfaults in the JIT loader), so each case runs in its own
subprocess, as in tests/test_boundary_interpolation_fixes.py.
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


def run_case(case, tmp_path, *extra):
    outdir = str(tmp_path / case)
    proc = subprocess.run([sys.executable, _WORKER, "--case", case, "--outdir", outdir] + list(extra),
                          cwd=_HERE, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, \
        "the %s worker failed:\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s" % (
            case, proc.stdout[-4000:], proc.stderr[-4000:])
    return proc


def records(proc, kind):
    out = []
    for line in proc.stdout.splitlines():
        if line.startswith("PYOOMPH_" + kind + " "):
            out.append(json.loads(line.split(" ", 1)[1]))
    return out


def one(proc, kind):
    got = records(proc, kind)
    assert len(got) == 1, "expected exactly one PYOOMPH_%s record, got %d:\n%s" % (
        kind, len(got), proc.stdout[-3000:])
    return got[0]


# --------------------------------------------------------------------------------------
# (i) rmin
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def thin_neck(tmp_path_factory):
    return run_case("detect_pinch", tmp_path_factory.mktemp("thin"), "--neck", "0.04")


@pytest.fixture(scope="module")
def thick_neck(tmp_path_factory):
    return run_case("detect_pinch", tmp_path_factory.mktemp("thick"), "--neck", "0.25")


def test_a_thin_neck_is_planned_and_queued(thin_neck):
    got = one(thin_neck, "DETECT")
    assert got["has_plan"], "a neck of 0.04 is well below rmin=0.12 but no surgery plan was parked"
    assert got["queued"], "the mesh template was not added to Problem._domains_to_remesh"


def test_the_plan_is_a_single_pinch_that_splits_the_volume(thin_neck):
    plan = one(thin_neck, "PLAN")
    assert plan["kinds"] == ["pinch"], "expected exactly one pinch event, got %s" % (plan["kinds"],)
    assert abs(plan["z"][0]) < 1e-6, \
        "the waist is at z=0 by construction but the event was reported at z=%g" % plan["z"][0]
    assert len(plan["before"]) == 1 and len(plan["after"]) == 2, \
        "one parent must become two fragments, got %d -> %d" % (len(plan["before"]), len(plan["after"]))
    assert abs(sum(plan["after"]) - sum(plan["before"])) < 1e-12 * sum(plan["before"]), \
        "the planned fragments do not add up to the parent volume: %s vs %s" % (
            plan["after"], plan["before"])


def test_a_thick_neck_is_left_alone(thick_neck):
    """Also the test of the dimensional path: rmin is given in metres against a 0.5 m spatial scale,
    so a missing nondimensionalisation would make 0.25 look thin."""
    got = one(thick_neck, "DETECT")
    assert not got["has_plan"], "a neck of 0.25 is above rmin=0.12 but a surgery plan was parked"
    assert not got["queued"], "nothing happened, yet a remesh was requested"


# --------------------------------------------------------------------------------------
# (ii) the overlap guard
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def guard(tmp_path_factory):
    return run_case("overlap_guard", tmp_path_factory.mktemp("guard"))


def test_the_guard_arms_itself_when_something_gets_close(guard):
    got = one(guard, "ARMED")
    assert got["armed"], "a gap of 2*distmin is within the arming factor of 3, but the guard is off"
    assert got["gap"] > 5 * got["limit"], \
        "the armed configuration is already inside the rejection limit, so the case below would " \
        "reject for the wrong reason (gap=%g, limit=%g)" % (got["gap"], got["limit"])


def test_a_harmless_step_is_not_rejected(guard):
    got = one(guard, "HARMLESS")
    assert not got["rejected"], "an armed guard rejected a step that stays far from the limit"
    assert got["gap"] > 0.01, "the control step already went below the limit, so it proves nothing"


def test_a_step_that_would_overlap_is_rejected(guard):
    got = one(guard, "GUARD")
    assert got["rejected"], \
        "a step driving the two tips through each other was accepted; the time stepper would " \
        "never get the chance to cut dt"


# --------------------------------------------------------------------------------------
# (iii) the mesh-velocity gate
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def separating_gated(tmp_path_factory):
    return run_case("separating_tips", tmp_path_factory.mktemp("sep_on"), "--check-motion")


@pytest.fixture(scope="module")
def separating_ungated(tmp_path_factory):
    return run_case("separating_tips", tmp_path_factory.mktemp("sep_off"))


def test_the_configuration_would_otherwise_coalesce(separating_ungated):
    """The premise: without the gate, this gap does produce a coalescence plan."""
    got = one(separating_ungated, "SEPARATING")
    assert got["gap"] < got["distmin"], \
        "the gap grew past distmin, so the gate is not what decides here (gap=%g, distmin=%g)" % (
            got["gap"], got["distmin"])
    assert got["has_plan"], "a gap below distmin did not produce a coalescence plan at all"


def test_separating_tips_do_not_coalesce(separating_gated):
    got = one(separating_gated, "SEPARATING")
    assert got["gap"] < got["distmin"], "the premise moved: the gap is no longer below distmin"
    assert not got["has_plan"], \
        "two tips flying apart were merged anyway; the mesh-velocity gate did not fire"
