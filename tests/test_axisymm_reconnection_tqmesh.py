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

"""The topological changes on TQMesh instead of gmsh.

``TopologicalChangesTemplate`` claims that the surgery is independent of the mesh generator behind
it: the plan is geometry, and the calls ``define_geometry`` makes to build it - ``spline_from_chain``,
``lines_from_axis_segments``, ``plane_surface`` - are spelled the same way by both backends. The
worker's ``BlobMesh`` is therefore written once and given to one class per backend, and this file
runs the same scenarios through ``--backend tqmesh`` that
tests/test_axisymm_reconnection_detect.py and tests/test_axisymm_reconnection_transfer.py run through
gmsh.

What is NOT claimed, and not tested here, is that the two produce the same mesh. Element sizes are
prescribed in completely different ways - gmsh has size fields, TQMesh has ``mesh_size`` and per-point
sizes - and TQMesh additionally takes the chain points as the boundary edges themselves, which is why
``TopologicalChangesTQMeshTemplate`` thins the chain first. Sizing stays the template author's job on
either backend, so the assertions below are about the topology, the volumes and the solvability.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("shapely")

import pyoomph._pyoomph_core as _pyoomph_core

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "axisymm_reconnection_worker.py")

pytestmark = [
    pytest.mark.skipif(not getattr(_pyoomph_core, "has_tqmesh", False),
                       reason="built without TQMesh (PYOOMPH_HAS_TQMESH=OFF)"),
    pytest.mark.skipif(shutil.which("gmsh") is None, reason="gmsh not found"),
]


def run_case(case, tmp_path, *extra):
    outdir = str(tmp_path / case)
    proc = subprocess.run([sys.executable, _WORKER, "--case", case, "--outdir", outdir,
                           "--backend", "tqmesh"] + list(extra),
                          cwd=_HERE, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, \
        "the %s worker failed on TQMesh:\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s" % (
            case, proc.stdout[-4000:], proc.stderr[-4000:])
    return proc


def records(proc, kind):
    return [json.loads(line.split(" ", 1)[1]) for line in proc.stdout.splitlines()
            if line.startswith("PYOOMPH_" + kind + " ")]


def one(proc, kind):
    got = records(proc, kind)
    assert len(got) == 1, "expected exactly one PYOOMPH_%s record, got %d:\n%s" % (
        kind, len(got), proc.stdout[-3000:])
    return got[0]


# --------------------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def detected(tmp_path_factory):
    return run_case("detect_pinch", tmp_path_factory.mktemp("tq_detect"), "--neck", "0.04")


def test_a_thin_neck_is_detected_on_tqmesh(detected):
    got = one(detected, "DETECT")
    assert got["has_plan"] and got["queued"]


def test_the_plan_is_the_same_single_pinch(detected):
    plan = one(detected, "PLAN")
    assert plan["kinds"] == ["pinch"]
    assert abs(plan["z"][0]) < 1e-6
    assert len(plan["before"]) == 1 and len(plan["after"]) == 2
    assert abs(sum(plan["after"]) - sum(plan["before"])) < 1e-12 * sum(plan["before"])


# --------------------------------------------------------------------------------------
# the remesh, which is where the two backends actually differ
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pinched(tmp_path_factory):
    return run_case("pinch_remesh", tmp_path_factory.mktemp("tq_pinch"))


def test_a_quality_remesh_reproduces_the_geometry(pinched):
    # The no-event path through the same define_geometry: one chain in, one chain out, and the axis
    # still covered by the single span it was.
    rep = [r for r in records(pinched, "REMESH_REPORT") if r["which"] == "quality"]
    assert len(rep) == 1 and not rep[0]["has_plan"] and rep[0]["n_chains"] == 1
    assert len(rep[0]["axis_spans"]) == 1

    before, after = records(pinched, "MESH")[0], records(pinched, "MESH")[1]
    assert after["n_interface"] == 1 and after["n_axis"] == 1
    assert after["volumes"][0] == pytest.approx(before["volumes"][0], rel=2e-3)


def test_the_pinch_is_built_into_a_tqmesh_mesh(pinched):
    rep = [r for r in records(pinched, "REMESH_REPORT") if r["which"] == "pinch"]
    assert len(rep) == 1 and rep[0]["has_plan"]
    assert [e[0] for e in rep[0]["events"]] == ["pinch"]
    assert rep[0]["n_chains"] == 2 and len(rep[0]["axis_spans"]) == 2

    after = [m for m in records(pinched, "MESH") if m["tag"] == "after_pinch"][0]
    assert after["n_interface"] == 2 and after["n_axis"] == 2
    # Two fragments, no element left inside out by the surgery.
    assert after["n_negative"] == 0 and after["n_positive"] == after["n_elements"]
    assert len(after["volumes"]) == 2
    assert after["volumes"][0] == pytest.approx(after["volumes"][1], rel=5e-3), \
        "the dumbbell is symmetric about its waist, so the two drops must match"


def test_the_volume_correction_balances_on_tqmesh(pinched):
    bal = one(pinched, "PLAN_BALANCE")
    assert sum(bal["after"]) == pytest.approx(sum(bal["before"]), rel=1e-9)


def test_the_new_mesh_is_solvable_and_does_not_re_detect(pinched):
    assert one(pinched, "RESOLVED")["has_plan"] is False


# --------------------------------------------------------------------------------------
# coalescence
# --------------------------------------------------------------------------------------

def test_two_blobs_coalesce_on_tqmesh(tmp_path):
    proc = run_case("coalescence_remesh", tmp_path)
    rep = [r for r in records(proc, "REMESH_REPORT") if r["which"] == "coalescence"]
    assert len(rep) == 1 and rep[0]["has_plan"]
    assert [e[0] for e in rep[0]["events"]] == ["coalescence"]
    assert rep[0]["n_chains"] == 1, "the two blobs must come back as one fragment"
    after = [m for m in records(proc, "MESH") if m["tag"] == "after_coalescence"][0]
    assert after["n_interface"] == 1 and after["n_negative"] == 0
    assert one(proc, "RESOLVED")["has_plan"] is False
