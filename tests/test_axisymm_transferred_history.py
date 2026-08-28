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

"""What the mesh-to-mesh transfer writes into the history across a topological change.

A node the surgery created did not exist one time step ago, and the mesh-to-mesh transfer gives it
the history of whichever old material point sits at its position - for a fresh pinch cap the middle
of the collapsing neck, and for a coalescence bridge, which is outside the old liquid altogether,
whatever the nearest-node fallback lands on.

Two things are pinned here.

* The premise: the fresh nodes really do come out of the transfer with a motion they never had,
  several times the ambient one, while the nodes away from the event keep the accurate history they
  are entitled to.
* The remedy: one time step with BDF1 weights after the event makes the transferred history
  irrelevant, exactly, and that is asserted here as an identity rather than as a tolerance. oomph
  shifts the history slots at the start of every step, so the first post-event step reads the
  transferred level 0 as its level 1 and the transferred level 1 as its level 2, and the second step
  has shifted both of the transferred levels out again; BDF1 weights ignore level 2. So two runs that
  differ only in what stands in the transferred history levels must produce the same numbers to the
  last bit once that first step is degraded - and must not, otherwise. The worker's
  ``--flatten-history`` is what makes them differ; it is a test instrument, and nothing in the
  library repairs the history per node. Section 8.4 of
  dev_docs/axisymmetric_topological_changes.md records the measurement that decided that, including
  the four per-node repairs that were tried and all came out worse than doing nothing.
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

# Not marked slow: the whole module is about ten seconds, the same order as its sibling
# test_axisymm_reconnection_* modules, and it is worth having in the default run.
pytestmark = pytest.mark.skipif(shutil.which("gmsh") is None, reason="gmsh not found")


def run_case(case, tmp_path, *extra):
    outdir = str(tmp_path / (case + "_" + "_".join(str(e) for e in extra).replace("-", "")))
    proc = subprocess.run([sys.executable, _WORKER, "--case", case, "--outdir", outdir] + list(extra),
                          cwd=_HERE, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, \
        "the %s worker failed:\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s" % (
            case, proc.stdout[-4000:], proc.stderr[-4000:])
    return proc


def one(proc, kind):
    got = [json.loads(line.split(" ", 1)[1]) for line in proc.stdout.splitlines()
           if line.startswith("PYOOMPH_" + kind + " ")]
    assert len(got) == 1, "expected one PYOOMPH_%s record, got %d" % (kind, len(got))
    return got[0]


@pytest.fixture(scope="module")
def histories(tmp_path_factory):
    """One coalescence, digested right after the transfer and before any further solve."""
    base = tmp_path_factory.mktemp("hr")
    return one(run_case("fresh_node_history", base), "HISTORY")


def test_the_transfer_gives_the_fresh_nodes_a_motion_they_never_had(histories):
    """The premise of the whole exercise, red without which nothing below means anything."""
    rec = histories
    assert rec["coalesced"] and rec["n_fresh_points"] > 0
    assert rec["inside"]["n"] > 0 and rec["outside"]["n"] > 0
    # The blobs move rigidly, so every node that was there before reports the same, correct
    # displacement per step. The nodes in the surgery's region report several times that.
    assert rec["inside"]["dx"] > 3.0 * rec["outside"]["dx"], \
        "the fresh nodes did not inherit an anomalous motion: %s" % (rec,)


@pytest.fixture(scope="module")
def bdf1_digests(tmp_path_factory):
    base = tmp_path_factory.mktemp("hb")
    out = {}
    for degrade in (False, True):
        flag = ["--degrade-bdf1"] if degrade else []
        for flatten in (False, True):
            args = (["--flatten-history"] if flatten else []) + flag
            out[(degrade, flatten)] = one(run_case("history_bdf1", base, *args), "DIGEST")
    return out


def test_one_bdf1_step_makes_the_transferred_history_irrelevant(bdf1_digests):
    a, b = bdf1_digests[(True, False)], bdf1_digests[(True, True)]
    assert a["coalesced"] and b["coalesced"]
    for key in ("u_sum", "u_min", "u_max", "x_sum", "n_nodes"):
        assert a[key] == b[key], \
            "with a degraded first step the transferred history still mattered (%s: %r vs %r)" % (
                key, a[key], b[key])


def test_without_the_degraded_step_the_transferred_history_does_matter(bdf1_digests):
    """The control: the identity above is a statement about BDF1, not about the two runs being the
    same run."""
    a, b = bdf1_digests[(False, False)], bdf1_digests[(False, True)]
    assert a["u_sum"] != b["u_sum"] or a["x_sum"] != b["x_sum"], \
        "the two histories produced identical results even with BDF2 throughout: %r %r" % (a, b)
