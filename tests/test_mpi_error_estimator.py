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

"""MPI coverage for the error-estimator paths the existing MPI suites do not reach.

test_mpi_adaptivity and friends all run --distribute, and none of them uses an interface error
estimator, compound-flux groups or desired_ndof -- so they verify that nothing was broken, not that
the new collectives are right. Each case here targets one of them; see mpi_error_estimator_worker.py
for which path each exercises and why.

Both oracles from test_mpi_adaptivity apply: every reported quantity is global, so all ranks must
agree, and the answer must equal the serial one.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_error_estimator_worker.py")


def _skip_reason():
    if shutil.which("mpirun") is None:
        return "mpirun not found"
    return None


_SKIP_REASON = _skip_reason()

pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]


def _run(case, nproc, tmpdir, distribute):
    cmd = ["mpirun", "-n", str(nproc)]
    if os.environ.get("PYOOMPH_MPI_OVERSUBSCRIBE", "1") == "1":
        cmd += ["--oversubscribe"]
    cmd += [sys.executable, _WORKER, "--case", case, "--outdir", str(tmpdir)]
    if distribute:
        cmd += ["--distribute"]
    # Same TMPDIR isolation as test_mpi_adaptivity: this pytest process has already called MPI_Init
    # by importing pyoomph, so a nested mpirun collides with its Open MPI session directory and dies
    # with exit code 1 and no diagnostics.
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=900, env=env)
    except subprocess.TimeoutExpired as e:
        raise AssertionError(
            "mpirun did not finish within 900 s for case %r -- suspect a distributed deadlock, which "
            "is exactly what a collective called with a rank-dependent count looks like.\n"
            "--- stdout tail ---\n%s" % (case, (e.stdout or b"")[-3000:]))
    results = []
    for line in proc.stdout.splitlines():
        if line.startswith("PYOOMPH_MPI_RESULT "):
            results.append(json.loads(line[len("PYOOMPH_MPI_RESULT "):]))
    if not results:
        raise AssertionError("no results from mpirun (exit %d)\n--- stdout ---\n%s\n--- stderr ---\n%s"
                             % (proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
    assert len(results) == nproc, "case %s reported from %d of %d ranks" % (case, len(results), nproc)
    for r in results:
        assert "error" not in r, "case %s failed on rank %d: %s\n%s" % (
            case, r["rank"], r["error"], r.get("traceback", ""))
    return results


def _serial(case, tmpdir):
    import importlib
    worker = importlib.import_module("mpi_error_estimator_worker")
    return worker.solve_case(case, os.path.join(str(tmpdir), "serial_"+case))


def _assert_ranks_agree(results, key):
    values = [r[key] for r in results]
    assert all(v == values[0] for v in values), \
        "%s is partition-dependent across ranks: %r" % (key, values)
    return values[0]


def test_interface_estimator_under_mpi(tmp_path):
    """The recovery frame, on a vertical boundary, with the coefficient broadcast in play.

    Deliberately NOT distributed: that is the only configuration in which a rank evaluates
    coefficients fitted in another rank's frame and has to rebuild that frame from the broadcast
    element list. If the rebuild disagreed with the sender -- an eigenvector sign, say -- the errors
    would differ between ranks and from serial.
    """
    results = _run("interface", 2, tmp_path, distribute=False)
    errs = _assert_ranks_agree(results, "interface_errors")
    _assert_ranks_agree(results, "ndof")
    serial = _serial("interface", tmp_path)
    assert len(errs) == len(serial["interface_errors"])
    for a, b in zip(errs, serial["interface_errors"]):
        assert abs(a-b) <= 1e-8*max(1.0, abs(b)), "interface error differs from serial: %r vs %r" % (a, b)
    assert results[0]["ndof"] == serial["ndof"]


def test_compound_flux_groups_under_mpi(tmp_path):
    """n_compound_flux is the element count of the MPI_Allreduce on flux_norm, so a rank that
    disagreed about it would reduce over a different number of entries -- a deadlock or worse, not a
    wrong answer. Distributed, so that a rank really can hold a different part of the mesh."""
    results = _run("groups", 2, tmp_path, distribute=True)
    ndof = _assert_ranks_agree(results, "ndof")
    serial = _serial("groups", tmp_path)
    assert ndof == serial["ndof"]


def test_desired_ndof_under_mpi(tmp_path):
    """The controller has to reach the same threshold on every rank -- it becomes a mesh-wide
    tolerance -- and its element count has to skip halo copies, or a shared element is counted once
    per process holding it and the budget is spent on elements that do not exist."""
    results = _run("ndof", 2, tmp_path, distribute=True)
    ndof = _assert_ranks_agree(results, "ndof")
    serial = _serial("ndof", tmp_path)
    # Within the controller's own dead band of the serial answer: the two runs see the same errors,
    # but 2:1 balancing acts on whatever the partition made adjacent, so the paths can differ by a
    # few elements without either being wrong.
    assert abs(ndof-serial["ndof"]) <= 0.15*serial["ndof"], \
        "distributed desired_ndof landed at %d, serial at %d" % (ndof, serial["ndof"])
