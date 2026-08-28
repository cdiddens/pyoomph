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

# --distribute on a problem that cannot be partitioned.
#
# oomph-lib's Problem::distribute() throws "there are less elements than processors" when the global
# mesh has fewer elements than there are ranks, which a pure ODE problem (a single element) always
# does. Every rank threw, so the job aborted with two interleaved tracebacks instead of running --
# the very first thing a user tries `mpirun --distribute` on is typically a small tutorial script.
#
# Nothing can be partitioned there, so the request cannot be honoured; but it can be ignored, which
# is what pyoomph now does: the problem stays replicated on every rank, exactly as the same script
# runs under mpirun WITHOUT --distribute, and the answer is the serial one. Hence the assertions:
# the run finishes, no rank reports distributed, and every rank reproduces the serial trajectory.
#
# The other two tests are the other side of it: the skip must trigger only where oomph-lib would
# have thrown. A problem with a real mesh next to that same ODE must still be distributed, and so
# must a single-element mesh that RefineToLevel grows before the distribution -- the element count
# is read after the initial uniform refinement, and reading it any earlier would refuse a problem
# that partitions fine. The error-driven initial adaption, on the other hand, runs after the
# distribution and cannot rescue a mesh that is too coarse: it leaves the mesh non-uniformly
# refined, which Problem::distribute() refuses outright.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_undistributable_worker.py")


def _mpi_reason():
    """None if a distributed run is possible here, else the reason to skip."""
    if shutil.which("mpirun") is None:
        return "mpirun not found"
    try:
        from pyoomph.generic.mpi import has_mpi, have_pymetis
        if not has_mpi():
            return "pyoomph was built without MPI"
        if not have_pymetis():
            return "PyMetis is not installed, so the mesh cannot be partitioned"
    except Exception as e:  # noqa: BLE001
        return "MPI unavailable: " + str(e)
    return None


_SKIP_REASON = _mpi_reason()
pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]


def _run(case, tmpdir, nproc, distribute=True, timeout=900):
    cmd = []
    if nproc is not None:
        cmd += ["mpirun", "-n", str(nproc)]
    cmd += [sys.executable, _WORKER, "--case", case,
            "--outdir", os.path.join(str(tmpdir), "out_" + case + "_" + str(nproc))]
    if distribute:
        cmd += ["--distribute"]
    # Importing pyoomph calls MPI_Init, so THIS pytest process is already a (singleton) MPI job and
    # owns an Open MPI session directory under TMPDIR. A nested mpirun collides with it and dies with
    # exit code 1 and no diagnostics. Give the child its own TMPDIR.
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        # Bounded on purpose: a refusal raised on some ranks but not others leaves the rest waiting in
        # the next collective, which has to surface as a failure rather than a suite that hangs.
        raise AssertionError(
            "mpirun did not finish within %d s -- suspect a deadlock (case=%s, nproc=%s).\n"
            "--- stdout tail ---\n%s" % (timeout, case, nproc, (e.stdout or "")[-3000:]))
    per_rank = [json.loads(l[len("PYOOMPH_MPI_RESULT "):]) for l in proc.stdout.splitlines()
                if l.startswith("PYOOMPH_MPI_RESULT ")]
    if not per_rank:
        raise AssertionError(
            "no results from mpirun (exit %d -- a negative value is the killing signal)\n"
            "--- stdout tail ---\n%s\n--- stderr tail ---\n%s"
            % (proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
    assert len(per_rank) == (nproc or 1), "reported from %d of %s ranks" % (len(per_rank), nproc)
    for r in per_rank:
        assert "error" not in r, "failed on rank %d: %s\n%s" % (
            r["rank"], r["error"], r.get("traceback", ""))
    assert proc.returncode == 0, \
        "mpirun exited %d after every rank reported -- suspect a crash during teardown" % proc.returncode
    return per_rank


@pytest.mark.parametrize("nproc", [2, 3])
def test_ode_only_problem_survives_a_distribute_request(tmp_path, nproc):
    serial = _run("ode", tmp_path, nproc=None, distribute=False)[0]
    for r in _run("ode", tmp_path, nproc=nproc):
        where = "rank %d/%d" % (r["rank"], nproc)
        assert not r["distributed"], \
            "%s: a %d-element problem reports itself as distributed over %d ranks" % (
                where, r["ndof"], nproc)
        assert r["final_time"] == pytest.approx(1.0, abs=1e-5), \
            "%s: the run stopped at t=%.6g" % (where, r["final_time"])
        # Replicated means every rank solved the whole thing, so this is the serial answer exactly.
        assert r["y"] == pytest.approx(serial["y"], rel=1e-12), \
            "%s: y=%.17g, serially %.17g" % (where, r["y"], serial["y"])


def test_a_single_element_refined_before_the_distribution_is_distributed(tmp_path):
    """One quad element, uniformly refined to four by RefineToLevel, then adapted."""
    serial = _run("refined", tmp_path, nproc=None, distribute=False)[0]
    for r in _run("refined", tmp_path, nproc=2):
        where = "rank %d" % r["rank"]
        assert r["distributed"], \
            "%s: refused a mesh that has %d elements by the time it is distributed" % (where, 4)
        assert r["ndof"] == serial["ndof"], \
            "%s: ndof=%d, serially %d -- the adaption did not follow the serial one" % (
                where, r["ndof"], serial["ndof"])


def test_a_partitionable_problem_next_to_an_ode_is_still_distributed(tmp_path):
    serial = _run("pde", tmp_path, nproc=None, distribute=False)[0]
    for r in _run("pde", tmp_path, nproc=2):
        where = "rank %d" % r["rank"]
        assert r["distributed"], "%s: the mesh was not distributed" % where
        assert r["ndof"] == serial["ndof"], \
            "%s: ndof=%d, serially %d" % (where, r["ndof"], serial["ndof"])
        # The ODE is kept as a halo on every rank, so its value is available everywhere.
        assert r["y"] == pytest.approx(serial["y"], rel=1e-9), \
            "%s: y=%.17g, serially %.17g" % (where, r["y"], serial["y"])
