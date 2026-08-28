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

# Work that only rank 0 does must fail for all ranks or for none.
#
# Both cases here used to hang instead of ending, and in both the run had already printed the right
# error message before it stopped returning:
#
# 1. A missing PyMetis. oomph-lib partitions on rank 0 and scatters the result, so the ImportError
#    from the METIS callback unwound rank 0 alone while everybody else sat in the scatter.
#    Problem.distribute() now checks PyMetis on all ranks before entering the distribution, and the
#    callback aborts the job for anything that still escapes it.
#
# 2. A failing gmsh.write(). The mesh files are written by rank 0 between two barriers, so an
#    unwritable output directory (or any other write error) left the other ranks in the barrier
#    behind it. run_on_rank_zero() now agrees on the outcome instead.
#
# The same asymmetry without a failure: a remesh requested by one rank. RemeshWhen judges the
# elements its own rank holds, while force_remesh() is collective throughout, so the ranks that saw
# nothing wrong used to skip it and wait in collectives the asking rank had already entered.
# Problem._agree_on_domains_to_remesh() makes the request unanimous first.

import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYMETIS_WORKER = os.path.join(_HERE, "mpi_missing_pymetis_worker.py")
_GMSH_WORKER = os.path.join(_HERE, "mpi_gmsh_write_worker.py")
_REMESH_WORKER = os.path.join(_HERE, "mpi_remesh_agreement_worker.py")


def _mpi_reason():
    if shutil.which("mpirun") is None:
        return "mpirun not found"
    try:
        from pyoomph.generic.mpi import has_mpi
        if not has_mpi():
            return "pyoomph was built without MPI"
    except Exception as e:  # noqa: BLE001
        return "MPI unavailable: " + str(e)
    return None


_SKIP_REASON = _mpi_reason()
pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]


def _run(worker, tmpdir, extra_args, nproc=2, timeout=300):
    cmd = ["mpirun", "-n", str(nproc)]
    if os.environ.get("PYOOMPH_MPI_OVERSUBSCRIBE", "1") == "1":
        cmd += ["--oversubscribe"]
    cmd += [sys.executable, worker, "--outdir", str(tmpdir)] + extra_args
    # Importing pyoomph calls MPI_Init, so this pytest process already owns an Open MPI session
    # directory under TMPDIR; a nested mpirun collides with it and dies with no diagnostics.
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        return subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout,
                              env=env)
    except subprocess.TimeoutExpired as e:
        # This is the regression itself: before the fix these runs hung rather than failing.
        raise AssertionError(
            "mpirun did not finish within %d s -- a failure on rank 0 deadlocked the other ranks "
            "instead of ending the run (%s).\n--- stdout tail ---\n%s" % (
                timeout, " ".join(extra_args) or "no extra args", (e.stdout or "")[-3000:]))


def _reported(proc):
    return [l for l in proc.stdout.splitlines() if l.startswith("PYOOMPH_MPI_RESULT ")]


def test_missing_pymetis_fails_on_every_rank(tmp_path):
    proc = _run(_PYMETIS_WORKER, tmp_path, ["--distribute"])
    assert proc.returncode != 0, "the run reported success although it could not partition"
    reported = _reported(proc)
    assert len(reported) == 2, \
        "expected both ranks to raise, got:\n%s\n--- stderr tail ---\n%s" % (
            "\n".join(reported), proc.stderr[-2000:])
    for line in reported:
        assert "ImportError" in line and "PyMetis" in line, \
            "unhelpful failure on a rank: %s" % line


def test_missing_pymetis_inside_the_callback_aborts_instead_of_hanging(tmp_path):
    """Last-resort net: the pre-flight is bypassed, so rank 0 fails alone inside METIS."""
    proc = _run(_PYMETIS_WORKER, tmp_path, ["--distribute", "--skip-preflight"])
    assert proc.returncode != 0, "the run reported success although it could not partition"
    out = proc.stdout + proc.stderr
    assert "PyMetis" in out, \
        "the abort did not say why:\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s" % (
            proc.stdout[-2000:], proc.stderr[-2000:])


@pytest.mark.parametrize("which", ["geo", "msh"])
def test_failing_gmsh_write_on_rank_zero_fails_on_every_rank(tmp_path, which):
    proc = _run(_GMSH_WORKER, tmp_path, ["--fail-write", which])
    assert proc.returncode != 0, "the run reported success although it could not write the mesh"
    reported = _reported(proc)
    assert len(reported) == 2, \
        "expected both ranks to raise, got:\n%s\n--- stderr tail ---\n%s" % (
            "\n".join(reported), proc.stderr[-2000:])
    for line in reported:
        # The ranks that did not write must still learn what went wrong, not just that it did.
        assert "simulated failure while writing" in line, \
            "a rank did not learn the real cause: %s" % line


def test_gmsh_meshing_under_mpi_still_works(tmp_path):
    """Control, plus the primitive that keeps the (re)meshing decision unanimous."""
    proc = _run(_GMSH_WORKER, tmp_path, [])
    assert proc.returncode == 0, \
        "the run failed:\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s" % (
            proc.stdout[-2000:], proc.stderr[-2000:])
    assert len(_reported(proc)) == 2, "not every rank finished:\n%s" % "\n".join(_reported(proc))
    assert os.path.exists(os.path.join(str(tmp_path), "_gmsh", "Disc.msh")), "no mesh file written"
    votes = [l for l in proc.stdout.splitlines() if l.startswith("PYOOMPH_MPI_ANY ")]
    assert len(votes) == 2, "get_mpi_any was not reported by both ranks:\n%s" % "\n".join(votes)
    for line in votes:
        # A single rank asking for a regeneration has to carry all of them into the collectives.
        assert "one=True" in line and "none=False" in line, "get_mpi_any disagrees: %s" % line


@pytest.mark.parametrize("asking_rank", [0, 1])
def test_remesh_requested_by_one_rank_is_carried_by_all(tmp_path, asking_rank):
    proc = _run(_REMESH_WORKER, tmp_path, ["--ask-on-rank", str(asking_rank)])
    assert proc.returncode == 0, \
        "the run failed:\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s" % (
            proc.stdout[-2000:], proc.stderr[-2000:])
    reported = _reported(proc)
    assert len(reported) == 2, "not every rank finished:\n%s" % "\n".join(reported)
    for line in reported:
        # Including the rank that saw nothing wrong: it must remesh with the others, not skip.
        assert "remeshed=True" in line, "a rank sat the remesh out: %s" % line
    assert any("asked=1" in l for l in reported) and any("asked=0" in l for l in reported), \
        "the test did not set up an asymmetric request:\n%s" % "\n".join(reported)
