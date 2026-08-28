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

# A solver that is not MPI-parallel, used under mpirun.
#
# oomph-lib routes EVERY run with nproc>1 through its distributed solver entry point, whether or not
# the problem was distributed -- SuperLUSolver::solve branches on communicator_pt()->nproc(), not on
# the mesh. So pardiso/superlu/umfpack/accelerate could not solve at all under mpirun: pardiso raised
# in its constructor, superlu raised "cannot solve distributed", accelerate inherited the base
# refusal. On a machine without PETSc that made mpirun unusable outright.
#
# GenericLinearSystemSolver now gathers the row-distributed system onto rank 0 and calls the backend's
# own solve_serial() there. The assembly stays parallel; the solve does not scale. What these tests
# check is that it is CORRECT and that it cannot hang:
#
#  - the answer matches the serial one, in both MPI regimes and at a non-divisible rank count;
#  - a failure only rank 0 can see still reaches every rank (this is the deadlock case, and a
#    timeout here IS the regression -- before the shim's MPI_Allreduce the other ranks would sit in
#    it forever);
#  - the waiting ranks sleep rather than spin, which is the entire reason for the polled waits in
#    pyoomph/generic/mpi.py. A blocking collective would pin N-1 cores at 100% and starve exactly the
#    OpenMP threads rank 0 was supposed to get.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_serial_solver_worker.py")


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


def _have_solver(idname):
    try:
        from pyoomph.solvers.generic import GenericLinearSystemSolver
        if idname in GenericLinearSystemSolver._registered_solvers:
            return True
        import importlib
        importlib.import_module("pyoomph.solvers." + ("scipy" if idname in ("superlu", "umfpack") else idname))
        return idname in GenericLinearSystemSolver._registered_solvers
    except Exception:  # noqa: BLE001
        return False


# superlu/umfpack are always there (scipy is a hard dependency); pardiso needs MKL. Accelerate is
# macOS-only and is covered by the shared code path plus the macOS workflow, not from here.
_SOLVERS = [s for s in ("superlu", "umfpack", "pardiso") if _have_solver(s)]


def _run(tmpdir, extra, nproc=2, timeout=300):
    cmd = ["mpirun", "-n", str(nproc)]
    if os.environ.get("PYOOMPH_MPI_OVERSUBSCRIBE", "1") == "1":
        cmd += ["--oversubscribe"]
    cmd += [sys.executable, _WORKER, "--outdir", str(tmpdir)] + extra
    # Importing pyoomph calls MPI_Init, so this pytest process already owns an Open MPI session
    # directory under TMPDIR; a nested mpirun collides with it and dies with no diagnostics.
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        return subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        raise AssertionError(
            "mpirun did not finish within %d s -- the gathered solve deadlocked (%s).\n"
            "--- stdout tail ---\n%s" % (timeout, " ".join(extra), (e.stdout or "")[-3000:]))


def _results(proc):
    out = []
    for line in proc.stdout.splitlines():
        if line.startswith("PYOOMPH_MPI_RESULT "):
            out.append(json.loads(line[len("PYOOMPH_MPI_RESULT "):]))
        # Under the condensed MPI console the marker is prefixed with "[rank N] ".
        elif "PYOOMPH_MPI_RESULT " in line:
            out.append(json.loads(line.split("PYOOMPH_MPI_RESULT ", 1)[1]))
    return out


def _serial_reference(tmpdir, solver, N=12):
    env = dict(os.environ)
    proc = subprocess.run([sys.executable, _WORKER, "--outdir", os.path.join(str(tmpdir), "serial"),
                           "--solver", solver, "--N", str(N)],
                          cwd=_HERE, capture_output=True, text=True, timeout=300, env=env)
    res = _results(proc)
    assert len(res) == 1 and "error" not in res[0], \
        "the serial reference itself failed:\n%s\n%s" % (proc.stdout[-2000:], proc.stderr[-2000:])
    return res[0]


@pytest.mark.parametrize("solver", _SOLVERS)
@pytest.mark.parametrize("nproc", [2, 3])
@pytest.mark.parametrize("distribute", [False, True])
def test_gathered_solve_matches_serial(tmp_path, solver, nproc, distribute):
    """nproc=3 is deliberate: 529 rows over 3 ranks is a non-divisible split, which is where an
    offset computed as 'rank * n // nproc' rather than from the reported first_row falls over."""
    ref = _serial_reference(tmp_path, solver)
    extra = ["--solver", solver, "--mpi-output=all"] + (["--distribute"] if distribute else [])
    proc = _run(tmp_path, extra, nproc=nproc)
    res = _results(proc)
    assert proc.returncode == 0, "the run failed:\n%s\n%s" % (proc.stdout[-3000:], proc.stderr[-2000:])
    assert len(res) == nproc, "expected one result per rank, got %d:\n%s" % (len(res), proc.stdout[-2000:])
    for r in res:
        assert "error" not in r, "rank %d failed: %s" % (r["rank"], r.get("error"))
        assert r["finite"], "rank %d has a non-finite dof" % r["rank"]
        assert r["ndof"] == ref["ndof"]
        # Every rank must hold the WHOLE dof vector: the solution is scattered back as row blocks and
        # oomph allgathers it onto the (replicated) dof distribution. A rank left holding only its own
        # rows is the failure the old pardiso refusal described.
        assert r["len_dofs"] == ref["ndof"], \
            "rank %d holds %d of %d dofs" % (r["rank"], r["len_dofs"], ref["ndof"])
        assert r["l2"] == pytest.approx(ref["l2"], rel=1e-9), \
            "rank %d disagrees with the serial solution (%r vs %r)" % (r["rank"], r["l2"], ref["l2"])
        assert r["maxres"] < 1e-7
    # And they must agree with EACH OTHER exactly, not merely to the serial tolerance: a rank solving
    # a slightly different system is a far worse failure than an inaccurate one.
    assert len({r["l2"] for r in res}) == 1, "the ranks disagree about the solution: %s" % [r["l2"] for r in res]


@pytest.mark.skipif(not _SOLVERS, reason="no serial solver available")
def test_solver_error_on_rank_zero_is_retried_not_hung(tmp_path):
    """A SolverError is seen by rank 0 alone. It must become a retryable failure on every rank.

    The route is the MPI_Allreduce in src/nanobind/solver.cpp, which was written for exactly this:
    rank 0 reports the failure, the others return normally, and all of them end up throwing
    NewtonSolverError. Without it the other ranks would wait in that reduce forever.
    """
    proc = _run(tmp_path, ["--solver", _SOLVERS[0], "--fail-mode", "solvererror", "--mpi-output=all"],
                nproc=2, timeout=180)
    assert proc.returncode != 0, "the run reported success although rank 0 could not factorise"
    out = proc.stdout + proc.stderr
    assert "THE LINEAR SOLVER FAILED" in out or "simulated factorisation failure" in out, \
        "the failure was not reported:\n%s\n%s" % (proc.stdout[-2000:], proc.stderr[-2000:])


@pytest.mark.skipif(not _SOLVERS, reason="no serial solver available")
def test_non_solver_error_on_rank_zero_ends_every_rank(tmp_path):
    """The other half of the failure story, and the one the C++ shim cannot cover.

    solver.cpp rethrows anything that is not a SolverError BEFORE it reaches its MPI_Allreduce, so
    rank 0 would unwind while the others sat in it. The Python side has to agree first.
    """
    proc = _run(tmp_path, ["--solver", _SOLVERS[0], "--fail-mode", "valueerror", "--mpi-output=all"],
                nproc=2, timeout=180)
    assert proc.returncode != 0, "the run reported success although rank 0 raised"
    out = proc.stdout + proc.stderr
    assert "simulated non-solver failure" in out, \
        "no rank learned the real cause:\n%s\n%s" % (proc.stdout[-2000:], proc.stderr[-2000:])


@pytest.mark.skipif(not _SOLVERS, reason="no serial solver available")
def test_waiting_ranks_do_not_burn_cpu(tmp_path):
    """The waiting ranks must sleep, not spin.

    Rank 0 sleeps inside its factorisation to stand in for a slow one. With the polled waits a
    waiting rank uses a percent or two of a core; with a blocking MPI collective it uses 100%, which
    would leave rank 0's OpenMP threads fighting for the cores this design exists to free. The
    threshold is deliberately loose -- the two cases differ by a factor of fifty.
    """
    delay = 1.0
    proc = _run(tmp_path, ["--solver", _SOLVERS[0], "--root-solve-delay", str(delay), "--N", "8",
                           "--mpi-output=all"], nproc=2, timeout=300)
    assert proc.returncode == 0, "the run failed:\n%s\n%s" % (proc.stdout[-2000:], proc.stderr[-2000:])
    res = {r["rank"]: r for r in _results(proc)}
    assert len(res) == 2
    waiting = res[1]
    assert waiting["solve_wall"] > delay, \
        "rank 0 did not actually hold the solve up, so this proves nothing (%r)" % waiting
    assert waiting["solve_cpu"] < 0.5 * waiting["solve_wall"], (
        "rank 1 burned %.2f s of CPU over %.2f s of waiting -- it is spinning in a blocking "
        "collective instead of sleeping" % (waiting["solve_cpu"], waiting["solve_wall"]))
