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

# The MPI gate for problem.keep_structural_zeros and the solver structure reuse it enables
# (dev_docs/structural_assembly.md sections 6 and 7).
#
# pytest itself runs serially, so each test launches tests/mpi_structural_worker.py under
# `mpirun -n N ... --distribute` and checks three independent things:
#
#   1. cross-rank agreement -- every measured quantity is global (get_residuals() is gathered to full
#      length, the integral observables are MPI_Allreduce-summed over non-halo elements), so all ranks
#      must report the same numbers. A partition-dependent answer means the structural pattern is
#      being built or reused on the wrong subset of the mesh.
#   2. agreement with the value-filtered run -- turning structural zeros on must not move the
#      solution. Explicit zeros are still zeros.
#   3. agreement with the SERIAL run, computed in-process from the same worker module. This is what
#      catches a globally consistent but wrong field, which per-rank agreement alone would pass.
#
# Why the worker solves twice: the FIRST solve can never reuse a factorisation, so only a repeat
# exercises the branch this work adds. A test that solved once would pass with the reuse path dead.
#
# What specifically could break here and nowhere else: PETSc's distributed matrix is rebuilt
# collectively, so if jacobian_structure_id ever diverged across ranks, some would reuse their Mat
# while others rebuilt theirs -- a deadlock, not a wrong answer. (There is also a PARANOID
# MPI_Allreduce check for exactly that inside pyoomph::Problem::assign_eqn_numbers.)

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_structural_worker.py")


def _mpi_reason():
    """None if a distributed run is possible here, else the reason to skip."""
    if shutil.which("mpirun") is None:
        return "mpirun not found"
    try:
        from pyoomph.generic.mpi import has_mpi
        if not has_mpi():
            return "pyoomph was built without MPI"
    except Exception as e:
        return "MPI unavailable: " + str(e)
    try:
        from petsc4py import PETSc  # type:ignore
        if not PETSc.Sys.hasExternalPackage("mumps"):
            return "PETSc has no MUMPS support (no distributed-capable direct solver)"
    except Exception:
        return "petsc4py not available (no distributed-capable direct solver)"
    return None


_SKIP_REASON = _mpi_reason()
pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]

# Tolerances. The two runs solve the identical discrete system; they differ only in whether entries
# that are exactly zero are stored, and in whether MUMPS redoes its analysis. So the results should
# agree to round-off accumulated over a Newton solve, not merely to engineering accuracy.
_OBS_RTOL = 1e-10
_RES_TOL = 1e-7


def _run_distributed(nproc, tmpdir, dim, size, structural, timeout=900):
    """Launch the worker under mpirun and return the list of per-rank result dicts."""
    cmd = ["mpirun", "-n", str(nproc)]
    if os.environ.get("PYOOMPH_MPI_OVERSUBSCRIBE", "1") == "1":
        # CI machines routinely have fewer slots than we ask for; without this OpenMPI refuses to start.
        cmd += ["--oversubscribe"]
    cmd += [sys.executable, _WORKER, "--outdir", str(tmpdir),
            "--dim", str(dim), "--size", str(size), "--structural", str(int(structural)),
            "--distribute"]
    # Importing pyoomph calls MPI_Init, so THIS pytest process is already a (singleton) MPI job and owns
    # an Open MPI session directory under TMPDIR. A nested mpirun collides with it and dies immediately
    # with exit code 1 and no diagnostics at all. Give the child its own TMPDIR.
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        # A bounded timeout on purpose: ranks disagreeing about whether to rebuild the matrix deadlock
        # rather than returning, and that must surface as a FAILURE, not as a suite that never returns.
        raise AssertionError(
            "mpirun did not finish within %d s -- suspect a distributed deadlock (dim=%d structural=%s)."
            "\n--- stdout tail ---\n%s" % (timeout, dim, structural, (e.stdout or "")[-3000:]))
    per_rank = []
    for line in proc.stdout.splitlines():
        if line.startswith("PYOOMPH_MPI_RESULT "):
            per_rank.append(json.loads(line[len("PYOOMPH_MPI_RESULT "):]))
    if not per_rank:
        raise AssertionError(
            "no results from mpirun (exit %d)\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s"
            % (proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
    assert len(per_rank) == nproc, "reported from %d of %d ranks" % (len(per_rank), nproc)
    for r in per_rank:
        assert "error" not in r, "failed on rank %d: %s\n%s" % (
            r["rank"], r["error"], r.get("traceback", ""))
    return per_rank


def _obs_keys(result):
    keys = [k for k in result if k.startswith("obs_")]
    assert keys, "no integral observables -- the field would go unchecked"
    return keys


def _assert_same_solution(what, a, b):
    assert a["ndof"] == b["ndof"], "%s: ndof %d vs %d" % (what, a["ndof"], b["ndof"])
    for k in _obs_keys(a):
        assert b[k] == pytest.approx(a[k], rel=_OBS_RTOL), \
            "%s: %s = %.17g vs %.17g" % (what, k, a[k], b[k])


def _serial_reference(tmpdir, dim, size, structural):
    sys.path.insert(0, _HERE)
    try:
        import mpi_structural_worker
        return mpi_structural_worker.solve_case(dim, size, structural,
                                                outdir=os.path.join(str(tmpdir), "serial"))
    finally:
        sys.path.remove(_HERE)


@pytest.mark.parametrize("nproc", [2, 4])
@pytest.mark.parametrize("dim,size", [(2, 16), (3, 5)], ids=["2d", "3d"])
def test_structural_zeros_agree_across_ranks_and_with_serial(tmp_path, nproc, dim, size):
    filtered = _run_distributed(nproc, tmp_path / "off", dim, size, structural=False)
    structural = _run_distributed(nproc, tmp_path / "on", dim, size, structural=True)

    # (1) every rank must agree with rank 0, in both configurations
    for name, per_rank in (("filtered", filtered), ("structural", structural)):
        for r in per_rank:
            assert r["maxres"] < _RES_TOL, \
                "%s: max|residual| = %.3e on rank %d" % (name, r["maxres"], r["rank"])
            _assert_same_solution("%s rank %d vs rank 0" % (name, r["rank"]), per_rank[0], r)
        ids = {r["structure_id"] for r in per_rank}
        assert len(ids) == 1, "%s: jacobian_structure_id diverged across ranks: %s" % (name, ids)

    # the whole point: with structural zeros there IS a reusable pattern, without them there is not
    assert filtered[0]["structure_id"] == 0
    assert structural[0]["structure_id"] != 0

    # (2) turning structural zeros on must not move the solution
    _assert_same_solution("structural vs filtered (distributed)", filtered[0], structural[0])

    # (3) and the distributed answer must be the serial one
    serial = _serial_reference(tmp_path, dim, size, structural=True)
    assert serial["maxres"] < _RES_TOL
    _assert_same_solution("distributed vs serial", serial, structural[0])
