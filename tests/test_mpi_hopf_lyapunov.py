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

# The first Lyapunov coefficient, and switching onto the orbit, under MPI.
#
# switch_to_hopf_orbit() used to refuse --distribute outright. Three things stood in the way, and all
# three are worth knowing about because none of them was the obvious one:
#
#   1. get_hopf_lyapunov_coefficient() assembled the eigenproblem pencil and used the local row block
#      as if it were the whole square matrix. It is now ALLgathered -- not gathered to rank 0 -- so
#      every rank runs the routine in lockstep: nodalf(), d2f() and d3f() go through set_current_dofs()
#      and get_residuals(), which are collective, and doing the work on one rank would deadlock.
#   2. Problem::get_second_order_directional_derivative indexed the caller's direction by GLOBAL
#      equation number while demanding it be nrow_local long, counted halo elements twice, and never
#      reduced across ranks -- a heap overrun under --distribute, correct under a replicated mpirun
#      only because the two lengths coincide there. It now takes and returns global-length vectors.
#   3. The final d(Re lambda)/d(parameter) walked one FD step with go_to_param(), i.e. by arclength
#      continuation, which is refused while a tracker is installed on a distributed problem. A step of
#      FD_param_delta from a converged eigenbranch does not need arclength.
#
# The test problem is the Hopf normal form applied pointwise on a 1D mesh plus diffusion (see
# tests/hopf_lyapunov_worker.py). Diffusion annihilates a uniform field, so the uniform state is still
# an exact solution with the same normal form, while the mesh is big enough to distribute.
#
# WHAT EACH ASSERTION IS FOR:
#
#   - ga and al against the serial values. These are the whole point: the coefficient is the thing
#     that was unavailable distributed. Note ga is NOT mesh-independent (it scales with the
#     normalisation of q), so the comparison has to be against the same discretisation, not a formula.
#   - the orbit radius and period. This is the end-to-end one -- ga, al and the guess construction
#     have to agree with each other for the radius to come out, so it fails if any of them is wrong.
#   - that the distributed run really was distributed, and that its global dof count agrees. A silent
#     fallback to replicated would otherwise pass everything else.

import json
import os
import shutil
import subprocess
import sys

import numpy
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "hopf_lyapunov_worker.py")
_N = 20
_NPROC = 4


def _skip_reason():
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
            return "PETSc has no MUMPS support"
        import slepc4py  # type:ignore  # noqa: F401
    except Exception:
        return "petsc4py/slepc4py not available (PYTHONPATH must carry a complex PETSc build)"
    return None


_SKIP = _skip_reason()
pytestmark = [pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP)), pytest.mark.slow]


def _run(tmpdir, nproc, distribute, what, timeout=1800):
    os.makedirs(str(tmpdir), exist_ok=True)
    cmd = [sys.executable, _WORKER, "--outdir", str(tmpdir), "--case", "pde",
           "--what", what, "--N", str(_N)]
    if nproc > 1:
        cmd = ["mpirun", "-n", str(nproc)] + cmd
    if distribute:
        cmd.append("--distribute")
    env = dict(os.environ)
    # Own TMPDIR for the nested mpirun; see tests/test_mpi_bifurcation_tracking.py for why.
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    out = subprocess.run(cmd, cwd=str(tmpdir), capture_output=True, text=True, timeout=timeout, env=env)
    lines = [l for l in out.stdout.splitlines() if l.startswith("PYOOMPH_HOPF_RESULT ")]
    assert len(lines) == 1, out.stdout[-4000:] + out.stderr[-4000:]
    res = json.loads(lines[0][len("PYOOMPH_HOPF_RESULT "):])
    assert "error" not in res, res.get("traceback", res.get("error"))
    assert out.returncode == 0, out.stdout[-4000:] + out.stderr[-4000:]
    return res


@pytest.fixture(scope="module")
def coeff_runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("mpi_hopf_coeff")
    return (_run(base / "serial", 1, False, "coeff"),
            _run(base / "replicated", _NPROC, False, "coeff"),
            _run(base / "distributed", _NPROC, True, "coeff"))


@pytest.fixture(scope="module")
def orbit_runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("mpi_hopf_orbit")
    return (_run(base / "serial", 1, False, "orbit"),
            _run(base / "distributed", _NPROC, True, "orbit"))


def test_the_distributed_run_really_was_distributed(coeff_runs):
    serial, replicated, dist = coeff_runs
    assert serial["distributed"] is False and replicated["distributed"] is False, (serial, replicated)
    assert dist["distributed"] is True and dist["nproc"] == _NPROC, dist
    # ndof is the global count, so it must agree whichever way the mesh was cut up.
    assert serial["ndof"] == replicated["ndof"] == dist["ndof"], coeff_runs


@pytest.mark.parametrize("which", [1, 2], ids=["replicated", "distributed"])
def test_coefficient_matches_serial(coeff_runs, which):
    serial = coeff_runs[0]
    other = coeff_runs[which]
    # ga scales with the normalisation of q, so this is a comparison against the SAME discretisation
    # solved a different way -- not against a formula. What is left is the assembly's summation order.
    assert abs(other["ga"] - serial["ga"]) < 1e-9 * abs(serial["ga"]), (serial, other)
    assert abs(other["al"] - serial["al"]) < 1e-6 * abs(serial["al"]), (serial, other)
    assert other["dlam"] == serial["dlam"], (serial, other)
    assert abs(other["omega"] - serial["omega"]) < 1e-9, (serial, other)


def test_orbit_matches_serial(orbit_runs):
    serial, dist = orbit_runs
    assert dist["distributed"] is True, dist
    assert abs(dist["T"] - serial["T"]) < 1e-9 * serial["T"], (serial, dist)
    assert abs(dist["radius_mean"] - serial["radius_mean"]) < 1e-8 * serial["radius_mean"], (serial, dist)
    assert dist["supercritical"] == serial["supercritical"] is True, (serial, dist)


def test_orbit_still_matches_the_exact_limit_cycle_when_distributed(orbit_runs):
    """Not just self-consistent with serial: still the right answer."""
    dist = orbit_runs[1]
    rel = abs(dist["radius_mean"] - dist["radius_exact"]) / dist["radius_exact"]
    assert rel < 5e-3, (rel, dist)
    assert abs(dist["T"] - dist["T_exact"]) < 1e-6 * dist["T_exact"], dist
