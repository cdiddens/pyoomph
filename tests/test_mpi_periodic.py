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

# Distributed (MPI) tests for periodic boundary conditions.
#
# Periodicity is pointer aliasing, not a constraint equation: BoundaryNodeBase::make_node_periodic
# points the copy node's Value/Eqn_number arrays at its master's, and the copy's assign_eqn_numbers()
# is a no-op. oomph-lib's distribution machinery knows nothing about that -- mesh.cc, problem.cc and
# refineable_mesh.cc do not contain the word "periodic" -- and pyoomph used to refuse the combination
# outright ("Distributed parallel with copied nodes ... does not work with nodal degrees of freedom").
# See dev_docs/distributed_periodic_bc.md for what had to change.
#
# pytest itself runs serially, so each test launches tests/mpi_periodic_worker.py under
# `mpirun -n N ... --distribute` in a subprocess and compares the per-rank results against a serial
# reference computed in-process from the SAME definitions (tests/periodic_cases.py). Three oracles:
#
#   1. ndof -- global, and EXACT. A periodic link that does not survive distribution is silent: the
#      master's destructor deep-copies every remaining copy and clears Copied_node_pt, after which the
#      copies contribute their own equations. So a lost seam shows up here as a larger ndof, with no
#      error and a solve that still converges.
#   2. serial agreement of the integral observables -- catches a globally consistent but WRONG field,
#      which the residual alone would happily pass. In particular it catches the shared Eqn_number
#      array being bumped twice by synchronise_eqn_numbers (invisible on rank 0, which has base 0).
#   3. cross-rank agreement -- every measured quantity is global, so all ranks must report the same.
#
# The controls (line1d_nonper, quad2d_nonper) are the same discretisations without periodicity: they
# are what shows the oracles can tell a periodic mesh from a non-periodic one at all.
#
# Every case builds its periodicity with PeriodicBC -- the mesh classes have no periodic argument.
#
# The tests skip (rather than fail) when mpirun or an MPI-capable solver is missing, so the suite stays
# runnable on a serial-only install.

import json
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import periodic_cases

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_periodic_worker.py")

# All cases are linear, so the residual after one Newton step is machine zero iff the (distributed)
# Jacobian is exact.
_RES_TOL = 1e-9
_NEWTON_REDUCTION = 1e-10
# numpy-allclose semantics, with the absolute part scaled by the largest observable of the case: intu
# vanishes by symmetry in every periodic case, and comparing round-off against round-off would fail
# for no physical reason.
_OBS_RTOL = 1e-9
_OBS_ATOL_REL = 1e-12
# The seam jump is exactly zero when the aliasing survives (the two nodes read the same doubles), and
# O(the solution) when it does not. The bound only has to sit far below the latter.
_SEAM_TOL = 1e-14

_N = 16


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
# Skipped when MPI is unavailable, and -- because each test launches mpirun and re-solves every case
# serially for its reference -- also held back from the fast run (see conftest.py).
pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]


def _run_distributed(cases, nproc, tmpdir, timeout=600, N=_N, allow_errors=False):
    """Launch the worker under mpirun and return {case: [per-rank result dicts]}."""
    cmd = ["mpirun", "-n", str(nproc)]
    if os.environ.get("PYOOMPH_MPI_OVERSUBSCRIBE", "1") == "1":
        # CI machines routinely have fewer slots than we ask for; without this OpenMPI refuses to start.
        cmd += ["--oversubscribe"]
    cmd += [sys.executable, _WORKER, "--spec", json.dumps(list(cases)), "--outdir", str(tmpdir),
            "--N", str(N), "--distribute"]
    # Importing pyoomph calls MPI_Init, so THIS pytest process is already a (singleton) MPI job and owns
    # an Open MPI session directory under TMPDIR. A nested mpirun collides with it and dies immediately
    # with exit code 1 and no diagnostics at all. Giving the child its own TMPDIR keeps them apart.
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    # A bounded timeout on purpose: a distributed deadlock must surface as a test FAILURE, not as a
    # suite that never returns. The whole sweep runs in a couple of seconds per rank.
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        raise AssertionError(
            "mpirun did not finish within %d s -- suspect a distributed deadlock.\ncases: %r\n"
            "--- stdout tail ---\n%s" % (timeout, cases, (e.stdout or b"")[-3000:]))
    results = {}
    for line in proc.stdout.splitlines():
        if line.startswith("PYOOMPH_MPI_RESULT "):
            payload = json.loads(line[len("PYOOMPH_MPI_RESULT "):])
            results.setdefault(payload["case"], []).append(payload)
    if not results:
        raise AssertionError(
            "no results from mpirun (exit %d)\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s"
            % (proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
    for case, per_rank in results.items():
        assert len(per_rank) == nproc, "case %s reported from %d of %d ranks" % (
            case, len(per_rank), nproc)
        if not allow_errors:
            for r in per_rank:
                assert "error" not in r, "case %s failed on rank %d: %s\n%s" % (
                    case, r["rank"], r["error"], r.get("traceback", ""))
    return results


def _obs_tol(sval, case_scale):
    return _OBS_RTOL * abs(sval) + _OBS_ATOL_REL * case_scale


def _assert_matches_serial(case, per_rank, serial):
    obs_keys = [k for k in serial if k.startswith("obs_")]
    assert obs_keys, "%s: no integral observables defined -- the field would go unchecked" % case
    case_scale = max(abs(serial[k]) for k in obs_keys)
    for r in per_rank:
        # (1) the distributed solve is itself converged, in one Newton step as a linear problem should
        assert r["maxres"] < _RES_TOL, "%s: max|residual| = %.3e on rank %d" % (
            case, r["maxres"], r["rank"])
        conv = r["newton_conv"]
        if len(conv) >= 2 and conv[0] > 1e-6:
            assert conv[1] / conv[0] < _NEWTON_REDUCTION, \
                "%s: one Newton step reduced the residual by only %.2e on rank %d (history %r)" % (
                    case, conv[1] / conv[0], r["rank"], conv)
        # (2) the periodic links survived distribution at all
        assert r["has_periodic_nodes"] == serial["has_periodic_nodes"], \
            "%s: rank %d %s periodic nodes, serially it %s" % (
                case, r["rank"], "has" if r["has_periodic_nodes"] else "has no",
                "does" if serial["has_periodic_nodes"] else "does not")
        assert r["seam_jump"] <= _SEAM_TOL, \
            "%s: the two sides of the periodic seam differ by %.3e on rank %d -- the copy node is no " \
            "longer reading its master's values" % (case, r["seam_jump"], r["rank"])
        # (3) the same global discretisation: a lost periodic link would show up as extra dofs
        assert r["ndof"] == serial["ndof"], "%s: ndof %d on rank %d vs %d serially" % (
            case, r["ndof"], r["rank"], serial["ndof"])
        # (4) the FIELD matches serial (global, halo-free, MPI-reduced integrals)
        for key in obs_keys:
            sval, dval = serial[key], r[key]
            assert abs(dval - sval) <= _obs_tol(sval, case_scale), \
                "%s: %s = %.16g on rank %d vs %.16g serially" % (case, key, dval, r["rank"], sval)
    # (5) all ranks agree with each other
    first = per_rank[0]
    for r in per_rank[1:]:
        assert r["ndof"] == first["ndof"], "%s: ndof differs between rank %d and rank %d" % (
            case, first["rank"], r["rank"])
        for key in obs_keys:
            assert abs(r[key] - first[key]) <= _obs_tol(first[key], case_scale), \
                "%s: %s differs between rank %d and rank %d" % (case, key, first["rank"], r["rank"])


def _serial(case, tmp_path):
    ref = periodic_cases.run_case(case, N=_N, outdir=str(tmp_path / "serial" / case))
    assert ref["maxres"] < _RES_TOL, \
        "%s: the SERIAL reference did not converge (max|residual| = %.3e)" % (case, ref["maxres"])
    return ref


# Two ranks and four. Two is the minimum that can put the seam's ends on different processors at all;
# four is where a doubly periodic mesh has partition boundaries crossing BOTH seams, and where the
# "overlooked halo node" reconciliation -- the part of oomph-lib that goes through an intermediate
# processor -- actually has intermediates to go through.
@pytest.mark.parametrize("nproc", [2, 4])
@pytest.mark.parametrize("case", periodic_cases.CASES)
def test_distributed_periodic_matches_serial(case, nproc, tmp_path):
    dist = _run_distributed([case], nproc, tmp_path / "dist")
    _assert_matches_serial(case, dist[case], _serial(case, tmp_path))


def test_periodicity_actually_removes_dofs_when_distributed(tmp_path):
    # The oracles above only mean something if a periodic mesh is distinguishable from a non-periodic
    # one by them. Both discretisations are the same 16x16 quad mesh, so ndof differs ONLY because the
    # periodic seam merges one column of nodes into the other; reproducing that difference under
    # --distribute is what shows the merge survived partitioning rather than being undone silently.
    cases = ["quad2d_x", "quad2d_xy", "quad2d_nonper"]
    dist = _run_distributed(cases, 4, tmp_path / "dist")
    got = {c: dist[c][0]["ndof"] for c in cases}
    assert got["quad2d_xy"] > got["quad2d_x"], \
        "doubly periodic should keep more free dofs than singly periodic + Dirichlet, got %r" % (got,)
    assert got["quad2d_x"] > got["quad2d_nonper"], \
        "x-periodic should keep more free dofs than fully pinned, got %r" % (got,)
    for c in cases:
        assert dist[c][0]["ndof"] == _serial(c, tmp_path)["ndof"], \
            "%s: distributed ndof %d != serial" % (c, dist[c][0]["ndof"])


def test_adapting_a_distributed_periodic_mesh_is_refused(tmp_path):
    # Mesh::distribute() ends with setup_tree_forest(), which rebuilds tree neighbours by matching
    # shared nodes; a periodic master and its copy are distinct Node pointers, so the
    # TreeRoot::Neighbour_periodic links do not survive. Refining afterwards would mint ordinary,
    # non-periodic nodes along the seam and the solution would quietly stop being periodic -- so the
    # refusal is the feature, and it has to name the reason rather than fail somewhere downstream.
    dist = _run_distributed(["quad2d_x_adaptive"], 2, tmp_path / "dist", allow_errors=True)
    per_rank = dist["quad2d_x_adaptive"]
    for r in per_rank:
        assert "error" in r, "rank %d refined a distributed periodic mesh instead of refusing" % r["rank"]
        assert "periodic" in r["error"] and "distribute" in r["error"], \
            "rank %d refused for the wrong reason: %s" % (r["rank"], r["error"])


def test_periodic_moving_mesh_is_refused_when_distributed(tmp_path):
    # make_periodic() aliases only the nodal values, never the positions, so on a moving mesh a
    # periodic copy carries position dofs of its own -- and the copy is deliberately kept out of the
    # distributed halo scheme, which is exactly what those dofs would need. Numbering them on every
    # rank that holds the copy would be silent, so the run is stopped instead. Every rank has to
    # refuse, including one whose partition holds no periodic node at all (the check is reduced over
    # ranks); a one-sided raise would deadlock the others in the next collective.
    dist = _run_distributed(["quad2d_x_ale"], 2, tmp_path / "dist", allow_errors=True)
    per_rank = dist["quad2d_x_ale"]
    for r in per_rank:
        assert "error" in r, \
            "rank %d distributed a periodic moving mesh instead of refusing" % r["rank"]
        assert "moving" in r["error"] and "distribute" in r["error"], \
            "rank %d refused for the wrong reason: %s" % (r["rank"], r["error"])
