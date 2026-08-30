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

# Distributed (MPI) adaptivity tests for branch mixed_adapt.
#
# pytest itself runs serially, so each test here launches tests/mpi_worker.py under
# `mpirun -n N ... --distribute` in a subprocess, then compares the per-rank results against a serial
# reference computed in-process from the SAME definitions (tests/box_cases.py). Two independent oracles:
#
#   1. cross-rank agreement -- every measured quantity is global (gathered residual, global ndof,
#      MPI_Allreduce'd integral observables), so all ranks must report the same numbers. A partition-
#      dependent answer means something is being computed on the wrong subset of the mesh.
#   2. serial agreement    -- the distributed numbers must match the serial run. This is what catches a
#      globally consistent but WRONG field, which a per-rank residual check alone would happily pass.
#
# Coverage: the box [-0.5,0.5]^2 as pure quads / two triangle splits / a MIXED quad+tri mesh, at three
# refinement states including a two-level non-uniform 2:1 jump, for C1 Poisson, C2 Poisson, coupled
# C2+C1 Poisson, and Taylor-Hood / Crouzeix-Raviart Stokes driven by f = (-y, x).
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

import box_cases

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_worker.py")

# Tolerances. The residual oracle is machine zero for these linear problems; Crouzeix-Raviart is more
# poorly conditioned than Taylor-Hood, hence its own (still tiny) bound. The serial-vs-distributed
# comparison is a comparison of two different summation orders of the same exact computation, so it is
# allowed a relative slack rather than being required bit-identical.
_RES_TOL = {"stokes_cr": 1e-6}
_RES_TOL_DEFAULT = 1e-9
# numpy-allclose semantics. The absolute part is scaled by the LARGEST observable of the case rather than
# being a fixed constant: an observable that happens to be ~0 (e.g. one that vanishes by symmetry) would
# otherwise be compared round-off against round-off and fail for no physical reason.
#
# Crouzeix-Raviart gets a looser relative tolerance for the same reason its residual bound is looser: it is
# markedly worse conditioned than Taylor-Hood (its serial residual sits around 1e-10 rather than at machine
# zero), so re-partitioning it perturbs the observables by ~1e-9 relative. That is solver noise on an
# ill-conditioned system, not a differing field -- a torn or mis-hung interface shifts these integrals by
# orders of magnitude more.
_OBS_RTOL = {"stokes_cr": 1e-7}
_OBS_RTOL_DEFAULT = 1e-9
_OBS_ATOL_REL = 1e-12
# Same Jacobian oracle and same Crouzeix-Raviart exemption as the serial campaign; see the calibration note
# in test_adaptive_2d_campaign.py.
_NEWTON_REDUCTION = 1e-10
_NO_REDUCTION_TEST = {"stokes_cr"}


def _obs_tol(sval, case_scale, eq):
    return _OBS_RTOL.get(eq, _OBS_RTOL_DEFAULT) * abs(sval) + _OBS_ATOL_REL * case_scale


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
              pytest.mark.slow, pytest.mark.campaign]


def _run_distributed(cases, nproc, tmpdir, cases_module="box_cases", timeout=900, extra_env=None):
    """Launch the worker under mpirun and return {case_id: [per-rank result dicts]}."""
    spec = json.dumps([[k, e, list(l)] for k, e, l in cases])
    cmd = ["mpirun", "-n", str(nproc)]
    # No --oversubscribe: this project's machines have the cores these ranks need, and an
    # oversubscribed run trades a deadlock for a machine that stops responding.
    cmd += [sys.executable, _WORKER, "--spec", spec, "--outdir", str(tmpdir),
            "--cases", cases_module, "--distribute"]
    # Importing pyoomph calls MPI_Init, so THIS pytest process is already a (singleton) MPI job and owns an
    # Open MPI session directory under TMPDIR. A nested mpirun collides with it and dies immediately with
    # exit code 1 and no diagnostics at all. Giving the child its own TMPDIR keeps the two session
    # directories apart. (Note this bites whenever any test module in the same pytest process has imported
    # pyoomph, which is essentially always -- so it cannot be avoided by import discipline here.)
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    if extra_env:
        env.update(extra_env)
    # A bounded timeout on purpose: a distributed deadlock (one rank stuck in a collective while the
    # other has finished) must surface as a test FAILURE, not as a suite that never returns. Seen at
    # ~50 minutes on a workload that takes 1.5 s serially, so anything past a few minutes is a hang.
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
    for cid, per_rank in results.items():
        assert len(per_rank) == nproc, "case %s reported from %d of %d ranks" % (cid, len(per_rank), nproc)
        for r in per_rank:
            assert "error" not in r, "case %s failed on rank %d: %s\n%s" % (
                cid, r["rank"], r["error"], r.get("traceback", ""))
    return results


def _assert_matches_serial(cid, per_rank, serial, eq):
    res_tol = _RES_TOL.get(eq, _RES_TOL_DEFAULT)
    obs_keys = [k for k in serial if k.startswith("obs_")]
    assert obs_keys, "%s: no integral observables defined -- the field would go unchecked" % cid
    case_scale = max(abs(serial[k]) for k in obs_keys)
    for r in per_rank:
        # (1) the distributed solve is itself converged
        assert r["maxres"] < res_tol, "%s: max|residual| = %.3e on rank %d" % (cid, r["maxres"], r["rank"])
        # (1b) and converged the way a linear problem should: one Newton step removes the whole residual.
        #      This is the DISTRIBUTED Jacobian being exact, which needs more than the serial case -- the
        #      hanging nodes' values must be collapsed from their (halo) masters before every assembly, or
        #      the Jacobian is built from stale values. Crouzeix-Raviart is exempt for the conditioning
        #      reason documented in test_adaptive_2d_campaign.py.
        conv = r["newton_conv"]
        if eq not in _NO_REDUCTION_TEST and len(conv) >= 2 and conv[0] > 1e-6:
            assert conv[1] / conv[0] < _NEWTON_REDUCTION, \
                "%s: one Newton step reduced the residual by only %.2e on rank %d (history %r)" % (
                    cid, conv[1] / conv[0], r["rank"], conv)
        # (2) same global discretisation -> same hanging-node structure was built
        assert r["ndof"] == serial["ndof"], "%s: ndof %d on rank %d vs %d serially" % (
            cid, r["ndof"], r["rank"], serial["ndof"])
        # (3) the FIELD matches serial (global, halo-free, MPI-reduced integrals)
        for key in obs_keys:
            sval, dval = serial[key], r[key]
            assert abs(dval - sval) <= _obs_tol(sval, case_scale, eq), \
                "%s: %s = %.16g on rank %d vs %.16g serially" % (cid, key, dval, r["rank"], sval)
    # (4) all ranks agree with each other (MPI_Allreduce hands every rank the same sum, so this should be
    #     exact; compared with a tolerance anyway so the test does not depend on that guarantee)
    first = per_rank[0]
    for r in per_rank[1:]:
        assert r["ndof"] == first["ndof"], "%s: ndof differs between rank %d and rank %d" % (
            cid, first["rank"], r["rank"])
        for key in obs_keys:
            assert abs(r[key] - first[key]) <= _obs_tol(first[key], case_scale, eq), \
                "%s: %s differs between rank %d and rank %d" % (cid, key, first["rank"], r["rank"])


def _check(cases, nproc, tmp_path, mod=None, timeout=900, extra_env=None):
    """Solve `cases` distributed and serially from the SAME case module and require them to agree.
    `mod` defaults to the 2D campaign; test_mpi_adaptivity_3d.py passes box_cases_3d."""
    mod = mod or box_cases
    dist = _run_distributed(cases, nproc, tmp_path / "dist", cases_module=mod.__name__, timeout=timeout,
                            extra_env=extra_env)
    for kind, eq, levels in cases:
        cid = mod.case_id(kind, eq, levels)
        serial = mod.solve_case(kind, eq, levels, outdir=str(tmp_path / "serial" / cid))
        assert serial["maxres"] < _RES_TOL.get(eq, _RES_TOL_DEFAULT), \
            "%s: the SERIAL reference did not converge (max|residual| = %.3e)" % (cid, serial["maxres"])
        _assert_matches_serial(cid, dist[cid], serial, eq)


# One test per equation system, sweeping all four discretisations and all three refinement states, so a
# failure names the physics immediately. Each invocation is a single mpirun (the worker loops over the
# cases), which keeps the MPI start-up cost to one per test rather than one per case.
@pytest.mark.parametrize("eq", ["poisson1", "poisson2", "mixed12", "constrain12", "unconstrain12",
                                "neumann", "stokes_th", "ale"])
def test_distributed_adaptive_matches_serial(eq, tmp_path):
    cases = [(kind, eq, lv) for kind in box_cases.MESH_KINDS for lv in box_cases.LEVELS]
    _check(cases, 2, tmp_path)


def test_distributed_c1_constraint_still_reduces_dofs(tmp_path):
    # The C1 field constraint must survive mesh DISTRIBUTION, not merely coexist with it. It is applied
    # from Python by walking the mesh's own elements, so on a distributed mesh each rank only sees its own
    # partition (plus halos) -- if the halo copies were skipped, or if a constrained node's C1 corners
    # landed on another rank, the constraint would silently apply to fewer dofs. Since ndof is global,
    # reproducing the serial ordering constrained < unconstrained-on-top < baseline under --distribute is
    # what shows the constraint was applied to the same set of dofs everywhere.
    levels = (1, 3)
    cases = [(kind, eq, levels) for kind in box_cases.MESH_KINDS
             for eq in ("mixed12", "constrain12", "unconstrain12")]
    dist = _run_distributed(cases, 2, tmp_path / "dist")
    for kind in box_cases.MESH_KINDS:
        got = {eq: dist[box_cases.case_id(kind, eq, levels)][0]["ndof"]
               for eq in ("mixed12", "constrain12", "unconstrain12")}
        assert got["constrain12"] < got["unconstrain12"] < got["mixed12"], \
            "%s under MPI: expected ndof(constrained) < ndof(unconstrained on top) < ndof(baseline), " \
            "got %r" % (kind, got)


def test_distributed_crouzeix_raviart(tmp_path):
    # CR pins an ELEMENT-INTERNAL pressure dof, i.e. the other pressure-fixation path; and its bubble
    # (C2TB) velocity allocates son internal data on refinement. Level 3 is dropped: CR is the most
    # expensive discretisation here and level (1,2) already covers the 2:1 hanging case.
    cases = [(kind, "stokes_cr", lv) for kind in box_cases.MESH_KINDS for lv in [(0, 0), (1, 1), (1, 2)]]
    _check(cases, 2, tmp_path)


def test_distributed_four_ranks(tmp_path):
    # More ranks than the 2-rank default: more partition boundaries cut through the refined band, and the
    # globally selected pressure node is more likely to live on a rank other than 0. Includes the ALE case,
    # where the mesh POSITIONS are distributed unknowns, and the constrained case.
    cases = [(kind, eq, (1, 3)) for kind in box_cases.MESH_KINDS
             for eq in ["poisson2", "unconstrain12", "stokes_th", "ale"]]
    _check(cases, 4, tmp_path)


def test_halo_consistency_check_stays_clean(tmp_path):
    # Runs the campaign with pyoomph's own halo-consistency check armed in THROW mode
    # (PYOOMPH_CHECK_HALO_CONSISTENCY=2), so any adapt during these cases that leaves the processes
    # disagreeing about the elements they share fails the run outright instead of quietly producing a
    # wrong mesh.
    #
    # This pins the invariant behind the defect-C fix (see dev_docs/adaptive_refinement.md section
    # 9.8): pyoomph applies per-element refinement overrides rank-locally after oomph-lib's estimator has
    # synchronised its errors, so without Mesh::synchronise_elemental_errors() the ranks refine different
    # elements. It also pins the CHECK -- if the check itself stops working, the negative test below
    # notices, and if it starts firing spuriously, this one does.
    #
    # Deliberately spans the mechanisms that can drive refinement apart: a mesh-level RefineToLevel plus a
    # boundary-restricted one (the pair that produced the original divergence), across both refinement
    # states and every discretisation.
    cases = [(kind, eq, lv) for kind in box_cases.MESH_KINDS
             for eq in ["poisson2", "unconstrain12", "stokes_th", "ale"]
             for lv in [(1, 1), (1, 3)]]
    _check(cases, 2, tmp_path, extra_env={"PYOOMPH_CHECK_HALO_CONSISTENCY": "2"})


@pytest.mark.parametrize("crit", ["size", "callback"])
def test_distributed_refinement_criteria(crit, tmp_path):
    # The defect-C fix (Mesh::synchronise_elemental_errors) works on the FINAL error vector, so it covers
    # every way of asking for refinement rather than just the RefineToLevel that exposed the bug -- there
    # is exactly one call site into Mesh::adapt(), so nothing can bypass it. This tests that claim instead
    # of relying on it, on the two other criteria pyoomph ships:
    #
    #   size     -- RefineMaxElementSize, stated on the bulk mesh AND on the "top" interface mesh. The
    #               interface-restricted form is the shape that broke: a rank holds halo copies of bulk
    #               elements without the interface elements that would override their error.
    #   callback -- RefineAccordingToElement, whose 2:1 interface runs through the mesh INTERIOR (x=0)
    #               rather than along a boundary, so a partition cut is far more likely to lie along it.
    #
    # Run with the halo-consistency check in throw mode, so a divergence fails at the adapt that caused it
    # rather than being inferred afterwards from a mismatched ndof.
    cases = [(kind, eq, (1, 2, crit)) for kind in box_cases.MESH_KINDS
             for eq in ["poisson2", "unconstrain12", "stokes_th"]]
    _check(cases, 2, tmp_path, extra_env={"PYOOMPH_CHECK_HALO_CONSISTENCY": "2"})
