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

# Adapting the mesh to an EIGENFUNCTION, under MPI.
#
# pytest runs serially, so each test launches tests/mpi_eigen_adapt_worker.py under mpirun and
# compares against a serial run of the same worker computed in-process.
#
# Problem.refine_eigenfunction() was refused outright on a distributed problem until the history dof
# accessors learnt to work there (commit 2531e00); the refusal was then kept deliberately, because
# the adaptation itself had never been validated distributed. This suite is that validation, and it
# is also the only coverage refine_eigenfunction has at all -- serial included.
#
# What each assertion is for, and why they are not all equalities:
#
#   * cross-rank agreement. Everything reported is global (gathered eigenvector, global ndof,
#     MPI_Allreduce'd integrals, a fingerprint gathered over all ranks), so every rank must report
#     the same numbers. Only the refinement COUNTS are rank-local and excluded.
#
#   * replicated `mpirun` == serial, exactly, mesh included. Without --distribute every rank holds
#     the whole mesh and every rank computes every Z2 patch, so there is nothing left to differ.
#
#   * the carry-across is exact. With ONE adaptation and no unrefinement, refining an element leaves
#     the FE function it interpolates unchanged, so the eigenfunction integrals must survive the
#     round trip through history levels 3 and 4 to round-off -- on any partition. This is the
#     mechanism the old refusal was about and the assertion that would catch it breaking.
#
#   * --distribute is compared to serial on PHYSICS, not on the mesh. oomph-lib's distributed Z2
#     recovery neglects the flux contributions of patches that can only be assembled from vertex
#     nodes owned by another process (the long "NOTE FOR FUTURE REFERENCE" in
#     LagrZ2ErrorEstimator::setup_patches, src/lagr_error_estimator.cpp), so elements sitting near
#     the refinement threshold can be decided differently and the meshes differ by a per cent or so.
#     test_mesh_difference_is_the_estimator_not_the_eigenfunction pins that reading down: the SAME
#     comparison driven by the base state alone has to differ in the same way, so a mesh difference
#     that appears only when adapting to an eigenfunction still fails.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_eigen_adapt_worker.py")


def _mpi_reason():
    """None if a distributed eigen run is possible here, else the reason to skip."""
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
        return "petsc4py not available"
    try:
        import slepc4py  # type:ignore  # noqa: F401
    except Exception:
        return "slepc4py not available"
    return None


_SKIP_REASON = _mpi_reason()
pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]

# Serial and a replicated mpirun compute the identical discrete problem and are compared exactly.
_EXACT_RTOL = 1e-9
# Between ranks of ONE run there is nothing left to differ -- the same collective produced them.
_RANK_RTOL = 1e-12
# The carry-across through the history levels. Exact for the Cartesian case; the azimuthal one comes
# back at ~2e-7, EQUALLY in serial and distributed, so it is a property of the m != 0 machinery (the
# axis dofs are forced to zero by a matrix manipulator rather than pinned) and not of MPI. The
# tolerance is per case for that reason, and the serial value is asserted alongside so a regression
# in the serial path cannot hide behind it.
_CARRY_TOL = {"cartesian": 1e-12, "azimuthal": 1e-5}
# Serial vs --distribute, on a mesh the two need not agree about. The eigenvalue moved by 2e-6
# relative at 4 ranks; anything structurally wrong moves it by percent.
_DIST_EVAL_RTOL = 1e-3
_DIST_NDOF_RTOL = 0.1

_CASES = {"cartesian": None, "azimuthal": 1}


def _run_mpi(nproc, tmpdir, distribute, case="cartesian", size=8, numadapt=2, driver="eigen",
             timeout=1200):
    """Launch the worker under mpirun and return the list of per-rank result dicts."""
    cmd = ["mpirun", "-n", str(nproc)]
    # No --oversubscribe: this project's machines have the cores these ranks need, and an
    # oversubscribed run trades a deadlock for a machine that stops responding.
    cmd += [sys.executable, _WORKER, "--outdir", str(tmpdir), "--case", case,
            "--size", str(size), "--numadapt", str(numadapt), "--driver", driver,
            "--azimuthal-m", str(-1 if _CASES[case] is None else _CASES[case]),
            # Without this only rank 0 reaches stdout (the default MPI output mode is "condensed"),
            # and a per-rank comparison would silently compare one rank with itself.
            "--mpi-output=all"]
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
        # Bounded on purpose: a refinement decision taken on one rank and not another leaves the
        # others waiting in a collective rather than returning. That has to surface as a FAILURE,
        # not as a suite that never finishes.
        raise AssertionError(
            "mpirun did not finish within %d s -- suspect a deadlock (nproc=%d distribute=%s case=%s)."
            "\n--- stdout tail ---\n%s" % (timeout, nproc, distribute, case, (e.stdout or "")[-3000:]))
    per_rank = []
    for line in proc.stdout.splitlines():
        marker = "PYOOMPH_MPI_RESULT "
        if marker in line:
            per_rank.append(json.loads(line[line.index(marker) + len(marker):]))
    if not per_rank:
        raise AssertionError(
            "no results from mpirun (exit %d)\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s"
            % (proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
    assert len(per_rank) == nproc, "reported from %d of %d ranks" % (len(per_rank), nproc)
    _check_no_errors(per_rank)
    return per_rank


def _check_no_errors(per_rank):
    for r in per_rank:
        assert "error" not in r, "failed on rank %d: %s\n%s" % (
            r["rank"], r["error"], r.get("traceback", ""))


_SERIAL_CACHE = {}


def _serial_reference(tmpdir, case="cartesian", size=8, numadapt=2, driver="eigen"):
    """The serial reference, computed once per configuration and reused.

    Cached both because it is half the runtime of the suite and because every call builds another
    Problem in the pytest process, which is worth not doing more often than necessary.
    """
    key = (case, size, numadapt, driver)
    if key in _SERIAL_CACHE:
        return _SERIAL_CACHE[key]
    sys.path.insert(0, _HERE)
    try:
        import mpi_eigen_adapt_worker
        res = mpi_eigen_adapt_worker.solve_case(
            case=case, N=size, numadapt=numadapt, azimuthal_m=_CASES[case], driver=driver,
            outdir=os.path.join(str(tmpdir), "serial_%s_%s_%d" % (case, driver, numadapt)))
        _SERIAL_CACHE[key] = res
        return res
    finally:
        sys.path.remove(_HERE)


def _assert_ranks_agree(per_rank):
    """Every reported quantity is global, so the ranks must report the same one.

    The refinement counts are the exception and are excluded: nrefined() counts what THIS rank
    refined, halo copies included, and the partition decides that.
    """
    rank_local = {"rank", "n_refined", "n_unrefined"}
    ref = per_rank[0]
    for r in per_rank[1:]:
        for k, v in ref.items():
            if k in rank_local:
                continue
            if isinstance(v, dict):
                for kk, vv in v.items():
                    assert r[k][kk] == pytest.approx(vv, rel=_RANK_RTOL), \
                        "rank %d disagrees about %s[%s]: %r vs %r" % (r["rank"], k, kk, r[k][kk], vv)
            elif isinstance(v, float):
                assert r[k] == pytest.approx(v, rel=_RANK_RTOL, abs=1e-14), \
                    "rank %d disagrees about %s: %r vs %r" % (r["rank"], k, r[k], v)
            else:
                assert r[k] == v, "rank %d disagrees about %s: %r vs %r" % (r["rank"], k, r[k], v)


def _assert_carry_is_faithful(res, case):
    """The eigenfunction must survive the round trip through history levels 3 and 4.

    Only meaningful for a SINGLE adaptation: with two, the value measured after the second one has a
    re-solve in between and is legitimately a different eigenfunction. And only while nothing is
    unrefined -- merging four sons back into a father does not preserve the FE function.
    """
    assert res["n_unrefined"] == 0, "an unrefinement invalidates the carry-across invariant"
    before, carry = res["eigfunc_before"], res["eigfunc_carry"]
    for k, v in before.items():
        assert carry[k] == pytest.approx(v, rel=_CARRY_TOL[case]), \
            "the eigenfunction did not survive the adaptation: %s went %r -> %r" % (k, v, carry[k])


@pytest.mark.parametrize("case", sorted(_CASES))
@pytest.mark.parametrize("nproc", [2, 4])
def test_replicated_matches_serial(tmp_path, case, nproc):
    """Plain `mpirun`, no --distribute: the same mesh, to the last element."""
    serial = _serial_reference(tmp_path, case=case)
    per_rank = _run_mpi(nproc, tmp_path, distribute=False, case=case)
    _assert_ranks_agree(per_rank)
    got = per_rank[0]
    assert got["distributed"] is False
    assert got["fingerprint"] == serial["fingerprint"]
    assert got["ndof"] == serial["ndof"]
    assert got["nelement"] == serial["nelement"]
    assert got["eval_re"] == pytest.approx(serial["eval_re"], rel=_EXACT_RTOL)
    for k, v in serial["eigfunc"].items():
        assert got["eigfunc"][k] == pytest.approx(v, rel=_EXACT_RTOL)


@pytest.mark.parametrize("case", sorted(_CASES))
@pytest.mark.parametrize("nproc", [2, 4])
def test_distributed_carry_across_the_adaptation(tmp_path, case, nproc):
    """--distribute, one adaptation: the eigenfunction comes back out of the history levels intact.

    This is the assertion the lifted refusal rests on, and it is partition-independent by
    construction -- it does not care which elements were refined.
    """
    serial = _serial_reference(tmp_path, case=case, numadapt=1)
    _assert_carry_is_faithful(serial, case)
    per_rank = _run_mpi(nproc, tmp_path, distribute=True, case=case, numadapt=1)
    _assert_ranks_agree(per_rank)
    got = per_rank[0]
    assert got["distributed"] is True
    _assert_carry_is_faithful(got, case)
    # ...and the eigenvector really is the global one on the NEW numbering, not a rank's row block.
    assert got["evect_len"] == got["ndof"]


@pytest.mark.parametrize("case", sorted(_CASES))
@pytest.mark.parametrize("nproc", [2, 4])
def test_distributed_matches_serial_physics(tmp_path, case, nproc):
    """--distribute: the eigenvalue and the eigenfunction's position agree with serial.

    Tolerances, not equalities, because the two need not have refined exactly the same elements --
    see the module docstring and the next test.
    """
    serial = _serial_reference(tmp_path, case=case)
    per_rank = _run_mpi(nproc, tmp_path, distribute=True, case=case)
    _assert_ranks_agree(per_rank)
    got = per_rank[0]
    assert got["distributed"] is True
    assert got["evect_len"] == got["ndof"]
    assert got["eval_re"] == pytest.approx(serial["eval_re"], rel=_DIST_EVAL_RTOL)
    assert got["ndof"] == pytest.approx(serial["ndof"], rel=_DIST_NDOF_RTOL)
    # Where the eigenfunction sits on the mesh, normalised out of its arbitrary amplitude. A
    # mis-scattered eigenvector keeps its norm and moves this.
    for k in ("usqr_x", "usqr_y"):
        assert got["eigfunc"][k] / got["eigfunc"]["usqr"] == pytest.approx(
            serial["eigfunc"][k] / serial["eigfunc"]["usqr"], rel=1e-3)


def test_mesh_difference_is_the_estimator_not_the_eigenfunction(tmp_path):
    """Whatever --distribute does to the refined mesh, it does without the eigenfunction too.

    The distributed Z2 recovery is missing the patches it cannot assemble locally, so a distributed
    mesh may differ from the serial one. That is allowed here -- but only if driving the SAME
    estimator from the base state alone differs in the same way. A mesh difference that shows up
    ONLY when adapting to an eigenfunction is a defect in the eigen path and fails this test.
    """
    both = {}
    for driver in ("eigen", "base"):
        serial = _serial_reference(tmp_path / driver, case="cartesian", driver=driver)
        got = _run_mpi(4, tmp_path / driver, distribute=True, case="cartesian", driver=driver)[0]
        both[driver] = (serial["fingerprint"] != got["fingerprint"],
                        abs(got["ndof"] - serial["ndof"]) / serial["ndof"])
    eigen_differs, eigen_dev = both["eigen"]
    base_differs, base_dev = both["base"]
    if eigen_differs:
        assert base_differs, (
            "the mesh differs from serial only when adapting to the eigenfunction "
            "(base-driven ndof deviation %.3g) -- that is not the estimator's known "
            "process-boundary effect" % base_dev)
    assert eigen_dev <= max(3 * base_dev, 0.02), \
        "eigen-driven ndof deviation %.3g is far beyond the base-driven one %.3g" % (eigen_dev, base_dev)
