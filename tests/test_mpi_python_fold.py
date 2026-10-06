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

# The PYTHON FoldTracker under mpirun -- the first of the custom-assembler family to run there.
#
# Everything the tracker stands on is tested on its own: the handoff
# (test_mpi_custom_assembler_handoff), the dof layout (test_mpi_augmentation_layout), the assembly
# (test_mpi_multiassembly), the backend (test_distributed_la, test_mpi_distributed_la) and a bordered
# solve through all of it (test_mpi_bordered_solve). What is left for here is the tracker.
#
# The problem is the one tests/mpi_bifurcation_worker.py uses for the C++ MyFoldHandler -- 2D Bratu,
# whose branch turns at a genuine limit point -- so the two routes can be compared against each
# other. Two independent implementations of the same augmented system agreeing is a stronger
# statement than either agreeing with itself across rank counts.
#
# --distribute is REFUSED, and that refusal is asserted here rather than left untested. The augmented
# system the tracker builds on a partitioned mesh matches the serial one to 1e-13 in every
# permutation invariant (Frobenius norm, trace, nnz, sorted diagonal and row sums, residual norm),
# but the Newton solve converges only linearly -- 6.7e-5 -> 2.3e-5 -> 7.7e-6 -- and stops at the
# iteration cap. The fault is therefore downstream of the assembly and is not yet found, so
# FoldTracker.supports_distributed() returns False and says so by name.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_python_fold_worker.py")


def _mpi_reason():
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
        return "slepc4py not available (the eigenvector guess needs an eigensolve)"
    return None


_SKIP_REASON = _mpi_reason()
pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]

# Serial, np=2 and np=3 all converge to lam = 6.808263809409476 -- every digit -- and the
# eigenfunction integral agrees to ~1e-18, because the replicated regime assembles the same global
# system and solves it the same way. The C++ route lands on the same parameter to 16 digits too.
# 1e-9 is far looser than any of that and still far tighter than a real defect: a misplaced border or
# a wrong constraint row moves a critical parameter by percent.
_PARAM_RTOL = 1e-9
_OBS_RTOL = 1e-8


def _run(nproc, tmpdir, distribute=False, cxx=False, nonlinear=False, N=8,
         expect_failure=False, timeout=900):
    outdir = os.path.join(str(tmpdir), "out")
    os.makedirs(outdir, exist_ok=True)
    cmd = []
    if nproc > 1:
        cmd += ["mpirun", "-n", str(nproc)]
    cmd += [sys.executable, _WORKER, "--outdir", outdir, "--N", str(N)]
    if distribute:
        cmd += ["--distribute"]
    if cxx:
        cmd += ["--cxx"]
    if nonlinear:
        cmd += ["--nonlinear-constraint"]
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        raise AssertionError(
            "mpirun did not finish within %d s -- suspect a deadlock (nproc=%d distribute=%s cxx=%s)."
            "\n--- stdout tail ---\n%s" % (timeout, nproc, distribute, cxx, (e.stdout or "")[-3000:]))
    per_rank = []
    for line in proc.stdout.splitlines():
        if line.startswith("PYOOMPH_MPI_RESULT "):
            per_rank.append(json.loads(line[len("PYOOMPH_MPI_RESULT "):]))
    assert len(per_rank) == nproc, (
        "reported from %d of %d ranks (exit %d)\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s"
        % (len(per_rank), nproc, proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
    if not expect_failure:
        for r in per_rank:
            assert r.get("ok"), "rank %s failed: %s\n%s" % (r.get("rank"), r.get("error"), r.get("traceback", ""))
    return sorted(per_rank, key=lambda r: r["rank"])


@pytest.mark.parametrize("nproc", [1, 2, 3])
def test_the_python_fold_tracker_finds_the_fold(tmp_path, nproc):
    per_rank = _run(nproc, tmp_path)
    for r in per_rank:
        assert r["supports_mpi"] is True
        # [base | V | parameter]: the group layout the bordered system is laid out from.
        assert r["groups"] == [False, False, True]
        assert r["ndof_aug"] == 2 * r["ndof_base"] + 1
        # The augmentation must be gone again afterwards.
        assert r["ndof_after"] == r["ndof_base"]
        # The eigenvector comes back globally replicated at full length, which every consumer of
        # get_last_eigenvectors() assumes (dev_docs/mpi_eigenproblems.md).
        assert r["evect_len"] == r["ndof_base"]
    # Every rank reads the same converged augmented state.
    for r in per_rank[1:]:
        assert r["critical"] == pytest.approx(per_rank[0]["critical"], rel=1e-12)
        assert r["eigfunc_usqr"] == pytest.approx(per_rank[0]["eigfunc_usqr"], rel=1e-12)


@pytest.mark.parametrize("nproc", [2, 3])
def test_mpirun_agrees_with_serial(tmp_path, nproc):
    serial = _run(1, tmp_path / "serial")[0]
    got = _run(nproc, tmp_path / ("np%d" % nproc))[0]
    assert got["critical"] == pytest.approx(serial["critical"], rel=_PARAM_RTOL), (
        "np=%d found the fold at %.17g, serial at %.17g" % (nproc, got["critical"], serial["critical"]))
    # The mesh integral of the squared eigenfunction: the one assertion that constrains WHERE on the
    # mesh the eigenvector's entries ended up, which a wrong translation would move while leaving the
    # critical parameter alone.
    assert got["eigfunc_usqr"] == pytest.approx(serial["eigfunc_usqr"], rel=_OBS_RTOL)


@pytest.mark.parametrize("nproc", [1, 2])
def test_the_python_route_agrees_with_the_cxx_handler(tmp_path, nproc):
    """Two independent implementations of the same augmented system, on the same problem.

    The C++ MyFoldHandler has been correct under MPI for a while, so this is the strongest check
    available on the Python one -- stronger than it agreeing with itself across rank counts.
    """
    py = _run(nproc, tmp_path / "py")[0]
    cxx = _run(nproc, tmp_path / "cxx", cxx=True)[0]
    assert py["ndof_aug"] == cxx["ndof_aug"], "the two routes build different-sized augmented systems"
    assert py["critical"] == pytest.approx(cxx["critical"], rel=_PARAM_RTOL), (
        "the Python tracker found the fold at %.17g, the C++ handler at %.17g"
        % (py["critical"], cxx["critical"]))
    assert py["eigfunc_usqr"] == pytest.approx(cxx["eigfunc_usqr"], rel=_OBS_RTOL)


def test_the_nonlinear_length_constraint_also_works_under_mpirun(tmp_path):
    """<V,V> instead of <V,V0>: the normalisation row then depends on V and has to be re-replicated
    on every assembly, which is a different code path through the border row."""
    serial = _run(1, tmp_path / "serial", nonlinear=True)[0]
    got = _run(2, tmp_path / "np2", nonlinear=True)[0]
    assert got["critical"] == pytest.approx(serial["critical"], rel=_PARAM_RTOL)
    # The same fold, whichever constraint is used to pin the eigenvector's length.
    plain = _run(1, tmp_path / "plain")[0]
    assert serial["critical"] == pytest.approx(plain["critical"], rel=1e-6)


def test_distribute_is_refused_by_name(tmp_path):
    """Not an oversight: the assembly is verified there but the solve does not converge yet.

    Asserted so that lifting supports_distributed() has to come with this test changing, rather than
    the refusal quietly outliving the problem it describes.
    """
    per_rank = _run(2, tmp_path, distribute=True, expect_failure=True)
    assert all(not r["ok"] for r in per_rank)
    for r in per_rank:
        msg = str(r.get("error", ""))
        assert "FoldTracker" in msg and "--distribute" in msg, msg
        assert "converge" in msg, "the refusal must say what is wrong, not just that it is: " + msg
