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

# A tracker-shaped bordered system, assembled and SOLVED, in every regime -- with no tracker.
#
# This is the whole chain a CustomBifurcationTracker will use, exercised one step before the tracker
# itself, which is the point: if this passes and a tracker then fails, the fault is in the tracker
# and not in the machinery under it.
#
#   a Python dof augmentation       the base layout and the equation table come from it
#   the multi-assembly              J and a Hessian-vector product as row blocks (B1)
#   backend.block()/stack()         the 2N+1 bordered system on the augmented layout
#   backend.solve()                 solve_python_built_distributed -> PETSc MPIAIJ + MUMPS
#
# Three regimes have to agree: serial, plain mpirun (replicated) and --distribute. They do, to every
# digit printed -- see the tolerances below.
#
# What is compared, and why only this. Under --distribute the dofs are RENUMBERED, so no
# dof-indexed quantity means the same thing in two regimes. The Frobenius norm of the bordered matrix
# and the norm of the solution are invariant under that renumbering; the entries of the solution are
# not, which is why the worker reports norms rather than vectors. The relative residual
# |Ax-b|/|b| is invariant too, and it is the one check that does not depend on any other run.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_bordered_solve_worker.py")


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
    return None


_SKIP_REASON = _mpi_reason()
pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]

# A direct solve of the same system in a different row order. Measured across serial, np=2 plain,
# np=2 --distribute and np=3 --distribute: the matrix norm and the solution norm agree to every digit
# printed (47.5907051286 and 1.0389625738116), and the relative residual stays between 2e-16 and
# 7e-16. 1e-9 is far looser than that and still far tighter than any real defect: a border in the
# wrong column or a block on the wrong rows moves these by percent.
_RTOL = 1e-9
_RESIDUAL = 1e-10


def _run(nproc, tmpdir, distribute=False, N=4, timeout=600):
    outdir = os.path.join(str(tmpdir), "out")
    os.makedirs(outdir, exist_ok=True)
    cmd = []
    if nproc > 1:
        cmd += ["mpirun", "-n", str(nproc)]
    cmd += [sys.executable, _WORKER, "--outdir", outdir, "--N", str(N)]
    if distribute:
        cmd += ["--distribute"]
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        # A bordered solve is where a layout disagreement turns into a hang rather than an error:
        # handing a solver row counts that do not tile makes it wait for rows nobody will send.
        raise AssertionError(
            "mpirun did not finish within %d s -- suspect a deadlock (nproc=%d distribute=%s)."
            "\n--- stdout tail ---\n%s" % (timeout, nproc, distribute, (e.stdout or "")[-3000:]))
    per_rank = []
    for line in proc.stdout.splitlines():
        if line.startswith("PYOOMPH_MPI_RESULT "):
            per_rank.append(json.loads(line[len("PYOOMPH_MPI_RESULT "):]))
    assert len(per_rank) == nproc, (
        "reported from %d of %d ranks (exit %d)\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s"
        % (len(per_rank), nproc, proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
    for r in per_rank:
        assert r.get("ok"), "rank %s failed: %s\n%s" % (r.get("rank"), r.get("error"), r.get("traceback", ""))
    return sorted(per_rank, key=lambda r: r["rank"])


@pytest.mark.parametrize("nproc,distribute", [(1, False), (2, False), (2, True), (3, True)])
def test_the_bordered_system_is_solved_in_every_regime(tmp_path, nproc, distribute):
    per_rank = _run(nproc, tmp_path, distribute=distribute)
    for r in per_rank:
        assert r["solve_residual"] < _RESIDUAL, (
            "rank %d: |Ax-b|/|b| = %.3e, so the bordered system was not actually solved"
            % (r["rank"], r["solve_residual"]))
        # Every rank must agree about a global reduction; if they do not, one of them is reducing
        # over a different set of rows.
        assert r["A_fro"] == pytest.approx(per_rank[0]["A_fro"], rel=_RTOL)
        assert r["x_norm"] == pytest.approx(per_rank[0]["x_norm"], rel=_RTOL)


@pytest.mark.parametrize("nproc,distribute", [(2, False), (2, True), (3, True)])
def test_mpi_agrees_with_serial(tmp_path, nproc, distribute):
    """The same bordered system, the same answer, however the rows are spread."""
    serial = _run(1, tmp_path / "serial")[0]
    got = _run(nproc, tmp_path / ("np%d%s" % (nproc, "d" if distribute else "")),
               distribute=distribute)[0]
    assert got["A_fro"] == pytest.approx(serial["A_fro"], rel=_RTOL), (
        "the bordered matrix differs from serial: |A|_F = %.14g against %.14g"
        % (got["A_fro"], serial["A_fro"]))
    assert got["rhs_norm"] == pytest.approx(serial["rhs_norm"], rel=_RTOL)
    assert got["naive_norm"] == pytest.approx(serial["naive_norm"], rel=_RTOL), (
        "the solution differs from serial: |x| = %.14g against %.14g"
        % (got["naive_norm"], serial["naive_norm"]))


@pytest.mark.parametrize("nproc", [2, 3])
def test_the_augmented_blocks_tile(tmp_path, nproc):
    """The bordered system is built on the augmented dof layout, so its blocks must tile it."""
    per_rank = _run(nproc, tmp_path, distribute=True)
    n = per_rank[0]["aug"][0]
    expect = 0
    for r in per_rank:
        aug_n, first_row, nrow_local, distributed = r["aug"]
        assert aug_n == n
        assert distributed
        assert first_row == expect, "the augmented blocks do not tile [0,%d): %s" % (
            n, [x["aug"] for x in per_rank])
        expect += nrow_local
    assert expect == n
    base_n = per_rank[0]["base"][0]
    assert n == 2 * base_n + 1, "one vector block plus one parameter over %d base dofs is 2N+1" % base_n


def test_the_replicated_regime_imposes_a_split_for_the_solve(tmp_path):
    """Plain mpirun holds the whole system on every rank, which the solver must NOT be told.

    Handing solve_python_built_distributed (nrow_local=n, first_row=0) from every rank makes PETSc
    read nproc*n global rows -- the counts are supposed to tile -- and the solve hangs. The backend
    imposes a contiguous split instead, exactly as PeriodicDrivingResponse does for its own
    replicated bordered system. The evidence that it works is that this returns at all, with the
    right answer; before the split was imposed, this configuration deadlocked.
    """
    serial = _run(1, tmp_path / "serial")[0]
    per_rank = _run(2, tmp_path / "np2")
    for r in per_rank:
        assert not r["aug"][3], "this test is about the non-distributed augmented layout"
        assert r["naive_norm"] == pytest.approx(serial["naive_norm"], rel=_RTOL)
        assert r["solve_residual"] < _RESIDUAL
