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

# The collective half of pyoomph/generic/distributed_la.py.
#
# tests/test_distributed_la.py fabricates a multi-rank split in one process, which covers every bit
# of layout and translation arithmetic and none of the communication. These cover what it cannot: the
# reductions, to_global(), global_value(), and the PETSc matrix -- its matrix-vector product and its
# transpose. The reference is a global system built identically on every rank from a fixed seed, so
# nothing has to be shipped between ranks to check an answer.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_distributed_la_worker.py")


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
        from petsc4py import PETSc  # type:ignore  # noqa: F401
    except Exception:
        return "petsc4py not available (the distributed matrix needs it)"
    return None


_SKIP_REASON = _mpi_reason()
pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]

# Every quantity here is a reduction over the same numbers in a different order, so round-off is the
# only difference there should be. Measured at np=2/3/4 on a 24-row system: 4e-16 on an inner
# product, 9e-16 on a norm, 1.8e-15 on a matrix-vector product and 3.6e-15 on the transpose
# identity. A layout off by one row moves these by O(1).
_ATOL = 1e-11


def _run(nproc, tmpdir, n=24, timeout=600):
    cmd = ["mpirun", "-n", str(nproc), sys.executable, _WORKER, "--n", str(n)]
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        raise AssertionError(
            "mpirun did not finish within %d s -- suspect a deadlock, which is what a collective "
            "reached by only some ranks looks like (nproc=%d).\n--- stdout tail ---\n%s"
            % (timeout, nproc, (e.stdout or "")[-3000:]))
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


@pytest.mark.parametrize("nproc", [2, 3, 4])
def test_the_layouts_tile_and_petsc_is_chosen(tmp_path, nproc):
    per_rank = _run(nproc, tmp_path)
    n = per_rank[0]["layout"][0]
    expect = 0
    for r in per_rank:
        layout_n, first_row, nrow_local, distributed = r["layout"]
        assert layout_n == n
        assert distributed, "rank %d's layout is not distributed, so nothing here is tested" % r["rank"]
        assert first_row == expect, "the layouts do not tile [0,%d): %s" % (n, [x["layout"] for x in per_rank])
        expect += nrow_local
        # More than one rank means PETSc, and the matrix must really be the PETSc one -- the scipy
        # fallback would replicate the operand instead of multiplying in parallel.
        assert r["backend"] == "petsc", "rank %d chose the %s backend" % (r["rank"], r["backend"])
        assert r["matrix_class"] == "_PETScMatrix"
    assert expect == n


@pytest.mark.parametrize("nproc", [2, 3, 4])
def test_the_reductions_agree_with_the_global_reference(tmp_path, nproc):
    for r in _run(nproc, tmp_path):
        assert r["dot"] == pytest.approx(r["dot_ref"], abs=_ATOL), "rank %d's dot product" % r["rank"]
        assert r["norm"] == pytest.approx(r["norm_ref"], abs=_ATOL), "rank %d's norm" % r["rank"]
        assert r["max_abs"] == pytest.approx(r["max_abs_ref"], abs=_ATOL), "rank %d's max" % r["rank"]
    # A reduction every rank must see the same answer to; disagreement means one of them is reducing
    # over a different set.
    ref = _run(nproc, tmp_path)[0]["dot"]
    for r in _run(nproc, tmp_path):
        assert r["dot"] == pytest.approx(ref, abs=_ATOL)


@pytest.mark.parametrize("nproc", [2, 3])
def test_the_named_escapes_replicate_correctly(tmp_path, nproc):
    """to_global() and global_value() are the two O(n) reads; they must give the global answer."""
    for r in _run(nproc, tmp_path):
        assert r["to_global_matches"], "rank %d's to_global() is not the global vector" % r["rank"]
        for got, want in zip(r["global_value"], r["global_value_ref"]):
            assert got == pytest.approx(want, abs=_ATOL), "rank %d's global_value()" % r["rank"]


@pytest.mark.parametrize("nproc", [2, 3, 4])
def test_the_petsc_matrix_multiplies_and_transposes(tmp_path, nproc):
    for r in _run(nproc, tmp_path):
        assert r["matvec_max_err"] == pytest.approx(0.0, abs=_ATOL), (
            "rank %d's A@x differs from the reference by %.3e" % (r["rank"], r["matvec_max_err"]))
        assert r["frobenius"] == pytest.approx(r["frobenius_ref"], abs=_ATOL)
        # The transpose is checked by the identity rather than by comparing row blocks, because the
        # transposed matrix need not land on the same split.
        assert r["ATx_dot_y"] == pytest.approx(r["x_dot_Ay"], abs=_ATOL), (
            "rank %d: <A^T x, y> = %.17g but <x, A y> = %.17g" % (r["rank"], r["ATx_dot_y"], r["x_dot_Ay"]))


@pytest.mark.parametrize("nproc", [2, 3, 4])
def test_transpose_onto_lands_on_the_given_layout(tmp_path, nproc):
    """Block-by-block against the global transpose, which M.transpose() cannot be compared to.

    transpose_onto exists because a bordered system needs J^T on the SAME row layout as its other
    blocks. PETSc's own transpose returns its ownership range -- a different partition of the same
    rows -- and putting that into block() would place every entry correctly by index and wrongly by
    owner, which is the B2 class of mistake: a plausible matrix, not an error. So this compares the
    actual entries, not an invariant.
    """
    per_rank = _run(nproc, tmp_path)
    for r in per_rank:
        assert r["transpose_onto_shape"] == r["transpose_onto_shape_ref"], (
            "rank %d got shape %r, the global transpose's block is %r"
            % (r["rank"], r["transpose_onto_shape"], r["transpose_onto_shape_ref"]))
        assert r["transpose_onto_nnz"] == r["transpose_onto_nnz_ref"], (
            "rank %d got %d nonzeros, expected %d"
            % (r["rank"], r["transpose_onto_nnz"], r["transpose_onto_nnz_ref"]))
        assert r["transpose_onto_max_err"] < 1e-14, (
            "rank %d: largest entry difference from the global transpose is %.3e"
            % (r["rank"], r["transpose_onto_max_err"]))
        # Canonical column order, because PETSc's createAIJ requires it and petsc.py's reuse digest
        # hashes the index arrays.
        assert r["transpose_onto_sorted"] is True, "rank %d returned an unsorted CSR" % r["rank"]
