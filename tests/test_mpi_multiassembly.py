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

# The base-problem multi-assembly under MPI: B1 of mpi_augmented_systems.md.
#
# Problem::sparse_assemble_row_or_column_compressed_base_problem() used to throw the moment
# nproc > 1 ("This likely does not work in parallel" / "...in distributed parallel"). It keeps this
# rank's element slice and nothing in it reduced over the ranks, so lifting the throw on its own
# would have handed each rank a partial residual and Jacobian. It now delegates to oomph's own
# parallel_sparse_assemble(), which does the off-processor row exchange -- and that is the point
# worth stating: the reduction is NOT a sum of disjoint slices, because a base equation on a
# partition boundary collects contributions from non-halo elements on several ranks.
#
# Two comparisons, because only one of them is available in each regime:
#
#   - REPLICATED (plain mpirun) keeps serial's dof numbering, so the gathered row blocks are compared
#     to the serial CSR entry by entry, pattern included. This is the assertion that a missing or
#     wrong reduction cannot survive.
#   - --distribute renumbers (distribute() gives each rank a contiguous block), so nothing indexed by
#     dof is comparable. The checks there are invariant under a permutation of the unknowns: total
#     nnz and the Frobenius norm of each matrix.
#
# Reached through MultiAssembleRequest directly. A dof augmentation is installed only because the
# base-problem routine insists on one -- it exists to assemble the base block OF an augmented system
# -- and no custom assembler is, so Problem.set_custom_assembler's nproc>1 refusal is not involved.

import json
import os
import shutil
import subprocess
import sys

import numpy
import pytest
import scipy.sparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_multiassembly_worker.py")


def _mpi_reason():
    if shutil.which("mpirun") is None:
        return "mpirun not found"
    try:
        from pyoomph.generic.mpi import has_mpi
        if not has_mpi():
            return "pyoomph was built without MPI"
    except Exception as e:
        return "MPI unavailable: " + str(e)
    return None


_SKIP_REASON = _mpi_reason()
pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]

# Replicated runs assemble over the SAME element set as serial and differ only in the order the
# contributions are summed, so they agree to round-off rather than to engineering accuracy. Measured
# spread at np=2/3/4 on the 49-dof case: 0, 0 and 4.4e-16 absolute on the Jacobian. A missing
# reduction moves entries by O(1) -- each rank would hold only its own slice's contributions.
_ATOL = 1e-12
# The Frobenius norm across a renumbering agrees to ~1e-14 relative (measured: 33.100820112786 in
# every regime). Same reasoning: a permutation cannot move it, a wrong reduction can.
_FRO_RTOL = 1e-10


def _run(nproc, tmpdir, distribute=False, N=4, timeout=900):
    """Run the worker and return (per-rank result dicts, the gathered J, the gathered R)."""
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

    # The row blocks come back as files, not on stdout: every rank writes the same pipe and a line of
    # a few thousand characters does not arrive atomically.
    blocks = []
    for rank in range(nproc):
        z = numpy.load(os.path.join(outdir, "block_rank%d.npz" % rank))
        first_row, nrow_local, n = int(z["first_row"][0]), int(z["nrow_local"][0]), int(z["n"][0])
        blocks.append((first_row,
                       scipy.sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(nrow_local, n)),
                       z["R"]))
    blocks.sort(key=lambda b: b[0])
    # The blocks come out on the BASE DOF layout. Under --distribute that is a genuine partition and
    # they concatenate; under a replicated mpirun the base layout is non-distributed, so every rank
    # returns the WHOLE system and concatenating would repeat it nproc times.
    replicated = all(b[1].shape[0] == b[1].shape[1] for b in blocks) and len(blocks) > 1 \
        and all(b[0] == 0 for b in blocks)
    if replicated:
        J, R = blocks[0][1].tocsr(), blocks[0][2]
    else:
        J = scipy.sparse.vstack([b[1] for b in blocks], format="csr").tocsr()
        R = numpy.concatenate([b[2] for b in blocks])
    return sorted(per_rank, key=lambda r: r["rank"]), J, R


def test_the_blocks_come_out_on_the_base_dof_layout(tmp_path):
    """Whatever the regime, the blocks must be on the BASE DOF layout, not some other split.

    That is the contract the rest of the machinery rests on: the augmented dof layout is built from
    the base one, so a tracker's bordered system can only line up if its base blocks do. An earlier
    version targeted a fresh uniform split of the base equations instead -- 24/25 against the dof
    layout's 21/28 at np=2 -- which is the same "two partitions of the same rows" mistake as B2.
    """
    for distribute in (False, True):
        per_rank, J, _R = _run(2, tmp_path / ("d" if distribute else "p"), distribute=distribute)
        n = per_rank[0]["n"]
        expect = 0
        for r in per_rank:
            assert r["n"] == n
            # Every returned matrix is a (nrow_local x n) block with global column indices.
            for shape in r["shapes"]:
                assert shape == [r["nrow_local"], n]
            for length in r["vec_lengths"]:
                assert length == r["nrow_local"]
            # oomph's distributed assembly returns each row in the order it met the entries; an
            # unsorted CSR is wrong to hand onward (PETSc's createAIJ wants ascending columns, and
            # petsc.py's reuse digest hashes the index arrays), so assemble() sorts.
            assert r["J_sorted"], "rank %d returned an unsorted CSR" % r["rank"]
            if distribute:
                assert r["first_row"] == expect, "the row blocks do not tile [0,%d): %s" % (
                    n, [(x["first_row"], x["nrow_local"]) for x in per_rank])
                expect += r["nrow_local"]
                assert r["nrow_local"] < n, "rank %d got the whole system under --distribute" % r["rank"]
            else:
                # Replicated: the base dof layout is non-distributed, so each rank holds all of it.
                assert (r["first_row"], r["nrow_local"]) == (0, n)
        if distribute:
            assert expect == n
        assert J.has_sorted_indices


@pytest.mark.parametrize("nproc", [2, 3, 4])
def test_replicated_assembly_reproduces_the_serial_system(tmp_path, nproc):
    """Plain mpirun keeps serial's numbering, so compare the whole CSR entry by entry.

    EVERY rank holds the whole system here -- the base dof layout is non-distributed, and the
    assembly is replicated to match it -- so this compares each rank's own copy against serial, which
    is a stronger statement than the blocks merely concatenating to it.
    """
    _s, Js, Rs = _run(1, tmp_path / "serial")
    _p, J, R = _run(nproc, tmp_path / ("np%d" % nproc))

    assert J.shape == Js.shape
    assert J.nnz == Js.nnz, "np=%d assembled %d nonzeros where serial has %d" % (nproc, J.nnz, Js.nnz)
    assert numpy.array_equal(J.indptr, Js.indptr), "the row counts differ from serial"
    assert numpy.array_equal(J.indices, Js.indices), "the sparsity pattern differs from serial"
    assert numpy.allclose(J.data, Js.data, rtol=0, atol=_ATOL), (
        "np=%d Jacobian differs from serial by up to %.3e" % (nproc, numpy.abs(J.data - Js.data).max()))
    assert numpy.allclose(R, Rs, rtol=0, atol=_ATOL), (
        "np=%d residual differs from serial by up to %.3e" % (nproc, numpy.abs(R - Rs).max()))


@pytest.mark.parametrize("nproc", [2, 3])
def test_distributed_assembly_agrees_up_to_the_renumbering(tmp_path, nproc):
    """--distribute renumbers, so compare only what a permutation cannot move."""
    _s, Js, _Rs = _run(1, tmp_path / "serial")
    per_rank, J, _R = _run(nproc, tmp_path / ("np%d" % nproc), distribute=True)

    assert J.nnz == Js.nnz, "--distribute at np=%d assembled %d nonzeros where serial has %d" % (
        nproc, J.nnz, Js.nnz)
    fro = float(numpy.sqrt((J.data ** 2).sum()))
    fro_s = float(numpy.sqrt((Js.data ** 2).sum()))
    assert fro == pytest.approx(fro_s, rel=_FRO_RTOL), (
        "--distribute at np=%d gives |J|_F = %.14g where serial gives %.14g" % (nproc, fro, fro_s))
    # Each of the other assembled quantities too, from the per-rank summaries.
    for name in ("J", "dJdp", "HV"):
        total_nnz = sum(r["nnz"][name] for r in per_rank)
        combined = float(numpy.sqrt(sum(r["fro"][name] ** 2 for r in per_rank)))
        assert total_nnz == Js.nnz, "%s has %d nonzeros, expected %d" % (name, total_nnz, Js.nnz)
        assert combined > 0.0, "%s came back empty" % name


def test_the_assembly_no_longer_refuses_mpi(tmp_path):
    """The throws this replaced were unconditional; a run that gets a result at all has cleared them."""
    per_rank, _J, _R = _run(2, tmp_path, distribute=False)
    assert all(r["ok"] for r in per_rank)
    per_rank, _J, _R = _run(2, tmp_path / "dist", distribute=True)
    assert all(r["ok"] for r in per_rank)
