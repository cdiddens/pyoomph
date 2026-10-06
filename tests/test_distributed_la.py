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

# pyoomph/generic/distributed_la.py, without mpirun.
#
# The row layouts and the equation table are plain data, so a multi-rank split can be FABRICATED in
# one process: build each would-be rank's layout and its slice of the operands, call block() once per
# rank, and reassemble. That is the whole value of these tests -- the offset and translation
# arithmetic is the error-prone part, a wrong offset produces a plausible matrix rather than a crash,
# and reproducing it under mpirun would make a fast unit test into a slow integration one.
# tests/test_mpi_row_layout.py tests mpi_row_layout_from_gathered the same way and for the same
# reason.
#
# The reference is always the idiom being replaced: scipy.sparse.block_array on the whole system.

import numpy
import pytest
import scipy.sparse

from pyoomph.generic.distributed_la import (DistMatrix, DistVector, RowLayout, ScipyBackend,
                                            get_la_backend)


def _rng():
    return numpy.random.default_rng(20261006)


def _fold_pieces(nbase, seed=0):
    """A fold-shaped bordered system: [[J, 0, dRdP], [HV, J, dJdPV], [0, V0^T, 0]], 2N+1 square."""
    rng = numpy.random.default_rng(1234 + seed)
    J = scipy.sparse.random(nbase, nbase, density=0.4, format="csr", random_state=int(rng.integers(1 << 30)))
    HV = scipy.sparse.random(nbase, nbase, density=0.3, format="csr", random_state=int(rng.integers(1 << 30)))
    dRdP = rng.normal(size=nbase)
    dJdPV = rng.normal(size=nbase)
    V0 = rng.normal(size=nbase)
    return J, HV, dRdP, dJdPV, V0


def _reference(J, HV, dRdP, dJdPV, V0):
    """What the trackers used to build, and what the backend has to reproduce."""
    col = lambda v: scipy.sparse.csr_matrix(numpy.asarray(v).reshape(-1, 1))
    row = lambda v: scipy.sparse.csr_matrix(numpy.asarray(v).reshape(1, -1))
    return scipy.sparse.block_array([[J, None, col(dRdP)],
                                     [HV, J, col(dJdPV)],
                                     [None, row(V0), None]]).tocsr()


# ----------------------------------------------------------------------------------------------
# RowLayout
# ----------------------------------------------------------------------------------------------

def test_a_serial_layout_owns_everything():
    L = RowLayout.serial(7)
    assert (L.n, L.first_row, L.nrow_local, L.distributed) == (7, 0, 7, False)
    assert L.local_slice == slice(0, 7)
    assert all(L.owns(i) for i in range(7))


def test_a_full_range_block_is_not_distributed():
    """Otherwise a one-rank run would take the distributed path and reach collectives alone."""
    assert RowLayout.block(5, 0, 5).distributed is False
    assert RowLayout.block(5, 0, 2).distributed is True


def test_impossible_layouts_are_refused():
    with pytest.raises(ValueError, match="does not fit"):
        RowLayout(n=5, first_row=3, nrow_local=4, distributed=True)
    with pytest.raises(ValueError, match="must own every row"):
        RowLayout(n=5, first_row=1, nrow_local=2, distributed=False)
    with pytest.raises(ValueError, match="negative"):
        RowLayout(n=-1, first_row=0, nrow_local=0, distributed=False)


def test_local_rows_of_keeps_only_this_ranks_rows():
    """A set of global equation numbers is reported identically on every rank; each must act on its
    own rows only, and the others reach theirs through their own copy of the same set."""
    L = RowLayout.block(10, 4, 3)  # owns 4,5,6
    assert L.local_rows_of([0, 4, 5, 9, 6, 6]).tolist() == [0, 1, 2]
    assert L.local_rows_of([0, 1, 9]).tolist() == []


# ----------------------------------------------------------------------------------------------
# DistVector
# ----------------------------------------------------------------------------------------------

def test_vector_reductions_match_numpy_serially():
    rng = _rng()
    a, b = rng.normal(size=9), rng.normal(size=9)
    L = RowLayout.serial(9)
    va, vb = DistVector(a, L), DistVector(b, L)
    assert va.dot(vb) == pytest.approx(float(numpy.dot(a, b)))
    assert va.norm() == pytest.approx(float(numpy.linalg.norm(a)))
    assert va.max_abs() == pytest.approx(float(numpy.max(numpy.abs(a))))
    assert va.sum() == pytest.approx(float(a.sum()))
    assert va.normalised().norm() == pytest.approx(1.0)


def test_vector_reductions_add_up_across_a_fabricated_split():
    """Each 'rank' holds a slice; the reductions are sums of the local parts by construction."""
    rng = _rng()
    a, b = rng.normal(size=12), rng.normal(size=12)
    cuts = [(0, 5), (5, 3), (8, 4)]
    dots = sum(float(numpy.dot(a[f:f + k], b[f:f + k])) for f, k in cuts)
    assert dots == pytest.approx(float(numpy.dot(a, b)))
    sq = sum(float(numpy.dot(a[f:f + k], a[f:f + k])) for f, k in cuts)
    assert numpy.sqrt(sq) == pytest.approx(float(numpy.linalg.norm(a)))
    # and the pieces really are what from_global hands each rank
    for f, k in cuts:
        L = RowLayout.block(12, f, k)
        assert DistVector.from_global(a, L).local.tolist() == a[f:f + k].tolist()


def test_a_vector_must_match_its_layout():
    with pytest.raises(ValueError, match="row block"):
        DistVector(numpy.zeros(3), RowLayout.serial(4))
    with pytest.raises(ValueError, match="expected 4 global values"):
        DistVector.from_global(numpy.zeros(3), RowLayout.serial(4))
    with pytest.raises(ValueError, match="different layouts"):
        DistVector(numpy.zeros(4), RowLayout.serial(4)).dot(DistVector(numpy.zeros(2), RowLayout.serial(2)))


def test_set_rows_writes_only_the_owned_ones():
    L = RowLayout.block(10, 4, 3)
    v = DistVector(numpy.arange(3.0) + 10.0, L)
    out = v.set_rows([0, 5, 9])  # only row 5 is ours, i.e. local index 1
    assert out.local.tolist() == [10.0, 0.0, 12.0]
    assert v.local.tolist() == [10.0, 11.0, 12.0], "set_rows must not modify in place"


# ----------------------------------------------------------------------------------------------
# DistMatrix
# ----------------------------------------------------------------------------------------------

def test_matrices_are_stored_canonically():
    """Unsorted column indices are what oomph's distributed assembly produces; PETSc's createAIJ
    wants them ascending and petsc.py's reuse digest hashes them."""
    unsorted = scipy.sparse.csr_matrix((numpy.array([1.0, 2.0, 3.0]),
                                        numpy.array([2, 0, 1]),
                                        numpy.array([0, 3])), shape=(1, 3))
    assert not unsorted.has_sorted_indices
    M = DistMatrix(unsorted, RowLayout.serial(1), 3)
    assert M.local.has_sorted_indices
    assert M.local.toarray().tolist() == [[2.0, 3.0, 1.0]]


def test_row_operations_are_local_and_match_the_helpers():
    rng = _rng()
    A = scipy.sparse.random(6, 6, density=0.5, format="csr", random_state=int(rng.integers(1 << 30))).tocsr()
    B = ScipyBackend()
    M = B.matrix(A, RowLayout.serial(6))
    from pyoomph.solvers.generic import csr_rows_to_identity, zero_csr_rows
    assert (M.rows_to_identity([1, 3]).local != csr_rows_to_identity(A, numpy.array([1, 3]))).nnz == 0
    assert (M.zero_rows([1, 3]).local != zero_csr_rows(A, numpy.array([1, 3]))).nnz == 0
    # on a block, the same rows are reached by global index
    blk = B.matrix(A[2:5, :], RowLayout.block(6, 2, 3))
    whole = csr_rows_to_identity(A, numpy.array([3]))
    assert (blk.rows_to_identity([3]).local != whole[2:5, :]).nnz == 0


def test_matvec_and_frobenius_serially():
    rng = _rng()
    A = scipy.sparse.random(6, 6, density=0.5, format="csr", random_state=int(rng.integers(1 << 30))).tocsr()
    x = rng.normal(size=6)
    B = ScipyBackend()
    L = RowLayout.serial(6)
    got = B.matrix(A, L).matvec(DistVector(x, L))
    assert numpy.allclose(got.local, A @ x)
    assert B.matrix(A, L).frobenius_norm() == pytest.approx(float(numpy.sqrt((A.data ** 2).sum())))


def test_a_distributed_transpose_is_refused_with_the_alternative_named():
    B = ScipyBackend()
    M = B.matrix(scipy.sparse.eye(3, 6, format="csr"), RowLayout.block(6, 0, 3))
    with pytest.raises(RuntimeError, match="transposed=True"):
        M.transpose()


# ----------------------------------------------------------------------------------------------
# the bordered system: block() and stack()
# ----------------------------------------------------------------------------------------------

_GROUPS = [False, False, True]  # [base | eigenvector block | scalar parameter]


def _build(backend, J, HV, dRdP, dJdPV, V0, base, augmented, table=None):
    """One 'rank's' contribution to the fold-shaped bordered matrix."""
    Jb = backend.matrix(J[base.local_slice, :], base, base.n)
    HVb = backend.matrix(HV[base.local_slice, :], base, base.n)
    return backend.block(
        [[Jb, None, backend.col(DistVector.from_global(dRdP, base))],
         [HVb, Jb, backend.col(DistVector.from_global(dJdPV, base))],
         [None, backend.row(DistVector(V0, RowLayout.serial(base.n))), None]],
        _GROUPS, base, augmented, table)


def test_block_reproduces_block_array_serially():
    nbase = 6
    J, HV, dRdP, dJdPV, V0 = _fold_pieces(nbase)
    ref = _reference(J, HV, dRdP, dJdPV, V0)
    B = ScipyBackend()
    base = RowLayout.serial(nbase)
    aug = RowLayout.serial(2 * nbase + 1)
    got = _build(B, J, HV, dRdP, dJdPV, V0, base, aug)
    assert got.local.shape == ref.shape
    diff = got.local - ref
    assert diff.nnz == 0 or abs(diff).max() == 0.0, "block() does not reproduce block_array"


def _naive_table(nbase, cuts, groups):
    """The layout AugmentedDofDistributionHelper installs, rebuilt here for a fabricated split.

    rank d owns its base rows, then its rows of each vector block, with each scalar on rank 0.
    """
    naive_start, acc = [0], 0
    for g in groups:
        naive_start.append(acc + (1 if g else nbase))
        acc = naive_start[-1]
    n_aug = naive_start[len(groups)]
    table = numpy.full(n_aug, -1, dtype=numpy.int64)
    counter = 0
    aug_blocks = []
    for d, (first, nloc) in enumerate(cuts):
        start = counter
        for g_index, is_scalar in enumerate(groups):
            if not is_scalar:
                for i in range(nloc):
                    table[naive_start[g_index] + first + i] = counter; counter += 1
            elif d == 0:
                table[naive_start[g_index]] = counter; counter += 1
        aug_blocks.append((start, counter - start))
    assert counter == n_aug and not (table < 0).any()
    return table, aug_blocks, n_aug


@pytest.mark.parametrize("cuts", [[(0, 3), (3, 3)], [(0, 2), (2, 2), (4, 2)], [(0, 5), (5, 1)]])
def test_block_over_a_fabricated_split_reassembles_to_the_serial_system(cuts):
    """The real test of the arithmetic: assemble per 'rank', reassemble, compare to the reference.

    The distributed system is the serial one with its rows and columns permuted by the equation
    table, so the comparison applies that permutation rather than hoping the two orders agree.
    """
    nbase = sum(k for _first, k in cuts)
    J, HV, dRdP, dJdPV, V0 = _fold_pieces(nbase, seed=len(cuts))
    ref = _reference(J, HV, dRdP, dJdPV, V0)
    table, aug_blocks, n_aug = _naive_table(nbase, cuts, _GROUPS)
    B = ScipyBackend()

    pieces = []
    for (first, nloc), (aug_first, aug_nloc) in zip(cuts, aug_blocks):
        base = RowLayout.block(nbase, first, nloc)
        aug = RowLayout.block(n_aug, aug_first, aug_nloc)
        pieces.append(_build(B, J, HV, dRdP, dJdPV, V0, base, aug, table))
        assert pieces[-1].local.shape == (aug_nloc, n_aug)

    got = scipy.sparse.vstack([p.local for p in pieces], format="csr").tocsr()
    # ref permuted: row/col i of the naive system becomes table[i]
    perm = numpy.asarray(table)
    expect = ref.tocoo()
    expect = scipy.sparse.coo_matrix((expect.data, (perm[expect.row], perm[expect.col])),
                                     shape=(n_aug, n_aug)).tocsr()
    diff = got - expect
    assert diff.nnz == 0 or abs(diff).max() == pytest.approx(0.0, abs=1e-15), (
        "the reassembled distributed system is not the permuted serial one (max %.3e)"
        % (abs(diff).max() if diff.nnz else 0.0))


@pytest.mark.parametrize("cuts", [[(0, 3), (3, 3)], [(0, 2), (2, 2), (4, 2)]])
def test_stack_over_a_fabricated_split_reassembles_to_the_serial_residual(cuts):
    nbase = sum(k for _first, k in cuts)
    rng = _rng()
    R, extra = rng.normal(size=nbase), rng.normal(size=nbase)
    scalar = 0.375
    ref = numpy.concatenate([R, extra, [scalar]])
    table, aug_blocks, n_aug = _naive_table(nbase, cuts, _GROUPS)
    B = ScipyBackend()

    out = numpy.zeros(n_aug)
    for (first, nloc), (aug_first, aug_nloc) in zip(cuts, aug_blocks):
        base = RowLayout.block(nbase, first, nloc)
        aug = RowLayout.block(n_aug, aug_first, aug_nloc)
        part = B.stack([DistVector.from_global(R, base), DistVector.from_global(extra, base), scalar],
                       _GROUPS, base, aug, table)
        assert len(part) == aug_nloc
        out[aug_first:aug_first + aug_nloc] = part.local
    # undo the permutation and compare
    inverse = numpy.empty(n_aug, dtype=numpy.int64)
    inverse[numpy.asarray(table)] = numpy.arange(n_aug)
    assert numpy.allclose(out[numpy.asarray(table)], ref)


def test_block_refuses_a_grid_that_does_not_match_the_groups():
    B = ScipyBackend()
    base, aug = RowLayout.serial(3), RowLayout.serial(7)
    with pytest.raises(ValueError, match="grid"):
        B.block([[None, None]], _GROUPS, base, aug)
    with pytest.raises(ValueError, match="group 0 is the base block"):
        B.block([[None]], [True], base, RowLayout.serial(1))


def test_block_refuses_a_group_layout_that_contradicts_the_dof_layout():
    """2N+1 is what [base | block | scalar] describes; anything else is a bookkeeping mistake."""
    B = ScipyBackend()
    with pytest.raises(ValueError, match="describes 7 augmented rows but the dof layout has 8"):
        B.block([[None] * 3 for _ in range(3)], _GROUPS, RowLayout.serial(3), RowLayout.serial(8))


def test_a_border_row_must_be_replicated():
    B = ScipyBackend()
    blocked = DistVector(numpy.zeros(2), RowLayout.block(6, 0, 2))
    with pytest.raises(ValueError, match="needs the whole vector"):
        B.row(blocked)


def test_the_backend_choice_is_scipy_on_one_process():
    assert isinstance(get_la_backend(None), ScipyBackend)
