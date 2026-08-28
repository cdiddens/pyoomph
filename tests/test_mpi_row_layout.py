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

# The offset arithmetic behind the gather-to-root linear solve, tested WITHOUT mpirun.
#
# A wrong displacement here does not crash: it produces a perfectly well-formed CSR matrix that
# describes a different system, and the Newton solve then simply fails to converge somewhere far
# away from the cause. That is why mpi_row_layout_from_gathered() is split out from the collective
# that feeds it -- so the part that can be wrong can be checked against fabricated inputs.
#
# The cases that matter are the ones oomph actually produces: a non-divisible split, and (under
# --distribute, or with fewer dofs than ranks) blocks that are empty or not in rank order.

import pytest

from pyoomph.generic.mpi import mpi_row_layout_from_gathered


def _layout(entries, n):
    """entries: [(first_row, nrow_local, nnz_local), ...] indexed by rank."""
    return mpi_row_layout_from_gathered(entries, n)


def test_uniform_split():
    lay = _layout([(0, 5, 20), (5, 5, 30)], 10)
    assert list(lay.vec_counts) == [5, 5]
    assert list(lay.vec_displs) == [0, 5]
    assert list(lay.nnz_counts) == [20, 30]
    assert list(lay.nnz_displs) == [0, 20]
    assert lay.nnz_total == 50


def test_non_divisible_split():
    """7 rows over 3 ranks -- oomph gives the remainder to the low ranks, so the blocks differ."""
    lay = _layout([(0, 3, 9), (3, 2, 4), (5, 2, 5)], 7)
    assert list(lay.vec_counts) == [3, 2, 2]
    assert list(lay.vec_displs) == [0, 3, 5]
    assert list(lay.nnz_displs) == [0, 9, 13]
    assert lay.nnz_total == 18


def test_ranks_owning_no_rows():
    """Fewer dofs than ranks: the empty ranks share a first_row with their successor."""
    lay = _layout([(0, 1, 1), (1, 1, 2), (2, 0, 0), (2, 0, 0)], 2)
    assert list(lay.vec_counts) == [1, 1, 0, 0]
    assert list(lay.vec_displs) == [0, 1, 2, 2]
    assert lay.nnz_total == 3
    # An empty rank contributes nothing, so it cannot displace anyone else's nonzeros.
    assert list(lay.nnz_displs)[:2] == [0, 1]


def test_blocks_not_in_rank_order():
    """Ordering must follow first_row, not the rank index.

    oomph's own uniform split happens to be ascending in rank, but a distributed problem's dof
    distribution carries no such promise, and ordering by rank would then interleave the rows of two
    ranks into one plausible-looking, wrong matrix.
    """
    lay = _layout([(6, 3, 7), (0, 4, 11), (4, 2, 5)], 9)
    assert list(lay.vec_displs) == [6, 0, 4]
    # Nonzeros are laid out in row order: rank 1 (rows 0-3) first, then rank 2, then rank 0.
    assert list(lay.nnz_displs) == [16, 0, 11]
    assert lay.nnz_total == 23


def test_gap_between_blocks_is_rejected():
    with pytest.raises(RuntimeError, match="do not tile"):
        _layout([(0, 3, 3), (4, 3, 3)], 7)


def test_overlapping_blocks_are_rejected():
    with pytest.raises(RuntimeError, match="do not tile"):
        _layout([(0, 4, 4), (3, 3, 3)], 7)


def test_row_count_mismatch_is_rejected():
    with pytest.raises(RuntimeError, match="rows, but the system has"):
        _layout([(0, 3, 3), (3, 3, 3)], 10)


def test_nnz_over_int32_is_rejected():
    """The gathered row_start array is int32; over the limit the offsets wrap silently negative."""
    with pytest.raises(RuntimeError, match="int32"):
        _layout([(0, 2, 2 ** 30), (2, 2, 2 ** 30 + 5)], 4)


def test_single_rank_holding_everything():
    lay = _layout([(0, 6, 17)], 6)
    assert list(lay.vec_counts) == [6] and list(lay.nnz_displs) == [0]
    assert lay.nnz_total == 17
