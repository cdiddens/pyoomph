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

"""Row-distributed vectors and matrices, and the operations an augmented system needs on them.

This replaces the direct use of ``scipy.sparse`` in the bordered-system algebra of
:py:mod:`pyoomph.generic.bifurcation_tools`. The point is not an abstraction for its own sake: every
object here carries a **row layout**, so one piece of tracker code is correct serially, under plain
``mpirun`` and under ``--distribute``, instead of three.

The degenerate case is the whole design. A :py:class:`RowLayout` with ``distributed=False`` says
"this rank owns every row", so on one process a :py:class:`DistVector` is a numpy array with an
offset of zero, ``dot`` is ``numpy.dot``, and nothing is communicated. Code written against this API
does not ask which regime it is in.

What it does NOT hide is cost. An operation that has to talk to other ranks is named so that the
reader can see it: :py:meth:`DistVector.to_global` and :py:meth:`DistMatrix.to_global_square`
replicate, :py:meth:`DistVector.global_value` is a broadcast, and the reductions are collective by
definition. Every collective here must be reached by every rank, so a branch around one has to be
decided on replicated data -- see ``dev_docs/replicated_mpi_correctness.md`` on why a branch taken on
a local length is a deadlock rather than a wrong answer.

Two backends. :py:class:`ScipyBackend` keeps everything on one process and is what runs serially;
:py:class:`PETScBackend` is used whenever ``nproc > 1``, and differs only where it has to -- the
matrix-vector product, the transpose, and the solve. The solve itself is not new code: it goes
through ``GenericLinearSystemSolver.solve_python_built_distributed``, which already takes exactly
this layout (a local CSR row block with global column indices) and which PETSc already implements
with a real MPIAIJ matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Union

import numpy
import scipy.sparse  # type:ignore

from ..typings import NPFloatArray


def _nproc() -> int:
    """Number of MPI ranks, 1 if there is no MPI.

    Imported lazily: importing ``pyoomph.generic.mpi`` initialises MPI, and this module is imported
    by things that may never need it. Note ``get_mpi_nproc()`` returns **0** without MPI, hence the
    ``max``.
    """
    from .mpi import get_mpi_nproc
    return max(int(get_mpi_nproc()), 1)


# ----------------------------------------------------------------------------------------------
# the layout
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RowLayout:
    """Who owns which rows of a row-partitioned vector or matrix.

    Immutable, and every field is either a global property or derived from allgathered data, which
    is what makes it safe to branch on. ``distributed=False`` means this rank owns all ``n`` rows.
    """

    n: int
    first_row: int
    nrow_local: int
    distributed: bool

    def __post_init__(self):
        if self.n < 0 or self.nrow_local < 0 or self.first_row < 0:
            raise ValueError("a row layout cannot have negative sizes: " + repr(self))
        if self.first_row + self.nrow_local > self.n:
            raise ValueError("the row block [%d,%d) does not fit in %d rows"
                             % (self.first_row, self.first_row + self.nrow_local, self.n))
        if not self.distributed and (self.first_row != 0 or self.nrow_local != self.n):
            raise ValueError("a non-distributed layout must own every row, got " + repr(self))

    # -- constructors ---------------------------------------------------------------------------

    @classmethod
    def serial(cls, n: int) -> "RowLayout":
        """One owner for everything. What serial code and a replicated ``mpirun`` both see."""
        return cls(n=int(n), first_row=0, nrow_local=int(n), distributed=False)

    @classmethod
    def block(cls, n: int, first_row: int, nrow_local: int) -> "RowLayout":
        """An explicit row block. Distributed unless it happens to be the whole range."""
        n, first_row, nrow_local = int(n), int(first_row), int(nrow_local)
        return cls(n=n, first_row=first_row, nrow_local=nrow_local,
                   distributed=not (first_row == 0 and nrow_local == n))

    @classmethod
    def base(cls, problem: Any) -> "RowLayout":
        """The problem's BASE (unaugmented) dof layout.

        This is the layout the multi-assembly returns its blocks on, and it is deliberately a
        different constructor from :py:meth:`augmented`: confusing the two is the same class of
        mistake as B2 in ``dev_docs/mpi_augmented_systems.md``, so the call site has to say which it
        means rather than passing a tuple around.
        """
        n, nrow_local, first_row, distributed = problem._get_base_dof_distribution_info()
        return cls(n=int(n), first_row=int(first_row), nrow_local=int(nrow_local), distributed=bool(distributed))

    @classmethod
    def augmented(cls, problem: Any) -> "RowLayout":
        """The problem's current dof layout, i.e. the augmented one while a tracker is installed."""
        n, nrow_local, first_row, distributed = problem._get_dof_distribution_info()
        return cls(n=int(n), first_row=int(first_row), nrow_local=int(nrow_local), distributed=bool(distributed))

    # -- queries --------------------------------------------------------------------------------

    @property
    def local_slice(self) -> slice:
        return slice(self.first_row, self.first_row + self.nrow_local)

    def owns(self, global_row: int) -> bool:
        return self.first_row <= int(global_row) < self.first_row + self.nrow_local

    def local_rows_of(self, global_rows: Iterable[int]) -> "numpy.ndarray":
        """The subset of ``global_rows`` this rank owns, as LOCAL indices, sorted.

        The filtering is the point, not an optimisation: a set of global equation numbers (the dofs a
        normal mode pins, say) is reported identically on every rank, and each rank must act on its
        own rows only -- the others reach theirs through their own copy of the same set.
        """
        g = numpy.asarray(list(global_rows), dtype=numpy.int64) if not isinstance(global_rows, numpy.ndarray) \
            else numpy.asarray(global_rows, dtype=numpy.int64)
        g = g[(g >= self.first_row) & (g < self.first_row + self.nrow_local)]
        return numpy.unique(g - self.first_row)

    def validate(self) -> "RowLayout":
        """Collective: prove the ranks' blocks tile ``[0,n)``. Returns self, for chaining.

        Reuses :py:func:`pyoomph.generic.mpi.mpi_row_layout_from_gathered`, which already orders the
        blocks by ``first_row`` rather than by rank and refuses a gap, an overlap or an nnz total
        that would overflow an int32 row-start array.
        """
        if _nproc() <= 1 or not self.distributed:
            return self
        from .mpi import mpi_row_layout
        mpi_row_layout(self.n, self.first_row, self.nrow_local, 0)
        return self


# ----------------------------------------------------------------------------------------------
# vectors
# ----------------------------------------------------------------------------------------------

class DistVector:
    """This rank's rows of a vector, plus the layout saying which they are."""

    __slots__ = ("local", "layout")

    def __init__(self, local: Any, layout: RowLayout):
        arr = numpy.asarray(local)
        if arr.ndim != 1:
            raise ValueError("a DistVector is one-dimensional, got shape " + str(arr.shape))
        if len(arr) != layout.nrow_local:
            raise ValueError("got %d values for a row block of %d (layout %r)"
                             % (len(arr), layout.nrow_local, layout))
        self.local = arr
        self.layout = layout

    # -- construction ---------------------------------------------------------------------------

    @classmethod
    def from_global(cls, values: Any, layout: RowLayout) -> "DistVector":
        """Keep this rank's slice of a globally indexed array.

        The array has to be the global one on every rank; that is how a guess arrives from Python
        (``dofs.add_vector(...)``) and how an eigenvector comes back from the eigensolver.
        """
        arr = numpy.asarray(values)
        if len(arr) != layout.n:
            raise ValueError("expected %d global values, got %d" % (layout.n, len(arr)))
        return cls(numpy.ascontiguousarray(arr[layout.local_slice]), layout)

    @classmethod
    def zeros(cls, layout: RowLayout, dtype: Any = numpy.float64) -> "DistVector":
        return cls(numpy.zeros(layout.nrow_local, dtype=dtype), layout)

    def like(self, local: Any) -> "DistVector":
        """A new vector on this one's layout."""
        return DistVector(local, self.layout)

    # -- reductions (collective when distributed) ------------------------------------------------

    def _check(self, other: "DistVector", what: str):
        if other.layout != self.layout:
            raise ValueError("cannot %s two vectors on different layouts: %r and %r"
                             % (what, self.layout, other.layout))

    def _sum(self, value: float) -> float:
        if not self.layout.distributed or _nproc() <= 1:
            return float(value)
        from .mpi import get_mpi_sum
        return float(get_mpi_sum(float(value)))

    def _max(self, value: float) -> float:
        if not self.layout.distributed or _nproc() <= 1:
            return float(value)
        from .mpi import get_mpi_max
        return float(get_mpi_max(float(value)))

    def dot(self, other: "DistVector") -> float:
        self._check(other, "dot")
        return self._sum(float(numpy.dot(self.local, other.local)))

    def vdot(self, other: "DistVector") -> complex:
        """Conjugating inner product. Summed as one complex value, so the ranks cannot disagree
        about the real and imaginary parts separately."""
        self._check(other, "vdot")
        local = complex(numpy.vdot(self.local, other.local))
        if not self.layout.distributed or _nproc() <= 1:
            return local
        from .mpi import get_mpi_sum
        return complex(get_mpi_sum(local))

    def norm(self) -> float:
        """Euclidean norm. Reduced as the sum of squares, not as a max of local norms."""
        return float(numpy.sqrt(self._sum(float(numpy.vdot(self.local, self.local).real))))

    def max_abs(self) -> float:
        local = float(numpy.max(numpy.abs(self.local))) if len(self.local) else 0.0
        return self._max(local)

    def sum(self) -> float:
        return self._sum(float(numpy.sum(self.local)))

    # -- elementwise (local, no communication) ---------------------------------------------------

    def __add__(self, other: "DistVector") -> "DistVector":
        self._check(other, "add")
        return self.like(self.local + other.local)

    def __sub__(self, other: "DistVector") -> "DistVector":
        self._check(other, "subtract")
        return self.like(self.local - other.local)

    def __mul__(self, scalar: Union[float, complex]) -> "DistVector":
        return self.like(self.local * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: Union[float, complex]) -> "DistVector":
        return self.like(self.local / scalar)

    def __neg__(self) -> "DistVector":
        return self.like(-self.local)

    def __len__(self) -> int:
        return len(self.local)

    def copy(self) -> "DistVector":
        return self.like(self.local.copy())

    def real(self) -> "DistVector":
        return self.like(numpy.real(self.local))

    def imag(self) -> "DistVector":
        return self.like(numpy.imag(self.local))

    def astype(self, dtype: Any) -> "DistVector":
        return self.like(self.local.astype(dtype))

    def normalised(self) -> "DistVector":
        """A copy scaled to unit norm. Collective, because the norm is."""
        nrm = self.norm()
        if nrm == 0.0:
            raise ValueError("cannot normalise a vector of norm zero")
        return self / nrm

    def set_rows(self, global_rows: Iterable[int], value: float = 0.0) -> "DistVector":
        """Write ``value`` into the given GLOBAL rows this rank owns. Returns a copy.

        Replaces the ``r[list_of_global_equations] = 0.0`` idiom, which is only correct when one rank
        holds everything.
        """
        out = self.local.copy()
        out[self.layout.local_rows_of(global_rows)] = value
        return self.like(out)

    # -- the named escapes -----------------------------------------------------------------------

    def to_global(self) -> NPFloatArray:
        """The whole vector, on every rank. O(n) memory per rank and a collective; named so that
        both are visible at the call site."""
        if not self.layout.distributed or _nproc() <= 1:
            return self.local
        from .mpi import mpi_allgather_vector
        return mpi_allgather_vector(self.layout.n, self.layout.first_row, self.layout.nrow_local,
                                    self.local, context="replicating a DistVector")

    def global_value(self, global_row: int) -> float:
        """One entry by global index, on every rank. A broadcast; collective."""
        g = int(global_row)
        if not self.layout.distributed or _nproc() <= 1:
            return float(self.local[g])
        from .mpi import get_mpi_sum
        mine = float(self.local[g - self.layout.first_row]) if self.layout.owns(g) else 0.0
        # Summed rather than broadcast from a computed owner: exactly one rank owns the row, so the
        # sum IS the value, and it needs no agreement about who that is.
        return float(get_mpi_sum(mine))

    def __repr__(self) -> str:
        return "DistVector(%r, %r)" % (self.local, self.layout)


# ----------------------------------------------------------------------------------------------
# matrices
# ----------------------------------------------------------------------------------------------

class DistMatrix:
    """This rank's rows of a sparse matrix: a ``(nrow_local, ncol)`` CSR with GLOBAL column indices.

    That is the layout oomph's distributed assembly produces, the one PETSc's MPIAIJ wants, and the
    one ``solve_python_built_distributed`` is documented against, so nothing is converted at either
    boundary.
    """

    __slots__ = ("local", "layout", "ncol")

    def __init__(self, local: Any, layout: RowLayout, ncol: Optional[int] = None):
        mat = local.tocsr() if hasattr(local, "tocsr") else scipy.sparse.csr_matrix(local)
        if mat.shape[0] != layout.nrow_local:
            raise ValueError("got %d rows for a row block of %d (layout %r)"
                             % (mat.shape[0], layout.nrow_local, layout))
        self.ncol = int(mat.shape[1]) if ncol is None else int(ncol)
        if mat.shape[1] != self.ncol:
            raise ValueError("matrix has %d columns, expected %d" % (mat.shape[1], self.ncol))
        # Canonical form, always: PETSc's createAIJ wants ascending column indices per row, and
        # petsc.py's structure-reuse digest hashes the index arrays, so an order that varied between
        # two assemblies of one pattern would defeat the reuse it exists to enable.
        if not mat.has_sorted_indices:
            mat = mat.copy()
            mat.sort_indices()
        self.local = mat
        self.layout = layout

    @property
    def shape(self):
        """The LOCAL shape. ``(layout.n, ncol)`` is the global one."""
        return self.local.shape

    @property
    def nnz(self) -> int:
        return int(self.local.nnz)

    def like(self, local: Any) -> "DistMatrix":
        return DistMatrix(local, self.layout, self.ncol)

    # -- local algebra --------------------------------------------------------------------------

    def __add__(self, other: "DistMatrix") -> "DistMatrix":
        if other.layout != self.layout or other.ncol != self.ncol:
            raise ValueError("cannot add matrices on different layouts")
        return self.like(self.local + other.local)

    def __mul__(self, scalar: Union[float, complex]) -> "DistMatrix":
        return self.like(self.local * scalar)

    __rmul__ = __mul__

    def scale_rows(self, factors: Any) -> "DistMatrix":
        """Multiply each owned row by its factor (a local array or a DistVector). Row-local."""
        f = factors.local if isinstance(factors, DistVector) else numpy.asarray(factors)
        if len(f) != self.layout.nrow_local:
            raise ValueError("expected %d row factors, got %d" % (self.layout.nrow_local, len(f)))
        return self.like(scipy.sparse.diags(f, 0, shape=(len(f), len(f)), format="csr") @ self.local)

    def zero_rows(self, global_rows: Iterable[int]) -> "DistMatrix":
        """Zero the given GLOBAL rows, pruning the explicit zeros. Row-local."""
        from ..solvers.generic import zero_csr_rows
        return self.like(zero_csr_rows(self.local, numpy.asarray(list(global_rows), dtype=numpy.int64),
                                       first_row=self.layout.first_row))

    def rows_to_identity(self, global_rows: Iterable[int]) -> "DistMatrix":
        """Zero the given GLOBAL rows and put 1 on their diagonal. Row-local."""
        from ..solvers.generic import csr_rows_to_identity
        return self.like(csr_rows_to_identity(self.local, numpy.asarray(list(global_rows), dtype=numpy.int64),
                                              first_row=self.layout.first_row))

    def frobenius_norm(self) -> float:
        local = float((numpy.abs(self.local.data) ** 2).sum())
        if not self.layout.distributed or _nproc() <= 1:
            return float(numpy.sqrt(local))
        from .mpi import get_mpi_sum
        return float(numpy.sqrt(float(get_mpi_sum(local))))

    # -- products (collective when distributed; the backend decides how) --------------------------

    def matvec(self, v: DistVector) -> DistVector:
        """``A @ v``. Needs entries of ``v`` this rank does not own, so collective when distributed."""
        raise NotImplementedError("use a backend's matrix, not DistMatrix directly")

    def transpose(self) -> "DistMatrix":
        raise NotImplementedError("use a backend's matrix, not DistMatrix directly")

    def to_global_square(self) -> Any:
        """The whole matrix as a scipy CSR, on every rank. O(nnz) per rank and a collective."""
        if not self.layout.distributed or _nproc() <= 1:
            return self.local
        from .mpi import mpi_allgather_square_csr
        if self.ncol != self.layout.n:
            raise ValueError("to_global_square() is for square systems; this one is %dx%d"
                             % (self.layout.n, self.ncol))
        return mpi_allgather_square_csr(self.layout.n, self.layout.first_row, self.layout.nrow_local,
                                        self.local, context="replicating a DistMatrix").tocsr()

    def __repr__(self) -> str:
        return "DistMatrix(%dx%d local of %dx%d, nnz=%d)" % (
            self.local.shape[0], self.local.shape[1], self.layout.n, self.ncol, self.nnz)


class _Border:
    """A border row or column, held until :py:meth:`LinearAlgebraBackend.block` places it.

    Kept as its own type rather than as a 1-row/1-column matrix because where it LIVES is not known
    until the block layout is: a border row occupies one global row, owned by one rank, and a border
    column spans every row. Deciding that at construction time is what makes a bordered system come
    out subtly wrong under MPI.
    """

    __slots__ = ("vector", "is_row")

    def __init__(self, vector: DistVector, is_row: bool):
        self.vector = vector
        self.is_row = is_row

    def __repr__(self) -> str:
        return "_Border(%s, %r)" % ("row" if self.is_row else "column", self.vector.layout)


# ----------------------------------------------------------------------------------------------
# the backend
# ----------------------------------------------------------------------------------------------

class LinearAlgebraBackend:
    """Builds and solves row-distributed systems. Subclassed per backend; most of it is shared.

    Only three operations genuinely differ between the scipy and PETSc implementations -- the
    matrix-vector product on a distributed layout, the transpose, and the solve. Everything else,
    including the bordered-system assembly in :py:meth:`block`, is layout arithmetic and lives here.
    """

    name = "generic"

    def __init__(self, problem: Any = None):
        self.problem = problem

    # -- plain construction ----------------------------------------------------------------------

    def vector(self, local: Any, layout: RowLayout) -> DistVector:
        return DistVector(local, layout)

    def matrix(self, local: Any, layout: RowLayout, ncol: Optional[int] = None) -> DistMatrix:
        return self._wrap(DistMatrix(local, layout, ncol))

    def zeros(self, layout: RowLayout, ncol: Optional[int] = None) -> DistMatrix:
        ncol = layout.n if ncol is None else int(ncol)
        return self.matrix(scipy.sparse.csr_matrix((layout.nrow_local, ncol)), layout, ncol)

    def diag(self, values: Any, layout: RowLayout) -> DistMatrix:
        """A square diagonal matrix from this rank's rows of the diagonal."""
        d = values.local if isinstance(values, DistVector) else numpy.asarray(values)
        if len(d) != layout.nrow_local:
            raise ValueError("expected %d diagonal values, got %d" % (layout.nrow_local, len(d)))
        rows = numpy.arange(len(d), dtype=numpy.int64)
        cols = rows + layout.first_row
        keep = d != 0.0
        mat = scipy.sparse.csr_matrix((d[keep], (rows[keep], cols[keep])), shape=(layout.nrow_local, layout.n))
        return self.matrix(mat, layout, layout.n)

    def _wrap(self, m: DistMatrix) -> DistMatrix:
        """Re-type a DistMatrix as this backend's, so matvec/transpose/solve resolve here."""
        m.__class__ = self._matrix_class()
        return m

    def _matrix_class(self):
        return DistMatrix

    # -- borders ---------------------------------------------------------------------------------

    def col(self, v: DistVector) -> _Border:
        """A border COLUMN: one global column, an entry in every row."""
        return _Border(v, is_row=False)

    def row(self, v: DistVector) -> _Border:
        """A border ROW: one global row, an entry in every column of its group.

        WHICH rank keeps it is not stated here and is not a parameter: it follows from the augmented
        layout passed to :py:meth:`block`, which is the only thing that knows where that row lives.
        AugmentedDofDistributionHelper puts every scalar unknown on rank 0 (src/bifurcation.hpp), so
        in practice that is where it lands, and the dense row of ``n`` entries it costs there is the
        cost the C++ trackers already pay -- see dev_docs/mpi_augmented_systems.md section 5.

        ``v`` must be on a NON-distributed layout, i.e. the whole vector. One rank keeps this row, so
        it needs every entry, and requiring the caller to hand over a replicated vector keeps the
        collective out of the assembly: it happens where the vector is made, once, instead of inside
        every block() call. It is also what the C++ handlers do -- their fixed normalisation and
        symmetry vectors stay fully replicated on purpose, because they are read-only after
        construction and that removes any obligation to synchronise them.
        """
        if v.layout.distributed:
            raise ValueError(
                "a border row needs the whole vector, but it was given a row block (%r). One rank "
                "keeps the row, so replicate it first -- DistVector.from_global(v.to_global(), "
                "RowLayout.serial(n)) -- or keep it replicated from the start, which is what the C++ "
                "trackers do with their normalisation vectors." % (v.layout,))
        return _Border(v, is_row=True)

    # -- the bordered system ---------------------------------------------------------------------

    def block(self, rows: Sequence[Sequence[Any]], group_is_scalar: Sequence[bool],
              base: RowLayout, augmented: RowLayout, eqn_table: Optional[Any] = None) -> DistMatrix:
        """Assemble a bordered matrix from blocks given in the NAIVE group order.

        ``rows`` is the list-of-lists a tracker writes, with ``None`` for a structurally absent
        block, a :py:class:`DistMatrix` for a base-sized one, a :py:class:`_Border` from
        :py:meth:`col`/:py:meth:`row`, and a plain number for a scalar-scalar entry.
        ``group_is_scalar`` says, per group, whether it is a single unknown or a base-sized vector --
        group 0 is the base block and is never scalar.

        ``eqn_table`` is ``Problem._get_augmented_eqn_table()``: the naive -> real translation, empty
        (or None) when the layout is not distributed, which is read as the identity. Python must not
        recompute it -- two tables that disagree is a silently wrong matrix.
        """
        n_groups = len(group_is_scalar)
        if any(len(r) != n_groups for r in rows) or len(rows) != n_groups:
            raise ValueError("block() wants a %d x %d grid, got %s"
                             % (n_groups, n_groups, [len(r) for r in rows]))
        if group_is_scalar[0]:
            raise ValueError("group 0 is the base block and cannot be a scalar")

        nbase = base.n
        # Naive start of each group, in the historical [base | block | ... | scalar | ...] numbering.
        naive_start = [0]
        for g in range(n_groups):
            naive_start.append(naive_start[g] + (1 if group_is_scalar[g] else nbase))
        n_aug_naive = naive_start[n_groups]
        if n_aug_naive != augmented.n:
            raise ValueError("the group layout describes %d augmented rows but the dof layout has %d"
                             % (n_aug_naive, augmented.n))

        table = None
        if eqn_table is not None and len(eqn_table) > 0:
            table = numpy.asarray(eqn_table, dtype=numpy.int64)
            if len(table) != n_aug_naive:
                raise ValueError("the equation table has %d entries for %d augmented rows"
                                 % (len(table), n_aug_naive))

        def to_real(naive: Any):
            return naive if table is None else table[naive]

        # Who owns a scalar group's single row is asked of the AUGMENTED LAYOUT, never of the process
        # rank. The layout is the only thing that knows -- it is what the dof distribution installed --
        # and taking it from get_mpi_rank() instead would both duplicate that knowledge and make the
        # arithmetic untestable without mpirun.
        def i_own_scalar(group: int) -> bool:
            return augmented.owns(int(to_real(naive_start[group])))

        data_list, row_list, col_list = [], [], []

        def emit(r_naive: Any, c_naive: Any, values: Any):
            if len(numpy.atleast_1d(values)) == 0:
                return
            row_list.append(numpy.asarray(to_real(r_naive), dtype=numpy.int64).ravel())
            col_list.append(numpy.asarray(to_real(c_naive), dtype=numpy.int64).ravel())
            data_list.append(numpy.asarray(values, dtype=numpy.float64).ravel())

        for gr in range(n_groups):
            for gc in range(n_groups):
                cell = rows[gr][gc]
                if cell is None:
                    continue
                if isinstance(cell, DistMatrix):
                    if group_is_scalar[gr] or group_is_scalar[gc]:
                        raise ValueError("block (%d,%d) is a matrix but one of its groups is a scalar" % (gr, gc))
                    coo = cell.local.tocoo()
                    # local row i  ->  base global row base.first_row+i  ->  naive  ->  real
                    emit(naive_start[gr] + base.first_row + coo.row.astype(numpy.int64),
                         naive_start[gc] + coo.col.astype(numpy.int64), coo.data)
                elif isinstance(cell, _Border):
                    if cell.is_row:
                        # One global row on cell.owner, spanning a whole base-sized group of columns.
                        # The owner needs the entire vector, so this replicates it -- deliberately, and
                        # only on the rank that keeps the row.
                        if group_is_scalar[gc]:
                            raise ValueError("a border row cannot span a scalar group (%d,%d)" % (gr, gc))
                        if not group_is_scalar[gr]:
                            raise ValueError("a border row must sit in a scalar group, not (%d,%d)" % (gr, gc))
                        full = cell.vector.local  # replicated by contract; see row()
                        if i_own_scalar(gr):
                            cols = numpy.arange(nbase, dtype=numpy.int64)
                            keep = full != 0.0
                            emit(numpy.full(int(keep.sum()), naive_start[gr], dtype=numpy.int64),
                                 naive_start[gc] + cols[keep], full[keep])
                    else:
                        # One global column, an entry in each of this rank's rows of group gr.
                        if group_is_scalar[gc] is False:
                            raise ValueError("a border column must sit in a scalar group, not (%d,%d)" % (gr, gc))
                        if group_is_scalar[gr]:
                            raise ValueError("a border column cannot span a scalar group (%d,%d)" % (gr, gc))
                        vals = cell.vector.local
                        rws = naive_start[gr] + base.first_row + numpy.arange(len(vals), dtype=numpy.int64)
                        keep = vals != 0.0
                        emit(rws[keep], numpy.full(int(keep.sum()), naive_start[gc], dtype=numpy.int64), vals[keep])
                else:
                    # A scalar entry, which only exists where both groups are scalars.
                    if not (group_is_scalar[gr] and group_is_scalar[gc]):
                        raise ValueError("a plain number at (%d,%d) needs both groups to be scalars" % (gr, gc))
                    if i_own_scalar(gr) and float(cell) != 0.0:
                        emit(numpy.array([naive_start[gr]], dtype=numpy.int64),
                             numpy.array([naive_start[gc]], dtype=numpy.int64),
                             numpy.array([float(cell)]))

        if data_list:
            data = numpy.concatenate(data_list)
            rws = numpy.concatenate(row_list) - augmented.first_row
            cls = numpy.concatenate(col_list)
        else:
            data = numpy.zeros(0); rws = numpy.zeros(0, dtype=numpy.int64); cls = numpy.zeros(0, dtype=numpy.int64)
        if len(rws) and (rws.min() < 0 or rws.max() >= augmented.nrow_local):
            raise ValueError("block() produced a row outside this rank's augmented block [0,%d): %d..%d"
                             % (augmented.nrow_local, int(rws.min()), int(rws.max())))
        mat = scipy.sparse.coo_matrix((data, (rws, cls)),
                                      shape=(augmented.nrow_local, augmented.n)).tocsr()
        mat.sum_duplicates()
        return self.matrix(mat, augmented, augmented.n)

    def stack(self, parts: Sequence[Any], group_is_scalar: Sequence[bool],
              base: RowLayout, augmented: RowLayout, eqn_table: Optional[Any] = None) -> DistVector:
        """The residual counterpart of :py:meth:`block`: one entry per group, in naive order."""
        n_groups = len(group_is_scalar)
        if len(parts) != n_groups:
            raise ValueError("stack() wants %d parts, got %d" % (n_groups, len(parts)))
        naive_start = [0]
        for g in range(n_groups):
            naive_start.append(naive_start[g] + (1 if group_is_scalar[g] else base.n))
        table = None
        if eqn_table is not None and len(eqn_table) > 0:
            table = numpy.asarray(eqn_table, dtype=numpy.int64)
        def real_of(naive_index):
            return naive_index if table is None else table[naive_index]

        out = numpy.zeros(augmented.nrow_local, dtype=numpy.float64)
        for g in range(n_groups):
            part = parts[g]
            if group_is_scalar[g]:
                # Owned by whichever rank the augmented layout puts this row on; see block().
                if not augmented.owns(int(real_of(naive_start[g]))):
                    continue
                naive = numpy.array([naive_start[g]], dtype=numpy.int64)
                vals = numpy.array([float(part.local[0] if isinstance(part, DistVector) else part)])
            else:
                vec = part.local if isinstance(part, DistVector) else numpy.asarray(part)
                if len(vec) != base.nrow_local:
                    raise ValueError("group %d has %d values, expected %d" % (g, len(vec), base.nrow_local))
                naive = naive_start[g] + base.first_row + numpy.arange(base.nrow_local, dtype=numpy.int64)
                vals = numpy.asarray(vec, dtype=numpy.float64)
            real = real_of(naive)
            out[numpy.asarray(real, dtype=numpy.int64) - augmented.first_row] = vals
        return DistVector(out, augmented)

    # -- solving ---------------------------------------------------------------------------------

    def solve(self, A: DistMatrix, b: DistVector) -> DistVector:
        """Solve ``A x = b`` for this rank's block of x. Collective.

        Goes through ``GenericLinearSystemSolver.solve_python_built_distributed``, whose contract is
        already this layout: a local CSR row block with global column indices. PETSc implements it
        with a real MPIAIJ matrix and its own KSP; the other backends get a replicating fallback, so
        pardiso/scipy/accelerate keep working under mpirun without this module knowing about them.
        """
        if A.layout != b.layout:
            raise ValueError("the matrix and the right-hand side are on different layouts")
        if A.ncol != A.layout.n:
            raise ValueError("cannot solve a non-square system (%d x %d)" % (A.layout.n, A.ncol))
        if self.problem is None:
            raise RuntimeError("this backend has no problem, so it has no linear solver to solve with")
        la = self.problem.get_la_solver()
        n = A.layout.n
        nproc = _nproc()

        if A.layout.distributed or nproc <= 1:
            out = la.solve_python_built_distributed(n, A.layout.nrow_local, A.layout.first_row,
                                                   A.local, b.local)
            return DistVector(numpy.asarray(out, dtype=numpy.float64), A.layout)

        # Replicated under mpirun: every rank holds the whole system, so the blocks the solver is
        # given have to be IMPOSED rather than taken from the layout. Handing it (nrow_local=n,
        # first_row=0) on every rank would make PETSc read nproc*n global rows -- the row counts are
        # supposed to tile, and n on each of nproc ranks does not -- and the solve then hangs.
        #
        # This is what PeriodicDrivingResponse already does with its own replicated bordered system
        # (pyoomph/utils/periodic_driving_response.py), and the reason it was worth reading before
        # writing this: impose a contiguous split, contribute only that slice, and replicate the
        # answer afterwards so the caller still sees a vector on its own layout.
        from .mpi import get_mpi_rank, mpi_allgather_vector
        rank = int(get_mpi_rank())
        base, rem = divmod(n, nproc)
        nrow_local = base + (1 if rank < rem else 0)
        first_row = rank * base + min(rank, rem)
        local = la.solve_python_built_distributed(n, nrow_local, first_row,
                                                  A.local[first_row:first_row + nrow_local, :].tocsr(),
                                                  numpy.ascontiguousarray(b.local[first_row:first_row + nrow_local]))
        full = mpi_allgather_vector(n, first_row, nrow_local, numpy.asarray(local, dtype=numpy.float64),
                                    context="replicating the solution of a Python-built system")
        return DistVector(numpy.asarray(full, dtype=numpy.float64), A.layout)


    def solve_many(self, A: DistMatrix, bs: Sequence[DistVector]) -> "list[DistVector]":
        """Solve ``A x = b`` for several right-hand sides against ONE factorisation of A. Collective.

        Every rank must pass the same number of right-hand sides in the same order, because the
        collectives below are per right-hand side. Returns one DistVector per input, on A's layout.

        This is not a convenience wrapper around :py:meth:`solve`: the factorisation is what costs,
        and a loop of single solves pays for it once per right-hand side on every backend -- the
        gathering ones refactorise outright, and PETSc re-assembles its Mat and so loses the numeric
        factors even though it keeps the symbolic analysis. A Lyapunov spectrum asks for k
        right-hand sides against one matrix, which is the case this exists for.
        """
        bs = list(bs)
        for b in bs:
            if A.layout != b.layout:
                raise ValueError("the matrix and a right-hand side are on different layouts")
        if A.ncol != A.layout.n:
            raise ValueError("cannot solve a non-square system (%d x %d)" % (A.layout.n, A.ncol))
        if self.problem is None:
            raise RuntimeError("this backend has no problem, so it has no linear solver to solve with")
        if not bs:
            return []
        la = self.problem.get_la_solver()
        n = A.layout.n
        nproc = _nproc()

        if A.layout.distributed or nproc <= 1:
            outs = la.solve_python_built_distributed_many(n, A.layout.nrow_local, A.layout.first_row,
                                                          A.local, [b.local for b in bs])
            return [DistVector(numpy.asarray(o, dtype=numpy.float64), A.layout) for o in outs]

        # Replicated under mpirun: impose a split, as solve() does and for the same reason -- handing
        # the solver (nrow_local=n, first_row=0) from every rank makes it read nproc*n rows and hang.
        from .mpi import get_mpi_rank, mpi_allgather_vector
        rank = int(get_mpi_rank())
        base, rem = divmod(n, nproc)
        nrow_local = base + (1 if rank < rem else 0)
        first_row = rank * base + min(rank, rem)
        sl = slice(first_row, first_row + nrow_local)
        locals_out = la.solve_python_built_distributed_many(
            n, nrow_local, first_row, A.local[sl, :].tocsr(),
            [numpy.ascontiguousarray(b.local[sl]) for b in bs])
        out = []
        for loc in locals_out:
            full = mpi_allgather_vector(n, first_row, nrow_local,
                                        numpy.asarray(loc, dtype=numpy.float64),
                                        context="replicating the solution of a Python-built system")
            out.append(DistVector(numpy.asarray(full, dtype=numpy.float64), A.layout))
        return out


class _ScipyMatrix(DistMatrix):
    """A DistMatrix whose products go through scipy, replicating the operand when distributed."""

    __slots__ = ()

    def matvec(self, v: DistVector) -> DistVector:
        if v.layout.n != self.ncol:
            raise ValueError("cannot multiply a %d-column matrix by a vector of %d rows" % (self.ncol, v.layout.n))
        # The product needs every entry of v, so a distributed operand costs one allgather per call.
        # That is the price of a backend that works without PETSc; it is reported once per run.
        full = v.to_global() if v.layout.distributed else v.local
        return DistVector(numpy.asarray(self.local @ full).ravel(), self.layout)

    def transpose(self) -> DistMatrix:
        if self.layout.distributed:
            raise RuntimeError(
                "transposing a row-distributed matrix needs an off-processor exchange, which this "
                "backend does not do. Ask the multi-assembly for the transposed product instead -- "
                "MultiAssembleRequest.dJdU(V, transposed=True) and the '_transposed' requests exist "
                "for exactly this -- or run with PETSc.")
        out = DistMatrix(self.local.transpose().tocsr(), self.layout, self.layout.n)
        out.__class__ = _ScipyMatrix
        return out


class ScipyBackend(LinearAlgebraBackend):
    """Serial and replicated: scipy throughout, nothing distributed unless an operand says so."""

    name = "scipy"

    def _matrix_class(self):
        return _ScipyMatrix


class _PETScMatrix(DistMatrix):
    """A DistMatrix whose products go through a real PETSc MPIAIJ matrix."""

    __slots__ = ()

    def _petsc_mat(self):
        from petsc4py import PETSc  # type:ignore
        if self.layout.distributed:
            return PETSc.Mat().createAIJ(size=((self.layout.nrow_local, self.layout.n), (self.layout.nrow_local, self.ncol)),
                                         csr=(self.local.indptr.astype(PETSc.IntType, copy=False),
                                              self.local.indices.astype(PETSc.IntType, copy=False),
                                              self.local.data.astype(PETSc.ScalarType, copy=False)))
        return PETSc.Mat().createAIJ(size=(self.layout.n, self.ncol),
                                     csr=(self.local.indptr.astype(PETSc.IntType, copy=False),
                                          self.local.indices.astype(PETSc.IntType, copy=False),
                                          self.local.data.astype(PETSc.ScalarType, copy=False)),
                                     comm=PETSc.COMM_SELF)

    def matvec(self, v: DistVector) -> DistVector:
        if not self.layout.distributed:
            return DistVector(numpy.asarray(self.local @ v.local).ravel(), self.layout)
        from petsc4py import PETSc  # type:ignore
        A = self._petsc_mat()
        A.assemble()
        x = PETSc.Vec().createWithArray(numpy.ascontiguousarray(v.local.astype(PETSc.ScalarType, copy=False)),
                                        size=(v.layout.nrow_local, v.layout.n))
        y = A.createVecLeft()
        A.mult(x, y)
        out = numpy.array(y.getArray(), copy=True)
        A.destroy(); x.destroy(); y.destroy()
        return DistVector(out.real if out.dtype.kind == "c" else out, self.layout)

    def transpose(self) -> DistMatrix:
        from petsc4py import PETSc  # type:ignore
        if not self.layout.distributed:
            out = DistMatrix(self.local.transpose().tocsr(), self.layout, self.layout.n)
            out.__class__ = _PETScMatrix
            return out
        A = self._petsc_mat()
        A.assemble()
        AT = A.transpose()
        start, end = AT.getOwnershipRange()
        indptr, indices, data = AT.getValuesCSR()
        local = scipy.sparse.csr_matrix((data, indices, indptr), shape=(end - start, self.layout.n))
        A.destroy(); AT.destroy()
        out = DistMatrix(local, RowLayout.block(self.layout.n, start, end - start), self.layout.n)
        out.__class__ = _PETScMatrix
        return out


class PETScBackend(LinearAlgebraBackend):
    """Distributed: PETSc MPIAIJ products and transposes, and its own KSP for the solve."""

    name = "petsc"

    def _matrix_class(self):
        return _PETScMatrix


def get_la_backend(problem: Any) -> LinearAlgebraBackend:
    """The backend to use for ``problem``.

    PETSc whenever there is more than one rank and petsc4py is importable, scipy otherwise. The
    decision is made from the rank count and an import, both of which every rank agrees on -- it must
    be, because the two backends reach different collectives and a split choice would deadlock rather
    than merely run slowly. See PETSCSolver._agree_on_reuse_structure_distributed for the same
    hazard in the solver.
    """
    if _nproc() <= 1:
        return ScipyBackend(problem)
    try:
        import petsc4py  # type:ignore  # noqa: F401
        from petsc4py import PETSc  # type:ignore  # noqa: F401
    except Exception:
        # Every rank fails this import or none does (it is the same interpreter and the same
        # PYTHONPATH), so this cannot split the ranks.
        return ScipyBackend(problem)
    return PETScBackend(problem)
