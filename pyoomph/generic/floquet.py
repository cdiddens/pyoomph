from __future__ import annotations
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

"""Floquet multipliers of a periodic orbit by structured condensation of the orbit Jacobian.

The periodic-orbit Jacobian is block bidiagonal in the time direction: with the unknowns ordered
time-major (``global_eqn = nbase*tindex + base_eqn``, see ``PeriodicOrbitHandler::eqn_number``), each
element of the time discretization writes the equations of its own time blocks and reads only the
time blocks it spans. Condensing every such element down to one ``nbase x nbase`` transfer matrix
turns the whole orbit into a product of transfer matrices - the monodromy matrix - whose eigenvalues
are the Floquet multipliers.

This is the classical condensation of Fairgrieve & Jepson (1991) that AUTO performs, and what
BifurcationKit's ``FloquetColl`` does. The alternative, which pyoomph used to do exclusively and
still offers as ``method="eigenproblem"``, is to hand the *whole* orbit Jacobian to a general
eigensolver paired with a mass matrix that is the identity on the wrap-around block alone. That
pencil is correct but has a rank-``nbase`` mass matrix in an ``nT*nbase``-dimensional space, so all
but ``nbase`` of its eigenvalues are infinite and have to be filtered out by a magnitude threshold -
which also throws away the genuinely small multipliers, and makes the number of returned multipliers
depend on the eigensolver. The condensation returns exactly ``nbase`` multipliers, deterministically,
without a shift and without a threshold.

A point worth spelling out, because it is what rules out the more common shooting formulation here:
the mass matrix is completely arbitrary, and may be **singular**. ``M`` never appears on its own in
the condensation - it is already folded into the element blocks, weighted by the collocation shape
derivatives - so no ``M^-1`` is ever formed, and a DAE (an incompressible pressure, an algebraic
constraint) goes through unchanged.

**But read the multipliers of a DAE with care.** Gauss-Legendre collocation is not stiffly accurate -
its stability function has ``|R(inf)| = 1``, not 0 - so an algebraic direction does not decay to a
zero multiplier. Its transfer factor per time element is exactly ``(-1)**order``: the perturbation is
the degree-``order`` polynomial vanishing at the ``order`` Gauss points, and by the symmetry of those
points about the midpoint its value at the end of the element is ``(-1)**order`` times its value at
the start. Over the whole orbit the algebraic directions therefore come out at exactly
``(-1)**(number of time intervals)`` - and with an odd number of intervals that is a spurious ``-1``,
sitting exactly where a period-doubling bifurcation would. This is a property of the time
discretization, not of the condensation (the ``"eigenproblem"`` method produces the same value, just
less accurately); an even number of intervals moves it to ``+1``, next to the trivial multiplier.
"""

from ..typings import *
import numpy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

if TYPE_CHECKING:
    from .. import _pyoomph_core as _pyoomph
    from ..solvers.generic import DefaultMatrixType


# Below this size the per-element blocks are inverted densely, which is considerably faster than a
# sparse LU for the small blocks an ODE orbit produces (nbase=3, order=3 gives a 9x9 block).
_DENSE_BLOCK_SIZE = 256

# How many multipliers to return when none was asked for and the monodromy is too large to form.
_DEFAULT_DOMINANT = 20

# Hard ceiling on forming the monodromy densely even when nearly all multipliers are wanted:
# 4000**2 doubles is 128 MB, and the eigendecomposition of it is the next thing to get expensive.
_DENSE_MONODROMY_CAP = 4000


class FloquetStructureError(RuntimeError):
    """Raised when the assembled orbit Jacobian does not have the block structure the handler describes."""


class _TimeElement:
    """One element of the time discretization, condensed to a transfer matrix.

    ``inds`` are its time-block indices in ascending order. The element fills the orbit-Jacobian rows
    of ``inds[:-1]`` from the columns of all of ``inds``, so writing that sub-block as ``[E0 | L]``
    the linearized equations of this element read ``L @ v[inds[1:]] = -E0 @ v[inds[0]]``, and the
    last ``nbase`` rows of the solution are ``v[inds[-1]]``.
    """

    def __init__(self, inds: Sequence[int], nbase: int):
        self.inds = [int(i) for i in inds]
        self.nbase = nbase
        if len(self.inds) < 2:
            raise FloquetStructureError("A time element must span at least two time blocks, got " + str(self.inds))
        if self.inds != list(range(self.inds[0], self.inds[0] + len(self.inds))):
            # The condensation slices contiguous row/column ranges out of the CSR matrix, which is
            # only the same thing as gathering the listed blocks while they are consecutive. They are,
            # by construction of the time mesh (node ie*order+in of element ie), but a future
            # discretization that reorders them would silently get the wrong sub-block otherwise.
            raise FloquetStructureError("Time element blocks are not consecutive: " + str(self.inds))
        self.row_start = self.inds[0] * nbase
        self.row_end = self.inds[-1] * nbase           # rows of inds[:-1]
        self.col_start = self.row_start
        self.col_end = (self.inds[-1] + 1) * nbase     # columns of all inds
        self._dense_lu: Any = None
        self._sparse_lu: Any = None
        self._E0: NPFloatArray = numpy.zeros((0, 0))

    def factorize(self, J: "DefaultMatrixType") -> None:
        sub = J[self.row_start:self.row_end, self.col_start:self.col_end]
        self._E0 = -sub[:, :self.nbase].toarray()
        L = sub[:, self.nbase:]
        if L.shape[0] <= _DENSE_BLOCK_SIZE:
            self._dense_lu = scipy.linalg.lu_factor(L.toarray())
        else:
            self._sparse_lu = scipy.sparse.linalg.splu(scipy.sparse.csc_matrix(L))

    def _solve(self, rhs: NPFloatArray) -> NPFloatArray:
        if self._dense_lu is not None:
            return scipy.linalg.lu_solve(self._dense_lu, rhs)
        if self._sparse_lu is None:
            raise FloquetStructureError("Time element used before factorize() was called")
        if numpy.iscomplexobj(rhs):
            # SuperLU factorizes the real block, so a complex right-hand side has to be split.
            return self._sparse_lu.solve(numpy.ascontiguousarray(rhs.real)) + 1j * self._sparse_lu.solve(numpy.ascontiguousarray(rhs.imag))
        return self._sparse_lu.solve(rhs)

    def transfer(self, columns: tuple[int, int] | None = None) -> NPFloatArray:
        """The dense ``nbase x nbase`` transfer matrix ``v[inds[-1]] = C @ v[inds[0]]``.

        With ``columns``, only that range of it -- the right-hand sides are independent, which is what
        lets the ranks share the solves out between them.
        """
        rhs = self._E0 if columns is None else self._E0[:, columns[0]:columns[1]]
        return self._solve(rhs)[-self.nbase:, :]

    def propagate(self, v0: NPFloatArray) -> NPFloatArray:
        """The interior states of this element, ``v[inds[1]], ..., v[inds[-1]]``, stacked."""
        return self._solve(self._E0 @ v0)


def time_elements(handler: "_pyoomph.PeriodicOrbitHandler") -> list[_TimeElement]:
    """The condensation elements of the currently active orbit handler, in orbit order."""
    nbase:int = handler.get_base_ndof()
    raw = handler.get_time_element_node_indices()
    if len(raw) == 0:
        raise FloquetStructureError(
            "The active periodic orbit discretization exposes no time-element structure. Floquet "
            "multipliers require mode='collocation' or mode='floquet'.")
    elems = [_TimeElement(inds, nbase) for inds in raw]
    for prev, nxt in zip(elems, elems[1:]):
        if prev.inds[-1] != nxt.inds[0]:
            raise FloquetStructureError(
                "Time elements do not chain: element ending at block " + str(prev.inds[-1]) +
                " is followed by one starting at block " + str(nxt.inds[0]))
    nT = handler.get_num_time_steps()
    if elems[0].inds[0] != 0 or elems[-1].inds[-1] != nT - 1:
        raise FloquetStructureError(
            "Time elements span blocks " + str(elems[0].inds[0]) + ".." + str(elems[-1].inds[-1]) +
            ", but the orbit has " + str(nT) + " time blocks")
    return elems


def check_orbit_jacobian_structure(J: "DefaultMatrixType", elems: list[_TimeElement]) -> None:
    """Verify that every equation row really is confined to the columns of its own time element.

    The condensation is only equivalent to the full pencil if it is, and getting this wrong produces
    plausible-looking multipliers rather than an error. The period column is the one legitimate
    exception: dR/dT is dense down the whole matrix and is dropped from the Floquet problem, exactly
    as the eigenproblem formulation drops it.
    """
    Tcol = J.shape[1] - 1
    indptr, indices = J.indptr, J.indices
    for el in elems:
        lo, hi = indptr[el.row_start], indptr[el.row_end]
        cols = indices[lo:hi]
        stray = cols[(cols < el.col_start) | ((cols >= el.col_end) & (cols != Tcol))]
        if stray.size:
            raise FloquetStructureError(
                "Orbit Jacobian rows " + str(el.row_start) + ":" + str(el.row_end) + " reach columns " +
                str(numpy.unique(stray)[:8]) + " outside the time element's own block range " +
                str(el.col_start) + ":" + str(el.col_end) + ". The Floquet condensation assumes the "
                "block bidiagonal structure that PeriodicOrbitHandler describes.")


def _to_time_major(J: "DefaultMatrixType", handler: "_pyoomph.PeriodicOrbitHandler") -> "DefaultMatrixType":
    """Put a gathered orbit Jacobian back into the naive [u_0 | u_1 | ... | T] row/column order.

    Under ``--distribute`` the augmented rows are interleaved per rank -- rank d's base rows, then its
    rows of each time block -- so the matrix that comes back from a gathered assembly is not in the
    time-major order the condensation slices along, and every block it cut would be the wrong one.
    The handler hands out the naive -> augmented translation; applying it to both axes undoes the
    interleaving. Returns the matrix untouched when the problem is not distributed, where the handler
    reports an empty translation because the two orders already agree.
    """
    order = handler.get_naive_equation_order()
    if len(order) == 0:
        return J
    perm = numpy.asarray(order, dtype=numpy.int64)
    if perm.size != J.shape[0]:
        raise FloquetStructureError(
            "The orbit's equation-order translation has " + str(perm.size) + " entries but the "
            "assembled Jacobian has " + str(J.shape[0]) + " rows")
    return scipy.sparse.csr_matrix(J[perm][:, perm])


class MonodromyOperator(scipy.sparse.linalg.LinearOperator):
    """The monodromy matrix as a matrix-free operator: one solve per time element, never formed.

    For a PDE orbit the transfer matrices are ``nbase x nbase`` *dense*, so the explicit product is
    out of reach long before the factorizations are. Applying the chain to a vector costs one
    triangular solve per time element against factorizations that are computed once and cached.
    """

    def __init__(self, elems: list[_TimeElement], nbase: int, dtype=numpy.dtype(float)):
        self.elems = elems
        self.nbase = nbase
        super().__init__(dtype=dtype, shape=(nbase, nbase))

    def _matvec(self, x: NPFloatArray) -> NPFloatArray:
        v = numpy.asarray(x).reshape(-1)
        for el in self.elems:
            v = el.propagate(v)[-self.nbase:]
        return v


class ShiftInvertMonodromyOperator(scipy.sparse.linalg.LinearOperator):
    """Applies ``(Mono - sigma*I)^-1`` without ever forming the monodromy.

    Plain Arnoldi on the monodromy converges badly when the wanted multipliers are clustered, which is
    the normal situation for a PDE orbit: on the 1D Brusselator, whose multipliers sit on top of each
    other around 0.992, eight of them cost 277 s matrix-free against 65 s for forming the whole
    monodromy and taking all 802 (see section 5). Shift-invert is the standard remedy, and it needs
    ``(Mono - sigma*I)^-1`` -- which looks impossible for an operator only available as a chain.

    It is not, because the chain came from a sparse system in the first place. Take the orbit Jacobian,
    keep its element rows, and replace the wrap-around row ``v_last - v_0 = 0`` with
    ``v_last - sigma*v_0 = b``. The element rows still force ``v_k+1 = C_k v_k``, so ``v_last = Mono v_0``
    and the last row reads ``(Mono - sigma*I) v_0 = b``. One solve of a matrix the size of the orbit
    system therefore delivers the shift-invert apply exactly -- and that matrix is factorized once and
    reused, at the same cost the orbit's own Newton step already pays every iteration.
    """

    def __init__(self, J: "DefaultMatrixType", nbase: int, nT: int, sigma: complex):
        self.nbase, self.nT, self.sigma = nbase, nT, complex(sigma)
        n = nT * nbase
        # Element equations: rows 0 .. (nT-1)*nbase, columns over every time block. Row block nT-1 of
        # the orbit Jacobian is the wrap-around identity and is the one being replaced.
        top = J[0:(nT - 1) * nbase, 0:n]
        eye = scipy.sparse.identity(nbase, format="csr")
        pad = scipy.sparse.csr_matrix((nbase, (nT - 2) * nbase))
        wrap = scipy.sparse.hstack([-self.sigma * eye, pad, eye], format="csr")
        A = scipy.sparse.vstack([top.astype(complex), wrap], format="csc")
        self._lu = scipy.sparse.linalg.splu(A)
        self._rhs = numpy.zeros(n, dtype=complex)
        super().__init__(dtype=numpy.dtype(complex), shape=(nbase, nbase))

    def _matvec(self, x: NPFloatArray) -> NPComplexArray:
        self._rhs[:] = 0.0
        self._rhs[(self.nT - 1) * self.nbase:] = numpy.asarray(x).reshape(-1)
        return self._lu.solve(self._rhs)[:self.nbase]


def _column_range(nbase: int, nproc: int, rank: int) -> tuple[int, int]:
    """The block of transfer-matrix columns this rank solves for. Derived only from
    (nbase, nproc, rank), so every rank agrees on every rank's share without communicating."""
    base, rem = divmod(nbase, nproc)
    lo = rank * base + min(rank, rem)
    return lo, lo + base + (1 if rank < rem else 0)


def transfer_matrices(elems: list[_TimeElement], quiet: bool = True) -> list[NPFloatArray]:
    """Every element's transfer matrix, with the right-hand sides shared out across the ranks.

    Solving each element against all ``nbase`` right-hand sides is the dominant cost of a Floquet
    solve -- 79% of a distributed one at nbase=1282, against 2% for gathering the Jacobian -- and
    every rank used to repeat all of it identically. The columns are independent, so each rank solves
    only its own range and the results are allgathered, in one collective for all elements together.

    This deliberately keeps forming each transfer matrix separately rather than propagating the
    identity through the whole chain, which would do the same solves with one block instead of
    ``Nelem`` of them: on a strongly graded, non-normal chain the accumulated block loses its small
    directions inside ``E0 @ X`` before the solve ever sees them, and the smallest multipliers go from
    ~1e-15 relative error to complete nonsense. ``test_periodic_schur_beats_the_product_at_the_bottom``
    catches that.
    """
    from .mpi import get_mpi_world_comm, get_mpi_rank, get_mpi_nproc
    comm = get_mpi_world_comm()
    nproc = get_mpi_nproc() if comm is not None else 1
    if comm is None or nproc <= 1:
        return [el.transfer() for el in elems]

    nbase = elems[0].nbase
    rank = get_mpi_rank()
    ranges = [_column_range(nbase, nproc, r) for r in range(nproc)]
    lo, hi = ranges[rank]
    if not quiet:
        print("Floquet: solving transfer columns %d:%d of %d on rank %d" % (lo, hi, nbase, rank))
    # Transposed and stacked, so this rank's whole contribution is one contiguous run.
    local = numpy.concatenate([numpy.ascontiguousarray(el.transfer((lo, hi)).T) for el in elems]) \
        if hi > lo else numpy.zeros((0, nbase))

    from mpi4py import MPI
    counts = [len(elems) * (h - l) * nbase for (l, h) in ranges]
    displs = numpy.concatenate(([0], numpy.cumsum(counts)[:-1])).astype(int)
    buf = numpy.empty(int(sum(counts)), dtype=float)
    comm.Allgatherv([numpy.ascontiguousarray(local.reshape(-1)), MPI.DOUBLE],
                    [buf, counts, displs, MPI.DOUBLE])

    out = [numpy.empty((nbase, nbase)) for _ in elems]
    for r, (l, h) in enumerate(ranges):
        if h <= l:
            continue
        chunk = buf[displs[r]:displs[r] + counts[r]].reshape(len(elems), h - l, nbase)
        for ie in range(len(elems)):
            out[ie][:, l:h] = chunk[ie].T
    return out


def monodromy_matrix(elems: list[_TimeElement], quiet: bool = True) -> NPFloatArray:
    """The dense ``nbase x nbase`` monodromy ``C_last @ ... @ C_0``."""
    Cs = transfer_matrices(elems, quiet=quiet)
    mono = Cs[0]
    for C in Cs[1:]:
        mono = C @ mono
    return mono


def monodromy_eigendecomposition(elems: list[_TimeElement], n: int,
                                 quiet: bool = True) -> tuple[NPComplexArray, NPComplexArray]:
    """The n dominant eigenpairs of the monodromy: transfers in parallel, the rest on rank 0.

    Only the transfer solves split by column (:func:`transfer_matrices`); the matrix product and the
    eigendecomposition after them are O(nbase**3) and every rank used to repeat both. That is not
    merely wasted -- measured at nbase=1282, the redundant `numpy.linalg.eig` went from 0.68 s on one
    rank to 1.76 s on four as the copies contended for memory bandwidth, which made a 4-rank Floquet
    solve slower overall than a serial one even though its solves were 2.8x faster. Rank 0 does them
    once and broadcasts the result.
    """
    from .mpi import get_mpi_world_comm, get_mpi_rank, get_mpi_nproc
    comm = get_mpi_world_comm()
    nproc = get_mpi_nproc() if comm is not None else 1
    Cs = transfer_matrices(elems, quiet=quiet)

    def decompose() -> tuple[NPComplexArray, NPComplexArray]:
        mono = Cs[0]
        for C in Cs[1:]:
            mono = C @ mono
        eigs, eigv = numpy.linalg.eig(mono)  # type: ignore[misc]
        if n < eigs.size:
            keep = numpy.argsort(-numpy.abs(eigs))[:n]  # the n that decide stability
            eigs, eigv = eigs[keep], eigv[:, keep]
        return eigs, eigv

    if comm is None or nproc <= 1:
        return decompose()
    pair = decompose() if get_mpi_rank() == 0 else None
    return comm.bcast(pair, root=0)


def _positive_diagonal_qr(A: NPComplexArray) -> tuple[NPComplexArray, NPComplexArray]:
    """QR with the sign freedom fixed so that R has a positive real diagonal.

    Without this the factorization is only unique up to a unit-modulus diagonal, the diagonal of R
    is not a magnitude, and the sweep-to-sweep comparison that detects convergence compares
    quantities that differ by an arbitrary phase.
    """
    Q, R = numpy.linalg.qr(A)
    d = numpy.diagonal(R).copy()
    phase = numpy.where(numpy.abs(d) > 0, d / numpy.where(numpy.abs(d) > 0, numpy.abs(d), 1.0), 1.0)
    return cast(NPComplexArray, Q * phase[None, :]), cast(NPComplexArray, numpy.conj(phase)[:, None] * R)


def _cluster_bounds(W: NPComplexArray, tol: float) -> list[tuple[int, int]]:
    """Split 0..n at every k whose whole lower-left corner of W is negligible.

    W is the overlap of consecutive sweeps' Schur bases. Where subspace iteration has separated the
    spectrum, W is upper triangular there and the split is exact; where two multipliers have (nearly)
    the same modulus -- a complex conjugate pair above all -- it never separates, and those indices
    stay together in one block that is diagonalized directly.
    """
    n = W.shape[0]
    cuts = [0]
    for k in range(1, n):
        if numpy.max(numpy.abs(W[k:, :k])) < tol:
            cuts.append(k)
    cuts.append(n)
    return list(zip(cuts[:-1], cuts[1:]))


def periodic_schur_multipliers(elems: list[_TimeElement], max_sweeps: int = 200,
                               tol: float = 1e-11, quiet: bool = True
                               ) -> tuple[NPComplexArray, int, int]:
    """Multipliers of the transfer-matrix product without ever forming the product.

    Periodic QR, i.e. subspace iteration run through the chain: starting from an orthonormal basis
    Q, sweeping ``V, R_ie = qr(C_ie V)`` along the orbit gives, exactly and at every sweep,

        Mono @ Q_old = Q_new @ S,     S = R_{p-1} ... R_0  upper triangular

    so with ``W = Q_old^H Q_new`` the multipliers are the eigenvalues of ``W @ S``. Subspace
    iteration drives W upper triangular (at rate ``|lambda_{k+1}/lambda_k|`` per sweep), and where it
    is, ``W @ S`` is upper triangular too and its diagonal is the spectrum -- read off as
    ``W[k,k] * exp(sum_ie log R_ie[k,k])``, which is where the overflow is avoided: the product of the
    diagonals is accumulated in logs and never formed.

    Indices that subspace iteration cannot separate (equal moduli) are left as a diagonal block and
    diagonalized directly, with each factor divided by the geometric mean of its own diagonal so the
    block product stays O(1) and the scale is carried alongside in logs.

    Returns ``(multipliers, sweeps_used, largest_block)``.
    """
    Cs = transfer_matrices(elems, quiet=quiet)
    n = Cs[0].shape[0]
    Q: NPComplexArray = numpy.eye(n, dtype=complex)
    Rs: list[NPComplexArray] = []
    W: NPComplexArray = numpy.eye(n, dtype=complex)
    sweeps = 0
    prev_off = numpy.inf
    stagnant = 0
    for sweeps in range(1, max_sweeps + 1):
        Q_old = Q
        Rs = []
        V = Q
        for Ci in Cs:
            V, R = _positive_diagonal_qr(Ci @ V)
            Rs.append(R)
        Q = V
        W = Q_old.conj().T @ Q
        off = float(numpy.max(numpy.abs(numpy.tril(W, -1))))
        if off < tol:
            break
        # Indices whose multipliers share a modulus -- a complex conjugate pair, or the +1 and -1 a
        # DAE produces -- can never be separated by subspace iteration: its rate is
        # |lambda_{k+1}/lambda_k|, which is 1 there. Those are diagonalized as a block below, so once
        # the off-triangular part stops shrinking there is nothing left for more sweeps to do.
        # Without this, the DAE cases spent all max_sweeps sweeps reaching the same answer.
        stagnant = stagnant + 1 if off > 0.9 * prev_off else 0
        prev_off = off
        if stagnant >= 3:
            break

    # log of the diagonal of S = prod(R_ie), accumulated rather than multiplied. A zero diagonal
    # entry (an exactly singular transfer matrix) is a zero multiplier, and -inf carries that.
    with numpy.errstate(divide="ignore"):
        logdiag = numpy.sum(numpy.log(numpy.array([numpy.real(numpy.diagonal(R)) for R in Rs])), axis=0)

    blocks = _cluster_bounds(W, tol)
    eigs = numpy.zeros(n, dtype=complex)
    for a, b in blocks:
        if b - a == 1:
            eigs[a] = W[a, a] * numpy.exp(logdiag[a]) if numpy.isfinite(logdiag[a]) else 0.0
            continue
        # Scale every factor by the geometric mean of its own (positive) diagonal, so that the block
        # product cannot overflow: within a block the moduli are equal by construction.
        blk: NPComplexArray = numpy.eye(b - a, dtype=complex)
        logscale = 0.0
        singular = False
        for R in Rs:
            sub = R[a:b, a:b]
            d = numpy.real(numpy.diagonal(sub))
            if numpy.any(d <= 0):
                singular = True
                break
            g = float(numpy.exp(numpy.mean(numpy.log(d))))
            blk = (sub / g) @ blk
            logscale += float(numpy.log(g))
        if singular:
            eigs[a:b] = 0.0
        else:
            eigs[a:b] = numpy.exp(logscale) * numpy.linalg.eigvals(W[a:a + (b - a), a:b] @ blk)
    largest_block = max(b - a for a, b in blocks)
    if not quiet:
        print("Floquet: periodic Schur used %d sweep(s), largest unseparated block %d"
              % (sweeps, largest_block))
    return eigs, sweeps, largest_block


def orbit_eigenfunctions(elems: list[_TimeElement], V: NPComplexArray, nbase: int, nT: int) -> NPComplexArray:
    """Push monodromy eigenvectors back through the chain to get their eigenfunctions over the orbit.

    ``V`` holds them as columns; the result has one row each, laid out as the eigenproblem
    formulation produces: ``nT*nbase`` entries, one time block after the other, plus a trailing zero
    for the period unknown that the Floquet problem does not carry.

    All of them go through the chain together. Done one eigenvector at a time -- which is what this
    did at first -- asking for every multiplier of an nbase=1282 orbit spent 185 s here against 3 s
    for the multipliers themselves, because each column was its own pass of sparse solves. The
    arithmetic per column is unchanged; only the batching is.
    """
    V = numpy.atleast_2d(numpy.asarray(V, dtype=complex))
    if V.shape[0] != nbase:
        V = V.T
    out = numpy.zeros((V.shape[1], nT * nbase + 1), dtype=complex)
    out[:, :nbase] = V.T
    X = cast(NPFloatArray, V)
    for el in elems:
        interior = el.propagate(X)
        out[:, el.inds[1] * nbase:(el.inds[-1] + 1) * nbase] = interior.T
        X = cast(NPFloatArray, interior[-nbase:])
    return out


def _eigenvector_by_inverse_iteration(mono: NPFloatArray, lam: complex,
                                      rng_index: int) -> NPComplexArray:
    """One eigenvector of the (small, dense) monodromy for an eigenvalue computed elsewhere.

    The periodic Schur route produces eigenvalues without ever forming a Schur vector in the original
    basis, so the vectors are recovered here instead. Inverse iteration is well conditioned precisely
    because lam is an accurate eigenvalue: mono - lam*I is nearly singular and the solve lands in its
    null direction in one or two steps.
    """
    n = mono.shape[0]
    A = mono.astype(complex) - lam * numpy.eye(n, dtype=complex)
    # A deterministic, non-special starting vector: a constant one is orthogonal to too many things.
    v = numpy.exp(1j * numpy.arange(n) * (1.0 + rng_index)) / numpy.sqrt(n)
    for _ in range(3):
        try:
            w = numpy.linalg.solve(A, v)
        except numpy.linalg.LinAlgError:
            # Exactly singular is the ideal case, not a failure: fall back to a least-squares null
            # vector, which is what the solve was approximating anyway.
            w = numpy.linalg.lstsq(A, v, rcond=None)[0]
        nrm = numpy.linalg.norm(w)
        if not numpy.isfinite(nrm) or nrm == 0:
            break
        v = w / nrm
    return v


def floquet_multipliers(problem, n: int | None = None, quiet: bool = True,
                        dense_threshold: int = 2000,
                        check_structure: bool = True,
                        periodic_schur: bool = False,
                        shift_invert: bool = True,
                        sigma: complex | None = None) -> tuple[NPComplexArray, NPComplexArray]:
    """Floquet multipliers and eigenfunctions of the currently tracked periodic orbit.

    ``n`` is the number of multipliers wanted, or ``None`` for all of them (which is only the default
    while the monodromy can be formed; above ``dense_threshold`` it becomes the dominant few).

    ``shift_invert`` (on by default) and ``sigma`` control the matrix-free route only: the multipliers
    are sought near ``sigma``, which defaults to just outside the unit circle, where the stability
    question lives. Set ``shift_invert=False`` for plain "largest magnitude" Arnoldi.

    ``periodic_schur`` selects :func:`periodic_schur_multipliers` instead of forming the monodromy.
    It is dense-only and costs ``sweeps * Nelem`` matrix products, so it is for accuracy at modest
    ``nbase``, not for scale.

    Returns ``(multipliers, eigenfunctions)`` unsorted; the caller applies its own ordering and
    filtering. The eigenfunctions are laid out as described in :func:`orbit_eigenfunctions`.
    """
    handler = problem.assembly_handler_pt()
    nbase:int = handler.get_base_ndof()
    nT = handler.get_num_time_steps()
    elems = time_elements(handler)

    J = problem.assemble_jacobian(with_residual=False)
    expected = nT * nbase + 1
    if J.shape[0] != expected:
        raise FloquetStructureError(
            "Orbit Jacobian has size " + str(J.shape[0]) + ", expected " + str(expected))
    J = _to_time_major(J, handler)
    if check_structure:
        check_orbit_jacobian_structure(J, elems)
    for el in elems:
        el.factorize(J)

    if n is None:
        n = nbase if nbase <= dense_threshold else min(_DEFAULT_DOMINANT, nbase - 2)
        if n != nbase and not quiet:
            print("Floquet: no count given for a problem with " + str(nbase) + " degrees of freedom; "
                  "computing the " + str(n) + " dominant multipliers")
    if n < 1:
        raise ValueError("Cannot request " + str(n) + " Floquet multipliers")
    n = min(n, nbase)

    # Which way round is cheaper is decided by n/nbase, not by nbase alone. Forming the monodromy
    # costs one solve per time element with nbase right-hand sides; applying it matrix-free costs one
    # solve per time element per Arnoldi iteration, so the matrix-free route only pays off while few
    # multipliers are wanted. Asking for nearly all of them through ARPACK is the worst of both -- it
    # was measurably slower than the dense route at nbase=802 -- and k must stay below nbase anyway.
    if periodic_schur:
        if nbase > _DENSE_MONODROMY_CAP:
            raise ValueError(
                "periodic_schur is a dense method and this problem has " + str(nbase) +
                " degrees of freedom; use the default method instead.")
        eigs, _, largest_block = periodic_schur_multipliers(elems, quiet=quiet)
        mono = monodromy_matrix(elems)
        order = numpy.argsort(-numpy.abs(eigs))[:n]
        eigs = eigs[order]
        eigv = numpy.column_stack([_eigenvector_by_inverse_iteration(mono, lam, i)
                                   for i, lam in enumerate(eigs)]) if eigs.size else numpy.zeros((nbase, 0), dtype=complex)
        eigfuncs = orbit_eigenfunctions(elems, eigv, nbase, nT)
        if not quiet and eigs.size:
            print("Floquet: trivial multiplier deviates from 1 by {:.3e}".format(
                numpy.min(numpy.abs(eigs - 1.0))))
        return numpy.asarray(eigs, dtype=complex), eigfuncs

    use_dense = nbase <= dense_threshold or (n * 4 > nbase and nbase <= _DENSE_MONODROMY_CAP)
    if use_dense:
        if nbase > dense_threshold and not quiet:
            print("Floquet: forming the " + str(nbase) + "x" + str(nbase) + " monodromy anyway, since "
                  + str(n) + " of its multipliers were asked for")
        eigs, eigv = monodromy_eigendecomposition(elems, n, quiet=quiet)
    else:
        if n >= nbase - 1:
            raise ValueError(
                "Cannot compute " + str(n) + " of " + str(nbase) + " Floquet multipliers matrix-free "
                "(Arnoldi needs strictly fewer). Raise dense_threshold to form the monodromy instead.")
        op = MonodromyOperator(elems, nbase)
        if shift_invert:
            # Shift-invert around sigma, which for a stability question belongs just outside the unit
            # circle. Plain "LM" Arnoldi is the right TARGET but converges badly when the wanted
            # multipliers are clustered, which is the normal case for a PDE orbit: 277 s for eight of
            # the Brusselator's, against 18 s this way, to 2.6e-13.
            if sigma is None:
                sigma = 1.0 + 1e-3
            OPinv = ShiftInvertMonodromyOperator(J, nbase, nT, sigma)
            if not quiet:
                print("Floquet: matrix-free, shift-inverted about {:.4g}".format(sigma)) #type:ignore[str-format] # complex formats fine with "g"
            eigs, eigv = scipy.sparse.linalg.eigs(op, k=n, sigma=sigma, OPinv=OPinv)  # type: ignore[misc]
        else:
            eigs, eigv = scipy.sparse.linalg.eigs(op, k=n, which="LM")  # type: ignore[misc]

    eigfuncs = orbit_eigenfunctions(elems, eigv, nbase, nT)
    if not quiet and eigs.size:
        trivial = numpy.min(numpy.abs(eigs - 1.0))
        print("Floquet: trivial multiplier deviates from 1 by {:.3e}".format(trivial))
    return numpy.asarray(eigs, dtype=complex), eigfuncs
