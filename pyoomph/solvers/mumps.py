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

"""The direct MUMPS backends (idname "mumps"), linear solver and eigensolver.

MUMPS is already reachable from pyoomph as "petsc_mumps"/"slepc_mumps", but only through PETSc and
SLEPc - which means a full PETSc+SLEPc stack in two scalar flavours, a real one for the linear solves
and a complex one for the azimuthal and Floquet stability problems, plus PETSc's Mat/Vec marshalling
on every solve. This module talks to dmumps/zmumps directly instead, through the separate
``pyoomph_mumps`` package.

Two backends are registered, both named "mumps" (the linear and eigen registries are separate, so the
name can serve both, exactly as "pardiso" already does):

* :class:`MumpsLinearSolver` - serial, or natively distributed under MPI (MUMPS ICNTL(18)=3, each
  rank supplying its own row block), including the sparsity-pattern reuse that lets a transient run
  pay for the ordering once;
* :class:`MumpsSpectraEigenSolver` - pyoomph's existing Spectra Arnoldi driver with MUMPS supplying
  the shift-and-invert factorisation, in real *and* complex arithmetic.

The eigensolver is a thirty-line subclass rather than a second implementation because the hard part
of an eigensolver here is not the Arnoldi iteration - Spectra provides that - but the factorisation
of ``J - sigma*M`` that the iteration calls at every step. Substituting MUMPS for Pardiso/SuperLU
there is the entire difference, so everything else (the choice of sigma, the nu = 1/(lambda-sigma)
back-transform, the ordering rules, the singular-mass-matrix cutoff, the retry ladder, the dense
fallback for tiny systems) is inherited unchanged and cannot drift away from the Spectra backend.
"""

from .generic import (GenericLinearSystemSolver, GenericEigenSolver, DefaultMatrixType, SolverError,
                      _MUMPS_ICNTL14_ERRORS, _MUMPS_ICNTL23_ERROR, _next_mumps_icntl14)

import numpy
import scipy.sparse  # type:ignore

from ..typings import *
if TYPE_CHECKING:
    from ..generic.problem import Problem

# Raised as an ImportError, and deliberately at import time, before the @register_solver decorators
# below have run: GenericLinearSystemSolver.factory_solver() imports this module to find the solver,
# so failing here means the classes are never registered and the user gets the standard "unknown
# solver, the following are available" message instead of one that appears to offer a backend which
# cannot run. Same mechanism as pyoomph/solvers/spectra.py's has_spectra check.
try:
    import pyoomph_mumps  # type:ignore
except ImportError as _e:
    raise ImportError("the pyoomph_mumps package is not installed, so the 'mumps' solvers are not "
                      "available. See https://github.com/pyoomph/pyoomph for how to build it; it "
                      "downloads and builds MUMPS itself by default.") from _e

from ..generic.mpi import has_mpi as _pyoomph_has_mpi

# The one mismatch that fails confusingly rather than loudly. A serial MUMPS links a stub library
# (libmpiseq) that defines its OWN MPI_Init, MPI_Comm_f2c and friends; put that in the same process
# as an MPI-enabled pyoomph and the two either initialise MPI twice or not at all, and what comes out
# is a hang, or a solve on a communicator nobody meant, rather than an error. Neither library
# notices, so the check has to be here.
if bool(pyoomph_mumps.has_mpi) != bool(_pyoomph_has_mpi()):
    raise ImportError(
        "pyoomph_mumps was built " + ("with" if pyoomph_mumps.has_mpi else "without") +
        " MPI support but pyoomph was built " + ("with" if _pyoomph_has_mpi() else "without") +
        " it. The two must agree: a serial MUMPS carries its own MPI stub, so mixing them gives a "
        "hang or a wrong communicator rather than an error message. Rebuild pyoomph_mumps with "
        "-DPYOOMPH_MUMPS_USE_MPI=" + ("ON" if _pyoomph_has_mpi() else "OFF") +
        " (its build_for_develop.sh reads pyoomph's setting and matches it automatically).")


class MumpsSolverError(SolverError):
    """MUMPS could not factorise or solve the system.

    A subclass of SolverError, so the nanobind shim in src/nanobind/solver.cpp turns it into an
    oomph NewtonSolverError: an adaptive time step or an arclength step then shrinks and retries,
    which is the right response to a Jacobian that has gone singular on the way to a fold, rather
    than ending the run.
    """


def _as_int32(a: Any) -> Any:
    return numpy.ascontiguousarray(a, dtype=numpy.int32)


def _as_f64(a: Any) -> Any:
    """The values array as contiguous float64.

    A no-op for what oomph-lib hands over, which is already exactly that. It is here so that a
    caller who is one conversion away gets a copy rather than nanobind's "incompatible function
    arguments", which names the C++ signature and not the array that failed to match it."""
    return numpy.ascontiguousarray(a, dtype=numpy.float64)


def _factorize_growing_workspace(solver: Any, max_retries: int, quiet: bool,
                                 on_grow: Callable[[int], None] | None = None) -> int:
    """Factorise, raising ICNTL(14) and trying again when MUMPS runs out of working space.

    MUMPS sizes an internal work array from the fill-in its ANALYSIS predicted. Numerical pivoting
    can need more room than that, and MUMPS then stops with one of _MUMPS_ICNTL14_ERRORS rather than
    reallocating. That is a routine, recoverable condition, not a failed solve: the remedy in MUMPS's
    own manual is to raise ICNTL(14) - the percentage of slack added to the prediction - and
    factorise again, which is what this does.

    Treating it as a failure instead is expensive in a way that does not look like a bug. The Newton
    step is abandoned, oomph shrinks the timestep and retries, the same analysis under-predicts the
    same fill-in, and the run crawls: nacl_capillary_evaporation.py spent 13297 factorisations on one
    2430-dof system that way and never finished, while the same script takes ten seconds with any
    other backend. Nothing in the output says "solver": it just looks like a hard problem.

    The raised value is kept by the caller for the rest of the run, not reverted after the retry - a
    matrix that needed the extra room once needs it again at the next step. Returns how many times
    the workspace was grown.

    Collective-safe under MPI without any agreement of its own: INFOG is global in MUMPS, so every
    rank sees the same code from the same collective call and takes the same branch.
    """
    grown = 0
    while True:
        try:
            solver.factorize()
            return grown
        except pyoomph_mumps.MumpsError:
            infog1 = solver.infog(1)
            if infog1 == _MUMPS_ICNTL23_ERROR:
                raise MumpsSolverError(
                    "MUMPS hit the working-memory cap set by ICNTL(23). Raise or remove that cap; "
                    "ICNTL(14) cannot buy room a cap forbids.")
            if infog1 not in _MUMPS_ICNTL14_ERRORS or grown >= max_retries:
                raise
            # Guaranteed by the C++ side, which keeps the analysis across exactly these codes so that
            # the documented recovery is possible at all.
            assert solver.last_error_was_workspace() and solver.has_analysis
            current = solver.get_icntl(14) or 20  # 0 means "MUMPS's own default", which is 20
            new_value = _next_mumps_icntl14(current)
            if new_value == current:
                raise  # already at the cap; more slack is not going to appear
            solver.set_icntl(14, new_value)
            grown += 1
            if on_grow is not None:
                on_grow(new_value)
            if not quiet:
                print("MUMPS ran out of working space (INFOG(1)=" + str(infog1) +
                      "); retrying with ICNTL(14)=" + str(new_value))


@GenericLinearSystemSolver.register_solver()
class MumpsLinearSolver(GenericLinearSystemSolver):
    """MUMPS as pyoomph's linear solver, serial or distributed.

    Distributed for real, not gathered: under mpirun every rank hands MUMPS its own row block
    (ICNTL(18)=3), so the factorisation itself is parallel. Only the right-hand side and the solution
    are centralized on rank 0 - they are n doubles against a matrix with orders of magnitude more
    entries, and MUMPS's distributed right-hand side (ICNTL(20)=10/11) is a refinement that can come
    later without changing anything a caller sees.
    """

    idname = "mumps"
    solves_natively_distributed = True
    # Deliberately left off: solve_distributed() below is a real distributed implementation, so the
    # base class's gather-onto-rank-0 fallback would never be reached, and claiming it would only
    # promise that solve_serial() issues no collective - which is true, but says nothing useful here.
    gathers_to_root_under_mpi = False

    def __init__(self, problem: "Problem", verbose: bool = False):
        super().__init__(problem)
        #: 0 keeps MUMPS silent (its INFOG codes reach you through MumpsSolverError either way);
        #: 1 turns its error stream back on, 2 and above its diagnostics and statistics.
        self.verbose = 1 if verbose else 0
        #: ICNTL entries applied to every new MUMPS instance, 1-based as in the user guide. The one
        #: worth knowing about is ICNTL(14), the percentage by which MUMPS oversizes its working
        #: space (default 20): a factorisation that fails with INFOG(1) around -9 wants more.
        self.icntl_override: dict[int, int] = {}
        #: Reuse the analysis (JOB=1) whenever the sparsity pattern has not moved, running only the
        #: numerical factorisation (JOB=2). Set to False for an A/B measurement of what that is
        #: worth - it then builds a fresh instance every time, which is the only way to be sure
        #: nothing is being reused.
        self.reuse_symbolic_factorisation = True
        #: Ask MUMPS for the determinant (ICNTL(33)=1), which is what makes get_determinant_sign()
        #: answer. Off by default because it is not free and only quick continuation wants it.
        self.compute_determinant_sign = False
        # Which path each op_flag==1 took. Counters rather than timings, for the same reason the
        # Pardiso backend keeps them: a benchmark cannot otherwise tell a win from a silent fallback.
        self.n_full_factorisations = 0     # JOB=1 + JOB=2
        self.n_numeric_factorisations = 0  # JOB=2 alone, the analysis reused
        self.n_symmetric_factorisations = 0
        #: How many times a factorisation ran out of working space and was retried with a larger
        #: ICNTL(14). A few at the start of a run are normal; a number that keeps climbing means the
        #: analysis is a poor fit for the values and is worth knowing about.
        self.n_workspace_growths = 0
        #: How far ICNTL(14) may be grown before a workspace failure is reported as one. Four
        #: doublings take the default 20 to 640, which is the whole useful range below the cap.
        self.max_workspace_retries = 4
        self._solver: Any = None
        self._structure_id: int = 0
        self._active_sym: int = 0
        self._nthreads: int | None = None
        self._layout: Any = None  # the distributed row layout of the current factorisation

    # ------------------------------------------------------------------ housekeeping

    def set_num_threads(self, nthreads: int | None):
        self._nthreads = None if not nthreads else int(nthreads)
        if self._solver is not None:
            # ICNTL(16) is read at each phase, so this takes effect from the next factorisation on;
            # it does not need the instance to be rebuilt.
            self._solver.set_icntl(16, self._nthreads or 0)

    def _invalidate(self) -> None:
        """Throw the MUMPS instance away, so the next factorisation starts from nothing.

        Called whenever something raised. MUMPS's state after a failed phase is not a base for the
        next one, and both reuse tiers key off the instance still being there - so without this a
        retry (which oomph makes with a SMALLER step, i.e. a different Jacobian that deserves its own
        ordering anyway) walks straight back into the handle that just failed.
        """
        self._solver = None
        self._structure_id = 0
        self._layout = None

    def _before_assigning_equation_numbers(self) -> None:
        # The dof numbering is about to change, so the pattern and everything derived from it is
        # meaningless. Nothing else in this class can notice that on its own: the pattern comparison
        # would simply see a different matrix and rebuild, but the row layout would not.
        self._invalidate()

    def _new_solver(self, sym: int, comm_fortran: int) -> Any:
        s = pyoomph_mumps.MumpsSolverReal(sym=sym, comm_fortran=comm_fortran, verbosity=self.verbose)
        if self.compute_determinant_sign:
            s.set_icntl(33, 1)
        if self._nthreads:
            s.set_icntl(16, self._nthreads)
        for i, v in self.icntl_override.items():
            s.set_icntl(int(i), int(v))
        return s

    def _decide_sym(self) -> int:
        """0 (LU) or 2 (LDL^T), from pyoomph's symbolic symmetry proof.

        Re-asked at every factorisation, never cached: the same problem flips between symmetric and
        not when a bifurcation tracker or a custom assembler is switched on. Never sym=1 - that is
        the Cholesky, and positive definiteness is not something that can be proven symbolically.
        """
        if self._use_symmetric_factorisation_now():
            self.n_symmetric_factorisations += 1
            return 2
        return 0

    def get_determinant_sign(self) -> int | None:
        # None means "cannot say", 0 means "exactly singular" - two different answers, so the
        # not-asked and not-factorised cases must not fall through to the sign itself.
        if self._solver is None or not self.compute_determinant_sign:
            return None
        if not self._solver.has_factorisation:
            return None
        return self._solver.determinant_sign()

    def get_last_used_mem_size_in_kb(self) -> int:
        """Peak memory of the last factorisation, in kB. INFOG(22) reports it in MB."""
        if self._solver is None:
            return 0
        try:
            return int(self._solver.infog(22)) * 1024
        except Exception:
            return 0

    # ------------------------------------------------------------------ serial

    def solve_serial(self, op_flag: int, n: int, nnz: int, nrhs: int, values: NPFloatArray,
                     rowind: NPAnyIntArray, colptr: NPAnyIntArray, b: NPFloatArray, ldb: int,
                     transpose: int) -> int:
        try:
            return self._solve_serial(op_flag, n, nnz, nrhs, values, rowind, colptr, b, ldb, transpose)
        except pyoomph_mumps.MumpsError as e:
            self._invalidate()
            raise MumpsSolverError(str(e)) from e
        except Exception:
            self._invalidate()
            raise

    def _solve_serial(self, op_flag: int, n: int, nnz: int, nrhs: int, values: NPFloatArray,
                      rowind: NPAnyIntArray, colptr: NPAnyIntArray, b: NPFloatArray, ldb: int,
                      transpose: int) -> int:
        if op_flag == 1:
            # The slots carry SuperLU's names but CSR data: `rowind` holds the column indices and
            # `colptr` the row starts, 0-based, columns not sorted within a row (which MUMPS does not
            # mind - see the test in pyoomph_mumps/tests). Handing this straight to MUMPS as row-major
            # triplets therefore gives A itself, where the scipy backend, which reads the same arrays
            # as CSC, gets A^T and undoes it with the transpose flag. See _apply_transpose below.
            sym = self._decide_sym()
            comm = pyoomph_mumps.USE_COMM_WORLD
            if self._solver is None or self._active_sym != sym or not self.reuse_symbolic_factorisation:
                # A symmetry flip changes which triangle is stored, so it needs a fresh instance
                # rather than a re-analysis - and it is an intentional flip (a tracker was toggled),
                # not a pattern bug, so it is done quietly.
                self._solver = self._new_solver(sym, comm)
                self._active_sym = sym
                self._structure_id = 0
            self._factorise(lambda s: s.set_matrix_csr(n, _as_int32(colptr), _as_int32(rowind),
                                                       _as_f64(values)))
        elif op_flag == 2:
            self.setup_solver()
            if self._solver is None or not self._solver.has_factorisation:
                raise MumpsSolverError("MUMPS was asked to resolve (op_flag=2) without a "
                                       "factorisation. Something discarded it in between - see "
                                       "_invalidate().")
            self._apply_transpose(transpose)
            # Through _solve_newton_step like every other backend, so deflation and an augmented
            # assembler's custom solve routine keep working. Serial entry point: no row offset and
            # no reduction.
            b[:] = self._solve_newton_step(self._back_substitute, b)
        else:
            raise RuntimeError("Cannot handle MUMPS op_flag " + str(op_flag) + " yet")
        return 0

    def _apply_transpose(self, transpose: int) -> None:
        """Tell MUMPS which of A and A^T to solve with.

        oomph passes transpose=1 because it hands SuperLU CSR data through CSC-named slots, i.e. it
        is asking for the transpose of what SuperLU would read. Reading the same arrays as CSR, as
        this backend does, already gives A - so transpose=1 means "solve with A", MUMPS's ICNTL(9)=1.
        Getting this the wrong way round solves the transposed system without any complaint, which is
        why it is spelled out rather than ignored.
        """
        assert self._solver is not None
        self._solver.set_icntl(9, 1 if transpose == 1 else 0)

    def _back_substitute(self, rhs: NPFloatArray) -> NPFloatArray:
        assert self._solver is not None
        x = numpy.ascontiguousarray(rhs, dtype=numpy.float64).copy()
        self._solver.solve(x)
        return x

    def _factorise(self, install: Callable[[Any], bool]) -> None:
        """Install the matrix through `install`, then analyse if needed and factorise.

        The pattern is COMPARED by pyoomph_mumps, not taken on trust: problem.jacobian_structure_id
        promises it has not moved, but that promise does not hold on an augmented (tracking) system,
        whose pattern is value-filtered - and applying an old elimination tree to a new pattern is a
        silently wrong answer, not a crash. The id is therefore used only to decide whether a
        mismatch is worth reporting.
        """
        assert self._solver is not None
        structure_id = self.problem.jacobian_structure_id
        unchanged = install(self._solver)
        if structure_id != 0 and structure_id == self._structure_id and not unchanged:
            self._report_structure_id_mismatch("the MUMPS analysis")
        if unchanged and self._solver.has_analysis:
            self.n_numeric_factorisations += 1
        else:
            self._solver.analyse()
            self.n_full_factorisations += 1
        self._grow_and_factorise()
        self._structure_id = structure_id

    def _grow_and_factorise(self) -> None:
        """factorize(), growing ICNTL(14) if MUMPS runs out of room. See _factorize_growing_workspace.

        The grown value is recorded in icntl_override so that it survives the instance: a failure
        elsewhere drops the MUMPS handle (_invalidate) and the next factorisation builds a fresh one,
        which would otherwise start again from the default that has already been shown to be too
        small.
        """
        assert self._solver is not None

        def remember(value: int) -> None:
            self.icntl_override[14] = value
            self.n_workspace_growths += 1

        _factorize_growing_workspace(self._solver, self.max_workspace_retries,
                                     self.problem.is_quiet(), remember)

    # ------------------------------------------------------------------ distributed

    def solve_distributed(self, op_flag: int, allow_permutations: int, n: int, nnz_local: int,
                          nrow_local: int, first_row: int, values: NPFloatArray,
                          col_index: NPIntArray, row_start: NPIntArray, b: NPFloatArray, nprow: int,
                          npcol: int, doc: int, data: NPUInt64Array, info: NPIntArray) -> None:
        from ..generic.mpi import get_mpi_nproc
        if get_mpi_nproc() <= 1:
            # Reachable when a solver is forced down the distributed path on one process; the "local"
            # block is then the whole system. Not just an optimisation: mpi_row_layout returns None
            # below nproc>1, so the distributed bookkeeping has nothing to work with here.
            self.solve_serial(op_flag, n, nnz_local, 1 if op_flag == 2 else 0, values, col_index,
                              row_start, b, n, 1)
            return
        try:
            self._solve_distributed(op_flag, n, nnz_local, nrow_local, first_row, values, col_index,
                                    row_start, b)
        except pyoomph_mumps.MumpsError as e:
            # No collective agreement needed for this one: INFOG is GLOBAL in MUMPS, so every rank
            # sees the same negative INFOG(1) from the same collective call and raises together.
            # src/nanobind/solver.cpp then MPI_Allreduces the failure flag and turns it into a
            # retryable NewtonSolverError everywhere.
            self._invalidate()
            raise MumpsSolverError(str(e)) from e
        except Exception:
            self._invalidate()
            raise

    def _solve_distributed(self, op_flag: int, n: int, nnz_local: int, nrow_local: int,
                           first_row: int, values: NPFloatArray, col_index: NPIntArray,
                           row_start: NPIntArray, b: NPFloatArray) -> None:
        from ..generic.mpi import (get_mpi_rank, get_mpi_any, mpi_row_layout, mpi_gather_vector,
                                   mpi_scatter_vector)
        if op_flag == 3:
            self._invalidate()
            return
        if op_flag not in (1, 2):
            raise RuntimeError("Cannot handle op_flag " + str(op_flag) + " in the distributed MUMPS solve")

        # Replicated condition, checked before the first collective so that all ranks refuse
        # together. An augmented handler's custom solve routine reaches back into the problem, which
        # is collective, from inside a callback this backend would have to drive on every rank in
        # lock-step; the custom assembler is not supported under MPI anyway.
        if self._custom_solve_routine_active():
            raise RuntimeError("An augmented assembly handler's custom solve routine cannot be used "
                               "with the distributed MUMPS solver. Run serially, or use petsc_mumps.")

        rank = get_mpi_rank()
        if op_flag == 1:
            sym = self._decide_sym()
            comm = pyoomph_mumps.comm_fortran_for(None)  # MPI_COMM_WORLD, which is what oomph uses
            # Replicated decision: _use_symmetric_factorisation_now() reads the problem's symbolic
            # symmetry, which is the same on every rank, and reuse_symbolic_factorisation is a
            # setting. Nothing here depends on a local count, so the ranks cannot disagree about
            # whether to build a new instance - and a disagreement would deadlock in MUMPS's next
            # collective phase rather than fail.
            if self._solver is None or self._active_sym != sym or not self.reuse_symbolic_factorisation:
                self._solver = self._new_solver(sym, comm)
                self._active_sym = sym
                self._structure_id = 0
            self._factorise_distributed(n, first_row, row_start, col_index, values)
            # Built after the factorisation, from one allgather, and reused for the right-hand side
            # and the solution below - they share the row split.
            self._layout = mpi_row_layout(n, first_row, nrow_local, nnz_local)
        else:
            layout = self._layout
            # Replicated: oomph gates the back-substitution on the factorisation having been
            # allocated, so op_flag==1 always precedes it and sets the layout on every rank.
            if layout is None or self._solver is None or not self._solver.has_factorisation:
                raise MumpsSolverError("A distributed back-substitution was requested without a "
                                       "preceding distributed factorisation.")
            # The only genuinely per-rank condition on this path, and therefore the only one that
            # has to be agreed on: a rank that raised here alone would leave the others waiting in
            # the gather below for a contribution that never comes. Everything else either branches
            # on replicated data or fails inside MUMPS, whose INFOG is global.
            bad = len(b) != int(layout.vec_counts[rank])
            if get_mpi_any(bad):
                raise MumpsSolverError("The row layout and the right-hand side disagree about how "
                                       "many rows a rank owns (this rank: expected " +
                                       str(int(layout.vec_counts[rank])) + ", got " + str(len(b)) +
                                       "). Raised on every rank, because at least one saw it.")
            # ICNTL(20)=0 / ICNTL(21)=0, MUMPS's defaults: the right-hand side and the solution are
            # dense and centralized on the host. The matrix - the part that actually costs - stays
            # distributed; this moves n doubles.
            b_global = mpi_gather_vector(layout, b)
            # Every rank calls solve(): the MUMPS phases are collective. Only the host passes an
            # array, which is exactly what the empty-rhs case in the binding is for.
            self._apply_transpose(1)
            self._solver.solve(b_global if rank == 0 else numpy.zeros(0, dtype=numpy.float64))
            mpi_scatter_vector(layout, b_global, b)
            # After the scatter, not before: every rank now holds its own row block of the increment
            # and can take part in the one reduction deflation needs. Doing it on rank 0 inside the
            # centralized solve would be a collective inside a single-rank branch.
            b[:] = self._postprocess_newton_step(b, first_row=int(layout.vec_displs[rank]),
                                                 reduce_dot=True)

    def _factorise_distributed(self, n: int, first_row: int, row_start: NPIntArray,
                               col_index: NPIntArray, values: NPFloatArray) -> None:
        from ..generic.mpi import get_mpi_any
        assert self._solver is not None
        structure_id = self.problem.jacobian_structure_id
        unchanged = self._solver.set_matrix_csr_distributed(n, first_row, _as_int32(row_start),
                                                            _as_int32(col_index), _as_f64(values))
        if structure_id != 0 and structure_id == self._structure_id and not unchanged:
            self._report_structure_id_mismatch("the MUMPS analysis")
        # Collective, and this is the one place it genuinely has to be: the pattern is compared per
        # rank, so one rank whose block moved while the others' did not would skip JOB=1 on every
        # rank but itself and deadlock in the next phase. Agreeing on "does ANYONE need a new
        # analysis" costs one allreduce of a bool against an analysis that costs seconds.
        need_analysis = get_mpi_any(not unchanged or not self._solver.has_analysis)
        if need_analysis:
            self._solver.analyse()
            self.n_full_factorisations += 1
        else:
            self.n_numeric_factorisations += 1
        self._grow_and_factorise()
        self._structure_id = structure_id


class MumpsInvOp(object):
    """The shift-invert operator: factorise A once, apply A^-1 on demand.

    The counterpart of PardisoInvOp (pyoomph/solvers/pardiso.py) and _SpluInvOp
    (pyoomph/solvers/spectra.py), duck-typed the same way so it can be dropped into either the
    Spectra driver or scipy's ARPACK wrapper.

    A is expected ALREADY SHIFTED - this does not form J - sigma*M itself. That is deliberate: the
    arithmetic follows the dtype of the shifted matrix, not of J and M, and those disagree in the
    ordinary case. A Hopf bifurcation has a real J and a real M and a complex sigma, so A is complex
    and the zmumps path is the common one, not the exotic one.
    """

    def __init__(self, A: DefaultMatrixType, sym: int = 0, icntl_override: dict[int, int] | None = None,
                 verbosity: int = 0, max_workspace_retries: int = 4, quiet: bool = True):
        A = A.tocsr()  # type:ignore
        cls = pyoomph_mumps.solver_for_dtype(A.dtype)  # type:ignore
        self.mat = A
        self._solver = cls(sym=sym, comm_fortran=pyoomph_mumps.USE_COMM_WORLD, verbosity=verbosity)
        for i, v in (icntl_override or {}).items():
            self._solver.set_icntl(int(i), int(v))
        self._solver.set_matrix_csr(A.shape[0], _as_int32(A.indptr), _as_int32(A.indices),  # type:ignore
                                    numpy.ascontiguousarray(A.data))  # type:ignore
        self._solver.analyse()
        # The same growth loop as the linear solver's: a shifted matrix J - sigma*M pivots at least
        # as hard as the Jacobian it came from, so the workspace prediction is at least as likely to
        # be short. Without this the Spectra driver would see a factorisation failure, nudge sigma and
        # try again - a retry that cannot help, because the shift is not what was wrong.
        self.n_workspace_growths = _factorize_growing_workspace(self._solver, max_workspace_retries,
                                                                quiet)

    def __call__(self, b: Any) -> Any:
        # A copy per matvec, because MUMPS solves in place and the caller's vector is Spectra's own
        # Arnoldi basis column. One n-vector against a back-substitution is not worth avoiding.
        x = numpy.ascontiguousarray(b, dtype=self.mat.dtype).copy()  # type:ignore
        self._solver.solve(x)
        return x

    matvec = __call__

    @property
    def shape(self) -> Any:
        return self.mat.shape  # type:ignore

    @property
    def dtype(self) -> Any:
        return self.mat.dtype  # type:ignore


# The Spectra backend is what supplies the Arnoldi iteration, and it is absent from a pyoomph built
# without Spectra (-DPYOOMPH_HAS_SPECTRA=OFF). That must not cost the LINEAR solver above, which
# needs none of it - hence the guarded import and the conditional registration rather than a plain
# import at the top of the module.
try:
    from .spectra import SpectraEigenSolver as _SpectraEigenSolver
except ImportError:
    _SpectraEigenSolver = None  # type:ignore


if _SpectraEigenSolver is not None:

    @GenericEigenSolver.register_solver()
    class MumpsSpectraEigenSolver(_SpectraEigenSolver):  # type:ignore[valid-type,misc]
        """Spectra's Arnoldi iteration with MUMPS doing the shift-and-invert factorisation.

        Everything except :py:meth:`_make_operator` is inherited, which is the point: the choice of
        sigma, the ``lambda = sigma + 1/nu`` back-transform, the ordering rules that
        get_hopf_lyapunov_coefficient relies on, the cutoff that drops a singular mass matrix's
        infinite eigenvalues, the retry ladder for a shift that landed on an eigenvalue and the dense
        fallback for a system too small for Arnoldi are all delicate and all already right in
        pyoomph/solvers/spectra.py. Reimplementing them here would only give them somewhere new to
        drift.

        The inherited ``use_pardiso`` flag has no effect on this subclass - the operator is always
        MUMPS - and is left in place rather than removed so that code configuring the Spectra
        backend generically does not have to know which subclass it has.
        """

        idname = "mumps"

        def __init__(self, problem: "Problem"):
            super().__init__(problem)
            #: ICNTL entries for the shift-invert factorisations, 1-based. ICNTL(14) is again the
            #: one to raise if a factorisation runs out of working space.
            self.icntl_override: dict[int, int] = {}
            #: 0 keeps MUMPS silent; see MumpsLinearSolver.verbose.
            self.mumps_verbosity: int = 0

        def _make_operator(self, J: DefaultMatrixType, M: DefaultMatrixType, sigma: complex,
                           use_sym: bool, quiet: bool, force_complex: bool) -> tuple[Any, bool]:
            # sigma is a python complex throughout, and multiplying a real matrix by one promotes it
            # to complex128 even when the imaginary part is zero - which would put every real problem
            # on the complex path at twice the cost. Demote it when it really is real. (Kept in step
            # with SpectraEigenSolver._make_operator, which does the same.)
            shift_scalar: float | complex = sigma.real if sigma.imag == 0.0 else sigma
            A = (J - shift_scalar * M).tocsr()  # type:ignore
            if force_complex and A.dtype.kind != "c":
                # A complex start vector cannot be handed to a real factorisation.
                A = A.astype(numpy.complex128)
            complex_op = (A.dtype.kind == "c")
            # MUMPS's sym=2 is "A = A^T", which for zmumps means complex SYMMETRIC and not Hermitian
            # - and complex symmetric is exactly what J - sigma*M is when J and M are symmetric. So
            # the verdict transfers to the complex path unchanged. (In practice it does not arise:
            # the inherited symmetry screen already requires a real sigma.)
            sym = 2 if use_sym else 0
            return MumpsInvOp(A, sym=sym, icntl_override=self.icntl_override,
                              verbosity=self.mumps_verbosity, quiet=quiet), complex_op


from ..typings import _set_public_api
_set_public_api(globals())
