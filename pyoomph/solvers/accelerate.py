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

from .generic import GenericLinearSystemSolver, GenericEigenSolver, SolverError
import sys
import numpy
from scipy.sparse import csr_matrix

from ..typings import *

if TYPE_CHECKING:
    from ..generic.problem import Problem


if sys.platform!="darwin":
    raise RuntimeError("The Accelerate sparse solvers are only available on Macs")

from .. import _pyoomph_core as _pyoomph

if not hasattr(_pyoomph,"MacAccelerateSparseSolver"):
    raise RuntimeError("This pyoomph build was not compiled with Apple Accelerate support (src/mac_accelerate.cpp)")

# Method names accepted by pyoomph._pyoomph_core.MacAccelerateSparseSolver.factorize()/refactorize():
# "qr" (default; general, also handles unsymmetric square systems), "cholesky"/"ldlt"/
# "ldlt_unpivoted"/"ldlt_sbk"/"ldlt_tpp" (symmetric, square matrices only - only the upper
# triangle of the given matrix is used), "cholesky_at_a" (least-squares via A^T A).
MacAccelerateMethod:TypeAlias = Literal["qr", "cholesky", "ldlt", "ldlt_unpivoted", "ldlt_sbk", "ldlt_tpp", "cholesky_at_a"]


class AccelerateSolverError(SolverError):
    """Accelerate could not factorize the matrix: SparseMatrixIsSingular or SparseFactorizationFailed.

    Raised from C++ rather than from here -- checkStatus() in src/mac_accelerate.cpp throws a
    MacAccelerateNumericalFailure for exactly those two statuses, and the nanobind translator in
    src/nanobind/solver.cpp turns it into this class (see the SolverError docstring for what that
    buys). Accelerate's other statuses describe the call, not the matrix, and keep arriving as plain
    RuntimeErrors; so do the argument checks in this module, e.g. refactorize() before any factorize.
    """


@GenericLinearSystemSolver.register_solver()
class MacAccelerateLinearSolver(GenericLinearSystemSolver):
    idname = "accelerate"
    # Accelerate is serial, but oomph routes every mpirun through the distributed entry point, so
    # without this a plain `mpirun -n 2` could not solve at all on a Mac. The base class gathers the
    # system onto rank 0 and calls solve_serial() below there (which issues no MPI collective, as the
    # flag requires, and keeps its symbolic-factorization reuse).
    gathers_to_root_under_mpi=True
    def __init__(self, problem:"Problem", method:MacAccelerateMethod="qr"):
        super().__init__(problem)
        self.solver=_pyoomph.MacAccelerateSparseSolver()
        self.method:MacAccelerateMethod=method
        # Reuse Accelerate's symbolic factorization across solves while the Jacobian sparsity pattern is
        # unchanged. Requires problem.keep_structural_zeros; without it jacobian_structure_id is 0 and
        # this switches itself off, leaving the previous behaviour untouched.
        self.reuse_symbolic_factorization=True
        self._structure_id:int=0
        self._last_factorize_method:MacAccelerateMethod | None=None # what the cached factorization was built with

    def set_method(self,method:MacAccelerateMethod)->None:
        """Select the factorization method used on the next factorize (i.e. the next Newton/time
        step). To re-factorize the *current* Jacobian with a different method right away, without
        waiting for the next step, call refactorize() instead."""
        self.method=method

    def refactorize(self,method:MacAccelerateMethod | None=None)->None:
        """Re-run the factorization of the last-assembled Jacobian, optionally switching to a
        different method (e.g. from "qr" to "cholesky")."""
        if method is not None:
            self.method=method
        if not self.solver.is_factorized():
            raise RuntimeError("refactorize() called before any system was factorized")
        self.solver.refactorize(self.method)
        self._last_factorize_method=self.method

    def resolve(self,b:NPFloatArray)->NPFloatArray:
        """Re-solve against a new right-hand side, reusing the cached factorization."""
        return self.solver.resolve(b)

    #: Backward error above which a symmetric factorisation is discarded and redone with QR. Far
    #: above any honest round-off (a good solve gives 1e-16 to 1e-12 here) and far below what a
    #: failed one produces (7.9e14 measured), so it does not need tuning.
    symmetric_fallback_tolerance:float=1e-6

    def _check_solves(self)->bool:
        """Whether to verify each solve against the matrix it was given.

        Off by default - it costs a sparse matrix-vector product and a copy of the matrix per
        factorisation - and switched on with PYOOMPH_ACCELERATE_CHECK_SOLVE=1. It exists because this
        backend has twice now returned an answer that does not solve the system while reporting
        success: Newton went from a residual of 0.118 to inf on a linear Poisson problem, and from
        1.016 to 9.0e15 on a constrained-adaptivity one. A backward error is the one measurement that
        distinguishes "the solver is wrong" from "the matrix it was handed is wrong", and neither
        Apple's status nor the Newton residual says it.
        """
        import os
        return os.environ.get("PYOOMPH_ACCELERATE_CHECK_SOLVE","") not in ("","0","false","False")

    def _solve_and_maybe_check(self,rhs:NPFloatArray)->NPFloatArray:
        """Solve, and - once per symmetric factorisation - check that the answer solves the system.

        Apple's symmetric factorisations return a vector that does not satisfy Ax=b, with status
        SparseStatusOK, on matrices pyoomph assembles routinely. Measured on macOS arm64: an
        unpivoted LDL^T on a 1600-dof saddle point took Newton from a residual of 0.118 to inf, and
        the PIVOTED ldlt_sbk on the 357-dof constrained-adaptivity system returned a backward error
        of 7.9e14 - both silently. So the answer is verified rather than trusted, and a bad one is
        recomputed with QR, which is general and has never done this.

        The cost is one matrix-vector product per factorisation. Using QR throughout would cost far
        more, on every symmetric problem that factorises perfectly well.
        """
        import numpy
        x=self.solver.solve(rhs)
        A=getattr(self,"_checked_matrix",None)
        if A is None or not getattr(self,"_verify_next_solve",False):
            return x
        self._verify_next_solve=False
        scale=float(numpy.linalg.norm(rhs)) or 1.0
        backward_error=float(numpy.linalg.norm(A@x-rhs))/scale
        if self._check_solves():
            # The inputs as well as the error: a backward error of 1e38 on a 357-dof system is not
            # something a factorisation can produce out of finite data, so "is what we were handed
            # finite, and how big is it" has to be part of the same line, or the next question is
            # another round trip.
            finite_A=bool(numpy.isfinite(A.data).all()) if A.nnz else True
            finite_b=bool(numpy.isfinite(rhs).all())
            print("PYOOMPH_ACCELERATE_CHECK n=%d method=%s backward_error=%.3e "
                  "|A|max=%.3e finite_A=%s |b|max=%.3e finite_b=%s"
                  %(A.shape[0],self._last_factorize_method,backward_error,
                    float(numpy.abs(A.data).max()) if A.nnz else 0.0,finite_A,
                    float(numpy.abs(rhs).max()),finite_b),flush=True)
        if backward_error>self.symmetric_fallback_tolerance:
            if not getattr(self,"_warned_about_fallback",False):
                self._warned_about_fallback=True
                print("NOTE: the Accelerate '"+str(self._last_factorize_method)+"' factorisation "
                      "returned a solution with a backward error of "+("%.3e"%backward_error)+
                      " and reported success; refactorising this system with QR. Symmetric "
                      "factorisations do this on indefinite and near-singular systems, which is why "
                      "the answer is checked. Pass exploit_proven_symmetry=False to skip them "
                      "entirely.",flush=True)
            indptr,indices,data=self._last_csr
            self.solver.factorize(A.shape[0],A.shape[0],indptr,indices,data,"qr")
            self._last_factorize_method="qr"
            self._structure_id=0   # the next factorisation must not refresh a QR as if symmetric
            x=self.solver.solve(rhs)
            # Checked too, and reported: if QR is no better, the fault is not the factorisation but
            # what it was given, and that is a different investigation.
            after=float(numpy.linalg.norm(A@x-rhs))/scale
            if after>self.symmetric_fallback_tolerance:
                print("NOTE: QR did not do better (backward error %.3e); the system itself is the "
                      "problem, not the factorisation."%after,flush=True)
        self._checked_matrix=None
        return x

    def solve_serial(self,op_flag:int,n:int,nnz:int,nrhs:int,values:NPFloatArray,rowind:NPAnyIntArray,colptr:NPAnyIntArray,b:NPFloatArray,ldb:int,transpose:int)->int:
        if op_flag==1:
            A=csr_matrix((values,rowind,colptr),shape=(n,n))
            A.sort_indices()
            # indptr/indices genuinely need int32->int64 for the Accelerate C++ wrapper, so those
            # astype calls copy. A.data is already float64, so astype(...,copy=False) is a no-op that
            # avoids a redundant ~nnz*8 byte copy.
            indptr=A.indptr.astype("int64"); indices=A.indices.astype("int64")
            data=A.data.astype("float64",copy=False)
            # Skip Accelerate's symbolic factorization (the fill-reducing ordering and elimination
            # structure) whenever the sparsity pattern is unchanged since the last one -- SparseRefactor
            # recomputes only the numbers. problem.jacobian_structure_id promises the pattern is the
            # same; refactorize_values_only() verifies it against the stored indices before acting, so a
            # stale id costs a full factorization rather than a wrong answer.
            # "ldlt_sbk", not "ldlt": Apple's SparseFactorizationLDLT is the UNPIVOTED factorization,
            # which is only valid for a definite matrix, while the symbolic proof behind
            # _use_symmetric_factorisation_now() establishes symmetry and nothing more. Every saddle
            # point pyoomph assembles - an interface Lagrange multiplier, a constraint, an augmented
            # tracker - is symmetric INDEFINITE, and unpivoted LDL^T meets a zero pivot there and
            # returns nonsense without reporting failure. Measured on the coupled curved-interface
            # Poisson problem of tests/test_curved_boundaries.py: 1600 dofs, symmetric to 2.2e-16,
            # 1472 positive and 128 negative eigenvalues (one per interface multiplier); Accelerate
            # reported a successful factorisation and Newton went from a residual of 0.118 to inf in
            # one step, on a LINEAR problem. SBK is Bunch-Kaufman with supernodes, i.e. the pivoted
            # variant the symmetry proof actually justifies; "ldlt_tpp" (threshold partial pivoting)
            # is the more conservative alternative, and plain "ldlt" remains available for anyone who
            # knows their system is definite. Cholesky is still not an option - that needs SPD, which
            # the proof cannot give either.
            method:MacAccelerateMethod="ldlt_sbk" if self._use_symmetric_factorisation_now() else self.method
            structure_id=self.problem.jacobian_structure_id
            if (self.reuse_symbolic_factorization and structure_id!=0 and structure_id==self._structure_id
                    and method==self._last_factorize_method
                    and self.solver.is_factorized()
                    and self.solver.refactorize_values_only(indptr,indices,data)):
                # method must match what the factorization was built with: a QR symbolic structure
                # must not be numerically refreshed as LDLT (or vice versa, after a tracker toggled).
                return 0
            self._structure_id=structure_id
            self.solver.factorize(n,n,indptr,indices,data,method)
            self._last_factorize_method=method
            # Kept only for the check below, and only when it is switched on: holding the matrix
            # otherwise would double the memory a factorisation costs.
            # Verified on the FIRST solve after every symmetric factorisation - one matrix-vector
            # product per factorisation, not per solve, and the matrix is dropped again as soon as it
            # has been used. A bad factorisation poisons every solve that follows it, so checking the
            # first is enough to catch it.
            symmetric_now=method in ("ldlt","ldlt_unpivoted","ldlt_sbk","ldlt_tpp","cholesky")
            self._checked_matrix=A if (self._check_solves() or symmetric_now) else None
            self._verify_next_solve=bool(symmetric_now or self._check_solves())
            self._last_csr=(indptr,indices,data)
        elif op_flag==2:
            if nrhs != 1:
                raise NotImplementedError("Only single right-hand side is supported")
            # _solve_newton_step, not solve()-and-return: it is the one entry point that applies BOTH
            # of the things installed on top of a plain solve, and every other backend goes through it
            # (pardiso.py, scipy.py, petsc.py). This one applied neither, so on a Mac falling back to
            # Accelerate - Apple silicon without PETSc/MUMPS - a deflated Newton took the UNDEFLATED
            # step and an augmented handler never got the solve at all.
            #
            #   * a deflation OPERATOR rescales the increment (_postprocess_newton_step). Adding only
            #     that fixed the two tests that install one, on arm64, in the run of 29th August 2026.
            #   * a custom assembler with has_custom_solve_routine() - the augmented trackers, and the
            #     DeflationAssemblyHandler shell - is handed the solve as a CALLABLE and does its own
            #     algebra with it. That is the half this backend was still missing, which is why the
            #     two handler-based tests went on failing with a trajectory identical to before the
            #     first fix: -41.6 at the first step, the undeflated increment exactly.
            b[:] = self._solve_newton_step(lambda rhs: self._solve_and_maybe_check(rhs), b)
        else:
            raise NotImplementedError("Only transpose operation is supported")

        return 0


from .scipy import ScipyEigenSolver,DefaultMatrixType

class AccelerateInvOp(object):
    def __init__(self, A:DefaultMatrixType, M:DefaultMatrixType | None=None,sigma:float | complex | None=None,method:MacAccelerateMethod="qr"):
        if sigma is None:
            self.mat=A
        else:
            self.mat=A-sigma*M #type:ignore
        if self.mat.dtype==numpy.dtype("complex128"):
            raise RuntimeError("The Accelerate sparse solvers only support real matrices, but a complex shifted matrix was passed. Use e.g. the pardiso or scipy eigensolver for complex eigenproblems instead")
        self._solver=_pyoomph.MacAccelerateSparseSolver()
        Acsr=csr_matrix(self.mat)
        Acsr.sort_indices()
        # data is already float64 -> copy=False avoids a redundant copy; the index arrays must widen to int64.
        self._solver.factorize(Acsr.shape[0],Acsr.shape[1],Acsr.indptr.astype("int64"),Acsr.indices.astype("int64"),Acsr.data.astype("float64",copy=False),method)

    def __call__(self, b): #type:ignore
        return numpy.array(self._solver.solve(b)) #type:ignore

    matvec = __call__ #type:ignore

    @property
    def shape(s): #type:ignore
        return s.mat.shape #type:ignore

    @property
    def dtype(s): #type:ignore
        return s.mat.dtype #type:ignore


@GenericEigenSolver.register_solver()
class AccelerateArpackEigenSolver(ScipyEigenSolver):
    idname = "accelerate"
    def __init__(self,problem:"Problem",method:MacAccelerateMethod="qr"):
        super().__init__(problem)
        self.method:MacAccelerateMethod=method

    def get_OPInv(self,M:DefaultMatrixType,J:DefaultMatrixType,shift:float | complex):
        if shift is None:
            OPinv = None
        else:
            # LDLT (indefinite-safe) when ScipyEigenSolver.solve decided the pencil is proven
            # symmetric - J-shift*M is then real symmetric. Otherwise the user's configured method.
            method:MacAccelerateMethod="ldlt" if self.last_symmetry_decision else self.method
            OPinv = AccelerateInvOp(J, M, sigma=shift,method=method)
        return OPinv


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
