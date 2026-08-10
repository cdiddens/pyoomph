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

    def resolve(self,b:NPFloatArray)->NPFloatArray:
        """Re-solve against a new right-hand side, reusing the cached factorization."""
        return self.solver.resolve(b)

    def solve_serial(self,op_flag:int,n:int,nnz:int,nrhs:int,values:NPFloatArray,rowind:NPIntArray,colptr:NPIntArray,b:NPFloatArray,ldb:int,transpose:int)->int:
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
            structure_id=self.problem.jacobian_structure_id
            if (self.reuse_symbolic_factorization and structure_id!=0 and structure_id==self._structure_id
                    and self.solver.is_factorized()
                    and self.solver.refactorize_values_only(indptr,indices,data)):
                return 0
            self._structure_id=structure_id
            self.solver.factorize(n,n,indptr,indices,data,self.method)
        elif op_flag==2:
            if nrhs != 1:
                raise NotImplementedError("Only single right-hand side is supported")
            x=self.solver.solve(b)
            b[:] = x[:]
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
            OPinv = AccelerateInvOp(J, M, sigma=shift,method=self.method)
        return OPinv
