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

from .generic import GenericLinearSystemSolver
import sys
from scipy.sparse import csr_matrix

from ..typings import *

if TYPE_CHECKING:
    from ..generic.problem import Problem


if sys.platform!="darwin":
    raise RuntimeError("The Accelerate sparse solvers are only available on Macs")

import pyoomph._pyoomph_core as _pyoomph

if not hasattr(_pyoomph,"MacAccelerateSparseSolver"):
    raise RuntimeError("This pyoomph build was not compiled with Apple Accelerate support (src/mac_accelerate.cpp)")

# Method names accepted by pyoomph._pyoomph_core.MacAccelerateSparseSolver.factorize()/refactorize():
# "qr" (default; general, also handles unsymmetric square systems), "cholesky"/"ldlt"/
# "ldlt_unpivoted"/"ldlt_sbk"/"ldlt_tpp" (symmetric, square matrices only - only the upper
# triangle of the given matrix is used), "cholesky_at_a" (least-squares via A^T A).
MacAccelerateMethod = Literal["qr", "cholesky", "ldlt", "ldlt_unpivoted", "ldlt_sbk", "ldlt_tpp", "cholesky_at_a"]


@GenericLinearSystemSolver.register_solver()
class MacAccelerateLinearSolver(GenericLinearSystemSolver):
    idname = "accelerate"
    def __init__(self, problem:"Problem", method:MacAccelerateMethod="qr"):
        super().__init__(problem)
        self.solver=_pyoomph.MacAccelerateSparseSolver()
        self.method:MacAccelerateMethod=method

    def set_method(self,method:MacAccelerateMethod)->None:
        """Select the factorization method used on the next factorize (i.e. the next Newton/time
        step). To re-factorize the *current* Jacobian with a different method right away, without
        waiting for the next step, call refactorize() instead."""
        self.method=method

    def refactorize(self,method:Optional[MacAccelerateMethod]=None)->None:
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
            self.solver.factorize(n,n,A.indptr.astype("int64"),A.indices.astype("int64"),A.data.astype("float64"),self.method)
        elif op_flag==2:
            if nrhs != 1:
                raise NotImplementedError("Only single right-hand side is supported")
            x=self.solver.solve(b)
            b[:] = x[:]
        else:
            raise NotImplementedError("Only transpose operation is supported")

        return 0
