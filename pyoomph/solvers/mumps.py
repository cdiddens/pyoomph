#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2026  Christian Diddens & Duarte Rocha
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
#  The authors may be contacted at c.diddens@utwente.nl and d.rocha@utwente.nl
#
# ========================================================================
 
from .generic import GenericLinearSystemSolver
from scipy.sparse import csr_matrix #type: ignore
from mumps import DMumpsContext #type: ignore # Requires PyMUMPS: python3 -m pip install PyMUMPS
import _pyoomph

from mpi4py import MPI

from ..typings import *
if TYPE_CHECKING:
    from ..generic.problem import Problem


@GenericLinearSystemSolver.register_solver()
class MUMPSSolver(GenericLinearSystemSolver):
    idname = "mumps"

    def __init__(self, problem:"Problem"):
        super().__init__(problem)
        self.ctx=None

    def __del__(self):
        if self.ctx is not None:
            self.ctx.destroy()

    def solve_serial(self,op_flag:int,n:int,nnz:int,nrhs:int,values:NPFloatArray,rowind:NPIntArray,colptr:NPIntArray,b:NPFloatArray,ldb:int,transpose:int)->int:
        if op_flag==1:
            if self.ctx is None:
                self.ctx=DMumpsContext()
                self.ctx.set_icntl(1, -1)  #type: ignore # output stream for error msgs 
                self.ctx.set_icntl(2, -1)  #type: ignore # otuput stream for diagnostic msgs
                self.ctx.set_icntl(3, -1)  #type: ignore # output stream for global info
            if self.ctx.myid==0: #type: ignore
                A = csr_matrix((values, rowind, colptr), shape=(n, n))
                self.ctx.set_centralized_sparse(A) #type: ignore
                self.ctx.run(job=4) #type: ignore
        elif op_flag == 2:
            if self.ctx is None:
                raise RuntimeError("Should not happen")
            self.ctx.set_rhs(b) #type: ignore
            self.ctx.run(job=3) #type: ignore
        return 0

    def solve_distributed(self, op_flag: int, allow_permutations: int, n: int, nnz_local: int, nrow_local: int, first_row: int, values: NPFloatArray, col_index: NPIntArray, row_start: NPIntArray, b: NPFloatArray, nprow: int, npcol: int, doc: int, data: NPUInt64Array, info: NPIntArray) -> None:
        if op_flag==1:
            if self.ctx is None:
                self.ctx=DMumpsContext()
                self.ctx.set_icntl(1, -1)  #type: ignore # output stream for error msgs 
                self.ctx.set_icntl(2, -1)  #type: ignore # otuput stream for diagnostic msgs
                self.ctx.set_icntl(3, -1)  #type: ignore # output stream for global info
                self.ctx.set_icntl(18, 3)  #type: ignore # output stream for global info         
                         
            self.ctx.set_shape(n) #type: ignore # set the matrix size
            col_loc=col_index[:]+1 # add 1 for FORTRAN array
            row_loc=_pyoomph.csr_rows_to_coo_rows(row_start,nnz_local,first_row+1) #CSR to COO including FORTRAN offset of 1
            self.ctx.set_distributed_assembled(row_loc,col_loc,values) #type: ignore # Pass the matrix to MUMPS
            
            #self.ctx.run(job=1) #type: ignore  # Analyse
            #workspace_scaling=5 # TODO: Potentially increase workspace here
            #self.ctx.set_icntl(24, workspace_scaling*self.ctx.id.infog[25])                
            #self.ctx.run(job=2) #type:ignore # Factor
            self.ctx.run(job=4) #type:ignore # Analyse and Factor            
        elif op_flag==2:
            bglobal=self.problem._redistribute_local_to_global_double_vector(b) # RHS must be global for MUMPS
            self.ctx.set_rhs(bglobal) #type: ignore # Set the RHS (and solution )
            self.ctx.run(job=3) #type:ignore # solve by backsubs
            MPI.COMM_WORLD.Bcast(bglobal) #type:ignore # Solution is only stored at root-> spread to other procs            
            b[:]=self.problem._redistribute_global_to_local_double_vector(bglobal) # Split solution again to local vectors and pass it via b
        else:
            raise RuntimeError("Unknown mode "+str(op_flag))
        
