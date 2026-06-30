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
import sys
from scipy.sparse import csr_matrix

from ..typings import *


if sys.platform!="darwin":
    raise RuntimeError("The Accelerate QR solver is only availabe on Macs")


try:
    import pyoomph_mac_accelerate
except:
    raise RuntimeError("Please install pyoomph_mac_accelerate from https://github.com/cdiddens/pyoomph_mac_accelerate")


@GenericLinearSystemSolver.register_solver()
class MacAccelerateLinearSolver(GenericLinearSystemSolver):
    idname = "mac_accelerate"
    def __init__(self, problem):
        super().__init__(problem)
        self.solver=pyoomph_mac_accelerate.SparseLU()
        
        
    def solve_serial(self,op_flag:int,n:int,nnz:int,nrhs:int,values:NPFloatArray,rowind:NPIntArray,colptr:NPIntArray,b:NPFloatArray,ldb:int,transpose:int)->int:
        if op_flag==1:
            A=csr_matrix((values,rowind,colptr),shape=(n,n))
            self.solver.factorize(A) 
        elif op_flag==2:
            if nrhs != 1:
                raise NotImplementedError("Only single right-hand side is supported")
            x=self.solver.solve(b)
            b[:] = x[:]
        else:
            raise NotImplementedError("Only transpose operation is supported")
        
        return 0
            