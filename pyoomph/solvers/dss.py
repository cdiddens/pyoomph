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


import ctypes.util
from ctypes import CDLL
import os
import sys
import numpy
from scipy.sparse import csr_matrix
from ..typings import *
from importlib import metadata
from more_itertools import first
from ctypes import POINTER, byref, c_longlong, c_int, Structure, c_char_p,c_double
from pathlib import Path
if TYPE_CHECKING:
    from ..generic.problem import Problem

from .generic import GenericLinearSystemSolver
    
# =======================================================================

def _try_to_find_lib(nam:Union[str,List[str]])->Optional[CDLL]:
    # First try to find the library via the packages
    try:
        mkl_rt=[p for p in metadata.files('mkl') if 'mkl_rt' in str(p)]
        if len(mkl_rt)==1:
            mkl_rt=first(mkl_rt)
            res=CDLL(mkl_rt.locate())
            if res is not None:
                return res
    except:
        pass
    if isinstance(nam,list):
        for l in nam:
            res=_try_to_find_lib(l)
            if res is not None:
                return res
        return None

    res = None
    try:
        resl=CDLL(nam)
        return resl
    except:
        expname=ctypes.util.find_library(nam)
        if expname is None:
            return None
        try:
            resl=CDLL(expname)
            if resl is None:
                return None
            return resl
        except:
            res = None

    return None


if sys.platform == "linux":
    if "PYOOMPH_MKLDSS_LIB" in os.environ.keys():
        DSSlib=CDLL(os.environ["PYOOMPH_MKLDSS_LIB"])
    else:
        DSSlib=_try_to_find_lib(["libmkl_rt.so",os.path.join(Path.home(), ".local/lib/libmkl_rt.so"),"mkl_rt",os.path.join(Path.home(), ".local/lib/libmkl_rt.so.2")])
elif sys.platform == "win32":
    if "PYOOMPH_MKLDSS_LIB" in os.environ.keys():
        DSSlib=CDLL(os.environ["PYOOMPH_MKLDSS_LIB"])
    else:
        DSSlib = _try_to_find_lib(["mkl_rt.dll", "mkl_rt.1.dll","mkl_rt.2.dll", "mkl_rt"])
elif sys.platform=="darwin":
    if "PYOOMPH_MKLDSS_LIB" in os.environ.keys():
        DSSlib=CDLL(os.environ["PYOOMPH_MKLDSS_LIB"])
    else:
        DSSlib = _try_to_find_lib(["libmkl_rt.dylib", "libmkl_rt.1.dylib", "libmkl_rt.2.dylib","mkl_rt"])
else:
    raise RuntimeError("Unknown platform: "+sys.platform)

if DSSlib is None:
    raise RuntimeError("MKL DSS not found")

dss_create=DSSlib.dss_create
dss_create.argtypes = [POINTER(c_longlong),
                        POINTER(c_int)]
dss_create.restype = c_int

dss_define_structure=DSSlib.dss_define_structure
dss_define_structure.argtypes = [POINTER(c_longlong),
                                 POINTER(c_int),
                                 POINTER(c_int),
                                 POINTER(c_int),
                                 POINTER(c_int),
                                 POINTER(c_int),
                                 POINTER(c_int)]
dss_define_structure.restype = c_int

dss_reorder=DSSlib.dss_reorder
dss_reorder.argtypes = [POINTER(c_longlong),
                        POINTER(c_int),
                        POINTER(c_int)]
dss_reorder.restype = c_int

dss_factor_real=DSSlib.dss_factor_real
dss_factor_real.argtypes = [POINTER(c_longlong),
                            POINTER(c_int),
                            POINTER(c_double)]
dss_factor_real.restype = c_int

dss_solve_real=DSSlib.dss_solve_real
dss_solve_real.argtypes = [POINTER(c_longlong),
                           POINTER(c_int),
                           POINTER(c_double),
                           POINTER(c_int),
                           POINTER(c_double)]
dss_solve_real.restype = c_int

dss_delete=DSSlib.dss_delete
dss_delete.argtypes = [POINTER(c_longlong),
                       POINTER(c_int)]
dss_delete.restype = c_int

dss_statistics=DSSlib.dss_statistics
dss_statistics.argtypes = [POINTER(c_longlong),
                           POINTER(c_int),
                           ctypes.c_char_p,
                           POINTER(c_double)]
dss_statistics.restype = c_int

# ========================================================================
# MKL DSS constants: Note: These are from the C header file mkl_dss.h, but it is not sure whether they might change in future versions of MKL
MKL_DSS_SUCCESS=0
MKL_DSS_ZERO_BASED_INDEXING=131072
MKL_DSS_MSG_LVL_WARNING=	-2147483644
MKL_DSS_TERM_LVL_ERROR=  	1073741864
MKL_DSS_NON_SYMMETRIC=536871104
MKL_DSS_NON_SYMMETRIC_COMPLEX=536871296
MKL_DSS_AUTO_ORDER=268435520
MKL_DSS_INDEFINITE=134217856
MKL_DSS_REFINEMENT_OFF=4096
MKL_DSS_REFINEMENT_ON =8192

# ========================================================================

@GenericLinearSystemSolver.register_solver()
class DSSSolver(GenericLinearSystemSolver):
    idname = "dss"

    def __init__(self, problem:"Problem"):
        super().__init__(problem)
        self.dss_pt = numpy.zeros(64, numpy.int64) #type:ignore
        self.DSS_PT = self.dss_pt.ctypes.data_as(POINTER(c_longlong))
        self.iparm = numpy.zeros(64, dtype=numpy.int32) #type:ignore
        self.iparm[0]=MKL_DSS_ZERO_BASED_INDEXING+MKL_DSS_MSG_LVL_WARNING+MKL_DSS_TERM_LVL_ERROR
        self.DSS_iparam = self.iparm.ctypes.data_as(POINTER(c_int))        
        self.statistics=numpy.zeros(64,dtype=numpy.float64) #type:ignore
        if (res:=dss_create(self.DSS_PT, self.DSS_iparam))!=MKL_DSS_SUCCESS:
            raise RuntimeError("Could not create MKL DSS object: {}".format(res))



    def get_jacobian_matrix(self,n:int,values:NPFloatArray, rowind:NPIntArray, colptr:NPIntArray)->Any:
        # TODO: Really a copy here? Valgrind can report problems otherwise
        return csr_matrix((values, rowind, colptr), shape=(n, n)).copy() #type:ignore

    def get_b(self,n:int,b:NPFloatArray):
        return b

    def solve_serial(self,op_flag:int,n:int,nnz:int,nrhs:int,values:NPFloatArray,rowind:NPIntArray,colptr:NPIntArray,b:NPFloatArray,ldb:int,transpose:int)->int:
        if op_flag == 1:
#            print("INFO",len(values),len(rowind),len(colptr))
            A = self.get_jacobian_matrix(n,values, rowind, colptr)  # That is not optimal, of course
            A.sort_indices()
            Nrows=ctypes.c_int(A.shape[0])
            Ncols=ctypes.c_int(A.shape[1])
            Nnz=ctypes.c_int(A.nnz)
            self.DSS_iparam[0]=MKL_DSS_NON_SYMMETRIC
            if (res:=dss_define_structure(self.DSS_PT,self.DSS_iparam,A.indptr.ctypes.data_as(POINTER(c_int)),Nrows,Ncols,A.indices.ctypes.data_as(POINTER(c_int)),Nnz))!=MKL_DSS_SUCCESS:
                raise RuntimeError("Could not define MKL DSS structure: {}".format(res))

            self.DSS_iparam[0]=MKL_DSS_AUTO_ORDER
            myorder=numpy.zeros(A.shape[0],dtype=numpy.int32) #type:ignore
            if (res:=dss_reorder(self.DSS_PT,self.DSS_iparam,myorder.ctypes.data_as(POINTER(c_int))))!=MKL_DSS_SUCCESS:
                raise RuntimeError("Could not reorder MKL DSS structure: {}".format(res))



            self.DSS_iparam[0]=MKL_DSS_INDEFINITE
            if (res:=dss_factor_real(self.DSS_PT,self.DSS_iparam,A.data.ctypes.data_as(POINTER(c_double))))!=MKL_DSS_SUCCESS:
                raise RuntimeError("Could not factor MKL DSS structure: {}".format(res))

        elif op_flag == 2:
            self.DSS_iparam[0]=MKL_DSS_REFINEMENT_ON            
            Nrhs=ctypes.c_int(nrhs)
            sol=numpy.zeros(n,dtype=numpy.float64) #type:ignore
            if (res:=dss_solve_real(self.DSS_PT,self.DSS_iparam,b.ctypes.data_as(POINTER(c_double)),Nrhs,sol.ctypes.data_as(POINTER(c_double))))!=MKL_DSS_SUCCESS:
                raise RuntimeError("Could not solve MKL DSS system: {}".format(res))
            b[:] = sol[:]
        else:
            raise RuntimeError("Cannot handle DSS mode " + str(op_flag) + " yet")
            return 666

        return 0  # TODO: Return sign of Jacobian
        
        
    
    

