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
import ctypes
import ctypes.util
from ctypes import POINTER, byref, c_longlong, c_int, Structure, c_char_p,c_double,c_bool, c_long, c_void_p
from ctypes import CDLL
import numpy,sys,os
from pathlib import Path
from numpy import ctypeslib
from ..typings import *

def _try_to_find_lib(nam:Union[str,List[str]])->Optional[CDLL]:
    # First try to find the library via the packages
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

@GenericLinearSystemSolver.register_solver()
class CKTSOLinearSolver(GenericLinearSystemSolver):
    idname = "cktso"
    def __init__(self, problem):
        super().__init__(problem)
        self._initialized=False
        
        if sys.platform == "linux":
            if "PYOOMPH_CKTSO_LIB" in os.environ.keys():
                dll=CDLL(os.environ["PYOOMPH_CKTSO_LIB"])
            else:
                dll=_try_to_find_lib(["libmkl_rt.so",os.path.join(Path.home(), ".local/lib/libcktso.so"),"mkl_rt",os.path.join(Path.home(), ".local/lib/libcktso.so.2")])
        elif sys.platform == "win32":
            if "PYOOMPH_CKTSO_LIB" in os.environ.keys():
                dll=CDLL(os.environ["PYOOMPH_CKTSO_LIB"])
            else:
                dll = _try_to_find_lib(["cktso.dll", "cktso.1.dll","cktso.2.dll", "cktso"])
        elif sys.platform=="darwin":
            if "PYOOMPH_CKTSO_LIB" in os.environ.keys():
                MKLlib=CDLL(os.environ["PYOOMPH_CKTSO_LIB"])
            else:
                MKLlib = _try_to_find_lib(["libcktso.dylib", "libcktso.1.dylib", "libcktso.2.dylib","cktso"])
        else:
            raise RuntimeError("Unknown platform: "+sys.platform)
        
        if dll is None:
            raise RuntimeError("Could not find the cktso library. Please set the environment variable PYOOMPH_CKTSO_LIB to the path of the cktso library.")
        self._create_solver=dll.CKTSO_CreateSolver
        self._create_solver.argtypes = [c_void_p,
                                POINTER(POINTER(c_int)),
                                POINTER(POINTER(c_longlong))]
        self._create_solver.restype = c_int

        self._destroy_solver=dll.CKTSO_DestroySolver
        self._destroy_solver.argtypes = [c_void_p]
        self._destroy_solver.restype = c_int

        self._analyze=dll.CKTSO_Analyze
        self._analyze.argtypes = [c_void_p, c_bool, c_int, POINTER(c_int), POINTER(c_int), POINTER(None), c_int ]
        self._analyze.restype = c_int

        self._factorize=dll.CKTSO_Factorize
        self._factorize.argtypes = [c_void_p,POINTER(None),c_bool]
        self._factorize.restype = c_int

        self._solve=dll.CKTSO_Solve
        self._solve.argtypes = [c_void_p, POINTER(None), POINTER(None),c_bool,c_bool]
        self._solve.restype = c_int


        self._solver =  numpy.zeros(64, numpy.int64).ctypes.data_as(POINTER(c_longlong))
        self._iparm = numpy.zeros(64, dtype=numpy.int32) #type:ignore
        self._oparm = numpy.zeros(64, dtype=numpy.int32) #type:ignore
        
        res=self._create_solver(byref(self._solver),byref(self._iparm.ctypes.data_as(POINTER(c_int))),byref(self._oparm.ctypes.data_as(POINTER(c_longlong))))
        if res != 0:
            raise RuntimeError(f"CKTSO_CreateSolver failed with error code {res}")
        self._initialized=True
        
    def __del__(self):
        if self._initialized:
            res=self._destroy_solver(byref(self._solver))
            if res != 0:
                raise RuntimeError(f"CKTSO_DestroySolver failed with error code {res}")
        
    def solve_serial(self,op_flag:int,n:int,nnz:int,nrhs:int,values:NPFloatArray,rowind:NPIntArray,colptr:NPIntArray,b:NPFloatArray,ldb:int,transpose:int)->int:
        if op_flag==1:
            ctypes_dtype = ctypeslib.ndpointer(ctypeslib.as_ctypes_type(values.dtype))
            res=self._analyze(self._solver,c_bool(False),c_int(n),colptr.ctypes.data_as(POINTER(c_int)),rowind.ctypes.data_as(POINTER(c_int)),values.ctypes.data_as(ctypes_dtype),c_int(-1))
            if res != 0:
                raise RuntimeError(f"CKTSO_Analyze failed with error code {res}")
            
            res=self._factorize(self._solver,values.ctypes.data_as(ctypes_dtype),c_bool(False))
            if res != 0:
                raise RuntimeError(f"CKTSO_Factorize failed with error code {res}")
        elif op_flag==2:
            if nrhs != 1:
                raise NotImplementedError("Only single right-hand side is supported")
            ctypes_dtype = ctypeslib.ndpointer(ctypeslib.as_ctypes_type(b.dtype))
            x=numpy.zeros(n,dtype=numpy.float64)
            res=self._solve(self._solver,b.ctypes.data_as(ctypes_dtype),x.ctypes.data_as(ctypes_dtype),c_bool(True),c_bool(False))
            b[:] = x[:]
        else:
            raise NotImplementedError("Only transpose operation is supported")
        
        return 0
            