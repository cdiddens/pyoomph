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
 
import ctypes.util


from ctypes import CDLL

import os
from pathlib import Path

import sys

from more_itertools import first

from ..generic.mpi import mpi_barrier,get_mpi_nproc,get_mpi_rank,get_mpi_world_comm
from ..typings import *
import numpy

from importlib import metadata

if TYPE_CHECKING:
    from ..generic.problem import Problem

def _try_to_find_lib(nam:str | list[str])->CDLL | None:
    # First try to find the library via the packages
    try:
        mkl_files=metadata.files('mkl')
        mkl_rt=[p for p in (mkl_files or []) if 'mkl_rt' in str(p)]
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
    if "PYOOMPH_PARDISO_LIB" in os.environ.keys():
        MKLlib=CDLL(os.environ["PYOOMPH_PARDISO_LIB"])
    else:
        MKLlib=_try_to_find_lib(["libmkl_rt.so",os.path.join(Path.home(), ".local/lib/libmkl_rt.so"),"mkl_rt",os.path.join(Path.home(), ".local/lib/libmkl_rt.so.2"),os.path.join(Path.home(), ".local/lib/libmkl_rt.so.3"),os.path.join(Path.home(), ".local/lib/libmkl_rt.so.4")])
elif sys.platform == "win32":
    if "PYOOMPH_PARDISO_LIB" in os.environ.keys():
        MKLlib=CDLL(os.environ["PYOOMPH_PARDISO_LIB"])
    else:
        MKLlib = _try_to_find_lib(["mkl_rt.dll", "mkl_rt.1.dll","mkl_rt.2.dll","mkl_rt.3.dll","mkl_rt.4.dll", "mkl_rt"])
elif sys.platform=="darwin":
    if "PYOOMPH_PARDISO_LIB" in os.environ.keys():
        MKLlib=CDLL(os.environ["PYOOMPH_PARDISO_LIB"])
    else:
        MKLlib = _try_to_find_lib(["libmkl_rt.dylib", "libmkl_rt.1.dylib", "libmkl_rt.2.dylib","libmkl_rt.3.dylib","libmkl_rt.4.dylib", "mkl_rt"])
else:
    raise RuntimeError("Unknown platform: "+sys.platform)

if MKLlib is None:
    raise RuntimeError("Pardiso not found")

from builtins import object

# from pyMKL import pardisoinit, pardiso, mkl_get_version
from ctypes import POINTER, byref, c_longlong, c_int, Structure, c_char_p
import numpy as np
import scipy.sparse as sp #type:ignore
from numpy import ctypeslib

from .generic import GenericLinearSystemSolver, GenericEigenSolver

####

pardisoinit = MKLlib.pardisoinit

pardisoinit.argtypes = [POINTER(c_longlong),
                        POINTER(c_int),
                        POINTER(c_int)]
pardisoinit.restype = None

feastinit = MKLlib.feastinit
# Sparse interfaces
# Real general
# {p}d{i}feast_gcsr{ev,gv}{x}
# where gv: generalized
# i inexact iterative
# p parallel

# feastcall=MKLlib.feast_gcsr
# print(dir(MKLlib))
# print()
# exit()


pardiso = MKLlib.pardiso

pardiso.argtypes = [POINTER(c_longlong),  # pt
                    POINTER(c_int),  # maxfct
                    POINTER(c_int),  # mnum
                    POINTER(c_int),  # mtype
                    POINTER(c_int),  # phase
                    POINTER(c_int),  # n
                    POINTER(None),  #type:ignore # a
                    POINTER(c_int),  # ia
                    POINTER(c_int),  # ja
                    POINTER(c_int),  # perm
                    POINTER(c_int),  # nrhs
                    POINTER(c_int),  # iparm
                    POINTER(c_int),  # msglvl
                    POINTER(None),  #type:ignore # b
                    POINTER(None),  #type:ignore # x
                    POINTER(c_int)]  # error)
pardiso.restype = None


class pyMKLVersion(Structure):
    _fields_ = [('MajorVersion', c_int),
                ('MinorVersion', c_int),
                ('UpdateVersion', c_int),
                ('ProductStatus', c_char_p),
                ('Build', c_char_p),
                ('Processor', c_char_p),
                ('Platform', c_char_p)]


_mkl_get_version = MKLlib.mkl_get_version
_mkl_get_version.argtypes = [POINTER(pyMKLVersion)]
_mkl_get_version.restype = None


def mkl_get_version():
    MKLVersion = pyMKLVersion()
    _mkl_get_version(MKLVersion)
    version = {'MajorVersion': MKLVersion.MajorVersion,
               'MinorVersion': MKLVersion.MinorVersion,
               'UpdateVersion': MKLVersion.UpdateVersion,
               'ProductStatus': MKLVersion.ProductStatus,
               'Build': MKLVersion.Build,
               'Platform': MKLVersion.Platform}

    versionString = 'Intel(R) Math Kernel Library Version {MajorVersion}.{MinorVersion}.{UpdateVersion} {ProductStatus} Build {Build} for {Platform} applications'.format(
        **version)

    return versionString


_mkl_get_max_threads = MKLlib.mkl_get_max_threads
_mkl_get_max_threads.argtypes = None #type:ignore
_mkl_get_max_threads.restype = c_int


def mkl_get_max_threads():
    max_threads = _mkl_get_max_threads()
    return max_threads


_mkl_set_num_threads = MKLlib.mkl_set_num_threads
_mkl_set_num_threads.argtypes = [POINTER(c_int)]
_mkl_set_num_threads.restype = None


def mkl_set_num_threads(num_threads:int):
    _mkl_set_num_threads(c_int(num_threads))


class pardisoSolver(object):
    
    def __init__(self, matA:Any, mtype:int=11, verbose:bool=False,iparm_override:dict[int,int]={}):
            #mode  11 : real, nonsymmetric
            #mode  13 : complex,  nonsymmetric

        self.mtype = mtype
        if mtype in [1, 3]:
            msg = "mtype = 1,3 not implemented yet."
            raise NotImplementedError(msg)
        elif mtype in [2, -2, 4, -4, 6, 11, 13]:
            pass
        else:
            msg = "Invalid mtype: mtype={}".format(mtype)
            raise ValueError(msg)

        self.n = matA.shape[0]

        if mtype in [4, -4, 6, 13]:
            # Complex matrix
            self.dtype = np.complex128
        elif mtype in [2, -2, 11]:
            # Real matrix
            self.dtype = np.float64
        self.ctypes_dtype = ctypeslib.ndpointer(self.dtype)

        if mtype in [2, -2, 4, -4, 6]:
            matA = sp.triu(matA, format='csr') #type:ignore
        elif mtype in [11, 13]:
            matA = matA.tocsr()

        if not matA.has_sorted_indices:
            matA.sort_indices()

        self.a = matA.data
        self.ia = matA.indptr
        self.ja = matA.indices

        self._MKL_a = self.a.ctypes.data_as(self.ctypes_dtype)
        self._MKL_ia = self.ia.ctypes.data_as(POINTER(c_int))
        self._MKL_ja = self.ja.ctypes.data_as(POINTER(c_int))

        self.maxfct = 1
        self.mnum = 1
        self.perm = 0

        if verbose:
            self.msglvl = 1
        else:
            self.msglvl = 0

        self.pt = np.zeros(64, np.int64) #type:ignore
        self._MKL_pt = self.pt.ctypes.data_as(POINTER(c_longlong))

        self.iparm = np.zeros(64, dtype=np.int32) #type:ignore
        self._MKL_iparm = self.iparm.ctypes.data_as(POINTER(c_int))

        # Init call
        pardisoinit(self._MKL_pt, byref(c_int(self.mtype)), self._MKL_iparm)

        verstring = mkl_get_version()

        if '11.3.3' in verstring:
            self.iparm[1] = 0
        else:
            self.iparm[1] = 3  
        self.iparm[23] = 1  
        self.iparm[34] = 1
          
        for k,v in iparm_override.items():
            self.iparm[k-1]=v

        self.last_mem_used_in_kb:int | None=None

    def update_matrix_values(self, matA:Any,mtype:int=11):
        if self.n != matA.shape[0]:
            return False
        if self.mtype != mtype:
            return False
        
        if len(matA.data) != len(self.a):
            return False
        if len(matA.indptr) != len(self.ia):
            return False
        if len(matA.indices) != len(self.ja):  
            return False
        
        if not matA.has_sorted_indices:
            matA.sort_indices()

        self.a[:] = matA.data[:]
        self.ia[:]=matA.indptr[:]
        self.ja[:]=matA.indices[:]
        
        self._MKL_a = self.a.ctypes.data_as(self.ctypes_dtype)
        self._MKL_ia = self.ia.ctypes.data_as(POINTER(c_int))
        self._MKL_ja = self.ja.ctypes.data_as(POINTER(c_int))
        return True

    def clear(self):        
        self.run_pardiso(phase=-1)

    def __del__(self, _is_finalizing:Any=sys.is_finalizing):
        # Skip once the interpreter itself is shutting down: this instance may only be
        # reachable this late because the owning Problem never called .release()/used
        # "with" (see project_nanobind_migration memory) - by that point module globals
        # this depends on (e.g. numpy as np, imported above) may already be cleared to
        # None, which would otherwise surface as a harmless but scary "Exception ignored
        # in __del__" (AttributeError: 'NoneType' object has no attribute 'zeros').
        # Freeing MKL's internal memory here is unnecessary anyway - the OS reclaims it
        # on process exit regardless.
        # _is_finalizing is bound as a default arg (evaluated once, at function-definition
        # time) rather than looked up as "sys.is_finalizing" at call time: CPython clears
        # nearly all module-level globals to None during interpreter shutdown, including
        # this module's own "sys" name - a bare "sys.is_finalizing()" call here can itself
        # raise "'NoneType' object has no attribute 'is_finalizing'" depending on shutdown
        # GC ordering (confirmed empirically). A default-arg value lives in the function
        # object itself, not the module dict, so it survives regardless.
        if _is_finalizing():
            return
        self.clear()

    def factor(self):
        #print("PARDISO FACTOR")
        out = self.run_pardiso(phase=12) #type:ignore
    
    def refactor(self):
        out = self.run_pardiso(phase=23) #type:ignore

    def solve(self, rhs:NPFloatArray | NPComplexArray)->NPFloatArray | NPComplexArray:
        #print("PARDISO SOLVE")
        x = self.run_pardiso(phase=33, rhs=rhs)
        return x

    def run_pardiso(self, phase:int, rhs:NPFloatArray | NPComplexArray | None=None)->NPFloatArray | NPComplexArray:
        
        if rhs is None:
            nrhs = 0
            x = np.zeros(1) #type:ignore
            rhs = np.zeros(1) #type:ignore
        else:
            if rhs.ndim == 1:
                nrhs = 1
            elif rhs.ndim == 2:
                nrhs = rhs.shape[1]
            else:
                msg = "Can only solve for 1 or 2 RHS"
                raise NotImplementedError(msg)
            rhs = rhs.astype(self.dtype).flatten(order='f') #type:ignore
            x = np.zeros(nrhs * self.n, dtype=self.dtype) #type:ignore

        MKL_rhs = rhs.ctypes.data_as(self.ctypes_dtype) #type:ignore
        MKL_x = x.ctypes.data_as(self.ctypes_dtype)
        ERR = 0

        pardiso(self._MKL_pt,  # pt
                byref(c_int(self.maxfct)),  # maxfct
                byref(c_int(self.mnum)),  # mnum
                byref(c_int(self.mtype)),  # mtype
                byref(c_int(phase)),  # phase
                byref(c_int(self.n)),  # n
                self._MKL_a,  # a
                self._MKL_ia,  # ia
                self._MKL_ja,  # ja
                byref(c_int(self.perm)),  # perm
                byref(c_int(nrhs)),  # nrhs
                self._MKL_iparm,  # iparm
                byref(c_int(self.msglvl)),  # msglvl
                MKL_rhs,  # b
                MKL_x,  # x
                byref(c_int(ERR)))  # error
        
        if ERR!=0:
            print("ERROR IN PARDISO",ERR)

        if self._MKL_iparm[14]!=0 or self._MKL_iparm[15]!=0 or self._MKL_iparm[16]!=0:
            self.last_mem_used_in_kb=max(self._MKL_iparm[14],self._MKL_iparm[15]+self._MKL_iparm[16])

        if nrhs > 1:
            x = x.reshape((self.n, nrhs), order='f') #type:ignore
        return x #type:ignore


from scipy.sparse import  csr_matrix #type:ignore


@GenericLinearSystemSolver.register_solver()
class PardisoSolver(GenericLinearSystemSolver):
    idname = "pardiso"

    def __init__(self, problem:"Problem",verbose:bool=False):
        super().__init__(problem)
        self._current_pardiso = None
        self.try_to_reuse_solver=False
        self.verbose=verbose
        self.iparm_override:dict[int,int]={}

    def set_num_threads(self,nthreads:int | None):
        if nthreads is None or nthreads==0:
            mkl_set_num_threads(mkl_get_max_threads())
        else:
            mkl_set_num_threads(nthreads)


    def get_last_used_mem_size_in_kb(self):
        if self._current_pardiso is None:
            return 0
        elif self._current_pardiso.last_mem_used_in_kb is None:
            return 0
        else:
            return self._current_pardiso.last_mem_used_in_kb

            

    def get_jacobian_matrix(self,n:int,values:NPFloatArray, rowind:NPIntArray, colptr:NPIntArray)->Any:
        # TODO: Really a copy here? Valgrind can report problems otherwise
        return csr_matrix((values, rowind, colptr), shape=(n, n)).copy() #type:ignore

    def get_b(self,n:int,b:NPFloatArray):
        return b

    def solve_serial(self,op_flag:int,n:int,nnz:int,nrhs:int,values:NPFloatArray,rowind:NPIntArray,colptr:NPIntArray,b:NPFloatArray,ldb:int,transpose:int)->int:
        #print("CALL WITH OP FLAG ",op_flag,ldb,transpose)
        #print("PARDISO ", op_flag)
        if op_flag == 1:
#            print("INFO",len(values),len(rowind),len(colptr))
            A = self.get_jacobian_matrix(n,values, rowind, colptr)  # That is not optimal, of course
            mode = 11
            if self.try_to_reuse_solver:
                self._lastA=A
                if self._current_pardiso is None:    
                    self._current_pardiso = pardisoSolver(A, mtype=mode, verbose=self.verbose,iparm_override=self.iparm_override)
                    print("CREATED NEW PARDISO AND FACTOR")
                    self._current_pardiso.factor()
                else:
                    if not self._current_pardiso.update_matrix_values(A):
                        self._current_pardiso.clear()  # TODO: Only if matrix is entirely changed                
                        self._current_pardiso = pardisoSolver(A, mtype=mode, verbose=self.verbose,iparm_override=self.iparm_override)
                        print("CREATED NEW PARDISO AND FACTOR")
                        self._current_pardiso.factor()                    
                        
                   
            else:
                if self._current_pardiso:
                    self._current_pardiso.clear()  # TODO: Only if matrix is entirely changed                
                self._current_pardiso = pardisoSolver(A, mtype=mode, verbose=self.verbose,iparm_override=self.iparm_override)
                self._current_pardiso.factor()
                if self.verbose:
                    print("PARDISO FACTOR IPARM",self._current_pardiso.iparm)                
        elif op_flag == 2:
            self.setup_solver()
            assert self._current_pardiso is not None
            if self.try_to_reuse_solver:
                maxiters=30
                self._current_pardiso.iparm[7]=maxiters
                #self._current_pardiso.iparm[8]=1
                
                self._current_pardiso.iparm[3] = 63
                bv=self.get_b(n,b)
                if self.problem._custom_assembler is not None and self.problem._custom_assembler.has_custom_solve_routine():
                    raise NotImplementedError("Custom solve not implemented for this case")
                sol=self._current_pardiso.solve(bv)
                #self._current_pardiso.iparm[3] = 0
                err=numpy.amax(numpy.absolute(self._lastA*sol-bv))
                if self._current_pardiso.iparm[6]==maxiters or err>1e-10:
                    print("MUST RECOMPUTE FACTORIZATION","ITER",self._current_pardiso.iparm[6],"ERR",err)
                    if self._current_pardiso:
                        self._current_pardiso.clear() 
                    mode=11
                    self._current_pardiso = pardisoSolver(self._lastA, mtype=mode, verbose=self.verbose,iparm_override=self.iparm_override)
                    self._current_pardiso.factor()
                    sol=self._current_pardiso.solve(bv)
                else:
                    print("REUSE PARDISO AND REFACTOR DONE, ERROR",err,"IN ",self._current_pardiso.iparm[6],"ITERATIONS")
                b[:]=sol[:]
            else:
                if self.problem._custom_assembler is not None and self.problem._custom_assembler.has_custom_solve_routine():
                    pd = self._current_pardiso
                    sol=self.problem._custom_assembler.custom_solve_routine(lambda rhs : pd.solve(rhs), b) #type:ignore
                else:
                    sol = self._current_pardiso.solve(self.get_b(n,b))
            if self.verbose:
                print("PARDISO SOLVE IPARM",self._current_pardiso.iparm)
            b[:] = sol[:]
        else:
            raise RuntimeError("Cannot handle Pardiso mode " + str(op_flag) + " yet")
            return 666

        return 0  # TODO: Return sign of Jacobian

    def solve_distributed(self, op_flag: int, allow_permutations: int, n: int, nnz_local: int, nrow_local: int, first_row: int, values: NPFloatArray, col_index: NPIntArray, row_start: NPIntArray, b: NPFloatArray, nprow: int, npcol: int, doc: int, data: NPUInt64Array, info: NPIntArray)->None:        
        # NOTE: This does not solve the system via MPI Pardiso. Instead it solves it on the root process and scatters the solution. This is not optimal, but MKL Pardiso is not MPI parallel. MKL cluster_sparse_solver is, but this must be accessed via PETSc using mkl_cpardiso
        from mpi4py import MPI
        rank=get_mpi_rank()
        nproc=get_mpi_nproc()
        if op_flag==1:
            global_col_index = col_index 
            rows = numpy.empty(len(col_index), dtype=np.int64)
            for i in range(nrow_local):
                rows[row_start[i]:row_start[i + 1]] = first_row + i            
            nnz_local = len(data)
            cols = global_col_index
            data_values = values
            comm=get_mpi_world_comm()
            assert comm is not None
            #all_nnz = comm.gather(nnz_local, root=0)
            all_rows = comm.gather(rows, root=0)
            all_cols = comm.gather(cols, root=0)
            all_data = comm.gather(data_values, root=0)

            if rank==0:
                assert all_rows is not None and all_cols is not None and all_data is not None
                global_rows = np.concatenate(all_rows)
                global_cols = np.concatenate(all_cols)
                global_data = np.concatenate(all_data)
                #assert isinstance(A,csr_matrix)
                A = csr_matrix((global_data, (global_rows,global_cols)),shape=(n, n))
                A.eliminate_zeros()
                A.sort_indices()                  
                if self._current_pardiso:
                    self._current_pardiso.clear()  # TODO: Only if matrix is entirely changed
                mode = 11                
                self._current_pardiso = pardisoSolver(A, mtype=mode, verbose=False)                
                self._current_pardiso.factor()

                
            mpi_barrier()            
        elif op_flag==2:
            comm=get_mpi_world_comm()
            assert comm is not None
            counts = comm.gather(len(b), root=0)
            if rank == 0:
                counts = np.array(counts, dtype=np.int32)    
                displs = np.zeros(len(counts), dtype=np.int32)
                displs[1:] = np.cumsum(counts[:-1])
#                x_global = sol.copy()
                b_global = np.empty(n, dtype=b.dtype)
            else:
                displs = None
                b_global = None
            
            comm.Gatherv(sendbuf=b,recvbuf=[b_global, counts, displs, MPI.DOUBLE],root=0)

            sol:NPFloatArray | NPComplexArray | None = None
            if rank==0:
                self.setup_solver()
                assert self._current_pardiso is not None
                assert b_global is not None
                pd = self._current_pardiso
                if self.try_to_reuse_solver:
                    raise NotImplementedError("try_to_reuse_solver not implemented yet when running with MPI")
                if self.problem._custom_assembler is not None and self.problem._custom_assembler.has_custom_solve_routine():
                    sol=self.problem._custom_assembler.custom_solve_routine(lambda rhs : pd.solve(rhs), b) #type:ignore
                else:
                    sol = self._current_pardiso.solve(self.get_b(n,b_global))

            if rank == 0:
                assert sol is not None
                counts = np.array(counts, dtype=np.int32)
                displs = np.zeros(len(counts), dtype=np.int32)
                displs[1:] = np.cumsum(counts[:-1])
                x_global = sol.copy()
            else:
                displs = None
                x_global = None
            #print("GATHERV SOLUTION",displs,counts)
            x_local = np.empty(len(b), dtype=np.float64)
            
            comm.Scatterv([x_global, counts, displs, MPI.DOUBLE],x_local,root=0)
            b[:] = x_local[:]                
            mpi_barrier()            
        else:
            raise RuntimeError("Not implemented")

from .scipy import ScipyEigenSolver,DefaultMatrixType

class PardisoInvOp(object):
    def __init__(self, A:DefaultMatrixType, M:DefaultMatrixType | None=None,sigma:float | complex | None=None,mode:int=11):
        if sigma is None:
            self.mat=A
        else:
            self.mat=A-sigma*M #type:ignore
        self._current_pardiso=pardisoSolver(self.mat, mtype=mode, verbose=False) #type:ignore
        self._current_pardiso.factor()


    def __call__(self, b): #type:ignore
        x = self._current_pardiso.solve(b) #type:ignore
        return x

    matvec  = __call__  #type:ignore # ? 

    @property
    def shape(s): #type:ignore
        return s.mat.shape #type:ignore

    @property
    def dtype(s): #type:ignore
        return s.mat.dtype #type:ignore


@GenericEigenSolver.register_solver()
class PardisoArpackEigenSolver(ScipyEigenSolver):
    idname = "pardiso"

    def get_OPInv(self,M:DefaultMatrixType,J:DefaultMatrixType,shift:float | complex):
        if shift is None:
            OPinv = None
        else:
            mode=11
            if M.dtype==numpy.dtype("complex128") or J.dtype==numpy.dtype("complex128"):
                mode=13
            OPinv = PardisoInvOp(J, M, sigma=shift,mode=mode)
        return OPinv


