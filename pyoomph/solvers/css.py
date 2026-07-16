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
from numpy import ctypeslib

from .generic import GenericLinearSystemSolver
import ctypes,numpy
import os
from mpi4py import MPI
from ..typings import *
from importlib import metadata
from scipy.sparse import csr_matrix
from more_itertools import first
# ============================================================
# Low-level Loader
# ============================================================

class _MKLLoader:
    MKL_INT = ctypes.c_longlong  # Use 64-bit integers for MKL (ilp64 interface)
    def __init__(self):
        mode = ctypes.RTLD_GLOBAL | getattr(os, "RTLD_LAZY", 1)
        def get_mkl_file(name):
            candidates=[f for f in metadata.files('mkl') if name in str(f)]
            if len(candidates)==0:
                raise RuntimeError(f"Could not find {name} in mkl package files")
            if (len(candidates)>1):
                raise RuntimeError(f"Warning: multiple candidates found for {name} in mkl package files. These are the candidates: {candidates}. Please report this issue to the developer (c.diddens@utwente.nl).")
            return first(candidates).locate() 
        
        
        self.mkl_rt=ctypes.CDLL(get_mkl_file("mkl_rt"), mode=mode)
        self.mkl_core   = ctypes.CDLL(get_mkl_file("mkl_core"), mode=mode)
        self.mkl_thread = ctypes.CDLL(get_mkl_file("mkl_gnu_thread"), mode=mode) # TODO: Other threading
        self.mkl_gf     = ctypes.CDLL(get_mkl_file("mkl_gf_ilp64"), mode=mode) # TODO: Other stuff
        self.mkl_blacs  = ctypes.CDLL(get_mkl_file("mkl_blacs_openmpi_ilp64"), mode=mode) # TODO: Other MPI implementations
        self.cluster_sparse_solver = self.mkl_gf.cluster_sparse_solver


# ============================================================
# Main Wrapper Class
# ============================================================

@GenericLinearSystemSolver.register_solver()        
class ClusterSparseSolver(GenericLinearSystemSolver):
    idname = "css"
    MKL_INT = _MKLLoader.MKL_INT  
    def __init__(self, problem):
        super().__init__(problem)
        self._initialized=False
        self.loader=_MKLLoader()
        self._setup_mpi()
        self.n = self.MKL_INT()
        self.mtype = self.MKL_INT()
        self.nrhs = self.MKL_INT()

        # CSR data
        self.ia = (self.MKL_INT * len([]))(*[])
        self.ja = (self.MKL_INT * len([]))(*[])
        self.a  = (ctypes.c_double * len([]))(*[])

        # Solver state
        self.pt = (ctypes.c_void_p * 64)()
        self.iparm = (self.MKL_INT * 64)()

        self._init_iparm()

        self.maxfct = self.MKL_INT(1)
        self.mnum   = self.MKL_INT(1)
        self.msglvl = self.MKL_INT(1)

        self.error  = self.MKL_INT(0)

        self.ddum = ctypes.c_double()
        self.idum = self.MKL_INT()

        self._init_iparm()
        self._configure_signature()        
        
        self._initialized=True
        
    def get_jacobian_matrix(self,n:int,values:NPFloatArray, rowind:NPIntArray, colptr:NPIntArray)->Any:
        # TODO: Really a copy here? Valgrind can report problems otherwise
        return csr_matrix((values, rowind, colptr), shape=(n, n)).copy() #type:ignore
    
    def solve_serial(self, op_flag, n, nnz, nrhs, values, rowind, colptr, b, ldb, transpose):
        self.ctypes_dtype = ctypeslib.ndpointer(numpy.float64)
        if op_flag==1:
            self.n.value = n
            self.mtype.value = 11 # Real unsymmetric matrix
            self.nrhs.value = nrhs
            self.matrix=self.get_jacobian_matrix(n,values, rowind, colptr)
            self.matrix.sort_indices()  # Ensure indices are sorted for MKL            
            
            #print("values dtype:", self.matrix.data.dtype,self.ctypes_dtype)
            indptr=self.matrix.indptr.astype(numpy.longlong)            
            inds=self.matrix.indices.astype(numpy.longlong)
            self.ia=indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))            
            self.ja=inds.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
            self.a=self.matrix.data.ctypes.data_as(self.ctypes_dtype)
            self._factorize()
        elif op_flag==2:
            print("OP_FLAG 2: SOLVE",n,nrhs,len(b))
            
            bvv=b.ctypes.data_as(self.ctypes_dtype)
            self._solve(bvv)            
        else:
            raise NotImplementedError(f"op_flag={op_flag} not supported")
        return 0
    
    

    # --------------------------------------------------------
    # MPI Setup
    # --------------------------------------------------------

    def _setup_mpi(self):
        #mpi = MPI
        comm=MPI.COMM_WORLD

        #self.MPI_Init = mpi.MPI_Init
        self.MPI_Comm_rank = comm.Get_rank
        #self.MPI_Comm_c2f = comm.Comm_c2f
        self.MPI_Finalize = MPI.Finalize



        argc = ctypes.c_int(0)
        argv = ctypes.POINTER(ctypes.c_char_p)()

        #self.MPI_Init(ctypes.byref(argc), ctypes.byref(argv))

        self.rank = ctypes.c_int(self.MPI_Comm_rank())
        

        #self.comm = self.MKL_INT(self.MPI_Comm_c2f(self.MPI_COMM_WORLD))
        self.comm = self.MKL_INT(comm.py2f()) # self.MKL_INT(comm.Get_attr(comm.Get_attr(MPI.TAG_UB)))  # Hack to get the Fortran communicator handle

    # --------------------------------------------------------
    # Solver Setup
    # --------------------------------------------------------

    def _configure_signature(self):
        f = self.loader.cluster_sparse_solver

        f.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(None), #ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(None), # ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(None), # ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(self.MKL_INT),
        ]

    def _init_iparm(self):
        ip = self.iparm

        ip[0] = 1 # Use the default values in iparm, except for the following:
        ip[1] = 2 # Use METIS for reordering
        ip[5] = 1 # Write solution into b
        ip[7] = 2 # Max number of iterative refinement steps
        ip[9] = 13 # Pivoting perturbation, eps=10^(-13) 
        ip[10] = 0 # Scaling vectors (NOTE: default is 1 for nonsymmetric matrices)
        ip[12] = 1 # Improved accuracy using (non-)symmetric weighted matching
        ip[17] = -1 # Report the number of non-zero elements in the factors
        ip[18] = -1 # Output: Mflops for LU factorization
        ip[26] = 1 # gf ??? XXX
        ip[27] = 1 # Single precision mode?? XXX
        ip[34] = 1 # C-style indexing (0-based) for ia and ja
        ip[39] = 0 # matrix/rhs/solution stored on master

    # --------------------------------------------------------
    # Internal Call
    # --------------------------------------------------------

    def _call(self, phase, b=None, x=None):
        print("CALL WITH PHASE", phase)
        phase = self.MKL_INT(phase)

        if b is None:
            b = ctypes.byref(self.ddum)
        if x is None:
            x = ctypes.byref(self.ddum)
        
        self.loader.cluster_sparse_solver(
            self.pt,
            ctypes.byref(self.maxfct),
            ctypes.byref(self.mnum),
            ctypes.byref(self.mtype),
            ctypes.byref(phase),
            ctypes.byref(self.n),
            self.a,
            self.ia,
            self.ja,
            ctypes.byref(self.idum),
            ctypes.byref(self.nrhs),
            self.iparm,
            ctypes.byref(self.msglvl),
            b,
            x,
            ctypes.byref(self.comm),
            ctypes.byref(self.error)
        )

        if self.error.value != 0:
            raise RuntimeError(f"MKL solver error: {self.error.value}")

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def _factorize(self):
        """Phase 12"""
        self._call(12)

    def _solve(self, b):
        """Phase 33"""        
        self._call(33, b)

        return [b[i] for i in range(len(b))]

    def finalize(self):
        """Phase -1 + finalize"""
        self._call(-1)        

    def __del__(self):
        if self._initialized:            
            self.finalize()
            
            
