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
 
"""
Pyoomph is a finite element framework based on oomph-lib and GiNaC. It is designed to be a high-level interface to the oomph-lib library, providing an alternative way of invoking the power of oomph-lib via just-in-time compiled equations in python instead of the C++ templates of oomph-lib. The definition of weak forms is designed to be used in a similar way to FEniCS, but in an object-oriented approach.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '4' 
os.environ['MKL_NUM_THREADS'] = '4'
# To Deactivate OpenMP parallelization
#os.environ['OPENBLAS_NUM_THREADS'] = '1' 
#os.environ['MKL_NUM_THREADS'] = '1'



from .generic import *
from .meshes import *
from .meshes.gmsh import GmshTemplate #type:ignore
from .output.meshio import MeshFileOutput #type:ignore
from .output.generic import ODEFileOutput,TextFileOutput,IntegralObservableOutput #type:ignore
from .expressions import var_and_test,var,nondim #type:ignore
from .generic.mpi import *
from .equations.generic import *
from .meshes.meshdatacache import MeshDataEigenModes #type:ignore

from .typings import *

import _pyoomph

_pyoomph.set_jit_include_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "jitbridge"))

_default_c_compiler="system"


def get_default_c_compiler():
	return _default_c_compiler


###DEVELOPMENT FLAGS, REMOVE AFTER SUCCESSFUL IMPLEMENTATION
_dev_opts:Dict[str,Any]= {}
_dev_opts["allow_tri_refine"]=False

def set_dev_option(name:str,val:Any):
	_dev_opts[name]=val

def get_dev_option(name:str)->Any:
	return _dev_opts[name]

#from .generic.ccompiler import *

#Set distutils compiler as default, otherwise intrinsic one
#du_compiler=DistUtilsCCompiler()
#set_ccompiler(du_compiler)

#if du_compiler.check_avail():
#	print("Using DISTUTILS compiler")
#	set_ccompiler(du_compiler)
#else:
#	print("No working compiler found by DISTUTILS, falling back to slow internal TinyCC compiler")
#	set_ccompiler(_pyoomph.CCompiler())


import numpy






######### Solver callback ###################


class GeneralSolverCallback(_pyoomph.GeneralSolverCallback):
	def __init__(self):
		super().__init__()
		self._current_problem=None

	def set_problem(self,problem:Problem):
#		if self._current_problem is not None:
#			if self._current_problem!=problem:
#				raise RuntimeError("Multiple problems probably not supported yet") #Make sure before each callback, the problem is updated accordingly
		self._current_problem=problem

	def solve_la_system_serial(self,op_flag:int,n:int,nnz:int,nrhs:int,values:NPFloatArray,rowind:NPIntArray,colptr:NPIntArray,b:NPFloatArray,ldb:int,transpose:int)->int:
		if self._current_problem is None:
			raise RuntimeError("The problem has not been set yet")
		solv=self._current_problem.get_la_solver()
		assert solv is not None
		return solv.solve_serial(op_flag,n,nnz,nrhs,values,rowind,colptr,b,ldb,transpose)


	def solve_la_system_distributed(self, op_flag: int, allow_permutations: int, n: int, nnz_local: int, nrow_local: int, first_row: int, values: NPFloatArray, col_index: NPIntArray, row_start: NPIntArray, b: NPFloatArray, nprow: int, npcol: int, doc: int, data: NPUInt64Array, info: NPIntArray) -> None:
		if self._current_problem is None:
			raise RuntimeError("The problem has not been set yet")
		return self._current_problem.get_la_solver().solve_distributed(op_flag,allow_permutations,n,nnz_local,nrow_local,first_row,values,col_index,row_start,b,nprow,npcol,doc,data,info) #,comm

	def metis_partgraph_kway(self, nvertex:int,xadj:NPIntArray,adjacency_vector:NPIntArray,vwgt:NPIntArray,adjwgt:NPIntArray,wgtflag:int, numflag:int,nparts:int,options:NPIntArray,edgecut:NPIntArray,part:NPIntArray):
		print("TODO: METIS")
		part[:]=numpy.arange(len(part))[:]/len(part)*nparts #type:ignore
		print("USING DEFAULT DISTRI on",nparts,"procs:",part)
#		print("IN PYMET", nvertex,xadj,adjacency_vector,vwgt,adjwgt,wgtflag, numflag,nparts,options,edgecut,part_Py)

		

solver_cb=GeneralSolverCallback()
_pyoomph.set_Solver_callback(solver_cb)

#Set pardiso as default
from .solvers.generic import set_default_linear_solver,set_default_eigen_solver
#from .solvers.pardiso import PardisoSolver
try:
	from .solvers.pardiso import PardisoSolver #type:ignore
	set_default_linear_solver("pardiso")
	set_default_eigen_solver("pardiso")
except:
	from .solvers.scipy import SuperLUSerial,ScipyEigenSolver #type:ignore
	set_default_linear_solver("superlu")
	set_default_eigen_solver("scipy")
