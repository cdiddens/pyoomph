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
 
"""
Pyoomph is a finite element framework based on oomph-lib and GiNaC. It is designed to be a high-level interface to the oomph-lib library, providing an alternative way of invoking the power of oomph-lib via just-in-time compiled equations in python instead of the C++ templates of oomph-lib. The definition of weak forms is designed to be used in a similar way to FEniCS, but in an object-oriented approach.
"""

import os
import platform
import sys
import weakref
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

from . import _pyoomph_core as _pyoomph

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
		# A weakref, not a strong reference: this singleton lives for the whole process, so a
		# strong reference here would keep whichever Problem last called solve() (and,
		# transitively, everything nb::keep_alive holds alive for it on the C++ side) alive
		# forever for any script that only ever creates a single Problem - Problem.release()
		# cannot break this cycle itself, since it isn't one: solver_cb is a perfectly reachable,
		# legitimate object, so nothing (no gc, no Problem.__del__) will ever consider the
		# Problem it points to as garbage. A weakref sidesteps the issue entirely: during an
		# actual solve, the caller still holds a strong reference to its Problem, so the weakref
		# resolves fine; once nothing else does, it simply (and correctly) starts resolving to
		# None instead of pinning the Problem alive.
		self._current_problem_ref:"Optional[weakref.ReferenceType[Problem]]"=None

	def set_problem(self,problem:Problem):
		self._current_problem_ref=weakref.ref(problem)

	def _get_current_problem(self)->Optional[Problem]:
		return self._current_problem_ref() if self._current_problem_ref is not None else None

	def solve_la_system_serial(self,op_flag:int,n:int,nnz:int,nrhs:int,values:NPFloatArray,rowind:NPIntArray,colptr:NPIntArray,b:NPFloatArray,ldb:int,transpose:int)->int:
		problem=self._get_current_problem()
		if problem is None:
			raise RuntimeError("The problem has not been set yet")
		solv=problem.get_la_solver()
		assert solv is not None
		return solv.solve_serial(op_flag,n,nnz,nrhs,values,rowind,colptr,b,ldb,transpose)


	def solve_la_system_distributed(self, op_flag: int, allow_permutations: int, n: int, nnz_local: int, nrow_local: int, first_row: int, values: NPFloatArray, col_index: NPIntArray, row_start: NPIntArray, b: NPFloatArray, nprow: int, npcol: int, doc: int, data: NPUInt64Array, info: NPIntArray) -> None:
		problem=self._get_current_problem()
		if problem is None:
			raise RuntimeError("The problem has not been set yet")
		return problem.get_la_solver().solve_distributed(op_flag,allow_permutations,n,nnz_local,nrow_local,first_row,values,col_index,row_start,b,nprow,npcol,doc,data,info) #,comm

	def metis_partgraph_kway(self, nvertex,nconnection, xadj, adjacency_vector, vwgt, nparts, options, edgecut, part):
		#print("IN PYMETIS")
		#print("nvertex",nvertex)
		#print("nconnection",nconnection)
		#print("xadj",xadj)
		#print("adjacency_vector",adjacency_vector)
		#print("vwgt",vwgt)
		#print("nparts",nparts)
		#print("options",options)
		#print("edgecut",edgecut)
		#print("part",part)
		
		try:
			import pymetis #type:ignore			
		except:
			raise ImportError("PyMetis is not installed, cannot perform graph partitioning for distributed meshes. Please install PyMetis via e.g. 'pip install pymetis'")
		adj=pymetis.CSRAdjacency(xadj, adjacency_vector) #type:ignore
		if len(vwgt)==0:
			vwgt=None
		opts=pymetis.Options()
		opts.set_defaults()
		if options[0]==0:
			opts.objtype=pymetis.ObjType.CUT
		elif options[0]==1:
			opts.objtype=pymetis.ObjType.VOL
		else:
			raise RuntimeError("ERROR: Unknown METIS option for OBJTYPE: " + str(options[0]))
		for i in range(1,len(options)):
			if options[i]!=0:
				raise RuntimeError("ERROR: METIS option " + str(i) + " is not supported")				
		print("Calling PyMetis with nparts=",nparts,"and objtype=",opts.objtype,"and vwgt=",vwgt)
		res=pymetis.part_graph(nparts,adjacency=adj,vweights=vwgt)
		part[:]=res[1] #type:ignore		
		edgecut[0]=res[0] #type:ignore
		#part[:]=numpy.arange(len(part))[:]/len(part)*nparts #type:ignore		
		return 0

solver_cb=GeneralSolverCallback()
_pyoomph.set_Solver_callback(solver_cb)

#Set best solver as default
from .solvers.generic import set_default_linear_solver,set_default_eigen_solver


def _set_accelerate_solver() -> bool:
	try:
		from .solvers import accelerate as _accelerate #type:ignore
		set_default_linear_solver("accelerate")
		set_default_eigen_solver("accelerate")
		return True
	except:
		return False


def _set_pardiso_solver() -> bool:
	try:
		from .solvers.pardiso import PardisoSolver #type:ignore
		set_default_linear_solver("pardiso")
		set_default_eigen_solver("pardiso")
		return True
	except:
		return False


def _set_petsc_mumps_solver() -> bool:
	try:
		from .solvers.petsc import PETSc,PETSCMUMPSSolver,SlepcMUMPSEigenSolver #type:ignore
		if not PETSc.Sys.hasExternalPackage("mumps"):
			return False
		set_default_linear_solver("petsc_mumps")
		set_default_eigen_solver("slepc_mumps")
		return True
	except:
		return False


def _set_superlu_fallback() -> None:
	from .solvers.scipy import SuperLUSerial,ScipyEigenSolver #type:ignore
	set_default_linear_solver("superlu")
	set_default_eigen_solver("scipy")



_is_macos = (sys.platform == "darwin")
_machine = platform.machine().lower()
_is_arm64 = _machine in ("arm64", "aarch64")

def _warn_suboptimal_solver(name:str) -> None:
	import warnings
	suggestion="PETSc/SLEPc compiled with MUMPS support" if (_is_macos and _is_arm64) else "pardiso (via Intel MKL)"
	warnings.warn(
		"pyoomph is falling back to the '"+name+"' solver, since no better solver was found. For better performance, consider "
		"installing "+suggestion+" -- see https://pyoomph.readthedocs.io/en/latest/tutorial/installation/ for "
		"instructions.",
		RuntimeWarning,
		stacklevel=2,
	)


if _is_macos and _is_arm64:
	if not _set_petsc_mumps_solver():
		if _set_accelerate_solver():
			_warn_suboptimal_solver("accelerate")
		else:
			_set_superlu_fallback()
			_warn_suboptimal_solver("superlu")
elif _is_macos:
	if not _set_pardiso_solver():
		if not _set_petsc_mumps_solver():
			if _set_accelerate_solver():
				_warn_suboptimal_solver("accelerate")
			else:
				_set_superlu_fallback()
				_warn_suboptimal_solver("superlu")
else:
	if not _set_pardiso_solver():
		if not _set_petsc_mumps_solver():
			_set_superlu_fallback()
			_warn_suboptimal_solver("superlu")
