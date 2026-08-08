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
 
"""
Pyoomph is a finite element framework based on oomph-lib and GiNaC. It is designed to be a high-level interface to the oomph-lib library, providing an alternative way of invoking the power of oomph-lib via just-in-time compiled equations in python instead of the C++ templates of oomph-lib. The definition of weak forms is designed to be used in a similar way to FEniCS, but in an object-oriented approach.
"""

import os
import platform
import sys
import weakref
os.environ.setdefault('OPENBLAS_NUM_THREADS', os.environ.get('PYOOMPH_OPENBLAS_NUM_THREADS', '4'))
os.environ.setdefault('MKL_NUM_THREADS', os.environ.get('PYOOMPH_MKL_NUM_THREADS', '4'))
# To Deactivate OpenMP parallelization, set PYOOMPH_OPENBLAS_NUM_THREADS=1 and PYOOMPH_MKL_NUM_THREADS=1



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

_default_c_compiler:Literal["tcc","system"]="system"


def get_default_c_compiler():
	return _default_c_compiler


###DEVELOPMENT FLAGS, REMOVE AFTER SUCCESSFUL IMPLEMENTATION
_dev_opts:dict[str,Any]= {}
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
		self._current_problem_ref:"weakref.ReferenceType[Problem] | None"=None

	def set_problem(self,problem:Problem):
		self._current_problem_ref=weakref.ref(problem)

	def _get_current_problem(self)->Problem | None:
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
		from .generic.mpi import get_mpi_nproc as _get_mpi_nproc, get_mpi_max as _get_mpi_max, get_mpi_min as _get_mpi_min #type:ignore
		if _get_mpi_nproc()>1:
			# n is the GLOBAL system size, so it must agree on every rank, distributed or not. When it
			# does not, PETSc fails deep inside the preallocation with "Row too large" - which says
			# nothing about the actual cause, namely that the ranks built different problems. That
			# happens whenever anything in the setup is drawn per rank rather than shared: an unshared
			# random initial condition makes the ranks solve slightly different problems, which makes
			# their spatial error estimates differ, which makes them refine differently. Every rank is
			# in this call, so the reduction below is safe and they all fail together.
			nmin,nmax=_get_mpi_min(n),_get_mpi_max(n)
			if nmin!=nmax:
				raise RuntimeError("The MPI ranks disagree about the size of the linear system ("+str(n)+" on this one, between "+str(nmin)+" and "+str(nmax)+" over all of them). They are solving different problems, which means the setup is not identical on all ranks - typically a random initial condition or mesh refinement criterion that was drawn per rank instead of being shared (see DeterministicRandomField), or a quantity read from a per-rank file.")
		return problem.get_la_solver().solve_distributed(op_flag,allow_permutations,n,nnz_local,nrow_local,first_row,values,col_index,row_start,b,nprow,npcol,doc,data,info) #,comm

	def metis_partgraph_kway(self, nvertex,nconnection, xadj, adjacency_vector, vwgt, nparts, options, edgecut, part):
		# oomph-lib partitions on rank 0 only and scatters the result, so all other ranks are already
		# sitting in the collective that follows this call. An exception escaping here would unwind
		# rank 0 alone and leave the rest of the job hanging forever (a missing pymetis used to do
		# exactly that). There is no way to hand the failure over to the other ranks from inside a
		# call they never make, so report it and take the whole job down instead.
		try:
			return self._metis_partgraph_kway(nvertex,nconnection, xadj, adjacency_vector, vwgt, nparts, options, edgecut, part)
		except BaseException:
			from .generic.mpi import get_mpi_nproc as _get_mpi_nproc, get_mpi_rank as _get_mpi_rank, mpi_abort as _mpi_abort #type:ignore
			if _get_mpi_nproc()<=1:
				raise  # serial run: nobody is waiting, a normal traceback is the friendlier failure
			import traceback
			traceback.print_exc()
			sys.stderr.write("pyoomph: mesh partitioning failed on rank "+str(_get_mpi_rank())+" (see the traceback above); aborting all MPI processes.\n")
			sys.stderr.flush()
			_mpi_abort(1)
			raise # not reached, Abort() does not return

	def _metis_partgraph_kway(self, nvertex,nconnection, xadj, adjacency_vector, vwgt, nparts, options, edgecut, part):
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
		except ImportError:
			from .generic.mpi import PYMETIS_MISSING_MESSAGE #type:ignore
			raise ImportError(PYMETIS_MISSING_MESSAGE)
		adj=pymetis.CSRAdjacency(xadj, adjacency_vector) #type:ignore
		# nanobind converts the C++-side null vwgt pointer (no vertex weights, the common case)
		# to None rather than an empty array, unlike pybind11 before it.
		if vwgt is None or len(vwgt)==0:
			vwgt=None
		opts=pymetis.Options()
		opts.set_defaults() #type:ignore
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
# set_default_eigen_solver is not used here anymore (the eigensolver default is resolved lazily, see
# below), but stays imported: it has been reachable as pyoomph.set_default_eigen_solver for a long
# time and user scripts call it that way.
from .solvers.generic import CoreEigenSolverEnum,set_default_linear_solver,set_default_eigen_solver,set_default_eigen_solver_resolver


# Availability probes. Linear solver and eigensolver are picked by separate cascades below (the best
# linear solver on a machine is not necessarily the backend of the best eigensolver), so these only
# answer "is it there", they do not select anything.
def _have_accelerate() -> bool:
	try:
		from .solvers import accelerate as _accelerate #type:ignore
		return True
	except:
		return False


def _have_pardiso() -> bool:
	try:
		from .solvers.pardiso import PardisoSolver #type:ignore
		return True
	except:
		return False


def _have_petsc_mumps() -> bool:
	try:
		from .solvers.petsc import PETSc,PETSCMUMPSSolver,SlepcMUMPSEigenSolver #type:ignore
		return bool(PETSc.Sys.hasExternalPackage("mumps")) #type:ignore
	except:
		return False


def _set_accelerate_linear_solver() -> bool:
	if not _have_accelerate():
		return False
	set_default_linear_solver("accelerate")
	return True


def _set_pardiso_linear_solver() -> bool:
	if not _have_pardiso():
		return False
	set_default_linear_solver("pardiso")
	return True


def _set_petsc_mumps_linear_solver() -> bool:
	if not _have_petsc_mumps():
		return False
	set_default_linear_solver("petsc_mumps")
	return True


def _set_superlu_linear_fallback() -> None:
	from .solvers.scipy import SuperLUSerial #type:ignore
	set_default_linear_solver("superlu")



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


def _running_under_mpi() -> bool:
	# True only for a genuine multi-process run (mpirun -n N with N>1). get_mpi_nproc() returns 0
	# when pyoomph was built without MPI, and 1 for a single-process run -- both mean "serial".
	try:
		from .generic.mpi import get_mpi_nproc #type:ignore
		return get_mpi_nproc() > 1
	except Exception:
		return False


def _warn_no_mpi_capable_solver(name:str) -> None:
	import warnings
	warnings.warn(
		"pyoomph is running with multiple MPI processes, but no MPI-capable direct solver was found; "
		"falling back to '"+name+"', which is not MPI-parallel. Install PETSc/SLEPc with MUMPS support "
		"(then --petsc_mumps / the automatic default applies) -- see "
		"https://pyoomph.readthedocs.io/en/latest/tutorial/installation/petscslepc.html",
		RuntimeWarning,
		stacklevel=2,
	)


# Under MPI (mpirun -n N, N>1) the platform-preferred serial solvers are not usable: MKL Pardiso is not
# MPI-parallel at all (it raises in its constructor) and Accelerate is macOS-serial. PETSc+MUMPS is the
# only distributed-capable direct solver pyoomph ships, so it becomes the default whenever it is present,
# regardless of platform. Falls through to the normal serial cascade (with a warning) if it is missing,
# so that a run without PETSc still starts and can be steered explicitly from the command line.
if _running_under_mpi() and _set_petsc_mumps_linear_solver():
	pass
elif _is_macos and _is_arm64:
	if not _set_petsc_mumps_linear_solver():
		if _set_accelerate_linear_solver():
			_warn_suboptimal_solver("accelerate")
		else:
			_set_superlu_linear_fallback()
			_warn_suboptimal_solver("superlu")
elif _is_macos:
	if not _set_pardiso_linear_solver():
		if not _set_petsc_mumps_linear_solver():
			if _set_accelerate_linear_solver():
				_warn_suboptimal_solver("accelerate")
			else:
				_set_superlu_linear_fallback()
				_warn_suboptimal_solver("superlu")
else:
	if not _set_pardiso_linear_solver():
		if not _set_petsc_mumps_linear_solver():
			_set_superlu_linear_fallback()
			_warn_suboptimal_solver("superlu")


# Eigensolver default: SLEPc with MUMPS first, then Pardiso, then ARPACK (i.e. what --arpack selects,
# scipy's ARPACK, rather than the ARPACK shipped with Pardiso). Accelerate stays ahead of scipy on
# macOS, where it is the platform-native option and the only fast one left once Pardiso is out - on
# arm64 Macs there is no MKL at all.
#
# This deliberately does not follow the linear solver chosen above: on Linux, Pardiso is the better
# linear solver but SLEPc/MUMPS is the better eigensolver, so the two defaults now differ there.
#
# Registered as a resolver rather than evaluated here, because _have_petsc_mumps() imports
# petsc4py/slepc4py (~0.4 s, more than the rest of `import pyoomph` together) and most scripts never
# solve an eigenproblem. The probe therefore runs on the first Problem.get_eigen_solver() call.
def _autodetect_eigen_solver() -> CoreEigenSolverEnum:
	if _have_petsc_mumps():
		return "slepc_mumps"
	if _have_pardiso():
		return "pardiso"
	if _is_macos and _have_accelerate():
		return "accelerate"
	return "scipy"

set_default_eigen_solver_resolver(_autodetect_eigen_solver)

if _running_under_mpi():
	from .solvers.generic import get_default_linear_solver as _get_default_linear_solver
	_chosen=_get_default_linear_solver()
	if _chosen not in ("petsc_mumps","petsc","mumps"):
		from .generic.mpi import get_mpi_rank as _get_mpi_rank #type:ignore
		if _get_mpi_rank()==0:  # one warning per run, not one per rank
			_warn_no_mpi_capable_solver(str(_chosen))
	del _chosen
