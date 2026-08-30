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
def _num_threads_from_command_line():
    """Value of a ``--omp N`` on the command line, or None.

    Read here, before anything else is imported, because MKL and PETSc latch their thread counts from
    the environment at *their* import time - which is long before Problem.parse_cmd_line() would see
    the switch. The real handling of --omp is in Problem.setup_cmd_line()/parse_cmd_line(); this is
    only the part that has to happen early.
    """
    for i, a in enumerate(sys.argv):
        if a == "--omp":
            try:
                return max(1, int(sys.argv[i + 1]))
            except (IndexError, ValueError):
                return None
        if a.startswith("--omp="):
            try:
                return max(1, int(a.split("=", 1)[1]))
            except ValueError:
                return None
    return None


_omp_threads = _num_threads_from_command_line()
_default_blas_threads = str(_omp_threads) if _omp_threads else '4'
os.environ.setdefault('OPENBLAS_NUM_THREADS', os.environ.get('PYOOMPH_OPENBLAS_NUM_THREADS', _default_blas_threads))
os.environ.setdefault('MKL_NUM_THREADS', os.environ.get('PYOOMPH_MKL_NUM_THREADS', _default_blas_threads))
# pyoomph's own element loop takes its thread count from Problem.set_num_threads and passes it on the
# OpenMP pragma itself, so OMP_NUM_THREADS does not steer it. It is pinned anyway, at 1 unless --omp
# says otherwise, so that a third-party OpenMP runtime inside the linear solver cannot quietly open a
# second pool of threads next to ours and oversubscribe the machine.
os.environ.setdefault('OMP_NUM_THREADS', os.environ.get('PYOOMPH_OMP_NUM_THREADS', str(_omp_threads) if _omp_threads else '1'))
# The three PYOOMPH_*_NUM_THREADS above are pyoomph-only aliases of the standard variables, for
# pinning a third-party runtime without also pinning it for everything else on the machine. None of
# them touches pyoomph's OWN assembly threads: those come from --omp / Problem.set_num_threads and
# are passed on the pragma's num_threads clause (src/parallel_assembly.cpp), so --omp 1 - not an
# environment variable - is how the element loop is made serial.




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
_resolved_default_c_compiler:str | None=None


def get_default_c_compiler():
	"""The compiler a fresh Problem starts with: "system", unless that one cannot compile here.

	The preference is unchanged - the system toolchain is faster than the bundled tcc and is what
	every working installation ends up with - but it is now VERIFIED once per process instead of
	being asserted. On Windows "system" means MSVC whatever else is installed (distutils has no
	other default there), and an MSVC that cannot compile is not hypothetical: the Windows job of
	the full-suite run of 30th August 2026 got one whose vcvars returned an INCLUDE without the
	Windows SDK, so every JIT compile died on "Cannot open include file: 'math.h'" - on an image
	carrying a Visual Studio 18 preview, and not reproducible on the next runner. tccbox was
	installed and working the whole time, and nothing consulted it, because this function returned
	a hardcoded name that Problem.__init__ passes straight to set_c_compiler.

	check_avail() compiles and links a small program, so it is memoised: it would otherwise be paid
	per Problem, and tests/test_multiple_problems.py builds plenty.
	"""
	global _resolved_default_c_compiler
	if _resolved_default_c_compiler is not None:
		return _resolved_default_c_compiler
	from .generic.ccompiler import BaseCCompiler
	resolved=_default_c_compiler
	try:
		avail=BaseCCompiler.available_compilers()
	except Exception:
		avail={}   # a probe that raises must not stop a run before it starts
	if avail and resolved not in avail:
		best=max(avail,key=lambda k:avail[k])
		import warnings
		warnings.warn(
			"The '"+resolved+"' C compiler cannot compile on this machine, so pyoomph will JIT with "
			"'"+best+"' instead. On Windows this usually means the Visual Studio installation has no "
			"Windows SDK on its include path (the symptom is \"Cannot open include file: 'math.h'\"); "
			"the fallback works, but it is slower.",
			RuntimeWarning,stacklevel=2)
		resolved=best
	_resolved_default_c_compiler=resolved
	return resolved


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


def _have_spectra() -> bool:
	try:
		from .solvers.spectra import SpectraEigenSolver
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
		"pyoomph is running with multiple MPI processes, but no distributed direct solver was found, so "
		"'"+name+"' will be used. It is not MPI-parallel: the assembled system is gathered onto rank 0 and "
		"solved there while the other ranks wait, so the assembly scales but the solve does not, and rank 0 "
		"needs the whole matrix in memory. Install PETSc/SLEPc with MUMPS support for a genuinely "
		"distributed solve (then --petsc_mumps / the automatic default applies) -- see "
		"https://pyoomph.readthedocs.io/en/latest/tutorial/installation/petscslepc.html",
		RuntimeWarning,
		stacklevel=2,
	)


# Under MPI (mpirun -n N, N>1) the platform-preferred serial solvers still work -- the base solver class
# gathers the system onto rank 0 for them -- but they do not scale: only the assembly is parallel.
# PETSc+MUMPS is the only distributed-capable direct solver pyoomph ships, so it becomes the default
# whenever it is present, regardless of platform. Falls through to the normal serial cascade (with a
# warning saying what that costs) if it is missing.
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


# Eigensolver default: SLEPc with MUMPS first, then Spectra, then Pardiso, then ARPACK (i.e. what
# --arpack selects, scipy's ARPACK, rather than the ARPACK shipped with Pardiso). Accelerate stays
# ahead of scipy on macOS, where it is the platform-native option and the only fast one left once
# Pardiso is out - on arm64 Macs there is no MKL at all.
#
# Spectra outranks Pardiso because "pardiso" and "accelerate" are scipy's ARPACK with a different
# factorisation behind it, and that backend raises on any target at all. Spectra targets both real and
# complex eigenvalues, which is what Hopf tracking and the normal-form calculations need, and it uses
# MKL Pardiso for the factorisation itself when MKL is there - so it is strictly more capable than the
# entry it displaces, not merely different. Unlike _have_petsc_mumps() the probe below is cheap.
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
	if _have_spectra():
		return "spectra"
	if _have_pardiso():
		return "pardiso"
	if _is_macos and _have_accelerate():
		return "accelerate"
	return "scipy"

set_default_eigen_solver_resolver(_autodetect_eigen_solver)

if _running_under_mpi():
	from .solvers.generic import get_default_linear_solver as _get_default_linear_solver
	_chosen=_get_default_linear_solver()
	if _chosen not in ("petsc_mumps","petsc"):
		from .generic.mpi import get_mpi_rank as _get_mpi_rank #type:ignore
		if _get_mpi_rank()==0:  # one warning per run, not one per rank
			_warn_no_mpi_capable_solver(str(_chosen))
	del _chosen


from .typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
