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
 
from ..meshes.mesh import AnyMesh, ODEStorageMesh
from ..typings import *
import numpy
import os
import weakref


import scipy.sparse #type:ignore


# Passed for the arrays a given op_flag leaves unused, so the backends see a real (empty) array rather
# than None: oomph passes a null pointer there and the shim then hands Python a zero-length ndarray.
_EMPTY_F64=numpy.zeros(0,dtype=numpy.float64)
_EMPTY_I32=numpy.zeros(0,dtype=numpy.int32)


class SolverError(RuntimeError):
	"""Raised by a linear solver backend when a system could not be solved: a singular matrix, a
	factorisation that ran out of memory, an iterative solve that gave up.

	What separates it from an ordinary ``RuntimeError`` (which it derives from, so existing ``except
	RuntimeError`` around a solve keeps working) is what pyoomph does with it. The solver shim
	(``src/nanobind/solver.cpp``) reports a ``SolverError`` to oomph-lib as a failed Newton solve, so an
	adaptive time step halves ``dt`` and an arclength step scales ``Ds`` by 2/3 and both retry, instead
	of the exception unwinding the whole run.

	Raise it only for failures a smaller step could plausibly fix. A solver that is misconfigured or
	unavailable -- no MUMPS in this PETSc build, an unmatched field-split name, Pardiso under MPI --
	must stay an ordinary error: it would fail identically on every retry, and the run would shrink its
	step until it underflowed and then blame the time step rather than showing the real message.

	Backends subclass it one per backend (``PardisoError``, ``PETScSolverError``, ``ScipySolverError``),
	so a caller can still tell which one gave up.
	"""

DefaultMatrixType:TypeAlias=scipy.sparse.csr_matrix # spelled as an alias: a bare assignment is a variable, which cannot be used in an annotation
_TypeGenericLASolver=TypeVar("_TypeGenericLASolver",bound=type["GenericLinearSystemSolver"])
_TypeGenericEigenSolver=TypeVar("_TypeGenericEigenSolver",bound=type["GenericEigenSolver"])

CoreLinearSolverEnum:TypeAlias=Literal["superlu","umfpack","petsc","pardiso","accelerate","petsc_mumps"]
CoreEigenSolverEnum:TypeAlias=Literal["scipy","pardiso","slepc","accelerate","slepc_mumps"]
EigenSolverWhich:TypeAlias=Literal["LM","SM","LR","SR","SI"]
_default_la_solver:"GenericLinearSystemSolver | CoreLinearSolverEnum | None"=None
_default_eigen_solver:"GenericEigenSolver | CoreEigenSolverEnum | None"=None
_default_eigen_solver_resolver:"Callable[[],GenericEigenSolver | CoreEigenSolverEnum] | None"=None

if TYPE_CHECKING:
    from ..generic.problem import Problem

_PETSCSLEPC_INSTALL_URL="https://pyoomph.readthedocs.io/en/latest/tutorial/installation/petscslepc.html"
_PYPA_INSTALL_URL="https://pyoomph.readthedocs.io/en/latest/tutorial/installation/pypa.html"
_SOLVER_INSTALL_HINTS:dict[str,tuple[str,str]]={
	"petsc":("PETSc",_PETSCSLEPC_INSTALL_URL),
	"petsc_mumps":("PETSc with MUMPS support",_PETSCSLEPC_INSTALL_URL),
	"slepc":("SLEPc",_PETSCSLEPC_INSTALL_URL),
	"slepc_mumps":("SLEPc with MUMPS support",_PETSCSLEPC_INSTALL_URL),
	"pardiso":("Intel MKL (Pardiso)",_PYPA_INSTALL_URL),
	"accelerate":("the macOS Accelerate framework",_PYPA_INSTALL_URL),
}

def _mpi_any_rank(flag:bool)->bool:
	"""True if ``flag`` holds on ANY rank.

	Imported lazily, and short-circuited on a single process: pyoomph.generic.mpi initialises MPI on
	import, and this module is imported by every script including the ones that never touch a solver.
	"""
	from ..generic.mpi import get_mpi_nproc,get_mpi_any
	if get_mpi_nproc()<=1:
		return bool(flag)
	return get_mpi_any(bool(flag))

def _unavailable_solver_message(kind:str,name:str,available:list[str],e:Exception)->str:
	msg=kind+" '"+name+"' is not available ("+type(e).__name__+": "+str(e)+"). Available: "+str(available)+"."
	hint=_SOLVER_INSTALL_HINTS.get(name)
	if hint is not None:
		msg+=" See "+hint[1]+" for how to install "+hint[0]+"."
	return msg

def set_default_linear_solver(solv:"GenericLinearSystemSolver | CoreLinearSolverEnum"):
	global _default_la_solver
	_default_la_solver=solv

def get_default_linear_solver()->"GenericLinearSystemSolver | CoreLinearSolverEnum | None":
	return _default_la_solver


def set_default_eigen_solver(solv:"GenericEigenSolver | CoreEigenSolverEnum"):
	global _default_eigen_solver,_default_eigen_solver_resolver
	_default_eigen_solver=solv
	_default_eigen_solver_resolver=None # an explicit choice wins over the autodetection below

def set_default_eigen_solver_resolver(resolver:Callable[[],"GenericEigenSolver | CoreEigenSolverEnum"]):
	"""Register a callable that picks the default eigensolver the first time one is actually needed.

	The autodetection in pyoomph/__init__.py has to import petsc4py/slepc4py to find out whether SLEPc
	with MUMPS is usable, which costs more time than the entire rest of `import pyoomph` and is wasted
	on the majority of scripts, which never solve an eigenproblem. Deferring the probe keeps the import
	cheap without changing which solver is eventually selected.
	"""
	global _default_eigen_solver,_default_eigen_solver_resolver
	_default_eigen_solver=None
	_default_eigen_solver_resolver=resolver

def get_default_eigen_solver()->"GenericEigenSolver | CoreEigenSolverEnum | None":
	global _default_eigen_solver
	if _default_eigen_solver is None and _default_eigen_solver_resolver is not None:
		_default_eigen_solver=_default_eigen_solver_resolver()
	return _default_eigen_solver

class GenericLinearSystemSolver:
	_registered_solvers:dict[str,type["GenericLinearSystemSolver"]]={}
	idname:str
	

	def __init__(self,problem:"Problem"):
		self.problem=problem

	@property
	def problem(self)->"Problem":
		# Stored as a weakref: a strong reference here would form a Problem<->solver
		# cycle that keeps the Problem alive as long as any mesh (itself pinned by the
		# Problem's own nb::keep_alive) transitively references this solver.
		p=self._problem_wr()
		assert p is not None, "The Problem this solver belonged to has already been destroyed"
		return p

	@problem.setter
	def problem(self,p:"Problem | None"):
		self._problem_wr=weakref.ref(p) if p is not None else (lambda:None)

	def requires_explicit_diagonal(self)->bool:
		"""Whether this solver needs an entry stored on every diagonal of the Jacobian, including where
		the equations put nothing there.

		Needing one is a property of the factorisation, not of the problem: PETSc's own LU rejects a
		matrix with a missing diagonal outright ("Matrix is missing diagonal entry i"), while MUMPS,
		Pardiso and SuperLU do not care. It is not free to say yes -- the extra stored zeros change the
		matrix the solver sees and therefore its pivoting -- so the default is no, and a solver should
		override this only when its configured factorisation genuinely requires it.

		A solver that answers wrongly in the "no" direction is not silently broken: the user gets the
		factoriser's own clear complaint and can force the issue with
		``problem.force_jacobian_diagonal_entries = True``.
		"""
		return False

	def setup_solver(self)->None:
		pass

	def _report_structure_id_mismatch(self,what:str)->None:
		"""Say that the sparsity pattern moved under an unchanged ``problem.jacobian_structure_id``.

		Both reuse paths (Pardiso's symbolic factorisation and PETSc's preallocated Mat) verify the
		pattern rather than trusting the id, and fall back correctly when it does not match, so this
		is never a correctness problem -- but whether it is a *defect* depends on the system.

		On an augmented one -- bifurcation, eigenbranch or periodic-orbit tracking -- it is expected:
		the elemental block there is larger than the field description, so no symbolic mask applies
		and the pattern falls back to being value-filtered, which the id cannot promise anything
		about. eigenbranch_continuation.py produces hundreds of these. On a plain system it really is
		a missing invalidation, and the wording used to say so unconditionally, which turned that
		tutorial's log into hundreds of requests to report a non-bug.

		Reported once per solver object either way: the second occurrence adds nothing the first did
		not, and staying silent altogether would hide the case that IS worth reporting.
		"""
		if getattr(self,"_structure_id_mismatch_reported",False):
			return
		self._structure_id_mismatch_reported=True
		try:
			augmented=self.problem._get_n_unaugmented_dofs()!=self.problem.ndof() #type:ignore
		except Exception:
			augmented=False
		msg=("the Jacobian sparsity pattern changed although problem.jacobian_structure_id did not, "
			 "so "+what+" was rebuilt instead of reused. ")
		if augmented:
			msg="NOTE: "+msg+("Expected on this augmented (tracking) system, whose pattern is "
							  "value-filtered. Costs a rebuild per solve; not an error. Reported once.")
		else:
			msg="WARNING: "+msg+("This system is not augmented, so it is a bug in the pattern "
								 "invalidation -- please report it. Reported once.")
		print(msg)

	def _before_assigning_equation_numbers(self)->None:		
		pass


	def solve_distributed(self, op_flag: int, allow_permutations: int, n: int, nnz_local: int, nrow_local: int, first_row: int, values: NPFloatArray, col_index: NPIntArray, row_start: NPIntArray, b: NPFloatArray, nprow: int, npcol: int, doc: int, data: NPUInt64Array, info: NPIntArray)->None:
		if self.gathers_to_root_under_mpi:
			return self._solve_distributed_on_root(op_flag,allow_permutations,n,nnz_local,nrow_local,first_row,values,col_index,row_start,b,nprow,npcol,doc,data,info)
		raise RuntimeError("This solver cannot be used with multiple MPI processes.")

	def solve_serial(self,op_flag:int,n:int,nnz:int,nrhs:int,values:NPFloatArray,rowind:NPIntArray,colptr:NPIntArray,b:NPFloatArray,ldb:int,transpose:int)->int:
		raise NotImplementedError("You need to specialise the function 'solve_serial'")


	def distributed_possible(self)->bool:
		return True

	#################### Gather-to-root: a serial backend under mpirun ####################

	# Whether solve_distributed() may fall back to gathering the whole system onto rank 0 and calling
	# this solver's own solve_serial() there. Opt-in, so a backend that has not thought about MPI keeps
	# getting the loud refusal above rather than a silent serialisation.
	#
	# Setting it asserts two things about solve_serial(): it performs no MPI collective (it runs on rank
	# 0 alone, so anything collective inside it hangs the job), and it does not mind being called with
	# a system much larger than this rank assembled.
	gathers_to_root_under_mpi:bool=False

	# True for a backend that genuinely solves in parallel. Only used to word the messages: a gathering
	# backend works, it just does not scale, and the two should not read the same.
	solves_natively_distributed:bool=False

	_gather_layout:"Any"=None
	_gather_notice_printed:bool=False

	def _note_external_serial_solve(self)->None:
		"""Tell the solver that someone called solve_serial() directly, outside solve_distributed().

		Several utilities (Lyapunov exponents, the periodic driving response, Halley's method) build a
		globally REPLICATED system in Python and call solve_serial() on every rank. That writes the same
		factorisation slot the gathered path keeps rank 0's factors in. Interleaved with a gathered
		Newton solve, rank 0 would then back-substitute against the wrong factors while the other ranks
		had no way to notice. Dropping the layout makes the next gathered back-substitution fail loudly
		instead.
		"""
		self._gather_layout=None

	def _agree_on_gathered_outcome(self,code:int)->int:
		"""Collective: agree on how the gathered solve went. 0 = fine, 1 = SolverError, 2 = anything else.

		Every rank must call this, and the answer decides what all of them do next -- which is the whole
		point: rank 0 is the only one that solves, so it is the only one that can see the failure, and a
		branch taken on anything else would split the ranks between different collectives.
		"""
		import numpy
		from mpi4py import MPI #type:ignore
		from ..generic.mpi import get_mpi_world_comm,mpi_wait_idle
		comm=get_mpi_world_comm()
		assert comm is not None
		send=numpy.array([code],dtype=numpy.int32)
		recv=numpy.zeros(1,dtype=numpy.int32)
		req=comm.Iallreduce(send,recv,op=MPI.MAX) #type:ignore
		mpi_wait_idle(req,"the gathered solve on rank 0")
		return int(recv[0])

	def _report_gather_to_root_once(self,n:int)->None:
		"""Say once, on rank 0, that the system is being solved on one rank -- and what that costs."""
		if self._gather_notice_printed:
			return
		self._gather_notice_printed=True
		from ..generic.mpi import get_mpi_rank,get_mpi_nproc
		if get_mpi_rank()!=0 or self.problem.is_quiet():
			return
		msg=("NOTE: the linear solver '"+str(getattr(self,"idname",type(self).__name__))+"' is not "
			 "MPI-parallel, so the assembled system ("+str(n)+" dofs) is gathered onto rank 0 and solved "
			 "there while the other "+str(get_mpi_nproc()-1)+" rank(s) wait. Assembly stays parallel; the "
			 "solve does not scale. Use petsc_mumps for a genuinely distributed solve.")
		# Freeing the other cores only helps if rank 0 is allowed to use them, and Open MPI binds by
		# core at -n 2 and by socket above it. pyoomph cannot change that from inside the process:
		# binding is applied by mpirun before exec.
		if hasattr(os,"sched_getaffinity"):
			allowed=len(os.sched_getaffinity(0))
			total=os.cpu_count() or allowed
			if allowed<total:
				msg+=("\n      Rank 0 may use "+str(allowed)+" of "+str(total)+" CPUs, so its threads cannot "
					  "take the cores the waiting ranks give up. Re-run with 'mpirun --bind-to none' and "
					  "set the thread count via problem.set_num_threads(...).")
		print(msg,flush=True)

	def _solve_distributed_on_root(self, op_flag: int, allow_permutations: int, n: int, nnz_local: int, nrow_local: int, first_row: int, values: NPFloatArray, col_index: NPIntArray, row_start: NPIntArray, b: NPFloatArray, nprow: int, npcol: int, doc: int, data: NPUInt64Array, info: NPIntArray)->None:
		"""Gather the row-distributed system onto rank 0, solve it there, scatter the solution back.

		oomph-lib routes every run with nproc>1 through the distributed entry point, whether or not the
		problem was distributed (SuperLUSolver::solve branches on nproc(), not on the mesh), so a serial
		backend is otherwise unusable under mpirun at all. This makes it usable: the element loop is
		still split across the ranks, only the factorisation is serialised.

		Nothing here writes ``data`` or ``info``. ``data`` arrives as None -- pyoomph's shim never puts a
		handle there, which is also why op_flag==3 never arrives (oomph guards its cleanup on that same
		handle being non-null).
		"""
		import numpy
		from ..generic.mpi import (get_mpi_rank,get_mpi_nproc,mpi_row_layout,mpi_gather_csr_rows,
								   mpi_gather_vector,mpi_scatter_vector,mpi_share_any_failure)
		# Every branch below is decided either on op_flag (which oomph passes identically to all ranks)
		# or on the agreed outcome code. Nothing branches on a local count.
		if op_flag==3:
			self._gather_layout=None
			return
		if op_flag not in (1,2):
			raise RuntimeError("Cannot handle op_flag "+str(op_flag)+" in the gather-to-root solve")
		if get_mpi_nproc()<=1:
			# Only reachable if a solver was forced to Distributed on a single process; the "local"
			# block is then the whole system and there is nothing to gather.
			self.solve_serial(op_flag,n,nnz_local,1 if op_flag==2 else 0,values,col_index,row_start,b,n,1)
			return
		# Replicated condition, checked before the first collective so all ranks refuse together. The
		# custom solve routine would run on rank 0 alone, and the augmented handlers reach back into
		# the problem from inside it; the custom assembler is not supported under MPI anyway.
		if self.problem._custom_assembler is not None and self.problem._custom_assembler.has_custom_solve_routine(): #type:ignore
			raise RuntimeError("A custom solve routine (deflation, an augmented assembly handler) cannot "
							   "be used together with a linear solver that gathers the system onto rank 0 "
							   "under MPI: it would run on rank 0 only. Use petsc_mumps, or run serially.")

		rank=get_mpi_rank()
		error:BaseException | None=None
		if op_flag==1:
			self._report_gather_to_root_once(n)
			layout=mpi_row_layout(n,first_row,nrow_local,nnz_local)
			assert layout is not None
			self._gather_layout=layout
			gathered=mpi_gather_csr_rows(layout,values,col_index,row_start)
			if rank==0:
				assert gathered is not None
				vals_g,cols_g,rs_g=gathered
				try:
					# The argument convention is the one oomph uses serially for a CRDoubleMatrix
					# (linear_solver.cc:2261-2264, 2303-2314): CSR data through the SuperLU-named
					# (values, rowind, colptr) slots, nrhs=0 and b unused on a factorise, transpose=1.
					self.solve_serial(1,n,layout.nnz_total,0,vals_g,cols_g,rs_g,_EMPTY_F64,n,1)
				except BaseException as e:
					error=e
			code=self._agree_on_gathered_outcome(0 if error is None else (1 if isinstance(error,SolverError) else 2))
		else:
			layout=self._gather_layout
			# Replicated: op_flag==1 always precedes a back-substitution (oomph gates the call on the
			# factorisation having been allocated) and sets the layout on every rank before it can fail.
			if layout is None:
				raise RuntimeError("A gathered back-substitution was requested without a preceding "
								   "gathered factorisation. Something solved with this backend outside "
								   "the distributed path in between.")
			local_code=0
			if len(b)!=int(layout.vec_counts[rank]):
				error=RuntimeError("The gathered layout expects "+str(int(layout.vec_counts[rank]))+
								   " local rows on this rank, but the right-hand side has "+str(len(b))+".")
				local_code=2
			b_global=mpi_gather_vector(layout,b if local_code==0 else numpy.zeros(int(layout.vec_counts[rank])))
			if rank==0 and local_code==0:
				assert b_global is not None
				try:
					self.solve_serial(2,n,0,1,_EMPTY_F64,_EMPTY_I32,_EMPTY_I32,b_global,n,1)
				except BaseException as e:
					error=e
					local_code=1 if isinstance(e,SolverError) else 2
			code=self._agree_on_gathered_outcome(local_code)
			if code==0:
				mpi_scatter_vector(layout,b_global,b)
		if code==0:
			return
		if code==1:
			# A SolverError is left to the shim in src/nanobind/solver.cpp, which MPI_Allreduces the
			# "did this rank report a solver failure" flag right after this call and turns it into a
			# retryable NewtonSolverError on every rank -- it says so explicitly, for exactly this case.
			# So only rank 0 raises, and b keeps the right-hand side rather than a fabricated solution:
			# if anything ever swallowed the NewtonSolverError, a bounded wrong step is far easier to
			# notice than a zero one, which looks like convergence.
			self._gather_layout=None
			if error is not None:
				raise error
			return
		# Anything else cannot go through the shim: it rethrows a non-SolverError BEFORE reaching its
		# Allreduce, so rank 0 would unwind while the others sat in it. Agree here instead.
		self._gather_layout=None
		mpi_share_any_failure(error,context="solving the gathered linear system on rank 0")

	@classmethod
	def register_solver(cls,*,override:bool=False)->Callable[[_TypeGenericLASolver],_TypeGenericLASolver]:
		def decorator(subclass:_TypeGenericLASolver)->_TypeGenericLASolver:
			name=subclass.idname
			if name in cls._registered_solvers.keys():
					if not override:
						raise RuntimeError("You tried to register the solver "+name+", but there is already one defined. Please add override=True to the arguments of @GenericLinearSystemSolver.register_solver(override=True)")
			cls._registered_solvers[name] = subclass
			return subclass
		return decorator

	@staticmethod
	def factory_solver(name:str,problem:"Problem") -> "GenericLinearSystemSolver":
		if name in GenericLinearSystemSolver._registered_solvers.keys():
			return GenericLinearSystemSolver._registered_solvers[name](problem)
		else:
			# The module a backend lives in, where it is not simply named after it. superlu/umfpack
			# were only ever registered by accident: pyoomph/__init__.py imports solvers.scipy inside
			# its superlu fallback, and solvers.pardiso pulls it in for the eigensolver. Under mpirun
			# the default cascade picks petsc_mumps, so neither happens and an explicit --superlu then
			# failed with "No module named 'pyoomph.solvers.superlu'".
			libname={"petsc_mumps":"petsc","superlu":"scipy","umfpack":"scipy"}.get(name,name)
			try:
				import importlib
				#__import__(name)
				importlib.import_module("pyoomph.solvers."+libname)
				if name in GenericLinearSystemSolver._registered_solvers.keys():
					return GenericLinearSystemSolver._registered_solvers[name](problem)
				else:
					raise RuntimeError("Unknown Linear Algebra solver: '"+name+"'. Following are defined (and included): "+str(list(GenericLinearSystemSolver._registered_solvers.keys())))
			except Exception as e:
				raise RuntimeError(_unavailable_solver_message("Linear Algebra solver",name,list(GenericLinearSystemSolver._registered_solvers.keys()),e)) from e


	def set_num_threads(self,nthreads:int | None) -> None:
		pass

##########



class EigenMatrixManipulatorBase:
	def __init__(self,problem:"Problem") -> None:
		super(EigenMatrixManipulatorBase, self).__init__()
		self.problem=problem

	def _not_present_locally(self,mesh:Any=None)->bool:
		"""Whether a named part of the mesh is simply absent from THIS rank.

		Only ever true on a distributed problem. A corner interface such as ``domain/bottom/left`` is a
		single point and so belongs to exactly one partition; on the others the submesh is either
		missing or present but empty, and asking it for its fields reports none.

		Distinguishing that from a mistyped field name is not possible from here, so a distributed run
		trades the typo diagnostic for the ability to run at all. It has to: raising on the ranks that
		do not hold the piece is not a clean error but a SPLIT, with those ranks unwinding while the
		owner walks into the next collective alone. That is how this surfaced -- as MPI_ERR_BUFFER out
		of PETSc, several stack frames away from the name that could not be resolved.
		"""
		if not self.problem.is_distributed():
			return False
		return mesh is None or mesh.nelement()==0

	def resolve_equations_by_name(self,name:str) -> set[int]:
		"""The global equation numbers of the dofs named by e.g. ``domain/left/velocity_y``.

		On a distributed problem this returns only what the CALLING RANK can see, which is the whole
		point: the caller restricts itself to the rows it owns anyway, and every owned row is reachable
		from a local non-halo element. A part of the mesh that lives entirely on another rank must
		therefore resolve to nothing here rather than to an error -- see _not_present_locally().
		"""
		from ..generic.problem import Problem
		from .. import _pyoomph_core as _pyoomph
		splt=name.split("/")
		root:"Problem | AnyMesh"=self.problem
		fieldname=None
		#print("IN ",name)
		for i,k in enumerate(splt):
			if not isinstance(root,ODEStorageMesh):
				nextone=root.get_mesh(k,return_None_if_not_found=True)
				if nextone is None and root==self.problem:
					#Try whether it is an ODE
					ode=self.problem.get_ode(k)
					root=ode
					if len(splt)!=2:
						raise RuntimeError("Cannot access the ODE variable "+name+". Happens when trying to access "+str(name))
					fieldname=splt[1]
					break
			else:
				nextone=None
			if nextone is None:
				if i<len(splt)-1:
					if self._not_present_locally():
						return set()
					print("Splitted is :",splt)
					raise RuntimeError("Cannot access the mesh "+str("/".join(splt[0:i-1]))+" to access the degrees of freedom "+str(name))
				else:
					fieldname=splt[-1]
			else:
				root=nextone
		if fieldname is None:
			raise RuntimeError("Cannot set a full mesh yet")
		assert root is not None and not isinstance(root,Problem)
		assert isinstance(root,_pyoomph.Mesh)
		fi = root.get_field_information()
		if fieldname not in fi.keys():
			# An EMPTY submesh reports no fields at all, which is not the same as the user naming a
			# field that does not exist.
			if self._not_present_locally(root):
				return set()
			raise RuntimeError("Cannot find field "+str(fieldname)+" in mesh "+root.get_full_name())
		res:set[int]=set()
		if  isinstance(root,ODEStorageMesh):
			odeelem = root.get_element()
			_, inds = odeelem._ode_elem_to_numpy()
			if not fieldname in inds.keys():
				raise RuntimeError("Cannot get the field '"+fieldname+"' on ODE domain "+root.get_full_name())
			eqn=odeelem.internal_data_pt(inds[fieldname]).eqn_number(0)
			if eqn>=0:
				res.add(eqn)
		else:
			coord_dir_index=None
			if fieldname=="mesh_x":
				coord_dir_index=0
			elif fieldname=="mesh_y":
				coord_dir_index=1
			elif fieldname=="mesh_z":
				coord_dir_index=2

			for e in root.elements():
				is_nodal=False
				if coord_dir_index is not None:
					for ni in range(e.nnode()):
						n=e.node_pt(ni)
						eqn=n.variable_position_pt().eqn_number(coord_dir_index)

						if eqn>=0:
							res.add(eqn)
				else:
					for ni in range(e.nnode()):
						n=e.node_pt(ni)
						val_index=e.get_nodal_index_by_name(n,fieldname)
						if val_index<0 and is_nodal:
							raise RuntimeError("Cannot access "+fieldname+" in node "+str(n)+" of element "+str(e)+" on mesh "+root.get_full_name())
						if val_index<0:
							is_nodal=False
						else:
							is_nodal=True
						if val_index<0:
							raise RuntimeError("TODO",fieldname,root.get_full_name())
						eqn=n.eqn_number(val_index)
						if eqn>=0:
							res.add(eqn)
					if not is_nodal:
						raise RuntimeError("DISCONT FIELDS HERE")
		return res

	def apply_on_J_and_M(self,solver:"GenericEigenSolver",J:DefaultMatrixType,M:DefaultMatrixType)->tuple[DefaultMatrixType,DefaultMatrixType]:
		return J,M

	def apply_on_distributed_J_and_M(self,solver:"GenericEigenSolver",J:Any,M:Any)->tuple[Any,Any]:
		"""Counterpart of apply_on_J_and_M for a row-partitioned eigenproblem.

		J and M are the eigensolver backend's own matrices (a petsc4py ``Mat`` for the SLEPc solver),
		not scipy ones: on a distributed problem no rank holds the whole matrix, so the row surgery
		apply_on_J_and_M does has to happen where the ownership range is known. Only the backend's
		matrix interface is used here, so this module still does not import PETSc.

		Manipulators that have no distributed equivalent inherit this and stop the run with a clear
		message rather than silently leaving the constraint unapplied.
		"""
		raise RuntimeError(type(self).__name__+" cannot be applied to a distributed (MPI) eigenproblem yet. Run the eigenproblem without --distribute, or drop the manipulator.")


class EigenMatrixSetDofsToZero(EigenMatrixManipulatorBase):
	def __init__(self,problem:"Problem",*doflist:str | int):
		super(EigenMatrixSetDofsToZero, self).__init__(problem)
		self.doflist:set[str | int]=set(doflist)
		self.zeromap:set[int]=set()
		self.last_zeroed_rows:int=0


	def setcsrrow2id(self,amat:DefaultMatrixType, rowind:int):
		indptr = amat.indptr #type:ignore
		values = amat.data #type:ignore
		indxs = amat.indices #type:ignore

		# get the range of the data that is changed
		rowpa = indptr[rowind] #type:ignore
		rowpb = indptr[rowind + 1] #type:ignore

		# new value and its new rowindex
		#print(rowind,rowpa,rowpb,values.shape,indxs.shape)
		if rowpa>=len(values): #type:ignore
			#raise RuntimeError("Here is still something strange")
			values=values.copy()
			indxs=indxs.copy()
			print(rowpa,len(values))
			values=numpy.pad(values, (0, rowpa-len(values)+1), 'constant')
			indxs = numpy.pad(indxs, (0, rowpa - len(indxs) + 1), 'constant')
			#values.resize([rowpa+1])
			#indxs.resize([rowpa+1])
		values[rowpa] = 1.0
		indxs[rowpa] = rowind

		# number of new zero values
		diffvals = rowpb - rowpa - 1 #type:ignore


		# filter the data and indices and adjust the range
		#values[rowpa+1:rowpb-1]=0.0
		#if diffvals >= 0:
		values = numpy.r_[values[:rowpa + 1], values[rowpb:]]
		indxs = numpy.r_[indxs[:rowpa + 1], indxs[rowpb:]]
		indptr = numpy.r_[indptr[:rowind + 1], indptr[rowind + 1:] - diffvals]

		# hard set the new sparse data
		amat.indptr = indptr
		amat.data = values
		amat.indices = indxs

	def set_rows_to_identity(self,A:DefaultMatrixType,rows:Iterable[int]) -> DefaultMatrixType:
		for i in rows:
			A.data[A.indptr[i]: A.indptr[i + 1]] = 0.0 #type:ignore
			A[i,i]=1 

		return A

	def _resolve_zeromap(self)->set[int]:
		"""The global equation numbers this manipulator constrains."""
		from .. import _pyoomph_core as _pyoomph
		zeromap:set[int]=set()
		for d in self.doflist:
			if isinstance(d,str):
				eqs=self.resolve_equations_by_name(d)
			else:
				eqs=set([d])
			if  _pyoomph.get_verbosity_flag() != 0:
				print("INFO ",d,eqs)
			zeromap=zeromap.union(eqs)
		return zeromap

	def apply_on_distributed_J_and_M(self,solver:"GenericEigenSolver",J:Any,M:Any) -> tuple[Any, Any]:
		# Same constraint as the scipy version below -- J's row becomes delta_ij and M's becomes zero --
		# expressed as the backend's own row operation, which knows the ownership range.
		#
		# Restricted to locally owned rows on purpose. resolve_equations_by_name() walks the local
		# elements, which on a distributed mesh includes halo elements, so it also reports equation
		# numbers this rank does not own; those rows belong to the rank that does own them, and that rank
		# reaches them through its own non-halo elements. Filtering here is therefore complete as well as
		# safe, and it keeps the call free of off-process row communication.
		_,nrow_local,first_row,_=solver.get_eigen_row_layout()
		zeromap=self._resolve_zeromap()
		self.zeromap=zeromap
		rows=sorted(r for r in zeromap if first_row<=r<first_row+nrow_local)
		self.last_zeroed_rows=len(rows)   # so a test can tell "applied nothing here" from "never ran"
		# Collective on both matrices, so every rank calls them even with nothing of its own to zero.
		J.zeroRows(rows,diag=1.0)
		M.zeroRows(rows,diag=0.0)
		return J,M

	def apply_on_J_and_M(self,solver:"GenericEigenSolver",J:DefaultMatrixType,M:DefaultMatrixType) -> tuple[DefaultMatrixType, DefaultMatrixType]:
		self.zeromap=self._resolve_zeromap()
		#print("GOING TO SET TO ZERO",self.zeromap)
		N=J.shape[0]
		Adiag=numpy.ones(N)
		Adiag[numpy.array(sorted(list(self.zeromap)),dtype=numpy.int64)] = 0.0
		Bdiag=1-Adiag
		A=scipy.sparse.spdiags(Adiag, [0], N, N).tocsr()
		B=scipy.sparse.spdiags(Bdiag, [0], N, N).tocsr()
		J=A@J+B # Set removed rows to delta_ij
		M=A@M # Set removed rows to zero
		return J,M
		


	def apply_on_J_and_M___OLD(self,solver:"GenericEigenSolver",J:DefaultMatrixType,M:DefaultMatrixType) -> tuple[DefaultMatrixType, DefaultMatrixType]:
		# TODO OLD VERSION: Slow, remove
		from .. import _pyoomph_core as _pyoomph
		self.zeromap=set()
		for d in self.doflist:
			if isinstance(d,str):
				eqs=self.resolve_equations_by_name(d)
			else:
				eqs=set([d])
			if  _pyoomph.get_verbosity_flag() != 0:
				print("INFO ",d,eqs)
			self.zeromap=self.zeromap.union(eqs)
		if len(self.zeromap)>0:
			#J=self.set_rows_to_identity(J,list(self.zeromap))
			for k in reversed(sorted(self.zeromap)):
				#print("SET TO ZERO",k)       
				self.setcsrrow2id(J,k)
			#J=self.set_rows_to_identity(J,list(self.zeromap))
			J.eliminate_zeros()
			for row in self.zeromap:
				M.data[M.indptr[row]:M.indptr[row + 1]] = 0 #type:ignore
			M.eliminate_zeros()
		return J,M


class GenericEigenSolver:
	_registered_solvers:dict[str,type["GenericEigenSolver"]]={}
	idname:str
	def __init__(self,problem:"Problem"):
		self.problem=problem
		self.matrix_manipulators:list[EigenMatrixManipulatorBase]=[]
		self.real_contribution:str=""
		self.imag_contribution:str | None=None
		self.ncv:int | None=None
		self.last_assembly_was_complex:bool=False

	@property
	def problem(self)->"Problem":
		# See GenericLinearSystemSolver.problem: kept as a weakref so this solver does not
		# form an uncollectible Problem<->solver reference cycle.
		p=self._problem_wr()
		assert p is not None, "The Problem this solver belonged to has already been destroyed"
		return p

	@problem.setter
	def problem(self,p:"Problem | None"):
		self._problem_wr=weakref.ref(p) if p is not None else (lambda:None)

	def _before_assigning_equation_numbers(self)->None:
		pass

	def supports_target(self)->bool:
		return False

	def setup_matrix_contributions(self,real_contribution:str,imag_contribution:str | None=None):
		self.real_contribution=real_contribution
		self.imag_contribution=imag_contribution
	
	def distributed_possible(self) -> bool:
		return True

	@classmethod
	def register_solver(cls,*,override:bool=False)->Callable[[_TypeGenericEigenSolver],_TypeGenericEigenSolver]:
		def decorator(subclass:_TypeGenericEigenSolver)->_TypeGenericEigenSolver:
			name=subclass.idname
			if name in cls._registered_solvers.keys():
					if not override:
						raise RuntimeError("You tried to register the solver "+name+", but there is already one defined. Please add override=True to the arguments of @GenericEigenSolver.register_solver(override=True)")
			cls._registered_solvers[name] = subclass
			return subclass
		return decorator

	def solve(self,neval:int,shift:float | complex | None=None,sort:bool=True,which:EigenSolverWhich="LM",OPpart:Literal["r", "i"] | None=None,v0:NPComplexArray | NPFloatArray | None=None,target:complex | None=None,custom_J_and_M:tuple[DefaultMatrixType,DefaultMatrixType] | None=None,with_left_eigenvectors:bool=False,quiet:bool=True)->tuple[NPComplexArray,NPComplexArray,DefaultMatrixType,DefaultMatrixType]:
		raise RuntimeError("Here")
	
	@staticmethod
	def factory_solver(name:str,problem:"Problem")->"GenericEigenSolver":
		if name in GenericEigenSolver._registered_solvers.keys():
			return GenericEigenSolver._registered_solvers[name](problem)
		else:
			libname=name
			if libname=="slepc" or libname=="slepc_mumps":
				libname="petsc"
			try:
				import importlib
				importlib.import_module("pyoomph.solvers."+libname)
				if name in GenericEigenSolver._registered_solvers.keys():
					return GenericEigenSolver._registered_solvers[name](problem)
				else:
					raise RuntimeError("Unknown Eigen solver: '"+name+"'. Following are defined (and included): "+str(list(GenericEigenSolver._registered_solvers.keys())))
			except Exception as e:
				raise RuntimeError(_unavailable_solver_message("Eigen solver",name,list(GenericEigenSolver._registered_solvers.keys()),e)) from e

			#raise RuntimeError("Unknown Eigen solver: '"+name+"'. Following are defined (and included): "+str(list(GenericEigenSolver._registered_solvers.keys())))

	def add_matrix_manipulator(self,manip:EigenMatrixManipulatorBase):
		self.matrix_manipulators.append(manip)

	def clear_matrix_manipulators(self):
		self.matrix_manipulators.clear()


	def get_eigen_row_layout(self)->tuple[int,int,int,bool]:
		"""Row layout ``(n, nrow_local, first_row, distributed)`` of the eigenproblem matrices.

		They are assembled on the problem's dof distribution, which is row-partitioned once the problem
		has been distributed: each rank then holds only rows ``first_row .. first_row+nrow_local-1``,
		with GLOBAL column indices. Serially, and under MPI without ``--distribute`` (where oomph-lib
		redistributes the assembled matrices back to a globally replicated form), ``nrow_local == n``,
		``first_row == 0`` and ``distributed`` is False.
		"""
		return self.problem._get_dof_distribution_info() #type:ignore

	def get_J_M_n_and_type(self)->tuple[DefaultMatrixType,DefaultMatrixType,int,bool]:
		"""Assemble the eigenproblem's J and M and report the global size and whether they are complex.

		The matrices are ``(nrow_local, n)`` rather than square whenever the problem is distributed;
		see get_eigen_row_layout(). Every caller that has to know the difference asks for the layout
		itself, so that the serial shape stays exactly what it always was.
		"""
		from scipy.sparse import csr_matrix #type:ignore
		if not self.problem._set_solved_residual(self.real_contribution,True,False):
			raise RuntimeError("Cannot set the residual "+self.real_contribution+" for eigen calculation since it has no contribution at all")
		n, M_nzz, M_nr, M_val, M_ci, M_rs, J_nzz, J_nr, J_val, J_ci, J_rs = self.problem.assemble_eigenproblem_matrices(0) #type:ignore
		# shape=(M_nr, n), not (n, n): M_nr is the LOCAL row count, and on a distributed problem it is
		# smaller than n. Passing (n,n) with a short indptr is what made every distributed eigen solve
		# die inside scipy before the eigensolver was ever reached.
		matM=csr_matrix((M_val, M_ci, M_rs), shape=(M_nr, n))	#TODO: Is csr or csc?
		matJ=csr_matrix((-J_val, J_ci, J_rs), shape=(J_nr, n))
		is_complex=False
		if self.imag_contribution is not None:
			if self.problem._set_solved_residual(self.imag_contribution,False,False):
				matM=cast(csr_matrix,matM.copy())
				matJ=cast(csr_matrix,matJ.copy())
				n, M_nzz, M_nr, M_val, M_ci, M_rs, J_nzz, J_nr, J_val, J_ci, J_rs = self.problem.assemble_eigenproblem_matrices(0) #type:ignore
				matMi = csr_matrix((M_val, M_ci, M_rs), shape=(M_nr, n))  # TODO: Is csr or csc?
				matJi = csr_matrix((-J_val, J_ci, J_rs), shape=(J_nr, n))
				# Both counts are per-rank, so on a distributed problem a partition whose local block
				# happens to carry no imaginary entries would answer differently from the others -- and
				# they would then disagree about whether this is a complex eigenproblem, which is not a
				# wrong answer but a deadlock, since the two branches issue different collectives
				# downstream. Decide it globally.
				has_Mi=_mpi_any_rank(M_nzz>0)
				has_Ji=_mpi_any_rank(J_nzz>0)
				if has_Mi:
					matM=cast(csr_matrix,matM+complex(0,1)*matMi)
					is_complex = True
				if has_Ji:
					matJ =cast(csr_matrix,matJ+ complex(0, 1) * matJi)
					is_complex=True

		self.problem._set_solved_residual("",True,True)

		#print("Applying Matrix manipulators")
		# Skipped, NOT refused, when distributed: these manipulators rewrite whole rows of a square
		# global matrix, which is not what a rank holds here. The eigensolver applies them afterwards to
		# its own distributed matrices, where the ownership range is known -- see
		# apply_on_distributed_J_and_M() and SlepcEigenSolver.solve(). A backend that neither handles
		# them there nor can work distributed at all is stopped earlier, by distributed_possible().
		if self.matrix_manipulators and not self.get_eigen_row_layout()[3]:
			for manip in self.matrix_manipulators:
				#print("APPLY MANIP",manip)
				#if isinstance(manip,EigenMatrixSetDofsToZero):
					#print("APPLY MANIP",manip,manip.doflist)
				matJ,matM=manip.apply_on_J_and_M(self,matJ,matM)

		if not _mpi_any_rank(matM.nnz>0): #type:ignore
			raise RuntimeError("The mass matrix has no entries. This likely means that you do not have any time derivatives in your system")

		if not self.problem.is_quiet():
			# Rank 0 only, and the GLOBAL shape: every rank reaching this line would otherwise print its
			# own local row count, which reads like a much smaller problem than the one being solved.
			from ..generic.mpi import get_mpi_rank
			if get_mpi_rank()==0:
				print("Matrices assembled ("+str(n)+" x "+str(n)+"). Invoking eigensolver")
		# Recorded so a caller (in practice tests/mpi_eigen_worker.py) can tell which branch was taken.
		# Under MPI it is agreed across ranks by construction, see the allreduce above.
		self.last_assembly_was_complex=is_complex
		return matJ,matM,n,is_complex


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
