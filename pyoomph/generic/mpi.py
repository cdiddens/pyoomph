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
 
 
import pathlib
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
	from mpi4py import MPI

no_mpi_file=pathlib.Path(__file__).parent.parent.joinpath("NO_MPI").resolve()

if no_mpi_file.exists():
	import sys
	from .. import _pyoomph_core as _pyoomph
	_pyoomph.InitMPI(sys.argv)
	
	def has_mpi()->bool:
		return False

	def get_mpi_world_comm()->'Optional[MPI.Intracomm]':
		return None

	def get_mpi_rank(comm=None)->int: #type:ignore
		return 0 #type:ignore

	def get_mpi_nproc(comm=None)->int: #type:ignore
		return 0 #type:ignore

	def mpi_barrier(comm=None)->None: #type:ignore
		pass #type:ignore

	def mpi_abort(errorcode:int=1)->None: #type:ignore
		import sys as _sys
		_sys.exit(errorcode)

	def get_mpi_sum(value, comm=None): #type:ignore
		return value #type:ignore

	def get_mpi_min(value, comm=None): #type:ignore
		return value #type:ignore

	def get_mpi_max(value, comm=None): #type:ignore
		return value #type:ignore

	def get_mpi_any(flag:bool, comm=None)->bool: #type:ignore
		return bool(flag) #type:ignore

	def get_mpi_any_list(flags, comm=None): #type:ignore
		return [bool(f) for f in flags] #type:ignore

	def get_mpi_bcast(value, root:int=0, comm=None): #type:ignore
		return value #type:ignore

	def mpi_share_root_failure(error:'Optional[BaseException]'=None, root:int=0, context:str="")->None: #type:ignore
		if error is not None:
			raise error

	def mpi_share_any_failure(error:'Optional[BaseException]'=None, context:str="")->None: #type:ignore
		if error is not None:
			raise error
else:
	from mpi4py import MPI #type:ignore
	import sys

	from .. import _pyoomph_core as _pyoomph

	_pyoomph.InitMPI(sys.argv)

	def has_mpi()->bool:
		return True

	def get_mpi_world_comm()->'Optional[MPI.Intracomm]':
		return MPI.COMM_WORLD

	def get_mpi_rank(comm=MPI.COMM_WORLD)->int: #type:ignore
		return comm.Get_rank() #type:ignore

	def get_mpi_nproc(comm=MPI.COMM_WORLD)->int: #type:ignore
		return comm.Get_size() #type:ignore

	def mpi_barrier(comm=MPI.COMM_WORLD)->None: #type:ignore
		comm.barrier() #type:ignore
  
  
	def get_mpi_sum(value, comm=MPI.COMM_WORLD):
		return comm.allreduce(value, op=MPI.SUM) #type:ignore

	def get_mpi_min(value, comm=MPI.COMM_WORLD):
		return comm.allreduce(value, op=MPI.MIN) #type:ignore

	def get_mpi_max(value, comm=MPI.COMM_WORLD):
		return comm.allreduce(value, op=MPI.MAX) #type:ignore

	def get_mpi_any(flag:bool, comm=MPI.COMM_WORLD)->bool: #type:ignore
		return bool(comm.allreduce(bool(flag), op=MPI.LOR)) #type:ignore

	def get_mpi_any_list(flags, comm=MPI.COMM_WORLD): #type:ignore
		# Elementwise OR. The lists must line up on all ranks, i.e. be built from something with a
		# rank-independent order.
		gathered=comm.allgather([bool(f) for f in flags]) #type:ignore
		return [any(votes) for votes in zip(*gathered)] #type:ignore

	def get_mpi_bcast(value, root:int=0, comm=MPI.COMM_WORLD): #type:ignore
		# For data that MUST be identical on every rank but is not by construction - anything drawn
		# from a random number generator, in particular. Collective: every rank has to call it.
		return comm.bcast(value, root=root) #type:ignore

	def mpi_abort(errorcode:int=1)->None: #type:ignore
		# Tears down the entire job, not just this rank. Only for failures a single rank cannot
		# communicate to the others (i.e. inside a section that only one rank executes while the
		# rest already wait in a collective) - raising there would hang the run instead.
		MPI.COMM_WORLD.Abort(errorcode) #type:ignore

	def mpi_share_root_failure(error:'Optional[BaseException]'=None, root:int=0, context:str="")->None: #type:ignore
		if get_mpi_nproc()<=1:
			if error is not None:
				raise error
			return
		# Only the description travels, not the exception object: an arbitrary exception is not
		# guaranteed to be picklable, and a broadcast that fails here would hang exactly the run it
		# is meant to rescue. The rank that saw the failure re-raises the original, so its traceback
		# survives.
		description=None if error is None else (type(error).__name__+": "+str(error))
		description=MPI.COMM_WORLD.bcast(description, root=root) #type:ignore
		if description is None:
			return
		if error is not None:
			raise error
		raise RuntimeError("MPI rank "+str(root)+" failed"+((" while "+context) if context else "")+
						   " ("+description+"). This rank did not run that section itself and raises "
						   "here so that the job ends rather than waiting for a rank that is gone.")

	def mpi_share_any_failure(error:'Optional[BaseException]'=None, context:str="")->None: #type:ignore
		"""Agree on a failure that ANY rank may have seen, and end the job on all of them.

		The rooted mpi_share_root_failure above covers a section that only rank 0 runs. This one is
		for a section every rank runs for itself, where any of them can be the one that fails -
		reading a state file, say. Being collective, it doubles as the barrier such a section needs.
		"""
		if get_mpi_nproc()<=1:
			if error is not None:
				raise error
			return
		# Descriptions only, for the same reason as above: an arbitrary exception need not be
		# picklable, and a gather that fails here would hang the run it is meant to rescue.
		descriptions=MPI.COMM_WORLD.allgather(None if error is None else (type(error).__name__+": "+str(error))) #type:ignore
		if error is not None:
			raise error # the rank that saw it re-raises the original, so its traceback survives
		for r,d in enumerate(descriptions): #type:ignore
			if d is not None:
				raise RuntimeError("MPI rank "+str(r)+" failed"+((" while "+context) if context else "")+
								   " ("+d+"). This rank got through it itself and raises here so that "
								   "the job ends rather than waiting for a rank that is gone.")

	if get_mpi_nproc()>1:
		print("MPI initialized, rank",get_mpi_rank(),"of",get_mpi_nproc())
		if get_mpi_rank()==0:
			import mpi4py
			mpi4py.rc(initialize=False) #type:ignore
			print("MPI config",mpi4py.get_config())


def run_on_rank_zero(func, context:str="")->None:
	"""Run ``func`` on rank 0 only (all ranks in a serial run) and let every rank see the outcome.

	Anything a single rank does while the others wait has to end for all of them or for none: a
	plain ``if get_mpi_rank()==0:`` block that raises unwinds that one rank while the rest sit in
	the next collective forever, which turns a file that could not be written into a job that never
	returns. This runs the section, catches whatever it raises and agrees on it, so the run ends
	with the real error message on every rank instead.

	``context`` describes the section ("writing the mesh file"), for the message the other ranks get.
	"""
	error=None
	if get_mpi_rank()==0 or get_mpi_nproc()<=1:
		try:
			func()
		except BaseException as e:
			error=e
	# Collective, and rooted at 0: no rank leaves before rank 0 has finished the section, so this
	# also does the job of the barrier such a block used to be followed by.
	mpi_share_root_failure(error, context=context)


PYMETIS_MISSING_MESSAGE="PyMetis is not installed, cannot perform graph partitioning for distributed meshes. Please install PyMetis via e.g. 'pip install pymetis'"


def have_pymetis()->bool:
	try:
		import pymetis #type:ignore
		return True
	except ImportError:
		return False


def ensure_pymetis_available()->None:
	"""Raise if PyMetis is missing, to be called on *every* rank before the mesh distribution starts.

	oomph-lib runs the actual partitioning on rank 0 only and scatters the result, so a missing
	PyMetis noticed down there raises on rank 0 while all other ranks are already waiting in the
	following collective - the job then hangs instead of failing. Checking here, where all ranks
	pass, turns that into an ordinary exception on all of them.
	"""
	if not have_pymetis():
		raise ImportError(PYMETIS_MISSING_MESSAGE)
