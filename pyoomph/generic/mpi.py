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
 
 
import os
import pathlib
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
	from mpi4py import MPI

no_mpi_file=pathlib.Path(__file__).parent.parent.joinpath("NO_MPI").resolve()

# Environment variables through which the common launchers announce the size of the run, checked
# BEFORE anything MPI-related is touched: in a build without MPI there is no communicator to ask, and
# every rank would otherwise happily believe it is the only process there is. One per launcher family,
# since a job may well be started by a launcher whose runtime is not the one pyoomph was built for.
# SLURM is deliberately represented by SLURM_STEP_NUM_TASKS rather than SLURM_NTASKS: the latter is
# set for the whole allocation, so a plain serial `python` inside an `sbatch --ntasks=4` script would
# look like a 4-rank run, while the step variable only appears for a step actually launched by srun.
_MPI_LAUNCHER_SIZE_ENV = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE",
                          "SLURM_STEP_NUM_TASKS", "PALS_NRANKS")
# Same launchers, their rank variables. Only used to keep the refusal below from printing the same
# paragraph once per rank.
_MPI_LAUNCHER_RANK_ENV = ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "MV2_COMM_WORLD_RANK",
                          "SLURM_PROCID", "PALS_RANKID")


def _launcher_rank() -> int:
	"""The rank an MPI launcher advertises, or 0 if it does not say (then everyone reports)."""
	for name in _MPI_LAUNCHER_RANK_ENV:
		try:
			return int(os.environ.get(name, ""))
		except ValueError:
			continue
	return 0


def _launcher_world_size() -> "tuple[str,int] | None":
	"""The (variable, size) an MPI launcher advertises, or None if this is not a launched run."""
	for name in _MPI_LAUNCHER_SIZE_ENV:
		val = os.environ.get(name, "")
		try:
			size = int(val)
		except ValueError:
			continue
		if size > 1:
			return name, size
	return None


if no_mpi_file.exists():
	import sys
	# Refuse a multi-rank launch of a build that has no MPI. Without this the run does not fail, which
	# is the problem: every rank runs the WHOLE simulation, none of them aware of the others, all of
	# them writing into the same output directory. That looks like a parallel run, takes as long as the
	# serial one, and leaves output files written by several processes at once.
	_launched = _launcher_world_size()
	if _launched is not None and os.environ.get("PYOOMPH_ALLOW_SERIAL_UNDER_MPIRUN", "") in ("", "0", "off", "false", "False"):
		_var, _size = _launched
		if _launcher_rank() != 0:
			# The full explanation once is enough; N copies of it bury the one that is read.
			raise RuntimeError("pyoomph was compiled WITHOUT MPI support and cannot be run with "
							   + str(_size) + " ranks -- see the message from rank 0.")
		raise RuntimeError(
			"pyoomph was compiled WITHOUT MPI support, but this process was started by an MPI launcher "
			"with " + str(_size) + " ranks (" + _var + "=" + str(_size) + ").\n"
			"Each rank would run the entire simulation on its own, unaware of the others, and all of them "
			"would write into the same output directory: " + str(_size) + " identical runs, no speed-up, "
			"and output files written by several processes at once.\n"
			"To run in parallel, build pyoomph from source with MPI support (the wheels on PyPI are built "
			"without it) and install mpi4py:\n"
			"    python -m pip install --no-build-isolation -e . "
			"--config-settings=cmake.define.PYOOMPH_USE_MPI=ON\n"
			"See https://pyoomph.readthedocs.io/en/latest/tutorial/parallel/mpi.html for the details, and "
			"`python -m pyoomph check mpi` to verify the result.\n"
			"If you meant to start " + str(_size) + " INDEPENDENT serial runs, set "
			"PYOOMPH_ALLOW_SERIAL_UNDER_MPIRUN=1 -- each rank must then be given its own output directory.")
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

	def get_mpi_elementwise_max(arr, comm=None): #type:ignore
		return arr #type:ignore

	def get_mpi_bcast(value, root:int=0, comm=None): #type:ignore
		return value #type:ignore

	def mpi_share_root_failure(error:'Optional[BaseException]'=None, root:int=0, context:str="")->None: #type:ignore
		if error is not None:
			raise error

	def mpi_share_any_failure(error:'Optional[BaseException]'=None, context:str="")->None: #type:ignore
		if error is not None:
			raise error

	def install_mpi_abort_excepthook()->None: #type:ignore
		pass

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

	def get_mpi_elementwise_max(arr, comm=MPI.COMM_WORLD): #type:ignore
		# Elementwise MAX over a numpy array of the same length and meaning on every rank - indexed
		# by something rank-independent, a global equation number say. Reduced in place (and the
		# array returned), so the caller must not hand in a view it still needs unreduced.
		import numpy
		out=numpy.ascontiguousarray(arr) #type:ignore
		comm.Allreduce(MPI.IN_PLACE, out, op=MPI.MAX) #type:ignore
		return out #type:ignore

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

	_abort_excepthook_installed=[False]

	def install_mpi_abort_excepthook()->None: #type:ignore
		"""Make an uncaught exception on ANY rank end the job, instead of hanging it.

		A rank that dies on its own leaves the others exactly where they were, which is almost always
		inside a collective they will now wait in forever. Aborting is the only way out: agreeing on
		anything first would need the very collective the dead rank is never going to call - the same
		reason mpi_abort() exists for the rank-0-only sections.

		This is what the nightly hit on 2026-08-18. One rank raised inside Problem.output() while the
		other three sat in a collective, and because OpenMPI polls rather than sleeps, the job did not
		fail - it spun at 100% CPU on four cores for three hours, until the tutorial step's 8h timeout
		would have killed it. A deadlock and real work look identical in top; the run had reported
		"simulation time: 3.07 s" for the script three hours earlier.

		The traceback is printed and both streams flushed BEFORE the abort: Abort() does not unwind
		and does not flush, so anything still buffered dies with the job - which is what would make
		this hard to diagnose from a log. Chained rather than replaced, so an application that has
		installed its own hook keeps it. SystemExit never reaches sys.excepthook, so a plain
		sys.exit() is unaffected.
		"""
		if _abort_excepthook_installed[0] or get_mpi_nproc()<=1:
			return
		_abort_excepthook_installed[0]=True
		_previous_hook=sys.excepthook
		def _abort_hook(exctype,value,tb): #type:ignore
			try:
				_previous_hook(exctype,value,tb)
				sys.stderr.write("pyoomph: uncaught "+exctype.__name__+" on MPI rank "+str(get_mpi_rank())+
								 " of "+str(get_mpi_nproc())+" -- aborting the whole job. The other ranks "
								 "are waiting in a collective that this rank will now never reach, so "
								 "letting it exit alone would leave them spinning rather than failing.\n")
				sys.stderr.flush()
				sys.stdout.flush()
			finally:
				mpi_abort(1)
		sys.excepthook=_abort_hook

	if get_mpi_nproc()>1:
		install_mpi_abort_excepthook()
		# Before the banner, not after: the banner is the first thing that would otherwise be
		# printed once per rank.
		from .logging import setup_mpi_console as _setup_mpi_console
		_mode=_setup_mpi_console(get_mpi_rank(),get_mpi_nproc())
		if get_mpi_rank()==0:
			import mpi4py
			mpi4py.rc(initialize=False) #type:ignore
			print("MPI initialized with "+str(get_mpi_nproc())+" ranks, config "+str(mpi4py.get_config()))
			if _mode=="condensed":
				print("  (showing rank 0 only; use --mpi-output=all for every rank, =off for the raw output)")


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


# How long a rank waiting for another one polls before it starts sleeping, and how long it sleeps at
# most once it does. The spin phase matters because time.sleep() cannot resolve better than roughly
# 60 us on Linux, which would dominate the thousands of short collectives a small problem issues; the
# 5 ms cap bounds the wake-up delay after a long factorisation to something negligible while leaving
# the rank at ~200 wakeups/s, i.e. no measurable CPU at all.
_MPI_IDLE_SPIN_CALLS=int(os.environ.get("PYOOMPH_MPI_IDLE_SPIN","2000"))
_MPI_IDLE_MIN_SLEEP=1e-4
_MPI_IDLE_MAX_SLEEP=float(os.environ.get("PYOOMPH_MPI_IDLE_MAX_SLEEP","5e-3"))
# Printed once, and only to say that a rank has been waiting long enough that something is probably
# wrong. Deliberately NOT a timeout that raises: raising on one rank is the deadlock it would be
# trying to report.
_MPI_IDLE_WARN_AFTER=float(os.environ.get("PYOOMPH_MPI_IDLE_WARN_AFTER","600"))
_mpi_idle_warned:set[str]=set()


def mpi_wait_idle(request, context:str="")->None:
	"""Wait for a non-blocking MPI ``request`` by polling, giving the CPU up in between.

	MPI's blocking calls busy-poll: with the default ``mpi_yield_when_idle=0``, a rank sitting in
	``Gatherv`` or ``Allreduce`` keeps a core at 100%. That is fine for the short, balanced
	collectives a distributed solver issues, and quite wrong for a gather-to-root solve, where N-1
	ranks wait out a whole factorisation on rank 0 - burning exactly the cores that rank 0's
	OpenMP/MKL threads were supposed to get.

	Spins briefly first (Test() is also what drives progress on a build without a progress thread,
	so it is not wasted), then backs off to sleeping. time.sleep releases the GIL and the ranks are
	single-threaded, so nothing else needs it.
	"""
	if request.Test():
		return
	for _ in range(_MPI_IDLE_SPIN_CALLS):
		if request.Test():
			return
	import time
	delay=_MPI_IDLE_MIN_SLEEP
	waited=0.0
	while True:
		time.sleep(delay)
		if request.Test():
			return
		waited+=delay
		delay=min(2*delay,_MPI_IDLE_MAX_SLEEP)
		if waited>_MPI_IDLE_WARN_AFTER and context and context not in _mpi_idle_warned:
			_mpi_idle_warned.add(context)
			print("NOTE: MPI rank "+str(get_mpi_rank())+" has been waiting "+str(int(waited))+
				  " s for "+context+". If the run never finishes, this is where it stopped: some rank "
				  "did not reach the matching collective.",flush=True)


class MPIRowLayout:
	"""Who owns which rows of a row-partitioned vector/matrix, and where their nonzeros go.

	Built once per factorisation from one allgather, then reused for the right-hand side and the
	solution, which share the row split. Every field is derived from allgathered data, so it is
	identical on every rank - which is what makes it safe to branch on.
	"""

	__slots__=("n","nproc","vec_counts","vec_displs","nnz_counts","nnz_displs","nnz_total")

	def __init__(self,n:int,vec_counts,vec_displs,nnz_counts,nnz_displs,nnz_total:int):
		self.n=n
		self.nproc=len(vec_counts)
		self.vec_counts=vec_counts
		self.vec_displs=vec_displs
		self.nnz_counts=nnz_counts
		self.nnz_displs=nnz_displs
		self.nnz_total=nnz_total


def mpi_row_layout_from_gathered(entries,n:int)->MPIRowLayout:
	"""Build an MPIRowLayout from the allgathered ``(first_row, nrow_local, nnz_local)`` per rank.

	Split out from the collective so it can be tested with fabricated inputs and no mpirun - the
	offsets are the error-prone part of a gather, and a wrong one produces a plausible matrix rather
	than a crash.

	The blocks are ordered by ``first_row``, never by rank: oomph hands out ascending first_row for
	its own uniform split, but a distributed problem's dof distribution need not, and a rank owning
	no rows at all has the same first_row as its successor.
	"""
	import numpy
	nproc=len(entries)
	# Ties can only involve a rank with nrow_local==0, which contributes no nonzeros either, so
	# breaking them by rank cannot move any other rank's offset.
	order=sorted(range(nproc),key=lambda r:(entries[r][0],r))
	nnz_displs=[0]*nproc
	nnz_total=0
	for r in order:
		nnz_displs[r]=nnz_total
		nnz_total+=int(entries[r][2])
	expect=0
	for r in order:
		if int(entries[r][0])!=expect:
			raise RuntimeError("The MPI ranks' row blocks do not tile [0,"+str(n)+") - rank "+str(r)+
							   " starts at "+str(entries[r][0])+" where "+str(expect)+" was expected. "
							   "The ranks disagree about the row distribution of the linear system.")
		expect+=int(entries[r][1])
	if expect!=n:
		raise RuntimeError("The MPI ranks' row blocks cover "+str(expect)+" rows, but the system has "+
						   str(n)+".")
	# The gathered row_start array is int32 (that is what oomph hands out), so the offsets have to fit.
	# Over the limit they wrap into negative values and describe a different, entirely plausible matrix.
	if nnz_total>=2**31-1:
		raise RuntimeError("The gathered system has "+str(nnz_total)+" nonzeros, which does not fit the "
						   "int32 row-start indices used here. Use a natively distributed solver "
						   "(petsc_mumps) for a system this size.")
	return MPIRowLayout(n,
						numpy.array([int(e[1]) for e in entries],dtype=numpy.int32),
						numpy.array([int(e[0]) for e in entries],dtype=numpy.int32),
						numpy.array([int(e[2]) for e in entries],dtype=numpy.int32),
						numpy.array(nnz_displs,dtype=numpy.int32),
						nnz_total)


def mpi_row_layout(n:int,first_row:int,nrow_local:int,nnz_local:int,comm=None)->'Optional[MPIRowLayout]':
	"""Collective: agree on the row layout of a distributed matrix. None on a single process."""
	if get_mpi_nproc()<=1:
		return None
	if comm is None:
		comm=get_mpi_world_comm()
	assert comm is not None
	entries=comm.allgather((int(first_row),int(nrow_local),int(nnz_local))) #type:ignore
	return mpi_row_layout_from_gathered(entries,n)


def mpi_gather_csr_rows(layout:MPIRowLayout,values,col_index,row_start,root:int=0,comm=None,context:str=""):
	"""Gather the local CSR row blocks into one globally indexed CSR triple on ``root``.

	Returns ``(values, col_index, row_start)`` on ``root`` and ``None`` everywhere else. The column
	indices of a distributed CRDoubleMatrix are already global, so only the rows have to be stitched.

	Typed Gatherv rather than the pickling ``comm.gather``: this runs once per Newton step on the
	largest object in the solve. Pickling would copy the block into a bytes object, unpickle it into
	a second array on the root and concatenate into a third - three copies of the whole matrix, on
	the one rank that is already the memory bottleneck.
	"""
	import numpy
	from mpi4py import MPI #type:ignore
	if comm is None:
		comm=get_mpi_world_comm()
	assert comm is not None
	rank=get_mpi_rank()
	n,nnz_total=layout.n,layout.nnz_total
	nrow_local=int(layout.vec_counts[rank])
	# The eigenproblem matrices are complex whenever an imaginary contribution was assembled, so the
	# element type follows the data rather than being assumed real.
	vdtype=numpy.dtype(numpy.asarray(values).dtype)
	if vdtype==numpy.complex128:
		vmpi=MPI.C_DOUBLE_COMPLEX
	else:
		vdtype=numpy.dtype(numpy.float64)
		vmpi=MPI.DOUBLE
	vals_g=numpy.empty(nnz_total,dtype=vdtype) if rank==root else None
	cols_g=numpy.empty(nnz_total,dtype=numpy.int32) if rank==root else None
	rs_g=numpy.empty(n+1,dtype=numpy.int32) if rank==root else None
	# Offset this rank's row-start heads by its own nonzero displacement before sending, so the root
	# can drop them straight into place instead of walking every block again afterwards. Kept in a
	# local until the request completes: it is the send buffer, and CPython would otherwise be free
	# to collect it while MPI is still reading it.
	rs_local=numpy.asarray(row_start[:nrow_local],dtype=numpy.int32)+layout.nnz_displs[rank]
	vals_l=numpy.asarray(values,dtype=vdtype)
	cols_l=numpy.asarray(col_index,dtype=numpy.int32)
	for send,recv,counts,displs,mtype in (
			(vals_l,vals_g,layout.nnz_counts,layout.nnz_displs,vmpi),
			(cols_l,cols_g,layout.nnz_counts,layout.nnz_displs,MPI.INT),
			(rs_local,rs_g,layout.vec_counts,layout.vec_displs,MPI.INT)):
		req=comm.Igatherv(send,[recv,counts,displs,mtype] if rank==root else None,root=root) #type:ignore
		mpi_wait_idle(req,context or "gathering the matrix onto rank "+str(root))
	if rank!=root:
		return None
	assert vals_g is not None and cols_g is not None and rs_g is not None # the receive buffers, allocated on the root
	rs_g[n]=nnz_total
	return vals_g,cols_g,rs_g


def mpi_gather_vector(layout:MPIRowLayout,b,root:int=0,comm=None,context:str=""):
	"""Gather the local row blocks of a vector into one global array on ``root`` (None elsewhere)."""
	import numpy
	from mpi4py import MPI #type:ignore
	if comm is None:
		comm=get_mpi_world_comm()
	assert comm is not None
	rank=get_mpi_rank()
	out=numpy.empty(layout.n,dtype=numpy.float64) if rank==root else None
	send=numpy.asarray(b,dtype=numpy.float64)
	req=comm.Igatherv(send,[out,layout.vec_counts,layout.vec_displs,MPI.DOUBLE] if rank==root else None,root=root) #type:ignore
	mpi_wait_idle(req,context or "gathering the right-hand side onto rank "+str(root))
	return out


def mpi_scatter_vector(layout:MPIRowLayout,x,out,root:int=0,comm=None,context:str=""):
	"""Scatter a global vector held on ``root`` back into each rank's own row block ``out``."""
	import numpy
	from mpi4py import MPI #type:ignore
	if comm is None:
		comm=get_mpi_world_comm()
	assert comm is not None
	rank=get_mpi_rank()
	send=None if rank!=root else numpy.asarray(x,dtype=numpy.float64)
	local=numpy.empty(int(layout.vec_counts[rank]),dtype=numpy.float64)
	req=comm.Iscatterv([send,layout.vec_counts,layout.vec_displs,MPI.DOUBLE] if rank==root else None,local,root=root) #type:ignore
	mpi_wait_idle(req,context or "scattering the solution from rank "+str(root))
	out[:]=local[:]
	return out


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
