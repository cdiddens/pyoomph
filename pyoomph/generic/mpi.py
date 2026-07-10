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

no_mpi_file=pathlib.Path(__file__).parent.parent.joinpath("NO_MPI").resolve()

if no_mpi_file.exists():
	import sys
	import _pyoomph
	_pyoomph.InitMPI(sys.argv)
	
	def has_mpi():
		return False
		
	def get_mpi_world_comm():
		return None

	def get_mpi_rank(comm=None)->int: #type:ignore
		return 0 #type:ignore

	def get_mpi_nproc(comm=None)->int: #type:ignore
		return 0 #type:ignore

	def mpi_barrier(comm=None)->None: #type:ignore
		pass #type:ignore
else:
	from mpi4py import MPI #type:ignore
	import sys

	import _pyoomph

	_pyoomph.InitMPI(sys.argv)

	def has_mpi():
		return True
		
	def get_mpi_world_comm():
		return MPI.COMM_WORLD

	def get_mpi_rank(comm=MPI.COMM_WORLD)->int: #type:ignore
		return comm.Get_rank() #type:ignore

	def get_mpi_nproc(comm=MPI.COMM_WORLD)->int: #type:ignore
		return comm.Get_size() #type:ignore

	def mpi_barrier(comm=MPI.COMM_WORLD)->None: #type:ignore
		comm.barrier() #type:ignore
  
  
	def get_mpi_sum(value, comm=MPI.COMM_WORLD):
		return comm.allreduce(value, op=MPI.SUM) #type:ignore
  
	if get_mpi_nproc()>1:
		print("MPI initialized, rank",get_mpi_rank(),"of",get_mpi_nproc())
		if get_mpi_rank()==0:
			import mpi4py
			mpi4py.rc(initialize=False) #type:ignore
			print("MPI config",mpi4py.get_config())
