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
 
from .generic import GenericLinearSystemSolver,GenericEigenSolver,DefaultMatrixType,EigenSolverWhich,SolverError
import scipy #type:ignore
import scipy.linalg #type:ignore
from scipy.sparse import csc_matrix,csr_matrix #type:ignore
import scipy.sparse.linalg #type:ignore

import numpy,numpy.typing

from ..typings import *
if TYPE_CHECKING:
    from ..generic.problem import Problem


class ScipySolverError(SolverError):
	"""SuperLU/UMFPACK could not factorise the matrix.

	A SolverError rather than the plain RuntimeError scipy raises, so that an adaptive time step or an
	arclength step rejects and retries with a smaller one instead of the run ending here.
	"""


def _permutation_is_odd(perm:Any)->bool:
	"""Parity of a permutation, by counting cycles: parity = (n - number of cycles) mod 2.

	A python loop over n, which is O(n) against the O(nnz*fill) of the factorisation whose sign is being
	read, so it costs nothing next to the work already done.
	"""
	perm=numpy.asarray(perm)
	n=int(perm.size)
	seen=numpy.zeros(n,dtype=bool)
	swaps=0
	for i in range(n):
		if seen[i]:
			continue
		j=i
		while not seen[j]:
			seen[j]=True
			j=int(perm[j])
			swaps+=1
		swaps-=1
	return bool(swaps&1)


@GenericLinearSystemSolver.register_solver()
class SuperLUSerial(GenericLinearSystemSolver):
	idname="superlu"
	# scipy's SuperLU/UMFPACK are serial, but oomph routes every mpirun through the distributed entry
	# point, so without this a plain `mpirun -n 2` could not solve at all. The base class gathers the
	# system onto rank 0 and calls solve_serial() below there. solve_serial() issues no collective, as
	# that flag requires.
	gathers_to_root_under_mpi=True
	def __init__(self,problem:"Problem",useUmfpack:bool=False):
		super().__init__(problem)
		scipy.sparse.linalg.use_solver(useUmfpack=useUmfpack) #type:ignore

	def get_determinant_sign(self)->int | None:
		"""From the factorisation the last solve already computed, so this costs nothing.

		SuperLU gives ``P_r A P_c = L U`` with a unit-diagonal L, so the sign is that of the product of
		U's diagonal times the parity of both permutations. Leaving the permutations out - as the
		commented-out determinant print in this file used to - gives a sign that flips whenever pivoting
		reorders, i.e. spurious bifurcation detections.
		"""
		lu=getattr(self,"_current_LU",None)
		if lu is None:
			return None
		try:
			diag=lu.U.diagonal()
		except Exception:
			return None
		if len(diag)==0:
			return None
		if numpy.any(diag==0):
			return 0
		sign=1 if int(numpy.count_nonzero(diag<0))%2==0 else -1
		for perm in (lu.perm_r,lu.perm_c):
			if _permutation_is_odd(perm):
				sign=-sign
		return sign

	def solve_serial(self,op_flag:int,n:int,nnz:int,nrhs:int,values:NPFloatArray,rowind:NPIntArray,colptr:NPIntArray,b:NPFloatArray,ldb:int,transpose:int)->int:
#		print("SOLVING system N=",n,"nnz=",nnz)
		
		if op_flag==1:
			A=csc_matrix((values, rowind, colptr), shape=(n, n))

			#arr=A.toarray()
			#maxzero=0
			#for l in arr:
			#	print(numpy.linalg.norm(l))
			try:
				self._current_LU = scipy.sparse.linalg.splu(A) #type:ignore
			except RuntimeError as re:
				if re.args[0]=="Factor is exactly singular":
					maxn=8000
					if n>maxn:
						print("Singular matrix detected. Nullspace investigation is only done if n="+str(n)+" <= "+str(maxn))
					else:
						print("Singular matrix detected!")
						print("Doing nullspace investigation!")
						nsp=scipy.linalg.null_space(A.todense()) #type:ignore
						print("Nullspace has shape",nsp.shape)
						for k in range(nsp.shape[1]): #type:ignore
							nspv=nsp[:,k] #type:ignore
							maxi=numpy.argsort(numpy.absolute(nspv)) #type:ignore
							maxi=maxi[-1:-10:-1] #type:ignore
							maxvs=numpy.absolute(nspv[maxi]) #type:ignore
							rel=maxvs/maxvs[0]
							crop=numpy.argwhere(rel<0.01) #type:ignore
							if len(crop):
								maxi=maxi[0:crop[0][0]] #type:ignore
							nsplead=nspv[maxi] #type:ignore
							descs=[self.problem.describe_equation(eq) for eq in maxi] #type:ignore
							print(k,nsplead,maxi,":\n\t\t"+"\n\t\t".join(descs)) #type:ignore
					# Re-raised as a SolverError: a singular factorisation is exactly the state an
					# adaptive time step or an arclength step should back away from, and scipy's plain
					# RuntimeError would end the run instead. Any OTHER RuntimeError from splu is a
					# genuine error and keeps its type.
					raise ScipySolverError("SuperLU could not factorise the matrix: "+str(re.args[0])) from re
				raise
#			print("det","DET",self._current_LU.L.diagonal().prod()*self._current_LU.U.diagonal().prod())
		elif op_flag==2:
			self.setup_solver()
			if self.problem._custom_assembler is not None and self.problem._custom_assembler.has_custom_solve_routine():
				sol=self.problem._custom_assembler.custom_solve_routine(lambda rhs : self._current_LU.solve(rhs,"T" if transpose==1 else "N"), b)
			else:
				sol=self._current_LU.solve(b,"T" if transpose==1 else "N") #type:ignore
			b[:]=sol[:] #type:ignore
		else:
			raise RuntimeError("Cannot handle SuperLU mode "+str(op_flag)+" yet")
			return 666
		return 0		#TODO: Return sign of Jacobian

@GenericLinearSystemSolver.register_solver()
class UMFPACKSerial(SuperLUSerial):
	idname="umfpack"
	def __init__(self,problem:"Problem"):
		super().__init__(problem=problem,useUmfpack=True) #type:ignore


class PardisoInvOp(object):
		def __init__(self,A:DefaultMatrixType,M:DefaultMatrixType | None=None):
			self.A = A
			self.M = M

		def __call__(s, b): #type:ignore
			return b  #type:ignore

		matvec = _matvec = dot = __call__  # type:ignore

		@property
		def shape(self):
			return self.A.shape

		@property
		def dtype(self):
			return self.A.dtype






@GenericEigenSolver.register_solver()
class ScipyEigenSolver(GenericEigenSolver):
	idname="scipy"
	def __init__(self,problem:"Problem"):
		super().__init__(problem)
		self.shift=1
		self.ncv:int | None=None
		self.tol=0

	def get_OPInv(self,M:DefaultMatrixType,J:DefaultMatrixType,shift:float | complex)->object | None:
		return None

	def distributed_possible(self) -> bool:
		# ARPACK through scipy sees one process' matrices only, so on a distributed problem it would
		# solve each rank's row block as if it were the whole eigenproblem. It can still be used there,
		# because solve() below gathers the blocks onto rank 0 first -- serialised, but correct. SLEPc
		# remains the way to do this in parallel.
		return True

	def _solve_gathered_on_root(self,neval:int,**kwargs:Any)->tuple[NPComplexArray,NPComplexArray,DefaultMatrixType,DefaultMatrixType]:
		"""Solve a distributed eigenproblem by gathering J and M onto rank 0.

		Under --distribute each rank assembles only its own (nrow_local x n) row block, which ARPACK
		cannot work with. Gathering them into one square matrix on rank 0 and solving there is not
		parallel -- SLEPc is, and is preferred -- but it is correct, and it is the difference between
		scipy/ARPACK being usable under --distribute and not being usable at all.
		"""
		from ..generic.mpi import (get_mpi_rank,get_mpi_world_comm,mpi_row_layout,mpi_gather_csr_rows,
								   mpi_share_root_failure)
		# Collective, and it must run on every rank: it is what assembles the local blocks.
		J,M,n,_is_complex=self.get_J_M_n_and_type()
		_n,nrow_local,first_row,_distributed=self.get_eigen_row_layout()

		def _gather(mat:DefaultMatrixType):
			layout=mpi_row_layout(n,first_row,nrow_local,mat.nnz)
			assert layout is not None
			got=mpi_gather_csr_rows(layout,mat.data,mat.indices,mat.indptr,
									context="gathering the eigenproblem onto rank 0")
			if got is None:
				return None
			vals,cols,rs=got
			return csr_matrix((vals,cols,rs),shape=(n,n))

		Jg,Mg=_gather(J),_gather(M)
		evals:Any=None
		evects:Any=None
		error:BaseException | None=None
		if get_mpi_rank()==0:
			assert Jg is not None and Mg is not None
			try:
				# get_J_M_n_and_type() skips the matrix manipulators when the problem is distributed --
				# they rewrite whole rows of a square global matrix, which is not what a rank holds
				# there. Here it is, so they apply.
				for manip in self.matrix_manipulators:
					Jg,Mg=manip.apply_on_J_and_M(self,Jg,Mg)
				evals,evects,_,_=self.solve(neval,custom_J_and_M=(Jg,Mg),**kwargs)
			except BaseException as e:
				error=e
		# Ends the job on every rank if rank 0 failed, rather than leaving them in the broadcast below.
		mpi_share_root_failure(error,context="solving the gathered eigenproblem on rank 0")
		comm=get_mpi_world_comm()
		assert comm is not None
		# Eigenvectors are contractually replicated at full global length on every rank (see
		# Problem.rotate_eigenvectors), so a broadcast is exactly the right shape here.
		evals,evects=comm.bcast((evals,evects) if get_mpi_rank()==0 else None,root=0) #type:ignore
		# The LOCAL blocks are returned, not the gathered ones: the caller's accuracy report multiplies
		# them by a globally replicated eigenvector and reduces over the ranks, which only works if each
		# rank contributes its own rows.
		return evals,evects,J,M

	def solve(self,neval:int,shift:float | complex | None=None,sort:bool=True,which:EigenSolverWhich="LM",OPpart:Literal["r", "i"] | None=None,v0:NPComplexArray | NPFloatArray | None=None,target:complex | None=None,custom_J_and_M:tuple[DefaultMatrixType,DefaultMatrixType] | None=None,with_left_eigenvectors:bool=False,quiet:bool=True)->tuple[NPComplexArray,NPComplexArray,DefaultMatrixType,DefaultMatrixType]:
		# custom_J_and_M is always a whole global matrix, so it never needs gathering -- and the
		# gathered path itself comes back through here with one.
		if custom_J_and_M is None and self.get_eigen_row_layout()[3]:
			return self._solve_gathered_on_root(neval,shift=shift,sort=sort,which=which,OPpart=OPpart,
												v0=v0,target=target,
												with_left_eigenvectors=with_left_eigenvectors,quiet=quiet)
		if shift is None:
			shift=self.shift
		if target is not None:
			raise RuntimeError("implement target for this eigensolver")
		self.problem._set_solved_residual(self.real_contribution,True,False)
		
		if with_left_eigenvectors:
			raise RuntimeError("Implement with_left_eigenvectors")    
        
		if custom_J_and_M is not None:
			J,M=custom_J_and_M
			n=J.shape[0]
		else:
			J,M,n,_=self.get_J_M_n_and_type()
		
		asym = abs(M - M.T)
		rel_asym = abs(M - M.T).max()/abs(M).max()
		if rel_asym>1e-7:
			print("WARNING: Mass matrix is asymmetric! The scipy eigensolver does not support that! Eigenvalues/vectors will be wrong!\nConsider switching to SLEPc! Max asymmetry:", asym.max(), "vs max |M|:", abs(M).max())
		

		if neval <= 0:
			neval=n

		if neval>=n-1:
			
			evals,evects=scipy.linalg.eig(J.toarray(),b=M.toarray(),left=False) #type:ignore			
			if sort:
				if target:
					srt = numpy.argsort(numpy.abs(evals-target))
				else:
					srt=numpy.argsort(-evals)[0:min(neval,n)] #type:ignore
				infcrop=numpy.argmax(numpy.isfinite((evals[srt[:]]))) #type:ignore
				srt=srt[infcrop:] #type:ignore
				#evals,evects=evals[srt],numpy.transpose(evects)[srt]
				evals= evals[srt] #type:ignore
				evects=evects[:,srt] #type:ignore
			evects=numpy.transpose(evects) #type:ignore
			evals=cast(NPComplexArray,evals)
			evects=cast(NPComplexArray,evects)
			return evals,evects,J,M
		else:
			OPInv=self.get_OPInv(M,J,shift)
			ncv=self.ncv
			max_retries=3
			evals=None
			evects=None
			for attempt in range(max_retries+1):
				try:
					evals,evects=scipy.sparse.linalg.eigs(J,M=M,sigma=shift,return_eigenvectors=True,k=neval,OPinv=OPInv,which=which,OPpart=OPpart,v0=v0,ncv=ncv,tol=self.tol) #type:ignore
					break
				except scipy.sparse.linalg.ArpackError:
					if attempt>=max_retries:
						raise
					ncv=max(2*neval+1,20,ncv*2 if ncv is not None else 0)
					ncv=min(ncv,n)
					if not quiet:
						print("ARPACK failed to converge, retrying with ncv="+str(ncv)+" (attempt "+str(attempt+1)+"/"+str(max_retries)+")")
			assert evals is not None and evects is not None
			if sort:
				if target:
					srt = numpy.argsort(numpy.abs(evals-target))
				else:
					srt = numpy.argsort(-evals)[0:min(neval, n)] #type:ignore
     
				infcrop = numpy.argmax(numpy.isfinite((evals[srt[:]]))) #type:ignore
				srt = srt[infcrop:] #type:ignore
				evals = evals[srt] #type:ignore
				evects = evects[:, srt] #type:ignore
			evects = numpy.transpose(evects) #type:ignore
			evals=cast(NPComplexArray,evals)
			evects=cast(NPComplexArray,evects)

			return evals, evects,J,M
			#return evals,numpy.transpose(evects)


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
