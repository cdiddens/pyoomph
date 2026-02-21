#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2025  Christian Diddens & Duarte Rocha
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
 
from .generic import GenericLinearSystemSolver,GenericEigenSolver,DefaultMatrixType,EigenSolverWhich
import scipy #type:ignore
import scipy.linalg #type:ignore
from scipy.sparse import csc_matrix #type:ignore
import scipy.sparse.linalg #type:ignore

import numpy,numpy.typing

from ..typings import *
if TYPE_CHECKING:
    from ..generic.problem import Problem


@GenericLinearSystemSolver.register_solver()
class SuperLUSerial(GenericLinearSystemSolver):
	idname="superlu"
	def __init__(self,problem:"Problem",useUmfpack:bool=False):
		super().__init__(problem)
		scipy.sparse.linalg.use_solver(useUmfpack=useUmfpack) #type:ignore

	def solve_serial(self,op_flag:int,n:int,nnz:int,nrhs:int,values:NPFloatArray,rowind:NPIntArray,colptr:NPIntArray,b:NPFloatArray,ldb:int,transpose:int)->int:
#		print("SOLVING system N=",n,"nnz=",nnz)
		
		if op_flag==1:
			A=csc_matrix((values, rowind, colptr), shape=(n, n))

			if False:
				lu=splu(A)
				diagL = lu.L.diagonal()
				diagU = lu.U.diagonal()
				diagL = diagL.astype(numpy.complex128)
				diagU = diagU.astype(numpy.complex128)
				logdet = numpy.log(diagL).sum() + numpy.log(diagU).sum()
				determ=numpy.exp(logdet)
				print("MATERIX DET",determ)
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

	def distributed_possible(self) -> bool:
		return False

	def solve_distributed(self, op_flag: int, allow_permutations: int, n: int, nnz_local: int, nrow_local: int, first_row: int, values: NPFloatArray, col_index: NPIntArray, row_start: NPIntArray, b: NPFloatArray, nprow: int, npcol: int, doc: int, data: NPUInt64Array, info: NPIntArray)->None:
		raise RuntimeError("SuperLU solver cannot solve distributed")

@GenericLinearSystemSolver.register_solver()
class UMFPACKSerial(SuperLUSerial):
	idname="umfpack"
	def __init__(self,problem:"Problem"):
		super().__init__(problem=problem,useUmfpack=True) #type:ignore


class PardisoInvOp(object):
		def __init__(self,A:DefaultMatrixType,M:Optional[DefaultMatrixType]=None):
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
		self.ncv:Optional[int]=None
		self.tol=0

	def get_OPInv(self,M:DefaultMatrixType,J:DefaultMatrixType,shift:Union[float,complex])->Optional[object]:
		return None


	def solve(self,neval:int,shift:Optional[Union[float,complex]]=None,sort:bool=True,which:EigenSolverWhich="LM",OPpart:Optional[Literal["r","i"]]=None,v0:Optional[Union[NPComplexArray,NPFloatArray]]=None,target:Optional[complex]=None,custom_J_and_M:Optional[Tuple[DefaultMatrixType]]=None,with_left_eigenvectors:bool=False,quiet:bool=True)->Tuple[NPComplexArray,NPComplexArray,DefaultMatrixType,DefaultMatrixType]:
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
			evals,evects=scipy.sparse.linalg.eigs(J,M=M,sigma=shift,return_eigenvectors=True,k=neval,OPinv=OPInv,which=which,OPpart=OPpart,v0=v0,ncv=self.ncv,tol=self.tol) #type:ignore
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

