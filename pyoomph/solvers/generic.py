
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
 
from ..meshes.mesh import ODEStorageMesh
from ..typings import *
import numpy


import scipy.sparse #type:ignore
import traceback

DefaultMatrixType=scipy.sparse.csr_matrix
_TypeGenericLASolver=TypeVar("_TypeGenericLASolver",bound=Type["GenericLinearSystemSolver"])
_TypeGenericEigenSolver=TypeVar("_TypeGenericEigenSolver",bound=Type["GenericEigenSolver"])

CoreLinearSolverEnum=Literal["superlu","umfpack","petsc","mumps","pardiso"]
CoreEigenSolverEnum=Literal["scipy","pardiso","slepc"]
EigenSolverWhich=Literal["LM","SM","LR","SR","SI"]
_default_la_solver:Optional[Union["GenericLinearSystemSolver",CoreLinearSolverEnum]]=None
_default_eigen_solver:Optional[Union["GenericEigenSolver",CoreEigenSolverEnum]]=None

if TYPE_CHECKING:
    from ..generic.problem import Problem

def set_default_linear_solver(solv:Union["GenericLinearSystemSolver",CoreLinearSolverEnum]):
	global _default_la_solver
	_default_la_solver=solv

def get_default_linear_solver()->Optional[Union["GenericLinearSystemSolver",CoreLinearSolverEnum]]:
	return _default_la_solver


def set_default_eigen_solver(solv:Union["GenericEigenSolver",CoreEigenSolverEnum]):
	global _default_eigen_solver
	_default_eigen_solver=solv

def get_default_eigen_solver()->Optional[Union["GenericEigenSolver",CoreEigenSolverEnum]]:
	return _default_eigen_solver

class GenericLinearSystemSolver:
	_registered_solvers:Dict[str,Type["GenericLinearSystemSolver"]]={}
	idname:str

	def __init__(self,problem:"Problem"):
		self.problem=problem

	def setup_solver(self)->None:
		pass

	def solve_distributed(self, op_flag: int, allow_permutations: int, n: int, nnz_local: int, nrow_local: int, first_row: int, values: NPFloatArray, col_index: NPIntArray, row_start: NPIntArray, b: NPFloatArray, nprow: int, npcol: int, doc: int, data: NPUInt64Array, info: NPIntArray)->None:
		raise RuntimeError("SuperLU solver cannot solve distributed")

	def solve_serial(self,op_flag:int,n:int,nnz:int,nrhs:int,values:NPFloatArray,rowind:NPIntArray,colptr:NPIntArray,b:NPFloatArray,ldb:int,transpose:int)->int:
		raise NotImplementedError("You need to specialise the function 'solve_serial'")

	
	def distributed_possible(self)->bool:
		return True

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
			try:
				import importlib
				#__import__(name)
				importlib.import_module("pyoomph.solvers."+name)
				if name in GenericLinearSystemSolver._registered_solvers.keys():
					return GenericLinearSystemSolver._registered_solvers[name](problem)
				else:
					raise RuntimeError("Unknown Linear Algebra solver: '"+name+"'. Following are defined (and included): "+str(list(GenericLinearSystemSolver._registered_solvers.keys())))
			except Exception as e:
				add_msg_str="When trying to import pyoomph.solvers."+str(name)+", the following error was raised:\n"
				add_msg_str+=''.join(traceback.format_exception(type(e), value=e, tb=e.__traceback__))
				raise RuntimeError("Unknown Linear Algebra solver: '"+name+"'. Following are defined (and included): "+str(list(GenericLinearSystemSolver._registered_solvers.keys()))+"\n"+add_msg_str)


	def set_num_threads(self,nthreads:Optional[int]) -> None:
		pass

##########



class EigenMatrixManipulatorBase:
	def __init__(self,problem:"Problem") -> None:
		super(EigenMatrixManipulatorBase, self).__init__()
		self.problem=problem

	def resolve_equations_by_name(self,name:str) -> Set[int]:
		from ..generic.problem import Problem
		import _pyoomph
		splt=name.split("/")
		root=self.problem
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
			raise RuntimeError("Cannot find field "+str(fieldname)+" in mesh "+root.get_full_name())
		res:Set[int]=set()
		if  isinstance(root,ODEStorageMesh):
			ode = root._get_ODE("ODE")
			_, inds = ode.to_numpy()
			if not fieldname in inds.keys():
				raise RuntimeError("Cannot get the field '"+fieldname+"' on ODE domain "+root.get_full_name())
			eqn=ode.internal_data_pt(inds[fieldname]).eqn_number(0)
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

	def apply_on_J_and_M(self,solver:"GenericEigenSolver",J:DefaultMatrixType,M:DefaultMatrixType)->Tuple[DefaultMatrixType,DefaultMatrixType]:
		return J,M


class EigenMatrixSetDofsToZero(EigenMatrixManipulatorBase):
	def __init__(self,problem:"Problem",*doflist:str):
		super(EigenMatrixSetDofsToZero, self).__init__(problem)
		self.doflist=set(doflist)
		self.zeromap:Set[int]=set()


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

	def apply_on_J_and_M(self,solver:"GenericEigenSolver",J:DefaultMatrixType,M:DefaultMatrixType) -> Tuple[DefaultMatrixType, DefaultMatrixType]:
		self.zeromap:Set[int]=set()
		import _pyoomph
		for d in self.doflist:
			if isinstance(d,str):
				eqs=self.resolve_equations_by_name(d)
			else:
				eqs=set([d])
			if  _pyoomph.get_verbosity_flag() != 0:
				print("INFO ",d,eqs)
			self.zeromap=self.zeromap.union(eqs)
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
		


	def apply_on_J_and_M___OLD(self,solver:"GenericEigenSolver",J:DefaultMatrixType,M:DefaultMatrixType) -> Tuple[DefaultMatrixType, DefaultMatrixType]:
		# TODO OLD VERSION: Slow, remove
		import _pyoomph
		self.zeromap:Set[int]=set()
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
	_registered_solvers:Dict[str,Type["GenericEigenSolver"]]={}
	idname:str
	def __init__(self,problem:"Problem"):
		self.problem=problem
		self.matrix_manipulators:List[EigenMatrixManipulatorBase]=[]
		self.real_contribution:str=""
		self.imag_contribution:Optional[str]=None

	def supports_target(self)->bool:
		return False

	def setup_matrix_contributions(self,real_contribution:str,imag_contribution:Optional[str]=None):
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

	def solve(self,neval:int,shift:Optional[Union[float,complex]]=None,sort:bool=True,which:EigenSolverWhich="LM",OPpart:Optional[Literal["r","i"]]=None,v0:Optional[Union[NPComplexArray,NPFloatArray]]=None,target:Optional[complex]=None,custom_J_and_M:Optional[Tuple[DefaultMatrixType]]=None,with_left_eigenvectors:bool=False,quiet:bool=True)->Tuple[NPComplexArray,NPComplexArray,DefaultMatrixType,DefaultMatrixType]:
		raise RuntimeError("Here")
	
	@staticmethod
	def factory_solver(name:str,problem:"Problem")->"GenericEigenSolver":
		if name in GenericEigenSolver._registered_solvers.keys():
			return GenericEigenSolver._registered_solvers[name](problem)
		else:
			libname=name
			if libname=="slepc":
				libname="petsc"
			try:
				import importlib
				importlib.import_module("pyoomph.solvers."+libname)
				if name in GenericEigenSolver._registered_solvers.keys():
					return GenericEigenSolver._registered_solvers[name](problem)
				else:
					raise RuntimeError("Unknown Eigen solver: '"+name+"'. Following are defined (and included): "+str(list(GenericEigenSolver._registered_solvers.keys())))
			except Exception as e:
				add_msg_str="When trying to import pyoomph.solvers."+str(name)+", the following error was raised:\n"
				add_msg_str+=''.join(traceback.format_exception(type(e), value=e, tb=e.__traceback__))
				if libname=="petsc":
					add_msg_str+="\n\nNote: If you want to use SLEPc, you must install it separately. Please see the manual ( https://pyoomph.readthedocs.io/en/latest/tutorial/installation/petscslepc.html ) for more information."
				raise RuntimeError("Unknown Eigen solver solver: '"+name+"'. Following are defined (and included): "+str(list(GenericEigenSolver._registered_solvers.keys()))+"\n"+add_msg_str)

			#raise RuntimeError("Unknown Eigen solver: '"+name+"'. Following are defined (and included): "+str(list(GenericEigenSolver._registered_solvers.keys())))

	def add_matrix_manipulator(self,manip:EigenMatrixManipulatorBase):
		self.matrix_manipulators.append(manip)

	def clear_matrix_manipulators(self):
		self.matrix_manipulators.clear()


	def get_J_M_n_and_type(self)->Tuple[DefaultMatrixType,DefaultMatrixType,int,bool]:
		from scipy.sparse import csr_matrix #type:ignore
		if not self.problem._set_solved_residual(self.real_contribution):
			raise RuntimeError("Cannot set the residual "+self.real_contribution+" for eigen calculation since it has no contribution at all")
		n, M_nzz, M_nr, M_val, M_ci, M_rs, J_nzz, J_nr, J_val, J_ci, J_rs = self.problem.assemble_eigenproblem_matrices(0) #type:ignore
		matM=csr_matrix((M_val, M_ci, M_rs), shape=(n, n))	#TODO: Is csr or csc?
		matJ=csr_matrix((-J_val, J_ci, J_rs), shape=(n, n))
		is_complex=False
		if self.imag_contribution is not None:      
			if self.problem._set_solved_residual(self.imag_contribution,raise_error=False):					
				matM=cast(csr_matrix,matM.copy())
				matJ=cast(csr_matrix,matJ.copy())
				n, M_nzz, M_nr, M_val, M_ci, M_rs, J_nzz, J_nr, J_val, J_ci, J_rs = self.problem.assemble_eigenproblem_matrices(0) #type:ignore
				matMi = csr_matrix((M_val, M_ci, M_rs), shape=(n, n))  # TODO: Is csr or csc?
				matJi = csr_matrix((-J_val, J_ci, J_rs), shape=(n, n))
				if M_nzz>0:
					matM=cast(csr_matrix,matM+complex(0,1)*matMi)
					is_complex = True
				if J_nzz>0:
					matJ =cast(csr_matrix,matJ+ complex(0, 1) * matJi)
					is_complex=True

		self.problem._set_solved_residual("")

		#print("Applying Matrix manipulators")
		for manip in self.matrix_manipulators:
			#print("APPLY MANIP",manip)
			#if isinstance(manip,EigenMatrixSetDofsToZero):      
				#print("APPLY MANIP",manip,manip.doflist)
			matJ,matM=manip.apply_on_J_and_M(self,matJ,matM)

		if matM.nnz==0: #type:ignore
			raise RuntimeError("The mass matrix has no entries. This likely means that you do not have any time derivatives in your system")

		if not self.problem.is_quiet():
			print("Matrices assembled. Invoking eigensolver")
		return matJ,matM,n,is_complex
