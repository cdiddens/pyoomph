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
 
from .generic import GenericLinearSystemSolver, GenericEigenSolver, EigenSolverWhich,DefaultMatrixType
from collections import OrderedDict
import petsc4py #type:ignore
import sys

petsc4py.init(sys.argv) #type:ignore

import slepc4py #type:ignore

slepc4py.init(sys.argv) #type:ignore

from petsc4py import PETSc #type:ignore
from slepc4py import SLEPc #type:ignore
from ..generic.mpi import *
from ..typings import *
import numpy

if TYPE_CHECKING:
    from ..generic.problem import Problem


@GenericLinearSystemSolver.register_solver()
class PETSCSolver(GenericLinearSystemSolver):
    idname = "petsc"

    def __init__(self, problem:"Problem"):
        super().__init__(problem)
        self._do_not_set_any_args:bool=False
        self.petsc_mat=None
        self.ksp=None
        self.x=None

    #		opts=PETSc.Options().getAll()
    #		if "add_zero_diagonal" in opts.keys():
    #			problem.set_diagonal_zero_entries(True)
    
    def use_mumps(self,mumps_param14:Optional[int]=None):
        _SetDefaultPetscOption("mat_mumps_icntl_6",5)
        _SetDefaultPetscOption("ksp_type","preonly")
        _SetDefaultPetscOption("pc_type","lu")
        _SetDefaultPetscOption("pc_factor_mat_solver_type","mumps")
        if mumps_param14 is not None:
            _SetDefaultPetscOption("mat_mumps_icntl_14",mumps_param14)
        return self    

    def set_default_petsc_option(self,name:str,val:Any=None,force:bool=False)->None:
        _SetDefaultPetscOption(name,val, force) #type:ignore

    def get_PETSc(self)->Any:
        """
        Returns access to PETSc
        If defining derived classes that need access to PETSc, get PETSc from here, do not import petsc4py again
        """
        return PETSc

    def setup_solver(self):        
        opts = PETSc.Options().getAll() #type:ignore
        if "add_zero_diagonal" in opts.keys(): #type:ignore
            #			print(dir(self.petsc_mat))
            self.petsc_mat.setOption(19, 0) #type:ignore
            self.petsc_mat.shift(0) #type:ignore

        self.ksp = PETSc.KSP().create() #type:ignore
        self.ksp.setOperators(self.petsc_mat) #type:ignore
        if not self._do_not_set_any_args:
            self.ksp.setType('preonly') #type:ignore
        pc = self.ksp.getPC() #type:ignore
        if not self._do_not_set_any_args:
            pc.setType('lu') #type:ignore
        opts = PETSc.Options().getAll() #type:ignore
        if "pc_factor_mat_solver_type" in opts.keys(): #type:ignore
            if hasattr(pc, "setFactorSolverPackage"): #type:ignore
                if not self._do_not_set_any_args: #type:ignore
                    pc.setFactorSolverPackage(opts["pc_factor_mat_solver_type"]) #type:ignore
        pc.setFromOptions() #type:ignore
        self.ksp.setFromOptions() #type:ignore
        #print('Solving with:', self.ksp.getType())  # ,dir(pc)

    def solve_serial(self,op_flag:int,n:int,nnz:int,nrhs:int,values:NPFloatArray,rowind:NPIntArray,colptr:NPIntArray,b:NPFloatArray,ldb:int,transpose:int)->int:
        if op_flag == 1:
            if self.petsc_mat is not None:
                self.petsc_mat.destroy()
                self.petsc_mat=None
            if self.ksp is not None:
                self.ksp.destroy() #type:ignore
                self.ksp=None
            if self.x is not None:
                self.x.destroy() #type:ignore
                self.x=None
                
            #self.petsc_mat.destroy() #type:ignore
            self.petsc_mat = PETSc.Mat().createAIJ(size=(n, n), csr=(colptr, rowind, values)) #type:ignore
            self.x = PETSc.Vec().createSeq(n) #type:ignore
        elif op_flag == 2:
            self.setup_solver()
            bv = PETSc.Vec().createWithArray(b) #type:ignore
            if self.problem._custom_assembler is not None and self.problem._custom_assembler.has_custom_solve_routine():
                raise RuntimeError("Cannot use custom solve routine with PETSc yet. Also, iterative solving might require different handling here")
            else:
                self.ksp.solve(bv, self.x) #type:ignore
                xv = self.x.getArray() #type:ignore
            b[:] = xv[:] #type:ignore

            #print('Converged in', self.ksp.getIterationNumber(), 'iterations.') #type:ignore

            
            
            bv.destroy() #type:ignore
        else:
            raise RuntimeError("Cannot handle Petsc mode " + str(op_flag) + " yet")
        return 0  # TODO: Return sign of Jacobian

    def solve_distributed(self, op_flag: int, allow_permutations: int, n: int, nnz_local: int, nrow_local: int, first_row: int, values: NPFloatArray, col_index: NPIntArray, row_start: NPIntArray, b: NPFloatArray, nprow: int, npcol: int, doc: int, data: NPUInt64Array, info: NPIntArray)->None:

        if op_flag == 1:
            #print("PETSCINF",nrow_local,n)
            self.petsc_mat = PETSc.Mat().createAIJ(size=((nrow_local, n), (nrow_local, n),),csr=(row_start, col_index, values)) #type:ignore
        #			print("PROCESSOR Ns",get_mpi_rank(),nrow_local,n)
        #			print("PROCESSOR RS",get_mpi_rank(),row_start)
        #			print("PROCESSOR CI",get_mpi_rank(),col_index)
        #			print("FIRST ROW",get_mpi_rank(),first_row)
        # self.petsc_mat
        elif op_flag == 2:
            self.setup_solver()
            bv = PETSc.Vec().createWithArray(b, (len(b), n)) #type:ignore
            xv = bv.duplicate() #type:ignore
            self.ksp.solve(bv, xv) #type:ignore
            res = xv.getArray() #type:ignore
            b[:] = res[:] #type:ignore

            self.petsc_mat.destroy() #type:ignore
            self.ksp.destroy() #type:ignore
            xv.destroy() #type:ignore
            bv.destroy() #type:ignore
        else:
            raise RuntimeError("Cannot handle Petsc mode " + str(op_flag) + " yet")

    def set_options(self,**kwargs:Any):
        for a,b in kwargs.items():
            PETSc.Options().setValue(a,b) #type:ignore

def _SetDefaultPetscOption(key:str, val:Any,force:bool=False):
    if force or (not PETSc.Options().hasName(key)): #type:ignore
        if isinstance(val, complex):
            print("GOT COMPLEX",val)
            val=str(val.real)+("+" if val.imag>=0 else "")+str(val.imag)+"i"
            print("CASTED TO",val)
        PETSc.Options().setValue(key, val) #type:ignore


@GenericEigenSolver.register_solver()
class SlepcEigenSolver(GenericEigenSolver):
    idname = "slepc"

    def __init__(self, problem:"Problem"):
        super().__init__(problem)
        self.spectral_transformation:Optional[str]="sinvert"
        self.store_basis:bool=False
        self._last_basis:Optional[Union[NPComplexArray,NPFloatArray]]=None
        
    def supports_target(self):
        return True
        
    def get_last_basis(self)->Optional[Union[NPComplexArray,NPFloatArray]]:
        return self._last_basis

    def further_setup(self,E): #type:ignore
        pass
    
    def set_default_option(self,name:str,val:Any=None,force:bool=False)->None:
        _SetDefaultPetscOption(name,val, force)
    
    def use_mumps(self,mumps_param14:Optional[int]=None):        
        _SetDefaultPetscOption("st_ksp_type","preonly")
        _SetDefaultPetscOption("st_pc_type","lu")
        _SetDefaultPetscOption("st_pc_factor_mat_solver_type","mumps")
        _SetDefaultPetscOption("st_mat_mumps_icntl_6",5)
        if mumps_param14 is not None:
            _SetDefaultPetscOption("st_mat_mumps_icntl_14",mumps_param14)
        return self

    def solve(self, neval:int, shift:Union[float,None,complex]=None,sort:bool=True,which:EigenSolverWhich="LM",OPpart:Optional[Literal["r","i"]]=None,v0:Optional[Union[NPComplexArray,NPFloatArray]]=None,target:Optional[complex]=None,custom_J_and_M:Optional[Tuple["DefaultMatrixType"]]=None,with_left_eigenvectors:bool=False,quiet:bool=True)->Tuple[NPComplexArray,NPComplexArray,"DefaultMatrixType","DefaultMatrixType"]:
        if which!="LM":
            raise RuntimeError("Implement which="+str(which))
        if OPpart is not None:
            raise RuntimeError("Implement OPpart="+str(OPpart))
#        if v0 is not None:
#            raise RuntimeError("Implement v0="+str(v0))
    
        if with_left_eigenvectors:
            raise RuntimeError("Implement with_left_eigenvectors")    
        if custom_J_and_M is not None:
            Jin=custom_J_and_M[0]
            Min=custom_J_and_M[1]
            n=Jin.shape[0]
            if not isinstance(Jin,DefaultMatrixType):
                Jin=Jin.tocsr()
                assert isinstance(Jin,DefaultMatrixType)
            if not isinstance(Min,DefaultMatrixType):
                Min=Min.tocsr()
                assert isinstance(Min,DefaultMatrixType)
                
            M=PETSc.Mat().createAIJ(size=((n, n), (n, n),), csr=(Min.indptr, Min.indices, Min.data))
            J=PETSc.Mat().createAIJ(size=((n, n), (n, n),), csr=(Jin.indptr, Jin.indices, Jin.data))
            
        else:
            Jin,Min,n,complex_mat=self.get_J_M_n_and_type()
            upscale_to_complex=complex_mat and (PETSc.ScalarType in {numpy.float64,numpy.float128,numpy.float32})
            if upscale_to_complex:
                raise RuntimeError("SLEPc cannot handle a complex matrix here...")
            M=PETSc.Mat().createAIJ(size=((n, n), (n, n),), csr=(Min.indptr, Min.indices, Min.data))
            J=PETSc.Mat().createAIJ(size=((n, n), (n, n),), csr=(Jin.indptr, Jin.indices, Jin.data))
            
        #if self.imag_contribution is not None:
        #    raise RuntimeError("Cannot have imaginary matrix contributions yet here")
#        for manip in self.matrix_manipulators:
#            raise RuntimeError("Cannot have MatrixManipulators yet here: "+str(manip))
            #J, M = manip.apply_on_J_and_M(self, J, M)

        # TODO: Working example
        ##--petsc -st_pc_type lu -st_pc_factor_mat_solver_type umfpack
        # print(dir(PETSc.Options.hasName))
        # exit()
        
        _SetDefaultPetscOption("eps_type", "krylovschur") # krylovschur
        target_set=target is not None
        if target is None:
            if shift is not None:
                target=shift

        
        if self.spectral_transformation:
            _SetDefaultPetscOption("st_ksp_type", "preonly")
            _SetDefaultPetscOption("st_type", self.spectral_transformation)
                            
        E = SLEPc.EPS()  #type:ignore
        E.create() #type:ignore
        if target is not None:
            E.setTarget(target)
            E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE) #type:ignore
        else:
            E.setTarget(0)
            E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL) #type:ignore
            
            
        
            
            #trgt=PETSc.toScalar(target)
            #print(trgt)
            #E.setTarget(trgt)
        E.setOperators(J, M) #type:ignore
        E.setProblemType(SLEPc.EPS.ProblemType.GNHEP) #type:ignore
        
        if neval==0:
            neval=1
        #E.setProblemType(SLEPc.EPS.ProblemType.PGNHEP)
        #ncv=max(2 * neval + 1, 5 + neval)
        ncv=max(2 * neval + 1, 5 + neval)
        mdp=ncv #TODO: Can be smaller for higher
        
        E.setDimensions(neval,ncv,mdp) #type:ignore
        
        if v0 is not None:
            if len(v0.shape)==1:
                _v0=PETSc.Vec().createWithArray(v0)
                E.setInitialSpace(_v0)
                _v0.destroy()
            else:
                ispace=[]
                for i in range(min(v0.shape[0],ncv)):
                    ispace.append(PETSc.Vec().createWithArray(v0[i,:]))
                E.setInitialSpace(ispace)
                for _v0 in ispace:
                    _v0.destroy()
        
        #print(dir(E))
        #exit()
        # E.setProblemType(SLEPc.EPS.ProblemType.PGNHEP)
        E.setFromOptions() #type:ignore

        if self.spectral_transformation and shift:
            E.getST().setShift(shift)
        self.further_setup(E) #type:ignore
        E.solve() #type:ignore

        if quiet:
            Print = lambda *pargs,**kwargs: None
        else:
            Print = PETSc.Sys.Print #type:ignore
        Print()
        Print("******************************")
        Print("*** SLEPc Solution Results ***")
        Print("******************************")
        Print()
        its = E.getIterationNumber() #type:ignore
        Print("Number of iterations of the method: %d" % its) #type:ignore
        eps_type = E.getType() #type:ignore
        Print("Solution method: %s" % eps_type) #type:ignore
        nev, ncv, mpd = E.getDimensions()  #type:ignore
        Print("Number of requested eigenvalues: %d" % nev)
        tol, maxit = E.getTolerances() #type:ignore
        Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit)) #type:ignore
        nconv = E.getConverged() #type:ignore
        Print("Number of converged eigenpairs %d" % nconv) #type:ignore

        #Print(M) #type:ignore

        evals = []
        evects = []
        if nconv > 0:
            # Create the results vectors

            vr, wr = J.getVecs() #type:ignore
            vi, wi = J.getVecs() #type:ignore
            #
            Print()
            Print(" k ||Ax-kx||/||kx|| ")
            Print("----------------- ------------------")
            #lastev = None
            for i in range(nconv): #type:ignore
                k = E.getEigenpair(i, vr, vi) #type:ignore
                #k=E.getEigenvalue(i) #type:ignore
                #E.getEigenvector(i, vr, vi) #type:ignore
                error = E.computeError(i) #type:ignore
                evals.append(k) #type:ignore
                _vr = 0+vr.getArray() #type:ignore
                #_vi=0+vi.getArray() #type:ignore
                # TODO: Something seems to be wrong in complex SLEPc. At least here, with complex shift, it can be messed up
                #Print("IN K %9f%+9f j"%(k.real,k.imag)+" error: %12g" % error) #type:ignore
                #print("EQ ",k*(Min*_vr)-Jin*_vr)
                if k.imag != 0.0: #type:ignore
                    #Print("LASTEV "+("None" if lastev is None else "NOTNONE"))

                    if False:
                        if lastev is not None:
                            Print("DIFF  "+str(numpy.abs(lastev - numpy.conjugate(k))))
                            if numpy.abs(lastev - numpy.conjugate(k)) == 0.0:
                                Print("ADDING VI  "+str(vi.getArray()))
                                evects.append(0+vi.getArray())
                            #lastev = None
                        else:
                            evects.append(0+_vr)
                            #lastev = k
                        
                    else:
                        evects.append(0+_vr+vi.getArray()*1j) #type:ignore
                        Print(" %9f%+9f j %12g" % (k.real, k.imag, error))
                else:
                    #lastev = None
                    evects.append(0+_vr) #type:ignore
                    Print(" %12f %12g" % (k.real, error)) #type:ignore

        evals = numpy.array(evals) #type:ignore
        if sort:
            if sort==True:
                if target_set:
                    srt = numpy.argsort(numpy.abs(evals-complex(target)))[0:min(neval, len(evals))]
                else:
                    srt = numpy.argsort(-evals)[0:min(neval, len(evals))] #type:ignore
            else:
                srt = numpy.argsort(numpy.array([sort(x) for x in evals]))[0:min(neval, len(evals))] #type:ignore
            #print("SORTING",evals,srt)
            evals = evals[srt] #type:ignore
            evects = numpy.array(evects)[srt] #type:ignore
        else:
            evects = numpy.array(evects) #type:ignore
            
        if self.store_basis:
            self._last_basis=[]
            basis=E.getBV()
            nbasis=basis.getSizes()[1]        
            for i in range(nbasis):
                bv=basis.createVec()
                basis.copyVec(i,bv)
                self._last_basis.append(bv.getArray())
                bv.destroy()
            self._last_basis=numpy.array(self._last_basis)
        else:
            self._last_basis=None
        
        M.destroy() #type:ignore
        J.destroy() #type:ignore    
        E.destroy() #type:ignore
        
        return numpy.array(evals), numpy.array(evects),Jin,Min #type:ignore

    def get_PETSc(self)->Any:
        """
        Returns access to PETSc
        If defining derived classes that need access to PETSc, get PETSc from here, do not import petsc4py again
        """
        return PETSc

    def get_SLEPc(self)->Any:
        """
        Returns access to SLEPc
        If defining derived classes that need access to SLEPc, get SLEPc from here, do not import slepc4py again
        """        
        return SLEPc

class FieldSplitPETSCSolver(PETSCSolver):
    def __init__(self,problem:"Problem"):
        super(FieldSplitPETSCSolver, self).__init__(problem)
        self._fieldsplit_map:Optional[NPIntArray]=None
        self._fieldsplit:Optional[List[Tuple[str,Any]]]=None
        self.default_field_split:Optional[int]=None
        self._fieldsplit_names:Dict[int,str]={}
        self.preconditioner_matrix_name=None
        self._nullspaces=[]
        
    def add_constant_nullspace(self,*dofnames):
        self._nullspaces.append(("constant",dofnames))
        
        
    def set_fieldsplit_names(self,**kargs:int)->None:
        for k,v in kargs.items():
            self._fieldsplit_names[v]=k

    def define_options(self):
        pass

    def define_field_split(self):
        #Call split_fields here (with e.g. domain/velocity_x=0, ...)
        pass

    def split_fields(self,**kwargs:int):
        meshblocks:OrderedDict[str,Dict[str,int]]=OrderedDict()
        wholemesh:OrderedDict[str,int]=OrderedDict()
        allkeys:OrderedDict[str,bool]=OrderedDict()
        for n,b in kwargs.items():
            tmesh=self.problem.get_mesh(n,return_None_if_not_found=True)
            if tmesh is None:
                #Simple field only
                sp=n.split("/")
                sn="/".join(sp[0:-1])
                tmesh = self.problem.get_mesh(sn, return_None_if_not_found=True)
                if tmesh is None:
                    if len(sp)==2 and self.problem._meshdict.get(sp[0], None) is not None:
                        ode=self.problem.get_ode(sp[0])
                        #ode_elem = ode._get_ODE("ODE")                        
                        inds=ode.get_code_gen()._code.get_elemental_field_indices()
                        if sp[-1] in inds.keys():
                            if not sn in meshblocks.keys():
                                meshblocks[sn]={}
                            meshblocks[sn][sp[-1]]=b
                            allkeys[sn]=True
                            continue
                    raise RuntimeError("Cannot perform a field split for the unknown field "+n)
                if not (sn in meshblocks.keys()):
                    meshblocks[sn]={}
                meshblocks[sn][sp[-1]]=b
                allkeys[sn]=True
            else: #Whole mesh
                if n in wholemesh.keys():
                    raise RuntimeError("Duplicated argument "+n)
                wholemesh[n]=b
                allkeys[n]=True

        
        for k in allkeys.keys():
            mesh=self.problem.get_mesh(k,return_None_if_not_found=True)
            if mesh is None:
                mesh=self.problem.get_ode(k)
               
            typesI, names = mesh.describe_global_dofs()
            name_look_up={v:i for i,v in enumerate(names)}
            types:NPIntArray = numpy.array(typesI,dtype=numpy.int32) #type:ignore
            if self._fieldsplit_map is None:
                self._fieldsplit_map=0*types-1
            if k in wholemesh.keys():
                dest=wholemesh[k]
                where=numpy.where(types>=0)[0] #type:ignore
                self._fieldsplit_map[where]=dest
            if k in meshblocks.keys():
                for vn,dest in meshblocks[k].items():
                    if not (vn in name_look_up.keys()):
                        raise RuntimeError("Cannot find the field "+vn+" on mesh "+k+" to split")
                    where = numpy.where(types == name_look_up[vn])[0] #type:ignore
                    self._fieldsplit_map[where] = dest


    def _perform_field_split(self):
        if len(self._nullspaces)>0:
            nsvects=[]
            for ns in self._nullspaces:
                if ns[0]=="constant":
                    alldofinds=None
                    for n in ns[1]:
                        splt=n.split("/")
                        mesh=self.problem.get_mesh("/".join(splt[:-1]),return_None_if_not_found=True)
                        if mesh is None:
                            mesh=self.problem.get_ode(n)
                        typesI, names = mesh.describe_global_dofs()
                        
                        name_look_up={v:i for i,v in enumerate(names)}
                        types:NPIntArray = numpy.array(typesI,dtype=numpy.int32)
                        if alldofinds is None:
                            alldofinds=types*0
                        where = numpy.where(types == name_look_up[splt[-1]])[0]
                        alldofinds[where]=1
                    nsvects.append(alldofinds)                    
                else:
                    raise RuntimeError("Unknown nullspace type "+ns[0])
                
            petscvects=[PETSc.Vec().createWithArray(v) for v in nsvects]
            ns=self.get_PETSc().NullSpace().create(constant=False,vectors=petscvects)
            self.petsc_mat.setNullSpace(ns) #type:ignore
            
        if self._fieldsplit_map is None:
            return
        where = numpy.where(self._fieldsplit_map == -1)[0] #type:ignore
        if len(where)>0:
            if self.default_field_split is None:
                raise RuntimeError("Found a defined field split. Use either default_field_split or specify all fields in the define_field_split")
            else:
                self._fieldsplit_map[where] = self.default_field_split

        numfields:int=numpy.amax(self._fieldsplit_map)+1 #type:ignore
        fields = []
        ownerrange=self.petsc_mat.getOwnershipRange()        
        globsize=self.petsc_mat.getSize()[0]
        
        for i in range(numfields):
            IS = PETSc.IS() #type:ignore
            subdofs = numpy.where(self._fieldsplit_map == i)[0] #type:ignore
            if ownerrange[0]>0 or ownerrange[1]<globsize:
                subdofs=subdofs[(subdofs < ownerrange[1]) & (subdofs >= ownerrange[0])]                               
            
            subdofs:NPIntArray  = numpy.array(subdofs, dtype="int32") #type:ignore
            name = self._fieldsplit_names.get(i,str(i))
            IS.createGeneral(subdofs) #type:ignore
            
            fields.append((name, IS)) #type:ignore
        pc = self.ksp.getPC() #type:ignore
        pc.setFieldSplitIS(*fields) #type:ignore
        self._fieldsplit=fields

    def assemble_preconditioner(self,name:str,restrict_on_field_split:Optional[int]=None)->Any:               
        _res, n, _M_nzz, nrow_local, M_values_arr, M_colindex_arr, M_row_start_arr=self.problem._assemble_residual_jacobian(name)
        P=PETSc.Mat().createAIJ(size=((nrow_local, n), (nrow_local, n),), csr=( M_row_start_arr,M_colindex_arr, M_values_arr)) # TODO: Must be destroyed!
        if restrict_on_field_split is not None:
            assert self._fieldsplit is not None
            ps = self._fieldsplit[restrict_on_field_split][1] #type:ignore
            P = P.createSubMatrix(ps, ps) #type:ignore
        return P #type:ignore

    def define_preconditioner(self):
        if self.preconditioner_matrix_name is not None:
            P=self.assemble_preconditioner(self.preconditioner_matrix_name)
            self.ksp.setOperators(self.petsc_mat,P)

    def setup_solver(self):
        
        self.define_options()
        super().setup_solver()
        self._fieldsplit_map=None
        self._nullspaces=[]        
        self.define_field_split()
        self._perform_field_split()
        self.define_preconditioner()

    def get_PC(self)->Any:
        pc = self.ksp.getPC() #type:ignore
        return pc #type:ignore





