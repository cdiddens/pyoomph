#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2026  Christian Diddens & Duarte Rocha
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
 
from ..typings import *
import numpy
from scipy.sparse import csr_matrix #type:ignore
import time

if TYPE_CHECKING:
    from .problem import Problem
    import _pyoomph

class CustomAssemblyBase:    
    def __init__(self) -> None:
        self.problem:Optional["Problem"]=None            

    def _set_problem(self,problem:"Problem"):
        self.problem=problem
        
    def has_custom_solve_routine(self)->bool:
        # You can override the default solve routine for J*dU=R
        return False
    
    def custom_solve_routine(self, solve_Jx_b:Callable[[NPFloatArray],NPFloatArray], b:NPFloatArray) -> NPFloatArray:
        # You can override the default solve routine for J*dU=R
        # This is only called if has_custom_solve_routine() returns True
        raise NotImplementedError("Custom solve routine not implemented")

    def invalidate_cache(self)->None:
        pass

    def actions_after_adapt(self)->None:
        self.invalidate_cache()

    def actions_after_remeshing(self)->None:
        self.invalidate_cache()

    def actions_after_equation_numbering(self)->None:
        self.invalidate_cache()

    def actions_after_setting_initial_condition(self)->None:
        self.invalidate_cache()

    def actions_after_successful_newton_solve(self)->None:
        pass

    def initialize(self)->None:
        pass
    
    def finalize(self)->None:
        pass

    @overload
    def get_residuals_and_jacobian(self,require_jacobian:Literal[False],dparameter:Optional[str]=None)->NPFloatArray: ...

    @overload
    def get_residuals_and_jacobian(self,require_jacobian:Literal[True],dparameter:Optional[str]=None)->Tuple[NPFloatArray,csr_matrix]: ...

    def get_residuals_and_jacobian(self,require_jacobian:bool,dparameter:Optional[str]=None)->Union[NPFloatArray,Tuple[NPFloatArray,csr_matrix]]:
        raise RuntimeError("Must be implemented")
        pass

    # Optionally: Return mass and jacobian matrices of the last step, can be used e.g. for Lyapunov exponent calculation
    def get_last_mass_and_jacobian_matrices(self)->Tuple[Optional[csr_matrix],Optional[csr_matrix]]:
        return None,None
        


class FixedMeshMaxQuadraticNonlinearAssembly(CustomAssemblyBase):
    """
    For a problem with max. quadratic non-linearities with non-moving, non-adaptive meshes, this assembly hander saves a lot of time. 
    It only works for first order time stepping and assumes BDF2 time-stepping, where the first step is degraded.
    The residuals must be writeable as 
    
      :math:`\\vec{R}(\\vec{U})=\\vec{R}_0+\\mathbf{M}_0\\partial_t \\vec{U}+\\mathbf{J}_0\\vec{U}+\\frac{1}{2}\\vec{U}\\cdot\\mathbf{H}_0\\cdot\\vec{U}`
      
    where :math:`\\vec{R}_0`, :math:`\\mathbf{M}_0`, :math:`\\mathbf{J}_0` and :math:`\\mathbf{H}_0` are the residual vector, mass matrix, Jacobian and Hessian rank-3-tensor evaluated at the trivial dof vector U=0.
    In particular, :math:`\\mathbf{H}_0` must be independent of :math:`\\vec{U}`.
    Parameters may appear in all terms, however, it depends on the passed parameter of `cache_at_fixed_parameters`.
    This controlls whether the tensor cache is evaluated at fixed or zero parameters. 
    In the latter case, the parameter contribution is added, which requires some overhead, but allows to vary the parameter without the demanding update of the tensor cache.
    If a global parameter frequenly changes its value, it is better to use cache_at_fixed_parameters=False or 
    to pass a set of all other (rather constant) parameters to cache_at_fixed_parameters.
    However, frequently varying parameters excluded by cache_at_fixed_parameters (either by passing False or a set not including those) 
    may not appear in the Hessian term H, i.e. they may appear at maximum as factor for linear or constant terms.
    
    Furthermore, you must activate the analytical Hessian via problem.set_analytic_hessian_products(True).

    In particular, the mesh(es) must be non-moving, as moving meshes are usually highly nonlinear.
    """

    def __init__(self,cache_at_fixed_parameters:Union[bool,Set[str]]=True) -> None:
        super().__init__()
        self._tensor_cache_valid=False
        self._M0:csr_matrix
        self._J0:csr_matrix
        self._R0:NPFloatArray
        self._HMat:csr_matrix
        self._HTens:_pyoomph.SparseRank3Tensor
        self._ndof:int=-1
        self._history_dofs:Dict[float,NPFloatArray]={}        
        self._last_current_dofs=None
        self._cache_at_fixed_parameters=cache_at_fixed_parameters
        self._param_values:Dict[str,float]={} # Parameter values at cache assembly 
        self._param_contribs:Dict[str,Tuple[NPFloatArray,csr_matrix,csr_matrix]] # Parameter contributions R,J,M
        self._lastMatM:Optional[csr_matrix]=None
        self._lastMatJ:Optional[csr_matrix]=None

    def invalidate_time_history(self)->None:
        """Since obtaining the history dofs takes some time, we store them in a ring buffer. This routine clears the buffer
        """
        self._history_dofs={}
        self._last_current_dofs=None        

    def invalidate_cache(self)->None:
        """Called when the system changes. This invokes a demanding recalculation of all cached tensors.
        """
        self._tensor_cache_valid=False
        self.invalidate_time_history()


    def actions_after_successful_newton_solve(self) -> None:
        """Whenever we successfully take a time step, we can backup the degrees of freedom for the next step to save some time
        """
        if self._last_current_dofs is not None:
            assert self.problem
            t=self.problem.time_stepper_pt().time_pt().time(0)
            self._history_dofs[t]=self._last_current_dofs # Store the degrees at the current time


    def update_tensor_cache(self)->None:
        """Recalculate all tensors R0, J0, M0 and H0 and potential parameter contributions to R, J and M (but not H)"""
        assert self.problem
        oldsetting=self.problem.use_custom_residual_jacobian
        self.problem.use_custom_residual_jacobian=False # Set it to false to get the right matrices by elemental assembly
        t0=time.time()

        dofs=self.problem.get_history_dofs(0) #store the dofs
        self.problem.set_current_dofs([0.0]*len(dofs)) # zero dofs

        ntstep=self.problem.ntime_stepper()
        was_steady=[False]*ntstep
        for i in range(ntstep):
            ts=self.problem.time_stepper_pt(i)
            was_steady[i]=ts.is_steady()
            ts.make_steady()

        pnames=self.problem.get_global_parameter_names()
        paramvals:Dict[str,float]={}
        self._param_values={}
        self._param_contribs={}
        needs_contrib:List[str]=[]
        for pname in pnames:
            paramvals[pname]=self.problem.get_global_parameter(pname).value
        
        for pname in pnames:
            param=self.problem.get_global_parameter(pname)
            if isinstance(self._cache_at_fixed_parameters,set):
                if pname in self._cache_at_fixed_parameters:
                    self._param_values[pname]=param.value
                    continue
            elif self._cache_at_fixed_parameters:
                self._param_values[pname]=param.value
                continue
            param.value=0.0
            needs_contrib.append(pname)
            

        t1=time.time()
        print("UPDATING TENSOR CACHE")
        self._R0=numpy.array(self.problem.get_residuals()) #type:ignore # Initial residual
        n, M_nzz, M_nr, M_val, M_ci, M_rs, J_nzz, J_nr, J_val, J_ci, J_rs = self.problem.assemble_eigenproblem_matrices(0.0) #type:ignore # Mass and zero Jacobian
        self._M0=csr_matrix((M_val, M_ci, M_rs), shape=(n, n)).copy()	#type:ignore
        self._J0=csr_matrix((J_val, J_ci, J_rs), shape=(n, n)).copy() #type:ignore                
        self._J0.eliminate_zeros() #type:ignore
        self._M0.eliminate_zeros() #type:ignore
        self._J0.sort_indices() #type:ignore
        self._M0.sort_indices() #type:ignore
        self._M0._has_canonical_format=True
        self._J0._has_canonical_format=True        
        t2=time.time()
        print("JACOBIAN AND MASS MATRIX DONE IN "+str(t2-t1)+" s")
        print("ASSEMBLING HESSIAN TENSOR")
        self._HTens=self.problem._assemble_hessian_tensor(False)        
        print("PREPARE FOR HESSIAN VECTOR PRODUCT")
        colinds,rowstart=self._HTens.finalize_for_vector_product()
        tempvals=numpy.zeros((rowstart[-1],),dtype=numpy.float64) #type:ignore
        self._HMat=csr_matrix((tempvals,colinds,rowstart),shape=self._J0.shape) #type:ignore
        self._HMat._has_canonical_format=True #type:ignore
        t3=time.time()
        print("HESSIAN TENSOR DONE in "+str(t3-t2)+" s")
        for pname in needs_contrib:
            print("CALCULATING CONTRIBUTION OF PARAMTER "+pname)
            self.problem._replace_RJM_by_param_deriv(pname,True)
            dRdp:NPFloatArray=numpy.array(self.problem.get_residuals()) #type:ignore # Initial residual
            n, M_nzz, M_nr, M_val, M_ci, M_rs, J_nzz, J_nr, J_val, J_ci, J_rs = self.problem.assemble_eigenproblem_matrices(0.0) #type:ignore # Mass and zero Jacobian
            dMdp:csr_matrix=csr_matrix((M_val, M_ci, M_rs), shape=(n, n)).copy()	#type:ignore
            dJdp:csr_matrix=csr_matrix((J_val, J_ci, J_rs), shape=(n, n)).copy() #type:ignore
            dMdp.eliminate_zeros()
            dJdp.eliminate_zeros()
            dMdp.sort_indices()
            dJdp.sort_indices()
            dMdp._has_canonical_format=True
            dJdp._has_canonical_format=True
            self._param_contribs[pname]=(dRdp,dJdp,dMdp) #type:ignore
            self.problem._replace_RJM_by_param_deriv(pname,False)

        self.problem.set_current_dofs(dofs) #type:ignore # restore dofs

        for i in range(ntstep):
            if not was_steady[i]:
                self.problem.time_stepper_pt(i).undo_make_steady()
        
        for pname in pnames:
            self.problem.get_global_parameter(pname).value=paramvals[pname]

        self.problem.use_custom_residual_jacobian=oldsetting #reset setting
        self._ndof=n
        self._tensor_cache_valid=True

        t4=time.time()
        print("TENSOR CACHE UPDATED in "+str(t4-t0)+" s")

    def get_history_dofs(self,index:int) -> NPFloatArray:
        """Get the history dofs at previous time step 'index'. These are buffered to save some time,

        Args:
            index (int): 0 means current step, 1 means last step and 2 means the degrees of freedom two time steps ago.

        Returns:
            NPFloatArray: Degrees of freedom of the system.
        """
        assert self.problem
        t=self.problem.timestepper.time_pt().time(index)
        if index==0:
            res=self.problem.get_history_dofs(index) # Index 0 can never be be cached. They change in each Newton step. But these are quickly obtainable
            self._last_current_dofs=res
        else:
            t=self.problem.timestepper.time_pt().time(index)
            if t in self._history_dofs.keys(): # We have buffered the dofs at this time
                res=self._history_dofs[t]
            else:
                res=self.problem.get_history_dofs(index) # We have to obtain them, which might require some time
                self._history_dofs[t]=res

        if len(self._history_dofs)>3: #Get rid of some history to not overcrowd memory
            times:List[float]=list(sorted(self._history_dofs.keys()))
            times_to_rem=times[:-3] # Let 3 entries alive
            for trem in times_to_rem:
                del self._history_dofs[trem]                   
        return res
        

    def get_residuals_and_jacobian(self,require_jacobian:bool,dparameter:Optional[str]=None)->Union[NPFloatArray,Tuple[NPFloatArray,csr_matrix]]:
        """Get the residual vector (and potentially the Jacobian) based on the current and history dofs using the cached tensors.

        When a parameter changes, for which we have evaluted the tensor, we have to recalculate the cache. 

        Args:
            require_jacobian (bool): Do we require the Jacobian or not?

        Returns:
            Union[NPFloatArray,Tuple[NPFloatArray,csr_matrix]]: Either the residual vector or the residual vector and the Jacobian.
        """
        assert self.problem
        assert dparameter is None


        for pname,oldval in self._param_values.items():
            param=self.problem.get_global_parameter(pname)
            if param.value!=oldval:
                self._tensor_cache_valid=False
                print("Invalidating Tensor cache, since parameter "+pname+" has changed. For often changing parameters, consider using cache_at_fixed_parameters=False or set a set of constant parameters")
                break

        if not self._tensor_cache_valid:
            self.update_tensor_cache()

        #t0=time.time()
        dofs0=self.get_history_dofs(0)
        
        if len(dofs0)!=self._ndof:
            self.invalidate_cache()
            self.update_tensor_cache()
            dofs0=self.get_history_dofs(0)        

        dofs1=self.get_history_dofs(1)
        dofs2=self.get_history_dofs(2)        
        #t1=time.time(); print("Getting history: "+str(t1-t0)); t0=t1
        if self.problem.timestepper.get_num_unsteady_steps_done()==0: # The first step is degraded to BDF1 by default
            w0=self.problem.timestepper.weightBDF1(1,0)
            w1=self.problem.timestepper.weightBDF1(1,1)
            w2=0
        else:
            w0=self.problem.timestepper.weightBDF2(1,0)
            w1=self.problem.timestepper.weightBDF2(1,1)
            w2=self.problem.timestepper.weightBDF2(1,2)
        
        matM=self._M0
        matJ=self._J0
        vecR=self._R0

        for pname,contrib in self._param_contribs.items():
            param=self.problem.get_global_parameter(pname)
            val=param.value
            vecR+=contrib[0]*val
            matJ+=contrib[1]*val #type:ignore
            matM+=contrib[2]*val #type:ignore

        #t1=time.time(); print("Before tens mult "+str(t1-t0)); t0=t1
        self._HMat.data=self._HTens.right_vector_mult(dofs0)
        #t1=time.time(); print("Tens mult: "+str(t1-t0)); t0=t1
        half_H_times_dof0:csr_matrix=0.5*self._HMat #type:ignore
        
        #print("HESSCONTRIB",H_times_dof0)
        #r=r0+JU+1/2*HUU
        #J=J+HU
        #t1=time.time(); print("Before mat asm: "+str(t1-t0)); t0=t1
        matJ:csr_matrix=matJ+w0*matM+half_H_times_dof0  #type:ignore               
        dtUold:NPFloatArray=matM@(w1*dofs1+w2*dofs2) #type:ignore
        residual:NPFloatArray=vecR+dtUold+matJ@dofs0 #type:ignore
        #t1=time.time(); print("RES asm: "+str(t1-t0)); t0=t1
        

        #self.use_custom_residual_jacobian=False
        if not require_jacobian:                  
            
            return residual #type:ignore
        else:                        
            matJ+=half_H_times_dof0 #type:ignore
            #t1=time.time(); print("Mat asm: "+str(t1-t0)); t0=t1
            self._lastMatJ=matJ
            self._lastMatM=matM
            return residual,matJ #type:ignore


    def get_last_mass_and_jacobian_matrices(self)->Tuple[Optional[csr_matrix],Optional[csr_matrix]]:
        return self._lastMatM,self._lastMatJ
