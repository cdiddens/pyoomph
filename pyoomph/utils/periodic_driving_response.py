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
 
 
"""
A module to compute the linear response of a system to a periodic driving force.
"""
 
import scipy.linalg
from .. import *
from ..expressions import ExpressionNumOrNone, partial_t, pi
import scipy

class _DrivingForResponse(ODEEquations):
    def __init__(self,omega:ExpressionOrNum,hopf_damping:ExpressionOrNum):
        super().__init__()
        self.omega=omega
        self.name="_driving"
        self.damp=hopf_damping

    def define_fields(self):
        self.define_ode_variable(self.name)
        self.define_ode_variable("_dt_"+self.name)

    def define_residuals(self):
        d,d_test=var_and_test(self.name)
        dp,dp_test=var_and_test("_dt_"+self.name)        
        EQ_y = partial_t(dp,nondim=True) -  d
        EQ_yp = partial_t(d,nondim=True) + self.omega**2*dp +2*self.omega*self.damp*d
        self.add_residual(EQ_y * d_test+EQ_yp*dp_test)



class PeriodicDrivingResponse():
    """
    Helper class to compute the linear response of a system to a periodic driving force.
    Replace the periodic driving, e.g. some ``cos(omega*var("time"))``, by :py:meth:`get_driving_mode` in the problem. Then, after finding a stationary solution, you can use :py:meth:`iterate_over_driving_frequencies` to iterate over driving frequencies and get the response of the system. 
    
    Args:
        problem: The problem to which the driving force is applied.
        omega_param_name: The name of the parameter that stores the current driving frequency. 
        hopf_param_name: The name of the parameter that is used to find the driving response.
    """
    def __init__(self,problem:Problem,omega_param_name:str="_driving_omega",driving_domain_name:str="_driving",hopf_param_name:str="_driving_damping") -> None:
        self.omega_param_name=omega_param_name # Parameter is meant to be 1/scaling("temporal")
        self.driving_domain_name=driving_domain_name
        self.hopf_param_name=hopf_param_name
        self.problem=problem
        if self.problem.is_initialised():
            raise RuntimeError("Create PeriodicDrivingResponse(...) before the problem is initialized")
        self.omega_param,self.hopf_param=self.problem.define_global_parameter(**{self.omega_param_name:1,self.hopf_param_name:0})
        problem+=_DrivingForResponse(self.omega_param,self.hopf_param)@self.driving_domain_name
        self._omega_val_before_init:ExpressionNumOrNone=None
        self.problem.setup_for_stability_analysis(analytic_hessian=True,improve_pitchfork_on_unstructured_mesh=False,azimuthal_stability=False)

    # Set this as your driving. Scale it with a potential dimensional amplitude!
    def get_driving_mode(self):
        """
        Must be used to replace the driving force in the problem.

        Returns:
            An expression that represents the driving force for finding the linear response.
        """
        return var("_driving",domain=self.driving_domain_name)
    
    # Set the driving omega
    def set_driving_omega(self,omega:ExpressionOrNum):
        """
        Sets the current angular frequency of the driving force.
        """
        if not self.problem.is_initialised():
            self._omega_val_before_init=omega
        else:
            self._omega_val_before_init=None
            self.omega_param.value=float(omega*self.problem.get_scaling("temporal"))

    def get_driving_omega(self):
        """
        Returns the current angular frequency of the driving force.
        """
        if self._omega_val_before_init is not None:
            return self._omega_val_before_init
        else:
            return self.omega_param.value/self.problem.get_scaling("temporal")
        
    def get_driving_frequency(self):
        """
        Returns the current frequency of the driving force.
        """
        return self.get_driving_omega()/(2*pi)
    
    # Set the driving amplitude
    def set_driving_frequency(self,freq:ExpressionOrNum):
        """
        Sets the current frequency of the driving force.
        """
        self.set_driving_omega(2*pi*freq)

    def iterate_over_driving_frequencies(self,*,omegas:Optional[List[ExpressionOrNum]]=None,freqs:Optional[List[ExpressionOrNum]]=None,unit:ExpressionOrNum=1,signum:int=1):
        """
        Iterator to iterate over the response of the system to different driving frequencies.

        Args:
            omegas: A list of angular frequencies to iterate over. You must either set ``omegas`` or ``freqs``.
            freqs: A list of frequencies to iterate over. You must either set ``omegas`` or ``freqs``.
            unit: An optional unit for the frequencies, e.g. ``kilo*hertz``. Defaults to 1.
            signum: The sign orientation in the complex plane. Defaults to 1.

        Yields:
            For each frequency, you get the current response as complex vector, with entries belonging to the degrees of freedom of the system.
        """
        if omegas is not None and freqs is not None:
            raise RuntimeError("Cannot set both omega and frequency")
        elif omegas is not None:
            if len(omegas)==0:
                return
            self.set_driving_omega(omegas[0]*unit)            
        elif freqs is not None:
            if len(freqs)==0:
                return
            self.set_driving_frequency(freqs[0]*unit)
            omegas=[2*pi*freq for freq in freqs ]
            
        if not self.problem.is_initialised():
            self.problem.initialise()            
        if self._omega_val_before_init is not None:
            self.set_driving_omega(self._omega_val_before_init)
            self._omega_val_before_init=None

        doftypes,dofnames=self.problem.get_dof_description()
        driveind=dofnames.index(self.driving_domain_name+"/_driving")
        drivedofind=numpy.argwhere(doftypes==driveind)
        if len(drivedofind)!=1:
            raise RuntimeError("Cannot find the driving degree of freedom for some some strange reason")
        drivedofind=drivedofind[0,0]      
        dtdriveind=dofnames.index(self.driving_domain_name+"/_dt__driving")
        dtdrivedofind=numpy.argwhere(doftypes==dtdriveind)
        dtdrivedofind=dtdrivedofind[0,0]              
        ntstep=self.problem.ntime_stepper()
        was_steady=[False]*ntstep
        self.hopf_param.value=0.0
        oldomega=self.omega_param.value
        self.omega_param.value=1.0
        for i in range(ntstep):
            ts=self.problem.time_stepper_pt(i)
            was_steady[i]=ts.is_steady()
            ts.make_steady()
        n, M_nzz, M_nr, M_val, M_ci, M_rs, J_nzz, J_nr, J_val, J_ci, J_rs = self.problem.assemble_eigenproblem_matrices(0) #type:ignore
        for i in range(ntstep):
            if not was_steady[i]:
                self.problem.time_stepper_pt(i).undo_make_steady()
        matM=scipy.sparse.csr_matrix((M_val, M_ci, M_rs), shape=(n, n)).copy()	#TODO: Is csr or csc?
        matJ=scipy.sparse.csr_matrix((-J_val, J_ci, J_rs), shape=(n, n)).copy() 
        self.omega_param.value=oldomega

        lambda_in_vr=numpy.zeros((n,2))            
        lambda_in_vr[drivedofind,0]=1
        lambda_in_vr[dtdrivedofind,1]=1
        vr_in_lambda=numpy.zeros((2,n))
        vr_in_lambda[0,drivedofind]=1
        vr_in_lambda[1,dtdrivedofind]=1
        rhs=numpy.zeros((2*n+2))                        
        rhs[-2]=1
        rhs[-1]=0
        

        for omega in omegas:
            self.set_driving_omega(omega*unit)
            matJ[dtdrivedofind,dtdrivedofind]=-self.omega_param.value**2
            fullmat=scipy.sparse.bmat([[signum*matJ,self.omega_param.value*matM,lambda_in_vr],[signum*self.omega_param.value*matM,-matJ,None],[vr_in_lambda,None,None]]).copy()        
            fullmat=fullmat.tocsr()                        
            sol=rhs.copy()

            self.problem.get_la_solver().solve_serial(1,fullmat.shape[0],fullmat.nnz,1,fullmat.data,fullmat.indices,fullmat.indptr,sol,0,1)        
            self.problem.get_la_solver().solve_serial(2,fullmat.shape[0],fullmat.nnz,1,fullmat.data,fullmat.indptr,fullmat.indptr,sol,0,1)            
            result=sol[:n]+sol[n:-2]*1j        
            result/=result[drivedofind]
            self.problem.invalidate_cached_mesh_data(only_eigens=True)
            self.problem._last_eigenvectors=numpy.array([result])
            self.problem._last_eigenvalues=numpy.array([0+1j*self.omega_param.value])            
            yield result        
        return

    def new_solve_driving_response(self,*,omega:ExpressionNumOrNone=None,freq:ExpressionNumOrNone=None):
        if omega is not None and freq is not None:
            raise RuntimeError("Cannot set both omega and frequency")
        elif omega is not None:
            self.set_driving_omega(omega)            
        elif freq is not None:
            self.set_driving_frequency(freq)
            
        if not self.problem.is_initialised():
            self.problem.initialise()            
        if self._omega_val_before_init is not None:
            self.set_driving_omega(self._omega_val_before_init)
            self._omega_val_before_init=None

        doftypes,dofnames=self.problem.get_dof_description()
        driveind=dofnames.index(self.driving_domain_name+"/_driving")
        drivedofind=numpy.argwhere(doftypes==driveind)
        if len(drivedofind)!=1:
            raise RuntimeError("Cannot find the driving degree of freedom for some some strange reason")
        drivedofind=drivedofind[0,0]      
        dtdriveind=dofnames.index(self.driving_domain_name+"/_dt__driving")
        dtdrivedofind=numpy.argwhere(doftypes==dtdriveind)
        dtdrivedofind=dtdrivedofind[0,0]              
        ntstep=self.problem.ntime_stepper()
        was_steady=[False]*ntstep
        for i in range(ntstep):
            ts=self.problem.time_stepper_pt(i)
            was_steady[i]=ts.is_steady()
            ts.make_steady()
        n, M_nzz, M_nr, M_val, M_ci, M_rs, J_nzz, J_nr, J_val, J_ci, J_rs = self.problem.assemble_eigenproblem_matrices(0) #type:ignore
        for i in range(ntstep):
            if not was_steady[i]:
                self.problem.time_stepper_pt(i).undo_make_steady()
        matM=scipy.sparse.csr_matrix((M_val, M_ci, M_rs), shape=(n, n)).copy()	#TODO: Is csr or csc?
        matJ=scipy.sparse.csr_matrix((-J_val, J_ci, J_rs), shape=(n, n)).copy() 

        lambda_in_vr=numpy.zeros((n,2))            
        lambda_in_vr[drivedofind,0]=1
        lambda_in_vr[dtdrivedofind,1]=1
        vr_in_lambda=numpy.zeros((2,n))
        vr_in_lambda[0,drivedofind]=1
        vr_in_lambda[1,dtdrivedofind]=1
        rhs=numpy.zeros((2*n+2))                        
        rhs[-2]=1
        rhs[-1]=0
        fullmat=scipy.sparse.bmat([[matJ,self.omega_param.value*matM,lambda_in_vr],[self.omega_param.value*matM,-matJ,None],[vr_in_lambda,None,None]]).copy()        
        fullmat=fullmat.tocsr()                        
        self.problem.get_la_solver().solve_serial(1,fullmat.shape[0],fullmat.nnz,1,fullmat.data,fullmat.indices,fullmat.indptr,rhs,0,0)        
        self.problem.get_la_solver().solve_serial(2,fullmat.shape[0],fullmat.nnz,1,fullmat.data,fullmat.indptr,fullmat.indptr,rhs,0,0)
        sol=rhs
        result=sol[:n]+sol[n:-2]*1j        
        result/=result[drivedofind]
        self.problem.invalidate_cached_mesh_data(only_eigens=True)
        self.problem._last_eigenvectors=numpy.array([result])
        self.problem._last_eigenvalues=numpy.array([0+1j*self.omega_param.value])
        
        return result
        


    def solve_driving_response(self,*,omega:ExpressionNumOrNone=None,freq:ExpressionNumOrNone=None,with_eigenvector_guess:bool=False,numeigen=4,eigen_thresh=1e-7,by_hopf_tracking:bool=False,use_target:bool=False):
        if omega is not None and freq is not None:
            raise RuntimeError("Cannot set both omega and frequency")
        elif omega is not None:
            self.set_driving_omega(omega)            
        elif freq is not None:
            self.set_driving_frequency(freq)
            
        if not self.problem.is_initialised():
            self.problem.initialise()            
        if self._omega_val_before_init is not None:
            self.set_driving_omega(self._omega_val_before_init)
            self._omega_val_before_init=None

        doftypes,dofnames=self.problem.get_dof_description()
        driveind=dofnames.index(self.driving_domain_name+"/_driving")
        drivedofind=numpy.argwhere(doftypes==driveind)

        
        dtdriveind=dofnames.index(self.driving_domain_name+"/_dt__driving")
        dtdrivedofind=numpy.argwhere(doftypes==dtdriveind)
        if len(drivedofind)!=1:
            raise RuntimeError("Cannot find the driving degree of freedom for some some strange reason")
        drivedofind=drivedofind[0,0]        
        dtdrivedofind=dtdrivedofind[0,0]        
        istracking=self.problem.get_bifurcation_tracking_mode()=="hopf" and self.problem._bifurcation_tracking_parameter_name==self.hopf_param_name

                
        if with_eigenvector_guess or (by_hopf_tracking and not istracking):
            v0=numpy.zeros((self.problem.ndof()))
            v0[drivedofind]=1        
        else:
            v0=None
    
        eigenfilter=lambda l : abs(numpy.real(l))<eigen_thresh and abs(numpy.imag(l)-self.omega_param.value)<eigen_thresh
        if by_hopf_tracking: 
            if istracking:
                # Solve at the new omega
                self.problem.solve(max_newton_iterations=20)
            else:                       
                self.problem.activate_bifurcation_tracking(self.hopf_param,"hopf",eigenvector=v0,omega=self.omega_param.value)
                self.problem.solve(max_newton_iterations=20)
        else:
            if self.problem.get_eigen_solver().idname=="slepc" and use_target:
                self.problem.solve_eigenproblem(numeigen,v0=v0,filter=eigenfilter,target=complex(0,self.omega_param.value))
            else:
                self.problem.solve_eigenproblem(numeigen,v0=v0,filter=eigenfilter)
        foundeigen=len(self.problem.get_last_eigenvectors())
        if foundeigen!=1:
            raise RuntimeError("Cannot find a single eigenvalue that corresponds to the driving: Got "+str(foundeigen))
        
        v0=self.problem.get_last_eigenvectors()[0]
        print("DRIVE",v0[drivedofind])
        print("DTDRIVE",v0[dtdrivedofind])
        self.problem._last_eigenvectors/=v0[drivedofind] 
        print("INFO",self.omega_param.value,self.problem._last_eigenvectors[0,drivedofind],self.problem._last_eigenvectors[0,dtdrivedofind])        

        return self.problem.get_last_eigenvectors()[0]       
        

    def split_response_amplitude_and_phase(self):
        """
        Splits the complex response vector into a real-valued amplitude and phase vector.

        Returns:
            The pair of amplitude and phase vectors.
        """
        if len(self.problem.get_last_eigenvectors())!=1:
            raise RuntimeError("Must solve the response first")
        v=self.problem.get_last_eigenvectors()[0]
        ampl=numpy.absolute(v)
        phase=numpy.angle(v)
        return ampl,phase
    
    
    def switch_to_orbit_tracking(self,*,omega:ExpressionNumOrNone=None,freq:ExpressionNumOrNone=None,mode:Literal["collocation","floquet","bspline","central","BDF2"]="collocation",  order:int=2,GL_order:int=-1,T_constraint:Literal["plane","phase"]="phase",NT:int=50):
        if freq is None:
            if omega is None:
                raise RuntimeError("Need to set either omega or frequency")            
            drivemode=self.solve_driving_response(omega=omega)
            T=2*pi/omega
        elif omega is None:
            drivemode=self.solve_driving_response(freq=freq)
            T=1/freq
        else:
            raise RuntimeError("Cannot set both omega and frequency")
        
        basesol=self.problem.get_current_dofs()[0]
        history_dofs=[basesol+numpy.real(numpy.exp(1j*phase)*drivemode) for phase in numpy.linspace(0,2*numpy.pi,NT,endpoint=False)]
        history_dofs=numpy.array(history_dofs)        
        self.problem.set_current_dofs(history_dofs[0])
        return self.problem.activate_periodic_orbit_handler(T,history_dofs=history_dofs[1:],mode=mode,order=order,GL_order=GL_order,T_constraint=T_constraint)