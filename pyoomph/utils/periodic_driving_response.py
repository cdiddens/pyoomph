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
 
 
"""
A module to compute the linear response of a system to a periodic driving force.
"""
 
import scipy.linalg
from .. import *
from ..typings import *
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
        self.omega_param=self.problem.define_global_parameter(**{self.omega_param_name:1})
        self.hopf_param=self.problem.define_global_parameter(**{self.hopf_param_name:0})
        problem+=_DrivingForResponse(Expression(self.omega_param),Expression(self.hopf_param))@self.driving_domain_name
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

    #################### Assembling and solving the bordered response system ####################
    #
    # The response at a driving frequency omega solves the bordered real system
    #
    #     [ s*J      omega*M   e_d  e_dt ] [xr ]   [0]
    #     [ s*omega*M   -J      .    .   ] [xi ] = [0]
    #     [ e_d^T       .       .    .   ] [l_1]   [1]
    #     [ e_dt^T      .       .    .   ] [l_2]   [0]
    #
    # with J the NEGATED assembled Jacobian (the (A,M)=(-J,M) convention of
    # dev_docs/mpi_eigenproblems.md section 0), s the sign orientation, and the two border rows pinning
    # the driving ODE's own dofs so that the driving is exactly cos(omega*t).
    #
    # The unknowns are ordered INTERLEAVED, xr_k -> 2k and xi_k -> 2k+1, with the two multipliers last.
    # Block ordering would give each rank two disjoint row ranges once the dofs are partitioned, which
    # MatCreateAIJ cannot express; interleaving makes rank r's rows the contiguous block
    # [2*first_row, 2*(first_row+nrow_local)) and turns the column map into j -> 2j / 2j+1, with the
    # column indices still global. No communication, and no owner lookup.

    def _find_driving_dofs(self)->tuple[int,int]:
        """Global equation numbers of the driving ODE's two dofs.

        get_dof_description() is collective and full-length on a distributed problem, so these are the
        same on every rank -- which they must be, since they index a globally replicated response.
        """
        doftypes,dofnames=self.problem.get_dof_description()
        driveind=dofnames.index(self.driving_domain_name+"/_driving")
        drivedofind=numpy.argwhere(doftypes==driveind)
        if len(drivedofind)!=1:
            raise RuntimeError("Cannot find the driving degree of freedom for some some strange reason")
        dtdriveind=dofnames.index(self.driving_domain_name+"/_dt__driving")
        dtdrivedofind=numpy.argwhere(doftypes==dtdriveind)
        if len(dtdrivedofind)!=1:
            raise RuntimeError("Cannot find the driving degree of freedom for some some strange reason")
        return int(drivedofind[0,0]),int(dtdrivedofind[0,0])

    def _assemble_response_pencil(self):
        """Assemble (J, M) once, on this rank's row block, ready for any driving frequency.

        Everything that does not depend on omega happens here: the matrices are assembled with the time
        steppers made steady and the frequency parameter at 1, and the one entry that DOES depend on it
        -- the driving ODE's (dt_driving, dt_driving) Jacobian entry, analytically -omega**2 -- is zeroed
        out to be re-supplied per frequency by _solve_at_omega(). That keeps the sparsity pattern of the
        bordered system independent of omega, which is what lets the solver keep its symbolic
        factorisation across a whole frequency scan.

        Returns ``(n, nrow_local, first_row, parallel, matJ, matM, drivedofind, dtdrivedofind)`` with the
        matrices shaped ``(nrow_local, n)`` and carrying GLOBAL column indices.
        """
        drivedofind,dtdrivedofind=self._find_driving_dofs()
        eigensolver=self.problem.get_eigen_solver()
        ntstep=self.problem.ntime_stepper()
        was_steady=[False]*ntstep
        self.hopf_param.value=0.0
        oldomega=self.omega_param.value
        self.omega_param.value=1.0
        for i in range(ntstep):
            ts=self.problem.time_stepper_pt(i)
            was_steady[i]=ts.is_steady()
            ts.make_steady()
        try:
            n, M_nzz, M_nr, M_val, M_ci, M_rs, J_nzz, J_nr, J_val, J_ci, J_rs = self.problem.assemble_eigenproblem_matrices(0) #type:ignore
        finally:
            for i in range(ntstep):
                if not was_steady[i]:
                    self.problem.time_stepper_pt(i).undo_make_steady()
            self.omega_param.value=oldomega
        # shape=(M_nr, n), not (n, n): M_nr is the LOCAL row count, and under --distribute it is smaller
        # than n. Passing (n,n) with a short indptr is what used to make this die inside scipy with
        # "index pointer size (3) should be (5)", three frames from the feature the user asked for.
        matM=scipy.sparse.csr_matrix((M_val, M_ci, M_rs), shape=(M_nr, n)).copy()
        matJ=scipy.sparse.csr_matrix((-J_val, J_ci, J_rs), shape=(J_nr, n)).copy()

        nrow_local,first_row,parallel=eigensolver.get_parallel_row_split(n)
        if parallel and not self.problem.is_distributed():
            # Plain mpirun: oomph replicated the assembled matrices, so every rank holds all of them and
            # contributes only the slice the split above assigned it. Without this the solve would be
            # done nproc times over instead of once in parallel.
            matM=eigensolver.local_row_block(matM,first_row,nrow_local)
            matJ=eigensolver.local_row_block(matJ,first_row,nrow_local)
        if not parallel:
            nrow_local,first_row=n,0

        # Drop the omega-dependent entry. The driving ODE contributes d(EQ_yp)/d(dp) = omega**2 there,
        # i.e. -omega**2 in matJ; _solve_at_omega() re-adds it explicitly. The old code overwrote it in
        # the assembled CSR every iteration, which is both a SparseEfficiencyWarning when that diagonal
        # is structurally absent and a GLOBAL index write no rank can do once the rows are partitioned.
        if first_row<=dtdrivedofind<first_row+nrow_local:
            p=dtdrivedofind-first_row
            lo,hi=int(matJ.indptr[p]),int(matJ.indptr[p+1])
            matJ.data[lo+numpy.flatnonzero(matJ.indices[lo:hi]==dtdrivedofind)]=0.0
        return n,nrow_local,first_row,parallel,matJ,matM,drivedofind,dtdrivedofind

    def _solve_at_omega(self,pencil,signum:int):
        """Build the bordered system at the current omega, solve it, and return the complex response.

        The response comes back REPLICATED at full global length on every rank, which is what
        set_eigenfunction_as_dofs(), the mesh data cache and the VTK output all expect -- they index an
        eigenvector by global equation number. See dev_docs/mpi_eigenproblems.md section 3.
        """
        from ..generic.mpi import get_mpi_nproc,get_mpi_rank,mpi_allgather_vector
        n,nrow_local,first_row,parallel,matJ,matM,drivedofind,dtdrivedofind=pencil
        omega=self.omega_param.value
        ntot=2*n+2
        last_rank=parallel and (get_mpi_rank()==get_mpi_nproc()-1)

        rows:list[Any]=[]
        cols:list[Any]=[]
        vals:list[Any]=[]
        def block(mat,row_off:int,col_off:int,factor:float):
            coo=mat.tocoo()
            rows.append(2*coo.row+row_off)
            cols.append(2*coo.col+col_off)
            vals.append(factor*coo.data)
        block(matJ,0,0,signum)          # row 2p, cols 2j:      s*J
        block(matM,0,1,omega)           # row 2p, cols 2j+1:    omega*M
        block(matM,1,0,signum*omega)    # row 2p+1, cols 2j:    s*omega*M
        block(matJ,1,1,-1.0)            # row 2p+1, cols 2j+1: -J
        # The omega-dependent driving entry, dropped in _assemble_response_pencil(), on its owner only.
        if first_row<=dtdrivedofind<first_row+nrow_local:
            p=dtdrivedofind-first_row
            rows.append(numpy.array([2*p,2*p+1],dtype=numpy.int64))
            cols.append(numpy.array([2*dtdrivedofind,2*dtdrivedofind+1],dtype=numpy.int64))
            vals.append(numpy.array([signum*(-omega**2),omega**2],dtype=numpy.float64))
        # The two border COLUMNS: the multipliers act on the driving dofs' first-block rows.
        for dof,col in ((drivedofind,2*n),(dtdrivedofind,2*n+1)):
            if first_row<=dof<first_row+nrow_local:
                rows.append(numpy.array([2*(dof-first_row)],dtype=numpy.int64))
                cols.append(numpy.array([col],dtype=numpy.int64))
                vals.append(numpy.array([1.0],dtype=numpy.float64))
        nrow_block=2*nrow_local+(2 if last_rank or not parallel else 0)
        b_local=numpy.zeros(nrow_block)
        if last_rank or not parallel:
            # The two border ROWS, pinning Re(v_drive)=1 and Re(v_dt_drive)=0. They go on the LAST rank
            # (serially, at the end of the only block), which is what makes every rank's row range
            # contiguous.
            base=2*nrow_local
            rows.append(numpy.array([base,base+1],dtype=numpy.int64))
            cols.append(numpy.array([2*drivedofind,2*dtdrivedofind],dtype=numpy.int64))
            vals.append(numpy.array([1.0,1.0],dtype=numpy.float64))
            b_local[base]=1.0
        fullmat=scipy.sparse.coo_matrix((numpy.concatenate(vals),
                                         (numpy.concatenate(rows),numpy.concatenate(cols))),
                                        shape=(nrow_block,ntot)).tocsr()
        fullmat.sum_duplicates()

        la=self.problem.get_la_solver()
        if not parallel:
            # Single process: the whole system is here, so go straight through the ordinary serial entry
            # point and keep every backend, MPI-capable or not, on the path it always had. Under mpirun
            # this branch is never taken -- get_parallel_row_split() reports parallel for any nproc>1 --
            # so unlike the old code there is no replicated solve_serial() to warn the gather path about.
            sol=b_local.copy()
            la.solve_serial(1,ntot,fullmat.nnz,1,fullmat.data,fullmat.indices,fullmat.indptr,sol,0,1)
            la.solve_serial(2,ntot,fullmat.nnz,1,fullmat.data,fullmat.indices,fullmat.indptr,sol,0,1)
        else:
            first_block_row=2*first_row
            local=la.solve_python_built_distributed(ntot,nrow_block,first_block_row,fullmat,b_local)
            sol=mpi_allgather_vector(ntot,first_block_row,nrow_block,local,
                                     context="replicating the driving response on every rank")
        result=sol[0:2*n:2]+1j*sol[1:2*n:2]
        result/=result[drivedofind]
        self.problem.invalidate_cached_mesh_data(only_eigens=True)
        self.problem._last_eigenvectors=numpy.array([result])
        self.problem._last_eigenvalues=numpy.array([0+1j*omega])
        return result

    def iterate_over_driving_frequencies(self,*,omegas:list[ExpressionOrNum] | None=None,freqs:list[ExpressionOrNum] | None=None,unit:ExpressionOrNum=1,signum:int=1):
        """
        Iterator to iterate over the response of the system to different driving frequencies.

        Args:
            omegas: A list of angular frequencies to iterate over. You must either set ``omegas`` or ``freqs``.
            freqs: A list of frequencies to iterate over. You must either set ``omegas`` or ``freqs``.
            unit: An optional unit for the frequencies, e.g. ``kilo*hertz``. Defaults to 1.
            signum: The sign orientation in the complex plane. Defaults to 1.

        Yields:
            For each frequency, you get the current response as complex vector, with entries belonging to the degrees of freedom of the system.

        Works under ``mpirun``, with and without ``--distribute``: the bordered system is assembled on
        each rank's own row block and solved on COMM_WORLD, and the response is replicated afterwards.
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
        else:
            raise RuntimeError("Must set either omegas or freqs")

        if not self.problem.is_initialised():
            self.problem.initialise()            
        if self._omega_val_before_init is not None:
            self.set_driving_omega(self._omega_val_before_init)
            self._omega_val_before_init=None

        pencil=self._assemble_response_pencil()
        for omega in omegas:
            self.set_driving_omega(omega*unit)
            yield self._solve_at_omega(pencil,signum)
        return

    def new_solve_driving_response(self,*,omega:ExpressionNumOrNone=None,freq:ExpressionNumOrNone=None,signum:int=1):
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

        # _assemble_response_pencil() forces omega to 1 while it assembles and puts it back afterwards,
        # so the frequency set above survives it.
        return self._solve_at_omega(self._assemble_response_pencil(),signum)

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
        
        eigenvects=self.problem.get_last_eigenvectors()
        vfound=eigenvects[0] # not v0, which is the starting guess handed to the eigensolver above
        print("DRIVE",vfound[drivedofind])
        print("DTDRIVE",vfound[dtdrivedofind])
        eigenvects/=vfound[drivedofind]
        self.problem._last_eigenvectors=eigenvects
        print("INFO",self.omega_param.value,eigenvects[0,drivedofind],eigenvects[0,dtdrivedofind])        

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
            T:ExpressionOrNum=2*pi/omega
        elif omega is None:
            drivemode=self.solve_driving_response(freq=freq)
            T=1/freq
        else:
            raise RuntimeError("Cannot set both omega and frequency")
        
        basesol=self.problem.get_current_dofs()[0]
        history_dofs=numpy.array([basesol+numpy.real(numpy.exp(1j*phase)*drivemode) for phase in numpy.linspace(0,2*numpy.pi,NT,endpoint=False)])
        self.problem.set_current_dofs(history_dofs[0])
        return self.problem.activate_periodic_orbit_handler(T,history_dofs=history_dofs[1:],mode=mode,order=order,GL_order=GL_order,T_constraint=T_constraint)


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
