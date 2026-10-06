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
from ..generic.problem import GenericProblemHooks
import numpy
from scipy.sparse import csr_matrix
from ..expressions import ExpressionNumOrNone
from ..typings import NPFloatArray
from collections import deque
from ..generic.mpi import get_mpi_rank
from ..generic.distributed_la import RowLayout, DistVector, get_la_backend

class LyapunovExponentCalculator(GenericProblemHooks):
    """
    A class for calculating multiple Lyapunov exponents. Add it to the problem by ``problem+=LyapunovExponentCalculator(...)`` and it will do the rest for you.
    However, note that we cannot use BDF2 time derivatives in the calculation of new pertubations (B matrix). 
    Therefore, we calculate trajectories using BFD2, but the perturbation vectors are updated using either implicit Euler (BDF1) or a Crank-Nicholson-like scheme.
    
    Args:
        k: The number of Lyapunov exponents to calculate. k>=2 will invoke Gram-Schmidt on the perturbation vectors. Defaults to 1.
        waiting_time: The time to wait before starting the Lyapunov calculation.
        prerelaxation_time: The time to prerelax the perturbation vectors before starting the Lyapunov calculation. This allows to bypass initial transients.
        use_crank_nicholson_integration: Whether to use a Crank-Nicholson-like integration for the perturbation vectors, instead of BDF1 integration. This may improve accuracy for problems with large time steps. Defaults to False.
        filename: The name of the output file. Defaults to "lyapunov.txt".
        relative_to_output: Whether to save the output file relative to the problem's output directory. Defaults to True.
        store_as_eigenvectors: Whether to store the perturbation vectors as eigenvectors. Defaults to False.
        random_seed: Seed for the initial random perturbation basis. The SAME value must be used on
            every rank -- it is a constructor argument, so it is -- because the basis is drawn
            identically everywhere and then sliced to each rank's rows. See ``_draw_initial_basis``.
    """
    def __init__(self,k: int = 1,waiting_time:ExpressionNumOrNone=None, prerelaxation_time:ExpressionNumOrNone=None,use_crank_nicholson_integration:bool=False, filename="lyapunov.txt",relative_to_output=True,store_as_eigenvectors:bool=False,gram_schmidt_dt:ExpressionNumOrNone=None,random_seed:int=0):
        super().__init__()
        self.k = k
        self.random_seed = int(random_seed)
        if self.k <= 0:
            raise ValueError("k must be a positive integer")
        self.waiting_time = waiting_time
        self.prerelaxation_time = prerelaxation_time
        self.use_crank_nicholson_integration = use_crank_nicholson_integration        
        
        self.filename = filename
        self.relative_to_output = relative_to_output
        self.store_as_eigenvectors = store_as_eigenvectors
        self.B:NPFloatArray=numpy.zeros((0,0)) # Storing the k perturbation vectors        
        self.Lambdas:NPFloatArray=numpy.zeros((0,)) # Storing the Lyapunov exponents sum
        self._Tstart1,self._Tstart2=0,0 # Nondimensional times when (1) the vector calculation starts and (2) the prerelaxation ends (i.e. the lambda calculation starts)
        self._oldJ=None
        self._outputfile=None
        self.gram_schmidt_dt=gram_schmidt_dt
        self._gram_schmidt_dt=0
        self._t_last_gram_schmidt=0
        self._layout:"RowLayout | None"=None

    def _row_layout(self,problem,nrow_from_assembly:int,n:int)->"RowLayout":
        """The row layout of the assembled J and M, validated rather than inferred.

        No augmentation is installed here, so the eigenproblem matrices come back on the problem's
        base dof layout. That is asked for by name and then CHECKED against the row count the
        assembly actually returned, because the two disagreeing is precisely the class of bug B2 in
        ``dev_docs/mpi_augmented_systems.md`` -- and silently taking the shorter of the two produces
        a plausible matrix rather than an error.
        """
        layout=RowLayout.base(problem) if problem.is_distributed() else RowLayout.serial(n)
        if layout.n!=n or layout.nrow_local!=nrow_from_assembly:
            raise RuntimeError("Lyapunov: the assembled matrices have %d of %d rows on this rank, but "
                               "the dof layout says %d of %d. Refusing to guess which is right."
                               %(nrow_from_assembly,n,layout.nrow_local,layout.n))
        return layout.validate()

    def _draw_initial_basis(self,layout:"RowLayout")->NPFloatArray:
        """An orthonormal random basis of k perturbations, as this rank's (nrow_local, k) block.

        Drawn GLOBALLY from a seeded generator and then sliced, not drawn per rank. Two reasons, and
        the second is the one that matters: the ranks must agree (an unseeded ``numpy.random.rand``
        made them perturb differently, take different branches and deadlock at the next collective --
        B6 in ``dev_docs/mpi_augmented_systems.md``), and drawing the global vector means serial,
        replicated ``mpirun`` and ``--distribute`` all start from the BIT-IDENTICAL basis, so their
        exponents are comparable to round-off instead of only statistically. The cost is an O(n*k)
        draw on every rank, once.
        """
        rng=numpy.random.default_rng(self.random_seed)
        full=rng.random((layout.n,self.k))
        return numpy.ascontiguousarray(full[layout.local_slice,:])

    def _as_vec(self,col:int,layout:"RowLayout")->DistVector:
        return DistVector(self.B[:,col],layout)

    def actions_after_initialise(self):
        problem=self.get_problem()
        T0=problem.get_current_time(dimensional=True,as_float=True)
        TS=problem.get_scaling("temporal")
        if self.waiting_time is not None:
            TW=float(self.waiting_time/TS)
        else:
            TW=0
        if self.prerelaxation_time is not None:
            TP=float(self.prerelaxation_time/TS)
        else:
            TP=0
        self._Tstart1=T0+TW
        self._t_last_gram_schmidt=self._Tstart2
        self._Tstart2=self._Tstart1+TP
        if self.gram_schmidt_dt is None:
            self._gram_schmidt_dt=0
        else:
            self._gram_schmidt_dt=float(self.gram_schmidt_dt/TS)
        super().actions_after_initialise()

    def actions_after_newton_solve(self):
        problem = self.get_problem()
        t = problem.get_current_time(dimensional=True,as_float=True)

        # Not started yet
        if t < self._Tstart1:
            return
        Tdiff = t - self._Tstart2


        # --- BDF weights ---
        ts = problem.timestepper
        w0=ts.weightBDF1(1,0)
        w1=ts.weightBDF1(1,1)
        

        if w0 == 0.0: # Stationary solve
            return
        
                

        # --- Matrices ---
        was_steady=[problem.time_stepper_pt(i).is_steady() for i in range(problem.ntime_stepper())]
        for i in range(problem.ntime_stepper()):
            problem.time_stepper_pt(i).make_steady()
        n, M_nzz, M_nr, M_val, M_ci, M_rs, J_nzz, J_nr, J_val, J_ci, J_rs = problem.assemble_eigenproblem_matrices(0.0) #type:ignore # Mass and zero Jacobian
        # shape=(M_nr, n), not (n, n): M_nr is this rank's LOCAL row count and the column indices are
        # global, so under --distribute the block is rectangular. Passing (n,n) with a short indptr is
        # the same mistake as B2 in dev_docs/mpi_augmented_systems.md and dies inside scipy with an
        # "index pointer size" complaint several frames from anything the user wrote.
        matJ=csr_matrix((J_val, J_ci, J_rs), shape=(J_nr, n)).copy()	#type:ignore        
        matM=csr_matrix((M_val, M_ci, M_rs), shape=(M_nr, n)).copy()	#type:ignore        
        for i,ws in enumerate(was_steady):
            if not ws:
                problem.time_stepper_pt(i).undo_make_steady()
        matM.eliminate_zeros() #type:ignore

        layout=self._row_layout(problem,int(M_nr),int(n))
        self._layout=layout
        la_backend=get_la_backend(problem)

        # --- Initialization ---
        # Deferred until the layout is known: the perturbations are row blocks of it, and before the
        # first assembly there is nothing that says which rows this rank owns.
        if self.B.shape[1] != self.k:
            if self.k > n:
                raise ValueError("number of Lyapunov exponents k must be less or equal to the number of degrees of freedom in the problem")
            self.B = self._draw_initial_basis(layout)
            # Prepare orthonormal random basis. The dots and norms are allreduces when distributed.
            for i in range(self.k):
                self.B[:, i] /= self._as_vec(i,layout).norm()
                for j in range(i + 1, self.k):
                    self.B[:, j] -= self._as_vec(i,layout).dot(self._as_vec(j,layout)) * self.B[:, i]
            self.Lambdas = numpy.zeros((self.k,))
        if self.B.shape[0] != layout.nrow_local:
            raise ValueError("Internal error: wrong size of perturbation vectors (%d rows, the layout "
                             "owns %d). Probably, you adapted or remeshed the problem during the "
                             "Lyapunov calculation, which is not supported."%(self.B.shape[0],layout.nrow_local))
        
        
        if not self.use_crank_nicholson_integration or self._oldJ is None: # Just implicit Euler
            if self.use_crank_nicholson_integration:
                self._oldJ=matJ.copy()
            matJ+=matM*w0
        else: # Crank-Nicolson-like update         
            tmp=self._oldJ.copy()
            self._oldJ=matJ.copy()
            matJ=matM*w0+0.5*(tmp+matJ) 

        matM.eliminate_zeros() #type:ignore
        matM.sort_indices()
        matJ.sort_indices()
        # --- Solve for the new perturbations ---
        # Through the backend rather than solve_serial: matM is this rank's row block with GLOBAL
        # column indices, so "matM @ pert" is only meaningful once the operand is whole, which is
        # what DistMatrix.matvec is for (a replicating allgather for scipy, Mat.mult for PETSc).
        matJd=la_backend.matrix(matJ,layout,int(n))
        matMd=la_backend.matrix(matM,layout,int(n))
        # One factorisation, k right-hand sides -- what the old solve_serial(1,...) followed by k
        # times solve_serial(2,...) did, kept explicitly rather than by luck. No
        # _note_external_serial_solve() here: the backend's distributed entry point owns its own
        # factorisation slot (PETSc's _aux_*), separate from the gathered Newton solve's.
        rhss=[matMd.matvec(self._as_vec(i,layout))*(-w1) for i in range(self.k)]  # No second history perturbation for Lyapunov calculation
        for i,sol in enumerate(la_backend.solve_many(matJd,rhss)):
            self.B[:,i]=sol.local

        if t-self._t_last_gram_schmidt>self._gram_schmidt_dt:
            # --- QR orthonormalization ---
            # Every norm and projection below is an allreduce when distributed, so the whole
            # factorisation R is identical on every rank and the exponents cannot drift apart.
            for i in range(self.k):
                norm=self._as_vec(i,layout).norm()
                if Tdiff>0:
                    #print("R_ii",i,norm)
                    self.Lambdas[i]+=numpy.log(norm)
                self.B[:,i]/=norm                                                
                # Gram-Schmidt
                for j in range(i+1,self.k):
                    proj=self._as_vec(i,layout).dot(self._as_vec(j,layout))
                    #print("R_ij",i,j,proj)
                    self.B[:,j]-=proj*self.B[:,i]
                    

            self._t_last_gram_schmidt=t

            # --- Output ---
            if self._outputfile is None and get_mpi_rank()==0:
                fname = (problem.get_output_directory(self.filename)if self.relative_to_output else self.filename)
                self._outputfile = open(fname, "w")

            if Tdiff > 0:
                lyap_estimate=self.Lambdas/Tdiff
                if get_mpi_rank()==0:
                    # self._outputfile is guaranteed to be opened above (on rank 0) either now or in a previous call
                    assert self._outputfile is not None
                    self._outputfile.write(f"{t}\t" + "\t".join(map(str, lyap_estimate)) + "\n")
                    self._outputfile.flush()

                if self.store_as_eigenvectors:
                    problem._last_eigenvalues = numpy.array(lyap_estimate, dtype=numpy.complex128)
                    problem._last_eigenvalues_m = numpy.zeros(len(lyap_estimate), dtype="int")
                    # Problem._last_eigenvectors is declared as a 2d complex ndarray (rows=eigenvectors), not a list:
                    # other code (e.g. Problem.calculate_eigenvalues, periodic_driving_response.py) indexes it as
                    # _last_eigenvectors[i,:] or _last_eigenvectors[0,dofidx], which fails with a TypeError on a plain list.
                    # Replicated at full GLOBAL length: an eigenvector is indexed by global equation
                    # number everywhere it is consumed (set_eigenfunction_as_dofs, the mesh data
                    # cache, the VTK output), so the perturbations are gathered here rather than
                    # handed over as row blocks. See dev_docs/mpi_eigenproblems.md section 3.
                    norms=[self._as_vec(i,layout).norm() for i in range(self.k)]
                    eigenvecs = numpy.array([self._as_vec(i,layout).to_global() for i in range(self.k)],
                                            dtype=numpy.complex128)
                    for i in range(eigenvecs.shape[0]):
                        eigenvecs[i] = eigenvecs[i] / norms[i]
                    problem._last_eigenvectors = eigenvecs
                    problem.invalidate_cached_mesh_data(only_eigens=True)
                    
                    

class LyapunovExponentCalculatorBDF2(GenericProblemHooks):
    """
    A class for calculating Lyapunov exponents. Add it to the problem by ``problem+=LyapunovExponentCalculator(...)`` and it will do the rest for you.
    It works a bit differently than the other Lyapunov exponent calculator: Here, we use the BDF2 time discretization to evolve both the state and the perturbation vectors.    
    However, note that we only may have first order time derivatives in the equations. Second order time derivatives must be rewritten as first order time derivatives before.
    Also, the time derivatives in the system must use the fully implicit "BDF2" time scheme, which is the default (unless set otherwise stated by either using ``scheme="..."`` in :py:func:`~pyoomph.expressions.generic.partial_t` or by altering :py:attr:`~pyoomph.generic.problem.Problem.default_timestepping_scheme` of the :py:class:`~pyoomph.generic.problem.Problem`).
    Gram-Schmidt ortho*normalization* is only performed if the vectors grow too large or too small, to avoid numerical issues. Otherwise only ortho*gonalization* is performed. This is required since BDF2 has multiple time levels.
    Also, instead of accumulating the Lyapunov exponents over time, we use a ring buffer to store recent growths and average over a specified time interval.

    Args:
        average_time: The time interval over which to average the Lyapunov exponents. If None, we average over the entire time
        N: The number of Lyapunov exponents to calculate. N>=2 will invoke Gram-Schmidt on the perturbation vectors. Defaults to 1.
        filename: The name of the output file. Defaults to "lyapunov.txt".
        relative_to_output: Whether to save the output file relative to the problem's output directory. Defaults to True.
        store_as_eigenvectors: Whether to store the perturbation vectors as eigenvectors. Defaults to False.
        random_seed: Seed for the initial random perturbations. The same value on every rank -- it is
            a constructor argument, so it is -- which is what keeps the ranks from diverging under
            ``mpirun``.
    """    
    def __init__(self,average_time:ExpressionNumOrNone=None,N:int=1,filename:str="lyapunov.txt",relative_to_output:bool=True,store_as_eigenvectors:bool=False,random_seed:int=0):
        super().__init__()
        self.random_seed=int(random_seed)
        self.filename=filename
        self.relative_to_output=relative_to_output
        self.store_as_eigenvectors=store_as_eigenvectors
        
        self.perturbation:list[NPFloatArray]=[] # Storing the last perturbation
        self.old_perturbation:list[NPFloatArray | None] | None=None # Storing the perturbation one step before (per-index None until it has been computed once)
        self.outputfile=None # Output file
        self.average_time=average_time
        self.ringbuffer:"deque[tuple[float,NPFloatArray]]"=deque() # (time, growth rates) of the averaging window
        self.N=N
        if self.N<=0:
            raise ValueError("N must be a positive integer")
        
    def renormalize(self,i:int):
        if self.old_perturbation is None:
            self.old_perturbation=[None for _ in range(self.N)]
        old_pert=self.old_perturbation
        nrm=numpy.linalg.norm(self.perturbation[i])
        if old_pert[i] is not None:
            # Scale the old perturbation. Note: We divide by the norm of self.perturbation to keep the ratio between both
            old_pert[i]=old_pert[i]/nrm #type:ignore # narrowed by the enclosing "is not None" check; pyright can't track subscripts by a loop variable
        # And renormalize the current perturbation to start_perturbation_norm
        self.perturbation[i]=self.perturbation[i]/nrm
    
    
    def actions_after_newton_solve(self):
        problem=self.get_problem()
        if len(self.perturbation)!=self.N:
            if self.N>problem.ndof():
                raise ValueError("number of Lyapunov exponents N must be less or equal to the number of degrees of freedom in the problem")
            # Placeholder vectors of size 0 (mismatching problem.ndof()) so the size check below triggers proper (re-)initialization
            self.perturbation=[numpy.zeros(0) for i in range(self.N)]
        if len(self.perturbation[0])!=problem.ndof():
            # Seeded, and drawn once for all N vectors from ONE generator: numpy.random.rand is not
            # identically seeded across ranks, so under mpirun the ranks perturbed differently,
            # converged differently and deadlocked at the next collective -- B6 in
            # dev_docs/mpi_augmented_systems.md. This class runs replicated (see the refusal below),
            # so every rank must draw the identical numbers, not merely reproducible ones.
            rng=numpy.random.default_rng(self.random_seed)
            draws=rng.random((self.N,problem.ndof()))*2-1
            for i in range(self.N):
                self.perturbation[i]=numpy.ascontiguousarray(draws[i])
                if self.old_perturbation is None:
                    self.old_perturbation=[None for _ in range(self.N)]
                self.old_perturbation[i]=None
                self.renormalize(i) # and scale it to the length
        
        # Open the file if necessary
        if self.outputfile is None and get_mpi_rank()==0:
            if self.relative_to_output:                
                self.outputfile=open(problem.get_output_directory(self.filename),"w")
            else:
                self.outputfile=open(self.filename,"w")
        
        t=problem.get_current_time(dimensional=True,as_float=True)
        # History time stepping weights
        if problem.timestepper.get_num_unsteady_steps_done()==0: # The first step is degraded to BDF1 by default
            w1=problem.timestepper.weightBDF1(1,1)
            w2=0            
        else:
            w1=problem.timestepper.weightBDF2(1,1)
            w2=problem.timestepper.weightBDF2(1,2)            
        # Second history perturbation
        
        if w1==0:
            return # Seems to be a stationary solve here
        

        # Get the mass matrix and the Jacobian
        # Still refused under --distribute, and for a reason specific to THIS class rather than to
        # Lyapunov exponents: it does not factorise anything of its own. It back-substitutes against
        # the factorisation the Newton solve just made, via solve_serial(op_flag=2), which needs no
        # matrix at all. There is no layout-agnostic form of that: the distributed analogue,
        # solve_distributed(op_flag=2), routes through _solve_newton_step and would apply Newton-step
        # post-processing to a Lyapunov right-hand side. LyapunovExponentCalculator (above) works
        # under --distribute because it builds and factorises its own BDF1/Crank-Nicolson operator.
        problem._require_non_distributed("Lyapunov exponent calculation with BDF2 perturbations")
        matM,matJ=None,None
        custom_assm=problem.get_custom_assembler()
        if custom_assm is not None:
            matM,matJ=custom_assm.get_last_mass_and_jacobian_matrices()
        
        if matM is None or matJ is None:
            n, M_nzz, M_nr, M_val, M_ci, M_rs, J_nzz, J_nr, J_val, J_ci, J_rs = problem.assemble_eigenproblem_matrices(0.0) #type:ignore # Mass and zero Jacobian
            # shape=(M_nr, n): M_nr is the local row count. Non-distributed here by the refusal
            # above, so the two agree today -- written correctly anyway, because a (n,n) with a short
            # indptr is exactly the B2 shape of mistake and it would be found the hard way.
            matM=csr_matrix((M_val, M_ci, M_rs), shape=(M_nr, n)).copy()	#type:ignore        
            matM.eliminate_zeros() #type:ignore
        else:
            n, J_nzz, J_val, J_rs, J_ci = problem.ndof(), len(matJ.data), matJ.data, matJ.indptr, matJ.indices
        
        # self.old_perturbation is guaranteed to be set by the initialization block above, either just now or in a previous call
        assert self.old_perturbation is not None
        old_pert=self.old_perturbation
        growths=[]
        for i in range(self.N):
            pert1=self.perturbation[i].copy() # First history perturbation
            pert2=(old_pert[i] if (old_pert[i] is not None) else self.perturbation[i]).copy() #type:ignore # pyright cannot narrow subscripts by a loop variable
            # Assemble the RHS
            rhs=-matM@(w1*pert1+w2*pert2)
            # And (re)solve the linear system for the new perturbation
            problem.get_la_solver().solve_serial(2,n,J_nzz,1,J_val,J_rs,J_ci,rhs,0,1) #type:ignore
            # Update the perturbation (rhs stores the solution after solving)
            old_pert[i]=self.perturbation[i]
            self.perturbation[i]=rhs.copy()
            # Check whether we have to renormalize

            # Calculate the growth, update the ring buffer and write the current estimate to the file
            # old_pert[i] was just set to self.perturbation[i] (a NPFloatArray) above; pyright cannot narrow
            # subscript expressions indexed by a loop variable, hence the non-None guarantee is asserted here
            growths.append(numpy.log(numpy.linalg.norm(self.perturbation[i])/numpy.linalg.norm(old_pert[i]))) #type:ignore


            ss=numpy.linalg.norm(self.perturbation[i])
            if ss>1e30 or ss<1e-10:
                self.renormalize(i)            
        
            
        # Gram-Schmidt
        if self.N>1:
            new_basis=self.perturbation.copy()
            for i in range(self.N):            
                for j in range(i):
                    new_basis[i]-=numpy.dot(self.perturbation[j],self.perturbation[i])/numpy.dot(self.perturbation[j],self.perturbation[j])*self.perturbation[j]
                    #new_basis[i]-=numpy.dot(self.perturbation[j],self.perturbation[i])*self.perturbation[j] # Using the fact the we renormalize every step
            self.perturbation=new_basis
        
        self.ringbuffer.append((t,numpy.array(growths)))
        # this is essentially 1/(t2-t1)*log(norm(t2)/norm(t1)) by accumulating over the buffer and using the logarithmic addition rule
        if len(self.ringbuffer)>=2:       
            # Must skip the first entry in the sum, since for 2 elements, we only have one dt differeces     
            accumulated=numpy.zeros_like(self.ringbuffer[-1][1])
            for i,r in enumerate(self.ringbuffer):
                if i>0:
                    accumulated=accumulated+r[1]
            ljap_estimate=accumulated/(self.ringbuffer[-1][0]-self.ringbuffer[0][0])
            if self.average_time is not None:
                while self.ringbuffer[0][0]<t-self.average_time and len(self.ringbuffer)>1:
                    self.ringbuffer.popleft()
            if get_mpi_rank()==0:
                # self.outputfile is guaranteed to be opened above (on rank 0) either now or in a previous call
                assert self.outputfile is not None
                self.outputfile.write(str(t)+"\t"+"\t".join(map(str,ljap_estimate))+"\n")
                self.outputfile.flush()

            if self.store_as_eigenvectors:
                problem._last_eigenvalues=numpy.array(ljap_estimate,dtype=numpy.complex128)
                problem._last_eigenvalues_m=numpy.zeros(len(ljap_estimate),dtype="int")
                # Problem._last_eigenvectors is declared as a 2d complex ndarray (rows=eigenvectors), not a list:
                # other code (e.g. Problem.calculate_eigenvalues, periodic_driving_response.py) indexes it as
                # _last_eigenvectors[i,:] or _last_eigenvectors[0,dofidx], which fails with a TypeError on a plain list.
                eigenvecs=numpy.array([p.copy() for p in self.perturbation],dtype=numpy.complex128)
                for i in range(eigenvecs.shape[0]):
                    eigenvecs[i]=eigenvecs[i]/numpy.linalg.norm(eigenvecs[i])
                problem._last_eigenvectors=eigenvecs

    
    def actions_after_initialise(self):
        problem=self.get_problem()
        if problem.is_distributed():
            raise RuntimeError("Lyapunov exponent calculation is not supported for distributed problems")
        super().actions_after_initialise()