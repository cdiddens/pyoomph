from ..generic.problem import GenericProblemHooks
import numpy
from scipy.sparse import csr_matrix
from ..expressions import ExpressionNumOrNone, ExpressionOrNum
from ..typings import NPFloatArray,List,Optional
from collections import deque

from ..generic.problem import GenericProblemHooks
import numpy
from scipy.sparse import csr_matrix
from ..expressions import ExpressionNumOrNone
from ..typings import NPFloatArray,List,Optional

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
    """
    def __init__(self,k: int = 1,waiting_time:ExpressionOrNum=None, prerelaxation_time:ExpressionOrNum=None,use_crank_nicholson_integration:bool=False, filename="lyapunov.txt",relative_to_output=True,store_as_eigenvectors:bool=False,gram_schmidt_dt:ExpressionNumOrNone=None):
        super().__init__()
        self.k = k
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

    def actions_after_initialise(self):
        problem=self.get_problem()
        T0=problem.get_current_time(as_float=True)
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

    def actions_after_newton_solve(self):
        problem = self.get_problem()
        t = problem.get_current_time(as_float=True)

        # Not started yet
        if t < self._Tstart1:
            return
        Tdiff = t - self._Tstart2

        # --- Initialization ---
        if self.B.shape[1] != self.k:
            if self.k > problem.ndof():
                raise ValueError("number of Lyapunov exponents k must be less or equal to the number of degrees of freedom in the problem")
            self.B = numpy.random.rand(problem.ndof(), self.k)            
            # Prepare orthonormal random basis
            for i in range(self.k):
                self.B[:, i] /= numpy.linalg.norm(self.B[:, i])
                for j in range(i + 1, self.k):
                    self.B[:, j] -= numpy.dot(self.B[:, i], self.B[:, j]) * self.B[:, i]            
            self.Lambdas = numpy.zeros((self.k,))
        if self.B.shape[0] != problem.ndof():
            print(self.B.shape)
            raise ValueError("Internal error: wrong size of perturbation vectors. Probably, you adapted or remeshed the problem during the Lyapunov calculation, which is not supported.")

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
        matJ=csr_matrix((J_val, J_ci, J_rs), shape=(n, n)).copy()	#type:ignore        
        matM=csr_matrix((M_val, M_ci, M_rs), shape=(n, n)).copy()	#type:ignore        
        for i,ws in enumerate(was_steady):
            if not ws:
                problem.time_stepper_pt(i).undo_make_steady()
        matM.eliminate_zeros() #type:ignore

        la = problem.get_la_solver()
        
        
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
        # --- Solve for the new perturbations ---
        la = problem.get_la_solver()
        la.solve_serial(1,n,matJ.nnz,1,matJ.data,matJ.indices,matJ.indptr,numpy.zeros(n),0,1)

        for i in range(self.k):
            pert = self.B[:,i].copy()
            rhs = -matM @ (w1 * pert)  # No second history perturbation for Lyapunov calculation
            la.solve_serial(2,n,matJ.nnz,1,matJ.data,matJ.indices,matJ.indptr,rhs,0,1)
            self.B[:,i] = rhs[:]

        if t-self._t_last_gram_schmidt>self._gram_schmidt_dt:
            # --- QR orthonormalization ---            
            for i in range(self.k):
                norm=numpy.linalg.norm(self.B[:,i])
                if Tdiff>0:
                    #print("R_ii",i,norm)
                    self.Lambdas[i]+=numpy.log(norm)
                self.B[:,i]/=norm                                                
                # Gram-Schmidt
                for j in range(i+1,self.k):
                    proj=numpy.dot(self.B[:,i],self.B[:,j])
                    #print("R_ij",i,j,proj)
                    self.B[:,j]-=proj*self.B[:,i]
                    

            self._t_last_gram_schmidt=t

            # --- Output ---
            if self._outputfile is None:
                fname = (problem.get_output_directory(self.filename)if self.relative_to_output else self.filename)
                self._outputfile = open(fname, "w")

            if Tdiff > 0:
                lyap_estimate=self.Lambdas/Tdiff
                self._outputfile.write(f"{t}\t" + "\t".join(map(str, lyap_estimate)) + "\n")
                self._outputfile.flush()

                if self.store_as_eigenvectors:
                    problem._last_eigenvalues = lyap_estimate.copy()
                    problem._last_eigenvalues_m = numpy.zeros(len(lyap_estimate), dtype="int")
                    problem._last_eigenvectors = [self.B[:, i].copy() for i in range(self.k)]
                    for i, ev in enumerate(problem._last_eigenvectors):
                        problem._last_eigenvectors[i] = ev / numpy.linalg.norm(ev)
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
    """    
    def __init__(self,average_time:ExpressionNumOrNone=None,N:int=1,filename:str="lyapunov.txt",relative_to_output:bool=True,store_as_eigenvectors:bool=False):
        super().__init__()
        self.filename=filename
        self.relative_to_output=relative_to_output
        self.store_as_eigenvectors=store_as_eigenvectors
        
        self.perturbation:List[NPFloatArray]=[] # Storing the last perturbation
        self.old_perturbation:Optional[List[NPFloatArray]]=None # Storing the perturbation one step before
        self.outputfile=None # Output file
        self.average_time=average_time
        self.ringbuffer=deque()
        self.N=N
        if self.N<=0:
            raise ValueError("N must be a positive integer")
        
    def renormalize(self,i:int):
        if self.old_perturbation is None:
            self.old_perturbation=[None]*self.N
        nrm=numpy.linalg.norm(self.perturbation[i])
        if self.old_perturbation[i] is not None:
            # Scale the old perturbation. Note: We divide by the norm of self.perturbation to keep the ratio between both
            self.old_perturbation[i]=self.old_perturbation[i]/nrm
        # And renormalize the current perturbation to start_perturbation_norm
        self.perturbation[i]=self.perturbation[i]/nrm
    
    
    def actions_after_newton_solve(self):
        problem=self.get_problem()
        if len(self.perturbation)!=self.N:
            if self.N>problem.ndof():
                raise ValueError("number of Lyapunov exponents N must be less or equal to the number of degrees of freedom in the problem")
            self.perturbation=[[] for i in range(self.N)]
        if len(self.perturbation[0])!=problem.ndof():
            for i in range(self.N):
                self.perturbation[i]=(numpy.random.rand(problem.ndof())*2-1)
                if self.old_perturbation is None:
                    self.old_perturbation=[None]*self.N
                self.old_perturbation[i]=None
                self.renormalize(i) # and scale it to the length
        
        # Open the file if necessary
        if self.outputfile is None:
            if self.relative_to_output:                
                self.outputfile=open(problem.get_output_directory(self.filename),"w")
            else:
                self.outputfile=open(self.filename,"w")
        
        t=problem.get_current_time(as_float=True)
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
        matM,matJ=None,None
        custom_assm=problem.get_custom_assembler()
        if custom_assm is not None:
            matM,matJ=custom_assm.get_last_mass_and_jacobian_matrices()
        
        if matM is None or matJ is None:
            n, M_nzz, M_nr, M_val, M_ci, M_rs, J_nzz, J_nr, J_val, J_ci, J_rs = problem.assemble_eigenproblem_matrices(0.0) #type:ignore # Mass and zero Jacobian
            matM=csr_matrix((M_val, M_ci, M_rs), shape=(n, n)).copy()	#type:ignore        
            matM.eliminate_zeros() #type:ignore
        else:
            n, J_nzz, J_val, J_rs, J_ci = problem.ndof(), len(matJ.data), matJ.data, matJ.indptr, matJ.indices
        
        growths=[]
        for i in range(self.N):
            pert1=self.perturbation[i].copy() # First history perturbation
            pert2=(self.old_perturbation[i] if (self.old_perturbation[i] is not None) else self.perturbation[i]).copy()
            # Assemble the RHS
            rhs=-matM@(w1*pert1+w2*pert2)
            # And (re)solve the linear system for the new perturbation
            problem.get_la_solver().solve_serial(2,n,J_nzz,1,J_val,J_rs,J_ci,rhs,0,1)
            # Update the perturbation (rhs stores the solution after solving)
            self.old_perturbation[i]=self.perturbation[i]
            self.perturbation[i]=rhs.copy()
            # Check whether we have to renormalize

            # Calculate the growth, update the ring buffer and write the current estimate to the file
            growths.append(numpy.log(numpy.linalg.norm(self.perturbation[i])/numpy.linalg.norm(self.old_perturbation[i])))


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
            ljap_estimate=sum(r[1] for i,r in enumerate(self.ringbuffer) if i>0)/(self.ringbuffer[-1][0]-self.ringbuffer[0][0])
            if self.average_time is not None:
                while self.ringbuffer[0][0]<t-self.average_time and len(self.ringbuffer)>1:
                    self.ringbuffer.popleft()
            self.outputfile.write(str(t)+"\t"+"\t".join(map(str,ljap_estimate))+"\n")
            self.outputfile.flush()

            if self.store_as_eigenvectors:
                problem._last_eigenvalues=numpy.array(ljap_estimate)
                problem._last_eigenvalues_m=numpy.zeros(len(ljap_estimate),dtype="int")
                problem._last_eigenvectors=self.perturbation.copy()
                for i,ev in enumerate(problem._last_eigenvectors):
                    problem._last_eigenvectors[i]=ev/numpy.linalg.norm(ev)
