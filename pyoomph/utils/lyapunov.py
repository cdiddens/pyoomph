from ..generic.problem import GenericProblemHooks
import numpy
from scipy.sparse import csr_matrix
from ..expressions import ExpressionNumOrNone
from ..typings import NPFloatArray,List,Optional
from collections import deque

class LyapunovExponentCalculator(GenericProblemHooks):
    """
    A class for calculating Lyapunov exponents. Add it to the problem by ``problem+=LyapunovExponentCalculator(...)`` and it will do the rest for you.
    However, note that we only may have first order time derivatives in the equations. Second order time derivatives must be rewritten as first order time derivatives before.
    Also, the time derivatives in the system must use the fully implicit "BDF2" time scheme, which is the default (unless set otherwise stated by either using ``scheme="..."`` in :py:func:`~pyoomph.expressions.generic.partial_t` or by altering :py:attr:`~pyoomph.generic.problem.Problem.default_timestepping_scheme` of the :py:class:`~pyoomph.generic.problem.Problem`).

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
