import scipy.sparse
import scipy.sparse.linalg
from .problem import Problem
from ..expressions import GlobalParameter,ExpressionNumOrNone
from ..typings import *
import _pyoomph
import numpy,scipy
from .assembly import CustomAssemblyBase
from ..solvers.generic import DefaultMatrixType,EigenMatrixManipulatorBase

from scipy.sparse import csr_matrix      

def get_hopf_lyapunov_coefficient(problem:Problem,param:Union[GlobalParameter,str],FD_delta:float=1e-5,FD_param_delta:float=1e-5,omega:Optional[float]=None,q:Optional[NPComplexArray]=None,mu0:float=0,omega_epsilon:float=1e-5):
    # Taken from § 10.2 of Yuri A. Kuznetsov, Elements of Applied Bifurcation Theory, Fourth Edition, Springer, 2004
    # Also implemented analogously in pde2path, file hogetnf.m 
    # XXX Here is the generalization of the code with mass matrix
    # In Kuznetsov, it is assumed that the dofs x can be cast to a complex number z via z=<p,x>
    # If you have a mass matrix, it must be z=<p,Mx> where M is the mass matrix
    # The rest stays the same
    # Thereby, when having the eigenvector q corresponding to lambda=i*omega (i*omega*M*q=A*q=-J*q)
    # we must find the adjoint eigenvector p, which fulfills A^T*p=-i*omega*M^T*p
    # and it must be normalized so that <p,Mq>=1 and <p,M(q*)>=0
    #
    # Thereby, you can take the dynamics close to the bifurcations:
    #
    #   M*partial_t(x)=A*x+F(x,alpha)
    #
    # Multiply it via <p| and use the definition of z (see above), to get
    #
    #   partial_t(z)=<p,A*x> + <p,F(x,alpha)>
    #
    # write <p,A*x>=<A^T*p,x>=<lambda* *M^T*p,x>=lambda*<p,Mx>=lambda*z
    # so that we end up at
    # 
    #   partial_t(z)=lambda*z + <p,F(x,alpha)>
    #   
    # Now, we can apply the method of Kuznetsov to get the Lyapunov coefficient.
    #
    # However, when mapping the expansion coefficients h_lk of the quadratic and cubic terms of the normal form
    #
    #      partial_t(z)=lambda*z + sum_{2<=l+k<=3} h_lk z^l z*^k
    #
    # We get the following equation:
    #
    #   h20 = (2*i*omega*M-A)^(-1) B(q,q)       instead of     h20 = (2*i*omega*I-A)^(-1) B(q,q) [5.30 in the book]
    #   h11= -A^(-1) B(q,qb)                    remains the same    [5.31 in the book]
    #
    # The cubic term [see 5.32 in the book] is also augmented with A mass matrix
    #
    #   (i*omega0*M-A)*h21=C(q,q,qb)+B(qb,h20)+2*B(q,h11)-2*c1*M*q
    #
    # But the argument is the same: When applying <p| to this equation, the lhs vanishes.
    # Since <p,q>=1, we get
    #
    #   c1=1/2*<p,C(q,q,qb)+2*B(q,h11)+B(qb,h20)>
    #
    # as in the book
    
    esolver=problem.get_eigen_solver()
    
    if isinstance(param,str):
        param=problem.get_global_parameter(param)

    eigensolve_kwargs={}
        
    
    u=problem.get_current_dofs()[0]
    def nodalf(up):
        problem.set_current_dofs(up)
        res=-numpy.array(problem.get_residuals())        
        return res
    
    def solve_mat(A,rhs):
        #return numpy.linalg.solve(A.toarray(),rhs) # TODO: Improve here
        return scipy.sparse.linalg.spsolve(A,rhs)# TODO: Improve here to use e.g. Pardiso (however, requires complex support)
    
    delt=FD_delta
    
    
        
    
    ntstep=problem.ntime_stepper()
    was_steady=[False]*ntstep
    for i in range(ntstep):
        ts=problem.time_stepper_pt(i)
        was_steady[i]=ts.is_steady()
        ts.make_steady()
    


    n, M_nzz, M_nr, M_val, M_ci, M_rs, J_nzz, J_nr, J_val, J_ci, J_rs = problem.assemble_eigenproblem_matrices(0) #type:ignore
    M=csr_matrix((M_val, M_ci, M_rs), shape=(n, n))	#TODO: Is csr or csc?
    A=csr_matrix((-J_val, J_ci, J_rs), shape=(n, n))
    AT=A.transpose().tocsr()
    MT=M.transpose().tocsr()
    
    if omega is None or q is None:
        print("Solving for omega and q")
        eval,evect,_,_=problem.get_eigen_solver().solve(2,custom_J_and_M=(A,M),**eigensolve_kwargs)                    
        omega0=numpy.imag(eval[0])
        if omega0<0:
            omega0=-omega0
            qR=numpy.real(evect[1])
            qI=numpy.imag(evect[1])
        else:
            qR=numpy.real(evect[0])
            qI=numpy.imag(evect[0])
        mu0=numpy.real(eval[0])
    else:
        qR=numpy.real(q)
        qI=numpy.imag(q)
        omega0=omega
    print("PREDOT",numpy.dot(qR,qI))
    qdenom=numpy.dot(qR,qR)+numpy.dot(qI,qI)
    qR/=numpy.sqrt(qdenom)
    qI/=numpy.sqrt(qdenom)
    print("POSTDOT",numpy.dot(qR,qI))
    
    if False or (numpy.amax(numpy.abs(A@qR+omega0*M@qI))>1e-7 or numpy.amax(-omega0*M@qR+A*qI)>1e-7):
        print("Given q does not fulfill the eigenvector equation. Resolving it.")
        esolv_kwargs=eigensolve_kwargs.copy()
        if esolver.supports_target():
            esolv_kwargs["target"]=1j*omega0
        eval,evects,_,_=problem.get_eigen_solver().solve(1,custom_J_and_M=(A,M),**esolv_kwargs,shift=(1j+omega_epsilon)*omega0,v0=numpy.array(q),sort=False,quiet=False)
        #for ev,evec in zip(eval,evects):
        #    print("EVAL",ev)
        #    #evec=numpy.conjugate(evec)
        #    print(numpy.amax(numpy.abs(-A@evec+ev*M@evec)))
        #    print(numpy.amax(numpy.abs(A@numpy.real(evec)+numpy.imag(ev)*M@numpy.imag(evec))))
        #    print(numpy.amax(numpy.abs(-numpy.imag(ev)*M@numpy.real(evec)+A@numpy.imag(evec))))
        omega0=numpy.imag(eval[0])
        if omega0<0:
            omega0=-omega0
            qR=numpy.real(evects[0])
            qI=-numpy.imag(evects[0])
        else:
            qR=numpy.real(evects[0])
            qI=numpy.imag(evects[0])
        
        #print("AFTER SETTING")    
        #print(numpy.amax(numpy.abs(-A@(qR+1j*qI)+1j*omega0*M@(qR+1j*qI))))
        #print(numpy.amax(numpy.abs(A@qR+omega0*M@qI)))
        #print(numpy.amax(numpy.abs(-omega0*M@qR+A@qI)))
        
    if numpy.abs(numpy.dot(qR,qI))>1e-7:
        raise ValueError("qR and qI are not orthogonal. This is likely an issue with the eigenvalue solver. Please check the eigenvalue solver settings.")
    
    esolv_kwargs=eigensolve_kwargs.copy()
    if esolver.supports_target():
        esolv_kwargs["target"]=-1j*omega0
        evalT,evectT,_,_=problem.get_eigen_solver().solve(1,custom_J_and_M=(AT,MT),**esolv_kwargs,shift=-(1j+omega_epsilon)*omega0,v0=numpy.conjugate(q),sort=False,quiet=False)   # TODO: Is MT right here?                 
    else:
        problem.deactivate_bifurcation_tracking()        
        problem.set_custom_assembler(HopfTracker(problem,param.get_name(),numpy.conjugate(q),omega=-omega0,left_eigenvector=True,eigenscale=1))
        problem.solve()
        evalT=problem.get_last_eigenvalues()
        evectT=problem.get_last_eigenvectors()
        if evalT*omega0>0:
            evalT=-evalT
            evectT=numpy.conjugate(evectT)        
        problem.set_custom_assembler(None)
        
        #raise RuntimeError("Eigenvalue solver does not support target. Please use a different eigenvalue solver.")
    #print("GOT",evalT,evectT)
    #print("Omega0",omega0)
    if numpy.imag(evalT[0])<0 and numpy.abs(numpy.imag(evalT[0])+omega0)<1e-3:                    
        #print("Omega0'[0]",numpy.imag(evalT[0]))
        pR=numpy.real(evectT[0])
        pI=numpy.imag(evectT[0])
        #print("Omega0 for q",omega0,"and for p",numpy.imag(evalT[0]),"sum",numpy.imag(evalT[0])+omega0)
        #print("Precheck: P Matrix equations (should be zero)")
        #print(numpy.amax(AT@pR-omega0*MT@pI))
        #print(numpy.amax(omega0*MT@pR+AT@pI))
    else:
        print("For omega0=",omega0,"we found ",numpy.imag(evalT[0]))
        raise ValueError("Could not find the correct eigenvector. This is likely an issue with the eigenvalue solver. Please check the eigenvalue solver settings.")

    #p=MT@(pR+pI*1j)
    #pR=numpy.real(p) 
    #pI=numpy.imag(p)
    #print("EIGENGL",-1j*omega0*(MT*p)-AT*p)
    #print("qR",qR)
    #print("qI",qI)
    #print("pR",pR)
    #print("pI",pI)
    verbose=True
    
    # XXX Here is the generalization of the code with mass matrix
    Mq=M@(qR+qI*1j)
    MqR=numpy.real(Mq)
    MqI=numpy.imag(Mq)
    #print("<p,Mq>",numpy.vdot(pR+1j*pI,M@(qR+1j*qI)))
    #print("<p,Mq*>",numpy.vdot(pR+1j*pI,M@(qR-1j*qI)))
    #exit()
    #theta=numpy.angle(numpy.dot(pR,qR)+numpy.dot(pI,qI)+(numpy.dot(pR,qI)-numpy.dot(pI,qR))*1j); 
    theta=numpy.angle(numpy.dot(pR,MqR)+numpy.dot(pI,MqI)+(numpy.dot(pR,MqI)-numpy.dot(pI,MqR))*1j); 
    p=(pR+pI*1j)*numpy.exp(1j*theta)
    pR=numpy.real(p) 
    pI=numpy.imag(p)
    pnorm=numpy.dot(pR,MqR)+numpy.dot(pI,MqI)
    if numpy.abs(pnorm)<1e-10:                
        print(numpy.dot(qR,qR)+numpy.dot(qI,qI),numpy.dot(qR,qI))
        print(numpy.dot(pR,qR)+numpy.dot(pI,qI),numpy.dot(pR,qI)-numpy.dot(pI,qR))
        print(numpy.linalg.norm(pR),numpy.linalg.norm(pI))
        print("<p,q>",numpy.vdot(pR+1j*pI,qR+1j*qI))
        print("<p,q*>",numpy.vdot(pR+1j*pI,qR-1j*qI))
        for theta in numpy.linspace(0,2*numpy.pi,20):
            p=(pR+pI*1j)*numpy.exp(1j*theta)
            pRt=numpy.real(p) 
            pIt=numpy.imag(p)
            print("Theta",theta,"<p,q>",numpy.vdot(pRt+1j*pIt,qR+1j*qI),"<p,q*>",numpy.vdot(pRt+1j*pIt,qR-1j*qI))
        #exit()
        #print("pnorm is very small. This is likely an issue with the eigenvalue solver. Please check the eigenvalue solver settings.")
        #raise ValueError("pnorm is very small. This is likely an issue with the eigenvalue solver. Please check the eigenvalue solver settings.")
        #p*=numpy.exp(1j*numpy.pi/3)
        #pR=numpy.real(p)
        #pI=numpy.imag(p)
        
        #print("q",numpy.dot(qR,qR)+numpy.dot(qI,qI))
        #print("<qR,qI>",numpy.dot(qR,qI))
       
        #pR=numpy.real(p) 
        #pI=numpy.imag(p)
        #pnorm=numpy.dot(pR,qR)+numpy.dot(pI,qI)
        #print(pnorm)
        
        #exit()
        #pnorm=1
    pR=pR/pnorm
    pI=pI/pnorm
    
    qunitscale=max(numpy.amax(numpy.absolute(qR)),numpy.amax(numpy.absolute(qI)))
    punitscale=max(numpy.amax(numpy.absolute(pR)),numpy.amax(numpy.absolute(pI)))
    print("Q unitscale",qunitscale)
    print("Q magnitude",numpy.linalg.norm(qR),numpy.linalg.norm(qI))
    print("P unitscale",punitscale)
    print("P magnitude",numpy.linalg.norm(pR),numpy.linalg.norm(pI))
    qRus=qR/qunitscale
    qIus=qI/qunitscale
    pRus=pR/punitscale
    pIus=pI/punitscale
    
    
    if verbose:
        print("Step 1 : Checking equations")
        print("Q Matrix equations (should be zero)")
        print(numpy.amax(numpy.abs(A@qR+omega0*M@qI)))
        print(numpy.amax(-omega0*M@qR+A@qI))
        print("P Matrix equations (should be zero)")
        print(numpy.amax(AT@pR-omega0*MT@pI))
        print(numpy.amax(omega0*MT@pR+AT@pI))
        print("Normalisation (1,0) required")
        print(numpy.dot(qR,qR)+numpy.dot(qI,qI),numpy.dot(qR,qI))
        print(numpy.dot(pR,MqR)+numpy.dot(pI,MqI),numpy.dot(pR,MqI)-numpy.dot(pI,MqR))
        #exit()
        #print("THIS gives:")
        #print("qR",qR)
        #print("qI",qI)
        #print("pR",pR)
        #print("pI",pI)
    
    
    
    
    f0=nodalf(u)
    def d2f(direct):                    
        res_hess=-numpy.array(problem.get_second_order_directional_derivative(direct))        
        if False:
            fp=nodalf(u+delt*direct)
            fm=nodalf(u-delt*direct)
            problem.set_current_dofs(u)
            res_fd=(fm-2*f0+fp)/(delt**2)        
            print("DIFFERENCE",numpy.amax(numpy.absolute(res_hess-res_fd)),"FD",numpy.amax(numpy.absolute(res_fd)),"HESS",numpy.amax(numpy.absolute(res_hess)),"DIRECT",numpy.amax(numpy.absolute(res_hess)))
            return res_hess
        return res_hess
        
    def d3f(direct):
        direct_scale=numpy.amax(numpy.abs(direct))                
        if direct_scale<1e-10:
            direct_scale=1
        else:
            direct_scale=1/direct_scale        
        direct_scale=1
        problem.set_current_dofs(u+delt*direct*direct_scale)
        res_hessp=-numpy.array(problem.get_second_order_directional_derivative(direct))
        problem.set_current_dofs(u-delt*direct*direct_scale)
        res_hessm=-numpy.array(problem.get_second_order_directional_derivative(direct))
        problem.set_current_dofs(u)        
        res_hess=0.5*(res_hessp-res_hessm)/(delt*direct_scale)
        
        if False:        
            fmm=nodalf(u-2*delt*direct)
            fm=nodalf(u-delt*direct)
            fp=nodalf(u+delt*direct)
            fpp=nodalf(u+2*delt*direct)
            res_fd=(-0.5*fmm+fm-fp+0.5*fpp)/(delt**3)
            print("DIFFERENCE 3d order",numpy.amax(numpy.absolute(res_hess-res_fd)),"FD",numpy.amax(numpy.absolute(res_fd)),"HESS",numpy.amax(numpy.absolute(res_hess)),"DIRECT",numpy.amax(numpy.absolute(direct)))
            #print("RESP",res_hessp,res_hessm,"DIRECT",direct)
            return res_hess # (-0.5*fmm+fm-fp+0.5*fpp)/(delt**3)
        return res_hess
    
    
    # Step 2 
    # TODO: Make via Hessian products instead
    
    
    
    a=d2f(qR)
    b=d2f(qI)
    c=0.25*(d2f(qR+qI)-d2f(qR-qI))    
    
    if verbose:
        print("Step 2")
        print("A magnitude",numpy.linalg.norm(a))
        print("B magnitude",numpy.linalg.norm(b))
        print("C magnitude",numpy.linalg.norm(c))
        #a/=numpy.linalg.norm(a)
        #b/=numpy.linalg.norm(b)
        #c/=numpy.linalg.norm(c)
        #print("a",a)
        #print("b",b)
        #print("c",c)

    #step 3
    # I don't think this part of pde2path is correct
    #r=solve_mat(A,M*(a+b))
    #sv=solve_mat(-A+2j*M*omega0,M*(a-b+2j*c))
    # I think it should be this
    r=solve_mat(A,a+b)
    sv=solve_mat(-A+2j*M*omega0,a-b+2j*c)
    sR=numpy.real(sv)
    sI=numpy.imag(sv)
    if verbose:
        print("Step 3")
        #print("r",r)
        #print("sR",sR)
        #print("sI",sI)
        print("CHECKING r",numpy.amax(numpy.absolute(A@r-M@(a+b))),"Rmagnitude",numpy.linalg.norm(r))
        print("CHECKING sR",numpy.amax(numpy.absolute(-A@sR-2*omega0*M@sI - M@(a-b))),"Sr magnitude",numpy.linalg.norm(sR))
        print("CHECKING sI",numpy.amax(numpy.absolute(2*omega0*M@sR-A@sI - 2*M@c)), "SI magnitude",numpy.linalg.norm(sI))

    # step 4
    sig1=0.25*numpy.dot(pR,d2f(qR+r)-d2f(qR-r))
    sig2=0.25*numpy.dot(pI,d2f(qI+r)-d2f(qI-r)) # TODO: In the book, it is pI, in pde2path is is pR    
    sig=sig1+sig2
    if verbose:
        print("Step 4")
        print("sig1",sig1)
        print("sig2",sig2)
        print("sig",sig)
    
    # step 5
    d1=0.25*numpy.dot(pR,d2f(qR+sR)-d2f(qR-sR))
    d2=0.25*numpy.dot(pR,d2f(qI+sI)-d2f(qI-sI))
    d3=0.25*numpy.dot(pI,d2f(qR+sI)-d2f(qR-sI)) # TODO: In the book, it is pI, in pde2path is is pR
    d4=0.25*numpy.dot(pI,d2f(qI+sR)-d2f(qI-sR)) # TODO: In the book, it is pI, in pde2path is is pR    
    d0=d1+d2+d3-d4
    if verbose:
        print("Step 5")
        print("d1",d1)
        print("d2",d2)
        print("d3",d3)
        print("d4",d4)
        print("d0",d0)
    
    # Step 6        
    g1=numpy.dot(pR,d3f(qR))
    g2=numpy.dot(pI,d3f(qI))
    g3=numpy.dot(pR+pI,d3f(qR+qI))
    g4=numpy.dot(pR-pI,d3f(qR-qI))    
    g0=2*(g1+g2)/3+(g3+g4)/6
    
    if verbose:
        print("Step 6")
        print("g1",g1)
        print("g2",g2)
        print("g3",g3)
        print("g4",g4)
        print("g0",g0)
    
    # step 7
    ga=(g0-2*sig+d0)/abs(2*omega0)
    c1=abs(omega0)*ga
    if verbose:
        print("Step 7")
        print("ga",ga)
        print("c1",c1)
  

    # Get dmu/dparam    
    old=param.value
    if False:
        param.value+=FD_param_delta    
        evalsP,_=problem.solve_eigenproblem(2,**eigensolve_kwargs,shift=(1j+omega_epsilon)*omega0)
        mu=numpy.real(evalsP[0])    
        mup=-(mu-mu0)/FD_param_delta
    else:
        problem.activate_eigenbranch_tracking("complex",eigenvector=q,eigenvalue=1j*omega0)        
        problem.solve()
        mu1=numpy.real(problem.get_last_eigenvalues()[0])
        problem.go_to_param(**{param.get_name():param.value+FD_param_delta})
        mu2=numpy.real(problem.get_last_eigenvalues()[0])
        mup=-(mu2-mu1)/FD_param_delta
        if abs(mup)>1e12*abs(c1):
            raise ValueError("Likely, the orbit originates like in the van der Pol oscillator. The first Lyapunov coefficient seems to be zero. Please manually specify dparam and the orbit amplitude")
            
        
        problem.deactivate_bifurcation_tracking()
        problem.set_current_dofs(u)                            

    param.value=old
    
    if ((c1>0 and mup>0) or (c1<0 and mup<0)):
        dlam=-1
    else:
        dlam=1
    al=numpy.sqrt(-dlam*mup/c1); 
    if verbose:
        print("dmu_dparam",mup,al)
        print("Will return",dlam,al)
    

    problem.set_current_dofs(u) 
    for i in range(ntstep):
        if not was_steady[i]:
            problem.time_stepper_pt(i).undo_make_steady()

    return ga,dlam,al,qR,qI


















DofAugmentationSpecifications=_pyoomph.DofAugmentations

class MultiAssembleRequest:
    def __init__(self,problem:Problem):
        self._what=[]
        self._contributions=[]
        self._hessian_vectors=[]
        self._hessian_vector_indices=[]
        self._parameters=[]
        self.problem=problem
        
    def _resolve_hessian_vector_index(self,V):
        for i,w in enumerate(self._hessian_vectors):
            if V is w:
                return i
        self._hessian_vectors.append(V)
        return len(self._hessian_vectors)-1
    
    def R(self,contribution=""):
        self._what.append("residuals")
        self._contributions.append(contribution)
        return self
        
    def J(self,contribution=""):
        self._what.append("jacobian")
        self._contributions.append(contribution)
        return self
        
    def M(self,contribution=""):
        self._what.append("mass_matrix")
        self._contributions.append(contribution)
        return self
        
    def dRdp(self,parameter:Union[str,GlobalParameter],contribution=""):
        self._what.append("dresiduals_dparameter")
        self._contributions.append(contribution)
        self._parameters.append(parameter)
        return self
    
    def dJdp(self,parameter:Union[str,GlobalParameter],contribution=""):
        self._what.append("djacobian_dparameter")
        self._contributions.append(contribution)
        self._parameters.append(parameter)
        return self
        
    def dMdp(self,parameter:Union[str,GlobalParameter],contribution=""):
        self._what.append("dmass_matrix_dparameter")
        self._contributions.append(contribution)
        self._parameters.append(parameter)
        return self
        
    def dJdU(self,vector:NPFloatArray,contribution="",transposed=False):
        if transposed:
            self._what.append("hessian_vector_product_transposed")
        else:
            self._what.append("hessian_vector_product")
        self._contributions.append(contribution)
        self._hessian_vector_indices.append(self._resolve_hessian_vector_index(vector))
        return self
        
    def dMdU(self,vector:NPFloatArray,contribution="",transposed=False):
        if transposed:
            self._what.append("mass_matrix_hessian_vector_product_transposed")
        else:
            self._what.append("mass_matrix_hessian_vector_product")
        self._contributions.append(contribution)
        self._hessian_vector_indices.append(self._resolve_hessian_vector_index(vector))
        return self
        
        
    def assemble(self):
        n,vectors,csrdatas,return_indices=self.problem._assemble_multiassembly(self._what,self._contributions,self._parameters,self._hessian_vectors,self._hessian_vector_indices)
        nmatrix=len(csrdatas)//2
        nvectors=len(vectors)-nmatrix
        matrices=[]
        #print("RETURN INDICES",return_indices)
        for i in range(nmatrix):
            matrices.append(scipy.sparse.csr_matrix((vectors[nvectors+i],csrdatas[2*i+1],csrdatas[2*i]),shape=(n,n)))
        res=[]
        for r in return_indices:
            if r>=0:
                res.append(vectors[r])
            else:
                res.append(matrices[-(r+1)])        
        return res

class AugmentedAssemblyHandler(CustomAssemblyBase):
    def __init__(self):
        super().__init__()
        self._augdof_spec=None
        
    def initialize(self):
        #self._augdof_spec._in_specification=True
        #self._augdof_spec._problem=self.problem
        self._augdof_spec=self.problem._create_dof_augmentation()
        self.define_augmented_dofs(self.get_augmented_dofs())
        self.problem._add_augmented_dofs(self._augdof_spec)
        print("Dofs after augmentation",self.problem.ndof())
        
        
    def finalize(self):
        self.problem._reset_augmented_dof_vector_to_nonaugmented()
        
        
    def get_augmented_dofs(self)->DofAugmentationSpecifications:
        return self._augdof_spec
            
    def define_augmented_dofs(self,dofs:DofAugmentationSpecifications):
        raise NotImplementedError("define_augmented_dofs not implemented")
    
    def define_augmented_residuals(self,dofs:DofAugmentationSpecifications):
        raise NotImplementedError("define_augmented_residuals not implemented")
    
    def define_augmented_residuals_and_jacobian(self,dofs:DofAugmentationSpecifications):
        raise NotImplementedError("define_augmented_residuals_and_jacobian not implemented")
    
    def get_base_residuals_and_jacobian(self)->Tuple[NPFloatArray,DefaultMatrixType]:
        old=self.problem.use_custom_residual_jacobian
        self.problem.use_custom_residual_jacobian=False
        R,J=self.problem.assemble_jacobian(with_residual=True)
        self.problem.use_custom_residual_jacobian=old
        return R,J
    
    def get_base_dresiduals_dparameter(self,parameter:Union[str,GlobalParameter])->NPFloatArray:
        raise NotImplementedError("get_base_dresiduals_dparameter not implemented")
    
    def get_base_dresiduals_and_djacobian_dparameter(self,parameter:Union[str,GlobalParameter])->Tuple[NPFloatArray,DefaultMatrixType]:
        raise NotImplementedError("get_base_dresiduals_and_djacobian_dparameter not implemented")
    
    def get_base_hessian_vector_product(self,vector:NPFloatArray)->NPFloatArray:
        raise NotImplementedError("get_base_hessian_vector_product not implemented")

    def start_multiassembly(self):
        return MultiAssembleRequest(self.problem)
    
    def as_matrix_column(self,arr):
        if isinstance(arr,list):
            arr=numpy.array(arr)
        return scipy.sparse.csr_matrix(arr.reshape(-1,1))
    
    def as_matrix_row(self,arr):
        if isinstance(arr,list):
            arr=numpy.array(arr)
        return scipy.sparse.csr_matrix(arr.reshape(1,-1))
    
    def get_residuals_and_jacobian(self,require_jacobian:bool,dparameter:Optional[str]=None)->Union[NPFloatArray,Tuple[NPFloatArray,DefaultMatrixType]]:               
        raise NotImplementedError("get_residuals_and_jacobian not implemented")

class CustomBifurcationTracker(AugmentedAssemblyHandler):
    """
    A generic class to write custom bifurcation trackers.
    """
    def __init__(self,problem:Problem):
        super().__init__()
        self.problem=problem
        
    def get_real_eigenvector_guess(self,eigenvector:Union[NPAnyArray,int]=0,normalize:bool=True)->NPAnyArray:
        """
        Get a real eigenvector guess. This can be either an index to the previously solved eigenvalues or an eigenvector.        

        Args:
            eigenvector: Index to a calculated eigenvector or the eigenvector itself
            normalize: Normalize the eigenvector to |V|=1

        Returns:
            The eigenvector as array
        """
        if eigenvector is None:
            eigenvector=0
        elif isinstance(eigenvector,int):
            if eigenvector>=len(self.problem.get_last_eigenvalues()):
                raise RuntimeError("Eigenvalue index out of range")
            eigenvector=numpy.real(self.problem.get_last_eigenvectors()[eigenvector])
        if normalize:
            eigenvector/=numpy.linalg.norm(eigenvector)
        return eigenvector

    def get_complex_eigenvector_guess(self,eigenvector:Union[NPAnyArray,int],normalize:bool=True)->NPAnyArray:
        """
        Get a complex eigenvector guess. This can be either an index to the previously solved eigenvalues or an eigenvector.        

        Args:
            eigenvector: Index to a calculated eigenvector or the eigenvector itself
            normalize: Normalize the eigenvector so that <Re(V),Im(V)>=0 and <Re(V),Re(V)>=1

        Returns:
            The eigenvector as array
        """
        if eigenvector is None:
            eigenvector=0
        elif isinstance(eigenvector,int):
            if eigenvector>=len(self.problem.get_last_eigenvalues()):
                raise RuntimeError("Eigenvalue index out of range")
            eigenvector=self.problem.get_last_eigenvectors()[eigenvector]
        if normalize:
            GrGr=numpy.dot(numpy.real(eigenvector),numpy.real(eigenvector))
            GrGi=numpy.dot(numpy.real(eigenvector),numpy.imag(eigenvector))
            GiGi=numpy.dot(numpy.imag(eigenvector),numpy.imag(eigenvector))
            n_phi_samples,n_iter,best_phi,best_GrGr=30,15,0,GrGr            
            for phi in numpy.linspace(0,2*numpy.pi,n_phi_samples):
                res=-GiGi*numpy.sin(phi)*numpy.cos(phi) - GrGi*numpy.sin(phi)*numpy.sin(phi) + GrGi*numpy.cos(phi)*numpy.cos(phi) + GrGr*numpy.sin(phi)*numpy.cos(phi)
                GrGr_new=numpy.dot(numpy.real(eigenvector)*numpy.cos(phi)+numpy.imag(eigenvector)*numpy.sin(phi),numpy.real(eigenvector)*numpy.cos(phi)+numpy.imag(eigenvector)*numpy.sin(phi))
                success=False
                for iter in range(n_iter):
                    J=GiGi*numpy.sin(phi)*numpy.sin(phi) - GiGi*numpy.cos(phi)*numpy.cos(phi) - 4*GrGi*numpy.sin(phi)*numpy.cos(phi) - GrGr*numpy.sin(phi)*numpy.sin(phi) + GrGr*numpy.cos(phi)*numpy.cos(phi)
                    if numpy.abs(J)<1.0e-10:
                        break # Singular Jacobian
                    phi-=res/J
                    res=-GiGi*numpy.sin(phi)*numpy.cos(phi) - GrGi*numpy.sin(phi)*numpy.sin(phi) + GrGi*numpy.cos(phi)*numpy.cos(phi) + GrGr*numpy.sin(phi)*numpy.cos(phi)
                    if numpy.abs(res)<1.0e-10:
                        success=True
                        break
                if not success:
                    continue                
                # Test whether it maximizes <Re(eigenvector),Re(eigenvector)>
                GrGr_new=GrGr*numpy.cos(phi)*numpy.cos(phi) + GiGi*numpy.sin(phi)*numpy.sin(phi) - 2*GrGi*numpy.sin(phi)*numpy.cos(phi)
                if GrGr_new>best_GrGr:      
                    best_GrGr=GrGr_new
                    best_phi=phi
            eigenvector=numpy.exp(1j*best_phi)*eigenvector
    
            if False:
                pR,pI=numpy.real(eigenvector),numpy.imag(eigenvector)            
                def optimize(theta):
                    p=(pR+pI*1j)*numpy.exp(1j*theta)
                    return abs(numpy.dot(numpy.real(p),numpy.imag(p))),abs(numpy.dot(numpy.imag(p),numpy.imag(p)))
                besttheta,bestval,smallest_im=0,None,None
                for testtheta in numpy.linspace(0,2*numpy.pi,100):
                    val,imim=optimize(testtheta)
                    if (bestval is None or val<bestval) and (smallest_im is None or imim<smallest_im):
                        bestval=val
                        besttheta=testtheta
                        smallest_im=imim
                eigenvector/=numpy.linalg.norm(numpy.real(eigenvector))
                #theta=scipy.optimize.minimize_scalar(optimize,bounds=(0,2*numpy.pi),method="bounded",options={"xatol":1e-15,"maxiter":100}).x
                theta=scipy.optimize.root_scalar(lambda t:optimize(t)[0],x0=besttheta).root
                eigenvector=(pR+pI*1j)*numpy.exp(1j*theta)
                eigenvector/=numpy.linalg.norm(numpy.real(eigenvector))
                
            eigenvector/=numpy.linalg.norm(numpy.real(eigenvector))
            
            #print("Eigenvector ReIm",numpy.dot(numpy.real(eigenvector),numpy.imag(eigenvector)))
            #print("Eigenvector ReRe",numpy.dot(numpy.real(eigenvector),numpy.real(eigenvector)))
            #print("Eigenvector ImIm",numpy.dot(numpy.imag(eigenvector),numpy.imag(eigenvector)))
            #exit()
        return eigenvector

    def store_eigenvector(self,eigenvects:Dict[Union[float,complex],NPAnyArray]):
        self.problem._last_eigenvalues=numpy.array(list(eigenvects.keys()))
        self.problem._last_eigenvectors=numpy.array([numpy.array(v) for v in eigenvects.values()])   
        self.problem._last_eigenvalues_m=None
        self.problem._last_eigenvalues_k=None          


class FoldTracker(CustomBifurcationTracker):
    """
    A custom fold tracker. This class can be used to track a fold bifurcation.
    However, it might be slightly slower compared to the internal fold tracker.
    Along this, you can develop your own bifurcation trackers of e.g. co-dimension-2-bifurcations.
    
    Args:
        problem: The problem to track the bifurcation for
        parameter: The parameter to track the bifurcation for
        eigenvector: The eigenvector to track the bifurcation for. This can be an index to the previously solved eigenvalues or the eigenvector itself
        eigenscale: The scale of the eigenvector internally considered. Internally, the eigenvalue will have the magnitude |V|=eigenscale, in the output, the eigenvector will be normalized to |V|=1
        nonlinear_length_constraint: If False, we demand <V,V0>=eigenscale, if True, we demand <V,V>=eigenscale^2. Nonlinear length constraints can require a more sophisticated initial guess, but can be better for arclength continuation along a long branch, where <V,V0>=0 could in principle occur.
    """
    def __init__(self,problem:Problem,parameter,eigenvector:Union[NPAnyArray,int]=0,eigenscale:float=1,nonlinear_length_constraint:bool=False):
        super().__init__(problem)
        self.parameter=parameter
        self.V0=self.get_real_eigenvector_guess(eigenvector,normalize=True)
        self.V0=self.V0/numpy.linalg.norm(self.V0)
        self.eigenscale=eigenscale
        self.nonlinear_length_constraint=nonlinear_length_constraint
        
        
        
    def define_augmented_dofs(self,dofs:DofAugmentationSpecifications):
        # dofs will be grouped in (U,V,p)
        dofs.add_vector(self.V0*self.eigenscale)
        dofs.add_parameter(self.parameter)

    def get_residuals_and_jacobian(self,require_jacobian:bool,dparameter:Optional[str]=None)->Union[NPFloatArray,Tuple[NPFloatArray,DefaultMatrixType]]:               
        V,=self.get_augmented_dofs().split(startindex=1,endindex=2) # Get the eigenvector solution
        # Request the residuals and Jacobian of the non-augmented system
        assembly=self.start_multiassembly()                                            
        if require_jacobian:
            assert dparameter is None, "dparameter not supported for require_jacobian=True"
            # If we need the augmented Jacobian, we also need dR/dP and dJ_ik/dU_j V_k
            R,J,dRdP,dJdP,HV=assembly.R().J().dRdp(self.parameter).dJdp(self.parameter).dJdU(V).assemble() 
        else:
            if dparameter:
                # This happens during arclength continuation in another parameter
                dRdp,dJdp=assembly.dRdp(dparameter).dJdp(dparameter).assemble()
                return numpy.hstack([dRdp,dJdp@V,0]) # leave here with the derivative of the residuals with respect to the other parameter
                                        
            R,J=assembly.R().J().assemble() # Only residuals and Jacobian are requested and required
            
        nl=self.nonlinear_length_constraint
        Raug=numpy.hstack([R,J@V,numpy.dot(V,(V if nl else self.V0))-self.eigenscale*(self.eigenscale if nl else 1)]) # Augmented dof vector
        if require_jacobian:            
            col=lambda C:self.as_matrix_column(C)
            row=lambda R:self.as_matrix_row(R)
            # Augmented Jacobian
            Jaug=scipy.sparse.block_array(
                [[J,None,col(dRdP)],
                 [HV,J,col(dJdP@V)],
                 [None,row(2*V if nl else self.V0),None]]).tocsr()            
            return Raug,Jaug
        else:
            return Raug

    def actions_after_successful_newton_solve(self)->None:
        V,=self.get_augmented_dofs().split(startindex=1,endindex=2)
        V=V/numpy.linalg.norm(V)
        self.store_eigenvector({0:numpy.array(V)})
          


class PitchForkTracker(CustomBifurcationTracker):
    """
    Simple pitchfork bifurcation tracker. This might be slightly slower than the internal pitchfork tracker.        
    
    Args:
        problem: The problem to track the bifurcation for
        parameter: The parameter to track the bifurcation for
        eigenvector: The eigenvector to track the bifurcation for. This can be an index to the previously solved eigenvalues or the eigenvector itself
        symmetry_vector: The symmetry vector to track the bifurcation for. This can be an index to the previously solved eigenvalues or custom vector. If None (default), the eigenvector will be used as symmetry vector.
        eigenscale: The scale of the eigenvector internally considered. Internally, the eigenvalue will have the magnitude |V|=eigenscale, in the output, the eigenvector will be normalized to |V|=1
        nonlinear_length_constraint: If False, we demand <V,V0>=eigenscale, if True, we demand <V,V>=eigenscale^2. Nonlinear length constraints can require a more sophisticated initial guess, but can be better for arclength continuation along a long branch, where <V,V0>=0 could in principle occur.
        
    """
    def __init__(self, problem, parameter,eigenvector=0,symmetry_vector=None,eigenscale:float=1,nonlinear_length_constraint:bool=False):
        super().__init__(problem)
        self.parameter=parameter
        self.V0=self.get_real_eigenvector_guess(eigenvector)
        self.V0/=numpy.linalg.norm(self.V0)
        if symmetry_vector is None:
            self.S=self.V0 # Symmetry constraint vector just copied
        else:
            # Or any prescribed symmetry vector
            self.S=self.get_real_eigenvector_guess(symmetry_vector)
        self.eigenscale=eigenscale    
        self.nonlinear_length_constraint=nonlinear_length_constraint
        
    def define_augmented_dofs(self, dofs):
        dofs.add_vector(self.V0*self.eigenscale)                
        dofs.add_parameter(self.parameter)
        dofs.add_scalar(0) # slack variable
        
    def get_residuals_and_jacobian(self,require_jacobian:bool,dparameter:Optional[str]=None)->Union[NPFloatArray,Tuple[NPFloatArray,DefaultMatrixType]]:               
        U,V,p,eps=self.get_augmented_dofs().split(startindex=0) # Get the eigenvector solution and the slack variable
        eps=eps[0] # Get the scalar value of the slack variable (split dofs are all vectors)
        # Request the residuals and Jacobian of the non-augmented system
        assembly=self.start_multiassembly()        
        if require_jacobian:
            assert dparameter is None, "dparameter not supported for require_jacobian=True"
            R,J,dRdP,dJdP,HV=assembly.R().J().dRdp(self.parameter).dJdp(self.parameter).dJdU(V).assemble() # Assemble all quantities, will be given in the order of the requests
        else:
            if dparameter is not None:            
                # This happens during arclength continuation in another parameter
                dRdp,dJdp=assembly.dRdp(dparameter).dJdp(dparameter).assemble() 
                # leave here with the derivative of the residuals with respect to the other parameter
                return numpy.hstack([dRdp,dJdp@V,0,0]) 
                        
            R,J=assembly.R().J().assemble() # Only residuals and Jacobian are requested and required
            
        nl=self.nonlinear_length_constraint
        Raug=numpy.hstack([R+eps*self.S,J@V,numpy.dot(V,(V if nl else self.V0))-self.eigenscale*(self.eigenscale if nl else 1),numpy.dot(U,self.S)]) 
        if require_jacobian:            
            col=lambda C:self.as_matrix_column(C)
            row=lambda R:self.as_matrix_row(R)
            # Augmented Jacobian
            Jaug=scipy.sparse.block_array(
                [[J,None,col(dRdP),col(self.S)],
                 [HV,J,col(dJdP@V),None],
                 [None,row(2*V if nl else self.V0),None,None],
                 [row(self.S),None,None,None]]).tocsr()            
            return Raug,Jaug
        else:
            return Raug

    def actions_after_successful_newton_solve(self)->None:
        V,=self.get_augmented_dofs().split(startindex=1,endindex=2)
        V=V/numpy.linalg.norm(V)
        self.store_eigenvector({0:V})        
        

class HopfTracker(CustomBifurcationTracker):
    """This class can be used to track a Hopf bifurcation.
    

    Args:
        problem: The problem to track the bifurcation for
        parameter: The parameter to track the bifurcation for
        eigenvector: The eigenvector to track the bifurcation for. This can be an index to the previously solved eigenvalues or the eigenvector itself
        omega: The frequency of the Hopf bifurcation. If None, the frequency of the eigenvector will be used.
        eigenscale: The scale of the eigenvector internally considered. Internally, the eigenvalue will have the magnitude |Re(V)|=eigenscale, in the output, the eigenvector will be normalized to |V|=1
        nonlinear_length_constraint: If False, we demand <Re(V),Re(V0)>=eigenscale, if True, we demand <Re(V),Re(V)>=eigenscale^2. Nonlinear length constraints can require a more sophisticated initial guess, but can be better for arclength continuation along a long branch, where <Re(V),Re(V0)>=0 could in principle occur.
    """
    def __init__(self,problem:Problem,parameter,eigenvector=0,omega=None,eigenscale:float=1,nonlinear_length_constraint:bool=False,left_eigenvector:bool=False):
        super().__init__(problem)
        self.parameter=parameter                    
        self.eigenvector=self.get_complex_eigenvector_guess(eigenvector)
        if omega is None:
            if isinstance(eigenvector,int):
                self.omega=numpy.imag(self.problem.get_last_eigenvalues()[0])
            else:
                raise RuntimeError("You need to provide the frequency of the Hopf bifurcation when using a custom eigenvector guess")
        else:
            self.omega=omega
        self.C=numpy.real(self.eigenvector)
        self.eigenscale=eigenscale
        self.nonlinear_length_constraint=nonlinear_length_constraint
        self.left_eigenvector=left_eigenvector
    
    def define_augmented_dofs(self, dofs):
        dofs.add_vector(numpy.real(self.eigenvector)*self.eigenscale)
        dofs.add_vector(numpy.imag(self.eigenvector)*self.eigenscale)
        dofs.add_parameter(self.parameter)
        dofs.add_scalar(self.omega)
        
    def get_residuals_and_jacobian(self,require_jacobian:bool,dparameter:Optional[str]=None)->Union[NPFloatArray,Tuple[NPFloatArray,DefaultMatrixType]]:               
        Vr,Vi,p,omega=self.get_augmented_dofs().split(startindex=1) # Get all the augmented dofs
        omega=omega[0] # Get the scalar value of the frequency variable (split dofs are all vectors)        
        assembly=self.start_multiassembly()        
        if require_jacobian:
            assert dparameter is None, "dparameter not supported for require_jacobian=True"
            # If we need the augmented Jacobian, we also need dR/dP and dJ_ik/dU_j V_k
            print("Currently at Hopf tracking with omega=",omega)
            R,J,M,dRdP,dJdP,dMdP,HVr,HVi,dMdUVr,dMdUVi=assembly.R().J().M().dRdp(self.parameter).dJdp(self.parameter).dMdp(self.parameter).dJdU(Vr,transposed=self.left_eigenvector).dJdU(Vi,transposed=self.left_eigenvector).dMdU(Vr,transposed=self.left_eigenvector).dMdU(Vi,transposed=self.left_eigenvector).assemble() # Assemble all quantities, will be given in the order of the requests
        else:
            if dparameter is not None:
                # This happens during arclength continuation in another parameter
                dRdp,dJdp,dMdp=assembly.dRdp(dparameter).dJdp(dparameter).dMdp(dparameter).assemble() 
                # leave here with the derivative of the residuals with respect to the other parameter
                if self.left_eigenvector:
                    return numpy.hstack([dRdp,-dJdp.transpose()@Vr+omega*dMdp.transpose()@Vi,-dJdp.transpose()@Vi-omega*dMdp.transpose()@Vr,0,0])
                else:
                    return numpy.hstack([dRdp,-dJdp@Vr+omega*dMdp@Vi,-dJdp@Vi-omega*dMdp@Vr,0,0])
            
            R,J,M=assembly.R().J().M().assemble() # Only residuals and Jacobian are requested and required
        
        nl=self.nonlinear_length_constraint
        if self.left_eigenvector:
            Raug=numpy.hstack([R,-J.transpose()@Vr+omega*M.transpose()@Vi,-J.transpose()@Vi-omega*M.transpose()@Vr,numpy.dot(Vr,Vr if nl else self.C)-self.eigenscale*(self.eigenscale if nl else 1),numpy.dot(Vi,Vr if nl else self.C)]) 
        else:
            Raug=numpy.hstack([R,-J@Vr+omega*M@Vi,-J@Vi-omega*M@Vr,numpy.dot(Vr,Vr if nl else self.C)-self.eigenscale*(self.eigenscale if nl else 1),numpy.dot(Vi,Vr if nl else self.C)]) 
        if require_jacobian:            
            col=lambda C:self.as_matrix_column(C)
            row=lambda R:self.as_matrix_row(R)
            if self.left_eigenvector:
                #raise NotImplementedError("Left eigenvector not implemented for augmented Jacobian")
                Jaug=scipy.sparse.block_array(
                    [[J,None,None,col(dRdP),None],
                    [-HVr+omega*dMdUVi,-J.transpose(),omega*M.transpose(),col(-dJdP.transpose()@Vr+omega*dMdP.transpose()@Vi),col(M.transpose()@Vi)],
                    [-HVi-omega*dMdUVr,-omega*M.transpose(),-J.transpose(),col(-dJdP.transpose()@Vi-omega*dMdP.transpose()@Vr),col(-M.transpose()@Vr)],
                    [None,row(2*Vr if nl else self.C),None,None,None],
                    [None,row(Vi) if nl else None,row(Vr if nl else self.C),None,None]]).tocsr()            
            else:
                # Augmented Jacobian
                Jaug=scipy.sparse.block_array(
                    [[J,None,None,col(dRdP),None],
                    [-HVr+omega*dMdUVi,-J,omega*M,col(-dJdP@Vr+omega*dMdP@Vi),col(M@Vi)],
                    [-HVi-omega*dMdUVr,-omega*M,-J,col(-dJdP@Vi-omega*dMdP@Vr),col(-M@Vr)],
                    [None,row(2*Vr if nl else self.C),None,None,None],
                    [None,row(Vi) if nl else None,row(Vr if nl else self.C),None,None]]).tocsr()            
            return Raug,Jaug
        else:
            return Raug
        
    def actions_after_successful_newton_solve(self)->None:
        Vr,Vi,p,omega=self.get_augmented_dofs().split(startindex=1)
        self.store_eigenvector({(1j*omega[0]):(numpy.array(Vr)+numpy.array(Vi)*1j)})        
        

class _NormalModeBifurcationTrackerBase(CustomBifurcationTracker):
    def __init__(self, problem:Problem,eigenvector:int=0,azimuthal_m:Optional[int]=None,cartesian_k:ExpressionNumOrNone=None,eigenscale:float=1,nonlinear_length_constraint:bool=False):    
        super().__init__(problem)
        self.eigenscale=eigenscale
        self.nonlinear_length_constraint=nonlinear_length_constraint
        if self.problem._azimuthal_mode_param_m is not None:
            self.azimuthal=True            
            if cartesian_k is not None:
                raise RuntimeError("Cannot supply a Cartesian wave number k for azimuthal mode bifurcation tracking")            
            if azimuthal_m is not None:
                self.azimuthal_m=azimuthal_m
            else:
                self.azimuthal_m=self.problem.get_last_eigenmodes_m()[eigenvector]
            self.problem._azimuthal_mode_param_m.value=self.azimuthal_m
            self.real_contribution=self.problem._azimuthal_stability.real_contribution_name
            self.imag_contribution=self.problem._azimuthal_stability.imag_contribution_name
        elif self.problem._normal_mode_param_k is not None:
            self.azimuthal=False
            if azimuthal_m is not None:
                raise RuntimeError("Cannot supply an azimuthal mode m for Cartesian normal mode bifurcation tracking")
            if cartesian_k is not None:
                self.cartesian_k=cartesian_k
            else:
                self.cartesian_k=self.problem.get_last_eigenmodes_k()[eigenvector]
            self.problem._normal_mode_param_k.value=self.cartesian_k
            self.real_contribution=self.problem._cartesian_normal_mode_stability.real_contribution_name
            self.imag_contribution=self.problem._cartesian_normal_mode_stability.imag_contribution_name
        else:
            raise RuntimeError("Normal mode bifurcation tracking requires either azimuthal mode or Cartesian normal mode by calling setup_for_stability_analysis with the right kwargs first") 
        
        self.eigenvector=self.get_complex_eigenvector_guess(eigenvector)
        #print(numpy.dot(numpy.imag(self.eigenvector),numpy.real(self.eigenvector)))
        #print(numpy.dot(numpy.imag(self.eigenvector),numpy.imag(self.eigenvector)))
        #print(numpy.dot(numpy.real(self.eigenvector),numpy.real(self.eigenvector)))
        
        
        
        self.has_imag=numpy.dot(numpy.imag(self.eigenvector),numpy.imag(self.eigenvector))>1e-15 # TODO: Make it adjustable
        if not self.has_imag:
            self.has_imag=self.problem._set_solved_residual(self.imag_contribution,False,False)
            self.problem._set_solved_residual("",False,False)
            if self.has_imag:
                print("Strange, eigenvector is real, but it has imaginary jacobian contribution")
                #raise RuntimeError("Strange, eigenvector is real, but it has imaginary jacobian contribution")
        self.lambda0=numpy.real(self.problem.get_last_eigenvalues()[eigenvector])
        if self.has_imag: # TODO: Is this really the case? Can't we have a Hopf bifurcation for normal mode expansions? I think so, e.g. on a partial_t(u,2)=div(grad(u))+alpha*u, we can have a Hopf bifurcation
            #self.eigenvector=self.get_complex_eigenvector_guess(eigenvector)
            self.V0=numpy.real(self.eigenvector)/numpy.linalg.norm(numpy.real(self.eigenvector))
            self.omega=numpy.imag(self.problem.get_last_eigenvalues()[eigenvector])            
        else:
            self.eigenvector=self.get_real_eigenvector_guess(eigenvector)
            self.V0=self.eigenvector/numpy.linalg.norm(self.eigenvector)
            self.omega=0
            
        self.base_zero_dofs,self.eigen_zero_dofs=self.get_forced_to_zero_dofs()

    def patch_residuals(self,eigen:bool,R:Union[List[NPFloatArray],NPFloatArray]):
        if not isinstance(R,list):
            R=[R]
        res=[]
        for r in R:
            r=numpy.array(r)            
            r[numpy.array(list(self.eigen_zero_dofs if eigen else self.base_zero_dofs),dtype="int64")]=0.0
            res.append(r)
        return res
    
    def patch_matrices(self,eigen:bool,J:Union[DefaultMatrixType,List[DefaultMatrixType]],M:Union[DefaultMatrixType,List[DefaultMatrixType]]=[])->Tuple[DefaultMatrixType,...]:
        if not isinstance(J,list):
            J=[J]
        if not isinstance(M,list):
            M=[M]
        N=J[0].shape[0]
        Adiag=numpy.ones(N)
        Adiag[numpy.array(sorted(list(self.eigen_zero_dofs if eigen else self.base_zero_dofs)),dtype=numpy.int64)] = 0.0
        Bdiag=1-Adiag
        A=scipy.sparse.spdiags(Adiag, [0], N, N).tocsr()
        B=scipy.sparse.spdiags(Bdiag, [0], N, N).tocsr()
        res=[]
        for Jmat in J:
            res.append(A@Jmat+B)
        for Mmat in M:
            res.append(A@Mmat)
        return res

    def get_forced_to_zero_dofs(self):
        if self.azimuthal:
            base_zero_dofs=self.problem._equation_system._get_forced_zero_dofs_for_eigenproblem(self.problem.get_eigen_solver(),0,None)                         
            eigen_zero_dofs=self.problem._equation_system._get_forced_zero_dofs_for_eigenproblem(self.problem.get_eigen_solver(),self.azimuthal_m,None) 
        else:            
            base_zero_dofs=self.problem._equation_system._get_forced_zero_dofs_for_eigenproblem(self.problem.get_eigen_solver(),None,0)                         
            eigen_zero_dofs=self.problem._equation_system._get_forced_zero_dofs_for_eigenproblem(self.problem.get_eigen_solver(),None,self.cartesian_k) 
        base_zero_dofs=self.problem.dof_strings_to_global_equations(base_zero_dofs)            
        eigen_zero_dofs=self.problem.dof_strings_to_global_equations(eigen_zero_dofs)
        return base_zero_dofs,eigen_zero_dofs
    
    def define_augmented_dofs(self, dofs):
        dofs.add_vector(0+numpy.real(self.eigenvector)*self.eigenscale)
        if self.has_imag:
            dofs.add_vector(numpy.imag(self.eigenvector)*self.eigenscale)
        if self.parameter is None:
            dofs.add_scalar(self.lambda0)
        else:
            dofs.add_parameter(self.parameter)
        if self.has_imag:
            dofs.add_scalar(self.omega)

    def actions_after_successful_newton_solve(self):
        if self.has_imag:
            Vr,Vi,lam,omega=self.get_augmented_dofs().split(startindex=1)            
            lam=lam[0] if self.parameter is None else 0
            self.store_eigenvector({(lam+1j*omega[0]):(numpy.array(Vr)+numpy.array(Vi)*1j)})
        else:
            Vr,lam=self.get_augmented_dofs().split(startindex=1)
            lam=lam[0] if self.parameter is None else 0
            self.store_eigenvector({lam:numpy.array(Vr)})            
            
            
class NormalModeBifurcationTracker(_NormalModeBifurcationTrackerBase):
    def __init__(self, problem:Problem,parameter:str,eigenvector:int=0,azimuthal_m:Optional[int]=None,cartesian_k:ExpressionNumOrNone=None,eigenscale:float=1,nonlinear_length_constraint:bool=False):
        super().__init__(problem,eigenvector,azimuthal_m,cartesian_k,eigenscale,nonlinear_length_constraint)
        self.parameter=parameter
        

                
    def get_residuals_and_jacobian(self,require_jacobian:bool,dparameter:Optional[str]=None)->Union[NPFloatArray,Tuple[NPFloatArray,DefaultMatrixType]]:
        nl=self.nonlinear_length_constraint
        if not self.has_imag:
            Vr,p=self.get_augmented_dofs().split(startindex=1)
            if not require_jacobian:
                if dparameter is not None:
                    dRdp,dJRdp=self.start_multiassembly().dRdp(dparameter).dJdp(dparameter,self.real_contribution)
                    dJRdp,=self.patch_matrices(eigen=True,J=dJRdp)
                    dRdp,=self.patch_residuals(eigen=False,R=[dRdp])
                    return numpy.hstack([dRdp,dJRdp@Vr,0.0])                                
                else:
                    R,JR=self.start_multiassembly().R().J(self.real_contribution).assemble()
                    JR,=self.patch_matrices(eigen=True,J=JR)
                    R,=self.patch_residuals(eigen=False,R=[R])
                    return numpy.hstack([R,JR@Vr,numpy.dot(Vr,Vr if self.nonlinear_length_constraint else self.V0)-self.eigenscale*(self.eigenscale if self.nonlinear_length_constraint else 1)])                                
            else:
                assert dparameter is None, "dparameter not supported for require_jacobian=True"
                R,J,JR,HJVr,dJRdP=self.start_multiassembly().R().J().J(self.real_contribution).dJdU(Vr).dJdp(self.parameter).assemble()
                J,=self.patch_matrices(eigen=False,J=J)
                R,=self.patch_residuals(eigen=False,R=[R])
                JR,dJRdP=self.patch_matrices(eigen=True,J=[JR,dJRdP])
                col=lambda C:self.as_matrix_column(C)
                row=lambda R:self.as_matrix_row(R)                                
                Raug=numpy.hstack([R,JR,numpy.dot(Vr,Vr if self.nonlinear_length_constraint else self.V0)-self.eigenscale*(self.eigenscale if self.nonlinear_length_constraint else 1)])                                
                Jaug=scipy.sparse.block_array(
                    [[J,None,None],
                     [HJVr,JR,col(dJRdP@Vr)],
                     [None,row(2*Vr if nl else self.V0),None]]).tocsr()
                return Raug,Jaug
        else:
            Vr,Vi,p,omega=self.get_augmented_dofs().split(startindex=1)
            omega=omega[0]
            if not require_jacobian:
                if dparameter is not None:
                    assm=self.start_multiassembly().dRdp(dparameter).dJdp(dparameter,self.real_contribution).dJdp(dparameter,self.imag_contribution)
                    assm.dMdp(dparameter,self.real_contribution).dMdp(dparameter,self.imag_contribution)
                    dRdp,dJRdp,dJIdp,dMRdp,dMIdp=assm.assemble()
                    dJRdp,dJIdp,dMRdp,dMIdp=self.patch_matrices(eigen=True,J=[dJRdp,dJIdp],M=[dMRdp,dMIdp])                    
                    d_eq_V_re_dp=-dJIdp*Vi + dJRdp*Vr +  omega*(-dMIdp*Vr - dMRdp*Vi)
                    d_eq_V_im_dp=dJIdp*Vr + dJRdp*Vi + omega*(-dMIdp*Vi + dMRdp*Vr)                    
                    return numpy.hstack([dRdp,d_eq_V_re_dp,d_eq_V_im_dp,0,0])                                
                else:
                    R,JR,JI,MR,MI=self.start_multiassembly().R().J(self.real_contribution).J(self.imag_contribution).M(self.real_contribution).M(self.imag_contribution).assemble()
                    R,=self.patch_residuals(eigen=False,R=[R])
                    JR,JI,MR,MI=self.patch_matrices(eigen=True,J=[JR,JI],M=[MR,MI])                    
                    eq_V_re=-JI*Vi + JR*Vr +  omega*(-MI*Vr - MR*Vi)
                    eq_V_im=JI*Vr + JR*Vi  + omega*(-MI*Vi + MR*Vr)
                    norm_constr=numpy.dot(Vr,Vr if self.nonlinear_length_constraint else self.V0)-self.eigenscale*(self.eigenscale if self.nonlinear_length_constraint else 1)
                    rot_constr=numpy.dot(Vi,Vr if self.nonlinear_length_constraint else self.V0)
                    eq_V_re,eq_V_im=self.patch_residuals(eigen=True,R=[eq_V_re,eq_V_im])
                    #print("MAX RES IN R",numpy.max(numpy.abs(R)))
                    #print("MAX RES IN eq_V_re",numpy.max(numpy.abs(eq_V_re)))
                    #print("MAX RES IN eq_V_im",numpy.max(numpy.abs(eq_V_im)))
                    #print("MAX RES IN norm_constr",numpy.max(numpy.abs(norm_constr)))
                    #print("MAX RES IN rot_constr",numpy.max(numpy.abs(rot_constr)))
                    return numpy.hstack([R,eq_V_re,eq_V_im,norm_constr,rot_constr])                                
            else:                
                assm=self.start_multiassembly().R().dRdp(self.parameter).J().J(self.real_contribution).J(self.imag_contribution).M(self.real_contribution).M(self.imag_contribution)
                assm.dJdU(Vr,self.real_contribution).dJdU(Vi,self.real_contribution).dJdU(Vr,self.imag_contribution).dJdU(Vi,self.imag_contribution)
                assm.dMdU(Vr,self.real_contribution).dMdU(Vi,self.real_contribution).dMdU(Vr,self.imag_contribution).dMdU(Vi,self.imag_contribution)
                assm.dJdp(self.parameter,self.real_contribution).dJdp(self.parameter,self.imag_contribution).dMdp(self.parameter,self.real_contribution).dMdp(self.parameter,self.imag_contribution)
                R,dRdp,J,JR,JI,MR,MI, HJRVR,HJRVI,HJIVR,HJIVI, HMRVR,HMRVI,HMIVR,HMIVI,dJRdp,dJIdp,dMRdp,dMIdp=assm.assemble()
                J,=self.patch_matrices(eigen=False,J=J)
                JR,JI,dJRdp,dJIdp,MR,MI,dMRdp,dMIdp=self.patch_matrices(eigen=True,J=[JR,JI,dJRdp,dJIdp],M=[MR,MI,dMRdp,dMIdp])                                    
                #HJRVR,HJRVI,HJIVR,HJIVI, HMRVR,HMRVI,HMIVR,HMIVI=self.patch_matrices(eigen=True,J=[HJRVR,HJRVI,HJIVR,HJIVI],M=[HMRVR,HMRVI,HMIVR,HMIVI])
                eq_V_re=JR@Vr -JI@Vi  - omega*(MI@Vr + MR@Vi)
                d_eq_V_re_dU=HJRVR - HJIVI - omega*(HMIVR + HMRVI)
                d_eq_V_re_dVr=JR  - omega*MI
                d_eq_V_re_dVi=-JI  - omega*MR
                d_eq_V_re_dp=dJRdp@Vr -dJIdp@Vi  - omega*(dMIdp@Vr + dMRdp@Vi)
                eq_V_im=JI@Vr + JR@Vi  + omega*(MR@Vr-MI@Vi)
                d_eq_V_im_dU=HJIVR + HJRVI  + omega*(HMRVR-HMIVI)
                d_eq_V_im_dVr=JI  + omega*MR
                d_eq_V_im_dVi=JR  - omega*MI
                d_eq_V_im_dp=dJIdp@Vr + dJRdp@Vi  + omega*(dMRdp@Vr-dMIdp@Vi)
                norm_constr=numpy.dot(Vr,Vr if self.nonlinear_length_constraint else self.V0)-self.eigenscale*(self.eigenscale if self.nonlinear_length_constraint else 1)
                rot_constr=numpy.dot(Vi,Vr if self.nonlinear_length_constraint else self.V0)
                Raug=numpy.hstack([R,eq_V_re,eq_V_im,norm_constr,rot_constr])                                
                col=lambda C:self.as_matrix_column(C)
                row=lambda R:self.as_matrix_row(R)                
                Jaug=scipy.sparse.block_array([
                    [J,None,None,col(dRdp),None],
                    [d_eq_V_re_dU,d_eq_V_re_dVr,d_eq_V_re_dVi,col(d_eq_V_re_dp),col(-(MI*Vr + MR*Vi))],
                    [d_eq_V_im_dU,d_eq_V_im_dVr,d_eq_V_im_dVi,col(d_eq_V_im_dp),col(MR*Vr-MI*Vi)],
                    [None,row(2*Vr if nl else self.V0),None,None,None],
                    [None,row(Vi) if nl else None,row(Vr if nl else self.V0),None,None]
                ]).tocsr()
                return Raug,Jaug        
        
        
class RealEigenbranchTracker(CustomBifurcationTracker):
    """
    Follows a real eigenbranch along a parameter
    """
    def __init__(self, problem,eigenvector:int,eigenscale:float=1,nonlinear_length_constraint:bool=False):
        super().__init__(problem)
        self.eigenscale=eigenscale
        self.lambda_Re0=numpy.real(problem.get_last_eigenvalues()[eigenvector])
        self.eigenvector=self.get_real_eigenvector_guess(eigenvector)
        self.nonlinear_length_constraint=nonlinear_length_constraint
        self.V0=self.eigenvector/numpy.linalg.norm(self.eigenvector)
    
    def define_augmented_dofs(self, dofs):
        dofs.add_vector(self.eigenvector*self.eigenscale)
        dofs.add_scalar(self.lambda_Re0)
        
    def get_residuals_and_jacobian(self,require_jacobian:bool,dparameter:Optional[str]=None)->Union[NPFloatArray,Tuple[NPFloatArray,DefaultMatrixType]]:               
        V,lam=self.get_augmented_dofs().split(startindex=1)
        lam=lam[0]
        assembly=self.start_multiassembly()
        if require_jacobian:
            assert dparameter is None, "dparameter not supported for require_jacobian=True"
            R,J,M,HJV,HMV=assembly.R().J().M().dJdU(V).dMdU(V).assemble()
        else:
            if dparameter is not None:                
                dRdp,dJdp,dMdp=assembly.dRdp(dparameter).dJdp(dparameter).dMdp(dparameter).assemble()
                return numpy.hstack([dRdp,lam*dMdp@V+dJdp@V,0])

            R,J,M=assembly.R().J().M().assemble()
            
        nl=self.nonlinear_length_constraint
        Raug=numpy.hstack([R,lam*M@V+J@V,numpy.dot(V,V if nl else self.V0)-self.eigenscale*(self.eigenscale if nl else 1)])
        if require_jacobian:
            col=lambda C:self.as_matrix_column(C)
            row=lambda R:self.as_matrix_row(R)
            Jaug=scipy.sparse.block_array(
                [[J,None,None],
                 [lam*HMV+HJV,lam*M+J,col(M@V)],
                 [None,row(2*V if nl else self.V0),None]]).tocsr()            
            return Raug,Jaug
        else:
            return Raug

    def actions_after_successful_newton_solve(self)->None:
        Vr,lam=self.get_augmented_dofs().split(startindex=1)
        self.store_eigenvector({lam[0]:numpy.array(Vr)})        



class ComplexEigenbranchTracker(CustomBifurcationTracker):
    """
    Follows a complex eigenbranch along a parameter
    """
    def __init__(self, problem,eigenvector:int,eigenscale:float=1,nonlinear_length_constraint:bool=False):
        super().__init__(problem)
        self.eigenscale=eigenscale
        self.lambda_Re0=numpy.real(problem.get_last_eigenvalues()[eigenvector])
        self.omega0=numpy.imag(problem.get_last_eigenvalues()[eigenvector])
        self.eigenvector=self.get_complex_eigenvector_guess(eigenvector)
        self.nonlinear_length_constraint=nonlinear_length_constraint
        self.V0=numpy.real(self.eigenvector)/numpy.linalg.norm(numpy.real(self.eigenvector))
    
    def define_augmented_dofs(self, dofs):
        dofs.add_vector(numpy.real(self.eigenvector)*self.eigenscale)
        dofs.add_vector(numpy.imag(self.eigenvector)*self.eigenscale)
        dofs.add_scalar(self.lambda_Re0)
        dofs.add_scalar(self.omega0)
        
    def get_residuals_and_jacobian(self,require_jacobian:bool,dparameter:Optional[str]=None)->Union[NPFloatArray,Tuple[NPFloatArray,DefaultMatrixType]]:               
        Vr,Vi,lam,omega=self.get_augmented_dofs().split(startindex=1)
        lam,omega=lam[0],omega[0]
        assembly=self.start_multiassembly()
        if require_jacobian:
            assert dparameter is None, "dparameter not supported for require_jacobian=True"
            R,J,M,HJVr,HJVi,HMVr,HMVi=assembly.R().J().M().dJdU(Vr).dJdU(Vi).dMdU(Vr).dMdU(Vi).assemble()
        else:
            if dparameter is not None:                
                dRdp,dJdp,dMdp=assembly.dRdp(dparameter).dJdp(dparameter).dMdp(dparameter).assemble()
                return numpy.hstack([dRdp,dMdp@(lam*Vr-omega*Vi)+dJdp@Vr,dMdp@(lam*Vi+omega*Vr)+dJdp@Vi, 0,0])
            
            R,J,M=assembly.R().J().M().assemble()
            
        nl=self.nonlinear_length_constraint
        Raug=numpy.hstack([R,M@(lam*Vr-omega*Vi)+J@Vr,M@(lam*Vi+omega*Vr)+J@Vi, numpy.dot(Vr,Vr if nl else self.V0)-self.eigenscale*(self.eigenscale if nl else 1),numpy.dot(Vi,Vr if nl else self.V0)])
        if require_jacobian:
            col=lambda C:self.as_matrix_column(C)
            row=lambda R:self.as_matrix_row(R)
            Jaug=scipy.sparse.block_array(
                [[J,None,None,None,None],
                 [lam*HMVr-omega*HMVi+HJVr,lam*M+J,-omega*M, col(M@Vr),col(-M@Vi)],
                 [lam*HMVi+omega*HMVr+HJVi,omega*M,lam*M+J, col(M@Vi),col(M@Vr)],
                 [None,row(2*Vr if nl else self.V0),None,None,None],
                 [None,row(Vi) if nl else None,row(Vr if nl else self.V0),None,None]]).tocsr()            
            return Raug,Jaug
        else:
            return Raug

    def actions_after_successful_newton_solve(self)->None:
        Vr,Vi,lam,omg=self.get_augmented_dofs().split(startindex=1)
        self.store_eigenvector({lam[0]+1j*omg[0]:numpy.array(Vr)+numpy.array(Vi)*1j})
        


class NormalModeEigenbranchTracker(_NormalModeBifurcationTrackerBase):
    def __init__(self, problem,eigenvector:int,azimuthal_m:Optional[int]=None,cartesian_k:ExpressionNumOrNone=None,eigenscale:float=1,nonlinear_length_constraint:bool=False):
        super().__init__(problem,eigenvector,azimuthal_m,cartesian_k,eigenscale,nonlinear_length_constraint)
        self.parameter=None # No parameter means essentially take the real part as adjustable parameter
                
    def get_residuals_and_jacobian(self,require_jacobian:bool,dparameter:Optional[str]=None)->Union[NPFloatArray,Tuple[NPFloatArray,DefaultMatrixType]]:
        nl=self.nonlinear_length_constraint
        if not self.has_imag:
            Vr,lamb=self.get_augmented_dofs().split(startindex=1)
            lamb=lamb[0]
            if not require_jacobian:
                if dparameter is not None:
                    dRdp,dJRdp,dMRdp=self.start_multiassembly().dRdp(dparameter).dJdp(dparameter,self.real_contribution).dMdp(dparameter,self.real_contribution).assemble()
                    dJRdp,dMRdp=self.patch_matrices(eigen=True,J=dJRdp,M=dMRdp)
                    dRdp,=self.patch_residuals(eigen=False,R=[dRdp])
                    return numpy.hstack([dRdp,lamb*dMRdp@Vr+dJRdp@Vr,0.0])                                
                else:
                    R,JR,MR=self.start_multiassembly().R().J(self.real_contribution).M(self.real_contribution).assemble()
                    JR,MR=self.patch_matrices(eigen=True,J=JR,M=MR)
                    return numpy.hstack([R,lamb*MR@Vr+JR@Vr,numpy.dot(Vr,Vr if self.nonlinear_length_constraint else self.V0)-self.eigenscale*(self.eigenscale if self.nonlinear_length_constraint else 1)])                                
            else:
                assert dparameter is None, "dparameter not supported for require_jacobian=True"
                R,J,JR,MR,HJVr,HMVr=self.start_multiassembly().R().J().J(self.real_contribution).M(self.real_contribution).dJdU(Vr).dMdU(Vr).assemble()
                R,=self.patch_residuals(eigen=False,R=[R])
                J,=self.patch_matrices(eigen=False,J=J)
                JR,MR=self.patch_matrices(eigen=True,J=JR,M=MR)
                col=lambda C:self.as_matrix_column(C)
                row=lambda R:self.as_matrix_row(R)                
                Raug=numpy.hstack([R,lamb*MR@Vr+JR@Vr,numpy.dot(Vr,Vr if self.nonlinear_length_constraint else self.V0)-self.eigenscale*(self.eigenscale if self.nonlinear_length_constraint else 1)])                                
                Jaug=scipy.sparse.block_array(
                    [[J,None,None],
                     [lamb*HMVr+HJVr,lamb*MR+JR,col(MR@Vr)],
                     [None,row(2*Vr if nl else self.V0),None]]).tocsr()
                return Raug,Jaug
        else:
            Vr,Vi,lamb,omega=self.get_augmented_dofs().split(startindex=1)
            lamb,omega=lamb[0],omega[0]
            if not require_jacobian:
                if dparameter is not None:
                    assm=self.start_multiassembly().dRdp(dparameter).dJdp(dparameter,self.real_contribution).dJdp(dparameter,self.imag_contribution)
                    assm.dMdp(dparameter,self.real_contribution).dMdp(dparameter,self.imag_contribution)
                    dRdp,=self.patch_residuals(eigen=False,R=[dRdp])
                    dRdp,dJRdp,dJIdp,dMRdp,dMIdp=assm.assemble()
                    dJRdp,dJIdp,dMRdp,dMIdp=self.patch_matrices(eigen=True,J=[dJRdp,dJIdp],M=[dMRdp,dMIdp])                    
                    d_eq_V_re_dp=-dJIdp*Vi + dJRdp*Vr + lamb*(-dMIdp*Vi + dMRdp*Vr) + omega*(-dMIdp*Vr - dMRdp*Vi)
                    d_eq_V_im_dp=dJIdp*Vr + dJRdp*Vi + lamb*(dMIdp*Vr + dMRdp*Vi) + omega*(-dMIdp*Vi + dMRdp*Vr)                    
                    return numpy.hstack([dRdp,d_eq_V_re_dp,d_eq_V_im_dp,0,0])                                
                else:
                    R,JR,JI,MR,MI=self.start_multiassembly().R().J(self.real_contribution).J(self.imag_contribution).M(self.real_contribution).M(self.imag_contribution).assemble()
                    R,=self.patch_residuals(eigen=False,R=[R])
                    JR,JI,MR,MI=self.patch_matrices(eigen=True,J=[JR,JI],M=[MR,MI])                    
                    eq_V_re=-JI*Vi + JR*Vr + lamb*(-MI*Vi + MR*Vr) + omega*(-MI*Vr - MR*Vi)
                    eq_V_im=JI*Vr + JR*Vi + lamb*(MI*Vr + MR*Vi) + omega*(-MI*Vi + MR*Vr)
                    norm_constr=numpy.dot(Vr,Vr if self.nonlinear_length_constraint else self.V0)-self.eigenscale*(self.eigenscale if self.nonlinear_length_constraint else 1)
                    rot_constr=numpy.dot(Vi,Vr if self.nonlinear_length_constraint else self.V0)                    
                    return numpy.hstack([R,eq_V_re,eq_V_im,norm_constr,rot_constr])                                
            else:                
                #import time
                #start=time.time()
                assm=self.start_multiassembly().R().J().J(self.real_contribution).J(self.imag_contribution).M(self.real_contribution).M(self.imag_contribution)
                assm.dJdU(Vr,self.real_contribution).dJdU(Vi,self.real_contribution).dJdU(Vr,self.imag_contribution).dJdU(Vi,self.imag_contribution)
                assm.dMdU(Vr,self.real_contribution).dMdU(Vi,self.real_contribution).dMdU(Vr,self.imag_contribution).dMdU(Vi,self.imag_contribution)
                R,J,JR,JI,MR,MI, HJRVR,HJRVI,HJIVR,HJIVI, HMRVR,HMRVI,HMIVR,HMIVI=assm.assemble()
                #end=time.time()
                #print("TIME TO ASSEMBLE",end-start)
                J,=self.patch_matrices(eigen=False,J=J)
                R,=self.patch_residuals(eigen=False,R=[R])
                JR,JI,MR,MI=self.patch_matrices(eigen=True,J=[JR,JI],M=[MR,MI])                    
                eq_V_re=JR@Vr -JI@Vi + lamb*(MR@Vr-MI@Vi) - omega*(MI@Vr + MR@Vi)
                d_eq_V_re_dU=-HJIVI + HJRVR + lamb*(HMRVR-HMIVI) - omega*(HMIVR + HMRVI)
                d_eq_V_re_dVr=JR + lamb*MR - omega*MI
                d_eq_V_re_dVi=-JI - lamb*MI - omega*MR
                eq_V_im=JI@Vr + JR@Vi + lamb*(MI@Vr + MR@Vi) + omega*(MR@Vr-MI@Vi)
                d_eq_V_im_dU=HJIVR + HJRVI + lamb*(HMIVR + HMRVI) + omega*(HMRVR-HMIVI)
                d_eq_V_im_dVr=JI + lamb*MI + omega*MR
                d_eq_V_im_dVi=JR + lamb*MR - omega*MI
                norm_constr=numpy.dot(Vr,Vr if self.nonlinear_length_constraint else self.V0)-self.eigenscale*(self.eigenscale if self.nonlinear_length_constraint else 1)
                rot_constr=numpy.dot(Vi,Vr if self.nonlinear_length_constraint else self.V0)
                Raug=numpy.hstack([R,eq_V_re,eq_V_im,norm_constr,rot_constr])                                
                col=lambda C:self.as_matrix_column(C)
                row=lambda R:self.as_matrix_row(R)                
                Jaug=scipy.sparse.block_array([
                    [J,None,None,None,None],
                    [d_eq_V_re_dU,d_eq_V_re_dVr,d_eq_V_re_dVi,col(-MI*Vi + MR*Vr),col(-MI*Vr - MR*Vi)],
                    [d_eq_V_im_dU,d_eq_V_im_dVr,d_eq_V_im_dVi,col(MI*Vr + MR*Vi),col(-MI*Vi + MR*Vr)],
                    [None,row(2*Vr if nl else self.V0),None,None,None],
                    [None,row(Vi) if nl else None,row(Vr if nl else self.V0),None,None]
                ]).tocsr()
                return Raug,Jaug
    




def EigenbranchTracker(problem:Problem,eigenvector:int=0,eigenscale:float=1,nonlinear_length_constraint:bool=False, complex_threshold:float=1e-8):
    # method to get the right eigenbranch tracker class depending on the type of the eigenvalue
    if eigenvector<0 or eigenvector>=len(problem.get_last_eigenvalues()):
        raise RuntimeError("Eigenvalue index out of range")
    normal_mode=False
    
    if problem.get_last_eigenmodes_k() is not None and eigenvector<len(problem.get_last_eigenmodes_k()):
        normal_mode=True 
    if problem.get_last_eigenmodes_m() is not None and eigenvector<len(problem.get_last_eigenmodes_m()):
        normal_mode=True
    if normal_mode:
        return NormalModeEigenbranchTracker(problem,eigenvector,eigenscale=eigenscale,nonlinear_length_constraint=nonlinear_length_constraint)
        #raise RuntimeError("Normal modes are not supported yet")
    if numpy.abs(numpy.imag(problem.get_last_eigenvalues()[eigenvector]))<complex_threshold:        
        return RealEigenbranchTracker(problem,eigenvector,eigenscale=eigenscale,nonlinear_length_constraint=nonlinear_length_constraint)
    else:
        return ComplexEigenbranchTracker(problem,eigenvector,eigenscale=eigenscale,nonlinear_length_constraint=nonlinear_length_constraint)
    pass




# Just a little helper to
class ResidualJacobianParameterDerivativeHandler(AugmentedAssemblyHandler):
    def define_augmented_dofs(self, dofs):
        pass
    
    def get_residuals_and_jacobian(self,require_jacobian:bool,dparameter:Optional[str]=None)->Union[NPFloatArray,Tuple[NPFloatArray,DefaultMatrixType]]:               
        if dparameter is None:
            raise ValueError("No parameter specified")
        assm=self.start_multiassembly()
        if require_jacobian is False:
            dRdp=assm.dRdp(dparameter).assemble()
            return numpy.array(dRdp)                
        else:
            dRdp,dJdp=assm.dRdp(dparameter).dJdp(dparameter).assemble()
            return numpy.array(dRdp),dJdp
        

# Just a little helper to return arbitrary stuff
class PerformCustomMultiAssembly(AugmentedAssemblyHandler):
    def __init__(self,problem:Problem,request:Callable[[MultiAssembleRequest],Any]):
        super().__init__()
        self.request=request
        self.problem=problem
        self.problem.set_custom_assembler(self)
        self.res=self.get_residuals_and_jacobian(require_jacobian=False)
        self.problem.set_custom_assembler(None)
        
    def result(self):
        return self.res
    
    def define_augmented_dofs(self, dofs):
        pass
    
    def get_residuals_and_jacobian(self,require_jacobian:bool,dparameter:Optional[str]=None)->Union[NPFloatArray,Tuple[NPFloatArray,DefaultMatrixType]]:                       
        assm=self.start_multiassembly()
        self.request(assm)
        res=assm.assemble()        
        for i,r in enumerate(res):
            if isinstance(r,list):
                res[i]=numpy.array(r)
        return res
        



class DeflationAssemblyHandler(AugmentedAssemblyHandler):
    # Deflation as described in https://arxiv.org/pdf/1410.5620
    def __init__(self,alpha=0.1,p=2):
        super().__init__()
        self.alpha=alpha
        self.p=p
        self.shift_mode:Literal["single","each","scaled"]="each"
        if not isinstance(self.p,int):
            raise ValueError("p must be an integer")
        if self.p<1:
            raise ValueError("p must be at least 1")
        self.Ws=[]
        
    def define_augmented_dofs(self, dofs):
        # No augmentation of dofs
        pass
        
    def add_known_solution(self,W):
        U,=self.get_augmented_dofs().split(startindex=0)
        if len(U)!=len(W):
            raise ValueError("Length of known solution and augmented dof do not match")
        self.Ws.append(W)
        
    def clear_known_solutions(self):
        self.Ws=[]
        
    
    def _get_alpha(self):
        if len(self.Ws)==0 or self.shift_mode=="single" or self.shift_mode=="each":
            return self.alpha
        else:
            return self.alpha**(1/len(self.Ws))
    
    def _get_eta_single(self,U,W):
        # Return the inverse of the diagonal of the deflation matrix (just a constant scalar) with respect to a known solution W
        #norm=numpy.sum((U-W)**self.p)
        norm=numpy.linalg.norm(U-W)**self.p
        if self.shift_mode=="single":
            return norm
        else:
            return 1/(1/norm+self._get_alpha())

    def _get_eta(self,U):
        # Return the inverse of the diagonal of the deflation matrix (just a constant scalar) with respect to a all known solutions
        res=1
        for W in self.Ws:
            res*=self._get_eta_single(U,W)
        if self.shift_mode=="single":
            if len(self.Ws)==0:
                return 1
            res=1/(1/res+self._get_alpha())
            return res
        else:
            return res        
    
    def _get_eta_prime_single(self,U,W):
        # Return derivative of eta_single(W) with respect to U   
        
        
        if self.shift_mode=="single":            
            raise RuntimeError("Implement single mode")
            #dnorm=self.p*n**(self.p-1)        
            return dnorm
        else:            
            n=numpy.linalg.norm(U-W)     
            return self.p*n**(self.p-2)/(self._get_alpha()*n**self.p+1)**2 *(U-W)
            #return dnorm/(self._get_alpha()*norm**2+1)

    def _get_eta_prime(self,U):    
        # Return derivative of eta([W1,W2,...]) with respect to U
        
        if self.shift_mode=="single":
            if len(self.Ws)==0:
                return 0
            sum_deta=0
            sum_eta=0
            for W in self.Ws:
                sum_eta+=self._get_eta_single(U,W)                
                sum_deta+=self._get_eta_prime_single(U,W)
            return sum_deta/(self._get_alpha()*sum_eta+1)**2
        else:
            factor=1
            sum=0
            for W in self.Ws:
                eta_s=self._get_eta_single(U,W)
                factor*=eta_s
                sum+=self._get_eta_prime_single(U,W)/eta_s
            return factor*sum
    
        
    def get_residuals_and_jacobian(self, require_jacobian, dparameter = None):
        assm=self.start_multiassembly()
        U,=self.get_augmented_dofs().split(startindex=0)
        eta=self._get_eta(U)
        if require_jacobian:
            if dparameter is not None:
                raise NotImplementedError("dparameter not implemented for jacobian")
            R,J=assm.R().J().assemble()            
            return numpy.array(R)/eta,J # NOTE: J is not deflated, it requires the special solving method
        else:
            if dparameter is not None:
                raise NotImplementedError("dparameter not implemented for residuals")
            R,=assm.R().assemble()            
            return numpy.array(R)/eta
        
    def has_custom_solve_routine(self):
        return True
        
    def custom_solve_routine(self, solve_Jx_b:Callable[[NPFloatArray],NPFloatArray], b:NPFloatArray) -> NPFloatArray:
        if len(self.Ws)==0:
            return solve_Jx_b(b)
        U,=self.get_augmented_dofs().split(startindex=0)
        eta=self._get_eta(U)
        d=self._get_eta_prime(U)
        f=-b/eta # Since R=R_nondeflated/eta, f=-R_nondeflated/eta**2
        fsol=solve_Jx_b(f)
        bsol=solve_Jx_b(b)
        fdbsol=f*numpy.dot(d,bsol)
        numer=solve_Jx_b(fdbsol)
        denom=1+eta*numpy.dot(d,fsol)                
        return eta*bsol-eta**2*numer/denom        

         
class NormalFormCalculator:
      def __init__(self,problem:Problem):
            self.problem=problem
            self.fd_eps=1e-7
                  
      def d2f(self,direct):                    
        res_hess=-numpy.array(self.problem.get_second_order_directional_derivative(direct))                
        return res_hess
  
      def d3f(self,direct,directfd=None):
            if directfd is None:
                directfd=direct
            u=self.problem.get_current_dofs()[0]
            direct_scale=1
            self.problem.set_current_dofs(u+self.fd_eps*directfd*direct_scale)
            res_hessp=-numpy.array(self.problem.get_second_order_directional_derivative(direct))
            self.problem.set_current_dofs(u-self.fd_eps*directfd*direct_scale)
            res_hessm=-numpy.array(self.problem.get_second_order_directional_derivative(direct))
            self.problem.set_current_dofs(u)        
            res_hess=0.5*(res_hessp-res_hessm)/(self.fd_eps*direct_scale)
            return res_hess
      
      

      def la_solve(self,A,rhs):
        #res=numpy.linalg.solve(A.toarray(),rhs) # TODO: Improve here        
        res= scipy.sparse.linalg.spsolve(A,rhs)# TODO: Improve here to use e.g. Pardiso (however, requires complex support)                   
        if numpy.isnan(numpy.sum(res)):
            print("Matrix rank warning. Going for a least squares solution")
            #res= numpy.linalg.lstsq(A.toarray(),rhs,rcond=None)[0]
            res=scipy.sparse.linalg.lsqr(A,rhs)[0]
        return res
            
      def get_left_eigenvector(self,lamb):            
            n, M_nzz, M_nr, M_val, M_ci, M_rs, J_nzz, J_nr, J_val, J_ci, J_rs = self.problem.assemble_eigenproblem_matrices(0) #type:ignore
            M=csr_matrix((M_val, M_ci, M_rs), shape=(n, n))	#TODO: Is csr or csc?
            A=csr_matrix((-J_val, J_ci, J_rs), shape=(n, n))
            AT=A.transpose().tocsr()
            MT=M.transpose().tocsr()
            if not self.problem.get_eigen_solver().supports_target():
                if numpy.abs(numpy.imag(lamb))>1e-8:
                    raise RuntimeError("Eigenvalue is not real. Complex left eigenvector must be implemented for a solver without target")
                evals,evects,_,_=self.problem.get_eigen_solver().solve(2,shift=1e-7,custom_J_and_M=(AT,MT))
            else:    
                evals,evects,_,_=self.problem.get_eigen_solver().solve(2,shift=1e-7,target=lamb,custom_J_and_M=(AT,MT))
            closest=numpy.argmin(numpy.abs(evals-lamb))
            return evects[closest],evals[closest]
      
      def get_normal_form(self,param:Optional[str]=None,eigenindex:int=0):
        if param is None:
            if self.problem._bifurcation_tracking_parameter_name is not None and self.problem._bifurcation_tracking_parameter_name!="":
                param=self.problem._bifurcation_tracking_parameter_name
            else:
                raise RuntimeError("Pass a parameter or use this with solved bifurcation tracking active")
        if self.problem.get_last_eigenvalues() is None or eigenindex>=len(self.problem.get_last_eigenvalues()):
            raise RuntimeError("Eigenpair at index "+str(eigenindex)+" not calculated!")
        lambd=self.problem.get_last_eigenvalues()[eigenindex]
        if numpy.abs(numpy.imag(lambd))>1e-8:
            return self.get_normal_form_hopf(param=param,eigenindex=eigenindex)
        else:
            return self.get_normal_form1d(param=param,eigenindex=eigenindex)
        
      def get_normal_form_hopf(self,param:Optional[str]=None,eigenindex:int=0):
            # Translated from Julia language code BifurcationKitDocs.jl (https://bifurcationkit.github.io/BifurcationKitDocs.jl)
            #raise RuntimeError("Hopf calculation does not really work without considering the mass matrix")
            # Generalized by a mass matrix
            if param is None:
                  if self.problem._bifurcation_tracking_parameter_name is not None and self.problem._bifurcation_tracking_parameter_name!="":
                        param=self.problem._bifurcation_tracking_parameter_name
                  else:
                        raise RuntimeError("Pass a parameter or use this with solved bifurcation tracking active")
            if self.problem.get_last_eigenvalues() is None or eigenindex>=len(self.problem.get_last_eigenvalues()):
                  raise RuntimeError("Eigenpair at index "+str(eigenindex)+" not calculated!")
            lambd=self.problem.get_last_eigenvalues()[eigenindex]
            # TODO: Check lambda small Re and nonvanishing imag
            omega=numpy.imag(lambd)
            
            
            zeta=self.problem.get_last_eigenvectors()[eigenindex]
            # TODO: Scale zeta reasonably
            zeta/=numpy.linalg.norm(zeta)
            czeta=numpy.conj(zeta)            
            self.problem.deactivate_bifurcation_tracking()
            self.problem.timestepper.make_steady()
            
            n, M_nzz, M_nr, M_val, M_ci, M_rs, J_nzz, J_nr, J_val, J_ci, J_rs = self.problem.assemble_eigenproblem_matrices(0) #type:ignore
            M=csr_matrix((M_val, M_ci, M_rs), shape=(n, n)).copy()	#TODO: Is csr or csc?
            
            
            zeta_star,lambd_star=self.get_left_eigenvector(numpy.conj(lambd))
            zeta_star /= numpy.vdot(M@zeta,zeta_star )                        
                        
            
            
            dRdp,J,dJdp,HzetaR,HzetaI=PerformCustomMultiAssembly(self.problem,lambda a : a.dRdp(param).J().dJdp(param).dJdU(numpy.real(zeta)).dJdU(numpy.imag(zeta))).result()
            Hzeta=(HzetaR+1j*HzetaI)/2
            R01=-dRdp
            L=-J
            psi001=self.la_solve(L,-R01) # TODO: This must be checked
            av=-dJdp@zeta
            av += 2 * Hzeta@ psi001 # TODO: This must be checked
            a = numpy.vdot(av, zeta_star)
            
            
            R20 = Hzeta@zeta
            psi200 = self.la_solve((2j*omega*M-L), R20)
            R20 =  Hzeta@czeta
            psi110 = self.la_solve(L, -R20)        
            
            # Third order term is a mess
            u=self.problem.get_current_dofs()[0]
            delt=1e-8
            
            def R3(dx1,dx2,dx3):
                self.problem.set_current_dofs(u+delt*dx1)
                dJp,=PerformCustomMultiAssembly(self.problem,lambda a : a.dJdU(dx3)).result()
                self.problem.set_current_dofs(u-delt*dx1)                
                dJm,=PerformCustomMultiAssembly(self.problem,lambda a : a.dJdU(dx3)).result()
                self.problem.set_current_dofs(u)
                return -((dJp-dJm)@dx2)/(2*delt)
            
            def third_order(R3,dx1, dx2, dx3): # x2=x1 assumed here
                dx1r = numpy.real(dx1);  dx2r=numpy.real(dx2); dx3r = numpy.real(dx3)
                dx1i = numpy.imag(dx1);  dx2i=numpy.imag(dx2); dx3i = numpy.imag(dx3)
                #outr =  R3(dx1r, dx2r, dx3r) - R3(dx1r, dx2i, dx3i) -R3(dx1i, dx2r, dx3i) - R3(dx1i, dx2i, dx3r)
                #outi =  R3(dx1r, dx2r, dx3i) + R3(dx1r, dx2i, dx3r) +R3(dx1i, dx2r, dx3r) - R3(dx1i, dx2i, dx3i)
                
                outr =  R3(dx1r, dx2r, dx3r) - R3(dx1r, dx2i, dx3i) - R3(dx1i, dx2r, dx3i) - R3(dx1i, dx2i, dx3r)
                outi =  R3(dx1r, dx2r, dx3i) + R3(dx1r, dx2i, dx3r) +R3(dx1i, dx2r, dx3r) - R3(dx1i, dx2i, dx3i)
                return (outr+1j*outi)/12

            
            
            #bv = 2 * Hzeta@psi110 + 2 * numpy.conj(Hzeta)@psi200 + 3 * third_order(lambda d1,d2,d3:R3(d1,d2,d3), zeta,zeta,czeta)
            # TODO: Here is still something not perfect in the quadratic terms...
            bv =  Hzeta@psi110 +  2*numpy.conj(Hzeta)@psi200 + 3 * third_order(lambda d1,d2,d3:R3(d1,d2,d3), zeta,zeta,czeta)
            print("HZETA1",Hzeta@psi110)
            print("HZERA2",numpy.conj(Hzeta)@psi200)
            
            
            
            b = numpy.vdot(bv, zeta_star)            
            #b*=2
            
            if omega>0:
                a=numpy.conj(a)
                b=numpy.conj(b)
            else:
                omega=-omega
                a=a
                b=b
                zeta=numpy.conj(zeta)
            
            # Nontrivial solution: A*exp(I*omega_l*t) where
            #   A**2*b + a*dp - I*omega + I*omega0=0
            # A=sqrt(-real(a)*dp/real(b))
            # omega_l=omega0+ imag(a)*dp - real(a)*imag(b)*dp/real(b) 
            
            
            print("omega",omega)
            print("a",a)
            print("b",b)
            res={}
            res["type"]="hopf"
            psign=1 if numpy.real(a)/numpy.real(b)<0 else -1
            res["psign"]=psign
            res["zeta"]=zeta
            res["param_predictor"]=lambda dp : psign*abs(dp)
            res["omega_predictor"]=lambda dp : omega+ numpy.imag(a)*dp - numpy.real(a)*numpy.imag(b)*dp/numpy.real(b) 
            res["perturbation_predictor"]=lambda dp,omegat: u+numpy.sqrt(numpy.abs(numpy.real(a)*dp/(2*numpy.real(b))))*numpy.real(zeta*numpy.exp(1j*omegat)+czeta*numpy.exp(-1j*omegat))
            return res
            
            
            
            
      def get_normal_form1d(self,param:Optional[str]=None,eigenindex:int=0):
            # Compute a normal form based on Golubitsky, Martin, David G Schaeffer, and Ian Stewart. Singularities and Groups in Bifurcation Theory. New York: Springer-Verlag, 1985, VI.1.d page 295.
            # Translated from Julia language code BifurcationKitDocs.jl (https://bifurcationkit.github.io/BifurcationKitDocs.jl)
            if param is None:
                  if self.problem._bifurcation_tracking_parameter_name is not None and self.problem._bifurcation_tracking_parameter_name!="":
                        param=self.problem._bifurcation_tracking_parameter_name
                  else:
                        raise RuntimeError("Pass a parameter or use this with solved bifurcation tracking active")
            if self.problem.get_last_eigenvalues() is None or eigenindex>=len(self.problem.get_last_eigenvalues()):
                  raise RuntimeError("Eigenpair at index "+str(eigenindex)+" not calculated!")
            
            lambd=self.problem.get_last_eigenvalues()[eigenindex]
            # TODO: Check lambda small Re and no imag
            lambd=numpy.real(lambd)
            zeta=self.problem.get_last_eigenvectors()[eigenindex]
            zeta=numpy.real(zeta)
            # TODO: Scale zeta reasonably
            zeta/=numpy.linalg.norm(zeta)
            self.problem.deactivate_bifurcation_tracking()
            self.problem.timestepper.make_steady()
            
            zeta_star,lambd_star=self.get_left_eigenvector(lambd)
            zeta_star = numpy.real(zeta_star)
            lambd_star = numpy.real(lambd_star)
            if abs(numpy.dot(zeta, zeta_star)) < 1e-10:
                  raise RuntimeError("The left and right eigenvectors are orthogonal, which should not be")
            zeta_star /= numpy.dot(zeta, zeta_star)
            dRdp,J,dJdp,Hzeta=PerformCustomMultiAssembly(self.problem,lambda a : a.dRdp(param).J().dJdp(param).dJdU(zeta)).result()
            R01=-dRdp
            a = numpy.dot(R01, zeta_star)
            L=-J
            
            E = lambda x: x - numpy.dot(x, zeta_star) *zeta
            #print("zeta",zeta)
            #print("zeta_star",zeta_star)
            #print("L",L)
            #print("R01",R01)
            ER01=E(R01)
            #print("ER01",ER01)
            psi01=E(self.la_solve(L,ER01))
            
            #print("psi01=",psi01,E(psi01))
            #psiJulia=numpy.array([-2.844267585714278, 0.1853775935202674, -0.004439531011646893])
            #print("RES PsiJulia",L@psiJulia-E(R01))
            #print("dot1",numpy.dot(psiJulia,zeta_star))
            #print("dot2",numpy.dot(psiJulia,zeta))
            #print("RES JUlia",psiJulia,E(psiJulia))
            #resPetsc=self.nullspace_la_solve(L,E(R01),zeta,zeta_star)
            #print("RES PsiPetsc",resPetsc,E(resPetsc))
            #exit()
            dF=lambda x: -dJdp@x
            R11=dF(zeta)
            
            
            b1 = numpy.dot(R11 + Hzeta@psi01, zeta_star)
            
            b2v = -Hzeta@zeta
            b2 = numpy.dot(b2v, zeta_star)
            
            wst = E(self.la_solve(L, E(b2v))) # Golub. Schaeffer Vol 1 page 33, eq 3.22    
            b3v = self.d3f(zeta) + 3 * Hzeta@wst
            b3 = numpy.dot(b3v, zeta_star)    
            res={"a":a,"b1":b1,"b2":b2,"b3":b3}
            print("a=",a)
            print("b1=",b1)
            print("b2=",b2)
            print("b3=",b3)
            tol_fold=0.001
            if abs(a) < tol_fold:
                if 100*abs(b2/2) < abs(b3/6):
                    print("Likely a pitchfork")
                    psign=1 if b1*b3<0 else -1
                    print("With sign",psign)
                    res["type"]="pitchfork"
                    res["psign"]=psign
                    res["zeta"]=zeta
                    res["param_predictor"]=lambda dp : psign*abs(dp)
                    res["perturbation_predictor"]=lambda dp: zeta*numpy.sqrt(abs(6*b1/b3*dp))
                    
                else:
                    print("Likely transcritical")
                    res["type"]="transcritical"
                    res["psign"]="arbitrary" # Can be chosen arbitrarily
                    res["param_predictor"]=lambda dp : dp
                    res["perturbation_predictor"]=lambda dp: -zeta*2*b1/b2*dp
            else:
                print("Likely fold")
                res["type"]="fold"
                res["psign"]=0 # Parameter may not change
                res["param_predictor"]=lambda dp : 0
                res["perturbation_predictor"]=lambda dp: zeta*dp
            return res
