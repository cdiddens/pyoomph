from __future__ import annotations
import scipy.sparse
import scipy.sparse.linalg
from .problem import Problem
from ..expressions import GlobalParameter,ExpressionNumOrNone
from ..typings import *
from .. import _pyoomph_core as _pyoomph
import numpy,scipy
from .assembly import CustomAssemblyBase
from ..solvers.generic import DefaultMatrixType

from scipy.sparse import csr_matrix      

def get_hopf_lyapunov_coefficient(problem:Problem,param:GlobalParameter | str,FD_delta:float=1e-5,FD_param_delta:float=1e-5,omega:float | None=None,q:NPComplexArray | None=None,omega_epsilon:float=1e-5,use_hopf_tracker_for_adjoint:bool=False,verbose:bool=True,residual_tolerance:float=1e-6,check_derivatives_by_fd:bool=False):
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

    eigensolve_kwargs:dict[str,Any]={}
        
    
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
    


    problem._require_non_distributed("Bifurcation tracking")
    n, M_nzz, M_nr, M_val, M_ci, M_rs, J_nzz, J_nr, J_val, J_ci, J_rs = problem.assemble_eigenproblem_matrices(0) #type:ignore
    # Compressed sparse ROW, per the binding's own docstring ("both matrices in local compressed
    # sparse row (CSR) format"), so (values, column_index, row_start) is the csr_matrix argument order
    # and A/M come out un-transposed. A=-J because J=dR/dU while the pencil below is A q = i*omega*M q.
    M=csr_matrix((M_val, M_ci, M_rs), shape=(n, n))
    A=csr_matrix((-J_val, J_ci, J_rs), shape=(n, n))
    AT=A.transpose().tocsr()
    MT=M.transpose().tocsr()
    
    if omega is None or q is None:
        if verbose:
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
    else:
        qR=numpy.real(q)
        qI=numpy.imag(q)
        omega0=omega
    qdenom=numpy.dot(qR,qR)+numpy.dot(qI,qI)
    qR/=numpy.sqrt(qdenom)
    qI/=numpy.sqrt(qdenom)
    
    if numpy.amax(numpy.abs(A@qR+omega0*M@qI))>1e-7 or numpy.amax(numpy.abs(-omega0*M@qR+A@qI))>1e-7:
        if verbose:
            print("Given q does not fulfill the eigenvector equation. Resolving it.")
        esolv_kwargs=eigensolve_kwargs.copy()
        if esolver.supports_target():
            esolv_kwargs["target"]=1j*omega0
        eval,evects,_,_=problem.get_eigen_solver().solve(1,custom_J_and_M=(A,M),**esolv_kwargs,shift=(1j+omega_epsilon)*omega0,v0=None if q is None else numpy.array(q),sort=False,quiet=False)
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
        # Try to rotate it
        tmp=HopfTracker(problem,param,eigenvector=qR+1j*qI,omega=omega0)
        q=tmp.eigenvector
        qR=numpy.real(q)
        qI=numpy.imag(q)
        if verbose:
            print("After rotation, dot product is",numpy.dot(qR,qI))
        if numpy.abs(numpy.dot(qR,qI))>1e-7:
            raise ValueError("qR and qI are not orthogonal, i.e. the dot product is not zero, but {}. This is likely an issue with the eigenvalue solver. Please check the eigenvalue solver settings.".format(numpy.dot(qR,qI)))
    q_resolved=qR+1j*qI # q (the parameter) might still be None from the caller/auto-solve above; qR/qI are always resolved by this point

    esolv_kwargs=eigensolve_kwargs.copy()
    # 'or' here used to mean that asking for the Hopf tracker selected the eigensolver instead.
    if esolver.supports_target() and not use_hopf_tracker_for_adjoint:
        esolv_kwargs["target"]=-1j*omega0
        evalT,evectT,_,_=problem.get_eigen_solver().solve(1,custom_J_and_M=(AT,MT),**esolv_kwargs,shift=-(1j+omega_epsilon)*omega0,v0=numpy.conjugate(q_resolved),sort=False,quiet=False)   # TODO: Is MT right here?                  #type:ignore
    else:
        problem.deactivate_bifurcation_tracking()        
        problem.set_custom_assembler(HopfTracker(problem,param.get_name(),numpy.conjugate(q_resolved),omega=-omega0,left_eigenvector=True,eigenscale=1))
        problem.solve()
        evalT=problem.get_last_eigenvalues()
        evectT=problem.get_last_eigenvectors()
        # evalT is an array; comparing the whole of it against a scalar raises. Only the mode we are
        # after matters, and we want the one at -i*omega0.
        if numpy.imag(evalT[0])*omega0>0:
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
        # <p,Mq> is what the whole projection rests on: z=<p,Mx> is only a coordinate if it is 1.
        # Near-zero means the adjoint the eigensolver returned does not pair with q -- usually the
        # wrong mode of the conjugate pair -- and dividing by it produces a confident wrong answer.
        raise ValueError("The adjoint eigenvector does not pair with the eigenvector: <p,Mq>={:.3e} is "
                         "essentially zero, so it cannot be normalised to 1. This is usually the "
                         "eigensolver returning the wrong member of the conjugate pair; check its "
                         "settings, or pass a better omega/q.".format(pnorm))
    pR=pR/pnorm
    pI=pI/pnorm
    
    # Every one of these must be zero for the projection to mean anything, and a wrong adjoint gives
    # a confidently wrong coefficient rather than a failure -- so they are checked, not just printed.
    # The abs() matters: amax() of a large NEGATIVE residual passes any positive threshold.
    checks={
        "A*qR + omega0*M*qI":   numpy.amax(numpy.abs(A@qR+omega0*M@qI)),
        "A*qI - omega0*M*qR":   numpy.amax(numpy.abs(A@qI-omega0*M@qR)),
        "AT*pR - omega0*MT*pI": numpy.amax(numpy.abs(AT@pR-omega0*MT@pI)),
        "AT*pI + omega0*MT*pR": numpy.amax(numpy.abs(AT@pI+omega0*MT@pR)),
        "<q,q> - 1":            numpy.abs(numpy.dot(qR,qR)+numpy.dot(qI,qI)-1),
        "<qR,qI>":              numpy.abs(numpy.dot(qR,qI)),
        "Re<p,Mq> - 1":         numpy.abs(numpy.dot(pR,MqR)+numpy.dot(pI,MqI)-1),
        # <p,Mq*>=0 is automatic for omega0!=0 (p and q* belong to different eigenvalues of the
        # pencil), so this one is a consistency check on the solver rather than a constraint.
        "Im<p,Mq>":             numpy.abs(numpy.dot(pR,MqI)-numpy.dot(pI,MqR)),
    }
    if verbose:
        print("Step 1: eigenvector/adjoint residuals (all should be zero)")
        for name,val in checks.items():
            print("   {:24s} {:.3e}".format(name,val))
    worst=max(checks.items(),key=lambda kv:kv[1])
    if worst[1]>residual_tolerance:
        print("WARNING: the Hopf eigenvector/adjoint pair is only satisfied to {:.3e} ({}); the first "
              "Lyapunov coefficient will be correspondingly inaccurate".format(worst[1],worst[0]))
        #exit()
        #print("THIS gives:")
        #print("qR",qR)
        #print("qI",qI)
        #print("pR",pR)
        #print("pI",pI)
    
    
    
    
    def d2f(direct):
        """B(v,v), from the analytic Hessian contraction."""
        res_hess=-numpy.array(problem.get_second_order_directional_derivative(direct))
        if check_derivatives_by_fd:
            # Second difference of the residual itself. Only worth running when a newly written
            # element's analytic Hessian is under suspicion -- it costs two extra assemblies per call
            # and is far less accurate than what it is checking.
            f0=nodalf(u)
            fp=nodalf(u+delt*direct)
            fm=nodalf(u-delt*direct)
            problem.set_current_dofs(u)
            res_fd=(fm-2*f0+fp)/(delt**2)
            print("d2f vs FD: difference",numpy.amax(numpy.absolute(res_hess-res_fd)),
                  "FD",numpy.amax(numpy.absolute(res_fd)),"Hessian",numpy.amax(numpy.absolute(res_hess)))
        return res_hess
        
    def d3f(direct):
        # C(v,v,v), as a central difference of the analytic Hessian contraction along the same
        # direction. There is no third derivative in the code generator, so this is the only route.
        problem.set_current_dofs(u+delt*direct)
        res_hessp=-numpy.array(problem.get_second_order_directional_derivative(direct))
        problem.set_current_dofs(u-delt*direct)
        res_hessm=-numpy.array(problem.get_second_order_directional_derivative(direct))
        problem.set_current_dofs(u)        
        res_hess=0.5*(res_hessp-res_hessm)/delt
        if check_derivatives_by_fd:
            fmm=nodalf(u-2*delt*direct)
            fm=nodalf(u-delt*direct)
            fp=nodalf(u+delt*direct)
            fpp=nodalf(u+2*delt*direct)
            problem.set_current_dofs(u)
            res_fd=(-0.5*fmm+fm-fp+0.5*fpp)/(delt**3)
            print("d3f vs FD: difference",numpy.amax(numpy.absolute(res_hess-res_fd)),
                  "FD",numpy.amax(numpy.absolute(res_fd)),"Hessian-FD",numpy.amax(numpy.absolute(res_hess)))
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

    # Step 3: the two normal-form solves.
    #   r  = A^-1 B(q,qb)        with B(q,qb) = a+b     (its imaginary part cancels by symmetry)
    #   sv = (2i*omega0*M - A)^-1 B(q,q)  with B(q,q) = a-b+2i*c      = h20
    # The mass matrix belongs in the h20 OPERATOR and nowhere else -- in particular not on either
    # right-hand side. pde2path multiplies both right-hand sides by M; that is what the two commented
    # lines below were, and the independent NormalFormCalculator.get_normal_form_hopf in this file
    # agrees with the form used here (its psi200/psi110 solves carry no M on the right either).
    #r=solve_mat(A,M*(a+b))
    #sv=solve_mat(-A+2j*M*omega0,M*(a-b+2j*c))
    r=solve_mat(A,a+b)
    sv=solve_mat(-A+2j*M*omega0,a-b+2j*c)
    sR=numpy.real(sv)
    sI=numpy.imag(sv)
    if verbose:
        # These used to check the pde2path form (an M@ on each right-hand side) while the lines above
        # solved this one, so they reported a large error for a correct solve and were ignored.
        print("Step 3: normal-form solve residuals (all should be zero)")
        print("   {:24s} {:.3e}   |r| ={:.3e}".format("A*r - B(q,qb)",
              numpy.amax(numpy.absolute(A@r-(a+b))),numpy.linalg.norm(r))) #type:ignore
        print("   {:24s} {:.3e}   |sR|={:.3e}".format("Re[(2i*w*M-A)s - B(q,q)]",
              numpy.amax(numpy.absolute(-A@sR-2*omega0*M@sI-(a-b))),numpy.linalg.norm(sR)))
        print("   {:24s} {:.3e}   |sI|={:.3e}".format("Im[(2i*w*M-A)s - B(q,q)]",
              numpy.amax(numpy.absolute(2*omega0*M@sR-A@sI-2*c)),numpy.linalg.norm(sI)))

    # step 4
    # sig = Re<p,B(q,r)>. With r real, B(q,r) = B(qR,r) + i*B(qI,r), and
    #   Re[(pR - i pI).(B(qR,r) + i B(qI,r))] = pR.B(qR,r) + pI.B(qI,r),
    # so the second term takes pI (Kuznetsov), not pR (pde2path). Same for d3/d4 below.
    sig1=0.25*numpy.dot(pR,d2f(qR+r)-d2f(qR-r))
    sig2=0.25*numpy.dot(pI,d2f(qI+r)-d2f(qI-r))
    sig=sig1+sig2
    if verbose:
        print("Step 4")
        print("sig1",sig1)
        print("sig2",sig2)
        print("sig",sig)
    
    # step 5
    # d0 = Re<p,B(qb,s)>. With qb = qR - i qI and s = sR + i sI,
    #   B(qb,s) = [B(qR,sR)+B(qI,sI)] + i[B(qR,sI)-B(qI,sR)],
    # so Re[(pR - i pI).that] = pR.B(qR,sR) + pR.B(qI,sI) + pI.B(qR,sI) - pI.B(qI,sR) = d1+d2+d3-d4.
    d1=0.25*numpy.dot(pR,d2f(qR+sR)-d2f(qR-sR))
    d2=0.25*numpy.dot(pR,d2f(qI+sI)-d2f(qI-sI))
    d3=0.25*numpy.dot(pI,d2f(qR+sI)-d2f(qR-sI))
    d4=0.25*numpy.dot(pI,d2f(qI+sR)-d2f(qI-sR))
    d0=d1+d2+d3-d4
    if verbose:
        print("Step 5")
        print("d1",d1)
        print("d2",d2)
        print("d3",d3)
        print("d4",d4)
        print("d0",d0)
    
    # Step 6: g0 = Re<p,C(q,q,qb)>. Expanding C(q,q,qb) = C(uuu)+C(uvv) + i[C(uuv)+C(vvv)] with
    # u=qR, v=qI gives pR.[C(uuu)+C(uvv)] + pI.[C(uuv)+C(vvv)], which is what the polarisation below
    # assembles out of the three same-direction contractions C(w,w,w).
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
  

    # d(Re lambda)/d(parameter), by continuing the eigenbranch rather than re-solving the
    # eigenproblem at a shifted parameter: q_resolved, not q -- q is still None whenever the caller
    # supplied neither omega nor an eigenvector and the auto-solve above produced them.
    old=param.value
    problem.activate_eigenbranch_tracking("complex",eigenvector=q_resolved,eigenvalue=1j*omega0)
    problem.solve()
    mu1=numpy.real(problem.get_last_eigenvalues()[0])
    problem.go_to_param({param.get_name():param.value+FD_param_delta})
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
        self._what:list[str]=[]
        self._contributions:list[str]=[]
        self._hessian_vectors:list[Any]=[]
        self._hessian_vector_indices:list[int]=[]
        self._parameters:list[str]=[] # by name: the C++ multiassembly takes strings
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
        
    def dRdp(self,parameter:str | GlobalParameter,contribution=""):
        self._what.append("dresiduals_dparameter")
        self._contributions.append(contribution)
        self._parameters.append(parameter if isinstance(parameter,str) else parameter.get_name())
        return self
    
    def dJdp(self,parameter:str | GlobalParameter,contribution=""):
        self._what.append("djacobian_dparameter")
        self._contributions.append(contribution)
        self._parameters.append(parameter if isinstance(parameter,str) else parameter.get_name())
        return self
        
    def dMdp(self,parameter:str | GlobalParameter,contribution=""):
        self._what.append("dmass_matrix_dparameter")
        self._contributions.append(contribution)
        self._parameters.append(parameter if isinstance(parameter,str) else parameter.get_name())
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
        self._augdof_spec=self.get_problem()._create_dof_augmentation()
        self.define_augmented_dofs(self.get_augmented_dofs())
        self.get_problem()._add_augmented_dofs(self._augdof_spec)
        print("Dofs after augmentation",self.get_problem().ndof())
        
        
    def finalize(self):
        self.get_problem()._reset_augmented_dof_vector_to_nonaugmented()
        
        
    def get_augmented_dofs(self)->DofAugmentationSpecifications:
        assert self._augdof_spec is not None, "initialize() must be called before get_augmented_dofs()"
        return self._augdof_spec
            
    def define_augmented_dofs(self,dofs:DofAugmentationSpecifications):
        raise NotImplementedError("define_augmented_dofs not implemented")
    
    def define_augmented_residuals(self,dofs:DofAugmentationSpecifications):
        raise NotImplementedError("define_augmented_residuals not implemented")
    
    def define_augmented_residuals_and_jacobian(self,dofs:DofAugmentationSpecifications):
        raise NotImplementedError("define_augmented_residuals_and_jacobian not implemented")
    
    def get_base_residuals_and_jacobian(self)->tuple[NPFloatArray,DefaultMatrixType]:
        old=self.get_problem().use_custom_residual_jacobian
        self.get_problem().use_custom_residual_jacobian=False
        R,J=self.get_problem().assemble_jacobian(with_residual=True)
        self.get_problem().use_custom_residual_jacobian=old
        return R,J
    
    def get_base_dresiduals_dparameter(self,parameter:str | GlobalParameter)->NPFloatArray:
        raise NotImplementedError("get_base_dresiduals_dparameter not implemented")
    
    def get_base_dresiduals_and_djacobian_dparameter(self,parameter:str | GlobalParameter)->tuple[NPFloatArray,DefaultMatrixType]:
        raise NotImplementedError("get_base_dresiduals_and_djacobian_dparameter not implemented")
    
    def get_base_hessian_vector_product(self,vector:NPFloatArray)->NPFloatArray:
        raise NotImplementedError("get_base_hessian_vector_product not implemented")

    def start_multiassembly(self):
        return MultiAssembleRequest(self.get_problem())
    
    def as_matrix_column(self,arr):
        if isinstance(arr,list):
            arr=numpy.array(arr)
        return scipy.sparse.csr_matrix(arr.reshape(-1,1))
    
    def as_matrix_row(self,arr):
        if isinstance(arr,list):
            arr=numpy.array(arr)
        return scipy.sparse.csr_matrix(arr.reshape(1,-1))
    
    def get_residuals_and_jacobian(self,require_jacobian:bool,dparameter:str | None=None)->NPFloatArray | tuple[NPFloatArray, DefaultMatrixType]: #type:ignore
        raise NotImplementedError("get_residuals_and_jacobian not implemented")

class CustomBifurcationTracker(AugmentedAssemblyHandler):
    """
    A generic class to write custom bifurcation trackers.
    """
    def __init__(self,problem:Problem):
        super().__init__()
        self.problem=problem
        
    def get_real_eigenvector_guess(self,eigenvector:NPAnyArray | int=0,normalize:bool=True)->NPAnyArray:
        """
        Get a real eigenvector guess. This can be either an index to the previously solved eigenvalues or an eigenvector.        

        Args:
            eigenvector: Index to a calculated eigenvector or the eigenvector itself
            normalize: Normalize the eigenvector to |V|=1

        Returns:
            The eigenvector as array
        """
        V:NPAnyArray
        if isinstance(eigenvector,int):
            if eigenvector>=len(self.get_problem().get_last_eigenvalues()):
                raise RuntimeError("Eigenvalue index out of range")
            V=numpy.real(self.get_problem().get_last_eigenvectors()[eigenvector])
        else:
            V=eigenvector
        if normalize:
            V/=numpy.linalg.norm(V)
        return V

    def get_complex_eigenvector_guess(self,eigenvector:NPAnyArray | int,normalize:bool=True)->NPAnyArray:
        """
        Get a complex eigenvector guess. This can be either an index to the previously solved eigenvalues or an eigenvector.        

        Args:
            eigenvector: Index to a calculated eigenvector or the eigenvector itself
            normalize: Normalize the eigenvector so that <Re(V),Im(V)>=0 and <Re(V),Re(V)>=1

        Returns:
            The eigenvector as array
        """
        V:NPAnyArray
        if isinstance(eigenvector,int):
            if eigenvector>=len(self.get_problem().get_last_eigenvalues()):
                raise RuntimeError("Eigenvalue index out of range")
            V=self.get_problem().get_last_eigenvectors()[eigenvector]
        else:
            V=eigenvector
        if normalize:
            GrGr=numpy.dot(numpy.real(V),numpy.real(V))
            GrGi=numpy.dot(numpy.real(V),numpy.imag(V))
            GiGi=numpy.dot(numpy.imag(V),numpy.imag(V))
            n_phi_samples,n_iter,best_phi,best_GrGr=30,15,0,GrGr
            for phi in numpy.linspace(0,2*numpy.pi,n_phi_samples):
                res=-GiGi*numpy.sin(phi)*numpy.cos(phi) - GrGi*numpy.sin(phi)*numpy.sin(phi) + GrGi*numpy.cos(phi)*numpy.cos(phi) + GrGr*numpy.sin(phi)*numpy.cos(phi)
                GrGr_new=numpy.dot(numpy.real(V)*numpy.cos(phi)+numpy.imag(V)*numpy.sin(phi),numpy.real(V)*numpy.cos(phi)+numpy.imag(V)*numpy.sin(phi))
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
            V=numpy.exp(1j*best_phi)*V

            if False:
                pR,pI=numpy.real(V),numpy.imag(V)
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
                V/=numpy.linalg.norm(numpy.real(V))
                #theta=scipy.optimize.minimize_scalar(optimize,bounds=(0,2*numpy.pi),method="bounded",options={"xatol":1e-15,"maxiter":100}).x
                theta=scipy.optimize.root_scalar(lambda t:optimize(t)[0],x0=besttheta).root
                V=(pR+pI*1j)*numpy.exp(1j*theta)
                V/=numpy.linalg.norm(numpy.real(V))

            V/=numpy.linalg.norm(numpy.real(V))

            #print("Eigenvector ReIm",numpy.dot(numpy.real(V),numpy.imag(V)))
            #print("Eigenvector ReRe",numpy.dot(numpy.real(V),numpy.real(V)))
            #print("Eigenvector ImIm",numpy.dot(numpy.imag(V),numpy.imag(V)))
            #exit()
        return V

    def store_eigenvector(self,eigenvects:dict[float | complex,NPAnyArray]):
        self.get_problem()._last_eigenvalues=numpy.array(list(eigenvects.keys()))
        self.get_problem()._last_eigenvectors=numpy.array([numpy.array(v) for v in eigenvects.values()])   
        self.get_problem()._last_eigenvalues_m=None
        self.get_problem()._last_eigenvalues_k=None          


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
    def __init__(self,problem:Problem,parameter,eigenvector:NPAnyArray | int=0,eigenscale:float=1,nonlinear_length_constraint:bool=False):
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

    def get_residuals_and_jacobian(self,require_jacobian:bool,dparameter:str | None=None)->NPFloatArray | tuple[NPFloatArray, DefaultMatrixType]: # type: ignore[override] # the base's overloads narrow the return by require_jacobian; every implementation here is the general one
        V,=self.get_augmented_dofs().split(startindex=1,endindex=2) # Get the eigenvector solution
        # Request the residuals and Jacobian of the non-augmented system
        assembly=self.start_multiassembly()
        dRdP=None
        dJdP=None
        HV=None
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
            assert dRdP is not None and dJdP is not None and HV is not None
            col=lambda C:self.as_matrix_column(C)
            row=lambda R:self.as_matrix_row(R)
            # Augmented Jacobian
            Jaug=scipy.sparse.block_array(
                [[J,None,col(dRdP)],
                 [HV,J,col(dJdP@V)],
                 [None,row(2*V if nl else self.V0),None]]).tocsr()
            return Raug,Jaug #type:ignore
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
        
    def get_residuals_and_jacobian(self,require_jacobian:bool,dparameter:str | None=None)->NPFloatArray | tuple[NPFloatArray, DefaultMatrixType]: # type: ignore[override] # the base's overloads narrow the return by require_jacobian; every implementation here is the general one
        U,V,p,eps=self.get_augmented_dofs().split(startindex=0) # Get the eigenvector solution and the slack variable
        eps=eps[0] # Get the scalar value of the slack variable (split dofs are all vectors)
        # Request the residuals and Jacobian of the non-augmented system
        assembly=self.start_multiassembly()
        dRdP=None
        dJdP=None
        HV=None
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
            assert dRdP is not None and dJdP is not None and HV is not None
            col=lambda C:self.as_matrix_column(C)
            row=lambda R:self.as_matrix_row(R)
            # Augmented Jacobian
            Jaug=scipy.sparse.block_array(
                [[J,None,col(dRdP),col(self.S)],
                 [HV,J,col(dJdP@V),None],
                 [None,row(2*V if nl else self.V0),None,None],
                 [row(self.S),None,None,None]]).tocsr()            
            return Raug,Jaug #type:ignore
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
    def __init__(self,problem:Problem,parameter,eigenvector:NPAnyArray | int=0,omega:float | None=None,eigenscale:float=1,nonlinear_length_constraint:bool=False,left_eigenvector:bool=False):
        super().__init__(problem)
        self.parameter=parameter
        self.eigenvector=self.get_complex_eigenvector_guess(eigenvector)
        if omega is None:
            if isinstance(eigenvector,int):
                self.omega=numpy.imag(self.get_problem().get_last_eigenvalues()[0])
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
        
    def get_residuals_and_jacobian(self,require_jacobian:bool,dparameter:str | None=None)->NPFloatArray | tuple[NPFloatArray, DefaultMatrixType]: # type: ignore[override] # the base's overloads narrow the return by require_jacobian; every implementation here is the general one
        Vr,Vi,p,omega=self.get_augmented_dofs().split(startindex=1) # Get all the augmented dofs
        omega=omega[0] # Get the scalar value of the frequency variable (split dofs are all vectors)        
        assembly=self.start_multiassembly()
        dRdP=None
        dJdP=None
        dMdP=None
        HVr=None
        HVi=None
        dMdUVr=None
        dMdUVi=None
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
            assert dRdP is not None and dJdP is not None and dMdP is not None and HVr is not None and HVi is not None and dMdUVr is not None and dMdUVi is not None
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
            return Raug,Jaug #type:ignore
        else:
            return Raug
        
    def actions_after_successful_newton_solve(self)->None:
        Vr,Vi,p,omega=self.get_augmented_dofs().split(startindex=1)
        self.store_eigenvector({(1j*omega[0]):(numpy.array(Vr)+numpy.array(Vi)*1j)})        
        

class _NormalModeBifurcationTrackerBase(CustomBifurcationTracker):
    parameter:str | None # Set by concrete subclasses: a parameter name, or None for eigenbranch tracking

    # azimuthal_m may be a non-integer: nothing on the tracking path needs it to be one (only an
    # eigensolve does), and CriticalWavenumberTracker makes it an unknown of the augmented system.
    # It still selects the axis conditions, which are a step function of it - see that class.
    def __init__(self, problem:Problem,eigenvector:int=0,azimuthal_m:float | None=None,cartesian_k:ExpressionNumOrNone=None,eigenscale:float=1,nonlinear_length_constraint:bool=False):
        super().__init__(problem)
        self.eigenscale=eigenscale
        self.nonlinear_length_constraint=nonlinear_length_constraint
        prob=self.get_problem()
        if prob._azimuthal_mode_param_m is not None:
            self.azimuthal=True
            if cartesian_k is not None:
                raise RuntimeError("Cannot supply a Cartesian wave number k for azimuthal mode bifurcation tracking")
            if azimuthal_m is not None:
                self.azimuthal_m=azimuthal_m
            else:
                last_modes_m=prob.get_last_eigenmodes_m()
                assert last_modes_m is not None
                self.azimuthal_m=int(last_modes_m[eigenvector])
            prob._azimuthal_mode_param_m.value=self.azimuthal_m
            assert prob._azimuthal_stability is not None
            self.real_contribution=prob._azimuthal_stability.real_contribution_name
            self.imag_contribution=prob._azimuthal_stability.imag_contribution_name
        elif prob._normal_mode_param_k is not None:
            self.azimuthal=False
            if azimuthal_m is not None:
                raise RuntimeError("Cannot supply an azimuthal mode m for Cartesian normal mode bifurcation tracking")
            if cartesian_k is not None:
                self.cartesian_k=cartesian_k
            else:
                last_modes_k=prob.get_last_eigenmodes_k()
                assert last_modes_k is not None
                self.cartesian_k=last_modes_k[eigenvector]
            prob._normal_mode_param_k.value=self.cartesian_k #type:ignore
            assert prob._cartesian_normal_mode_stability is not None
            self.real_contribution=prob._cartesian_normal_mode_stability.real_contribution_name
            self.imag_contribution=prob._cartesian_normal_mode_stability.imag_contribution_name
        else:
            raise RuntimeError("Normal mode bifurcation tracking requires either azimuthal mode or Cartesian normal mode by calling setup_for_stability_analysis with the right kwargs first")

        self.eigenvector=self.get_complex_eigenvector_guess(eigenvector)
        #print(numpy.dot(numpy.imag(self.eigenvector),numpy.real(self.eigenvector)))
        #print(numpy.dot(numpy.imag(self.eigenvector),numpy.imag(self.eigenvector)))
        #print(numpy.dot(numpy.real(self.eigenvector),numpy.real(self.eigenvector)))



        self.has_imag=numpy.dot(numpy.imag(self.eigenvector),numpy.imag(self.eigenvector))>1e-15 # TODO: Make it adjustable
        if not self.has_imag:
            self.has_imag=prob._set_solved_residual(self.imag_contribution,False,False)
            prob._set_solved_residual("",False,False)
            if self.has_imag:
                print("Strange, eigenvector is real, but it has imaginary jacobian contribution")
                #raise RuntimeError("Strange, eigenvector is real, but it has imaginary jacobian contribution")
        self.lambda0=numpy.real(prob.get_last_eigenvalues()[eigenvector])
        if self.has_imag: # TODO: Is this really the case? Can't we have a Hopf bifurcation for normal mode expansions? I think so, e.g. on a partial_t(u,2)=div(grad(u))+alpha*u, we can have a Hopf bifurcation
            #self.eigenvector=self.get_complex_eigenvector_guess(eigenvector)
            self.V0=numpy.real(self.eigenvector)/numpy.linalg.norm(numpy.real(self.eigenvector))
            self.omega=numpy.imag(prob.get_last_eigenvalues()[eigenvector])
        else:
            self.eigenvector=self.get_real_eigenvector_guess(eigenvector)
            self.V0=self.eigenvector/numpy.linalg.norm(self.eigenvector)
            self.omega=0
            
        self.base_zero_dofs,self.eigen_zero_dofs=self.get_forced_to_zero_dofs()

    def patch_residuals(self,eigen:bool,R:list[NPFloatArray] | NPFloatArray):
        if not isinstance(R,list):
            R=[R]
        res=[]
        for r in R:
            r=numpy.array(r)            
            r[numpy.array(list(self.eigen_zero_dofs if eigen else self.base_zero_dofs),dtype="int64")]=0.0
            res.append(r)
        return res
    
    def patch_matrices(self,eigen:bool,J:DefaultMatrixType | list[DefaultMatrixType],M:DefaultMatrixType | list[DefaultMatrixType]=[])->tuple[DefaultMatrixType,...]:
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
        return tuple(res)

    def get_forced_to_zero_dofs(self):
        prob=self.get_problem()
        if self.azimuthal:
            base_zero_dofs=prob._equation_system._get_forced_zero_dofs_for_eigenproblem(prob.get_eigen_solver(),0,None)
            eigen_zero_dofs=prob._equation_system._get_forced_zero_dofs_for_eigenproblem(prob.get_eigen_solver(),self.azimuthal_m,None)
        else:
            base_zero_dofs=prob._equation_system._get_forced_zero_dofs_for_eigenproblem(prob.get_eigen_solver(),None,0)
            eigen_zero_dofs=prob._equation_system._get_forced_zero_dofs_for_eigenproblem(prob.get_eigen_solver(),None,self.cartesian_k) #type:ignore
        base_zero_dofs=prob.dof_strings_to_global_equations(base_zero_dofs)
        eigen_zero_dofs=prob.dof_strings_to_global_equations(eigen_zero_dofs)
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
    parameter:str # always a real parameter name in this subclass (never None, unlike the base class)

    def __init__(self, problem:Problem,parameter:str,eigenvector:int=0,azimuthal_m:float | None=None,cartesian_k:ExpressionNumOrNone=None,eigenscale:float=1,nonlinear_length_constraint:bool=False):
        super().__init__(problem,eigenvector,azimuthal_m,cartesian_k,eigenscale,nonlinear_length_constraint)
        self.parameter=parameter #type:ignore
        

                
    def get_residuals_and_jacobian(self,require_jacobian:bool,dparameter:str | None=None)->NPFloatArray | tuple[NPFloatArray, DefaultMatrixType]: # type: ignore[override] # see the other implementations
        nl=self.nonlinear_length_constraint
        if not self.has_imag:
            Vr,p=self.get_augmented_dofs().split(startindex=1)
            if not require_jacobian:
                if dparameter is not None:
                    dRdp,dJRdp=self.start_multiassembly().dRdp(dparameter).dJdp(dparameter,self.real_contribution).assemble()
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
                # Four things were wrong here, and none of them could show up until a problem reached
                # this branch at all -- which needs a STATIONARY neutral mode AND no imaginary
                # contribution, i.e. a scalar-only normal-mode problem. Anything with a vector field
                # takes the has_imag branch below, which had none of these.
                #  - dR/dparameter was neither assembled nor placed, so the base rows carried no
                #    parameter column at all and Newton could not move the parameter;
                #  - Raug carried the MATRIX JR instead of JR@Vr, which makes numpy.hstack produce an
                #    object array and Problem.get_custom_residuals_jacobian assert;
                #  - the Hessian and dJ/dp were asked of the BASE residual, but the row they
                #    differentiate is J_real*V, so both have to be the real contribution's (the
                #    dparameter branch above already gets this right);
                #  - dJ_real/dp was patched like a Jacobian, which puts a 1 on the diagonal of every
                #    forced-zero eigen row. That row is the equation V_j=0, whose parameter
                #    derivative is zero, so the derivative goes through the M slot instead.
                R,J,dRdp,JR,HJVr,dJRdP=self.start_multiassembly().R().J().dRdp(self.parameter).J(self.real_contribution).dJdU(Vr,self.real_contribution).dJdp(self.parameter,self.real_contribution).assemble()
                J,=self.patch_matrices(eigen=False,J=J)
                R,dRdp=self.patch_residuals(eigen=False,R=[R,dRdp])
                JR,dJRdP=self.patch_matrices(eigen=True,J=[JR],M=[dJRdP])
                col=lambda C:self.as_matrix_column(C)
                row=lambda R:self.as_matrix_row(R)
                Raug=numpy.hstack([R,JR@Vr,numpy.dot(Vr,Vr if self.nonlinear_length_constraint else self.V0)-self.eigenscale*(self.eigenscale if self.nonlinear_length_constraint else 1)])                                 #type:ignore
                Jaug=scipy.sparse.block_array(
                    [[J,None,col(dRdp)],
                     [HJVr,JR,col(dJRdP@Vr)],
                     [None,row(2*Vr if nl else self.V0),None]]).tocsr()
                return Raug,Jaug #type:ignore
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
                return Raug,Jaug         #type:ignore


class CriticalWavenumberTracker(_NormalModeBifurcationTrackerBase):
    """
    Co-dimension-2 tracker for the *critical* wavenumber of a normal mode instability, either a
    Cartesian mode ~exp(I*k*z) or an azimuthal mode ~exp(I*m*phi).

    :py:class:`NormalModeBifurcationTracker` finds, for a FIXED wavenumber, the parameter value at
    which Re(lambda)=0. The neutral curve gamma_c(k) traced out that way is a one-parameter family;
    this class instead finds its minimum, i.e. the point where in addition dRe(lambda)/dk=0, by making
    the wavenumber a second unknown. That is the wavenumber at which the instability actually sets in.

    Below, k stands for whichever of the two mode parameters is in play: the Cartesian wavenumber
    ``normal_mode_k`` or the azimuthal mode number ``azimuthal_m``. Both are ordinary global
    parameters occurring in the generated eigen contributions, which is what makes either of them
    usable as an unknown; for the azimuthal case this means m is treated as a REAL number (see the
    note below).

    The extra equation is obtained by differentiating the eigenproblem (J_c+lambda*M_c)V=0 with
    respect to k at fixed base state and fixed parameter -- legitimate because the base residual has
    k substituted to zero, so the base state does not depend on k at all::

        (dJ_c/dk + lambda*dM_c/dk)V + (dlambda/dk)*M_c*V + (J_c + lambda*M_c)*dV/dk = 0

    dV/dk enters as a further vector of unknowns. Criticality is imposed the same way the other
    trackers impose Re(lambda)=0: by simply not introducing Re(lambda) and Re(dlambda/dk) as unknowns,
    so lambda=I*omega and dlambda/dk=I*mu. The augmented system therefore has 5N+4 unknowns for an
    oscillatory neutral mode and 3N+2 for a stationary one, and it is square in both cases.

    Once converged, the critical point can be arclength-continued in a FURTHER parameter, giving a
    curve of critical points in a two-parameter plane.

    Args:
        problem: The problem, set up with ``setup_for_stability_analysis`` and either
            ``additional_cartesian_mode=True`` or ``azimuthal_stability=True``
        parameter: Name of the parameter to adjust along with the wavenumber
        eigenvector: Index into the previously solved eigenvalues
        azimuthal_m: Starting azimuthal mode number, real-valued. Defaults to the m of the selected
            eigenvalue. Only for a problem set up for azimuthal stability.
        cartesian_k: Starting wavenumber. Defaults to the k of the selected eigenvalue. Only for a
            problem set up for an additional Cartesian mode.
        eigenscale: Internal magnitude of the eigenvector, as in the other trackers
        k_fd_step: Relative step of the finite difference in the mode parameter used for the
            Jacobian (see below)
        exact_k_derivative_jacobian: If False, the finite-differenced blocks are dropped entirely.
            The converged answer is unchanged (they only enter the Jacobian), but Newton will need
            more steps. Mostly useful to test the analytic blocks in isolation.

    Note:
        The Jacobian of the extra equations needs d2J/dU dk, d2J/dgamma dk and d2J/dk2, none of which
        pyoomph generates. They are obtained by finite-differencing the corresponding first-derivative
        blocks in k. The RESIDUAL stays exact -- it is built from the analytic dJ/dk and dM/dk -- so
        Newton still converges to the exact critical point; only the convergence rate suffers.

    Note:
        For an azimuthal mode, m becomes a REAL unknown. Physically only integer m are admissible, so
        the answer is a critical m to be read as "the instability sets in between these two integer
        modes"; the usual next step is to run the ordinary tracker at the neighbouring integers.
        Numerically the awkward part is that the axis conditions are a DISCRETE function of m (m==0,
        |m|==1, |m|>1 give different pinned dofs), which a continuous m cannot express. This class
        therefore assumes the |m|>1 regime throughout: the mask is frozen at m=2 when the tracker is
        created and is not revisited, and starting at |m|<=1 is refused. It follows that an eigensolve
        after such a run needs m to be put back to an integer, since
        ``Problem.setup_forced_zero_dof_list_for_eigenproblems`` refuses a non-integer one.

    Note:
        On a problem whose residual has terms of odd order in k (so that the imaginary contribution
        exists), the multi-assembly may be refused by the frozen sparsity path -- the Hessian
        products of that contribution reach entries outside the Jacobian's symbolic pattern. Set
        ``problem.use_frozen_sparsity=False``. This is not specific to this class;
        :py:class:`NormalModeBifurcationTracker` is refused in exactly the same situation.
    """

    # The mask of dofs pinned at the axis is a discrete function of m; a real m is taken to mean the
    # |m|>1 regime, which this integer stands for when the mask is computed.
    HIGH_AZIMUTHAL_MODE_REGIME=2
    parameter:str # always a real parameter name in this subclass (never None, unlike the base class)

    def __init__(self, problem:Problem,parameter:str,eigenvector:int=0,azimuthal_m:float | None=None,cartesian_k:ExpressionNumOrNone=None,eigenscale:float=1,k_fd_step:float=1e-6,exact_k_derivative_jacobian:bool=True):
        super().__init__(problem,eigenvector,azimuthal_m,cartesian_k,eigenscale,nonlinear_length_constraint=False) #type:ignore[arg-type] # m is real here, see the class docstring
        self.parameter=parameter
        # The mode number IS a global parameter, which is the whole reason it can be made an unknown:
        # add_parameter pushes its value pointer into the dof vector, and the generated eigen
        # contributions read it from there.
        if self.azimuthal:
            assert problem._azimuthal_stability is not None and problem._azimuthal_mode_param_m is not None
            self.mode_parameter=problem._azimuthal_stability.azimuthal_param_m_name
            self._mode_param=problem._azimuthal_mode_param_m
            if abs(float(self._mode_param.value))<=1+1e-10:
                # The axis conditions differ between m==0, |m|==1 and |m|>1, and the mask is frozen at
                # the |m|>1 one when the tracker is created (see get_forced_to_zero_dofs), so the low
                # modes cannot be represented at all.
                raise RuntimeError("Cannot track a critical azimuthal mode at |m|<=1: this tracker treats m as real and assumes the |m|>1 axis regime throughout, which the low modes do not fall into. Got m="+str(float(self._mode_param.value))+".")
        else:
            assert problem._cartesian_normal_mode_stability is not None and problem._normal_mode_param_k is not None
            self.mode_parameter=problem._cartesian_normal_mode_stability.normal_mode_param_k_name
            self._mode_param=problem._normal_mode_param_k
            if abs(float(self._mode_param.value))<1e-10:
                # The forced-zero dof masks are taken once, from _get_forced_zero_dofs_for_eigenproblem,
                # which returns a DIFFERENT set for k==0 than for k!=0. Starting there (or passing through
                # zero) would silently keep the wrong mask.
                raise RuntimeError("Cannot start the critical wavenumber search at k=0: the set of dofs forced to zero in the eigenproblem is different at k=0 than for k!=0, and it is frozen when the tracker is created.")
        self.k_fd_step=k_fd_step
        self.exact_k_derivative_jacobian=exact_k_derivative_jacobian
        # A complex eigenvalue does NOT imply a complex J_c: the imaginary contribution only exists
        # when the residual has terms of odd order in k (a single derivative along the extra
        # direction). A problem with only Laplacians has an oscillatory mode and no imaginary
        # contribution at all, and asking the multi-assembly for a contribution the problem does not
        # have is a hard error, so every request for it is guarded and replaced by a zero matrix.
        self.has_imag_contribution=bool(problem._set_solved_residual(self.imag_contribution,False,False))
        problem._set_solved_residual("",False,False)
        self.dVdk,dlambda_dk=self._guess_k_derivative()
        self.mu=float(numpy.imag(dlambda_dk))
        print("Starting critical wavenumber tracking at",self.mode_parameter,"=",float(self._mode_param.value),
              "with dlambda/d("+self.mode_parameter+") =",dlambda_dk,"(its real part is the tangency residual to be driven to zero)")

    def get_forced_to_zero_dofs(self):
        """Which dofs the eigenproblem pins, frozen once when the tracker is created.

        For a Cartesian mode this is the base class's answer. For an azimuthal one it is not: m is a
        real unknown here, but the axis conditions are a DISCRETE function of it and
        _get_forced_zero_dofs_for_eigenproblem truncates m towards zero before branching, so the mask
        would silently change under the solver as m wandered across an integer. It is therefore taken
        at the |m|>1 regime, which this class documents itself as assuming.
        """
        if not self.azimuthal:
            return super().get_forced_to_zero_dofs()
        prob=self.get_problem()
        base=prob._equation_system._get_forced_zero_dofs_for_eigenproblem(prob.get_eigen_solver(),0,None)
        eigen=prob._equation_system._get_forced_zero_dofs_for_eigenproblem(prob.get_eigen_solver(),self.HIGH_AZIMUTHAL_MODE_REGIME,None)
        return prob.dof_strings_to_global_equations(base),prob.dof_strings_to_global_equations(eigen)

    def _guess_k_derivative(self)->tuple[NPAnyArray,complex]:
        """Initial guess for dV/dk and dlambda/dk, from one bordered solve on the converged eigenpair.

        Cannot use an eigensolve: solve_eigenproblem is refused once a Python augmentation is
        installed, and by the time this runs the tracker is about to become one. Differentiating
        (J_c+lambda*M_c)V=0 in k at fixed base state gives a bordered system whose border is exactly
        the normalisation constraint <V0,V>=const differentiated, i.e. <V0,dV/dk>=0.
        """
        prob=self.get_problem()
        rc,ic,kn=self.real_contribution,self.imag_contribution,self.mode_parameter
        V=self.eigenvector*self.eigenscale
        try:
            if self.has_imag:
                def req(a):
                    a.J(rc).M(rc).dJdp(kn,rc).dMdp(kn,rc)
                    if self.has_imag_contribution:
                        a.J(ic).M(ic).dJdp(kn,ic).dMdp(kn,ic)
                res=list(PerformCustomMultiAssembly(prob,req).result())
                JR,MR,dJRdk,dMRdk=res[:4]
                if self.has_imag_contribution:
                    JI,MI,dJIdk,dMIdk=res[4:]
                else:
                    Z=self._zero_matrix()
                    JI,MI,dJIdk,dMIdk=Z,Z,Z,Z
                JR,JI,MR,MI,dJRdk,dJIdk,dMRdk,dMIdk=self.patch_matrices(eigen=True,J=[JR,JI],M=[MR,MI,dJRdk,dJIdk,dMRdk,dMIdk])
                Jc,Mc=JR+1j*JI,MR+1j*MI
                Jck,Mck=dJRdk+1j*dJIdk,dMRdk+1j*dMIdk
                lam=1j*self.omega
            else:
                assm=PerformCustomMultiAssembly(prob,lambda a:a.J(rc).M(rc).dJdp(kn,rc))
                JR,MR,dJRdk=assm.result()
                JR,MR,dJRdk=self.patch_matrices(eigen=True,J=[JR],M=[MR,dJRdk])
                Jc,Mc,Jck,Mck=JR,MR,dJRdk,None
                lam=0.0
            N=Jc.shape[0]
            A=Jc+lam*Mc
            rhs=-(Jck@V+(lam*(Mck@V) if Mck is not None else 0))
            border=Mc@V # dlambda/dk multiplies M_c*V
            Abord=scipy.sparse.vstack([
                scipy.sparse.hstack([A,self.as_matrix_column(border)]),
                scipy.sparse.hstack([self.as_matrix_row(self.V0),scipy.sparse.csr_matrix((1,1),dtype=A.dtype)])]).tocsc()
            sol=numpy.asarray(scipy.sparse.linalg.spsolve(Abord,numpy.hstack([rhs,0]).astype(Abord.dtype))).ravel()
            if not numpy.all(numpy.isfinite(sol)):
                raise RuntimeError("non-finite solution of the bordered system")
            return sol[:N],complex(sol[N])
        except Exception as e:
            print("Could not compute an initial guess for dV/dk ("+str(e)+"), starting from zero instead")
            return numpy.zeros_like(V),0j

    def define_augmented_dofs(self, dofs):
        # dV/dk is stored already at the eigenscale (the bordered solve above was fed V*eigenscale),
        # hence no further scaling here, unlike the eigenvector itself.
        dofs.add_vector(numpy.real(self.eigenvector)*self.eigenscale)
        if self.has_imag:
            dofs.add_vector(numpy.imag(self.eigenvector)*self.eigenscale)
        dofs.add_vector(numpy.real(self.dVdk))
        if self.has_imag:
            dofs.add_vector(numpy.imag(self.dVdk))
        dofs.add_parameter(self.parameter)
        if self.has_imag:
            dofs.add_scalar(self.omega)
        dofs.add_parameter(self.mode_parameter)
        if self.has_imag:
            dofs.add_scalar(self.mu)

    # ---------------------------------------------------------------------------------------------
    # small algebra helpers. (J_c+I*omega*M_c) applied to a complex pair shows up in four guises
    # here -- as the eigen rows themselves, as their derivative w.r.t. the base dofs (Hessian
    # blocks), and with the k- or parameter-derivatives of the matrices in place of the matrices --
    # so the combination is written once and fed different operands.
    # ---------------------------------------------------------------------------------------------
    @staticmethod
    def _cplx_rows(ops:Any,omega:float):
        """Real and imaginary part of (J_c+I*omega*M_c) acting on a complex pair. ops is (JR,JI,MR,MI)
        as callables mapping "re"/"im" to that matrix's product with the real/imaginary component."""
        Jr,Ji,Mr,Mi=ops
        re=Jr("re")-Ji("im")-omega*(Mr("im")+Mi("re"))
        im=Ji("re")+Jr("im")+omega*(Mr("re")-Mi("im"))
        return re,im

    @staticmethod
    def _matvec(mats:Any,ar:Any,ai:Any)->Any:
        """Operand set for _cplx_rows built from four matrices and one complex pair."""
        return tuple((lambda A: (lambda s:A@(ar if s=="re" else ai)))(A) for A in mats)

    @staticmethod
    def _pairs(products:Any)->Any:
        """Operand set for _cplx_rows built from eight already-contracted products, given as
        (JR_re,JR_im, JI_re,JI_im, MR_re,MR_im, MI_re,MI_im)."""
        return tuple((lambda a,b: (lambda s:a if s=="re" else b))(products[2*i],products[2*i+1]) for i in range(4))

    @staticmethod
    def _omega_rows(MR:Any,MI:Any,ar:Any,ai:Any):
        """d/d(omega) of the eigen rows: I*M_c applied to the complex pair."""
        return -(MR@ai+MI@ar),MR@ar-MI@ai

    def _pairs_assemble(self,add:Callable[[Any,str],Any],extra:Callable[[Any],Any] | None=None)->Any:
        """One multi-assembly asking `add(request,contribution)` for the real and, if the problem has
        one, the imaginary contribution. Returns (extras, real results, imaginary results), the last
        one filled with zero matrices when there is no imaginary contribution.

        `extra` adds contribution-independent requests (the base residual and its parameter
        derivative), which must not be issued twice."""
        a=self.start_multiassembly()
        nextra=0
        if extra is not None:
            extra(a)
            nextra=len(a._what)
        add(a,self.real_contribution)
        npair=len(a._what)-nextra
        if self.has_imag_contribution:
            add(a,self.imag_contribution)
            res=list(a.assemble())
            return res[:nextra],res[nextra:nextra+npair],res[nextra+npair:]
        res=list(a.assemble())
        Z=self._zero_matrix()
        return res[:nextra],res[nextra:],[Z]*npair

    def _zero_matrix(self)->DefaultMatrixType:
        n=self.get_problem()._get_n_unaugmented_dofs() or self.get_problem().ndof()
        return scipy.sparse.csr_matrix((n,n))

    def _kfd_delta(self)->float:
        return self.k_fd_step*max(abs(float(self._mode_param.value)),1.0)

    def _at_k_offset(self,offset:float,build:Callable[[],Any])->Any:
        """Run build() with the mode parameter shifted. It is one of the augmented dofs, i.e. a raw
        pointer into the global parameter's value, so it must be put back exactly even if the
        assembly throws."""
        kp=self._mode_param
        k0=kp.value
        try:
            kp.value=k0+offset
            return build()
        finally:
            kp.value=k0

    def get_residuals_and_jacobian(self,require_jacobian:bool,dparameter:str | None=None)->NPFloatArray | tuple[NPFloatArray, DefaultMatrixType]: # type: ignore[override] # see the other implementations
        # The imaginary contribution is never named here: every request for it goes through
        # _pairs_assemble, which knows whether the problem has one.
        rc,kn=self.real_contribution,self.mode_parameter
        col=lambda C:self.as_matrix_column(C)
        row=lambda R:self.as_matrix_row(R)
        # Derivative matrices are patched like a mass matrix (rows of forced-zero eigen dofs simply
        # zeroed), NOT like a Jacobian (which additionally gets a 1 on the diagonal). The Jacobian
        # patch turns the eigen row of such a dof into the equation V_j=0; the k- or parameter-
        # derivative of that equation is zero, so putting the identity back there would contradict it.
        if not self.has_imag:
            Vr,Wr,p,k=self.get_augmented_dofs().split(startindex=1)
            if not require_jacobian and dparameter is None:
                R,JR,dJRdk=self.start_multiassembly().R().J(rc).dJdp(kn,rc).assemble()
                R,=self.patch_residuals(eigen=False,R=[R])
                JR,dJRdk=self.patch_matrices(eigen=True,J=[JR],M=[dJRdk])
                return numpy.hstack([R,JR@Vr,dJRdk@Vr+JR@Wr,
                                     numpy.dot(Vr,self.V0)-self.eigenscale,numpy.dot(Wr,self.V0)])
            if not require_jacobian:
                assert dparameter is not None
                def _dp_real():
                    dRdp,dJRdp=self.start_multiassembly().dRdp(dparameter).dJdp(dparameter,rc).assemble()
                    dRdp,=self.patch_residuals(eigen=False,R=[dRdp])
                    # patch_matrices needs one J just to size itself; the parameter derivative goes
                    # through the M slot (zeroed rows, no identity) and the J output is discarded.
                    _,dJRdp=self.patch_matrices(eigen=True,J=[dJRdp],M=[dJRdp])
                    return dRdp,dJRdp
                dRdp,dJRdp=_dp_real()
                # d/dsigma of the eigen rows is analytic; d/dsigma of the tangency rows also needs
                # d2J/dsigma dk, which is not, hence the second pass at a shifted k.
                dEdp=dJRdp@Vr
                delta=self._kfd_delta()
                _,dJRdp_s=self._at_k_offset(delta,_dp_real)
                dFdp=(dJRdp_s@Vr-dEdp)/delta+dJRdp@Wr
                return numpy.hstack([dRdp,dEdp,dFdp,0.0,0.0])
            assert dparameter is None, "dparameter not supported for require_jacobian=True"
            assm=self.start_multiassembly().R().J().dRdp(self.parameter)
            assm.J(rc).dJdp(self.parameter,rc).dJdp(kn,rc).dJdU(Vr,rc).dJdU(Wr,rc)
            R,J,dRdp,JR,dJRdp,dJRdk,HVr,HWr=assm.assemble()
            R,=self.patch_residuals(eigen=False,R=[R])
            J,=self.patch_matrices(eigen=False,J=[J])
            JR,dJRdp,dJRdk,HVr,HWr=self.patch_matrices(eigen=True,J=[JR],M=[dJRdp,dJRdk,HVr,HWr])
            dEdU,dEdp,dEdk=HVr,dJRdp@Vr,dJRdk@Vr
            if self.exact_k_derivative_jacobian:
                def _shifted_real():
                    a=self.start_multiassembly().dJdp(self.parameter,rc).dJdp(kn,rc).dJdU(Vr,rc)
                    s_dJRdp,s_dJRdk,s_HVr=a.assemble()
                    s_dJRdp,s_dJRdk,s_HVr=self.patch_matrices(eigen=True,J=[JR],M=[s_dJRdp,s_dJRdk,s_HVr])[1:]
                    return s_HVr,s_dJRdp@Vr,s_dJRdk@Vr
                delta=self._kfd_delta()
                s_dEdU,s_dEdp,s_dEdk=self._at_k_offset(delta,_shifted_real)
                dkdEdU,dkdEdp,dkdEdk=(s_dEdU-dEdU)/delta,(s_dEdp-dEdp)/delta,(s_dEdk-dEdk)/delta
            else:
                dkdEdU,dkdEdp,dkdEdk=0*dEdU,0*dEdp,0*dEdk
            Raug=numpy.hstack([R,JR@Vr,dJRdk@Vr+JR@Wr,
                               numpy.dot(Vr,self.V0)-self.eigenscale,numpy.dot(Wr,self.V0)])
            Jaug=scipy.sparse.block_array([
                [J,        None,   None, col(dRdp),  None],
                [dEdU,     JR,     None, col(dEdp),  col(dEdk)],
                [dkdEdU+HWr,dJRdk, JR,   col(dkdEdp+dJRdp@Wr),col(dkdEdk+dJRdk@Wr)],
                [None,     row(self.V0),None,None,   None],
                [None,     None,   row(self.V0),None,None]]).tocsr()
            return Raug,Jaug #type:ignore
        else:
            Vr,Vi,Wr,Wi,p,omega,k,mu=self.get_augmented_dofs().split(startindex=1)
            omega,mu=omega[0],mu[0]
            def _plain():
                _,re_,im_=self._pairs_assemble(lambda a,c:a.J(c).M(c).dJdp(kn,c).dMdp(kn,c))
                JR,MR,dJRdk,dMRdk=re_
                JI,MI,dJIdk,dMIdk=im_
                return self.patch_matrices(eigen=True,J=[JR,JI],M=[MR,MI,dJRdk,dJIdk,dMRdk,dMIdk])
            if not require_jacobian and dparameter is None:
                R,=self.start_multiassembly().R().assemble()
                R,=self.patch_residuals(eigen=False,R=[R])
                JR,JI,MR,MI,dJRdk,dJIdk,dMRdk,dMIdk=_plain()
                Er,Ei=self._cplx_rows(self._matvec((JR,JI,MR,MI),Vr,Vi),omega)
                Pr,Pi=self._cplx_rows(self._matvec((dJRdk,dJIdk,dMRdk,dMIdk),Vr,Vi),omega)
                Qr,Qi=self._cplx_rows(self._matvec((JR,JI,MR,MI),Wr,Wi),omega)
                Owr,Owi=self._omega_rows(MR,MI,Vr,Vi)
                return numpy.hstack([R,Er,Ei,Pr+Qr+mu*Owr,Pi+Qi+mu*Owi,
                                     numpy.dot(Vr,self.V0)-self.eigenscale,numpy.dot(Vi,self.V0),
                                     numpy.dot(Wr,self.V0),numpy.dot(Wi,self.V0)])
            if not require_jacobian:
                assert dparameter is not None
                def _dp():
                    ex,re_,im_=self._pairs_assemble(lambda a,c:a.dJdp(dparameter,c).dMdp(dparameter,c),
                                                    lambda a:a.dRdp(dparameter))
                    dRdp,=self.patch_residuals(eigen=False,R=[ex[0]])
                    # patch_matrices needs one J just to size itself; every parameter derivative goes
                    # through the M slot (zeroed rows, no identity) and the J output is discarded.
                    _,dJRdp,dJIdp,dMRdp,dMIdp=self.patch_matrices(eigen=True,J=[re_[0]],M=[re_[0],im_[0],re_[1],im_[1]])
                    return dRdp,(dJRdp,dJIdp,dMRdp,dMIdp)
                dRdp,dmats=_dp()
                dEr,dEi=self._cplx_rows(self._matvec(dmats,Vr,Vi),omega)
                dQr,dQi=self._cplx_rows(self._matvec(dmats,Wr,Wi),omega)
                dOwr,dOwi=self._omega_rows(dmats[2],dmats[3],Vr,Vi)
                delta=self._kfd_delta()
                _,smats=self._at_k_offset(delta,_dp)
                sEr,sEi=self._cplx_rows(self._matvec(smats,Vr,Vi),omega)
                dFr=(sEr-dEr)/delta+dQr+mu*dOwr
                dFi=(sEi-dEi)/delta+dQi+mu*dOwi
                return numpy.hstack([dRdp,dEr,dEi,dFr,dFi,0,0,0,0])
            assert dparameter is None, "dparameter not supported for require_jacobian=True"
            R,J,dRdp=self.start_multiassembly().R().J().dRdp(self.parameter).assemble()
            R,=self.patch_residuals(eigen=False,R=[R])
            J,=self.patch_matrices(eigen=False,J=[J])
            JR,JI,MR,MI,dJRdk,dJIdk,dMRdk,dMIdk=_plain()
            def _add_param_and_hessians(a,c):
                a.dJdp(self.parameter,c).dMdp(self.parameter,c)
                a.dJdU(Vr,c).dJdU(Vi,c).dMdU(Vr,c).dMdU(Vi,c)
                a.dJdU(Wr,c).dJdU(Wi,c).dMdU(Wr,c).dMdU(Wi,c)
            _,gr,gi=self._pairs_assemble(_add_param_and_hessians)
            gr=self.patch_matrices(eigen=True,J=[JR],M=list(gr))[1:]
            gi=self.patch_matrices(eigen=True,J=[JR],M=list(gi))[1:]
            dJRdp,dMRdp=gr[0],gr[1]
            dJIdp,dMIdp=gi[0],gi[1]
            # _pairs takes (JR_re,JR_im, JI_re,JI_im, MR_re,MR_im, MI_re,MI_im), i.e. the eight
            # Hessian-vector products interleaved by contribution.
            HV=[gr[2],gr[3],gi[2],gi[3],gr[4],gr[5],gi[4],gi[5]]
            HW=[gr[6],gr[7],gi[6],gi[7],gr[8],gr[9],gi[8],gi[9]]
            hess=lambda H:self._pairs(H)
            dEdU_re,dEdU_im=self._cplx_rows(hess(HV),omega)
            dEdp_re,dEdp_im=self._cplx_rows(self._matvec((dJRdp,dJIdp,dMRdp,dMIdp),Vr,Vi),omega)
            Pr,Pi=self._cplx_rows(self._matvec((dJRdk,dJIdk,dMRdk,dMIdk),Vr,Vi),omega)
            if self.exact_k_derivative_jacobian:
                delta=self._kfd_delta()
                def _add_shifted(s,c):
                    s.dJdp(kn,c).dMdp(kn,c).dJdp(self.parameter,c).dMdp(self.parameter,c)
                    s.dJdU(Vr,c).dJdU(Vi,c).dMdU(Vr,c).dMdU(Vi,c)
                def _shifted():
                    _,sr,si=self._pairs_assemble(_add_shifted)
                    sr=self.patch_matrices(eigen=True,J=[JR],M=list(sr))[1:]
                    si=self.patch_matrices(eigen=True,J=[JR],M=list(si))[1:]
                    sP=self._cplx_rows(self._matvec((sr[0],si[0],sr[1],si[1]),Vr,Vi),omega)
                    sdEdp=self._cplx_rows(self._matvec((sr[2],si[2],sr[3],si[3]),Vr,Vi),omega)
                    sdEdU=self._cplx_rows(hess([sr[4],sr[5],si[4],si[5],sr[6],sr[7],si[6],si[7]]),omega)
                    return sP,sdEdp,sdEdU
                sP,sdEdp,sdEdU=self._at_k_offset(delta,_shifted)
                dkP=((sP[0]-Pr)/delta,(sP[1]-Pi)/delta)
                dkdEdp=((sdEdp[0]-dEdp_re)/delta,(sdEdp[1]-dEdp_im)/delta)
                dkdEdU=((sdEdU[0]-dEdU_re)/delta,(sdEdU[1]-dEdU_im)/delta)
            else:
                dkP=(0*Pr,0*Pi)
                dkdEdp=(0*dEdp_re,0*dEdp_im)
                dkdEdU=(0*dEdU_re,0*dEdU_im)
            # Everything below is analytic.
            Qr,Qi=self._cplx_rows(self._matvec((JR,JI,MR,MI),Wr,Wi),omega)
            Er,Ei=self._cplx_rows(self._matvec((JR,JI,MR,MI),Vr,Vi),omega)
            Owr,Owi=self._omega_rows(MR,MI,Vr,Vi)               # dE/domega, and dF/dmu
            Owr_W,Owi_W=self._omega_rows(MR,MI,Wr,Wi)           # dE/domega with V->W
            Owr_k,Owi_k=self._omega_rows(dMRdk,dMIdk,Vr,Vi)     # d/dk of dE/domega (analytic)
            Owr_p,Owi_p=self._omega_rows(dMRdp,dMIdp,Vr,Vi)     # d/dparameter of dE/domega
            dEdU_W_re,dEdU_W_im=self._cplx_rows(hess(HW),omega)
            dEdp_W_re,dEdp_W_im=self._cplx_rows(self._matvec((dJRdp,dJIdp,dMRdp,dMIdp),Wr,Wi),omega)
            P_W_re,P_W_im=self._cplx_rows(self._matvec((dJRdk,dJIdk,dMRdk,dMIdk),Wr,Wi),omega)
            # d/du of dE/domega, i.e. the mass-matrix half of the Hessian combination
            OwU_re,OwU_im=-(HV[5]+HV[6]),HV[4]-HV[7]
            # dE/dV blocks, and their k-derivative, which IS analytic (the primed matrices)
            EVr_re,EVr_im=JR-omega*MI,JI+omega*MR
            EVi_re,EVi_im=-JI-omega*MR,JR-omega*MI
            kEVr_re,kEVr_im=dJRdk-omega*dMIdk,dJIdk+omega*dMRdk
            kEVi_re,kEVi_im=-dJIdk-omega*dMRdk,dJRdk-omega*dMIdk
            Raug=numpy.hstack([R,Er,Ei,Pr+Qr+mu*Owr,Pi+Qi+mu*Owi,
                               numpy.dot(Vr,self.V0)-self.eigenscale,numpy.dot(Vi,self.V0),
                               numpy.dot(Wr,self.V0),numpy.dot(Wi,self.V0)])
            Jaug=scipy.sparse.block_array([
                # u                        Vr                   Vi                   Wr        Wi        gamma                            omega                    k                          mu
                [J,                        None,                None,                None,     None,     col(dRdp),                       None,                    None,                      None],
                [dEdU_re,                  EVr_re,              EVi_re,              None,     None,     col(dEdp_re),                    col(Owr),                col(Pr),                   None],
                [dEdU_im,                  EVr_im,              EVi_im,              None,     None,     col(dEdp_im),                    col(Owi),                col(Pi),                   None],
                [dkdEdU[0]+dEdU_W_re+mu*OwU_re, kEVr_re-mu*MI,  kEVi_re-mu*MR,       EVr_re,   EVi_re,   col(dkdEdp[0]+dEdp_W_re+mu*Owr_p),col(Owr_k+Owr_W),       col(dkP[0]+P_W_re+mu*Owr_k),col(Owr)],
                [dkdEdU[1]+dEdU_W_im+mu*OwU_im, kEVr_im+mu*MR,  kEVi_im-mu*MI,       EVr_im,   EVi_im,   col(dkdEdp[1]+dEdp_W_im+mu*Owi_p),col(Owi_k+Owi_W),       col(dkP[1]+P_W_im+mu*Owi_k),col(Owi)],
                [None,row(self.V0),None,None,None,None,None,None,None],
                [None,None,row(self.V0),None,None,None,None,None,None],
                [None,None,None,row(self.V0),None,None,None,None,None],
                [None,None,None,None,row(self.V0),None,None,None,None]]).tocsr()
            return Raug,Jaug #type:ignore

    def actions_after_successful_newton_solve(self):
        prob=self.get_problem()
        if self.has_imag:
            Vr,Vi,Wr,Wi,p,omega,k,mu=self.get_augmented_dofs().split(startindex=1)
            self.store_eigenvector({(1j*omega[0]):(numpy.array(Vr)+numpy.array(Vi)*1j)})
        else:
            Vr,Wr,p,k=self.get_augmented_dofs().split(startindex=1)
            self.store_eigenvector({0.0:numpy.array(Vr)})
        # store_eigenvector clears the mode bookkeeping; put the wavenumber we just found back, so
        # that get_last_eigenmodes_k()/get_last_eigenmodes_m() still describes the stored
        # eigenvector. The azimuthal one is deliberately NOT cast to an integer here: it is a real
        # unknown of this system, and rounding it would report something that was never solved for.
        if self.azimuthal:
            prob._last_eigenvalues_m=numpy.array([k[0]])
        else:
            prob._last_eigenvalues_k=numpy.array([k[0]])

    def get_critical_mode(self)->float:
        """The converged mode unknown: the Cartesian wavenumber k or the (real) azimuthal mode m."""
        return float(self.get_augmented_dofs().split(startindex=1)[-2 if self.has_imag else -1][0])

    def get_critical_wavenumber(self)->float:
        """The critical Cartesian wavenumber k, non-dimensional -- divide by the spatial scaling for
        a dimensional one, as Problem.get_current_normal_mode_k does."""
        if self.azimuthal:
            raise RuntimeError("This tracker follows an azimuthal mode; use get_critical_azimuthal_m().")
        return self.get_critical_mode()

    def get_critical_azimuthal_m(self)->float:
        """The critical azimuthal mode number, as a REAL number. Only integers are physical, so read
        it as "the instability sets in between the neighbouring integer modes"."""
        if not self.azimuthal:
            raise RuntimeError("This tracker follows a Cartesian mode; use get_critical_wavenumber().")
        return self.get_critical_mode()

    def get_critical_omega(self)->float:
        """Imaginary part of the neutral eigenvalue; zero for a stationary instability."""
        if not self.has_imag:
            return 0.0
        return float(self.get_augmented_dofs().split(startindex=1)[-3][0])

    def get_dlambda_dmode(self)->complex:
        """dlambda/dk (or dlambda/dm) at the critical point. Its real part is zero by construction --
        that is the equation this tracker adds -- so only the imaginary part carries information."""
        if not self.has_imag:
            return 0j
        return 1j*float(self.get_augmented_dofs().split(startindex=1)[-1][0])

    def get_dlambda_dk(self)->complex:
        """Alias of :py:meth:`get_dlambda_dmode`, for the Cartesian case."""
        return self.get_dlambda_dmode()


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
        
    def get_residuals_and_jacobian(self,require_jacobian:bool,dparameter:str | None=None)->NPFloatArray | tuple[NPFloatArray, DefaultMatrixType]: # type: ignore[override] # the base's overloads narrow the return by require_jacobian; every implementation here is the general one
        V,lam=self.get_augmented_dofs().split(startindex=1)
        lam=lam[0]
        assembly=self.start_multiassembly()
        HJV=None
        HMV=None
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
            assert HJV is not None and HMV is not None
            col=lambda C:self.as_matrix_column(C)
            row=lambda R:self.as_matrix_row(R)
            Jaug=scipy.sparse.block_array(
                [[J,None,None],
                 [lam*HMV+HJV,lam*M+J,col(M@V)],
                 [None,row(2*V if nl else self.V0),None]]).tocsr()            
            return Raug,Jaug #type:ignore
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
        
    def get_residuals_and_jacobian(self,require_jacobian:bool,dparameter:str | None=None)->NPFloatArray | tuple[NPFloatArray, DefaultMatrixType]: # type: ignore[override] # the base's overloads narrow the return by require_jacobian; every implementation here is the general one
        Vr,Vi,lam,omega=self.get_augmented_dofs().split(startindex=1)
        lam,omega=lam[0],omega[0]
        assembly=self.start_multiassembly()
        HJVr=None
        HJVi=None
        HMVr=None
        HMVi=None
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
            assert HJVr is not None and HJVi is not None and HMVr is not None and HMVi is not None
            col=lambda C:self.as_matrix_column(C)
            row=lambda R:self.as_matrix_row(R)
            Jaug=scipy.sparse.block_array(
                [[J,None,None,None,None],
                 [lam*HMVr-omega*HMVi+HJVr,lam*M+J,-omega*M, col(M@Vr),col(-M@Vi)],
                 [lam*HMVi+omega*HMVr+HJVi,omega*M,lam*M+J, col(M@Vi),col(M@Vr)],
                 [None,row(2*Vr if nl else self.V0),None,None,None],
                 [None,row(Vi) if nl else None,row(Vr if nl else self.V0),None,None]]).tocsr()            
            return Raug,Jaug #type:ignore
        else:
            return Raug

    def actions_after_successful_newton_solve(self)->None:
        Vr,Vi,lam,omg=self.get_augmented_dofs().split(startindex=1)
        self.store_eigenvector({lam[0]+1j*omg[0]:numpy.array(Vr)+numpy.array(Vi)*1j})
        


class NormalModeEigenbranchTracker(_NormalModeBifurcationTrackerBase):
    def __init__(self, problem,eigenvector:int,azimuthal_m:float | None=None,cartesian_k:ExpressionNumOrNone=None,eigenscale:float=1,nonlinear_length_constraint:bool=False):
        super().__init__(problem,eigenvector,azimuthal_m,cartesian_k,eigenscale,nonlinear_length_constraint)
        self.parameter=None # No parameter means essentially take the real part as adjustable parameter
                
    def get_residuals_and_jacobian(self,require_jacobian:bool,dparameter:str | None=None)->NPFloatArray | tuple[NPFloatArray, DefaultMatrixType]: # type: ignore[override] # see the other implementations
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
                    R,=self.patch_residuals(eigen=False,R=[R])
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
                return Raug,Jaug #type:ignore
        else:
            Vr,Vi,lamb,omega=self.get_augmented_dofs().split(startindex=1)
            lamb,omega=lamb[0],omega[0]
            if not require_jacobian:
                if dparameter is not None:
                    assm=self.start_multiassembly().dRdp(dparameter).dJdp(dparameter,self.real_contribution).dJdp(dparameter,self.imag_contribution)
                    assm.dMdp(dparameter,self.real_contribution).dMdp(dparameter,self.imag_contribution)
                    dRdp,dJRdp,dJIdp,dMRdp,dMIdp=assm.assemble()
                    dRdp,=self.patch_residuals(eigen=False,R=[dRdp])                    
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
                return Raug,Jaug #type:ignore
    




def EigenbranchTracker(problem:Problem,eigenvector:int=0,eigenscale:float=1,nonlinear_length_constraint:bool=False, complex_threshold:float=1e-8):
    # method to get the right eigenbranch tracker class depending on the type of the eigenvalue
    if eigenvector<0 or eigenvector>=len(problem.get_last_eigenvalues()):
        raise RuntimeError("Eigenvalue index out of range")
    normal_mode=False

    last_modes_k=problem.get_last_eigenmodes_k()
    if last_modes_k is not None and eigenvector<len(last_modes_k):
        normal_mode=True
    last_modes_m=problem.get_last_eigenmodes_m()
    if last_modes_m is not None and eigenvector<len(last_modes_m):
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
    
    def get_residuals_and_jacobian(self,require_jacobian:bool,dparameter:str | None=None)->NPFloatArray | tuple[NPFloatArray, DefaultMatrixType]: # type: ignore[override] # the base's overloads narrow the return by require_jacobian; every implementation here is the general one
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
        previous=getattr(self.get_problem(),"_custom_assembler",None)
        self.get_problem().set_custom_assembler(self)
        try:
            self.res=self.get_residuals_and_jacobian(require_jacobian=False)
        finally:
            # In a finally, and restoring what was there rather than clearing to None. Without this an
            # assembly that raises - a normal form asked of a problem built without an analytic Hessian
            # is the easy way to trigger it - left this handler installed on the Problem, and every
            # later assembly in the run went through it. The symptom was the Hessian complaint
            # reappearing from an unrelated arclength step, long after the call that caused it.
            self.get_problem().set_custom_assembler(previous)
        
    def result(self)->Any:
        return self.res

    def define_augmented_dofs(self, dofs):
        pass

    def get_residuals_and_jacobian(self,require_jacobian:bool,dparameter:str | None=None)->Any:
        # This class deliberately returns whatever the request lambda assembled (an arbitrary-length
        # tuple of arrays/matrices), not the fixed (residual)/(residual,jacobian) shape the base class
        # signature promises - hence the Any return type override.
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
        if numpy.isnan(numpy.sum(res)): #type:ignore
            print("Matrix rank warning. Going for a least squares solution")
            #res= numpy.linalg.lstsq(A.toarray(),rhs,rcond=None)[0]
            res=scipy.sparse.linalg.lsqr(A,rhs)[0] #type:ignore
        return res
            
      def get_left_eigenvector(self,lamb):
            self.problem._require_non_distributed("Left eigenvector calculation")
            n, M_nzz, M_nr, M_val, M_ci, M_rs, J_nzz, J_nr, J_val, J_ci, J_rs = self.problem.assemble_eigenproblem_matrices(0) #type:ignore
            M=csr_matrix((M_val, M_ci, M_rs), shape=(n, n))	#TODO: Is csr or csc?
            A=csr_matrix((-J_val, J_ci, J_rs), shape=(n, n))
            AT=A.transpose().tocsr()
            MT=M.transpose().tocsr()
            if not self.problem.get_eigen_solver().supports_target():
                if numpy.abs(numpy.imag(lamb))>1e-8:
                    raise RuntimeError("Eigenvalue is not real. Complex left eigenvector must be implemented for a solver without target")
                evals,evects,_,_=self.problem.get_eigen_solver().solve(2,shift=1e-7,custom_J_and_M=(AT,MT)) #type:ignore
            else:    
                evals,evects,_,_=self.problem.get_eigen_solver().solve(2,shift=1e-7,target=lamb,custom_J_and_M=(AT,MT)) #type:ignore
            closest=numpy.argmin(numpy.abs(evals-lamb))
            return evects[closest],evals[closest]
      
      def get_normal_form(self,param:str | None=None,eigenindex:int=0,assume:str | None=None):
        """Normal form of the bifurcation the problem is sitting at, Hopf or real.

        ``assume`` is passed on to :py:meth:`get_normal_form1d` and forces its fold/branch-point
        decision; it has no meaning for a Hopf and is rejected there.
        """
        if param is None:
            if self.problem._bifurcation_tracking_parameter_name is not None and self.problem._bifurcation_tracking_parameter_name!="":
                param=self.problem._bifurcation_tracking_parameter_name
            else:
                raise RuntimeError("Pass a parameter or use this with solved bifurcation tracking active")
        if self.problem.get_last_eigenvalues() is None or eigenindex>=len(self.problem.get_last_eigenvalues()):
            raise RuntimeError("Eigenpair at index "+str(eigenindex)+" not calculated!")
        lambd=self.problem.get_last_eigenvalues()[eigenindex]
        if numpy.abs(numpy.imag(lambd))>1e-8:
            if assume is not None:
                raise ValueError("assume applies to the fold/branch-point decision of a real "
                                 "bifurcation; this eigenvalue is complex, so it is a Hopf")
            return self.get_normal_form_hopf(param=param,eigenindex=eigenindex)
        else:
            return self.get_normal_form1d(param=param,eigenindex=eigenindex,assume=assume)
        
      def get_normal_form_hopf(self,param:str | None=None,eigenindex:int=0):
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
            
            self.problem._require_non_distributed("Bifurcation branch switching")
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
            res:dict[str,Any]={}
            res["type"]="hopf"
            psign=1 if numpy.real(a)/numpy.real(b)<0 else -1
            res["psign"]=psign
            res["zeta"]=zeta
            res["param_predictor"]=lambda dp : psign*abs(dp)
            res["omega_predictor"]=lambda dp : omega+ numpy.imag(a)*dp - numpy.real(a)*numpy.imag(b)*dp/numpy.real(b) 
            res["perturbation_predictor"]=lambda dp,omegat: u+numpy.sqrt(numpy.abs(numpy.real(a)*dp/(2*numpy.real(b))))*numpy.real(zeta*numpy.exp(1j*omegat)+czeta*numpy.exp(-1j*omegat))
            return res
            
            
            
            
      def get_normal_form1d(self,param:str | None=None,eigenindex:int=0,assume:str | None=None,tol_fold:float=1e-3):
            """Normal form of a real (non-Hopf) bifurcation the problem is sitting at.

            ``assume`` overrides the fold/branch-point decision with "fold" or "branch_point" - useful
            when the geometry of the branch already answers it, e.g. when the continuation walked
            THROUGH the point without the parameter turning around, which no fold does. ``tol_fold`` is
            compared against the scale-free measure described below, not against a itself.
            """
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
            # "Is a zero?" is the fold/branch-point question, but a itself carries no scale: zeta_star is
            # normalised by 1/<zeta,zeta_star>, so a mode whose left and right null vectors overlap
            # poorly - which is exactly what happens when a second eigenvalue sits close by - inflates a
            # by that factor alone. An absolute threshold then reads a branch point as a fold. What does
            # not move is the ANGLE between R01=-dR/dp and the left null vector: a=0 means dR/dp lies in
            # the range of J, i.e. the parameter cannot move the solution off the branch, and the cosine
            # says that scale-free. Measured: a genuine fold gives O(1) here, a branch point 1e-5 or less.
            a_rel=abs(a)/max(numpy.linalg.norm(R01)*numpy.linalg.norm(zeta_star),1e-300)
            res={"a":a,"b1":b1,"b2":b2,"b3":b3,"a_rel":a_rel}
            print("a=",a," (relative to |dR/dp| and the left null vector:",a_rel,")")
            print("b1=",b1)
            print("b2=",b2)
            print("b3=",b3)
            if assume not in (None,"fold","branch_point"):
                raise ValueError("assume must be None, 'fold' or 'branch_point', got "+repr(assume))
            is_fold=(a_rel>=tol_fold) if assume is None else (assume=="fold")
            if assume is not None:
                print("Classified as a "+("fold" if is_fold else "branch point")+" on request, not from a")
            elif tol_fold/10<a_rel<tol_fold*10:
                # Within a decade of the threshold either way. Measured, the two cases sit three decades
                # apart (a Bratu fold gives 0.94, a branch point 3e-5), so anything this close is worth
                # saying out loud rather than deciding silently.
                print("WARNING: this is only",a_rel,"against a threshold of",tol_fold,
                      "- the fold/branch-point verdict is not clear-cut here. Pass assume='fold' or "
                      "assume='branch_point' if you know which it is.")
            if not is_fold:
                if 100*abs(b2/2) < abs(b3/6):
                    print("Likely a pitchfork")
                    psign=1 if b1*b3<0 else -1
                    print("With sign",psign)
                    res["type"]="pitchfork"
                    res["psign"]=psign
                else:
                    print("Likely transcritical")
                    res["type"]="transcritical"
                    res["psign"]="arbitrary" # Can be chosen arbitrarily
            else:
                print("Likely fold")
                res["type"]="fold"
                res["psign"]=0 # Parameter may not change
            res["zeta"]=zeta
            attach_normal_form_predictors(res)
            return res


def attach_normal_form_predictors(nf:dict[str,Any])->dict[str,Any]:
      """Fill in a real normal form's predictors from its coefficients and its null vector.

      Kept apart from the classification that computes those coefficients so that there is one copy of
      each formula rather than two: a normal form read back from a saved bifurcation diagram keeps the
      coefficients, which are a handful of numbers, but not zeta, which is the size of the problem. Give
      it a freshly computed zeta and this makes it predict exactly what the original did - provided zeta
      is normalised the same way, to unit length, because b2 and b3 scale with it.

      Hopf is not among the types handled here. Its predictor is parameterised by the phase omega*t as
      well and returns an absolute state rather than a perturbation, and nothing reads it back; a
      restored Hopf therefore carries its coefficients and its zeta, and no predictors.
      """
      zeta=nf.get("zeta")
      if zeta is None:
            return nf
      t=nf.get("type")
      if t=="pitchfork":
            b1,b3=nf["b1"],nf["b3"]
            psign=nf["psign"]
            nf["param_predictor"]=lambda dp : psign*abs(dp)
            # The SIGN of dp picks which of the two symmetric branches is predicted: they sit at
            # +-zeta and share the same parameter side, so without this both of switch_branch's
            # two directions asked for the very same point and one of the two branches could
            # never be reached.
            nf["perturbation_predictor"]=lambda dp: numpy.sign(dp)*zeta*numpy.sqrt(abs(6*b1/b3*dp))
      elif t=="transcritical":
            b1,b2=nf["b1"],nf["b2"]
            nf["param_predictor"]=lambda dp : dp
            nf["perturbation_predictor"]=lambda dp: -zeta*2*b1/b2*dp
      elif t=="fold":
            nf["param_predictor"]=lambda dp : 0
            nf["perturbation_predictor"]=lambda dp: zeta*dp
      return nf


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
