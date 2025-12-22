from ..generic.problem import Problem
from ..typings import *
from ..generic.bifurcation_tools import MultiAssembleRequest
import numpy

class HalleySolver:
    def __init__(self,problem:Problem):
        super().__init__()
        self.problem=problem
             
                
    def solve(self,*,max_iterations:Optional[int]=None,accuracy:Optional[float]=None):
        
        # Currently, only stationary solves supported
        if not self.problem.is_initialised():
            self.problem.initialise()

        if max_iterations is None:
            max_iterations=self.problem.max_newton_iterations
        if accuracy is None:
            accuracy=self.problem.newton_solver_tolerance
                    
        ntstep=self.problem.ntime_stepper()
        was_steady=[False]*ntstep
        for i in range(ntstep):
            ts=self.problem.time_stepper_pt(i)
            was_steady[i]=ts.is_steady()
            ts.make_steady()
        
        self.problem.actions_before_stationary_solve()
        self.problem.actions_before_newton_solve()
        step=1
        R=numpy.array(self.problem.get_residuals())        
        dofs,_=self.problem.get_current_dofs()
        while True:
            Rorig=R.copy()
            J=self.problem.assemble_jacobian(with_residual=False)            
            self.problem.get_la_solver().solve_serial(1,J.shape[0],J.nnz,1,J.data,J.indices,J.indptr,R,0,0)
            self.problem.get_la_solver().solve_serial(2,J.shape[0],J.nnz,1,J.data,J.indices,J.indptr,R,0,0)
            
            _augdof_spec=self.problem._create_dof_augmentation()            
            self.problem._add_augmented_dofs(_augdof_spec)
            request=MultiAssembleRequest(self.problem)
            request.dJdU(R)
            dJdU,=request.assemble()
            self.problem._reset_augmented_dof_vector_to_nonaugmented()
            J=J-dJdU/2
            self.problem.get_la_solver().solve_serial(1,J.shape[0],J.nnz,1,J.data,J.indices,J.indptr,Rorig,0,0)
            self.problem.get_la_solver().solve_serial(2,J.shape[0],J.nnz,1,J.data,J.indices,J.indptr,Rorig,0,0)
                        
            dofs=dofs-Rorig
            self.problem.set_current_dofs(dofs)
            self.problem.invalidate_cached_mesh_data()
            self.problem.actions_before_newton_convergence_check()
            R=numpy.array(self.problem.get_residuals())
            err=numpy.linalg.norm(R, ord=numpy.inf)
            print("Halley step ",str(step)+":","Residual norm:",err)            
            if err<accuracy:
                print("Converged!")
                break            
            if step>=max_iterations:
                raise RuntimeError("Halley solver did not converge within the maximum number of iterations")
            step+=1
            self.problem.actions_after_newton_step()
            
                                                    
        self.problem.actions_after_newton_solve()                                                            
        for i in range(ntstep):
            if not was_steady[i]:
                self.problem.time_stepper_pt(i).undo_make_steady()     
