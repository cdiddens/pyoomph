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
from ..generic.problem import Problem
from ..typings import *
from ..generic.bifurcation_tools import MultiAssembleRequest
import numpy

class HalleySolver:
    """Halley's method -- cubically convergent -- in place of Newton, for a stationary solve.

    Replicated, not distributed. Everything here works on GLOBAL vectors: ``get_residuals()`` and
    ``set_current_dofs()`` take the whole dof vector, and the two factorisations go through
    ``solve_serial`` with the whole Jacobian. Under a replicated ``mpirun`` that is correct (every
    rank holds all of it, and every rank does the same work); under ``--distribute`` it is not, and
    the refusal in :py:meth:`solve` says so rather than letting it return a plausible wrong answer.

    Making it distributed is not a retyping job: ``MultiAssembleRequest.assemble()`` hands back this
    rank's ``(nrow_local, n)`` row block with global column indices, so ``J - dJdU/2`` and both
    solves would have to move onto the LA backend -- which is mechanical -- but the method itself has
    no caller, no tutorial and no test anywhere in the tree, so there would be nothing to verify the
    migration against. The guard and a serial test are the honest state.
    """

    def __init__(self,problem:Problem):
        super().__init__()
        self.problem=problem
             
                
    def solve(self,*,max_iterations:int | None=None,accuracy:float | None=None):
        
        # Currently, only stationary solves supported
        if not self.problem.is_initialised():
            self.problem.initialise()
        # See the class docstring: global dof vectors and solve_serial throughout. Without this the
        # method runs under --distribute and converges to something, because each rank would treat
        # its own row block as the whole system.
        self.problem._require_non_distributed("Halley's method")

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
            J=self.problem.assemble_jacobian(with_residual=False,which_one="")
            # Tell the solver its factorisation slot is being reused for a system pyoomph built here, not
            # for the one solve_distributed() gathered. Under mpirun the gathered Newton solve keeps
            # rank 0's factors in that same slot, and a back-substitution landing on these ones instead
            # would be silently wrong on every rank at once.
            self.problem.get_la_solver()._note_external_serial_solve()
            self.problem.get_la_solver().solve_serial(1,J.shape[0],J.nnz,1,J.data,J.indices,J.indptr,R,0,0) #type:ignore[attr-defined] # scipy.sparse.csr_matrix attrs unresolved without scipy-stubs (blocked on numpy<2 pin)
            self.problem.get_la_solver().solve_serial(2,J.shape[0],J.nnz,1,J.data,J.indices,J.indptr,R,0,0) #type:ignore[attr-defined]
            
            _augdof_spec=self.problem._create_dof_augmentation()            
            self.problem._add_augmented_dofs(_augdof_spec)
            request=MultiAssembleRequest(self.problem)
            request.dJdU(R)
            dJdU,=request.assemble()
            self.problem._reset_augmented_dof_vector_to_nonaugmented()
            J=J-dJdU/2
            self.problem.get_la_solver().solve_serial(1,J.shape[0],J.nnz,1,J.data,J.indices,J.indptr,Rorig,0,0) #type:ignore[attr-defined]
            self.problem.get_la_solver().solve_serial(2,J.shape[0],J.nnz,1,J.data,J.indices,J.indptr,Rorig,0,0) #type:ignore[attr-defined]
                        
            dofs=dofs-Rorig
            self.problem.set_current_dofs(dofs.tolist())
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


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
