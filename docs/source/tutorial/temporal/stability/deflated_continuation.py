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



from pyoomph import *
from pyoomph.expressions import *

class PitchForkNormalForm(ODEEquations):
    def __init__(self,r,sign=-1):
        super(PitchForkNormalForm, self).__init__()
        self.r=r
        self.sign=sign

    def define_fields(self):
        self.define_ode_variable("x")

    def define_residuals(self):
        x,x_test=var_and_test("x") 
        self.add_residual((partial_t(x)-(self.r*x+self.sign*x**3))*x_test)

if __name__=="__main__":
    problem=Problem()
    r=problem.define_global_parameter(r=-1)
    problem+=PitchForkNormalForm(r=r,sign=-1)@"pitchfork"
    
    # Storage for the output files: Branch index -> output file
    output_files={}
    
    # Scan r from -1 to 1, apply deflated continuation
    for branch_index,rvalue,sol in problem.deflated_continuation(r=numpy.linspace(-1,1,50)):
        # we get the branch_index (increasing), the value of the parameter and the degrees of freedom
        if branch_index not in output_files:
            # Create an output file for the new branch
            output_files[branch_index]=problem.create_text_file_output("branch_{:02d}.txt".format(branch_index))
        # We can e.g. solve eigenproblems, or output solutions here
        problem.solve_eigenproblem(1)
        Re_ev=numpy.real(problem.get_last_eigenvalues()[0])
        # Write the output
        output_files[branch_index].add_row(rvalue,sol[0],Re_ev)
        