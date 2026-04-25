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
    # Simple problems can be assembled without a specific class
    problem=Problem()
    problem+=PitchForkNormalForm(r=1,sign=-1)@"pitchfork"
    # Find the solutions by deflation
    solutions=[]
    for sol in problem.iterate_over_multiple_solutions_by_deflation(deflation_alpha=0.1,deflation_p=2,perturbation_amplitude=0.1,num_random_tries=2):
        solutions.append(sol)
    print("Found solutions at r=1 are x = ",solutions)
        