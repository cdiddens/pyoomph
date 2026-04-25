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

class FoldNormalForm(ODEEquations):
    def __init__(self,r):
        super(FoldNormalForm, self).__init__()
        self.r=r

    def define_fields(self):
        self.define_ode_variable("x")

    def define_residuals(self):
        x,x_test=var_and_test("x") 
        self.add_residual((partial_t(x)-self.r+x**2)*x_test)


class FoldProblem(Problem):
    def __init__(self):
        super(FoldProblem, self).__init__()
        # bifurcation parameter
        self.r=self.define_global_parameter(r=1) 
        self.x0=1

    def define_problem(self):
        eq=FoldNormalForm(r=self.r) #Pass the paramter 'r' here
        eq+=InitialCondition(x=self.x0)
        # Instead of having a file with time as first column, we want to have the parameter value r
        eq+=ODEFileOutput(first_column=self.r)
        
        self.add_equations(eq@"fold")

if __name__=="__main__":
    with FoldProblem() as problem:
        while True:
            problem.solve()
            problem.output_at_increased_time()
            problem.r.value-=0.02

