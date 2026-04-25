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

class TranscriticalNormalForm(ODEEquations):
    def __init__(self,r):
        super(TranscriticalNormalForm, self).__init__()
        self.r=r

    def define_fields(self):
        self.define_ode_variable("x")

    def define_residuals(self):
        x,x_test=var_and_test("x") #Shortcut to get var("x") and testfunction("x")
        self.add_residual((partial_t(x)-self.r*x+x**2)*x_test)


class TranscriticalProblem(Problem):
    def __init__(self):
        super(TranscriticalProblem, self).__init__()
        # Bifuraction parameter with default value
        self.r=self.define_global_parameter(r=1) 
        self.x0=1

    def define_problem(self):
        eq=TranscriticalNormalForm(r=self.r) #Pass the symbolic 'r' here
        eq+=InitialCondition(x=self.x0)
        eq+=ODEFileOutput()
        self.add_equations(eq@"transcritical")

if __name__=="__main__":
    with TranscriticalProblem() as problem:

        problem.r.value=1 # You can set the value here
        problem.x0=0.001 # Start slightly above the unstable solution x=0
        problem.run(endtime=20,numouts=100) # And let it evolve towards the stable branch x=r

        problem.r.value=-1 # Change the parameter value, x=1 is now not a stationary solution anymore
        problem.run(endtime=40, numouts=100)  # And let it evolve towards the now stable branch x=0
