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
from pyoomph.expressions import weak,dot,grad
from poisson_neumann import PoissonEquation


# Inherit from the normal InterfaceEquations
class PoissonRobinCondition(InterfaceEquations):
    def __init__(self,name,alpha,beta,g):
        super(PoissonRobinCondition, self).__init__()
        self.name=name # Store all passed values
        self.alpha=alpha
        self.beta=beta
        self.g=g

    def define_fields(self):
        # Define a Lagrange multiplier (field) at the interface with some unique name
        self.define_scalar_field("_lagr_robin_"+self.name,"C2")

    def define_residuals(self):
        l,ltest=var_and_test("_lagr_robin_"+self.name) # get the Lagrange multiplier
        u,utest=var_and_test(self.name) # the value of u on the interface
        # For the gradient grad(u), we need the function u inside the domain as well to calculate the gradient there
        # This is done by changing the domain to the parent domain, i.e. the domain where this InterfaceEquation is attached to
        ubulk,ubulk_test=var_and_test(self.name,domain=self.get_parent_domain())
        n=self.get_normal() # Normal to calculate dot(grad(u),n)
        condition=self.alpha*u+self.beta*dot(grad(ubulk),n)-self.g # The condition to enforce
        self.add_residual(weak(condition,ltest)+weak(l,utest)) # Lagrange multiplier pair to enforce it



class RobinPoissonProblem(Problem):
    def define_problem(self):
        mesh = LineMesh(minimum=-1, size=2, N=100)
        self.add_mesh(mesh)
        equations = PoissonEquation(source=1)
        equations+=TextFileOutput()
        equations += PoissonRobinCondition("u",0,0.5,1) @ "left"
        equations += PoissonRobinCondition("u",-1,0,-1) @ "right"
        self.add_equations(equations@ "domain")


if __name__ == "__main__":
    with RobinPoissonProblem() as problem:
        problem.solve()  # Solve the problem
        problem.output()  # Write output
