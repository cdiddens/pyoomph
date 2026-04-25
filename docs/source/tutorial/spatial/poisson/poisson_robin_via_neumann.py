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
from poisson_neumann import PoissonEquation, PoissonNeumannCondition


# The Robin condition is just inherited from the Neumann condition
class PoissonRobinCondition(PoissonNeumannCondition):
    def __init__(self,name,alpha,beta,g):
        u=var(name) # Get the variable itself
        flux=1/beta*(g-alpha*u) # Calculate the Neumann flux term to impose
        super(PoissonRobinCondition, self).__init__(name,flux) # and pass it to the Neumann class


class RobinPoissonProblem(Problem):
    def define_problem(self):
        mesh = LineMesh(minimum=-1, size=2, N=100)
        self.add_mesh(mesh)
        equations = PoissonEquation(source=1)
        equations+=TextFileOutput()
        equations += PoissonRobinCondition("u",1,0.5,1) @ "left"
        equations += PoissonRobinCondition("u",-1,2,-1) @ "right"
        self.add_equations(equations@ "domain")


if __name__ == "__main__":
    with RobinPoissonProblem() as problem:
        problem.solve()  # Solve the problem
        problem.output()  # Write output
