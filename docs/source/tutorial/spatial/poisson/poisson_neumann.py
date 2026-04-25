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
from poisson import PoissonEquation,weak  # Load the Poisson equation for the previous class


# We create a new class, which adds the boundary integration term -<j,v>
class PoissonNeumannCondition(InterfaceEquations):
    # Makes sure that we can only use it as boundaries for a Poisson equation
    required_parent_type = PoissonEquation

    def __init__(self,name,flux):
        super(PoissonNeumannCondition, self).__init__()
        self.name=name # store the variable name and the flux
        self.flux=flux

    def define_residuals(self):
        u,v=var_and_test(self.name) # Get the test function by the name
        self.add_residual(-weak(self.flux,v)) # and add it to the residual
        # weak(j,v) is now <j,v>, i.e. a boundary integral

class NeumannPoissonProblem(Problem):
    def define_problem(self):
        mesh = LineMesh(minimum=-1, size=2, N=100)
        self.add_mesh(mesh)

        # Alternative way to assemble the system by restricting directly
        # Poisson equation and output on bulk domain
        equations = (PoissonEquation(source=1)+TextFileOutput()) @ "domain"
        equations += DirichletBC(u=0) @ "domain/left" # Dirichlet BC on domain/left
        equations += PoissonNeumannCondition("u",-1.5) @ "domain/right" # Neumann BC on domain/right
        self.add_equations(equations)


if __name__ == "__main__":
    with NeumannPoissonProblem() as problem:
        problem.solve()  # Solve the problem
        problem.output()  # Write output
