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
from pyoomph.expressions import *  # Import grad & weak


class PoissonEquation(Equations):
    def __init__(self, *, name="u", space="C2", source=0):
        super(PoissonEquation, self).__init__()
        self.name = name  # store the variable name
        self.space = space  # the finite element space
        self.source = source  # and the source function g

    def define_fields(self):
        self.define_scalar_field(self.name, self.space)  # define the unknown scalar field

    def define_residuals(self):
        u, v = var_and_test(self.name)  # get the unknown field and the corresponding test function
        # weak formulation in residual form: (grad(u),grad(v))-(g,v)
        residual = weak(grad(u), grad(v)) - weak(self.source, v)
        self.add_residual(residual)  # add it to the residual


class PoissonProblem(Problem):
    def define_problem(self):
        mesh = LineMesh(minimum=-1, size=2, N=100)  # Line mesh from [-1:1] with 100 elements
        # Add the mesh (default name is "domain" with boundaries "left" and "right")
        self.add_mesh(mesh)

        # Assemble the system
        equations = PoissonEquation(source=1)  # create a Poisson equation with source g=1
        equations += DirichletBC(u=0) @ "left"  # Dirichlet condition u=0 on the left boundary
        equations += DirichletBC(u=0) @ "right"  # and u=0 on the right boundary
        equations += TextFileOutput()  # Add a simple text file output
        self.add_equations(equations @ "domain")  # Add the equation system on the domain named "domain"


if __name__ == "__main__":
    with PoissonProblem() as problem:
        problem.solve()  # Solve the problem
        problem.output()  # Write output
