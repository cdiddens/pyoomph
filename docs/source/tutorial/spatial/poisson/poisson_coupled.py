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
from poisson import PoissonEquation  # Load the Poisson equation for the previous class


class CoupledPoissonProblem(Problem):
    def define_problem(self):
        mesh = LineMesh(minimum=-1, size=2, N=100)
        self.add_mesh(mesh)

        u, w = var(["u", "w"])  # Bind the variables to use them mutually as sources
        # Create two instances of Poisson equations with different names and coupled sources
        equations = PoissonEquation(name="u", source=w) + PoissonEquation(name="w", source=-10 * u)
        equations += DirichletBC(u=0, w=1) @ "left"  # Dirichlet conditions u=0, w=1 on the left boundary
        equations += DirichletBC(u=0, w=-1) @ "right"  # and u=0, w=-1 on the right boundary
        equations += TextFileOutput()
        self.add_equations(equations @ "domain")


if __name__ == "__main__":
    with CoupledPoissonProblem() as problem:
        problem.solve()  # Solve the problem
        problem.output()  # Write output
