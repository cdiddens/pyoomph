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


# Import all PoissonEquation, Neumann and Robin condition as before
from poisson_robin_via_neumann import *
from pyoomph.expressions import * # Import vector


class PoissonProblem2d(Problem):
    def define_problem(self):
        # Create a 2d mesh, 10x10 elements, spanning from (0,0) to (1,1)
        mesh=RectangularQuadMesh(size=[1,1],lower_left=[0,0],N=10)
        self.add_mesh(mesh)

        # Assemble the system, bulk, then boundaries
        x=var("coordinate")-vector([0.5,0.5]) # position vector shifted by 0.5,0.5
        peak_source=100*exp(-100*dot(x,x))
        equations=PoissonEquation(source=peak_source)
        equations += DirichletBC(u=0) @ "left"
        equations += DirichletBC(u=0) @ "right"
        equations += PoissonNeumannCondition("u",2) @ "bottom"
        equations += PoissonRobinCondition("u", 1,1,-1) @ "top"
        # Also add an output. This VTU output can be viewed with Paraview
        equations += MeshFileOutput()

        self.add_equations(equations@"domain")

if __name__=="__main__":
    with PoissonProblem2d() as problem:
        problem.solve()
        problem.output()
