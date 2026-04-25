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
from pyoomph.equations.poisson import *  # use the pre-defined Poisson equation


class HelicalLineMesh(MeshTemplate):
    def __init__(self, N=100, radius=1, length=5, windings=4, domain_name="helix"):
        super(HelicalLineMesh, self).__init__()
        self.N = N
        self.radius = radius
        self.length = length
        self.windings = windings
        self.domain_name = domain_name

    def define_geometry(self):
        domain = self.new_domain(self.domain_name, 3)  # Domain, but with 3d nodes

        # function to get the node based on a parameter l from [0:1]
        def node_at_parameter(l):
            x = self.radius * cos(2 * pi * self.windings * l)
            y = self.radius * sin(2 * pi * self.windings * l)
            z = self.length * l
            return self.add_node_unique(x, y, z)

        # loop to generate the elements
        for i in range(self.N):
            n0 = node_at_parameter(i / self.N)  # constructing nodes
            n1 = node_at_parameter((i + 0.5) / self.N)
            n2 = node_at_parameter((i + 1) / self.N)
            domain.add_line_1d_C2(n0, n1, n2)  # add a second order line element
            if i == 0:  # Marking the start boundary:
                self.add_nodes_to_boundary("start", [n0])
            elif i == self.N - 1:  # Marking the end boundary:
                self.add_nodes_to_boundary("end", [n2])


class MeshTestProblem(Problem):
    def define_problem(self):
        self.add_mesh(HelicalLineMesh())
        eqs = MeshFileOutput()
        x, y, z = var(["coordinate_x", "coordinate_y", "coordinate_z"])
        source = x ** 2 + 5 * y * z + z
        eqs += PoissonEquation(name="u", source=source, space="C2")
        eqs += DirichletBC(u=0) @ "start"
        eqs += DirichletBC(u=0) @ "end"
        self.add_equations(eqs @ "helix")


if __name__ == "__main__":
    with MeshTestProblem() as problem:
        problem.solve(spatial_adapt=4)
        problem.output_at_increased_time()
