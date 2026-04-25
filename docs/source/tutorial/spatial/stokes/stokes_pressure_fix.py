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
from pyoomph.equations.navier_stokes import *


class StokesProblemWithPressureFixation(Problem):
    def __init__(self):
        super(StokesProblemWithPressureFixation, self).__init__()
        self.mode="CR" # use Crouzeix-Raviart ("CR"), alternatively "TH" for Taylor-Hood

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh())
        eqs=MeshFileOutput()

        x, y = var(["coordinate_x", "coordinate_y"])
        # Create a Stokes equation with bulk force
        # and combine it with the appropriate pressure fixation
        eqs+=StokesEquations(dynamic_viscosity=1,bulkforce=vector(-y,x),mode=self.mode).with_pressure_fixation()
        for b in ["top","left","bottom","right"]: # No in/outflow allowed on all sides, one pressure must be fixed
            eqs+=DirichletBC(velocity_x=0,velocity_y=0)@b

        self.add_equations(eqs@"domain")

with StokesProblemWithPressureFixation() as problem:
    problem.solve()
    problem.output()