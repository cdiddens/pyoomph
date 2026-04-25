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
from pyoomph.equations.multi_component import *
from pyoomph.materials import *
import pyoomph.materials.default_materials # Alternatively, define the materials by hand

class RayleighTaylorProblem(Problem):
    def __init__(self):
        super(RayleighTaylorProblem, self).__init__()
        self.box_height,self.box_width=1*milli*meter,0.25*milli*meter # box size
        self.Nx=5 # Num elements in x direction
        self.mixture=Mixture(get_pure_liquid("water")+0.5*get_pure_liquid("glycerol")) # Default mixture
        self.temperature=20*celsius # Temperature : Required for some properties
        self.gravity=9.81*vector(0,-1)*meter/second**2

    def define_problem(self):
        # Mesh
        self.add_mesh(RectangularQuadMesh(size=[self.box_width,self.box_height],N=[self.Nx,int(self.Nx*self.box_height/self.box_width)]))
        # Spatial and temporal scales must be set by hand
        self.set_scaling(spatial=self.box_width,temporal=1*second)
        # Set remaining scales by the liquid properties
        self.mixture.set_reference_scaling_to_problem(self,temperature=self.temperature)
        # define global constants "temperature" and "absolute_pressure". It might be required by the fluid properties
        self.define_named_var(temperature=self.temperature,absolute_pressure=1*atm)

        eqs=MeshFileOutput()
        eqs+=CompositionFlowEquations(self.mixture,spatial_errors=True,gravity=self.gravity)
        for side in ["left","right","bottom"]:
            eqs+=DirichletBC(velocity_x=0,velocity_y=True)@side
            # Top side must be open: The density is not constant and hence we require in/outflow somewhere!

        self.add_equations(eqs@"domain")


with RayleighTaylorProblem() as problem:
    # Let the user select any mixture
    problem.mixture = Mixture(get_pure_liquid("water") + 0.5 * get_pure_liquid("glycerol"))

    # And also formulate the initial condition
    x,y=var(["coordinate_x","coordinate_y"])
    xrel, yrel = var("coordinate_x") / problem.box_width, var("coordinate_y") / problem.box_height - 0.5
    wg_init = 0.5*(1+tanh(100 * (yrel - 0.0125 * cos(2 * pi * xrel))))
    problem.additional_equations+=InitialCondition(massfrac_glycerol=wg_init)@"domain"

    problem.max_refinement_level=4
    problem.run(10*second,startstep=0.1*second,maxstep=0.5*second,outstep=True,spatial_adapt=1,temporal_error=1)





