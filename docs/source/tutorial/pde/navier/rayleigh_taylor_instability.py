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
# Use the pre-defined equations for Navier-Stokes and advection-diffusion
from pyoomph.equations.navier_stokes import * 
from pyoomph.equations.advection_diffusion import *


class RayleighTaylorProblem(Problem):
	def __init__(self):
		super(RayleighTaylorProblem,self).__init__()
		self.W,self.H=0.25, 1 # Size of the box
		self.rho,self.mu=0.01, 1 # density and viscosity
		self.Nx=4 # elmenents in x-direction
		self.max_refinement_level=4 # max. 4 times refining
		
	def define_problem(self):
		# add the mesh
		self.add_mesh(RectangularQuadMesh(size=[self.W,self.H],N=[self.Nx,int(self.Nx*self.H/self.W)]))
		eqs=MeshFileOutput() # output
		bulkforce=100*var("c")*vector(0,-1) # bulkforce: depends on the composition c
		# Advection diffusion equation: Advected by the velocity
		eqs+=AdvectionDiffusionEquations(fieldnames="c",wind=var("velocity"),diffusivity=0.0001,space="C1")
		# Navier-Stokes with the c-dependent bulk formce
		eqs+=NavierStokesEquations(bulkforce=bulkforce,dynamic_viscosity=self.mu,mass_density=self.rho)
		# Initial condition
		xrel,yrel=var("coordinate_x")/self.W,var("coordinate_y")/self.H-0.5
		eqs+=InitialCondition(c=tanh(100*(yrel-0.0125*cos(2*pi*xrel))))
		# Refinements based on c and velocity
		eqs+=SpatialErrorEstimator(c=1,velocity=1)
		# Adding no-slip conditions
		for wall in ["left","right","top","bottom"]:
			eqs+=DirichletBC(velocity_x=0,velocity_y=0)@wall
		# Fix one pressure degree
		eqs+=DirichletBC(pressure=0)@"bottom/left"
		self.add_equations(eqs@"domain")
		

if __name__=="__main__":
	with RayleighTaylorProblem() as problem:
		problem.run(10,numouts=50,spatial_adapt=1)
