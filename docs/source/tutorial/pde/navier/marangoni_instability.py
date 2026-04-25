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

# Dimensional problem
from pyoomph.expressions.units import *

class MarangoniProblem(Problem):
	def __init__(self):
		super(MarangoniProblem,self).__init__()
		self.W,self.H=1*milli*meter, 0.25*milli*meter # Size of the box
		self.rho,self.mu=1000*kilogram/meter**3, 1*milli*pascal*second  # density and viscosity
		self.D=1e-9*meter**2/second # diffusivity
		self.Nx=10 # elmenents in x-direction
		self.max_refinement_level=3 # max. 4 times refining
		self.dsigma_dc=-0.1*milli*newton/meter
		
	def define_problem(self):
		self.set_scaling(spatial=self.W,temporal=0.1*second)
		self.set_scaling(velocity=scale_factor("spatial")/scale_factor("temporal"))
		self.set_scaling(pressure=1*pascal)						
		# add the mesh		
		self.add_mesh(RectangularQuadMesh(size=[self.W,self.H],N=[self.Nx,int(self.Nx*self.H/self.W)]))
		eqs=MeshFileOutput() # output

		eqs+=AdvectionDiffusionEquations(fieldnames="c",wind=var("velocity"),diffusivity=self.D,space="C1")
		eqs+=NavierStokesEquations(dynamic_viscosity=self.mu,mass_density=self.rho)
		# Initial condition
		xrel,yrel=var("coordinate_x")/self.W,var("coordinate_y")/self.H
		eqs+=InitialCondition(c=yrel*(1+0.01*cos(2*pi*xrel)+0.001*sin(4*pi*xrel)))
		# Refinements based on c and velocity
		eqs+=SpatialErrorEstimator(c=1,velocity=1)
		# Adding no-slip conditions
		for wall in ["left","right","bottom"]:
			eqs+=DirichletBC(velocity_x=0,velocity_y=0)@wall

		# "Free" surface: fixed y-velocity, Marangoni force
		sigma=self.dsigma_dc*var("c")		# Surface tension
		eqs+=(DirichletBC(velocity_y=0)+NeumannBC(velocity=-grad(sigma)))@"top" 
		# Fix one pressure degree
		eqs+=DirichletBC(pressure=0)@"bottom/left"
		self.add_equations(eqs@"domain")
		

if __name__=="__main__":
	with MarangoniProblem() as problem:
		# problem.dsigma_dc=0.1*milli*newton/meter
		problem.run(1*second,outstep=True,startstep=0.01*second,maxstep=0.1*second,spatial_adapt=1,temporal_error=1)
