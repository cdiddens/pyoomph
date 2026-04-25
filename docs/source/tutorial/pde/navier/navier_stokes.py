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
from pyoomph.expressions import *


class NavierStokesEquations(Equations):
	def __init__(self,rho,mu):
		super(NavierStokesEquations, self).__init__()
		self.rho, self.mu= rho,mu # Store viscosity and density

	def define_fields(self):
		self.define_vector_field("velocity","C2") # Taylor-Hood pair
		self.define_scalar_field("pressure","C1") 
		
	def define_residuals(self):
		u,v=var_and_test("velocity") # get the fields and the corresponding test functions
		p,q=var_and_test("pressure")
		stress=-p*identity_matrix()+2*self.mu*sym(grad(u)) # see Stokes equation
		inertia=self.rho*material_derivative(u,u) # lhs of the Navier-Stokes eq. rho*Du/dt
		self.add_residual(weak(inertia,v)+ weak(stress,grad(v)) + weak(div(u),q)) 


class WomersleyFlowProblem(Problem):
	def __init__(self):
		super(WomersleyFlowProblem, self).__init__()
		self.rho,self.mu=10,1 # density and viscosity
		self.omega,self.delta_p=10,10 # frequency and pressure amplitude
		self.L,self.R=1,1 # size of the pipe
		# Corresponding to a Womersley number of 10
		self.max_refinement_level=3 # refine due to the velocity profile
		
	def define_problem(self):
		self.set_coordinate_system("axisymmetric") # Pipe: Axisymmetric
		Nr=4 # number of radial mesh elements
		self.add_mesh(RectangularQuadMesh(N=[Nr,int(self.L/self.R*Nr)],size=[self.R,self.L]))
		
		eqs=NavierStokesEquations(self.rho,self.mu)
		eqs+=MeshFileOutput()
		eqs+=DirichletBC(velocity_x=0)@"left" # no r-velocity at the axis
		eqs+=DirichletBC(velocity_x=0,velocity_y=0)@"right" # no-slip at the wall
		eqs+=DirichletBC(velocity_x=0)@"top" # no r-velocity at in and outflow
		eqs+=DirichletBC(velocity_x=0)@"bottom"														
		# impose oscillating pressure
		eqs+=NeumannBC(velocity_y=-self.delta_p*cos(self.omega*var("time")))@"bottom"
		eqs+=SpatialErrorEstimator(velocity=1) # Refine where necessary
		# eqs+=DirichletBC(velocity_x=0) # We can also deactivate the entire x-velocity in this problem
		self.add_equations(eqs@"domain") # adding the equation
		
if __name__=="__main__":
	with WomersleyFlowProblem() as problem:
		problem.run(1,outstep=True,startstep=0.01,spatial_adapt=1)
