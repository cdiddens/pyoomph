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


class ConvectionDiffusionEquation(Equations):
	def __init__(self,u,D,advection_in_partial_integration,space="C2"):
		super(ConvectionDiffusionEquation, self).__init__()
		self.u=u # advection velocity
		self.D=D # diffusivity		
		self.space=space # space of the field c
		self.advection_in_partial_integration=advection_in_partial_integration # Which weak form to use
		
	def define_fields(self):
		self.define_scalar_field("c",self.space) # The scalar field to advect
		
	def define_residuals(self):
		c,phi=var_and_test("c")
		# Advection either intergrated by parts or not
		advection=-weak(self.u*c,grad(phi)) if self.advection_in_partial_integration else weak(div(self.u*c),phi)
		# TPZ or MPT time stepping can be of advantage compared to BDF2
		self.add_residual(time_scheme("TPZ",weak(partial_t(c),phi)+advection+weak(self.D*grad(c),grad(phi))))

		
		
class ConvectionDiffusionProblem(Problem):
	def __init__(self):
		super(ConvectionDiffusionProblem, self).__init__()
		self.u=2*pi*vector([-var("coordinate_y"),var("coordinate_x")]) # Circular flow, one rotation at t=1
		self.D=0.001 # diffusivity
		self.L=1 # size of the mesh
		self.N=4 # number of elements of the coarsest mesh in each direction		
		self.max_refinement_level=5 # max refinement level
		self.advection_in_partial_integration=False # which weak form to choose
		
	def define_problem(self):
		self.add_mesh(RectangularQuadMesh(lower_left=-self.L/2,size=self.L,N=self.N))
		
		eqs=ConvectionDiffusionEquation(self.u,self.D,self.advection_in_partial_integration)
		eqs+=MeshFileOutput() # output
		
		# use a bump as initial condition
		bump_pos=vector([-self.L/5,0]) # center pos of the bump
		bump_width=0.005*self.L # width of the bump
		bump_amplitude=1 # amplitude of the bump
		xdiff=var("coordinate")-bump_pos # difference between coordinate and bump center
		bump=bump_amplitude*exp(-dot(xdiff,xdiff)/bump_width) # Gaussian bump
		eqs+=InitialCondition(c=bump)

		# Set the boundaries to 0
		for b in ["top","left","right","bottom"]:
			eqs+=DirichletBC(c=0)@b

		# Errors: We evaluate the jumps in the gradients of c at the element boundaries, i.e. when crossing to the next element
		# this is not only done at the current time step, but also on the previous one
		error_fluxes=[grad(var("c")),evaluate_in_past(grad(var("c")))]
		eqs+=SpatialErrorEstimator(*error_fluxes)
		
		self.add_equations(eqs@"domain") # adding the equation
		
if __name__=="__main__":
	with ConvectionDiffusionProblem() as problem:
		problem.advection_in_partial_integration=True # Can also set it to false
		problem.D=0.0001 # diffusivity		
		problem.run(1,outstep=0.01,maxstep=0.0025,spatial_adapt=1,temporal_error=1)
