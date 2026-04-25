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

class StokesEquations(Equations):
	# Passing the viscosity and the basis function space ("C1" or "C2") for velocity and pressure
	def __init__(self,mu,uspace,pspace):
		super(StokesEquations, self).__init__()
		self.mu=mu # Store viscosity and the selected spaces
		self.uspace=uspace
		self.pspace=pspace
		
	def define_fields(self):
		self.define_vector_field("velocity",self.uspace) # define a vector field called "velocity" on space uspace
		self.define_scalar_field("pressure",self.pspace) # and a scalar field "pressure"
		
	def define_residuals(self):
		u,v=var_and_test("velocity") # get the fields and the corresponding test functions
		p,q=var_and_test("pressure")
		# stress tensor, sym(A) applied on a matrix gives 1/2*(A+A^t), so this is -p+mu*(grad(u)+grad(u)^t)
		stress=-p*identity_matrix()+2*self.mu*sym(grad(u))
		self.add_residual(weak(stress,grad(v)) + weak(div(u),q)) # weak form of Stokes flow
		
		
class StokesSpaceTestProblem(Problem):
	# Taking viscosity and basis function spaces for the velocity and pressure
	def __init__(self,mu,uspace,pspace):
		super(StokesSpaceTestProblem, self).__init__()
		self.mu=mu
		self.uspace=uspace
		self.pspace=pspace
		
	def define_problem(self):
		self.add_mesh(RectangularQuadMesh()) # add a rectangular quad mesh
		eqs=StokesEquations(self.mu,self.uspace,self.pspace) # Stokes equation using the viscosity and the spaces
		eqs+=MeshFileOutput() # Add output to write PVD/VTU files to be viewed in paraview
		
		# Inflow: Parabolic u_x=y*(1-y), u_y=0
		y=var("coordinate_y")
		u_x_inflow=y*(1-y)
		# Components of vector quantities can be accessed with the suffix "_x" or "_y" (or "_z" in 3d)
		eqs+=DirichletBC(velocity_x=u_x_inflow,velocity_y=0)@"left"
		
		# Outflow, u_y=0, no Dirichlet on u_x, i.e. stress free outlet
		eqs+=DirichletBC(velocity_y=0)@"right"
		
		# No slip conditions at top and bottom
		eqs+=DirichletBC(velocity_x=0,velocity_y=0)@"bottom"
		eqs+=DirichletBC(velocity_x=0,velocity_y=0)@"top"
		
		# Adding this to the default domain name "domain" of the RectangularQuadMesh above
		self.add_equations(eqs@"domain")
	
		
if __name__ == "__main__":		
	# Create a Stokes problem with viscosity 1, quadratic velocity basis functions and linear pressure basis functions
	with StokesSpaceTestProblem(1.0,"C2","C1") as problem: 
		problem.solve() # solve and output
		problem.output()
	
		
