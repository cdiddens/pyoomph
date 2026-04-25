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
from pyoomph.expressions.units import * # Import units as e.g. meter, second, etc.


class StokesEquations(Equations):
	# Passing the viscosity 
	def __init__(self,mu):
		super(StokesEquations, self).__init__()
		self.mu=mu # Store viscosity
		
	def define_fields(self):
		
		X=scale_factor("spatial") # spatial scale (will be supplied by the Problem class)
		U=scale_factor("velocity")
		P=scale_factor("pressure")
		mu=self.mu
		
		#Taylor-Hood pair, with testscale we can set the definition of the test scales V and Q
		self.define_vector_field("velocity","C2",testscale=X**2/(mu*U)) 
		self.define_scalar_field("pressure","C1",testscale=X/U)
		
	def define_residuals(self):
		# Fields and test functions are dimensional here!
		u,v=var_and_test("velocity") 
		p,q=var_and_test("pressure")
		stress=-p*identity_matrix()+2*self.mu*sym(grad(u))
		self.add_residual(weak(stress,grad(v)) + weak(div(u),q)) 
		
		
class DimStokesProblem(Problem):
	def __init__(self):
		super(DimStokesProblem, self).__init__()
		# we are now using units for the viscosity
		self.mu=1*milli*pascal*second
		self.boxsize=1*milli*meter # the size of the box 
		self.imposed_traction=1*pascal # and the imposed traction on the left

		
	def define_problem(self):
		# setting the spatial scale X by the boxsize and the pressure scale P by the imposed traction
		self.set_scaling(spatial=self.boxsize,pressure=self.imposed_traction)
		# the velocity scale is now calculated based on these scales. scale_factor will expand to P and X, respectively
		self.set_scaling(velocity=scale_factor("pressure")*scale_factor("spatial")/self.mu)
		# alternatively, you can just set directly
		# self.set_scaling(velocity=self.imposed_traction*self.boxsize/self.mu)
		
		self.add_mesh(RectangularQuadMesh(size=self.boxsize)) # we have to tell the mesh that it has a dimensional size now
		eqs=StokesEquations(self.mu) # passing the dimensional viscosity to the Stokes equations
		eqs+=MeshFileOutput() 
		
		# A traction is just the Neumann term
		eqs+=NeumannBC(velocity_x=-self.imposed_traction)@"left"
		# zero y velocity at left and right
		eqs+=DirichletBC(velocity_y=0)@"left"
		eqs+=DirichletBC(velocity_y=0)@"right"
		# No slip conditions at top and bottom
		eqs+=DirichletBC(velocity_x=0,velocity_y=0)@"bottom"
		eqs+=DirichletBC(velocity_x=0,velocity_y=0)@"top"

		
		# Adding this to the default domain name "domain" of the RectangularQuadMesh above
		self.add_equations(eqs@"domain")
	
		
if __name__ == "__main__":		
	# Create a Stokes problem with viscosity 1, quadratic velocity basis functions and linear pressure basis functions
	with DimStokesProblem() as problem: 
		problem.solve() # solve and output
		problem.output()
	
		
