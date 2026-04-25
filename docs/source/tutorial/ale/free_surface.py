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

# use the pre-defined Navier-Stokes ( it will use partial_t(...,ALE="auto") )
from pyoomph.equations.navier_stokes import * 
# and the pre-defined LaplaceSmoothedMesh
from pyoomph.equations.ALE import * 

class KinematicBC(InterfaceEquations):
	def define_fields(self):
		self.define_scalar_field("_kin_bc","C2") # second order field lambda
		
	def define_residuals(self):
		n,u=var(["normal","velocity"])
		l,eta=var_and_test("_kin_bc") # Lagrange multiplier
		x,chi=var_and_test("mesh") # unknown mesh coordinates
		# Let the normal mesh velocity follow the normal fluid velocity
		self.add_residual(weak(dot(n,u-mesh_velocity()),eta)-weak(l,dot(n,chi)))

	def before_assigning_equations_postorder(self, mesh):
		# pin the Lagrange multiplier, when the mesh is locally entirely pinned
		self.pin_redundant_lagrange_multipliers(mesh, "_kin_bc", "mesh") 
		

class DynamicBC(InterfaceEquations):
	def __init__(self,sigma):
		super(DynamicBC,self).__init__()
		self.sigma=sigma
		
	def define_residuals(self):
		v=testfunction("velocity")
		self.add_residual(weak(self.sigma,div(v)))
		

# Shortcut to create both conditions simultaneously
def FreeSurface(sigma):
	return KinematicBC()+DynamicBC(sigma)
	
	
class SurfaceRelaxationProblem(Problem):	
	def define_problem(self):
		# Shallow 2d mesh
		self.add_mesh(RectangularQuadMesh(N=[80,4],size=[1,0.05]))
		eqs=NavierStokesEquations(mass_density=0.01,dynamic_viscosity=1) # equations
		eqs+=LaplaceSmoothedMesh() # Laplace smoothed mesh
		eqs+=DirichletBC(mesh_x=True) # We can fix all x-coordinates, since the problem is rather shallow
		eqs+=MeshFileOutput() # output	
		eqs+=DirichletBC(velocity_x=0,velocity_y=0,mesh_y=0)@"bottom" # no slip at bottom and fix the mesh there
		eqs+=DirichletBC(velocity_x=0)@"left" # no in/outflow at the sides
		eqs+=DirichletBC(velocity_x=0)@"right"
		eqs+=FreeSurface(sigma=1)@"top" # free surface at the top
		# Deform the initial mesh
		X,Y=var(["lagrangian_x","lagrangian_y"])
		eqs+=InitialCondition(mesh_y=Y*(1+0.25*cos(2*pi*X)))  # small height with a modulation
		self.add_equations(eqs@"domain") # adding it to the system

		
if __name__=="__main__":
	with SurfaceRelaxationProblem() as problem:
		problem.run(50,outstep=True,startstep=0.25)	
