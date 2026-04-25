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
from pyoomph.equations.navier_stokes import *
from pyoomph.equations.ALE import *

# Create a rectangular mesh with an interface in between
class RectangularQuadMeshWithInterface(RectangularQuadMesh):
	def __init__(self,N,size,interface_name,interface_y_offset):
		super().__init__(N=N,size=size)
		self.interface_name=interface_name # Name of the interface
		self.interface_y_offset=interface_y_offset # Offset (in elements) of the interface in y direction
  
	def define_geometry(self):
		super().define_geometry() # Define the normal rectangular mesh
		# Y position of the interface
		y=self.interface_y_offset*self.size[1]/self.N[1]+self.lower_left[1]
		for ix in range(self.N[0]):
			# Find the nodes of the interface
			nl = self.add_node_unique(ix * self.size[0] / self.N[0] + self.lower_left[0], y)
			nr = self.add_node_unique((ix+1) * self.size[0] / self.N[0] + self.lower_left[0], y)
   			# And add them to the internal interface boundary
			self.add_nodes_to_boundary(self.interface_name, [nl, nr]) 
		# Also mark it as an internal boundary
		self.set_boundary_as_interior(self.interface_name)
  

class TwoLayerFlowProblem(Problem):
	def __init__(self):
		super(TwoLayerFlowProblem, self).__init__()
		self.W=1
		self.H1=0.1
		self.H2=0.1
		self.Nx,self.Ny1,self.Ny2=100,10,10
		

	def define_problem(self):		
		self.add_mesh(RectangularQuadMeshWithInterface([self.Nx,self.Ny1+self.Ny2], [self.W, self.H1+self.H2], "interface", self.Ny1)) # Create the mesh

		# Only one domain now, so we just assemble a single set of equations
		eqs=LaplaceSmoothedMesh()
		eqs+=MeshFileOutput()
		eqs+=DirichletBC(mesh_x=True)
		eqs += DirichletBC(velocity_x=0) @ "left"  # no in/outflow at the sides
		eqs += DirichletBC(velocity_x=0) @ "right"

		# Add a fixed, elemental domain marker
		eqs+=ScalarField("domain_marker",space="D0") # constant per element
		domain_marker_expr=heaviside(var("lagrangian_y")-self.H1) # Expression to mark the domains (based on the Lagrangian y-position)
		eqs+=DirichletBC(domain_marker=domain_marker_expr) # Fix the domain marker everywhere
	
		# Blend the viscosity with the domain marker
		mu1=1
		mu2=0.01
		mu=mu1*(1-var("domain_marker"))+var("domain_marker")*mu2

		# Use the Navier-Stokes equations with CR elements and the elementally varying viscosity
		eqs += NavierStokesEquations(mass_density=0.01, dynamic_viscosity=mu,mode="CR")  # NS equations with CR elements
		
		# no slip at top and bottom
		eqs += DirichletBC(velocity_x=0, velocity_y=0, mesh_y=0) @ "bottom"  # no slip at bottom and fix the mesh there
		eqs += DirichletBC(velocity_x=0, velocity_y=0, mesh_y=self.H1+self.H2) @ "top"  # no slip at bottom and fix the mesh there
		eqs += AverageConstraint(pressure=0) # Fix the average pressure to 0

		# Free surface, mesh connection and velocity connection
		ieqs= NavierStokesFreeSurface(surface_tension=1)
		ieqs+=InteriorBoundaryOrientation(var("normal_y")) # Important. Only create interface elements with an upwards normal
		eqs+=ieqs@ "interface"  
  		
		# Deform the initial mesh
		X, Y = var(["lagrangian_x", "lagrangian_y"])
		deform_bottom=Y * (1 + 0.25 * cos(2 * pi * X))
		deform_top=Y+ (self.H1+self.H2-Y)*(0.25 * cos(2 * pi * X))
		deform=deform_bottom*(1-domain_marker_expr)+deform_top*domain_marker_expr
		eqs += InitialCondition(mesh_y=deform)  # small height with a modulation		
  
		self.add_equations(eqs @ "domain") 


if __name__=="__main__":
	with TwoLayerFlowProblem() as problem:
		problem.run(50,outstep=True,startstep=0.25)
