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


from free_surface import * # Load our free surface implementation
from pyoomph.meshes.simplemeshes import CircularMesh # Import a curved mesh


class EquilibriumContactAngle(InterfaceEquations):

	required_parent_type=DynamicBC # Must be attached to an interface with a DynamicBC
	
	def __init__(self,N):
		super(EquilibriumContactAngle,self).__init__()
		self.N=N # equilibrium vector
		
	def define_residuals(self):
		# get sigma from the DynamicBC object of the interface
		sigma=self.get_parent_equations().sigma 
		# Contact line contribution
		v=testfunction("velocity")
		self.add_residual(-weak(sigma*self.N,v))

	
class DropletSpreadingProblem(Problem):
	def __init__(self):
		super(DropletSpreadingProblem,self).__init__()
		self.contact_angle=45*degree # equilibrium contact angle
		
	def define_problem(self):
		# hemi-circle mesh, i.e. initial contact angle of 90 degree, free interface "interface", symmetry axis "axis" and bottom interface "substrate"
		mesh=CircularMesh(radius=1,segments=["NE"],straight_interface_name={"center_to_north":"axis","center_to_east":"substrate"},outer_interface="interface")
		self.add_mesh(mesh)
		
		self.set_coordinate_system("axisymmetric") # axisymmetry

		eqs=NavierStokesEquations(mass_density=0.01,dynamic_viscosity=1) # flow
		eqs+=LaplaceSmoothedMesh() # Laplace smoothed mesh
		eqs+=RefineToLevel(4) # refine, since the CircularMesh is coarse by default
		eqs+=DirichletBC(mesh_x=0,velocity_x=0)@"axis" # fix mesh x-position, no flow through the axis
		eqs+=DirichletBC(mesh_y=0,velocity_y=0)@"substrate" # fix substrate at y=0, no flow through the substrate				
		# free surface at the interface, equilibrium contact angle at the contact with the substrate
		N=vector(cos(self.contact_angle),-sin(self.contact_angle))
		eqs+=(FreeSurface(sigma=1)+EquilibriumContactAngle(N)@"substrate")@"interface" 
		eqs+=MeshFileOutput() # output	
		
		self.add_equations(eqs@"domain") # adding it to the system

		
if __name__=="__main__":
	with DropletSpreadingProblem() as problem:
		problem.run(50,outstep=True,startstep=0.25)	
