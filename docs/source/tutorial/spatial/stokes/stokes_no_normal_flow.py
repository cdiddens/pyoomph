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


from stokes_dimensional import * # Import the dimensional Stokes equation from the previous section
from pyoomph.meshes.simplemeshes import CircularMesh # Import a curved mesh


class StokesFlowZeroNormalFlux(InterfaceEquations):
	required_parent_type = StokesEquations # Must be attached to an interface of a Stokes equation
	
	def define_fields(self):
		# Velocity space is C2, so we must create the Lagrange multipliers on the same space
		# Note how we set the scale and the testscale here: In both cases, we absorb the test scale or the scale of the velocity
		self.define_scalar_field("noflux_lambda","C2",scale=1/test_scale_factor("velocity"),testscale=1/scale_factor("velocity")) 
		
	def define_residuals(self):
		# Binding variables
		l,ltest=var_and_test("noflux_lambda")
		u,utest=var_and_test("velocity")
		n=var("normal")
		# Add the residual: The scales will cancel out: u~U, ltest~1/U and l~1/V, utest~V
		self.add_residual(weak(dot(u,n),ltest)+weak(l,dot(utest,n)))
		
	# This will be called before the equations are numbered. This is the last chance to apply any pinning (i.e. Dirichlet conditions)
	def before_assigning_equations_postorder(self, mesh):
		# If the velocity is entirely pinned at any node (e.g. no slip), we also have to set the Lagrange multiplier to zero
		# This can be done with the helper function: we set noflux_lambda=0 whenever "velocity" (i.e. "velocity_x" & "velocity_y) are pinned
		self.pin_redundant_lagrange_multipliers(mesh, "noflux_lambda", "velocity")
		

class StokesBulkForce(Equations):
	def __init__(self,force_density):
		super(StokesBulkForce, self).__init__()
		self.force_density=force_density
		
	def define_residuals(self):
		utest=testfunction("velocity")
		self.add_residual(-weak(self.force_density,utest))
		
		
class NoFluxStokesProblem(Problem):
	def __init__(self):
		super(NoFluxStokesProblem, self).__init__()
		self.mu=1*milli*pascal*second # dynamic viscosity
		self.radius=1*milli*meter # the radius of the circular mesh
		
	def define_problem(self):
		# Setting reasonable scales
		self.set_scaling(spatial=self.radius,velocity=1*milli*meter/second,pressure=1*pascal)

		# Changing to an axisymmetric coordinate system
		self.set_coordinate_system("axisymmetric")
		
		# Taking the north east segment of a circle as mesh, set the radius and rename the interfaces
		mesh=CircularMesh(radius=self.radius,segments=["NE"],straight_interface_name={"center_to_north":"axis","center_to_east":"bottom"},outer_interface="interface")
		self.add_mesh(mesh)
		
		eqs=StokesEquations(self.mu) # passing the dimensional viscosity to the Stokes equations
		eqs+=RefineToLevel(3) # Refine the mesh, which is otherwise too coarse
		eqs+=MeshFileOutput() # and output
				
		#Imposing gravity as bulk force
		rho=1000*kilogram/meter**3 # mass density
		g=9.81*meter/second**2 # gravity
		gdir=vector(0,-1)	# direction of the gravity
		eqs+=StokesBulkForce(rho*g*gdir)
		
		#adding some artificial bulk force as well
		f=1000*rho/second**2 * vector(-var("coordinate_y"),var("coordinate_x"))
		eqs+=StokesBulkForce(f)
				
		# No slip at substrate
		eqs+=DirichletBC(velocity_x=0,velocity_y=0)@"bottom"
		# No flow through the axis of symmtery
		eqs+=DirichletBC(velocity_x=0)@"axis"
		# Use our zero flux interface
		eqs+=StokesFlowZeroNormalFlux()@"interface"
				
		# Adding these equations to the default domain name "domain" of the CircularMesh above
		self.add_equations(eqs@"domain")
	
		
if __name__ == "__main__":		
	with NoFluxStokesProblem() as problem: 
		problem.solve() 
		problem.output()
	
		
