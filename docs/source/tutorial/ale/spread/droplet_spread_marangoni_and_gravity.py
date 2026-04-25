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


from pyoomph import * # main pyoomph
from pyoomph.expressions.units import * # units like meter, etc.

from pyoomph.equations.navier_stokes import * # Navier-Stokes equations, including free surface, contact angles
from pyoomph.equations.ALE import * # Moving mesh

from pyoomph.meshes.simplemeshes import CircularMesh # Mesh


class StationaryDropletProblem(Problem):
	def __init__(self):
		super(StationaryDropletProblem, self).__init__()
		self.volume=0.25*milli*liter
		self.rho=1000*kilogram/meter**3
		self.mu=1*milli*pascal*second
		self.sigma0=72*milli*newton/meter
		self.slip_length=1*micro*meter

		# Parameter to modify the gravity (in terms of g), initially 0
		self.param_gravity_factor = self.define_global_parameter(gravity_factor=0)
		# Parameter to dynamically change the contact angle at run time
		# Initially 90 deg coinciding with the initial mesh
		self.param_contact_angle=self.define_global_parameter(contact_angle=pi/2)		
		# Parameter to control the surface tension, sigma=sigma0*(1+sigma_gradient*(var("coordinate_x")/R0)**2), initially 0
		self.param_sigma_gradient = self.define_global_parameter(sigma_gradient=0)
		self.gravity=9.81*meter/second**2 # overall gravity when param_gravity_factor.value=1


	def define_problem(self):
		R0=square_root(3*self.volume/(4*pi)*2,3) # Radius of the initial hemi-sphere
		mesh=CircularMesh(radius=R0,segments=["NE"],outer_interface="interface",straight_interface_name={"center_to_north":"axis","center_to_east":"substrate"})
		self.add_mesh(mesh)

		# Find good scales to nondimensionalize the space, the time, velocity and pressure
		self.set_scaling(spatial=R0,temporal=1*second,velocity=scale_factor("spatial")/scale_factor("temporal"))
		self.set_scaling(pressure=self.sigma0/R0)

		self.set_coordinate_system("axisymmetric")

		eqs=MeshFileOutput()

		# Navier-Stokes with gravity
		g=self.gravity*self.param_gravity_factor*vector(0,-1) # We can change the influence of gravity by the parameter
		eqs+=NavierStokesEquations(mass_density=self.rho,dynamic_viscosity=self.mu,gravity=g)
		# Mesh motion and refinement of the coarse CircularMesh
		eqs+=PseudoElasticMesh()
		eqs+=RefineToLevel(self.initial_adaption_steps) # Refine slightly in the beginning to find the solution quickly


		# Dirichlet boundary conditions and slip length
		eqs+=DirichletBC(velocity_x=0,mesh_x=0)@"axis"
		eqs += DirichletBC(velocity_y=0, mesh_y=0) @ "substrate"
		eqs +=NavierStokesSlipLength(self.slip_length)@"substrate"

		# Free surface and contact angle. Both surface tension and contact angle depend on parameters
		sigma=self.sigma0*(1+self.param_sigma_gradient*(var("coordinate_x")/R0)**2)
		eqs+=NavierStokesFreeSurface(surface_tension=sigma)@"interface"
		eqs+=NavierStokesContactAngle(self.param_contact_angle,wall_normal=vector(0,1),wall_tangent=vector(-1,0))@"interface/substrate"

		# The volume is not conserved during stationary solving
		# So we must add a single global Lagrange multiplier that ensures the volume

		# Create the Lagrange multiplier "volume_lagrange", add an offset of -self.volume to the residual
		vol_constr_eqs=GlobalLagrangeMultiplier(volume_lagrange=-self.volume)
		vol_constr_eqs+=Scaling(volume_lagrange=scale_factor("pressure")) # Introduce the scale to nondimensionalize
		vol_constr_eqs += TestScaling(volume_lagrange=1/self.volume) # And also a test scale, which is just the inverse
		self.add_equations(vol_constr_eqs@"volume_constraint") # add it to an "ODE" domain called "volume_constraint"

		# bind the volume enforcing Lagrange multiplier
		vol_constr_lagr=var("volume_lagrange",domain="volume_constraint")
		# And add the dimensional integral 1*dx over the droplet to the residual of the Lagrange multiplier
		# In total, we then solve [integral_droplet 1*dx - self. volume] on the test space of the Lagrange multiplier
		# This is indeed the volume constraint
		eqs+=WeakContribution(1,testfunction(vol_constr_lagr),dimensional_dx=True)

		# Finally, the Lagrange multiplier must also yield feedback to the problem
		# It can be done easily by adding additional pressure whenever the droplet's volume does not match
		# We hence add a normal traction (pressure) proportional to the volume Lagrange multiplier
		eqs+=NeumannBC(velocity=vol_constr_lagr*var("normal"))@"interface"

		eqs+=SpatialErrorEstimator(velocity=1)

		self.add_equations(eqs@"domain")


if __name__=="__main__":
	with StationaryDropletProblem() as problem:

		problem.initial_adaption_steps=2 # adapt the initial mesh only a bit for fast calculation
		problem.max_refinement_level=7 # final adaption, very fine

		# Start without gravity, no Marangoni flow and at 90 deg contact angle
		problem.param_gravity_factor.value=0
		problem.param_sigma_gradient.value = 0
		problem.param_contact_angle.value = pi/2

		problem.solve() # This stationary solve is trivial
		problem.output() # Output first step

		# Ramp up gravity by arclength continuation
		problem.go_to_param(gravity_factor=1,startstep=0.2,final_adaptive_solve=False)
		problem.output_at_increased_time()

		# Ramp up the contact angle by arclength continuation
		problem.go_to_param(contact_angle=110*degree, startstep=5*degree,final_adaptive_solve=False)
		problem.output_at_increased_time()

		# Ramp up Marangoni flow by arclength continuation
		problem.go_to_param(sigma_gradient=0.001, startstep=0.001,final_adaptive_solve=True)
		problem.output_at_increased_time()

