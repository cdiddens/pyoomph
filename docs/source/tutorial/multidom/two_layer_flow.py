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

class EnforceContinuousVelocity(InterfaceEquations):
	def define_fields(self):
		self.define_vector_field("_couple_velo","C2")
		
	def define_residuals(self):
		l,ltest=var_and_test("_couple_velo")
		ui,uitest=var_and_test("velocity") # inner velocity at the interface
		uo,uotest=var_and_test("velocity",domain=self.get_opposite_side_of_interface()) # outer velocity 
		self.add_residual(weak(ui-uo,ltest)+weak(l,uitest)-weak(l,uotest))

	def before_assigning_equations_postorder(self, mesh):
		# pin Lagrange multiplier if both velocities are pinned
		# we have to iterate over the directions x,y,z (if present)
		for d in ["x","y","z"][0:self.get_nodal_dimension()]:
			self.pin_redundant_lagrange_multipliers(mesh,"_couple_velo_"+d,"velocity_"+d,opposite_interface="velocity_"+d)


class TwoLayerFlowProblem(Problem):
	def __init__(self):
		super(TwoLayerFlowProblem, self).__init__()
		self.W=1
		self.H1=0.1
		self.H2=0.1
		self.quad_size=0.01

	def define_problem(self):
		domain_names=lambda x,y: "lower" if y<self.H1 else "upper" # Name lower half lower, upper half upper
		self.add_mesh(RectangularQuadMesh(N=[math.ceil(self.W/self.quad_size), math.ceil((self.H1+self.H2)/self.quad_size)], size=[self.W, self.H1+self.H2],name=domain_names,boundary_names={"lower_upper":"interface"}))

		# Add the same required equations to both domains
		for dom in ["lower","upper"]:
			eqs=LaplaceSmoothedMesh()
			eqs+=MeshFileOutput()
			eqs+=DirichletBC(mesh_x=True)
			eqs += DirichletBC(velocity_x=0) @ "left"  # no in/outflow at the sides
			eqs += DirichletBC(velocity_x=0) @ "right"
			self.add_equations(eqs@dom)

		# Different fluids
		l_eqs = NavierStokesEquations(mass_density=0.01, dynamic_viscosity=1)  # NS equations
		u_eqs = NavierStokesEquations(mass_density=0.01, dynamic_viscosity=0.01)  # NS equations

		# no slip at top and bottom
		l_eqs += DirichletBC(velocity_x=0, velocity_y=0, mesh_y=0) @ "bottom"  # no slip at bottom and fix the mesh there
		u_eqs += DirichletBC(velocity_x=0, velocity_y=0, mesh_y=self.H1+self.H2) @ "top"  # no slip at bottom and fix the mesh there
		l_eqs += DirichletBC(pressure=0) @"bottom/left" # pin one pressure degree

		# Free surface, mesh connection and velocity connection
		l_eqs += NavierStokesFreeSurface(surface_tension=1) @ "interface"  # free surface at the top
		l_eqs += ConnectMeshAtInterface()@"interface"
		l_eqs += EnforceContinuousVelocity()@"interface"

		# Deform the initial mesh
		X, Y = var(["lagrangian_x", "lagrangian_y"])
		l_eqs += InitialCondition(mesh_y=Y * (1 + 0.25 * cos(2 * pi * X)))  # small height with a modulation
		u_eqs += InitialCondition(mesh_y=Y+ (self.H1+self.H2-Y)*(0.25 * cos(2 * pi * X)))  # small height with a modulation
		self.add_equations(l_eqs @ "lower" + u_eqs @ "upper")  # adding it to the system


if __name__=="__main__":
	with TwoLayerFlowProblem() as problem:
		problem.run(50,outstep=True,startstep=0.25)
