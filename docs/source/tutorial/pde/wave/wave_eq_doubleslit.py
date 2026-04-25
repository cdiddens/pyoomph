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


from wave_eq import *  # Import the wave equation from the previous example


class DoubleSlitMesh(GmshTemplate):
	def __init__(self,problem):
		super(DoubleSlitMesh,self).__init__()
		self.problem=problem # store the problem to access the properties
		self.default_resolution=self.problem.resolution
		self.mesh_mode="tris" # use triangles here
		
	def define_geometry(self):		
		p=self.problem

		# corner points of the inlet
		p_in_0=self.point(-p.inlet_length,0)
		p_in_H = self.point(-p.inlet_length, p.domain_height)
		p_wl_0=self.point(0,0)
		p_wl_H = self.point(0, p.domain_height)

		# slit corner points, upper slit
		p_wl_ub = self.point(0, (p.domain_height-p.slit_width+p.slit_distance)*0.5)
		p_wl_uu = self.point(0, (p.domain_height+p.slit_width+p.slit_distance)*0.5)
		p_wr_ub = self.point(p.wall_thickness, (p.domain_height - p.slit_width + p.slit_distance) * 0.5)
		p_wr_uu = self.point(p.wall_thickness, (p.domain_height + p.slit_width + p.slit_distance) * 0.5)

		# slit corner points, bottom slit
		p_wl_bb = self.point(0, (p.domain_height - p.slit_width - p.slit_distance) * 0.5)
		p_wl_bu = self.point(0, (p.domain_height + p.slit_width - p.slit_distance) * 0.5)
		p_wr_bb = self.point(p.wall_thickness, (p.domain_height - p.slit_width - p.slit_distance) * 0.5)
		p_wr_bu = self.point(p.wall_thickness, (p.domain_height + p.slit_width - p.slit_distance) * 0.5)

		# corner points of the domain behind the slit
		p_wr_0 = self.point(p.wall_thickness, 0)
		p_wr_H = self.point(p.wall_thickness, p.domain_height)
		p_scr_0 = self.point(p.screen_distance, 0)
		p_scr_H = self.point(p.screen_distance, p.domain_height)
	
		# Create a line loop of the exterior boundary
		lines=self.create_lines(p_in_0,"inlet",p_in_H,"top",p_wl_H,"wall_left",p_wl_uu,"wall",p_wr_uu,"wall",p_wr_H,"out_top",p_scr_H,"screen",p_scr_0,"out_bottom",p_wr_0,"wall",p_wr_bb,"wall",p_wl_bb,"wall_left",p_wl_0,"bottom",p_in_0)		
		# The center part of the wall is a hole in the mesh
		holes=self.create_lines(p_wl_bu,"wall_left",p_wl_ub,"wall",p_wr_ub,"wall",p_wr_bu,"wall",p_wl_bu)
		self.plane_surface(*lines,name="domain",holes=[holes]) # create the domain


# Measure the intensity of the waves at the screen
class WaveEquationScreen(InterfaceEquations):
	required_parent_type =  WaveEquation

	def define_fields(self):
		self.define_scalar_field("I","C2") # intensity as interface field

	def define_residuals(self):
		I,J=var_and_test("I")
		self.add_residual(weak(partial_t(I)-var("u")**2,J)) # I=integral of u^2 dt


class DoubleSlitProblem(Problem):
	def __init__(self):
		super(DoubleSlitProblem, self).__init__()
		self.c = 1  # speed
		self.omega=10 # wave frequency		
		self.inlet_length=0.4 # length of the inlet
		self.wall_thickness=0.1 # thickness of the wall
		self.screen_distance=2 # distance of the screen
		self.domain_height=4 # height of the entire domain
		self.slit_width=0.2 # width of a slit
		self.slit_distance=0.5 # distance of the slits
		self.resolution=0.04 # mesh resolution, the smaller the finer

	

	def define_problem(self):
		mesh=DoubleSlitMesh(self) # mesh

		#mesh=LineMesh(size=6,N=200)
		self.add_mesh(mesh)

		eqs = WaveEquation(c=self.c)  # wave equation
		eqs += MeshFileOutput()  # mesh output

		# initial condition: incoming wave wave, exponentially damped 
		u0=cos(self.omega*(var("coordinate_x")+self.inlet_length-self.c*var("time")))*exp(-10*(var("coordinate_x")+self.inlet_length))
		eqs +=InitialCondition(u=u0)				
		eqs += DirichletBC(u=u0) @ "inlet"  # incoming wave

		# Let the waves just flow out/absorb without any reflection at some interfaces
		u=var("u")
		eqs += NeumannBC(u=self.c*partial_t(u)) @ "wall_left"
		eqs += NeumannBC(u=self.c*partial_t(u)) @ "screen"
		eqs += NeumannBC(u=self.c*partial_t(u)) @ "out_top"
		eqs += NeumannBC(u=self.c*partial_t(u)) @ "out_bottom"

		# Measure the intesity at the screen
		eqs += WaveEquationScreen()@"screen"
		eqs += TextFileOutput() @ "screen"

		self.add_equations(eqs @ "domain")  


if __name__ == "__main__":
	with DoubleSlitProblem() as problem:
		problem.c=3 # increase the wave speed
		problem.omega=20 # and the wave frequency
		problem.run(1, outstep=True, startstep=0.01)
