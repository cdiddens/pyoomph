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
from pyoomph.equations.poisson import *  # use the pre-defined Poisson equation

from pyoomph.expressions.units import *

class GmshFishMesh(GmshTemplate):
    def __init__(self,  size=1, mouth_angle=45*degree, fin_angle=50*degree,mouth_depth_factor=0.5,fin_length_factor=0.45,fin_height_factor=0.8,domain_name="fish"):
        super(GmshFishMesh, self).__init__()
        self.size = size # all as before
        self.mouth_angle=mouth_angle 
        self.mouth_depth_factor=mouth_depth_factor 
        self.fin_angle=fin_angle 
        self.fin_length_factor=fin_length_factor
        self.fin_height_factor = fin_height_factor
        self.domain_name = domain_name

    def define_geometry(self):    	
        S=self.size # gmsh does not require to nondimensionalize the size, it will be done automatically
        # Corner nodes of the fish: instead of "add_node_unique", we use "point". 
        # We do not need p_center_body_fin and p_center_fin_end here
        p_mouth_center=self.point(-(1-self.mouth_depth_factor)*S,0)
        p_upper_jaw = self.point(-cos(self.mouth_angle / 2) * S, sin(self.mouth_angle / 2)*S)
        p_lower_jaw=self.point(-cos(self.mouth_angle/2)*S,-sin(self.mouth_angle/2)*S)
        p_upper_body_fin=self.point(cos(self.fin_angle/2)*S,sin(self.fin_angle/2)*S)
        p_lower_body_fin = self.point(cos(self.fin_angle / 2) * S, -sin(self.fin_angle / 2) * S)        
        p_upper_fin_corner=self.point((cos(self.fin_angle / 2)+self.fin_length_factor) * S, self.fin_height_factor * S)
        p_lower_fin_corner = self.point((cos(self.fin_angle / 2) + self.fin_length_factor) * S,-self.fin_height_factor * S)

        # Instead of starting with the elements, we start with the outlines
        # Create lines from lower jaw, to mouth center and to upper jaw, all named "mouth"
        self.create_lines(p_lower_jaw,"mouth",p_mouth_center,"mouth",p_upper_jaw)
        # Create the fin, also here, just a chain of straight lines, all named "fin"
        self.create_lines(p_lower_body_fin,"fin",p_lower_fin_corner,"fin",p_upper_fin_corner,"fin",p_upper_body_fin)
        # Create the body curves
        self.circle_arc(p_lower_jaw,p_lower_body_fin,center=[0,0],name="curved")
        self.circle_arc(p_upper_jaw,p_upper_body_fin,center=[0,0],name="curved")
        
        # Now, generate the surface, i.e. the domain
        self.plane_surface("mouth","fin","curved",name=self.domain_name)


class MeshTestProblem(Problem):
    def __init__(self):
        super(MeshTestProblem, self).__init__()
        self.fish_size=0.5*meter # quite large fish, isn't it...?
        self.resolution = 0.1 # Resolution of the mesh
        self.mesh_mode="quads" # Try to use quadrilateral elements
        self.space="C2"

    def define_problem(self):
        mesh=GmshFishMesh(size=self.fish_size)
        mesh.default_resolution=self.resolution
        mesh.mesh_mode=self.mesh_mode
        self.add_mesh(mesh)        
        self.set_scaling(spatial=self.fish_size) # Nondimensionalize space by the fish size

        eqs = MeshFileOutput()
        eqs += PoissonEquation(name="u", source=1, space=self.space,coefficient=1*meter**2)

        # Boundaries all u=0
        eqs += DirichletBC(u=0)@"fin"
        eqs += DirichletBC(u=0) @ "mouth"
        eqs += DirichletBC(u=0) @ "curved"

        self.add_equations(eqs @ "fish")


if __name__ == "__main__":
    with MeshTestProblem() as problem:
        problem.solve()
        problem.output_at_increased_time()


