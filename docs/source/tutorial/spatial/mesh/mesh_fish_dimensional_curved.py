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

class FishMesh(MeshTemplate):
    def __init__(self,  size=1, mouth_angle=45*degree, fin_angle=50*degree,mouth_depth_factor=0.5,fin_length_factor=0.45,fin_height_factor=0.8,domain_name="fish"):
        super(FishMesh, self).__init__()
        self.size = size # overall size of the fish (potentially dimesional)
        self.mouth_angle=mouth_angle # angle of the mouth-opening (with respect to the body center)
        self.mouth_depth_factor=mouth_depth_factor # depth of the mouth
        self.fin_angle=fin_angle # angle of the fin-body-connection (with respect to the body center)
        self.fin_length_factor=fin_length_factor
        self.fin_height_factor = fin_height_factor
        self.domain_name = domain_name # name of the fish domain

    def define_geometry(self):
        domain = self.new_domain(self.domain_name)

        S=self.nondim_size(self.size) # Important: Nondimensionalize the potentially dimensional size

        # Corner nodes of the fish
        n_mouth_center=self.add_node_unique(-(1-self.mouth_depth_factor)*S,0)
        n_upper_jaw = self.add_node_unique(-cos(self.mouth_angle / 2) * S, sin(self.mouth_angle / 2)*S)
        n_lower_jaw=self.add_node_unique(-cos(self.mouth_angle/2)*S,-sin(self.mouth_angle/2)*S)
        n_upper_body_fin=self.add_node_unique(cos(self.fin_angle/2)*S,sin(self.fin_angle/2)*S)
        n_lower_body_fin = self.add_node_unique(cos(self.fin_angle / 2) * S, -sin(self.fin_angle / 2) * S)
        n_center_body_fin = self.add_node_unique(cos(self.fin_angle / 2) * S, 0)
        n_upper_fin_corner=self.add_node_unique((cos(self.fin_angle / 2)+self.fin_length_factor) * S, self.fin_height_factor * S)
        n_lower_fin_corner = self.add_node_unique((cos(self.fin_angle / 2) + self.fin_length_factor) * S,-self.fin_height_factor * S)
        n_center_fin_end = self.add_node_unique((cos(self.fin_angle / 2) + self.fin_length_factor) * S, 0)

        # Elements
        domain.add_quad_2d_C1(n_lower_jaw,n_lower_body_fin,n_mouth_center,n_center_body_fin) # lower body part
        domain.add_quad_2d_C1(n_mouth_center, n_center_body_fin,n_upper_jaw, n_upper_body_fin ) # upper body part
        domain.add_quad_2d_C1(n_lower_body_fin,n_lower_fin_corner,n_center_body_fin,n_center_fin_end) # lower fin part
        domain.add_quad_2d_C1(n_center_body_fin,n_center_fin_end,n_upper_body_fin,n_upper_fin_corner) # upper fin part

        # Curved entities
        upper_body_curve=self.create_curved_entity("circle_arc",n_upper_jaw,n_upper_body_fin,center=[0,0])
        lower_body_curve = self.create_curved_entity("circle_arc", n_lower_jaw, n_lower_body_fin, center=[0, 0])
        self.add_facet_to_curve_entity([n_upper_jaw,n_upper_body_fin],upper_body_curve) # top body curve
        self.add_facet_to_curve_entity([n_lower_body_fin, n_lower_jaw], lower_body_curve) # bottom body curve

        # Add nodes to boundaries
        self.add_nodes_to_boundary("curved",[n_upper_body_fin, n_lower_body_fin,n_lower_jaw,n_upper_jaw]) # nodes on curved body parts
        self.add_nodes_to_boundary("mouth", [ n_lower_jaw,n_upper_jaw,n_mouth_center])  # nodes of the mouth
        self.add_nodes_to_boundary("fin",[n_upper_body_fin,n_lower_body_fin,n_center_fin_end,n_upper_fin_corner,n_lower_fin_corner]) # fin


class MeshTestProblem(Problem):
    def __init__(self):
        super(MeshTestProblem, self).__init__()
        self.fish_size=1*meter # quite large fish, isn't it...?
        self.max_refinement_level = 5 # maximum level of refinements
        self.space="C2"

    def define_problem(self):
        self.add_mesh(FishMesh(size=self.fish_size))
        self.set_scaling(spatial=self.fish_size) # Nondimensionalize space by the fish size

        eqs = MeshFileOutput()
        # We must set a meter^2 coefficient to be consistent with the units
        eqs += PoissonEquation(name="u", source=1, space=self.space,coefficient=1*meter**2)

        # Boundaries all u=0
        eqs += DirichletBC(u=0)@"fin"
        eqs += DirichletBC(u=0) @ "mouth"
        eqs += DirichletBC(u=0) @ "curved"

        # refine the curved boundary to the highest order (i.e. max_refinement_level) during adaptive solves
        eqs += RefineToLevel("max")@"curved"
        eqs += SpatialErrorEstimator(u=1) # and adapt on all other elements based on the error

        self.add_equations(eqs @ "fish")


if __name__ == "__main__":
    with MeshTestProblem() as problem:
        problem.solve(spatial_adapt=problem.max_refinement_level)
        problem.output_at_increased_time()


