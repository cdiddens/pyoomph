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
from pyoomph.equations.poisson import * # use the pre-defined Poisson equation


class LShapedMesh(MeshTemplate):
	def __init__(self,Nx=10,Ny=5,H=1,W=1,domain_name="domain"):
		super(LShapedMesh,self).__init__()
		self.Nx=Nx
		self.Ny=Ny
		self.H=H
		self.W=W
		self.domain_name=domain_name

	def define_geometry(self):
		domain=self.new_domain(self.domain_name)
		# row of quads in x direction
		for ix in range(self.Nx):
			x_l,x_u=self.W*ix,self.W*(ix+1) # lower and upper x coordinate
			y_l, y_u = 0, self.H # lower and upper y coordinate
			# add the corner nodes. These will be unique, i.e. no additional node will be added if it is already present
			node_ll=self.add_node_unique(x_l,y_l)
			node_ul = self.add_node_unique(x_u, y_l)
			node_lu = self.add_node_unique(x_l, y_u)
			node_uu = self.add_node_unique(x_u, y_u)
			# add a quadrilateral element from (x_l,y_l) to (x_u,y_u)
			domain.add_quad_2d_C1(node_ll,node_ul,node_lu,node_uu)
			if ix==0: # Marking the left boundary:
				self.add_nodes_to_boundary("left",[node_ll,node_lu])

		# row of quads in y direction
		for iy in range(1,self.Ny): # we must start from 1, since the element in the corner is already present
			x_l,x_u=self.W*(self.Nx-1),self.W*self.Nx # lower and upper x coordinate
			y_l, y_u = self.H*iy, self.H*(iy+1) # lower and upper y coordinate
			node_ll=self.add_node_unique(x_l,y_l)
			node_ul = self.add_node_unique(x_u, y_l)
			node_lu = self.add_node_unique(x_l, y_u)
			node_uu = self.add_node_unique(x_u, y_u)
			domain.add_quad_2d_C1(node_ll,node_ul,node_lu,node_uu)
			if iy == self.Ny-1: # Marking the top boundary:
				self.add_nodes_to_boundary("top",[node_lu, node_uu])


class MeshTestProblem(Problem):
	def define_problem(self):
		self.add_mesh(LShapedMesh(Nx=6,Ny=4))
		eqs=MeshFileOutput()
		eqs+=PoissonEquation(name="u",source=0)
		eqs+=DirichletBC(u=0)@"left"
		eqs += DirichletBC(u=1) @ "top"
		eqs+=SpatialErrorEstimator(u=1)
		self.add_equations(eqs@"domain")

if __name__=="__main__":
	with MeshTestProblem() as problem:
		problem.initial_adaption_steps=0
		problem.solve(spatial_adapt=4)
		problem.output_at_increased_time()


		

