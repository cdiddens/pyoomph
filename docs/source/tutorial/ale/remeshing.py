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


from laplace_smoothed_mesh import *
from pyoomph.meshes.remesher import *
		
class RemeshingProblem(Problem):
	def __init__(self):
		super(RemeshingProblem, self).__init__()
		self.remeshing=True # shall we remesh or not
		self.remesh_options=RemeshingOptions(max_expansion=2,min_expansion=0.3,min_quality_decrease=0.2) # when to remesh

	def define_problem(self):
		# Create a mesh and add a remesher
		mesh=RectangularQuadMesh(N=6)
		mesh.remesher=Remesher2d(mesh)

		# Add the mesh and use the Lagrange smoothed mesh
		self.add_mesh(mesh)
		eqs=LaplaceSmoothedMesh()
		eqs+=MeshFileOutput()
		# Fix some interfaces
		eqs+=DirichletBC(mesh_x=0,mesh_y=True)@"left"
		eqs+=DirichletBC(mesh_x=True,mesh_y=0)@"bottom"
		eqs+=DirichletBC(mesh_y=1)@"top"

		# Moving boundary
		xi=var("lagrangian")
		eqs+=DirichletBC(mesh_x=1+0.5*xi[1]*var("time"))@"right" # move the right interface with time

		# Remeshing based on the options
		eqs+=RemeshWhen(self.remesh_options)
		# optional: setting particular sizes at interfaces or corners
		eqs+=RemeshMeshSize(size=0.2)@"left" # size of 0.2 at the left interface
		eqs += RemeshMeshSize(size=0.02) @ "right/top" # size of 0.02 at the top right corner

		self.add_equations(eqs@"domain")
		
if __name__=="__main__":		
	with RemeshingProblem() as problem:
		# problem.remesh_options.active=False we can deactivate remeshing by the remesh options as well
		problem.run(10,outstep=True,startstep=0.5,maxstep=0.5)
