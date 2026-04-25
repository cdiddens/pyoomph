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

class LaplaceSmoothedMesh(Equations):

	def define_fields(self):
		# let the mesh coordinates become a variables, approximated with second order Lagrange basis functions
		self.activate_coordinates_as_dofs(coordinate_space="C2") 
		
	def define_residuals(self):
		x,xtest=var_and_test("mesh") # Eulerian mesh coordinates
		xi=var("lagrangian") # fixed Lagrangian coordinates
		d=x-xi # displacement
		# Weak formulation: gradients and integrals are carried out with respect to the Lagrangian coordinates
		self.add_residual(weak(grad(d,lagrangian=True), grad(xtest, lagrangian=True),lagrangian=True) )
		
		
class LaplaceSmoothProblem(Problem):
	def define_problem(self):
		self.initial_adaption_steps=0
		self.add_mesh(RectangularQuadMesh(N=6))
		eqs=LaplaceSmoothedMesh()
		eqs+=MeshFileOutput()
		eqs+=DirichletBC(mesh_x=0,mesh_y=True)@"left" # fix the mesh at x=0 and keep y in place
		eqs+=DirichletBC(mesh_x=True,mesh_y=0)@"bottom" # fix the mesh at y=0 and keep x in place		
		xi=var("lagrangian") # Lagrangian coordinate
		eqs+=DirichletBC(mesh_x=1+0.5*xi[1])@"right" # linear slope at the left
		eqs+=DirichletBC(mesh_y=1+0.25*xi[0]*(1-xi[0]))@"top" # quadratic deformation at the top
		eqs+=SpatialErrorEstimator(mesh=1) # Adapt where large deformations are present
		
		self.add_equations(eqs@"domain")
		
if __name__=="__main__":		
	with LaplaceSmoothProblem() as problem:
		problem.output()
		problem.solve(spatial_adapt=4)
		problem.output_at_increased_time()
