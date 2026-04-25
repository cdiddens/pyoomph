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

class MovingMesh(Equations):
	def define_fields(self):
		self.activate_coordinates_as_dofs() 
		
	def define_residuals(self):
		x,xtest=var_and_test("mesh_x") 
		xi,t=var(["lagrangian_x","time"]) 
		desired_pos=xi+0.125*sin(2*pi*t)
		self.add_residual(weak(x-desired_pos,xtest,lagrangian=True) )

		
class DiffusionEquation(Equations):
	def define_fields(self):
		self.define_scalar_field("c","C2")
		
	def define_residuals(self):
		c,ctest=var_and_test("c")
		self.add_residual(weak(partial_t(c,ALE="auto"),ctest)+weak(0.01*grad(c),grad(ctest)))
		
		
class ALEProblem(Problem):
	def define_problem(self):
		self.add_mesh(RectangularQuadMesh(N=32,lower_left=[-0.5,-0.5]))
		eqs=MovingMesh()
		eqs+=MeshFileOutput()
		eqs+=DiffusionEquation()
		eqs+=DirichletBC(mesh_y=True)
		x=var("coordinate")
		eqs+=InitialCondition(c=exp(-dot(x,x)*100))
		self.add_equations(eqs@"domain")
		
if __name__=="__main__":		
	with ALEProblem() as problem:
		problem.run(1,numouts=20)
