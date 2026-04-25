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


class WaveEquation(Equations):
	def __init__(self,c=1):
		super(WaveEquation, self).__init__()
		self.c=c # speed
		
	def define_fields(self):
		self.define_scalar_field("u","C2")
		
	def define_residuals(self):
		u,w=var_and_test("u")
		self.add_residual(weak(partial_t(u,2),w)+weak(self.c**2*grad(u),grad(w)))
		
		
class WaveProblem(Problem):
	def __init__(self):
		super(WaveProblem, self).__init__()
		self.c=1 # speed
		self.L=10 # domain length
		self.N=100 # number of elements
		
	def define_problem(self):
		# interval mesh from [-L/2 : L/2 ] with N elements
		self.add_mesh(LineMesh(N=self.N,size=self.L,minimum=-self.L/2))
		
		eqs=WaveEquation() # equation
		eqs+=TextFileOutput() # output
		eqs+=DirichletBC(u=0)@"left" # fixed knots at the end points
		eqs+=DirichletBC(u=0)@"right"		
		
		# Initial condition
		x,t=var(["coordinate_x","time"])
		#u_init=exp(-x**2) # This has no initial motion, i.e. partial_t(u)=0
		u_init=exp(-(x-self.c*t)**2) # This one moves initially to the right
		eqs+=InitialCondition(u=u_init) # setting the initial condition
		
		self.add_equations(eqs@"domain") # adding the equation
		
if __name__=="__main__":
	with WaveProblem() as problem:
		problem.run(20,outstep=True,startstep=0.1)
