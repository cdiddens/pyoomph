#  @file
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


from pyoomph import * # Import pyoomph 
from pyoomph.expressions import * # Import some additional things to express e.g. partial_t

class PendulumEquations(ODEEquations):
	def __init__(self,*,g=1,L=1): 
		super(PendulumEquations,self).__init__()
		self.g=g
		self.L=L
		
	def define_fields(self):
		self.define_ode_variable("phi") #Angle
		
	def define_residuals(self):
		phi=var("phi")
		residual=partial_t(phi,2)+self.g/self.L*sin(phi)
		self.add_residual(residual*testfunction(phi))
		


class PendulumProblem(Problem):

	def __init__(self):
		super(PendulumProblem,self).__init__() 
		self.g=1 #Gravity
		self.L=1 #Length
	
	def define_problem(self):
		eqs=PendulumEquations(g=self.g,L=self.L)
		eqs+=InitialCondition(phi=0.9*pi) #High initial position
		eqs+=ODEFileOutput() 
		self.add_equations(eqs@"pendulum") 		

if __name__=="__main__":
	with PendulumProblem() as problem:
		problem.run(endtime=100,numouts=1000)
