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

class VanDerPolOscillator(ODEEquations):
	def __init__(self,mu): #Requires the parameter mu
		super(VanDerPolOscillator,self).__init__()
		self.mu=mu #Store the value of mu		
		
	def define_fields(self):
		self.define_ode_variable("y") #same as usual
		
	def define_residuals(self):
		y=var("y")
		residual=partial_t(y,2)-self.mu*(1-y**2)*partial_t(y)+y
		self.add_residual(residual*testfunction(y))
		


class VanDerPolProblem(Problem):

	def __init__(self):
		super(VanDerPolProblem,self).__init__() 
		self.mu=5
	
	def define_problem(self):
		eqs=VanDerPolOscillator(self.mu) #passing mu
		eqs+=InitialCondition(y=1) 
		eqs+=ODEFileOutput() 
		self.add_equations(eqs@"vdPol_oscillator") 		

if __name__=="__main__":
	with VanDerPolProblem() as problem:
		problem.run(endtime=100,numouts=1000)
