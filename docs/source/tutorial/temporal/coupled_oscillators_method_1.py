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

class TwoCoupledHarmonicOscillator(ODEEquations):
	def __init__(self,Kmatrix): #Required to pass the coupling matrix here
		super(TwoCoupledHarmonicOscillator,self).__init__()
		self.Kmatrix=Kmatrix #Store the matrix in the equations object
		
	def define_fields(self):
		self.define_ode_variable("y1") #define both unknowns
		self.define_ode_variable("y2")		
		
	def define_residuals(self):
		y1,y2=var(["y1","y2"]) #Bind both unknowns to variables
		# Calculate the residuals
		residual1=partial_t(y1,2)+self.Kmatrix[0][0]*y1+self.Kmatrix[0][1]*y2
		residual2=partial_t(y2,2)+self.Kmatrix[1][0]*y1+self.Kmatrix[1][1]*y2		

		# Add both residuals
		self.add_residual(residual1*testfunction(y1)+residual2*testfunction(y2))
		

# The remaining part is analogous to the normal oscillator
class TwoCoupledHarmonicOscillatorProblem(Problem):

	def __init__(self):
		super(TwoCoupledHarmonicOscillatorProblem,self).__init__() 
		self.Kmatrix=[[1,-0.5],[-0.2,0.4]] # Some coefficient matrix
	
	def define_problem(self):
		eqs=TwoCoupledHarmonicOscillator(self.Kmatrix)
		eqs+=InitialCondition(y1=1,y2=0) #Setting initial condition
		eqs+=ODEFileOutput() 
		self.add_equations(eqs@"coupled_oscillator") 		

if __name__=="__main__":
	with TwoCoupledHarmonicOscillatorProblem() as problem:
		problem.run(endtime=100,numouts=1000)
