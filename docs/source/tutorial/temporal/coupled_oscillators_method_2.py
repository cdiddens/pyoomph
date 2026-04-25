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

class SingleHarmonicOscillator(ODEEquations):
	def __init__(self,name,terms): #Pass the name of the unknown and the terms T
		super(SingleHarmonicOscillator,self).__init__()
		self.name=name #Store the name of the unknown
		self.terms=terms #and the terms to consider
		
	def define_fields(self):
		self.define_ode_variable(self.name) 
		
	def define_residuals(self):
		y=var(self.name) #Bind the single variable
		# Calculate the residuals
		residual=partial_t(y,2)+self.terms #Just add the passed terms here
		self.add_residual(residual*testfunction(y))
		


class TwoCoupledHarmonicOscillatorProblem(Problem):

	def __init__(self):
		super(TwoCoupledHarmonicOscillatorProblem,self).__init__() 
		self.Kmatrix=[[1,-0.5],[-0.2,0.4]] # Some coefficient matrix
	
	def define_problem(self):
		y1,y2=var(["y1","y2"]) #bind the variables here
		K=self.Kmatrix # shorthand
		#Adding two single oscillators, but passing the coupling
		eqs=SingleHarmonicOscillator("y1",K[0][0]*y1+K[0][1]*y2)
		eqs+=SingleHarmonicOscillator("y2",K[1][0]*y1+K[1][1]*y2)		
		eqs+=InitialCondition(y1=1,y2=0) #Setting initial condition
		eqs+=ODEFileOutput() 
		self.add_equations(eqs@"coupled_oscillator") 		

if __name__=="__main__":
	with TwoCoupledHarmonicOscillatorProblem() as problem:
		problem.run(endtime=100,numouts=1000)
