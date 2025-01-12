#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2025  Christian Diddens & Duarte Rocha
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
		#bind the variables. Both are called just "y", but on different domains
		y1=var("y",domain="oscillator1") #note the important keyword argument domain!
		y2=var("y",domain="oscillator2") 
		K=self.Kmatrix # shorthand

		#Define the first harmonic oscillator
		eqs1=SingleHarmonicOscillator("y",K[0][0]*y1+K[0][1]*y2)
		eqs1+=InitialCondition(y=1) 
		eqs1+=ODEFileOutput() 
		
		#Define the second harmonic oscillator
		eqs2=SingleHarmonicOscillator("y",K[1][0]*y1+K[1][1]*y2)		
		eqs2+=InitialCondition(y=0) 		
		eqs2+=ODEFileOutput() 		
		
		#Add both to different domains
		self.add_equations(eqs1@"oscillator1"+eqs2@"oscillator2") 		

if __name__=="__main__":
	with TwoCoupledHarmonicOscillatorProblem() as problem:
		problem.run(endtime=100,numouts=1000)
