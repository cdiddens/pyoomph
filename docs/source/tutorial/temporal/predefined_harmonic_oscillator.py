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

# Import pyoomph
from pyoomph import *
# Also import the predefined harmonic oscillator equation
from pyoomph.equations.harmonic_oscillator import HarmonicOscillator


# Create a specific problem class to solve your problem. It is inherited from the generic problem class 'Problem'
class HarmonicOscillatorProblem(Problem):

	# In the constructor of the problem, we can set some default values. here omega
	def __init__(self):
		super(HarmonicOscillatorProblem,self).__init__() #we have to call the constructor of the parent class
		self.omega=1 #Set the default value of omega to 1
	

	# The method define_problem will define the entire problem you want to solve. Here, it is quite simple...
	def define_problem(self):
		eqs=HarmonicOscillator(omega=self.omega,name="y") #Create the equation, passing omega
		eqs+=InitialCondition(y=1-var("time")) #We can set both initial conditions for y and y' simultaneously
		eqs+=ODEFileOutput() #Add an output of the ODE to a file
		self.add_equations(eqs@"harmonic_oscillator") #And finally, add this combined set to the problem with the name "harmonic_oscillator"
		

if __name__=="__main__":
	with HarmonicOscillatorProblem() as problem:
		problem.run(endtime=100,numouts=1000)
