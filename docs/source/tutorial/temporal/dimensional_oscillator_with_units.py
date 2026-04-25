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


from pyoomph import *
from pyoomph.expressions import * #Import the basic expressions
from pyoomph.expressions.units import * #Import units like meter and so on


class DimensionalOscillator(ODEEquations):
	def __init__(self,*,m=1,k=1): #Default values can be nondimensional
		super(DimensionalOscillator,self).__init__()
		self.m=m
		self.k=k
		
	def define_fields(self):
		# bind the scaling of time
		T=scale_factor("temporal")
		X=scale_factor("x") # and of the variable x
		self.define_ode_variable("x",testscale=T**2/X) #set the test function scale here
				
	def define_residuals(self):
		x=var("x") # dimensional x
		# write the equation as before with dimensions
		eq=partial_t(x,2)+self.k/self.m*x
		self.add_residual(eq*testfunction(x)) # testfunction(x) will expand to T**2/X * ~chi

class DimensionalOscillatorProblem(Problem):

	def __init__(self):
		super(DimensionalOscillatorProblem,self).__init__() 
		self.mass=100*kilogram  #Specifying dimensional parameters
		self.spring_constant=1000*newton/meter
		self.initial_displacement=2*centi*meter #and a dimensional initial condition	
	
	def define_problem(self):
		eqs=DimensionalOscillator(m=self.mass,k=self.spring_constant) #Setting dimensional parameters
		eqs+=InitialCondition(x=self.initial_displacement) #Setting a dimensional displacement
		eqs+=ODEFileOutput() 
		
		#Important step: Introduce a good scaling
		T=square_root(self.mass/self.spring_constant)
		self.set_scaling(temporal=T,x=self.initial_displacement) #and set it to the problem
		
		self.add_equations(eqs@"harmonic_oscillator") 
		

if __name__=="__main__":
	with DimensionalOscillatorProblem() as problem:
		problem.run(endtime=10*second,numouts=1000) #endime is now also in seconds!
