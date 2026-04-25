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

# We define a new class called HarmonicOscillator, which is inherited from the generic ODEEquations
class HarmonicOscillator(ODEEquations):
	# Constructor, allow to set some parameters like the name of the variable, omega, damping and driving
	def __init__(self,*,name="y",omega=1,damping=0,driving=0):
		super(HarmonicOscillator,self).__init__()
		self.name=name #Store these as members of the equation object
		self.omega=omega
		self.damping=damping
		self.driving=driving
		
	# This function is called to define all fields in this ODE (system)
	def define_fields(self):
		self.define_ode_variable(self.name)
		
	# This function will finally define the equations
	def define_residuals(self):
		y=var(self.name) # bind the local variable y to the ODE variable
		# Write the equation in residual form, i.e. lhs-rhs=0
		residual=partial_t(y,2)+2*self.damping*partial_t(y)+self.omega**2 *y - self.driving
		# And add the residual to the equation. Here, we have to project it on the test function.
		self.add_residual(residual*testfunction(y))
		


# The remainder is almost the same is in the example nondim_harmonic_osci.py
class HarmonicOscillatorProblem(Problem):

	def __init__(self):
		super(HarmonicOscillatorProblem,self).__init__() 
		self.omega=1
		self.damping=0.1 #But we add some default damping here
		t=var("time")
		self.driving=0.2*cos(0.2*t) #and some driving
	

	def define_problem(self):
		eqs=HarmonicOscillator(omega=self.omega,damping=self.damping,driving=self.driving,name="y") #We also pass the damping and driving here
		eqs+=InitialCondition(y=1-var("time"))
		eqs+=ODEFileOutput() 
		self.add_equations(eqs@"harmonic_oscillator") 
		

if __name__=="__main__":
	with HarmonicOscillatorProblem() as problem:
		problem.run(endtime=100,numouts=1000)
