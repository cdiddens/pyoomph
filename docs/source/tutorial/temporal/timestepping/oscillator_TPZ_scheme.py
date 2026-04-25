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

# Harmonic oscillator by the trapezoidal rule
class HarmonicOscillator(ODEEquations):
	def __init__(self,*,omega=1):
		super(HarmonicOscillator,self).__init__()
		self.omega=omega
		
	def define_fields(self):
		self.define_ode_variable("y")
		self.define_ode_variable("dot_y")
		
	def define_residuals(self):
		y=var("y") 
		dot_y=var("dot_y")
		residual=(partial_t(dot_y,scheme="BDF1")+evaluate_in_past(self.omega**2*y,0.5))*testfunction(dot_y)
		residual += (partial_t(y,scheme="BDF1")-evaluate_in_past(dot_y,0.5)) * testfunction(y)
		self.add_residual(residual)
		

class HarmonicOscillatorProblem(Problem):
	def __init__(self): # Passing scheme here
		super(HarmonicOscillatorProblem,self).__init__() 
		self.omega=1

	def define_problem(self):
		eqs=HarmonicOscillator(omega=self.omega)
		
		t=var("time") # Time variable
		Ampl, phi=1, 0 #Amplitude and phase
		y0=Ampl*cos(self.omega*t+phi) #Initial condition with full time depencency
		dot_y0 = -self.omega*Ampl * sin(self.omega * t + phi) #derivative of it
		eqs+=InitialCondition(y=y0) #Set initial condition for y(t) at t=0
		eqs += InitialCondition(dot_y=dot_y0)  # And if required also for dot_y

		#Calculate the total energy. Important to also stick to the convention: BDF1 derivative and evaluate_in_past(...,0.5)
		y=var("y")
		total_energy=1/2*partial_t(y,scheme="BDF1")**2+1/2*evaluate_in_past(self.omega*y,0.5)**2
		eqs+=ODEObservables(Etot=total_energy) # Add the total energy as observable

		eqs+=ODEFileOutput() 
		self.add_equations(eqs@"harmonic_oscillator") 
		

if __name__=="__main__":
	with HarmonicOscillatorProblem() as problem:
		problem.run(endtime=100,numouts=1000)
