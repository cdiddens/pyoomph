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


class HarmonicOscillator(ODEEquations):
	def __init__(self,*,omega=1,scheme="Newmark2"): #Passing a time stepping scheme
		super(HarmonicOscillator,self).__init__()
		allowed_schemes={"Newmark2","BDF1","BDF2"} #Possible values
		if not (scheme in allowed_schemes): #Test for valid input
			raise ValueError("Unknown time stepping scheme: "+str(scheme)+". Allowed: "+str(allowed_schemes))
		self.scheme=scheme
		self.omega=omega
		
	def define_fields(self):
		self.define_ode_variable("y")
		if self.scheme!="Newmark2":
			self.define_ode_variable("dot_y") #Additional variable for first order ODE system
		
	def define_residuals(self):
		y=var("y") 
		if self.scheme=="Newmark2": #One second order ODE
			residual=(partial_t(y,2)+self.omega**2 *y)*testfunction(y)  
		else:	#Two first order ODEs
			dot_y=var("dot_y")
			residual=(partial_t(dot_y,scheme=self.scheme)+self.omega**2*y)*testfunction(dot_y)
			residual+=(partial_t(y,scheme=self.scheme)-dot_y)*testfunction(y)
		self.add_residual(residual)
		

class HarmonicOscillatorProblem(Problem):
	def __init__(self,scheme="Newmark2"): # Passing scheme here
		super(HarmonicOscillatorProblem,self).__init__() 
		self.omega=1
		self.scheme=scheme

	def define_problem(self):
		eqs=HarmonicOscillator(omega=self.omega,scheme=self.scheme)
		
		t=var("time") # Time variable
		Ampl, phi=1, 0 #Amplitude and phase
		y0=Ampl*cos(self.omega*t+phi) #Initial condition with full time depencency
		dot_y0 = -self.omega*Ampl * sin(self.omega * t + phi) #derivative of it
		eqs+=InitialCondition(y=y0) #Set initial condition for y(t) at t=0
		if self.scheme!="Newmark2":
			eqs += InitialCondition(dot_y=dot_y0)  # And if required also for dot_y

		#Calculate the total energy
		y=var("y")
		total_energy=1/2*partial_t(y,scheme=self.scheme)**2+1/2*(self.omega*y)**2
		eqs+=ODEObservables(Etot=total_energy) # Add the total energy as observable

		eqs+=ODEFileOutput() 
		self.add_equations(eqs@"harmonic_oscillator") 
		

if __name__=="__main__":
	for scheme in {"Newmark2","BDF1","BDF2"}:
		with HarmonicOscillatorProblem(scheme=scheme) as problem:
			problem.set_output_directory("osci_timestepping_"+scheme)
			problem.run(endtime=100,numouts=1000)
