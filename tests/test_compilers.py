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
from pyoomph.expressions import *
# Also import the predefined harmonic oscillator equation
from pyoomph.equations.harmonic_oscillator import HarmonicOscillator

class ErrorToAnalyticalSolution(ODEEquations):
    def __init__(self,analytical_solution):
       super().__init__()
       self.analytical_solution=analytical_solution
  
    def define_fields(self):
       self.define_ode_variable("error")
       self.define_ode_variable("anasol")
    
    def define_residuals(self):
       error,error_test=var_and_test("error")
       difference=var("y")-self.analytical_solution
       self.add_weak(partial_t(error)-difference**2,error_test)
       ya,ya_test=var_and_test("anasol")
       self.add_weak(ya-self.analytical_solution,ya_test)
  

# Create a specific problem class to solve your problem. It is inherited from the generic problem class 'Problem'
class HarmonicOscillatorProblem(Problem):

	# In the constructor of the problem, we can set some default values. here omega
	def __init__(self):
		super(HarmonicOscillatorProblem,self).__init__() #we have to call the constructor of the parent class
		self.omega=1 #Set the default value of omega to 1
	

	# The method define_problem will define the entire problem you want to solve. Here, it is quite simple...
	def define_problem(self):
		eqs=HarmonicOscillator(omega=self.omega,name="y") #Create the equation, passing omega
		eqs+=InitialCondition(y=cos(var("time"))) #We can set both initial conditions for y and y' simultaneously
		eqs+=ODEFileOutput() #Add an output of the ODE to a file
		self.add_equations(eqs@"harmonic_oscillator") #And finally, add this combined set to the problem with the name "harmonic_oscillator"
		

def test_system():
	with HarmonicOscillatorProblem() as problem:
		problem.set_c_compiler("system")
		problem+=ErrorToAnalyticalSolution(cos(var("time")))@"harmonic_oscillator"
		problem.run(endtime=2*pi,maxstep=2*pi/100)
		err=problem.get_ode("harmonic_oscillator").get_value("error",as_float=True)
		assert err<1e-4
  
def test_tcc():
	with HarmonicOscillatorProblem() as problem:
		problem.set_c_compiler("tcc")
		problem+=ErrorToAnalyticalSolution(cos(var("time")))@"harmonic_oscillator"
		problem.run(endtime=2*pi,maxstep=2*pi/100)
		err=problem.get_ode("harmonic_oscillator").get_value("error",as_float=True)
		assert err<1e-4
  
