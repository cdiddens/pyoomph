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
from pyoomph.equations.harmonic_oscillator import *
from pyoomph.expressions import * # subexpression
from pyoomph.expressions.cb import * # Custom math expressions


class TrapezoidalFunction(CustomMathExpression):
	def __init__(self,*,wait_time=5, flank_time=0.25,high_time=10,amplitude=1):
		super(TrapezoidalFunction, self).__init__()
		self.wait_time=wait_time # Pass some parameters to the function already in the constructor
		self.flank_time=flank_time
		self.high_time=high_time
		self.amplitude=amplitude

	# This method will be called whenever the function must be evaluated
	def eval(self,arg_array):
		t=arg_array[0] # Bind local t to the first passed argument
		if t<self.wait_time:
			return 0.0 # Before the pulse
		elif t<self.wait_time+self.flank_time:
			return self.amplitude*(t-self.wait_time)/self.flank_time # flank up
		elif t<self.wait_time+self.flank_time+self.high_time:
			return self.amplitude # at the plateau
		elif t<self.wait_time+2*self.flank_time+self.high_time:
			return self.amplitude*(1-(t-self.wait_time-self.flank_time-self.high_time)/self.flank_time) # flank down
		else:
			return 0.0 # after the plateau


class TrapezoidallyDrivenOscillatorProblem(Problem):

	def define_problem(self):
		t = var("time")
		# Create a trapezoidal driving
		driving = TrapezoidalFunction(wait_time=10, high_time=20, flank_time=1)
		# Evaluate at t (which is the current time) and wrap it in a subexpression (optional, but recommended)
		driving = subexpression(driving(t))
		# pass the driving function evaluated at t here
		eqs=HarmonicOscillator(omega=1,damping=0.2,driving=driving,name="y")
		eqs+=InitialCondition(y=0.1)
		eqs+=ODEFileOutput()
		eqs+=ODEObservables(driving=driving) # Also output the driving to the file
		self.add_equations(eqs@"harmonic_oscillator") 

if __name__=="__main__":
	with TrapezoidallyDrivenOscillatorProblem() as problem:
		problem.run(endtime=100,numouts=1000)
