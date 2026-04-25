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

# import the predefined harmonic oscillator equation
from pyoomph.equations.harmonic_oscillator import HarmonicOscillator

#Inherit from the HarmonicOscillator class
class VanDerPolOscillator(HarmonicOscillator):
	def __init__(self,mu): 
		damping=-mu/2*(1-var("y")**2)
		super(VanDerPolOscillator,self).__init__(name="y",damping=damping,omega=1)

		
class VanDerPolProblem(Problem):

	def __init__(self):
		super(VanDerPolProblem,self).__init__() 
		self.mu=5
	
	def define_problem(self):
		eqs=VanDerPolOscillator(self.mu) #passing mu
		eqs+=InitialCondition(y=1) 
		eqs+=ODEFileOutput() 
		self.add_equations(eqs@"vdPol_oscillator") 		

if __name__=="__main__":
	with VanDerPolProblem() as problem:
		problem.run(endtime=100,numouts=1000)
