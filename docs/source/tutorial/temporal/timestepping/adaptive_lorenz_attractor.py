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


class LorenzSystem(ODEEquations):
	def __init__(self,*,sigma=10,rho=28,beta=8/3,scheme="BDF2"): # Default parameters used by Lorenz
		super(LorenzSystem,self).__init__()
		self.sigma=sigma
		self.rho=rho
		self.beta=beta
		self.scheme=scheme

	def define_fields(self):
		self.define_ode_variable("x","y","z") 
	
	def define_residuals(self):
		x,y,z=var(["x","y","z"])
		residual=(partial_t(x)-self.sigma*(y-x))*testfunction(x)
		residual+=(partial_t(y)-x*(self.rho-z)+y)*testfunction(y)
		residual+=(partial_t(z)-x*y+self.beta*z)*testfunction(z)
		self.add_residual(time_scheme(self.scheme,residual))


class LorenzProblem(Problem):
	
	def define_problem(self):
		eqs=LorenzSystem(scheme="BDF2") # Temporal adaptivity works best with BDF2
		eqs+=InitialCondition(x=0.01)  # Some non-trivial initial position
		eqs+=TemporalErrorEstimator(x=1,y=1,z=1) # Weight all temporal error with unity
		eqs+=ODEFileOutput()  
		self.add_equations(eqs@"lorenz_attractor") 		

if __name__=="__main__":
	with LorenzProblem() as problem:
		# outstep=True means output every step
		# startstep is the first time step
		# temporal_error controls the maximum difference between prediction and actual result
		problem.run(endtime=100,outstep=True,startstep=0.0001,temporal_error=0.005)
