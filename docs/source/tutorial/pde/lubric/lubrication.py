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
from pyoomph.expressions import *

class LubricationEquations(Equations):
	def __init__(self,sigma=1,mu=1,disjoining_pressure=0):
		super(LubricationEquations, self).__init__()
		self.sigma=sigma
		self.mu=mu
		self.disjoining_pressure=disjoining_pressure
		
	def define_fields(self):
		self.define_scalar_field("h","C2")
		self.define_scalar_field("p","C2")		
		
	def define_residuals(self):
		h,eta=var_and_test("h")
		p,q=var_and_test("p")		
		self.add_residual(weak(partial_t(h),eta)+weak(1/self.mu*(h**3/3*grad(p)-h**2/2*grad(self.sigma)),grad(eta)))
		self.add_residual(weak(p-self.disjoining_pressure,q)-weak(self.sigma*grad(h),grad(q)))
		
		
class LubricationProblem(Problem):	
	def define_problem(self):
		self.add_mesh(LineMesh(N=100)) # simple line mesh		
		eqs=LubricationEquations() # equations
		eqs+=TextFileOutput() # output	
		eqs+=InitialCondition(h=0.05*(1+0.25*cos(2*pi*var("coordinate_x"))))  # small height with a modulation
		self.add_equations(eqs@"domain") # adding the equation

		
if __name__=="__main__":
	with LubricationProblem() as problem:
		problem.run(50,outstep=True,startstep=0.25)
