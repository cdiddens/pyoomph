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
from stokes import * # Import the previous Stokes problem
		
if __name__ == "__main__":		

	# bind the velocity as variable
	u=var("velocity")
	# symmetrized shear
	S=sym(grad(u))
	# get the scalar shear rate by sqrt(2*S:S)
	shear_rate=square_root(2*contract(S,S))
	# Shear thickening fluid -> viscosity increases with increasing shear
	# we wrap it in a subexpression since the viscosity appears in the equation for u_x and u_y
	# Thereby, the viscosity will be only evaluated once in the compiled code, which speeds up the calculation
	mu=subexpression(1+shear_rate)
	
	# pass it to the Stokes problem, which will pass it to the equations
	with StokesSpaceTestProblem(mu,"C2","C1") as problem: 
	
		# Since the problem is nonlinear, it is essential to provide a good guess for the initial condition
		# Here, we use the one of the Poiseuille flow for Newtonian liquids
		y=var("coordinate_y")
		ux_init=y*(1-y)
		# We can add arbitrary equations/conditions to the problem by adding them to additional_equations
		# Here, we set the initial condition to help the nonlinear problem to converge
		problem.additional_equations+=InitialCondition(velocity_x=ux_init)@"domain"
		
		problem.solve() # solve and output
		problem.output()
	
		
