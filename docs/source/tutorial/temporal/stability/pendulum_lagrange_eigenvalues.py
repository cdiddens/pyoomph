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


#We can reuse the old class, since it is already a system of first order ODEs
from pendulum_lagrange_multiplier import *


if __name__=="__main__":
	with PendulumProblem() as problem:
		problem.quiet()
		ode = problem.get_ode("pendulum")

		problem.solve(timestep=0.001) #Make one little time step to find a good guess for lambda
		problem.solve()
		x,y,lambd,xdot,ydot=ode.get_value(["x","y","lambda_pendulum","xdot","ydot"])
		print(f"Solution: x={x}, y={y}, lambda={lambd}, xdot={xdot}, ydot={ydot}")
		eig_vals,eig_vects=problem.solve_eigenproblem(5)
		print(eig_vals)


		# Just flip the solution upside down
		ode.set_value(x=0.0,y=-y,lambda_pendulum=-lambd) # We also need to flip the rod tension lambda
		problem.solve() # We can still solve
		x, y, lambd, xdot, ydot = ode.get_value(["x", "y", "lambda_pendulum", "xdot", "ydot"])
		print(f"Solution: x={x}, y={y}, lambda={lambd}, xdot={xdot}, ydot={ydot}")
		eig_vals, eig_vects = problem.solve_eigenproblem(5)
		print(eig_vals)
