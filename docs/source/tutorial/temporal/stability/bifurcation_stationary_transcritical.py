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


from bifurcation_transient_transcritical import * #Import the equation and problem from the previous script

if __name__=="__main__":
    with TranscriticalProblem() as problem:
        problem.quiet() # Do not pollute our output with all the messages

        problem.r.value=1 # Parameter value
        problem.x0=0.001 # Start slightly above the unstable solution x=0

        ode = problem.get_ode("transcritical")  # Get access the ODE (note: it will initialize the problem!)

        xvalue = ode.get_value("x") #Get the current value of x
        print(f"We are starting at x={xvalue}")

        problem.solve(timestep=None) # Solve without a timestep means stationary solve
        xvalue = ode.get_value("x") # Get the current value of x
        print(f"Currently, we are at the stationary solution x={xvalue} with r={problem.r.value}")

        ode.set_value(x=0.8) # Set the current value of x
        problem.solve() # we can omit timestep=None, since it is default
        xvalue = ode.get_value("x")
        print(F"Currently, we are at the stationary solution x={xvalue} with r={problem.r.value}")

