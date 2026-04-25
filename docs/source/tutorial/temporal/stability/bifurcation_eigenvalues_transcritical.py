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

        ode = problem.get_ode("transcritical")  # Get access the ODE (note: it will initialize the problem!)

        for startpoint in [0.001,0.8]: #Take different start points
            ode.set_value(x=startpoint) #Start there
            problem.solve() # Solve for the stationary solution
            xvalue = ode.get_value("x") # Get the current value of x
            print(f"Starting at {startpoint} gives the stationary solution x={xvalue} with r={problem.r.value}")
            eigen_vals,eigen_vects=problem.solve_eigenproblem(1)
            print("Eigenvalues are "+str(eigen_vals))
            print("Thus, this solution is "+("unstable" if eigen_vals[0].real>0 else "stable"))
