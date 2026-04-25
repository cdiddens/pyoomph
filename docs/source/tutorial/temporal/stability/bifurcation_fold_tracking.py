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



from bifurcation_fold_param_change import *

if __name__=="__main__":
    with FoldProblem() as problem:

        # Find any start solution, which must be close to the bifurcation
        problem.r.value=1
        problem.get_ode("fold").set_value(x=1)
        problem.solve()

        # Find a guess for the normalization constraint
        problem.solve_eigenproblem(1)
        vguess=problem.get_last_eigenvectors()[0] # use the eigenvector as guess

        # Activate fold bifurcation tracking in parameter r and solve the augmented system
        problem.activate_bifurcation_tracking(problem.r,"fold",eigenvector=vguess)
        problem.solve()

        print(f"Critical at r_c={problem.r.value} and x_c={problem.get_ode('fold').get_value('x')}")
