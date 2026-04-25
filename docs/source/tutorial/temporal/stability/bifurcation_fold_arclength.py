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

        # Find start solution
        problem.r.value=1
        problem.get_ode("fold").set_value(x=1)
        problem.solve()
        problem.output()

        # Initialize ds (the first step is in direction of the parameter r, i.e. we decrease r first)
        ds=-0.02

        # Scan as long as x>-1                
        while problem.get_ode("fold").get_value("x",as_float=True)>-1:
            # adjust r, solve for x along the tangent and return a good new ds
            ds=problem.arclength_continuation(problem.r,ds,max_ds=0.025)
            problem.output()


