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


# Load the previous code
from poisson_2d import *

# Inherit from the previous problem
class AdaptivePoissonProblem2d(PoissonProblem2d):
    def define_problem(self):
        # define the previous problem
        super(AdaptivePoissonProblem2d, self).define_problem()

        # add a spatial error estimator for u (errors weighted by the coefficient 1.0)
        additional_equations = SpatialErrorEstimator(u=1.0)
        self.add_equations(additional_equations @ "domain")


if __name__=="__main__":
    with AdaptivePoissonProblem2d() as problem:
        # Maximum refinement level
        problem.max_refinement_level = 5
        # Refine elements with error larger than that
        problem.max_permitted_error = 0.0005
        # Unrefine elements with elements smaller than that
        problem.min_permitted_error = 0.00005
        # Solve with full refinement
        problem.solve(spatial_adapt=problem.max_refinement_level)
        problem.output()