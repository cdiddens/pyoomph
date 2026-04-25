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
# Load the Poisson equation for the previous class
from poisson_neumann import PoissonEquation,PoissonNeumannCondition

# Create a new Poisson equation that fixes the average with a Lagrange multiplier
class PoissonEquationWithNullspaceRemoval(PoissonEquation):
    def __init__(self,lagrange_multiplier,average_value,*,name="u",space="C2",source=0):
        # Initialize as before
        super(PoissonEquationWithNullspaceRemoval, self).__init__(name=name,source=source,space=space)
        # And store the lagrange multiplier reference
        self.lagrange_multiplier=lagrange_multiplier
        self.average_value=average_value # and the desired average value
        
    def define_residuals(self):
        # Add the normal Poisson residuals
        super(PoissonEquationWithNullspaceRemoval, self).define_residuals()

        # Add the contributions from the Lagrange multiplier
        l,ltest=self.lagrange_multiplier,testfunction(self.lagrange_multiplier)
        u,utest=var_and_test(self.name)
        self.add_residual(weak(u-self.average_value,ltest)+weak(l,utest))

# A single "ODE", which is used as storage for the Lagrange multiplier value
class LagrangeMultiplierForPoisson(ODEEquations):
    def __init__(self,name):
        super(LagrangeMultiplierForPoisson, self).__init__()
        self.name=name

    def define_fields(self):
        self.define_ode_variable(self.name)


class PureNeumannPoissonProblem(Problem):
    def define_problem(self):
        mesh = LineMesh(minimum=-1, size=2, N=100)
        self.add_mesh(mesh)

        # Create the Lagrange multiplier (just a single value)
        lagrange = LagrangeMultiplierForPoisson("lambda")
        lagrange += ODEFileOutput() # Output it to file as well
        self.add_equations(lagrange@"lambda_space") # And add it to a space called "lambda_space"

        l=var("lambda",domain="lambda_space") # Bind it. Important to pass the domain name here!
        # and pass it to the augmented Poisson equation
        equations = PoissonEquationWithNullspaceRemoval(l,10,source=0)
        equations += TextFileOutput() # output
        # And the Neumann conditions
        equations += PoissonNeumannCondition("u",-1) @ "left"
        equations += PoissonNeumannCondition("u",1) @ "right"
        self.add_equations(equations@"domain")


if __name__ == "__main__":
    with PureNeumannPoissonProblem() as problem:
        problem.solve()  # Solve the problem
        problem.output()  # Write output
