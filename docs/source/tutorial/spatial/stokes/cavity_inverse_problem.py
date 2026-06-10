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
# Use the predefined NavierStokesEquations
from pyoomph.equations.navier_stokes import *

# Create a problem and define it in the run script directly instead of using inheritance
problem = Problem()

#T = problem.define_global_parameter(T=0) # T is not a parameter anymore
Udesired=problem.define_global_parameter(Udesired=1) # Some value we want to reach for the rms-average velocity U=1
# T is now an unknown, adjusted to reach the desired U, so we define it as a dof instead of a parameter
# We must pass a reasonable initial condition for T to assist Newton's method for convergence
T,Ttest=problem.add_global_dof("T",initial_condition=10)

# Assemble the equations
eqs = NavierStokesEquations(mass_density=10, dynamic_viscosity=1)
eqs += MeshFileOutput()
eqs += NoSlipBC()@["left", "right", "bottom"]
eqs += (DirichletBC(velocity_y=0)+NeumannBC(velocity_x=T))@"top"
# Since we have no inflow/outflow, we need to fix the pressure, since p->p+const is a nullspace transformation.
eqs += AverageConstraint(pressure=0)

# To monitor the induced velocity, we define IntegralObservables
# First we calculate the integral of u^2 over the domain
# We also need the area of the domain, which is 1 in this case, but we calculate via integral of 1, which is relevant for more complicated domain
# Finally, U is given by sqrt(integral(u^2)/Area)
eqs += IntegralObservables(U_sqr=dot(var("velocity"), var("velocity")),
                           Area=1, U=lambda U_sqr, Area: square_root(U_sqr)/Area)


# We want to adjust T, so that the residual R=integral(u^2) - Udesired^2 = 0
# So we just add the contribution integral(u^2) - Udesired^2 to the residual of T
eqs += WeakContribution(dot(var("velocity"), var("velocity")) - Udesired*Udesired, Ttest)

# Add mesh and equations
problem += RectangularQuadMesh(N=25, name="domain")
problem += eqs@'domain'

# Newton's method cannot start with a zero velocity field, since u^2 will produce an empty Jacobian row.
# So we first have to solve the problem with T fixed, to get a reasonable starting guess for the velocity field
with problem.select_dofs() as dofs:
    # Remove T from the dofs, so that it is fixed in the first solve
    dofs.unselect("globals/T") 
    # Solve the velocity and pressure for this fixed T
    problem.solve()
    # When leaving the 'with' block, T is an unknown again, and we can solve the full problem to adjust T to reach the desired U

# Solve to adjust T to reach the desired U
problem.solve()

# Output results
print("To reach U=", Udesired.value, "we need T=", problem.get_ode("globals").get_value("T"))
print("U (calculated)=", problem.get_mesh("domain").evaluate_observable("U"))

