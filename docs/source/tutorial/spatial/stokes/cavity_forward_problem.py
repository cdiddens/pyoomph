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
# The tangential traction at the top boundary is a parameter that we can adjust.
T = problem.define_global_parameter(T=0)

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

# Add mesh and equations
problem += RectangularQuadMesh(N=25, name="domain")
problem += eqs@'domain'

problem.solve()

# Scan to get a curve of U vs T
out_file = problem.create_text_file_output("U_vs_T.txt", header=["T", "U"])
for T_val in numpy.linspace(0, 100, 20, endpoint=True):
    problem.go_to_param(T=T_val)
    problem.output()
    U_val = problem.get_mesh("domain").evaluate_observable("U")
    out_file.add_row(T, U_val)
