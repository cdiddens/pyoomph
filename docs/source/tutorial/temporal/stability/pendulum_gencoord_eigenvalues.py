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


from pyoomph import * # Import pyoomph 
from pyoomph.expressions import * # Import some additional things to express e.g. partial_t

class PendulumEquations(ODEEquations):
    # Lets assume g=L=1
    def define_fields(self):
        self.define_ode_variable("phi","psi") # angle and angular velocity

    def define_residuals(self):
        phi,phi_test=var_and_test("phi")
        psi, psi_test = var_and_test("psi")
        self.add_residual((partial_t(psi)+sin(phi))*phi_test) # psi'=phi''=-sin(phi)
        self.add_residual((partial_t(phi)-psi) * psi_test) # psi=dot(phi)


class PendulumProblem(Problem):
    def define_problem(self):
        eqs=PendulumEquations() #No output or initial condition required
        self.add_equations(eqs@"pendulum")

    # A function to investigate the stability of solutions
    def investigate_stability_close_to(self,phi_guess):
        ode=self.get_ode("pendulum")
        ode.set_value(phi=phi_guess,psi=0) # set the guess
        self.solve() # stationary solve
        eigvals,eigvects=self.solve_eigenproblem(2) # get eigenvectors
        phi_in_terms_of_pi=(ode.get_value("phi")/pi).evalf() # output phi as multiple as pi
        print(f"Eigenvalues at phi={phi_in_terms_of_pi}*pi are {eigvals[0]} and {eigvals[1]}")

if __name__=="__main__":
    with PendulumProblem() as problem:
        problem.quiet()
        problem.investigate_stability_close_to(phi_guess=0.01) # pendulum is almost hanging straight down
        problem.investigate_stability_close_to(phi_guess=0.9*pi)  # pendulum is almost at the apex