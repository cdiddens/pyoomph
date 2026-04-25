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

from adaptive_lorenz_attractor import * # import the Lorenz problem

# Simple Lorenz system where all parameters can be changed at runtime
class LorenzBifurcationProblem(Problem):
    def __init__(self):
        super(LorenzBifurcationProblem, self).__init__()
        self.rho=self.define_global_parameter(rho=0)
        self.sigma = self.define_global_parameter(sigma=0)
        self.beta = self.define_global_parameter(beta=0)

    def define_problem(self):
        ode=LorenzSystem(sigma=self.sigma,beta=self.beta,rho=self.rho)
        self.add_equations(ode@"lorenz")


if __name__=="__main__":
    with LorenzBifurcationProblem() as problem:
        problem.quiet() # shut up and use the symbolical Hessian terms
        problem.setup_for_stability_analysis(analytic_hessian=True)
        
        # Start near the pitchfork at rho=1
        problem.rho.value=0.5
        problem.sigma.value=10
        problem.beta.value=8/3
        problem.solve(timestep=0.1) # Get the initial solution (trivial solution here)
        problem.solve()
        problem.solve_eigenproblem(0) # get an eigenvector as guess

        # Find the pitchfork in terms of rho
        problem.activate_bifurcation_tracking(problem.rho,"pitchfork",eigenvector=problem.get_last_eigenvectors()[0])
        problem.solve()
        x,y,z=problem.get_ode("lorenz").get_value(["x","y","z"])
        print(f"Pitchfork starts at rho={problem.rho.value}, x,y,z={x,y,z}")

        # this will be now the critical eigenvector at the bifurcation
        perturb=numpy.real(problem.get_last_eigenvectors()[0])
        # deactivate bifurcation tracking: Solve again the normal Lorenz system
        problem.deactivate_bifurcation_tracking()

        problem.perturb_dofs(perturb) # Go in the direction of the critical eigenvector
        problem.rho.value+=0.1 # and go a bit higher with the rho value
        problem.solve(timestep=[0.1,1,2,None]) # do a few time steps and then a stationary solve (timestep=None)
        eigvals, eigvects=problem.solve_eigenproblem(0) # get the initial eigenvalues

        # Scan rho to the Hopf bifurcation
        ds=0.001
        while eigvals[0].real<-0.001:
            ds=problem.arclength_continuation(problem.rho,ds)
            x, y, z = problem.get_ode("lorenz").get_value(["x", "y", "z"])
            eigvals, eigvects = problem.solve_eigenproblem(0)
            print(f"On pitchfork branch rho={problem.rho.value}, x,y,z={x, y, z}, eigenvalue={eigvals[0]}")

        # Jump on the Hopf bifurcation
        problem.activate_bifurcation_tracking(problem.rho,"hopf",eigenvector=problem.get_last_eigenvectors()[0],omega=numpy.imag(problem.get_last_eigenvalues()[0]))
        problem.solve()
        x, y, z = problem.get_ode("lorenz").get_value(["x", "y", "z"])
        print(f"On Hopf branch rho={problem.rho.value}, x,y,z={x, y, z}, omega={numpy.imag(problem.get_last_eigenvalues()[0])}")

        # Go down with sigma but staying on the Hopf bifurcation (i.e. do not call deactivate_bifurcation_tracking)
        ds=-0.001
        while problem.sigma.value>2+problem.beta.value:
            ds=problem.arclength_continuation(problem.sigma,ds,max_ds=0.1)
            x, y, z = problem.get_ode("lorenz").get_value(["x", "y", "z"])
            print(f"On Hopf branch rho,sigma={problem.rho.value,problem.sigma.value}, x,y,z={x, y, z}, omega={numpy.imag(problem.get_last_eigenvalues()[0])}")
