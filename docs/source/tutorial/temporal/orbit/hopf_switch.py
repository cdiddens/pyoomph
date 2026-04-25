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

from pyoomph import * 
from pyoomph.expressions import * 

# Lorenz system as before
class LorenzSystem(ODEEquations):
	def __init__(self,*,sigma=10,rho=28,beta=8/3,scheme="BDF2"): # Default parameters used by Lorenz
		super(LorenzSystem,self).__init__()
		self.sigma=sigma
		self.rho=rho
		self.beta=beta
		self.scheme=scheme

	def define_fields(self):
		self.define_ode_variable("x","y","z") 
	
	def define_residuals(self):
		x,y,z=var(["x","y","z"])
		residual=(partial_t(x)-self.sigma*(y-x))*testfunction(x)
		residual+=(partial_t(y)-x*(self.rho-z)+y)*testfunction(y)
		residual+=(partial_t(z)-x*y+self.beta*z)*testfunction(z)
		self.add_residual(time_scheme(self.scheme,residual))


# Lorenz problem as before
class LorenzProblem(Problem):
    def __init__(self):
        super(LorenzProblem, self).__init__()
        self.rho=self.define_global_parameter(rho=0)
        self.sigma = 10
        self.beta = 8/3

    def define_problem(self):
        ode=LorenzSystem(sigma=self.sigma,beta=self.beta,rho=self.rho)
        ode+=ODEFileOutput()
        self.add_equations(ode@"lorenz")


if __name__=="__main__":
    with LorenzProblem() as problem:
        # To calculate c_1, we need the Hessian, so the symbolical code must be generated and compiled
        problem.setup_for_stability_analysis(analytic_hessian=True)
        # Add a non-trivial initial condition
        problem+=InitialCondition(x=1,z=24)@"lorenz"
        problem.rho.value=24 # Start close to the Hopf
        
        problem.solve() # Find a stationary solution (will be on one of the pitchfork branches)        
        problem.solve_eigenproblem(n=1) # And get some eigenvalue for the Hopf tracker
        
        problem.activate_bifurcation_tracking("rho","hopf") # Activate the Hopf tracking
        problem.solve() # Find the Hopf bifurcation by adjusting rho
        
        # Since we are on the Hopf bifurcation, we can switch to the orbit
        # We chose NT=100 time points for the orbit
        # The initial period T and the initial guess of the orbit will be calculated automatically
        with problem.switch_to_hopf_orbit(NT=100) as orbit:
            print("Bifurcation is supercritical: "+str(orbit.starts_supercritically()))
            print("Period at rho=",problem.rho.value, " is ",orbit.get_T())
            # This function will write the output along the orbit to a subdirectory in the output directory
            orbit.output_orbit("orbit_at_rho_{:.4f}".format(problem.rho.value))
            # Perform continuation in rho
            # We do not know in which direction we have to go (depends on the nature of the Hopf)
            # But a good guess including direction can be obtained from the Lyapunov coefficient
            ds=orbit.get_init_ds() 
            while problem.rho.value>16:
                ds=problem.arclength_continuation("rho",ds)
                print("Period at rho=",problem.rho.value, " is ",orbit.get_T())
                orbit.output_orbit("orbit_at_rho_{:.4f}".format(problem.rho.value))            
                