#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2025  Christian Diddens & Duarte Rocha
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


# Same Lorenz system as before
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


# Problem as before
class LorenzProblem(Problem):    
    def define_problem(self):
        eqs=LorenzSystem()
        eqs+=InitialCondition(x=-1.49160012010492360e+00, y=-7.07637299749291571e-01, z=2.07295045876768036e+01)  # Some non-trivial initial position
        #eqs+=TemporalErrorEstimator(x=1,y=1,z=1) # Weight all temporal error with unity
        eqs+=ODEFileOutput()  
        self+=eqs@"lorenz_attractor"


if __name__=="__main__":
    
    # Import the LyapunovExponentCalculator from the utils module
    from pyoomph.utils.lyapunov import LyapunovExponentCalculator
    
    with LorenzProblem() as problem:
        # We want to save memory, since we have a fine temporal discretization. 
        # So we do not write state files for continue simulations
        problem.write_states=False 
        # Add the LyapunovExponentCalculator to the problem. 
        # Calculating k=3 Lyapunov exponents. Starting after t=10, then relaxing perturbation vectors until t=10+5
        # Then start the actual Lyapunov exponent calculation                
        problem+=LyapunovExponentCalculator(k=3,waiting_time=10,prerelaxation_time=10,store_as_eigenvectors=False,use_crank_nicholson_integration=False)
        # Run it with a rather fine time step 
        problem.run(endtime=200,outstep=0.001)        
