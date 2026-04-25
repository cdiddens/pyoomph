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

from langford_floquet import *
        
if __name__=="__main__":
     with LangfordProblem() as problem:
        # Use again an analytic Hessian for the determination of the first Lyapunov coefficient
        problem.setup_for_stability_analysis(analytic_hessian=True)        
        # We also need the SLEPc eigensolver here
        problem.set_eigensolver("slepc").use_mumps() 
        
        problem+=InitialCondition(x=0.01,z=1.1)@"langford"  # Some non-trivial initial position        
        
        # Find the Hopf bifurcation as usual
        problem.solve()
        problem.solve_eigenproblem(3)
        problem.activate_bifurcation_tracking("mu")
        problem.solve()                
        
        # Switch again to the orbits originating from the Hopf bifurcation
        with problem.switch_to_hopf_orbit(NT=50,order=3) as orbit:          
                ds=orbit.get_init_ds()       
                maxds=ds*100 # Limit the maximum step size
                problem.go_to_param(mu=2.005,startstep=ds,max_step=maxds,call_after_step= lambda ds: orbit.output_orbit("orbit_at_mu_"+str(problem.mu.value)))                
                T=orbit.get_T() # Get the period
                NT=orbit.get_num_time_steps() # Get the number of time steps
                dt=T/NT # And calculate a good time step
                
        # Running a transient integration starting on the orbit
        problem.run(40*T,outstep=dt/4)
                        
                        