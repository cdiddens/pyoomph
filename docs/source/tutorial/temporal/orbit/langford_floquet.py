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

#See https://arxiv.org/abs/2407.18230v1
class LangfordSystem(ODEEquations):
     def __init__(self,mu): 
             super(LangfordSystem,self).__init__()
             self.mu=mu

     def define_fields(self):
             self.define_ode_variable("x","y","z")

     def define_residuals(self):
             x,y,z=var(["x","y","z"])             
             xrhs=(self.mu-3)*x-0.25*y+x*(z+0.2*(1-z**2))
             yrhs=(self.mu-3)*y+0.25*x+y*(z+0.2*(1-z**2))
             zrhs=self.mu*z-(x**2+y**2+z**2)
             residual=(partial_t(x)-xrhs)*testfunction(x)
             residual+=(partial_t(y)-yrhs)*testfunction(y)
             residual+=(partial_t(z)-zrhs)*testfunction(z)
             self.add_residual(residual)
                
    
class LangfordProblem(Problem):
    def __init__(self):
         super().__init__()
         self.mu=self.define_global_parameter(mu=1.6)
         
    def define_problem(self):
        eqs=LangfordSystem(self.mu) 
        eqs+=ODEFileOutput()
        self.add_equations(eqs@"langford")

    def get_analytical_nontrivial_floquet_multiplier(self):
        # Calculate the analytical nontrivial Floquet multiplier
        muv=self.mu.value
        z=2.5*(1-numpy.sqrt(0.8*muv-1.24))
        r=numpy.sqrt(z*(muv-z))
        exponent=(muv-2*z+numpy.emath.sqrt((muv-2*z)**2-8*r*(r-0.4*r*z)))/2
        T=4*2*numpy.pi
        multiplier=numpy.exp(exponent*T)
        if numpy.imag(multiplier)<0:
            multiplier=numpy.conjugate(multiplier) # We always consider the one with positive imaginary part
        return multiplier
        
        
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
        
        # Output file to compare the numerical and analytical Floquet multipliers
        floquet_output=problem.create_text_file_output("floquet.txt",header=["mu","num_real","num_imag","ana_real","ana_imag"])        
        
        # Switch again to the orbits originating from the Hopf bifurcation
        with problem.switch_to_hopf_orbit(NT=50,order=3) as orbit:          
                ds=orbit.get_init_ds()       
                maxds=ds*100 # Limit the maximum step size
                while problem.mu.value<2.05:
                        ds=problem.arclength_continuation("mu",ds,max_ds=maxds)                      
                        F=orbit.get_floquet_multipliers(n=3,shift=3) # Calculate some Floquet multipliers 
                        # However, not always three multipliers are found. We have to consider the cases                                                             
                        if len(F)==3:
                                # Three multipliers found: The trivial one and two complex conjugate ones
                                F=numpy.delete(F,numpy.argmin(numpy.abs(F-1)))
                                nontrivial_floquet=F[0] # Take one of the complex conjugate multipliers
                        elif len(F)==2:
                                # Only two multipliers found: The trivial one and one real one
                                F=numpy.delete(F,numpy.argmin(numpy.abs(F-1)))
                                nontrivial_floquet=F[0] # Take the remaining multiplier                         
                        else:
                                # Only one multiplier found: The trivial one
                                nontrivial_floquet=0 # The others are then very close to 0
                                
                        if numpy.imag(nontrivial_floquet)<0:
                                # conjugate a multiplier with negative imaginary part
                                nontrivial_floquet=numpy.conjugate(nontrivial_floquet)
                                
                        # Output the orbit
                        odir="orbit_{:.3f}".format(problem.mu.value)
                        orbit.output_orbit(odir)
                        
                        # Write to output for comparison
                        floq_ana=problem.get_analytical_nontrivial_floquet_multiplier()
                        floquet_output.add_row(problem.mu, nontrivial_floquet.real,nontrivial_floquet.imag,floq_ana.real,floq_ana.imag)
                        
                        
                        