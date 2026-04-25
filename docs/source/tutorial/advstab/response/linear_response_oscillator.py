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
from pyoomph.expressions.units import *
# Load tools for periodic driving response and text file output
from pyoomph.utils.periodic_driving_response import *
from pyoomph.utils.num_text_out import *

# Driven damped harmonic oscillator
class DampedHarmonicOscillatorEquations(ODEEquations):
    def __init__(self,omega0,delta,driving):
        super().__init__()
        self.omega0,self.delta,self.driving=omega0,delta,driving

    def define_fields(self):
        # Must be formulated first order here
        self.define_ode_variable("y",testscale=scale_factor("temporal")**2/scale_factor("y"))
        self.define_ode_variable("ydot",scale=scale_factor("y")/scale_factor("temporal"),testscale=scale_factor("temporal")/scale_factor("y"))        

    def define_residuals(self):
        y,y_test=var_and_test("y")
        ydot,ydot_test=var_and_test("ydot")
        self.add_weak(partial_t(y)-ydot,ydot_test)
        self.add_weak(partial_t(ydot)+self.delta*ydot +self.omega0**2*y-self.driving,y_test)
        

class DampedHarmonicOscillatorProblem(Problem):
    def __init__(self):
        super().__init__()
        self.omega0=1/second
        self.delta=0.1/second
        # Default driving
        self.driving=meter/second**2 *cos(0.3/second*var("time"))

    def define_problem(self):
        self.set_scaling(y=2*meter,temporal=1*second)
        eqs=DampedHarmonicOscillatorEquations(self.omega0,self.delta,self.driving)
        eqs+=ODEFileOutput()
        self+=eqs@"oscillator"

with DampedHarmonicOscillatorProblem() as problem:
    if False: 
        # Trivial, but long way: integrate in time, extract response manually from the output        
        problem.run(100*second,outstep=0.1*second)
    else: 
        # Quick way of scanning
        # Create the PeriodicDrivingResponse before the problem is initialized        
        pdr=PeriodicDrivingResponse(problem)

        F=1*meter/second**2 # Driving amplitude, does not really matter, will cancel out in the normalized response 
        problem.driving=F*pdr.get_driving_mode() # means F*cos(omega*t)

        # solve for a stationary state
        problem.solve()

        # Get the equation index to y
        dofindices,dofnames=problem.get_dof_description()
        yindex=numpy.argwhere(dofindices==dofnames.index("oscillator/y"))[0,0]

        # Factor to absorb the dimensions. We want to have response amplitude divided by driving at the end
        response_dim_factor=F/meter

        # Scan the frequency and write output
        outfile=NumericalTextOutputFile(problem.get_output_directory("response.txt"))
        outfile.header("omega[1/s]","(A/F)_num[m/(m/s^2)]","phi_num","(A/F)_ana[m/(m/s^2)]")

        omegas=numpy.linspace(0.01,3,300)
        for response in pdr.iterate_over_driving_frequencies(omegas=omegas,unit=1/second):        
            response_ampl,phi=pdr.split_response_amplitude_and_phase() # nondimensional response amplitude and angle            
            omega=pdr.get_driving_omega() # current omega
            # redimensionalize the response amplitude and divide by the driving, afterwards nondimensionalize
            A_num=response_ampl[yindex]*problem.get_scaling("y")/F*response_dim_factor
            # Analytic solution
            A_analytic=1/square_root((problem.omega0**2-omega**2)**2+(problem.delta*omega)**2)*response_dim_factor
            phi_analytic=atan2(-problem.delta*omega,problem.omega0**2-omega**2)            
            # outpuf
            outfile.add_row(omega*second,A_num,phi[yindex],A_analytic,phi_analytic)

        
