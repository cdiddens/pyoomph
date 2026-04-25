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


# Load the equation and problem class (with the reaction terms)
from turing_dispersion import *
                
class TransientTuringProblem(TuringProblem):
    def __init__(self):
        super().__init__()
        self.kc=0.46 # We use the dominant wavenumber from the dispersion analysis
        self.Nperiods=4 # Number of periods per x and y direction to consider
        self.Nelem=40 # Number of elements in x and y direction    
        
    def define_problem(self):        
        # We entirely override this method
        L=2*pi/self.kc*self.Nperiods
        self+=RectangularQuadMesh(size=L,N=self.Nelem)        
        eqs=TuringEquations(self.d,self.f,self.g)
        # Periodic BCs
        eqs+=PeriodicBC("left",offset=[-L,0])@"right"
        eqs+=PeriodicBC("bottom",offset=[0,-L])@"top"
        # Some reasonable guess here (not the exact solution)
        eqs+=InitialCondition(u=(self.a + 1)/self.b, v=(self.a + 1)**2/self.b**2)
        eqs+=MeshFileOutput()
        self+=eqs@"domain"
        
with TransientTuringProblem() as problem:    
    # Solve for the stationary solution
    problem.solve()
    # Add some random perturbation to the stationary solution
    problem.perturb_dofs(numpy.random.rand(problem.ndof())*0.001)
    # Adaptive time stepping control
    problem.DTSF_max_increase_factor=1.25    
    problem.DTSF_min_decrease_factor=1.25    
    # Run it transiently, adaptive time stepping with output every step, but do not set the initial condition again
    problem.run(2000,outstep=True,startstep=0.1,temporal_error=1,do_not_set_IC=True,maxstep=100,max_newton_iterations=4)
        
        
        