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
from pyoomph.equations.solid import * # nonlinear solid equations


class AiryCantileverProblem(Problem):
    def __init__(self):
        super().__init__()
        self.H=0.5 # half height of the cantilever
        self.L=10 # length of the cantilever
        self.E=1 # Young's modulus
        self.nu=0.3 # Poisson's ratio
        self.G=0 # Optional gravity vector (not considered here)
        self.P=self.define_global_parameter(P=0) # Pressure on the top of the cantilever
        
    def define_problem(self):        
        self+=RectangularQuadMesh(size=[self.L,2*self.H],N=[20,2],lower_left=[0,-self.H])
        
        eqs=MeshFileOutput()
        if True:
            # Take the constitutive law for a generalized Hookean solid (compressible)
            claw=GeneralizedHookeanSolidConstitutiveLaw(E=self.E,nu=self.nu)
        else:
            # Alternatively, take the constitutive law for an incompressible solid
            # This will introduce a pressure variable to ensure incompressibility, but we must remove the null space of the pressure variable 
            claw=IncompressibleHookeanSolidConstitutiveLaw(E=self.E)
            eqs+=AverageConstraint(pressure=0) # remove the null space of the pressure variable
            
        eqs+=DeformableSolidEquations(constitutive_law=claw,mass_density=0,bulkforce=vector(0,self.G),coordinate_space="C2",pressure_space="DL",with_error_estimator=True)
        eqs+=DirichletBC(mesh_x=0,mesh_y=True)@"left" # Fix the left side of the cantilever
        eqs+=SolidNormalTraction(self.P)@"top" # Apply pressure on the top of the cantilever
        
        # To compare the numerical solution with the analytical solution, we write the numerical and analytical stress tensors
        eqs+=LocalExpressions(sigma_num=claw.get_sigma(2,pressure_var=var("pressure"))) # Numerical stress tensor
        # Constants for exact (St. Venant) solution
        a=-1.0/4.0*self.P
        b=-3.0/8.0*self.P/self.H
        c=1.0/8.0*self.P/self.H**3
        d=1.0/20.0*self.P/self.H
        xi=var("lagrangian")
        xx=self.L-xi[0]
        yy=xi[1] 
        # Approximate analytical (St. Venant) solution of the stress tensor
        sigma=matrix([[c*(6.0*xx*xx*yy-4.0*yy*yy*yy)+6.0*d*yy, 2.0*(b*xx+3.0*c*xx*yy*yy)],[2.0*(b*xx+3.0*c*xx*yy*yy),2.0*(a+b*yy+c*yy*yy*yy)]])
        eqs+=LocalExpressions(sigma_ana=sigma) # approximate analytical stress tensor
          
        self+=eqs@"domain"
        
        
with AiryCantileverProblem() as problem:
     max_adapt=3
     nstep=5
     p_increment=1.0e-5
     
     problem.initialise()
     problem.refine_uniformly()
          
     for i in range(nstep): 
        problem.P.value+=p_increment
        problem.solve(spatial_adapt=max_adapt)
        problem.output_at_increased_time()
        
        
   
 