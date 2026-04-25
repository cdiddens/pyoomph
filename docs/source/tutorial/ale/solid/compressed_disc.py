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
from pyoomph.equations.solid import *
from pyoomph.meshes.simplemeshes import CircularMesh

class CompressedDiscProblem(Problem):
    def __init__(self):
        super().__init__()
        self.Gamma=1.1 # isotropic growth factor
        self.claw=GeneralizedHookeanSolidConstitutiveLaw(E=1,nu=0.3) # Generalized Hookean solid constitutive law
        self.P=self.define_global_parameter(P=0) # Pressure on the circumference of the disc
        self.polar_implementation=True # Use radial polar coordinates only
        
    def define_problem(self):        
        # Base equations, irrespective of the coordinate system
        eqs=MeshFileOutput()        
        eqs+=DeformableSolidEquations(constitutive_law=self.claw,coordinate_space="C2",isotropic_growth_factor=self.Gamma)
        eqs+=SolidNormalTraction(self.P)@"circumference"                                        
        
        # Mesh, coordinate system and boundary conditions depending on whether we solve a 2d Cartesian or polar 1d problem
        if self.polar_implementation:
            self+=LineMesh(size=1,N=20,left_name="center",right_name="circumference") # Create a line mesh for the radial direction
            self.set_coordinate_system("axisymmetric") # Polar coordinate system
            eqs+=DirichletBC(mesh_x=0)@"center" # Fixed in the center of the disc
        else:
            # Case of Cartesian coordinates, we create a quarter circular mesh
            self+=CircularMesh(radius=1,segments=["NE"])
            eqs+=DirichletBC(mesh_x=0)@"center_to_north" # and fix the positions at the symmetry axes
            eqs+=DirichletBC(mesh_y=0)@"center_to_east"        
                                                            
        # To monitor the radius of the disc, we can use IntegralObservables. We integrate over the circumference of the disc to the the line length
        # and we also integrate over r*dl
        eqs+=IntegralObservables(_linelength=1,_radius_integral=square_root(dot(var("coordinate"),var("coordinate"))))@"circumference" 
        # The radius is then given by the ratio of the integral of r and the line length
        eqs+=IntegralObservables(radius=lambda _radius_integral,_linelength:_radius_integral/_linelength)@"circumference"   
        
        self+=eqs@"domain"
        
    
        
        
with CompressedDiscProblem() as problem:
    delta_p=0.0125
    nstep=21
     
    problem.P.value=-delta_p*(nstep-1)*0.5 # Start with a negative pressure (pulling the disc outwards)
    problem.initialise()
    problem.refine_uniformly()        
        
    # Write a comparison output file with the radius computed from the linearized analytical solution and the numerical solution
    outf=problem.create_text_file_output("disc_output.txt",header=["P","r_numeric","r_linear"])    
              
    for i in range(nstep):         
        problem.solve()
        problem.output_at_increased_time()
        rlinear=square_root(problem.Gamma)*(1-problem.P*(1+problem.claw.nu)*(1-2*problem.claw.nu))
        rnumeric=problem.get_mesh("domain/circumference").evaluate_observable("radius")
        outf.add_row(problem.P,rnumeric,rlinear)
        problem.P.value+=delta_p
        
        
   
 