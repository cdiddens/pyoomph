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
from pyoomph.meshes.simplemeshes import CuboidBrickMesh
from pyoomph.expressions.units import *

class OscillatingSolidProblem(Problem):
    def __init__(self):
        super().__init__()
        self.rho=1000*kilogram/meter**3 # Density of the solid
        self.E=2.5*giga*pascal # Young's modulus of the solid
        self.nu=0.38 # Poisson's ratio of the solid        
        self.L=1*meter # Length of the beam
        self.H=5*centi*meter # thickness of the beam in the y and z direction
        self.Nh=2 # Number of elements in the y and z direction
        self.Nl=20 # Number of elements in the x direction
        self.torsion=90*degree/meter # Torsion of the beam, in torsion angle per meter
        
    def get_characteristic_time_scale(self):
        # Some typical time scale for the oscillation of the solid
        return self.L*square_root(self.rho/self.E)
    
    def define_problem(self):                        
        # Scales to nondimensionalize the equations
        self.set_scaling(spatial=self.L,mass_density=self.rho,temporal=self.get_characteristic_time_scale())
        self+=CuboidBrickMesh(size=[self.L,self.H,self.H],N=[self.Nl,self.Nh,self.Nh])
        eqs=MeshFileOutput()        
        claw=GeneralizedHookeanSolidConstitutiveLaw(E=self.E,nu=self.nu)
        eqs+=DeformableSolidEquations(constitutive_law=claw,coordinate_space="C2",mass_density=self.rho)
        # Apply the torsion to the solid by expression the mesh coordinates in terms of the torsion angle and the Lagrangian coordinates (undeformed mesh coordinates)
        X=var("lagrangian")        
        theta=self.torsion*X[0]
        eqs+=InitialCondition(mesh_y=X[1]*cos(theta)+X[2]*sin(theta),mesh_z=-X[1]*sin(theta)+X[2]*cos(theta))
        eqs+=DirichletBC(mesh_x=0,mesh_y=True,mesh_z=True)@"left" # Fix the left side of the beam to the solid wall
        self+=eqs@"domain"
        
    
with OscillatingSolidProblem() as problem:        
    T=problem.get_characteristic_time_scale()
    problem.run(10*T,outstep=0.1*T,temporal_error=1)
        