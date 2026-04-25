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
from pyoomph.equations.navier_stokes import * # Navier-Stokes for the flow
from pyoomph.equations.ALE import * # Moving mesh equations
from pyoomph.meshes.remesher import Remesher2d # Remeshing
from pyoomph.utils.num_text_out import NumericalTextOutputFile # Tool to write a file with numbers

# Make a mesh of a hanging hemispherical droplet with radius 1
class HangingDropletMesh(GmshTemplate):
    def define_geometry(self):
        self.default_resolution=0.05
        self.mesh_mode="tris"
        self.create_lines((0,-1),"axis",(0,0),"wall",(1,0))
        self.circle_arc((0,-1),(1,0),center=(0,0),name="interface")
        self.plane_surface("axis","wall","interface",name="droplet")
        self.remesher=Remesher2d(self) # attach a remesher


class HangingDropletProblem(Problem):
    def __init__(self):
        super().__init__()
        # Initially no gravity
        self.Bo=self.define_global_parameter(Bo=0)        
        # Initial volume of a hemispherical droplet with radius R=1
        self.V=self.define_global_parameter(V=2*pi/3)

    def define_problem(self):
        self.set_coordinate_system("axisymmetric") # axisymmetry 
        self+=HangingDropletMesh() # Add the mesh

        # Bulk equations: Navier-Stokes + Moving Mesh
        eqs=MeshFileOutput() # Paraview output
        # Mass density can be set arbitrarly with the same results. But must be >0 to get a du/dt -> mass matrix
        eqs+=NavierStokesEquations(dynamic_viscosity=1,mass_density=1,bulkforce=self.Bo*vector(0,-1))
        eqs+=LaplaceSmoothedMesh()

        # Boundary conditions:
        eqs+=(DirichletBC(mesh_y=0,mesh_x=True)+NoSlipBC())@"wall" # static no-slip wall
        eqs+=AxisymmetryBC()@"axis"
        eqs+=NavierStokesFreeSurface(surface_tension=1)@"interface" # free surface

        # Remeshing
        eqs+=RemeshWhen(RemeshingOptions())
        
        # For stationary solutions, we must enforce the droplet volume to be the given one
        # Add a global Lagrange multiplier -> something like the gas pressure that acts to ensure the volume
        # we want to solve p_gas by integral_droplet(1*dx)-Paramter_V=0, so we subtract the symbolic volume parameter
        self+=GlobalLagrangeMultiplier(p_gas=-self.V,only_for_stationary_solve=True)@"globals"
        p_gas=var("p_gas",domain="globals") # bind the gas pressure
        # And rewrite 1*dx=div(x)/3*dx=1/3*x*n*dS, add it to the p_gas equation to complete it
        eqs+=WeakContribution(1/3*var("mesh"),var("normal")*testfunction(p_gas))@"interface"
        # p_gas must now act somewhere. We just let it act on the contact line, and only if solved stationary
        # The kinematic BC at the rest of the interface will adjust accordingly
        eqs+=EnforcedBC(pressure=p_gas,only_for_stationary_solve=True)@"interface/wall"        

        # Add the equation system to the droplet
        self+=eqs@"droplet"


if __name__=="__main__":
    with HangingDropletProblem() as problem:
        # Calculate the Hessian symbolically/analytically -> faster and more accurate than finite differences
        problem.setup_for_stability_analysis()
        problem.do_call_remeshing_when_necessary=False # Don't auto-remesh. Can be problematic during continuation

        # Increase Bond number towards the bifurcation, do remeshing if necessary during that
        problem.go_to_param(Bo=2.8, call_after_step=lambda a : problem.remesh_handler_during_continuation())
        problem.force_remesh() # Force a new mesh
        problem.solve() # and resolve (mainly for the hydrostatic pressure and small shape adjustments of the new mesh)
        
        # Solve the eigenvalues to get a good guess for the critical eigenfunction
        problem.solve_eigenproblem(5)
        # Looking for a fold bifurcation when the droplet pinches off
        problem.activate_bifurcation_tracking("Bo","fold")
        # and solve for it. Bo will be adjusted to the critical Bond number at the initial volume
        problem.solve()

        # Create an output file for the curve Bo_c(V)
        critical_curve_out=NumericalTextOutputFile(problem.get_output_directory("critical_curve.txt"))
        critical_curve_out.header("V","Bo_c") # header line
        critical_curve_out.add_row(problem.V.value,problem.Bo.value) # V and Bo_c -> file
        problem.output_at_increased_time() # also Paraview output

        # Increase the volume, still tracking for the critical Bond number:
        dV=0.1*problem.V.value
        while problem.V.value<5:
            dV=problem.arclength_continuation("V",dV,max_ds=0.1*problem.V.value) 
            problem.remesh_handler_during_continuation()
            critical_curve_out.add_row(problem.V.value,problem.Bo.value)
            problem.output_at_increased_time()


        
        


