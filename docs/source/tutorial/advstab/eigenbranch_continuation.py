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
from pyoomph.equations.navier_stokes import *
from pyoomph.equations.ALE import *
from pyoomph.utils.num_text_out import NumericalTextOutputFile
from pyoomph.meshes.meshdatacache import MeshDataCombineWithEigenfunction

class LiquidBridgeProblem(Problem):
    def __init__(self):
        super().__init__()
        # Length of the domain
        self.L=self.define_global_parameter(L=2*pi)
        self.Bo=self.define_global_parameter(Bo=0)
        self.R=1 # Radius of the cylinder
        self.Nr=6 # Number of elements in the radial direction
        
    def define_problem(self):
        # Axisymmetric problem
        self.set_coordinate_system("axisymmetric")
        
        # Calculate the number of elements in the axial direction
        aspect0=float(self.L/self.R)
        Nl=round(aspect0*self.Nr)
        
        # Add the mesh
        self+=RectangularQuadMesh(size=[self.R,self.L],N=[self.Nr,Nl])
        
        # Bulk equations are: Stokes equations, pseudo-elastic mesh, mesh file output
        eqs=StokesEquations(dynamic_viscosity=1,bulkforce=self.Bo*vector(0,-1))
        eqs+=PseudoElasticMesh()
        eqs+=MeshFileOutput(operator=MeshDataCombineWithEigenfunction(0)) # Add also the zeroth eigenfunction to the output
        
        # Boundary conditions: Axisymmetry, no-slip at the bottom, no-slip at the top, no-slip at the right, free surface at the left
        eqs+=AxisymmetryBC()@"left"
        eqs+=DirichletBC(velocity_x=0,velocity_y=0,mesh_y=0,mesh_x=True)@"bottom"
        eqs+=DirichletBC(velocity_x=0,velocity_y=0,mesh_x=True)@"top"
        # However, since we want to vary the length, we must trick a bit
        # First, we enforce that mesh_y=L at the top (i.e. we adjust mesh_y so that var("mesh_y")-self.L=0)
        eqs+=EnforcedDirichlet(mesh_y=self.L)@"top"
        # However, thereby, the Lagrange multiplier for the kinematic boundary condition is not automatically pinned to zero
        # Since mesh_y is a degree of freedom now at the right/top corner, the kinematic BC constraint is not pinned automatically
        # So we must pin it manually
        eqs+=DirichletBC(_kin_bc=0)@"right/top"
        
        # Free surface at the left
        eqs+=NavierStokesFreeSurface(surface_tension=1)@"right"
        
        # Volume constraint for the pressure to fix the volume
        Vdest=pi*self.R**2*self.L
        P,Ptest=self.add_global_dof("P",equation_contribution=-Vdest) # Subtract the desired volume
        eqs+=WeakContribution(1,Ptest) # Integrate the actually present volume, P is now determined by V_act-V_desired=0
        #eqs+=WeakContribution(P,"pressure") # And this pressure is added to the pressure field
        eqs+=AverageConstraint(_kin_bc=P)@"right" # Average the normal traction to agree with the gas pressure
        
        # Add the equations to the problem
        self+=eqs@"domain"
        
        
        
with LiquidBridgeProblem() as problem:
    
    # Generate analytically derived C code for the Hessian (for the eigenbranch tracking)
    problem.setup_for_stability_analysis(analytic_hessian=True)
    problem.set_c_compiler("system").optimize_for_max_speed()
    # Solve the base problem
    problem.solve()
    L0=float(problem.L) # Store the initial length
    minL=0.8*L0
    maxL=1.2*L0
    problem.go_to_param(L=minL) # Go to the stable length
    problem.save_state("start.dump") # Save the initial state
    
    neigen=2
    def create_Bond_curve(Bo,eigenindex,startfile, postfix,start_high_L=False):
        problem.load_state(startfile,ignore_outstep=True) # Load the initial state
        # Go to the desired Bond number and length
        problem.go_to_param(Bo=Bo)
        problem.go_to_param(L=(maxL if start_high_L else minL))
        # Create and output file for this Bond number
        curve=NumericalTextOutputFile(problem.get_output_directory("curve_Bo_"+str(Bo)+"_"+str(eigenindex)+"_"+postfix+".txt"),header=["L","ReLambda","ImLambda"])    
        # Solve the eigenproblem and add the first eigenvalue to the curve
        problem.solve_eigenproblem(neigen)        
        # We need to solve one eigenproblem only
        # Now we activate eigenbranch tracking
        problem.activate_eigenbranch_tracking(eigenvector=eigenindex)
        problem.solve() # And solve for it
        
        # Scan the curve
        curve.add_row(problem.L,numpy.real(problem.get_last_eigenvalues()[0]),numpy.imag(problem.get_last_eigenvalues()[0]))
        dL0=(maxL-minL)/20*(-1 if start_high_L else 1) # Initial step size
        dL=dL0 # Current step size
        while problem.L.value<=maxL and problem.L.value>=minL:
            # We must use arclength continuation here, since we hit fold bifurcations if Bo!=0
            dL=problem.arclength_continuation("L",dL,max_ds=dL0)        
            curve.add_row(problem.L,numpy.real(problem.get_last_eigenvalues()[0]),numpy.imag(problem.get_last_eigenvalues()[0]))
        problem.deactivate_bifurcation_tracking() # Stop the bifurcation tracking (here, eigenbranch tracking)
        
    
    # Create the Bond curve for Bo=0
    create_Bond_curve(0,0,"start.dump","std")
    # Save the end state for later (high L)
    problem.save_state("end.dump")
    # Create the Bond curve for Bo=1
    create_Bond_curve(0,1,"start.dump","std")
    
    # Now create the lower L curves for Bo=0.0025
    create_Bond_curve(0.0025,0,"start.dump","fold")
    create_Bond_curve(0.0025,1,"start.dump","fold")
    
    # And also the higher L curves for Bo=0.0025
    create_Bond_curve(0.0025,0,"end.dump","unstab",start_high_L=True)
    # Save the end state for later, when going back to Bo=0 
    problem.save_state("end2.dump")
    create_Bond_curve(0.0025,1,"end.dump","unstab",start_high_L=True)    
    
    # Now we have found a rather strange unstable branch, where the interface is not straight despite of Bo=0. 
    # Here, the two curvatures cancel each other out, but it is not stable.
    create_Bond_curve(0,0,"end2.dump","unstab")    