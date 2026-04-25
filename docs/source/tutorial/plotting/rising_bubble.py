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
from pyoomph.equations.ALE import *
from pyoomph.equations.navier_stokes import *
from pyoomph.meshes.meshdatacache import MeshDataCombineWithEigenfunction
       
        
class StructuredBubbleMesh(MeshTemplate):
    # Due to the rather high Reynolds number, it is better to work with a structured adaptive mesh
    # This is a bit more work to set up, but it is worth it
    def define_geometry(self):        
        pr=cast("RisingBubbleProblem",self.get_problem())
        R=pr.R # radius of the bubble
        W=self.nondim_size(pr.W) # width of the domain        
        L_top=self.nondim_size(pr.L_top) # length of the top part of the domain
        L_bottom=self.nondim_size(pr.L_bottom) # length of the bottom part of the domain
        
        nw=int(W/R/2) # number of elements in the width
        dx=W/nw # width of the elements
        nt=int(L_top/R/2) # number of elements in the top part
        dyt=L_top/nt # height of the elements in the top part        
        nb=int(L_bottom/R/2) # number of elements in the bottom part
        dyb=L_bottom/nb # height of the elements in the bottom part
        
        dom=self.new_domain("domain")
        # Nodes at the bubble interface
        pnorth=self.add_node(0,R)
        pne=self.add_node(float(R/square_root(2)),float(R/square_root(2)))
        peast=self.add_node(R,0)
        pse=self.add_node(float(R/square_root(2)),-float(R/square_root(2)))
        psouth=self.add_node(0,-R)
        
        # And the nodes in the rectangular around it
        n0t=self.add_node(0,dyt)
        nxt=self.add_node(dx,dyt)
        nx0=self.add_node(dx,0)
        nxb=self.add_node(dx,-dyb)
        n0b=self.add_node(0,-dyb)
        # Quads to comprise the bubble
        dom.add_quad_2d_C1(pnorth,pne,n0t,nxt)
        dom.add_quad_2d_C1(pne,peast,nxt,nx0)
        dom.add_quad_2d_C1(peast,pse,nx0,nxb)
        dom.add_quad_2d_C1(pse,psouth,nxb,n0b)
        
        # Make circular segements to make a curved interface on initial refinement by oomph-lib's MacroElements
        # Also, mark the boundaries
        nthalf=self.create_curved_entity("circle_arc",pnorth,peast,center=(0,0))
        nbhalf=self.create_curved_entity("circle_arc",psouth,peast,center=(0,0))
        self.add_nodes_to_boundary("interface",[pnorth,pne])
        self.add_nodes_to_boundary("interface",[pne,peast])
        self.add_nodes_to_boundary("interface",[peast,pse])
        self.add_nodes_to_boundary("interface",[pse,psouth])
        self.add_facet_to_curve_entity([pnorth,pne],nthalf)
        self.add_facet_to_curve_entity([pne,peast],nthalf)
        self.add_facet_to_curve_entity([peast,pse],nbhalf)
        self.add_facet_to_curve_entity([pse,psouth],nbhalf)
        self.add_nodes_to_boundary("axis",[pnorth,n0t])
        self.add_nodes_to_boundary("axis",[psouth,n0b])                
        
        # Fill the strip right of the bubble to the width of the domain
        for i in range(1,nw):
            # Top part of the strip
            n00=self.add_node_unique(i*dx,0)
            n10=self.add_node_unique((i+1)*dx,0)
            n01=self.add_node_unique(i*dx,dyt)
            n11=self.add_node_unique((i+1)*dx,dyt)
            if i==nw-1:
                self.add_nodes_to_boundary("side",[n10,n11])
            dom.add_quad_2d_C1(n00,n10,n01,n11)
            # Bottom part of the strip
            n00=self.add_node_unique(i*dx,-dyb)
            n10=self.add_node_unique((i+1)*dx,-dyb)
            n01=self.add_node_unique(i*dx,0)
            n11=self.add_node_unique((i+1)*dx,0)
            if i==nw-1:
                self.add_nodes_to_boundary("side",[n10,n11])
            dom.add_quad_2d_C1(n00,n10,n01,n11)
        # Make the top part
        for j in range(1,nt):
            for i in range(0,nw):
                n00=self.add_node_unique(i*dx,j*dyt)
                n10=self.add_node_unique((i+1)*dx,j*dyt)
                n01=self.add_node_unique(i*dx,(j+1)*dyt)
                n11=self.add_node_unique((i+1)*dx,(j+1)*dyt)
                if i==nw-1:
                    self.add_nodes_to_boundary("side",[n10,n11])
                if i==0:
                    self.add_nodes_to_boundary("axis",[n00,n01])
                if j==nt-1:
                    self.add_nodes_to_boundary("top",[n01,n11])
                dom.add_quad_2d_C1(n00,n10,n01,n11)
        # Make the bottom part
        for j in range(1,nb):
            for i in range(0,nw):
                n00=self.add_node_unique(i*dx,-(j+1)*dyb)
                n10=self.add_node_unique((i+1)*dx,-(j+1)*dyb)
                n01=self.add_node_unique(i*dx,-j*dyb)
                n11=self.add_node_unique((i+1)*dx,-j*dyb)
                if i==nw-1:
                    self.add_nodes_to_boundary("side",[n10,n11])
                if i==0:
                    self.add_nodes_to_boundary("axis",[n00,n01])
                if j==nb-1:
                    self.add_nodes_to_boundary("bottom",[n00,n10])
                dom.add_quad_2d_C1(n00,n10,n01,n11)


class RisingBubbleProblem(Problem):
    def __init__(self):
        super().__init__()
        self.R=0.5
        self.Mo=6.2e-7 # Morton number selects the fluid 
        self.Bo=self.define_global_parameter(Bo=0.4) # Bond number, effectively selects the bubble size                                            
         # This helps a lot in reducing the code size: We calculate Ga from Bo with an rational exponent. 
         # Since we must separate the real and imaginary part of the azimuthal mode, this would generate a lot of code if Bo could be negative, meaning that Ga could become complex according to the definition
        self.Bo.restrict_to_positive_values()
        
        self.L_top=15/4 # Far sizes. These are considerably smaller than in the literature
        self.L_bottom=30/4
        self.W=15/4
        
        self.max_refinement_level=4 # Do not refine more than 4 times (we want to have it fast, not perfectly accurate)
    
    def define_problem(self):        
        Ga=(self.Bo**3/self.Mo)**rational_num(1,4) # Galilei number                                        
                
        self.set_coordinate_system("axisymmetric")

        self+=StructuredBubbleMesh()  # Add the mesh
        
        # Assemble the equations: First, output with eigenfunction included
        eqs=MeshFileOutput(operator=MeshDataCombineWithEigenfunction(0))
        
        # Unknown bubble velocity and bubble pressure (global degrees)
        U,Utest=self.add_global_dof("U")
        P,Ptest=self.add_global_dof("P",equation_contribution=-4/3*pi*self.R**3,initial_condition=8/self.Bo)
            
        # Bulk equations: Navier-Stokes in the co-moving frame with inertia correction of a potentially accelerating frame
        eqs+=NavierStokesEquations(dynamic_viscosity=1/Ga ,mass_density=1,gravity=vector(0,-1)*partial_t(U),mode="CR")                
        
        # Free surface with the additional pressure of the bubble and the absorbed hydrostatic pressure
        eqs+=NavierStokesFreeSurface(surface_tension=1/self.Bo,additional_normal_traction=P+var("coordinate_y"))@"interface"
        
        # Constraints fixing the bubble velocity U and the bubble pressure P        
        eqs+=WeakContribution(1/2*var("coordinate_y")**2*var("normal_y"),Utest)@"interface"        
        eqs+=WeakContribution(-dot(var("coordinate"),var("normal"))/3,Ptest)@"interface"
        
        # Boundary conditions
        eqs+=AxisymmetryBC()@"axis"        
        eqs+=DirichletBC(mesh_x=self.W,velocity_x=0,velocity_phi=0)@"side"         
        eqs+=DirichletBC(mesh_y=-self.L_bottom)@"bottom"         
        eqs+=DirichletBC(mesh_y=self.L_top,velocity_x=0,velocity_phi=0)@"top"
        eqs+=EnforcedDirichlet(velocity_y=-U)@"top" # Adjust the far field velocity
                        
        # Add a moving mesh
        eqs+=PseudoElasticMesh()
        # But pin in further away from the bubble to save degrees of freedom
        eqs+=PinWhere(mesh_x=True,mesh_y=True,where=lambda x,y : x**2+y**2>4)
        
        # Refinement strategy: Max level at the interface
        eqs+=RefineToLevel()@"interface"
        # And also, refine according velocity gradients, both for the base solution and the eigenfunction
        eqs+=SpatialErrorEstimator(velocity=1)                                                                                            
                                                            
        self+=eqs@"domain"
        
    def process_eigenvectors(self, eigenvectors):
        # This function is called whenever the eigenvectors are calculated.
        # Eigenvectors are arbitrary up to a scalar constant. 
        # We can multiply it by such a constant that the average x-displacement of the interface mesh has positive real part and zero imaginary part (on average)
        # This is optional, but makes the results more consistent, since the multiplicative constant is otherwise arbitrary
        return self.rotate_eigenvectors(eigenvectors,"domain/interface/mesh_x",normalize_amplitude=0.2,normalize_dofs=True)

if __name__=="__main__":
    with RisingBubbleProblem() as problem:
            
        # Make sure to get the most optimized code available
        problem.set_c_compiler("system").optimize_for_max_speed()
        # Use SLEPc for the eigenvalue problem, use MUMPS as linear solver, since we have constraints.
        # These have a zero diagonal and give problems in the default LU decomposition of PETSc
        problem.set_eigensolver("slepc").use_mumps()
        
        # Setup the problem for azimuthal stability analysis. We don't use the analytic Hessian, since we don't do any bifurcation tracking
        # This saves some code generation and compilations time
        problem.setup_for_stability_analysis(azimuthal_stability=True,analytic_hessian=False)
        
        # Settings
        problem.Mo=6.2e-7 # Morton number selects the fluid 
        problem.Bo.value=3 # Start at Bo=3
        BoMax=10 # Maximum Bond number
        dBond=0.25 # Step size in Bond number        
        m=1 # Azimuthal mode number
        lambd=-0.1+0.75j # Guess for the eigenvalue
                    
        # Relax to the base state, then solve for the stationary solution
        problem.run(10,startstep=0.1,outstep=False,temporal_error=1)
        problem.solve(max_newton_iterations=20,spatial_adapt=4)
        
        # Now we can start the eigenanalysis        
        outfile=problem.create_text_file_output("m1_instability.txt",header=["Bo","ReLambda","ImLambda"])
            
        # Scan the branch
        while problem.Bo.value<BoMax:                                                
            # Solve it with a shift-inverted method close to the guess
            problem.solve_eigenproblem(1,azimuthal_m=m,shift=lambd,target=lambd)
            # Refine the mesh according to the eigenfunction and recalculate the eigenproblem
            problem.refine_eigenfunction(use_startvector=True)
            # And update the eigenvalue and the eigenvector guess
            lambd=problem.get_last_eigenvalues()[0] # Update the eigenvalue for the next iteration
            # Store it to the text file
            outfile.add_row(problem.Bo,numpy.real(lambd),numpy.imag(lambd))                
            # Output the solution with eigenfunction
            problem.output_at_increased_time()
            # And continue in Bo
            problem.go_to_param(Bo=problem.Bo.value+dBond)
            
        