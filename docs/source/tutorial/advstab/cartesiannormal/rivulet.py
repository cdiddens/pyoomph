from pyoomph import *
from pyoomph.equations.navier_stokes import *
from pyoomph.equations.ALE import *
from pyoomph.utils.dropgeom import *
        
class RivuletMesh(GmshTemplate):
    def define_geometry(self):
        self.default_resolution=0.1
        self.mesh_mode="tris"
        cl_factor=0.5 # Make it finer at the contact line
        pr=cast(RivuletProblem,self.get_problem())
        geom=DropletGeometry(volume=pi/2,rivulet_instead=True,contact_angle=pr.theta)
        p00=self.point(0,0)
        prl=self.point(-geom.base_radius,0,size=cl_factor*self.default_resolution) # Mirrored point for the circle_arc
        prr=self.point(geom.base_radius,0,size=cl_factor*self.default_resolution) # contact line
        p0h=self.point(0,geom.apex_height) # Top of the droplet

        self.circle_arc(prr,p0h,through_point=prl,name="interface")        
        self.line(prr,p00,name="substrate")
                
        self.line(p00,p0h,name="axis")
        self.plane_surface("interface","substrate","axis",name="liquid")


class RivuletProblem(Problem):        
    def __init__(self):
        super().__init__()
        # Contact angle and slip length
        self.theta=self.define_global_parameter(theta=90*degree)
        self.sliplength=self.define_global_parameter(sliplength=1)
        
    def define_problem(self):        
        self+=RivuletMesh() # Add a 2d mesh       
        
        # Assemble the equation system
        eqs=HyperelasticSmoothedMesh() # Moving mesh, Hyperelastic mesh is quite robust, since we do not remesh in this particular tutorial
        eqs+=NavierStokesEquations(dynamic_viscosity=1) # bulk flow
        # Boundary conditions:
        # Navier-slip and no penetration at the substrate
        eqs+=( NavierStokesSlipLength(sliplength=self.sliplength) + DirichletBC(velocity_y=0,mesh_y=0) )@"substrate"        
        # Free surface at the interface
        eqs+=NavierStokesFreeSurface(surface_tension=1)@"interface"        
        
        # Impose a contact angle at the contact line
        wall_normal=vector(0,1,0) # The substrate has an upwards normal (to be read as 0*e_r+1*e_z+0*e_phi)
        ninter=var("normal",domain="..") # The normal at the free surface (one domain up when evaluated at the contact line)
        wall_tangent=wall_normal*dot(wall_normal,ninter)-ninter # double cross product bac-cab rule (with dot(wall_normal,wall_normal)=1)
        wall_tangent=wall_tangent/square_root(dot(wall_tangent,wall_tangent)) # Normalize it                
        eqs+=NavierStokesContactAngle(contact_angle=self.theta,wall_normal=wall_normal,wall_tangent=wall_tangent)@"interface/substrate"  
        # Symmetry at the axis
        eqs+=DirichletBC(mesh_x=0,velocity_x=0)@"axis" 
        # Enforce the volume/area of the liquid by a pressure constraint
        eqs+=EnforceVolumeByPressure(volume=pi/4)        
        
        eqs+=MeshFileOutput()              
         # Apply the equation system to the liquid domain
        self+=eqs@"liquid"
        
       

problem=RivuletProblem() # Create the problem
# Setup the problem for k-stability analysis, we do not need an analytic Hessian, since we don't do any bifurcation tracking
problem.setup_for_stability_analysis(additional_cartesian_mode=True,analytic_hessian=False) 
# Use the SLEPc eigensolver with MUMPS
problem.set_eigensolver("slepc").use_mumps()
problem.solve() # Solve the base state
problem.save_state("start.dump") # Save the start case at 90°


# Scan the contact angle
for theta_deg in [60,90,120]: 
    problem.load_state("start.dump",ignore_outstep=True)
    problem.go_to_param(theta=theta_deg*degree,reset_pars=False)        
    # Scan the slip length (either essentially free slip or quite low slip length)
    for sl in [10000,0.01]:        
        problem.go_to_param(sliplength=sl)    
    
        outf=problem.create_text_file_output("for_"+str(round(float(problem.theta/degree)))+"_deg_SL_"+str(sl)+".txt",header=["k","Lambda"])

        for k in numpy.linspace(0.01,1.5,50):
            problem.solve_eigenproblem(1,normal_mode_k=k) # Solve the k-dependent eigenproblem
            evs=problem.get_last_eigenvalues()    
            outf.add_row(k,numpy.real(evs[0]))
    


