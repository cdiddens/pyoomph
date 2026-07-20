from pyoomph import *
from pyoomph.expressions import *

# Import the preCICE adapter of pyoomph. 
from pyoomph.solvers.precice_adapter import *

# Heat conduction equation with a source term
class HeatEquation(Equations):
    def __init__(self,f):
        super().__init__()
        self.f=f
        
    def define_fields(self):
        self.define_scalar_field("u","C2")
        
    def define_residuals(self):
        u,v=var_and_test("u")
        self.add_weak(partial_t(u),v).add_weak(grad(u),grad(v)).add_weak(-self.f,v)
        
        
class RectMeshWithCircleHole(GmshTemplate):
    def define_geometry(self):
        pr=self.get_problem()
        y_bottom, y_top = 0, 1
        x_left, x_right = 0, 2
        radius = 0.2
        self.mesh_mode="tris"
        self.default_resolution=0.1
        midpoint = self.point(0.5, 0.5)
        outer_lines=self.create_lines((x_left, y_bottom), "bottom", (x_right,y_bottom),"right",(x_right,y_top),"top",(x_left,y_top),"left")
        
        # Make non-matching meshes for testing
        circle_res=0.025 if pr.precice_participant=="Neumann" else 0.05 
        
        circle_lines=self.create_circle_lines(midpoint,radius=radius,line_name=None if pr.precice_participant=="" else "interface",mesh_size=circle_res)
        if pr.precice_participant!="Neumann":
            self.plane_surface(*outer_lines,holes=[circle_lines],name="domain")        
        if pr.precice_participant!="Dirichlet":
            self.plane_surface(*circle_lines,name="domain")

# Generic heat conduction problem. Can be run without preCICE on the full domain or as Dirichlet or Neumann participant
class HeatConductionProblem(Problem):
    def __init__(self):
        super().__init__()
        self.alpha=3 # Parameters
        self.beta=1.2    
        # Config file
        self.precice_config_file="precice-config-circle.xml"   
        
    def get_f(self):
        # Source term
        return self.beta-2-2*self.alpha
    
    def get_u_analyical(self):
        # Analytical solution
        return 1+var("coordinate_x")**2+self.alpha*var("coordinate_y")**2+self.beta*var("time")
        
    def define_problem(self):
        # Depending on the participant, set up the coupling equations
        
        # First of all, we must provide the mesh at the coupling interface to preCICE
        # If we run without preCICE, this equation part is not used, so it will be just discarded
        coupling_eqs=PreciceProvideMesh(self.precice_participant+"-Mesh")
        if self.precice_participant=="Dirichlet":
            coupling_boundary="interface"
            coupling_eqs+=PreciceWriteData(**{"Heat-Flux":grad(var("u",domain=".."))},vector_dim=2)
            coupling_eqs+=PreciceReadData(uD="Temperature")            
            coupling_eqs+=EnforcedBC(u=var("u")-var("uD"))
        elif self.precice_participant=="Neumann":
            coupling_boundary="interface"
            coupling_eqs+=PreciceWriteData(Temperature=var("u"))
            coupling_eqs+=PreciceReadData(flux="Heat-Flux",vector_dim=2)                        
            coupling_eqs+=NeumannBC(u=-dot(var("flux"),var("normal")))
        elif self.precice_participant=="":
            # If we run without preCICE, we use the full domain and have no coupling boundary
            coupling_boundary=None
        else:
            raise Exception("Unknown participant. Choose 'Dirichlet', 'Neumann' or an empty string")
        

        self+=RectMeshWithCircleHole()
        
        # Assemble the base equations
        eqs=MeshFileOutput()
        eqs+=HeatEquation(self.get_f())
        eqs+=InitialCondition(u=self.get_u_analyical())
        

        if self.precice_participant!="":        
            eqs+=coupling_eqs@coupling_boundary
        if self.precice_participant!="Neumann":            
            eqs+=DirichletBC(u=self.get_u_analyical())@["bottom","top","left","right"]
        
        # Calculate the error
        eqs+=IntegralObservables(error=(var("u")-self.get_u_analyical())**2)
        eqs+=IntegralObservableOutput()
                            
        self+=eqs@"domain"
        
        
if __name__=="__main__":
    with HeatConductionProblem() as problem:
        problem.initialise() # After this, precice_participant could have been set via command line, e.g. by  -P precice_participant=Neumann
        if problem.precice_participant=="":
            # Just run it manually without preCICE
            problem.run(1,outstep=0.1)
        else:
            # Run it with preCICE. Time stepping is taken from the config file
            problem.precice_run()