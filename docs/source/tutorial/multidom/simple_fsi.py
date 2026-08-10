from pyoomph import *
from pyoomph.equations.navier_stokes import *
from pyoomph.equations.ALE import *
from pyoomph.equations.solid import *

        
class SimpleFSIProblem(Problem):
    def __init__(self):
        super().__init__()
        self.Pinlet=15 # Inlet pressure
        self.claw=GeneralizedHookeanSolidConstitutiveLaw(E=40000,nu=0.3) # Solid dynamics
        self.rho, self.mu=1000 , 1 # Liquid density and dynamic viscosity        
        self.max_refinement_level=2 # Adaptivity level        
    
    def define_problem(self):
        
        # Mark individual domains in the RectangularQuadMesh
        def domain_name(x,y):
            if (x>1.8 and x<2 and y<1.5) or (x>2.8 and x<3 and y>0.5):
                return "solid"
            else:
                return "liquid"            
        self+=RectangularQuadMesh(size=[5,2],name=domain_name,N=[50,20])
        
        # Liquid equations
        leqs=MeshFileOutput()
        leqs+=NavierStokesEquations(mass_density=self.rho,dynamic_viscosity=self.mu)
        leqs+=PseudoElasticMesh()
        leqs+=PinMeshCoordinates()@["left","right"]
        leqs+=DirichletBC(mesh_y=0)@"bottom"
        leqs+=DirichletBC(mesh_y=2)@"top"
        leqs+=NoSlipBC()@["top","bottom"]
        leqs+=(DirichletBC(velocity_y=0)+NeumannBC(velocity_x=-self.Pinlet))@"left"
        leqs+=DirichletBC(velocity_y=0)@"right"
                
        # Solid equations
        seqs=MeshFileOutput()
        seqs+=DeformableSolidEquations(self.claw,mass_density=2,coordinate_space="C2",scale_for_FSI=True)
        seqs+=PinMeshCoordinates()@"bottom"
        seqs+=PinMeshCoordinates()@"top"
        
        # Fluid-structure interaction at the mutual interface
        leqs+=FSIConnection()@"liquid_solid"
        # The FSIConnection only pulls on the liquid nodes lying on the interface. Around the tips of the
        # leaflets, where the imposed motion varies quickly along the interface, the first layer of liquid
        # elements has to absorb the whole mismatch to the almost stationary interior and eventually inverts.
        # Stiffening that layer spreads the deformation over the elements further inside.
        leqs+=InterfaceMeshStiffening(5)@"liquid_solid"

        # Adaptivity. The two domains are adapted individually, but pyoomph keeps both sides of the
        # mutual interface at the same refinement, so the error estimator can drive it freely. This used
        # to require RefineToLevel()@"liquid_solid" on BOTH sides -- forcing the interface to maximum
        # refinement everywhere so that the two sides could not disagree -- which is no longer needed.
        leqs+=SpatialErrorEstimator(velocity=1)

        self+=leqs@"liquid"+seqs@"solid"
        
        
with SimpleFSIProblem() as problem:
    problem.run(50,outstep=0.5,temporal_error=1,spatial_adapt=1)
