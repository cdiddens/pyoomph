.. _simplefsi:

Fluid-Structure Interaction
---------------------------

Having deformable solids (:numref:`secALEsolid`) and the Navier-Stokes equations on moving domains (:numref:`secALEfreesurfNS`) available, obviously, both can be combined for fluid-structure interaction scenarios.
We consider a 2d channel with two leaflets that will deform by the flow and thereby change the flow as well. The problem case is rather short, just combining the Navier-Stokes equations in the liquid domain and the solid equations in the deformable leaflets:

.. code:: python

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
		
		# Adaptivity
		leqs+=SpatialErrorEstimator(velocity=1)
		leqs+=RefineToLevel()@"liquid_solid"
		seqs+=RefineToLevel()@"liquid_solid"
		
		self+=leqs@"liquid"+seqs@"solid"
		
		
	with SimpleFSIProblem() as problem:    
	    problem.run(50,outstep=0.5,temporal_error=1,spatial_adapt=1)


The real main part is the :py:class:`~pyoomph.equations.solid.FSIConnection`, which must be added to the liquid side of the mutual interface. In :numref:`secconnectfluids`, we discussed how the enforcing of continuous velocity between two liquid domains via Lagrange multipliers actually ensures the balance of tractions. The same idea is used in the :py:class:`pyoomph.equations.solid.FSIConnection` interface. We enforce that the liquid velocity agrees with the solid velocity and thereby also ensure the balance of the tractions at the shared interface. Moreover, the fluid mesh is moved with the solid mesh. As discussed in :numref:`secmultidomheatcond`, it is important to use the same scale of the test function on both sides to balance the tractions. Therefore, one has to set ``scale_for_FSI=True`` in the :py:class:`~pyoomph.equations.solid.DeformableSolidEquations`.

As opposed to the :py:class:`~pyoomph.equations.ALE.ConnectMeshAtInterface` class, which moves the nodes of the meshes on both sides, the :py:class:`~pyoomph.equations.solid.FSIConnection` only moves the nodes of the liquid mesh to match those of the solid mesh. Otherwise, the particular moving mesh dynamics of the fluid domain, which does not reflect any physics, would add additional unphysical tractions to the system.

.. only:: html

	.. raw:: html 

		<figure class="align-center" id="vidsimplefsi"><video autoplay="True" preload="auto" width="80%" loop=""><source src="../../_static/simple_fsi.mp4" type="video/mp4"></video><figcaption><p><span class="caption-text">Fluid-Structure Interaction</span></p></figcaption></figure>
	
	
.. only:: latex

	..  figure:: simple_fsi.*
		:name: figsimplefsi
		:align: center
		:alt: Fluid-Structure Interaction
		:class: with-shadow
		:width: 80%

		Fluid-Structure Interaction



.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <simple_fsi.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   	
		    		
