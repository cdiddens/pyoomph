.. _solidoscillations:

Oscillations of a released torsion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So far, only stationary solutions of deformed solid bodies were considered. Now, we go for transient dynamics of a long three-dimensional beam. We start with a deformed configuration, specifically, by applying a torsion of the beam. On one side, the beam is fixed to a solid wall.

With the previous examples in mind, it is trivial to set up the problem case. However, here we consider physical units, i.e. we have to set typical scalings to let pyoomph nondimensionalize the equations and redimensionalize the output automatically. In particular, we need a spatial scale, a temporal scale and a scale for the mass density. With these, the equations can be nondimensionalized. A typical time scale for the solid dynamics can be obtained from Young's modulus, the density and the length:


.. code:: python

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


For the torsion, we use the original undeformed beam by accessing the Lagrangian coordinates and apply the deformation by setting an initial condition for the Eulerian mesh coordinates. Note that the code generation and compilation take a while, since the three-dimensional dynamics involves a lot of higher order tensors, inverse matrices (the contravariant metric tensor) and nonlinearities. In particular, the entries of the analytical Jacobian with respect to the moving mesh coordinates hence constitute long expressions, which bloat up the generated C code to over 3 megabytes.

..  figure:: solid_oscillations.*
	:name: figalesolidoscillations
	:align: center
	:alt: Oscillations of a beam when releasing a torsion
	:class: with-shadow
	:width: 90%

	Oscillations of a beam when releasing a torsion



.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <solid_oscillations.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    		
