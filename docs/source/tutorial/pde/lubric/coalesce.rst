.. _secpdelubric_coalescence:

Coalescence of droplets
~~~~~~~~~~~~~~~~~~~~~~~

For a reasonable coalescence, we have to solve the lubrication equation on a two-dimensional lateral plane. Due to symmetry, only one half of this plane is solved. Of course, it is beneficial to use the spatial adaptivity to resolve the domain accurately and optimize the required computational effort:

.. code:: python

   from lubrication_spreading import * # Import the previous example problem
   		
   class DropletCoalescence(DropletSpreading):	
   	def __init__(self):
   		super(DropletCoalescence,self).__init__()
   		self.distance=2.5 # droplet distance
   		self.Lx=7.5
   		self.max_refinement_level=6
   			
   					
   	def define_problem(self):
   		self.add_mesh(RectangularQuadMesh(N=[10,5],size=[self.Lx,self.Lx/2],lower_left=[-self.Lx*0.5,0])) 
   		
   		h=var("h") # Building disjoining pressure
   		disjoining_pressure=5*self.sigma*self.hp**2*self.theta_eq**2*(h**3 - self.hp**3)/(3*h**6)
   		
   		eqs=LubricationEquations(sigma=self.sigma,disjoining_pressure=disjoining_pressure) # equations
   		eqs+=MeshFileOutput() # output	
   		x=var("coordinate")
   		dist1=x-vector(-self.distance/2,0) # distance to the centers of the droplets
   		dist2=x-vector(self.distance/2,0)
   		h1=self.h_center*(1-dot(dist1,dist1)/self.R**2) # height functions of the droplets
   		h2=self.h_center*(1-dot(dist2,dist2)/self.R**2)		
   		h_init=maximum(maximum(h1,h2),self.hp) # Initial height: maximum of h1, h2 and precursor
   		eqs+=InitialCondition(h=h_init) 
   		
   		eqs+=SpatialErrorEstimator(h=1) # refine based on the height field
   		
   		self.add_equations(eqs@"domain") # adding the equation

   		
   if __name__=="__main__":
   	with DropletCoalescence() as problem:
   		problem.run(1000,outstep=True,startstep=0.01,maxstep=10,temporal_error=1,spatial_adapt=1)


We just reuse the previous problem by inheritance to get access to the parameters as e.g. ``R``, ``sigma``, etc. Of course, the parameter ``distance`` and the size of the mesh ``Lx`` are additionally required. With the :py:attr:`~pyoomph.genric.problem.Problem.max_refinement_level` of the :py:class:`~pyoomph.generic.problem.Problem` base class, the maximum refinement is controlled. The rest is analogous to the previous example, however, in Cartesian coordinates with a 2d mesh and with two droplets.

..  figure:: lubric_coalescence.*
	:name: figpdelubriccoalescence
	:align: center
	:alt: Coalescence of two droplets
	:class: with-shadow
	:width: 80%

	Coalescence of two droplets.


One can rather easily add e.g. (in)soluble surfactants or a mixture composition field by adding a corresponding advection-diffusion field on the domain. When redefining the surface tension ``sigma`` to be dependent on this additional field, it is easy to reproduce *delayed coalescence* due to Marangoni dynamics. Similarly, it is also straightforward to use dimensions here and use the non-dimensionalization in pyoomph to solve the dynamics of real droplets.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <lubrication_coalescence.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
