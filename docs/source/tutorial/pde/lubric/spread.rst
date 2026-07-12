.. _eqpdelubric_spread:

Spreading of a droplet
~~~~~~~~~~~~~~~~~~~~~~

We can use the same equation class to calculate the spreading of a droplet. For that, we have to switch to an axisymmetric coordinate system and also make sure that the droplet can spread at all. For the latter, we must make sure that the droplet does not end at its radius :math:`R` with :math:`h(R,t)=0`, since this would exclude any change in the height according to :math:numref:`eqpdelubricationstrong`. A conventional way to resolve this is the addition of a precursor film with a thin thickness compared to the droplet. The thickness of this film will control the spreading velocity. Additionally, one may add a disjoining pressure. Thereby, one can e.g. enforce the spreading to stop at a finite contact angle:

.. code:: python

   from lubrication import *
   		
   class DropletSpreading(Problem):	
   	def __init__(self):
   		super(DropletSpreading,self).__init__()
   		self.hp=0.0075 # precursor height
   		self.sigma=1 # surface tension
   		self.R,self.h_center=1,0.5 # initial radius and height of the droplet
   		self.theta_eq=pi/8  # equilibrium contact angle
   			
   					
   	def define_problem(self):
   		self.set_coordinate_system("axisymmetric")	
   		self.add_mesh(LineMesh(N=500,size=5)) # simple line mesh		
   		
   		h=var("h") # Building disjoining pressure
   		disjoining_pressure=5*self.sigma*self.hp**2*self.theta_eq**2*(h**3 - self.hp**3)/(3*h**6)
   		
   		eqs=LubricationEquations(sigma=self.sigma,disjoining_pressure=disjoining_pressure) # equations
   		eqs+=TextFileOutput() # output	
   		h_init=maximum(self.h_center*(1-(var("coordinate_x")/self.R)**2),self.hp) # Initial height
   		eqs+=InitialCondition(h=h_init) 
   		
   		self.add_equations(eqs@"domain") # adding the equation

   		
   if __name__=="__main__":
   	with DropletSpreading() as problem:
   		problem.run(1000,outstep=True,startstep=0.01,maxstep=10,temporal_error=1)


..  figure:: lubric_spreading.*
	:name: figpdelubricspreading
	:align: center
	:alt: Spreading of a droplet
	:class: with-shadow
	:width: 80%

	Spreading of a droplet until the equilibrium contact angle is reached, which is enforced by the disjoining pressure.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <lubrication_spreading.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
