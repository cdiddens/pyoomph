With Navier-slip at the substrate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As mentioned, the free slip, i.e. the unhindered :math:`x`-velocity, at the substrate is not physical. Often, one imposes no slip at solid substrates, but adding ``DirichletBC(velocity_x=0)@"substrate"`` would lead to a pinned contact line. Since the :math:`x`-velocity will be zero, the only way to fulfill the ``KinematicBC`` will be that also the mesh velocity in :math:`x`-direction will be zero. Hence, the contact line won't move. Furthermore, any contribution to the contact line boundary condition :math:`[\sigma\vec{N},\vec{v}]` would vanish in that case, since the velocity test function :math:`\vec{v}` must be zero on strongly enforced boundaries.

A solution to this dilemma is the consideration of a *slip length*, which essentially does what the precursor film has done in the corresponding lubrication theory example in :numref:`eqpdelubric_spread`: It allows for the motion of the contact line by introducing a small length scale.

The *Navier-slip* boundary condition does not set the velocity to zero at the substrate but allows a damped tangential motion. This can be done by imposing a term proportional to the tangential velocity as tangential Neumann condition, i.e. as traction, we add

.. math::

   \begin{aligned}
   \left\langle \frac{\mu}{L_\text{s}} \mathbf{P}_t \vec{u}, \mathbf{P}_t \vec{v} \right\rangle
   \end{aligned}

to the Neumann contribution, where :math:`\mathbf{P}_t=\mathbf{1}-\mathbf{nn}` is the tangential projector, :math:`L_\text{s}` is the slip length and :math:`\mu` is the dynamic viscosity in the bulk. If the fluid wants to move tangentially along the substrate, a counter-acting traction proportional to :math:`\frac{\mu}{L_\text{s}} \mathbf{P}_t \vec{u}` will hamper this motion. An implementation could read

.. code:: python

   from droplet_spread_free_slip import * # Load the problem without slip

   class SlipLength(InterfaceEquations):
    	# must be attached to a domain with NavierStokesEquations
   	required_parent_type = NavierStokesEquations

   	def __init__(self, slip_length):
   		super(SlipLength, self).__init__()
   		self.slip_length = slip_length # store the slip length

   	def define_residuals(self):
   		n = var("normal")
   		u, utest = var_and_test("velocity")
   		utang = u - dot(u, n) * n # tangential velocity
   		utest_tang = utest - dot(utest, n) * n # tangential test function
   		mu=self.get_parent_equations().dynamic_viscosity # get mu from the parent equations
   		factor = mu / (self.slip_length) # add the weak contribution
   		self.add_residual(weak(factor * utang, utest_tang))

Again, we use :py:attr:`~pyoomph.generic.codegen.InterfaceEquations.required_parent_type` in combination with :py:meth:`~pyoomph.generic.codegen.InterfaceEquations.get_parent_equations` to access the ``dynamic_viscosity`` of the bulk, i.e. of the :py:class:`~pyoomph.equations.navier_stokes.NavierStokesEquations` in the parent domain of this interface. Note that this equation does not add contributions normal to the substrate. Here, we rely that a ``DirichletBC(velocity_y=0)`` takes care of preventing any flow through the substrate in :math:`y`-direction.

To add the slip length to the problem, we use inheritance, i.e. our new problem will be the same as the old problem, except that a slip length will be additionally added to the system:

.. code:: python

   # Inherit from the problem without slip
   class DropletSpreadingWithSliplength(DropletSpreadingProblem):
   	def __init__(self):
   		super(DropletSpreadingWithSliplength,self).__init__()
   		self.slip_length=0.001 # tiny slip length
   		
   		
   	def define_problem(self):
   		super(DropletSpreadingWithSliplength,self).define_problem() # define the old problem
   		self.add_equations(SlipLength(self.slip_length)@"domain/substrate") # add a slip length to the substrate
   		
   		# Refinement
   		self.max_refinement_level=6 # level 4 is already the base refinement, allow additional refinment
   		self.add_equations(SpatialErrorEstimator(velocity=1)@"domain") # allow for refinement to resolve the strong stresses near the contact line

   		
   if __name__=="__main__":
   	with DropletSpreadingWithSliplength() as problem:
   		problem.run(50,outstep=True,startstep=0.25,spatial_adapt=1)	

Note that we anticipate high stresses near the contact line. The droplet wants to spread due to the stress stemming from the equilibrium contact angle contribution, but it will be hampered due to the slip length near the substrate. Hence, we add mesh refinement to resolve this more accurately.

A comparison of results without and with slip length can be seen on the right side of :numref:`figaledropletspread` in the previous section. The case with slip length is definitely more realistic. The smaller the slip length, the slower the spreading will take place. The slip length can hence be used to match the spreading velocity with experiments.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <droplet_spread_sliplength.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
