.. _secspatialzeroflowenforcing:

Enforcing zero normal flow
~~~~~~~~~~~~~~~~~~~~~~~~~~

Until now, the mesh was always aligned with the axes so that it was easy to just impose e.g. ``DirichletBC(velocity_x=0)`` to prevent any flow in the :math:`x`-direction. However, on curved interfaces of a mesh, one sometimes want to enforce that there is no in- or outflow. This occurs e.g. when one tries to enforce a kinematic boundary condition :math:`\vec{u}\cdot\vec{n}=0` on a curved quasi-stationary free surface, e.g. of a droplet. Then, one cannot fix neither ``velocity_x`` nor ``velocity_y``, but only the projection ``dot(var("velocity"),var("normal"))`` must be zero, which is one constraint for two separate degrees of freedom. In these cases, the typical approach is to use a field of Lagrange multipliers :math:`\lambda` (with test function :math:`\eta`) to enforce this constraint, one adds

.. math:: :label: eqspatialnofluxlagrange

   \left\langle \vec{u}\cdot\vec{n} ,\,\eta \right\rangle  + \left\langle \lambda ,\, \vec{n}\cdot\vec{v} \right\rangle 

to the interface residuals where the normal flow should vanish. This weak form can only be fulfilled if :math:`\vec{u}\cdot\vec{n}=0`. :math:`\lambda` is then the normal traction (:math:`\approx` pressure), required to push or pull the fluid so that the constraint is fulfilled.

We want to combine it with the dimensional equations from the previous section to illustrate how this can be done:

.. code:: python

   from stokes_dimensional import * # Import the dimensional Stokes equation from the previous section
   from pyoomph.meshes.simplemeshes import CircularMesh # Import a curved mesh


   class StokesFlowZeroNormalFlux(InterfaceEquations):
   	required_parent_type = StokesEquations # Must be attached to an interface of a Stokes equation
   	
   	def define_fields(self):
   		# Velocity space is C2, so we must create the Lagrange multipliers on the same space
   		# Note how we set the scale and the testscale here: In both cases, we absorb the test scale or the scale of the velocity
   		self.define_scalar_field("noflux_lambda","C2",scale=1/test_scale_factor("velocity"),testscale=1/scale_factor("velocity")) 
   		
   	def define_residuals(self):
   		# Binding variables
   		l,ltest=var_and_test("noflux_lambda")
   		u,utest=var_and_test("velocity")
   		n=var("normal")
   		# Add the residual: The scales will cancel out: u~U, ltest~1/U and l~1/V, utest~V
   		self.add_residual(weak(dot(u,n),ltest)+weak(l,dot(utest,n)))
   		
   	# This will be called before the equations are numbered. This is the last chance to apply any pinning (i.e. Dirichlet conditions)
   	def before_assigning_equations_postorder(self, mesh):
   		# If the velocity is entirely pinned at any node (e.g. no slip), we also have to set the Lagrange multiplier to zero
   		# This can be done with the helper function: we set noflux_lambda=0 whenever "velocity" (i.e. "velocity_x" & "velocity_y) are pinned
   		self.pin_redundant_lagrange_multipliers(mesh, "noflux_lambda", "velocity")

We introduce new interface equations ``StokesFlowZeroNormalFlux`` which must be attached to a domain having the ``StokesEquations`` in the bulk. These equations will introduce a new Lagrange multiplier field at the interface on space ``"C2"``, i.e. the same space as the velocity is defined on. Previously, we have set the scaling on a problem level, using the method :py:meth:`~pyoomph.generic.problem.Problem.set_scaling` in the :py:meth:`~pyoomph.generic.problem.Problem.define_problem` method of the :py:class:`~pyoomph.generic.problem.Problem` class. However, it is complicated for the user to set also the scales (i.e. the dimensions) for the Lagrange multipliers by hand. Instead, we see immediately from :math:numref:`eqspatialnofluxlagrange` that the test function :math:`\eta` should scale inverse to the velocity scale, i.e. :math:`\eta=\tilde{\eta}/U` to cancel out the dimensions in the first term. The Lagrange multiplier field :math:`\lambda` in the second term of :math:numref:`eqspatialnofluxlagrange` must scale as :math:`1/V` i.e. the inverse of the velocity test function scale to kill all dimensional contributions. Thereby, the nondimensionalization is automatically done and the used just have to set the velocity scale on a problem level.

Furthermore, there is one caveat here: When the interface meets with another interface where e.g. a no-slip boundary condition :math:`\vec{u}=0` is set, we run into troubles. In fact, the first term of :math:numref:`eqspatialnofluxlagrange` will be automatically zero, since :math:`\vec{u}\cdot\vec{n}=0` holds due to the no-slip condition. Hence, we do not add any contribution to the test space of the Lagrange multiplier. Also, the second term will be problematic, since the Dirichlet condition for the velocity requires that the velocity test function :math:`\vec{v}` vanishes at this intersection of the two interfaces. Therefore, the entire residual will be zero irrespectively of the local value of :math:`\lambda` here. This eventually leads to a degenerate matrix (i.e. having a zero row/column) and finding a unique solution becomes impossible.

One either can leave this caveat to the user, who has to make sure at the problem level that also :math:`\lambda=0` is imposed at these particular interface intersections. A better way, which leads to less complications, is to give this responsibility to the :py:class:`~pyoomph.generic.codegen.InterfaceEquations` class itself. The method :py:meth:`~pyoomph.generic.codegen.BaseEquations.before_assigning_equations_postorder` will be called whenever the degrees of freedom are about to be numbered internally. This is the last chance to pin individual degrees of freedom, i.e. setting :math:`\lambda=0` here. Therefore, we call the helper function :py:meth:`~pyoomph.generic.codegen.InterfaceEquations.pin_redundant_lagrange_multipliers`, which will check if indeed both degrees of freedom for the velocity are pinned. If so, we set the local value of :math:`\lambda=0` and remove it from the list of unknowns. Note that it might not always work entirely automatically, namely in the case that we e.g. have only a Dirichlet condition for the velocity in :math:`x`-direction, but not in :math:`y`-direction. Since the :math:`y`-velocity is still an unknown, the Lagrange multiplier :math:`\lambda` will not be pinned to zero by :py:meth:`~pyoomph.generic.codegen.InterfaceEquations.pin_redundant_lagrange_multipliers`. However, if the normal :math:`\vec{n}` happens to have a vanishing :math:`y`-component, the entire issue persists and is not resolved. Due to :math:`n_y=0`, :math:`u_x=0` and :math:`v_x=0`, all terms in :math:numref:`eqspatialnofluxlagrange` are again zero and the system cannot be solved for a unique value of :math:`\lambda` for this particular interface intersection. However, this rarely happens and in this case, the responsibility to treat for it is by the user.

Next, we also want to add a bulk force density :math:`\vec{f}` to the Stokes flow, so we write another bulk equation:

.. code:: python

   class StokesBulkForce(Equations):
   	def __init__(self,force_density):
   		super(StokesBulkForce, self).__init__()
   		self.force_density=force_density
   		
   	def define_residuals(self):
   		utest=testfunction("velocity")
   		self.add_residual(-weak(self.force_density,utest))

We can just use this equation to add e.g. gravity or other bulk forces to the momentum equation. Both new equations are now used in the problem class:

.. code:: python

   class NoFluxStokesProblem(Problem):
   	def __init__(self):
   		super(NoFluxStokesProblem, self).__init__()
   		self.mu=1*milli*pascal*second # dynamic viscosity
   		self.radius=1*milli*meter # the radius of the circular mesh
   		
   	def define_problem(self):
   		# Setting reasonable scales
   		self.set_scaling(spatial=self.radius,velocity=1*milli*meter/second,pressure=1*pascal)

   		# Changing to an axisymmetric coordinate system
   		self.set_coordinate_system("axisymmetric")
   		
   		# Taking the north east segment of a circle as mesh, set the radius and rename the interfaces
   		mesh=CircularMesh(radius=self.radius,segments=["NE"],straight_interface_name={"center_to_north":"axis","center_to_east":"bottom"},outer_interface="interface")
   		self.add_mesh(mesh)
   		
   		eqs=StokesEquations(self.mu) # passing the dimensional viscosity to the Stokes equations
   		eqs+=RefineToLevel(3) # Refine the mesh, which is otherwise too coarse
   		eqs+=MeshFileOutput() # and output
   				
   		#Imposing gravity as bulk force
   		rho=1000*kilogram/meter**3 # mass density
   		g=9.81*meter/second**2 # gravity
   		gdir=vector(0,-1)	# direction of the gravity
   		eqs+=StokesBulkForce(rho*g*gdir)
   		
   		#adding some artificial bulk force as well
   		f=1000*rho/second**2 * vector(-var("coordinate_y"),var("coordinate_x"))
   		eqs+=StokesBulkForce(f)
   				
   		# No slip at substrate
   		eqs+=DirichletBC(velocity_x=0,velocity_y=0)@"bottom"
   		# No flow through the axis of symmtery
   		eqs+=DirichletBC(velocity_x=0)@"axis"
   		# Use our zero flux interface
   		eqs+=StokesFlowZeroNormalFlux()@"interface"
   				
   		# Adding these equations to the default domain name "domain" of the CircularMesh above
   		self.add_equations(eqs@"domain")

We use a quarter circle mesh with the :py:class:`~pyoomph.meshes.simplemeshes.CircularMesh` class. This one has a curved interface and is hence ideal to test the no-flux condition. It is also important to set the spatial scale and typical scales for the velocity and the pressure here. Since both are not imposed strongly, it is hard to determine a good scale a priori. We just take any reasonable values (which can later on be checked by comparing with the typical orders of magnitude in the output). We switch to an axisymmetric coordinate system, so our quarter circle mesh is in fact an axisymmetric hemisphere with the :math:`y`-axis as axis of symmetry. The :py:class:`~pyoomph.meshes.simplemeshes.CircularMesh` has options to rename the interfaces, which we use here to name the interface aligned with the :math:`x`-axis as ``"bottom"``, the one aligned with the :math:`y`-axis as ``"axis"`` and the curved interface is named as ``"interface"``. Furthermore, the :py:class:`~pyoomph.meshes.simplemeshes.CircularMesh` is very coarse by default, but we can refine it three times by adding a :py:class:`~pyoomph.equations.generic.RefineToLevel` to the equation class. We add some bulk forces, i.e. the gravity :math:`-\rho g\vec{e}_y` and some artificial force to bring the fluid into motion. Since we apply a no-slip condition at the ``"bottom"`` interface, the contact line between ``"bottom"`` and ``"interface"`` is actually a case where the discussed problem with the Lagrange multiplier of the ``StokesFlowZeroNormalFlux`` class arises. However, the implemented method :py:meth:`~pyoomph.generic.codegen.BaseEquations.before_assigning_equations_postorder` will take care of this and pin the local Lagrange multiplier at this point automatically.

The run code is trivial:

.. code:: python

   if __name__ == "__main__":		
   	with NoFluxStokesProblem() as problem: 
   		problem.solve() 
   		problem.output()

The results are shown in :numref:`figspatialstokesnoflux`. It is apparent how the spatially varying body force drives the flow. At the curved interface, the action of the no-flux condition prevents any in-/outflow, but allows for free tangential flow.

..  figure:: stokes_noflux.*
	:name: figspatialstokesnoflux
	:align: center
	:alt: Stokes flow with zero normal flux
	:class: with-shadow
	:width: 70%

	Velocity and pressure field of the Stokes flow with enforced zero outflow at the curved interface and bulk force driving the flow.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <stokes_no_normal_flow.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
