.. _secspatialstokes_law:

Stokes' law - Obtaining forces by traction integrals and using global Lagrange multipliers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Stokes' law* describes the flow field around a spherical rigid object. Here, we consider a solid sphere sinking down due to buoyancy. Obviously, treating this in the lab frame would require to solve for the motion of the object directly, which can be done with a moving mesh method as described later on in :numref:`secALE`. However, we can also transform into the coordinate system co-moving with the object. In this case, a fixed mesh can be used.

Stokes' law states that the terminal velocity will be given by

.. math:: :label: eqspatialstokeslawuterm

   U=\frac{2}{9}\frac{\rho_\text{o}-\rho_\text{f}}{\mu}gR^2\,,

where :math:`\rho_\text{o}` and :math:`\rho_\text{f}` are the mass densities of the spherical object and the fluid, respectively, :math:`g` is the gravitational acceleration, :math:`\mu` is the fluid's dynamic viscosity and :math:`R` is the radius of the object. This equation can be derived by balancing the net force due to gravity

.. math:: F_g=\Delta \rho g V=\left(\rho_\text{o}-\rho_\text{f}\right) g \frac{4}{3}\pi R^3

acting on the object with the drag force :math:`F_\text{d}` into the direction of :math:`\vec{e}_z`, i.e. in direction of the gravity. The drag force can be obtained by the integration over the :math:`z`-projected traction of the object

.. math:: :label: eqspatialstokeslawtraction

   F_\text{d}=\int_\text{obj}  \vec{n}\cdot\left[-p \mathbf{1}+\mu\left(\nabla \vec{u}+(\nabla \vec{u})^\text{t} \right) \right]\cdot\vec{e}_z \:\mathrm{d}S

When solving this, the analytical radial and axial velocity (in frame of the object) reads

.. math:: :label: eqspatialstokeslawfarfield

   \begin{aligned}
   u_r&=\frac{3R^3}{4} \: \frac{rzU}{d^5}-\frac{3R}{4}\:\frac{rzU}{d^3} \nonumber \\
   u_z&=\frac{R^3}{4}\left(\frac{3Uz^2}{d^5}-\frac{U}{d^3}\right)+U-\frac{3R}{4}\left(\frac{U}{d}+\frac{Uz^2}{d^3}\right)\\
   \text{with }d&=\sqrt{r^2+z^2} \nonumber
   \end{aligned}

In our problem, we will not use the analytical velocity :math:numref:`eqspatialstokeslawuterm`, but indeed solve for it. In fact, we use the terminal velocity :math:`U` as global Lagrange multiplier (with test function :math:`V`) to enforce the force balance. Hence, we minimize with respect to the constraint

.. math:: U\cdot\left(F_\text{d}-F_g\right)=0

to determine :math:`U`. This value of :math:`U` is then used as far field condition by virtue of :math:numref:`eqspatialstokeslawfarfield`. :math:`U` is hence determined by the weak form

.. math:: :label: eqspatialstokeslawcontraint

   V\cdot\left(F_\text{d}-F_g\right)=0   

and the feedback of :math:`U` via the traction is given by the far field, which depends on :math:`U` and changes the flow to modify the value of :math:`F_\text{d}` until this constraint is met.

As a first step, we must build a mesh with a spherical object in the center. The far field boundary may be more or less arbitrary, but we chose a larger spherical shell. According to the mesh creation tutorial in :numref:`secspatialmeshgen`, this can be done e.g. by

.. code:: python

   from stokes_dimensional import * # Import dimensional Stokes from before

   # Mesh: Two concentrical hemi-circles => axisymmetric => concentric spheres
   class StokesLawMesh(GmshTemplate):
   	def define_geometry(self):
   		self.default_resolution=0.05 # make it a bit finer
   		self.mesh_mode="tris"
   		p=self.get_problem() # get the problem to obtain parameters
   		Rs=p.sphere_radius # bind sphere radius
   		Ro=p.outer_radius # and outer radius
   		self.far_size = self.default_resolution*float(Ro/Rs) # Make the far field coarser
   		p00=self.point(0,0) # center
   		pSnorth=self.point(0,Rs) # points along the sphere
   		pSeast=self.point(Rs,0)
   		pSsouth=self.point(0,-Rs)
   		pOnorth=self.point(0,Ro,size=self.far_size) # points of the far field
   		pOeast=self.point(Ro,0,size=self.far_size)
   		pOsouth=self.point(0,-Ro,size=self.far_size)
   		self.line(pOsouth,pSsouth,name="axisymm_lower") # axisymmetric lines, we have two since we
   		self.line(pOnorth,pSnorth,name="axisymm_upper")	# want to fix p=0 at a single p-DoF at axisymm_upper
   		self.circle_arc(pOsouth,pOeast,center=p00,name="far_field") # far field hemi-circle
   		self.circle_arc(pOnorth,pOeast,center=p00,name="far_field")
   		self.circle_arc(pSsouth,pSeast,center=p00,name="liquid_sphere") # sphere hemi-circle
   		self.circle_arc(pSnorth,pSeast,center=p00,name="liquid_sphere")						
   		self.plane_surface("axisymm_lower","axisymm_upper","far_field","liquid_sphere",name="liquid") # liquid domain

We split the axis of symmetry into two parts, namely the lower and upper one. Thereby, we can later on pin a single degree of the pressure at e.g. ``"liquid_object/axisymm_lower"`` to remove the pressure nullspace.

Next, we require a possibility to calculate the drag force :math:`F_\text{d}` and add this contribution to the test space of :math:`U`, i.e. add it to the residuals with test function :math:`V`. To that end, we will later pass :math:`V` to the constructor of our new class

.. code:: python

   class DragContribution(InterfaceEquations):
   	required_parent_type = StokesEquations
   	def __init__(self,lagr_mult,direction=vector(0,-1)):
   		super(DragContribution, self).__init__()
   		self.lagr_mult=lagr_mult # Store the destination Lagrange multiplier U
   		self.direction=direction # and the e_z direction

   	def define_residuals(self):
   		u=var("velocity",domain=self.get_parent_domain()) # Important: we want to calculate grad with respect to the bulk
   		strain=2*self.get_parent_equations().mu*sym(grad(u)) # get mu from the parent equations
   		p=var("pressure")
   		stress = -p * identity_matrix() + strain  # T=-p*1 + mu*(grad(u)+grad(u)^t))
   		n = var("normal")  # interface normal pointing outwards
   		traction = matproduct(stress, n)  # traction vector by projection
   		ltest=testfunction(self.lagr_mult) # test function V of the Lagrange multiplier U
   		self.add_residual(weak(dot(traction,self.direction),ltest,dimensional_dx=True)) # Integrate dimensionally over the traction

One important trick is here that we pass ``domain=self.get_parent_domain()`` when we bind the field ``"velocity"`` to ``"u"``. Thereby, we do not get the interfacial velocity, but the full velocity of the bulk. While the values of the bulk and interfacial velocity coincide on the interface, spatial derivatives do not! If we bound ``u=var("velocity")`` without the ``domain`` argument, :math:`\nabla\vec{u}` would take the surface gradient :math:`\nabla_S \vec{u}`, not the bulk gradient :math:`\nabla \vec{u}`. Alternatively, we could have used ``u=var("velocity",domain="..")`` as shortcut to bind the bulk velocity.

Then we add the integral :math:numref:`eqspatialstokeslawtraction` to the test space of :math:`U`, i.e on ``testfunction(U)``, which is :math:`V`. However, since :py:func:`~pyoomph.expressions.generic.weak` by default calculates integrals to the non-dimensional differential, i.e. to :math:`\mathrm{d}\tilde{S}` instead of :math:`\mathrm{d}S`, we would not get the unit of a force. Therefore, we have to tell :py:func:`~pyoomph.expressions.generic.weak` by passing ``dimensional_dx=True`` that we want to integrate dimensionally.

The :py:class:`~pyoomph.generic.problem.Problem` class uses physical dimensions and we set the default values in the constructor. Furthermore, we add a method that allows us to calculate the analytical terminal velocity according to :math:numref:`eqspatialstokeslawuterm`:

.. code:: python

   class StokesLawProblem(Problem):
   	def __init__(self):
   		super(StokesLawProblem, self).__init__()
   		self.sphere_radius=1*milli*meter # radius of the spherical object
   		self.outer_radius=10*milli*meter # radius of the far boundary
   		self.gravity=9.81*meter/second**2 # gravitational acceleration
   		self.sphere_density=1200*kilogram/meter**3 # density of the sphere
   		self.fluid_density=1000*kilogram/meter**3 # density of the liquid
   		self.fluid_viscosity=1*milli*pascal*second # viscosity

   	def get_theoretical_velocity(self): # get the analytical terminal velocity
   		return 2 / 9 * (self.sphere_density - self.fluid_density) / self.fluid_viscosity * self.gravity * self.sphere_radius ** 2

The problem definition will now use our mesh, set an axisymmetric coordinate system and introduce scalings, namely the object radius as spatial scale and the theoretical velocity as velocity scale. The pressure scale is set by the viscous pressure scale and we furthermore introduce a scale for any ``"force"``, which is initialized by the buoyancy force. This one will be used in a minute.

.. code:: python

   	def define_problem(self):
   		self.set_coordinate_system("axisymmetric") # axisymmetric
   		self.set_scaling(spatial=self.sphere_radius) # use the radius as spatial scale 

   		# Use the theoretical value as scaling for the velocity
   		UStokes_ana=self.get_theoretical_velocity()
   		self.set_scaling(velocity=UStokes_ana)		
   		self.set_scaling(pressure=scale_factor("velocity")*self.fluid_viscosity/scale_factor("spatial"))
   		# Buoyancy force
   		F_buo=(self.sphere_density-self.fluid_density)*self.gravity*4/3*pi*self.sphere_radius**3
   		self.set_scaling(force=F_buo) # define the scale "force" by the value of the gravity force

   		self.add_mesh(StokesLawMesh()) 		# Mesh

The first part of the equations is trivial, just ``StokesEquations`` with output and a few boundary conditions:

.. code:: python

   		eqs=StokesEquations(self.fluid_viscosity) # Stokes equation and output
   		eqs+=MeshFileOutput() 
   	
   		eqs+=DirichletBC(velocity_x=0)@"axisymm_lower" # No flow through the axis of symmetry
   		eqs += DirichletBC(velocity_x=0) @ "axisymm_upper"  # No flow through the axis of symmetry
   		eqs+=DirichletBC(velocity_x=0,velocity_y=0)@"liquid_sphere" # and no-slip on the object

Then, the Lagrange multiplier, i.e. the terminal velocity :math:`U`, is introduced. We use :py:class:`~pyoomph.generic.codegen.GlobalLagrangeMultiplier` for that, which will introduce a single global degree of freedom ``UStokes``. Furthermore, the constant offset of :math:`F_g` (``F_buo``) is subtracted, i.e. accounting for this term in :math:numref:`eqspatialstokeslawcontraint`. Both the definition of ``UStokes`` and the offset term are simultaneously done by passing ``UStokes=-F_buo`` to the :py:class:`~pyoomph.generic.codegen.GlobalLagrangeMultiplier`. The Lagrange multiplier equation is then augmented by a :py:class:`~pyoomph.equations.generic.Scaling` and a :py:class:`~pyoomph.equations.generic.TestScaling`, which sets the scale of ``UStokes`` to the ``"velocity"`` scale and the scale of its test function, i.e. :math:`V`, to an inverse of the ``"force"`` scale. With the latter, :math:numref:`eqspatialstokeslawcontraint` will become nondimensional, i.e. the units of force will cancel out upon the internal replacement of the variables and test functions by their non-dimensional counterparts:

.. code:: python

   		# Define the Lagrange multiplier U
   		U_eqs = GlobalLagrangeMultiplier(UStokes=-F_buo) # name if "UStokes" and add an offset of -F_buo to test space of U
   		U_eqs += Scaling(UStokes=scale_factor("velocity")) # "UStokes" scales as a velocity
   		U_eqs += TestScaling(UStokes=1/scale_factor("force")) # and V scales as 1/[F]
   		self.add_equations(U_eqs @ "globals") # add it to an ODE domain named "globals"

.. note::
	The :py:class:`~pyoomph.generic.problem.Problem` class has a method :py:meth:`~pyoomph.generic.problem.Problem.add_global_dof`, which simplifies the addition of a :py:class:`~pyoomph.generic.codegen.GlobalLagrangeMultiplier` with a :py:class:`~pyoomph.equations.generic.Scaling` and a :py:class:`~pyoomph.equations.generic.TestScaling` and a potential global contribution to its residual.

Since the Lagrange multiplier is global, we cannot add it to any mesh. Instead, it has to be added to an own domain, which we call ``"globals"`` here.


..  figure:: stokes_law.*
	:name: figspatialstokeslaw
	:align: center
	:alt: Velocity around objects according to Stokes law
	:class: with-shadow
	:width: 80%

	(left) Velocity around a spherical object according to Stokes law. (right) With adjustments of the mesh, one easily can replace the shape of the object.




We then bind this variable, where again the ``domain`` argument is crucial and pass it to our developed class ``DragContribution``. The ``DragContribution`` has to be attached to the ``"liquid/liquid_sphere"`` interface, since we must integrate over this interface to obtain the drag:

.. code:: python

   		U=var("UStokes",domain="globals") # bind U from the domain "globals"
   		# Add the traction integral, i.e. the drag force to U
   		eqs += DragContribution(U)@"liquid_sphere" # The constraint is now fully assembled

Finally, the value of :math:`U` must be used as far field condition. To that end, we implement the analytical solution :math:numref:`eqspatialstokeslawfarfield` into pyoomph and enforce it at the far field boundary. We cannot use a :py:class:`~pyoomph.meshes.bcs.DirichletBC` here, since the analytical solution depends on :math:`U`, which is part of the unknowns, but :py:class:`~pyoomph.meshes.bcs.DirichletBC` terms should only depend on independent variables as e.g. ``"time"``:

.. code:: python

   		# Far field condition
   		R=self.sphere_radius
   		r,z=var(["coordinate_x","coordinate_y"])
   		d=subexpression(square_root(r**2+z**2)) # precalcuate d in the generated C code for faster computation
   		ur_far=3*R**3/4*r*z*U/d**5-3*R/4*r*z*U/d**3 # u_r as function of U
   		uz_far=R**3/4*(3*U*z**2/d**5-U/d**3)+U-3*R/4*(U/d+U*z**2/d**3) # u_z as function of U

   		# Since U is an unknown, DirichletBC should not be used here. Instead, we enforce the velocity components to the far field by Lagrange multipliers
   		eqs+=EnforcedBC(velocity_x=var("velocity_x")-ur_far,velocity_y=var("velocity_y")-uz_far)@"far_field"
   		eqs += DirichletBC(pressure=0) @ "liquid_sphere/axisymm_upper"  # fix one pressure degree

   		self.add_equations(eqs@"liquid")

The run code is again short, but we compare the analytical and numerical values, leading to an error of :math:`\sim 0.024\:\mathrm{\%}` for this mesh resolution:

.. code:: python

   if __name__ == "__main__":		
   	with StokesLawProblem() as problem: 
   		problem.solve() # solve and output
   		problem.output()
   		# Compare numerical and analytical velocity
   		U_num=problem.get_ode("globals").get_value("UStokes")
   		U_ana=problem.get_theoretical_velocity()
   		print("NUMERICAL: ",U_num,"ANALYTICAL:",U_ana,"ERROR [%]:",abs(float((U_num-U_ana)/U_ana*100)))

The result is plotted in :numref:`figspatialstokeslaw`. We can easily change the mesh to calculate the terminal velocity around differently shaped objects. The far field solution won't be exact, but for a sufficiently large exterior mesh, the resulting error becomes small due to the convergence of :math:numref:`eqspatialstokeslawfarfield` to :math:`(u_r,u_z)=(0,U)`.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <stokes_flow_around_object.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
