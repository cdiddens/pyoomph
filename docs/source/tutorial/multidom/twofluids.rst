Connecting two fluid domains
----------------------------

In :numref:`secALEfreesurfNS`, we have developed the equations of a free liquid surface. However, so far we have not considered the presence of the opposite side of the free surface, which is usually also a fluid. So let's do this now...

Let us consider two phases ``"inside"`` and ``"outside"`` with respect of the free surface, which we will address by ``"interface"`` or ``"inside/interface"``, since we will add all necessary :py:class:`~pyoomph.generic.codegen.InterfaceEquations` on the ``"interface"`` from the domain ``"inside"``. Actually, the kinematic boundary condition :math:numref:`eqalekinbcstrong` has to be fulfilled on both sides. When we use again a phase superscript :math:`\phi`, we can state

.. math:: :label: eqmultidomkinbcstrong

   \begin{aligned}
   \vec{n}^\text{i}\cdot\left(\vec{u}^\phi-\dot{\vec{x}}\right)=0\quad\text{for}\quad\phi=\text{i,o for inside and outside}\,,
   \end{aligned}

where the normal :math:`\vec{n}^\text{i}` is pointing from ``"inside"`` to ``"outside"`` (actually, which direction does not matter here). However, if one just adds one ``KinematicBC`` object per side, i.e. on ``"inside/interface"`` and ``"outside/interface"``, the kinematic boundary condition would be fulfilled on both sides, but once the mesh is connected with a :py:class:`~pyoomph.equations.ALE.ConnectMeshAtInterface` object, the normal mesh motion would be over-constrained.

Both kinematic boundary conditions together can be also formulated by the kinematic boundary condition for the ``"inside"`` domain and the requirement that the normal velocity is continuous across the ``"interface"``:

.. math:: :label: eqmultidomkinbcstrong2

   \begin{aligned}
   \vec{n}^\text{i}\cdot\left(\vec{u}^\text{i}-\dot{\vec{x}}\right)&=0\\
   \vec{n}^\text{i}\cdot\left(\vec{u}^\text{i}-\vec{u}^\text{o}\right)&=0
   \end{aligned}

When we further demand that the tangential velocity should be also continuous (which is a reasonable assumption), :math:numref:`eqmultidomkinbcstrong2` reads

.. math:: :label: eqmultidomveloconti

   \begin{aligned}
   \vec{u}^\text{i}-\vec{u}^\text{o}&=\vec{0}\,
   \end{aligned}

which can be enforced by a vectorial Lagrange multiplier field :math:`\vec\lambda` (with test function :math:`\vec{\eta}`) as usual

.. math:: :label: eqmultidomvelocontiweak

   \begin{aligned}
   \left\langle\vec{u}^\text{i}-\vec{u}^\text{o},\vec\eta\right\rangle+\left\langle \vec\lambda, \vec{v}^\text{i} \right\rangle+\left\langle -\vec\lambda, \vec{v}^\text{o} \right\rangle\,.
   \end{aligned}

The dynamic boundary condition :math:numref:`eqaledynbcstrong` must be generalized to

.. math:: :label: eqmultidomdynbcweak

   \begin{aligned}
   \vec{n}^\text{i}\cdot\left[-p^\text{i}\mathbf{1}+\mu^\text{i}\left(\nabla\vec{u}^\text{i}+(\nabla\vec{u}^\text{i})^\text{t}\right)\right]+\vec{n}^\text{o}\cdot\left[-p^\text{o}\mathbf{1}+\mu^\text{o}\left(\nabla\vec{u}^\text{o}+(\nabla\vec{u}^\text{o})^\text{t}\right)\right]=\sigma\kappa\vec{n}^\text{i}+\nabla_S \sigma\,,
   \end{aligned}

where the lhs can be simplified due to :math:`\vec{n}^\text{i}=-\vec{n}^\text{o}`. Indeed, analogous to the heat flux in :numref:`secmultidomheatcond`, this equation is automatically fulfilled if we add a ``DynamicBC`` to the ``"inside/interface"`` and couple the velocities on both sides via :math:numref:`eqmultidomvelocontiweak`: On the ``"inside/interface"``, we then have the Neumann contribution

.. math::

   \begin{aligned}
   \vec{n}^\text{i}\cdot\left[-p^\text{i}\mathbf{1}+\mu^\text{i}\left(\nabla\vec{u}^\text{i}+(\nabla\vec{u}^\text{i})^\text{t}\right)\right]=\sigma\kappa\vec{n}^\text{i}+\nabla_S \sigma+\vec\lambda
   \end{aligned}

and on the ``"outside/interface"``

.. math::

   \begin{aligned}
   \vec{n}^\text{o}\cdot\left[-p^\text{o}\mathbf{1}+\mu^\text{o}\left(\nabla\vec{u}^\text{o}+(\nabla\vec{u}^\text{o})^\text{t}\right)\right]=-\vec{\lambda}\,.
   \end{aligned}

It is apparent, that the sum of the latter two equations indeed gives the dynamic boundary condition :math:numref:`eqmultidomdynbcweak`.

So the only additional work we have to do is to couple the velocities by a Lagrange multiplier, which can be implemented in pyoomph as

.. code:: python

   from pyoomph import *
   from pyoomph.expressions import *
   from pyoomph.equations.navier_stokes import *
   from pyoomph.equations.ALE import *

   class EnforceSteadyVelocity(InterfaceEquations):
   	def define_fields(self):
   		self.define_vector_field("_couple_velo","C2")
   		
   	def define_residuals(self):
   		l,ltest=var_and_test("_couple_velo")
   		ui,uitest=var_and_test("velocity") # inner velocity at the interface
   		uo,uotest=var_and_test("velocity",domain=self.get_opposite_side_of_interface()) # outer velocity 
   		self.add_residual(weak(ui-uo,ltest)+weak(l,uitest)-weak(l,uotest))

   	def before_assigning_equations_postorder(self, mesh):
   		# pin Lagrange multiplier if both velocities are pinned
   		# we have to iterate over the directions x,y,z (if present)
   		for d in ["x","y","z"][0:self.get_nodal_dimension()]:
   			self.pin_redundant_lagrange_multipliers(mesh,"_couple_velo_"+d,"velocity_"+d,opposite_interface="velocity_"+d)

Again, we have to tell :py:func:`~pyoomph.expressions.generic.var` with ``domain=self.get_opposite_side_of_interface()`` that we want to have the outer velocity field, whereas without this argument, the inner velocity is meant. When both velocities are prescribed with a :py:class:`~pyoomph.meshes.bcs.DirichletBC`, i.e. pinned, the Lagrange multiplier would either lead to a null space (if the strongly imposed velocities matching) or to the absence of any solution (if the strongly imposed velocities are mismatching). We have to do this per component, which is done in the ``for`` loop. Here, only the components are considered, which are actually present in the actual nodal dimension of the mesh via :py:meth:`~pyoomph.generic.codegen.BaseEquations.get_nodal_dimension`. We also use the argument ``opposite_interface=...`` to tell :py:meth:`~pyoomph.generic.codegen.InterfaceEquations.pin_redundant_lagrange_multipliers` that each component of the Lagrange multiplier :math:`\vec\lambda` is only redundant if both the inside and the outside velocity component is pinned. Note that the predefined :py:class:`~pyoomph.generic.codegen.InterfaceEquations` class :py:class:`~pyoomph.equations.ALE.ConnectMeshAtInterface` does exactly the same but on the mesh positions.

The rest of the code is rather straight-forward, however, we use the :py:class:`~pyoomph.meshes.simplemeshes.RectangularQuadMesh` with a ``lambda`` ``callable`` as argument for ``name``:

.. code:: python

   class TwoLayerFlowProblem(Problem):
   	def __init__(self):
   		super(TwoLayerFlowProblem, self).__init__()
   		self.W=1
   		self.H1=0.1
   		self.H2=0.1
   		self.quad_size=0.01

   	def define_problem(self):
   		domain_names=lambda x,y: "lower" if y<self.H1 else "upper" # Name lower half lower, upper half upper
   		self.add_mesh(RectangularQuadMesh(N=[math.ceil(self.W/self.quad_size), math.ceil((self.H1+self.H2)/self.quad_size)], size=[self.W, self.H1+self.H2],name=domain_names,boundary_names={"lower_upper":"interface"}))

With this argument, we can split the :py:class:`~pyoomph.meshes.simplemeshes.RectangularQuadMesh` into multiple domains. The ``callable`` passed to ``name`` receives nondimensional :math:`x,y` coordinates of the element centers and is expected to return the name of the domain. Interfaces between the different domains are automatically marked by ``"domain1_domain2"`` with the adjacent domain names ``"domain1"`` and ``"domain2"`` (in alphabetic order). Here, we rename this interface ``"lower_upper"`` via the ``boundary_names`` ``dict`` to ``"interface"``.

The equations are assembled and added:

.. code:: python

   		# Add the same required equations to both domains
   		for dom in ["lower","upper"]:
   			eqs=LaplaceSmoothedMesh()
   			eqs+=MeshFileOutput()
   			eqs+=DirichletBC(mesh_x=True)
   			eqs += DirichletBC(velocity_x=0) @ "left"  # no in/outflow at the sides
   			eqs += DirichletBC(velocity_x=0) @ "right"
   			self.add_equations(eqs@dom)

   		# Different fluids
   		l_eqs = NavierStokesEquations(mass_density=0.01, dynamic_viscosity=1)  # NS equations
   		u_eqs = NavierStokesEquations(mass_density=0.01, dynamic_viscosity=0.01)  # NS equations

   		# no slip at top and bottom
   		l_eqs += DirichletBC(velocity_x=0, velocity_y=0, mesh_y=0) @ "bottom"  # no slip at bottom and fix the mesh there
   		u_eqs += DirichletBC(velocity_x=0, velocity_y=0, mesh_y=self.H1+self.H2) @ "top"  # no slip at bottom and fix the mesh there
   		l_eqs += DirichletBC(pressure=0) @"bottom/left" # pin one pressure degree

   		# Free surface, mesh connection and velocity connection
   		l_eqs += NavierStokesFreeSurface(surface_tension=1) @ "interface"  # free surface at the top
   		l_eqs += ConnectMeshAtInterface()@"interface"
   		l_eqs += EnforceSteadyVelocity()@"interface"

   		# Deform the initial mesh
   		X, Y = var(["lagrangian_x", "lagrangian_y"])
   		l_eqs += InitialCondition(mesh_y=Y * (1 + 0.25 * cos(2 * pi * X)))  # small height with a modulation
   		u_eqs += InitialCondition(mesh_y=Y+ (self.H1+self.H2-Y)*(0.25 * cos(2 * pi * X)))  # small height with a modulation
   		self.add_equations(l_eqs @ "lower" + u_eqs @ "upper")  # adding it to the system

We use the predefined :py:class:`~pyoomph.equations.navier_stokes.NavierStokesFreeSurface` instead of our free surface consisting of ``KinematicBC`` and ``DynamicBC`` developed in :numref:`secALEfreesurfNS`, but it does essentially the same. With the ``EnforceSteadyVelocity``, the velocities are enforced to be continuous, whereas the Lagrange multiplier :math:`\lambda_x` in :math:`x`-direction will be pinned to :math:`0` automatically on the ``"left"`` and ``"right"``, since both inside and outside velocity are prescribed by a :py:class:`~pyoomph.meshes.bcs.DirichletBC`.

The run code reads

.. code:: python

   if __name__=="__main__":
   	with TwoLayerFlowProblem() as problem:
   		problem.run(50,outstep=True,startstep=0.25)

and the results are depicted in :numref:`figmultidomtwolayer`.

..  figure:: two_layer.*
	:name: figmultidomtwolayer
	:align: center
	:alt: Two-layer flow with connected velocity at the interface.
	:class: with-shadow
	:width: 100%

	Two-layer flow with connected velocity at the interface. By the velocity coupling, the stress is correctly distributed between both domains.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <two_layer_flow.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   	
		    

.. tip::

   There is a similar example case in oomph-lib at https://oomph-lib.github.io/oomph-lib/doc/navier_stokes/two_layer_interface/html/index.html. However, in their case, a single mesh (i.e. domain) is used, but with varying viscosity and mass densities per elements. The free surface is just added at an interior interface. Thereby, the continuity of the velocity field and the mesh position across the interface is automatically fulfilled, i.e. no Lagrange multipliers to connect the velocity and mesh are necessary. However, since the pressure has a jump at the interface due to the Laplace pressure, the pressure space must be discontinuous, i.e. in the oomph-lib example, Crouzeix-Raviart instead of Taylor-Hood elements are used. While it is possible to follow the same approach in pyoomph, it is not discussed here. The moment, mass transfer between both phases is considered, the normal velocity has a jump at the interface as well, provided the mass densities in both phases are different. Then, Lagrange multipliers are definitely required.

