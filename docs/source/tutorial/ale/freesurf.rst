.. _secALEfreesurfNS:

Free surface Navier-Stokes equation
-----------------------------------

We will now combine the Laplace smoothed moving mesh with a Navier-Stokes equation. Having a moving mesh allows us to track a free surface and impose a surface tension on it. We will have two boundary conditions that must be enforced, namely the kinematic and the dynamic boundary condition.

The kinematic boundary condition reads

.. math:: :label: eqalekinbcstrong

   \begin{aligned}
   \vec{n}\cdot\left(\vec{u}-\dot{\vec{x}}\right)=0\,,
   \end{aligned}

i.e. the mesh has to move with the normal fluid velocity :cite:`Cairncross2000`. This is obviously a constraint which narrows the potential solutions of the mesh coordinates. We therefore add a field of Lagrange multipliers :math:`\lambda` with test function :math:`\eta` on each free surface that enforces this constraint. Since we want the normal mesh motion to follow the fluid, and not the velocity following the mesh motion, we add the action of the Lagrange multiplier to the test space of the mesh, not the velocity. If we added it to the test space of the velocity, the particular mesh equations (e.g. the Laplace smoothed mesh) would influence the flow, which is not reasonable. Hence, the weak form at the free surfaces for the kinematic boundary condition reads

.. math:: :label: eqalekinbcweak

   \left\langle \vec{n}\cdot\left(\vec{u}-\dot{\vec{x}}\right), \eta\right\rangle - \left\langle \lambda, \vec{n}\cdot\vec{\chi}\right\rangle\,.

The implementation of the kinematic boundary condition could read as follows

.. literalinclude:: free_surface.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: self.pin_redundant_lagrange_multipliers(mesh, "_kin_bc", "mesh")

Note that we use :py:func:`~pyoomph.expressions.generic.mesh_velocity` instead of :py:func:`~pyoomph.expressions.generic.partial_t` to calculate the mesh velocity. In fact, :py:func:`~pyoomph.expressions.generic.mesh_velocity` is just ``partial_t(var("mesh"),ALE=False)``. This is reasonable, since we want the mesh velocity co-moving with the interface, i.e. directly at the nodes.

Again, we use :py:meth:`~pyoomph.generic.codegen.InterfaceEquations.pin_redundant_lagrange_multipliers` in the :py:meth:`~pyoomph.generic.codegen.BaseEquations.before_assigning_equations_postorder` method to automatically pin the local Lagrange multiplier, if the mesh position is entirely pinned at any node. If the mesh cannot move, the Lagrange multiplier would otherwise over-constrain the problem.

..  figure:: normals.*
	:name: figalenormals
	:align: center
	:alt: Normals at the interface
	:class: with-shadow
	:width: 70%

	Interface normals :math:`\vec{n}` (normal to the interface, pointing outward from the parent domain) and the interface boundary normals :math:`\vec{N}` (tangentially to the interface, pointing outwards) for a 1d interface of a 2d bulk domain and a 2d interface of a 3d bulk domain.

The second condition is the dynamic boundary condition. This one states that the traction is given by a combination of the interface curvature :math:`\kappa`, the surface tension :math:`\sigma` and potential tangential gradients of the latter, i.e.

.. math:: :label: eqaledynbcstrong

   \begin{aligned}
   \vec{n}\cdot\left[-p\mathbf{1}+\mu\left(\nabla\vec{u}+(\nabla\vec{u})^\text{t}\right)\right]=\sigma\kappa\vec{n}+\nabla_S \sigma
   \end{aligned}

:math:`\nabla_S` is the surface gradient operator, sometimes written as :math:`\nabla_S=\left(\mathbf{1}-\mathbf{nn}\right)\nabla`, and will only have tangential contributions. Obviously, the lhs of this equation is the negative Neumann contribution we can add to the (Navier-)Stokes equation, cf. :math:numref:`eqspatialstokesweak`. It could be hence implemented by adding

.. math::

   \begin{aligned}
   -\left\langle \sigma\kappa\vec{n}+\nabla_S \sigma, \vec{v} \right\rangle
   \end{aligned}

as interface contribution to the velocity test function :math:`\vec{v}`. However, it is not trivial to calculate the curvature :math:`\kappa=-\nabla_S\cdot \vec{n}`. In fact, pyoomph does not allow one to calculate the surface divergence of the normal yet. Instead, we make use of the *surface divergence theorem*. For an arbitrary vector field :math:`\vec{w}` defined on the interface :math:`\Gamma`, we have the relation

.. math:: \int_\Gamma \nabla_S\cdot\vec{w}\, \mathrm{d}A = \int_\Gamma \left(\nabla_S\cdot\vec{n}\right) \left(\vec{n}\cdot\vec{w}\right) \mathrm{d}A +\int_{\partial\Gamma} \vec{w}\cdot\vec{N}\, \mathrm{d}l

where the last integral comprises the boundary of the surface with outward normal :math:`\vec{N}` (cf. :numref:`figalenormals` for an illustration of both kinds of normals). When selecting :math:`\vec{w}=\sigma\vec{v}`, this can be arranged to

.. math:: \int_\Gamma \left(-\nabla_S\cdot\vec{n}\right) \left(\sigma\vec{n}\cdot\vec{v}\right) \mathrm{d}A =-\int_\Gamma \nabla_S\cdot\left(\sigma\vec{v}\right)\, \mathrm{d}A   +\int_{\partial\Gamma} \sigma\vec{v}\cdot\vec{N}\, \mathrm{d}l\,,

or, alternatively, using the product rule

.. math:: \int_\Gamma \left[\sigma\left(-\nabla_S\cdot\vec{n}\right)\vec{n}\right]\cdot\vec{v} \,\mathrm{d}A =-\int_\Gamma \left[\left(\nabla_S\sigma\right)\cdot\vec{v}+\sigma\left(\nabla_S\cdot\vec{v}\right)\right]\, \mathrm{d}A   +\int_{\partial\Gamma} \sigma\vec{v}\cdot\vec{N}\, \mathrm{d}l

and by moving the surface tension gradient :math:`\nabla_S \sigma` from the right to the left, we get

.. math:: \int_\Gamma \left[\sigma\left(-\nabla_S\cdot\vec{n}\right)\vec{n}+\nabla_S\sigma)\right]\cdot\vec{v}\, \mathrm{d}A =-\int_\Gamma \sigma\left(\nabla_S\cdot\vec{v}\right)\, \mathrm{d}A   +\int_{\partial\Gamma} \sigma\vec{v}\cdot\vec{N}\, \mathrm{d}l\,.

Upon negation, we can identify the weak forms

.. math:: :label: eqaleweaksigmafs

   -\left\langle \sigma\kappa\vec{n}+\nabla_S\sigma,\vec{v}\right\rangle =\left\langle\sigma,\nabla_S\cdot\vec{v}\right\rangle   -\left[ \sigma\vec{N},\vec{v}\right]\,,


So instead calculating the curvature, it is sufficient to add :math:`\langle \sigma, \nabla_S\cdot\vec{v}\rangle` to get both normal traction due to the Laplace pressure and tangential Marangoni stresses simultaneously. Additionally, there is another term :math:`[\cdot,\cdot]` arising, which allows weak Neumann contributions at the ends of the free surface, which will help us to impose contact angles soon.

.. tip::

   oomph-lib also covers the boundary conditions of a free surface in the tutorial at https://oomph-lib.github.io/oomph-lib/doc/navier_stokes/surface_theory/html/index.html.

   Also, an analogous implementation of the following free surface can be found in oomph-lib at https://oomph-lib.github.io/oomph-lib/doc/navier_stokes/single_layer_free_surface/html/index.html

But first, let us now implement the dynamic boundary condition which can be added to the free surface itself, i.e. the :math:`\langle \cdot , \cdot \rangle` contribution:

.. literalinclude:: free_surface.py
   :language: python
   :start-at: class DynamicBC(InterfaceEquations):
   :end-at: self.add_residual(weak(self.sigma,div(v)))

One might wonder whether :py:func:`~pyoomph.expressions.div` is indeed the surface divergence operator :math:`\nabla_S`. But when this equation is added to an interface, it will indeed expand to this. There is no other reasonable way to calculate the divergence of a field defined on a manifold embedded in a higher order space. The same applies for :py:func:`~pyoomph.expressions.generic.grad`: In the bulk, i.e. on domains with zero *co-dimension*, it is indeed the conventional gradient, but on manifolds (surfaces) with nonzero co-dimension, it will be the corresponding surface gradient.

Before defining the problem, we can combine both boundary conditions in a short-hand notation:

.. literalinclude:: free_surface.py
   :language: python
   :start-at: # Shortcut to create both conditions simultaneously
   :end-at: return KinematicBC()+DynamicBC(sigma)

Now, as example problem, let us do the same as before on the basis of lubrication theory in :numref:`eqpdelubric_relax`, but this time solving the exact flow and the exact free surface dynamics:

..  figure:: free_surface_ns.*
	:name: figalefreesurfacens
	:align: center
	:alt: Free surface combined with Navier-Stokes and a Laplace-smoothed mesh
	:class: with-shadow
	:width: 100%

	Free surface combined with Navier-Stokes and a Laplace-smoothed mesh.


.. literalinclude:: free_surface.py
   :language: python
   :start-at: class SurfaceRelaxationProblem(Problem):
   :end-at: problem.run(50,outstep=True,startstep=0.25)

As opposed to the lubrication example in :numref:`eqpdelubric_relax`, we use a :py:class:`~pyoomph.meshes.simplemeshes.RectangularQuadMesh` to resolve the entire bulk flow. We add the predefined :py:class:`~pyoomph.equations.navier_stokes.NavierStokesEquations`, which also - opposed to lubrication theory - allows for inertia due to the finite mass density. In order to use the free surface equations we have just defined, we must allow the mesh nodes to move, since the ``KinematicBC`` requires adding weak contributions to the test function of the mesh coordinates. We therefore use the predefined :py:class:`~pyoomph.equations.ALE.LaplaceSmoothedMesh`. However, since this particular problem is shallow, it is sufficient to only consider motion in :math:`y`-direction. This can be achieved by just pinning all :math:`x`-positions to their initial values by the ``DirichletBC(mesh_x=True)``. We impose no slip and a zero :math:`y`-coordinate at the ``"bottom"`` interface and prevent any in- or outflow at the ``"left"`` and ``"right"`` interfaces. Finally, we deform the initial mesh by adding an :py:class:`~pyoomph.equations.generic.InitialCondition`, which sets the :math:`y`-position based on the ``"lagrangian"`` coordinate, which corresponds to the undeformed mesh by default.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <free_surface.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   	
		    
