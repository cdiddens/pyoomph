Laplace smoothed mesh
---------------------

Since the Lagrangian coordinates are initialized with the undeformed initial Eulerian coordinates, we can define the displacement from the initial configuration as :math:`\vec{d}=\vec{x}-\vec{\xi}`. One can smooth this displacement by solving a Laplace equation for :math:`\vec{d}`, i.e. :math:`\nabla_\xi^2\vec{d}=0`, where :math:`\nabla_\xi` denotes the derivatives with respect to the Lagrangian coordinates. Thereby, any deformation that is imposed e.g. on the boundaries, will be smoothed out along the mesh.

The weak formulation with test function :math:`\vec{\chi}` reads

.. math:: :label: eqalelaplsmooth

   \begin{aligned}
   \left(\nabla_\xi\left(\vec{x}-\vec{\xi}\right),\nabla_\xi \vec{\chi}\right)_\xi-\left\langle \vec{n}_\xi\cdot\nabla_\xi \left(\vec{x}-\vec{\xi}\right) ,\vec{\chi} \right\rangle_\xi =0 
   \end{aligned}

Let us hence define this equation class:

.. literalinclude:: laplace_smoothed_mesh.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: self.add_residual(weak(grad(d,lagrangian=True), grad(xtest, lagrangian=True),lagrangian=True) )

We do not define fields, but we activate the mesh coordinates as dependent variables with the call :py:meth:`~pyoomph.generic.codegen.Equations.activate_coordinates_as_dofs`. You can pass an argument ``coordinate_space`` to select the space. If further fields are added, the coordinate space must at least comprise the highest space of all defined fields, i.e. we cannot have a ``coordinate_space`` of ``"C1"`` and define other fields on the space ``"C2"``. If the argument is omitted, the coordinate space will be automatically determined by the highest space of all added fields. The rest is trivial, except the usage of the variables ``"mesh"`` and ``"lagrangian"`` and the keyword arguments ``lagrangian=True`` to the :py:func:`~pyoomph.expressions.generic.grad` and :py:func:`~pyoomph.expressions.generic.weak` calls.

As an example problem, let us deform a rectangular mesh by prescribing Dirichlet boundary conditions at the interfaces and let the internal mesh relax based on the Laplace smoothing:

.. literalinclude:: laplace_smoothed_mesh.py
   :language: python
   :start-at: class LaplaceSmoothProblem(Problem):
   :end-at: problem.output_at_increased_time()

A few new things occur here. First, we set the property :py:attr:`~pyoomph.generic.problem.Problem.initial_adaption_steps` of the problem class to ``0``. This controls the initial adaption, i.e. the adaption steps taken after the first solve. We deactivate this to get the middle mesh in :numref:`figalelaplacesmooth`. If this is not set, but a :py:class:`~pyoomph.equations.generic.SpatialErrorEstimator` is present, pyoomph will already adapt with respect to the initial condition. Then, the :py:class:`~pyoomph.meshes.bcs.DirichletBC` terms have values that are set to ``True`` instead to some value. This will fix the value of the variable at the interface, but it will not influence its value. Thereby, we can e.g. fix the :math:`y`-coordinates of the ``"left"`` interface. Finally, note that we use the Lagrangian coordinate to prescribe the deformation in the :py:class:`~pyoomph.meshes.bcs.DirichletBC` term. We cannot use the Eulerian coordinate (i.e. ``var("mesh")`` or ``var("coordinate")``) here, since these are now unknowns. Dirichlet boundary conditions may only depend on independent variables.

Finally, the :py:class:`~pyoomph.equations.generic.SpatialErrorEstimator` will refine the mesh where the deformation is rather discontinuous (cf. right panel in :numref:`figalelaplacesmooth`).


..  figure:: laplacesmooth.*
	:name: figalelaplacesmooth
	:align: center
	:alt: Laplace-smoothed mesh
	:class: with-shadow
	:width: 100%

	Laplace smoothing: (left) undeformed mesh. (center) mesh after applying the Dirichlet boundary conditions that deform the mesh at the interfaces. (right) Relaxed mesh.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <laplace_smoothed_mesh.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   	
		    
