Laplace smoothed mesh
---------------------

Since the Lagrangian coordinates are initialized with the undeformed initial Eulerian coordinates, we can define the displacement from the initial configuration as :math:`\vec{d}=\vec{x}-\vec{\xi}`. One can smooth this displacement by solving a Laplace equation for :math:`\vec{d}`, i.e. :math:`\nabla_\xi^2\vec{d}=0`, where :math:`\nabla_\xi` denotes the derivatives with respect to the Lagrangian coordinates. Thereby, any deformation that is imposed e.g. on the boundaries, will be smoothed out along the mesh.

The weak formulation with test function :math:`\vec{\chi}` reads

.. math:: :label: eqalelaplsmooth

   \begin{aligned}
   \left(\nabla_\xi\left(\vec{x}-\vec{\xi}\right),\nabla_\xi \vec{\chi}\right)_\xi-\left\langle \vec{n}_\xi\cdot\nabla_\xi \left(\vec{x}-\vec{\xi}\right) ,\vec{\chi} \right\rangle_\xi =0 
   \end{aligned}

Let us hence define this equation class:

.. code:: python

   from pyoomph import *
   from pyoomph.expressions import *

   class LaplaceSmoothedMesh(Equations):

   	def define_fields(self):
   		# let the mesh coordinates become a variables, approximated with second order Lagrange basis functions
   		self.activate_coordinates_as_dofs(coordinate_space="C2") 
   		
   	def define_residuals(self):
   		x,xtest=var_and_test("mesh") # Eulerian mesh coordinates
   		xi=var("lagrangian") # fixed Lagrangian coordinates
   		d=x-xi # displacement
   		# Weak formulation: gradients and integrals are carried out with respect to the Lagrangian coordinates
   		self.add_residual(weak(grad(d,lagrangian=True), grad(xtest, lagrangian=True),lagrangian=True) )

We do not define fields, but we activate the mesh coordinates as dependent variables with the call :py:meth:`~pyoomph.generic.codegen.Equations.activate_coordinates_as_dofs`. You can pass an argument ``coordinate_space`` to select the space. If further fields are added, the coordinate space must at least comprise the highest space of all defined fields, i.e. we cannot have a ``coordinate_space`` of ``"C1"`` and define other fields on the space ``"C2"``. If the argument is omitted, the coordinate space will be automatically determined by the highest space of all added fields. The rest is trivial, except the usage of the variables ``"mesh"`` and ``"lagrangian"`` and the keyword arguments ``lagrangian=True`` to the :py:func:`~pyoomph.expressions.generic.grad` and :py:func:`~pyoomph.expressions.generic.weak` calls.

As an example problem, let us deform a rectangular mesh by prescribing Dirichlet boundary conditions at the interfaces and let the internal mesh relax based on the Laplace smoothing:

.. code:: python

   class LaplaceSmoothProblem(Problem):
   	def define_problem(self):
   		self.initial_adaption_steps=0
   		self.add_mesh(RectangularQuadMesh(N=6))
   		eqs=LaplaceSmoothedMesh()
   		eqs+=MeshFileOutput()
   		eqs+=DirichletBC(mesh_x=0,mesh_y=True)@"left" # fix the mesh at x=0 and keep y in place
   		eqs+=DirichletBC(mesh_x=True,mesh_y=0)@"bottom" # fix the mesh at y=0 and keep x in place		
   		xi=var("lagrangian") # Lagrangian coordinate
   		eqs+=DirichletBC(mesh_x=1+0.5*xi[1])@"right" # linear slope at the left
   		eqs+=DirichletBC(mesh_y=1+0.25*xi[0]*(1-xi[0]))@"top" # quadratic deformation at the top
   		eqs+=SpatialErrorEstimator(mesh=1) # Adapt where large deformations are present
   		
   		self.add_equations(eqs@"domain")
   		
   if __name__=="__main__":		
   	with LaplaceSmoothProblem() as problem:
   		problem.output()
   		problem.solve(spatial_adapt=4)
   		problem.output_at_increased_time()

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
		    
