.. _secALEtimediff:

Time derivatives of fields on moving meshes
-------------------------------------------

We will now move a mesh around and solve a diffusion equation on the moving mesh to see what happens. First of all, the mesh shall now move to the left and right by a sine oscillation, which can be realized as follows

.. code:: python

   from pyoomph import *
   from pyoomph.expressions import *

   class MovingMesh(Equations):
   	def define_fields(self):
   		self.activate_coordinates_as_dofs() 
   		
   	def define_residuals(self):
   		x,xtest=var_and_test("mesh_x") 
   		xi,t=var(["lagrangian_x","time"]) 
   		desired_pos=xi+0.125*sin(2*pi*t)
   		self.add_residual(weak(x-desired_pos,xtest,lagrangian=True) )

We have omitted the ``coordinate_space`` argument in :py:meth:`~pyoomph.generic.codegen.Equations.activate_coordinates_as_dofs`, so that the coordinate space is automatically adjusted by the space of the diffusion equation. The diffusion equation with diffusivity :math:`0.01` could be written as follows:

.. code:: python

   class DiffusionEquation(Equations):
   	def define_fields(self):
   		self.define_scalar_field("c","C2")
   		
   	def define_residuals(self):
   		c,ctest=var_and_test("c")
   		# Note the ALE=False statement here. We come to it in the text
   		self.add_residual(weak(partial_t(c,ALE=False),ctest)+weak(0.01*grad(c),grad(ctest)))

And the :py:class:`~pyoomph.generic.problem.Problem` class combining the equations reads

.. code:: python

   class ALEProblem(Problem):
   	def define_problem(self):
   		self.add_mesh(RectangularQuadMesh(N=32,lower_left=[-0.5,-0.5]))
   		eqs=MovingMesh()
   		eqs+=MeshFileOutput()
   		eqs+=DiffusionEquation()
   		eqs+=DirichletBC(mesh_y=True)
   		x=var("coordinate")
   		eqs+=InitialCondition(c=exp(-dot(x,x)*100))
   		self.add_equations(eqs@"domain")
   		
   if __name__=="__main__":		
   	with ALEProblem() as problem:
   		problem.run(1,numouts=20)

Note how we pin the :math:`y`-coordinate by a :py:class:`~pyoomph.meshes.bcs.DirichletBC` applied on the entire ``"domain"``. There is no equation for the :math:`y`-coordinate, so we have to fix it. We furthermore set a Gaussian initial condition for the diffusion field.

..  figure:: ALE1.*
	:name: figaleale1
	:align: center
	:alt: ALE correction of time derivatives
	:class: with-shadow
	:width: 100%

	When the mesh is oscillating to the left and right, all fields move along with it. The center of the Gaussian spot is always in the center of the mesh, not of the Eulerian coordinate system. When the keyword argument ``ALE`` of the function :py:func:`~pyoomph.expressions.generic.partial_t` is set to ``True`` or ``"auto"``, it is compensated for the mesh motion. Thereby, the maximum of the field stays in the center of the coordinates, not of the mesh. Since it gets into contact with the boundaries, the field is slightly deformed.

What happens is naively counter-intuitive, namely the diffusion field :math:`c` is moving along with the oscillating mesh, see :numref:`figaleale1`. One might expect, that the maximum of the field :math:`c` will be always at :math:`\vec{x}=0`, but it is not the case. Instead, it will be always in the center of the mesh, i.e. at :math:`\vec{\xi}=0`. This can be understood, when considering that fields are always approximated as functions of the nodal values :math:`c_l(t)`. Here, we have

.. math:: c(\vec{x},t)=\sum_l c_l(t) \psi(\vec{x},t)

where :math:`\psi(\vec{x},t)` are the shape/basis functions, which are due to the moving mesh now also a function of time :math:`t` and :math:`l` is a summation over all nodes. When adding ``ALE=False`` to ``partial_t(c)``, we will just calculate the temporal derivatives of the coefficients :math:`c_l(t)`, i.e.

.. math:: :label: eqalepartialtnoale

   \text{partial}\_\text{t}^{\text{ALE=False}}\text{(c)}=\sum_l \dot{c}_l(t) \psi(\vec{x},t)

Thereby, when the mesh moves, the entire field (including the time derivative) will co-move with the mesh. If we want to compensate for the mesh motion, we have to compensate for the term originating from the chain rule due to the time-dependence of the mesh coordinates. To that end :py:func:`~pyoomph.expressions.generic.partial_t` has an optional argument ``ALE``, which defaults to ``ALE="auto"``. If ``ALE`` is ``False``, we calculate the time derivative according to :math:numref:`eqalepartialtnoale`. However, if ``ALE=True`` is passed, we get

.. math:: :label: eqalepartialtwithale

   \text{partial}\_\text{t}^{\text{ALE=True}}\text{(c)}=\text{partial}\_\text{t}^{\text{ALE=False}}\text{(c)}-\dot{\vec{x}}\cdot\nabla c

Thereby, the field :math:`c` is moved against the mesh motion and hence stays in place when the mesh moves. Since ``ALE=True`` requires the evaluation of the mesh velocity :math:`\dot{\vec{x}}`, it is not required on static meshes. On a static mesh, :math:`\dot{\vec{x}}=0` holds, and hence the calculation is redundantly time-consuming during the assembly of the system. Pyoomph also allows one to pass ``ALE="auto"`` (the default value) to set it to ``False`` if the mesh is static, i.e. no :py:class:`~pyoomph.generic.codegen.Equations` are added that call :py:meth:`~pyoomph.generic.codegen.Equations.activate_coordinates_as_dofs`. Thereby, the redundant computation of :math:`\dot{\vec{x}}` is not carried out. If the mesh is moving, i.e. an equation for the mesh coordinates is present, ``ALE="auto"`` will become ``ALE=True``, i.e. expanding according to :math:numref:`eqalepartialtwithale`.

.. warning::

   The predefined variables ``var("coordinate")`` and ``var("mesh")`` are in principle the same (and so their components ``var("coordinate_x")``, ``var("mesh_x")``, etc). However, there are two fundamental differences: ``var("mesh")`` can have test functions, when :py:meth:`~pyoomph.generic.codegen.Equations.activate_coordinates_as_dofs` has been called. Furthermore, ``partial_t(var("mesh"))`` may be non-zero on a moving mesh, i.e. yielding the mesh velocity :math:`\dot{\vec{x}}`, whereas ``partial_t(var("coordinate"))`` is always zero.

In conclusion, if one has a moving mesh, but wants to keep spatio-temporal fields independent of the mesh motion, one should augment all :py:func:`~pyoomph.expressions.generic.partial_t` calls with an ``ALE="auto"`` (or leave it out, since it is the default value). This has been done in the bottom row of :numref:`figaleale1`, where the weak formulation of the diffusion equation has been changed to

.. code:: python

   		self.add_residual(weak(partial_t(c,ALE="auto"),ctest)+weak(0.01*grad(c),grad(ctest)))

The field :math:`c` is now not following the oscillatory motion of the mesh, but stays in place.


.. note::
	When demanding conservation of quantities, this approach is often not sufficient. This is discussed in detail later in :numref:`secgcl`.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <ALE_correction.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   	
		    
