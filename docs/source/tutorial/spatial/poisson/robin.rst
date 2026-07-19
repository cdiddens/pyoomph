Robin boundary conditions
~~~~~~~~~~~~~~~~~~~~~~~~~

Robin boundary conditions are sort of a weighted average of a Dirichlet and Neumann condition, i.e. one imposes

.. math:: \alpha u + \beta \nabla u\cdot \vec{n} = g

for some nonzero coefficients :math:`\alpha` and :math:`\beta`. The easiest way to implement these conditions is to rewrite this definition to

.. math:: j=\nabla u\cdot \vec{n} = \frac{1}{\beta}\left( g-\alpha u\right)

and just add this term in the same manner as the Neumann condition, i.e. via :math:`\langle j,v\rangle`:

.. literalinclude:: poisson_robin_via_neumann.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: problem.output()  # Write output

..  figure:: robin_poisson1.*
	:name: figspatialrobinpoisson1
	:align: center
	:alt: Poisson equation with Robin boundary conditions.
	:class: with-shadow
	:width: 50%
	
	Poisson equation with Robin boundary conditions.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <poisson_robin_via_neumann.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    

Of course, you can recover the Neumann condition as a special case by setting :math:`\alpha=0`, but you cannot recover the Dirichlet condition, since :math:`\beta=0` will induce a division by zero.




To overcome this, one can enforce this particular boundary condition with arbitrary values of :math:`\alpha` and :math:`\beta`, by introducing a Lagrange multiplier at the boundary to enforce the condition

.. math:: \alpha u + \beta \nabla u\cdot \vec{n} - g=0\,.

The same idea also works for other kinds of generalized boundary conditions, also non-linear ones. One just has to exchange the definition of the ``PoissonRobinCondition`` as follows:

.. literalinclude:: poisson_robin_via_lagrange.py
   :language: python
   :start-at: # Inherit from the normal InterfaceEquations
   :end-at: self.add_residual(weak(condition,ltest)+weak(l,utest)) # Lagrange multiplier pair to enforce it

The main idea is to create a Lagrange multiplier :math:`\lambda` with test function :math:`\mu` on the interface and add the weak contributions

.. math:: \left\langle\alpha u + \beta \nabla u\cdot \vec{n} - g,\mu \right\rangle+\left\langle \lambda,v \right\rangle \,.

Thereby, the value of :math:`u` on the interface is adjusted until this condition holds. 

.. important::
    It is important to note that the term :math:`\nabla u\cdot \vec{n}` requires some extra caution in pyoomph. To calculate the bulk gradient, it is required to evaluate :math:`u` also in the bulk. Therefore, it is required to obtain the bulk field :math:`u` by using ``var(...,domain=self.get_parent_domain())``. Without the specification of the ``domain`` via :py:meth:`~pyoomph.generic.codegen.BaseEquations.get_parent_domain`, one would obtain the value :math:`u` on the interface and the calculation of the gradient will hence give the *surface gradient*, i.e. it would lead to wrong results here. Alternatively, one can use ``domain=".."`` instead of ``domain=self.get_parent_domain()``.

The outward unit normal is obtained by :py:meth:`~pyoomph.generic.codegen.BaseEquations.get_normal` (or by ``var("normal")``) and :py:func:`~pyoomph.expressions.generic.dot` represents the dot product of two vectors.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <poisson_robin_via_lagrange.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    

For the latter approach, there is also a generic class :py:class:`~pyoomph.meshes.bcs.EnforcedBC`, which allows us to enforce arbitrary boundary conditions. To get the same result as with the custom implemented class ``PoissonRobinCondition("u",alpha,beta,g)``, the generic class requires to cast it into residual form, i.e. ``EnforcedBC(u=alpha*var("u")+beta*dot(grad(var("u",domain="..")),var("normal"))-g)``.
