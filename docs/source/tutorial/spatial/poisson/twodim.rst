Two-dimensional Poisson equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All previous discussions so far were exemplified on a 1d domain. Pyoomph makes it very simple to use equations on arbitrary domains. Since we have formulated the ``PoissonEquation`` and the Neumann boundary conditions in :download:`poisson.py` and :download:`poisson_robin_via_neumann.py` with :py:func:`~pyoomph.expressions.generic.grad`, the definition is not restricted to any particular number of dimensions. To solve it on a 2d rectangular domain, we can hence directly reuse the equation classes and the boundary conditions defined above. To solve i.e. the system

.. math::

   \begin{aligned}
   -\nabla^2 u(x,y) &=100\exp\left(-100(x-0.5)^2-100(y-0.5)^2 \right) \\
   u&=0 \quad \text{at} \quad x=0 \\
   u&=0 \quad \text{at} \quad x=1 \\
   \partial_y u&=-2 \quad \text{at} \quad y=0 \\
   u+\partial_y u&=-1 \quad \text{at} \quad y=1
   \end{aligned}

We just have to assemble the system on a 2d geometry, which is predefined in pyoomph in the :py:class:`~pyoomph.meshes.simplemeshes.RectangularQuadMesh`:

.. literalinclude:: poisson_2d.py
   :language: python
   :start-at: # Import all PoissonEquation, Neumann and Robin condition as before
   :end-at: problem.output()

Obviously, the definition of the system in pyoomph is almost identical to the mathematical definition above. One only needs to know the default names of the :py:class:`~pyoomph.meshes.simplemeshes.RectangularQuadMesh` class, which are ``"domain"`` for the inner domain and ``"left"``, ``"right"``, ``"bottom"`` and ``"top"`` for the boundaries. The source function now depends on the coordinate vector :math:`\vec{x}`. This one can be accessed via ``var("coordinate")``. Since it is a vector, one has to use e.g. :py:func:`~pyoomph.expressions.generic.dot` to calculate the square and subtract also the vectorial offset :math:`(0.5,0.5)` via ``vector([0.5,0.5])``. Elements of vectors can be accessed by e.g. ``var("coordinate")[0]`` and ``var("coordinate")[1]``.

If the same condition is to be imposed on several boundaries, they can be listed at once, e.g. ``DirichletBC(u=0)@["left","right","top","bottom"]``. Instead of listing them, the name may also be a wildcard pattern, which is matched against the names the mesh actually provides:

.. code:: python

	eqs+=DirichletBC(u=0)@"*"           # on all boundaries of the domain
	eqs+=DirichletBC(u=0)@"[lr]*"       # only on "left" and "right"
	eqs+=DirichletBC(u=0)@"domain/*"    # the same as the first line, written as a path

The usual wildcards ``*``, ``?``, ``[abc]`` and ``[!abc]`` are supported, but a pattern never spans a ``/``, i.e. it always selects one level of the equation tree. Patterns can be used for the bulk domains themselves (``eqs@"*"`` at the problem level), for the boundaries of a domain, and for the intersections of those boundaries, e.g. ``@"left/*"`` for all points where another boundary meets ``"left"``. Since the names are only known once the meshes have been created, the expansion happens when the problem is initialised; a pattern that does not match any name raises an error listing the available names, so that a typo cannot pass silently.

If a boundary is selected by a pattern and also addressed explicitly, both are applied, with the explicitly named one taking precedence. Thus, ``DirichletBC(u=0)@"*"+DirichletBC(u=1)@"left"`` sets :math:`u=1` on ``"left"`` and :math:`u=0` on all other boundaries.

.. warning::

	On a mesh with multiple bulk domains, the interface between two domains is a boundary of both of them. It is hence also selected by ``@"*"``, which is usually not intended. In that case, the boundaries should be listed explicitly.


..  figure:: poisson2d.*
	:name: figspatialpoisson2d
	:align: center
	:alt: Two-dimensional Poisson equation
	:class: with-shadow
	:width: 50%
	
	Two-dimensional Poisson equation


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <poisson_2d.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    