Coupled one-dimensional Poisson equations with Dirichlet boundary conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The power of defining the equations in classes easily lets you combine multiple equations. Let us now solve the following system

.. math::

   \begin{aligned}
   -\nabla^2 u&=w \\
   -\nabla^2 w&=-10u \\
   \end{aligned}

subject to :math:`u(-1)=u(1)=0` and :math:`w(-1)=-w(1)=1`. The code just creates two instances of the ``PoissonEquation`` from the previous file :download:`poisson.py` with different names, combines both and couple both equations via the source terms:

.. literalinclude:: poisson_coupled.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: problem.output()  # Write output

Also note how :py:class:`~pyoomph.meshes.bcs.DirichletBC` takes multiple keyword arguments to set multiple boundary values.

..  figure:: coupled_poisson.*
	:name: figspatialcoupledpoisson
	:align: center
	:alt: Coupled Poisson equations.
	:class: with-shadow
	:width: 50%
	
	Coupled Poisson equations with Dirichlet boundaries.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <poisson_coupled.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    