Non-Newtonian fluids
~~~~~~~~~~~~~~~~~~~~~

We now want to consider a nonlinear Stokes equation by letting the viscosity depend on the shear rate. Thereby, the normally linear Stokes equation becomes nonlinear and highly dependent on the initial guess. However, shear-thickening or -thinning fluids can be found in reality. We can easily reuse our previous implementation of the Stokes problem by passing a shear dependent viscosity to the :py:class:`~pyoomph.generic.problem.Problem` class which forwards it to the :py:class:`~pyoomph.generic.codegen.Equations` class, where it is ultimately evaluated in the weak form implementation. For simplicity, we set the viscosity to :math:`\mu=1+\dot{\gamma}`, where :math:`\dot{\gamma}=\sqrt{2\mathbf{S}:\mathbf{S}}` with the shear tensor :math:`\mathbf{S}=\frac{1}{2}\left(\nabla\vec{u}+\nabla\vec{u}^\text{t}\right)`:

.. literalinclude:: stokes_nonnewtonian.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: problem.output()

This example again shows the power of pyoomph: At any stage, we can define quite arbitrary expressions and pass them via the :py:class:`~pyoomph.generic.problem.Problem` class to the :py:class:`~pyoomph.generic.codegen.Equations` class, where they are evaluated in the weak form. To get the strain rate, we first bind the variable field by ``u=var("velocity")``, although it is not known here at all. The symmetrized strain tensor is again calculated via :py:func:`~pyoomph.expressions.sym` and :py:func:`~pyoomph.expressions.generic.grad`, ``S=sym(grad(u))``, and we get :math:`\dot \gamma` via applying :py:func:`~pyoomph.expressions.square_root` on the contraction :math:`\mathbf{S}:\mathbf{S}` (using :py:func:`~pyoomph.expressions.generic.contract`). To speed up the code, the dynamic viscosity is again wrapped in a :py:func:`~pyoomph.expressions.generic.subexpression`.

Since the problem is now strongly nonlinear, it is essential to provide a reasonable initial guess. We could write our own :py:class:`~pyoomph.generic.problem.Problem` class, but we also can add further equations to a given :py:class:`~pyoomph.generic.problem.Problem` class by adding those to the property :py:attr:`~pyoomph.generic.problem.Problem.additional_equations`. Of course, we have to restrict them to the ``"domain"``, i.e. on the domain where the equation should be solved.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <stokes_nonnewtonian.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
