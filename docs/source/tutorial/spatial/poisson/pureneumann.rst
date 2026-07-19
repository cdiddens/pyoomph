.. _secspatialpoissonpureneumann:

Pure Neumann boundary conditions for the Poisson equation - Using a Lagrange multiplier to remove the nullspace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First of all, in order to have only Neumann conditions, the source term :math:`j` of the Poisson equation and the imposed Neumann fluxes have to fulfill a relation, which can be seen by setting the test function :math:`v=1` (possible, since test functions are allowed to be arbitrary) in :math:numref:`eqspatialpoissonweak`, which gives

.. math:: :label: eqspatialpoissonneumconstr
    
    \int_\Omega g \:\mathrm{d}^n x=-\int_{\Gamma}  j_\text{N} \:\mathrm{d}S \,.

If this relation is not fulfilled, the equation is not well posed. If we had a single Dirichlet condition, setting :math:`v=1` is not allowed, since test functions have to vanish on the Dirichlet boundaries and this requirement is not necessary any more.

There is another complication to obtain the solution, namely the fact that the solution :math:`u` is not unique: If :math:`u` is a solution, :math:`u+c` for any constant :math:`c` is a solution as well. Any Dirichlet condition would again specify a unique constant :math:`c` for which this Dirichlet condition is fulfilled, which renders the solution unique again, but in absence of any Dirichlet condition, there is an infinite number of solutions.

Both problems can be tackled simultaneously by introducing a Lagrange multiplier :math:`\lambda`, which fixes the spatial average of :math:`u`, i.e. for some prescribed :math:`u_\text{avg}`, we demand that the spatial average of :math:`u` is :math:`u_\text{avg}`, i.e.

.. math:: \frac{\int_\Omega u \: \mathrm{d}^{n}x}{\int_\Omega 1 \: \mathrm{d}^{n}x}= u_\text{avg}

Given the fact that :math:`u_\text{avg}` is a constant, we can write this as an implicit constraint

.. math:: \int_\Omega \left(u-u_\text{avg}\right) \: \mathrm{d}^{n}x=0

As before in :numref:`secODEpendulum`, this constraint can be enforced by the Lagrange multiplier :math:`\lambda` by minimizing

.. math:: \lambda \int_\Omega \left(u-u_\text{avg}\right) \: \mathrm{d}^{n}x= \int_\Omega \lambda\left(u-u_\text{avg}\right) \: \mathrm{d}^{n}x

with respect to :math:`u` and :math:`\lambda`, which gives by variation the additional weak contributions

.. math:: \left(\lambda, v\right)+\left(u-u_\text{avg},\mu\right)

where :math:`\mu` is the test function corresponding to :math:`\lambda`. Therefore, we need to augment our Poisson equation from :download:`poisson_neumann.py` by adding these terms to the residual as well:

.. literalinclude:: poisson_pure_neumann_nullspace.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: self.add_residual(weak(u-self.average_value,ltest)+weak(l,utest))

Obviously, we load the old class ``PoissonEquation`` and inherit from it. In the :py:meth:`~pyoomph.generic.codegen.BaseEquations.define_residuals`, we first use a ``super`` call to call :py:meth:`~pyoomph.generic.codegen.BaseEquations.define_residuals` of the non-augmented ``PoissonEquation`` to add the normal residuals to the weak form. Then, the missing terms are added. However, the Lagrange multiplier :math:`\lambda` is not defined yet and it cannot be defined in the ``PoissonEquationWithNullspaceRemoval`` class, since the latter will be restricted to the domain :math:`\Omega`. Any field definitions in the :py:meth:`~pyoomph.generic.codegen.BaseEquations.define_fields` method would hence add fields on :math:`\Omega`, but :math:`\lambda` is just a simple real number, not a field.

Therefore, :math:`\lambda` will be introduced by an own class. Since :math:`\lambda` has no spatial dependence, it can be best done by using the :py:class:`~pyoomph.generic.codegen.ODEEquations` class (see chapter :numref:`secode`), which allows the definition of real valued variables:

.. literalinclude:: poisson_pure_neumann_nullspace.py
   :language: python
   :start-at: # A single "ODE", which is used as storage for the Lagrange multiplier value
   :end-at: self.define_ode_variable(self.name)

Note that we do not add any residuals for :math:`\lambda` inside this class. The single purpose of this class is to provide the storage of a real-valued variable :math:`\lambda`, whereas the contributions to the residuals are entirely done in the weak form of the augmented Poisson equation.

Finally, we need to couple both parts in the definition of the problem:

.. literalinclude:: poisson_pure_neumann_nullspace.py
   :language: python
   :start-at: class PureNeumannPoissonProblem(Problem):
   :end-at: problem.output()  # Write output

In the :py:meth:`~pyoomph.generic.problem.Problem.define_problem` method, it is noteworthy that the ``LagrangeMultiplierForPoisson`` object is stored in another domain (named ``"lambda_space"``) than the Poisson equation. This is necessary, since the domain ``"domain"`` is already bound to the interval :math:`\Omega=[-1,1]` of the line mesh. By the :py:func:`~pyoomph.expressions.generic.var` statement, we bind the just allocated variable :math:`\lambda` to the local variable ``l`` and pass it together with the desired average value :math:`u_\text{avg}=10` to the ``PoissonEquationWithNullspaceRemoval`` object. Thereby, both parts, the augmented Poisson equation on :math:`\Omega` and the separately defined Lagrange multiplier :math:`\lambda` are coupled. The remainder is as before, but now two ``PoissonNeumannCondition`` objects are created.

..  figure:: pure_neumann_poisson.*
	:name: figspatialpureneumannpoisson
	:align: center
	:alt: Poisson equation with pure Neumann conditions.
	:class: with-shadow
	:width: 50%
	
	Poisson equation with pure Neumann conditions, i.e. Neumann boundary conditions at the left and right. We have to enforce an average value (here :math:`10`) to make the solution unique.


The output (plotted in :numref:`figspatialpureneumannpoisson`) indeed shows that the average of :math:`u` is :math:`10` and that the value of the Lagrange multiplier :math:`\lambda=0`. If one instead violates the condition :math:numref:`eqspatialpoissonneumconstr` by imposing an ill-posed combination of the Neumann fluxes and the source :math:`g`, the Lagrange multiplier will attain a non-zero value, which can be seen as follows: Let us first write down the full augmented residual form:

.. math:: \left(\nabla u,\nabla v\right)-\left(g, v\right)+\left(\lambda, v\right)+\left(u-u_\text{avg},\mu\right)-\left\langle j_\text{N}, v\right\rangle=0\,.

Upon selection :math:`v=1` and :math:`\mu=0` as well as :math:`v=0` and :math:`\mu=1`, we arrive at

.. math::

   \begin{aligned}
   \int_\Omega \left(g-\lambda\right) \:\mathrm{d}^n x&=-\int_{\Gamma}  j_\text{N} \:\mathrm{d}S \\
   \int_\Omega \left(u-u_\text{avg}\right) \: \mathrm{d}^{n}x&=0
   \end{aligned}

Obviously, the previous constraint :math:numref:`eqspatialpoissonneumconstr` can now be fulfilled by solving for the correct value :math:`\lambda`, which introduces a new source function :math:`g^*=g-\lambda`, so that this constraint is fulfilled. The test space of :math:`\lambda` on the other hand is used to set the average of :math:`u` to :math:`u_\text{avg}`.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <poisson_pure_neumann_nullspace.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
