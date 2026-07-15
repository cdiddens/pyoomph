.. _secmultidomicefront:

Propagation of an ice front
---------------------------

The next problem is very related to the previous one. We will again solve two temperature conduction equations, but this time, condition :math:numref:`eqmultidomcontitqflux` will be slightly different. Furthermore, we will consider also the temporal behavior and use physical dimensions.

In fact, we want to solve the propagation of an ice front, i.e. how ice is solidifying or melting in the presence of a temperature gradient. Since this phase transition obviously leads to a growth of the ice domain and a corresponding shrinkage of the liquid domain or vice versa, a moving mesh/ALE method will be used.

Mathematically, we have the transient heat conduction equations

.. math:: :label: eqmultidomtempconduct

   \begin{aligned}
   \rho^\phi c_p^\phi \partial_t T^\phi =\nabla\cdot\left(k^\phi \nabla T\right)
   \end{aligned}

where the phase superscript :math:`\phi` can be either ice or liquid, depending on whether we apply this equation on either the ``"ice"`` or the ``"liquid"`` domain. The equation can be easily cast into its weak form and implemented:

.. literalinclude:: temperature_conduction_propagation.py
   :language: python
   :start-at: from temperature_conduction import *
   :end-at: self.add_residual(weak(self.rho*self.c_p*partial_t(T),T_test)+weak(self.k*grad(T),grad(T_test)))

Note that we bind the test scale of the temperature field ``"T"`` to ``1/scale_factor("thermal_equation")``. This means essentially, that :math:numref:`eqmultidomtempconduct` is multiplied by a factor :math:`1/S` during non-dimensionalization, i.e. that we actually solve

.. math:: :label: eqmultidomtempconductnd

   \begin{aligned}
   S^{-1}\rho^\phi c_p^\phi \partial_t T^\phi =S^{-1}\nabla\cdot\left(k^\phi \nabla T\right)
   \end{aligned}

One could now choose e.g. :math:`S=\rho^\text{ice} c_p^\text{ice} [T]/[t]`, where :math:`[T]` is the scaling of the temperature field, e.g. :math:`1\:\mathrm{K}` and :math:`[t]` is the characteristic time scale, e.g. :math:`1\:\mathrm{s}`. Thereby, the nondimensional lhs would have a unity factor. Alternatively, we can also set the factor of the nondimensional conduction term on the rhs to unity by selecting :math:`S=k^\text{ice}[T]/[X]^2` with the spatial scale :math:`X`. Of course, one can also use the properties of the ``"liquid"`` domain instead of the ``"ice"`` domain. Eventually, :math:`S` will be set at :py:class:`~pyoomph.generic.problem.Problem` level with the ``set_scaling(thermal_equation=...)`` method. Thereby, on both domains, the equations will have the same test scale, i.e. are nondimensionalized with respect to the same scale. That way, the problem regarding the consistency of the heat flux at the interface, as discussed in the previous example, will be circumvented. Therefore, this approach is a good practice.

Next, we must couple the interface motion, i.e. the propagation of the ice front, with the heat fluxes. The interface :math:`x_\text{I}` will move, according to

.. math::

   \begin{aligned}
   \partial_t x_\text{I}=\frac{k^\text{ice}\partial_x T^\text{ice}-k^\text{liq}\partial_x T^\text{liq}}{\rho^\text{ice}\Lambda}\,,
   \end{aligned}

where :math:`\Lambda` is the latent heat of solidification. We have used :math:`\rho^\text{ice}` in the denominator, since the liquid will actually be subject to a tiny normal velocity at the interface due to the density difference. But this small contribution is disregarded here, since only conduction equations are solved.

As usual in pyoomph, we should write this equation independent of the chosen coordinate system to make this equation applicable to any problem. This is obviously given by

.. math::

   \begin{aligned}
   \vec{n}\cdot\partial_t \vec{x}_\text{I}=\frac{k^\text{ice}\nabla T^\text{ice}-k^\text{liq}\nabla T^\text{liq}}{\rho^\text{ice}\Lambda}\cdot\vec{n}\,,
   \end{aligned}

In this formulation with interface normal :math:`\vec{n}`, we also notice that it is a constraint for the normal motion of the mesh, whereas the tangential motion is not affected. Since it is a constraint, the typical Lagrange multiplier approach is again the way to take. As usual, with :math:`\vec{\chi}` and :math:`\eta` being the test functions of the mesh position and the Lagrange multiplier :math:`\lambda`, respectively, we get the weak formulation for the constraint:

.. math:: :label: eqmultidomtempconductispeed

   \begin{aligned}
   \left\langle \vec{n}^\text{ice}\cdot\partial_t \vec{x}-\frac{k^\text{ice}\nabla T^\text{ice}\cdot\vec{n}-k^\text{liq}\nabla T^\text{liq}\cdot\vec{n}^\text{ice}}{\rho^\text{ice}\Lambda},\eta\right\rangle+\left\langle \lambda,\vec{n}^\text{ice}\cdot\vec{\chi}\right\rangle
   \end{aligned}

The implementation is rather straight-forward:

.. literalinclude:: temperature_conduction_propagation.py
   :language: python
   :start-at: class IceFrontSpeed(InterfaceEquations):
   :end-at: self.add_residual(weak(l,dot(xtest,n)))

With the :py:attr:`~pyoomph.generic.codegen.InterfaceEquations.required_parent_type` and :py:attr:`~pyoomph.generic.codegen.InterfaceEquations.required_opposite_parent_type`, we inform pyoomph that it is only allowed to attach this constraint to an interface that has a ``TemperatureConductionEquation`` on both the inside bulk and the outside bulk of this interface. Otherwise, an error will be thrown. Due to these statements, we also get automatically the inside and outside ``TemperatureConductionEquation`` of the bulk phases when calling :py:meth:`~pyoomph.generic.codegen.InterfaceEquations.get_parent_equations` and :py:meth:`~pyoomph.generic.codegen.InterfaceEquations.get_opposite_parent_equations`. This is used to obtain the required properties :math:`k^\phi` and :math:`\rho` in the :py:meth:`~pyoomph.generic.codegen.BaseEquations.define_residuals` method here. The interface property ``latent_heat``, however, has to be passed to the constructor and is stored internally.

The scaling has to fit, i.e. upon non-dimensionalization of :math:numref:`eqmultidomtempconductispeed`, all weak forms must yield non-dimensional results. Indeed, if we scale :math:`\lambda` with the inverse of the scaling of :math:`\chi` and nondimensionalize the test function :math:`\eta` as :math:`\eta=[T]/[X]\tilde\eta`, all units will cancel out in :math:numref:`eqmultidomtempconductispeed`.

There is another very relevant aspect to consider, namely:

.. warning::

   One fundamental aspect is that we want to take the bulk gradient for the :math:`\nabla T` terms in :math:numref:`eqmultidomtempconductispeed`. Since we are on an interface, i.e. on a manifold with co-dimension 1, the simple statement ``grad(var("T"))`` would expand to the surface gradient :math:`\nabla_S T` of the temperature field of the inside domain (cf. :numref:`secspatialhelicalmesh`), which will be always tangential to :math:`\vec{n}`. The bulk gradients are only obtained if the temperature fields of the bulk phases are passed to :py:func:`~pyoomph.expressions.generic.grad`. These can be obtained by adding :py:meth:`~pyoomph.generic.codegen.BaseEquations.get_parent_domain` and :py:meth:`~pyoomph.generic.codegen.Equations.get_opposite_parent_domain` (for the inside and outside bulk, respectively) as ``domain=`` keyword argument in the bindings via :py:func:`~pyoomph.expressions.generic.var`.
   Alternatively, you can also use ``domain=".."`` instead of ``domain=self.get_parent_domain()`` and ``domain="|.."`` instead of ``domain=self.get_opposite_parent_domain()``.

In the constructor of the :py:class:`~pyoomph.generic.problem.Problem` class, nothing spectacular happens. We just initialize a few default parameters:

.. literalinclude:: temperature_conduction_propagation.py
   :language: python
   :start-at: class IceFrontProblem(Problem):
   :end-at: self.T_right=1*celsius

In the :py:meth:`~pyoomph.generic.problem.Problem.define_problem` method, we have to set the scales for nondimensionalization and we make use of a ``for`` loop to construct similar equations on both domains:

.. literalinclude:: temperature_conduction_propagation.py
   :language: python
   :start-at: def define_problem(self):
   :end-at: self.add_equations(interf_eqs@"ice/interface")

The interface equations consist of an instance of our just developed class ``IceFrontSpeed`` and the predefined class :py:class:`~pyoomph.equations.ALE.ConnectMeshAtInterface`. The latter will introduce Lagrange multipliers so that the nodes of the ``"liquid"`` and ``"ice"`` domain at the mutual ``"interface"`` will be enforced to coincide. Without this, only the ``"ice"`` mesh would move, whereas the ``"liquid"`` mesh would remain static. As an alternative to adding ``interf_eqs@"ice/interface"`` to the problem, we could also add ``interf_eqs`` to the ``"liquid"`` side of the ``"interface"``. In that case, however, we would have to negate the ``latent_heat``.

The code to run this problem is simple, but we use temporal and spatial adaptivity to properly resolve the initial temperature discontinuity at the ``"interface"``:

.. literalinclude:: temperature_conduction_propagation.py
   :language: python
   :start-at: if __name__=="__main__":
   :end-at: problem.run(1000*second,startstep=0.00001*second,outstep=True,temporal_error=1,spatial_adapt=1,maxstep=2*second)

The results are shown in :numref:`figmultidomiceprop1d`.


..  figure:: iceprop1d.*
	:name: figmultidomiceprop1d
	:align: center
	:alt: Propagation of an ice front
	:class: with-shadow
	:width: 80%

	Propagation of the front between solid ice (left) and liquid water (right) due to a temperature gradient at different times.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <temperature_conduction_propagation.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   	
		    
