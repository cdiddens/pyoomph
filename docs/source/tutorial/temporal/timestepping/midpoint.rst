Trapezoidal rule and the implicit midpoint rule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So far, we have used different time stepping methods, but there are obviously more possibilities. Let us focus on systems of first order ODEs only, which can be written in general as

.. math:: \partial_t \vec{U}=\vec{F}(\vec{U},t)

with the vector of unknowns :math:`\vec{U}` and the rhs :math:`\vec{F}(\vec{U},t)`. Of course, as usual in numerical simulations, :math:`\vec{U}` is discretized in time, i.e. we have the time steps :math:`\vec{U}^{(0)}`, :math:`\vec{U}^{(1)}`, :math:`\ldots`, :math:`\vec{U}^{(n-1)}` already computed and we want to compute the step :math:`\vec{U}^{(n)}`. With this notation, an approximation of the time derivative can be written as

.. math:: 
	:label: eqodetsteppweight
	
	\partial_t \vec{U}\approx\sum_{i=0}^{k} w_i \vec{U}^{(n-k)}\,. 

Here, :math:`k` is the order of the time stepper, e.g. :math:`k=1` for ``"BDF1"`` and :math:`k=2` for ``"BDF2"``, and :math:`w_i` are the time stepper weights. For example, in ``"BDF1"``, the weights are :math:`w_0=-w_1=1/\Delta t^{(n)}`, which gives

.. math:: \partial_t \vec{U}\approx\sum_{i=0}^{1} w_i \vec{U}^{(n-k)}=\frac{1}{\Delta t^{(n)}}\left(\vec{U}^{(n)}-\vec{U}^{(n-1)}\right)\,,

where :math:`\Delta t^{(n)}` is the time step we want to take to obtain :math:`\vec{U}^{(n)}`.

Until now, we have not addressed the right hand side, i.e. :math:`\vec{F}(\vec{U})`. In fact, if we use any variable bound by :py:func:`~pyoomph.expressions.generic.var` in the code, it always means the value corresponding to the time step :math:`n`, i.e. the right hand side up to now was always fully implicit. This means e.g. for ``"BDF1"`` time stepping that the discretized equations read

.. math:: \frac{1}{\Delta t^{(n)}}\left(\vec{U}^{(n)}-\vec{U}^{(n-1)}\right)=\vec{F}\left(\vec{U}^{(n)},t^{(n)}\right)\,,

where :math:`t^{(n)}=t^{(n-1)}+\Delta t^{(n)}` is the time we are currently solving for. Indeed, this is the implicit Euler integration.

However, there are time integration methods, where also previous values of :math:`\vec{U}`, i.e. :math:`\vec{U}^{(n-1)}` etc., are used in the rhs. One particular example is the *trapezoidal rule* (sometimes also called *Crank-Nicolson method*). This one reads

.. math:: 
	:label: eqodetrapzrule
	
	\frac{1}{\Delta t^{(n)}}\left(\vec{U}^{(n)}-\vec{U}^{(n-1)}\right)=\frac{1}{2}\left[\vec{F}\left(\vec{U}^{(n)},t^{(n)}\right)+\vec{F}\left(\vec{U}^{(n-1)},t^{(n-1)}\right)\right]\,,

In pyoomph, the lhs can easily be obtained by writing ``partial_t(...,scheme="BDF1")``, but how to evaluate the right hand side appropriately? As mentioned before, :py:func:`~pyoomph.expressions.generic.var` statements are always evaluated at the current time step :math:`n` we are currently solving for, so how to access the values of :math:`\vec{U}` at previous time steps?

The answer is the function :py:func:`~pyoomph.expressions.generic.evaluate_in_past`, which takes an expression as argument and evaluates it at previous time steps. Without an argument, ``evaluate_in_past(expr)`` evaluates ``expr`` at the time step :math:`n-1`. A second optional argument gives the offset, e.g. ``evaluate_in_past(expr,2)`` evaluates the expression ``expr`` at the time step :math:`n-2` and, beyond that, e.g. ``evaluate_in_past(expr,0.5)`` is equal to ``0.5*expr+0.5*evaluate_in_past(expr,1)``, i.e. with fractional values as second argument, one can linearly interpolate between two time steps.

This means that the trapezoidal rule :math:numref:`eqodetrapzrule` in residual form in pyoomph can be written as ``partial_t(...,scheme="BDF1")-evaluate_in_past(...,0.5)``, where we have cast the equation already in residual form, i.e. all terms are on the lhs and the rhs is zero.

As an example, let us - once again - consider the harmonic oscillator, but this time integrated with the trapezoidal rule (``"TPZ"``). The code for the equation class now reads

.. literalinclude:: oscillator_TPZ_scheme.py
   :language: python
   :start-at: class HarmonicOscillator(ODEEquations):
   :end-at: self.add_residual(residual)

It can be seen that indeed :math:numref:`eqodetrapzrule` is recovered by using ``scheme="BDF1"`` for the lhs and ``evaluate_in_past(...,0.5)`` for the rhs.

When using the trapezoidal rule for advancing in time, one also has to be consistent with the definition of the total energy, i.e. the argument which is passed to the :py:class:`~pyoomph.equations.generic.ODEObservables` class (as in the previous section) now has to take the time derivative of :math:`y` in the kinetic energy via the ``"BDF1"`` scheme, whereas the value of :math:`y` in the potential energy has to be evaluated at the average between the time steps :math:`n` and :math:`n-1`, i.e. via ``evaluate_in_past(...,0.5)``:

.. literalinclude:: oscillator_TPZ_scheme.py
   :language: python
   :start-at: #Calculate the total energy. Important to also stick to the convention: BDF1 derivative and evaluate_in_past(...,0.5)
   :end-at: eqs+=ODEObservables(Etot=total_energy) # Add the total energy as observable

Of course, this time stepping can be generalized, which is the so-called :math:`\theta`-*method*:

.. math:: \frac{1}{\Delta t^{(n)}}\left(\vec{U}^{(n)}-\vec{U}^{(n-1)}\right)=(1-\theta) \vec{F}\left(\vec{U}^{(n)},t^{(n)}\right)+\theta \vec{F}\left(\vec{U}^{(n-1)},t^{(n-1)}\right)\,

Obviously, :math:`\theta=0` coincides with ``"BDF1"`` and :math:`\theta=1/2` with the trapezoidal rule. To blend between these methods or any :math:`0\leq \theta <1`, one can just adjust the second argument of ``evaluate_in_past(...,theta)``. The case :math:`\theta=1` corresponds to the *explicit (forward) Euler method*, which is highly unstable and inaccurate.

Another method is the *midpoint rule* (``"MPT"``), which reads

.. math:: \frac{1}{\Delta t^{(n)}}\left(\vec{U}^{(n)}-\vec{U}^{(n-1)}\right)=\vec{F}\left(\frac{\vec{U}^{(n)}+\vec{U}^{(n-1)}}{2},\frac{t^{(n)}+t^{(n-1)}}{2}\right)\,.

If :math:`\vec{F}` is linear, the midpoint rule and the trapezoidal rule are identical, but for nonlinear :math:`\vec{F}` it is not. The midpoint rule can be used well for *symplectic integration*, i.e. it is expected that the total energy of the harmonic oscillator is quite well conserved with this method. As explained above, the function ``evaluate_in_past(...,0.5)`` gives the trapezoidal rule, not the midpoint rule. To obtain the midpoint rule, one has to use the function :py:func:`~pyoomph.expressions.generic.evaluate_at_midpoint` instead.

.. warning::

   If you use any history values, e.g. by :py:func:`~pyoomph.expressions.generic.evaluate_in_past` or :py:func:`~pyoomph.expressions.generic.evaluate_at_midpoint`, the compilation of the generated code is slower, i.e. you might experience some additional waiting time at the beginning. The reason is that one additional routine has to be generated and compiled to solve for steady solutions in that case, where all history values are identical to the current value. More on steady solutions can be found in :numref:`secODEstationarysolve`.
   

.. only:: html   
	
	.. container:: downloadbutton

		:download:`Download this example <oscillator_TPZ_scheme.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   
