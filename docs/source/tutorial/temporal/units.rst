.. _secodephysdims:

Using physical units and nondimensionalization
----------------------------------------------

As every numerical software in its internal core, pyoomph is calculating just with floating-point arithmetic, i.e. with numbers of limited precision and in particular without physical units. Due to the limited precision and numerical bounds, it is beneficial to nondimensionalize the equations in a way that the processor never has to deal with very tiny or huge numbers during the calculation. Therefore, one should avoid nondimensionalizing the equations by just omitting the units, i.e. just substituting :math:`1000` for a mass of :math:`1000\:\mathrm{kg}`, which is even more important for e.g. calculations on extreme scales, e.g. in quantum mechanics or astrophysics.

Nondimensionalization furthermore comes with the benefit to identify nondimensional numbers that entirely characterize the system dynamics. However, sometimes, in terms of usability of a numerical code for real applications, it is easier to just input the dimensional quantities directly instead of reading through the particular required nondimensional characteristic numbers required for the numerical code. pyoomph offers therefore the possibility to input quantities with units and automatically nondimensionalize the equations.

Let us discuss this feature once more on the simple basis of the harmonic oscillator. In a mass spring system, one has the mass :math:`m` (in :math:`\:\mathrm{kg}`), the spring constant :math:`k` (in :math:`\:\mathrm{N}/\mathrm{m}`), the dimensional time :math:`t` (in :math:`\:\mathrm{s}`) and the displacement :math:`x` (in :math:`\:\mathrm{m}`), i.e. the dimensional equation reads

.. math:: \partial_t^2x+\frac{k}{m}x=0\,.

However, in pyoomph, we write these equations as a product with a test function :math:`\chi` corresponding to :math:`x`, i.e.

.. math:: :label: eqodedimhostart

   \left(\partial_t^2x+\frac{k}{m}x\right)\chi=0\,.

To nondimensionalize this equation, one introduces characteristic scales, namely a typical displacement :math:`X` and a time scale :math:`T`, defines the nondimensional quantities via :math:`x=X\tilde{x}` and :math:`t=T\tilde{t}`. Additionally, we allow the test function :math:`\chi` to be associated with a dimensional unit, i.e. :math:`\chi=C\tilde{\chi}`. This gives the equation

.. math:: \left(\frac{X}{T^2}\partial_{\tilde{t}}^2\tilde{x}+X\frac{k}{m}\tilde{x}\right)C\tilde{\chi}=\left(\frac{CX}{T^2}\partial_{\tilde{t}}^2\tilde{x}+CX\frac{k}{m}\tilde{x}\right)\tilde{\chi}=0\,,

The scale :math:`C` of the test function is arbitrary at the moment, but we can choose it in such a way that the coefficient in front of the term :math:`\partial_{\tilde{t}}^2\tilde{x}` becomes unity by setting :math:`C=T^2/X`. When using this test function scale, we arrive at

.. math:: :label: eqodedimhoend

   \left(\partial_{\tilde{t}}^2\tilde{x}+\frac{kT^2}{m}\tilde{x}\right)\tilde{\chi}=0\,,
   

It is important that the residual form may not have any dimensional quantities left. When the dimensions of :math:`k` and :math:`m` and the time scale :math:`T` are correctly chosen, this will be the case also for the second term in the brackets. Obviously, since we have a homogeneous linear equation, no suitable scale for the displacement :math:`x`, i.e. the scale factor :math:`X`, can be found. However, we can identify a reasonable time scale by e.g. :math:`T=\sqrt{\frac{m}{k}}`, which simplifies the equation to having only coefficients of unity. The displacement scale :math:`X` can e.g. be set by the initial displacement :math:`X=x(t=0)`.

In pyoomph, we can do the same steps as follows within the definition of our equation class:

.. literalinclude:: dimensional_oscillator_with_units.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: self.add_residual(eq*testfunction(x)) # testfunction(x) will expand to T**2/X * ~chi

We have implemented the equation as ``partial_t(x,2)+m/k*x``, i.e. it will have the units of ``x`` divided by the unit of the time squared. To get rid of these scales, we have passed the argument ``testscale`` to the :py:meth:`~pyoomph.generic.codegen.ODEEquations.define_ode_variable` method. Thereby, we set the scale factor :math:`C` of the test function to be :math:`T^2/X`, where :math:`T` and :math:`X` can be obtained by the function :py:func:`~pyoomph.expressions.generic.scale_factor`, i.e. by ``scale_factor("temporal")`` and ``scale_factor("x")``, respectively. pyoomph will automatically expand all parts that are added to the residuals into scales times the nondimensional equivalent, i.e. internally exactly the same steps are done from :math:numref:`eqodedimhostart` to :math:numref:`eqodedimhoend`.

We have not yet specified the scales :math:`X` and :math:`T`. Both will be done at problem level:

.. literalinclude:: dimensional_oscillator_with_units.py
   :language: python
   :start-at: class DimensionalOscillatorProblem(Problem):
   :end-at: problem.run(endtime=10*second,numouts=1000) #endime is now also in seconds!

In the constructor, we see how we can simply define dimensional units, i.e. just by multiplying or dividing with units like ``meter``, ``newton`` or whatever. In the :py:meth:`~pyoomph.generic.problem.Problem.define_problem` method, we pass these dimensional parameters to the equation and also the initial condition for ``x`` gets a dimensional value. However, in order to make this work, we have to introduce the typical scalings for the time and the displacement. The time scale can be set via the argument ``temporal`` in the method :py:meth:`~pyoomph.generic.problem.Problem.set_scaling`, whereas the scaling of any other variable, i.e. here ``x``, can also be set with this method. As discussed before, a good time scale is :math:`\sqrt{m/k}` and the initial displacement is used as scale for :math:`x`. The rest remains the same as before, except that we have to use a dimensional time for the ``endtime`` keyword argument in :py:meth:`~pyoomph.generic.problem.Problem.run`, since our time is now dimensional. Also, after running the simulation, the output file will have units in the header. This is beneficial, since it is not required to redimensionalize the output to compare it e.g. against experimental data. It is important to note that :py:func:`~pyoomph.expressions.generic.scale_factor` calls in the equation class gives unity if there is no scaling set in the problem class. This allows using the same equation class for dimensional and nondimensional calculations. For the latter case, of course, the variables ``mass``, ``spring_constant`` and ``initial_displacement`` may not contain units and also the :py:meth:`~pyoomph.generic.problem.Problem.run` statement needs a nondimensional numeric value for the ``endtime``. Unfortunately, one still has to set good values for the scaling by hand via the :py:meth:`~pyoomph.generic.problem.Problem.set_scaling` method. While for the harmonic oscillator, it would be possible to guess a good scaling just from the equation (as we have done with the time scale), it is in general, in particular for highly coupled systems with multiple driving mechanisms, not feasible.

Let us discuss once more what is happening internally in pyoomph here: After :py:meth:`~pyoomph.generic.problem.Problem.define_problem` is processed, the C code generation is invoked, which will call among others the functions :py:meth:`~pyoomph.generic.codegen.BaseEquations.define_fields` and :py:meth:`~pyoomph.generic.codegen.BaseEquations.define_residuals` of the equation class. Here, the :py:func:`~pyoomph.expressions.generic.scale_factor` quantities will be expanded to the quantities set by the :py:meth:`~pyoomph.generic.problem.Problem.set_scaling` method of the problem class. Furthermore, when evaluating the added residuals, all occurrences of :py:func:`~pyoomph.expressions.generic.var` will be replaced by the scale factor times the nondimensional variable, which is accessible in pyoomph via :py:func:`~pyoomph.expressions.generic.nondim`. Here, ``var("x")`` will be replaced by ``scale_factor("x")*nondim("x")``, which is exactly the step we performed analytically before, i.e. :math:`x=X\tilde{x}`. In a similar manner, ``partial_t(...,2)`` will be nondimensionalized to a nondimensional variant of ``partial_t(...,2)`` divided by the square of the temporal scale. When the scaling and the equation is set up correctly, all units, e.g. ``meter`` etc., will cancel out and one arrives at a nondimensional equation. In fact, due to the chosen scaling in this particular problem, always the same nondimensional equation will be solved, namely :math:`\partial_{\tilde{t}}^2\tilde{x}+x=0` with :math:`\tilde{x}(\tilde{t}{=}0)=1`, no matter what values are set for ``mass``, ``spring_constant`` and ``initial_displacement``.

The usage of units is also helpful to check for consistency: Whenever pyoomph cannot cancel the units in the residual, there is either a scaling not set appropriately, the equation is inconsistent or a parameter has a wrong unit. In this case, pyoomph will report an error.

Finally, instead of setting a scale at problem level, it is also possible to set a scale only for a particular domain, i.e. here for the domain ``"harmonic_oscillator"``. Instead of passing ``x=self.initial_displacement`` as argument for the :py:meth:`~pyoomph.generic.problem.Problem.set_scaling` method at problem level, one can also augment the equation ``eqs`` with a :py:class:`~pyoomph.equations.generic.Scaling` object, i.e. via ``eqs+=Scaling(x=self.initial_displacement)`` instead. This allows introducing different scalings for variables with the same name on different domains. Of course, it does not make any sense to have individual temporal scales, as the time is a global variable.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <dimensional_oscillator_with_units.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   	
