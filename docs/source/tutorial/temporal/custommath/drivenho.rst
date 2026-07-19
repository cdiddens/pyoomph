A harmonic oscillator driven by a trapezoidal forcing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let us again consider a nondimensional harmonic oscillator, but now with a custom driving function, which resembles a trapezoidal pulse. This custom pulse function can be implemented in pyoomph via the :py:class:`~pyoomph.expressions.cb.CustomMathExpression` class from the :py:mod:`pyoomph.expressions.cb` module as follows:

.. literalinclude:: custom_math_driven_oscillator.py
   :language: python
   :start-at: from pyoomph.expressions.cb import * # Custom math expressions
   :end-at: return 0.0 # after the plateau

In the constructor, we take the parameters that are fixed during the simulation, namely the quantities describing the shape of the pulse. Then, the method :py:meth:`~pyoomph.expressions.cb.CustomMathExpression.eval` has to be implemented, which takes a list object ``arg_array`` as parameters. In this list object, the current numerical values of the passed parameters (here, it will be the time :math:`t` later on) are stored. Based on this value, we return the current value of the pulse.

..  figure:: trapezoidal_driving.*
	:name: figodetrapezoidaldriving
	:align: center
	:alt: Using custom math expressions for a trapezoidal driving
	:class: with-shadow
	:width: 70%
	
	Using a :py:class:`~pyoomph.expressions.cb.CustomMathExpression`, we can implement custom functions, here the trapezoidal driving.


.. warning::

   All custom functions must be deterministic on their input arguments, i.e. evaluating the function :py:meth:`~pyoomph.expressions.cb.CustomMathExpression.eval` multiple times for the same input must yield the same result. This rules out any contribution of random numbers or any dependence on the degrees of freedom or parameters which are not passed via the argument list ``arg_array``.

The problem class looks like this, where we reuse the predefined :py:class:`~pyoomph.equations.harmonic_oscillator.HarmonicOscillator` equation class:

.. literalinclude:: custom_math_driven_oscillator.py
   :language: python
   :start-at: class TrapezoidallyDrivenOscillatorProblem(Problem):
   :end-at: problem.run(endtime=100,numouts=1000)

The result is depicted in :numref:`figodetrapezoidaldriving`.

.. only:: html
	
	.. container:: downloadbutton

		:download:`Download this example <custom_math_driven_oscillator.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`  
