.. _secodecustomharmosci:

Defining your own harmonic oscillator equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Up to now, we have loaded the predefined harmonic oscillator equations with the line

.. code:: python

   from pyoomph.equations.harmonic_oscillator import HarmonicOscillator

However, one of the main features of pyoomph is actually the definition of arbitrary equations. Instead of including the predefined equation class, we will write our own class, inheriting from the generic :py:class:`~pyoomph.generic.codegen.ODEEquations`. This will serve as an example of how to express arbitrary ODEs within pyoomph. At the same time, let us generalize the equation to include damping :math:`\delta` and driving :math:`f(t)` as follows

.. math:: \partial_t^2 y + 2\delta\partial_t y +\omega^2 y = f

The corresponding code reads as follows:

.. literalinclude:: custom_harmonic_oscillator.py
   :language: python
   :start-at: from pyoomph import * # Import pyoomph
   :end-at: self.add_residual(residual*testfunction(y))
   		

In the :py:meth:`~pyoomph.generic.codegen.ODEEquations.__init__` method, we take optional keyword arguments which have default values. These are then stored as members of the class. Of course, we have to call again the constructor of the super-class :py:class:`~pyoomph.generic.codegen.ODEEquations` using the Python builtin ``super``.

In the next step, the method :py:meth:`~pyoomph.generic.codegen.BaseEquations.define_fields` is overloaded. Inside this function, all unknowns of the ODE system have to be defined. Here, it is just the single unknown :math:`y`.

Finally, the nitty-gritty, the equation has to be defined. This happens in the method :py:meth:`~pyoomph.generic.codegen.BaseEquations.define_residuals`. To that end, the equation first has to be cast to a residual form, which can be done by putting all terms on one side of the equation:

.. math:: \partial_t^2 y + 2\delta\partial_t y +\omega^2 y - f =0

Of course, only the lhs of this equation is of relevance. Before adding it to the system of equations with the :py:meth:`~pyoomph.generic.codegen.BaseEquations.add_residual` method, it must be multiplied by a so-called *test function*, which can be obtained by the function :py:func:`~pyoomph.expressions.generic.testfunction`. This all happens in the line ``self.add_residual(residual*testfunction(y))``. Here, we only have one equation to be solved, i.e. the harmonic oscillator. In general, one will have a system of ODEs, i.e. multiple equations for multiple unknowns. This is where test functions become important, which will be explained in the next section.

The remainder of the code is very similar to before, but now also damping and driving will be considered:

.. literalinclude:: custom_harmonic_oscillator.py
   :language: python
   :start-at: # The remainder is almost the same is in the example nondim_harmonic_osci.py
   :end-at: problem.run(endtime=100,numouts=1000)

The output is plotted in :numref:`fignondimhocustom`.

.. _fignondimhocustom:

..  figure:: nondimhocustom.*
    :name: figcustomnondimho
    :align: center
    :alt: User defined harmonic oscillator
    :class: with-shadow
    :width: 100%
    
    Output for the user-defined harmonic oscillator equation with damping and driving.
    
.. only:: html    

	.. container:: downloadbutton

		:download:`Download this example <custom_harmonic_oscillator.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`    
