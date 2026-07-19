A nonlinear oscillator
~~~~~~~~~~~~~~~~~~~~~~

So far, all considered equations were linear ODEs or linear ODE systems. However, unlike e.g. FEniCS, there is no difference in defining linear and nonlinear equations in pyoomph, a feature which is directly inherited from oomph-lib. One famous nonlinear oscillator is the *Van der Pol oscillator*

.. math:: :label: eqvdposci
	
	\partial_t^2 y-\mu\left(1-y^2\right)\partial_t y+y=0\,. 

Here, the parameter :math:`\mu` controls the nonlinearity. It is obvious that a positive value of :math:`\mu` will damp the oscillation whenever :math:`y^2>1` and enhance the amplitude whenever :math:`y^2<1`. The straightforward way of implementing this equation would be again to write an equation class:

.. literalinclude:: van_der_pol_method_1.py
   :language: python
   :start-at: class VanDerPolOscillator(ODEEquations):
   :end-at: self.add_residual(residual*testfunction(y))

It is not a problem to add the nonlinear damping term to the residuals - a step where FEniCS would complain unless explicitly implemented as a nonlinear problem. 

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <van_der_pol_method_1.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`    

There is also another way to implement exactly the same equation by using already implemented equations. In the following, we make use of the predefined harmonic oscillator that comes with pyoomph:

.. literalinclude:: van_der_pol_method_2.py
   :language: python
   :start-at: # import the predefined harmonic oscillator equation
   :end-at: super(VanDerPolOscillator,self).__init__(name="y",damping=damping,omega=1)

Here, we use inheritance, i.e. the methods :py:meth:`~pyoomph.generic.codegen.BaseEquations.define_fields` and :py:meth:`~pyoomph.generic.codegen.BaseEquations.define_residuals` will be inherited from the super-class :py:class:`~pyoomph.equations.harmonic_oscillator.HarmonicOscillator`, which will calculate

.. math:: \partial_t^2 y+2\delta \partial_t y +\omega^2 y=0\,.

If we replace :math:`\delta=-1/2 \mu(1-y)^2`, we get in fact the Van der Pol oscillator :math:numref:`eqvdposci`. This fact is used here: when calling the constructor ``__init__`` of the super-class :py:class:`~pyoomph.equations.harmonic_oscillator.HarmonicOscillator`, we pass exactly this expression as damping parameter and thereby recover the Van der Pol oscillator. Obviously, there are always multiple ways in pyoomph to achieve the same goal. A plot of the numerical solution is shown in :numref:`figodevdpoln`.

..  figure:: vdpol.*
	:name: figodevdpoln
	:align: center
	:alt: Van der Pol oscillator
	:class: with-shadow
	:width: 70%
	
	Output for the nonlinear Van der Pol oscillator.
	
.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <van_der_pol_method_2.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   
