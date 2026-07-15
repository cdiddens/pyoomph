Damped harmonic oscillator
~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider a damped harmonic oscillator with driving, i.e.

.. math::

   \begin{aligned}
   \partial_t^2 y+\delta\partial_t y+\omega_0^2 y = F\cos(\omega t)\,.
   \end{aligned}

Due to the damping, all transients will decay and after some time, the system will converge to the response to the driving frequency :math:`\omega`, i.e. to

.. math::

   \begin{aligned}
   y=A\cos(\omega t + \varphi)
   \end{aligned}

Here :math:`A` is the response amplitude and :math:`\varphi` is the phase shift with respect to the driving. pyoomph can calculate this response automatically for arbitrary problems, i.e. also complex PDEs on moving meshes. To that end, any potential nonlinearities will be linearized around the stationary solution (here, :math:`y=0`). First, we define the harmonic oscillator with arbitrary driving and assemble it in a problem. The oscillator must be written as a first order system in time, because eventually, an eigenproblem will be solved:

.. literalinclude:: linear_response_oscillator.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: self+=eqs@"oscillator"

The trivial way of getting the response is to just integrate over a sufficiently long time and extract the response :math:`A` and the angle :math:`\varphi` from the output, i.e.

.. code:: python

   with DampedHarmonicOscillatorProblem() as problem:
        # Trivial, but long way: integrate in time, extract response manually from the output        
        problem.run(100*second,outstep=0.1*second)

However, with the periodic driving response tool, you can scan the linear response quickly:

.. literalinclude:: linear_response_oscillator.py
   :language: python
   :start-at: # Quick way of scanning
   :end-at: outfile.add_row(omega*second,A_num,phi[yindex],A_analytic,phi_analytic)

Before the problem is initialized, we must create a :py:class:`~pyoomph.utils.periodic_driving_response.PeriodicDrivingResponse` object and attach it to the problem. This one will introduce a nondimensional, undamped harmonic oscillator with a variable angular frequency :math:`\omega`, i.e. :math:`\partial_t z+\omega^2 z=0` to the problem. The method :py:meth:`~pyoomph.utils.periodic_driving_response.PeriodicDrivingResponse.get_driving_mode` gives back :math:`z`, i.e. internally, we use this auxiliary harmonic oscillator to impose the driving on the harmonic oscillator equation for :math:`y`. To obtain the response, we first must find the equation number corresponding to :math:`y`, which can be done by describing the degrees of freedom of the problem with the :py:meth:`~pyoomph.generic.problem.Problem.get_dof_description` and finding the correct index in the degrees of freedom. By :py:meth:`~pyoomph.utils.periodic_driving_response.PeriodicDrivingResponse.iterate_over_driving_frequencies`, we can scan a full range of driving frequencies :math:`\omega` in a loop. The ``response`` is a complex eigenvector, but it can be split into amplitude and phase by :py:meth:`~pyoomph.utils.periodic_driving_response.PeriodicDrivingResponse.split_response_amplitude_and_phase`. By extracting the right component corresponding to :math:`y`, we get the amplitude and phase directly, correctly account for any physical dimensions and compare it with the analytical solution in the output. The result is plotted in :numref:`figstabilitypdrosci`.

..  figure:: pdrosci.*
	:name: figstabilitypdrosci
	:align: center
	:alt: Linear response of a damped harmonic oscillator to a periodic driving
	:class: with-shadow
	:width: 90%

	Numerical linear response to a periodic driving with :math:`F\cos(\omega t)` of a harmonic oscillator with :math:`\omega_0=1` and damping :math:`\delta=0.1`. The analytical result is also plotted.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <linear_response_oscillator.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
 
