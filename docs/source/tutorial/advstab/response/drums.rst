.. _secadvstabdrumresponse:

Drums getting excited by a guitar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A band has a rest during rehearsal. As usual, the guitar guy cannot resist playing. By the incident sound wave, the drums are put into motion. We solve the drum excitation height :math:`h` by a driven damped wave equation on a circular membrane, where we assume axisymmetric modes only, i.e.

.. math:: \partial_t^2 h + \delta \partial_t h=c^2 \nabla^2 h + F\cos \omega t

where :math:`\omega=2\pi f` is the frequency of the guitar sound incident as a planar wave with a forcing amplitude of :math:`F`. :math:`c` depends on the drum and the surrounding gas and :math:`\delta` is a damping coefficient. We demand that :math:`h(r=R)=0` at the radius :math:`R` of the drum.

As usual, we start with the drum equation in a coordinate-system-independent weak formulation:

.. literalinclude:: linear_response_drum.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: self.add_weak(self.c**2*grad(h),grad(h_test))

For the problem, we just define it on a line mesh in axisymmetric coordinates, which are indeed axisymmetric polar coordinates. Thereby, only axisymmetric modes are allowed. Alternatively, one could use a 2d circular Cartesian domain to also allow for nonaxisymmetric modes.

.. literalinclude:: linear_response_drum.py
   :language: python
   :start-at: class DrumProblem(Problem):
   :end-at: self+=eqs@"drum"

The general procedure is the same as before for a simple harmonic oscillator. However, here, we project the response on different Bessel modes:

.. literalinclude:: linear_response_drum.py
   :language: python
   :start-at: with DrumProblem() as problem:
   :end-at: outfile.add_row(pdr.get_driving_frequency()/hertz,*bessel_data)

To obtain the full response data, we can access the eigenvector of the problem. Here, the response is stored in the ``eigenvector=0``. The nondimensional eigenfunction is stored as real and imaginary contributions, which can be accessed by :py:meth:`~pyoomph.generic.problem.Problem.get_cached_mesh_data` for each mesh. Then, cubic interpolators are generated and in the loop, the individual Bessel modes are projected. The result is plotted in :numref:`figstabilitypdrdrum`.

..  figure:: pdrdrum.*
	:name: figstabilitypdrdrum
	:align: center
	:alt: Response of an excited drum
	:class: with-shadow
	:width: 90%

	Response of the drum separated into different Bessel modes that fulfill :math:`h(r=R)=0`. The dashed lines indicate the analytical eigenfrequencies of the undamped wave equation.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <linear_response_drum.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
