.. _eqpdelubric_spread:

Spreading of a droplet
~~~~~~~~~~~~~~~~~~~~~~

We can use the same equation class to calculate the spreading of a droplet. For that, we have to switch to an axisymmetric coordinate system and also make sure that the droplet can spread at all. For the latter, we must make sure that the droplet does not end at its radius :math:`R` with :math:`h(R,t)=0`, since this would exclude any change in the height according to :math:numref:`eqpdelubricationstrong`. A conventional way to resolve this is the addition of a precursor film with a thin thickness compared to the droplet. The thickness of this film will control the spreading velocity. Additionally, one may add a disjoining pressure. Thereby, one can e.g. enforce the spreading to stop at a finite contact angle:

.. literalinclude:: lubrication_spreading.py
   :language: python
   :start-at: from lubrication import *
   :end-at: problem.run(1000,outstep=True,startstep=0.01,maxstep=10,temporal_error=1)

..  figure:: lubric_spreading.*
	:name: figpdelubricspreading
	:align: center
	:alt: Spreading of a droplet
	:class: with-shadow
	:width: 80%

	Spreading of a droplet until the equilibrium contact angle is reached, which is enforced by the disjoining pressure.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <lubrication_spreading.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
