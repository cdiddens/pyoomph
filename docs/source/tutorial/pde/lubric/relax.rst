.. _eqpdelubric_relax:

Relaxation of a perturbation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As a first problem class, let us calculate the relaxation of a thin film with a modulation:

.. literalinclude:: lubrication.py
   :language: python
   :start-at: class LubricationProblem(Problem):
   :end-at: problem.run(50,outstep=True,startstep=0.25)

The result is depicted in :numref:`figpdelubrication`.

..  figure:: lubrication.*
	:name: figpdelubrication
	:align: center
	:alt: Relaxation of a perturbed surface $h$ in the lubrication limit
	:class: with-shadow
	:width: 70%

	Relaxation of a perturbed surface :math:`h` in the lubrication limit.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <lubrication.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
