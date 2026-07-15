Womersley flow
~~~~~~~~~~~~~~

As an example for the action of inertia in this equation, we calculate a Womersley pipe flow, i.e. a flow through a pipe driven by an oscillating pressure:

.. literalinclude:: navier_stokes.py
   :language: python
   :start-at: class WomersleyFlowProblem(Problem):
   :end-at: problem.run(1,outstep=True,startstep=0.01,spatial_adapt=1)

Due to the inertia, the flow reversal does not happen instantaneously, but shows a Womersley flow profile, see :numref:`figpdewomersley`.

..  figure:: womersley.*
	:name: figpdewomersley
	:align: center
	:alt: Womersley flow in a pipe
	:class: with-shadow
	:width: 100%

	Womersley flow in a pipe


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <navier_stokes.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
