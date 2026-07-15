.. _secpdelubric_coalescence:

Coalescence of droplets
~~~~~~~~~~~~~~~~~~~~~~~

For a reasonable coalescence, we have to solve the lubrication equation on a two-dimensional lateral plane. Due to symmetry, only one half of this plane is solved. Of course, it is beneficial to use the spatial adaptivity to resolve the domain accurately and optimize the required computational effort:

.. literalinclude:: lubrication_coalescence.py
   :language: python
   :start-at: from lubrication_spreading import * # Import the previous example problem
   :end-at: problem.run(1000,outstep=True,startstep=0.01,maxstep=10,temporal_error=1,spatial_adapt=1)

We just reuse the previous problem by inheritance to get access to the parameters as e.g. ``R``, ``sigma``, etc. Of course, the parameter ``distance`` and the size of the mesh ``Lx`` are additionally required. With the :py:attr:`~pyoomph.genric.problem.Problem.max_refinement_level` of the :py:class:`~pyoomph.generic.problem.Problem` base class, the maximum refinement is controlled. The rest is analogous to the previous example, however, in Cartesian coordinates with a 2d mesh and with two droplets.

..  figure:: lubric_coalescence.*
	:name: figpdelubriccoalescence
	:align: center
	:alt: Coalescence of two droplets
	:class: with-shadow
	:width: 80%

	Coalescence of two droplets.


One can rather easily add e.g. (in)soluble surfactants or a mixture composition field by adding a corresponding advection-diffusion field on the domain. When redefining the surface tension ``sigma`` to be dependent on this additional field, it is easy to reproduce *delayed coalescence* due to Marangoni dynamics. Similarly, it is also straightforward to use dimensions here and use the non-dimensionalization in pyoomph to solve the dynamics of real droplets.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <lubrication_coalescence.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
