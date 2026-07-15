.. _solidoscillations:

Oscillations of a released torsion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So far, only stationary solutions of deformed solid bodies were considered. Now, we go for transient dynamics of a long three-dimensional beam. We start with a deformed configuration, specifically, by applying a torsion of the beam. On one side, the beam is fixed to a solid wall.

With the previous examples in mind, it is trivial to set up the problem case. However, here we consider physical units, i.e. we have to set typical scalings to let pyoomph nondimensionalize the equations and redimensionalize the output automatically. In particular, we need a spatial scale, a temporal scale and a scale for the mass density. With these, the equations can be nondimensionalized. A typical time scale for the solid dynamics can be obtained from Young's modulus, the density and the length:


.. literalinclude:: solid_oscillations.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: problem.run(10*T,outstep=0.1*T,temporal_error=1)

For the torsion, we use the original undeformed beam by accessing the Lagrangian coordinates and apply the deformation by setting an initial condition for the Eulerian mesh coordinates. Note that the code generation and compilation take a while, since the three-dimensional dynamics involves a lot of higher order tensors, inverse matrices (the contravariant metric tensor) and nonlinearities. In particular, the entries of the analytical Jacobian with respect to the moving mesh coordinates hence constitute long expressions, which bloat up the generated C code to over 3 megabytes.

..  figure:: solid_oscillations.*
	:name: figalesolidoscillations
	:align: center
	:alt: Oscillations of a beam when releasing a torsion
	:class: with-shadow
	:width: 90%

	Oscillations of a beam when releasing a torsion



.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <solid_oscillations.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    		
