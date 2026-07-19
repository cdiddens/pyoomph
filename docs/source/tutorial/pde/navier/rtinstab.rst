.. _secpdertinstab:

Rayleigh-Taylor instability
---------------------------

Now, we can make use of the power of pyoomph to easily couple the Navier-Stokes equations with an advection-diffusion equation. Here, we will solve a denser fluid residing atop a less dense fluid. We apply the *Boussinesq approximation*, i.e. the mass density :math:`\rho` is assumed to be constant in the continuity equation and also in the inertia term, but the bulk force will depend on the varying density. We set the bulk force to :math:`\vec{f}=-100 c \vec{e}_y` and solve for the composition field :math:`c` that ranges from :math:`0` (bottom light fluid) to :math:`1` (heavier top fluid). We choose a small diffusivity so that advection can dominate. For the equations for the advection-diffusion of :math:`c` and the velocity-pressure pair, we use the predefined equations from pyoomph. The corresponding classes are :py:class:`~pyoomph.equations.navier_stokes.NavierStokesEquations` and :py:class:`~pyoomph.equations.advection_diffusion.AdvectionDiffusionEquations`. However, we could also use the equations we have developed in the previous sections instead.

.. literalinclude:: rayleigh_taylor_instability.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: problem.run(10,numouts=50,spatial_adapt=1)

If you have read the tutorial up to here, you should understand all steps. The idea is to create a :py:class:`~pyoomph.equations.navier_stokes.NavierStokesEquations` object with a body force depending on the variable ``var("c")``. This variable is defined and solved in the :py:class:`~pyoomph.equations.advection_diffusion.AdvectionDiffusionEquations`, which in turn get the field ``var("velocity")`` for the advection. Thereby, both parts are coupled. Since we allow no normal outflow, we have to fix a single pressure degree, which we do in the lower left corner. The results are depicted in :numref:`figpderayleightaylor`.

..  figure:: rayleigh_taylor.*
	:name: figpderayleightaylor
	:align: center
	:alt: Miscible Rayleigh-Taylor instability.
	:class: with-shadow
	:width: 100%

	Miscible Rayleigh-Taylor instability by coupling the Navier-Stokes equations with an advection-diffusion equation.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <rayleigh_taylor_instability.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
