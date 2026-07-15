Naive implementation
~~~~~~~~~~~~~~~~~~~~

Let us first ignore this complication and implement the equation naively. We will add a flag ``advection_in_partial_integration`` to choose between both different weak forms of the advection term:

.. literalinclude:: convdiffu_simple.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: self.add_residual(time_scheme("TPZ",weak(partial_t(c),phi)+advection+weak(self.D*grad(c),grad(phi))))

Depending on the value of ``advection_in_partial_integration``, we either use :math:numref:`eqpdeconvdiffuweakA` or :math:numref:`eqpdeconvdiffuweakB` for the weak form. Furthermore, we changed the time stepping from the default ``"BDF2"`` to ``"TPZ"``, which can be of advantage (cf. :numref:`secODEtimescheme` for time stepping methods).

As a problem class, we use a bump which is swirled around by one period. When the diffusivity is low, we expect the bump to be only slightly smaller in amplitude and only slightly coarser due to diffusion. The problem class hence reads:

.. literalinclude:: convdiffu_simple.py
   :language: python
   :start-at: class ConvectionDiffusionProblem(Problem):
   :end-at: self.add_equations(eqs@"domain") # adding the equation

The most interesting thing here is that we can also define a :py:class:`~pyoomph.equations.generic.SpatialErrorEstimator` based on history values. Instead of passing keyword arguments, we can also pass positional arguments to the :py:class:`~pyoomph.equations.generic.SpatialErrorEstimator`. The error estimator requires gradients, but these can also be evaluated at previous time steps. This ensures that the wake remains more finely resolved.

The run code is again simple:

.. literalinclude:: convdiffu_simple.py
   :language: python
   :start-at: if __name__=="__main__":
   :end-at: problem.run(1,outstep=0.01,maxstep=0.0025,spatial_adapt=1,temporal_error=1)

Using ``outstep=0.01`` in the :py:meth:`~pyoomph.generic.problem.Problem.run`, we will get 100 outputs, but due to ``maxstep=0.0025``, we solve at least 4 times per output. ``spatial_adapt=1`` will perform, as usual, one spatial adaption per solve, whereas ``temporal_error=1`` just ensures that the time step gets reduced when it does not converge.

Results at different times are depicted in :numref:`figpdesimpleconvdiffu`.

..  figure:: simpleconvdiffu.*
	:name: figpdesimpleconvdiffu
	:align: center
	:alt: Advecting a bump with low diffusivity.
	:class: with-shadow
	:width: 100%

	Advecting a bump with low diffusivity.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <convdiffu_simple.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
