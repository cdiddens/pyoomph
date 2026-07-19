.. _secODEtemporaladapt:

Temporal adaptivity
~~~~~~~~~~~~~~~~~~~

It is not always easy to find a good time step that is sufficiently accurate. Of course, you can make a guess and rerun the simulation with reduced time step to see whether the dynamics is affected by the time step. If not, you might have found a good time step. However, some systems have unpredictable dynamics, as e.g. chaotic systems, which might show very strongly fluctuating dynamics.

pyoomph allows one to flexibly adjust the time step during the simulation depending on an estimate of the error the current time step might have. To that end, the next time step is first predicted based on the previous steps, then a step is taken and the result is compared with the prediction. If the difference is too large, pyoomph will discard this step and retry with a smaller time step. At the same time, if the prediction and the computed solution match very well, pyoomph gradually tries to increase the time step.

Here, we will use the chaotic Lorenz system, known for its chaotic *strange attractor*, to illustrate this feature. The Lorenz system consists of three coupled non-linear ODEs of first order

.. math:: :label: eqodelorenz

   \begin{aligned}
   \partial_t x&=\sigma(y-x) \nonumber \\ 
   \partial_t y&=x(\rho-z)-y \\
   \partial_t z&=xy-\beta z \nonumber
   \end{aligned}

with three parameters :math:`(\sigma,\rho,\beta)`. Implementing this in pyoomph is trivial:

.. literalinclude:: adaptive_lorenz_attractor.py
   :language: python
   :start-at: class LorenzSystem(ODEEquations):
   :end-at: class LorenzProblem(Problem):

To add the feature of temporal adaptivity to the system, it is sufficient to combine the equations with a :py:class:`~pyoomph.equations.generic.TemporalErrorEstimator`

.. literalinclude:: adaptive_lorenz_attractor.py
   :language: python
   :start-at: class LorenzProblem(Problem):
   :end-at: self.add_equations(eqs@"lorenz_attractor")

:py:class:`~pyoomph.equations.generic.TemporalErrorEstimator` takes keyword arguments of the form ``variable_name=error_weight``, i.e. we can set to each of the ODE variables a different weighting for the computation of the temporal error between prediction and actual solution. In the code here, all weights have been set to unity, i.e. all variables :math:`x`, :math:`y`, :math:`z` are weighted equally in the determination of the temporal error.

Finally, the :py:meth:`~pyoomph.generic.problem.Problem.run` statement takes a keyword argument ``temporal_error``, which defines the error we are ready to accept. The smaller the temporal error, the finer the dynamics time steps, but the longer the computation takes.

.. literalinclude:: adaptive_lorenz_attractor.py
   :language: python
   :start-at: if __name__=="__main__":
   :end-at: problem.run(endtime=100,outstep=True,startstep=0.0001,temporal_error=0.005)

Note that also ``outstep=True`` was passed instead of ``numouts``. It will just output each step. Of course, also a ``startstep`` should be set so that the problem has a good guess how to start. If the value of ``temporal_error`` is set too large, the system might not show the correct dynamics, see :numref:`figodelorenzdyndt`. The accepted time steps are displayed in :numref:`figodelorenzdyndt2`.


..  figure:: plot_lorenz.*
	:name: figodelorenzdyndt
	:align: center
	:alt: Temporal adaptivity
	:class: with-shadow
	:width: 100%
	
	Dynamics of the Lorenz system with adaptive time stepping. Allowing too large errors will give the wrong dynamics. 

..  figure:: plot_lorenz_tstep.*
	:name: figodelorenzdyndt2
	:align: center
	:alt: accepted time steps
	:class: with-shadow
	:width: 70%
	
	Accepted time steps in the case of ``temporal_error=0.005``
	
.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <adaptive_lorenz_attractor.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		
