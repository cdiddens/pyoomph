.. _secODEstationarysolve:

Stationary solutions
~~~~~~~~~~~~~~~~~~~~

By simple time integration, we have seen how the transcritical normal form avoids unstable stationary solutions and tries to approach stable stationary solutions. While time integration can help to approach a stable stationary solution, one is unable to approach unstable stationary solutions that way. Also, it takes a lot of time steps to approach the stable ones. Instead, one can just jump to a stationary solution, both stable and unstable ones, by solving the stationary problem:

.. math:: 0=rx-x^2\,,

Of course, one could implement this equation by hand again, but it is better to reuse the full equation with :math:`\partial_t x`-term, i.e. :math:numref:`eqodetranscriticnf`, and just tell pyoomph to search for a stationary solution instead of integrating the temporal evolution. To that end, we first import the transcritical bifurcation equation and problem from the previous example :download:`bifurcation_transient_transcritical.py`. Afterwards, we use the :py:meth:`~pyoomph.generic.problem.Problem.solve` method of the problem, which exactly looks for a stationary solution:

.. code:: python

   from bifurcation_transient_transcritical import * #Import the equation and problem from the previous script

   if __name__=="__main__":
       with TranscriticalProblem() as problem:
           problem.quiet() # Do not pollute our output with all the messages

           problem.r.value=1 # Parameter value
           problem.x0=0.001 # Start slightly above the unstable solution x=0

           ode = problem.get_ode("transcritical")  # Get access the ODE (note: it will initialize the problem!)

           xvalue = ode.get_value("x") #Get the current value of x
           print(f"We are starting at x={xvalue}")

           problem.solve(timestep=None) # Solve without a timestep means stationary solve
           xvalue = ode.get_value("x") # Get the current value of x
           print(f"Currently, we are at the stationary solution x={xvalue} with r={problem.r.value}")

           ode.set_value(x=0.8) # Set the current value of x
           problem.solve() # we can omit timestep=None, since it is default
           xvalue = ode.get_value("x")
           print(F"Currently, we are at the stationary solution x={xvalue} with r={problem.r.value}")

First, we tell the problem class to be :py:meth:`~pyoomph.generic.problem.Problem.quiet`, so that there is no output from the code generation and solution. Thereby, the ``print`` statements can be better found in the output. We can get access to the ODE by using the :py:meth:`~pyoomph.generic.problem.Problem.get_ode` method. Note that this will initialize the problem, i.e. generate the code and so on, since before that the problem does not know about the existence of an ODE domain called ``"transcritical"``. This domain is added in :py:meth:`~pyoomph.generic.problem.Problem.define_problem`. With the reference to the ODE, we can use :py:meth:`~pyoomph.meshes.mesh.ODEStorageMesh.set_value` and :py:meth:`~pyoomph.meshes.mesh.ODEStorageMesh.get_value` to access the current value of :math:`x`. Initially, we are close to the unstable solution :math:`x=0`, so the :py:meth:`~pyoomph.generic.problem.Problem.solve` statement will likely run into this solution, which is indeed the case. In the second solve, we are close to the stable solution at :math:`x=r=1`, so we run into this with the second invocation of :py:meth:`~pyoomph.generic.problem.Problem.solve`.

Obviously, one can easily find a stationary solution by the :py:meth:`~pyoomph.generic.problem.Problem.solve` function. However, this only works if one is close to that solution. If one tries to e.g. start at :math:`x=0.5`, i.e. by ``ode.set_value(x=0.5)`` followed by a ``problem.solve()`` statement, it will not find any solution. The reason lies in the internal *Newton method*, which requires the *Jacobian*, which is singular at :math:`x=0.5`. In fact, we are exactly between both solutions, so the system cannot decide which way to go. In that case, one can solve a few steps by temporal integration, either via the :py:meth:`~pyoomph.generic.problem.Problem.run` command or via ``solve(timestep=...)`` to move a bit away from the singularity. After that, the normal :py:meth:`~pyoomph.generic.problem.Problem.solve` command will converge to a stationary solution again. Note that the numerical stationary solutions are not exactly :math:`0` and :math:`1`. Deviations come from the fact that (i) we have a nonlinear problem, i.e. the solution procedure is stopped after reaching sufficient accuracy and (ii) numerical calculations on computers always have a limited accuracy due to the finite accuracy of floating point numbers. The first issue can be improved by passing ``newton_solver_tolerance=1e-20`` or even smaller numbers to the :py:meth:`~pyoomph.generic.problem.Problem.solve` command. However, in larger systems the latter, i.e. the finite accuracy of float arithmetic, might hamper reaching tiny accuracy thresholds.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <bifurcation_stationary_transcritical.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		