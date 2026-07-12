Calculating eigenvalues and eigenfunctions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the previous example, we have found stationary solutions, but we could not obtain the stability of these solutions. To explicitly calculate the eigenvalues, we can use the method :py:meth:`~pyoomph.generic.problem.Problem.solve_eigenproblem` of the problem class, which requires the number of eigenvalues we want to calculate. Here, there is just one degree of freedom, namely :math:`x`, so we should only aim for a single eigenvalue. The method will return a list of eigenvalues (complex numbers) and a list of the corresponding eigenvectors. Of course, using :py:meth:`~pyoomph.generic.problem.Problem.solve_eigenproblem` is only meaningful if the problem is currently on a stationary solution.

The rest is more or less the same as the previous code, i.e. first importing the problem and equation from the previous example :download:`bifurcation_transient_transcritical.py` and then using :py:meth:`~pyoomph.generic.problem.Problem.solve_eigenproblem` after finding the stationary solutions via the :py:meth:`~pyoomph.generic.problem.Problem.solve` call.

.. code:: python

   from bifurcation_transient_transcritical import * #Import the equation and problem from the previous script

   if __name__=="__main__":
       with TranscriticalProblem() as problem:
           problem.quiet() # Do not pollute our output with all the messages

           problem.r.value=1 # Parameter value

           ode = problem.get_ode("transcritical")  # Get access the ODE (note: it will initialize the problem!)

           for startpoint in [0.001,0.8]: #Take different start points
               ode.set_value(x=startpoint) #Start there
               problem.solve() # Solve for the stationary solution
               xvalue = ode.get_value("x") # Get the current value of x
               print(f"Starting at {startpoint} gives the stationary solution x={xvalue} with r={problem.r.value}")
               eigen_vals,eigen_vects=problem.solve_eigenproblem(1)
               print("Eigenvalues are "+str(eigen_vals))
               print("Thus, this solution is "+("unstable" if eigen_vals[0].real>0 else "stable"))


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <bifurcation_eigenvalues_transcritical.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		