
.. _secpdeksebiftrack:

Stability via bifurcation tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We could now perform similar scans for different :math:`\delta`, but there is a simpler route, namely bifurcation tracking. We can instruct pyoomph to find the fold bifurcation and its corresponding value of :math:`\gamma\approx 0.2826` directly:

.. literalinclude:: kuramoto_sivanshinsky_bifurcation.py
   :language: python
   :start-at: from kuramoto_sivanshinsky_arclength_eigen import * # Import the previous example problem
   :end-at: print("FOLD BIFURCATION HAPPENS AT",problem.param_gamma.value)

To that end, we first move close to the bifurcation, i.e. to :math:`\gamma=0.28` and :py:meth:`~pyoomph.generic.problem.Problem.solve` to find a good guess. Then, we :py:meth:`~pyoomph.generic.problem.Problem.activate_bifurcation_tracking` for a ``"fold"`` bifurcation in :math:`\gamma`. Within the next :py:meth:`~pyoomph.generic.problem.Problem.solve` command, the value of :math:`\gamma` will be adjusted (i.e. :math:`\gamma` is in fact a degree of freedom) so that the system is directly at the fold bifurcation. We also output the eigenvector directly at the fold bifurcation. To that end, another :py:class:`~pyoomph.output.meshio.MeshFileOutput` is added, but with the arguments ``eigenvector=0`` (meaning the zeroth eigenvector) and ``eigenmode="real"`` (i.e. considering the real part, although this particular eigenvector is real anyhow). We furthermore must supply a ``filetrunk`` to prevent overwriting of the output files of the solution itself.

Once we are on the bifurcation, we can sweep over :math:`\delta` and follow the position of the fold bifurcation. As long as the bifurcation tracking is active :math:`\gamma` will be adjusted to stay on the fold bifurcation, i.e. we get a curve :math:`\gamma_\text{fold}(\delta)`, which is written to file:

.. literalinclude:: kuramoto_sivanshinsky_bifurcation.py
   :language: python
   :start-at: hexfold_file = open(os.path.join(problem.get_output_directory(), "hexfold.txt"), "w")

The result, i.e. the location of the fold bifurcation, is depicted in :numref:`figpdeksefold2`.

..  figure:: kse_fold2.*
	:name: figpdeksefold2
	:align: center
	:alt: Temporal integration of the damped Kuramoto-Sivashinsky equation
	:class: with-shadow
	:width: 70%

	Emergence of a hexagonal dot pattern by the damped Kuramoto-Sivashinsky equation starting from a random initial condition.

Similarly, we can set the other :py:class:`~pyoomph.equations.generic.InitialCondition` to start with hexagonal holes or stripe patterns and find the bifurcations.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <kuramoto_sivanshinsky_bifurcation.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    

