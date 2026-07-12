.. _secspatialadapt:

Spatial adaptivity
~~~~~~~~~~~~~~~~~~

The previous example has a very steep source function and due to the boundary conditions, also steep gradients in the solution :math:`u` can be expected near the corners due to the different types of imposed boundary conditions. Pyoomph offers a simple way to automatically refine the mesh where necessary by adding a :py:class:`~pyoomph.equations.generic.SpatialErrorEstimator` object to the equations. We could either modify the previous example directly or use inheritance to re-use the previous example and just add the :py:class:`~pyoomph.equations.generic.SpatialErrorEstimator` object to the equation system. The latter approach reads:

.. code:: python

   # Load the previous code
   from poisson_2d import *

   # Inherit from the previous problem
   class AdaptivePoissonProblem2d(PoissonProblem2d):
       def define_problem(self):
           # define the previous problem
           super(AdaptivePoissonProblem2d, self).define_problem()

           # add a spatial error estimator for u (errors weighted by the coefficient 1.0)
           additional_equations = SpatialErrorEstimator(u=1.0)
           self.add_equations(additional_equations @ "domain")


   if __name__=="__main__":
       with AdaptivePoissonProblem2d() as problem:
           # Maximum refinement level
           problem.max_refinement_level = 5
           # Refine elements with error larger than that
           problem.max_permitted_error = 0.0005
           # Unrefine elements with elements smaller than that
           problem.min_permitted_error = 0.00005
           # Solve with full refinement
           problem.solve(spatial_adapt=problem.max_refinement_level)
           problem.output()

By using the ``super`` call, the original non-adaptive problem is defined. Then, the :py:class:`~pyoomph.equations.generic.SpatialErrorEstimator` is applied on the field ``u`` with a weighting factor of ``1.0``. If you have multiple fields, you can weight the estimated errors of the individual fields differently. Finally, the :py:meth:`~pyoomph.generic.problem.Problem.solve` call has to be augmented with a ``spatial_adapt`` keyword to specify the number of spatial adaption steps. However, the problem will never refine to a level finer than :py:attr:`~pyoomph.genric.problem.Problem.max_refinement_level`. To check whether any element has to be refined, the estimated error is compared to the :py:attr:`~pyoomph.genric.problem.Problem.max_permitted_error` value. If this threshold is exceeded, the element is refined. When any element got refined, the solution has to be recalculated and the next adaption step starts. If neighboring previously refined elements have an error lower than :py:attr:`~pyoomph.genric.problem.Problem.min_permitted_error`, they also might be recombined to a coarser element within this adaption routine.

The result is depicted in :numref:`figspatialpoissonadapt`. Obviously, the adaption is done near the corners and in the center, where the source function is prominent.

..  figure:: poisson2d_adapt.*
	:name: figspatialpoissonadapt
	:align: center
	:alt: Automatic spatial adaptivity
	:class: with-shadow
	:width: 50%
	
	Automatic spatial adaptivity based on error estimation in the Poisson problem.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <poisson_2d_adaptive.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    

.. tip::

   More details on how the error is estimated can be found e.g. in the oomph-lib documentation, e.g. here: https://oomph-lib.github.io/oomph-lib/doc/the_data_structure/html/classoomph_1_1Z2ErrorEstimator.html
