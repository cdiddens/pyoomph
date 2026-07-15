.. _secODEarclength:

Following a stationary solution along a parameter by pseudo-arc length continuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the light of knowing now how to consider parameters, solve for stationary solutions and calculate the stability, it is time to move on to another feature, which is helpful. The question is simple: *How does the stationary solution change if we gradually increase or decrease a parameter?* To answer this question, *pseudo-arc length continuation* is a powerful tool.

The simplest problem where arc length continuation becomes handy is the normal form of another kind of bifurcation, namely the *fold bifurcation*, which reads

.. math:: :label: eqodefoldnf

   \partial_t x=r-x^2\,.

Obviously, there is no stationary solution for :math:`r<0`, whereas we have a stable stationary solution at :math:`x_0=\sqrt{r}` and an unstable stationary solution at :math:`x_0=-\sqrt{r}` for :math:`r>0`. Let us implement this equation and gradually reduce from a positive :math:`r` and follow the stationary solution:

.. literalinclude:: bifurcation_fold_param_change.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: problem.r.value-=0.02

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <bifurcation_fold_param_change.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		
               
Of course, this code will crash the moment we decrease :math:`r<0`, since there is no stationary solution and hence :py:meth:`~pyoomph.generic.problem.Problem.solve` will fail. Since we know that the fold bifurcation takes place at :math:`r=0`, one could try to increase the parameter again after reaching :math:`r=0`, but then - which branch of the solution will be taken? The stable one at :math:`x=\sqrt{r}` or the unstable one at :math:`x=-\sqrt{r}`.

Obviously, in this situation, the parameter :math:`r` is not the best quantity to vary in order to obtain the entire solution curve as a function of the parameter. One could prescribe a value of :math:`x_0` and determine :math:`r` so that this value of :math:`x_0(r)` is indeed a stationary solution. However, there is an even better way: The fundamental idea is to neither solve for solutions :math:`x_0` for varying :math:`r`, nor solve for the parameter :math:`r` for which a varying :math:`x_0` is the stationary solution, but instead vary both at the same time. However, what to choose as the independent variable in that case? A good choice is obviously the arc length :math:`s` along the curve :math:`(x_0(r),r)`. This means that :math:`r` becomes part of the unknowns and we are solving the system

.. math:: :label: eqodearclengthcontr

   \begin{aligned}
   F(x,r)=r-x^2&=0\\
   (x-x^*) \partial_x F +(r-r^*)\partial_r F &=\Delta s
   \end{aligned}

where :math:`(x^*,r^*)` is a starting point for which :math:`F(x^*,r^*)=0` holds. The second equation now prescribes a step :math:`\Delta s` in the tangent direction, i.e. along the tangent :math:`(\partial_x F,\partial_r F)` along the curve :math:`F(x,r)=0`. In pyoomph, this can be done with the method :py:meth:`~pyoomph.generic.problem.Problem.arclength_continuation`:

.. literalinclude:: bifurcation_fold_arclength.py
   :language: python
   :start-at: from bifurcation_fold_param_change import *

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <bifurcation_fold_arclength.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		

First, reuse the classes from the beginning of this page, i.e. from :download:`bifurcation_fold_param_change.py`. We get a start solution :math:`(x^*,r^*)`. Then, we use :py:meth:`~pyoomph.generic.problem.Problem.arclength_continuation` to solve the system above for a step :math:`\Delta s`. In the first step the sign of :math:`\Delta s` (``ds`` in python here) gives the initial direction for the parameter, i.e. ``ds<0`` means we want to initially decrease the parameter :math:`r`. :py:meth:`~pyoomph.generic.problem.Problem.arclength_continuation` will now adjust :math:`r` and solve for the stationary solution at the same time. Furthermore, it returns a new guess for ``ds`` for the next step. In order to capture the curve well, we can add the optional argument ``max_ds`` to limit the maximum step along the tangential direction. After the first call of :py:meth:`~pyoomph.generic.problem.Problem.arclength_continuation`, the direction of the tangent and further parameters required for the next steps are implicitly stored in the :py:class:`~pyoomph.generic.problem.Problem` class. These are reset whenever one performs an :py:meth:`~pyoomph.generic.problem.Problem.arclength_continuation` with respect to another parameter or explicitly calls :py:meth:`~pyoomph.generic.problem.Problem.reset_arc_length_parameters`.

We can combine the :py:meth:`~pyoomph.generic.problem.Problem.arclength_continuation` with the calculation of eigenvalues to get a bifurcation diagram of the fold bifurcation:

.. literalinclude:: bifurcation_fold_arclength_eigen.py
   :language: python
   :start-at: from bifurcation_fold_param_change import *
   :end-at: write_to_eigen_file() # write the updated state

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <bifurcation_fold_arclength_eigen.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		


Here, we use the classes from :download:`bifurcation_fold_param_change.py` to continue along the branch and solve for the eigenvalues. We write the eigenvalue with the largest real part (by default stored at index 0) to a file. A plot of this can be found in :numref:`figodebifurcs`, where we marked the stability of the branches.


..  figure:: bifurcs.*
	:name: figodebifurcs
	:align: center
	:alt: Diagram of multiple bifurcations
	:class: with-shadow
	:width: 100%
	
	Bifurcation diagrams of the fold, transcritical and pitchfork (super- and subcritical) bifurcations determined by arc length continuation and eigenvalues.


With the same approach, we can find the bifurcation diagram of the transcritical normal form :math:numref:`eqodetranscriticnf` and the pitchfork normal form

.. math:: :label: eqodepitchforknf

   \partial_t x=rx\pm x^3\,,   

where a minus sign gives the supercritical and a plus sign the subcritical version of the pitchfork bifurcation. All bifurcations are plotted in :numref:`figodebifurcs`. The corresponding codes are not discussed here, since they all are similar to the code of the fold bifurcation above. However, they are shipped along with this tutorial and can be found in the files :download:`bifurcation_transcritital_arclength_eigen.py` and :download:`bifurcation_pitchfork_arclength_eigen.py`. For the pitchfork bifurcation, we had to set ``set_arc_length_parameter(scale_arc_length=False)`` to stay on the non-trivial branch. If ``scale_arc_length`` is ``True``, the taken arc length step :math:`\Delta s` will be scaled with the magnitudes of :math:`\partial_x F` and :math:`\partial_r F` in :math:numref:`eqodearclengthcontr`. However, at the pitchfork bifurcation, both will approach zero when approaching the bifurcation at :math:`r=0`.

The approach presented here can not only be applied to the simple normal forms but to arbitrary systems, also on discretized spatio-temporal partial differential equations, which will be done later in :numref:`secpdekse`. This provides an easy way to investigate the stability of complicated highly nonlinear systems.
