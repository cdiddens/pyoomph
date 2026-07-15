.. _secploteigendynamics:

Generating a movie of eigendynamics
-----------------------------------

Eigendynamics, i.e. the exponential growth of a perturbation with a potential oscillation, can be nicely visualized in a movie.
To that end, we can generate a sequence of images with an eigenperturbation that grows (and oscillates) like :math:`exp(\lambda t)`.
As an example of such a movie, refer to :numref:`secadvstabrisingbubble`, where the azimuthal instability of a bubble is investigated. Here, due to the azimuthal instability, it is effectively a three-dimensional scenario which is even harder to visualize in any other way. 

.. only:: html

	.. raw:: html 

		<figure class="align-center" id="vidrisingbubble"><video autoplay="True" preload="auto" width="60%" loop=""><source src="../../_static/rising_bubble.mp4" type="video/mp4"></video><figcaption><p><span class="caption-text">Example of the eigendynamics of the rising bubble case.</span> </span></p></figcaption></figure>

We first start by importing the problem class from :download:`rising_bubble.py` and defining a conventional plotter class. We don't have to do anything regarding the eigendynamics here. This plotter can therefore be also used for the base state, as done in the previous examples.

.. literalinclude:: eigendynamics.py
   :language: python
   :start-at: from rising_bubble import *
   :end-at: self.add_plot("domain/interface",transform=[None,"mirror_x"])

We can then just find a stationary solution, solve (and refine) the eigensolution:

.. literalinclude:: eigendynamics.py
   :language: python
   :start-at: with RisingBubbleProblem() as problem:
   :end-at: problem.create_eigendynamics_animation("eigenanim",RisingBubblePlotter(),max_amplitude=1,numperiods=4,numouts=4*25)


The final call of the method :py:meth:`~pyoomph.generic.problem.Problem.create_eigendynamics_animation` does the following: It creates a subdirectory (here ``eigenanim``) in the output directory. It uses the plotter class shipped as an argument to perform plots. However, it won't plot the base state, but it will perturb the base state by a perturbation with the eigenfunction which grows (and oscillates) over time like :math:`exp(\lambda t)`. We can specify the maximum amplitude, from which the initial amplitude is calculated. Note that the total amplitude is given by the amplitude of the eigenfunction times the growing amplitude. Therefore, it also depends on the way we implemented :py:meth:`~pyoomph.generic.problem.Problem.process_eigenvectors` in :download:`rising_bubble.py`. We can specify the number of periods to consider. In case the eigenvalue is real, i.e. no oscillation amplitude given, this time is given by :math:`2\pi/|\lambda|`. For complex eigenvalues, the reasonable choice for the period is :math:`2\pi/|\mathrm{Im}(\lambda)|`. The amount of output steps then controls the time step. 

Once done, you can assemble all generated images to a movie. It is noteworthy that for azimuthal instabilities, the transform ``"mirror_x"`` will also invert the eigenperturbation on the left part of the plot in a reasonable way.

If you do not want to create a movie, but e.g. a sequence of images for a static document, consider adding ``fileext="pdf"`` to the constructor of the plotter class.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <eigendynamics.py>`

		:download:`Download all examples <../tutorial_example_scripts.zip>`


