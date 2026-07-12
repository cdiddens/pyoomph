Plotting of eigenfunctions
--------------------------

When investigating the stability of stationary solutions by bifurcation analysis, one is frequently interested in the eigenfunction corresponding to the critical eigenvalue whose real value passes through zero. To plot these functions, we can directly use the very same plotting class, but just pass a few additional arguments to the constructor.

As an illustration, let us consider the problem of :numref:`secpdeksebiftrack`, which depends on the scripts :download:`kuramoto_sivanshinsky_bifurcation.py <../pde/patterns/kuramoto_sivanshinsky_bifurcation.py>`, :download:`kuramoto_sivanshinsky_arclength_eigen.py <../pde/patterns/kuramoto_sivanshinsky_arclength_eigen.py>` and :download:`kuramoto_sivanshinsky.py <../pde/patterns/kuramoto_sivanshinsky.py>`. The plotting class can be rather short, since we only have to plot the height field. However, we directly add the functionality for the plots of the eigenfunction:

.. code:: python

   class KSEPlotter(MatplotlibPlotter):
       def define_plot(self):
           cb=self.add_colorbar("h",position="top center") # Add a color bar, but do not show it
           cb.invisible=True

           self.add_plot("domain/h",colorbar=cb) # plot the height field (or its eigenfunction)

           # Parameters as text
           self.add_text(r"$\gamma={:5.4f}, \delta={:5.4f}$".format(self.get_problem().param_gamma.value,self.get_problem().param_delta.value), position="bottom left",textsize=15)

           # Information text, based on whether we plot the eigenfunction or the normal solution
           if self.eigenvector is not None: # eigenfunction is plotted
               self.add_text("eigenvalue={:5.4f}".format(self.get_eigenvalue()),position="bottom right",textsize=15) # eigenvalue (will be 0 anyhow)
               if self.eigenmode=="abs":
                   self.add_text("eigenfunction magnitude",textsize=20,position="top center") # title
               elif self.eigenmode=="real":
                   self.add_text("eigenfunction (real)", textsize=20, position="top center") # title
               elif self.eigenmode == "imag":
                   self.add_text("eigenfunction (imag.)", textsize=20, position="top center") # title
           else:
               self.add_text("height field", textsize=20, position="top center") # title

The plotting is done as before, via a color bar (which is hidden here by setting :py:attr:`~pyoomph.output.plotting.MatplotLibColorbar.invisible`) and :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_plot` of the height field. With the :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_text`, the parameters are added. Depending on whether we want to plot an eigenfunction or the normal solution, we add additional text to the plot. If an eigenfunction is plotted, :py:attr:`~pyoomph.output.plotting.BasePlotter.eigenvector` will refer to the index of the eigenfunction, whereas it will be ``None`` if no eigenfunction (i.e. the normal solution) is about to be plotted. With :py:meth:`~pyoomph.output.plotting.BasePlotter.get_eigenvalue` we can get the eigenvalue corresponding to the plotted eigenfunction. :py:attr:`~pyoomph.output.plotting.BasePlotter.eigenmode` can hold different modes to plot the eigenfunction. Since it is in general complex, we have to choose whether we want to plot the ``"real"`` part, the ``"imag"`` part, the ``"abs"`` magnitude or the ``"angle"`` (which is the phase).

To use this plotting class for the normal solution and different eigenfunction plots, we just have to create an instance of this plotter for each desired plot:

.. code:: python

   if __name__ == "__main__":
       with KSEBifurcationProblem() as problem:
           # Add a bunch of plotters
           problem.plotter=[KSEPlotter(problem)] # Plot the normal solution
           problem.plotter.append(KSEPlotter(problem,eigenvector=0,eigenmode="real",filetrunk="eigenreal_{:05d}")) # real part of the eigenfunction
           problem.plotter.append(KSEPlotter(problem, eigenvector=0, eigenmode="imag", filetrunk="eigenimag_{:05d}")) # imag. part
           problem.plotter.append(KSEPlotter(problem, eigenvector=0, eigenmode="abs", filetrunk="eigenabs_{:05d}")) # magnitude

           for p in problem.plotter:
               p.file_ext=["png","pdf"] # plot both png and pdf for all plotters

           # ...
           # the rest is the same
           # ...

The :py:attr:`~pyoomph.generic.problem.Problem.plotter` property is now a ``list`` containing multiple plotters. Without further arguments, the plotter will plot the normal solution. If ``eigenvector`` is set, it indexes the desired eigenfunction to be plotted and ``eigenmode`` selects the desired mode (see above). Since all plots would write to the same file, we also have to specify ``filetrunk``, which takes a format string, which is formatted based on the output step. All plotters are instructed to plot both *pdf* and *png* files.

Some plots are depicted in :numref:`figplottingeigenkse`. As expected, the critical eigenfunction is only real valued and similar to the solution itself, i.e. the collapse to the flat solution beyond the fold bifurcation is apparent.

..  figure:: eigenkse.*
	:name: figplottingeigenkse
	:align: center
	:alt: Fold bifurcation of the damped Kuramoto-Sivashinsky equation
	:class: with-shadow
	:width: 70%

	Critical solution and corresponding eigenfunction at the fold bifurcation of the damped Kuramoto-Sivashinsky equation :math:numref:`eqpdeksestrong`.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <plotting_eigenmodes.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   	