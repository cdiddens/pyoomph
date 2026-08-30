Plotting interface
==================

Since meshes are a quite complicated data structure (at least compared to simple grids), pyoomph has a built-in feature to plot meshes and fields. Two-dimensional domains are drawn as spatial maps - colour-keyed fields over the domain, with arrows and streamlines on top - by :py:class:`~pyoomph.output.plotting.MatplotlibPlotter`. One-dimensional domains have no area to fill, so they are drawn as ordinary x-y graphs instead, by :py:class:`~pyoomph.output.plotting1d.MatplotlibPlotter1D`. Three-dimensional domains are not covered yet; export them and use e.g. Paraview.

To activate plotting, one has to set the ``plotter`` property of the :py:class:`~pyoomph.generic.problem.Problem` class to either an instance or a ``list`` of instances of one of these plotter classes, defined in the modules :py:mod:`pyoomph.output.plotting` and :py:mod:`pyoomph.output.plotting1d`. After each :py:meth:`~pyoomph.generic.problem.Problem.output` call, plots will be generated automatically.

.. toctree::
   :maxdepth: 5
   :hidden:

   plotting/droplet.rst
   plotting/tracers.rst
   plotting/replotting.rst
   plotting/eigenfuncs.rst
   plotting/eigendynamics.rst
   plotting/one_dimensional.rst
