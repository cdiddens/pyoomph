Plotting interface
==================

Since meshes are a quite complicated data structure (at least compared to simple grids), pyoomph has a built-in feature to plot meshes and fields. Currently, this only works for two-dimensional meshes, but it is envisioned to also support this for one- and three-dimensional domains in the future.

To activate plotting, one has to set the ``plotter`` property of the :py:class:`~pyoomph.generic.problem.Problem` class to either an instance or a ``list`` of instances of the class :py:class:`~pyoomph.output.plotting.MatplotlibPlotter`, which is defined in the module :py:mod:`pyoomph.output.plotting`. After each :py:meth:`~pyoomph.generic.problem.Problem.output` call, plots will be generated automatically.

.. toctree::
   :maxdepth: 5
   :hidden:

   plotting/droplet.rst
   plotting/replotting.rst
   plotting/eigenfuncs.rst
   plotting/eigendynamics.rst
