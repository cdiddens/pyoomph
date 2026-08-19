.. _secplottingonedim:

Plotting one-dimensional domains
--------------------------------

While two-dimensional plots are the real tough ones (unstructured, potentially locally adapted meshes are hard to plot), 
pyoomph also offers to plot one-dimensional domains, although this can be easily done with matplotlib directly.
For one-dimensional domains, we have the :py:class:`~pyoomph.output.plotting1d.MatplotlibPlotter1D` class. 
It's API is similar to the two-dimensional :py:class:`~pyoomph.output.plotting.MatplotlibPlotter`, in fact, it is a subclass of it.
Again, we have to specify :py:meth:`~pyoomph.output.plotting.BasePlotter.define_plot`.

Take a diffusing mode on a line mesh:

.. literalinclude:: one_dimensional.py
   :language: python
   :start-at: class DiffusionPlotter(MatplotlibPlotter1D):
   :end-before: class DiffusionProblem

.. figure:: one_dim_diffusion.*
   :name: figplottingonedimdiffusion
   :align: center
   :alt: A diffusing mode on a line mesh
   :class: with-shadow
   :width: 80%

   A field against its coordinate, along with the element boundaries and an analytical solution.

First, we have to call :py:meth:`~pyoomph.output.plotting1d.MatplotlibPlotter1D.set_axes` to configure the graph: limits,
labels, log scales, a grid, a legend, units, tick formats. 

``rangemode_y="grow"`` will adjust the y-range so that it only grows, never shrinks, which can be relevant e.g. for movies.
By default the y-axis is rescaled to each output step, so a decaying solution fills the frame
at every step and two frames cannot be compared by eye. ``"grow"`` takes the union over all steps
instead. The other choices are ``"auto"`` (the default), ``"fixed"`` (use the limits given to
:py:meth:`~pyoomph.output.plotting1d.MatplotlibPlotter1D.set_axes`), ``"firststep"`` (lock onto the
first output step) and an explicit ``(lo,hi)`` pair. The accumulated ranges are written next to the
colorbar ranges under ``_plots/_cb_ranges``, so a replot run (``--runmode p``) can pick them up again.

:py:meth:`~pyoomph.output.plotting1d.MatplotlibPlotter1D.add_plot` takes the usual domain and field path, as in
two dimensions. The abscissa is ``coordinate_x`` unless ``xaxis=`` says otherwise. 
The axis labels are derived from the field names and their units, only an explicit ``xlabel``/``ylabel`` overrides that.

:py:meth:`~pyoomph.output.plotting1d.MatplotlibPlotter1D.add_nodes` marks the mesh nodes, with
``only_vertex_nodes=True`` skipping the interior node of each second-order element, and
:py:meth:`~pyoomph.output.plotting1d.MatplotlibPlotter1D.add_element_borders` draws a thin line at
every element boundary. Both is helpful to visualize spatial adaptivity. 
External data that does not come from a mesh, e.g. an analytical solution or any other externally loaded data, can be added with :py:meth:`~pyoomph.output.plotting1d.MatplotlibPlotter1D.add_analytical` or 
:py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_external_data`.
:py:meth:`~pyoomph.output.plotting1d.MatplotlibPlotter1D.add_analytical` evaluates a function on a uniform grid over the current x-range.


A one-dimensional mesh may also be embedded in a higher dimensional space. Such a domain has a shape of its own, so it might become hard to put it into a simple plot.
To that end, :py:meth:`~pyoomph.output.plotting1d.MatplotlibPlotter1D.add_curve` draws the shape, which can optionally be color-coded by a field defined on the mesh:

.. literalinclude:: one_dimensional.py
   :language: python
   :start-at: class CurvePlotter(MatplotlibPlotter1D):
   :end-before: class ArclengthPlotter

.. figure:: one_dim_curve.*
   :name: figplottingonedimcurve
   :align: center
   :alt: The curve a bent line mesh traces in the plane
   :class: with-shadow
   :width: 80%

   The bend mesh shape, colored by a field solved on it.

The points are automatically ordered by the mesh connectivity. A domain made of several disconnected pieces is drawn as several curves.

Alternatively, ``xaxis="arclength"`` is available: the distance travelled along the curve.

.. literalinclude:: one_dimensional.py
   :language: python
   :start-at: class ArclengthPlotter(MatplotlibPlotter1D):
   :end-before: class CurvedProblem

.. figure:: one_dim_arclength.*
   :name: figplottingonedimarclength
   :align: center
   :alt: The same field against x and against arclength
   :class: with-shadow
   :width: 80%

   The same field, drawn against :math:`x` and against the arclength :math:`s` along the curve.

Besides ``"arclength"``, ``xaxis`` accepts any coordinate (``"coordinate_y"``, ``"coordinate_z"`` and
the ``"lagrangian_*"`` equivalents), the name of any nodal field or local expression - so a field can
be plotted against another field.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <one_dimensional.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`
