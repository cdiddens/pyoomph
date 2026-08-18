.. _secplottingonedim:

Plotting one-dimensional domains
--------------------------------

A two-dimensional domain is drawn as a spatial map: a colour-keyed field over an area, with arrows
and streamlines on top of it. A one-dimensional domain has no area to fill. What one wants there is a
graph - a curve, with visible axes, labels and a legend - and that is what
:py:class:`~pyoomph.output.plotting1d.MatplotlibPlotter1D` draws. It is used exactly like
:py:class:`~pyoomph.output.plotting.MatplotlibPlotter`: subclass it, implement
:py:meth:`~pyoomph.output.plotting.BasePlotter.define_plot`, and assign an instance to the problem's
``plotter``. Everything that is not about the geometry is inherited unchanged - file names and
formats, eigenvector plotting, the merge of a mesh distributed over MPI ranks, and all the overlays.

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

   A field against its coordinate, with the element boundaries and the exact solution drawn in.

:py:meth:`~pyoomph.output.plotting1d.MatplotlibPlotter1D.set_axes` configures the graph: limits,
labels, log scales, a grid, a legend, units, tick formats. Anything it does not cover can be set on
the object it returns.

``rangemode_y="grow"`` deserves a word, because it is what makes a series of plots into a usable
movie. By default the y-axis is rescaled to each output step, so a decaying solution fills the frame
at every step and two frames cannot be compared by eye. ``"grow"`` takes the union over all steps
instead. The other choices are ``"auto"`` (the default), ``"fixed"`` (use the limits given to
:py:meth:`~pyoomph.output.plotting1d.MatplotlibPlotter1D.set_axes`), ``"firststep"`` (lock onto the
first output step) and an explicit ``(lo,hi)`` pair. The accumulated ranges are written next to the
colorbar ranges under ``_plots/_cb_ranges``, so a replot run can pick them up again.

:py:meth:`~pyoomph.output.plotting1d.MatplotlibPlotter1D.add_plot` takes a domain and a field, as in
two dimensions. The abscissa is ``coordinate_x`` unless ``xaxis=`` says otherwise, and it stays
``coordinate_x`` whether the domain lives in one, two or three dimensions of space - it is never
guessed from the mesh. The axis labels are derived from the field names and their units, and an
explicit ``xlabel``/``ylabel`` overrides that.

Two decorations are worth knowing about:
:py:meth:`~pyoomph.output.plotting1d.MatplotlibPlotter1D.add_nodes` marks the mesh nodes, with
``only_vertex_nodes=True`` skipping the interior node of each second-order element, and
:py:meth:`~pyoomph.output.plotting1d.MatplotlibPlotter1D.add_element_borders` draws a thin line at
every element boundary. Both make spatial adaptivity visible on a curve that would otherwise look
equally smooth everywhere. Data that does not come from a mesh - an analytical solution, a
measurement - goes in with
:py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_external_data` or, as above, with
:py:meth:`~pyoomph.output.plotting1d.MatplotlibPlotter1D.add_analytical`, which evaluates a function
on a uniform grid over the current x-range.

Curves in space
~~~~~~~~~~~~~~~

A one-dimensional domain need not live in one-dimensional space. An interface of a two-dimensional
problem is one, and so is a line mesh created with ``nodal_dimension=2`` and then bent. Such a domain
has a shape of its own, and
:py:meth:`~pyoomph.output.plotting1d.MatplotlibPlotter1D.add_curve` draws it - one coordinate against
another, optionally colour-coded by a field:

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

   The curve the mesh itself traces, coloured by the field solved on it.

The points are ordered by the mesh connectivity, not by ascending :math:`x`. That is the whole reason
this works: a curve that folds back on itself, a closed interface, or a helix has no coordinate that
increases monotonically along it, and sorting by one would replace the curve with a zig-zag. It also
means that a domain made of several disconnected pieces is drawn as several curves rather than being
joined across the gaps, and that the interior nodes of second-order elements land in their proper
place, so the line is smooth instead of saw-toothed.

For the same reason, ``xaxis="arclength"`` is available: the distance travelled along the curve,
which is often the natural coordinate on a bent domain even though ``coordinate_x`` remains the
default.

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
be plotted against another field - ``"index"`` for the position along the curve, which is useful when
a connectivity problem is suspected, and a plain numpy array with one entry per node.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <one_dimensional.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`
