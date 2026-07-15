Two-dimensional plotting example
--------------------------------

As an example case, let us take the evaporating droplet case from :numref:`secmultidomdropevap`, i.e. we refer to the script :download:`evaporating_water_droplet.py <../multidom/evaporating_water_droplet.py>`. To add plotting, first a specialization of the :py:class:`~pyoomph.output.plotting.MatplotlibPlotter` class must be implemented. The entire plot is defined in the method :py:meth:`~pyoomph.output.plotting.BasePlotter.define_plot`, in which one first has to define the field of view, i.e. the area which should be covered by the plot. Since this area often depends on the problem settings, one can access the problem with the :py:meth:`~pyoomph.output.plotting.BasePlotter.get_problem` method. Here, the considered area depends on the radius of the droplet:

.. literalinclude:: plotting_evaporating_droplet.py
   :language: python
   :start-at: from evaporating_water_droplet import * # Import the problem
   :end-at: self.set_view(-xrange,-0.165*xrange,xrange,0.9*xrange)

Also the :py:attr:`~pyoomph.output.plotting.MatplotlibPlotter.background_color` can be set, where either hex-codes for the color or predefined colors from the python package ``matplotlib`` can be used.

Once the desired plot area is selected by the :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.set_view` method (arguments are minimum :math:`x`, minimum :math:`y`, maximum :math:`x` and maximum :math:`y`, in (potentially dimensional) spatial coordinates), we can start to add parts to the plot. Usually, one starts with color bars with :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_colorbar`, that can be used later on to plot the fields. This can read e.g. as follows:

.. literalinclude:: plotting_evaporating_droplet.py
   :language: python
   :start-at: # Second step: add colorbars with different colormaps at different positions
   :end-at: cb_vap=self.add_colorbar("water vapor [g/m^3]",cmap="Blues",position="top right",factor=1000)

Each color bar gets first a title and can also contain LaTeX code, usually by a ``r``-string, e.g. ``r"$\phi$"`` to obtain :math:`\phi`. ``cmap`` selects the color map, see the documentation of ``matplotlib`` for a reference. With ``position``, we can control the location of the color bar. This can either be a tuple of graph coordinates or a string indicating the position as shown in the example above. By default, all fields are plotted in the normal *SI* units without any prefixes. If a color bar should indicate the range e.g. in :math:`\:\mathrm{mm}/\mathrm{s}`, one must set the ``factor`` to :math:`1000` to compensate for the milli prefix.

Color bars have additional properties, which can be set, e.g. the :py:attr:`~pyoomph.output.plotting.MatplotLibColorbar.length`, :py:attr:`~pyoomph.output.plotting.MatplotLibColorbar.thickness`, :py:attr:`~pyoomph.output.plotting.MatplotLibOverlayBase.xshift` and :py:attr:`~pyoomph.output.plotting.MatplotLibOverlayBase.yshift`, :py:attr:`~pyoomph.output.plotting.MatplotLibOverlayBase.xmargin` and :py:attr:`~pyoomph.output.plotting.MatplotLibOverlayBase.ymargin` (all in graph coordinates). For a complete list of settings, read e.g. the output of ``print(dir(cb_v))`` or have a look at the :py:class:`~pyoomph.output.plotting.MatplotLibColorbar` class in the module :py:mod:`pyoomph.output.plotting`.

Once the color bars are set up, one can plot fields with those. Basically all plots of field data can be done by the :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_plot` method, e.g.

.. literalinclude:: plotting_evaporating_droplet.py
   :language: python
   :start-at: # Now, we can add all kinds of plots
   :end-at: self.add_plot("droplet/velocity", mode="arrows",linecolor="green",transform=["mirror_x",None])

Each :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_plot` call requires you to pass the data to plot as a string, e.g. ``"droplet/velocity"``. When a color bar is supplied by the ``colorbar`` argument, it will be plotted as color map. Vectorial fields, as e.g. the velocity, will be plotted as magnitude. The color bars will automatically increase in range to comprise the visible data range of all plots with the same color bar.

The argument ``transform`` (default ``None``) will apply a transform on the plot, which can e.g. by ``"mirror_x"`` to mirror the data (and the vector fields) along the :math:`x`-axis. You can also supply a list of transforms to plot all transformed data simultaneously. In that case, the return value of :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_plot` is also a ``list`` of the individual plots. Further strings indicating transforms are ``"rotate_cw"``, ``"rotate_ccw"`` and ``"rotate_ccw_mirror"`` for clock-wise, counter-clockwise and counter-clockwise rotation including mirroring, respectively. If a custom transform is required, you can overload the base class :py:class:`~pyoomph.output.plotting.PlotTransform` of :py:mod:`pyoomph.output.plotting` accordingly and pass an instance of your custom transform class as ``transform``.

If no ``colorbar`` is set, you have to specify the plotting ``mode``. To plot e.g. arrows indicating the direction of a vector field, you can use ``mode="arrows"``. Alternatively, you can also use ``mode="streamlines"``. Each ``mode`` has a different class with different settings creating the desired part of the plot. In the :py:mod:`pyoomph.output.plotting` module, you find all available classes for plot modes. These are decorated by ``@MatplotLibPart.register()`` and their class string ``mode`` indicates the plotting mode. You can furthermore see the attributes that you can set from the class definitions.

Again, you can access the problem to select a reasonable field to plot, e.g. the vapor on both sides:

.. literalinclude:: plotting_evaporating_droplet.py
   :language: python
   :start-at: # Plot the vapor in the gas phase
   :end-at: self.add_plot("gas/c_vap",colorbar=cb_vap,transform=["mirror_x",None])

To plot interface lines, just use :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_plot` where the first argument indicates an interface mesh. This will automatically plot the interface lines:

.. literalinclude:: plotting_evaporating_droplet.py
   :language: python
   :start-at: # at the interface lines
   :end-at: self.add_plot("gas/gas_substrate",transform=["mirror_x",None])

You cannot plot an interface if there is no single equation defined on this interface. In that case, just add a dummy equation to this interface when defining the problem. A dummy equation instance can be just e.g. the base class :py:class:`~pyoomph.generic.codegen.Equations` (or :py:class:`~pyoomph.generic.codegen.InterfaceEquations`), which neither defines any fields nor residuals, nor does anything else.

Finally, you can also plot interface fields. These can be either plotted as color maps (``mode="interfacecmap"``, which is selected automatically if you pass a ``colorbar`` to :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_plot`) or as ``"interfacearrows"``. The latter ``mode`` will be selected automatically, if you pass an ``arrowkey`` argument to :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_plot`. However, therefore, you first have to add the arrow key, the same way as done with the color bars above by using :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_arrow_key`:

.. literalinclude:: plotting_evaporating_droplet.py
   :language: python
   :start-at: # For the evaporation rate, we require an arrow key, again with a factor 1000, since we convert from kg to g per m^2s
   :end-at: arrs=self.add_plot("droplet/droplet_gas/evap_rate",arrowkey=ak_evap,transform=["mirror_x",None])

Finally, there are few additional global parts (i.e. parts without any field data) you can add. These are e.g. :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_text`, :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_time_label` or :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_scale_bar`. To add e.g. the current time and a scale bar to the plot, you can call

.. literalinclude:: plotting_evaporating_droplet.py
   :language: python
   :start-at: # and a time label and a scale bar
   :end-at: self.add_scale_bar(position="bottom center").textsize*=0.7

That is all you have to do in the plotter class. To use it, you just have to create an instance of it to the :py:attr:`~pyoomph.generic.problem.Problem.plotter` property of the :py:class:`~pyoomph.generic.problem.Problem` class:

.. literalinclude:: plotting_evaporating_droplet.py
   :language: python
   :start-at: if __name__=="__main__":
   :end-at: # .....

Alternatively, you can also set :py:attr:`~pyoomph.generic.problem.Problem.plotter` to a ``list`` of multiple plotters.

On each output, the plotter(s) will be invoked to create each a plot in the ``_plots`` folder of the output directory. For an example of the resulting plot with this plotter class, refer to :numref:`figmultidomdropevap`.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <plotting_evaporating_droplet.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   	
