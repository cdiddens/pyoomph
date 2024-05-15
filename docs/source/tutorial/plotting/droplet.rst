Two-dimensional plotting example
--------------------------------

As an example case, let us take the evaporating droplet case from :numref:`secmultidomdropevap`, i.e. we refer to the script :download:`evaporating_water_droplet.py <../multidom/evaporating_water_droplet.py>`. To add plotting, first a specialization of the :py:class:`~pyoomph.output.plotting.MatplotlibPlotter` class must be implemented. The entire plot is defined in the method :py:meth:`~pyoomph.output.plotting.BasePlotter.define_plot`, in which the one first have to define the field of view, i.e. the area which should be covered by the plot. Since this area often depends on the problem settings, one can access the problem with the :py:meth:`~pyoomph.output.plotting.BasePlotter.get_problem` method. Here, the considered area depends on the radius of the droplet:

.. code:: python

    from evaporating_water_droplet import * # Import the problem
    from pyoomph.output.plotting import *


    # Specialize the MatplotlibPlotter for the droplet problem
    class DropPlotter(MatplotlibPlotter):
        # This method defines the entire plot
        def define_plot(self):    
            p=self.get_problem() # we can get access to the problem by get_problem()
            r=p.droplet_radius # and hence can access the droplet base radius
            xrange=1.5*r # in x-direction, the image will be from [-1.5*r : 1.5*r ]
            self.background_color="darkgrey" # background color

            # First step: Set the field of view (xmin,ymin,xmax,ymax)
            self.set_view(-xrange,-0.165*xrange,xrange,0.9*xrange)


Also the :py:attr:`~pyoomph.output.plotting.MatplotlibPlotter.background_color` can be set, where either hex-codes for the color or predefined colors from the python package ``matplotlib`` can be used.

Once the desired plot area is selected by the :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.set_view` method (arguments are minimum :math:`x`, minimum :math:`y`, maximum :math:`x` and maximum :math:`y`, in (potentially dimensional) spatial coordinates), we can start to add parts to the plot. Usually, one starts with color bars with :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_colorbar`, that can are used later on to plot the fields. This can read e.g. as follows:

.. code:: python

            # Second step: add colorbars with different colormaps at different positions
            # with 'factor', you can control the multiplicative factor (i.e. mm/s requires a factor of 1000 to convert the m/s to mm/s)
            cb_v=self.add_colorbar("velocity [mm/s]",cmap="seismic",position="bottom right",factor=1000)
            cb_vap=self.add_colorbar("water vapor [g/m^3]",cmap="Blues",position="top right",factor=1000)


Each color bar gets first a title and can also contain LaTeXcode, usually by a ``r``-string, e.g. ``r"$\phi$"`` to obtain :math:`\phi`. ``cmap`` selects the color map, see the documentation of ``matplotlib`` for a reference. With ``position``, we can control the location of the color bar. This can either be a tuple of graph coordinates or a string indicating the position as shown in the example above. By default, all fields are plotted in the normal *SI* units without any prefixes. If a color bar should indicate the range e.g. in :math:`\:\mathrm{mm}/\mathrm{s}`, one must set the ``factor`` to :math:`1000` to compensate for the milli prefix.

Color bars have additional properties, which can be set, e.g. the :py:attr:`~pyoomph.output.plotting.MatplotLibColorbar.length`, :py:attr:`~pyoomph.output.plotting.MatplotLibColorbar.thickness`, :py:attr:`~pyoomph.output.plotting.MatplotLibOverlayBase.xshift` and :py:attr:`~pyoomph.output.plotting.MatplotLibOverlayBase.yshift`, :py:attr:`~pyoomph.output.plotting.MatplotLibOverlayBase.xmargin` and :py:attr:`~pyoomph.output.plotting.MatplotLibOverlayBase.ymargin` (all in graph coordinates). For a complete list of settings, read e.g. the output of ``print(dir(cb_v))`` or have a look at the :py:class:`~pyoomph.output.plotting.MatplotLibColorbar` class in the module :py:mod:`pyoomph.output.plotting`.

Once the color bars are set up, one can plot fields with those. Basically all plots of field data can be done by the :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_plot` method, e.g.

.. code:: python

            # Now, we can add all kinds of plots
            # plot the velocity (magnitude, since it is a vector) of the droplet domain (on both sides)
            self.add_plot("droplet/velocity",colorbar=cb_v,transform=["mirror_x",None])
            # add velocity arrows on both sides
            self.add_plot("droplet/velocity", mode="arrows",linecolor="green",transform=["mirror_x",None])
            
                               

Each :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_plot` call requires to pass the data to plot as string, e.g. ``"droplet/velocity"``. When a color bar is supplied by the ``colorbar`` argument, it will be plotted as color map. Vectorial fields, as e.g. the velocity, will be plotted as magnitude. The color bars will automatically increase in range to comprise the visible data range of all plots with the same color bar.

The argument ``transform`` (default ``None``) will apply a transform on the plot, which can e.g. by ``"mirror_x"`` to mirror the data (and the vector fields) along the :math:`x`-axis. You can also supply a list of transforms to plot all transformed data simultaneously. In that case, the return value of :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_plot` is also a ``list`` of the individual plots. Further strings indicating transforms are ``"rotate_cw"``, ``"rotate_ccw"`` and ``"rotate_ccw_mirror"`` for clock-wise, counter-clockwise and counter-clockwise rotation including mirroring, respectively. If a custom transform is required, you can overload the base class :py:class:`~pyoomph.output.plotting.PlotTransform` of :py:mod:`pyoomph.output.plotting` accordingly and pass an instance of your custom transform class as ``transform``.

If no ``colorbar`` is set, you have to specify the plotting ``mode``. To plot e.g. arrows indicating the direction of a vector field, you can use ``mode="arrows"``. Alternatively, you can also use ``mode="streamlines"``. Each ``mode`` has a different class with different settings creating the desired part of the plot. In the :py:mod:`pyoomph.output.plotting` module, you find all available classes for plot modes. These are decorated by ``@MatplotLibPart.register()`` and their class string ``mode`` indicates the plotting mode. You can furthermore see the attributes that you can set from the class definitions.

Again, you can access the problem to select a reasonable field to plot, e.g. the vapor on both sides:

.. code:: python

            # Plot the vapor in the gas phase
            self.add_plot("gas/c_vap",colorbar=cb_vap,transform=["mirror_x",None])

To plot interface lines, just use :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_plot` where the first argument indicates an interface mesh. This will automatically plot the interface lines:

.. code:: python

            # at the interface lines
            self.add_plot("droplet/droplet_gas",linecolor="yellow",transform=["mirror_x",None])
            self.add_plot("droplet/droplet_substrate",transform=["mirror_x",None])
            self.add_plot("gas/gas_substrate",transform=["mirror_x",None])


You cannot plot an interface if there is no single equation defined on this interface. In that case, just add a dummy equation to this interface when defining the problem. A dummy equation instances can be just e.g. the base class :py:class:`~pyoomph.generic.codegen.Equations` (or :py:class:`~pyoomph.generic.codegen.InterfaceEquations`), which neither define any fields nor residuals nor doing anything else.

Finally, you can also plot interface fields. These can be either plotted as color maps (``mode="interfacecmap"``, which is selected automatically if you pass a ``colorbar`` to :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_plot`) or as ``"interfacearrows"``. The latter ``mode`` will be selected automatically, if you pass a ``arrowkey`` argument to :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_plot`. However, therefore, you first have to add the arrow key, similar as done with the color bars above by using :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_arrow_key`:

.. code:: python

            # For the evaporation rate, we require an arrow key, again with a factor 1000, since we convert from kg to g per m^2s
            ak_evap=self.add_arrow_key(position="top center",title="evap. rate water [g/(m^2s)]",factor=1000)
            # We can hide instances by setting invisible=True:
            #       ak_evap.invisible=True
            # or we can move it a bit relative to the "position" by the xmargin and ymargin
            # or with xshift and yshift. All are in graph coordinates, i.e. 1 means the width/height of the entire image
            ak_evap.ymargin+=0.2
            ak_evap.xmargin *= 2

            # add the evaporation arrows at the interface, both sides
            arrs=self.add_plot("droplet/droplet_gas/evap_rate",arrowkey=ak_evap,transform=["mirror_x",None])

            

Finally, there are few additional global parts (i.e. parts without any field data) you can add. These are e.g. :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_text`, :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_time_label` or :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_scale_bar`. To add e.g. the current time and a scale bar to the plot, you can call

.. code:: python

           # and a time label and a scale bar
            self.add_time_label(position="top left")
            self.add_scale_bar(position="bottom center").textsize*=0.7

That is all you have to do in the plotter class. To use it, you just have to create an instance of it to the :py:attr:`~pyoomph.generic.problem.Problem.plotter` property of the :py:class:`~pyoomph.generic.problem.Problem` class:

.. code:: python

   if __name__=="__main__":
       with EvaporatingDroplet() as problem:
           problem.plotter=DropPlotter(problem) # set the plotter and pass the problem itself
           # The rest is the same as before
           # .....
           #

Alternatively, you can also set :py:attr:`~pyoomph.generic.problem.Problem.plotter` to a ``list`` of multiple plotters.

On each output, the plotter(s) will be invoked to create each a plot in the ``_plots`` folder of the output directory. For an example of the resulting plot with this plotter class, refer to :numref:`figmultidomdropevap`.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <plotting_evaporating_droplet.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   	
