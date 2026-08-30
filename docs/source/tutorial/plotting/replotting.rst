.. _secreplotting:

Replotting of existing data
---------------------------

The default file extension of plots is *png* files. You can change this by the :py:attr:`~pyoomph.output.plotting.MatplotlibPlotter.file_ext` to e.g. *pdf* files. *Png* files are more suitable to assemble a movie out of the plots, whereas *pdf* files are better for inclusion in publications.

To change the file extension, you can just set it after :py:attr:`~pyoomph.generic.problem.Problem.plotter` has been assigned, e.g.

.. code:: python

           # Changing the file extension (also a list works, e.g. ["pdf","png"]
           problem.plotter.file_ext = "pdf"
           

It can also be a ``list`` of file extensions to be created simultaneously.

Furthermore, you can change e.g. the :py:attr:`~pyoomph.output.plotting.MatplotlibPlotter.dpi` of the plots or change the default settings of individual plotting parts here:

.. code:: python

           # Changing e.g. the dpi or default settings of the velocity arrows:
           problem.plotter.dpi *= 1.5
           # problem.plotter.defaults("arrows").arrowdensity /= 2
           # problem.plotter.defaults("arrows").arrowlength *= 1.5        

However, after having simulated a long simulation, you do not want to run it again to create the new plots. Instead, you can instruct pyoomph to just redo all plots, if you supply the command line argument ``--runmode p`` to the call (cf. also :numref:`secodecmdline`):

.. code:: bash

   python evap_droplet_thermal_plot.py --runmode p

If you only want to recreate e.g. the plot number :math:`10`, you can add the ``--where`` argument as well

.. code:: bash

   python evap_droplet_thermal_plot.py --runmode p --where step==10

Alternatively, you can also create something like ``--where 'step in [10,11,20]'`` or alternative python ``bool`` expressions involving the output step ``step``.

Note that this replotting only works, if :py:attr:`~pyoomph.generic.problem.Problem.write_states` is ``True`` (which is the default). Replotting, just like continuing a simulation (cf. :numref:`secpdecontinue`), relies on the state files which contain all information of the current state of the simulation.
