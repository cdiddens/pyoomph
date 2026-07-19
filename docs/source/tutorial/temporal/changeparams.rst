Modifying simulation parameters
-------------------------------

Up to now, we always have used the predefined parameter values which were set in the constructor of the problem class. As an example, we have set values for ``mass``, ``spring_constant`` and ``initial_displacement`` in the problem class of the harmonic oscillator in the previous section. These were then passed to the equation class and used as initial condition, respectively. Each change of the problem parameters would naively require editing these values in the constructor of this class. The idea of a complete problem class is, however, that only reasonable default parameters are set and all parameters can be modified easily. In fact for complicated problems, it is a good practice to even put the problem class in another Python file than the script actually used to start the simulation.

Inside Python
~~~~~~~~~~~~~

The direct way of modifying parameters before running the simulation is directly in Python. As described, we can just import the previously discussed Python file where the problem is defined.

.. code:: python

   #Import the script where the problem is defined (here, the file dimensional_oscillator_with_units.py is in the same directory)
   from dimensional_oscillator_with_units import *

   if __name__=="__main__":
   	with DimensionalOscillatorProblem() as problem:
   	
   		#Modify the parameters
   		problem.initial_displacement=-10*centi*meter
   		problem.mass=1*kilogram
   		problem.spring_constant=1*newton/meter
   		
   		problem.run(endtime=10*second,numouts=1000)

Since the default output directory will have the same name as the script name without the ``.py`` extension, i.e. ``dim_osci_run`` here, the output directories are normally different by default. You can create multiple run scripts of this type with different parameters and each will write to its own output folder. To change the folder, you still can use :py:meth:`~pyoomph.generic.problem.Problem.set_output_directory`.

Note that after the first call of :py:meth:`~pyoomph.generic.problem.Problem.run` (or also several other methods like :py:meth:`~pyoomph.generic.problem.Problem.output` and so on), one cannot change the output directory or any parameters that easily. The reason is that all these settings are required for the code generation, e.g. for the nondimensionalization and so on. If it is really necessary, one can invoke a recompilation by :py:meth:`~pyoomph.generic.problem.Problem.redefine_problem`, but this is not discussed here. Therefore, it is important to set all parameters before the first :py:meth:`~pyoomph.generic.problem.Problem.run` statement.

Sometimes, it is not beneficial to have an individual run script for each simulation intended to be performed. If you want to run e.g. hundreds or thousands of simulations to perform a parameter scan, it would require writing a lot of script files. Of course, you could loop through it in python, e.g.

.. code:: python

   from dimensional_oscillator_with_units import *

   if __name__=="__main__":
   	for k_in_N_per_m in [0.1,0.2,0.5,1,2,5]: #Scan the spring constant
   		with DimensionalOscillatorProblem() as problem:
   		
   			#Modify the parameters
   			problem.initial_displacement=-10*centi*meter
   			problem.mass=1*kilogram
   			problem.spring_constant=k_in_N_per_m*newton/meter
   			
   			problem.set_output_directory("dim_osci_seq_run_k_"+str(k_in_N_per_m))
   			
   			problem.run(endtime=10*second,numouts=1000)

However, this would run all simulations sequentially. For this simple problem, it is not an issue, but if each simulation would take several hours or days, it is not an option.

.. _secodecmdline:

Via the command line
~~~~~~~~~~~~~~~~~~~~

You can always override simulation parameters from the command line. For this, you can add the command line arguments *--outdir* to change the output directory and *-P* to modify parameters, e.g.

.. code:: bash

      python dimensional_oscillator_with_units.py --outdir dim_osci_run_modified_params -P spring_constant='1.5*newton/meter' initial_displacement='0.25*meter'

invokes the script :download:`dimensional_oscillator_with_units.py`, output will be written to the directory ``dim_osci_run_modified_params`` and we set the spring constant :math:`k=1.5 \:\mathrm{N}/\mathrm{m}` and the initial displacement to :math:`x_0=0.25\:\mathrm{m}`. Note that the parameters passed via the command line will be set after the parameters set in :download:`dimensional_oscillator_with_units.py` before the :py:meth:`~pyoomph.generic.problem.Problem.run` statement. Hence, the command line will override the parameters set in the script.

This allows one to use e.g. bash in Linux or a batch script in Windows to call multiple simulations in a loop.

Parameter scans via Python
~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, you can also scan through parameters in parallel in Python. Again, we want to run the script :download:`dimensional_oscillator_with_units.py` with multiple parameter settings. pyoomph comes with a class that can take care of looping over parameters and invoking a call of the simulation script with each particular parameter combination as follows:

.. literalinclude:: parallel_running.py
   :language: python
   :start-at: #Import the parallel parameter scanner
   :end-at: scanner.run_all(skip_done=False)

First, a :py:class:`~pyoomph.utils.paramscan.ParallelParameterScan` object is created, passing the script that should be started in parallel and an optional argument of how many processes to be used. If the latter is omitted, it will default to the number of CPUs on the system. The script to run must be either in the same directory or the corresponding full or relative directory has to be passed. Then, for each parameter combination, we add a new simulation to the list using :py:meth:`~pyoomph.utils.paramscan.ParallelParameterScan.new_sim` with the output directory of this particular script. Note that these output directories will be sub-directories of the directory of the parameter scan, which defaults to ``dim_osci_para_run`` here, but can be set via the keyword argument ``output_dir`` in the constructor of the :py:class:`~pyoomph.utils.paramscan.ParallelParameterScan` object. We can then modify any parameter we like to adjust for the simulation by setting them to the object obtained by :py:meth:`~pyoomph.utils.paramscan.ParallelParameterScan.new_sim`.

Finally, if all simulations to be started are added, we can invoke the :py:meth:`~pyoomph.utils.paramscan.ParallelParameterScan.run_all` method to start them in parallel. The moment one simulation finishes, the next one is started, but the maximum number of parallel processes is never exceeded. The optional argument ``skip_done`` lets you control whether you want to rerun already completed simulations or not. If you make e.g. relevant changes in the script :download:`dimensional_oscillator_with_units.py`, it obviously has to be set to ``False``, since the finished results may differ. Otherwise, you can easily continue a previously interrupted scanning process by setting ``skip_done=True``, which will skip all simulations that were already completed previously.

.. only:: html
	
	.. container:: downloadbutton

		:download:`Download this example <parallel_running.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   		

