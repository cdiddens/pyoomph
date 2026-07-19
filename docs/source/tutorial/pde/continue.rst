.. _secpdecontinue:

Continuing a stopped simulation
-------------------------------

Spatio-temporal simulations can be time consuming and you might have to reboot for whatever reasons, causing your simulation to stop. In that case, you can continue the simulation at the last written output by adding the command line argument ``--runmode c`` to the call:

.. code:: bash
   
   python MY_SIMULATION.py --runmode c

where ``MY_SIMULATION.py`` is of course your simulation script. You can only continue if the property :py:attr:`~pyoomph.generic.problem.Problem.write_states` is ``True``, which is the default. The state files (found in the sub-directory ``_states`` of your output directory) contain all information of the current state of the simulation which allows it to continue.

.. warning::

   Continuing a stopped simulation does not really work if you do stationary solves, arc length continuations, etc. It only works correctly if you just use the :py:meth:`~pyoomph.generic.problem.Problem.run` call to perform a temporal integration of the system. In future, it might change to be more flexible.
