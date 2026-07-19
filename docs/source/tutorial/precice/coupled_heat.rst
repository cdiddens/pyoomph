Solving the heat equation on a domain by two simulations
---------------------------------------------------------

To show how the preCICE adapter in pyoomph works, we cover the preCICE tutorial example `Partitioned heat conduction <https://precice.org/tutorials-partitioned-heat-conduction.html>`__.

A rectangular domain of size :math:`2 \times 1` is separated into a left (Dirichlet) and right (Neumann) participant, each of size :math:`1 \times 1`.
On the full domain, we solve a heat conduction equation, i.e. we also solve it in both participants. preCICE will take the lead, i.e. controls the time stepping in both running pyoomph simulations and transfers the data at the mutual coupling interface from one participant to the other.

Via preCICE, the Dirichlet participant (left half) will receive the values of the temperature at the coupling boundary, impose these values as a Dirichlet condition, solve the system and feed back the heat flux to the Neumann participant (right half). The latter will solve the system with the received heat flux as a Neumann condition and feed the temperature Dirichlet values again to the Dirichlet participant.

More details can be found in the `preCICE tutorial <https://precice.org/tutorials-partitioned-heat-conduction.html>`__.


To use preCICE in pyoomph, you can just import the module :py:mod:`pyoomph.solvers.precice_adapter`. As mentioned on the previous page, you must have installed preCICE and the preCICE python bindings to import it.

We want to formulate the problem in a way that it can be either run monolithically in a single simulation, which calculates the full domain. Alternatively, we can run either the Dirichlet or the Neumann participant. If both are running simultaneously, they will interact via preCICE.

pyoomph's :py:class:`~pyoomph.generic.problem.Problem` has the attributes :py:attr:`~pyoomph.generic.problem.Problem.precice_participant` and :py:attr:`~pyoomph.generic.problem.Problem.precice_config_file`. When using preCICE, you must specify the preCICE config file with the latter and the participant name with the former.
Here, the config file :download:`precice-config.xml` of this example defines the participants ``"Dirichlet"`` and ``"Neumann"``.

As usual, we start by importing the required modules and defining the heat equation, i.e. besides importing pyoomph's preCICE adapter, nothing spectacular is happening here:

.. literalinclude:: partitioned_heat_conduction.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: self.add_weak(partial_t(u),v).add_weak(grad(u),grad(v)).add_weak(-self.f,v)

The magic happens in the definition of the problem, where we use several classes from the :py:mod:`~pyoomph.solvers.precice_adapter` module:

.. literalinclude:: partitioned_heat_conduction.py
   :language: python
   :start-at: # Generic heat conduction problem. Can be run without preCICE on the full domain or as Dirichlet or Neumann participant
   :end-at: self+=eqs@"domain"

If we use preCICE, we only define half of the domain. The mesh of the ``"Neumann"`` participant is furthermore shifted to the right. Depending on the side we solve, the ``coupling_boundary`` is either the ``"left"`` or ``"right"`` boundary of the domain. When we set ``precice_participant=""`` (default value), we just solve the full problem and do not add any coupling. 

If we select one of the participants, however, we have to set up the coupling. This happens in multiple steps. First of all, we must export the interface mesh at the coupling boundary to preCICE, which is done by the class :py:class:`~pyoomph.solvers.precice_adapter.PreciceProvideMesh`. You must supply a mesh name agreeing with the ``provide-mesh`` definition in the config file :download:`precice-config.xml`. This will tell preCICE where the nodes are located, so that it can be connected to the other participant. 

Then, both participants have to exchange data. For writing data from the current participant to the other, you can use the class :py:class:`~pyoomph.solvers.precice_adapter.PreciceWriteData`. It takes arguments of the form ``PRECICE_NAME = PYOOMPH_EXPRESSION``, where ``PRECICE_NAME`` must coincide with the name of a ``data`` declaration in the preCICE config file. Since ``Heat-Flux`` cannot be used as a keyword argument (due to the dash), we instead supply it via a ``dict`` using the ``**{}`` syntax in the Dirichlet participant. In pyoomph, we calculate the normal gradient and send it to the ``"Heat-Flux"`` data of preCICE. In the Neumann participant, we just write the nodal values of ``var("u")`` to the ``"Temperature"`` data of preCICE. 

For the opposite direction, we can use :py:class:`~pyoomph.solvers.precice_adapter.PreciceReadData`. It takes arguments like ``PYOOMPH_NAME = PRECICE_NAME`` and defines a pyoomph variable given by ``PYOOMPH_NAME``, which will hold the values of the preCICE data given by ``PRECICE_NAME``. Again, all used ``PRECICE_NAME`` must be declared in the config file to be readable from the mesh.

Thereby, the transfer of data is complete, but you still have to use the data read from the other participant in the current participant. In the Dirichlet case, we use :py:class:`~pyoomph.meshes.bcs.EnforcedBC`, since a :py:class:`~pyoomph.meshes.bcs.DirichletBC` may only depend on the time, Lagrangian and Eulerian (for a static mesh only) coordinates. However, in fact the :py:class:`~pyoomph.meshes.bcs.EnforcedBC` does exactly the same as a :py:class:`~pyoomph.meshes.bcs.DirichletBC` here. The Neumann part just imposes the read ``var("flux")`` as :py:class:`~pyoomph.meshes.bcs.NeumannBC`. Here, one has to be careful with the signs. From the weak form of the heat equation, the Neumann term would require a minus sign, but since the normal is pointing in the negative x-direction on the Neumann side of the interface, it cancels out.

Eventually, we just have to attach all coupling equations to the corresponding boundary and make sure to not apply the Dirichlet boundary conditions of the far field here. If we do not use preCICE, we just discard the coupling equations.

The coupling is complete, but for running a coupled simulation, preCICE must take the lead for the time stepping. Typical time steps and the maximum simulation time are given in the config file. Therefore, pyoomph's :py:meth:`~pyoomph.generic.problems.Problem.run` method cannot be used. Instead, the method :py:meth:`~pyoomph.generic.problems.Problem.precice_run` has to be used:

.. literalinclude:: partitioned_heat_conduction.py
   :language: python
   :start-at: if __name__=="__main__":
   :end-at: problem.precice_run()

This completes the simulation. For running with preCICE, you have to run the script two times, passing the participant name as a command line parameter (see :numref:`secodecmdline`).

.. code:: bash

      python partitioned_heat_conduction.py --outdir Dirichlet -P precice_participant=Dirichlet &
      python partitioned_heat_conduction.py --outdir Neumann -P precice_participant=Neumann
      
Of course, you must place the config file :download:`precice-config.xml` in the same directory.
If you run the scripts without setting :py:attr:`~pyoomph.generic.problem.Problem.precice_participant`, it will just run the monolithic case without preCICE.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <partitioned_heat_conduction.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   	
		    
