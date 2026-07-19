.. _secprecicenonmatch:

Non-matching meshes and passing vectorial quantities
----------------------------------------------------

The preCICE tutorial also covers a `generalization of the previous case <https://precice.org/tutorials-partitioned-heat-conduction-complex.html>`__.

It solves exactly the same equation as in the previous case, but the domains are different. One participant is now a rectangle with a circular hole and the other is the missing circular domain that fits in the hole. Most interestingly, the nodes do not have to match at the coupling boundary. While such non-matching coupling is possible in oomph-lib, it cannot be done in pyoomph directly yet. preCICE, however, does not care about the resolution and node positioning at the coupling boundaries, provided that it can find a corresponding node in the coupled participant.

The changes compared to the previous case are minor. Mainly, we need another mesh:

.. literalinclude:: partitioned_heat_conduction_circle.py
   :language: python
   :start-at: class RectMeshWithCircleHole(GmshTemplate):
   :end-at: self.plane_surface(*circle_lines,name="domain")

We create the full mesh for the monolithic run; otherwise, we either create a fine circle for the Neumann problem or a box with the corresponding circular hole in the Dirichlet case. The meshes do not match, since the resolution of the Neumann mesh is smaller. For the coupling interface, we define the boundary ``"interface"``, which is not required for the monolithic case.

Furthermore, as in the `preCICE example <https://precice.org/tutorials-partitioned-heat-conduction-complex.html>`__, we pass the heat flux as a vector now. This means we have to change the :py:class:`~pyoomph.solvers.precice_adapter.PreciceWriteData` in the Dirichlet participant to  

.. literalinclude:: partitioned_heat_conduction_circle.py
   :language: python
   :start-at: coupling_eqs+=PreciceWriteData(**{"Heat-Flux":grad(var("u",domain=".."))},vector_dim=2)
   :end-at: coupling_eqs+=PreciceWriteData(**{"Heat-Flux":grad(var("u",domain=".."))},vector_dim=2)

Vectorial quantities must be marked with ``vector_dim``; otherwise, it is a scalar. However, it also must agree with the definition in the config file :download:`precice-config-circle.xml`.

Likewise, reading and imposing the vectorial flux on the Neumann participant is now different:

.. literalinclude:: partitioned_heat_conduction_circle.py
   :language: python
   :start-at: coupling_eqs+=PreciceReadData(flux="Heat-Flux",vector_dim=2)
   :end-at: coupling_eqs+=NeumannBC(u=-dot(var("flux"),var("normal")))

Here, we have to consider the minus sign in the :py:class:`~pyoomph.meshes.bcs.NeumannBC`, since it has to agree with the weak formulation.

The rest is mainly the same. However, in the config file :download:`precice-config-circle.xml`, we relax a bit the threshold for convergence and use `acceleration <https://precice.org/configuration-acceleration.html>`__. Therefore, this problem requires less simulation time than the previous example.


For running with preCICE, you must place the config file :download:`precice-config-circle.xml` in the same directory and run again the script two times simultaneously:

.. code:: bash

      python partitioned_heat_conduction_circle.py --outdir Dirichlet -P precice_participant=Dirichlet &
      python partitioned_heat_conduction_circle.py --outdir Neumann -P precice_participant=Neumann
      
      
..  figure:: heat_circle.*
	:name: figpreciceheatcircle
	:align: center
	:alt: Non-matching meshes work fine with preCICE
	:class: with-shadow
	:width: 100%

	Dirichlet/Neumann coupling with preCICE on non-matching meshes and with vectorial heat fluxes
      

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <partitioned_heat_conduction_circle.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   	
		    
