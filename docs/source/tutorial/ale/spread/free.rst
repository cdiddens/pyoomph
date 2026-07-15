With free slip at the substrate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similarly as done with lubrication theory in :numref:`eqpdelubric_spread`, we now want to let a droplet spread until it reaches its equilibrium contact angle. This time, however, we want to solve the full bulk flow including inertia and considering the full interface curvature. Hence, we use again the free surface equations from the previous example.

The key part to impose an equilibrium contact angle is the :math:`[\cdot,\cdot]` term in :math:numref:`eqaleweaksigmafs`. It has not been considered so far, but it will become relevant now. This term can be added to boundaries of the free surface, i.e. to contact lines. The surface tension is fully balanced if :math:`\vec{N}` is the outward pointing normal of the contact line, which is the outward pointing tangential continuation of the free interface at the boundaries. Let :math:`\theta` be the equilibrium contact angle, then :math:`\vec{N}` will read :math:`(\cos(\theta),-\sin(\theta))` if the droplet is at the equilibrium contact angle. Let us define the corresponding equation class that has to be added to the contact line to enforce this contact angle:

.. literalinclude:: droplet_spread_free_slip.py
   :language: python
   :start-at: from free_surface import * # Load our free surface implementation
   :end-at: self.add_residual(-weak(sigma*self.N,v))

It is rather trivial, how :math:`-[\sigma\vec{N},\vec{v}]` is implemented here. Note how we can access the free interface by accessing the ``DynamicBC`` equation of the parent domain, i.e. of the free surface, by :py:meth:`~pyoomph.generic.codegen.InterfaceEquations.get_parent_equations`. Since we have set ``required_parent_type=DynamicBC``, pyoomph knows that :py:meth:`~pyoomph.generic.codegen.InterfaceEquations.get_parent_equations` should give the ``DynamicBC`` and not the ``KinematicBC`` contribution. Only the former has the surface tension property ``sigma`` defined.

The problem class itself is very similar to the previous example:

.. literalinclude:: droplet_spread_free_slip.py
   :language: python
   :start-at: class DropletSpreadingProblem(Problem):
   :end-at: problem.run(50,outstep=True,startstep=0.25)

Important differences are the mesh, which is now the north-east (``"NE"``) quarter of a :py:class:`~pyoomph.meshes.simplemeshes.CircularMesh` with renamed boundaries and a suitable :py:class:`~pyoomph.equations.generic.RefineToLevel` by thereof, the selection of an ``"axisymmetric"`` coordinate system and the fact that now both :math:`x` and :math:`y` mesh coordinates are allowed to move. We set the :math:`x`-coordinate (i.e. the :math:`r`-coordinate) of the mesh and the :math:`x`-velocity (:math:`r`-velocity) to zero at the axis of symmetry and likewise the :math:`y`-coordinate and :math:`y`-velocity to zero at the substrate. Note that the :math:`x`-velocity at the substrate is completely free, i.e. it corresponds to an (unrealistic) free slip boundary at the moment. This can be seen on the left side of :numref:`figaledropletspread`. The droplet spreads quickly and the fluid can flow unhindered tangentially along the substrate.


..  figure:: droplet_spread.*
	:name: figaledropletspread
	:align: center
	:alt: Droplet spreading
	:class: with-shadow
	:width: 70%

	(left) Spreading with free slip at the substrate. (right) Spreading with a tiny slip length.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <droplet_spread_free_slip.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
