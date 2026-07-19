.. _secpdestokes_law:

Transient and nonlinear generalization of Stokes' law
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please refer to :numref:`secspatialstokes_law` for understanding the problem setup. We will only address the required changes here.

First of all, we do not only have to balance the drag force :math:`F_\text{d}` with the buoyancy force :math:`F_g`, but also the inertia :math:`M\dot{U}` matters, where :math:`M` is the mass of the spherical object. To that end, instead of using the :py:class:`~pyoomph.generic.codegen.GlobalLagrangeMultiplier` we write a simple :py:class:`~pyoomph.generic.codegen.ODEEquations` class to account for Newton's law of motion:

.. literalinclude:: navier_stokes_around_object.py
   :language: python
   :start-at: class NewtonEquation(ODEEquations):
   :end-at: self.add_residual(weak(self.M*partial_t(U)-self.F_buo,V))

As in the previous example, to this equation of motion, the drag force will be added externally by the ``DragContribution`` added to the interface of fluid and object.

In the problem class, to minimize errors stemming from the imposition of the pure Stokes solution at the far field, we shrink the spherical object and increase the radius of the far field:

.. literalinclude:: navier_stokes_around_object.py
   :language: python
   :start-at: self.sphere_radius = 0.25 * milli * meter  # radius of the spherical object
   :end-at: self.outer_radius = 50 * milli * meter  # radius of the far boundary

Since we have a temporal problem, also a time scale has to be set

.. literalinclude:: navier_stokes_around_object.py
   :language: python
   :start-at: self.set_scaling(temporal=self.sphere_radius/UStokes_ana)
   :end-at: self.set_scaling(temporal=self.sphere_radius/UStokes_ana)

We then have to exchange the ``StokesEquations`` by the ``NavierStokesEquations`` to account for the inertia. However, when transforming into the co-moving frame of reference, we have to correct for the acceleration of the co-moving system. This can be added via the ``bulkforce`` argument, which adds a contribution proportional to :math:`\rho_\text{l}\dot{U}\vec{e}_z` to the ``NavierStokesEquations``. This effectively changes the time derivative in the inertia to :math:`\rho_\text{l}(\partial_t \vec{u}-\dot{U} \vec{e}_z)` so that the acceleration of the coordinate system :math:`\dot{U}` cancels out:

.. literalinclude:: navier_stokes_around_object.py
   :language: python
   :start-at: U = var("UStokes", domain="globals")  # bind U from the domain "globals"
   :end-at: eqs = NavierStokesEquations(dynamic_viscosity=self.fluid_viscosity,mass_density=self.fluid_density,bulkforce=inertia_correction)  # Stokes equation and output

Instead of the :py:class:`~pyoomph.generic.codegen.GlobalLagrangeMultiplier` we add the developed ``NewtonEquation``:

.. literalinclude:: navier_stokes_around_object.py
   :language: python
   :start-at: # Define the Lagrange multiplier U
   :end-at: self.add_equations(U_eqs @ "globals")  # add it to an ODE domain named "globals"

Note that we combine it with an :py:class:`~pyoomph.output.generic.ODEFileOutput` to write the time evolution of :math:`U(t)` to a file.

Finally, the run code must be transient now:

.. literalinclude:: navier_stokes_around_object.py
   :language: python
   :start-at: if __name__ == "__main__":
   :end-at: problem.run(0.5*second,startstep=0.05*second,outstep=True)  # solve and output

As seen in :numref:`figpdetransientstokeslaw`, the final velocity field is not symmetric anymore and a transient dynamics of :math:`U(t)` is plotted.

..  figure:: transient_stokes.*
	:name: figpdetransientstokeslaw
	:align: center
	:alt: Velocity around a spherical object with consideration of inertia.
	:class: with-shadow
	:width: 100%

	(left) Velocity around a spherical object with consideration of inertia. (right) Evolution of the velocity :math:`U(t)`.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <navier_stokes_around_object.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
