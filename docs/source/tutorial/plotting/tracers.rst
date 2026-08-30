.. _secplottingtracers:

Plotting tracer particles
-------------------------

Tracer particles are massless points that are carried along by a prescribed vector field - usually
the flow velocity - without feeding back into it. They are a visualisation tool first of all:
pathlines, streak patterns, "where does the fluid at the inlet end up". The plotter draws them like
any other field, along with a trail of where each one has recently been.

In order to plot them, their trajectories must be calculated along the transient simulation. Following pyoomph's design philosophy, tracers are added to the equations of a domain:

.. code:: python

   eqs += TracerParticles(var("velocity"), seed=TracerSeedGrid(0.15))

The first argument is the field that advects them - it defaults to ``var("velocity")`` - and
``seed`` says where they start. :py:class:`~pyoomph.equations.tracers.TracerSeedGrid` lays a lattice over
the domain's bounding box and drops the candidates that fall outside the mesh, so it is safe on a
domain with a hole in it. The alternatives are
:py:class:`~pyoomph.equations.tracers.TracerSeedPoints` (explicit positions),
:py:class:`~pyoomph.equations.tracers.TracerSeedElement` (one per element, which follows a graded mesh
rather than a box), :py:class:`~pyoomph.equations.tracers.TracerSeedRandom` and
:py:class:`~pyoomph.equations.tracers.TracerSeedCallable`. All of them work in one, two and three
dimensions.

Let us put them in a channel whose top wall oscillates, so that the mesh is in constant motion:

.. literalinclude:: tracers.py
   :language: python
   :start-at: class WavyChannel(Problem):
   :end-at: self += small

Two of the keyword arguments deserve a word.

``history_time`` keeps a rolling window of each particle's recent positions, i.e. the last
:math:`0.4` time units here. That is what a trail is drawn from.

``payloads`` gives each particle scalars that are integrated along its own path, by the same
sub-steps that move it. ``{"residence": 1}`` is therefore just a simple example, it accumulates the time the particle has spent
in the domain; a strain rate would accumulate strain, and a concentration would accumulate exposure.
The sources must be dimensionless, and are integrated over nondimensional time. Read them back with
:py:meth:`~pyoomph.meshes.mesh.BaseMesh.get_tracers` and ``get_payloads()``.

In the plotter, a tracer collection is addressed by its name just like a field:

.. literalinclude:: tracers.py
   :language: python
   :start-at: class TracerPlotter(MatplotlibPlotter):
   :end-before: class WavyChannel

The same plot is written twice here, once as a vector graphic and once as a raster image at half
the resolution. Note that ``image_size`` alone fixes the pixel count while ``dpi`` says how many
inches those pixels are, so halving *both* keeps the figure the same size in inches: the labels,
the markers and the colorbar keep their proportion instead of growing to twice their share of a
half-size canvas.

The last line of the equation system is what keeps the picture worth looking at. A particle that
reaches the outlet has left the mesh and is simply dropped, so the channel would empty out from the
left over the course of the run. :py:class:`~pyoomph.equations.tracers.TracerPeriodicBoundaryCondition`
puts it back in at the periodic image of its position instead:

.. code:: python

   eqs += TracerPeriodicBoundaryCondition(vector(-self.length, 0)) @ "right"

The particle keeps its identity and its payloads - the residence time here goes on accumulating
across the jump - and it finishes the rest of its timestep from the inlet rather than stopping at
the boundary, so a wrap costs no accuracy. Note that only the *particles* are made periodic by this,
not the flow: the solution in this example is a Poiseuille inflow and a free outflow, and stays so.
The shift is registered on the collection rather than on the boundary, so it does not matter which
end of a periodic pair you attach it to, and attaching it to both is harmless.

.. figure:: tracers_channel.*
	:name: figplottingtracers
	:align: center
	:alt: Tracer pathlines in a channel with an oscillating wall
	:class: with-shadow
	:width: 100%

	Tracer particles and their trails in a channel whose upper wall oscillates. The mesh moves
	considerably over a period, but the particles follow only the flow.

Tracers on an interface
~~~~~~~~~~~~~~~~~~~~~~~

Adding the same equation to an interface instead of a bulk domain confines the particles to that
interface. There is no separate class and no flag - where you attach it is the statement:

.. code:: python

   eqs += TracerParticles(var("velocity"), tracer_name="surface_tracers") @ "top"

Such a particle is advected by the tangential component of the field only, and follows the
interface exactly in the normal direction. So on a free surface it stays on the surface as the
surface deforms, moving along it at the tangential flow speed, and a purely normal flow does not
slide it sideways. 

Some further remarks
~~~~~~~~~~~~~~~~~~~~

* Particles that leave the domain are removed, unless you say where they should go instead: to a
  neighbouring domain sharing an interface, with a
  :py:class:`~pyoomph.equations.tracers.TracerTransferAtInterface` (both sides must carry tracers);
  onto the domain's own interface, with a
  :py:class:`~pyoomph.equations.tracers.TracerTransferToInterface`, which is what a free surface
  losing mass wants - the surface recedes past the particle rather than the particle swimming out;
  or back in at the far end, with a
  :py:class:`~pyoomph.equations.tracers.TracerPeriodicBoundaryCondition` as above.
* A trail outlives the particle that drew it. One that leaves is gone from the simulation
  immediately - it stops being advected and drops out of ``get_positions()`` - but its trail stays
  in the picture and fades out over ``history_time`` instead of blinking out with the marker.
* A particle confined to an interface cannot leave it, only reach the end of it, and it stays
  there. On a free surface with a pinned contact line the particles therefore accumulate at the
  rim, which is the transport behind the coffee-ring effect rather than an artefact.
* Advection happens once per accepted timestep. Stationary solves, continuation steps and the
  intermediate solves of spatial adaptation do not move the particles.
* The accuracy is bounded by the mesh: within a timestep the solver only knows where the nodes were
  at the beginning and at the end, so the in-step mesh configuration is interpolated, and that
  interpolation - not the accuracy of the particle integrator - is what limits the result on a
  strongly moving mesh.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <tracers.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`
