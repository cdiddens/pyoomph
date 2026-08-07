.. _secALEtracers:

Tracer particles on moving meshes
---------------------------------

Tracer particles are massless points that are carried along by a prescribed vector field - usually
the flow velocity - without feeding back into it. They are a visualisation tool: pathlines, streak
patterns, "where does the fluid at the inlet end up".

On a **moving mesh** they are less trivial than they look, because the mesh slides underneath them.
It is worth being explicit about what pyoomph guarantees here, since it is the property that makes
the results trustworthy: in a bulk domain, a particle sitting in a mesh that moves while the
advection field is zero **does not move at all**. Not approximately - the mesh motion cancels out of
the equation of motion analytically, so it is never even computed. Whatever your mesh smoother, your
remeshing or your free surface does to the nodes, the particles follow the flow and nothing else.

Adding tracers to a domain is one line:

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
   :end-at: self += TracerPlotter(self)

Two of the keyword arguments deserve a word.

``history_time`` keeps a rolling window of each particle's recent positions, i.e. the last
:math:`0.4` time units here. That is what a **trail** is drawn from, and it costs nothing when you
do not ask for it. In the plotter, a tracer collection is addressed by its name just like a field:

.. literalinclude:: tracers.py
   :language: python
   :start-at: class TracerPlotter(MatplotlibPlotter):
   :end-before: class WavyChannel

``payloads`` gives each particle scalars that are **integrated along its own path**, by the same
sub-steps that move it. ``{"residence": 1}`` therefore accumulates the time the particle has spent
in the domain; a strain rate would accumulate strain, and a concentration would accumulate exposure.
The sources must be dimensionless, and are integrated over nondimensional time. Read them back with
:py:meth:`~pyoomph.meshes.mesh.BaseMesh.get_tracers` and ``get_payloads()``.

.. figure:: tracers_channel.*
	:name: figaletracers
	:align: center
	:alt: Tracer pathlines in a channel with an oscillating wall
	:class: with-shadow
	:width: 100%

	Tracer particles and their trails in a channel whose upper wall oscillates. The mesh moves
	considerably over a period, but the particles follow only the flow.

Tracers on an interface
~~~~~~~~~~~~~~~~~~~~~~~

Adding the same equation to an **interface** instead of a bulk domain confines the particles to that
interface. There is no separate class and no flag - where you attach it is the statement:

.. code:: python

   eqs += TracerParticles(var("velocity"), tracer_name="surface_tracers") @ "top"

Such a particle is advected by the **tangential** component of the field only, and follows the
interface exactly in the **normal** direction. So on a free surface it stays on the surface as the
surface deforms, moving along it at the tangential flow speed, and a purely normal flow does not
slide it sideways. The normal offset stays at machine precision however long you run.

Some further remarks
~~~~~~~~~~~~~~~~~~~~

* Particles that leave the domain are removed. If two domains share an interface and both carry
  tracers, add a :py:class:`~pyoomph.equations.tracers.TracerTransferAtInterface` to hand them over
  instead.
* Tracers work under ``--distribute``. Each particle is owned by one process, migrates when it
  crosses a partition boundary, and ``gather_positions()`` gives the same answer on every process.
  State files hold the whole set and can be written at one process count and read at another.
* Advection happens once per accepted timestep. Stationary solves, continuation steps and the
  intermediate solves of spatial adaptation do not move the particles.
* The accuracy is bounded by the mesh: within a timestep the solver only knows where the nodes were
  at the beginning and at the end, so the in-step mesh configuration is interpolated, and that
  interpolation - not the ``rtol`` of the particle integrator - is what limits the result on a
  strongly moving mesh. See ``dev_docs/tracers.md`` for the details.
