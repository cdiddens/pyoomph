.. _secpdeadaptmoffatt:

Moffatt eddies: adaptivity when no tolerance can be chosen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As a first example, let us consider Stokes flow in a sharp corner, which does something remarkable.
Moffatt :cite:`Moffatt1964` has shown that, whenever the opening angle is below about :math:`146\:\mathrm{^\circ}`,
the flow near the apex is not a single decaying motion, but an *infinite* sequence of counter-rotating
eddies. Writing the stream function in polar coordinates as :math:`\psi\sim r^{\lambda}f(\theta)`, the
exponent :math:`\lambda` that satisfies the no-slip conditions on both walls turns out to be complex,
and its imaginary part converts the radial decay into a decaying oscillation. Each eddy is a scaled
copy of the previous one, shrunken by a fixed geometric factor and considerably weaker.

We take a circular sector of half-angle :math:`30\:\mathrm{^\circ}`, meshed with triangles by a
:py:class:`~pyoomph.meshes.gmsh.GmshTemplate`, with no-slip on the two straight walls and a rotating
outer arc to stir the fluid:

.. literalinclude:: moffatt_eddies.py
   :language: python
   :start-at: class WedgeMesh(GmshTemplate):
   :end-at: self.plane_surface("lower_wall", "driven", "upper_wall", name="wedge")

The equations are just Stokes flow with the driving velocity imposed on the arc:

.. literalinclude:: moffatt_eddies.py
   :language: python
   :start-at: eqs = StokesEquations(dynamic_viscosity=1, mode="TH")
   :end-at: eqs += AverageConstraint(pressure=0)

Note that the drive is tapered smoothly to zero where the arc meets the walls. This is done on
purpose: an abrupt velocity jump there would introduce a second singularity of the lid-driven-cavity
kind, which would compete with the apex for the elements of the mesh and hence spoil the experiment.
With the taper, the apex is the only singular point of the domain. Since the velocity is prescribed
on every boundary, the pressure is only determined up to a constant, which is why we also add an
:py:class:`~pyoomph.equations.generic.AverageConstraint`.

To see what the flow looks like, we can simply zoom in. Streamlines are the appropriate tool here,
since they follow the *direction* of the flow and are unaffected by its magnitude, and the velocity is
shown on a logarithmic colour scale, which is unavoidable when several decades must fit into a single
picture.

The result is depicted in :numref:`figpdeadaptmoffattstreamlines`. Each magnification shows a closed
eddy with the next one just appearing in the corner, and each panel looks like the previous one, only
with the colour bar shifted by a few decades. The cascade is self-similar in :math:`\log r`, and we
shall exploit exactly this property below.

..  figure:: moffatt_streamlines.*
	:name: figpdeadaptmoffattstreamlines
	:align: center
	:alt: Streamlines and velocity magnitude at three magnifications about the apex
	:class: with-shadow
	:width: 100%

	Streamlines and velocity magnitude (logarithmic colour scale) on the adapted mesh, at one, ten and
	a hundred times magnification about the apex. Note how the colour bar shifts by roughly four
	decades from one panel to the next.

Just how much weaker each eddy is can be read off :numref:`figpdeadaptmoffattcascade`: the amplitude
falls by a few hundred from one eddy to the next, i.e. by more than nine orders of magnitude in total
over the four sign reversals that are resolved at this budget.

..  figure:: moffatt_cascade.*
	:name: figpdeadaptmoffattcascade
	:align: center
	:alt: Eddy amplitude against radius, showing plateaus of alternating rotation sense
	:class: with-shadow
	:width: 75%

	Amplitude of the azimuthal velocity against the radius. Each plateau corresponds to one eddy, the
	dashed lines mark the sign reversals separating them and the colour indicates the sense of
	rotation.

This is what makes the corner such an awkward customer for adaptive refinement: **there is no
tolerance we could sensibly choose**. However small we make
:py:attr:`~pyoomph.generic.problem.Problem.max_permitted_error`, there is always another eddy below
it, so the mesh would refine towards the apex forever and the adaptation loop would merely run until
it reaches its step limit. The meaningful question is not *how accurate?*, but *how much mesh are we
willing to pay for?* -- and this is precisely what
:py:attr:`~pyoomph.generic.problem.Problem.desired_ndof` expresses. When it is set to an integer,
pyoomph no longer aims at an error tolerance, but refines the elements with the largest errors until
the number of degrees of freedom is approximately the requested one:

.. literalinclude:: moffatt_eddies.py
   :language: python
   :start-at: self.desired_ndof = budget
   :end-at: self.initial_adaption_steps = 0

The adaptation itself then needs nothing special: a plain
:py:meth:`~pyoomph.generic.problem.Problem.solve` with ``spatial_adapt`` adapts up to that many times
and stops as soon as the controller reports that there is nothing left to refine or unrefine.

.. literalinclude:: moffatt_eddies.py
   :language: python
   :start-at: problem.solve(spatial_adapt=30)
   :end-at: problem.solve(spatial_adapt=30)

Let us first try the obvious error criterion, namely the energy-norm one, which estimates the error in
:math:`\operatorname{sym}(\nabla\vec{u})`, i.e. the strain rate that actually carries the viscous
dissipation. Granting it a budget of 80 000 degrees of freedom, we obtain

.. code:: text

   criterion                        ndof      reach elements per decade of r
   sym(grad(u))       (k=0)        73729   9.91e-03 [0, 0, 27, 16971]

where *reach* denotes the smallest nodal radius, i.e. how far into the corner the mesh actually gets,
and the last column counts the elements in each decade of :math:`r`. Of 16 998 elements, 16 971 end up
in the outermost decade alone and the corner is never touched at all -- see the left panel of
:numref:`figpdeadaptmoffattmesh`. Raising the budget does not help either: at 5 000, 20 000 and 80 000
degrees of freedom, the reach is :math:`9.91\times 10^{-3}` every single time, which is nothing but
the spacing of the initial mesh at the apex.

This is not a bug, and it is worth understanding why. A Zienkiewicz-Zhu estimator equidistributes the
error in the *energy norm*, and by :numref:`figpdeadaptmoffattcascade` the Moffatt eddies carry
essentially no energy at all. Two eddies inwards, the velocity has already dropped by five orders of
magnitude, so the elemental error contribution near the apex is utterly negligible compared to
anything happening close to the arc. Inspecting the elemental errors confirms this: they are nearly
uniform across the entire domain -- at this budget their median is :math:`9.7\times10^{-7}`
against a maximum of :math:`4.4\times10^{-6}`, i.e. a spread of less than a decade, where the velocity
itself spans nine. The estimator does exactly
what we asked of it; we simply asked the wrong question.

.. note::

   This failure mode is worth keeping in mind whenever a solution has features that matter physically
   but are weak in magnitude -- a trace species, a small recirculation, or a boundary layer in an
   otherwise quiescent region. If the refinement stubbornly ignores something you care about, compare
   the elemental errors there with those in the region where the mesh *is* being spent, before
   concluding that the adaptivity is broken.

The remedy follows from the fact that the estimator measures whichever flux we hand to it. Since the
eddies are self-similar in :math:`\log r`, dividing the flux by a power of the radius removes that
scaling and lets every decade of :math:`r` compete for the budget on equal terms:

.. literalinclude:: moffatt_eddies.py
   :language: python
   :start-at: r = square_root(x**2+y**2+1e-18)
   :end-at: eqs += SpatialErrorEstimator(flux, normalize_relative=0)

The small constant under the square root merely keeps the expression finite at the apex itself. The
argument ``normalize_relative=0`` asks for the *absolute* error instead of the relative one used by
default. This matters here because we want to compare the two runs with each other: by default, the
elemental errors are divided by the recovered flux norm of the entire mesh, so they are relative to
that particular mesh and consequently cannot be compared between meshes at all. With
``normalize_relative=0``, the errors are absolute and do shrink upon refinement.

At the very same budget, the two criteria now buy us rather different meshes:

.. code:: text

   criterion                        ndof      reach elements per decade of r
   sym(grad(u))       (k=0)        73729   9.91e-03 [0, 0, 27, 16971]
   sym(grad(u))/r**3 (k=3)         74812   6.19e-04 [1, 76, 2673, 15091]

i.e. the same cost, a sixteen-fold deeper reach and a mesh graded over three decades instead of piled
up in a single one, cf. :numref:`figpdeadaptmoffattmesh`.

..  figure:: moffatt_mesh.*
	:name: figpdeadaptmoffattmesh
	:align: center
	:alt: The adapted meshes for the two error criteria at the same budget
	:class: with-shadow
	:width: 90%

	The same budget spent in two ways. Left
	(:math:`k=0`): the plain energy-norm criterion, which places everything near the driven arc.
	Right (:math:`k=3`): the scale-compensated criterion, which grades the mesh towards the apex.
	Neither mesh is wrong -- they simply minimise different quantities.

With a criterion that can actually see the corner, the budget finally becomes a meaningful dial:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20

   * - ``desired_ndof``
     - achieved ndof
     - reach
   * - 5 000
     - 4 750
     - :math:`2.5\times10^{-3}`
   * - 20 000
     - 18 701
     - :math:`1.2\times10^{-3}`
   * - 80 000
     - 74 812
     - :math:`6.2\times10^{-4}`

Each factor of four in the budget halves the reach, i.e. buys one further level of geometric grading
towards the apex, which is just the signature of a self-similar cascade being resolved. It also
illustrates what the budget really purchases here: not *more accuracy* in any single number, but one
more turn of the spiral.

Note that the achieved size always lands a few percent below the target rather than exactly on it.
This is the dead band of the controller: it stops as soon as it is within
:py:attr:`~pyoomph.generic.problem.Problem.desired_ndof_tolerance` (10\% by default) of the target,
since a controller that kept chasing the exact number would never report that there is nothing left to
do, and the adaptation loop would consequently never terminate.

.. warning::

   While :py:attr:`~pyoomph.generic.problem.Problem.desired_ndof` is set, the attributes
   :py:attr:`~pyoomph.generic.problem.Problem.min_permitted_error` and
   :py:attr:`~pyoomph.generic.problem.Problem.max_permitted_error` become *outputs* of the controller,
   which recomputes them before every adaptation, rather than settings under your control. They are
   restored to your values once ``desired_ndof`` is set back to ``None``.

Finally, hard limits such as :py:class:`~pyoomph.equations.generic.RefineToLevel` are deliberately
*not* subject to the budget: a mandatory refinement is still carried out, even if it takes the mesh
beyond the target. The budget only decides how the discretionary part of the mesh is spent, not
whether your constraints are honoured.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <moffatt_eddies.py>`

		:download:`Download all examples <../../tutorial_example_scripts.zip>`
