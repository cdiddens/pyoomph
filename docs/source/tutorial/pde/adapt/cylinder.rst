.. _secpdeadaptcylinder:

A heated cylinder: mixed meshes and competing error criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The previous example had a single field and a single criterion. Let us now look at a problem with
several fields at once, where the interesting question is not *how much* mesh to spend, but *on which
field* to spend it.

We consider the classical flow past a cylinder at :math:`\mathrm{Re}=40`, which is steady and
therefore cheap. The cylinder is held at a fixed temperature, so a thermal wake trails behind it.
In addition, a *trace species* enters through the inlet as a thin filament well above the cylinder
and is carried straight downstream. The tracer is deliberately weak -- its concentration is
:math:`10^{-3}` times the temperature scale -- and it passes through a region where nothing else of
interest happens.

Both scalars obey the same advection-diffusion equation, so we write it once and instantiate it
twice:

.. literalinclude:: heated_cylinder.py
   :language: python
   :start-at: class AdvectionDiffusion(Equations):
   :end-at: + weak(grad(c)/self.peclet, grad(q)))

Before coming to the error criteria, the mesh deserves a look of its own. A boundary layer is
resolved far more economically by quadrilaterals aligned with the wall than by triangles, whereas
the far field is much easier to fill with triangles. pyoomph lets us have both in a single domain:
we build a structured O-grid from four transfinite sectors of an annulus and recombine them into
quadrilaterals, while everything outside stays triangular. Making the radial lines transfinite with
a ``"Progression"`` coefficient grades them, so the layers are thin at the wall and coarsen
outwards.

.. literalinclude:: heated_cylinder.py
   :language: python
   :start-at: class CylinderMesh(GmshTemplate):
   :end-at: self.plane_surface(*box, holes=[ring], name="fluid")

The initial mesh then consists of 288 quadrilaterals in the boundary layer and 1348 triangles
outside it, cf. :numref:`figpdeadaptcylmixed`. Both element types are refined by the adaptivity
without any further ado -- a quadrilateral is split into four quadrilaterals, a triangle into four
triangles, and the hanging nodes on the interface between the two families are taken care of
automatically.

..  figure:: cylinder_mixed.*
	:name: figpdeadaptcylmixed
	:align: center
	:alt: Mixed mesh with quadrilaterals in the boundary layer and triangles outside
	:class: with-shadow
	:width: 40%

	The initial mixed mesh around the cylinder: a structured, radially graded O-grid of
	quadrilaterals in the boundary layer, unstructured triangles in the far field.

As in the previous example, we fix the cost of the computation rather than an error tolerance, by
setting :py:attr:`~pyoomph.generic.problem.Problem.desired_ndof` to 80 000 and letting
:py:meth:`~pyoomph.generic.problem.Problem.solve` adapt until the controller is satisfied. The only
thing we vary is *how the error is measured*.

The obvious choice is to hand all three fields to a single
:py:class:`~pyoomph.equations.generic.SpatialErrorEstimator`:

.. code:: python

   eqs += SpatialErrorEstimator(velocity=1, temperature=1, tracer=1)

This puts all of them into one *compound flux group*, which means their elemental errors are summed
before being normalised by the recovered flux norm of that group. And here lies the catch: the tracer
is a thousand times weaker than the temperature, so its contribution to that sum is a *million* times
smaller. It is, for all practical purposes, invisible. The alternative is to give each field a group
of its own:

.. literalinclude:: heated_cylinder.py
   :language: python
   :start-at: eqs += SpatialErrorEstimator(velocity=1, group="flow")
   :end-at: eqs += SpatialErrorEstimator(tracer=1, group="trace",weight=100)

Now every field is divided by *its own* recovered flux norm and is therefore judged on its own scale,
and the three groups are combined by taking the maximum. A common factor on a field cancels out of
its own group exactly, so it no longer matters that the tracer is small -- only that it is
under-resolved. The ``weight`` on the tracer group is applied *after* that normalisation and hence
does not cancel: it is the knob that says how much we care about the filament relative to the rest.
Running both variants at the same budget gives

.. code:: text

   criterion   ndof  quads   tris filament cylinder       Nu
   joint      77742   1404   6826      376     2054    6.662
   grouped    74146    150   9028     1433      488    7.070

where *filament* counts the elements in the band that the tracer passes through and *cylinder* those
within one diameter of the cylinder. The difference is exactly what the grouping was meant to
achieve: the joint criterion spends its budget near the cylinder and leaves the tracer filament with
376 elements, whereas the grouped criterion gives the filament 1433 and takes the difference out of
the cylinder region. :numref:`figpdeadaptcylmeshes` shows the two meshes; in the lower panel the
filament is plainly visible as a refined stripe running the full length of the domain.

..  figure:: cylinder_meshes.*
	:name: figpdeadaptcylmeshes
	:align: center
	:alt: The adapted meshes for the joint and the grouped error criterion
	:class: with-shadow
	:width: 50%

	The same budget of 80 000 degrees of freedom, spent under the two criteria. Top: all fields in
	one group, so the tracer never influences the mesh. Bottom: one group per field, whereupon the
	tracer filament is resolved as a refined stripe.

Note also how the quadrilateral count follows along: refining near the cylinder subdivides
boundary-layer quadrilaterals, so the joint criterion ends up with 1404 of them while the grouped one
keeps only 150. The mixed mesh is adapted as a mixed mesh throughout.

What that buys is shown in :numref:`figpdeadaptcylfields`. The temperature is essentially the same
either way -- it is a strong field and both criteria resolve it. The tracer is not: under the joint
criterion its filament is smeared out and visibly ragged, the signature of a streak carried on a
mesh too coarse to hold it, while under the grouped criterion it stays narrow and clean all the way
to the outlet.

..  figure:: cylinder_fields.*
	:name: figpdeadaptcylfields
	:align: center
	:alt: Temperature and tracer under the two error criteria
	:class: with-shadow
	:width: 100%

	Temperature (top) and tracer concentration (bottom, in units of its inlet value) under the joint
	(left) and the grouped (right) criterion. The temperature is unaffected by the choice; the
	tracer filament is smeared by the joint criterion and sharp under the grouped one.

One issue is noteworthy: The mean Nusselt numbers are :math:`6.662` and
:math:`7.070`, a difference of six percent -- so the choice of error criterion does not merely
rearrange the picture, it changes a physically meaningful output.

The sound conclusion is therefore: at a fixed cost, what you ask the estimator to look at changes
what you get. The grouped criterion is not "more accurate"; it resolves the tracer, and it pays for
that out of the only purse available.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <heated_cylinder.py>`

		:download:`Download all examples <../../tutorial_example_scripts.zip>`
