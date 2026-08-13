.. _secdgfacetfields:

Unknowns on the interior facets
--------------------------------

In the previous examples, the facets were only used to couple bulk fields, i.e. the interior facet terms were assembled from the values of the bulk field on both sides of each facet. However, the interior facets form a mesh of their own, the *skeleton* mesh, and this mesh can also carry its own unknowns. Note again that pyoomph, other than e.g. NGsolve, only counts the interior facets to the skeleton mesh, not the exterior facets, which must/can be treated separately. 
Unknowns on facets (i.e. on the skeleton mesh) are the essential ingredient of hybridized and hybridizable discontinuous Galerkin (HDG) methods: instead of gluing the bulk elements together by penalty terms, one introduces a single-valued quantity on each facet, e.g. the trace of the solution or the flux through the facet, and lets it mediate the coupling.

In pyoomph, the interior skeleton mesh is available under the reserved domain name ``"_internal_facets_"``. Whenever equations are added to this domain, i.e. by ``eqs+=MyFacetEquations()@"_internal_facets_"``, the skeleton is generated, and any field defined in ``define_fields`` of these equations is a genuine unknown on the facets. In particular, one does not have to set ``requires_interior_facet_terms`` by hand in that case, this is implied by the presence of equations on the skeleton. Instead one has to explitly define an equation class for the skeleton mesh.

The degrees of freedom of such a facet field are stored in the facet element itself, not at nodes. Consequently, only discontinuous spaces, i.e. ``"D0"``, ``"DL"``, ``"D1"``, ``"D1TB"``, ``"D2"`` and ``"D2TB"``, can be used here. Each facet then owns its own set of values: two facets meeting at a common vertex do not share anything, which is exactly what one wants for a facet-wise trace. 

As an example, we consider the Poisson equation

.. math::

   -\nabla^2 u = f

on a fully discontinuous space, i.e. we do not add any penalty term to enforce the continuity of :math:`u`. The element-wise weak form reads

.. math::

   \sum_E \left\{ \left(\nabla u,\nabla v\right)_E - \left(f,v\right)_E\right\} + \sum_F \left\langle \vec{n}^+\cdot\nabla u , v^+-v^-\right\rangle_F = 0\,,

where, as before, :math:`\vec{n}^+` is the facet normal pointing from :math:`E^+` to :math:`E^-`. Without any further terms, the facet contribution is unknown and the elements are entirely decoupled. In the hybridized formulation, one therefore introduces an unknown :math:`\lambda` on each facet, replaces the unknown facet flux by it and enforces the continuity of :math:`u` by using :math:`\lambda` as a Lagrange multiplier, i.e. we solve

.. math::

   \begin{aligned}
   \sum_E \left\{ \left(\nabla u,\nabla v\right)_E - \left(f,v\right)_E\right\} + \sum_F \left\{\left\langle \lambda , \operatorname{jump}(v)\right\rangle_F +   \left\langle \operatorname{jump}(u) , \mu\right\rangle_F \right\} &= 0
   \end{aligned}

By comparing both expressions, it is apparent that :math:`\lambda=-\vec{n}^+\cdot\nabla u` at the solution, i.e. the multiplier is nothing but the single-valued flux transmitted through the facet. This is what we will check numerically below.

The bulk equation is just the plain Poisson equation, written on a discontinuous space and without any facet terms at all:

.. literalinclude:: hybrid_poisson.py
   :language: python
   :start-at: class HybridizedPoissonEquation(Equations):
   :end-at: self.add_residual(weak(grad(u),grad(v))-weak(self.source,v))

The coupling is a separate equation class, which will be added to the skeleton domain. It defines the facet unknown ``lam`` and adds both of the above facet terms. Note how the bulk field ``u`` is bound by :py:func:`~pyoomph.expressions.generic.var` without any ``domain`` argument, exactly as in the facet terms of the previous examples, whereas ``lam`` is a field of the facet equations themselves:

.. literalinclude:: hybrid_poisson.py
   :language: python
   :start-at: class HDGCoupling(Equations):
   :end-at: self.set_facet_recovery("lam",-dot(var("normal"),avg(grad(u))))

The call of :py:meth:`~pyoomph.generic.codegen.BaseEquations.set_facet_recovery` will be discussed at the end of this section, it is only relevant when the mesh changes (remeshing, spatial adaptivity).

In the problem class, the facet equations are attached to ``"_internal_facets_"`` of the parent domain. We use a manufactured solution :math:`u=\sin(\pi x)`, i.e. :math:`f=\pi^2\sin(\pi x)`, so that we can monitor both the error of :math:`u` and the deviation of :math:`\lambda` from the exact flux by :py:class:`~pyoomph.equations.generic.IntegralObservables`:

.. literalinclude:: hybrid_poisson.py
   :language: python
   :start-at: class HybridizedPoissonProblem(Problem):
   :end-at: self.max_refinement_level=1

We use a one-dimensional mesh here, since the method cannot be trivially extended to higher dimensions. Since a facet of a one-dimensional mesh is just a point, the "integrals" over the facets are plain sums over all interior facets.

Running the script, i.e.

.. literalinclude:: hybrid_poisson.py
   :language: python
   :start-at: if __name__=="__main__":
   :end-at: problem.output()

gives, with ``N=10`` elements,

.. code:: text

   solution: L2 error of u = 5.798e-03, |jump(u)| = 1.108e-08, error of lam = 1.421e-05, unfilled facets = 0

The jump of :math:`u` vanishes up to the tolerance of the Newton solver, i.e. the multiplier really does glue the discontinuous field back together. The scheme is therefore equivalent to the continuous first order Galerkin discretization and converges with second order: after the uniform refinement, the error of :math:`u` drops from :math:`5.8\cdot 10^{-3}` to :math:`1.5\cdot 10^{-3}`. The multiplier agrees with the exact flux :math:`-\vec{n}^+\cdot\nabla u` to about :math:`10^{-5}`, which confirms the interpretation given above.

Facet fields under mesh adaptation and remeshing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The skeleton is never adapted incrementally: it is discarded and generated from scratch on the refined bulk mesh. The element-owned values of the ``"D0"`` and ``"DL"`` facet fields are hence sampled before the adaptation and fitted back onto the new facets afterwards, including all time history levels. Facets which are just split by the refinement thereby keep their values, but the facets which are created entirely new at refined elements are a problematic: there is no previous value available at that position in the old skeleton, so no interpolation can produce a value there.

By default, these values are set to zero, and a warning is issued:

.. code:: text

   WARNING: transferring the discontinuous (DL/D0) fields of interface '_internal_facets_' across an
   adaptation left 10 of 19 new element(s) without a single sample point. [...]

Since a facet unknown is usually determined by the bulk solution anyhow, the better answer is to say how it should be reconstructed there, which is what :py:meth:`~pyoomph.generic.codegen.BaseEquations.set_facet_recovery` in the ``HDGCoupling`` class above does: on facets without any transferred value, ``lam`` is evaluated from the surrounding bulk solution, here just by the flux :math:`-\vec{n}\cdot\operatorname{avg}(\nabla u)`. In the output of the script, the difference is clearly visible. With the recovery expression, we get

.. code:: text

   after refinement, before solving: L2 error of u = 6.320e-03, |jump(u)| = 1.249e-08, error of lam = 2.881e-02, unfilled facets = 0
   after refinement, solved:         L2 error of u = 1.453e-03, |jump(u)| = 1.219e-08, error of lam = 1.329e-06, unfilled facets = 0

whereas without it, the ten facets created inside the refined elements are reported as unfilled and their multiplier is off by orders of magnitude before the next solve:

.. code:: text

   after refinement, before solving: L2 error of u = 6.320e-03, |jump(u)| = 1.249e-08, error of lam = 7.025e+00, unfilled facets = 10

In a stationary problem, this is repaired by the next Newton solve, which is why zero is a tolerable default, but in a transient simulation the wrong values would enter the time derivatives. The recovery expression is written to all time levels of the new facets, so that no spurious time derivative is generated by a facet which just came into existence.

The same transfer is used upon remeshing (cf. :numref:`secaleremeshing`), where each new facet takes the values of the closest facet of the old skeleton within the same old bulk element. Only for an identical remesh, this is exact, otherwise the error is of the order of the distance to the closest old facet times the gradient of the facet field. For a facet field which is a trace or a flux of the bulk solution, :py:meth:`~pyoomph.generic.codegen.BaseEquations.set_facet_recovery` is therefore the recommended choice also here.

.. note::

   Facet fields work under MPI, in both modes of :numref:`secmpimodes`. With ``--distribute`` the mesh
   is partitioned and a facet whose two elements land on different processes is *owned* by the one that
   assembles it, the other holding a halo copy - so it stays one unknown, numbered once. The same
   ``"DL"``/``"D0"`` restriction as for adaptivity applies, and for the same reason: distributing
   rebuilds every facet element, and only those two spaces are carried across the rebuild. A nodal
   ``"D1"``/``"D2"`` facet field is refused by ``distribute()`` with a message saying so.

Condensing the bulk unknowns away
'''''''''''''''''''''''''''''''''

An unknown on the facets also changes what the *bulk* unknowns are coupled to. Without one, the facet terms of a discontinuous Galerkin method connect each element directly to its neighbours. With one, the elements only ever talk to the facets between them, so the bulk unknowns of an element couple to that element alone - which is precisely the condition under which they can be eliminated from the linear system element by element. That is the subject of the next section, :numref:`secdghdg`.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <hybrid_poisson.py>`

		:download:`Download all examples <../tutorial_example_scripts.zip>`
