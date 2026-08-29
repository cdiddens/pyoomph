.. _secdghdg:

Hybridizable DG: eliminating the bulk unknowns
-----------------------------------------------

The multiplier of the previous section glues the discontinuous field back together, but it does so in a way that does not carry over to higher dimensions. A *hybridizable* discontinuous Galerkin (HDG) method arranges the same idea differently, and the rearrangement buys something substantial: the bulk unknowns can be removed from the linear system entirely, leaving only the unknowns on the skeleton.

The starting point is the Poisson equation on a discontinuous space, element by element. Multiplying :math:`-\nabla^2u=f` by a test function and integrating by parts over a single element :math:`K` gives

.. math:: :label: eqdghdgelement

   \int_K \nabla u\cdot\nabla v \;-\; \oint_{\partial K} \partial_n u\, v \;=\; \int_K f v

with :math:`\vec{n}` the outward normal of :math:`K`. On its own this is a collection of independent Neumann problems. In HDG, the coupling is introduced by a single-valued unknown :math:`\hat{u}` on each facet, which plays the role of the trace of :math:`u` there, and the element contribution becomes

.. math:: :label: eqdghdgform

   \int_K \nabla u\cdot\nabla v
   -\oint_{\partial K} \partial_n u\,(v-\hat{v})
   -\oint_{\partial K} (u-\hat{u})\,\partial_n v
   +\oint_{\partial K} \tau\,(u-\hat{u})(v-\hat{v})
   \;=\;\int_K f v

Both added terms vanish at the exact solution, where :math:`u=\hat{u}` on every facet, so the method is consistent. The second one restores the symmetry of the bilinear form, and the third, with a stabilization parameter :math:`\tau` scaling like :math:`1/h`, is what makes it stable.

The unknowns of an element couple to that element and to the :math:`\hat{u}` of its own facets - and to nothing else. In particular they never couple to the unknowns of the neighbouring element, which in an interior penalty formulation they would.

Both parts can be written in the same class: ``add_interior_facet_residual`` assembles a weak form on the skeleton, and ``at_internal_facets=True`` declares a field there, so :math:`\hat{u}` is a genuine facet unknown rather than a bulk one. Both need ``requires_interior_facet_terms`` in the constructor, which is how pyoomph learns about the skeleton while the equation tree is still being assembled - long before ``define_fields`` is called. Splitting the facet part off into its own class added with ``@"_internal_facets_"``, as in :numref:`secdgfacetfields`, is equivalent.

.. literalinclude:: hdg_poisson.py
   :language: python
   :start-at: class HDGPoissonEquations(Equations):
   :end-at: self.add_residual(weak(grad(u), grad(v)) - weak(self.source, v))

The facet residual adds the three boundary integrals of :math:numref:`eqdghdgform`, summed over the two elements sharing the facet. Since :math:`\vec{n}` is the outward normal of the element the facet is attached to, the other element contributes the same expression with :math:`-\vec{n}`. The one-sided values are recovered from the jump and the average, using :math:`a_\text{near}=\operatorname{avg}(a)+\operatorname{jump}(a)/2` and :math:`a_\text{far}=\operatorname{avg}(a)-\operatorname{jump}(a)/2`:

.. literalinclude:: hdg_poisson.py
   :language: python
   :dedent: 8
   :start-at: uhat, vhat = var_and_test("uhat")
   :end-at: self.add_interior_facet_residual(r)

Since the skeleton comprises the *interior* facets only, the exterior boundary needs its own terms: the same expression with :math:`\hat{u}` replaced by the prescribed value, i.e. Nitsche's method.

.. warning::

   In that boundary term, the gradient must be bound through the parent domain, i.e. by
   ``var_and_test("u", domain=self.get_parent_domain())``. A ``"D1"``/``"D2"`` field can be read
   directly on an interface, as discussed in :numref:`secdg`, but ``grad()`` of it is then the
   *surface* gradient. With the interface binding, ``dot(n, grad(u))`` quietly evaluates to something
   that is not the normal derivative, the boundary condition contributes nothing, and the method
   silently degrades to first order instead of failing.


Because the bulk unknowns of an element couple to nothing outside it, they can be eliminated by a small dense Schur complement per element before the Jacobian reaches the solver, and reconstructed afterwards. That is what :py:class:`~pyoomph.equations.generic.StaticCondensation` does (cf. :numref:`secspatialcrcondensation`), and one line is enough:

.. literalinclude:: hdg_poisson.py
   :language: python
   :dedent: 12
   :start-at: eqs += StaticCondensation("u")
   :end-at: eqs += StaticCondensation("u")

Running :download:`hdg_poisson.py` on an :math:`8\times 8` mesh with ``"D2"`` for both spaces reports

.. code:: text

     elements                : 64
     interior facets         : 112
     degrees of freedom      : 912
     condensed away          : 576 in 64 blocks of 9
     seen by the solver      : 336
     L2 error of u           : 5.1415e-05

i.e. exactly one block per element, each of the size of the element's own unknowns (:math:`9` for ``"D2"`` on a quadrilateral), and the solver is handed the :math:`112\times 3=336` facet unknowns alone. That the global system is the size of the skeleton rather than of the bulk is the defining property of an HDG method, and here it is simply what the elimination leaves behind.

The elimination is exact, so the accuracy is that of the unreduced scheme. Refining the mesh gives the expected orders, with ``"D1"``/``"DL"`` converging quadratically and ``"D2"``/``"D2"`` cubically:

.. list-table::
   :header-rows: 1
   :widths: 20 26 26

   * - mesh
     - ``"D1"``, ``"DL"``
     - ``"D2"``, ``"D2"``
   * - :math:`4\times 4`
     - 1.028e-02
     - 4.093e-04
   * - :math:`8\times 8`
     - 2.570e-03
     - 5.142e-05
   * - :math:`16\times 16`
     - 6.426e-04
     - 6.435e-06


.. note::

   The hybridization is what makes this work, not the discontinuous space. If the facet terms couple
   the two sides of a facet directly - an interior penalty formulation with a ``jump(u)*jump(v)``
   term, say - then the bulk unknowns of the whole mesh form a single coupled block, and the
   elimination is refused with an explanation rather than silently performing a second, worse, direct
   solve. The stabilization term is likewise not optional: without it each element-local problem is a
   pure Neumann problem, hence singular, and the elimination breaks down on it.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <hdg_poisson.py>`

		:download:`Download all examples <../tutorial_example_scripts.zip>`
