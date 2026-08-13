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

The decisive property is which unknowns talk to each other. The unknowns of an element couple to that element and to the :math:`\hat{u}` of its own facets - and to nothing else. In particular they never couple to the unknowns of the neighbouring element, which in an interior penalty formulation they would.

Setting it up
~~~~~~~~~~~~~

The bulk equation has no facet terms whatsoever - everything that connects the elements sits on the skeleton:

.. code:: python

   class HDGPoissonEquations(Equations):
       def __init__(self, source, space="D2"):
           super().__init__()
           self.source, self.space = source, space

       def define_fields(self):
           self.define_scalar_field("u", self.space)

       def define_residuals(self):
           u, v = var_and_test("u")
           self.add_residual(weak(grad(u), grad(v)) - weak(self.source, v))

The facet equations own :math:`\hat{u}` and add the three boundary integrals of :math:numref:`eqdghdgform`, summed over the two elements sharing the facet. Since :math:`\vec{n}` is the outward normal of the element the facet is attached to, the other element contributes the same expression with :math:`-\vec{n}`. The one-sided values are recovered from the jump and the average, using :math:`a_\text{near}=\operatorname{avg}(a)+\operatorname{jump}(a)/2` and :math:`a_\text{far}=\operatorname{avg}(a)-\operatorname{jump}(a)/2`:

.. code:: python

   u_n, u_f = avg(u) + jump(u) / 2, avg(u) - jump(u) / 2
   ...
   r = -weak(dot(n, gu_n), v_n - vhat) - weak(-dot(n, gu_f), v_f - vhat)
   r += -weak(u_n - uhat, dot(n, gv_n)) - weak(u_f - uhat, -dot(n, gv_f))
   r += weak(self.tau * (u_n - uhat), v_n - vhat) + weak(self.tau * (u_f - uhat), v_f - vhat)

Since the skeleton comprises the *interior* facets only, the exterior boundary needs its own terms: the same expression with :math:`\hat{u}` replaced by the prescribed value, i.e. Nitsche's method.

.. warning::

   In that boundary term, the gradient must be bound through the parent domain, i.e. by
   ``var_and_test("u", domain=self.get_parent_domain())``. A ``"D1"``/``"D2"`` field can be read
   directly on an interface, as discussed in :numref:`secdg`, but ``grad()`` of it is then the
   *surface* gradient. With the interface binding, ``dot(n, grad(u))`` quietly evaluates to something
   that is not the normal derivative, the boundary condition contributes nothing, and the method
   silently degrades to first order instead of failing.

Condensing
~~~~~~~~~~

Because the bulk unknowns of an element couple to nothing outside it, they can be eliminated by a small dense Schur complement per element before the Jacobian reaches the solver, and reconstructed afterwards. That is what :py:class:`~pyoomph.equations.generic.StaticCondensation` does (cf. :numref:`secspatialcrcondensation`), and one line is enough:

.. code:: python

   eqs += StaticCondensation("u")

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

   Two practical points. The automatic selection
   :py:meth:`~pyoomph.generic.problem.Problem.condense_element_private_dofs` does **not** pick up the
   bulk field here, because the facet elements read it as external data; it has to be named
   explicitly, as above. And the example keeps adaptivity off simply to compare a fixed sequence of
   meshes; every discontinuous facet space, the nodal ``"D1"``/``"D2"`` included, is carried through
   a spatial adaptation, a remesh, a state file and ``--distribute``.

Under MPI
~~~~~~~~~

Both parallel modes of :numref:`secmpimodes` run this example, condensation included.

With ``--distribute`` the mesh is partitioned, and a facet whose two elements land on different processes is *owned* by the one that assembles it while the other holds a halo copy - so the trace :math:`\hat{u}` is one unknown, numbered once, exactly as it is serially. Distributing rebuilds every facet element, but the trace is carried across that rebuild whatever its space is, so ``facet_space="DL"`` (with ``space="D1"`` for the bulk) is a modelling choice here rather than a restriction.

Without ``--distribute`` the run is *replicated* - every process holds the whole mesh and only the assembly and the linear system are split - and the elimination is served there too. That it is, is a property of *which* dofs are being eliminated. A block can only be condensed on the process that owns all of its rows, and the rows are cut into contiguous ranges; pyoomph moves those cut points off the blocks, which it can do exactly when each element's selected dofs are numbered together. Element-internal values - the bulk field here, and ``"DL"``/``"D0"`` fields generally - are. A selection mixing *nodal* and element-internal dofs is not, because oomph-lib numbers every nodal value before any internal one: the Crouzeix-Raviart selection of :numref:`secspatialcrcondensation`, which pairs the bubble velocity (nodal) with the pressure gradients (internal), therefore has to be run with ``--distribute``, where each process's dofs are renumbered contiguously and the question does not arise. That case is refused with a message saying so, not silently skipped.

.. warning::

   The hybridization is what makes this work, not the discontinuous space. If the facet terms couple
   the two sides of a facet *directly* - an interior penalty formulation with a ``jump(u)*jump(v)``
   term, say - then the bulk unknowns of the whole mesh form a single coupled block, and the
   elimination is refused with an explanation rather than silently performing a second, worse, direct
   solve. The stabilization term is likewise not optional: without it each element-local problem is a
   pure Neumann problem, hence singular, and the elimination breaks down on it.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <hdg_poisson.py>`

		:download:`Download all examples <../tutorial_example_scripts.zip>`
