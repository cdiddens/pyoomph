.. _secspatialcrcondensation:

Static condensation of the element-local degrees of freedom
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The previous section covered Crouzeix-Raviart element, which has the benefit of elementwise exact continuity
equation, but it comes at the cost of additional degrees of freedom. These additional degrees of freedom will impact the performance of the linear solver, however, several degrees of freedom can actually be removed before they enter the linear solver at all.

Comparing to Taylor-Hood, the additional of the Crouzeix-Raviart element are element-local. The additional velocity bubble and the discontinuous pressure field (cf. :math:numref:`eqspatialdlpressure`) are accessed only within each element, i.e. they are not shared with neighboring elements.
In particular, the gradients :math:`p_x`, :math:`p_y` (and :math:`p_z` in 3d) are irrelevant for neighboring elements since they just enforce the continuity equation within the element. Such unknowns, which do not couple to anything outside its own element can be eliminated from the linear system by a small dense Schur complement before the Jacobian is factorized, and reconstructed from the solution afterwards. This is called *static condensation*, and it is what :py:class:`~pyoomph.equations.generic.StaticCondensation` does.

When using pyoomph's predefined implementation the :py:class:`~pyoomph.equations.NavierStokesEquations`, eliminating both types of condensable degrees of freedom can be realizes via:

.. code:: python

    eqs = NavierStokesEquations(mass_density=100, dynamic_viscosity=1, mode="CR")
    eqs += StaticCondensation(velocity="bubble", pressure="DL_gradients")


We indicate the the ``"bubble"`` degree of the velocity and the gradients terms of the ``"DL"`` space of the pressure should be consensed, the rest is taken care of automatically.
Instead of ``"DL_gradients"``, also ``pressure=[1,2]`` in 2d and ``pressure=[1,2,3]`` in 3d is permitted and resulting in the same -- but the former variant is valid for arbitrary dimensions.

Note that the elimination via condensation is exact: It ends up in the same Newton iterations and the same solution to the last digit, as without it.

One cannot condense arbitrary degrees of freedom. In particular, one cannot condense either the velocity bubble *or* the pressure gradients.
The continuity equation is tested against the pressure test function, but it does not contain any pressure at all. Selecting only the pressure therefore leaves a block that
is structurally zero. Vice versa, selecting only the bubble velocities leaves out the pressure gradients, which the bubble has to couples to.

Despite of being local to each element only, the constant pressure mode :math:`p_0`, must remain a global unknown: The bubble velocity has no coupling to it, which can be seen when applying the divergence theorem on the continuity equation :math:`\nabla\cdot\vec{u}` integrated over an element in combination with the fact that a bubble is zero along the elements edges.

.. note::

   ``velocity="bubble"`` selects the bubble depending on the element type.
   On a triangle or tetrahedron, ``"C2TB"`` enriches ``"C2"`` by a cubic bubble, i.e. this bubble will be selected.
   On a quadrilateral or brick, ``"C2TB"`` is the very same as ``"C2"``, but the node at the centroid has exactly the properties required for the condensation and is hence selected as well when the velocity is defined on ``"C2"``, e.g. for Taylor-Hood elements.


The example code :download:`cr_static_condensation.py` itself is not discussed in detail here, but the outcome is shown in the following table:

.. list-table::
   :header-rows: 1
   :widths: 26 18 18 18 20

   * -
     - elements
     - unknowns
     - condensed
     - seen by the solver
   * - 2d tris, :math:`N=24`
     - 2304
     - 20545
     - 9216
     - 11329
   * - 3d tetras, :math:`N=6`
     - 1115
     - 16612
     - 6690
     - 9922
   * - 2d quads, :math:`N=24`
     - 576
     - 6145
     - 2304
     - 3841
   * - 3d hexs, :math:`N=8`
     - 512
     - 12172
     - 3072
     - 9100

Almost half of the degrees of freedom never end up in the solver, and simultaneously, the resulting matrix did not increase complexity.
In the first 2d case, e.g., the non-zeros drop from 504532 to 265559.

The factorization time and memory requirement is of course also reduced (measured with ``--superlu``):

.. list-table::
   :header-rows: 1
   :widths: 22 14 16 16 12 20

   * -
     - unknowns
     - factorization
     - with condensation
     - saved
     - peak memory
   * - 2d tris, :math:`N=24`
     - 20545
     - 0.416 s
     - 0.193 s
     - 54 %
     - 388 -> 299 MB
   * - 2d tris, :math:`N=48`
     - 82561
     - 4.562 s
     - 1.813 s
     - 60 %
     - 1594 -> 849 MB
   * - 3d tetra, :math:`N=6`
     - 16612
     - 7.419 s
     - 2.406 s
     - 68 %
     - 1096 -> 599 MB
   * - 2d quads, :math:`N=24`
     - 6145
     - 0.073 s
     - 0.037 s
     - 49 %
     - 221 -> 194 MB
   * - 3d hexs, :math:`N=8`
     - 12172
     - 3.657 s
     - 1.944 s
     - 47 %
     - 779 -> 586 MB

So the performance of the factorization can become more than twice as efficient and costs only about half the memory.
However, there is of course overhead associated with the condensation in the assembly (Schur complement, inversion of small dense matrices and the reconstruction after the linear solve) associated with it.


.. note::
	A block of selected degrees of freedom can only be eliminated on the process that owns all of its rows. With ``--distribute`` each process renumbers its own degrees of freedom contiguously, so that is automatic. In a *replicated* run (``mpirun`` without ``--distribute``) the mesh is not partitioned but the rows of the linear system still are, and this selection pairs the bubble velocity (nodal) with the pressure gradients (element-internal) - which oomph-lib numbers far apart, every nodal value coming before any internal one. The two halves of a block would then land in different processes' row ranges.

	That is why the script above sets

	.. code:: python

		problem.dof_ordering = ElementBlockOrdering("domain/velocity_*", "domain/pressure")

	before ``initialise()``: it numbers each element's velocities and pressure together, so the blocks are short enough for pyoomph to move the row split off them. In serial this changes nothing at all - the same blocks, the same non-zeros and the same answer - and with ``--distribute`` it is equally harmless. Without it, a replicated run is refused with a message saying so. A selection of purely element-internal degrees of freedom, e.g. the one of :numref:`secdghdg`, needs none of this and is served in both modes either way.


.. warning::
	:py:class:`~pyoomph.equations.generic.StaticCondensation` is experimental. Only conventional Newton solves benefit:
	residual evaluations, eigenvalue and Hessian assemblies and arclength continuation always see the full system. 
	Jacobian reuse and the globally convergent (line search) Newton method are not supported.
	

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <cr_static_condensation.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
