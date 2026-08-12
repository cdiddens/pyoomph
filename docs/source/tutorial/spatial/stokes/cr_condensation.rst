.. _secspatialcrcondensation:

Static condensation of the element-local degrees of freedom
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The previous section closed on the price of the Crouzeix-Raviart element: the elementwise continuity
equation is bought with additional degrees of freedom. It is worth looking at where exactly those extra
unknowns sit, because most of them never have to reach the linear solver at all.

Both additions of the Crouzeix-Raviart element are **element-local**. The velocity carries one node in
the interior of each element, which no neighbour shares. The pressure
of :math:numref:`eqspatialdlpressure` is stored inside the element as well, and of its degrees of
freedom the gradients :math:`p_x`, :math:`p_y` (and :math:`p_z` in 3d) are what the neighbours do not
see. An unknown that couples to nothing outside its own element can be eliminated from the linear
system by a small dense Schur complement before the Jacobian is factorized, and reconstructed from the
solution afterwards. This is called *static condensation*, and it is what
:py:class:`~pyoomph.equations.generic.StaticCondensation` does::

    eqs = NavierStokesEquations(mass_density=100, dynamic_viscosity=1, mode="CR")
    eqs += StaticCondensation(velocity="bubble", pressure="DL_gradients")

Adding the class anywhere in the equation tree also switches the feature on. It contributes no
residuals whatsoever - it only states which unknowns may be eliminated - and the elimination is
**exact**: the same Newton iterations, and the same solution to the last digit, as without it.

Why both halves are required
""""""""""""""""""""""""""""

The two selections above have to be made together. Neither is invertible on its own, and the reason is
worth understanding rather than memorizing.

The continuity equation, tested against the pressure test function, contains no pressure at all - it is
:math:`\langle \nabla\cdot\vec{u}, q\rangle`. Selecting only the pressure therefore leaves a block that
is structurally zero, and pyoomph refuses it with exactly that explanation. Selecting only the bubble
velocities leaves out the pressure gradients, which the bubble is what couples to.

The constant pressure mode :math:`p_0`, on the other hand, must **stay** a global unknown. The bubble
velocity has no coupling to it: the integral of :math:`\nabla\cdot\vec{u}` of a bubble against a
constant test function vanishes, since the bubble is zero on the whole element boundary. Taking it
along makes the little block singular, and pyoomph says so, naming the offending degrees of freedom.

That is what ``pressure="DL_gradients"`` means: every value of the ``"DL"`` field except the constant.
Spelling the indices out - ``pressure=[1,2]`` in 2d and ``pressure=[1,2,3]`` in 3d - does exactly the
same thing, but has to be changed when the same script is run in another dimension, whereas
``"DL_gradients"`` is resolved from the dimension of the domain it is added to.

.. note::

   ``velocity="bubble"`` works on every element family, although the node it selects gets there by
   different routes. On a triangle or tetrahedron, ``"C2TB"`` genuinely enriches ``"C2"`` by a cubic
   bubble. On a quadrilateral or brick, ``"C2TB"`` *is* ``"C2"`` - as discussed in the previous
   section - but the ``"C2"`` element already carries a node at its centroid, and that node's shape
   function vanishes on the entire element boundary just as a bubble does. So it is interior to its
   element by the same argument, and is condensed by the same rule.

   Either way it is exactly one velocity node, i.e. one velocity vector plus the pressure gradients
   per element: 4 unknowns per element in 2d and 6 in 3d, on simplices and on tensor-product elements
   alike. The example below defaults to triangles and tetrahedra and takes ``--quads`` for the
   quadrilateral and brick case; the measurements further down cover both.

An example
""""""""""

:download:`cr_static_condensation.py` solves the classical lid-driven cavity with the predefined
:py:class:`~pyoomph.equations.navier_stokes.NavierStokesEquations` in ``mode="CR"``, and reports the
size of the system the solver is handed:

.. code:: bash

   python3 cr_static_condensation.py --condense           # triangles
   python3 cr_static_condensation.py --3d --condense      # tetrahedra
   python3 cr_static_condensation.py --quads --condense   # quadrilaterals

Since every velocity boundary is prescribed, the pressure is fixed only up to a constant, which is why
the script calls ``with_pressure_fixation()`` as discussed in :numref:`secspatialstokespuredirichlet`.
Note that this and the condensation do not collide: the fixation pins the *constant* mode, which is
exactly the one the selection leaves alone, and a pinned value is never condensed anyway.

The counts come out per element, as they must: in 2d two interior velocity components and two pressure
gradients, in 3d three of each - whatever the element shape:

.. list-table::
   :header-rows: 1
   :widths: 26 18 18 18 20

   * -
     - elements
     - unknowns
     - condensed
     - seen by the solver
   * - 2d, ``--N 24``
     - 2304
     - 20545
     - 9216
     - 11329
   * - 3d, ``--3d --N 6``
     - 1115
     - 16612
     - 6690
     - 9922
   * - 2d, ``--quads --N 24``
     - 576
     - 6145
     - 2304
     - 3841
   * - 3d, ``--quads --3d --N 8``
     - 512
     - 12172
     - 3072
     - 9100

So a little under half of the system never reaches the solver, and the matrix that does is
substantially sparser as well - in the first 2d case the non-zeros drop from 504532 to 265559.

What it buys
""""""""""""

The point of all this is the factorization, so that is what to measure. The numbers below are the time
the linear solver spends factorizing the Jacobian, and the peak memory of the whole process, with and
without the condensation:

.. list-table::
   :header-rows: 1
   :widths: 22 14 16 16 12 20

   * -
     - unknowns
     - factorization
     - with condensation
     - saved
     - peak memory
   * - 2d, ``--N 24``
     - 20545
     - 0.416 s
     - 0.193 s
     - 54 %
     - 388 -> 299 MB
   * - 2d, ``--N 48``
     - 82561
     - 4.562 s
     - 1.813 s
     - 60 %
     - 1594 -> 849 MB
   * - 3d, ``--3d --N 6``
     - 16612
     - 7.419 s
     - 2.406 s
     - 68 %
     - 1096 -> 599 MB
   * - 2d, ``--quads --N 24``
     - 6145
     - 0.073 s
     - 0.037 s
     - 49 %
     - 221 -> 194 MB
   * - 3d, ``--quads --3d --N 8``
     - 12172
     - 3.657 s
     - 1.944 s
     - 47 %
     - 779 -> 586 MB

So the factorization costs roughly a third to a half of what it did - on simplices and on
tensor-product elements alike - and the memory it needs falls by a comparable amount. In 3d the latter
is often the more pressing of the two: it is the memory of the factors, not the time, that decides
whether a direct solve is possible at all.

.. note::

   These were measured with ``--superlu``, i.e. a plain single-threaded direct solver, deliberately.
   The saving comes from the sparsity of the matrix that reaches the solver, so it is the *ratio* that
   carries over, not the absolute times. Measuring the same thing with a heavily optimized threaded
   solver such as Pardiso or MUMPS gives much smaller absolute times, in which fixed overheads and
   thread scheduling swamp the effect - on the 2d case above, Pardiso showed a 28 % saving on a
   factorization of 10 ms, which says more about the measurement than about the elimination.

   And a saved factorization is only worth something if the factorization is what you are paying for.
   For a problem dominated by the assembly instead, none of this shows up in the runtime at all;
   :numref:`secmpidofreduction` works through exactly such a case and shows how to tell the difference.


Limitations
"""""""""""

:py:class:`~pyoomph.equations.generic.StaticCondensation` is experimental. Only Newton solves benefit:
residual evaluations, eigenvalue and Hessian assemblies and arclength continuation always see the full
system. Jacobian reuse and the globally convergent (line search) Newton method are refused with an
error rather than silently ignored, and so is a *replicated* ``mpirun`` - more than one process without
``--distribute`` - while a distributed run is supported. See :numref:`secmpidofreduction` for the same
technique applied to a different element pair, and for how to find out whether your problem is limited
by the factorization at all.
