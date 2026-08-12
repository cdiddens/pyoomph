.. _secmpidofreduction:

Making the system smaller instead
---------------------------------

More processes are not the only way to get a result sooner, and often not the first one to try. A
Newton step consists of two rather different pieces of work - assembling the residual and the Jacobian,
and factorizing that Jacobian - and they respond to completely different remedies. Assembly is a loop
over elements and is what ``mpirun`` splits between processes (:numref:`secmpimodes`). The factorization
is the solver's business, it grows considerably faster than linearly with the number of unknowns, and
the way to make it cheaper is to hand the solver fewer of them.

This section shows two ways of doing that on the same example, and - just as importantly - how to find
out whether either of them can help your problem at all.

The example
~~~~~~~~~~~

A droplet oscillating under surface tension, discretized with MINI elements on tetrahedra: the velocity
lives on ``"C1TB"``, i.e. piecewise linear plus one bubble node per tetrahedron, and the pressure on
``"C1"``. The mesh moves with the free surface, so the nodal positions are unknowns as well. The full
script is :download:`oscillating_droplet_dofs.py <oscillating_droplet_dofs.py>`; the two reductions
below are switched on by its ``--constrain`` and ``--condense`` flags.

The mesh has 2046 tetrahedra and the problem 16342 unknowns. Where they sit is worth spelling out,
because it is what both reductions exploit: each tetrahedron carries a velocity bubble node of its own
(3 unknowns), and since ``"C1TB"`` is the highest continuous space in the problem, pyoomph also uses it
as the *coordinate* space - so every tetrahedron carries a mesh-position bubble too, another 3. Two
sets of 3 x 2046 = 6138 unknowns, three quarters of the system between them, and every one of them
element-local.

Constraining the mesh bubbles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mesh-position bubbles are the ones nobody asked for: they exist because the velocity needs the
bubble, not because the geometry does. :py:class:`~pyoomph.equations.ALE.ConstrainPositionsToC1Space`
ties every non-vertex position to the linear interpolation of its element's corners, which is what a
straight-sided tetrahedron has anyway::

    eqs += ConstrainPositionsToC1Space()

That removes exactly those 6138 unknowns, 16342 -> 10204.

.. warning::

   This is a modelling decision, not free bookkeeping. The bubble positions were genuine degrees of
   freedom of the moving mesh, and taking them away is a different discretization of the mesh motion:
   the computed solution changes slightly - in the example, an oscillation amplitude of 0.11182319
   becomes 0.11182170. That is a perfectly reasonable thing to accept, but it has to be an accepted
   approximation rather than an assumed identity. Note also that the coordinate space cannot simply be
   lowered instead: pyoomph refuses a coordinate space below the highest nodal field space, which with
   ``mode="mini"`` is ``"C1TB"``.

Condensing the velocity bubbles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The velocity bubbles cannot be removed - they are what makes the element pair stable - but they need
not reach the solver. A bubble unknown couples only to the other unknowns of its own element, so it can
be eliminated by a small dense Schur complement before the matrix is factorized and reconstructed from
the result afterwards. That is :py:class:`~pyoomph.equations.generic.StaticCondensation`::

    eqs += StaticCondensation(velocity="bubble")

Adding it anywhere in the equation tree also switches the feature on. The elimination is **exact**: the
same Newton iterations and, in the example, a solution identical to the last printed digit. What
changes is only what the solver is given - here 2046 blocks of at most 3, 6138 unknowns eliminated and
841122 non-zeros reduced to 721612.

The two combine, and then the solver sees 4066 of the original 16342 unknowns, with the non-zeros down
from 546030 to 346726:

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - variant
     - unknowns
     - condensed
     - seen by the solver
   * - plain
     - 16342
     - -
     - 16342
   * - ``--constrain``
     - 10204
     - -
     - 10204
   * - ``--condense``
     - 16342
     - 6138
     - 10204
   * - both
     - 10204
     - 6138
     - 4066

But does it help?
~~~~~~~~~~~~~~~~~

A quarter of the original system sounds decisive, so it is worth being precise about what it buys. On
this problem, with a direct solver, measured per Newton step:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - unknowns
     - factorization
     - with condensation
     - saved
     - Jacobian assembly
   * - 16342
     - 0.068 s
     - 0.040 s
     - 41 %
     - 1.16 s
   * - 43515
     - 0.260 s
     - 0.146 s
     - 44 %
     - 3.03 s
   * - 91267
     - 0.958 s
     - 0.532 s
     - 44 %
     - 6.11 s

Static condensation does exactly what it promises - it takes a reliable 40 to 45 % off the
factorization at every size - and it makes **no difference whatsoever to the runtime**, because on this
problem the factorization is not where the time goes. Even at 91267 unknowns, assembling the Jacobian
costs six times as much as factorizing it, so removing 44 % of the factorization removes about 6 % of
the step, which is lost in the noise of everything else.

This is not a defect of the method, it is a statement about the problem. The two costs scale
differently: the assembly is a loop over elements and grows linearly with the mesh, while the
factorization in these measurements grows roughly like :math:`n^{1.8}`. Extrapolating the two lines,
they cross about an order of magnitude further up, i.e. somewhere around a million unknowns for this
particular problem - and a Crouzeix-Raviart discretization, whose condensable block is much larger than
MINI's 3x3, reaches that point far sooner and saves closer to half of the factorization.

.. note::

   The lesson is the one worth taking from this whole chapter: **find out which half of the step you
   are paying for before optimizing either.** The solver prints both numbers each step - the time to
   set up the Jacobian and the time to factorize it - and their ratio tells you immediately whether to
   reduce unknowns or to parallelize the assembly.

   The droplet here is firmly assembly-bound, and that makes it a poor candidate for condensation and
   an excellent one for ``mpirun`` *without* ``--distribute``: that mode parallelizes precisely the
   element loop that dominates, and needs no change to the script at all.

Limitations
~~~~~~~~~~~

:py:class:`~pyoomph.equations.generic.StaticCondensation` is experimental and **serial only**. An MPI
run is refused with an error rather than silently ignored, and so are Jacobian reuse and the globally
convergent (line search) Newton method. Only Newton solves benefit: residual evaluations, eigenvalue
and Hessian assemblies and arclength continuation always see the full system. The two halves of this
chapter are therefore mutually exclusive today - condense, or run in parallel, but not both. The
example script queries the number of processes and simply leaves condensation off when there is more
than one::

    if get_mpi_nproc() > 1:
        print("More than one process: leaving static condensation off, it is serial only.")
    else:
        eqs += StaticCondensation(velocity="bubble")

Without an argument, ``StaticCondensation()`` selects every element-private unknown of its domain,
which is the convenient form for auxiliary fields projected onto a discontinuous space. For a
Crouzeix-Raviart discretization the bubble velocities and the pressure gradient modes have to be named
together - neither is invertible on its own - while the constant pressure mode must remain global::

    eqs = NavierStokesEquations(mode="CR", dynamic_viscosity=1, mass_density=100)
    eqs += StaticCondensation(velocity="bubble", pressure=[1, 2])   # [1,2,3] in 3d
