.. _secopenmp:

OpenMP threads within one process: 
----------------------------------

This parallelizes the evaluation of each element's residual vector and Jacobian block.

Pass ``--omp N`` on the command line (cf. :numref:`installcmdlineoptions`) or call
:py:meth:`~pyoomph.generic.problem.Problem.set_num_threads` in the script:

.. code:: bash

   python3 my_simulation.py --omp 4


Both ``--omp`` and :py:meth:`~pyoomph.generic.problem.Problem.set_num_threads` set the threading of the
**linear solver** as well, which most direct solvers bring by themselves. Without either, pyoomph's own
element loop stays serial and ``OMP_NUM_THREADS`` is pinned to one, so that a third-party OpenMP runtime
in the same process cannot quietly open a second pool of threads next to pyoomph's. The BLAS thread
counts (``MKL_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``) default to four instead, since that is what the
solvers were already using. All of these are overridable through the corresponding ``PYOOMPH_*``
environment variables (cf. :numref:`installenvvars`).

The threaded assembly requires a build with OpenMP, which is the ``PYOOMPH_USE_OPENMP`` CMake option
(cf. :numref:`installcmakeoptions`). Its default ``AUTO`` links OpenMP if the toolchain provides it,
so whether a given installation has it is answered by

.. code:: python

   from pyoomph import _pyoomph_core
   print(_pyoomph_core.has_openmp)

.. note::

   The threaded assembly is **bit-identical** to the serial one, i.e. it changes the time a run takes
   and nothing else. That is a deliberate design property rather than a happy accident, see below, and
   it means ``--omp`` can be switched on and off without re-validating any result.

   A threaded direct *solver* is a different matter: it is not bit-reproducible, so a converged state
   can differ in its last digits between one thread and several.

Computing the element blocks in parallel is easy; writing them into the sparse matrix is not, because
two elements sharing a degree of freedom write the same entry. Guarding those writes with atomics
would work, but it would make the *summation order* - and hence the last bits of every matrix entry -
depend on the thread schedule.

Pyoomph therefore inverts the *scatter* into a *gather*. The elements are processed in chunks, and
each chunk is done in two phases: first every thread computes its elements' blocks into its own slice
of a scratch buffer, then each thread is given a disjoint range of *target entries* of the matrix and
sums the contributions belonging to them. Since the gather index is sorted by target entry, each entry
receives its contributions in element order - the very order the serial loop adds them in - and the
result agrees with the serial assembly bit for bit.

The threaded loop covers the Newton residual and Jacobian, the eigenvalue and mass matrices, the
augmented systems of bifurcation tracking and arclength continuation, and the distributed assembly
under MPI.

.. warning::

   There are situations in which pyoomph *declines* to thread the assembly. It then says so once and
   runs the serial loop, so the run stays correct and merely does not get faster. The most common ones
   are a domain using a finite-difference Jacobian (whose perturbations touch data shared with the
   neighbouring elements) or a build without OpenMP.

   Because a silent fallback and a working fast path give exactly the same answer, check
   ``problem._get_parallel_assemblies_done()``, which counts the loops that actually ran threaded,
   before believing any speed measurement.


The element loop scales close to linearly; the whole ``assemble_jacobian`` call does not, because part
of it is inherently serial - handing out the CSR structure and, on a mesh with hanging nodes, the
pre-pass that interpolates them. On a 2d lid-driven cavity with Q2/Q1 Navier-Stokes and 89 402 degrees
of freedom, on four cores:

.. list-table::
    :widths: 40 15 15 15
    :header-rows: 1

    *   -
        - 1 thread
        - 2
        - 4
    *   - element loop
        - 143 ms
        - 73 ms
        - 38 ms
    *   - whole Jacobian assembly
        - 193 ms
        - 132 ms
        - 97 ms

So the loop itself gains 3.7x and the call as a whole 2.0x.

Elements are handed to the threads by work stealing rather than in equal counts, so a problem whose
elements cost wildly different amounts - two domains with different physics, interface elements next to
bulk elements, a cheap scalar field next to a 3d Navier-Stokes block - balances itself without anyone
having to estimate a cost per element.

Finally, if a Python callback is evaluated during assembly, e.g. a
:py:class:`~pyoomph.expressions.cb.CustomMathExpression`, the threads have to take the Python global
interpreter lock in turn for it. In a callback-heavy problem, you therefore should expect only little gain.

``--omp`` combines with ``mpirun``: the threads run inside each rank. Keep the product of ranks and
threads at or below the number of physical cores, and see :numref:`secmpimodes` - in particular the
warning about ``mpirun`` pinning each rank to a single core, which turns ``--omp`` into a no-op.
