.. _secmpi:
.. _secmpimodes:

Processes and distributed meshes: MPI
-------------------------------------

Where threads only split the element loop within one process, MPI splits the whole simulation over
several processes, including the linear solve and - if you ask for it - the storage of the mesh itself.
That is what lets a problem run which does not fit into one machine, and it is the mode to use on a
cluster.

Requirements
~~~~~~~~~~~~

MPI support is a compile-time option. The CMake switch ``PYOOMPH_USE_MPI`` defaults to ``OFF``
(cf. :numref:`installcmakeoptions`), and the precompiled wheels on PyPI are built with that default, so
a ``pip install pyoomph`` cannot be run in parallel. To use MPI you have to
build pyoomph from source with the switch turned on (cf. :numref:`installcompile`), e.g.

.. code:: bash

   python -m pip install --no-build-isolation -e . --config-settings=cmake.define.PYOOMPH_USE_MPI=ON

On the Python side, ``mpi4py`` must be importable. You can test whether your pyoomph installation is ready for MPI with

.. code:: bash

   python -m pyoomph check mpi

From within a script, the same question is answered by

.. code:: python

   from pyoomph.generic.mpi import has_mpi
   print(has_mpi())



Distributing the mesh (the third mode below) additionally requires `PyMetis <https://pypi.org/project/PyMetis/>`__
for the graph partitioning:

.. code:: bash

   python -m pip install pymetis


A parallel run is started with ``mpirun`` (or ``mpiexec``, or whatever your MPI installation and queueing
system provide) as any other MPI program:

.. code:: bash

   mpirun -n 4 python3 my_simulation.py # run with 4 processes, non-distributed meshes
   mpirun -n 4 python3 my_simulation.py --distribute # run with 4 processes, distributed meshes

As usual in MPI running, the actual run script cannot select or change the number of MPI processes, it has to be specified by the mpirun/mpiexec launcher.

The three modes
~~~~~~~~~~~~~~~

Besides the serial run without MPI, there are two ways of running with MPI which differ in what is shared between the processes.
They are shown, together with a serial run for reference, in :numref:`figmpimodes`.

..  figure:: parallel_modes.*
	:name: figmpimodes
	:align: center
	:alt: The three ways of running a pyoomph simulation
	:class: with-shadow
	:width: 100%

	The same 8x8 quadrilateral mesh in the three modes. (left) Serial: one process owns everything.
	(centre) ``mpirun`` without ``--distribute``: every process still holds the entire mesh, but the
	element loop of the assembly is split between them - the colors show which process integrates
	which elements. (right) ``mpirun --distribute``: the mesh itself is partitioned, and each process
	only holds its own part (plus a thin halo of neighbouring elements, not drawn).

**Serial, without** ``mpirun``. One process holds the mesh, assembles the residual and Jacobian and
solves the linear system. This is not necessarily single-threaded: with ``--omp N`` the element loop of
the assembly is threaded (:numref:`secopenmp`), and most direct solvers are internally OpenMP threaded
as well, so a single-process pyoomph run can keep several cores busy in both halves of a Newton step.
For small to medium problems, in particular for simple ODEs or 1d problems, this is very often the
fastest/easiest option.

**With** ``mpirun``, **without** ``--distribute``. Every process holds the whole mesh(es) and the whole
vector of unknowns. The assembly is parallel - oomph-lib splits the element loop between the
processes  - and the linear solve, for which the assembled Jacobian is laid out across the processes by rows. 
Everything else is replicated: each process computes (and must compute) the same numbers.

It does not reduce the memory per process at all: every process stores the whole mesh, and the assembled matrix rows on top of it.

**With** ``mpirun --distribute``. The mesh is partitioned with METIS and each process keeps only its own
part (together with *halos* of neighbouring elements). Assembly, storage and the linear solve are all distributed,
which leads to still more total memory requirement than a serial run, but considerably less than without ``--distribute``.
Note that the initial mesh is still constructed on all processes, then METIS splits it on rank 0 and afterwards, it is pruned on each rank to only hold the assigned subset of elements.
This mode is intended for large problems or HPC clusters.


.. warning::

   MPI requires that each process agrees with the others on the state of the simulation. 
   Any random number or non-deterministic behaviour can end up in serious problems, which eventually usually manifest as a deadlock, where all processes are stuck.
   Use e.g. :py:class:`~pyoomph.expressions.utils.DeterministicRandomField` if you need random numbers. 


Combining MPI with threads
~~~~~~~~~~~~~~~~~~~~~~~~~~

The two are independent and may be used together: ``mpirun -n 2 python3 my_simulation.py --omp 2``
runs two processes of two threads each. Keep the product of ranks and threads at or below the number of
physical cores.

.. warning::

   ``mpirun`` pins each rank to a single core by default, which makes ``--omp N`` a **no-op that still
   looks like it works**: the threads are created, the loop does run threaded and every number is
   right - they simply take turns on the one core the rank is allowed to use. Launch with
   ``mpirun --bind-to none``, or with a binding that gives each rank as many cores as it has threads.
   Pyoomph prints a one-off note when it notices the mismatch.

With the binding fixed, ranks and threads are roughly interchangeable for the *Jacobian* assembly,
while ranks are clearly better for the *residual*: its threaded share is smaller, so the parts MPI also
splits dominate. Hence the rule of thumb from :numref:`secparallel` - take ranks first, and reach for
``--omp`` under ``mpirun`` for what more ranks cannot give you: memory headroom, or a core count beyond
what the mesh partitioner divides sensibly.

Choosing the linear solver
~~~~~~~~~~~~~~~~~~~~~~~~~~

When running with MPI with more than one process, the linear solver will default to PETSc with MUMPS (``--petsc_mumps``) if it is available. 
PETSc is the only supported backend that is genuinely parallel, and it is the one that scales to large problems.

The serial backends - ``pardiso``, ``superlu``, ``umfpack`` and Apple's ``accelerate`` - are not simply
refused under ``mpirun``. Pyoomph gathers the assembled system onto process 0, solves it there (potentially with OpenMP threads, see :numref:`secopenmp`), and
scatters the result back, while the other processes wait. You will receive a warning message if this fallback is used.
It is a real speed-up for an assembly-dominated problem, but process 0 needs the entire matrix in memory.


**For a genuinely parallel solve, use PETSc.** Two of its configurations are directly available:

``--petsc_mumps``
      PETSc with MUMPS as the underlying direct solver, i.e. a distributed sparse LU factorization. This
      is the robust default choice, and the one pyoomph selects by itself under ``mpirun`` when it is
      available. Like every direct solver it is limited by the memory the factors need, which grows much
      faster than the matrix itself, especially in 3d.

``--petsc``
      PETSc's Krylov solvers, i.e. an iterative solve. This is what eventually scales - both in time and
      in memory, since no factors are formed - but an iterative solver only works with a preconditioner
      suited to the equations, and a poor choice will simply not converge. Choosing and configuring them
      is a subject of its own and will be covered later.

Both require PETSc to be installed and configured (cf. :numref:`petscslepc`); MUMPS in particular has to
be part of that PETSc build (``--download-mumps=yes``). The solver can be selected from the command line
(cf. :numref:`installcmdlineoptions`) or in the script with
:py:meth:`~pyoomph.generic.problem.Problem.set_linear_solver`.

