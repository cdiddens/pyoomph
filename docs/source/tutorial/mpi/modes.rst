.. _secmpimodes:

Requirements and the parallel modes
-----------------------------------

Requirements
~~~~~~~~~~~~

MPI support is a **compile-time** option. The CMake switch ``PYOOMPH_USE_MPI`` defaults to ``OFF``
(cf. :numref:`installcmakeoptions`), and the precompiled wheels on PyPI are built with that default, so
a ``pip install pyoomph`` cannot be run in parallel no matter how it is launched. To use MPI you have to
build pyoomph from source (cf. :numref:`installcompile`) with the switch turned on, e.g.

.. code:: bash

   python -m pip install --no-build-isolation -e . --config-settings=cmake.define.PYOOMPH_USE_MPI=ON

On the Python side, ``mpi4py`` must be importable. Whether the pyoomph you are actually running has MPI
is best asked directly rather than inferred from how it was installed:

.. code:: python

   from pyoomph.generic.mpi import has_mpi
   print(has_mpi())

A build without MPI is not broken, it is simply serial: it reports ``has_mpi() == False``, and the
process count it reports is ``0`` rather than ``1``, which is how pyoomph tells "built without MPI" from
"built with MPI but started as a single process".

Distributing the mesh (the third mode below) additionally requires `PyMetis <https://pypi.org/project/PyMetis/>`__
for the graph partitioning::

   python -m pip install pymetis

A parallel run is started with ``mpirun`` (or ``mpiexec``, or whatever your MPI installation and queueing
system provide) as any other MPI program:

.. code:: bash

   mpirun -n 4 python3 my_simulation.py

The number of processes is not something the script chooses; it comes from the launcher. The same script
runs unchanged on one process and on many.

The three modes
~~~~~~~~~~~~~~~

The two ways of running in parallel differ in *what* is shared between the processes, which is far more
consequential than it may first sound. They are shown, together with a serial run for reference, in
:numref:`figmpimodes`.

..  figure:: parallel_modes.*
	:name: figmpimodes
	:align: center
	:alt: The three ways of running a pyoomph simulation
	:class: with-shadow
	:width: 100%

	The same 8x8 quadrilateral mesh in the three modes. (left) Serial: one process owns everything.
	(centre) ``mpirun`` without ``--distribute``: every process still holds the entire mesh, but the
	element loop of the assembly is split between them - the colours show which process integrates
	which elements. (right) ``mpirun --distribute``: the mesh itself is partitioned, and each process
	only ever holds its own part (plus a thin halo of neighbouring elements, not drawn). METIS
	partitions to minimise the number of cut element faces, which is why the parts come out as
	compact blocks rather than as strips: the four blocks drawn here share 16 faces, four strips of
	the same size would share 24.

**Serial, without** ``mpirun``. One process holds the mesh, assembles the residual and Jacobian and
solves the linear system. This is not necessarily single-threaded: most direct solvers are internally
threaded, so a serial pyoomph run can still keep several cores busy inside the factorization. The thread
count is set with :py:meth:`~pyoomph.generic.problem.Problem.set_num_threads`. For small and medium
problems this is very often the fastest option, and it is always the simplest to reason about.

**With** ``mpirun``, **without** ``--distribute``. Every process holds the *whole* mesh and the *whole*
vector of unknowns. What is parallel is the assembly - oomph-lib splits the element loop between the
processes as soon as more than one is present - and the linear solve, for which the assembled Jacobian is
laid out across the processes by rows. Everything else is *replicated*: each process computes the same
numbers, and it must, since they all keep their own copy of the state.

This mode is easy to adopt, because nothing about the problem setup has to change, and it helps most for
problems whose cost is dominated by the assembly - typically strongly nonlinear equations whose residuals
are expensive to evaluate. It does not reduce the memory per process at all: every process stores the
whole mesh, and the assembled matrix rows on top of it.

.. warning::

   Because the state is replicated, anything that lets the processes disagree breaks the run, and it
   rarely breaks where the mistake is. A random initial condition drawn per process, a refinement
   criterion evaluated from something that differs in the last bit, a value read from a per-process
   file: from that point on the processes are solving different problems and stitching one Jacobian out
   of them. Use :py:class:`~pyoomph.expressions.utils.DeterministicRandomField` instead of drawing
   randomly, and keep every decision a pure function of the replicated state. Pyoomph checks the global
   system size across processes and refuses to continue when they disagree, which catches the common
   cases early with an explicit message.

**With** ``mpirun --distribute``. The mesh is partitioned with METIS and each process keeps only its own
part, together with a halo of the neighbouring elements it needs to assemble its own. Assembly, storage
and the linear solve are all distributed, so this is the mode that lets a problem exceed the memory of a
single machine, and the one that actually scales to larger process counts. The price is that a
distributed run is a genuinely different object: a process no longer knows the whole solution, so
anything that wants global mesh data has to ask for it collectively, and a number of features are
restricted or unavailable on a distributed problem.

Choosing the linear solver
~~~~~~~~~~~~~~~~~~~~~~~~~~

Splitting the assembly is only half of the work, and usually the smaller half: for anything but the most
expensive constitutive laws, the linear solve dominates. Whether that part is parallel at all is decided
entirely by the linear solver, and most of the solvers pyoomph can use are not MPI-parallel.

The serial backends - ``pardiso``, ``superlu``, ``umfpack`` and Apple's ``accelerate`` - are not simply
refused under ``mpirun``. Pyoomph gathers the assembled system onto process 0, solves it there, and
scatters the result back, while the other processes wait. That fallback is deliberate and it is honest
about itself: it prints a one-off note saying which solver it applies to, that the assembly stays
parallel, and that the solve does not scale. It is a real speed-up for an assembly-dominated problem,
but process 0 needs the entire matrix in memory, so it defeats the main purpose of ``--distribute``.

.. note::

   The waiting processes sleep rather than spin, so that process 0 may use their cores for its own
   threads. Whether it actually can is up to the launcher: most MPI implementations pin each process to
   a core by default, which stops process 0 from spreading out. Run ``mpirun --bind-to none`` and set
   the thread count with :py:meth:`~pyoomph.generic.problem.Problem.set_num_threads` if you want it to.
   Pyoomph prints a note when it detects this situation.

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
      is a subject of its own and is covered later.

Both require PETSc to be installed and configured (cf. :numref:`petscslepc`); MUMPS in particular has to
be part of that PETSc build (``--download-mumps=yes``). The solver can be selected from the command line
(cf. :numref:`installcmdlineoptions`) or in the script with
:py:meth:`~pyoomph.generic.problem.Problem.set_linear_solver`.

.. warning::

   Do not force one linear solver on a set of scripts merely to make a comparison uniform. The solver is
   part of what is being computed: augmented systems, as they arise in bifurcation tracking and
   arclength continuation, are not solved equally well - or at all - by every backend. Compare a solver
   choice per problem, not across problems.
