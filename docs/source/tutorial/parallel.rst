.. _secparallel:

Parallelization
===============

Pyoomph can spread the work of a simulation over several cores in two rather different ways, and both ways in principle
compose: *OpenMP* threads inside one process, and *MPI* processes that may additionally hold only
their own part of the mesh. Neither is on by default, i.e. ``python3 my_simulation.py`` assembles
on a single thread in a single process. Note that, depending on the choice, the linear solver still might be *OpenMP* parallel.


*   **OpenMP Threads (**\ ``--omp N``\ **)** are the cheap option. They only parallelize the
    element assembly (and applies the same number of threads on the linear solver, if supported).
*   **MPI processes (**\ ``mpirun``\ **)** parallelize the assembly, the linear solve, and with ``--distribute``
    the storage of the mesh too.
    MPI has to be compiled in, that every process must agree with the others on the state of the simulation.

.. note::

	All pyoomph tutorial scripts have been tested against both OpenMP and MPI. However, in particular MPI is still under development.


.. toctree::
   :maxdepth: 5
   :hidden:

   parallel/openmp.rst
   parallel/mpi.rst
