.. _installenvvars:

Environment variables
----------------------

pyoomph reads a number of environment variables at runtime (and a few more at build time, see :numref:`installcmakeoptions`). None of these are required -- pyoomph works fine without any of them set -- but they can be useful to tweak caching, threading, debugging or solver library locations.

Threading
~~~~~~~~~

``PYOOMPH_OPENBLAS_NUM_THREADS``, ``PYOOMPH_MKL_NUM_THREADS``
      Number of threads used by OpenBLAS and MKL, respectively. Both default to the ``N`` of a ``--omp N`` given on the command line, and to ``4`` without it. On import, pyoomph sets the corresponding ``OPENBLAS_NUM_THREADS``/``MKL_NUM_THREADS`` environment variables to these values, unless they are already set in the environment (e.g. by your shell or by a launcher script), in which case the existing value is left untouched. They exist so that a third-party library can be pinned for pyoomph runs alone, without also pinning it for everything else on the machine.

      They bound the threading *inside* BLAS/MKL only. They do not make a pyoomph run serial: the element loop takes its thread count from ``--omp``/:py:meth:`~pyoomph.generic.problem.Problem.set_num_threads` and passes it to its threading backend itself, so with ``--omp 4`` the assembly runs on four threads whatever these variables say. Run without ``--omp`` (or with ``--omp 1``) for a serial run.

``PYOOMPH_OMP_NUM_THREADS``
      Value pyoomph sets ``OMP_NUM_THREADS`` to on import, again only if that variable is not already set. It defaults to the ``--omp N`` given on the command line, or to ``1`` without it -- pyoomph's own threaded element loop takes its thread count from ``--omp``/:py:meth:`~pyoomph.generic.problem.Problem.set_num_threads` and not from the environment, so the pin only keeps a third-party OpenMP runtime in the same process from opening a second pool of threads. See :numref:`secopenmp`.

``PYOOMPH_ASSEMBLY_CHUNK_DOUBLES``
      Size, in doubles, of the scratch buffer the threaded element assembly (``--omp N``, see :numref:`secopenmp`) fills before gathering it into the global matrices. Defaults to ``131072``, i.e. 1 MB per matrix, and is divided by the number of matrices being assembled. Chunking is what keeps the threaded summation order identical to the serial one, so this only trades cache behaviour against barrier overhead and never changes the result. The default was measured, and everything from about 16 K upwards lies within a few percent of it -- much smaller chunks pay their two barriers for too little work.

JIT code cache
~~~~~~~~~~~~~~

pyoomph compiles the equations of a problem into native code just-in-time. To avoid recompiling identical code again and again, compiled objects are cached on disk, keyed by the content of the generated code. The following variables control this cache:

``PYOOMPH_JIT_CACHE``
      Set to ``0``, ``false``, ``False`` or an empty string to disable the JIT cache for the current process. Defaults to enabled (``1``). Note that the cache can also be permanently disabled at build time (see ``PYOOMPH_ENABLE_JIT_CACHE`` in :numref:`installcmakeoptions`); a build-time ``OFF`` cannot be re-enabled via this variable.

``PYOOMPH_JIT_CACHE_DIR``
      Directory used to store cached compiled objects. If unset, a platform-specific default cache directory is used (respecting ``XDG_CACHE_HOME`` on Linux, ``LOCALAPPDATA`` on Windows, or ``~/Library/Caches`` on Mac).

``PYOOMPH_JIT_CACHE_MAX_MB``
      Maximum size of the JIT cache in megabytes. Defaults to ``2048``.

``PYOOMPH_JIT_CACHE_MAX_FINGERPRINTS``
      Maximum number of entries kept in the cache's Tier-2 fingerprint bookkeeping. Defaults to ``100000``.

``PYOOMPH_JIT_CACHE_TIER2``
      Set to ``0``, ``false``, ``False`` or an empty string to disable the (more experimental) Tier-2 caching. Defaults to enabled (``1``), but only has any effect while ``PYOOMPH_JIT_CACHE`` is also enabled.

MPI
~~~

These are only read when running under ``mpirun`` (cf. :numref:`secmpi`).

``PYOOMPH_MPI_OUTPUT``
      Console output mode when several ranks share one terminal: ``condensed`` (the default -- only rank 0 prints, while the stderr of every rank still gets through, tagged), ``all`` (every rank, each line written in one piece and tagged ``[rank N]``) or ``off`` (no filtering at all, i.e. the raw interleaved output). The same choice is available as ``--mpi-output`` on the command line, which takes precedence; an unrecognized value here is ignored rather than raising, because this is read while ``pyoomph`` is being imported. See :numref:`secmpioutput`.

``PYOOMPH_ALLOW_SERIAL_UNDER_MPIRUN``
      Set to ``1`` to allow a pyoomph built **without** MPI support to be started by an MPI launcher with more than one rank, which is refused by default (cf. :numref:`secmpi`). The ranks are then independent serial runs which know nothing of each other, so each of them must be given its own output directory. This has no effect on a build with MPI support.

``PYOOMPH_MPI_IDLE_SPIN``
      How many polls a rank waiting for another one does before it starts sleeping between polls. Defaults to ``2000``. The spin phase is there because ``time.sleep()`` cannot resolve much better than about 60 microseconds, which would otherwise dominate the many short collectives of a small problem.

``PYOOMPH_MPI_IDLE_MAX_SLEEP``
      Longest sleep, in seconds, between two polls once a rank has given up spinning. Defaults to ``5e-3``, which leaves a waiting rank at roughly 200 wake-ups per second, i.e. no measurable CPU load, while bounding the delay with which it notices that a long factorization on another rank has finished.

``PYOOMPH_MPI_IDLE_WARN_AFTER``
      Seconds a rank may wait before pyoomph prints a one-off notice naming what it is waiting for. Defaults to ``600``. It is deliberately a notice and not a timeout: raising on the one rank that noticed would create exactly the deadlock it is trying to report.

Compilation and debugging
~~~~~~~~~~~~~~~~~~~~~~~~~~

``PYOOMPH_DEBUG``
      Set to ``1`` to compile the just-in-time generated code with debug information/flags instead of the default optimized build. Useful when debugging a crash inside generated code with e.g. ``gdb``.

``PYOOMPH_FULL_UNIT_ERROR``
      Set to ``1`` to print the offending expression in full when a residual contribution does not come out dimensionless. By default such an expression is truncated, since a single Navier-Stokes contribution prints as tens of thousands of characters and buries the part of the message that says what is actually wrong.

Plotting
~~~~~~~~

``PYOOMPH_MPLBACKEND``
      Overrides the matplotlib backend used by pyoomph's plotting utilities (e.g. ``Agg``, ``TkAgg``, ``Qt5Agg``). If unset, ``Agg`` is used, since these utilities render to files rather than to a window. This does not affect the bifurcation GUI, which embeds its figure in its own tkinter window and therefore selects no backend at all.

Alternative solver libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, pyoomph loads the MKL/Pardiso solver library from its usual system location. The following variable lets you point at a specific shared library instead, e.g. a custom build or a non-standard install location:

``PYOOMPH_PARDISO_LIB``
      Full path to the shared library providing the (MKL) Pardiso solver.

Third-party variables
~~~~~~~~~~~~~~~~~~~~~~

A few environment variables belonging to third-party dependencies are also relevant when working with pyoomph, most notably ``PETSC_DIR``, ``PETSC_ARCH`` and ``PYTHONPATH`` for selecting a PETSc/SLEPc installation, see :numref:`petscslepc` for details.
