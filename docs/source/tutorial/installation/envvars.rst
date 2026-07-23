.. _installenvvars:

Environment variables
----------------------

pyoomph reads a number of environment variables at runtime (and a few more at build time, see :numref:`installcmakeoptions`). None of these are required -- pyoomph works fine without any of them set -- but they can be useful to tweak caching, threading, debugging or solver library locations.

Threading
~~~~~~~~~

``PYOOMPH_OPENBLAS_NUM_THREADS``, ``PYOOMPH_MKL_NUM_THREADS``
      Number of threads used by OpenBLAS and MKL, respectively. Both default to ``4``. On import, pyoomph sets the corresponding ``OPENBLAS_NUM_THREADS``/``MKL_NUM_THREADS`` environment variables to these values, unless they are already set in the environment (e.g. by your shell or by a launcher script), in which case the existing value is left untouched. To fully disable OpenMP-style parallelization, set both to ``1``.

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

Compilation and debugging
~~~~~~~~~~~~~~~~~~~~~~~~~~

``PYOOMPH_DEBUG``
      Set to ``1`` to compile the just-in-time generated code with debug information/flags instead of the default optimized build. Useful when debugging a crash inside generated code with e.g. ``gdb``.

Plotting
~~~~~~~~

``PYOOMPH_MPLBACKEND``
      Overrides the matplotlib backend used by pyoomph's plotting utilities and the bifurcation GUI (e.g. ``Agg``, ``TkAgg``, ``Qt5Agg``). If unset, matplotlib's own auto-detected backend is used.

Alternative solver libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, pyoomph loads the MKL/Pardiso solver library from its usual system location. The following variable lets you point at a specific shared library instead, e.g. a custom build or a non-standard install location:

``PYOOMPH_PARDISO_LIB``
      Full path to the shared library providing the (MKL) Pardiso solver.

Third-party variables
~~~~~~~~~~~~~~~~~~~~~~

A few environment variables belonging to third-party dependencies are also relevant when working with pyoomph, most notably ``PETSC_DIR``, ``PETSC_ARCH`` and ``PYTHONPATH`` for selecting a PETSc/SLEPc installation, see :numref:`petscslepc` for details.
