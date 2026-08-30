.. _installcmakeoptions:

CMake build options
--------------------

Building pyoomph from source (see :numref:`installcompile`) is driven by CMake. The build options below are declared in the top-level ``CMakeLists.txt`` and can be overridden either by passing ``-DPYOOMPH_...=...`` directly to ``cmake``, or, when building via ``pip``, through ``--config-settings=cmake.define.PYOOMPH_...=...``, e.g.

.. code:: bash

      python -m pip install --no-build-isolation -e . --config-settings=cmake.define.PYOOMPH_USE_MPI=ON

Feature switches
~~~~~~~~~~~~~~~~

``PYOOMPH_USE_MPI`` (default ``OFF``)
      Build with MPI support (defines ``OOMPH_HAS_MPI``).

``PYOOMPH_PARANOID`` (default ``OFF``)
      Build with extra runtime sanity checks (defines ``PARANOID``).

``PYOOMPH_USE_OPENMP`` (default ``AUTO``)
      OpenMP backend for the threaded element assembly (``--omp N``, see :numref:`installcmdlineoptions`). ``AUTO`` links OpenMP if the toolchain provides it (Linux/Windows always do); ``ON`` makes a missing OpenMP a configure error, ``OFF`` never looks for it. On macOS this is **off by default** -- macOS threads through GCD instead (next option), because an OpenMP runtime there would clash with the one MKL loads; pass ``ON`` to force the OpenMP path on macOS anyway (it then wins over GCD and needs a ``libomp``, e.g. via ``citools/build_static_libomp.sh``).

``PYOOMPH_USE_GCD`` (macOS only, default ``AUTO``)
      GCD/libdispatch backend for the threaded assembly -- the macOS default. It links no OpenMP runtime at all, so it cannot collide with MKL's. ``AUTO`` uses it unless ``PYOOMPH_USE_OPENMP=ON`` is given, ``ON`` forces it, ``OFF`` falls back to OpenMP. Ignored off macOS.

``PYOOMPH_ENABLE_JIT_CACHE`` (default ``ON``)
      Build-time kill switch for the JIT code cache (see :numref:`installenvvars`). If set to ``OFF``, the cache is permanently disabled for this build, regardless of any runtime flag or environment variable (``--no-cache``, ``PYOOMPH_JIT_CACHE``, ...) -- those can only ever narrow an ``ON`` down to ``OFF``, never override an ``OFF`` set here.

``PYOOMPH_HAS_TQMESH`` (default ``ON``)
      Build with the `TQMesh <https://github.com/FloSewn/TQMesh>`__ two-dimensional triangle/quad mesh generator. With ``OFF``, TQMesh is not downloaded and its sources are dropped from the build: the extension module then reports ``has_tqmesh == False`` and none of the ``TQMesh*`` classes exist.

``PYOOMPH_HAS_SPECTRA`` (default ``ON``)
      Build with the built-in `Spectra <https://spectralib.org>`__/`Eigen <https://eigen.tuxfamily.org>`__ serial eigensolver (the ``--spectra`` backend, see :numref:`installcmdlineoptions`). It is the only eigensolver that can target a given eigenvalue without PETSc/SLEPc, which is what makes it the fallback on Windows. With ``OFF``, neither library is downloaded, ``has_spectra == False``, ``pyoomph.solvers.spectra`` refuses to import, and the eigensolver autodetection falls back to the ARPACK-based backends, which cannot target an eigenvalue at all.

``PYOOMPH_GENERATE_STUBS`` (default ``ON``)
      Generate ``.pyi`` type stubs for the compiled ``_core`` extension via nanobind's ``stubgen`` (best effort, failures are non-fatal).

``PYOOMPH_COPY_STUBS_TO_SOURCE_TREE`` (default ``ON``)
      Also mirror the generated ``.pyi`` stub into the source-tree ``pyoomph/`` directory, so Pylance/Pyright/mypy can resolve ``pyoomph._core`` while editing without a full ``pip install``.

CLN/GiNaC dependencies
~~~~~~~~~~~~~~~~~~~~~~

pyoomph depends on `CLN <https://www.ginac.de/CLN/>`__ and `GiNaC <https://www.ginac.de/>`__. By default, both are downloaded and built automatically as part of the CMake configuration; the following options control this.

``PYOOMPH_DOWNLOAD_CLN`` (default ``ON``)
      Download and build CLN from source via its own autotools ``./configure``.

``PYOOMPH_DOWNLOAD_GINAC`` (default ``ON``)
      Download and build GiNaC from source via its own autotools ``./configure``.

``PYOOMPH_CLN_VERSION`` / ``PYOOMPH_GINAC_VERSION`` (default: empty)
      Pin a specific CLN/GiNaC version to download. Left empty by default, in which case the current version is auto-detected by scraping ginac.de's download pages at configure time.

``PYOOMPH_THIRDPARTY_PREFIX`` (default ``${CMAKE_BINARY_DIR}/thirdparty-install``)
      Install prefix used for a downloaded/built CLN and/or GiNaC.

``CLN_EXTRA_CONFIGURE_FLAGS`` / ``GINAC_EXTRA_CONFIGURE_FLAGS`` (default: empty)
      Extra flags passed through to CLN's/GiNaC's own autotools ``./configure``.

``PYOOMPH_ASSUME_GINAC_HASH_PATCHED`` (default ``OFF``)
      Only meaningful together with ``PYOOMPH_DOWNLOAD_GINAC=OFF``: asserts that a system-supplied GiNaC has been patched by other means to make its term/hash ordering deterministic across process runs. Leave this ``OFF`` unless you have actually verified this yourself -- a wrong ``ON`` here would silently defeat the JIT code cache's safety check, since it would assume that the generated code is reproducible when it might not be.

If you use a system-installed CLN/GiNaC instead (``PYOOMPH_DOWNLOAD_CLN``/``PYOOMPH_DOWNLOAD_GINAC`` set to ``OFF``), the following hints tell CMake where to find them:

``PYOOMPH_GINAC_INCLUDE_DIR`` / ``PYOOMPH_GINAC_LIB_DIR``
      Directory containing ``ginac/ginac.h`` / the directory containing ``libginac``.

``PYOOMPH_CLN_INCLUDE_DIR`` / ``PYOOMPH_CLN_LIB_DIR``
      Directory containing ``cln/cln.h`` / the directory containing ``libcln``.

Downloaded TQMesh/Spectra/Eigen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TQMesh, Spectra and Eigen are header-only and, unlike CLN/GiNaC, are always downloaded rather than taken from the system. Each of the three has the same trio of options, which is what allows an offline build from an already available source tree:

``PYOOMPH_TQMESH_REF`` / ``PYOOMPH_SPECTRA_REF`` / ``PYOOMPH_EIGEN_REF``
      Git ref (commit, tag or branch) to download. Defaults to a pinned commit for TQMesh and Spectra and to ``3.4.0`` for Eigen.

``PYOOMPH_TQMESH_URL`` / ``PYOOMPH_SPECTRA_URL`` / ``PYOOMPH_EIGEN_URL`` (default: empty)
      Full URL of a source archive to download, overriding the corresponding ``..._REF``.

``PYOOMPH_TQMESH_SOURCE_DIR`` / ``PYOOMPH_SPECTRA_SOURCE_DIR`` / ``PYOOMPH_EIGEN_SOURCE_DIR`` (default: empty)
      An already available source tree to use instead of downloading anything. It is copied into the build directory and left untouched, since the sources are patched afterwards.

Other standard CMake settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Besides the ``PYOOMPH_*`` options above, a few standard CMake variables are also relevant:

``CMAKE_BUILD_TYPE``
      Defaults to ``Release`` if not set explicitly (and no multi-config generator is in use). Set to e.g. ``RelWithDebInfo`` or ``Debug`` for a debuggable build.

``CMAKE_OSX_ARCHITECTURES`` (Mac only)
      Target architecture(s) for macOS. If left unset on Apple Silicon, pyoomph auto-detects whether the current CMake configure run is itself running under Rosetta 2 translation and, if so, forces a single-arch ``x86_64`` build to keep CLN/GiNaC's sub-builds consistent with the rest of the build. Set this explicitly to override that detection.
