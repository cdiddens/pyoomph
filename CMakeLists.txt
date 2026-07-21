
cmake_minimum_required(VERSION 3.22)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# Project version is normally supplied by scikit-build-core (via
# setuptools-scm, see pyproject.toml) as SKBUILD_PROJECT_VERSION. Fall back
# to a placeholder for plain, non-Python CMake configuration/testing.
if(NOT DEFINED SKBUILD_PROJECT_VERSION)
  set(SKBUILD_PROJECT_VERSION "0.0.0")
endif()

project(pyoomph LANGUAGES C CXX VERSION ${SKBUILD_PROJECT_VERSION})

# On Apple Silicon, detect whether this CMake configure is itself running
# under Rosetta 2 translation (an x86_64 process on arm64 hardware) - e.g.
# when following the documented "Rosetta 2 terminal" route to get the MKL
# Pardiso solver, which has no arm64 build. If so, default to a matching
# single-arch x86_64 build rather than letting the compiler pick its native
# default (arm64). Without this, CLN/GiNaC's autotools ExternalProject_Add
# subbuilds (see cmake/ThirdPartyGiNaC.cmake, which reads
# CMAKE_OSX_ARCHITECTURES back out to keep their -arch flags in sync) can
# end up targeting a different architecture than the rest of the build,
# which surfaces as clang being invoked for both -arch x86_64 and -arch
# arm64 at once and failing with "redefinition of '_OSSwapInt32'" (the SDK's
# per-architecture endian headers collide). Only takes effect if the caller
# hasn't already set CMAKE_OSX_ARCHITECTURES explicitly.
if(APPLE AND NOT CMAKE_OSX_ARCHITECTURES)
  execute_process(
    COMMAND sysctl -n sysctl.proc_translated
    OUTPUT_VARIABLE PYOOMPH_MACOS_PROC_TRANSLATED
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(PYOOMPH_MACOS_PROC_TRANSLATED STREQUAL "1")
    message(STATUS "pyoomph: detected Rosetta 2 translation - forcing a single-arch x86_64 build (pass -DCMAKE_OSX_ARCHITECTURES=... to override)")
    set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE STRING "Build architecture(s) for macOS" FORCE)
  endif()
endif()

# ----------------------------------------------------------------------------
# Build options
#
# These mirror the old PYOOMPH_* environment variables read by setup.py, and
# are what the `configure` wrapper script (see ./configure) sets via
# `-DPYOOMPH_...=...` / `pip install --config-settings=cmake.define.X=Y`.
# ----------------------------------------------------------------------------
option(PYOOMPH_USE_MPI       "Build with MPI support (defines OOMPH_HAS_MPI)"        OFF)
option(PYOOMPH_PARANOID      "Build with extra runtime sanity checks (defines PARANOID)" OFF)
option(PYOOMPH_DOWNLOAD_CLN  "Download and build CLN from source via its own autotools ./configure" ON)
option(PYOOMPH_DOWNLOAD_GINAC "Download and build GiNaC from source via its own autotools ./configure" ON)
option(PYOOMPH_GENERATE_STUBS "Generate .pyi stubs for the _core extension via nanobind.stubgen (best effort, non-fatal)"  ON)
option(PYOOMPH_COPY_STUBS_TO_SOURCE_TREE "Also mirror the generated .pyi stub into the source-tree pyoomph/ directory, so Pylance/Pyright/mypy can resolve pyoomph._core while editing without a full pip install" ON)

# Left empty by default: ginac.de only ever hosts the *current* release
# tarball, so a version pinned here goes dead as soon as a new release is
# published. When empty, cmake/ThirdPartyGiNaC.cmake auto-detects the
# current version by scraping ginac.de's own download pages at configure
# time. Set explicitly to pin a specific (still-live) version instead.
set(PYOOMPH_CLN_VERSION "" CACHE STRING "CLN version to download when PYOOMPH_DOWNLOAD_CLN=ON (auto-detected from ginac.de if empty)")
set(PYOOMPH_GINAC_VERSION "" CACHE STRING "GiNaC version to download when PYOOMPH_DOWNLOAD_GINAC=ON (auto-detected from ginac.de if empty)")
set(PYOOMPH_THIRDPARTY_PREFIX "${CMAKE_BINARY_DIR}/thirdparty-install" CACHE PATH
    "Install prefix used for a downloaded/built CLN and/or GiNaC")

# Optional hints for a pre-existing, system-installed CLN/GiNaC (used only
# when PYOOMPH_DOWNLOAD_CLN / PYOOMPH_DOWNLOAD_GINAC are OFF). These mirror
# the old PYOOMPH_GINAC_INCLUDE_DIR / PYOOMPH_GINAC_LIB_DIR / ... env vars.
set(PYOOMPH_GINAC_INCLUDE_DIR "" CACHE PATH "Hint: directory containing ginac/ginac.h")
set(PYOOMPH_GINAC_LIB_DIR     "" CACHE PATH "Hint: directory containing libginac")
set(PYOOMPH_CLN_INCLUDE_DIR   "" CACHE PATH "Hint: directory containing cln/cln.h")
set(PYOOMPH_CLN_LIB_DIR       "" CACHE PATH "Hint: directory containing libcln")

if(PYOOMPH_USE_MPI)
  set(PYOOMPH_NO_MPI_DEFAULT OFF)
else()
  set(PYOOMPH_NO_MPI_DEFAULT ON)
endif()

set(_pyoomph_no_mpi_marker "${CMAKE_SOURCE_DIR}/pyoomph/NO_MPI")
if(PYOOMPH_USE_MPI)
  file(REMOVE "${_pyoomph_no_mpi_marker}")
else()
  file(TOUCH "${_pyoomph_no_mpi_marker}")
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

include(cmake/ThirdPartyGiNaC.cmake)

# ----------------------------------------------------------------------------
# nanobind, via CMake.
#
# NOTE: nanobind itself (headers/CMake package) is looked up here, but is
# only actually *used* by the _pyoomph target further down, whose sources
# are restricted to src/nanobind/*.cpp. Everything else in this project
# (libgeneric.a and the pyoomph_core static library) is fully Python-version
# independent and never sees nanobind or the Python headers/flags.
# ----------------------------------------------------------------------------
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT
  COMMAND_ERROR_IS_FATAL ANY)

find_package(nanobind CONFIG REQUIRED)

# ----------------------------------------------------------------------------
# libgeneric.a: all .cc files from oomph-lib's include/ directory. This must
# be built first; pyoomph_core and _pyoomph both depend on it, which
# target_link_libraries() below expresses (CMake/the generator then takes
# care of the ordering automatically, no manual sequencing needed).
#
# Python-independent: built once, reused unchanged across every Python
# version when iterating with `./configure && make` for multiple wheels.
# ----------------------------------------------------------------------------
file(GLOB_RECURSE OOMPH_GENERIC_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_SOURCE_DIR}/src/thirdparty/oomph-lib/include/*.cc")

if(NOT OOMPH_GENERIC_SOURCES)
  message(FATAL_ERROR "No .cc sources found under src/thirdparty/oomph-lib/include - "
                       "is the oomph-lib submodule/sources checked out?")
endif()

# These are template-only translation units (meant to be #included where
# needed, not compiled standalone as part of libgeneric.a) and must be
# excluded from the glob above.
set(OOMPH_GENERIC_EXCLUDED_SOURCES
  "tree.template.cc"
  "multi_domain.template.cc"
  "refineable_mesh.template.cc"
)
foreach(_excluded_name IN LISTS OOMPH_GENERIC_EXCLUDED_SOURCES)
  list(FILTER OOMPH_GENERIC_SOURCES EXCLUDE REGEX "/${_excluded_name}$")
endforeach()

add_library(generic STATIC ${OOMPH_GENERIC_SOURCES})
target_include_directories(generic PUBLIC
  "${CMAKE_SOURCE_DIR}/src/thirdparty/oomph-lib/include"
)
target_compile_definitions(generic PUBLIC
  $<$<BOOL:${PYOOMPH_USE_MPI}>:OOMPH_HAS_MPI>
  $<$<BOOL:${PYOOMPH_PARANOID}>:PARANOID>
)

# ----------------------------------------------------------------------------
# pyoomph_core: all of src/*.cpp EXCEPT src/nanobind/*.cpp. This is the bulk of
# pyoomph's own C++ code, and it does not depend on nanobind or the Python
# headers at all - it only needs to be recompiled when the sources
# themselves change, not when switching Python versions.
# ----------------------------------------------------------------------------
file(GLOB PYOOMPH_CORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_library(pyoomph_core STATIC ${PYOOMPH_CORE_SOURCES})

target_include_directories(pyoomph_core PUBLIC
  "${CMAKE_SOURCE_DIR}/src"
  "${CMAKE_SOURCE_DIR}/src/thirdparty"
  "${PYOOMPH_GINAC_INCLUDE_DIR_RESOLVED}"
  "${PYOOMPH_CLN_INCLUDE_DIR_RESOLVED}"
)

target_link_libraries(pyoomph_core PUBLIC generic)

target_compile_definitions(pyoomph_core PUBLIC
  $<$<BOOL:${PYOOMPH_USE_MPI}>:OOMPH_HAS_MPI>
  $<$<BOOL:${PYOOMPH_PARANOID}>:PARANOID>
)

if(PYOOMPH_DOWNLOAD_GINAC)
  add_dependencies(pyoomph_core ginac_external)
endif()
if(PYOOMPH_DOWNLOAD_CLN)
  add_dependencies(pyoomph_core cln_external)
endif()

if(PYOOMPH_USE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_link_libraries(generic PUBLIC MPI::MPI_CXX)
endif()

# ----------------------------------------------------------------------------
# The actual Python extension module: ONLY src/nanobind/*.cpp. This is the
# only target that pulls in nanobind/Python compile flags and include
# directories, and hence the only target that needs rebuilding when
# iterating over multiple Python versions - pyoomph_core and generic above
# are reused as-is.
#
# Built/installed as the "_core" submodule of the "pyoomph" package, i.e.
# importable as `pyoomph._core` (see NB_MODULE(_core, ...) in
# src/nanobind - the module name there must match this target name).
# ----------------------------------------------------------------------------
file(GLOB PYOOMPH_NANOBIND_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/nanobind/*.cpp")

if(NOT PYOOMPH_NANOBIND_SOURCES)
  message(FATAL_ERROR "No .cpp sources found under src/nanobind - "
                       "cannot build the Python extension module.")
endif()

nanobind_add_module(_pyoomph_core
  ${PYOOMPH_NANOBIND_SOURCES}
)

target_link_libraries(_pyoomph_core PRIVATE
  pyoomph_core
  ${PYOOMPH_GINAC_LIBRARY}
  ${PYOOMPH_CLN_LIBRARY}
)

if(NOT WIN32)
  target_link_libraries(_pyoomph_core PRIVATE dl)
endif()

if(APPLE)
  # Needed by src/mac_accelerate.cpp (Apple's Accelerate "Sparse Solvers" framework), which is
  # part of pyoomph_core's sources but only compiles to non-empty code on macOS (see the
  # __APPLE__ guard in src/mac_accelerate.hpp/.cpp).
  find_library(PYOOMPH_ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  target_link_libraries(pyoomph_core PUBLIC "${PYOOMPH_ACCELERATE_FRAMEWORK}")
endif()

target_compile_definitions(_pyoomph_core PRIVATE
  $<$<BOOL:${PYOOMPH_USE_MPI}>:OOMPH_HAS_MPI>
  $<$<BOOL:${PYOOMPH_PARANOID}>:PARANOID>
)

if(PYOOMPH_USE_MPI)
  target_link_libraries(_pyoomph_core PRIVATE MPI::MPI_CXX)
endif()


 
if(PYOOMPH_GENERATE_STUBS)
  set(PYOOMPH_STUB_STAGE_DIR "${CMAKE_BINARY_DIR}/pyi-stage")
 
  set(PYOOMPH_STUB_EXTRA_COPY_ARGS "")
  if(PYOOMPH_COPY_STUBS_TO_SOURCE_TREE)
    list(APPEND PYOOMPH_STUB_EXTRA_COPY_ARGS
         --extra-copy-dir "${CMAKE_SOURCE_DIR}/pyoomph")
  endif()

  add_custom_command(TARGET _pyoomph_core POST_BUILD
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/cmake/generate_stubs.py"
            --module-dir "$<TARGET_FILE_DIR:_pyoomph_core>"
            --module-name "_pyoomph_core"
            --stage-dir "${PYOOMPH_STUB_STAGE_DIR}"
            --patch-script "${CMAKE_SOURCE_DIR}/src/nanobind/patch_stubs.py"
            --python "${Python_EXECUTABLE}"
            ${PYOOMPH_STUB_EXTRA_COPY_ARGS}
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Generating .pyi stubs for pyoomph._pyoomph_core (best effort)"
    VERBATIM
  )
 
  # Whatever generate_stubs.py normalizes into PYOOMPH_STUB_STAGE_DIR (either
  # "_core.pyi" or a "_core/" package directory) is copied straight next to
  # the built extension. OPTIONAL means a missing stage dir (stub generation
  # skipped/failed) is silently ignored rather than breaking `make install`.
  install(DIRECTORY "${PYOOMPH_STUB_STAGE_DIR}/" DESTINATION pyoomph OPTIONAL)
endif()


# ----------------------------------------------------------------------------
# Install: _core lands inside the pyoomph/ package directory, so it is
# importable as `pyoomph._core` rather than as a top-level module.
# ----------------------------------------------------------------------------
install(TARGETS _pyoomph_core LIBRARY DESTINATION pyoomph)

# Ensure wheel installs carry the NO_MPI marker whenever MPI support is
# intentionally disabled so runtime imports skip mpi4py.
if(NOT PYOOMPH_USE_MPI)
  install(FILES "${_pyoomph_no_mpi_marker}" DESTINATION pyoomph)
endif()
