# Acquires Eigen (https://eigen.tuxfamily.org) and Spectra (https://spectralib.org), which together
# provide the serial eigensolver backend exposed to python by src/nanobind/spectra/ as the eigensolver
# named "spectra". Its reason for existing is Windows, where PETSc/SLEPc is not available at all.
#
# Both are header-only, so - as for TQMesh, see ThirdPartyTQMesh.cmake - there is nothing to
# configure, build or install here: this only obtains the sources and hands the include directories
# back. Unlike TQMesh there is no patch step, which is why a source tree supplied by the user can be
# used in place directly instead of being copied.
#
# Sets, for use by the top-level CMakeLists.txt:
#   PYOOMPH_SPECTRA_INCLUDE_DIRS   the include directories, to be attached to the spectra sources ONLY
#   PYOOMPH_SPECTRA_DEPENDS        the ExternalProject targets the extension module must depend on
#                                  (empty when both trees were supplied rather than downloaded)

include(ExternalProject)

# NOT the v1.0.0 release. Spectra 1.0.0 cannot solve complex eigenproblems at all: its GenEigsBase
# hardcodes "using Complex = std::complex<Scalar>", i.e. Scalar is assumed real, so the template does
# not even instantiate for std::complex<double>. Complex support (a RealScalar/Scalar split and a
# RestartArnoldi specialisation for complex scalars) landed on master afterwards and has never been
# released. pyoomph needs it for the azimuthal and normal-mode stability problems, whose matrix pair
# is genuinely complex, so master it is. The website documentation describes master too, which is why
# it appears to promise complex support that the release tarball does not have.
set(PYOOMPH_SPECTRA_REF "db1d5cc3279752ca7ea3e33da44ba2a85e4e4a95" CACHE STRING
    "Git ref (commit, tag or branch) of Spectra to download")
set(PYOOMPH_SPECTRA_URL "" CACHE STRING
    "Full URL of a Spectra source archive to download, overriding PYOOMPH_SPECTRA_REF")
set(PYOOMPH_SPECTRA_SOURCE_DIR "" CACHE PATH
    "An already available Spectra source tree to use instead of downloading one. It is used in place and left untouched.")

set(PYOOMPH_EIGEN_REF "3.4.0" CACHE STRING
    "Git ref (commit, tag or branch) of Eigen to download")
set(PYOOMPH_EIGEN_URL "" CACHE STRING
    "Full URL of an Eigen source archive to download, overriding PYOOMPH_EIGEN_REF")
set(PYOOMPH_EIGEN_SOURCE_DIR "" CACHE PATH
    "An already available Eigen source tree (the directory containing Eigen/Dense, e.g. /usr/include/eigen3) to use instead of downloading one. It is used in place and left untouched.")

set(PYOOMPH_SPECTRA_INCLUDE_DIRS "")
set(PYOOMPH_SPECTRA_DEPENDS "")

# --- Eigen ------------------------------------------------------------------------------------
# Note that Eigen's include root is the source root itself (#include <Eigen/Dense>), whereas
# Spectra's is its include/ subdirectory (#include <Spectra/GenEigsSolver.h>).
if(PYOOMPH_EIGEN_SOURCE_DIR)
  if(NOT EXISTS "${PYOOMPH_EIGEN_SOURCE_DIR}/Eigen/Dense")
    message(FATAL_ERROR
      "PYOOMPH_EIGEN_SOURCE_DIR=${PYOOMPH_EIGEN_SOURCE_DIR} does not contain Eigen/Dense, "
      "so it is not an Eigen include root. A distribution's Eigen usually lives in /usr/include/eigen3.")
  endif()
  set(_pyoomph_eigen_inc "${PYOOMPH_EIGEN_SOURCE_DIR}")
  message(STATUS "pyoomph: using the Eigen headers in ${PYOOMPH_EIGEN_SOURCE_DIR}")
else()
  if(PYOOMPH_EIGEN_URL)
    set(_pyoomph_eigen_url "${PYOOMPH_EIGEN_URL}")
  else()
    set(_pyoomph_eigen_url "https://gitlab.com/libeigen/eigen/-/archive/${PYOOMPH_EIGEN_REF}/eigen-${PYOOMPH_EIGEN_REF}.tar.gz")
  endif()
  # No URL_HASH, and no reachability pre-check, for the reasons spelled out in ThirdPartyTQMesh.cmake:
  # the forges regenerate these archives on demand, so the ref is what pins the sources.
  set(_pyoomph_eigen_inc "${CMAKE_BINARY_DIR}/eigen/source")
  ExternalProject_Add(eigen_external
    URL "${_pyoomph_eigen_url}"
    PREFIX "${CMAKE_BINARY_DIR}/eigen"
    SOURCE_DIR "${_pyoomph_eigen_inc}"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS "${_pyoomph_eigen_inc}/Eigen/Dense"
  )
  list(APPEND PYOOMPH_SPECTRA_DEPENDS eigen_external)
  message(STATUS "pyoomph: using Eigen ${PYOOMPH_EIGEN_REF}")
endif()

# --- Spectra ----------------------------------------------------------------------------------
if(PYOOMPH_SPECTRA_SOURCE_DIR)
  if(NOT EXISTS "${PYOOMPH_SPECTRA_SOURCE_DIR}/include/Spectra/GenEigsSolver.h")
    message(FATAL_ERROR
      "PYOOMPH_SPECTRA_SOURCE_DIR=${PYOOMPH_SPECTRA_SOURCE_DIR} does not contain "
      "include/Spectra/GenEigsSolver.h, so it is not a Spectra source tree.")
  endif()
  if(NOT EXISTS "${PYOOMPH_SPECTRA_SOURCE_DIR}/include/Spectra/HermEigsSolver.h")
    # Cheapest available discriminator between the v1.0.0 release and a master that has the complex
    # support: HermEigsSolver.h was added by the same series of changes. Without it the complex
    # instantiation in src/nanobind/spectra/spectra.cpp does not compile, and the error it produces
    # deep inside Spectra's templates says nothing about the actual problem.
    message(FATAL_ERROR
      "The Spectra tree in ${PYOOMPH_SPECTRA_SOURCE_DIR} predates the complex-scalar support "
      "(include/Spectra/HermEigsSolver.h is missing) - the v1.0.0 release is too old for pyoomph. "
      "Use a master checkout, or leave PYOOMPH_SPECTRA_SOURCE_DIR empty to download one.")
  endif()
  set(_pyoomph_spectra_inc "${PYOOMPH_SPECTRA_SOURCE_DIR}/include")
  message(STATUS "pyoomph: using the Spectra headers in ${PYOOMPH_SPECTRA_SOURCE_DIR}")
else()
  if(PYOOMPH_SPECTRA_URL)
    set(_pyoomph_spectra_url "${PYOOMPH_SPECTRA_URL}")
  else()
    set(_pyoomph_spectra_url "https://github.com/yixuan/spectra/archive/${PYOOMPH_SPECTRA_REF}.tar.gz")
  endif()
  set(_pyoomph_spectra_src "${CMAKE_BINARY_DIR}/spectra/source")
  set(_pyoomph_spectra_inc "${_pyoomph_spectra_src}/include")
  ExternalProject_Add(spectra_external
    URL "${_pyoomph_spectra_url}"
    PREFIX "${CMAKE_BINARY_DIR}/spectra"
    SOURCE_DIR "${_pyoomph_spectra_src}"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS "${_pyoomph_spectra_inc}/Spectra/GenEigsSolver.h"
  )
  list(APPEND PYOOMPH_SPECTRA_DEPENDS spectra_external)
  message(STATUS "pyoomph: using Spectra ${PYOOMPH_SPECTRA_REF}")
endif()

set(PYOOMPH_SPECTRA_INCLUDE_DIRS "${_pyoomph_eigen_inc}" "${_pyoomph_spectra_inc}")
