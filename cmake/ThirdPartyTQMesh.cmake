# Acquires TQMesh (https://github.com/FloSewn/TQMesh), the two-dimensional triangle/quad mesh
# generator exposed to python by src/nanobind/tqmesh/.
#
# Unlike CLN/GiNaC (see ThirdPartyGiNaC.cmake), TQMesh is header-only: there is nothing to configure,
# build or install, so this only downloads the sources, prepares them
# (citools/patches/patch_tqmesh.cmake - which is also where the two changes pyoomph makes to them are
# explained) and hands the include directories back.
#
# Sets, for use by the top-level CMakeLists.txt:
#   PYOOMPH_TQMESH_INCLUDE_DIRS
# and the ExternalProject target tqmesh_external, which the extension module must depend on.

include(ExternalProject)

# TQMesh's release tag v1.4.0 is what this defaults to in spirit, but the commit below additionally
# carries the missing virtual destructor in VtkIO.h that the VTU export relies on. Their src/algorithm
# is otherwise identical to the tag. Any git ref works here - a tag ("v1.4.0"), a branch or a commit.
set(PYOOMPH_TQMESH_REF "d77ad3084d4c8746ef99bd03c2dd156532dd7178" CACHE STRING
    "Git ref (commit, tag or branch) of TQMesh to download")
set(PYOOMPH_TQMESH_URL "" CACHE STRING
    "Full URL of a TQMesh source archive to download, overriding PYOOMPH_TQMESH_REF")
set(PYOOMPH_TQMESH_SOURCE_DIR "" CACHE PATH
    "An already available TQMesh source tree to use instead of downloading one. It is copied into the build directory and left untouched.")

if(PYOOMPH_TQMESH_SOURCE_DIR)
  if(NOT EXISTS "${PYOOMPH_TQMESH_SOURCE_DIR}/src/algorithm/TQMesh.h")
    message(FATAL_ERROR
      "PYOOMPH_TQMESH_SOURCE_DIR=${PYOOMPH_TQMESH_SOURCE_DIR} does not contain "
      "src/algorithm/TQMesh.h, so it is not a TQMesh source tree.")
  endif()
  # ExternalProject copies a local directory given as URL, which is what we want: the sources are
  # patched afterwards and the user's own copy must not be touched.
  set(_pyoomph_tqmesh_url "${PYOOMPH_TQMESH_SOURCE_DIR}")
  message(STATUS "pyoomph: using the TQMesh sources in ${PYOOMPH_TQMESH_SOURCE_DIR}")
else()
  if(PYOOMPH_TQMESH_URL)
    set(_pyoomph_tqmesh_url "${PYOOMPH_TQMESH_URL}")
  else()
    set(_pyoomph_tqmesh_url "https://github.com/FloSewn/TQMesh/archive/${PYOOMPH_TQMESH_REF}.tar.gz")
  endif()
  # No URL_HASH: github regenerates these archives, so a pinned hash would eventually fail a build
  # that is otherwise perfectly fine - the ref itself is what pins the sources. (CLN/GiNaC are
  # downloaded without a hash for the same reason.)
  #
  # Also no reachability pre-check as for CLN/GiNaC, which would only cost time here without
  # answering anything: github generates these archives on demand and takes the better part of a
  # minute to answer the HEAD request such a check makes (while serving the actual download in a
  # couple of seconds), so the check timed out on a URL that downloads perfectly well. A wrong ref
  # is reported by the download step below instead.
  message(STATUS "pyoomph: using TQMesh ${PYOOMPH_TQMESH_REF}")
endif()

set(_pyoomph_tqmesh_src "${CMAKE_BINARY_DIR}/tqmesh/source")
set(_pyoomph_tqmesh_config "${CMAKE_BINARY_DIR}/tqmesh/config")

ExternalProject_Add(tqmesh_external
  URL "${_pyoomph_tqmesh_url}"
  PREFIX "${CMAKE_BINARY_DIR}/tqmesh"
  SOURCE_DIR "${_pyoomph_tqmesh_src}"
  PATCH_COMMAND "${CMAKE_COMMAND}"
                -D "TQMESH_SOURCE_DIR=${_pyoomph_tqmesh_src}"
                -D "TQMESH_CONFIG_DIR=${_pyoomph_tqmesh_config}"
                -P "${CMAKE_SOURCE_DIR}/citools/patches/patch_tqmesh.cmake"
  # Header-only: nothing to configure, build or install
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS "${_pyoomph_tqmesh_src}/src/algorithm/TQMesh.h"
                   "${_pyoomph_tqmesh_config}/TQMeshConfig.h"
)

# TQMesh includes its own headers unqualified ("Mesh.h", <TQMeshConfig.h>, ...), so each directory
# has to be on the include path separately.
set(PYOOMPH_TQMESH_INCLUDE_DIRS
  "${_pyoomph_tqmesh_config}"
  "${_pyoomph_tqmesh_src}/src/algorithm"
  "${_pyoomph_tqmesh_src}/src/utils"
)
