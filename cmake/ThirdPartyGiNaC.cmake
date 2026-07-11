# Finds (or downloads and builds) CLN and GiNaC.
#
# GiNaC and CLN are autotools projects, not CMake projects, so "downloading
# and building them during the CMake build" means invoking their own real
# ./configure && make && make install via ExternalProject_Add - CMake itself
# never compiles their sources directly.
#
# NOTE: the exact ./configure flags GiNaC/CLN accept (e.g. how GiNaC is told
# where to find CLN) can vary a bit by version. The flags below work for
# recent GiNaC/CLN autotools releases as of this writing, but if you hit a
# "./configure: unrecognized option" style failure, check
# `<extracted-source>/configure --help` for that exact version and adjust
# GINAC_EXTRA_CONFIGURE_FLAGS / CLN_EXTRA_CONFIGURE_FLAGS below (or pass them
# in via -DGINAC_EXTRA_CONFIGURE_FLAGS=... / -DCLN_EXTRA_CONFIGURE_FLAGS=...).
#
# Sets, for use by the top-level CMakeLists.txt:
#   PYOOMPH_CLN_INCLUDE_DIR_RESOLVED / PYOOMPH_GINAC_INCLUDE_DIR_RESOLVED
#   PYOOMPH_CLN_LIBRARY / PYOOMPH_GINAC_LIBRARY
# and (when downloading) the ExternalProject targets cln_external /
# ginac_external, which the extension module is made to depend on.

include(ExternalProject)
include(GNUInstallDirs)

set(CLN_EXTRA_CONFIGURE_FLAGS "" CACHE STRING "Extra flags passed to CLN's ./configure")
set(GINAC_EXTRA_CONFIGURE_FLAGS "" CACHE STRING "Extra flags passed to GiNaC's ./configure")

set(_pyoomph_autotools_common_flags
    "--enable-static" "--disable-shared" "CFLAGS=-fPIC" "CXXFLAGS=-fPIC")

# ---------------------------------------------------------------- CLN -----
if(PYOOMPH_DOWNLOAD_CLN)
  set(_cln_lib "${PYOOMPH_THIRDPARTY_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libcln.a")
  ExternalProject_Add(cln_external
    URL "https://www.ginac.de/CLN/cln-${PYOOMPH_CLN_VERSION}.tar.bz2"
    PREFIX "${CMAKE_BINARY_DIR}/cln_build"
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND <SOURCE_DIR>/configure
                       --prefix=${PYOOMPH_THIRDPARTY_PREFIX}
                       ${_pyoomph_autotools_common_flags}
                       ${CLN_EXTRA_CONFIGURE_FLAGS}
    BUILD_COMMAND $(MAKE)
    INSTALL_COMMAND $(MAKE) install
    BUILD_BYPRODUCTS "${_cln_lib}"
  )
  set(PYOOMPH_CLN_INCLUDE_DIR_RESOLVED "${PYOOMPH_THIRDPARTY_PREFIX}/include")
  set(PYOOMPH_CLN_LIBRARY "${_cln_lib}")
else()
  find_path(_pyoomph_cln_include NAMES cln/cln.h
    HINTS "${PYOOMPH_CLN_INCLUDE_DIR}" ENV PYOOMPH_CLN_INCLUDE_DIR)
  find_library(_pyoomph_cln_lib NAMES cln
    HINTS "${PYOOMPH_CLN_LIB_DIR}" ENV PYOOMPH_CLN_LIB_DIR)
  if(NOT _pyoomph_cln_include OR NOT _pyoomph_cln_lib)
    message(FATAL_ERROR
      "CLN not found. Either install it system-wide (optionally hint its "
      "location via -DPYOOMPH_CLN_INCLUDE_DIR=... -DPYOOMPH_CLN_LIB_DIR=...), "
      "or configure with -DPYOOMPH_DOWNLOAD_CLN=ON to build it from source.")
  endif()
  set(PYOOMPH_CLN_INCLUDE_DIR_RESOLVED "${_pyoomph_cln_include}")
  set(PYOOMPH_CLN_LIBRARY "${_pyoomph_cln_lib}")
endif()

# --------------------------------------------------------------- GiNaC ----
if(PYOOMPH_DOWNLOAD_GINAC)
  set(_ginac_lib "${PYOOMPH_THIRDPARTY_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libginac.a")

  # GiNaC's ./configure locates CLN via pkg-config (cln.pc) by default, so
  # point PKG_CONFIG_PATH at wherever CLN's .pc file ended up - whether from
  # our own just-built CLN, or a system/hinted one.
  if(PYOOMPH_DOWNLOAD_CLN)
    set(_ginac_depends cln_external)
    set(_cln_pkgconfig_dir "${PYOOMPH_THIRDPARTY_PREFIX}/${CMAKE_INSTALL_LIBDIR}/pkgconfig")
  else()
    set(_ginac_depends "")
    get_filename_component(_cln_lib_dir "${PYOOMPH_CLN_LIBRARY}" DIRECTORY)
    set(_cln_pkgconfig_dir "${_cln_lib_dir}/pkgconfig")
  endif()

  ExternalProject_Add(ginac_external
    URL "https://www.ginac.de/ginac-${PYOOMPH_GINAC_VERSION}.tar.bz2"
    PREFIX "${CMAKE_BINARY_DIR}/ginac_build"
    DEPENDS ${_ginac_depends}
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env
                       "PKG_CONFIG_PATH=${_cln_pkgconfig_dir}"
                       "CPPFLAGS=-I${PYOOMPH_CLN_INCLUDE_DIR_RESOLVED}"
                       "LDFLAGS=-L${PYOOMPH_THIRDPARTY_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
                       <SOURCE_DIR>/configure
                       --prefix=${PYOOMPH_THIRDPARTY_PREFIX}
                       ${_pyoomph_autotools_common_flags}
                       ${GINAC_EXTRA_CONFIGURE_FLAGS}
    BUILD_COMMAND $(MAKE)
    INSTALL_COMMAND $(MAKE) install
    BUILD_BYPRODUCTS "${_ginac_lib}"
  )
  set(PYOOMPH_GINAC_INCLUDE_DIR_RESOLVED "${PYOOMPH_THIRDPARTY_PREFIX}/include")
  set(PYOOMPH_GINAC_LIBRARY "${_ginac_lib}")
else()
  find_path(_pyoomph_ginac_include NAMES ginac/ginac.h
    HINTS "${PYOOMPH_GINAC_INCLUDE_DIR}" ENV PYOOMPH_GINAC_INCLUDE_DIR)
  find_library(_pyoomph_ginac_lib NAMES ginac
    HINTS "${PYOOMPH_GINAC_LIB_DIR}" ENV PYOOMPH_GINAC_LIB_DIR)
  if(NOT _pyoomph_ginac_include OR NOT _pyoomph_ginac_lib)
    message(FATAL_ERROR
      "GiNaC not found. Either install it system-wide (optionally hint its "
      "location via -DPYOOMPH_GINAC_INCLUDE_DIR=... -DPYOOMPH_GINAC_LIB_DIR=...), "
      "or configure with -DPYOOMPH_DOWNLOAD_GINAC=ON to build it from source.")
  endif()
  set(PYOOMPH_GINAC_INCLUDE_DIR_RESOLVED "${_pyoomph_ginac_include}")
  set(PYOOMPH_GINAC_LIBRARY "${_pyoomph_ginac_lib}")
endif()
