# Prepares a freshly downloaded TQMesh source tree for use inside pyoomph. Run as
#
#   cmake -D TQMESH_SOURCE_DIR=<tree> -D TQMESH_CONFIG_DIR=<dir> -P patch_tqmesh.cmake
#
# from cmake/ThirdPartyTQMesh.cmake's PATCH_COMMAND, i.e. once per download.
#
# TQMesh itself is header-only, so there is nothing to build - but two things have to be arranged
# before its headers can be included:
#
# 1) TQMesh must not end the process. Its ASSERT/assert_msg calls assert() in a debug build, has an
#    empty body in a release one, and TERMINATE calls exit(EXIT_FAILURE). That is a reasonable choice
#    for the command line application TQMesh ships as, but pyoomph runs it inside the user's python
#    session, where neither killing the process nor carrying on past a violated invariant is
#    acceptable - the latter is what turns a mesh size function returning zero, or a quad layer
#    requested between positions that are not connected by boundary edges, into a segmentation fault
#    rather than an exception. Both therefore throw here, and src/nanobind/tqmesh/ turns the
#    exception into a python error. The assertions are consequently also active in release builds,
#    which is the point: they are cheap, and continuing past one is what caused the crash.
#
#    Making them throw has one consequence of its own: the ASSERT in ~QuadTree() would then throw
#    from a destructor and call std::terminate, so it becomes a plain null check.
#
# 2) Several TQMesh headers include <TQMeshConfig.h> unconditionally. Upstream generates it from
#    auxiliary/TQMeshConfig.h.in through configure_file() in its own CMakeLists.txt, which pyoomph
#    does not run - it compiles the headers directly. So generate it here, from that very template
#    and with the version taken from TQMesh's own CMakeLists.txt.
#
# Idempotent: ExternalProject can re-run a PATCH_COMMAND on an already-patched tree. Each
# replacement is skipped when its result is already present, and a replacement that finds neither
# its result nor the text it applies to is a hard error - a TQMesh version whose sources moved on must be looked at
# rather than silently built without the safety net above.

if(NOT TQMESH_SOURCE_DIR OR NOT TQMESH_CONFIG_DIR)
  message(FATAL_ERROR "patch_tqmesh.cmake needs -D TQMESH_SOURCE_DIR=... -D TQMESH_CONFIG_DIR=...")
endif()

# Replaces `match` by `replacement` in `file`, unless it is already there. Line endings and
# trailing whitespace are normalized first, so that the passages to be found below can be written
# out plainly and stay found when upstream reflows them.
function(pyoomph_tqmesh_patch file match replacement)
  set(_path "${TQMESH_SOURCE_DIR}/${file}")
  if(NOT EXISTS "${_path}")
    message(FATAL_ERROR "pyoomph: ${_path} does not exist - is this a TQMesh source tree?")
  endif()
  file(READ "${_path}" _content)
  string(REGEX REPLACE "\r\n" "\n" _content "${_content}")
  string(REGEX REPLACE "[ \t]+\n" "\n" _content "${_content}")
  # Per replacement rather than per file - two of them share a file, and a marker somewhere in it
  # says nothing about this one
  string(FIND "${_content}" "${replacement}" _already)
  if(NOT _already EQUAL -1)
    return()
  endif()
  string(FIND "${_content}" "${match}" _found)
  if(_found EQUAL -1)
    message(FATAL_ERROR
      "pyoomph: could not find the passage to patch in ${file} of TQMesh. This version of TQMesh "
      "differs from the one pyoomph knows how to make safe to embed (see the comment at the top of "
      "citools/patches/patch_tqmesh.cmake). Either pin the known-good TQMesh with "
      "-DPYOOMPH_TQMESH_REF=..., or update this script for the new sources.")
  endif()
  string(REPLACE "${match}" "${replacement}" _content "${_content}")
  file(WRITE "${_path}" "${_content}")
  message(STATUS "pyoomph: patched ${file} of TQMesh")
endfunction()

# --- 1a) assertions and TERMINATE throw instead of aborting, exiting or vanishing --------------
pyoomph_tqmesh_patch("src/utils/Helpers.h"
"#ifndef NDEBUG
static inline void assert_msg(bool cond, const std::string& msg,
                              const std::string& filename, int line)
{
  if (!cond)
  {
    std::cerr << \"[ERROR] \" << msg << \" (\" << filename
              << \" - Line \" << line << \")\" << std::endl;
    assert(cond);
  }
}
#else
static inline void assert_msg(bool cond, const std::string& msg,
                              const std::string& filename, int line)
{}
#endif"
"//FOR PYOOMPH: throws instead of calling assert(), or of doing nothing at all in a release build -
// see citools/patches/patch_tqmesh.cmake for why, and note that this makes the assertions active in
// release builds as well.
#include <stdexcept>
static inline void assert_msg(bool cond, const std::string& msg,
                              const std::string& filename, int line)
{
  if (!cond)
    throw std::runtime_error(\"TQMesh: \" + msg + \" (\" + filename
                             + \" - Line \" + std::to_string(line) + \")\");
}")

pyoomph_tqmesh_patch("src/utils/Helpers.h"
"static inline void TERMINATE(const std::string& msg)
{
  std::cerr << \"[ERROR] \" << msg << std::endl;
  exit(EXIT_FAILURE);
}"
"//FOR PYOOMPH: throws instead of exiting the process, see the comment above assert_msg()
static inline void TERMINATE(const std::string& msg)
{
  throw std::runtime_error(\"TQMesh: \" + msg);
}")

# --- 1b) ... which means this one must not be an assertion any more ----------------------------
pyoomph_tqmesh_patch("src/utils/QuadTree.h"
"        ASSERT( child, \"QuadTree structure is corrupted.\" );
        delete  child;"
"        //FOR PYOOMPH: was an ASSERT, which now throws (see Helpers.h) - and throwing from a
        // destructor would call std::terminate. Skipping the corrupted child leaks it at worst,
        // which beats killing the process while the quadtree is already being torn down.
        if ( !child )
          continue;
        delete  child;")

# --- 2) the configuration header TQMesh's own CMakeLists.txt would have generated ---------------
file(READ "${TQMESH_SOURCE_DIR}/CMakeLists.txt" _tqmesh_cmakelists)
# Anchored on project(), since the cmake_minimum_required() above it carries a VERSION as well
if(NOT _tqmesh_cmakelists MATCHES "project[ \t\r\n]*\\([^)]*VERSION[ \t\r\n]+([0-9]+)\\.([0-9]+)")
  message(FATAL_ERROR "pyoomph: cannot read TQMesh's version from its CMakeLists.txt")
endif()
set(TQMesh_VERSION_MAJOR "${CMAKE_MATCH_1}")
set(TQMesh_VERSION_MINOR "${CMAKE_MATCH_2}")
# The template also refers to ${CMAKE_SOURCE_DIR}, which upstream fills with the root of the TQMesh
# tree; in script mode that variable would be wherever cmake -P happened to be invoked.
set(CMAKE_SOURCE_DIR "${TQMESH_SOURCE_DIR}")
configure_file("${TQMESH_SOURCE_DIR}/auxiliary/TQMeshConfig.h.in" "${TQMESH_CONFIG_DIR}/TQMeshConfig.h")
message(STATUS "pyoomph: using TQMesh ${TQMesh_VERSION_MAJOR}.${TQMesh_VERSION_MINOR} from ${TQMESH_SOURCE_DIR}")
