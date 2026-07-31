// TQMeshConfig.h
//
//FOR PYOOMPH: This file is not part of TQMesh. Upstream generates it from auxiliary/TQMeshConfig.h.in
// via configure_file() in its own CMakeLists.txt, which pyoomph does not run - it compiles the
// vendored headers directly (see src/thirdparty/INFO_tqmesh). Several TQMesh headers include
// <TQMeshConfig.h> unconditionally, so a replacement has to exist somewhere on the include path.
//
// Only the version macros are reproduced. TQMESH_SOURCE_DIR pointed at the source tree of the
// TQMesh build, which upstream only uses to place the output files of its examples and tests -
// neither of which is copied here - so it is deliberately left out rather than being baked with a
// path that does not exist on a user's machine.
#pragma once

#define TQMESH_VERSION_MAJOR 1
#define TQMESH_VERSION_MINOR 4
