/*
* This file is part of the CppUtils library.  
* This code was written by Florian Setzwein in 2022, 
* and is covered under the MIT License
* Refer to the accompanying documentation for details
* on usage and license.
*/
#pragma once

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <stdexcept>
#include <stdlib.h>

namespace CppUtils {

/*********************************************************************
* Asserts messages
*********************************************************************/
//FOR PYOOMPH: Both of these throw in pyoomph, instead of aborting, exiting or - in a release
// build, where the original assert_msg() has an empty body - letting the algorithm carry on with
// a violated invariant. TQMesh is a command line application upstream, where ending the process
// is a legitimate way to report that something went wrong; here it runs inside the user's python
// session, which must not be killed by a mesh that cannot be generated. Both conditions do fire
// on input that is merely unsuitable rather than wrong: a size function returning zero, or a
// quad layer requested between positions that are not connected by boundary edges, for instance,
// so this is the difference between a python exception and a segmentation fault. The assertions
// are therefore also kept active in release builds - they are cheap, and continuing past one is
// exactly what turned a failed quad layer into a crash. src/nanobind/tqmesh/ turns these
// exceptions into python errors.
static inline void assert_msg(bool cond, const std::string& msg,
                              const std::string& filename, int line)
{
  if (!cond)
    throw std::runtime_error("TQMesh: " + msg + " (" + filename
                             + " - Line " + std::to_string(line) + ")");
}

#define ASSERT(cond, msg) assert_msg(cond, msg, __FILE__, __LINE__)

/*********************************************************************
* Terminate 
*********************************************************************/
//FOR PYOOMPH: see the comment above assert_msg()
static inline void TERMINATE(const std::string& msg)
{
  throw std::runtime_error("TQMesh: " + msg);
}

} // namespace CppUtils
