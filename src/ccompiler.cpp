/*================================================================================
pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
Copyright (C) 2021-2026  Christian Diddens, Duarte Rocha & Maxim de Wildt

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. 

The main author may be contacted at c.diddens@utwente.nl

================================================================================*/

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "ccompiler.hpp"
#include "exception.hpp"



#include <cmath>

namespace pyoomph
{

  CCompiler::CCompiler() : code_trunk(""), code(""), current_handle(NULL) {}

  CCompiler::~CCompiler()
  {
  }

  // Platform-specific extension used for on-disk shared libraries produced by compile().
  std::string CCompiler::get_shared_lib_extension()
  {
#ifdef _WIN32
    return ".dll";
#else
#if defined(__APPLE__)
    return ".dylib";
#else
    return ".so";
#endif
#endif
  }

  // Default: assume in-memory compilation (the historical TCC-based mode). Subclasses that
  // compile to an on-disk shared library (e.g. via a system C++ compiler from Python) override
  // this to return false.
  bool CCompiler::compile_to_memory()
  {
    return true;
  }

  // Actual compilation is delegated to a derived class (typically a Python-side compiler
  // implementation that shells out to a system C++ compiler); the C++ base class does not
  // implement compilation itself.
  bool CCompiler::compile(bool suppress_compilation, bool suppress_code_writing, bool quiet, const std::vector<std::string> &extra_flags)
  {
     throw_runtime_error("This method should be implemented in a derived class");
    return false;

  }

  // Compile-if-needed and resolve the JIT_ELEMENT_init entry point of the generated code: for
  // in-memory compilation, current_handle must already have been set by a prior compile() call;
  // for on-disk shared libraries, the library is dlopen'd/LoadLibrary'd here and the symbol
  // looked up by name.
  JIT_ELEMENT_init_SPEC CCompiler::get_init_func()
  {
    if (this->compile_to_memory())
    {
      if (!current_handle)
      {
        throw_runtime_error("No code handle found in memory");
      }
      return init_in_mem;
    }
    else
    {
      std::string fnam = this->get_shared_library(this->code_trunk);
      std::string full_fnam = this->expand_full_library_name(fnam);
#ifndef _WIN32
      void *h = dlopen(fnam.c_str(), RTLD_LOCAL | RTLD_NOW);
      if (!h)
      {
        throw_runtime_error(dlerror());
      }
      current_handle = h;
      dlerror();
      JIT_ELEMENT_init_SPEC initfunc = (JIT_ELEMENT_init_SPEC)dlsym(h, "JIT_ELEMENT_init");
      char *err = dlerror();
      if (err)
      {
        throw_runtime_error(err);
      }
#else
      void *h = (void *)LoadLibrary(fnam.c_str());
      if (!h)
      {
        h = (void *)LoadLibrary(full_fnam.c_str());
        if (!h)
        {
          auto errcode = GetLastError();
          throw_runtime_error("DLL " + fnam + ", i.e. " + full_fnam + " could not be loaded. Error code: " + std::to_string(errcode));
        }
      }
      current_handle = h;
      JIT_ELEMENT_init_SPEC initfunc = (JIT_ELEMENT_init_SPEC)GetProcAddress((HMODULE)h, "JIT_ELEMENT_init");
      if (!initfunc)
      {
        throw_runtime_error("Cannot find entry point in " + fnam);
      }
#endif
      return initfunc;
    }
  }

  // Unload a previously loaded shared library handle; no-op for in-memory compilation, since
  // there is no separate OS-level library to unload.
  void CCompiler::close_handle(void *handle)
  {
    if (this->compile_to_memory())
    {
    }
    else
    {
#ifdef _WIN32
      FreeLibrary((HMODULE)handle);
#else
      dlclose(handle);
#endif
    }
  }

  // Directory containing jitbridge.h and friends, set once at startup from the Python side.
  std::string g_jit_include_dir = "";
}
