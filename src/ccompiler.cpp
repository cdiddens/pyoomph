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

#ifndef PYOOMPH_NO_TCC
#include "thirdparty/tinycc/libtcc.h"
#endif

#include <cmath>

namespace pyoomph
{

  CCompiler::CCompiler() : code_trunk(""), code(""), current_handle(NULL) {}

  CCompiler::~CCompiler()
  {
  }

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

#ifndef PYOOMPH_NO_TCC
  static void tcc_err_callback(void *err_opaque, const char *msg)
  {
    std::cerr << "##TCC: " << msg << std::endl;
  };
#endif

  bool CCompiler::compile_to_memory()
  {
    return true;
  }

  bool CCompiler::compile(bool suppress_compilation, bool suppress_code_writing, bool quiet, const std::vector<std::string> &extra_flags)
  {
  #ifndef PYOOMPH_NO_TCC
    init_in_mem = NULL;
    TCCState *s = tcc_new();
    tcc_set_error_func(s, NULL, tcc_err_callback);
    if (!s)
    {
      throw_runtime_error("Can't create a TCC context");
      return false;
    }
    tcc_set_options(s, "-O3");
    tcc_set_options(s, "-m64");
    if (this->compile_to_memory())
    {
      tcc_set_output_type(s, TCC_OUTPUT_MEMORY);
      tcc_define_symbol(s, "PYOOMPH_TCC_TO_MEMORY", NULL);

      // TODO: Add more

      tcc_add_symbol(s, "acos", (void *)static_cast<double (*)(double)>(std::acos));
      tcc_add_symbol(s, "asin", (void *)static_cast<double (*)(double)>(std::asin));
      tcc_add_symbol(s, "atan", (void *)static_cast<double (*)(double)>(std::atan));
      tcc_add_symbol(s, "atan2", (void *)static_cast<double (*)(double, double)>(std::atan2));
      tcc_add_symbol(s, "acosh", (void *)static_cast<double (*)(double)>(std::acosh));
      tcc_add_symbol(s, "asinh", (void *)static_cast<double (*)(double)>(std::asinh));
      tcc_add_symbol(s, "atanh", (void *)static_cast<double (*)(double)>(std::atanh));
      tcc_add_symbol(s, "cos", (void *)static_cast<double (*)(double)>(std::cos));
      tcc_add_symbol(s, "sin", (void *)static_cast<double (*)(double)>(std::sin));
      tcc_add_symbol(s, "tan", (void *)static_cast<double (*)(double)>(std::tan));
      tcc_add_symbol(s, "cosh", (void *)static_cast<double (*)(double)>(std::cosh));
      tcc_add_symbol(s, "sinh", (void *)static_cast<double (*)(double)>(std::sinh));
      tcc_add_symbol(s, "tanh", (void *)static_cast<double (*)(double)>(std::tanh));
      tcc_add_symbol(s, "exp", (void *)static_cast<double (*)(double)>(std::exp));
      tcc_add_symbol(s, "log", (void *)static_cast<double (*)(double)>(std::log));
      tcc_add_symbol(s, "log10", (void *)static_cast<double (*)(double)>(std::log10));
      tcc_add_symbol(s, "pow", (void *)static_cast<double (*)(double, double)>(std::pow));
      tcc_add_symbol(s, "sqrt", (void *)static_cast<double (*)(double)>(std::sqrt));
      tcc_add_symbol(s, "fabs", (void *)static_cast<double (*)(double)>(std::fabs));
      tcc_add_symbol(s, "fmax", (void *)static_cast<double (*)(double, double)>(std::fmax));
      tcc_add_symbol(s, "fmin", (void *)static_cast<double (*)(double, double)>(std::fmin));
      tcc_add_symbol(s, "free", (void *)static_cast<void (*)(void *)>(free));

#ifndef _WIN32
      tcc_define_symbol(s, "size_t", "long unsigned int");
      tcc_add_symbol(s, "strlen", (void *)static_cast<long unsigned int (*)(const char *)>(strlen));
      tcc_add_symbol(s, "strncpy", (void *)static_cast<char *(*)(char *, const char *, long unsigned int)>(strncpy));
      tcc_add_symbol(s, "malloc", (void *)static_cast<void *(*)(long unsigned int)>(malloc));
      tcc_add_symbol(s, "calloc", (void *)static_cast<void *(*)(long unsigned int, long unsigned int)>(calloc));
#else
      tcc_define_symbol(s, "size_t", "long long unsigned int");
      tcc_add_symbol(s, "strlen", (void *)static_cast<long long unsigned int (*)(const char *)>(strlen));
      tcc_add_symbol(s, "strncpy", (void *)static_cast<char *(*)(char *, const char *, long long unsigned int)>(strncpy));
      tcc_add_symbol(s, "malloc", (void *)static_cast<void *(*)(long long unsigned int)>(malloc));
      tcc_add_symbol(s, "calloc", (void *)static_cast<void *(*)(long long unsigned int, long long unsigned int)>(calloc));
#endif

      tcc_define_symbol(s, "NULL", "(void*)(0)");

      tcc_add_symbol(s, "strdup", (void *)static_cast<char *(*)(const char *)>(strdup));
    }
    else
    {
      tcc_set_output_type(s, TCC_OUTPUT_DLL);
    }
    std::string code_file = code_trunk + ".c";
    if (!quiet)
    {
      std::cout << "ADDING LIB PATH " << g_jit_include_dir << std::endl;
    }
    tcc_set_lib_path(s, g_jit_include_dir.c_str());
    tcc_add_library_path(s, g_jit_include_dir.c_str());
    if (tcc_add_include_path(s, g_jit_include_dir.c_str()))
    {
      throw_runtime_error("Problem setting TCC include path");
    }

    if (tcc_add_file(s, code_file.c_str()))
    {
      throw_runtime_error("Problem adding TCC source");
    }

    if (this->compile_to_memory())
    {
      unsigned long memsize = tcc_relocate(s, NULL);
      if (!memsize)
      {
        throw_runtime_error("Empty code result for relocation");
      }
      current_handle = malloc(memsize);
      if (!current_handle)
      {
        throw_runtime_error("Cannot aquire " + std::to_string(memsize) + " bytes for relocation");
      }
      tcc_relocate(s, current_handle);
      init_in_mem = (JIT_ELEMENT_init_SPEC)tcc_get_symbol(s, "JIT_ELEMENT_init");
      if (!init_in_mem)
      {
        throw_runtime_error("Cannot find init function in memory");
      }
    }
    else
    {
      std::string so_file = code_trunk + this->get_shared_lib_extension();
      if (tcc_output_file(s, so_file.c_str()))
      {
        printf("Compilation error !\n");
        return false;
      }
    }
    tcc_delete(s);
    return true;
    #else
     throw_runtime_error("Cannot compile via TCC since pyoomph was configures with PYOOMPH_NO_TCC. Use the method set_c_compiler('distutils') of the Problem class to use the system compiler or add the command line argument --distutils");
     return false;
    #endif
    
    
  }

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

  std::string g_jit_include_dir = "";
}
