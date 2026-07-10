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


#pragma once
#include <vector>
#include <string>
#include "jitbridge.h"
#include <memory>

namespace pyoomph
{

   extern std::string g_jit_include_dir;

   // This is the TCC compiler
   class CCompiler
   {
   protected:
      std::string code_trunk;
      std::string code;
      void *current_handle;
      JIT_ELEMENT_init_SPEC init_in_mem;

   public:
      CCompiler();
      virtual std::string get_jit_include_dir() { return g_jit_include_dir; }
      virtual void set_code_from_string(const std::string &_code)
      {
         code = _code;
         code_trunk = "";
         current_handle = NULL;
      }
      virtual void set_code_from_file(const std::string &ftrunk)
      {
         code_trunk = ftrunk;
         code = "";
         current_handle = NULL;
      }
      virtual bool support_code_from_string() { return false; }
      virtual std::string get_code_trunk() { return code_trunk; }
      virtual void *get_current_handle() { return current_handle; }
      virtual void reset_current_handle() { current_handle = NULL; }

      virtual std::string get_shared_lib_extension();

      // To implement in C++
      virtual std::string get_shared_library(std::string code_trunk) { return code_trunk + this->get_shared_lib_extension(); }
      virtual std::string expand_full_library_name(std::string relname) { return relname; }
      virtual void close_handle(void *handle);
      virtual JIT_ELEMENT_init_SPEC get_init_func();

      // To implement in python
      virtual bool compile(bool suppress_compilation, bool suppress_code_writing, bool quiet, const std::vector<std::string> &extra_flags);
      virtual bool sanity_check() { return false; }

      virtual bool compile_to_memory();

      virtual ~CCompiler();
   };

   class CustomCCompiler : public CCompiler
   {
   protected:
   public:
      virtual bool compile_to_memory() { return false; }
      virtual bool compile(bool suppress_compilation, bool suppress_code_writing, bool quiet, const std::vector<std::string> &extra_flags) { return false; }
   };

}
