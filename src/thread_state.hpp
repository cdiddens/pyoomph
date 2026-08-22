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

// Per-thread channels that the parallel element loop needs, and the hooks that let pyoomph_core -
// which links neither Python nor nanobind - hand the GIL back and forth. Kept separate from
// parallel_assembly.hpp because this half is included by elements.hpp, i.e. by nearly everything,
// while the engine itself only concerns problem.cpp. See dev_docs/openmp_assembly.md.

#pragma once

#include "jitbridge.h"

namespace pyoomph
{

  // ---------------------------------------------------------------------------------------------
  // current_res_jac: which residual form a code is currently assembling.
  //
  // It lives in the function table, which is shared by every element of a code - and the pitchfork
  // and azimuthal handlers WRITE it per element, in the middle of their own get_residuals() /
  // get_jacobian() (MyPitchForkHandler::set_assembled_residual,
  // AzimuthalSymmetryBreakingHandler::set_assembled_residual). Two threads assembling two elements
  // of the same code are then at different points of that sequence and overwrite each other. The
  // note left on the field by the previous commit - "set once per solve, read-only during a sweep" -
  // holds for the default assembly handler only.
  //
  // The field itself cannot become thread_local: it sits in a C struct that generated code includes
  // (jitbridge.h is part of the JIT cache key), even though no generated code reads it. So the
  // WRITES are diverted instead. Off the parallel path __crj_thread_override is false and both
  // accessors are the plain field access they replace, which is what keeps the serial cost at a
  // predicted branch on an already-hot thread_local flag.
  extern thread_local bool __crj_thread_override;

  int __crj_get(const JITFuncSpec_Table_FiniteElement_t *ft);
  void __crj_set(JITFuncSpec_Table_FiniteElement_t *ft, int v);

  inline int get_current_res_jac(const JITFuncSpec_Table_FiniteElement_t *ft)
  {
    return __crj_thread_override ? __crj_get(ft) : ft->current_res_jac;
  }

  inline void set_current_res_jac(JITFuncSpec_Table_FiniteElement_t *ft, int v)
  {
    if (__crj_thread_override)
      __crj_set(ft, v);
    else
      ft->current_res_jac = v;
  }

  // Opens the per-thread override for its lifetime, seeded from whatever the tables currently say,
  // so a worker that never calls set_current_res_jac() reads exactly the value the serial run would.
  // Constructed inside each worker of a parallel element loop.
  class CurrentResJacThreadScope
  {
    bool prev;

  public:
    CurrentResJacThreadScope();
    ~CurrentResJacThreadScope();
  };

  // ---------------------------------------------------------------------------------------------
  // GIL hooks. Generated code can call back into Python (CustomMathExpression,
  // CustomMultiReturnExpression) through functable->invoke_callback, and a worker thread holds no
  // GIL. pyoomph_core cannot include Python.h, so the nanobind layer installs these two at import
  // time (see install_gil_hooks in src/nanobind/problem.cpp), exactly as it installs
  // invoke_callback itself. Both are NULL in a build or context with no interpreter, and every
  // caller must tolerate that.
  typedef void *(*gil_release_fn)();   // Releases the GIL, returns an opaque token
  typedef void (*gil_acquire_fn)(void *); // Re-acquires with that token
  extern gil_release_fn __gil_release_hook;
  extern gil_acquire_fn __gil_acquire_hook;
  void install_gil_hooks(gil_release_fn rel, gil_acquire_fn acq);

  // Releases the GIL for its lifetime if a hook is installed, otherwise a no-op. Wrapped around a
  // parallel region: the workers then take it back one at a time in the callback trampolines.
  class GILReleaseScope
  {
    void *token;

  public:
    GILReleaseScope() : token(__gil_release_hook ? __gil_release_hook() : NULL) {}
    ~GILReleaseScope()
    {
      if (token && __gil_acquire_hook) __gil_acquire_hook(token);
    }
    GILReleaseScope(const GILReleaseScope &) = delete;
    GILReleaseScope &operator=(const GILReleaseScope &) = delete;
  };

}
