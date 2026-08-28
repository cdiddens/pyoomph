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

// Definitions for the per-thread channels declared in thread_state.hpp.

#include "thread_state.hpp"

#include <unordered_map>

namespace pyoomph
{

  thread_local bool __crj_thread_override = false;

  // Only the tables this thread has actually written are in here. A miss falls back to the table
  // value, which is the value the serial run would have read at that point, so no seeding pass over
  // all codes is needed when the scope opens.
  static thread_local std::unordered_map<const JITFuncSpec_Table_FiniteElement_t *, int> __crj_thread_values;

  int __crj_get(const JITFuncSpec_Table_FiniteElement_t *ft)
  {
    auto it = __crj_thread_values.find(ft);
    return (it != __crj_thread_values.end() ? it->second : ft->current_res_jac);
  }

  void __crj_set(JITFuncSpec_Table_FiniteElement_t *ft, int v)
  {
    __crj_thread_values[ft] = v;
  }

  CurrentResJacThreadScope::CurrentResJacThreadScope() : prev(__crj_thread_override)
  {
    // Cleared rather than carried over: a previous parallel region's last mode must not leak into
    // this one, whose handler may set nothing at all and expect the table default.
    __crj_thread_values.clear();
    __crj_thread_override = true;
  }

  CurrentResJacThreadScope::~CurrentResJacThreadScope()
  {
    __crj_thread_values.clear();
    __crj_thread_override = prev;
  }

  gil_release_fn __gil_release_hook = NULL;
  gil_acquire_fn __gil_acquire_hook = NULL;

  void install_gil_hooks(gil_release_fn rel, gil_acquire_fn acq)
  {
    __gil_release_hook = rel;
    __gil_acquire_hook = acq;
  }

}
