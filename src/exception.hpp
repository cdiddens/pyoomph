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

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace pyoomph
{
  // A std::runtime_error whose what() message is prefixed with "file:line: " of the throw
  // site, so error messages surfacing in Python (via the pybind exception translation) are
  // traceable back to the offending C++ location. Always construct via the
  // throw_runtime_error(arg) macro below rather than directly, so file/line are filled in
  // automatically.
  class runtime_error_with_line : public std::runtime_error
  {
  protected:
    std::string msg;

  public:
    runtime_error_with_line(const std::string &arg, const char *file, int line);
    ~runtime_error_with_line() throw() {}
    const char *what() const throw();

  };

  extern int pyoomph_verbose; // Global verbosity level, checked by various parts of the code to decide whether to print diagnostic output

}
// Throw a runtime_error_with_line carrying the current file and line number; use this
// instead of throwing runtime_error_with_line directly.
#define throw_runtime_error(arg) throw pyoomph::runtime_error_with_line(arg, __FILE__, __LINE__);
