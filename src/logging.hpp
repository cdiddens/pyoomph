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


// Simple logging facility that mirrors ("tees") pyoomph's/oomph-lib's console output into
// an optional log file, without requiring callers to change how they write to
// std::cout/std::cerr (see logging.cpp's TeeToLogFile for the actual mechanism).
//
// The same tee also implements the MPI console (setup_mpi_console below): under mpirun every
// rank writes to the same terminal, so without filtering the output is both N-fold duplicated
// and chopped up mid-line. See logging.cpp for what the filter does.

#pragma once
#include <iostream>
#include <string>
namespace pyoomph
{

  extern std::ostream * g_current_log_stream; // The active log file stream, or NULL if not logging to a file
  void set_logging_stream(std::ostream * logstream); // Start (or stop, if NULL) mirroring console output into logstream
  std::ostream * get_logging_stream(); // Return the currently active log file stream (or NULL)
  void write_to_log_file(const std::string & message); // Write directly to the log file only, bypassing the console

  // The stream oomph::oomph_info must be pointed at for its output to reach the log file and the
  // MPI console filter. Anything that temporarily redirects oomph_info (Problem::quiet) has to
  // restore it to this, not to std::cout.
  std::ostream * get_console_stream();

  // Install the MPI console filter for a job of `nproc` ranks, this one being `rank`.
  // `mode` is "condensed" (only rank 0 reaches the terminal), "all" (every rank, but tagged and
  // line-atomic) or "off" (no filtering at all, i.e. the raw pre-existing behaviour).
  void setup_mpi_console(int rank, int nproc, const std::string & mode);
  // What a Python-side console wrapper needs to reproduce the same policy for print().
  std::string get_mpi_console_mode();
  int get_mpi_console_rank();
  int get_mpi_console_nproc();
  // Push out a partial line held in the line buffer (at the end of a run, or before Python writes).
  void flush_console();
};
