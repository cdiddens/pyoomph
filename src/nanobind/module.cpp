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


#include <nanobind/nanobind.h>

namespace nb = nanobind;

// "PyDecl_*" functions (implemented in the correspondingly named files under src/nanobind/) only
// call nb::class_<...>(m, "Name") to register a class's Python *name*, without yet adding any of
// its methods/properties. This lets classes from different translation units refer to each
// other's Python types (e.g. as a function argument or return type) regardless of registration
// order, as long as the referenced class has at least been declared before it is first used.
void PyDecl_Mesh(nb::module_ &m);
void PyDecl_CodeGen(nb::module_ &m);
void PyDecl_Problem(nb::module_ &m);

// "PyReg_*" functions do the actual binding work: adding methods, properties, operators and
// free functions to the module (and to the classes declared above via PyDecl_*).
void PyReg_Problem(nb::module_ &m);
void PyReg_TimeStepper(nb::module_ &m);
void PyReg_CodeGen(nb::module_ &m);
void PyReg_Expressions(nb::module_ &m);
void PyReg_Mesh(nb::module_ &m);
void PyReg_Solvers(nb::module_ &m);
void PyReg_GeomObjects(nb::module_ &m);
void PyReg_Vector(nb::module_ &m);
#ifdef PYOOMPH_HAS_TQMESH
// Bindings for the TQMesh mesh generator (src/nanobind/tqmesh; the library itself is downloaded by
// cmake/ThirdPartyTQMesh.cmake).
// Optional: the build option PYOOMPH_HAS_TQMESH can leave TQMesh out entirely, in which case
// neither this function nor its translation unit exists.
void PyReg_TQMesh(nb::module_ &m);
#endif

// Compiled and installed as pyoomph/_pyoomph_core*.so, i.e. importable as pyoomph._pyoomph_core.
#define PYOOMPH_MODULE_NAME _pyoomph_core

NB_MODULE(PYOOMPH_MODULE_NAME, m)
{
    m.doc() = "This module exposes the compiled C++ core of pyoomph via nanobind to python. Here, the relevant C++ base classes and further low-level functions can be found. Usually, it is not necessary for a user to use these functions directly.";

    // Declaration phase: register the bare Python types first, before any of them are used as
    // parameter/return types by the PyReg_* calls below.
    PyDecl_Mesh(m);
    PyDecl_CodeGen(m);
    PyDecl_Problem(m);

    // Registration phase: fill in the actual bindings. Order mostly doesn't matter here since
    // all cross-referenced types were already declared above, but PyReg_Expressions() must run
    // before PyReg_Problem()/PyReg_CodeGen()/PyReg_Mesh(), which bind functions returning or
    // accepting GiNaC expression types defined there.
    PyReg_TimeStepper(m);
    PyReg_GeomObjects(m);

    PyReg_Expressions(m);
    PyReg_Problem(m);
    PyReg_CodeGen(m);
    PyReg_Mesh(m);
    PyReg_Solvers(m);
    PyReg_Vector(m);

    // Whether the threaded element loop (--omp N) is compiled in at all. Same purpose as
    // has_tqmesh below, but it answers a question that is otherwise unanswerable from Python: a
    // build without OpenMP accepts --omp N, prints one note and assembles serially, so a wheel that
    // lost OpenMP at configure time is indistinguishable from a working one until someone measures.
    // tests/test_openmp_assembly.py skips on it, and the wheel CI asserts it is true.
#ifdef PYOOMPH_HAS_OPENMP
    m.attr("has_openmp") = true;
#else
    m.attr("has_openmp") = false;
#endif

    // The GCD/libdispatch backend for the same threaded element loop (macOS only). It is an
    // ALTERNATIVE to OpenMP that ships no OpenMP runtime, so a mac wheel can offer --omp N without a
    // libomp that collides with MKL's libiomp5 (OMP: Error #15). has_threaded_assembly is the one the
    // bit-identity tests should gate on - either backend gives a threaded loop to compare.
#ifdef PYOOMPH_HAS_GCD
    m.attr("has_gcd") = true;
#else
    m.attr("has_gcd") = false;
#endif
#if defined(PYOOMPH_HAS_OPENMP) || defined(PYOOMPH_HAS_GCD)
    m.attr("has_threaded_assembly") = true;
#else
    m.attr("has_threaded_assembly") = false;
#endif

    // Lets python find out whether this build can mesh with TQMesh, without having to probe for
    // the classes registered by PyReg_TQMesh().
#ifdef PYOOMPH_HAS_TQMESH
    m.attr("has_tqmesh") = true;
    PyReg_TQMesh(m);
#else
    m.attr("has_tqmesh") = false;
#endif
}
