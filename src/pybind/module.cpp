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

The authors may be contacted at c.diddens@utwente.nl and d.rocha@utwente.nl

================================================================================*/


#include <pybind11/pybind11.h>

namespace py = pybind11;

void PyDecl_Mesh(py::module &m);
void PyDecl_CodeGen(py::module &m);
void PyDecl_Problem(py::module &m);

void PyReg_Problem(py::module &m);
void PyReg_TimeStepper(py::module &m);
void PyReg_CodeGen(py::module &m);
void PyReg_Expressions(py::module &m);
void PyReg_Mesh(py::module &m);
void PyReg_Solvers(py::module &m);
void PyReg_GeomObjects(py::module &m);
void PyReg_Vector(py::module &m);

#define PYOOMPH_MODULE_NAME _pyoomph

PYBIND11_MODULE(PYOOMPH_MODULE_NAME, m)
{
    m.doc() = "This module exposes the compiled C++ core of pyoomph via pybind11 to python. Here, the relevant C++ base classes and further low-level functions can be found. Usually, it is not necessary for a user to use these functions directly.";

    PyDecl_Mesh(m);
    PyDecl_CodeGen(m);
    PyDecl_Problem(m);

    PyReg_TimeStepper(m);
    PyReg_GeomObjects(m);

    PyReg_Expressions(m);
    PyReg_Problem(m);    
    PyReg_CodeGen(m);
    PyReg_Mesh(m);
    PyReg_Solvers(m);
    PyReg_Vector(m);
}
