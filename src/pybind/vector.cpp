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
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
namespace py = pybind11;

#include <sstream>

#include "../oomph_lib.hpp"

void PyReg_Vector(py::module &m)
{

	py::class_<oomph::Vector<double>>(m, "VectorDouble")
		.def("__getitem__", [](const oomph::Vector<double> *self, const int &i)
			 { return (*self)[i]; })
		.def("__setitem__", [](oomph::Vector<double> *self, const int &i, const double &v)
			 { (*self)[i] = v; })
		.def("size", &oomph::Vector<double>::size);
}
