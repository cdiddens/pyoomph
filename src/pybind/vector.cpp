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


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
namespace py = pybind11;

#include <sstream>

#include "../oomph_lib.hpp"

void PyReg_Vector(py::module &m)
{

	py::class_<oomph::Vector<double>>(
		m, "VectorDouble",
		"Thin Python wrapper around oomph-lib's oomph::Vector<double>, a std::vector<double> subclass with "
		"optional range checking. Used to expose plain C++ double vectors (e.g. eigenvectors, arclength "
		"direction vectors) to Python without copying them into a numpy array.")
		.def(
			"__getitem__", [](const oomph::Vector<double> *self, const int &i)
			{ return (*self)[i]; },
			py::arg("i"), "Return the entry at index ``i``.")
		.def(
			"__setitem__", [](oomph::Vector<double> *self, const int &i, const double &v)
			{ (*self)[i] = v; },
			py::arg("i"), py::arg("value"), "Set the entry at index ``i`` to ``value``.")
		.def("size", &oomph::Vector<double>::size, "Return the number of entries in the vector.");
}
