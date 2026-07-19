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

namespace py = pybind11;

#include <sstream>

#include "../oomph_lib.hpp"
#include "../exception.hpp"
namespace pyoomph
{

	// Base class for a parametric geometric object (curve/surface), i.e. a mapping from a local
	// coordinate zeta to a global position r. Meant to be subclassed from Python (see
	// PyGeomObject below) to describe e.g. curved mesh boundaries.
	class GeomObject : public oomph::GeomObject
	{
	public:
		void position(const oomph::Vector<double> &zeta, oomph::Vector<double> &r) const override
		{
			throw_runtime_error("GeomObject::position not specialised");
		}
	};

	// pybind11 trampoline: forwards the virtual position() call to a Python override, so a
	// Python class deriving from GeomObject can implement the actual parametrization.
	class PyGeomObject : public GeomObject
	{
	public:
		using GeomObject::GeomObject;

		void position(const oomph::Vector<double> &zeta, oomph::Vector<double> &r) const override
		{
			PYBIND11_OVERLOAD(void, GeomObject, position, zeta, r);
		}
	};

	// Base class for a (macro-element) domain, i.e. a collection of macro elements together with
	// the mapping from each macro element's local coordinate s to a boundary position f, used by
	// oomph-lib's macro-element/Q-mapping to build curved meshes. Meant to be subclassed from
	// Python (see PyDomain below); the numpy-array-based macro_element_boundary() overload is the
	// one exposed to/overridden by Python, while the oomph::Vector-based overload (required by
	// the oomph-lib base class interface) just marshals data into/out of small reusable numpy
	// buffers (PyS, PyF) and forwards to it, to avoid reallocating on every call.
	class Domain : public oomph::Domain
	{
	protected:
		py::array_t<double> PyS, PyF;
		py::buffer_info PyS_buff, PyF_buff;

	public:
		Domain() : oomph::Domain(), PyS(1), PyF(1)
		{
			PyS_buff = PyS.request();
			PyF_buff = PyF.request();
		}
		// Overridden (from Python, via PyDomain) to compute the global position f on the
		// boundary i_direct of macro element i_macro, at local coordinate s and history time t.
		virtual void macro_element_boundary(const unsigned &t, const unsigned &i_macro, const unsigned &i_direct, const py::array_t<double> &s, py::array_t<double> &f)
		{
			throw_runtime_error("Domain::macro_element_boundary not specialised");
		}
		using oomph::Domain::macro_element_boundary;
		// oomph-lib-facing overload: copies s/f between oomph::Vector<double> and the reusable
		// numpy buffers PyS/PyF (resizing them only if the dimensionality changed), then calls
		// the numpy-array-based overload above.
		void macro_element_boundary(const unsigned &t, const unsigned &i_macro, const unsigned &i_direct, const oomph::Vector<double> &s, oomph::Vector<double> &f) override
		{
			if (PyS_buff.shape[0] != (int)s.size())
			{
				PyS.resize({s.size()});
				PyS_buff = PyS.request();
			}
			for (unsigned int i = 0; i < s.size(); i++)
				((double *)(PyS_buff.ptr))[i] = s[i];

			if (PyF_buff.shape[0] != (int)f.size())
			{
				PyF.resize({f.size()});
				PyF_buff = PyF.request();
			}
			macro_element_boundary(t, i_macro, i_direct, PyS, PyF);
			for (unsigned int i = 0; i < f.size(); i++)
				f[i] = ((double *)(PyF_buff.ptr))[i];
		}
	};

	// pybind11 trampoline: forwards the virtual macro_element_boundary() call to a Python
	// override.
	class PyDomain : public Domain
	{
	public:
		using Domain::Domain;
		void macro_element_boundary(const unsigned &t, const unsigned &i_macro, const unsigned &i_direct, const py::array_t<double> &s, py::array_t<double> &f) override
		{
			PYBIND11_OVERLOAD(void, Domain, macro_element_boundary, t, i_macro, i_direct, s, f);
		}
	};

}

void PyReg_GeomObjects(py::module &m)
{

	py::class_<pyoomph::GeomObject, pyoomph::PyGeomObject>(
		m, "GeomObject",
		"Base class for a parametric geometric object (a mapping from a local coordinate to a global "
		"position), to be subclassed in Python and used e.g. to describe curved mesh boundaries via "
		"an overridden position(zeta) method.")
		.def(py::init<>());

	py::class_<oomph::Domain>(m, "OomphDomain",
							   "Base class (from oomph-lib) for a macro-element domain.");
	py::class_<pyoomph::Domain, pyoomph::PyDomain, oomph::Domain>(
		m, "Domain",
		"Base class for a macro-element domain, to be subclassed in Python and used to describe "
		"curved meshes via an overridden macro_element_boundary(t, i_macro, i_direct, s, f) method.")
		.def(py::init<>());

	py::class_<oomph::MacroElement>(m, "MacroElement",
									 "Base class (from oomph-lib) for a single macro element of a Domain.");
	py::class_<oomph::QMacroElement<2>, oomph::MacroElement>(
		m, "QMacroElement2",
		"A 2d quadrilateral macro element belonging to a Domain, identified by its index therein.")
		.def(py::init<oomph::Domain *, const unsigned &>(), py::arg("domain"), py::arg("macro_element_index"));
}
