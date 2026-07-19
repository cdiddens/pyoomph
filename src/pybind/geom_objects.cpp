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
#include <nanobind/ndarray.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>

namespace nb = nanobind;

#include <sstream>
#include <vector>

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

	// nanobind trampoline: forwards the virtual position() call to a Python override, so a
	// Python class deriving from GeomObject can implement the actual parametrization.
	class PyGeomObject : public GeomObject
	{
	public:
		NB_TRAMPOLINE(GeomObject, 1);

		void position(const oomph::Vector<double> &zeta, oomph::Vector<double> &r) const override
		{
			NB_OVERRIDE(position, zeta, r);
		}
	};

	// Base class for a (macro-element) domain, i.e. a collection of macro elements together with
	// the mapping from each macro element's local coordinate s to a boundary position f, used by
	// oomph-lib's macro-element/Q-mapping to build curved meshes. Meant to be subclassed from
	// Python (see PyDomain below); the numpy-array-based macro_element_boundary() overload is the
	// one exposed to/overridden by Python, while the oomph::Vector-based overload (required by
	// the oomph-lib base class interface) just marshals data into/out of small reusable buffers
	// (PyS, PyF), viewed as numpy arrays only for the duration of the Python call, and forwards
	// to it, to avoid reallocating on every call.
	class Domain : public oomph::Domain
	{
	protected:
		std::vector<double> PyS, PyF;

	public:
		Domain() : oomph::Domain(), PyS(1), PyF(1)
		{
		}
		// Overridden (from Python, via PyDomain) to compute the global position f on the
		// boundary i_direct of macro element i_macro, at local coordinate s and history time t.
		virtual void macro_element_boundary(const unsigned &t, const unsigned &i_macro, const unsigned &i_direct, const nb::ndarray<nb::numpy, double> &s, nb::ndarray<nb::numpy, double> &f)
		{
			throw_runtime_error("Domain::macro_element_boundary not specialised");
		}
		using oomph::Domain::macro_element_boundary;
		// oomph-lib-facing overload: copies s/f between oomph::Vector<double> and the reusable
		// buffers PyS/PyF (resizing them only if the dimensionality changed), wraps them as
		// (non-owning) numpy array views for the duration of the call, then calls the
		// numpy-array-based overload above.
		void macro_element_boundary(const unsigned &t, const unsigned &i_macro, const unsigned &i_direct, const oomph::Vector<double> &s, oomph::Vector<double> &f) override
		{
			if (PyS.size() != s.size())
				PyS.resize(s.size());
			for (unsigned int i = 0; i < s.size(); i++)
				PyS[i] = s[i];

			if (PyF.size() != f.size())
				PyF.resize(f.size());

			nb::ndarray<nb::numpy, double> PyS_view(PyS.data(), {PyS.size()});
			nb::ndarray<nb::numpy, double> PyF_view(PyF.data(), {PyF.size()});
			macro_element_boundary(t, i_macro, i_direct, PyS_view, PyF_view);
			for (unsigned int i = 0; i < f.size(); i++)
				f[i] = PyF[i];
		}
	};

	// nanobind trampoline: forwards the virtual macro_element_boundary() call to a Python
	// override.
	class PyDomain : public Domain
	{
	public:
		NB_TRAMPOLINE(Domain, 1);

		void macro_element_boundary(const unsigned &t, const unsigned &i_macro, const unsigned &i_direct, const nb::ndarray<nb::numpy, double> &s, nb::ndarray<nb::numpy, double> &f) override
		{
			NB_OVERRIDE(macro_element_boundary, t, i_macro, i_direct, s, f);
		}
	};

}

void PyReg_GeomObjects(nb::module_ &m)
{

	nb::class_<pyoomph::GeomObject, pyoomph::PyGeomObject>(
		m, "GeomObject",
		"Base class for a parametric geometric object (a mapping from a local coordinate to a global "
		"position), to be subclassed in Python and used e.g. to describe curved mesh boundaries via "
		"an overridden position(zeta) method.")
		.def(nb::init<>());

	nb::class_<oomph::Domain>(m, "OomphDomain",
							   "Base class (from oomph-lib) for a macro-element domain.");
	nb::class_<pyoomph::Domain, pyoomph::PyDomain, oomph::Domain>(
		m, "Domain",
		"Base class for a macro-element domain, to be subclassed in Python and used to describe "
		"curved meshes via an overridden macro_element_boundary(t, i_macro, i_direct, s, f) method.")
		.def(nb::init<>());

	nb::class_<oomph::MacroElement>(m, "MacroElement",
									 "Base class (from oomph-lib) for a single macro element of a Domain.");
	nb::class_<oomph::QMacroElement<2>, oomph::MacroElement>(
		m, "QMacroElement2",
		"A 2d quadrilateral macro element belonging to a Domain, identified by its index therein.")
		.def(nb::init<oomph::Domain *, const unsigned &>(), nb::arg("domain"), nb::arg("macro_element_index"));
}
