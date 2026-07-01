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

namespace py = pybind11;

#include <sstream>

#include "../oomph_lib.hpp"
#include "../exception.hpp"
namespace pyoomph
{

	class GeomObject : public oomph::GeomObject
	{
	public:
		void position(const oomph::Vector<double> &zeta, oomph::Vector<double> &r) const
		{
			throw_runtime_error("GeomObject::position not specialised");
		}
	};

	class PyGeomObject : public GeomObject
	{
	public:
		using GeomObject::GeomObject;

		void position(const oomph::Vector<double> &zeta, oomph::Vector<double> &r) const
		{
			PYBIND11_OVERLOAD(void, GeomObject, position, zeta, r);
		}
	};

	///

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
		virtual void macro_element_boundary(const unsigned &t, const unsigned &i_macro, const unsigned &i_direct, const py::array_t<double> &s, py::array_t<double> &f)
		{
			throw_runtime_error("Domain::macro_element_boundary not specialised");
		}

		void macro_element_boundary(const unsigned &t, const unsigned &i_macro, const unsigned &i_direct, const oomph::Vector<double> &s, oomph::Vector<double> &f)
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

	class PyDomain : public Domain
	{
	public:
		using Domain::Domain;
		void macro_element_boundary(const unsigned &t, const unsigned &i_macro, const unsigned &i_direct, const py::array_t<double> &s, py::array_t<double> &f)
		{
			PYBIND11_OVERLOAD(void, Domain, macro_element_boundary, t, i_macro, i_direct, s, f);
		}
	};

}

void PyReg_GeomObjects(py::module &m)
{

	py::class_<pyoomph::GeomObject, pyoomph::PyGeomObject>(m, "GeomObject")
		.def(py::init<>());

	py::class_<oomph::Domain>(m, "OomphDomain");
	py::class_<pyoomph::Domain, pyoomph::PyDomain, oomph::Domain>(m, "Domain")
		.def(py::init<>());

	py::class_<oomph::MacroElement>(m, "MacroElement");
	py::class_<oomph::QMacroElement<2>, oomph::MacroElement>(m, "QMacroElement2")
		.def(py::init<oomph::Domain *, const unsigned &>());
}
