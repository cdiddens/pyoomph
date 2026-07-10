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

namespace py = pybind11;

#include "../oomph_lib.hpp"
#include "../timestepper.hpp"

void PyReg_TimeStepper(py::module &m)
{

	py::class_<oomph::Time>(m, "Time")
		.def("time", (double(oomph::Time::*)(const unsigned &) const) & oomph::Time::time)
		.def("time", (double &(oomph::Time::*)()) & oomph::Time::time)
		.def("set_time", [](oomph::Time &self, double t)
			 { self.time() = t; })
		.def("ndt", [](oomph::Time &self)
			 { return self.ndt(); })
		.def("dt", (double(oomph::Time::*)(const unsigned &) const) & oomph::Time::dt)
		.def("set_dt", [](oomph::Time *self, const unsigned &index, const double &v)
			 { self->dt(index) = v; });

	py::class_<oomph::TimeStepper>(m, "TimeStepper")
		.def("make_steady", [](oomph::TimeStepper &self)
			 { self.make_steady(); })
		.def("time_pt", (oomph::Time * &(oomph::TimeStepper::*)()) & oomph::TimeStepper::time_pt, py::return_value_policy::reference)
		.def("undo_make_steady", [](oomph::TimeStepper &self)
			 { self.undo_make_steady(); })
		.def("is_steady", [](oomph::TimeStepper &self)
			 { return self.is_steady(); })
		.def("set_weights", [](oomph::TimeStepper &self)
			 { return self.set_weights(); })
		.def("ntstorage", &oomph::TimeStepper::ntstorage)
		.def("nprev_values", &oomph::TimeStepper::nprev_values);

	py::class_<pyoomph::MultiTimeStepper, oomph::TimeStepper>(m, "MultiTimeStepper")
		.def("get_num_unsteady_steps_done", &pyoomph::MultiTimeStepper::get_num_unsteady_steps_done)
		.def("weightBDF1",&pyoomph::MultiTimeStepper::weightBDF1)
		.def("weightBDF2",&pyoomph::MultiTimeStepper::weightBDF2)
		.def("weightNewmark2",&pyoomph::MultiTimeStepper::weightNewmark2)						
		.def("set_Newmark2_coeffs",&pyoomph::MultiTimeStepper::setNewmark2Coeffs)						
		.def("set_num_unsteady_steps_done", &pyoomph::MultiTimeStepper::set_num_unsteady_steps_done)
		.def("increment_num_unsteady_steps_done", &pyoomph::MultiTimeStepper::increment_num_unsteady_steps_done)
		.def(py::init<bool>());

}
