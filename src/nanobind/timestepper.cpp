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

#include "../oomph_lib.hpp"
#include "../timestepper.hpp"

void PyReg_TimeStepper(nb::module_ &m)
{

	nb::class_<oomph::Time>(
		m, "Time",
		"Holds the current continuous simulation time and the history of previous time step sizes (dt), "
		"shared by all TimeStepper objects of a Problem (although we only have one time stepper).")		
		.def("time", (double(oomph::Time::*)(const unsigned &) const) & oomph::Time::time, nb::arg("t"),
			 "Return the continuous time value at history level ``t`` (t=0: current time, t>0: the time ``t`` steps ago).")
		.def("time", (double &(oomph::Time::*)()) & oomph::Time::time,
			 "Return the current continuous nondimensional time value (equivalent to time(0)).")
		.def(
			"set_time", [](oomph::Time &self, double t)
			{ self.time() = t; },
			nb::arg("t"), "Set the current continuous nondimensional time value.")
		.def(
			"ndt", [](oomph::Time &self)
			{ return self.ndt(); },
			"Return the number of previous time step sizes stored.")
		.def("dt", (double(oomph::Time::*)(const unsigned &) const) & oomph::Time::dt, nb::arg("t") = 0,
			 "Return the value of the nondimensional  time steps ``t``-th stored time step size (t=0: current/most recent step size, t>0: further in the past).")
		.def(
			"set_dt", [](oomph::Time *self, const unsigned &index, const double &v)
			{ self->dt(index) = v; },
			nb::arg("index"), nb::arg("value"), "Set the value of the ``index``-th stored nondimensional time step size.");

	nb::class_<oomph::TimeStepper>(
		m, "TimeStepper",
		"Base class (from oomph-lib) for time-stepping schemes: approximates the time derivatives of Data as a "
		"weighted sum of its current and historically stored values, i.e. weight(i,t) is the contribution of the "
		"t-th stored history value to the i-th time derivative.")
		.def(
			"make_steady", [](oomph::TimeStepper &self)
			{ self.make_steady(); },
			"Temporarily deactivate all time-dependence by setting all weights to zero (except the weight of the "
			"current value), so the time stepper effectively solves a steady-state problem. Reversible via undo_make_steady().")
		.def("time_pt", (oomph::Time * &(oomph::TimeStepper::*)()) & oomph::TimeStepper::time_pt, nb::rv_policy::reference,
			 "Return the Time object (current time and stored time step sizes) associated with this time stepper.")
		.def(
			"undo_make_steady", [](oomph::TimeStepper &self)
			{ self.undo_make_steady(); },
			"Restore the normal (unsteady) weights after a previous make_steady() call.")
		.def(
			"is_steady", [](oomph::TimeStepper &self)
			{ return self.is_steady(); },
			"Whether this time stepper has been temporarily made steady via make_steady().")
		.def(
			"set_weights", [](oomph::TimeStepper &self)
			{ return self.set_weights(); },
			"(Re-)compute the finite-difference weights used to approximate time derivatives from the current and "
			"stored time step sizes; must be called whenever a time step size changes.")
		.def("ntstorage", &oomph::TimeStepper::ntstorage,
			 "Return the number of history values (doubles) stored per degree of freedom to represent its time derivatives (1 for a steady/static time stepper).")
		.def("nprev_values", &oomph::TimeStepper::nprev_values,
			 "Return the number of history values that represent actual previous values of the degree of freedom, as opposed to "
			 "other stored quantities such as previous derivatives (0 for a static time stepper, 1 for BDF1, ...).");

	nb::class_<pyoomph::MultiTimeStepper, oomph::TimeStepper>(
		m, "MultiTimeStepper",
		"pyoomph time stepper that simultaneously evaluates the weights of several schemes (BDF1, BDF2 and Newmark2) "
		"on the same stored history, so that e.g. the temporal error can be estimated via BDF2 while a different "
		"scheme is actually used to advance the solution.")
		.def("get_num_unsteady_steps_done", &pyoomph::MultiTimeStepper::get_num_unsteady_steps_done,
			 "Return how many unsteady (non-steady) time steps have been taken so far with this time stepper; used to "
			 "'degrade' the scheme order during start-up (e.g. use BDF1 for the very first step(s), then BDF2).")
		.def("weightBDF1", &pyoomph::MultiTimeStepper::weightBDF1, nb::arg("i"), nb::arg("j"),
			 "Return the BDF1 (first order backward differentiation) weight of history value ``j`` for the ``i``-th time derivative.")
		.def("weightBDF2", &pyoomph::MultiTimeStepper::weightBDF2, nb::arg("i"), nb::arg("j"),
			 "Return the BDF2 (second order backward differentiation) weight of history value ``j`` for the ``i``-th time derivative.")
		.def("weightNewmark2", &pyoomph::MultiTimeStepper::weightNewmark2, nb::arg("i"), nb::arg("j"),
			 "Return the Newmark2 weight of history value ``j`` for the ``i``-th time derivative.")
		.def("set_Newmark2_coeffs", &pyoomph::MultiTimeStepper::setNewmark2Coeffs, nb::arg("beta1"), nb::arg("beta2"),
			 "Set the two Newmark-beta coefficients used to compute the Newmark2 weights.")
		.def("set_num_unsteady_steps_done", &pyoomph::MultiTimeStepper::set_num_unsteady_steps_done, nb::arg("n"),
			 "Directly set the internal counter of unsteady time steps taken so far (see get_num_unsteady_steps_done()).")
		.def("increment_num_unsteady_steps_done", &pyoomph::MultiTimeStepper::increment_num_unsteady_steps_done,
			 "Increment the internal counter of unsteady time steps taken so far by one.")
		.def(nb::init<bool>(), nb::arg("adaptive") = false,
			 "Create a MultiTimeStepper. If ``adaptive`` is True, additional storage for a predictor step and "
			 "temporal error estimation (based on the BDF2 weights) is allocated.");

}
