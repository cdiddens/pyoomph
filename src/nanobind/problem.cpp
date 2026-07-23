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
#include <nanobind/trampoline.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <fstream>
#include <tuple>

#include "nb_array_utils.hpp"

namespace nb = nanobind;

#include <fstream>

#include "../problem.hpp"
#include "../bifurcation.hpp"
#include "../expressions.hpp"
#include "../mesh.hpp"
#include "mesh_handle.hpp"
#include "../elements.hpp"
#include "../codegen.hpp"
#include "../ccompiler.hpp"
#include "../logging.hpp"
#ifdef __gnu_linux__
#include <fenv.h>
#include "problem.hpp"
#endif

namespace pyoomph
{

	// Trampoline class forwarding the virtual hook functions of pyoomph::Problem to Python,
	// so that a Python class derived from Problem can override e.g. setup_pinning() or
	// set_initial_condition() and have the C++ core call back into Python.
	class PyProblemTrampoline : public pyoomph::Problem
	{
	public:
		NB_TRAMPOLINE(pyoomph::Problem, 16);

		void setup_pinning() override
		{
			NB_OVERRIDE(setup_pinning);
		}

		void set_initial_condition() override
		{
			NB_OVERRIDE(set_initial_condition);
		}

		std::pair<unsigned, unsigned> _adapt() override
		{
			NB_OVERRIDE(_adapt);
		}

		void actions_before_newton_solve() override
		{
			NB_OVERRIDE(actions_before_newton_solve);
		}

		void actions_after_newton_solve() override
		{
			NB_OVERRIDE(actions_after_newton_solve);
		}

		void actions_before_newton_convergence_check() override
		{
			NB_OVERRIDE(actions_before_newton_convergence_check);
		}

		void actions_after_change_in_global_parameter(const std::string &paramname) override
		{
			NB_OVERRIDE(actions_after_change_in_global_parameter, paramname);
		}

		void actions_after_parameter_increase(const std::string &paramname) override
		{
			NB_OVERRIDE(actions_after_parameter_increase, paramname);
		}

		void actions_after_newton_step() override
		{
			NB_OVERRIDE(actions_after_newton_step);
		}

		void actions_before_newton_step() override
		{
			NB_OVERRIDE(actions_before_newton_step);
		}

		void actions_before_adapt() override
		{
			NB_OVERRIDE(actions_before_adapt);
		}

		void actions_after_adapt() override
		{
			NB_OVERRIDE(actions_after_adapt);
		}

		void actions_before_distribute() override
		{
			NB_OVERRIDE(actions_before_distribute);
		}

		void actions_after_distribute() override
		{
			NB_OVERRIDE(actions_after_distribute);
		}

		void get_custom_residuals_jacobian(pyoomph::CustomResJacInformation *info) override
		{
			NB_OVERRIDE(get_custom_residuals_jacobian, info);
		}

		void _build_mesh() override
		{
			NB_OVERRIDE(_build_mesh);
		}
	};

}

static nb::class_<GiNaC::GiNaCGlobalParameterWrapper> *py_decl_GlobalParam = NULL;
// static nb::class_<pyoomph::DofAugmentations> *py_decl_DofAugmentations=NULL;
void PyDecl_Problem(nb::module_ &m)
{
	// Forward-declared here (before PyReg_Problem runs) since other translation units may need
	// to refer to the "GiNaC_GlobalParam" Python type before all nanobind classes are registered.
	py_decl_GlobalParam = new nb::class_<GiNaC::GiNaCGlobalParameterWrapper>(
		m, "GiNaC_GlobalParam",
		"A global (scalar) parameter of a Problem, wrapped so that it can be used directly inside "
		"symbolic (GiNaC) expressions, e.g. as a bifurcation-tracking or arclength-continuation parameter. "
		"Obtained via Problem.get_global_parameter().");
	// py_decl_DofAugmentations=new nb::class_<pyoomph::DofAugmentations>(m,"DofAugmentations");
}

void PyReg_Problem(nb::module_ &m)
{

	m.def(
		"InitMPI", [](std::vector<std::string> &argv)
		{
			std::vector<char *> c_argv(argv.size() + 1, NULL); // +1 for the trailing NULL sentinel, as in a real argv
			for (unsigned int i = 0; i < argv.size(); i++)
			{
				unsigned l = strlen(argv[i].c_str());
				c_argv[i] = (char *)malloc(sizeof(char) * (l + 1));
				strncpy(c_argv[i], argv[i].c_str(), l);
				c_argv[i][l] = '\0';
			}
			int c_argc = argv.size();
			oomph::MPI_Helpers::init(c_argc, &(c_argv[0]));
			for (unsigned int i = 0; i < argv.size(); i++)
			{
				free(c_argv[i]);
			}
		},
		nb::arg("argv"),
		"Initialize MPI. Must be called (if at all) before any other MPI-related functionality is used. "
		"``argv`` is the list of command line arguments, conventionally with the program name as its first entry.");

	m.def("FinaliseMPI", &oomph::MPI_Helpers::finalize,
		  "Finalize MPI. Must be called once before the program exits if InitMPI() was called.");

	m.def("_write_to_log_file", &pyoomph::write_to_log_file, nb::arg("message"),
		  "Write ``message`` to the currently open problem log file (see Problem._open_log_file()); does nothing if no log file is open.");

	m.def(
		"_get_core_information", []()
		{
			std::map<std::string, std::string> info;
			/*#ifdef VERSION_INFO
				info["core_version"]=VERSION_INFO;
			#endif*/
			return info;
		},
		"Return miscellaneous information about the compiled C++ core (e.g. build/version information) as a dict of strings.");

	m.def("get_verbosity_flag", []()
		  { return pyoomph::pyoomph_verbose; }, "Return the current verbosity level of the C++ core (0: quiet).");
	m.def(
		"set_verbosity_flag", [](int level)
		{ pyoomph::pyoomph_verbose = level; },
		nb::arg("level"),
		"Set the verbosity level of the C++ core (0: quiet, higher values print more diagnostic information).");

	m.def(
		"feenableexcept", []()
		{
#ifdef __gnu_linux__
			feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#else
			throw_runtime_error("feenableexcept not supported on this OS");
#endif
		},
		"Enable floating point exceptions (SIGFPE) on division-by-zero, invalid operations and overflow. "
		"Useful to trap the origin of NaN/Inf values in a debugger. Only supported on Linux.");


	// py_decl_DofAugmentations->
	nb::class_<pyoomph::DofAugmentations>(
		m, "DofAugmentations",
		"Collects extra ('augmented') degrees of freedom that are appended to the regular degrees of freedom "
		"of a Problem, e.g. eigenvector components or Lagrange multipliers required for bifurcation tracking "
		"or arclength continuation. Create one via Problem._create_dof_augmentation() and pass it to Problem._add_augmented_dofs().")
		.def("add_vector", &pyoomph::DofAugmentations::add_vector, nb::arg("values"),
			 "Append a vector-valued block of augmented degrees of freedom (e.g. an eigenvector), initialized with ``values``. Returns the start index of this block in the augmented dof vector.")
		.def("add_scalar", &pyoomph::DofAugmentations::add_scalar, nb::arg("value"),
			 "Append a single scalar augmented degree of freedom, initialized with ``value``. Returns its index in the augmented dof vector.")
		.def("add_parameter", &pyoomph::DofAugmentations::add_parameter, nb::arg("parameter_name"),
			 "Append the global parameter ``parameter_name`` as an augmented degree of freedom (e.g. for arclength continuation). Returns its index in the augmented dof vector.")
		.def(
			"split", [](pyoomph::DofAugmentations &augs, unsigned startindex, int endindex)
			{
				std::vector<std::vector<double>> res = augs.split(startindex, endindex);
				std::vector<nb::ndarray<nb::numpy, double>> resA(res.size());
				for (unsigned int i = 0; i < res.size(); i++)
					resA[i] = vector_to_ndarray(res[i]);
				return resA;
			},
			nb::arg("startindex") = 1, nb::arg("endindex") = -1,
			"Split the augmented dof vector into its individual components, one array per add_vector()/add_scalar()/add_parameter() "
			"call. By default (startindex=1), only the augmented dofs are returned, skipping the base problem dofs at index 0; "
			"endindex=-1 means up to and including the last block.");

	py_decl_GlobalParam
		->def_prop_rw(
			"value",
			[](GiNaC::GiNaCGlobalParameterWrapper *self)
			{ return self->get_struct().cme->value(); },
			[](GiNaC::GiNaCGlobalParameterWrapper *self, const double &v)
			{
				if (v < 0 && self->get_struct().cme->is_restricted_to_positive_values())
					throw_runtime_error("Cannot set the parameter " + self->get_struct().cme->get_name() + " to a negative value of " + std::to_string(v) + " since it is restricted to positive values.");
				self->get_struct().cme->value() = v;
			},
			"The current numerical value of this global parameter.")
		.def_prop_rw(
			"analytical_derivative",
			[](GiNaC::GiNaCGlobalParameterWrapper *self)
			{ return self->get_struct().cme->get_analytic_derivative(); },
			[](GiNaC::GiNaCGlobalParameterWrapper *self, const bool &v)
			{ self->get_struct().cme->set_analytic_derivative(v); },
			"Whether derivatives of the residuals with respect to this parameter are computed analytically via generated code (True) or by finite differences (False).")
		.def(
			"get_symbol", [](GiNaC::GiNaCGlobalParameterWrapper *self) -> GiNaC::ex
			{ return 0 + (*self); },
			"Return this parameter as a plain GiNaC symbolic expression, detached from the wrapper.")
		.def(
			"get_name", [](GiNaC::GiNaCGlobalParameterWrapper *self)
			{ return self->get_struct().cme->get_name(); },
			"Return the name of this global parameter.")
		.def(
			"restrict_to_positive_values", [](GiNaC::GiNaCGlobalParameterWrapper *self)
			{ self->get_struct().cme->restrict_to_positive_values(); },
			"Mark this parameter as restricted to non-negative values; assigning it a negative value afterwards raises a RuntimeError.")

		.def(-nb::self, "Symbolic negation, returns a GiNaC expression.")

		.def(nb::self + nb::self, "Symbolic addition of two global parameters, returns a GiNaC expression.")
		.def(nb::self + GiNaC::ex(), "Symbolic addition with a GiNaC expression, returns a GiNaC expression.")
		.def(int() + nb::self, "Symbolic addition with an int, returns a GiNaC expression.")
		.def(nb::self + int(), "Symbolic addition with an int, returns a GiNaC expression.")
		.def(float() + nb::self, "Symbolic addition with a float, returns a GiNaC expression.")
		.def(nb::self + float(), "Symbolic addition with a float, returns a GiNaC expression.")

		.def(nb::self - nb::self, "Symbolic subtraction of two global parameters, returns a GiNaC expression.")
		.def(nb::self - GiNaC::ex(), "Symbolic subtraction with a GiNaC expression, returns a GiNaC expression.")
		.def(int() - nb::self, "Symbolic subtraction with an int, returns a GiNaC expression.")
		.def(nb::self - int(), "Symbolic subtraction with an int, returns a GiNaC expression.")
		.def(float() - nb::self, "Symbolic subtraction with a float, returns a GiNaC expression.")
		.def(nb::self - float(), "Symbolic subtraction with a float, returns a GiNaC expression.")

		.def(nb::self * nb::self, "Symbolic multiplication of two global parameters, returns a GiNaC expression.")
		.def(nb::self * GiNaC::ex(), "Symbolic multiplication with a GiNaC expression, returns a GiNaC expression.")
		.def(int() * nb::self, "Symbolic multiplication with an int, returns a GiNaC expression.")
		.def(nb::self * int(), "Symbolic multiplication with an int, returns a GiNaC expression.")
		.def(float() * nb::self, "Symbolic multiplication with a float, returns a GiNaC expression.")
		.def(nb::self * float(), "Symbolic multiplication with a float, returns a GiNaC expression.")

		.def(nb::self / nb::self, "Symbolic division of two global parameters, returns a GiNaC expression.")
		.def(nb::self / GiNaC::ex(), "Symbolic division with a GiNaC expression, returns a GiNaC expression.")
		.def(int() / nb::self, "Symbolic division with an int, returns a GiNaC expression.")
		.def(nb::self / int(), "Symbolic division with an int, returns a GiNaC expression.")
		.def(float() / nb::self, "Symbolic division with a float, returns a GiNaC expression.")
		.def(nb::self / float(), "Symbolic division with a float, returns a GiNaC expression.")

		.def(
			"__repr__", [](const GiNaC::GiNaCGlobalParameterWrapper &self)
			{
				std::ostringstream oss;
				GiNaC::print_python pypc(oss);
				(self + 0).print(pypc);
				return oss.str();
			},
			"Return the Python-style string representation of the parameter's symbol.")

		.def(
			"__pow__", [](const GiNaC::GiNaCGlobalParameterWrapper &lh, const GiNaC::ex &rh)
			{ return GiNaC::pow(lh, rh); },
			nb::is_operator(), "Symbolic exponentiation with a GiNaC expression exponent, returns a GiNaC expression.")
		.def(
			"__pow__", [](const GiNaC::GiNaCGlobalParameterWrapper &lh, const int &rh)
			{ return GiNaC::pow(lh, rh); },
			nb::is_operator(), "Symbolic exponentiation with an int exponent, returns a GiNaC expression.")
		.def(
			"__pow__", [](const GiNaC::GiNaCGlobalParameterWrapper &lh, const double &rh)
			{ return GiNaC::pow(lh, rh); },
			nb::is_operator(), "Symbolic exponentiation with a float exponent, returns a GiNaC expression.")

		.def(
			"__rpow__", [](const GiNaC::GiNaCGlobalParameterWrapper &rh, const int &lh)
			{ return GiNaC::pow(lh, rh); },
			nb::is_operator(), "Symbolic exponentiation with an int base and this parameter as exponent, returns a GiNaC expression.")
		.def(
			"__rpow__", [](const GiNaC::GiNaCGlobalParameterWrapper &rh, const double &lh)
			{ return GiNaC::pow(lh, rh); },
			nb::is_operator(), "Symbolic exponentiation with a float base and this parameter as exponent, returns a GiNaC expression.")
		.def(
			"__float__", [](const GiNaC::GiNaCGlobalParameterWrapper &self)
			{ return self.get_struct().cme->value(); },
			"Return the current numerical value of this parameter as a Python float (same as the ``value`` property).")

		;

	nb::class_<pyoomph::SparseRank3Tensor>(
		m, "SparseRank3Tensor",
		"Sparse rank-3 tensor storing the Hessian of the residuals with respect to the degrees of freedom, "
		"i.e. entry (i,j,k) corresponds to d^2 R_i / (du_j du_k). Can optionally be symmetric in the last two indices. "
		"Obtained via Problem._assemble_hessian_tensor().")
		.def("get_entries", &pyoomph::SparseRank3Tensor::get_entries,
			 "Return all non-zero entries as a list of (i, j, k, value) tuples.")
		.def(
			"finalize_for_vector_product", [](pyoomph::SparseRank3Tensor &self) -> std::tuple<nb::ndarray<nb::numpy, int>, nb::ndarray<nb::numpy, int>>
			{
				std::vector<int> col_index;
				std::vector<int> row_start;
				std::tie(col_index, row_start) = self.finalize_for_vector_product();
				return std::make_tuple(vector_to_ndarray(col_index), vector_to_ndarray(row_start));
			},
			"Build the compressed-row-storage (CSR) index arrays (column_index, row_start) required by right_vector_mult(); "
			"must be called once before right_vector_mult() is used.")
		.def(
			"right_vector_mult", [](pyoomph::SparseRank3Tensor &self, const nb::ndarray<nb::numpy, double> &vec) -> nb::ndarray<nb::numpy, double>
			{
				std::vector<double> v = ndarray_to_vector(vec);
				std::vector<double> vals = self.right_vector_mult(v);
				return vector_to_ndarray(vals);
			},
			nb::arg("vector"),
			"Contract the tensor with ``vector`` over its last index. Used e.g. to assemble directional Hessian-vector products; "
			"requires finalize_for_vector_product() to have been called beforehand.");

	nb::class_<pyoomph::CustomResJacInformation>(
		m, "CustomResJacInfo",
		"Passed to Problem.get_custom_residuals_jacobian() to hand a fully custom (Python-assembled) residual vector, "
		"and if requested a Jacobian matrix, back to the C++ core. Used when Problem.use_custom_residual_jacobian is set.")
		.def("require_jacobian", &pyoomph::CustomResJacInformation::require_jacobian,
			 "Whether the Jacobian is required in addition to the residual vector for the current assembly.")
		.def("get_parameter_name", &pyoomph::CustomResJacInformation::get_parameter_name,
			 "Name of the global parameter with respect to which a derivative is requested, or an empty string if the plain residual/Jacobian is requested.")
		.def(
			"set_custom_residuals", [](pyoomph::CustomResJacInformation &self, nb::ndarray<nb::numpy, double> r)
			{
				std::vector<double> R = ndarray_to_vector(r);
				self.set_custom_residuals(R);
			},
			nb::arg("residuals"), "Set the custom residual vector (length must match Problem.ndof()).")
		.def(
			"set_custom_jacobian", [](pyoomph::CustomResJacInformation &self, nb::ndarray<nb::numpy, double> jvals, nb::ndarray<nb::numpy, int> colindex, nb::ndarray<nb::numpy, int> rowstart)
			{
				std::vector<double> V = ndarray_to_vector(jvals);
				std::vector<int> I = ndarray_to_vector(colindex);
				std::vector<int> J = ndarray_to_vector(rowstart);

				self.set_custom_jacobian(V, I, J);
			},
			nb::arg("values"), nb::arg("column_index"), nb::arg("row_start"),
			"Set the custom Jacobian matrix in compressed sparse row (CSR) format: ``values`` are the non-zero entries, "
			"``column_index`` their column indices and ``row_start`` the row offsets into ``values``/``column_index`` (length ndof()+1).");

	nb::class_<oomph::AssemblyHandler>(
		m, "AssemblyHandler",
		"Base class (from oomph-lib) for objects that override how residuals and Jacobians of a Problem are assembled, "
		"e.g. to add the extra equations required for bifurcation tracking or periodic orbit continuation.");

	nb::class_<pyoomph::MyFoldHandler, oomph::AssemblyHandler>(
		m, "FoldHandler",
		"AssemblyHandler activated during fold (limit point / saddle-node) bifurcation tracking. Augments the "
		"system with the null-eigenvector equations and a normalization condition needed to track a fold in a control parameter.")
		.def(
			"get_eigenfunction", [](pyoomph::MyFoldHandler *self) -> nb::ndarray<nb::numpy, double>
			{
				oomph::Vector<oomph::DoubleVector> efuncs;
				self->get_eigenfunction(efuncs);

				size_t n_eigvec = efuncs.size(), ndof = self->get_problem_ndof();
				double *dest = new double[n_eigvec * ndof];
				double *ptr = dest;

				for (unsigned int i = 0; i < efuncs.size(); i++)
				{
					for (unsigned int j = 0; j < self->get_problem_ndof(); j++)
					{
						*ptr = efuncs[i][j];
						ptr++;
					}
				}
				nb::capsule owner(dest, [](void *p) noexcept { delete[] (double *) p; });
				return nb::ndarray<nb::numpy, double>(dest, {n_eigvec, ndof}, owner);
			},
			"Return the null eigenvector(s) associated with the fold, as an array of shape (n_eigenvectors, ndof).")
		.def_prop_rw(
			"FD_step",
			[](pyoomph::MyFoldHandler *h)
			{ return h->FD_step; },
			[](pyoomph::MyFoldHandler *h, double s)
			{ h->FD_step = s; },
			"Step size used for the finite-difference derivatives required internally by the fold tracking equations.")
		.def("get_base_ndof", &pyoomph::MyFoldHandler::get_problem_ndof,
			 "Number of degrees of freedom of the underlying (non-augmented) problem.")
		.def("set_eigenweight", &pyoomph::MyFoldHandler::set_eigenweight, nb::arg("weight"),
			 "Set the weight factor used to scale the eigenvector normalization equation.")
		.def_prop_rw(
			"symmetric_FD",
			[](pyoomph::MyFoldHandler *h)
			{ return h->symmetric_FD; },
			[](pyoomph::MyFoldHandler *h, bool s)
			{ h->symmetric_FD = s; },
			"Whether the internal finite-difference derivatives are evaluated with a symmetric (central) stencil instead of a one-sided one.");

	nb::class_<pyoomph::MyPitchForkHandler, oomph::AssemblyHandler>(
		m, "PitchForkHandler",
		"AssemblyHandler activated during pitchfork bifurcation tracking. Augments the system with the eigenvector "
		"and symmetry-breaking constraint equations required to track a pitchfork bifurcation in a control parameter.")
		.def("get_base_ndof", &pyoomph::MyPitchForkHandler::get_problem_ndof,
			 "Number of degrees of freedom of the underlying (non-augmented) problem.")
		.def("set_eigenweight", &pyoomph::MyPitchForkHandler::set_eigenweight, nb::arg("weight"),
			 "Set the weight factor used to scale the eigenvector normalization equation.");

	nb::class_<pyoomph::MyHopfHandler, oomph::AssemblyHandler>(
		m, "HopfHandler",
		"AssemblyHandler activated during Hopf bifurcation tracking. Augments the system with the (complex) "
		"eigenvector equations and normalization conditions required to track a Hopf bifurcation, including the "
		"associated oscillation frequency, in a control parameter.")
		.def("get_nicely_rotated_eigenfunction", &pyoomph::MyHopfHandler::get_nicely_rotated_eigenfunction,
			 "Return the complex eigenvector of the Hopf mode, rotated in phase so that it has a canonical (reproducible) orientation.")
		.def("set_eigenweight", &pyoomph::MyHopfHandler::set_eigenweight, nb::arg("weight"),
			 "Set the weight factor used to scale the eigenvector normalization equations.")
		.def("get_base_ndof", &pyoomph::MyHopfHandler::get_problem_ndof,
			 "Number of degrees of freedom of the underlying (non-augmented) problem.")
		.def(
			"debug_analytical_filling", [](pyoomph::MyHopfHandler *self, oomph::GeneralisedElement *elem, double eps)
			{ self->debug_analytical_filling(elem, eps); },
			nb::arg("element"), nb::arg("eps"),
			"Debugging helper: compare the analytically assembled Jacobian entries of the Hopf system for ``element`` "
			"against a finite-difference approximation using step size ``eps``.");

	nb::class_<pyoomph::AzimuthalSymmetryBreakingHandler, oomph::AssemblyHandler>(
		m, "AzimuthalSymmetryBreakingHandler",
		"AssemblyHandler used to track azimuthal symmetry-breaking bifurcations on problems with a rotationally "
		"symmetric (axisymmetric) base state, i.e. bifurcations to a non-axisymmetric ('m != 0') mode.")
		.def("set_eigenweight", &pyoomph::AzimuthalSymmetryBreakingHandler::set_eigenweight, nb::arg("weight"),
			 "Set the weight factor used to scale the eigenvector normalization equations.")
		.def("get_base_ndof", &pyoomph::AzimuthalSymmetryBreakingHandler::get_problem_ndof,
			 "Number of degrees of freedom of the underlying (non-augmented, axisymmetric) problem.")
		.def("set_global_equations_forced_zero", &pyoomph::AzimuthalSymmetryBreakingHandler::set_global_equations_forced_zero,
			 nb::arg("base_equations"), nb::arg("eigen_equations"),
			 "Mark global equation numbers of the base state (``base_equations``) and of the eigenfunction (``eigen_equations``) "
			 "that must be forced to zero, e.g. due to boundary conditions enforced on the symmetry axis.");

	nb::class_<pyoomph::PeriodicOrbitHandler, oomph::AssemblyHandler>(
		m, "PeriodicOrbitHandler",
		"AssemblyHandler used to compute and continue periodic orbits (limit cycles) of a Problem in time, discretizing "
		"one period either via a B-spline basis, finite differences between nodal time points, or Floquet mode collocation.")
		.def("backup_dofs", &pyoomph::PeriodicOrbitHandler::backup_dofs,
			 "Backup the current degrees of freedom of the periodic orbit (e.g. before a trial step).")
		.def("restore_dofs", &pyoomph::PeriodicOrbitHandler::restore_dofs,
			 "Restore the degrees of freedom previously saved by backup_dofs().")
		.def("get_base_ndof", &pyoomph::PeriodicOrbitHandler::get_problem_ndof,
			 "Number of degrees of freedom of the underlying (non-augmented) problem.")
		.def("is_floquet_mode", &pyoomph::PeriodicOrbitHandler::is_floquet_mode,
			 "Whether the handler discretizes the orbit using the explicit Floquet mode (with an explicit degree of freedom "
			 "at the end of the period) rather than a B-spline or finite-difference representation.")
		.def("get_T", &pyoomph::PeriodicOrbitHandler::get_T, "Return the current period of the tracked periodic orbit.")
		.def("get_num_time_steps", &pyoomph::PeriodicOrbitHandler::n_tsteps,
			 "Return the number of time steps used to discretize one period of the orbit.")
		.def("get_s_integration_samples", &pyoomph::PeriodicOrbitHandler::get_s_integration_samples,
			 "Return a list of (s, weight) sample points in normalized time s in [0,1], such that "
			 "integral_0^1 f(U(s)) ds is approximated by sum(weight_i * f(U(s_i))).")
		.def("update_phase_constraint_information", &pyoomph::PeriodicOrbitHandler::update_phase_constraint_information,
			 "Update the internally stored derivative of the reference orbit, used to enforce the phase constraint that "
			 "pins the otherwise arbitrary time-shift of the periodic orbit.")
		.def("set_dofs_to_interpolated_values", &pyoomph::PeriodicOrbitHandler::set_dofs_to_interpolated_values, nb::arg("s"),
			 "Set the problem's degrees of freedom to the orbit's state interpolated at normalized time ``s`` in [0,1].");

/*
	class PythonAssemblyHandlerTrampoline : public pyoomph::PythonAssemblyHandler
	{
	public:
		using pyoomph::PythonAssemblyHandler::PythonAssemblyHandler;

	};

	nb::class_<pyoomph::PythonAssemblyHandler,PythonAssemblyHandlerTrampoline,oomph::AssemblyHandler>(m,"PythonAssemblyHandler")
		.def(nb::init<>());
		//.def("_after_construction", &pyoomph::PythonAssemblyHandler::_after_construction);
*/


	nb::class_<pyoomph::DynamicBulkElementInstance>(
		m, "DynamicBulkElementInstance",
		"A JIT-compiled and instantiated finite element code, attached to a particular mesh. Created via "
		"Problem.generate_and_compile_bulk_element_code(); provides the low-level bookkeeping (field indices, "
		"external data links, ...) required by the Python-level Equations/Mesh classes.")
		.def("_exchange_mesh", [](pyoomph::DynamicBulkElementInstance *self, MeshHandleBase *mesh_h)
			 { self->set_bulk_mesh(mesh_h->mesh()); }, nb::arg("mesh"),
			 "Change the bulk mesh this element code instance is associated with.")
		.def("link_external_data", &pyoomph::DynamicBulkElementInstance::link_external_data,
			 nb::arg("name"), nb::arg("data"), nb::arg("index"), nb::arg("full_source_name"),
			 "Link an external (non-local) Data value, identified by ``name`` in the generated code, to the given "
			 "oomph-lib ``data`` object and value ``index``; ``full_source_name`` is used for diagnostics.")
		.def("get_nodal_field_index", &pyoomph::DynamicBulkElementInstance::get_nodal_field_index, nb::arg("name"),
			 "Return the local index of the nodal field ``name`` in this element code, or a negative value if it is not present.")
		.def("get_discontinuous_field_index", &pyoomph::DynamicBulkElementInstance::get_discontinuous_field_index, nb::arg("name"),
			 "Return the local index of the discontinuous (elemental, DG-type) field ``name`` in this element code, or a negative value if it is not present.")
		.def("has_moving_nodes", &pyoomph::DynamicBulkElementInstance::has_moving_nodes,
			 "Whether this element code moves its nodes, i.e. uses an ALE (Arbitrary Lagrangian-Eulerian) formulation.")
		.def("get_max_dt_order", &pyoomph::DynamicBulkElementInstance::get_max_dt_order,
			 "Return the highest time derivative order (0: steady, 1: first order in time, ...) occurring in this element code.")
		.def("can_be_time_adaptive", &pyoomph::DynamicBulkElementInstance::can_be_time_adaptive,
			 "Whether this element code provides the temporal error estimators required for adaptive time stepping.")
		.def("has_parameter_contribution", &pyoomph::DynamicBulkElementInstance::has_parameter_contribution, nb::arg("parameter_name"),
			 "Whether the residuals of this element code depend on the global parameter ``parameter_name``.")
		.def("setup_interface_dof_indices", &pyoomph::DynamicBulkElementInstance::setup_interface_dof_indices,
			 "Populate and return the mapping of interface degree-of-freedom names to indices, required for continuous "
			 "fields defined only on parts of the mesh boundary (e.g. C2TB-C1 interface fields).")
		.def("get_nodal_field_indices", &pyoomph::DynamicBulkElementInstance::get_nodal_field_indices,
			 "Return a mapping from nodal field name to its local index in this element code.")
		.def(
			"set_analytical_jacobian", [](pyoomph::DynamicBulkElementInstance *self, bool ana, bool anapos)
			{
				self->get_func_table()->fd_jacobian = !ana;
				self->get_func_table()->fd_position_jacobian = !anapos;
			},
			nb::arg("analytic"), nb::arg("analytic_positions"),
			"Select whether the Jacobian is assembled analytically (via the generated derivative code) or by finite "
			"differences, separately for the ordinary degrees of freedom (``analytic``) and for the nodal position "
			"degrees of freedom in moving-mesh problems (``analytic_positions``).")
		.def("get_elemental_field_indices", &pyoomph::DynamicBulkElementInstance::get_elemental_field_indices,
			 "Return a mapping from elemental (discontinuous) field name to its local index in this element code.");

	nb::class_<pyoomph::Problem, pyoomph::PyProblemTrampoline>(
		m, "Problem",
		"C++ core counterpart of a pyoomph Problem: owns the meshes, degrees of freedom, time steppers and the "
		"Newton/arclength/bifurcation-tracking machinery inherited from oomph-lib. The Python-level Problem class "
		"derives from this and overrides the virtual hooks (setup_pinning(), set_initial_condition(), ...).")
		.def(nb::init<>(), "Create a new, empty Problem.")
		.def("assembly_handler_pt", (oomph::AssemblyHandler * &(pyoomph::Problem::*)()) & pyoomph::Problem::assembly_handler_pt, nb::rv_policy::reference,
			 "Return the AssemblyHandler currently used to assemble residuals and Jacobians, e.g. a bifurcation-tracking handler if one is active.")
		.def("enable_store_local_dof_pt_in_elements", &pyoomph::Problem::enable_store_local_dof_pt_in_elements,
			 "Enable storing direct pointers to the local degrees of freedom in each element (an internal performance optimization).")
		.def("setup_pinning", &pyoomph::Problem::setup_pinning,
			 "Hook, overridable in Python, to pin (fix) degrees of freedom before the equation numbers are assigned.")
		.def("set_initial_condition", &pyoomph::Problem::set_initial_condition,
			 "Hook, overridable in Python, to set the initial condition of the problem.")
		// The following are hooks, overridable in Python, called at the corresponding point of a
		// Newton solve/adaptation/distribution; all default to a no-op in the C++ core. Unlike
		// pybind11, nanobind's virtual-override dispatch (see PyProblemTrampoline above) requires
		// every overridable hook to also have a plain (non-overridden) binding under the same name
		// here, used to detect whether a Python subclass has actually overridden it.
		.def("actions_before_newton_solve", &pyoomph::Problem::actions_before_newton_solve,
			 "Hook, overridable in Python, called right before a Newton solve.")
		.def("actions_after_newton_solve", &pyoomph::Problem::actions_after_newton_solve,
			 "Hook, overridable in Python, called right after a Newton solve.")
		.def("actions_before_newton_convergence_check", &pyoomph::Problem::actions_before_newton_convergence_check,
			 "Hook, overridable in Python, called before each Newton convergence check.")
		.def("actions_after_change_in_global_parameter", (void (pyoomph::Problem::*)(const std::string &)) & pyoomph::Problem::actions_after_change_in_global_parameter, nb::arg("parameter_name"),
			 "Hook, overridable in Python, called after the named global parameter has changed (e.g. during continuation).")
		.def("actions_after_parameter_increase", (void (pyoomph::Problem::*)(const std::string &)) & pyoomph::Problem::actions_after_parameter_increase, nb::arg("parameter_name"),
			 "Hook, overridable in Python, called after the named global parameter has been increased.")
		.def("actions_after_newton_step", &pyoomph::Problem::actions_after_newton_step,
			 "Hook, overridable in Python, called after each Newton step.")
		.def("actions_before_newton_step", &pyoomph::Problem::actions_before_newton_step,
			 "Hook, overridable in Python, called before each Newton step.")
		.def("actions_before_adapt", &pyoomph::Problem::actions_before_adapt,
			 "Hook, overridable in Python, called before spatial mesh adaptation.")
		.def("actions_after_adapt", &pyoomph::Problem::actions_after_adapt,
			 "Hook, overridable in Python, called after spatial mesh adaptation.")
		.def("actions_before_distribute", &pyoomph::Problem::actions_before_distribute,
			 "Hook, overridable in Python, called before the problem's meshes are distributed across MPI processes.")
		.def("actions_after_distribute", &pyoomph::Problem::actions_after_distribute,
			 "Hook, overridable in Python, called after the problem's meshes are distributed across MPI processes.")
		.def("_get_jacobian_information_string", &pyoomph::Problem::get_jacobian_information_string,
			 "Return a human-readable summary of the defined fields, residuals and their Jacobian coupling structure, "
			 "together with a flag whether the structure looks consistent; used for diagnostics/debugging.")
		.def("refine_uniformly", (void(pyoomph::Problem::*)()) & pyoomph::Problem::refine_uniformly,
			 "Uniformly refine all refineable meshes of the problem by one level.")
		.def("unrefine_uniformly", (unsigned(pyoomph::Problem::*)()) & pyoomph::Problem::unrefine_uniformly,
			 "Uniformly unrefine all refineable meshes of the problem by one level. Returns the number of elements that could not be unrefined further.")
		.def("assign_eqn_numbers", &pyoomph::Problem::assign_eqn_numbers, nb::arg("assign_local_eqn_numbers") = true,
			 "(Re-)assign the global equation numbers to all degrees of freedom of the problem. Must be called whenever "
			 "the problem structure (meshes, pinning, ...) changes. Returns the total number of degrees of freedom.")
		.def("initialise_dt", (void(pyoomph::Problem::*)(const double &)) & pyoomph::Problem::initialise_dt, nb::arg("dt"),
			 "Set all time steps of all time steppers in the problem to ``dt`` and (re-)assign their weights.")
		.def("assign_initial_values_impulsive", (void(pyoomph::Problem::*)(const double &)) & pyoomph::Problem::assign_initial_values_impulsive, nb::arg("dt"),
			 "Initialize data and nodal positions to simulate an impulsive start from the current configuration, also setting the initial and previous time step to ``dt``.")
		.def("assign_initial_values_impulsive", (void(pyoomph::Problem::*)()) & pyoomph::Problem::assign_initial_values_impulsive,
			 "Initialize data and nodal positions to simulate an impulsive start from the current configuration/solution.")
		.def("get_last_jacobian_setup_time", [](pyoomph::Problem *self)
			 { return self->linear_solver_pt()->jacobian_setup_time(); }, "Return the wall-clock time (in seconds) spent setting up the Jacobian in the most recent linear solve.")
		.def("get_last_linear_solver_solution_time", [](pyoomph::Problem *self)
			 { return self->linear_solver_pt()->linear_solver_solution_time(); }, "Return the wall-clock time (in seconds) spent in the linear solver during the most recent linear solve.")
		.def_prop_rw(
			"max_residuals",
			[](pyoomph::Problem &p)
			{ return p.max_residuals(); },
			[](pyoomph::Problem &p, const double &r)
			{ p.max_residuals() = r; },
			"Maximum desired residual: if the maximum residual exceeds this value at any point during a Newton solve, the solver gives up.")
		.def_prop_rw(
			"apply_Dirichlet_BCs_by_dof_removing",
			[](pyoomph::Problem &p)
			{ return p.are_Dirichlets_by_removing_from_dof_vector(); },
			[](pyoomph::Problem &p, const bool &r)
			{ p.set_Dirichlets_by_removing_from_dof_vector(r); },
			"If True (default), Dirichlet conditions are enforced the oomph-lib way, by removing the corresponding entries from the degree-of-freedom vector. "
			"If False, all Dirichlet dofs are kept as regular degrees of freedom and the assembled system is manipulated afterwards instead.")
		.def_prop_rw(
			"nodal_block_dof_arrangement_used",
			[](pyoomph::Problem &p)
			{ return p.is_block_dof_arrangement_used(); },
			[](pyoomph::Problem &p, const bool &r)
			{ p.set_block_dof_arrangement_used(r); },
			"Whether the degrees of freedom are arranged nodal-block-wise (all values of a node contiguous), which is required by some block-preconditioned solvers.")
		.def_prop_rw(
			"DTSF_minimum_dt",
			[](pyoomph::Problem &p)
			{ return p.minimum_dt(); },
			[](pyoomph::Problem &p, double v)
			{ p.minimum_dt() = v; },
			"Minimum desired time step during adaptive time stepping; the solve aborts with an error if dt falls below this value.")
		.def_prop_rw(
			"DTSF_maximum_dt",
			[](pyoomph::Problem &p)
			{ return p.maximum_dt(); },
			[](pyoomph::Problem &p, double v)
			{ p.maximum_dt() = v; },
			"Maximum desired time step during adaptive time stepping.")
		.def(
			"_set_globally_convergent_newton_method", [](pyoomph::Problem &p, bool r)
			{ if (r) p.enable_globally_convergent_newton_method(); else p.disable_globally_convergent_newton_method(); },
			nb::arg("active"), "Enable or disable the globally convergent (line-search damped) Newton method.")
		.def_prop_rw(
			"max_newton_iterations",
			[](pyoomph::Problem &p)
			{ return p.max_newton_iterations(); },
			[](pyoomph::Problem &p, const unsigned &r)
			{ p.max_newton_iterations() = r; },
			"Maximum number of Newton iterations for solving before giving up.")
		.def_prop_rw(
			"newton_solver_tolerance",
			[](pyoomph::Problem &p)
			{ return p.newton_solver_tolerance(); },
			[](pyoomph::Problem &p, const double &r)
			{ p.newton_solver_tolerance() = r; },
			"Maximum value in the residual vector to consider the solution as converged during the Newton method.")
		.def_prop_rw(
			"always_take_one_newton_step",
			[](pyoomph::Problem &p)
			{ return p.always_take_one_newton_step(); },
			[](pyoomph::Problem &p, const bool &b)
			{ p.always_take_one_newton_step() = b; },
			"If True, a Newton step is always taken even if the initial residuals are already below the required tolerance.")
		.def_prop_rw(
			"newton_relaxation_factor",
			[](pyoomph::Problem &p)
			{ return p.newton_relaxation_factor(); },
			[](pyoomph::Problem &p, const double &r)
			{ p.newton_relaxation_factor() = r; },
			"Relaxation factor for the Newton method: only a fraction (this value) of the full Newton correction is applied if less than 1 (default 1).")
		.def_prop_rw(
			"DTSF_max_increase_factor",
			[](pyoomph::Problem &p)
			{ return p.DTSF_max_increase_factor(); },
			[](pyoomph::Problem &p, const double &r)
			{ p.DTSF_max_increase_factor() = r; },
			"Maximum possible increase factor of the time step dt between successive time steps in adaptive time stepping.")
		.def_prop_rw(
			"DTSF_min_decrease_factor",
			[](pyoomph::Problem &p)
			{ return p.DTSF_min_decrease_factor(); },
			[](pyoomph::Problem &p, const double &r)
			{ p.DTSF_min_decrease_factor() = r; },
			"Minimum allowed decrease factor of the time step dt between successive time steps in adaptive time stepping; "
			"a proposed reduction below this factor causes the time step to be rejected and retried with a smaller dt.")
		.def_prop_rw(
			"minimum_arclength_ds",
			[](pyoomph::Problem &p)
			{ return p.minimum_ds(); },
			[](pyoomph::Problem &p, const double &r)
			{ p.minimum_ds() = r; },
			"Minimum desired value of the arclength step ds during arclength continuation.")
		.def_prop_rw(
			"keep_temporal_error_below_tolerance",
			[](pyoomph::Problem &p)
			{ return p.get_Keep_temporal_error_below_tolerance(); },
			[](pyoomph::Problem &p, bool s)
			{ p.set_Keep_temporal_error_below_tolerance(s); },
			"If True (default), a time step is rejected and retried with a smaller dt if the temporal error estimate after solving exceeds the requested tolerance.")
		.def_prop_rw(
			"use_custom_residual_jacobian",
			[](pyoomph::Problem &p)
			{ return p.use_custom_residual_jacobian; },
			[](pyoomph::Problem &p, bool s)
			{ p.use_custom_residual_jacobian = s; },
			"If True, residuals/Jacobians are obtained by calling the (Python-overridable) get_custom_residuals_jacobian() hook instead of the generated finite element code.")
		.def_prop_rw(
			"_improved_pitchfork_tracking_on_unstructured_meshes",
			[](pyoomph::Problem &p)
			{ return p.improved_pitchfork_tracking_on_unstructured_meshes; },
			[](pyoomph::Problem &p, bool s)
			{ p.improved_pitchfork_tracking_on_unstructured_meshes = s; },
			"Whether an improved (but more expensive) pitchfork bifurcation tracking scheme is used, required for some unstructured meshes.")
		.def_prop_rw("sparse_assembly_method", &pyoomph::Problem::get_sparse_assembly_method, &pyoomph::Problem::set_sparse_assembly_method,
					  "Method used to assemble the sparse Jacobian matrix: one of \"vectors_of_pairs\", \"two_vectors\", \"maps\", \"lists\" or \"two_arrays\" (see oomph-lib for details).")
		.def_prop_rw("dist_problem_matrix_distribution", &pyoomph::Problem::get_dist_problem_matrix_distribution, &pyoomph::Problem::set_dist_problem_matrix_distribution,
					  "How the Jacobian matrix rows are distributed across MPI processes in a distributed problem: one of \"default\", \"problem\" or \"uniform\". No-op without MPI support.")
		.def("adaptive_unsteady_newton_solve", (double(pyoomph::Problem::*)(const double &, const double &)) & pyoomph::Problem::adaptive_unsteady_newton_solve,
			 nb::arg("dt_desired"), nb::arg("epsilon"),
			 "Attempt to advance time by ``dt_desired``, halving the time step on non-convergence until it succeeds or falls below the minimum time step. "
			 "``epsilon`` is the desired magnitude of the global temporal error at each time step. Always shifts the time history values. "
			 "Returns the suggested next time step, to be passed again as ``dt_desired``.")
		.def("_adapt", &pyoomph::Problem::_adapt, "Perform a single spatial adaptation step (mesh refinement/unrefinement). Returns (n_refined, n_unrefined).")
		.def("adaptive_unsteady_newton_solve", (double(pyoomph::Problem::*)(const double &, const double &, const bool &)) & pyoomph::Problem::adaptive_unsteady_newton_solve,
			 nb::arg("dt_desired"), nb::arg("epsilon"), nb::arg("shift_values"),
			 "Same as the two-argument overload, but ``shift_values`` controls whether the time history values are shifted.")
		.def("unsteady_newton_solve", (void(pyoomph::Problem::*)(const double &, const bool &)) & pyoomph::Problem::unsteady_newton_solve,
			 nb::arg("dt"), nb::arg("shift_values"),
			 "Advance time by ``dt`` and solve by Newton's method. ``shift_values`` controls whether the current data values are shifted "
			 "to the storage locations of the previous time step before the solve (must be True on the very first time step).")
		.def("unsteady_newton_solve", (void(pyoomph::Problem::*)(const double &, const unsigned &, const bool &, const bool &)) & pyoomph::Problem::unsteady_newton_solve,
			 nb::arg("dt"), nb::arg("max_adapt"), nb::arg("first"), nb::arg("shift") = true,
			 "Advance time by ``dt`` with up to ``max_adapt`` rounds of spatial mesh adaptation to satisfy the error targets of the refineable submeshes. "
			 "If ``first`` is True, the initial conditions are re-assigned after each mesh adaptation. ``shift`` controls whether time history values are shifted.")
		.def("doubly_adaptive_unsteady_newton_solve",
			 (double(pyoomph::Problem::*)(const double &, const double &, const unsigned &, const unsigned &, const bool &, const bool &)) & pyoomph::Problem::doubly_adaptive_unsteady_newton_solve,
			 nb::arg("dt"), nb::arg("epsilon"), nb::arg("max_adapt"), nb::arg("suppress_resolve_after_spatial_adapt_flag"), nb::arg("first"), nb::arg("shift") = true,
			 "Advance time by ``dt``, first performing temporal adaptation (adjusting dt until the temporal error tolerance ``epsilon`` is met), then up to "
			 "``max_adapt`` rounds of spatial adaptation without re-checking the temporal error. If ``first`` is True, the initial conditions are re-assigned "
			 "after mesh adaptation. ``shift`` controls whether time history values are shifted. Returns the suggested next time step.")
		.def("newton_solve", (void(pyoomph::Problem::*)(unsigned const &)) & pyoomph::Problem::newton_solve, "Perform a newton solve", nb::arg("max_adapt") = 0)
		.def("steady_newton_solve", (void(pyoomph::Problem::*)(unsigned const &)) & pyoomph::Problem::steady_newton_solve, "Perform a steady newton solve", nb::arg("max_adapt") = 0)
		.def("_arc_length_step", &pyoomph::Problem::arc_length_step, nb::arg("parameter_name"), nb::arg("ds"), nb::arg("max_adapt"),
			 "Perform one pseudo-arclength continuation step of size ``ds`` in the global parameter ``parameter_name``, with up to ``max_adapt`` rounds of spatial adaptation.")
		.def("get_arc_length_parameter_derivative", &pyoomph::Problem::get_arc_length_parameter_derivative,
			 "Return the current derivative of the continuation parameter with respect to arclength.")
		.def("_set_arc_length_parameter_derivative", &pyoomph::Problem::set_arc_length_parameter_derivative, nb::arg("dp_ds"),
			 "Set the derivative of the continuation parameter with respect to arclength.")
		.def("_update_dof_vectors_for_continuation", &pyoomph::Problem::update_dof_vectors_for_continuation, nb::arg("ddof"), nb::arg("current"),
			 "Update the internally stored dof-derivative (``ddof``) and current-dof (``current``) vectors used by arclength continuation.")
		.def("_update_param_info_for_continuation", &pyoomph::Problem::update_param_info_for_continuation, nb::arg("dparam_ds"), nb::arg("param0"),
			 "Update the internally stored continuation parameter derivative and reference value used by arclength continuation.")
		.def("get_arc_length_theta_sqr", &pyoomph::Problem::get_arc_length_theta_sqr,
			 "Return theta^2, the weighting parameter that controls the proportion of the arclength taken up by the continuation parameter.")
		.def("_set_arc_length_theta_sqr", &pyoomph::Problem::set_arc_length_theta_sqr, nb::arg("theta_sqr"),
			 "Set theta^2, the weighting parameter that controls the proportion of the arclength taken up by the continuation parameter.")
		.def("_set_arclength_parameter", &pyoomph::Problem::set_arclength_parameter, nb::arg("name"), nb::arg("value"),
			 "Set an internal named arclength-continuation setting (e.g. \"use_continuation_timestepper\" or \"Desired_newton_iterations_ds\") to ``value``.")
		.def("_start_bifurcation_tracking", &pyoomph::Problem::start_bifurcation_tracking,
			 nb::arg("parameter_name"), nb::arg("bifurcation_type"), nb::arg("block_solve"), nb::arg("eigenvector1"), nb::arg("eigenvector2"), nb::arg("omega"), nb::arg("special_residual_forms"),
			 "Activate bifurcation tracking with respect to the global parameter ``parameter_name``. ``bifurcation_type`` selects the kind of bifurcation "
			 "(e.g. fold, pitchfork, Hopf or azimuthal), ``eigenvector1``/``eigenvector2`` are the (real/imaginary) starting eigenvector guess(es), ``omega`` "
			 "the starting guess for the (Hopf/azimuthal) eigenvalue frequency, and ``special_residual_forms`` maps residual names to alternative forms used for the tracking equations.")
		.def("_start_orbit_tracking", &pyoomph::Problem::start_orbit_tracking,
			 nb::arg("history"), nb::arg("period"), nb::arg("bspline_order"), nb::arg("gl_order"), nb::arg("knots"), nb::arg("T_constraint_mode"),
			 "Activate periodic orbit tracking (see PeriodicOrbitHandler), starting from the time ``history`` of dof snapshots covering one period ``period``. "
			 "``bspline_order``/``knots`` select a B-spline discretization of the orbit if >=0, ``gl_order`` the Gauss-Legendre collocation order, and "
			 "``T_constraint_mode`` selects how the period/phase of the orbit is pinned (0: plane constraint, 1: period constraint).")
		.def("_get_max_dt_order", &pyoomph::Problem::get_max_dt_order,
			 "Return the highest time derivative order (0: steady, 1: first order, ...) occurring anywhere in the problem.")
		//.def("_start_custom_augmented_system", &pyoomph::Problem::start_custom_augmented_system)
		.def("_reset_augmented_dof_vector_to_nonaugmented", &pyoomph::Problem::reset_augmented_dof_vector_to_nonaugmented,
			 "Deactivate any augmented degrees of freedom, restoring the plain (non-augmented) dof vector, e.g. after bifurcation/orbit tracking has been stopped.")
		.def("_create_dof_augmentation", &pyoomph::Problem::create_dof_augmentation, nb::rv_policy::take_ownership,
			 "Create a new (empty) DofAugmentations object for this problem, to be filled and passed to _add_augmented_dofs().")
		.def("_get_n_unaugmented_dofs", &pyoomph::Problem::get_n_unaugmented_dofs,
			 "Return the number of degrees of freedom of the underlying problem, excluding any currently active augmented degrees of freedom.")
		.def("_add_augmented_dofs", &pyoomph::Problem::add_augmented_dofs, nb::arg("augmentations"),
			 "Append the degrees of freedom collected in ``augmentations`` (a DofAugmentations object) to the problem's dof vector and re-build the dof distribution.")
		.def("_enable_store_local_dof_pt_in_elements", &pyoomph::Problem::enable_store_local_dof_pt_in_elements,
			 "Enable storing direct pointers to the local degrees of freedom in each element (an internal performance optimization).")
		.def("after_bifurcation_tracking_step", &pyoomph::Problem::after_bifurcation_tracking_step,
			 "Hook called after each solve step while bifurcation tracking is active, to update internally cached tracking information.")
		.def("get_custom_residuals_jacobian", &pyoomph::Problem::get_custom_residuals_jacobian, nb::arg("info"),
			 "Hook, overridable in Python, that fills ``info`` (a CustomResJacInfo) with a custom residual vector and, if requested, Jacobian; used together with use_custom_residual_jacobian.")
		.def("_unpin_Dirichlet_dofs_for_matrix_manipulation", &pyoomph::Problem::unpin_Dirichlet_dofs_for_matrix_manipulation,
			 "Unpin the Dirichlet degrees of freedom again after they were removed from the linear system by matrix manipulation (only relevant if apply_Dirichlet_BCs_by_dof_removing is False).")
		.def("get_bifurcation_tracking_mode", &pyoomph::Problem::get_bifurcation_tracking_mode,
			 "Return the currently active bifurcation tracking mode as a string (e.g. \"fold\", \"pitchfork\", \"hopf\", \"azimuthal\"), or an empty string if no bifurcation is being tracked.")
		.def("_get_bifurcation_eigenvector", &pyoomph::Problem::get_bifurcation_eigenvector,
			 "Return the (possibly complex) eigenvector currently associated with the active bifurcation tracking, evaluated at the base (non-augmented) degrees of freedom.")
		.def("_get_bifurcation_tracking_info", &pyoomph::Problem::get_bifurcation_tracking_info,
			 "Return a tuple (is_tracking, eigenvalue) describing whether a bifurcation is currently being tracked and, if so, its (possibly complex) eigenvalue.")
		.def("_get_bifurcation_omega", &pyoomph::Problem::get_bifurcation_omega,
			 "Return the imaginary part (angular frequency) of the eigenvalue of the currently tracked Hopf or azimuthal bifurcation.")
		.def("_get_lambda_tracking_real", [](pyoomph::Problem *self)
			 { return *self->get_lambda_tracking_real(); }, "Return the real part of lambda used for eigenbranch tracking.")
		.def("_set_lambda_tracking_real", [](pyoomph::Problem *self, double lr)
			 { *self->get_lambda_tracking_real() = lr; }, nb::arg("lambda_real"), "Set the real part of lambda used for eigenbranch tracking.")
		//.def("reset_arc_length_parameters", &pyoomph::Problem::reset_arc_length_parameters)
		.def("reset_arc_length_parameters", [](pyoomph::Problem *self)
			 { self->reset_arc_length_parameters(); }, "Reset all internal arclength continuation parameters (theta^2, sign of Jacobian, continuation direction, parameter derivative, ...) to their default starting values.")
		.def("_set_dof_direction_arclength", &pyoomph::Problem::set_dof_direction_arclength, nb::arg("direction"),
			 "Set the direction vector along which the dof-derivative for arclength continuation is initialized/oriented.")
		.def(
			"get_parameter_derivative", [](pyoomph::Problem *self, const std::string &parameter_name)
			{ return vector_to_ndarray(self->get_parameter_derivative(parameter_name)); },
			nb::arg("parameter_name"),
			"Return the derivative of the residual vector with respect to the global parameter ``parameter_name`` (i.e. dR/dparam), evaluated at the current state.")
		.def(
			"get_arclength_dof_derivative_vector", [](pyoomph::Problem *self)
			{ return vector_to_ndarray(self->get_arclength_dof_derivative_vector()); },
			"Return the current derivative of the degrees of freedom with respect to arclength, as used/updated during arclength continuation.")
		.def(
			"get_arclength_dof_current_vector", [](pyoomph::Problem *self)
			{ return vector_to_ndarray(self->get_arclength_dof_current_vector()); },
			"Return the degrees of freedom at the last converged point on the arclength continuation branch.")
		.def(
			"get_global_parameter", [](pyoomph::Problem *self, const std::string &n) -> GiNaC::GiNaCGlobalParameterWrapper
			{ auto *gpd = self->assert_global_parameter(n); return GiNaC::GiNaCGlobalParameterWrapper(gpd); },
			nb::rv_policy::reference, nb::arg("parameter_name"),
			"Return a global parameter. If it does not exist, it will be added and initialized with value 0.")
		.def("get_global_parameter_names", &pyoomph::Problem::get_global_parameter_names,
			 "Return the set of names of all global parameters currently defined on this problem.")
		.def(
			"get_current_dofs", [](pyoomph::Problem *self)
			{
				auto rs = self->get_current_dofs();
				return std::make_tuple(vector_to_ndarray(std::get<0>(rs)), std::get<1>(rs));
				// return self->get_current_dofs();
			},
			"Return a tuple (values, is_pinned) with the current values of all degrees of freedom and, for each, whether it is currently pinned.")
		.def(
			"get_history_dofs", [](pyoomph::Problem *self, unsigned t)
			{
				auto rs = self->get_history_dofs(t);
				return vector_to_ndarray(rs);
			},
			nb::arg("t"), "Return the values of all degrees of freedom at history/time level ``t`` (t=0: current values, t=1: previous time step, ...).")
		.def("get_last_residual_convergence", &pyoomph::Problem::get_last_residual_convergence,
			 "Return the maximum residual recorded at the start of, and after each iteration of, the most recent Newton solve; useful for diagnosing convergence behavior.")
		.def(
			"get_residuals", [](pyoomph::Problem *self)
			{
				oomph::DoubleVector ov;
				self->get_residuals(ov);
				std::vector<double> res(self->ndof());
				for (unsigned int i = 0; i < self->ndof(); i++)
					res[i] = ov[i];
				return vector_to_ndarray(res);
			},
			"Assemble and return the current residual vector (without the Jacobian) at the current state.")
		.def(
			"get_current_pinned_values", [](pyoomph::Problem *self, bool with_pos)
			{ return vector_to_ndarray(self->get_current_pinned_values(with_pos)); },
			nb::arg("with_position_dofs"), "Return the current values of all pinned degrees of freedom; also includes pinned nodal position dofs if ``with_position_dofs`` is True.")
		.def(
			"set_current_dofs", [](pyoomph::Problem *self, const std::vector<double> &inp)
			{ return self->set_current_dofs(inp); },
			nb::arg("values"), "Set the current values of all (unpinned and pinned) degrees of freedom from ``values``.")
		.def(
			"set_history_dofs", [](pyoomph::Problem *self, unsigned t, const std::vector<double> &inp)
			{ return self->set_history_dofs(t, inp); },
			nb::arg("t"), nb::arg("values"), "Set the values of all degrees of freedom at history/time level ``t`` from ``values``.")
		.def(
			"set_current_pinned_values", [](pyoomph::Problem *self, const std::vector<double> &inp, bool with_pos, unsigned t)
			{ return self->set_current_pinned_values(inp, with_pos, t); },
			nb::arg("values"), nb::arg("with_position_dofs"), nb::arg("t") = 0,
			"Set the values of all pinned degrees of freedom (and, if ``with_position_dofs`` is True, pinned nodal position dofs) at history/time level ``t`` from ``values``.")
		.def(
			"assemble_eigenproblem_matrices", [](pyoomph::Problem *self, double sigma_r)
			{
				oomph::CRDoubleMatrix *M = NULL, *J = NULL;
				self->assemble_eigenproblem_matrices(M, J, sigma_r);

				unsigned M_nzz = M->nnz();
				unsigned J_nzz = J->nnz();
				unsigned n = M->distribution_pt()->nrow();

				double *M_values = M->value(); // nnz_local
				double *J_values = J->value(); // nnz_local

				int *M_colindex = M->column_index(); // nnz_local
				int *J_colindex = J->column_index(); // nnz_local
				int M_nrow_local = M->nrow_local();
				int J_nrow_local = J->nrow_local();
				int *M_row_start = M->row_start(); // nrow_local+1
				int *J_row_start = J->row_start(); // nrow_local+1

				nb::ndarray<nb::numpy, double> M_values_arr(M_values, {(size_t)M_nzz}, nb::capsule(M_values, [](void *f) noexcept {}));
				nb::ndarray<nb::numpy, int> M_colindex_arr(M_colindex, {(size_t)M_nzz}, nb::capsule(M_colindex, [](void *f) noexcept {}));
				nb::ndarray<nb::numpy, int> M_row_start_arr(M_row_start, {(size_t)M_nrow_local + 1}, nb::capsule(M_row_start, [](void *f) noexcept {}));

				nb::ndarray<nb::numpy, double> J_values_arr(J_values, {(size_t)J_nzz}, nb::capsule(J_values, [](void *f) noexcept {}));
				nb::ndarray<nb::numpy, int> J_colindex_arr(J_colindex, {(size_t)J_nzz}, nb::capsule(J_colindex, [](void *f) noexcept {}));
				nb::ndarray<nb::numpy, int> J_row_start_arr(J_row_start, {(size_t)J_nrow_local + 1}, nb::capsule(J_row_start, [](void *f) noexcept {}));

				return std::make_tuple(n, M_nzz, M_nrow_local, M_values_arr, M_colindex_arr, M_row_start_arr, J_nzz, J_nrow_local, J_values_arr, J_colindex_arr, J_row_start_arr);
			},
			nb::arg("sigma_r"),
			"Assemble the mass matrix M and Jacobian J needed to set up the generalized eigenproblem J x = lambda M x (e.g. for a shift-invert "
			"eigensolver with real shift ``sigma_r``). Returns (n, M_nnz, M_nrow_local, M_values, M_column_index, M_row_start, J_nnz, J_nrow_local, "
			"J_values, J_column_index, J_row_start), both matrices in local compressed sparse row (CSR) format.")
		.def(
			"_assemble_residual_jacobian", [](pyoomph::Problem *self, std::string name)
			{
				std::string oldresi = self->_get_solved_residual();
				if (name != oldresi)
					self->_set_solved_residual(name, true, false);
				oomph::DoubleVector resi;
				oomph::CRDoubleMatrix J;
				self->get_jacobian(resi, J);
				unsigned J_nzz = J.nnz();
				unsigned n = J.distribution_pt()->nrow();
				double *J_values = J.value();
				unsigned int J_nrow_local = J.nrow_local();
				int *J_row_start = J.row_start();	 // nrow_local+1
				int *J_colindex = J.column_index(); // nnz_local
				if (name != oldresi)
					self->_set_solved_residual(oldresi, true, true);

				std::vector<double> J_values_vec(J_values, J_values + J_nzz);
				std::vector<int> J_colindex_vec(J_colindex, J_colindex + J_nzz);
				std::vector<int> J_row_start_vec(J_row_start, J_row_start + J_nrow_local + 1);
				auto J_values_arr = vector_to_ndarray(J_values_vec);
				auto J_colindex_arr = vector_to_ndarray(J_colindex_vec);
				auto J_row_start_arr = vector_to_ndarray(J_row_start_vec);

				std::vector<double> res(n);
				for (unsigned int i = 0; i < n; i++)
					res[i] = resi[i];
				return std::make_tuple(vector_to_ndarray(res), n, J_nzz, J_nrow_local, J_values_arr, J_colindex_arr, J_row_start_arr);
			},
			nb::arg("residual_name"),
			"Temporarily activate the residual/Jacobian combination named ``residual_name`` (restoring the previously active one afterwards), "
			"and assemble and return (residuals, n, J_nnz, J_nrow_local, J_values, J_column_index, J_row_start) with the Jacobian in local CSR format.")
		.def("quiet", &pyoomph::Problem::quiet, nb::arg("quiet") = true, "Deactivate output messages from the oomph-lib and pyoomph C++ core")
		.def("_open_log_file", &pyoomph::Problem::open_log_file, nb::arg("fname"), nb::arg("activate_logging") = true, "Open a log file for the problem")
		.def("_assemble_hessian_tensor", &pyoomph::Problem::assemble_hessian_tensor, nb::arg("symmetric"),
			 "Assemble and return the Hessian (second derivative of the residuals with respect to the degrees of freedom) as a SparseRank3Tensor. "
			 "If ``symmetric`` is True, the tensor is assumed/exploited to be symmetric in its last two indices, halving the assembly cost.")
		.def("is_quiet", &pyoomph::Problem::is_quiet, "Return whether output messages from the oomph-lib and pyoomph C++ core are currently suppressed.")
		.def("_unload_all_dlls", &pyoomph::Problem::unload_all_dlls,nb::arg("clear_all") = true,
			 "Unload all dynamically loaded/compiled shared libraries (generated element codes) currently linked into this problem. With clear_all, also remove all meshes, elements, nodes")
		.def("add_time_stepper_pt", &pyoomph::Problem::add_time_stepper_pt, nb::keep_alive<1, 2>(), nb::arg("time_stepper"),
			 "Add a time stepper to the problem, automatically (re-)sizing the Time object to hold the required number of history levels.")
		.def("set_mesh_pt", [](pyoomph::Problem *self, MeshHandleBase *mesh_h)
			 { self->set_mesh_pt(mesh_h->mesh()); }, nb::keep_alive<1, 2>(), nb::arg("mesh"), "Set the (single) global mesh of the problem.")
		.def("add_sub_mesh", [](pyoomph::Problem *self, MeshHandleBase *mesh_h)
			 { return self->add_sub_mesh(mesh_h->oomph_mesh()); }, nb::keep_alive<1, 2>(), nb::arg("mesh"),
			 "Add ``mesh`` as a sub-mesh of the problem; combine all sub-meshes into the global mesh afterwards via build_global_mesh().")
		.def("flush_sub_meshes", &pyoomph::Problem::flush_sub_meshes, "Remove all sub-meshes previously added via add_sub_mesh(), without deleting them.")
		.def("_get_global_field_names", &pyoomph::Problem::get_global_field_names,
			 "Return the names of all fields defined anywhere in the problem, in the (stable) global field index order.")
		.def(
			"_get_dof_to_global_field_index_mapping", [](pyoomph::Problem *self)
			{ return vector_to_ndarray(self->get_dof_to_global_field_index_mapping()); },
			"Return, for each degree of freedom, the index of the global field it belongs to (or a negative value if it does not correspond to a named field, e.g. a Lagrange multiplier).")
		.def(
			"get_second_order_directional_derivative", [](pyoomph::Problem *self, std::vector<double> direction)
			{ return vector_to_ndarray(self->get_second_order_directional_derivative(direction)); },
			nb::arg("direction"),
			"Return the second order directional derivative of the residuals along ``direction``, i.e. the contraction of the Hessian with ``direction`` twice (a Hessian-vector-vector product).")
		.def("nsub_mesh", &pyoomph::Problem::nsub_mesh, "Return the number of sub-meshes added via add_sub_mesh().")
		.def(
			"adapt", [](pyoomph::Problem &self)
			{ unsigned nref, nunref; self.adapt(nref, nunref); return std::make_tuple(nref, nunref); },
			"Perform spatial mesh adaptation (refinement/unrefinement) on all refineable submeshes. Returns (n_refined, n_unrefined).")
		.def("_replace_RJM_by_param_deriv", &pyoomph::Problem::_replace_RJM_by_param_deriv, nb::arg("parameter_name"), nb::arg("active"),
			 "If ``active`` is True, replace the assembled residuals/Jacobian/mass-matrix by their derivative with respect to the global parameter "
			 "``parameter_name`` in all subsequent assembly calls; if False, restore the plain residuals/Jacobian/mass-matrix.")
		.def("_set_solved_residual", &pyoomph::Problem::_set_solved_residual, nb::arg("name"), nb::arg("raise_error") = true, nb::arg("remove_dofs_without_jacobian_row") = true,
			 "Activate the residual/Jacobian combination named ``name`` as the one assembled by subsequent get_residuals()/get_jacobian() calls. Raises an error if "
			 "no equation contributes to it and ``raise_error`` is True. If ``remove_dofs_without_jacobian_row`` is True, degrees of freedom without a corresponding "
			 "Jacobian row/column for this residual are pinned to keep the Jacobian non-singular.")
		.def(
			"set_analytic_hessian_products", [](pyoomph::Problem *self, bool active, bool use_symmetry)
			{ if (active) self->set_analytic_hessian_products();  else self->unset_analytic_hessian_products(); self->set_symmetric_hessian_assembly(use_symmetry); },
			nb::arg("active"), nb::arg("use_symmetry") = false,
			"Select whether Hessian-vector products are computed analytically via the generated code (``active`` True) or by finite differences, and whether "
			"the Hessian assembly exploits its symmetry in the last two indices (``use_symmetry``).")
		.def("set_FD_step_used_in_get_hessian_vector_products", &pyoomph::Problem::set_FD_step_used_in_get_hessian_vector_products, nb::arg("step"),
			 "Set the finite-difference step size used when Hessian-vector products are computed by finite differences (see set_analytic_hessian_products()).")
		.def("are_hessian_products_calculated_analytically", &pyoomph::Problem::are_hessian_products_calculated_analytically,
			 "Whether Hessian-vector products are currently set to be computed analytically via the generated code (see set_analytic_hessian_products()), "
			 "rather than by finite differences.")
		.def("get_symmetric_hessian_assembly", &pyoomph::Problem::get_symmetric_hessian_assembly,
			 "Whether Hessian assembly currently exploits its symmetry in the last two indices (see set_analytic_hessian_products()).")
		.def("build_global_mesh", &pyoomph::Problem::build_global_mesh, "Combine all sub-meshes added via add_sub_mesh() into the problem's global mesh.")
		.def("rebuild_global_mesh", &pyoomph::Problem::rebuild_global_mesh, "Re-build the problem's global mesh from the current set of sub-meshes, e.g. after sub-meshes were modified.")
		.def("mesh_pt", [](pyoomph::Problem *self)
			 { return pyoomph_find_mesh_handle(self->mesh_pt()); }, "Return the problem's global mesh.")
		.def("mesh_pt", [](pyoomph::Problem *self, unsigned const &i)
			 { return pyoomph_find_mesh_handle(self->mesh_pt(i)); }, nb::arg("i"), "Return the ``i``-th sub-mesh of the problem.")
		.def("time_pt", (oomph::Time * &(pyoomph::Problem::*)()) & pyoomph::Problem::time_pt, nb::rv_policy::reference, "Return the problem's Time object, holding the current time and time step history.")
		.def("time_stepper_pt", (oomph::TimeStepper * &(pyoomph::Problem::*)(const unsigned &)) & pyoomph::Problem::time_stepper_pt, nb::rv_policy::reference, nb::arg("i") = 0,
			 "Return the ``i``-th time stepper added to the problem via add_time_stepper_pt() (default: the first/main one).")
		.def("shift_time_values", &pyoomph::Problem::shift_time_values,
			 "Shift the stored time history values of all data (copy current values to the storage locations of the previous time step), in preparation for the next time step.")
		.def("get_ccompiler", &pyoomph::Problem::get_ccompiler, "Return the C compiler currently used to compile generated element code.")
		.def("_set_ccompiler", &pyoomph::Problem::set_ccompiler, nb::keep_alive<1, 2>(), nb::arg("compiler"), "Set the C compiler used to compile generated element code.")
		.def("ntime_stepper", &pyoomph::Problem::ntime_stepper, "Return the number of time steppers added to the problem.")
		.def(
			"_assemble_multiassembly", [](pyoomph::Problem *p, std::vector<std::string> what, std::vector<std::string> contributions, std::vector<std::string> params, std::vector<std::vector<double>> hessian_vectors, std::vector<unsigned> &hessian_vector_indices)
			{
				std::vector<std::vector<double>> data;
				std::vector<std::vector<int>> csrdata;
				std::vector<int> return_indices;
				unsigned ndof;
				p->assemble_multiassembly(what, contributions, params, hessian_vectors, hessian_vector_indices, data, csrdata, ndof, return_indices);
				// data/csrdata are ragged (rows have different lengths, e.g. a dense
				// residual vector vs. a sparse matrix's CSR arrays), so each row is
				// individually wrapped as its own numpy array rather than nesting into
				// a single (non-rectangular) 2D array.
				std::vector<nb::ndarray<nb::numpy, double>> data_arrs;
				data_arrs.reserve(data.size());
				for (auto &row : data)
					data_arrs.push_back(vector_to_ndarray(row));
				std::vector<nb::ndarray<nb::numpy, int>> csrdata_arrs;
				csrdata_arrs.reserve(csrdata.size());
				for (auto &row : csrdata)
					csrdata_arrs.push_back(vector_to_ndarray(row));
				return std::make_tuple(ndof, data_arrs, csrdata_arrs, return_indices);
			},
			nb::arg("what"), nb::arg("contributions"), nb::arg("params"), nb::arg("hessian_vectors"), nb::arg("hessian_vector_indices"),
			"Assemble several residual vectors and/or Jacobian-like matrices (and, if requested, Hessian-vector products) in a single sweep over the elements. "
			"``what``/``contributions`` are parallel lists selecting what to assemble (e.g. \"residual\"/\"jacobian\") and with respect to which named residual "
			"combination; ``params`` optionally selects a parameter derivative for each entry; ``hessian_vectors``/``hessian_vector_indices`` request additional "
			"directional Hessian-vector products. Returns (ndof, data, csrdata, return_indices), where ``data``/``csrdata`` hold the assembled vectors and the CSR "
			"data (values, row_start, column_index pairs) of the assembled matrices.")
		.def("_assemble_defined_field_list", &pyoomph::Problem::assemble_defined_field_list,
			 "(Re-)build the internal list of all fields defined anywhere in the problem and their contributions to each residual/Jacobian combination; "
			 "must be called after the set of equations/fields changes, before assign_eqn_numbers().")
		.def(
			"distribute", [](pyoomph::Problem *self)
			{
#ifdef OOMPH_HAS_MPI
				self->distribute();
#endif
			},
			"Distribute the problem's meshes across the available MPI processes. No-op if pyoomph was built without MPI support.")
		.def(
			"load_balance", [](pyoomph::Problem *self)
			{
#ifdef OOMPH_HAS_MPI
				self->load_balance();
#endif
			},
			"Re-balance the distribution of the problem's meshes across the available MPI processes. No-op if pyoomph was built without MPI support.")
		.def("_build_mesh", &pyoomph::Problem::_build_mesh,
			 "Hook, overridable in Python, used to (re-)build the problem's mesh when required for load balancing in a distributed (MPI) problem.")
		.def("is_distributed", &pyoomph::Problem::distributed, "Return whether the problem's meshes are currently distributed across multiple MPI processes.")
		.def(
			"_redistribute_local_to_global_double_vector", [](pyoomph::Problem *self, const nb::ndarray<nb::numpy, double> &local_v)
			{
				double *in_ptr = local_v.data();
				size_t nloc = local_v.shape(0);
				auto *loc_distribution = self->linear_solver_pt()->distribution_pt();
				oomph::DoubleVector tmp(loc_distribution);
				int ndof = loc_distribution->nrow();
				for (unsigned int i = 0; i < nloc; i++)
					tmp[i] = in_ptr[i];
				oomph::LinearAlgebraDistribution global_distribution(loc_distribution->communicator_pt(), ndof, false);
				tmp.redistribute(&global_distribution);
				std::vector<double> res_vec(tmp.nrow());
				for (unsigned int i = 0; i < tmp.nrow(); i++)
					res_vec[i] = tmp[i];
				return vector_to_ndarray(res_vec);
			},
			nb::arg("local_vector"),
			"MPI utility: redistribute a vector given in the linear solver's local (per-process) distribution to a non-distributed, globally replicated vector.")
		.def(
			"_redistribute_global_to_local_double_vector", [](pyoomph::Problem *self, const nb::ndarray<nb::numpy, double> &global_v)
			{
				double *in_ptr = global_v.data();
				size_t nglob = global_v.shape(0);
				auto *loc_distribution = self->linear_solver_pt()->distribution_pt();
				oomph::LinearAlgebraDistribution global_distribution(loc_distribution->communicator_pt(), nglob, false);
				oomph::DoubleVector tmp(&global_distribution);
				for (unsigned int i = 0; i < nglob; i++)
					tmp[i] = in_ptr[i];
				tmp.redistribute(loc_distribution);
				unsigned nloc = loc_distribution->nrow_local();
				std::vector<double> res_vec(nloc);
				for (unsigned int i = 0; i < nloc; i++)
					res_vec[i] = tmp[i];
				return vector_to_ndarray(res_vec);
			},
			nb::arg("global_vector"),
			"MPI utility: redistribute a non-distributed, globally replicated vector to the linear solver's local (per-process) distribution.")
		.def("ndof", &pyoomph::Problem::ndof, "Returns the number of equations, i.e. degrees of freedom")
		.def("ensure_dummy_values_to_be_dummy", &pyoomph::Problem::ensure_dummy_values_to_be_dummy,
			 "Sanity-check/reset helper that ensures dummy (unused, placeholder) degrees of freedom hold their expected dummy value.")
		.def(
			"generate_and_compile_bulk_element_code", [](pyoomph::Problem *problem, pyoomph::FiniteElementCode *my_element, std::string code_trunk, bool suppress_writing, bool suppress_compilation, MeshHandleBase *bulkmesh_h, bool quiet, const std::vector<std::string> &extra_flags)
			{
				pyoomph::Mesh *bulkmesh = bulkmesh_h->mesh();
				// Generate Hessian if desired
				my_element->set_problem(problem);
				my_element->generate_hessian = problem->are_hessian_products_calculated_analytically();
				my_element->assemble_hessian_by_symmetry = problem->get_symmetric_hessian_assembly();
				if (suppress_writing)
				{
					std::ostringstream oss; // TODO Null stream instead
					if (!quiet)
						std::cout << "Generating equation C code, but do not write to any file" << std::endl;
					my_element->write_code(oss);
				}
				else
				{
					std::ofstream ofs(code_trunk + ".c");
					if (!quiet)
						std::cout << "Generating equation C code: " << code_trunk << std::endl;
					my_element->write_code(ofs);
					// std::ofstream hfs(code_trunk+".gar",std::ios::binary);
					// hfs << my_element->archive;
					// hfs.close();
				}

#ifdef OOMPH_HAS_MPI
				MPI_Barrier(problem->communicator_pt()->mpi_comm());
#endif

				pyoomph::CCompiler *compiler = problem->get_ccompiler();
				if (!compiler)
				{
					throw_runtime_error("No C compiler set");
				}
				compiler->set_code_from_file(code_trunk);

				if (!suppress_compilation)
				{
					// "Compiling equation C code" is printed from CCompiler::compile() itself (see
					// BaseCCompiler.compile() in ccompiler.py), not here - it is only known once
					// compile() actually decides it needs to invoke the compiler, i.e. never on a
					// JIT cache hit.
					compiler->compile(suppress_compilation, suppress_writing, quiet, extra_flags);
				}

#ifdef OOMPH_HAS_MPI
				MPI_Barrier(problem->communicator_pt()->mpi_comm());
#endif

				std::string lib = compiler->get_shared_library(code_trunk);
				pyoomph::DynamicBulkElementCode *code = problem->load_dynamic_bulk_element_code(lib, my_element);
				pyoomph::DynamicBulkElementInstance *code_instance = code->factory_instance(bulkmesh);
				return code_instance;
			},
			nb::rv_policy::reference,
			nb::arg("my_element"), nb::arg("code_trunk"), nb::arg("suppress_writing"), nb::arg("suppress_compilation"), nb::arg("bulkmesh"), nb::arg("quiet"), nb::arg("extra_flags"),
			"Generate the C source code for the symbolic finite element definition ``my_element`` (writing it to ``code_trunk``.c unless ``suppress_writing`` is True), "
			"compile it with the problem's C compiler (unless ``suppress_compilation`` is True, passing ``extra_flags`` to the compiler) and load the resulting shared "
			"library, instantiating it on ``bulkmesh``. Returns the resulting DynamicBulkElementInstance. ``quiet`` suppresses progress output.");
}
