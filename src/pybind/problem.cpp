/*================================================================================
pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
Copyright (C) 2021-2025  Christian Diddens & Duarte Rocha

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
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <fstream>
#include <tuple>

namespace py = pybind11;

#include <fstream>

#include "../problem.hpp"
#include "../bifurcation.hpp"
#include "../expressions.hpp"
#include "../mesh.hpp"
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

	class PyProblemTrampoline : public pyoomph::Problem
	{
	public:
		using pyoomph::Problem::Problem;

		void setup_pinning() override
		{
			PYBIND11_OVERLOAD(
				void,			  /* Return type */
				pyoomph::Problem, /* Parent class */
				setup_pinning	  //,          /* Name of function in C++ (must match Python name) */
								  //            n_times      /* Argument(s) */
			);
		}

		void set_initial_condition() override
		{
			PYBIND11_OVERLOAD(
				void,				  /* Return type */
				pyoomph::Problem,	  /* Parent class */
				set_initial_condition //,          /* Name of function in C++ (must match Python name) */
									  //            n_times      /* Argument(s) */
			);
		}

		std::pair<unsigned, unsigned> _adapt() override
		{
			typedef std::pair<double, double> pairuint;
			PYBIND11_OVERLOAD(pairuint, pyoomph::Problem, _adapt);
		}

		void actions_before_newton_solve() override
		{
			PYBIND11_OVERLOAD(void, pyoomph::Problem, actions_before_newton_solve);
		}

		void actions_after_newton_solve() override
		{
			PYBIND11_OVERLOAD(void, pyoomph::Problem, actions_after_newton_solve);
		}

		void actions_before_newton_convergence_check() override
		{
			PYBIND11_OVERLOAD(void, pyoomph::Problem, actions_before_newton_convergence_check);
		}

		void actions_after_change_in_global_parameter(const std::string &paramname) override
		{
			PYBIND11_OVERLOAD(void, pyoomph::Problem, actions_after_change_in_global_parameter, paramname);
		}

		void actions_after_parameter_increase(const std::string &paramname) override
		{
			PYBIND11_OVERLOAD(void, pyoomph::Problem, actions_after_parameter_increase, paramname);
		}

		void actions_after_newton_step() override
		{
			PYBIND11_OVERLOAD(void, pyoomph::Problem, actions_after_newton_step);
		}

		void actions_before_newton_step() override
		{
			PYBIND11_OVERLOAD(void, pyoomph::Problem, actions_before_newton_step);
		}

		void actions_before_adapt() override
		{
			PYBIND11_OVERLOAD(void, pyoomph::Problem, actions_before_adapt);
		}

		void actions_after_adapt() override
		{
			PYBIND11_OVERLOAD(void, pyoomph::Problem, actions_after_adapt);
		}

		void actions_before_distribute() override
		{
			PYBIND11_OVERLOAD(void, pyoomph::Problem, actions_before_distribute);
		}

		void actions_after_distribute() override
		{
			PYBIND11_OVERLOAD(void, pyoomph::Problem, actions_after_distribute);
		}

		void get_custom_residuals_jacobian(pyoomph::CustomResJacInformation *info) override
		{
			PYBIND11_OVERLOAD(void, pyoomph::Problem, get_custom_residuals_jacobian, info);
		}
	};

}

static py::class_<GiNaC::GiNaCGlobalParameterWrapper> *py_decl_GlobalParam = NULL;
//static py::class_<pyoomph::DofAugmentations> *py_decl_DofAugmentations=NULL;
void PyDecl_Problem(py::module &m)
{
	py_decl_GlobalParam = new py::class_<GiNaC::GiNaCGlobalParameterWrapper>(m, "GiNaC_GlobalParam");
//	py_decl_DofAugmentations=new py::class_<pyoomph::DofAugmentations>(m,"DofAugmentations");
}

void PyReg_Problem(py::module &m)
{

	m.def("InitMPI", [](std::vector<std::string> &args)
		  {
				std::vector<char*> argv(args.size(),NULL);
				for (unsigned int i=0;i<args.size();i++) {
						unsigned l=strlen(args[i].c_str());
						argv[i]=(char*)malloc(sizeof(char)*(l+1));
						strncpy(argv[i],args[i].c_str(),l);
						argv[i][l]='\0';
				}
				oomph::MPI_Helpers::init(args.size(),&(argv[0])); });

	m.def("FinaliseMPI", &oomph::MPI_Helpers::finalize);
	
	m.def("_write_to_log_file",pyoomph::write_to_log_file);

	m.def("_get_core_information",[](){
		std::map<std::string,std::string> info;
		/*#ifdef VERSION_INFO
			info["core_version"]=VERSION_INFO;
		#endif*/
		return info;
	});

	m.def("get_verbosity_flag", []()
		  { return pyoomph::pyoomph_verbose; });
	m.def("set_verbosity_flag", [](int v)
		  { pyoomph::pyoomph_verbose = v; });

	m.def("feenableexcept", []()
		  {
#ifdef __gnu_linux__
			  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#else
			  throw_runtime_error("feenableexcept not supported on this OS");
#endif
		  }

	);


	//py_decl_DofAugmentations->
	py::class_<pyoomph::DofAugmentations>(m,"DofAugmentations")
		.def("add_vector", &pyoomph::DofAugmentations::add_vector)
		.def("add_scalar", &pyoomph::DofAugmentations::add_scalar)
		.def("add_parameter", &pyoomph::DofAugmentations::add_parameter)
		.def("split", [](pyoomph::DofAugmentations &augs, unsigned startindex,int endindex)
			{
				std::vector<std::vector<double>> res=augs.split(startindex,endindex);
				std::vector<py::array_t<double>> resA(res.size());
				for (unsigned int i=0;i<res.size();i++) resA[i]=py::cast(res[i]);
				return resA;
			},"splits the augmented dof vector into its components. By default, all augmented dofs (without base dofs), but can be controlled by startindex and endindex",py::arg("startindex")=1,py::arg("endindex")=-1);

	py_decl_GlobalParam->def_property(
						   "value", [](GiNaC::GiNaCGlobalParameterWrapper *self)
						   { return self->get_struct().cme->value(); },
						   [](GiNaC::GiNaCGlobalParameterWrapper *self, const double &v)
						   { 
							if (v<0 && self->get_struct().cme->is_restricted_to_positive_values()) throw_runtime_error("Cannot set the parameter "+self->get_struct().cme->get_name()+" to a negative value of "+std::to_string(v)+" since it is restricted to positive values.");
							self->get_struct().cme->value() = v; 
						   })
		.def_property(
			"analytical_derivative", [](GiNaC::GiNaCGlobalParameterWrapper *self)
			{ return self->get_struct().cme->get_analytic_derivative(); },
			[](GiNaC::GiNaCGlobalParameterWrapper *self, const bool &v)
			{ self->get_struct().cme->set_analytic_derivative(v); })
		.def("get_symbol", [](GiNaC::GiNaCGlobalParameterWrapper *self) -> GiNaC::ex
			 { return 0 + (*self); })
		.def("get_name", [](GiNaC::GiNaCGlobalParameterWrapper *self)
			 { return self->get_struct().cme->get_name(); })
		.def("restrict_to_positive_values",[](GiNaC::GiNaCGlobalParameterWrapper *self){self->get_struct().cme->restrict_to_positive_values();})

		.def(-py::self)

		.def(py::self + py::self)
		.def(py::self + GiNaC::ex())
		.def(int() + py::self)
		.def(py::self + int())
		.def(float() + py::self)
		.def(py::self + float())

		.def(py::self - py::self)
		.def(py::self - GiNaC::ex())
		.def(int() - py::self)
		.def(py::self - int())
		.def(float() - py::self)
		.def(py::self - float())

		.def(py::self * py::self)
		.def(py::self * GiNaC::ex())
		.def(int() * py::self)
		.def(py::self * int())
		.def(float() * py::self)
		.def(py::self * float())

		.def(py::self / py::self)
		.def(py::self / GiNaC::ex())
		.def(int() / py::self)
		.def(py::self / int())
		.def(float() / py::self)
		.def(py::self / float())

		.def("__repr__", [](const GiNaC::GiNaCGlobalParameterWrapper &self)
			 {   	 std::ostringstream oss; 	 GiNaC::print_python pypc(oss); 	 (self+0).print(pypc);	 return oss.str(); })

		.def("__pow__", [](const GiNaC::GiNaCGlobalParameterWrapper &lh, const GiNaC::ex &rh)
			 { return GiNaC::pow(lh, rh); }, py::is_operator())
		.def("__pow__", [](const GiNaC::GiNaCGlobalParameterWrapper &lh, const int &rh)
			 { return GiNaC::pow(lh, rh); }, py::is_operator())
		.def("__pow__", [](const GiNaC::GiNaCGlobalParameterWrapper &lh, const double &rh)
			 { return GiNaC::pow(lh, rh); }, py::is_operator())

		.def("__rpow__", [](const GiNaC::GiNaCGlobalParameterWrapper &rh, const int &lh)
			 { return GiNaC::pow(lh, rh); }, py::is_operator())
		.def("__rpow__", [](const GiNaC::GiNaCGlobalParameterWrapper &rh, const int &lh)
			 { return GiNaC::pow(lh, rh); }, py::is_operator())
		.def("__rpow__", [](const GiNaC::GiNaCGlobalParameterWrapper &rh, const double &lh)
			 { return GiNaC::pow(lh, rh); }, py::is_operator())
		.def("__float__", [](const GiNaC::GiNaCGlobalParameterWrapper &self)
			 { return self.get_struct().cme->value(); })

		;

	py::class_<pyoomph::SparseRank3Tensor>(m, "SparseRank3Tensor")
		.def("get_entries", &pyoomph::SparseRank3Tensor::get_entries)
		.def("finalize_for_vector_product", [](pyoomph::SparseRank3Tensor &self) -> std::tuple<py::array_t<int>, py::array_t<int>>
			 {
       std::vector<int> col_index;
       std::vector<int> row_start;
       std::tie(col_index,row_start)=self.finalize_for_vector_product();
       return std::make_tuple(py::cast(col_index),py::cast(row_start)); })
		.def("right_vector_mult", [](pyoomph::SparseRank3Tensor &self, const py::array_t<double> &vec) -> py::array_t<double>
			 {
       py::buffer_info binfo = vec.request();
       auto ptr = static_cast<double *>(binfo.ptr);
       size_t size=1; for (auto s : binfo.shape) size*=s;
       std::vector<double> v(ptr,ptr+size);
       
       std::vector<double> vals=self.right_vector_mult(v);
       return py::cast(vals); });

	py::class_<pyoomph::CustomResJacInformation>(m, "CustomResJacInfo")
		.def("require_jacobian", &pyoomph::CustomResJacInformation::require_jacobian)
		.def("get_parameter_name", &pyoomph::CustomResJacInformation::get_parameter_name)
		.def("set_custom_residuals", [](pyoomph::CustomResJacInformation &self, py::array_t<double> r)
			 {
       py::buffer_info binfo = r.request();
       auto ptr = static_cast<double *>(binfo.ptr);
       size_t size=1; for (auto s : binfo.shape) size*=s;
       std::vector<double> R(ptr,ptr+size);
       self.set_custom_residuals(R); })
		.def("set_custom_jacobian", [](pyoomph::CustomResJacInformation &self, py::array_t<double> jvals, py::array_t<int> colindex, py::array_t<int> rowstart)
			 {
       py::buffer_info binfo = jvals.request();
       auto dptr = static_cast<double *>(binfo.ptr);
       size_t size=1; for (auto s : binfo.shape) size*=s;       
       std::vector<double> V(dptr,dptr+size);
       
       binfo = colindex.request();
       auto iptr = static_cast<int *>(binfo.ptr);
       size=1; for (auto s : binfo.shape) size*=s;              
       std::vector<int> I(iptr,iptr+size);       
       
       binfo = rowstart.request();
       iptr = static_cast<int *>(binfo.ptr);
       size=1; for (auto s : binfo.shape) size*=s;                     
       std::vector<int> J(iptr,iptr+size);              
       
       self.set_custom_jacobian(V,I,J); });

	py::class_<oomph::AssemblyHandler>(m, "AssemblyHandler");

	py::class_<pyoomph::MyFoldHandler, oomph::AssemblyHandler>(m, "FoldHandler")
		.def("get_eigenfunction", [](pyoomph::MyFoldHandler *self)
			 {
	 oomph::Vector<oomph::DoubleVector> efuncs;
	 self->get_eigenfunction(efuncs);

	 std::vector<unsigned> dims{static_cast<unsigned>(efuncs.size()),self->get_problem_ndof()};
	 py::array_t<double> res=py::array_t<double>(dims);
	 double * dest=(double*)res.request().ptr;

	 for (unsigned int i=0;i<efuncs.size();i++)
	 {
  	   for (unsigned int j=0;j<efuncs.size();j++)
	   {
	    *dest=efuncs[i][j];
	    dest++;
	   }
	 }
	 return res; })
		.def_property("FD_step", [](pyoomph::MyFoldHandler *h)
					  { return h->FD_step; }, [](pyoomph::MyFoldHandler *h, double s)
					  { h->FD_step = s; })
		.def("set_eigenweight", &pyoomph::MyFoldHandler::set_eigenweight)
		.def_property("symmetric_FD", [](pyoomph::MyFoldHandler *h)
					  { return h->symmetric_FD; }, [](pyoomph::MyFoldHandler *h, bool s)
					  { h->symmetric_FD = s; });

	py::class_<pyoomph::MyPitchForkHandler, oomph::AssemblyHandler>(m, "PitchForkHandler")
		.def("set_eigenweight", &pyoomph::MyPitchForkHandler::set_eigenweight);

	py::class_<pyoomph::MyHopfHandler, oomph::AssemblyHandler>(m, "HopfHandler")
		.def("get_nicely_rotated_eigenfunction", &pyoomph::MyHopfHandler::get_nicely_rotated_eigenfunction)
		.def("set_eigenweight", &pyoomph::MyHopfHandler::set_eigenweight)	
		.def("debug_analytical_filling", [](pyoomph::MyHopfHandler *self, oomph::GeneralisedElement *elem, double eps)		
			 { self->debug_analytical_filling(elem, eps); });

	py::class_<pyoomph::AzimuthalSymmetryBreakingHandler, oomph::AssemblyHandler>(m, "AzimuthalSymmetryBreakingHandler")
		.def("set_eigenweight", &pyoomph::AzimuthalSymmetryBreakingHandler::set_eigenweight)	
		.def("set_global_equations_forced_zero", &pyoomph::AzimuthalSymmetryBreakingHandler::set_global_equations_forced_zero);

	py::class_<pyoomph::PeriodicOrbitHandler, oomph::AssemblyHandler>(m, "PeriodicOrbitHandler")
		.def("backup_dofs", &pyoomph::PeriodicOrbitHandler::backup_dofs)	
		.def("restore_dofs", &pyoomph::PeriodicOrbitHandler::restore_dofs)	
		.def("get_base_ndof",&pyoomph::PeriodicOrbitHandler::get_problem_ndof)
		.def("is_floquet_mode",&pyoomph::PeriodicOrbitHandler::is_floquet_mode)
		.def("get_T",&pyoomph::PeriodicOrbitHandler::get_T)
		.def("get_num_time_steps",&pyoomph::PeriodicOrbitHandler::n_tsteps)
		.def("get_s_integration_samples",&pyoomph::PeriodicOrbitHandler::get_s_integration_samples)
		.def("update_phase_constraint_information",&pyoomph::PeriodicOrbitHandler::update_phase_constraint_information)
		.def("set_dofs_to_interpolated_values", &pyoomph::PeriodicOrbitHandler::set_dofs_to_interpolated_values)	;

/*
	class PythonAssemblyHandlerTrampoline : public pyoomph::PythonAssemblyHandler
	{
	public:
		using pyoomph::PythonAssemblyHandler::PythonAssemblyHandler;

	};

	py::class_<pyoomph::PythonAssemblyHandler,PythonAssemblyHandlerTrampoline,oomph::AssemblyHandler>(m,"PythonAssemblyHandler")	
		.def(py::init<>());
		//.def("_after_construction", &pyoomph::PythonAssemblyHandler::_after_construction);
*/


	py::class_<pyoomph::DynamicBulkElementInstance>(m, "DynamicBulkElementInstance")
		.def("_exchange_mesh", &pyoomph::DynamicBulkElementInstance::set_bulk_mesh)
		.def("link_external_data", &pyoomph::DynamicBulkElementInstance::link_external_data)
		.def("get_nodal_field_index", &pyoomph::DynamicBulkElementInstance::get_nodal_field_index)
		.def("get_discontinuous_field_index", &pyoomph::DynamicBulkElementInstance::get_discontinuous_field_index)
		.def("has_moving_nodes", &pyoomph::DynamicBulkElementInstance::has_moving_nodes)
		.def("get_max_dt_order", &pyoomph::DynamicBulkElementInstance::get_max_dt_order)
		.def("can_be_time_adaptive", &pyoomph::DynamicBulkElementInstance::can_be_time_adaptive)
		.def("has_parameter_contribution",&pyoomph::DynamicBulkElementInstance::has_parameter_contribution)
		.def("get_nodal_field_indices", &pyoomph::DynamicBulkElementInstance::get_nodal_field_indices)
		.def("set_analytical_jacobian", [](pyoomph::DynamicBulkElementInstance *self, bool ana, bool anapos)
			 {
		    self->get_func_table()->fd_jacobian=!ana;
		    self->get_func_table()->fd_position_jacobian=!anapos; }, py::arg("analytic"), py::arg("analytic_positions"))
		.def("get_elemental_field_indices", &pyoomph::DynamicBulkElementInstance::get_elemental_field_indices);

	py::class_<pyoomph::Problem, pyoomph::PyProblemTrampoline>(m, "Problem")
		.def(py::init<>())
		.def("assembly_handler_pt", (oomph::AssemblyHandler * &(pyoomph::Problem::*)()) & pyoomph::Problem::assembly_handler_pt, py::return_value_policy::reference)
		.def("enable_store_local_dof_pt_in_elements", &pyoomph::Problem::enable_store_local_dof_pt_in_elements)
		.def("setup_pinning", &pyoomph::Problem::setup_pinning)
		.def("set_initial_condition", &pyoomph::Problem::set_initial_condition)
		.def("refine_uniformly", (void(pyoomph::Problem::*)()) & pyoomph::Problem::refine_uniformly)
		.def("unrefine_uniformly", (unsigned(pyoomph::Problem::*)()) & pyoomph::Problem::unrefine_uniformly)
		.def("assign_eqn_numbers", &pyoomph::Problem::assign_eqn_numbers)
		.def("initialise_dt", (void(pyoomph::Problem::*)(const double &)) & pyoomph::Problem::initialise_dt)
		.def("assign_initial_values_impulsive", (void(pyoomph::Problem::*)(const double &)) & pyoomph::Problem::assign_initial_values_impulsive)
		.def("assign_initial_values_impulsive", (void(pyoomph::Problem::*)()) & pyoomph::Problem::assign_initial_values_impulsive)
		.def("get_last_jacobian_setup_time", [](pyoomph::Problem *self)
			 { return self->linear_solver_pt()->jacobian_setup_time(); })
		.def("get_last_linear_solver_solution_time", [](pyoomph::Problem *self)
			 { return self->linear_solver_pt()->linear_solver_solution_time(); })
		.def_property(
			"max_residuals", [](pyoomph::Problem &p)
			{ return p.max_residuals(); },
			[](pyoomph::Problem &p, const double &r)
			{ p.max_residuals() = r; })
		.def_property("DTSF_minimum_dt", [](pyoomph::Problem &p) {return p.minimum_dt();}, [](pyoomph::Problem &p, double v) {p.minimum_dt()=v;})
		.def("_set_globally_convergent_newton_method", [](pyoomph::Problem &p, bool r)
			 {if (r) p.enable_globally_convergent_newton_method(); else p.disable_globally_convergent_newton_method(); })
		.def_property(
			"max_newton_iterations", [](pyoomph::Problem &p)
			{ return p.max_newton_iterations(); },
			[](pyoomph::Problem &p, const unsigned &r)
			{ p.max_newton_iterations() = r; },"Maximum number of Newton iterations for solving before giving up.")
		.def_property(
			"newton_solver_tolerance", [](pyoomph::Problem &p)
			{ return p.newton_solver_tolerance(); },
			[](pyoomph::Problem &p, const double &r)
			{ p.newton_solver_tolerance() = r; },"Maximum value in the residual vector to consider the solution as converged during Newton method")
		.def_property(
			"always_take_one_newton_step", [](pyoomph::Problem &p)
			{ return p.always_take_one_newton_step(); },
			[](pyoomph::Problem &p, const bool &b)
			{ p.always_take_one_newton_step() = b; })
		.def_property(
			"newton_relaxation_factor", [](pyoomph::Problem &p)
			{ return p.newton_relaxation_factor(); },
			[](pyoomph::Problem &p, const double &r)
			{ p.newton_relaxation_factor() = r; })
		.def_property(
			"DTSF_max_increase_factor", [](pyoomph::Problem &p)
			{ return p.DTSF_max_increase_factor(); },
			[](pyoomph::Problem &p, const double &r)
			{ p.DTSF_max_increase_factor() = r; })
		.def_property(
			"DTSF_min_decrease_factor", [](pyoomph::Problem &p)
			{ return p.DTSF_min_decrease_factor(); },
			[](pyoomph::Problem &p, const double &r)
			{ p.DTSF_min_decrease_factor() = r; })
		.def_property(
			"minimum_arclength_ds", [](pyoomph::Problem &p)
			{ return p.minimum_ds(); },
			[](pyoomph::Problem &p, const double &r)
			{ p.minimum_ds() = r; })
		.def_property(
			"keep_temporal_error_below_tolerance", [](pyoomph::Problem &p)
			{ return p.get_Keep_temporal_error_below_tolerance(); },
			[](pyoomph::Problem &p, bool s)
			{ p.set_Keep_temporal_error_below_tolerance(s); })
		.def_property("use_custom_residual_jacobian", [](pyoomph::Problem &p)
					  { return p.use_custom_residual_jacobian; }, [](pyoomph::Problem &p, bool s)
					  { p.use_custom_residual_jacobian = s; })
		.def_property("_improved_pitchfork_tracking_on_unstructured_meshes", [](pyoomph::Problem &p)
					  { return p.improved_pitchfork_tracking_on_unstructured_meshes; }, [](pyoomph::Problem &p, bool s)
					  { p.improved_pitchfork_tracking_on_unstructured_meshes = s; })
		.def_property("sparse_assembly_method", &pyoomph::Problem::get_sparse_assembly_method,&pyoomph::Problem::set_sparse_assembly_method)
		.def("adaptive_unsteady_newton_solve", (double(pyoomph::Problem::*)(const double &, const double &)) & pyoomph::Problem::adaptive_unsteady_newton_solve)
		.def("_adapt", &pyoomph::Problem::_adapt)
		.def("adaptive_unsteady_newton_solve", (double(pyoomph::Problem::*)(const double &, const double &, const bool &)) & pyoomph::Problem::adaptive_unsteady_newton_solve)
		.def("unsteady_newton_solve", (void(pyoomph::Problem::*)(const double &, const bool &)) & pyoomph::Problem::unsteady_newton_solve)
		.def("unsteady_newton_solve", (void(pyoomph::Problem::*)(const double &, const unsigned &, const bool &, const bool &)) & pyoomph::Problem::unsteady_newton_solve)
		.def("doubly_adaptive_unsteady_newton_solve", (double(pyoomph::Problem::*)(const double &, const double &, const unsigned &, const unsigned &, const bool &, const bool &)) & pyoomph::Problem::doubly_adaptive_unsteady_newton_solve)
		.def("newton_solve", (void(pyoomph::Problem::*)(unsigned const &)) & pyoomph::Problem::newton_solve, "Perform a newton solve", py::arg("max_adapt") = 0)
		.def("steady_newton_solve", (void(pyoomph::Problem::*)(unsigned const &)) & pyoomph::Problem::steady_newton_solve, "Perform a steady newton solve", py::arg("max_adapt") = 0)
		.def("_arc_length_step", &pyoomph::Problem::arc_length_step)
		.def("get_arc_length_parameter_derivative", &pyoomph::Problem::get_arc_length_parameter_derivative)
		.def("_set_arc_length_parameter_derivative", &pyoomph::Problem::set_arc_length_parameter_derivative)
		.def("_update_dof_vectors_for_continuation", &pyoomph::Problem::update_dof_vectors_for_continuation)
		.def("_update_param_info_for_continuation", &pyoomph::Problem::update_param_info_for_continuation)
		.def("get_arc_length_theta_sqr", &pyoomph::Problem::get_arc_length_theta_sqr)
		.def("_set_arc_length_theta_sqr", &pyoomph::Problem::set_arc_length_theta_sqr)
		.def("_set_arclength_parameter", &pyoomph::Problem::set_arclength_parameter)
		.def("_start_bifurcation_tracking", &pyoomph::Problem::start_bifurcation_tracking)
		.def("_start_orbit_tracking", &pyoomph::Problem::start_orbit_tracking)
		.def("_get_max_dt_order", &pyoomph::Problem::get_max_dt_order)
		//.def("_start_custom_augmented_system", &pyoomph::Problem::start_custom_augmented_system)
		.def("_reset_augmented_dof_vector_to_nonaugmented", &pyoomph::Problem::reset_augmented_dof_vector_to_nonaugmented)
		.def("_create_dof_augmentation",&pyoomph::Problem::create_dof_augmentation,py::return_value_policy::take_ownership)
		.def("_get_n_unaugmented_dofs", &pyoomph::Problem::get_n_unaugmented_dofs)		
		.def("_add_augmented_dofs", &pyoomph::Problem::add_augmented_dofs)
		.def("_enable_store_local_dof_pt_in_elements", &pyoomph::Problem::enable_store_local_dof_pt_in_elements)
		.def("after_bifurcation_tracking_step", &pyoomph::Problem::after_bifurcation_tracking_step)
		.def("get_custom_residuals_jacobian", &pyoomph::Problem::get_custom_residuals_jacobian, py::arg("info"))
		.def("get_bifurcation_tracking_mode", &pyoomph::Problem::get_bifurcation_tracking_mode)
		.def("_get_bifurcation_eigenvector", &pyoomph::Problem::get_bifurcation_eigenvector)
		.def("_get_bifurcation_omega", &pyoomph::Problem::get_bifurcation_omega)
		.def("_get_lambda_tracking_real", [](pyoomph::Problem * self) {return *self->get_lambda_tracking_real(); })
		.def("_set_lambda_tracking_real", [](pyoomph::Problem * self,double lr) {*self->get_lambda_tracking_real()=lr; })
		//.def("reset_arc_length_parameters", &pyoomph::Problem::reset_arc_length_parameters)
		.def("reset_arc_length_parameters", [](pyoomph::Problem * self) {self->reset_arc_length_parameters(); })
		.def("_set_dof_direction_arclength", &pyoomph::Problem::set_dof_direction_arclength)		
		.def("get_parameter_derivative", &pyoomph::Problem::get_parameter_derivative)
		.def("get_arclength_dof_derivative_vector", &pyoomph::Problem::get_arclength_dof_derivative_vector)
		.def("get_arclength_dof_current_vector", &pyoomph::Problem::get_arclength_dof_current_vector)
		.def("get_global_parameter", [](pyoomph::Problem *self, const std::string &n) -> GiNaC::GiNaCGlobalParameterWrapper
			 {auto * gpd=self->assert_global_parameter(n); return GiNaC::GiNaCGlobalParameterWrapper(gpd); }, py::return_value_policy::reference, py::arg("parameter_name"), "Return a global parameter. If it does not exist, it will be added and initialized with value 0.")
		.def("get_global_parameter_names", &pyoomph::Problem::get_global_parameter_names)
		.def("get_current_dofs", [](pyoomph::Problem *self)
			 { 
				auto rs=self->get_current_dofs();
				return std::make_tuple(py::array_t<double>(std::get<0>(rs).size(),std::get<0>(rs).data()), std::get<1>(rs));
				//return self->get_current_dofs(); 
			})
		.def("get_history_dofs", [](pyoomph::Problem *self, unsigned t)
			 { 
			   auto rs=self->get_history_dofs(t);			   
			   return py::array_t<double>(rs.size(), rs.data()); })
		.def("get_last_residual_convergence", &pyoomph::Problem::get_last_residual_convergence)
		.def("get_residuals", [](pyoomph::Problem *self)
			 {
			oomph::DoubleVector ov;
			self->get_residuals(ov);
			std::vector<double> res(self->ndof());
		   for (unsigned int i=0;i<self->ndof();i++) res[i]=ov[i];
			return res; })
		.def("get_current_pinned_values", [](pyoomph::Problem *self, bool with_pos)
			 { return self->get_current_pinned_values(with_pos); })
		.def("set_current_dofs", [](pyoomph::Problem *self, const std::vector<double> &inp)
			 { return self->set_current_dofs(inp); })
		.def("set_history_dofs", [](pyoomph::Problem *self, unsigned t, const std::vector<double> &inp)
			{ return self->set_history_dofs(t, inp); })
		.def("set_current_pinned_values", [](pyoomph::Problem *self, const std::vector<double> &inp, bool with_pos,unsigned t)
			 { return self->set_current_pinned_values(inp, with_pos,t); },py::arg("inp"),py::arg("with_pos"),py::arg("t")=0)
		.def("assemble_eigenproblem_matrices", [](pyoomph::Problem *self, double sigma_r)
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

				 py::array_t<double> M_values_arr({M_nzz}, {sizeof(double)}, M_values, py::capsule(M_values, [](void *f) {}));
				 py::array_t<int> M_colindex_arr({M_nzz}, {sizeof(int)}, M_colindex, py::capsule(M_colindex, [](void *f) {}));
				 py::array_t<int> M_row_start_arr({M_nrow_local + 1}, {sizeof(int)}, M_row_start, py::capsule(M_row_start, [](void *f) {}));

				 py::array_t<double> J_values_arr({J_nzz}, {sizeof(double)}, J_values, py::capsule(J_values, [](void *f) {}));
				 py::array_t<int> J_colindex_arr({J_nzz}, {sizeof(int)}, J_colindex, py::capsule(J_colindex, [](void *f) {}));
				 py::array_t<int> J_row_start_arr({J_nrow_local + 1}, {sizeof(int)}, J_row_start, py::capsule(J_row_start, [](void *f) {}));

				 return std::make_tuple(n, M_nzz, M_nrow_local, M_values_arr, M_colindex_arr, M_row_start_arr, J_nzz, J_nrow_local, J_values_arr, J_colindex_arr, J_row_start_arr); })
		.def("_assemble_residual_jacobian", [](pyoomph::Problem *self, std::string name)
			 {
	    std::string oldresi=self->_get_solved_residual();
	    if (name!=oldresi) self->_set_solved_residual(name);
	    oomph::DoubleVector resi;
	    oomph::CRDoubleMatrix J;
	    self->get_jacobian(resi,J);
	    unsigned J_nzz=J.nnz();
		 unsigned n=J.distribution_pt()->nrow();
       double* J_values = J.value();
		 unsigned int J_nrow_local=J.nrow_local();
 		 int *J_row_start=J.row_start(); //nrow_local+1
		 int * J_colindex=J.column_index(); //nnz_local
       if (name!=oldresi) self->_set_solved_residual(oldresi);

 	    py::array_t<double> J_values_arr=py::array_t<double>({J_nzz});
	    double * dest=(double*)J_values_arr.request().ptr;
	    for (unsigned int j=0;j<J_nzz;j++) dest[j]=J_values[j];

 	    py::array_t<int> J_colindex_arr=py::array_t<int>({J_nzz});
	    int *idest=(int*)J_colindex_arr.request().ptr;
	    for (unsigned int j=0;j<J_nzz;j++) idest[j]=J_colindex[j];

 	    py::array_t<int> J_row_start_arr=py::array_t<int>({J_nrow_local+1});
	    idest=(int*)J_row_start_arr.request().ptr;
	    for (unsigned int j=0;j<J_nrow_local+1;j++) idest[j]=J_row_start[j];


		 std::vector<double> res(n);
		 for (unsigned int i=0;i<n;i++) res[i]=resi[i];
       return std::make_tuple(res,n,J_nzz,J_nrow_local,J_values_arr,J_colindex_arr,J_row_start_arr); })
		.def("quiet", &pyoomph::Problem::quiet, py::arg("quiet") = true, "Deactivate output messages from the oomph-lib and pyoomph C++ core")
		.def("_open_log_file", &pyoomph::Problem::open_log_file,py::arg("fname"),py::arg("activate_logging")=true,"Open a log file for the problem")
		.def("_assemble_hessian_tensor", &pyoomph::Problem::assemble_hessian_tensor)
		.def("is_quiet", &pyoomph::Problem::is_quiet)
		.def("_unload_all_dlls", &pyoomph::Problem::unload_all_dlls)
		.def("add_time_stepper_pt", &pyoomph::Problem::add_time_stepper_pt, py::keep_alive<1, 2>())
		.def("set_mesh_pt", &pyoomph::Problem::set_mesh_pt, py::keep_alive<1, 2>())
		.def("add_sub_mesh", &pyoomph::Problem::add_sub_mesh, py::keep_alive<1, 2>())
		.def("flush_sub_meshes", &pyoomph::Problem::flush_sub_meshes)
		.def("get_second_order_directional_derivative", &pyoomph::Problem::get_second_order_directional_derivative)
		.def("nsub_mesh", &pyoomph::Problem::nsub_mesh)
		.def("adapt", [](pyoomph::Problem &self)
			 {unsigned nref,nunref; self.adapt(nref,nunref); return std::make_tuple(nref,nunref); })
		.def("_replace_RJM_by_param_deriv", &pyoomph::Problem::_replace_RJM_by_param_deriv)
		.def("_set_solved_residual", &pyoomph::Problem::_set_solved_residual, py::arg("name"),py::arg("raise_error")=true)
		.def("set_analytic_hessian_products", [](pyoomph::Problem *self, bool active, bool use_symmetry)
			 { if (active) self->set_analytic_hessian_products();  else self->unset_analytic_hessian_products(); self->set_symmetric_hessian_assembly(use_symmetry); }, py::arg("active"), py::arg("use_symmetry") = false)
		.def("set_FD_step_used_in_get_hessian_vector_products", &pyoomph::Problem::set_FD_step_used_in_get_hessian_vector_products)
		.def("build_global_mesh", &pyoomph::Problem::build_global_mesh)
		.def("rebuild_global_mesh", &pyoomph::Problem::rebuild_global_mesh)
		.def("mesh_pt", (oomph::Mesh * &(pyoomph::Problem::*)()) & pyoomph::Problem::mesh_pt, py::return_value_policy::reference)
		.def("mesh_pt", (oomph::Mesh * &(pyoomph::Problem::*)(unsigned const &)) & pyoomph::Problem::mesh_pt, py::return_value_policy::reference)
		.def("time_pt", (oomph::Time * &(pyoomph::Problem::*)()) & pyoomph::Problem::time_pt, py::return_value_policy::reference)
		.def("time_stepper_pt", (oomph::TimeStepper * &(pyoomph::Problem::*)(const unsigned &)) & pyoomph::Problem::time_stepper_pt, py::return_value_policy::reference, py::arg("i") = 0)
		.def("shift_time_values", &pyoomph::Problem::shift_time_values)
		.def("get_ccompiler", &pyoomph::Problem::get_ccompiler)
		.def("_set_ccompiler", &pyoomph::Problem::set_ccompiler, py::keep_alive<1, 2>())
		.def("ntime_stepper", &pyoomph::Problem::ntime_stepper)
		.def("_assemble_multiassembly", [](pyoomph::Problem *p,std::vector<std::string> what,std::vector<std::string> contributions,std::vector<std::string> params,std::vector<std::vector<double>> hessian_vectors,std::vector<unsigned> & hessian_vector_indices)
			 {
				std::vector<std::vector<double>> data;
				std::vector<std::vector<int>> csrdata;
				std::vector<int> return_indices;
				unsigned ndof;
				p->assemble_multiassembly(what,contributions,params,hessian_vectors,hessian_vector_indices,data,csrdata,ndof,return_indices);
				return std::make_tuple(ndof,data,csrdata,return_indices);
			 })
		.def("distribute", [](pyoomph::Problem *self)
			 {
#ifdef OOMPH_HAS_MPI
				 self->distribute();
#endif
			 })
		.def("is_distributed", &pyoomph::Problem::distributed)
		.def("_redistribute_local_to_global_double_vector", [](pyoomph::Problem *self, const py::array_t<double> &local_v)
			 {
		  py::buffer_info buf = local_v.request();
		  double *in_ptr = (double *)buf.ptr;
		  size_t nloc = buf.shape[0];   
		  auto * loc_distribution=self->linear_solver_pt()->distribution_pt();
		  oomph::DoubleVector tmp(loc_distribution);
        int ndof = loc_distribution->nrow();		  
		  for (unsigned int i=0;i<nloc;i++) tmp[i]=in_ptr[i];
        oomph::LinearAlgebraDistribution global_distribution(loc_distribution->communicator_pt(),ndof,false);
		  tmp.redistribute(&global_distribution);	
        auto res=py::array_t<double>({tmp.nrow()});
        double *res_buff=(double*)res.request().ptr;
        for (unsigned int i=0;i<tmp.nrow();i++) res_buff[i]=tmp[i];
        return res; })
		.def("_redistribute_global_to_local_double_vector", [](pyoomph::Problem *self, const py::array_t<double> &global_v)
			 {
		  py::buffer_info buf = global_v.request();
		  double *in_ptr = (double *)buf.ptr;
		  size_t nglob = buf.shape[0];   
		  auto * loc_distribution=self->linear_solver_pt()->distribution_pt();		  
        oomph::LinearAlgebraDistribution global_distribution(loc_distribution->communicator_pt(),nglob,false);		  
		  oomph::DoubleVector tmp(&global_distribution);
		  for (unsigned int i=0;i<nglob;i++) tmp[i]=in_ptr[i];
		  tmp.redistribute(loc_distribution);	
		  unsigned nloc=loc_distribution->nrow_local();
        auto res=py::array_t<double>({nloc});
        double *res_buff=(double*)res.request().ptr;
        for (unsigned int i=0;i<nloc;i++) res_buff[i]=tmp[i];
        return res; })
		.def("ndof", &pyoomph::Problem::ndof,"Returns the number of equations, i.e. degrees of freedom")
		.def("ensure_dummy_values_to_be_dummy", &pyoomph::Problem::ensure_dummy_values_to_be_dummy)
		.def("generate_and_compile_bulk_element_code", [](pyoomph::Problem *problem, pyoomph::FiniteElementCode *my_element, std::string code_trunk, bool suppress_writing, bool suppress_compilation, pyoomph::Mesh *bulkmesh, bool quiet, const std::vector<std::string> &extra_flags)
			 {
	
	       // Generate Hessian if desired
		   my_element->set_problem(problem);
	       my_element->generate_hessian=problem->are_hessian_products_calculated_analytically();
	       my_element->assemble_hessian_by_symmetry=problem->get_symmetric_hessian_assembly();
			 if (suppress_writing)
			 {
				 std::ostringstream oss; //TODO Null stream instead
				 if (!quiet) std::cout << "Generating equation C code, but do not write to any file"<< std::endl;
				 my_element->write_code(oss);
			 }
			 else
			 {
				 std::ofstream ofs(code_trunk+".c");
				 if (!quiet) std::cout << "Generating equation C code: " << code_trunk << std::endl;
				 my_element->write_code(ofs);
				 std::ofstream hfs(code_trunk+".gar",std::ios::binary);
				 hfs << my_element->archive;
				 hfs.close();
          }

#ifdef OOMPH_HAS_MPI
          MPI_Barrier(problem->communicator_pt()->mpi_comm());
#endif          

			 pyoomph::CCompiler * compiler=problem->get_ccompiler();
			 if (!compiler)
			 {
				  throw_runtime_error("No C compiler set");
			 }
			 compiler->set_code_from_file(code_trunk);

			 if (!suppress_compilation)
			 {
				 if (!quiet) std::cout << "Compiling equation C code" << std::endl;
				 compiler->compile(suppress_compilation,suppress_writing,quiet,extra_flags);
			 }

#ifdef OOMPH_HAS_MPI
          MPI_Barrier(problem->communicator_pt()->mpi_comm());
#endif

			 std::string lib=compiler->get_shared_library(code_trunk);
			 pyoomph::DynamicBulkElementCode * code=problem->load_dynamic_bulk_element_code(lib,my_element);
			 pyoomph::DynamicBulkElementInstance * code_instance=code->factory_instance(bulkmesh);
			 return code_instance; }, py::return_value_policy::reference);
}
