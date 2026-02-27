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
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include <sstream>

#include "../codegen.hpp"
#include "../ccompiler.hpp"
#include "../expressions.hpp"
#include "../exception.hpp"
#include "../problem.hpp"


namespace pyoomph
{

    class PyLaTeXPrinter : public pyoomph::LaTeXPrinter
    {
    public:
        using LaTeXPrinter::LaTeXPrinter;
        void _add_LaTeX_expression(std::map<std::string, std::string> info, std::string expr, FiniteElementCode *code) override
        {
            PYBIND11_OVERLOAD(void, LaTeXPrinter, _add_LaTeX_expression, info, expr, code);
        }
        std::string _get_LaTeX_expression(std::map<std::string, std::string> info, FiniteElementCode *code) override
        {
            PYBIND11_OVERLOAD(std::string, LaTeXPrinter, _get_LaTeX_expression, info, code);
        }
    };

    class PyFiniteElementCode : public pyoomph::FiniteElementCode
    {
    public:
        using FiniteElementCode::FiniteElementCode;

        bool _is_ode_element() const override
        {
            PYBIND11_OVERLOAD(
                bool,              /* Return type */
                FiniteElementCode, /* Parent class */
                _is_ode_element);
        }

        Equations *get_equations() override
        {
            PYBIND11_OVERLOAD(
                Equations *,       /* Return type */
                FiniteElementCode, /* Parent class */
                get_equations);
        }

        GiNaC::ex get_integral_dx(bool use_scaling, bool lagrangian, CustomCoordinateSystem *coordsys) override
        {
            PYBIND11_OVERLOAD(
                GiNaC::ex,         /* Return type */
                FiniteElementCode, /* Parent class */
                get_integral_dx,
                use_scaling,
                lagrangian,
                coordsys);
        }

        GiNaC::ex get_element_size(bool use_scaling, bool lagrangian, bool with_coordsys, CustomCoordinateSystem *coordsys) override
        {
            PYBIND11_OVERLOAD(
                GiNaC::ex,         /* Return type */
                FiniteElementCode, /* Parent class */
                get_element_size,
                use_scaling,
                lagrangian,
                with_coordsys,
                coordsys);
        }

        void _register_external_ode_linkage(std::string my_fieldname, FiniteElementCode *odecode, std::string odefieldname) override
        {
            PYBIND11_OVERLOAD(
                void,              /* Return type */
                FiniteElementCode, /* Parent class */
                _register_external_ode_linkage,
                my_fieldname,
                odecode,
                odefieldname);
        }
        GiNaC::ex get_scaling(std::string name, bool testscale = false) override
        {
            PYBIND11_OVERLOAD(
                GiNaC::ex,         /* Return type */
                FiniteElementCode, /* Parent class */
                get_scaling,
                name,
                testscale);
        }

        std::string get_domain_name() override
        {
            PYBIND11_OVERLOAD(
                std::string,       /* Return type */
                FiniteElementCode, /* Parent class */
                get_domain_name);
        }

        CustomCoordinateSystem *get_coordinate_system() override
        {
            PYBIND11_OVERLOAD(
                CustomCoordinateSystem *, /* Return type */
                FiniteElementCode,        /* Parent class */
                get_coordinate_system);
        }

        GiNaC::ex expand_additional_field(const std::string &name, const bool &dimensional, const GiNaC::ex &expr, pyoomph::FiniteElementCode *in_domain, bool no_jacobian, bool no_hessian, std::string where) override
        {
            PYBIND11_OVERLOAD(
                GiNaC::ex,         /* Return type */
                FiniteElementCode, /* Parent class */
                expand_additional_field, name, dimensional, expr, in_domain, no_jacobian, no_hessian, where);
        }

        GiNaC::ex expand_additional_testfunction(const std::string &name, const GiNaC::ex &expr, pyoomph::FiniteElementCode *in_domain) override
        {
            PYBIND11_OVERLOAD(
                GiNaC::ex,         /* Return type */
                FiniteElementCode, /* Parent class */
                expand_additional_testfunction, name, expr, in_domain);
        }

        std::string get_default_timestepping_scheme(unsigned int dt_order) override
        {
            PYBIND11_OVERLOAD(
                std::string,
                FiniteElementCode,
                get_default_timestepping_scheme, dt_order);
        }

        unsigned get_default_spatial_integration_order() override
        {
            PYBIND11_OVERLOAD(
                unsigned,
                FiniteElementCode,
                get_default_spatial_integration_order);
        }

        pyoomph::FiniteElementCode *_resolve_based_on_domain_name(std::string name) override
        {
            PYBIND11_OVERLOAD(
                pyoomph::FiniteElementCode *,
                FiniteElementCode,
                _resolve_based_on_domain_name, name);
        }
    };

    class PyEquations : public pyoomph::Equations
    {
    public:
        using Equations::Equations;

        void _define_fields() override
        {
            PYBIND11_OVERLOAD_PURE(
                void,           /* Return type */
                Equations,      /* Parent class */
                _define_fields, //,          /* Name of function in C++ (must match Python name) */
                                //            n_times      /* Argument(s) */
            );
        }
        void _define_element() override
        {
            PYBIND11_OVERLOAD_PURE(
                void,            /* Return type */
                Equations,       /* Parent class */
                _define_element, //,          /* Name of function in C++ (must match Python name) */
                                 //            n_times      /* Argument(s) */
            );
        }
    };

    class PyCCompiler : public pyoomph::CustomCCompiler
    {
    public:
        using CustomCCompiler::CustomCCompiler;

        /*
             void set_final_element(FiniteElementCode * fin)  override {
                PYBIND11_OVERLOAD(
                    void,
                    FiniteElementCode,
                    set_final_element, //,
                        fin
                );
            }
        */

        bool compile(bool suppress_compilation, bool suppress_code_writing, bool quiet, const std::vector<std::string> &extra_flags) override
        {
            PYBIND11_OVERLOAD_PURE(
                bool,            /* Return type */
                CustomCCompiler, /* Parent class */
                compile,
                suppress_compilation,
                suppress_code_writing,
                quiet,
                extra_flags);
        }

        bool sanity_check() override
        {
            PYBIND11_OVERLOAD(
                bool,            /* Return type */
                CustomCCompiler, /* Parent class */
                sanity_check);
        }

        std::string expand_full_library_name(std::string relname) override
        {
            PYBIND11_OVERLOAD(
                std::string,     /* Return type */
                CustomCCompiler, /* Parent class */
                expand_full_library_name, relname);
        }

        std::string get_shared_lib_extension() override
        {
            PYBIND11_OVERLOAD(
                std::string,     /* Return type */
                CustomCCompiler, /* Parent class */
                get_shared_lib_extension);
        }
    };


}

static py::class_<pyoomph::CCompiler> *py_decl_PyoomphCCompiler = NULL;
static py::class_<pyoomph::FiniteElementCode, pyoomph::PyFiniteElementCode> *py_decl_PyoomphFiniteElementCode = NULL;

void PyDecl_CodeGen(py::module &m)
{
    py_decl_PyoomphCCompiler = new py::class_<pyoomph::CCompiler>(m, "CCompiler");
    py_decl_PyoomphFiniteElementCode = new py::class_<pyoomph::FiniteElementCode, pyoomph::PyFiniteElementCode>(m, "FiniteElementCode");
}

void PyReg_CodeGen(py::module &m)
{

    py::class_<pyoomph::FiniteElementField>(m, "FiniteElementField"); // TODO: Add stuff

    py::class_<GiNaC::print_FEM_options>(m, "GiNaC_print_FEM_options")
        .def(py::init<>())
        .def("get_code", [](GiNaC::print_FEM_options *self)
             { return self->for_code; });



    py::class_<pyoomph::Equations, pyoomph::PyEquations>(m, "Equations")
        .def(py::init<>())
        .def("_get_current_codegen", &pyoomph::Equations::_get_current_codegen)
        .def("_define_fields", &pyoomph::Equations::_define_fields)
        .def("_define_element", &pyoomph::Equations::_define_element)
        .def("_set_current_codegen", &pyoomph::Equations::_set_current_codegen);

    py::class_<pyoomph::LaTeXPrinter, pyoomph::PyLaTeXPrinter>(m, "LaTeXPrinter")
        .def(py::init<>());

    py_decl_PyoomphFiniteElementCode->def(py::init<>())
        .def("_find_all_accessible_spaces", &pyoomph::FiniteElementCode::find_all_accessible_spaces)
        .def("_set_equations", &pyoomph::FiniteElementCode::set_equations)
        .def("get_equations", &pyoomph::FiniteElementCode::get_equations)
        .def("get_scaling", &pyoomph::FiniteElementCode::get_scaling)
        .def("_is_ode_element", &pyoomph::FiniteElementCode::_is_ode_element)
        .def("get_coordinate_system", &pyoomph::FiniteElementCode::get_coordinate_system, py::return_value_policy::reference)
        .def("_set_nodal_dimension", &pyoomph::FiniteElementCode::set_nodal_dimension)
        .def("get_nodal_dimension", &pyoomph::FiniteElementCode::nodal_dimension)
        .def("_set_lagrangian_dimension", &pyoomph::FiniteElementCode::set_lagrangian_dimension)
        .def("get_lagrangian_dimension", &pyoomph::FiniteElementCode::lagrangian_dimension)
        .def("_set_integration_order", &pyoomph::FiniteElementCode::set_integration_order)
        .def("_get_integration_order", &pyoomph::FiniteElementCode::get_integration_order)
        .def("_set_problem", &pyoomph::FiniteElementCode::set_problem)
        .def("expand_additional_field", &pyoomph::FiniteElementCode::expand_additional_field, py::arg("name"), py::arg("dimensional"), py::arg("expression"), py::arg("in_domain"), py::arg("no_jacobian"), py::arg("no_hessian"), py::arg("where"))
        .def("_register_external_ode_linkage", &pyoomph::FiniteElementCode::_register_external_ode_linkage, py::arg("myfieldname"), py::arg("odecodegen"), py::arg("odefieldname"))
        .def("_activate_residual", &pyoomph::FiniteElementCode::_activate_residual)
        .def(
            "expand_placeholders", [](pyoomph::FiniteElementCode *c, GiNaC::ex expr, bool raise_error)
            { return c->expand_placeholders(expr, "Python", raise_error).evalm(); },
            py::return_value_policy::reference)
        .def("expand_additional_testfunction", &pyoomph::FiniteElementCode::expand_additional_testfunction, py::arg("name"), py::arg("expression"), py::arg("in_domain"))
        .def("derive_expression", &pyoomph::FiniteElementCode::derive_expression)
        .def("get_default_timestepping_scheme", &pyoomph::FiniteElementCode::get_default_timestepping_scheme)
        .def("get_default_spatial_integration_order", &pyoomph::FiniteElementCode::get_default_spatial_integration_order)
        .def("_set_initial_condition", &pyoomph::FiniteElementCode::set_initial_condition)
        .def("_set_Dirichlet_bc", &pyoomph::FiniteElementCode::set_Dirichlet_bc)
        .def("_register_integral_function", &pyoomph::FiniteElementCode::register_integral_function)
        .def("_register_tracer_advection", &pyoomph::FiniteElementCode::set_tracer_advection_velocity)
        .def("_register_local_function", &pyoomph::FiniteElementCode::register_local_expression)
        .def("_register_extremum_function", &pyoomph::FiniteElementCode::register_extremum_expression)
        .def("_get_integral_function_unit_factor", &pyoomph::FiniteElementCode::get_integral_expression_unit_factor)
        .def("_get_local_expression_unit_factor", &pyoomph::FiniteElementCode::get_local_expression_unit_factor)
        .def("_get_extremum_expression_unit_factor", &pyoomph::FiniteElementCode::get_extremum_expression_unit_factor)
        .def("_add_residual", &pyoomph::FiniteElementCode::add_residual)
        .def("_add_Z2_flux", &pyoomph::FiniteElementCode::add_Z2_flux)
        .def("_register_field", &pyoomph::FiniteElementCode::register_field, py::return_value_policy::reference)
        .def_readwrite("_coordinates_as_dofs", &pyoomph::FiniteElementCode::coordinates_as_dofs)
        .def_readwrite("_coordinate_space", &pyoomph::FiniteElementCode::coordinate_space)
        .def("_set_bulk_element", &pyoomph::FiniteElementCode::set_bulk_element)
        .def("_nullify_bulk_residual", &pyoomph::FiniteElementCode::nullify_bulk_residual)
        .def("_get_parent_domain", &pyoomph::FiniteElementCode::get_bulk_element, py::return_value_policy::reference)
        .def("_get_opposite_interface", &pyoomph::FiniteElementCode::get_opposite_interface_code, py::return_value_policy::reference)
        .def("_set_opposite_interface", &pyoomph::FiniteElementCode::set_opposite_interface_code)
        .def("get_space_of_field", [](pyoomph::FiniteElementCode *code, std::string name)
             {
       pyoomph::FiniteElementField * f=code->get_field_by_name(name);
       if (!f) return std::string("");
       else return f->get_space()->get_name(); })
        .def("get_all_fieldnames", [](pyoomph::FiniteElementCode *code, std::set<std::string> only_spaces)
             {
                std::set<std::string> res;
                std::vector<pyoomph::FiniteElementSpace*> spaces=code->get_all_spaces();
                for (auto s : spaces)
                {
                    if (only_spaces.size())
                    {
                        if (!only_spaces.count(s->get_name())) 
                        { 
                        continue;
                        }
                    }
                    std::set<pyoomph::FiniteElementField*> fields_on_s=code->get_fields_on_space(s);
                    for (auto f : fields_on_s)
                    {
                        res.insert(f->get_name());
                    }
                }       
                return res; })
        .def("_resolve_based_on_domain_name", &pyoomph::FiniteElementCode::_resolve_based_on_domain_name, py::arg("domainname"))
        .def("_finalise", &pyoomph::FiniteElementCode::finalise)
        .def("_get_dx", &pyoomph::FiniteElementCode::get_dx, py::return_value_policy::reference)
        .def("_get_element_size_symbol", &pyoomph::FiniteElementCode::get_element_size_symbol, py::return_value_policy::reference)
        .def("get_integral_dx", [](pyoomph::FiniteElementCode *self, bool use_scaling, bool lagrangian, pyoomph::CustomCoordinateSystem *coordsys)
             { return self->get_integral_dx(use_scaling, lagrangian, coordsys); }, py::return_value_policy::reference, py::arg("use_scaling"), py::arg("lagrangian"), py::arg("coordsys"))
        .def("get_element_size", [](pyoomph::FiniteElementCode *self, bool use_scaling, bool lagrangian, bool with_coordsys, pyoomph::CustomCoordinateSystem *coordsys)
             { return self->get_element_size(use_scaling, lagrangian, with_coordsys, coordsys); }, py::return_value_policy::reference, py::arg("use_scaling"), py::arg("lagrangian"), py::arg("with_coordsys"), py::arg("coordsys"))
        .def("_get_nodal_delta", &pyoomph::FiniteElementCode::get_nodal_delta, py::return_value_policy::reference)
        .def("_get_normal_component", &pyoomph::FiniteElementCode::get_normal_component, py::return_value_policy::reference)
        .def("set_ignore_residual_assembly", &pyoomph::FiniteElementCode::set_ignore_residual_assembly)
        .def("set_derive_jacobian_by_expansion_mode", &pyoomph::FiniteElementCode::set_derive_jacobian_by_expansion_mode)
        .def("set_derive_hessian_by_expansion_mode", &pyoomph::FiniteElementCode::set_derive_hessian_by_expansion_mode)        
        .def("set_ignore_dpsi_coord_diffs_in_jacobian", &pyoomph::FiniteElementCode::set_ignore_dpsi_coord_diffs_in_jacobian)
        .def("_set_temporal_error", &pyoomph::FiniteElementCode::set_temporal_error)
        .def("_set_discontinuous_refinement_exponent", &pyoomph::FiniteElementCode::set_discontinuous_refinement_exponent)
        .def("get_time", [](pyoomph::FiniteElementCode &self)
             { return 0.0 + pyoomph::expressions::t; }, py::return_value_policy::reference)
        .def("get_dt", [](pyoomph::FiniteElementCode &self)
             { return 0.0 + pyoomph::expressions::dt; }, py::return_value_policy::reference)
        .def_property_readonly("dimension", &pyoomph::FiniteElementCode::get_dimension)
        .def_readwrite("analytical_jacobian", &pyoomph::FiniteElementCode::analytical_jacobian)
        .def_readwrite("analytical_position_jacobian", &pyoomph::FiniteElementCode::analytical_position_jacobian)
        .def("_debug_second_order_Hessian_deriv", &pyoomph::FiniteElementCode::debug_second_order_Hessian_deriv)
        .def("_do_define_fields", &pyoomph::FiniteElementCode::_do_define_fields)
        .def("_define_fields", &pyoomph::FiniteElementCode::_define_fields)
        .def("_define_element", &pyoomph::FiniteElementCode::_define_element)
        .def("_set_reference_point_for_IC_and_DBC", &pyoomph::FiniteElementCode::set_reference_point_for_IC_and_DBC)
        .def("_index_fields", &pyoomph::FiniteElementCode::index_fields)
        .def("get_domain_name", &pyoomph::FiniteElementCode::get_domain_name)
        .def("set_latex_printer", &pyoomph::FiniteElementCode::set_latex_printer)
        .def_readwrite("bulk_position_space_to_C1", &pyoomph::FiniteElementCode::bulk_position_space_to_C1)
        .def_readwrite("debug_jacobian_epsilon", &pyoomph::FiniteElementCode::debug_jacobian_epsilon)
        .def_readwrite("with_adaptivity", &pyoomph::FiniteElementCode::with_adaptivity)
        .def_readwrite("ccode_expression_mode", &pyoomph::FiniteElementCode::ccode_expression_mode)
        .def_readwrite("use_shared_shape_buffer_during_multi_assemble", &pyoomph::FiniteElementCode::use_shared_shape_buffer_during_multi_assemble)
        .def_readwrite("warn_on_large_numerical_factor", &pyoomph::FiniteElementCode::warn_on_large_numerical_factor)
        .def_readwrite("stop_on_jacobian_difference", &pyoomph::FiniteElementCode::stop_on_jacobian_difference);

    m.def(
        "_currently_generated_element", []()
        { return pyoomph::__current_code; },
        py::return_value_policy::reference);

    py_decl_PyoomphCCompiler->def(py::init<>())
        .def("compile", [](pyoomph::CCompiler *self, bool suppress1, bool suppress2, bool quiet, const std::vector<std::string> &extra_flags)
             { return self->compile(suppress1, suppress2, quiet, extra_flags); })
        .def("get_code_trunk", &pyoomph::CCompiler::get_code_trunk)
        .def("compiling_to_memory", &pyoomph::CCompiler::compile_to_memory)
        .def("sanity_check", &pyoomph::CCompiler::sanity_check);

    py::class_<pyoomph::CustomCCompiler, pyoomph::PyCCompiler, pyoomph::CCompiler>(m, "SharedLibCCompiler")
        .def(py::init<>())
        .def("compile", [](pyoomph::CustomCCompiler *self, bool suppress1, bool suppress2, bool quiet, const std::vector<std::string> &extra_flags)
             { return self->compile(suppress1, suppress2, quiet, extra_flags); }, py::arg("suppress_compilation"), py::arg("suppress_code_writing"), py::arg("quiet"), py::arg("extra_flags"))
        .def("sanity_check", &pyoomph::CustomCCompiler::sanity_check)
        .def("expand_full_library_name", &pyoomph::CustomCCompiler::expand_full_library_name)
        .def("get_jit_include_dir", &pyoomph::CustomCCompiler::get_jit_include_dir)
        .def("get_shared_lib_extension", &pyoomph::CustomCCompiler::get_shared_lib_extension);

    m.def("set_jit_include_dir", [](std::string dir)
          { return pyoomph::g_jit_include_dir = dir; });
    m.def("has_tcc",[]()->bool
    {
     #ifdef PYOOMPH_NO_TCC
      return false;
     #else
      return true;
     #endif
    });

    delete py_decl_PyoomphCCompiler;
    delete py_decl_PyoomphFiniteElementCode;
}
