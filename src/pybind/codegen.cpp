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
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include <sstream>
#include <map>

#include "../codegen.hpp"
#include "../ccompiler.hpp"
#include "../expressions.hpp"
#include "../exception.hpp"
#include "../problem.hpp"


namespace pyoomph
{

    // pybind11 trampoline: forwards LaTeXPrinter's two virtual hooks to a Python override, so a
    // Python class can implement custom LaTeX rendering/collection of the symbolic expressions
    // making up the residuals of a FiniteElementCode.
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

    // pybind11 trampoline: forwards FiniteElementCode's virtual hooks to Python overrides. Most
    // of the actual "how is this domain's weak form built up" logic lives in the Python
    // FiniteElementCode subclasses (see pyoomph/generic/codegen.py); the C++ base class provides
    // the bookkeeping (fields, spaces, residuals, ...) that the generated C code plugs into.
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

    // pybind11 trampoline: forwards Equations's two pure-virtual hooks (_define_fields,
    // _define_element) to Python, since every concrete Equations subclass is written in Python.
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

    // pybind11 trampoline: forwards CustomCCompiler's virtual hooks to Python, so a Python class
    // can implement the actual invocation of a C compiler (or an in-memory backend like TinyCC)
    // to build the code generated by src/codegen.cpp.
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

    py::class_<pyoomph::FiniteElementField>(m, "FiniteElementField",
        "A single named field (e.g. a velocity component or pressure) defined on a FiniteElementSpace within a FiniteElementCode."); // TODO: Add stuff

    py::class_<GiNaC::print_FEM_options>(m, "GiNaC_print_FEM_options",
        "Options controlling how a GiNaC expression is printed as (pyoomph-flavored) C code or LaTeX.")
        .def(py::init<>())
        .def("get_code", [](GiNaC::print_FEM_options *self)
             { return self->for_code; }, "Return the FiniteElementCode these print options are associated with.");



    py::class_<pyoomph::Equations, pyoomph::PyEquations>(
        m, "Equations",
        "Base class, to be subclassed in Python, for a set of finite-element equations (fields, weak-form "
        "residual contributions, boundary conditions, ...) contributed to a FiniteElementCode.")
        .def(py::init<>())
        .def("_get_current_codegen", &pyoomph::Equations::_get_current_codegen, py::return_value_policy::reference,
             "Return the FiniteElementCode these equations are currently being defined on/attached to.")
        .def("_define_fields", &pyoomph::Equations::_define_fields,
             "Hook, overridden in Python, that declares the fields these equations require.")
        .def("_define_element", &pyoomph::Equations::_define_element,
             "Hook, overridden in Python, that adds the actual weak-form residual contributions of these equations.")
        .def("_set_current_codegen", &pyoomph::Equations::_set_current_codegen, py::arg("codegen"),
             "Set the FiniteElementCode these equations are currently being defined on/attached to.");

    py::class_<pyoomph::LaTeXPrinter, pyoomph::PyLaTeXPrinter>(
        m, "LaTeXPrinter",
        "Base class, to be subclassed in Python, that collects/renders the LaTeX representation of the "
        "symbolic expressions making up a FiniteElementCode's residuals; installed via FiniteElementCode.set_latex_printer().")
        .def(py::init<>());

    py_decl_PyoomphFiniteElementCode->def(py::init<>())
        .def("_find_all_accessible_spaces", &pyoomph::FiniteElementCode::find_all_accessible_spaces,
             "(Re-)discover all FiniteElementSpaces reachable from this code (including bulk/interface/coupled-ODE domains) and cache them for later lookups.")
        .def("_set_equations", &pyoomph::FiniteElementCode::set_equations, py::arg("equations"),
             "Set the top-level Equations object that defines this code's fields and residuals.")
        .def("get_equations", &pyoomph::FiniteElementCode::get_equations, py::return_value_policy::reference,
             "Return the top-level Equations object of this code.")
        .def("get_scaling", &pyoomph::FiniteElementCode::get_scaling, py::arg("name"), py::arg("testscale") = false,
             "Return the (nondimensionalization) scaling factor registered under ``name``, or the corresponding test-function scale if ``testscale`` is True.")
        .def("_is_ode_element", &pyoomph::FiniteElementCode::_is_ode_element,
             "Whether this code describes a (spaceless) system of ODEs rather than a spatially discretized PDE domain.")
        .def("get_coordinate_system", &pyoomph::FiniteElementCode::get_coordinate_system, py::return_value_policy::reference,
             "Return the CustomCoordinateSystem (e.g. Cartesian, axisymmetric, ...) this code is formulated in.")
        .def("_set_nodal_dimension", &pyoomph::FiniteElementCode::set_nodal_dimension, py::arg("dim"),
             "Set the number of nodal position coordinates (spatial dimension of the embedding space).")
        .def("get_nodal_dimension", &pyoomph::FiniteElementCode::nodal_dimension,
             "Return the number of nodal position coordinates (spatial dimension of the embedding space).")
        .def("_set_lagrangian_dimension", &pyoomph::FiniteElementCode::set_lagrangian_dimension, py::arg("dim"),
             "Set the number of Lagrangian (reference/undeformed) position coordinates.")
        .def("get_lagrangian_dimension", &pyoomph::FiniteElementCode::lagrangian_dimension,
             "Return the number of Lagrangian (reference/undeformed) position coordinates.")
        .def("_set_integration_order", &pyoomph::FiniteElementCode::set_integration_order, py::arg("order"),
             "Set the quadrature (Gauss integration) order used to numerically integrate this code's residuals.")
        .def("_get_integration_order", &pyoomph::FiniteElementCode::get_integration_order,
             "Return the quadrature (Gauss integration) order used to numerically integrate this code's residuals, or a negative value if not explicitly set.")
        .def("_set_problem", &pyoomph::FiniteElementCode::set_problem, py::arg("problem"),
             "Associate this code with the given Problem.")
        .def("expand_additional_field", &pyoomph::FiniteElementCode::expand_additional_field, py::arg("name"), py::arg("dimensional"), py::arg("expression"), py::arg("in_domain"), py::arg("no_jacobian"), py::arg("no_hessian"), py::arg("where"),
             "Hook, overridable in Python, allowing an unknown field ``name`` referenced in ``expression`` to be expanded/substituted by a custom symbolic definition instead of raising an error.")
        .def("_register_external_ode_linkage", &pyoomph::FiniteElementCode::_register_external_ode_linkage, py::arg("myfieldname"), py::arg("odecodegen"), py::arg("odefieldname"),
             "Register that this code's field ``myfieldname`` is linked to (couples with) the field ``odefieldname`` of the external coupled-ODE code ``odecodegen``.")
        .def("_activate_residual", &pyoomph::FiniteElementCode::_activate_residual, py::arg("name"),
             "Select which named residual/Jacobian combination subsequent equation contributions are added to.")
        .def(
            "expand_placeholders", [](pyoomph::FiniteElementCode *c, GiNaC::ex expr, bool raise_error)
            { return c->expand_placeholders(expr, "Python", raise_error).evalm(); },
            py::return_value_policy::reference, py::arg("expression"), py::arg("raise_error") = true,
            "Recursively resolve/expand any placeholder sub-expressions (e.g. fields, scalings, coordinate-system-dependent operators) in ``expression`` into their final symbolic form.")
        .def("expand_additional_testfunction", &pyoomph::FiniteElementCode::expand_additional_testfunction, py::arg("name"), py::arg("expression"), py::arg("in_domain"),
             "Hook, overridable in Python, allowing an unknown test function ``name`` referenced in ``expression`` to be expanded/substituted by a custom symbolic definition instead of raising an error.")
        .def("derive_expression", &pyoomph::FiniteElementCode::derive_expression, py::arg("expression"), py::arg("by"),
             "Return the symbolic (GiNaC) derivative of ``expression`` with respect to ``by``.")
        .def("get_default_timestepping_scheme", &pyoomph::FiniteElementCode::get_default_timestepping_scheme, py::arg("dt_order"),
             "Return the name of the default time-stepping scheme (e.g. \"BDF2\", \"Newmark2\") used for a time derivative of order ``dt_order`` if none is explicitly specified.")
        .def("get_default_spatial_integration_order", &pyoomph::FiniteElementCode::get_default_spatial_integration_order,
             "Return the default quadrature order used if none is explicitly set via _set_integration_order().")
        .def("_set_initial_condition", &pyoomph::FiniteElementCode::set_initial_condition, py::arg("name"), py::arg("expression"), py::arg("degraded_start"), py::arg("ic_name"),
             "Register ``expression`` as the initial condition for field ``name``, optionally under a degraded (lower-order) start scheme ``degraded_start``, tagged with the name ``ic_name``.")
        .def("_set_Dirichlet_bc", &pyoomph::FiniteElementCode::set_Dirichlet_bc, py::arg("name"), py::arg("expression"), py::arg("use_identity"),
             "Register a Dirichlet boundary condition, pinning field ``name`` to ``expression``. If ``use_identity`` is True, the pinning is enforced by an explicit identity residual rather than by removing the degree of freedom.")
        .def("_register_integral_function", &pyoomph::FiniteElementCode::register_integral_function, py::arg("name"), py::arg("expression"),
             "Register ``expression`` as an integral observable named ``name`` (integrated over the domain); can later be queried e.g. via IntegralObservableOutput.")
        .def("_register_tracer_advection", &pyoomph::FiniteElementCode::set_tracer_advection_velocity, py::arg("name"), py::arg("expression"),
             "Register ``expression`` as the advection velocity field ``name`` used to move Lagrangian tracer particles through this domain.")
        .def("_register_local_function", &pyoomph::FiniteElementCode::register_local_expression, py::arg("name"), py::arg("expression"),
             "Register ``expression`` as a local (per-point, non-integrated) named output field ``name``.")
        .def("_register_extremum_function", &pyoomph::FiniteElementCode::register_extremum_expression, py::arg("name"), py::arg("expression"),
             "Register ``expression`` as an extremum observable named ``name`` (its minimum/maximum over the domain is tracked).")
        .def("_get_integral_function_unit_factor", &pyoomph::FiniteElementCode::get_integral_expression_unit_factor, py::arg("name"),
             "Return the physical unit (dimensional scaling factor) of the integral observable ``name`` registered via _register_integral_function().")
        .def("_get_local_expression_unit_factor", &pyoomph::FiniteElementCode::get_local_expression_unit_factor, py::arg("name"),
             "Return the physical unit (dimensional scaling factor) of the local expression ``name`` registered via _register_local_function().")
        .def("_get_extremum_expression_unit_factor", &pyoomph::FiniteElementCode::get_extremum_expression_unit_factor, py::arg("name"),
             "Return the physical unit (dimensional scaling factor) of the extremum observable ``name`` registered via _register_extremum_function().")
        .def("_add_residual", &pyoomph::FiniteElementCode::add_residual, py::arg("contribution"), py::arg("allow_contributions_without_dx"),
             "Add ``contribution`` (a weak-form residual term, typically of the form weak(...,...)) to the currently active residual. ``allow_contributions_without_dx`` permits terms not already wrapped in an integration measure.")
        .def("_add_Z2_flux", &pyoomph::FiniteElementCode::add_Z2_flux, py::arg("flux"), py::arg("for_eigen"),
             "Register ``flux`` as a contribution to the Z2 (flux-recovery) spatial error estimator used for adaptive refinement; ``for_eigen`` selects whether it applies to the base state or an eigenmode.")
        .def("_register_field", &pyoomph::FiniteElementCode::register_field, py::return_value_policy::reference, py::arg("name"), py::arg("spacename"),
             "Declare a new field ``name`` living in the finite element space ``spacename`` (e.g. \"C2\", \"C1\", \"D0\").")
        .def_readwrite("_coordinates_as_dofs", &pyoomph::FiniteElementCode::coordinates_as_dofs,
                        "Whether the nodal position coordinates are themselves treated as degrees of freedom (i.e. a moving-mesh/ALE formulation).")
        .def_readwrite("_coordinate_space", &pyoomph::FiniteElementCode::coordinate_space,
                        "The finite element space (e.g. \"C2\", \"C1\") the nodal position coordinates are interpolated in.")
        .def("_set_bulk_element", &pyoomph::FiniteElementCode::set_bulk_element, py::arg("bulk_code"),
             "Set the bulk-domain FiniteElementCode this (interface/facet) code is attached to.")
        .def("_nullify_bulk_residual", &pyoomph::FiniteElementCode::nullify_bulk_residual, py::arg("field_or_position_index"),
             "Mark the bulk residual contribution of the given continuous field (or the position dof, if negative) as nullified at this interface, so it doesn't get contributions from both the bulk and interface elements redundantly.")
        .def("_get_parent_domain", &pyoomph::FiniteElementCode::get_bulk_element, py::return_value_policy::reference,
             "Return the bulk-domain FiniteElementCode this (interface/facet) code is attached to, or None.")
        .def("_get_opposite_interface", &pyoomph::FiniteElementCode::get_opposite_interface_code, py::return_value_policy::reference,
             "Return the FiniteElementCode of the interface on the opposite side of a shared boundary (e.g. the other domain across a two-domain interface), or None.")
        .def("_set_opposite_interface", &pyoomph::FiniteElementCode::set_opposite_interface_code, py::arg("opposite_code"),
             "Set the FiniteElementCode of the interface on the opposite side of a shared boundary.")
        .def("get_space_of_field", [](pyoomph::FiniteElementCode *code, std::string name)
             {
       pyoomph::FiniteElementField * f=code->get_field_by_name(name);
       if (!f) return std::string("");
       else return f->get_space()->get_name(); }, py::arg("name"),
             "Return the name of the finite element space field ``name`` lives in, or an empty string if the field does not exist.")
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
                return res; }, py::arg("only_spaces") = std::set<std::string>(),
             "Return the names of all fields defined on this code, optionally restricted to fields living in one of the space names in ``only_spaces``.")
        .def("_resolve_based_on_domain_name", &pyoomph::FiniteElementCode::_resolve_based_on_domain_name, py::arg("domainname"), py::return_value_policy::reference,
             "Resolve a (possibly relative, e.g. \"..\"-prefixed) domain name to the corresponding FiniteElementCode, relative to this one.")
        .def("_finalise", &pyoomph::FiniteElementCode::finalise,
             "Finalize this code's definition (fields, residuals, spaces, ...) so it can be handed to the code generator; no further fields/equations may be added afterwards.")
        .def("_get_dx", &pyoomph::FiniteElementCode::get_dx, py::return_value_policy::reference,
             "Return the symbolic integration measure (volume/surface element) of this domain.")
        .def("_get_element_size_symbol", &pyoomph::FiniteElementCode::get_element_size_symbol, py::return_value_policy::reference,
             "Return the symbolic placeholder representing the local element's size/volume.")
        .def("get_integral_dx", [](pyoomph::FiniteElementCode *self, bool use_scaling, bool lagrangian, pyoomph::CustomCoordinateSystem *coordsys)
             { return self->get_integral_dx(use_scaling, lagrangian, coordsys); }, py::return_value_policy::reference, py::arg("use_scaling"), py::arg("lagrangian"), py::arg("coordsys"),
             "Return the symbolic integration measure (volume/surface element), optionally in nondimensional (``use_scaling``) and/or Lagrangian (undeformed, ``lagrangian``) form, evaluated with the given coordinate system.")
        .def("get_element_size", [](pyoomph::FiniteElementCode *self, bool use_scaling, bool lagrangian, bool with_coordsys, pyoomph::CustomCoordinateSystem *coordsys)
             { return self->get_element_size(use_scaling, lagrangian, with_coordsys, coordsys); }, py::return_value_policy::reference, py::arg("use_scaling"), py::arg("lagrangian"), py::arg("with_coordsys"), py::arg("coordsys"),
             "Return the symbolic expression for the (possibly nondimensional/Lagrangian) size of the local element, i.e. the integral of get_integral_dx() over the element.")
        .def("_get_nodal_delta", &pyoomph::FiniteElementCode::get_nodal_delta, py::return_value_policy::reference,
             "Return the symbolic Kronecker-delta-like placeholder used to select individual shape function/nodal contributions.")
        .def("_get_normal_component", &pyoomph::FiniteElementCode::get_normal_component, py::return_value_policy::reference,
             "Return the symbolic expression for the (outer) normal vector's component, as used at interfaces/boundaries.")
        .def("set_ignore_residual_assembly", &pyoomph::FiniteElementCode::set_ignore_residual_assembly, py::arg("ignore"),
             "If ``ignore`` is True, skip assembling the residual contributions of this code entirely (e.g. for a purely auxiliary/ghost code).")
        .def("set_derive_jacobian_by_expansion_mode", &pyoomph::FiniteElementCode::set_derive_jacobian_by_expansion_mode, py::arg("residual_name"), py::arg("expansion_mode"),
             "Select the symbolic expansion mode used when deriving the Jacobian of the residual named ``residual_name`` from the residual.")
        .def("set_derive_hessian_by_expansion_mode", &pyoomph::FiniteElementCode::set_derive_hessian_by_expansion_mode, py::arg("residual_name"), py::arg("expansion_mode"),
             "Select the symbolic expansion mode used when deriving the Hessian of the residual named ``residual_name`` from the residual.")
        .def("set_ignore_dpsi_coord_diffs_in_jacobian", &pyoomph::FiniteElementCode::set_ignore_dpsi_coord_diffs_in_jacobian, py::arg("ignore"),
             "If ``ignore`` is True, omit the (usually small) Jacobian contributions from the derivative of shape function gradients with respect to moving nodal coordinates.")
        .def("_set_temporal_error", &pyoomph::FiniteElementCode::set_temporal_error, py::arg("name"), py::arg("expression"),
             "Register ``expression`` as the temporal error estimator contribution of field ``name``, used for adaptive time stepping.")
        .def("_set_discontinuous_refinement_exponent", &pyoomph::FiniteElementCode::set_discontinuous_refinement_exponent, py::arg("field"), py::arg("exponent"),
             "Set the exponent used to scale the discontinuous (DG-type) field ``field``'s values when the mesh is refined/unrefined.")
        .def("get_time", [](pyoomph::FiniteElementCode &self)
             { return 0.0 + pyoomph::expressions::t; }, py::return_value_policy::reference,
             "Return the symbolic placeholder for the current continuous time.")
        .def("get_dt", [](pyoomph::FiniteElementCode &self)
             { return 0.0 + pyoomph::expressions::dt; }, py::return_value_policy::reference,
             "Return the symbolic placeholder for the current time step size.")
        .def_property_readonly("dimension", &pyoomph::FiniteElementCode::get_dimension,
                                "The spatial dimension of this domain (0 for ODE domains).")
        .def_readwrite("analytical_jacobian", &pyoomph::FiniteElementCode::analytical_jacobian,
                        "Whether the Jacobian is assembled analytically (via generated derivative code) rather than by finite differences.")
        .def_readwrite("analytical_position_jacobian", &pyoomph::FiniteElementCode::analytical_position_jacobian,
                        "Whether the Jacobian entries with respect to nodal position degrees of freedom (moving-mesh problems) are assembled analytically rather than by finite differences.")
        .def("_debug_second_order_Hessian_deriv", &pyoomph::FiniteElementCode::debug_second_order_Hessian_deriv,
             "Debugging helper comparing the analytically derived second-order (Hessian) derivatives against a finite-difference approximation.")
        .def("_do_define_fields", &pyoomph::FiniteElementCode::_do_define_fields,
             "Invoke the Equations' _define_fields() hook to declare this code's fields.")
        .def("_define_fields", &pyoomph::FiniteElementCode::_define_fields,
             "Declare this code's fields (delegates to the attached Equations object).")
        .def("_define_element", &pyoomph::FiniteElementCode::_define_element,
             "Add this code's weak-form residual contributions (delegates to the attached Equations object).")
        .def("_set_reference_point_for_IC_and_DBC", &pyoomph::FiniteElementCode::set_reference_point_for_IC_and_DBC,
             py::arg("x"), py::arg("y"), py::arg("z"), py::arg("t"), py::arg("nx"), py::arg("ny"), py::arg("nz"),
             "Set the reference point (position x,y,z, time t, and normal nx,ny,nz) at which spatially constant initial conditions/Dirichlet values are evaluated.")
        .def("_index_fields", &pyoomph::FiniteElementCode::index_fields,
             "Assign local indices to all fields/spaces of this code, in preparation for code generation.")
        .def("get_domain_name", &pyoomph::FiniteElementCode::get_domain_name,
             "Return the (dotted, possibly hierarchical) name of this code's domain.")
        .def("set_latex_printer", &pyoomph::FiniteElementCode::set_latex_printer, py::arg("printer"),
             "Install a LaTeXPrinter used to render/collect this code's residual expressions as LaTeX.")
        .def_readwrite("debug_jacobian_epsilon", &pyoomph::FiniteElementCode::debug_jacobian_epsilon,
                        "Finite-difference step size used when comparing the analytical Jacobian against a numerical one for debugging.")
        .def_readwrite("with_adaptivity", &pyoomph::FiniteElementCode::with_adaptivity,
                        "Whether this code generates the additional error-estimator code required for spatial mesh adaptivity.")
        .def_readwrite("ccode_expression_mode", &pyoomph::FiniteElementCode::ccode_expression_mode,
                        "Selects how symbolic expressions are emitted in the generated C code (e.g. common-subexpression-optimized vs. plain).")
        .def_readwrite("use_shared_shape_buffer_during_multi_assemble", &pyoomph::FiniteElementCode::use_shared_shape_buffer_during_multi_assemble,
                        "Whether shape function buffers are shared/reused across the different residual/Jacobian/Hessian combinations assembled in a single multi-assembly pass.")
        .def_readwrite("warn_on_large_numerical_factor", &pyoomph::FiniteElementCode::warn_on_large_numerical_factor,
                        "Whether to emit a warning when a very large or very small numerical prefactor is encountered while generating code (often indicating a units/scaling mistake).")
        .def_readwrite("stop_on_jacobian_difference", &pyoomph::FiniteElementCode::stop_on_jacobian_difference,
                        "Whether to raise an error (rather than just warn) when the analytical and finite-difference Jacobians disagree beyond debug_jacobian_epsilon.");

    m.def(
        "_currently_generated_element", []()
        { return pyoomph::__current_code; },
        py::return_value_policy::reference,
        "Return the FiniteElementCode currently being defined/generated (used internally while executing a Python _define_fields()/_define_element() hook), or None.");

    py_decl_PyoomphCCompiler->def(py::init<>())
        .def("compile", [](pyoomph::CCompiler *self, bool suppress1, bool suppress2, bool quiet, const std::vector<std::string> &extra_flags)
             { return self->compile(suppress1, suppress2, quiet, extra_flags); },
             py::arg("suppress_compilation"), py::arg("suppress_code_writing"), py::arg("quiet"), py::arg("extra_flags"),
             "Compile the currently set generated C code into a shared library (or an in-memory module, depending on the backend).")
        .def("get_code_trunk", &pyoomph::CCompiler::get_code_trunk,
             "Return the file path (without extension) the generated C code was/will be written to.")
        .def("compiling_to_memory", &pyoomph::CCompiler::compile_to_memory,
             "Whether this compiler backend compiles directly to memory instead of writing/loading an actual shared library file.")
        .def("sanity_check", &pyoomph::CCompiler::sanity_check,
             "Verify that this compiler backend is usable (e.g. the compiler executable can be found), raising/returning False otherwise.");

    py::class_<pyoomph::CustomCCompiler, pyoomph::PyCCompiler, pyoomph::CCompiler>(
        m, "SharedLibCCompiler",
        "Base class, to be subclassed in Python, for a C compiler backend that builds the code generated by "
        "FiniteElementCode into a shared library, loaded at runtime (see e.g. pyoomph/generic/ccompiler.py "
        "for the system-gcc/clang/MSBuild and TinyCC-based implementations).")
        .def(py::init<>())
        .def("compile", [](pyoomph::CustomCCompiler *self, bool suppress1, bool suppress2, bool quiet, const std::vector<std::string> &extra_flags)
             { return self->compile(suppress1, suppress2, quiet, extra_flags); }, py::arg("suppress_compilation"), py::arg("suppress_code_writing"), py::arg("quiet"), py::arg("extra_flags"),
             "Compile the currently set generated C code into a shared library.")
        .def("sanity_check", &pyoomph::CustomCCompiler::sanity_check,
             "Verify that this compiler backend is usable, raising/returning False otherwise.")
        .def("expand_full_library_name", &pyoomph::CustomCCompiler::expand_full_library_name, py::arg("relative_name"),
             "Turn a relative shared-library base name into the full, platform-specific file path (e.g. adding \"lib\"/\".so\").")
        .def("get_jit_include_dir", &pyoomph::CustomCCompiler::get_jit_include_dir,
             "Return the include directory containing pyoomph's JIT bridge headers (jitbridge.h and friends), required to compile the generated code.")
        .def("get_shared_lib_extension", &pyoomph::CustomCCompiler::get_shared_lib_extension,
             "Return the platform-specific shared library file extension (e.g. \".so\", \".dll\").");

    m.def(
        "set_jit_include_dir", [](std::string dir)
        { return pyoomph::g_jit_include_dir = dir; },
        py::arg("dir"), "Set the include directory containing pyoomph's JIT bridge headers, passed to the C compiler when building generated element code.");
   

    delete py_decl_PyoomphCCompiler;
    delete py_decl_PyoomphFiniteElementCode;
}
