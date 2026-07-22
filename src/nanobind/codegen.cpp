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
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>

namespace nb = nanobind;

#include <sstream>
#include <map>

#include "../codegen.hpp"
#include "../ccompiler.hpp"
#include "../expressions.hpp"
#include "../exception.hpp"
#include "../problem.hpp"


namespace pyoomph
{

    // nanobind trampoline: forwards LaTeXPrinter's two virtual hooks to a Python override, so a
    // Python class can implement custom LaTeX rendering/collection of the symbolic expressions
    // making up the residuals of a FiniteElementCode.
    class PyLaTeXPrinter : public pyoomph::LaTeXPrinter
    {
    public:
        NB_TRAMPOLINE(LaTeXPrinter, 2);
        void _add_LaTeX_expression(std::map<std::string, std::string> info, std::string expr, FiniteElementCode *code) override
        {
            NB_OVERRIDE(_add_LaTeX_expression, info, expr, code);
        }
        std::string _get_LaTeX_expression(std::map<std::string, std::string> info, FiniteElementCode *code) override
        {
            NB_OVERRIDE(_get_LaTeX_expression, info, code);
        }
    };

    // nanobind trampoline: forwards FiniteElementCode's virtual hooks to Python overrides. Most
    // of the actual "how is this domain's weak form built up" logic lives in the Python
    // FiniteElementCode subclasses (see pyoomph/generic/codegen.py); the C++ base class provides
    // the bookkeeping (fields, spaces, residuals, ...) that the generated C code plugs into.
    class PyFiniteElementCode : public pyoomph::FiniteElementCode
    {
    public:
        NB_TRAMPOLINE(FiniteElementCode, 12);

        bool _is_ode_element() const override
        {
            NB_OVERRIDE(_is_ode_element);
        }

        Equations *get_equations() override
        {
            NB_OVERRIDE(get_equations);
        }

        GiNaC::ex get_integral_dx(bool use_scaling, bool lagrangian, CustomCoordinateSystem *coordsys) override
        {
            NB_OVERRIDE(
                get_integral_dx,
                use_scaling,
                lagrangian,
                coordsys);
        }

        GiNaC::ex get_element_size(bool use_scaling, bool lagrangian, bool with_coordsys, CustomCoordinateSystem *coordsys) override
        {
            NB_OVERRIDE(
                get_element_size,
                use_scaling,
                lagrangian,
                with_coordsys,
                coordsys);
        }

        void _register_external_ode_linkage(std::string my_fieldname, FiniteElementCode *odecode, std::string odefieldname) override
        {
            NB_OVERRIDE(
                _register_external_ode_linkage,
                my_fieldname,
                odecode,
                odefieldname);
        }
        GiNaC::ex get_scaling(std::string name, bool testscale = false) override
        {
            NB_OVERRIDE(
                get_scaling,
                name,
                testscale);
        }

        std::string get_domain_name() override
        {
            NB_OVERRIDE(get_domain_name);
        }

        CustomCoordinateSystem *get_coordinate_system() override
        {
            NB_OVERRIDE(get_coordinate_system);
        }

        GiNaC::ex expand_additional_field(const std::string &name, const bool &dimensional, const GiNaC::ex &expr, pyoomph::FiniteElementCode *in_domain, bool no_jacobian, bool no_hessian, std::string where) override
        {
            NB_OVERRIDE(
                expand_additional_field, name, dimensional, expr, in_domain, no_jacobian, no_hessian, where);
        }

        GiNaC::ex expand_additional_testfunction(const std::string &name, const GiNaC::ex &expr, pyoomph::FiniteElementCode *in_domain) override
        {
            NB_OVERRIDE(
                expand_additional_testfunction, name, expr, in_domain);
        }

        std::string get_default_timestepping_scheme(unsigned int dt_order) override
        {
            NB_OVERRIDE(get_default_timestepping_scheme, dt_order);
        }

        unsigned get_default_spatial_integration_order() override
        {
            NB_OVERRIDE(get_default_spatial_integration_order);
        }

        pyoomph::FiniteElementCode *_resolve_based_on_domain_name(std::string name) override
        {
            NB_OVERRIDE(_resolve_based_on_domain_name, name);
        }
    };

    // nanobind trampoline: forwards Equations's two pure-virtual hooks (_define_fields,
    // _define_element) to Python, since every concrete Equations subclass is written in Python.
    class PyEquations : public pyoomph::Equations
    {
    public:
        NB_TRAMPOLINE(Equations, 2);

        void _define_fields() override
        {
            NB_OVERRIDE_PURE(_define_fields);
        }
        void _define_element() override
        {
            NB_OVERRIDE_PURE(_define_element);
        }
    };

    // nanobind trampoline: forwards CustomCCompiler's virtual hooks to Python, so a Python class
    // can implement the actual invocation of a C compiler (or an in-memory backend like TinyCC)
    // to build the code generated by src/codegen.cpp.
    class PyCCompiler : public pyoomph::CustomCCompiler
    {
    public:
        NB_TRAMPOLINE(CustomCCompiler, 4);

        bool compile(bool suppress_compilation, bool suppress_code_writing, bool quiet, const std::vector<std::string> &extra_flags) override
        {
            NB_OVERRIDE_PURE(
                compile,
                suppress_compilation,
                suppress_code_writing,
                quiet,
                extra_flags);
        }

        bool sanity_check() override
        {
            NB_OVERRIDE(sanity_check);
        }

        std::string expand_full_library_name(std::string relname) override
        {
            NB_OVERRIDE(expand_full_library_name, relname);
        }

        std::string get_shared_lib_extension() override
        {
            NB_OVERRIDE(get_shared_lib_extension);
        }
    };


}

static nb::class_<pyoomph::CCompiler> *py_decl_PyoomphCCompiler = NULL;
static nb::class_<pyoomph::FiniteElementCode, pyoomph::PyFiniteElementCode> *py_decl_PyoomphFiniteElementCode = NULL;

void PyDecl_CodeGen(nb::module_ &m)
{
    py_decl_PyoomphCCompiler = new nb::class_<pyoomph::CCompiler>(m, "CCompiler");
    py_decl_PyoomphFiniteElementCode = new nb::class_<pyoomph::FiniteElementCode, pyoomph::PyFiniteElementCode>(m, "FiniteElementCode");
}

void PyReg_CodeGen(nb::module_ &m)
{

    nb::class_<pyoomph::FiniteElementField>(m, "FiniteElementField",
        "A single named field (e.g. a velocity component or pressure) defined on a FiniteElementSpace within a FiniteElementCode."); // TODO: Add stuff

    nb::class_<GiNaC::print_FEM_options>(m, "GiNaC_print_FEM_options",
        "Options controlling how a GiNaC expression is printed as (pyoomph-flavored) C code or LaTeX.")
        .def(nb::init<>())
        .def("get_code", [](GiNaC::print_FEM_options *self)
             { return self->for_code; }, "Return the FiniteElementCode these print options are associated with.");



    nb::class_<pyoomph::Equations, pyoomph::PyEquations>(
        m, "Equations",
        "Base class, to be subclassed in Python, for a set of finite-element equations (fields, weak-form "
        "residual contributions, boundary conditions, ...) contributed to a FiniteElementCode.")
        .def(nb::init<>())
        .def("_get_current_codegen", &pyoomph::Equations::_get_current_codegen, nb::rv_policy::reference,
             "Return the FiniteElementCode these equations are currently being defined on/attached to.")
        .def("_define_fields", &pyoomph::Equations::_define_fields,
             "Hook, overridden in Python, that declares the fields these equations require.")
        .def("_define_element", &pyoomph::Equations::_define_element,
             "Hook, overridden in Python, that adds the actual weak-form residual contributions of these equations.")
        .def("_set_current_codegen", &pyoomph::Equations::_set_current_codegen, nb::arg("codegen").none(),
             "Set the FiniteElementCode these equations are currently being defined on/attached to.");

    nb::class_<pyoomph::LaTeXPrinter, pyoomph::PyLaTeXPrinter>(
        m, "LaTeXPrinter",
        "Base class, to be subclassed in Python, that collects/renders the LaTeX representation of the "
        "symbolic expressions making up a FiniteElementCode's residuals; installed via FiniteElementCode.set_latex_printer().")
        .def(nb::init<>());

    py_decl_PyoomphFiniteElementCode->def(nb::init<>())
        .def("_find_all_accessible_spaces", &pyoomph::FiniteElementCode::find_all_accessible_spaces,
             "(Re-)discover all FiniteElementSpaces reachable from this code (including bulk/interface/coupled-ODE domains) and cache them for later lookups.")
        .def("_set_equations", &pyoomph::FiniteElementCode::set_equations, nb::arg("equations"),
             "Set the top-level Equations object that defines this code's fields and residuals.")
        .def("get_equations", &pyoomph::FiniteElementCode::get_equations, nb::rv_policy::reference,
             "Return the top-level Equations object of this code.")
        .def("get_scaling", &pyoomph::FiniteElementCode::get_scaling, nb::arg("name"), nb::arg("testscale") = false,
             "Return the (nondimensionalization) scaling factor registered under ``name``, or the corresponding test-function scale if ``testscale`` is True.")
        .def("_is_ode_element", &pyoomph::FiniteElementCode::_is_ode_element,
             "Whether this code describes a (spaceless) system of ODEs rather than a spatially discretized PDE domain.")
        .def("get_coordinate_system", &pyoomph::FiniteElementCode::get_coordinate_system, nb::rv_policy::reference,
             "Return the CustomCoordinateSystem (e.g. Cartesian, axisymmetric, ...) this code is formulated in.")
        .def("_set_nodal_dimension", &pyoomph::FiniteElementCode::set_nodal_dimension, nb::arg("dim"),
             "Set the number of nodal position coordinates (spatial dimension of the embedding space).")
        .def("get_nodal_dimension", &pyoomph::FiniteElementCode::nodal_dimension,
             "Return the number of nodal position coordinates (spatial dimension of the embedding space).")
        .def("_set_lagrangian_dimension", &pyoomph::FiniteElementCode::set_lagrangian_dimension, nb::arg("dim"),
             "Set the number of Lagrangian (reference/undeformed) position coordinates.")
        .def("get_lagrangian_dimension", &pyoomph::FiniteElementCode::lagrangian_dimension,
             "Return the number of Lagrangian (reference/undeformed) position coordinates.")
        .def("_set_integration_order", &pyoomph::FiniteElementCode::set_integration_order, nb::arg("order"),
             "Set the quadrature (Gauss integration) order used to numerically integrate this code's residuals.")
        .def("_get_integration_order", &pyoomph::FiniteElementCode::get_integration_order,
             "Return the quadrature (Gauss integration) order used to numerically integrate this code's residuals, or a negative value if not explicitly set.")
        .def("_set_problem", &pyoomph::FiniteElementCode::set_problem, nb::arg("problem").none(),
             "Associate this code with the given Problem (or None to clear it).")
        .def("_get_problem", &pyoomph::FiniteElementCode::get_problem, nb::rv_policy::reference,
             "Return the Problem this code is associated with (or None if not yet set).")
        .def("expand_additional_field", &pyoomph::FiniteElementCode::expand_additional_field, nb::arg("name"), nb::arg("dimensional"), nb::arg("expression"), nb::arg("in_domain"), nb::arg("no_jacobian"), nb::arg("no_hessian"), nb::arg("where"),
             "Hook, overridable in Python, allowing an unknown field ``name`` referenced in ``expression`` to be expanded/substituted by a custom symbolic definition instead of raising an error.")
        .def("_register_external_ode_linkage", &pyoomph::FiniteElementCode::_register_external_ode_linkage, nb::arg("myfieldname"), nb::arg("odecodegen"), nb::arg("odefieldname"),
             "Register that this code's field ``myfieldname`` is linked to (couples with) the field ``odefieldname`` of the external coupled-ODE code ``odecodegen``.")
        .def("_activate_residual", &pyoomph::FiniteElementCode::_activate_residual, nb::arg("name"),
             "Select which named residual/Jacobian combination subsequent equation contributions are added to.")
        .def(
            "expand_placeholders", [](pyoomph::FiniteElementCode *c, GiNaC::ex expr, bool raise_error)
            { return c->expand_placeholders(expr, "Python", raise_error).evalm(); },
            nb::rv_policy::reference, nb::arg("expression"), nb::arg("raise_error") = true,
            "Recursively resolve/expand any placeholder sub-expressions (e.g. fields, scalings, coordinate-system-dependent operators) in ``expression`` into their final symbolic form.")
        .def("expand_additional_testfunction", &pyoomph::FiniteElementCode::expand_additional_testfunction, nb::arg("name"), nb::arg("expression"), nb::arg("in_domain"),
             "Hook, overridable in Python, allowing an unknown test function ``name`` referenced in ``expression`` to be expanded/substituted by a custom symbolic definition instead of raising an error.")
        .def("derive_expression", &pyoomph::FiniteElementCode::derive_expression, nb::arg("expression"), nb::arg("by"),
             "Return the symbolic (GiNaC) derivative of ``expression`` with respect to ``by``.")
        .def("get_default_timestepping_scheme", &pyoomph::FiniteElementCode::get_default_timestepping_scheme, nb::arg("dt_order"),
             "Return the name of the default time-stepping scheme (e.g. \"BDF2\", \"Newmark2\") used for a time derivative of order ``dt_order`` if none is explicitly specified.")
        .def("get_default_spatial_integration_order", &pyoomph::FiniteElementCode::get_default_spatial_integration_order,
             "Return the default quadrature order used if none is explicitly set via _set_integration_order().")
        .def("_set_initial_condition", &pyoomph::FiniteElementCode::set_initial_condition, nb::arg("name"), nb::arg("expression"), nb::arg("degraded_start"), nb::arg("ic_name"),
             "Register ``expression`` as the initial condition for field ``name``, optionally under a degraded (lower-order) start scheme ``degraded_start``, tagged with the name ``ic_name``.")
        .def("_set_Dirichlet_bc", &pyoomph::FiniteElementCode::set_Dirichlet_bc, nb::arg("name"), nb::arg("expression"), nb::arg("use_identity"),
             "Register a Dirichlet boundary condition, pinning field ``name`` to ``expression``. If ``use_identity`` is True, the pinning is enforced by an explicit identity residual rather than by removing the degree of freedom.")
        .def("_register_integral_function", &pyoomph::FiniteElementCode::register_integral_function, nb::arg("name"), nb::arg("expression"),
             "Register ``expression`` as an integral observable named ``name`` (integrated over the domain); can later be queried e.g. via IntegralObservableOutput.")
        .def("_register_tracer_advection", &pyoomph::FiniteElementCode::set_tracer_advection_velocity, nb::arg("name"), nb::arg("expression"),
             "Register ``expression`` as the advection velocity field ``name`` used to move Lagrangian tracer particles through this domain.")
        .def("_register_local_function", &pyoomph::FiniteElementCode::register_local_expression, nb::arg("name"), nb::arg("expression"),
             "Register ``expression`` as a local (per-point, non-integrated) named output field ``name``.")
        .def("_register_extremum_function", &pyoomph::FiniteElementCode::register_extremum_expression, nb::arg("name"), nb::arg("expression"),
             "Register ``expression`` as an extremum observable named ``name`` (its minimum/maximum over the domain is tracked).")
        .def("_get_integral_function_unit_factor", &pyoomph::FiniteElementCode::get_integral_expression_unit_factor, nb::arg("name"),
             "Return the physical unit (dimensional scaling factor) of the integral observable ``name`` registered via _register_integral_function().")
        .def("_get_local_expression_unit_factor", &pyoomph::FiniteElementCode::get_local_expression_unit_factor, nb::arg("name"),
             "Return the physical unit (dimensional scaling factor) of the local expression ``name`` registered via _register_local_function().")
        .def("_get_extremum_expression_unit_factor", &pyoomph::FiniteElementCode::get_extremum_expression_unit_factor, nb::arg("name"),
             "Return the physical unit (dimensional scaling factor) of the extremum observable ``name`` registered via _register_extremum_function().")
        .def("_add_residual", &pyoomph::FiniteElementCode::add_residual, nb::arg("contribution"), nb::arg("allow_contributions_without_dx"),
             "Add ``contribution`` (a weak-form residual term, typically of the form weak(...,...)) to the currently active residual. ``allow_contributions_without_dx`` permits terms not already wrapped in an integration measure.")
        .def("_add_Z2_flux", &pyoomph::FiniteElementCode::add_Z2_flux, nb::arg("flux"), nb::arg("for_eigen"),
             "Register ``flux`` as a contribution to the Z2 (flux-recovery) spatial error estimator used for adaptive refinement; ``for_eigen`` selects whether it applies to the base state or an eigenmode.")
        .def("_register_field", &pyoomph::FiniteElementCode::register_field, nb::rv_policy::reference, nb::arg("name"), nb::arg("spacename"),
             "Declare a new field ``name`` living in the finite element space ``spacename`` (e.g. \"C2\", \"C1\", \"D0\").")
        .def_rw("_coordinates_as_dofs", &pyoomph::FiniteElementCode::coordinates_as_dofs,
                        "Whether the nodal position coordinates are themselves treated as degrees of freedom (i.e. a moving-mesh/ALE formulation).")
        .def_rw("_coordinate_space", &pyoomph::FiniteElementCode::coordinate_space,
                        "The finite element space (e.g. \"C2\", \"C1\") the nodal position coordinates are interpolated in.")
        .def("_set_bulk_element", &pyoomph::FiniteElementCode::set_bulk_element, nb::arg("bulk_code"),
             "Set the bulk-domain FiniteElementCode this (interface/facet) code is attached to.")
        .def("_nullify_bulk_residual", &pyoomph::FiniteElementCode::nullify_bulk_residual, nb::arg("field_or_position_index"),
             "Mark the bulk residual contribution of the given continuous field (or the position dof, if negative) as nullified at this interface, so it doesn't get contributions from both the bulk and interface elements redundantly.")
        .def("_get_parent_domain", &pyoomph::FiniteElementCode::get_bulk_element, nb::rv_policy::reference,
             "Return the bulk-domain FiniteElementCode this (interface/facet) code is attached to, or None.")
        .def("_get_opposite_interface", &pyoomph::FiniteElementCode::get_opposite_interface_code, nb::rv_policy::reference,
             "Return the FiniteElementCode of the interface on the opposite side of a shared boundary (e.g. the other domain across a two-domain interface), or None.")
        .def("_set_opposite_interface", &pyoomph::FiniteElementCode::set_opposite_interface_code, nb::arg("opposite_code"),
             "Set the FiniteElementCode of the interface on the opposite side of a shared boundary.")
        .def("get_space_of_field", [](pyoomph::FiniteElementCode *code, std::string name)
             {
       pyoomph::FiniteElementField * f=code->get_field_by_name(name);
       if (!f) return std::string("");
       else return f->get_space()->get_name(); }, nb::arg("name"),
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
                    auto fields_on_s=code->get_fields_on_space(s);
                    for (auto f : fields_on_s)
                    {
                        res.insert(f->get_name());
                    }
                }
                return res; }, nb::arg("only_spaces") = std::set<std::string>(),
             "Return the names of all fields defined on this code, optionally restricted to fields living in one of the space names in ``only_spaces``.")
        .def("_resolve_based_on_domain_name", &pyoomph::FiniteElementCode::_resolve_based_on_domain_name, nb::arg("domainname"), nb::rv_policy::reference,
             "Resolve a (possibly relative, e.g. \"..\"-prefixed) domain name to the corresponding FiniteElementCode, relative to this one.")
        .def("_finalise", &pyoomph::FiniteElementCode::finalise,
             "Finalize this code's definition (fields, residuals, spaces, ...) so it can be handed to the code generator; no further fields/equations may be added afterwards.")
        .def("_get_dx", &pyoomph::FiniteElementCode::get_dx, nb::rv_policy::reference,
             "Return the symbolic integration measure (volume/surface element) of this domain.")
        .def("_get_element_size_symbol", &pyoomph::FiniteElementCode::get_element_size_symbol, nb::rv_policy::reference,
             "Return the symbolic placeholder representing the local element's size/volume.")
        .def("get_integral_dx", [](pyoomph::FiniteElementCode *self, bool use_scaling, bool lagrangian, pyoomph::CustomCoordinateSystem *coordsys)
             { return self->get_integral_dx(use_scaling, lagrangian, coordsys); }, nb::rv_policy::reference, nb::arg("use_scaling"), nb::arg("lagrangian"), nb::arg("coordsys"),
             "Return the symbolic integration measure (volume/surface element), optionally in nondimensional (``use_scaling``) and/or Lagrangian (undeformed, ``lagrangian``) form, evaluated with the given coordinate system.")
        .def("get_element_size", [](pyoomph::FiniteElementCode *self, bool use_scaling, bool lagrangian, bool with_coordsys, pyoomph::CustomCoordinateSystem *coordsys)
             { return self->get_element_size(use_scaling, lagrangian, with_coordsys, coordsys); }, nb::rv_policy::reference, nb::arg("use_scaling"), nb::arg("lagrangian"), nb::arg("with_coordsys"), nb::arg("coordsys"),
             "Return the symbolic expression for the (possibly nondimensional/Lagrangian) size of the local element, i.e. the integral of get_integral_dx() over the element.")
        .def("_get_nodal_delta", &pyoomph::FiniteElementCode::get_nodal_delta, nb::rv_policy::reference,
             "Return the symbolic Kronecker-delta-like placeholder used to select individual shape function/nodal contributions.")
        .def("_get_normal_component", &pyoomph::FiniteElementCode::get_normal_component, nb::rv_policy::reference,
             "Return the symbolic expression for the (outer) normal vector's component, as used at interfaces/boundaries.")
        .def("set_ignore_residual_assembly", &pyoomph::FiniteElementCode::set_ignore_residual_assembly, nb::arg("ignore"),
             "If ``ignore`` is True, skip assembling the residual contributions of this code entirely (e.g. for a purely auxiliary/ghost code).")
        .def("set_derive_jacobian_by_expansion_mode", &pyoomph::FiniteElementCode::set_derive_jacobian_by_expansion_mode, nb::arg("residual_name"), nb::arg("expansion_mode"),
             "Select the symbolic expansion mode used when deriving the Jacobian of the residual named ``residual_name`` from the residual.")
        .def("set_derive_hessian_by_expansion_mode", &pyoomph::FiniteElementCode::set_derive_hessian_by_expansion_mode, nb::arg("residual_name"), nb::arg("expansion_mode"),
             "Select the symbolic expansion mode used when deriving the Hessian of the residual named ``residual_name`` from the residual.")
        .def("set_ignore_dpsi_coord_diffs_in_jacobian", &pyoomph::FiniteElementCode::set_ignore_dpsi_coord_diffs_in_jacobian, nb::arg("ignore"),
             "If ``ignore`` is True, omit the (usually small) Jacobian contributions from the derivative of shape function gradients with respect to moving nodal coordinates.")
        .def("_set_temporal_error", &pyoomph::FiniteElementCode::set_temporal_error, nb::arg("name"), nb::arg("expression"),
             "Register ``expression`` as the temporal error estimator contribution of field ``name``, used for adaptive time stepping.")
        .def("_set_discontinuous_refinement_exponent", &pyoomph::FiniteElementCode::set_discontinuous_refinement_exponent, nb::arg("field"), nb::arg("exponent"),
             "Set the exponent used to scale the discontinuous (DG-type) field ``field``'s values when the mesh is refined/unrefined.")
        .def("get_time", [](pyoomph::FiniteElementCode &self)
             { return 0.0 + pyoomph::expressions::t; }, nb::rv_policy::reference,
             "Return the symbolic placeholder for the current continuous time.")
        .def("get_dt", [](pyoomph::FiniteElementCode &self)
             { return 0.0 + pyoomph::expressions::dt; }, nb::rv_policy::reference,
             "Return the symbolic placeholder for the current time step size.")
        .def_prop_ro("dimension", &pyoomph::FiniteElementCode::get_dimension,
                                "The spatial dimension of this domain (0 for ODE domains).")
        .def_rw("analytical_jacobian", &pyoomph::FiniteElementCode::analytical_jacobian,
                        "Whether the Jacobian is assembled analytically (via generated derivative code) rather than by finite differences.")
        .def_rw("analytical_position_jacobian", &pyoomph::FiniteElementCode::analytical_position_jacobian,
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
             nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("t"), nb::arg("nx"), nb::arg("ny"), nb::arg("nz"),
             "Set the reference point (position x,y,z, time t, and normal nx,ny,nz) at which spatially constant initial conditions/Dirichlet values are evaluated.")
        .def("_index_fields", &pyoomph::FiniteElementCode::index_fields,
             "Assign local indices to all fields/spaces of this code, in preparation for code generation.")
        .def("get_domain_name", &pyoomph::FiniteElementCode::get_domain_name,
             "Return the (dotted, possibly hierarchical) name of this code's domain.")
        .def("set_latex_printer", &pyoomph::FiniteElementCode::set_latex_printer, nb::arg("printer").none(),
             "Install a LaTeXPrinter used to render/collect this code's residual expressions as LaTeX.")
        .def_rw("debug_jacobian_epsilon", &pyoomph::FiniteElementCode::debug_jacobian_epsilon,
                        "Finite-difference step size used when comparing the analytical Jacobian against a numerical one for debugging.")
        .def_rw("with_adaptivity", &pyoomph::FiniteElementCode::with_adaptivity,
                        "Whether this code generates the additional error-estimator code required for spatial mesh adaptivity.")
        .def_rw("ccode_expression_mode", &pyoomph::FiniteElementCode::ccode_expression_mode,
                        "Selects how symbolic expressions are emitted in the generated C code (e.g. common-subexpression-optimized vs. plain).")
        .def_rw("use_shared_shape_buffer_during_multi_assemble", &pyoomph::FiniteElementCode::use_shared_shape_buffer_during_multi_assemble,
                        "Whether shape function buffers are shared/reused across the different residual/Jacobian/Hessian combinations assembled in a single multi-assembly pass.")
        .def_rw("warn_on_large_numerical_factor", &pyoomph::FiniteElementCode::warn_on_large_numerical_factor,
                        "Whether to emit a warning when a very large or very small numerical prefactor is encountered while generating code (often indicating a units/scaling mistake).")
        .def_rw("stop_on_jacobian_difference", &pyoomph::FiniteElementCode::stop_on_jacobian_difference,
                        "Whether to raise an error (rather than just warn) when the analytical and finite-difference Jacobians disagree beyond debug_jacobian_epsilon.");

    m.def(
        "_currently_generated_element", []()
        { return pyoomph::__current_code; },
        nb::rv_policy::reference,
        "Return the FiniteElementCode currently being defined/generated (used internally while executing a Python _define_fields()/_define_element() hook), or None.");

    py_decl_PyoomphCCompiler->def(nb::init<>())
        .def("compile", [](pyoomph::CCompiler *self, bool suppress1, bool suppress2, bool quiet, const std::vector<std::string> &extra_flags)
             { return self->compile(suppress1, suppress2, quiet, extra_flags); },
             nb::arg("suppress_compilation"), nb::arg("suppress_code_writing"), nb::arg("quiet"), nb::arg("extra_flags"),
             "Compile the currently set generated C code into a shared library (or an in-memory module, depending on the backend).")
        .def("get_code_trunk", &pyoomph::CCompiler::get_code_trunk,
             "Return the file path (without extension) the generated C code was/will be written to.")
        .def("compiling_to_memory", &pyoomph::CCompiler::compile_to_memory,
             "Whether this compiler backend compiles directly to memory instead of writing/loading an actual shared library file.")
        .def("sanity_check", &pyoomph::CCompiler::sanity_check,
             "Verify that this compiler backend is usable (e.g. the compiler executable can be found), raising/returning False otherwise.");

    nb::class_<pyoomph::CustomCCompiler, pyoomph::PyCCompiler, pyoomph::CCompiler>(
        m, "SharedLibCCompiler",
        "Base class, to be subclassed in Python, for a C compiler backend that builds the code generated by "
        "FiniteElementCode into a shared library, loaded at runtime (see e.g. pyoomph/generic/ccompiler.py "
        "for the system-gcc/clang/MSBuild and TinyCC-based implementations).")
        .def(nb::init<>())
        .def("compile", [](pyoomph::CustomCCompiler *self, bool suppress1, bool suppress2, bool quiet, const std::vector<std::string> &extra_flags)
             { return self->compile(suppress1, suppress2, quiet, extra_flags); }, nb::arg("suppress_compilation"), nb::arg("suppress_code_writing"), nb::arg("quiet"), nb::arg("extra_flags"),
             "Compile the currently set generated C code into a shared library.")
        .def("sanity_check", &pyoomph::CustomCCompiler::sanity_check,
             "Verify that this compiler backend is usable, raising/returning False otherwise.")
        .def("expand_full_library_name", &pyoomph::CustomCCompiler::expand_full_library_name, nb::arg("relative_name"),
             "Turn a relative shared-library base name into the full, platform-specific file path (e.g. adding \"lib\"/\".so\").")
        .def("get_jit_include_dir", &pyoomph::CustomCCompiler::get_jit_include_dir,
             "Return the include directory containing pyoomph's JIT bridge headers (jitbridge.h and friends), required to compile the generated code.")
        .def("get_shared_lib_extension", &pyoomph::CustomCCompiler::get_shared_lib_extension,
             "Return the platform-specific shared library file extension (e.g. \".so\", \".dll\").");

    m.def(
        "set_jit_include_dir", [](std::string dir)
        { return pyoomph::g_jit_include_dir = dir; },
        nb::arg("dir"), "Set the include directory containing pyoomph's JIT bridge headers, passed to the C compiler when building generated element code.");
   

    delete py_decl_PyoomphCCompiler;
    delete py_decl_PyoomphFiniteElementCode;
}
