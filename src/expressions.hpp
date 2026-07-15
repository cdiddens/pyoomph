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


#pragma once

#include "ginac.hpp"
#include "pyginacstruct.hpp"
#include <vector>
#include <map>
#include "exception.hpp"
#include <complex>

namespace pyoomph
{

  // Default spatial dimension used for fixed-size vector/tensor storage (e.g. buffers sized independent of the actual problem dimension)
  const unsigned the_vector_dim = 3;

  // Cache mapping a name (e.g. a field or parameter name) to the unique GiNaC symbol representing it, so that repeated
  // lookups of the same name always return the identical GiNaC handle (required since GiNaC compares/sorts symbols by identity)
  extern std::map<std::string, GiNaC::ex> __field_name_cache;
  GiNaC::ex _get_field_name_cache(const std::string &id);

  // Base class for a user-defined scalar callback function (exposed to Python as CustomMathExpression) that can be embedded
  // as a leaf in a GiNaC expression tree, evaluated numerically at runtime and differentiated symbolically/numerically.
  class CustomMathExpressionBase
  {
  protected:
    static unsigned unique_counter;
    unsigned unique_id;                    // Unique id. These are necessary for sorting the expressions in the order they are created by Python => Required for Parallel processes and missing GiNaC order
    int jit_index;                         // temorarary index of the current jit compilation
    CustomMathExpressionBase *diff_parent; // Parent function of the derivative, i.e this function is a derivative of diff_parent
    int diff_index;                        // along this index
  public:
    static std::map<CustomMathExpressionBase *, int> code_map; // Maps each instance to its index in the generated JIT code (assigned during code generation)
    CustomMathExpressionBase() : unique_id(unique_counter++), jit_index(-1), diff_parent(NULL), diff_index(-1) {}
    virtual ~CustomMathExpressionBase() { std::cout << "FREEING Expr " << unique_id << std::endl; }
    virtual CustomMathExpressionBase *get_diff_parent() const { return diff_parent; }
    virtual int get_diff_index() const { return diff_index; }
    // Marks this instance as being the derivative of "parent" with respect to its "index"-th argument (used to relate auto-generated derivative callbacks back to the original function)
    void set_as_derivative(CustomMathExpressionBase *parent, int index)
    {
      diff_parent = parent;
      diff_index = index;
    }
    virtual double _call(double *, unsigned int ) { return 0.0; } // Numerically evaluate the callback (overridden by the Python binding to dispatch to the Python eval() method)
    int get_jit_index() { return jit_index; }
    unsigned get_unique_id() { return unique_id; }
    virtual GiNaC::ex outer_derivative(const GiNaC::ex , int ) { return 0; }         // Symbolic derivative of the function call w.r.t. its "index"-th argument, given the (already substituted) argument list
    virtual GiNaC::ex real_part(GiNaC::ex , std::vector<GiNaC::ex> ) { return 0; }    // Symbolic real part of an invocation of this function (for complex-valued usage)
    virtual GiNaC::ex imag_part(GiNaC::ex , std::vector<GiNaC::ex> ) { return 0; }    // Symbolic imaginary part of an invocation of this function
    virtual std::string get_id_name() { return "unknown cb"; }         // Human-readable identifier, used e.g. in generated code and error messages
    virtual GiNaC::ex get_argument_unit(unsigned int ) { return 1; }  // Physical unit expected for the i-th argument (arguments are nondimensionalized by this before the callback is invoked)
    virtual GiNaC::ex get_result_unit() { return 1; }                  // Physical unit of the callback's return value (the numeric result is rescaled by this)
  };

  // Thin pointer wrapper so a CustomMathExpressionBase* can be embedded as a leaf inside a GiNaC expression tree via PYGINACSTRUCT
  class CustomMathExpressionWrapper
  {
  public:
    CustomMathExpressionBase *cme;
    CustomMathExpressionWrapper(CustomMathExpressionBase *c) : cme(c) {}
    CustomMathExpressionWrapper(const CustomMathExpressionWrapper &c) : cme(c.cme) {}
  };

  // Comparison operators required by the pyginacstruct comparison policy (compares by wrapped pointer, i.e. by identity)
  bool operator==(const CustomMathExpressionWrapper &lhs, const CustomMathExpressionWrapper &rhs);
  bool operator<(const CustomMathExpressionWrapper &lhs, const CustomMathExpressionWrapper &rhs);

  // Base class for a user-defined callback that can return multiple values at once (and optionally their Jacobian w.r.t. the
  // arguments), e.g. for numerically defined constitutive laws. Exposed to Python as CustomMultiReturnExpression.
  // To return multiple values
  class CustomMultiReturnExpressionBase
  {
  public:
    static unsigned unique_counter;
    unsigned unique_id;
    double debug_c_code_epsilon; // If >=0, generated C code calls are cross-checked against the Python eval() result and a warning is printed if they differ by more than this tolerance
    static std::map<CustomMultiReturnExpressionBase *, int> code_map; // Maps each instance to its index in the generated JIT code
    CustomMultiReturnExpressionBase() : unique_id(unique_counter++), debug_c_code_epsilon(-1.0) {}
    virtual std::string get_id_name() { return "unknown multi-ret cb"; }
    virtual std::string _get_c_code() { return ""; } // Optionally return literal C code implementing this function directly (inlined at code generation instead of calling back into Python); empty means no C code present
    virtual void _call(int , double *, unsigned int , double *, unsigned int , double *) { throw_runtime_error("Should not end up here"); } // Numerically evaluate; if flag is set, also fill the nres x nargs derivative matrix
    virtual std::pair<bool, GiNaC::ex> _get_symbolic_derivative(const std::vector<GiNaC::ex> &, const int &, const int &) { return std::make_pair(false, 0); } // Optionally provide an analytic symbolic derivative of result i_res w.r.t. argument j_arg (first=false means "not available", fall back to numerical differentiation)
  };

  // Thin pointer wrapper so a CustomMultiReturnExpressionBase* can be embedded as a GiNaC leaf
  class CustomMultiReturnExpressionWrapper
  {
  public:
    CustomMultiReturnExpressionBase *cme;
    CustomMultiReturnExpressionWrapper(CustomMultiReturnExpressionBase *c) : cme(c) {}
    CustomMultiReturnExpressionWrapper(const CustomMultiReturnExpressionWrapper &c) : cme(c.cme) {}
  };
  bool operator==(const CustomMultiReturnExpressionWrapper &lhs, const CustomMultiReturnExpressionWrapper &rhs);
  bool operator<(const CustomMultiReturnExpressionWrapper &lhs, const CustomMultiReturnExpressionWrapper &rhs);

  // Since a multi-return callback yields several values at once but a single GiNaC expression can only represent one scalar,
  // an invocation is represented by one CustomMultiReturnExpressionResultSymbol per requested output component: it records
  // which function was called, with what arguments, and which result component ("index") this particular leaf stands for.
  // Stores its function and its result
  class CustomMultiReturnExpressionResultSymbol
  {
  public:
    CustomMultiReturnExpressionBase *func;
    GiNaC::lst arglist;
    unsigned index;
  };
  bool operator==(const CustomMultiReturnExpressionResultSymbol &lhs, const CustomMultiReturnExpressionResultSymbol &rhs);
  bool operator<(const CustomMultiReturnExpressionResultSymbol &lhs, const CustomMultiReturnExpressionResultSymbol &rhs);

  // Wraps a C++ closure that lazily produces a GiNaC expression on demand (e.g. resolving a Python callable into an
  // expression only once it is actually needed during expansion/code generation, instead of eagerly at construction time)
  class DelayedPythonCallbackExpansion
  {
  public:
    std::function<GiNaC::ex()> f;
    DelayedPythonCallbackExpansion(std::function<GiNaC::ex()> func) : f(func) {}
  };

  // Thin pointer wrapper so a DelayedPythonCallbackExpansion* can be embedded as a GiNaC leaf
  class DelayedPythonCallbackExpansionWrapper
  {
  public:
    DelayedPythonCallbackExpansion *cme;
    DelayedPythonCallbackExpansionWrapper(DelayedPythonCallbackExpansion *c) : cme(c) {}
    DelayedPythonCallbackExpansionWrapper(const DelayedPythonCallbackExpansionWrapper &c) : cme(c.cme) {}
    virtual ~DelayedPythonCallbackExpansionWrapper() {}
  };

  bool operator==(const DelayedPythonCallbackExpansionWrapper &lhs, const DelayedPythonCallbackExpansionWrapper &rhs);
  bool operator<(const DelayedPythonCallbackExpansionWrapper &lhs, const DelayedPythonCallbackExpansionWrapper &rhs);

  class GlobalParameterDescriptor;

  // Thin pointer wrapper so a GlobalParameterDescriptor* (a Python-exposed mutable scalar parameter) can be embedded as a
  // GiNaC leaf inside expressions; the wrapped descriptor is later substituted by its current numerical value where needed
  class GlobalParameterWrapper
  {
  public:
    GlobalParameterDescriptor *cme;
    GlobalParameterWrapper(GlobalParameterDescriptor *c) : cme(c) {}
    GlobalParameterWrapper(const GlobalParameterWrapper &c) : cme(c.cme) {}
    virtual ~GlobalParameterWrapper() {}
  };

  bool operator==(const GlobalParameterWrapper &lhs, const GlobalParameterWrapper &rhs);
  bool operator<(const GlobalParameterWrapper &lhs, const GlobalParameterWrapper &rhs);

  //

  class FiniteElementCode;
  // Base class (overridable from Python via CustomCoordinateSystem) implementing the differential operators (grad, div, ...)
  // for a particular coordinate system (e.g. axisymmetric, spherical). An instance is carried along symbolically as an
  // argument of the grad/div/... GiNaC placeholder functions and invoked during code generation to expand them.
  class CustomCoordinateSystem
  {
  public:
    CustomCoordinateSystem() {}
    virtual ~CustomCoordinateSystem() {}
    virtual int vector_gradient_dimension(unsigned int basedim, bool ) { return basedim; } // Number of components of the gradient of a vector field in this coordinate system (may differ from basedim, e.g. when azimuthal derivatives add an extra component)
    // f: expression to differentiate; dim: nodal/physical dimension (or -1 if not fixed); edim: element (local) dimension; flags: bitmask of expansion options (e.g. whether to attach dimensional prefactors)
    virtual GiNaC::ex grad(const GiNaC::ex &, int , int , int ) { throw_runtime_error("grad not implemented for this coordinate system"); }
    virtual GiNaC::ex directional_derivative(const GiNaC::ex &, const GiNaC::ex &, int , int , int ) { throw_runtime_error("directional derivative not implemented for this coordinate system"); }

    // Generic hook for coordinate-system-specific weak-form contributions that cannot be expressed via grad/div alone
    // (e.g. extra curvature terms); funcname identifies which high-level operator requested the contribution
    virtual GiNaC::ex general_weak_differential_contribution(std::string , std::vector<GiNaC::ex> , GiNaC::ex , int , int , int )
    {
      throw_runtime_error("general_weak_differential_contribution not implemented for this coordinate system");
    }
    virtual GiNaC::ex div(const GiNaC::ex &, int , int , int ) { throw_runtime_error("div not implemented for this coordinate system"); }
    virtual GiNaC::ex geometric_jacobian() { return 1.0; }          // Extra metric-determinant factor (e.g. r in cylindrical coordinates) multiplying the integration measure
    virtual GiNaC::ex jacobian_for_element_size() { return 1.0; }   // Jacobian factor used specifically when computing an element's physical size (may differ from geometric_jacobian)
    virtual std::string get_id_name() { return "<unknown coordinate system>"; }
    // Hook allowing the coordinate system to modify how a field/test function is expanded for a given expansion mode (e.g. Fourier/azimuthal mode decomposition); expr is the yet-unexpanded expression, where gives the calling context
    virtual GiNaC::ex get_mode_expansion_of_var_or_test(pyoomph::FiniteElementCode *, std::string , bool , bool , GiNaC::ex expr, std::string , int ) { return expr; }
  };

  // Thin pointer wrapper so a CustomCoordinateSystem* can be embedded as a GiNaC leaf and passed as an argument to grad/div/...
  class CustomCoordinateSystemWrapper
  {
  public:
    CustomCoordinateSystem *cme;
    CustomCoordinateSystemWrapper(CustomCoordinateSystem *c) : cme(c) {}
    CustomCoordinateSystemWrapper(const CustomCoordinateSystemWrapper &c) : cme(c.cme) {}
  };

  bool operator==(const CustomCoordinateSystemWrapper &lhs, const CustomCoordinateSystemWrapper &rhs);
  bool operator<(const CustomCoordinateSystemWrapper &lhs, const CustomCoordinateSystemWrapper &rhs);

  class FiniteElementCode;
  // Carries the context (the FiniteElementCode being generated, plus a list of free-form qualifier tags, e.g. domain/space
  // selectors) needed to resolve a symbolic placeholder (field(), scale(), testfunction(), ...) into a concrete expression.
  // Embedded as the second argument of those placeholder functions.
  class PlaceHolderResolveInfo
  {
  public:
    FiniteElementCode *code;
    std::vector<std::string> tags;
    PlaceHolderResolveInfo() : code(NULL), tags() {}
    PlaceHolderResolveInfo(FiniteElementCode *_code, const std::vector<std::string> &_tags) : code(_code), tags(_tags) {}
  };

  bool operator==(const PlaceHolderResolveInfo &lhs, const PlaceHolderResolveInfo &rhs);
  bool operator<(const PlaceHolderResolveInfo &lhs, const PlaceHolderResolveInfo &rhs);

  // Symbolic reference to a specific history/timestep value of the time variable, used e.g. by eval_in_past() and by the
  // BDF/Newmark time discretization machinery. index==0 refers to the current time, index>0 to previous timesteps.
  class TimeSymbol
  {
  public:
    int index;
    // TODO: Add a previous order here as well?
    TimeSymbol() : index(0) {}
    TimeSymbol(int history_index) : index(history_index) {}
  };

  bool operator==(const TimeSymbol &lhs, const TimeSymbol &rhs);
  bool operator<(const TimeSymbol &lhs, const TimeSymbol &rhs);

  // An exponential mode exp(arg) which derives as arg'*exp(arg), but in the code it will be exp->1
  // (used e.g. for azimuthal/Floquet mode ansatzes where the exponential prefactor is handled abstractly and never
  // literally evaluated in the generated code; "dual" marks the conjugate/adjoint mode, e.g. for bifurcation tracking)
  class FakeExponentialMode
  {
  public:
    GiNaC::ex arg;
    bool dual;
    FakeExponentialMode(GiNaC::ex _arg, bool _dual = false) : arg(_arg), dual(_dual) {}
  };

  bool operator==(const FakeExponentialMode &lhs, const FakeExponentialMode &rhs);
  bool operator<(const FakeExponentialMode &lhs, const FakeExponentialMode &rhs);

}

namespace GiNaC
{

  // Each PYGINACSTRUCT(WrapperType, GiNaCName) below declares GiNaCName as a GiNaC atom type (a specialization of
  // pyginacstruct<WrapperType>, see pyginacstruct.hpp) so that the corresponding C++ wrapper object can appear as a leaf
  // node inside a GiNaC expression tree. Each one gets a specialized print() (and, where relevant, derivative()/info())
  // implementation defined in expressions.cpp; comparisons/hashing use the operator==/operator< defined above.

  // Leaf representing an invocation target of a user-defined scalar callback (CustomMathExpressionBase)
  PYGINACSTRUCT(pyoomph::CustomMathExpressionWrapper, GiNaCCustomMathExpressionWrapper);
  template <>
  void GiNaC::GiNaCCustomMathExpressionWrapper::print(const print_context &c, unsigned level) const;

  // Leaf representing an invocation target of a user-defined multi-return callback (CustomMultiReturnExpressionBase)
  PYGINACSTRUCT(pyoomph::CustomMultiReturnExpressionWrapper, GiNaCCustomMultiReturnExpressionWrapper);
  template <>
  void GiNaCCustomMultiReturnExpressionWrapper::print(const print_context &c, unsigned level) const;

  // Leaf representing one indexed result component of a multi-return callback invocation
  PYGINACSTRUCT(pyoomph::CustomMultiReturnExpressionResultSymbol, GiNaCCustomMultiReturnExpressionResultSymbol);
  template <>
  void GiNaCCustomMultiReturnExpressionResultSymbol::print(const print_context &c, unsigned level) const;

  // Leaf wrapping a CustomCoordinateSystem object, passed as an argument to grad/div/... so the coordinate system can be
  // recovered again during code generation
  PYGINACSTRUCT(pyoomph::CustomCoordinateSystemWrapper, GiNaCCustomCoordinateSystemWrapper);
  template <>
  void GiNaCCustomCoordinateSystemWrapper::print(const print_context &c, unsigned level) const;

  // Leaf wrapping a global parameter; info() is specialized so that GiNaC treats it as a real-valued (info_flags::real) atom
  PYGINACSTRUCT(pyoomph::GlobalParameterWrapper, GiNaCGlobalParameterWrapper);
  template <>
  void GiNaCGlobalParameterWrapper::print(const print_context &c, unsigned level) const;
  template <>
  bool GiNaCGlobalParameterWrapper::info(unsigned inf) const;


  // Leaf carrying the resolve context (FiniteElementCode + tags) for placeholder expansion functions (field(), scale(), ...)
  PYGINACSTRUCT(pyoomph::PlaceHolderResolveInfo, GiNaCPlaceHolderResolveInfo);
  template <>
  void GiNaCPlaceHolderResolveInfo::print(const print_context &c, unsigned level) const;

  // Leaf wrapping a lazily-evaluated Python-expression-producing callback
  PYGINACSTRUCT(pyoomph::DelayedPythonCallbackExpansionWrapper, GiNaCDelayedPythonCallbackExpansion);
  template <>
  void GiNaCDelayedPythonCallbackExpansion::print(const print_context &c, unsigned level) const;

  // Leaf representing a symbolic time-history reference (see TimeSymbol); has a custom derivative() since differentiating
  // w.r.t. time must be handled specially rather than falling back to GiNaC's default (derivative w.r.t. a generic symbol)
  PYGINACSTRUCT(pyoomph::TimeSymbol, GiNaCTimeSymbol);
  template <>
  void GiNaCTimeSymbol::print(const print_context &c, unsigned level) const;
  template <>
  GiNaC::ex GiNaCTimeSymbol::derivative(const GiNaC::symbol &s) const;

  // An exponential mode exp(arg) which derives as arg'*exp(arg), but in the code it will be exp->1
  PYGINACSTRUCT(pyoomph::FakeExponentialMode, GiNaCFakeExponentialMode);
  template <>
  void GiNaCFakeExponentialMode::print(const print_context &c, unsigned level) const;
  template <>
  GiNaC::ex GiNaCFakeExponentialMode::derivative(const GiNaC::symbol &s) const;

  // Symbol type used throughout pyoomph for quantities that are always real-valued; currently pinned to GiNaC::realsymbol
  // (the commented-out alternative would allow switching to plain, possibly-complex symbols)
  //typedef symbol potential_real_symbol;
  typedef realsymbol potential_real_symbol;




}

namespace pyoomph
{

  // Sentinel/default coordinate system (plain Cartesian, i.e. all operators reduce to their flat-space form) used whenever
  // no explicit CustomCoordinateSystem has been supplied by Python
  extern CustomCoordinateSystem __no_coordinate_system;
  extern GiNaC::GiNaCCustomCoordinateSystemWrapper __no_coordinate_system_wrapper;

  // Registry of user-defined physical base units (name -> positive symbol), populated on demand from Python (GiNaC_unit())
  extern std::map<std::string, GiNaC::possymbol> base_units;

  namespace expressions
  {

    extern GiNaC::symbol nnode; // Symbol representing "number of nodes" as used in generated code
    // Eulerian (current, deformed) global coordinates
    extern GiNaC::potential_real_symbol x;
    extern GiNaC::potential_real_symbol y;
    extern GiNaC::potential_real_symbol z;
    // Lagrangian (undeformed/reference) global coordinates
    extern GiNaC::potential_real_symbol X;
    extern GiNaC::potential_real_symbol Y; // Lagrangian
    extern GiNaC::potential_real_symbol Z;
    // Element-local (reference element) coordinates, i.e. the "s" coordinates ranging e.g. over [-1,1] or the unit triangle
    extern GiNaC::potential_real_symbol local_coordinate_1,local_coordinate_2,local_coordinate_3;
    // Macro-element intrinsic coordinates (used e.g. for mesh generation on curved macro elements)
    extern GiNaC::potential_real_symbol zeta_coordinate_1, zeta_coordinate_2, zeta_coordinate_3;
    // Components of the outer unit normal vector
    extern GiNaC::potential_real_symbol nx;
    extern GiNaC::potential_real_symbol ny;
    extern GiNaC::potential_real_symbol nz;

    // Time and per-scheme (sub-)timestep symbols used by the generated code's time discretization
    extern GiNaC::potential_real_symbol t, _dt_BDF1, _dt_BDF2, _dt_Newmark2;
    extern GiNaC::potential_real_symbol __partial_t_mass_matrix; // This symbol is used to identify partial_t terms to put in the mass matrix
    extern GiNaC::potential_real_symbol dt;
    extern GiNaC::potential_real_symbol timefrac_tracer; // Fractional-in-time weight used when interpolating quantities at tracer particle positions between timesteps
    extern GiNaC::idx l_shape; // Symbolic loop index over shape functions/nodes (corresponds to the "l_shape" loop variable in generated C code)
    extern GiNaC::idx l_test;  // Symbolic loop index over test functions (corresponds to the "l_test" loop variable in generated C code)
    extern GiNaC::potential_real_symbol *proj_on_test_function; // If set, restricts weak-form assembly to a single, given test function (projection) instead of looping over all of them
    extern int el_dim; // Currently active element (local) dimension during code generation

    GiNaC::ex diff(const GiNaC::ex &what, const GiNaC::ex &wrto); // Symbolic differentiation that additionally understands pyoomph's placeholder functions (fields, test functions, ...), unlike plain GiNaC::diff
    bool collect_base_units(GiNaC::ex arg, GiNaC::ex &factor, GiNaC::ex &units, GiNaC::ex &rest); // Splits arg into a numeric factor, a product of base units, and a dimensionless remainder; returns false if arg is not unit-consistent

    // Substitutes field()/nondimfield() placeholders and global parameters occurring in arg by concrete expressions (used e.g. to numerically evaluate an expression by "calling" it with concrete field values)
    GiNaC::ex subs_fields(const GiNaC::ex &arg, const std::map<std::string, GiNaC::ex> &fields, const std::map<std::string, GiNaC::ex> &nondimfields, const std::map<std::string, GiNaC::ex> &globalparams);

    GiNaC::ex replace_global_params_by_current_values(const GiNaC::ex &in); // Replaces every GlobalParameterWrapper leaf in "in" by its current numerical value

    double eval_to_double(const GiNaC::ex &inp);                // Forces full numeric evaluation of an expression to a real double (throws if not possible)
    std::complex<double> eval_to_complex(const GiNaC::ex &inp); // Forces full numeric evaluation of an expression to a complex double

    // The following DECLARE_FUNCTION_NP macros (from GiNaC) each declare a symbolic GiNaC function of N arguments that acts
    // as a placeholder/operator node in the expression tree. They are not evaluated immediately; their _eval/print/derivative
    // behavior is registered via REGISTER_FUNCTION in expressions.cpp, where they are expanded/lowered during code generation.

    DECLARE_FUNCTION_5P(grad) // 1: what to grad, 2: nodal dimension or -1, 3: element dimension or -1, 4: Coordinate System object, 5: withdim(0,1)
    DECLARE_FUNCTION_6P(directional_derivative) // Like grad, but projected onto a given direction: 1: what, 2: direction, 3: nodal dim, 4: element dim, 5: coordinate system, 6: withdim
    DECLARE_FUNCTION_7P(general_weak_differential_contribution) // Generic coordinate-system-dependent weak-form differential term; forwards to CustomCoordinateSystem::general_weak_differential_contribution
    DECLARE_FUNCTION_2P(dot)        // Dot (inner) product of two vectors
    DECLARE_FUNCTION_2P(double_dot) // Double-contraction (Frobenius inner product) of two matrices/tensors
    DECLARE_FUNCTION_2P(contract)   // Generic index contraction, used e.g. to implement the "@" (matmul) operator
    DECLARE_FUNCTION_4P(weak)       // Marks a weak-form contribution (lhs paired with a test function), later expanded into residual contributions
    DECLARE_FUNCTION_5P(div)        // Divergence: 1: what, 2: nodal dim, 3: element dim, 4: coordinate system, 5: withdim
    DECLARE_FUNCTION_1P(transpose)  // Matrix transpose
    DECLARE_FUNCTION_1P(trace)      // Matrix trace
    DECLARE_FUNCTION_2P(determinant)    // 1: matrix, 2: dimension (in case of a non-square/degenerate matrix)
    DECLARE_FUNCTION_3P(inverse_matrix) // 1: matrix, 2: dimension, 3: flags

    DECLARE_FUNCTION_4P(minimize_functional_derivative) // Functional derivative used for minimization-type weak formulations (e.g. gradient flow / energy minimization problems)

    DECLARE_FUNCTION_4P(unitvect) // Unit basis vector in a given direction/coordinate system

    DECLARE_FUNCTION_1P(subexpression) // Wraps an expression so it is factored out as a named local subexpression in the generated code instead of being inlined everywhere it is used

    DECLARE_FUNCTION_1P(get_real_part) // Real part of a (possibly complex-valued) expression
    DECLARE_FUNCTION_1P(get_imag_part) // Imaginary part of a (possibly complex-valued) expression
    DECLARE_FUNCTION_1P(split_subexpressions_in_real_and_imaginary_parts) // Rewrites any subexpression() leaves inside the argument into separate real/imaginary subexpression() leaves

    DECLARE_FUNCTION_3P(symbol_subs) // Substitutes one symbol by another expression within arg, deferred until code generation
    DECLARE_FUNCTION_3P(remove_mode_from_jacobian_or_hessian) // Removes contributions of a given expansion mode from the Jacobian/Hessian assembly (used in mode-coupling analyses)
    DECLARE_FUNCTION_1P(debug_ex) // Passes the argument through unchanged but prints/logs it during evaluation, for debugging expression trees

    DECLARE_FUNCTION_1P(heaviside) // Heaviside step function
    DECLARE_FUNCTION_1P(absolute)  // Absolute value; differentiates as signum(f)*f' (i.e. treated as smooth away from f=0)
    DECLARE_FUNCTION_1P(signum)    // Signum function; differentiates to 0 everywhere, including at the origin
    DECLARE_FUNCTION_2P(minimum)   // min(a,b)
    DECLARE_FUNCTION_2P(maximum)   // max(a,b)
    DECLARE_FUNCTION_3P(piecewise_geq0) // Returns a if the condition (1st arg) is >=0, otherwise b

    // Thin GiNaC-function wrappers around the corresponding GiNaC::* simplification routines, applied after first expanding
    // all pyoomph placeholder functions (fields, test functions, ...) contained in the argument
    DECLARE_FUNCTION_1P(ginac_expand)
    DECLARE_FUNCTION_1P(ginac_normal)
    DECLARE_FUNCTION_1P(ginac_factor)
    DECLARE_FUNCTION_2P(ginac_collect)
    DECLARE_FUNCTION_1P(ginac_collect_common_factors)
    DECLARE_FUNCTION_4P(ginac_series)

    // For expansion: We have first argument: name, second argument: GiNaCPlaceHolderResolveInfo
    DECLARE_FUNCTION_2P(scale)      // Placeholder for the (dimensional) scale of a field/quantity, resolved against the code's scaling system
    DECLARE_FUNCTION_2P(test_scale) // Placeholder for the scale associated with a test function
    DECLARE_FUNCTION_2P(field)      // Placeholder for a (dimensional) field value, resolved to the actual interpolated field expression
    DECLARE_FUNCTION_2P(nondimfield) // Placeholder for the nondimensional counterpart of a field
    DECLARE_FUNCTION_2P(eval_in_domain) // Evaluates the wrapped expression as if in a different (named) domain/region, resolved via PlaceHolderResolveInfo
    DECLARE_FUNCTION_4P(eval_in_past)   // Evaluates the wrapped expression at a previous time-history index (see TimeSymbol); also carries a timestepper action flag and an "apply on integral dx" flag
    DECLARE_FUNCTION_2P(eval_at_expansion_mode) // Evaluates the wrapped expression at a specific expansion mode index (e.g. a particular Fourier/azimuthal mode)
    DECLARE_FUNCTION_2P(testfunction)    // Placeholder for a test function of a field -> expanded to test_function later on
    DECLARE_FUNCTION_2P(dimtestfunction) // Dimensional test function

    DECLARE_FUNCTION_2P(matproduct) // Matrix-matrix (or matrix-vector) product

    DECLARE_FUNCTION_2P(single_index) // Indexing a vector-valued expression by a single index (fallback used when the expression is not already a concrete GiNaC::matrix)
    DECLARE_FUNCTION_3P(double_index) // Indexing a matrix-valued expression by a pair of indices



    DECLARE_FUNCTION_2P(Diff) // Deferred/symbolic differentiation placeholder: differentiate 1st argument w.r.t. 2nd argument, resolved later by expressions::diff

    DECLARE_FUNCTION_2P(internal_function_with_element_arg) // Calls a named internal (C++-implemented) function that additionally needs access to the current element during code generation

    DECLARE_FUNCTION_2P(python_cb_function) // Invocation of a user-defined scalar callback (CustomMathExpressionWrapper) with a list of (already nondimensionalized) arguments
    DECLARE_FUNCTION_3P(python_multi_cb_function) // Invocation of a user-defined multi-return callback (CustomMultiReturnExpressionWrapper) with an argument list and the requested number of return values
    DECLARE_FUNCTION_2P(python_multi_cb_indexed_result) // Extracts one indexed result component from a python_multi_cb_function invocation

    DECLARE_FUNCTION_3P(time_stepper_weight) // Symbolic weight of a given time-derivative order/history index for a named time discretization scheme, resolved to a concrete coefficient during code generation

    DECLARE_FUNCTION_1P(eval_flag) // Placeholder that resolves to a boolean/integer evaluation flag (e.g. whether Jacobian/Hessian is currently being assembled) known only at code-generation time

  }

}
