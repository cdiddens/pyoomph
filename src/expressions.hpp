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


#pragma once

#include "ginac.hpp"
#include "pyginacstruct.hpp"
#include <vector>
#include <map>
#include "exception.hpp"
#include <complex>

namespace pyoomph
{

  const unsigned the_vector_dim = 3;

  extern std::map<std::string, GiNaC::ex> __field_name_cache;
  GiNaC::ex _get_field_name_cache(const std::string &id);

  class CustomMathExpressionBase
  {
  protected:
    static unsigned unique_counter;
    unsigned unique_id;                    // Unique id. These are necessary for sorting the expressions in the order they are created by Python => Required for Parallel processes and missing GiNaC order
    int jit_index;                         // temorarary index of the current jit compilation
    CustomMathExpressionBase *diff_parent; // Parent function of the derivative, i.e this function is a derivative of diff_parent
    int diff_index;                        // along this index
  public:
    static std::map<CustomMathExpressionBase *, int> code_map;
    CustomMathExpressionBase() : unique_id(unique_counter++), jit_index(-1), diff_parent(NULL), diff_index(-1) {}
    virtual ~CustomMathExpressionBase() { std::cout << "FREEING Expr " << unique_id << std::endl; }
    virtual CustomMathExpressionBase *get_diff_parent() const { return diff_parent; }
    virtual int get_diff_index() const { return diff_index; }
    void set_as_derivative(CustomMathExpressionBase *parent, int index)
    {
      diff_parent = parent;
      diff_index = index;
    }
    virtual double _call(double *args, unsigned int nargs) { return 0.0; }
    int get_jit_index() { return jit_index; }
    unsigned get_unique_id() { return unique_id; }
    virtual GiNaC::ex outer_derivative(const GiNaC::ex arglist, int index) { return 0; }
    virtual GiNaC::ex real_part(GiNaC::ex invok, std::vector<GiNaC::ex> arglist) { return 0; }
    virtual GiNaC::ex imag_part(GiNaC::ex invok, std::vector<GiNaC::ex> arglist) { return 0; }
    virtual std::string get_id_name() { return "unknown cb"; }
    virtual GiNaC::ex get_argument_unit(unsigned int i) { return 1; }
    virtual GiNaC::ex get_result_unit() { return 1; }
  };

  class CustomMathExpressionWrapper
  {
  public:
    CustomMathExpressionBase *cme;
    CustomMathExpressionWrapper(CustomMathExpressionBase *c) : cme(c) {}
    CustomMathExpressionWrapper(const CustomMathExpressionWrapper &c) : cme(c.cme) {}
  };

  bool operator==(const CustomMathExpressionWrapper &lhs, const CustomMathExpressionWrapper &rhs);
  bool operator<(const CustomMathExpressionWrapper &lhs, const CustomMathExpressionWrapper &rhs);

  // To return multiple values
  class CustomMultiReturnExpressionBase
  {
  public:
    static unsigned unique_counter;
    unsigned unique_id;
    double debug_c_code_epsilon;
    static std::map<CustomMultiReturnExpressionBase *, int> code_map;
    CustomMultiReturnExpressionBase() : unique_id(unique_counter++), debug_c_code_epsilon(-1.0) {}
    virtual std::string get_id_name() { return "unknown multi-ret cb"; }
    virtual std::string _get_c_code() { return ""; } // No C code present
    virtual void _call(int flag, double *args, unsigned int nargs, double *res, unsigned int nres, double *derivs) { throw_runtime_error("Should not end up here"); }
    virtual std::pair<bool, GiNaC::ex> _get_symbolic_derivative(const std::vector<GiNaC::ex> &arg_list, const int &i_res, const int &j_arg) { return std::make_pair(false, 0); }
  };

  class CustomMultiReturnExpressionWrapper
  {
  public:
    CustomMultiReturnExpressionBase *cme;
    CustomMultiReturnExpressionWrapper(CustomMultiReturnExpressionBase *c) : cme(c) {}
    CustomMultiReturnExpressionWrapper(const CustomMultiReturnExpressionWrapper &c) : cme(c.cme) {}
  };
  bool operator==(const CustomMultiReturnExpressionWrapper &lhs, const CustomMultiReturnExpressionWrapper &rhs);
  bool operator<(const CustomMultiReturnExpressionWrapper &lhs, const CustomMultiReturnExpressionWrapper &rhs);

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

  class DelayedPythonCallbackExpansion
  {
  public:
    std::function<GiNaC::ex()> f;
    DelayedPythonCallbackExpansion(std::function<GiNaC::ex()> func) : f(func) {}
  };

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
  class CustomCoordinateSystem
  {
  public:
    CustomCoordinateSystem() {}
    virtual ~CustomCoordinateSystem() {}
    virtual int vector_gradient_dimension(unsigned int basedim, bool lagrangian) { return basedim; }
    virtual GiNaC::ex grad(const GiNaC::ex &f, int dim, int edim, int flags) { throw_runtime_error("grad not implemented for this coordinate system"); }
    virtual GiNaC::ex directional_derivative(const GiNaC::ex &f, const GiNaC::ex &d, int dim, int edim, int flags) { throw_runtime_error("directional derivative not implemented for this coordinate system"); }

    virtual GiNaC::ex general_weak_differential_contribution(std::string funcname, std::vector<GiNaC::ex> lhs, GiNaC::ex rhs, int dim, int edim, int flags)
    {
      throw_runtime_error("general_weak_differential_contribution not implemented for this coordinate system");
    }
    virtual GiNaC::ex div(const GiNaC::ex &v, int dim, int edim, int flags) { throw_runtime_error("div not implemented for this coordinate system"); }
    virtual GiNaC::ex geometric_jacobian() { return 1.0; }
    virtual GiNaC::ex jacobian_for_element_size() { return 1.0; }
    virtual std::string get_id_name() { return "<unknown coordinate system>"; }
    virtual GiNaC::ex get_mode_expansion_of_var_or_test(pyoomph::FiniteElementCode *mycode, std::string fieldname, bool is_field, bool is_dim, GiNaC::ex expr, std::string where, int expansion_mode) { return expr; }
  };

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

  PYGINACSTRUCT(pyoomph::CustomMathExpressionWrapper, GiNaCCustomMathExpressionWrapper);
  template <>
  void GiNaC::GiNaCCustomMathExpressionWrapper::print(const print_context &c, unsigned level) const;

  PYGINACSTRUCT(pyoomph::CustomMultiReturnExpressionWrapper, GiNaCCustomMultiReturnExpressionWrapper);
  template <>
  void GiNaCCustomMultiReturnExpressionWrapper::print(const print_context &c, unsigned level) const;

  PYGINACSTRUCT(pyoomph::CustomMultiReturnExpressionResultSymbol, GiNaCCustomMultiReturnExpressionResultSymbol);
  template <>
  void GiNaCCustomMultiReturnExpressionResultSymbol::print(const print_context &c, unsigned level) const;

  PYGINACSTRUCT(pyoomph::CustomCoordinateSystemWrapper, GiNaCCustomCoordinateSystemWrapper);
  template <>
  void GiNaCCustomCoordinateSystemWrapper::print(const print_context &c, unsigned level) const;

  PYGINACSTRUCT(pyoomph::GlobalParameterWrapper, GiNaCGlobalParameterWrapper);
  template <>
  void GiNaCGlobalParameterWrapper::print(const print_context &c, unsigned level) const;
  template <>
  bool GiNaCGlobalParameterWrapper::info(unsigned inf) const;


  PYGINACSTRUCT(pyoomph::PlaceHolderResolveInfo, GiNaCPlaceHolderResolveInfo);
  template <>
  void GiNaCPlaceHolderResolveInfo::print(const print_context &c, unsigned level) const;

  PYGINACSTRUCT(pyoomph::DelayedPythonCallbackExpansionWrapper, GiNaCDelayedPythonCallbackExpansion);
  template <>
  void GiNaCDelayedPythonCallbackExpansion::print(const print_context &c, unsigned level) const;

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

  //typedef symbol potential_real_symbol;
  typedef realsymbol potential_real_symbol;


  
    
}

namespace pyoomph
{

  extern CustomCoordinateSystem __no_coordinate_system;
  extern GiNaC::GiNaCCustomCoordinateSystemWrapper __no_coordinate_system_wrapper;

  extern std::map<std::string, GiNaC::possymbol> base_units;

  namespace expressions
  {

    extern GiNaC::symbol nnode;
    extern GiNaC::potential_real_symbol x;
    extern GiNaC::potential_real_symbol y;
    extern GiNaC::potential_real_symbol z;
    extern GiNaC::potential_real_symbol X;
    extern GiNaC::potential_real_symbol Y; // Lagrangian
    extern GiNaC::potential_real_symbol Z;
    extern GiNaC::potential_real_symbol local_coordinate_1,local_coordinate_2,local_coordinate_3;
    extern GiNaC::potential_real_symbol zeta_coordinate_1, zeta_coordinate_2, zeta_coordinate_3; 
    extern GiNaC::potential_real_symbol nx;
    extern GiNaC::potential_real_symbol ny;
    extern GiNaC::potential_real_symbol nz;

    extern GiNaC::potential_real_symbol t, _dt_BDF1, _dt_BDF2, _dt_Newmark2;
    extern GiNaC::potential_real_symbol __partial_t_mass_matrix; // This symbol is used to identify partial_t terms to put in the mass matrix
    extern GiNaC::potential_real_symbol dt;
    extern GiNaC::potential_real_symbol timefrac_tracer;
    extern GiNaC::idx l_shape;
    extern GiNaC::idx l_test;
    extern GiNaC::potential_real_symbol *proj_on_test_function;
    extern int el_dim;

    GiNaC::ex diff(const GiNaC::ex &what, const GiNaC::ex &wrto);
    bool collect_base_units(GiNaC::ex arg, GiNaC::ex &factor, GiNaC::ex &units, GiNaC::ex &rest);

    GiNaC::ex subs_fields(const GiNaC::ex &arg, const std::map<std::string, GiNaC::ex> &fields, const std::map<std::string, GiNaC::ex> &nondimfields, const std::map<std::string, GiNaC::ex> &globalparams);

    GiNaC::ex replace_global_params_by_current_values(const GiNaC::ex &in);

    double eval_to_double(const GiNaC::ex &inp);
    std::complex<double> eval_to_complex(const GiNaC::ex &inp);    

    DECLARE_FUNCTION_5P(grad) // 1: what to grad, 2: nodal dimension or -1, 3: element dimension or -1, 4: Coordinate System object, 5: withdim(0,1)
    DECLARE_FUNCTION_6P(directional_derivative)
    DECLARE_FUNCTION_7P(general_weak_differential_contribution)
    DECLARE_FUNCTION_2P(dot)
    DECLARE_FUNCTION_2P(double_dot)
    DECLARE_FUNCTION_2P(contract)
    DECLARE_FUNCTION_4P(weak)
    DECLARE_FUNCTION_5P(div)
    DECLARE_FUNCTION_1P(transpose)
    DECLARE_FUNCTION_1P(trace)
    DECLARE_FUNCTION_2P(determinant)    
    DECLARE_FUNCTION_3P(inverse_matrix)    

    DECLARE_FUNCTION_4P(minimize_functional_derivative)    

    DECLARE_FUNCTION_4P(unitvect)

    DECLARE_FUNCTION_1P(subexpression)

    DECLARE_FUNCTION_1P(get_real_part)
    DECLARE_FUNCTION_1P(get_imag_part)
    DECLARE_FUNCTION_1P(split_subexpressions_in_real_and_imaginary_parts)

    DECLARE_FUNCTION_3P(symbol_subs)
    DECLARE_FUNCTION_3P(remove_mode_from_jacobian_or_hessian)
    DECLARE_FUNCTION_1P(debug_ex)

    DECLARE_FUNCTION_1P(heaviside)
    DECLARE_FUNCTION_1P(absolute)
    DECLARE_FUNCTION_1P(signum)
    DECLARE_FUNCTION_2P(minimum)
    DECLARE_FUNCTION_2P(maximum)
    DECLARE_FUNCTION_3P(piecewise_geq0)

    DECLARE_FUNCTION_1P(ginac_expand)
    DECLARE_FUNCTION_1P(ginac_normal)
    DECLARE_FUNCTION_1P(ginac_factor)
    DECLARE_FUNCTION_2P(ginac_collect)
    DECLARE_FUNCTION_1P(ginac_collect_common_factors)
    DECLARE_FUNCTION_4P(ginac_series)

    // For expansion: We have first argument: name, second argument: GiNaCPlaceHolderResolveInfo
    DECLARE_FUNCTION_2P(scale)
    DECLARE_FUNCTION_2P(test_scale)
    DECLARE_FUNCTION_2P(field)
    DECLARE_FUNCTION_2P(nondimfield)
    DECLARE_FUNCTION_2P(eval_in_domain)
    DECLARE_FUNCTION_3P(eval_in_past)
    DECLARE_FUNCTION_2P(eval_at_expansion_mode)
    DECLARE_FUNCTION_2P(testfunction)    // Placeholder for a test function of a field -> expanded to test_function later on
    DECLARE_FUNCTION_2P(dimtestfunction) // Dimensional test function

    DECLARE_FUNCTION_2P(matproduct)

    DECLARE_FUNCTION_2P(single_index)
    DECLARE_FUNCTION_3P(double_index)

    
    

    DECLARE_FUNCTION_2P(Diff)

    DECLARE_FUNCTION_2P(internal_function_with_element_arg)

    DECLARE_FUNCTION_2P(python_cb_function)
    DECLARE_FUNCTION_3P(python_multi_cb_function)
    DECLARE_FUNCTION_2P(python_multi_cb_indexed_result)

    DECLARE_FUNCTION_3P(time_stepper_weight)

    DECLARE_FUNCTION_1P(eval_flag)

  }

}
