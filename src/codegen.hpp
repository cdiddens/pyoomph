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
#include <set>
#include "expressions.hpp"
#include "jitbridge.h"

namespace pyoomph
{

   class FiniteElementCode;

   class Equations
   {
   protected:
      FiniteElementCode *current_codegen;

   public:
      Equations() : current_codegen(NULL) {}
      virtual void _set_current_codegen(FiniteElementCode *cg) { current_codegen = cg; }
      virtual FiniteElementCode *_get_current_codegen() { return current_codegen; }
      virtual void _define_element() = 0;
      virtual void _define_fields() = 0;
   };

   class BasisFunction;
   class FiniteElementField;

   class SpatialIntegralSymbol
   {
   protected:
      FiniteElementCode *code;
      bool lagrangian;
      bool derived, derived2, derived_by_second_index; // Last one indicates that we have derived with respect to l_shape2 in the Hessian. Important for first order derivatives only
      int deriv_direction;
      int deriv_direction2;

   public:
      int expansion_mode = 0; // For mode expansions
      unsigned history_step=0; // For evaluations in past
      bool no_jacobian = false;
      bool no_hessian = false;
      bool simple_unity_integral = false;  // If true, dx []=J=sqrt(det(g))] from the element won't be considered

      bool is_lagrangian() const { return lagrangian; }
      bool is_derived() const { return derived; }
      int get_derived_direction() const { return deriv_direction; }
      bool is_derived2() const { return derived2; }
      int get_derived_direction2() const { return deriv_direction2; }
      bool is_derived_by_lshape2() const { return derived_by_second_index; }
      const FiniteElementCode *get_code() const { return code; }
      SpatialIntegralSymbol(FiniteElementCode *_code, bool _lagrangian) : code(_code), lagrangian(_lagrangian), derived(false), derived2(false), derived_by_second_index(false), deriv_direction(0), deriv_direction2(0) {}
      SpatialIntegralSymbol(FiniteElementCode *_code, bool _lagrangian, int direction) : code(_code), lagrangian(_lagrangian), derived(true), derived2(false), derived_by_second_index(false), deriv_direction(direction), deriv_direction2(0) {}
      SpatialIntegralSymbol(FiniteElementCode *_code, bool _lagrangian, int direction, std::string second_index_dummy) : code(_code), lagrangian(_lagrangian), derived(true), derived2(false), derived_by_second_index(true), deriv_direction(direction), deriv_direction2(0) {}
      SpatialIntegralSymbol(FiniteElementCode *_code, bool _lagrangian, int direction, int direction2) : code(_code), lagrangian(_lagrangian), derived(true), derived2(true), derived_by_second_index(false), deriv_direction(direction), deriv_direction2(direction2) {}
   };

   bool operator==(const SpatialIntegralSymbol &lhs, const SpatialIntegralSymbol &rhs);
   bool operator<(const SpatialIntegralSymbol &lhs, const SpatialIntegralSymbol &rhs);

   class ElementSizeSymbol
   {
   protected:
      FiniteElementCode *code;
      bool lagrangian, consider_coordsys;
      bool derived, derived2, derived_by_second_index;
      int deriv_direction;
      int deriv_direction2;

   public:
      bool is_lagrangian() const { return lagrangian; }
      bool is_with_coordsys() const { return consider_coordsys; } // If true, it will be the integral including terms like 2*pi*r for axisymm, otherwise not
      bool is_derived() const { return derived; }
      int get_derived_direction() const { return deriv_direction; }
      bool is_derived_by_lshape2() const { return derived_by_second_index; }
      bool is_derived2() const { return derived2; }
      int get_derived_direction2() const { return deriv_direction2; }
      const FiniteElementCode *get_code() const { return code; }
      ElementSizeSymbol(FiniteElementCode *_code, bool _lagrangian, bool _consider_coordsys) : code(_code), lagrangian(_lagrangian), consider_coordsys(_consider_coordsys), derived(false), derived2(false), derived_by_second_index(false), deriv_direction(0), deriv_direction2(0) {}
      ElementSizeSymbol(FiniteElementCode *_code, bool _lagrangian, bool _consider_coordsys, int direction) : code(_code), lagrangian(_lagrangian), consider_coordsys(_consider_coordsys), derived(true), derived2(false), derived_by_second_index(false), deriv_direction(direction), deriv_direction2(0) {}
      ElementSizeSymbol(FiniteElementCode *_code, bool _lagrangian, bool _consider_coordsys, int direction, std::string second_index_dummy) : code(_code), lagrangian(_lagrangian), consider_coordsys(_consider_coordsys), derived(true), derived2(false), derived_by_second_index(true), deriv_direction(direction), deriv_direction2(0) {}
      ElementSizeSymbol(FiniteElementCode *_code, bool _lagrangian, bool _consider_coordsys, int direction, int direction2) : code(_code), lagrangian(_lagrangian), consider_coordsys(_consider_coordsys), derived(true), derived2(true), derived_by_second_index(false), deriv_direction(direction), deriv_direction2(direction2) {}
   };

   bool operator==(const ElementSizeSymbol &lhs, const ElementSizeSymbol &rhs);
   bool operator<(const ElementSizeSymbol &lhs, const ElementSizeSymbol &rhs);

   class NodalDeltaSymbol
   {
   protected:
      FiniteElementCode *code;

   public:
      const FiniteElementCode *get_code() const { return code; }
      NodalDeltaSymbol(FiniteElementCode *_code) : code(_code) {}
   };

   bool operator==(const NodalDeltaSymbol &lhs, const NodalDeltaSymbol &rhs);
   bool operator<(const NodalDeltaSymbol &lhs, const NodalDeltaSymbol &rhs);

   class NormalSymbol
   {
   protected:
      FiniteElementCode *code;
      unsigned direction;
      int deriv_direction;
      int deriv_direction2;
      bool derived_by_second_index; // indicates that we have derived with respect to l_shape2 in the Hessian. Important for first order derivatives only
   public:
      //bool is_eigenexpansion = false; // Used for symmetry breaking: It gives then dn_i/dX^{0l}_j* X^{ml}_j
      int expansion_mode = 0;         // For mode expansions
      bool no_jacobian = false;
      bool no_hessian = false;
      bool is_derived_by_lshape2() const { return derived_by_second_index; }
      const FiniteElementCode *get_code() const { return code; }
      unsigned get_direction() const { return direction; }
      int get_derived_direction() const { return deriv_direction; }
      int get_derived_direction2() const { return deriv_direction2; }
      NormalSymbol(FiniteElementCode *_code, unsigned _direction, int _deriv_direction = -1) : code(_code), direction(_direction), deriv_direction(_deriv_direction), deriv_direction2(-1), derived_by_second_index(false) {}
      NormalSymbol(FiniteElementCode *_code, unsigned _direction, int _deriv_direction, int _deriv_direction2) : code(_code), direction(_direction), deriv_direction(_deriv_direction), deriv_direction2(_deriv_direction2), derived_by_second_index(false) {}
      NormalSymbol(FiniteElementCode *_code, unsigned _direction, int _deriv_direction, int _deriv_direction2, bool _derived_by_second_index) : code(_code), direction(_direction), deriv_direction(_deriv_direction), deriv_direction2(_deriv_direction2), derived_by_second_index(_derived_by_second_index) {}
   };

   bool operator==(const NormalSymbol &lhs, const NormalSymbol &rhs);
   bool operator<(const NormalSymbol &lhs, const NormalSymbol &rhs);

   class ShapeExpansion
   {
   protected:
   public:
      FiniteElementField *field;
      unsigned dt_order;
      std::string dt_scheme;
      BasisFunction *basis;
      bool is_derived;             // A derived shape expansion is not sum(u_i(t) * phi_i(x))_i, but just a symbolic (phi_i(x))_i -> required for jacobian entries etc
      bool is_derived_other_index; // For hessian, we have two looping indices. This is accounted for here
      int nodal_coord_dir;         // If this is !=-1, it means that it is the derivative d(dpsidx(l,i)*u^l)/d(nodal_coordinate_X_j^k)
      int nodal_coord_dir2;        // Second coordinate derivatives
      int time_history_index;
      bool no_jacobian, no_hessian;
      int expansion_mode; // For mode expansions

      ShapeExpansion(FiniteElementField *_field, unsigned _dt_order, BasisFunction *_basis, bool _is_derived = false, int _nodal_coord_dir = -1) : field(_field), dt_order(_dt_order), dt_scheme("TIME_DIFF_SCHEME_NOT_SET"), basis(_basis), is_derived(_is_derived), is_derived_other_index(false), nodal_coord_dir(_nodal_coord_dir), nodal_coord_dir2(-1), time_history_index(0), no_jacobian(false), no_hessian(false), expansion_mode(0) {}

      ShapeExpansion(FiniteElementField *_field, unsigned _dt_order, BasisFunction *_basis, std::string ts, bool _is_derived = false, int _nodal_coord_dir = -1) : field(_field), dt_order(_dt_order), dt_scheme(ts), basis(_basis), is_derived(_is_derived), is_derived_other_index(false), nodal_coord_dir(_nodal_coord_dir), nodal_coord_dir2(-1), time_history_index(0), no_jacobian(false), no_hessian(false), expansion_mode(0) {}

      ShapeExpansion(FiniteElementField *_field, unsigned _dt_order, BasisFunction *_basis, std::string ts, bool _is_derived, int _nodal_coord_dir, bool _is_derived_other_index) : field(_field), dt_order(_dt_order), dt_scheme(ts), basis(_basis), is_derived(_is_derived), is_derived_other_index(_is_derived_other_index), nodal_coord_dir(_nodal_coord_dir), nodal_coord_dir2(-1), time_history_index(0), no_jacobian(false), no_hessian(false), expansion_mode(0) {}

      ShapeExpansion(FiniteElementField *_field, unsigned _dt_order, BasisFunction *_basis, std::string ts, bool _is_derived, int _nodal_coord_dir, bool _is_derived_other_index, int _nodal_coord_dir2) : field(_field), dt_order(_dt_order), dt_scheme(ts), basis(_basis), is_derived(_is_derived), is_derived_other_index(_is_derived_other_index), nodal_coord_dir(_nodal_coord_dir), nodal_coord_dir2(_nodal_coord_dir2), time_history_index(0), no_jacobian(false), no_hessian(false), expansion_mode(0) {}

      virtual std::string get_dt_values_name(FiniteElementCode *forcode) const;
      virtual std::string get_timedisc_scheme(FiniteElementCode *forcode) const;
      virtual std::string get_spatial_interpolation_name(FiniteElementCode *forcode) const;
      virtual std::string get_num_nodes_str(FiniteElementCode *forcode) const;
      virtual std::string get_nodal_index_str(FiniteElementCode *forcode) const;
      virtual std::string get_nodal_data_string(FiniteElementCode *forcode, std::string indexstr) const;
      virtual std::string get_shape_string(FiniteElementCode *forcode, std::string nodal_index) const;
      virtual FiniteElementCode *can_be_a_positional_derivative_symbol(const GiNaC::symbol &s, FiniteElementCode *domain_to_check = NULL) const;
   };

   bool operator==(const ShapeExpansion &lhs, const ShapeExpansion &rhs);
   bool operator<(const ShapeExpansion &lhs, const ShapeExpansion &rhs);

   class TestFunction
   {
   protected:
   public:
      FiniteElementField *field;
      BasisFunction *basis;
      int nodal_coord_dir;         // If this is !=-1, it means that it is the derivative d(dpsidx(l,i)*u^l)/d(nodal_coordinate_X_j^k)
      int nodal_coord_dir2;        // If this is !=-1, it means that it is the derivative d(dpsidx(l,i)*u^l)/d(nodal_coordinate_X_j^k)
      bool is_derived_other_index; // For hessian, we have two looping indices. This is accounted for here
      TestFunction(FiniteElementField *_field, BasisFunction *_basis, int _nodal_coord_dir = -1) : field(_field), basis(_basis), nodal_coord_dir(_nodal_coord_dir), nodal_coord_dir2(-1), is_derived_other_index(false) {}
      TestFunction(FiniteElementField *_field, BasisFunction *_basis, int _nodal_coord_dir, bool _is_derived_other_index) : field(_field), basis(_basis), nodal_coord_dir(_nodal_coord_dir), nodal_coord_dir2(-1), is_derived_other_index(_is_derived_other_index) {}
      TestFunction(FiniteElementField *_field, BasisFunction *_basis, int _nodal_coord_dir, bool _is_derived_other_index, int _nodal_coord_dir2) : field(_field), basis(_basis), nodal_coord_dir(_nodal_coord_dir), nodal_coord_dir2(_nodal_coord_dir2), is_derived_other_index(_is_derived_other_index) {}
   };

   bool operator==(const TestFunction &lhs, const TestFunction &rhs);
   bool operator<(const TestFunction &lhs, const TestFunction &rhs);

   class FiniteElementCodeSubExpression;
   class SubExpression
   {
   public:
      FiniteElementCode *code;
      GiNaC::ex expr; // Expression
      SubExpression(FiniteElementCode *c, const GiNaC::ex &e) : code(c), expr(e) {}
   };

   class MultiRetCallback
   {
   public:
      FiniteElementCode *code;
      GiNaC::ex invok;    // Full invokation to sort things
      int retindex;       // Return value index
      int derived_by_arg; // Return value index
      MultiRetCallback(FiniteElementCode *c, const GiNaC::ex &inv, const int &index) : code(c), invok(inv), retindex(index), derived_by_arg(-1) {}
      MultiRetCallback(FiniteElementCode *c, const GiNaC::ex &inv, const int &index, const int &derived) : code(c), invok(inv), retindex(index), derived_by_arg(derived) {}
   };
   bool operator==(const MultiRetCallback &lhs, const MultiRetCallback &rhs);
   bool operator<(const MultiRetCallback &lhs, const MultiRetCallback &rhs);
}

namespace GiNaC
{

   PYGINACSTRUCT(pyoomph::SpatialIntegralSymbol, GiNaCSpatialIntegralSymbol);
   template <>
   void GiNaCSpatialIntegralSymbol::print(const print_context &c, unsigned level) const;
   template <>
   GiNaC::ex GiNaCSpatialIntegralSymbol::derivative(const GiNaC::symbol &s) const;

   PYGINACSTRUCT(pyoomph::ElementSizeSymbol, GiNaCElementSizeSymbol);
   template <>
   void GiNaCElementSizeSymbol::print(const print_context &c, unsigned level) const;
   template <>
   GiNaC::ex GiNaCElementSizeSymbol::derivative(const GiNaC::symbol &s) const;

   PYGINACSTRUCT(pyoomph::NodalDeltaSymbol, GiNaCNodalDeltaSymbol);
   template <>
   void GiNaCNodalDeltaSymbol::print(const print_context &c, unsigned level) const;
   template <>
   GiNaC::ex GiNaCNodalDeltaSymbol::derivative(const GiNaC::symbol &s) const;

   PYGINACSTRUCT(pyoomph::NormalSymbol, GiNaCNormalSymbol);
   template <>
   void GiNaCNormalSymbol::print(const print_context &c, unsigned level) const;
   template <>
   GiNaC::ex GiNaCNormalSymbol::derivative(const GiNaC::symbol &s) const;

   PYGINACSTRUCT(pyoomph::ShapeExpansion, GiNaCShapeExpansion);
   template <>
   void GiNaCShapeExpansion::print(const print_context &c, unsigned level) const;
   template <>
   GiNaC::ex GiNaCShapeExpansion::derivative(const GiNaC::symbol &s) const;
   // template <> GiNaC::ex GiNaCShapeExpansion::real_part() const;
   // template <> GiNaC::ex GiNaCShapeExpansion::imag_part() const;

   /*
   class GiNaCShapeExpansion : public GiNaC::structure<pyoomph::ShapeExpansion, GiNaC::compare_std_less>
   {
    public:
     GiNaCShapeExpansion(const pyoomph::ShapeExpansion & s) : GiNaC::structure<pyoomph::ShapeExpansion, GiNaC::compare_std_less>(s) {}
     void print(const print_context & c, unsigned level) const;
     GiNaC::ex derivative(const GiNaC::symbol & s) const;
   };*/
   // template <> GiNaC::ex GiNaCShapeExpansion::real_part() const;
   // template <> GiNaC::ex GiNaCShapeExpansion::imag_part() const;

   PYGINACSTRUCT(pyoomph::SubExpression, GiNaCSubExpression);
   template <>
   void GiNaCSubExpression::print(const print_context &c, unsigned level) const;
   template <>
   GiNaC::ex GiNaCSubExpression::derivative(const GiNaC::symbol &s) const;

   PYGINACSTRUCT(pyoomph::MultiRetCallback, GiNaCMultiRetCallback);
   template <>
   void GiNaCMultiRetCallback::print(const print_context &c, unsigned level) const;
   template <>
   GiNaC::ex GiNaCMultiRetCallback::derivative(const GiNaC::symbol &s) const;
   template <>
   GiNaC::ex GiNaCMultiRetCallback::subs(const GiNaC::exmap &m, unsigned options) const;

   PYGINACSTRUCT(pyoomph::TestFunction, GiNaCTestFunction);
   template <>
   void GiNaCTestFunction::print(const print_context &c, unsigned level) const;
   template <>
   GiNaC::ex GiNaCTestFunction::derivative(const GiNaC::symbol &s) const;

   // New print context which automatically writes the C++ code by expansion
   class print_FEM_options
   {
   public:
      pyoomph::FiniteElementCode *for_code;
      bool in_subexpr_deriv;
      bool ignore_custom;
      print_FEM_options() : for_code(NULL), in_subexpr_deriv(false), ignore_custom(false) {}
   };

   class print_csrc_FEM : public GiNaC::print_csrc_double
   {
      //    GINAC_DECLARE_PRINT_CONTEXT(print_csrc_FEM, GiNaC::print_dflt)
   public:
      print_FEM_options *FEM_opts;
      print_csrc_FEM();
      print_csrc_FEM(std::ostream &, print_FEM_options *fem_opts, unsigned options = 0);
   };

   class print_latex_FEM : public GiNaC::print_latex
   {
   public:
      print_FEM_options *FEM_opts;
      print_latex_FEM();
      print_latex_FEM(std::ostream &, print_FEM_options *fem_opts, unsigned options = 0);
   };

   class SortedGiNaC
    {
      public:
        std::vector<SortedGiNaC*> op;        
        virtual ~SortedGiNaC() ;        
        static SortedGiNaC * factory(const ex & e,std::ostream &os, GiNaC::print_FEM_options &csrc_opts);                
        virtual std::string to_string(std::ostream &os, GiNaC::print_FEM_options &csrc_opts) =0;       
        virtual int add_order()=0;
        virtual int mul_order()=0;        
        bool add_sort_compare(SortedGiNaC * other, std::ostream &os, GiNaC::print_FEM_options &csrc_opts);        
        bool mul_sort_compare(SortedGiNaC * other, std::ostream &os, GiNaC::print_FEM_options &csrc_opts);        
    };

    class SortedGiNaCNumeric : public SortedGiNaC
    {
      public:
        GiNaC::numeric value;

        SortedGiNaCNumeric(GiNaC::numeric v) : SortedGiNaC(), value(v) {}
        virtual std::string to_string(std::ostream &os, GiNaC::print_FEM_options &csrc_opts) override;
        virtual int add_order() override { return 0; }
        virtual int mul_order() override { return 0; }
        
    };

    class SortedGiNaCAdd : public SortedGiNaC
    {
      public:
        SortedGiNaCAdd(const std::vector<SortedGiNaC*> & ops) : SortedGiNaC()  {op=ops;}
        virtual std::string to_string(std::ostream &os, GiNaC::print_FEM_options &csrc_opts) override;        
        virtual int add_order() override;        
        virtual int mul_order() override { return 5; }
            
    };

    class SortedGiNaCMul : public SortedGiNaC
    {
      public:
        SortedGiNaCMul(const std::vector<SortedGiNaC*> & ops) : SortedGiNaC(){op=ops;}
        virtual std::string to_string(std::ostream &os, GiNaC::print_FEM_options &csrc_opts) override;        
        virtual int add_order() override { return 3; }
        virtual int mul_order() override;        
    };

    class SortedGiNaCPow : public SortedGiNaC
    {
      public:
        SortedGiNaCPow(SortedGiNaC * base, SortedGiNaC * exp) : SortedGiNaC()
        {
            op.push_back(base);
            op.push_back(exp);
        }
        virtual std::string to_string(std::ostream &os, GiNaC::print_FEM_options &csrc_opts) override;        
        virtual int add_order() override { return 4; }
        virtual int mul_order() override { return 4; }
    };

    class SortedGiNaCFunction : public SortedGiNaC
    {
      public:
        std::string fname;
        SortedGiNaCFunction(const std::string & fname, const std::vector<SortedGiNaC*> & ops) : SortedGiNaC(), fname(fname)     {            op=ops;        }
        virtual std::string to_string(std::ostream &os, GiNaC::print_FEM_options &csrc_opts) override;        
        virtual int add_order() override { return 1; }
        virtual int mul_order() override { return 1; }
    };

    class SortedGiNaCSymbol : public SortedGiNaC
    {
      public:
        std::string vname;        
        SortedGiNaCSymbol(const std::string & vname) : SortedGiNaC(), vname(vname) {}
        virtual std::string to_string(std::ostream &os, GiNaC::print_FEM_options &csrc_opts) override {return vname;}
        virtual int add_order() override {return 2;}
        virtual int mul_order() override {return 2;}
    };

    class SortedGiNaCStruct : public SortedGiNaC
    {
      public:
        ex contents;
        SortedGiNaCStruct(GiNaC::ex _contents) : SortedGiNaC(), contents(_contents) {}
        virtual std::string to_string(std::ostream &os, GiNaC::print_FEM_options &csrc_opts) override;        
        virtual int add_order() override { return 6; }
        virtual int mul_order() override;
    };

    std::ostream & print_sorted_GiNaC(ex  e,std::ostream &os, GiNaC::print_FEM_options &csrc_opts);

}

namespace pyoomph
{
   void print_simplest_form(GiNaC::ex expr, std::ostream &os, GiNaC::print_FEM_options &csrc_opts);


   class LaTeXPrinter
   {
   public:
      virtual void _add_LaTeX_expression(std::map<std::string, std::string> info, std::string expr, FiniteElementCode *code) {}  // Will be implemented by python
      virtual std::string _get_LaTeX_expression(std::map<std::string, std::string> info, FiniteElementCode *code) { return ""; } // Will be implemented by python
      void print(std::map<std::string, std::string> info, const GiNaC::ex &expression, GiNaC::print_FEM_options &ops)
      {
         std::ostringstream oss;
         expression.eval().print(GiNaC::print_latex_FEM(oss, &ops));
         this->_add_LaTeX_expression(info, oss.str(), ops.for_code);
      }
   };

   class DrawUnitsOutOfSubexpressions : public GiNaC::map_function
   {
   protected:
      FiniteElementCode *code;

   public:
      DrawUnitsOutOfSubexpressions(FiniteElementCode *code_) : code(code_) {}
      GiNaC::ex operator()(const GiNaC::ex &inp);
   };

   class RemoveSubexpressionsByIndentity : public GiNaC::map_function
   {
   protected:
      FiniteElementCode *code;

   public:
      RemoveSubexpressionsByIndentity(FiniteElementCode *code_) : code(code_) {}
      GiNaC::ex operator()(const GiNaC::ex &inp);
   };

   class ReplaceFieldsToNonDimFields : public GiNaC::map_function
   {
   protected:
      FiniteElementCode *code;
      std::string where;

   public:
      unsigned repl_count;
      GiNaC::ex extra_test_scale;
      ReplaceFieldsToNonDimFields(FiniteElementCode *code_, std::string _where) : code(code_), where(_where), repl_count(0), extra_test_scale(1) {}
      GiNaC::ex operator()(const GiNaC::ex &inp);
   };

   class DerivedShapeExpansionsToUnity : public GiNaC::map_function
   {
   protected:
     BasisFunction * ensure_basis; // If set, derived shape expansions on other basis functions will become zero
     int ensure_dt_order; // Same as above but for time derivaties, -1 deactivates it
     std::string ensure_dt_scheme;
   public:
      DerivedShapeExpansionsToUnity(BasisFunction * _ensure_basis=NULL,int _ensure_dt_order=-1,std::string _ensure_dt_scheme=""): ensure_basis(_ensure_basis), ensure_dt_order(_ensure_dt_order), ensure_dt_scheme(_ensure_dt_scheme) {}
      GiNaC::ex operator()(const GiNaC::ex &inp)
      {
         if (GiNaC::is_a<GiNaC::GiNaCShapeExpansion>(inp))
         {
            auto &shp = (GiNaC::ex_to<GiNaC::GiNaCShapeExpansion>(inp)).get_struct();
            if (shp.is_derived)
            {
               if (this->ensure_basis)
               {
                 if (shp.basis!=this->ensure_basis) return 0;
               }
               if (this->ensure_dt_order!=-1)
               {
                 if (shp.dt_order!=this->ensure_dt_order) return 0;
               }
               if (this->ensure_dt_scheme!="")
               {
                if (shp.dt_scheme!=this->ensure_dt_scheme) return 0;
               }
               return 1;
            }
            else
               return inp.map(*this);
         }
         else
            return inp.map(*this);
      }
   };

   // Each space will be interpolated individually
   // A space also defines the basis functions
   // The space can be nodal of this element, internal (for discontinous spaces) or part of bulk or external elements

   class FiniteElementSpace;
   class BasisFunction
   {
   protected:
      FiniteElementSpace *space;
      std::vector<BasisFunction *> basis_deriv_x, lagr_deriv_x, local_coord_deriv_x;

   public:
      BasisFunction(FiniteElementSpace *_space) : space(_space) {}
      virtual ~BasisFunction();
      virtual BasisFunction *get_diff_x(unsigned direction);
      virtual BasisFunction *get_diff_X(unsigned direction);
      virtual BasisFunction *get_diff_S(unsigned direction);
      virtual std::string to_string();
      virtual const FiniteElementSpace *get_space() const { return space; }
      virtual std::string get_dx_str() const { return "d0x"; }
      virtual std::string get_shape_string(FiniteElementCode *forcode, std::string nodal_index) const;
      virtual std::string get_c_varname(FiniteElementCode *forcode, std::string test_index);
   };

   class D1XBasisFunction : public BasisFunction
   {
   protected:
      unsigned direction;

   public:
      D1XBasisFunction(FiniteElementSpace *_space, unsigned _direction) : BasisFunction(_space), direction(_direction) {}
      virtual BasisFunction *get_diff_x(unsigned direction);
      virtual BasisFunction *get_diff_X(unsigned direction);
      virtual BasisFunction *get_diff_S(unsigned direction);
      virtual std::string to_string();
      virtual std::string get_dx_str() const { return "d1x" + std::to_string(direction); }
      virtual std::string get_shape_string(FiniteElementCode *forcode, std::string nodal_index) const;
      virtual std::string get_c_varname(FiniteElementCode *forcode, std::string test_index);
      virtual unsigned get_direction() const { return direction; }
   };

   class D1XBasisFunctionLagr : public D1XBasisFunction
   {
   public:
      D1XBasisFunctionLagr(FiniteElementSpace *_space, unsigned _direction) : D1XBasisFunction(_space, _direction) {}
      virtual std::string to_string();
      virtual std::string get_dx_str() const { return "d1X" + std::to_string(direction); }
      virtual std::string get_shape_string(FiniteElementCode *forcode, std::string nodal_index) const;
      virtual std::string get_c_varname(FiniteElementCode *forcode, std::string test_index);
   };

   class D1XBasisFunctionLocalCoord : public D1XBasisFunctionLagr
   {
   public:
      D1XBasisFunctionLocalCoord(FiniteElementSpace *_space, unsigned _direction) : D1XBasisFunctionLagr(_space, _direction) {}
      virtual std::string to_string();
      virtual std::string get_dx_str() const { return "d1S" + std::to_string(direction); }
      virtual std::string get_shape_string(FiniteElementCode *forcode, std::string nodal_index) const;
      virtual std::string get_c_varname(FiniteElementCode *forcode, std::string test_index);
   };

   class FiniteElementCode;
   class FiniteElementSpace
   {
   protected:
      FiniteElementCode *code;
      std::string name;
      BasisFunction *Basis;

   public:
      virtual std::string get_eqn_number_str(FiniteElementCode *forcode) const;
      virtual bool is_external() { return false; }
      virtual bool is_basis_derivative_zero(BasisFunction *b, unsigned dir) { return false; }
      virtual std::string get_num_nodes_str(FiniteElementCode *forcode) const;
      virtual void write_spatial_interpolation(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, std::set<ShapeExpansion> &required_shapeexps, bool including_nodal_diffs, bool for_hessian);
      virtual void write_nodal_time_interpolation(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, std::set<ShapeExpansion> &required_shapeexps);
      virtual bool write_generic_RJM_contribution(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, GiNaC::ex for_what, bool hessian);
      virtual void write_generic_RJM_jacobian_contribution(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, GiNaC::ex for_what, bool hanging_eqns,FiniteElementField * residual_field);
      virtual bool write_generic_Hessian_contribution(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, GiNaC::ex for_what, bool hanging_eqns);
      FiniteElementSpace(FiniteElementCode *_code, const std::string &_name) : code(_code), name(_name), Basis(new BasisFunction(this)) {}
      virtual ~FiniteElementSpace()
      {
         if (Basis)
            delete Basis;
      }
      BasisFunction *get_basis() { return Basis; }
      std::string get_name() const { return name; }
      virtual std::string get_shape_name() const { return name; }
      virtual std::string get_hang_name() const { return name; }
      virtual bool can_have_hanging_nodes() { return false; }
      virtual bool need_interpolation_loop() { return true; }
      FiniteElementCode *get_code() const { return code; }
   };

   class ContinuousFiniteElementSpace : public FiniteElementSpace
   {
   public:
      bool can_have_hanging_nodes() override;
      ContinuousFiniteElementSpace(FiniteElementCode *_code, const std::string &_name) : FiniteElementSpace(_code, _name) {}
   };

   class PositionFiniteElementSpace : public ContinuousFiniteElementSpace
   {
   public:
      virtual std::string get_num_nodes_str(FiniteElementCode *forcode) const;
      virtual std::string get_eqn_number_str(FiniteElementCode *forcode) const;
      PositionFiniteElementSpace(FiniteElementCode *_code, const std::string &_name) : ContinuousFiniteElementSpace(_code, _name) {}
      virtual void write_generic_RJM_jacobian_contribution(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, GiNaC::ex for_what, bool hanging_eqns,FiniteElementField * residual_field);
      virtual bool write_generic_Hessian_contribution(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, GiNaC::ex for_what, bool hanging_eqns);
   };

   class DGFiniteElementSpace : public FiniteElementSpace
   {
   protected:
      FiniteElementSpace *conti_space;

   public:
      FiniteElementSpace *get_corresponding_continuous_space() { return conti_space; }
      virtual std::string get_shape_name() const { return "C" + name.substr(1); }
      virtual std::string get_hang_name() const { return "Discont"; }
      bool can_have_hanging_nodes() { return false; }
      DGFiniteElementSpace(FiniteElementCode *_code, const std::string &_name, FiniteElementSpace *_conti_space) : FiniteElementSpace(_code, _name), conti_space(_conti_space) {}
   };

   // DL and D0
   class DiscontinuousFiniteElementSpace : public FiniteElementSpace
   {
   public:
      bool can_have_hanging_nodes() { return false; }
      virtual std::string get_hang_name() const { return "Discont"; }
      DiscontinuousFiniteElementSpace(FiniteElementCode *_code, const std::string &_name) : FiniteElementSpace(_code, _name) {}
   };

   class D0BasisFunction : public BasisFunction
   {
   public:
      virtual std::string get_c_varname(FiniteElementCode *forcode, std::string test_index) { return "1"; }
      virtual std::string get_shape_string(FiniteElementCode *forcode, std::string nodal_index) const { return "1"; }
      D0BasisFunction(FiniteElementSpace *_space) : BasisFunction(_space) {}
   };

   class D0FiniteElementSpace : public DiscontinuousFiniteElementSpace
   {
   public:
      virtual std::string get_num_nodes_str(FiniteElementCode *forcode) const;
      virtual bool is_basis_derivative_zero(BasisFunction *b, unsigned dir) { return true; } // All spatial derivatives are zero
      D0FiniteElementSpace(FiniteElementCode *_code, const std::string &_name) : DiscontinuousFiniteElementSpace(_code, _name)
      {
         if (Basis)
            delete Basis;
         Basis = new D0BasisFunction(this);
      }
      virtual bool need_interpolation_loop() { return false; }
      virtual void write_spatial_interpolation(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, std::set<ShapeExpansion> &required_shapeexps, bool including_nodal_diffs, bool for_hessian);
   };

   class ExternalD0Space : public virtual D0FiniteElementSpace
   {
   public:
      virtual bool is_external() { return true; }
      ExternalD0Space(FiniteElementCode *_code, const std::string &_name) : D0FiniteElementSpace(_code, _name) {}
   };

   class FiniteElementField
   {
   protected:
      std::string name;
      FiniteElementSpace *space;
      GiNaC::symbol symb;
      std::map<FiniteElementCode *, std::set<unsigned>> residual_contribution_for_code; // For each code, the residual indices for which this field has a contribution
      std::map<FiniteElementCode *, std::map<unsigned ,std::set<FiniteElementField*> >> jacobian_contribution_for_code; // For each code, the residual indices for which this field has a contribution
      FiniteElementField * defined_on_domain_equivalent=NULL; // If a field is already defined on a bulk domain, it is transferred to interfaces and corners. This goes to the top level, i.e. where it is defined
   public:
      double discontinuous_refinement_exponent = 0.0;
      bool no_jacobian_at_all; // used for Lagrangian entries
      double temporal_error_factor;
      std::map<std::string, GiNaC::ex> initial_condition;
      std::map<std::string, bool> degraded_start;
      GiNaC::ex Dirichlet_condition;
      bool Dirichlet_condition_set = false;
      bool Dirichlet_condition_pin_only = false;
      const GiNaC::symbol &get_symbol() const { return symb; }
      int index;
      std::string get_name() { return name; }
      virtual std::string get_nodal_index_str(FiniteElementCode *forcode) const;
      virtual std::string get_equation_str(FiniteElementCode *forcode, std::string index) const;
      FiniteElementSpace *get_space() { return space; }
      FiniteElementField(const std::string &_name, FiniteElementSpace *_space) : name(_name), space(_space), symb(_name), no_jacobian_at_all(false), temporal_error_factor(0) {}
      GiNaC::ex get_shape_expansion(bool no_jacobian = false, bool no_hessian = false)
      {
         auto se = ShapeExpansion(this, 0, space->get_basis());
         if (no_jacobian)
            se.no_jacobian = true;
         if (no_hessian)
            se.no_hessian = no_hessian;
         return 0 + GiNaC::GiNaCShapeExpansion(se);
      }
      GiNaC::ex get_test_function() { return 0 + GiNaC::GiNaCTestFunction(TestFunction(this, space->get_basis())); }
      virtual std::string get_hanginfo_str(FiniteElementCode *forcode) const;
      bool has_residual_contribution_for_code(FiniteElementCode *code,unsigned residual_index);
      bool has_jacobian_contribution_for_code(FiniteElementCode *code,unsigned residual_index, FiniteElementField *other);
      void mark_residual_contribution_for_code(FiniteElementCode *code,unsigned residual_index);
      void mark_jacobian_contribution_for_code(FiniteElementCode *code,unsigned residual_index, FiniteElementField *other);
      FiniteElementField * get_defined_on_domain_equivalent_field();
      void set_defined_on_domain_equivalent_field(FiniteElementField *equiv_field);
   };

   class FiniteElementCodeSubExpression
   {
   protected:
      GiNaC::ex expr;
      GiNaC::potential_real_symbol cvar;

   public:
      std::set<ShapeExpansion> req_fields;
      std::map<GiNaC::symbol, GiNaC::ex, GiNaC::ex_is_less> derivsyms;
      GiNaC::ex expr_subst;
      GiNaC::ex &get_expression() { return expr; }
      GiNaC::potential_real_symbol &get_cvar() { return cvar; }
      FiniteElementCodeSubExpression(const GiNaC::ex &expr_, const GiNaC::potential_real_symbol &cvar_, const std::set<ShapeExpansion> &req_fields_) : expr(expr_), cvar(cvar_), req_fields(req_fields_) {}
   };

   class FiniteElementFieldTagInfo
   {
   public:
      bool no_jacobian = false;
      bool no_hessian = false;
      int expansion_mode = 0;
   };

   class Problem;

   class FiniteElementCode
   {
   protected:
      unsigned residual_index;
      Problem *problem=NULL;
      std::vector<std::string> residual_names;
      std::vector<double> reference_pos_for_IC_and_DBC = {0, 0, 0, 0, 0, 0, 0}; // 0-2: x,y, 3: t, 4-6: nx,ny,nz
      Equations *equations;
      FiniteElementCode *bulk_code;                   // Code of the bulk element
      FiniteElementCode *opposite_interface_code;     // Code of the interface elements at the opposite side of the interface
      std::vector<FiniteElementSpace *> spaces;       // Spaces in descending complexity order
      std::vector<FiniteElementCode *> required_odes; // Codes of coupled ODEs

      std::vector<GiNaC::ex> residual;
      std::set<std::string> ignore_assemble_residuals; // E.g. for azimuthal eigenvalue matrices. Residual is not used => don't assemble
      //std::map<std::string,std::set<int> > remove_underived_modes; // If in the Jacobian still modes are present that are not derived from interpolated_... to shape_..., they are removed. They can appear in eigenderivatives
      std::map<std::string,int> derive_jacobian_by_expansion_mode; // Derive the Jacobian by the given expansion mode only
      std::set<std::string> ignore_dpsi_coord_diffs_in_jacobian_set; // Usually, if we derive df/dx=f^l*dpsi^l/dx on a moving mesh, we also get a contribution how dpsi^l/dx changes with the moving mesh. For eigenexpansions, this might be wrong
      std::map<std::string,int> derive_hessian_by_expansion_mode; // Derive the Jacobian by the given expansion mode only
      SpatialIntegralSymbol dx, dX, dx_unity;
      ElementSizeSymbol elemsize_Eulerian, elemsize_Lagrangian, elemsize_Eulerian_Cart, elemsize_Lagrangian_Cart;
      NodalDeltaSymbol nodal_delta;
      std::vector<SpatialIntegralSymbol> dx_derived, dx_derived_lshape2_for_Hessian;
      std::vector<std::vector<SpatialIntegralSymbol>> dx_derived2;
      std::vector<ElementSizeSymbol> elemsize_derived, elemsize_derived_lshape2_for_Hessian, elemsize_Cart_derived, elemsize_Cart_derived_lshape2_for_Hessian;
      std::vector<std::vector<ElementSizeSymbol>> elemsize_derived2, elemsize_Cart_derived2;

      bool geometric_jac_for_elemsize_has_spatial_deriv, geometric_jac_for_elemsize_has_second_spatial_deriv;

      FiniteElementSpace *name_to_space(std::string name);

      std::vector<FiniteElementSpace *> allspaces;
      std::vector<FiniteElementField *> myfields;
      std::set<FiniteElementField*> contributing_fields;
      int stage; // 0: we can register fields, 1: fields are registered (cannot add any more), but now we can add residuals

      unsigned nodal_dim, lagr_dim;
      CustomCoordinateSystem *coordinate_sys;
      GiNaC::indexed _x, _y, _z;
      std::vector<CustomMathExpressionBase *> cb_expressions;
      std::vector<CustomMultiReturnExpressionBase *> multi_ret_expressions;

      std::map<std::string, std::map<FiniteElementSpace *, std::map<std::string, bool>>> required_shapes;
      unsigned max_dt_order = 0;
      std::vector<GiNaC::ex> Z2_fluxes,Z2_fluxes_for_eigen;
      std::map<std::string, GiNaC::ex> integral_expressions;
      std::map<std::string, GiNaC::ex> integral_expression_units;

      std::map<std::string, GiNaC::ex> tracer_advection_terms;
      std::map<std::string, GiNaC::ex> tracer_advection_units;

      std::map<std::string, GiNaC::ex> local_expressions;
      std::map<std::string, GiNaC::ex> local_expression_units;

      std::map<std::string, GiNaC::ex> extremum_expressions;
      std::map<std::string, GiNaC::ex> extremum_expression_units;

      std::vector<std::string> nullified_bulk_residuals;
      unsigned integration_order = 0;
      std::vector<bool> extra_steady_routine = {false};     // Time steppings involving explicit dependence of the previous DoFs, e.g. MPT, TPZ etc, require an additional routine for steady solving
      std::vector<bool> has_hessian_contribution = {false}; // Which of the residuals have hessian contributions
      std::vector<std::string> IC_names;                    // Names of the initial conditions
      std::vector<bool> has_constant_mass_matrix_for_sure;
      
      virtual void write_code_initial_condition(std::ostream &os, unsigned int index, std::string name);
      virtual void write_code_Dirichlet_condition(std::ostream &os);
      virtual void write_code_integral_or_local_expressions(std::ostream &os, std::map<std::string, GiNaC::ex> &exprs, std::map<std::string, GiNaC::ex> &units, std::string funcname, std::string reqname, bool integrate);
      virtual void write_code_integral_expressions(std::ostream &os);
      virtual void write_code_tracer_advection(std::ostream &os);
      virtual void write_code_local_expressions(std::ostream &os);
      virtual void write_code_extremum_expressions(std::ostream &os);
      virtual void write_code_header(std::ostream &os);
      virtual void write_code_info(std::ostream &os);
      virtual void write_code_geometric_jacobian(std::ostream &os);
      virtual void write_code_get_z2_flux(std::ostream &os,bool for_eigen);
      virtual void check_for_external_ode_dependencies();
      virtual void write_code_multi_ret_call(std::ostream &os, std::string indent, GiNaC::ex for_what, unsigned i, std::set<int> *multi_return_calls_written = NULL, GiNaC::ex *invok = NULL);
      virtual GiNaC::ex write_code_subexpressions(std::ostream &os, std::string indent, GiNaC::ex for_what, const std::set<ShapeExpansion> &required_shapeexps, bool hessian);
      virtual GiNaC::ex expand_initial_or_Dirichlet(const std::string &fieldname, GiNaC::ex expression);
      virtual GiNaC::ex extract_spatial_integral_part(const GiNaC::ex &inp, bool eulerian, bool lagrangian);

   public:
      void add_contributing_field(FiniteElementField *field) { contributing_fields.insert(field); }
      unsigned get_current_residual_index() const { return residual_index; }
      virtual void mark_nonconstant_mass_matrix() {has_constant_mass_matrix_for_sure[residual_index]=false;}
      virtual void set_reference_point_for_IC_and_DBC(double x, double y, double z, double t, double nx, double ny, double nz)
      {
         reference_pos_for_IC_and_DBC[0] = x;
         reference_pos_for_IC_and_DBC[1] = y;
         reference_pos_for_IC_and_DBC[2] = z;
         reference_pos_for_IC_and_DBC[3] = t;
         reference_pos_for_IC_and_DBC[4] = nx;
         reference_pos_for_IC_and_DBC[5] = ny;
         reference_pos_for_IC_and_DBC[6] = nz;
      }
      GiNaC::archive archive;
      std::map<std::string, GiNaC::ex> expanded_scales;
      GiNaC::ex expand_placeholders(GiNaC::ex inp, std::string where, bool raise_error = true);
      // To prevent tons of Python callbacks in e.g. UNIFAC to substitute molefraction by subexpressions, we cache the expanded callbacks
      std::map<std::tuple<std::string, const bool, const GiNaC::ex, FiniteElementCode *, bool, bool, std::string>, GiNaC::ex> expanded_additional_field_cache;
      virtual void set_equations(Equations *eqs) { equations = eqs; }
      virtual Equations *get_equations() { return equations; }
      virtual void index_fields();
      virtual void _activate_residual(std::string name);
      virtual void debug_second_order_Hessian_deriv(GiNaC::ex inp, std::string dx1, std::string dx2);
      const SpatialIntegralSymbol &get_dx_derived(int dir);
      const SpatialIntegralSymbol &get_dx_derived2(int dir, int dir2) { return dx_derived2[dir][dir2]; }
      const ElementSizeSymbol &get_elemsize_derived(int dir, bool _consider_coordsys);
      const ElementSizeSymbol &get_elemsize_derived2(int dir, int dir2, bool _consider_coordsys) { return (_consider_coordsys ? elemsize_derived2[dir][dir2] : elemsize_Cart_derived2[dir][dir2]); }
      const std::vector<FiniteElementSpace *> get_all_spaces() { return allspaces; }
      std::set<FiniteElementField *> get_fields_on_space(FiniteElementSpace *space);
      PositionFiniteElementSpace *get_my_position_space();
      void find_all_accessible_spaces();
      FiniteElementCodeSubExpression *resolve_subexpression(const GiNaC::ex &e);
      int resolve_multi_return_call(const GiNaC::ex &invok);
      int element_dim;
      bool analytical_jacobian;
      bool analytical_position_jacobian;
      double debug_jacobian_epsilon;
      bool with_adaptivity;
      bool coordinates_as_dofs;
      bool generate_hessian, assemble_hessian_by_symmetry;
      std::string coordinate_space;
      bool stop_on_jacobian_difference;
      std::string ccode_expression_mode = ""; // Mode to write expressions
      std::map<unsigned, unsigned> global_parameter_to_local_indices;
      std::vector<std::vector<bool>> local_parameter_has_deriv;
      std::vector<GiNaC::ex> local_parameter_symbols;
      std::vector<FiniteElementCodeSubExpression> subexpressions;
      std::vector<GiNaC::ex> multi_return_calls;
      std::map<CustomMultiReturnExpressionBase *, std::pair<unsigned, std::string>> multi_return_ccodes;
      void set_integration_order(unsigned order) { integration_order = order; }
      int get_integration_order() { return integration_order; }
      virtual GiNaC::ex eval_flag(std::string flagname);
      virtual void set_bulk_element(FiniteElementCode *_bulk_code) { bulk_code = _bulk_code; }
      virtual FiniteElementCode *get_bulk_element() { return bulk_code; }

      virtual void set_opposite_interface_code(FiniteElementCode *_opposite_interface_code) { opposite_interface_code = _opposite_interface_code; }
      virtual FiniteElementCode *get_opposite_interface_code() { return opposite_interface_code; }

      virtual FiniteElementCode *_resolve_based_on_domain_name(std::string name) { return NULL; }

      virtual std::string get_shapes_required_string(std::string func_type, FiniteElementSpace *space, std::string dx_type);
      virtual void write_required_shapes(std::ostream &os, const std::string indent, std::string func_type);
      virtual void mark_further_required_fields(GiNaC::ex expr, const std::string &for_what);
      virtual void mark_shapes_required(std::string func_type, FiniteElementSpace *space, std::string dx_type);
      virtual void mark_shapes_required(std::string func_type, FiniteElementSpace *space, BasisFunction *bf);
      virtual GiNaC::ex get_scaling(std::string name, bool testscale = false) { return 1; }

      virtual void add_Z2_flux(GiNaC::ex flux,bool for_eigen);
      virtual int get_dimension() const { return element_dim; }
      void set_nodal_dimension(unsigned d) { nodal_dim = d; }
      unsigned nodal_dimension() const { return nodal_dim; }

      void set_lagrangian_dimension(unsigned d) { lagr_dim = d; }
      unsigned lagrangian_dimension() const { return lagr_dim; }

      void nullify_bulk_residual(std::string dofname);

      virtual CustomCoordinateSystem *get_coordinate_system() { return coordinate_sys; } // To be overloaded to get it from the element as well

      FiniteElementCode();
      virtual ~FiniteElementCode();

      std::set<ShapeExpansion> get_all_shape_expansions_in(GiNaC::ex inp, bool merge_no_jacobian = true, bool merge_expansion_modes = true, bool merge_no_hessian = true);
      std::set<TestFunction> get_all_test_functions_in(GiNaC::ex inp);

      void fill_callback_info(JITFuncSpec_Table_FiniteElement_t *ft);

      virtual std::vector<std::string> register_integral_function(std::string name, GiNaC::ex expr);
      virtual GiNaC::ex get_integral_expression_unit_factor(std::string name);
      virtual std::vector<std::string> get_integral_expressions();

      virtual void set_tracer_advection_velocity(std::string name, GiNaC::ex expr);

      virtual std::pair<std::vector<std::string>, int> register_local_expression(std::string name, GiNaC::ex expr);
      virtual std::vector<std::string> get_local_expressions();
      virtual GiNaC::ex get_local_expression_unit_factor(std::string name);

      virtual void register_extremum_expression(std::string name, GiNaC::ex expr);
      virtual std::vector<std::string> get_extremum_expressions();
      virtual GiNaC::ex get_extremum_expression_unit_factor(std::string name);

      virtual void set_temporal_error(std::string f, double factor);
      // This will resolve the code (either itself, or bulk/otherbulk, external), func=field,nondimfield,scale,testfunction
      virtual FiniteElementCode *resolve_corresponding_code(GiNaC::ex func, std::string *fname, FiniteElementFieldTagInfo *taginfo);
      void set_problem(Problem * p);
	   Problem * get_problem();

      FiniteElementField *get_field_by_name(std::string name);
      FiniteElementField *register_field(std::string name, std::string spacename);
      GiNaC::ex expand_all_and_ensure_nondimensional(GiNaC::ex what, std::string where, GiNaC::ex *collected_units_and_factor = NULL);
      virtual void add_residual(GiNaC::ex add, bool allow_contributions_without_dx);
      virtual void write_generic_spatial_integration_header(std::ostream &os, std::string indent, GiNaC::ex eulerian_part, GiNaC::ex lagrangian_part, std::string required_table_and_flag);
      virtual void write_generic_spatial_integration_footer(std::ostream &os, std::string indent);
      virtual void write_generic_nodal_delta_header(std::ostream &os, std::string indent);
      virtual void write_generic_nodal_delta_footer(std::ostream &os, std::string indent);
      virtual void write_generic_RJM(std::ostream &os, std::string funcname, GiNaC::ex resi, bool with_hang);     // Generic Residual/Jacobian/Mass matrix (also for parameter derivatives)
      virtual bool write_generic_Hessian(std::ostream &os, std::string funcname, GiNaC::ex resi, bool with_hang); // Generic Hessian vector product
      virtual void write_code(std::ostream &os);
      virtual GiNaC::ex get_dx(bool lagrangian,bool unity_only=false);      
      virtual GiNaC::ex get_element_size_symbol(bool lagrangian, bool with_coordsys);
      virtual GiNaC::ex get_integral_dx(bool use_scaling, bool lagrangian, CustomCoordinateSystem *coordsys) { return get_dx(lagrangian); }
      virtual GiNaC::ex get_element_size(bool use_scaling, bool lagrangian, bool with_coordsys, CustomCoordinateSystem *coordsys) { return get_element_size_symbol(lagrangian, with_coordsys); }
      virtual GiNaC::ex get_nodal_delta();
      virtual GiNaC::ex get_normal_component(unsigned i);
      //virtual GiNaC::ex get_normal_component_eigenexpansion(unsigned i); // Used for azimuthal eigenstab only. Gives dn_i/dX^{0l}_j * X^{ml}_j
      virtual void set_derive_jacobian_by_expansion_mode(std::string residual_name,int expansion_mode) { derive_jacobian_by_expansion_mode[residual_name]=expansion_mode; }      
      virtual void set_derive_hessian_by_expansion_mode(std::string residual_name,int expansion_mode) { derive_hessian_by_expansion_mode[residual_name]=expansion_mode; }            
      virtual void set_ignore_dpsi_coord_diffs_in_jacobian(std::string residual_name) { ignore_dpsi_coord_diffs_in_jacobian_set.insert(residual_name); }
      virtual void set_ignore_residual_assembly(std::string residual_name) { ignore_assemble_residuals.insert(residual_name); }   
      virtual bool is_current_residual_assembly_ignored() { return ignore_assemble_residuals.count(residual_names[residual_index]); }
      virtual int * get_derive_jacobian_by_expansion_mode() { if (!derive_jacobian_by_expansion_mode.count(residual_names[residual_index])) return NULL; else return &(derive_jacobian_by_expansion_mode[residual_names[residual_index]]) ; }      
      virtual int * get_derive_hessian_by_expansion_mode() { if (!derive_hessian_by_expansion_mode.count(residual_names[residual_index])) return NULL; else return &(derive_hessian_by_expansion_mode[residual_names[residual_index]]) ; }            
      virtual bool ignore_dpsi_coord_diffs_in_jacobian() { return ignore_dpsi_coord_diffs_in_jacobian_set.count(residual_names[residual_index]); }

      virtual int classify_space_type(const FiniteElementSpace *s); // Returns 0 if the space is defined on this element, -1 for bulk element, -2 for other side of interface, >0 for external elements [-1]
      virtual std::string get_owner_prefix(const FiniteElementSpace *sp);
      virtual std::string get_shape_info_str(const FiniteElementSpace *sp);
      virtual std::string get_elem_info_str(const FiniteElementSpace *sp);
      virtual std::string get_nodal_data_string(const FiniteElementSpace *sp);
      virtual void finalise();
      virtual void _do_define_fields(int element_dimension);
      virtual GiNaC::ex expand_additional_field(const std::string &name, const bool &dimensional, const GiNaC::ex &expr, FiniteElementCode *in_domain, bool no_jacobian, bool no_hessian, std::string where) { return expr; }
      virtual GiNaC::ex expand_additional_testfunction(const std::string &name, const GiNaC::ex &expr, FiniteElementCode *in_domain) { return expr; }

      virtual std::string get_default_timestepping_scheme(unsigned int dt_order) { return (dt_order == 1 ? "BDF2" : "Newmark2"); }
      virtual unsigned get_default_spatial_integration_order() { return 0; }
      virtual void set_initial_condition(const std::string &name, GiNaC::ex expr, std::string degraded_start, const std::string &ic_name);
      virtual void set_Dirichlet_bc(const std::string &name, GiNaC::ex expr, bool use_identity);
      virtual void _define_element();
      virtual void _register_external_ode_linkage(std::string my_fieldname, FiniteElementCode *odecode, std::string odefieldname) {}

      virtual GiNaC::ex derive_expression(const GiNaC::ex &what, const GiNaC::ex by);

      virtual void _define_fields();
      virtual bool _is_ode_element() const { return false; }
      virtual std::string get_domain_name()
      {
         std::ostringstream oss;
         oss << this;
         return oss.str();
      }
      virtual std::string get_full_domain_name()
      {
         if (this->get_bulk_element()) return this->get_bulk_element()->get_full_domain_name()+"/"+this->get_domain_name();
         else return this->get_domain_name();
      }      
      virtual void set_discontinuous_refinement_exponent(std::string field, double exponent);
      double warn_on_large_numerical_factor = 0.0;
      bool bulk_position_space_to_C1 = false;
      bool use_shared_shape_buffer_during_multi_assemble = false;
      LaTeXPrinter *latex_printer;
      virtual void set_latex_printer(LaTeXPrinter *lp) { latex_printer = lp; }


      std::set<FiniteElementField *> Hessian_symmetric_fields_completed;
   };

   extern FiniteElementCode *__current_code;

}
