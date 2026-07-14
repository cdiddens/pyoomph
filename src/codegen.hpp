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
#include <set>
#include "expressions.hpp"
#include "jitbridge.h"

namespace pyoomph
{

   class FiniteElementCode;

   // Base class for the Python-side "Equations" objects (weak forms) that fill a FiniteElementCode.
   // _define_fields() registers the fields/spaces, _define_element() adds the residuals; both are
   // pure virtual and implemented in Python via pybind11 overrides.
   class Equations
   {
   protected:
      FiniteElementCode *current_codegen; // The FiniteElementCode currently being assembled by this Equations instance

   public:
      Equations() : current_codegen(NULL) {}
      virtual void _set_current_codegen(FiniteElementCode *cg) { current_codegen = cg; }
      virtual FiniteElementCode *_get_current_codegen() { return current_codegen; }
      virtual void _define_element() = 0;
      virtual void _define_fields() = 0;
   };

   class BasisFunction;
   class FiniteElementField;

   // GiNaC symbol (wrapped as GiNaCSpatialIntegralSymbol via PYGINACSTRUCT) representing the
   // integration measure "dx" (Eulerian) or "dX" (Lagrangian) of the element, i.e. J=sqrt(det(g)).
   // Also used, via the derived/derived2 flags, to represent the derivative of that measure with
   // respect to a nodal coordinate direction (needed for the Jacobian/Hessian of moving-mesh problems).
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
      bool is_derived() const { return derived; } // Is this the derivative w.r.t. a nodal coordinate (first order)?
      int get_derived_direction() const { return deriv_direction; }
      bool is_derived2() const { return derived2; } // Is this the second derivative (Hessian) w.r.t. two nodal coordinates?
      int get_derived_direction2() const { return deriv_direction2; }
      bool is_derived_by_lshape2() const { return derived_by_second_index; }
      const FiniteElementCode *get_code() const { return code; }
      // Plain dx/dX (not derived)
      SpatialIntegralSymbol(FiniteElementCode *_code, bool _lagrangian) : code(_code), lagrangian(_lagrangian), derived(false), derived2(false), derived_by_second_index(false), deriv_direction(0), deriv_direction2(0) {}
      // First derivative w.r.t. nodal coordinate "direction"
      SpatialIntegralSymbol(FiniteElementCode *_code, bool _lagrangian, int direction) : code(_code), lagrangian(_lagrangian), derived(true), derived2(false), derived_by_second_index(false), deriv_direction(direction), deriv_direction2(0) {}
      // First derivative, but taken w.r.t. the second shape-function loop index (used when assembling the Hessian)
      SpatialIntegralSymbol(FiniteElementCode *_code, bool _lagrangian, int direction, std::string second_index_dummy) : code(_code), lagrangian(_lagrangian), derived(true), derived2(false), derived_by_second_index(true), deriv_direction(direction), deriv_direction2(0) {}
      // Second (mixed) derivative w.r.t. nodal coordinates "direction" and "direction2"
      SpatialIntegralSymbol(FiniteElementCode *_code, bool _lagrangian, int direction, int direction2) : code(_code), lagrangian(_lagrangian), derived(true), derived2(true), derived_by_second_index(false), deriv_direction(direction), deriv_direction2(direction2) {}
   };

   // Ordering/equality used to store SpatialIntegralSymbol in std::set/std::map (e.g. for caching per-code instances)
   bool operator==(const SpatialIntegralSymbol &lhs, const SpatialIntegralSymbol &rhs);
   bool operator<(const SpatialIntegralSymbol &lhs, const SpatialIntegralSymbol &rhs);

   // Analogous to SpatialIntegralSymbol, but represents the element size h (not the pointwise Jacobian dx),
   // i.e. the symbol used e.g. for stabilization terms that require a characteristic element length.
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

   // Symbol representing a Dirac-delta-like nodal weighting (used for point-source-type contributions
   // evaluated at the nodes rather than integrated over the element). Carries no extra state besides
   // the owning code, since (unlike the symbols above) it has no derived/direction variants.
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

   // Symbol representing a component of the outward unit normal vector at the current integration/evaluation
   // point (only meaningful on interfaces/boundaries). Like the symbols above, also supports first/second
   // derivatives w.r.t. nodal coordinates for the Jacobian/Hessian of moving-mesh/interface problems.
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

   // Symbolic representation of a field's finite-element interpolation, i.e. sum_i u_i(t)*phi_i(x),
   // where phi_i is given by "basis" and u_i are the nodal/internal degrees of freedom of "field".
   // Also doubles as the "derived" (bare, un-summed) shape function phi_i itself when is_derived is
   // set, which is what appears in individual Jacobian/Hessian entries. dt_order/dt_scheme select a
   // time derivative of a given order/scheme (e.g. for time-dependent problems), and the
   // nodal_coord_dir(2) fields select derivatives w.r.t. moving-mesh nodal coordinates.
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

      // Additionally distinguishes derivatives taken w.r.t. the second shape-function loop index (used for the Hessian)
      ShapeExpansion(FiniteElementField *_field, unsigned _dt_order, BasisFunction *_basis, std::string ts, bool _is_derived, int _nodal_coord_dir, bool _is_derived_other_index) : field(_field), dt_order(_dt_order), dt_scheme(ts), basis(_basis), is_derived(_is_derived), is_derived_other_index(_is_derived_other_index), nodal_coord_dir(_nodal_coord_dir), nodal_coord_dir2(-1), time_history_index(0), no_jacobian(false), no_hessian(false), expansion_mode(0) {}

      // Full form including the second nodal coordinate direction, for mixed second-order (Hessian) nodal-coordinate derivatives
      ShapeExpansion(FiniteElementField *_field, unsigned _dt_order, BasisFunction *_basis, std::string ts, bool _is_derived, int _nodal_coord_dir, bool _is_derived_other_index, int _nodal_coord_dir2) : field(_field), dt_order(_dt_order), dt_scheme(ts), basis(_basis), is_derived(_is_derived), is_derived_other_index(_is_derived_other_index), nodal_coord_dir(_nodal_coord_dir), nodal_coord_dir2(_nodal_coord_dir2), time_history_index(0), no_jacobian(false), no_hessian(false), expansion_mode(0) {}

      virtual std::string get_dt_values_name(FiniteElementCode *forcode) const;
      virtual std::string get_timedisc_scheme(FiniteElementCode *forcode) const;
      virtual std::string get_spatial_interpolation_name(FiniteElementCode *forcode) const;
      virtual std::string get_num_nodes_str(FiniteElementCode *forcode) const;
      virtual std::string get_nodal_index_str(FiniteElementCode *forcode) const;
      virtual std::string get_nodal_data_string(FiniteElementCode *forcode, std::string indexstr) const;
      virtual std::string get_shape_string(FiniteElementCode *forcode, std::string nodal_index) const;
      // Checks whether symbol s is the nodal-position derivative symbol matching this shape expansion
      // (used to detect Jacobian entries w.r.t. moving-mesh coordinates); returns the code it belongs to, or NULL
      virtual FiniteElementCode *can_be_a_positional_derivative_symbol(const GiNaC::symbol &s, FiniteElementCode *domain_to_check = NULL) const;
   };

   bool operator==(const ShapeExpansion &lhs, const ShapeExpansion &rhs);
   bool operator<(const ShapeExpansion &lhs, const ShapeExpansion &rhs);

   // Symbolic representation of a weak-form test function (the "phi_i" multiplying the residual before
   // integration), analogous to ShapeExpansion but without a time-history/nodal-value sum attached.
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
   // Lightweight GiNaC-wrapped handle to a common-subexpression-eliminated (CSE) expression: just
   // pairs the owning code with the underlying GiNaC expression. The actual bookkeeping (substitution
   // symbol, required fields, etc.) lives in FiniteElementCodeSubExpression, looked up via resolve_subexpression().
   class SubExpression
   {
   public:
      FiniteElementCode *code;
      GiNaC::ex expr; // Expression
      SubExpression(FiniteElementCode *c, const GiNaC::ex &e) : code(c), expr(e) {}
   };

   // Represents one return value of an invocation of a Python-defined multi-return callback
   // (CustomMultiReturnExpressionBase), i.e. a C function call with several outputs. "invok" is the
   // full call (function + args), "retindex" selects which output this expression stands for, and
   // "derived_by_arg" (if >=0) selects that this represents the derivative of that output w.r.t. the
   // given argument index instead of the value itself.
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
   // The PYGINACSTRUCT(...) macro (see pyginacstruct.hpp) wraps each pyoomph symbol struct above into a
   // GiNaC::structure<...> so it can be embedded in GiNaC::ex expression trees. The specialized print()
   // methods define how each symbol is rendered to C++/LaTeX source (see print_csrc_FEM/print_latex_FEM
   // below), and the specialized derivative() methods define symbolic differentiation rules used when
   // building Jacobian/Hessian expressions.

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
      pyoomph::FiniteElementCode *for_code; // Code being printed for; gives access to spaces/fields/etc. needed to render symbols
      bool in_subexpr_deriv; // True while printing the derivative expression of a subexpression (affects how subexpressions are re-expanded)
      bool ignore_custom;    // If true, custom (Python callback) expressions are not substituted/expanded during printing
      print_FEM_options() : for_code(NULL), in_subexpr_deriv(false), ignore_custom(false) {}
   };

   // GiNaC print context that renders an expression tree as C++ source code (for the generated JIT element code)
   class print_csrc_FEM : public GiNaC::print_csrc_double
   {
      //    GINAC_DECLARE_PRINT_CONTEXT(print_csrc_FEM, GiNaC::print_dflt)
   public:
      print_FEM_options *FEM_opts;
      print_csrc_FEM();
      print_csrc_FEM(std::ostream &, print_FEM_options *fem_opts, unsigned options = 0);
   };

   // GiNaC print context that renders an expression tree as LaTeX (used for the auto-generated equation documentation)
   class print_latex_FEM : public GiNaC::print_latex
   {
   public:
      print_FEM_options *FEM_opts;
      print_latex_FEM();
      print_latex_FEM(std::ostream &, print_FEM_options *fem_opts, unsigned options = 0);
   };

   // Helper hierarchy used by print_sorted_GiNaC()/print_simplest_form() (mode "deterministic") to
   // convert a GiNaC expression tree into a form with a fixed, reproducible term/factor ordering before
   // printing it as C++ source. This avoids nondeterministic reordering of sums/products between GiNaC
   // versions or runs, which would otherwise make generated code (and thus floating-point rounding)
   // vary from build to build. Each subclass mirrors one GiNaC expression kind (numeric, add, mul, pow,
   // function, symbol, generic struct) and knows how to print itself and where it ranks in add/mul sort order.
   class SortedGiNaC
    {
      public:
        std::vector<SortedGiNaC*> op; // Child nodes (e.g. summands, factors, function args), in original (pre-sort) order
        virtual ~SortedGiNaC() ;
        static SortedGiNaC * factory(const ex & e,std::ostream &os, GiNaC::print_FEM_options &csrc_opts); // Recursively builds a SortedGiNaC tree mirroring the given GiNaC expression
        virtual std::string to_string(std::ostream &os, GiNaC::print_FEM_options &csrc_opts) =0; // Renders this node (with its children sorted) as C++ source
        virtual int add_order()=0; // Rank of this node's kind when sorting terms of a sum
        virtual int mul_order()=0; // Rank of this node's kind when sorting factors of a product
        bool add_sort_compare(SortedGiNaC * other, std::ostream &os, GiNaC::print_FEM_options &csrc_opts); // Strict-weak-order comparison used to sort summands deterministically
        bool mul_sort_compare(SortedGiNaC * other, std::ostream &os, GiNaC::print_FEM_options &csrc_opts); // Strict-weak-order comparison used to sort factors deterministically
    };

    // Leaf node for a plain numeric literal
    class SortedGiNaCNumeric : public SortedGiNaC
    {
      public:
        GiNaC::numeric value;

        SortedGiNaCNumeric(GiNaC::numeric v) : SortedGiNaC(), value(v) {}
        virtual std::string to_string(std::ostream &os, GiNaC::print_FEM_options &csrc_opts) override;
        virtual int add_order() override { return 0; }
        virtual int mul_order() override { return 0; }
        
    };

    // Node for a sum (GiNaC::add); to_string() sorts its "op" summands via add_sort_compare() before printing
    class SortedGiNaCAdd : public SortedGiNaC
    {
      public:
        SortedGiNaCAdd(const std::vector<SortedGiNaC*> & ops) : SortedGiNaC()  {op=ops;}
        virtual std::string to_string(std::ostream &os, GiNaC::print_FEM_options &csrc_opts) override;        
        virtual int add_order() override;        
        virtual int mul_order() override { return 5; }
            
    };

    // Node for a product (GiNaC::mul); to_string() sorts its "op" factors via mul_sort_compare() before printing
    class SortedGiNaCMul : public SortedGiNaC
    {
      public:
        SortedGiNaCMul(const std::vector<SortedGiNaC*> & ops) : SortedGiNaC(){op=ops;}
        virtual std::string to_string(std::ostream &os, GiNaC::print_FEM_options &csrc_opts) override;        
        virtual int add_order() override { return 3; }
        virtual int mul_order() override;        
    };

    // Node for a power base^exp (GiNaC::power); op[0] is the base, op[1] the exponent
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

    // Node for a GiNaC function call (sin, cos, ...); "op" holds the (already sorted-tree) arguments
    class SortedGiNaCFunction : public SortedGiNaC
    {
      public:
        std::string fname;
        SortedGiNaCFunction(const std::string & fname, const std::vector<SortedGiNaC*> & ops) : SortedGiNaC(), fname(fname)     {            op=ops;        }
        virtual std::string to_string(std::ostream &os, GiNaC::print_FEM_options &csrc_opts) override;
        virtual int add_order() override { return 1; }
        virtual int mul_order() override { return 1; }
    };

    // Leaf node for a plain GiNaC::symbol, printed verbatim as its already-resolved C++ variable name
    class SortedGiNaCSymbol : public SortedGiNaC
    {
      public:
        std::string vname;
        SortedGiNaCSymbol(const std::string & vname) : SortedGiNaC(), vname(vname) {}
        virtual std::string to_string(std::ostream &os, GiNaC::print_FEM_options &csrc_opts) override {return vname;}
        virtual int add_order() override {return 2;}
        virtual int mul_order() override {return 2;}
    };

    // Catch-all leaf for any of the pyoomph GiNaCStruct-wrapped symbols (ShapeExpansion, TestFunction,
    // SpatialIntegralSymbol, ...) that aren't otherwise decomposed; delegates printing to their own print() specialization
    class SortedGiNaCStruct : public SortedGiNaC
    {
      public:
        ex contents;
        SortedGiNaCStruct(GiNaC::ex _contents) : SortedGiNaC(), contents(_contents) {}
        virtual std::string to_string(std::ostream &os, GiNaC::print_FEM_options &csrc_opts) override;
        virtual int add_order() override { return 6; }
        virtual int mul_order() override;
    };

    // Entry point: builds a SortedGiNaC tree for e (via SortedGiNaC::factory) and writes its deterministically-sorted C++ form to os
    std::ostream & print_sorted_GiNaC(ex  e,std::ostream &os, GiNaC::print_FEM_options &csrc_opts);

}

namespace pyoomph
{
   // Prints expr to os as C++ source, choosing the simplification/canonicalization strategy according
   // to csrc_opts.for_code->ccode_expression_mode (e.g. "deterministic" uses print_sorted_GiNaC() for
   // reproducible output, others apply GiNaC::factor/normal/expand/collect_common_factors first).
   // Also archives expr (for later inspection/serialization) via for_code->archive.
   void print_simplest_form(GiNaC::ex expr, std::ostream &os, GiNaC::print_FEM_options &csrc_opts);


   // Bridges generated LaTeX snippets (for equation documentation) back to Python; the actual storage
   // (e.g. writing to a .tex/.html file) is implemented by the Python subclass overriding _add_LaTeX_expression.
   class LaTeXPrinter
   {
   public:
      virtual void _add_LaTeX_expression(std::map<std::string, std::string> info, std::string expr, FiniteElementCode *code) {}  // Will be implemented by python
      virtual std::string _get_LaTeX_expression(std::map<std::string, std::string> info, FiniteElementCode *code) { return ""; } // Will be implemented by python
      // Renders expression to LaTeX and forwards it (together with the free-form "info" tags) to the Python-side sink
      void print(std::map<std::string, std::string> info, const GiNaC::ex &expression, GiNaC::print_FEM_options &ops)
      {
         std::ostringstream oss;
         expression.eval().print(GiNaC::print_latex_FEM(oss, &ops));
         this->_add_LaTeX_expression(info, oss.str(), ops.for_code);
      }
   };

   // GiNaC::map_function that descends into expressions::subexpression(...) wrapped terms and pulls
   // their physical units out to the top level, leaving a purely nondimensional subexpression behind
   // (used so that CSE'd subexpressions can be substituted by a single nondimensional C variable).
   class DrawUnitsOutOfSubexpressions : public GiNaC::map_function
   {
   protected:
      FiniteElementCode *code;

   public:
      DrawUnitsOutOfSubexpressions(FiniteElementCode *code_) : code(code_) {}
      GiNaC::ex operator()(const GiNaC::ex &inp);
   };

   // GiNaC::map_function that strips expressions::subexpression(...) markers back out again (replacing
   // them by their contained expression), effectively undoing the common-subexpression tagging; also
   // re-registers any multi-return callback invocations it encounters with the owning code.
   class RemoveSubexpressionsByIndentity : public GiNaC::map_function
   {
   protected:
      FiniteElementCode *code;

   public:
      RemoveSubexpressionsByIndentity(FiniteElementCode *code_) : code(code_) {}
      GiNaC::ex operator()(const GiNaC::ex &inp);
   };

   // GiNaC::map_function that resolves eval_in_domain(...)/field placeholders to the concrete,
   // nondimensionalized field expressions of the resolved domain code, tracking the extra test-function
   // scale factor picked up along the way (e.g. from facet/domain-side tags) and how many replacements were made.
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

   // GiNaC::map_function that replaces "derived" (bare, un-summed) ShapeExpansions by 1, optionally only
   // for a specific basis function / time-derivative order+scheme (others become 0). Used to turn a
   // symbolic Jacobian/Hessian entry expression (linear in the derived shape functions) into the
   // residual-independent coefficient that multiplies a given trial/test function pair.
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
                 if ((int)shp.dt_order!=this->ensure_dt_order) return 0;
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
   // Represents the (undifferentiated) shape function phi of a FiniteElementSpace. Subclasses
   // (D1XBasisFunction and its descendants) represent spatial derivatives of phi instead; the
   // get_diff_x/X/S() methods lazily create and cache those derivative BasisFunction instances
   // (in basis_deriv_x/lagr_deriv_x/local_coord_deriv_x) so each distinct derivative is only built once.
   class BasisFunction
   {
   protected:
      FiniteElementSpace *space;
      std::vector<BasisFunction *> basis_deriv_x, lagr_deriv_x, local_coord_deriv_x; // Cached derivative BasisFunctions, indexed by spatial direction: Eulerian (x), Lagrangian (X), local/reference (S)

   public:
      BasisFunction(FiniteElementSpace *_space) : space(_space) {}
      virtual ~BasisFunction();
      virtual BasisFunction *get_diff_x(unsigned direction); // Eulerian spatial derivative d(phi)/dx_direction
      virtual BasisFunction *get_diff_X(unsigned direction); // Lagrangian spatial derivative d(phi)/dX_direction
      virtual BasisFunction *get_diff_S(unsigned direction); // Local/reference-coordinate derivative d(phi)/dS_direction
      virtual std::string to_string();
      virtual const FiniteElementSpace *get_space() const { return space; }
      virtual std::string get_dx_str() const { return "d0x"; } // Suffix identifying which (if any) derivative table this basis function reads from in the generated code
      virtual std::string get_shape_string(FiniteElementCode *forcode, std::string nodal_index) const; // C++ expression for this basis function's shape value/derivative at the given nodal index
      virtual std::string get_c_varname(FiniteElementCode *forcode, std::string test_index);
   };

   // First Eulerian spatial derivative of a basis function, d(phi)/dx_direction
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

   // First Lagrangian (material/reference-configuration) spatial derivative of a basis function, d(phi)/dX_direction
   class D1XBasisFunctionLagr : public D1XBasisFunction
   {
   public:
      D1XBasisFunctionLagr(FiniteElementSpace *_space, unsigned _direction) : D1XBasisFunction(_space, _direction) {}
      virtual std::string to_string();
      virtual std::string get_dx_str() const { return "d1X" + std::to_string(direction); }
      virtual std::string get_shape_string(FiniteElementCode *forcode, std::string nodal_index) const;
      virtual std::string get_c_varname(FiniteElementCode *forcode, std::string test_index);
   };

   // First local/reference-element-coordinate derivative of a basis function, d(phi)/dS_direction
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
   // Base class for a finite-element interpolation space (e.g. C1/C2 continuous nodal spaces, the
   // position space, discontinuous/DG spaces, D0). Owns the (undifferentiated) BasisFunction for this
   // space and knows how to emit the C++ code that interpolates fields on it (spatial/time
   // interpolation, and the residual/Jacobian/Hessian contribution loops over its shape functions).
   class FiniteElementSpace
   {
   protected:
      FiniteElementCode *code;
      std::string name;
      BasisFunction *Basis;

   public:
      virtual std::string get_eqn_number_str(FiniteElementCode *forcode) const;
      virtual bool is_external() { return false; } // True for spaces belonging to an externally coupled element (e.g. external data/ODE)
      virtual bool is_basis_derivative_zero(BasisFunction *b, unsigned dir) { return false; } // True if the spatial derivative of b in direction dir is identically zero (e.g. piecewise-constant D0 spaces)
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
      virtual std::string get_shape_name() const { return name; } // Name used to look up the shape-function table in the generated code
      virtual bool can_have_hanging_nodes() { return false; }
      virtual bool need_interpolation_loop() { return true; } // False for spaces (e.g. D0) where a single value replaces the usual per-node interpolation loop
      FiniteElementCode *get_code() const { return code; }
   };

   // A standard continuous (C1/C2-type) nodal Lagrange space, i.e. the usual "H1-conforming" FE space that can have hanging nodes on refined meshes
   class ContinuousFiniteElementSpace : public FiniteElementSpace
   {
   public:
      bool can_have_hanging_nodes() override;
      ContinuousFiniteElementSpace(FiniteElementCode *_code, const std::string &_name) : FiniteElementSpace(_code, _name) {}
   };

   // The special continuous space that carries the element's own (Eulerian) nodal position/coordinates,
   // as opposed to a regular unknown field; the Jacobian/Hessian contributions w.r.t. this space are
   // the moving-mesh (shape-derivative) terms and are therefore handled by dedicated overrides.
   class PositionFiniteElementSpace : public ContinuousFiniteElementSpace
   {
   public:
      virtual std::string get_num_nodes_str(FiniteElementCode *forcode) const;
      virtual std::string get_eqn_number_str(FiniteElementCode *forcode) const;
      PositionFiniteElementSpace(FiniteElementCode *_code, const std::string &_name) : ContinuousFiniteElementSpace(_code, _name) {}
      virtual void write_generic_RJM_jacobian_contribution(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, GiNaC::ex for_what, bool hanging_eqns,FiniteElementField * residual_field);
      virtual bool write_generic_Hessian_contribution(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, GiNaC::ex for_what, bool hanging_eqns);
   };

   // Discontinuous-Galerkin space; keeps a pointer to its "shadow" continuous counterpart space of the
   // same polynomial order (conti_space), which is used e.g. to look up shared geometric shape data.
   class DGFiniteElementSpace : public FiniteElementSpace
   {
   protected:
      FiniteElementSpace *conti_space;

   public:
      FiniteElementSpace *get_corresponding_continuous_space() { return conti_space; }
      virtual std::string get_shape_name() const { return "C" + name.substr(1); }
      bool can_have_hanging_nodes() { return false; }
      DGFiniteElementSpace(FiniteElementCode *_code, const std::string &_name, FiniteElementSpace *_conti_space) : FiniteElementSpace(_code, _name), conti_space(_conti_space) {}
   };

   // DL and D0
   // Base for spaces with independent, non-shared degrees of freedom per element (DL: discontinuous Lagrange, D0: piecewise constant); never has hanging nodes
   class DiscontinuousFiniteElementSpace : public FiniteElementSpace
   {
   public:
      bool can_have_hanging_nodes() { return false; }
      DiscontinuousFiniteElementSpace(FiniteElementCode *_code, const std::string &_name) : FiniteElementSpace(_code, _name) {}
   };

   // Basis function of a D0 (piecewise-constant) space: the shape value is simply the constant 1, and (via
   // FiniteElementSpace::is_basis_derivative_zero) all its spatial derivatives are zero
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

   // A D0 space whose single value lives on an externally coupled element rather than on this element itself
   class ExternalD0Space : public virtual D0FiniteElementSpace
   {
   public:
      virtual bool is_external() { return true; }
      ExternalD0Space(FiniteElementCode *_code, const std::string &_name) : D0FiniteElementSpace(_code, _name) {}
   };

   // A named unknown (or position) field defined on a FiniteElementSpace. Provides the symbolic
   // building blocks (via get_shape_expansion()/get_test_function()) used to write weak-form residuals,
   // and tracks, per FiniteElementCode/residual, whether this field actually contributes to the
   // residual/Jacobian there (so unnecessary shape/derivative tables can be skipped in codegen).
   class FiniteElementField
   {
   protected:
      std::string name;
      FiniteElementSpace *space;
      GiNaC::symbol symb; // Plain GiNaC symbol identifying this field, used e.g. as a key in nondimensionalization/substitution maps
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
      int index; // Position of this field's degree of freedom within its space's per-node data
      std::string get_name() { return name; }
      virtual std::string get_nodal_index_str(FiniteElementCode *forcode) const;
      virtual std::string get_equation_str(FiniteElementCode *forcode, std::string index) const;
      FiniteElementSpace *get_space() { return space; }
      FiniteElementField(const std::string &_name, FiniteElementSpace *_space) : name(_name), space(_space), symb(_name), no_jacobian_at_all(false), temporal_error_factor(0) {}
      // Symbolic sum_i u_i(t)*phi_i(x) representing this field's FE interpolation (see ShapeExpansion)
      GiNaC::ex get_shape_expansion(bool no_jacobian = false, bool no_hessian = false)
      {
         auto se = ShapeExpansion(this, 0, space->get_basis());
         if (no_jacobian)
            se.no_jacobian = true;
         if (no_hessian)
            se.no_hessian = no_hessian;
         return 0 + GiNaC::GiNaCShapeExpansion(se);
      }
      // Symbolic weak-form test function phi_i associated with this field (see TestFunction)
      GiNaC::ex get_test_function() { return 0 + GiNaC::GiNaCTestFunction(TestFunction(this, space->get_basis())); }
      virtual std::string get_hanginfo_str(FiniteElementCode *forcode) const;
      bool has_residual_contribution_for_code(FiniteElementCode *code,unsigned residual_index);
      bool has_jacobian_contribution_for_code(FiniteElementCode *code,unsigned residual_index, FiniteElementField *other);
      void mark_residual_contribution_for_code(FiniteElementCode *code,unsigned residual_index);
      void mark_jacobian_contribution_for_code(FiniteElementCode *code,unsigned residual_index, FiniteElementField *other);
      FiniteElementField * get_defined_on_domain_equivalent_field();
      void set_defined_on_domain_equivalent_field(FiniteElementField *equiv_field);
   };

   // Bookkeeping entry for one common-subexpression-eliminated (CSE) piece of a residual/Jacobian
   // expression: the original expression, the C++ variable ("cvar") it gets substituted by, which
   // shape expansions it depends on (req_fields, used to know when it must be recomputed), and the
   // derivatives of the substituted variable w.r.t. other symbols (derivsyms), cached to avoid
   // re-differentiating the (possibly expensive) original expression repeatedly.
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

   // Small flag bundle passed around while resolving/expanding a field reference (e.g. via
   // resolve_corresponding_code()), carrying whether Jacobian/Hessian contributions should be
   // suppressed for it and which eigen-expansion mode it belongs to.
   class FiniteElementFieldTagInfo
   {
   public:
      bool no_jacobian = false;
      bool no_hessian = false;
      int expansion_mode = 0;
   };

   class Problem;

   // The central code-generation object: one FiniteElementCode instance corresponds to one compiled
   // "element type" (bulk element, interface, ODE, ...) as defined by a Python Equations object. It
   // collects the fields/spaces/residuals defined via the various register_*()/add_residual() calls,
   // then write_code() drives symbolic differentiation (via GiNaC) of the residuals to produce the C++
   // source of the JIT-compiled element (residual, analytic Jacobian, mass matrix, and optionally
   // Hessian vector products), which is what actually gets compiled and loaded by the C compiler
   // backend (see ccompiler.hpp) and executed by the oomph-lib element machinery (see elements.hpp).
   // Python subclasses this (see pybind/codegen.cpp's PyFiniteElementCode) to hook in domain-specific
   // callbacks such as expand_additional_field.
   class FiniteElementCode
   {
   protected:
      unsigned residual_index; // Index of the residual currently being assembled/derived (into residual/residual_names and the various per-residual vectors below)
      Problem *problem=NULL;
      std::vector<std::string> residual_names;
      std::vector<double> reference_pos_for_IC_and_DBC = {0, 0, 0, 0, 0, 0, 0}; // 0-2: x,y, 3: t, 4-6: nx,ny,nz
      Equations *equations;
      FiniteElementCode *bulk_code;                   // Code of the bulk element
      FiniteElementCode *opposite_interface_code;     // Code of the interface elements at the opposite side of the interface
      std::vector<FiniteElementSpace *> spaces;       // Spaces in descending complexity order
      std::vector<FiniteElementCode *> required_odes; // Codes of coupled ODEs
      std::vector<GiNaC::ex> residual; // One weak-form residual expression per named residual (index by residual_index)
      std::set<std::string> ignore_assemble_residuals; // E.g. for azimuthal eigenvalue matrices. Residual is not used => don't assemble
      //std::map<std::string,std::set<int> > remove_underived_modes; // If in the Jacobian still modes are present that are not derived from interpolated_... to shape_..., they are removed. They can appear in eigenderivatives
      std::map<std::string,int> derive_jacobian_by_expansion_mode; // Derive the Jacobian by the given expansion mode only
      std::set<std::string> ignore_dpsi_coord_diffs_in_jacobian_set; // Usually, if we derive df/dx=f^l*dpsi^l/dx on a moving mesh, we also get a contribution how dpsi^l/dx changes with the moving mesh. For eigenexpansions, this might be wrong
      std::map<std::string,int> derive_hessian_by_expansion_mode; // Derive the Jacobian by the given expansion mode only
      SpatialIntegralSymbol dx, dX, dx_unity; // Cached "canonical" dx (Eulerian), dX (Lagrangian) and unity-Jacobian integration-measure symbols for this code, handed out by get_dx()
      ElementSizeSymbol elemsize_Eulerian, elemsize_Lagrangian, elemsize_Eulerian_Cart, elemsize_Lagrangian_Cart; // Cached element-size symbols (with/without coordinate-system weighting), handed out by get_element_size_symbol()
      NodalDeltaSymbol nodal_delta; // Cached nodal-delta symbol, handed out by get_nodal_delta()
      std::vector<SpatialIntegralSymbol> dx_derived, dx_derived_lshape2_for_Hessian; // Cache of first-derivative dx symbols, indexed by nodal coordinate direction, returned by get_dx_derived()
      std::vector<std::vector<SpatialIntegralSymbol>> dx_derived2; // Cache of second-derivative dx symbols, indexed by [dir][dir2], returned by get_dx_derived2()
      std::vector<ElementSizeSymbol> elemsize_derived, elemsize_derived_lshape2_for_Hessian, elemsize_Cart_derived, elemsize_Cart_derived_lshape2_for_Hessian; // Analogous first-derivative caches for the element-size symbol, see get_elemsize_derived()
      std::vector<std::vector<ElementSizeSymbol>> elemsize_derived2, elemsize_Cart_derived2; // Analogous second-derivative caches for the element-size symbol, see get_elemsize_derived2()

      bool geometric_jac_for_elemsize_has_spatial_deriv, geometric_jac_for_elemsize_has_second_spatial_deriv; // Whether write_code_geometric_jacobian() needs to emit the (more expensive) spatial-derivative terms of the element-size Jacobian

      FiniteElementSpace *name_to_space(std::string name); // Looks up an already-registered space by name (NULL if not found)

      std::vector<FiniteElementSpace *> allspaces; // All spaces reachable from this code (own + bulk/interface/external), populated by find_all_accessible_spaces()
      std::vector<FiniteElementField *> myfields; // Fields registered directly on this code (owns them)
      std::set<FiniteElementField*> contributing_fields; // Fields (possibly from other codes, e.g. bulk) that actually appear in this code's residuals
      int stage; // 0: we can register fields, 1: fields are registered (cannot add any more), but now we can add residuals

      unsigned nodal_dim, lagr_dim; // Number of Eulerian / Lagrangian coordinate dimensions of the nodes
      CustomCoordinateSystem *coordinate_sys;
      GiNaC::indexed _x, _y, _z; // Indexed placeholder symbols for the (interpolated) global coordinates, used while building coordinate-system-dependent expressions
      std::vector<CustomMathExpressionBase *> cb_expressions; // Python callback math expressions (single return value) referenced by this code, in order of first use
      std::vector<CustomMultiReturnExpressionBase *> multi_ret_expressions; // Python callback math expressions (multiple return values) referenced by this code, in order of first use

      std::map<std::string, std::map<FiniteElementSpace *, std::map<std::string, bool>>> required_shapes; // [func_type][space][shape/dx flavor] -> required; tracks which shape-function tables must be filled for which generated routine
      unsigned max_dt_order = 0; // Highest time-derivative order appearing in any residual of this code
      std::vector<GiNaC::ex> Z2_fluxes,Z2_fluxes_for_eigen; // Flux expressions used for the Zienkiewicz-Zhu (Z2) error estimator, registered via add_Z2_flux()
      std::map<std::string, GiNaC::ex> integral_expressions; // Named domain-integral output expressions, registered via register_integral_function()
      std::map<std::string, GiNaC::ex> integral_expression_units;

      std::map<std::string, GiNaC::ex> tracer_advection_terms; // Named advection-velocity expressions for tracer/particle advection, registered via set_tracer_advection_velocity()
      std::map<std::string, GiNaC::ex> tracer_advection_units;

      std::map<std::string, GiNaC::ex> local_expressions; // Named pointwise (non-integrated) output expressions, registered via register_local_expression()
      std::map<std::string, GiNaC::ex> local_expression_units;

      std::map<std::string, GiNaC::ex> extremum_expressions; // Named expressions whose extremum (over the domain) is tracked, registered via register_extremum_expression()
      std::map<std::string, GiNaC::ex> extremum_expression_units;

      std::vector<std::string> nullified_bulk_residuals; // Names of bulk residuals that this (interface/facet) code forces to zero, see nullify_bulk_residual()
      unsigned integration_order = 0;
      std::vector<bool> extra_steady_routine = {false};     // Time steppings involving explicit dependence of the previous DoFs, e.g. MPT, TPZ etc, require an additional routine for steady solving
      std::vector<bool> has_hessian_contribution = {false}; // Which of the residuals have hessian contributions
      std::vector<std::string> IC_names;                    // Names of the initial conditions
      std::vector<bool> has_constant_mass_matrix_for_sure; // Per-residual flag: true if the mass matrix is known a priori to be constant (enables solver optimizations); may be downgraded to false while writing the Hessian

      // The write_code_*() methods each emit one section/routine of the generated C++ element source
      // (invoked in sequence from write_code()); they operate on the already-built symbolic residual(s).
      virtual void write_code_initial_condition(std::ostream &os, unsigned int index, std::string name);
      virtual void write_code_Dirichlet_condition(std::ostream &os);
      virtual void write_code_integral_or_local_expressions(std::ostream &os, std::map<std::string, GiNaC::ex> &exprs, std::map<std::string, GiNaC::ex> &units, std::string funcname, std::string reqname, bool integrate); // Shared implementation behind write_code_integral/local/extremum_expressions()
      virtual void write_code_integral_expressions(std::ostream &os);
      virtual void write_code_tracer_advection(std::ostream &os);
      virtual void write_code_local_expressions(std::ostream &os);
      virtual void write_code_extremum_expressions(std::ostream &os);
      virtual void write_code_header(std::ostream &os);
      virtual void write_code_info(std::ostream &os);
      virtual void write_code_geometric_jacobian(std::ostream &os); // Emits the code computing d(dx)/d(nodal coordinate) etc. for moving-mesh Jacobian contributions
      virtual void write_code_get_z2_flux(std::ostream &os,bool for_eigen);
      virtual void check_for_external_ode_dependencies();
      virtual void write_code_multi_ret_call(std::ostream &os, std::string indent, GiNaC::ex for_what, unsigned i, std::set<int> *multi_return_calls_written = NULL, GiNaC::ex *invok = NULL); // Emits the C call to a multi-return Python callback and stores its outputs in local variables
      virtual GiNaC::ex write_code_subexpressions(std::ostream &os, std::string indent, GiNaC::ex for_what, const std::set<ShapeExpansion> &required_shapeexps, bool hessian); // Emits local-variable definitions for all CSE'd subexpressions occurring in for_what and returns the substituted expression
      virtual GiNaC::ex expand_initial_or_Dirichlet(const std::string &fieldname, GiNaC::ex expression);
      virtual GiNaC::ex extract_spatial_integral_part(const GiNaC::ex &inp, bool eulerian, bool lagrangian); // Splits off the dx/dX integration-measure factor from inp, returning the remaining integrand
      virtual void write_required_shapes_for_code(std::ostream & os, std::string func_type, std::string indent, FiniteElementCode *for_code, int type);
   public:
      void add_contributing_field(FiniteElementField *field) { contributing_fields.insert(field); }
      unsigned get_current_residual_index() const { return residual_index; }
      virtual void mark_nonconstant_mass_matrix() {has_constant_mass_matrix_for_sure[residual_index]=false;}
      // Sets the reference point (position, time, normal) used to evaluate initial conditions/Dirichlet
      // conditions that are only enforced/degraded relative to a single representative point rather than pointwise
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
      GiNaC::archive archive; // Accumulates every expression passed through print_simplest_form(), for later inspection/debugging/serialization
      std::map<std::string, GiNaC::ex> expanded_scales;
      GiNaC::ex expand_placeholders(GiNaC::ex inp, std::string where, bool raise_error = true);
      // To prevent tons of Python callbacks in e.g. UNIFAC to substitute molefraction by subexpressions, we cache the expanded callbacks
      std::map<std::tuple<std::string, const bool, const GiNaC::ex, FiniteElementCode *, bool, bool, std::string>, GiNaC::ex> expanded_additional_field_cache;
      virtual void set_equations(Equations *eqs) { equations = eqs; }
      virtual Equations *get_equations() { return equations; }
      virtual void index_fields(); // Assigns each registered field's per-node data index within its space
      virtual void _activate_residual(std::string name); // Selects (creating if necessary) the named residual as the current one (residual_index) for subsequent add_residual() calls
      virtual void debug_second_order_Hessian_deriv(GiNaC::ex inp, std::string dx1, std::string dx2);
      const SpatialIntegralSymbol &get_dx_derived(int dir); // Cached first nodal-coordinate derivative of dx in direction dir (lazily filled in dx_derived)
      const SpatialIntegralSymbol &get_dx_derived2(int dir, int dir2) { return dx_derived2[dir][dir2]; }
      const ElementSizeSymbol &get_elemsize_derived(int dir, bool _consider_coordsys); // Cached first nodal-coordinate derivative of the element size in direction dir
      const ElementSizeSymbol &get_elemsize_derived2(int dir, int dir2, bool _consider_coordsys) { return (_consider_coordsys ? elemsize_derived2[dir][dir2] : elemsize_Cart_derived2[dir][dir2]); }
      const std::vector<FiniteElementSpace *> get_all_spaces() { return allspaces; }
      std::set<FiniteElementField *> get_fields_on_space(FiniteElementSpace *space);
      PositionFiniteElementSpace *get_my_position_space(); // Returns the space holding this code's own nodal (Eulerian) coordinates
      void find_all_accessible_spaces(); // Populates allspaces from this code plus bulk/opposite-interface/external codes
      FiniteElementCodeSubExpression *resolve_subexpression(const GiNaC::ex &e); // Looks up (or fails) the CSE bookkeeping entry for a GiNaCSubExpression-wrapped expression
      int resolve_multi_return_call(const GiNaC::ex &invok); // Looks up the index of an already-registered multi-return call matching invok, or -1
      int element_dim;
      bool analytical_jacobian; // If true, differentiate the residual symbolically for the Jacobian; otherwise the Jacobian is obtained by finite differences at runtime
      bool analytical_position_jacobian; // Same, but specifically for the moving-mesh (nodal position) Jacobian entries
      double debug_jacobian_epsilon; // Step size used when numerically cross-checking the analytical Jacobian against finite differences
      bool with_adaptivity;
      bool coordinates_as_dofs; // If true, the nodal positions are themselves unknowns with residual/Jacobian entries (moving-mesh/ALE elements)
      bool generate_hessian, assemble_hessian_by_symmetry; // Whether to generate Hessian-vector-product code at all, and whether to exploit its symmetry to only derive half of the entries
      std::string coordinate_space;
      bool stop_on_jacobian_difference; // If true, raise an error (rather than just warn) when the analytical and numerical Jacobians disagree beyond tolerance
      std::string ccode_expression_mode = ""; // Mode to write expressions
      std::map<unsigned, unsigned> global_parameter_to_local_indices; // Maps global parameter IDs to their local index within this code's parameter list
      std::vector<std::vector<bool>> local_parameter_has_deriv; // Per residual, per local parameter: whether a derivative w.r.t. that parameter is required
      std::vector<GiNaC::ex> local_parameter_symbols;
      std::vector<FiniteElementCodeSubExpression> subexpressions; // All CSE'd subexpressions registered for this code, in creation order (indices referenced by GiNaCSubExpression)
      std::vector<GiNaC::ex> multi_return_calls; // All distinct multi-return callback invocations registered for this code, in creation order
      std::map<CustomMultiReturnExpressionBase *, std::pair<unsigned, std::string>> multi_return_ccodes; // Per multi-return callback: its assigned numeric id and generated C function name
      void set_integration_order(unsigned order) { integration_order = order; }
      int get_integration_order() { return integration_order; }
      virtual GiNaC::ex eval_flag(std::string flagname); // Evaluates a named boolean/numeric compile-time flag (e.g. from CustomCoordinateSystem) to a GiNaC expression
      virtual void set_bulk_element(FiniteElementCode *_bulk_code) { bulk_code = _bulk_code; }
      virtual FiniteElementCode *get_bulk_element() { return bulk_code; }

      virtual void set_opposite_interface_code(FiniteElementCode *_opposite_interface_code) { opposite_interface_code = _opposite_interface_code; }
      virtual FiniteElementCode *get_opposite_interface_code() { return opposite_interface_code; }

      virtual FiniteElementCode *_resolve_based_on_domain_name(std::string name) { return NULL; } // Overloaded in Python to resolve a domain-name string to the corresponding FiniteElementCode

      virtual std::string get_shapes_required_string(std::string func_type, FiniteElementSpace *space, std::string dx_type);
      virtual void write_required_shapes(std::ostream &os, const std::string indent, std::string func_type); // Emits the "which shapes to fill" flags for func_type based on the required_shapes map
      virtual void mark_further_required_fields(GiNaC::ex expr, const std::string &for_what); // Scans expr for shape expansions/test functions and marks their shapes/spaces as required (and, for other codes, their contribution)
      virtual void mark_shapes_required(std::string func_type, FiniteElementSpace *space, std::string dx_type);
      virtual void mark_shapes_required(std::string func_type, FiniteElementSpace *space, BasisFunction *bf);
      virtual GiNaC::ex get_scaling(std::string name, bool testscale = false) { return 1; } // Nondimensionalization scale factor for field/parameter "name"; overloaded in Python, default is unscaled (1)

      virtual void add_Z2_flux(GiNaC::ex flux,bool for_eigen); // Registers a flux expression for the Z2 error estimator (expands vector/matrix expressions into one entry per component)
      virtual int get_dimension() const { return element_dim; }
      void set_nodal_dimension(unsigned d) { nodal_dim = d; }
      unsigned nodal_dimension() const { return nodal_dim; }

      void set_lagrangian_dimension(unsigned d) { lagr_dim = d; }
      unsigned lagrangian_dimension() const { return lagr_dim; }

      void nullify_bulk_residual(std::string dofname); // Forces the bulk element's residual for field dofname to zero wherever this (interface) code is active

      virtual CustomCoordinateSystem *get_coordinate_system() { return coordinate_sys; } // To be overloaded to get it from the element as well

      FiniteElementCode();
      virtual ~FiniteElementCode();

      // Collects, respectively, all distinct ShapeExpansions / TestFunctions occurring anywhere in expression inp
      // (used to determine which shape-function tables must be computed before evaluating inp).
      std::set<ShapeExpansion> get_all_shape_expansions_in(GiNaC::ex inp, bool merge_no_jacobian = true, bool merge_expansion_modes = true, bool merge_no_hessian = true);
      std::set<TestFunction> get_all_test_functions_in(GiNaC::ex inp);

      void fill_callback_info(JITFuncSpec_Table_FiniteElement_t *ft); // Fills the JIT function table's callback-function-pointer entries (parameters, custom math functions, multi-return calls) for the compiled element

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
      GiNaC::ex expand_all_and_ensure_nondimensional(GiNaC::ex what, std::string where, GiNaC::ex *collected_units_and_factor = NULL); // Expands placeholders/fields in "what" and checks/divides out that the result is unit-free, optionally returning the collected unit+numeric factor
      virtual void add_residual(GiNaC::ex add, bool allow_contributions_without_dx); // Adds "add" to the currently active residual (residual_index); allow_contributions_without_dx permits terms without an explicit dx/dX integration measure (e.g. nodal-delta contributions)
      virtual void write_generic_spatial_integration_header(std::ostream &os, std::string indent, GiNaC::ex eulerian_part, GiNaC::ex lagrangian_part, std::string required_table_and_flag);
      virtual void write_generic_spatial_integration_footer(std::ostream &os, std::string indent);
      virtual void write_generic_nodal_delta_header(std::ostream &os, std::string indent);
      virtual void write_generic_nodal_delta_footer(std::ostream &os, std::string indent);
      virtual void write_generic_RJM(std::ostream &os, std::string funcname, GiNaC::ex resi, bool with_hang);     // Generic Residual/Jacobian/Mass matrix (also for parameter derivatives)
      virtual bool write_generic_Hessian(std::ostream &os, std::string funcname, GiNaC::ex resi, bool with_hang); // Generic Hessian vector product
      virtual void write_code(std::ostream &os); // Top-level driver: writes the full generated C++ source for this element (calls the write_code_*()/write_generic_*() methods for every residual)
      virtual GiNaC::ex get_dx(bool lagrangian,bool unity_only=false); // Returns the (cached) dx/dX integration-measure symbol wrapped as a GiNaC::ex
      virtual GiNaC::ex get_element_size_symbol(bool lagrangian, bool with_coordsys);
      virtual GiNaC::ex get_integral_dx(bool use_scaling, bool lagrangian, CustomCoordinateSystem *coordsys) { return get_dx(lagrangian); }
      virtual GiNaC::ex get_element_size(bool use_scaling, bool lagrangian, bool with_coordsys, CustomCoordinateSystem *coordsys) { return get_element_size_symbol(lagrangian, with_coordsys); }
      virtual GiNaC::ex get_nodal_delta(); // Returns the (cached) nodal-delta symbol wrapped as a GiNaC::ex
      virtual GiNaC::ex get_normal_component(unsigned i); // Returns component i of the outward normal as a GiNaC::ex (NormalSymbol)
      //virtual GiNaC::ex get_normal_component_eigenexpansion(unsigned i); // Used for azimuthal eigenstab only. Gives dn_i/dX^{0l}_j * X^{ml}_j
      virtual void set_derive_jacobian_by_expansion_mode(std::string residual_name,int expansion_mode) { derive_jacobian_by_expansion_mode[residual_name]=expansion_mode; }
      virtual void set_derive_hessian_by_expansion_mode(std::string residual_name,int expansion_mode) { derive_hessian_by_expansion_mode[residual_name]=expansion_mode; }
      virtual void set_ignore_dpsi_coord_diffs_in_jacobian(std::string residual_name) { ignore_dpsi_coord_diffs_in_jacobian_set.insert(residual_name); }
      virtual void set_ignore_residual_assembly(std::string residual_name) { ignore_assemble_residuals.insert(residual_name); }
      virtual bool is_current_residual_assembly_ignored() { return ignore_assemble_residuals.count(residual_names[residual_index]); }
      virtual bool is_residual_assembly_ignored(std::string residual_name) { return ignore_assemble_residuals.count(residual_name); }
      virtual int * get_derive_jacobian_by_expansion_mode() { if (!derive_jacobian_by_expansion_mode.count(residual_names[residual_index])) return NULL; else return &(derive_jacobian_by_expansion_mode[residual_names[residual_index]]) ; }
      virtual int * get_derive_hessian_by_expansion_mode() { if (!derive_hessian_by_expansion_mode.count(residual_names[residual_index])) return NULL; else return &(derive_hessian_by_expansion_mode[residual_names[residual_index]]) ; }
      virtual bool ignore_dpsi_coord_diffs_in_jacobian() { return ignore_dpsi_coord_diffs_in_jacobian_set.count(residual_names[residual_index]); }

      // Classifies where space s "lives" relative to this code, to decide how to access its data in generated code:
      // 0 if the space is defined on this element, -1 for bulk element, -2 for other side of interface, >0 for external elements [-1]
      virtual int classify_space_type(const FiniteElementSpace *s);
      virtual std::string get_owner_prefix(const FiniteElementSpace *sp); // C++ expression prefix to access data owned by the element that "sp" belongs to (self/bulk/opposite/external), based on classify_space_type()
      virtual std::string get_shape_info_str(const FiniteElementSpace *sp);
      virtual std::string get_elem_info_str(const FiniteElementSpace *sp);
      virtual std::string get_nodal_data_string(const FiniteElementSpace *sp);
      virtual void finalise(); // Locks in the residual definitions and prepares the code for write_code() (space/field discovery, index assignment, etc.)
      virtual void _do_define_fields(int element_dimension);
      virtual GiNaC::ex expand_additional_field(const std::string &name, const bool &dimensional, const GiNaC::ex &expr, FiniteElementCode *in_domain, bool no_jacobian, bool no_hessian, std::string where) { return expr; } // Overloaded in Python to expand domain-specific pseudo-fields (e.g. derived/auxiliary quantities) into real expressions
      virtual GiNaC::ex expand_additional_testfunction(const std::string &name, const GiNaC::ex &expr, FiniteElementCode *in_domain) { return expr; } // Overloaded in Python, analogous to expand_additional_field() but for test functions

      virtual std::string get_default_timestepping_scheme(unsigned int dt_order) { return (dt_order == 1 ? "BDF2" : "Newmark2"); }
      virtual unsigned get_default_spatial_integration_order() { return 0; }
      virtual void set_initial_condition(const std::string &name, GiNaC::ex expr, std::string degraded_start, const std::string &ic_name);
      virtual void set_Dirichlet_bc(const std::string &name, GiNaC::ex expr, bool use_identity);
      virtual void _define_element(); // Forwards to equations->_define_element() (adds the residual contributions); overridden/called from Python
      virtual void _register_external_ode_linkage(std::string my_fieldname, FiniteElementCode *odecode, std::string odefieldname) {} // Overloaded in Python to record that my_fieldname is coupled to odefieldname on an external ODE code

      virtual GiNaC::ex derive_expression(const GiNaC::ex &what, const GiNaC::ex by); // Symbolically differentiates "what" w.r.t. "by", accounting for the pyoomph-specific symbol types (ShapeExpansion, TestFunction, etc.)

      virtual void _define_fields(); // Forwards to equations->_define_fields() (registers fields/spaces); overridden/called from Python
      virtual bool _is_ode_element() const { return false; }
      // Default domain name: the object's own address, stringified (unique per code); overridden in Python to return the actual mesh/domain path
      virtual std::string get_domain_name()
      {
         std::ostringstream oss;
         oss << this;
         return oss.str();
      }
      // Full "bulk/.../interface" path built by recursing through get_bulk_element()
      virtual std::string get_full_domain_name()
      {
         if (this->get_bulk_element()) return this->get_bulk_element()->get_full_domain_name()+"/"+this->get_domain_name();
         else return this->get_domain_name();
      }
      virtual void set_discontinuous_refinement_exponent(std::string field, double exponent);
      double warn_on_large_numerical_factor = 0.0; // If nonzero, warn (or, if negative, only warn without further action) when a generated numerical coefficient exceeds this magnitude, which often indicates a nondimensionalization issue
      bool use_shared_shape_buffer_during_multi_assemble = false; // If true, elements sharing the same shape-function buffer during a multi-assemble pass reuse it instead of recomputing it (performance optimization, see elements.cpp)
      LaTeXPrinter *latex_printer;
      virtual void set_latex_printer(LaTeXPrinter *lp) { latex_printer = lp; }


      std::set<FiniteElementField *> Hessian_symmetric_fields_completed; // Fields whose Hessian block has already been written for the current residual; used together with assemble_hessian_by_symmetry to avoid deriving symmetric entries twice, reset per residual

   };

   extern FiniteElementCode *__current_code; // The FiniteElementCode currently being defined/assembled (set while running Python-side _define_fields()/_define_element() callbacks); used as an implicit context by free functions that need "the current code"

}
