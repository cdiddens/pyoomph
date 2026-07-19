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

/*********************************
 Older versions of GiNaC do not have all methods implemented in the structure class. This file is a workaround to make it work with older versions of GiNaC.
 I.e. this is based of the structure.h file from the GiNaC source code.
 Also, it is ensured that all structures are assumed to be real-valued.

 See https://www.ginac.de/ginac.git/?p=ginac.git;a=blob_plain;f=ginac/structure.h;hb=HEAD for the original file of GiNaC
**********************************/



#pragma once

#include "ginac.hpp"

#include <functional>


// __NO_PYGINAC_STRUCT is always defined here (unconditionally, right below), so the
// NO_PYGINAC_STRUCT branch (falling back to GiNaC's own GiNaC::structure<T>) is normally
// dead code, kept only in case a future/older GiNaC version needs it again; the pyginacstruct
// implementation below is what pyoomph actually uses via the PYGINACSTRUCT(x,n) macro.
#define __NO_PYGINAC_STRUCT
#ifdef NO_PYGINAC_STRUCT
#define PYGINACSTRUCT(x,n) typedef GiNaC::structure<x> n;
#else
namespace GiNaC
{

  // Select default comparison policy
  template <class T, template <class> class ComparisonPolicy = compare_all_equal>
  class pyginacstruct;

  // VARIATION: THIS ONE IS REAL ONLY!
  /** Wrapper template for making GiNaC classes out of C++ structures. Wraps an arbitrary
   * C++ struct T as a GiNaC::basic expression node (so instances of T can appear inside
   * GiNaC expression trees), forwarding GiNaC's structural-equality/ordering machinery to
   * T via the ComparisonPolicy (which is expected to provide struct_is_equal/struct_compare
   * on T - see GiNaC's compare_all_equal/compare_std_less policies), and treating the
   * wrapped value as an opaque, real-valued, atomic (nops()==0) leaf: none of GiNaC's
   * structural operations (differentiation, substitution, expansion, ...) look inside T,
   * they either no-op or delegate straight to the inherited GiNaC::basic default. Used via
   * the PYGINACSTRUCT(x,n) macro at the bottom of this file, which defines a concrete
   * typedef n for pyginacstruct<x, compare_std_less> (see pyginacstruct.hpp usage). */
  template <class T, template <class> class ComparisonPolicy>
  class pyginacstruct : public basic, public ComparisonPolicy<T>
  {
    GINAC_DECLARE_REGISTERED_CLASS(pyginacstruct, basic)

    // helpers
    static const char *get_class_name() { return "pyginacstruct"; }
    // constructors
  public:
    /** Construct structure as a copy of a given C++ structure. */
    pyginacstruct(const T &t) : obj(t) {}

    // functions overriding virtual functions from base classes
    // All these are just defaults that can be specialized by the user
  public:
    // evaluation
    ex eval() const override { return hold(); }
    ex evalm() const override { return inherited::evalm(); }

  protected:
    ex eval_ncmul(const exvector &v) const override { return hold_ncmul(v); }

  public:
    ex eval_indexed(const basic &i) const override { return i.hold(); }

    // printing
    void print(const print_context &c, unsigned level = 0) const override { inherited::print(c, level); }
    unsigned precedence() const override { return 70; }

    // info
    bool info(unsigned inf) const override 
    { 
      switch (inf)
      {
        case info_flags::real:
          return true;
          break;			
        default:
          break;
      }
      return basic::info(inf); 
    }

    // operand access
    size_t nops() const override { return 0; }
    ex op(size_t i) const override { return inherited::op(i); }
    ex operator[](const ex &index) const override { return inherited::operator[](index); }
    ex operator[](size_t i) const override { return inherited::operator[](i); }
    ex &let_op(size_t i) override { return inherited::let_op(i); }
    ex &operator[](const ex &index) override { return inherited::operator[](index); }
    ex &operator[](size_t i) override { return inherited::operator[](i); }

    // pattern matching
    bool has(const ex &other, unsigned options = 0) const override { return inherited::has(other, options); }
    bool match(const ex &pattern, exmap &repl_lst) const override { return inherited::match(pattern, repl_lst); }

  protected:
    bool match_same_type(const basic &) const override { return true; }

  public:
    // substitutions
    ex subs(const exmap &m, unsigned options = 0) const override { return inherited::subs(m, options); }

    // function mapping
    ex map(map_function &f) const override { return inherited::map(f); }

    // degree/coeff
    int degree(const ex &s) const override { return inherited::degree(s); }
    int ldegree(const ex &s) const override { return inherited::ldegree(s); }
    ex coeff(const ex &s, int n = 1) const override { return inherited::coeff(s, n); }

    // expand/collect
    ex expand(unsigned options = 0) const override { return inherited::expand(options); }
    ex collect(const ex &s, bool distributed = false) const override { return inherited::collect(s, distributed); }

    // differentiation and series expansion
  protected:
    ex derivative(const symbol &s) const override { return inherited::derivative(s); }

  public:
    ex series(const relational &r, int order, unsigned options = 0) const override { return inherited::series(r, order, options); }

    ex real_part() const override { return *this; }
    ex imag_part() const override { return 0; }
    // rational functions
    ex normal(exmap &repl, exmap &rev_lookup, lst &modifier) const override { return inherited::normal(repl, rev_lookup, modifier); }
    ex to_rational(exmap &repl) const override { return inherited::to_rational(repl); }
    ex to_polynomial(exmap &repl) const override { return inherited::to_polynomial(repl); }

    // polynomial algorithms
    numeric integer_content() const override { return 1; }
    ex smod(const numeric &) const override { return *this; }
    numeric max_coefficient() const override { return 1; }

    // indexed objects
    exvector get_free_indices() const override { return exvector(); }
    ex add_indexed(const ex &self, const ex &other) const override { return self + other; }
    ex scalar_mul_indexed(const ex &self, const numeric &other) const override { return self * ex(other); }
    bool contract_with(exvector::iterator , exvector::iterator , exvector &) const override { return false; }

    // noncommutativity
    unsigned return_type() const override { return return_types::commutative; }
    return_type_t return_type_tinfo() const override
    {
      return_type_t r;
      r.rl = 0;
      r.tinfo = &typeid(*this);
      return r;
    }

    // unsigned get_domain() const override { return domain::real; }

    ex conjugate() const override { return *this; }

  protected:
    bool is_equal_same_type(const basic &other) const override
    {
      GINAC_ASSERT(is_a<pyginacstruct>(other));
      const pyginacstruct &o = static_cast<const pyginacstruct &>(other);

      return this->struct_is_equal(&obj, &o.obj);
    }

    unsigned calchash() const override { return inherited::calchash(); }

    // non-virtual functions in this class
  public:
    // access to embedded structure
    const T *operator->() const { return &obj; }
    T &get_struct() { return obj; }
    const T &get_struct() const { return obj; }

  private:
    T obj;
  };

  /** Default constructor */
  template <class T, template <class> class CP>
  pyginacstruct<T, CP>::pyginacstruct() {}

  /** Compare two structures of the same type. */
  template <class T, template <class> class CP>
  int pyginacstruct<T, CP>::compare_same_type(const basic &other) const
  {
    GINAC_ASSERT(is_a<pyginacstruct>(other));
    const pyginacstruct &o = static_cast<const pyginacstruct &>(other);

    return this->struct_compare(&obj, &o.obj);
  }

  template <class T, template <class> class CP>
  registered_class_info pyginacstruct<T, CP>::reg_info = registered_class_info(registered_class_options(pyginacstruct::get_class_name(), "basic", typeid(pyginacstruct<T, CP>)));

} // namespace GiNaC
//#define PYGINACSTRUCT(x) GiNaC::pyginacstruct<x,GiNaC::compare_std_less>
#define PYGINACSTRUCT(x,n) typedef GiNaC::pyginacstruct<x,GiNaC::compare_std_less> n;
#endif
