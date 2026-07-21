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


#include "codegen.hpp"
#include "expressions.hpp"
#include "exception.hpp"
#include "problem.hpp"
#include <limits>

namespace pyoomph
{

	/*
	   class GatherDistributiveNumericalFactors : public GiNaC::map_function
		{
		   GiNaC::ex operator()(const GiNaC::ex &inp)
			{
			 if (GiNaC::is_a<GiNaC::mul>(inp))
			 {
			   GiNaC::ex newres=1;
			   for (unsigned int i=0;i<inp.nops();i++)
			   {
				if (GiNaC::is_a<GiNaC::add>(inp.op(i)))
				{
				 GiNaC::ex applied=inp.op(i).map(*this);
				 if (GiNaC::is_a<GiNaC::add>(applied))
				 {
				  //Find the largest numerical coefficient
				  raise TODO

				 }
				 else
				 {
				  newres*=inp.op(i);
				 }
				}
				else
				{
				 newres*=inp.op(i);
				}
			   }
			   return newres;
			 }
			 else
			 {
			   return inp.map(*this);
			 }
			}
		};
		*/

	// Prints a GiNaC expression as C source code, after applying an optional simplification
	// strategy selected at runtime via FiniteElementCode::ccode_expression_mode (e.g. "factor",
	// "normal", "expand", "collect_common_factors" - mostly useful for debugging/benchmarking
	// how different GiNaC simplifications affect the generated code). The expression is also
	// archived (csrc_opts.for_code->archive) so that it can be inspected/replayed later.
	void print_simplest_form(GiNaC::ex expr, std::ostream &os, GiNaC::print_FEM_options &csrc_opts)
	{
		GiNaC::ex towrite;
		std::string mode = csrc_opts.for_code->ccode_expression_mode;
		csrc_opts.for_code->archive.archive_ex(expr, ("expression_"+std::to_string(csrc_opts.for_code->archive.num_expressions())).c_str());
		if (mode == "deterministic")
		{
			//GiNaC::print_sorted_GiNaC(GiNaC::expand(GiNaC::expand(expr)),os,csrc_opts);
			GiNaC::print_sorted_GiNaC(expr,os,csrc_opts);
			return;
		}
		else if (mode == "factor")
			towrite = GiNaC::factor(GiNaC::normal(GiNaC::expand(GiNaC::expand(expr).evalf())));
		else if (mode == "normal")
			towrite = GiNaC::normal(GiNaC::expand(GiNaC::expand(expr).evalf()));
		else if (mode == "expand")
			towrite = GiNaC::expand(GiNaC::expand(expr).evalf()).evalf();
		else if (mode == "collect_common_factors")
			towrite = GiNaC::collect_common_factors(GiNaC::expand(GiNaC::expand(expr).evalf()).evalf());
		else if (mode == "test")
			towrite = GiNaC::normal(GiNaC::factor(GiNaC::collect_common_factors(GiNaC::expand(GiNaC::expand(expr).evalf()))));
		else if (mode == "test2")
			towrite = GiNaC::normal(GiNaC::factor(GiNaC::collect_common_factors(GiNaC::expand(GiNaC::expand(expr))))).evalf();
		else if (mode == "test3")
			towrite = GiNaC::normal(GiNaC::expand(expr));
		else if (mode == "expand_no_evalf")
			towrite = GiNaC::expand(expr);
		else if (mode == "ccf_no_evalf")
			towrite = GiNaC::collect_common_factors(GiNaC::expand(expr));		
		else
			towrite = expr.evalf();
		//	std::cout << "MODE WAS " << mode << std::endl;
		towrite.print(GiNaC::print_csrc_FEM(os, &csrc_opts));
	}

	//////////////

	// Global code-generation state. These globals are set/restored around the symbolic
	// differentiation and code-emission passes below and are read by the custom GiNaC
	// structure classes (ShapeExpansion, SpatialIntegralSymbol, ...) whose derivative()/print()
	// implementations otherwise have no direct access to "which pass are we currently in".
	FiniteElementCode *__current_code;                                // Code object currently being processed (residual/Jacobian/Hessian assembly)
	const ShapeExpansion *__deriv_subexpression_wrto = NULL;          // Set while differentiating a subexpression w.r.t. a specific field/direction
	bool __derive_shapes_by_second_index = false;                     // True while building the "second" (l_shape2) index of a Hessian double-loop
	bool __in_pitchfork_symmetry_constraint = false;                  // True while generating the extra pitchfork-symmetry-breaking constraint equations
	int * __derive_only_by_expansion_mode = NULL;                     // If set, only derivatives w.r.t. this azimuthal/Fourier expansion mode are kept
	bool __ignore_dpsi_coord_diffs_in_jacobian = false;                // Suppresses dpsi/dX (moving-mesh) contributions in the Jacobian for the current residual
	std::set<ShapeExpansion> __all_Hessian_shapeexps;                 // Accumulates all shape expansions encountered while building a Hessian contribution
	std::set<TestFunction> __all_Hessian_testfuncs;                   // Accumulates all test functions encountered while building a Hessian contribution
	std::set<FiniteElementField *> __all_Hessian_indices_required;    // Fields that need a Hessian index (i.e. contribute to the outer derivative direction)
	bool __in_hessian = false;                                        // True while performing second-order (Hessian) differentiation

	// Pitchfork/symmetry-breaking constraint equations must not additionally pick up the usual
	// Jacobian contribution from moving nodal positions (they are handled separately), unless we
	// are currently building the second index of a Hessian double loop.
	bool ignore_nodal_position_derivatives_for_pitchfork_symmetry()
	{
		return __in_pitchfork_symmetry_constraint && !pyoomph::__derive_shapes_by_second_index;
	}

	


	// GiNaC tree-mapper that replaces every explicit expressions::subexpression(...) marker in an
	// expression by a GiNaCSubExpression structure, collecting the underlying (mapped) expressions
	// into `subexpressions` and de-duplicating identical ones (so the same subexpression is only
	// emitted/evaluated once in the generated C code). While doing so it also resolves nested
	// multi-return callback invocations, and - if the element uses moving-mesh coordinates as
	// degrees of freedom - strips out shape expansions of the nodal position field that must be
	// ignored for pitchfork-symmetry-breaking constraint equations.
	class SubExpressionsToStructs : public GiNaC::map_function
	{
	protected:
		FiniteElementCode *code;

	public:
		std::vector<FiniteElementCodeSubExpression> subexpressions;
		SubExpressionsToStructs(FiniteElementCode *code_) : code(code_) {}
		GiNaC::ex operator()(const GiNaC::ex &inp) override
		{
			if (is_ex_the_function(inp, expressions::subexpression))
			{
				GiNaC::ex mapped_ex = inp.op(0).map(*this);
				if (GiNaC::is_a<GiNaC::GiNaCMultiRetCallback>(mapped_ex))
				{
					const auto &sp = GiNaC::ex_to<GiNaC::GiNaCMultiRetCallback>(mapped_ex).get_struct();
					GiNaC::ex invok = expressions::python_multi_cb_function(sp.invok.op(0), sp.invok.op(1).map(*this), sp.invok.op(2));
					mapped_ex = GiNaC::GiNaCMultiRetCallback(pyoomph::MultiRetCallback(sp.code, invok.map(*this), sp.retindex, sp.derived_by_arg));
					invok = GiNaC::ex_to<GiNaC::GiNaCMultiRetCallback>(mapped_ex).get_struct().invok;
					if (code->resolve_multi_return_call(invok) < 0)
					{
						code->multi_return_calls.push_back(invok);
					}
				}

				GiNaC::ex res = GiNaC::GiNaCSubExpression(SubExpression(code, mapped_ex));
				auto &st = GiNaC::ex_to<GiNaC::GiNaCSubExpression>(res).get_struct();
				bool found = false;
				for (unsigned int j = 0; j < subexpressions.size(); j++)
					if (st.expr.is_equal(subexpressions[j].get_expression()))
					{
						found = true;
						break;
					}
				if (!found)
				{
					std::set<ShapeExpansion> sub_shapeexps = code->get_all_shape_expansions_in(st.expr);
					std::set<TestFunction> sub_testfuncs = code->get_all_test_functions_in(st.expr);
					if (!sub_testfuncs.empty())
					{
						throw_runtime_error("Subexpressions may not depend on test functions!");
					}
					/*					for (GiNaC::const_preorder_iterator i = st.expr.preorder_begin(); i != st.expr.preorder_end(); ++i)
										{
										 if (GiNaC::is_a<GiNaC::GiNaCMultiRetCallback>(*i))
										 {
										  std::ostringstream oss;
										  oss << std::endl << "Happened in: " << std::endl << st.expr << std::endl << "where the following was found:" << std::endl << (*i);
										  throw_runtime_error("Results of Multi-Return-Expressions cannot be wrapped in subexpressions yet. Adjust your multi-return-expression that way that it returns already the term you want to wrap into subexpression nicely."+oss.str());
										 }
										}
					*/
					if (code->coordinates_as_dofs && !pyoomph::ignore_nodal_position_derivatives_for_pitchfork_symmetry())
					{
						// Now this is a bit harder: We need to remove this spaces here!
						for (auto d : std::vector<std::string>{"x", "y", "z"})
						{
							FiniteElementField *cf = code->get_field_by_name("coordinate_" + d);
							if (cf)
							{
								std::vector<std::string> time_schemes = {"BDF1", "BDF2", "Newmark2", "TIME_DIFF_SCHEME_NOT_SET"};
								std::vector<BasisFunction *> bases = {cf->get_space()->get_basis()};
								for (unsigned ib = 0; ib < 3; ib++)
									bases.push_back(bases[0]->get_diff_x(ib));
								for (auto ts : time_schemes)
								{
									for (auto bas : bases)
									{
										for (unsigned int ti = 0; ti <= 2; ti++)
										{
											ShapeExpansion se(cf, ti, bas, ts);
											sub_shapeexps.erase(se);
										}
									}
								}

								if (this->code->get_bulk_element())
								{
									FiniteElementField *cf = code->get_bulk_element()->get_field_by_name("coordinate_" + d);
									bases = {cf->get_space()->get_basis()};
									for (unsigned ib = 0; ib < 3; ib++)
										bases.push_back(bases[0]->get_diff_x(ib));
									for (auto ts : time_schemes)
									{
										for (auto bas : bases)
										{
											for (unsigned int ti = 0; ti <= 2; ti++)
											{
												ShapeExpansion se(cf, ti, bas, ts);
												sub_shapeexps.erase(se);
											}
										}
									}
									if (this->code->get_bulk_element()->get_bulk_element())
									{
										FiniteElementField *cf = code->get_bulk_element()->get_bulk_element()->get_field_by_name("coordinate_" + d);
										bases = {cf->get_space()->get_basis()};
										for (unsigned ib = 0; ib < 3; ib++)
											bases.push_back(bases[0]->get_diff_x(ib));
										for (auto ts : time_schemes)
										{
											for (auto bas : bases)
											{
												for (unsigned int ti = 0; ti <= 2; ti++)
												{
													ShapeExpansion se(cf, ti, bas, ts);
													sub_shapeexps.erase(se);
												}
											}
										}
									}
								}
							}
						}
					}
					subexpressions.push_back(FiniteElementCodeSubExpression(st.expr.map(*this), GiNaC::potential_real_symbol("subexpr_" + std::to_string(subexpressions.size())), sub_shapeexps));
				}

				return res;
			}
			/*			else if (is_ex_the_function(inp, expressions::python_multi_cb_function))
						{
						 return expressions::python_multi_cb_function(inp.op(0),inp.op(1).map(*this),inp.op(2));
						}
			*/
			else if (GiNaC::is_a<GiNaC::GiNaCMultiRetCallback>(inp))
			{
				const auto &sp = GiNaC::ex_to<GiNaC::GiNaCMultiRetCallback>(inp).get_struct();
				GiNaC::ex invok = expressions::python_multi_cb_function(sp.invok.op(0), sp.invok.op(1).map(*this), sp.invok.op(2));
				GiNaC::ex res = GiNaC::GiNaCMultiRetCallback(pyoomph::MultiRetCallback(sp.code, invok.map(*this), sp.retindex, sp.derived_by_arg));
				invok = GiNaC::ex_to<GiNaC::GiNaCMultiRetCallback>(res).get_struct().invok;

				if (code->resolve_multi_return_call(invok) < 0)
				{
					code->multi_return_calls.push_back(invok);
				}
				return res;
			}
			else
			{
				GiNaC::ex res = inp.map(*this);
				return res;
			}
		}
	};

	SubExpressionsToStructs *__SE_to_struct_hessian = NULL;

	// GiNaC tree-mapper used to isolate the part of a residual expression that belongs to a single
	// test-function space (and, if `varname` is given, to a single field's test function): any
	// GiNaCTestFunction on a different space is replaced by 0, so that mapping this over the full
	// residual yields exactly the terms that must be assembled into that field's residual/Jacobian
	// row. Also validates that global parameters referenced in the expression belong to the same
	// Problem as the space being processed (parameter indices are only meaningful within one problem).
	class MapOnTestSpace : public GiNaC::map_function
	{
	protected:
		FiniteElementSpace *space;
		std::string varname;
		FiniteElementField *field;

	public:
		FiniteElementField *get_field() { return field; }
		MapOnTestSpace(FiniteElementSpace *sp, std::string vn) : space(sp), varname(vn), field(NULL) {}
		GiNaC::ex operator()(const GiNaC::ex &inp) override
		{
			if (GiNaC::is_a<GiNaC::GiNaCTestFunction>(inp))
			{
				auto &tst = (GiNaC::ex_to<GiNaC::GiNaCTestFunction>(inp)).get_struct();
				if (tst.basis->get_space() != this->space)
					return 0;
				else if (varname != "")
				{
					if (varname == tst.field->get_name())
					{
						if (!field)
							field = tst.field;
						return inp.map(*this);
					}
					else
						return 0;
				}
				else
					return inp.map(*this);
			}
			else if (GiNaC::is_a<GiNaC::GiNaCGlobalParameterWrapper>(inp))
			{
				if (!this->space->get_code()->get_problem()) 
				{
					//throw_runtime_error("For some reason, the code generator is not able to access the problem here. Please report this bug to the developers. Happened on a variable named '"+varname+"'. The code is "+this->space->get_code()->get_full_domain_name()+". The space is "+this->space->get_name()+".");
					return inp.map(*this); // Just do not check in this case...
				}
				// check whether the parameter belongs to the same problem. Otherwise, things get messed up in the parameter indicies
				auto &p = (GiNaC::ex_to<GiNaC::GiNaCGlobalParameterWrapper>(inp)).get_struct();
				if (!(p.cme->get_problem() == this->space->get_code()->get_problem()))
				{
					std::ostringstream oss; 
					oss<< "Problem of Parameter: " << p.cme->get_problem() ;
					oss<< " vs. Current Problem: " << this->space->get_code()->get_problem();
					oss << " Are the same? " << (p.cme->get_problem() == this->space->get_code()->get_problem() ? " yes " : "no");
					throw_runtime_error("You added a global parameter '" + p.cme->get_name() + "' defined in one problem to the residuals of a different problem. This is not allowed.... "+oss.str());				
				}
				return inp.map(*this);
			}
			else
				return inp.map(*this);
		}
	};

	// GiNaC tree-mapper that rewrites a residual to its "steady" form: any shape expansion or
	// spatial-integral symbol that explicitly references a past time-history slot (time_history_index
	// / history_step > 0) but has no actual time derivative is redirected to the current-time value.
	// This is needed for time-stepping schemes with explicit dependence on previous-step DoFs (e.g.
	// MPT, TPZ), which therefore require a separate "extra steady" residual routine for steady solves;
	// require_extra_steady_routine() reports whether such a rewrite actually occurred.
	class MakeResidualSteady : public GiNaC::map_function
	{
	protected:
		FiniteElementCode *code;
		bool extra_steady_routine;

	public:
		MakeResidualSteady(FiniteElementCode *_code) : code(_code), extra_steady_routine(false) {}
		GiNaC::ex operator()(const GiNaC::ex &inp) override
		{
			if (GiNaC::is_a<GiNaC::GiNaCShapeExpansion>(inp))
			{
				auto &shp = (GiNaC::ex_to<GiNaC::GiNaCShapeExpansion>(inp)).get_struct();
				if (shp.dt_order == 0 && shp.time_history_index) // No time derivative, but in history
				{
					ShapeExpansion repl = shp;
					repl.time_history_index = 0; // Evaluate at current time
					extra_steady_routine = true; // We require an extra steady routine in that case
					return GiNaC::GiNaCShapeExpansion(repl);
				}
				return inp;
			}
			else if (GiNaC::is_a<GiNaC::GiNaCSpatialIntegralSymbol>(inp))
			{
				auto &si = (GiNaC::ex_to<GiNaC::GiNaCSpatialIntegralSymbol>(inp)).get_struct();
				if (si.history_step > 0)
				{
					SpatialIntegralSymbol repl = si;
					repl.history_step = 0; // Evaluate at current time
					extra_steady_routine = true; // We require an extra steady routine in that case
					return GiNaC::GiNaCSpatialIntegralSymbol(repl);
				}
				else
				{
					return inp.map(*this);
				}
			}
			else
				return inp.map(*this);
		}

		bool require_extra_steady_routine() const { return extra_steady_routine; }
	};

	// GiNaC tree-mapper that replaces every global-parameter wrapper by its current numerical value,
	// used where a fully numeric evaluation (rather than a symbolic parameter dependency) is required.
	class GlobalParamsToValues : public GiNaC::map_function
	{
	public:
		GiNaC::ex operator()(const GiNaC::ex &inp) override
		{
			if (GiNaC::is_a<GiNaC::GiNaCGlobalParameterWrapper>(inp))
			{
				auto &p = (GiNaC::ex_to<GiNaC::GiNaCGlobalParameterWrapper>(inp)).get_struct();
				return p.cme->value();
			}
			else
				return inp.map(*this);
		}
	};

	// For every expressions::subexpression(...) or expressions::Diff(...) node, factors the argument
	// into a pure number, a dimensional unit, and a dimensionless "rest", then rebuilds the node so
	// that only the dimensionless rest stays wrapped in the subexpression - the numeric factor and
	// the unit are pulled out and multiplied back in unwrapped. This keeps subexpression C variables
	// (and their common-subexpression caching) purely numeric, while units are still tracked and
	// cancelled symbolically by GiNaC outside of the subexpression boundary.
	GiNaC::ex DrawUnitsOutOfSubexpressions::operator()(const GiNaC::ex &inp)
	{
		//	std::cout << "INP " <<inp << std::endl;
		if (is_ex_the_function(inp, expressions::subexpression))
		{
			if (pyoomph_verbose)
				std::cout << "INP SE:  " << inp << std::endl;
			GiNaC::ex factor, unit, rest;
			GiNaC::ex arg = inp.map(*this);
			if (GiNaC::is_a<GiNaC::numeric>(arg) ) return arg; // No units here
			arg=arg.op(0); // Descent recursively through nested subexpressions
			if (pyoomph_verbose)
				std::cout << "PROCESSING " << inp << std::endl
						  << "YIELDS " << arg << std::endl
						  << std::endl;
			if (!expressions::collect_base_units(arg, factor, unit, rest))
			{
				std::ostringstream oss;
				oss << std::endl
					<< "INPUT: " << inp << std::endl
					<< "PROCESSED ARG:" << arg << std::endl
					<< "numerical part: " << factor << "unit part:" << unit << "rest part:" << rest << std::endl;
				throw_runtime_error("Cannot extract the unit from the subexpression:" + oss.str());
			}
			if (pyoomph_verbose)
				std::cout << "SEP: " << arg << "  n " << factor << " u " << unit << "  r  " << rest << std::endl;
			if (pyoomph_verbose)
				std::cout << "RET: " << (factor * unit * expressions::subexpression(rest)) << std::endl;
			return factor * unit * expressions::subexpression(rest);
		}
		else if (is_ex_the_function(inp, expressions::Diff))
		{
			GiNaC::ex factor, unit, rest;
			GiNaC::ex arg = inp.map(*this).op(0); // Descent recursively through nested subexpressions
			if (!expressions::collect_base_units(arg, factor, unit, rest))
			{
				std::ostringstream oss;
				oss << std::endl
					<< "INPUT: " << inp << std::endl
					<< "PROCESSED ARG:" << arg << std::endl
					<< "numerical part: " << factor << "unit part:" << unit << "rest part:" << rest << std::endl;
				throw_runtime_error("Cannot extract the unit from the derivative numerator:" + oss.str());
			}
			GiNaC::ex factor2, unit2, rest2;
			GiNaC::ex arg2 = inp.map(*this).op(1); // Descent recursively through nested subexpressions
			if (!expressions::collect_base_units(arg2, factor2, unit2, rest2))
			{
				std::ostringstream oss;
				oss << std::endl
					<< "INPUT: " << inp << std::endl
					<< "PROCESSED ARG:" << arg2 << std::endl
					<< "numerical part: " << factor2 << "unit part:" << unit2 << "rest part:" << rest2 << std::endl;
				throw_runtime_error("Cannot extract the unit from the derivative denominator:" + oss.str());
			}
			//		return (unit/unit2)*expressions::Diff(factor*rest,factor2*rest2);
			return (factor / factor2) * (unit / unit2) * expressions::Diff(rest, factor2 * rest2);
		}

		return inp.map(*this);
	}

	// GiNaC tree-mapper that undoes subexpression wrapping, i.e. unwraps every
	// expressions::subexpression(...) marker back to its plain argument. Used where the
	// subexpression-CSE optimization is not desired/applicable and the raw expression is needed instead.
	GiNaC::ex RemoveSubexpressionsByIndentity::operator()(const GiNaC::ex &inp)
	{
		if (is_ex_the_function(inp, expressions::subexpression))
		{
			return inp.op(0).map(*this);
		}
		else if (GiNaC::is_a<GiNaC::GiNaCMultiRetCallback>(inp))
		{
			const auto &sp = GiNaC::ex_to<GiNaC::GiNaCMultiRetCallback>(inp).get_struct();
			GiNaC::ex invok = expressions::python_multi_cb_function(sp.invok.op(0), sp.invok.op(1).map(*this), sp.invok.op(2));
			GiNaC::ex res = GiNaC::GiNaCMultiRetCallback(pyoomph::MultiRetCallback(sp.code, invok.map(*this), sp.retindex, sp.derived_by_arg));
			invok = GiNaC::ex_to<GiNaC::GiNaCMultiRetCallback>(res).get_struct().invok;

			if (code->resolve_multi_return_call(invok) < 0)
			{
				code->multi_return_calls.push_back(invok);
			}
			return res;
		}
		else
			return inp.map(*this);
	}

	// Core placeholder-expansion pass: walks a user-supplied symbolic expression and replaces every
	// high-level placeholder function (field(...), nondimfield(...), testfunction(...),
	// dimtestfunction(...), scale(...), test_scale(...), eval_flag(...), eval_in_domain(...),
	// python_multi_cb_function(...), delayed-callback expansions, ...) by the concrete GiNaC
	// expression it stands for: a dimensional/nondimensional ShapeExpansion or TestFunction,
	// possibly multiplied by its scaling factor, resolved in the correct FiniteElementCode/domain
	// (`code->resolve_corresponding_code`, respecting eval_in_domain(...) to jump to bulk/interface
	// codes). Recurses into itself (via a freshly constructed instance bound to the resolved domain)
	// so cross-domain expressions are expanded fully. `where` records the code-generation context
	// (residual/Jacobian/...) which affects some scaling factors; `repl_count` is incremented on every
	// substitution so callers can detect when no further expansion is possible.
	GiNaC::ex ReplaceFieldsToNonDimFields::operator()(const GiNaC::ex &inp)
	{
		std::string fieldname;
		//		std::cout <<"ENTERING "<<inp <<std::endl <<std::flush;
		if (is_ex_the_function(inp, expressions::eval_in_domain))
		{
			FiniteElementCode *mycode = code->resolve_corresponding_code(inp, &fieldname, NULL);
			if (pyoomph_verbose)
				std::cout << "Expanding eval_in_domain (this " << code << " , domain " << mycode << " ): " << inp << " || fieldname: " << fieldname << std::endl;

			GiNaC::GiNaCPlaceHolderResolveInfo resolve_info = GiNaC::ex_to<GiNaC::GiNaCPlaceHolderResolveInfo>(inp.op(1));
			auto tags = resolve_info.get_struct().tags;
			GiNaC::ex extra_test_scale_due_to_facets = 1;
			for (auto &t : tags)
			{
				if (t == "domain:+")
				{
					extra_test_scale_due_to_facets = 1 / mycode->get_scaling("spatial", false);
					break;
				}
				if (t == "domain:-")
				{
					extra_test_scale_due_to_facets = 1 / mycode->get_scaling("spatial", false);
					break;
				}
			}
			GiNaC::ex expr = inp.op(0);
			if (pyoomph_verbose)
				std::cout << "Evaluation expression " << expr << " @ CODE " << mycode << std::endl;
			repl_count++;
			// Go through all fields and nondim fields
			ReplaceFieldsToNonDimFields repl(mycode, where);
			repl.extra_test_scale = this->extra_test_scale * extra_test_scale_due_to_facets;
			return repl(expr).map(*this);
		}
		/*   else if (is_ex_the_function(inp,expressions::eval_in_past))
			{
			  GiNaC::ex expr=inp.op(0);
			  GiNaC::ex index_e=inp.op(1);
			  if not (GiNaC::is_a<GiNaC::numeric>(index_e))
			  {
			   throw_runtime_error("Cannot use evaluate_in_paste(expression,timeoffet) with a non numeric timeoffset");
			  }
			  GiNaC::numeric index_n=GiNaC::ex_to<GiNaC::numeric>(index_e);
			  if (index_n.is_zero())
			  {
				 repl_count++;
				return expr.map(*this);
			  }
			  else if (index_n.is_pos_integer())
			  {
				 repl_count++;
				 throw_runtime_error("TODO: Eval in past!");
			  }
			  else
			  {
			   throw_runtime_error("Cannot use evaluate_in_paste(expression,timeoffet) with a non positive or non-integer timeoffset");
			  }
			}*/
		else if (is_ex_the_function(inp, expressions::field))
		{
			FiniteElementFieldTagInfo taginfo;
			FiniteElementCode *mycode = code->resolve_corresponding_code(inp, &fieldname, &taginfo);
			GiNaC::ex scale = mycode->get_scaling(fieldname);
			code->expanded_scales["field(" + mycode->get_domain_name() + "): " + fieldname] = scale;
			if (pyoomph_verbose)
				std::cout << "Expanding field " << fieldname << " @ CODE " << mycode << std::endl;
			repl_count++;
			if (mycode->get_field_by_name(fieldname))
			{
				if (pyoomph_verbose)
					std::cout << "Found field by name in code " << mycode << " NO JACOBIAN IS " << taginfo.no_jacobian << " NO HESSIAN IS " << taginfo.no_hessian << std::endl;
				auto *coordsys = mycode->get_coordinate_system();
				return scale * coordsys->get_mode_expansion_of_var_or_test(mycode, fieldname, true, true, mycode->get_field_by_name(fieldname)->get_shape_expansion(taginfo.no_jacobian, taginfo.no_hessian), where, taginfo.expansion_mode);
			}

			std::tuple<std::string, const bool, const GiNaC::ex, FiniteElementCode *, bool, bool, std::string> cache_key = std::make_tuple(fieldname, true, inp, code, taginfo.no_jacobian, taginfo.no_hessian, where);
			GiNaC::ex res;
			bool add_to_cache;
			if (false && mycode->expanded_additional_field_cache.count(cache_key)) //Do not use the cache for the moment
			{
				res = mycode->expanded_additional_field_cache[cache_key];
				add_to_cache = false;
			}
			else
			{
				res = mycode->expand_additional_field(fieldname, true, inp, code, taginfo.no_jacobian, taginfo.no_hessian, where);
				add_to_cache = true;
			}
			if (pyoomph_verbose)
				std::cout << "expand_additional_field of " << inp << " @ CODE " << mycode << " gave " << res << std::endl;

			//			res=res.map(*this);
			ReplaceFieldsToNonDimFields further_expansion(mycode, where);
			res = res.map(further_expansion);
			if (add_to_cache)
			{
				mycode->expanded_additional_field_cache[cache_key] = res;
			}
			if (pyoomph_verbose)
				std::cout << "which was further expanded from " << inp << " @ CODE " << mycode << " to " << res << std::endl;

			return res;
		}
		else if (is_ex_the_function(inp, expressions::eval_flag))
		{
			std::ostringstream os;
			os << inp.op(0);
			std::string flag = os.str();
			GiNaC::ex ret = code->eval_flag(flag);
			if (pyoomph_verbose)
				std::cout << "Expanding flag " + flag + " gives " << ret << std::endl;
			return ret;
		}
		else if (is_ex_the_function(inp, expressions::scale))
		{
			FiniteElementCode *mycode = code->resolve_corresponding_code(inp, &fieldname, NULL);
			GiNaC::ex scale = mycode->get_scaling(fieldname);
			code->expanded_scales["scale(" + mycode->get_domain_name() + "): " + fieldname] = scale;
			//				std::cout << "EXPANDED scaLE FACTOR " << fieldname << "  "  << scale << "  " << mycode << " " << code << std::endl;
			repl_count++;
			return scale;
		}
		else if (is_ex_the_function(inp, expressions::test_scale))
		{
			FiniteElementCode *mycode = code->resolve_corresponding_code(inp, &fieldname, NULL);
			GiNaC::ex scale = mycode->get_scaling(fieldname, true);
			code->expanded_scales["testscale(" + mycode->get_domain_name() + "): " + fieldname] = scale;
			//				std::cout << "EXPANDED scaLE FACTOR " << fieldname << "  "  << scale << "  " << mycode << " " << code << std::endl;
			repl_count++;
			return scale;
		}
		else if (is_ex_the_function(inp, expressions::nondimfield))
		{
			FiniteElementFieldTagInfo taginfo;
			FiniteElementCode *mycode = code->resolve_corresponding_code(inp, &fieldname, &taginfo);
			if (pyoomph_verbose)
				std::cout << "Expanding nondim field " << fieldname << std::endl;
			repl_count++;
			if (mycode->get_field_by_name(fieldname))
			{
				if (pyoomph_verbose)
					std::cout << "Found field by name in code " << mycode << std::endl;
				auto *coordsys = mycode->get_coordinate_system();
				return coordsys->get_mode_expansion_of_var_or_test(mycode, fieldname, true, false, mycode->get_field_by_name(fieldname)->get_shape_expansion(taginfo.no_jacobian, taginfo.no_hessian), where, taginfo.expansion_mode);
			}
			std::tuple<std::string, const bool, const GiNaC::ex, FiniteElementCode *, bool, bool, std::string> cache_key = std::make_tuple(fieldname, false, inp, code, taginfo.no_jacobian, taginfo.no_hessian, where);
			GiNaC::ex res;
			bool add_to_cache;
			if (false && mycode->expanded_additional_field_cache.count(cache_key)) // Do not use the cache for the moment
			{
				res = mycode->expanded_additional_field_cache[cache_key];
				add_to_cache = false;
			}
			else
			{
				res = mycode->expand_additional_field(fieldname, false, inp, code, taginfo.no_jacobian, taginfo.no_hessian, where);
				add_to_cache = true;
			}
			ReplaceFieldsToNonDimFields further_expansion(mycode, where);
			res = res.map(further_expansion);
			if (add_to_cache)
			{
				mycode->expanded_additional_field_cache[cache_key] = res;
			}
			return res;
		}
		else if (is_ex_the_function(inp, expressions::dimtestfunction))
		{
			FiniteElementCode *mycode = code->resolve_corresponding_code(inp, &fieldname, NULL);
			GiNaC::ex scale = mycode->get_scaling(fieldname, true);
			scale *= this->extra_test_scale;
			code->expanded_scales["test(" + mycode->get_domain_name() + "): " + fieldname] = scale;
			// Check if + or - is used. If so, divide by scale spatial
			GiNaC::GiNaCPlaceHolderResolveInfo resolve_info = GiNaC::ex_to<GiNaC::GiNaCPlaceHolderResolveInfo>(inp.op(1));
			auto tags = resolve_info.get_struct().tags;
			for (auto &t : tags)
			{
				if (t == "domain:+")
				{
					scale /= mycode->get_scaling("spatial", false);
					break;
				}
				if (t == "domain:-")
				{
					scale /= mycode->get_scaling("spatial", false);
					break;
				}
			}
			if (pyoomph_verbose)
				std::cout << "Expanding dim testfunction " << fieldname << std::endl;
			repl_count++;
			if (mycode->get_field_by_name(fieldname))
			{
				if (pyoomph_verbose)
					std::cout << "Found testfunction by name in code " << mycode << std::endl;
				auto *coordsys = mycode->get_coordinate_system();
				return scale * coordsys->get_mode_expansion_of_var_or_test(mycode, fieldname, false, true, mycode->get_field_by_name(fieldname)->get_test_function(), where, 0);
			}
			GiNaC::ex res = scale * mycode->expand_additional_testfunction(fieldname, inp, code);
			ReplaceFieldsToNonDimFields further_expansion(mycode, where);
			res = res.map(further_expansion);
			return res;
		}
		else if (is_ex_the_function(inp, expressions::testfunction))
		{
			FiniteElementCode *mycode = code->resolve_corresponding_code(inp, &fieldname, NULL);
			if (pyoomph_verbose)
				std::cout << "Expanding testfunction " << fieldname << std::endl;
			repl_count++;
			if (mycode->get_field_by_name(fieldname))
			{
				if (pyoomph_verbose)
					std::cout << "Found testfunction by name in code " << mycode << std::endl;
				auto *coordsys = mycode->get_coordinate_system();
				return coordsys->get_mode_expansion_of_var_or_test(mycode, fieldname, false, false, mycode->get_field_by_name(fieldname)->get_test_function(), where, 0);
			}
			GiNaC::ex res = mycode->expand_additional_testfunction(fieldname, inp, code);
			ReplaceFieldsToNonDimFields further_expansion(mycode, where);
			res = res.map(further_expansion);
			return res;
		}
		else if (GiNaC::is_a<GiNaC::GiNaCDelayedPythonCallbackExpansion>(inp))
		{
			//   std::cout << "FOUND DELAYED CALLBACK" <<std::endl << std::flush;
			GiNaC::ex func_res = GiNaC::ex_to<GiNaC::GiNaCDelayedPythonCallbackExpansion>(inp).get_struct().cme->f();
			//   std::cout << "FUNC RES" << func_res << std::endl << std::flush;
			return func_res.map(*this);
		}
		else if (is_ex_the_function(inp, expressions::python_multi_cb_function))
		{
			GiNaC::ex invok = inp.map(*this);
			std::cout << "ON INVOK " << invok << std::endl
					  << std::flush;
			if (GiNaC::is_a<GiNaC::lst>(invok))
				return invok; // We might be able to evaluate directly if all args are replaced by constants
			int numret = GiNaC::ex_to<GiNaC::numeric>(invok.op(2)).to_int();
			//int numargs = GiNaC::ex_to<GiNaC::lst>(invok.op(1)).nops();
			CustomMultiReturnExpressionBase *func = GiNaC::ex_to<GiNaC::GiNaCCustomMultiReturnExpressionWrapper>(invok.op(0)).get_struct().cme;
			std::string ccode = func->_get_c_code();
			if (ccode != "")
			{
				unsigned index = code->multi_return_ccodes.size();
				if (code->multi_return_ccodes.count(func))
				{
					if (code->multi_return_ccodes[func].second != ccode)
					{
						throw_runtime_error("The same multi-ret generates different C code at successive calls!");
					}
				}
				else
				{
					code->multi_return_ccodes[func] = std::make_pair(index, ccode);
				}
			}
			std::vector<GiNaC::ex> ret;
			for (int i = 0; i < numret; i++)
			{
				ret.push_back(GiNaC::GiNaCMultiRetCallback(MultiRetCallback(code, invok, i)));
			}
			return GiNaC::lst(ret.begin(), ret.end()).map(*this);
		}
		else if (GiNaC::is_a<GiNaC::GiNaCMultiRetCallback>(inp))
		{
			const auto &wrappi = GiNaC::ex_to<GiNaC::GiNaCMultiRetCallback>(inp).get_struct();
			GiNaC::ex invok = wrappi.invok.map(*this);
			std::set<TestFunction> sub_testfuncs = code->get_all_test_functions_in(invok);
			if (!sub_testfuncs.empty())
			{
				std::ostringstream oss;
				oss << invok;
				throw_runtime_error("Multi-return functions may not have testfunctions as arguments!\nHappened in:\n" + oss.str());
			}
			return GiNaC::GiNaCMultiRetCallback(MultiRetCallback(wrappi.code, invok, wrappi.retindex, wrappi.derived_by_arg));
		}

		return inp.map(*this);
	}

	// GiNaC tree-mapper that substitutes one FiniteElementField for another wherever it appears as a
	// ShapeExpansion or TestFunction, keeping all other properties (time-derivative order, basis,
	// nodal-coordinate-derivative direction, jacobian/hessian flags, expansion mode) unchanged. Used
	// e.g. to re-target an expression originally written for one field onto an equivalent field
	// defined on a different (but structurally identical) domain. Only un-spatially-derived basis
	// functions are supported; remapping a spatially derived ShapeExpansion/TestFunction is not
	// implemented and raises an error.
	class RemapFieldsInExpression : public GiNaC::map_function
	{
	protected:
		std::map<FiniteElementField *, FiniteElementField *> remapping;

	public:
		RemapFieldsInExpression(std::map<FiniteElementField *, FiniteElementField *> remap) : remapping(remap) {}
		GiNaC::ex operator()(const GiNaC::ex &inp) override
		{
			if (GiNaC::is_a<GiNaC::GiNaCShapeExpansion>(inp))
			{
				auto &se = GiNaC::ex_to<GiNaC::GiNaCShapeExpansion>(inp).get_struct();
				if (!remapping.count(se.field))
					return inp;
				else
				{
					FiniteElementField *newfield = remapping[se.field];
					if (se.field->get_space()->get_basis() != se.basis)
					{
						throw_runtime_error("Cannot remap spatially derived ShapeExpansion yet");
					}
					ShapeExpansion repl(newfield, se.dt_order, newfield->get_space()->get_basis(), se.dt_scheme, se.is_derived, se.nodal_coord_dir);
					repl.no_jacobian = se.no_jacobian;
					repl.no_hessian = se.no_hessian;
					repl.expansion_mode = se.expansion_mode;
					repl.is_derived_other_index = se.is_derived_other_index;
					return 0 + GiNaC::GiNaCShapeExpansion(repl);
				}
			}
			else if (GiNaC::is_a<GiNaC::GiNaCMultiRetCallback>(inp))
			{
				const auto &sp = GiNaC::ex_to<GiNaC::GiNaCMultiRetCallback>(inp).get_struct();
				GiNaC::ex invok = expressions::python_multi_cb_function(sp.invok.op(0), sp.invok.op(1).map(*this), sp.invok.op(2));
				GiNaC::ex res = GiNaC::GiNaCMultiRetCallback(pyoomph::MultiRetCallback(sp.code, invok.map(*this), sp.retindex, sp.derived_by_arg));
				invok = GiNaC::ex_to<GiNaC::GiNaCMultiRetCallback>(res).get_struct().invok;

				/*if (code->resolve_multi_return_call(invok) < 0)
				{
					code->multi_return_calls.push_back(invok);
				}*/
				return res;
			}
			else if (GiNaC::is_a<GiNaC::GiNaCTestFunction>(inp))
			{
				auto &se = GiNaC::ex_to<GiNaC::GiNaCTestFunction>(inp).get_struct();
				if (!remapping.count(se.field))
					return inp;
				else
				{
					FiniteElementField *newfield = remapping[se.field];
					if (se.field->get_space()->get_basis() != se.basis)
					{
						throw_runtime_error("Cannot remap spatially derived TestFunctions yet");
					}
					TestFunction repl(newfield, newfield->get_space()->get_basis(), se.nodal_coord_dir);
					return 0 + GiNaC::GiNaCTestFunction(repl);
				}
			}
			else
			{
				return inp.map(*this);
			}
		}
	};

	//////////////

	// The following operator==/operator< overloads implement value equality and a strict weak
	// ordering (lexicographic over all data members, most significant first) for the small "tag"
	// structs used as GiNaC custom structures (SpatialIntegralSymbol, ElementSizeSymbol,
	// NodalDeltaSymbol, NormalSymbol, SubExpression, MultiRetCallback, ShapeExpansion, TestFunction).
	// GiNaC needs these so that the structures can be stored/deduplicated in std::set/std::map (e.g.
	// the sets of "required shape expansions" collected while emitting code) and so that structurally
	// identical symbols compare equal regardless of where in the expression tree they were created.
	bool operator==(const SpatialIntegralSymbol &lhs, const SpatialIntegralSymbol &rhs)
	{
		return lhs.get_code() == rhs.get_code() && 
		       lhs.is_lagrangian() == rhs.is_lagrangian() && 
			   lhs.is_derived() == rhs.is_derived() && 
			   lhs.get_derived_direction() == rhs.get_derived_direction() && 
			   lhs.is_derived2() == rhs.is_derived2() && 
			   lhs.get_derived_direction2() == rhs.get_derived_direction2() && 
			   lhs.is_derived_by_lshape2() == rhs.is_derived_by_lshape2() && 
			   lhs.expansion_mode == rhs.expansion_mode && 
			   lhs.no_jacobian == rhs.no_jacobian && 
			   lhs.no_hessian == rhs.no_hessian && 
			   lhs.history_step == rhs.history_step &&
			   lhs.simple_unity_integral == rhs.simple_unity_integral;
	}
	bool operator<(const SpatialIntegralSymbol &lhs, const SpatialIntegralSymbol &rhs)
	{		
		return lhs.get_code() < rhs.get_code() || 
		       (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() < rhs.is_lagrangian()) || 
			   (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() < rhs.is_derived()) || 
			   (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() < rhs.get_derived_direction()) ||			   
			   (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.is_derived2() < rhs.is_derived2()) ||
			   (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.is_derived2() == rhs.is_derived2() && lhs.get_derived_direction2() < rhs.get_derived_direction2()) ||
			   (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.is_derived2() == rhs.is_derived2() && lhs.get_derived_direction2() == rhs.get_derived_direction2() && lhs.is_derived_by_lshape2() < rhs.is_derived_by_lshape2()) ||
			   (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.is_derived2() == rhs.is_derived2() && lhs.get_derived_direction2() == rhs.get_derived_direction2() && lhs.is_derived_by_lshape2() == rhs.is_derived_by_lshape2() && lhs.expansion_mode < rhs.expansion_mode) ||
			   //(lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.is_derived2() == rhs.is_derived2() && lhs.get_derived_direction2() == rhs.get_derived_direction2() && lhs.is_derived_by_lshape2() < rhs.is_derived_by_lshape2()) ||
			   (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.is_derived2() == rhs.is_derived2() && lhs.get_derived_direction2() == rhs.get_derived_direction2() && lhs.is_derived_by_lshape2() == rhs.is_derived_by_lshape2() && lhs.expansion_mode == rhs.expansion_mode && lhs.no_jacobian < rhs.no_jacobian) ||
			   //(lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.is_derived2() == rhs.is_derived2() && lhs.get_derived_direction2() == rhs.get_derived_direction2() && lhs.is_derived_by_lshape2() < rhs.is_derived_by_lshape2()) ||
			   (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.is_derived2() == rhs.is_derived2() && lhs.get_derived_direction2() == rhs.get_derived_direction2() && lhs.is_derived_by_lshape2() == rhs.is_derived_by_lshape2() && lhs.expansion_mode == rhs.expansion_mode && lhs.no_jacobian == rhs.no_jacobian && lhs.no_hessian < rhs.no_hessian) || 
			   (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.is_derived2() == rhs.is_derived2() && lhs.get_derived_direction2() == rhs.get_derived_direction2() && lhs.is_derived_by_lshape2() == rhs.is_derived_by_lshape2() && lhs.expansion_mode == rhs.expansion_mode && lhs.no_jacobian == rhs.no_jacobian && lhs.no_hessian == rhs.no_hessian && lhs.history_step<rhs.history_step) ||
			   (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.is_derived2() == rhs.is_derived2() && lhs.get_derived_direction2() == rhs.get_derived_direction2() && lhs.is_derived_by_lshape2() == rhs.is_derived_by_lshape2() && lhs.expansion_mode == rhs.expansion_mode && lhs.no_jacobian == rhs.no_jacobian && lhs.no_hessian == rhs.no_hessian && lhs.history_step==rhs.history_step && lhs.simple_unity_integral<rhs.simple_unity_integral) 
			;
	}

	bool operator==(const ElementSizeSymbol &lhs, const ElementSizeSymbol &rhs)
	{
		return lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.is_derived2() == rhs.is_derived2() && lhs.get_derived_direction2() == rhs.get_derived_direction2() && lhs.is_with_coordsys() == rhs.is_with_coordsys() && lhs.is_derived_by_lshape2() == rhs.is_derived_by_lshape2();
	}
	bool operator<(const ElementSizeSymbol &lhs, const ElementSizeSymbol &rhs)
	{
		return lhs.get_code() < rhs.get_code() || (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() < rhs.is_lagrangian()) || (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() < rhs.is_derived()) || (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() < rhs.get_derived_direction()) ||
			   (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.is_derived2() < rhs.is_derived2()) ||
			   (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.is_derived2() == rhs.is_derived2() && lhs.get_derived_direction2() < rhs.get_derived_direction2()) ||
			   (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.is_derived2() == rhs.is_derived2() && lhs.get_derived_direction2() == rhs.get_derived_direction2() && lhs.is_with_coordsys() < rhs.is_with_coordsys()) ||
			   (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.is_derived2() == rhs.is_derived2() && lhs.get_derived_direction2() == rhs.get_derived_direction2() && lhs.is_with_coordsys() == rhs.is_with_coordsys() && lhs.is_derived_by_lshape2() < rhs.is_derived_by_lshape2());
	}

	bool operator==(const NodalDeltaSymbol &lhs, const NodalDeltaSymbol &rhs)
	{
		return lhs.get_code() == rhs.get_code();
	}
	bool operator<(const NodalDeltaSymbol &lhs, const NodalDeltaSymbol &rhs)
	{
		return lhs.get_code() < rhs.get_code();
	}
      
	bool operator==(const NormalSymbol &lhs, const NormalSymbol &rhs)
	{
		return lhs.get_code() == rhs.get_code() && lhs.get_direction() == rhs.get_direction() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.get_derived_direction2() == rhs.get_derived_direction2() && lhs.is_derived_by_lshape2() == rhs.is_derived_by_lshape2()  && lhs.expansion_mode == rhs.expansion_mode && lhs.no_jacobian == rhs.no_jacobian && lhs.no_hessian == rhs.no_hessian;
	}
	bool operator<(const NormalSymbol &lhs, const NormalSymbol &rhs)
	{
		return lhs.get_code() < rhs.get_code() 
		 || (lhs.get_code() == rhs.get_code() && lhs.get_direction() < rhs.get_direction()) 
		 || (lhs.get_code() == rhs.get_code() && lhs.get_direction() == rhs.get_direction() && lhs.get_derived_direction() < rhs.get_derived_direction()) 
		 || (lhs.get_code() == rhs.get_code() && lhs.get_direction() == rhs.get_direction() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.get_derived_direction2() < rhs.get_derived_direction2()) 
		 || (lhs.get_code() == rhs.get_code() && lhs.get_direction() == rhs.get_direction() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.get_derived_direction2() == rhs.get_derived_direction2() && lhs.is_derived_by_lshape2() < rhs.is_derived_by_lshape2()) 		
		 || (lhs.get_code() == rhs.get_code() && lhs.get_direction() == rhs.get_direction() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.get_derived_direction2() == rhs.get_derived_direction2() && lhs.is_derived_by_lshape2() == rhs.is_derived_by_lshape2() && lhs.expansion_mode < rhs.expansion_mode) 
		 || (lhs.get_code() == rhs.get_code() && lhs.get_direction() == rhs.get_direction() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.get_derived_direction2() == rhs.get_derived_direction2() && lhs.is_derived_by_lshape2() == rhs.is_derived_by_lshape2() && lhs.expansion_mode == rhs.expansion_mode && lhs.no_jacobian < rhs.no_jacobian) 
		 || (lhs.get_code() == rhs.get_code() && lhs.get_direction() == rhs.get_direction() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.get_derived_direction2() == rhs.get_derived_direction2() && lhs.is_derived_by_lshape2() == rhs.is_derived_by_lshape2() && lhs.expansion_mode == rhs.expansion_mode && lhs.no_jacobian == rhs.no_jacobian && lhs.no_hessian < rhs.no_hessian);
	}

	bool operator<(const SubExpression &lhs, const SubExpression &rhs)
	{
		return GiNaC::ex_is_less()(lhs.expr, rhs.expr);
	}

	bool operator==(const SubExpression &lhs, const SubExpression &rhs)
	{
		return lhs.expr.is_equal(rhs.expr);
	}

	bool operator<(const MultiRetCallback &lhs, const MultiRetCallback &rhs)
	{
		return GiNaC::ex_is_less()(lhs.invok, rhs.invok) || (lhs.invok.is_equal(rhs.invok) && lhs.retindex < rhs.retindex) || (lhs.invok.is_equal(rhs.invok) && lhs.retindex == rhs.retindex && lhs.derived_by_arg < rhs.derived_by_arg) || (lhs.invok.is_equal(rhs.invok) && lhs.retindex == rhs.retindex && lhs.derived_by_arg == rhs.derived_by_arg && lhs.code < rhs.code);
	}

	bool operator==(const MultiRetCallback &lhs, const MultiRetCallback &rhs)
	{
		return lhs.invok.is_equal(rhs.invok) && lhs.retindex == rhs.retindex && lhs.derived_by_arg == rhs.derived_by_arg && lhs.code == rhs.code;
	}
	bool operator==(const ShapeExpansion &lhs, const ShapeExpansion &rhs)
	{
		return lhs.field == rhs.field && lhs.dt_order == rhs.dt_order && lhs.basis == rhs.basis && lhs.is_derived == rhs.is_derived && lhs.is_derived_other_index == rhs.is_derived_other_index && lhs.basis == rhs.basis && lhs.nodal_coord_dir == rhs.nodal_coord_dir && lhs.time_history_index == rhs.time_history_index && (lhs.dt_order == 0 || lhs.dt_scheme == rhs.dt_scheme) && (lhs.no_jacobian == rhs.no_jacobian) && (lhs.no_hessian == rhs.no_hessian) && (lhs.expansion_mode == rhs.expansion_mode) && (lhs.nodal_coord_dir2 == rhs.nodal_coord_dir2);
	}
	bool operator<(const ShapeExpansion &lhs, const ShapeExpansion &rhs)
	{
		return lhs.field < rhs.field || (lhs.field == rhs.field && lhs.dt_order < rhs.dt_order) || (lhs.field == rhs.field && lhs.dt_order == rhs.dt_order && lhs.basis < rhs.basis) || (lhs.field == rhs.field && lhs.dt_order == rhs.dt_order && lhs.basis == rhs.basis && lhs.is_derived < rhs.is_derived) || (lhs.field == rhs.field && lhs.dt_order == rhs.dt_order && lhs.basis == rhs.basis && lhs.is_derived == rhs.is_derived && lhs.is_derived_other_index < rhs.is_derived_other_index) || (lhs.field == rhs.field && lhs.dt_order == rhs.dt_order && lhs.basis == rhs.basis && lhs.is_derived == rhs.is_derived && lhs.is_derived_other_index == rhs.is_derived_other_index && lhs.nodal_coord_dir < rhs.nodal_coord_dir) || (lhs.field == rhs.field && lhs.dt_order == rhs.dt_order && lhs.basis == rhs.basis && lhs.is_derived == rhs.is_derived && lhs.is_derived_other_index == rhs.is_derived_other_index && lhs.nodal_coord_dir == rhs.nodal_coord_dir && lhs.time_history_index < rhs.time_history_index) || (lhs.field == rhs.field && lhs.dt_order == rhs.dt_order && lhs.basis == rhs.basis && lhs.is_derived == rhs.is_derived && lhs.is_derived_other_index == rhs.is_derived_other_index && lhs.nodal_coord_dir == rhs.nodal_coord_dir && lhs.time_history_index == rhs.time_history_index && (lhs.dt_order > 0 && lhs.dt_scheme < rhs.dt_scheme)) || (lhs.field == rhs.field && lhs.dt_order == rhs.dt_order && lhs.basis == rhs.basis && lhs.is_derived == rhs.is_derived && lhs.is_derived_other_index == rhs.is_derived_other_index && lhs.nodal_coord_dir == rhs.nodal_coord_dir && lhs.time_history_index == rhs.time_history_index && (lhs.dt_order == 0 || lhs.dt_scheme == rhs.dt_scheme) && (lhs.no_jacobian < rhs.no_jacobian)) || (lhs.field == rhs.field && lhs.dt_order == rhs.dt_order && lhs.basis == rhs.basis && lhs.is_derived == rhs.is_derived && lhs.is_derived_other_index == rhs.is_derived_other_index && lhs.nodal_coord_dir == rhs.nodal_coord_dir && lhs.time_history_index == rhs.time_history_index && (lhs.dt_order == 0 || lhs.dt_scheme == rhs.dt_scheme) && (lhs.no_jacobian == rhs.no_jacobian) && (lhs.no_hessian < rhs.no_hessian)) || (lhs.field == rhs.field && lhs.dt_order == rhs.dt_order && lhs.basis == rhs.basis && lhs.is_derived == rhs.is_derived && lhs.is_derived_other_index == rhs.is_derived_other_index && lhs.nodal_coord_dir == rhs.nodal_coord_dir && lhs.time_history_index == rhs.time_history_index && (lhs.dt_order == 0 || lhs.dt_scheme == rhs.dt_scheme) && (lhs.no_jacobian == rhs.no_jacobian) && (lhs.no_hessian == rhs.no_hessian) && (lhs.expansion_mode < rhs.expansion_mode)) || (lhs.field == rhs.field && lhs.dt_order == rhs.dt_order && lhs.basis == rhs.basis && lhs.is_derived == rhs.is_derived && lhs.is_derived_other_index == rhs.is_derived_other_index && lhs.nodal_coord_dir == rhs.nodal_coord_dir && lhs.time_history_index == rhs.time_history_index && (lhs.dt_order == 0 || lhs.dt_scheme == rhs.dt_scheme) && (lhs.no_jacobian == rhs.no_jacobian) && (lhs.no_hessian == rhs.no_hessian) && (lhs.expansion_mode == rhs.expansion_mode) && (lhs.nodal_coord_dir2 < rhs.nodal_coord_dir2));
	}

	bool operator==(const TestFunction &lhs, const TestFunction &rhs)
	{
		return lhs.field == rhs.field && lhs.basis == rhs.basis && lhs.nodal_coord_dir == rhs.nodal_coord_dir && lhs.is_derived_other_index == rhs.is_derived_other_index && lhs.nodal_coord_dir2 == rhs.nodal_coord_dir2;
	}
	bool operator<(const TestFunction &lhs, const TestFunction &rhs)
	{
		return lhs.field < rhs.field || (lhs.field == rhs.field && lhs.basis < rhs.basis) || (lhs.field == rhs.field && lhs.basis == rhs.basis && lhs.nodal_coord_dir < rhs.nodal_coord_dir) || (lhs.field == rhs.field && lhs.basis == rhs.basis && lhs.nodal_coord_dir == rhs.nodal_coord_dir && lhs.is_derived_other_index < rhs.is_derived_other_index) || (lhs.field == rhs.field && lhs.basis == rhs.basis && lhs.nodal_coord_dir == rhs.nodal_coord_dir && lhs.is_derived_other_index == rhs.is_derived_other_index && lhs.nodal_coord_dir2 < rhs.nodal_coord_dir2);
	}

	// Checks whether the GiNaC symbol `s` is the raw position-coordinate symbol of the nodal
	// position field on some accessible domain (this element itself, its bulk element(s), or the
	// opposite-interface element and its bulk), so that differentiating w.r.t. `s` can be
	// interpreted as a derivative w.r.t. the moving mesh coordinates. If `domain_to_check` is NULL,
	// recursively probes __current_code and its related domains (bulk / bulk-of-bulk / opposite
	// interface / opposite interface's bulk); otherwise checks only the given domain. Returns the
	// FiniteElementCode on which `s` was found as a position symbol, or NULL if it is not one.
	FiniteElementCode *ShapeExpansion::can_be_a_positional_derivative_symbol(const GiNaC::symbol &s, FiniteElementCode *domain_to_check) const
	{
		if (!domain_to_check)
		{
			if (!__current_code)
				throw_runtime_error("DD");
			FiniteElementCode *res = can_be_a_positional_derivative_symbol(s, __current_code);
			if (res)
				return res;
			if (__current_code->get_bulk_element())
			{
				res = can_be_a_positional_derivative_symbol(s, __current_code->get_bulk_element());
				if (res)
					return res;
				if (__current_code->get_bulk_element()->get_bulk_element())
				{
					res = can_be_a_positional_derivative_symbol(s, __current_code->get_bulk_element()->get_bulk_element());
					if (res)
						return res;
				}
			}
			if (__current_code->get_opposite_interface_code())
			{
				res = can_be_a_positional_derivative_symbol(s, __current_code->get_opposite_interface_code());
				if (res)
					return res;
				if (__current_code->get_opposite_interface_code()->get_bulk_element())
				{
					res = can_be_a_positional_derivative_symbol(s, __current_code->get_opposite_interface_code()->get_bulk_element());
					if (res)
						return res;
				}
			}
		}
		else
		{
			std::ostringstream oss;
			oss << s;
			std::string sname = oss.str();
			if (this->basis->get_space()->get_code() == domain_to_check)
			{
				auto *posspace = domain_to_check->get_my_position_space();
				for (auto *f : domain_to_check->get_fields_on_space(posspace))
				{
					if (f->get_name() == sname)
					{
						if (f->get_symbol() == s)
						{
							return domain_to_check;
						}
					}
				}
			}
		}
		return NULL;
	}

	// The following ShapeExpansion::get_*_name/str methods construct the C variable/array names used
	// in the generated code for a given shape expansion: e.g. get_dt_values_name names the array
	// holding the time-derivative-weighted nodal values, get_spatial_interpolation_name names the
	// (possibly nodal-coordinate-derivative-tagged) interpolated field value at the integration
	// point, and get_shape_string/get_nodal_data_string/get_num_nodes_str/get_nodal_index_str
	// delegate to the basis function / field / space to build the corresponding shape-function or
	// nodal-data array access expression as a string of C code.
	std::string ShapeExpansion::get_dt_values_name(FiniteElementCode *forcode) const
	{
		std::string code_type = forcode->get_owner_prefix(this->basis->get_space());
		std::string dtstring = "d" + std::to_string(this->dt_order) + "t" + std::to_string(time_history_index);
		if (this->dt_order > 0)
			dtstring += this->dt_scheme;
		return code_type + dtstring + "_" + this->field->get_name();
	}

	std::string ShapeExpansion::get_timedisc_scheme(FiniteElementCode *) const
	{
		return this->dt_scheme;
	}

	std::string ShapeExpansion::get_spatial_interpolation_name(FiniteElementCode *forcode) const
	{
		std::string code_type = forcode->get_owner_prefix(this->basis->get_space());
		std::string dtstring = "d" + std::to_string(this->dt_order) + "t" + std::to_string(time_history_index);
		if (this->dt_order > 0)
			dtstring += this->dt_scheme;
		if (nodal_coord_dir == -1)
		{
			return code_type + "intrp_" + dtstring + "_" + this->basis->get_dx_str() + "_" + this->field->get_name();
		}
		else if (nodal_coord_dir2 == -1)
		{
			return code_type + "intrp_" + dtstring + "_" + this->basis->get_dx_str() + "_COORDDIFF_" + std::to_string(this->nodal_coord_dir) + "_" + this->field->get_name() + "[" + (this->is_derived_other_index ? "l_shape2" : "l_shape") + "]";
		}
		else
		{
			// Second nodal-coordinate derivative: sort the two directions so that the same C array name
			// is used irrespective of the order the derivatives were taken in (mixed partials commute),
			// and remember whether we swapped so the l_shape/l_shape2 loop indices are swapped to match.
			int ind1 = this->nodal_coord_dir;
			int ind2 = this->nodal_coord_dir2;
			// TODO: Symmetrize?
			bool swapped = false;
			if (ind2 < ind1)
			{
				ind1 = this->nodal_coord_dir2;
				ind2 = this->nodal_coord_dir;
				swapped = true;
			}
			return code_type + "intrp_" + dtstring + "_" + this->basis->get_dx_str() + "_2ndCOORDDIFF_" + std::to_string(ind1) + "_" + std::to_string(ind2) + "_" + this->field->get_name() + "[" + (swapped ? "l_shape2" : "l_shape") + "][" + (swapped ? "l_shape" : "l_shape2") + "]";
		}
	}

	std::string ShapeExpansion::get_nodal_index_str(FiniteElementCode *forcode) const
	{
		return field->get_nodal_index_str(forcode);
	}

	// Bookkeeping of which (code, residual_index[, other field]) combinations a field actually
	// contributes to. This is filled in while residuals are added/derived (mark_*) and later queried
	// (has_*) to skip generating code for residual/Jacobian entries that are structurally zero,
	// avoiding wasted symbolic differentiation and C code for terms that can never be nonzero.
	bool FiniteElementField::has_residual_contribution_for_code(FiniteElementCode *code,unsigned residual_index)
	{
		return this->residual_contribution_for_code.count(code) > 0 && this->residual_contribution_for_code[code].count(residual_index) > 0;
	}

    bool FiniteElementField::has_jacobian_contribution_for_code(FiniteElementCode *code,unsigned residual_index, FiniteElementField *other)
	{
		return this->jacobian_contribution_for_code.count(code) > 0 && this->jacobian_contribution_for_code[code].count(residual_index) > 0 && this->jacobian_contribution_for_code[code][residual_index].count(other) > 0;
	}
    void FiniteElementField::mark_residual_contribution_for_code(FiniteElementCode *code,unsigned residual_index)
	{
	  if (!this->residual_contribution_for_code.count(code))
	  {
		this->residual_contribution_for_code[code]=std::set<unsigned>();
	  }
      this->residual_contribution_for_code[code].insert(residual_index);
	}
    void FiniteElementField::mark_jacobian_contribution_for_code(FiniteElementCode *code,unsigned residual_index, FiniteElementField *other)
	{
		if (!this->jacobian_contribution_for_code.count(code))
		{
			this->jacobian_contribution_for_code[code]=std::map<unsigned,std::set<FiniteElementField*>>();
		}
		if (!this->jacobian_contribution_for_code[code].count(residual_index))
		{
			this->jacobian_contribution_for_code[code][residual_index]=std::set<FiniteElementField*>();
		}
		this->jacobian_contribution_for_code[code][residual_index].insert(other);

	}

	// If a field is already defined on a bulk domain and merely re-exposed on an interface/corner
	// domain, returns the top-level (bulk) field it is equivalent to; otherwise returns itself.
	FiniteElementField * FiniteElementField::get_defined_on_domain_equivalent_field()
	{
		if (defined_on_domain_equivalent)
			return defined_on_domain_equivalent;
		else
			return this;
	}
	// Records that this field is just the interface/corner-level view of `equiv_field`, which is
	// defined on a bulk domain (see get_defined_on_domain_equivalent_field()).
    void FiniteElementField::set_defined_on_domain_equivalent_field(FiniteElementField *equiv_field)
	{
		this->defined_on_domain_equivalent = equiv_field;
	}

	std::string FiniteElementField::get_nodal_index_str(FiniteElementCode *forcode) const
	{
		std::string code_type = forcode->get_owner_prefix(space);
		return code_type + "nodalind_" + name;
	}

	std::string FiniteElementField::get_equation_str(FiniteElementCode *forcode, std::string index) const
	{
		std::string nodal_index = get_nodal_index_str(forcode);
		//     std::string eleminfo=forcode->get_elem_info_str(space);
		std::string eqnstr = space->get_eqn_number_str(forcode);
		return eqnstr + "[" + index + "][" + nodal_index + "]";
	}

	std::string FiniteElementField::get_hanginfo_str(FiniteElementCode *forcode) const
	{
		// The position space has its own dimension-indexed buffer; every other space (continuous,
		// DG, DL, D0) shares one unified buffer indexed by this field's global nodal-data index.
		if (dynamic_cast<PositionFiniteElementSpace *>(space))
		{
			return forcode->get_shape_info_str(space) + "->hanginfo_Pos[" + get_nodal_index_str(forcode) + "]";
		}
		else
		{
			return forcode->get_shape_info_str(space) + "->hanginfo[" + get_nodal_index_str(forcode) + "]";
		}
	}

	std::string ShapeExpansion::get_num_nodes_str(FiniteElementCode *forcode) const
	{
		return this->basis->get_space()->get_num_nodes_str(forcode);
	}

	std::string ShapeExpansion::get_shape_string(FiniteElementCode *forcode, std::string nodal_index) const
	{
		std::string shape_str = basis->get_shape_string(forcode, nodal_index);
		if (shape_str == "1")
			return shape_str;
		else
			return forcode->get_shape_info_str(basis->get_space()) + "->" + shape_str;
	}

	std::string ShapeExpansion::get_nodal_data_string(FiniteElementCode *forcode, std::string indexstr) const
	{
		if (this->dt_order > 0)
			return this->get_dt_values_name(forcode) + "[" + indexstr + "]";
		std::string nds = forcode->get_nodal_data_string(this->basis->get_space());
		return forcode->get_elem_info_str(this->basis->get_space()) + "->" + nds + "[" + indexstr + "][" + get_nodal_index_str(forcode) + "][" + std::to_string(time_history_index) + "]";
	}

	// BasisFunction represents an (undifferentiated or spatially differentiated) shape function of a
	// FiniteElementSpace. get_diff_x/X/S lazily create and cache, per Cartesian/Lagrangian/local
	// direction, the BasisFunction object representing its derivative w.r.t. Eulerian coordinate x,
	// Lagrangian coordinate X, or local element coordinate S respectively; the caches are owned by
	// this object and freed in the destructor. Subclasses (D1XBasisFunction and its Lagrangian/local
	// coordinate variants below) implement the actual differentiated shape function's C code names.
	std::string BasisFunction::get_c_varname(FiniteElementCode *, std::string test_index)
	{
		return "testfunction[" + test_index + "]";
	}

	BasisFunction *BasisFunction::get_diff_x(unsigned direction)
	{
		if (basis_deriv_x.empty())
		{
			basis_deriv_x.resize(3); // TODO: Let this depend on the space
			for (unsigned int i = 0; i < basis_deriv_x.size(); i++)
				basis_deriv_x[i] = new D1XBasisFunction(space, i);
		}
		return basis_deriv_x[direction];
	}

	BasisFunction *BasisFunction::get_diff_X(unsigned direction)
	{
		if (lagr_deriv_x.empty())
		{
			lagr_deriv_x.resize(3); // TODO: Let this depend on the space
			for (unsigned int i = 0; i < lagr_deriv_x.size(); i++)
				lagr_deriv_x[i] = new D1XBasisFunctionLagr(space, i);
		}
		return lagr_deriv_x[direction];
	}

	BasisFunction *BasisFunction::get_diff_S(unsigned direction)
	{
		if (local_coord_deriv_x.empty())
		{
			local_coord_deriv_x.resize(3); // TODO: Let this depend on the space
			for (unsigned int i = 0; i < local_coord_deriv_x.size(); i++)
				local_coord_deriv_x[i] = new D1XBasisFunctionLocalCoord(space, i);
		}
		return local_coord_deriv_x[direction];
	}

	std::string BasisFunction::get_shape_string(FiniteElementCode *, std::string nodal_index) const
	{
		if (space->get_shape_name()=="C2TB" || space->get_shape_name()=="C2" || space->get_shape_name()=="C1TB" || space->get_shape_name()=="C1")
		{
			return "shapes[SPACE_INDEX_" + space->get_shape_name() + "][" + nodal_index + "]";
		}
		else
		{
			return "shape_" + space->get_shape_name() + "[" + nodal_index + "]";
		}
	}

	BasisFunction::~BasisFunction()
	{
		for (unsigned int i = 0; i < basis_deriv_x.size(); i++)
			if (basis_deriv_x[i])
				delete basis_deriv_x[i];
		for (unsigned int i = 0; i < lagr_deriv_x.size(); i++)
			if (lagr_deriv_x[i])
				delete lagr_deriv_x[i];
	}
	std::string BasisFunction::to_string()
	{
		return "BASIS of " + space->get_name();
	}

	// Second spatial derivatives of basis functions (i.e. differentiating an already-once-differentiated
	// D1XBasisFunction again) are not supported by the code generator.
	BasisFunction *D1XBasisFunction::get_diff_x(unsigned)
	{
		throw_runtime_error("Cannot handle second order derivatives of basis functions yet");
	}

	BasisFunction *D1XBasisFunction::get_diff_X(unsigned)
	{
		throw_runtime_error("Cannot handle second order derivatives of basis functions yet");
	}

	BasisFunction *D1XBasisFunction::get_diff_S(unsigned)
	{
		throw_runtime_error("Cannot handle second order derivatives of basis functions yet");
	}

	std::string D1XBasisFunction::get_c_varname(FiniteElementCode *, std::string test_index)
	{
		return "dx_testfunction[" + test_index + "][" + std::to_string(direction) + "]";
	}
	std::string D1XBasisFunction::to_string()
	{
		std::string dx;
		if (direction == 0)
			dx = "d/dx ";
		else if (direction == 1)
			dx = "d/dy ";
		else if (direction == 2)
			dx = "d/dz ";
		return dx + "of BASIS of " + space->get_name();
	}

	std::string D1XBasisFunction::get_shape_string(FiniteElementCode *, std::string nodal_index) const
	{
		if (space->get_shape_name()=="C2TB" || space->get_shape_name()=="C2" || space->get_shape_name()=="C1TB" || space->get_shape_name()=="C1")
		{
			return "dx_shapes[SPACE_INDEX_" + space->get_shape_name() + "][" + nodal_index + "][" + std::to_string(direction) + "]";
		}
		else
		{
			return "dx_shape_" + space->get_shape_name() + "[" + nodal_index + "][" + std::to_string(direction) + "]";
		}
	}

	std::string D1XBasisFunctionLagr::get_c_varname(FiniteElementCode *, std::string test_index)
	{
		return "dX_testfunction[" + test_index + "][" + std::to_string(direction) + "]";
	}
	std::string D1XBasisFunctionLagr::to_string()
	{
		std::string dx;
		if (direction == 0)
			dx = "d/dX ";
		else if (direction == 1)
			dx = "d/dY ";
		else if (direction == 2)
			dx = "d/dZ ";
		return dx + "of BASIS of " + space->get_name();
	}

	std::string D1XBasisFunctionLagr::get_shape_string(FiniteElementCode *, std::string nodal_index) const
	{
		if (space->get_shape_name()=="C2TB" || space->get_shape_name()=="C2" || space->get_shape_name()=="C1TB" || space->get_shape_name()=="C1")
		{
			return "dX_shapes[SPACE_INDEX_" + space->get_shape_name() + "][" + nodal_index + "][" + std::to_string(direction) + "]";
		}
		else
		{
			return "dX_shape_" + space->get_shape_name() + "[" + nodal_index + "][" + std::to_string(direction) + "]";
		}
	}



	std::string D1XBasisFunctionLocalCoord::get_c_varname(FiniteElementCode *, std::string test_index)
	{
		return "dS_testfunction[" + test_index + "][" + std::to_string(direction) + "]";
	}
	std::string D1XBasisFunctionLocalCoord::to_string()
	{
		std::string dx;
		if (direction == 0)
			dx = "d/ds^1 ";
		else if (direction == 1)
			dx = "d/ds^2 ";
		else if (direction == 2)
			dx = "d/ds^3 ";
		return dx + "of BASIS of " + space->get_name();
	}

	std::string D1XBasisFunctionLocalCoord::get_shape_string(FiniteElementCode *, std::string nodal_index) const
	{
		if (space->get_shape_name()=="C2TB" || space->get_shape_name()=="C2" || space->get_shape_name()=="C1TB" || space->get_shape_name()=="C1")
		{
			return "dS_shapes[SPACE_INDEX_" + space->get_shape_name() + "][" + nodal_index + "][" + std::to_string(direction) + "]";
		}
		else
		{
			return "dS_shape_" + space->get_shape_name() + "[" + nodal_index + "][" + std::to_string(direction) + "]";
		}
	}



	// The following get_eqn_number_str/get_num_nodes_str overrides build the C expression used to
	// access, respectively, the local equation-number lookup array and the node count for a given
	// FiniteElementSpace within the generated element code: ordinary nodal spaces use the shared
	// "nodal_local_eqn"/"nnode_of_space[...]" oomph-lib element members, while the position space
	// (PositionFiniteElementSpace) uses the dedicated "pos_local_eqn"/"nnode", and the discontinuous
	// D0 space (one DoF per element, not per node) always reports a single "node".
	std::string FiniteElementSpace::get_eqn_number_str(FiniteElementCode *forcode) const
	{
		std::string eleminfo = forcode->get_elem_info_str(this);
		return eleminfo + "->nodal_local_eqn";
	}

	std::string FiniteElementSpace::get_num_nodes_str(FiniteElementCode *forcode) const
	{
		std::string eleminfo = forcode->get_elem_info_str(this);
		if (this->get_shape_name()=="DL") //TODO: Make this in another way
		{
			return eleminfo + "->" + "nnode_DL";
		}
		else return eleminfo + "->" + "nnode_of_space[SPACE_INDEX_" + this->get_shape_name() + "]";
	}

	std::string D0FiniteElementSpace::get_num_nodes_str(FiniteElementCode *) const
	{
		return "1";
	}

	std::string PositionFiniteElementSpace::get_num_nodes_str(FiniteElementCode *forcode) const
	{
		std::string eleminfo = forcode->get_elem_info_str(this);
		return eleminfo + "->" + "nnode";
	}

	std::string PositionFiniteElementSpace::get_eqn_number_str(FiniteElementCode *forcode) const
	{
		std::string eleminfo = forcode->get_elem_info_str(this);
		return eleminfo + "->pos_local_eqn";
	}

	// Emits C code that pre-computes, for every distinct time-derivative shape expansion in
	// `required_shapeexps` that lives on this space, an array of per-node time-derivative values
	// (weighted sum over the timestepper's history storage using the timestepper's finite-difference
	// weights for the requested order/scheme, with an optional "degraded start" scheme override for
	// the first BDF2 step). Declarations are emitted first (PYOOMPH_AQUIRE_ARRAY macro), then a
	// loop over nodes zero-initializes the arrays, followed by a loop over time-history storage that
	// accumulates weight*nodal_value into them. Lines are collected into vectors and sorted before
	// being written so the generated code (and thus its diff / compiled hash) is deterministic
	// regardless of the (unordered) traversal order of `required_shapeexps`.
	void FiniteElementSpace::write_nodal_time_interpolation(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, std::set<ShapeExpansion> &required_shapeexps)
	{
		bool hascontrib = false;
		std::string range = "";
		std::string shapeinfo = "";
		std::string eleminfo = "";
		std::set<std::string> handled;
		std::vector<std::string> decl_lines;
		for (auto &s : required_shapeexps)
		{
			if (s.dt_order == 0 || s.basis->get_space() != this)
				continue;
			if (s.basis->get_space() == this) // Only expand if it is my task
			{
				std::string varname = s.get_dt_values_name(for_code);
				if (!hascontrib)
				{
					range = s.get_num_nodes_str(for_code);
					shapeinfo = for_code->get_shape_info_str(s.basis->get_space());
					eleminfo = for_code->get_elem_info_str(s.basis->get_space());
					hascontrib = true;
				}
				if (handled.count(varname))
					continue;
				handled.insert(varname);
				// os << indent << "double "<<varname << "["<< range << "];" << std::endl;
				//os << indent << "PYOOMPH_AQUIRE_ARRAY(double, " << varname << ", " << range << ")" << std::endl;
				decl_lines.push_back(indent + "PYOOMPH_AQUIRE_ARRAY(double, " + varname + ", " + range + ")");
			}
		}
		std::sort(decl_lines.begin(), decl_lines.end());
		for (auto &l : decl_lines)
		{
			os << l << std::endl;
		}

		if (!hascontrib)
			return;

		handled.clear();
		//		bool req_loop=this->need_interpolation_loop();
		os << indent << "for (unsigned int l_shape=0;l_shape<" + range + ";l_shape++)" << std::endl;
		os << indent << "{" << std::endl;
		std::vector<std::string> init_lines;
		for (auto &s : required_shapeexps)
		{
			if (s.dt_order == 0 || s.basis->get_space() != this)
				continue;
			if (s.basis->get_space() == this) // Only expand if it is my task
			{
				std::string varname = s.get_dt_values_name(for_code);
				if (handled.count(varname))
					continue;
				handled.insert(varname);
				//os << indent << "  " << varname << "[l_shape]=0.0;" << std::endl;
				init_lines.push_back(indent + "  " + varname + "[l_shape]=0.0;");
			}
		}
		std::sort(init_lines.begin(), init_lines.end());
		for (auto &l : init_lines)
		{
			os << l << std::endl;
		}

		handled.clear();
		os << indent << "  for (unsigned tindex=0;tindex<" << "shapeinfo->timestepper_ntstorage;tindex++)" << std::endl;
		os << indent << "  {" << std::endl;


		std::vector<std::string> compute_lines;
		for (auto &s : required_shapeexps)
		{
			if (s.dt_order == 0 || s.basis->get_space() != this)
				continue;
			std::string nds = for_code->get_nodal_data_string(s.basis->get_space());
			if (s.basis->get_space() == this) // Only expand if it is my task
			{
				std::string varname = s.get_dt_values_name(for_code);
				std::string timedisc_scheme = s.get_timedisc_scheme(for_code);
				bool dgs = true;
				if (s.field->degraded_start.count(""))
					dgs = s.field->degraded_start[""]; // A bit anoying here... Only the default IC can be checked for degraded start
				if (dgs && s.dt_order == 1 && timedisc_scheme != "BDF1")
				{
					timedisc_scheme += "_degr";
				}
				if (handled.count(varname))
					continue;
				handled.insert(varname);
				std::string nodalindex = s.get_nodal_index_str(for_code);
				if (s.dt_order == 1)
				{
					//os << indent << "    " << varname << "[l_shape] += " <<  "shapeinfo->timestepper_weights_dt_" << timedisc_scheme << "[tindex]*" << eleminfo << "->" << nds << "[l_shape][" << nodalindex << "][tindex];" << std::endl;
					compute_lines.push_back(indent + "    " + varname + "[l_shape] += " + "shapeinfo->timestepper_weights_dt_" + timedisc_scheme + "[tindex]*" + eleminfo + "->" + nds + "[l_shape][" + nodalindex + "][tindex];");
				}
				else if (s.dt_order == 2)
				{
					//os << indent << "    " << varname << "[l_shape] += " <<   "shapeinfo->timestepper_weights_d2t_" << timedisc_scheme << "[tindex]*" << eleminfo << "->" << nds << "[l_shape][" << nodalindex << "][tindex];" << std::endl;
					compute_lines.push_back(indent + "    " + varname + "[l_shape] += " + "shapeinfo->timestepper_weights_d2t_" + timedisc_scheme + "[tindex]*" + eleminfo + "->" + nds + "[l_shape][" + nodalindex + "][tindex];");
				}
				else
					throw_runtime_error("TODO Higher order time derivatives");
			}
		}
		std::sort(compute_lines.begin(), compute_lines.end());
		for (auto &l : compute_lines)
		{
			os << l << std::endl;
		}

		os << indent << "  }" << std::endl;
		os << indent << "}" << std::endl;
	}

	// Emits C code that interpolates every required shape expansion on this space at the current
	// integration point: sum_l nodal_value[l] * shape[l] over a loop on l_shape. This is the
	// workhorse that turns symbolic ShapeExpansion nodes into actual C loops/arrays.
	//
	// If `including_nodal_diffs` is set, this additionally emits the "coordinate-diff" arrays
	// needed for analytical Jacobian/Hessian contributions of moving-mesh (ALE) problems: for every
	// spatially-once-differentiated (D1XBasisFunction, Eulerian only - not Lagrangian) shape
	// expansion, it computes d(dpsi_l/dx_dir)/dX_j^m, i.e. how the *derivative* of the shape function
	// itself changes as nodal position m in direction j is perturbed (this is a genuinely nontrivial
	// geometric quantity, since the mapping from local to global coordinates depends on nodal
	// positions). If `for_hessian` is additionally set, the second nodal-coordinate derivative
	// d^2(dpsi_l/dx_dir)/dX_j^m dX_j2^m2 is also emitted (needed for exact Hessian-vector products),
	// looping symmetrically over m2>=m nodal-coordinate-derivative pairs. All emitted lines are again
	// collected and sorted before being written to keep the generated code deterministic.
	void FiniteElementSpace::write_spatial_interpolation(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, std::set<ShapeExpansion> &required_shapeexps, bool including_nodal_diffs, bool for_hessian)
	{
		bool hascontrib = false;
		std::string range = "";
		std::string posrange = "";
		std::set<ShapeExpansion> required_coorddiffs;
		std::vector<std::string> decl_lines;
		for (auto &s : required_shapeexps)
		{
			if (s.basis->get_space() != this)
				continue;
			std::string varname = s.get_spatial_interpolation_name(for_code);
			if (!hascontrib)
			{
				range = s.get_num_nodes_str(for_code);
				posrange = for_code->get_elem_info_str(s.basis->get_space()) + "->nnode";
				hascontrib = true;
			}
			//os << indent << "double " << varname << "=0.0;" << std::endl;
			decl_lines.push_back(indent + "double " + varname + "=0.0;");
			if (including_nodal_diffs)
			{
				if (dynamic_cast<D1XBasisFunction *>(s.basis) && !dynamic_cast<D1XBasisFunctionLagr *>(s.basis))
				{
					required_coorddiffs.insert(s);
				}
			}
		}

		if (!hascontrib)
			return;


		std::sort(decl_lines.begin(), decl_lines.end());
		for (auto &l : decl_lines)
		{
			os << l << std::endl;
		}
		os << indent << "for (unsigned int l_shape=0;l_shape<" + range + ";l_shape++)" << std::endl;
		os << indent << "{" << std::endl;
		std::vector<std::string> calc_lines;
		for (auto &s : required_shapeexps)
		{
			if (s.basis->get_space() != this)
				continue;
			std::string varname = s.get_spatial_interpolation_name(for_code);
			std::string nodal_data = s.get_nodal_data_string(for_code, "l_shape");
			std::string shapestr = s.get_shape_string(for_code, "l_shape");
			//os << indent << "  " << varname << "+= " << nodal_data << " * " << shapestr << ";" << std::endl;
			calc_lines.push_back(indent + "  " + varname + "+= " + nodal_data + " * " + shapestr + ";");
		}
		std::sort(calc_lines.begin(), calc_lines.end());
		for (auto &l : calc_lines)
		{
			os << l << std::endl;
		}
		os << indent << "}" << std::endl;

		
		if (!required_coorddiffs.empty())
		{
			decl_lines.clear();
			for (auto s : required_coorddiffs)
			{
				std::string dtstring = "d" + std::to_string(s.dt_order) + "t" + std::to_string(s.time_history_index);
				if (s.dt_order > 0)
					dtstring += s.dt_scheme;
				for (unsigned int i = 0; i < for_code->nodal_dimension(); i++)
				{
					std::string code_type = for_code->get_owner_prefix(s.basis->get_space());
					std::string coorddiffname = code_type + "intrp_" + dtstring + "_" + s.basis->get_dx_str() + "_COORDDIFF_" + std::to_string(i) + "_" + s.field->get_name();
					//os << indent << "PYOOMPH_AQUIRE_ARRAY(double," << coorddiffname << "," << posrange << ");" << std::endl;
					decl_lines.push_back(indent + "PYOOMPH_AQUIRE_ARRAY(double," + coorddiffname + "," + posrange + ");");
				}
				if (for_hessian)
				{
					for (unsigned int i = 0; i < for_code->nodal_dimension(); i++)
					{
						for (unsigned int j = i; j < for_code->nodal_dimension(); j++) // TODO: Symmetrize? Go from j=i?
						{
							std::string code_type = for_code->get_owner_prefix(s.basis->get_space());
							std::string coorddiffname = code_type + "intrp_" + dtstring + "_" + s.basis->get_dx_str() + "_2ndCOORDDIFF_" + std::to_string(i) + "_" + std::to_string(j) + "_" + s.field->get_name();
							//os << indent << "PYOOMPH_AQUIRE_TWO_D_ARRAY(double," << coorddiffname << "," << posrange << "," << posrange << ");" << std::endl;
							decl_lines.push_back(indent + "PYOOMPH_AQUIRE_TWO_D_ARRAY(double," + coorddiffname + "," + posrange + "," + posrange + ");");
						}
					}
				}
				
			}
			std::sort(decl_lines.begin(), decl_lines.end());
			for (auto &l : decl_lines)
			{
				os << l << std::endl;
			}
			if (!for_hessian)
				os << indent << "if (flag)" << std::endl;
			os << indent << "{" << std::endl
			   << indent << " for (unsigned int m=0;m<" << posrange << ";m++)" << std::endl
			   << indent << " {" << std::endl;
			std::vector<std::string> init_lines;
			calc_lines.clear();
			for (auto s : required_coorddiffs)
			{
				std::string dtstring = "d" + std::to_string(s.dt_order) + "t" + std::to_string(s.time_history_index);
				if (s.dt_order > 0)
					dtstring += s.dt_scheme;
				for (unsigned int i = 0; i < for_code->nodal_dimension(); i++)
				{
					std::string code_type = for_code->get_owner_prefix(s.basis->get_space());
					std::string coorddiffname = code_type + "intrp_" + dtstring + "_" + s.basis->get_dx_str() + "_COORDDIFF_" + std::to_string(i) + "_" + s.field->get_name();
					//os << indent << "    " << coorddiffname << "[m]=0.0;" << std::endl;
					init_lines.push_back(indent + "    " + coorddiffname + "[m]=0.0;");
				}				
			}
			std::sort(init_lines.begin(), init_lines.end());
			for (auto &l : init_lines)
			{
				os << l << std::endl;
			}	

			os << indent << "    for (unsigned int l_shape=0;l_shape<" + range + ";l_shape++)" << std::endl
			   << indent << "    {" << std::endl;
			for (auto s : required_coorddiffs)
			{
				std::string dtstring = "d" + std::to_string(s.dt_order) + "t" + std::to_string(s.time_history_index);
				if (s.dt_order > 0)
					dtstring += s.dt_scheme;
				for (unsigned int i = 0; i < for_code->nodal_dimension(); i++)
				{
					std::string code_type = for_code->get_owner_prefix(s.basis->get_space());
					std::string coorddiffname = code_type + "intrp_" + dtstring + "_" + s.basis->get_dx_str() + "_COORDDIFF_" + std::to_string(i) + "_" + s.field->get_name();
					std::string nodal_data = s.get_nodal_data_string(for_code, "l_shape");
					std::string shapename=s.basis->get_space()->get_shape_name();
					if (shapename=="C2TB" || shapename=="C2" || shapename=="C1TB" || shapename=="C1")
					{
						shapename="d_dx_shape_dcoord[SPACE_INDEX_"+shapename+"]";
					}
					else
					{
						shapename="d_dx_shape_dcoord_" + s.basis->get_space()->get_shape_name();
					}
					std::string shapestr = for_code->get_shape_info_str(s.basis->get_space()) + "->" + shapename + "[l_shape][" + std::to_string(dynamic_cast<D1XBasisFunction *>(s.basis)->get_direction()) + "][m][" + std::to_string(i) + "]";
					//os << indent << "       " << coorddiffname << "[m]+=" << nodal_data << " * " << shapestr << ";" << std::endl;
					calc_lines.push_back(indent + "       " + coorddiffname + "[m]+=" + nodal_data + " * " + shapestr + ";");
				}
			}
			std::sort(calc_lines.begin(), calc_lines.end());
			for (auto &l : calc_lines)
			{
				os << l << std::endl;
			}
			os << indent << "    }" << std::endl;			
			if (for_hessian)
			{
				init_lines.clear();
				std::vector<std::string> hess_lines;
				os << indent << "    for (unsigned int m2=0;m2<" << posrange << ";m2++)" << std::endl
				   << indent << "    {" << std::endl;

				for (auto s : required_coorddiffs)
				{
					std::string dtstring = "d" + std::to_string(s.dt_order) + "t" + std::to_string(s.time_history_index);
					if (s.dt_order > 0)
						dtstring += s.dt_scheme;
					for (unsigned int i = 0; i < for_code->nodal_dimension(); i++)
					{
						for (unsigned int j = i; j < for_code->nodal_dimension(); j++) // TODO: Symmetrize? Go from j=i?
						{
							std::string code_type = for_code->get_owner_prefix(s.basis->get_space());
							std::string coorddiffname = code_type + "intrp_" + dtstring + "_" + s.basis->get_dx_str() + "_2ndCOORDDIFF_" + std::to_string(i) + "_" + std::to_string(j) + "_" + s.field->get_name();
							//os << indent << "       " << coorddiffname << "[m][m2]=0.0;" << std::endl;
							init_lines.push_back(indent + "       " + coorddiffname + "[m][m2]=0.0;");
						}
					}
					std::sort(init_lines.begin(), init_lines.end());
					//os << indent << "       // INIT LINES RIGHT NOW" << std::endl;
					for (auto &l : init_lines)
					{
						os << l << std::endl;
					}
					init_lines.clear();
					os << indent << "       for (unsigned int l_shape=0;l_shape<" + range + ";l_shape++)" << std::endl
					   << indent << "       {" << std::endl;
					/*					os << indent << "         for (unsigned int l_shape2=0;l_shape2<" + range + ";l_shape2++)" << std::endl
											<< indent << "         {" << std::endl;						*/
					//					for (auto s : required_coorddiffs)
					//					{
					//						std::string dtstring = "d" + std::to_string(s.dt_order) + "t" + std::to_string(s.time_history_index);
					//						if (s.dt_order > 0)
					//							dtstring += s.dt_scheme;
					for (unsigned int i = 0; i < for_code->nodal_dimension(); i++)
					{
						for (unsigned int j = i; j < for_code->nodal_dimension(); j++) // TODO: Symmetrize? Go from j=i?
						{
							std::string code_type = for_code->get_owner_prefix(s.basis->get_space());
							std::string coorddiffname = code_type + "intrp_" + dtstring + "_" + s.basis->get_dx_str() + "_2ndCOORDDIFF_" + std::to_string(i) + "_" + std::to_string(j) + "_" + s.field->get_name();
							std::string nodal_data = s.get_nodal_data_string(for_code, "l_shape");
							std::string shapename=s.basis->get_space()->get_shape_name();
							if (shapename=="C2TB" || shapename=="C2" || shapename=="C1TB" || shapename=="C1")
							{
								shapename="d2_dx2_shape_dcoord[SPACE_INDEX_"+shapename+"]";
							}
							else
							{
								shapename="d2_dx2_shape_dcoord_" + s.basis->get_space()->get_shape_name();
							}
							std::string shapestr = for_code->get_shape_info_str(s.basis->get_space()) + "->" + shapename + "[l_shape][" + std::to_string(dynamic_cast<D1XBasisFunction *>(s.basis)->get_direction()) + "][m][" + std::to_string(i) + "][m2][" + std::to_string(j) + "]";
							//os << indent << "             " << coorddiffname << "[m][m2]+=" << nodal_data << " * " << shapestr << ";" << std::endl;
							hess_lines.push_back(indent + "             " + coorddiffname + "[m][m2]+=" + nodal_data + " * " + shapestr + ";");
						}
					}
					std::sort(hess_lines.begin(), hess_lines.end());
					for (auto &l : hess_lines)
					{
						os << l << std::endl;
					}
					hess_lines.clear();
					//					}
					//					os << indent << "         }" << std::endl;
					os << indent << "       }" << std::endl;
				}

				os << indent << "    }" << std::endl;
			}

			os << indent << "  }" << std::endl
			   << indent << "}";
		}
	}

	// D0 spaces have a single (element-local, not nodal) DoF per field, so "interpolation" is trivial:
	// no loop is required, the value is simply the field's (only) nodal data entry. This overrides
	// the generic FiniteElementSpace::write_spatial_interpolation loop-based implementation.
	void D0FiniteElementSpace::write_spatial_interpolation(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, std::set<ShapeExpansion> &required_shapeexps, bool, bool)
	{
		bool hascontrib = false;
		std::string range = "";
		for (auto &s : required_shapeexps)
		{
			//		   os << " // NEWSHAPE "  << std::endl;
			//		   os << " //" ;
			//		   GiNaC::GiNaCShapeExpansion(s).print(GiNaC::print_dflt(os));
			//		   os << std::endl;
			if (s.basis->get_space() != this)
				continue;
			std::string varname = s.get_spatial_interpolation_name(for_code);
			if (!hascontrib)
			{
				range = s.get_num_nodes_str(for_code);
				hascontrib = true;
			}
			os << indent << "double " << varname << ";" << std::endl;
		}

		if (!hascontrib)
			return;

		for (auto &s : required_shapeexps)
		{
			if (s.basis->get_space() != this)
				continue;
			std::string varname = s.get_spatial_interpolation_name(for_code);
			std::string nodal_data = s.get_nodal_data_string(for_code, "0");
			os << indent << "  " << varname << "= " << nodal_data << ";" << std::endl;
		}
	}

	// The nodal position space only carries real degrees of freedom (and thus needs
	// Jacobian/Hessian rows/columns of its own) when the mesh coordinates are themselves unknowns
	// being solved for (moving-mesh/ALE problems with coordinates_as_dofs); otherwise positions are
	// prescribed data and there is nothing to differentiate w.r.t. them.
	bool PositionFiniteElementSpace::write_generic_Hessian_contribution(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, GiNaC::ex for_what, bool hanging_eqns)
	{
		// Only do it if the coordinates are Dofs
		if (for_code->coordinates_as_dofs)
			return FiniteElementSpace::write_generic_Hessian_contribution(for_code, os, indent, for_what, hanging_eqns);
		else
			return false;
	}

	void PositionFiniteElementSpace::write_generic_RJM_jacobian_contribution(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, GiNaC::ex for_what, bool hanging_eqns,FiniteElementField * residual_field)
	{
		// Only do it if the coordinates are Dofs
		if (this->code->coordinates_as_dofs)
			FiniteElementSpace::write_generic_RJM_jacobian_contribution(for_code, os, indent, for_what, hanging_eqns,residual_field);
	}

	// Emits the C code for the Hessian (second-derivative) contribution of `for_what` (typically a
	// residual expression, already differentiated once w.r.t. the "outer" field of this space) with
	// respect to every field defined on every other relevant FiniteElementSpace ("inner" fields).
	// Algorithm, per outer field f on this space:
	//   1. Symbolically differentiate for_what w.r.t. f's raw GiNaC symbol to get diffpart = dR/df.
	//      Also isolate its mass-matrix part (derivative w.r.t. the special "partial_t" marker) since
	//      the mass-matrix Hessian is *not* symmetric and must be tracked separately.
	//   2. Determine the set of FiniteElementSpaces that actually occur (as shape expansions) in
	//      diffpart - these are the spaces that can contribute a nonzero second derivative. Moving
	//      mesh coordinate fields are added explicitly here (and further above, for the outer field)
	//      because they enter through the geometric Jacobian/shape-function derivatives rather than
	//      directly as GiNaC symbols, so they would otherwise be missed by a naive symbol scan.
	//   3. For every inner field f2 on every such space: differentiate diffpart (and masspart) again
	//      w.r.t. f2 to get the true second derivative diffpart2. If for_code->assemble_hessian_by_symmetry
	//      is enabled and the symmetric (f2,f) combination has already been emitted, the almost-symmetric
	//      Hessian entry is skipped entirely (mass part only, since that part is not symmetric).
	//   4. Convert all remaining subexpression(...) markers to structs (via __SE_to_struct_hessian) and
	//      emit nested loops over the outer node index l_shape and inner node index l_shape2 (using the
	//      BEGIN_HESSIAN_SHAPE_LOOP1[_CONTINUOUS_SPACE] macros for hanging-node-aware assembly), writing
	//      the C statement that adds the Hessian entry into the sparse Hessian tensor.
	// Global state (__derive_shapes_by_second_index, __all_Hessian_shapeexps/testfuncs/indices_required,
	// __derive_only_by_expansion_mode) is toggled around the differentiation calls so that the custom
	// GiNaC structures' derivative() implementations know which index of the double loop is being
	// derived, and so the caller can later learn which shape expansions/spaces/fields must have their
	// interpolation code emitted to support this Hessian.
	bool FiniteElementSpace::write_generic_Hessian_contribution(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, GiNaC::ex for_what, bool hanging_eqns)
	{
		GiNaC::print_FEM_options csrc_opts;
		csrc_opts.for_code = for_code;

		std::set<ShapeExpansion> jacobian_shapes = for_code->get_all_shape_expansions_in(for_what);
		bool has_contribs = false;
		// TODO: This is only necessary if a dx portion or dxdpsi is present
		if (for_code->coordinates_as_dofs)
		{
			for (auto d : std::vector<std::string>{"x", "y", "z"})
			{
				if (for_code->get_field_by_name("coordinate_" + d))
				{
					jacobian_shapes.insert(ShapeExpansion(for_code->get_field_by_name("coordinate_" + d), 0, for_code->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
					if (for_code->get_bulk_element())
					{
						jacobian_shapes.insert(ShapeExpansion(for_code->get_bulk_element()->get_field_by_name("coordinate_" + d), 0, for_code->get_bulk_element()->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
						if (for_code->get_bulk_element()->get_bulk_element())
						{
							jacobian_shapes.insert(ShapeExpansion(for_code->get_bulk_element()->get_bulk_element()->get_field_by_name("coordinate_" + d), 0, for_code->get_bulk_element()->get_bulk_element()->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
						}
					}					
				}
			}
		}
		if (for_code->get_opposite_interface_code() && for_code->get_opposite_interface_code()->coordinates_as_dofs)
		{
			for (auto d : std::vector<std::string>{"x", "y", "z"})
			{
				if (for_code->get_opposite_interface_code()->get_field_by_name("coordinate_" + d))
				{
					jacobian_shapes.insert(ShapeExpansion(for_code->get_opposite_interface_code()->get_field_by_name("coordinate_" + d), 0, for_code->get_opposite_interface_code()->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
					if (for_code->get_opposite_interface_code()->get_bulk_element())
					{
						jacobian_shapes.insert(ShapeExpansion(for_code->get_opposite_interface_code()->get_bulk_element()->get_field_by_name("coordinate_" + d), 0, for_code->get_opposite_interface_code()->get_bulk_element()->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
					}
				}
			}
		}

		std::set<FiniteElementField *> jacobian_fields;
		for (auto &s : jacobian_shapes)
		{
			if (s.field->get_space() == this)
			{
				if (!s.field->no_jacobian_at_all)
				{
					jacobian_fields.insert(s.field);
				}
			}
		}
		if (jacobian_fields.empty())
			return false;

		std::string numnodes_str = this->get_num_nodes_str(for_code);

		bool hang = this->can_have_hanging_nodes() || this->code != for_code;

		bool loop1_written = false;

		for (auto &f : jacobian_fields)
		{
			__all_Hessian_indices_required.insert(f);
			bool loop2_written = false;

			__derive_only_by_expansion_mode=for_code->get_derive_jacobian_by_expansion_mode();
			__ignore_dpsi_coord_diffs_in_jacobian=for_code->ignore_dpsi_coord_diffs_in_jacobian();						
			GiNaC::ex diffpart = GiNaC::diff(for_what, f->get_symbol());
			__derive_only_by_expansion_mode=NULL;
			__ignore_dpsi_coord_diffs_in_jacobian=false;

			for_code->subexpressions = __SE_to_struct_hessian->subexpressions;
			//			std::cout << "HESSIAN  CONTRIBU " << for_what << std::endl;
			//			std::cout << "DHESSIAN  CONTRIBU " << diffpart << std::endl;
			if (diffpart.is_zero())
			{
				for_code->Hessian_symmetric_fields_completed.insert(f);
				continue;
			}
			if (pyoomph_verbose)
				std::cout << "DIFF PART IS " << diffpart << std::endl;

			GiNaC::ex masspart = GiNaC::diff(diffpart, pyoomph::expressions::__partial_t_mass_matrix);
			//			  std::cout << "00 POTENTIAL MASS CONTRIB " << f->get_symbol() << " : " << for_what << std::endl;
			//			  std::cout << "00 DERIV " << diffpart << std::endl;
			//				if (!masspart.is_zero())
			//			{
			//		  std::cout << "11 MASSPART BY" << f->get_symbol()<< " : " << masspart << std::endl;
			//		}

			std::string l_shape;
			if (numnodes_str == "1")
			{
				l_shape = "0";
			}
			else
			{
				l_shape = "l_shape";
			}
			std::string eqn_index = f->get_equation_str(for_code, l_shape);

			std::string nodal_index;
			std::string hang_info;
			if (hang)
			{
				nodal_index = f->get_nodal_index_str(for_code);
				hang_info = f->get_hanginfo_str(for_code);
			}

			std::set<ShapeExpansion> hessian_shapes = for_code->get_all_shape_expansions_in(diffpart);
			std::set<FiniteElementSpace *> hessian_spaces;

			// TODO: This is only necessary if a dx portion or dxdpsi is present
			if (for_code->coordinates_as_dofs)
			{
				for (auto d : std::vector<std::string>{"x", "y", "z"})
				{
					if (for_code->get_field_by_name("coordinate_" + d))
					{
						hessian_shapes.insert(ShapeExpansion(for_code->get_field_by_name("coordinate_" + d), 0, for_code->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
						if (for_code->get_bulk_element())
						{
							hessian_shapes.insert(ShapeExpansion(for_code->get_bulk_element()->get_field_by_name("coordinate_" + d), 0, for_code->get_bulk_element()->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
							if (for_code->get_bulk_element()->get_bulk_element())
							{
								hessian_shapes.insert(ShapeExpansion(for_code->get_bulk_element()->get_bulk_element()->get_field_by_name("coordinate_" + d), 0, for_code->get_bulk_element()->get_bulk_element()->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
							}
						}
					}
				}
			}
			if (for_code->get_opposite_interface_code() && for_code->get_opposite_interface_code()->coordinates_as_dofs)
			{
				for (auto d : std::vector<std::string>{"x", "y", "z"})
				{
					if (for_code->get_opposite_interface_code()->get_field_by_name("coordinate_" + d))
					{
						hessian_shapes.insert(ShapeExpansion(for_code->get_opposite_interface_code()->get_field_by_name("coordinate_" + d), 0, for_code->get_opposite_interface_code()->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
						if (for_code->get_opposite_interface_code()->get_bulk_element())
						{
							hessian_shapes.insert(ShapeExpansion(for_code->get_opposite_interface_code()->get_bulk_element()->get_field_by_name("coordinate_" + d), 0, for_code->get_opposite_interface_code()->get_bulk_element()->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
						}
					}
				}
			}

			for (auto s2 : hessian_shapes)
			{
				hessian_spaces.insert(s2.field->get_space());
				__all_Hessian_indices_required.insert(s2.field);
			}
			for (auto *s2 : hessian_spaces)
			{
				if (dynamic_cast<PositionFiniteElementSpace *>(s2))
				{
					if (for_code->coordinates_as_dofs)
					{
						// throw_runtime_error("TODO: Coordinates as dofs in Hessian");
					}
					else
					{
						continue;
					}
				}
				std::set<FiniteElementField *> hessian_fields;
				for (auto &s3 : hessian_shapes)
				{
					if (!s3.field->no_jacobian_at_all && s3.field->get_space() == s2)
					{
						hessian_fields.insert(s3.field);
					}
				}
				if (hessian_fields.empty())
				{
					continue;
				}

				std::string numnodes_str2 = s2->get_num_nodes_str(for_code);
				std::string l_shape2;
				if (numnodes_str2 == "1")
				{
					l_shape2 = "0";
				}
				else
				{
					//					 os << indent << "   for (unsigned int l_shape2=0;l_shape2<" << numnodes_str2 << ";l_shape2++)" << std::endl;
					//					 os << indent << "   {" << std::endl;
					l_shape2 = "l_shape2";
				}

				bool hang2 = s2->can_have_hanging_nodes() || this->code != for_code;

				for (auto f2 : hessian_fields)
				{
					__derive_shapes_by_second_index = true;
					GiNaC::ex masspart2 = GiNaC::diff(masspart, f2->get_symbol());
					bool only_mass_part = false; // Since the mass Hessian is NOT symmetric!
					if (for_code->assemble_hessian_by_symmetry && for_code->Hessian_symmetric_fields_completed.count(f2))
					{
						if (masspart2.is_zero())
						{
							os << "//SYMMETRY: SKIPPING FIELD COMBINATION:  " << f->get_equation_str(for_code, "any") << " & " << f2->get_equation_str(for_code, "any") << std::endl;
							continue;
						}
						else
						{
							only_mass_part = true;
						}
					}

					__derive_only_by_expansion_mode=for_code->get_derive_hessian_by_expansion_mode();
					__ignore_dpsi_coord_diffs_in_jacobian=false;
					GiNaC::ex diffpart2 = GiNaC::diff(diffpart, f2->get_symbol());
					__derive_only_by_expansion_mode=NULL;
					__ignore_dpsi_coord_diffs_in_jacobian=false;

					/*if (!masspart.is_zero())
					{
					  std::cout << "22 MASSPART " << masspart << std::endl;
					  std::cout << "22 MASSPART BY" << f2->get_symbol()<< " : " << masspart2 << std::endl;
					}*/
					for_code->subexpressions = __SE_to_struct_hessian->subexpressions;
					for (auto &s : for_code->subexpressions)
					{
						auto se_shapes=for_code->get_all_shape_expansions_in(s.get_expression());
						for (auto & se : se_shapes) {
							if (!se.is_derived && !se.is_derived_other_index)
							{
								__all_Hessian_shapeexps.insert(se);
							}
							__all_Hessian_indices_required.insert(se.field);
						}
					}
					__derive_shapes_by_second_index = false;
					if (diffpart2.is_zero() && masspart2.is_zero()) // &&  masspart2.is_zero()
						continue;

					auto shapeexps = for_code->get_all_shape_expansions_in(diffpart2);
					auto shapeexpsM = for_code->get_all_shape_expansions_in(masspart2);
					
					for (auto sexpa : shapeexps)
					{
						if ((!sexpa.is_derived && !sexpa.is_derived_other_index) || sexpa.nodal_coord_dir != -1 || sexpa.nodal_coord_dir2 != -1)
							__all_Hessian_shapeexps.insert(sexpa);
						__all_Hessian_indices_required.insert(sexpa.field);
					}
					for (auto sexpa : shapeexpsM)
					{
						if ((!sexpa.is_derived && !sexpa.is_derived_other_index) || sexpa.nodal_coord_dir != -1 || sexpa.nodal_coord_dir2 != -1)
							__all_Hessian_shapeexps.insert(sexpa);
						__all_Hessian_indices_required.insert(sexpa.field);
					}
					//		  		   __all_Hessian_shapeexps.insert(shapeexps.begin(),shapeexps.end());
					auto testfuncs = for_code->get_all_test_functions_in(diffpart2);
					__all_Hessian_testfuncs.insert(testfuncs.begin(), testfuncs.end());
					auto testfuncsM = for_code->get_all_test_functions_in(masspart2);
					__all_Hessian_testfuncs.insert(testfuncsM.begin(), testfuncsM.end());

					std::string eqn_index2 = f2->get_equation_str(for_code, l_shape2);
					std::string hang_info2;
					if (hang2)
					{
						hang_info2 = f2->get_hanginfo_str(for_code);
					}

					if (!loop1_written)
					{
						if (numnodes_str != "1")
						{
							os << indent << "for (unsigned int l_shape=0;l_shape<" << numnodes_str << ";l_shape++)" << std::endl;
							os << indent << "{" << std::endl;
						}
						else
						{
							os << indent << "{" << std::endl;
							os << indent << "   const unsigned int l_shape=0;" << std::endl;
						}
						loop1_written = true;
					}

					if (!loop2_written)
					{
						if (hang)
						{
							os << indent << "  BEGIN_HESSIAN_SHAPE_LOOP1_CONTINUOUS_SPACE(" << eqn_index << "," << hang_info << "," << l_shape << ")" << std::endl;
						}
						else
						{
							os << indent << "  BEGIN_HESSIAN_SHAPE_LOOP1(" << eqn_index << ")" << std::endl;
						}
						loop2_written = true;
					}

					has_contribs = true;

					if (numnodes_str2 != "1")
					{
						os << indent << "     for (unsigned int l_shape2=0;l_shape2<" << numnodes_str2 << ";l_shape2++)" << std::endl;
						os << indent << "     {" << std::endl;
					}

					//		   os << indent << "   //HESSIAN SHAPE CONTRIB: " << f2->get_nodal_index_str(for_code) << ": " << diffpart2 ;

					os << std::endl;
					//	 		                        std::cout << "DIFFPART2 " << diffpart2 << std::endl;
					GiNaC::ex diffpart2_se = (*__SE_to_struct_hessian)(diffpart2);
					//						 		                        std::cout << "DIFFPART2 SE" << diffpart2_se << std::endl;
					for_code->subexpressions = __SE_to_struct_hessian->subexpressions;
					if (hang2)
					{
						os << indent << "        BEGIN_HESSIAN_SHAPE_LOOP2_CONTINUOUS_SPACE(" << eqn_index2 << ",";
						if (only_mass_part)
							os << "0";
						else
							print_simplest_form(diffpart2_se, os, csrc_opts);
						os << "," << hang_info2 << "," << l_shape2 << ")" << std::endl;
					}
					else
					{
						os << indent << "        BEGIN_HESSIAN_SHAPE_LOOP2(" << eqn_index2 << ", ";
						if (only_mass_part)
							os << "0";
						else
							print_simplest_form(diffpart2_se, os, csrc_opts);
						os << ")" << std::endl;
					}
					if (for_code->assemble_hessian_by_symmetry)
					{
						if (f == f2)
							os << indent << "           const bool symmetry_assembly_same_field=true;" << std::endl;
						else
							os << indent << "           const bool symmetry_assembly_same_field=false;" << std::endl;
					}
					if (!only_mass_part)
					{
						os << indent << "           ADD_TO_HESSIAN_" << (hanging_eqns ? "HANG" : "NOHANG") << "_" << (hang ? "HANG" : "NOHANG") << "_" << (hang2 ? "HANG" : "NOHANG") << "()" << std::endl;
						//std::cout << "        HESSIAN CONTRIB: " << f->get_equation_str(for_code, l_shape) << " & " << f2->get_equation_str(for_code, l_shape2) << std::endl;
					}

					//					GiNaC::ex mass_part2 = GiNaC::diff(mass_part, pyoomph::expressions::__partial_t_mass_matrix);
					for_code->subexpressions = __SE_to_struct_hessian->subexpressions;
					// std::cout << "CHECKING MASS PART " << (masspart2-GiNaC::diff(diffpart2,pyoomph::expressions::__partial_t_mass_matrix)) << std::endl;
					// std::cout << " MA " << masspart2 << std::endl;
					// std::cout << " MB " << GiNaC::diff(diffpart2,pyoomph::expressions::__partial_t_mass_matrix) << std::endl;
					//				__derive_shapes_by_second_index = true;
					//					GiNaC::ex masspart2=GiNaC::diff(diffpart2,pyoomph::expressions::__partial_t_mass_matrix);
					//					__derive_shapes_by_second_index = false;
					if (!masspart2.is_zero())
					{
						GiNaC::ex mass_part_se = (*__SE_to_struct_hessian)(masspart2);
						for_code->subexpressions = __SE_to_struct_hessian->subexpressions;
						for_code->mark_nonconstant_mass_matrix(); // If we have a Hessian contribution, then we clearly have a changing mass matrix
						os << indent << "           ADD_TO_MASS_HESSIAN_" << (hanging_eqns ? "HANG" : "NOHANG") << "_" << (hang ? "HANG" : "NOHANG") << "_" << (hang2 ? "HANG" : "NOHANG") << "(";
						print_simplest_form(mass_part_se, os, csrc_opts);
						os << ")" << std::endl;
					}

					if (hang2)
					{
						os << indent << "        END_HESSIAN_SHAPE_LOOP2_CONTINUOUS_SPACE()" << std::endl;
					}
					else
					{
						os << indent << "        END_HESSIAN_SHAPE_LOOP2()" << std::endl;
					}

					if (numnodes_str2 != "1")
					{
						os << indent << "     }" << std::endl;
					}
				}
				__derive_shapes_by_second_index = false;
			}
			__derive_shapes_by_second_index = false;

			if (loop2_written)
			{
				if (hang)
				{
					os << indent << "  END_HESSIAN_SHAPE_LOOP1_CONTINUOUS_SPACE() // " << nodal_index << std::endl;
				}
				else
				{
					os << indent << "  END_HESSIAN_SHAPE_LOOP1() // " << nodal_index << std::endl;
				}
			}

			for_code->Hessian_symmetric_fields_completed.insert(f);
		}
		

		if (loop1_written)
		{
			os << indent << "}" << std::endl;
		}
		__derive_shapes_by_second_index = false; // Just to make sure....
		return has_contribs;
	}

	// Emits the C code for the (first-order) Jacobian and mass-matrix contribution of residual
	// expression `for_what` w.r.t. every field defined on this space ("residual/Jacobian/Mass
	// matrix" = RJM). For each candidate field f found among the shape expansions occurring in
	// for_what (plus, for moving-mesh problems, the nodal position fields, which enter implicitly
	// through geometric/shape-function factors and must be added explicitly), symbolically
	// differentiates for_what w.r.t. f's GiNaC symbol; the coefficient of the special
	// __partial_t_mass_matrix marker within that derivative is the mass-matrix entry, the rest is
	// the (stiffness) Jacobian entry. Loops over the node index l_shape (or "0" for D0/single-DoF
	// spaces) and, for hanging-node-capable spaces, uses the BEGIN/END_JACOBIAN_HANG macros so the
	// hanging-node constraint is distributed correctly; otherwise the simpler _NOHANG variants are
	// used. `residual_field` is recorded (mark_jacobian_contribution_for_code) so later passes know
	// which field pairs actually contribute and can skip generating dead code for structurally-zero
	// combinations.
	void FiniteElementSpace::write_generic_RJM_jacobian_contribution(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, GiNaC::ex for_what, bool hanging_eqns,FiniteElementField * residual_field)
	{
		GiNaC::print_FEM_options csrc_opts;
		csrc_opts.for_code = for_code;

		std::set<ShapeExpansion> jacobian_shapes = for_code->get_all_shape_expansions_in(for_what);

		// TODO: This is only necessary if a dx portion or dxdpsi is present
		if (for_code->coordinates_as_dofs)
		{
			for (auto d : std::vector<std::string>{"x", "y", "z"})
			{
				if (for_code->get_field_by_name("coordinate_" + d))
				{
					jacobian_shapes.insert(ShapeExpansion(for_code->get_field_by_name("coordinate_" + d), 0, for_code->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
					if (for_code->get_bulk_element())
					{
						jacobian_shapes.insert(ShapeExpansion(for_code->get_bulk_element()->get_field_by_name("coordinate_" + d), 0, for_code->get_bulk_element()->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
						if (for_code->get_bulk_element()->get_bulk_element())
						{
							jacobian_shapes.insert(ShapeExpansion(for_code->get_bulk_element()->get_bulk_element()->get_field_by_name("coordinate_" + d), 0, for_code->get_bulk_element()->get_bulk_element()->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
						}
					}					
				}
			}
		}
		if (for_code->get_opposite_interface_code() && for_code->get_opposite_interface_code()->coordinates_as_dofs)
		{
			for (auto d : std::vector<std::string>{"x", "y", "z"})
			{
				if (for_code->get_opposite_interface_code()->get_field_by_name("coordinate_" + d))
				{
					jacobian_shapes.insert(ShapeExpansion(for_code->get_opposite_interface_code()->get_field_by_name("coordinate_" + d), 0, for_code->get_opposite_interface_code()->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
					if (for_code->get_opposite_interface_code()->get_bulk_element())
					{
						jacobian_shapes.insert(ShapeExpansion(for_code->get_opposite_interface_code()->get_bulk_element()->get_field_by_name("coordinate_" + d), 0, for_code->get_opposite_interface_code()->get_bulk_element()->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
					}
				}
			}
		}
		
		auto cmp = [&for_code](FiniteElementField * a, FiniteElementField * b) 
		{ 			
			return a->get_nodal_index_str(for_code) < b->get_nodal_index_str(for_code); 
		};
		std::set<FiniteElementField *,decltype(cmp)> jacobian_fields(cmp);
		//std::set<FiniteElementField *> jacobian_fields;
		
		for (auto &s : jacobian_shapes)
		{
			if (s.field->get_space() == this)
			{
				if (!s.field->no_jacobian_at_all)
				{
					jacobian_fields.insert(s.field);
				}
			}
		}
		/*
		std::cout << " IN RJM JACOBIAN CONTRIB, NUMBER OF SHAPES: " << jacobian_shapes.size() << std::endl;
		for (auto &s : jacobian_shapes)
		{
			std::cout << "  SHAPE: " << s.get_nodal_data_string(for_code,"INDEX") << " " << s.get_shape_string(for_code,"INDEX") << "  " << s.get_nodal_index_str(for_code) << std::endl;
		}
		std::cout << " IN RJM JACOBIAN CONTRIB, NUMBER OF FIELDS: " << jacobian_fields.size() << std::endl;
		for (auto &f : jacobian_fields)
		{
			std::cout << "  FIELD: " << f->get_name() << "   NODAL INDEX: " << f->get_nodal_index_str(for_code) << std::endl;
		}		
		*/
		if (jacobian_fields.empty())
			return;
		std::string numnodes_str = this->get_num_nodes_str(for_code);
		std::string l_shape;
		if (numnodes_str == "1")
		{
			l_shape = "0";
		}
		else
		{
			os << indent << "for (unsigned int l_shape=0;l_shape<" << numnodes_str << ";l_shape++)" << std::endl;
			os << indent << "{" << std::endl;
			l_shape = "l_shape";
		}

		bool hang = this->can_have_hanging_nodes() || this->code != for_code;

		for (auto &f : jacobian_fields)
		{
			if (pyoomph_verbose)
			{
				std::cout << "DIFFING FOR JACOBIAN " << for_what << "   WRT.  " << f->get_symbol() << std::endl
						  << std::flush;
			}
			__derive_only_by_expansion_mode=for_code->get_derive_jacobian_by_expansion_mode();
			__ignore_dpsi_coord_diffs_in_jacobian=for_code->ignore_dpsi_coord_diffs_in_jacobian();
			GiNaC::ex diffpart = GiNaC::diff(for_what, f->get_symbol());
			__derive_only_by_expansion_mode=NULL;
			__ignore_dpsi_coord_diffs_in_jacobian=false;

		
			if (diffpart.is_zero())
				continue;
			if (pyoomph_verbose)
				std::cout << "DIFF PART IS " << diffpart << std::endl;
			std::string eqn_index = f->get_equation_str(for_code, l_shape);
			
			for_code->add_contributing_field(residual_field);
			for_code->add_contributing_field(f);
			residual_field->mark_jacobian_contribution_for_code(for_code,for_code->get_current_residual_index(),f);
			if (hang)
			{
				std::string hang_info = f->get_hanginfo_str(for_code);
				os << indent << "  BEGIN_JACOBIAN_HANG(" << eqn_index << ", ";
				//if (for_code->get_derive_jacobian_by_expansion_mode())
				//{
				//	os << indent << " /* SYMBOLIC FORM  " << std::endl << diffpart << std::endl << " */ ";
				//}
				print_simplest_form(diffpart, os, csrc_opts);
				os << "," << hang_info << "," << l_shape << ")" << std::endl;
			}
			else
			{
				//	    os << indent << "  //TODO Jacobian of ext data must be always hanging!!! " <<std::endl;
				os << indent << "  BEGIN_JACOBIAN_NOHANG(" << eqn_index << ", ";
				//if (for_code->get_derive_jacobian_by_expansion_mode())
				//{
				//	os << indent << " /* SYMBOLIC FORM  " << std::endl << diffpart << std::endl << " */ ";
				//}
				print_simplest_form(diffpart, os, csrc_opts);
				os << indent << ")" << std::endl;
			}
			os << indent << "    ADD_TO_JACOBIAN_" << (hanging_eqns ? "HANG" : "NOHANG") << "_" << (hang ? "HANG" : "NOHANG") << "()" << std::endl;
			// diffpart.evalf().print(GiNaC::print_csrc_FEM(os,&csrc_opts));
			// GiNaC::factor(GiNaC::normal(GiNaC::expand(GiNaC::expand(diffpart).evalf()))).print(GiNaC::print_csrc_FEM(os,&csrc_opts));

			//	    os <<")" <<std::endl;

			GiNaC::ex mass_part = GiNaC::diff(diffpart, pyoomph::expressions::__partial_t_mass_matrix);
			if (!mass_part.is_zero())
			{
				os << indent << "    ADD_TO_MASS_MATRIX_" << (hanging_eqns ? "HANG" : "NOHANG") << "_" << (hang ? "HANG" : "NOHANG") << "(";
				//		    mass_part.evalf().print(GiNaC::print_csrc_FEM(os,&csrc_opts));
				//          GiNaC::factor(GiNaC::normal(GiNaC::expand(GiNaC::expand(mass_part).evalf()))).print(GiNaC::print_csrc_FEM(os,&csrc_opts));
				//std::cout << mass_part << std::endl;
				print_simplest_form(mass_part, os, csrc_opts);
				os << ")" << std::endl;
			}
			os << indent << "  END_JACOBIAN_" << (hang ? "HANG" : "NOHANG") << "()" << std::endl;
		}

		if (numnodes_str != "1")
		{
			os << indent << "}" << std::endl;
		}
	}

	// Top-level driver that emits the full residual + Jacobian (+ optionally Hessian, if `hessian`)
	// code for the part of `for_what` that is tested against test functions living on this space.
	// Uses MapOnTestSpace to isolate, per field with a test function on this space, exactly the
	// terms of the residual belonging to that field's equation; loops over the local test-node index
	// l_test (emitting the required shape/dx/dX/dS-shape pointers first), and for every present test
	// field: emits the residual contribution (BEGIN_RESIDUAL[_CONTINUOUS_SPACE]/ADD_TO_RESIDUAL...,
	// skipped if `hessian` is true since the Hessian pass only needs Jacobian/mass matrix code, or if
	// the residual assembly for the currently active residual name has been explicitly disabled via
	// set_ignore_residual_assembly), then delegates to write_generic_RJM_jacobian_contribution (or,
	// for `hessian`, to write_generic_Hessian_contribution) on every FiniteElementSpace that
	// contributes shape expansions to that field's residual term, to produce the first (or second)
	// derivative code. Returns whether any nonzero contribution was written at all, which callers use
	// to decide whether to skip emitting entire (empty) code blocks/functions.
	bool FiniteElementSpace::write_generic_RJM_contribution(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, GiNaC::ex for_what, bool hessian)
	{
		bool has_contribs = false;
		GiNaC::print_FEM_options csrc_opts;
		csrc_opts.for_code = for_code;
		// First step -> Map the residual on this space only
		MapOnTestSpace mapper(this, "");
		GiNaC::ex mypart = mapper(for_what);
		if (pyoomph_verbose)
			std::cout << "MYPART " << mypart << std::endl;
		if (pyoomph_verbose)
			std::cout << "FORWHAT " << for_what << std::endl;
		if (mypart.is_zero())
			return false;
		// Gather all test functions
		std::set<TestFunction> alltests = for_code->get_all_test_functions_in(mypart);
		std::set<std::string> present_tests;
		for (auto &a : alltests)
			present_tests.insert(a.field->get_name());
		if (!for_code->coordinates_as_dofs)
		{
			for (auto &n : present_tests)
			{
				if (n == "coordinate_x" || n == "coordinate_y" || n == "coordinate_z")
				{
					std::string info=for_code->get_full_domain_name();
					throw_runtime_error("Cannot add residual contributions on the position test space as long as the bulk element has not activated the positions as dofs (i.e. via calling BulkElement.activate_coordinates_as_dofs for domain "+info+")");
				}
			}
		}
		std::ostringstream oss;
		std::string numnodes_str = this->get_num_nodes_str(for_code);
		oss << indent << "{" << std::endl;
		std::string shapeinfo = for_code->get_shape_info_str(this);

		std::string l_test;
		if (numnodes_str != "1")
		{

			if (this->get_shape_name()=="C2TB" || this->get_shape_name()=="C2" || this->get_shape_name()=="C1TB" || this->get_shape_name()=="C1")
			{
				oss << indent << "  double const * testfunction = " << shapeinfo << "->shapes[SPACE_INDEX_" << this->get_shape_name() << "];" << std::endl;
				oss << indent << "  DX_SHAPE_FUNCTION_DECL(dx_testfunction) = " << shapeinfo << "->dx_shapes[SPACE_INDEX_" << this->get_shape_name() << "];" << std::endl;
				oss << indent << "  DX_SHAPE_FUNCTION_DECL(dX_testfunction) = " << shapeinfo << "->dX_shapes[SPACE_INDEX_" << this->get_shape_name() << "];" << std::endl;
				oss << indent << "  DX_SHAPE_FUNCTION_DECL(dS_testfunction) = " << shapeinfo << "->dS_shapes[SPACE_INDEX_" << this->get_shape_name() << "];" << std::endl;
			}
			else
			{
				oss << indent << "  double const * testfunction = " << shapeinfo << "->shape_" << this->get_shape_name() << ";" << std::endl;
				oss << indent << "  DX_SHAPE_FUNCTION_DECL(dx_testfunction) = " << shapeinfo << "->dx_shape_" << this->get_shape_name() << ";" << std::endl;
				oss << indent << "  DX_SHAPE_FUNCTION_DECL(dX_testfunction) = " << shapeinfo << "->dX_shape_" << this->get_shape_name() << ";" << std::endl;
				oss << indent << "  DX_SHAPE_FUNCTION_DECL(dS_testfunction) = " << shapeinfo << "->dS_shape_" << this->get_shape_name() << ";" << std::endl;
			}

			oss << indent << "  for (unsigned int l_test=0;l_test<" << numnodes_str << ";l_test++)" << std::endl;
			oss << indent << "  {" << std::endl;
			l_test = "l_test";
		}
		else
		{
			l_test = "0";
		}
		for (auto &test_name : present_tests)
		{
			for_code->Hessian_symmetric_fields_completed.clear();			
			MapOnTestSpace var_mapper(this, test_name);
			GiNaC::ex var_part = var_mapper(mypart);
			if (var_part.is_zero())
				continue;
			FiniteElementField *field = var_mapper.get_field();
			//if (hessian) std::cout << "HESSIAN TEST: " << test_name << std::endl;
			std::string eqn_index = field->get_equation_str(for_code, l_test);
			std::string hang_info = field->get_hanginfo_str(for_code);
			bool hessian_loop1_written = false;
			bool can_have_hanging = this->can_have_hanging_nodes() || for_code != this->code; // Always hang for external spaces
			if (!hessian)
			{
				field->mark_residual_contribution_for_code(for_code,for_code->get_current_residual_index());
				//std::cout << "MARKING RESIDUAL CONTRIBUTION " << field->get_space()->get_code()->get_full_domain_name()+"/"+field->get_name() << " for " << for_code->get_full_domain_name()<< " PTR " << field << " FOR CODE " << for_code <<std::endl;
				for_code->add_contributing_field(field);
				has_contribs = true;
				if (can_have_hanging)
				{
					oss << indent << "    BEGIN_RESIDUAL_CONTINUOUS_SPACE(" << eqn_index << ",";
					if (for_code->is_current_residual_assembly_ignored())
					{
						oss << "0 /* IGNORED RESIDUAL " << std::endl; //<< var_part << std::endl << std::endl ;
					}
					//else
					//{
						print_simplest_form(var_part, oss, csrc_opts);
					//}
					if (for_code->is_current_residual_assembly_ignored())
					{
						oss << std::endl << "*/" << std::endl;
					}
					if (for_code->latex_printer)
					{
						std::map<std::string, std::string> latexinfo = {{"typ", "final_residual"}, {"test_name", test_name}};
						for_code->latex_printer->print(latexinfo, var_part, csrc_opts);
					}
					oss << ", " << hang_info << "," << l_test << ")" << std::endl;
					oss << indent << "      ADD_TO_RESIDUAL_CONTINUOUS_SPACE()" << std::endl;
				}
				else
				{
					oss << indent << "    BEGIN_RESIDUAL(" << eqn_index << ", ";
					if (for_code->is_current_residual_assembly_ignored())
					{
						oss << "0 /* IGNORED RESIDUAL: " << std::endl /*<< var_part << std::endl << std::endl */;
					}
					/*else
					{*/
						print_simplest_form(var_part, oss, csrc_opts);
					//}
					if (for_code->is_current_residual_assembly_ignored())
					{
						oss << std::endl << "*/" << std::endl;
					}
					oss << ")" << std::endl;
					oss << indent << "      ADD_TO_RESIDUAL()" << std::endl;
				}
			}

			//    print_simplest_form(var_part,os,csrc_opts);
			//      os << ")" << std::endl;

			// Now test for any remaining shape expansions, if there are present, we need to add it to the Jacobian //TODO: This needs to be handled with care in case of moving nodes
			std::set<ShapeExpansion> jacobian_shapes = for_code->get_all_shape_expansions_in(var_part);
			// Make sure to include all coordinates if we have coordinates as dofs (TODO: This should be only necessary if dx or dpsidx is present)
			if (for_code->coordinates_as_dofs)
			{
				for (auto d : std::vector<std::string>{"x", "y", "z"})
				{
					if (for_code->get_field_by_name("coordinate_" + d))
					{
						jacobian_shapes.insert(ShapeExpansion(for_code->get_field_by_name("coordinate_" + d), 0, for_code->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
						if (for_code->get_bulk_element())
						{
							jacobian_shapes.insert(ShapeExpansion(for_code->get_bulk_element()->get_field_by_name("coordinate_" + d), 0, for_code->get_bulk_element()->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
							if (for_code->get_bulk_element()->get_bulk_element())
							{
								jacobian_shapes.insert(ShapeExpansion(for_code->get_bulk_element()->get_bulk_element()->get_field_by_name("coordinate_" + d), 0, for_code->get_bulk_element()->get_bulk_element()->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
							}
						}						
					}
				}
			}
			if (for_code->get_opposite_interface_code() && for_code->get_opposite_interface_code()->coordinates_as_dofs)
			{
				for (auto d : std::vector<std::string>{"x", "y", "z"})
				{					
					if (for_code->get_opposite_interface_code()->get_field_by_name("coordinate_" + d))
					{
						//std::cout << "Adding coordinate " << d << " from opposite interface code " << for_code->get_opposite_interface_code()->get_full_domain_name() << " to Jacobian shapes for code " << for_code->get_full_domain_name() << std::endl;
						jacobian_shapes.insert(ShapeExpansion(for_code->get_opposite_interface_code()->get_field_by_name("coordinate_" + d), 0, for_code->get_opposite_interface_code()->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
						if (for_code->get_opposite_interface_code()->get_bulk_element())
						{
							jacobian_shapes.insert(ShapeExpansion(for_code->get_opposite_interface_code()->get_bulk_element()->get_field_by_name("coordinate_" + d), 0, for_code->get_opposite_interface_code()->get_bulk_element()->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
						}			
					}
				}
			}
			if (!jacobian_shapes.empty())
			{
				if (!hessian)
					oss << indent << "      BEGIN_JACOBIAN()" << std::endl;
				
				auto cmp=[&for_code](FiniteElementField * a, FiniteElementField * b) 
				{ 			
					return a->get_nodal_index_str(for_code) < b->get_nodal_index_str(for_code); 
				};
				std::set<FiniteElementField *, decltype(cmp)> jacobian_fields(cmp);
				//std::set<FiniteElementField *> jacobian_fields;
				for (auto &s : jacobian_shapes)
				{					
					//std::cout << "Test function " << test_name << " Jacobian Field " << s.field->get_name() << " Space " << s.field->get_space()->get_name() << " on " << s.field->get_space()->get_code()->get_full_domain_name() << std::endl;
					jacobian_fields.insert(s.field);
				}
				// This might be problematic for DG methods... HDG e.g. accesses both sides, but they are somehow the same
				
				auto cmp_spaces=[&for_code](FiniteElementSpace * a, FiniteElementSpace * b) 
				{ 			
					return (a->get_name()<b->get_name()) || 
						   ((a->get_name()==b->get_name()) &&  (a->get_code()->get_full_domain_name() < b->get_code()->get_full_domain_name())) || 
						   ((a->get_name()==b->get_name()) &&  (a->get_code()->get_full_domain_name() == b->get_code()->get_full_domain_name()) && (a->get_num_nodes_str(for_code)<b->get_num_nodes_str(for_code))) || 
						   ((a->get_name()==b->get_name()) &&  (a->get_code()->get_full_domain_name() == b->get_code()->get_full_domain_name()) && (a->get_num_nodes_str(for_code)==b->get_num_nodes_str(for_code)) && (for_code->get_elem_info_str(a)<for_code->get_elem_info_str(b)));
				};
				std::set<FiniteElementSpace *, decltype(cmp_spaces)> jacobian_spaces(cmp_spaces);
				
				//std::set<FiniteElementSpace *> jacobian_spaces;
				
				for (auto *s : jacobian_fields)
					jacobian_spaces.insert(s->get_space());
				if (pyoomph_verbose)
					std::cout << "VAR PART IS " << var_part << std::endl;
				for (auto *s : jacobian_spaces)
				{
					if (pyoomph_verbose)
						std::cout << "writing contrib of domain " << s->get_code() << std::endl;
					if (!hessian)
						s->write_generic_RJM_jacobian_contribution(for_code, oss, indent + "        ", var_part, can_have_hanging,field);
					else
					{
						std::ostringstream hessian_inner;
						//        	    std::cout << "HESSIAN INNER " << var_part <<std::endl;
						bool has_hessian = s->write_generic_Hessian_contribution(for_code, hessian_inner, indent + "        ", var_part, can_have_hanging);
						if (has_hessian)
						{
							has_contribs = true;
							if (!hessian_loop1_written)
							{
								if (can_have_hanging)
								{
									oss << indent << "    BEGIN_HESSIAN_TEST_LOOP_CONTINUOUS_SPACE(" << eqn_index << ", " << hang_info << "," << l_test << ")" << std::endl;
								}
								else
								{
									oss << indent << "    BEGIN_HESSIAN_TEST_LOOP(" << eqn_index << ")" << std::endl;
								}
								hessian_loop1_written = true;
							}
							oss << hessian_inner.str();
						}
					}
				}
				if (!hessian)
					oss << indent << "      END_JACOBIAN()" << std::endl;
			}

			if (!hessian)
			{
				if (can_have_hanging)
				{
					oss << indent << "    END_RESIDUAL_CONTINUOUS_SPACE()" << std::endl;
				}
				else
				{
					oss << indent << "    END_RESIDUAL()" << std::endl;
				}
			}
			else if (hessian_loop1_written)
			{
				if (can_have_hanging)
				{
					oss << indent << "    END_HESSIAN_TEST_LOOP_CONTINUOUS_SPACE()" << std::endl;
				}
				else
				{
					oss << indent << "    END_HESSIAN_TEST_LOOP()" << std::endl;
				}
			}
		}
		if (numnodes_str != "1")
		{
			oss << indent << "  }" << std::endl;
		}
		oss << indent << "}" << std::endl;
		if (has_contribs)
		{
			os << oss.str();
		}
		return has_contribs;
	}

	// Finds the (unique) PositionFiniteElementSpace owned by this code (as opposed to inherited
	// from a bulk/external element); raises an error if somehow more than one is registered.
	PositionFiniteElementSpace *FiniteElementCode::get_my_position_space()
	{
		PositionFiniteElementSpace *res = NULL;
		for (auto *s : allspaces)
		{
			if (dynamic_cast<PositionFiniteElementSpace *>(s))
			{
				if (s->get_code() == this)
				{
					if (!res)
						res = dynamic_cast<PositionFiniteElementSpace *>(s);
					else
					{
						throw_runtime_error("Code has multiple position spaces");
					}
				}
			}
		}
		return res;
	}

	std::set<FiniteElementField *> FiniteElementCode::get_fields_on_space(FiniteElementSpace *space)
	{
		std::set<FiniteElementField *> res;
		for (auto *f : myfields)
		{
			if (f->get_space() == space)
				res.insert(f);
		}
		return res;
	}

	// The next two overloads flag, for a given generated-function category (`func_type`, e.g.
	// "residual" or "jacobian"), that shape functions of a particular kind (plain "psi",
	// Eulerian-derivative "dx_psi", or Lagrangian-derivative "dX_psi") must be computed/provided for
	// `space` when writing that function - this drives write_required_shapes()/mark_further_required_fields()
	// so that only the shape data actually used by the generated code is computed at runtime.
	// DG spaces are recorded under their underlying continuous space, since the shape functions are
	// identical; external D0 spaces never need shape data (their DoF is not spatially interpolated).
	void FiniteElementCode::mark_shapes_required(std::string func_type, FiniteElementSpace *space, BasisFunction *bf)
	{
		std::string dx_type = "psi";
		if (dynamic_cast<D1XBasisFunction *>(bf))
		{
			dx_type = "dx_psi";
			if (dynamic_cast<D1XBasisFunctionLagr *>(bf))
			{
				dx_type = "dX_psi";
			}
		}
		this->mark_shapes_required(func_type, space, dx_type);
	}

	void FiniteElementCode::mark_shapes_required(std::string func_type, FiniteElementSpace *space, std::string dx_type)
	{
		if (dynamic_cast<ExternalD0Space *>(space))
			return;
		if (!required_shapes.count(func_type))
			required_shapes[func_type] = std::map<FiniteElementSpace *, std::map<std::string, bool>>();
		if (dynamic_cast<DGFiniteElementSpace *>(space))
		{
			space = dynamic_cast<DGFiniteElementSpace *>(space)->get_corresponding_continuous_space(); // We only mark the continuous spaces here. Shape functions are identical
		}
		if (!required_shapes[func_type].count(space))
			required_shapes[func_type][space] = std::map<std::string, bool>();
		required_shapes[func_type][space][dx_type] = true;
	}

	// Returns the C literal "true"/"false" indicating whether mark_shapes_required() was previously
	// called for this (func_type, space, dx_type) combination - used to conditionally emit shape
	// data computation only where actually needed.
	std::string FiniteElementCode::get_shapes_required_string(std::string func_type, FiniteElementSpace *space, std::string dx_type)
	{
		if (required_shapes.count(func_type))
		{
			if (required_shapes[func_type].count(space))
			{
				if (required_shapes[func_type][space].count(dx_type))
				{
					if (required_shapes[func_type][space][dx_type])
						return "true";
					else
						return "false";
				}
				else
					return "false";
			}
			else
				return "false";
		}
		else
			return "false";
	}

	// Looks up one of this code's registered FiniteElementSpace objects (Pos, C2, C1, D0, ...) by its
	// short name; raises a descriptive error listing the available spaces if not found.
	FiniteElementSpace *FiniteElementCode::name_to_space(std::string name)
	{
		for (unsigned int i = 0; i < spaces.size(); i++)
			if (spaces[i]->get_name() == name)
				return spaces[i];
		std::string avail = "Cannot resolve the field space name '" + name + "' on this element. Possible spaces are:";
		for (unsigned int i = 0; i < spaces.size(); i++)
		{
			if (spaces[i]->get_name() != "ED0")
				avail = avail + "\n" + spaces[i]->get_name();
		}
		throw_runtime_error(avail);
		return NULL;
	}

	// Registers a new field `name` on the space named `spacename` (creating a FiniteElementField), or
	// returns the existing one if already registered on the same space (raises an error if the same
	// name is registered on two different spaces). Fields may only be added before stage 1, i.e.
	// before any residual has been added - the element's field set is fixed after that point.
	FiniteElementField *FiniteElementCode::register_field(std::string name, std::string spacename)
	{

		for (unsigned int i = 0; i < this->myfields.size(); i++)
		{
			if (myfields[i]->get_name() == name)
			{
				if (myfields[i]->get_space()->get_name() == spacename)
					return myfields[i];
				else
					throw_runtime_error("Field '" + name + "' is defined on two different spaces, namely '" + myfields[i]->get_space()->get_name() + "' and '" + spacename + "'");
			}
		}
		if (stage != 0)
			throw_runtime_error("Can only add fields before adding residuals: Trying to add " + name + " on space " + spacename);
		FiniteElementField *res = new FiniteElementField(name, this->name_to_space(spacename));
		myfields.push_back(res);
		return res;
	}

	bool ContinuousFiniteElementSpace::can_have_hanging_nodes()
	{
		return code->with_adaptivity;
	}

	// Constructs the built-in FiniteElementSpace hierarchy shared by every element (Pos, the
	// continuous C2TB/C2/C1TB/C1 spaces, their discontinuous-Galerkin D2TB/D2/D1TB/D1 counterparts,
	// the fully discontinuous DL/D0 spaces, and the external ED0 space), plus the symbolic
	// "derived" dx/element-size symbol families (dx_derived[i], dx_derived2[i][j], elemsize_derived,
	// elemsize_Cart_derived, ... and their Hessian "second index" variants) used to represent
	// spatial derivatives of the integration measure w.r.t. moving nodal coordinates - these are
	// pre-built for all 3 spatial directions upfront so that GiNaC::diff() on dx/element-size
	// symbols can return the correctly-tagged symbol without having to construct new ones on the fly.
	FiniteElementCode::FiniteElementCode() : residual_index(0), residual_names({""}), equations(NULL), bulk_code(NULL), opposite_interface_code(NULL), residual(std::vector<GiNaC::ex>{0}), dx(this, false), dX(this, true), dx_unity(this, false), elemsize_Eulerian(this, false, true), elemsize_Lagrangian(this, true, true), elemsize_Eulerian_Cart(this, false, false), elemsize_Lagrangian_Cart(this, true, false), nodal_delta(this), stage(0), nodal_dim(0), lagr_dim(0), coordinate_sys(&__no_coordinate_system), _x(GiNaC::indexed(GiNaC::potential_real_symbol("interpolated_x"), GiNaC::idx(0, 3))),
											 _y(GiNaC::indexed(GiNaC::potential_real_symbol("interpolated_x"), GiNaC::idx(1, 3))), _z(GiNaC::indexed(GiNaC::potential_real_symbol("interpolated_x"))), integration_order(0), IC_names({""}), has_constant_mass_matrix_for_sure(std::vector<bool>{false}), element_dim(-1), analytical_jacobian(true), analytical_position_jacobian(true), debug_jacobian_epsilon(0.0), with_adaptivity(true),
											 coordinates_as_dofs(false), generate_hessian(false), assemble_hessian_by_symmetry(true), coordinate_space(""), stop_on_jacobian_difference(false), latex_printer(NULL)
	{
		dx_unity.simple_unity_integral=true;
		spaces.push_back(new PositionFiniteElementSpace(this, "Pos"));
		spaces.push_back(new ContinuousFiniteElementSpace(this, "C2TB"));
		spaces.push_back(new ContinuousFiniteElementSpace(this, "C2"));
		spaces.push_back(new ContinuousFiniteElementSpace(this, "C1TB"));
		spaces.push_back(new ContinuousFiniteElementSpace(this, "C1"));

		spaces.push_back(new DGFiniteElementSpace(this, "D2TB", spaces[1]));
		spaces.push_back(new DGFiniteElementSpace(this, "D2", spaces[2]));
		spaces.push_back(new DGFiniteElementSpace(this, "D1TB", spaces[3]));
		spaces.push_back(new DGFiniteElementSpace(this, "D1", spaces[4]));

		spaces.push_back(new DiscontinuousFiniteElementSpace(this, "DL"));
		spaces.push_back(new D0FiniteElementSpace(this, "D0"));
		spaces.push_back(new ExternalD0Space(this, "ED0"));
		for (unsigned int i = 0; i < 3; i++)
		{
			dx_derived.push_back(SpatialIntegralSymbol(this, false, i));
			dx_derived_lshape2_for_Hessian.push_back(SpatialIntegralSymbol(this, false, i, "second_index"));
			dx_derived2.push_back(std::vector<SpatialIntegralSymbol>());
			for (unsigned int j = 0; j < 3; j++)
			{
				dx_derived2.back().push_back(SpatialIntegralSymbol(this, false, i, j)); // TODO: Potentially use the symmetry
			}
		}
		for (unsigned int i = 0; i < 3; i++)
		{
			elemsize_derived.push_back(ElementSizeSymbol(this, false, true, i));
			elemsize_derived_lshape2_for_Hessian.push_back(ElementSizeSymbol(this, false, true, i, "second_index"));
			elemsize_derived2.push_back(std::vector<ElementSizeSymbol>());
			elemsize_Cart_derived.push_back(ElementSizeSymbol(this, false, false, i));
			elemsize_Cart_derived_lshape2_for_Hessian.push_back(ElementSizeSymbol(this, false, false, i, "second_index"));
			elemsize_Cart_derived2.push_back(std::vector<ElementSizeSymbol>());
			for (unsigned int j = 0; j < 3; j++)
			{
				elemsize_derived2.back().push_back(ElementSizeSymbol(this, false, true, i, j));		  // TODO: Potentially use the symmetry
				elemsize_Cart_derived2.back().push_back(ElementSizeSymbol(this, false, false, i, j)); // TODO: Potentially use the symmetry
			}
		}
	}

	// Switches the "current residual" context to the named residual, creating a new (empty, zero)
	// residual slot for it if it doesn't exist yet. All subsequent add_residual() calls accumulate
	// into residual[residual_index] until the next _activate_residual() call.
	void FiniteElementCode::_activate_residual(std::string name)
	{
		for (unsigned int i = 0; i < residual_names.size(); i++)
		{
			if (name == residual_names[i])
			{
				residual_index = i;
				return;
			}
		}
		residual_index = residual_names.size();
		residual_names.push_back(name);
		has_constant_mass_matrix_for_sure.push_back(false);
		residual.push_back(0);
	}

	FiniteElementCode::~FiniteElementCode()
	{
		for (auto *s : spaces)
			if (s)
				delete s;
		for (auto *f : myfields)
			if (f)
				delete f;
	}

	// Collects every distinct ShapeExpansion appearing anywhere in expression `inp`, recursing into
	// subexpression(...) wrappers and multi-return callback invocation arguments (since those are
	// opaque to a plain preorder traversal otherwise). If any of the merge_* flags are set, entries
	// that are identical except for the no_jacobian/no_hessian/expansion_mode "tag" flags are merged
	// into a single canonical entry (flags cleared) - i.e. once *any* variant of a shape expansion is
	// found, the plain untagged variant is guaranteed to be requested, which is what the shape/data
	// interpolation code generator (which only understands the untagged shapes) needs to see.
	std::set<ShapeExpansion> FiniteElementCode::get_all_shape_expansions_in(GiNaC::ex inp, bool merge_no_jacobian, bool merge_expansion_modes, bool merge_no_hessian)
	{
		std::set<ShapeExpansion> res;
		for (GiNaC::const_preorder_iterator i = inp.preorder_begin(); i != inp.preorder_end(); ++i)
		{
			//			std::cout << *i << std::endl;
			if (GiNaC::is_a<GiNaC::GiNaCShapeExpansion>(*i))
			{
				auto &shapeexp = (GiNaC::ex_to<GiNaC::GiNaCShapeExpansion>(*i)).get_struct();
				//&		  	std::cout << "FOUND SHAPE EXPANSION  " << &shapeexp << std::endl;
				res.insert(shapeexp);
			}
			else if (GiNaC::is_a<GiNaC::GiNaCSubExpression>(*i))
			{
				GiNaC::GiNaCSubExpression se = GiNaC::ex_to<GiNaC::GiNaCSubExpression>(*i);
				std::set<ShapeExpansion> sub = get_all_shape_expansions_in(se.get_struct().expr, merge_no_jacobian, merge_expansion_modes, merge_no_hessian);
				for (auto &se : sub)
				{
					res.insert(se);
				}
			}
			else if (GiNaC::is_a<GiNaC::GiNaCMultiRetCallback>(*i))
			{
				//std::cout << "GOT MULTIRET CB "  << (*i) << std::endl;
				
				GiNaC::GiNaCMultiRetCallback se = GiNaC::ex_to<GiNaC::GiNaCMultiRetCallback>(*i);
				//std::cout << "GOT MULTIRET CB " << "INVOK "  << (se.get_struct().invok) << std::endl;
				//std::cout << "GOT MULTIRET CB " << "INVOK OP1 "  << (se.get_struct().invok.op(1)) << std::endl;
				std::set<ShapeExpansion> sub = get_all_shape_expansions_in(se.get_struct().invok.op(1), merge_no_jacobian, merge_expansion_modes, merge_no_hessian);
				for (auto &se : sub)
				{
					//std::cout << "GOT MULTIRET CB " << "INSERTING "  << GiNaC::GiNaCShapeExpansion(se) << std::endl;
					res.insert(se);
				}
			}
		}

		if (merge_no_jacobian || merge_expansion_modes || merge_no_hessian)
		{
			std::set<ShapeExpansion> newres;
			// Remove them which are already in there, but with a different value of the flags (e.g. no Jacobian)
			for (auto it = res.begin(); it != res.end();)
			{
				ShapeExpansion sp_test = *it;
				if (merge_no_jacobian && sp_test.no_jacobian)
				{
					sp_test.no_jacobian = false;
				}
				if (merge_no_hessian && sp_test.no_hessian)
				{
					sp_test.no_hessian = false;
				}
				if (merge_expansion_modes && sp_test.expansion_mode)
				{
					sp_test.expansion_mode = 0;
				}
				newres.insert(sp_test);
				it++;
			}
			res = newres;
		}
		return res;
	}

	// Collects every distinct TestFunction structure appearing anywhere in expression `inp` (simple
	// preorder scan; unlike get_all_shape_expansions_in, test functions are not expected inside
	// subexpression(...) wrappers since subexpressions may not depend on test functions - see
	// SubExpressionsToStructs above).
	std::set<TestFunction> FiniteElementCode::get_all_test_functions_in(GiNaC::ex inp)
	{
		std::set<TestFunction> res;
		for (GiNaC::const_preorder_iterator i = inp.preorder_begin(); i != inp.preorder_end(); ++i)
		{
			//			std::cout << *i << std::endl;
			if (GiNaC::is_a<GiNaC::GiNaCTestFunction>(*i))
			{
				auto &test = (GiNaC::ex_to<GiNaC::GiNaCTestFunction>(*i)).get_struct();
				//&		  	std::cout << "FOUND SHAPE EXPANSION  " << &shapeexp << std::endl;
				res.insert(test);
			}
		}
		return res;
	}

	
	// GiNaC tree-mapper that replaces occurrences of the auxiliary "mesh_x"/"mesh_y"/"mesh_z" fields
	// (used only so that partial_t(mesh_x) can be nonzero while partial_t(coordinate_x) is defined
	// to be identically zero, allowing mesh-velocity terms to be expressed) by the corresponding real
	// "coordinate_x"/"coordinate_y"/"coordinate_z" position field, once the mesh-velocity-specific
	// differentiation is no longer needed.
	class MeshToCoordinateShapes : public GiNaC::map_function
	{
	protected:
		FiniteElementCode *code;

	public:
		MeshToCoordinateShapes(FiniteElementCode *code_) : code(code_) {}
		GiNaC::ex operator()(const GiNaC::ex &inp) override
		{
			std::vector<std::string> dirs{"x", "y", "z"};
			if (GiNaC::is_a<GiNaC::GiNaCShapeExpansion>(inp))
			{
				auto &shapeexp = (GiNaC::ex_to<GiNaC::GiNaCShapeExpansion>(inp)).get_struct();
				for (auto d : dirs)
				{
					if (shapeexp.field->get_name() == "mesh_" + d)
					{
						ShapeExpansion repl = shapeexp;
						repl.field = shapeexp.field->get_space()->get_code()->get_field_by_name("coordinate_" + d);
						return GiNaC::GiNaCShapeExpansion(repl);
					}
				}
			}
			else if (GiNaC::is_a<GiNaC::GiNaCTestFunction>(inp))
			{
				auto &testf = (GiNaC::ex_to<GiNaC::GiNaCTestFunction>(inp)).get_struct();
				for (auto d : dirs)
				{
					if (testf.field->get_name() == "mesh_" + d)
					{
						TestFunction repl = testf;
						repl.field = testf.field->get_space()->get_code()->get_field_by_name("coordinate_" + d);
						return GiNaC::GiNaCTestFunction(repl);
					}
				}
			}

			return inp.map(*this);
		}
	};

	// Repeatedly applies ReplaceFieldsToNonDimFields until the expression stops changing (a single
	// pass may not fully expand nested field(...)/eval_in_domain(...) placeholders, since expanding
	// one placeholder can reveal further placeholders inside the code it expands to). If a pass
	// reports substitutions were made (repl_count>0) but the expression is unchanged, expansion is
	// stuck (e.g. a self-referential definition) and either raises an error or silently gives up,
	// depending on `raise_error`. Finally rewrites any remaining "mesh_*" pseudo-fields back to the
	// real "coordinate_*" position fields (see MeshToCoordinateShapes above) - this substitution is
	// deliberately done only once, at the very end, since keeping "mesh_*" distinct from
	// "coordinate_*" during expansion is what allows a nonzero mesh velocity partial_t(mesh_x) while
	// partial_t(coordinate_x) stays zero.
	GiNaC::ex FiniteElementCode::expand_placeholders(GiNaC::ex inp, std::string where, bool raise_error)
	{
		this->expanded_scales.clear();
		ReplaceFieldsToNonDimFields repl_dim_fields(this, where);
		GiNaC::ex repl = inp;
		do
		{
			GiNaC::ex old = repl;
			if (pyoomph_verbose)
				std::cout << "EXPAND LOOP START (@CODE " << this << "): " << repl << std::endl;
			repl_dim_fields.repl_count = 0;
			repl = repl_dim_fields(repl);
			if (pyoomph_verbose)
				std::cout << "EXPANDED " << repl_dim_fields.repl_count << " WITH RESULT: " << repl << std::endl;
			if (repl_dim_fields.repl_count && (old - repl).is_zero())
			{
				if (raise_error)
				{
					throw_runtime_error("Cannot expand the expression any further");
				}
				else
				{
					break;
				}
			}
		} while (repl_dim_fields.repl_count);

		// Finally, replace all mesh coordinates to normal coordinates
		// We just need this temporarily, since we want to be able to calculate partial_t(mesh_x), which is non-zero, whereas partial_t(coordinate_x) =0
		MeshToCoordinateShapes msh2x(this);

		return msh2x(repl);
	}

	// Assigns each field's global DoF-slot `index` (used to lay out nodal/internal data and equation
	// numbers consistently between a bulk element and its interfaces/attached elements). Must run
	// exactly once per code (guarded by `stage`); recurses into the bulk code first, since interface
	// element fields must reuse the *same* index as the corresponding bulk field so hanging-node and
	// DoF bookkeeping between bulk and interface elements stays consistent. Roughly:
	//   1. If there is a bulk element: recursively index it first, inherit its coordinate_space and
	//      coordinates_as_dofs flag, then copy over (register, without re-indexing) every position-space
	//      field from the bulk (skipping zeta/local coordinates beyond this element's dimension), and
	//      separately index any zeta-coordinate fields that only exist on this (lower-dimensional) code.
	//   2. Walk from the outermost ("deepest") bulk code down to this one, registering every
	//      continuous/DG field (except position fields) that is not yet present here, inheriting the
	//      index from the deepest bulk if it already has one there, else marking it "-2" (to be indexed
	//      before genuinely interface-local fields) or "-1" (ordinary interface-local field).
	//   3. Assign fresh consecutive indices to all remaining fields per space, "-2"-tagged bulk-inherited
	//      fields first, then interface-local ("-1"-tagged) fields.
	//   4. Position-space fields get their own separate index sequence, with mesh_x/mesh_y/mesh_z always
	//      pinned to indices 0/1/2 respectively (mirroring the fixed nodal coordinate ordering).
	// Finally marks stage=1 (fields now fixed) and, if this code has an "opposite side" interface code
	// that hasn't been indexed yet, recursively indexes it too (the stage-1 guard above prevents infinite
	// SideA<->SideB recursion).
	void FiniteElementCode::index_fields()
	{
		if (pyoomph_verbose)
			std::cout << "ENTERING INDEX FIELDS " << this << " @ STAGE " << stage << "  WITH BULK " << bulk_code << " AND OPP " << opposite_interface_code << std::endl;
		if (stage >= 1)
			return;

		for (unsigned int i = 0; i < myfields.size(); i++)
			myfields[i]->index = -1;
		int walking_index = 0;

		// If we have a bulk element, we need to make sure to map the data exactly
		if (bulk_code)
		{
			bulk_code->index_fields();
			this->coordinate_space = bulk_code->coordinate_space;

			/*	for (auto * s : spaces)
				{
					if (s->get_name()=="C2TB")
					{
						for (unsigned int i=0;i<myfields.size();i++)
						{
							if (myfields[i]->get_space()==s)
							{
								throw_runtime_error("Field "+myfields[i]->get_name()+" is defined on an interface on space C2TB, which is not possible.");
							}
						}
					}
				}*/

			coordinates_as_dofs = bulk_code->coordinates_as_dofs; // We need to transfer the information regarding moving nodes
																  // Copy the coordinates

			int bulk_coordinate_index_max=0;
			for (unsigned int j = 0; j < bulk_code->myfields.size(); j++)
			{
				FiniteElementSpace *bulkspace = bulk_code->myfields[j]->get_space();
				if (dynamic_cast<PositionFiniteElementSpace *>(bulkspace))
				{
					std::string n=bulk_code->myfields[j]->get_name();					
					if (n.rfind("zeta_coordinate_", 0) == 0)
					{
						continue; // We do not index or copy zeta coordinates
					}
					if (n.rfind("local_coordinate_", 0) == 0)
					{
						int index=std::stoi(n.substr(17));												
						if (index>element_dim) continue;
					}
					FiniteElementField *f = this->register_field(bulk_code->myfields[j]->get_name(), bulk_code->myfields[j]->get_space()->get_name());
					f->index = bulk_code->myfields[j]->index;
					f->set_defined_on_domain_equivalent_field(bulk_code->myfields[j]->get_defined_on_domain_equivalent_field());
					bulk_coordinate_index_max= std::max(bulk_coordinate_index_max, f->index);
				}
			}

			// Now check the zeta coordinates, which are not indexed yet
			for (unsigned int j = 0; j < this->myfields.size(); j++)
			{
				std::string n=this->myfields[j]->get_name();					
				if (n.rfind("zeta_coordinate_", 0) == 0)
				{
					this->myfields[j]->index = bulk_coordinate_index_max + 1; // We just take the next index
					bulk_coordinate_index_max++;
				}
			}

			// Go from deepest bulk upwards
			std::list<FiniteElementCode *> parent_codes;
			FiniteElementCode *deepest_bulk = bulk_code;
			while (deepest_bulk->bulk_code)
			{
				parent_codes.push_front(deepest_bulk);
				deepest_bulk = deepest_bulk->bulk_code;
			}
			parent_codes.push_front(deepest_bulk);

			for (auto pc : parent_codes)
			{
				for (unsigned int j = 0; j < pc->myfields.size(); j++)
				{
					FiniteElementSpace *bulkspace = pc->myfields[j]->get_space();
					if ((dynamic_cast<ContinuousFiniteElementSpace *>(bulkspace) || dynamic_cast<DGFiniteElementSpace *>(bulkspace)) && !dynamic_cast<PositionFiniteElementSpace *>(bulkspace))
					{
						FiniteElementField *fpresent = this->get_field_by_name(pc->myfields[j]->get_name());
						if (fpresent)
						{
							if (fpresent->get_space()->get_name() != pc->myfields[j]->get_space()->get_name())
							{
								throw_runtime_error("Field " + pc->myfields[j]->get_name() + " is defined on different spaces, namely " + fpresent->get_space()->get_name() + " and " + pc->myfields[j]->get_space()->get_name());
							}
							if (pc == deepest_bulk)
							{
								fpresent->index = pc->myfields[j]->index;
								if (fpresent->index >= walking_index)
									walking_index = fpresent->index + 1;
								
							}
							continue;
						}
						std::string pspacename = pc->myfields[j]->get_space()->get_name();
						/*   if (pspacename=="C2TB")
						   {
							pspacename="C2"; //Bubble does not transfer to the interfaces
						   }*/
						FiniteElementField *f = this->register_field(pc->myfields[j]->get_name(), pspacename);
						f->set_defined_on_domain_equivalent_field(pc->myfields[j]->get_defined_on_domain_equivalent_field());
						if (pc == deepest_bulk)
						{
							f->index = pc->myfields[j]->index;
							if (f->index >= walking_index)
								walking_index = f->index + 1;						
						}
						else
						{
							f->index = -1;
						}
					}
				}
			}
			// Now go again, and index all missing fields sorted by spaces
			for (auto pc : parent_codes)
			{
				for (auto *s : pc->spaces)
				{
					if ((dynamic_cast<ContinuousFiniteElementSpace *>(s) || dynamic_cast<DGFiniteElementSpace *>(s)) && !dynamic_cast<PositionFiniteElementSpace *>(s))
					{
						for (unsigned int j = 0; j < pc->myfields.size(); j++)
						{
							if (pc->myfields[j]->get_space() != s)
								continue;
							FiniteElementField *f = this->get_field_by_name(pc->myfields[j]->get_name());
							if (f->index == -1)
							{
								f->index = -2; // Prefer these
							}
						}
					}
				}
			}

			/*
		   for (unsigned int j=0;j<bulk_code->myfields.size();j++)
		   {
			   FiniteElementSpace * bulkspace=bulk_code->myfields[j]->get_space();
			   if (dynamic_cast<ContinuousFiniteElementSpace*>(bulkspace))
			   {
				   FiniteElementField *f=this->register_field(bulk_code->myfields[j]->get_name(),bulk_code->myfields[j]->get_space()->get_name());
				   // Set the index only for position space and if on deepest bulk space
				   bool must_set_index=dynamic_cast<PositionFiniteElementSpace*>(bulkspace);
				   // See if the field is defined on the deepest bulk
				   FiniteElementField * dbf = deepest_bulk->get_field_by_name(bulk_code->myfields[j]->get_name());
				   must_set_index|=(dbf!=NULL);
				   if (must_set_index)
				   {
					   f->index=bulk_code->myfields[j]->index;
					   if (!dynamic_cast<PositionFiniteElementSpace*>(bulkspace))
					   {
						   if (f->index>=walking_index) walking_index=f->index+1;
						 }
					  }
					  else
					  {
					   f->index=-1;
					  }

			   }
		   }*/
			// for (auto & f : myfields_backup) myfields.push_back(f);

			// Now the additional interface dofs. Here, we first do the discontinuous fields!
			/*
				for (auto * s : spaces)
				{
					std::cout << "INTERF " << s->get_name()<<std::endl;
					if (dynamic_cast<PositionFiniteElementSpace*>(s)) continue; //Position space has own indices
					std::cout << "	A1 " << s->get_name()<<std::endl;
					if (!dynamic_cast<DiscontinuousFiniteElementSpace*>(s)) continue; //Skip the continuous fields first, since they are additional nodal values
					std::cout << "	B1 " << s->get_name()<<std::endl;
					for (unsigned int i=0;i<myfields.size();i++)
					{
						if (myfields[i]->get_space()==s && myfields[i]->index==-1)
						{
							std::cout << "	ADDING " << myfields[i]->get_name() << " to index " << walking_index <<std::endl;
							myfields[i]->index=walking_index++;
						}

					}
				}
				//Now do the additional nodal values
				for (auto * s : spaces)
				{
								std::cout << "INTERF " << s->get_name()<<std::endl;
					if (dynamic_cast<PositionFiniteElementSpace*>(s)) continue; //Position space has own indices
					std::cout << "	A2 " << s->get_name()<<std::endl;
					if (!dynamic_cast<ContinuousFiniteElementSpace*>(s)) continue; //Only continuous spaces
					std::cout << "	B2 " << s->get_name()<<std::endl;
					for (unsigned int i=0;i<myfields.size();i++)
					{
						if (myfields[i]->get_space()==s && myfields[i]->index==-1)
						{
							std::cout << "	ADDING " << myfields[i]->get_name() << " to index " << walking_index <<std::endl;
							myfields[i]->index=walking_index++;
						}

					}
				}
				*/
		}
		//  	   else
		//  	   {
		for (auto *s : spaces)
		{
			if (dynamic_cast<PositionFiniteElementSpace *>(s))
				continue; // Position space has own indices
			for (unsigned int i = 0; i < myfields.size(); i++)
			{
				if (myfields[i]->get_space() == s && myfields[i]->index == -2)
				{
					myfields[i]->index = walking_index++;
				}
			}
			for (unsigned int i = 0; i < myfields.size(); i++)
			{
				if (myfields[i]->get_space() == s && myfields[i]->index == -1)
				{
					myfields[i]->index = walking_index++;
				}
			}
		}
		//	   }

		unsigned posindex = 0;
		for (auto *s : spaces)
		{
			if (!dynamic_cast<PositionFiniteElementSpace *>(s))
				continue; // Position space has own indices
			for (unsigned int i = 0; i < myfields.size(); i++)
			{
				if (myfields[i]->get_space() == s && myfields[i]->index == -1)
				{
					myfields[i]->index = posindex++;
				}
				// Patch the mesh indices
				if (myfields[i]->get_name() == "mesh_x")
					myfields[i]->index = 0;
				else if (myfields[i]->get_name() == "mesh_y")
					myfields[i]->index = 1;
				else if (myfields[i]->get_name() == "mesh_z")
					myfields[i]->index = 2;
			}
		}

		stage = 1;

		// Call after stetting stage=1 to prevent infinite loop SideA->SideB->SideA-> ...
		if (opposite_interface_code)
		{
			if (!opposite_interface_code->stage)
			{
				opposite_interface_code->index_fields();
			}
		}
	}

	// Registers a Z2-error-estimator flux expression (used by oomph-lib's Z2 error estimator to drive
	// mesh adaptation). The (possibly matrix/vector-valued, after evalm()) expression is expanded and
	// every non-constant scalar component is stored, either in the normal flux list or, if `for_eigen`,
	// in the separate list used for eigenproblem/azimuthal error estimation.
	void FiniteElementCode::add_Z2_flux(GiNaC::ex flux,bool for_eigen)
	{
		if (stage > 1)
			throw_runtime_error("Cannot add error estimators any more");
		GiNaC::ex expanded = this->expand_placeholders(flux, "Z2Flux");

		GiNaC::ex evm = expanded.evalm();
		if (GiNaC::is_a<GiNaC::matrix>(evm))
		{
			GiNaC::matrix m = GiNaC::ex_to<GiNaC::matrix>(evm);
			for (unsigned int i = 0; i < m.rows(); i++)
			{
				for (unsigned int j = 0; j < m.cols(); j++)
				{
					if (!GiNaC::is_a<GiNaC::numeric>(m(i, j)))
					{
						if (for_eigen)
						{
							this->Z2_fluxes_for_eigen.push_back(m(i, j));
						}
						else
						{
							this->Z2_fluxes.push_back(m(i, j));
						}
					}
				}
			}
		}
		else if (!GiNaC::is_a<GiNaC::numeric>(evm))
		{
			if (for_eigen)
			{
				this->Z2_fluxes_for_eigen.push_back(evm);
			}
			else
			{
				this->Z2_fluxes.push_back(evm);
			}
		}
	}

	// Expands all placeholders in `what` (see expand_placeholders) and then checks that the result is
	// dimensionally consistent: every base unit occurring in it must cancel out exactly (to power 1,
	// via the `sublist` substitution built from `base_units`), i.e. the final expression must be
	// nondimensional. If `collected_units_and_factor` is given, units are instead factored out and
	// returned there rather than required to fully cancel (used for expressions - such as scales or
	// individual vector/matrix components - that are allowed to carry a consistent overall unit); for
	// matrix-valued input, every component's units are cross-checked for consistency against the
	// first component's unit. Raises a detailed error (showing the offending term and units) if the
	// units cannot be separated or do not cancel/match as required.
	GiNaC::ex FiniteElementCode::expand_all_and_ensure_nondimensional(GiNaC::ex what, std::string where, GiNaC::ex *collected_units_and_factor)
	{
		GiNaC::ex expanded = this->expand_placeholders(what, where);
		if (expanded.is_zero())
			return 0;
		DrawUnitsOutOfSubexpressions units_out_of_subexpressions(this);
		GiNaC::ex repl = units_out_of_subexpressions(expanded);
		GiNaC::ex expa = repl.expand().evalm().normal();
		GiNaC::lst sublist;
		if (collected_units_and_factor)
		{
			if (GiNaC::is_a<GiNaC::matrix>(expa))
			{
				GiNaC::ex component;
				GiNaC::matrix expam = GiNaC::ex_to<GiNaC::matrix>(expa);
				std::vector<GiNaC::ex> newvect;
				for (unsigned int cd = 0; cd < expa.nops(); cd++)
				{

					GiNaC::ex factor, unit, rest;
					if (!expressions::collect_base_units(expa[cd], factor, unit, rest))
					{
						std::ostringstream oss;
						oss << std::endl
							<< "INPUT FORM:" << what << std::endl;
						oss << "EXPANDED FORM, component " << cd << ":" << expa[cd] << std::endl;
						oss << "CANNOT SEPARATE UNITS AND REST" << std::endl;
						oss << "NUMERICAL FACTOR: " << factor << std::endl;
						oss << "COLLECTED UNITS: " << unit << std::endl;
						oss << "REMAINING PART: " << rest << std::endl;
						throw_runtime_error("Found a inseparable units in the added expression:" + oss.str());
					}
					else
					{
						if (cd == 0)
						{
							for (auto &bu : base_units)
							{
								sublist.append(bu.second == 1);
							}
							component = rest;
							(*collected_units_and_factor) = factor * unit;
						}
						else
						{
							GiNaC::ex conversion = (factor * unit / (*collected_units_and_factor)).expand().evalm().normal();
							GiNaC::ex rest2;

							if (!expressions::collect_base_units(conversion, factor, unit, rest2))
							{
								std::ostringstream oss;
								oss << std::endl
									<< "INPUT FORM:" << what << std::endl;
								oss << "EXPANDED FORM, component " << cd << ":" << expa[cd] << std::endl;
								oss << "CANNOT SEPARATE UNITS AND REST, when comparing to base unit of first vector component, namely " << (*collected_units_and_factor) << std::endl;
								oss << "NUMERICAL FACTOR: " << factor << std::endl;
								oss << "COLLECTED UNITS: " << unit << std::endl;
								oss << "REMAINING PART: " << rest2 << std::endl;
								throw_runtime_error("Found a inseparable units in the added expression:" + oss.str());
							}

							component = rest * conversion;
						}
					}
					newvect.push_back(component);
				}
				expa = 0 + GiNaC::matrix(expam.cols(), expam.rows(), GiNaC::lst(newvect.begin(), newvect.end()));
			}
			else
			{
				GiNaC::ex factor, unit, rest;
				if (!expressions::collect_base_units(expa, factor, unit, rest))
				{
					std::ostringstream oss;
					oss << std::endl
						<< "INPUT FORM:" << what << std::endl;
					oss << "EXPANDED FORM:" << expa << std::endl;
					oss << "CANNOT SEPARATE UNITS AND REST" << std::endl;
					oss << "NUMERICAL FACTOR: " << factor << std::endl;
					oss << "COLLECTED UNITS: " << unit << std::endl;
					oss << "REMAINING PART: " << rest << std::endl;
					throw_runtime_error("Found a inseparable units in the added expression:" + oss.str());
				}
				else
				{
					for (auto &bu : base_units)
					{
						sublist.append(bu.second == 1);
					}
					expa = rest;
					(*collected_units_and_factor) = factor * unit;
				}
			}
		}
		else
		{
			for (auto &bu : base_units)
			{
				if (expa.has(bu.second))
				{
					std::ostringstream oss;
					oss << std::endl
						<< "INPUT FORM:" << what << std::endl;
					oss << "EXPANDED FORM:" << expa << std::endl;
					GiNaC::ex factor, unit, rest;
					if (!expressions::collect_base_units(expa, factor, unit, rest))
					{
						oss << "CANNOT SEPARATE UNITS AND REST" << std::endl;
					}
					else
					{
						oss << "UNITS AND REST ARE SEPARABLE" << std::endl;
						// Last chance:
						if (unit.is_equal(1))
						{
							sublist.append(bu.second == 1);
							continue;
						}
					}
					oss << "NUMERICAL FACTOR: " << factor << std::endl;
					oss << "COLLECTED UNITS: " << unit << std::endl;
					oss << "REMAINING PART: " << rest << std::endl;

					throw_runtime_error("Found a dimensional contribution in the added expression:" + oss.str());
				}
				sublist.append(bu.second == 1);
			}
		}
		// GiNaC::ex final_contrib=repl.subs(sublist);
		GiNaC::ex finalres = expa.subs(sublist);
		return finalres;
	}

	// NOTE: this looks like unfinished/debugging code - it unconditionally prints diagnostic output
	// and calls exit(0), so it currently terminates the whole process rather than returning a usable
	// symbolic derivative to the caller. Left untouched (out of scope for a comments-only pass), but
	// flagged here since it is surprising behavior for a public API method.
	GiNaC::ex FiniteElementCode::derive_expression(const GiNaC::ex &what, const GiNaC::ex by)
	{
		if (stage == 0)
			index_fields();
		GiNaC::ex expanded = this->expand_placeholders(what, "DerivativeNumer");
		if (expanded.is_zero())
			return 0;
		GiNaC::ex bw = this->expand_placeholders(by, "DerivativeDenom");
		std::cout << "TRY TO DIFF " << expanded << " WRTO " << by << std::endl;
		GiNaC::ex deriv = expressions::Diff(expanded, bw);
		DrawUnitsOutOfSubexpressions units_out_of_subexpressions(this);
		GiNaC::ex repl = units_out_of_subexpressions(deriv);
		std::cout << " RES " << repl << std::endl;
		exit(0);
		return 0;
		/*		DrawUnitsOutOfSubexpressions units_out_of_subexpressions(this);
				GiNaC::ex repl=units_out_of_subexpressions(expanded);
				GiNaC::ex expa=repl.expand().normal();*/
	}

	// Adds a user-supplied weak-form contribution `add` to the currently active residual
	// (residual[residual_index]). Expands all placeholders, and - for ODE elements only - multiplies
	// by the (unity) integration measure if none is present yet, so ODE residuals integrate
	// correctly alongside PDE ones. Pulls dimensional units out of subexpressions and verifies that,
	// after cancellation, the residual is fully nondimensional (every base unit's substituted test
	// value must leave the expression unchanged) - raising a detailed error otherwise, since a
	// dimensionally-inconsistent residual signals a modeling error in the user's equations. Also
	// rejects matrix/vector-valued residual terms (they must be fully contracted to scalars by the
	// user) and, if `warn_on_large_numerical_factor` is set, warns/errors when any numerical
	// coefficient in the expanded residual exceeds that threshold (a common sign of an accidental
	// unit-scaling mistake). The lengthy commented-out block below is leftover exploratory code for a
	// (currently disabled) stricter check of Eulerian/Lagrangian dx-degree consistency.
	void FiniteElementCode::add_residual(GiNaC::ex add, bool)
	{
		if (stage > 1)
			throw_runtime_error("Cannot add residuals any more");
		if (stage == 0)
			index_fields();
		// Checking the contribution

		//      GiNaC::ex expanded=expand_all_and_ensure_nondimensional(add);

		GiNaC::ex expanded = this->expand_placeholders(add, "Residual");
		if (expanded.is_zero())
			return;
		if (this->_is_ode_element())
		{
			unsigned ldeg = expanded.ldegree(this->get_dx(false));		
			if (ldeg==0)
			{
			  expanded = expanded * get_dx(false);
			}
		}			
		// TODO: Further checking

		/*
				// Check for Eulerian and Lagrangian integerals
				if (expanded.degree(this->get_dx(false)) > 1)
					throw_runtime_error("Found a dx contribution of higher than linear order");
				unsigned ldeg = expanded.ldegree(this->get_dx(false));
				if (ldeg < 0)
				{
					throw_runtime_error("Negative dx degree");
				}
				if (ldeg == 0)
				{
					GiNaC::ex remain = expanded.coeff(get_dx(false), 0);
					if (expanded.degree(this->get_dx(true)) > 1)
						throw_runtime_error("Found a dX contribution of higher than linear order");
					unsigned ldeg = expanded.ldegree(this->get_dx(true));
					if (this->_is_ode_element() && ldeg == 0)
					{
						expanded = expanded * get_dx(false);
					}
					else if (ldeg == 0 && allow_contributions_without_dx)
					{
					}
					// This part could be a Lagrangian contribution
					else if (ldeg < 1)
					{
						// Now it can only be a nodal_delta
						unsigned nddeg = expanded.degree(this->get_nodal_delta());
						if (ldeg <= 0 && nddeg == 0)
						{
								std::cerr << "IN: " << expanded << std::endl;
							throw_runtime_error("Found a dx (Eulerian or Lagrangian) contribution of lower than linear order");
						}
						if (nddeg > 1)
						{
							throw_runtime_error("Nonlinear nodal_delta degree");
						}
					}
				}
				else
				{
					// Check for mixed contribution
					GiNaC::ex remain = expanded.coeff(get_dx(false), 1);
					if (remain.has(get_dx(true)))
						throw_runtime_error("Mixed Lagragian and Eulerian integral contribution");
					if (remain.has(this->get_nodal_delta()))
						throw_runtime_error("Mixed spatial integral and nodal delta contribution");
				}
		*/

		DrawUnitsOutOfSubexpressions units_out_of_subexpressions(this);
		GiNaC::ex repl = units_out_of_subexpressions(expanded);
		GiNaC::ex expa = repl.expand().normal();
		GiNaC::lst sublist;
		for (auto &bu : base_units)
		{
			if (expa.has(bu.second))
			{
				std::ostringstream oss;
				oss << std::endl
					<< "INPUT FORM:" << add << std::endl;
				oss << "EXPANDED FORM:" << expa << std::endl;
				GiNaC::ex factor, unit, rest;
				if (!expressions::collect_base_units(expa, factor, unit, rest))
				{
					oss << "CANNOT SEPARATE UNITS AND REST" << std::endl;
				}
				else
				{
					oss << "UNITS AND REST ARE SEPARABLE" << std::endl;
					// Last chance:
					if (unit.is_equal(1))
					{
						sublist.append(bu.second == 1);
						continue;
					}
				}
				oss << "NUMERICAL FACTOR: " << factor << std::endl;
				oss << "COLLECTED UNITS: " << unit << std::endl;
				oss << "REMAINING PART: " << rest << std::endl;
				oss << "USED SCALES: " << std::endl;
				for (auto entry : this->expanded_scales)
				{
					oss << "  " << entry.first << " = " << entry.second << std::endl;
				}

				throw_runtime_error("Found a dimensional contribution in the added residual:" + oss.str());
			}
			sublist.append(bu.second == 1);
		}

		GiNaC::ex final_contrib = repl.subs(sublist);
		//		 GiNaC::ex final_contrib=expa.subs(sublist);
		//		  GiNaC::ex final_contrib=expanded;
		if (pyoomph_verbose)
			std::cout << "Adding residual " << final_contrib << std::endl;

		for (GiNaC::const_preorder_iterator i = final_contrib.preorder_begin(); i != final_contrib.preorder_end(); ++i)
		{
			if (GiNaC::is_a<GiNaC::matrix>(*i))
			{
				std::ostringstream oss;
				oss << std::endl
					<< *i << std::endl;
				throw_runtime_error("Apparently, the added residual contains vectors or matrices. Please contract everything to scalar via dot or double_dot. Problematic term:" + oss.str());
			}
		}

		if (warn_on_large_numerical_factor)
		{
			GiNaC::ex expa = final_contrib.expand();
			double maxf = 0.0;
			for (GiNaC::const_postorder_iterator it = expa.postorder_begin(); it != expa.postorder_end(); it++)
			{
				if (GiNaC::is_a<GiNaC::numeric>(*it))
				{
					double f = GiNaC::ex_to<GiNaC::numeric>(*it).to_double();
					if (fabs(f) > maxf)
						maxf = fabs(f);
				}
			}
			if (maxf > fabs(warn_on_large_numerical_factor))
			{
				std::ostringstream oss;
				oss << "WARNING: NUMERICAL FACTOR OF " << maxf << " IN " << std::endl
					<< final_contrib << std::endl
					<< "STEMMING FROM " << std::endl
					<< add << std::endl;
				if (warn_on_large_numerical_factor > 0)
				{
					std::cout << oss.str();
				}
				else
				{
					throw_runtime_error(oss.str());
				}
			}
		}

		residual[residual_index] += final_contrib;
	}

	// Emits the opening of the standard integration-point loop ("for(ipt=0;...)"), filling the shape
	// buffer for the current point (fill_shape_buffer_for_point), and declaring the local dx/dX/
	// dx_unity integration-weight variables actually needed (dx only if `eulerian_part` is present,
	// dX only if `lagrangian_part` is present). write_generic_spatial_integration_footer closes the
	// loop opened here. When use_shared_shape_buffer_during_multi_assemble is enabled, the shape
	// buffer is only re-filled if not already shared from an enclosing multi-assemble call.
	void FiniteElementCode::write_generic_spatial_integration_header(std::ostream &os, std::string indent, GiNaC::ex eulerian_part, GiNaC::ex lagrangian_part, std::string required_table_and_flag)
	{
		if (this->use_shared_shape_buffer_during_multi_assemble)
		{
			os << indent << "unsigned n_int_pt=(my_func_table->during_shared_multi_assembling ? 1 : shapeinfo->n_int_pt);" << std::endl;
			os << indent << "for(unsigned ipt=0;ipt<n_int_pt;ipt++)" << std::endl;
		}
		else
		{
			os << indent << "for(unsigned ipt=0;ipt<shapeinfo->n_int_pt;ipt++)" << std::endl;
		}
		os << indent << "{" << std::endl;
		if (this->use_shared_shape_buffer_during_multi_assemble)
		{
			os << indent << "   if (!my_func_table->during_shared_multi_assembling)" << std::endl;
			os << indent << "   {" << std::endl;
		}
		os << indent << "  my_func_table->fill_shape_buffer_for_point(ipt, " << required_table_and_flag << ");" << std::endl;
		if (this->use_shared_shape_buffer_during_multi_assemble)
		{
			os << indent << "   }" << std::endl;
		}
		if (!eulerian_part.is_zero())
		{
			os << indent << "  const double dx = shapeinfo->int_pt_weight[0];" << std::endl;
		}
		if (!lagrangian_part.is_zero())
		{
			os << indent << "  const double dX = shapeinfo->int_pt_weight_Lagrangian;" << std::endl;
		}
		os << indent << "  const double dx_unity = shapeinfo->int_pt_weight_unity;" << std::endl;		
	}
	void FiniteElementCode::write_generic_spatial_integration_footer(std::ostream &os, std::string indent)
	{
		os << indent << "}" << std::endl;
	}

	// Emits a loop over all element nodes for residual contributions expressed via a Kronecker-delta
	// "nodal_delta" (point contributions at nodes, as opposed to spatial dx/dX integrals); the code
	// comment acknowledges this is not the most efficient approach (delta_ij is zero off-diagonal)
	// but is kept simple. write_generic_nodal_delta_footer closes the loop opened here.
	void FiniteElementCode::write_generic_nodal_delta_header(std::ostream &os, std::string indent)
	{
		os << indent << "//This is not the best approach... But it is okay to loop over all nodes, although delta_ij=0 for all i!=j" << std::endl;
		os << indent << "for(unsigned ipt=0;ipt<eleminfo->nnode;ipt++)" << std::endl;
		os << indent << "{" << std::endl;
	}
	void FiniteElementCode::write_generic_nodal_delta_footer(std::ostream &os, std::string indent)
	{
		os << indent << "}" << std::endl;
	}

	// Emits the C call that invokes a registered multi-return Python/C callback (multi_return_calls[i]),
	// storing its `nret` return values and `nret*nargs` derivatives into freshly acquired arrays
	// (multi_ret_i / dmulti_ret_i). If this call's arguments themselves reference other multi-return
	// calls (nested calls), those are recursively emitted first via `multi_return_calls_written` (a
	// set of already-written call indices shared across the whole subexpression pass, to avoid
	// emitting the same call twice). Prefers a directly generated C implementation
	// (multi_ret_ccode_<index>, from multi_return_ccodes) over the generic Python callback dispatch
	// (my_func_table->invoke_multi_ret) when available; if the callback additionally requests
	// C-vs-Python cross-checking (debug_c_code_epsilon>0), both are emitted.
	void FiniteElementCode::write_code_multi_ret_call(std::ostream &os, std::string indent, GiNaC::ex for_what, unsigned i, std::set<int> *multi_return_calls_written, GiNaC::ex *invok)
	{
		if (multi_return_calls_written && invok)
		{
			// Recursively write the inner multi-rets first
			for (GiNaC::const_preorder_iterator it = (*invok).preorder_begin(); it != (*invok).preorder_end(); ++it)
			{
				if (GiNaC::is_a<GiNaC::GiNaCMultiRetCallback>(*it))
				{
					GiNaC::ex invok2 = GiNaC::ex_to<GiNaC::GiNaCMultiRetCallback>(*it).get_struct().invok;
					int mr_index = this->resolve_multi_return_call(invok2);
					if (mr_index < 0)
					{
						std::ostringstream oss;
						oss << std::endl
							<< "When looking for:" << std::endl
							<< invok2 << std::endl
							<< "Present:" << std::endl;
						for (unsigned int _i = 0; _i < multi_return_calls.size(); _i++)
							oss << multi_return_calls[_i] << std::endl;
						throw_runtime_error("Cannot resolve multi-return call" + oss.str());
					}
					if (!multi_return_calls_written->count(mr_index))
					{
						this->write_code_multi_ret_call(os, indent, for_what, mr_index, multi_return_calls_written, &invok2);
						multi_return_calls_written->insert(mr_index);
					}
				}
			}
		}
		int nret = GiNaC::ex_to<GiNaC::numeric>(multi_return_calls[i].op(2)).to_int();
		int nargs = GiNaC::ex_to<GiNaC::lst>(multi_return_calls[i].op(1)).nops();
		GiNaC::print_FEM_options csrc_opts;
		csrc_opts.for_code = this;
		if (nret > 0)
		{
			os << indent << "PYOOMPH_AQUIRE_ARRAY(double,multi_ret_" << i << "," << nret << ");" << std::endl;
			os << indent << "PYOOMPH_AQUIRE_ARRAY(double,dmulti_ret_" << i << "," << nret << "*" << nargs << ");" << std::endl;
			CustomMultiReturnExpressionBase *func = GiNaC::ex_to<GiNaC::GiNaCCustomMultiReturnExpressionWrapper>(multi_return_calls[i].op(0)).get_struct().cme;
			if (!CustomMultiReturnExpressionBase::code_map.count(func))
			{
				CustomMultiReturnExpressionBase::code_map[func] = CustomMultiReturnExpressionBase::code_map.size();
			}
			unsigned index = CustomMultiReturnExpressionBase::code_map[func];
			if (multi_return_ccodes.count(func))
			{
				os << indent << "multi_ret_ccode_" << multi_return_ccodes[func].first << "(flag,(double []){";
				for (int l = 0; l < nargs; l++)
				{
					print_simplest_form(multi_return_calls[i].op(1).op(l), os, csrc_opts);
					if (l < nargs - 1)
						os << ", ";
				}
				os << "} , multi_ret_" << i << ", dmulti_ret_" << i;
				os << ", " << nargs << ", " << nret << ");" << std::endl
				   << std::endl;
				if (func->debug_c_code_epsilon > 0)
				{
					os << indent << "//DEBUG CALL WITH EPSILON " << func->debug_c_code_epsilon << std::endl;
					os << indent << "my_func_table->invoke_multi_ret(my_func_table, " << index << " , flag|128, (double []){";
					for (int l = 0; l < nargs; l++)
					{
						print_simplest_form(multi_return_calls[i].op(1).op(l), os, csrc_opts);
						if (l < nargs - 1)
							os << ", ";
					}
					os << "} , multi_ret_" << i << ", dmulti_ret_" << i;
					os << ", " << nargs << ", " << nret << ");" << std::endl
					   << std::endl;
				}
			}
			else
			{
				os << indent << "my_func_table->invoke_multi_ret(my_func_table, " << index << " , flag, (double []){";
				for (int l = 0; l < nargs; l++)
				{
					print_simplest_form(multi_return_calls[i].op(1).op(l), os, csrc_opts);
					if (l < nargs - 1)
						os << ", ";
				}
				os << "} , multi_ret_" << i << ", dmulti_ret_" << i;
				os << ", " << nargs << ", " << nret << ");" << std::endl
				   << std::endl;
			}
		}
	}

	// Common-subexpression-elimination (CSE) code emitter: extracts every expressions::subexpression(...)
	// marker from `for_what` into `this->subexpressions` (via SubExpressionsToStructs, unless `hessian`
	// is set - Hessian passes reuse the subexpression list already built for the first-order pass) and
	// returns `for_what` with those markers replaced by GiNaCSubExpression references. For each
	// collected subexpression it then emits:
	//   1. a "double <cvar> = <expr>;" declaration computing its value once, and
	//   2. inside "if (flag) { ... }" (Jacobian/Hessian assembly only runs when flag requests
	//      derivatives), a "double d_<cvar>_d_<field>" declaration plus assignment for every field
	//      the subexpression actually depends on (subexpressions[j].req_fields), computed via
	//      pyoomph::expressions::diff and then DerivedShapeExpansionsToUnity to turn the "which shape
	//      basis was it derived w.r.t." bookkeeping into a plain 0/1 factor. These pre-computed
	//      per-subexpression derivatives are what let the outer Jacobian/Hessian code reuse a
	//      subexpression's value and derivative without re-differentiating the (potentially large)
	//      subexpression body at every Jacobian entry - this is the key common-subexpression
	//      elimination performed by the code generator to keep generated code size and cost manageable.
	// Any multi-return callback invocations referenced by a subexpression are emitted (recursively,
	// respecting dependency order) via write_code_multi_ret_call before the subexpression that needs
	// them. Nested-coordinate (position-space) dependencies are skipped unless coordinates_as_dofs is
	// set, and only the current-time history slot (time_history_index==0) is differentiated here.
	GiNaC::ex FiniteElementCode::write_code_subexpressions(std::ostream &os, std::string indent, GiNaC::ex for_what, const std::set<ShapeExpansion> &, bool hessian)
	{
		GiNaC::ex res;
		os << " //Subexpressions // TODO: Check whether it is constant to take it out of the loop" << std::endl;

		if (!hessian)
		{
			subexpressions.clear();
			multi_return_calls.clear();
			SubExpressionsToStructs SE_to_struct(this);
			res = SE_to_struct(for_what);
			subexpressions = SE_to_struct.subexpressions;
		}
		else
		{
			res = for_what;
		}
		/*
			 for (GiNaC::const_postorder_iterator i = res.postorder_begin(); i != res.postorder_end(); ++i)
			 {
				if (GiNaC::is_a<GiNaC::GiNaCSubExpression>(*i)) //TODO: Check constant numbers or simple expressions and untreat them as subexpressions
				{
					bool found=false;
					auto & st=GiNaC::ex_to<GiNaC::GiNaCSubExpression>(*i).get_struct();
					for (unsigned int j=0;j<subexpressions.size();j++) if (st.expr.is_equal(subexpressions[j].get_expression())) {found=true; break;}
					if (!found)
					{
						 std::set<ShapeExpansion> sub_shapeexps=get_all_shape_expansions_in(st.expr);
						 std::set<TestFunction> sub_testfuncs=get_all_test_functions_in(st.expr);
						 if (!sub_testfuncs.empty()) { throw_runtime_error("Subexpressions may not depend on test functions!"); }
						 subexpressions.push_back(FiniteElementCodeSubExpression(st.expr,GiNaC::symbol("subexpr_"+std::to_string(subexpressions.size())),sub_shapeexps) );
						 //st.fe_subexpr=&(subexpressions[subexpressions.size()-1]);
					}
				}
			 }
			 */

		// Remove the subexpression functions and fill the objects

		GiNaC::print_FEM_options csrc_opts;
		csrc_opts.for_code = this;

		//	 ReplaceShapeExpansionToCVars shape_to_c(this,&required_shapeexps);
		//	 ReplaceSubexprToCVar rem_subexpr(this);
		//	 os << "  //Subexpressions" << std::endl;
		// if (!hessian)
		std::set<int> multi_return_calls_written;
		for (unsigned int j = 0; j < subexpressions.size(); j++)
		{

			// Test if the subexpression has results of multi-return calls. If so, we must write these earlier
			GiNaC::ex sexpr = subexpressions[j].get_expression();
			for (GiNaC::const_preorder_iterator it = sexpr.preorder_begin(); it != sexpr.preorder_end(); ++it)
			{
				if (GiNaC::is_a<GiNaC::GiNaCMultiRetCallback>(*it))
				{
					GiNaC::ex invok = GiNaC::ex_to<GiNaC::GiNaCMultiRetCallback>(*it).get_struct().invok;
					int mr_index = this->resolve_multi_return_call(invok);
					if (mr_index < 0)
					{
						std::ostringstream oss;
						oss << std::endl
							<< "When looking for:" << std::endl
							<< invok << std::endl
							<< "Present:" << std::endl;
						for (unsigned int _i = 0; _i < multi_return_calls.size(); _i++)
							oss << multi_return_calls[_i] << std::endl;
						throw_runtime_error("Cannot resolve multi-return call" + oss.str());
					}
					if (!multi_return_calls_written.count(mr_index))
					{
						this->write_code_multi_ret_call(os, indent, for_what, mr_index, &multi_return_calls_written, &invok);
						multi_return_calls_written.insert(mr_index);
					}
				}
			}

			//  if (hessian) throw_runtime_error("Hessian subexpressions!");

			/*if (!GiNaC::is_zero(GiNaC::imag_part(subexpressions[j].get_expression())))
			{
				os << "    double RE_" << subexpressions[j].get_cvar() << " = ";
				print_simplest_form(GiNaC::real_part(subexpressions[j].get_expression()), os, csrc_opts);
				os << ";" << std::endl;
				os << "    double IM_" << subexpressions[j].get_cvar() << " = ";
				print_simplest_form(GiNaC::imag_part(subexpressions[j].get_expression()), os, csrc_opts);
				os << ";" << std::endl;
			}
			else
			{*/
				os << "    double " << subexpressions[j].get_cvar() << " = ";
				print_simplest_form(subexpressions[j].get_expression(), os, csrc_opts);
				os << ";" << std::endl;
			//}
		}

		// if (!hessian) //Derivatives of subexpressions are treated in another way in Hessian
		
		{
			csrc_opts.in_subexpr_deriv = true;
			os << "    //Derivatives of subexpressions" << std::endl;
			std::set<std::string> subexpr_defined_written_in_hessian;
			for (unsigned int j = 0; j < subexpressions.size(); j++)
			{

				for (auto &f : subexpressions[j].req_fields)
				{
					if (!coordinates_as_dofs && dynamic_cast<PositionFiniteElementSpace *>(f.field->get_space()))
						continue;
					if (f.time_history_index != 0)
						continue;
					//				GiNaC::ex dsub=subexpressions[j].expr_subst.diff(f.get_cpp_symbol());
					//				if (!dsub.is_zero())
					//				{
					std::string wrto = f.get_spatial_interpolation_name(this);
					std::ostringstream derivname;
					derivname << "d_" << subexpressions[j].get_cvar() << "_d_" << wrto;
					if (hessian && subexpr_defined_written_in_hessian.count(derivname.str()))
						continue;
					os << "    double " << derivname.str() << ";" << std::endl;
					subexpr_defined_written_in_hessian.insert(derivname.str());
					//	subexpressions[j].derivsyms[f.get_cpp_symbol()]=GiNaC::symbol(derivname.str());
					//			}
				}
				// Additional derivatives with respect to coordinates
			}
			if (!hessian)
				os << "    if (flag)" << std::endl;
			os << "    {" << std::endl;

			std::set<std::string> subexpr_rhs_written_in_hessian;
			for (unsigned int j = 0; j < subexpressions.size(); j++)
			{
				for (auto &f : subexpressions[j].req_fields)
				{
					if (!coordinates_as_dofs && dynamic_cast<PositionFiniteElementSpace *>(f.field->get_space()))
						continue;
					if (f.time_history_index != 0)
						continue;
					//				GiNaC::ex dsub=subexpressions[j].expr_subst.diff(f.get_cpp_symbol());
					//				if (!dsub.is_zero())
					//				{
					std::string wrto = f.get_spatial_interpolation_name(this);
					__deriv_subexpression_wrto = &f;
					if (pyoomph::pyoomph_verbose)
					{
						std::cout << "DERIVING SUBSEXPRESSION " << subexpressions[j].get_expression() << " BY " << f.field->get_symbol() << ", more specifically by " << (0 + GiNaC::GiNaCShapeExpansion(f)) << std::endl;
					}
					if (hessian) throw_runtime_error("Hessian subexpressions!");
					__derive_only_by_expansion_mode=this->get_derive_jacobian_by_expansion_mode();
					GiNaC::ex dsdf = pyoomph::expressions::diff(subexpressions[j].get_expression(), f.field->get_symbol());
					__derive_only_by_expansion_mode=NULL;
					__deriv_subexpression_wrto = NULL;
					DerivedShapeExpansionsToUnity deriv_se_to_1(f.basis,f.dt_order,f.dt_scheme); // Map all other expanded basis functions to zero to separate between e.g. d/dx or nonderived shapes
					GiNaC::ex dsub = deriv_se_to_1(dsdf);
					if (pyoomph::pyoomph_verbose)
					{
						std::cout << "DERIVING SUBSEXPRESSION RESULT " << dsdf << " OR " << dsub << std::endl;
					}
					
					// if (!dsub.is_zero())
					{
						std::ostringstream derivname;
						derivname << "d_" << subexpressions[j].get_cvar() << "_d_" << wrto;
						if (hessian && subexpr_rhs_written_in_hessian.count(derivname.str())) continue;
						os << "     " << derivname.str() << " = ";
						subexpr_rhs_written_in_hessian.insert(derivname.str());
						// dsub.evalf().print(GiNaC::print_csrc_FEM(os,&csrc_opts));
						// GiNaC::factor(GiNaC::normal(GiNaC::expand(GiNaC::expand(dsub).evalf()))).print(GiNaC::print_csrc_FEM(os,&csrc_opts));
						print_simplest_form(dsub, os, csrc_opts);
						os << ";" << std::endl; // " // " << dsub << std::endl;
					}
					//	subexpressions[j].derivsyms[f.get_cpp_symbol()]=GiNaC::symbol(derivname.str());
					//			}
				}
			}

			os << "    }" << std::endl;
		}
		for (unsigned int i = 0; i < multi_return_calls.size(); i++)
		{
			if (!multi_return_calls_written.count(i))
			{
				this->write_code_multi_ret_call(os, indent, for_what, i);
				multi_return_calls_written.insert(i);
			}
		}

		return res;
	}

	// Scans `expr` for occurrences of NormalSymbol, SpatialIntegralSymbol, and ElementSizeSymbol
	// nodes (i.e. dependence on the outward normal, the dx/dX integration measure's history, or the
	// element size) and marks the corresponding shape data (normal vectors, position-space psi shapes,
	// element-size arrays, ...) as required for the `for_what` code-generation pass via
	// mark_shapes_required(), resolving which domain (this code / its bulk / the opposite interface
	// code / that interface's bulk) actually owns the position space each symbol refers to. This
	// ensures the shape/interpolation code emitted elsewhere actually computes the data these symbols
	// will be substituted by when the expression is printed as C code.
	void FiniteElementCode::mark_further_required_fields(GiNaC::ex expr, const std::string &for_what)
	{
		// Mark other requirements
		for (GiNaC::const_preorder_iterator i = expr.preorder_begin(); i != expr.preorder_end(); ++i)
		{
			if (GiNaC::is_a<GiNaC::GiNaCNormalSymbol>(*i))
			{
				const pyoomph::NormalSymbol &sp = GiNaC::ex_to<GiNaC::GiNaCNormalSymbol>(*i).get_struct();
				if (sp.get_code() == this || sp.get_code() == NULL)
				{
					this->mark_shapes_required(for_what, this->get_my_position_space(), "normal");
					if (this->bulk_code)
					{
						this->mark_shapes_required(for_what, this->bulk_code->get_my_position_space(), "psi");
					}
					else
					{
						this->mark_shapes_required(for_what, this->get_my_position_space(), "psi");
					}
				}
				else if (this->bulk_code && sp.get_code() == this->bulk_code)
				{
					this->mark_shapes_required(for_what, this->bulk_code->get_my_position_space(), "normal");
					if (this->bulk_code->bulk_code)
					{
						this->mark_shapes_required(for_what, this->bulk_code->bulk_code->get_my_position_space(), "psi");
					}
					else
					{
						this->mark_shapes_required(for_what, this->bulk_code->get_my_position_space(), "psi");
					}
				}
				else if (this->opposite_interface_code && sp.get_code() == this->opposite_interface_code)
				{
					this->mark_shapes_required(for_what, this->opposite_interface_code->get_my_position_space(), "normal");
					if (this->opposite_interface_code->bulk_code)
					{
						this->mark_shapes_required(for_what, this->opposite_interface_code->bulk_code->get_my_position_space(), "psi");
					}
					else
					{
						this->mark_shapes_required(for_what, this->opposite_interface_code->get_my_position_space(), "psi");
					}
				}
				else
				{
					throw_runtime_error("Normal of this domain not accessible");
				}
			}
			else if (GiNaC::is_a<GiNaC::GiNaCSpatialIntegralSymbol>(*i))
			{				
				auto &sp = GiNaC::ex_to<GiNaC::GiNaCSpatialIntegralSymbol>(*i).get_struct();				
				if (!sp.is_lagrangian())
				{
					  if (sp.history_step!=0) this->mark_shapes_required(for_what, this->get_my_position_space(), "history_integral_dx"+std::to_string(sp.history_step));
				}									
			}
			else if (GiNaC::is_a<GiNaC::GiNaCElementSizeSymbol>(*i))
			{
				const pyoomph::ElementSizeSymbol &sp = GiNaC::ex_to<GiNaC::GiNaCElementSizeSymbol>(*i).get_struct();
				std::string es_name = (sp.is_lagrangian() ? "elemsize_Lagrangian" : "elemsize_Eulerian");
				es_name += (sp.is_with_coordsys() ? "" : "_cartesian");
				if (sp.get_code() == this || sp.get_code() == NULL)
				{
					this->mark_shapes_required(for_what, this->get_my_position_space(), es_name);
					if (this->coordinates_as_dofs && !sp.is_lagrangian())
					{
						this->mark_shapes_required(for_what, this->get_my_position_space(), "psi");
					}
				}
				else if (this->bulk_code && sp.get_code() == this->bulk_code)
				{
					this->mark_shapes_required(for_what, this->bulk_code->get_my_position_space(), es_name);
					if (this->bulk_code->coordinates_as_dofs && !sp.is_lagrangian())
					{
						this->mark_shapes_required(for_what, this->bulk_code->get_my_position_space(), "psi");
					}
				}
				else if (this->opposite_interface_code && sp.get_code() == this->opposite_interface_code)
				{
					this->mark_shapes_required(for_what, this->opposite_interface_code->get_my_position_space(), es_name);
					if (this->opposite_interface_code->coordinates_as_dofs && !sp.is_lagrangian())
					{
						this->mark_shapes_required(for_what, this->opposite_interface_code->get_my_position_space(), "psi");
					}
				}
				else if (this->opposite_interface_code->bulk_code && sp.get_code() == this->opposite_interface_code->bulk_code)
				{
					this->mark_shapes_required(for_what, this->opposite_interface_code->bulk_code->get_my_position_space(), es_name);
					if (this->opposite_interface_code->bulk_code->coordinates_as_dofs && !sp.is_lagrangian())
					{
						this->mark_shapes_required(for_what, this->opposite_interface_code->bulk_code->get_my_position_space(), "psi");
					}
				}
				else
				{
					throw_runtime_error("Element size of this domain not accessible");
				}
			}
		}
	}

	// Extracts the part of residual expression `inp` that is multiplied by an Eulerian dx (if
	// `eulerian`) and/or a Lagrangian dX (if `lagrangian`) spatial-integral symbol, i.e. the part
	// that must be assembled inside the integration-point loop. Different SpatialIntegralSymbol
	// instances (e.g. tagged with different history_step or expansion_mode) are treated as distinct
	// "variables" whose linear coefficient is collected and re-multiplied back in, so multiple
	// differently-tagged dx/dX contributions present in the same residual are all picked up rather
	// than just the plain untagged one.
	GiNaC::ex FiniteElementCode::extract_spatial_integral_part(const GiNaC::ex &inp, bool eulerian, bool lagrangian)
	{
		//std::set<GiNaC::GiNaCSpatialIntegralSymbol> dx_symbs;
		std::set<GiNaC::ex, GiNaC::ex_is_less> dx_symbs;
		// First, gather all dx terms
		for (GiNaC::const_preorder_iterator i = inp.preorder_begin(); i != inp.preorder_end(); ++i)
		{
			if (GiNaC::is_a<GiNaC::GiNaCSpatialIntegralSymbol>(*i))
			{
				if (pyoomph_verbose)
				{
					std::cout << "	CHECKING DX CONTIBUTION " << (*i) << " FOR eulerian " << eulerian << " lagrangian " << lagrangian << " ALREADY FOUND " << dx_symbs.count(0+GiNaC::ex_to<GiNaC::GiNaCSpatialIntegralSymbol>(*i)) << std::endl;
					for (auto &dx : dx_symbs)
						std::cout << 	" DIFFERENCE BETWEEN the current " << (*i) << " and the already added " << dx << " is : " << (*i) - dx << " IS ZERO " << GiNaC::is_zero((*i) - dx) << std::endl;
				}
				auto &sp = GiNaC::ex_to<GiNaC::GiNaCSpatialIntegralSymbol>(*i).get_struct();
				if ((sp.is_lagrangian() && lagrangian) || (!sp.is_lagrangian() && eulerian)) // Only the ones of interest
				{
					if (pyoomph_verbose)
					{
						std::cout << " ADDING IT TO THE SET " << (*i) << " which has currently " << dx_symbs.size() << " elements " << std::endl;
					}
					/*if (eulerian)
					{
					  if (sp.history_step!=0) this->integral_dx_history_required.insert(sp.history_step);
					}*/
					dx_symbs.insert(0+GiNaC::ex_to<GiNaC::GiNaCSpatialIntegralSymbol>(*i));
					if (pyoomph_verbose)
					{
						std::cout << " Afterwards, the set has " << dx_symbs.size() << " elements " << std::endl;
					}
				}
			}
		}
		// And now assemble it again
		GiNaC::ex res = 0;
		for (auto &dx : dx_symbs)
		{
			if (pyoomph_verbose)
				std::cout << "	USING DX CONTRIBUTION " << dx << " FOR eulerian " << eulerian << " lagrangian " << lagrangian << std::endl;
			GiNaC::ex contrib = inp.coeff(dx, 1);
			// We could check here for another dx in contrib. If present, raise error
			res += contrib * dx;
		}
		return res;
	}

	// Top-level generator for the exact (analytical) Hessian-vector-product C function `funcname` of
	// residual `resi`. High-level structure:
	//   1. Split off the Eulerian/Lagrangian spatially-integrated part of the residual (nodal-delta
	//      Hessian contributions are not supported and raise an error if present), strip any leftover
	//      subexpression(...) markers back to plain form, then re-wrap them via a fresh
	//      SubExpressionsToStructs instance (__SE_to_struct_hessian) dedicated to this Hessian pass.
	//   2. Call write_generic_RJM_contribution(..., hessian=true) on every FiniteElementSpace, which
	//      (via write_generic_Hessian_contribution, see above) performs the actual double symbolic
	//      differentiation and accumulates, as a side effect, the global __all_Hessian_shapeexps/
	//      testfuncs/indices_required sets describing everything the emitted code needs at runtime.
	//   3. Emit the function header/signature and the boilerplate that allocates the dense
	//      n_dof^3 Hessian (and, if needed, mass-Hessian) scratch buffers, based on the `flag`
	//      parameter selecting between building the full third-order tensor, a directional
	//      Hessian-vector product, or its transpose (see the ASSEMBLE_*/SET_DIRECTIONAL_* macros used
	//      at the end).
	//   4. Emit per-field nodal-index constants, time-history interpolation, and (for D0-like spaces)
	//      pre-loop spatial interpolation, then - if there is a nonzero spatially-integrated part -
	//      the full integration-point loop with per-point interpolation and the CSE subexpression code
	//      (write_code_subexpressions), followed by the actual Hessian-assembly code collected into
	//      `osm` in step 2 above.
	// `__in_hessian` is set for the duration of this function so that GiNaC derivative()
	// implementations elsewhere know a Hessian derivative is in progress.
	bool FiniteElementCode::write_generic_Hessian(std::ostream &os, std::string funcname, GiNaC::ex resi, bool)
	{
		__in_hessian = true;
		bool has_contribs = false;
		std::ostringstream osh; // Header
		std::ostringstream osm; // Main contribution

		__all_Hessian_shapeexps.clear();
		__all_Hessian_testfuncs.clear();
		__all_Hessian_indices_required.clear();
		if (__SE_to_struct_hessian)
			delete __SE_to_struct_hessian;
		__SE_to_struct_hessian = new SubExpressionsToStructs(this);

		GiNaC::ex spatial_integral_portion_Eulerian = extract_spatial_integral_part(resi, true, false);	  // resi.coeff(get_dx(false), 1) * get_dx(false);
		GiNaC::ex spatial_integral_portion_Lagrangian = extract_spatial_integral_part(resi, false, true); // resi.coeff(get_dx(true), 1) * get_dx(true);
		GiNaC::ex spatial_integral_portion_NodalDelta = resi.coeff(get_nodal_delta(), 1);

		// if (!spatial_integral_portion_Lagrangian.is_zero()) this->mark_shapes_required("ResJac["+std::to_string(residual_index)+"]",spaces[0],"psi");
		GiNaC::ex spatial_integral_portion = spatial_integral_portion_Eulerian + spatial_integral_portion_Lagrangian;

		// REMOVE ALL SUBEXPRESSIONS FOR THE TIME BEING
		RemoveSubexpressionsByIndentity rem_ses(this);
		spatial_integral_portion = rem_ses(spatial_integral_portion);

		spatial_integral_portion = (*__SE_to_struct_hessian)(spatial_integral_portion);

		osm << "    //START: Contribution of the spaces" << std::endl;
		osm << "    double _H_contrib;" << std::endl;
		for (auto *sp : allspaces)
		{
			has_contribs = sp->write_generic_RJM_contribution(this, osm, "    ", spatial_integral_portion, true) || has_contribs;
		}
		osm << "    //END: Contribution of the spaces" << std::endl;

		if (!has_contribs)
		{
			__in_hessian = false;
			return has_contribs;
		}

		if (!spatial_integral_portion_NodalDelta.is_zero())
		{
			throw_runtime_error("Nodal Delta in Hessian!");
		}

		osh << "static void " << funcname << "(const JITElementInfo_t * eleminfo, const JITShapeInfo_t * shapeinfo,const double * Y, double *Cs, double *product,unsigned numvectors,unsigned flag)" << std::endl;
		osh << "{" << std::endl;
		osh << "  unsigned n_dof=shapeinfo->jacobian_size; // Since product, Y and Cs might be larger than eleminfo->ndof... " << std::endl;
		osh << "  int local_eqn, local_unknown, local_deriv;" << std::endl;
		osh << "  unsigned nummaster,nummaster2,nummaster3;" << std::endl;
		osh << "  double hang_weight,hang_weight2,hang_weight3;" << std::endl;
		osh << "  const double * t=shapeinfo->t;" << std::endl;
		osh << "  const double * dt=shapeinfo->dt;" << std::endl;
		osh << "  double * hessian_buffer;" << std::endl; // TODO: Potentially with allocate array instead
		osh << "  double * hessian_M_buffer;" << std::endl;
		//		if (this->assemble_hessian_by_symmetry)
		//		{
		osh << "  if (flag==3) " << std::endl;
		osh << "  {" << std::endl;
		osh << "    hessian_buffer=product; //Assign directly to the product" << std::endl;
		osh << "  }" << std::endl;
		osh << "  else" << std::endl;
		osh << "  {" << std::endl;
		osh << "    hessian_buffer=(double*)calloc(n_dof*n_dof*n_dof,sizeof(double));" << std::endl;
		osh << "  }" << std::endl;
		osh << "  if (flag==2 || flag==5) " << std::endl;
		osh << "  {" << std::endl;
		osh << "    hessian_M_buffer=(double*)calloc(n_dof*n_dof*n_dof,sizeof(double));" << std::endl;
		osh << "  }" << std::endl;
		osh << "  else if (flag==3) " << std::endl;
		osh << "  {" << std::endl;
		osh << "    hessian_M_buffer=Cs;" << std::endl;
		osh << "  }" << std::endl;
		/*		}
				else
				{
				  osh << "  PYOOMPH_AQUIRE_ARRAY(double, dJij_Yj_duk, n_dof*n_dof) " << std::endl;
				  osh << "  for (unsigned iv=0;iv<n_dof*n_dof;iv++) dJij_Yj_duk[iv]=0.0;" << std::endl;
				  osh << "  if (flag==2) " << std::endl;
				  osh << "  {" << std::endl;
				  osh << "    hessian_M_buffer=(double*)calloc(n_dof*n_dof*n_dof,sizeof(double));" << std::endl;
				  osh << "  }" << std::endl;
				osh << "  else if (flag==3) " << std::endl;
				  osh << "  {" << std::endl;
				  osh << "    hessian_M_buffer=Cs;" << std::endl;
				  osh << "  }" << std::endl;
				}*/
		std::set<ShapeExpansion> all_shapeexps = __all_Hessian_shapeexps;
		std::set<TestFunction> all_testfuncs = __all_Hessian_testfuncs;
		std::set<FiniteElementField *> indices_required = __all_Hessian_indices_required;

		std::set<ShapeExpansion> merged_shapeexps;
		for (auto &sp : all_shapeexps)
		{
			indices_required.insert(sp.field);
			max_dt_order = std::max(max_dt_order, sp.dt_order);
			this->mark_shapes_required("Hessian[" + std::to_string(residual_index) + "]", sp.field->get_space(), sp.basis);
			ShapeExpansion sp_for_merge = sp;
			sp_for_merge.nodal_coord_dir = -1;
			sp_for_merge.nodal_coord_dir2 = -1;
			sp_for_merge.is_derived = false;
			sp_for_merge.is_derived_other_index = false;
			sp_for_merge.expansion_mode = 0;
			merged_shapeexps.insert(sp_for_merge);
		}
		for (auto &tf : all_testfuncs)
		{
			indices_required.insert(tf.field);
			this->mark_shapes_required("Hessian[" + std::to_string(residual_index) + "]", tf.field->get_space(), tf.basis);
		}
		mark_further_required_fields(resi, "Hessian[" + std::to_string(residual_index) + "]");
		if (this->coordinates_as_dofs)
		{
			//			throw_runtime_error("You cannot use analyical Hessian yet if a mesh has moving nodes");

			for (auto d : std::vector<std::string>{"x", "y", "z"})
			{
				if (this->get_field_by_name("coordinate_" + d))
				{
					indices_required.insert(this->get_field_by_name("coordinate_" + d));
					if (this->bulk_code)
					{
						indices_required.insert(this->bulk_code->get_field_by_name("coordinate_" + d));
						if (this->bulk_code->bulk_code)
						{
							indices_required.insert(this->bulk_code->bulk_code->get_field_by_name("coordinate_" + d));
						}
					}					
				}
			}
		}
		if (this->opposite_interface_code && this->opposite_interface_code->coordinates_as_dofs)
		{
			for (auto d : std::vector<std::string>{"x", "y", "z"})
			{
				if (this->opposite_interface_code->get_field_by_name("coordinate_" + d))
				{
					indices_required.insert(this->opposite_interface_code->get_field_by_name("coordinate_" + d));
					if (this->opposite_interface_code->bulk_code)
					{
						indices_required.insert(this->opposite_interface_code->bulk_code->get_field_by_name("coordinate_" + d));
					}
				}
			}
		}

		std::vector<std::string> indices_lines;
		for (auto *f : indices_required)
		{
			//osh << "  const unsigned " << f->get_nodal_index_str(this) << " = " << f->index << ";" << std::endl;
			indices_lines.push_back("  const unsigned " + f->get_nodal_index_str(this) + " = " + std::to_string(f->index) + ";");
		}
		std::sort(indices_lines.begin(), indices_lines.end());
		for (auto &l : indices_lines)
		{
			osh << l << std::endl;
		}
		osh << "  //START: Precalculate time derivatives of the necessary data" << std::endl;
		for (auto *sp : allspaces)
		{
			sp->write_nodal_time_interpolation(this, osh, "  ", all_shapeexps);
		}
		osh << "  //END: Precalculate time derivatives of the necessary data" << std::endl
			<< std::endl;
		// First assign the "interpolated D0" values
		for (auto *sp : allspaces)
		{
			if (!sp->need_interpolation_loop())
			{
				sp->write_spatial_interpolation(this, osh, "    ", all_shapeexps, this->coordinates_as_dofs, true);
			}
		}

		if (!spatial_integral_portion.is_zero())
		{
			osh << "  //START: Spatial integration loop" << std::endl;
			std::string required_name = "&(my_func_table->shapes_required_Hessian[" + std::to_string(residual_index) + "]), 3";
			write_generic_spatial_integration_header(osh, "  ", spatial_integral_portion_Eulerian, spatial_integral_portion_Lagrangian, required_name);
			std::set<ShapeExpansion> spatial_shape_exps = get_all_shape_expansions_in(spatial_integral_portion); // TODO: This is wrong!
			std::set<ShapeExpansion> shape_intersect;

			/*			std::cout << "all_shapeexps" << "__________________" << std::endl;
						for (auto & s :  all_shapeexps)
						{
						  std::cout << GiNaC::GiNaCShapeExpansion(s) << std::endl;
						}
						std::cout << "merged_shapeexps" << "__________________" << std::endl;
						for (auto & s :  merged_shapeexps)
						{
						  std::cout << GiNaC::GiNaCShapeExpansion(s) << std::endl;
						}
			*/

			//			std::set_intersection(spatial_shape_exps.begin(), spatial_shape_exps.end(), all_shapeexps.begin(), all_shapeexps.end(), std::inserter(shape_intersect, shape_intersect.begin()));
			//			std::set<ShapeExpansion> spatial_shape_exps = get_all_shape_expansions_in(spatial_integral_portion);
			osh << "    //START: Interpolate all required fields" << std::endl;
			for (auto *sp : allspaces)
			{
				if (sp->need_interpolation_loop())
				{
					//					sp->write_spatial_interpolation(this, osh, "    ", spatial_shape_exps, this->coordinates_as_dofs,true);
					//					sp->write_spatial_interpolation(this, osh, "    ", all_shapeexps, this->coordinates_as_dofs,true);
					sp->write_spatial_interpolation(this, osh, "    ", merged_shapeexps, this->coordinates_as_dofs, true);
					//					sp->write_spatial_interpolation(this, osh, "    ", shape_intersect, false,true);
				}
			}
			osh << "    // SUBEXPRESSIONS" << std::endl
				<< std::endl;
			spatial_integral_portion = this->write_code_subexpressions(osh, "     ", spatial_integral_portion, spatial_shape_exps, true);
		}
		osh << "    //END: Interpolate all required fields" << std::endl
			<< std::endl;

		osh << std::endl;

		if (!has_contribs)
		{
			__in_hessian = false;
			return has_contribs;
		}

		os << osh.str();
		os << osm.str();
		write_generic_spatial_integration_footer(os, "  ");
		os << "  //END: Spatial integration loop" << std::endl
		   << std::endl;
		//	 os << " // TODO"  << std::endl;
		//  	 os << "  printf(\"TODO: Implement HessianVector products\\n\");" << std::endl;
		/*		if (!this->assemble_hessian_by_symmetry)
				{
					os << "  if (flag==3)" << std::endl;
					os << "  {" << std::endl;
					os << "    " << std::endl;
					os << "  }" << std::endl;
					os << "  else if (!flag)" << std::endl;
					os << "  {" << std::endl;
					os << "    ASSEMBLE_HESSIAN_VECTOR_PRODUCTS_FROM(dJij_Yj_duk,Cs,n_dof,numvectors,product)" << std::endl;
					os << "  }" << std::endl;
					os << "  else" << std::endl;
					os << "  {" << std::endl;
					os << "    SET_DIRECTIONAL_HESSIAN_FROM(dJij_Yj_duk,n_dof,product)" << std::endl;
					os << "  }" << std::endl;
				}
				else
				{*/
		os << "  if (!flag)" << std::endl;
		os << "  {" << std::endl;
		os << "     ASSEMBLE_SYMMETRIC_HESSIAN_VECTOR_PRODUCTS_FROM(Y,Cs,n_dof,numvectors,product)" << std::endl;
		os << "     free(hessian_buffer);" << std::endl;
		os << "  }" << std::endl;
		os << "  else if (flag!=3) " << std::endl;
		os << "  {" << std::endl;
		os << "     if (flag==5 || flag==4) " << std::endl;
		os << "     {" << std::endl;
		os << "        SET_DIRECTIONAL_SYMMETRIC_HESSIAN_FROM_TRANSPOSED(hessian_buffer,Y,n_dof,product)" << std::endl;
		os << "     }" << std::endl;
		os << "     else" << std::endl;
		os << "     {" << std::endl;
		os << "        SET_DIRECTIONAL_SYMMETRIC_HESSIAN_FROM(hessian_buffer,Y,n_dof,product)" << std::endl;
		os << "     }" << std::endl;
		os << "     free(hessian_buffer); " << std::endl;
		os << "  }" << std::endl;
		os << "  if (flag==2)" << std::endl;
		os << "  {" << std::endl;
		os << "     SET_DIRECTIONAL_SYMMETRIC_HESSIAN_FROM(hessian_M_buffer,Y,n_dof,Cs)" << std::endl;
		os << "     free(hessian_M_buffer);" << std::endl;
		os << "  }" << std::endl;
		os << "  else if (flag==5)" << std::endl;
		os << "  {" << std::endl;
		os << "     SET_DIRECTIONAL_SYMMETRIC_HESSIAN_FROM_TRANSPOSED(hessian_M_buffer,Y,n_dof,Cs)" << std::endl;
		os << "     free(hessian_M_buffer);" << std::endl;
		os << "  }" << std::endl;
		//		}
		os << "}" << std::endl;
		__in_hessian = false;
		return has_contribs;
	}

	// Top-level generator for the combined Residual/Jacobian/Mass-matrix C function `funcname` of
	// residual `resi` (the first-order analytical-differentiation counterpart of write_generic_Hessian
	// above). Emits the function signature and boilerplate declarations, then - analogous to
	// write_generic_Hessian's structure - collects all shape expansions/test functions/required field
	// indices occurring in `resi`, emits nodal-index constants and time/spatial interpolation code,
	// the CSE subexpression code, and finally delegates the actual residual/Jacobian/mass-matrix
	// emission to write_generic_RJM_contribution() on every FiniteElementSpace inside the
	// integration-point loop.
	void FiniteElementCode::write_generic_RJM(std::ostream &os, std::string funcname, GiNaC::ex resi, bool)
	{
		__in_hessian = false;
		os << "static void " << funcname << "(const JITElementInfo_t * eleminfo, const JITShapeInfo_t * shapeinfo,double * PYOOMPH_RESTRICT residuals, double * PYOOMPH_RESTRICT jacobian, double * PYOOMPH_RESTRICT mass_matrix,unsigned flag)" << std::endl;
		os << "{" << std::endl;
		os << "  int local_eqn, local_unknown;" << std::endl;
		os << "  bool _has_residual_contribution,_has_jacobian_contribution;" << std::endl;

		// TODO: Only if hanging allowed
		os << "  unsigned nummaster,nummaster2;" << std::endl;
		os << "  double hang_weight,hang_weight2;" << std::endl;

		os << "  const double * t=shapeinfo->t;" << std::endl;
		os << "  const double * dt=shapeinfo->dt;" << std::endl;
		if (stage == 0)
			index_fields();

		std::set<ShapeExpansion> all_shapeexps = get_all_shape_expansions_in(resi, true);

		std::set<TestFunction> all_testfuncs = get_all_test_functions_in(resi);
		std::set<FiniteElementField *> indices_required;
		for (auto &sp : all_shapeexps)
		{
			if (pyoomph_verbose)
				std::cout << "RJM " << this << " HAVING SHAPE EXPANSION " << sp.field->get_name() << "@" << sp.field->get_space()->get_name() << " @ code " << sp.field->get_space()->get_code() << std::endl;
			indices_required.insert(sp.field);
			max_dt_order = std::max(max_dt_order, sp.dt_order);
			// if (!dynamic_cast<D0FiniteElementSpace*>(sp.field->get_space()))
			//{
			this->mark_shapes_required("ResJac[" + std::to_string(residual_index) + "]", sp.field->get_space(), sp.basis);
			//}
		}
		for (auto &tf : all_testfuncs)
		{
			indices_required.insert(tf.field);
			// if (!dynamic_cast<D0FiniteElementSpace*>(tf.field->get_space()))
			//{
			this->mark_shapes_required("ResJac[" + std::to_string(residual_index) + "]", tf.field->get_space(), tf.basis);
			//}
		}

		// Mark other requirements
		mark_further_required_fields(resi, "ResJac[" + std::to_string(residual_index) + "]");


		if (this->coordinates_as_dofs)
		{
			for (auto d : std::vector<std::string>{"x", "y", "z"})
			{
				if (this->get_field_by_name("coordinate_" + d))
				{
					indices_required.insert(this->get_field_by_name("coordinate_" + d));
					if (this->bulk_code)
					{
						indices_required.insert(this->bulk_code->get_field_by_name("coordinate_" + d));
						if (this->bulk_code->bulk_code)
						{
							indices_required.insert(this->bulk_code->bulk_code->get_field_by_name("coordinate_" + d));
						}
					}					
				}
			}
		}
		if (this->opposite_interface_code && this->opposite_interface_code->coordinates_as_dofs)
		{
			for (auto d : std::vector<std::string>{"x", "y", "z"})
			{
				if (this->opposite_interface_code->get_field_by_name("coordinate_" + d))
				{
					indices_required.insert(this->opposite_interface_code->get_field_by_name("coordinate_" + d));
					if (this->opposite_interface_code->bulk_code)
					{
						indices_required.insert(this->opposite_interface_code->bulk_code->get_field_by_name("coordinate_" + d));
					}
				}
			}
		}

		std::vector<std::string> indices_lines;
		for (auto *f : indices_required)
		{
			//os << "  const unsigned " << f->get_nodal_index_str(this) << " = " << f->index << ";" << std::endl;
			indices_lines.push_back("  const unsigned " + f->get_nodal_index_str(this) + " = " + std::to_string(f->index) + ";");
		}
		std::sort(indices_lines.begin(), indices_lines.end());
		for (auto &l : indices_lines)
		{
			os << l << std::endl;
		}

		os << "  //START: Precalculate time derivatives of the necessary data" << std::endl;
		for (auto *sp : allspaces)
		{
			sp->write_nodal_time_interpolation(this, os, "  ", all_shapeexps);
		}
		os << "  //END: Precalculate time derivatives of the necessary data" << std::endl
		   << std::endl;

		// First assign the "interpolated D0" values
		for (auto *sp : allspaces)
		{
			if (!sp->need_interpolation_loop())
			{
				sp->write_spatial_interpolation(this, os, "    ", all_shapeexps, false, false);
			}
		}

		GiNaC::ex spatial_integral_portion_Eulerian = extract_spatial_integral_part(resi, true, false);	  // resi.coeff(get_dx(false), 1) * get_dx(false);
		if (pyoomph::pyoomph_verbose)
		{
			std::cout << "Full residual: " << resi << std::endl;
			std::cout << "Eulerian part of the residual: " << spatial_integral_portion_Eulerian << std::endl;
		}
		GiNaC::ex spatial_integral_portion_Lagrangian = extract_spatial_integral_part(resi, false, true); // resi.coeff(get_dx(true), 1) * get_dx(true);
		GiNaC::ex spatial_integral_portion_NodalDelta = resi.coeff(get_nodal_delta(), 1);

		if (!spatial_integral_portion_Lagrangian.is_zero())
			this->mark_shapes_required("ResJac[" + std::to_string(residual_index) + "]", spaces[0], "psi");

		// GiNaC::ex spatial_integral_portion=GiNaC::diff(resi,this->spatial_integral_dx); //TODO
		GiNaC::ex spatial_integral_portion = spatial_integral_portion_Eulerian + spatial_integral_portion_Lagrangian;

		if (!spatial_integral_portion.is_zero())
		{
			os << "  //START: Spatial integration loop" << std::endl;
			std::string required_name = "&(my_func_table->shapes_required_ResJac[" + std::to_string(residual_index) + "]), flag";
			write_generic_spatial_integration_header(os, "  ", spatial_integral_portion_Eulerian, spatial_integral_portion_Lagrangian, required_name);

			std::set<ShapeExpansion> spatial_shape_exps = get_all_shape_expansions_in(spatial_integral_portion);
			os << "    //START: Interpolate all required fields" << std::endl;
			for (auto *sp : allspaces)
			{
				if (sp->need_interpolation_loop())
				{
					sp->write_spatial_interpolation(this, os, "    ", spatial_shape_exps, this->coordinates_as_dofs, false);
				}
			}
			os << "    //END: Interpolate all required fields" << std::endl
			   << std::endl;

			os << std::endl;

			os << "    // SUBEXPRESSIONS" << std::endl
			   << std::endl;
			spatial_integral_portion = this->write_code_subexpressions(os, "     ", spatial_integral_portion, spatial_shape_exps, false);

			os << "    //START: Contribution of the spaces" << std::endl;
			os << "    double _res_contrib,_J_contrib;" << std::endl;
			for (auto *sp : allspaces)
			{
				sp->write_generic_RJM_contribution(this, os, "    ", spatial_integral_portion, false);
			}
			os << "    //END: Contribution of the spaces" << std::endl;

			write_generic_spatial_integration_footer(os, "  ");
			os << "  //END: Spatial integration loop" << std::endl
			   << std::endl;
		}

		if (!spatial_integral_portion_NodalDelta.is_zero())
		{
			os << "  //START: Nodal delta" << std::endl;
			os << "  //END: Nodal delta" << std::endl;
			//	write_generic_nodal_delta_header(os,"  "); //TODO

			std::set<ShapeExpansion> nodal_shape_exps = get_all_shape_expansions_in(spatial_integral_portion_NodalDelta);
			os << "    //START: Interpolate all required fields" << std::endl;
			os << "    double _res_contrib,_J_contrib;" << std::endl;
			// 			throw_runtime_error("TODO: Spatial interpolation! Psi->nodal_Psi");
			for (auto *sp : allspaces)
			{
				/*	if (sp->need_interpolation_loop())
					{
						throw_runtime_error("Non-D0 nodal delta");
						//sp->write_spatial_interpolation(this,os,"    ",nodal_shape_exps,this->coordinates_as_dofs);
					}*/
				/* 	std::cout << spatial_integral_portion_NodalDelta << std::endl;
									std::cerr << spatial_integral_portion_NodalDelta << std::endl;*/
				for (auto &se : nodal_shape_exps)
				{
					if (se.field->get_space() != sp)
					{
						continue;
					}
					if (dynamic_cast<D0FiniteElementSpace *>(se.field->get_space()))
					{
						sp->write_generic_RJM_contribution(this, os, "    ", spatial_integral_portion_NodalDelta, false);
					}
					else
					{
						throw_runtime_error("Non-D0 nodal delta");
					}
				}
			}
			os << "    //END: Interpolate all required fields" << std::endl
			   << std::endl;

			//			write_generic_nodal_delta_footer(os,"  "); //TODO

			os << std::endl;
		}

		os << "}" << std::endl
		   << std::endl;
	}

	// Emits the top-of-file preprocessor boilerplate for the generated element's C source: the
	// shared-library marker, an optional flag enabling Hessian-by-symmetry assembly, and the
	// jitbridge header that declares the runtime data structures (JITElementInfo_t, JITShapeInfo_t,
	// hanging-node macros, ...) used throughout the rest of the generated code.
	void FiniteElementCode::write_code_header(std::ostream &os)
	{
		os << "#define JIT_ELEMENT_SHARED_LIB" << std::endl;
		if (this->assemble_hessian_by_symmetry)
		{
			os << "#define ASSEMBLE_HESSIAN_VIA_SYMMETRY" << std::endl;
		}
		os << "#include \"jitbridge.h\"" << std::endl
		   << std::endl;
		os << "static JITFuncSpec_Table_FiniteElement_t * my_func_table;" << std::endl
		   << std::endl;
	}

	// Shared implementation for user-registered "integral expressions" (global quantities integrated
	// over the element, `integrate=true`) and "local expressions" (pointwise quantities evaluated at
	// the first integration point only, `integrate=false`): emits a single dispatcher function
	// `funcname(eleminfo, shapeinfo, index)` that, given the numeric `index` of the requested
	// expression (in `exprs`, keyed by name), computes and returns its value. All expressions are
	// gathered into one combined expression (each multiplied by a distinct GiNaC::wild() placeholder,
	// so structurally-identical terms across different expressions are not accidentally cancelled/
	// merged by GiNaC's simplification) purely to determine the union of required shape data once;
	// the actual per-expression C code is then emitted individually inside a "switch(index)" block. For
	// local expressions, also guards against a name collision with an already-registered field of the
	// same name (which would make the two indistinguishable to users of the generated evaluation API).
	void FiniteElementCode::write_code_integral_or_local_expressions(std::ostream &os, std::map<std::string, GiNaC::ex> &exprs, std::map<std::string, GiNaC::ex> &units, std::string funcname, std::string reqname, bool integrate)
	{
		os << "static double " << funcname << "(const JITElementInfo_t * eleminfo, const JITShapeInfo_t * shapeinfo, unsigned index)" << std::endl;
		os << "{" << std::endl;
		os << "  const unsigned flag=0;" << std::endl;
		GiNaC::ex gathered;
		unsigned cnt = 0;
		for (auto &e : exprs)
		{
			gathered += e.second * GiNaC::wild(cnt++); // Wild important to prevent that terms are cancelling out
		}

		os << "  const double * t=shapeinfo->t;" << std::endl
		   << "  const double * dt=shapeinfo->dt;" << std::endl
		   << std::endl;

		std::set<ShapeExpansion> all_shapeexps = get_all_shape_expansions_in(gathered);
		std::set<TestFunction> all_testfuncs = get_all_test_functions_in(gathered);
		if (!all_testfuncs.empty())
		{
			throw_runtime_error("Found test function in a custom integral/local expression");
		}
		std::set<FiniteElementField *> indices_required;
		for (auto &sp : all_shapeexps)
		{
			indices_required.insert(sp.field);
			max_dt_order = std::max(max_dt_order, sp.dt_order);
			// if (!dynamic_cast<D0FiniteElementSpace*>(sp.field->get_space()))
			//{
			this->mark_shapes_required(reqname, sp.field->get_space(), sp.basis);
			//}
		}

		mark_further_required_fields(gathered, reqname);

		for (auto *f : indices_required)
		{
			os << "  const unsigned " << f->get_nodal_index_str(this) << " = " << f->index << ";" << std::endl;
		}

		os << "  //START: Precalculate time derivatives of the necessary data" << std::endl;
		for (auto *sp : allspaces)
		{
			sp->write_nodal_time_interpolation(this, os, "  ", all_shapeexps);
		}
		os << "  //END: Precalculate time derivatives of the necessary data" << std::endl
		   << std::endl;

		if (integrate)
		{
			os << "  double res=0.0;" << std::endl;
			os << "  for(unsigned ipt=0;ipt<shapeinfo->n_int_pt;ipt++)" << std::endl;
			os << "  {" << std::endl;
			os << "    my_func_table->fill_shape_buffer_for_point(ipt, &(my_func_table->shapes_required_IntegralExprs), 0);" << std::endl;
		}
		else
		{
			os << "  double res;" << std::endl;
			os << "  unsigned ipt=0;" << std::endl;
			//	os << "  my_func_table->fill_shape_buffer_for_point(ipt, &(my_func_table->shapes_required_IntegralExprs), 0);" << std::endl;
		}

		std::set<ShapeExpansion> spatial_shape_exps = get_all_shape_expansions_in(gathered);
		os << "    //START: Interpolate all required fields" << std::endl;
		for (auto *sp : allspaces)
		{
			sp->write_spatial_interpolation(this, os, "    ", spatial_shape_exps, false, false);
		}
		os << "    //END: Interpolate all required fields" << std::endl
		   << std::endl;

		GiNaC::print_FEM_options csrc_opts;
		csrc_opts.for_code = this;

		RemoveSubexpressionsByIndentity sub_to_id(this);
		std::set<int> multi_return_calls_written;
		std::map<std::string, GiNaC::ex> sexprs;
		for (auto &e : exprs)
		{
			GiNaC::ex flux = 0 + e.second;
			flux = sub_to_id(flux.subs(GiNaC::lst{expressions::x, expressions::y, expressions::z}, {_x, _y, _z}));
			sexprs[e.first] = flux;
			for (GiNaC::const_preorder_iterator it = flux.preorder_begin(); it != flux.preorder_end(); ++it)
			{
				if (GiNaC::is_a<GiNaC::GiNaCMultiRetCallback>(*it))
				{
					GiNaC::ex invok = GiNaC::ex_to<GiNaC::GiNaCMultiRetCallback>(*it).get_struct().invok;
					int mr_index = this->resolve_multi_return_call(invok);
					if (mr_index < 0)
					{
						std::ostringstream oss;
						oss << std::endl
							<< "When looking for:" << std::endl
							<< invok << std::endl
							<< "Present:" << std::endl;
						for (unsigned int _i = 0; _i < multi_return_calls.size(); _i++)
							oss << multi_return_calls[_i] << std::endl;
						throw_runtime_error("Cannot resolve multi-return call" + oss.str());
					}
					if (!multi_return_calls_written.count(mr_index))
					{
						this->write_code_multi_ret_call(os, "    ", flux, mr_index);
						multi_return_calls_written.insert(mr_index);
					}
				}
			}
		}

		os << "    const double dx = shapeinfo->int_pt_weight[0];" << std::endl;
		os << "    const double dX = shapeinfo->int_pt_weight_Lagrangian;" << std::endl;
		os << "    const double dx_unity = shapeinfo->int_pt_weight_unity;" << std::endl;
		os << "    switch (index)" << std::endl;
		os << "    {" << std::endl;
		unsigned index = 0;
		for (auto &e : sexprs)
		{
			if (!integrate)
			{
				// Check whether there is a field with the same name accessible
				FiniteElementField * f=this->get_field_by_name(e.first);					
				if (f) throw_runtime_error("The name '" + e.first + "' cannot be used for a local expression on '"+this->get_full_domain_name()+"', because it is already used for a here accessible field defined on the domain '"+f->get_defined_on_domain_equivalent_field()->get_space()->get_code()->get_full_domain_name()+"'");
			}

			os << "      case " << index << " :  res" << (integrate ? "+" : "") << "= ";

			// flux.evalf().print(GiNaC::print_csrc_FEM(os,&csrc_opts));
			//        GiNaC::factor(GiNaC::normal(GiNaC::expand(GiNaC::expand(flux).evalf()))).print(GiNaC::print_csrc_FEM(os,&csrc_opts));
			print_simplest_form(e.second, os, csrc_opts);
			os << "; break; // " << e.first << " [ " << units[e.first] << " ]" << std::endl;
			index++;
		}
		os << "    }" << std::endl;
		if (integrate)
		{
			os << "  }" << std::endl;
		}
		os << "   return res;" << std::endl;
		os << "}" << std::endl;
	}

	// Emits the EvalTracerAdvection function that evaluates the registered tracer-advection velocity
	// field(s) (used to advect passive tracer particles through the flow field) at the first
	// integration point, blended over a time fraction `timefrac_tracer` for sub-stepping between two
	// stored time levels. Structurally similar to write_code_integral_or_local_expressions but
	// specialized for the tracer velocity's vector output (`result_velo`) and time-fraction blending.
	void FiniteElementCode::write_code_tracer_advection(std::ostream &os)
	{
		os << "static void EvalTracerAdvection(const JITElementInfo_t * eleminfo, const JITShapeInfo_t * shapeinfo, unsigned index, double timefrac_tracer, double * result_velo)" << std::endl;
		os << "{" << std::endl;
		GiNaC::ex gathered;
		unsigned cnt = 0;
		for (auto &e : tracer_advection_terms)
		{
			gathered += e.second * GiNaC::wild(cnt++); // Wild important to prevent that terms are cancelling out
		}

		os << "  const double * t=shapeinfo->t;" << std::endl
		   << "  const double * dt=shapeinfo->dt;" << std::endl
		   << std::endl;

		std::set<ShapeExpansion> all_shapeexps = get_all_shape_expansions_in(gathered);
		std::set<TestFunction> all_testfuncs = get_all_test_functions_in(gathered);
		if (!all_testfuncs.empty())
		{
			throw_runtime_error("Found test function in tracer advection terms");
		}
		std::set<FiniteElementField *> indices_required;
		for (auto &sp : all_shapeexps)
		{
			indices_required.insert(sp.field);
			max_dt_order = std::max(max_dt_order, sp.dt_order);
			// if (!dynamic_cast<D0FiniteElementSpace*>(sp.field->get_space()))
			//{
			this->mark_shapes_required("TracerAdvection", sp.field->get_space(), sp.basis);
			//}
		}

		mark_further_required_fields(gathered, "TracerAdvection");

		for (auto *f : indices_required)
		{
			os << "  const unsigned " << f->get_nodal_index_str(this) << " = " << f->index << ";" << std::endl;
		}

		os << "  //START: Precalculate time derivatives of the necessary data" << std::endl;
		for (auto *sp : allspaces)
		{
			sp->write_nodal_time_interpolation(this, os, "  ", all_shapeexps);
		}
		os << "  //END: Precalculate time derivatives of the necessary data" << std::endl
		   << std::endl;

		os << "  unsigned ipt=0;" << std::endl;

		std::set<ShapeExpansion> spatial_shape_exps = get_all_shape_expansions_in(gathered);
		os << "    //START: Interpolate all required fields" << std::endl;
		for (auto *sp : allspaces)
		{
			sp->write_spatial_interpolation(this, os, "    ", spatial_shape_exps, false, false);
		}
		os << "    //END: Interpolate all required fields" << std::endl
		   << std::endl;

		GiNaC::print_FEM_options csrc_opts;
		csrc_opts.for_code = this;

		os << "    const double dx = shapeinfo->int_pt_weight[0];" << std::endl; // TODO: Lagrangian part
		os << "    switch (index)" << std::endl;
		os << "    {" << std::endl;
		unsigned index = 0;
		for (auto &e : tracer_advection_terms)
		{
			os << "      case " << index << " :" << std::endl;
			GiNaC::ex flux = (0 + e.second).evalm();
			flux = flux.subs(GiNaC::lst{expressions::x, expressions::y, expressions::z}, {_x, _y, _z});
			if (!GiNaC::is_a<GiNaC::matrix>(flux))
			{
				std::ostringstream oss;
				oss << "Tracer advection flux for tracers '" << e.first << "' is not a vector, but ";
				print_simplest_form(flux, oss, csrc_opts);
				throw_runtime_error(oss.str());
			}
			for (unsigned int cd = 0; cd < flux.nops(); cd++)
			{
				if (!GiNaC::is_zero(flux[cd]))
				{
					os << "        result_velo[" << cd << "]= ";
					print_simplest_form(flux[cd], os, csrc_opts);
					os << ";" << std::endl;
				}
			}
			// flux.evalf().print(GiNaC::print_csrc_FEM(os,&csrc_opts));
			//        GiNaC::factor(GiNaC::normal(GiNaC::expand(GiNaC::expand(flux).evalf()))).print(GiNaC::print_csrc_FEM(os,&csrc_opts));
			//			print_simplest_form(flux,os,csrc_opts);
			os << "        break; // " << e.first << " [ " << tracer_advection_units[e.first] << " ]" << std::endl;
			index++;
		}
		os << "    }" << std::endl;

		// os <<"   return res;" << std::endl;
		os << "}" << std::endl;
	}

	// The following three thin wrappers instantiate write_code_integral_or_local_expressions() for
	// the three kinds of user-registered scalar expressions: pointwise "local" expressions,
	// pointwise "extremum" expressions (evaluated the same way as local ones, just semantically used
	// to track a max/min elsewhere), and element-integrated "integral" expressions.
	void FiniteElementCode::write_code_local_expressions(std::ostream &os)
	{
		this->write_code_integral_or_local_expressions(os, local_expressions, local_expression_units, "EvalLocalExpression", "LocalExprs", false);
	}

	void FiniteElementCode::write_code_extremum_expressions(std::ostream &os)
	{
		this->write_code_integral_or_local_expressions(os, extremum_expressions, extremum_expression_units, "EvalExtremumExpression", "ExtremumExprs", false);
	}

	void FiniteElementCode::write_code_integral_expressions(std::ostream &os)
	{
		this->write_code_integral_or_local_expressions(os, integral_expressions, integral_expression_units, "EvalIntegralExpression", "IntegralExprs", true);
		/*	os <<"static double EvalIntegralExpression(const JITElementInfo_t * eleminfo, const JITShapeInfo_t * shapeinfo, unsigned index)"<< std::endl;
			os <<"{" << std::endl;



			 GiNaC::ex gathered;
			 unsigned cnt=0;
			 for (auto & e : integral_expressions)
			 {
				 gathered+=e.second*GiNaC::wild(cnt++);
			 }

			 os << "  double * t=shapeinfo->t;" << std::endl <<"  double * dt=shapeinfo->dt;" << std::endl << std::endl;

			std::set<ShapeExpansion> all_shapeexps=get_all_shape_expansions_in(gathered);
			std::set<TestFunction> all_testfuncs=get_all_test_functions_in(gathered);
			if (!all_testfuncs.empty()) {throw_runtime_error("Found test function in a custom integral expression");}
			std::set<FiniteElementField*> indices_required;
			 for (auto & sp : all_shapeexps)
			 {
				indices_required.insert(sp.field);
				max_dt_order=std::max(max_dt_order,sp.dt_order);
				//if (!dynamic_cast<D0FiniteElementSpace*>(sp.field->get_space()))
				//{
					this->mark_shapes_required("IntegralExprs",sp.field->get_space(),sp.basis);
				//}
			 }

			mark_further_required_fields(gathered,"IntegralExprs");

			 for (auto * f : indices_required)
			 {
			  os << "  const unsigned " << f->get_nodal_index_str(this) <<" = " << f->index << ";" << std::endl;
			 }

			 os << "  //START: Precalculate time derivatives of the necessary data" << std::endl;
			 for (auto * sp : allspaces)
			 {
				sp->write_nodal_time_interpolation(this,os,"  ",all_shapeexps);
			 }
			 os << "  //END: Precalculate time derivatives of the necessary data" << std::endl << std::endl;



			 os << "  double res=0.0;" << std::endl;

			 os << "  for(unsigned ipt=0;ipt<shapeinfo->n_int_pt;ipt++)" << std::endl;
			 os << "  {" << std::endl;


				std::set<ShapeExpansion> spatial_shape_exps=get_all_shape_expansions_in(gathered);
			   os << "    //START: Interpolate all required fields" << std::endl;
				for (auto * sp : allspaces)
				{
					sp->write_spatial_interpolation(this,os,"    ",spatial_shape_exps,false);
				}
			   os << "    //END: Interpolate all required fields" << std::endl << std::endl;


			   GiNaC::print_FEM_options csrc_opts;
			   csrc_opts.for_code=this;

			 os << "    const double dx = shapeinfo->int_pt_weights;"  << std::endl; //TODO: Lagrangian part
			 os << "    switch (index)" << std::endl;
			 os << "    {" << std::endl;
			 unsigned index=0;
			  for (auto & e : integral_expressions)
			  {
				  os << "      case " << index << " :  res+= ";
				 GiNaC::ex flux=0+e.second;
				flux=flux.subs(GiNaC::lst{expressions::x,expressions::y,expressions::z},{_x,_y,_z});
				//flux.evalf().print(GiNaC::print_csrc_FEM(os,&csrc_opts));
		//        GiNaC::factor(GiNaC::normal(GiNaC::expand(GiNaC::expand(flux).evalf()))).print(GiNaC::print_csrc_FEM(os,&csrc_opts));
					print_simplest_form(flux,os,csrc_opts);
					os<< "; break; // " << e.first << " [ " << integral_expression_units[e.first] <<" ]" <<std::endl;
					index++;
				}
			 os << "    }" << std::endl;
			 os << "  }"<< std::endl;
			 os <<"   return res;" << std::endl;
			 os <<"}" << std::endl;
			 */
	}

	// Emits GetZ2Fluxes[ForEigen](...), evaluating every registered Z2-error-estimator flux
	// expression at the first integration point (used by oomph-lib's Z2 error estimator to drive
	// mesh adaptivity); `for_eigen` selects the separate flux list used for eigenproblem/azimuthal
	// error estimation instead of the regular one.
	void FiniteElementCode::write_code_get_z2_flux(std::ostream &os,bool for_eigen)
	{
		os << "static void GetZ2Fluxes"<<(for_eigen ? "ForEigen" : "")<<"(const JITElementInfo_t * eleminfo, const JITShapeInfo_t * shapeinfo, double * Z2Flux)" << std::endl;
		os << "{" << std::endl;
		os << std::endl;

		GiNaC::ex gathered;
		unsigned cnt = 0;
		auto & fluxes=(for_eigen ? Z2_fluxes_for_eigen : Z2_fluxes);
		for (unsigned int i = 0; i < fluxes.size(); i++)
		{
			gathered += fluxes[i] * GiNaC::wild(cnt++);
		}

		std::set<ShapeExpansion> all_shapeexps = get_all_shape_expansions_in(gathered);
		std::set<TestFunction> all_testfuncs = get_all_test_functions_in(gathered);
		if (!all_testfuncs.empty())
		{
			throw_runtime_error("Found test function in spatial error estimator");
		}
		std::set<FiniteElementField *> indices_required;
		for (auto &sp : all_shapeexps)
		{
			indices_required.insert(sp.field);
			max_dt_order = std::max(max_dt_order, sp.dt_order);
			// if (!dynamic_cast<D0FiniteElementSpace*>(sp.field->get_space()))
			//{
			this->mark_shapes_required("Z2Fluxes", sp.field->get_space(), sp.basis);
			//}
		}

		mark_further_required_fields(gathered, "Z2Fluxes");

		for (auto *f : indices_required)
		{
			os << "  const unsigned " << f->get_nodal_index_str(this) << " = " << f->index << ";" << std::endl;
		}

		os << "  //START: Precalculate time derivatives of the necessary data" << std::endl;
		for (auto *sp : allspaces)
		{
			sp->write_nodal_time_interpolation(this, os, "  ", all_shapeexps);
		}
		os << "  //END: Precalculate time derivatives of the necessary data" << std::endl
		   << std::endl;

		std::set<ShapeExpansion> spatial_shape_exps = get_all_shape_expansions_in(gathered);
		os << "    //START: Interpolate all required fields" << std::endl;
		for (auto *sp : allspaces)
		{
			sp->write_spatial_interpolation(this, os, "    ", spatial_shape_exps, false, false);
		}
		os << "    //END: Interpolate all required fields" << std::endl
		   << std::endl;

		GiNaC::print_FEM_options csrc_opts;
		csrc_opts.for_code = this;

		for (unsigned int i = 0; i < fluxes.size(); i++)
		{
			os << "  Z2Flux[" << i << "] = ";
			GiNaC::ex flux = 0 + fluxes[i];
			RemoveSubexpressionsByIndentity sub_to_id(this);
			flux = sub_to_id(flux);
			flux = flux.subs(GiNaC::lst{expressions::x, expressions::y, expressions::z}, {_x, _y, _z});
			// flux.evalf().print(GiNaC::print_csrc_FEM(os,&csrc_opts));
			//        GiNaC::factor(GiNaC::normal(GiNaC::expand(GiNaC::expand(flux).evalf()))).print(GiNaC::print_csrc_FEM(os,&csrc_opts));
			print_simplest_form(flux, os, csrc_opts);
			os << ";" << std::endl;
		}
		os << "}" << std::endl;
	}

	// Detects fields/test functions referenced in this code's residuals/integral/local expressions
	// that live on a domain not directly reachable from here (i.e. not this code itself, its bulk
	// element(s), or the opposite-interface element and its bulk) - the only way such a reference can
	// be legitimate is if the owning domain is a coupled ODE domain (external, 0-dimensional
	// "ED0"-space data, e.g. a lumped-parameter ODE coupled to this PDE). For every such field/test
	// function found, registers a proxy external-ODE field ("__EXT_ODE_<n>" on the ED0 space) linked
	// to the real field via _register_external_ode_linkage, and finally rewrites every residual/
	// integral/local expression (via RemapFieldsInExpression) to reference the proxy field instead of
	// the original one - since the original one is not something this element's generated code can
	// otherwise access. Fields/test functions are processed in a name-sorted (not pointer/insertion)
	// order so that the assigned proxy indices are deterministic and reproducible.
	void FiniteElementCode::check_for_external_ode_dependencies()
	{
		std::map<FiniteElementField *, FiniteElementField *> remapping;
		std::string ode_ext_name_trunk = "__EXT_ODE_";
		unsigned cnt = 0;
		int oldstage = stage;
		stage = 0; // To register further fields

		int walking_index = -1;
		for (unsigned int i = 0; i < myfields.size(); i++)
		{
			if (!dynamic_cast<PositionFiniteElementSpace *>(myfields[i]->get_space()))
			{
				walking_index = std::max(myfields[i]->index, walking_index);
			}
		}
		walking_index++;

		std::set<ShapeExpansion> shapeexps;
		for (unsigned int i = 0; i < residual.size(); i++)
		{
			std::set<ShapeExpansion> lshapeexp = this->get_all_shape_expansions_in(residual[i]);
			shapeexps.insert(lshapeexp.begin(), lshapeexp.end());
		}
		for (auto &ie : integral_expressions)
		{
			std::set<ShapeExpansion> lshapeexp = this->get_all_shape_expansions_in(ie.second);
			shapeexps.insert(lshapeexp.begin(), lshapeexp.end());
		}
		for (auto &le : local_expressions)
		{
			std::set<ShapeExpansion> lshapeexp = this->get_all_shape_expansions_in(le.second);
			shapeexps.insert(lshapeexp.begin(), lshapeexp.end());
		}

		std::vector<ShapeExpansion> ordered_shapeexps;
		for (auto &sp : shapeexps)
		{
			ordered_shapeexps.push_back(sp);
		}
		auto shape_order = [](ShapeExpansion &a, ShapeExpansion &b)
		{
		   std::string sa=a.field->get_space()->get_code()->get_domain_name()+"/"+a.field->get_name();
		   std::string sb=b.field->get_space()->get_code()->get_domain_name()+"/"+b.field->get_name();		   
		   return sa<sb; };
		std::sort(ordered_shapeexps.begin(), ordered_shapeexps.end(), shape_order);

		for (auto &sp : ordered_shapeexps)
		{
			// 		  std::cout << "CHECKING  " << GiNaC::GiNaCShapeExpansion(sp) << "  " << sp.field->get_space()->get_code() << " vs " << this << std::endl;
			auto *code_to_check = sp.field->get_space()->get_code();
			if (code_to_check != this && code_to_check != bulk_code && (!bulk_code || code_to_check != bulk_code->bulk_code) && code_to_check != opposite_interface_code && (opposite_interface_code ? opposite_interface_code->bulk_code != code_to_check : true))
			{
				if (!code_to_check->_is_ode_element())
				{
					std::ostringstream oss;
					oss << "Found a shape expansion " << GiNaC::GiNaCShapeExpansion(sp) << " which is neither defined on the current domain, nor on the parent or the domain opposite of the interface. It is also not and ODE. This does not work right now!";
					throw_runtime_error(oss.str());
				}
				if (!remapping.count(sp.field))
				{
					std::string myname = ode_ext_name_trunk + std::to_string(cnt);
					FiniteElementField *ext = this->register_field(myname, "ED0");
					this->_register_external_ode_linkage(myname, code_to_check, sp.field->get_name());
					ext->index = walking_index++;
					cnt++;
					remapping[sp.field] = ext;
				}
			}
		}

		std::set<TestFunction> testfuncs;
		for (unsigned int i = 0; i < residual.size(); i++)
		{
			std::set<TestFunction> ltestfuncs = this->get_all_test_functions_in(residual[i]);
			testfuncs.insert(ltestfuncs.begin(), ltestfuncs.end());
		}

		std::vector<TestFunction> ordered_testfuncs;
		for (auto &sp : testfuncs)
			ordered_testfuncs.push_back(sp);
		auto test_order = [](TestFunction &a, TestFunction &b)
		{
		   std::string sa=a.field->get_space()->get_code()->get_domain_name()+"/"+a.field->get_name();
		   std::string sb=b.field->get_space()->get_code()->get_domain_name()+"/"+b.field->get_name();		   
		   return sa<sb; };
		std::sort(ordered_testfuncs.begin(), ordered_testfuncs.end(), test_order);

		for (auto &tg : ordered_testfuncs)
		{
			auto *code_to_check = tg.field->get_space()->get_code();
			if (code_to_check != this && code_to_check != bulk_code && (!bulk_code || code_to_check != bulk_code->bulk_code) && code_to_check != opposite_interface_code && (opposite_interface_code ? opposite_interface_code->bulk_code != code_to_check : true))
			{
				if (!code_to_check->_is_ode_element())
				{
					std::ostringstream oss;
					oss << "Found a test function " << GiNaC::GiNaCTestFunction(tg) << " which is neither defined on the current domain, nor on the parent or the domain opposite of the interface. It is also not and ODE. This does not work right now!";
					throw_runtime_error(oss.str());
				}
				if (!remapping.count(tg.field))
				{
					std::string myname = ode_ext_name_trunk + std::to_string(cnt);
					FiniteElementField *ext = this->register_field(myname, "ED0");
					this->_register_external_ode_linkage(myname, code_to_check, tg.field->get_name());
					ext->index = walking_index++;
					cnt++;
					remapping[tg.field] = ext;
				}
			}
		}

		if (!remapping.empty())
		{
			RemapFieldsInExpression remap(remapping);
			for (unsigned int i = 0; i < residual.size(); i++)
			{
				residual[i] = remap(residual[i]);
			}
			for (auto &ie : integral_expressions)
			{
				integral_expressions[ie.first] = remap(ie.second);
			}
			for (auto &le : local_expressions)
			{
				local_expressions[le.first] = remap(le.second);
			}
		}

		stage = oldstage;
	}

	// Rebuilds `allspaces`: the flat list of every FiniteElementSpace reachable from this code,
	// i.e. its own spaces plus (if present) those of the bulk element, the bulk's bulk element, the
	// opposite-interface element, and that interface's bulk element. This is the set of spaces that
	// interpolation/Jacobian/Hessian code generation iterates over.
	void FiniteElementCode::find_all_accessible_spaces()
	{
		allspaces.clear();
		for (unsigned int i = 0; i < spaces.size(); i++)
			allspaces.push_back(spaces[i]);
		if (bulk_code)
		{
			for (unsigned int i = 0; i < bulk_code->spaces.size(); i++)
				allspaces.push_back(bulk_code->spaces[i]);
			if (bulk_code->bulk_code)
			{
				for (unsigned int i = 0; i < bulk_code->bulk_code->spaces.size(); i++)
					allspaces.push_back(bulk_code->bulk_code->spaces[i]);
			}
		}
		if (opposite_interface_code)
		{
			for (unsigned int i = 0; i < opposite_interface_code->spaces.size(); i++)
				allspaces.push_back(opposite_interface_code->spaces[i]);
			if (opposite_interface_code->bulk_code)
			{
				for (unsigned int i = 0; i < opposite_interface_code->bulk_code->spaces.size(); i++)
					allspaces.push_back(opposite_interface_code->bulk_code->spaces[i]);
			}
		}
	}
	
	

	// Top-level entry point that generates the *entire* C source file for this element: resolves
	// external-ODE dependencies, then for every registered residual, emits the analytical Residual/
	// Jacobian/Mass-matrix function (write_generic_RJM); if the timestepping scheme requires it
	// (detected via MakeResidualSteady), also a dedicated "steady" variant; if Hessian generation is
	// enabled, the Hessian-vector-product function (write_generic_Hessian); and, for every global
	// parameter the steady residual actually depends on, a dedicated parameter-derivative RJM
	// function (needed for parameter continuation/bifurcation tracking). Afterwards emits
	// initial-condition, Dirichlet-condition, geometric-Jacobian, Z2-flux, integral/local/extremum-
	// expression, tracer-advection, and finally the master write_code_info() dispatch-table function
	// that ties all the above together into the runtime-loadable JIT function table. `stage` is
	// advanced to 2 (fully finalized) once done.
	void FiniteElementCode::write_code(std::ostream &os)
	{
		__current_code = this;
		this->archive.clear();
		CustomMathExpressionBase::code_map.clear();
		CustomMultiReturnExpressionBase::code_map.clear();
		find_all_accessible_spaces();
		// Investigate the residual for external ODE variables
		check_for_external_ode_dependencies();

		write_code_header(os);
		os << std::endl;
		local_parameter_has_deriv.resize(residual.size());
		extra_steady_routine.resize(residual.size(), false);
		has_hessian_contribution.resize(residual.size(), false);
		for (auto &entry : multi_return_ccodes)
		{
			unsigned index = entry.second.first;
			std::string body = entry.second.second;
			os << "#define CURRENT_MULTIRET_FUNCTION multi_ret_ccode_" << index << std::endl;
			os << "static void multi_ret_ccode_" << index << "(int flag, double *arg_list, double *result_list, double *derivative_matrix,int nargs,int nret)" << std::endl
			   << "{" << std::endl;
			os << body << std::endl;
			os << "}" << std::endl;
			os << "#undef CURRENT_MULTIRET_FUNCTION" << std::endl
			   << std::endl;
		}
		for (unsigned int resind = 0; resind < residual.size(); resind++)
		{
			residual_index = resind;
			__in_pitchfork_symmetry_constraint = (residual_names[resind] == "_simple_mass_matrix_of_defined_fields");
			if (!residual[resind].is_zero())
			{
				write_generic_RJM(os, "ResidualAndJacobian" + std::to_string(resind), residual[resind], true); // Hanging unsteady routine
				os << std::endl;

				// Check if we need a dedicated steady routine. This happens, if you use e.g. MPT or TPZ time integration, which use history values
				MakeResidualSteady make_steady(this);
				GiNaC::ex steady_residual = make_steady(residual[resind]);
				extra_steady_routine[resind] = make_steady.require_extra_steady_routine();

				if (extra_steady_routine[resind])
				{
					os << std::endl;
					write_generic_RJM(os, "ResidualAndJacobianSteady" + std::to_string(resind), steady_residual, true); // Hanging unsteady routine
					os << std::endl;
				}

				if (generate_hessian)
				{
					has_constant_mass_matrix_for_sure[resind]=true; // Might change during writing the Hessian
					has_hessian_contribution[resind] = write_generic_Hessian(os, "HessianVectorProduct" + std::to_string(resind), residual[resind], true);
					os << std::endl;
				}

				GiNaC::potential_real_symbol gp_dummy("_global_param_");
				for (unsigned int i = 0; i < local_parameter_symbols.size(); i++) // Only parameters in Residuals releveant (e.g. not in integral expressions)
				{
					GiNaC::ex p = local_parameter_symbols[i];
					GiNaC::ex dres_dp = steady_residual.subs(p == gp_dummy).diff(gp_dummy); // Take the steady residual only here
					if (!dres_dp.is_zero())													// Need to write the dresidual_dparameter function
					{
						dres_dp = dres_dp.subs(gp_dummy == p);
						os << std::endl;
						os << "//Derivative wrt. global parameter " << p << std::endl;
						std::ostringstream oss;
						oss << "dResidual" + std::to_string(resind) + "dParameter_" << i;
						write_generic_RJM(os, oss.str(), dres_dp, true);
						local_parameter_has_deriv[resind].push_back(true);
					}
					else
						local_parameter_has_deriv[resind].push_back(false);
				}
			}
			else
			{
				extra_steady_routine[resind] = false;
			}
			__in_pitchfork_symmetry_constraint = false;
		}

		residual_index = 0;

		for (unsigned int i = 0; i < IC_names.size(); i++)
		{
			write_code_initial_condition(os, i, IC_names[i]);
			os << std::endl;
		}
		write_code_Dirichlet_condition(os);
		os << std::endl;
		write_code_geometric_jacobian(os);
		os << std::endl;
		if (Z2_fluxes.size())
		{
			write_code_get_z2_flux(os,false);
			os << std::endl;
		}
		if (Z2_fluxes_for_eigen.size())
		{
			write_code_get_z2_flux(os,true);
			os << std::endl;
		}
		os << std::endl;
		if (integral_expressions.size())
		{
			write_code_integral_expressions(os);
			os << std::endl;
		}
		if (local_expressions.size())
		{
			write_code_local_expressions(os);
			os << std::endl;
		}
		if (extremum_expressions.size())
		{
			write_code_extremum_expressions(os);
			os << std::endl;
		}
		if (tracer_advection_terms.size())
		{
			write_code_tracer_advection(os);
			os << std::endl;
		}
		os << std::endl;
		write_code_info(os);
		stage = 2;
		__current_code = NULL;
		//			std::set<ShapeExpansion*> allshapes=FiniteElementCode::get_all_shape_expansions_in(GiNaC::ex inp)
	}

	// Returns 0 if the space is defined on this element, -1 for bulk element, -2 for other side of interface, >0 for external elements [-1]
	//  -3 for opposite bulk
	//  -4 for bulk->bulk
	int FiniteElementCode::classify_space_type(const FiniteElementSpace *s)
	{
		for (unsigned int i = 0; i < spaces.size(); i++)
			if (s == spaces[i])
				return 0;
		if (bulk_code)
			for (unsigned int i = 0; i < bulk_code->spaces.size(); i++)
				if (s == bulk_code->spaces[i])
					return -1;
		if (opposite_interface_code)
		{
			for (unsigned int i = 0; i < opposite_interface_code->spaces.size(); i++)
				if (s == opposite_interface_code->spaces[i])
					return -2;
			if (opposite_interface_code->bulk_code)
			{
				for (unsigned int i = 0; i < opposite_interface_code->bulk_code->spaces.size(); i++)
					if (s == opposite_interface_code->bulk_code->spaces[i])
						return -3;
			}
		}
		if (bulk_code && bulk_code->bulk_code)
			for (unsigned int i = 0; i < bulk_code->bulk_code->spaces.size(); i++)
				if (s == bulk_code->bulk_code->spaces[i])
					return -4;
		/*
		for (unsigned ie=0;ie<required_odes.size();ie++)
		{
		   for (unsigned int i=0;i<required_odes[ie]->spaces.size();i++) if (s==required_odes[ie]->spaces[i]) return ie+1;
		}
		//Not found yet, check if it is an ODE, then we could add it
		if (s->get_code()->_is_ode_element())
		{
		  unsigned index=required_odes.size();
		  required_odes.push_back(s->get_code());
		  return index+1;
		}
  */
		throw_runtime_error("Error in classify_space_type");
		return -666;
	}

	// The following get_owner_prefix/get_shape_info_str/get_nodal_data_string/get_elem_info_str all
	// use classify_space_type() to translate "which domain does this space belong to" into the
	// matching C variable-name prefix / pointer-chase expression used in the generated code: the
	// current element's own data (this_/shapeinfo/eleminfo/nodal_data), its bulk element's data
	// (blk_/shapeinfo->bulk_shapeinfo/...), the opposite interface element's data (opp_/...), or
	// combinations thereof (oppblk_, blkblk_) for bulk-of-bulk / bulk-of-opposite-interface access.
	std::string FiniteElementCode::get_owner_prefix(const FiniteElementSpace *sp)
	{
		int typ = classify_space_type(sp);
		if (typ == 0)
			return "this_";
		else if (typ == -1)
			return "blk_";
		else if (typ == -2)
			return "opp_";
		else if (typ == -3)
			return "oppblk_";
		else if (typ == -4)
			return "blkblk_";
		/*     	for (unsigned ie=0;ie<required_odes.size();ie++)
			  {
				 for (unsigned int i=0;i<required_odes[ie]->spaces.size();i++) if (sp==required_odes[ie]->spaces[i]) return "ode"+std::to_string(ie)+"_";
			  }     	*/
		throw_runtime_error("TODO: add external spaces");
	}

	std::string FiniteElementCode::get_shape_info_str(const FiniteElementSpace *sp)
	{
		int typ = classify_space_type(sp);
		if (typ == 0)
			return "shapeinfo";
		else if (typ == -1)
			return "shapeinfo->bulk_shapeinfo";
		else if (typ == -2)
			return "shapeinfo->opposite_shapeinfo";
		else if (typ == -3)
			return "shapeinfo->opposite_shapeinfo->bulk_shapeinfo";
		else if (typ == -4)
			return "shapeinfo->bulk_shapeinfo->bulk_shapeinfo";
		//      else if (typ>0) return "shapeinfo"; //Use the fact that D0 is the same in all kinds
		throw_runtime_error("TODO: add bulk and external spaces");
	}

	std::string FiniteElementCode::get_nodal_data_string(const FiniteElementSpace *sp)
	{
		int typ = classify_space_type(sp);
		if (typ == 0)
		{
			if (dynamic_cast<const PositionFiniteElementSpace *>(sp))
				return "nodal_coords";
			else
				return "nodal_data";
		}
		else if (typ == -1)
		{
			if (sp->get_code() == this->bulk_code)
			{
				if (dynamic_cast<const PositionFiniteElementSpace *>(sp))
					return "nodal_coords";
				else
					return "nodal_data";
			}
			else
				throw_runtime_error("TODO: add external spaces");
		}
		else if (typ == -2)
		{
			if (sp->get_code() == this->opposite_interface_code)
			{
				if (dynamic_cast<const PositionFiniteElementSpace *>(sp))
					return "nodal_coords";
				else
					return "nodal_data";
			}
			else
				throw_runtime_error("TODO: add external spaces");
		}
		else if (typ == -3)
		{
			if (sp->get_code() == this->opposite_interface_code->bulk_code)
			{
				if (dynamic_cast<const PositionFiniteElementSpace *>(sp))
					return "nodal_coords";
				else
					return "nodal_data";
			}
			else
				throw_runtime_error("TODO: add  external spaces");
		}
		else if (typ == -4)
		{
			if (sp->get_code() == this->bulk_code->bulk_code)
			{
				if (dynamic_cast<const PositionFiniteElementSpace *>(sp))
					return "nodal_coords";
				else
					return "nodal_data";
			}
			else
				throw_runtime_error("TODO: add  external spaces");
		}
		/*     	else if (typ>0)
				{
				 return "external_data"; //TODO
				}*/
		else
			throw_runtime_error("TODO: add external spaces");
	}

	std::string FiniteElementCode::get_elem_info_str(const FiniteElementSpace *sp)
	{
		int typ = classify_space_type(sp);
		if (typ == 0)
			return "eleminfo";
		else if (typ == -1)
			return "eleminfo->bulk_eleminfo";
		else if (typ == -2)
			return "eleminfo->opposite_eleminfo";
		else if (typ == -3)
			return "eleminfo->opposite_eleminfo->bulk_eleminfo";
		else if (typ == -4)
			return "eleminfo->bulk_eleminfo->bulk_eleminfo";
		//     	else if (typ>0) return "eleminfo->external_data";
		else
			throw_runtime_error("TODO: add bulk and external spaces");
	}

	// The following get_dx/get_element_size_symbol/get_nodal_delta/get_normal_component factory
	// methods wrap this code's pre-built SpatialIntegralSymbol/ElementSizeSymbol/NodalDeltaSymbol/
	// NormalSymbol members as GiNaC expressions, so user-facing weak-form code can multiply residual
	// terms by "dx"/"dX"/element size/nodal delta/normal-vector components symbolically.
	GiNaC::ex FiniteElementCode::get_dx(bool lagrangian, bool unity_only)
	{
		if (unity_only) return 0+GiNaC::GiNaCSpatialIntegralSymbol(dx_unity);
		if (lagrangian)
		{
			return 0 + GiNaC::GiNaCSpatialIntegralSymbol(dX);
		}
		else
		{
			return 0 + GiNaC::GiNaCSpatialIntegralSymbol(dx);
		}
	}

	GiNaC::ex FiniteElementCode::get_element_size_symbol(bool lagrangian, bool with_coordsys)
	{
		if (lagrangian)
		{
			return 0 + GiNaC::GiNaCElementSizeSymbol(!with_coordsys ? elemsize_Lagrangian_Cart : elemsize_Lagrangian);
		}
		else
		{
			return 0 + GiNaC::GiNaCElementSizeSymbol(!with_coordsys ? elemsize_Eulerian_Cart : elemsize_Eulerian);
		}
	}

	GiNaC::ex FiniteElementCode::get_nodal_delta()
	{
		return 0 + GiNaC::GiNaCNodalDeltaSymbol(nodal_delta);
	}

	GiNaC::ex FiniteElementCode::get_normal_component(unsigned i)
	{
		return 0 + GiNaC::GiNaCNormalSymbol(NormalSymbol(this, i));
	}


	// Emits the ElementalInitialConditions<ic_index> function that evaluates the user-supplied
	// initial-condition expression for the initial-condition set named `ic_name`, for whichever field
	// `field_index` the runtime asks for. Substitutes the raw position/Lagrangian-coordinate C
	// arguments (_x[i]/_xlagr[i]) for the corresponding coordinate/lagrangian ShapeExpansion symbols
	// (since initial conditions are evaluated directly from given spatial coordinates, not via the
	// usual shape-function interpolation) before printing each field's expression in an if/else-if
	// chain keyed by field_index.
	void FiniteElementCode::write_code_initial_condition(std::ostream &os, unsigned int ic_index, std::string ic_name)
	{
		os << "// INITIAL CONDITION " << ic_name << std::endl;
		os << "static double ElementalInitialConditions" << ic_index << "(const JITElementInfo_t * eleminfo, int field_index,double *_x, double *_xlagr,double *_normal,double t,int flag,double default_val)" << std::endl;
		os << "{" << std::endl;
		//		os << "  const unsigned " << std::endl;
		GiNaC::lst sublist;
		std::vector<std::string> dir{"x", "y", "z"};
		for (unsigned int i = 0; i < this->nodal_dim; i++)
		{
			sublist.append(this->get_field_by_name("coordinate_" + dir[i])->get_shape_expansion() == GiNaC::potential_real_symbol("_x[" + std::to_string(i) + "]"));
			sublist.append(this->get_field_by_name("mesh_" + dir[i])->get_shape_expansion() == GiNaC::potential_real_symbol("_x[" + std::to_string(i) + "]"));
		}

		for (unsigned int i = 0; i < this->lagr_dim; i++)
		{
			sublist.append(this->get_field_by_name("lagrangian_" + dir[i])->get_shape_expansion() == GiNaC::potential_real_symbol("_xlagr[" + std::to_string(i) + "]"));
		}

		bool no_else = true;

		GiNaC::print_FEM_options csrc_opts;
		csrc_opts.for_code = this;
		RemoveSubexpressionsByIndentity sub_to_id(this);
		for (auto *f : myfields)
		{
			if (f->initial_condition.count(ic_name))
			{
				GiNaC::ex ic = f->initial_condition[ic_name];
				// Replace all stuff in the initial condition
				multi_return_calls.clear();
				ic = sub_to_id(ic.subs(sublist));
				int myindex = f->index;
				std::string nam = f->get_name();

				if (nam == "mesh_x")
					nam = "coordinate_x";
				else if (nam == "mesh_y")
					nam = "coordinate_y";
				else if (nam == "mesh_z")
					nam = "coordinate_z";

				if (nam == "coordinate_x")
					myindex = -1;
				else if (nam == "coordinate_y")
					myindex = -2;
				else if (nam == "coordinate_z")
					myindex = -3;

				os << "  " << (no_else ? "" : "else ") << "if (field_index==" << myindex << ") // IC of field " << nam << std::endl;
				os << "  {" << std::endl;
				std::set<int> multi_return_calls_written;
				for (GiNaC::const_preorder_iterator it = ic.preorder_begin(); it != ic.preorder_end(); ++it)
				{
					if (GiNaC::is_a<GiNaC::GiNaCMultiRetCallback>(*it))
					{
						GiNaC::ex invok = GiNaC::ex_to<GiNaC::GiNaCMultiRetCallback>(*it).get_struct().invok;
						int mr_index = this->resolve_multi_return_call(invok);
						if (mr_index < 0)
						{
							std::ostringstream oss;
							oss << std::endl
								<< "When looking for:" << std::endl
								<< invok << std::endl
								<< "Present:" << std::endl;
							for (unsigned int _i = 0; _i < multi_return_calls.size(); _i++)
								oss << multi_return_calls[_i] << std::endl;
							throw_runtime_error("Cannot resolve multi-return call" + oss.str());
						}
						if (!multi_return_calls_written.count(mr_index))
						{
							this->write_code_multi_ret_call(os, "    ", ic, mr_index);
							multi_return_calls_written.insert(mr_index);
						}
					}
				}

				os << "    if (!flag) return ";
				// 			   ic.evalf().print(GiNaC::print_csrc_double(os)); os << "; " << std::endl;
				// ic.evalf().print(GiNaC::print_csrc_FEM(os,&csrc_opts)); os << "; " << std::endl;
				print_simplest_form(ic, os, csrc_opts);
				os << "; " << std::endl;

				GiNaC::ex dtcond = ic.diff(pyoomph::expressions::t);
				os << "    if (flag==1) return ";
				//				dtcond.evalf().print(GiNaC::print_csrc_FEM(os,&csrc_opts)); os << "; " << std::endl;
				print_simplest_form(dtcond, os, csrc_opts);
				os << "; " << std::endl;
				// dtcond.evalf().print(GiNaC::print_csrc_double(os)); os << "; " << std::endl;
				GiNaC::ex dt2cond = dtcond.diff(pyoomph::expressions::t);
				os << "    if (flag==2) return ";
				//				dt2cond.evalf().print(GiNaC::print_csrc_FEM(os,&csrc_opts)); os << "; " << std::endl;
				print_simplest_form(dt2cond, os, csrc_opts);
				os << "; " << std::endl;
				// dt2cond.evalf().print(GiNaC::print_csrc_double(os)); os << "; " << std::endl;
				os << "  }" << std::endl;
				no_else = false;
			}
		}

		os << "  return default_val;" << std::endl;
		os << "}" << std::endl;
	}

	// Emits the ElementalDirichletConditions function, structurally analogous to
	// write_code_initial_condition() above but for user-set Dirichlet boundary values: for each field
	// with a Dirichlet condition set, either returns the default (pin-only, value left to the caller)
	// or evaluates and returns the prescribed value expression at the given position.
	void FiniteElementCode::write_code_Dirichlet_condition(std::ostream &os)
	{

		os << "static double ElementalDirichletConditions(const JITElementInfo_t * eleminfo, int field_index,double *_x, double *_xlagr,double *_normal,double t,double default_val)" << std::endl;
		os << "{" << std::endl;
		os << "  const unsigned flag=0;" << std::endl;
		GiNaC::lst sublist;
		std::vector<std::string> dir{"x", "y", "z"};
		for (unsigned int i = 0; i < this->nodal_dim; i++)
		{
			sublist.append(this->get_field_by_name("coordinate_" + dir[i])->get_shape_expansion() == GiNaC::potential_real_symbol("_x[" + std::to_string(i) + "]"));
			sublist.append(this->get_field_by_name("mesh_" + dir[i])->get_shape_expansion() == GiNaC::potential_real_symbol("_x[" + std::to_string(i) + "]"));
		}

		for (unsigned int i = 0; i < this->lagr_dim; i++)
		{
			sublist.append(this->get_field_by_name("lagrangian_" + dir[i])->get_shape_expansion() == GiNaC::potential_real_symbol("_xlagr[" + std::to_string(i) + "]"));
		}

		bool no_else = true;

		GiNaC::print_FEM_options csrc_opts;
		csrc_opts.for_code = this;
		RemoveSubexpressionsByIndentity sub_to_id(this);
		for (auto *f : myfields)
		{
			if (f->Dirichlet_condition_set)
			{
				GiNaC::ex dc = f->Dirichlet_condition;
				// Replace all stuff in the initial condition
				multi_return_calls.clear();
				dc = sub_to_id(dc.subs(sublist));
				int myindex = f->index;
				std::string nam = f->get_name();
				if (nam == "mesh_x")
					nam = "coordinate_x";
				else if (nam == "mesh_y")
					nam = "coordinate_y";
				else if (nam == "mesh_z")
					nam = "coordinate_z";
				if (nam == "coordinate_x")
					myindex = -1;
				else if (nam == "coordinate_y")
					myindex = -2;
				else if (nam == "coordinate_z")
					myindex = -3;
				os << "  " << (no_else ? "" : "else ") << "if (field_index==" << myindex << ") // DC of field " << nam << std::endl;
				os << "  {" << std::endl;
				if (f->Dirichlet_condition_pin_only)
				{
					os << "    return default_val;" << std::endl;
				}
				else
				{

					std::set<int> multi_return_calls_written;
					for (GiNaC::const_preorder_iterator it = dc.preorder_begin(); it != dc.preorder_end(); ++it)
					{
						if (GiNaC::is_a<GiNaC::GiNaCMultiRetCallback>(*it))
						{
							GiNaC::ex invok = GiNaC::ex_to<GiNaC::GiNaCMultiRetCallback>(*it).get_struct().invok;
							int mr_index = this->resolve_multi_return_call(invok);
							if (mr_index < 0)
							{
								std::ostringstream oss;
								oss << std::endl
									<< "When looking for:" << std::endl
									<< invok << std::endl
									<< "Present:" << std::endl;
								for (unsigned int _i = 0; _i < multi_return_calls.size(); _i++)
									oss << multi_return_calls[_i] << std::endl;
								throw_runtime_error("Cannot resolve multi-return call" + oss.str());
							}
							if (!multi_return_calls_written.count(mr_index))
							{
								this->write_code_multi_ret_call(os, "    ", dc, mr_index, &multi_return_calls_written, &invok);
								multi_return_calls_written.insert(mr_index);
							}
						}
					}

					os << "    return ";
					print_simplest_form(dc, os, csrc_opts);
					os << "; " << std::endl;
				}
				os << "  }" << std::endl;
				no_else = false;
			}
		}

		os << "  return default_val;" << std::endl;
		os << "}" << std::endl;
	}

	FiniteElementField *FiniteElementCode::get_field_by_name(std::string name)
	{
		for (unsigned int i = 0; i < myfields.size(); i++)
			if (myfields[i]->get_name() == name)
				return myfields[i];
		return NULL;
	}

	// Resolves the placeholder call `func` (a field(...)/nondimfield(...)/testfunction(...)/
	// eval_in_domain(...)/... expression) to the FiniteElementCode that actually owns it, based on
	// the domain tags attached to its GiNaCPlaceHolderResolveInfo argument. If given, extracts the
	// referenced field's plain name into `*fname`, and pulls the recognized "flag:*" tags
	// (no_jacobian/no_hessian/only_base_mode/only_perturbation_mode) out into `*taginfo`, leaving
	// only the remaining ("domain:*") tags to resolve. If the resolve-info already carries a
	// concrete code pointer, it is used directly (after checking it is actually reachable from here).
	// Otherwise, walks the domain tags to interpret shorthand relative-domain syntax: "." (self),
	// ".."/"..." (bulk / bulk-of-bulk), "|."/"|.." (opposite interface / its bulk), and the internal-
	// facet-only "+"/"-"/"+|"/"|-" tags used to access the two sides of a DG/HDG internal-facet
	// element; unrecognized domain names fall back to the element-type-specific
	// _resolve_based_on_domain_name() hook. If no domain tag is present at all, defaults to `this`.
	FiniteElementCode *FiniteElementCode::resolve_corresponding_code(GiNaC::ex func, std::string *fname, FiniteElementFieldTagInfo *taginfo)
	{

		std::ostringstream os;
		bool eval_in_domain = is_ex_the_function(func, expressions::eval_in_domain);
		std::string funcname;
		if (!eval_in_domain)
		{
			os << func.op(0);
			funcname = os.str();
			if (fname)
				*fname = funcname;
		}

		GiNaC::GiNaCPlaceHolderResolveInfo resolve_info = GiNaC::ex_to<GiNaC::GiNaCPlaceHolderResolveInfo>(func.op(1));
		auto intags = resolve_info.get_struct().tags;
		auto tags = intags;
		if (taginfo)
		{
			tags.clear();
			for (auto &t : intags)
			{
				if (t == "flag:no_jacobian")
					taginfo->no_jacobian = true;
				else if (t == "flag:no_hessian")
					taginfo->no_hessian = true;
				else if (t == "flag:only_base_mode")
					taginfo->expansion_mode = -1;
				else if (t == "flag:only_perturbation_mode")
					taginfo->expansion_mode = -2;
				else
					tags.push_back(t);
			}
		}
		else
		{
			tags = intags;
		}

		if (resolve_info.get_struct().code)
		{
			if (resolve_info->code != this && resolve_info->code != this->bulk_code && (!this->bulk_code || resolve_info->code != this->bulk_code->bulk_code) && resolve_info->code != this->opposite_interface_code && (!this->opposite_interface_code || resolve_info->code != this->opposite_interface_code->bulk_code))
			{
				if (eval_in_domain)
				{
					os << func.op(0);
					throw_runtime_error("The desired domain is not within the scope for the expression: " + os.str());
				}
				else
				{
					throw_runtime_error("Field " + funcname + " is not within the scope of the current equation domain");
				}
			}
			return resolve_info->code;
		}
		if (tags.empty())
		{
			// if (eval_in_domain) throw_runtime_error("Cannot evaluate in a domain which is not specified");
			return this;
		}

		for (auto &t : tags)
		{
			if (t.find("domain:") == 0)
			{
				std::string domname = t.substr(7);
				if (domname == ".")
					return this;
				else if (domname == "..")
				{
					if (!this->bulk_code)
						throw_runtime_error("Cannot access the parent domain by '..' when no parent domain is present");
					return this->bulk_code;
				}
				else if (domname == "...")
				{
					if (!this->bulk_code || (!this->bulk_code->bulk_code))
						throw_runtime_error("Cannot access the parent->parent domain by '...' when no parent->parent domain is present");
					return this->bulk_code->bulk_code;
				}
				else if (domname == "|.")
				{
					if (!this->opposite_interface_code)
						throw_runtime_error("Cannot access the opposing interface domain by '|.' when no opposing interface is present");
					return this->opposite_interface_code;
				}
				else if (domname == "|..")
				{
					if (!this->opposite_interface_code || (!this->opposite_interface_code->bulk_code))
						throw_runtime_error("Cannot access the opposing parent domain by '|..' when no opposing parent domain is present");
					return this->opposite_interface_code->bulk_code;
				}
				else if (domname == "+")
				{
					if (this->get_domain_name() != "_internal_facets_" || !this->bulk_code)
						throw_runtime_error("Accessing the bulk domain of the + side element only works on the '_internal_facets_' subdomain, not on the domain " + this->get_domain_name());
					return this->bulk_code;
				}
				else if (domname == "-")
				{
					if (this->get_domain_name() != "_internal_facets_" || !this->opposite_interface_code || !this->opposite_interface_code->bulk_code)
						throw_runtime_error("Accessing the bulk domain of the - side element only works on the '_internal_facets_' subdomain, not on the domain " + this->get_domain_name());
					return this->opposite_interface_code->bulk_code;
				}
				else if (domname == "+|")
				{
					if (this->get_domain_name() != "_internal_facets_")
						throw_runtime_error("Accessing the facet domain of the +| side element only works on the '_internal_facets_' subdomain, not on the domain " + this->get_domain_name());
					return this;
				}
				else if (domname == "|-")
				{
					if (this->get_domain_name() != "_internal_facets_" || !this->opposite_interface_code || !this->opposite_interface_code->bulk_code)
						throw_runtime_error("Accessing the facet domain of the |- side element only works on the '_internal_facets_' subdomain, not on the domain " + this->get_domain_name());
					return this->opposite_interface_code;
				}
				FiniteElementCode *bydomname = this->_resolve_based_on_domain_name(domname);
				if (!bydomname)
				{
					throw_runtime_error("Cannot resolve the domain name " + domname);
				}
				else
					return bydomname;
			}
		}

		throw_runtime_error("TODO: Resolve based on tags");
	}

	// For parallel problems, the derivatives  etc of CB functions may be in another order. This functions sorts them out by unique id (counted in order of their creation in python)
	// Derivatives can also be out of order due to GiNaCs missing ordering. Hence, we have to reconstruct this based on the derived parents
	void FiniteElementCode::fill_callback_info(JITFuncSpec_Table_FiniteElement_t *ft)
	{
		if (ft->numcallbacks != cb_expressions.size())
			throw_runtime_error("Mismatch in number of callback functions");
		std::vector<bool> used_once(cb_expressions.size(), false);
		unsigned numgood = 0;
		for (unsigned i = 0; i < ft->numcallbacks; i++)
			ft->callback_infos[i].cb_obj = NULL;

		for (unsigned i = 0; i < ft->numcallbacks; i++)
		{
			auto &ci = ft->callback_infos[i];
			if (ci.is_deriv_of == -1)
			{
				for (unsigned int j = 0; j < cb_expressions.size(); j++)
				{
					if ((!cb_expressions[j]->get_diff_parent()) && cb_expressions[j]->get_id_name() == std::string(ci.idname))
					{
						if (cb_expressions[j]->get_unique_id() == ci.unique_id)
						{
							if (used_once[j])
								throw_runtime_error("Ambigous callback functions");
							used_once[j] = true;
							ci.cb_obj = (void *)cb_expressions[j];
							numgood++;
							break;
						}
					}
				}
				if (!ci.cb_obj)
					throw_runtime_error("Cannot identify callback function (by unique id)");
			}
		}

		while (numgood != ft->numcallbacks)
		{
			unsigned oldnumgood = numgood;
			for (unsigned i = 0; i < ft->numcallbacks; i++)
			{
				auto &ci = ft->callback_infos[i];
				if (ci.is_deriv_of > -1)
				{
					auto &pci = ft->callback_infos[ci.is_deriv_of];
					if (pci.cb_obj) // Derivative parent already registered
					{
						for (unsigned int j = 0; j < cb_expressions.size(); j++)
						{
							if (!cb_expressions[j]->get_diff_parent() || used_once[j])
								continue;
							if (cb_expressions[j]->get_diff_parent() == pci.cb_obj && ci.deriv_index == cb_expressions[j]->get_diff_index())
							{
								if (cb_expressions[j]->get_id_name() == std::string(ci.idname))
								{
									used_once[j] = true;
									ci.cb_obj = (void *)cb_expressions[j];
									numgood++;
									break;
								}
							}
						}
					}
				}
			}
			if (numgood == oldnumgood)
				throw_runtime_error("Cannot identify all callback functions (via derivative parents)");
		}

		// Multi returns
		if (ft->num_multi_rets != multi_ret_expressions.size())
			throw_runtime_error("Mismatch in number of multi-return functions");
		used_once.clear();
		used_once.resize(multi_ret_expressions.size(), false);
		numgood = 0;
		for (unsigned i = 0; i < ft->num_multi_rets; i++)
			ft->multi_ret_infos[i].cb_obj = NULL;

		for (unsigned i = 0; i < ft->num_multi_rets; i++)
		{
			auto &ci = ft->multi_ret_infos[i];
			for (unsigned int j = 0; j < multi_ret_expressions.size(); j++)
			{
				if (multi_ret_expressions[j]->get_id_name() == std::string(ci.idname))
				{
					if (multi_ret_expressions[j]->unique_id == ci.unique_id)
					{
						if (used_once[j])
							throw_runtime_error("Ambigous multi-return functions");
						used_once[j] = true;
						ci.cb_obj = (void *)multi_ret_expressions[j];
						numgood++;
						break;
					}
				}
			}
			if (!ci.cb_obj)
				throw_runtime_error("Cannot identify multi-return function (by unique id)");
		}
	}

	// Sets the weighting factor used to include field `f` in the adaptive-timestepping temporal error
	// estimate (0 excludes it).
	void FiniteElementCode::set_temporal_error(std::string f, double factor)
	{
		auto *field = this->get_field_by_name(f);
		if (!field)
		{
			throw_runtime_error("Cannot set temporal error of an undefined field: " + f);
		}
		field->temporal_error_factor = factor;
	}

	// Expands and validates an initial-condition or Dirichlet-condition expression for `fieldname`:
	// after placeholder expansion, checks the expression is dimensionally consistent (units cancel to
	// 1), then performs a "dry run" evaluation, substituting the free spatial/Lagrangian/normal/time
	// symbols by the fixed reference point/time configured via set_reference_point_for_IC_and_DBC(),
	// all fields' own shape expansions by generic symbols, and global parameters by their current
	// numeric value - the resulting expression must reduce to a plain number. This "can this be
	// evaluated numerically at all" check catches, at element-generation time rather than at runtime,
	// initial/Dirichlet conditions that accidentally reference undefined variables or have leftover
	// (non-cancelling) units, which would otherwise silently propagate NaNs. Returns the expanded
	// (not evaluated at the reference point) expression for actual use by write_code_initial_condition/
	// write_code_Dirichlet_condition.
	GiNaC::ex FiniteElementCode::expand_initial_or_Dirichlet(const std::string &fieldname, GiNaC::ex expression)
	{

		// ReplaceFieldsToNonDimFields repl(this);
		// expression=0+repl(expression)/this->get_scaling(fieldname);
		expression = this->expand_placeholders(expression, "IC_or_DBC");
		RemoveSubexpressionsByIndentity sub_to_id(this);
		expression = sub_to_id(expression);
		GiNaC::ex units = 1;
		GiNaC::ex factor = 1;
		GiNaC::ex rest = 1;
		if ((!pyoomph::expressions::collect_base_units(expression, factor, units, rest)) || (units != 1))
		{
			std::ostringstream oss;
			oss << "Wrong physical dimensions [got " << units << "] in Dirichlet or initial condition for field '" << fieldname << "': " << expression << std::endl
				<< " GOT UNITS " << units << "  FACTOR " << factor << " REST " << rest;
			throw_runtime_error(oss.str());
		}
		expression = factor * units * rest;

		// GiNaC::lst sublist;

		/* std::vector<std::string> dir{"x","y","z"};
		 for (unsigned int i=0;i<this->nodal_dim;i++)
		 {
			 sublist.append(this->get_field_by_name("coordinate_"+dir[i])->get_shape_expansion()==GiNaC::symbol("_x["+std::to_string(i)+"]"));
			 sublist.append(this->get_field_by_name("mesh_"+dir[i])->get_shape_expansion()==GiNaC::symbol("_x["+std::to_string(i)+"]"));
		 }*/

		// Test if the initial condition is nondimensional and has no free parameters
		GiNaC::lst subslist;
		GiNaC::lst subslist2;
		GiNaC::potential_real_symbol interpolated_x("interpolated_x"), interpolated_y("interpolated_y"), interpolated_z("interpolated_z");
		GiNaC::potential_real_symbol normal_x("_normal[0]"), normal_y("_normal[1]"), normal_z("_normal[2]");
		subslist.append(this->get_normal_component(0) == normal_x);
		subslist.append(this->get_normal_component(1) == normal_y);
		subslist.append(this->get_normal_component(2) == normal_z);
		subslist2.append(this->get_normal_component(0) == normal_x);
		subslist2.append(this->get_normal_component(1) == normal_y);
		subslist2.append(this->get_normal_component(2) == normal_z);
		if (this->nodal_dim > 0)
		{
			subslist.append(pyoomph::expressions::x == interpolated_x);

			subslist2.append(this->get_field_by_name("coordinate_x")->get_shape_expansion() == interpolated_x);
			subslist2.append(this->get_field_by_name("mesh_x")->get_shape_expansion() == interpolated_x);
			if (this->nodal_dim > 1)
			{
				subslist.append(pyoomph::expressions::y == interpolated_y);
				subslist2.append(this->get_field_by_name("coordinate_y")->get_shape_expansion() == interpolated_y);
				subslist2.append(this->get_field_by_name("mesh_y")->get_shape_expansion() == interpolated_y);
				if (this->nodal_dim > 2)
				{
					subslist.append(pyoomph::expressions::z == interpolated_z);
					subslist2.append(this->get_field_by_name("coordinate_z")->get_shape_expansion() == interpolated_z);
					subslist2.append(this->get_field_by_name("mesh_z")->get_shape_expansion() == interpolated_z);
				}
			}
		}
		GiNaC::potential_real_symbol lagrangian_x("lagrangian_x"), lagrangian_y("lagrangian_y"), lagrangian_z("lagrangian_z");

		if (this->lagr_dim > 0)
		{
			subslist2.append(this->get_field_by_name("lagrangian_x")->get_shape_expansion() == lagrangian_x);
			if (this->lagr_dim > 1)
			{
				subslist2.append(this->get_field_by_name("lagrangian_y")->get_shape_expansion() == lagrangian_y);
				if (this->lagr_dim > 2)
				{
					subslist2.append(this->get_field_by_name("lagrangian_z")->get_shape_expansion() == lagrangian_z);
				}
			}
		}
		auto ts = GiNaC::GiNaCTimeSymbol(pyoomph::TimeSymbol());
		subslist.append(ts == pyoomph::expressions::t);
		GiNaC::ex subst = expression.subs(subslist);

		const std::vector<double> &ref = reference_pos_for_IC_and_DBC;
		GiNaC::ex substv = subst.subs(GiNaC::lst{interpolated_x, interpolated_y, interpolated_z, pyoomph::expressions::t, lagrangian_x, lagrangian_y, lagrangian_z, normal_x, normal_y, normal_z}, {ref[0], ref[1], ref[2], ref[3], ref[0], ref[1], ref[2], ref[4], ref[5], ref[6]});
		GiNaC::ex substv2 = substv.subs(subslist2);
		substv2 = substv2.subs(GiNaC::lst{interpolated_x, interpolated_y, interpolated_z, pyoomph::expressions::t, lagrangian_x, lagrangian_y, lagrangian_z, normal_x, normal_y, normal_z}, {ref[0], ref[1], ref[2], ref[3], ref[0], ref[1], ref[2], ref[4], ref[5], ref[6]});
		GlobalParamsToValues gp2val;
		substv2 = gp2val(substv2);
		try
		{

			substv2 = (0 + substv2).evalf();
			//	 		 std::cout << "WHAT " << substv2 << std::endl;
			if (!GiNaC::is_a<GiNaC::numeric>(substv2))
			{
				std::ostringstream oss;
				oss << "not a numeric: " << substv2;
				throw std::runtime_error(oss.str());
			}
			GiNaC::numeric num = GiNaC::ex_to<GiNaC::numeric>(substv2);
		}
		catch (const std::runtime_error &error)
		{
			std::ostringstream oss;
			oss << subst;
			substv2 = (0 + substv2).evalf();
			oss << std::endl
				<< "AFTER applying (float) :" << substv2;
			throw std::runtime_error("Cannot evaluate the following initial/Dirichlet condition, since it has unknown variables or units in it: " + oss.str());
		}
		GiNaC::lst sublist;
		for (auto &bu : base_units)
		{
			sublist.append(bu.second == 1);
		}

		return sub_to_id(subst.subs(sublist));
	}

	// Registers an initial condition expression for `fieldname` under the named IC set `ic_name`
	// (adding it to IC_names if new). `degraded_start` controls whether the first timestep uses a
	// lower-order (degraded) start scheme; "auto" picks it based on whether the IC actually depends
	// on time (if it doesn't, a normal/non-degraded start is safe).
	void FiniteElementCode::set_initial_condition(const std::string &fieldname, GiNaC::ex expression, std::string degraded_start, const std::string &ic_name)
	{
		FiniteElementField *field = this->get_field_by_name(fieldname);
		if (!field)
		{
			std::ostringstream oss;
			oss << std::endl;
			for (auto present_field : myfields)
			{
				oss << present_field->get_name() << std::endl;
			}
			throw_runtime_error("Cannot set initial condition of field '" + fieldname + "', since it is not defined in the element. Possible fields are:" + oss.str());
		}
		if (pyoomph_verbose)
			std::cout << "SETTING INIT COND " << expression << std::endl;
		int ic_index = -1;
		for (unsigned int i = 0; i < IC_names.size(); i++)
		{
			if (ic_name == IC_names[i])
			{
				ic_index = i;
				break;
			}
		}
		if (ic_index == -1)
			IC_names.push_back(ic_name);
		field->initial_condition[ic_name] = this->expand_initial_or_Dirichlet(fieldname, expression);
		if (degraded_start == "auto")
		{
			degraded_start = (field->initial_condition[ic_name].has(pyoomph::expressions::t) ? "no" : "yes");
		}
		field->degraded_start[ic_name] = (degraded_start == "yes");
		if (pyoomph_verbose)
			std::cout << "INIT COND SET: " << field->initial_condition[ic_name] << std::endl;
	}

	// get_dx_derived/get_elemsize_derived return the pre-built symbolic tag representing "the
	// derivative of the (Eulerian) integration measure / element size w.r.t. nodal coordinate
	// direction `dir`", picking the "second index" (l_shape2) variant instead when
	// __derive_shapes_by_second_index is currently set (i.e. while building the inner/second
	// derivative of a Hessian double loop) - this is what lets the same differentiation code path be
	// reused for both first derivatives (Jacobian) and the two independent index slots of second
	// derivatives (Hessian).
	const SpatialIntegralSymbol &FiniteElementCode::get_dx_derived(int dir)
	{
		if (__derive_shapes_by_second_index)
		{
			return dx_derived_lshape2_for_Hessian[dir];
		}
		else
		{
			return dx_derived[dir];
		}
	}

	const ElementSizeSymbol &FiniteElementCode::get_elemsize_derived(int dir, bool _consider_coordsys)
	{
		if (__derive_shapes_by_second_index)
		{
			return (_consider_coordsys ? elemsize_derived_lshape2_for_Hessian[dir] : elemsize_Cart_derived_lshape2_for_Hessian[dir]);
		}
		else
		{
			return (_consider_coordsys ? elemsize_derived[dir] : elemsize_Cart_derived[dir]);
		}
	}

	// Registers a Dirichlet boundary condition expression for `fieldname`. If `use_identity` is set,
	// the field is merely pinned to whatever value it already has (an "identity" Dirichlet condition,
	// e.g. for freezing a DoF without prescribing a specific value) rather than being set to the
	// evaluated expression.
	void FiniteElementCode::set_Dirichlet_bc(const std::string &fieldname, GiNaC::ex expression, bool use_identity)
	{
		FiniteElementField *field = this->get_field_by_name(fieldname);
		if (!field)
		{
		     std::string avfields="";
		     for (const auto & f : myfields) { if (avfields!="") avfields+=", "; avfields+=f->get_name(); }
		    throw_runtime_error("Cannot set Dirichlet condition of field '" + fieldname + " in domain '"+this->get_full_domain_name() +"', since it is not defined in the element.\nAvailable fields:\n"+avfields);
		}
		if (pyoomph_verbose)
			std::cout << "SETTING DIRICHLET COND " << expression << std::endl;
		field->Dirichlet_condition = this->expand_initial_or_Dirichlet(fieldname, expression);
		field->Dirichlet_condition_set = true;
		field->Dirichlet_condition_pin_only = use_identity;
		if (pyoomph_verbose)
			std::cout << "DIRICHLET COND SET: " << field->Dirichlet_condition << std::endl;
	}

	// Delegates to the user-supplied Equations object's _define_fields()/_define_element() hooks
	// (the Python-side equation definitions), temporarily binding `this` as their "current codegen"
	// target so field registration / residual calls issued from Python land on this code object.
	void FiniteElementCode::_define_fields()
	{
		if (!equations)
			throw_runtime_error("codegen: Cannot define the fields if no equations are set!");
		equations->_set_current_codegen(this);
		equations->_define_fields();
		equations->_set_current_codegen(NULL);
	}

	void FiniteElementCode::_define_element()
	{
		if (!equations)
			throw_runtime_error("codegen: Cannot define the equations if no equations are set!");
		equations->_set_current_codegen(this);
		equations->_define_element();
		equations->_set_current_codegen(NULL);
	}

	// Called once per code object to register the built-in position-related fields alongside the
	// user's own fields (_define_fields()): the nodal (Eulerian) coordinate_x/y/z fields, the fixed
	// Lagrangian reference coordinates, the element-local coordinates, (on codimension-1 interfaces
	// only) the surface parametrization zeta_coordinate_*, and the "mesh_*" pseudo-fields used purely
	// to let mesh-velocity terms have a nonzero time derivative while true position fields don't (see
	// MeshToCoordinateShapes above). Fixes `element_dim` for this code object; may only run once.
	void FiniteElementCode::_do_define_fields(int element_dimension)
	{
		if (this->element_dim != -1)
			throw_runtime_error("Equation element dimension was aready set. This usually happens, if you use the same codegen class instance multiple times in the problem");
		this->element_dim = element_dimension;
		this->_define_fields();

		for (unsigned int i = 0; i < this->nodal_dim; i++)
		{
			std::vector<std::string> dir{"x", "y", "z"};
			this->register_field("coordinate_" + dir[i], "Pos");
		}

		for (unsigned int i = 0; i < this->lagr_dim; i++)
		{
			std::vector<std::string> dir{"x", "y", "z"};
			this->register_field("lagrangian_" + dir[i], "Pos")->no_jacobian_at_all = true; // Lagrangian coordinates never have Jacobian entries, since they are fixed
		}

		for (unsigned int i = 0; i < static_cast<unsigned int>(this->element_dim); i++)
		{
			std::vector<std::string> dir{"1", "2", "3"};
			this->register_field("local_coordinate_" + dir[i], "Pos")->no_jacobian_at_all = true; // Lagrangian coordinates never have Jacobian entries, since they are fixed
		}		

		if (this->bulk_code && !this->bulk_code->bulk_code) 
		{
			// Only do this on co-dim 1 interfaces for now, then zetas are unique
			for (unsigned int i = 0; i < static_cast<unsigned int>(this->element_dim); i++)
			{
				std::vector<std::string> dir{"1", "2", "3"};
				this->register_field("zeta_coordinate_" + dir[i], "Pos")->no_jacobian_at_all = true; // Lagrangian coordinates never have Jacobian entries, since they are fixed
			}		
		}

		for (unsigned int i = 0; i < this->nodal_dim; i++) // Adding the mesh coordinates -> They in fact can be derived by t, whereas the partial_t( coordinate) =0
		{
			std::vector<std::string> dir{"x", "y", "z"};
			this->register_field("mesh_" + dir[i], "Pos");
		}
	}

	// Resets the residual accumulator(s) and invokes the user's _define_element() (weak-form
	// definition) hook, with expressions::el_dim temporarily set to this element's dimension so that
	// dimension-dependent symbolic constructs (e.g. spatial vectors of the right size) resolve
	// correctly during that call.
	void FiniteElementCode::finalise()
	{
		// residual[0]=0;
		residual.clear();
		residual_index = 0;
		residual_names.clear();
		residual.push_back(0);
		residual_names.push_back("");
		expressions::el_dim = this->element_dim;
		__current_code = this;
		_define_element();
		__current_code = NULL;
		expressions::el_dim = -1;
	}

	void FiniteElementCode::set_problem(Problem * p) {problem=p;}
	Problem * FiniteElementCode::get_problem() {return problem;}

	// Emits GeometricJacobian() (the coordinate-system Jacobian factor used by the Z2 error
	// estimator) and JacobianForElementSize() (the analogous factor used for element-size
	// computations, e.g. 2*pi*r terms in axisymmetric coordinates), both evaluated directly from the
	// raw position array `_x` rather than via shape-function interpolation. Additionally
	// differentiates JacobianForElementSize symbolically w.r.t. every spatial direction (and, if
	// nonzero, a second time) to emit JacobianForElementSizeSpatialDerivatives/
	// ...SecondSpatialDerivatives, but only if those derivatives are not identically zero
	// (geometric_jac_for_elemsize_has_[second_]spatial_deriv flags this so callers can skip invoking
	// functions that were not emitted at all).
	void FiniteElementCode::write_code_geometric_jacobian(std::ostream &os)
	{
		os << "// Used for Z2 error estimators" << std::endl;
		os << "static double GeometricJacobian(const JITElementInfo_t * eleminfo, const double * _x)" << std::endl;
		os << "{" << std::endl;
		GiNaC::ex geom_jacobian = expand_placeholders(this->get_coordinate_system()->geometric_jacobian(), "GeometricJacobian");
		GiNaC::lst sublist;
		//		std::cout << "NODAL DIM " << this->nodal_dim << " @ " << this << std::endl;

		std::vector<std::string> dir{"x", "y", "z"};
		std::vector<GiNaC::symbol> dir_syms;
		for (unsigned int i = 0; i < this->nodal_dim; i++)
		{
			//		    std::cout << "coordinate_"+dir[i] << "  : " << this->get_field_by_name("coordinate_"+dir[i])->get_shape_expansion() << std::endl;
			dir_syms.push_back(GiNaC::potential_real_symbol("_x[" + std::to_string(i) + "]"));
			sublist.append(this->get_field_by_name("coordinate_" + dir[i])->get_shape_expansion() == dir_syms.back());
			sublist.append(this->get_field_by_name("mesh_" + dir[i])->get_shape_expansion() == dir_syms.back());
		}
		/*
				 for (unsigned int i=0;i<this->lagr_dim;i++)
				 {
					 sublist.append(this->get_field_by_name("lagrangian_"+dir[i])->get_shape_expansion()==GiNaC::symbol("_xlagr["+std::to_string(i)+"]"));
				 }
			*/

		GiNaC::ex subst = geom_jacobian.subs(sublist);
		GiNaC::print_FEM_options csrc_opts;
		csrc_opts.for_code = this;
		os << "  return ";
		// subst.evalf().print(GiNaC::print_csrc_FEM(os,&csrc_opts));
		print_simplest_form(subst, os, csrc_opts);
		os << ";" << std::endl;
		os << "}" << std::endl;

		os << "// Used for elemsize_Eulerian etc" << std::endl;
		os << "static double JacobianForElementSize(const JITElementInfo_t * eleminfo, const double * _x)" << std::endl;
		os << "{" << std::endl;
		geom_jacobian = expand_placeholders(this->get_coordinate_system()->jacobian_for_element_size(), "JacobianForElementSize");
		subst = geom_jacobian.subs(sublist);
		os << "  return ";
		// subst.evalf().print(GiNaC::print_csrc_FEM(os,&csrc_opts));
		print_simplest_form(subst, os, csrc_opts);
		os << ";" << std::endl;
		os << "}" << std::endl
		   << std::endl;

		std::vector<GiNaC::ex> Jgrad;
		std::vector<GiNaC::ex> JHess;
		this->geometric_jac_for_elemsize_has_spatial_deriv = false;
		this->geometric_jac_for_elemsize_has_second_spatial_deriv = false;
		for (unsigned int i = 0; i < this->nodal_dim; i++)
		{
			GiNaC::ex deriv = GiNaC::diff(subst, dir_syms[i]);
			Jgrad.push_back(deriv);
			if (!GiNaC::is_zero(deriv))
				this->geometric_jac_for_elemsize_has_spatial_deriv = true;
			for (unsigned int j = 0; j < this->nodal_dim; j++)
			{
				GiNaC::ex second_deriv = GiNaC::diff(deriv, dir_syms[j]);
				JHess.push_back(second_deriv);
				if (!GiNaC::is_zero(second_deriv))
					this->geometric_jac_for_elemsize_has_second_spatial_deriv = true;
			}
		}
		if (this->geometric_jac_for_elemsize_has_spatial_deriv)
		{
			// Spatial Derivatives of the JacobianForElementSize
			os << "static void JacobianForElementSizeSpatialDerivatives(const JITElementInfo_t * eleminfo, const double * _x,double *grad)" << std::endl;
			os << "{" << std::endl;
			for (unsigned int i = 0; i < this->nodal_dim; i++)
			{
				os << "   grad[" << i << "] = ";
				print_simplest_form(Jgrad[i], os, csrc_opts);
				os << ";" << std::endl;
			}
			os << "}" << std::endl;
			if (this->geometric_jac_for_elemsize_has_second_spatial_deriv)
			{
				// Spatial Derivatives of the JacobianForElementSize
				os << "static void JacobianForElementSizeSecondSpatialDerivatives(const JITElementInfo_t * eleminfo, const double * _x,double *hessian)" << std::endl;
				os << "{" << std::endl;
				for (unsigned int i = 0; i < this->nodal_dim; i++)
				{
					for (unsigned int j = 0; j < this->nodal_dim; j++)
					{
						if (i != j)
							os << "   hessian[" << j * this->nodal_dim + i << "] = ";
						os << "   hessian[" << i * this->nodal_dim + j << "] = ";
						print_simplest_form(JHess[i * this->nodal_dim + j], os, csrc_opts);
						os << ";" << std::endl;
					}
				}
				os << "}" << std::endl;
			}
		}
	}

	// Emits the C statements that set the "shapes_required_<func_type>" runtime flags describing
	// exactly which shape data must be computed before invoking the `func_type` generated function
	// (e.g. "ResJac[0]" or "Hessian[2]") - populated earlier via mark_shapes_required(). First
	// determines *which* of this/bulk/bulk-of-bulk/opposite-interface/opposite-interface's-bulk
	// domains actually own any of the required spaces (raising an error if a required space cannot
	// be found in any reachable domain - a sign of an internal consistency bug in the generator),
	// then delegates the actual per-domain flag emission to write_required_shapes_for_code().
	void FiniteElementCode::write_required_shapes(std::ostream &os, const std::string indent, std::string func_type)
	{
		auto &entry = this->required_shapes[func_type];
		bool require_bulk = false;
		bool require_bulk_bulk = false;
		bool require_opposite_interface = false;
		bool require_opposite_bulk = false;
		for (auto &fieldentry : entry)
		{
			if (fieldentry.first == NULL)
			{
				continue; // No space attached
			}
			bool is_in_my_space = false;
			for (auto &s : spaces)
			{
				if (s == fieldentry.first)
				{
					is_in_my_space = true;
					break;
				}
			}
			if (is_in_my_space) continue;			
			bool found_elsewhere = false;
			if (bulk_code)
			{
				for (auto &s : bulk_code->spaces)
				{
					if (s == fieldentry.first)
					{
						require_bulk = true;
						found_elsewhere = true;
						break;
					}
				}
				if (!found_elsewhere && bulk_code->bulk_code)
				{
					for (auto &s : bulk_code->bulk_code->spaces)
					{
						if (s == fieldentry.first)
						{
							require_bulk = true;
							require_bulk_bulk = true;
							found_elsewhere = true;
							break;
						}
					}
				}
			}
			if (!found_elsewhere && opposite_interface_code)
			{
				for (auto &s : opposite_interface_code->spaces)
				{
					if (s == fieldentry.first)
					{
						require_opposite_interface = true;
						found_elsewhere = true;
						break;
					}
				}
			}
			if (!found_elsewhere && opposite_interface_code && opposite_interface_code->bulk_code)
			{
				for (auto &s : opposite_interface_code->bulk_code->spaces)
				{
					if (s == fieldentry.first)
					{
						require_opposite_interface = true;
						require_opposite_bulk = true;
						found_elsewhere = true;
						break;
					}
				}
			}
			if (!found_elsewhere)
			{
				std::ostringstream oss;
				oss << "Cannot find a required space " << fieldentry.first;
				throw_runtime_error(oss.str());
			}						
		}

		write_required_shapes_for_code(os,func_type,indent,this,0);
		if (require_bulk)
		{
			write_required_shapes_for_code(os,func_type,indent,this->bulk_code,1);
			if (require_bulk_bulk)
			{
				write_required_shapes_for_code(os,func_type,indent,this->bulk_code->bulk_code,2);
			}
		}
		
		if (require_opposite_interface)
		{
			write_required_shapes_for_code(os,func_type,indent,this->opposite_interface_code,-1);
			if (require_opposite_bulk)
			{
				write_required_shapes_for_code(os,func_type,indent,this->opposite_interface_code->bulk_code,-2);
			}
		}
		
	}


	// Emits the "functable->shapes_required_<func_type>[...].<flag> = true;" assignments for the
	// subset of required-shape entries whose FiniteElementSpace belongs to `for_code`. `type`
	// selects (and, for the nested bulk_shapes/opposite_shapes pointers, first calloc's) the correct
	// pointer-chased sub-struct to write into: 0 = this code itself (no prefix needed), 1/2 = the
	// bulk element / bulk-of-bulk element, -1/-2 = the opposite interface element / its bulk. The
	// continuous C1/C1TB/C2/C2TB spaces are addressed through a shared `continuous_spaces[SPACE_INDEX_*]`
	// array rather than individually named struct members, since they share the same shape layout.
	void FiniteElementCode::write_required_shapes_for_code(std::ostream & os, std::string func_type, std::string indent, FiniteElementCode *for_code, int type)
	{
		auto &entry = this->required_shapes[func_type];
		std::string prefix=indent+"functable->shapes_required_" + func_type+".";
		if (type==1)
		{
			os << " functable->shapes_required_" << func_type << ".bulk_shapes=(JITFuncSpec_RequiredShapes_FiniteElement_t*)calloc(sizeof(JITFuncSpec_RequiredShapes_FiniteElement_t),1);" << std::endl;
			prefix+= "bulk_shapes->";
		}
		else if (type==2)
		{
			os << " functable->shapes_required_" << func_type << ".bulk_shapes->bulk_shapes=(JITFuncSpec_RequiredShapes_FiniteElement_t*)calloc(sizeof(JITFuncSpec_RequiredShapes_FiniteElement_t),1);" << std::endl;
			prefix+= "bulk_shapes->bulk_shapes->";
		}
		else if (type==-1)
		{
			os << " functable->shapes_required_" << func_type << ".opposite_shapes=(JITFuncSpec_RequiredShapes_FiniteElement_t*)calloc(sizeof(JITFuncSpec_RequiredShapes_FiniteElement_t),1);" << std::endl;
			prefix+= "opposite_shapes->";
		}
		else if (type==-2)
		{
			os << " functable->shapes_required_" << func_type << ".opposite_shapes->bulk_shapes=(JITFuncSpec_RequiredShapes_FiniteElement_t*)calloc(sizeof(JITFuncSpec_RequiredShapes_FiniteElement_t),1);" << std::endl;
			prefix+= "opposite_shapes->bulk_shapes->";
		}
			
		for (auto &fieldentry : entry)
		{
			bool is_in_my_space = false;
			for (auto &s : for_code->spaces)
			{
				if (s == fieldentry.first)
				{
					is_in_my_space = true;
					break;
				}
			}
			if  (!is_in_my_space) continue;

			if (fieldentry.first == NULL)
			{
				// Write the stuff without a space
				for (auto &subentry : fieldentry.second)
				{
					if (subentry.second)
					{
						os << prefix  << subentry.first << " = true; THESE SHOULD NOT APPEAR" << std::endl;
					}
				}
				continue;
			}

			for (auto &psientry : fieldentry.second)
			{
				if (psientry.second)
				{
					if (psientry.first=="psi" || psientry.first=="dx_psi" || psientry.first=="dX_psi")
					{
						if (fieldentry.first->get_name()=="C1" || fieldentry.first->get_name()=="C1TB"||  fieldentry.first->get_name()=="C2" || fieldentry.first->get_name()=="C2TB")
						{
							os << prefix  << "continuous_spaces[SPACE_INDEX_"+fieldentry.first->get_name()+"]" << "." << psientry.first << " = true;" << std::endl;
						}						
						else
						{
							os << prefix  << fieldentry.first->get_name() << "." << psientry.first << " = true;" << std::endl;
						}												
					}
					else
					{
						os << prefix  << psientry.first  << " = true;" << std::endl;
					}
				}
			}
		}
	}

	

	// Resolves a named runtime "flag" symbol (eval_flag(...) placeholder) to its GiNaC expression:
	// "moving_mesh" becomes the constant 0/1 depending on coordinates_as_dofs, "timefrac_tracer"
	// becomes the special time-fraction symbol used for tracer-advection sub-stepping.
	GiNaC::ex FiniteElementCode::eval_flag(std::string flagname)
	{
		if (flagname == "moving_mesh")
		{
			return (coordinates_as_dofs ? 1 : 0);
		}
		if (flagname == "timefrac_tracer")
		{
			return pyoomph::expressions::timefrac_tracer;
		}
		else
			throw_runtime_error("Unknown flag name: " + flagname);
	}

	// The following resolve_subexpression/resolve_multi_return_call perform a linear search
	// (structural GiNaC equality) over the already-collected subexpressions/multi-return-call
	// invocations for one matching `e`/`invok`, returning a pointer / index, or NULL / -1 if not
	// found yet (a bit expensive for very large expressions, but these lists are typically small and
	// simplicity/correctness of the CSE bookkeeping is preferred here).
	FiniteElementCodeSubExpression *FiniteElementCode::resolve_subexpression(const GiNaC::ex &e)
	{
		if (pyoomph_verbose)
			std::cout << "SE RESOLVE " << e << std::endl;
		for (unsigned int i = 0; i < subexpressions.size(); i++)
		{
			if (pyoomph_verbose)
				std::cout << "TRYING " << i << subexpressions[i].get_expression() << std::endl;
			if (subexpressions[i].get_expression().is_equal(e))
				return &(subexpressions[i]);
		}
		return NULL;
	}

	int FiniteElementCode::resolve_multi_return_call(const GiNaC::ex &invok)
	{
		for (unsigned int i = 0; i < multi_return_calls.size(); i++)
		{
			if (multi_return_calls[i].is_equal(invok))
				return i;
		}
		return -1;
	}

	// Intended to suppress a bulk element's residual contribution for a given DoF on an interface
	// (e.g. to avoid double-counting), but currently unconditionally disabled (unimplemented feature,
	// per the leading throw_runtime_error) - the remaining validation logic below is dead code kept
	// for a possible future re-enablement.
	void FiniteElementCode::nullify_bulk_residual(std::string dofname)
	{
		throw_runtime_error("Nullified dofs are deactivated for now... Never used so far");
		if (!bulk_code)
		{
			throw_runtime_error("Cannot nullify bulk residuals without bulk element");
		}
		if (stage >= 2)
		{
			throw_runtime_error("Cannot nullify bulk residuals at this stage " + std::to_string(stage));
		}
		auto *bf = bulk_code->get_field_by_name(dofname);
		if (!bf)
		{
			throw_runtime_error("Cannot nullify bulk residuals of non-present DoF " + dofname);
		}
		if (!dynamic_cast<ContinuousFiniteElementSpace *>(bf->get_space()))
		{
			throw_runtime_error("Can only nullify bulk residuals on continuous spaces, but the DoF is discontinuous " + dofname);
		}
		for (auto a : nullified_bulk_residuals)
			if (dofname == a)
				return;
		nullified_bulk_residuals.push_back(dofname);
	}

	// Registers a user-defined "integral expression" named `name`. Scalar expressions are stored
	// directly; matrix/vector-valued expressions are decomposed component-wise into separately named
	// integral expressions (name_x/name_y/name_z for vectors, name_xx/name_xy/... for matrices up to
	// 3x3), skipping structurally-zero components (recorded as "" placeholders in the returned name
	// list so callers can still map component index to name/absence). Returns the list of actually
	// registered component names (empty vector for a plain scalar expression, which is stored under
	// `name` itself).
	std::vector<std::string> FiniteElementCode::register_integral_function(std::string name, GiNaC::ex expr)
	{
		RemoveSubexpressionsByIndentity sub_to_id(this);
		this->integral_expression_units[name] = 1;
		GiNaC::ex expanded = sub_to_id(expand_all_and_ensure_nondimensional(expr, "IntegralFunction", &(this->integral_expression_units[name]))).evalm();
		if (!GiNaC::is_a<GiNaC::matrix>(expanded))
		{
			this->integral_expressions[name] = expanded;
			return std::vector<std::string>();
		}
		else
		{
			GiNaC::matrix expam = GiNaC::ex_to<GiNaC::matrix>(expanded);
			std::vector<std::string> dirindex = {"x", "y", "z"};
			std::vector<std::string> res;
			if (expam.rows()>1 || expam.cols()>1)
			{
				
				for (unsigned int row = 0; row < std::min(expam.rows(), (unsigned int)3); row++)
				{
					for (unsigned int col = 0; col < std::min(expam.cols(), (unsigned int)3); col++)
					{
						std::string nam = name + "_" + dirindex[row] + dirindex[col];
						if (!GiNaC::is_zero(expam(row, col)))
						{
							this->integral_expressions[nam] = expam(row, col);
							this->integral_expression_units[nam] = this->integral_expression_units[name];
							res.push_back(nam);
						}
						else
						{
							res.push_back("");
						}
					}
				}
			}
			else
			{						
				for (unsigned int cd = 0; cd < std::max(expanded.nops(), (size_t)(3)); cd++)
				{
					std::string nam = name + "_" + dirindex[cd];
					if (!GiNaC::is_zero(expanded[cd]))
					{
						this->integral_expressions[nam] = expanded[cd];
						this->integral_expression_units[nam] = this->integral_expression_units[name];
						res.push_back(nam);
					}
					else
					{
						res.push_back("");
					}
				}
				
			}
			return res;
		}
	}

	// Registers the advection velocity field used to move passive tracer particles named `name`.
	// After nondimensionalizing, the unit factor must reduce to exactly [spatial]/[temporal] (a
	// velocity) - any other combination of units is a user error and raises a descriptive exception.
	void FiniteElementCode::set_tracer_advection_velocity(std::string name, GiNaC::ex expr)
	{
		RemoveSubexpressionsByIndentity sub_to_id(this);
		this->tracer_advection_units[name] = 1;
		this->tracer_advection_terms[name] = sub_to_id(expand_all_and_ensure_nondimensional(expr, "TracerVelocity", &(this->tracer_advection_units[name])));

		tracer_advection_units[name] = tracer_advection_units[name].evalf();
		if (GiNaC::is_a<GiNaC::numeric>(tracer_advection_units[name]))
		{
			this->tracer_advection_terms[name] *= tracer_advection_units[name];
			tracer_advection_units[name] = 1;
		}
		else
		{
			std::ostringstream oss;
			oss << "Nondimensionalized tracer velocity of tracer '" << name << "' has the unit " << tracer_advection_units[name] << " * [spatial]/[temporal], but should be [spatial]/[temporal] only";
			throw_runtime_error(oss.str());
		}
	}

	// Registers a scalar "extremum expression" (tracked for max/min over the mesh, e.g. for
	// diagnostics) named `name`. Unlike register_integral_function, the dimensional unit is not
	// required to be an integer power that cancels neatly - instead, the purely numeric/rational part
	// of the unit factor is folded back into the expression itself (so the stored expression is in
	// "as dimensional as needed, scaled by any leftover numeric factor" form) while the remaining
	// (purely symbolic) unit is kept in extremum_expression_units for later reporting. Only scalar
	// expressions are supported; vector/matrix-valued input raises an error.
	void FiniteElementCode::register_extremum_expression(std::string name, GiNaC::ex expr)
	{
		RemoveSubexpressionsByIndentity sub_to_id(this);
		this->local_expression_units[name] = 1;
		GiNaC::ex expanded = sub_to_id(expand_all_and_ensure_nondimensional(expr, "ExtremumExpression", &(this->extremum_expression_units[name])));
		GiNaC::ex factor, unit, rest;
		expressions::collect_base_units(this->extremum_expression_units[name], factor, unit, rest);
		this->extremum_expression_units[name] /= (factor * rest);
		expanded = (expanded * (factor * rest)).evalm();
		if (!GiNaC::is_a<GiNaC::matrix>(expanded))
		{
			this->extremum_expressions[name] = expanded;			
		}
		else
		{
			throw_runtime_error("Extremum expressions cannot be vectors or matrices");
		}
	}

	// Registers a "local expression" named `name`, analogous to register_integral_function but for
	// pointwise (non-integrated) expressions, with additional support for symmetric-matrix
	// components: for a matrix-valued expression, off-diagonal entries that are symbolically equal to
	// their transpose counterpart are not stored twice - the second name is simply aliased to the
	// first (via res containing the first name at the second position) so callers can still look them
	// up by either index. Returns the component name list together with a "shape code" second value:
	// -1 for a scalar, 0 for a vector, or the number of columns for a matrix (so callers can
	// distinguish vector vs. matrix layout from a single int).
	std::pair<std::vector<std::string>, int> FiniteElementCode::register_local_expression(std::string name, GiNaC::ex expr)
	{
		//			std::cout << "EXPR " << expr << std::endl;
		RemoveSubexpressionsByIndentity sub_to_id(this);
		this->local_expression_units[name] = 1;
		GiNaC::ex expanded = sub_to_id(expand_all_and_ensure_nondimensional(expr, "LocalExpression", &(this->local_expression_units[name])));
		//			std::cout << "EXPA " << expanded << std::endl;
		// std::cout << "MAATRIX " << expanded << std::endl;
		// Make sure it is positive and just the unit which is split up
		GiNaC::ex factor, unit, rest;
		expressions::collect_base_units(this->local_expression_units[name], factor, unit, rest);
		this->local_expression_units[name] /= (factor * rest);
		// std::cout << "MAATRIX " << expanded* (factor * rest) << std::endl;
		expanded = (expanded * (factor * rest)).evalm();
		// std::cout << "MAATRIX " << expanded << std::endl;
		if (!GiNaC::is_a<GiNaC::matrix>(expanded))
		{
			//      std::cout << "NO MATRIX" << expanded << std::endl;
			this->local_expressions[name] = expanded;
			return std::make_pair(std::vector<std::string>(), -1);
		}
		else
		{

			//      std::cout << "IS MATRIX" << expanded << std::endl;
			std::vector<std::string> dirindex = {"x", "y", "z"};
			std::vector<std::string> res;
			GiNaC::matrix expam = GiNaC::ex_to<GiNaC::matrix>(expanded);
			//			std::cout << "EXPAM " << expam << std::endl;
			if (expam.rows() <= 1 || expam.cols() <= 1)
			{
				for (unsigned int cd = 0; cd < std::max(expanded.nops(), (size_t)(3)); cd++)
				{
					std::string nam = name + "_" + dirindex[cd];
					if (!GiNaC::is_zero(expanded[cd]))
					{
						this->local_expressions[nam] = expanded[cd];
						this->local_expression_units[nam] = this->local_expression_units[name];
						res.push_back(nam);
					}
					else
					{
						res.push_back("");
					}
				}
				return std::make_pair(res, 0);
			}
			else
			{
				for (unsigned int ci = 0; ci < std::max(expam.cols(), (unsigned int)3); ci++)
				{
					for (unsigned int cj = 0; cj < std::max(expam.rows(), (unsigned int)3); cj++)
					{
						std::string nam = name + "_" + dirindex[ci] + dirindex[cj];
						if (!GiNaC::is_zero(expam(ci, cj)))
						{
							if (ci > cj && GiNaC::is_zero(expam(ci, cj) - expam(cj, ci)))
							{
								res.push_back(name + "_" + dirindex[cj] + dirindex[ci]);
							}
							else
							{
								this->local_expressions[nam] = expam(ci, cj);
								this->local_expression_units[nam] = this->local_expression_units[name];
								res.push_back(nam);
							}
						}
						else
						{
							res.push_back("");
						}
					}
				}
				return std::make_pair(res, (int)expam.cols());
			}
		}
	}

	// The following get_*_expression_unit_factor / get_*_expressions accessors expose the units and
	// registered names of integral/extremum/local expressions to callers (e.g. the Python binding
	// layer), defaulting to a unit factor of 1 for unknown names.
	GiNaC::ex FiniteElementCode::get_integral_expression_unit_factor(std::string name)
	{
		if (this->integral_expression_units.count(name))
		{
			return this->integral_expression_units[name];
		}
		else
		{
			return 1;
		}
	}



	GiNaC::ex FiniteElementCode::get_extremum_expression_unit_factor(std::string name)
	{
		if (this->extremum_expression_units.count(name))
		{
			return this->extremum_expression_units[name];
		}
		else
		{
			return 1;
		}
	}

	GiNaC::ex FiniteElementCode::get_local_expression_unit_factor(std::string name)
	{
		if (this->local_expression_units.count(name))
		{
			return this->local_expression_units[name];
		}
		else
		{
			return 1;
		}
	}

	std::vector<std::string> FiniteElementCode::get_integral_expressions()
	{
		std::vector<std::string> res;
		for (auto &e : this->integral_expressions)
			res.push_back(e.first);
		return res;
	}

	std::vector<std::string> FiniteElementCode::get_local_expressions()
	{
		std::vector<std::string> res;
		for (auto &e : this->local_expressions)
			res.push_back(e.first);
		return res;
	}

	std::vector<std::string> FiniteElementCode::get_extremum_expressions()
	{
		std::vector<std::string> res;
		for (auto &e : this->extremum_expressions)
			res.push_back(e.first);
		return res;		
	}

	// The master "glue" function: emits JIT_ELEMENT_init()/JIT_ELEMENT_finalize(), the two functions
	// that populate/tear down the runtime JITFuncSpec_Table_FiniteElement_t struct describing this
	// generated element to the rest of pyoomph (see jitbridge.h/elements.cpp). This is by far the
	// largest single function in the file; at a high level it emits code that:
	//   - runs ABI/struct-layout sanity checks (functable->check_compiler_size(...)) comparing the
	//     sizes the C compiler that will *load* this shared library sees against the sizes recorded
	//     when *this* generator was compiled, catching struct-layout mismatches between the JIT
	//     compiler and the host process early instead of via memory corruption;
	//   - records per-space field counts/names/index offsets (nodal Pos/DL/D0 spaces, the shared
	//     continuous C1/C1TB/C2/C2TB spaces, and the DG D1/D1TB/D2/D2TB spaces), validating that all
	//     fields sharing this element agree on which space is used as "the" coordinate space;
	//   - wires up function pointers for every previously emitted Residual/Jacobian/Hessian/steady/
	//     parameter-derivative routine, the required-shapes flags (via write_required_shapes), initial/
	//     Dirichlet condition callbacks, Z2 flux / integral / local / extremum / tracer-advection
	//     dispatchers, and the callback/multi-return function tables (fill_callback_info);
	//   - emits the mirrored JIT_ELEMENT_finalize() that frees everything allocated in _init().
	// Given its size and the fact that it is almost entirely straight-line "os << ... ;" statements
	// mirroring the JITFuncSpec_Table_FiniteElement_t struct layout field-by-field, only the
	// noteworthy/non-obvious steps are commented individually below rather than every assignment.
	void FiniteElementCode::write_code_info(std::ostream &os)
	{
	   std::ostringstream init,cleanup;
		init << "JIT_API void JIT_ELEMENT_init(JITFuncSpec_Table_FiniteElement_t *functable)" << std::endl;
		init << "{" << std::endl;

		init << " functable->check_compiler_size(sizeof(char),"<<sizeof(char)<<", \"char\");" << std::endl;		
		init << " functable->check_compiler_size(sizeof(unsigned short),"<<sizeof(unsigned short)<<", \"unsigned short\");" << std::endl;
      init << " functable->check_compiler_size(sizeof(unsigned int),"<<sizeof(unsigned int)<<", \"unsigned int\");" << std::endl;
      init << " functable->check_compiler_size(sizeof(unsigned long int),"<<sizeof(unsigned long int)<<", \"unsigned long int\");" << std::endl;
      init << " functable->check_compiler_size(sizeof(unsigned long long int),"<<sizeof(unsigned long long int)<<", \"unsigned long long int\");" << std::endl;
      init << " functable->check_compiler_size(sizeof(float),"<<sizeof(float)<<", \"float\");" << std::endl;
      init << " functable->check_compiler_size(sizeof(double),"<<sizeof(double)<<", \"double\");" << std::endl;      
      init << " functable->check_compiler_size(sizeof(size_t),"<<sizeof(size_t)<<", \"size_t\");" << std::endl;
      
      init << " functable->check_compiler_size(sizeof(struct JITElementInfo),"<<sizeof(struct JITElementInfo)<<", \"struct JITElementInfo\");" << std::endl;
      
      init << " functable->check_compiler_size(sizeof(struct JITHangInfoEntry),"<<sizeof(struct JITHangInfoEntry)<<", \"struct JITHangInfoEntry\");" << std::endl;
      init << " functable->check_compiler_size(sizeof(struct JITHangInfo),"<<sizeof(struct JITHangInfo)<<", \"struct JITHangInfo\");" << std::endl;
      init << " functable->check_compiler_size(sizeof(struct JITShapeInfo),"<<sizeof(struct JITShapeInfo)<<", \"struct JITShapeInfo\");" << std::endl;
      init << " functable->check_compiler_size(sizeof(struct JITFuncSpec_RequiredShapes_FiniteElement),"<<sizeof(struct JITFuncSpec_RequiredShapes_FiniteElement)<<", \"struct JITFuncSpec_RequiredShapes_FiniteElement\");" << std::endl;
      init << " functable->check_compiler_size(sizeof(struct JITFuncSpec_Callback_Entry),"<<sizeof(struct JITFuncSpec_Callback_Entry)<<", \"struct JITFuncSpec_Callback_Entry\");" << std::endl;
      init << " functable->check_compiler_size(sizeof(struct JITFuncSpec_MultiRet_Entry),"<<sizeof(struct JITFuncSpec_MultiRet_Entry)<<", \"struct JITFuncSpec_MultiRet_Entry\");" << std::endl;
      init << " functable->check_compiler_size(sizeof(struct JITFuncSpec_Table_FiniteElement),"<<sizeof(struct JITFuncSpec_Table_FiniteElement)<<", \"struct JITFuncSpec_Table_FiniteElement\");" << std::endl;


		init << " functable->nodal_dim=" << this->nodal_dim << ";" << std::endl;
		init << " functable->lagr_dim=" << this->lagr_dim << ";" << std::endl;

		init << " functable->fd_jacobian=" << (analytical_jacobian ? "false" : "true") << "; " << std::endl;
		init << " functable->fd_position_jacobian=" << (analytical_position_jacobian ? "false" : "true") << "; " << std::endl;
		init << " functable->with_adaptivity=" << (with_adaptivity ? "true" : "false") << "; " << std::endl;
		init << " functable->debug_jacobian_epsilon = " << debug_jacobian_epsilon << ";" << std::endl;
		init << " functable->stop_on_jacobian_difference = " << (stop_on_jacobian_difference ? "true" : "false") << ";" << std::endl;

		
		for (std::string my_sp : std::vector<std::string>{"Pos","DL","D0"})
		{
			init << " strcpy(functable->info_" << my_sp << ".space_name, \"" << my_sp << "\");" << std::endl;
		}
		for (std::string my_sp : std::vector<std::string>{"C2TB","C2","C1TB","C1"})
		{
			init << " strcpy(functable->continuous_spaces[SPACE_INDEX_" << my_sp << "].space_name, \"" << my_sp << "\");" << std::endl;
		}		
		for (std::string my_sp : std::vector<std::string>{"D2TB","D2","D1TB","D1"})
		{
			init << " strcpy(functable->dg_spaces[SPACE_INDEX_" << my_sp << "].space_name, \"" << my_sp << "\");" << std::endl;
		}

		int index_offset = 0;

		for (auto &space : spaces)
		{
			if (!dynamic_cast<PositionFiniteElementSpace *>(space))
				continue; // Separate both space types
			int numfields = 0;
			for (auto &f : myfields)
			{
				if (f->get_space() == space)
				{
					if (f->get_name() != "mesh_x" && f->get_name() != "mesh_y" && f->get_name() != "mesh_z")
						numfields++;
				}
			}
			if (numfields)
			{
								
				init << " functable->info_" << space->get_name() << ".numfields=" << numfields << ";" << std::endl;
				init << " functable->info_" << space->get_name() << ".fieldnames=(char **)malloc(sizeof(char*)*functable->info_" << space->get_name() << ".numfields);" << std::endl;				
				for (auto &f : myfields)
				{
					if (f->get_space() != space)
						continue;
					if (f->get_name() == "mesh_x" || f->get_name() == "mesh_y" || f->get_name() == "mesh_z")
						continue;
					
					
					init << " SET_INTERNAL_FIELD_NAME(functable->info_" << space->get_name() << ".fieldnames," << (f->index - index_offset) << ", \"" << f->get_name() << "\" );" << std::endl;										
					cleanup << " pyoomph_tested_free(functable->info_" << space->get_name() << ".fieldnames[" << (f->index - index_offset) << "]); functable->info_" << space->get_name() << ".fieldnames[" << (f->index - index_offset) << "]=PYOOMPH_NULL; " << std::endl;
				}
								
				cleanup << " pyoomph_tested_free(functable->info_" << space->get_name() << ".fieldnames); functable->info_" << space->get_name() << ".fieldnames=PYOOMPH_NULL; " << std::endl;
				index_offset += numfields;
			}
		}

		bool coordinate_space_validated = false;
		bool has_C1TB_fields=false;
		index_offset = 0;
		unsigned int base_bulk_nodal_offset = 0;
		unsigned int internal_data_offset = 0;
		unsigned int DG_external_offset = 0;
//		unsigned int interf_buffer_offset = 0;
		for (auto &space : spaces)
		{
			if (dynamic_cast<PositionFiniteElementSpace *>(space))
				continue; // Separate both space types
			//		std::cout << "MY SPACE " << space->get_name() << std::endl;
			int numfields = 0;

			for (auto &f : myfields)
			{
				if (f->get_space() == space)
					numfields++;
			}
			std::string info_name = "info_" + space->get_name();
			if (space->get_name() == "C1" || space->get_name() == "C1TB" || space->get_name() == "C2" || space->get_name() == "C2TB")
			{
				info_name = "continuous_spaces[SPACE_INDEX_" + space->get_name() + "]";
			}
			else if (space->get_name() == "D2TB" || space->get_name() == "D2" || space->get_name() == "D1TB" || space->get_name() == "D1" )
			{
				info_name = "dg_spaces[SPACE_INDEX_" + space->get_name() + "]";
			}			
			//		std::cout << "NUMFIELDS " << numfields << std::endl;
			if (numfields)
			{
				//  std::cout << "ENTERING " << space->get_name() << "  " << coordinate_space << std::endl;
				if (coordinate_space == "")
				{
					coordinate_space = space->get_name();
					coordinate_space_validated = true;
				}
				else if (!coordinate_space_validated)
				{
					if (coordinate_space != space->get_name())
					{
						throw_runtime_error("Cannot use a coordinate space of " + coordinate_space + ", which is inferior to the required nodal field space " + space->get_name());
					}
					else
						coordinate_space_validated = true;
				}
				init << " functable->" << info_name << ".numfields=" << numfields << ";" << std::endl;

				if (dynamic_cast<ContinuousFiniteElementSpace *>(space) || dynamic_cast<DGFiniteElementSpace *>(space))
				{
					// Find out the fields which are really defined on the bulk
					// Other fields stem from the interface
					if (!bulk_code)
					{
						
						init << " functable->" << info_name << ".numfields_bulk=" << numfields << ";" << std::endl;
						init << " functable->" << info_name << ".numfields_basebulk=" << numfields << ";" << std::endl;
						init << " functable->" << info_name << ".numfields_new=" << numfields << ";" << std::endl;

						if (space->get_name()=="C1TB" && numfields>0) has_C1TB_fields=true;
						
						if (dynamic_cast<ContinuousFiniteElementSpace *>(space))
						{
							init << " functable->" << info_name << ".nodal_offset_basebulk =" << base_bulk_nodal_offset << ";" << std::endl;
						}
						else if (dynamic_cast<DGFiniteElementSpace *>(space))
						{
							init << " functable->" << info_name << ".internal_offset_new =" << internal_data_offset << ";" << std::endl;
							internal_data_offset += numfields;
						}
						init << " functable->" << info_name << ".buffer_offset_basebulk =" << base_bulk_nodal_offset << ";" << std::endl;
						base_bulk_nodal_offset += numfields;
					}
					else
					{
						// Count the fields which are defined on the bulk
						FiniteElementCode *bc = bulk_code;
						// while (bc->bulk_code) bc=bc->bulk_code; //Step down to the actual bulk (for eg contact line elements)
						unsigned ncbulk = 0;
						for (auto &s : bc->spaces)
						{
							std::string bspn = s->get_name();
							//              if (bspn=="C2TB") bspn="C2";
							if (bspn == space->get_name())
							{
								for (auto &f : bc->myfields)
								{
									if (f->get_space() == s)
										ncbulk++;
								}
								break; // May not be used due to C2TB
							}
						}

						bc = bulk_code;
						while (bc->bulk_code)
							bc = bc->bulk_code; // Step down to the actual bulk (for eg contact line elements)
						unsigned ncbasebulk = 0;
						for (auto &s : bc->spaces)
						{
							std::string bspn = s->get_name();
							//	  std::cout << "IN " <<  s->get_name() << " " << std::endl;
							//              if (bspn=="C2TB") bspn="C2";
							if (bspn == space->get_name())
							{
								for (auto &f : bc->myfields)
								{
									//			 	      std::cout << "  FIELD ENTRY " <<  f->get_name() << " " << f->get_space()->get_name() << "  " << f->get_space() <<"==" << s<<  " || " <<f->get_space()->get_name()<< " == " << "C2TB" << std::endl;
									if (f->get_space() == s)
										ncbasebulk++;
								}
								break; // May not be used due to C2TB
							}
						}

						init << " functable->" << info_name << ".numfields_bulk=" << ncbulk << ";" << std::endl;
						init << " functable->" << info_name << ".numfields_basebulk=" << ncbasebulk << ";" << std::endl;
						init << " functable->" << info_name << ".numfields_new=" << numfields - ncbulk << ";" << std::endl;
						if (dynamic_cast<ContinuousFiniteElementSpace *>(space))
						{

							init << " functable->" << info_name << ".nodal_offset_basebulk =" << base_bulk_nodal_offset << ";" << std::endl;
						}
						else if (dynamic_cast<DGFiniteElementSpace *>(space))
						{
							init << " functable->" << info_name << ".internal_offset_new =" << internal_data_offset << ";" << std::endl;							
							init << " functable->" << info_name << ".external_offset_bulk = " << DG_external_offset << ";" << std::endl;
							internal_data_offset += (numfields - ncbulk);
							DG_external_offset += ncbulk;
						}

						
						init << " functable->" << info_name << ".buffer_offset_basebulk =" << base_bulk_nodal_offset << ";" << std::endl;
						base_bulk_nodal_offset += ncbasebulk;
					}
				}
				else if (dynamic_cast<DiscontinuousFiniteElementSpace *>(space))
				{					
					init << " functable->" << info_name << ".buffer_offset_basebulk =" << index_offset << "; // Using _basebulk here" << std::endl;
					if (!dynamic_cast<ExternalD0Space *>(space))
					{						
						init << " functable->" << info_name << ".internal_offset_new =" << internal_data_offset << "; // using _new here" << std::endl;
						internal_data_offset += numfields;
					}
					else
					{						
						init << " functable->" << info_name << ".external_offset_bulk = " << DG_external_offset << "; // using _bulk here" << std::endl;
						DG_external_offset += numfields;
					}
				}
				init << " functable->" << info_name << ".fieldnames=(char **)malloc(sizeof(char*)*functable->" << info_name << ".numfields);" << std::endl;
				std::map<unsigned, FiniteElementField *> reindex;
				for (auto &f : myfields)
				{
					if (f->get_space() != space)
						continue;
					reindex.insert(std::make_pair(f->index, f));
				}
				std::map<unsigned, int> reindex2;
				unsigned cnt = 0;
				for (auto &pair : reindex)
				{
					reindex2.insert(std::make_pair(pair.first, cnt++));
				}
				for (auto &f : myfields)
				{
					if (f->get_space() != space)
						continue;
					unsigned contiindex = reindex2[f->index];
					init << " SET_INTERNAL_FIELD_NAME(functable->" << info_name << ".fieldnames," << contiindex << ", \"" << f->get_name() << "\" );" << std::endl;					
					cleanup << " pyoomph_tested_free(functable->" << info_name << ".fieldnames[" << (contiindex) <<"]); functable->" << info_name << ".fieldnames[" <<(contiindex) << "]=PYOOMPH_NULL; " << std::endl;
				}
				cleanup << " pyoomph_tested_free(functable->" << info_name << ".fieldnames); functable->" << info_name << ".fieldnames=PYOOMPH_NULL; " << std::endl;
				index_offset += numfields;
			}
			else if (!coordinate_space_validated && coordinate_space != "")
			{
				if (coordinate_space == space->get_name())
				{
					coordinate_space_validated = true;
				}
			}
		}

		if (bulk_code)
		{
					
			init << " functable->continuous_spaces[SPACE_INDEX_C2TB].buffer_offset_interf=functable->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk+functable->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk+functable->continuous_spaces[SPACE_INDEX_C1TB].numfields_basebulk+functable->continuous_spaces[SPACE_INDEX_C1].numfields_basebulk" << std::endl;
			init << "                                     +functable->dg_spaces[SPACE_INDEX_D2TB].numfields_basebulk+functable->dg_spaces[SPACE_INDEX_D2].numfields_basebulk+functable->dg_spaces[SPACE_INDEX_D1TB].numfields_basebulk+functable->dg_spaces[SPACE_INDEX_D1].numfields_basebulk;" << std::endl;
			init << " functable->continuous_spaces[SPACE_INDEX_C2].buffer_offset_interf=functable->continuous_spaces[SPACE_INDEX_C2TB].buffer_offset_interf+(functable->continuous_spaces[SPACE_INDEX_C2TB].numfields-functable->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk);" << std::endl;
			init << " functable->continuous_spaces[SPACE_INDEX_C1TB].buffer_offset_interf=functable->continuous_spaces[SPACE_INDEX_C2].buffer_offset_interf+(functable->continuous_spaces[SPACE_INDEX_C2].numfields-functable->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk);" << std::endl;
			init << " functable->continuous_spaces[SPACE_INDEX_C1].buffer_offset_interf=functable->continuous_spaces[SPACE_INDEX_C1TB].buffer_offset_interf+(functable->continuous_spaces[SPACE_INDEX_C1TB].numfields-functable->continuous_spaces[SPACE_INDEX_C1TB].numfields_basebulk);" << std::endl;
			init << " functable->dg_spaces[SPACE_INDEX_D2TB].buffer_offset_interf=functable->continuous_spaces[SPACE_INDEX_C1].buffer_offset_interf+(functable->continuous_spaces[SPACE_INDEX_C1].numfields-functable->continuous_spaces[SPACE_INDEX_C1].numfields_basebulk);" << std::endl;
			init << " functable->dg_spaces[SPACE_INDEX_D2].buffer_offset_interf=functable->dg_spaces[SPACE_INDEX_D2TB].buffer_offset_interf+(functable->dg_spaces[SPACE_INDEX_D2TB].numfields-functable->dg_spaces[SPACE_INDEX_D2TB].numfields_basebulk);" << std::endl;
			init << " functable->dg_spaces[SPACE_INDEX_D1TB].buffer_offset_interf=functable->dg_spaces[SPACE_INDEX_D2].buffer_offset_interf+(functable->dg_spaces[SPACE_INDEX_D2].numfields-functable->dg_spaces[SPACE_INDEX_D2].numfields_basebulk);" << std::endl;
			init << " functable->dg_spaces[SPACE_INDEX_D1].buffer_offset_interf=functable->dg_spaces[SPACE_INDEX_D1TB].buffer_offset_interf+(functable->dg_spaces[SPACE_INDEX_D1TB].numfields-functable->dg_spaces[SPACE_INDEX_D1TB].numfields_basebulk);" << std::endl;
			init << "#ifndef PYOOMPH_TCC_TO_MEMORY" << std::endl;
			init << " if (functable->dg_spaces[SPACE_INDEX_D1].buffer_offset_interf+(functable->dg_spaces[SPACE_INDEX_D1].numfields-functable->dg_spaces[SPACE_INDEX_D1].numfields_basebulk)+functable->info_DL.numfields+functable->info_D0.numfields+functable->info_ED0.numfields!=" << index_offset << ")" << std::endl;
			init << " {" << std::endl;
			init << "   printf(\"Error in the buffer offsets. Please report with the script you have used to create this error!\\nbuffer_offset_C2TB_interf=%d\\nbuffer_offset_C2_interf=%d\\nbuffer_offset_C1TB_interf=%d\\nbuffer_offset_C1_interf=%d\\nbuffer_offset_D2TB_interf=%d\\nbuffer_offset_D2_interf=%d\\nbuffer_offset_D1TB_interf=%d\\nbuffer_offset_D1_interf=%d\\n\",functable->continuous_spaces[SPACE_INDEX_C2TB].buffer_offset_interf,functable->continuous_spaces[SPACE_INDEX_C2].buffer_offset_interf,functable->continuous_spaces[SPACE_INDEX_C1TB].buffer_offset_interf,functable->continuous_spaces[SPACE_INDEX_C1].buffer_offset_interf,functable->dg_spaces[SPACE_INDEX_D2TB].buffer_offset_interf,functable->dg_spaces[SPACE_INDEX_D2].buffer_offset_interf,functable->dg_spaces[SPACE_INDEX_D1TB].buffer_offset_interf,functable->dg_spaces[SPACE_INDEX_D1].buffer_offset_interf);" << std::endl;
			init << "   exit(1);" << std::endl;
			init << " }" << std::endl;
			init << "#endif" << std::endl;
		}
		if (coordinate_space == "D0" || coordinate_space == "DL" || coordinate_space == "D1")
			coordinate_space = "C1";
		else if (coordinate_space == "D2")
			coordinate_space = "C2";
		else if (coordinate_space == "C1TB")
			coordinate_space = "C1TB";
		else if (coordinate_space == "D2TB")
			coordinate_space = "C2TB";
		if (coordinate_space=="C2" && has_C1TB_fields ) coordinate_space="C2TB"; // Only here, we have the bubble
		if (coordinate_space == "" || coordinate_space=="ED0")
			throw_runtime_error("Cannot deduce the coordinate space of domain " + this->get_domain_name() + ". Please specify it explicitly by adding an ElementSpace().");
		//   if (coordinate_space=="C2TB" && this->bulk_code) coordinate_space="C2";
		init << " functable->dominant_space=strdup(\"" << coordinate_space << "\");" << std::endl;

		init << " functable->info_Pos.hangindex=-1; //Position always hangs on the max space" << std::endl;
		if (coordinate_space == "C1" || coordinate_space == "C1TB")
		{			
			init << " functable->continuous_spaces[SPACE_INDEX_C2TB].hangindex=-1;" << std::endl;
			init << " functable->continuous_spaces[SPACE_INDEX_C2].hangindex=-1;" << std::endl;
			init << " functable->continuous_spaces[SPACE_INDEX_C1TB].hangindex=-1;" << std::endl;
			init << " functable->continuous_spaces[SPACE_INDEX_C1].hangindex=-1;" << std::endl;
		}
		else
		{			
			init << " functable->continuous_spaces[SPACE_INDEX_C2TB].hangindex=-1;" << std::endl;
			init << " functable->continuous_spaces[SPACE_INDEX_C2].hangindex=-1;" << std::endl;
			init << " functable->continuous_spaces[SPACE_INDEX_C1TB].hangindex=functable->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk+functable->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk;" << std::endl;
			init << " functable->continuous_spaces[SPACE_INDEX_C1].hangindex=functable->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk+functable->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk;" << std::endl;
		}

		init << " functable->max_dt_order=" << this->max_dt_order << ";" << std::endl;
		init << " functable->moving_nodes=" << (this->coordinates_as_dofs ? "true" : "false") << ";" << std::endl;

		if (!nullified_bulk_residuals.empty())
		{
			throw_runtime_error("Nullified dofs are deactivated for now... Never used so far");
			init << " functable->num_nullified_bulk_residuals=" << nullified_bulk_residuals.size() << ";" << std::endl;
			init << " functable->nullified_bulk_residuals=(char **)malloc(sizeof(char*)*functable->num_nullified_bulk_residuals);" << std::endl;
			for (unsigned int i = 0; i < nullified_bulk_residuals.size(); i++)
			{
				init << " SET_INTERNAL_FIELD_NAME(functable->nullified_bulk_residuals," << i << ", \"" << nullified_bulk_residuals[i] << "\" );" << std::endl;
				cleanup << " pyoomph_tested_free(functable->nullified_bulk_residuals["<<i<<"]); functable->nullified_bulk_residuals["<<i<<"]=PYOOMPH_NULL; " << std::endl;								
			}
			cleanup << " pyoomph_tested_free(functable->nullified_bulk_residuals); functable->nullified_bulk_residuals=PYOOMPH_NULL; " << std::endl;											
		}

		init << " functable->num_res_jacs=" << residual.size() << ";" << std::endl;
		if (!global_parameter_to_local_indices.empty())
		{
			init << " functable->numglobal_params=" << global_parameter_to_local_indices.size() << ";" << std::endl;
			init << " functable->global_paramindices=(unsigned *)malloc(sizeof(unsigned)*functable->numglobal_params);" << std::endl;
			cleanup << " pyoomph_tested_free(functable->global_paramindices); functable->global_paramindices=PYOOMPH_NULL; " << std::endl;
			init << " functable->global_parameters=(double **)calloc(functable->numglobal_params,sizeof(double*));" << std::endl;
			cleanup << " pyoomph_tested_free(functable->global_parameters); functable->global_parameters=PYOOMPH_NULL; " << std::endl;			
			for (auto &gp : global_parameter_to_local_indices)
			{
				init << " functable->global_paramindices[" << gp.second << "]=" << gp.first << ";" << std::endl;
			}
			init << " functable->ParameterDerivative=(JITFuncSpec_ResidualAndJacobian_FiniteElement **)malloc(sizeof(JITFuncSpec_ResidualAndJacobian_FiniteElement)*functable->num_res_jacs);" << std::endl;

			for (unsigned int i = 0; i < residual.size(); i++)
			{
				init << " functable->ParameterDerivative[" << i << "]=(JITFuncSpec_ResidualAndJacobian_FiniteElement *)malloc(sizeof(JITFuncSpec_ResidualAndJacobian_FiniteElement)*functable->numglobal_params);" << std::endl;
				local_parameter_has_deriv[i].resize(global_parameter_to_local_indices.size(), false);
				for (auto &gp : global_parameter_to_local_indices)
				{
					if (local_parameter_has_deriv[i][gp.second])
					{
						init << " functable->ParameterDerivative[" << i << "][" << gp.second << "]=&dResidual" << i << "dParameter_" << gp.second << ";" << std::endl;
					}
					else
					{
						init << " functable->ParameterDerivative[" << i << "][" << gp.second << "]=PYOOMPH_NULL;" << std::endl;
					}
				}
			  cleanup << " pyoomph_tested_free(functable->ParameterDerivative["<<i<<"]); functable->ParameterDerivative["<<i<<"]=PYOOMPH_NULL; " << std::endl;					
			}
			
			cleanup << " pyoomph_tested_free(functable->ParameterDerivative); functable->ParameterDerivative=PYOOMPH_NULL; " << std::endl;	
		}

		init << " functable->ResidualAndJacobian_NoHang=(JITFuncSpec_ResidualAndJacobian_FiniteElement *)calloc(functable->num_res_jacs,sizeof(JITFuncSpec_ResidualAndJacobian_FiniteElement));" << std::endl;
		cleanup << " pyoomph_tested_free(functable->ResidualAndJacobian_NoHang); functable->ResidualAndJacobian_NoHang=PYOOMPH_NULL; " << std::endl;			
		init << " functable->ResidualAndJacobian=(JITFuncSpec_ResidualAndJacobian_FiniteElement *)calloc(functable->num_res_jacs,sizeof(JITFuncSpec_ResidualAndJacobian_FiniteElement));" << std::endl;
		cleanup << " pyoomph_tested_free(functable->ResidualAndJacobian); functable->ResidualAndJacobian=PYOOMPH_NULL; " << std::endl;					
		init << " functable->ResidualAndJacobianSteady=(JITFuncSpec_ResidualAndJacobian_FiniteElement *)calloc(functable->num_res_jacs,sizeof(JITFuncSpec_ResidualAndJacobian_FiniteElement));" << std::endl;
		cleanup << " pyoomph_tested_free(functable->ResidualAndJacobianSteady); functable->ResidualAndJacobianSteady=PYOOMPH_NULL; " << std::endl;							
		init << " functable->shapes_required_ResJac=(JITFuncSpec_RequiredShapes_FiniteElement_t *)calloc(functable->num_res_jacs,sizeof(JITFuncSpec_RequiredShapes_FiniteElement_t));" << std::endl;
		cleanup << " pyoomph_tested_free(functable->shapes_required_ResJac); functable->shapes_required_ResJac=PYOOMPH_NULL; " << std::endl;									
		init << " functable->shapes_required_Hessian=(JITFuncSpec_RequiredShapes_FiniteElement_t *)calloc(functable->num_res_jacs,sizeof(JITFuncSpec_RequiredShapes_FiniteElement_t));" << std::endl;
		cleanup << " pyoomph_tested_free(functable->shapes_required_Hessian); functable->shapes_required_Hessian=PYOOMPH_NULL; " << std::endl;											
		init << " functable->HessianVectorProduct=(JITFuncSpec_HessianVectorProduct_FiniteElement *)calloc(functable->num_res_jacs,sizeof(JITFuncSpec_HessianVectorProduct_FiniteElement));" << std::endl;
		cleanup << " pyoomph_tested_free(functable->HessianVectorProduct); functable->HessianVectorProduct=PYOOMPH_NULL; " << std::endl;													
		init << " functable->res_jac_names=(char**)calloc(functable->num_res_jacs,sizeof(char*));" << std::endl;
		init << " functable->missing_residual_assembly=(bool*)calloc(functable->num_res_jacs,sizeof(bool));" << std::endl;
		cleanup << " pyoomph_tested_free(functable->missing_residual_assembly); functable->missing_residual_assembly=PYOOMPH_NULL; " << std::endl;
		init << " functable->has_constant_mass_matrix_for_sure=(bool*)calloc(functable->num_res_jacs,sizeof(bool));" << std::endl;		
		cleanup << " pyoomph_tested_free(functable->has_constant_mass_matrix_for_sure); functable->has_constant_mass_matrix_for_sure=PYOOMPH_NULL; " << std::endl;

		
		
		

		for (unsigned int resiind = 0; resiind < residual.size(); resiind++)
		{
			init << " SET_INTERNAL_FIELD_NAME(functable->res_jac_names," << resiind << ", \"" << residual_names[resiind] << "\" );" << std::endl;
			cleanup << " pyoomph_tested_free(functable->res_jac_names["<<resiind<<"]); functable->res_jac_names["<<resiind<<"]=PYOOMPH_NULL; " << std::endl;		
			if (!residual[resiind].is_zero())
			{
				init << " functable->ResidualAndJacobian_NoHang[" << resiind << "]=&ResidualAndJacobian" << resiind << ";" << std::endl;
				init << " functable->ResidualAndJacobian[" << resiind << "]=&ResidualAndJacobian" << resiind << ";" << std::endl;				
				if (extra_steady_routine[resiind])
				{
					init << " functable->ResidualAndJacobianSteady[" << resiind << "]=&ResidualAndJacobianSteady" << resiind << ";" << std::endl;
				}
				else
				{
					init << " functable->ResidualAndJacobianSteady[" << resiind << "]=&ResidualAndJacobian" << resiind << ";" << std::endl;
				}
				if (generate_hessian)
				{
					if (has_hessian_contribution[resiind])
					{
						init << " functable->HessianVectorProduct[" << resiind << "]=&HessianVectorProduct" << resiind << ";" << std::endl;
					}
				}

				this->write_required_shapes(init, "  ", "ResJac[" + std::to_string(resiind) + "]");
				if (generate_hessian)
					this->write_required_shapes(init, "  ", "Hessian[" + std::to_string(resiind) + "]");
			}
			init << " functable->missing_residual_assembly[" << resiind << "] = " << (ignore_assemble_residuals.count(residual_names[resiind]) ? "true" : "false") << ";" << std::endl;
			init << " functable->has_constant_mass_matrix_for_sure[" << resiind << "] = " << (has_constant_mass_matrix_for_sure[resiind] ? "true" : "false") << ";" << std::endl;	
		}
		cleanup << " pyoomph_tested_free(functable->res_jac_names); functable->res_jac_names=PYOOMPH_NULL; " << std::endl;	

		

		if (generate_hessian)
			init << " functable->hessian_generated=true;" << std::endl
			   << std::endl;
		if (use_shared_shape_buffer_during_multi_assemble)
			init << " functable->use_shared_shape_buffer_during_multi_assemble=true;" << std::endl
			   << std::endl;
		init << std::endl;
		init << " functable->num_Z2_flux_terms = " << this->Z2_fluxes.size() << ";" << std::endl;
		if (this->Z2_fluxes.size())
		{
			init << " functable->GetZ2Fluxes=&GetZ2Fluxes;" << std::endl;
		}
		init << " functable->num_Z2_flux_terms_for_eigen = " << this->Z2_fluxes_for_eigen.size() << ";" << std::endl;
		if (this->Z2_fluxes_for_eigen.size())
		{
			init << " functable->GetZ2FluxesForEigen=&GetZ2FluxesForEigen;" << std::endl;
		}

		if (this->Z2_fluxes.size() || this->Z2_fluxes_for_eigen.size())
		{
			this->write_required_shapes(init, "  ", "Z2Fluxes");
		}

		init << " functable->temporal_error_scales=calloc(" + std::to_string(myfields.size()) + ",sizeof(double)); " << std::endl;
	   cleanup << " pyoomph_tested_free(functable->temporal_error_scales); functable->temporal_error_scales=PYOOMPH_NULL; " << std::endl;							
		// TODO: discontinuous_refinement_exponents
		init << " functable->discontinuous_refinement_exponents=calloc(" << std::to_string(myfields.size()) << ",sizeof(double));" << std::endl;
      cleanup << " pyoomph_tested_free(functable->discontinuous_refinement_exponents); functable->discontinuous_refinement_exponents=PYOOMPH_NULL; " << std::endl;
		index_offset = 0;
		bool has_temporal_estimators = false;
		for (auto &f : myfields)
		{
			if (f->temporal_error_factor != 0.0)
			{
				init << "  functable->temporal_error_scales[" << f->index << "] = " + std::to_string(f->temporal_error_factor) << ";" << std::endl;
				has_temporal_estimators = true;
			}
			if (f->discontinuous_refinement_exponent != 0.0)
			{
				init << "  functable->discontinuous_refinement_exponents[" << f->index << "] = " + std::to_string(f->discontinuous_refinement_exponent) << ";" << std::endl;
			}
		}
		if (has_temporal_estimators)
			init << "  functable->has_temporal_estimators=true;" << std::endl;

		init << " functable->num_ICs=" << IC_names.size() << ";" << std::endl;
		init << " functable->IC_names=(char**)calloc(functable->num_ICs,sizeof(char*));" << std::endl;
		init << " functable->InitialConditionFunc=(JITFuncSpec_InitialCondition_FiniteElement*)calloc(functable->num_ICs,sizeof(JITFuncSpec_InitialCondition_FiniteElement));" << std::endl;

		for (unsigned int i = 0; i < IC_names.size(); i++)
		{
			init << " SET_INTERNAL_FIELD_NAME(functable->IC_names," << i << ", \"" << IC_names[i] << "\" );" << std::endl;
			init << " functable->InitialConditionFunc[" << i << "]=&ElementalInitialConditions" << i << ";" << std::endl;
         cleanup << " pyoomph_tested_free(functable->IC_names[" << i << "]); functable->IC_names[" << i << "]=PYOOMPH_NULL; " << std::endl;
		}
		init << " functable->DirichletConditionFunc=&ElementalDirichletConditions;" << std::endl;
      cleanup << " pyoomph_tested_free(functable->IC_names); functable->IC_names=PYOOMPH_NULL; " << std::endl;			         					
      cleanup << " pyoomph_tested_free(functable->InitialConditionFunc); functable->InitialConditionFunc=PYOOMPH_NULL; " << std::endl;			         			

		std::vector<std::string> dirichlet_set_names;
		std::vector<bool> dirichlet_set_true;
		std::map<int, FiniteElementField*> dirichlet_index_to_field;
		for (auto *f : myfields)
		{
			int myindex = f->index;
			std::string nam = f->get_name();
			if (nam == "lagrangian_x" || nam == "lagrangian_y" || nam == "lagrangian_z")
				continue;
			if (nam == "local_coordinate_1" || nam == "local_coordinate_2" || nam == "local_coordinate_3")
				continue;				
			if (nam == "zeta_coordinate_1" || nam == "zeta_coordinate_2" || nam == "zeta_coordinate_3")
				continue;								
			if (nam == "mesh_x")
				nam = "coordinate_x";
			else if (nam == "mesh_y")
				nam = "coordinate_y";
			else if (nam == "mesh_z")
				nam = "coordinate_z";
			if (nam == "coordinate_x")
				myindex = -1;
			else if (nam == "coordinate_y")
				myindex = -2;
			else if (nam == "coordinate_z")
				myindex = -3;

			myindex += 3;
			if (myindex >= (int)dirichlet_set_names.size())
			{
				dirichlet_set_names.resize(myindex + 1, "");
				dirichlet_set_true.resize(myindex + 1, false);
			}
			// std::cout << "DIRICHLET INFO " << nam << "INDEX " << myindex << " SET " <<  f->Dirichlet_condition_set << std::endl;
			dirichlet_set_names[myindex] = nam;
			dirichlet_set_true[myindex] = f->Dirichlet_condition_set;
			dirichlet_index_to_field[myindex] = f;
		}

		init << " functable->Dirichlet_set_size=" << dirichlet_set_names.size() << ";" << std::endl;
		init << " functable->Dirichlet_set=(bool *)calloc(functable->Dirichlet_set_size,sizeof(bool)); " << std::endl;
		init << " functable->Dirichlet_names=(char**)calloc(functable->Dirichlet_set_size,sizeof(char*));" << std::endl;
		for (unsigned int i = 0; i < dirichlet_set_names.size(); i++)
		{
			init << " SET_INTERNAL_FIELD_NAME(functable->Dirichlet_names," << i << ", \"" << dirichlet_set_names[i] << "\" ); ";
			cleanup << " pyoomph_tested_free(functable->Dirichlet_names["<<i<<"]); functable->Dirichlet_names["<<i<<"]=PYOOMPH_NULL; " << std::endl;
			if (i >= 3)
				init << "// nodal_data index is " << i - 3 << std::endl;
			else
				init << "// nodal_coords index is " << 2 - i << std::endl;
			if (dirichlet_set_true[i])
			{
				init << " functable->Dirichlet_set[" << i << "]=true; //" << dirichlet_set_names[i] << std::endl;
			}
		}
      cleanup << " pyoomph_tested_free(functable->Dirichlet_names); functable->Dirichlet_names=PYOOMPH_NULL; " << std::endl;			         							
      cleanup << " pyoomph_tested_free(functable->Dirichlet_set); functable->Dirichlet_set=PYOOMPH_NULL; " << std::endl;


	  // Build the contribution mapping
	  std::vector<std::string> contribution_names;
	  std::map<std::string, unsigned> contribution_name_to_index;	  
	  std::map<FiniteElementField*, unsigned> contribution_field_to_index;	  
	  std::map<FiniteElementField*,FiniteElementField*> to_where_it_was_defined;  
	  for (auto *f : contributing_fields)
	  {
		FiniteElementField *wheredef = f->get_defined_on_domain_equivalent_field();
		to_where_it_was_defined[f] = wheredef;
		std::string nn=wheredef->get_name();
		if (nn=="mesh_x") nn="coordinate_x";
		else if (nn=="mesh_y") nn="coordinate_y";
		else if (nn=="mesh_z") nn="coordinate_z";
		std::string n=wheredef->get_space()->get_code()->get_full_domain_name()+"/"+nn;
		//std::cout << "CONTRIBUTING FIELD " << f->get_space()->get_code()->get_full_domain_name() << "/" << f->get_name() << " defined on " << n << std::endl;
		//std::cout << "  name already in there " << n << " ? " << (contribution_name_to_index.count(n) ? "YES" : "NO") << std::endl;
		if (contribution_name_to_index.count(n)==0)
		{
			int index=contribution_names.size();
			contribution_names.push_back(n);
			contribution_name_to_index[n]=index;
			contribution_field_to_index[f]=index;
		}	
		else
		{
			contribution_field_to_index[f]=contribution_name_to_index[n];
		}	
		//std::cout << "  setting index to " << contribution_field_to_index[f] << std::endl;
	  }

	  /*
	  for (unsigned int i=0;i<contribution_names.size();i++)
	  {
		  std::cout << "CONTRIBUTION NAME " << contribution_names[i] << " INDEX " << i << " Other indeX " << contribution_name_to_index[contribution_names[i]] << std::endl;
	  }	

	  for (auto &pair : to_where_it_was_defined)
	  {
		  FiniteElementField *f = pair.first;
		  FiniteElementField *wheredef = pair.second;
		  std::string n1=f->get_space()->get_code()->get_full_domain_name()+"/"+f->get_name();
		  std::string n2=wheredef->get_space()->get_code()->get_full_domain_name()+"/"+wheredef->get_name();
		  std::cout << "CONTRIBUTION INFO " << n1 << " defined on " << n2 << " PTRS " << f << " " << wheredef << " for code " << this << std::endl;
	  }
		*/
	  init << " functable->contributes_to_residual=(bool**)calloc(functable->num_res_jacs,sizeof(*functable->contributes_to_residual));" << std::endl;
	  init << " functable->contributes_to_jacobian=(bool***)calloc(functable->num_res_jacs,sizeof(*functable->contributes_to_jacobian));" << std::endl;
	  init << " functable->contribution_entries_size=" << contribution_names.size() << ";" << std::endl;
	  if (contribution_names.size()>0)
	  {
	  	init << " functable->contribution_names=(char**)calloc(functable->contribution_entries_size,sizeof(char*));" << std::endl;
	  	cleanup << " for (unsigned int i=0;i<functable->contribution_entries_size;i++) { pyoomph_tested_free(functable->contribution_names[i]); functable->contribution_names[ i]=PYOOMPH_NULL; }" << std::endl;
	  	cleanup << " pyoomph_tested_free(functable->contribution_names); functable->contribution_names=PYOOMPH_NULL; " << std::endl;
	  	for (unsigned int i=0;i<contribution_names.size();i++)
	  	{
		  init << " SET_INTERNAL_FIELD_NAME(functable->contribution_names," << i << ", \"" << contribution_names[i] << "\" );" << std::endl;
		  
	  	}
		
	  
	  	for (unsigned int resiind = 0; resiind < residual.size(); resiind++)
	  	{
				init << " functable->contributes_to_residual[" << resiind << "]=(bool*)calloc("<< contribution_names.size() <<",sizeof(bool));" << std::endl;				
				init << " functable->contributes_to_jacobian[" << resiind << "]=(bool**)calloc("<< contribution_names.size() <<",sizeof(**functable->contributes_to_jacobian));" << std::endl;				
				init << " for (unsigned int _i=0;_i<"<< contribution_names.size() <<";_i++) { functable->contributes_to_jacobian[" << resiind << "][_i]=(bool*)calloc("<< contribution_names.size() <<",sizeof(bool)); }" << std::endl;				
				std::vector<bool> written_residual_contribution(contribution_names.size(), false);
				std::vector<std::vector<bool>> written_jacobian_contribution(contribution_names.size(), std::vector<bool>(contribution_names.size(), false));
				for (auto &pair1 : to_where_it_was_defined)
				{
					FiniteElementField *f = pair1.first;					
					int i1 = contribution_field_to_index[f];
					//std::cout << "CHECK CONTRIB " << f->get_space()->get_code()->get_full_domain_name() << "/" << f->get_name() << " for residual " << residual_names[resiind] << " contributes to residual? " << f->has_residual_contribution_for_code(this,resiind) << " Corresponding index " << i1 << std::endl;
					
					if ((f->has_residual_contribution_for_code(this,resiind) || pair1.second->has_residual_contribution_for_code(this,resiind)) && !written_residual_contribution[i1])
					{
						init << " functable->contributes_to_residual[" << resiind << "][" << i1 << "]=true; //" << contribution_names[i1] << std::endl;
						written_residual_contribution[i1] = true;
					}
					for (auto &pair2 : to_where_it_was_defined)
					{
						FiniteElementField *f2 = pair2.first;
						int i2 = contribution_field_to_index[f2];
						
						if ((f->has_jacobian_contribution_for_code(this,resiind,f2) || f->has_jacobian_contribution_for_code(this,resiind,pair2.second)) && !written_jacobian_contribution[i1][i2])
						{
							init << " functable->contributes_to_jacobian[" << resiind << "][" << i1 << "][" << i2 << "]=true; //" << contribution_names[i1] << " vs " << contribution_names[i2] << std::endl;
							written_jacobian_contribution[i1][i2] = true;
						}
					}
				}
				cleanup << " for (unsigned int _i=0;_i<functable->contribution_entries_size;_i++) { pyoomph_tested_free(functable->contributes_to_jacobian[" << resiind << "][_i]); functable->contributes_to_jacobian[" << resiind << "][_i]=PYOOMPH_NULL; }" << std::endl;
				cleanup << " pyoomph_tested_free(functable->contributes_to_jacobian[" << resiind << "]); functable->contributes_to_jacobian[" << resiind << "]=PYOOMPH_NULL; " << std::endl;
				cleanup << " pyoomph_tested_free(functable->contributes_to_residual[" << resiind << "]); functable->contributes_to_residual[" << resiind << "]=PYOOMPH_NULL; " << std::endl;
	  
	  	}

	  	cleanup << " pyoomph_tested_free(functable->contributes_to_residual); functable->contributes_to_residual=PYOOMPH_NULL; " << std::endl;
	  	cleanup << " pyoomph_tested_free(functable->contributes_to_jacobian); functable->contributes_to_jacobian=PYOOMPH_NULL; " << std::endl;
	   }

	   init << " functable->dirichlet_field_index_to_global_field_index=(int*)calloc(functable->Dirichlet_set_size,sizeof(int)); // Filling is done in the problem once all fields are defined" << std::endl;
	   init << " for (unsigned int i=0;i<functable->Dirichlet_set_size;i++) { functable->dirichlet_field_index_to_global_field_index[i]=-1; }" << std::endl;
	   cleanup << " pyoomph_tested_free(functable->dirichlet_field_index_to_global_field_index); functable->dirichlet_field_index_to_global_field_index=PYOOMPH_NULL; " << std::endl;

	   std::vector<std::string> defined_fields_on_this_domain;
	   for (auto &f : myfields)
	   {
		if (f->get_defined_on_domain_equivalent_field()==f) // Not transferred from parent
		{
			if (f->get_name() == "lagrangian_x" || f->get_name() == "lagrangian_y" || f->get_name() == "lagrangian_z") continue;
			if (f->get_name() == "local_coordinate_1" || f->get_name() == "local_coordinate_2" || f->get_name() == "local_coordinate_3") continue;
			if (f->get_name() == "zeta_coordinate_1" || f->get_name() == "zeta_coordinate_2" || f->get_name() == "zeta_coordinate_3") continue;
			if ((f->get_name() == "mesh_x" || f->get_name() == "mesh_y" || f->get_name() == "mesh_z")) continue;
			if (!this->coordinates_as_dofs &&  (f->get_name() == "coordinate_x" || f->get_name() == "coordinate_y" || f->get_name() == "coordinate_z")) continue;
			if (dynamic_cast<ExternalD0Space*>(f->get_space())) continue;
			defined_fields_on_this_domain.push_back(f->get_space()->get_code()->get_full_domain_name()+"/"+f->get_name());
		}
	   }
	   if (defined_fields_on_this_domain.size()>0)
	   {
		init << " functable->num_defined_fields_on_this_domain=" << defined_fields_on_this_domain.size() << ";" << std::endl;
		init << " functable->defined_field_names_on_this_domain=(char**)calloc(functable->num_defined_fields_on_this_domain,sizeof(char*));" << std::endl;
		for (unsigned int i=0;i<defined_fields_on_this_domain.size();i++)
		{
			init << " SET_INTERNAL_FIELD_NAME(functable->defined_field_names_on_this_domain," << i << ", \"" << defined_fields_on_this_domain[i] << "\" );" << std::endl;
			cleanup << " pyoomph_tested_free(functable->defined_field_names_on_this_domain[" << i << "]); functable->defined_field_names_on_this_domain[" << i << "]=PYOOMPH_NULL; " << std::endl;		
		}
		cleanup << " pyoomph_tested_free(functable->defined_field_names_on_this_domain); functable->defined_field_names_on_this_domain=PYOOMPH_NULL; " << std::endl;
	 }

		// TODO: Numextdata?
		int numcallbacks = CustomMathExpressionBase::code_map.size();
		if (numcallbacks > 0)
		{
			// Allocate missing diff parents
			//		unsigned index=numcallbacks;
			while (true)
			{
				std::vector<CustomMathExpressionBase *> missing;
				for (auto &cb : CustomMathExpressionBase::code_map)
				{
					//				std::cout << "CB " << cb.first << std::endl;
					auto *diffparent = cb.first->get_diff_parent();
					if (diffparent && (!CustomMathExpressionBase::code_map.count(diffparent)))
					{
						bool found = false;
						for (auto &m : missing)
						{
							if (m == diffparent)
							{
								found = true;
								break;
							}
						}
						if (found)
							break;
						//				std::cout << "ADD DP " << diffparent << std::endl;
						missing.push_back(diffparent);
					}
				}

				if (missing.empty())
					break;
				for (unsigned int i = 0; i < missing.size(); i++)
					CustomMathExpressionBase::code_map.insert(std::make_pair(missing[i], numcallbacks++));
			}
			init << " functable->numcallbacks= " << numcallbacks << ";" << std::endl;
			init << " functable->callback_infos= (JITFuncSpec_Callback_Entry_t*)calloc(" << numcallbacks << ",sizeof(JITFuncSpec_Callback_Entry_t));" << std::endl;
			cb_expressions.resize(numcallbacks, NULL);
			for (auto &cb : CustomMathExpressionBase::code_map)
			{
				//			std::cout << "CENTRY " << cb.first << std::endl;
				int i = cb.second;
				if (i < 0 || i >= numcallbacks)
					throw_runtime_error("Strange problem to locate the callback function");
				cb_expressions[i] = cb.first;
			}

			for (unsigned int i = 0; i < cb_expressions.size(); i++)
			{

				init << "   SET_INTERNAL_NAME(functable->callback_infos[" << i << "].idname, \"" << cb_expressions[i]->get_id_name() << "\");" << std::endl;
				cleanup << " pyoomph_tested_free(functable->callback_infos[" << i << "].idname); functable->callback_infos[" << i << "].idname=PYOOMPH_NULL; " << std::endl;
				init << "   functable->callback_infos[" << i << "].unique_id=" << cb_expressions[i]->get_unique_id() << ";" << std::endl;
				auto *diffparent = cb_expressions[i]->get_diff_parent();
				int diff_pi = -1;
				if (diffparent)
				{
					if (CustomMathExpressionBase::code_map.count(diffparent))
						diff_pi = CustomMathExpressionBase::code_map[diffparent];
					else
						throw_runtime_error("Problem allocating a diff-parent");
				}
				init << "   functable->callback_infos[" << i << "].is_deriv_of=" << diff_pi << ";" << std::endl;
				init << "   functable->callback_infos[" << i << "].deriv_index=" << cb_expressions[i]->get_diff_index() << ";" << std::endl;
			}
			//			cb << "   functable->callback_infos[" <<i<<"].unique_id=" << cb.first->get_unique_id() <<std::endl;
			//			JITFuncSpec_Callback_Entry_t * callback_infos;
			cleanup << " pyoomph_tested_free(functable->callback_infos); functable->callback_infos=PYOOMPH_NULL; " << std::endl;
		}

		int nummultiret = CustomMultiReturnExpressionBase::code_map.size();
		if (nummultiret > 0)
		{
			init << " functable->num_multi_rets= " << nummultiret << ";" << std::endl;
			init << " functable->multi_ret_infos= (JITFuncSpec_MultiRet_Entry_t*)calloc(" << nummultiret << ",sizeof(JITFuncSpec_MultiRet_Entry_t));" << std::endl;
			multi_ret_expressions.resize(nummultiret, NULL);
			for (auto &cb : CustomMultiReturnExpressionBase::code_map)
			{
				//			std::cout << "CENTRY " << cb.first << std::endl;
				int i = cb.second;
				if (i < 0 || i >= nummultiret)
					throw_runtime_error("Strange problem to locate the multi-return function");
				multi_ret_expressions[i] = cb.first;
			}
			for (unsigned int i = 0; i < multi_ret_expressions.size(); i++)
			{

				init << "   SET_INTERNAL_NAME(functable->multi_ret_infos[" << i << "].idname, \"" << multi_ret_expressions[i]->get_id_name() << "\");" << std::endl;
				cleanup << " pyoomph_tested_free(functable->multi_ret_infos[" << i << "].idname); functable->multi_ret_infos[" << i << "].idname=PYOOMPH_NULL; " << std::endl;
				init << "   functable->multi_ret_infos[" << i << "].unique_id=" << multi_ret_expressions[i]->unique_id << ";" << std::endl;
			}
			cleanup << " pyoomph_tested_free(functable->multi_ret_infos); functable->multi_ret_infos=PYOOMPH_NULL; " << std::endl;
		}

		if (integral_expressions.size())
		{
			init << " functable->numintegral_expressions=" << integral_expressions.size() << ";" << std::endl;
			init << " functable->integral_expressions_names=(char **)malloc(sizeof(char*)*functable->numintegral_expressions);" << std::endl;
			unsigned ie_index = 0;
			for (auto &e : integral_expressions)
			{
				init << " SET_INTERNAL_FIELD_NAME(functable->integral_expressions_names," << ie_index << ",\"" << e.first << "\");" << std::endl;
				cleanup << " pyoomph_tested_free(functable->integral_expressions_names["<< ie_index<<"]); functable->integral_expressions_names["<< ie_index<<"]=PYOOMPH_NULL; " << std::endl;
				ie_index++;
			}
			init << " functable->EvalIntegralExpression=&EvalIntegralExpression;" << std::endl;
			cleanup << " pyoomph_tested_free(functable->integral_expressions_names); functable->integral_expressions_names=PYOOMPH_NULL; " << std::endl;

			this->write_required_shapes(init, "  ", "IntegralExprs");
		}

		if (local_expressions.size())
		{
			init << " functable->numlocal_expressions=" << local_expressions.size() << ";" << std::endl;
			init << " functable->local_expressions_names=(char **)malloc(sizeof(char*)*functable->numlocal_expressions);" << std::endl;
			unsigned ie_index = 0;
			for (auto &e : local_expressions)
			{
				init << " SET_INTERNAL_FIELD_NAME(functable->local_expressions_names," << ie_index << ",\"" << e.first << "\");" << std::endl;
				cleanup << " pyoomph_tested_free(functable->local_expressions_names["<<ie_index<<"]); functable->local_expressions_names["<<ie_index<<"]=PYOOMPH_NULL; " << std::endl;
				ie_index++;
			}
			init << " functable->EvalLocalExpression=&EvalLocalExpression;" << std::endl;
			cleanup << " pyoomph_tested_free(functable->local_expressions_names); functable->local_expressions_names=PYOOMPH_NULL; " << std::endl;

			this->write_required_shapes(init, "  ", "LocalExprs");
		}

		if (extremum_expressions.size())
		{
			init << " functable->numextremum_expressions=" << extremum_expressions.size() << ";" << std::endl;
			init << " functable->extremum_expressions_names=(char **)malloc(sizeof(char*)*functable->numextremum_expressions);" << std::endl;
			unsigned ie_index = 0;
			for (auto &e : extremum_expressions)
			{
				init << " SET_INTERNAL_FIELD_NAME(functable->extremum_expressions_names," << ie_index << ",\"" << e.first << "\");" << std::endl;
				cleanup << " pyoomph_tested_free(functable->extremum_expressions_names["<<ie_index<<"]); functable->extremum_expressions_names["<<ie_index<<"]=PYOOMPH_NULL; " << std::endl;
				ie_index++;
			}
			init << " functable->EvalExtremumExpression=&EvalExtremumExpression;" << std::endl;
			cleanup << " pyoomph_tested_free(functable->extremum_expressions_names); functable->extremum_expressions_names=PYOOMPH_NULL; " << std::endl;

			this->write_required_shapes(init, "  ", "ExtremumExprs");
		}

		if (tracer_advection_terms.size())
		{
			init << " functable->numtracer_advections=" << tracer_advection_terms.size() << ";" << std::endl;
			init << " functable->tracer_advection_names=(char **)malloc(sizeof(char*)*functable->numtracer_advections);" << std::endl;
			unsigned ie_index = 0;
			for (auto &e : tracer_advection_terms)
			{
				init << " SET_INTERNAL_FIELD_NAME(functable->tracer_advection_names," << ie_index << ",\"" << e.first << "\");" << std::endl;
				cleanup << " pyoomph_tested_free(functable->tracer_advection_names["<<ie_index<<"]); functable->tracer_advection_names["<<ie_index<<"]=PYOOMPH_NULL; " << std::endl;
				ie_index++;
			}
			init << " functable->EvalTracerAdvection=&EvalTracerAdvection;" << std::endl;
			cleanup << " pyoomph_tested_free(functable->tracer_advection_names); functable->tracer_advection_names=PYOOMPH_NULL; " << std::endl;

			this->write_required_shapes(init, "  ", "TracerAdvection");
		}

		if (!this->integration_order)
			this->integration_order = this->get_default_spatial_integration_order();
		init << " functable->integration_order=" << this->integration_order << ";" << std::endl;
		init << " functable->GeometricJacobian=&GeometricJacobian;" << std::endl;
		init << " functable->JacobianForElementSize=&JacobianForElementSize;" << std::endl;
		if (this->geometric_jac_for_elemsize_has_spatial_deriv)
		{
			init << " functable->JacobianForElementSizeSpatialDerivative=&JacobianForElementSizeSpatialDerivatives;" << std::endl;
			if (this->geometric_jac_for_elemsize_has_second_spatial_deriv)
			{
				init << " functable->JacobianForElementSizeSecondSpatialDerivative=&JacobianForElementSizeSecondSpatialDerivatives;" << std::endl;
			}
		}
		
		init << " SET_INTERNAL_NAME(functable->domain_name,\"" << this->get_domain_name() << "\");" << std::endl;
		cleanup << " pyoomph_tested_free(functable->domain_name); functable->domain_name=PYOOMPH_NULL; " << std::endl;
		init << " functable->clean_up=&clean_up;" << std::endl;
		init << " my_func_table=functable;" << std::endl;
		init << "}" << std::endl;
		
		init << std::endl << std::endl;
		
		os << "static void clean_up(JITFuncSpec_Table_FiniteElement_t *functable)" << std::endl;
		os << "{" << std::endl;
		os << "#ifndef NULL" << std::endl << "#define PYOOMPH_NULL (void *)0" << std::endl << "#else" << std::endl << "#define PYOOMPH_NULL NULL" << std::endl << "#endif" << std::endl;
		os << cleanup.str() ;
		os << "}" << std::endl << std::endl ;
		os << init.str();
	}

	// Sets the exponent used to scale a discontinuous (DG) field's value across mesh refinement
	// levels - used to keep DG jump terms well-scaled as elements are refined/coarsened.
	void FiniteElementCode::set_discontinuous_refinement_exponent(std::string field, double exponent)
	{
		auto *f = this->get_field_by_name(field);
		f->discontinuous_refinement_exponent = exponent;
	}

	// Developer/debugging helper (exposed to Python for interactive use) that expands `inp`,
	// symbolically differentiates it once w.r.t. `dx1` and, if given, a second time w.r.t. `dx2`
	// (using __derive_shapes_by_second_index for the second derivative, exactly as the real Hessian
	// code generator does), printing the symbolic result and its generated C code to stdout after each
	// step. `dx1`/`dx2` accept either a field name or one of the special coordinate/time tokens
	// "__x__"/"__y__"/"__z__"/"__X__"/"__Y__"/"__Z__"/"__t__". Used to manually inspect/verify
	// individual Hessian derivative terms without generating a full element.
	void FiniteElementCode::debug_second_order_Hessian_deriv(GiNaC::ex inp, std::string dx1, std::string dx2)
	{
		auto *old = pyoomph::__current_code;
		pyoomph::__current_code = this;
		std::cout << "ENTER DEBUG SECOND DERIV " << inp << std::endl;
		;
		GiNaC::ex curr = this->expand_placeholders(inp, "Residual");
		std::cout << "EXPANDED " << inp << std::endl;
		GiNaC::print_FEM_options csrc_opts;
		csrc_opts.for_code = this;
		std::cout << "C CODE: ";
		print_simplest_form(curr, std::cout, csrc_opts);
		std::cout << std::endl;
		if (dx1 != "")
		{
			GiNaC::symbol dxs;
			if (dx1 == "__x__")
				dxs = expressions::x;
			else if (dx1 == "__y__")
				dxs = expressions::y;
			else if (dx1 == "__z__")
				dxs = expressions::z;
			else if (dx1 == "__X__")
				dxs = expressions::X;
			else if (dx1 == "__Y__")
				dxs = expressions::Y;
			else if (dx1 == "__Z__")
				dxs = expressions::Z;
			else if (dx1 == "__t__")
				dxs = expressions::t;
			else
			{
				auto *dx = this->get_field_by_name(dx1);
				if (!dx)
					throw_runtime_error("UNKNOWN FIELD " + dx1);
				dxs = dx->get_symbol();
			}
			std::cout << "DERIVATIVE WRT " << dx1 << " : " << dxs << std::endl;
			curr = GiNaC::diff(curr, dxs);
			std::cout << "GIVES " << curr << std::endl;
			std::cout << "C CODE: ";
			print_simplest_form(curr, std::cout, csrc_opts);
			std::cout << std::endl;
		}
		if (dx2 != "")
		{
//			auto *dx = this->get_field_by_name(dx2);
			GiNaC::symbol dxs;
			if (dx2 == "__x__")
				dxs = expressions::x;
			else if (dx2 == "__y__")
				dxs = expressions::y;
			else if (dx2 == "__z__")
				dxs = expressions::z;
			else if (dx2 == "__X__")
				dxs = expressions::X;
			else if (dx2 == "__Y__")
				dxs = expressions::Y;
			else if (dx2 == "__Z__")
				dxs = expressions::Z;
			else if (dx2 == "__t__")
				dxs = expressions::t;
			else
			{
				auto *dx = this->get_field_by_name(dx2);
				if (!dx)
					throw_runtime_error("UNKNOWN FIELD " + dx2);
				dxs = dx->get_symbol();
			}
			std::cout << "DERIVATIVE WRT " << dx2 << " : " << dxs << std::endl;
			__derive_shapes_by_second_index = true;
			curr = GiNaC::diff(curr, dxs);
			__derive_shapes_by_second_index = false;
			std::cout << "GIVES " << curr << std::endl;
			std::cout << "C CODE: ";
			print_simplest_form(curr, std::cout, csrc_opts);
			std::cout << std::endl;
		}
		pyoomph::__current_code = old;
	}

}

namespace GiNaC
{


	/// SORTED PRINTS
	// SortedGiNaC and its subclasses (declared in codegen.hpp) implement a small parallel expression
	// tree, mirrored from a GiNaC::ex, whose sole purpose is to print C code with a *deterministic*
	// term/factor ordering: plain GiNaC printing order can vary depending on internal hashing/pointer
	// values, which would make generated code (and hence compiled-library caching/diffing) spuriously
	// change between runs on otherwise-identical input. Each node type (Numeric/Add/Mul/Pow/Function/
	// Symbol/Struct) defines an add_order()/mul_order() "sort class" priority plus a to_string(); the
	// add_sort_compare()/mul_sort_compare() helpers first order by that priority and break ties by
	// comparing the already-stringified sub-terms lexicographically, so the final printed C expression
	// is fully reproducible. See print_sorted_GiNaC() below for the entry point that (via
	// SortedGiNaC::factory) builds this tree from a plain GiNaC::ex and prints it.
	SortedGiNaC::~SortedGiNaC()
	{
		for (auto ptr : op) {                
			delete ptr;
		}
	}

	bool SortedGiNaC::add_sort_compare(SortedGiNaC * other,std::ostream &os, GiNaC::print_FEM_options &csrc_opts)
	{
		int add_order1 = this->add_order();
		int add_order2 = other->add_order();
		if (add_order1 != add_order2) {
			return add_order1 < add_order2;
		}
		else {
			return this->to_string(os, csrc_opts) < other->to_string(os, csrc_opts);
		}
	}

    bool SortedGiNaC::mul_sort_compare(SortedGiNaC * other,std::ostream &os, GiNaC::print_FEM_options &csrc_opts)
	{
		int mul_order1 = this->mul_order();
		int mul_order2 = other->mul_order();
		if (mul_order1 != mul_order2) {
			return mul_order1 < mul_order2;
		}
		else {
			return this->to_string(os, csrc_opts) < other->to_string(os, csrc_opts);
		}
	}


	std::string SortedGiNaCNumeric::to_string(std::ostream &, GiNaC::print_FEM_options &csrc_opts)
	{
		std::ostringstream ss;
		GiNaC::print_csrc_FEM p(ss, &csrc_opts);
		value.print(p);
		return ss.str();
	}

	std::string SortedGiNaCAdd::to_string(std::ostream &os, GiNaC::print_FEM_options &csrc_opts)
	{
		std::string res="(";
		for (size_t i=0;i<op.size();i++) {
			if (i>0) res+="+";
			res+=op[i]->to_string(os, csrc_opts);
		}
		res+=")";
		return res;
	}
    int SortedGiNaCAdd::add_order() 
    {
        throw std::runtime_error("Not implemented. Add order for add makes no sense for expanded expressions.");
    }

	std::string SortedGiNaCMul::to_string(std::ostream &os, GiNaC::print_FEM_options &csrc_opts) 
	{
		std::string res="(";
		for (size_t i=0;i<op.size();i++) {
			if (i>0) res+="*";
			res+=op[i]->to_string(os, csrc_opts);
		}
		res+=")";
		return res;
	}
    int SortedGiNaCMul::mul_order() 
    {
            throw_runtime_error("Not implemented. Mul order for mul makes no sense for expanded expressions.");
    }

	std::string SortedGiNaCPow::to_string(std::ostream &os, GiNaC::print_FEM_options &csrc_opts) 
	{
		std::string res="pow(";
		res+=op[0]->to_string(os, csrc_opts);
		res+=",";
		res+=op[1]->to_string(os, csrc_opts);
		res+=")";
		return res;
	}

	std::string SortedGiNaCFunction::to_string(std::ostream &os, GiNaC::print_FEM_options &csrc_opts)
	{
		std::string res=fname+"(";
		for (size_t i=0;i<op.size();i++) {
			if (i>0) res+=",";
			res+=op[i]->to_string(os, csrc_opts);
		}
		res+=")";
		return res;
	}

	std::string SortedGiNaCStruct::to_string(std::ostream &, GiNaC::print_FEM_options &csrc_opts) 
	{
		std::ostringstream ss;
		GiNaC::print_csrc_FEM p(ss, &csrc_opts);
		contents.print(p);
		return ss.str();
	}
	int SortedGiNaCStruct::mul_order() 
	{
		return 6; 
	}

	// Builds a SortedGiNaC tree from GiNaC expression `e` (after expand()): numbers/constants become
	// SortedGiNaCNumeric, sums/products become SortedGiNaCAdd/Mul with their operands recursively
	// built and then *sorted* (via add_sort_compare/mul_sort_compare) so the operand order is
	// deterministic, powers/functions/symbols map to their respective node types, and any of this
	// codebase's custom GiNaC structures (test functions, shape expansions, subexpressions, spatial-
	// integral/normal symbols, fake-exponential-mode markers, global-parameter wrappers) are treated
	// as opaque leaves (SortedGiNaCStruct) printed via the normal (non-sorted) print_csrc_FEM path,
	// since their internal structure is not a plain algebraic sum/product to be reordered.
	SortedGiNaC * SortedGiNaC::factory(const ex & e,std::ostream &os, GiNaC::print_FEM_options &csrc_opts)
    {
		ex expa=e.expand();
		if (is_a<numeric>(expa)) 
		{
			/*if (ex_to<numeric>(expa).is_crational() && !is_zero(ex_to<numeric>(expa).denom())-1) 
			{
				return new SortedGiNaCNumeric(ex_to<numeric>(expa).to_int64());
			}*/
			return new SortedGiNaCNumeric(ex_to<numeric>(expa));
		}
		else if (is_a<constant>(expa)) {
			return new SortedGiNaCNumeric(ex_to<numeric>(ex_to<constant>(expa).evalf()));
		}
		else if (is_a<add>(expa)) {
			std::vector<SortedGiNaC*> ops;
			for (size_t i=0;i<expa.nops();i++) {
				ops.push_back(SortedGiNaC::factory(expa.op(i), os, csrc_opts));
			}
			std::sort(ops.begin(), ops.end(),
						[&os,&csrc_opts](SortedGiNaC * a, SortedGiNaC * b) {
							return a->add_sort_compare(b,os, csrc_opts);
						});
			return new SortedGiNaCAdd(ops);
		}
		else if (is_a<mul>(expa)) {
			std::vector<SortedGiNaC*> ops;
			for (size_t i=0;i<expa.nops();i++) {
				ops.push_back(SortedGiNaC::factory(expa.op(i), os, csrc_opts));
			}
			std::sort(ops.begin(), ops.end(),                          
						[&os,&csrc_opts](SortedGiNaC * a, SortedGiNaC * b) {
							return a->mul_sort_compare(b,os, csrc_opts);
						});
			return new SortedGiNaCMul(ops);
		}
		else if (is_a<power>(expa)) {
			SortedGiNaC * base=SortedGiNaC::factory(expa.op(0), os, csrc_opts);
			SortedGiNaC * exp=SortedGiNaC::factory(expa.op(1), os, csrc_opts);
			GINAC_ASSERT(expa.nops()==2);
			return new SortedGiNaCPow(base, exp);
		}
		else if (is_a<function>(expa)) {
			std::vector<SortedGiNaC*> ops;
			for (size_t i=0;i<expa.nops();i++) {
				ops.push_back(SortedGiNaC::factory(expa.op(i), os, csrc_opts));
			}
			return new SortedGiNaCFunction(ex_to<function>(expa).get_name(), ops);
		}
		else if (is_a<symbol>(expa)) {
			return new SortedGiNaCSymbol(ex_to<symbol>(expa).get_name());
		}
		else if (is_a<GiNaCTestFunction>(expa) || is_a<GiNaCShapeExpansion>(expa) || is_a<GiNaCSubExpression>(expa) || is_a<GiNaCSpatialIntegralSymbol>(expa) || is_a<GiNaCNormalSymbol>(expa) || is_a<GiNaCFakeExponentialMode>(expa) || is_a<GiNaCGlobalParameterWrapper>(expa)) 
		{
			return new SortedGiNaCStruct(expa);
		}
		else if (expa.is_zero())
		{
			return new SortedGiNaCStruct(0+expa);
		}
		else {
			std::ostringstream err;
			err << "Non implemented type in SortedGiNaC factory, got: " << expa;
			throw_runtime_error(err.str());
		}
    }

	// Entry point for the deterministic ("sorted") C-code printing mode (ccode_expression_mode ==
	// "deterministic"): builds the SortedGiNaC tree for `e` and writes its canonically-ordered string
	// form to `os`.
	std::ostream &  print_sorted_GiNaC(ex  e,std::ostream &os, GiNaC::print_FEM_options &csrc_opts)
    {
        SortedGiNaC * root=SortedGiNaC::factory(e, os, csrc_opts);
        os << root->to_string(os, csrc_opts);
        delete root;
        return os; 
    }

	// print_csrc_FEM/print_latex_FEM are GiNaC print_context subclasses that carry a pointer to
	// print_FEM_options (`FEM_opts`), which in turn carries the FiniteElementCode currently being
	// printed for; this is how the custom structures' print()/derivative() implementations below know
	// which code's subexpression list/resolve_multi_return_call/etc. to use, since GiNaC's print
	// machinery itself has no notion of "current element being generated".
	print_csrc_FEM::print_csrc_FEM() : GiNaC::print_csrc_double(std::cout)
	{
	}
	print_csrc_FEM::print_csrc_FEM(std::ostream &os, print_FEM_options *fem_opts, unsigned opt) : GiNaC::print_csrc_double(os, opt), FEM_opts(fem_opts)
	{
	}

	print_latex_FEM::print_latex_FEM() : GiNaC::print_latex(std::cout)
	{
	}

	print_latex_FEM::print_latex_FEM(std::ostream &os, print_FEM_options *fem_opts, unsigned opt) : GiNaC::print_latex(os, opt), FEM_opts(fem_opts)
	{
	}

	// Prints a GiNaCSubExpression: in C-code printing mode, resolves it back to the FiniteElementCode's
	// registered subexpression list and prints the corresponding pre-computed C variable name
	// (subexpr_N) instead of re-expanding the (potentially large) underlying expression - this is the
	// actual mechanism by which the CSE optimization performed in write_code_subexpressions() takes
	// effect at print time. In any other (e.g. debug/plain) print mode, falls back to printing the
	// raw wrapped expression.
	template <>
	void GiNaCSubExpression::print(const print_context &c, unsigned) const
	{
		if (GiNaC::is_a<print_csrc_FEM>(c))
		{
			const auto &femprint = dynamic_cast<const print_csrc_FEM &>(c);
			if (femprint.FEM_opts->for_code)
			{
				auto *se = femprint.FEM_opts->for_code->resolve_subexpression(get_struct().expr);
				if (!se)
					throw_runtime_error("Cannot resolve subexpressions");
				c.s << se->get_cvar();
			}
			else
			{
				throw_runtime_error("No code supplied");
			}
		}
		else
			c.s << "<SUBEXPRESSION: " << get_struct().expr << ">";
	}

	// Implements GiNaC's chain rule for a GiNaCSubExpression node w.r.t. raw symbol `s` (normally a
	// field's underlying GiNaC::symbol). This is the counterpart to the CSE derivative pre-computation
	// done in write_code_subexpressions(): rather than re-differentiating the (potentially large)
	// subexpression body every time it appears, it looks up which of the subexpression's required
	// fields (`se->req_fields`) actually correspond to `s`, and returns (symbolically) the *product
	// rule term* d(subexpr)/d(field) * d(field)/d(s) where the first factor is represented either by
	// re-differentiating on the spot into a fresh nested subexpression (only during the very first,
	// "outer", index of a Hessian double-differentiation, __in_hessian && !__derive_shapes_by_second_index)
	// or, in the common (non-Hessian, or Hessian-inner-index) case, by a symbolic placeholder symbol
	// named "d_<cvar>_d_<field>" that refers to the C variable already emitted by
	// write_code_subexpressions() - i.e. the actual numeric derivative value is computed once in C,
	// not re-derived symbolically at every use site. The second factor is represented by a "derived"
	// ShapeExpansion of that field (GiNaCShapeExpansion's own derivative() then reduces this to 0 or 1
	// depending on which basis function was differentiated - see DerivedShapeExpansionsToUnity).
	// Additionally, if this code has moving-mesh coordinates as DoFs and `s` happens to be one of the
	// (Eulerian/bulk/opposite-interface) position symbols, the subexpression must also be
	// differentiated directly (bypassing the cached-C-derivative mechanism above) since geometric
	// quantities such as d(dpsi/dx * u)/dX^l_j genuinely depend on the per-node/per-direction loop
	// index and cannot be captured by a single precomputed scalar derivative variable.
	template <>
	GiNaC::ex GiNaCSubExpression::derivative(const GiNaC::symbol &s) const
	{
		auto *se = get_struct().code->resolve_subexpression(get_struct().expr);
		if (!se)
			throw_runtime_error("Cannot resolve subexpressions");
		GiNaC::ex res = 0;
		bool found = false;
		for (auto &shape_exp : se->req_fields)
		{
			if (shape_exp.field->get_symbol() == s)
			{
				if (shape_exp.time_history_index != 0)
					continue; // Only with respect to the actual time
				if (pyoomph::__in_hessian && !pyoomph::__derive_shapes_by_second_index)
				{
					GiNaC::ex inner = GiNaC::diff(get_struct().expr, s);
					GiNaC::ex newse = (*pyoomph::__SE_to_struct_hessian)(pyoomph::expressions::subexpression(inner));
					auto sexp = pyoomph::ShapeExpansion(shape_exp.field, shape_exp.dt_order, shape_exp.basis, shape_exp.dt_scheme, true);
					// if (pyoomph::__derive_shapes_by_second_index) sexp.is_derived_other_index=true;
					res += newse * GiNaCShapeExpansion(sexp);
					found = true;
				}
				else
				{
					std::string wrto = shape_exp.get_spatial_interpolation_name(get_struct().code);
					std::ostringstream derivname;
					derivname << "d_" << se->get_cvar() << "_d_" << wrto;
					if (!pyoomph::__field_name_cache.count(derivname.str()))
						pyoomph::__field_name_cache.insert(std::make_pair(derivname.str(), GiNaC::potential_real_symbol(derivname.str())));
					auto sexp = pyoomph::ShapeExpansion(shape_exp.field, shape_exp.dt_order, shape_exp.basis, shape_exp.dt_scheme, true);
					if (pyoomph::__derive_shapes_by_second_index)
						sexp.is_derived_other_index = true;
					res += pyoomph::__field_name_cache[derivname.str()] * GiNaCShapeExpansion(sexp);
					found = true;
				}
			}
		}
		// HOWEVER, if we have moving nodes, there is no way but to derive it by hand here, we cannot put it into the subexpression, since a lot of things, like d_(dpsidx*u)_dX^li depend on l_shape in the jacobian loop
		if (get_struct().code->coordinates_as_dofs && !pyoomph::ignore_nodal_position_derivatives_for_pitchfork_symmetry())
		{
			bool is_coordinate = false;
			for (auto d : std::vector<std::string>{"x", "y", "z"})
			{
				auto *f = get_struct().code->get_field_by_name("coordinate_" + d);
				if (f)
				{
					if (f->get_symbol() == s)
					{
						is_coordinate = true;
						break;
					}
					if (get_struct().code->get_bulk_element())
					{
						f = get_struct().code->get_bulk_element()->get_field_by_name("coordinate_" + d);
						if (f && f->get_symbol() == s)
						{
							is_coordinate = true;
							break;
						}
						if (get_struct().code->get_bulk_element()->get_bulk_element())
						{
							f = get_struct().code->get_bulk_element()->get_bulk_element()->get_field_by_name("coordinate_" + d);
							if (f && f->get_symbol() == s)
							{
								is_coordinate = true;
								break;
							}
						}
					}
					if (get_struct().code->get_opposite_interface_code())
					{
						f = get_struct().code->get_opposite_interface_code()->get_field_by_name("coordinate_" + d);
						if (f && f->get_symbol() == s)
						{
							is_coordinate = true;
							break;
						}
						if (get_struct().code->get_opposite_interface_code()->get_bulk_element())
						{
							f = get_struct().code->get_opposite_interface_code()->get_bulk_element()->get_field_by_name("coordinate_" + d);
							if (f && f->get_symbol() == s)
							{
								is_coordinate = true;
								break;
							}
						}
					}
				}
			}
			if (is_coordinate)
			{
				GiNaC::ex deriv = GiNaC::diff(get_struct().expr, s);
				//				std::cout << "DERIV OF " << get_struct().expr << " WRTO " << s << " IS " << deriv << std::endl;
				if (!deriv.is_zero())
				{
					if (found)
					{
						std::ostringstream oss;
						oss << "subexpression derivative wrto " << s << " is non-zero, but we already have a contribution before..." << std::endl;
						oss << "DERIV IS (should be 0): " << deriv << std::endl;
						oss << "EXPRESSION IS " << get_struct().expr << std::endl;
						//throw_runtime_error(oss.str());
						return deriv;
					}
					else
					{
						//		    std::cout << "DERIVED SUBEXPRESSIONS "
					}
					return deriv;
				}
			}
		}

		// throw_runtime_error("TODO");
		return res;
	}

	// Prints a GiNaCMultiRetCallback: resolves it to its slot in the enclosing FiniteElementCode's
	// multi_return_calls list and prints either "multi_ret_<index>[<retindex>]" (the callback's
	// plain return value) or, if this node represents a derivative w.r.t. one of the callback's
	// arguments (derived_by_arg>=0), "dmulti_ret_<index>[nargs*retindex+derived_by_arg]" - matching
	// the flattened derivative-matrix layout filled in by write_code_multi_ret_call().
	template <>
	void GiNaCMultiRetCallback::print(const print_context &c, unsigned) const
	{
		const auto &sp = get_struct();

		if (GiNaC::is_a<print_csrc_FEM>(c))
		{
			const auto &femprint = dynamic_cast<const print_csrc_FEM &>(c);
			if (femprint.FEM_opts->for_code)
			{
				int index = femprint.FEM_opts->for_code->resolve_multi_return_call(sp.invok);
				if (index < 0)
				{
					std::ostringstream oss;
					oss << std::endl
						<< "When looking for:" << std::endl
						<< sp.invok << std::endl
						<< "Present:" << std::endl;
					for (unsigned int _i = 0; _i < femprint.FEM_opts->for_code->multi_return_calls.size(); _i++)
						oss << femprint.FEM_opts->for_code->multi_return_calls[_i] << std::endl;
					throw_runtime_error("Cannot resolve multi_return_call" + oss.str());
				}

				if (sp.derived_by_arg >= 0)
				{
					//				  int nret=GiNaC::ex_to<GiNaC::numeric>(sp.invok.op(2)).to_int();
					int nargs = GiNaC::ex_to<GiNaC::lst>(sp.invok.op(1)).nops();
					//				  c.s << "dmulti_ret_"<<index<<"["<<sp.retindex<<"+"<<nret<<"*"<< sp.derived_by_arg<<"]";
					c.s << "dmulti_ret_" << index << "[" << nargs << "*" << sp.retindex << "+" << sp.derived_by_arg << "]";
				}
				else
				{
					c.s << "multi_ret_" << index << "[" << sp.retindex << "]";
				}
			}
			else
			{
				throw_runtime_error("No code supplied");
			}
		}
		else
		{
			if (sp.derived_by_arg < 0)
			{
				c.s << "<MULTIRET_CB: " << sp.invok << " at index " << sp.retindex << ">";
			}
			else
			{
				c.s << "<DERIVED MULTIRET_CB: " << sp.invok << " at index " << sp.retindex << " wrt. " << sp.derived_by_arg << ">";
			}
		}
	}

	// Chain rule for a multi-return callback's return value: only first derivatives are supported
	// (differentiating an already-derived node, derived_by_arg>=0, raises an error - except w.r.t.
	// the mass-matrix marker, which is trivially zero). For a plain (non-derived) node, differentiates
	// every argument expression w.r.t. `s` and, for each nonzero argument derivative, asks the
	// underlying callback whether it can supply a closed-form symbolic derivative
	// (_get_symbolic_derivative); if not, falls back to a "derived by argument i" GiNaCMultiRetCallback
	// node (whose value is the numerically-computed Jacobian entry dmulti_ret_.../nargs*retindex+i,
	// filled in at runtime by the invoked C/Python callback itself) multiplied by the chain-rule factor.
	template <>
	GiNaC::ex GiNaCMultiRetCallback::derivative(const GiNaC::symbol &s) const
	{
		const auto &sp = get_struct();
		if (sp.derived_by_arg >= 0)
		{
			if (s == pyoomph::expressions::__partial_t_mass_matrix)
			{
				return 0;
			}
			std::ostringstream oss;
			oss << std::endl
				<< "happes when deriving " << (*this) << std::endl
				<< " by " << s;
			throw_runtime_error("Multi-Return Callbacks can only be derived to the first order at the moment!" + oss.str());
		}
		else
		{
			GiNaC::ex args = sp.invok.op(1);
			GiNaC::ex res = 0;
			pyoomph::CustomMultiReturnExpressionBase *func = GiNaC::ex_to<GiNaC::GiNaCCustomMultiReturnExpressionWrapper>(sp.invok.op(0)).get_struct().cme;
			std::vector<GiNaC::ex> argvect;
			for (unsigned int i = 0; i < args.nops(); i++)
			{
				argvect.push_back(args.op(i));
			}
			for (unsigned int i = 0; i < args.nops(); i++)
			{
				GiNaC::ex inner = GiNaC::diff(args.op(i), s);
				if (!GiNaC::is_zero(inner))
				{
					std::pair<bool, GiNaC::ex> symderiv = func->_get_symbolic_derivative(argvect, sp.retindex, i);
					if (symderiv.first)
					{
						res += inner * symderiv.second;
					}
					else
					{
						res += inner * GiNaCMultiRetCallback(pyoomph::MultiRetCallback(sp.code, sp.invok, sp.retindex, i));
					}
				}
			}
			return res;
		}
	}

	// Custom substitution: applies `m` to the callback's invocation arguments; if that substitution
	// causes GiNaC to fully evaluate the invocation down to a concrete list of numeric results (`lst`,
	// e.g. all arguments became numbers), directly returns the requested return-value component
	// instead of rebuilding another (now meaningless) GiNaCMultiRetCallback wrapper.
	template <>
	GiNaC::ex GiNaCMultiRetCallback::subs(const GiNaC::exmap &m, unsigned options) const
	{
		const auto &sp = get_struct();
		GiNaC::ex invok = sp.invok.subs(m, options);
		if (GiNaC::is_a<GiNaC::lst>(invok)) // Substition causes the numerical eval
		{
			if (sp.derived_by_arg < 0)
			{
				return invok.op(sp.retindex);
			}
			else
			{
				throw_runtime_error("Should not get here");
			}
		}
		else
		{
			return GiNaCMultiRetCallback(pyoomph::MultiRetCallback(sp.code, invok, sp.retindex, sp.derived_by_arg));
		}
	}

	// NodalDeltaSymbol has no configurable state (it always represents the same Kronecker-delta
	// nodal-contribution marker), so print() just emits the fixed C identifier "nodal_delta_sym", and
	// derivative() is always 0 (it does not depend on any field/coordinate symbol).
	template <>
	void GiNaCNodalDeltaSymbol::print(const print_context &c, unsigned) const
	{
		if (GiNaC::is_a<print_csrc_FEM>(c))
		{
			// const auto &femprint = dynamic_cast<const print_csrc_FEM &>(c);
			/*			if (femprint.FEM_opts->for_code)
						{*/
			c.s << "nodal_delta_sym";
			//			}
		}
		else
		{
			c.s << "<Nodal Delta>";
		}
	}

	template <>
	GiNaC::ex GiNaCNodalDeltaSymbol::derivative(const GiNaC::symbol &) const
	{
		return 0;
	}

	// Prints a SpatialIntegralSymbol (dx/dX and its derivatives-w.r.t.-nodal-coordinates variants) as
	// the matching runtime C expression: the plain "dx_unity"/"dX"/"dx" local variables declared by
	// write_generic_spatial_integration_header() for the untagged cases, or, for a symbol tagged as
	// "derived" (i.e. representing d(dx)/dX^dir at the current l_shape[2] index), the corresponding
	// precomputed shapeinfo->int_pt_weights_d_coords[...]/..._d2_coords[...] array entry. A nonzero
	// history_step instead selects the corresponding stored-history integration weight. In LaTeX
	// printing mode, delegates to the (Python-implemented) LaTeXPrinter via a descriptive info-map
	// instead. Falls back to a human-readable "<DX ...>" placeholder in any other print mode (e.g.
	// plain debug printing).
	template <>
	void GiNaCSpatialIntegralSymbol::print(const print_context &c, unsigned) const
	{
		if (GiNaC::is_a<print_csrc_FEM>(c))
		{
			const auto &femprint = dynamic_cast<const print_csrc_FEM &>(c);
			if (femprint.FEM_opts->for_code)
			{
				if (get_struct().simple_unity_integral)
				{
					c.s << "dx_unity";
					return;
				}
				else if (get_struct().is_lagrangian())
					c.s << "dX";
				else if (!get_struct().is_derived())
				{
					if (get_struct().history_step==0) c.s << "dx";				
					else c.s << "shapeinfo->int_pt_weight[" << get_struct().history_step << "]";
				}
				else if (!get_struct().is_derived2())
				{
					c.s << "shapeinfo->int_pt_weights_d_coords[" << get_struct().get_derived_direction() << "][" << (get_struct().is_derived_by_lshape2() ? "l_shape2" : "l_shape") << "]"; // TODO: Other spaces, e.g. bulk
				}
				else
				{
					c.s << "shapeinfo->int_pt_weights_d2_coords[" << get_struct().get_derived_direction() << "][" << get_struct().get_derived_direction2() << "][l_shape][l_shape2]";
				}
				return;
			}
		}
		if (GiNaC::is_a<print_latex_FEM>(c))
		{
			const auto &femprint = dynamic_cast<const print_latex_FEM &>(c);
			if (femprint.FEM_opts->for_code && femprint.FEM_opts->for_code->latex_printer)
			{
				std::map<std::string, std::string> texinfo;
				texinfo["typ"] = "spatial_integral_symbol";
				texinfo["lagrangian"] = get_struct().is_lagrangian() ? "true" : "false";
				texinfo["derived_in_direction"] = get_struct().is_derived() ? std::to_string(get_struct().get_derived_direction()) : "none";
				texinfo["derived_in_direction2"] = get_struct().is_derived2() ? std::to_string(get_struct().get_derived_direction2()) : "none";
				texinfo["derived_to_lshape2"] = get_struct().is_derived_by_lshape2() ? "true" : "false";
				texinfo["simple_unity_integral"] = get_struct().simple_unity_integral ? "true" : "false";
				texinfo["history_step"] = std::to_string(get_struct().history_step);
				c.s << femprint.FEM_opts->for_code->latex_printer->_get_LaTeX_expression(texinfo, femprint.FEM_opts->for_code);
				return;
			}
		}
		std::string modestr=(get_struct().expansion_mode ? " | MODE "+std::to_string(get_struct().expansion_mode) +" " : "");
		if (get_struct().simple_unity_integral)
		{
			c.s << "<DX"+modestr+"Unity>";
		}
		else
		if (get_struct().is_lagrangian())
		{
			c.s << "<DX "+modestr+"Lagrangian>";
		}
		else
		{
			if (get_struct().is_derived())
			{

				c.s << "<DX "+modestr;

				c.s << " derived by position direction " << get_struct().get_derived_direction();
				if (get_struct().is_derived2())
				{
					c.s << " and " << get_struct().get_derived_direction2();
				}
				else
				{
					if (get_struct().is_derived_by_lshape2())
						c.s << " in second shape index for Hessian";
				}
				c.s << ">";
			}
			else
			{
				if (get_struct().history_step > 0)
				{
					c.s << "<DX"+modestr<< "|History step " << get_struct().history_step << "|" << modestr <<">";
				}
				else
				c.s << "<DX"+modestr+">";
			}
		}
	}
	// Implements d(dx)/ds for the Eulerian integration-measure symbol: the Lagrangian dX, the plain
	// "unity" measure, and history-tagged (history_step>0) variants never depend on the moving mesh
	// (their derivative is 0 by construction/convention). Otherwise, if the mesh coordinates are
	// active DoFs and `s` is one of this domain's raw coordinate_{x,y,z} symbols, returns the matching
	// pre-built "derived" (or, if already once-derived, "twice-derived") SpatialIntegralSymbol from
	// FiniteElementCode::get_dx_derived[2](), preserving the expansion_mode tag; any other symbol -
	// or a request filtered out by __derive_only_by_expansion_mode / the no_jacobian/no_hessian tags
	// (checked against which Hessian index is currently being derived, __derive_shapes_by_second_index)
	// - yields 0, since dx genuinely only depends on the moving nodal positions.
	template <>
	GiNaC::ex GiNaCSpatialIntegralSymbol::derivative(const GiNaC::symbol &s) const
	{
		if (get_struct().is_lagrangian() || get_struct().simple_unity_integral || get_struct().history_step>0)
			return 0;
		
		if (pyoomph::__derive_only_by_expansion_mode &&  get_struct().expansion_mode!=*pyoomph::__derive_only_by_expansion_mode)
			return 0;
		pyoomph::FiniteElementCode *code = (pyoomph::FiniteElementCode *)(get_struct().get_code()); // Cast aways the constness
		pyoomph::FiniteElementField *testf;
		if (!code->coordinates_as_dofs || pyoomph::ignore_nodal_position_derivatives_for_pitchfork_symmetry())
			return 0;

		if ((get_struct().no_jacobian && (!pyoomph::__derive_shapes_by_second_index)) || (get_struct().no_hessian && pyoomph::__derive_shapes_by_second_index))
			return 0;

		// TODO: Other spaces, e.g. bulk
		if (!get_struct().is_derived())
		{
			testf = code->get_field_by_name("coordinate_x");
			if (testf && s == testf->get_symbol())
			{
				pyoomph::SpatialIntegralSymbol sder = code->get_dx_derived(0);
				sder.expansion_mode = get_struct().expansion_mode;
				return 0 + GiNaC::GiNaCSpatialIntegralSymbol(sder);
			}
			testf = code->get_field_by_name("coordinate_y");
			if (testf && s == testf->get_symbol())
			{
				pyoomph::SpatialIntegralSymbol sder = code->get_dx_derived(1);
				sder.expansion_mode = get_struct().expansion_mode;
				return 0 + GiNaC::GiNaCSpatialIntegralSymbol(sder);
			}
			testf = code->get_field_by_name("coordinate_z");
			if (testf && s == testf->get_symbol())
			{
				pyoomph::SpatialIntegralSymbol sder = code->get_dx_derived(2);
				sder.expansion_mode = get_struct().expansion_mode;
				return 0 + GiNaC::GiNaCSpatialIntegralSymbol(sder);
			}
		}
		else if (!get_struct().is_derived2())
		{
			int dir1 = get_struct().get_derived_direction();
			testf = code->get_field_by_name("coordinate_x");
			if (testf && s == testf->get_symbol())
			{
				pyoomph::SpatialIntegralSymbol sder = code->get_dx_derived2(dir1, 0);
				sder.expansion_mode = get_struct().expansion_mode;
				return 0 + GiNaC::GiNaCSpatialIntegralSymbol(sder);
			}
			testf = code->get_field_by_name("coordinate_y");
			if (testf && s == testf->get_symbol())
			{
				pyoomph::SpatialIntegralSymbol sder = code->get_dx_derived2(dir1, 1);
				sder.expansion_mode = get_struct().expansion_mode;
				return 0 + GiNaC::GiNaCSpatialIntegralSymbol(sder);
			}
			testf = code->get_field_by_name("coordinate_z");
			if (testf && s == testf->get_symbol())
			{
				pyoomph::SpatialIntegralSymbol sder = code->get_dx_derived2(dir1, 2);
				sder.expansion_mode = get_struct().expansion_mode;
				return 0 + GiNaC::GiNaCSpatialIntegralSymbol(sder);
			}
		}
		return 0;
	}

	// Prints an ElementSizeSymbol analogously to GiNaCSpatialIntegralSymbol::print above, selecting
	// between the plain elemsize_Eulerian/elemsize_Lagrangian[_cartesian] runtime fields and, for
	// "derived" variants, the corresponding elemsize[_Cart]_d[2]_coords[...] array entries that hold
	// the (first or second) derivative of the element size w.r.t. nodal coordinates.
	template <>
	void GiNaCElementSizeSymbol::print(const print_context &c, unsigned) const
	{
		if (GiNaC::is_a<print_csrc_FEM>(c))
		{
			const auto &femprint = dynamic_cast<const print_csrc_FEM &>(c);
			if (femprint.FEM_opts->for_code)
			{
				pyoomph::FiniteElementCode *code = (pyoomph::FiniteElementCode *)(get_struct().get_code());						   // Cast aways the constness
				std::string shapeinfo_str = femprint.FEM_opts->for_code->get_shape_info_str(code->get_my_position_space()) + "->"; // "shapeinfo->"; //XXX TODO Other codes!
				if (get_struct().is_lagrangian())
				{
					if (get_struct().is_with_coordsys())
					{
						c.s << shapeinfo_str << "elemsize_Lagrangian";
					}
					else
					{
						c.s << shapeinfo_str << "elemsize_Lagrangian_cartesian";
					}
				}
				else if (!get_struct().is_derived())
				{
					if (get_struct().is_with_coordsys())
					{
						c.s << shapeinfo_str << "elemsize_Eulerian";
					}
					else
					{
						c.s << shapeinfo_str << "elemsize_Eulerian_cartesian";
					}
				}
				else if (!get_struct().is_derived2())
				{
					c.s << shapeinfo_str << "elemsize" << (get_struct().is_with_coordsys() ? "" : "_Cart") << "_d_coords[" << get_struct().get_derived_direction() << "][" << (get_struct().is_derived_by_lshape2() ? "l_shape2" : "l_shape") << "]"; // TODO: Other spaces, e.g. bulk
				}
				else
				{
					c.s << shapeinfo_str << "elemsize" << (get_struct().is_with_coordsys() ? "" : "_Cart") << "_d2_coords[" << get_struct().get_derived_direction() << "][" << get_struct().get_derived_direction2() << "][l_shape][l_shape2]";
				}
				return;
			}
		}
		if (GiNaC::is_a<print_latex_FEM>(c))
		{
			const auto &femprint = dynamic_cast<const print_latex_FEM &>(c);
			if (femprint.FEM_opts->for_code && femprint.FEM_opts->for_code->latex_printer)
			{
				std::map<std::string, std::string> texinfo;
				texinfo["typ"] = "element_size_symbol";
				texinfo["lagrangian"] = get_struct().is_lagrangian() ? "true" : "false";
				texinfo["with_coordsys"] = get_struct().is_with_coordsys() ? "true" : "false";
				texinfo["derived_in_direction"] = get_struct().is_derived() ? std::to_string(get_struct().get_derived_direction()) : "none";
				texinfo["derived_in_direction2"] = get_struct().is_derived2() ? std::to_string(get_struct().get_derived_direction2()) : "none";
				texinfo["derived_to_lshape2"] = get_struct().is_derived_by_lshape2() ? "true" : "false";
				c.s << femprint.FEM_opts->for_code->latex_printer->_get_LaTeX_expression(texinfo, femprint.FEM_opts->for_code);
				return;
			}
		}
		if (get_struct().is_lagrangian())
		{
			c.s << "<Elemsize Lagrangian " << (get_struct().is_with_coordsys() ? "with coordsys" : "cartesian") << ">";
		}
		else
		{
			if (get_struct().is_derived())
			{

				c.s << "<Elemsize Eulerian " << (get_struct().is_with_coordsys() ? "with coordsys" : "cartesian");

				c.s << " derived by position direction " << get_struct().get_derived_direction();
				if (get_struct().is_derived2())
				{
					c.s << " and " << get_struct().get_derived_direction2();
				}
				else if (get_struct().is_derived_by_lshape2())
				{
					c.s << " with respect to second shape index";
				}
				c.s << ">";
			}
			else
			{
				c.s << "<Elemsize Eulerian>";
			}
		}
	}
	// Mirrors GiNaCSpatialIntegralSymbol::derivative above for element-size symbols: Lagrangian
	// element size never depends on the moving mesh; otherwise, differentiating w.r.t. one of this
	// domain's coordinate_{x,y,z} symbols (when coordinates_as_dofs) yields the matching pre-built
	// "derived" (or twice-derived) ElementSizeSymbol; any other symbol yields 0.
	template <>
	GiNaC::ex GiNaCElementSizeSymbol::derivative(const GiNaC::symbol &s) const
	{
		if (get_struct().is_lagrangian())
			return 0;
		pyoomph::FiniteElementCode *code = (pyoomph::FiniteElementCode *)(get_struct().get_code()); // Cast aways the constness
		pyoomph::FiniteElementField *testf;
		if (!code->coordinates_as_dofs || pyoomph::ignore_nodal_position_derivatives_for_pitchfork_symmetry())
			return 0;
		// TODO: Other spaces, e.g. bulk
		if (!get_struct().is_derived())
		{
			testf = code->get_field_by_name("coordinate_x");
			if (testf && s == testf->get_symbol())
			{
				return 0 + GiNaCElementSizeSymbol(code->get_elemsize_derived(0, get_struct().is_with_coordsys()));
			}
			testf = code->get_field_by_name("coordinate_y");
			if (testf && s == testf->get_symbol())
			{
				return 0 + GiNaCElementSizeSymbol(code->get_elemsize_derived(1, get_struct().is_with_coordsys()));
			}
			testf = code->get_field_by_name("coordinate_z");
			if (testf && s == testf->get_symbol())
			{
				return 0 + GiNaCElementSizeSymbol(code->get_elemsize_derived(2, get_struct().is_with_coordsys()));
			}
		}
		else if (!get_struct().is_derived2())
		{
			int dir1 = get_struct().get_derived_direction();
			testf = code->get_field_by_name("coordinate_x");
			if (testf && s == testf->get_symbol())
			{
				return 0 + GiNaCElementSizeSymbol(code->get_elemsize_derived2(dir1, 0, get_struct().is_with_coordsys()));
			}
			testf = code->get_field_by_name("coordinate_y");
			if (testf && s == testf->get_symbol())
			{
				return 0 + GiNaCElementSizeSymbol(code->get_elemsize_derived2(dir1, 1, get_struct().is_with_coordsys()));
			}
			testf = code->get_field_by_name("coordinate_z");
			if (testf && s == testf->get_symbol())
			{
				return 0 + GiNaCElementSizeSymbol(code->get_elemsize_derived2(dir1, 2, get_struct().is_with_coordsys()));
			}
		}
		return 0;
	}

	template <>
	void GiNaCNormalSymbol::print(const print_context &c, unsigned) const
	{
		const pyoomph::NormalSymbol &sp = get_struct();
		if (GiNaC::is_a<print_csrc_FEM>(c))
		{
			const auto &femprint = dynamic_cast<const print_csrc_FEM &>(c);
			if (femprint.FEM_opts->for_code)
			{

				std::string prefix = "shapeinfo->";
				if (femprint.FEM_opts->for_code == sp.get_code())
				{
				}
				else if (femprint.FEM_opts->for_code->get_bulk_element() && femprint.FEM_opts->for_code->get_bulk_element() == sp.get_code())
				{
					prefix = "shapeinfo->bulk_shapeinfo->";
				}
				else if (femprint.FEM_opts->for_code->get_opposite_interface_code() && femprint.FEM_opts->for_code->get_opposite_interface_code() == sp.get_code())
				{
					prefix = "shapeinfo->opposite_shapeinfo->";
				}
				else
				{
					throw_runtime_error("Normal may not be used in an external element yet");
				}
				if (sp.get_derived_direction() == -1)
				{
					c.s << prefix << "normal[" << sp.get_direction() << "]";
				}
				else if (sp.get_derived_direction2() == -1)
				{
					c.s << prefix << "d_normal_dcoord[" << sp.get_direction() << "][" << (sp.is_derived_by_lshape2() ? "l_shape2" : "l_shape") << "][" << sp.get_derived_direction() << "]";
				}
				else
				{
					c.s << prefix << "d2_normal_d2coord[" << sp.get_direction() << "][l_shape][" << sp.get_derived_direction() << "][l_shape2][" << sp.get_derived_direction2() << "]";
				}
				return;
			}
		}
		std::string expansion_mode_str = (sp.expansion_mode != 0 ? "| MODE " + std::to_string(sp.expansion_mode) : "");
		if (sp.get_derived_direction() == -1)
		{
			c.s << "<" <<  "NORMAL COMPONENT " << sp.get_direction() << " @ " << sp.get_code() << expansion_mode_str << ">";
		}
		else if (sp.get_derived_direction2() == -1)
		{
			c.s << "<" <<  "NORMAL COMPONENT " << sp.get_direction() << " DERIVED in DIR " << sp.get_derived_direction() << " @ " << sp.get_code() << expansion_mode_str << ">";
		}
		else
		{
			c.s << "<" <<  "NORMAL COMPONENT " << sp.get_direction() << " DERIVED in DIRs " << sp.get_derived_direction() << " and " << sp.get_derived_direction2() << " @ " << sp.get_code() << expansion_mode_str << ">";
		}
	}
	template <>
	GiNaC::ex GiNaCNormalSymbol::derivative(const GiNaC::symbol &s) const
	{
        
		if (s == pyoomph::expressions::t || s == pyoomph::expressions::x || s == pyoomph::expressions::y || s == pyoomph::expressions::z || s == pyoomph::expressions::X || s == pyoomph::expressions::Y || s == pyoomph::expressions::Z)
		{
			throw_runtime_error("Cannot derive the normal with respect to space or time yet");
		}
		else
		{

			const pyoomph::NormalSymbol &sp = get_struct();
			if (pyoomph::__derive_only_by_expansion_mode && sp.expansion_mode != *pyoomph::__derive_only_by_expansion_mode)
				return 0;

//      std::cout << "ENTERING NORMAL DIFF " << sp.no_jacobian << " " << pyoomph::__derive_shapes_by_second_index <<  " " << sp.no_hessian << std::endl;
//      std::cout << " BY WHAT " << s << std::endl;
			if ((sp.no_jacobian && (!pyoomph::__derive_shapes_by_second_index)) || (sp.no_hessian && pyoomph::__derive_shapes_by_second_index))
				return 0;

			std::ostringstream oss;
			oss << s;
			std::string sname = oss.str();
			if (sname == "coordinate_x" || sname == "coordinate_y" || sname == "coordinate_z")
			{
				//    std::cout << "IN NORMAL DERIVATIVE wrt " << s  << std::endl;
				int coord_dir = (sname == "coordinate_x" ? 0 : (sname == "coordinate_y" ? 1 : 2));
				if (sp.get_code() == pyoomph::__current_code)
				{
					//    	   std::cout << " MY NORMAL DERIV " << s  << std::endl;
					// Here, we have to be careful! The normal of a facet element depends on the bulk element coordinates
					if (!pyoomph::__current_code->get_bulk_element())
					{
						auto *posspace = pyoomph::__current_code->get_my_position_space();
						bool found = false;
						for (auto *f : pyoomph::__current_code->get_fields_on_space(posspace))
						{
							if (f->get_name() == sname)
							{
								if (f->get_symbol() == s)
								{
									found = true;
									break;
								}
							}
						}
						if (found)
						{
							if (sp.get_derived_direction() == -1)
							{
								pyoomph::NormalSymbol nret(pyoomph::__current_code, sp.get_direction(), coord_dir, -1, pyoomph::__derive_shapes_by_second_index);
								nret.no_jacobian = sp.no_jacobian;
								nret.no_hessian = sp.no_hessian;
								nret.expansion_mode = sp.expansion_mode;
								return GiNaCNormalSymbol(nret);
							}
							else
							{
								pyoomph::NormalSymbol nret(pyoomph::__current_code, sp.get_direction(), sp.get_derived_direction(), coord_dir, pyoomph::__derive_shapes_by_second_index);
								nret.no_jacobian = sp.no_jacobian;
								nret.no_hessian = sp.no_hessian;
								nret.expansion_mode = sp.expansion_mode;
								return GiNaCNormalSymbol(nret);
							}
						}
					}
					else // We need to make sure the normal is position-diffed wrto the bulk element!
					{
						auto *posspace = pyoomph::__current_code->get_bulk_element()->get_my_position_space();
						bool found = false;
						for (auto *f : pyoomph::__current_code->get_bulk_element()->get_fields_on_space(posspace))
						{
							if (f->get_name() == sname)
							{
								if (f->get_symbol() == s)
								{
									found = true;
									break;
								}
							}
						}
						if (found)
						{
							if (sp.get_derived_direction() == -1)
							{
								pyoomph::NormalSymbol nret(pyoomph::__current_code, sp.get_direction(), coord_dir, -1, pyoomph::__derive_shapes_by_second_index);
								nret.no_jacobian = sp.no_jacobian;
								nret.no_hessian = sp.no_hessian;
								nret.expansion_mode = sp.expansion_mode;
								return GiNaCNormalSymbol(nret);
							}
							else
							{
								pyoomph::NormalSymbol nret(pyoomph::__current_code, sp.get_direction(), sp.get_derived_direction(), coord_dir, pyoomph::__derive_shapes_by_second_index);
								nret.no_jacobian = sp.no_jacobian;
								nret.no_hessian = sp.no_hessian;
								nret.expansion_mode = sp.expansion_mode;
								return GiNaCNormalSymbol(nret);
							}
						}
					}
				}
				else if (sp.get_code() && sp.get_code() == pyoomph::__current_code->get_bulk_element())
				{
					// std::cout << "DERIVING PARENT NORMAL " << sname << " " << s << " " << sp.get_direction() << "  " << coord_dir << std::endl;
					if (!pyoomph::__current_code->get_bulk_element()->get_bulk_element())
					{
						// 	   		 std::cout << " MODE 1 "<< std::endl;
						auto *posspace = pyoomph::__current_code->get_bulk_element()->get_my_position_space();
						bool found = false;
						for (auto *f : pyoomph::__current_code->get_bulk_element()->get_fields_on_space(posspace))
						{
							if (f->get_name() == sname)
							{
								if (f->get_symbol() == s)
								{
									found = true;
									break;
								}
							}
						}
						if (found)
						{
							if (sp.get_derived_direction() == -1)
							{
								pyoomph::NormalSymbol nret(pyoomph::__current_code->get_bulk_element(), sp.get_direction(), coord_dir, -1, pyoomph::__derive_shapes_by_second_index);
								nret.no_jacobian = sp.no_jacobian;
								nret.no_hessian = sp.no_hessian;
								nret.expansion_mode = sp.expansion_mode;
								return GiNaCNormalSymbol(nret);
							}
							else
							{
								pyoomph::NormalSymbol nret(pyoomph::__current_code->get_bulk_element(), sp.get_direction(), sp.get_derived_direction(), coord_dir, pyoomph::__derive_shapes_by_second_index);
								nret.no_jacobian = sp.no_jacobian;
								nret.no_hessian = sp.no_hessian;
								nret.expansion_mode = sp.expansion_mode;
								return GiNaCNormalSymbol(nret);
							}
						}
					}
					else // We need to make sure the normal is position-diffed wrto the bulk element!
					{
						//			 	   		 std::cout << " MODE 2 "<< std::endl;
						auto *posspace = pyoomph::__current_code->get_bulk_element()->get_bulk_element()->get_my_position_space();
						bool found = false;
						for (auto *f : pyoomph::__current_code->get_bulk_element()->get_bulk_element()->get_fields_on_space(posspace))
						{
							// std::cout << "  CHECKING FIELD "  << f->get_name() << "  " << sname << std::endl;
							if (f->get_name() == sname)
							{
								//					 std::cout << "  CHECKING SYMBOL "  << f->get_symbol() << "  " << s << "  " << (f->get_symbol()==s? "TRUE" : "FALSE") << std::endl;
								if (f->get_symbol() == s)
								{
									found = true;
									break;
								}
							}
						}
						if (found)
						{
							// std::cout << "  FOUND MNODE 2" 	 << std::endl;
							if (sp.get_derived_direction() == -1)
							{

								pyoomph::NormalSymbol nret(pyoomph::__current_code->get_bulk_element(), sp.get_direction(), coord_dir, -1, pyoomph::__derive_shapes_by_second_index);
								nret.no_jacobian = sp.no_jacobian;
								nret.no_hessian = sp.no_hessian;
								nret.expansion_mode = sp.expansion_mode;
								return GiNaCNormalSymbol(nret);
							}
							else
							{
								pyoomph::NormalSymbol nret(pyoomph::__current_code->get_bulk_element(), sp.get_direction(), sp.get_derived_direction(), coord_dir, pyoomph::__derive_shapes_by_second_index);
								nret.no_jacobian = sp.no_jacobian;
								nret.no_hessian = sp.no_hessian;
								nret.expansion_mode = sp.expansion_mode;
								return GiNaCNormalSymbol(nret);
							}
						}
					}
				}
				else if (sp.get_code() && sp.get_code() == pyoomph::__current_code->get_opposite_interface_code())
				{
					// std::cout << "DERIVING PARENT NORMAL " << sname << " " << s << " " << sp.get_direction() << "  " << coord_dir << std::endl;
					if (!pyoomph::__current_code->get_bulk_element()->get_opposite_interface_code())
					{
						// 	   		 std::cout << " MODE 1 "<< std::endl;
						auto *posspace = pyoomph::__current_code->get_opposite_interface_code()->get_my_position_space();
						bool found = false;
						for (auto *f : pyoomph::__current_code->get_opposite_interface_code()->get_fields_on_space(posspace))
						{
							if (f->get_name() == sname)
							{
								if (f->get_symbol() == s)
								{
									found = true;
									break;
								}
							}
						}
						if (found)
						{
							if (sp.get_derived_direction() == -1)
							{
								pyoomph::NormalSymbol nret(pyoomph::__current_code->get_opposite_interface_code(), sp.get_direction(), coord_dir, -1, pyoomph::__derive_shapes_by_second_index);
								nret.no_jacobian = sp.no_jacobian;
								nret.no_hessian = sp.no_hessian;
								nret.expansion_mode = sp.expansion_mode;
								return GiNaCNormalSymbol(nret);
							}
							else
							{
								pyoomph::NormalSymbol nret(pyoomph::__current_code->get_opposite_interface_code(), sp.get_direction(), sp.get_derived_direction(), coord_dir, pyoomph::__derive_shapes_by_second_index);
								nret.no_jacobian = sp.no_jacobian;
								nret.no_hessian = sp.no_hessian;
								nret.expansion_mode = sp.expansion_mode;
								return GiNaCNormalSymbol(nret);
							}
						}
					}
					else // We need to make sure the normal is position-diffed wrto the bulk element!
					{
						//			 	   		 std::cout << " MODE 2 "<< std::endl;
						auto *posspace = pyoomph::__current_code->get_opposite_interface_code()->get_bulk_element()->get_my_position_space();
						bool found = false;
						for (auto *f : pyoomph::__current_code->get_opposite_interface_code()->get_bulk_element()->get_fields_on_space(posspace))
						{
							// std::cout << "  CHECKING FIELD "  << f->get_name() << "  " << sname << std::endl;
							if (f->get_name() == sname)
							{
								//					 std::cout << "  CHECKING SYMBOL "  << f->get_symbol() << "  " << s << "  " << (f->get_symbol()==s? "TRUE" : "FALSE") << std::endl;
								if (f->get_symbol() == s)
								{
									found = true;
									break;
								}
							}
						}
						if (found)
						{
							// std::cout << "  FOUND MNODE 2" 	 << std::endl;
							if (sp.get_derived_direction() == -1)
							{
								pyoomph::NormalSymbol nret(pyoomph::__current_code->get_opposite_interface_code(), sp.get_direction(), coord_dir, -1, pyoomph::__derive_shapes_by_second_index);
								nret.no_jacobian = sp.no_jacobian;
								nret.no_hessian = sp.no_hessian;
								nret.expansion_mode = sp.expansion_mode;
								return GiNaCNormalSymbol(nret);
							}
							else
							{
								pyoomph::NormalSymbol nret(pyoomph::__current_code->get_opposite_interface_code(), sp.get_direction(), sp.get_derived_direction(), coord_dir, pyoomph::__derive_shapes_by_second_index);
								nret.no_jacobian = sp.no_jacobian;
								nret.no_hessian = sp.no_hessian;
								nret.expansion_mode = sp.expansion_mode;
								return GiNaCNormalSymbol(nret);
							}
						}
					}
				}
				else
				{
					throw_runtime_error("Cannot access the normal of this domain");
				}
			}
		}
		return 0;
	}

	template <>
	void GiNaCShapeExpansion::print(const print_context &c, unsigned) const
	{
		const pyoomph::ShapeExpansion &sp = get_struct();
		std::string dt = "";
		if (sp.dt_order == 1)
			dt = "d/dt ";
		else if (sp.dt_order > 1)
			dt = "d^" + std::to_string(sp.dt_order) + "/dt^" + std::to_string(sp.dt_order);
		if (GiNaC::is_a<print_csrc_FEM>(c))
		{
			const auto &femprint = dynamic_cast<const print_csrc_FEM &>(c);
			if (femprint.FEM_opts->for_code)
			{
				if (!sp.is_derived)
				{
					c.s << sp.get_spatial_interpolation_name(femprint.FEM_opts->for_code);
				}
				else
				{
					std::string timedisc_scheme = sp.get_timedisc_scheme(femprint.FEM_opts->for_code);
					bool dgs = true;
					if (sp.field->degraded_start.count(""))
						dgs = sp.field->degraded_start[""];
					if (sp.dt_order == 1 && dgs && timedisc_scheme != "BDF1")
					{
						timedisc_scheme += "_degr";
					}
					if (sp.dt_order > 2)
					{
						throw_runtime_error("Too high dt order");
					}
					else if (sp.dt_order == 2)
						c.s << "shapeinfo->timestepper_weights_d2t_" << timedisc_scheme << "[0]*";
					else if (sp.dt_order == 1)
					{
						c.s << "shapeinfo->timestepper_weights_dt_" << timedisc_scheme << "[0]*";
					}
					if (femprint.FEM_opts->in_subexpr_deriv)
					{
						if (sp.is_derived)
						{
							c.s << "1";
						}
					}
					else
					{
						if (sp.is_derived && (sp.nodal_coord_dir >= 0 || sp.nodal_coord_dir2 >= 0))
						{
							if (sp.nodal_coord_dir >= 0 && sp.nodal_coord_dir2 >= 0)
							{
								throw_runtime_error("DD")
							}
							std::string shapename=sp.basis->get_space()->get_shape_name();
							if (shapename=="C2TB" || shapename=="C2" || shapename=="C1TB" || shapename=="C1")
							{
								shapename="d_dx_shape_dcoord[SPACE_INDEX_" + sp.basis->get_space()->get_shape_name() + "]";
							}						
							else
							{
								shapename="d_dx_shape_dcoord_" + sp.basis->get_space()->get_shape_name();
							}
							if (sp.nodal_coord_dir >= 0)
							{
								std::string shapestr = femprint.FEM_opts->for_code->get_shape_info_str(sp.basis->get_space()) + "->" + shapename;
								shapestr += "[l_shape2][" + std::to_string(dynamic_cast<pyoomph::D1XBasisFunction *>(sp.basis)->get_direction()) + "][l_shape][" + std::to_string(sp.nodal_coord_dir) + "]";
								c.s << shapestr;
							}
							else
							{
								std::string shapestr = femprint.FEM_opts->for_code->get_shape_info_str(sp.basis->get_space()) + "->" + shapename;
								shapestr += "[l_shape][" + std::to_string(dynamic_cast<pyoomph::D1XBasisFunction *>(sp.basis)->get_direction()) + "][" + "l_shape2" + "][" + std::to_string(sp.nodal_coord_dir2) + "]";
								c.s << shapestr;
							}
						}
						else
							c.s << sp.get_shape_string(femprint.FEM_opts->for_code, (sp.is_derived_other_index ? "l_shape2" : "l_shape"));
					}
				}
				return;
			}
		}
		else if (GiNaC::is_a<print_latex_FEM>(c))
		{
			const auto &femprint = dynamic_cast<const print_latex_FEM &>(c);
			if (femprint.FEM_opts->for_code && femprint.FEM_opts->for_code->latex_printer)
			{
				std::map<std::string, std::string> texinfo;
				texinfo["typ"] = "field";
				texinfo["name"] = sp.field->get_name();
				texinfo["timediff"] = dt;
				texinfo["basis"] = sp.basis->to_string();
				texinfo["domain"] = sp.field->get_space()->get_code()->get_domain_name();
				texinfo["derived"] = (sp.is_derived ? "true" : "false");
				texinfo["is_derived_other_index"] = (sp.is_derived_other_index ? "true" : "false");
				texinfo["no_jacobian"] = (sp.no_jacobian ? "true" : "false");
				texinfo["no_hessian"] = (sp.no_hessian ? "true" : "false");
				texinfo["nodal_coord_dir"] = std::to_string(sp.nodal_coord_dir);
				texinfo["nodal_coord_dir2"] = std::to_string(sp.nodal_coord_dir2);
				texinfo["expansion_mode"] = std::to_string(sp.expansion_mode);
				c.s << femprint.FEM_opts->for_code->latex_printer->_get_LaTeX_expression(texinfo, femprint.FEM_opts->for_code);
				return;
			}
		}
		c.s << "<" << (sp.is_derived ? (sp.is_derived_other_index ? "ALT.DERIVED " : "DERIVED ") : "") << (sp.nodal_coord_dir == -1 ? "" : "COORDINATE_DIFF_" + std::to_string(sp.nodal_coord_dir) + " ") << "SHAPEEXP of " << dt << sp.field->get_name() << " of " << sp.field->get_space()->get_code()->get_domain_name() << " @ " << sp.basis->to_string() << (sp.no_jacobian ? " | NO_JACOBIAN" : "") << (sp.no_hessian ? " | NO_HESSIAN" : "") << (sp.expansion_mode ? (" | MODE " + std::to_string(sp.expansion_mode)) : "") << ">";
	}


	template <>
	GiNaC::ex GiNaCShapeExpansion::derivative(const GiNaC::symbol &s) const
	{
		const pyoomph::ShapeExpansion &sp = get_struct();
		std::ostringstream oss;
		oss << s;
		std::string sname = oss.str();
		if (pyoomph::__derive_only_by_expansion_mode && sp.expansion_mode != *pyoomph::__derive_only_by_expansion_mode)
			return 0;
		//			   std::cout << " ENTER diff "  << (*this) << "  " << sp.field->get_name() <<  "   WRT " << s <<  std::endl;

		   /*if (pyoomph::pyoomph_verbose)
		   {
			std::cout << "DERIV SHAPE EXP  " <<(*this) << " by " << s << " which is a realsymb? " << (GiNaC::is_a<GiNaC::realsymbol>(s) ? " true " : "false") << std::endl;
					 std::cout << "DERIV SHAPE EXP  " << (GiNaC::ex_to<GiNaC::realsymbol>(s)==pyoomph::expressions::t ? " same" : "not" )<< " namely : " << s << " vs " << pyoomph::expressions::t << " SUB "  << (pyoomph::expressions::t-s) <<std::endl;
		   }

		 if (pyoomph::pyoomph_verbose)
		 {std::cout << "SYMBOLIC MATCH IN DERIV  " << sp.field->get_symbol()<< " == " <<s<< "  :  "  << ( sp.field->get_symbol()==s ? "true" : "false") <<  std::endl;
		  if (GiNaC::is_a<GiNaC::realsymbol>(s)) { std::cout << "   " << GiNaC::ex_to<GiNaC::realsymbol>(s)-GiNaC::ex_to<GiNaC::realsymbol>(sp.field->get_symbol()) << std::endl; }
		  std::cout << " HASHES " << sp.field->get_symbol().gethash() << "  "<< GiNaC::ex_to<GiNaC::symbol>(sp.field->get_symbol().gethash()) << "  " << s.gethash() << "  " << GiNaC::ex_to<GiNaC::realsymbol>(s) << std::endl;
		  std::cout << "EQUAL " << (GiNaC::ex_to<GiNaC::realsymbol>(s).is_equal(s) ? "Y" : "N") <<std::endl;
		 }*/

		// Derivatives with respect to the time
		if (s == pyoomph::expressions::t || s == pyoomph::expressions::_dt_BDF1 || s == pyoomph::expressions::_dt_BDF2 || s == pyoomph::expressions::_dt_Newmark2)
		{
			if (sp.time_history_index != 0)
			{
				throw_runtime_error("Cannot derive with respect to time in the past yet");
			}
			if (dynamic_cast<pyoomph::PositionFiniteElementSpace *>(sp.field->get_space())) // First check, to prevent any derivatives as partial_t(x)!=0
			{
				if (sp.field->get_name() == "coordinate_x" || sp.field->get_name() == "coordinate_y" || sp.field->get_name() == "coordinate_z")
					return 0;
				if (sp.field->get_name() == "lagrangian_x" || sp.field->get_name() == "lagrangian_y" || sp.field->get_name() == "lagrangian_z")
					return 0;
				if (sp.field->get_name() == "local_coordinate_1" || sp.field->get_name() == "local_coordinate_2" || sp.field->get_name() == "local_coordinate_3")
					return 0;
				if (sp.field->get_name() == "zeta_coordinate_1" || sp.field->get_name() == "zeta_coordinate_2" || sp.field->get_name() == "zeta_coordinate_3")
					return 0;
			}
			std::string timescheme;
			unsigned dt_order = sp.dt_order + 1;
			if (s == pyoomph::expressions::t)
			{
				timescheme = pyoomph::__current_code->get_default_timestepping_scheme(dt_order);
			}
			else if (s == pyoomph::expressions::_dt_BDF1)
				timescheme = "BDF1";
			else if (s == pyoomph::expressions::_dt_BDF2)
				timescheme = "BDF2";
			else if (s == pyoomph::expressions::_dt_Newmark2)
				timescheme = "Newmark2";

			// Auto switch second order to Newmark
			if (dt_order == 2)
				timescheme = "Newmark2";
			auto se = pyoomph::ShapeExpansion(sp.field, dt_order, sp.basis, timescheme);
			if (sp.no_jacobian)
				se.no_jacobian = true;
			if (sp.no_hessian)
				se.no_hessian = true;
			if (sp.expansion_mode)
				se.expansion_mode = sp.expansion_mode;
			return GiNaCShapeExpansion(se);
		}
		else if (sp.time_history_index != 0)
		{
			return 0; // All derivatives in the past are zero => No contrib to Jacobian //TODO: IS this always true? What about positions?
		}
		else if (s == pyoomph::expressions::__partial_t_mass_matrix)
		{
			if (sp.is_derived && sp.dt_order == 1)
			{
				auto se = pyoomph::ShapeExpansion(sp.field, 0, sp.basis, sp.dt_scheme, true);
				if (sp.no_jacobian)
					se.no_jacobian = true;
				if (sp.no_hessian)
					se.no_hessian = true;
				if (sp.expansion_mode)
					se.expansion_mode = sp.expansion_mode;
				se.is_derived_other_index = sp.is_derived_other_index;
				return GiNaCShapeExpansion(se);
			}
			else
				return 0;
		}
		// Eulerian derivatives
		else if (s == pyoomph::expressions::x || s == pyoomph::expressions::y || s == pyoomph::expressions::z)
		{
			if (pyoomph::pyoomph_verbose)
			{
			 std::cout << "   IS COORD DIFF "  << (*this) << "  " << sp.field->get_name() <<  "   WRT " << s <<  std::endl;
			}
			unsigned dir = (s == pyoomph::expressions::x ? 0 : (s == pyoomph::expressions::y ? 1 : 2));
			if (dynamic_cast<pyoomph::PositionFiniteElementSpace *>(sp.field->get_space()))
			{
				bool no_codimension = (sp.basis->get_space()->get_code()->element_dim == (int)sp.basis->get_space()->get_code()->nodal_dimension());
				bool no_eigenexpansion = (sp.expansion_mode==0 || pyoomph::__derive_only_by_expansion_mode);
				if (pyoomph::pyoomph_verbose)
				{
			 		std::cout << "   NO CODIM, NO EIGENEXPANSION "  << no_codimension << "  ,  "  << no_eigenexpansion <<  std::endl;
				}
				if (!sp.dt_order && no_codimension && no_eigenexpansion) // TODO: Check for no co-dimension relevant?
				{
					//	   std::cout << "   BB alt diff "  << (*this) << "  " << sp.field->get_name() << std::endl;
					//   	     std::cout << "GRAD TEST " <<  << std::endl;
					if (sp.field->get_name() == "coordinate_x" || sp.field->get_name() == "mesh_x")
					{
						if (dir == 0)
							return 1;
						else
							return 0;
					}
					else if (sp.field->get_name() == "coordinate_y" || sp.field->get_name() == "mesh_y")
					{
						if (dir == 1)
							return 1;
						else
							return 0;
					}
					else if (sp.field->get_name() == "coordinate_z" || sp.field->get_name() == "mesh_z")
					{
						if (dir == 2)
							return 1;
						else
							return 0;
					}
					else
					{
						std::ostringstream sn;
						sn << s;
						throw_runtime_error("Generic Position derivatives of " + sp.field->get_name() + " with respect to " + sn.str());
					}
				}
				else
				{
					//				   std::cout << "   AA alt diff "  << (*this) << "  " << sp.field->get_name() << std::endl;
					/*if (sp.field->get_name() == "coordinate_x" || sp.field->get_name() == "coordinate_y" || sp.field->get_name() == "coordinate_z")
					{
						if (pyoomph::pyoomph_verbose)
						{
							std::cout << "   HIT ELSE CASE IN COORD DIFF "  << (*this) << "  " << sp.field->get_name() <<  "   WRT " << s <<  std::endl;
						}
						return 0;
					}
					else*/
					{
						auto se = pyoomph::ShapeExpansion(sp.field, sp.dt_order, sp.basis->get_diff_x(dir), sp.dt_scheme, sp.is_derived, sp.nodal_coord_dir);
						if (sp.no_jacobian)
							se.no_jacobian = true;
						if (sp.no_hessian)
							se.no_hessian = true;
						if (sp.expansion_mode)
							se.expansion_mode = sp.expansion_mode;
						se.is_derived_other_index = sp.is_derived_other_index;
						return GiNaCShapeExpansion(se);
					}
				}
			}

			if (sp.field->get_space()->is_basis_derivative_zero(sp.basis, dir))
			{
				{
					//std::cout << "WARNING: Spatial derivative of basis of field " << sp.field->get_name() << " is zero. Please consider this!" << std::endl;
					// throw_runtime_error("Basis derivative is zero, TODO: make this an optional warning");
				}
				return 0;
			}
			else
			{
				auto se = pyoomph::ShapeExpansion(sp.field, sp.dt_order, sp.basis->get_diff_x(dir), sp.dt_scheme, sp.is_derived, sp.nodal_coord_dir);
				if (sp.no_jacobian)
					se.no_jacobian = true;
				if (sp.no_hessian)
					se.no_hessian = true;
				if (sp.expansion_mode)
					se.expansion_mode = sp.expansion_mode;
				se.is_derived_other_index = sp.is_derived_other_index;
				return GiNaCShapeExpansion(se);
			}
		}
		// Lagrangian diffs
		else if (s == pyoomph::expressions::X || s == pyoomph::expressions::Y || s == pyoomph::expressions::Z)
		{
			unsigned dir = (s == pyoomph::expressions::X ? 0 : (s == pyoomph::expressions::Y ? 1 : 2));
			if (dynamic_cast<pyoomph::PositionFiniteElementSpace *>(sp.field->get_space()))
			{
				if (sp.field->get_name() == "lagrangian_x")
				{
					if (dir == 0)
						return 1;
					else
						return 0;
				}
				else if (sp.field->get_name() == "lagrangian_y")
				{
					if (dir == 1)
						return 1;
					else
						return 0;
				}
				else if (sp.field->get_name() == "lagrangian_z")
				{
					if (dir == 2)
						return 1;
					else
						return 0;
				}
			}
			auto se = pyoomph::ShapeExpansion(sp.field, sp.dt_order, sp.basis->get_diff_X(dir), sp.dt_scheme);
			if (sp.no_jacobian)
				se.no_jacobian = true;
			if (sp.no_hessian)
				se.no_hessian = true;
			if (sp.expansion_mode)
				se.expansion_mode = sp.expansion_mode;
			se.is_derived_other_index = sp.is_derived_other_index;
			return GiNaCShapeExpansion(se);
		}

		else if (s == pyoomph::expressions::zeta_coordinate_1 || s == pyoomph::expressions::zeta_coordinate_2 || s == pyoomph::expressions::zeta_coordinate_3)
		{
			throw_runtime_error("Cannot derive with respect to zeta coordinates yet. This is not implemented in the code, as it is not needed for the current applications. If you need this, please contact the developers.");
		}
		// Local coordinate diffs
		else if (s == pyoomph::expressions::local_coordinate_1 || s == pyoomph::expressions::local_coordinate_2 || s == pyoomph::expressions::local_coordinate_3)
		{
			unsigned dir = (s == pyoomph::expressions::local_coordinate_1 ? 0 : (s == pyoomph::expressions::local_coordinate_2 ? 1 : 2));
			if (dynamic_cast<pyoomph::PositionFiniteElementSpace *>(sp.field->get_space()))
			{
				if (sp.field->get_name() == "local_coordinate_1")
				{
					if (dir == 0)
						return 1;
					else
						return 0;
				}
				else if (sp.field->get_name() == "local_coordinate_2")
				{
					if (dir == 1)
						return 1;
					else
						return 0;
				}
				else if (sp.field->get_name() == "local_coordinate_3")
				{
					if (dir == 2)
						return 1;
					else
						return 0;
				}
			}
			auto se = pyoomph::ShapeExpansion(sp.field, sp.dt_order, sp.basis->get_diff_S(dir), sp.dt_scheme);
			if (sp.no_jacobian)
				se.no_jacobian = true;
			if (sp.no_hessian)
				se.no_hessian = true;
			if (sp.expansion_mode)
				se.expansion_mode = sp.expansion_mode;
			se.is_derived_other_index = sp.is_derived_other_index;
			return GiNaCShapeExpansion(se);
		}

		else if (sp.is_derived) // We just have a shape term left (without the nodal weighting)
		{
			// std::cout << "HIT If derived case" << std::endl;

			if (sp.nodal_coord_dir >= 0 || sp.nodal_coord_dir2 >= 0)
				throw_runtime_error("We have a derived shape expansion-> only psi^l. If it is dxpsi^l, we might have a COORDDIFF, which might give an u term ");
			int coord_dir = (sname == "coordinate_x" ? 0 : (sname == "coordinate_y" ? 1 : 2));
			if ((dynamic_cast<pyoomph::D1XBasisFunction *>(sp.basis) && !(dynamic_cast<pyoomph::D1XBasisFunctionLagr *>(sp.basis))))
			{
				if (sname == "coordinate_x" || sname == "coordinate_y" || sname == "coordinate_z")
				{
					pyoomph::FiniteElementCode *posspace_domain = sp.can_be_a_positional_derivative_symbol(s);
					if (posspace_domain)
					{
						//						   std::cout << "FOUND AND " << sp.basis->get_space()->get_code()->coordinates_as_dofs << " NODAL COORD DIR " << sp.nodal_coord_dir << std::endl;
						if (sp.basis->get_space()->get_code()->coordinates_as_dofs && !pyoomph::ignore_nodal_position_derivatives_for_pitchfork_symmetry())
						{
							if (sp.nodal_coord_dir >= 0)
								throw_runtime_error("DD");
							// Ugly construct, but we have to call one of the constructors...
							auto se = pyoomph::ShapeExpansion(sp.field, sp.dt_order, sp.basis, sp.dt_scheme, sp.is_derived, sp.nodal_coord_dir, pyoomph::__derive_shapes_by_second_index, coord_dir);
							if (sp.no_jacobian)
								se.no_jacobian = true;
							if (sp.no_hessian)
								se.no_hessian = true;
							if (sp.expansion_mode)
								se.expansion_mode = sp.expansion_mode;
							//								se.is_derived_other_index = sp.is_derived_other_index;
							return GiNaCShapeExpansion(se);
						}
					}
				}
			}
			return 0;
		}
		else if (sp.field->get_symbol() == s || sp.field->get_symbol().is_equal(s))
		{
			const pyoomph::ShapeExpansion *SEwr = pyoomph::__deriv_subexpression_wrto;

			if (pyoomph::pyoomph_verbose)
				std::cout << "ENTERING EQUAL DIFF " << SEwr << std::endl;
			if (!SEwr)
			{
				if ((sp.no_jacobian && (!pyoomph::__derive_shapes_by_second_index)) || (sp.no_hessian && pyoomph::__derive_shapes_by_second_index))
					return 0;
				auto se = pyoomph::ShapeExpansion(sp.field, sp.dt_order, sp.basis, sp.dt_scheme, true, sp.nodal_coord_dir, pyoomph::__derive_shapes_by_second_index);
				if (pyoomph::__derive_shapes_by_second_index)
					se.is_derived_other_index = true;
				// Here we have to check wether we derive e.g. dX_l/dt*dphi_l/dx. It will give another contribution from the coordinate diff
				if (sp.dt_order && (dynamic_cast<pyoomph::D1XBasisFunction *>(sp.basis) && !(dynamic_cast<pyoomph::D1XBasisFunctionLagr *>(sp.basis))))
				{
					std::ostringstream ossn;
					ossn << s;
					std::string sname = ossn.str();
					int coord_dir = (sname == "coordinate_x" ? 0 : (sname == "coordinate_y" ? 1 : 2));
					if (sp.nodal_coord_dir >= 0)
						throw_runtime_error("Handle second order derivative here");
					//std::cout << "SHOULD NOT GENERATE A TERM HERE " << pyoomph::__ignore_dpsi_coord_diffs_in_jacobian << "  " << GiNaCShapeExpansion(se) << std::endl;
					return GiNaCShapeExpansion(se) + (pyoomph::__ignore_dpsi_coord_diffs_in_jacobian ? 0 : 1 )*GiNaCShapeExpansion(pyoomph::ShapeExpansion(sp.field, sp.dt_order, sp.basis, sp.dt_scheme, false, coord_dir, pyoomph::__derive_shapes_by_second_index));
					/*				  std::ostringstream oss;
									  oss << (*this) << " WT " << s;
									  throw_runtime_error("TODO: Derive here "+oss.str()); */
				}
				return GiNaCShapeExpansion(se);
			}
			else
			{
				if (SEwr->field == sp.field && sp.dt_order == SEwr->dt_order && sp.basis == SEwr->basis && sp.dt_scheme == SEwr->dt_scheme)
				{
					if ((sp.no_jacobian && (!pyoomph::__derive_shapes_by_second_index)) || (sp.no_hessian && pyoomph::__derive_shapes_by_second_index))
						return 0;
					auto se = pyoomph::ShapeExpansion(sp.field, sp.dt_order, sp.basis, sp.dt_scheme, true, sp.nodal_coord_dir, pyoomph::__derive_shapes_by_second_index);
					if (pyoomph::__derive_shapes_by_second_index)
					{
						if (sp.is_derived)
							throw_runtime_error("DD");
						se.is_derived_other_index = true;
					}
					// Here we have to check wether we derive e.g. dX_l/dt*dphi_l/dx. It will give another contribution from the coordinate diff
					if (sp.dt_order && (dynamic_cast<pyoomph::D1XBasisFunction *>(sp.basis) && !(dynamic_cast<pyoomph::D1XBasisFunctionLagr *>(sp.basis))))
					{
						std::ostringstream oss;
						oss << (*this) << " WT " << s;
						throw_runtime_error("TODO: Derive here " + oss.str());
					}
					return GiNaCShapeExpansion(se);
				}
				else
				{
					return 0;
				}
			}
		}
		else
		{

			int coord_dir = (sname == "coordinate_x" ? 0 : (sname == "coordinate_y" ? 1 : 2));
			if ((dynamic_cast<pyoomph::D1XBasisFunction *>(sp.basis) && !(dynamic_cast<pyoomph::D1XBasisFunctionLagr *>(sp.basis))))
			{
				//		   std::cout << "  hit else case "  << (*this) << "  " << sp.field->get_name() <<  "   WRT " << s << " sname " << sname << " dir " << coord_dir<<  std::endl;
				if (sname == "coordinate_x" || sname == "coordinate_y" || sname == "coordinate_z")
				{
					pyoomph::FiniteElementCode *posspace_domain = sp.can_be_a_positional_derivative_symbol(s);
					if (posspace_domain)
					{
						//					   std::cout << "FOUND AND " << sp.basis->get_space()->get_code()->coordinates_as_dofs << " NODAL COORD DIR " << sp.nodal_coord_dir << std::endl;
						if (sp.basis->get_space()->get_code()->coordinates_as_dofs && !pyoomph::ignore_nodal_position_derivatives_for_pitchfork_symmetry())
						{
							if ((sp.no_jacobian && (!pyoomph::__derive_shapes_by_second_index)) || (sp.no_hessian && pyoomph::__derive_shapes_by_second_index))
								return 0;
							if (sp.nodal_coord_dir <0 && !pyoomph::__derive_shapes_by_second_index && pyoomph::__ignore_dpsi_coord_diffs_in_jacobian)
								return 0;
							// Ugly construct, but we have to call one of the constructors...
							auto se = (sp.nodal_coord_dir >= 0
										   ? pyoomph::ShapeExpansion(sp.field, sp.dt_order, sp.basis, sp.dt_scheme, sp.is_derived, sp.nodal_coord_dir, pyoomph::__derive_shapes_by_second_index, coord_dir)
										   : pyoomph::ShapeExpansion(sp.field, sp.dt_order, sp.basis, sp.dt_scheme, sp.is_derived, coord_dir, pyoomph::__derive_shapes_by_second_index));
							if (sp.no_jacobian)
								se.no_jacobian = true;
							if (sp.no_hessian)
								se.no_hessian = true;
							if (sp.expansion_mode)
								se.expansion_mode = sp.expansion_mode;
							//								se.is_derived_other_index = sp.is_derived_other_index;
							return GiNaCShapeExpansion(se);
						}
						else
						{
							return 0;
						}
					}
				}
			}
		}
		return 0;
	}

	template <>
	void GiNaCTestFunction::print(const print_context &c, unsigned) const
	{
		const pyoomph::TestFunction &sp = get_struct();
		if (GiNaC::is_a<print_csrc_FEM>(c))
		{
			const auto &femprint = dynamic_cast<const print_csrc_FEM &>(c);
			if (femprint.FEM_opts->for_code)
			{
				std::string shapename = sp.basis->get_space()->get_shape_name();
				std::string shapename2;
				if (shapename == "C2TB" || shapename == "C2" || shapename == "C1TB" || shapename == "C1")
				{
					shapename = "d_dx_shape_dcoord[SPACE_INDEX_" + sp.basis->get_space()->get_shape_name() + "]";
					shapename2="d2_dx2_shape_dcoord[SPACE_INDEX_" + sp.basis->get_space()->get_shape_name() + "]";
				}
				else
				{
					shapename = "d_dx_shape_dcoord_" + sp.basis->get_space()->get_shape_name();
					shapename2 = "d2_dx2_shape_dcoord_" + sp.basis->get_space()->get_shape_name();
				}
				if (sp.nodal_coord_dir == -1)
				{
					c.s << sp.basis->get_c_varname(femprint.FEM_opts->for_code, "l_test");
				}
				else if (sp.nodal_coord_dir2 == -1)
				{
					std::string shapestr = femprint.FEM_opts->for_code->get_shape_info_str(sp.basis->get_space()) + "->"+ shapename;
					shapestr += "[l_test][" + std::to_string(dynamic_cast<pyoomph::D1XBasisFunction *>(sp.basis)->get_direction()) + "][" + (sp.is_derived_other_index ? "l_shape2" : "l_shape") + "][" + std::to_string(sp.nodal_coord_dir) + "]";
					c.s << shapestr;
				}
				else
				{
					std::string shapestr = femprint.FEM_opts->for_code->get_shape_info_str(sp.basis->get_space()) + "->"+ shapename2;
					shapestr += "[l_test][" + std::to_string(dynamic_cast<pyoomph::D1XBasisFunction *>(sp.basis)->get_direction()) + "][l_shape][" + std::to_string(sp.nodal_coord_dir) + "][l_shape2][" + std::to_string(sp.nodal_coord_dir2) + "]";
					c.s << shapestr;
				}
				return;
			}
		}
		else if (GiNaC::is_a<print_latex_FEM>(c))
		{
			const auto &femprint = dynamic_cast<const print_latex_FEM &>(c);
			if (femprint.FEM_opts->for_code && femprint.FEM_opts->for_code->latex_printer)
			{
				std::map<std::string, std::string> texinfo;
				texinfo["typ"] = "testfunction";
				texinfo["name"] = sp.field->get_name();
				texinfo["basis"] = sp.basis->to_string();
				texinfo["domain"] = sp.field->get_space()->get_code()->get_domain_name();
				texinfo["nodal_coord_dir"] = std::to_string(sp.nodal_coord_dir);
				texinfo["nodal_coord_dir2"] = std::to_string(sp.nodal_coord_dir2);
				texinfo["is_derived_other_index"] = (sp.is_derived_other_index ? "true" : "false");
				c.s << femprint.FEM_opts->for_code->latex_printer->_get_LaTeX_expression(texinfo, femprint.FEM_opts->for_code);
				return;
			}
		}
		else
		{
			c.s << "<" << (sp.nodal_coord_dir == -1 ? "" : "COORDINATE_DIFF_" + std::to_string(sp.nodal_coord_dir) + " ") << "TESTFUNC of " << sp.field->get_name() << " of " << sp.field->get_space()->get_code()->get_domain_name() << (sp.is_derived_other_index ? " wrt. l_shape2" : "") << " @ " << sp.basis->to_string() << ">"; //<< sp.basis->get_name() << ">";
		}
	}

	template <>
	GiNaC::ex GiNaCTestFunction::derivative(const GiNaC::symbol &s) const
	{
		const pyoomph::TestFunction &sp = get_struct();
		if (s == pyoomph::expressions::X)
		{
			if (dynamic_cast<const pyoomph::D0FiniteElementSpace *>(sp.basis->get_space()))
				return 0;
			else
				return GiNaCTestFunction(pyoomph::TestFunction(sp.field, sp.basis->get_diff_X(0)));
		}
		else if (s == pyoomph::expressions::Y)
		{
			if (dynamic_cast<const pyoomph::D0FiniteElementSpace *>(sp.basis->get_space()))
				return 0;
			else
				return GiNaCTestFunction(pyoomph::TestFunction(sp.field, sp.basis->get_diff_X(1)));
		}
		else if (s == pyoomph::expressions::Z)
		{
			if (dynamic_cast<const pyoomph::D0FiniteElementSpace *>(sp.basis->get_space()))
				return 0;
			else
				return GiNaCTestFunction(pyoomph::TestFunction(sp.field, sp.basis->get_diff_X(2)));
		}
		else if (s == pyoomph::expressions::x)
		{
			if (dynamic_cast<const pyoomph::D0FiniteElementSpace *>(sp.basis->get_space()))
				return 0;
			else
				return GiNaCTestFunction(pyoomph::TestFunction(sp.field, sp.basis->get_diff_x(0)));
		}
		else if (s == pyoomph::expressions::y)
		{
			if (dynamic_cast<const pyoomph::D0FiniteElementSpace *>(sp.basis->get_space()))
				return 0;
			else
				return GiNaCTestFunction(pyoomph::TestFunction(sp.field, sp.basis->get_diff_x(1)));
		}
		else if (s == pyoomph::expressions::z)
		{
			if (dynamic_cast<const pyoomph::D0FiniteElementSpace *>(sp.basis->get_space()))
				return 0;
			else
				return GiNaCTestFunction(pyoomph::TestFunction(sp.field, sp.basis->get_diff_x(2)));
		}
		else if (s == pyoomph::expressions::local_coordinate_1)
		{
			if (dynamic_cast<const pyoomph::D0FiniteElementSpace *>(sp.basis->get_space()))
				return 0;
			else
				return GiNaCTestFunction(pyoomph::TestFunction(sp.field, sp.basis->get_diff_S(0)));
		}
		else if (s == pyoomph::expressions::local_coordinate_2)
		{
			if (dynamic_cast<const pyoomph::D0FiniteElementSpace *>(sp.basis->get_space()))
				return 0;
			else
				return GiNaCTestFunction(pyoomph::TestFunction(sp.field, sp.basis->get_diff_S(1)));
		}
		else if (s == pyoomph::expressions::local_coordinate_3)
		{
			if (dynamic_cast<const pyoomph::D0FiniteElementSpace *>(sp.basis->get_space()))
				return 0;
			else
				return GiNaCTestFunction(pyoomph::TestFunction(sp.field, sp.basis->get_diff_S(2)));
		}			
		else if (s==pyoomph::expressions::zeta_coordinate_1 || s==pyoomph::expressions::zeta_coordinate_2 || s==pyoomph::expressions::zeta_coordinate_3)	
		{
			throw_runtime_error("Cannot derive with respect to zeta coordinates yet. This is not implemented in the code, as it is not needed for the current applications. If you need this, please contact the developers.");
		}
		else
		{
			std::ostringstream oss;
			oss << s;
			std::string sname = oss.str();
			if (dynamic_cast<pyoomph::D1XBasisFunction *>(sp.basis) && !(dynamic_cast<pyoomph::D1XBasisFunctionLagr *>(sp.basis)))
			{
				if (sname == "coordinate_x" || sname == "coordinate_y" || sname == "coordinate_z")
				{
					int coord_dir = (sname == "coordinate_x" ? 0 : (sname == "coordinate_y" ? 1 : 2));
					if (sp.basis->get_space()->get_code() == pyoomph::__current_code)
					{
						auto *posspace = pyoomph::__current_code->get_my_position_space();
						bool found = false;
						for (auto *f : pyoomph::__current_code->get_fields_on_space(posspace))
						{
							if (f->get_name() == sname)
							{
								if (f->get_symbol() == s)
								{
									found = true;
									break;
								}
							}
						}
						if (found)
						{
							if (dynamic_cast<const pyoomph::D0FiniteElementSpace *>(sp.basis->get_space()))
								return 0;
							else
							{
								if (sp.nodal_coord_dir >= 0)
								{
									return GiNaCTestFunction(pyoomph::TestFunction(sp.field, sp.basis, sp.nodal_coord_dir, pyoomph::__derive_shapes_by_second_index, coord_dir));
								}
								else
								{
									if (pyoomph::__ignore_dpsi_coord_diffs_in_jacobian) return 0;
									return GiNaCTestFunction(pyoomph::TestFunction(sp.field, sp.basis, coord_dir, pyoomph::__derive_shapes_by_second_index));
								}
							}
						}
					}
					else if (sp.basis->get_space()->get_code() == pyoomph::__current_code->get_bulk_element())
					{
						auto *posspace = pyoomph::__current_code->get_bulk_element()->get_my_position_space();
						bool found = false;
						for (auto *f : pyoomph::__current_code->get_bulk_element()->get_fields_on_space(posspace))
						{
							if (f->get_name() == sname)
							{
								if (f->get_symbol() == s)
								{
									found = true;
									break;
								}
							}
						}
						if (found)
						{
							if (dynamic_cast<const pyoomph::D0FiniteElementSpace *>(sp.basis->get_space()))
								return 0;
							else
							{
								if (sp.nodal_coord_dir >= 0)
								{
									GiNaCTestFunction(pyoomph::TestFunction(sp.field, sp.basis, sp.nodal_coord_dir, pyoomph::__derive_shapes_by_second_index, coord_dir));
								}
								else
								{
									if (pyoomph::__ignore_dpsi_coord_diffs_in_jacobian) return 0;
									return GiNaCTestFunction(pyoomph::TestFunction(sp.field, sp.basis, coord_dir, pyoomph::__derive_shapes_by_second_index));
								}
							}
						}
					}
					else if (sp.basis->get_space()->get_code() == pyoomph::__current_code->get_opposite_interface_code())
					{
						auto *posspace = pyoomph::__current_code->get_opposite_interface_code()->get_my_position_space();
						bool found = false;
						for (auto *f : pyoomph::__current_code->get_opposite_interface_code()->get_fields_on_space(posspace))
						{
							if (f->get_name() == sname)
							{
								if (f->get_symbol() == s)
								{
									found = true;
									break;
								}
							}
						}
						if (found)
						{
							if (dynamic_cast<const pyoomph::D0FiniteElementSpace *>(sp.basis->get_space()))
								return 0;
							else
							{
								if (sp.nodal_coord_dir >= 0)
								{
									return GiNaCTestFunction(pyoomph::TestFunction(sp.field, sp.basis, sp.nodal_coord_dir, pyoomph::__derive_shapes_by_second_index, coord_dir));
								}
								else
								{
									if (pyoomph::__ignore_dpsi_coord_diffs_in_jacobian) return 0;
									return GiNaCTestFunction(pyoomph::TestFunction(sp.field, sp.basis, coord_dir, pyoomph::__derive_shapes_by_second_index));
								}
							}
						}
					}
					else if (pyoomph::__current_code->get_opposite_interface_code() && sp.basis->get_space()->get_code() == pyoomph::__current_code->get_opposite_interface_code()->get_bulk_element())
					{
						auto *posspace = pyoomph::__current_code->get_opposite_interface_code()->get_bulk_element()->get_my_position_space();
						bool found = false;
						for (auto *f : pyoomph::__current_code->get_opposite_interface_code()->get_bulk_element()->get_fields_on_space(posspace))
						{
							if (f->get_name() == sname)
							{
								if (f->get_symbol() == s)
								{
									found = true;
									break;
								}
							}
						}
						if (found)
						{
							if (dynamic_cast<const pyoomph::D0FiniteElementSpace *>(sp.basis->get_space()))
								return 0;
							else
							{
								if (sp.nodal_coord_dir >= 0)
								{
									return GiNaCTestFunction(pyoomph::TestFunction(sp.field, sp.basis, sp.nodal_coord_dir, pyoomph::__derive_shapes_by_second_index, coord_dir));
								}
								else
								{
									if (pyoomph::__ignore_dpsi_coord_diffs_in_jacobian) return 0;
									return GiNaCTestFunction(pyoomph::TestFunction(sp.field, sp.basis, coord_dir, pyoomph::__derive_shapes_by_second_index));
								}
							}
						}
					}
				}
			}
		}
		/*  	Cannot be used now: For jacobian terms, we need to call derivative -> gives problems here
	std::ostringstream oss;
	oss << "Deriving: " << (*this) << "    with respect to  " << s ;
	throw_runtime_error("Cannot derive test function with respect to unknown symbol: Happend in: "+oss.str());
	  */
		return 0;
	}

}
