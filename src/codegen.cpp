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


#include "codegen.hpp"
#include "expressions.hpp"
#include "exception.hpp"
#include "problem.hpp"

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

	void print_simplest_form(GiNaC::ex expr, std::ostream &os, GiNaC::print_FEM_options &csrc_opts)
	{
		GiNaC::ex towrite;
		std::string mode = csrc_opts.for_code->ccode_expression_mode;

		if (mode == "factor")
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

	FiniteElementCode *__current_code;
	const ShapeExpansion *__deriv_subexpression_wrto = NULL;
	bool __derive_shapes_by_second_index = false;
	bool __in_pitchfork_symmetry_constraint = false;
	int * __derive_only_by_expansion_mode = NULL;
	bool __ignore_dpsi_coord_diffs_in_jacobian = false;
	std::set<ShapeExpansion> __all_Hessian_shapeexps;
	std::set<TestFunction> __all_Hessian_testfuncs;
	std::set<FiniteElementField *> __all_Hessian_indices_required;
	bool __in_hessian = false;

	bool ignore_nodal_position_derivatives_for_pitchfork_symmetry()
	{
		return __in_pitchfork_symmetry_constraint && !pyoomph::__derive_shapes_by_second_index;
	}

	


	class SubExpressionsToStructs : public GiNaC::map_function
	{
	protected:
		FiniteElementCode *code;

	public:
		std::vector<FiniteElementCodeSubExpression> subexpressions;
		SubExpressionsToStructs(FiniteElementCode *code_) : code(code_) {}
		GiNaC::ex operator()(const GiNaC::ex &inp)
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

	class MapOnTestSpace : public GiNaC::map_function
	{
	protected:
		FiniteElementSpace *space;
		std::string varname;
		FiniteElementField *field;

	public:
		FiniteElementField *get_field() { return field; }
		MapOnTestSpace(FiniteElementSpace *sp, std::string vn) : space(sp), varname(vn), field(NULL) {}
		GiNaC::ex operator()(const GiNaC::ex &inp)
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

	// This will map all history DoFs to the current
	class MakeResidualSteady : public GiNaC::map_function
	{
	protected:
		FiniteElementCode *code;
		bool extra_steady_routine;

	public:
		MakeResidualSteady(FiniteElementCode *_code) : code(_code), extra_steady_routine(false) {}
		GiNaC::ex operator()(const GiNaC::ex &inp)
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
			else
				return inp.map(*this);
		}

		bool require_extra_steady_routine() const { return extra_steady_routine; }
	};

	class GlobalParamsToValues : public GiNaC::map_function
	{
	public:
		GiNaC::ex operator()(const GiNaC::ex &inp)
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

	GiNaC::ex DrawUnitsOutOfSubexpressions::operator()(const GiNaC::ex &inp)
	{
		//	std::cout << "INP " <<inp << std::endl;
		if (is_ex_the_function(inp, expressions::subexpression))
		{
			if (pyoomph_verbose)
				std::cout << "INP SE:  " << inp << std::endl;
			GiNaC::ex factor, unit, rest;
			GiNaC::ex arg = inp.map(*this).op(0); // Descent recursively through nested subexpressions
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

	class RemapFieldsInExpression : public GiNaC::map_function
	{
	protected:
		std::map<FiniteElementField *, FiniteElementField *> remapping;

	public:
		RemapFieldsInExpression(std::map<FiniteElementField *, FiniteElementField *> remap) : remapping(remap) {}
		GiNaC::ex operator()(const GiNaC::ex &inp)
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

	std::string ShapeExpansion::get_dt_values_name(FiniteElementCode *forcode) const
	{
		std::string code_type = forcode->get_owner_prefix(this->basis->get_space());
		std::string dtstring = "d" + std::to_string(this->dt_order) + "t" + std::to_string(time_history_index);
		if (this->dt_order > 0)
			dtstring += this->dt_scheme;
		return code_type + dtstring + "_" + this->field->get_name();
	}

	std::string ShapeExpansion::get_timedisc_scheme(FiniteElementCode *forcode) const
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
		// if (!space->can_have_hanging_nodes()) return "No_Hang_Possible";
		return forcode->get_shape_info_str(space) + "->hanginfo_" + space->get_hang_name();
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

	std::string BasisFunction::get_c_varname(FiniteElementCode *forcode, std::string test_index)
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

	std::string BasisFunction::get_shape_string(FiniteElementCode *forcode, std::string nodal_index) const
	{
		return "shape_" + space->get_shape_name() + "[" + nodal_index + "]";
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

	BasisFunction *D1XBasisFunction::get_diff_x(unsigned direction)
	{
		throw_runtime_error("Cannot handle second order derivatives of basis functions yet");
	}

	BasisFunction *D1XBasisFunction::get_diff_X(unsigned direction)
	{
		throw_runtime_error("Cannot handle second order derivatives of basis functions yet");
	}

	BasisFunction *D1XBasisFunction::get_diff_S(unsigned direction)
	{
		throw_runtime_error("Cannot handle second order derivatives of basis functions yet");
	}

	std::string D1XBasisFunction::get_c_varname(FiniteElementCode *forcode, std::string test_index)
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

	std::string D1XBasisFunction::get_shape_string(FiniteElementCode *forcode, std::string nodal_index) const
	{
		return "dx_shape_" + space->get_shape_name() + "[" + nodal_index + "][" + std::to_string(direction) + "]";
	}

	std::string D1XBasisFunctionLagr::get_c_varname(FiniteElementCode *forcode, std::string test_index)
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

	std::string D1XBasisFunctionLagr::get_shape_string(FiniteElementCode *forcode, std::string nodal_index) const
	{
		return "dX_shape_" + space->get_shape_name() + "[" + nodal_index + "][" + std::to_string(direction) + "]";
	}



	std::string D1XBasisFunctionLocalCoord::get_c_varname(FiniteElementCode *forcode, std::string test_index)
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

	std::string D1XBasisFunctionLocalCoord::get_shape_string(FiniteElementCode *forcode, std::string nodal_index) const
	{
		return "dS_shape_" + space->get_shape_name() + "[" + nodal_index + "][" + std::to_string(direction) + "]";
	}



	std::string FiniteElementSpace::get_eqn_number_str(FiniteElementCode *forcode) const
	{
		std::string eleminfo = forcode->get_elem_info_str(this);
		return eleminfo + "->nodal_local_eqn";
	}

	std::string FiniteElementSpace::get_num_nodes_str(FiniteElementCode *forcode) const
	{
		std::string eleminfo = forcode->get_elem_info_str(this);
		return eleminfo + "->" + "nnode_" + this->get_shape_name();
	}

	std::string D0FiniteElementSpace::get_num_nodes_str(FiniteElementCode *forcode) const
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

	void FiniteElementSpace::write_nodal_time_interpolation(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, std::set<ShapeExpansion> &required_shapeexps)
	{
		bool hascontrib = false;
		std::string range = "";
		std::string shapeinfo = "";
		std::string eleminfo = "";
		std::set<std::string> handled;
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
				os << indent << "PYOOMPH_AQUIRE_ARRAY(double, " << varname << ", " << range << ")" << std::endl;
			}
		}

		if (!hascontrib)
			return;

		handled.clear();
		//		bool req_loop=this->need_interpolation_loop();
		os << indent << "for (unsigned int l_shape=0;l_shape<" + range + ";l_shape++)" << std::endl;
		os << indent << "{" << std::endl;

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
				os << indent << "  " << varname << "[l_shape]=0.0;" << std::endl;
			}
		}

		handled.clear();
		os << indent << "  for (unsigned tindex=0;tindex<" << "shapeinfo->timestepper_ntstorage;tindex++)" << std::endl;
		os << indent << "  {" << std::endl;

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
					os << indent << "    " << varname << "[l_shape] += " <<  "shapeinfo->timestepper_weights_dt_" << timedisc_scheme << "[tindex]*" << eleminfo << "->" << nds << "[l_shape][" << nodalindex << "][tindex];" << std::endl;
				}
				else if (s.dt_order == 2)
				{
					os << indent << "    " << varname << "[l_shape] += " <<   "shapeinfo->timestepper_weights_d2t_" << timedisc_scheme << "[tindex]*" << eleminfo << "->" << nds << "[l_shape][" << nodalindex << "][tindex];" << std::endl;
				}
				else
					throw_runtime_error("TODO Higher order time derivatives");
			}
		}

		os << indent << "  }" << std::endl;
		os << indent << "}" << std::endl;
	}

	void FiniteElementSpace::write_spatial_interpolation(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, std::set<ShapeExpansion> &required_shapeexps, bool including_nodal_diffs, bool for_hessian)
	{
		bool hascontrib = false;
		std::string range = "";
		std::string posrange = "";
		std::set<ShapeExpansion> required_coorddiffs;
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
			os << indent << "double " << varname << "=0.0;" << std::endl;
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

		os << indent << "for (unsigned int l_shape=0;l_shape<" + range + ";l_shape++)" << std::endl;
		os << indent << "{" << std::endl;
		for (auto &s : required_shapeexps)
		{
			if (s.basis->get_space() != this)
				continue;
			std::string varname = s.get_spatial_interpolation_name(for_code);
			std::string nodal_data = s.get_nodal_data_string(for_code, "l_shape");
			std::string shapestr = s.get_shape_string(for_code, "l_shape");
			os << indent << "  " << varname << "+= " << nodal_data << " * " << shapestr << ";" << std::endl;
		}
		os << indent << "}" << std::endl;

		if (!required_coorddiffs.empty())
		{
			for (auto s : required_coorddiffs)
			{
				std::string dtstring = "d" + std::to_string(s.dt_order) + "t" + std::to_string(s.time_history_index);
				if (s.dt_order > 0)
					dtstring += s.dt_scheme;
				for (unsigned int i = 0; i < for_code->nodal_dimension(); i++)
				{
					std::string code_type = for_code->get_owner_prefix(s.basis->get_space());
					std::string coorddiffname = code_type + "intrp_" + dtstring + "_" + s.basis->get_dx_str() + "_COORDDIFF_" + std::to_string(i) + "_" + s.field->get_name();
					os << indent << "PYOOMPH_AQUIRE_ARRAY(double," << coorddiffname << "," << posrange << ");" << std::endl;
				}
				if (for_hessian)
				{
					for (unsigned int i = 0; i < for_code->nodal_dimension(); i++)
					{
						for (unsigned int j = i; j < for_code->nodal_dimension(); j++) // TODO: Symmetrize? Go from j=i?
						{
							std::string code_type = for_code->get_owner_prefix(s.basis->get_space());
							std::string coorddiffname = code_type + "intrp_" + dtstring + "_" + s.basis->get_dx_str() + "_2ndCOORDDIFF_" + std::to_string(i) + "_" + std::to_string(j) + "_" + s.field->get_name();
							os << indent << "PYOOMPH_AQUIRE_TWO_D_ARRAY(double," << coorddiffname << "," << posrange << "," << posrange << ");" << std::endl;
						}
					}
				}
			}
			if (!for_hessian)
				os << indent << "if (flag)" << std::endl;
			os << indent << "{" << std::endl
			   << indent << " for (unsigned int m=0;m<" << posrange << ";m++)" << std::endl
			   << indent << " {" << std::endl;
			for (auto s : required_coorddiffs)
			{
				std::string dtstring = "d" + std::to_string(s.dt_order) + "t" + std::to_string(s.time_history_index);
				if (s.dt_order > 0)
					dtstring += s.dt_scheme;
				for (unsigned int i = 0; i < for_code->nodal_dimension(); i++)
				{
					std::string code_type = for_code->get_owner_prefix(s.basis->get_space());
					std::string coorddiffname = code_type + "intrp_" + dtstring + "_" + s.basis->get_dx_str() + "_COORDDIFF_" + std::to_string(i) + "_" + s.field->get_name();
					os << indent << "    " << coorddiffname << "[m]=0.0;" << std::endl;
				}
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
					std::string shapestr = for_code->get_shape_info_str(s.basis->get_space()) + "->d_dx_shape_dcoord_" + s.basis->get_space()->get_shape_name() + "[l_shape][" + std::to_string(dynamic_cast<D1XBasisFunction *>(s.basis)->get_direction()) + "][m][" + std::to_string(i) + "]";
					os << indent << "       " << coorddiffname << "[m]+=" << nodal_data << " * " << shapestr << ";" << std::endl;
				}
			}
			os << indent << "    }" << std::endl;
			if (for_hessian)
			{
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
							os << indent << "       " << coorddiffname << "[m][m2]=0.0;" << std::endl;
						}
					}
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
							std::string shapestr = for_code->get_shape_info_str(s.basis->get_space()) + "->d2_dx2_shape_dcoord_" + s.basis->get_space()->get_shape_name() + "[l_shape][" + std::to_string(dynamic_cast<D1XBasisFunction *>(s.basis)->get_direction()) + "][m][" + std::to_string(i) + "][m2][" + std::to_string(j) + "]";
							os << indent << "             " << coorddiffname << "[m][m2]+=" << nodal_data << " * " << shapestr << ";" << std::endl;
						}
					}
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

	void D0FiniteElementSpace::write_spatial_interpolation(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, std::set<ShapeExpansion> &required_shapeexps, bool including_nodal_diffs, bool for_hessian)
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

	bool PositionFiniteElementSpace::write_generic_Hessian_contribution(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, GiNaC::ex for_what, bool hanging_eqns)
	{
		// Only do it if the coordinates are Dofs
		if (for_code->coordinates_as_dofs)
			return FiniteElementSpace::write_generic_Hessian_contribution(for_code, os, indent, for_what, hanging_eqns);
		else
			return false;
	}

	void PositionFiniteElementSpace::write_generic_RJM_jacobian_contribution(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, GiNaC::ex for_what, bool hanging_eqns)
	{
		// Only do it if the coordinates are Dofs
		if (for_code->coordinates_as_dofs)
			FiniteElementSpace::write_generic_RJM_jacobian_contribution(for_code, os, indent, for_what, hanging_eqns);
	}

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
					if (for_code->get_opposite_interface_code())
					{
						jacobian_shapes.insert(ShapeExpansion(for_code->get_opposite_interface_code()->get_field_by_name("coordinate_" + d), 0, for_code->get_opposite_interface_code()->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
						if (for_code->get_opposite_interface_code()->get_bulk_element())
						{
							jacobian_shapes.insert(ShapeExpansion(for_code->get_opposite_interface_code()->get_bulk_element()->get_field_by_name("coordinate_" + d), 0, for_code->get_opposite_interface_code()->get_bulk_element()->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
						}
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
						if (for_code->get_opposite_interface_code())
						{
							hessian_shapes.insert(ShapeExpansion(for_code->get_opposite_interface_code()->get_field_by_name("coordinate_" + d), 0, for_code->get_opposite_interface_code()->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
							if (for_code->get_opposite_interface_code()->get_bulk_element())
							{
								hessian_shapes.insert(ShapeExpansion(for_code->get_opposite_interface_code()->get_bulk_element()->get_field_by_name("coordinate_" + d), 0, for_code->get_opposite_interface_code()->get_bulk_element()->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
							}
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
					std::string nodal_index2;
					std::string hang_info2;
					if (hang2)
					{
						nodal_index2 = f2->get_nodal_index_str(for_code);
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
							os << indent << "  BEGIN_HESSIAN_SHAPE_LOOP1_CONTINUOUS_SPACE(" << eqn_index << "," << hang_info << "," << nodal_index << "," << l_shape << ")" << std::endl;
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
						os << "," << hang_info2 << "," << nodal_index2 << "," << l_shape2 << ")" << std::endl;
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
						os << indent << "           ADD_TO_HESSIAN_" << (hanging_eqns ? "HANG" : "NOHANG") << "_" << (hang ? "HANG" : "NOHANG") << "_" << (hang2 ? "HANG" : "NOHANG") << "()" << std::endl;

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
			}

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
		return has_contribs;
	}

	void FiniteElementSpace::write_generic_RJM_jacobian_contribution(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, GiNaC::ex for_what, bool hanging_eqns)
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
					if (for_code->get_opposite_interface_code())
					{
						jacobian_shapes.insert(ShapeExpansion(for_code->get_opposite_interface_code()->get_field_by_name("coordinate_" + d), 0, for_code->get_opposite_interface_code()->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
						if (for_code->get_opposite_interface_code()->get_bulk_element())
						{
							jacobian_shapes.insert(ShapeExpansion(for_code->get_opposite_interface_code()->get_bulk_element()->get_field_by_name("coordinate_" + d), 0, for_code->get_opposite_interface_code()->get_bulk_element()->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
						}
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

			if (hang)
			{
				std::string nodal_index = f->get_nodal_index_str(for_code);
				std::string hang_info = f->get_hanginfo_str(for_code);
				os << indent << "  BEGIN_JACOBIAN_HANG(" << eqn_index << ", ";
				//if (for_code->get_derive_jacobian_by_expansion_mode())
				//{
				//	os << indent << " /* SYMBOLIC FORM  " << std::endl << diffpart << std::endl << " */ ";
				//}
				print_simplest_form(diffpart, os, csrc_opts);
				os << "," << hang_info << "," << nodal_index << "," << l_shape << ")" << std::endl;
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
					throw_runtime_error("Cannot add residual contributions on the position test space as long as the bulk element has not activated the positions as dofs (i.e. via calling BulkElement.activate_coordinates_as_dofs");
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

			oss << indent << "  double const * testfunction = " << shapeinfo << "->shape_" << this->get_shape_name() << ";" << std::endl;
			oss << indent << "  DX_SHAPE_FUNCTION_DECL(dx_testfunction) = " << shapeinfo << "->dx_shape_" << this->get_shape_name() << ";" << std::endl;
			oss << indent << "  DX_SHAPE_FUNCTION_DECL(dX_testfunction) = " << shapeinfo << "->dX_shape_" << this->get_shape_name() << ";" << std::endl;
			oss << indent << "  DX_SHAPE_FUNCTION_DECL(dS_testfunction) = " << shapeinfo << "->dS_shape_" << this->get_shape_name() << ";" << std::endl;

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
			std::string eqn_index = field->get_equation_str(for_code, l_test);
			std::string nodal_index = field->get_nodal_index_str(for_code);
			std::string hang_info = field->get_hanginfo_str(for_code);
			bool hessian_loop1_written = false;
			bool can_have_hanging = this->can_have_hanging_nodes() || for_code != this->code; // Always hang for external spaces
			if (!hessian)
			{
				has_contribs = true;
				if (can_have_hanging)
				{
					oss << indent << "    BEGIN_RESIDUAL_CONTINUOUS_SPACE(" << eqn_index << ",";
					if (for_code->is_current_residual_assembly_ignored())
					{
						oss << "0 /* IGNORED RESIDUAL: " << std::endl << var_part << std::endl << std::endl ;
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
					oss << ", " << hang_info << "," << nodal_index << "," << l_test << ")" << std::endl;
					oss << indent << "      ADD_TO_RESIDUAL_CONTINUOUS_SPACE()" << std::endl;
				}
				else
				{
					oss << indent << "    BEGIN_RESIDUAL(" << eqn_index << ", ";
					if (for_code->is_current_residual_assembly_ignored())
					{
						oss << "0 /* IGNORED RESIDUAL: " << std::endl << var_part << std::endl << std::endl ;
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
						if (for_code->get_opposite_interface_code())
						{
							jacobian_shapes.insert(ShapeExpansion(for_code->get_opposite_interface_code()->get_field_by_name("coordinate_" + d), 0, for_code->get_opposite_interface_code()->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
							if (for_code->get_opposite_interface_code()->get_bulk_element())
							{
								jacobian_shapes.insert(ShapeExpansion(for_code->get_opposite_interface_code()->get_bulk_element()->get_field_by_name("coordinate_" + d), 0, for_code->get_opposite_interface_code()->get_bulk_element()->get_field_by_name("coordinate_" + d)->get_space()->get_basis()));
							}
						}
					}
				}
			}
			if (!jacobian_shapes.empty())
			{
				if (!hessian)
					oss << indent << "      BEGIN_JACOBIAN()" << std::endl;
				std::set<FiniteElementField *> jacobian_fields;
				for (auto &s : jacobian_shapes)
					jacobian_fields.insert(s.field);
				std::set<FiniteElementSpace *> jacobian_spaces;
				for (auto *s : jacobian_fields)
					jacobian_spaces.insert(s->get_space());
				if (pyoomph_verbose)
					std::cout << "VAR PART IS " << var_part << std::endl;
				for (auto *s : jacobian_spaces)
				{
					if (pyoomph_verbose)
						std::cout << "writing contrib of domain " << s->get_code() << std::endl;
					if (!hessian)
						s->write_generic_RJM_jacobian_contribution(for_code, oss, indent + "        ", var_part, can_have_hanging);
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
									oss << indent << "    BEGIN_HESSIAN_TEST_LOOP_CONTINUOUS_SPACE(" << eqn_index << ", " << hang_info << "," << nodal_index << "," << l_test << ")" << std::endl;
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

	FiniteElementCode::FiniteElementCode() : residual_index(0), residual_names({""}), equations(NULL), bulk_code(NULL), opposite_interface_code(NULL), residual(std::vector<GiNaC::ex>{0}), dx(this, false), dX(this, true),dx_unity(this, false), elemsize_Eulerian(this, false, true), elemsize_Lagrangian(this, true, true), elemsize_Eulerian_Cart(this, false, false), elemsize_Lagrangian_Cart(this, true, false), nodal_delta(this), stage(0), nodal_dim(0), lagr_dim(0), coordinate_sys(&__no_coordinate_system), _x(GiNaC::indexed(GiNaC::potential_real_symbol("interpolated_x"), GiNaC::idx(0, 3))),
											 _y(GiNaC::indexed(GiNaC::potential_real_symbol("interpolated_x"), GiNaC::idx(1, 3))), _z(GiNaC::indexed(GiNaC::potential_real_symbol("interpolated_x"))), integration_order(0), IC_names({""}), element_dim(-1), analytical_jacobian(true), analytical_position_jacobian(true), debug_jacobian_epsilon(0.0), with_adaptivity(true),
											 coordinates_as_dofs(false), generate_hessian(false), assemble_hessian_by_symmetry(true), coordinate_space(""), stop_on_jacobian_difference(false), latex_printer(NULL), has_constant_mass_matrix_for_sure(std::vector<bool>{false})
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
	}

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

	
	class MeshToCoordinateShapes : public GiNaC::map_function
	{
	protected:
		FiniteElementCode *code;

	public:
		MeshToCoordinateShapes(FiniteElementCode *code_) : code(code_) {}
		GiNaC::ex operator()(const GiNaC::ex &inp)
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

	void FiniteElementCode::add_residual(GiNaC::ex add, bool allow_contributions_without_dx)
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
			os << indent << "  const double dx = shapeinfo->int_pt_weight;" << std::endl;
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

	GiNaC::ex FiniteElementCode::write_code_subexpressions(std::ostream &os, std::string indent, GiNaC::ex for_what, const std::set<ShapeExpansion> &required_shapeexps, bool hessian)
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
				}
				else if (this->bulk_code && sp.get_code() == this->bulk_code)
				{
					this->mark_shapes_required(for_what, this->bulk_code->get_my_position_space(), "normal");
				}
				else if (this->opposite_interface_code && sp.get_code() == this->opposite_interface_code)
				{
					this->mark_shapes_required(for_what, this->opposite_interface_code->get_my_position_space(), "normal");
				}
				else
				{
					throw_runtime_error("Normal of this domain not accessible");
				}
			}
			if (GiNaC::is_a<GiNaC::GiNaCElementSizeSymbol>(*i))
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

	bool FiniteElementCode::write_generic_Hessian(std::ostream &os, std::string funcname, GiNaC::ex resi, bool with_hang)
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
					if (this->opposite_interface_code)
					{
						indices_required.insert(this->opposite_interface_code->get_field_by_name("coordinate_" + d));
						if (this->opposite_interface_code->bulk_code)
						{
							indices_required.insert(this->opposite_interface_code->bulk_code->get_field_by_name("coordinate_" + d));
						}
					}
				}
			}
		}

		for (auto *f : indices_required)
		{
			osh << "  const unsigned " << f->get_nodal_index_str(this) << " = " << f->index << ";" << std::endl;
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

	void FiniteElementCode::write_generic_RJM(std::ostream &os, std::string funcname, GiNaC::ex resi, bool with_hang)
	{
		__in_hessian = false;
		os << "static void " << funcname << "(const JITElementInfo_t * eleminfo, const JITShapeInfo_t * shapeinfo,double * residuals, double *jacobian, double *mass_matrix,unsigned flag)" << std::endl;
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
		/*for (GiNaC::const_preorder_iterator i = resi.preorder_begin(); i != resi.preorder_end(); ++i)
		{
		 if (GiNaC::is_a<GiNaC::GiNaCNormalSymbol>(*i))
		 {
			   this->mark_shapes_required("ResJac",NULL,"normal");
		 }
		}*/

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
					if (this->opposite_interface_code)
					{
						indices_required.insert(this->opposite_interface_code->get_field_by_name("coordinate_" + d));
						if (this->opposite_interface_code->bulk_code)
						{
							indices_required.insert(this->opposite_interface_code->bulk_code->get_field_by_name("coordinate_" + d));
						}
					}
				}
			}
		}

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

	void FiniteElementCode::write_code_header(std::ostream &os)
	{
		os << "#define JIT_ELEMENT_SHARED_LIB" << std::endl;
		if (this->assemble_hessian_by_symmetry)
		{
			os << "#define ASSEMBLE_HESSIAN_VIA_SYMMETRY" << std::endl;
		}
		os << "#include \"jitbridge.h\"" << std::endl
		   << std::endl;
		os << "static JITFuncSpec_Table_FiniteElement_t * my_func_table;" << std::endl;
		os << "#include \"jitbridge_hang.h\"" << std::endl
		   << std::endl;
	}

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

		os << "    const double dx = shapeinfo->int_pt_weight;" << std::endl;
		os << "    const double dX = shapeinfo->int_pt_weight_Lagrangian;" << std::endl;
		os << "    const double dx_unity = shapeinfo->int_pt_weight_unity;" << std::endl;
		os << "    switch (index)" << std::endl;
		os << "    {" << std::endl;
		unsigned index = 0;
		for (auto &e : sexprs)
		{
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

		os << "    const double dx = shapeinfo->int_pt_weight;" << std::endl; // TODO: Lagrangian part
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

	void FiniteElementCode::write_code(std::ostream &os)
	{
		__current_code = this;
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

	void FiniteElementCode::set_temporal_error(std::string f, double factor)
	{
		auto *field = this->get_field_by_name(f);
		if (!field)
		{
			throw_runtime_error("Cannot set temporal error of an undefined field: " + f);
		}
		field->temporal_error_factor = factor;
	}

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

		for (unsigned int i = 0; i < this->element_dim; i++)
		{
			std::vector<std::string> dir{"1", "2", "3"};
			this->register_field("local_coordinate_" + dir[i], "Pos")->no_jacobian_at_all = true; // Lagrangian coordinates never have Jacobian entries, since they are fixed
		}		

		if (this->bulk_code && !this->bulk_code->bulk_code) 
		{
			// Only do this on co-dim 1 interfaces for now, then zetas are unique
			for (unsigned int i = 0; i < this->element_dim; i++)
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
				// Write the stuff like normal and so on
				for (auto &subentry : fieldentry.second)
				{
					if (subentry.second)
					{
						os << indent << "functable->shapes_required_" << func_type << "." << subentry.first << " = true;" << std::endl;
					}
				}
				continue;
			}

			if (pyoomph_verbose)
				std::cout << "REQUIRED FIELDS FOR " << func_type << "  " << fieldentry.first->get_name() << "  code " << fieldentry.first->get_code() << " [this " << this << "]" << std::endl;
			bool is_in_my_space = false;
			for (auto &s : spaces)
			{
				if (s == fieldentry.first)
				{
					is_in_my_space = true;
					break;
				}
			}
			if (!is_in_my_space)
			{
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
					if (pyoomph_verbose)
						std::cout << "IIIII Test if func in bulk space " << fieldentry.first << " returns in: " << require_bulk << std::endl;
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
						if (pyoomph_verbose)
							std::cout << "IIIII Test if func in bulk->bulk space " << fieldentry.first << " returns in: " << require_bulk_bulk << std::endl;
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
					if (pyoomph_verbose)
						std::cout << "IIIII Test if func in opposite interface space " << fieldentry.first << " returns in: " << require_opposite_interface << std::endl;
				}
				if (!found_elsewhere && opposite_interface_code && opposite_interface_code->bulk_code)
				{
					for (auto &s : opposite_interface_code->bulk_code->spaces)
					{
						if (s == fieldentry.first)
						{
							require_opposite_bulk = true;
							found_elsewhere = true;
							break;
						}
					}
					if (pyoomph_verbose)
						std::cout << "IIIII Test if func in opposite interface space " << fieldentry.first << " returns in: " << require_opposite_bulk << std::endl;
				}
				if (!found_elsewhere)
				{
					std::ostringstream oss;
					oss << "Cannot find a required space " << fieldentry.first;
					throw_runtime_error(oss.str());
				}
				continue;
			}
			for (auto &psientry : fieldentry.second)
			{
				if (psientry.second)
				{
					os << indent << "functable->shapes_required_" << func_type << "." << psientry.first << "_" << fieldentry.first->get_name() << " = true;" << std::endl;
				}
			}
		}

		// Check if we need the bulk for the normal
		bool just_the_normal = false;
		if (!require_bulk && bulk_code)
		{
			if (this->required_shapes[func_type].count(get_my_position_space()) && this->required_shapes[func_type][get_my_position_space()].count("normal"))
			{
				just_the_normal = true;
				require_bulk = true;
			}
		}

		bool just_the_parent_normal = false;
		if (!require_bulk_bulk && bulk_code && bulk_code->bulk_code)
		{
			if (this->required_shapes[func_type].count(bulk_code->get_my_position_space()) && this->required_shapes[func_type][bulk_code->get_my_position_space()].count("normal"))
			{
				just_the_parent_normal = true;
				require_bulk = true;
				require_bulk_bulk = true;
			}
		}

		// Check if we need the bulk for the normal
		bool just_the_opposite_normal = false;
		if (!require_opposite_interface && opposite_interface_code)
		{
			if (this->required_shapes[func_type].count(opposite_interface_code->get_my_position_space()) && this->required_shapes[func_type][opposite_interface_code->get_my_position_space()].count("normal"))
			{
				just_the_opposite_normal = true;
				require_opposite_interface = true;
			}
		}

		if (require_bulk)
		{
			os << " functable->shapes_required_" << func_type << ".bulk_shapes=(JITFuncSpec_RequiredShapes_FiniteElement_t*)calloc(sizeof(JITFuncSpec_RequiredShapes_FiniteElement_t),1);" << std::endl;
			for (auto &fieldentry : entry)
			{
				bool is_in_my_space = false;
				for (auto &s : bulk_code->spaces)
				{
					if (s == fieldentry.first)
					{
						is_in_my_space = true;
						break;
					}
				}
				if (!is_in_my_space)
				{
					continue;
				}
				for (auto &psientry : fieldentry.second)
				{
					if (psientry.second)
					{
						os << indent << "functable->shapes_required_" << func_type << ".bulk_shapes->" << psientry.first << "_" << fieldentry.first->get_name() << " = true;" << std::endl;
					}
				}
			}

			if (just_the_normal)
			{
				os << indent << "functable->shapes_required_" << func_type << ".bulk_shapes->psi_Pos = true;" << std::endl;
			}

			if (require_bulk_bulk)
			{

				os << " functable->shapes_required_" << func_type << ".bulk_shapes->bulk_shapes=(JITFuncSpec_RequiredShapes_FiniteElement_t*)calloc(sizeof(JITFuncSpec_RequiredShapes_FiniteElement_t),1);" << std::endl;
				for (auto &fieldentry : entry)
				{
					bool is_in_my_space = false;
					for (auto &s : bulk_code->bulk_code->spaces)
					{
						if (s == fieldentry.first)
						{
							is_in_my_space = true;
							break;
						}
					}
					if (!is_in_my_space)
					{
						continue;
					}
					for (auto &psientry : fieldentry.second)
					{
						if (psientry.second)
						{
							os << indent << "functable->shapes_required_" << func_type << ".bulk_shapes->bulk_shapes->" << psientry.first << "_" << fieldentry.first->get_name() << " = true;" << std::endl;
						}
					}
				}
				if (just_the_parent_normal)
				{
					os << indent << "functable->shapes_required_" << func_type << ".bulk_shapes->bulk_shapes->psi_Pos = true;" << std::endl;
				}
			}
		}

		/* //Opposite side of the interface
		just_the_normal=false;
		if (!require_opposite_interface && opposite_interface_code) //TODO: THis required here?
		{
		 if (this->required_shapes[func_type].count(NULL) && this->required_shapes[func_type][NULL].count("normal"))
		 {
		  just_the_normal=true;
		  require_bulk=true;
		 }
		}*/

		if (require_opposite_interface || require_opposite_bulk)
		{
			os << " functable->shapes_required_" << func_type << ".opposite_shapes=(JITFuncSpec_RequiredShapes_FiniteElement_t*)calloc(sizeof(JITFuncSpec_RequiredShapes_FiniteElement_t),1);" << std::endl;

			for (auto &fieldentry : entry)
			{
				bool is_in_my_space = false;
				for (auto &s : opposite_interface_code->spaces)
				{
					if (s == fieldentry.first)
					{
						is_in_my_space = true;
						break;
					}
				}
				if (!is_in_my_space)
				{
					continue;
				}
				for (auto &psientry : fieldentry.second)
				{
					if (psientry.second)
					{
						os << indent << "functable->shapes_required_" << func_type << ".opposite_shapes->" << psientry.first << "_" << fieldentry.first->get_name() << " = true;" << std::endl;
					}
				}
			}

			if (just_the_opposite_normal)
			{
				os << indent << "functable->shapes_required_" << func_type << ".opposite_shapes->psi_Pos = true;" << std::endl;
			}
			/*
			if (just_the_normal) //TODO: THis required here?
			{
			  os << indent << "functable->shapes_required_"  << func_type << ".bulk_shapes->psi_Pos = true;" << std::endl;
			}
			*/
		}

		if (require_opposite_bulk)
		{

			os << " functable->shapes_required_" << func_type << ".opposite_shapes->bulk_shapes=(JITFuncSpec_RequiredShapes_FiniteElement_t*)calloc(sizeof(JITFuncSpec_RequiredShapes_FiniteElement_t),1);" << std::endl;

			for (auto &fieldentry : entry)
			{
				bool is_in_my_space = false;
				for (auto &s : opposite_interface_code->bulk_code->spaces)
				{
					if (s == fieldentry.first)
					{
						is_in_my_space = true;
						break;
					}
				}
				if (!is_in_my_space)
				{
					continue;
				}
				for (auto &psientry : fieldentry.second)
				{
					if (psientry.second)
					{
						os << indent << "functable->shapes_required_" << func_type << ".opposite_shapes->bulk_shapes->" << psientry.first << "_" << fieldentry.first->get_name() << " = true;" << std::endl;
					}
				}
			}
			/*
			if (just_the_normal) //TODO: THis required here?
			{
			  os << indent << "functable->shapes_required_"  << func_type << ".bulk_shapes->psi_Pos = true;" << std::endl;
			}
			*/
		}
	}

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
			std::vector<std::string> dirindex = {"x", "y", "z"};
			std::vector<std::string> res;
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
			return res;
		}
	}

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
				init << " functable->numfields_" << space->get_name() << "=" << numfields << ";" << std::endl;
				init << " functable->fieldnames_" << space->get_name() << "=(char **)malloc(sizeof(char*)*functable->numfields_" << space->get_name() << ");" << std::endl;
				for (auto &f : myfields)
				{
					if (f->get_space() != space)
						continue;
					if (f->get_name() == "mesh_x" || f->get_name() == "mesh_y" || f->get_name() == "mesh_z")
						continue;
					init << " SET_INTERNAL_FIELD_NAME(functable->fieldnames_" << space->get_name() << "," << (f->index - index_offset) << ", \"" << f->get_name() << "\" );" << std::endl;
					cleanup << " pyoomph_tested_free(functable->fieldnames_" << space->get_name() << "[ " << (f->index - index_offset) <<"]); functable->fieldnames_" << space->get_name()<< "[" <<(f->index - index_offset) << "]=PYOOMPH_NULL; " << std::endl;
				}
				cleanup << " pyoomph_tested_free(functable->fieldnames_" << space->get_name() << "); functable->fieldnames_" << space->get_name() << "=PYOOMPH_NULL; " << std::endl;
				index_offset += numfields;
			}
		}

		bool coordinate_space_validated = false;
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
						throw_runtime_error("Cannot use a coordinate space on " + coordinate_space + ", which is inferior to the required nodal field space " + space->get_name());
					}
					else
						coordinate_space_validated = true;
				}
				init << " functable->numfields_" << space->get_name() << "=" << numfields << ";" << std::endl;

				if (dynamic_cast<ContinuousFiniteElementSpace *>(space) || dynamic_cast<DGFiniteElementSpace *>(space))
				{
					// Find out the fields which are really defined on the bulk
					// Other fields stem from the interface
					if (!bulk_code)
					{
						init << " functable->numfields_" << space->get_name() << "_bulk=" << numfields << ";" << std::endl;
						init << " functable->numfields_" << space->get_name() << "_basebulk=" << numfields << ";" << std::endl;
						init << " functable->numfields_" << space->get_name() << "_new=" << numfields << ";" << std::endl;
						if (dynamic_cast<ContinuousFiniteElementSpace *>(space))
						{
							init << " functable->nodal_offset_" << space->get_name() << "_basebulk =" << base_bulk_nodal_offset << ";" << std::endl;
						}
						else if (dynamic_cast<DGFiniteElementSpace *>(space))
						{
							init << " functable->internal_offset_" << space->get_name() << "_new =" << internal_data_offset << ";" << std::endl;
							internal_data_offset += numfields;
						}
						init << " functable->buffer_offset_" << space->get_name() << "_basebulk =" << base_bulk_nodal_offset << ";" << std::endl;
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

						init << " functable->numfields_" << space->get_name() << "_bulk=" << ncbulk << ";" << std::endl;
						init << " functable->numfields_" << space->get_name() << "_basebulk=" << ncbasebulk << ";" << std::endl;
						init << " functable->numfields_" << space->get_name() << "_new=" << numfields - ncbulk << ";" << std::endl;
						if (dynamic_cast<ContinuousFiniteElementSpace *>(space))
						{
							init << " functable->nodal_offset_" << space->get_name() << "_basebulk =" << base_bulk_nodal_offset << ";" << std::endl;
						}
						else if (dynamic_cast<DGFiniteElementSpace *>(space))
						{
							init << " functable->internal_offset_" << space->get_name() << "_new =" << internal_data_offset << ";" << std::endl;
							internal_data_offset += (numfields - ncbulk);
							init << " functable->external_offset_" << space->get_name() << "_bulk = " << DG_external_offset << ";" << std::endl;
							DG_external_offset += ncbulk;
						}
						init << " functable->buffer_offset_" << space->get_name() << "_basebulk =" << base_bulk_nodal_offset << ";" << std::endl;
						base_bulk_nodal_offset += ncbasebulk;
					}
				}
				else if (dynamic_cast<DiscontinuousFiniteElementSpace *>(space))
				{
					init << " functable->buffer_offset_" << space->get_name() << " =" << index_offset << ";" << std::endl;
					if (!dynamic_cast<ExternalD0Space *>(space))
					{
						init << " functable->internal_offset_" << space->get_name() << " =" << internal_data_offset << ";" << std::endl;
						internal_data_offset += numfields;
					}
					else
					{
						init << " functable->external_offset_" << space->get_name() << " = " << DG_external_offset << ";" << std::endl;
						DG_external_offset += numfields;
					}
				}

				init << " functable->fieldnames_" << space->get_name() << "=(char **)malloc(sizeof(char*)*functable->numfields_" << space->get_name() << ");" << std::endl;
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
					init << " SET_INTERNAL_FIELD_NAME(functable->fieldnames_" << space->get_name() << "," << contiindex << ", \"" << f->get_name() << "\" );" << std::endl;
					cleanup << " pyoomph_tested_free(functable->fieldnames_" << space->get_name() << "[ " << (contiindex) <<"]); functable->fieldnames_" << space->get_name()<<    "[" <<(contiindex) << "]=PYOOMPH_NULL; " << std::endl;
				}
				cleanup << " pyoomph_tested_free(functable->fieldnames_" << space->get_name() << "); functable->fieldnames_" << space->get_name() << "=PYOOMPH_NULL; " << std::endl;								
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
			init << " functable->buffer_offset_C2TB_interf=functable->numfields_C2TB_basebulk+functable->numfields_C2_basebulk+functable->numfields_C1TB_basebulk+functable->numfields_C1_basebulk" << std::endl;
			init << "                                     +functable->numfields_D2TB_basebulk+functable->numfields_D2_basebulk+functable->numfields_D1TB_basebulk+functable->numfields_D1_basebulk;" << std::endl;
			init << " functable->buffer_offset_C2_interf=functable->buffer_offset_C2TB_interf+(functable->numfields_C2TB-functable->numfields_C2TB_basebulk);" << std::endl;
			init << " functable->buffer_offset_C1TB_interf=functable->buffer_offset_C2_interf+(functable->numfields_C2-functable->numfields_C2_basebulk);" << std::endl;
			init << " functable->buffer_offset_C1_interf=functable->buffer_offset_C1TB_interf+(functable->numfields_C1TB-functable->numfields_C1TB_basebulk);" << std::endl;
			init << " functable->buffer_offset_D2TB_interf=functable->buffer_offset_C1_interf+(functable->numfields_C1-functable->numfields_C1_basebulk);" << std::endl;
			init << " functable->buffer_offset_D2_interf=functable->buffer_offset_D2TB_interf+(functable->numfields_D2TB-functable->numfields_D2TB_basebulk);" << std::endl;
			init << " functable->buffer_offset_D1TB_interf=functable->buffer_offset_D2_interf+(functable->numfields_D2-functable->numfields_D2_basebulk);" << std::endl;
			init << " functable->buffer_offset_D1_interf=functable->buffer_offset_D1TB_interf+(functable->numfields_D1TB-functable->numfields_D1TB_basebulk);" << std::endl;
			init << "#ifndef PYOOMPH_TCC_TO_MEMORY" << std::endl;
			init << " if (functable->buffer_offset_D1_interf+(functable->numfields_D1-functable->numfields_D1_basebulk)+functable->numfields_DL+functable->numfields_D0+functable->numfields_ED0!=" << index_offset << ")" << std::endl;
			init << " {" << std::endl;
			init << "   printf(\"Error in the buffer offsets. Please report with the script you have used to create this error!\\nbuffer_offset_C2TB_interf=%d\\nbuffer_offset_C2_interf=%d\\nbuffer_offset_C1TB_interf=%d\\nbuffer_offset_C1_interf=%d\\nbuffer_offset_D2TB_interf=%d\\nbuffer_offset_D2_interf=%d\\nbuffer_offset_D1TB_interf=%d\\nbuffer_offset_D1_interf=%d\\n\",functable->buffer_offset_C2TB_interf,functable->buffer_offset_C2_interf,functable->buffer_offset_C1TB_interf,functable->buffer_offset_C1_interf,functable->buffer_offset_D2TB_interf,functable->buffer_offset_D2_interf,functable->buffer_offset_D1TB_interf,functable->buffer_offset_D1_interf);" << std::endl;
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
		if (coordinate_space == "" || coordinate_space=="ED0")
			throw_runtime_error("Cannot deduce the coordinate space of domain " + this->get_domain_name() + ". Please specify it explicitly by adding an ElementSpace().");
		//   if (coordinate_space=="C2TB" && this->bulk_code) coordinate_space="C2";
		init << " functable->dominant_space=strdup(\"" << coordinate_space << "\");" << std::endl;

		init << " functable->hangindex_Pos=-1; //Position always hangs on the max space" << std::endl;
		if (coordinate_space == "C1" || coordinate_space == "C1TB")
		{
			init << " functable->hangindex_C2TB=-1;" << std::endl;
			init << " functable->hangindex_C2=-1;" << std::endl;
			init << " functable->hangindex_C1TB=-1;" << std::endl;
			init << " functable->hangindex_C1=-1;" << std::endl;
		}
		else
		{
			init << " functable->hangindex_C2TB=-1;" << std::endl;
			init << " functable->hangindex_C2=-1;" << std::endl;
			init << " functable->hangindex_C1TB=functable->numfields_C2TB_basebulk+functable->numfields_C2_basebulk;" << std::endl;
			init << " functable->hangindex_C1=functable->numfields_C2TB_basebulk+functable->numfields_C2_basebulk;" << std::endl;
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
		if (this->bulk_position_space_to_C1)
		{
			init << "   functable->bulk_position_space_to_C1=true;" << std::endl;
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

	void FiniteElementCode::set_discontinuous_refinement_exponent(std::string field, double exponent)
	{
		auto *f = this->get_field_by_name(field);
		f->discontinuous_refinement_exponent = exponent;
	}

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

	template <>
	void GiNaCSubExpression::print(const print_context &c, unsigned level) const
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

	template <>
	void GiNaCMultiRetCallback::print(const print_context &c, unsigned level) const
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

	template <>
	void GiNaCNodalDeltaSymbol::print(const print_context &c, unsigned level) const
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
	GiNaC::ex GiNaCNodalDeltaSymbol::derivative(const GiNaC::symbol &s) const
	{
		return 0;
	}

	template <>
	void GiNaCSpatialIntegralSymbol::print(const print_context &c, unsigned level) const
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
					c.s << "dx";
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
				c.s << "<DX"+modestr+">";
			}
		}
	}
	template <>
	GiNaC::ex GiNaCSpatialIntegralSymbol::derivative(const GiNaC::symbol &s) const
	{
		if (get_struct().is_lagrangian() || get_struct().simple_unity_integral)
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

	template <>
	void GiNaCElementSizeSymbol::print(const print_context &c, unsigned level) const
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
	void GiNaCNormalSymbol::print(const print_context &c, unsigned level) const
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
	void GiNaCShapeExpansion::print(const print_context &c, unsigned level) const
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
								throw_runtime_error("DD");
							if (sp.nodal_coord_dir >= 0)
							{
								std::string shapestr = femprint.FEM_opts->for_code->get_shape_info_str(sp.basis->get_space()) + "->d_dx_shape_dcoord_" + sp.basis->get_space()->get_shape_name();
								shapestr += "[l_shape2][" + std::to_string(dynamic_cast<pyoomph::D1XBasisFunction *>(sp.basis)->get_direction()) + "][l_shape][" + std::to_string(sp.nodal_coord_dir) + "]";
								c.s << shapestr;
							}
							else
							{
								std::string shapestr = femprint.FEM_opts->for_code->get_shape_info_str(sp.basis->get_space()) + "->d_dx_shape_dcoord_" + sp.basis->get_space()->get_shape_name();
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
	void GiNaCTestFunction::print(const print_context &c, unsigned level) const
	{
		const pyoomph::TestFunction &sp = get_struct();
		if (GiNaC::is_a<print_csrc_FEM>(c))
		{
			const auto &femprint = dynamic_cast<const print_csrc_FEM &>(c);
			if (femprint.FEM_opts->for_code)
			{
				if (sp.nodal_coord_dir == -1)
				{
					c.s << sp.basis->get_c_varname(femprint.FEM_opts->for_code, "l_test");
				}
				else if (sp.nodal_coord_dir2 == -1)
				{
					std::string shapestr = femprint.FEM_opts->for_code->get_shape_info_str(sp.basis->get_space()) + "->d_dx_shape_dcoord_" + sp.basis->get_space()->get_shape_name();
					shapestr += "[l_test][" + std::to_string(dynamic_cast<pyoomph::D1XBasisFunction *>(sp.basis)->get_direction()) + "][" + (sp.is_derived_other_index ? "l_shape2" : "l_shape") + "][" + std::to_string(sp.nodal_coord_dir) + "]";
					c.s << shapestr;
				}
				else
				{
					std::string shapestr = femprint.FEM_opts->for_code->get_shape_info_str(sp.basis->get_space()) + "->d2_dx2_shape_dcoord_" + sp.basis->get_space()->get_shape_name();
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
