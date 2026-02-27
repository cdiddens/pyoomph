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

#include "expressions.hpp"
#include "exception.hpp"
#include "codegen.hpp"
#include "problem.hpp"
#include <cassert>
#include <sstream>
#include <limits>

using namespace GiNaC;

namespace GiNaC
{

	

	/////////////////

	template <>
	void GiNaCPlaceHolderResolveInfo::print(const print_context &c, unsigned level) const
	{
		const pyoomph::PlaceHolderResolveInfo &sp = get_struct();
		c.s << "< code=" << sp.code << " , tags=";
		for (unsigned int i = 0; i < sp.tags.size(); i++)
			c.s << (i == 0 ? "" : ", ") << sp.tags[i];
		c.s << ">";
	}

	template <>
	void GiNaCCustomMultiReturnExpressionWrapper::print(const print_context &c, unsigned level) const
	{
		const pyoomph::CustomMultiReturnExpressionWrapper &sp = get_struct();
		c.s << "<" << sp.cme->get_id_name() << " @" << sp.cme << ">";
	}

	template <>
	void GiNaCCustomMultiReturnExpressionResultSymbol::print(const print_context &c, unsigned level) const
	{
		const pyoomph::CustomMultiReturnExpressionResultSymbol &sp = get_struct();
		c.s << "<CB_RESULT " << sp.index << " of " << sp.func->get_id_name() << " called with " << sp.arglist << ">";
	}

	template <>
	void GiNaCCustomMathExpressionWrapper::print(const print_context &c, unsigned level) const
	{
		//        std::cout << "ENTERING CME "<<std::flush <<std::endl;
		//        c.s << "CME" <<std::endl;
		//       std::cout << "RET CME "<<std::flush <<std::endl;
		//       return;
		const pyoomph::CustomMathExpressionWrapper &sp = get_struct();
		c.s << "<" << sp.cme->get_id_name() << " @" << sp.cme << ">";
	}

	template <>
	void GiNaCCustomCoordinateSystemWrapper::print(const print_context &c, unsigned level) const
	{
		const pyoomph::CustomCoordinateSystemWrapper &sp = get_struct();
		c.s << "<" << sp.cme->get_id_name() << " @" << sp.cme << ">";
	}

	template <>
	void GiNaCGlobalParameterWrapper::print(const print_context &c, unsigned level) const
	{
		const pyoomph::GlobalParameterWrapper &sp = get_struct();
		if (dynamic_cast<const print_csrc *>(&c))
		{
			unsigned local_index;
			if (pyoomph::pyoomph_verbose)
				std::cout << "CURRENT CODE " << pyoomph::__current_code << std::endl;
			if (pyoomph::__current_code->global_parameter_to_local_indices.count(sp.cme->get_global_index()))
			{
				local_index = pyoomph::__current_code->global_parameter_to_local_indices[sp.cme->get_global_index()];
			}
			else
			{
				local_index = pyoomph::__current_code->global_parameter_to_local_indices.size();
				pyoomph::__current_code->local_parameter_symbols.push_back(*this);
				pyoomph::__current_code->global_parameter_to_local_indices.insert(std::pair<unsigned, unsigned>(sp.cme->get_global_index(), local_index));
			}
			c.s << "(*(my_func_table->global_parameters[" << local_index << "]))";
		}
		else
		{
			c.s << "<global param: " << sp.cme->get_name() << ">";
		}
	}

	template <>
	bool GiNaCGlobalParameterWrapper::info(unsigned inf) const
	{		
		if (inf == info_flags::real)
			return true;
		if (inf==info_flags::positive || inf==info_flags::nonnegative)
			return get_struct().cme->is_restricted_to_positive_values();
		else
			return inherited::info(inf);
	}

	template <>
	void GiNaCDelayedPythonCallbackExpansion::print(const print_context &c, unsigned level) const
	{
		// const pyoomph::DelayedPythonCallbackExpansionWrapper &sp = get_struct();
		if (dynamic_cast<const print_csrc *>(&c))
		{
			throw_runtime_error("Should not happen");
		}
		else
		{
			c.s << "<delayed lambda>";
		}
	}

	template <>
	void GiNaCTimeSymbol::print(const print_context &c, unsigned level) const
	{
		const pyoomph::TimeSymbol &sp = get_struct();
		std::string indstring = std::to_string(sp.index);
		if (dynamic_cast<const print_csrc *>(&c))
		{

			c.s << "t[" + indstring + "]";
		}
		else
		{
			c.s << "<time" + (indstring == "0" ? "" : indstring) + ">";
		}
	}

	template <>
	GiNaC::ex GiNaCTimeSymbol::derivative(const GiNaC::symbol &s) const
	{
		if (s == pyoomph::expressions::t || s == pyoomph::expressions::_dt_BDF1 || s == pyoomph::expressions::_dt_BDF2 || s == pyoomph::expressions::_dt_Newmark2)
		{
			if (get_struct().index == 0) // TODO: Is that correct?
			{
				return 1;
			}
			else
			{
				return 0;
			}
		}
		else
		{
			return 0;
		}
	}

	template <>
	void GiNaCFakeExponentialMode::print(const print_context &c, unsigned level) const
	{
		const pyoomph::FakeExponentialMode &sp = get_struct();
		if (dynamic_cast<const print_csrc *>(&c))
		{
			c.s << "1"; // Just return unity in the code
		}
		else
		{
			c.s << "<" << (sp.dual ? "Dual" : "") << "FakeExponentialMode: " << sp.arg << ">";
		}
	}

	template <>
	GiNaC::ex GiNaCFakeExponentialMode::derivative(const GiNaC::symbol &s) const
	{
		return (*this) * GiNaC::diff(this->get_struct().arg, s);
	}

}

namespace pyoomph
{

	CustomCoordinateSystem __no_coordinate_system;
	GiNaCCustomCoordinateSystemWrapper __no_coordinate_system_wrapper(&__no_coordinate_system);

	std::map<std::string, GiNaC::ex> __field_name_cache;

	GiNaC::ex _get_field_name_cache(const std::string &id)
	{
		if (!__field_name_cache.count(id))
			pyoomph::__field_name_cache.insert(std::make_pair(id, GiNaC::symbol(id)));
		return __field_name_cache[id];
	}

	bool operator==(const CustomMathExpressionWrapper &lhs, const CustomMathExpressionWrapper &rhs)
	{
		return lhs.cme == rhs.cme;
	}

	bool operator<(const CustomMathExpressionWrapper &lhs, const CustomMathExpressionWrapper &rhs)
	{
		return lhs.cme < rhs.cme;
	}

	bool operator==(const CustomMultiReturnExpressionWrapper &lhs, const CustomMultiReturnExpressionWrapper &rhs)
	{
		return lhs.cme == rhs.cme;
	}

	bool operator<(const CustomMultiReturnExpressionWrapper &lhs, const CustomMultiReturnExpressionWrapper &rhs)
	{
		return lhs.cme < rhs.cme;
	}

	bool operator==(const CustomMultiReturnExpressionResultSymbol &lhs, const CustomMultiReturnExpressionResultSymbol &rhs)
	{
		return lhs.func == rhs.func && lhs.arglist == rhs.arglist && lhs.index == rhs.index;
	}

	bool operator<(const CustomMultiReturnExpressionResultSymbol &lhs, const CustomMultiReturnExpressionResultSymbol &rhs)
	{
		return (lhs.func < rhs.func) || ((lhs.func == rhs.func) && (lhs.arglist < rhs.arglist)) || ((lhs.func == rhs.func) && (lhs.arglist == rhs.arglist) && (lhs.index < rhs.index));
	}

	bool operator==(const DelayedPythonCallbackExpansionWrapper &lhs, const DelayedPythonCallbackExpansionWrapper &rhs)
	{
		return lhs.cme == rhs.cme;
	}

	bool operator<(const DelayedPythonCallbackExpansionWrapper &lhs, const DelayedPythonCallbackExpansionWrapper &rhs)
	{
		return lhs.cme < rhs.cme;
	}

	bool operator==(const CustomCoordinateSystemWrapper &lhs, const CustomCoordinateSystemWrapper &rhs)
	{
		return lhs.cme == rhs.cme;
	}

	bool operator<(const CustomCoordinateSystemWrapper &lhs, const CustomCoordinateSystemWrapper &rhs)
	{
		return lhs.cme < rhs.cme;
	}

	bool operator==(const GlobalParameterWrapper &lhs, const GlobalParameterWrapper &rhs)
	{
		return lhs.cme == rhs.cme;
	}

	bool operator<(const GlobalParameterWrapper &lhs, const GlobalParameterWrapper &rhs)
	{
		return lhs.cme < rhs.cme;
	}

	bool operator==(const PlaceHolderResolveInfo &lhs, const PlaceHolderResolveInfo &rhs)
	{
		if (lhs.code != rhs.code)
			return false;
		if (lhs.tags.size() != rhs.tags.size())
			return false;
		for (unsigned int i = 0; i < lhs.tags.size(); i++)
			if (lhs.tags[i] != rhs.tags[i])
				return false;
		return true;
	}

	bool operator<(const PlaceHolderResolveInfo &lhs, const PlaceHolderResolveInfo &rhs)
	{
		if (lhs.code < rhs.code)
			return true;
		else if (lhs.code > rhs.code)
			return false;
		if (lhs.tags.size() < rhs.tags.size())
			return true;
		else if (lhs.tags.size() > rhs.tags.size())
			return false;
		for (unsigned int i = 0; i < lhs.tags.size(); i++)
		{
			if (lhs.tags[i] < rhs.tags[i])
				return true;
			else if (lhs.tags[i] > rhs.tags[i])
				return false;
		}
		return false;
	}

	bool operator==(const TimeSymbol &lhs, const TimeSymbol &rhs)
	{
		return lhs.index == rhs.index;
	}

	bool operator<(const TimeSymbol &lhs, const TimeSymbol &rhs)
	{
		return lhs.index < rhs.index;
	}

	bool operator==(const FakeExponentialMode &lhs, const FakeExponentialMode &rhs)
	{
		return lhs.arg.is_equal(rhs.arg) && lhs.dual == rhs.dual;
	}

	bool operator<(const FakeExponentialMode &lhs, const FakeExponentialMode &rhs)
	{
		return (lhs.arg.compare(rhs.arg) < 0) || (lhs.arg.compare(rhs.arg) == 0 && (lhs.dual < rhs.dual));
	}

	std::map<CustomMathExpressionBase *, int> CustomMathExpressionBase::code_map; // Mapping for indices
	unsigned CustomMathExpressionBase::unique_counter = 0;

	std::map<CustomMultiReturnExpressionBase *, int> CustomMultiReturnExpressionBase::code_map; // Mapping for indices
	unsigned CustomMultiReturnExpressionBase::unique_counter = 0;

	std::map<std::string, GiNaC::possymbol> base_units;

	namespace expressions
	{

		bool need_to_hold(const ex &arg)
		{
			for (GiNaC::const_preorder_iterator i = arg.preorder_begin(); i != arg.preorder_end(); ++i)
			{
				if (is_ex_the_function(*i, testfunction) || 
					is_ex_the_function(*i, dimtestfunction) || 
					is_ex_the_function(*i, nondimfield) || 
					is_ex_the_function(*i, scale) || 
					is_ex_the_function(*i, test_scale) || 
					is_ex_the_function(*i, field) || 
					is_ex_the_function(*i, unitvect) || 
					is_ex_the_function(*i, symbol_subs) || 
					is_ex_the_function(*i, Diff) || 
					is_ex_the_function(*i, eval_in_domain) || 
					is_ex_the_function(*i,ginac_expand) || 
					is_ex_the_function(*i,grad) || 
					is_ex_the_function(*i,div) || 
					is_ex_the_function(*i,transpose) ||
					is_ex_the_function(*i,determinant) ||
					is_ex_the_function(*i,trace) || // TODO: Add more!
					is_ex_the_function(*i,minimize_functional_derivative) 
					)
				{
					return true;
				}
			}
			return false;
		}

		int el_dim = -1;

		potential_real_symbol x("x");
		potential_real_symbol y("y");
		potential_real_symbol z("z");
		potential_real_symbol X("X");
		potential_real_symbol Y("Y");
		potential_real_symbol Z("Z");
		potential_real_symbol nx("normal_x");
		potential_real_symbol ny("normal_y");
		potential_real_symbol nz("normal_z");
		potential_real_symbol local_coordinate_1("local_coordinate_1");
		potential_real_symbol local_coordinate_2("local_coordinate_2");
		potential_real_symbol local_coordinate_3("local_coordinate_3");
		potential_real_symbol zeta_coordinate_1("zeta_coordinate_1");
		potential_real_symbol zeta_coordinate_2("zeta_coordinate_2");
		potential_real_symbol zeta_coordinate_3("zeta_coordinate_3");
		potential_real_symbol timefrac_tracer("timefrac_tracer");
		potential_real_symbol t("t");
		potential_real_symbol _dt_BDF1("_dt_BDF1");
		potential_real_symbol _dt_BDF2("_dt_BDF2");
		potential_real_symbol _dt_Newmark2("_dt_Newmark2");
		potential_real_symbol __partial_t_mass_matrix("__partial_t_mass_matrix");
		potential_real_symbol dt("dt");
		symbol nnode("nnode");

		potential_real_symbol *proj_on_test_function = NULL;

		idx l_shape(symbol("l_shape"), nnode);
		idx l_test(symbol("l_test"), nnode);

		// TODO: Not sure whether this is correct in all cases
		bool collect_base_units(GiNaC::ex arg, GiNaC::ex &factor, GiNaC::ex &units, GiNaC::ex &rest)
		{
			if (pyoomph_verbose)
				std::cout << "COLLECTING BASE UNITS IN " << arg << std::endl
						  << "WITH CURRENT FACTOR " << factor << " UNITS " << units << " REST " << rest << std::endl
						  << std::endl;
			rest = 1;
			factor = 1;
			units = 1;
			GiNaC::ex cl = GiNaC::collect_common_factors(GiNaC::expand(arg));
			// GiNaC::ex cl=GiNaC::expand(arg);
			//  std::cout << "CL "  << cl <<  std::endl;
			//  std::cout << "NOPS "  << cl.nops() <<  std::endl;
			if (!GiNaC::is_exactly_a<GiNaC::mul>(cl))
			{
				if (GiNaC::is_a<GiNaC::symbol>(cl))
				{
					bool found = false;
					for (auto &bu : base_units)
					{
						if (bu.second == cl)
						{
							units *= cl;
							units=GiNaC::expand(units);
							found = true;
							break;
						}
					}
					if (!found)
						rest *= cl;
				}
				else if (GiNaC::is_a<GiNaC::numeric>(cl) || GiNaC::is_a<GiNaC::constant>(cl))
				{
					if (GiNaC::to_double(GiNaC::ex_to<GiNaC::numeric>(cl.evalf()))<0)
					{						
							factor *= -cl;
							rest *=-1;
					}
					else
					{
						factor*=cl;
					}					
				}
				else if (GiNaC::is_exactly_a<GiNaC::power>(cl))
				{
					GiNaC::ex srest = 1;
					GiNaC::ex sfactor = 1;
					GiNaC::ex sunits = 1;
					if (!collect_base_units(cl.op(0), sfactor, sunits, srest))
					{
						std::cerr << "Problem collecting units in " << cl.op(0) << std::endl;
						return false;
					}
					if (pyoomph_verbose)
						std::cout << "  APPLY POWER " << cl.op(1) << " on " << sunits << "  ,  " << sfactor << " , " << srest << std::endl;
					units *= GiNaC::power(sunits, cl.op(1));
					units=GiNaC::expand(units);
					factor *= GiNaC::power(sfactor, cl.op(1));
					rest *= GiNaC::power(srest, cl.op(1));
				}
				else if (GiNaC::is_a<GiNaC::add>(cl))
				{
					// Find some dominant numerical factor to pull out
					std::vector<GiNaC::ex> terms(cl.nops(), 0);
					std::vector<GiNaC::ex> factors(cl.nops(), 0);
					GiNaC::ex common_unit = 0;
					GiNaC::ex dominant_factor = 0;
					for (unsigned int i = 0; i < cl.nops(); i++)
					{
						GiNaC::ex srest = 1;
						GiNaC::ex sfactor = 1;
						GiNaC::ex sunits = 1;
						if (!collect_base_units(cl.op(i), sfactor, sunits, srest))
						{
							std::cerr << "Problem collecting units in " << cl.op(i) << std::endl;
							return false;
						}
						sunits=GiNaC::expand(sunits);
						if (i == 0)
							common_unit = sunits;
						else
						{
							if (!common_unit.is_equal(sunits))
							{
								std::cerr << "Problem: Adding/subtracting different units [" << common_unit << "] and [" << sunits << "] in " << cl << std::endl;
								return false;
							}
						}
						if (abs(sfactor) > dominant_factor)
							dominant_factor = abs(sfactor);
						terms[i] = srest;
						factors[i] = sfactor;
					}
					units *= common_unit;
					units=GiNaC::expand(units);
					factor *= dominant_factor;
					GiNaC::ex normalized_rest = 0;
					for (unsigned int i = 0; i < cl.nops(); i++)
					{
						normalized_rest += (factors[i] / dominant_factor) * terms[i];
					}
					rest *= normalized_rest;
				}
				else if (is_ex_the_function(cl, subexpression))
				{
					GiNaC::ex srest = 1;
					GiNaC::ex sfactor = 1;
					GiNaC::ex sunits = 1;
					if (!collect_base_units(cl.op(0), sfactor, sunits, srest))
					{
						std::cerr << "Problem collecting units in subexpression " << cl.op(0) << std::endl;
						return false;
					}
					units *= sunits;
					units=GiNaC::expand(units);
					factor *= sfactor;
					rest *= subexpression(srest);
				}
				else if (is_ex_the_function(cl, expressions::maximum) || is_ex_the_function(cl, expressions::minimum) || is_ex_the_function(cl, expressions::absolute) || is_ex_the_function(cl, expressions::heaviside) || is_ex_the_function(cl, expressions::signum))
				{
					std::vector<GiNaC::ex> args(cl.nops());
					for (unsigned int i = 0; i < cl.nops(); i++)
					{
						args[i] = cl.op(i);
					}
					std::vector<GiNaC::ex> rest_args(cl.nops(), 1);
					std::vector<GiNaC::ex> scales(cl.nops(), 1);
					GiNaC::ex common_unit = 0;
					GiNaC::ex dominant_factor = 0;
					for (unsigned int i = 0; i < cl.nops(); i++)
					{
						GiNaC::ex sfactor = 1;
						GiNaC::ex sunit = 1;
						GiNaC::ex srest = 1;
						if (!expressions::collect_base_units(args[i], sfactor, sunit, srest))
						{
							std::ostringstream oss;
							oss << std::endl
								<< "INPUT: " << cl << std::endl
								<< "PROCESSED ARG:" << args[i] << std::endl
								<< "numerical part: " << sfactor << "unit part:" << sunit << "rest part:" << srest << std::endl;
							throw_runtime_error("Cannot extract the unit from the argument:" + oss.str());
						}
						sunit=GiNaC::expand(sunit);
						if (sfactor.is_zero())
						{
						}
						else
						{
							if (common_unit.is_zero())
							{
								common_unit = sunit;
							}
							else
							{
								if ((!sunit.is_equal(common_unit)))
								{
									std::ostringstream oss;
									oss << "Nonmatching units in function " << cl << ":  " << common_unit << " vs " << sunit << " in argument " << i << " : " << cl.op(i) << std::endl;
									throw_runtime_error(oss.str());
								}
							}
						}
						if (abs(sfactor) > dominant_factor)
							dominant_factor = abs(sfactor);
						rest_args[i] = srest;
						scales[i] = sfactor;
					}
					if (dominant_factor.is_zero())
						return 0;
					for (unsigned int i = 0; i < cl.nops(); i++)
					{
						rest_args[i] *= scales[i] / dominant_factor;
					}
					if (is_ex_the_function(cl, expressions::heaviside))
					{
						rest *= pyoomph::expressions::heaviside(rest_args[0]);
					}
					else if (is_ex_the_function(cl, expressions::signum))
					{
						rest *= pyoomph::expressions::signum(rest_args[0]);
					}
					else
					{
						units *= common_unit;
						units=GiNaC::expand(units);
						factor *= dominant_factor;
						if (is_ex_the_function(cl, expressions::maximum))
						{
							rest *= pyoomph::expressions::maximum(rest_args[0], rest_args[1]);
						}
						else if (is_ex_the_function(cl, expressions::minimum))
						{
							rest *= pyoomph::expressions::minimum(rest_args[0], rest_args[1]);
						}
						else if (is_ex_the_function(cl, expressions::absolute))
						{
							rest *= pyoomph::expressions::absolute(rest_args[0]);
						}
						else
						{
							throw_runtime_error("Should not end up here");
						}
					}
				}
				else if (GiNaC::is_a<GiNaC::function>(cl))
				{
					// Test if there are units left in the function args
					bool units_left = false;
					for (auto &bu : base_units)
					{
						if (GiNaC::has(cl, bu.second))
						{
							units_left = true;
							break;
						}
					}
					if (!units_left)
					{
						rest *= cl;
					}
					else
					{
						GiNaC::exvector newargs(cl.nops());
						if (pyoomph_verbose)
							std::cout << "SIMPLIFYING " << cl << std::endl;
						for (unsigned int i = 0; i < cl.nops(); i++)
						{
							if (pyoomph_verbose)
								std::cout << "		SIMPLIFYING ARG " << cl.op(i) << std::endl;
							GiNaC::ex srest = 1;
							GiNaC::ex sfactor = 1;
							GiNaC::ex sunits = 1;
							if (!collect_base_units(cl.op(i), sfactor, sunits, srest))
							{
								std::cerr << "Problem collecting units in arg " << i << ", i.e. " << cl.op(i) << std::endl
										  << " of " << std::endl
										  << cl << std::endl;
								return false;
							}
							sunits=GiNaC::expand(sunits);
							newargs[i] = sfactor * sunits * srest;
							if (pyoomph_verbose)
								std::cout << "		NEW ARG " << newargs[i] << std::endl;
						}
						GiNaC::ex nf = GiNaC::function(GiNaC::ex_to<GiNaC::function>(cl).get_serial(), newargs);
						if (pyoomph_verbose)
							std::cout << "SIMPLIFY " << cl << " RESULTED IN " << std::endl
									  << nf << std::endl;
						rest *= nf;
					}
				}
				else
				{
					rest *= cl;
					//			std::cout << "SETTING REST: " <<  rest << std::endl;
				}
			}
			else
			{
				for (unsigned int i = 0; i < cl.nops(); i++)
				{
					//  std::cout << "FI " << i << "  " << cl.op(i) << "  " << std::endl;
					if (GiNaC::is_a<GiNaC::numeric>(cl.op(i)) || GiNaC::is_a<GiNaC::constant>(cl.op(i)))
					{
						factor *= cl.op(i); /*std::cout << "   NUM" << std::endl;*/
					}
					else if (GiNaC::is_a<GiNaC::symbol>(cl.op(i)))
					{
						//			std::cout << "   SYMBOL" << std::endl;
						bool found = false;
						for (auto &bu : base_units)
						{
							if (bu.second == cl.op(i))
							{
								units *= cl.op(i);
								units=GiNaC::expand(units);
								found = true;
								break;
							}
						}
						if (!found)
							rest *= cl.op(i);
					}
					else if (GiNaC::is_exactly_a<GiNaC::power>(cl.op(i)))
					{
						//			std::cout << "   POWER: " << cl << "  || i=" << i << "  op(i)= " << cl.op(i) <<  std::endl;
						GiNaC::ex srest = 1;
						GiNaC::ex sfactor = 1;
						GiNaC::ex sunits = 1;
						if (!collect_base_units(cl.op(i).op(0), sfactor, sunits, srest))
						{
							std::cerr << "Problem collecting units in basis " << cl.op(i).op(0) << std::endl;
							return false;
						}
						sunits=GiNaC::expand(sunits);
						GiNaC::ex erest = 1;
						GiNaC::ex efactor = 1;
						GiNaC::ex eunits = 1;
						if (!collect_base_units(cl.op(i).op(1), efactor, eunits, erest))
						{
							std::cerr << "Problem collecting units in exponent " << cl.op(i).op(1) << std::endl;
							return false;
						}						
						eunits=GiNaC::expand(eunits);
						if (pyoomph_verbose)
							std::cout << "  APPLY POWER " << cl.op(i).op(1) << " on " << sunits << "  ,  " << sfactor << " , " << srest << std::endl;
						//std::cout << "EXPONENT FACTOR UNIT AND REST " << efactor << "  " << eunits << "  " << erest << std::endl;
						
						units *= GiNaC::power(sunits, cl.op(i).op(1));
						units=GiNaC::expand(units);
						factor *= GiNaC::power(sfactor, cl.op(i).op(1));
						rest *= GiNaC::power(srest, cl.op(i).op(1));
						if (pyoomph_verbose)
							std::cout << "   RES " << units << "  " << factor << "  " << rest << std::endl;
					}
					else
					{
						GiNaC::ex srest = 1;
						GiNaC::ex sfactor = 1;
						GiNaC::ex sunits = 1;						
						if (!collect_base_units(cl.op(i), sfactor, sunits, srest))
						{
							std::cerr << "Problem collecting units in " << cl.op(i) << std::endl;
							return false;
						}
						units *= sunits;
						units=GiNaC::expand(units);
						factor *= sfactor;
						rest *= srest;
					}
					//		else {rest*=cl.op(i); 		/*	std::cout << "   REST" << std::endl;*/}
				}
			}
			// TODO Check the rest for missing units
			for (auto &bu : base_units)
			{
				//		if (bu.second==cl.op(i))  {units*=cl.op(i); found=true; break;}
				if (GiNaC::has(rest, bu.second))
					return false;
			}

			// Check whether the factor is positive, try to go for the positive one
			//std::cout << "FACTOR " << factor << std::endl;
			GiNaC::ex factorf=factor.evalf();
			if (!GiNaC::is_a<GiNaC::numeric>(factorf))
			{
				rest*=factor;
				factor=1;
			}
			else if (GiNaC::to_double(GiNaC::ex_to<GiNaC::numeric>(factorf))<0)
			{
				factor *= -1;
				rest *= -1;
			}

			// However, if the rest is just a negative number, we must make the factor negative instead, so that rest=1 factor=-1 unit=meter in e.g. -1*meter
			if (GiNaC::is_a<GiNaC::numeric>(rest) || GiNaC::is_a<GiNaC::constant>(rest))
			{
				if (GiNaC::to_double(GiNaC::ex_to<GiNaC::numeric>(rest.evalf()))<0)
				{
					factor *= -1;
					rest *= -1;
				}
			}
			return true;
		}

		GiNaC::ex diff(const GiNaC::ex &what, const GiNaC::ex &wrto)
		{
			if (pyoomph::pyoomph_verbose)
			{
				std::cout << "  in diff " << what << " BY " << wrto << std::endl;
			}
			if (GiNaC::is_a<GiNaC::realsymbol>(wrto))
			{
				if (pyoomph::pyoomph_verbose)
				{
					std::cout << "  in diff " << what << " BY REALSYMB" << wrto << std::endl;
				}
				GiNaC::realsymbol wrto_sym = GiNaC::ex_to<GiNaC::realsymbol>(wrto);
				return GiNaC::diff(what, wrto_sym);
			}
			if (GiNaC::is_a<GiNaC::symbol>(wrto))
			{
				if (pyoomph::pyoomph_verbose)
				{
					std::cout << "  in diff " << what << " BY SYMB" << wrto << std::endl;
				}
				GiNaC::symbol wrto_sym = GiNaC::ex_to<GiNaC::symbol>(wrto);
				return GiNaC::diff(what, wrto_sym);
			}
			// Test for x,y,z which are shape expansions
			if (GiNaC::is_a<GiNaC::GiNaCShapeExpansion>(wrto))
			{
				if (pyoomph::pyoomph_verbose)
				{
					std::cout << "  in diff " << what << " SHAPE " << wrto << std::endl;
				}
				GiNaC::GiNaCShapeExpansion se = GiNaC::ex_to<GiNaC::GiNaCShapeExpansion>(wrto);
				auto &sp = se.get_struct();
				if (sp.is_derived)
					throw_runtime_error("Deriving wrt. a derived shape expansion");
				if (sp.dt_order)
					throw_runtime_error("Deriving wrt. a time-derived shape expansion");
				if (dynamic_cast<D1XBasisFunction *>(sp.basis))
					throw_runtime_error("Deriving wrt. a spatially-derived shape expansion");

				if (sp.field->get_name() == "coordinate_x")
					return diff(what, x);
				else if (sp.field->get_name() == "coordinate_y")
					return diff(what, y);
				else if (sp.field->get_name() == "coordinate_z")
					return diff(what, z);
				else if (sp.field->get_name() == "lagrangian_x")
					return diff(what, X);
				else if (sp.field->get_name() == "lagrangian_y")
					return diff(what, Y);
				else if (sp.field->get_name() == "lagrangian_z")
					return diff(what, Z);
				else if (sp.field->get_name() == "local_coordinate_1")
					return diff(what, local_coordinate_1);
				else if (sp.field->get_name() == "local_coordinate_2")
					return diff(what, local_coordinate_2);
				else if (sp.field->get_name() == "local_coordinate_3")
					return diff(what, local_coordinate_3);					

				else
				{
					pyoomph::DerivedShapeExpansionsToUnity deriv_se_to_1;
					return deriv_se_to_1(diff(what, sp.field->get_symbol()));
					//    throw_runtime_error("Deriving wrt. an arbitrary shape expansion");
				}
			}

			// Otherwise, it may only be a value and unit symbols... Sep them from the normal variable
			GiNaC::ex normvar = 0;
			for (GiNaC::const_preorder_iterator it = wrto.preorder_begin(); it != wrto.preorder_end(); it++)
			{
				if (GiNaC::is_a<GiNaC::symbol>(*it))
				{
					bool is_base_unit = false;
					for (auto &bu : pyoomph::base_units)
					{
						if (*it == bu.second)
						{
							is_base_unit = true;
							break;
						}
					}
					if (!is_base_unit)
					{
						if (normvar.is_zero())
							normvar = (*it);
						else
						{
							normvar = 0;
							break;
						}
					}
				}
				else if (GiNaC::is_a<GiNaC::GiNaCShapeExpansion>(*it))
				{
					if (normvar.is_zero())
						normvar = (*it);
					else
					{
						normvar = 0;
						break;
					}
				}
			}
			if (!normvar.is_zero())
			{
				GiNaC::ex remainder = wrto / (normvar);
				return 1 / remainder * (diff(what, normvar));
			}
			std::ostringstream oss;
			oss << wrto;
			throw_runtime_error("Can only take a diff(f,x) if x is symbol, not an expressions: " + oss.str()) return 0.0;
		}

		GiNaC::ex diff_x(const GiNaC::ex &what, unsigned i)
		{
			if (i == 0)
				return GiNaC::diff(what, x);
			else if (i == 1)
				return GiNaC::diff(what, y);
			else if (i == 2)
				return GiNaC::diff(what, z);
			else
				return 0.0;
		}

		////////////////

		static ex unitvect_eval(const ex &dir, const ex &basedim, const ex &coordsys, const ex &flags)
		{
			std::cout << "ENTERING UNITVECT EVAL A" << std::endl;
			GiNaCCustomCoordinateSystemWrapper w = ex_to<GiNaCCustomCoordinateSystemWrapper>(coordsys);
			const pyoomph::CustomCoordinateSystemWrapper &sp = w.get_struct();
			CustomCoordinateSystem *sys = &__no_coordinate_system;
			std::cout << "ENTERING UNITVECT EVAL B" << std::endl;
			if (sp.cme == &__no_coordinate_system)
			{
				if (__current_code)
				{
					sys = __current_code->get_coordinate_system();
					if (pyoomph::pyoomph_verbose)
					{
						std::cout << "Got the coordinate system from element " << sys << std::endl;
					}
				}
				if (sys == &__no_coordinate_system)
				{
					std::cerr << "CANNOT RESOLVE COORD SYS" << std::endl;
					return unitvect(dir, basedim, coordsys, flags).hold();
				}
			}
			else
			{
				sys = sp.cme;
			}
			std::cout << "ENTERING UNITVECT EVAL C" << std::endl;
			int _flags = GiNaC::ex_to<GiNaC::numeric>(flags.evalf()).to_double();
			int ndim = GiNaC::ex_to<GiNaC::numeric>(basedim.evalf()).to_double();
			if (ndim < 0)
			{
				if (__current_code)
				{
					if (_flags & 8)
					{
						ndim = __current_code->lagrangian_dimension();
					}
					else
					{
						ndim = __current_code->nodal_dimension();
					}
					if (pyoomph::pyoomph_verbose)
					{
						std::cout << "NDIM WAS SET TO " << ndim << std::endl;
					}
				}
				else
					return unitvect(dir, basedim, coordsys, flags).hold();
			}

			std::cout << "ENTERING UNITVECT EVAL D" << std::endl;
			int direction = GiNaC::ex_to<GiNaC::numeric>(dir.evalf()).to_double();
			std::vector<GiNaC::ex> v(pyoomph::the_vector_dim, 0);
			v[direction] = 1;
			return 0 + GiNaC::matrix(v.size(), 1, GiNaC::lst(v.begin(), v.end()));
		}

		REGISTER_FUNCTION(unitvect, eval_func(unitvect_eval).set_return_type(GiNaC::return_types::noncommutative))

		////////////////

		static ex grad_eval(const ex &f, const ex &nodal_dim, const ex &elem_dim, const ex &coordsys, const ex &flags)
		{
			if (f == wild())
				return grad(f, nodal_dim, elem_dim, coordsys, flags).hold();
			if (pyoomph::pyoomph_verbose)
				std::cout << "ENTERING GRAD  " << f << "  " << nodal_dim << "  " << elem_dim << "  " << coordsys << "   " << flags << std::endl;
			if (need_to_hold(f))
				return grad(f, nodal_dim, elem_dim, coordsys, flags).hold();
			// Resolve coordinate system
			GiNaCCustomCoordinateSystemWrapper w = ex_to<GiNaCCustomCoordinateSystemWrapper>(coordsys);
			const pyoomph::CustomCoordinateSystemWrapper &sp = w.get_struct();

			CustomCoordinateSystem *sys = &__no_coordinate_system;
			if (sp.cme == &__no_coordinate_system)
			{
				if (__current_code)
				{
					sys = __current_code->get_coordinate_system();
					if (pyoomph::pyoomph_verbose)
					{
						std::cout << "Got the coordinate system from element " << sys << std::endl;
					}
				}
				if (sys == &__no_coordinate_system)
				{
					std::cerr << "CANNOT RESOLVE COORD SYS" << std::endl;
					return grad(f, nodal_dim, elem_dim, coordsys, flags).hold();
				}
			}
			else
			{
				sys = sp.cme;
			}
			int _flags = GiNaC::ex_to<GiNaC::numeric>(flags.evalf()).to_double();
			int ndim = GiNaC::ex_to<GiNaC::numeric>(nodal_dim.evalf()).to_double();
			if (ndim < 0)
			{
				if (__current_code)
				{
					if (_flags & 8)
					{
						ndim = __current_code->lagrangian_dimension();
					}
					else
					{
						ndim = __current_code->nodal_dimension();
					}
					if (pyoomph::pyoomph_verbose)
					{
						std::cout << "NDIM WAS SET TO " << ndim << std::endl;
					}
				}
				else
					return grad(f, nodal_dim, elem_dim, coordsys, flags).hold();
			}
			int edim = GiNaC::ex_to<GiNaC::numeric>(elem_dim.evalf()).to_double();
			if (edim < 0)
			{
				if (__current_code)
				{
					edim = __current_code->get_dimension();
					if (pyoomph::pyoomph_verbose)
					{
						std::cout << "EDIM WAS SET TO " << edim << std::endl;
					}
				}
				else
					return grad(f, nodal_dim, elem_dim, coordsys, flags).hold();
			}

			//	 std::cout << "CALLING GRAD " << typeid(sys).name() << "   " <<  f << "  " << ndim << "  " << _flags << std::endl;
			if (pyoomph::pyoomph_verbose)
				std::cout << "CALLING GRAD " << sys << std::endl;
			return sys->grad(f, ndim, edim, _flags);
		}

		REGISTER_FUNCTION(grad, eval_func(grad_eval).set_return_type(GiNaC::return_types::noncommutative))

		static ex directional_derivative_eval(const ex &f, const ex &d, const ex &nodal_dim, const ex &elem_dim, const ex &coordsys, const ex &flags)
		{
			if (f == wild())
				return directional_derivative(f, d, nodal_dim, elem_dim, coordsys, flags).hold();
			if (pyoomph::pyoomph_verbose)
				std::cout << "ENTERING DIRECTIONAL DERIVATIVE  " << f << "  " << d << "  " << nodal_dim << "  " << elem_dim << "  " << coordsys << "   " << flags << std::endl;
			if (need_to_hold(f) || need_to_hold(d))
			{
				if (pyoomph::pyoomph_verbose)
					std::cout << "		MUST BE HELD  " << f << "  " << d << "  " << nodal_dim << "  " << elem_dim << "  " << coordsys << "   " << flags << std::endl;
				return directional_derivative(f, d, nodal_dim, elem_dim, coordsys, flags).hold();
			}
				
			// Resolve coordinate system
			GiNaCCustomCoordinateSystemWrapper w = ex_to<GiNaCCustomCoordinateSystemWrapper>(coordsys);
			const pyoomph::CustomCoordinateSystemWrapper &sp = w.get_struct();

			CustomCoordinateSystem *sys = &__no_coordinate_system;
			if (sp.cme == &__no_coordinate_system)
			{
				if (__current_code)
				{
					sys = __current_code->get_coordinate_system();
					if (pyoomph::pyoomph_verbose)
					{
						std::cout << "Got the coordinate system from element " << sys << std::endl;
					}
				}
				if (sys == &__no_coordinate_system)
				{
					std::cerr << "CANNOT RESOLVE COORD SYS" << std::endl;
					return directional_derivative(f, d, nodal_dim, elem_dim, coordsys, flags).hold();
				}
			}
			else
			{
				sys = sp.cme;
			}
			if (pyoomph::pyoomph_verbose)
				std::cout << "ENTERING DIRECTIONAL DERIVATIVE2  " << f << "  " << d << "  " << nodal_dim << "  " << elem_dim << "  " << coordsys << "   " << flags << std::endl;

			int _flags = GiNaC::ex_to<GiNaC::numeric>(flags.evalf()).to_double();
			int ndim = GiNaC::ex_to<GiNaC::numeric>(nodal_dim.evalf()).to_double();

			if (pyoomph::pyoomph_verbose)
				std::cout << "ENTERING DIRECTIONAL DERIVATIVE3  " << f << "  " << d << "  " << nodal_dim << "  " << elem_dim << "  " << coordsys << "   " << flags << std::endl;
			if (ndim < 0)
			{
				if (__current_code)
				{
					if (_flags & 8)
					{
						ndim = __current_code->lagrangian_dimension();
					}
					else
					{
						ndim = __current_code->nodal_dimension();
					}
					if (pyoomph::pyoomph_verbose)
					{
						std::cout << "NDIM WAS SET TO " << ndim << std::endl;
					}
				}
				else
					return directional_derivative(f, d, nodal_dim, elem_dim, coordsys, flags).hold();
			}
			int edim = GiNaC::ex_to<GiNaC::numeric>(elem_dim.evalf()).to_double();
			if (edim < 0)
			{
				if (__current_code)
				{
					edim = __current_code->get_dimension();
					if (pyoomph::pyoomph_verbose)
					{
						std::cout << "EDIM WAS SET TO " << edim << std::endl;
					}
				}
				else
					return directional_derivative(f, d, nodal_dim, elem_dim, coordsys, flags).hold();
			}

			//	 std::cout << "CALLING GRAD " << typeid(sys).name() << "   " <<  f << "  " << ndim << "  " << _flags << std::endl;
			if (pyoomph::pyoomph_verbose)
				std::cout << "CALLING DIRECTIONAL DERIVATIVE " << sys << std::endl;
			GiNaC::ex eva = f.evalm();
			if (GiNaC::is_a<GiNaC::matrix>(eva))
			{
				GiNaC::matrix evm = GiNaC::ex_to<GiNaC::matrix>(eva);
				if (evm.cols() > 1 && evm.rows() > 1)
					_flags |= 16; // tensor
				else
					_flags |= 4; // vector
			}
			// Check if we have a vectorial direction
			GiNaC::ex evd = d.evalm();
			if (GiNaC::is_zero(evd)) return 0;
			if (!GiNaC::is_a<GiNaC::matrix>(evd))
			{
				std::ostringstream oss;
				oss << evd;
				throw_runtime_error("Second argument of directional derivative must be a vector, but is: " + oss.str());
			}
			GiNaC::matrix evdm = GiNaC::ex_to<GiNaC::matrix>(evd);
			if (evdm.cols() > 1 && evdm.rows() > 1)
			{
				std::ostringstream oss;
				oss << evd;
				throw_runtime_error("Second argument of directional derivative must be a vector, but is: " + oss.str());
			}

			return sys->directional_derivative(f, d, ndim, edim, _flags);
		}

		REGISTER_FUNCTION(directional_derivative, eval_func(directional_derivative_eval)) //.set_return_type(GiNaC::return_types::noncommutative))

		// General weak contribution of the form WEAKCONTRIB("name",{f,g,h,..},testfunc)
		// The specific coordinate system will take care of the evaluation
		static ex general_weak_differential_contribution_eval(const ex &funcname, const ex &f, const ex &test, const ex &nodal_dim, const ex &elem_dim, const ex &coordsys, const ex &flags)
		{
			if (f == wild())
				return general_weak_differential_contribution(funcname, f, test, nodal_dim, elem_dim, coordsys, flags).hold();
			if (pyoomph::pyoomph_verbose)
				std::cout << "ENTERING GENERAL WEAK DIFFERENTIAL CONTRIBUTION  " << funcname << "  " << f << "  " << test << "  " << nodal_dim << "  " << elem_dim << "  " << coordsys << "   " << flags << std::endl;
			if (need_to_hold(f))
				return general_weak_differential_contribution(funcname, f, test, nodal_dim, elem_dim, coordsys, flags).hold();
			// Resolve coordinate system
			GiNaCCustomCoordinateSystemWrapper w = ex_to<GiNaCCustomCoordinateSystemWrapper>(coordsys);
			const pyoomph::CustomCoordinateSystemWrapper &sp = w.get_struct();

			CustomCoordinateSystem *sys = &__no_coordinate_system;
			if (sp.cme == &__no_coordinate_system)
			{
				if (__current_code)
				{
					sys = __current_code->get_coordinate_system();
					if (pyoomph::pyoomph_verbose)
					{
						std::cout << "Got the coordinate system from element " << sys << std::endl;
					}
				}
				if (sys == &__no_coordinate_system)
				{
					std::cerr << "CANNOT RESOLVE COORD SYS" << std::endl;
					return general_weak_differential_contribution(funcname, f, test, nodal_dim, elem_dim, coordsys, flags).hold();
				}
			}
			else
			{
				sys = sp.cme;
			}
			int _flags = GiNaC::ex_to<GiNaC::numeric>(flags.evalf()).to_double();
			int ndim = GiNaC::ex_to<GiNaC::numeric>(nodal_dim.evalf()).to_double();
			if (ndim < 0)
			{
				if (__current_code)
				{
					if (_flags & 8)
					{
						ndim = __current_code->lagrangian_dimension();
					}
					else
					{
						ndim = __current_code->nodal_dimension();
					}
					if (pyoomph::pyoomph_verbose)
					{
						std::cout << "NDIM WAS SET TO " << ndim << std::endl;
					}
				}
				else
					return general_weak_differential_contribution(funcname, f, test, nodal_dim, elem_dim, coordsys, flags).hold();
			}
			int edim = GiNaC::ex_to<GiNaC::numeric>(elem_dim.evalf()).to_double();
			if (edim < 0)
			{
				if (__current_code)
				{
					edim = __current_code->get_dimension();
					if (pyoomph::pyoomph_verbose)
					{
						std::cout << "EDIM WAS SET TO " << edim << std::endl;
					}
				}
				else
					return general_weak_differential_contribution(funcname, f, test, nodal_dim, elem_dim, coordsys, flags).hold();
			}

			//	 std::cout << "CALLING GRAD " << typeid(sys).name() << "   " <<  f << "  " << ndim << "  " << _flags << std::endl;
			if (pyoomph::pyoomph_verbose)
				std::cout << "CALLING GENERAL WEAK DIFFERENTIAL CONTRIBUTION " << sys << std::endl;
			std::ostringstream oss;
			oss << funcname;
			std::string funcname_str = oss.str();
			std::vector<GiNaC::ex> lhs;
			if (!GiNaC::is_a<GiNaC::lst>(f))
				lhs.push_back(f);
			else
			{
				for (unsigned int ie = 0; ie < f.nops(); ie++)
					lhs.push_back(f.op(ie));
			}
			return sys->general_weak_differential_contribution(funcname_str, lhs, test, ndim, edim, _flags);
		}

		REGISTER_FUNCTION(general_weak_differential_contribution, eval_func(general_weak_differential_contribution_eval).set_return_type(GiNaC::return_types::noncommutative))

		////////////////

		static ex div_eval(const ex &v, const ex &nodal_dim, const ex &elem_dim, const ex &coordsys, const ex &flags)
		{
			if (v == wild())
				return div(v, nodal_dim, elem_dim, coordsys, flags).hold();
			if (need_to_hold(v))
				return div(v, nodal_dim, elem_dim, coordsys, flags).hold();
			ex eva = v.evalm();
			if (!is_a<matrix>(eva))
			{
				if (eva.is_zero())
					return 0;
				throw_runtime_error("Tried to take the divergence of something which is not a matrix or a vector");
			}
			GiNaC::matrix evam = GiNaC::ex_to<GiNaC::matrix>(eva);
			GiNaCCustomCoordinateSystemWrapper w = ex_to<GiNaCCustomCoordinateSystemWrapper>(coordsys);
			const pyoomph::CustomCoordinateSystemWrapper &sp = w.get_struct();
			CustomCoordinateSystem *sys = &__no_coordinate_system;
			if (sp.cme == &__no_coordinate_system)
			{
				if (__current_code)
				{
					sys = __current_code->get_coordinate_system();
					if (pyoomph::pyoomph_verbose)
					{
						std::cout << "Got the coordinate system from element " << sys << std::endl;
					}
				}
				if (sys == &__no_coordinate_system)
				{
					std::cerr << "CANNOT RESOLVE COORD SYS" << std::endl;
					return div(v, nodal_dim, elem_dim, coordsys, flags).hold();
				}
			}
			else
			{
				sys = sp.cme;
			}

			int _flags = GiNaC::ex_to<GiNaC::numeric>(flags.evalf()).to_double();
			int ndim = GiNaC::ex_to<GiNaC::numeric>(nodal_dim.evalf()).to_double();
			if (ndim < 0)
			{
				if (__current_code)
				{
					if (_flags & 8)
					{
						ndim = __current_code->lagrangian_dimension();
					}
					else
					{
						ndim = __current_code->nodal_dimension();
					}

					if (pyoomph::pyoomph_verbose)
						std::cout << "NDIM WAS SET TO " << ndim << std::endl;
				}
				else
					return div(v, nodal_dim, elem_dim, coordsys, flags).hold();
			}
			int edim = GiNaC::ex_to<GiNaC::numeric>(elem_dim.evalf()).to_double();
			if (edim < 0)
			{
				if (__current_code)
				{
					edim = __current_code->get_dimension();
					if (pyoomph::pyoomph_verbose)
						std::cout << "EDIM WAS SET TO " << edim << std::endl;
				}
				else
					return div(v, nodal_dim, elem_dim, coordsys, flags).hold();
			}

			if (pyoomph::pyoomph_verbose)
				std::cout << "CALLING DIV " << sys << std::endl;

			if (evam.cols() > 1 && evam.rows() > 1)
			{
				_flags |= 16;
			} // Tensor divergence!

			return sys->div(v, ndim, edim, _flags);
		}

		REGISTER_FUNCTION(div, eval_func(div_eval))

		////////////////

		static ex transpose_eval(const ex &v)
		{
			if (need_to_hold(v))
				return transpose(v).hold();
			ex eva = v.evalm();
			if (!is_a<matrix>(eva))
			{
				std::ostringstream oss;
				oss << eva;
				throw_runtime_error("Cannot transpose anything else as matrices, but got " + oss.str());
			}
			else
			{
				matrix ma = ex_to<matrix>(eva);
				return ma.transpose();
			}
		}

		REGISTER_FUNCTION(transpose, eval_func(transpose_eval))

		static ex trace_eval(const ex &m)
		{
			if (need_to_hold(m))
				return trace(m).hold();
			ex eva = m.evalm();
			if (!is_a<matrix>(eva))
			{
				std::ostringstream oss;
				oss << eva;
				throw_runtime_error("Cannot apply trace on anything else as matrices, but got " + oss.str());
			}
			else
			{
				matrix ma = ex_to<matrix>(eva);
				return ma.trace();
			}
		}

		REGISTER_FUNCTION(trace, eval_func(trace_eval).set_return_type(GiNaC::return_types::commutative))

		static ex matproduct_eval(const ex &m1, const ex &m2)
		{
			if (pyoomph::pyoomph_verbose)
				std::cout << "Entering matprod " << std::endl
						  << m1 << std::endl
						  << m2 << std::endl
						  << std::endl;
			if (need_to_hold(m1) || need_to_hold(m2))
				return matproduct(m1, m2).hold();
			if (pyoomph::pyoomph_verbose)
				std::cout << " MATPROD NOT HELD " << std::endl;
			ex evm1 = m1.evalm();
			ex evm2 = m2.evalm();
			if (!is_a<matrix>(evm1) || (!is_a<matrix>(evm2)))
			{
				if (evm1.is_zero() || evm2.is_zero())
					return 0;
				std::ostringstream oss;
				oss << "Cannot calculate the matrix product between non-matrices: " << std::endl
					<< evm1 << std::endl
					<< evm2 << std::endl;
				throw_runtime_error(oss.str());
				return 0;
			}
			else
			{
				if (pyoomph::pyoomph_verbose)
					std::cout << " MATPROD RESULT " << std::endl
							  << (evm1 * evm2).evalm() << std::endl;
				return (evm1 * evm2).evalm();
			}
		}

		REGISTER_FUNCTION(matproduct, eval_func(matproduct_eval))

		////////////////

		static ex dot_eval(const ex &a, const ex &b)
		{
			if (pyoomph::pyoomph_verbose)
				std::cout << "Entering dot " << std::endl
						  << a << std::endl
						  << b << std::endl
						  << std::endl;
			if (need_to_hold(a) || need_to_hold(b))
				return dot(a, b).hold();
			if (a.has(grad(wild(), wild(), wild(), wild(), wild())))
				return dot(a, b).hold();
			if (b.has(grad(wild(), wild(), wild(), wild(), wild())))
				return dot(a, b).hold();
			if (pyoomph::pyoomph_verbose)
				std::cout << " DOT NOT HELD " << std::endl;
			ex eva = a.evalm();
			ex evb = b.evalm();
			if (eva.is_zero() || evb.is_zero())
				return 0;
			if (is_a<matrix>(eva) && is_a<matrix>(evb))
			{
				matrix ma = ex_to<matrix>(eva);
				matrix mb = ex_to<matrix>(evb);

				if (ma.cols() != 1 || mb.cols() != 1)
				{
					std::ostringstream oss;
					oss << std::endl
						<< " a = " << a << std::endl
						<< " b = " << b << std::endl;
					throw_runtime_error("dot is only allowed between vectors, but got: " + oss.str());
				}
				GiNaC::ex ret;
				if (ma.rows() != mb.rows())
				{
					if (ma.rows() > mb.rows())
					{
						for (unsigned int i = mb.rows(); i < ma.rows(); i++)
						{
							if (!ma(i, 0).is_zero())
							{
								std::ostringstream oss;
								oss << std::endl
									<< " a = " << a << std::endl
									<< " b = " << b << std::endl;
								throw_runtime_error("dot got different dimensions in both vectors:" + oss.str());
							}
						}
						for (unsigned int i = 0; i < mb.rows(); i++)
							ret += ma(i, 0) * mb(i, 0);
						return ret;
					}
					else
					{
						for (unsigned int i = ma.rows(); i < mb.rows(); i++)
						{
							if (!mb(i, 0).is_zero())
							{
								std::ostringstream oss;
								oss << std::endl
									<< " a = " << a << std::endl
									<< " b = " << b << std::endl;
								throw_runtime_error("dot got different dimensions in both vectors:" + oss.str());
							}
						}
						for (unsigned int i = 0; i < ma.rows(); i++)
							ret += ma(i, 0) * mb(i, 0);
						return ret;
					}
				}
				for (unsigned int i = 0; i < ma.rows(); i++)
				{
					ret += ma(i, 0) * mb(i, 0);
				}
				return ret;
			}
			else
			{
				std::ostringstream oss;
				oss << std::endl
					<< " a = " << a << std::endl
					<< " b = " << b << std::endl;
				throw_runtime_error("dot is only allowed between vectors, but got: " + oss.str());
			}
		}

		REGISTER_FUNCTION(dot, eval_func(dot_eval).set_return_type(GiNaC::return_types::commutative))

		static ex debug_ex_eval(const ex &a)
		{
			if (need_to_hold(a))
			{
				std::cout << "DEBUG EXPRESSION HOLD: " << a << std::endl;
				return debug_ex(a).hold();
			}
			std::cout << "DEBUG EXPRESSION FULLY EXPANDED: " << a << std::endl;
			std::cout << "EVALM: " << std::flush;
			std::cout << a.evalm() << std::endl;
			exit(0);
			return 0;
		}

		REGISTER_FUNCTION(debug_ex, eval_func(debug_ex_eval).set_return_type(GiNaC::return_types::commutative))

		////////////////

		static ex double_dot_eval(const ex &a, const ex &b)
		{
			if (pyoomph::pyoomph_verbose)
				std::cout << "Trying to eval double dot of " << std::endl
						  << a << std::endl
						  << b << std::endl
						  << std::endl;
			ex evma = a.evalm();
			ex evmb = b.evalm();
			if (is_a<matrix>(evma) && ex_to<matrix>(evma).is_zero_matrix())
				return 0;
			if (is_a<matrix>(evmb) && ex_to<matrix>(evmb).is_zero_matrix())
				return 0;

			if (need_to_hold(evma) || need_to_hold(evmb))
				return double_dot(evma, evmb).hold();

			if (is_a<matrix>(evma) && is_a<matrix>(evmb))
			{
				matrix ma = ex_to<matrix>(evma);
				matrix mb = ex_to<matrix>(evmb);

				if (ma.cols() != mb.cols() || ma.rows() != mb.rows())
				{
					GiNaC::ex ret;
					for (unsigned int i = 0; i < std::min(ma.rows(), mb.rows()); i++)
					{
						for (unsigned int j = 0; j < std::min(ma.cols(), mb.cols()); j++)
						{
							ret += ma(i, j) * mb(i, j);
						}
					}
					return ret;
					/*		  std::ostringstream oss;
							  oss << std::endl << " a = " << ma << std::endl << " b = " << mb << std::endl;
							  throw_runtime_error("Contraction between differently sized matrices, got"+oss.str());*/
				}
				GiNaC::ex ret;
				for (unsigned int i = 0; i < ma.rows(); i++)
				{
					for (unsigned int j = 0; j < ma.cols(); j++)
					{
						ret += ma(i, j) * mb(i, j);
					}
				}
				return ret;
			}
			throw_runtime_error("double_dot is only allowed between vectors or matrices");
		}

		REGISTER_FUNCTION(double_dot, eval_func(double_dot_eval).set_return_type(GiNaC::return_types::commutative))

		////

		static ex contract_eval(const ex &a, const ex &b)
		{
			if (pyoomph::pyoomph_verbose)
				std::cout << "Trying to eval contract dot of " << std::endl
						  << a << std::endl
						  << b << std::endl
						  << std::endl;
			ex evma = a.evalm();
			ex evmb = b.evalm();
			if (is_a<matrix>(evma) && ex_to<matrix>(evma).is_zero_matrix())
				return 0;
			if (is_a<matrix>(evmb) && ex_to<matrix>(evmb).is_zero_matrix())
				return 0;

			if (need_to_hold(evma) || need_to_hold(evmb))
				return contract(evma, evmb).hold();

			if (is_a<matrix>(evma) && is_a<matrix>(evmb))
			{
				matrix ma = ex_to<matrix>(evma);
				matrix mb = ex_to<matrix>(evmb);
				// std::cout << "COL ROW INFO" << ma.cols() << " " << mb.cols() << "    " << ma.rows() << "  " << mb.rows() << std::endl;
				if (ma.cols() == 1 && mb.cols() == 1)
					return dot(evma, evmb);
				else if (ma.cols() == 1)
				{
					std::vector<GiNaC::ex> v(std::min(ma.rows(), mb.rows()), 0);
					for (unsigned int i = 0; i < v.size(); i++)
					{
						for (unsigned int j = 0; j < mb.rows(); j++)
							v[i] += ma(j, 0) * mb(i, j);
					}
					return 0 + GiNaC::matrix(v.size(), 1, GiNaC::lst(v.begin(), v.end()));
				}
				else if (mb.cols() == 1)
				{
					//				std::cout << "COL ROW INFO" << ma.cols() << " " << mb.cols() << "    " << ma.rows() << "  " << mb.rows() << std::endl;
					//				std::cout << "a=" << ma << "  " <<"  b=" << mb   << std::endl;
					std::vector<GiNaC::ex> v(std::min(ma.rows(), mb.rows()), 0);
					for (unsigned int i = 0; i < v.size(); i++)
					{
						for (unsigned int j = 0; j < ma.rows(); j++)
						{
							//		      std::cout << "  CONTRIB " << i << "  " << j << "  is " << mb(j,0) << "  " << ma(j,i) << std::endl;
							v[i] += mb(j, 0) * ma(j, i);
						}
						//  				std::cout << "v[" <<i << "] = " << v[i]   << std::endl;
					}

					return 0 + GiNaC::matrix(v.size(), 1, GiNaC::lst(v.begin(), v.end()));
				}
				else
					return double_dot(evma, evmb);
			}
			if (!is_a<matrix>(evma) && !is_a<matrix>(evmb))
			{
				return evma * evmb;
			}
			else
			{
				return evma * evmb;
			}
			std::ostringstream oss;
			oss << std::endl
				<< " a = " << evma << std::endl
				<< " b = " << evmb << std::endl;
			throw_runtime_error("Cannot contract the following:" + oss.str());
			return 0;
		}

		REGISTER_FUNCTION(contract, eval_func(contract_eval).set_return_type(GiNaC::return_types::commutative))

		////////////////

		static ex single_index_eval(const ex &v, const ex &i)
		{
			ex evmv = v.evalm();
			if (need_to_hold(evmv))
				return single_index(evmv, i).hold();

			if (is_a<matrix>(evmv))
			{
				matrix ma = ex_to<matrix>(evmv);
				int _i = GiNaC::ex_to<GiNaC::numeric>(i.evalf()).to_double();
				if (ma.cols() == 1)
				{
					return ma(_i, 0);
				}
				else
				{
					matrix subvec(3, 1);
					for (unsigned _j = 0; _j < ma.rows(); _j++)
					{
						subvec[_j] = ma(_i, _j);
					}
					return subvec;
				}
			}
			throw_runtime_error("single_index cannot be applied on a non-matrix/vector object");
		}

		REGISTER_FUNCTION(single_index, eval_func(single_index_eval).set_return_type(GiNaC::return_types::commutative))

		//

		static ex double_index_eval(const ex &v, const ex &i, const ex &j)
		{
			ex evmv = v.evalm();
			if (need_to_hold(evmv))
				return double_index(evmv, i, j).hold();
			if (is_a<matrix>(evmv))
			{
				matrix ma = ex_to<matrix>(evmv);
				int _i = GiNaC::ex_to<GiNaC::numeric>(i.evalf()).to_double();
				int _j = GiNaC::ex_to<GiNaC::numeric>(j.evalf()).to_double();
				return ma(_i, _j);
			}
			throw_runtime_error("double_index cannot be applied on a non-matrix object");
		}

		REGISTER_FUNCTION(double_index, eval_func(double_index_eval).set_return_type(GiNaC::return_types::commutative))

		////////////////

		static int get_nontrivial_matrix_dimension(const matrix &ma)
		{
			int _n=std::min(ma.cols(),ma.rows());			
			while (_n>0)
			{
				bool found_nonzero=false;
				for (unsigned int i=0;i<_n;i++)
				{							
					if (!ma(i,_n-1).is_zero())
					{
						found_nonzero=true;
						break;
					}
					if (!ma(_n-1,i).is_zero())
					{
						found_nonzero=true;
						break;
					}
				}
				if (found_nonzero) break;
				_n--;						
			}
			return _n;
		}

		static ex determinant_eval(const ex &v,const ex &n)
		{
			ex evmv = v.evalm();
			if (need_to_hold(evmv))
				return determinant(evmv,n).hold();

			if (is_a<matrix>(evmv))
			{
				int _n = GiNaC::ex_to<GiNaC::numeric>(n.evalf()).to_double();
				matrix ma = ex_to<matrix>(evmv);
				//std::cout << "N IS " << _n << std::endl;
				if (_n<0) return ma.determinant(); // Determinant of the whole matrix
				else if (_n==0)
				{					
					_n=get_nontrivial_matrix_dimension(ma);										
					if (_n==0) return 0;
				}
								
				// Extract the block
				if (ma.cols() < _n) throw_runtime_error("Block size is larger than the matrix (colums)");
				if (ma.rows() < _n) throw_runtime_error("Block size is larger than the matrix (rows)");
				std::vector<GiNaC::ex> entries;
				for (unsigned int i = 0; i < _n; i++)
				{
					for (unsigned int j = 0; j < _n; j++)
					{
						entries.push_back(ma(i, j));
					}
				}
				GiNaC::lst entries_lst(GiNaC::lst(entries.begin(), entries.end()));
				return GiNaC::matrix(_n, _n, entries_lst).determinant();
				
			}
			throw_runtime_error("determinant cannot be applied on a non-matrix/vector object");
		}

		REGISTER_FUNCTION(determinant, eval_func(determinant_eval).set_return_type(GiNaC::return_types::commutative))

		////////////////


		static ex inverse_matrix_eval(const ex &v,const ex &n,const ex &flags)
		{
			ex evmv = v.evalm();
			if (need_to_hold(evmv))
				return inverse_matrix(evmv,n,flags).hold();

			if (is_a<matrix>(evmv))
			{
				int _n = GiNaC::ex_to<GiNaC::numeric>(n.evalf()).to_double();
				int flag = GiNaC::ex_to<GiNaC::numeric>(flags.evalf()).to_double();
				matrix ma = ex_to<matrix>(evmv);

				if (flag & 4) // Identify the zero columns and rows
				{
					if (ma.cols()!=ma.rows())
					{
						throw_runtime_error("Matrix is not square, cannot identify zero rows/columns");
					}
					std::vector<bool> zero_cols(ma.cols(), true);
					unsigned new_n=ma.cols();
					unsigned zero_count=0;
					for (unsigned int i = 0; i < ma.rows(); i++)
					{
						for (unsigned int j = 0; j < ma.cols(); j++)
						{
							if (!ma(i, j).is_zero())
							{
								zero_cols[j] = false;
								zero_cols[i] = false;
							}
						}
					}
					for (unsigned int i = 0; i < ma.rows(); i++)
					{
						if (zero_cols[i])
						{
							//std::cout << "Zero row/column found at index " << i << std::endl;
							new_n-=1;
							zero_count+=1;
						}
					}
					std::vector<GiNaC::ex> nonzero_entries;
					for (unsigned int i = 0; i < ma.rows(); i++)
					{
						for (unsigned int j = 0; j < ma.cols(); j++)
						{
							if (!zero_cols[i] && !zero_cols[j])
							{
								nonzero_entries.push_back(ma(i, j));
							}
						}
					}
					GiNaC::lst nonzero_entries_lst(GiNaC::lst(nonzero_entries.begin(), nonzero_entries.end()));
					if (new_n*new_n!= nonzero_entries.size()) throw_runtime_error("Something strange happened, the number of non-zero entries does not match the expected size");
					//std::cout << "Reduced matrix size is " << new_n << " and _n=" << _n << " and n=" << n << std::endl;
				  	GiNaC::matrix reduced_mat(new_n, new_n, nonzero_entries_lst);				
					GiNaC::ex reduced_ex =inverse_matrix_eval(reduced_mat,n,flags-4-2);
					reduced_mat= GiNaC::ex_to<GiNaC::matrix>(reduced_ex.evalm());
					unsigned int refilled_size=std::min((unsigned)3,reduced_mat.rows()+zero_count);
					std::vector<GiNaC::ex> refilled_entries(refilled_size*refilled_size,0);
					unsigned ii=0;
					for (unsigned int i = 0; i < ma.rows(); i++)
					{
						if (zero_cols[i])	continue;
						unsigned jj=0;
						for (unsigned int j = 0; j < ma.cols(); j++)
						{
							if (zero_cols[j]) continue;							
							refilled_entries[i*refilled_size+j] = reduced_mat(ii,jj);
							jj++;
						}
						ii++;
					}
					return GiNaC::matrix(refilled_size,refilled_size, GiNaC::lst(refilled_entries.begin(), refilled_entries.end()));				
				}
				
				if (_n<0) return ma.inverse(); // Inverse of the whole matrix
				else if (_n==0)
				{					
					_n=get_nontrivial_matrix_dimension(ma);										
					if (_n==0) throw_runtime_error("Matrix is empty and cannot be inverted");
				}
												
				if (ma.cols() < _n) throw_runtime_error("Block size is larger than the matrix (colums)");
				if (ma.rows() < _n) throw_runtime_error("Block size is larger than the matrix (rows)");
				std::vector<GiNaC::ex> entries;
				if (_n==1) { entries.push_back(1/ma(0,0));}
				else if (_n==2) {
					GiNaC::ex det=ma(0,0)*ma(1,1)-ma(0,1)*ma(1,0);
					if (flag & 1) det=subexpression(det);
					entries.push_back(ma(1,1)/det);
					entries.push_back(-ma(0,1)/det);
					entries.push_back(-ma(1,0)/det);
					entries.push_back(ma(0,0)/det);
				}
				else if (_n==3) {
					GiNaC::ex det=ma(0,0)*(ma(1,1)*ma(2,2)-ma(1,2)*ma(2,1))
						-ma(0,1)*(ma(1,0)*ma(2,2)-ma(1,2)*ma(2,0))
						+ma(0,2)*(ma(1,0)*ma(2,1)-ma(1,1)*ma(2,0));
					if (flag & 1) det=subexpression(det);
					entries.push_back((ma(1,1)*ma(2,2)-ma(1,2)*ma(2,1))/det);
					entries.push_back((ma(0,2)*ma(2,1)-ma(0,1)*ma(2,2))/det);
					entries.push_back((ma(0,1)*ma(1,2)-ma(0,2)*ma(1,1))/det);
					entries.push_back((ma(1,2)*ma(2,0)-ma(1,0)*ma(2,2))/det);
					entries.push_back((ma(0,0)*ma(2,2)-ma(0,2)*ma(2,0))/det);
					entries.push_back((ma(0,2)*ma(1,0)-ma(0,0)*ma(1,2))/det);
					entries.push_back((ma(1,0)*ma(2,1)-ma(1,1)*ma(2,0))/det);
					entries.push_back((ma(0,1)*ma(2,0)-ma(0,0)*ma(2,1))/det);
					entries.push_back((ma(0,0)*ma(1,1)-ma(0,1)*ma(1,0))/det);					
				}
				
				if (flag & 2==0)
				{
				  GiNaC::lst entries_lst(GiNaC::lst(entries.begin(), entries.end()));
				  return GiNaC::matrix(_n, _n, entries_lst);				
				}
				else
				{
					//Return a 3x3 matrix with zero filling
					std::vector<GiNaC::ex> entries3x3(9,0);
					for (unsigned int i = 0; i < _n; i++)
					{
						for (unsigned int j = 0; j < _n; j++)
						{
						  entries3x3[3*i+j] = entries[_n*i+j];
						}
					}
					// TODO: Diag to 1 ?
					GiNaC::lst entries_lst(GiNaC::lst(entries3x3.begin(), entries3x3.end()));
					return GiNaC::matrix(3, 3, entries_lst);				
				}
			}
			throw_runtime_error("inverse_matrix cannot be applied on a non-matrix/vector object");
		}

		REGISTER_FUNCTION(inverse_matrix, eval_func(inverse_matrix_eval).set_return_type(GiNaC::return_types::noncommutative))


		////////////////
		

		static ex subexpression_eval(const ex &wrapped)
		{
			if (GiNaC::is_a<GiNaC::constant>(wrapped) || GiNaC::is_a<GiNaC::numeric>(wrapped))
				return wrapped; // TODO: Simplify something like 3/4*Pi
			else if (is_ex_the_function(wrapped, subexpression))
				return wrapped; // Simplify these
			else
			{
				GiNaC::ex evm = wrapped.evalm();
				if (GiNaC::is_a<GiNaC::matrix>(evm))
				{
					GiNaC::matrix inp = GiNaC::ex_to<GiNaC::matrix>(evm);
					GiNaC::matrix res(inp.rows(), inp.cols());
					for (unsigned int r = 0; r < inp.rows(); r++)
					{
						for (unsigned int c = 0; c < inp.cols(); c++)
						{
							res(r, c) = 0 + subexpression(inp(r, c));
						}
					}
					return 0 + res;
				}
				return subexpression(wrapped).hold();
			}
		}

		static ex subexpression_evalf(const ex &wrapped)
		{

			if (GiNaC::is_a<GiNaC::constant>(wrapped) || GiNaC::is_a<GiNaC::numeric>(wrapped))
				return wrapped.evalf(); // TODO: Simplify something like 3/4*Pi
			else if (is_ex_the_function(wrapped, subexpression))
				return wrapped.evalf(); // Simplify these
			else
				return subexpression(wrapped).hold();
		}

		static ex subexpression_deriv(const ex &wrapped, unsigned deriv_arg)
		{
			throw_runtime_error("Cannot derive a subexpression");
		}

		static ex subexpression_expl_deriv(const ex &wrapped, const symbol &deriv_arg)
		{
			return subexpression(wrapped.diff(deriv_arg));
		}

		REGISTER_FUNCTION(subexpression, eval_func(subexpression_eval).evalf_func(subexpression_evalf).derivative_func(subexpression_deriv).expl_derivative_func(subexpression_expl_deriv))

		////////////////

		static ex get_real_part_eval(const ex &wrapped)
		{
			if (GiNaC::is_a<GiNaC::constant>(wrapped) || GiNaC::is_a<GiNaC::numeric>(wrapped))
				return GiNaC::real_part(wrapped);
			else
			{
				GiNaC::ex evm = wrapped.evalm();
				if (GiNaC::is_a<GiNaC::matrix>(evm))
				{
					GiNaC::matrix inp = GiNaC::ex_to<GiNaC::matrix>(evm);
					GiNaC::matrix res(inp.rows(), inp.cols());
					for (unsigned int r = 0; r < inp.rows(); r++)
					{
						for (unsigned int c = 0; c < inp.cols(); c++)
						{
							res(r, c) = 0 + get_real_part(inp(r, c));
						}
					}
					return 0 + res;
				}
				else if (need_to_hold(wrapped))
				{
					if (pyoomph::pyoomph_verbose)
						std::cout << " GETTING REAL PART OF  " << wrapped << std::endl;
					return GiNaC::real_part(wrapped);
				}
				else
					return get_real_part(wrapped).hold();
			}
		}

		static ex get_real_part_evalf(const ex &wrapped)
		{
			return get_real_part_eval(wrapped).evalf();
		}

		static ex get_real_part_deriv(const ex &wrapped, unsigned deriv_arg)
		{
			throw_runtime_error("Cannot derive get_real_part");
		}

		static ex get_real_part_expl_deriv(const ex &wrapped, const symbol &deriv_arg)
		{
			return get_real_part(wrapped.diff(deriv_arg));
		}

		REGISTER_FUNCTION(get_real_part, eval_func(get_real_part_eval).evalf_func(get_real_part_evalf).derivative_func(get_real_part_deriv).expl_derivative_func(get_real_part_expl_deriv))

		////////////////

		static ex get_imag_part_eval(const ex &wrapped)
		{
			if (GiNaC::is_a<GiNaC::constant>(wrapped) || GiNaC::is_a<GiNaC::numeric>(wrapped))
				return GiNaC::imag_part(wrapped);
			else
			{
				GiNaC::ex evm = wrapped.evalm();
				if (GiNaC::is_a<GiNaC::matrix>(evm))
				{
					GiNaC::matrix inp = GiNaC::ex_to<GiNaC::matrix>(evm);
					GiNaC::matrix res(inp.rows(), inp.cols());
					for (unsigned int r = 0; r < inp.rows(); r++)
					{
						for (unsigned int c = 0; c < inp.cols(); c++)
						{
							res(r, c) = 0 + get_imag_part(inp(r, c));
						}
					}
					return 0 + res;
				}
				else if (need_to_hold(wrapped))
					return GiNaC::imag_part(wrapped);
				else
					return get_imag_part(wrapped).hold();
			}
		}

		static ex get_imag_part_evalf(const ex &wrapped)
		{
			return get_imag_part_eval(wrapped).evalf();
		}

		static ex get_imag_part_deriv(const ex &wrapped, unsigned deriv_arg)
		{
			throw_runtime_error("Cannot derive get_imag_part");
		}

		static ex get_imag_part_expl_deriv(const ex &wrapped, const symbol &deriv_arg)
		{
			return get_imag_part(wrapped.diff(deriv_arg));
		}

		REGISTER_FUNCTION(get_imag_part, eval_func(get_imag_part_eval).evalf_func(get_imag_part_evalf).derivative_func(get_imag_part_deriv).expl_derivative_func(get_imag_part_expl_deriv))


		////////////////

		class SubExpressionsToRealAndImag : public GiNaC::map_function
		{
		public:
			GiNaC::ex operator()(const GiNaC::ex & inp)
			{
				if (is_ex_the_function(inp, expressions::subexpression))
				{
					GiNaC::ex mapped_ex = inp.op(0).map(*this);
					if (GiNaC::is_zero(GiNaC::imag_part(mapped_ex)))
					{
						return inp.map(*this);
					}
					else
					{
						return (pyoomph::expressions::subexpression(GiNaC::real_part(mapped_ex)) + GiNaC::I * pyoomph::expressions::subexpression(GiNaC::imag_part(mapped_ex))).map(*this);
					}

				}
				else 
				{
					return inp.map(*this);
				}
			}
		};


		static ex split_subexpressions_in_real_and_imaginary_parts_eval(const ex &wrapped)
		{
			if (need_to_hold(wrapped))
				return split_subexpressions_in_real_and_imaginary_parts(wrapped).hold();
			else
			{
				SubExpressionsToRealAndImag repl;
				return repl(wrapped);
			}
		}


		REGISTER_FUNCTION(split_subexpressions_in_real_and_imaginary_parts, eval_func(split_subexpressions_in_real_and_imaginary_parts_eval))


		////////////////

		static ex Diff_eval(const ex &arg, const ex &wrto)
		{
			if (pyoomph::pyoomph_verbose)
				std::cout << "ENTERING DIFF " << arg << " wrtO " << wrto << std::endl;
			if (arg.is_zero() || GiNaC::is_a<GiNaC::numeric>(arg))
				return 0;

			if (need_to_hold(arg) || need_to_hold(wrto))
			{
				return Diff(arg, wrto).hold(); // Do not differentiate fields until required
			}
			else
			{
				if (pyoomph::pyoomph_verbose)
					std::cout << " DIFF NOT HOLD" << std::endl;
				return diff(arg, wrto);
			}
		}

		static ex Diff_expl_deriv(const ex &arg, const ex &wrto, const symbol &deriv_arg)
		{
			return Diff(arg, wrto).hold(); // TODO always best way?
		}

		REGISTER_FUNCTION(Diff, eval_func(Diff_eval).expl_derivative_func(Diff_expl_deriv).set_return_type(GiNaC::return_types::commutative))

		static ex testfunction_eval(const ex &name, const ex &resolve)
		{
			return testfunction(name, resolve).hold();
		}

		REGISTER_FUNCTION(testfunction, eval_func(testfunction_eval).set_return_type(GiNaC::return_types::commutative))

		static ex dimtestfunction_eval(const ex &name, const ex &resolve)
		{
			return dimtestfunction(name, resolve).hold();
		}

		REGISTER_FUNCTION(dimtestfunction, eval_func(dimtestfunction_eval).set_return_type(GiNaC::return_types::commutative))

		static ex scale_eval(const ex &name, const ex &resolve)
		{
			return scale(name, resolve).hold();
		}

		REGISTER_FUNCTION(scale, eval_func(scale_eval).set_return_type(GiNaC::return_types::commutative))

		static ex test_scale_eval(const ex &name, const ex &resolve)
		{
			return test_scale(name, resolve).hold();
		}

		REGISTER_FUNCTION(test_scale, eval_func(test_scale_eval).set_return_type(GiNaC::return_types::commutative))

		static ex field_eval(const ex &name, const ex &resolve)
		{
			return field(name, resolve).hold();
		}

		REGISTER_FUNCTION(field, eval_func(field_eval).set_return_type(GiNaC::return_types::commutative))

		static ex eval_flag_eval(const ex &name)
		{
			return eval_flag(name).hold();
		}

		REGISTER_FUNCTION(eval_flag, eval_func(eval_flag_eval).set_return_type(GiNaC::return_types::commutative))

		static ex nondimfield_eval(const ex &name, const ex &resolve)
		{
			return nondimfield(name, resolve).hold();
		}

		REGISTER_FUNCTION(nondimfield, eval_func(nondimfield_eval).set_return_type(GiNaC::return_types::commutative))

		static ex eval_in_domain_eval(const ex &expr, const ex &resolve)
		{
			return eval_in_domain(expr, resolve).hold();
		}

		REGISTER_FUNCTION(eval_in_domain, eval_func(eval_in_domain_eval).set_return_type(GiNaC::return_types::commutative))

		//****
		class ReplaceGlobalParamsByCurrentValues : public GiNaC::map_function
		{
		public:
			GiNaC::ex operator()(const GiNaC::ex &inp)
			{
				if (GiNaC::is_a<GiNaC::GiNaCGlobalParameterWrapper>(inp))
				{
					GiNaC::GiNaCGlobalParameterWrapper se = GiNaC::ex_to<GiNaC::GiNaCGlobalParameterWrapper>(inp);
					auto &sp = se.get_struct();
					return sp.cme->value();
				}
				else
				{
					return inp.map(*this);
				}
			}
		};

		GiNaC::ex replace_global_params_by_current_values(const GiNaC::ex &in)
		{
			ReplaceGlobalParamsByCurrentValues repl;
			return repl(in);
		}

		class EvaluateShapeExpansionsInPast : public GiNaC::map_function
		{
		protected:
			int index;
			bool is_int;
			double frac_part;
			int partial_t_action; // 0: No change, 1: change all partial_t schemes of first order to BDF1
		public:
			EvaluateShapeExpansionsInPast(int _index, int tstep_action) : index(_index), is_int(true), frac_part(0), partial_t_action(tstep_action)
			{
				if (index > 2)
				{
					throw_runtime_error("Cannot evaluate earlier in past than two steps");
				}
			}
			EvaluateShapeExpansionsInPast(double frac, int tstep_action) : index(std::floor(frac)), is_int(false), frac_part(frac - std::floor(frac)), partial_t_action(tstep_action)
			{
				if (index > 2 || (index > 1 && frac > 0))
				{
					throw_runtime_error("Cannot evaluate earlier in past than two steps");
				}
				if (frac < 1e-9)
				{
					frac_part = 0;
					is_int = true;
				}
				if (frac > 1 - 1e-9)
				{
					frac_part = 0;
					is_int = true;
					index++;
				}
			}
			GiNaC::ex operator()(const GiNaC::ex &inp)
			{
				if (GiNaC::is_a<GiNaC::GiNaCShapeExpansion>(inp))
				{
					GiNaC::GiNaCShapeExpansion se = GiNaC::ex_to<GiNaC::GiNaCShapeExpansion>(inp);
					auto &sp = se.get_struct();
					ShapeExpansion sp_past = sp;
					if (partial_t_action && sp.dt_order)
					{
						// Change to BDF1, but do not evaluate in past
						if (sp.dt_order == 1)
						{
							if (partial_t_action == 1)
							{
								sp_past.dt_scheme = "BDF1";
							}
							else if (partial_t_action == 2)
							{
								sp_past.dt_scheme = "BDF2";
							}
							else if (partial_t_action == 3)
							{
								sp_past.dt_scheme = "Newmark2";
							}
							else
							{
								throw_runtime_error("Wrong argument for the partial_t_action");
							}
						}
						return GiNaC::GiNaCShapeExpansion(sp_past);
					}
					sp_past.time_history_index = index;
					if (is_int)
					{
						return GiNaC::GiNaCShapeExpansion(sp_past);
					}
					else
					{
						ShapeExpansion sp_past1 = sp;
						sp_past1.time_history_index = index + 1;
						return (1 - frac_part) * GiNaC::GiNaCShapeExpansion(sp_past) + frac_part * GiNaC::GiNaCShapeExpansion(sp_past1);
					}
				}
				if (GiNaC::is_a<GiNaC::GiNaCTimeSymbol>(inp))
				{
					GiNaC::GiNaCTimeSymbol se = GiNaC::ex_to<GiNaC::GiNaCTimeSymbol>(inp);
					auto &sp = se.get_struct();
					TimeSymbol sp_past = sp;
					sp_past.index = index;
					if (is_int)
					{
						return GiNaC::GiNaCTimeSymbol(sp_past);
					}
					else
					{
						TimeSymbol sp_past1 = sp;
						sp_past1.index = index + 1;
						return (1 - frac_part) * GiNaC::GiNaCTimeSymbol(sp_past) + frac_part * GiNaC::GiNaCTimeSymbol(sp_past1);
					}
				}
				// TODO: Also dx for ALE
				else
				{
					return inp.map(*this);
				}
			}
		};

		static ex eval_in_past_eval(const ex &expr, const ex &index, const ex &tstep_action)
		{
			if (need_to_hold(expr))
				return eval_in_past(expr, index, tstep_action).hold();

			if (!GiNaC::is_a<GiNaC::numeric>(index))
			{
				throw_runtime_error("Cannot use evaluate_in_past(expression,timeoffset,timestepper_action) with a non-numeric timeoffset");
			}
			if (!GiNaC::is_a<GiNaC::numeric>(tstep_action))
			{
				throw_runtime_error("Cannot use evaluate_in_past(expression,timeoffset,timestepper_action) with a non-numeric timestepper_action (0: nothing, 1: convert all to BDF1)");
			}
			GiNaC::numeric index_n = GiNaC::ex_to<GiNaC::numeric>(index);
			GiNaC::numeric index_ts = GiNaC::ex_to<GiNaC::numeric>(tstep_action);
			if (index_n.is_zero() && index_ts.is_zero())
			{
				return expr;
			}
			else if (index_n.is_pos_integer())
			{
				EvaluateShapeExpansionsInPast in_past(index_n.to_int(), index_ts.to_int());
				return in_past(expr);
			}
			else if (!index_n.is_negative())
			{
				EvaluateShapeExpansionsInPast in_past(index_n.to_double(), index_ts.to_int());
				return in_past(expr);
			}
			else
			{
				throw_runtime_error("Cannot use evaluate_in_past(expression,timeoffet,timestepper_action) with a non positive  timeoffset");
			}
		}

		REGISTER_FUNCTION(eval_in_past, eval_func(eval_in_past_eval).set_return_type(GiNaC::return_types::commutative))
		//****

		class EvaluateShapeExpansionsAtExpansionMode : public GiNaC::map_function
		{
		protected:
			int index;

		public:
			EvaluateShapeExpansionsAtExpansionMode(int _index) : index(_index) {}
			GiNaC::ex operator()(const GiNaC::ex &inp)
			{
				if (GiNaC::is_a<GiNaC::GiNaCShapeExpansion>(inp))
				{
					GiNaC::GiNaCShapeExpansion se = GiNaC::ex_to<GiNaC::GiNaCShapeExpansion>(inp);
					ShapeExpansion sp = se.get_struct();
					if (sp.expansion_mode == index)
						return inp;
					else
					{
						sp.expansion_mode = index;
						return GiNaC::GiNaCShapeExpansion(sp);
					}
				}
				if (GiNaC::is_a<GiNaC::GiNaCNormalSymbol>(inp))
				{
					GiNaC::GiNaCNormalSymbol se = GiNaC::ex_to<GiNaC::GiNaCNormalSymbol>(inp);
					NormalSymbol sp = se.get_struct();
					if (sp.expansion_mode == index) //  || sp.is_eigenexpansion
						return inp;
					else
					{
						sp.expansion_mode = index;
						return GiNaC::GiNaCNormalSymbol(sp);
					}
				}
				if (GiNaC::is_a<GiNaC::GiNaCSpatialIntegralSymbol>(inp))
				{
					GiNaC::GiNaCSpatialIntegralSymbol se = GiNaC::ex_to<GiNaC::GiNaCSpatialIntegralSymbol>(inp);
					SpatialIntegralSymbol sp = se.get_struct();
					if (sp.expansion_mode == index) //  || sp.is_eigenexpansion
						return inp;
					else
					{
						sp.expansion_mode = index;
						return GiNaC::GiNaCSpatialIntegralSymbol(sp);
					}
				}
				else
				{
					return inp.map(*this);
				}
			}
		};

		static ex eval_at_expansion_mode_eval(const ex &expr, const ex &index)
		{
			if (need_to_hold(expr))
				return eval_at_expansion_mode(expr, index).hold();

			if (!GiNaC::is_a<GiNaC::numeric>(index))
			{
				throw_runtime_error("Cannot use eval_at_expansion_mode(expression,index) with a non-numeric index");
			}
			GiNaC::numeric index_n = GiNaC::ex_to<GiNaC::numeric>(index);
			if (index_n.is_zero())
			{
				return expr;
			}
			else
			{
				EvaluateShapeExpansionsAtExpansionMode at_mode(index_n.to_int());
				return at_mode(expr);
			}
		}

		REGISTER_FUNCTION(eval_at_expansion_mode, eval_func(eval_at_expansion_mode_eval).set_return_type(GiNaC::return_types::commutative))
		//****

		static ex symbol_subs_eval(const ex &expr, const ex &what, const ex &by_what)
		{
			if (need_to_hold(expr))
				return symbol_subs(expr, what, by_what).hold();
			GiNaC::exmap mapping;
			if (GiNaC::is_a<GiNaC::lst>(what))
			{
				if (!GiNaC::is_a<GiNaC::lst>(by_what))
				{
					throw_runtime_error("symbol_subst got a list as sources, but not a list as destinations");
				}
				if (what.nops() != by_what.nops())
				{
					throw_runtime_error("symbol_subst got a lists of different sizes");
				}
				for (unsigned int i = 0; i < what.nops(); i++)
					mapping[what[i]] = by_what[i];
			}
			else if (GiNaC::is_a<GiNaC::lst>(by_what))
			{
				throw_runtime_error("symbol_subst got a list as destinations, but not a list as sources");
			}
			else
			{
				mapping[what] = by_what;
			}
			return expr.subs(mapping);
		}
		REGISTER_FUNCTION(symbol_subs, eval_func(symbol_subs_eval).set_return_type(GiNaC::return_types::commutative))

		class DeactivateJacobianOfExpansionMode : public GiNaC::map_function
		{
		protected:
			int index, flag;

		public:
			DeactivateJacobianOfExpansionMode(int _index, int _flag) : index(_index), flag(_flag) {}
			GiNaC::ex operator()(const GiNaC::ex &inp)
			{
				if (GiNaC::is_a<GiNaC::GiNaCShapeExpansion>(inp))
				{
					GiNaC::GiNaCShapeExpansion se = GiNaC::ex_to<GiNaC::GiNaCShapeExpansion>(inp);
					ShapeExpansion sp = se.get_struct();
					if (sp.expansion_mode != index)
						return inp;
					else
					{
						if (flag == 0 || flag == 1)
							sp.no_jacobian = true;
						if (flag == 0 || flag == 2)
							sp.no_hessian = true;
						return GiNaC::GiNaCShapeExpansion(sp);
					}
				}
				else if (GiNaC::is_a<GiNaC::GiNaCNormalSymbol>(inp))
				{
					GiNaC::GiNaCNormalSymbol se = GiNaC::ex_to<GiNaC::GiNaCNormalSymbol>(inp);
					NormalSymbol sp = se.get_struct();
					if (sp.expansion_mode != index)
						return inp;
					else
					{
						if (flag == 0 || flag == 1)
							sp.no_jacobian = true;
						if (flag == 0 || flag == 2)
							sp.no_hessian = true;
						return GiNaC::GiNaCNormalSymbol(sp);
					}
				}
				else if (GiNaC::is_a<GiNaC::GiNaCSpatialIntegralSymbol>(inp))
				{
					GiNaC::GiNaCSpatialIntegralSymbol se = GiNaC::ex_to<GiNaC::GiNaCSpatialIntegralSymbol>(inp);
					SpatialIntegralSymbol sp = se.get_struct();
					if (sp.expansion_mode != index)
						return inp;
					else
					{
						if (flag == 0 || flag == 1)
							sp.no_jacobian = true;
						if (flag == 0 || flag == 2)
							sp.no_hessian = true;
						return GiNaC::GiNaCSpatialIntegralSymbol(sp);
					}
				}

				else
				{
					return inp.map(*this);
				}
			}
		};

		static ex remove_mode_from_jacobian_or_hessian_eval(const ex &expr, const ex &which, const ex &flag)
		{
			if (need_to_hold(expr))
				return remove_mode_from_jacobian_or_hessian(expr, which, flag).hold();

			if (!GiNaC::is_a<GiNaC::numeric>(which) || !GiNaC::is_a<GiNaC::numeric>(flag))
			{
				throw_runtime_error("Cannot use eval_at_expansion_mode(expression,index,flag) with a non-numeric index or flag");
			}
			GiNaC::numeric index_n = GiNaC::ex_to<GiNaC::numeric>(which);
			GiNaC::numeric flag_n = GiNaC::ex_to<GiNaC::numeric>(flag); // flag 0: remove from jacobian and hessian, 1: only remove jacobian, 2: only remove hessian
			DeactivateJacobianOfExpansionMode set_mode_jacobian_zero(index_n.to_int(), flag_n.to_int());
			return set_mode_jacobian_zero(expr);
		}
		REGISTER_FUNCTION(remove_mode_from_jacobian_or_hessian, eval_func(remove_mode_from_jacobian_or_hessian_eval).set_return_type(GiNaC::return_types::commutative))

		//****

		static ex time_stepper_weight_eval(const ex &order, const ex &index, const ex &scheme)
		{
			return time_stepper_weight(order, index, scheme).hold();
		}

		static void time_stepper_weight_eval_csrc_float(const ex &order, const ex &index, const ex &scheme, const print_context &c)
		{
			int iorder = GiNaC::ex_to<GiNaC::numeric>(order).to_double();
			int iindex = GiNaC::ex_to<GiNaC::numeric>(index).to_double();
			std::ostringstream oss;
			oss << scheme;
			std::string scheme_str = oss.str();
			if (scheme_str != "BDF1")
			{
				throw_runtime_error("Strange time scheme");
			}
			if (iorder == 1)
			{
				c.s << "shapeinfo->timestepper_weights_dt_" + scheme_str + "[" << iindex << "]";
			}
			else
			{
				throw_runtime_error("Strange order in time stepper"); // TODO: Second order and symbol (0,0) for dt
			}
		}

		REGISTER_FUNCTION(time_stepper_weight, eval_func(time_stepper_weight_eval)
												   .print_func<print_csrc_float>(time_stepper_weight_eval_csrc_float)
												   .print_func<print_csrc_double>(time_stepper_weight_eval_csrc_float)
												   .set_return_type(GiNaC::return_types::commutative))

		static ex heaviside_eval(const ex &arg)
		{
			if (GiNaC::is_a<GiNaC::numeric>(arg))
			{
				double argd = GiNaC::ex_to<GiNaC::numeric>(arg).to_double();
				//  std::cout << "HEAVISIDE " << arg << "  " << argd << std::endl;
				if (argd > 0)
					return 1;
				else if (argd < 0)
					return 0;
				else
					return GiNaC::numeric(1, 2);
			}
			else
				return heaviside(arg).hold();
		}
		
		static ex heaviside_evalf(const ex &arg)
		{
			if (GiNaC::is_a<GiNaC::numeric>(arg))
			{
				double argd = GiNaC::ex_to<GiNaC::numeric>(arg).to_double();
				//  std::cout << "HEAVISIDE " << arg << "  " << argd << std::endl;
				if (argd > 0)
					return 1;
				else if (argd < 0)
					return 0;
				else
					return GiNaC::numeric(1, 2);
			}
			else
				return heaviside(arg).hold();
		}

		static ex heaviside_real_part(const ex &arg)
		{
			return heaviside(arg).hold();
		}

		static ex heaviside_imag_part(const ex &arg)
		{
			return 0;
		}

		static void heaviside_csrc_float(const ex &arg, const print_context &c)
		{
			c.s << "step(";
			arg.print(c);
			c.s << ")";
		}

		static ex heaviside_expl_derivative(const ex &arg, const symbol &deriv_arg)
		{
			//	 return arg.diff(deriv_arg)*heaviside(arg);
			return 0;
		}

		REGISTER_FUNCTION(heaviside, eval_func(heaviside_eval).evalf_func(heaviside_evalf)
										 .print_func<print_csrc_float>(heaviside_csrc_float)
										 .print_func<print_csrc_double>(heaviside_csrc_float)
										 .expl_derivative_func(heaviside_expl_derivative)
										 .set_return_type(GiNaC::return_types::commutative)
										 .real_part_func(heaviside_real_part)
										 .imag_part_func(heaviside_imag_part))

		static ex absolute_eval(const ex &arg)
		{
			if (GiNaC::is_a<GiNaC::numeric>(arg))
			{
				double darg = GiNaC::ex_to<GiNaC::numeric>(arg).to_double();
				if (darg > 0)
					return arg;
				else if (darg < 0)
					return -arg;
				else
					return 0;
			}
			else
				return absolute(arg).hold();
		}

		static void absolute_csrc_float(const ex &arg, const print_context &c)
		{
			c.s << "fabs(";
			arg.print(c);
			c.s << ")";
		}

		static ex absolute_expl_derivative(const ex &arg, const symbol &deriv_arg)
		{
			return arg.diff(deriv_arg) * signum(arg);
		}

		REGISTER_FUNCTION(absolute, eval_func(absolute_eval)
										.print_func<print_csrc_float>(absolute_csrc_float)
										.print_func<print_csrc_double>(absolute_csrc_float)
										.expl_derivative_func(absolute_expl_derivative)
										.set_return_type(GiNaC::return_types::commutative))

		static ex signum_eval(const ex &arg)
		{
			if (GiNaC::is_a<GiNaC::numeric>(arg))
			{
				double argv = GiNaC::ex_to<GiNaC::numeric>(arg).to_double();
				if (argv > 0)
					return 1;
				else if (argv < 0)
					return -1;
				else
					return 0;
			}
			else
				return signum(arg).hold();
		}

		static void signum_csrc_float(const ex &arg, const print_context &c)
		{
			c.s << "signum(";
			arg.print(c);
			c.s << ")";
		}

		static ex signum_expl_derivative(const ex &arg, const symbol &deriv_arg)
		{
			return 0; // TODO: Singularity, but this does not really matter here
		}

		REGISTER_FUNCTION(signum, eval_func(signum_eval)
									  .print_func<print_csrc_float>(signum_csrc_float)
									  .print_func<print_csrc_double>(signum_csrc_float)
									  .expl_derivative_func(signum_expl_derivative)
									  .set_return_type(GiNaC::return_types::commutative))

		static ex minimum_eval(const ex &a, const ex &b)
		{
			if (GiNaC::is_a<GiNaC::numeric>(a) && GiNaC::is_a<GiNaC::numeric>(b))
			{
				GiNaC::numeric A = GiNaC::ex_to<GiNaC::numeric>(a);
				GiNaC::numeric B = GiNaC::ex_to<GiNaC::numeric>(b);
				if (A <= B)
					return A;
				else
					return B;
			}
			return minimum(a, b).hold();
		}

		static ex minimum_evalf(const ex &a, const ex &b)
		{
			if (GiNaC::is_a<GiNaC::numeric>(a) && GiNaC::is_a<GiNaC::numeric>(b))
			{
				GiNaC::numeric A = GiNaC::ex_to<GiNaC::numeric>(a);
				GiNaC::numeric B = GiNaC::ex_to<GiNaC::numeric>(b);
				if (A <= B)
					return A;
				else
					return B;
			}
			return minimum(a, b).hold();
		}


		static ex minimum_real_part(const ex &arg1,const ex &arg2)
		{
			return minimum(arg1,arg2).hold();
		}

		static ex minimum_imag_part(const ex &arg1,const ex &arg2)
		{
			return 0;
		}

		static void minimum_csrc_float(const ex &a, const ex &b, const print_context &c)
		{
			c.s << "fmin(";
			a.print(c);
			c.s << ", ";
			b.print(c);
			c.s << ")";
		}

		static ex minimum_expl_derivative(const ex &a, const ex &b, const symbol &deriv_arg)
		{
			return a.diff(deriv_arg) * heaviside(b - a) + b.diff(deriv_arg) * heaviside(a - b);
		}

		REGISTER_FUNCTION(minimum, eval_func(minimum_eval)
									   .print_func<print_csrc_float>(minimum_csrc_float)
									   .print_func<print_csrc_double>(minimum_csrc_float)
									   .expl_derivative_func(minimum_expl_derivative)
									   .real_part_func(minimum_real_part).imag_part_func(minimum_imag_part)
									   .evalf_func(minimum_evalf)
									   .set_return_type(GiNaC::return_types::commutative))

		static ex maximum_evalf(const ex &a, const ex &b)
		{
			if (GiNaC::is_a<GiNaC::numeric>(a) && GiNaC::is_a<GiNaC::numeric>(b))
			{
				GiNaC::numeric A = GiNaC::ex_to<GiNaC::numeric>(a);
				GiNaC::numeric B = GiNaC::ex_to<GiNaC::numeric>(b);
				if (A <= B)
					return B;
				else
					return A;
			}
			return maximum(a, b).hold();
		}

		static ex maximum_eval(const ex &a, const ex &b)
		{
			if (GiNaC::is_a<GiNaC::numeric>(a) && GiNaC::is_a<GiNaC::numeric>(b))
			{
				GiNaC::numeric A = GiNaC::ex_to<GiNaC::numeric>(a);
				GiNaC::numeric B = GiNaC::ex_to<GiNaC::numeric>(b);
				if (A <= B)
					return B;
				else
					return A;
			}
			return maximum(a, b).hold();
		}

		static void maximum_csrc_float(const ex &a, const ex &b, const print_context &c)
		{
			c.s << "fmax(";
			a.print(c);
			c.s << ", ";
			b.print(c);
			c.s << ")";
		}

		static ex maximum_expl_derivative(const ex &a, const ex &b, const symbol &deriv_arg)
		{
			return a.diff(deriv_arg) * heaviside(a - b) + b.diff(deriv_arg) * heaviside(b - a);
		}

		static ex maximum_real_part(const ex &arg1,const ex &arg2)
		{
			return maximum(arg1,arg2).hold();
		}

		static ex maximum_imag_part(const ex &arg1,const ex &arg2)
		{
			return 0;
		}

		REGISTER_FUNCTION(maximum, eval_func(maximum_eval)
									   .print_func<print_csrc_float>(maximum_csrc_float)
									   .print_func<print_csrc_double>(maximum_csrc_float)
									   .expl_derivative_func(maximum_expl_derivative)
									   .real_part_func(maximum_real_part).imag_part_func(maximum_imag_part)
									   .evalf_func(maximum_evalf)
									   .set_return_type(GiNaC::return_types::commutative))

		static ex piecewise_geq0_eval(const ex &cond, const ex &a, const ex &b)
		{
			/*throw_runtime_error("PIECEWISE does not work right now: Reason: condition -> relational is problematic. It will require to overload all the ==, >=, ... operators in python");
			if (!GiNaC::is_a<GiNaC::relational>(cond))
			{
				throw_runtime_error("piecewise(condition, true_result, false_result) requires the condition to be a relational");
			}
			GiNaC::relational rel = GiNaC::ex_to<GiNaC::relational>(cond);
			GiNaC::ex diff = rel.lhs() - rel.rhs();
			GiNaC::ex_to<GiNaC::numeric>(diff);
			*/			
			/*
			GiNaC::numeric B=GiNaC::ex_to<GiNaC::numeric>(rel.op(1));

				   case info_flags::relation_equal:
					   return o==equal;
				   case info_flags::relation_not_equal:
					   return o==not_equal;
				   case info_flags::relation_less:
					   return o==less;
				   case info_flags::relation_less_or_equal:
					   return o==less_or_equal;
				   case info_flags::relation_greater:
					   return o==greater;
				   case info_flags::relation_greater_or_equal:
					   return o==greater_or_equal;
			*/
			if (GiNaC::is_a<GiNaC::numeric>(cond))
			{
				//std::cout << "NUMERIC  " << cond << "  " << GiNaC::ex_to<GiNaC::numeric>(cond).is_positive() << std::endl;
				if (!GiNaC::ex_to<GiNaC::numeric>(cond).is_positive())
					return b;
				else
					return a;
			}
			else if (GiNaC::is_a<GiNaC::constant>(cond))
			{
				if (!GiNaC::ex_to<GiNaC::numeric>(GiNaC::ex_to<GiNaC::constant>(cond).evalf()).is_positive())
					return b;
				else
					return a;
			}
			// TODO: SIMPLIFICATION HERE IF POSSIBLE
			return piecewise_geq0(cond, a, b).hold();
		}

		static void piecewise_geq0_csrc_float(const ex &cond, const ex &a, const ex &b, const print_context &c)
		{
			c.s << "(";
			cond.print(c);
			c.s << " >=0 ? ";
			a.print(c);
			c.s << " : ";
			b.print(c);
			c.s << ")";
		}

		static ex piecewise_geq0_expl_derivative(const ex &cond, const ex &a, const ex &b, const symbol &deriv_arg)
		{
			return piecewise_geq0(cond, a.diff(deriv_arg), b.diff(deriv_arg));
		}

		REGISTER_FUNCTION(piecewise_geq0, eval_func(piecewise_geq0_eval)
										 .print_func<print_csrc_float>(piecewise_geq0_csrc_float)
										 .print_func<print_csrc_double>(piecewise_geq0_csrc_float)
										 .expl_derivative_func(piecewise_geq0_expl_derivative)
										 .set_return_type(GiNaC::return_types::commutative))

		////////////////

		static ex internal_function_with_element_arg_eval(const ex &n, const ex &args)
		{
			return internal_function_with_element_arg(n, args).hold();
		}

		static void internal_function_with_element_arg_csrc_float(const ex &n, const ex &args, const print_context &c)
		{
			c.s << "my_func_table->";
			n.print(c);
			c.s << "(eleminfo->elem_ptr";
			//		c.s << "(my_func_table";
			lst l = ex_to<lst>(args);
			for (unsigned i = 0; i < l.nops(); i++)
			{
				c.s << ", ";
				l.op(i).print(c);
			}
			c.s << ")";
		}

		REGISTER_FUNCTION(internal_function_with_element_arg, eval_func(internal_function_with_element_arg_eval)
																  .print_func<print_csrc_float>(internal_function_with_element_arg_csrc_float)
																  .print_func<print_csrc_double>(internal_function_with_element_arg_csrc_float))

		/////////////////

		static ex python_cb_function_eval(const ex &func, const ex &arglst)
		{
			lst l = ex_to<lst>(arglst);
			lst l2;
			// Expand vectors and matrices within the args
			for (unsigned i = 0; i < l.nops(); i++)
			{
				GiNaC::ex a = l.op(i);
				if (GiNaC::is_a<GiNaC::matrix>(a))
				{
					GiNaC::matrix m = GiNaC::ex_to<GiNaC::matrix>(a);
					for (unsigned i = 0; i < m.rows(); i++)
						for (unsigned j = 0; j < m.cols(); j++)
							l2.append(m(i, j));
				}
				else
					l2.append(a);
			}

			std::vector<double> dv(l2.nops());
			bool success = true;
			for (unsigned i = 0; i < l2.nops(); i++)
			{
				if (GiNaC::is_a<GiNaC::numeric>(l2.op(i)) || GiNaC::is_a<GiNaC::constant>(l2.op(i)))
				{
					dv[i] = GiNaC::ex_to<GiNaC::numeric>(l2.op(i)).to_double();
				}
				else
				{
					success = false;
					break;
				}
			}
			if (success)
			{
				GiNaCCustomMathExpressionWrapper w = ex_to<GiNaCCustomMathExpressionWrapper>(func);
				const pyoomph::CustomMathExpressionWrapper &sp = w.get_struct();
				try
				{
					return sp.cme->_call(&(dv[0]), dv.size());
				}
				catch (const std::exception &exc)
				{
					std::cerr << exc.what();
				}
			}
			return python_cb_function(func, l2).hold();
		}

		static ex python_cb_function_evalf(const ex &func, const ex &arglst)
		{
			GiNaCCustomMathExpressionWrapper w = ex_to<GiNaCCustomMathExpressionWrapper>(func);
			const pyoomph::CustomMathExpressionWrapper &sp = w.get_struct();
			lst argf = ex_to<lst>(arglst.evalf());
			std::vector<double> dv(argf.nops());
			bool success = true;
			for (unsigned i = 0; i < argf.nops(); i++)
			{
				if (GiNaC::is_a<GiNaC::numeric>(argf.op(i)) || GiNaC::is_a<GiNaC::constant>(argf.op(i)))
				{
					dv[i] = GiNaC::ex_to<GiNaC::numeric>(argf.op(i)).to_double();
				}
				else
				{
					success = false;
					break;
				}
			}
			if (success)
			{
				try
				{
					return sp.cme->_call(&(dv[0]), dv.size());
				}
				catch (const std::exception &exc)
				{
					std::cerr << exc.what();
					//		return python_cb_function(func,argf);
				}
				return python_cb_function(func, arglst);
			}
			else
			{
				return python_cb_function(func, argf);
			}
		}

		static void python_cb_function_print_python(const ex &func, const ex &arglist, const print_context &c)
		{
			c.s << "python_callback(";
			GiNaCCustomMathExpressionWrapper w = ex_to<GiNaCCustomMathExpressionWrapper>(func);
			const pyoomph::CustomMathExpressionWrapper &sp = w.get_struct();
			c.s << sp.cme->get_id_name();

			lst l = ex_to<lst>(arglist);
			lst l2;
			// Expand vectors and matrices within the args
			for (unsigned i = 0; i < l.nops(); i++)
			{
				GiNaC::ex a = l.op(i);
				if (GiNaC::is_a<GiNaC::matrix>(a))
				{
					GiNaC::matrix m = GiNaC::ex_to<GiNaC::matrix>(a);
					for (unsigned i = 0; i < m.rows(); i++)
						for (unsigned j = 0; j < m.cols(); j++)
							l2.append(m(i, j));
				}
				else
					l2.append(a);
			}

			for (unsigned i = 0; i < l2.nops(); i++)
			{
				c.s << ", ";
				l2.op(i).print(c);
			}
			c.s << ")";
		}

		static void python_cb_function_csrc_float(const ex &func, const ex &arglst, const print_context &c)
		{
			c.s << "my_func_table->invoke_callback";
			//		 if (is_a<GiNaCCustomMathExpressionWrapper>(func)) std::cout << "SRCS " << std::endl;
			GiNaCCustomMathExpressionWrapper w = ex_to<GiNaCCustomMathExpressionWrapper>(func);
			//		 std::cout << w.get_struct().cme << std::endl;
			//		 std::cout << w.get_struct().cme->get_id_name() << std::endl;
			const pyoomph::CustomMathExpressionWrapper &sp = w.get_struct();
			int codeindex = -1;
			if (!CustomMathExpressionBase::code_map.count(sp.cme))
			{
				codeindex = CustomMathExpressionBase::code_map.size();
				CustomMathExpressionBase::code_map.insert(std::make_pair(sp.cme, codeindex));
			}
			else
			{
				codeindex = CustomMathExpressionBase::code_map[sp.cme];
			}
			//     c.s << "(eleminfo->elem_ptr, " << std::to_string(codeindex) << " , (double []){";
			c.s << "(my_func_table, " << std::to_string(codeindex) << " , (double []){";
			lst l = ex_to<lst>(arglst);
			lst l2;
			// Expand vectors and matrices within the args
			for (unsigned i = 0; i < l.nops(); i++)
			{
				GiNaC::ex a = l.op(i);
				if (GiNaC::is_a<GiNaC::matrix>(a))
				{
					GiNaC::matrix m = GiNaC::ex_to<GiNaC::matrix>(a);
					for (unsigned i = 0; i < m.rows(); i++)
						for (unsigned j = 0; j < m.cols(); j++)
							l2.append(m(i, j));
				}
				else
					l2.append(a);
			}

			for (unsigned i = 0; i < l2.nops(); i++)
			{
				if (i > 0)
					c.s << ", ";
				l2.op(i).print(c);
			}
			c.s << "}, " << l2.nops() << " )";
		}

		static ex python_cb_function_expl_deriv(const ex &func, const ex &arglst, const symbol &deriv_arg)
		{
			GiNaCCustomMathExpressionWrapper w = ex_to<GiNaCCustomMathExpressionWrapper>(func);
			const pyoomph::CustomMathExpressionWrapper &sp = w.get_struct();
			lst l = ex_to<lst>(arglst);
			ex res = 0;
			for (unsigned i = 0; i < l.nops(); i++)
			{
				ex diff_i = l.op(i).diff(deriv_arg);
				if (!diff_i.is_zero())
				{
					GiNaC::ex diff_o;
					try
					{
						diff_o = sp.cme->outer_derivative(arglst, i);
					}
					catch (const std::exception &exc)
					{
						std::cerr << exc.what();
					}
					//					ex_to<GiNaCCustomMathExpressionWrapper>(diff_o).get_struct().cme->set_as_derivative(sp.cme,i);
					res = res + diff_i * diff_o;
				}
			}
			return res;
		}

		static ex python_cb_function_real_part(const ex &func, const ex &arglst)
		{
			GiNaCCustomMathExpressionWrapper w = ex_to<GiNaCCustomMathExpressionWrapper>(func);
			const pyoomph::CustomMathExpressionWrapper &sp = w.get_struct();
			lst l = ex_to<lst>(arglst);
			std::vector<GiNaC::ex> arglist(l.nops());
			for (unsigned i = 0; i < l.nops(); i++)
				arglist[i] = l.op(i);
			return sp.cme->real_part(func, arglist);
		}

		static ex python_cb_function_imag_part(const ex &func, const ex &arglst)
		{
			GiNaCCustomMathExpressionWrapper w = ex_to<GiNaCCustomMathExpressionWrapper>(func);
			const pyoomph::CustomMathExpressionWrapper &sp = w.get_struct();
			lst l = ex_to<lst>(arglst);
			std::vector<GiNaC::ex> arglist(l.nops());
			for (unsigned i = 0; i < l.nops(); i++)
				arglist[i] = l.op(i);
			return sp.cme->imag_part(func, arglist);
		}

		REGISTER_FUNCTION(python_cb_function, eval_func(python_cb_function_eval).evalf_func(python_cb_function_evalf).print_func<print_csrc_float>(python_cb_function_csrc_float).print_func<print_csrc_double>(python_cb_function_csrc_float).expl_derivative_func(python_cb_function_expl_deriv).print_func<print_python>(python_cb_function_print_python).real_part_func(python_cb_function_real_part).imag_part_func(python_cb_function_imag_part))

		static ex python_multi_cb_function_eval(const ex &func, const ex &arglst, const ex &numret)
		{
			std::vector<double> dv(arglst.nops());
			bool success = true;
			int numret_i = GiNaC::ex_to<GiNaC::numeric>(numret).to_int();
			for (unsigned i = 0; i < arglst.nops(); i++)
			{
				if (GiNaC::is_a<GiNaC::numeric>(arglst.op(i)))
				{
					dv[i] = GiNaC::ex_to<GiNaC::numeric>(arglst.op(i)).to_double();
				}
				else if (GiNaC::is_a<GiNaC::constant>(arglst.op(i)))
				{
					dv[i] = GiNaC::ex_to<GiNaC::numeric>(GiNaC::ex_to<GiNaC::constant>(arglst.op(i)).evalf()).to_double();
				}
				else
				{
					success = false;
					break;
				}
			}
			if (success)
			{
				GiNaCCustomMultiReturnExpressionWrapper w = ex_to<GiNaCCustomMultiReturnExpressionWrapper>(func);
				const pyoomph::CustomMultiReturnExpressionWrapper &sp = w.get_struct();
				try
				{
					std::vector<double> ret(numret_i, 0.0);
					std::vector<double> dummy_derivs(1);
					sp.cme->_call(0, &(dv[0]), dv.size(), &(ret[0]), numret_i, &(dummy_derivs[0]));
					std::vector<GiNaC::ex> res(numret_i);
					for (int i = 0; i < numret_i; i++)
					{
						res[i] = GiNaC::ex(ret[i]);
					}
					return GiNaC::lst(res.begin(), res.end());
				}
				catch (const std::exception &exc)
				{
					std::cerr << exc.what();
				}
			}
			return python_multi_cb_function(func, arglst, numret).hold();
		}

		REGISTER_FUNCTION(python_multi_cb_function, eval_func(python_multi_cb_function_eval) //.evalf_func(python_cb_function_evalf).print_func<print_csrc_float>(python_cb_function_csrc_float).print_func<print_csrc_double>(python_cb_function_csrc_float).expl_derivative_func(python_cb_function_expl_deriv)
																							 //            .print_func<print_python>(python_cb_function_print_python)
		)

		static ex python_multi_cb_indexed_result_eval(const ex &func, const ex &index)
		{
			if (GiNaC::is_a<GiNaC::lst>(func))
			{
				int index_i = GiNaC::ex_to<GiNaC::numeric>(index).to_int();
				return GiNaC::ex_to<GiNaC::lst>(func)[index_i];
			}
			return python_multi_cb_indexed_result(func, index).hold();
		}

		REGISTER_FUNCTION(python_multi_cb_indexed_result, eval_func(python_multi_cb_indexed_result_eval) //.evalf_func(python_cb_function_evalf).print_func<print_csrc_float>(python_cb_function_csrc_float).print_func<print_csrc_double>(python_cb_function_csrc_float).expl_derivative_func(python_cb_function_expl_deriv)
																										 //            .print_func<print_python>(python_cb_function_print_python)
		)

		static ex ginac_expand_eval(const ex &v)
		{
			if (need_to_hold(v))
				return ginac_expand(v).hold();
			return GiNaC::expand(v);
		}

		REGISTER_FUNCTION(ginac_expand, eval_func(ginac_expand_eval))

		static ex ginac_normal_eval(const ex &v)
		{
			if (need_to_hold(v))
				return ginac_normal(v).hold();
			return GiNaC::normal(v);
		}

		REGISTER_FUNCTION(ginac_normal, eval_func(ginac_normal_eval))

		static ex ginac_factor_eval(const ex &v)
		{
			if (need_to_hold(v))
				return ginac_factor(v).hold();
			return GiNaC::factor(v);
		}

		REGISTER_FUNCTION(ginac_factor, eval_func(ginac_factor_eval))

		static ex ginac_collect_eval(const ex &v, const ex &s)
		{
			if (need_to_hold(v) || need_to_hold(s))
				return ginac_collect(v, s).hold();
			return GiNaC::collect(v, s);
		}

		REGISTER_FUNCTION(ginac_collect, eval_func(ginac_collect_eval))

		static ex ginac_collect_common_factors_eval(const ex &v)
		{
			if (need_to_hold(v))
				return ginac_collect_common_factors(v).hold();
			return GiNaC::collect_common_factors(v);
		}

		REGISTER_FUNCTION(ginac_collect_common_factors, eval_func(ginac_collect_common_factors_eval))

		static ex ginac_series_eval(const ex &expr, const ex &x, const ex &x0, const ex &order)
		{
			if (need_to_hold(expr) || need_to_hold(x) || need_to_hold(x0) || need_to_hold(order))
				return ginac_series(expr, x, x0, order).hold();

			if (!GiNaC::is_a<GiNaC::numeric>(order))
			{
				throw_runtime_error("Series order must be an integer");
			}

			return series_to_poly(expr.series(x == x0, GiNaC::ex_to<GiNaC::numeric>(order).to_int()));
		}

		REGISTER_FUNCTION(ginac_series, eval_func(ginac_series_eval))

		////////////////

		static ex weak_eval(const ex &a, const ex &b, const ex &flags, const ex &coordsys)
		{
			if (pyoomph::pyoomph_verbose)
				std::cout << "Trying to eval weak of " << std::endl
						  << a << std::endl
						  << b << std::endl
						  << "with flags " << flags << "and coordsys " << coordsys << std::endl;
			ex evma = a.evalm();
			ex evmb = b.evalm();
			if (evma.is_zero() || evmb.is_zero())
				return 0;
			if (is_a<matrix>(evma) && ex_to<matrix>(evma).is_zero_matrix())
				return 0;
			if (is_a<matrix>(evmb) && ex_to<matrix>(evmb).is_zero_matrix())
				return 0;

			if (need_to_hold(evma) || need_to_hold(evmb))
				return weak(evma, evmb, flags, coordsys).hold();

			CustomCoordinateSystem *sys = NULL;

			if (!coordsys.is_zero())
			{
				GiNaCCustomCoordinateSystemWrapper w = ex_to<GiNaCCustomCoordinateSystemWrapper>(coordsys);
				const pyoomph::CustomCoordinateSystemWrapper &sp = w.get_struct();

				if (sp.cme == &__no_coordinate_system)
				{
					if (__current_code)
					{
						sys = __current_code->get_coordinate_system();
						std::cout << "COORDINATE SYSTEM " << sys << std::endl;
						if (pyoomph::pyoomph_verbose)
						{
							std::cout << "Got the coordinate system from element " << sys << std::endl;
						}
					}
					if (sys == &__no_coordinate_system)
					{
						std::cerr << "CANNOT RESOLVE COORD SYS" << std::endl;
						return weak(a, b, flags, coordsys).hold();
					}
				}
				else
				{
					sys = sp.cme;
				}
			}
			int flag = GiNaC::ex_to<GiNaC::numeric>(flags).to_int();

			bool use_scaling = (flag & 2);
			bool lagrangian = (flag & 1);

			if (!__current_code)
				throw_runtime_error("Cannot use weak outside of a code generation");
			//  std::cout << "GETTING DX " << __current_code << "  " << use_scaling << "  " << lagrangian << "  " << sys << std::endl;
			GiNaC::ex dx = __current_code->get_integral_dx(use_scaling, lagrangian, sys);

			if (is_a<matrix>(evma) && is_a<matrix>(evmb))
			{
				matrix ma = ex_to<matrix>(evma);
				matrix mb = ex_to<matrix>(evmb);
				if (ma.cols() == 1 && mb.cols() == 1)
					return dot(evma, evmb) * dx; // Vector
				else if (ma.cols() == mb.cols())
					return double_dot(evma, evmb) * dx; // Matrix
				else
				{
					std::ostringstream oss;
					oss << std::endl
						<< " a = " << evma << std::endl
						<< " b = " << evmb << std::endl;
					throw_runtime_error("Arity not matching in the following weak form contribution:" + oss.str());
				}
			}
			if (!is_a<matrix>(evma) && !is_a<matrix>(evmb))
			{
				return evma * evmb * dx; // Scalar
			}
			else
			{
				std::ostringstream oss;
				oss << std::endl
					<< " a = " << evma << std::endl
					<< " b = " << evmb << std::endl;
				throw_runtime_error("Arity not matching in the following weak form contribution:" + oss.str());
				return 0;
			}
		}

		REGISTER_FUNCTION(weak, eval_func(weak_eval).set_return_type(GiNaC::return_types::commutative))

		/////////////////

		class ReplaceFieldsAndSubfields : public GiNaC::map_function
		{
		protected:
			const std::map<std::string, GiNaC::ex> &fields, nondimfields, globalparams;

		public:
			ReplaceFieldsAndSubfields(const std::map<std::string, GiNaC::ex> &_fields, const std::map<std::string, GiNaC::ex> &_nondimfields, const std::map<std::string, GiNaC::ex> &_globalparams) : fields(_fields), nondimfields(_nondimfields), globalparams(_globalparams) {}
			GiNaC::ex operator()(const GiNaC::ex &inp)
			{
				if (is_ex_the_function(inp, field))
				{
					std::ostringstream os;
					os << inp.op(0);
					std::string fname = os.str();
					if (fields.count(fname))
						return fields.at(fname);
					else
						return inp.map(*this);
				}
				else if (is_ex_the_function(inp, nondimfield))
				{
					std::ostringstream os;
					os << inp.op(0);
					std::string fname = os.str();
					if (nondimfields.count(fname))
						return nondimfields.at(fname);
					else
						return inp.map(*this);
				}
				else if (is_a<GiNaCGlobalParameterWrapper>(inp))
				{
					const GiNaCGlobalParameterWrapper &gp = ex_to<GiNaCGlobalParameterWrapper>(inp);
					std::string fname = gp.get_struct().cme->get_name();
					if (globalparams.count(fname))
						return globalparams.at(fname);
					else
						return inp.map(*this);
				}
				else
					return inp.map(*this);
			}
		};

		GiNaC::ex subs_fields(const GiNaC::ex &arg, const std::map<std::string, GiNaC::ex> &fields, const std::map<std::string, GiNaC::ex> &nondimfields, const std::map<std::string, GiNaC::ex> &globalparams)
		{
			ReplaceFieldsAndSubfields repl(fields, nondimfields, globalparams);
			DrawUnitsOutOfSubexpressions uos(NULL);
			return uos(repl(arg));
		}

		class GlobalParamsToDouble : public GiNaC::map_function
		{
		protected:
		public:
			GiNaC::ex operator()(const GiNaC::ex &inp)
			{
				if (is_a<GiNaCGlobalParameterWrapper>(inp))
				{
					const GiNaCGlobalParameterWrapper &gp = ex_to<GiNaCGlobalParameterWrapper>(inp);
					return gp.get_struct().cme->value();
				}
				else
					return inp.map(*this);
			}
		};

		double eval_to_double(const GiNaC::ex &inp)
		{
			GlobalParamsToDouble expand_params;
			DrawUnitsOutOfSubexpressions uos(NULL);
			GiNaC::ex gpsubst = uos(expand_params(inp));
			GiNaC::ex v = GiNaC::evalf(gpsubst);
			if (GiNaC::is_a<GiNaC::numeric>(v))
			{
				return GiNaC::ex_to<GiNaC::numeric>(v).to_double();
			}
			else
			{
				// There might be units which have not been cancelled out
				GiNaC::ex factor, unit, rest;			  
			    if (expressions::collect_base_units(inp, factor, unit, rest))
				{
					GiNaC::ex inp2=factor * unit * expressions::subexpression(rest);
					gpsubst = uos(expand_params(inp2));
					v = GiNaC::evalf(gpsubst);
					if (GiNaC::is_a<GiNaC::numeric>(v))
					{
						return GiNaC::ex_to<GiNaC::numeric>(v).to_double();
					}
				}
				std::ostringstream oss;
				oss << "Cannot cast the following into a double: " << v;
				throw_runtime_error(oss.str());
			}
		}

		std::complex<double> eval_to_complex(const GiNaC::ex &inp)
		{
			GlobalParamsToDouble expand_params;
			DrawUnitsOutOfSubexpressions uos(NULL);
			GiNaC::ex gpsubst = uos(expand_params(inp));
			GiNaC::ex v = GiNaC::evalf(gpsubst);
			if (GiNaC::is_a<GiNaC::numeric>(v))
			{
				GiNaC::numeric n = GiNaC::ex_to<GiNaC::numeric>(v);
				return std::complex<double>(n.real().to_double(), n.imag().to_double());
			}
			else
			{
				std::ostringstream oss;
				oss << "Cannot cast the following into a complex: " << v;
				throw_runtime_error(oss.str());
			}
		}

		/////////////////

		static ex minimize_functional_derivative_eval(const ex &F, const ex &only_wrto, const ex &flags, const ex &coordsys)
		{
			if (need_to_hold(F) || need_to_hold(only_wrto) || need_to_hold(flags) || need_to_hold(coordsys))
				return minimize_functional_derivative(F, only_wrto, flags, coordsys).hold();
			

			int flag = GiNaC::ex_to<GiNaC::numeric>(flags).to_int();
			bool dim_testfunc=flag & 64;


			if (!__current_code)
				throw_runtime_error("Cannot use functional minimization outside of a code generation");

			GiNaC::lst wrto;
			if (GiNaC::is_a<GiNaC::lst>(only_wrto) && only_wrto.nops() > 0)
			{
				std::set<pyoomph::ShapeExpansion> wrto_set;
				std::cout << "NOPS " << only_wrto.nops() << std::endl;
				for (unsigned i = 0; i < only_wrto.nops(); i++)
				{
					//std::cout << "NOP " <<i << " of NOPS " << only_wrto.nops()<< " : " << only_wrto.op(i) << std::endl;
					//std::cout << "GETTING SHAPES IN " << only_wrto.op(i) << std::endl;
					for (const pyoomph::ShapeExpansion & se:  __current_code->get_all_shape_expansions_in(only_wrto.op(i)))
					{
						
						GiNaC::GiNaCShapeExpansion gse(se);
						//std::cout << "  SHAPE  " << gse << " IN " << only_wrto.op(i) << " SET COUNT " << wrto_set.count(se) << std::endl;
						if (!wrto_set.count(se))
						{
							//std::cout << "  ADDING SHAPE  " << gse << " IN " << only_wrto.op(i) << std::endl;
							//std::cout << "SHAPE " << gse << std::endl;
							wrto.append(gse);
							wrto_set.insert(se);
						}
					}					
				}
				
			}
			else
			{
				for (const pyoomph::ShapeExpansion & se:  __current_code->get_all_shape_expansions_in(F))
				{
					wrto.append(GiNaC::GiNaCShapeExpansion(se));
				}
			}
			//std::cout << "Minimizing " << F << " wrt " << wrto << std::endl;
			GiNaC::ex res = 0;
			GiNaC::symbol derive_dummy("__dummy__for_minimization");
			for (GiNaC::ex wrt_ex : wrto)
			{
				if (is_a<GiNaCShapeExpansion>(wrt_ex))
				{
					const GiNaCShapeExpansion &se = ex_to<GiNaCShapeExpansion>(wrt_ex);
					const pyoomph::ShapeExpansion sexp=se.get_struct();
					if (sexp.time_history_index!=0) throw_runtime_error("Cannot minimize wrt a time history");
					if (sexp.expansion_mode!=0) throw_runtime_error("Cannot minimize wrt an expansion mode");
					if (sexp.nodal_coord_dir!=-1 || sexp.nodal_coord_dir2!=-1) throw_runtime_error("Cannot minimize wrt a nodal coordinate derivative");
					if (sexp.is_derived) throw_runtime_error("Cannot minimize wrt a derived shape expansion");
					if (sexp.is_derived_other_index) throw_runtime_error("Cannot minimize wrt a derived shape expansion with respect to another index");
					if (sexp.no_jacobian || sexp.no_hessian) throw_runtime_error("Cannot minimize wrt a shape expansion with no_jacobian or no_hessian");
					if (sexp.dt_order!=0) throw_runtime_error("Cannot minimize wrt a shape expansion with a time derivative");
					// TODO: This should be possible for e.g. bulk domain -> bulk test function
					//if (sexp.field->get_space()->get_code()!=__current_code) throw_runtime_error("Cannot minimize wrt a shape expansion from another domain");

					//std::cout << "Minimizing wrt " << se << std::endl;					
					GiNaC::ex F_dummy = F.subs(wrt_ex == derive_dummy);
					GiNaC::ex dF = GiNaC::diff(F_dummy, derive_dummy).subs(derive_dummy == wrt_ex);
//					std::cout << " dF " << dF << std::endl;					
					pyoomph::TestFunction testfunc(sexp.field,sexp.basis);
					GiNaC::ex tf=0+GiNaC::GiNaCTestFunction(testfunc);
					if (dim_testfunc)
					{
						//throw_runtime_error("Dimensional test function here");
						std::string id=sexp.field->get_name();
						if (!pyoomph::__field_name_cache.count(id)) pyoomph::__field_name_cache.insert(std::make_pair(id,GiNaC::realsymbol(id)));						
	  	  				GiNaC::GiNaCPlaceHolderResolveInfo ri(pyoomph::PlaceHolderResolveInfo(NULL,std::vector<std::string>{"domain:"+sexp.basis->get_space()->get_code()->get_full_domain_name()}));						
						tf*=(0+pyoomph::expressions::test_scale(pyoomph::__field_name_cache[id],ri));
					}
					//std::cout << "dF " << dF << std::endl;
					res = res + weak(dF,tf,flags,coordsys);
				}
				else
				{
					std::ostringstream oss;
					oss << "Cannot minimize wrt " << wrt_ex;
					throw_runtime_error(oss.str());
				}
			}



			return res;
		}

		
		REGISTER_FUNCTION(minimize_functional_derivative, eval_func(minimize_functional_derivative_eval).set_return_type(GiNaC::return_types::commutative))

	}

}
