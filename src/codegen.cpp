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
#include <chrono>
#include <unordered_set>
#include <cmath>
#include <functional>

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

	// An exponent that is a floating point number happening to be a whole number, e.g. -1.0 rather
	// than -1. GiNaC's C-source printer for products loses such a factor entirely: mul::do_print_csrc
	// decides whether to emit "1.0/" or "/" from info_flags::negint, which an inexact -1 does not
	// satisfy, but decides whether to leave the exponent out from is_equal(-1), which - GiNaC's
	// numeric comparison being by value, not by representation - it does. So x*y^(-1.0) is printed as
	// "x*y": a silently wrong reciprocal, not an error. Nothing else distinguishes an exact from an
	// inexact whole-number exponent, so they are made exact before printing.
	static bool is_inexact_whole_number(const GiNaC::ex &e)
	{
		if (!GiNaC::is_a<GiNaC::numeric>(e))
			return false;
		const GiNaC::numeric &n = GiNaC::ex_to<GiNaC::numeric>(e);
		if (n.is_crational() || !n.is_real())
			return false;
		const double d = n.to_double();
		return d == std::floor(d) && std::fabs(d) < 1e15;
	}

	class ExactifyWholeNumberExponents : public GiNaC::map_function
	{
	public:
		GiNaC::ex operator()(const GiNaC::ex &inp) override
		{
			if (GiNaC::is_a<GiNaC::power>(inp) && is_inexact_whole_number(inp.op(1)))
				return GiNaC::power(inp.op(0).map(*this), GiNaC::numeric((long)GiNaC::ex_to<GiNaC::numeric>(inp.op(1)).to_double()));
			return inp.map(*this);
		}
	};

	// The rewrite above rebuilds the whole expression, so it is only run when this cheaper scan -
	// which allocates nothing - finds something to rewrite. It normally does not.
	static bool has_inexact_whole_number_exponent(const GiNaC::ex &e)
	{
		for (GiNaC::const_preorder_iterator it = e.preorder_begin(); it != e.preorder_end(); ++it)
			if (GiNaC::is_a<GiNaC::power>(*it) && is_inexact_whole_number(it->op(1)))
				return true;
		return false;
	}

	static const bool __reciprocals_disabled = getenv("PYOOMPH_DISABLE_PARAM_RECIPROCALS") != NULL;

	// A global parameter appearing with a negative exponent is printed by GiNaC's mul printer as an
	// infix division - `.../pyoomph_gparam_0` - and GCC will not turn that into a reciprocal multiply
	// without -freciprocal-math. Across the tutorial corpus there are 137 such divisions, and they sit
	// inside Jacobian and Hessian ENTRIES, i.e. in the innermost trial loop, so each is a hardware
	// divide per matrix entry. Substituting the reciprocal of the parameter (declared once per function
	// by GlobalParameterFunctionScope, right next to the parameter itself) turns every one of them into
	// a multiply: measured at -14.8% on the azimuthal element, composing with the buffer aliases of 13
	// to -25.8%.
	//
	// This is the ONE place in code generation that deliberately changes the arithmetic: a*(1/b) is not
	// bit-identical to a/b. It is a reassociation, nothing more - the same class of difference gcc and
	// tcc already show against each other through FMA contraction (12).
	//
	// Only negative INTEGER exponents, and only global parameters, whose value is fixed for the whole
	// assembly. Do not widen this to arbitrary printed subtrees: that is the general CSE pass of 8.3,
	// which was built, measured at -1.9% and reverted.
	class ParameterReciprocals : public GiNaC::map_function
	{
		FiniteElementCode *code;

	public:
		ParameterReciprocals(FiniteElementCode *code_) : code(code_) {}
		GiNaC::ex operator()(const GiNaC::ex &inp) override
		{
			if (GiNaC::is_a<GiNaC::power>(inp) && GiNaC::is_a<GiNaC::GiNaCGlobalParameterWrapper>(inp.op(0)) &&
				GiNaC::is_a<GiNaC::numeric>(inp.op(1)) && inp.op(1).info(GiNaC::info_flags::negint))
			{
				const unsigned global_index = GiNaC::ex_to<GiNaC::GiNaCGlobalParameterWrapper>(inp.op(0)).get_struct().cme->get_global_index();
				auto it = code->global_parameter_to_local_indices.find(global_index);
				// Only where the parameter really is a local of this function: outside a
				// GlobalParameterFunctionScope it prints as the indirect table access, which is not what
				// _rgp_<i> was declared from.
				if (it != code->global_parameter_to_local_indices.end() && code->params_declared_in_current_function.count(it->second))
				{
					const unsigned local_index = it->second;
					const GiNaC::ex recip = GiNaC::potential_real_symbol("_rgp_" + std::to_string(local_index));
					const long n = -(long)GiNaC::ex_to<GiNaC::numeric>(inp.op(1)).to_long();
					return (n == 1 ? recip : GiNaC::power(recip, GiNaC::numeric(n)));
				}
			}
			return inp.map(*this);
		}
	};

	static bool has_parameter_reciprocal(const GiNaC::ex &e)
	{
		for (GiNaC::const_preorder_iterator it = e.preorder_begin(); it != e.preorder_end(); ++it)
			if (GiNaC::is_a<GiNaC::power>(*it) && GiNaC::is_a<GiNaC::GiNaCGlobalParameterWrapper>(it->op(0)) &&
				GiNaC::is_a<GiNaC::numeric>(it->op(1)) && it->op(1).info(GiNaC::info_flags::negint))
				return true;
		return false;
	}

	// Prints a GiNaC expression as C source code, after applying an optional simplification
	// strategy selected at runtime via FiniteElementCode::ccode_expression_mode (e.g. "factor",
	// "normal", "expand", "collect_common_factors" - mostly useful for debugging/benchmarking
	// how different GiNaC simplifications affect the generated code). The expression is also
	// archived (csrc_opts.for_code->archive) so that it can be inspected/replayed later.
	void print_simplest_form(GiNaC::ex expr, std::ostream &os, GiNaC::print_FEM_options &csrc_opts)
	{
		GiNaC::ex towrite;
		std::string mode = csrc_opts.for_code->ccode_expression_mode;
		// Opt-in; nothing reads the archive - see FiniteElementCode::archive.
		static const bool do_archive = getenv("PYOOMPH_ARCHIVE_EXPRESSIONS") != NULL;
		if (do_archive)
			csrc_opts.for_code->archive.archive_ex(expr, ("expression_" + std::to_string(csrc_opts.for_code->archive.num_expressions())).c_str());
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
		if (has_inexact_whole_number_exponent(towrite))
		{
			ExactifyWholeNumberExponents exactify;
			towrite = exactify(towrite);
		}
		// After exactify, so that an inexact -1.0 exponent is an exact -1 by the time negint is tested.
		if (!__reciprocals_disabled && csrc_opts.for_code && has_parameter_reciprocal(towrite))
		{
			ParameterReciprocals reciprocals(csrc_opts.for_code);
			towrite = reciprocals(towrite);
		}
		towrite.print(GiNaC::print_csrc_FEM(os, &csrc_opts));
	}

	// Emits one residual entry, or "0" when this residual's assembly has been switched off with
	// set_ignore_residual_assembly() - which the azimuthal and Cartesian normal-mode stability
	// contributions and the pitchfork mass matrix all do (see Problem's stability setup): they exist
	// to supply Jacobian and mass-matrix entries for the eigenproblem, never a residual.
	//
	// Historically the ignored case still printed the expression into a discarded stream, because
	// GiNaCGlobalParameterWrapper::print allocated a code's local parameter slot on first encounter
	// and a parameter occurring only in a field-independent term of an ignored residual would
	// otherwise never have been registered. Registration now happens print-free and up front
	// (GlobalParameterFunctionScope below screens the full residual before any of the function body
	// is printed), so the ignored case emits a bare 0 without any traversal. Even earlier, the
	// printed form was kept as a /* ... */ comment - in one production element 66.5% of a 5.8 MB
	// generated file, recompiled on every JIT cache miss for no purpose.
	static void print_residual_entry(FiniteElementCode *for_code, const GiNaC::ex &res_part,
									 std::ostream &os, GiNaC::print_FEM_options &csrc_opts)
	{
		if (for_code->is_current_residual_assembly_ignored())
		{
			os << "0 /* IGNORED RESIDUAL */";
		}
		else
		{
			print_simplest_form(res_part, os, csrc_opts);
		}
	}

	// Registers every global parameter occurring in e in this code's local slot table (same
	// first-encounter numbering GiNaCGlobalParameterWrapper::print used to establish as a print side
	// effect) and collects the local indices into used_local_indices. Descends into CSE subexpression
	// wrappers and multi-ret invocation arguments, which a plain preorder walk would treat as opaque.
	void FiniteElementCode::register_global_parameters_in(const GiNaC::ex &e, std::set<unsigned> &used_local_indices)
	{
		for (GiNaC::const_preorder_iterator i = e.preorder_begin(); i != e.preorder_end(); ++i)
		{
			if (GiNaC::is_a<GiNaC::GiNaCGlobalParameterWrapper>(*i))
			{
				const auto &p = GiNaC::ex_to<GiNaC::GiNaCGlobalParameterWrapper>(*i).get_struct();
				unsigned global_index = p.cme->get_global_index();
				if (!global_parameter_to_local_indices.count(global_index))
				{
					unsigned local_index = global_parameter_to_local_indices.size();
					local_parameter_symbols.push_back(*i);
					global_parameter_to_local_indices.insert(std::make_pair(global_index, local_index));
				}
				used_local_indices.insert(global_parameter_to_local_indices[global_index]);
			}
			else if (GiNaC::is_a<GiNaC::GiNaCSubExpression>(*i))
			{
				register_global_parameters_in(GiNaC::ex_to<GiNaC::GiNaCSubExpression>(*i).get_struct().expr, used_local_indices);
			}
			else if (GiNaC::is_a<GiNaC::GiNaCMultiRetCallback>(*i))
			{
				register_global_parameters_in(GiNaC::ex_to<GiNaC::GiNaCMultiRetCallback>(*i).get_struct().invok.op(1), used_local_indices);
			}
		}
	}

	// Screens the symbolic source expression(s) of ONE generated C function for global parameters
	// before anything of the function body is printed. This replaces the former lazy registration
	// inside GiNaCGlobalParameterWrapper::print, which made slot numbering depend on print order and
	// forced print_residual_entry above to print ignored residuals into a discarded stream just for
	// the side effect. The scope also hoists each parameter into a
	//   const double pyoomph_gparam_<i> = *(my_func_table->global_parameters[<i>]);
	// local (write_declarations, to be emitted right after the function's opening lines), which
	// GiNaCGlobalParameterWrapper::print then references by name instead of repeating the double
	// indirection at every use. Printing paths not covered by such a scope (multi-ret callback
	// bodies, the geometric-Jacobian family) keep working through print's fallback: lazy
	// registration plus the self-contained indirect access.
	//
	// Everything printed inside a scoped function must derive from the screened sources (residual
	// derivatives, CSE bodies, ...), so the declared set is always a superset of what gets printed;
	// a parameter screened but eliminated from every printed derivative merely leaves an unused
	// local behind.
	class GlobalParameterFunctionScope
	{
		FiniteElementCode *code;
		std::set<unsigned> used;

	public:
		GlobalParameterFunctionScope(FiniteElementCode *code_, const std::vector<GiNaC::ex> &sources) : code(code_)
		{
			for (auto &e : sources)
				code->register_global_parameters_in(e, used);
			code->params_declared_in_current_function = used;
		}
		void write_declarations(std::ostream &os, const std::string &indent)
		{
			for (unsigned i : used)
			{
				const auto &p = GiNaC::ex_to<GiNaC::GiNaCGlobalParameterWrapper>(code->local_parameter_symbols[i]).get_struct();
				os << indent << "const double pyoomph_gparam_" << i << " = *(my_func_table->global_parameters[" << i << "]); // " << p.cme->get_name() << std::endl;
				// Emitted for every declared parameter rather than only for the ones that turn out to be
				// divided by: which those are is only known once the derivatives have been printed, and
				// these declarations have to come first. An unused one is a dead local that the compiler
				// drops; a division by a zero-valued parameter that is never read yields an unused inf,
				// and pyoomph does not unmask FP exceptions (see the note on BEGIN_RESIDUAL_* skipping
				// contribution evaluation in jitbridge.h).
				if (!__reciprocals_disabled)
					os << indent << "const double _rgp_" << i << " = 1.0/pyoomph_gparam_" << i << ";" << std::endl;
			}
		}
		~GlobalParameterFunctionScope() { code->params_declared_in_current_function.clear(); }
	};

	//////////////

	// Buffer aliases, see FiniteElementCode::alias_buffer_access and the comment on the members in
	// codegen.hpp. The switch is the A/B lever: it leaves the emitted code exactly as it was before
	// the aliases existed, on the same binary, which is the only way to re-measure them later.
	static const bool __buffer_aliases_disabled = getenv("PYOOMPH_DISABLE_BUFFER_ALIASES") != NULL;

	// One per emitted function. Aliases are function-scoped (legal because the shape buffers are
	// re-pointed by prepare_shape_buffer_for_integration, i.e. once per assembly and before the
	// generated function is entered), so the registry is cleared per function and the declarations
	// go at the top of it, after the nodal-index constants the hanginfo aliases reference.
	class BufferAliasFunctionScope
	{
		FiniteElementCode *code;
		bool was_active;

	public:
		BufferAliasFunctionScope(FiniteElementCode *code_) : code(code_), was_active(code_->buffer_aliases_active)
		{
			code->buffer_alias_by_access.clear();
			code->buffer_alias_decls.clear();
			code->buffer_aliases_active = true;
		}
		std::string declarations(const std::string &indent, const std::string &body) const { return code->write_buffer_alias_declarations(indent, body); }
		~BufferAliasFunctionScope()
		{
			code->buffer_aliases_active = was_active;
			code->buffer_alias_by_access.clear();
			code->buffer_alias_decls.clear();
		}
	};

	// Turns aliasing off for a stretch of printing that lands OUTSIDE the region the declarations will
	// be emitted into. The Hessian needs it: its assembly body is printed before its header, so the
	// scope is open while the pre-loop time-interpolation is printed, and an alias declared inside the
	// integration loop must not be referenced above it.
	class BufferAliasSuspend
	{
		FiniteElementCode *code;
		bool was_active;

	public:
		BufferAliasSuspend(FiniteElementCode *code_) : code(code_), was_active(code_->buffer_aliases_active) { code->buffer_aliases_active = false; }
		~BufferAliasSuspend() { code->buffer_aliases_active = was_active; }
	};

	std::string FiniteElementCode::alias_buffer_access(const std::string &access, const std::string &name, const std::string &decl)
	{
		if (!buffer_aliases_active || __buffer_aliases_disabled)
			return access;
		auto it = buffer_alias_by_access.find(access);
		if (it != buffer_alias_by_access.end())
			return it->second;
		// A name must stand for exactly one access. If it ever did not, the generated code would read
		// the wrong buffer and still compile - so this is checked rather than assumed.
		auto clash = buffer_alias_decls.find(name);
		if (clash != buffer_alias_decls.end())
			throw_runtime_error("Buffer alias '" + name + "' would stand for '" + access + "' as well as for '" + clash->second + "'");
		buffer_alias_by_access[access] = name;
		buffer_alias_decls[name] = decl + " = " + access + ";";
		return name;
	}

	std::string FiniteElementCode::alias_nodal_data(const FiniteElementSpace *sp)
	{
		const std::string nds = get_nodal_data_string(sp);
		const std::string access = get_elem_info_str(sp) + "->" + nds;
		const std::string name = "_" + get_owner_prefix(sp) + (nds == "nodal_coords" ? "ncoord" : "ndata");
		return alias_buffer_access(access, name, "double *** const " + name);
	}

	// Only the aliases the body actually mentions. Registration happens while printing, and printing is
	// not the same as emitting: write_generic_RJM_contribution buffers a whole space block and throws it
	// away when the space turns out to contribute nothing, and the Hessian harvests over spaces it then
	// skips. A left-over shape alias would merely be an unused local, but a left-over hanginfo alias is
	// indexed by a this_nodalind_* constant that is only declared for the fields that survived - which
	// is a compile error in the generated C, and was one on the azimuthal element's external-ODE field.
	std::string FiniteElementCode::write_buffer_alias_declarations(const std::string &indent, const std::string &body) const
	{
		std::ostringstream oss;
		for (const auto &e : buffer_alias_decls)
			if (body.find(e.first) != std::string::npos)
				oss << indent << e.second << std::endl;
		return oss.str();
	}

	//////////////

	// PYOOMPH_TIME_ADD_RESIDUAL=1 prints a per-phase breakdown of add_residual/expand_placeholders
	// on stderr - how dev_docs/code_generation.md was measured. That phase has no other visibility.
	// Namespace scope, not a function-local static: these are read once per visited expression node,
	// and a function-local static would cost a guard-variable acquire load every time.
	static const bool __time_add_residual_on = getenv("PYOOMPH_TIME_ADD_RESIDUAL") != NULL;
	static const bool __expand_memo_on = getenv("PYOOMPH_DISABLE_EXPAND_MEMO") == NULL;
	static inline bool __time_add_residual() { return __time_add_residual_on; }
	struct __phase_timer
	{
		std::chrono::steady_clock::time_point t0;
		const char *name;
		__phase_timer(const char *n) : t0(std::chrono::steady_clock::now()), name(n) {}
		~__phase_timer()
		{
			if (__time_add_residual())
			{
				double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
				if (dt > 1e-3)
					std::cerr << "[add_residual] " << name << " " << dt << " s" << std::endl;
			}
		}
	};

	// Ambient code-generation state, set and restored around the symbolic differentiation and
	// code-emission passes below and read by GiNaC hooks that have no other channel.
	//
	// State that BELONGS to one code has been moved onto FiniteElementCode (the Hessian accumulators,
	// the Hessian subexpression mapper, the hoisted-coefficient counter). What is left is genuinely
	// ambient and stays that way on purpose: the readers are GiNaC `_eval` functions and print hooks
	// whose only inputs are `ex` arguments - unitvect_eval, grad, div, the callback printers - so
	// threading a code in would mean putting one into the expression itself, changing the symbolic
	// representation and with it every generated-code hash and the archive format. Nothing is bought
	// by that: one FiniteElementCode belongs to exactly one Problem, and code generation is
	// one-code-at-a-time by construction.
	//
	// They are thread_local so that two threads generating code never see each other's pass, and
	// every set site is an RAII scope so that no path - including the several that can throw
	// mid-differentiation - can leave one latched.
	thread_local FiniteElementCode *__current_code = NULL;             // Code object currently being processed (residual/Jacobian/Hessian assembly)
	thread_local const ShapeExpansion *__deriv_subexpression_wrto = NULL; // Set while differentiating a subexpression w.r.t. a specific field/direction
	thread_local bool __derive_shapes_by_second_index = false;         // True while building the "second" (l_shape2) index of a Hessian double-loop
	thread_local bool __in_pitchfork_symmetry_constraint = false;      // True while generating the extra pitchfork-symmetry-breaking constraint equations
	thread_local int * __derive_only_by_expansion_mode = NULL;         // If set, only derivatives w.r.t. this azimuthal/Fourier expansion mode are kept
	thread_local bool __ignore_dpsi_coord_diffs_in_jacobian = false;   // Suppresses dpsi/dX (moving-mesh) contributions in the Jacobian for the current residual
	// Whether eval_at_expansion_mode() also tags the element size, i.e. whether tau and anything else
	// built from the element size is perturbed along with the mesh in an azimuthal/normal-mode
	// expansion. False (the default) freezes it at the base state, which is both the meaningful
	// reading for a Cartesian mesh metric and the one whose augmented Jacobian is exact. True
	// reproduces the behaviour that existed before this flag.
	bool expand_element_size_in_expansion_modes = false;
	thread_local bool __in_hessian = false;                            // True while performing second-order (Hessian) differentiation

	// Sets one of the ambient flags above for its lifetime and RESTORES the previous value, rather than
	// clearing to a hard default the way the hand-written pairs did. Restoring matters because these
	// regions nest (a subexpression derivative is taken in the middle of a Hessian pass), and because
	// several of the differentiation calls in between can throw - an aborted code generation used to
	// leave a flag latched for everything generated afterwards in the same process.
	template <typename T>
	class AmbientCodegenScope
	{
		T &ref;
		T prev;

	public:
		AmbientCodegenScope(T &r, T v) : ref(r), prev(r) { ref = v; }
		~AmbientCodegenScope() { ref = prev; }
		AmbientCodegenScope(const AmbientCodegenScope &) = delete;
		AmbientCodegenScope &operator=(const AmbientCodegenScope &) = delete;
	};

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

	// Removes the __partial_t_mass_matrix probe from an expression. A term carrying that symbol as a
	// factor (added by time_derivative_of_integral, i.e. by add_dweak_dt) exists only so that
	// diff(., __partial_t_mass_matrix) below finds dI/dU, which is what d/dt of an integral
	// differentiates to. The symbol has no value, so it has to be substituted away before the
	// residual or the Jacobian half is printed or tested for being zero.
	static inline GiNaC::ex strip_mass_matrix_marker(const GiNaC::ex &e)
	{
		if (!e.has(pyoomph::expressions::__partial_t_mass_matrix))
			return e;
		return e.subs(pyoomph::expressions::__partial_t_mass_matrix == 0);
	}

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
					repl.history_geometry = false;
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
			// The two geometric symbols below carry a history level for the same reason and must be
			// redirected as well: a steady solve never fills the history slots, so anything left
			// pointing at them would read uninitialised geometry.
			else if (GiNaC::is_a<GiNaC::GiNaCNormalSymbol>(inp))
			{
				auto &ns = (GiNaC::ex_to<GiNaC::GiNaCNormalSymbol>(inp)).get_struct();
				if (ns.history_step > 0)
				{
					NormalSymbol repl = ns;
					repl.history_step = 0;
					extra_steady_routine = true;
					return GiNaC::GiNaCNormalSymbol(repl);
				}
				return inp.map(*this);
			}
			else if (GiNaC::is_a<GiNaC::GiNaCElementSizeSymbol>(inp))
			{
				auto &es = (GiNaC::ex_to<GiNaC::GiNaCElementSizeSymbol>(inp)).get_struct();
				if (es.history_step > 0)
				{
					ElementSizeSymbol repl = es;
					repl.history_step = 0;
					extra_steady_routine = true;
					return GiNaC::GiNaCElementSizeSymbol(repl);
				}
				return inp.map(*this);
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
	// How often the mapper is entered vs. how many *distinct* subexpressions it is entered on -
	// the measurement that identified the DAG-walked-as-a-tree blowup. GiNaC expressions are reference-counted DAGs but
	// ex::map walks them as trees, so a subtree that is shared k times is visited k times.
	// Distinctness is approximated by GiNaC's cached expression hash - good enough to tell an
	// order-of-magnitude gap between "visited once" and "visited a million times" apart.
	unsigned long __rfnd_calls = 0;
	unsigned long __rfnd_hits = 0;
	std::unordered_set<unsigned> __rfnd_distinct;

	// The opposite-side facet placeholder code (see FiniteElementCode::internal_facet_opposite_dummy)
	// mirrors the skeleton's own field declarations, but its elements are never added to a mesh and
	// hence never get equation numbers: reading such a field through '|-' silently evaluated to zero.
	// A field living on the skeleton is single-valued per facet anyway, so there is nothing sensible to
	// return. Bulk fields reached through the placeholder (the usual jump(...,at_facet=True) pattern)
	// stay legal, which is why only fields that no code in the bulk chain declares are rejected.
	static void reject_opposite_facet_field(FiniteElementCode *mycode, const std::string &fieldname)
	{
		if (!mycode || !mycode->internal_facet_opposite_dummy || fieldname.empty())
			return;
		if (!mycode->get_field_by_name(fieldname))
			return;
		for (FiniteElementCode *b = mycode->get_bulk_element(); b; b = b->get_bulk_element())
			if (b->get_field_by_name(fieldname))
				return;
		throw_runtime_error("Field '" + fieldname + "' is defined on the interior-facet skeleton '_internal_facets_' itself, i.e. it is single-valued on each facet. It therefore cannot be evaluated on the opposite side (domain '|-', which also happens inside jump(...,at_facet=True) and avg(...,at_facet=True)). Just use it without any domain specification.");
	}

	// Memoising wrapper around do_replace(): rewriting a subexpression is a pure function of it and
	// the mapper's code/where/extra_test_scale, so a DAG node ex::map hands us repeatedly can come
	// from the table. Two things must survive a hit: repl_count (the fixpoint loop reads it as "did
	// this pass change anything", so the first expansion's delta is replayed), and
	// code->expanded_scales (only ever assigned, same value for the same field, so a skipped repeat
	// assignment changes nothing).
	// Nothing whose expansion touched a Python-overridable hook is cached - those are user code, and
	// there is already a disabled cache for exactly them (expanded_additional_field_cache, "Do not
	// use the cache for the moment") that this must not quietly re-enable. The exclusion propagates
	// to ancestors via python_hook_calls, since caching a parent freezes the hook's result just as
	// effectively; only excluding the calling node was tried first and was not enough.
	GiNaC::ex ReplaceFieldsToNonDimFields::operator()(const GiNaC::ex &inp)
	{
		if (__time_add_residual())
		{
			__rfnd_calls++;
			__rfnd_distinct.insert(inp.gethash());
		}
		// On by default; PYOOMPH_DISABLE_EXPAND_MEMO=1 turns it off. On material-library-heavy
		// models the memo changes how GiNaC orders and signs the factors of a product - verified
		// numerically equivalent, see dev_docs/code_generation.md.
		if (!__expand_memo_on)
			return do_replace(inp);
		// Numbers are never cached. GiNaC compares and hashes them by value, not by representation,
		// so an exact -1 and an inexact -1.0 are one and the same key here and the memo hands back
		// whichever of the two was met first. Which is harmless for every later computation and not
		// harmless at all for the generated C source: an exponent that comes back inexact makes
		// print_simplest_form's ExactifyWholeNumberExponents necessary, and without it a reciprocal
		// x^(-1) is printed as a multiplication by x. A leaf costs nothing to expand anyway.
		if (GiNaC::is_a<GiNaC::numeric>(inp))
			return do_replace(inp);
		const unsigned h = inp.gethash();
		auto it = memo.find(h);
		if (it != memo.end())
		{
			for (auto &e : it->second)
			{
				if (e.key.is_equal(inp))
				{
					if (__time_add_residual())
						__rfnd_hits++;
					repl_count += e.repl_delta;
					return e.value;
				}
			}
		}
		const unsigned repl_before = repl_count;
		const unsigned long hooks_before = python_hook_calls;
		GiNaC::ex res = do_replace(inp);
		if (python_hook_calls == hooks_before)
			memo[h].push_back(MemoEntry{inp, res, repl_count - repl_before});
		return res;
	}

	GiNaC::ex ReplaceFieldsToNonDimFields::do_replace(const GiNaC::ex &inp)
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
			GiNaC::ex sub = repl(expr);
			// The nested mapper has its own hook counter and its own memo; fold its hook calls into
			// ours so this node is not cached if the delegated expansion touched Python.
			python_hook_calls += repl.python_hook_calls;
			return sub.map(*this);
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
			reject_opposite_facet_field(mycode, fieldname);
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
				python_hook_calls++; // see the comment on operator()
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
			reject_opposite_facet_field(mycode, fieldname);
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
				python_hook_calls++; // see the comment on operator()
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
			reject_opposite_facet_field(mycode, fieldname);
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
			python_hook_calls++; // see the comment on operator()
			GiNaC::ex res = scale * mycode->expand_additional_testfunction(fieldname, inp, code);
			ReplaceFieldsToNonDimFields further_expansion(mycode, where);
			res = res.map(further_expansion);
			return res;
		}
		else if (is_ex_the_function(inp, expressions::testfunction))
		{
			FiniteElementCode *mycode = code->resolve_corresponding_code(inp, &fieldname, NULL);
			reject_opposite_facet_field(mycode, fieldname);
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
			python_hook_calls++; // see the comment on operator()
			GiNaC::ex res = mycode->expand_additional_testfunction(fieldname, inp, code);
			ReplaceFieldsToNonDimFields further_expansion(mycode, where);
			res = res.map(further_expansion);
			return res;
		}
		else if (GiNaC::is_a<GiNaC::GiNaCDelayedPythonCallbackExpansion>(inp))
		{
			//   std::cout << "FOUND DELAYED CALLBACK" <<std::endl << std::flush;
			python_hook_calls++; // see the comment on operator()
			GiNaC::ex func_res = GiNaC::ex_to<GiNaC::GiNaCDelayedPythonCallbackExpansion>(inp).get_struct().cme->f();
			//   std::cout << "FUNC RES" << func_res << std::endl << std::flush;
			return func_res.map(*this);
		}
		else if (is_ex_the_function(inp, expressions::python_multi_cb_function))
		{
			python_hook_calls++; // Python hook, and mutates code->multi_return_ccodes
			GiNaC::ex invok = inp.map(*this);
			// Was unguarded since the initial commit, unlike every other trace in this function, so
			// any model using a multi-return callback (solid mechanics, the material library) dumped
			// one expanded GiNaC expression per invocation to stdout - 72 of cantilever's 169 output
			// lines, 36 of simple_fsi's 198.
			if (pyoomph_verbose)
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
		return lhs.get_code()->get_creation_index() < rhs.get_code()->get_creation_index() || 
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
		return lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.is_derived2() == rhs.is_derived2() && lhs.get_derived_direction2() == rhs.get_derived_direction2() && lhs.is_with_coordsys() == rhs.is_with_coordsys() && lhs.is_derived_by_lshape2() == rhs.is_derived_by_lshape2() && lhs.history_step == rhs.history_step && lhs.expansion_mode == rhs.expansion_mode;
	}
	bool operator<(const ElementSizeSymbol &lhs, const ElementSizeSymbol &rhs)
	{
		return lhs.get_code()->get_creation_index() < rhs.get_code()->get_creation_index() || (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() < rhs.is_lagrangian()) || (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() < rhs.is_derived()) || (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() < rhs.get_derived_direction()) ||
			   (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.is_derived2() < rhs.is_derived2()) ||
			   (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.is_derived2() == rhs.is_derived2() && lhs.get_derived_direction2() < rhs.get_derived_direction2()) ||
			   (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.is_derived2() == rhs.is_derived2() && lhs.get_derived_direction2() == rhs.get_derived_direction2() && lhs.is_with_coordsys() < rhs.is_with_coordsys()) ||
			   (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.is_derived2() == rhs.is_derived2() && lhs.get_derived_direction2() == rhs.get_derived_direction2() && lhs.is_with_coordsys() == rhs.is_with_coordsys() && lhs.is_derived_by_lshape2() < rhs.is_derived_by_lshape2()) ||
			   (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.is_derived2() == rhs.is_derived2() && lhs.get_derived_direction2() == rhs.get_derived_direction2() && lhs.is_with_coordsys() == rhs.is_with_coordsys() && lhs.is_derived_by_lshape2() == rhs.is_derived_by_lshape2() && lhs.history_step < rhs.history_step) ||
			   (lhs.get_code() == rhs.get_code() && lhs.is_lagrangian() == rhs.is_lagrangian() && lhs.is_derived() == rhs.is_derived() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.is_derived2() == rhs.is_derived2() && lhs.get_derived_direction2() == rhs.get_derived_direction2() && lhs.is_with_coordsys() == rhs.is_with_coordsys() && lhs.is_derived_by_lshape2() == rhs.is_derived_by_lshape2() && lhs.history_step == rhs.history_step && lhs.expansion_mode < rhs.expansion_mode);
	}

	bool operator==(const NodalDeltaSymbol &lhs, const NodalDeltaSymbol &rhs)
	{
		return lhs.get_code() == rhs.get_code();
	}
	bool operator<(const NodalDeltaSymbol &lhs, const NodalDeltaSymbol &rhs)
	{
		return lhs.get_code()->get_creation_index() < rhs.get_code()->get_creation_index();
	}
      
	bool operator==(const NormalSymbol &lhs, const NormalSymbol &rhs)
	{
		return lhs.get_code() == rhs.get_code() && lhs.get_direction() == rhs.get_direction() && lhs.get_derived_direction() == rhs.get_derived_direction() && lhs.get_derived_direction2() == rhs.get_derived_direction2() && lhs.is_derived_by_lshape2() == rhs.is_derived_by_lshape2()  && lhs.expansion_mode == rhs.expansion_mode && lhs.no_jacobian == rhs.no_jacobian && lhs.no_hessian == rhs.no_hessian && lhs.history_step == rhs.history_step && lhs.spatial_deriv_direction == rhs.spatial_deriv_direction;
	}
	// Ordering by a tuple rather than the former chain of nested equality tests, in which every new
	// clause had to repeat all the preceding ones - adding spatial_deriv_direction to that chain
	// would have meant a tenth copy of the whole prefix.
	bool operator<(const NormalSymbol &lhs, const NormalSymbol &rhs)
	{
		return std::make_tuple(lhs.get_code()->get_creation_index(), lhs.get_direction(), lhs.get_derived_direction(),
							   lhs.get_derived_direction2(), lhs.is_derived_by_lshape2(), lhs.expansion_mode,
							   lhs.no_jacobian, lhs.no_hessian, lhs.history_step, lhs.spatial_deriv_direction) <
			   std::make_tuple(rhs.get_code()->get_creation_index(), rhs.get_direction(), rhs.get_derived_direction(),
							   rhs.get_derived_direction2(), rhs.is_derived_by_lshape2(), rhs.expansion_mode,
							   rhs.no_jacobian, rhs.no_hessian, rhs.history_step, rhs.spatial_deriv_direction);
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
		return lhs.field == rhs.field && lhs.dt_order == rhs.dt_order && lhs.basis == rhs.basis && lhs.is_derived == rhs.is_derived && lhs.is_derived_other_index == rhs.is_derived_other_index && lhs.basis == rhs.basis && lhs.nodal_coord_dir == rhs.nodal_coord_dir && lhs.time_history_index == rhs.time_history_index && lhs.history_geometry == rhs.history_geometry && (lhs.dt_order == 0 || lhs.dt_scheme == rhs.dt_scheme) && (lhs.no_jacobian == rhs.no_jacobian) && (lhs.no_hessian == rhs.no_hessian) && (lhs.expansion_mode == rhs.expansion_mode) && (lhs.nodal_coord_dir2 == rhs.nodal_coord_dir2);
	}
	bool operator<(const ShapeExpansion &lhs, const ShapeExpansion &rhs)
	{
		return lhs.field->get_creation_index() < rhs.field->get_creation_index() || (lhs.field == rhs.field && lhs.dt_order < rhs.dt_order) || (lhs.field == rhs.field && lhs.dt_order == rhs.dt_order && lhs.basis->get_creation_index() < rhs.basis->get_creation_index()) || (lhs.field == rhs.field && lhs.dt_order == rhs.dt_order && lhs.basis == rhs.basis && lhs.is_derived < rhs.is_derived) || (lhs.field == rhs.field && lhs.dt_order == rhs.dt_order && lhs.basis == rhs.basis && lhs.is_derived == rhs.is_derived && lhs.is_derived_other_index < rhs.is_derived_other_index) || (lhs.field == rhs.field && lhs.dt_order == rhs.dt_order && lhs.basis == rhs.basis && lhs.is_derived == rhs.is_derived && lhs.is_derived_other_index == rhs.is_derived_other_index && lhs.nodal_coord_dir < rhs.nodal_coord_dir) || (lhs.field == rhs.field && lhs.dt_order == rhs.dt_order && lhs.basis == rhs.basis && lhs.is_derived == rhs.is_derived && lhs.is_derived_other_index == rhs.is_derived_other_index && lhs.nodal_coord_dir == rhs.nodal_coord_dir && lhs.time_history_index < rhs.time_history_index) || (lhs.field == rhs.field && lhs.dt_order == rhs.dt_order && lhs.basis == rhs.basis && lhs.is_derived == rhs.is_derived && lhs.is_derived_other_index == rhs.is_derived_other_index && lhs.nodal_coord_dir == rhs.nodal_coord_dir && lhs.time_history_index == rhs.time_history_index && (lhs.dt_order > 0 && lhs.dt_scheme < rhs.dt_scheme)) || (lhs.field == rhs.field && lhs.dt_order == rhs.dt_order && lhs.basis == rhs.basis && lhs.is_derived == rhs.is_derived && lhs.is_derived_other_index == rhs.is_derived_other_index && lhs.nodal_coord_dir == rhs.nodal_coord_dir && lhs.time_history_index == rhs.time_history_index && (lhs.dt_order == 0 || lhs.dt_scheme == rhs.dt_scheme) && (lhs.no_jacobian < rhs.no_jacobian)) || (lhs.field == rhs.field && lhs.dt_order == rhs.dt_order && lhs.basis == rhs.basis && lhs.is_derived == rhs.is_derived && lhs.is_derived_other_index == rhs.is_derived_other_index && lhs.nodal_coord_dir == rhs.nodal_coord_dir && lhs.time_history_index == rhs.time_history_index && (lhs.dt_order == 0 || lhs.dt_scheme == rhs.dt_scheme) && (lhs.no_jacobian == rhs.no_jacobian) && (lhs.no_hessian < rhs.no_hessian)) || (lhs.field == rhs.field && lhs.dt_order == rhs.dt_order && lhs.basis == rhs.basis && lhs.is_derived == rhs.is_derived && lhs.is_derived_other_index == rhs.is_derived_other_index && lhs.nodal_coord_dir == rhs.nodal_coord_dir && lhs.time_history_index == rhs.time_history_index && (lhs.dt_order == 0 || lhs.dt_scheme == rhs.dt_scheme) && (lhs.no_jacobian == rhs.no_jacobian) && (lhs.no_hessian == rhs.no_hessian) && (lhs.expansion_mode < rhs.expansion_mode)) || (lhs.field == rhs.field && lhs.dt_order == rhs.dt_order && lhs.basis == rhs.basis && lhs.is_derived == rhs.is_derived && lhs.is_derived_other_index == rhs.is_derived_other_index && lhs.nodal_coord_dir == rhs.nodal_coord_dir && lhs.time_history_index == rhs.time_history_index && (lhs.dt_order == 0 || lhs.dt_scheme == rhs.dt_scheme) && (lhs.no_jacobian == rhs.no_jacobian) && (lhs.no_hessian == rhs.no_hessian) && (lhs.expansion_mode == rhs.expansion_mode) && (lhs.nodal_coord_dir2 < rhs.nodal_coord_dir2)) || (lhs.field == rhs.field && lhs.dt_order == rhs.dt_order && lhs.basis == rhs.basis && lhs.is_derived == rhs.is_derived && lhs.is_derived_other_index == rhs.is_derived_other_index && lhs.nodal_coord_dir == rhs.nodal_coord_dir && lhs.time_history_index == rhs.time_history_index && (lhs.dt_order == 0 || lhs.dt_scheme == rhs.dt_scheme) && (lhs.no_jacobian == rhs.no_jacobian) && (lhs.no_hessian == rhs.no_hessian) && (lhs.expansion_mode == rhs.expansion_mode) && (lhs.nodal_coord_dir2 == rhs.nodal_coord_dir2) && (lhs.history_geometry < rhs.history_geometry));
	}

	bool operator==(const TestFunction &lhs, const TestFunction &rhs)
	{
		return lhs.field == rhs.field && lhs.basis == rhs.basis && lhs.nodal_coord_dir == rhs.nodal_coord_dir && lhs.is_derived_other_index == rhs.is_derived_other_index && lhs.nodal_coord_dir2 == rhs.nodal_coord_dir2;
	}
	bool operator<(const TestFunction &lhs, const TestFunction &rhs)
	{
		return lhs.field->get_creation_index() < rhs.field->get_creation_index() || (lhs.field == rhs.field && lhs.basis->get_creation_index() < rhs.basis->get_creation_index()) || (lhs.field == rhs.field && lhs.basis == rhs.basis && lhs.nodal_coord_dir < rhs.nodal_coord_dir) || (lhs.field == rhs.field && lhs.basis == rhs.basis && lhs.nodal_coord_dir == rhs.nodal_coord_dir && lhs.is_derived_other_index < rhs.is_derived_other_index) || (lhs.field == rhs.field && lhs.basis == rhs.basis && lhs.nodal_coord_dir == rhs.nodal_coord_dir && lhs.is_derived_other_index == rhs.is_derived_other_index && lhs.nodal_coord_dir2 < rhs.nodal_coord_dir2);
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
		// Only the interpolated value is affected by the history geometry; the nodal values behind it
		// (get_dt_values_name) are geometry-independent and stay shared.
		dtstring += get_history_geometry_suffix();
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

	unsigned FiniteElementField::next_creation_index = 0;

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
			this->jacobian_contribution_for_code[code]=std::map<unsigned,std::set<FiniteElementField*,FiniteElementFieldPtrLess>>();
		}
		if (!this->jacobian_contribution_for_code[code].count(residual_index))
		{
			this->jacobian_contribution_for_code[code][residual_index]=std::set<FiniteElementField*,FiniteElementFieldPtrLess>();
		}
		this->jacobian_contribution_for_code[code][residual_index].insert(other);

	}

    bool FiniteElementField::has_mass_matrix_contribution_for_code(FiniteElementCode *code,unsigned residual_index, FiniteElementField *other)
	{
		return this->mass_matrix_contribution_for_code.count(code) > 0 && this->mass_matrix_contribution_for_code[code].count(residual_index) > 0 && this->mass_matrix_contribution_for_code[code][residual_index].count(other) > 0;
	}
    bool FiniteElementField::has_hessian_contribution_for_code(FiniteElementCode *code,unsigned residual_index, FiniteElementField *other)
	{
		return this->hessian_contribution_for_code.count(code) > 0 && this->hessian_contribution_for_code[code].count(residual_index) > 0 && this->hessian_contribution_for_code[code][residual_index].count(other) > 0;
	}
    void FiniteElementField::mark_hessian_contribution_for_code(FiniteElementCode *code,unsigned residual_index, FiniteElementField *other)
	{
		this->hessian_contribution_for_code[code][residual_index].insert(other);
	}
    void FiniteElementField::mark_mass_matrix_contribution_for_code(FiniteElementCode *code,unsigned residual_index, FiniteElementField *other)
	{
		if (!this->mass_matrix_contribution_for_code.count(code))
		{
			this->mass_matrix_contribution_for_code[code]=std::map<unsigned,std::set<FiniteElementField*,FiniteElementFieldPtrLess>>();
		}
		if (!this->mass_matrix_contribution_for_code[code].count(residual_index))
		{
			this->mass_matrix_contribution_for_code[code][residual_index]=std::set<FiniteElementField*,FiniteElementFieldPtrLess>();
		}
		this->mass_matrix_contribution_for_code[code][residual_index].insert(other);
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
		// The whole hanginfo table is loop-invariant; the [node] index that follows at the use site is
		// not. Aliasing the row (i.e. including the field index, which is a const declared above) means
		// the macros index a local instead of walking shapeinfo-> once per test/trial node.
		const bool is_pos = dynamic_cast<PositionFiniteElementSpace *>(space) != NULL;
		const std::string access = forcode->get_shape_info_str(space) + (is_pos ? "->hanginfo_Pos[" : "->hanginfo[") + get_nodal_index_str(forcode) + "]";
		const std::string alias_name = "_" + forcode->get_owner_prefix(space) + "hang_" + (is_pos ? "Pos_" : "") + this->name;
		return forcode->alias_buffer_access(access, alias_name, "JITHangInfo_t * const " + alias_name);
	}

	std::string ShapeExpansion::get_num_nodes_str(FiniteElementCode *forcode) const
	{
		return this->basis->get_space()->get_num_nodes_str(forcode);
	}

	// The single place that mints buffer-alias names for the six shape buffers. Both the trial-side
	// accesses printed inside the loops (ShapeExpansion::get_shape_string) and the test-function
	// pointers emitted per space block go through it, so they share one declaration instead of loading
	// the same buffer twice. Only the Eulerian derivatives carry a history index.
	static std::string alias_shape_buffer(FiniteElementCode *forcode, const FiniteElementSpace *sp,
										  std::string base, const std::string &stem, int history,
										  const std::function<std::string(const std::string &)> &decl)
	{
		std::string hist_suffix;
		if (history >= 0)
		{
			size_t br = base.find('[');
			if (br == std::string::npos)
				base = base + "[" + std::to_string(history) + "]"; // flat per-space buffer, e.g. dx_shape_DL
			else
				base = base.substr(0, br) + "[" + std::to_string(history) + "]" + base.substr(br);
			if (history)
				hist_suffix = "_h" + std::to_string(history);
		}
		const std::string access = forcode->get_shape_info_str(sp) + "->" + base;
		// Collision-free by construction: (owner, buffer, space, history) determines the access.
		const std::string name = "_" + forcode->get_owner_prefix(sp) + stem + "_" + sp->get_shape_name() + hist_suffix;
		return forcode->alias_buffer_access(access, name, decl(name));
	}

	// The timestepper weight arrays are re-pointed by prepare_shape_buffer_for_integration (the
	// "_degr" variants switch between BDF1 and BDF2 on the first unsteady step), i.e. before the
	// generated function is entered, so they alias like the shape buffers. This one matters more than
	// it looks: weights[0] appears inside Jacobian ENTRIES, i.e. in the innermost trial loop, and one
	// surviving shapeinfo-> read in there costs the hoist of all the others.
	static std::string alias_timestepper_weights(FiniteElementCode *forcode, const std::string &kind, const std::string &scheme)
	{
		const std::string access = "shapeinfo->timestepper_weights_" + kind + "_" + scheme;
		const std::string name = "_tsw_" + kind + "_" + scheme;
		return forcode->alias_buffer_access(access, name, "const double * const " + name);
	}

	// Slot 0 of the same array, as a plain double. The Jacobian entries only ever read that one slot,
	// and they read it in the innermost trial loop - aliasing the ARRAY there would still cost a load
	// per iteration, which is most of what the aliases are trying to remove.
	static std::string alias_timestepper_weight0(FiniteElementCode *forcode, const std::string &kind, const std::string &scheme)
	{
		const std::string access = "shapeinfo->timestepper_weights_" + kind + "_" + scheme + "[0]";
		const std::string name = "_tsw0_" + kind + "_" + scheme;
		return forcode->alias_buffer_access(access, name, "const double " + name);
	}

	std::string ShapeExpansion::get_shape_string(FiniteElementCode *forcode, std::string nodal_index) const
	{
		// D0 basis functions print as the literal 1 - nothing to index and nothing to alias.
		if (basis->get_shape_string(forcode, nodal_index) == "1")
			return "1";
		// Only the EULERIAN derivative of a shape function depends on where the nodes are, so only that
		// one carries a history index: dx_shapes[k][...] rather than dx_shapes[...], with 0 being the
		// current configuration - the same convention as int_pt_weight[]. The undifferentiated shapes
		// and the Lagrangian/local-coordinate derivatives are properties of the reference element and
		// do not move with the mesh, so they have no such index.
		int history = -1;
		if (basis->is_eulerian_deriv())
		{
			const bool past = history_geometry && time_history_index > 0 && forcode->history_geometry_is_relevant();
			history = past ? (int)time_history_index : 0;
		}
		BasisFunction *bf = basis;
		const std::string alias = alias_shape_buffer(forcode, bf->get_space(), bf->get_shape_buffer_base(),
													bf->get_shape_alias_stem(), history,
													[bf](const std::string &n) { return bf->get_shape_alias_decl(n); });
		return alias + "[" + nodal_index + "]" + bf->get_shape_index_suffix();
	}

	std::string ShapeExpansion::get_nodal_data_string(FiniteElementCode *forcode, std::string indexstr) const
	{
		if (this->dt_order > 0)
			return this->get_dt_values_name(forcode) + "[" + indexstr + "]";
		return forcode->alias_nodal_data(this->basis->get_space()) + "[" + indexstr + "][" + get_nodal_index_str(forcode) + "][" + std::to_string(time_history_index) + "]";
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

	unsigned BasisFunction::next_creation_index = 0;

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

	// The four continuous spaces share one buffer indexed by SPACE_INDEX_*; every other space (DL,
	// D0, the DG family) has a buffer of its own named after the space.
	static bool basis_space_is_indexed(const std::string &sn)
	{
		return sn == "C2TB" || sn == "C2" || sn == "C1TB" || sn == "C1";
	}

	static std::string basis_buffer_base(const std::string &array, const std::string &sn)
	{
		if (basis_space_is_indexed(sn))
			return array + "s[SPACE_INDEX_" + sn + "]";
		return array + "_" + sn;
	}

	std::string BasisFunction::get_shape_buffer_base() const
	{
		return basis_buffer_base("shape", space->get_shape_name());
	}

	// Assembled from the three pieces above rather than spelled out per subclass, so that a buffer
	// alias built from get_shape_buffer_base() cannot drift away from the access it replaces.
	std::string BasisFunction::get_shape_string(FiniteElementCode *, std::string nodal_index) const
	{
		return get_shape_buffer_base() + "[" + nodal_index + "]" + get_shape_index_suffix();
	}

	// Second derivatives. The cache deliberately sits here, on the space's root BasisFunction, so
	// that both routes to the same mixed derivative land in the same slot - see the declaration in
	// codegen.hpp. All slots are filled at once (as get_diff_x does for the first derivatives) so
	// that creation_index assignment stays reproducible across separate runs of the same input.
	BasisFunction *BasisFunction::get_second_diff_x(unsigned dir_inner, unsigned dir_outer)
	{
		if (basis_deriv2_x.empty())
		{
			basis_deriv2_x.resize(MAX_N2DERIV, NULL);
			for (unsigned int i = 0; i < 3; i++)
				for (unsigned int j = 0; j < 3; j++)
					basis_deriv2_x[PYOOMPH_D2_SLOT(i, j)] = new D2XBasisFunction(space, i, j);
		}
		return basis_deriv2_x[PYOOMPH_D2_SLOT(dir_inner, dir_outer)];
	}

	BasisFunction *BasisFunction::get_second_diff_S(unsigned dir_inner, unsigned dir_outer)
	{
		if (local_coord_deriv2_x.empty())
		{
			local_coord_deriv2_x.resize(MAX_N2DERIV, NULL);
			for (unsigned int i = 0; i < 3; i++)
				for (unsigned int j = 0; j < 3; j++)
					local_coord_deriv2_x[PYOOMPH_D2_SLOT(i, j)] = new D2XBasisFunctionLocalCoord(space, i, j);
		}
		return local_coord_deriv2_x[PYOOMPH_D2_SLOT(dir_inner, dir_outer)];
	}

	BasisFunction::~BasisFunction()
	{
		for (unsigned int i = 0; i < basis_deriv_x.size(); i++)
			if (basis_deriv_x[i])
				delete basis_deriv_x[i];
		for (unsigned int i = 0; i < lagr_deriv_x.size(); i++)
			if (lagr_deriv_x[i])
				delete lagr_deriv_x[i];
		// local_coord_deriv_x used to be leaked here
		for (unsigned int i = 0; i < local_coord_deriv_x.size(); i++)
			if (local_coord_deriv_x[i])
				delete local_coord_deriv_x[i];
		for (unsigned int i = 0; i < basis_deriv2_x.size(); i++)
			if (basis_deriv2_x[i])
				delete basis_deriv2_x[i];
		for (unsigned int i = 0; i < local_coord_deriv2_x.size(); i++)
			if (local_coord_deriv2_x[i])
				delete local_coord_deriv2_x[i];
	}
	std::string BasisFunction::to_string()
	{
		return "BASIS of " + space->get_name();
	}

	// Differentiating an already-once-differentiated basis a second time. The cache is looked up on
	// the space's ROOT basis, and the index pair is canonicalised to (min, max) - but only where
	// d/dx_i d/dx_j is actually symmetric, i.e. on a domain without codimension. On a surface
	// pyoomph's metric-based derivative gives the tangential derivative of the surface gradient,
	// which is genuinely asymmetric (see jitbridge.h), so there the two orders must stay distinct.
	BasisFunction *D1XBasisFunction::get_diff_x(unsigned dir_outer)
	{
		unsigned a = direction, b = dir_outer;
		const FiniteElementCode *code = space->get_code();
		if (code && code->get_dimension() == (int)code->nodal_dimension() && a > b)
		{
			std::swap(a, b);
		}
		return space->get_basis()->get_second_diff_x(a, b);
	}

	BasisFunction *D1XBasisFunction::get_diff_X(unsigned)
	{
		throw_runtime_error("Cannot mix an Eulerian and a Lagrangian spatial derivative of a basis function, i.e. d/dX(d/dx(...))");
	}

	BasisFunction *D1XBasisFunction::get_diff_S(unsigned)
	{
		throw_runtime_error("Cannot mix an Eulerian and a local-coordinate derivative of a basis function, i.e. d/dS(d/dx(...))");
	}

	BasisFunction *D1XBasisFunctionLagr::get_diff_x(unsigned)
	{
		throw_runtime_error("Cannot mix a Lagrangian and an Eulerian spatial derivative of a basis function, i.e. d/dx(d/dX(...))");
	}

	BasisFunction *D1XBasisFunctionLagr::get_diff_X(unsigned)
	{
		throw_runtime_error("Second order LAGRANGIAN derivatives of basis functions are not implemented. Only the Eulerian ones, e.g. grad(grad(...)) or div(grad(...)), are available.");
	}

	BasisFunction *D1XBasisFunctionLagr::get_diff_S(unsigned)
	{
		throw_runtime_error("Cannot mix a Lagrangian and a local-coordinate derivative of a basis function");
	}

	BasisFunction *D1XBasisFunctionLocalCoord::get_diff_S(unsigned dir_outer)
	{
		unsigned a = direction, b = dir_outer;
		if (a > b) std::swap(a, b); // d2/(ds_a ds_b) is always symmetric - these are reference-element coordinates
		return space->get_basis()->get_second_diff_S(a, b);
	}

	// Third and higher derivatives
	BasisFunction *D2XBasisFunction::get_diff_x(unsigned)
	{
		throw_runtime_error("Cannot handle third order derivatives of basis functions");
	}

	BasisFunction *D2XBasisFunction::get_diff_X(unsigned)
	{
		throw_runtime_error("Cannot handle third order derivatives of basis functions");
	}

	BasisFunction *D2XBasisFunction::get_diff_S(unsigned)
	{
		throw_runtime_error("Cannot handle third order derivatives of basis functions");
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

	std::string D1XBasisFunction::get_shape_buffer_base() const
	{
		return basis_buffer_base("dx_shape", space->get_shape_name());
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

	std::string D1XBasisFunctionLagr::get_shape_buffer_base() const
	{
		return basis_buffer_base("dX_shape", space->get_shape_name());
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

	std::string D1XBasisFunctionLocalCoord::get_shape_buffer_base() const
	{
		return basis_buffer_base("dS_shape", space->get_shape_name());
	}

	// -------------------------------------------------------------------------------------------
	// Second derivatives. The slot index is resolved to an integer literal HERE, at generation
	// time, by calling the very same PYOOMPH_D2_SLOT macro that the runtime fill uses - so the two
	// cannot drift apart.
	// -------------------------------------------------------------------------------------------

	std::string D2XBasisFunction::get_c_varname(FiniteElementCode *, std::string test_index)
	{
		return "d2x_testfunction[" + test_index + "][" + std::to_string(get_slot()) + "]";
	}

	static std::string d2_direction_string(unsigned dir)
	{
		return (dir == 0 ? "x" : (dir == 1 ? "y" : "z"));
	}

	std::string D2XBasisFunction::to_string()
	{
		return "d^2/(d" + d2_direction_string(direction) + " d" + d2_direction_string(direction2) + ") of BASIS of " + space->get_name();
	}

	std::string D2XBasisFunction::get_shape_buffer_base() const
	{
		return basis_buffer_base("d2x_shape", space->get_shape_name());
	}

	std::string D2XBasisFunctionLocalCoord::get_c_varname(FiniteElementCode *, std::string test_index)
	{
		return "d2S_testfunction[" + test_index + "][" + std::to_string(get_slot()) + "]";
	}

	std::string D2XBasisFunctionLocalCoord::to_string()
	{
		return "d^2/(ds^" + std::to_string(direction + 1) + " ds^" + std::to_string(direction2 + 1) + ") of BASIS of " + space->get_name();
	}

	std::string D2XBasisFunctionLocalCoord::get_shape_buffer_base() const
	{
		return basis_buffer_base("d2S_shape", space->get_shape_name());
	}

	// Name of, and index into, the shape-function sensitivity arrays w.r.t. the nodal coordinates
	// (the "COORDDIFF" family), for a first OR a second Eulerian derivative of a basis function.
	// The second-derivative arrays are addressed by the packed PYOOMPH_D2_SLOT(inner,outer) rather
	// than by a single direction, and live in their own d_d2x_/d2_d2x2_ arrays.
	static std::string dcoord_shape_array(const BasisFunction *bf, bool second_coord_derivative)
	{
		const bool is_d2 = (bf->deriv_order() == 2);
		std::string base = second_coord_derivative ? (is_d2 ? "d2_d2x2_shape_dcoord" : "d2_dx2_shape_dcoord")
												   : (is_d2 ? "d_d2x_shape_dcoord" : "d_dx_shape_dcoord");
		const std::string sn = bf->get_space()->get_shape_name();
		if (sn == "C2TB" || sn == "C2" || sn == "C1TB" || sn == "C1")
			return base + "[SPACE_INDEX_" + sn + "]";
		return base + "_" + sn;
	}

	static std::string dcoord_shape_index(BasisFunction *bf)
	{
		if (D2XBasisFunction *d2 = dynamic_cast<D2XBasisFunction *>(bf))
			return std::to_string(d2->get_slot());
		return std::to_string(dynamic_cast<D1XBasisFunction *>(bf)->get_direction());
	}



	// The following get_eqn_number_str/get_num_nodes_str overrides build the C expression used to
	// access, respectively, the local equation-number lookup array and the node count for a given
	// FiniteElementSpace within the generated element code: ordinary nodal spaces use the shared
	// "nodal_local_eqn"/"nnode_of_space[...]" oomph-lib element members, while the position space
	// (PositionFiniteElementSpace) uses the dedicated "pos_local_eqn"/"nnode", and the discontinuous
	// D0 space (one DoF per element, not per node) always reports a single "node".
	unsigned FiniteElementSpace::next_creation_index = 0;

	std::string FiniteElementSpace::get_eqn_number_str(FiniteElementCode *forcode) const
	{
		const std::string access = forcode->get_elem_info_str(this) + "->nodal_local_eqn";
		const std::string name = "_" + forcode->get_owner_prefix(this) + "leqn";
		return forcode->alias_buffer_access(access, name, "int * const * const " + name);
	}

	std::string FiniteElementSpace::get_num_nodes_str(FiniteElementCode *forcode) const
	{
		const std::string eleminfo = forcode->get_elem_info_str(this);
		// Aliased like the buffers, and for the same reason: this is a LOOP BOUND, re-read on every
		// iteration of every trial loop because the jacobian[] store in the body may alias eleminfo.
		std::string access, name;
		if (this->get_shape_name() == "DL") // TODO: Make this in another way
		{
			access = eleminfo + "->nnode_DL";
			name = "_" + forcode->get_owner_prefix(this) + "nnode_DL";
		}
		else
		{
			access = eleminfo + "->nnode_of_space[SPACE_INDEX_" + this->get_shape_name() + "]";
			name = "_" + forcode->get_owner_prefix(this) + "nnode_" + this->get_shape_name();
		}
		return forcode->alias_buffer_access(access, name, "const unsigned " + name);
	}

	std::string D0FiniteElementSpace::get_num_nodes_str(FiniteElementCode *) const
	{
		return "1";
	}

	std::string PositionFiniteElementSpace::get_num_nodes_str(FiniteElementCode *forcode) const
	{
		const std::string access = forcode->get_elem_info_str(this) + "->nnode";
		const std::string name = "_" + forcode->get_owner_prefix(this) + "nnode";
		return forcode->alias_buffer_access(access, name, "const unsigned " + name);
	}

	std::string PositionFiniteElementSpace::get_eqn_number_str(FiniteElementCode *forcode) const
	{
		const std::string access = forcode->get_elem_info_str(this) + "->pos_local_eqn";
		const std::string name = "_" + forcode->get_owner_prefix(this) + "posleqn";
		return forcode->alias_buffer_access(access, name, "int * const * const " + name);
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
					compute_lines.push_back(indent + "    " + varname + "[l_shape] += " + alias_timestepper_weights(for_code, "dt", timedisc_scheme) + "[tindex]*" + for_code->alias_nodal_data(s.basis->get_space()) + "[l_shape][" + nodalindex + "][tindex];");
				}
				else if (s.dt_order == 2)
				{
					//os << indent << "    " << varname << "[l_shape] += " <<   "shapeinfo->timestepper_weights_d2t_" << timedisc_scheme << "[tindex]*" << eleminfo << "->" << nds << "[l_shape][" << nodalindex << "][tindex];" << std::endl;
					compute_lines.push_back(indent + "    " + varname + "[l_shape] += " + alias_timestepper_weights(for_code, "d2t", timedisc_scheme) + "[tindex]*" + for_code->alias_nodal_data(s.basis->get_space()) + "[l_shape][" + nodalindex + "][tindex];");
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
				if (s.basis->is_eulerian_deriv())
				{
					required_coorddiffs.insert(s);
					// The rank-4 nodal-coordinate sensitivity of this space is about to be READ below.
					// Marking the flag from this very set is what keeps the fill and the reads in step -
					// deriving it a second time from the residual would be another predicate to keep
					// synchronised by hand, which is how the Hessian flags went wrong (see 9.4.15).
					if (!for_code->current_shapeflag_func_type.empty())
						for_code->mark_shapes_required(for_code->current_shapeflag_func_type, const_cast<FiniteElementSpace *>(s.basis->get_space()), "dx_psi_dcoord");
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
				std::string dtstring = "d" + std::to_string(s.dt_order) + "t" + std::to_string(s.time_history_index) + s.get_history_geometry_suffix();
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
				std::string dtstring = "d" + std::to_string(s.dt_order) + "t" + std::to_string(s.time_history_index) + s.get_history_geometry_suffix();
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
				std::string dtstring = "d" + std::to_string(s.dt_order) + "t" + std::to_string(s.time_history_index) + s.get_history_geometry_suffix();
				if (s.dt_order > 0)
					dtstring += s.dt_scheme;
				for (unsigned int i = 0; i < for_code->nodal_dimension(); i++)
				{
					std::string code_type = for_code->get_owner_prefix(s.basis->get_space());
					std::string coorddiffname = code_type + "intrp_" + dtstring + "_" + s.basis->get_dx_str() + "_COORDDIFF_" + std::to_string(i) + "_" + s.field->get_name();
					std::string nodal_data = s.get_nodal_data_string(for_code, "l_shape");
					std::string shapename = dcoord_shape_array(s.basis, false);
					std::string shapestr = for_code->get_shape_info_str(s.basis->get_space()) + "->" + shapename + "[l_shape][" + dcoord_shape_index(s.basis) + "][m][" + std::to_string(i) + "]";
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
					std::string dtstring = "d" + std::to_string(s.dt_order) + "t" + std::to_string(s.time_history_index) + s.get_history_geometry_suffix();
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
					//						std::string dtstring = "d" + std::to_string(s.dt_order) + "t" + std::to_string(s.time_history_index) + s.get_history_geometry_suffix();
					//						if (s.dt_order > 0)
					//							dtstring += s.dt_scheme;
					for (unsigned int i = 0; i < for_code->nodal_dimension(); i++)
					{
						for (unsigned int j = i; j < for_code->nodal_dimension(); j++) // TODO: Symmetrize? Go from j=i?
						{
							std::string code_type = for_code->get_owner_prefix(s.basis->get_space());
							std::string coorddiffname = code_type + "intrp_" + dtstring + "_" + s.basis->get_dx_str() + "_2ndCOORDDIFF_" + std::to_string(i) + "_" + std::to_string(j) + "_" + s.field->get_name();
							std::string nodal_data = s.get_nodal_data_string(for_code, "l_shape");
							std::string shapename = dcoord_shape_array(s.basis, true);
							std::string shapestr = for_code->get_shape_info_str(s.basis->get_space()) + "->" + shapename + "[l_shape][" + dcoord_shape_index(s.basis) + "][m][" + std::to_string(i) + "][m2][" + std::to_string(j) + "]";
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

	// ---------------------------------------------------------------------------------------------
	// Exact hoisting of a Jacobian entry out of the l_shape loop. See dev_docs/code_generation.md 9.4.
	//
	// A Jacobian entry is a Frechet derivative, hence EXACTLY linear in the quantities carrying the
	// trial index l_shape: the trial shape functions, and on a moving mesh every geometric sensitivity,
	// which are indexed by l_shape too. Its coefficients therefore do not depend on l_shape - yet the
	// emitted C recomputes the whole entry nnode times per (integration point, l_test). Naming each
	// coefficient once above the loop is not a heuristic CSE (contrast 8.3, which was reverted): it is
	// that linearity, used. nnode*|entry| becomes |entry| + nnode*K for K atoms, and K is 3-4 where
	// nnode is 27 on the 3D solid element that motivated this.
	//
	// If the split fails for any reason the entry is emitted unchanged, so a case the mathematics did
	// not anticipate costs performance, never correctness.
	static const bool __hoist_disabled = getenv("PYOOMPH_DISABLE_JACOBIAN_HOIST") != NULL;

	// Coefficients below this many expression nodes are left inline rather than named. The default is 1,
	// i.e. hoist everything: naming a coefficient costs one value live across the trial loop, and once
	// the assembly function is split by its flag (9.4.6) that value lives in a body the residual-only
	// call never enters, so it costs nothing there. Measured on the split build, 3D solid element:
	// 1 -> -63.7%, 8 -> -63.2%, 32 -> -62.6%, 64 -> -48.5%, with residual-only flat at -8..-10%
	// throughout and 9 kB of extra C between the extremes.
	//
	// The knob is kept because the answer inverts without the split: with PYOOMPH_DISABLE_RJM_SPLIT the
	// coefficients DO sit in the residual-only path, hoisting cheap ones slowed it 3.6x, and 32 was the
	// right cut (9.4.5).
	static const unsigned __hoist_min_cost = getenv("PYOOMPH_JACOBIAN_HOIST_MIN")
												? (unsigned)atoi(getenv("PYOOMPH_JACOBIAN_HOIST_MIN"))
												: 1;

	// Leaves count 1 as well: a leaf here is a shape-function or interpolated-value array read, and on
	// this weak form the loads are what dominate (the same correction 8.3 had to make).
	static unsigned hoist_expr_cost(const GiNaC::ex &e)
	{
		unsigned c = 1;
		for (size_t i = 0; i < e.nops(); i++)
			c += hoist_expr_cost(e.op(i));
		return c;
	}

	// The quantities indexed by the trial index. A plain TestFunction is indexed by l_test and a plain
	// (non-derived) ShapeExpansion is an interpolated value, so neither qualifies; but a test function
	// or an interpolated value differentiated by a nodal COORDINATE does, because its C array carries
	// the coordinate node as an index (see dcoord_shape_array and the _COORDDIFF_ arrays).
	static bool is_l_shape_atom(const GiNaC::ex &e)
	{
		if (GiNaC::is_a<GiNaC::GiNaCShapeExpansion>(e))
		{
			const pyoomph::ShapeExpansion &sp = GiNaC::ex_to<GiNaC::GiNaCShapeExpansion>(e).get_struct();
			return sp.is_derived || sp.nodal_coord_dir >= 0 || sp.nodal_coord_dir2 >= 0;
		}
		if (GiNaC::is_a<GiNaC::GiNaCTestFunction>(e))
		{
			const pyoomph::TestFunction &tf = GiNaC::ex_to<GiNaC::GiNaCTestFunction>(e).get_struct();
			return tf.nodal_coord_dir >= 0 || tf.nodal_coord_dir2 >= 0;
		}
		if (GiNaC::is_a<GiNaC::GiNaCSpatialIntegralSymbol>(e))
			return GiNaC::ex_to<GiNaC::GiNaCSpatialIntegralSymbol>(e).get_struct().is_derived();
		if (GiNaC::is_a<GiNaC::GiNaCElementSizeSymbol>(e))
			return GiNaC::ex_to<GiNaC::GiNaCElementSizeSymbol>(e).get_struct().is_derived();
		if (GiNaC::is_a<GiNaC::GiNaCNormalSymbol>(e))
			return GiNaC::ex_to<GiNaC::GiNaCNormalSymbol>(e).get_struct().get_derived_direction() >= 0;
		return false;
	}

	// Which trial index an atom carries, as BITS: 1 = l_shape, 2 = l_shape2, 3 = both, 0 = not an atom.
	// Test with `cls & 2` for "carries l_shape2" - `2 | 3` is just 3 and matches everything, which is a
	// mistake worth making only once. Only the
	// Hessian has two indices; there, an atom carries BOTH exactly when it is a second derivative, and
	// otherwise carries l_shape2 iff its "other index" flag is set. Kept in step with the print sites -
	// GiNaCShapeExpansion::print, GiNaCTestFunction::print, and the SpatialIntegral/ElementSize/Normal
	// printers - which is where the actual [l_shape]/[l_shape2] subscripts are chosen.
	static unsigned l_shape_atom_class(const GiNaC::ex &e)
	{
		if (!is_l_shape_atom(e))
			return 0;
		bool second = false, by_lshape2 = false;
		if (GiNaC::is_a<GiNaC::GiNaCShapeExpansion>(e))
		{
			const pyoomph::ShapeExpansion &sp = GiNaC::ex_to<GiNaC::GiNaCShapeExpansion>(e).get_struct();
			// A DERIVED shape expansion carrying a nodal-coordinate direction prints as
			// d_dx_shape_dcoord[l_shape2][dir][l_shape][cdir] - the trial function of one index
			// differentiated by the nodal coordinate of the other - so it carries BOTH. The
			// second-coordinate-derivative forms (2ndCOORDDIFF, d2_dx2_shape_dcoord) do too.
			second = (sp.nodal_coord_dir2 >= 0) || (sp.is_derived && sp.nodal_coord_dir >= 0);
			by_lshape2 = sp.is_derived_other_index;
		}
		else if (GiNaC::is_a<GiNaC::GiNaCTestFunction>(e))
		{
			const pyoomph::TestFunction &tf = GiNaC::ex_to<GiNaC::GiNaCTestFunction>(e).get_struct();
			second = (tf.nodal_coord_dir2 >= 0);
			by_lshape2 = tf.is_derived_other_index;
		}
		else if (GiNaC::is_a<GiNaC::GiNaCSpatialIntegralSymbol>(e))
		{
			const auto &si = GiNaC::ex_to<GiNaC::GiNaCSpatialIntegralSymbol>(e).get_struct();
			second = si.is_derived2();
			by_lshape2 = si.is_derived_by_lshape2();
		}
		else if (GiNaC::is_a<GiNaC::GiNaCElementSizeSymbol>(e))
		{
			const auto &es = GiNaC::ex_to<GiNaC::GiNaCElementSizeSymbol>(e).get_struct();
			second = es.is_derived2();
			by_lshape2 = es.is_derived_by_lshape2();
		}
		else if (GiNaC::is_a<GiNaC::GiNaCNormalSymbol>(e))
		{
			const auto &ns = GiNaC::ex_to<GiNaC::GiNaCNormalSymbol>(e).get_struct();
			second = (ns.get_derived_direction2() >= 0);
			by_lshape2 = ns.is_derived_by_lshape2();
		}
		if (second)
			return 3;
		return by_lshape2 ? 2 : 1;
	}

	// Atoms whose class shares any bit with `mask`, in first-appearance order.
	static void collect_atoms_of_class(const GiNaC::ex &e, unsigned mask, std::vector<GiNaC::ex> &out)
	{
		const unsigned cls = l_shape_atom_class(e);
		if (cls)
		{
			if (cls & mask)
			{
				for (const auto &a : out)
					if (a.is_equal(e))
						return;
				out.push_back(e);
			}
			return;
		}
		for (size_t i = 0; i < e.nops(); i++)
			collect_atoms_of_class(e.op(i), mask, out);
	}

	static void collect_l_shape_atoms(const GiNaC::ex &e, std::vector<GiNaC::ex> &out)
	{
		if (is_l_shape_atom(e))
		{
			for (const auto &a : out)
				if (a.is_equal(e))
					return;
			out.push_back(e);
			return;
		}
		for (size_t i = 0; i < e.nops(); i++)
			collect_l_shape_atoms(e.op(i), out);
	}

	// Accumulate `factor * (linear form of t)` into acc[]. Structural rather than via expand(): the
	// entries reach hundreds of kilobytes and expanding them is exactly the cost we are removing.
	static const char *__hoist_fail_reason = "";
	static GiNaC::ex __hoist_fail_at;

	static bool accumulate_linear_in(const GiNaC::ex &t, const GiNaC::ex &factor,
									 const std::vector<GiNaC::symbol> &syms, std::vector<GiNaC::ex> &acc)
	{
		auto has_any = [&syms](const GiNaC::ex &x)
		{
			for (const auto &sy : syms)
				if (x.has(sy))
					return true;
			return false;
		};
		if (GiNaC::is_a<GiNaC::add>(t))
		{
			for (size_t i = 0; i < t.nops(); i++)
			{
				// An additive term free of every atom would be a Jacobian contribution that survives a
				// vanishing trial function. That cannot happen, so treat it as a failure rather than
				// silently dropping it.
				if (!has_any(t.op(i)))
				{
					__hoist_fail_reason = "additive term with no atom";
					__hoist_fail_at = t.op(i);
					return false;
				}
				if (!accumulate_linear_in(t.op(i), factor, syms, acc))
					return false;
			}
			return true;
		}
		if (GiNaC::is_a<GiNaC::mul>(t))
		{
			GiNaC::ex rest = 1, core;
			unsigned ncore = 0;
			for (size_t i = 0; i < t.nops(); i++)
			{
				if (has_any(t.op(i)))
				{
					core = t.op(i);
					ncore++;
				}
				else
					rest *= t.op(i);
			}
			if (ncore != 1)
			{
				__hoist_fail_reason = (ncore == 0 ? "product with no atom" : "product of two or more atoms");
				__hoist_fail_at = t;
				return false; // quadratic in the atoms, so not a first derivative
			}
			return accumulate_linear_in(core, factor * rest, syms, acc);
		}
		for (size_t k = 0; k < syms.size(); k++)
			if (t.is_equal(syms[k]))
			{
				acc[k] += factor;
				return true;
			}
		__hoist_fail_reason = "atom inside a power or a function";
		__hoist_fail_at = t;
		return false; // a power of an atom, or an atom inside a function: not linear
	}

	// entry == sum_k coeffs[k] * atoms[k], or false and nothing usable.
	static bool split_entry_by_l_shape(const GiNaC::ex &entry, const std::vector<GiNaC::ex> &atoms,
									   std::vector<GiNaC::ex> &coeffs)
	{
		std::vector<GiNaC::symbol> syms;
		GiNaC::exmap m;
		for (size_t k = 0; k < atoms.size(); k++)
		{
			syms.push_back(GiNaC::symbol("__hoist_atom_" + std::to_string(k)));
			m[atoms[k]] = syms.back();
		}
		const GiNaC::ex sub = entry.subs(m, GiNaC::subs_options::no_pattern);
		std::vector<GiNaC::ex> acc(atoms.size(), GiNaC::ex(0));
		if (!accumulate_linear_in(sub, GiNaC::ex(1), syms, acc))
			return false;
		// The coefficients must be free of the atoms, or the loop body would be wrong.
		for (const auto &c : acc)
			for (const auto &sy : syms)
				if (c.has(sy))
					return false;
		coeffs = acc;
		return true;
	}

	// The Hessian counterpart. A Hessian entry is a SECOND derivative, hence bilinear in the two trial
	// indices: linear in the l_shape2-carrying atoms with coefficients that still depend on l_shape.
	// Naming those coefficients just above the inner loop turns nnode*nnode2 evaluations of the whole
	// entry into nnode evaluations plus nnode*nnode2 multiply-adds. Atoms carrying BOTH indices (the
	// second-derivative family: int_pt_weights_d2_coords, d2_normal_d2coord, the 2ndCOORDDIFF arrays,
	// d2_dx2_shape_dcoord) belong to the inner set too - the entry is linear in them as well, and their
	// coefficients carry no index at all.
	//
	// A second split of those coefficients by the l_shape-carrying atoms would let the constants move
	// above the OUTER loop as well; that is left for later, because the outer loop is opened lazily and
	// shared between field pairs, and this level already removes the nnode2 factor.
	static std::string hoist_hessian_entry(const GiNaC::ex &entry, std::ostream &pre,
										   const std::string &indent, GiNaC::print_FEM_options &csrc_opts)
	{
		static const bool dbg = getenv("PYOOMPH_DEBUG_HOIST") != NULL;
		if (__hoist_disabled || entry.is_zero())
			return "";
		std::vector<GiNaC::ex> atoms;
		collect_atoms_of_class(entry, 2, atoms); // bit 1: everything carrying l_shape2, i.e. classes 2 and 3
		if (atoms.empty())
		{
			if (dbg)
			{
				std::vector<GiNaC::ex> all;
				collect_atoms_of_class(entry, 3, all);
				std::cerr << "HOIST-HESS: no l_shape2 atom (" << all.size() << " atoms of any class)" << std::endl;
				for (auto &a : all)
					std::cerr << "    class " << l_shape_atom_class(a) << " : " << a << std::endl;
			}
			return "";
		}
		std::vector<GiNaC::ex> coeffs;
		if (!split_entry_by_l_shape(entry, atoms, coeffs))
		{
			if (dbg)
			{
				std::cerr << "HOIST-HESS: split failed (" << __hoist_fail_reason << ") at: " << __hoist_fail_at << std::endl;
				for (size_t k = 0; k < atoms.size(); k++)
					std::cerr << "    atom " << k << " class " << l_shape_atom_class(atoms[k]) << " : " << atoms[k] << std::endl;
			}
			return "";
		}
		std::ostringstream body;
		bool any = false;
		const unsigned min_cost = (csrc_opts.for_code && csrc_opts.for_code->jacobian_hoist_min_cost >= 0)
									  ? (unsigned)csrc_opts.for_code->jacobian_hoist_min_cost
									  : __hoist_min_cost;
		for (size_t k = 0; k < atoms.size(); k++)
		{
			if (coeffs[k].is_zero())
				continue;
			if (!body.str().empty())
				body << "+";
			if (hoist_expr_cost(coeffs[k]) >= min_cost)
			{
				const std::string nm = "_hc" + std::to_string(csrc_opts.for_code->hoist_coeff_counter++);
				pre << indent << "const double " << nm << " = ";
				print_simplest_form(coeffs[k], pre, csrc_opts);
				pre << ";" << std::endl;
				body << nm;
				any = true;
			}
			else
			{
				body << "(";
				print_simplest_form(coeffs[k], body, csrc_opts);
				body << ")";
			}
			body << "*(";
			print_simplest_form(atoms[k], body, csrc_opts);
			body << ")";
		}
		if (!any)
			return "";
		return body.str().empty() ? std::string("0.0") : body.str();
	}

	// Emits `double _jcN = <coeff>;` lines into `pre` and returns the loop-body expression, or "" when
	// the entry could not be split and must be printed as it always was.
	// Coefficients are named one per surviving atom, independently per entry, so the same coefficient
	// can be emitted several times into the same `pre` stream - the Jacobian half and the mass half of
	// one field share that stream, and so do all the fields of one trial space. Across the tutorial
	// corpus 179 of 639 coefficients were an exact duplicate of a sibling in the same BEGIN_JACOBIAN()
	// block. GCC's GCSE removes them, so this is not about run time under the system compiler: it is
	// generated bytes and JIT compile time, and it IS run time under tcc, which has no CSE at all.
	// Keyed by the PRINTED text rather than by GiNaC structure - that is what "duplicate" means for the
	// emitted C, and it costs no expression comparisons.
	static std::string hoist_jacobian_entry(const GiNaC::ex &entry, std::ostream &pre,
											const std::string &indent, GiNaC::print_FEM_options &csrc_opts,
											std::map<std::string, std::string> &seen)
	{
		if (__hoist_disabled || entry.is_zero())
			return "";
		std::vector<GiNaC::ex> atoms;
		collect_l_shape_atoms(entry, atoms);
		if (atoms.empty())
			return "";
		std::vector<GiNaC::ex> coeffs;
		if (!split_entry_by_l_shape(entry, atoms, coeffs))
			return "";
		std::ostringstream body;
		bool any_hoisted = false;
		const unsigned min_cost = (csrc_opts.for_code && csrc_opts.for_code->jacobian_hoist_min_cost >= 0)
									  ? (unsigned)csrc_opts.for_code->jacobian_hoist_min_cost
									  : __hoist_min_cost;
		for (size_t k = 0; k < atoms.size(); k++)
		{
			if (coeffs[k].is_zero())
				continue;
			if (!body.str().empty())
				body << "+";
			if (hoist_expr_cost(coeffs[k]) >= min_cost)
			{
				std::ostringstream rhs;
				print_simplest_form(coeffs[k], rhs, csrc_opts);
				const std::string rhs_str = rhs.str();
				std::string nm;
				auto it = seen.find(rhs_str);
				if (it != seen.end())
				{
					nm = it->second; // already named in this same pre-stream
				}
				else
				{
					nm = "_jc" + std::to_string(csrc_opts.for_code->hoist_coeff_counter++);
					pre << indent << "const double " << nm << " = " << rhs_str << ";" << std::endl;
					seen[rhs_str] = nm;
				}
				body << nm;
				any_hoisted = true;
			}
			else
			{
				body << "(";
				print_simplest_form(coeffs[k], body, csrc_opts);
				body << ")";
			}
			body << "*(";
			print_simplest_form(atoms[k], body, csrc_opts);
			body << ")";
		}
		// Nothing was worth naming: leave the entry exactly as it was rather than emit a re-associated
		// copy of it for no gain.
		if (!any_hoisted)
			return "";
		return body.str().empty() ? std::string("0.0") : body.str();
	}

	// The nodal position space only carries real degrees of freedom (and thus needs
	// Jacobian/Hessian rows/columns of its own) when the mesh coordinates are themselves unknowns
	// being solved for (moving-mesh/ALE problems with coordinates_as_dofs); otherwise positions are
	// prescribed data and there is nothing to differentiate w.r.t. them.
	bool PositionFiniteElementSpace::write_generic_Hessian_contribution(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, GiNaC::ex for_what, bool hanging_eqns, FiniteElementField *residual_field)
	{
		// Only do it if the coordinates are Dofs
		if (for_code->coordinates_as_dofs)
			return FiniteElementSpace::write_generic_Hessian_contribution(for_code, os, indent, for_what, hanging_eqns, residual_field);
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
	//   4. Convert all remaining subexpression(...) markers to structs (via se_to_struct_hessian) and
	//      emit nested loops over the outer node index l_shape and inner node index l_shape2 (using the
	//      BEGIN_HESSIAN_SHAPE_LOOP1[_CONTINUOUS_SPACE] macros for hanging-node-aware assembly), writing
	//      the C statement that adds the Hessian entry into the sparse Hessian tensor.
	// Global state (__derive_shapes_by_second_index, all_Hessian_shapeexps/testfuncs/indices_required,
	// __derive_only_by_expansion_mode) is toggled around the differentiation calls so that the custom
	// GiNaC structures' derivative() implementations know which index of the double loop is being
	// derived, and so the caller can later learn which shape expansions/spaces/fields must have their
	// interpolation code emitted to support this Hessian.
	bool FiniteElementSpace::write_generic_Hessian_contribution(FiniteElementCode *for_code, std::ostream &os, const std::string &indent, GiNaC::ex for_what, bool hanging_eqns, FiniteElementField *residual_field)
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

		std::set<FiniteElementField *,FiniteElementFieldPtrLess> jacobian_fields;
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
			for_code->all_Hessian_indices_required.insert(f);
			bool loop2_written = false;

			__derive_only_by_expansion_mode=for_code->get_derive_jacobian_by_expansion_mode();
			__ignore_dpsi_coord_diffs_in_jacobian=for_code->ignore_dpsi_coord_diffs_in_jacobian();						
			GiNaC::ex diffpart = GiNaC::diff(for_what, f->get_symbol());
			__derive_only_by_expansion_mode=NULL;
			__ignore_dpsi_coord_diffs_in_jacobian=false;

			for_code->subexpressions = for_code->se_to_struct_hessian->subexpressions;
			//			std::cout << "HESSIAN  CONTRIBU " << for_what << std::endl;
			//			std::cout << "DHESSIAN  CONTRIBU " << diffpart << std::endl;
			// Same split as in write_generic_RJM_jacobian_contribution: take the mass half off the
			// derivative first, then strip the probe, so the Hessian half never carries it.
			GiNaC::ex masspart = strip_mass_matrix_marker(GiNaC::diff(diffpart, pyoomph::expressions::__partial_t_mass_matrix));
			diffpart = strip_mass_matrix_marker(diffpart);
			if (diffpart.is_zero() && masspart.is_zero())
			{
				for_code->Hessian_symmetric_fields_completed.insert(f);
				continue;
			}
			if (pyoomph_verbose)
				std::cout << "DIFF PART IS " << diffpart << std::endl;
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
			{
				// The mass half can depend on fields the Hessian half does not (a pure mass
				// contribution has an empty Hessian half altogether), so its second derivative has to
				// be taken with respect to those fields as well.
				std::set<ShapeExpansion> mass_shapes = for_code->get_all_shape_expansions_in(masspart);
				hessian_shapes.insert(mass_shapes.begin(), mass_shapes.end());
			}
			std::set<FiniteElementSpace *,FiniteElementSpacePtrLess> hessian_spaces;

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
				for_code->all_Hessian_indices_required.insert(s2.field);
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
				std::set<FiniteElementField *,FiniteElementFieldPtrLess> hessian_fields;
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
					// Not an AmbientCodegenScope, unlike the other flags: the true-region here spans a
					// loop body with a `continue` in the middle and three separate clears at the loop
					// exits below, and the two differentiations that need it are separated by the
					// symmetry-skip branch. Restructuring the double loop to scope it precisely is a
					// change to the most intricate part of the Hessian generator for no gain against
					// either of the two reasons the other flags were scoped (nesting, and throwing
					// mid-differentiation) - this one is set and cleared inside a single pass.
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
					for_code->subexpressions = for_code->se_to_struct_hessian->subexpressions;
					// The subexpression shape expansions used to be harvested into the Hessian-wide sets
					// here. They are not any more: this point is skipped by the assemble_hessian_by_symmetry
					// "continue" above, and it runs before the mappings further down, so the last (f,f2)
					// pair's subexpressions were never seen. write_generic_Hessian() now sweeps the
					// finished list once instead.
					__derive_shapes_by_second_index = false;
					if (diffpart2.is_zero() && masspart2.is_zero()) // &&  masspart2.is_zero()
						continue;

					// Record the second-derivative coupling. BOTH directions are marked: d2R/df df2 is
					// symmetric in (f,f2), and a Hessian contracted with a vector sums over one of the two
					// indices, so the surviving (row, column) pattern is the union over both.
					if (residual_field)
					{
						const unsigned ri = for_code->get_current_residual_index();
						residual_field->mark_hessian_contribution_for_code(for_code, ri, f);
						residual_field->mark_hessian_contribution_for_code(for_code, ri, f2);
					}

					auto shapeexps = for_code->get_all_shape_expansions_in(diffpart2);
					auto shapeexpsM = for_code->get_all_shape_expansions_in(masspart2);
					
					for (auto sexpa : shapeexps)
					{
						// The derived ones are excluded HERE only (they would collide as interpolation
						// variables), but they are still read by the emitted code - see
						// all_Hessian_shapeexps_for_shapeflags.
						if ((!sexpa.is_derived && !sexpa.is_derived_other_index) || sexpa.nodal_coord_dir != -1 || sexpa.nodal_coord_dir2 != -1)
							for_code->all_Hessian_shapeexps.insert(sexpa);
						for_code->all_Hessian_shapeexps_for_shapeflags.insert(sexpa);
						for_code->all_Hessian_indices_required.insert(sexpa.field);
					}
					for (auto sexpa : shapeexpsM)
					{
						if ((!sexpa.is_derived && !sexpa.is_derived_other_index) || sexpa.nodal_coord_dir != -1 || sexpa.nodal_coord_dir2 != -1)
							for_code->all_Hessian_shapeexps.insert(sexpa);
						for_code->all_Hessian_shapeexps_for_shapeflags.insert(sexpa);
						for_code->all_Hessian_indices_required.insert(sexpa.field);
					}
					//		  		   all_Hessian_shapeexps.insert(shapeexps.begin(),shapeexps.end());
					auto testfuncs = for_code->get_all_test_functions_in(diffpart2);
					for_code->all_Hessian_testfuncs.insert(testfuncs.begin(), testfuncs.end());
					auto testfuncsM = for_code->get_all_test_functions_in(masspart2);
					for_code->all_Hessian_testfuncs.insert(testfuncsM.begin(), testfuncsM.end());

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

					// The entry is built BEFORE the inner loop header is written, so that the coefficients
					// hoisted out of it (9.4.8) can be emitted above the loop. The subexpression pass of
					// 9.3 runs first either way; only the emission order moves.
					//	 		                        std::cout << "DIFFPART2 " << diffpart2 << std::endl;
					GiNaC::ex diffpart2_se = (*for_code->se_to_struct_hessian)(diffpart2);
					//						 		                        std::cout << "DIFFPART2 SE" << diffpart2_se << std::endl;
					for_code->subexpressions = for_code->se_to_struct_hessian->subexpressions;
					std::ostringstream hess_pre;
					const std::string hess_hoisted =
						(only_mass_part || l_shape2 != "l_shape2")
							? std::string("")
							: hoist_hessian_entry(diffpart2_se, hess_pre, indent + "     ", csrc_opts);
					os << hess_pre.str();

					if (numnodes_str2 != "1")
					{
						os << indent << "     for (unsigned int l_shape2=0;l_shape2<" << numnodes_str2 << ";l_shape2++)" << std::endl;
						os << indent << "     {" << std::endl;
					}

					//		   os << indent << "   //HESSIAN SHAPE CONTRIB: " << f2->get_nodal_index_str(for_code) << ": " << diffpart2 ;

					os << std::endl;
					if (hang2)
					{
						os << indent << "        BEGIN_HESSIAN_SHAPE_LOOP2_CONTINUOUS_SPACE(" << eqn_index2 << ",";
						if (only_mass_part)
							os << "0";
						else if (!hess_hoisted.empty())
							os << hess_hoisted;
						else
							print_simplest_form(diffpart2_se, os, csrc_opts);
						os << "," << hang_info2 << "," << l_shape2 << ")" << std::endl;
					}
					else
					{
						os << indent << "        BEGIN_HESSIAN_SHAPE_LOOP2(" << eqn_index2 << ", ";
						if (only_mass_part)
							os << "0";
						else if (!hess_hoisted.empty())
							os << hess_hoisted;
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
					for_code->subexpressions = for_code->se_to_struct_hessian->subexpressions;
					// std::cout << "CHECKING MASS PART " << (masspart2-GiNaC::diff(diffpart2,pyoomph::expressions::__partial_t_mass_matrix)) << std::endl;
					// std::cout << " MA " << masspart2 << std::endl;
					// std::cout << " MB " << GiNaC::diff(diffpart2,pyoomph::expressions::__partial_t_mass_matrix) << std::endl;
					//				__derive_shapes_by_second_index = true;
					//					GiNaC::ex masspart2=GiNaC::diff(diffpart2,pyoomph::expressions::__partial_t_mass_matrix);
					//					__derive_shapes_by_second_index = false;
					if (!masspart2.is_zero())
					{
						GiNaC::ex mass_part_se = (*for_code->se_to_struct_hessian)(masspart2);
						for_code->subexpressions = for_code->se_to_struct_hessian->subexpressions;
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
		__derive_shapes_by_second_index = false; // the loop above has three exits; this catches them all
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
		//std::set<FiniteElementField *,FiniteElementFieldPtrLess> jacobian_fields;
		
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
			l_shape = "l_shape";
		}

		// "" means "no hanging possible here"; otherwise the token to hand the hang macros as their
		// HANGON argument (a literal 1, or the constant _impl parameter when this routine is split).
		const std::string hang_on = this->get_hang_on_str(for_code);
		bool hang = !hang_on.empty();

		// The hoisted coefficients (9.4) have to be emitted BEFORE the loop, so the loop body is
		// buffered and the two streams are concatenated at the end. With hoisting off or unavailable
		// `pre` stays empty and the emitted code is exactly what it always was.
		std::ostringstream pre, body;
		// Shared by every entry written into `pre` (all fields of this trial space, Jacobian and mass
		// halves alike), so an identical coefficient is named once - see hoist_jacobian_entry.
		std::map<std::string, std::string> hoisted_coeffs;

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

			// Split the derivative into its mass-matrix and its Jacobian half before doing anything
			// else with it. The mass half is whatever multiplies the __partial_t_mass_matrix probe -
			// either because a partial_t shape expansion responded to it, or because the term was
			// flagged as a pure mass contribution by add_dweak_dt. Only the remainder is the Jacobian,
			// so the probe has to be stripped from diffpart; leaving it in would print an undefined
			// symbol and would mark Jacobian couplings that do not exist.
			GiNaC::ex mass_part = strip_mass_matrix_marker(GiNaC::diff(diffpart, pyoomph::expressions::__partial_t_mass_matrix));
			diffpart = strip_mass_matrix_marker(diffpart);

			if (diffpart.is_zero() && mass_part.is_zero())
				continue;
			if (pyoomph_verbose)
				std::cout << "DIFF PART IS " << diffpart << std::endl;
			std::string eqn_index = f->get_equation_str(for_code, l_shape);

			for_code->add_contributing_field(residual_field);
			for_code->add_contributing_field(f);
			// diffpart/mass_part ARE the elemental block expressions: the row role is carried by the
			// TestFunction of residual_field, the column role by the derived ShapeExpansions of f.
			// write_code_info() proves the JACOBIAN_BLOCK_* properties from them.
			if (for_code->record_jacobian_blocks_for_flags)
				for_code->record_jacobian_block_expr(residual_field, f, diffpart, mass_part);
			if (!diffpart.is_zero())
				residual_field->mark_jacobian_contribution_for_code(for_code,for_code->get_current_residual_index(),f);
			// Only worth doing where there is a loop to hoist out of: a single-DoF space indexes with
			// the literal "0" and pays nothing per trial node.
			const std::string hoisted = (l_shape == "l_shape")
											? hoist_jacobian_entry(diffpart, pre, indent, csrc_opts, hoisted_coeffs)
											: std::string("");
			if (hang)
			{
				std::string hang_info = f->get_hanginfo_str(for_code);
				body << indent << "  BEGIN_JACOBIAN_HANG(" << hang_on << ", " << eqn_index << ", ";
				if (hoisted.empty())
					print_simplest_form(diffpart, body, csrc_opts);
				else
					body << hoisted;
				body << "," << hang_info << "," << l_shape << ")" << std::endl;
			}
			else
			{
				body << indent << "  BEGIN_JACOBIAN_NOHANG(" << eqn_index << ", ";
				if (hoisted.empty())
					print_simplest_form(diffpart, body, csrc_opts);
				else
					body << hoisted;
				body << indent << ")" << std::endl;
			}
			if (!diffpart.is_zero()) // A pure mass contribution still needs the block, but adds no Jacobian
				body << indent << "    ADD_TO_JACOBIAN_" << (hanging_eqns ? "HANG" : "NOHANG") << "_" << (hang ? "HANG" : "NOHANG") << "()" << std::endl;
			// diffpart.evalf().print(GiNaC::print_csrc_FEM(os,&csrc_opts));
			// GiNaC::factor(GiNaC::normal(GiNaC::expand(GiNaC::expand(diffpart).evalf()))).print(GiNaC::print_csrc_FEM(os,&csrc_opts));

			//	    os <<")" <<std::endl;

			if (!mass_part.is_zero())
			{
				// Same bookkeeping as the Jacobian marking above, but for the mass-matrix half of this
				// contribution. "mass_part is not zero" is exactly "this (row field, column field) pair
				// can appear in the mass matrix", decided symbolically and therefore a superset of
				// whatever the numbers turn out to be - which is what a sparsity pattern must be. The
				// information was already computed here and discarded; recording it is what lets the
				// mass matrix get a tight, value-independent pattern of its own instead of either
				// inheriting the Jacobian's (3x too dense) or being value-filtered (not reusable).
				residual_field->mark_mass_matrix_contribution_for_code(for_code,for_code->get_current_residual_index(),f);
				for_code->emitted_mass_matrix_contribution = true;
				const std::string hoisted_mass = (l_shape == "l_shape")
													 ? hoist_jacobian_entry(mass_part, pre, indent, csrc_opts, hoisted_coeffs)
													 : std::string("");
				body << indent << "    ADD_TO_MASS_MATRIX_" << (hanging_eqns ? "HANG" : "NOHANG") << "_" << (hang ? "HANG" : "NOHANG") << "(";
				if (hoisted_mass.empty())
					print_simplest_form(mass_part, body, csrc_opts);
				else
					body << hoisted_mass;
				body << ")" << std::endl;
			}
			body << indent << "  END_JACOBIAN_" << (hang ? "HANG" : "NOHANG") << "()" << std::endl;
		}

		os << pre.str();
		if (numnodes_str != "1")
		{
			os << indent << "for (unsigned int l_shape=0;l_shape<" << numnodes_str << ";l_shape++)" << std::endl;
			os << indent << "{" << std::endl;
		}
		os << body.str();
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

			// Sourced from the function-scope buffer aliases rather than from shapeinfo directly, so
			// that a space block re-entered on every integration point does not re-walk the struct, and
			// so that the test and trial sides of the same buffer share one load. When the aliases are
			// switched off, alias_shape_buffer hands back the plain access and this is what it was.
			const bool indexed_space = (this->get_shape_name()=="C2TB" || this->get_shape_name()=="C2" || this->get_shape_name()=="C1TB" || this->get_shape_name()=="C1");
			const std::string sn = this->get_shape_name();
			auto ptr_decl = [](const std::string &n) { return "double const * " + n; };
			auto dx_decl = [](const std::string &n) { return "DX_SHAPE_FUNCTION_DECL(" + n + ")"; };
			auto d2x_decl = [](const std::string &n) { return "D2X_SHAPE_FUNCTION_DECL(" + n + ")"; };
			auto buf = [&](const std::string &array) { return indexed_space ? array + "s[SPACE_INDEX_" + sn + "]" : array + "_" + sn; };
			oss << indent << "  double const * testfunction = " << alias_shape_buffer(for_code, this, buf("shape"), "psi", -1, ptr_decl) << ";" << std::endl;
			oss << indent << "  DX_SHAPE_FUNCTION_DECL(dx_testfunction) = " << alias_shape_buffer(for_code, this, buf("dx_shape"), "dxpsi", 0, dx_decl) << ";" << std::endl;
			oss << indent << "  DX_SHAPE_FUNCTION_DECL(dX_testfunction) = " << alias_shape_buffer(for_code, this, buf("dX_shape"), "dXpsi", -1, dx_decl) << ";" << std::endl;
			oss << indent << "  DX_SHAPE_FUNCTION_DECL(dS_testfunction) = " << alias_shape_buffer(for_code, this, buf("dS_shape"), "dSpsi", -1, dx_decl) << ";" << std::endl;
			oss << indent << "  D2X_SHAPE_FUNCTION_DECL(d2x_testfunction) = " << alias_shape_buffer(for_code, this, buf("d2x_shape"), "d2xpsi", 0, d2x_decl) << ";" << std::endl;
			oss << indent << "  D2X_SHAPE_FUNCTION_DECL(d2S_testfunction) = " << alias_shape_buffer(for_code, this, buf("d2S_shape"), "d2Spsi", -1, d2x_decl) << ";" << std::endl;

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
			// var_part keeps the __partial_t_mass_matrix probe, because the Jacobian/mass split below
			// is derived from it; the residual itself only ever sees the part without the probe.
			GiNaC::ex res_part = strip_mass_matrix_marker(var_part);
			FiniteElementField *field = var_mapper.get_field();
			//if (hessian) std::cout << "HESSIAN TEST: " << test_name << std::endl;
			std::string eqn_index = field->get_equation_str(for_code, l_test);
			std::string hang_info = field->get_hanginfo_str(for_code);
			bool hessian_loop1_written = false;
			// "" means "no hanging possible here"; otherwise the token to hand the hang macros as
			// their HANGON argument. External/bulk spaces always hang - their hang buffers carry the
			// equation remap (see FiniteElementSpace::get_hang_on_str).
			const std::string hang_on = this->get_hang_on_str(for_code);
			bool can_have_hanging = !hang_on.empty();
			if (!hessian)
			{
				field->mark_residual_contribution_for_code(for_code,for_code->get_current_residual_index());
				//std::cout << "MARKING RESIDUAL CONTRIBUTION " << field->get_space()->get_code()->get_full_domain_name()+"/"+field->get_name() << " for " << for_code->get_full_domain_name()<< " PTR " << field << " FOR CODE " << for_code <<std::endl;
				for_code->add_contributing_field(field);
				has_contribs = true;
				if (can_have_hanging)
				{
					oss << indent << "    BEGIN_RESIDUAL_CONTINUOUS_SPACE(" << hang_on << ", " << eqn_index << ",";
					print_residual_entry(for_code, res_part, oss, csrc_opts);
					if (for_code->latex_printer)
					{
						std::map<std::string, std::string> latexinfo = {{"typ", "final_residual"}, {"test_name", test_name}};
						for_code->latex_printer->print(latexinfo, res_part, csrc_opts);
					}
					oss << ", " << hang_info << "," << l_test << ")" << std::endl;
					oss << indent << "      ADD_TO_RESIDUAL_CONTINUOUS_SPACE()" << std::endl;
				}
				else
				{
					oss << indent << "    BEGIN_RESIDUAL(" << eqn_index << ", ";
					print_residual_entry(for_code, res_part, oss, csrc_opts);
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
				//std::set<FiniteElementField *,FiniteElementFieldPtrLess> jacobian_fields;
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
						bool has_hessian = s->write_generic_Hessian_contribution(for_code, hessian_inner, indent + "        ", var_part, can_have_hanging, field);
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

	std::set<FiniteElementField *,FiniteElementFieldPtrLess> FiniteElementCode::get_fields_on_space(FiniteElementSpace *space)
	{
		std::set<FiniteElementField *,FiniteElementFieldPtrLess> res;
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
		// Order matters: D2XBasisFunction derives from D1XBasisFunction, so the second derivatives
		// have to be classified first or they would be marked as "dx_psi" and their buffer would
		// never be filled.
		std::string dx_type = "psi";
		if (bf->deriv_order() == 2)
		{
			// d2S_shapes rides on the same flag as d2x_shapes, exactly as dS_shapes does on dx_psi.
			dx_type = "d2x_psi";
		}
		else if (dynamic_cast<D1XBasisFunction *>(bf))
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
			required_shapes[func_type] = std::map<FiniteElementSpace *, std::map<std::string, bool>, FiniteElementSpacePtrLess>();
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

	// Three-way decision behind the hanging-node macros, see the declaration in codegen.hpp.
	//
	// The middle case is the specialisation: the hang macros are emitted, but with the constant
	// `_impl` parameter as their HANGON argument, so the same body compiles to a hanging and a
	// non-hanging entry point and elements_assembly.cpp picks per element.
	//
	// The FIRST case is a correctness trap and not an optimisation opportunity: for a space living on
	// another code (a bulk or opposite-side space accessed from an interface element),
	// fill_hang_info_with_equations reuses these very hang buffers as the local-equation REMAP channel
	// - it writes nummaster=1, weight=1.0 and the remapped local_eqn for every dof, hanging or not
	// (src/elements_hanging.cpp:1062-1099). With HANGON folded to 0 those entries would be skipped and
	// the element would scatter into the wrong rows/columns, so they stay unconditional.
	std::string FiniteElementSpace::get_hang_on_str(FiniteElementCode *for_code)
	{
		if (for_code != this->code)
			return "1"; // external/bulk/opposite space: the hang buffers carry the equation remap
		if (!this->can_have_hanging_nodes())
			return ""; // no hanging possible at all -> the plain non-hang macros
		if (!for_code->emitting_hang_parameter)
			return "1";
		for_code->hang_parameter_was_used = true;
		return "pyoomph_hang_on";
	}

	unsigned FiniteElementCode::next_creation_index = 0;

	// Constructs the built-in FiniteElementSpace hierarchy shared by every element (Pos, the
	// continuous C2TB/C2/C1TB/C1 spaces, their discontinuous-Galerkin D2TB/D2/D1TB/D1 counterparts,
	// the fully discontinuous DL/D0 spaces, and the external ED0 space), plus the symbolic
	// "derived" dx/element-size symbol families (dx_derived[i], dx_derived2[i][j], elemsize_derived,
	// elemsize_Cart_derived, ... and their Hessian "second index" variants) used to represent
	// spatial derivatives of the integration measure w.r.t. moving nodal coordinates - these are
	// pre-built for all 3 spatial directions upfront so that GiNaC::diff() on dx/element-size
	// symbols can return the correctly-tagged symbol without having to construct new ones on the fly.
	FiniteElementCode::FiniteElementCode() : creation_index(next_creation_index++), residual_index(0), residual_names({""}), equations(NULL), bulk_code(NULL), opposite_interface_code(NULL), residual(std::vector<GiNaC::ex>{0}), dx(this, false), dX(this, true), dx_unity(this, false), elemsize_Eulerian(this, false, true), elemsize_Lagrangian(this, true, true), elemsize_Eulerian_Cart(this, false, false), elemsize_Lagrangian_Cart(this, true, false), nodal_delta(this), stage(0), nodal_dim(0), lagr_dim(0), coordinate_sys(&__no_coordinate_system), _x(GiNaC::indexed(GiNaC::potential_real_symbol("interpolated_x"), GiNaC::idx(0, 3))),
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
		delete se_to_struct_hessian; // the last Hessian pass's mapper; leaked while it was a global
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
		unsigned __iters = 0;
		do
		{
			GiNaC::ex old = repl;
			if (pyoomph_verbose)
				std::cout << "EXPAND LOOP START (@CODE " << this << "): " << repl << std::endl;
			repl_dim_fields.repl_count = 0;
			{
				__phase_timer __t("  ph:mapper_pass");
				repl = repl_dim_fields(repl);
			}
			__iters++;
			if (pyoomph_verbose)
				std::cout << "EXPANDED " << repl_dim_fields.repl_count << " WITH RESULT: " << repl << std::endl;
			bool __stuck;
			{
				__phase_timer __t("  ph:stuck_check");
				__stuck = repl_dim_fields.repl_count && (old - repl).is_zero();
			}
			if (__stuck)
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
		if (__time_add_residual())
			std::cerr << "[add_residual]   ph:iterations " << __iters
					  << " mapper_calls " << __rfnd_calls
					  << " memo_hits " << __rfnd_hits
					  << " distinct " << __rfnd_distinct.size() << std::endl;

		// Finally, replace all mesh coordinates to normal coordinates
		// We just need this temporarily, since we want to be able to calculate partial_t(mesh_x), which is non-zero, whereas partial_t(coordinate_x) =0
		__phase_timer __t_m2x("  ph:MeshToCoordinateShapes");
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
	unsigned FiniteElementCode::get_Z2_group_index(const std::string &name,bool for_eigen)
	{
		auto &names = (for_eigen ? Z2_group_names_for_eigen : Z2_group_names);
		for (unsigned int i = 0; i < names.size(); i++)
			if (names[i] == name) return i;
		names.push_back(name);
		// New groups start at the historical settings: fully relative, unweighted. add_Z2_flux
		// overwrites them with what the caller asked for.
		(for_eigen ? Z2_group_normalize_relative_for_eigen : Z2_group_normalize_relative).push_back(1.0);
		(for_eigen ? Z2_group_weight_for_eigen : Z2_group_weight).push_back(1.0);
		return names.size() - 1;
	}

	void FiniteElementCode::add_Z2_flux(GiNaC::ex flux,bool for_eigen,const std::string &group,double normalize_relative,double weight)
	{
		if (stage > 1)
			throw_runtime_error("Cannot add error estimators any more");
		GiNaC::ex expanded = this->expand_placeholders(flux, "Z2Flux");

		const unsigned grp = this->get_Z2_group_index(group, for_eigen);
		// Every flux of one group has to agree about how that group is normalised - the norm being
		// divided out is a property of the group, not of the individual expression. Disagreement is
		// a mistake worth reporting rather than silently resolving to whichever was registered last.
		auto &normrel = (for_eigen ? Z2_group_normalize_relative_for_eigen : Z2_group_normalize_relative);
		auto &wgt = (for_eigen ? Z2_group_weight_for_eigen : Z2_group_weight);
		auto &fluxlist = (for_eigen ? Z2_fluxes_for_eigen : Z2_fluxes);
		auto &grouplist = (for_eigen ? Z2_flux_groups_for_eigen : Z2_flux_groups);
		const bool group_is_new = (std::count(grouplist.begin(), grouplist.end(), grp) == 0);
		if (!group_is_new && (normrel[grp] != normalize_relative || wgt[grp] != weight))
		{
			throw_runtime_error("Conflicting normalization for spatial error estimator group '" + group +
								"': got normalize_relative=" + std::to_string(normalize_relative) + ", weight=" + std::to_string(weight) +
								" but the group was already registered with normalize_relative=" + std::to_string(normrel[grp]) +
								", weight=" + std::to_string(wgt[grp]) +
								". Use a different group name to normalize them independently.");
		}
		normrel[grp] = normalize_relative;
		wgt[grp] = weight;

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
						fluxlist.push_back(m(i, j));
						grouplist.push_back(grp);
					}
				}
			}
		}
		else if (!GiNaC::is_a<GiNaC::numeric>(evm))
		{
			fluxlist.push_back(evm);
			grouplist.push_back(grp);
		}
	}

	// Truncates the string form of an expression: a single residual contribution of, say, the
	// Navier-Stokes equations prints as tens of thousands of characters and buries the part of the
	// message that actually says what is wrong. PYOOMPH_FULL_UNIT_ERROR=1 keeps everything.
	static std::string shortened_expression(const GiNaC::ex &e, size_t maxlen)
	{
		std::ostringstream oss;
		oss << e;
		std::string s = oss.str();
		if (getenv("PYOOMPH_FULL_UNIT_ERROR") || s.size() <= maxlen)
			return s;
		return s.substr(0, maxlen) + " [...] (truncated, set PYOOMPH_FULL_UNIT_ERROR=1 for the full expression)";
	}

	// Report for a contribution that is not dimensionless after nondimensionalisation.
	//
	// The old message dumped the raw GiNaC input form, the expanded form, and a single
	// (factor, unit, rest) split of the whole expression. For the by far most common mistake - a sum
	// whose terms do not share the same units - that split fails outright ("CANNOT SEPARATE UNITS AND
	// REST"), so the message reported the failure of the diagnostic rather than the mistake, and never
	// even named the base unit that survived. Splitting the sum first and collecting the units per
	// term instead separates the two failure modes that need entirely different fixes: terms that
	// disagree with each other (something in the equation is wrong) versus terms that agree but are
	// not dimensionless (a scale is missing or wrong).
	std::string FiniteElementCode::format_dimensional_error(const GiNaC::ex &input, const GiNaC::ex &expanded, const std::string &what)
	{
		std::ostringstream oss;
		oss << std::endl;

		std::vector<std::string> leftover;
		for (auto &bu : base_units)
			if (expanded.has(bu.second))
				leftover.push_back(bu.first);

		oss << what << " is not dimensionless." << std::endl;
		if (!leftover.empty())
		{
			oss << "It still carries the base unit";
			if (leftover.size() > 1) oss << "s";
			oss << ": ";
			for (size_t i = 0; i < leftover.size(); i++)
				oss << (i ? ", " : "") << leftover[i];
			oss << std::endl;
		}
		oss << "Every contribution must be free of units once the scales have been divided out." << std::endl;

		// Split the sum and classify each term by its units. Expand first: the caller hands in a
		// .normal()ised expression, in which a mixed-unit sum typically hides inside a product
		// ((a*kelvin + b*meter*second)/second) and would be classified as one inseparable term.
		GiNaC::ex flat = GiNaC::expand(expanded);
		std::vector<GiNaC::ex> terms;
		if (GiNaC::is_a<GiNaC::add>(flat))
			for (size_t i = 0; i < flat.nops(); i++)
				terms.push_back(flat.op(i));
		else
			terms.push_back(flat);

		std::vector<GiNaC::ex> group_units;
		std::vector<std::vector<GiNaC::ex>> group_terms;
		std::vector<GiNaC::ex> inseparable;
		for (auto &t : terms)
		{
			GiNaC::ex factor, unit, rest;
			if (!expressions::collect_base_units(t, factor, unit, rest))
			{
				inseparable.push_back(t);
				continue;
			}
			size_t g = 0;
			for (; g < group_units.size(); g++)
				if ((unit / group_units[g]).normal().is_equal(GiNaC::ex(1)))
					break;
			if (g == group_units.size())
			{
				group_units.push_back(unit);
				group_terms.push_back({});
			}
			group_terms[g].push_back(factor * rest);
		}

		if (!inseparable.empty())
		{
			oss << std::endl
				<< "The units of " << inseparable.size() << " term(s) cannot be separated from the rest at all." << std::endl
				<< "This happens when a dimensional quantity ends up as the argument of a function that" << std::endl
				<< "requires a dimensionless one (sin, exp, log, ...), or when terms of different units are" << std::endl
				<< "added inside such an argument. The first such term is:" << std::endl
				<< "  " << shortened_expression(inseparable[0], 600) << std::endl;
		}

		if (group_units.size() > 1)
		{
			oss << std::endl
				<< "The " << terms.size() << " additive terms carry " << group_units.size()
				<< " different units - no choice of scales can make such a sum dimensionless:" << std::endl;
			for (size_t g = 0; g < group_units.size(); g++)
			{
				oss << "  [" << (g + 1) << "] unit " << group_units[g] << "   (" << group_terms[g].size() << " term"
					<< (group_terms[g].size() > 1 ? "s" : "") << ")" << std::endl;
				for (size_t i = 0; i < group_terms[g].size() && i < 3; i++)
					oss << "        " << shortened_expression(group_terms[g][i], 300) << std::endl;
				if (group_terms[g].size() > 3)
					oss << "        ... and " << (group_terms[g].size() - 3) << " further term(s) of this unit" << std::endl;
			}
			oss << std::endl;
			for (size_t g = 1; g < group_units.size(); g++)
				oss << "  [" << (g + 1) << "]/[1] = " << (group_units[g] / group_units[0]).normal()
					<< "  -> one of the two is off by exactly this factor." << std::endl;
		}
		else if (group_units.size() == 1 && !group_units[0].is_equal(GiNaC::ex(1)))
		{
			oss << std::endl
				<< (terms.size() == 1 ? "It carries the unit " : "All terms agree on the unit ")
				<< group_units[0] << ", i.e. it is consistent with itself but" << std::endl
				<< "not dimensionless. Usually a scale is missing or wrong: check set_scaling(...) at problem level" << std::endl
				<< "and define_scaling()/set_scaling(...) at equation level. For a residual contribution, also check" << std::endl
				<< "the test function scale (test_scale_factor) of the field it is projected on." << std::endl;
		}

		if (!this->expanded_scales.empty())
		{
			oss << std::endl
				<< "Scales used in domain '" << this->get_full_domain_name() << "':" << std::endl;
			for (auto &entry : this->expanded_scales)
				oss << "  " << entry.first << " = " << entry.second << std::endl;
		}

		oss << std::endl
			<< "As written:" << std::endl
			<< "  " << shortened_expression(input, 1200) << std::endl;
		if (!input.is_equal(expanded))
			oss << "Fully expanded:" << std::endl
				<< "  " << shortened_expression(expanded, 1200) << std::endl;
		return oss.str();
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
				GiNaC::matrix expam = GiNaC::ex_to<GiNaC::matrix>(expa);
				const unsigned ncomp = (unsigned)expa.nops();

				// Split every component into (numeric factor, unit, rest) first, then decide which
				// one supplies the reference unit the others are converted to.
				std::vector<GiNaC::ex> facs(ncomp), units(ncomp), rests(ncomp);
				for (unsigned int cd = 0; cd < ncomp; cd++)
				{
					if (!expressions::collect_base_units(expa[cd], facs[cd], units[cd], rests[cd]))
					{
						std::ostringstream oss;
						oss << std::endl
							<< "INPUT FORM:" << what << std::endl;
						oss << "EXPANDED FORM, component " << cd << ":" << expa[cd] << std::endl;
						oss << "CANNOT SEPARATE UNITS AND REST" << std::endl;
						oss << "NUMERICAL FACTOR: " << facs[cd] << std::endl;
						oss << "COLLECTED UNITS: " << units[cd] << std::endl;
						oss << "REMAINING PART: " << rests[cd] << std::endl;
						throw_runtime_error("Found a inseparable units in the added expression:" + oss.str());
					}
				}

				// An identically zero component carries no unit, and its factor comes back as 0.
				// Taking component 0 as the reference unconditionally therefore divided every other
				// component by zero as soon as the first one happened to vanish - vector(0, 1) was
				// not expressible at all, in any expression that reaches here.
				int ref = -1;
				for (unsigned int cd = 0; cd < ncomp; cd++)
				{
					if (!GiNaC::is_zero(rests[cd]) && !GiNaC::is_zero((facs[cd] * units[cd]).evalf()))
					{
						ref = (int)cd;
						break;
					}
				}

				for (auto &bu : base_units)
				{
					sublist.append(bu.second == 1);
				}

				std::vector<GiNaC::ex> newvect;
				if (ref < 0)
				{
					// Every component vanishes: there is nothing to nondimensionalize.
					(*collected_units_and_factor) = 1;
					for (unsigned int cd = 0; cd < ncomp; cd++)
						newvect.push_back(0);
				}
				else
				{
					(*collected_units_and_factor) = facs[ref] * units[ref];
					for (unsigned int cd = 0; cd < ncomp; cd++)
					{
						if (GiNaC::is_zero(rests[cd]))
						{
							newvect.push_back(0);
							continue;
						}
						if ((int)cd == ref)
						{
							newvect.push_back(rests[cd]);
							continue;
						}
						GiNaC::ex conversion = (facs[cd] * units[cd] / (*collected_units_and_factor)).expand().evalm().normal();
						GiNaC::ex factor2, unit2, rest2;
						if (!expressions::collect_base_units(conversion, factor2, unit2, rest2))
						{
							std::ostringstream oss;
							oss << std::endl
								<< "INPUT FORM:" << what << std::endl;
							oss << "EXPANDED FORM, component " << cd << ":" << expa[cd] << std::endl;
							oss << "CANNOT SEPARATE UNITS AND REST, when comparing to the base unit of vector component "
								<< ref << ", namely " << (*collected_units_and_factor) << std::endl;
							oss << "NUMERICAL FACTOR: " << factor2 << std::endl;
							oss << "COLLECTED UNITS: " << unit2 << std::endl;
							oss << "REMAINING PART: " << rest2 << std::endl;
							throw_runtime_error("Found a inseparable units in the added expression:" + oss.str());
						}
						newvect.push_back(rests[cd] * conversion);
					}
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
					// Last chance: the units may still cancel once collected
					GiNaC::ex factor, unit, rest;
					if (expressions::collect_base_units(expa, factor, unit, rest) && unit.is_equal(1))
					{
						sublist.append(bu.second == 1);
						continue;
					}
					throw_runtime_error(this->format_dimensional_error(what, expa, "The added expression (" + where + ")"));
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
		__phase_timer __t_all("TOTAL");
		if (stage > 1)
			throw_runtime_error("Cannot add residuals any more");
		if (stage == 0)
			index_fields();
		// Checking the contribution

		//      GiNaC::ex expanded=expand_all_and_ensure_nondimensional(add);

		GiNaC::ex expanded;
		{
			__phase_timer __t("expand_placeholders");
			expanded = this->expand_placeholders(add, "Residual");
		}
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

		GiNaC::ex repl;
		{
			__phase_timer __t("DrawUnitsOutOfSubexpressions");
			DrawUnitsOutOfSubexpressions units_out_of_subexpressions(this);
			repl = units_out_of_subexpressions(expanded);
		}
		// `expa` is read only by the base-unit check below; the residual actually stored is
		// repl.subs(sublist). Normalising a residual with rational nonlinearities puts all the
		// denominators over a common one and costs seconds per contribution, all discarded. Since
		// expand()/normal() can only cancel base units, never introduce them, a residual that
		// mentions none can skip the normalisation entirely (4-species benchmark: 58.9 s -> 1.8 s).
		// Caveat: not calling normal() means it allocates no temporary symbols, which shifts GiNaC's
		// symbol serials and hence its canonical factor order - 2 of 2537 lines of the benchmark's
		// generated C came out as permuted products, so last-bit results can move and Tier-1 JIT
		// cache entries miss once. PYOOMPH_DISABLE_UNIT_PRESCAN restores the old path;
		// PYOOMPH_PARANOID_UNIT_PRESCAN re-verifies the invariant the expensive way.
		static const bool prescan_disabled = getenv("PYOOMPH_DISABLE_UNIT_PRESCAN") != NULL;
		bool may_be_dimensional = prescan_disabled;
		{
			__phase_timer __t("base_unit_prescan");
			for (GiNaC::const_preorder_iterator i = repl.preorder_begin(); i != repl.preorder_end() && !may_be_dimensional; ++i)
			{
				if (!GiNaC::is_a<GiNaC::symbol>(*i))
					continue;
				for (auto &bu : base_units)
				{
					if (i->is_equal(bu.second))
					{
						may_be_dimensional = true;
						break;
					}
				}
			}
		}
		// Same idea for contributions that genuinely are dimensional. The check below, when it finds
		// a surviving unit, already asks collect_base_units() for the verdict - so ask it first, on
		// the un-normalised expression. It is a structural dimensional analysis with no rational
		// normalisation, so it skips the expensive part; if it cannot prove the units cancel we fall
		// through to the old code, which decides. It can only turn a rejection into an acceptance.
		// Opt-in because the payoff is narrow and the perturbation is not: 1.17 s -> 0.39 s on a
		// dimensional 4-species benchmark, but a wash on twelve dimensional tutorials, while
		// skipping normal() renumbers the CSE temporaries and reassociates products (100+ lines on
		// marangoni_instability - all verified numerically equivalent, but that is not a reason to
		// reassociate every dimensional Jacobian by default). See dev_docs/code_generation.md.
		static const bool fastcheck_enabled = getenv("PYOOMPH_UNIT_FASTCHECK") != NULL;
		bool units_proven_to_cancel = false;
		if (may_be_dimensional && fastcheck_enabled && !prescan_disabled)
		{
			__phase_timer __t("collect_base_units_fastcheck");
			GiNaC::ex f_, u_, r_;
			if (expressions::collect_base_units(repl, f_, u_, r_) && u_.is_equal(1))
				units_proven_to_cancel = true;
		}
		GiNaC::ex expa;
		if (may_be_dimensional && !units_proven_to_cancel)
		{
			__phase_timer __t("expand().normal()");
			expa = repl.expand().normal();
		}
		else if (units_proven_to_cancel && getenv("PYOOMPH_PARANOID_UNIT_PRESCAN"))
		{
			// Cross-check of the fast path: run the authoritative computation and insist it would
			// have accepted too.
			GiNaC::ex check = repl.expand().normal();
			for (auto &bu : base_units)
			{
				if (check.has(bu.second))
				{
					GiNaC::ex f2, u2, r2;
					if (!expressions::collect_base_units(check, f2, u2, r2) || !u2.is_equal(1))
					{
						std::ostringstream oss;
						oss << std::endl
							<< "collect_base_units() on the un-normalised contribution said the units cancel, "
							<< "but the normalised form still carries " << bu.first << ":" << std::endl
							<< check << std::endl;
						throw_runtime_error("PARANOID_UNIT_PRESCAN violated (fastcheck)" + oss.str());
					}
				}
			}
		}
		else if (!may_be_dimensional && getenv("PYOOMPH_PARANOID_UNIT_PRESCAN"))
		{
			// Opt-in cross-check of the invariant above: do the work the prescan just skipped and
			// insist it agrees. Only for validating this optimisation against real (dimensional)
			// models; it is strictly slower than not having the prescan at all.
			GiNaC::ex check = repl.expand().normal();
			for (auto &bu : base_units)
			{
				if (check.has(bu.second))
				{
					std::ostringstream oss;
					oss << std::endl << "Base unit " << bu.first << " appeared only after expand().normal():" << std::endl << check << std::endl;
					throw_runtime_error("PARANOID_UNIT_PRESCAN violated" + oss.str());
				}
			}
		}
		GiNaC::lst sublist;
		__phase_timer *__t_bu = new __phase_timer("base_unit_has_loop");
		for (auto &bu : base_units)
		{
			if (may_be_dimensional && !units_proven_to_cancel && expa.has(bu.second))
			{
				// Last chance: the units may still cancel once collected
				GiNaC::ex factor, unit, rest;
				if (expressions::collect_base_units(expa, factor, unit, rest) && unit.is_equal(1))
				{
					sublist.append(bu.second == 1);
					continue;
				}
				throw_runtime_error(this->format_dimensional_error(add, expa, "The added residual contribution"));
			}
			sublist.append(bu.second == 1);
		}
		delete __t_bu;

		__phase_timer __t_subs("repl.subs(sublist)");
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
			os << indent << "unsigned n_int_pt=(shapeinfo->during_shared_multi_assembling ? 1 : shapeinfo->n_int_pt);" << std::endl;
			os << indent << "for(unsigned ipt=0;ipt<n_int_pt;ipt++)" << std::endl;
		}
		else
		{
			os << indent << "for(unsigned ipt=0;ipt<shapeinfo->n_int_pt;ipt++)" << std::endl;
		}
		os << indent << "{" << std::endl;
		if (this->use_shared_shape_buffer_during_multi_assemble)
		{
			os << indent << "   if (!shapeinfo->during_shared_multi_assembling)" << std::endl;
			os << indent << "   {" << std::endl;
		}
		os << indent << "  my_func_table->fill_shape_buffer_for_point(eleminfo, ipt, " << required_table_and_flag << ");" << std::endl;
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
			std::set<std::string> subexpr_decls_written;
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
					// Two req_fields entries of one subexpression can differ only in no_jacobian /
					// no_hessian / expansion_mode - flags that ShapeExpansion::operator< discriminates on
					// but get_spatial_interpolation_name does not - and would then declare the same C
					// variable twice. Sharing one variable is safe because the fill below matches on
					// (field, dt_order, basis, dt_scheme) only, so every variant produces the same value.
					// The fill loop applies the identical rule, so declaration and assignment agree.
					if (subexpr_decls_written.count(derivname.str()))
						continue;
					os << "    double " << derivname.str() << ";" << std::endl;
					subexpr_decls_written.insert(derivname.str());
					//	subexpressions[j].derivsyms[f.get_cpp_symbol()]=GiNaC::symbol(derivname.str());
					//			}
				}
				// Additional derivatives with respect to coordinates
			}
			if (!hessian)
				os << "    if (flag)" << std::endl;
			os << "    {" << std::endl;

			std::set<std::string> subexpr_fills_written;
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
					// Same collapsing rule as the declaration loop above, and applied before the diff so
					// the skipped variant costs nothing.
					if (subexpr_fills_written.count(derivname.str()))
						continue;
					if (pyoomph::pyoomph_verbose)
					{
						std::cout << "DERIVING SUBSEXPRESSION " << subexpressions[j].get_expression() << " BY " << f.field->get_symbol() << ", more specifically by " << (0 + GiNaC::GiNaCShapeExpansion(f)) << std::endl;
					}
					// The four flags below are set for the derivative and nothing else: the scope closes
					// before deriv_se_to_1 runs, which must not see them.
					//
					// __derive_only_by_expansion_mode: every d_subexpr_N_d_<field> the Hessian actually
					// reads is a *second*-index derivative - the outer index never touches the cache, it
					// builds a nested subexpression instead (GiNaCSubExpression::derivative). The second
					// index is differentiated under the Hessian expansion mode, the base state, mode 0,
					// not the Jacobian's perturbation mode 1; filling with the Jacobian mode would cache
					// the wrong derivative for azimuthal and Cartesian normal-mode stability.
					//
					// __in_hessian off: left set, a nested subexpression in the body would take the
					// outer-index branch and append a *new* entry to se_to_struct_hessian->subexpressions
					// - after this list was snapshotted and after the declarations above were emitted, so
					// it would have neither a "double subexpr_N =" line nor a declared derivative, and
					// resolve_subexpression would fail on it. Reading the nested subexpression's own cache
					// is also what the chain rule wants here.
					//
					// __ignore_dpsi_coord_diffs_in_jacobian on: the result is cached in a scalar C
					// variable, so it has to be the PURE partial derivative with respect to this one
					// interpolated quantity. Differentiating by a coordinate field also produces the
					// moving-mesh dpsi/dX contributions, and those are l_shape-indexed arrays that a
					// scalar cannot hold: they came out inside d_subexpr_N_d_..._d2x00_coordinate_x as
					// "..._COORDDIFF_0_u[l_shape]" and the generated code then failed to compile with
					// "'l_shape' undeclared". They do not belong here anyway - the chain rule that
					// consumes this cache sums over every required field and multiplies by dq/dX itself,
					// so keeping them would double count. Only azimuthal/normal-mode expansions on a
					// moving mesh reach this, since only they put second spatial derivatives of the
					// coordinates into a subexpression.
					GiNaC::ex dsdf;
					{
						AmbientCodegenScope<const ShapeExpansion *> wrto_scope(__deriv_subexpression_wrto, &f);
						AmbientCodegenScope<int *> expansion_mode_scope(__derive_only_by_expansion_mode,
																		hessian ? this->get_derive_hessian_by_expansion_mode() : this->get_derive_jacobian_by_expansion_mode());
						AmbientCodegenScope<bool> not_in_hessian(pyoomph::__in_hessian, false);
						AmbientCodegenScope<bool> ignore_dpsi(pyoomph::__ignore_dpsi_coord_diffs_in_jacobian, true);
						dsdf = pyoomph::expressions::diff(subexpressions[j].get_expression(), f.field->get_symbol());
					}
					DerivedShapeExpansionsToUnity deriv_se_to_1(f.basis,f.dt_order,f.dt_scheme); // Map all other expanded basis functions to zero to separate between e.g. d/dx or nonderived shapes
					GiNaC::ex dsub = deriv_se_to_1(dsdf);
					if (pyoomph::pyoomph_verbose)
					{
						std::cout << "DERIVING SUBSEXPRESSION RESULT " << dsdf << " OR " << dsub << std::endl;
					}
					
					// if (!dsub.is_zero())
					{
						os << "     " << derivname.str() << " = ";
						subexpr_fills_written.insert(derivname.str());
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
				if (sp.history_step != 0 && this->history_geometry_is_relevant())
					this->mark_shapes_required(for_what, this->get_my_position_space(), "history_geometry" + std::to_string(sp.history_step));
				// The position-space "psi" that goes with a normal is the POSITION-JACOBIAN channel, not a
				// shape the residual reads: the normal's nodal-coordinate sensitivities are expressed in
				// the position-owning element's nodes, so that element's position dofs have to be
				// attached, remapped and given hang info. On a static mesh there are no position dofs and
				// none of that exists, yet the mark used to be unconditional - the structurally identical
				// elemsize branch below has always had the guard. The consequence was not merely a set
				// flag: a non-NULL bulk_shapes forces a full bulk shape fill per integration point, a
				// bulk interpolate_hang_values, bulk equation remapping, and (because the interface
				// fill_hang_info_with_equations returns true whenever it recursed) the HANG entry point
				// for EVERY interface element of a static mesh. Corpus: 175 of the generated codes
				// setting it never touch bulk_shapeinfo at all.
				//
				// Guarded on the code OWNING the marked position space, exactly as the elemsize branch
				// does. No is_lagrangian() term: a normal is never Lagrangian.
				if (sp.get_code() == this || sp.get_code() == NULL)
				{
					this->mark_shapes_required(for_what, this->get_my_position_space(), "normal");
					if (sp.spatial_deriv_direction >= 0)
						this->mark_shapes_required(for_what, this->get_my_position_space(), "normal_deriv");
					if (this->bulk_code)
					{
						if (this->bulk_code->coordinates_as_dofs)
							this->mark_shapes_required(for_what, this->bulk_code->get_my_position_space(), "psi");
					}
					else
					{
						if (this->coordinates_as_dofs)
							this->mark_shapes_required(for_what, this->get_my_position_space(), "psi");
					}
				}
				else if (this->bulk_code && sp.get_code() == this->bulk_code)
				{
					this->mark_shapes_required(for_what, this->bulk_code->get_my_position_space(), "normal");
					if (sp.spatial_deriv_direction >= 0)
						this->mark_shapes_required(for_what, this->bulk_code->get_my_position_space(), "normal_deriv");
					if (this->bulk_code->bulk_code)
					{
						if (this->bulk_code->bulk_code->coordinates_as_dofs)
							this->mark_shapes_required(for_what, this->bulk_code->bulk_code->get_my_position_space(), "psi");
					}
					else
					{
						if (this->bulk_code->coordinates_as_dofs)
							this->mark_shapes_required(for_what, this->bulk_code->get_my_position_space(), "psi");
					}
				}
				else if (this->opposite_interface_code && sp.get_code() == this->opposite_interface_code)
				{
					this->mark_shapes_required(for_what, this->opposite_interface_code->get_my_position_space(), "normal");
					if (sp.spatial_deriv_direction >= 0)
						this->mark_shapes_required(for_what, this->opposite_interface_code->get_my_position_space(), "normal_deriv");
					if (this->opposite_interface_code->bulk_code)
					{
						if (this->opposite_interface_code->bulk_code->coordinates_as_dofs)
							this->mark_shapes_required(for_what, this->opposite_interface_code->bulk_code->get_my_position_space(), "psi");
					}
					else
					{
						if (this->opposite_interface_code->coordinates_as_dofs)
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
			else if (GiNaC::is_a<GiNaC::GiNaCShapeExpansion>(*i))
			{
				// An Eulerian shape derivative evaluated in the past has to be taken on the mesh as it
				// was then, so the geometry of that history level must be computed as well.
				const pyoomph::ShapeExpansion &sp = GiNaC::ex_to<GiNaC::GiNaCShapeExpansion>(*i).get_struct();
				if (sp.history_geometry && sp.time_history_index != 0 && this->history_geometry_is_relevant() && sp.basis->is_eulerian_deriv())
				{
					this->mark_shapes_required(for_what, this->get_my_position_space(), "history_geometry" + std::to_string(sp.time_history_index));
				}
			}
			else if (GiNaC::is_a<GiNaC::GiNaCElementSizeSymbol>(*i))
			{
				const pyoomph::ElementSizeSymbol &sp = GiNaC::ex_to<GiNaC::GiNaCElementSizeSymbol>(*i).get_struct();
				if (sp.history_step != 0 && this->history_geometry_is_relevant())
					this->mark_shapes_required(for_what, this->get_my_position_space(), "history_geometry" + std::to_string(sp.history_step));
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
				// Note the opposite_interface_code null check: unlike the branches above, this one is also reached on a
				// plain interface without an opposite side (e.g. a contact line asking for an element size that is
				// neither its own nor its bulk's), where dereferencing it segfaults instead of reporting the error below.
				else if (this->opposite_interface_code && this->opposite_interface_code->bulk_code && sp.get_code() == this->opposite_interface_code->bulk_code)
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
	//      Hessian contributions are not supported and raise an error if present) and turn its
	//      subexpression(...) markers into structs via a fresh SubExpressionsToStructs instance
	//      (se_to_struct_hessian) dedicated to this Hessian pass. The double differentiation in step 2
	//      then grows that list further, with one nested subexpression per (subexpression, outer field)
	//      pair - see GiNaCSubExpression::derivative, whose first-derivative cache is the Hessian's
	//      second derivative.
	//   2. Call write_generic_RJM_contribution(..., hessian=true) on every FiniteElementSpace, which
	//      (via write_generic_Hessian_contribution, see above) performs the actual double symbolic
	//      differentiation and accumulates, as a side effect, the code's all_Hessian_shapeexps/
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
		this->current_shapeflag_func_type = "Hessian[" + std::to_string(residual_index) + "]";
		AmbientCodegenScope<bool> in_hessian_scope(__in_hessian, true); // cleared on every exit, including the throwing ones
		bool has_contribs = false;
		std::ostringstream osh; // Header
		std::ostringstream osm; // Main contribution

		this->all_Hessian_shapeexps.clear();
		this->all_Hessian_shapeexps_for_shapeflags.clear();
		this->all_Hessian_testfuncs.clear();
		this->all_Hessian_indices_required.clear();
		delete this->se_to_struct_hessian;
		this->se_to_struct_hessian = new SubExpressionsToStructs(this);

		GiNaC::ex spatial_integral_portion_Eulerian = extract_spatial_integral_part(resi, true, false);	  // resi.coeff(get_dx(false), 1) * get_dx(false);
		GiNaC::ex spatial_integral_portion_Lagrangian = extract_spatial_integral_part(resi, false, true); // resi.coeff(get_dx(true), 1) * get_dx(true);
		GiNaC::ex spatial_integral_portion_NodalDelta = resi.coeff(get_nodal_delta(), 1);

		// if (!spatial_integral_portion_Lagrangian.is_zero()) this->mark_shapes_required("ResJac["+std::to_string(residual_index)+"]",spaces[0],"psi");
		GiNaC::ex spatial_integral_portion = spatial_integral_portion_Eulerian + spatial_integral_portion_Lagrangian;

		spatial_integral_portion = (*this->se_to_struct_hessian)(spatial_integral_portion);
		// Needed before the first differentiation, not merely for tidiness: GiNaCSubExpression::derivative
		// resolves against this list, and write_generic_Hessian_contribution only refreshes it *after* its
		// first diff() call.
		subexpressions = this->se_to_struct_hessian->subexpressions;

		// Constructed before the osm expression printing below; the declarations are emitted into the
		// osh header stream later, which precedes osm in the assembled function text
		GlobalParameterFunctionScope gp_scope(this, {resi});

		// Same buffer aliases as the Residual/Jacobian/Mass path, and they pay more here: a Hessian entry
		// sits in a THREE-deep loop (l_test, l_shape, l_shape2) whose innermost body stores into
		// hessian_buffer, so every shapeinfo-> read in there was reloaded per iteration of the innermost
		// loop. The scope has to open here, before the contributions are printed into osm, because osm is
		// generated before the osh header it will be concatenated behind; the declarations are emitted
		// into osh, right after the integration-loop header (see below).
		BufferAliasFunctionScope alias_scope(this);

		osm << "    //START: Contribution of the spaces" << std::endl;
		osm << "    double _H_contrib;" << std::endl;
		for (auto *sp : allspaces)
		{
			has_contribs = sp->write_generic_RJM_contribution(this, osm, "    ", spatial_integral_portion, true) || has_contribs;
		}
		osm << "    //END: Contribution of the spaces" << std::endl;

		// Authoritative harvest of everything the subexpression bodies read. The incremental sweep in
		// write_generic_Hessian_contribution() cannot be relied on for this: it sits inside the (f,f2)
		// loop, so the assemble_hessian_by_symmetry "continue" skips it, and the subexpressions that the
		// last (f,f2) pair appends are never scanned at all. A field missed here is one that
		// write_spatial_interpolation is never asked to produce, i.e. an undeclared identifier in the
		// emitted C rather than a wrong number.
		subexpressions = this->se_to_struct_hessian->subexpressions;
		for (auto &se : subexpressions)
		{
			for (auto &sh : get_all_shape_expansions_in(se.get_expression()))
			{
				if (!sh.is_derived && !sh.is_derived_other_index)
					this->all_Hessian_shapeexps.insert(sh);
				this->all_Hessian_shapeexps_for_shapeflags.insert(sh);
				this->all_Hessian_indices_required.insert(sh.field);
			}
		}

		if (!has_contribs)
		{
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
		gp_scope.write_declarations(osh, "  ");
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
		std::set<ShapeExpansion> all_shapeexps = this->all_Hessian_shapeexps;
		std::set<TestFunction> all_testfuncs = this->all_Hessian_testfuncs;
		std::set<FiniteElementField *,FiniteElementFieldPtrLess> indices_required = this->all_Hessian_indices_required;

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
		// The basis functions of the Jacobian COLUMNS (phi_i, dphi_i/dx, ... at l_shape/l_shape2), which
		// all_shapeexps deliberately does not contain - see all_Hessian_shapeexps_for_shapeflags. They
		// are only marked, never merged into the interpolation sets.
		for (auto &sp : this->all_Hessian_shapeexps_for_shapeflags)
		{
			this->mark_shapes_required("Hessian[" + std::to_string(residual_index) + "]", sp.field->get_space(), sp.basis);
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
		{
			// Above the integration loop, i.e. above where the alias declarations will land - see
			// BufferAliasSuspend. The scope is still open here because osm was printed with it.
			BufferAliasSuspend no_aliases(this);
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
		}

		if (!spatial_integral_portion.is_zero())
		{
			osh << "  //START: Spatial integration loop" << std::endl;
			std::string required_name = "&(my_func_table->shapes_required_Hessian[" + std::to_string(residual_index) + "]), 3";
			write_generic_spatial_integration_header(osh, "  ", spatial_integral_portion_Eulerian, spatial_integral_portion_Lagrangian, required_name);
			std::ostringstream osh_ipt; // in-loop part, so the alias declarations can be emitted ahead of it
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
			osh_ipt << "    //START: Interpolate all required fields" << std::endl;
			for (auto *sp : allspaces)
			{
				if (sp->need_interpolation_loop())
				{
					sp->write_spatial_interpolation(this, osh_ipt, "    ", merged_shapeexps, this->coordinates_as_dofs, true);
				}
			}
			osh_ipt << "    // SUBEXPRESSIONS" << std::endl
				<< std::endl;
			spatial_integral_portion = this->write_code_subexpressions(osh_ipt, "     ", spatial_integral_portion, spatial_shape_exps, true);
			// osm as well as osh_ipt: the Hessian assembly body was printed first and lives in osm, which
			// is concatenated after this header.
			osh << alias_scope.declarations("    ", osh_ipt.str() + osm.str());
			osh << osh_ipt.str();
		}
		osh << "    //END: Interpolate all required fields" << std::endl
			<< std::endl;

		osh << std::endl;

		if (!has_contribs)
		{
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
	void FiniteElementCode::write_generic_RJM(std::ostream &os, std::string funcname, GiNaC::ex resi, bool, bool may_be_asked_for_mass_matrix, bool allow_hang_split)
	{
		this->current_shapeflag_func_type = "ResJac[" + std::to_string(residual_index) + "]";
		// Redundant since write_generic_Hessian restores the flag on every exit (AmbientCodegenScope);
		// kept because it costs nothing and this function is also reached from paths that predate that.
		__in_hessian = false;
		emitted_mass_matrix_contribution = false;
		// Emitted as an implementation with a CONSTANT flag plus a dispatcher below, so that the three
		// assembly modes become three specialised bodies. See PYOOMPH_RJM_IMPL in jitbridge.h.
		static const bool split_rjm_globally = getenv("PYOOMPH_DISABLE_RJM_SPLIT") == NULL;
		const bool split_rjm = split_rjm_globally && this->split_rjm_by_flag;
		// The same trick once more, for the hanging-node machinery: a second constant parameter turns
		// the one emitted body into a hanging and a non-hanging entry point, and the assembly picks per
		// element (elements_assembly.cpp). Requires split_rjm - without the _impl there is nothing to
		// add a constant parameter to - and with_adaptivity, since otherwise no hang macro is emitted
		// at all and the two entry points would be the same code.
		static const bool hang_split_globally = getenv("PYOOMPH_DISABLE_HANG_SPLIT") == NULL;
		const bool split_hang = split_rjm && hang_split_globally && this->split_rjm_by_hang && this->with_adaptivity && allow_hang_split;
		this->emitting_hang_parameter = split_hang;
		this->hang_parameter_was_used = false;
		const std::string hangparam = (split_hang ? ",const int pyoomph_hang_on" : "");
		if (split_rjm)
			os << "PYOOMPH_RJM_IMPL " << funcname << "_impl(const JITElementInfo_t * eleminfo, const JITShapeInfo_t * shapeinfo,double * PYOOMPH_RESTRICT residuals, double * PYOOMPH_RESTRICT jacobian, double * PYOOMPH_RESTRICT mass_matrix,const unsigned flag" << hangparam << ")" << std::endl;
		else
			os << "static void " << funcname << "(const JITElementInfo_t * eleminfo, const JITShapeInfo_t * shapeinfo,double * PYOOMPH_RESTRICT residuals, double * PYOOMPH_RESTRICT jacobian, double * PYOOMPH_RESTRICT mass_matrix,unsigned flag)" << std::endl;
		os << "{" << std::endl;
		os << "  int local_eqn, local_unknown;" << std::endl;
		os << "  bool _has_residual_contribution,_has_jacobian_contribution;" << std::endl;

		// TODO: Only if hanging allowed
		os << "  unsigned nummaster,nummaster2;" << std::endl;
		os << "  double hang_weight,hang_weight2;" << std::endl;

		os << "  const double * t=shapeinfo->t;" << std::endl;
		os << "  const double * dt=shapeinfo->dt;" << std::endl;
		// The scatter macros index with these instead of re-reading shapeinfo->jacobian_size /
		// ->mass_matrix_size, which they may not keep in a register: the jacobian[]/mass_matrix[] store
		// in the same statement can alias shapeinfo (PYOOMPH_RESTRICT is empty and does not help, see
		// jitbridge.h), so both the load and the row multiply were repeated per inner-loop iteration.
		// Both are set by prepare_shape_buffer_for_integration before this function is entered and do
		// not change inside it. Hoisting them alone is worth nothing - it only pays together with the
		// buffer aliases, because one surviving shapeinfo-> read in the loop forces the reload of all
		// the others. HessianVectorProduct has had the same local (n_dof) all along.
		os << "  const unsigned _jacsize = shapeinfo->jacobian_size;" << std::endl;
		os << "  const unsigned _msize = shapeinfo->mass_matrix_size;" << std::endl;
		if (stage == 0)
			index_fields();

		// Everything printed below (residual entries, Jacobian/mass derivatives, CSE bodies) derives
		// from resi, so screening resi covers the whole function
		GlobalParameterFunctionScope gp_scope(this, {resi});
		gp_scope.write_declarations(os, "  ");

		std::set<ShapeExpansion> all_shapeexps = get_all_shape_expansions_in(resi, true);

		std::set<TestFunction> all_testfuncs = get_all_test_functions_in(resi);
		std::set<FiniteElementField *,FiniteElementFieldPtrLess> indices_required;
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

		std::ostringstream body;

		body << "  //START: Precalculate time derivatives of the necessary data" << std::endl;
		for (auto *sp : allspaces)
		{
			sp->write_nodal_time_interpolation(this, body, "  ", all_shapeexps);
		}
		body << "  //END: Precalculate time derivatives of the necessary data" << std::endl
		   << std::endl;

		// First assign the "interpolated D0" values
		for (auto *sp : allspaces)
		{
			if (!sp->need_interpolation_loop())
			{
				sp->write_spatial_interpolation(this, body, "    ", all_shapeexps, false, false);
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
			body << "  //START: Spatial integration loop" << std::endl;
			std::string required_name = "&(my_func_table->shapes_required_ResJac[" + std::to_string(residual_index) + "]), flag";
			write_generic_spatial_integration_header(body, "  ", spatial_integral_portion_Eulerian, spatial_integral_portion_Lagrangian, required_name);

			// The buffer aliases are declared HERE, at the top of the integration-point body, and not at
			// function scope - even though function scope would be equally correct, since
			// prepare_shape_buffer_for_integration re-points the buffers once per assembly and
			// fill_shape_buffer_for_point never does. It was tried, and it is measurably worse: the
			// aliases would then be live across the indirect fill_shape_buffer_for_point call, there are
			// far more of them than there are callee-saved registers, and every use pays a stack reload
			// instead of a register read. Callgrind on the heated-cylinder element, 300 assemblies:
			// 107.6M data reads unaliased, 108.0M with function-scope aliases (i.e. nothing gained),
			// 102.5M with these. Redeclaring per integration point costs a handful of loads per point
			// and buys the whole trial loop.
			BufferAliasFunctionScope alias_scope(this);
			std::ostringstream ipt_body;

			std::set<ShapeExpansion> spatial_shape_exps = get_all_shape_expansions_in(spatial_integral_portion);
			ipt_body << "    //START: Interpolate all required fields" << std::endl;
			for (auto *sp : allspaces)
			{
				if (sp->need_interpolation_loop())
				{
					sp->write_spatial_interpolation(this, ipt_body, "    ", spatial_shape_exps, this->coordinates_as_dofs, false);
				}
			}
			ipt_body << "    //END: Interpolate all required fields" << std::endl
			   << std::endl;

			ipt_body << std::endl;

			ipt_body << "    // SUBEXPRESSIONS" << std::endl
			   << std::endl;
			spatial_integral_portion = this->write_code_subexpressions(ipt_body, "     ", spatial_integral_portion, spatial_shape_exps, false);

			ipt_body << "    //START: Contribution of the spaces" << std::endl;
			ipt_body << "    double _res_contrib,_J_contrib;" << std::endl;
			for (auto *sp : allspaces)
			{
				sp->write_generic_RJM_contribution(this, ipt_body, "    ", spatial_integral_portion, false);
			}
			ipt_body << "    //END: Contribution of the spaces" << std::endl;

			const std::string ipt_text = ipt_body.str();
			body << alias_scope.declarations("    ", ipt_text);
			body << ipt_text;

			write_generic_spatial_integration_footer(body, "  ");
			body << "  //END: Spatial integration loop" << std::endl
			   << std::endl;
		}

		if (!spatial_integral_portion_NodalDelta.is_zero())
		{
			body << "  //START: Nodal delta" << std::endl;
			body << "  //END: Nodal delta" << std::endl;
			//	write_generic_nodal_delta_header(os,"  "); //TODO

			std::set<ShapeExpansion> nodal_shape_exps = get_all_shape_expansions_in(spatial_integral_portion_NodalDelta);
			body << "    //START: Interpolate all required fields" << std::endl;
			body << "    double _res_contrib,_J_contrib;" << std::endl;
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
						sp->write_generic_RJM_contribution(this, body, "    ", spatial_integral_portion_NodalDelta, false);
					}
					else
					{
						throw_runtime_error("Non-D0 nodal delta");
					}
				}
			}
			body << "    //END: Interpolate all required fields" << std::endl
			   << std::endl;

			//			write_generic_nodal_delta_footer(os,"  "); //TODO

			body << std::endl;
		}

		os << body.str();

		os << "}" << std::endl
		   << std::endl;

		// The registered entry point. Keeping the name means nothing outside this file changes: the
		// function table, the C++ callers and the ABI are all as they were.
		this->emitting_hang_parameter = false;
		if (!split_rjm)
			return;
		// Two entry points only if the body actually reads pyoomph_hang_on somewhere; see
		// hang_parameter_was_used in codegen.hpp.
		const bool write_nohang_entry = split_hang && this->hang_parameter_was_used;
		if (write_nohang_entry)
			this->emitted_nohang_entry_points.insert(funcname);
		const std::string args = "eleminfo, shapeinfo, residuals, jacobian, mass_matrix";
		// hangarg is what the dispatcher hands the constant parameter: the normal entry point keeps the
		// hanging-node machinery, the _NoHang twin folds it away.
		auto write_dispatcher = [&](const std::string &entryname, const std::string &hangarg)
		{
		os << "static void " << entryname << "(const JITElementInfo_t * eleminfo, const JITShapeInfo_t * shapeinfo,double * PYOOMPH_RESTRICT residuals, double * PYOOMPH_RESTRICT jacobian, double * PYOOMPH_RESTRICT mass_matrix,unsigned flag)" << std::endl;
		os << "{" << std::endl;
		os << "  if (flag == 0) " << funcname << "_impl(" << args << ", 0" << hangarg << ");" << std::endl;
		if (!emitted_mass_matrix_contribution)
		{
			// No mass-matrix entry was emitted anywhere in this routine, so `if (flag == 2)` guards
			// nothing and the flag==1 and flag==2 bodies are the same code. Ask for one of them.
			os << "  else " << funcname << "_impl(" << args << ", 1" << hangarg << "); /* no mass matrix in this routine */" << std::endl;
		}
		else if (!may_be_asked_for_mass_matrix)
		{
			// This routine has mass-matrix entries but is the UNSTEADY one of a residual that also got a
			// separate steady routine, i.e. MPT/TPZ time integration. A mass matrix is requested through
			// whichever routine tstepper->is_steady() picks (src/elements_assembly.cpp:157,378), and
			// every consumer that asks for one either makes the timesteppers steady first
			// (pyoomph/generic/problem.py:5374 and the make_steady/undo pairs in bifurcation_tools,
			// assembly, periodic_driving_response, lyapunov:109) or is BDF2-only and therefore never
			// reaches this branch at all - LyapunovExponentCalculatorBDF2, the one caller that assembles
			// a mass matrix mid-transient, is BDF2 by construction and BDF2 needs no steady routine.
			//
			// So flag==2 should not arrive here, and specialising to 1 saves the third body. The assert
			// is what keeps that an assumption we would HEAR about rather than one that silently returns
			// a zero mass matrix: it costs one predictable branch per element per assembly.
			os << "  else { assert(flag < 2 && \"mass matrix requested from the unsteady routine of an MPT/TPZ residual - make the timesteppers steady first\"); "
			   << funcname << "_impl(" << args << ", 1" << hangarg << "); }" << std::endl;
		}
		else
		{
			os << "  else if (flag == 1) " << funcname << "_impl(" << args << ", 1" << hangarg << ");" << std::endl;
			os << "  else " << funcname << "_impl(" << args << ", 2" << hangarg << ");" << std::endl;
		}
		os << "}" << std::endl
		   << std::endl;
		};
		write_dispatcher(funcname, split_hang ? ", 1" : "");
		if (write_nohang_entry)
			write_dispatcher(funcname + "_NoHang", ", 0");
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
		// my_func_table used to be a file-scope `static JITFuncSpec_Table_FiniteElement_t *` here,
		// assigned by JIT_ELEMENT_init. One .so serving two Problems then shared one pointer: dlopen
		// dedupes by inode, so the second Problem's init silently repointed it and the first
		// Problem's elements started reading the second's global_parameters (and dangled once that
		// Problem was destroyed). It is now the code's own table reached through the element info
		// that every generated entry point already receives. Kept as a macro rather than rewritten
		// at ~15 emission sites so that the compiler flags any body that has no `eleminfo` in scope
		// instead of that being found at runtime.
		os << "#define my_func_table (eleminfo->functable)" << std::endl
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
		// Not an assembled contribution: nothing here may mark dx_psi_dcoord, and leaving the
		// last routine's value standing would attribute its reads to the wrong contribution.
		this->current_shapeflag_func_type.clear();
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

		GlobalParameterFunctionScope gp_scope(this, {gathered});
		gp_scope.write_declarations(os, "  ");

		std::set<ShapeExpansion> all_shapeexps = get_all_shape_expansions_in(gathered);
		std::set<TestFunction> all_testfuncs = get_all_test_functions_in(gathered);
		if (!all_testfuncs.empty())
		{
			throw_runtime_error("Found test function in a custom integral/local expression");
		}
		std::set<FiniteElementField *,FiniteElementFieldPtrLess> indices_required;
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
			os << "    my_func_table->fill_shape_buffer_for_point(eleminfo, ipt, &(my_func_table->shapes_required_IntegralExprs), 0);" << std::endl;
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

	// Emits the EvalTracerAdvection function that evaluates one registered tracer-advection velocity
	// field (used to advect passive tracer particles through the flow field) at the local coordinate
	// the shape buffer was filled at. Structurally similar to
	// write_code_integral_or_local_expressions but specialized for the vector output.
	//
	// There is no time blending here. Each tracer name registers one entry per nodal time-history
	// level (see TracerParticles in pyoomph/equations/tracers.py), and the caller blends them with
	// weights it derives from t(0),t(1),t(2) at run time. Those weights cannot be baked in: for BDF2
	// with a changing dt they depend on the actual time levels, which are unknown here. The old code
	// passed a `timefrac_tracer` scalar and blended two levels linearly inside the generated code,
	// which could only ever be first order in a varying-dt run.
	void FiniteElementCode::write_code_tracer_advection(std::ostream &os)
	{
		// Not an assembled contribution: nothing here may mark dx_psi_dcoord, and leaving the
		// last routine's value standing would attribute its reads to the wrong contribution.
		this->current_shapeflag_func_type.clear();
		// Past-level geometry must be materialised even on a code without position dofs: a mesh moved
		// by macro elements or by direct node manipulation still has a nodal position history, and a
		// gradient inside a past-level advection expression has to be taken on the matching
		// configuration. history_geometry_is_relevant() gates that on coordinates_as_dofs, which is
		// the right default for residuals (it avoids duplicating interpolations that cannot differ)
		// but silently wrong here.
		struct ForceHistoryGeometry
		{
			FiniteElementCode *c;
			ForceHistoryGeometry(FiniteElementCode *code) : c(code) { c->force_history_geometry = true; }
			~ForceHistoryGeometry() { c->force_history_geometry = false; }
		} force_hist(this);

		os << "static void EvalTracerAdvection(const JITElementInfo_t * eleminfo, const JITShapeInfo_t * shapeinfo, unsigned index, double * result_velo)" << std::endl;
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

		GlobalParameterFunctionScope gp_scope(this, {gathered});
		gp_scope.write_declarations(os, "  ");

		std::set<ShapeExpansion> all_shapeexps = get_all_shape_expansions_in(gathered);
		std::set<TestFunction> all_testfuncs = get_all_test_functions_in(gathered);
		if (!all_testfuncs.empty())
		{
			throw_runtime_error("Found test function in tracer advection terms");
		}
		std::set<FiniteElementField *,FiniteElementFieldPtrLess> indices_required;
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

		// No `dx` here on purpose. The shape buffer is filled at an arbitrary local coordinate, not at
		// an integration point, so shapeinfo->int_pt_weight[0] holds whatever the last assembly left
		// in it. Emitting it as `dx` made a stale value silently available to any expression.
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
			std::set<FiniteElementField*,FiniteElementFieldPtrLess> indices_required;
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
		// Not an assembled contribution: nothing here may mark dx_psi_dcoord, and leaving the
		// last routine's value standing would attribute its reads to the wrong contribution.
		this->current_shapeflag_func_type.clear();
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

		GlobalParameterFunctionScope gp_scope(this, {gathered});
		gp_scope.write_declarations(os, "  ");

		std::set<ShapeExpansion> all_shapeexps = get_all_shape_expansions_in(gathered);
		std::set<TestFunction> all_testfuncs = get_all_test_functions_in(gathered);
		if (!all_testfuncs.empty())
		{
			throw_runtime_error("Found test function in spatial error estimator");
		}
		std::set<FiniteElementField *,FiniteElementFieldPtrLess> indices_required;
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
	
	

	// See get_precodegen_fingerprint_text() declaration in codegen.hpp for rationale: a canonical,
	// cheap-to-compute text serialization of everything write_code() reads, captured BEFORE it does
	// any (expensive) symbolic differentiation/CSE/printing. Used only in shadow mode (see
	// pyoomph/generic/jit_cache.py): never to actually skip codegen, only to predict whether it would
	// produce the same output as a previous run, so that gaps in this fingerprint's coverage (a
	// mismatch between the prediction and the real write_code() output) can be caught empirically
	// before ever trusting it to skip codegen for real.
	std::string FiniteElementCode::get_precodegen_fingerprint_text()
	{
		std::ostringstream os;
		GiNaC::print_dflt pc(os);
		auto print_ex = [&](const GiNaC::ex &e)
		{ e.print(pc); os << "\n"; };
		auto print_named_ex_map = [&](const std::map<std::string, GiNaC::ex> &m)
		{ for (auto &kv : m) { os << kv.first << "="; print_ex(kv.second); } };

		os << "FMT10\n"; // Bump whenever this function's coverage/format changes
		// FMT10: the hanging-node split (split_rjm_by_hang, PYOOMPH_DISABLE_HANG_SPLIT) is covered, and
		// so is PYOOMPH_DISABLE_RJM_SPLIT, which decides the same kind of thing and was missing.
		// FMT9: the Z2 compound-flux grouping and its per-group normalization/weight are covered
		// (see the Z2 block below). Before that, the same estimator expressions in different groups
		// or with a different normalize_relative/weight shared one fingerprint.
		// FMT8: the printed form of a shape expansion now carries its time history level and whether
		// its Eulerian shape derivative is taken on that level's geometry, and the normal/element-size
		// symbols carry their history level. Before that, evaluate_in_past() variants of the same
		// expression printed identically, so residuals that differ only in a history level shared one
		// fingerprint while generating different code -- which is exactly what Tier-2 shadow mode
		// reported once apply_on_others made both variants appear in one residual.
		// The codegen switches that can change what write_code() emits (see
		// dev_docs/code_generation.md). Without these the same fingerprint would map to two different
		// generated-code hashes depending on the environment, which makes Tier-2 shadow mode report
		// a mismatch it cannot explain - and would let a future codegen-skipping Tier-2 reuse code
		// generated under a different setting. PYOOMPH_ARCHIVE_EXPRESSIONS is deliberately absent:
		// the archive is write-only and cannot affect the emitted code.
		os << "sw_memo=" << __expand_memo_on
		   << " sw_unit_fastcheck=" << (getenv("PYOOMPH_UNIT_FASTCHECK") != NULL)
		   << " sw_no_unit_prescan=" << (getenv("PYOOMPH_DISABLE_UNIT_PRESCAN") != NULL)
		   << " sw_no_rjm_split=" << (getenv("PYOOMPH_DISABLE_RJM_SPLIT") != NULL)
		   << " sw_no_hang_split=" << (getenv("PYOOMPH_DISABLE_HANG_SPLIT") != NULL) << "\n";
		os << "dim=" << nodal_dim << " lagr_dim=" << lagr_dim << " max_dt_order=" << max_dt_order << " integration_order=" << integration_order << "\n";
		os << "generate_hessian=" << generate_hessian << " assemble_hessian_by_symmetry=" << assemble_hessian_by_symmetry << "\n";
		os << "analytical_jacobian=" << analytical_jacobian << " analytical_position_jacobian=" << analytical_position_jacobian << "\n";
		os << "with_adaptivity=" << with_adaptivity << " ccode_expression_mode=" << ccode_expression_mode << "\n";
		os << "use_shared_shape_buffer_during_multi_assemble=" << use_shared_shape_buffer_during_multi_assemble << "\n";
		os << "coordinates_as_dofs=" << coordinates_as_dofs << "\n"; // moving-mesh/ALE flag (functable->moving_nodes)
		// The emission choices of 9.4, recorded so a generated file can be traced back to the settings
		// that produced it. They need no cache epoch of their own: both change the generated C, and the
		// JIT cache key is the contents of that file (pyoomph/generic/ccompiler.py).
		os << "jacobian_hoist_min_cost=" << jacobian_hoist_min_cost << " split_rjm_by_flag=" << split_rjm_by_flag << " split_rjm_by_hang=" << split_rjm_by_hang << "\n";
		// NOTE: typeid(*coordinate_sys).name() is deliberately NOT used here (unlike elsewhere in
		// this function) - every Python-level CustomCoordinateSystem subclass (Cartesian,
		// Axisymmetric, ...) is wrapped by the same single nanobind trampoline class on the C++
		// side, so its RTTI type is identical regardless of which one it actually is. This was
		// tried and found completely non-discriminating (see branch jit_cache's investigation of
		// cross-script Tier-2 fingerprint collisions between e.g. poisson_2d.py and
		// poisson_axisymm_adaptive.py, whose "left" boundary codes have identical fingerprints
		// under a typeid-based check but differ in the actual generated GeometricJacobian(),
		// since it embeds the coordinate system's geometric_jacobian(), not its C++ type). Using
		// get_id_name() (a virtual, Python-overridden, human-readable identifier - "Cartesian",
		// "Axisymmetric", ...) plus the actual symbolic geometric_jacobian()/
		// jacobian_for_element_size() expressions instead directly captures what the coordinate
		// system actually *does*, the same way `residual` below captures behavior rather than a
		// type name. Uses get_coordinate_system() (virtual - "to be overloaded to get it from the
		// element as well", see its declaration) rather than the raw coordinate_sys member
		// directly: boundary/interface codes typically leave their own coordinate_sys member
		// unset and resolve it from the bulk domain through this override instead, so reading the
		// raw member here returned the same base-class default ("<unknown coordinate system>",
		// geometric_jacobian()==1.0) for every boundary code regardless of its actual coordinate
		// system - exactly the same non-discriminating result as the typeid() attempt above.
		CustomCoordinateSystem *effective_coordsys = this->get_coordinate_system();
		if (effective_coordsys)
		{
			os << "coordsys_id=" << effective_coordsys->get_id_name() << "\n";
			os << "coordsys_geometric_jacobian=";
			print_ex(effective_coordsys->geometric_jacobian());
			os << "coordsys_jacobian_for_element_size=";
			print_ex(effective_coordsys->jacobian_for_element_size());
		}
		else
		{
			os << "coordsys_id=NULL\n";
		}
		os << "reference_pos=";
		for (double v : reference_pos_for_IC_and_DBC)
			os << v << ",";
		os << "\n";

		// Which fields/spaces this code actually operates on: two codes can have textually
		// identical residual expressions (e.g. trivial/near-empty point constraints at an
		// axisymmetric axis) while still emitting different C code, because that code also
		// iterates over/emits per-field entries (e.g. SET_INTERNAL_FIELD_NAME(...), the
		// Dirichlet-condition-name table) for whichever fields are actually registered here -
		// which residual/settings alone do not capture. get_full_domain_name() is included for
		// the same reason: it is literally embedded into the generated code as
		// SET_INTERNAL_NAME(functable->domain_name, ...), so two codes on different domains with
		// otherwise-identical fingerprints would otherwise silently collide (this is exactly the
		// gap a shadow-mode mismatch on branch jit_cache's rising_bubble.py test caught, between
		// two distinct axis-point element codes on different parent domains).
		os << "full_domain_name=" << get_full_domain_name() << "\n";
		// Per-field Dirichlet/initial-condition state: two codes can agree on residual, settings,
		// coordinate system and even the plain name@space list above while still differing here -
		// e.g. a NoSlipBC() pinning both velocity components vs. a symmetry condition pinning only
		// one, or the same field with different initial conditions. Found missing this exact case
		// (a moving-mesh free_surface.py boundary colliding with a fixed-mesh cavity_forward_
		// problem.py boundary that pins one fewer component) while investigating Tier-2 shadow-
		// mode mismatches on branch jit_cache.
		os << "myfields=";
		for (auto *f : myfields)
		{
			os << f->get_name() << "@" << f->get_space()->get_name();
			if (f->Dirichlet_condition_set)
			{
				os << "|DBC" << (f->Dirichlet_condition_pin_only ? "(pin_only)" : "") << "=";
				print_ex(f->Dirichlet_condition);
			}
			for (auto &kv : f->initial_condition)
			{
				os << "|IC[" << kv.first << "]=";
				print_ex(kv.second);
			}
			os << ",";
		}
		os << "\n";
		os << "spaces=";
		for (auto *s : spaces)
			os << s->get_name() << ",";
		os << "\n";

		os << "residual_names=";
		for (auto &n : residual_names)
			os << n << ",";
		os << "\n";
		os << "residuals:\n";
		for (auto &r : residual)
			print_ex(r);

		os << "ignore_assemble_residuals=";
		for (auto &n : ignore_assemble_residuals)
			os << n << ",";
		os << "\n";
		os << "nullified_bulk_residuals=";
		for (auto &n : nullified_bulk_residuals)
			os << n << ",";
		os << "\n";
		os << "derive_jacobian_by_expansion_mode=";
		for (auto &kv : derive_jacobian_by_expansion_mode)
			os << kv.first << ":" << kv.second << ",";
		os << "\n";
		os << "derive_hessian_by_expansion_mode=";
		for (auto &kv : derive_hessian_by_expansion_mode)
			os << kv.first << ":" << kv.second << ",";
		os << "\n";
		os << "ignore_dpsi_coord_diffs_in_jacobian_set=";
		for (auto &n : ignore_dpsi_coord_diffs_in_jacobian_set)
			os << n << ",";
		os << "\n";

		os << "IC_names=";
		for (auto &n : IC_names)
			os << n << ",";
		os << "\n";

		os << "has_hessian_contribution=";
		for (bool b : has_hessian_contribution)
			os << b << ",";
		os << "\n";
		os << "has_constant_mass_matrix_for_sure=";
		for (bool b : has_constant_mass_matrix_for_sure)
			os << b << ",";
		os << "\n";
		os << "extra_steady_routine=";
		for (bool b : extra_steady_routine)
			os << b << ",";
		os << "\n";

		os << "Z2_fluxes:\n";
		for (auto &e : Z2_fluxes)
			print_ex(e);
		os << "Z2_fluxes_for_eigen:\n";
		for (auto &e : Z2_fluxes_for_eigen)
			print_ex(e);
		// The compound-flux grouping: the same flux expressions split over different groups, or one
		// group normalized/weighted differently, emit different Z2_flux_group_index/
		// Z2_group_normalize_relative/Z2_group_weight arrays (see write_group_arrays() in
		// write_code()) while the expressions above are identical - which is exactly the Tier-2
		// shadow-mode mismatch the grouping introduced. The settings are printed the same way
		// write_code() prints them into the arrays, so equal fingerprints mean equal code. Group
		// *names* are deliberately not included: they only pick the index and never reach the
		// generated code, so hashing them would split the cache for no reason.
		auto print_Z2_groups = [&](const std::vector<unsigned> &groups, const std::vector<double> &normrel, const std::vector<double> &wgt, const std::string &sfx)
		{
			os << "Z2_flux_groups" << sfx << "=";
			for (unsigned g : groups)
				os << g << ",";
			os << "\n";
			os << "Z2_group_settings" << sfx << "=";
			for (unsigned int g = 0; g < normrel.size(); g++)
				os << std::to_string(normrel[g]) << ":" << std::to_string(wgt[g]) << ",";
			os << "\n";
		};
		print_Z2_groups(Z2_flux_groups, Z2_group_normalize_relative, Z2_group_weight, "");
		print_Z2_groups(Z2_flux_groups_for_eigen, Z2_group_normalize_relative_for_eigen, Z2_group_weight_for_eigen, "_for_eigen");

		os << "integral_expressions:\n";
		print_named_ex_map(integral_expressions);
		os << "local_expressions:\n";
		print_named_ex_map(local_expressions);
		os << "extremum_expressions:\n";
		print_named_ex_map(extremum_expressions);
		os << "tracer_advection_terms:\n";
		print_named_ex_map(tracer_advection_terms);

		// Only the (deterministic, construction-order-based) identity of referenced Python callbacks is
		// captured here, not whether the Python-side callable they wrap changed - a known coverage gap,
		// left for shadow mode to surface empirically rather than solved here.
		os << "cb_expressions_ids=";
		for (auto *c : cb_expressions)
			os << (c ? (long)c->get_unique_id() : -1L) << ",";
		os << "\n";
		os << "multi_ret_expressions_ids=";
		for (auto *c : multi_ret_expressions)
			os << (c ? (long)c->unique_id : -1L) << ",";
		os << "\n";

		os << "bulk_code=" << (bulk_code ? "1" : "0") << " opposite_interface_code=" << (opposite_interface_code ? "1" : "0") << "\n";
		os << "required_odes_count=" << required_odes.size() << "\n";

		return os.str();
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
		this->hoist_coeff_counter = 0; // names are per code, not per process - see codegen.hpp
		this->emitted_nohang_entry_points.clear(); // Refilled by write_generic_RJM; a rewrite must not inherit the last one's decisions
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
			AmbientCodegenScope<bool> pitchfork_scope(__in_pitchfork_symmetry_constraint,
													  residual_names[resind] == "_simple_mass_matrix_of_defined_fields");
			if (!residual[resind].is_zero())
			{
				// Record the block expressions for the JACOBIAN_BLOCK_* flags, but only here: the steady
				// and dResidual/dParameter variants below assemble something else entirely. Residual sets
				// derived by expansion mode (azimuthal/Cartesian normal-mode stability) are skipped too -
				// their blocks are complex and would need conjugate-transpose semantics, so their flags
				// stay 0, which the contract reads as "nothing proven".
				// Check if we need a dedicated steady routine. This happens, if you use e.g. MPT or TPZ
				// time integration, which use history values. Decided BEFORE the unsteady routine is
				// written, because it decides how many specialised bodies that routine needs:
				// assemble_eigenproblem_matrices asks for a mass matrix through whichever routine
				// tstepper->is_steady() selects (src/elements_assembly.cpp:157,378), so once a separate
				// steady routine exists, the unsteady one is the unusual place to ask - not an
				// impossible one, which is why write_generic_RJM keeps a runtime flag there rather than
				// specialising the branch away. MakeResidualSteady is a pure GiNaC map, so hoisting it
				// above the write changes nothing else.
				MakeResidualSteady make_steady(this);
				GiNaC::ex steady_residual = make_steady(residual[resind]);
				extra_steady_routine[resind] = make_steady.require_extra_steady_routine();

				record_jacobian_blocks_for_flags = (get_derive_jacobian_by_expansion_mode() == NULL);
				write_generic_RJM(os, "ResidualAndJacobian" + std::to_string(resind), residual[resind], true, !extra_steady_routine[resind], true); // Hanging unsteady routine
				record_jacobian_blocks_for_flags = false;
				os << std::endl;

				if (extra_steady_routine[resind])
				{
					os << std::endl;
					// The steady twin gets the hang split too: a stationary solve is the common case, and
					// elements_assembly.cpp picks the Steady slot without ever looking at the unsteady one.
					write_generic_RJM(os, "ResidualAndJacobianSteady" + std::to_string(resind), steady_residual, true, true, true); // Hanging steady routine
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

	// As classify_space_type(), but reports failure instead of throwing. Callers that only want to
	// REFINE a decision must use this one: block_contribution_class_name() splits a contribution class
	// by side, and a split it cannot justify has to fall back to the merged (conservative) class rather
	// than abort code generation. The throwing wrapper below stays for the code-emission paths, where
	// an unclassifiable space genuinely means the generated C code could not be written.
	bool FiniteElementCode::try_classify_space_type(const FiniteElementSpace *s, int &out) const
	{
		for (unsigned int i = 0; i < spaces.size(); i++)
			if (s == spaces[i])
			{
				out = 0;
				return true;
			}
		if (bulk_code)
			for (unsigned int i = 0; i < bulk_code->spaces.size(); i++)
				if (s == bulk_code->spaces[i])
				{
					out = -1;
					return true;
				}
		if (opposite_interface_code)
		{
			for (unsigned int i = 0; i < opposite_interface_code->spaces.size(); i++)
				if (s == opposite_interface_code->spaces[i])
				{
					out = -2;
					return true;
				}
			if (opposite_interface_code->bulk_code)
			{
				for (unsigned int i = 0; i < opposite_interface_code->bulk_code->spaces.size(); i++)
					if (s == opposite_interface_code->bulk_code->spaces[i])
					{
						out = -3;
						return true;
					}
			}
		}
		if (bulk_code && bulk_code->bulk_code)
			for (unsigned int i = 0; i < bulk_code->bulk_code->spaces.size(); i++)
				if (s == bulk_code->bulk_code->spaces[i])
				{
					out = -4;
					return true;
				}
		return false;
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

	// A code only has a distinguishable history configuration if its nodes can move. On a static mesh
	// the geometry is the same at every history level, so the history slots are never filled and the
	// symbols keep reading slot 0 - which is both cheaper and avoids reading a slot nobody wrote.
	// Interfaces inherit the answer from their bulk, whose nodes they share.
	bool FiniteElementCode::history_geometry_is_relevant() const
	{
		if (this->force_history_geometry)
			return true; // see write_code_tracer_advection
		if (this->coordinates_as_dofs)
			return true;
		for (const FiniteElementCode *c = this->bulk_code; c; c = c->bulk_code)
		{
			if (c->coordinates_as_dofs)
				return true;
		}
		return false;
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
		GiNaC::ex gathered_ics;
		unsigned ic_wild_cnt = 0;
		for (auto *f : myfields)
			if (f->initial_condition.count(ic_name))
				gathered_ics += f->initial_condition[ic_name] * GiNaC::wild(ic_wild_cnt++); // Wild against accidental cancellation of parameters between fields
		GlobalParameterFunctionScope gp_scope(this, {gathered_ics});
		gp_scope.write_declarations(os, "  ");
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
		GiNaC::ex gathered_dcs;
		unsigned dc_wild_cnt = 0;
		for (auto *f : myfields)
			if (f->Dirichlet_condition_set)
				gathered_dcs += f->Dirichlet_condition * GiNaC::wild(dc_wild_cnt++); // Wild against accidental cancellation of parameters between fields
		GlobalParameterFunctionScope gp_scope(this, {gathered_dcs});
		gp_scope.write_declarations(os, "  ");
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
			// The expression arriving here has already been divided by the scale of the field, so a
			// leftover unit is exactly the mismatch between the value the user gave and that scale.
			throw_runtime_error(this->format_dimensional_error(expression, expression,
															   "The Dirichlet or initial condition for field '" + fieldname + "' (divided by the scale of '" + fieldname + "')"));
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
		// Only the plain normal has a counterpart in the initial/Dirichlet condition C signature
		// (the "_normal" argument). A spatially derived one would survive the substitution below and
		// then fail the "not a numeric" dry run with an unintelligible message.
		for (GiNaC::const_preorder_iterator it = expression.preorder_begin(); it != expression.preorder_end(); ++it)
		{
			if (GiNaC::is_a<GiNaC::GiNaCNormalSymbol>(*it) &&
				GiNaC::ex_to<GiNaC::GiNaCNormalSymbol>(*it).get_struct().spatial_deriv_direction >= 0)
			{
				throw_runtime_error("Spatial derivatives of the normal, e.g. grad(normal) or div(normal), cannot be used in initial or Dirichlet conditions - only the normal itself is available there.");
			}
		}
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
	// definition) hook, with __current_code pointing at this code so that the GiNaC structures
	// built during that call can reach their coordinate system and dimensions.
	void FiniteElementCode::finalise()
	{
		// residual[0]=0;
		residual.clear();
		residual_index = 0;
		residual_names.clear();
		residual.push_back(0);
		residual_names.push_back("");
		__current_code = this;
		_define_element();
		__current_code = NULL;
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
					if (psientry.first=="psi" || psientry.first=="dx_psi" || psientry.first=="dX_psi" || psientry.first=="d2x_psi" || psientry.first=="d2X_psi" || psientry.first=="dx_psi_dcoord")
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
	// "moving_mesh" becomes the constant 0/1 depending on coordinates_as_dofs.
	GiNaC::ex FiniteElementCode::eval_flag(std::string flagname)
	{
		if (flagname == "moving_mesh")
		{
			return (coordinates_as_dofs ? 1 : 0);
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
		// An identically zero field has no unit to infer, and asking for one divides by it. Store it
		// as-is instead of failing: "advect by nothing while the mesh moves" is the sharpest test
		// there is of the ALE handling, so it must be expressible.
		{
			GiNaC::ex probe = (0 + expr).evalm();
			bool all_zero = true;
			if (GiNaC::is_a<GiNaC::matrix>(probe))
			{
				for (size_t i = 0; i < probe.nops() && all_zero; i++)
					if (!GiNaC::is_zero(probe.op(i)))
						all_zero = false;
			}
			else
				all_zero = GiNaC::is_zero(probe);
			if (all_zero)
			{
				this->tracer_advection_terms[name] = probe;
				this->tracer_advection_units[name] = 1;
				return;
			}
		}

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

	////////////////////////////////////////////////////////////////////////////////////////////////
	// Per-block property analysis: the JACOBIAN_BLOCK_* bits of jitbridge.h.
	//
	// Input are the symbolic (row test field, column unknown field) block expressions recorded by
	// write_generic_RJM_jacobian_contribution(); output are the two flag tables emitted by
	// write_code_info() below. A SET bit is a proof, an unset bit means "not proven" - so every case
	// that is not fully understood must fall through to unset.
	//
	// NOTHING in here may print a GiNaC expression, not even into a discarded stream: global
	// parameters are registered up front by GlobalParameterFunctionScope, but
	// GiNaCGlobalParameterWrapper::print() still lazily registers a slot on unscreened paths, so a
	// printing analysis pass could still renumber the generated dResidual<N>dParameter_<i> routines
	// and functable->global_parameters.
	////////////////////////////////////////////////////////////////////////////////////////////////

	// Suffix marking the contribution class of a field seen on the FAR side of an interior facet. Kept
	// as a named constant because problem.cpp strips it again when the per-element classes are
	// aggregated into the problem-wide field list, and elements.cpp appends it when an interface
	// element adopts the attribution of its opposite neighbour: three places that must agree.
	const std::string OPPOSITE_CONTRIBUTION_CLASS_SUFFIX = "@opposite";

	// True for a space whose values live in Data owned by ONE element (DG/DL/D0), as opposed to a
	// continuous space, whose nodal values on a facet are shared by both adjacent elements.
	//
	// This is the condition under which the two sides of an interior facet may be split into separate
	// contribution classes at all. For a continuous field the split would be unsound: a facet dof is
	// literally the same unknown on both sides, so it is reachable through both class columns, and a
	// one-sided facet term would then declare a genuinely nonzero entry structurally absent.
	static bool is_element_private_space(pyoomph::FiniteElementSpace *sp)
	{
		if (!sp || sp->is_external())
			return false;
		return dynamic_cast<pyoomph::DGFiniteElementSpace *>(sp) != NULL ||
			   dynamic_cast<pyoomph::DiscontinuousFiniteElementSpace *>(sp) != NULL;
	}

	// The contribution CLASS of a field: the domain it is DEFINED on plus its name, with mesh_* and
	// coordinate_* being the same dof and hence the same class. Used both to index the functable
	// tables (write_code_info below) and to name the row/column roles in the symmetry comparison; the
	// two must not drift apart, which is why they share this function.
	//
	// for_code, when given, additionally splits off the far side of an interior facet as its own class
	// "<domain>/<field>@opposite". Without that split the two sides of a facet are indistinguishable
	// here - the opposite side is a dummy FiniteElementCode carrying the same domain path - so the
	// coupling table cannot state that the near-side and far-side copies of an element-local field do
	// NOT couple. That statement is what lets static condensation decompose an HDG system into one
	// block per element instead of one block spanning the whole mesh.
	//
	// Only the opposite roles (-2, -3) are split, and only for element-private spaces. Everything else
	// keeps the merged name, deliberately: roles 0/-1/-4 are the bulk-versus-interface views of the
	// SAME dof, which get_defined_on_domain_equivalent_field() exists to alias together. Any case that
	// cannot be classified falls back to the merged name as well - merging only ever over-reports
	// coupling, which costs a matrix entry, while an unjustified split would drop a real one.
	static std::string block_contribution_class_name(pyoomph::FiniteElementField *f, pyoomph::FiniteElementCode *for_code = NULL)
	{
		pyoomph::FiniteElementField *wheredef = f->get_defined_on_domain_equivalent_field();
		std::string nn = wheredef->get_name();
		if (nn == "mesh_x")
			nn = "coordinate_x";
		else if (nn == "mesh_y")
			nn = "coordinate_y";
		else if (nn == "mesh_z")
			nn = "coordinate_z";
		std::string res = wheredef->get_space()->get_code()->get_full_domain_name() + "/" + nn;
		if (for_code)
		{
			int role;
			if (for_code->try_classify_space_type(wheredef->get_space(), role) && (role == -2 || role == -3) &&
				is_element_private_space(wheredef->get_space()))
			{
				res += OPPOSITE_CONTRIBUTION_CLASS_SUFFIX;
			}
		}
		return res;
	}

	static bool is_position_field(pyoomph::FiniteElementField *f)
	{
		return dynamic_cast<pyoomph::PositionFiniteElementSpace *>(f->get_space()) != NULL;
	}

	// Preorder scan setting `nonconstant` as soon as the block can change between two assemblies, and
	// `dt_dependent` when the only thing it depends on is the time-stepper weights. Descends into
	// subexpressions by hand: CSE hides shape expansions and parameters behind an atom, so a plain
	// traversal (or .has()) would report a block as constant that is not.
	static void scan_block_constancy(const GiNaC::ex &e, bool moving, bool &nonconstant, bool &dt_dependent)
	{
		for (GiNaC::const_preorder_iterator i = e.preorder_begin(); i != e.preorder_end(); ++i)
		{
			if (nonconstant)
				return;
			const GiNaC::ex &c = *i;
			if (GiNaC::is_a<GiNaC::GiNaCShapeExpansion>(c))
			{
				auto &se = GiNaC::ex_to<GiNaC::GiNaCShapeExpansion>(c).get_struct();
				if (se.is_derived)
				{
					// A bare shape function or one of its spatial derivatives, i.e. pure geometry.
					if (moving || se.nodal_coord_dir != -1 || se.nodal_coord_dir2 != -1 || se.time_history_index != 0)
						nonconstant = true;
					else if (se.dt_order > 0)
						dt_dependent = true; // carries the BDF/Newmark weight
				}
				else if (!is_position_field(se.field) || moving)
				{
					// An interpolated unknown. The interpolated POSITION of a fixed mesh is not one -
					// which matters because an axisymmetric coordinate system puts it into every block.
					nonconstant = true;
				}
			}
			else if (GiNaC::is_a<GiNaC::GiNaCTestFunction>(c))
			{
				auto &tf = GiNaC::ex_to<GiNaC::GiNaCTestFunction>(c).get_struct();
				if (moving || tf.nodal_coord_dir != -1 || tf.nodal_coord_dir2 != -1)
					nonconstant = true;
			}
			else if (GiNaC::is_a<GiNaC::GiNaCSubExpression>(c))
			{
				scan_block_constancy(GiNaC::ex_to<GiNaC::GiNaCSubExpression>(c).get_struct().expr, moving, nonconstant, dt_dependent);
			}
			else if (GiNaC::is_a<GiNaC::GiNaCCustomMathExpressionWrapper>(c))
			{
				// The function IDENTITY of a scalar callback, not a value: a callback is required to be
				// a deterministic function of its arguments, so the invocation is constant whenever the
				// arguments are. python_cb_function(wrapper, args) is an ordinary two-operand GiNaC
				// function, so this walk visits those arguments by itself.
			}
			else if (GiNaC::is_a<GiNaC::GiNaCMultiRetCallback>(c))
			{
				// Same for multi-return callbacks (result value or derivative alike), but these are
				// opaque structures - the arguments have to be visited explicitly.
				scan_block_constancy(GiNaC::ex_to<GiNaC::GiNaCMultiRetCallback>(c).get_struct().invok.op(1), moving, nonconstant, dt_dependent);
			}
			else if (GiNaC::is_a<GiNaC::GiNaCSpatialIntegralSymbol>(c) || GiNaC::is_a<GiNaC::GiNaCElementSizeSymbol>(c) ||
					 GiNaC::is_a<GiNaC::GiNaCNormalSymbol>(c) || GiNaC::is_a<GiNaC::GiNaCNodalDeltaSymbol>(c))
			{
				// dx, the element size, the normal and the nodal delta are functions of the nodal
				// positions alone, hence constants on a fixed mesh. dx sits in every integrated block,
				// so without this exemption CONSTANT would never be reachable.
				if (moving)
					nonconstant = true;
			}
			else if (GiNaC::is_a<GiNaC::symbol>(c))
			{
				if (c.is_equal(pyoomph::expressions::_dt_BDF1) || c.is_equal(pyoomph::expressions::_dt_BDF2) ||
					c.is_equal(pyoomph::expressions::_dt_Newmark2) || c.is_equal(pyoomph::expressions::dt))
					dt_dependent = true;
				else
					nonconstant = true; // the explicit time symbol, and anything unrecognised
			}
			else if (c.nops() == 0 && !GiNaC::is_a<GiNaC::numeric>(c) && !GiNaC::is_a<GiNaC::constant>(c))
			{
				// Any other leaf: global parameters, delayed python expansions, mode expansions, ... -
				// every one of them can change between assemblies, and an unclassified one must be
				// assumed to.
				nonconstant = true;
			}
		}
	}

	static unsigned char block_constancy_flags(const GiNaC::ex &e, bool moving)
	{
		bool nonconstant = false, dt_dependent = false;
		scan_block_constancy(e, moving, nonconstant, dt_dependent);
		if (nonconstant)
			return 0;
		if (dt_dependent)
			return JACOBIAN_BLOCK_CONSTANT_FIXED_DT;
		return JACOBIAN_BLOCK_CONSTANT | JACOBIAN_BLOCK_CONSTANT_FIXED_DT;
	}

	// Size guard for the symmetry comparison below, whose expand() is superlinear. Counts subexpression
	// bodies at every reference, since the comparison inlines them.
	static size_t count_block_nodes(const GiNaC::ex &e, size_t cap)
	{
		size_t n = 0;
		for (GiNaC::const_preorder_iterator i = e.preorder_begin(); i != e.preorder_end(); ++i)
		{
			if (++n > cap)
				return n;
			if (GiNaC::is_a<GiNaC::GiNaCSubExpression>(*i))
			{
				n += count_block_nodes(GiNaC::ex_to<GiNaC::GiNaCSubExpression>(*i).get_struct().expr, cap - n);
				if (n > cap)
					return n;
			}
		}
		return n;
	}

	// Identifies one row/column role (or one time-stepper weight) in the canonicalised block. `tags`
	// carries the remaining attributes that make two otherwise identical shape expansions DIFFERENT
	// functions (mode expansion, history geometry): two roles sharing a key are treated as the same
	// function, so anything left out of the key that does change the function could turn into a wrong
	// symmetry proof.
	struct BlockRoleKey
	{
		int role; // 0 = row (test function), 1 = column (unknown), 2 = time-stepper weight
		std::string field;
		unsigned basis;
		int dt_order;
		std::string dt_scheme;
		std::string tags;
		bool operator<(const BlockRoleKey &o) const
		{
			return std::tie(role, field, basis, dt_order, dt_scheme, tags) < std::tie(o.role, o.field, o.basis, o.dt_order, o.dt_scheme, o.tags);
		}
	};

	// Replaces the row (test function) and column (derived shape function) roles of a block expression
	// by plain symbols, so that two blocks can be compared by subtraction. In transpose mode the two
	// roles are swapped - which is exactly what transposing the block does, since the test and the
	// trial function of one field are the same function (Galerkin).
	//
	// Anything that depends on BOTH loop indices at once - every moving-mesh nodal-coordinate
	// derivative, where the row test function itself carries a column index - cannot be expressed by a
	// single role symbol and sets `unsupported`, giving up on the block rather than risking a wrong
	// proof.
	class BlockRoleCanonicalizer : public GiNaC::map_function
	{
	protected:
		std::map<BlockRoleKey, GiNaC::symbol> *registry;
		bool transpose;
		bool *unsupported;
		// The code the blocks are being generated for, so that role symbols are keyed by the same
		// contribution class the functable tables are indexed by. Without it the near-side and far-side
		// test functions of an interior facet canonicalise to ONE role symbol, i.e. two different
		// functions sharing a key - exactly the collision BlockRoleKey warns about, and a source of
		// symmetry "proofs" that only hold once the two sides are conflated.
		pyoomph::FiniteElementCode *for_code;

		GiNaC::ex get(const BlockRoleKey &key)
		{
			auto it = registry->find(key);
			if (it != registry->end())
				return it->second;
			std::ostringstream oss;
			oss << "__blockrole" << registry->size();
			return registry->emplace(key, GiNaC::symbol(oss.str())).first->second;
		}
		GiNaC::ex role_symbol(int role, pyoomph::FiniteElementField *f, pyoomph::BasisFunction *basis, const std::string &tags)
		{
			return get(BlockRoleKey{transpose ? 1 - role : role, block_contribution_class_name(f, for_code), basis->get_creation_index(), 0, "", tags});
		}
		GiNaC::ex weight_symbol(unsigned dt_order, const std::string &scheme)
		{
			return get(BlockRoleKey{2, "", 0, (int)dt_order, scheme, ""});
		}

	public:
		BlockRoleCanonicalizer(std::map<BlockRoleKey, GiNaC::symbol> &reg, bool tr, bool &unsup, pyoomph::FiniteElementCode *fc) : registry(&reg), transpose(tr), unsupported(&unsup), for_code(fc) {}
		GiNaC::ex operator()(const GiNaC::ex &inp) override
		{
			if (*unsupported)
				return inp;
			if (GiNaC::is_a<GiNaC::GiNaCTestFunction>(inp))
			{
				auto &tf = GiNaC::ex_to<GiNaC::GiNaCTestFunction>(inp).get_struct();
				if (tf.nodal_coord_dir != -1 || tf.nodal_coord_dir2 != -1 || tf.is_derived_other_index || !tf.basis)
				{
					*unsupported = true;
					return inp;
				}
				return role_symbol(0, tf.field, tf.basis, "");
			}
			if (GiNaC::is_a<GiNaC::GiNaCShapeExpansion>(inp))
			{
				auto &se = GiNaC::ex_to<GiNaC::GiNaCShapeExpansion>(inp).get_struct();
				if (se.nodal_coord_dir != -1 || se.nodal_coord_dir2 != -1)
				{
					*unsupported = true;
					return inp;
				}
				if (!se.is_derived)
					return inp; // an interpolated value: the same factor in both blocks, so compare it as it stands
				if (se.is_derived_other_index || se.time_history_index != 0 || !se.basis)
				{
					*unsupported = true;
					return inp;
				}
				// Empty for the ordinary case, so that a plain shape expansion and the test function of
				// the same field share a role symbol - which is the whole point, since the two are the
				// same function. A mode expansion or a shape derivative taken on the history geometry is
				// NOT, so it gets its own symbol and simply fails to match, which is the safe direction.
				// no_jacobian/no_hessian are deliberately left out: they only suppress differentiation
				// elsewhere and describe the very same function.
				std::ostringstream tags;
				if (se.expansion_mode)
					tags << "M" << se.expansion_mode;
				if (se.history_geometry)
					tags << "G";
				GiNaC::ex res = role_symbol(1, se.field, se.basis, tags.str());
				if (se.dt_order > 0)
					res = res * weight_symbol(se.dt_order, se.dt_scheme); // the weight travels with the side carrying d/dt
				return res;
			}
			if (GiNaC::is_a<GiNaC::GiNaCSubExpression>(inp))
			{
				// Inlined, not reused via RemoveSubexpressionsByIdentity: that one re-registers
				// multi-return calls as a side effect, which an analysis pass must not do.
				return (*this)(GiNaC::ex_to<GiNaC::GiNaCSubExpression>(inp).get_struct().expr);
			}
			if (GiNaC::is_a<GiNaC::GiNaCMultiRetCallback>(inp))
			{
				// Opaque, but its arguments are ordinary values. If a row/column role does hide in
				// there, the atom cannot be compared meaningfully, so give up.
				GiNaC::ex invok = GiNaC::ex_to<GiNaC::GiNaCMultiRetCallback>(inp).get_struct().invok;
				if (!(*this)(invok).is_equal(invok))
					*unsupported = true;
				return inp;
			}
			// The derived variants of these carry a nodal-coordinate (i.e. column) index of their own.
			if (GiNaC::is_a<GiNaC::GiNaCSpatialIntegralSymbol>(inp))
			{
				if (GiNaC::ex_to<GiNaC::GiNaCSpatialIntegralSymbol>(inp).get_struct().is_derived())
					*unsupported = true;
				return inp;
			}
			if (GiNaC::is_a<GiNaC::GiNaCElementSizeSymbol>(inp))
			{
				if (GiNaC::ex_to<GiNaC::GiNaCElementSizeSymbol>(inp).get_struct().is_derived())
					*unsupported = true;
				return inp;
			}
			if (GiNaC::is_a<GiNaC::GiNaCNormalSymbol>(inp))
			{
				if (GiNaC::ex_to<GiNaC::GiNaCNormalSymbol>(inp).get_struct().get_derived_direction() != -1)
					*unsupported = true;
				return inp;
			}
			return inp.map(*this);
		}
	};

	// SYMMETRIC/ANTISYMMETRIC for the accumulated block A = block(i,j) against B = block(j,i)
	// (A and B are the same expression for a diagonal block). Compares the sums rather than term by
	// term: ANDing per term would be sound but would miss every cancellation.
	static unsigned char block_symmetry_flags(const GiNaC::ex &A, const GiNaC::ex &B, pyoomph::FiniteElementCode *for_code)
	{
		const size_t node_cap = 20000; // the expand() below is superlinear; over the cap, prove nothing
		if (count_block_nodes(A, node_cap) > node_cap || count_block_nodes(B, node_cap) > node_cap)
			return 0;
		std::map<BlockRoleKey, GiNaC::symbol> registry;
		bool unsupported = false;
		BlockRoleCanonicalizer plain(registry, false, unsupported, for_code), transposed(registry, true, unsupported, for_code);
		GiNaC::ex cA = plain(A);
		GiNaC::ex cBT = transposed(B);
		if (unsupported)
			return 0;
		if (cA.is_equal(cBT))
			return JACOBIAN_BLOCK_SYMMETRIC;
		if ((cA - cBT).expand().is_zero())
			return JACOBIAN_BLOCK_SYMMETRIC;
		if ((cA + cBT).expand().is_zero())
			return JACOBIAN_BLOCK_ANTISYMMETRIC;
		return 0;
	}

	void FiniteElementCode::record_jacobian_block_expr(FiniteElementField *rowf, FiniteElementField *colf, const GiNaC::ex &jac_part, const GiNaC::ex &mass_part)
	{
		auto &entry = jacobian_block_exprs[residual_index][std::make_pair(rowf, colf)];
		entry.first += jac_part;
		entry.second += mass_part;
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
	   // Which functable SpaceInfo slot each field ends up in, recorded while the per-space field names
	   // are emitted below. Read further down - once the contribution indices have been assigned - to
	   // emit the slot -> contribution-index map, so that the per-space index arithmetic (which differs
	   // between the continuous and the discontinuous branch) lives in exactly one place.
	   // Entries are (SpaceInfo member name, index within that space, field).
	   std::vector<std::tuple<std::string,unsigned,FiniteElementField*>> field_contribution_slots;
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
				// Allocated here, alongside fieldnames and with exactly the same count, so the two can never
				// disagree: the element walk indexes it by the space's field count, and sizing it from the
				// slots recorded below would under-allocate whenever a field takes part in no contribution.
				init << " functable->info_" << space->get_name() << ".field_contribution_index=(int*)malloc(sizeof(int)*functable->info_" << space->get_name() << ".numfields);" << std::endl;
				init << " for (unsigned int _i=0;_i<functable->info_" << space->get_name() << ".numfields;_i++) { functable->info_" << space->get_name() << ".field_contribution_index[_i]=-2; }" << std::endl; // -2: present but contributes to nothing; see jitbridge.h
				cleanup << " pyoomph_tested_free(functable->info_" << space->get_name() << ".field_contribution_index); functable->info_" << space->get_name() << ".field_contribution_index=PYOOMPH_NULL; " << std::endl;
				for (auto &f : myfields)
				{
					if (f->get_space() != space)
						continue;
					if (f->get_name() == "mesh_x" || f->get_name() == "mesh_y" || f->get_name() == "mesh_z")
						continue;
					
					
					init << " SET_INTERNAL_FIELD_NAME(functable->info_" << space->get_name() << ".fieldnames," << (f->index - index_offset) << ", \"" << f->get_name() << "\" );" << std::endl;										
					cleanup << " pyoomph_tested_free(functable->info_" << space->get_name() << ".fieldnames[" << (f->index - index_offset) << "]); functable->info_" << space->get_name() << ".fieldnames[" << (f->index - index_offset) << "]=PYOOMPH_NULL; " << std::endl;
					field_contribution_slots.push_back(std::make_tuple("info_"+space->get_name(), f->index - index_offset, f));
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
				// Same count as fieldnames; see the note at the other emission site.
				init << " functable->" << info_name << ".field_contribution_index=(int*)malloc(sizeof(int)*functable->" << info_name << ".numfields);" << std::endl;
				init << " for (unsigned int _i=0;_i<functable->" << info_name << ".numfields;_i++) { functable->" << info_name << ".field_contribution_index[_i]=-2; }" << std::endl; // -2: present but contributes to nothing; see jitbridge.h
				cleanup << " pyoomph_tested_free(functable->" << info_name << ".field_contribution_index); functable->" << info_name << ".field_contribution_index=PYOOMPH_NULL; " << std::endl;
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
					field_contribution_slots.push_back(std::make_tuple(info_name, contiindex, f));
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
			// Position + the dominant (max) space share the geometric slot -1. When the dominant space is
			// C2TB (either C2TB fields are present, a C1TB-enriched coordinate at line ~7753, or a manual
			// C2TB element upgrade), the position/-1 interpolation is the BUBBLE-ENRICHED one. On a
			// tetrahedron the C2TB face-centroid bubbles live ON the faces, so the enriched face trace
			// (7 masters) differs from the plain C2 face trace (6) -- unlike edges (2D) / hex faces, where
			// the bubble vanishes and the two coincide. So a plain C2 field must NOT read the enriched -1
			// hang: give C2 its OWN hang slot (its first value index) exactly as C1TB/C1 already do, so it
			// carries the plain trace. Dimension-independent (like the C1 slot) to stay consistent between a
			// bulk element and its interface facets, which inherit dominant_space but have a lower
			// element_dim; on 2D/hex the C2 slot is a harmless mirror of -1. When the dominant space is C2
			// (no enrichment) C2 shares -1 with position as before.
			init << " functable->continuous_spaces[SPACE_INDEX_C2TB].hangindex=-1;" << std::endl;
			if (coordinate_space == "C2TB")
				init << " functable->continuous_spaces[SPACE_INDEX_C2].hangindex=functable->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk;" << std::endl;
			else
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
		init << " functable->ResidualAndJacobianSteady_NoHang=(JITFuncSpec_ResidualAndJacobian_FiniteElement *)calloc(functable->num_res_jacs,sizeof(JITFuncSpec_ResidualAndJacobian_FiniteElement));" << std::endl;
		cleanup << " pyoomph_tested_free(functable->ResidualAndJacobianSteady_NoHang); functable->ResidualAndJacobianSteady_NoHang=PYOOMPH_NULL; " << std::endl;							
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
				// The _NoHang slots point at the specialised entry point where one was emitted and at the
				// hanging one otherwise, so the assembly can dereference them unconditionally.
				const std::string unsteady_name = "ResidualAndJacobian" + std::to_string(resiind);
				const std::string steady_name = (extra_steady_routine[resiind] ? "ResidualAndJacobianSteady" + std::to_string(resiind) : unsteady_name);
				const std::string unsteady_nohang = (emitted_nohang_entry_points.count(unsteady_name) ? unsteady_name + "_NoHang" : unsteady_name);
				const std::string steady_nohang = (emitted_nohang_entry_points.count(steady_name) ? steady_name + "_NoHang" : steady_name);
				init << " functable->ResidualAndJacobian[" << resiind << "]=&" << unsteady_name << ";" << std::endl;
				init << " functable->ResidualAndJacobian_NoHang[" << resiind << "]=&" << unsteady_nohang << ";" << std::endl;
				init << " functable->ResidualAndJacobianSteady[" << resiind << "]=&" << steady_name << ";" << std::endl;
				init << " functable->ResidualAndJacobianSteady_NoHang[" << resiind << "]=&" << steady_nohang << ";" << std::endl;
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

		// The compound-flux grouping and its per-group normalisation. Emitted as calloc'd arrays in
		// the same style as temporal_error_scales above, rather than as static tables, so that the
		// cleanup path can free them uniformly. The arrays are only emitted when there is something
		// to say: a single group with the default settings is exactly the historical behaviour and
		// leaves the pointers null, which is what the estimator checks to take its old fast path.
		auto write_group_arrays = [&](bool for_eigen)
		{
			const auto &groups = (for_eigen ? this->Z2_flux_groups_for_eigen : this->Z2_flux_groups);
			const auto &normrel = (for_eigen ? this->Z2_group_normalize_relative_for_eigen : this->Z2_group_normalize_relative);
			const auto &wgt = (for_eigen ? this->Z2_group_weight_for_eigen : this->Z2_group_weight);
			const std::string sfx = (for_eigen ? "_for_eigen" : "");
			if (groups.empty()) return;
			bool nondefault = (normrel.size() > 1);
			for (unsigned int g = 0; g < normrel.size(); g++)
				if (normrel[g] != 1.0 || wgt[g] != 1.0) nondefault = true;
			if (!nondefault) return; // one group, relative, unweighted: nothing the estimator needs to know
			init << " functable->num_Z2_compound_fluxes" << sfx << " = " << normrel.size() << ";" << std::endl;
			init << " functable->Z2_flux_group_index" << sfx << "=calloc(" << groups.size() << ",sizeof(unsigned));" << std::endl;
			cleanup << " pyoomph_tested_free(functable->Z2_flux_group_index" << sfx << "); functable->Z2_flux_group_index" << sfx << "=PYOOMPH_NULL; " << std::endl;
			for (unsigned int i = 0; i < groups.size(); i++)
				if (groups[i]) init << "  functable->Z2_flux_group_index" << sfx << "[" << i << "] = " << groups[i] << ";" << std::endl;
			init << " functable->Z2_group_normalize_relative" << sfx << "=calloc(" << normrel.size() << ",sizeof(double));" << std::endl;
			cleanup << " pyoomph_tested_free(functable->Z2_group_normalize_relative" << sfx << "); functable->Z2_group_normalize_relative" << sfx << "=PYOOMPH_NULL; " << std::endl;
			init << " functable->Z2_group_weight" << sfx << "=calloc(" << wgt.size() << ",sizeof(double));" << std::endl;
			cleanup << " pyoomph_tested_free(functable->Z2_group_weight" << sfx << "); functable->Z2_group_weight" << sfx << "=PYOOMPH_NULL; " << std::endl;
			for (unsigned int g = 0; g < normrel.size(); g++)
			{
				init << "  functable->Z2_group_normalize_relative" << sfx << "[" << g << "] = " << std::to_string(normrel[g]) << ";" << std::endl;
				init << "  functable->Z2_group_weight" << sfx << "[" << g << "] = " << std::to_string(wgt[g]) << ";" << std::endl;
			}
		};
		write_group_arrays(false);
		write_group_arrays(true);

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
	  std::map<FiniteElementField*, unsigned,FiniteElementFieldPtrLess> contribution_field_to_index;
	  std::map<FiniteElementField*,FiniteElementField*,FiniteElementFieldPtrLess> to_where_it_was_defined;
	  // The name of the contribution CLASS a field belongs to: the domain it is DEFINED on plus its
	  // name. Deliberately not the code the field is seen from -- an interface or facet element refers
	  // to bulk fields, and those must land in the same class as in the bulk code, or the two
	  // descriptions of the same dof would disagree. Used both to register the classes below and to
	  // resolve field_contribution_index further down; they must not drift apart, which is also why the
	  // block-flag analysis above names its row/column roles through the very same function.
	  // Captures "this" so the class name is resolved relative to the code being generated, which is
	  // what splits the far side of an interior facet into its own class. Every consumer below (the
	  // registration loop, class_index_of for the block flags, and the field_contribution_index name
	  // fallback) goes through this one lambda, so they cannot disagree about the naming.
	  auto contribution_class_name = [this](FiniteElementField *f) -> std::string { return block_contribution_class_name(f, this); };
	  for (auto *f : contributing_fields)
	  {
		FiniteElementField *wheredef = f->get_defined_on_domain_equivalent_field();
		to_where_it_was_defined[f] = wheredef;
		std::string n=contribution_class_name(f);
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
	  init << " functable->contributes_to_mass_matrix=(bool***)calloc(functable->num_res_jacs,sizeof(*functable->contributes_to_mass_matrix));" << std::endl;
	  init << " functable->contributes_to_hessian=(bool***)calloc(functable->num_res_jacs,sizeof(*functable->contributes_to_hessian));" << std::endl;
	  init << " functable->jacobian_block_flags=(unsigned char***)calloc(functable->num_res_jacs,sizeof(*functable->jacobian_block_flags));" << std::endl;
	  init << " functable->mass_matrix_block_flags=(unsigned char***)calloc(functable->num_res_jacs,sizeof(*functable->mass_matrix_block_flags));" << std::endl;
	  init << " functable->contribution_entries_size=" << contribution_names.size() << ";" << std::endl;

	  // Does anything in reach of this code move its nodes? Then dx, the normal, the element size and
	  // every shape derivative depend on unknowns, and hardly any block can be constant. Taken over the
	  // bulk/opposite-interface relatives as well, exactly like the coordinate-shape injection in
	  // write_generic_RJM_jacobian_contribution().
	  bool moving_for_flags = false;
	  for (FiniteElementCode *c = this; c; c = c->get_bulk_element())
		 if (c->coordinates_as_dofs) moving_for_flags = true;
	  for (FiniteElementCode *c = this->get_opposite_interface_code(); c; c = c->get_bulk_element())
		 if (c->coordinates_as_dofs) moving_for_flags = true;
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
				init << " functable->contributes_to_mass_matrix[" << resiind << "]=(bool**)calloc("<< contribution_names.size() <<",sizeof(**functable->contributes_to_mass_matrix));" << std::endl;				
				init << " for (unsigned int _i=0;_i<"<< contribution_names.size() <<";_i++) { functable->contributes_to_mass_matrix[" << resiind << "][_i]=(bool*)calloc("<< contribution_names.size() <<",sizeof(bool)); }" << std::endl;				
				init << " functable->contributes_to_hessian[" << resiind << "]=(bool**)calloc("<< contribution_names.size() <<",sizeof(**functable->contributes_to_hessian));" << std::endl;
				init << " for (unsigned int _i=0;_i<"<< contribution_names.size() <<";_i++) { functable->contributes_to_hessian[" << resiind << "][_i]=(bool*)calloc("<< contribution_names.size() <<",sizeof(bool)); }" << std::endl;
				init << " functable->jacobian_block_flags[" << resiind << "]=(unsigned char**)calloc("<< contribution_names.size() <<",sizeof(**functable->jacobian_block_flags));" << std::endl;
				init << " for (unsigned int _i=0;_i<"<< contribution_names.size() <<";_i++) { functable->jacobian_block_flags[" << resiind << "][_i]=(unsigned char*)calloc("<< contribution_names.size() <<",sizeof(unsigned char)); }" << std::endl;
				init << " functable->mass_matrix_block_flags[" << resiind << "]=(unsigned char**)calloc("<< contribution_names.size() <<",sizeof(**functable->mass_matrix_block_flags));" << std::endl;
				init << " for (unsigned int _i=0;_i<"<< contribution_names.size() <<";_i++) { functable->mass_matrix_block_flags[" << resiind << "][_i]=(unsigned char*)calloc("<< contribution_names.size() <<",sizeof(unsigned char)); }" << std::endl;

				std::vector<bool> written_residual_contribution(contribution_names.size(), false);
				std::vector<std::vector<bool>> written_jacobian_contribution(contribution_names.size(), std::vector<bool>(contribution_names.size(), false));
				std::vector<std::vector<bool>> written_mass_matrix_contribution(contribution_names.size(), std::vector<bool>(contribution_names.size(), false));
				std::vector<std::vector<bool>> written_hessian_contribution(contribution_names.size(), std::vector<bool>(contribution_names.size(), false));
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
						if ((f->has_hessian_contribution_for_code(this,resiind,f2) || f->has_hessian_contribution_for_code(this,resiind,pair2.second)) && !written_hessian_contribution[i1][i2])
						{
							init << " functable->contributes_to_hessian[" << resiind << "][" << i1 << "][" << i2 << "]=true; //" << contribution_names[i1] << " vs " << contribution_names[i2] << std::endl;
							written_hessian_contribution[i1][i2] = true;
						}
						if ((f->has_mass_matrix_contribution_for_code(this,resiind,f2) || f->has_mass_matrix_contribution_for_code(this,resiind,pair2.second)) && !written_mass_matrix_contribution[i1][i2])
						{
							init << " functable->contributes_to_mass_matrix[" << resiind << "][" << i1 << "][" << i2 << "]=true; //" << contribution_names[i1] << " vs " << contribution_names[i2] << std::endl;
							written_mass_matrix_contribution[i1][i2] = true;
						}
					}
				}

				// The JACOBIAN_BLOCK_* tables. The recorded per-field blocks are first SUMMED into their
				// contribution class pair - the expression-level counterpart of the boolean OR above.
				// Summing is what makes the comparison exact (ANDing the individual terms would be sound
				// but would lose every cancellation) and makes it immune to the different orderings of
				// jacobian_fields and present_tests.
				if (jacobian_block_exprs.count(resiind))
				{
					auto class_index_of = [&](FiniteElementField *f) -> int
					{
						auto it = contribution_field_to_index.find(f);
						if (it != contribution_field_to_index.end()) return (int)it->second;
						auto alias = to_where_it_was_defined.find(f);
						if (alias != to_where_it_was_defined.end())
						{
							auto it2 = contribution_field_to_index.find(alias->second);
							if (it2 != contribution_field_to_index.end()) return (int)it2->second;
						}
						auto it3 = contribution_name_to_index.find(contribution_class_name(f));
						if (it3 != contribution_name_to_index.end()) return (int)it3->second;
						return -1;
					};
					std::map<std::pair<int, int>, std::pair<GiNaC::ex, GiNaC::ex>> class_blocks;
					for (auto &entry : jacobian_block_exprs[resiind])
					{
						int i1 = class_index_of(entry.first.first), i2 = class_index_of(entry.first.second);
						if (i1 < 0 || i2 < 0) continue;
						auto &dst = class_blocks[std::make_pair(i1, i2)];
						dst.first += entry.second.first;
						dst.second += entry.second.second;
					}

					std::vector<std::vector<unsigned char>> jflags(contribution_names.size(), std::vector<unsigned char>(contribution_names.size(), 0));
					std::vector<std::vector<unsigned char>> mflags = jflags;
					for (auto &b : class_blocks)
					{
						// A block that is identically zero is not present at all: its flags stay 0 and
						// the consumer is told to check contributes_to_* first.
						if (!b.second.first.is_zero())
							jflags[b.first.first][b.first.second] |= block_constancy_flags(b.second.first, moving_for_flags);
						if (!b.second.second.is_zero())
							mflags[b.first.first][b.first.second] |= block_constancy_flags(b.second.second, moving_for_flags);
					}
					for (auto &b : class_blocks)
					{
						int i1 = b.first.first, i2 = b.first.second;
						if (i2 < i1) continue; // the mirror pair is handled from its lower triangle entry
						auto mirror = class_blocks.find(std::make_pair(i2, i1));
						// An off-diagonal block that WAS recorded is not identically zero, so it cannot be
						// the (anti)transpose of an absent one.
						if (i1 != i2 && mirror == class_blocks.end()) continue;
						const std::pair<GiNaC::ex, GiNaC::ex> &other = (i1 == i2 ? b.second : mirror->second);
						if (!b.second.first.is_zero() && !other.first.is_zero())
						{
							unsigned char f = block_symmetry_flags(b.second.first, other.first, this);
							jflags[i1][i2] |= f;
							jflags[i2][i1] |= f;
						}
						if (!b.second.second.is_zero() && !other.second.is_zero())
						{
							unsigned char f = block_symmetry_flags(b.second.second, other.second, this);
							mflags[i1][i2] |= f;
							mflags[i2][i1] |= f;
						}
					}

					auto flag_expr = [](unsigned char f) -> std::string
					{
						std::string res;
						if (f & JACOBIAN_BLOCK_SYMMETRIC) res += (res.empty() ? "" : "|") + std::string("JACOBIAN_BLOCK_SYMMETRIC");
						if (f & JACOBIAN_BLOCK_ANTISYMMETRIC) res += (res.empty() ? "" : "|") + std::string("JACOBIAN_BLOCK_ANTISYMMETRIC");
						if (f & JACOBIAN_BLOCK_CONSTANT) res += (res.empty() ? "" : "|") + std::string("JACOBIAN_BLOCK_CONSTANT");
						if (f & JACOBIAN_BLOCK_CONSTANT_FIXED_DT) res += (res.empty() ? "" : "|") + std::string("JACOBIAN_BLOCK_CONSTANT_FIXED_DT");
						return res;
					};
					for (unsigned int i1 = 0; i1 < contribution_names.size(); i1++)
					{
						for (unsigned int i2 = 0; i2 < contribution_names.size(); i2++)
						{
							// The comment carries the class NAMES only - printing the block expression
							// would allocate global parameter slots as a side effect.
							if (jflags[i1][i2])
								init << " functable->jacobian_block_flags[" << resiind << "][" << i1 << "][" << i2 << "]=" << flag_expr(jflags[i1][i2]) << "; //" << contribution_names[i1] << " vs " << contribution_names[i2] << std::endl;
							if (mflags[i1][i2])
								init << " functable->mass_matrix_block_flags[" << resiind << "][" << i1 << "][" << i2 << "]=" << flag_expr(mflags[i1][i2]) << "; //" << contribution_names[i1] << " vs " << contribution_names[i2] << std::endl;
						}
					}
				}

				cleanup << " for (unsigned int _i=0;_i<functable->contribution_entries_size;_i++) { pyoomph_tested_free(functable->jacobian_block_flags[" << resiind << "][_i]); functable->jacobian_block_flags[" << resiind << "][_i]=PYOOMPH_NULL; }" << std::endl;
				cleanup << " pyoomph_tested_free(functable->jacobian_block_flags[" << resiind << "]); functable->jacobian_block_flags[" << resiind << "]=PYOOMPH_NULL; " << std::endl;
				cleanup << " for (unsigned int _i=0;_i<functable->contribution_entries_size;_i++) { pyoomph_tested_free(functable->mass_matrix_block_flags[" << resiind << "][_i]); functable->mass_matrix_block_flags[" << resiind << "][_i]=PYOOMPH_NULL; }" << std::endl;
				cleanup << " pyoomph_tested_free(functable->mass_matrix_block_flags[" << resiind << "]); functable->mass_matrix_block_flags[" << resiind << "]=PYOOMPH_NULL; " << std::endl;
				cleanup << " for (unsigned int _i=0;_i<functable->contribution_entries_size;_i++) { pyoomph_tested_free(functable->contributes_to_hessian[" << resiind << "][_i]); functable->contributes_to_hessian[" << resiind << "][_i]=PYOOMPH_NULL; }" << std::endl;
				cleanup << " pyoomph_tested_free(functable->contributes_to_hessian[" << resiind << "]); functable->contributes_to_hessian[" << resiind << "]=PYOOMPH_NULL; " << std::endl;
				cleanup << " for (unsigned int _i=0;_i<functable->contribution_entries_size;_i++) { pyoomph_tested_free(functable->contributes_to_jacobian[" << resiind << "][_i]); functable->contributes_to_jacobian[" << resiind << "][_i]=PYOOMPH_NULL; }" << std::endl;
				cleanup << " pyoomph_tested_free(functable->contributes_to_jacobian[" << resiind << "]); functable->contributes_to_jacobian[" << resiind << "]=PYOOMPH_NULL; " << std::endl;
				cleanup << " for (unsigned int _i=0;_i<functable->contribution_entries_size;_i++) { pyoomph_tested_free(functable->contributes_to_mass_matrix[" << resiind << "][_i]); functable->contributes_to_mass_matrix[" << resiind << "][_i]=PYOOMPH_NULL; }" << std::endl;
				cleanup << " pyoomph_tested_free(functable->contributes_to_mass_matrix[" << resiind << "]); functable->contributes_to_mass_matrix[" << resiind << "]=PYOOMPH_NULL; " << std::endl;
				cleanup << " pyoomph_tested_free(functable->contributes_to_residual[" << resiind << "]); functable->contributes_to_residual[" << resiind << "]=PYOOMPH_NULL; " << std::endl;
	  
	  	}

	   }
	  // The outer arrays are calloc'd above regardless of whether any contribution classes exist, so
	  // their frees must be emitted unconditionally as well - inside the block above they leaked for
	  // codes with zero contribution classes.
	  cleanup << " pyoomph_tested_free(functable->contributes_to_residual); functable->contributes_to_residual=PYOOMPH_NULL; " << std::endl;
	  cleanup << " pyoomph_tested_free(functable->contributes_to_jacobian); functable->contributes_to_jacobian=PYOOMPH_NULL; " << std::endl;
	  cleanup << " pyoomph_tested_free(functable->contributes_to_hessian); functable->contributes_to_hessian=PYOOMPH_NULL; " << std::endl;
	  cleanup << " pyoomph_tested_free(functable->contributes_to_mass_matrix); functable->contributes_to_mass_matrix=PYOOMPH_NULL; " << std::endl;
	  cleanup << " pyoomph_tested_free(functable->jacobian_block_flags); functable->jacobian_block_flags=PYOOMPH_NULL; " << std::endl;
	  cleanup << " pyoomph_tested_free(functable->mass_matrix_block_flags); functable->mass_matrix_block_flags=PYOOMPH_NULL; " << std::endl;
	  jacobian_block_exprs.clear(); // the flag tables are emitted; holding on to the expressions only costs memory

	  // Emit, per space, the map from that space's field slot to the contribution index used by
	  // contributes_to_jacobian / contributes_to_mass_matrix. The elements need it to translate a LOCAL
	  // DOF (which they know as "field f of space s at node n") into the row/column class those tables
	  // are indexed by, and hence to decide which entries of their dense elemental block are
	  // structurally present. -2 means the field is known to take part in NO contribution of this code,
	  // so its row and column of the elemental block are empty; -1 (never written here) is reserved for
	  // "not attributed", which the reader must treat as coupled to everything.
	  // Cannot be emitted next to the field names above: the contribution indices are only assigned
	  // here, from the set of fields that turned out to contribute.
	  {
		// The arrays themselves are allocated next to fieldnames above, with the space's own field count,
		// so only the values are filled in here.
		for (auto &slot : field_contribution_slots)
		{
			FiniteElementField *f = std::get<2>(slot);
			int idx = -1;
			auto it = contribution_field_to_index.find(f);
			if (it != contribution_field_to_index.end()) idx = (int)it->second;
			else
			{
				// contributing_fields holds the field object as the RESIDUALS refer to it, which for an
				// interface or facet code is the bulk code's object; field_contribution_slots holds this
				// code's own object for the same dof. Those are different pointers, so the lookup above
				// misses and the index used to stay at the -2 sentinel -- "contributes to nothing" -- for
				// every field of every facet element. That made the whole elemental block read as
				// structurally empty, which the frozen-sparsity verification then (correctly) refused:
				// "A Jacobian entry appeared outside the symbolic sparsity pattern". Matching on the
				// contribution class instead is what the coupling table itself is keyed by.
				auto nit = contribution_name_to_index.find(contribution_class_name(f));
				if (nit != contribution_name_to_index.end()) idx = (int)nit->second;
			}
			if (idx < 0) continue; // Genuinely part of no contribution of this code: stays -2
			init << " functable->" << std::get<0>(slot) << ".field_contribution_index[" << std::get<1>(slot)
				 << "]=" << idx << "; //" << contribution_names[idx] << std::endl;
		}
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
		
		// domain_name is deliberately NOT set here (nor freed in cleanup below): baking a
		// per-domain string into the generated C code made otherwise textually-identical
		// codes on different domains (e.g. a DirichletBC(u=0) on both "left" and "right" of a
		// rectangle) compile to different .c/.so content, defeating the JIT code cache's
		// ability to share one compiled artifact between them. It is instead set by the host
		// (DynamicJITCode's constructor in problem.cpp, right after calling this
		// init function) and freed there too - see the comment there for why that keeps
		// alloc/free symmetric regardless of the compiler backend's C runtime.
		init << " functable->clean_up=&clean_up;" << std::endl;
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
		for (auto &shape_exp : se->req_fields)
		{
			if (shape_exp.field->get_symbol() == s)
			{
				if (shape_exp.time_history_index != 0)
					continue; // Only with respect to the actual time
				if (pyoomph::__in_hessian && !pyoomph::__derive_shapes_by_second_index)
				{
					// Outer Hessian index. The cached scalar d_subexpr_N_d_<field> is a plain GiNaC symbol
					// and cannot be differentiated a second time, so instead of reading it we wrap
					// d(body)/d(field) in a *nested* subexpression: its own first-derivative cache, filled
					// by write_code_subexpressions like any other, then is the second derivative. That is
					// what makes a separate d2_subexpr_N_d_f_d_g cache unnecessary.
					//
					// The wrapped body has to come out as a loop-index-independent scalar, exactly as in
					// the derivative fill of write_code_subexpressions: __deriv_subexpression_wrto restricts
					// the raw diff to this one (field, basis, dt) combination, and DerivedShapeExpansionsToUnity
					// divides the derived shape expansion back out - it is re-applied on the outside below.
					// Without both, the shape function would end up inside a value hoisted above the shape
					// loop, and every req_fields entry of the same field would add the whole sum again.
					// Save and restore rather than clearing: diff() re-enters this branch through nested
					// subexpressions, so a NULL here would corrupt the enclosing differentiation.
					const pyoomph::ShapeExpansion *outer_wrto = pyoomph::__deriv_subexpression_wrto;
					pyoomph::__deriv_subexpression_wrto = &shape_exp;
					GiNaC::ex inner = GiNaC::diff(get_struct().expr, s);
					pyoomph::__deriv_subexpression_wrto = outer_wrto;
					pyoomph::DerivedShapeExpansionsToUnity strip_derived(shape_exp.basis, shape_exp.dt_order, shape_exp.dt_scheme);
					inner = strip_derived(inner);
					if (inner.is_zero())
						continue; // nothing to cache, and wrapping it would emit a dead subexpr_N
					GiNaC::ex newse = (*get_struct().code->se_to_struct_hessian)(pyoomph::expressions::subexpression(inner));
					auto sexp = pyoomph::ShapeExpansion(shape_exp.field, shape_exp.dt_order, shape_exp.basis, shape_exp.dt_scheme, true);
					res += newse * GiNaCShapeExpansion(sexp);
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
				// The direct derivative is the complete one - GiNaCShapeExpansion::derivative already
				// turns every position expansion inside the body into its derived form - so whatever the
				// req_fields loop above put into res is the *same* contribution, only re-expressed
				// through the cached d_subexpr_N_d_<coordinate> scalar. res is therefore dropped rather
				// than added: returning res+deriv would count the coordinate term twice. That is also why
				// those cached coordinate derivatives are emitted and never read (see the dead-store
				// section of dev_docs/code_generation.md).
				//
				// This used to assemble an error message claiming a non-zero deriv with found==true was
				// impossible, print deriv as "should be 0", and then not throw it. found is in fact
				// reachable: SubExpressionsToStructs erases position expansions from req_fields, but that
				// erasure does not cover the opposite interface code, which the is_coordinate test above
				// does cover.
				GiNaC::ex deriv = GiNaC::diff(get_struct().expr, s);
				if (!deriv.is_zero())
					return deriv;
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
				// An Eulerian element size evaluated in the past is the size of the element where it was,
				// hence the history index (0 = current configuration). The Lagrangian sizes need none.
				std::string es_hist = "[0]";
				if (get_struct().history_step > 0 && femprint.FEM_opts->for_code->history_geometry_is_relevant())
					es_hist = "[" + std::to_string(get_struct().history_step) + "]";
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
						c.s << shapeinfo_str << "elemsize_Eulerian" << es_hist;
					}
					else
					{
						c.s << shapeinfo_str << "elemsize_Eulerian_cartesian" << es_hist;
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
		std::string hist_str = (get_struct().history_step ? (" | HIST " + std::to_string(get_struct().history_step)) : "");
		if (get_struct().is_lagrangian())
		{
			c.s << "<Elemsize Lagrangian " << (get_struct().is_with_coordsys() ? "with coordsys" : "cartesian") << ">";
		}
		else
		{
			if (get_struct().is_derived())
			{

				c.s << "<Elemsize Eulerian " << (get_struct().is_with_coordsys() ? "with coordsys" : "cartesian") << hist_str;

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
				c.s << "<Elemsize Eulerian" << hist_str << ">";
			}
		}
	}
	// Mirrors GiNaCSpatialIntegralSymbol::derivative above for element-size symbols: Lagrangian
	// element size never depends on the moving mesh; otherwise, differentiating w.r.t. one of this
	// domain's coordinate_{x,y,z} symbols (when coordinates_as_dofs) yields the matching pre-built
	// "derived" (or twice-derived) ElementSizeSymbol; any other symbol yields 0.
	// The prebuilt derived element-size symbols are shared, so the expansion-mode tag of the symbol
	// being differentiated has to be stamped onto a copy rather than onto the original.
	static pyoomph::ElementSizeSymbol _tag_elemsize_mode(const pyoomph::ElementSizeSymbol &src, int mode)
	{
		pyoomph::ElementSizeSymbol res = src;
		res.expansion_mode = mode;
		return res;
	}

	template <>
	GiNaC::ex GiNaCElementSizeSymbol::derivative(const GiNaC::symbol &s) const
	{
		if (get_struct().is_lagrangian())
			return 0;
		// The size of a PREVIOUS configuration is built from nodal positions that are history values,
		// not from the current unknowns, so it contributes nothing to the Jacobian (and hence nothing
		// to the mass matrix extracted from it). Same rule as for history-tagged dx and for shape
		// expansions with time_history_index != 0.
		if (get_struct().history_step > 0)
			return 0;
		pyoomph::FiniteElementCode *code = (pyoomph::FiniteElementCode *)(get_struct().get_code()); // Cast aways the constness
		pyoomph::FiniteElementField *testf;
		if (!code->coordinates_as_dofs || pyoomph::ignore_nodal_position_derivatives_for_pitchfork_symmetry())
			return 0;
		// Same expansion-mode rule as for shape expansions, normals and dx: a derivative only belongs to
		// the family of position dofs this symbol is tagged with. An already-derived size is a function
		// of the base geometry alone, so - as for the normal - it stays differentiable.
		//
		// The flag is needed on top of the tag because of WHEN the symbol appears: the element size
		// enters a residual as var("cartesian_element_length_h"), a placeholder that is only resolved
		// into this symbol while the code is generated, i.e. long after eval_at_expansion_mode() has
		// walked the expression. So the mapper cannot tag it there, and the flag says what an untagged
		// size means: expanded (differentiated in every pass, the exact derivative of the discrete
		// residual, what bifurcation tracking needs) or frozen at the base state.
		if (!pyoomph::expand_element_size_in_expansion_modes && pyoomph::__derive_only_by_expansion_mode &&
			get_struct().expansion_mode != *pyoomph::__derive_only_by_expansion_mode && !get_struct().is_derived())
			return 0;
		// TODO: Other spaces, e.g. bulk
		if (!get_struct().is_derived())
		{
			testf = code->get_field_by_name("coordinate_x");
			if (testf && s == testf->get_symbol())
			{
				return 0 + GiNaCElementSizeSymbol(_tag_elemsize_mode(code->get_elemsize_derived(0, get_struct().is_with_coordsys()), get_struct().expansion_mode));
			}
			testf = code->get_field_by_name("coordinate_y");
			if (testf && s == testf->get_symbol())
			{
				return 0 + GiNaCElementSizeSymbol(_tag_elemsize_mode(code->get_elemsize_derived(1, get_struct().is_with_coordsys()), get_struct().expansion_mode));
			}
			testf = code->get_field_by_name("coordinate_z");
			if (testf && s == testf->get_symbol())
			{
				return 0 + GiNaCElementSizeSymbol(_tag_elemsize_mode(code->get_elemsize_derived(2, get_struct().is_with_coordsys()), get_struct().expansion_mode));
			}
		}
		else if (!get_struct().is_derived2())
		{
			int dir1 = get_struct().get_derived_direction();
			testf = code->get_field_by_name("coordinate_x");
			if (testf && s == testf->get_symbol())
			{
				return 0 + GiNaCElementSizeSymbol(_tag_elemsize_mode(code->get_elemsize_derived2(dir1, 0, get_struct().is_with_coordsys()), get_struct().expansion_mode));
			}
			testf = code->get_field_by_name("coordinate_y");
			if (testf && s == testf->get_symbol())
			{
				return 0 + GiNaCElementSizeSymbol(_tag_elemsize_mode(code->get_elemsize_derived2(dir1, 1, get_struct().is_with_coordsys()), get_struct().expansion_mode));
			}
			testf = code->get_field_by_name("coordinate_z");
			if (testf && s == testf->get_symbol())
			{
				return 0 + GiNaCElementSizeSymbol(_tag_elemsize_mode(code->get_elemsize_derived2(dir1, 2, get_struct().is_with_coordsys()), get_struct().expansion_mode));
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
				// A normal evaluated in the past is the normal of that configuration, hence the leading
				// history index (0 = current configuration).
				std::string n_hist = "[0]";
				if (sp.history_step > 0 && femprint.FEM_opts->for_code->history_geometry_is_relevant())
					n_hist = "[" + std::to_string(sp.history_step) + "]";
				// A spatially derived normal reads from the parallel dnormal_dx family; the
				// nodal-coordinate derivative index means exactly the same thing in both.
				const std::string base = (sp.spatial_deriv_direction >= 0 ? "dnormal_dx" : "normal");
				const std::string spatial_idx = (sp.spatial_deriv_direction >= 0 ? "[" + std::to_string(sp.spatial_deriv_direction) + "]" : "");
				if (sp.get_derived_direction() == -1)
				{
					c.s << prefix << base << n_hist << "[" << sp.get_direction() << "]" << spatial_idx;
				}
				else if (sp.get_derived_direction2() == -1)
				{
					c.s << prefix << "d_" << base << "_dcoord[" << sp.get_direction() << "]" << spatial_idx << "[" << (sp.is_derived_by_lshape2() ? "l_shape2" : "l_shape") << "][" << sp.get_derived_direction() << "]";
				}
				else
				{
					c.s << prefix << "d2_" << base << "_d2coord[" << sp.get_direction() << "]" << spatial_idx << "[l_shape][" << sp.get_derived_direction() << "][l_shape2][" << sp.get_derived_direction2() << "]";
				}
				return;
			}
		}
		std::string expansion_mode_str = (sp.expansion_mode != 0 ? "| MODE " + std::to_string(sp.expansion_mode) : "");
		if (sp.get_derived_direction() == -1)
		{
			c.s << "<" <<  "NORMAL COMPONENT " << sp.get_direction() << " @ " << sp.get_code() << expansion_mode_str << (sp.history_step ? (" | HIST " + std::to_string(sp.history_step)) : "") << ">";
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
        
		// First EULERIAN spatial derivative: dn_i/dx_j, i.e. minus the Weingarten map. See
		// fill_shape_info_at_s for the formula. Second spatial derivatives are not implemented.
		if (s == pyoomph::expressions::x || s == pyoomph::expressions::y || s == pyoomph::expressions::z)
		{
			const pyoomph::NormalSymbol &sp = get_struct();
			if (sp.spatial_deriv_direction >= 0)
				throw_runtime_error("Second spatial derivatives of the normal are not implemented, i.e. grad(grad(normal)) or div(grad(normal)).");
			if (sp.get_derived_direction() != -1)
				throw_runtime_error("Cannot take a spatial derivative of a normal that has already been derived with respect to the nodal coordinates.");
			if (sp.history_step > 0)
				return 0; // the geometry of a past configuration does not depend on the current unknowns
			pyoomph::NormalSymbol nret = sp;
			nret.spatial_deriv_direction = (s == pyoomph::expressions::x ? 0 : (s == pyoomph::expressions::y ? 1 : 2));
			return GiNaC::GiNaCNormalSymbol(nret);
		}
		if (s == pyoomph::expressions::t)
		{
			throw_runtime_error("Cannot derive the normal with respect to time yet. Use partial_t(...) of the mesh positions instead.");
		}
		if (s == pyoomph::expressions::X || s == pyoomph::expressions::Y || s == pyoomph::expressions::Z)
		{
			throw_runtime_error("Cannot derive the normal with respect to the Lagrangian coordinates yet. Only the Eulerian derivative, e.g. grad(normal) or div(normal), is available.");
		}
		// Local element coordinates would otherwise fall through to the "return 0" at the very end of
		// this function. That was unreachable while every spatial derivative threw above, but is now
		// reached through AxisymmetryBreakingCoordinateSystem::vector_divergence, which contains
		// diff(arg[i], local_coordinate_1) terms - and a silent zero there is a plausible-looking
		// wrong eigenvalue rather than an error.
		if (s == pyoomph::expressions::local_coordinate_1 || s == pyoomph::expressions::local_coordinate_2 || s == pyoomph::expressions::local_coordinate_3)
		{
			throw_runtime_error("Cannot derive the normal with respect to the local element coordinates yet. Only the Eulerian derivative, e.g. grad(normal) or div(normal), is available.");
		}
		else
		{

			const pyoomph::NormalSymbol &sp = get_struct();
			// A normal of a previous configuration depends on history positions only, never on the
			// current unknowns -- see the note in GiNaCElementSizeSymbol::derivative.
			if (sp.history_step > 0)
				return 0;
			// The expansion-mode tag says which family of position dofs the FIRST derivative refers to:
			// mode 1 is the eigenfunction's mesh perturbation, mode 0 the base state. Once the normal
			// has been derived by a nodal coordinate, what is left - dn/dX - is a function of the base
			// geometry alone and carries the tag only as an inherited label, so the Hessian's second
			// derivative (taken under mode 0) has to be let through. Vetoing it dropped every
			// d2_normal_d2coord term from the Hessian of the azimuthal contributions: the generated
			// HessianVectorProduct for the m!=0 residual had all 32 dnormal terms but not one d2normal.
			// That is what made azimuthal bifurcation tracking diverge on moving meshes with a free
			// surface (the only place a normal enters), while the base-state Hessian stayed correct.
			if (pyoomph::__derive_only_by_expansion_mode && sp.expansion_mode != *pyoomph::__derive_only_by_expansion_mode && sp.get_derived_direction() == -1)
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
								nret.spatial_deriv_direction = sp.spatial_deriv_direction; // carry the spatial index into the coordinate-derived symbol
								return GiNaCNormalSymbol(nret);
							}
							else
							{
								pyoomph::NormalSymbol nret(pyoomph::__current_code, sp.get_direction(), sp.get_derived_direction(), coord_dir, pyoomph::__derive_shapes_by_second_index);
								nret.no_jacobian = sp.no_jacobian;
								nret.no_hessian = sp.no_hessian;
								nret.expansion_mode = sp.expansion_mode;
								nret.spatial_deriv_direction = sp.spatial_deriv_direction; // carry the spatial index into the coordinate-derived symbol
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
								nret.spatial_deriv_direction = sp.spatial_deriv_direction; // carry the spatial index into the coordinate-derived symbol
								return GiNaCNormalSymbol(nret);
							}
							else
							{
								pyoomph::NormalSymbol nret(pyoomph::__current_code, sp.get_direction(), sp.get_derived_direction(), coord_dir, pyoomph::__derive_shapes_by_second_index);
								nret.no_jacobian = sp.no_jacobian;
								nret.no_hessian = sp.no_hessian;
								nret.expansion_mode = sp.expansion_mode;
								nret.spatial_deriv_direction = sp.spatial_deriv_direction; // carry the spatial index into the coordinate-derived symbol
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
								nret.spatial_deriv_direction = sp.spatial_deriv_direction; // carry the spatial index into the coordinate-derived symbol
								return GiNaCNormalSymbol(nret);
							}
							else
							{
								pyoomph::NormalSymbol nret(pyoomph::__current_code->get_bulk_element(), sp.get_direction(), sp.get_derived_direction(), coord_dir, pyoomph::__derive_shapes_by_second_index);
								nret.no_jacobian = sp.no_jacobian;
								nret.no_hessian = sp.no_hessian;
								nret.expansion_mode = sp.expansion_mode;
								nret.spatial_deriv_direction = sp.spatial_deriv_direction; // carry the spatial index into the coordinate-derived symbol
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
								nret.spatial_deriv_direction = sp.spatial_deriv_direction; // carry the spatial index into the coordinate-derived symbol
								return GiNaCNormalSymbol(nret);
							}
							else
							{
								pyoomph::NormalSymbol nret(pyoomph::__current_code->get_bulk_element(), sp.get_direction(), sp.get_derived_direction(), coord_dir, pyoomph::__derive_shapes_by_second_index);
								nret.no_jacobian = sp.no_jacobian;
								nret.no_hessian = sp.no_hessian;
								nret.expansion_mode = sp.expansion_mode;
								nret.spatial_deriv_direction = sp.spatial_deriv_direction; // carry the spatial index into the coordinate-derived symbol
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
								nret.spatial_deriv_direction = sp.spatial_deriv_direction; // carry the spatial index into the coordinate-derived symbol
								return GiNaCNormalSymbol(nret);
							}
							else
							{
								pyoomph::NormalSymbol nret(pyoomph::__current_code->get_opposite_interface_code(), sp.get_direction(), sp.get_derived_direction(), coord_dir, pyoomph::__derive_shapes_by_second_index);
								nret.no_jacobian = sp.no_jacobian;
								nret.no_hessian = sp.no_hessian;
								nret.expansion_mode = sp.expansion_mode;
								nret.spatial_deriv_direction = sp.spatial_deriv_direction; // carry the spatial index into the coordinate-derived symbol
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
								nret.spatial_deriv_direction = sp.spatial_deriv_direction; // carry the spatial index into the coordinate-derived symbol
								return GiNaCNormalSymbol(nret);
							}
							else
							{
								pyoomph::NormalSymbol nret(pyoomph::__current_code->get_opposite_interface_code(), sp.get_direction(), sp.get_derived_direction(), coord_dir, pyoomph::__derive_shapes_by_second_index);
								nret.no_jacobian = sp.no_jacobian;
								nret.no_hessian = sp.no_hessian;
								nret.expansion_mode = sp.expansion_mode;
								nret.spatial_deriv_direction = sp.spatial_deriv_direction; // carry the spatial index into the coordinate-derived symbol
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
						c.s << alias_timestepper_weight0(femprint.FEM_opts->for_code, "d2t", timedisc_scheme) << "*";
					else if (sp.dt_order == 1)
					{
						c.s << alias_timestepper_weight0(femprint.FEM_opts->for_code, "dt", timedisc_scheme) << "*";
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
							std::string shapename = dcoord_shape_array(sp.basis, false);
							// Second site that emits a rank-4 read; marks through the same channel as the
							// interpolation loop so the flag stays a superset of the reads.
							if (!femprint.FEM_opts->for_code->current_shapeflag_func_type.empty())
								femprint.FEM_opts->for_code->mark_shapes_required(femprint.FEM_opts->for_code->current_shapeflag_func_type, const_cast<pyoomph::FiniteElementSpace *>(sp.basis->get_space()), "dx_psi_dcoord");
							const std::string dirstr = dcoord_shape_index(sp.basis);
							if (sp.nodal_coord_dir >= 0)
							{
								std::string shapestr = femprint.FEM_opts->for_code->get_shape_info_str(sp.basis->get_space()) + "->" + shapename;
								shapestr += "[l_shape2][" + dirstr + "][l_shape][" + std::to_string(sp.nodal_coord_dir) + "]";
								c.s << shapestr;
							}
							else
							{
								std::string shapestr = femprint.FEM_opts->for_code->get_shape_info_str(sp.basis->get_space()) + "->" + shapename;
								shapestr += "[l_shape][" + dirstr + "][" + "l_shape2" + "][" + std::to_string(sp.nodal_coord_dir2) + "]";
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
		c.s << "<" << (sp.is_derived ? (sp.is_derived_other_index ? "ALT.DERIVED " : "DERIVED ") : "") << (sp.nodal_coord_dir == -1 ? "" : "COORDINATE_DIFF_" + std::to_string(sp.nodal_coord_dir) + " ") << "SHAPEEXP of " << dt << sp.field->get_name() << " of " << sp.field->get_space()->get_code()->get_domain_name() << " @ " << sp.basis->to_string() << (sp.no_jacobian ? " | NO_JACOBIAN" : "") << (sp.no_hessian ? " | NO_HESSIAN" : "") << (sp.expansion_mode ? (" | MODE " + std::to_string(sp.expansion_mode)) : "") << (sp.time_history_index ? (" | HIST " + std::to_string(sp.time_history_index)) : "") << (sp.history_geometry ? " | HISTGEOM" : "") << ">";
	}


	template <>
	GiNaC::ex GiNaCShapeExpansion::derivative(const GiNaC::symbol &s) const
	{
		const pyoomph::ShapeExpansion &sp = get_struct();
		// sname is only consulted in the nodal-position branches far below, but this runs once per
		// shape-expansion node per GiNaC::diff() - so building the ostringstream up front cost a
		// stream and a heap string per node, almost always discarded. Computed on demand instead.
		std::string sname;
		bool sname_valid = false;
		auto get_sname = [&]() -> const std::string &
		{
			if (!sname_valid)
			{
				std::ostringstream oss;
				oss << s;
				sname = oss.str();
				sname_valid = true;
			}
			return sname;
		};
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
			int coord_dir = (get_sname() == "coordinate_x" ? 0 : (get_sname() == "coordinate_y" ? 1 : 2));
			if (sp.basis->is_eulerian_deriv())
			{
				if (get_sname() == "coordinate_x" || get_sname() == "coordinate_y" || get_sname() == "coordinate_z")
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
				if (sp.dt_order && sp.basis->is_eulerian_deriv())
				{
					std::ostringstream ossn;
					ossn << s;
					std::string sname = ossn.str();
					// Only a NODAL COORDINATE can contribute the moving-mesh dpsi/dX term. This branch
					// is reached whenever we differentiate by the field's OWN symbol, and for anything
					// that is not a position field -- a potential, a concentration, ... -- that symbol
					// is not a coordinate at all. The ternary below has no such case and fell through
					// to direction 2, so the Jacobian asked for a COORDDIFF_2 array that the
					// declaration loop (which runs to nodal_dimension()) never declares. That made
					// partial_t(grad(u)) of a bulk field, evaluated on an interface, fail to compile
					// with "COORDDIFF_2_u undeclared" - in every dimension and coordinate system, on a
					// static mesh as well as a moving one. The residual was always correct; only the
					// Jacobian carried the bogus term.
					// The dimension check covers the same invariant from the other side: a direction
					// the declaration loop does not reach cannot be referenced here either.
					if (sname != "coordinate_x" && sname != "coordinate_y" && sname != "coordinate_z")
						return GiNaCShapeExpansion(se);
					int coord_dir = (sname == "coordinate_x" ? 0 : (sname == "coordinate_y" ? 1 : 2));
					if (coord_dir >= (int)sp.basis->get_space()->get_code()->nodal_dimension())
						return GiNaCShapeExpansion(se);
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
					if (sp.dt_order && sp.basis->is_eulerian_deriv())
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

			int coord_dir = (get_sname() == "coordinate_x" ? 0 : (get_sname() == "coordinate_y" ? 1 : 2));
			if (sp.basis->is_eulerian_deriv())
			{
				//		   std::cout << "  hit else case "  << (*this) << "  " << sp.field->get_name() <<  "   WRT " << s << " sname " << sname << " dir " << coord_dir<<  std::endl;
				if (get_sname() == "coordinate_x" || get_sname() == "coordinate_y" || get_sname() == "coordinate_z")
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
				const std::string shapename = dcoord_shape_array(sp.basis, false);
				const std::string shapename2 = dcoord_shape_array(sp.basis, true);
				if (sp.nodal_coord_dir == -1)
				{
					c.s << sp.basis->get_c_varname(femprint.FEM_opts->for_code, "l_test");
				}
				else if (sp.nodal_coord_dir2 == -1)
				{
					// Third site emitting a rank-4 read (test-function side); same channel again. Only
					// THIS branch reads d_dx_shape_dcoord - the one above reads no sensitivity at all and
					// the one below reads the rank-6 array, so marking before the branch over-requests
					// and gives the space back its full fill for nothing.
					if (!femprint.FEM_opts->for_code->current_shapeflag_func_type.empty())
						femprint.FEM_opts->for_code->mark_shapes_required(femprint.FEM_opts->for_code->current_shapeflag_func_type, const_cast<pyoomph::FiniteElementSpace *>(sp.basis->get_space()), "dx_psi_dcoord");
					std::string shapestr = femprint.FEM_opts->for_code->get_shape_info_str(sp.basis->get_space()) + "->"+ shapename;
					shapestr += "[l_test][" + dcoord_shape_index(sp.basis) + "][" + (sp.is_derived_other_index ? "l_shape2" : "l_shape") + "][" + std::to_string(sp.nodal_coord_dir) + "]";
					c.s << shapestr;
				}
				else
				{
					std::string shapestr = femprint.FEM_opts->for_code->get_shape_info_str(sp.basis->get_space()) + "->"+ shapename2;
					shapestr += "[l_test][" + dcoord_shape_index(sp.basis) + "][l_shape][" + std::to_string(sp.nodal_coord_dir) + "][l_shape2][" + std::to_string(sp.nodal_coord_dir2) + "]";
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
			if (sp.basis->is_eulerian_deriv())
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
