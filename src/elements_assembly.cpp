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


// Residual, Jacobian, mass-matrix and Hessian assembly: the generic JIT contribution entry points,
// the multi-assembly driver, the finite-difference fallbacks for nodal and Lagrangian positions,
// and the analytic-vs-FD debugging comparisons.

#include <array>
#include <map>
#include <set>

#include "macroelements.hpp"
#include "elements.hpp"
#include "exception.hpp"
#include "problem.hpp"
#include "nodes.hpp"
#include "meshtemplate.hpp"
#include "expressions.hpp"
#include "timestepper.hpp"

namespace pyoomph
{

	// Per-element hanging/non-hanging dispatch (dev_docs/code_generation.md 9.4.14).
	//
	// PYOOMPH_DISABLE_NOHANG_DISPATCH forces every element through the hanging entry point again, on an
	// unchanged binary and unchanged generated code. That is what makes the specialised bodies testable:
	// the two arms of the comparison differ in nothing but which entry point was called.
	static const bool __nohang_dispatch_disabled = getenv("PYOOMPH_DISABLE_NOHANG_DISPATCH") != NULL;

	// PYOOMPH_REPORT_NOHANG_DISPATCH counts which entry point each element assembly took and prints the
	// tally at exit. Diagnostic only. It exists for the same reason as the ALE-identity report in
	// elements_shapeinfo.cpp: a comparison in which the specialised path never engaged produces two
	// identical numbers and looks like a perfect result, so the engagement has to be counted rather
	// than inferred.
	static const bool __report_nohang_dispatch = getenv("PYOOMPH_REPORT_NOHANG_DISPATCH") != NULL;
	// Diagnostic only (see the note in elements_hanging.cpp): process-wide and not thread-safe.
	static unsigned long __nohang_dispatch_count = 0, __hang_dispatch_count = 0;
	static struct __NoHangDispatchReport
	{
		~__NoHangDispatchReport()
		{
			if (!__report_nohang_dispatch)
				return;
			std::cout << "PYOOMPH_REPORT_NOHANG_DISPATCH: " << __nohang_dispatch_count
					  << " element assemblies took the NoHang entry point, " << __hang_dispatch_count
					  << " took the hanging one";
			if (!__nohang_dispatch_count)
				std::cout << " - the specialised path NEVER engaged, so nothing about it was proven";
			std::cout << std::endl;
		}
	} __nohang_dispatch_report;

	// PYOOMPH_REPORT_EXT_DATA builds a per-element-code histogram of (nexternal_data, external dofs,
	// ndof), sampled once per element at its first residual/Jacobian assembly. External data attached
	// per element CODE from merged_required_shapes (the OR over every contribution, including integral
	// and output expressions) inflates ndof, and the dense element block is ndof^2 - so this is what
	// says whether that inflation is worth attacking.
	static const bool __report_ext_data = getenv("PYOOMPH_REPORT_EXT_DATA") != NULL;
	static std::map<std::string, std::map<std::array<unsigned, 3>, unsigned long>> __ext_data_hist;
	static std::set<const void *> __ext_data_seen;
	static struct __ExtDataReport
	{
		~__ExtDataReport()
		{
			if (!__report_ext_data)
				return;
			if (__ext_data_hist.empty())
			{
				std::cout << "PYOOMPH_REPORT_EXT_DATA: no element was ever assembled - nothing was sampled" << std::endl;
				return;
			}
			for (const auto &c : __ext_data_hist)
			{
				unsigned long nel = 0, sum_ndof = 0, sum_extdof = 0;
				for (const auto &e : c.second)
				{
					nel += e.second;
					sum_ndof += (unsigned long)e.first[2] * e.second;
					sum_extdof += (unsigned long)e.first[1] * e.second;
				}
				std::cout << "PYOOMPH_REPORT_EXT_DATA: " << c.first << ": " << nel << " elements, "
						  << sum_extdof << " of " << sum_ndof << " local dofs come from external data ("
						  << (sum_ndof ? 100.0 * sum_extdof / sum_ndof : 0.0) << "%)" << std::endl;
				for (const auto &e : c.second)
					std::cout << "    nexternal_data=" << e.first[0] << " external_dofs=" << e.first[1]
							  << " ndof=" << e.first[2] << " : " << e.second << " elements" << std::endl;
			}
		}
	} __ext_data_report;

	void BulkElementBase::__sample_ext_data_stats()
	{
		if (!__report_ext_data)
			return;
		if (!__ext_data_seen.insert((const void *)this).second)
			return;
		unsigned next = this->nexternal_data(), extdofs = 0;
		for (unsigned ed = 0; ed < next; ed++)
		{
			oomph::Data *d = this->external_data_pt(ed);
			for (unsigned i = 0; i < d->nvalue(); i++)
				if (this->external_local_eqn(ed, i) >= 0)
					extdofs++;
		}
		__ext_data_hist[std::string(jitcode->get_file_name())]
					   [{next, extdofs, (unsigned)this->ndof()}]++;
	}

	// Whether this element gets the specialised, hanging-node-free entry point. has_hang is what
	// fill_hang_info_with_equations reported, i.e. "some dof of this element really hangs, or an
	// additional dof constraint reduced it".
	static inline bool __use_nohang_entry(bool has_hang)
	{
		const bool nohang = !has_hang && !__nohang_dispatch_disabled;
		if (__report_nohang_dispatch)
		{
			if (nohang) __nohang_dispatch_count++;
			else __hang_dispatch_count++;
		}
		return nohang;
	}

	// Whether contribution `c` really has a SEPARATE NoHang entry point. When the code generator found
	// no use for pyoomph_hang_on in the body (codegen.cpp, hang_parameter_was_used) it emits only one
	// routine and the function table aliases both slots - the "NoHang" call is then the ordinary body,
	// which still loads the hang buffers. Stage 3a's skip is keyed on this pointer, i.e. on the very
	// thing that selects the entry point, so the two cannot drift apart.
	static inline bool __has_separate_nohang(const JITFuncSpec_Table_FiniteElement_t *ft, int c, bool steady)
	{
		JITFuncSpec_ResidualAndJacobian_FiniteElement *const hang_tab = (steady ? ft->ResidualAndJacobianSteady : ft->ResidualAndJacobian);
		JITFuncSpec_ResidualAndJacobian_FiniteElement *const nohang_tab = (steady ? ft->ResidualAndJacobianSteady_NoHang : ft->ResidualAndJacobian_NoHang);
		if (!hang_tab || !nohang_tab)
			return false;
		return nohang_tab[c] && nohang_tab[c] != hang_tab[c];
	}

	// Multi-assembly: performs several residual/Jacobian/mass-matrix/Hessian-vector-product/
	// parameter-derivative contributions (described by `info`, one entry per "contribution",
	// e.g. bulk + several attached interfaces) for this element in a single pass, sharing one
	// shape_info fill per integration point instead of recomputing shapes from scratch for each
	// contribution separately. First merges the RequiredShapes of all requested contributions
	// (so the shape buffer is filled generously enough for all of them at once) and determines
	// the maximum needed shapeflag (0=residuals,1=+Jacobian,2=+mass matrix,3=+Hessian). Then
	// loops over integration points (or just once, if the code was compiled with a private,
	// non-shared shape buffer per contribution) and, for each requested contribution, calls the
	// appropriate JIT-generated residual/Jacobian/mass-matrix function, parameter-derivative
	// function, and/or Hessian-vector-product function.
	void BulkElementBase::get_multi_assembly(std::vector<SinglePassMultiAssembleInfo> &info)
	{
		JITShapeInfo_t *const shape_info = this->get_shape_info();
		JITFuncSpec_Table_FiniteElement_t *functable = jitcode->get_func_table();
		JITFuncSpec_RequiredShapes_FiniteElement_t *required_shapes = (JITFuncSpec_RequiredShapes_FiniteElement_t *)std::calloc(1, sizeof(JITFuncSpec_RequiredShapes_FiniteElement_t));
		int shapeflag = -1;
		// std::cout << "MERGED ASSEMBLY " << std::endl;
		// First pass: merge required shapes across all contributions and figure out the highest
		// shapeflag (residuals/Jacobian/mass-matrix/Hessian) needed overall.
		for (auto &inf : info)
		{
			if (inf.contribution < 0)
				continue;
			bool resjac_merged = false;
			if (inf.residuals || inf.jacobian || inf.mass_matrix)
			{
				resjac_merged = true;
				if (functable->fd_jacobian || functable->fd_position_jacobian)
					throw_runtime_error("Multi-assembly does not work with fd_jacobian or fd_position_jacobian");
				//    std::cout << "  MERGED ResJac " << inf.contribution << std::endl;
				RequiredShapes_merge(&functable->shapes_required_ResJac[inf.contribution], required_shapes);
			}
			if (inf.residuals)
				shapeflag = 0;
			if (inf.jacobian && shapeflag < 1)
				shapeflag = 1;
			if (inf.mass_matrix && shapeflag < 2)
				shapeflag = 2;
			if (inf.hessians.size())
			{
				RequiredShapes_merge(&functable->shapes_required_Hessian[inf.contribution], required_shapes);
				shapeflag = 3;
				//    std::cout << "  MERGED HEssian " << inf.contribution << std::endl;
			}
			if (functable->ParameterDerivative)
			{
				for (auto &pdiff : inf.dparams)
				{
					unsigned global_param_index = jitcode->get_problem()->resolve_parameter_value_ptr(pdiff.parameter);
					int paramindex = -1;
					for (unsigned int i = 0; i < functable->numglobal_params; i++)
					{
						if (functable->global_paramindices[i] == global_param_index)
						{
							paramindex = i;
							break;
						}
					}
					if (paramindex >= 0)
					{
						if (functable->ParameterDerivative[inf.contribution] && functable->ParameterDerivative[inf.contribution][paramindex])
						{
							if (!resjac_merged && (pdiff.dRdparam || pdiff.dJdparam || pdiff.dMdparam))
							{
								resjac_merged = true;
								if (functable->fd_jacobian || functable->fd_position_jacobian)
									throw_runtime_error("Multi-assembly does not work with fd_jacobian or fd_position_jacobian");
								//            std::cout << "  MERGED dParamResJac " << inf.contribution <<"   " << paramindex << "  " << pdiff.parameter << std::endl;
								RequiredShapes_merge(&functable->shapes_required_ResJac[inf.contribution], required_shapes);
							}
							if (pdiff.dMdparam && shapeflag < 2)
								shapeflag = 2;
							else if (pdiff.dJdparam && shapeflag < 1)
								shapeflag = 1;
							else if (pdiff.dRdparam && shapeflag < 0)
								shapeflag = 0;
						}
					}
				}
			}
		}
		// std::cout << " SHAPEFLAG " << shapeflag << std::endl;
		if (shapeflag < 0)
		{
			RequiredShapes_free(required_shapes);
			return; // Nothing to assemble at all
		}

		// This is the only benefit of this approach: We only have to do this once!
		// Elements in which nothing hangs take the specialised entry points below. The answer comes
		// from the cheap predicate rather than from the fill's return value, because Stage 3a needs it
		// BEFORE the fill in order to skip it.
		this->__sample_ext_data_stats();
		const bool has_hang = this->hang_fill_would_report_hang(*required_shapes);
		const bool use_nohang = __use_nohang_entry(has_hang);
		// Stage 3a. The fill is only skippable if EVERY function this pass is about to call is a
		// separately compiled NoHang body: those have pyoomph_hang_on folded to 0 and provably never
		// load a hang buffer. Parameter derivatives and Hessian-vector products have no NoHang twin at
		// all (only ResidualAndJacobian[Steady] do, see jitbridge.h), so a pass carrying one of those
		// keeps the fill unconditional.
		bool may_skip_hang_fill = use_nohang && !__disable_hang_fill_cache;
		if (may_skip_hang_fill)
		{
			int steady0 = -1; // lazily resolved: the timestepper lookup below assumes nodes or internal data
			for (auto &inf : info)
			{
				if (inf.contribution < 0)
					continue;
				if (inf.dparams.size() || inf.hessians.size())
				{
					may_skip_hang_fill = false;
					break;
				}
				if (inf.residuals || inf.jacobian || inf.mass_matrix)
				{
					if (steady0 < 0)
					{
						const oomph::TimeStepper *ts0 = (this->nnode() ? this->node_pt(0)->time_stepper_pt() : this->internal_data_pt(0)->time_stepper_pt());
						steady0 = (ts0->is_steady() ? 1 : 0);
					}
					if (!__has_separate_nohang(functable, inf.contribution, steady0 == 1))
					{
						may_skip_hang_fill = false;
						break;
					}
				}
			}
		}
		if (may_skip_hang_fill)
		{
			if (__paranoid_hang_fill_cache)
			{
				const bool real = this->fill_hang_info_with_equations(*required_shapes, shape_info, NULL);
				if (real)
					throw_runtime_error("PYOOMPH_PARANOID_HANG_FILL_CACHE: the hang predicate said 'nothing hangs' but fill_hang_info_with_equations reported a hang (multi-assembly)");
				this->poison_hang_info(shape_info);
			}
			if (__report_hang_fill_cache)
				__hang_fill_cache_count(HANGCACHE_FILL_SKIPPED);
		}
		else
		{
			this->fill_hang_info_with_equations(*required_shapes, shape_info, NULL);
			if (__report_hang_fill_cache)
				__hang_fill_cache_count(HANGCACHE_FILL_RUN);
		}
		this->interpolate_hang_values();
		prepare_shape_buffer_for_integration(*required_shapes, shapeflag);
      bool shared_multi_assemble=functable->use_shared_shape_buffer_during_multi_assemble;
      shape_info->during_shared_multi_assembling=shared_multi_assemble;
      unsigned n_int_pt=(shared_multi_assemble ? shape_info->n_int_pt : 1);
      for (unsigned int i_int_pt=0;i_int_pt<n_int_pt;i_int_pt++)
      {

			for (auto &inf : info)
			{
				if (inf.contribution < 0)
					continue;
				// Fill the shape buffer once per integration point, shared across all contributions
				// (only if the code table allows sharing; otherwise each contribution's function
				// call internally handles filling the buffer for all points itself).
				if (shared_multi_assemble)
				{
				  this->fill_shape_buffer_for_integration_point(i_int_pt,*required_shapes,shapeflag);
				}

				// Base contribution: residuals / Jacobian / mass matrix
				if (inf.residuals || inf.jacobian || inf.mass_matrix)
				{
					JITFuncSpec_ResidualAndJacobian_FiniteElement func;
					const oomph::TimeStepper *tstepper = (this->nnode() ? this->node_pt(0)->time_stepper_pt() : this->internal_data_pt(0)->time_stepper_pt());

					if (tstepper->is_steady())
					{
						func = (use_nohang ? functable->ResidualAndJacobianSteady_NoHang[inf.contribution]
										   : functable->ResidualAndJacobianSteady[inf.contribution]);
					}
					else
					{
						func = (use_nohang ? functable->ResidualAndJacobian_NoHang[inf.contribution]
										   : functable->ResidualAndJacobian[inf.contribution]);
					}
					if (func)
					{
						if (inf.mass_matrix) // residuals, Jacobian, Mass matrix
						{
							if (!inf.jacobian || !inf.residuals)
								throw_runtime_error("Cannot multiassemble a mass matrix without setting Jacobian and residual (possibly dummies)");
							shape_info->jacobian_size = inf.jacobian->nrow();
							shape_info->mass_matrix_size = inf.mass_matrix->nrow();
							//             std::cout << " AEESMBLE RJM " << inf.contribution << std::endl;
							func(&eleminfo, shape_info, &(((*inf.residuals)[0])), &(inf.jacobian->entry(0, 0)), &(inf.mass_matrix->entry(0, 0)), 2);
						}
						else if (inf.jacobian) // residuals, Jacobian
						{
							if (!inf.residuals)
								throw_runtime_error("Cannot multiassemble a Jacobian without setting residual (possibly dummy)");
							//             std::cout << " AEESMBLE RJ " << inf.contribution << std::endl;
							shape_info->jacobian_size = inf.jacobian->nrow();
							func(&eleminfo, shape_info, &(((*inf.residuals)[0])), &(inf.jacobian->entry(0, 0)), NULL, 1);
						}
						else if (inf.residuals)
						{
							//             std::cout << " AEESMBLE R " << inf.contribution << std::endl;
							func(&eleminfo, shape_info, &(((*inf.residuals)[0])), NULL, NULL, 0);
						}
					}
				}

				// Parameter derivatives
				if (functable->ParameterDerivative)
				{
					for (auto &pinf : inf.dparams)
					{
						if (!functable->ParameterDerivative[inf.contribution])
							continue;
						unsigned global_param_index = jitcode->get_problem()->resolve_parameter_value_ptr(pinf.parameter);
						int paramindex = -1;
						for (unsigned int i = 0; i < functable->numglobal_params; i++)
						{
							if (functable->global_paramindices[i] == global_param_index)
							{
								paramindex = i;
								break;
							}
						}
						if (paramindex < 0)
							continue;
						if (!functable->ParameterDerivative[inf.contribution][paramindex])
							continue;
						if (pinf.dMdparam) // residuals, Jacobian, Mass matrix
						{
							if (!pinf.dJdparam || !pinf.dRdparam)
								throw_runtime_error("Cannot multiassemble a mass matrix without setting Jacobian and residual (possibly dummies). Happens in parameter derivative");
							//             std::cout << " AEESMBLE PARAMDERIV RJM " << inf.contribution << "  " << paramindex << "  " << pinf.parameter << std::endl;
							shape_info->jacobian_size = pinf.dJdparam->nrow();
							shape_info->mass_matrix_size = pinf.dMdparam->nrow();
							functable->ParameterDerivative[inf.contribution][paramindex](&eleminfo, shape_info, &(((*pinf.dRdparam)[0])), &(pinf.dJdparam->entry(0, 0)), &(pinf.dMdparam->entry(0, 0)), 2);
						}
						else if (pinf.dJdparam) // residuals, Jacobian
						{
							if (!pinf.dRdparam)
								throw_runtime_error("Cannot multiassemble a Jacobian without setting residual (possibly dummy). Happens in parameter derivative");
							//             std::cout << " AEESMBLE PARAMDERIV RJ " << inf.contribution << "  " << paramindex << "  " << pinf.parameter << std::endl;
							shape_info->jacobian_size = pinf.dJdparam->nrow();
							functable->ParameterDerivative[inf.contribution][paramindex](&eleminfo, shape_info, &(((*pinf.dRdparam)[0])), &(pinf.dJdparam->entry(0, 0)), NULL, 1);
						}
						else if (pinf.dRdparam)
						{
							//             std::cout << " AEESMBLE PARAMDERIV R " << inf.contribution << "  " << paramindex << "  " << pinf.parameter << std::endl;
							functable->ParameterDerivative[inf.contribution][paramindex](&eleminfo, shape_info, &(((*pinf.dRdparam)[0])), NULL, NULL, 0);
						}
					}
				}

				// Hessians
				if (inf.hessians.size())
				{
					if (!functable->hessian_generated)
						throw_runtime_error("You want to calculate Hessian contributions, but analytical Hessian were not set. Please call problem.setup_for_stability_analysis(analytic_hessian=True) before just-in-time compilation");

					for (auto &hinf : inf.hessians)
					{
						if (!functable->HessianVectorProduct || !functable->HessianVectorProduct[inf.contribution])
							continue;
						if (!hinf.M_Hessian && !hinf.J_Hessian)
							continue;
						if (hinf.M_Hessian && !hinf.J_Hessian)
							throw_runtime_error("You want to calculate Hessian mass contributions, but you must set a potentially dummy Hessian Jacobian.");
						unsigned n_var = hinf.Y.size();
						unsigned n_vec = hinf.J_Hessian->ncol();
						if (n_var%n_vec!=0) throw_runtime_error("Y and Hessian must fulfill #Y modulo ncol(H) =0. Thereby, you can assembly multiple vectors products at once");
						shape_info->jacobian_size = n_vec;
						n_vec=n_var/n_vec;
						//std::cout << "NVEC " << n_vec << std::endl;
						if (hinf.M_Hessian)
						{
							//             std::cout << " AEESMBLE HESS JM " << inf.contribution << "  " << &hinf.Y << "  " <<  std::endl;
							functable->HessianVectorProduct[inf.contribution](&eleminfo, shape_info, &hinf.Y[0], &(hinf.M_Hessian->entry(0, 0)), &(hinf.J_Hessian->entry(0, 0)), n_vec, (hinf.transposed ? 5:  2));
						}
						else
						{
							//            std::cout << " AEESMBLE HESS J " << inf.contribution << "  " << &hinf.Y << "  " <<  std::endl;
							functable->HessianVectorProduct[inf.contribution](&eleminfo, shape_info, &hinf.Y[0], NULL, &(hinf.J_Hessian->entry(0, 0)), n_vec, (hinf.transposed ? 4:  1));
						}
					}
				}
			}
		}
      shape_info->during_shared_multi_assembling=false;
		RequiredShapes_free(required_shapes);
	}

	///\short Compute the derivatives of the
	/// residuals with respect to a parameter
	/// Flag=1 (or 0): do (or don't) compute the Jacobian as well.
	/// Flag=2: Fill in mass matrix too.
	void BulkElementBase::fill_in_generic_dresidual_contribution_jit(double *const &parameter_pt, oomph::Vector<double> &dres_dparam, oomph::DenseMatrix<double> &djac_dparam, oomph::DenseMatrix<double> &dmass_matrix_dparam, unsigned flag)
	{
		JITShapeInfo_t *const shape_info = this->get_shape_info();

		const JITFuncSpec_Table_FiniteElement_t *functable = jitcode->get_func_table();
		const int __crj = get_current_res_jac(functable); // hoisted: read once per element, see thread_state.hpp
		if (__crj < 0)
			return;
		if (!functable->ParameterDerivative)
			return;
		if (!functable->ParameterDerivative[__crj])
			return;
		unsigned global_param_index = jitcode->get_problem()->resolve_parameter_value_ptr(parameter_pt);
		int paramindex = -1;
		for (unsigned int i = 0; i < functable->numglobal_params; i++)
		{
			if (functable->global_paramindices[i] == global_param_index)
			{
				paramindex = i;
				break;
			}
		}
		if (paramindex < 0)
			return; // Nothing to do -> Element does not depend on this parameter
		if (!functable->ParameterDerivative[__crj][paramindex])
			return;
		// Unconditional, unlike the two residual/Jacobian paths: ParameterDerivative has no _NoHang
		// twin in the function table, so the body that runs here always loads the hang buffers.
		this->fill_hang_info_with_equations(functable->shapes_required_ResJac[__crj], shape_info, NULL);
		this->interpolate_hang_values(); // XXX This should be moved to somewhere else, after each update of any values
		prepare_shape_buffer_for_integration(functable->shapes_required_ResJac[__crj], flag);
		shape_info->jacobian_size = djac_dparam.nrow();
		shape_info->mass_matrix_size = dmass_matrix_dparam.nrow();

		if (!functable->ParameterDerivative[__crj][paramindex])
			return;
		if (flag)
		{
			if (flag >= 2) // residuals, Jacobian, Mass matrix
			{
				functable->ParameterDerivative[__crj][paramindex](&eleminfo, shape_info, &(dres_dparam[0]), &(djac_dparam.entry(0, 0)), &(dmass_matrix_dparam.entry(0, 0)), flag);
			}
			else // residuals, Jacobian
			{
				functable->ParameterDerivative[__crj][paramindex](&eleminfo, shape_info, &(dres_dparam[0]), &(djac_dparam.entry(0, 0)), NULL, flag);
			}
		}
		else // Only residuals
		{
			functable->ParameterDerivative[__crj][paramindex](&eleminfo, shape_info, &(dres_dparam[0]), NULL, NULL, flag);
		}
	}

	// The main JIT-driven residual/Jacobian/mass-matrix assembly entry point, called from
	// oomph-lib's fill_in_contribution_to_* machinery (flag: 0=residuals only, 1=+Jacobian,
	// >=2=+mass matrix). Two special redirections take priority over normal assembly: if
	// the Problem's replace_RJM_by_param_deriv is set (used while computing derivatives w.r.t. a parameter
	// via finite differences elsewhere), delegates to fill_in_generic_dresidual_contribution_jit
	// instead; if enable_zeta_projection is set, assembles the (unrelated) zeta-projection
	// residuals instead of the physical equations. Otherwise prepares the shape buffer, resolves
	// hanging-node equation info, and calls the appropriate JIT-generated ResidualAndJacobian
	// function (steady/unsteady, hanging/non-hanging variant). The hanging/non-hanging half of that
	// choice is a real one since 9.4.14 - it used to be hard-wired to "hanging" for every element.
	void BulkElementBase::fill_in_generic_residual_contribution_jit(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian, oomph::DenseMatrix<double> &mass_matrix, unsigned flag)
	{
		JITShapeInfo_t *const shape_info = this->get_shape_info();
		if (double *param_deriv = jitcode->get_problem()->get_replace_RJM_by_param_deriv())
		{
			fill_in_generic_dresidual_contribution_jit(param_deriv, residuals, jacobian, mass_matrix, flag);
			return;
		}

		if (this->enable_zeta_projection)
		{
			// Deliberately NOT cleared here. It used to be, which meant the projection residual was
			// assembled exactly once per element and every subsequent assembly of the same element -
			// the second Newton iteration, the residual-only pass, a re-solve - silently returned to
			// the physical equations. The projection driver clears it when it is done.
			residuals_for_zeta_projection(residuals, jacobian, flag);
			return;
		}

		const JITFuncSpec_Table_FiniteElement_t *functable = jitcode->get_func_table();
		const int __crj = get_current_res_jac(functable); // hoisted: read once per element, see thread_state.hpp
		if (__crj < 0)
			return;
		if (!this->ndof())
			return;

		if (!functable->ResidualAndJacobian[__crj])
			return;
		this->__sample_ext_data_stats();
		prepare_shape_buffer_for_integration(functable->shapes_required_ResJac[__crj], flag);
		shape_info->jacobian_size = jacobian.nrow();
		shape_info->mass_matrix_size = mass_matrix.nrow();
		// The predicate is the real answer ("some dof of this element hangs, a dof constraint reduced
		// it, or an attached element's equations are remapped through the same buffers"); it used to be
		// the fill's return value, but Stage 3a needs the answer BEFORE the fill in order to skip it.
		const bool has_hang = this->hang_fill_would_report_hang(functable->shapes_required_ResJac[__crj]);
		const bool use_nohang = __use_nohang_entry(has_hang);

		JITFuncSpec_ResidualAndJacobian_FiniteElement func;
		const oomph::TimeStepper *tstepper = (this->nnode() ? this->node_pt(0)->time_stepper_pt() : this->internal_data_pt(0)->time_stepper_pt());
		if (tstepper->is_steady())
		{
			func = (use_nohang ? functable->ResidualAndJacobianSteady_NoHang[__crj]
							   : functable->ResidualAndJacobianSteady[__crj]);
		}
		else
		{
			func = (use_nohang ? functable->ResidualAndJacobian_NoHang[__crj]
							   : functable->ResidualAndJacobian[__crj]);
		}
		// Stage 3a: derived from the very pointer selected just above, so the skip and the dispatch
		// cannot diverge. A separately compiled _NoHang body has pyoomph_hang_on folded to 0 and
		// therefore never loads a hang buffer (jitbridge.h, "Hanging macros"); where codegen emitted no
		// such twin the two table slots alias and the fill stays.
		if (use_nohang && !__disable_hang_fill_cache &&
			__has_separate_nohang(functable, __crj, tstepper->is_steady()))
		{
			if (__paranoid_hang_fill_cache)
			{
				const bool real = this->fill_hang_info_with_equations(functable->shapes_required_ResJac[__crj], shape_info, NULL);
				if (real)
					throw_runtime_error("PYOOMPH_PARANOID_HANG_FILL_CACHE: the hang predicate said 'nothing hangs' but fill_hang_info_with_equations reported a hang");
				this->poison_hang_info(shape_info);
			}
			if (__report_hang_fill_cache)
				__hang_fill_cache_count(HANGCACHE_FILL_SKIPPED);
		}
		else
		{
			this->fill_hang_info_with_equations(functable->shapes_required_ResJac[__crj], shape_info, NULL);
			if (__report_hang_fill_cache)
				__hang_fill_cache_count(HANGCACHE_FILL_RUN);
		}
		this->interpolate_hang_values(); // XXX This should be moved to somewhere else, after each update of any values
		// std::cout << "RESIDUAL LENGTH  " << residuals.size() << "  " << this->nexternal_data() <<  std::endl;

		if (flag)
		{
			if (flag >= 2) // residuals, Jacobian, Mass matrix
			{
				/* for (unsigned int i=0;i<mass_matrix.nrow();i++)
					 for (unsigned int j=0;j<mass_matrix.nrow();j++) if (mass_matrix(i,j)!=0.0) std::cout << "PREE " << mass_matrix(i,j) << std::endl ;
			*/
				func(&eleminfo, shape_info, &(residuals[0]), &(jacobian.entry(0, 0)), &(mass_matrix.entry(0, 0)), flag);
				/* for (unsigned int i=0;i<mass_matrix.nrow();i++)
					 for (unsigned int j=0;j<mass_matrix.nrow();j++) if (std::fabs(mass_matrix(i,j))>100.0) std::cout << "POST " << mass_matrix(i,j) << std::endl ;*/
			}
			else // residuals, Jacobian
			{
				func(&eleminfo, shape_info, &(residuals[0]), &(jacobian.entry(0, 0)), NULL, flag);
			}
		}
		else // Only residuals
		{
			func(&eleminfo, shape_info, &(residuals[0]), NULL, NULL, flag);
		}
		/*
		 std::cout << "C JACO " << std::endl;
		 for (unsigned int i=0;i<jacobian.nrow();i++)
		 {
			 for (unsigned int j=0;j<jacobian.ncol();j++) std::cout << "\t" << jacobian.entry(i,j) ;
			std::cout << std::endl;
		 }
		*/
	}

	// Debug helper: compares the analytical Hessian-vector product (Y,C) -> product against a
	// finite-difference approximation obtained by perturbing the dofs along each C vector and
	// re-assembling the Jacobian, to validate the JIT-generated analytical Hessian code.
	//
	// The reference is the *analytical* Jacobian, so this is only as good as that Jacobian - use it
	// together with Problem.debug_jacobian_by_fd_epsilon, which validates the Jacobian itself against
	// finite differences of the residual. The two together pin down the whole chain.
	//
	// Returns the largest absolute discrepancy found. Not used in production assembly.
	//
	// Call it through Problem.debug_analytic_hessian_by_fd() rather than directly: the
	// finite-difference half perturbs the dofs through dof_pt_vector(), and oomph-lib only fills the
	// Dof_pt array behind it when Problem::enable_store_local_dof_pt_in_elements() was switched on
	// before the equation numbers were assigned. Its own check for that sits inside #ifdef PARANOID,
	// so in this build the omission is a segfault rather than a message, and Dof_pt is private so it
	// cannot be probed from here.
	double BulkElementBase::debug_hessian(std::vector<double> Y, std::vector<std::vector<double>> C, double epsilon)
	{
		if (Y.size() != this->ndof())
			throw_runtime_error("Y vector is wrong in size " + std::to_string(Y.size()) + "  vs.  " + std::to_string(this->ndof()));
		if (!C.size())
			throw_runtime_error("Empty C matrix");
		oomph::Vector<double> Ys(Y.size());
		for (unsigned int i = 0; i < Ys.size(); i++)
			Ys[i] = Y[i];
		oomph::DenseMatrix<double> Cs(C.size(), C[0].size());
		for (unsigned int iv = 0; iv < C.size(); iv++)
		{
			if (C[iv].size() != this->ndof())
				throw_runtime_error("C vector entry " + std::to_string(iv) + " has wrong size");
			for (unsigned int id = 0; id < this->ndof(); id++)
				Cs(iv, id) = C[iv][id];
		}
		oomph::DenseMatrix<double> anaprod(C.size(), this->ndof(), 0.0);
		this->fill_in_contribution_to_hessian_vector_products(Ys, Cs, anaprod);

		// Now FDing it
		oomph::DenseMatrix<double> fdprod(C.size(), this->ndof(), 0.0);
		oomph::Vector<double> dummy_res(this->ndof());
		oomph::DenseMatrix<double> jac_base(this->ndof());
		this->get_jacobian(dummy_res, jac_base);
		oomph::Vector<double *> dof_pt;
		this->dof_pt_vector(dof_pt);
		oomph::Vector<double> dofbackup(dof_pt.size());
		for (unsigned int i = 0; i < dof_pt.size(); i++)
			dofbackup[i] = *(dof_pt[i]);

		////////////////////

		const double FD_step = 1.0e-7;

		// We can now construct our multipliers
		// Prepare to scale
		double dof_length = 0.0;
		oomph::Vector<double> C_length(C.size(), 0.0);

		for (unsigned n = 0; n < this->ndof(); n++)
		{
			if (std::fabs(dofbackup[n]) > dof_length)
			{
				dof_length = std::fabs(dofbackup[n]);
			}
		}

		// C is assumed to have the same distribution as the dofs
		for (unsigned i = 0; i < C.size(); i++)
		{
			for (unsigned n = 0; n < this->ndof(); n++)
			{
				if (std::fabs(C[i][n]) > C_length[i])
				{
					C_length[i] = std::fabs(C[i][n]);
				}
			}
		}
		///////////////////////////////7
		// Form the multipliers
		oomph::Vector<double> C_mult(C.size(), 0.0);
		for (unsigned i = 0; i < C.size(); i++)
		{
			C_mult[i] = dof_length / C_length[i];
			C_mult[i] += FD_step;
			C_mult[i] *= FD_step;
		}

		// Central difference along each C direction. A one-sided difference against jac_base is only
		// first order accurate, which on a moving mesh leaves a residue around 1e-4 - the same order
		// as a real error - and makes the check useless there.
		for (unsigned i = 0; i < C.size(); i++)
		{
			oomph::DenseMatrix<double> jac_plus(this->ndof()), jac_minus(this->ndof());
			for (unsigned n = 0; n < this->ndof(); n++)
				*dof_pt[n] += C_mult[i] * C[i][n];
			this->get_jacobian(dummy_res, jac_plus);
			for (unsigned n = 0; n < this->ndof(); n++)
				*(dof_pt[n]) = dofbackup[n] - C_mult[i] * C[i][n];
			this->get_jacobian(dummy_res, jac_minus);
			for (unsigned n = 0; n < this->ndof(); n++)
				*(dof_pt[n]) = dofbackup[n];

			for (unsigned n = 0; n < this->ndof(); n++)
			{
				double prod_c = 0.0;
				for (unsigned m = 0; m < this->ndof(); m++)
				{
					prod_c += (jac_plus(n, m) - jac_minus(n, m)) * Y[m];
				}
				fdprod(i, n) += prod_c / (2.0 * C_mult[i]);
			}
		}

		// Report a RELATIVE discrepancy: the products themselves range over many orders of magnitude
		// across elements and dof pairs, so an absolute threshold would either drown in noise or miss
		// errors in the small entries.
		double worst = 0.0;
		for (unsigned int iv = 0; iv < C.size(); iv++)
		{
			bool Cheader_written = false;
			for (unsigned int id = 0; id < this->ndof(); id++)
			{
				const double scale = std::max(1.0, std::max(std::fabs(anaprod(iv, id)), std::fabs(fdprod(iv, id))));
				const double delta = std::fabs(fdprod(iv, id) - anaprod(iv, id)) / scale;
				if (delta > worst)
					worst = delta;
				if (epsilon <= 0 || delta > epsilon)
				{
					if (!Cheader_written)
					{
						std::cout << "  FOR C VECTOR " << iv << " : ";
						for (unsigned k = 0; k < C[iv].size(); k++)
							std::cout << C[iv][k] << "  ";
						std::cout << std::endl;
						Cheader_written = true;
					}
					std::cout << "     COMPONENT " << id << " : FD: " << fdprod(iv, id) << " ANA: " << anaprod(iv, id) << " REL.DELTA: " << delta << std::endl;
				}
			}
		}
		return worst;
	}

	// Assembles the full (dense, flattened as ndof x ndof*ndof) Hessian tensor d^2R/dU^2 by
	// calling fill_in_generic_hessian with a dummy Y=0 vector (so no vector-product contraction
	// happens) and flag=3 (request full Hessian assembly rather than a vector product).
	void BulkElementBase::assemble_hessian_tensor(oomph::DenseMatrix<double> &hbuffer)
	{
		oomph::DenseMatrix<double> dummy(this->ndof(),this->ndof()*this->ndof(),0.0);// For the mass matrix
		fill_in_generic_hessian(oomph::Vector<double>(this->ndof(),0.0), dummy, hbuffer, 3);
	}

	// Assembles both the residual Hessian (d^2R/dU^2) and the mass-matrix Hessian (d^2M/dU^2) as
	// full rank-3 tensors directly via the JIT-generated HessianVectorProduct function (called
	// with a NULL Y so it fills the full tensors instead of contracting with a vector). Requires
	// the code to have been JIT-compiled with analytic Hessians enabled.
   void BulkElementBase::assemble_hessian_and_mass_hessian(oomph::RankThreeTensor<double> & hbuffer,oomph::RankThreeTensor<double> & mbuffer)
   {
		JITShapeInfo_t *const shape_info = this->get_shape_info();
		const JITFuncSpec_Table_FiniteElement_t *functable = jitcode->get_func_table();
		const int __crj = get_current_res_jac(functable); // hoisted: read once per element, see thread_state.hpp
		if (__crj < 0)
			return;
		if (!this->ndof())
			return;
		if (!functable->hessian_generated)
			throw_runtime_error("Tried to calculate an analytical Hessian, but the corresponding C code was not generated. Please call setup_for_stability_analysis(analytic_hessian=True) of the Problem instance before initialization of the Problem.");
		if (!functable->HessianVectorProduct[__crj])
			return;

      hbuffer.resize(this->ndof(),this->ndof(),this->ndof());
      hbuffer.initialise(0.0);
      mbuffer.resize(this->ndof(),this->ndof(),this->ndof());
      mbuffer.initialise(0.0);      
		prepare_shape_buffer_for_integration(functable->shapes_required_Hessian[__crj], 3);
		shape_info->jacobian_size = this->ndof();
		// Unconditional: HessianVectorProduct has no _NoHang twin either.
		this->fill_hang_info_with_equations(functable->shapes_required_Hessian[__crj], shape_info, NULL);
		this->interpolate_hang_values(); // This should be done elsewhere
		JITFuncSpec_HessianVectorProduct_FiniteElement func = functable->HessianVectorProduct[__crj];

		func(&eleminfo, shape_info, NULL, &(mbuffer(0, 0,0)), &(hbuffer(0, 0,0)), 1, 3);		   
   }
   

	
	// Residuals passed to fill_in_generic_residual_contribution_jit for solving projection of coordinates and fields.
	void BulkElementBase::residuals_for_zeta_projection(oomph::Vector<double>& residuals, oomph::DenseMatrix<double>& jacobian, const unsigned& do_fill_jacobian){
		
		// Store element in variable.
		FiniteElement *elem = dynamic_cast<FiniteElement *>(this);

		// Element's dimension.
		unsigned dim = elem->dim();

		// Local coordinates.
		oomph::Vector<double> s(dim,0.0);

		// Number of nodes.
		unsigned n_node = this->nnode();
		// Number of positional dofs.
      	const unsigned n_position_type = this->nnodal_position_type();
		// Set the value of n_intpt.
		const unsigned n_intpt = integral_pt()->nweight();

		// Get projection time.
		unsigned t =this->projection_time;

		// Create a field map to loop through all fields in element.
		auto *jitcode = this->get_jit_code();
		auto *func_table = jitcode->get_func_table();
		std::vector<int> field_map;
		for (unsigned int si=0;si<func_table->num_present_dg_spaces;si++)
		{
			if (func_table->dg_spaces[si].numfields)
		    {
		     throw_runtime_error("Cannot interpolate discontinuous fields yet");
		    }
		}		
		for (unsigned int si=0;si<func_table->num_present_continuous_spaces;si++)
		{
			if (func_table->continuous_spaces[si].numfields-func_table->continuous_spaces[si].numfields_basebulk)
		    {
		     throw_runtime_error("Cannot interpolate interface fields yet");
		    }
		}
		

		
		field_map.resize(ncont_interpolated_values());
		for (unsigned int i = 0; i < field_map.size(); i++){field_map[i] = i;}

		// Loop over integration points.
		for (unsigned ipt=0; ipt<n_intpt;ipt++){

			// Get local coordinates at integration point.
			for(unsigned i=0;i<dim;i++){s[i] = integral_pt()->knot(ipt, i);}

			// Old element pointer.
			BulkElementBase *old_elem = coords_oldmesh[ipt].first;
			// NULL when this integration point could not be located in the old mesh at all - a point
			// of the new mesh lying outside it, which a remesh of a curved boundary produces. There
			// is nothing to project from, so the point contributes nothing rather than being
			// dereferenced.
			if (!old_elem)
				continue;
			oomph::Vector<double> old_s = coords_oldmesh[ipt].second;

			// Shape functions.
			oomph::Shape psi(n_node,n_position_type);
			this->shape(s,psi);

			// Jacobian of mapping from local to global coordinates.
            double J = this->J_eulerian(s);

			// Get weight at ipt.
			double w = integral_pt()->weight(ipt);

			// Premultiply weights with Jacobian.
			double W = w * J;

			// Current position at current mesh.
			oomph::Vector<double> interpolated_zeta_curr(dim,0.0);
			oomph::Vector<double> interpolated_x_curr(dim,0.0);
			this->interpolated_zeta(s, interpolated_zeta_curr);
			this->interpolated_x(0, s, interpolated_x_curr);

			// Position in old element.
			oomph::Vector<double> interpolated_zeta_old(dim,0.0);
			oomph::Vector<double> interpolated_x_old(dim,0.0);
			old_elem->interpolated_zeta(old_s, interpolated_zeta_old);
			old_elem->interpolated_x(t, old_s, interpolated_x_old);

			// Initialise local equation and local unknown.
			int local_eqn=0;
			int local_unknown=0;

			// Loop through nodes.
			for(unsigned l=0;l<n_node;l++){
				
				// Loop through position types.
				for(unsigned k = 0; k < n_position_type; k++){

					//======= Coordinates: FROZEN, not projected =========//
					//
					// The positions are not solved for here at all. The current ones are what the mesh
					// generator produced and are already the answer; the history ones are what the mesh
					// velocity is built from, and the nodal transfer that seeds this projection already
					// delivers them to ~1e-13, which is better than this could.
					//
					// Projecting them was tried and does not work: the integration weights
					// W = J_eulerian(s) * w depend on the very position dofs being solved for, and the
					// Jacobian assembled here does not include that dependence, so the iteration is a
					// fixed point rather than a Newton method. Started 3% away it diverged outright
					// (1.7e-3, 7.1e-4, 4.7e-2, 1.6e+47, NaN, as elements inverted), and seeding it with
					// the converged nodal answer did not save it either.
					//
					// They still need an equation, or the matrix is singular wherever positions are
					// unknowns. A unit diagonal with a zero residual is that equation: it leaves them
					// exactly where they are, and - the reason this matters for cost - it also keeps
					// the geometry fixed, so the field system below is exactly linear and one
					// factorisation serves every field and every history level.
					if (do_fill_jacobian == 1)
					{
						for(unsigned i=0; i<dim; i++){
							local_eqn = this->position_local_eqn(l, k, i);
							if(local_eqn >= 0){
								jacobian(local_eqn, local_eqn) += 1.0;
							}
						}
					}
				}  // End of the loop over position types: the field block below is per node only.

				
				//======= Fill residuals for fields =========//

				// Get interpolated values for current mesh.
				oomph::Vector<double> interpolated_values_curr;
				this->get_interpolated_values(0, s, interpolated_values_curr);

				// Get interpolated values for old mesh.
				oomph::Vector<double> interpolated_values_old;
				// old_s, not s: s is this element's local coordinate, which means nothing in the old
				// element. The mapping from one to the other is the whole point of coords_oldmesh.
				old_elem->get_interpolated_values(t, old_s, interpolated_values_old);

				// Loop through every field.
				for(unsigned field=0; field<field_map.size(); field++){

					// field indexes a nodal VALUE, but field_map is sized by
					// ncont_interpolated_values(), which counts the fields of the element's function
					// space and need not equal the number of values a given node carries - a vertex
					// and a mid-side node of a mixed C2/C1 element do not carry the same set. Asking
					// nodal_local_eqn for a value the node does not have gives an index that is not a
					// dof of this element, and writing at it corrupted the assembly (an abort inside
					// sparse_assemble_row_or_column_compressed).
					if (field >= elem->node_pt(l)->nvalue())
						continue;

					// Get local equation number.
					local_eqn = elem->nodal_local_eqn(l, field);

					// If it is a degree of freedom.
					if(local_eqn >= 0){
						// Add residuals.
						residuals[local_eqn]+=(interpolated_values_curr[field]-interpolated_values_old[field]) * psi(l) * W;

					// The Jacobian must be assembled INSIDE the local_eqn >= 0 test. It used to sit
					// outside it, so for any pinned value - a Dirichlet condition, say - local_eqn is
					// -1 and jacobian(-1, ...) wrote before the start of the elemental matrix. That
					// is a heap overwrite, and it surfaced only later and elsewhere, as
					// "free(): invalid size" inside sparse_assemble_row_or_column_compressed.
					if (do_fill_jacobian == 1)
					{
						for (unsigned l2 = 0; l2 < n_node; l2++)
						{
							// The unknown is node l2's dof, not node l's. Indexing it by l made every
							// entry of the row land in the SAME column, summing psi(l2)*psi(l) over l2
							// into it - so the assembled matrix was not the mass matrix at all.
							if (field >= elem->node_pt(l2)->nvalue())
								continue;
							local_unknown = elem->nodal_local_eqn(l2, field);

							if (local_unknown >= 0)
							{	
								//Add Jacobian
								jacobian(local_eqn, local_unknown) += psi(l2) * psi(l) * W;
							}	
						}
					}
					}
				}
			}
		}
	}
   

	// Shared implementation behind fill_in_contribution_to_hessian_vector_products() and
	// assemble_hessian_tensor()/assemble_hessian_and_mass_hessian(): calls the JIT-generated
	// HessianVectorProduct function to contract the residual Hessian d^2R/dU^2 with the given
	// eigenvector Y and a set of directions C (flag selects vector-product vs. full-tensor mode,
	// see the JITFuncSpec_HessianVectorProduct_FiniteElement calling convention).
	void BulkElementBase::fill_in_generic_hessian(oomph::Vector<double> const &Y, oomph::DenseMatrix<double> &C, oomph::DenseMatrix<double> &product, unsigned flag)
	{
		JITShapeInfo_t *const shape_info = this->get_shape_info();
		const JITFuncSpec_Table_FiniteElement_t *functable = jitcode->get_func_table();
		const int __crj = get_current_res_jac(functable); // hoisted: read once per element, see thread_state.hpp
		if (__crj < 0)
			return;
		if (!this->ndof())
			return;
		if (!functable->hessian_generated)
			throw_runtime_error("Tried to calculate an analytical Hessian, but the corresponding C code was not generated. Please call setup_for_stability_analysis(analytic_hessian=True) of the Problem instance before initialization of the Problem.");
		if (!functable->HessianVectorProduct[__crj])
			return;

		unsigned n_vec = C.nrow();
		unsigned n_var = Y.size();
		if (flag == 3)
			n_var = product.nrow();

		prepare_shape_buffer_for_integration(functable->shapes_required_Hessian[__crj], 3);
		shape_info->jacobian_size = n_var; // Storing the number of dofs now
										   //& shape_info->mass_matrix_size=n_vec; // Won't be used, but storing the numbers of vects
		// Unconditional: HessianVectorProduct has no _NoHang twin, so this body always reads the hang
		// buffers and Stage 3a's skip does not apply here.
		this->fill_hang_info_with_equations(functable->shapes_required_Hessian[__crj], shape_info, NULL);
		this->interpolate_hang_values(); // XXX This should be moved to somewhere else, after each update of any values
		JITFuncSpec_HessianVectorProduct_FiniteElement func = functable->HessianVectorProduct[__crj];

		// const double * Cs=&(const_cast<oomph::DenseMatrix<double>*>(&C)->entry(0,0)); // XXX: Dirty hack, but otherwise not possibility to call this
		func(&eleminfo, shape_info, &(Y[0]), &(C.entry(0, 0)), &(product.entry(0, 0)), n_vec, flag);
	}

	// oomph-lib hook: computes product = sum_dofs Hessian(Y) * C^T, i.e. the Hessian-vector
	// products needed for e.g. bifurcation tracking / stability analysis. Delegates to
	// fill_in_generic_hessian in plain (non-tensor) vector-product mode.
	void BulkElementBase::fill_in_contribution_to_hessian_vector_products(oomph::Vector<double> const &Y, oomph::DenseMatrix<double> const &C, oomph::DenseMatrix<double> &product)
	{
		oomph::DenseMatrix<double> Ccopy = C;
		this->fill_in_generic_hessian(Y, Ccopy, product, 0);
	}

	void BulkElementBase::fill_in_jacobian_from_nodal_by_fd(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian)
	{
		const unsigned n_node = this->nnode();
		if (n_node == 0) return;
		// Every perturbation below re-enters get_residuals with a DIFFERENT dof vector, so the hanging
		// values genuinely change inside this one enclosing assembly sweep: the pass dedupe has to be
		// off for the whole loop (and a fresh pass opened afterwards, since the loop leaves the last
		// perturbed interpolation in the nodes).
		HangInterpPassSuspension __no_pass;

		this->update_before_nodal_fd();

		const unsigned n_dof = this->ndof();
		oomph::Vector<double> newres(n_dof);
		const double fd_step = this->Default_fd_jacobian_step;
		int local_unknown = 0;

		const std::vector<std::vector<unsigned>> & space_node_to_elem_node_map=this->get_nodal_space_index_to_element_index_map();
		for (unsigned int ispace=0;ispace<jitcode->get_func_table()->num_present_continuous_spaces;ispace++)
		{
			auto *space_info=jitcode->get_func_table()->present_continuous_spaces[ispace];
			if (space_info->numfields_basebulk)
			{
				for (unsigned n = 0; n < this->eleminfo.nnode_of_space[space_info->space_index]; n++)
				{
					oomph::Node *const local_node_pt = this->node_pt(space_node_to_elem_node_map[space_info->space_index][n]);
					for (unsigned i = space_info->nodal_offset_basebulk; i < space_info->nodal_offset_basebulk+space_info->numfields_basebulk; i++)
					{
						if (local_node_pt->is_hanging(space_info->hangindex) == false)
						{
							local_unknown = this->nodal_local_eqn(space_node_to_elem_node_map[space_info->space_index][n], i);
							if (local_unknown >= 0)
							{
								double *const value_pt = local_node_pt->value_pt(i);
								const double old_var = *value_pt;
								*value_pt += fd_step;
								this->update_in_nodal_fd(i);
								this->get_residuals(newres);
								for (unsigned m = 0; m < n_dof; m++)
								{
									jacobian(m, local_unknown) = (newres[m] - residuals[m]) / fd_step;
								}
								*value_pt = old_var;
								this->reset_in_nodal_fd(i);
							}
						}
						else
						{
							oomph::HangInfo *hang_info_pt = local_node_pt->hanging_pt(space_info->hangindex);
							const unsigned n_master = hang_info_pt->nmaster();
							for (unsigned m = 0; m < n_master; m++)
							{
								oomph::Node *const master_node_pt = hang_info_pt->master_node_pt(m);					
								local_unknown = this->local_hang_eqn(master_node_pt, i);						
								if (local_unknown >= 0)
								{
									double *const value_pt = master_node_pt->value_pt(i);
									const double old_var = *value_pt;
									*value_pt += fd_step;
									this->update_in_nodal_fd(i);
									this->get_residuals(newres);
									for (unsigned mm = 0; mm < n_dof; mm++)
									{
										jacobian(mm, local_unknown) = (newres[mm] - residuals[mm]) / fd_step;
									}
									*value_pt = old_var;
									this->reset_in_nodal_fd(i);
								}
							}
						}
					}
				}
			}
		}

		this->reset_after_nodal_fd();
	}

	// Debug helper: recomputes the Jacobian via the (base oomph-lib) finite-difference path and
	// compares it entry-by-entry to the already-assembled analytical (JIT) `jacobian`, printing
	// every entry that differs by more than diff_eps (with dof names) to help track down bugs in
	// generated residual/Jacobian code. Optionally aborts (stop_on_jacobian_difference) with a
	// detailed dump of the element's dof/equation-number bookkeeping.
	void BulkElementBase::debug_analytical_jacobian(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian, double diff_eps)
	{
		JITShapeInfo_t *const shape_info = this->get_shape_info();
		// oomph-lib's generic FD path below perturbs dofs and re-enters get_residuals, exactly like our own
		// FD loops do.
		HangInterpPassSuspension __no_pass;
		oomph::Vector<double> fd_residuals(residuals.size(), 0.0);
//		std::cout << "DB NDOF " << this->ndof() << std::endl  << std::flush;
//		std::cout << "   J" << jacobian.nrow() << " x " << jacobian.ncol() << std::endl << std::flush;
		oomph::DenseMatrix<double> fd_jacobian(jacobian.nrow(), jacobian.ncol(), 0.0);
		if (jitcode->get_func_table()->missing_residual_assembly[get_current_res_jac(jitcode->get_func_table())])
		{
		    throw_runtime_error("The Jacobian of the residual "+std::string(jitcode->get_func_table()->res_jac_names[get_current_res_jac(jitcode->get_func_table())])+" cannot be calculated by finite differences, since the residual is not calculated at all.");
		}
		this->RefineableSolidElement::fill_in_contribution_to_jacobian(fd_residuals, fd_jacobian);
		//	this->fill_in_jacobian_from_lagragian_by_fd(fd_residuals,fd_jacobian);
		std::vector<std::string> dofnames = get_dof_names();
		bool header_written = false;
		for (unsigned int i = 0; i < jacobian.nrow(); i++)
		{
			for (unsigned int j = 0; j < jacobian.ncol(); j++)
			{
				double diff = fd_jacobian(i, j) - jacobian(i, j);
				diff = fabs(diff);
				if (diff > diff_eps)
				{
					if (!header_written)
					{
						std::cout << "DIFFERENCES IN JACOBIAN ndof=" << this->ndof() << " ELEM_DIM " << this->dim() << std::endl;
						std::cout << "#I\tJ\tDOF_i\tDOF_j\tDIFF\tJana\tJfd\tRana_i\tRfd_i\tRana_j\tRfd_j" << std::endl;
						header_written = true;
					}
					std::cout << i << "\t" << j << "\t" << dofnames[i] << "\t" << dofnames[j] << "\t" << diff << "\t" << jacobian(i, j) << "\t" << fd_jacobian(i, j) << "\t" << residuals[i] << "\t" << fd_residuals[i] << "\t" << residuals[j] << "\t" << fd_residuals[j] << std::endl;
				}
			}
		}
		if (header_written && jitcode->get_func_table()->stop_on_jacobian_difference)
		{
			const JITFuncSpec_Table_FiniteElement_t *functable = jitcode->get_func_table();
			// Now a very detailed list:
			std::cout << "DOF LIST" << std::endl;
			for (unsigned int i = 0; i < dofnames.size(); i++)
			{
				std::cout << "\t" << i << "\t" << dofnames[i] << std::endl;
			}
			std::cout << "NODAL VALUE EQ BUFFER" << std::endl;
			for (unsigned int l = 0; l < this->nnode(); l++)
			{
				std::cout << "\t" << l;
				for (unsigned int i = 0; i < this->node_pt(l)->nvalue(); i++)
				{
					std::cout << "\t" << eleminfo.nodal_local_eqn[l][i];
				}
				std::cout << std::endl;
			}
			std::cout << "POS VALUE EQ BUFFER" << std::endl;
			for (unsigned int l = 0; l < this->nnode(); l++)
			{
				std::cout << "\t" << l;
				for (unsigned int i = 0; i < static_cast<Node *>(this->node_pt(l))->variable_position_pt()->nvalue(); i++)
				{
					std::cout << "\t" << eleminfo.pos_local_eqn[l][i];
				}
				std::cout << "\t@\t";
				for (unsigned int i = 0; i < static_cast<Node *>(this->node_pt(l))->variable_position_pt()->nvalue(); i++)
				{
					std::cout << "\t" << static_cast<Node *>(this->node_pt(l))->variable_position_pt()->value(i);
				}
				std::cout << std::endl;
			}
			std::cout << "HANG INFO POS" << std::endl;
			for (unsigned int l = 0; l < this->nnode(); l++)
			{
				std::cout << "\t" << l;
				for (unsigned int j = 0; j < eleminfo.nodal_dim; j++)
				{
					std::cout << "\t[dim " << j << "] nmst: " << shape_info->hanginfo_Pos[j][l].nummaster;
					for (int m = 0; m < shape_info->hanginfo_Pos[j][l].nummaster; m++)
					{
						std::cout << " (weight:" << shape_info->hanginfo_Pos[j][l].masters[m].weight << " eq:" << shape_info->hanginfo_Pos[j][l].masters[m].local_eqn << ")";
					}
				}
				std::cout << std::endl;
			}

			auto print_hanginfo_for_space = [](JITShapeInfo_t * si_shape_info, const JITFuncSpec_Table_FiniteElement_SpaceInfo_t & space_info, unsigned nnode_space)
			{
				std::cout << "HANG INFO " << space_info.space_name << std::endl;
				for (unsigned int l = 0; l < nnode_space; l++)
				{
					std::cout << "\t" << l;
					for (unsigned int f = 0; f < space_info.numfields_basebulk; f++)
					{
						JITHangInfo_t * hangbuffer = si_shape_info->hanginfo[f + space_info.buffer_offset_basebulk];
						std::cout << "\t[f" << f << "] nmst: " << hangbuffer[l].nummaster;
						for (int m = 0; m < hangbuffer[l].nummaster; m++)
							std::cout << " (weight:" << hangbuffer[l].masters[m].weight << " eq:" << hangbuffer[l].masters[m].local_eqn << ")";
					}
					for (unsigned int f = 0; f < space_info.numfields-space_info.numfields_basebulk; f++)
					{
						JITHangInfo_t * hangbuffer = si_shape_info->hanginfo[f + space_info.buffer_offset_interf];
						std::cout << "\t[if" << f << "] nmst: " << hangbuffer[l].nummaster;
						for (int m = 0; m < hangbuffer[l].nummaster; m++)
							std::cout << " (weight:" << hangbuffer[l].masters[m].weight << " eq:" << hangbuffer[l].masters[m].local_eqn << ")";
					}
					std::cout << std::endl;
				}
			};

			for (unsigned int si=0;si<NUM_CONTINUOUS_SPACES;si++)
			{
				if (eleminfo.nnode_of_space[si])
				{
					print_hanginfo_for_space(shape_info, functable->continuous_spaces[si], eleminfo.nnode_of_space[si]);
				}
			}

			if (functable->shapes_required_ResJac[get_current_res_jac(functable)].bulk_shapes && this->as_interface_element())
			{
				BulkElementBase *bel = dynamic_cast<BulkElementBase *>(this->as_interface_element()->bulk_element_pt());
				std::cout << "BULK HANG INFO POS" << std::endl;
				for (unsigned int l = 0; l < bel->nnode(); l++)
				{
					std::cout << "\t" << l;
					for (unsigned int j = 0; j < bel->eleminfo.nodal_dim; j++)
					{
						std::cout << "\t[dim " << j << "] nmst: " << shape_info->bulk_shapeinfo->hanginfo_Pos[j][l].nummaster;
						for (int m = 0; m < shape_info->bulk_shapeinfo->hanginfo_Pos[j][l].nummaster; m++)
						{
							std::cout << " (weight:" << shape_info->bulk_shapeinfo->hanginfo_Pos[j][l].masters[m].weight << " eq:" << shape_info->bulk_shapeinfo->hanginfo_Pos[j][l].masters[m].local_eqn << ")";
						}
					}
					std::cout << std::endl;
				}

				for (unsigned int si=0;si<NUM_CONTINUOUS_SPACES;si++)
				{
					if (bel->eleminfo.nnode_of_space[si])
					{
						std::cout << "BULK ";
						print_hanginfo_for_space(shape_info->bulk_shapeinfo, functable->continuous_spaces[si], bel->eleminfo.nnode_of_space[si]);
					}
				}
			}

			InterfaceElementBase *ie = this->as_interface_element();
			std::string prefix = "";
			while (ie)
			{
				prefix = prefix + "BULK_PARENT:";
				BulkElementBase *be = dynamic_cast<BulkElementBase *>(ie->bulk_element_pt());
				std::vector<std::string> pdofnames = be->get_dof_names();
				std::cout << "DOFS FOR " << prefix << std::endl;
				for (unsigned int i = 0; i < pdofnames.size(); i++)
				{
					std::cout << "\t" << i << "\t" << pdofnames[i] << std::endl;
				}
				ie = be->as_interface_element();
			}

			throw_runtime_error("Mismatch in Jacobian in code: " + this->jitcode->get_file_name());
		}
	}

	// Fills in the Jacobian columns corresponding to nodal position (Lagrangian/geometric) dofs
	// by finite-differencing the residuals with respect to each nodal coordinate in turn (used
	// when the JIT-generated code cannot or does not provide the position-Jacobian analytically,
	// e.g. moving-mesh problems with fd_position_jacobian set). Columns belonging to non-position
	// ("Lagrangian") dofs are left untouched here (is_lagrangian_dof is currently unused/disabled
	// via the commented-out block, so effectively every dof's row is updated for every perturbed
	// position dof). Perturbs one nodal coordinate at a time (looping master nodes for hanging
	// nodes), re-evaluates the residuals, and forms the FD column, restoring the coordinate
	// afterwards.
	void BulkElementBase::fill_in_jacobian_from_lagragian_by_fd(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian)
	{
		HangInterpPassSuspension __no_pass; // see fill_in_jacobian_from_nodal_by_fd

		const unsigned n_node = this->nnode();
		if (n_node == 0)
		{
			return;
		}

		// Test if this is a complete finite difference loop
		//  const JITFuncSpec_Table_FiniteElement_t * functable=jitcode->get_func_table();

		update_before_solid_position_fd();
		const unsigned n_position_type = this->nnodal_position_type();
		const unsigned nodal_dim = this->nodal_dimension();
		const unsigned n_dof = this->ndof();
		oomph::Vector<double> newres(n_dof);
		const double fd_step = this->Default_fd_jacobian_step;
		int local_unknown = 0;

		std::vector<bool> is_lagrangian_dof(this->ndof(), false);

		/*
		  for(unsigned l=0;l<n_node;l++)
		   {
			oomph::Node* const local_node_pt = this->node_pt(l);
			if(local_node_pt->is_hanging()==false)
			 {
			  for(unsigned k=0;k<n_position_type;k++)
			   {
				for(unsigned i=0;i<nodal_dim;i++)
				 {
				  local_unknown = this->position_local_eqn(l,k,i);
				  if(local_unknown >= 0)
				   {
					 is_lagrangian_dof[local_unknown]=true;
				   }
				 }
			   }
			 }
			 else
			 {
			  oomph::HangInfo* hang_info_pt = local_node_pt->hanging_pt();
			  const unsigned n_master = hang_info_pt->nmaster();
			  for(unsigned m=0;m<n_master;m++)
			   {
				oomph::Node* const master_node_pt = hang_info_pt->master_node_pt(m);
				oomph::DenseMatrix<int> Position_local_eqn_at_node = this->local_position_hang_eqn(master_node_pt);
				for(unsigned k=0;k<n_position_type;k++)
				 {
				  for(unsigned i=0;i<nodal_dim;i++)
				   {
					local_unknown = Position_local_eqn_at_node(k,i);
					if(local_unknown >= 0)
					 {
						 is_lagrangian_dof[local_unknown]=true;
					 }
				   }
				 }
			   }

			 }
			}      //TODO: Bulk element external position data

		*/
		for (unsigned l = 0; l < n_node; l++)
		{
			oomph::Node *const local_node_pt = this->node_pt(l);
			if (local_node_pt->is_hanging() == false)
			{
				for (unsigned k = 0; k < n_position_type; k++)
				{
					for (unsigned i = 0; i < nodal_dim; i++)
					{
						local_unknown = this->position_local_eqn(l, k, i);
						if (local_unknown >= 0)
						{
							double *const value_pt = &(local_node_pt->x_gen(k, i));
							const double old_var = *value_pt;
							*value_pt += fd_step;
							//            local_node_pt->perform_auxiliary_node_update_fct();
							update_in_solid_position_fd(l);
							get_residuals(newres);
							for (unsigned m = 0; m < n_dof; m++)
							{
								if (!is_lagrangian_dof[m])
								{
									// std::cout << "PERTURBED RESIDUALS " << l << "  " << k << "  " << i << "  at m " << m << " is " << (newres[m] - residuals[m])/fd_step << " WRITING TO (" << m << ", " << local_unknown << ")" << std::endl;
									jacobian(m, local_unknown) = (newres[m] - residuals[m]) / fd_step;
								}
							}
							*value_pt = old_var;
							// local_node_pt->perform_auxiliary_node_update_fct();
							// reset_in_solid_position_fd(l);
						}
					}
				}
			}
			// Otherwise it's a hanging node
			else
			{
				oomph::HangInfo *hang_info_pt = local_node_pt->hanging_pt();
				const unsigned n_master = hang_info_pt->nmaster();
				for (unsigned m = 0; m < n_master; m++)
				{
					oomph::Node *const master_node_pt = hang_info_pt->master_node_pt(m);
					oomph::DenseMatrix<int> Position_local_eqn_at_node = this->local_position_hang_eqn(master_node_pt);
					for (unsigned k = 0; k < n_position_type; k++)
					{
						for (unsigned i = 0; i < nodal_dim; i++)
						{
							local_unknown = Position_local_eqn_at_node(k, i);
							if (local_unknown >= 0)
							{
								double *const value_pt = &(master_node_pt->x_gen(k, i));
								const double old_var = *value_pt;
								*value_pt += fd_step;
								 master_node_pt->perform_auxiliary_node_update_fct();
								update_in_solid_position_fd(l);
								get_residuals(newres);

								for (unsigned m = 0; m < n_dof; m++)
								{
									if (!is_lagrangian_dof[m])
										jacobian(m, local_unknown) = (newres[m] - residuals[m]) / fd_step;
								}

								*value_pt = old_var;
								// master_node_pt->perform_auxiliary_node_update_fct();
								// reset_in_solid_position_fd(l);
							}
						}
					}
				}
			} // End of hanging node case
		}	  // End of loop over nodes
     reset_after_solid_position_fd();
		this->interpolate_hang_values();
	}
	
	
	void BulkElementBase::update_in_solid_position_fd(const unsigned &) // For FD with element_sizes, we have to update the element size buffer
	{
		JITShapeInfo_t *const shape_info = this->get_shape_info();
	 const JITFuncSpec_Table_FiniteElement_t *functable = jitcode->get_func_table();
		const int __crj = get_current_res_jac(functable); // hoisted: read once per element, see thread_state.hpp
	 if (functable->moving_nodes && (functable->shapes_required_ResJac[__crj].elemsize_Eulerian_cartesian || functable->shapes_required_ResJac[__crj].elemsize_Eulerian))
	 {
//	  std::cout << "UPDATE CALL" << std::endl;
	  this->fill_shape_info_element_sizes(functable->shapes_required_ResJac[__crj],shape_info,0);
	 }
	}

	// oomph-lib hook for residual + Jacobian assembly. Normally delegates to the JIT-generated
	// analytical assembly; if the code table requests position-Jacobian entries by finite
	// differences (fd_position_jacobian, for moving-mesh problems), those columns are patched in
	// afterwards via fill_in_jacobian_from_lagragian_by_fd(). If debug_jacobian_epsilon is set,
	// cross-checks the analytical Jacobian against a full FD one. If fd_jacobian is set for the
	// whole element (rather than just positions), falls back entirely to oomph-lib's generic FD
	// Jacobian.
	void BulkElementBase::fill_in_contribution_to_jacobian(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = jitcode->get_func_table();
		const int __crj = get_current_res_jac(functable); // hoisted: read once per element, see thread_state.hpp
		if (!functable->fd_jacobian)
		{
			fill_in_generic_residual_contribution_jit(residuals, jacobian, oomph::GeneralisedElement::Dummy_matrix, 1);
			if (functable->moving_nodes && functable->fd_position_jacobian)
			{
				this->fill_in_jacobian_from_lagragian_by_fd(residuals, jacobian);
			}

			if (functable->debug_jacobian_epsilon != 0.0 && __crj>=0)
				debug_analytical_jacobian(residuals, jacobian, functable->debug_jacobian_epsilon);
		}
		else
		{
		   if (__crj<0) return;
		   if (functable->missing_residual_assembly[__crj])
		   {
		    throw_runtime_error("The Jacobian of the residual "+std::string(functable->res_jac_names[__crj])+" cannot be calculated by finite differences, since the residual is not calculated at all.");
		   }
			// oomph-lib's own FD Jacobian: same reason as in fill_in_jacobian_from_nodal_by_fd.
			HangInterpPassSuspension __no_pass;
			this->RefineableSolidElement::fill_in_contribution_to_jacobian(residuals, jacobian);
		}

		/*
			 std::vector<std::string> dofnames=get_dof_names();
			 for (unsigned int i=0;i<jacobian.nrow();i++)
			 {
			   double minv=0;
			   double maxv=0;
			   for (unsigned int j=0;j<jacobian.ncol();j++)
			   {
				if (jacobian(i,j)<minv) minv=jacobian(i,j);
				if (jacobian(i,j)>maxv) maxv=jacobian(i,j);
			   }
			   if (minv==0 && maxv==0)
			   {
				std::cout << "EMPTY JACOBIAN CONTRIBTUION IN ROW " << i << " corresponding to eq " << this->eqn_number(i) << "  which is " << dofnames[i] << std::endl;
				std::cout << "ALL DOFS ARE " << std::endl;
				for (unsigned int k=0;k<dofnames.size();k++) std::cout << "  " << k << "  " << dofnames[k] << std::endl;
				std::cout << "HANGING INFO " << std::endl;
				for (unsigned l=0;l<this->nnode();l++)
				{
				 if (this->node_pt(l)->is_hanging())
				 {
					  oomph::HangInfo* hang_info_pt = this->node_pt(l)->hanging_pt();
						const unsigned n_master = hang_info_pt->nmaster();
						std::cout << "  " << l << " master " << n_master << "  :  " ;
						for(unsigned m=0;m<n_master;m++)
						 {
						  oomph::Node* const master_node_pt = hang_info_pt->master_node_pt(m);
						  oomph::DenseMatrix<int> Position_local_eqn_at_node = this->local_position_hang_eqn(master_node_pt);
						 for(unsigned ii=0;ii<this->node_pt(l)->ndim();ii++)
						 {
							  std::cout << " " << Position_local_eqn_at_node(0,ii);
						  }
						 }
						 std::cout << std::endl;
				 }
				 else
				 {
					std::cout << "  " << l << " not hanging, eqs for direction: ";
						 for(unsigned ii=0;ii<this->node_pt(l)->ndim();ii++)
						 {
							  std::cout << " " << this->position_local_eqn(l,0,ii);
						  }
					std::cout << std::endl;
				 }
				}
				std::cout << "  N  EXTERNAL " << this->nexternal_data() << std::endl;
			   }
			 }
			*/
	}

	// oomph-lib hook for residual + Jacobian + mass-matrix assembly (used e.g. for eigenproblems).
	// FD mass matrices are not implemented (the fd_jacobian branch below is dead code, left in
	// for reference, and would throw before reaching it); the normal path always calls the
	// JIT-generated assembly with flag=2 (residuals + Jacobian + mass matrix).
	void BulkElementBase::fill_in_contribution_to_jacobian_and_mass_matrix(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian, oomph::DenseMatrix<double> &mass_matrix)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = jitcode->get_func_table();
		if (functable->fd_jacobian)
		{
			throw_runtime_error("FD Mass matrix not implemented");
			//WARNING: This takes the analytic mass matrix
			jitcode->get_func_table()->fd_jacobian=false;
			fill_in_generic_residual_contribution_jit(residuals, jacobian, mass_matrix, 2);
			jitcode->get_func_table()->fd_jacobian=true;
			residuals.initialise(0.0);
			jacobian.initialise(0.0);
			fill_in_generic_residual_contribution_jit(residuals, jacobian, mass_matrix, 1);
			
		}
		fill_in_generic_residual_contribution_jit(residuals, jacobian, mass_matrix, 2);
	}
}
