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

#include "problem.hpp"
#include "elements.hpp"
#include "jitbridge.h"
#include "exception.hpp"
#include "nodes.hpp"
#include "codegen.hpp"
#include "bifurcation.hpp"
#include "ccompiler.hpp"
#include "logging.hpp"

extern "C"
{
	void _pyoomph_check_compiler_size(unsigned long long jitsize, unsigned long long internal_size, char *name)
	{
		if (jitsize != internal_size)
		{
			std::ostringstream errmsg;
			std::string nam = name;
			errmsg << "Mismatch between compiler sizes. Test failed: " << nam << std::endl
				   << "Expected " << internal_size << ", but got " << jitsize;
			throw_runtime_error(errmsg.str());
		}
	}
}

namespace pyoomph
{

	void RequiredShapes_merge(JITFuncSpec_RequiredShapes_FiniteElement_t *src, JITFuncSpec_RequiredShapes_FiniteElement_t *dest)
	{
		dest->psi_C1 |= src->psi_C1;
		dest->psi_C2 |= src->psi_C2;
		dest->psi_C2TB |= src->psi_C2TB;
		dest->psi_C1TB |= src->psi_C1TB;
		dest->psi_DL |= src->psi_DL;
		dest->psi_D0 |= src->psi_D0;
		dest->dx_psi_C1 |= src->dx_psi_C1;
		dest->dx_psi_C2 |= src->dx_psi_C2;
		dest->dx_psi_C2TB |= src->dx_psi_C2TB;
		dest->dx_psi_C1TB |= src->dx_psi_C1TB;
		dest->dx_psi_DL |= src->dx_psi_DL;
		dest->dx_psi_D0 |= src->dx_psi_D0;
		dest->dX_psi_C1 |= src->dX_psi_C1;
		dest->dX_psi_C2 |= src->dX_psi_C2;
		dest->dX_psi_C2TB |= src->dX_psi_C2TB;
		dest->dX_psi_C1TB |= src->dX_psi_C1TB;
		dest->dX_psi_DL |= src->dX_psi_DL;
		dest->dX_psi_D0 |= src->dX_psi_D0;
		dest->psi_Pos |= src->psi_Pos;
		dest->dx_psi_Pos |= src->dx_psi_Pos;
		dest->dX_psi_Pos |= src->dX_psi_Pos;
		dest->normal_Pos |= src->normal_Pos;
		dest->elemsize_Eulerian_Pos |= src->elemsize_Eulerian_Pos;
		dest->elemsize_Lagrangian_Pos |= src->elemsize_Lagrangian_Pos;
		dest->elemsize_Eulerian_cartesian_Pos |= src->elemsize_Eulerian_cartesian_Pos;
		dest->elemsize_Lagrangian_cartesian_Pos |= src->elemsize_Lagrangian_cartesian_Pos;
		if (src->bulk_shapes)
		{
			if (!dest->bulk_shapes)
				dest->bulk_shapes = (JITFuncSpec_RequiredShapes_FiniteElement_t *)std::calloc(1, sizeof(JITFuncSpec_RequiredShapes_FiniteElement_t));
			RequiredShapes_merge(src->bulk_shapes, dest->bulk_shapes);
		}
		if (src->opposite_shapes)
		{
			if (!dest->opposite_shapes)
				dest->opposite_shapes = (JITFuncSpec_RequiredShapes_FiniteElement_t *)std::calloc(1, sizeof(JITFuncSpec_RequiredShapes_FiniteElement_t));
			RequiredShapes_merge(src->opposite_shapes, dest->opposite_shapes);
		}
	}

	void RequiredShapes_free(JITFuncSpec_RequiredShapes_FiniteElement_t *p)
	{
		if (p->bulk_shapes)
			RequiredShapes_free(p->bulk_shapes);
		if (p->opposite_shapes)
			RequiredShapes_free(p->opposite_shapes);
		std::free(p);
	}

	DynamicBulkElementCode::DynamicBulkElementCode(Problem *prob, CCompiler *ccompiler, std::string fnam, FiniteElementCode *elem) : problem(prob), compiler(ccompiler), filename(fnam), functable(NULL), element_class(elem), so_handle(NULL)
	{
		JIT_ELEMENT_init_SPEC initfunc = ccompiler->get_init_func();
		if (!initfunc)
		{
			throw_runtime_error("Cannot load the JIT code entry point");
		}
		so_handle = ccompiler->get_current_handle();
		ccompiler->reset_current_handle();
		functable = new JITFuncSpec_Table_FiniteElement_t;
		memset(functable, 0, sizeof(JITFuncSpec_Table_FiniteElement_t));

		functable->check_compiler_size = _pyoomph_check_compiler_size;
		initfunc(functable);

		// Merge the required shapes to add all external data
		JITFuncSpec_RequiredShapes_FiniteElement *merged = &(functable->merged_required_shapes);

		for (unsigned int i = 0; i < functable->num_res_jacs; i++)
		{
			RequiredShapes_merge(&functable->shapes_required_ResJac[i], merged);
			RequiredShapes_merge(&functable->shapes_required_Hessian[i], merged);
		}
		RequiredShapes_merge(&functable->shapes_required_IntegralExprs, merged);
		RequiredShapes_merge(&functable->shapes_required_LocalExprs, merged);
		RequiredShapes_merge(&functable->shapes_required_ExtremumExprs, merged);
		RequiredShapes_merge(&functable->shapes_required_Z2Fluxes, merged);
		RequiredShapes_merge(&functable->shapes_required_TracerAdvection, merged);

		functable->handle = so_handle;

		// Export the functions to call
		functable->get_element_size = _pyoomph_get_element_size;
		functable->invoke_callback = _pyoomph_invoke_callback;
		functable->invoke_multi_ret = _pyoomph_invoke_multi_ret;
		functable->fill_shape_buffer_for_point = _pyoomph_fill_shape_buffer_for_point;

		for (unsigned int i = 0; i < functable->numintegral_expressions; i++)
			integral_function_map[functable->integral_expressions_names[i]] = i;
		for (unsigned int i = 0; i < functable->numextremum_expressions; i++)
			extremum_function_map[functable->extremum_expressions_names[i]] = i;
	}

	DynamicBulkElementCode::~DynamicBulkElementCode()
	{
		// std::cout << "UNLOADING ELEMENT CODE " << filename << " FUNCTABLE " << functable << " SO HANDLE " << so_handle  << std::endl << std::flush;
		// std::cout << "COMPILER  " << compiler  << std::endl << std::flush;

		if (functable)
		{
			if (pyoomph_verbose)
			{
				std::cout << "Cleaning memory of functable" << std::endl << std::flush;
			}
			if (functable->clean_up) functable->clean_up(functable);
			delete functable; // TODO: Also delete the malloced subentries here
		}

		if (pyoomph_verbose)
		{
				std::cout << "Closing library handle " << this->get_file_name() << std::endl << std::flush;
		}
		compiler->close_handle(so_handle);
		if (pyoomph_verbose)
		{
				std::cout << "Closed library handle " << std::endl << std::flush;
		}
		so_handle = NULL;
		functable = NULL;
	}

	int DynamicBulkElementCode::get_integral_function_index(std::string n)
	{
		if (!integral_function_map.count(n))
			return -1;
		return integral_function_map[n];
	}

	int DynamicBulkElementCode::get_extremum_function_index(std::string n)
	{
		if (!extremum_function_map.count(n))
			return -1;
		return extremum_function_map[n];
	}

	unsigned DynamicBulkElementCode::_set_solved_residual(std::string name)
	{
		int res_jac_index = -1;
		for (unsigned int i = 0; i < functable->num_res_jacs; i++)
		{
			std::string n = functable->res_jac_names[i];
			//std::cout << this->get_file_name() << " " << i << " : " << n << " PRT " << functable->ResidualAndJacobian[i] << std::endl;
			if (n == name && functable->ResidualAndJacobian[i])
			{
				res_jac_index = i;
				break;
			}
		}
		functable->current_res_jac = res_jac_index;
		if (res_jac_index >= 0)
			return 1;
		else
			return 0;
	}
	DynamicBulkElementInstance *DynamicBulkElementCode::factory_instance(pyoomph::Mesh *bulkmesh)
	{
		return new DynamicBulkElementInstance(this, bulkmesh);
	}

	//////////////////////////////////////////

	void ExternalDataLinkVector::reindex_elemental_data()
	{
		elemental_data.clear();
		for (unsigned int i = 0; i < this->size(); i++)
		{
			int found = -1;
			for (unsigned int e = 0; e < elemental_data.size(); e++)
			{
				if (elemental_data[e] == this->at(i).data)
				{
					found = e;
					break;
				}
			}
			if (found < 0)
			{
				found = elemental_data.size();
				elemental_data.push_back(this->at(i).data);
			}
			this->at(i).elemental_index = found;
		}
	}

	///////////////////////////////////////////

	DynamicBulkElementInstance::DynamicBulkElementInstance(DynamicBulkElementCode *d, pyoomph::Mesh *bm) : dyn(d), // local_field_to_global_field_index_C1(d->functable->numfields_C1,-1),
																												   //		local_field_to_global_field_index_C2(d->functable->numfields_C2,-1),
																												   //		local_global_parameter_to_global_index(d->functable->numglobal_params,-1),
																										   linked_external_data(d->functable->numfields_ED0),
																										   bulkmesh(bm)
	{
		/*
		  for (unsigned int i=0; i<d->functable->num_nullified_bulk_residuals;i++)
		  {
			std::string fn=d->functable->nullified_bulk_residuals[i];
			int index;
			if (fn=="coordinate_x") index=-1;
			else if (fn=="coordinate_y") index=-2;
			else if (fn=="coordinate_z") index=-3;
			else
			{
			  index=get_nodal_field_index(fn);
			  if (index==-1) throw_runtime_error("Cannot nullify the bulk DoF " +fn);
			}
			nullify_bulk_residuals.insert(index);
		  }
		 */
	}

	void DynamicBulkElementInstance::link_external_data(std::string name, oomph::Data *data, int index,std::string full_source_name)
	{
		int found = -1;
		for (unsigned int i = 0; i < dyn->functable->numfields_ED0; i++)
		{
			if (name == std::string(dyn->functable->fieldnames_ED0[i]))
			{
				found = i;
				break;
			}
		}
		if (found == -1)
			throw_runtime_error("Cannot link external data '" + name + "' since this is not required by the equation code");
		linked_external_data[found] = ExternalDataLink(data, index);
		// Replace the residual and jacobian information as well
		std::string look_for=this->get_element_class()->get_full_domain_name()+"/"+name;
		for (unsigned int i = 0; i < dyn->functable->contribution_entries_size; i++)
		{
			std::string res_name = dyn->functable->contribution_names[i];
			if (res_name == look_for)
			{
				free((char*)dyn->functable->contribution_names[i]);
				dyn->functable->contribution_names[i] = strdup(full_source_name.c_str());
				break;
			}
		}
		linked_external_data.reindex_elemental_data();
	}

	std::map<std::string, unsigned> DynamicBulkElementInstance::get_nodal_field_indices()
	{
		std::map<std::string, unsigned> res;
		unsigned offs = 0;
		for (unsigned int i = 0; i < dyn->functable->numfields_C2TB_basebulk; i++)
		{
			res[dyn->functable->fieldnames_C2TB[i]] = offs + i;
		}
		offs += dyn->functable->numfields_C2TB_basebulk;
		for (unsigned int i = 0; i < dyn->functable->numfields_C2_basebulk; i++)
		{
			res[dyn->functable->fieldnames_C2[i]] = offs + i;
		}
		offs += dyn->functable->numfields_C2_basebulk;
		for (unsigned int i = 0; i < dyn->functable->numfields_C1TB_basebulk; i++)
		{
			res[dyn->functable->fieldnames_C1TB[i]] = offs + i;
		}
		offs += dyn->functable->numfields_C1TB_basebulk;
		for (unsigned int i = 0; i < dyn->functable->numfields_C1_basebulk; i++)
		{
			res[dyn->functable->fieldnames_C1[i]] = offs + i;
		}
		offs += dyn->functable->numfields_C1_basebulk;

		for (unsigned int i = 0; i < dyn->functable->numfields_D2TB_basebulk; i++)
		{
			res[dyn->functable->fieldnames_D2TB[i]] = offs + i;
		}
		offs += dyn->functable->numfields_D2TB_basebulk;
		for (unsigned int i = 0; i < dyn->functable->numfields_D2_basebulk; i++)
		{
			res[dyn->functable->fieldnames_D2[i]] = offs + i;
		}
		offs += dyn->functable->numfields_D2_basebulk;
		for (unsigned int i = 0; i < dyn->functable->numfields_D1TB_basebulk; i++)
		{
			res[dyn->functable->fieldnames_D1TB[i]] = offs + i;
		}
		offs += dyn->functable->numfields_D1TB_basebulk;
		for (unsigned int i = 0; i < dyn->functable->numfields_D1_basebulk; i++)
		{
			res[dyn->functable->fieldnames_D1[i]] = offs + i;
		}
		offs += dyn->functable->numfields_D1_basebulk;

		// Now the additional ones
		for (unsigned int i = 0; i < dyn->functable->numfields_C2TB - dyn->functable->numfields_C2TB_basebulk; i++)
		{
			res[dyn->functable->fieldnames_C2TB[i + dyn->functable->numfields_C2TB_basebulk]] = offs + i;
		}
		offs += dyn->functable->numfields_C2TB - dyn->functable->numfields_C2TB_basebulk;
		for (unsigned int i = 0; i < dyn->functable->numfields_C2 - dyn->functable->numfields_C2_basebulk; i++)
		{
			res[dyn->functable->fieldnames_C2[i + dyn->functable->numfields_C2_basebulk]] = offs + i;
		}
		offs += dyn->functable->numfields_C2 - dyn->functable->numfields_C2_basebulk;
		for (unsigned int i = 0; i < dyn->functable->numfields_C1TB - dyn->functable->numfields_C1TB_basebulk; i++)
		{
			res[dyn->functable->fieldnames_C1TB[i + dyn->functable->numfields_C1TB_basebulk]] = offs + i;
		}
		offs += dyn->functable->numfields_C1TB - dyn->functable->numfields_C1TB_basebulk;
		for (unsigned int i = 0; i < dyn->functable->numfields_C1 - dyn->functable->numfields_C1_basebulk; i++)
		{
			res[dyn->functable->fieldnames_C1[i + dyn->functable->numfields_C1_basebulk]] = offs + i;
		}
		offs += dyn->functable->numfields_C1 - dyn->functable->numfields_C1_basebulk;

		// Now the additional ones
		for (unsigned int i = 0; i < dyn->functable->numfields_D2TB - dyn->functable->numfields_D2TB_basebulk; i++)
		{
			res[dyn->functable->fieldnames_D2TB[i + dyn->functable->numfields_D2TB_basebulk]] = offs + i;
		}
		offs += dyn->functable->numfields_D2TB - dyn->functable->numfields_D2TB_basebulk;
		for (unsigned int i = 0; i < dyn->functable->numfields_D2 - dyn->functable->numfields_D2_basebulk; i++)
		{
			res[dyn->functable->fieldnames_D2[i + dyn->functable->numfields_D2_basebulk]] = offs + i;
		}
		offs += dyn->functable->numfields_D2 - dyn->functable->numfields_D2_basebulk;
		for (unsigned int i = 0; i < dyn->functable->numfields_D1TB - dyn->functable->numfields_D1TB_basebulk; i++)
		{
			res[dyn->functable->fieldnames_D1TB[i + dyn->functable->numfields_D1TB_basebulk]] = offs + i;
		}
		offs += dyn->functable->numfields_D1TB - dyn->functable->numfields_D1TB_basebulk;
		for (unsigned int i = 0; i < dyn->functable->numfields_D1 - dyn->functable->numfields_D1_basebulk; i++)
		{
			res[dyn->functable->fieldnames_D1[i + dyn->functable->numfields_D1_basebulk]] = offs + i;
		}

		return res;
	}

	std::map<std::string, unsigned> DynamicBulkElementInstance::get_elemental_field_indices()
	{
		std::map<std::string, unsigned> res;
		for (unsigned int i = 0; i < dyn->functable->numfields_DL; i++)
		{
			res[dyn->functable->fieldnames_DL[i]] = i;
		}
		for (unsigned int i = 0; i < dyn->functable->numfields_D0; i++)
		{
			res[dyn->functable->fieldnames_D0[i]] = i + dyn->functable->numfields_DL;
		}
		return res;
	}

	int DynamicBulkElementInstance::get_discontinuous_field_index(std::string name)
	{
		for (unsigned int i = 0; i < dyn->functable->numfields_DL; i++)
		{
			if (!strcmp(name.c_str(), dyn->functable->fieldnames_DL[i]))
			{
				return i + dyn->functable->internal_offset_DL;
			}
		}
		for (unsigned int i = 0; i < dyn->functable->numfields_D0; i++)
		{
			if (!strcmp(name.c_str(), dyn->functable->fieldnames_D0[i]))
			{
				return i + dyn->functable->internal_offset_D0;
			}
		}
		return -1;
	}

	int DynamicBulkElementInstance::get_nodal_field_index(std::string name)
	{
		for (unsigned int i = 0; i < dyn->functable->numfields_C2TB_basebulk; i++)
		{
			if (!strcmp(name.c_str(), dyn->functable->fieldnames_C2TB[i]))
			{
				return i;
			}
		}
		for (unsigned int i = 0; i < dyn->functable->numfields_C2_basebulk; i++)
		{
			if (!strcmp(name.c_str(), dyn->functable->fieldnames_C2[i]))
			{
				return i+ dyn->functable->numfields_C2TB_basebulk;
			}
		}
		for (unsigned int i = 0; i < dyn->functable->numfields_C1_basebulk; i++)
		{
			if (!strcmp(name.c_str(), dyn->functable->fieldnames_C1[i]))
			{
				return i + dyn->functable->numfields_C2_basebulk+ dyn->functable->numfields_C2TB_basebulk;
			}
		}
		return -1;
	}

	unsigned DynamicBulkElementInstance::resolve_interface_dof_id(std::string n)
	{
		return this->get_bulk_mesh()->resolve_interface_dof_id(n);
	}

	std::string DynamicBulkElementInstance::get_space_of_field(std::string name)
	{
		for (unsigned int i = 0; i < dyn->functable->numfields_C2TB_basebulk; i++)
		{
			if (!strcmp(name.c_str(), dyn->functable->fieldnames_C2TB[i]))
			{
				return "C2TB";
			}
		}
		for (unsigned int i = 0; i < dyn->functable->numfields_C2_basebulk; i++)
		{
			if (!strcmp(name.c_str(), dyn->functable->fieldnames_C2[i]))
			{
				return "C2";
			}
		}
		for (unsigned int i = 0; i < dyn->functable->numfields_C1_basebulk; i++)
		{
			if (!strcmp(name.c_str(), dyn->functable->fieldnames_C1[i]))
			{
				return "C1";
			}
		}
		for (unsigned int i = 0; i < dyn->functable->numfields_DL; i++)
		{
			if (!strcmp(name.c_str(), dyn->functable->fieldnames_DL[i]))
			{
				return "DL";
			}
		}
		for (unsigned int i = 0; i < dyn->functable->numfields_D0; i++)
		{
			if (!strcmp(name.c_str(), dyn->functable->fieldnames_D0[i]))
			{
				return "D0";
			}
		}
		return "";
	}

	void DynamicBulkElementInstance::sanity_check()
	{
		/*
		 for (unsigned int i=0;i<dyn->functable->numglobal_params;i++)
		 {
			if (local_global_parameter_to_global_index[i]<0) throw_runtime_error("Elemental parameter "+std::string(dyn->functable->global_paramnames[i])+" not bound");
		 }
		*/
		/*
		 for (unsigned int i=0;i<dyn->functable->numfields_C2;i++)
		 {
			if (local_field_to_global_field_index_C2[i]<0) throw_runtime_error("C2 field "+std::string(dyn->functable->fieldnames_C2[i])+" not bound");
		 }
		 for (unsigned int i=0;i<dyn->functable->numfields_C1;i++)
		 {
			if (local_field_to_global_field_index_C1[i]<0) throw_runtime_error("C1 field "+std::string(dyn->functable->fieldnames_C1[i])+" not bound");
		 }
		*/
	}

    bool DynamicBulkElementInstance::has_parameter_contribution(const std::string &param)
	{
		if (!this->get_problem()->has_global_parameter(param))
			return false;
		pyoomph::GlobalParameterDescriptor * parameter=this->get_problem()->get_global_parameter(param);
		for (unsigned int i = 0; i < dyn->functable->numglobal_params; i++)
		{
			if (dyn->functable->global_paramindices[i] == parameter->get_global_index())
				return true;
		}
		return false;
	}

    void CustomResJacInformation::set_custom_jacobian(const std::vector<double> &Jv, const std::vector<int> &col_index, const std::vector<int> &row_start)
	{
		Jvals.resize(Jv.size());
		Jcolumn_index.resize(col_index.size());
		Jrow_start.resize(row_start.size());
		for (unsigned int i = 0; i < Jv.size(); i++)
			Jvals[i] = Jv[i];
		for (unsigned int i = 0; i < col_index.size(); i++)
			Jcolumn_index[i] = col_index[i];
		for (unsigned int i = 0; i < row_start.size(); i++)
			Jrow_start[i] = row_start[i];
	}

	unsigned Problem::get_max_dt_order() const
	{
		unsigned max_order = 0;
		for (unsigned int i = 0; i < bulk_element_codes.size(); i++)
		{
			max_order = std::max(max_order, (unsigned)bulk_element_codes[i]->functable->max_dt_order);
		}
		return max_order;
	}

	void Problem::unload_all_dlls()
	{
		if (pyoomph_verbose)
			std::cout << "Unloading all DLLs" << std::endl
					  << std::flush;
		for (unsigned int i = 0; i < bulk_element_codes.size(); i++)
		{
			if (pyoomph_verbose)
				std::cout << "Unloading DLL " << bulk_element_codes[i]->get_file_name() << std::endl
						  << std::flush;
			delete bulk_element_codes[i];
		}
		if (pyoomph_verbose)
			std::cout << "DLLs unloaded " << std::endl
					  << std::flush;
		for (auto &gp : global_params_by_name)
		{

			delete gp.second;
		}

		bulk_element_codes.clear();

		global_params_by_name.clear();

		if (eigen_MassMatrixPt)
			delete eigen_MassMatrixPt;
		if (eigen_JacobianMatrixPt)
			delete eigen_JacobianMatrixPt;
	}

	Problem::~Problem()
	{
		// if (meshtemplate) delete meshtemplate; meshtemplate=NULL;
		// for (unsigned int i=0;i<fields_by_index.size();i++) delete fields_by_index[i];
		unload_all_dlls();
		// if (this->compiler) delete this->compiler;
		if (logfile)
		{
		  if (pyoomph::get_logging_stream()==logfile) pyoomph::set_logging_stream(NULL);
		  delete logfile;
		  logfile=NULL;
		  
		}
	}

	Problem::Problem() : oomph::Problem(), compiler(NULL), logfile(NULL), _is_quiet(false), bulk_element_codes(0) // , meshtemplate(new MeshTemplate(this))
	{
	}

	DynamicBulkElementCode *Problem::load_dynamic_bulk_element_code(std::string dynamic_lib, FiniteElementCode *element_class)
	{
		for (unsigned int i = 0; i < bulk_element_codes.size(); i++)
		{
			if (bulk_element_codes[i]->get_file_name() == dynamic_lib)
				return bulk_element_codes[i];
		}
		CCompiler *ccompiler = this->get_ccompiler();
		bulk_element_codes.push_back(new DynamicBulkElementCode(this, ccompiler, dynamic_lib, element_class));
		element_class->fill_callback_info(bulk_element_codes.back()->functable);
		auto *ft = bulk_element_codes.back()->functable;
		for (unsigned int i = 0; i < ft->numglobal_params; i++)
		{
			//		std::cout << "LINKING GLOBAL PARAM " << i << " of " << functable->numglobal_params << std::endl;
			//		std::cout << "codeinst->get_problem()->get_global_parameter(functable->global_paramindices[i]) << std::endl;
			ft->global_parameters[i] = &(this->get_global_parameter(ft->global_paramindices[i])->value());
		}

		return bulk_element_codes.back();
	}

	/*
	const FieldDescriptor * Problem::assert_field(const std::string & name,const FieldSpace & space )
	{
	 if (!this->has_field(name))
	 {
	  FieldDescriptor *res=new FieldDescriptor(this,name,space,fields_by_index.size());
	  fields_by_name.insert(std::pair<std::string,FieldDescriptor *>(name,res));
	  fields_by_index.push_back(res);
		return res;
	 }
	 else
	 {
	  const FieldDescriptor * res=get_field(name);
	  if (res->get_space()!=space) throw_runtime_error("Field '"+name+"' is defined on different spaces");
	  return res;
	 }
	}

	*/

	bool Problem::_set_solved_residual(std::string name,bool raise_error,bool remove_dofs_without_jacobian_row)
	{
		unsigned numfound = 0;
		if (this->_solved_residual != name) 
		{
			for (unsigned int i=0;i<removed_fields_due_to_missing_jacobian_row.size();i++) removed_fields_due_to_missing_jacobian_row[i]=false;
		}
		for (unsigned int i = 0; i < bulk_element_codes.size(); i++)
		{
			numfound += bulk_element_codes[i]->_set_solved_residual(name);
		}
		if (!numfound && raise_error)
		{
			throw_runtime_error("Cannot activate the residual-Jacobian pair named '" + name + "', since it is defined in no equations at all");
		}
		this->_solved_residual = name;
		unsigned resind=std::find(residual_names.begin(),residual_names.end(),name)-residual_names.begin();				
		std::set<unsigned> fields_with_missing_jacobian_row;
		if (remove_dofs_without_jacobian_row)
		{
			for (unsigned int i=0;i<removed_fields_due_to_missing_jacobian_row.size();i++) 
			{
				removed_fields_due_to_missing_jacobian_row[i]=pin_due_to_empty_jacobian_row[resind][i];
				if (removed_fields_due_to_missing_jacobian_row[i]) fields_with_missing_jacobian_row.insert(i);
			}
		}
		/*if (!fields_with_missing_jacobian_row.empty())
		{
			std::cout << "NOTE: The following fields have no Jacobian row in the active residual and will be pinned to their current value:" << std::endl;
			for (unsigned int i=0;i<removed_fields_due_to_missing_jacobian_row.size();i++) 
			{
				if (removed_fields_due_to_missing_jacobian_row[i]) std::cout << "  - " << this->defined_fields[i] << std::endl;
			}
		}*/
		return numfound;
	}

	bool Problem::has_empty_jacobian_rows_marked() const
	{
		for (unsigned int i=0;i<removed_fields_due_to_missing_jacobian_row.size();i++) if (removed_fields_due_to_missing_jacobian_row[i]) return true;
		return false;
	}


	double &Problem::global_parameter(const std::string &n)
	{
		GlobalParameterDescriptor *res = assert_global_parameter(n);
		return res->value();
	}

	GlobalParameterDescriptor *Problem::assert_global_parameter(const std::string &name)
	{
		if (!this->has_global_parameter(name))
		{
			GlobalParameterDescriptor *res = new GlobalParameterDescriptor(this, name, global_params_by_index.size());
			global_params_by_name.insert(std::pair<std::string, GlobalParameterDescriptor *>(name, res));
			global_params_by_index.push_back(res);
			double *valptr = &(res->value());
			this->set_analytic_dparameter(valptr); // Default to analytic derivative
			return res;
		}
		else
		{
			GlobalParameterDescriptor *res = get_global_parameter(name);
			return res;
		}
	}

	double Problem::global_temporal_error_norm()
	{
		double global_error = 0.0;
		for (unsigned int ns = 0; ns < this->nsub_mesh(); ns++)
		{
			global_error += dynamic_cast<pyoomph::Mesh *>(this->mesh_pt(ns))->get_temporal_error_norm_contribution();
		}
		if (!_is_quiet)
			std::cout << "GLOBAL TEMPORAL ERROR " << sqrt(global_error) << std::endl;
		return sqrt(global_error);
	}

	void Problem::ensure_dummy_values_to_be_dummy()
	{
		for (unsigned nmi = 0; nmi < this->nsub_mesh(); nmi++)
		{
			unsigned nelem = mesh_pt(nmi)->nelement();
			//		std::cout << "ENSURE PINNING NEL " << nelem << std::endl;
			for (unsigned n = 0; n < nelem; n++)
			{
				auto el = dynamic_cast<BulkElementBase *>(mesh_pt(nmi)->element_pt(n));
				if (el)
					el->unpin_dummy_values();
			}
		}
		for (unsigned nmi = 0; nmi < this->nsub_mesh(); nmi++)
		{
			unsigned nelem = mesh_pt(nmi)->nelement();
			for (unsigned n = 0; n < nelem; n++)
			{
				auto el = dynamic_cast<BulkElementBase *>(mesh_pt(nmi)->element_pt(n));
				if (el)
					el->pin_dummy_values();
			}
		}
	}

	void Problem::actions_after_adapt()
	{
		for (unsigned nmi = 0; nmi < this->nsub_mesh(); nmi++)
		{
			if (dynamic_cast<Mesh *>(this->mesh_pt(nmi)))
			{
				dynamic_cast<Mesh *>(this->mesh_pt(nmi))->invalidate_lagrangian_kdtree();
			}
		}
		ensure_dummy_values_to_be_dummy();
		setup_pinning();
	}

	void Problem::set_initial_condition()
	{
		oomph::Problem::set_initial_condition();
	}

	void Problem::assemble_eigenproblem_matrices(oomph::CRDoubleMatrix *&M, oomph::CRDoubleMatrix *&J, double sigma_r)
	{
		if (!M)
		{
			if (eigen_MassMatrixPt)
			{
				delete eigen_MassMatrixPt;
			}
			eigen_MassMatrixPt = new oomph::CRDoubleMatrix(this->dof_distribution_pt());
			M = eigen_MassMatrixPt;
		}
		if (!J)
		{
			if (eigen_JacobianMatrixPt)
			{
				delete eigen_JacobianMatrixPt;
			}
			eigen_JacobianMatrixPt = new oomph::CRDoubleMatrix(this->dof_distribution_pt());
			J = eigen_JacobianMatrixPt;
		}
		this->get_eigenproblem_matrices(*M, *J, sigma_r);
	}

	std::vector<double> Problem::get_history_dofs(unsigned t)
	{
		std::vector<double> res(this->ndof(), 0.0);
		oomph::DoubleVector dofs;
		if (t == 0)
			this->get_dofs(dofs);
		else
			this->get_dofs(t, dofs);
		for (unsigned int i = 0; i < this->ndof(); i++)
			res[i] = dofs[i];
		return res;
	}

	std::tuple<std::vector<double>, std::vector<bool>> Problem::get_current_dofs()
	{
		std::vector<double> res(this->ndof(), 0.0);
		std::vector<bool> is_positional(this->ndof(), false);
		oomph::DoubleVector dofs;
		this->get_dofs(dofs);
		for (unsigned int i = 0; i < this->ndof(); i++)
			res[i] = dofs[i];

		for (unsigned int ism = 0; ism < this->nsub_mesh(); ism++)
		{
			pyoomph::Mesh *m = dynamic_cast<pyoomph::Mesh *>(this->mesh_pt(ism));
			for (unsigned int in = 0; in < m->nnode(); in++)
			{
				auto *n = dynamic_cast<pyoomph::Node *>(m->node_pt(in));
				auto *vp = n->variable_position_pt();
				for (unsigned int iv = 0; iv < vp->nvalue(); iv++)
				{
					if (vp->eqn_number(iv) >= 0)
						is_positional[vp->eqn_number(iv)] = true;
				}
			}
		}

		return std::make_tuple(res, is_positional);
	}

	void Problem::set_current_dofs(const std::vector<double> &inp)
	{
		oomph::DoubleVector dofs;
		dofs.build(this->dof_distribution_pt(), 0.0);
		if (inp.size() != this->ndof())
			throw_runtime_error("Mismatch in dof vector size");
		for (unsigned int i = 0; i < this->ndof(); i++)
			dofs[i] = inp[i];
		this->set_dofs(dofs);
	}


    void Problem::get_dofs(const unsigned& t, oomph::DoubleVector& dofs) const
	{	
      oomph::Problem::get_dofs(t,dofs);
	  //std::cout << "GET HISTORY DOFS " << t << std::endl;
	  for (unsigned i = 0, ni = mesh_pt()->nnode(); i < ni; i++)
      {
       pyoomph::Node* node_pt = dynamic_cast<pyoomph::Node*>(mesh_pt()->node_pt(i));
	   if (!node_pt) continue;
       for (unsigned j = 0, nj = node_pt->variable_position_pt()->nvalue(); j < nj; j++)
       {        
        int eqn_number = node_pt->variable_position_pt()->eqn_number(j);
        if (eqn_number >= 0)
        {
          dofs[eqn_number]=node_pt->variable_position_pt()->value(t, j);
        }
       }
      }	
	}

	void Problem::set_dofs(const unsigned& t, oomph::DoubleVector& dof_pt) 
	{
	 oomph::Problem::set_dofs(t,dof_pt);
	 //std::cout << "SET HISTORY DOFS " << t << std::endl;
	 // But oomph-lib forgot the variable position pt of moving nodes...
     for (unsigned i = 0, ni = mesh_pt()->nnode(); i < ni; i++)
     {
      pyoomph::Node* node_pt = dynamic_cast<pyoomph::Node*>(mesh_pt()->node_pt(i));
	  if (!node_pt) continue;
      for (unsigned j = 0, nj = node_pt->variable_position_pt()->nvalue(); j < nj; j++)
      {        
        int eqn_number = node_pt->variable_position_pt()->eqn_number(j);
        if (eqn_number >= 0)
        {
          node_pt->variable_position_pt()->set_value(t, j, dof_pt[eqn_number]);
        }
      }
     }		 	
	}

    void Problem::set_dofs(const unsigned& t, oomph::Vector<double*>& dof_pt) 
	{
     oomph::Problem::set_dofs(t,dof_pt);
	 //std::cout << "SET HISTORY DOFS " << t << std::endl;
	 // But oomph-lib forgot the variable position pt of moving nodes...
     for (unsigned i = 0, ni = mesh_pt()->nnode(); i < ni; i++)
     {
      pyoomph::Node* node_pt = dynamic_cast<pyoomph::Node*>(mesh_pt()->node_pt(i));
	  if (!node_pt) continue;
      for (unsigned j = 0, nj = node_pt->variable_position_pt()->nvalue(); j < nj; j++)
      {        
        int eqn_number = node_pt->variable_position_pt()->eqn_number(j);
        if (eqn_number >= 0)
        {
          node_pt->variable_position_pt()->set_value(t, j, *(dof_pt[eqn_number]));
        }
      }
     }
	}

	void Problem::set_history_dofs(unsigned t, const std::vector<double> &inp)
	{
		oomph::DoubleVector dofs;
		dofs.build(this->dof_distribution_pt(), 0.0);
		if (inp.size() != this->ndof())
			throw_runtime_error("Mismatch in dof vector size");
		if (t>=this->time_stepper_pt()->ntstorage()) 
		        throw_runtime_error("Wrong history offset");
		for (unsigned int i = 0; i < this->ndof(); i++)
			dofs[i] = inp[i];
		this->set_dofs(t, dofs);
	}

	std::vector<double> Problem::get_current_pinned_values(bool with_pos)
	{
		std::vector<double> res;
		for (unsigned int ism = 0; ism < this->nsub_mesh(); ism++)
		{
			pyoomph::Mesh *m = dynamic_cast<pyoomph::Mesh *>(this->mesh_pt(ism));
			for (unsigned int in = 0; in < m->nnode(); in++)
			{
				auto *n = m->node_pt(in);
				for (unsigned int iv = 0; iv < n->nvalue(); iv++)
				{
					if (n->is_pinned(iv))
						res.push_back(n->value(iv));
				}
				if (with_pos)
				{
					for (unsigned int iv = 0; iv < n->ndim(); iv++)
					{
						if (dynamic_cast<pyoomph::Node *>(n)->variable_position_pt()->is_pinned(iv))
							res.push_back(dynamic_cast<pyoomph::Node *>(n)->variable_position_pt()->value(iv));
					}
				}
			}
			for (unsigned int ie = 0; ie < m->nelement(); ie++)
			{
				auto *e = m->element_pt(ie);
				for (unsigned int iid = 0; iid < e->ninternal_data(); iid++)
				{
					auto *id = e->internal_data_pt(iid);
					for (unsigned int iv = 0; iv < id->nvalue(); iv++)
					{
						if (id->is_pinned(iv))
							res.push_back(id->value(iv));
					}
				}
			}
		}
		return res;
	}

	void Problem::get_residuals(oomph::DoubleVector &residuals)
	{
		if (!use_custom_residual_jacobian)
		{
			get_residuals_by_elemental_assembly(residuals);
		}
		else
		{
			CustomResJacInformation info(false,"");
			get_custom_residuals_jacobian(&info);
			if (!residuals.built())
			{
				oomph::LinearAlgebraDistribution dist(this->communicator_pt(), info.residuals.size(), false);
				residuals.build(&dist, 0.0);
			}
			for (unsigned int i = 0; i < info.residuals.size(); i++)
				residuals[i] = info.residuals[i];
		}
	}

	void Problem::get_derivative_wrt_global_parameter(double* const& parameter_pt,oomph::DoubleVector& result)
	{
		if (!use_custom_residual_jacobian)
		{
			get_derivative_wrt_global_parameter_elemental_assembly(parameter_pt,result);
		}
		else
		{
			int pindex=this->resolve_parameter_value_ptr(parameter_pt);
			if (pindex<0) throw_runtime_error("Cannot resolve the double pointer of a global parameter to this problem");			
			CustomResJacInformation info(false,global_params_by_index[pindex]->get_name());
			get_custom_residuals_jacobian(&info);
			if (!result.built())
			{
				oomph::LinearAlgebraDistribution dist(this->communicator_pt(), info.residuals.size(), false);
				result.build(&dist, 0.0);
			}
			for (unsigned int i = 0; i < info.residuals.size(); i++)
				result[i] = info.residuals[i];
		}

	}
	void Problem::get_jacobian(oomph::DoubleVector &residuals, oomph::CRDoubleMatrix &jacobian)
	{
		if (!use_custom_residual_jacobian)
		{
			get_jacobian_by_elemental_assembly(residuals, jacobian);
		}
		else
		{
			CustomResJacInformation info(true,"");
			get_custom_residuals_jacobian(&info);
			//       std::cout << "RET FROM PYTH" << std::endl;

			if (!residuals.built())
			{
				oomph::LinearAlgebraDistribution dist(this->communicator_pt(), info.residuals.size(), false);
				residuals.build(&dist, 0.0);
			}
			for (unsigned int i = 0; i < info.residuals.size(); i++)
				residuals[i] = info.residuals[i];

			//       std::cout << "BUILD J  "<< info.residuals.size() << "  " << info.Jcolumn_index.size() << "  " << info.Jrow_start.size() << std::endl;
			jacobian.build(info.residuals.size(), info.Jvals, info.Jcolumn_index, info.Jrow_start);
			//       std::cout << "DONE BUILD J" << std::endl;
		}
	}

	int Problem::resolve_parameter_value_ptr(double *ptr)
	{
		for (const auto &a : global_params_by_name)
		{
			if ((&a.second->value()) == ptr)
				return a.second->get_global_index();
		}
		throw_runtime_error("Cannot resolve the double pointer of a global parameter to this problem");
		return -1;
	}

	void Problem::set_arclength_parameter(std::string nam, double val)
	{
		if (nam == "Desired_proportion_of_arc_length")
			Desired_proportion_of_arc_length = val;
		else if (nam == "Scale_arc_length")
			Scale_arc_length = (val > 0.5 ? true : false);
		else if (nam == "Use_finite_differences_for_continuation_derivatives")
			Use_finite_differences_for_continuation_derivatives = (val > 0.5 ? true : false);
		else if (nam == "Use_continuation_timestepper")
			Use_continuation_timestepper = (val > 0.5 ? true : false);
		else if (nam == "Desired_newton_iterations_ds")
		   Desired_newton_iterations_ds=val;
		else
			throw_runtime_error("Unknown param to set " + nam);
	}

	void Problem::_replace_RJM_by_param_deriv(std::string name, bool active)
	{
		if (!active)
			__replace_RJM_by_param_deriv = NULL;
		else
		{
			if (!global_params_by_name.count(name))
				throw_runtime_error("Cannot replace residuals/jacobian/mass matrix by parameter derivatives for global parameter " + name + ", since it is not present in the problem");
			auto *p = global_params_by_name[name];
			__replace_RJM_by_param_deriv = &(p->value());
		}
	}

	double Problem::arc_length_step(const std::string param, const double &ds, unsigned max_adapt)
	{
		if (!global_params_by_name.count(param))
			throw_runtime_error("Cannot continue in the global parameter " + param + ", since it is not present in the problem");
		auto *p = global_params_by_name[param];
		double *valptr = &(p->value());
		//		this->set_analytic_dparameter(valptr);
		return this->arc_length_step_solve(valptr, ds, max_adapt);
	}

	std::vector<double> Problem::get_arclength_dof_derivative_vector()
	{
		std::vector<double> res(Dof_derivative.size());
		for (unsigned i = 0; i < res.size(); i++)
			res[i] = dof_derivative(i);
		return res;
	}

	std::vector<double> Problem::get_arclength_dof_current_vector()
	{
		std::vector<double> res(Dof_current.size());
		for (unsigned i = 0; i < res.size(); i++)
			res[i] = dof_current(i);
		return res;
	}

    void Problem::update_dof_vectors_for_continuation(const std::vector<double> &ddof, const std::vector<double> &curr)
    {
		if (ddof.size() != curr.size()) throw_runtime_error("Mismatch in size of ddof and curr");
		unsigned ndof_local = Dof_distribution_pt->nrow_local();
		if (ddof.size() != ndof_local)
		{
			throw_runtime_error("Mismatch in size of ddof and current dof vectors");
		}
		if (Dof_derivative.size() != ndof_local)
		{
			Dof_derivative.resize(ndof_local, 0.0);
		}
		if (Dof_current.size() != ndof_local)
		{
			Dof_current.resize(ndof_local, 0.0);
		}
		for (unsigned i = 0; i < ndof_local; i++)
		{
			Dof_derivative[i] = ddof[i];
			Dof_current[i] = curr[i];
		}
    }

	void Problem::update_param_info_for_continuation(double dp,double p0)
	{
		Parameter_current=p0;
		if (dp>0) Continuation_direction=1; else  if (dp<0) Continuation_direction=-1; 
		Parameter_derivative=abs(dp);
		
		Arc_length_step_taken=true;
	}

	

    void Problem::activate_my_fold_tracking(double *const &parameter_pt, const oomph::DoubleVector &eigenvector, const bool &block_solve)
	{
		reset_assembly_handler_to_default();
		this->assembly_handler_pt() = new MyFoldHandler(this, parameter_pt, eigenvector);
		if (block_solve)
		{
			this->linear_solver_pt() = new oomph::AugmentedBlockFoldLinearSolver(this->linear_solver_pt());
		}
	}

	void Problem::activate_my_fold_tracking(double *const &parameter_pt, const bool &block_solve)
	{
		reset_assembly_handler_to_default();
		this->assembly_handler_pt() = new MyFoldHandler(this, parameter_pt);
		if (block_solve)
		{
			this->linear_solver_pt() = new oomph::AugmentedBlockFoldLinearSolver(this->linear_solver_pt());
		}
	}

	void Problem::activate_my_hopf_tracking(double *const &parameter_pt, const double &omega, const oomph::DoubleVector &null_real, const oomph::DoubleVector &null_imag, const bool &block_solve)
	{
		reset_assembly_handler_to_default();
		this->assembly_handler_pt() = new MyHopfHandler(this, parameter_pt, omega, null_real, null_imag);
		if (block_solve)
		{
			this->linear_solver_pt() = new oomph::BlockHopfLinearSolver(this->linear_solver_pt());
		}
	}

	void Problem::actions_after_change_in_global_parameter(double *const &parameter_pt)
	{
		for (auto &p : this->global_params_by_index)
		{
			if (&(p->value()) == parameter_pt)
			{
				this->actions_after_change_in_global_parameter(p->get_name());
			}
		}
	}

	void Problem::actions_after_parameter_increase(double *const &parameter_pt)
	{
		for (auto &p : this->global_params_by_index)
		{
			if (&(p->value()) == parameter_pt)
			{
				this->actions_after_parameter_increase(p->get_name());
			}
		}
	}

	void Problem::activate_my_azimuthal_tracking(double *const &parameter_pt, const double &omega, const oomph::DoubleVector &null_real, const oomph::DoubleVector &null_imag, std::map<std::string, std::string> special_residual_forms)
	{
		reset_assembly_handler_to_default();
		if (!special_residual_forms.count("azimuthal_real_eigen"))
		{
			throw_runtime_error("You have not specified a azimuthal_real_eigen as special residual");
		}
		if (!special_residual_forms.count("azimuthal_imag_eigen"))
		{
			throw_runtime_error("You have not specified a azimuthal_imag_eigen as special residual");
		}
		bool has_imag=special_residual_forms["azimuthal_imag_eigen"]!="<NONE>";
		AzimuthalSymmetryBreakingHandler *azi = new AzimuthalSymmetryBreakingHandler(this, parameter_pt, null_real, null_imag, omega,has_imag);

		azi->setup_solved_azimuthal_contributions(special_residual_forms["azimuthal_real_eigen"], special_residual_forms["azimuthal_imag_eigen"]);
		this->assembly_handler_pt() = azi;
	}

	void Problem::activate_my_pitchfork_tracking(double *const &parameter_pt, const oomph::DoubleVector &symmetry_vector, const bool &block_solve)
	{
		//	this->activate_pitchfork_tracking(parameter_pt, symmetry_vector, block_solve);
		reset_assembly_handler_to_default();
		this->assembly_handler_pt() = new MyPitchForkHandler(this, parameter_pt, symmetry_vector);
	}

	std::vector<double> Problem::get_parameter_derivative(const std::string param)
	{
		if (!global_params_by_name.count(param))
			throw_runtime_error("Cannot derive wrt unknown global parameter " + param);
		auto *p = global_params_by_name[param];
		double *valptr = &(p->value());
		//		this->set_analytic_dparameter(valptr);
		oomph::DoubleVector resdv(this->dof_distribution_pt());
		resdv.clear();
		get_derivative_wrt_global_parameter(valptr, resdv);
		std::vector<double> res(this->ndof());
		for (unsigned int i = 0; i < res.size(); i++)
			res[i] = resdv[i];
		return res;
	}

	void Problem::after_bifurcation_tracking_step()
	{
		if (dynamic_cast<MyFoldHandler *>(this->assembly_handler_pt()))
		{
			dynamic_cast<MyFoldHandler *>(this->assembly_handler_pt())->realign_C_vector();
		}
	}

	void Problem::set_dof_direction_arclength(std::vector<double> ddir)
	{
		this->reset_arc_length_parameters();
		const unsigned long ndof_local = this->Dof_distribution_pt->nrow_local();
		if (ddir.size() != ndof_local)
			throw_runtime_error("Mismatching size in the dof direction vector and the actual number of DoFs:" + std::to_string(ddir.size()) + " vs " + std::to_string(ndof_local));
		this->Arc_length_step_taken = true;
		if (!this->Use_continuation_timestepper)
		{
			if (this->Dof_derivative.size() != ndof_local)
			{
				this->Dof_derivative.resize(ndof_local, 0.0);
			}
		}
		for (unsigned int i = 0; i < ddir.size(); i++)
			dof_derivative(i) = ddir[i];
	}

	double Problem::get_bifurcation_omega()
	{
		if (bifurcation_tracking_mode == "hopf" && (dynamic_cast<MyHopfHandler *>(this->assembly_handler_pt())))
		{
			return dynamic_cast<MyHopfHandler *>(this->assembly_handler_pt())->omega();
		}
		else if (bifurcation_tracking_mode == "azimuthal" && (dynamic_cast<AzimuthalSymmetryBreakingHandler *>(this->assembly_handler_pt())))
		{
			return dynamic_cast<AzimuthalSymmetryBreakingHandler *>(this->assembly_handler_pt())->omega();
		}
		else
		{
			return 0.0;
		}
	}

    void Problem::set_sparse_assembly_method(const std::string &method)
    {
		/*Perform_assembly_using_vectors_of_pairs,
      Perform_assembly_using_two_vectors,
      Perform_assembly_using_maps,
      Perform_assembly_using_lists,
      Perform_assembly_using_two_arrays*/
		if (method == "vectors_of_pairs")
		{
			Sparse_assembly_method=Perform_assembly_using_vectors_of_pairs;
		}
		else if (method == "two_vectors")
		{
			Sparse_assembly_method=Perform_assembly_using_two_vectors;
		}
		else if (method == "maps")
		{
			Sparse_assembly_method=Perform_assembly_using_maps;
		}
		else if (method == "lists")
		{
			Sparse_assembly_method=Perform_assembly_using_lists;
		}
		else if (method == "two_arrays")
		{
			Sparse_assembly_method=Perform_assembly_using_two_arrays;
		}
		else
		{
			throw_runtime_error("Unknown sparse assembly method: " + method);
		}
    }


	std::string Problem::get_sparse_assembly_method()
	{
		switch (Sparse_assembly_method)
		{
		case Perform_assembly_using_vectors_of_pairs:
			return "vectors_of_pairs";
		case Perform_assembly_using_two_vectors:
			return "two_vectors";
		case Perform_assembly_using_maps:
			return "maps";
		case Perform_assembly_using_lists:
			return "lists";
		case Perform_assembly_using_two_arrays:
			return "two_arrays";
		default:
			return "unknown";
		}
	}

    std::vector<std::complex<double>> Problem::get_bifurcation_eigenvector()
    {
		if (bifurcation_tracking_mode == "")
			return std::vector<std::complex<double>>();
		oomph::Vector<oomph::DoubleVector> be;
		this->get_bifurcation_eigenfunction(be);
		std::vector<std::complex<double>> res(be[0].nrow());
		if (be.size() == 1)
		{
			for (unsigned int i = 0; i < be[0].nrow(); i++)
				res[i] = std::complex<double>(be[0][i], 0.0);
		}
		else
		{
			for (unsigned int i = 0; i < be[0].nrow(); i++)
				res[i] = std::complex<double>(be[0][i], be[1][i]);
		}
		return res;
	}

	void Problem::start_orbit_tracking(const std::vector<std::vector<double>> &history, const double &T,int bspline_order,int gl_order,std::vector<double> knots,unsigned T_constraint_mode)
	
	{
		reset_assembly_handler_to_default();
		this->assembly_handler_pt() = new PeriodicOrbitHandler(this, T,history,bspline_order,gl_order,knots,T_constraint_mode);
	}

	void Problem::reset_assembly_handler_to_default()
	{
		/*if (dynamic_cast<pyoomph::PythonAssemblyHandler *>(assembly_handler_pt()))
		{
      		dynamic_cast<pyoomph::PythonAssemblyHandler *>(assembly_handler_pt())->finalize(this);
			assembly_handler_pt()=new oomph::AssemblyHandler(); // Dummy to be deleted by the super call
			oomph::Problem::reset_assembly_handler_to_default();
		}
		else
		{*/
			oomph::Problem::reset_assembly_handler_to_default();
		//}
	}

	void Problem::reset_augmented_dof_vector_to_nonaugmented()
	{
		if (n_unaugmented_dofs == 0)
			return;		
		this->GetDofPtr().resize(n_unaugmented_dofs);
    	this->GetDofDistributionPt()->build(this->communicator_pt(),n_unaugmented_dofs, false);    
    	this->GetSparcseAssembleWithArraysPA().resize(0);
		n_unaugmented_dofs=0;
	}

	/*void Problem::start_custom_augmented_system(oomph::AssemblyHandler *handler)
	{
		
		reset_assembly_handler_to_default();
		if (dynamic_cast<pyoomph::PythonAssemblyHandler *>(handler))
		{
			dynamic_cast<pyoomph::PythonAssemblyHandler *>(handler)->initialize(this);
			this->assembly_handler_pt() = handler;
		}		
		else
		{
			throw_runtime_error("Cannot set a non-python assembly handler");
		}
		
	}*/


	void Problem::start_bifurcation_tracking(const std::string param, const std::string typus, const bool &blocksolve, const std::vector<double> &eigenv1, const std::vector<double> &eigenv2, const double &omega, std::map<std::string, std::string> special_residual_forms)
	{
		if (param == "" || typus == "" || typus == "none")
		{
			bifurcation_tracking_mode = "";
			this->deactivate_bifurcation_tracking();
			return;
		}
		double *valptr;
		if (param!="<LAMBDA_TRACKING>")
		{
			if (!global_params_by_name.count(param))
				throw_runtime_error("Cannot track a bifuraciton in the global parameter " + param + ", since it is not present in the problem");
			auto *p = global_params_by_name[param];
			valptr = &(p->value());
		}
		else
		{
			valptr=&this->lambda_tracking_real;
		}
		
		//		this->set_analytic_dparameter(valptr);
		oomph::DoubleVector ev1(this->dof_distribution_pt());
		for (unsigned i = 0; i < std::min((size_t)eigenv1.size(), (size_t)this->ndof()); i++)
		{
			ev1[i] = eigenv1[i];
		}
		oomph::DoubleVector ev2(this->dof_distribution_pt());
		for (unsigned i = 0; i < std::min((size_t)eigenv2.size(), (size_t)this->ndof()); i++)
		{
			ev2[i] = eigenv2[i];
		}
		if (typus == "fold")
		{
			bifurcation_tracking_mode = "fold";
			if (eigenv1.empty())
				this->activate_my_fold_tracking(valptr, blocksolve);
			else
				this->activate_my_fold_tracking(valptr, ev1, blocksolve);
		}
		else if (typus == "hopf")
		{
			bifurcation_tracking_mode = "hopf";
			this->activate_my_hopf_tracking(valptr, omega, ev1, ev2, blocksolve);
			//    this->activate_hopf_tracking(valptr,omega,ev1,ev2,blocksolve);
		}
		else if (typus == "azimuthal")
		{
			bifurcation_tracking_mode = "azimuthal";
			this->activate_my_azimuthal_tracking(valptr, omega, ev1, ev2, special_residual_forms);
		}
		else if (typus == "cartesian_normal_mode")
		{
			bifurcation_tracking_mode = "cartesian_normal_mode";
			this->activate_my_azimuthal_tracking(valptr, omega, ev1, ev2, special_residual_forms);
		}		
		else if (typus == "pitchfork")
		{
			bifurcation_tracking_mode = "pitchfork";
			this->activate_my_pitchfork_tracking(valptr, ev1, blocksolve);
			//    this->activate_hopf_tracking(valptr,omega,ev1,ev2,blocksolve);
		}
		else
			throw_runtime_error("Cannot track unknown bifurcation type: " + typus);
	}

	void Problem::set_current_pinned_values(const std::vector<double> &inp, bool with_pos,unsigned t)
	{
		unsigned int pos = 0;
		unsigned mpos = inp.size();
		for (unsigned int ism = 0; ism < this->nsub_mesh(); ism++)
		{
			pyoomph::Mesh *m = dynamic_cast<pyoomph::Mesh *>(this->mesh_pt(ism));
			for (unsigned int in = 0; in < m->nnode(); in++)
			{
				auto *n = m->node_pt(in);
				for (unsigned int iv = 0; iv < n->nvalue(); iv++)
				{
					if (n->is_pinned(iv))
					{
						n->set_value(t,iv, inp[pos++]);
						if (pos > mpos)
							throw_runtime_error("Mismatch in value vector size: " + std::to_string(mpos) + " given, but reached index " + std::to_string(pos));
					}
				}
				if (with_pos)
				{
					for (unsigned int iv = 0; iv < n->ndim(); iv++)
					{
						if (dynamic_cast<pyoomph::Node *>(n)->variable_position_pt()->is_pinned(iv))
							dynamic_cast<pyoomph::Node *>(n)->variable_position_pt()->set_value(t,iv, inp[pos++]);
					}
				}
			}
			for (unsigned int ie = 0; ie < m->nelement(); ie++)
			{
				auto *e = m->element_pt(ie);
				for (unsigned int iid = 0; iid < e->ninternal_data(); iid++)
				{
					auto *id = e->internal_data_pt(iid);
					for (unsigned int iv = 0; iv < id->nvalue(); iv++)
					{
						if (id->is_pinned(iv))
						{
							id->set_value(t,iv, inp[pos++]);
							if (pos > mpos)
								throw_runtime_error("Mismatch in value vector size: " + std::to_string(mpos) + " given, but reached index " + std::to_string(pos));
						}
					}
				}
			}
		}
	}
	
	
	void Problem::open_log_file(const std::string &fname,const bool & activate_logging)
	{

		if (fname=="")
		{
			if (activate_logging) pyoomph::set_logging_stream(this->logfile);
			else 
			{
				if (pyoomph::get_logging_stream()==logfile) pyoomph::set_logging_stream(NULL);
				if (logfile) delete logfile;
				logfile=NULL;
			}
			return;
		}
		if (activate_logging && logfile)
		{
			if (pyoomph::get_logging_stream()==logfile) pyoomph::set_logging_stream(NULL);
			delete logfile;
			logfile=NULL;
		}
		logfile=new std::ofstream(fname.c_str());
		if (!logfile->is_open()) throw_runtime_error("Cannot open log file "+fname);
		if (activate_logging) pyoomph::set_logging_stream(logfile);
	}

	void Problem::quiet(bool _quiet)
	{
		_is_quiet = _quiet;
		Shut_up_in_newton_solve = _quiet;
		if (_quiet)
		{
			this->linear_solver_pt()->disable_doc_time();
			oomph::oomph_info.stream_pt() = &oomph::oomph_nullstream;
		}
		else
		{
			this->linear_solver_pt()->enable_doc_time();
			oomph::oomph_info.stream_pt() = &std::cout;
		}
	}

	std::vector<double> Problem::get_second_order_directional_derivative(std::vector<double> dir)
	{
		if (dof_distribution_pt()->nrow_local()!=dir.size()) throw_runtime_error("Mismatch in size of dir vector and the number of DoFs");
		
		/*
		oomph::DoubleVectorWithHaloEntries d1;
		oomph::Vector<oomph::DoubleVectorWithHaloEntries> d2(1);
		oomph::Vector<oomph::DoubleVectorWithHaloEntries> res(1);
    	d1.build(dof_distribution_pt(), 0.0);
    	d2[0].build(dof_distribution_pt(), 0.0);
		res[0].build(dof_distribution_pt(), 0.0);
		for (unsigned int i=0;i<dir.size();i++) 
		{
			d1[i]=dir[i];
			d2[0][i]=dir[i];
		}
		this->get_hessian_vector_products(d1,d2,res);
		std::vector<double> result(this->ndof(), 0.0);
		for (unsigned int i=0;i<this->ndof();i++) result[i]=0.5*res[0][i];
		return result;
		*/

		std::vector<double> result(this->ndof(), 0.0);
		const unsigned long n_elements = mesh_pt()->nelement();
		for (unsigned int ne = 0; ne < n_elements; ne++)
		{
			BulkElementBase *elem_pt = dynamic_cast<BulkElementBase *>(mesh_pt()->element_pt(ne));
			const unsigned nvar = assembly_handler_pt()->ndof(elem_pt);
			oomph::DenseMatrix<double> hessian_buffer(nvar, nvar * nvar, 0.0);
			elem_pt->assemble_hessian_tensor(hessian_buffer);
			for (unsigned int i = 0; i < nvar; i++)
			{
				unsigned iG = assembly_handler_pt()->eqn_number(elem_pt, i);
				for (unsigned int j = 0; j < nvar; j++)
				{
					unsigned jG = assembly_handler_pt()->eqn_number(elem_pt, j);
					for (unsigned int k = 0; k < nvar; k++)
					{
						double hval = hessian_buffer(i, k * nvar + j);						
						unsigned kG = assembly_handler_pt()->eqn_number(elem_pt, k);
						result[iG]+=hval*dir[jG]*dir[kG];
					}
				}
			}
		}
		return result;
	}

	SparseRank3Tensor Problem::assemble_hessian_tensor(bool symmetric)
	{
		SparseRank3Tensor result(this->ndof(), symmetric);
		const unsigned long n_elements = mesh_pt()->nelement();
		for (unsigned int ne = 0; ne < n_elements; ne++)
		{
			BulkElementBase *elem_pt = dynamic_cast<BulkElementBase *>(mesh_pt()->element_pt(ne));
			const unsigned nvar = assembly_handler_pt()->ndof(elem_pt);
			oomph::DenseMatrix<double> hessian_buffer(nvar, nvar * nvar, 0.0);
			elem_pt->assemble_hessian_tensor(hessian_buffer);
			for (unsigned int i = 0; i < nvar; i++)
			{
				unsigned iG = assembly_handler_pt()->eqn_number(elem_pt, i);
				for (unsigned int j = 0; j < nvar; j++)
				{
					unsigned jG = assembly_handler_pt()->eqn_number(elem_pt, j);
					for (unsigned int k = 0; k < nvar; k++)
					{
						double hval = hessian_buffer(i, k * nvar + j);
						if (std::fabs(hval) > Numerical_zero_for_sparse_assembly)
						{
							unsigned kG = assembly_handler_pt()->eqn_number(elem_pt, k);
							result.accumulate(iG, jG, kG, hval);
						}
					}
				}
			}
		}
		return result;
	}


//#define PYOOMPH_PERIODIC_ORBIT_BAND_MATRIX  
#ifdef PYOOMPH_PERIODIC_ORBIT_BAND_MATRIX  
	class PeriodicOrbitAssemblyBlockDenseMatrix : public oomph::DenseMatrix<double>
	{
		private:
			unsigned NT;
			unsigned base_ndof;
			//std::map<unsigned,std::map<unsigned,oomph::DenseMatrix<double>>> block_data;
			double ***block_data;
		public:
			PeriodicOrbitAssemblyBlockDenseMatrix(unsigned _NT) : oomph::DenseMatrix<double>(), NT(_NT), base_ndof(0), block_data(NULL)
			{
				block_data=new double**[NT+1]();
				for (unsigned i = 0; i < NT+1; i++)
				{
					block_data[i]=new double*[NT+1]();					
				}

			}

			void clear_block_data()
			{
					for (unsigned i = 0; i < NT+1; i++)
					{
						for (unsigned j = 0; j < NT+1; j++)
						{
							if (block_data[i][j])delete block_data[i][j];
						}
						delete block_data[i];
					}
			}

			~PeriodicOrbitAssemblyBlockDenseMatrix()
			{
				if (block_data)
				{
					clear_block_data();
					delete block_data;
				}
			}
			
			void resize(const unsigned long& n)
    		{				
      			oomph::DenseMatrix<double>::resize(n); //TODO: Remove
				N=n;
				M=n;
				if ((n-1)%NT!=0) throw_runtime_error("Invalid size for block matrix");
				//if (base_ndof!=(n-1)/NT) block_data.clear();							
				if (block_data)
				{
					clear_block_data();
					delete block_data;
				}
				block_data=new double**[NT+1]();
				for (unsigned i = 0; i < NT+1; i++)
				{
					block_data[i]=new double*[NT+1]();					
				}
				base_ndof=(n-1)/NT;				
    		}
        		
    		void initialise(const double& val)
    		{
				oomph::DenseMatrix<double>::initialise(val); //TODO: Remove
				if (val!=0.0) throw_runtime_error("Cannot initialise block matrix with non-zero value");
				if (block_data) clear_block_data();
    		}
			
    		void resize(const unsigned long& n, const unsigned long& m)
			{
				throw_runtime_error("Cannot resize block matrix like this");
			}
    
    		void resize(const unsigned long& n,const unsigned long& m,const double& initial_value)
			{
				throw_runtime_error("Cannot resize block matrix like this");
			}

    		inline double& entry(const unsigned long& i, const unsigned long& j) override
			{		
				unsigned ib=i/base_ndof;
				unsigned jb=j/base_ndof;
				unsigned ioff=i%base_ndof;
				unsigned joff=j%base_ndof;
				if (!block_data[ib]) block_data[ib]=new double*[NT+1]();
				if (!block_data[ib][jb]) block_data[ib][jb]=new double[base_ndof*base_ndof]();
				return block_data[ib][jb][ioff*base_ndof+joff];
			}
    
    		inline double get_entry(const unsigned long& i, const unsigned long& j) const
    		{      
				unsigned ib=i/base_ndof;
				unsigned jb=j/base_ndof;
				unsigned ioff=i%base_ndof;
				unsigned joff=j%base_ndof;
				if (!block_data[ib]) return 0.0;
				if (!block_data[ib][jb]) return 0.0;
				return block_data[ib][jb][ioff*base_ndof+joff];			
			}
				
			inline double operator()(const unsigned long& i, const unsigned long& j) const
    		{
      			return (this)->get_entry(i, j);
    		}
    
    		inline double& operator()(const unsigned long& i, const unsigned long& j)
    		{
      			return (this)->entry(i, j);
    		}

			const double ***get_block_data() const
			{
				return (const double ***)block_data;
			}
			unsigned get_numblocks() const
			{
				return NT+1;
			}

			unsigned get_nbasedof() const
			{
				return base_ndof;
			}

	};




	class PeriodicOrbitAssemblyBlockBandMatrix : public oomph::DenseMatrix<double>
	{
		/* 
			A periodic band matrix (consisting of NTxNT blocks) with bandwidth b
			Also, an additional row and column is added at the end (for the period constraint)
		*/
		protected:
			unsigned NT; // Number of blocks
			unsigned bandwidth; // Bandwidth
			unsigned base_ndof; // Number of dofs per block
			oomph::Vector<double> data;
		public:
			PeriodicOrbitAssemblyBlockBandMatrix(unsigned _NT,unsigned _b) : oomph::DenseMatrix<double>(), NT(_NT), bandwidth(_b), base_ndof(0), data()
			{
				
			}

			~PeriodicOrbitAssemblyBlockBandMatrix()
			{
				//if (data) delete data;
			}

			void resize(const unsigned long& n)
    		{		
				// TODO: Potentially do not realloc here if N==M==n 
      			oomph::DenseMatrix<double>::resize(n); //TODO: Remove
				N=n;
				M=n;
				if ((n-1)%NT!=0) throw_runtime_error("Invalid size for block matrix");
				base_ndof=(n-1)/NT;
				data.resize(((2*bandwidth+1)*base_ndof*base_ndof+1)*NT+n);						
    		}
        		
    		void initialise(const double& val)
    		{
				std::cout << "INITIALISE " << val << std::endl;
				oomph::DenseMatrix<double>::initialise(val); //TODO: Remove
				data.initialise(val);
    		}
			
    		void resize(const unsigned long& n, const unsigned long& m)
			{
				throw_runtime_error("Cannot resize block matrix like this");
			}
    
    		void resize(const unsigned long& n,const unsigned long& m,const double& initial_value)
			{
				throw_runtime_error("Cannot resize block matrix like this");
			}

			inline unsigned get_dataindex(const unsigned long& i, const unsigned long& j) const
			{
				std::cout << "GET DATA INDEX " << i << " " << j << std::endl;
				unsigned ib=i/base_ndof;
				if (ib>=NT) 
				{
					throw_runtime_error("TODO TIME COL");
				}
				unsigned jb=j/base_ndof;
				if (jb>=NT) 
				{
					throw_runtime_error("TODO TIME ROW");
				}
				int diff=(int)jb-(int)ib;
				if (diff>(int)bandwidth)
				{
					throw_runtime_error("TODO BANDWIDTH1");
				}
				else if  (-diff>(int)bandwidth)
				{
					throw_runtime_error("TODO BANDWIDTH2");
				}
				unsigned offset=ib*((2*bandwidth+1)*base_ndof*base_ndof+1); // row block offset
				int blockindexj=bandwidth+diff;
				offset+=(bandwidth+diff)*base_ndof*base_ndof; // column block offset
				unsigned ioff=i%base_ndof;
				unsigned joff=j%base_ndof;

				return offset+(ioff*base_ndof+joff);
			}

			inline double& entry(const unsigned long& i, const unsigned long& j) override
			{		
				return data[get_dataindex(i,j)];
			}
    
    		inline double get_entry(const unsigned long& i, const unsigned long& j) const override
    		{      
				return data[get_dataindex(i,j)];
			}
				
			 double operator()(const unsigned long& i, const unsigned long& j) const override
    		{
      			return (this)->get_entry(i, j);
    		}
    
    		 double& operator()(const unsigned long& i, const unsigned long& j) override
    		{
				std::cout << "OPERATOR " << i << " " << j << std::endl;
      			return (this)->entry(i, j);
    		}

			

	};

#endif

 	void Problem::sparse_assemble_row_or_column_compressed_for_periodic_orbit(oomph::Vector<int*>& column_or_row_index,oomph::Vector<int*>& row_or_column_start,oomph::Vector<double*>& value,oomph::Vector<unsigned>& nnz,oomph::Vector<double*>& residuals,bool compressed_row_flag)
  	{    
		// Periodic orbits would have very huge elemental Jacobians, so we must assemble them with block jacobians
		
    	const unsigned long n_elements = mesh_pt()->nelement();
    	unsigned long el_lo = 0;
    	unsigned long el_hi = n_elements - 1;

#ifdef OOMPH_HAS_MPI    
		if (!Problem_has_been_distributed)
		{
		el_lo = First_el_for_assembly[Communicator_pt->my_rank()];
		el_hi = Last_el_plus_one_for_assembly[Communicator_pt->my_rank()] - 1;
		}
#endif

		unsigned ndof = this->ndof();
		const unsigned n_vector = residuals.size();    
		const unsigned n_matrix = column_or_row_index.size();    
		std::cout << "Sparse assembly for periodic orbit:"  << n_vector << "  " << n_matrix << std::endl;
		if (n_vector != 1 || n_matrix != 1)
		{
			throw_runtime_error("Periodic orbit assembly only supports one vector and one matrix");
		}
		//oomph::AssemblyHandler* const assembly_handler_pt = this->assembly_handler_pt();
		PeriodicOrbitHandler* const assembly_handler_pt = dynamic_cast<PeriodicOrbitHandler*>(this->assembly_handler_pt());
		if (!assembly_handler_pt)
		{
			throw_runtime_error("Periodic orbit assembly only supports PeriodicOrbitHandler");
		}

#ifdef OOMPH_HAS_MPI
    	bool doing_residuals = false;
		if (dynamic_cast<oomph::ParallelResidualsHandler*>(this->assembly_handler_pt()) != 0)
		{
			doing_residuals = true;
		}
#endif

#ifdef PARANOID
		if (row_or_column_start.size() != n_matrix)
		{
		std::ostringstream error_stream;
		error_stream << "Error: " << std::endl
					<< "row_or_column_start.size() "
					<< row_or_column_start.size() << " does not equal "
					<< "column_or_row_index.size() "
					<< column_or_row_index.size() << std::endl;
		throw oomph::OomphLibError(
			error_stream.str(), OOMPH_CURRENT_FUNCTION, OOMPH_EXCEPTION_LOCATION);
		}

		if (value.size() != n_matrix)
		{
		std::ostringstream error_stream;
		error_stream
			<< "Error in Problem::sparse_assemble_row_or_column_compressed "
			<< std::endl
			<< "value.size() " << value.size() << " does not equal "
			<< "column_or_row_index.size() " << column_or_row_index.size()
			<< std::endl
			<< std::endl
			<< std::endl;
		throw oomph::OomphLibError(
			error_stream.str(), OOMPH_CURRENT_FUNCTION, OOMPH_EXCEPTION_LOCATION);
		}
#endif

		//oomph::Vector<oomph::Vector<std::map<unsigned, double>>> matrix_data_map(n_matrix);
		/*for (unsigned m = 0; m < n_matrix; m++)
		{
			matrix_data_map[m].resize(ndof);
		}*/
		oomph::Vector<std::map<unsigned, double>> matrix_data_map(ndof);		

		for (unsigned v = 0; v < n_vector; v++)
		{
			residuals[v] = new double[ndof];
			for (unsigned i = 0; i < ndof; i++)
			{
				residuals[v][i] = 0;
			}
		}


#ifdef OOMPH_HAS_MPI
    	double t_assemble_start = 0.0;
		if ((!doing_residuals) && Must_recompute_load_balance_for_assembly)
		{
		Elemental_assembly_time.resize(n_elements);
		}
#endif


    	{


      		//oomph::Vector<oomph::Vector<double>> el_residuals(n_vector);
      		//oomph::Vector<oomph::DenseMatrix<double>> el_jacobian(n_matrix);
			oomph::Vector<double> el_residuals;
	#ifdef PYOOMPH_PERIODIC_ORBIT_BAND_MATRIX
			//PeriodicOrbitAssemblyBlockDenseMatrix el_jacobian(assembly_handler_pt->n_tsteps());
			PeriodicOrbitAssemblyBlockBandMatrix el_jacobian(assembly_handler_pt->n_tsteps(),3); // TODO: Bandwidth
	#else
			oomph::DenseMatrix<double> el_jacobian;
    #endif

      		for (unsigned long e = el_lo; e <= el_hi; e++)
      		{
#ifdef OOMPH_HAS_MPI
				if ((!doing_residuals) && Must_recompute_load_balance_for_assembly)
				{
					t_assemble_start = oomph::TimingHelpers::timer();
				}
#endif
        		oomph::GeneralisedElement* elem_pt = mesh_pt()->element_pt(e);

#ifdef OOMPH_HAS_MPI
        		if (!elem_pt->is_halo())
        		{
#endif
          			const unsigned nvar = assembly_handler_pt->ndof(elem_pt);
					/*for (unsigned v = 0; v < n_vector; v++)
					{
						el_residuals[v].resize(nvar);
					}
					for (unsigned m = 0; m < n_matrix; m++)
					{
						el_jacobian[m].resize(nvar);
					}*/
					el_residuals.resize(nvar);
					el_jacobian.resize(nvar);

          
					//assembly_handler_pt->get_all_vectors_and_matrices(elem_pt, el_residuals, el_jacobian);
					assembly_handler_pt->get_jacobian(elem_pt, el_residuals, el_jacobian);

#ifdef PYOOMPH_PERIODIC_ORBIT_BAND_MATRIX
					
						//throw_runtime_error("TODO: Fill it in")		
					
#else
					
					for (unsigned i = 0; i < nvar; i++)
					{
						unsigned eqn_number = assembly_handler_pt->eqn_number(elem_pt, i);
						residuals[0][eqn_number] += el_residuals[i];
						for (unsigned j = 0; j < nvar; j++)
						{
							double value = el_jacobian(i, j);
							if (std::fabs(value) > Numerical_zero_for_sparse_assembly)
							{
								unsigned unknown = assembly_handler_pt->eqn_number(elem_pt, j);	
								if (compressed_row_flag)
								{
									matrix_data_map[eqn_number][unknown] += value;
								}							
								else
								{	
									matrix_data_map[unknown][eqn_number] += value;
								}
							}
						}
					}
#endif

#ifdef OOMPH_HAS_MPI
        		} // endif halo element
#endif


#ifdef OOMPH_HAS_MPI        
				if ((!doing_residuals) && Must_recompute_load_balance_for_assembly)
				{
					Elemental_assembly_time[e] =oomph::TimingHelpers::timer() - t_assemble_start;
				}
#endif
      		} // End of loop over the elements
    	} // End of map assembly


#ifdef OOMPH_HAS_MPI
    	if ((!doing_residuals) && (!Problem_has_been_distributed) && Must_recompute_load_balance_for_assembly)
    	{
      		recompute_load_balanced_assembly();
    	}

    
    	if ((!doing_residuals) && Must_recompute_load_balance_for_assembly)
    	{
      		Must_recompute_load_balance_for_assembly = false;
    	}
#endif


    
    	//for (unsigned m = 0; m < n_matrix; m++)
    	{
			const unsigned m=0;
      
			row_or_column_start[m] = new int[ndof + 1];      
			unsigned long entry_count = 0;
			row_or_column_start[m][0] = entry_count;

			
			nnz[m] = 0;
			for (unsigned long i_global = 0; i_global < ndof; i_global++)
			{
				//nnz[m] += matrix_data_map[m][i_global].size();
				nnz[m] += matrix_data_map[i_global].size();
			}
      
			column_or_row_index[m] = new int[nnz[m]];
			value[m] = new double[nnz[m]];


			for (unsigned long i_global = 0; i_global < ndof; i_global++)
			{
				row_or_column_start[m][i_global] = entry_count;
				//if (matrix_data_map[m][i_global].empty())
				if (matrix_data_map[i_global].empty())
				{
					continue;
				}
				//for (std::map<unsigned, double>::iterator it =matrix_data_map[m][i_global].begin();it != matrix_data_map[m][i_global].end();++it)
				for (std::map<unsigned, double>::iterator it =matrix_data_map[i_global].begin();it != matrix_data_map[i_global].end();++it)
				{
					column_or_row_index[m][entry_count] = it->first;
					value[m][entry_count] = it->second;				
					entry_count++;
				}
			}
      		row_or_column_start[m][ndof] = entry_count;
    	}

		if (Pause_at_end_of_sparse_assembly)
		{
			oomph::oomph_info << "Pausing at end of sparse assembly." << std::endl;
			oomph::pause("Check memory usage now.");
		}
  	}

    void Problem::sparse_assemble_row_or_column_compressed(oomph::Vector<int*>& column_or_row_index,oomph::Vector<int*>& row_or_column_start,oomph::Vector<double*>& value,oomph::Vector<unsigned>& nnz,oomph::Vector<double*>& residual,bool compressed_row_flag)
	{
		if (dynamic_cast<PeriodicOrbitHandler*>(this->assembly_handler_pt()))
		{
			sparse_assemble_row_or_column_compressed_for_periodic_orbit(column_or_row_index,row_or_column_start,value,nnz,residual,compressed_row_flag);
		}
		else
		{
			oomph::Problem::sparse_assemble_row_or_column_compressed(column_or_row_index,row_or_column_start,value,nnz,residual,compressed_row_flag);
		}
		
	}



 	void Problem::sparse_assemble_row_or_column_compressed_base_problem(oomph::Vector<int*>& column_or_row_index,oomph::Vector<int*>& row_or_column_start,oomph::Vector<double*>& value,oomph::Vector<unsigned>& nnz,oomph::Vector<double*>& residuals,bool compressed_row_flag)
  	{    				
    	const unsigned long n_elements = mesh_pt()->nelement();
    	unsigned long el_lo = 0;
    	unsigned long el_hi = n_elements - 1;

#ifdef OOMPH_HAS_MPI    
		if (!Problem_has_been_distributed)
		{
		if (Communicator_pt->nproc() > 1) throw_runtime_error("This likely does not work in parallel");
		el_lo = First_el_for_assembly[Communicator_pt->my_rank()];
		el_hi = Last_el_plus_one_for_assembly[Communicator_pt->my_rank()] - 1;
		} else throw_runtime_error("This likely does not work in distributed parallel");
#endif

		unsigned ndof = this->get_n_unaugmented_dofs();
		if (this->get_n_unaugmented_dofs()==0) throw_runtime_error("This only works if you have augmented dofs");
		const unsigned n_vector = residuals.size();    
		const unsigned n_matrix = column_or_row_index.size();    		
		oomph::AssemblyHandler* const assembly_handler_pt = this->assembly_handler_pt();
				
#ifdef OOMPH_HAS_MPI
    	bool doing_residuals = false;
		if (dynamic_cast<oomph::ParallelResidualsHandler*>(this->assembly_handler_pt()) != 0)
		{
			doing_residuals = true;
		}
#endif

#ifdef PARANOID
		if (row_or_column_start.size() != n_matrix)
		{
		std::ostringstream error_stream;
		error_stream << "Error: " << std::endl
					<< "row_or_column_start.size() "
					<< row_or_column_start.size() << " does not equal "
					<< "column_or_row_index.size() "
					<< column_or_row_index.size() << std::endl;
		throw oomph::OomphLibError(
			error_stream.str(), OOMPH_CURRENT_FUNCTION, OOMPH_EXCEPTION_LOCATION);
		}

		if (value.size() != n_matrix)
		{
		std::ostringstream error_stream;
		error_stream
			<< "Error in Problem::sparse_assemble_row_or_column_compressed "
			<< std::endl
			<< "value.size() " << value.size() << " does not equal "
			<< "column_or_row_index.size() " << column_or_row_index.size()
			<< std::endl
			<< std::endl
			<< std::endl;
		throw oomph::OomphLibError(
			error_stream.str(), OOMPH_CURRENT_FUNCTION, OOMPH_EXCEPTION_LOCATION);
		}
#endif

		oomph::Vector<oomph::Vector<std::map<unsigned, double>>> matrix_data_map(n_matrix);
		for (unsigned m = 0; m < n_matrix; m++)
		{
			matrix_data_map[m].resize(ndof);
		}		

		for (unsigned v = 0; v < n_vector; v++)
		{
			residuals[v] = new double[ndof];
			for (unsigned i = 0; i < ndof; i++)
			{
				residuals[v][i] = 0;
			}
		}


#ifdef OOMPH_HAS_MPI
    	double t_assemble_start = 0.0;
		if ((!doing_residuals) && Must_recompute_load_balance_for_assembly)
		{
		Elemental_assembly_time.resize(n_elements);
		}
#endif


    	{


      		oomph::Vector<oomph::Vector<double>> el_residuals(n_vector);
      		oomph::Vector<oomph::DenseMatrix<double>> el_jacobian(n_matrix);
			//oomph::Vector<double> el_residuals;
	
      		for (unsigned long e = el_lo; e <= el_hi; e++)
      		{
#ifdef OOMPH_HAS_MPI
				if ((!doing_residuals) && Must_recompute_load_balance_for_assembly)
				{
					t_assemble_start = oomph::TimingHelpers::timer();
				}
#endif
        		oomph::GeneralisedElement* elem_pt = mesh_pt()->element_pt(e);

#ifdef OOMPH_HAS_MPI
        		if (!elem_pt->is_halo())
        		{
#endif
          			const unsigned nvar = assembly_handler_pt->ndof(elem_pt);
					for (unsigned v = 0; v < n_vector; v++)
					{
						el_residuals[v].resize(nvar);
					}
					for (unsigned m = 0; m < n_matrix; m++)
					{
						el_jacobian[m].resize(nvar);
					}
					//el_residuals.resize(nvar);
					//el_jacobian.resize(nvar);

          
					assembly_handler_pt->get_all_vectors_and_matrices(elem_pt, el_residuals, el_jacobian);
					//assembly_handler_pt->get_jacobian(elem_pt, el_residuals, el_jacobian);


					
					
					
					for (unsigned i = 0; i < nvar; i++)
					{
						unsigned eqn_number = assembly_handler_pt->eqn_number(elem_pt, i);
						// Add the contribution to the residuals
            			for (unsigned v = 0; v < n_vector; v++)
            			{
							residuals[v][eqn_number] += el_residuals[v][i];
						}
						
						for (unsigned j = 0; j < nvar; j++)
						{
							// Loop over the matrices
              				for (unsigned m = 0; m < n_matrix; m++)
              				{
								double value = el_jacobian[m](i, j);
								if (std::fabs(value) > Numerical_zero_for_sparse_assembly)
								{
									unsigned unknown = assembly_handler_pt->eqn_number(elem_pt, j);	
									if (compressed_row_flag)
									{
										matrix_data_map[m][eqn_number][unknown] += value;
									}							
									else
									{	
										matrix_data_map[m][unknown][eqn_number] += value;
									}
								}
							}
						}
					}

#ifdef OOMPH_HAS_MPI
        		} // endif halo element
#endif


#ifdef OOMPH_HAS_MPI        
				if ((!doing_residuals) && Must_recompute_load_balance_for_assembly)
				{
					Elemental_assembly_time[e] =oomph::TimingHelpers::timer() - t_assemble_start;
				}
#endif
      		} // End of loop over the elements
    	} // End of map assembly


#ifdef OOMPH_HAS_MPI
    	if ((!doing_residuals) && (!Problem_has_been_distributed) && Must_recompute_load_balance_for_assembly)
    	{
      		recompute_load_balanced_assembly();
    	}

    
    	if ((!doing_residuals) && Must_recompute_load_balance_for_assembly)
    	{
      		Must_recompute_load_balance_for_assembly = false;
    	}
#endif


    
    	for (unsigned m = 0; m < n_matrix; m++)
    	{		
      
			row_or_column_start[m] = new int[ndof + 1];      
			unsigned long entry_count = 0;
			row_or_column_start[m][0] = entry_count;

			
			nnz[m] = 0;
			for (unsigned long i_global = 0; i_global < ndof; i_global++)
			{
				nnz[m] += matrix_data_map[m][i_global].size();
				//nnz[m] += matrix_data_map[i_global].size();
			}
      
			column_or_row_index[m] = new int[nnz[m]];
			value[m] = new double[nnz[m]];


			for (unsigned long i_global = 0; i_global < ndof; i_global++)
			{
				row_or_column_start[m][i_global] = entry_count;
				if (matrix_data_map[m][i_global].empty())
				//if (matrix_data_map[i_global].empty())
				{
					continue;
				}
				for (std::map<unsigned, double>::iterator it =matrix_data_map[m][i_global].begin();it != matrix_data_map[m][i_global].end();++it)
				//for (std::map<unsigned, double>::iterator it =matrix_data_map[i_global].begin();it != matrix_data_map[i_global].end();++it)
				{
					column_or_row_index[m][entry_count] = it->first;
					value[m][entry_count] = it->second;				
					entry_count++;
				}
			}
      		row_or_column_start[m][ndof] = entry_count;
    	}

		if (Pause_at_end_of_sparse_assembly)
		{
			oomph::oomph_info << "Pausing at end of sparse assembly." << std::endl;
			oomph::pause("Check memory usage now.");
		}
  	}




	void Problem::add_augmented_dofs(DofAugmentations &aug)
	{
		if (this->n_unaugmented_dofs!=0)
		{
			throw_runtime_error("Cannot add augmented dofs to a problem that already has augmented dofs");
		}
		this->n_unaugmented_dofs=this->ndof();
		unsigned vindex=0,sindex=0,pindex=0;
		for (unsigned int ti=0;ti<aug.types.size();ti++)
		{
			if (aug.types[ti]==0)
			{
				auto &v=aug.augmented_vectors[vindex];
				for (unsigned i=0;i<v.size();i++)
				{
					this->GetDofPtr().push_back(&(v[i]));
				}
				vindex++;
			}
			else if (aug.types[ti]==1)
			{
				this->GetDofPtr().push_back(&(aug.augmented_scalars[sindex]));
				sindex++;
			}
			else if (aug.types[ti]==2)
			{
				this->GetDofPtr().push_back(&this->get_global_parameter(aug.augmented_parameters[sindex])->value());
				pindex++;
			}
		}
		aug.split_offsets.push_back(this->GetDofPtr().size());
		aug.finalized=true;

		this->GetDofDistributionPt()->build(this->communicator_pt(),this->GetDofPtr().size(), false);
	}


	void Problem::assemble_multiassembly(std::vector<std::string> what,std::vector<std::string> contributions,std::vector<std::string> params,std::vector<std::vector<double>> & hessian_vectors,std::vector<unsigned> & hessian_vector_indices,std::vector<std::vector<double>> & data,std::vector<std::vector<int>> &csrdata,unsigned & ndof,std::vector<int> & return_indices)
	{
		if (what.size()!=contributions.size()) throw_runtime_error("Number of what and contributions must match");
		oomph::Vector<int*> column_or_row_index,row_or_column_start;		
		oomph::Vector<double*> value;
		oomph::Vector<unsigned> nnz;
		oomph::Vector<double*> residuals;

		oomph::AssemblyHandler * old_handler=this->assembly_handler_pt();
		pyoomph::CustomMultiAssembleHandler * new_handler=new pyoomph::CustomMultiAssembleHandler(this,what,contributions,params,hessian_vectors,hessian_vector_indices,return_indices);
	    ndof = this->get_n_unaugmented_dofs();
		this->assembly_handler_pt()=new_handler;
		unsigned nvector=new_handler->n_vector();
		unsigned nmatrix=new_handler->n_matrix();
		column_or_row_index.resize(nmatrix);
		row_or_column_start.resize(nmatrix);
		value.resize(nmatrix);
		nnz.resize(nmatrix);
		residuals.resize(nvector);
		this->sparse_assemble_row_or_column_compressed_base_problem(column_or_row_index,row_or_column_start,value,nnz,residuals,true);
		this->assembly_handler_pt()=old_handler;
		data.resize(nvector+nmatrix);
		csrdata.resize(2*nmatrix);
		for (unsigned int i=0;i<nvector;i++) 
		{
			data[i].resize(ndof);
			for (unsigned int j=0;j<ndof;j++) data[i][j]=residuals[i][j];
			delete [] residuals[i];
		}
		for (unsigned int i=0;i<nmatrix;i++) 
		{
			data[nvector+i].resize(nnz[i]);
			
			for (unsigned int j=0;j<nnz[i];j++) data[nvector+i][j]=value[i][j];
			csrdata[2*i].resize(ndof+1);
			for (unsigned int j=0;j<ndof+1;j++) csrdata[2*i][j]=row_or_column_start[i][j];
			csrdata[2*i+1].resize(nnz[i]);
			for (unsigned int j=0;j<nnz[i];j++) csrdata[2*i+1][j]=column_or_row_index[i][j];
			delete [] value[i];
			delete [] row_or_column_start[i];
			delete [] column_or_row_index[i];
		}

		//TODO: Fill the data
	}


	void Problem::assemble_defined_field_list()
	{
		defined_fields.clear();
		residual_names.clear();
		std::set<std::string> field_set;
		std::set<std::string> res_jac_combis;
		std::map<std::string,unsigned> field_name_to_index;
		
		for (auto * dc : bulk_element_codes)
		{
			auto *ft=dc->get_func_table();
			//std::cout << "Processing element code " << dc->get_file_name() << " has fields " << ft->num_defined_fields_on_this_domain << std::endl;
			for (unsigned i=0;i<ft->num_defined_fields_on_this_domain;i++)
			{
				std::string fn=ft->defined_field_names_on_this_domain[i];
				if (field_set.find(fn)==field_set.end())
				{
					field_set.insert(fn);
					field_name_to_index[fn]=defined_fields.size();	
					defined_fields.push_back(fn);	
				}
			}
			for (unsigned i=0;i<ft->num_res_jacs;i++)
			{				
				std::string combi=ft->res_jac_names[i];
				if (res_jac_combis.find(combi)==res_jac_combis.end())
				{
					res_jac_combis.insert(combi);							
					residual_names.push_back(combi);
				}
			}
		}

		residual_contributing_fields.resize(residual_names.size());
		jacobian_contributing_fields.resize(residual_names.size());
		jacobian_contributing_codes.resize(residual_names.size());
		residual_contributing_codes.resize(residual_names.size());
		for (unsigned i=0;i<residual_names.size();i++)
		{
			residual_contributing_fields[i].resize(defined_fields.size(),false);
			jacobian_contributing_fields[i].resize(defined_fields.size(),std::vector<bool>(defined_fields.size(),false));
			jacobian_contributing_codes[i].resize(defined_fields.size(),std::vector<std::set<DynamicBulkElementCode*>>(defined_fields.size(),std::set<DynamicBulkElementCode*>()));
			residual_contributing_codes[i].resize(defined_fields.size(),std::set<DynamicBulkElementCode*>());
			// Go over all bulk codes and check the contributions
			for (auto * dc : bulk_element_codes)
			{
				auto *ft=dc->get_func_table();
				int my_i=-1;
				for (unsigned j=0;j<ft->num_res_jacs;j++) 
				{
					if (ft->res_jac_names[j]==residual_names[i])
					{
						my_i=j;
						break;
					}
				}
				if (my_i==-1) continue; // This code does not contribute to this residual at all
				for (unsigned j=0;j<ft->contribution_entries_size;j++)
				{
					std::string fn=ft->contribution_names[j];
					if (field_name_to_index.find(fn)==field_name_to_index.end())
					{
						throw_runtime_error("Undefined field " + fn + " in contribution entry for residual/jacobian combination " + residual_names[i]);
					}
					unsigned row_index=field_name_to_index[fn];
					residual_contributing_fields[i][row_index]=residual_contributing_fields[i][row_index]| ft->contributes_to_residual[i][j];

					if (ft->contributes_to_residual[i][j])
					{
						residual_contributing_codes[i][row_index].insert(dc);
					}

					for (unsigned k=0;k<ft->contribution_entries_size;k++)
					{
						std::string fn2=ft->contribution_names[k];	
						if (field_name_to_index.find(fn2)==field_name_to_index.end())
						{
							throw_runtime_error("Undefined field " + fn2 + " in contribution entry for residual/jacobian combination " + residual_names[i]);
						}
						unsigned col_index=field_name_to_index[fn2];
						jacobian_contributing_fields[i][row_index][col_index]=jacobian_contributing_fields[i][row_index][col_index] | ft->contributes_to_jacobian[i][j][k];
						if (ft->contributes_to_jacobian[i][j][k])
						{
							jacobian_contributing_codes[i][row_index][col_index].insert(dc);
						}
					}
				}
				
			}
		}
		pin_due_to_empty_jacobian_row.resize(residual_names.size(),std::vector<bool>(defined_fields.size(),false));
		for (unsigned i=0;i<residual_names.size();i++)
		{
			for (unsigned j=0;j<defined_fields.size();j++)
			{
				bool has_contribs=false;
				for (unsigned k=0;k<defined_fields.size();k++)
				{
					if (jacobian_contributing_fields[i][j][k])
					{
						has_contribs=true;
						break;
					}
				}
				if (!has_contribs)
				{
					pin_due_to_empty_jacobian_row[i][j]=true;
				}
			}
		}

		
		// Loop once more to fill all dirichlet_field_index_to_global_field_index
		for (auto * dc : bulk_element_codes)
		{
			auto *ft=dc->get_func_table();
			//std::cout << "Processing element code " << dc->get_file_name() << " has dirichlet fields " << ft->Dirichlet_set_size << std::endl;
			for (unsigned int i=0;i<ft->Dirichlet_set_size;i++)
			{
				std::string dn=ft->Dirichlet_names[i];
				if (dn=="" || dn.find("__EXT_ODE_")==0) continue;
				if (!ft->moving_nodes && (dn=="coordinate_x" || dn=="coordinate_y" || dn=="coordinate_z")) continue;				
				//std::cout << "Looking for a field index for dirichlet field " << dn << std::endl;
				FiniteElementCode *current=dc->get_code();
				bool found=false;
				while (current)
				{
					std::string fullname=current->get_full_domain_name()+"/"+dn;
					if (field_name_to_index.find(fullname)!=field_name_to_index.end())
					{
						unsigned index=field_name_to_index[fullname];
						ft->dirichlet_field_index_to_global_field_index[i]=index;
						//std::cout << "Found field index " << index << " for dirichlet field " << dn << " in code of " << current->get_full_domain_name() << std::endl;
						found=true;
						break;
					}
					current=current->get_bulk_element();
				}
				if (!found)
				{
					throw_runtime_error("Could not find a global field index for dirichlet field " + dn + " in code of " + dc->get_file_name()+ " or any of the parents. This should not happen");
				}
			}
		}
		
		removed_fields_due_to_missing_jacobian_row.resize(defined_fields.size(),false);
		
	}


	bool Problem::is_field_removed_from_dofs_due_to_missing_jacobian_row(int global_field_index)
	{
		if (global_field_index<0) return false;
		else return removed_fields_due_to_missing_jacobian_row[global_field_index];
	}
	std::string Problem::get_jacobian_information_string()
	{
		std::ostringstream ss;
		ss << "Defined fields: " << std::endl;
		for (unsigned int i=0;i<defined_fields.size();i++)
		{
			ss << "\t" << i << "\t" << defined_fields[i] << std::endl;
		}
		ss << std::endl;
		for (unsigned int ri=0;ri<residual_names.size();ri++)
		{
			std::string combi=residual_names[ri];
			if (combi=="") ss << "Jacobian Structure -- Default Residuals" << std::endl;
			else ss << "Jacobian Structure -- Custom Residuals \"" << combi << "\"" << std::endl;
			if (defined_fields.size()>999)
			{
				throw_runtime_error("Too many defined fields to print jacobian structure");
			}
			else if (defined_fields.size()>99)
			{
				ss << "\t    | ";
				for (unsigned int i=0;i<defined_fields.size();i++)				
				{
					if (pin_due_to_empty_jacobian_row[ri][i]) continue;
					else if (i/100) ss << i/100 << " ";
					else ss << "  ";
				}
				ss << "|" << std::endl;
				ss << "\t    | ";
				for (unsigned int i=0;i<defined_fields.size();i++)				
				{
					if (pin_due_to_empty_jacobian_row[ri][i]) continue;
					else if ((i%100)/10) ss << (i%100)/10 << " ";
					else ss << "  ";
				}
				ss <<"|" << std::endl;
				ss << "\t    | ";
				for (unsigned int i=0;i<defined_fields.size();i++)				
				{
					if (pin_due_to_empty_jacobian_row[ri][i]) continue;
					else ss << i%10 << " ";
				}
				ss << "|" << std::endl;
				ss << "\t----|";
				for (unsigned int i=0;i<defined_fields.size();i++)				
				{
					if (pin_due_to_empty_jacobian_row[ri][i]) continue;
					else ss  << "--";
				}
				ss <<"-|" << std::endl;
			}
			else if (defined_fields.size()>9)
			{
				ss << "\t    | ";
				for (unsigned int i=0;i<defined_fields.size();i++)				
				{
					if (pin_due_to_empty_jacobian_row[ri][i]) continue;
					else if (i/10) ss << i/10 << " ";
					else ss << "  ";
				}
				ss << "|" << std::endl;
				ss << "\t    | ";
				for (unsigned int i=0;i<defined_fields.size();i++)				
				{
					if (pin_due_to_empty_jacobian_row[ri][i]) continue;
					else ss << i%10 << " ";
				}
				ss << "|" << std::endl;
				ss << "\t----|";
				for (unsigned int i=0;i<defined_fields.size();i++)				
				{
					if (pin_due_to_empty_jacobian_row[ri][i]) continue;
					else ss  << "--";
				}
				ss << "-|" <<std::endl;
			}
			else
			{
				ss << "\t    | ";
				for (unsigned int i=0;i<defined_fields.size();i++)				
				{
					if (pin_due_to_empty_jacobian_row[ri][i]) continue;
					else ss << i << " ";
				}
				ss <<"|"<< std::endl;
				ss << "\t----|";
				for (unsigned int i=0;i<defined_fields.size();i++)				
				{
					if (pin_due_to_empty_jacobian_row[ri][i]) continue;
					else ss  << "--";
				}
				ss << "-|" << std::endl;
			}
			
			std::vector<std::set<DynamicBulkElementCode*>> listed_domain_contributions;
			for (unsigned int i=0;i<defined_fields.size();i++)
			{				
				if (pin_due_to_empty_jacobian_row[ri][i]) continue;
				else ss << "  ";
				ss<<"\t" << std::setfill(' ') << std::setw(3) << i << " | ";		
				for (unsigned int j=0;j<defined_fields.size();j++)
				{
					if (pin_due_to_empty_jacobian_row[ri][j]) continue;
					if (jacobian_contributing_fields[ri][i][j]) 
					{
						unsigned found=listed_domain_contributions.size();
						for (unsigned int k=0;k<listed_domain_contributions.size();k++)
						{
							if (listed_domain_contributions[k]==jacobian_contributing_codes[ri][i][j])
							{
								found=k;
								break;
							}
						}
						if (found==listed_domain_contributions.size())
						{
							listed_domain_contributions.push_back(jacobian_contributing_codes[ri][i][j]);
						}
						std::string symbol=" ";
						symbol[0]=(char)('A' + found + (found>25 ? 6 : 0));
						
						ss << symbol << " ";
					}
					else ss << "  ";
				}
				ss <<"|" << std::endl;				
			}
			//Residual separator
			ss << "\t----|";
			for (unsigned int i=0;i<defined_fields.size();i++)
			{
				if (pin_due_to_empty_jacobian_row[ri][i]) continue;
				ss << "--";
			}
			ss << "-|" << std::endl;
			ss<< "\tRes | " ;
			std::set<unsigned> residual_contributions_with_zero_jacobian_row;
			for (unsigned int i=0;i<defined_fields.size();i++)
			{
				if (pin_due_to_empty_jacobian_row[ri][i]) 
				{
					if (residual_contributing_fields[ri][i]) residual_contributions_with_zero_jacobian_row.insert(i);
					continue;
				}
				if (residual_contributing_fields[ri][i]) 
					{
						unsigned found=listed_domain_contributions.size();
						for (unsigned int k=0;k<listed_domain_contributions.size();k++)
						{
							if (listed_domain_contributions[k]==residual_contributing_codes[ri][i])
							{
								found=k;
								break;
							}
						}
						if (found==listed_domain_contributions.size())
						{
							listed_domain_contributions.push_back(residual_contributing_codes[ri][i]);
						}
						std::string symbol=" ";
						symbol[0]=(char)('A' + found + (found>25 ? 6 : 0));						
						ss << symbol << " ";
					}
					else ss << "  ";
			}
			ss << "|" << std::endl;
			ss << std::endl;

			for (unsigned int k=0;k<listed_domain_contributions.size();k++)
			{
				ss << "\t" << (char)('A' + k + (k>25 ? 6 : 0)) << ": from:  ";
				unsigned int count=listed_domain_contributions[k].size();
				for (auto * dc : listed_domain_contributions[k])
				{
					ss << dc->get_code()->get_full_domain_name() << (--count ? " & " : "");
				}
				ss << std::endl;
			}
			ss << std::endl;

			if (residual_contributions_with_zero_jacobian_row.size()>0)
			{
				ss << "\t|WARNING|: Following fields have residual contributions, but a zero Jacobian row: ";
				unsigned int count=residual_contributions_with_zero_jacobian_row.size();
				for (auto i : residual_contributions_with_zero_jacobian_row)
				{
					ss << i << (--count ? ", " : "");
				}
				ss << std::endl;
				ss << std::endl;
			}
			
		}
		return ss.str();
		
	}





	void GlobalParameterDescriptor::set_analytic_derivative(bool active)
	{
		if (active)
			problem->set_analytic_dparameter(&Value);
		else
			problem->unset_analytic_dparameter(&Value);
	}
	bool GlobalParameterDescriptor::get_analytic_derivative()
	{
		return problem->is_dparameter_calculated_analytically(&Value);
	}




	DofAugmentations::DofAugmentations(Problem * _problem) : problem(_problem)
	{
		total_length=problem->ndof();
		finalized=false;
		split_offsets.push_back(0);
	}
    
	unsigned DofAugmentations::add_vector(const std::vector<double> & v) 
	{
		if (finalized) throw_runtime_error("Cannot modify the augmented DoFs once they are finalized");
		augmented_vectors.push_back(v); 
		types.push_back(0); 
		unsigned start=total_length; 
		split_offsets.push_back(start);
		total_length+=v.size(); 
		return start;
	}
    
	unsigned DofAugmentations::add_scalar(const double & s) 
	{
		if (finalized) throw_runtime_error("Cannot modify the augmented DoFs once they are finalized");
		augmented_scalars.push_back(s);
		types.push_back(1); 		
		unsigned start=total_length; 
		split_offsets.push_back(start);
		total_length+=1; 
		return start;
	}
    unsigned DofAugmentations::add_parameter(std::string p) 
	{
		if (finalized) throw_runtime_error("Cannot modify the augmented DoFs once they are finalized");
		augmented_parameters.push_back(p); 
		types.push_back(2); 		
		unsigned start=total_length; 
		split_offsets.push_back(start);
		total_length+=1; 
		return start;
	}      

	std::vector<std::vector<double>> DofAugmentations::split(unsigned int startindex,int endindex)
	{
		if (!finalized) throw_runtime_error("Cannot split non-finalized dofs");
		auto  dofptr=this->problem->GetDofPtr();		
		if (dofptr.size()!=split_offsets.back()) throw_runtime_error("Invalid number of dofs. Likely, the dofs has changed meanwhile");
		std::vector<std::vector<double>> res;
		if (endindex<0) endindex=split_offsets.size()+(endindex);		
		if (endindex>=split_offsets.size())  throw_runtime_error("Invalid end index");
		for (unsigned int i=startindex;i<endindex;i++)
		{
			unsigned length=split_offsets[i+1]-split_offsets[i];
			//std::cout << "SPlIT INDEX "<< i << " " << length << " FROM " << split_offsets[i] << " TO " << split_offsets[i+1] <<std::endl;
			res.push_back(std::vector<double>(length));
			for (unsigned int vi=0;vi<length;vi++) 
			{
				//std::cout << "DOFPTR" << " at " << split_offsets[i]+vi << "  " << dofptr[split_offsets[i]+vi] <<std::endl << std::flush;
				res.back()[vi]=*dofptr[split_offsets[i]+vi];
			}
		}
		return res;
	}

}
