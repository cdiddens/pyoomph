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
	// Called from JIT-generated code right after loading a shared library: verifies that the ABI
	// layout sizes baked into the compiled code (jitsize) match what this build of pyoomph expects
	// (internal_size), catching stale/incompatible compiled equation code early instead of crashing.
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

	// Recursively merges the "required shapes" flags of src into dest (bitwise OR of all shape/derivative
	// flags for every continuous space, DL/D0/position spaces, normals, element sizes, etc.), including
	// the nested bulk_shapes/opposite_shapes sub-structures (allocating them in dest on demand). Used to
	// accumulate, over all residual/Jacobian/Hessian/integral/etc. contributions of a code, the union of
	// shape information that must be computed for a given element.
	void RequiredShapes_merge(JITFuncSpec_RequiredShapes_FiniteElement_t *src, JITFuncSpec_RequiredShapes_FiniteElement_t *dest)
	{
		for (unsigned int i = 0; i < NUM_CONTINUOUS_SPACES; i++)
		{
			dest->continuous_spaces[i].psi |= src->continuous_spaces[i].psi;
			dest->continuous_spaces[i].dx_psi |= src->continuous_spaces[i].dx_psi;
			dest->continuous_spaces[i].dX_psi |= src->continuous_spaces[i].dX_psi;
		}		
		dest->DL.psi |= src->DL.psi;
		dest->D0.psi |= src->D0.psi;
	
		dest->DL.dx_psi |= src->DL.dx_psi;
		dest->D0.dx_psi |= src->D0.dx_psi;
		dest->DL.dX_psi |= src->DL.dX_psi;
		dest->D0.dX_psi |= src->D0.dX_psi;
		dest->Pos.psi |= src->Pos.psi;
		dest->Pos.dx_psi |= src->Pos.dx_psi;
		dest->Pos.dX_psi |= src->Pos.dX_psi;
		

		dest->normal |= src->normal;
		dest->elemsize_Eulerian |= src->elemsize_Eulerian;
		dest->elemsize_Lagrangian |= src->elemsize_Lagrangian;
		dest->elemsize_Eulerian_cartesian |= src->elemsize_Eulerian_cartesian;
		dest->elemsize_Lagrangian_cartesian |= src->elemsize_Lagrangian_cartesian;		

		dest->history_integral_dx1 |= src->history_integral_dx1;
		dest->history_integral_dx2 |= src->history_integral_dx2;

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

	// Recursively frees a JITFuncSpec_RequiredShapes_FiniteElement_t allocated via calloc (including the
	// nested bulk_shapes/opposite_shapes sub-structures created on demand by RequiredShapes_merge).
	void RequiredShapes_free(JITFuncSpec_RequiredShapes_FiniteElement_t *p)
	{
		if (p->bulk_shapes)
			RequiredShapes_free(p->bulk_shapes);
		if (p->opposite_shapes)
			RequiredShapes_free(p->opposite_shapes);

		std::free(p);
	}

	// Loads and initializes the function table exported by a freshly dlopen'ed JIT-compiled equation
	// shared library: takes ownership of the handle from the compiler, populates the function table via
	// the library's init entry point, then post-processes it (assigns space indices, collects only the
	// continuous/DG spaces that are actually present, determines the dominant space used for the element's
	// nodal positions, and merges the "required shapes" flags of all residual/Jacobian/Hessian/integral/
	// extremum/Z2-flux/tracer-advection contributions into a single merged_required_shapes).
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

		functable->info_Pos.space_index=0;


		for (unsigned int i=0;i<NUM_CONTINUOUS_SPACES;i++)
		{
			functable->continuous_spaces[i].space_index=i;
			functable->dg_spaces[i].space_index=i;
		}

		functable->total_num_fields=0;
		functable->total_num_fields_basebulk=0;

	// Only add the spaces which are really present to the present continuous_spaces array, in order of dominance
		functable->num_present_continuous_spaces=0;
		for (unsigned int i=0;i<NUM_CONTINUOUS_SPACES;i++)
		{
			if (functable->continuous_spaces[i].numfields>0)
			{
				functable->present_continuous_spaces[functable->num_present_continuous_spaces]=&functable->continuous_spaces[i];
				functable->total_num_fields+=functable->continuous_spaces[i].numfields;
				functable->total_num_fields_basebulk+=functable->continuous_spaces[i].numfields_basebulk;
				functable->num_present_continuous_spaces++;
			}
			if (functable->dg_spaces[i].numfields>0)
			{
				functable->present_dg_spaces[functable->num_present_dg_spaces]=&functable->dg_spaces[i];
				functable->total_num_fields+=functable->dg_spaces[i].numfields;
				functable->num_present_dg_spaces++;
			}
		}
		functable->total_num_fields+=functable->info_D0.numfields+functable->info_DL.numfields;
		// Find the continuous space matching the code's declared "dominant" space (the one whose nodes
		// carry the element's Eulerian/Lagrangian position) and remember its index for info_Pos.
		std::string dominant_space=functable->dominant_space;
		bool found_dominant=false;
		for (unsigned int i=0;i<NUM_CONTINUOUS_SPACES;i++)
		{
			if (std::string(functable->continuous_spaces[i].space_name)==dominant_space)
			{
				found_dominant=true;
				functable->info_Pos.space_index=i;
				break;
			}
		}

		if(!found_dominant)
		{
			std::ostringstream errmsg;
			errmsg << "Unknown dominant space " << dominant_space << " in JIT code " << filename;
			throw_runtime_error(errmsg.str());
		}



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

	// Cleans up the function table (invoking the code's own clean_up callback first, if any) and closes
	// the dlopen handle of the compiled shared library.
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

	// Looks up the residual/Jacobian contribution named "name" in this code and, if found and non-NULL,
	// makes it the active one (functable->current_res_jac) for subsequent assembly calls into this code.
	// Returns 1 on success, 0 if no matching (non-NULL) contribution exists.
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

	// Rebuilds the deduplicated elemental_data list from the (name-indexed) link entries: for each link,
	// finds or inserts its Data pointer in elemental_data and records the resulting position as the
	// link's elemental_index (the index the compiled element code uses to address this external data).
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

	// Constructs a new instance of code d bound to mesh bm; allocates the (initially empty) external-data
	// link slots sized to the code's declared ED0 (external-data-field) count.
	DynamicBulkElementInstance::DynamicBulkElementInstance(DynamicBulkElementCode *d, pyoomph::Mesh *bm) : dyn(d), // local_field_to_global_field_index_C1(d->functable->numfields_C1,-1),
																												   //		local_field_to_global_field_index_C2(d->functable->numfields_C2,-1),
																												   //		local_global_parameter_to_global_index(d->functable->numglobal_params,-1),
																										   linked_external_data(d->functable->info_ED0.numfields),
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

	// Binds the external field named "name" (as declared by the compiled code's ED0 space) to a specific
	// (Data, index) pair, and patches the code's contribution name for that field (used in diagnostic
	// output such as get_jacobian_information_string()) to full_source_name so it reflects where the
	// linked value actually comes from. Throws if "name" is not among the code's required external fields.
	void DynamicBulkElementInstance::link_external_data(std::string name, oomph::Data *data, int index,std::string full_source_name)
	{
		int found = -1;
		for (unsigned int i = 0; i < dyn->functable->info_ED0.numfields; i++)
		{
			if (name == std::string(dyn->functable->info_ED0.fieldnames[i]))
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

	// Maps every nodal field name to its index into the node's value array, in the order in which the
	// element code lays out nodal fields: first the "base bulk" fields of all present continuous spaces,
	// then the base-bulk fields of all present DG spaces, then the remaining (non-base-bulk, i.e.
	// interface-only) fields of the continuous spaces, then the remaining fields of the DG spaces.
	std::map<std::string, unsigned> DynamicBulkElementInstance::get_nodal_field_indices()
	{
		std::map<std::string, unsigned> res;
		unsigned offs = 0;
		for (unsigned int si=0;si<dyn->functable->num_present_continuous_spaces;si++)
		{
			auto *space = dyn->functable->present_continuous_spaces[si];
			for (unsigned int i = 0; i < space->numfields_basebulk; i++)
			{
				res[space->fieldnames[i]] = offs + i;
			}
			offs += space->numfields_basebulk;
		}		


		for (unsigned int si=0;si<dyn->functable->num_present_dg_spaces;si++)
		{
			auto *space = dyn->functable->present_dg_spaces[si];
			for (unsigned int i = 0; i < space->numfields_basebulk; i++)
			{
				res[space->fieldnames[i]] = offs + i;
			}
			offs += space->numfields_basebulk;
		}


		// Now the additional ones
		for (unsigned int si=0;si<dyn->functable->num_present_continuous_spaces;si++)
		{
			auto *space = dyn->functable->present_continuous_spaces[si];
			for (unsigned int i = 0; i < space->numfields - space->numfields_basebulk; i++)
			{
				res[space->fieldnames[i + space->numfields_basebulk]] = offs + i;
			}
			offs += space->numfields - space->numfields_basebulk;
		}

		for (unsigned int si=0;si<dyn->functable->num_present_dg_spaces;si++)
		{
			auto *space = dyn->functable->present_dg_spaces[si];
			for (unsigned int i = 0; i < space->numfields - space->numfields_basebulk; i++)
			{
				res[space->fieldnames[i + space->numfields_basebulk]] = offs + i;
			}
			offs += space->numfields - space->numfields_basebulk;
		}
		

		return res;
	}

	// Maps every elemental (internal, non-nodal) field name to its index among the element's internal
	// Data values: DL (discontinuous Lagrange, one value per element node) fields first, then D0
	// (piecewise-constant) fields, offset after the DL ones.
	std::map<std::string, unsigned> DynamicBulkElementInstance::get_elemental_field_indices()
	{
		std::map<std::string, unsigned> res;
		for (unsigned int i = 0; i < dyn->functable->info_DL.numfields; i++)
		{
			res[dyn->functable->info_DL.fieldnames[i]] = i;
		}
		for (unsigned int i = 0; i < dyn->functable->info_D0.numfields; i++)
		{
			res[dyn->functable->info_D0.fieldnames[i]] = i + dyn->functable->info_DL.numfields;
		}
		return res;
	}

	// Index of a DL or D0 (discontinuous/elemental) field by name, offset by internal_offset_new so it
	// directly indexes into the element's internal Data array; -1 if not a discontinuous field.
	int DynamicBulkElementInstance::get_discontinuous_field_index(std::string name)
	{
		for (unsigned int i = 0; i < dyn->functable->info_DL.numfields; i++)
		{
			if (!strcmp(name.c_str(), dyn->functable->info_DL.fieldnames[i]))
			{
				return i + dyn->functable->info_DL.internal_offset_new;
			}
		}
		for (unsigned int i = 0; i < dyn->functable->info_D0.numfields; i++)
		{
			if (!strcmp(name.c_str(), dyn->functable->info_D0.fieldnames[i]))
			{
				return i + dyn->functable->info_D0.internal_offset_new;
			}
		}
		return -1;
	}

	// Index of a base-bulk nodal field by name (offset by the space's nodal_offset_basebulk into the
	// node's value array); -1 if "name" is not a base-bulk field of any present continuous space.
	int DynamicBulkElementInstance::get_nodal_field_index(std::string name)
	{
		for (unsigned int si=0;si<dyn->functable->num_present_continuous_spaces;si++)
		{
			auto *space = dyn->functable->present_continuous_spaces[si];
			for (unsigned int i = 0; i < space->numfields_basebulk; i++)
			{
				if (!strcmp(name.c_str(), space->fieldnames[i]))
				{
					return i + space->nodal_offset_basebulk;
				}
			}
		}
		return -1;
	}

	// Forwards to the bulk mesh's interface-dof-id registry, adding a new id for field n if not present yet
	unsigned DynamicBulkElementInstance::resolve_interface_dof_id(std::string n)
	{
		//std::cout << "->Resolving interface dof for field " << n << " on mesh " <<  this->get_bulk_mesh() << std::endl;
		return this->get_bulk_mesh()->resolve_interface_dof_id(n);
	}

	// For every present continuous space's interface-only fields (the fields beyond numfields_basebulk,
	// i.e. those that only live on interface nodes of a C2TB-C1-type space), resolves and stores their
	// interface dof id in the function table's interface_dof_indices array so that interface elements can
	// find the additional dof slot on shared nodes. Required whenever this instance participates in an
	// interface coupling. Returns the resolved field-name -> dof-id map for convenience.
	std::map<std::string, unsigned> DynamicBulkElementInstance::setup_interface_dof_indices()
	{
		std::map<std::string, unsigned> res;		

		for (unsigned int i = 0; i < dyn->functable->num_present_continuous_spaces; i++	)
		{
			JITFuncSpec_Table_FiniteElement_SpaceInfo_t * space_info= dyn->functable->present_continuous_spaces[i];

			if (space_info->interface_dof_indices) {
				free(space_info->interface_dof_indices);
				space_info->interface_dof_indices=NULL;
			}
			if (space_info->numfields-space_info->numfields_basebulk>0)
			{
				space_info->interface_dof_indices=(unsigned int*)std::calloc(space_info->numfields-space_info->numfields_basebulk,sizeof(unsigned int));
				for (unsigned int i=0;i<space_info->numfields-space_info->numfields_basebulk;i++)
				{
					std::string field_name=space_info->fieldnames[i+space_info->numfields_basebulk];
					unsigned dof_index=this->resolve_interface_dof_id(field_name);					
					space_info->interface_dof_indices[i]=dof_index;
					res[field_name]=dof_index;
				}
			}			
		}

		return res;

	}

	// Returns the name of the function space ("C2","C1",...,"DL","D0") a field belongs to, or "" if unknown
	std::string DynamicBulkElementInstance::get_space_of_field(std::string name)
	{
		for (unsigned int si=0;si<dyn->functable->num_present_continuous_spaces;si++)
		{
			auto *space = dyn->functable->present_continuous_spaces[si];
			for (unsigned int i = 0; i < space->numfields_basebulk; i++)
			{
				if (!strcmp(name.c_str(), space->fieldnames[i]))
				{
					return space->space_name;
				}
			}
		}

		for (unsigned int si=0;si<dyn->functable->num_present_dg_spaces;si++)
		{
			auto *space = dyn->functable->present_dg_spaces[si];
			for (unsigned int i = 0; i < space->numfields_basebulk; i++)
			{
				if (!strcmp(name.c_str(), space->fieldnames[i]))
				{
					return space->space_name;
				}
			}
		}
		
		for (unsigned int i = 0; i < dyn->functable->info_DL.numfields; i++)
		{
			if (!strcmp(name.c_str(), dyn->functable->info_DL.fieldnames[i]))
			{
				return "DL";
			}
		}
		for (unsigned int i = 0; i < dyn->functable->info_D0.numfields; i++)
		{
			if (!strcmp(name.c_str(), dyn->functable->info_D0.fieldnames[i]))
			{
				return "D0";
			}
		}
		return "";
	}

	// Placeholder for consistency checks of the instance's field/parameter binding; currently a no-op,
	// the binding checks it used to perform are now handled elsewhere (kept here, commented out, for reference)
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

    // Whether this instance's compiled code has any residual/Jacobian contribution that depends on the
    // named global parameter (i.e. the parameter is among the code's registered global params).
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

///////////////////////////////////

	void DirichletMatrixManipulationInfo::clear()
	{
		data_to_dirichlet_dof_indices.clear();
		//throw_runtime_error("DirichletMatrixManipulationInfo::clear() is not implemented yet. This should be implemented if you want to use the Dirichlet matrix manipulation feature");
	}

	// Registers (d, dof_index) as Dirichlet-constrained and immediately unpins it, since under the
	// matrix-manipulation strategy Dirichlet dofs are kept as real (unpinned) dofs in the dof vector and
	// the constraint is instead enforced afterwards by zeroing rows/columns in the Jacobian/residual.
	void DirichletMatrixManipulationInfo::add_dirichlet_dof(oomph::Data *d, unsigned dof_index)
	{
		data_to_dirichlet_dof_indices[d].insert(dof_index);
		d->unpin(dof_index);
	}

	// (Re)computes global_pinned_dof_set, the set of global equation numbers corresponding to all
	// registered Dirichlet (Data, dof_index) pairs, using the problem's current equation numbering.
	// Must be called after equation numbers are (re)assigned. In a distributed run, each process only
	// knows the equation numbers of its own Data, so the sets are all-gathered and merged across ranks.
	void DirichletMatrixManipulationInfo::build_global_pinned_equation_set(Problem *prob)
	{
		global_pinned_dof_set.clear();
		//std::cout << "REMOVE DIRICHLET ENTRIES: " << data_to_dirichlet_dof_indices.size() << std::endl;
		for (const auto &pair : data_to_dirichlet_dof_indices)
		{
			for (unsigned dof_index : pair.second)
			{
				unsigned long global_eqn=pair.first->eqn_number(dof_index);
				//std::cout << "Mapping equation " << global_eqn << " to value pointer for data " << pair.first << " dof index " << dof_index << std::endl;
				//eqn_number_to_value_ptr[global_eqn] = pair.first->value_pt(dof_index);
				global_pinned_dof_set.insert(global_eqn);
			}
		}

		// If distributed, we have to merge the map from all processes and make sure that the global equation numbers are consistent across processes.
#ifdef OOMPH_HAS_MPI
		if (prob->distributed())
		{
			
			int size=prob->communicator_pt()->nproc();			
			std::vector<unsigned long> local_vec(global_pinned_dof_set.begin(), global_pinned_dof_set.end());
			int local_count = static_cast<int>(local_vec.size());
			std::vector<int> recvcounts(size);
			MPI_Allgather(&local_count, 1, MPI_INT,recvcounts.data(), 1, MPI_INT,prob->communicator_pt()->mpi_comm());

			std::vector<int> displs(size, 0);
			int total_count = recvcounts[0];
			for (int i = 1; i < size; ++i)
			{
				displs[i] = displs[i - 1] + recvcounts[i - 1];
				total_count += recvcounts[i];
			}
			
			std::vector<unsigned long> global_vec(total_count);

			MPI_Allgatherv(local_vec.data(),local_count,MPI_UNSIGNED_LONG,global_vec.data(),recvcounts.data(),displs.data(),MPI_UNSIGNED_LONG,prob->communicator_pt()->mpi_comm());

			global_pinned_dof_set=std::unordered_set<unsigned long>(global_vec.begin(),global_vec.end());

		}
#endif
	}



///////////////////////////////////



    // Stores a user-supplied Jacobian in CSR form (values/column indices/row starts), copied into the
    // oomph::Vector members used by the rest of the custom-residual/Jacobian assembly path.
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

	// Maximum time-derivative order required by any currently loaded equation code
	unsigned Problem::get_max_dt_order() const
	{
		unsigned max_order = 0;
		for (unsigned int i = 0; i < bulk_element_codes.size(); i++)
		{
			max_order = std::max(max_order, (unsigned)bulk_element_codes[i]->functable->max_dt_order);
		}
		return max_order;
	}

	// Deletes all loaded DynamicBulkElementCode objects (closing their shared libraries) and all global
	// parameter descriptors, and releases the cached eigenproblem matrices. Used both from the destructor
	// and explicitly (e.g. before recompiling/reloading equation code).
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

	// Unloads all equation code and closes the log file (if any and if it is still the active logging stream)
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

	// Loads (or returns the already-loaded) DynamicBulkElementCode for the shared library dynamic_lib,
	// associates it with element_class (which fills in the callback function pointers of the function
	// table), and binds each of the code's declared global parameters to the corresponding
	// GlobalParameterDescriptor's value in this problem (creating parameters as needed).
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

	// Activates the residual/Jacobian contribution named "name" in every loaded code (multi-residual
	// problems can define several independently solvable residuals). If remove_dofs_without_jacobian_row
	// is set, also updates removed_fields_due_to_missing_jacobian_row_or_col from the precomputed
	// pin_due_to_empty_jacobian_row_or_col table for this residual, so fields with no Jacobian row/column
	// under the newly active residual get pinned. Returns whether the residual was found in at least one
	// code; if not and raise_error is set, throws.
	bool Problem::_set_solved_residual(std::string name,bool raise_error,bool remove_dofs_without_jacobian_row)
	{
		unsigned numfound = 0;
		if (this->_solved_residual != name) 
		{
			for (unsigned int i=0;i<removed_fields_due_to_missing_jacobian_row_or_col.size();i++) removed_fields_due_to_missing_jacobian_row_or_col[i]=false;
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
		std::set<unsigned> fields_with_missing_jacobian_row_or_col;
		if (remove_dofs_without_jacobian_row)
		{
			for (unsigned int i=0;i<removed_fields_due_to_missing_jacobian_row_or_col.size();i++) 
			{
				removed_fields_due_to_missing_jacobian_row_or_col[i]=pin_due_to_empty_jacobian_row_or_col[resind][i];
				if (removed_fields_due_to_missing_jacobian_row_or_col[i]) fields_with_missing_jacobian_row_or_col.insert(i);
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
		for (unsigned int i=0;i<removed_fields_due_to_missing_jacobian_row_or_col.size();i++) if (removed_fields_due_to_missing_jacobian_row_or_col[i]) return true;
		return false;
	}


	double &Problem::global_parameter(const std::string &n)
	{
		GlobalParameterDescriptor *res = assert_global_parameter(n);
		return res->value();
	}

	// Returns the descriptor for global parameter "name", creating a new one (value 0, analytic
	// derivative enabled by default) if it does not exist yet.
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

	// Combines the per-submesh temporal error norm contributions (sum of squares) into a single global
	// RMS-like error estimate used by adaptive timestepping to decide whether/how to change dt.
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

	// For every dof, the index into get_global_field_names() of the field it belongs to (-1 if it cannot
	// be attributed to a single global field, e.g. augmentation dofs); delegates to each submesh.
	std::vector<int> Problem::get_dof_to_global_field_index_mapping()
	{
		std::vector<int> res(this->ndof(), -1);
		for (unsigned int ism = 0; ism < this->nsub_mesh(); ism++)
		{
			pyoomph::Mesh *m = dynamic_cast<pyoomph::Mesh *>(this->mesh_pt(ism));
			m->fill_dof_to_global_field_index_buffer(res);
		}
		return res;
	}

	std::vector<std::string> Problem::get_global_field_names()
	{
		return defined_fields;
	}

	// Two-pass reset of "dummy" values (unused position/field dofs kept only to satisfy oomph-lib's
	// element interface, e.g. unused position dofs of non-moving-mesh problems): first unpin all of them
	// (across every element on every submesh) so any stale pinning state is cleared, then re-pin them all
	// so they are guaranteed to actually be dummy (not real dofs) afterwards. Two passes are needed because
	// dummy-ness is a per-Data-object property that may be shared between elements.
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


	// Clears and rebuilds the Dirichlet bookkeeping used for the matrix-manipulation strategy: unpins the
	// dofs of all Dirichlet conditions on every element (registering them in dirichlet_info instead so
	// they remain assembled and are handled by remove_dirichlets_by_matrix_manipulation() later).
	void Problem::unpin_Dirichlet_dofs_for_matrix_manipulation()
	{
		dirichlet_info.clear();
		for (unsigned nmi = 0; nmi < this->nsub_mesh(); nmi++)
		{
			unsigned nelem = mesh_pt(nmi)->nelement();
			//		std::cout << "ENSURE PINNING NEL " << nelem << std::endl;
			for (unsigned n = 0; n < nelem; n++)
			{
				auto el = dynamic_cast<BulkElementBase *>(mesh_pt(nmi)->element_pt(n));
				if (el) el->unpin_Dirichlet_dofs_for_matrix_manipulation(dirichlet_info);
			}
		}
		/////
		for (unsigned int f=0; f<defined_fields.size();f++)
		{
			if (is_field_removed_from_dofs_due_to_missing_jacobian_row(f))
			{
				throw_runtime_error("This must be implemented. The dofs of removed fields should not be unpinned. Likely has to be done on an elemental level");
			}
		}
		/////		
	}

	// Performs the standard oomph-lib equation numbering, then lets every InterfaceMesh update its
	// equation remapping (interface dofs referencing bulk equation numbers that may have just changed),
	// and finally, when Dirichlet conditions are enforced by matrix manipulation rather than dof removal,
	// rebuilds the set of globally pinned equation numbers (which depends on the numbering just assigned).
	unsigned long Problem::assign_eqn_numbers(const bool& assign_local_eqn_numbers)
	{
      unsigned long res=oomph::Problem::assign_eqn_numbers(assign_local_eqn_numbers);
	  for (unsigned nmi = 0; nmi < this->nsub_mesh(); nmi++)
	  {
		if (dynamic_cast<InterfaceMesh *>(this->mesh_pt(nmi)))
		{
			dynamic_cast<InterfaceMesh *>(this->mesh_pt(nmi))->update_equation_remapping();
		}
	  }
	  if (!this->dirichlets_by_removing_from_dof_vector)
	  {
		//dirichlet_info.build_equation_to_value_map();
		dirichlet_info.build_global_pinned_equation_set(this);
	  }
	  return res;
	}

	// After mesh adaptation: invalidates any cached Lagrangian KD-trees (node positions may have changed),
	// re-establishes the dummy-value pinning invariant, and re-runs problem-specific pinning setup.
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

	// Default implementation just forwards to oomph-lib; Python subclasses typically override this
	// (via the trampoline) to set up problem-specific initial conditions.
	void Problem::set_initial_condition()
	{
		oomph::Problem::set_initial_condition();
	}

	// Assembles (or reuses/reallocates the cached eigen_MassMatrixPt/eigen_JacobianMatrixPt) the mass
	// matrix M and Jacobian J for the generalized eigenproblem J*v = sigma*M*v around the shift sigma_r.
	// Note: Dirichlet-by-matrix-manipulation is not yet supported here (see thrown error below).
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
		if (!this->dirichlets_by_removing_from_dof_vector)
		{
			throw_runtime_error("This must be implemented. The rows and columns of the eigenproblem matrices corresponding to Dirichlet dofs should be removed. Likely has to be done on an elemental level");
		}
	}

	// Dof values at history time level t (t=0 is the current time), as a plain std::vector<double>
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

	// Current dof values, together with a per-dof flag marking whether the dof is a nodal position dof
	// (i.e. belongs to a node's variable_position_pt(), as opposed to a "physical" field dof); the flag
	// is determined by scanning all nodes of all submeshes for equation numbers coming from their position data.
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


    // oomph-lib's get_dofs(t,...) does not account for the variable_position_pt() dofs of moving nodes
    // (see the analogous comment in set_dofs below) - this override fixes that up afterwards.
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

	// oomph-lib's set_dofs(t,...) does not update the variable_position_pt() dofs of moving nodes; this
	// override does the base-class assignment and then additionally writes back the position dofs.
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

    // Same fix-up as the DoubleVector overload above, for the raw-pointer-array variant of set_dofs
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

	// Sets the dof values at history time level t from a plain std::vector, validating both the vector
	// length (must match ndof()) and that t is within the timestepper's available history storage.
	void Problem::set_history_dofs(unsigned t, const std::vector<double> &inp)
	{
		oomph::DoubleVector dofs;
		dofs.build(this->dof_distribution_pt(), 0.0);
		if (inp.size() != this->ndof())
		{
			std::ostringstream oss; oss << "Try to set dofs of length " << inp.size() << " while problem has " << this->ndof() << " dofs";
			throw_runtime_error("Mismatch in dof vector size. " + oss.str());
		}
		if (t>=this->time_stepper_pt()->ntstorage()) 
		        throw_runtime_error("Wrong history offset");
		for (unsigned int i = 0; i < this->ndof(); i++)
			dofs[i] = inp[i];
		this->set_dofs(t, dofs);
	}

	// Collects the current values of all pinned Data across every submesh: nodal values, optionally
	// (with_pos) pinned nodal position dofs, and pinned internal (elemental) Data values. This is
	// independent of the (unpinned) dof vector and is used to save/restore state that is not part of
	// get_current_dofs()/set_current_dofs() (e.g. across mesh adaptation or continuation branch switches).
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

	// Enforces Dirichlet conditions in-place on an already-assembled (residuals, jacobian) pair for the
	// matrix-manipulation strategy: for every globally Dirichlet-pinned equation number, the residual
	// entry is set to 0 (dof is not part of the nonlinear system to solve for) and, if a Jacobian is
	// given, its row is replaced by the identity row (1 on the diagonal, 0 elsewhere) while any column
	// entries in *other* rows referring to a pinned dof are zeroed out (since that dof's value does not
	// change during the solve, so its contribution to other residuals' derivatives must not appear).
	// Operates directly on the local (possibly MPI-distributed) chunk of the matrix/vector.
	void Problem::remove_dirichlets_by_matrix_manipulation(oomph::DoubleVector &residuals,oomph::CRDoubleMatrix *jacobian)
	{
		const std::unordered_set<unsigned long> dof_set=dirichlet_info.get_global_pinned_equation_set();


		if (!dof_set.empty())
		{
			for (const auto &eqn_number : dof_set)
			{
				int eqn_number_local=static_cast<int>(eqn_number)-residuals.first_row();
				if (eqn_number_local>=0 && eqn_number_local<(int)residuals.nrow_local())
				{
					residuals[eqn_number_local] = 0.0;
				}
			}
			if (jacobian)
			{

				//const int num_rows = jacobian->nrow();
				//const int num_cols = jacobian->ncol();
				const int first_row=jacobian->distribution_pt()->first_row();
				const int num_local_rows=jacobian->nrow_local();
				//std::cout << "Jacobian size: " << num_rows << " x " << num_cols << std::endl << "LOCAL START " << first_row << " LOCAL ROWS " << num_local_rows << std::endl << std::flush;

				double * values=jacobian->value();
				const int * col_index=jacobian->column_index();
				const int * row_start=jacobian->row_start();

				for (int row = 0; row < num_local_rows; ++row) {
					bool is_constrained_row = dof_set.count(row+first_row);

					for (int i = row_start[row]; i < row_start[row + 1]; ++i) {
						int col = col_index[i];

						if (is_constrained_row)
						{
							values[i] = (col == row+first_row) ? 1.0 : 0.0;
						} else if (dof_set.count(col))
						{
							values[i] = 0.0;
						}
					}
				}
			}
		}
	}

	// Top-level residual assembly entry point: either the normal elemental assembly (with, if the
	// matrix-manipulation Dirichlet strategy is active, a subsequent zeroing of the Dirichlet rows via
	// remove_dirichlets_by_matrix_manipulation), or the user-supplied custom residual when
	// use_custom_residual_jacobian is set.
	void Problem::get_residuals(oomph::DoubleVector &residuals)
	{
		if (!use_custom_residual_jacobian)
		{
			get_residuals_by_elemental_assembly(residuals);
			if (!this->dirichlets_by_removing_from_dof_vector)
			{
				if (this->bifurcation_tracking_mode!="") throw_runtime_error("TODO: Cannot remove dirichlet dofs from the residual vector by matrix manipulation when bifurcation tracking is active, since the user-provided residual vector would still contain contributions from the dirichlet dofs, which would be wrong");
				this->remove_dirichlets_by_matrix_manipulation(residuals);
			}
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
			if (!this->dirichlets_by_removing_from_dof_vector)
			{
				throw_runtime_error("TODO: Cannot use custom residuals when dirichlet conditions are not removed from the dof vector, since the user-provided residual vector would still contain contributions from the dirichlet dofs, which would be wrong");
			}

			for (unsigned int i = 0; i < info.residuals.size(); i++)
				residuals[i] = info.residuals[i];
		}
	}

	// d(residuals)/d(parameter) entry point; same dispatch as get_residuals()/get_jacobian() between
	// normal elemental assembly and the custom-residual-Jacobian path (there identified by parameter name).
	void Problem::get_derivative_wrt_global_parameter(double* const& parameter_pt,oomph::DoubleVector& result)
	{
		if (!use_custom_residual_jacobian)
		{
			get_derivative_wrt_global_parameter_elemental_assembly(parameter_pt,result);
			if (!this->dirichlets_by_removing_from_dof_vector)
			{
				 //throw_runtime_error("TODO: Cannot remove dirichlet dofs from the derivative by a global parameter by matrix manipulation yet.");

				 //This is of course problematic if the DirichletBC depends on the parameter, but it is also problematic in the case of removing Dirichlets from the dofs
				 //TODO: However, here, it could be patched by providing a corresponding function to calculate dDirichlet/dparam 
				 this->remove_dirichlets_by_matrix_manipulation(result);
			}
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
			if (!this->dirichlets_by_removing_from_dof_vector)
			{
				 throw_runtime_error("TODO: Cannot remove dirichlet dofs from the derivative by a global parameter by matrix manipulation yet.");
			}

			for (unsigned int i = 0; i < info.residuals.size(); i++)
				result[i] = info.residuals[i];
		}

	}
	// Top-level Jacobian assembly entry point; same dispatch/Dirichlet-handling pattern as get_residuals()
	void Problem::get_jacobian(oomph::DoubleVector &residuals, oomph::CRDoubleMatrix &jacobian)
	{
		if (!use_custom_residual_jacobian)
		{
			get_jacobian_by_elemental_assembly(residuals, jacobian);
			if (!this->dirichlets_by_removing_from_dof_vector)
			{
				if (this->bifurcation_tracking_mode!="") throw_runtime_error("TODO: Cannot remove dirichlet dofs from the jacobian matrix by matrix manipulation when bifurcation tracking is active, since the user-provided jacobian matrix would still contain contributions from the dirichlet dofs, which would be wrong");
				this->remove_dirichlets_by_matrix_manipulation(residuals,&jacobian);
			}
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
			if (!this->dirichlets_by_removing_from_dof_vector)
			{
				throw_runtime_error("TODO: Cannot remove dirichlet dofs from the jacobian matrix by matrix manipulation when using custom jacobian, since the user-provided jacobian matrix would still contain contributions from the dirichlet dofs, which would be wrong");
			}
		}
	}

	// Finds the global parameter index whose value() address is exactly ptr; throws if not found (e.g. if
	// ptr does not belong to any registered global parameter of this problem)
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

	// Sets one of oomph-lib's named arclength-continuation control parameters (exposed by name since
	// they are protected data members of oomph::Problem without individual setters); boolean-valued ones
	// are encoded as val>0.5. Throws on an unrecognized name.
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

	// Toggles the internal __replace_RJM_by_param_deriv pointer, which, when non-NULL, makes subsequent
	// residual/Jacobian/mass-matrix assembly calls return the derivative w.r.t. the named global
	// parameter instead of the normal residual/Jacobian/mass matrix (used internally e.g. for parameter
	// derivative computations that must reuse the same elemental assembly loop machinery).
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

	// Performs one pseudo-arclength continuation step in the named global parameter, forwarding to
	// oomph-lib's arc_length_step_solve with the parameter's value() as the tracked parameter pointer.
	double Problem::arc_length_step(const std::string param, const double &ds, unsigned max_adapt)
	{
		if (!global_params_by_name.count(param))
			throw_runtime_error("Cannot continue in the global parameter " + param + ", since it is not present in the problem");
		auto *p = global_params_by_name[param];
		double *valptr = &(p->value());
		//		this->set_analytic_dparameter(valptr);
		return this->arc_length_step_solve(valptr, ds, max_adapt);
	}

	// d(dof)/ds (arclength derivative) as a plain std::vector, exposing oomph-lib's internal Dof_derivative
	std::vector<double> Problem::get_arclength_dof_derivative_vector()
	{
		std::vector<double> res(Dof_derivative.size());
		for (unsigned i = 0; i < res.size(); i++)
			res[i] = dof_derivative(i);
		return res;
	}

	// Dof values at the last continuation step (oomph-lib's internal Dof_current), as a std::vector
	std::vector<double> Problem::get_arclength_dof_current_vector()
	{
		std::vector<double> res(Dof_current.size());
		for (unsigned i = 0; i < res.size(); i++)
			res[i] = dof_current(i);
		return res;
	}

    // Writes ddof/curr (locally distributed dof-derivative and current-dof vectors) into oomph-lib's
    // internal Dof_derivative/Dof_current, resizing them first if necessary; used to seed/restore the
    // arclength continuation state (e.g. from Python-side branch switching).
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

	// Seeds oomph-lib's internal continuation state (current parameter value, direction, step-size
	// derivative magnitude) so that a subsequent arclength step continues smoothly from (p0, dp).
	void Problem::update_param_info_for_continuation(double dp,double p0)
	{
		Parameter_current=p0;
		if (dp>0) Continuation_direction=1; else  if (dp<0) Continuation_direction=-1;
		Parameter_derivative=abs(dp);

		Arc_length_step_taken=true;
	}



    // Installs oomph-lib's MyFoldHandler (fold/limit-point bifurcation tracking assembly handler) with a
    // given starting null-eigenvector guess; if block_solve, additionally switches to the augmented block
    // linear solver that exploits the bordered system's block structure instead of a full dense solve.
    void Problem::activate_my_fold_tracking(double *const &parameter_pt, const oomph::DoubleVector &eigenvector, const bool &block_solve)
	{
		reset_assembly_handler_to_default();
		this->assembly_handler_pt() = new MyFoldHandler(this, parameter_pt, eigenvector);
		if (block_solve)
		{
			this->linear_solver_pt() = new oomph::AugmentedBlockFoldLinearSolver(this->linear_solver_pt());
		}
	}

	// Same as above, but lets MyFoldHandler determine the initial null-eigenvector itself
	void Problem::activate_my_fold_tracking(double *const &parameter_pt, const bool &block_solve)
	{
		reset_assembly_handler_to_default();
		this->assembly_handler_pt() = new MyFoldHandler(this, parameter_pt);
		if (block_solve)
		{
			this->linear_solver_pt() = new oomph::AugmentedBlockFoldLinearSolver(this->linear_solver_pt());
		}
	}

	// Installs the Hopf-bifurcation tracking assembly handler (bordered system with complex null vector
	// null_real+i*null_imag and angular frequency omega); optionally with the block Hopf linear solver.
	void Problem::activate_my_hopf_tracking(double *const &parameter_pt, const double &omega, const oomph::DoubleVector &null_real, const oomph::DoubleVector &null_imag, const bool &block_solve)
	{
		reset_assembly_handler_to_default();
		this->assembly_handler_pt() = new MyHopfHandler(this, parameter_pt, omega, null_real, null_imag);
		if (block_solve)
		{
			this->linear_solver_pt() = new oomph::BlockHopfLinearSolver(this->linear_solver_pt());
		}
	}

	// oomph-lib hook (raw parameter_pt) resolved to the parameter's name and forwarded to the
	// string-named virtual so Python subclasses can react to global-parameter changes by name.
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

	// Same name-resolution dispatch as actions_after_change_in_global_parameter above, but for parameter increases
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

	// Installs the azimuthal-symmetry-breaking bifurcation tracking assembly handler, which requires two
	// named special residual forms ("azimuthal_real_eigen" and, unless "<NONE>", "azimuthal_imag_eigen")
	// describing the linearized real/imaginary azimuthal-mode residual contributions.
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

	// Installs the pitchfork-bifurcation tracking assembly handler with the given symmetry-breaking
	// eigenvector (block-solve variant is not wired up here, only the plain handler)
	void Problem::activate_my_pitchfork_tracking(double *const &parameter_pt, const oomph::DoubleVector &symmetry_vector, const bool &)
	{
		//	this->activate_pitchfork_tracking(parameter_pt, symmetry_vector, block_solve);
		reset_assembly_handler_to_default();
		this->assembly_handler_pt() = new MyPitchForkHandler(this, parameter_pt, symmetry_vector);
	}

	// d(residuals)/d(param), by name, as a plain std::vector (thin convenience wrapper around get_derivative_wrt_global_parameter)
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

	// Post-continuation-step hook: for fold tracking, re-aligns the handler's constraint vector C with
	// the current null-eigenvector estimate to keep the bordering well-conditioned along the branch.
	void Problem::after_bifurcation_tracking_step()
	{
		if (dynamic_cast<MyFoldHandler *>(this->assembly_handler_pt()))
		{
			dynamic_cast<MyFoldHandler *>(this->assembly_handler_pt())->realign_C_vector();
		}
	}

	// Directly sets oomph-lib's dof_derivative (d(dof)/ds) to a prescribed direction ddir and marks
	// Arc_length_step_taken, effectively priming the arclength continuation tangent by hand (e.g. for
	// branch switching) instead of computing it from a previous step.
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

	// Angular frequency omega of the currently tracked bifurcation, if it is a Hopf or azimuthal
	// (which are the two oscillatory/complex bifurcation types); 0 otherwise.
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

	// Whether a bifurcation is currently being tracked and, if so, the complex eigenvalue lambda =
	// Re(lambda) + i*omega associated with it (Re(lambda) is only meaningful/nonzero when tracking via
	// the lambda_tracking_real helper parameter, e.g. for eigenbranch tracking; otherwise 0).
	std::tuple<bool,std::complex<double>> Problem::get_bifurcation_tracking_info()
	{
		bool active=false;
		std::complex<double> lambda(0.0,0.0);
		if (bifurcation_tracking_mode == "hopf" && (dynamic_cast<MyHopfHandler *>(this->assembly_handler_pt())))
		{
			auto *h = dynamic_cast<MyHopfHandler *>(this->assembly_handler_pt());
			active=true; lambda=std::complex<double>((h->bifurcation_parameter_pt()==&this->lambda_tracking_real ? this->lambda_tracking_real :0.0),h->omega());			
		}
		else if (bifurcation_tracking_mode == "azimuthal" && (dynamic_cast<AzimuthalSymmetryBreakingHandler *>(this->assembly_handler_pt())))
		{
			auto *h = dynamic_cast<AzimuthalSymmetryBreakingHandler *>(this->assembly_handler_pt());
			active=true; lambda=std::complex<double>((h->bifurcation_parameter_pt()==&this->lambda_tracking_real ? this->lambda_tracking_real :0.0),h->omega()); 
		}
		else if (bifurcation_tracking_mode == "fold" && (dynamic_cast<MyFoldHandler *>(this->assembly_handler_pt())))
		{
			auto *h = dynamic_cast<MyFoldHandler *>(this->assembly_handler_pt());
			active=true; lambda=std::complex<double>((h->bifurcation_parameter_pt()==&this->lambda_tracking_real ? this->lambda_tracking_real : 0.0),0.0);
		}
		else if (bifurcation_tracking_mode == "pitchfork" && (dynamic_cast<MyPitchForkHandler *>(this->assembly_handler_pt())))
		{
			auto *h = dynamic_cast<MyPitchForkHandler *>(this->assembly_handler_pt());
			active=true; lambda=std::complex<double>((h->bifurcation_parameter_pt()==&this->lambda_tracking_real ? this->lambda_tracking_real : 0.0),0.0);
		}
		return std::make_tuple(active,lambda);
	}

    // Selects one of oomph-lib's built-in sparse assembly implementations by name (they differ in the
    // container used to accumulate (row,col)->value triplets before compressing to CSR: vectors of
    // pairs, two parallel vectors, maps, lists, or two plain arrays - a performance/memory trade-off).
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

	// Selects how the distributed Jacobian matrix is partitioned across MPI ranks ("default" defers to
	// oomph-lib's own default, "problem" mirrors the problem's dof distribution, "uniform" splits rows
	// evenly regardless of dof distribution); no-op when built without MPI.
	void Problem::set_dist_problem_matrix_distribution(const std::string & mode)
	{
		#ifdef OOMPH_HAS_MPI
		if (mode =="default") {this->distributed_problem_matrix_distribution()=oomph::Problem::Default_matrix_distribution;}
		else if (mode=="problem") {this->distributed_problem_matrix_distribution()=oomph::Problem::Problem_matrix_distribution;}
		else if (mode=="uniform") {this->distributed_problem_matrix_distribution()=oomph::Problem::Uniform_matrix_distribution;}
		else
		{
			throw_runtime_error("Unknown distributed problem matrix distribution mode: " + mode);
		}
		#endif
	}
    std::string Problem::get_dist_problem_matrix_distribution() 
	{
		#ifdef OOMPH_HAS_MPI
		switch (this->distributed_problem_matrix_distribution())
		{
		case oomph::Problem::Default_matrix_distribution:
			return "default";
		case oomph::Problem::Problem_matrix_distribution:
			return "problem";
		case oomph::Problem::Uniform_matrix_distribution:
			return "uniform";
		default:
			return "unknown";
		}
		#else
		return "nompi";
		#endif
	}

    // The (real, or complex for Hopf/azimuthal) eigenvector of the currently tracked bifurcation; empty
    // if no bifurcation is being tracked.
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

	// Installs the periodic-orbit tracking assembly handler, representing the orbit as a B-spline of the
	// given order through the initial guessed history (time series of dof snapshots) with period T; knots
	// gives the (possibly non-uniform) B-spline knot vector and T_constraint_mode selects how the period
	// is constrained (e.g. fixed phase condition) in the augmented system.
	void Problem::start_orbit_tracking(const std::vector<std::vector<double>> &history, const double &T,int bspline_order,int gl_order,std::vector<double> knots,unsigned T_constraint_mode)

	{
		reset_assembly_handler_to_default();
		this->assembly_handler_pt() = new PeriodicOrbitHandler(this, T,history,bspline_order,gl_order,knots,T_constraint_mode);
	}

	// Restores the default (non-augmented) assembly handler; also clears bifurcation_tracking_mode
	// implicitly via the callers (this function itself only deals with the oomph-lib assembly handler).
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

	// Truncates the dof vector/distribution back down to n_unaugmented_dofs entries, discarding any
	// augmentation dofs (bifurcation tracking, arclength, custom DofAugmentations) appended on top of the
	// physical dofs, and resets the sparse-assembly scratch buffers (whose size depended on the dof count).
	// No-op if no augmentation is currently active (n_unaugmented_dofs==0 is used as the "not augmented" sentinel).
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


	// High-level entry point for (de)activating bifurcation tracking, dispatching to the appropriate
	// activate_my_*_tracking() based on typus ("fold","pitchfork","hopf","azimuthal"). param selects the
	// tracked parameter by name, or the special sentinel "<LAMBDA_TRACKING>" to track the (helper)
	// lambda_tracking_real parameter instead (used for eigenbranch tracking not tied to an actual problem
	// parameter). eigenv1/eigenv2 seed the real/imaginary parts of the initial null-eigenvector guess
	// (truncated/zero-padded to ndof()). Passing param=="" (or typus=="" / "none") deactivates tracking.
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

	// Inverse of get_current_pinned_values(): writes inp (in the same nodal-value / [position] / internal
	// order) back into the pinned Data entries at history time level t. Throws if inp is shorter than the
	// number of pinned entries encountered (a size mismatch is only detected once too many values would be consumed).
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
	
	
	// Opens fname as the problem's log file, closing/replacing any previously open one; if activate_logging
	// is true, immediately makes it the active global logging stream (pyoomph::set_logging_stream), else
	// it is only prepared and can be attached to logging later. fname=="" closes the current log file
	// (deactivating logging), independent of any other state.
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

	// Suppresses (or restores) oomph-lib's own console output (Newton solve messages, linear solver
	// timing) and redirects oomph::oomph_info to a null stream when quiet.
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

	// Computes the directional second derivative H[dir,dir] of the residuals (i.e. the Hessian tensor
	// contracted twice with the same direction dir), element by element, without ever forming the full
	// sparse Hessian tensor (unlike assemble_hessian_tensor below). Used e.g. for second-order Newton
	// corrector steps / normal-form coefficients along a single direction, where only this contraction
	// is needed. The commented-out block below is an earlier alternative implementation via
	// get_hessian_vector_products() (kept for reference).
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

	// Assembles the full sparse rank-3 Hessian tensor d^2(residual_i)/d(dof_j)d(dof_k), element by
	// element: each element contributes a dense nvar x nvar x nvar block (flattened to nvar x nvar*nvar
	// by elem_pt->assemble_hessian_tensor), which is then scattered into the global sparse tensor using
	// the assembly handler's local-to-global equation numbering. Entries below Numerical_zero_for_sparse_assembly
	// in magnitude are dropped to keep the tensor sparse. If symmetric is set, result exploits (and the
	// caller must ensure) symmetry under exchange of the last two indices (j,k) to roughly halve the work.
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


// Experimental (currently disabled, since the macro below is commented out) specialized dense-matrix
// representations for the periodic-orbit elemental Jacobian, which is a large NT x NT block matrix (NT
// = number of time slices of the orbit discretization) with (in general) only a banded subset of blocks
// actually populated. Storing it as a plain oomph::DenseMatrix would waste O(NT^2) memory/time; these
// classes instead only store per-block (or per-band) dense sub-blocks. Not used unless
// PYOOMPH_PERIODIC_ORBIT_BAND_MATRIX is defined; kept here for future optimization work.
//#define PYOOMPH_PERIODIC_ORBIT_BAND_MATRIX
#ifdef PYOOMPH_PERIODIC_ORBIT_BAND_MATRIX
	// Sparse-by-block dense matrix: only allocates a base_ndof x base_ndof dense sub-block for a given
	// (block-row, block-col) pair the first time it is written to; missing blocks read as all-zero.
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




	// Periodic band-block matrix: only the bandwidth-b diagonal band of NTxNT blocks is stored (plus one
	// extra row/column for the period constraint), further reducing memory/time compared to
	// PeriodicOrbitAssemblyBlockDenseMatrix above when the true coupling between time slices is local.
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

 	// Specialized sparse assembly for the periodic-orbit augmented system (adapted from oomph-lib's
 	// generic sparse_assemble_row_or_column_compressed, see the base-problem variant below). Periodic
 	// orbit elements have very large elemental Jacobians (they couple all NT time-slice copies of the
 	// underlying dofs), so instead of using one of the configurable Sparse_assembly_method strategies,
 	// this always accumulates entries in per-row/column std::map<unsigned,double> buffers (matrix_data_map)
 	// before compressing to CSR. Only supports a single residual vector and a single matrix at a time
 	// (n_vector==1, n_matrix==1) and requires a PeriodicOrbitHandler to be the active assembly handler.
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

    // Overrides oomph-lib's sparse assembly dispatcher: routes to the periodic-orbit-specific
    // implementation when a PeriodicOrbitHandler is active, otherwise falls back to the normal
    // oomph-lib assembly (which itself uses whichever Sparse_assembly_method is configured).
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


	// NOTE: currently unimplemented (unconditionally throws below) - intended to precompute, for every
	// global equation, the set of Jacobian entries (global_eqs_to_jacobian_buffer_index) it contributes
	// to, so a later assembly pass could write directly into a preallocated buffer instead of building
	// the sparsity pattern from scratch each time. Left in place (with the working code commented out)
	// as a starting point for a future performance optimization.
	void Problem::update_jacobian_csr_structure()
	{
		//unsigned ndof = this->get_n_unaugmented_dofs();
		if (this->get_n_unaugmented_dofs()!=0) throw_runtime_error("This does not work if you have augmented dofs");
		global_eqs_to_jacobian_buffer_index.resize(this->ndof());
		for (unsigned i = 0; i < this->ndof(); i++)
		{
			global_eqs_to_jacobian_buffer_index[i].clear();
		}
		throw_runtime_error("Implement and check performance");
		for (unsigned int ie=0;ie<mesh_pt()->nelement();ie++)
		{
			/*
			oomph::GeneralisedElement* elem_pt = mesh_pt()->element_pt(ie);
			const unsigned nvar = assembly_handler_pt()->ndof(elem_pt);
			for (unsigned i = 0; i < nvar; i++)
			{
				unsigned eqn_number = assembly_handler_pt()->eqn_number(elem_pt, i);
				for (unsigned j = 0; j < nvar; j++)
				{
					double value = el_jacobian(i, j);
					if (std::fabs(value) > Numerical_zero_for_sparse_assembly)
					{
						unsigned unknown = assembly_handler_pt()->eqn_number(elem_pt, j);	
						global_eqs_to_jacobian_buffer_index[eqn_number].insert(unknown);
					}
				}
			}
				*/
		}
	}

 	// Sparse assembly of only the "base" (unaugmented) part of the problem, i.e. restricted to the first
 	// get_n_unaugmented_dofs() equations/dofs - used while an augmented system (bifurcation tracking,
 	// arclength, custom DofAugmentations) is active and the base-problem contribution to the bordered
 	// system's residual/Jacobian block must be assembled separately from the augmentation rows/columns.
 	// Structurally mirrors sparse_assemble_row_or_column_compressed_for_periodic_orbit / oomph-lib's own
 	// generic sparse assembly (map-based per-row/column accumulation, then compression to CSR); only
 	// supports the non-distributed, non-MPI-parallel case for now (see the throws below).
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




	// Appends the vectors/scalars/parameters registered in aug to the problem's dof pointer array (as raw
	// pointers to the augmentation's own storage, or to a global parameter's value()), remembers the
	// original (unaugmented) dof count in n_unaugmented_dofs, finalizes aug's split_offsets so aug.split()
	// can later decompose the augmented dof vector, and rebuilds the dof distribution to the new size.
	// Only one augmentation may be active at a time (see reset_augmented_dof_vector_to_nonaugmented()).
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
				this->GetDofPtr().push_back(&this->get_global_parameter(aug.augmented_parameters[pindex])->value());
				pindex++;
			}
		}
		aug.split_offsets.push_back(this->GetDofPtr().size());
		aug.finalized=true;

		this->GetDofDistributionPt()->build(this->communicator_pt(),this->GetDofPtr().size(), false);
	}


	// Assembles several requested quantities ("what", paired with "contributions" restricting which
	// residual contribution each applies to, and "params" naming any parameters needed for parameter
	// derivatives/Hessian-vector products) in a single elemental assembly pass: temporarily installs a
	// CustomMultiAssembleHandler as the active assembly handler (which knows how to pack all requested
	// residual vectors/Jacobian-like matrices as the "vectors"/"matrices" of a generic sparse assembly),
	// runs sparse_assemble_row_or_column_compressed_base_problem(), then unpacks the resulting raw
	// buffers into plain std::vectors (data/csrdata) for return to the caller (e.g. the Python binding),
	// freeing the raw buffers as it goes. Always operates on the unaugmented dof count.
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

	}


	// Rebuilds the global bookkeeping of "defined fields" (all field names known across every loaded
	// bulk/interface code) and, for every named residual/Jacobian combination, which fields contribute to
	// its residual and which (field,field) pairs contribute to its Jacobian - both as booleans
	// (residual_contributing_fields/jacobian_contributing_fields) and as the actual set of codes
	// responsible (residual_contributing_codes/jacobian_contributing_codes), the latter mainly used for
	// diagnostics (see get_jacobian_information_string()). Finally, for each residual, marks fields that
	// have no Jacobian contribution in either their row or their column
	// (pin_due_to_empty_jacobian_row_or_col) - such fields would make the Jacobian singular and must be
	// pinned when that residual is active (see _set_solved_residual()). Must be called whenever the set
	// of loaded codes changes (loading new equations) and before equation numbering can rely on the
	// pinning information being up to date.
	void Problem::assemble_defined_field_list()
	{
		defined_fields.clear();
		defined_fields_to_domain.clear();
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
					defined_fields_to_domain.push_back(dc);
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
				if (my_i==-1) 
				{
					//std::cout << "Warning: Could not find a contribution entry for residual/jacobian combination " << residual_names[i] << " in code of " << dc->get_file_name() << ". This code will not contribute to this residual/jacobian combination." << std::endl;
					continue; // This code does not contribute to this residual at all
				}
				for (unsigned j=0;j<ft->contribution_entries_size;j++)
				{
					std::string fn=ft->contribution_names[j];
					if (field_name_to_index.find(fn)==field_name_to_index.end())
					{
						throw_runtime_error("Undefined field " + fn + " in contribution entry for residual/jacobian combination " + residual_names[i]);
					}
					unsigned row_index=field_name_to_index[fn];
					residual_contributing_fields[i][row_index]=residual_contributing_fields[i][row_index]| ft->contributes_to_residual[my_i][j];

					if (ft->contributes_to_residual[my_i][j])
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
						//std::cout << "in code " << dc->get_file_name() << ": Checking jacobian contribution for residual/jacobian combination " << residual_names[i] << " for row field " << fn << " and column field " << fn2 << " indices " << row_index << "," << col_index << "  my_i " << my_i << " j,k "<< j << "," << k << " VALUE " << ft->contributes_to_jacobian[my_i][j][k] << std::endl;
						jacobian_contributing_fields[i][row_index][col_index]=jacobian_contributing_fields[i][row_index][col_index] | ft->contributes_to_jacobian[my_i][j][k];
						if (ft->contributes_to_jacobian[my_i][j][k])
						{
							jacobian_contributing_codes[i][row_index][col_index].insert(dc);
						}
					}
				}
				
			}
		}
		// A field must be pinned for a given residual if it has no Jacobian contribution as a row
		// (nothing derives w.r.t. it, i.e. no equation actually constrains it) or as a column (its own
		// equation - if any - does not depend on any field, which cannot happen if it has a row contribution,
		// but is checked independently since row and column may come from different res/jac entries).
		pin_due_to_empty_jacobian_row_or_col.resize(residual_names.size(),std::vector<bool>(defined_fields.size(),false));
		for (unsigned i=0;i<residual_names.size();i++)
		{
			for (unsigned j=0;j<defined_fields.size();j++)
			{
				bool has_row_contribs=false;
				bool has_col_contribs=false;
				for (unsigned k=0;k<defined_fields.size();k++)
				{
					if (jacobian_contributing_fields[i][j][k])
					{
						has_row_contribs=true;		
						if (has_col_contribs) break;				
					}
					if (jacobian_contributing_fields[i][k][j])
					{
						has_col_contribs=true;
						if (has_row_contribs) break;						
					}
				}
				if (!has_row_contribs || !has_col_contribs)
				{
					//std::cout << "Pinning field " << defined_fields[j] << " for residual/jacobian combination " << residual_names[i] << " because it has no jacobian contributions in row or column direction" << std::endl;
					pin_due_to_empty_jacobian_row_or_col[i][j]=true;
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
		
		removed_fields_due_to_missing_jacobian_row_or_col.resize(defined_fields.size(),false);
		
	}


	// Whether the field with the given global field index has actually been removed from the dofs
	// because it had no Jacobian row/column contribution under the currently active residual (see
	// _set_solved_residual()); negative indices (fields that cannot be attributed to a single global
	// field, e.g. augmentation dofs) are always reported as not removed.
	bool Problem::is_field_removed_from_dofs_due_to_missing_jacobian_row(int global_field_index)
	{
		if (global_field_index<0) return false;
		else return removed_fields_due_to_missing_jacobian_row_or_col[global_field_index];
	}

	// Builds a human-readable report of the Jacobian sparsity structure computed by
	// assemble_defined_field_list() - for every named residual/Jacobian combination, which fields are
	// defined, which contribute to the residual, and (as an ASCII matrix, for small enough problems) the
	// Jacobian contribution pattern between fields, flagging fields that had to be pinned due to a
	// missing row/column. Returns the report string together with a bool that is false if any problem
	// (e.g. a field with an empty Jacobian row/column) was found, intended to help users debug singular
	// Jacobians / incompletely specified equation systems.
	std::tuple<std::string,bool> Problem::get_jacobian_information_string()
	{
		std::ostringstream ss;
		bool all_good=true;
		ss << "Defined fields: " << std::endl;
		for (unsigned int i=0;i<defined_fields.size();i++)
		{
			ss << "\t" << i << "\t" << defined_fields[i] << std::endl;
		}
		ss << std::endl;
		std::vector<bool> has_any_contributions(defined_fields.size(),false);
		for (unsigned int ri=0;ri<residual_names.size();ri++)
		{
			std::string combi=residual_names[ri];
			bool ignored_residuals=true; // Happens e.g. for azimuthal contributions
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
					if (pin_due_to_empty_jacobian_row_or_col[ri][i]) continue;
					else if (i/100) ss << i/100 << " ";
					else ss << "  ";
					has_any_contributions[i]=true;
				}
				ss << "|" << std::endl;
				ss << "\t    | ";
				for (unsigned int i=0;i<defined_fields.size();i++)				
				{
					if (pin_due_to_empty_jacobian_row_or_col[ri][i]) continue;
					else if ((i%100)/10) ss << (i%100)/10 << " ";
					else ss << "  ";
				}
				ss <<"|" << std::endl;
				ss << "\t    | ";
				for (unsigned int i=0;i<defined_fields.size();i++)				
				{
					if (pin_due_to_empty_jacobian_row_or_col[ri][i]) continue;
					else ss << i%10 << " ";
				}
				ss << "|" << std::endl;
				ss << "\t----|";
				for (unsigned int i=0;i<defined_fields.size();i++)				
				{
					if (pin_due_to_empty_jacobian_row_or_col[ri][i]) continue;
					else ss  << "--";
				}
				ss <<"-|" << std::endl;
			}
			else if (defined_fields.size()>9)
			{
				ss << "\t    | ";
				for (unsigned int i=0;i<defined_fields.size();i++)				
				{
					if (pin_due_to_empty_jacobian_row_or_col[ri][i]) continue;
					else if (i/10) ss << i/10 << " ";
					else ss << "  ";
					has_any_contributions[i]=true;
				}
				ss << "|" << std::endl;
				ss << "\t    | ";
				for (unsigned int i=0;i<defined_fields.size();i++)				
				{
					if (pin_due_to_empty_jacobian_row_or_col[ri][i]) continue;
					else ss << i%10 << " ";
				}
				ss << "|" << std::endl;
				ss << "\t----|";
				for (unsigned int i=0;i<defined_fields.size();i++)				
				{
					if (pin_due_to_empty_jacobian_row_or_col[ri][i]) continue;
					else ss  << "--";
				}
				ss << "-|" <<std::endl;
			}
			else
			{
				ss << "\t    | ";
				for (unsigned int i=0;i<defined_fields.size();i++)				
				{
					if (pin_due_to_empty_jacobian_row_or_col[ri][i]) continue;
					else ss << i << " ";
					has_any_contributions[i]=true;
				}
				ss <<"|"<< std::endl;
				ss << "\t----|";
				for (unsigned int i=0;i<defined_fields.size();i++)				
				{
					if (pin_due_to_empty_jacobian_row_or_col[ri][i]) continue;
					else ss  << "--";
				}
				ss << "-|" << std::endl;
			}
			
			std::vector<std::set<DynamicBulkElementCode*>> listed_domain_contributions;
			for (unsigned int i=0;i<defined_fields.size();i++)
			{				
				if (pin_due_to_empty_jacobian_row_or_col[ri][i]) continue;
				else ss << "  ";
				ss<<"\t" << std::setfill(' ') << std::setw(3) << i << " | ";		
				for (unsigned int j=0;j<defined_fields.size();j++)
				{
					if (pin_due_to_empty_jacobian_row_or_col[ri][j]) continue;
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
						// Each distinct set of contributing codes gets its own letter (A, B, C, ...), printed
						// in the matrix cell and resolved to the actual domain name(s) below the matrix.
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
				if (pin_due_to_empty_jacobian_row_or_col[ri][i]) continue;
				ss << "--";
			}
			ss << "-|" << std::endl;
			ss<< "\tRes | " ;
			std::set<unsigned> residual_contributions_with_zero_jacobian_row_or_col;
			for (unsigned int i=0;i<defined_fields.size();i++)
			{
				if (pin_due_to_empty_jacobian_row_or_col[ri][i]) 
				{
					if (residual_contributing_fields[ri][i]) residual_contributions_with_zero_jacobian_row_or_col.insert(i);
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
						for  (auto * dc : listed_domain_contributions[found]) if (!dc->get_code()->is_residual_assembly_ignored(residual_names[ri])) ignored_residuals=false;
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

			if (residual_contributions_with_zero_jacobian_row_or_col.size()>0 && !ignored_residuals)
			{
				ss << "\t|WARNING|: Following fields have residual contributions, but a zero Jacobian row/column: ";
				unsigned int count=residual_contributions_with_zero_jacobian_row_or_col.size();
				for (auto i : residual_contributions_with_zero_jacobian_row_or_col)
				{
					ss << i << +" ("+defined_fields[i]+")"+ (--count ? ", " : "");
				}
				ss << std::endl;
				ss << std::endl;
				all_good=false;
			}
			
		}

		for (unsigned int i=0;i<defined_fields.size();i++)
		{
			if (!has_any_contributions[i])
			{
				// Check if it is pinned, then it is fine
				DynamicBulkElementCode * dc=defined_fields_to_domain[i];
				auto *ft=dc->get_func_table();
				bool pinned=false;
				for (unsigned int j=0;j<ft->Dirichlet_set_size;j++)
				{
					if (ft->dirichlet_field_index_to_global_field_index[j]==(int)i)
					{						
						if (ft->Dirichlet_set[j])
						{
							//std::cout << "IT IS PINNED " << i << "  " << j << std::endl;
							pinned=true;
							break;
						}
					}
				}
				if (pinned) continue;
				ss << "\t|WARNING|: Field " << i << " \"" << defined_fields[i] << "\" has an empty row or column in all Jacobians." << std::endl;
				all_good=false;
			}
		}

		return std::make_tuple(ss.str(), all_good);
		
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
		if (endindex<0) return res;
		if (endindex>=(int)split_offsets.size())  throw_runtime_error("Invalid end index");
		for (int i=(int)startindex;i<endindex;i++)
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
