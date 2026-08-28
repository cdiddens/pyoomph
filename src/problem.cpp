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

#include <limits>
#include <memory>
#include <cstring>
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
	// Merges one per-space flag set. Kept as a helper so that adding a flag to
	// JITFuncSpec_RequiredShapes_For_Space_t cannot be half-forgotten in one of the seven call sites
	// below - a missed flag leaves the corresponding buffer unfilled and is read as garbage rather
	// than crashing.
	static inline void RequiredShapes_merge_space(const JITFuncSpec_RequiredShapes_For_Space_t &src, JITFuncSpec_RequiredShapes_For_Space_t &dest)
	{
		dest.psi |= src.psi;
		dest.dx_psi |= src.dx_psi;
		dest.dX_psi |= src.dX_psi;
		dest.d2x_psi |= src.d2x_psi;
		dest.d2X_psi |= src.d2X_psi;
		dest.dx_psi_dcoord |= src.dx_psi_dcoord;
	}

	void RequiredShapes_merge(JITFuncSpec_RequiredShapes_FiniteElement_t *src, JITFuncSpec_RequiredShapes_FiniteElement_t *dest)
	{
		for (unsigned int i = 0; i < NUM_CONTINUOUS_SPACES; i++)
		{
			RequiredShapes_merge_space(src->continuous_spaces[i], dest->continuous_spaces[i]);
		}
		RequiredShapes_merge_space(src->DL, dest->DL);
		RequiredShapes_merge_space(src->D0, dest->D0);
		RequiredShapes_merge_space(src->Pos, dest->Pos);

		dest->normal |= src->normal;
		dest->normal_deriv |= src->normal_deriv;
		dest->elemsize_Eulerian |= src->elemsize_Eulerian;
		dest->elemsize_Lagrangian |= src->elemsize_Lagrangian;
		dest->elemsize_Eulerian_cartesian |= src->elemsize_Eulerian_cartesian;
		dest->elemsize_Lagrangian_cartesian |= src->elemsize_Lagrangian_cartesian;		

		dest->history_integral_dx1 |= src->history_integral_dx1;
		dest->history_integral_dx2 |= src->history_integral_dx2;
		dest->history_geometry1 |= src->history_geometry1;
		dest->history_geometry2 |= src->history_geometry2;

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
	DynamicJITCode::DynamicJITCode(Problem *prob, CCompiler *ccompiler, std::string fnam, FiniteElementCode *cg, pyoomph::Mesh *bm) : problem(prob), compiler(ccompiler), filename(fnam), functable(NULL), code_gen(cg), so_handle(NULL), bulkmesh(bm)
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

		// Set (and, in the destructor, free) domain_name from the host side rather than in
		// generated code: codegen.cpp deliberately no longer bakes this in, since doing so made
		// otherwise textually-identical generated code (e.g. the same DirichletBC on two
		// different domains, such as "left" and "right" of a rectangle) differ only by this one
		// string, defeating the JIT code cache's ability to share a single compiled .so between
		// them. Allocating AND freeing it here (both via the host's own malloc/free, never
		// touching whatever C runtime the generated code itself was compiled/linked against)
		// sidesteps any cross-DLL-heap mismatch between the compiler backend's runtime and
		// pyoomph_core's own - which would otherwise be a real hazard on Windows, where a DLL's
		// static CRT can have its own private heap distinct from the calling process's.
		{
			std::string dn = cg->get_domain_name();
			functable->domain_name = (char *)malloc(sizeof(char) * (dn.size() + 1));
			memcpy(functable->domain_name, dn.c_str(), dn.size() + 1);
		}

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
		// The assembled contributions are merged separately as well; see the field's declaration.
		JITFuncSpec_RequiredShapes_FiniteElement *assembly = &(functable->assembly_required_shapes);

		for (unsigned int i = 0; i < functable->num_res_jacs; i++)
		{
			RequiredShapes_merge(&functable->shapes_required_ResJac[i], merged);
			RequiredShapes_merge(&functable->shapes_required_Hessian[i], merged);
			RequiredShapes_merge(&functable->shapes_required_ResJac[i], assembly);
			RequiredShapes_merge(&functable->shapes_required_Hessian[i], assembly);
		}
		RequiredShapes_merge(&functable->shapes_required_IntegralExprs, merged);
		RequiredShapes_merge(&functable->shapes_required_LocalExprs, merged);
		RequiredShapes_merge(&functable->shapes_required_ExtremumExprs, merged);
		RequiredShapes_merge(&functable->shapes_required_Z2Fluxes, merged);
		RequiredShapes_merge(&functable->shapes_required_TracerAdvection, merged);

		functable->handle = so_handle;

		// Export the functions to call
		functable->invoke_callback = _pyoomph_invoke_callback;
		functable->invoke_multi_ret = _pyoomph_invoke_multi_ret;
		functable->fill_shape_buffer_for_point = _pyoomph_fill_shape_buffer_for_point;

		for (unsigned int i = 0; i < functable->numintegral_expressions; i++)
			integral_function_map[functable->integral_expressions_names[i]] = i;
		for (unsigned int i = 0; i < functable->numextremum_expressions; i++)
			extremum_function_map[functable->extremum_expressions_names[i]] = i;

		// Sized here rather than in the member initialiser list: the ED0 field count only
		// becomes known once initfunc() above has filled the function table.
		linked_external_data = ExternalDataLinkVector(functable->info_ED0.numfields);
	}

	// Every shared library currently loaded by ANY Problem in this process, mapped to the code that
	// loaded it. dlopen dedupes by inode and only bumps a refcount, so a second Problem that resolves
	// an element code to a path already loaded gets the FIRST Problem's compiled image handed back -
	// silently, and after its own compiler has already overwritten that file underneath the live
	// mapping. The observed symptom was a second Problem quietly computing the first one's equations;
	// on codes whose layout differs enough, the rewritten mapping instead takes the process down
	// inside dlsym. Neither is detectable from the return value of dlopen, so the collision is caught
	// here instead. Entries are removed by ~DynamicJITCode, so loading the same path again after the
	// first Problem has been destroyed is fine - which is what redefine_problem() relies on.
	static std::map<std::string, DynamicJITCode *> __loaded_jit_libraries;

	// Whether some Problem in this process currently has this shared library loaded. The Python side
	// asks before it compiles, so that a second Problem can be given its own code directory rather
	// than run into the error in load_jit_code below.
	bool jit_library_is_loaded(const std::string &fname)
	{
		return __loaded_jit_libraries.count(fname) > 0;
	}

	// Cleans up the function table (invoking the code's own clean_up callback first, if any) and closes
	// the dlopen handle of the compiled shared library.
	DynamicJITCode::~DynamicJITCode()
	{
		// std::cout << "UNLOADING ELEMENT CODE " << filename << " FUNCTABLE " << functable << " SO HANDLE " << so_handle  << std::endl << std::flush;
		// std::cout << "COMPILER  " << compiler  << std::endl << std::flush;

		if (functable)
		{
			if (pyoomph_verbose)
			{
				std::cout << "Cleaning memory of functable" << std::endl << std::flush;
			}
			// Host-allocated (see the constructor) - freed here with the host's own free(),
			// before clean_up()/close_handle() below ever touch the compiled library itself.
			if (functable->domain_name)
			{
				free(functable->domain_name);
				functable->domain_name = NULL;
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

		auto it = __loaded_jit_libraries.find(filename);
		if (it != __loaded_jit_libraries.end() && it->second == this)
			__loaded_jit_libraries.erase(it);
	}

	int DynamicJITCode::get_integral_function_index(std::string n)
	{
		if (!integral_function_map.count(n))
			return -1;
		return integral_function_map[n];
	}

	int DynamicJITCode::get_extremum_function_index(std::string n)
	{
		if (!extremum_function_map.count(n))
			return -1;
		return extremum_function_map[n];
	}

	// Looks up the residual/Jacobian contribution named "name" in this code and, if found and non-NULL,
	// makes it the active one (get_current_res_jac(functable)) for subsequent assembly calls into this code.
	// Returns 1 on success, 0 if no matching (non-NULL) contribution exists.
	unsigned DynamicJITCode::_set_solved_residual(std::string name)
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
		set_current_res_jac(functable, res_jac_index);
		if (res_jac_index >= 0)
			return 1;
		else
			return 0;
	}

	//////////////////////////////////////////

	// Rebuilds the deduplicated elemental_data list from the (name-indexed) link entries: for each link,
	// finds or inserts its Data pointer in elemental_data and records the resulting position as the
	// link's elemental_index (the index the compiled element code uses to address this external data).
	// Only links that some contribution actually assembles are registered - see the header for why the
	// output-only ones are left out. The two passes matter: a Data object shared between an
	// output-only field and an assembled one is registered by the latter, and the former then simply
	// finds it and addresses it through the element as before.
	void ExternalDataLinkVector::reindex_elemental_data(const int *field_contribution_index)
	{
		// Same lever as the bulk/opposite attachment split, so the whole stage can be A/B-ed on one binary.
		static const bool __disabled = getenv("PYOOMPH_DISABLE_ASSEMBLY_EXTDATA_SPLIT") != NULL;
		elemental_data.clear();
		for (unsigned int i = 0; i < this->size(); i++)
		{
			if (!__disabled && field_contribution_index && field_contribution_index[i] == -2)
				continue;
			if (std::find(elemental_data.begin(), elemental_data.end(), this->at(i).data) == elemental_data.end())
				elemental_data.push_back(this->at(i).data);
		}
		for (unsigned int i = 0; i < this->size(); i++)
		{
			auto found = std::find(elemental_data.begin(), elemental_data.end(), this->at(i).data);
			this->at(i).elemental_index = (found == elemental_data.end() ? -1 : found - elemental_data.begin());
		}
	}

	///////////////////////////////////////////

	// Binds the external field named "name" (as declared by the compiled code's ED0 space) to a specific
	// (Data, index) pair, and patches the code's contribution name for that field (used in diagnostic
	// output such as get_jacobian_information_string()) to full_source_name so it reflects where the
	// linked value actually comes from. Throws if "name" is not among the code's required external fields.
	void DynamicJITCode::link_external_data(std::string name, oomph::Data *data, int index,std::string full_source_name)
	{
		int found = -1;
		for (unsigned int i = 0; i < functable->info_ED0.numfields; i++)
		{
			if (name == std::string(functable->info_ED0.fieldnames[i]))
			{
				found = i;
				break;
			}
		}
		if (found == -1)
			throw_runtime_error("Cannot link external data '" + name + "' since this is not required by the equation code");
		linked_external_data[found] = ExternalDataLink(data, index);
		// Replace the residual and jacobian information as well
		std::string look_for=this->get_code_gen()->get_full_domain_name()+"/"+name;
		for (unsigned int i = 0; i < functable->contribution_entries_size; i++)
		{
			std::string res_name = functable->contribution_names[i];
			if (res_name == look_for)
			{
				free((char*)functable->contribution_names[i]);
				functable->contribution_names[i] = strdup(full_source_name.c_str());
				break;
			}
		}
		linked_external_data.reindex_elemental_data(functable->info_ED0.field_contribution_index);
	}

	// Maps every nodal field name to its index into the node's value array, in the order in which the
	// element code lays out nodal fields: first the "base bulk" fields of all present continuous spaces,
	// then the base-bulk fields of all present DG spaces, then the remaining (non-base-bulk, i.e.
	// interface-only) fields of the continuous spaces, then the remaining fields of the DG spaces.
	std::map<std::string, unsigned> DynamicJITCode::get_nodal_field_indices()
	{
		std::map<std::string, unsigned> res;
		unsigned offs = 0;
		for (unsigned int si=0;si<functable->num_present_continuous_spaces;si++)
		{
			auto *space = functable->present_continuous_spaces[si];
			for (unsigned int i = 0; i < space->numfields_basebulk; i++)
			{
				res[space->fieldnames[i]] = offs + i;
			}
			offs += space->numfields_basebulk;
		}		


		for (unsigned int si=0;si<functable->num_present_dg_spaces;si++)
		{
			auto *space = functable->present_dg_spaces[si];
			for (unsigned int i = 0; i < space->numfields_basebulk; i++)
			{
				res[space->fieldnames[i]] = offs + i;
			}
			offs += space->numfields_basebulk;
		}


		// Now the additional ones
		for (unsigned int si=0;si<functable->num_present_continuous_spaces;si++)
		{
			auto *space = functable->present_continuous_spaces[si];
			for (unsigned int i = 0; i < space->numfields - space->numfields_basebulk; i++)
			{
				res[space->fieldnames[i + space->numfields_basebulk]] = offs + i;
			}
			offs += space->numfields - space->numfields_basebulk;
		}

		for (unsigned int si=0;si<functable->num_present_dg_spaces;si++)
		{
			auto *space = functable->present_dg_spaces[si];
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
	std::map<std::string, unsigned> DynamicJITCode::get_elemental_field_indices()
	{
		std::map<std::string, unsigned> res;
		for (unsigned int i = 0; i < functable->info_DL.numfields; i++)
		{
			res[functable->info_DL.fieldnames[i]] = i;
		}
		for (unsigned int i = 0; i < functable->info_D0.numfields; i++)
		{
			res[functable->info_D0.fieldnames[i]] = i + functable->info_DL.numfields;
		}
		return res;
	}

	// Index of a DL or D0 (discontinuous/elemental) field by name, offset by internal_offset_new so it
	// directly indexes into the element's internal Data array; -1 if not a discontinuous field.
	int DynamicJITCode::get_discontinuous_field_index(std::string name)
	{
		for (unsigned int i = 0; i < functable->info_DL.numfields; i++)
		{
			if (!strcmp(name.c_str(), functable->info_DL.fieldnames[i]))
			{
				return i + functable->info_DL.internal_offset_new;
			}
		}
		for (unsigned int i = 0; i < functable->info_D0.numfields; i++)
		{
			if (!strcmp(name.c_str(), functable->info_D0.fieldnames[i]))
			{
				return i + functable->info_D0.internal_offset_new;
			}
		}
		return -1;
	}

	// Index of a base-bulk nodal field by name (offset by the space's nodal_offset_basebulk into the
	// node's value array); -1 if "name" is not a base-bulk field of any present continuous space.
	int DynamicJITCode::get_nodal_field_index(std::string name)
	{
		for (unsigned int si=0;si<functable->num_present_continuous_spaces;si++)
		{
			auto *space = functable->present_continuous_spaces[si];
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
	unsigned DynamicJITCode::resolve_interface_dof_id(std::string n)
	{
		//std::cout << "->Resolving interface dof for field " << n << " on mesh " <<  this->get_bulk_mesh() << std::endl;
		return this->get_bulk_mesh()->resolve_interface_dof_id(n);
	}

	// For every present continuous space's interface-only fields (the fields beyond numfields_basebulk,
	// i.e. those that only live on interface nodes of a C2TB-C1-type space), resolves and stores their
	// interface dof id in the function table's interface_dof_indices array so that interface elements can
	// find the additional dof slot on shared nodes. Required whenever this instance participates in an
	// interface coupling. Returns the resolved field-name -> dof-id map for convenience.
	std::map<std::string, unsigned> DynamicJITCode::setup_interface_dof_indices()
	{
		std::map<std::string, unsigned> res;		

		for (unsigned int i = 0; i < functable->num_present_continuous_spaces; i++	)
		{
			JITFuncSpec_Table_FiniteElement_SpaceInfo_t * space_info= functable->present_continuous_spaces[i];

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
	std::string DynamicJITCode::get_space_of_field(std::string name)
	{
		for (unsigned int si=0;si<functable->num_present_continuous_spaces;si++)
		{
			auto *space = functable->present_continuous_spaces[si];
			for (unsigned int i = 0; i < space->numfields_basebulk; i++)
			{
				if (!strcmp(name.c_str(), space->fieldnames[i]))
				{
					return space->space_name;
				}
			}
		}

		for (unsigned int si=0;si<functable->num_present_dg_spaces;si++)
		{
			auto *space = functable->present_dg_spaces[si];
			for (unsigned int i = 0; i < space->numfields_basebulk; i++)
			{
				if (!strcmp(name.c_str(), space->fieldnames[i]))
				{
					return space->space_name;
				}
			}
		}
		
		for (unsigned int i = 0; i < functable->info_DL.numfields; i++)
		{
			if (!strcmp(name.c_str(), functable->info_DL.fieldnames[i]))
			{
				return "DL";
			}
		}
		for (unsigned int i = 0; i < functable->info_D0.numfields; i++)
		{
			if (!strcmp(name.c_str(), functable->info_D0.fieldnames[i]))
			{
				return "D0";
			}
		}
		return "";
	}

	// Placeholder for consistency checks of the instance's field/parameter binding; currently a no-op,
	// the binding checks it used to perform are now handled elsewhere (kept here, commented out, for reference)
	void DynamicJITCode::sanity_check()
	{
		/*
		 for (unsigned int i=0;i<functable->numglobal_params;i++)
		 {
			if (local_global_parameter_to_global_index[i]<0) throw_runtime_error("Elemental parameter "+std::string(functable->global_paramnames[i])+" not bound");
		 }
		*/
		/*
		 for (unsigned int i=0;i<functable->numfields_C2;i++)
		 {
			if (local_field_to_global_field_index_C2[i]<0) throw_runtime_error("C2 field "+std::string(functable->fieldnames_C2[i])+" not bound");
		 }
		 for (unsigned int i=0;i<functable->numfields_C1;i++)
		 {
			if (local_field_to_global_field_index_C1[i]<0) throw_runtime_error("C1 field "+std::string(functable->fieldnames_C1[i])+" not bound");
		 }
		*/
	}

    // Whether this instance's compiled code has any residual/Jacobian contribution that depends on the
    // named global parameter (i.e. the parameter is among the code's registered global params).
    bool DynamicJITCode::has_parameter_contribution(const std::string &param)
	{
		if (!this->get_problem()->has_global_parameter(param))
			return false;
		pyoomph::GlobalParameterDescriptor * parameter=this->get_problem()->get_global_parameter(param);
		for (unsigned int i = 0; i < functable->numglobal_params; i++)
		{
			if (functable->global_paramindices[i] == parameter->get_global_index())
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
		for (unsigned int i = 0; i < jit_codes.size(); i++)
		{
			max_order = std::max(max_order, (unsigned)jit_codes[i]->functable->max_dt_order);
		}
		return max_order;
	}

	// Deletes all loaded DynamicJITCode objects (closing their shared libraries) and all global
	// parameter descriptors, and releases the cached eigenproblem matrices. Used both from the destructor
	// and explicitly (e.g. before recompiling/reloading equation code).
	void Problem::unload_all_dlls(bool clear_all)
	{		
		// A loop over all elements used to sit here, cross-casting each one to the JIT code class and
		// clearing its external data links. BulkElementBase and the JIT code class are unrelated
		// types, so that dynamic_cast always yielded NULL and the clear() never ran once - removed
		// rather than "fixed", since nothing has ever depended on it happening.
		if (clear_all)
		{
			for (unsigned int i=0;i< this->nsub_mesh();i++)
			{
				Mesh *m=dynamic_cast<Mesh *>(this->mesh_pt(i));
				// Kill the tree forest (if any) first, while the compiled element code is still
				// loaded: Tree::~Tree() destroys the "father" (non-leaf, already-refined-away)
				// elements it still owns, which are no longer reachable via element_pt() below -
				// see TemplatedMeshBase::_kill_tree_forest_now() for why this is safe to do before
				// the node/element deletion loops (leaf elements are explicitly left untouched).
				TemplatedMeshBase *tbm=dynamic_cast<TemplatedMeshBase *>(m);
				if (tbm) tbm->_kill_tree_forest_now();
				for (unsigned int j=m->nnode();j>0;j--)
				{
					delete m->node_pt(j-1);
					m->node_pt(j-1)=NULL;
				}
				for (unsigned int j=m->nelement();j>0;j--)
				{
					BulkElementBase *e=dynamic_cast<BulkElementBase *>(m->element_pt(j-1));
					delete e;
					m->element_pt(j-1)=NULL;
				}
				m->flush_element_and_node_storage();
			}
		}
		if (pyoomph_verbose)
			std::cout << "Unloading all DLLs" << std::endl
					  << std::flush;
		for (unsigned int i = 0; i < jit_codes.size(); i++)
		{
			if (pyoomph_verbose)
				std::cout << "Unloading DLL " << jit_codes[i]->get_file_name() << std::endl
						  << std::flush;
			delete jit_codes[i];
		}
		if (pyoomph_verbose)
			std::cout << "DLLs unloaded " << std::endl
					  << std::flush;
		for (auto &gp : global_params_by_name)
		{

			delete gp.second;
		}

		jit_codes.clear();

		global_params_by_name.clear();

		// unload_all_dlls() can run twice (once explicitly, e.g. from Problem.release(), then
		// again from ~Problem()) - nulling these after delete makes that safe; the two blocks
		// above are already idempotent (delete-and-clear a possibly-already-empty container).
		if (eigen_MassMatrixPt)
		{
			delete eigen_MassMatrixPt;
			eigen_MassMatrixPt = NULL;
		}
		if (eigen_JacobianMatrixPt)
		{
			delete eigen_JacobianMatrixPt;
			eigen_JacobianMatrixPt = NULL;
		}
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

	Problem::Problem() : oomph::Problem(), compiler(NULL), logfile(NULL), _is_quiet(false), jit_codes(0) // , meshtemplate(new MeshTemplate(this))
	{
	}

	// Loads the shared library dynamic_lib as a new DynamicJITCode bound to bulkmesh, associates it
	// with code_gen (which fills in the callback function pointers of the function table), and binds
	// each of the code's declared global parameters to the corresponding GlobalParameterDescriptor's
	// value in this problem (creating parameters as needed).
	DynamicJITCode *Problem::load_jit_code(std::string dynamic_lib, FiniteElementCode *code_gen, pyoomph::Mesh *bulkmesh)
	{
		// This used to silently hand back the already-loaded code for a repeated file name, on the
		// premise that one compiled library could serve several domains. It cannot: the returned
		// object kept the *first* compilation's code generator, which redefine_problem() has by then
		// replaced, and it now also carries the first compilation's bulk mesh. Every domain and
		// interface compiles to its own uniquely named trunk, so a collision here means two of them
		// were written to the same path - report it instead of corrupting one of them.
		//
		// Checked against every library loaded in the PROCESS, not just this Problem's own jit_codes:
		// the damage dlopen's inode dedupe does (see __loaded_jit_libraries) is worst precisely when
		// the two codes belong to different Problems, because nothing else in either of them would
		// ever notice.
		auto already = __loaded_jit_libraries.find(dynamic_lib);
		if (already != __loaded_jit_libraries.end())
		{
			const bool other_problem = already->second->get_problem() != this;
			throw_runtime_error("Two element codes resolved to the same shared library '" + dynamic_lib + "'." +
								(other_problem
									 ? std::string(" They belong to two different Problems that are alive at the same time. Give each Problem its own output directory (set_output_directory) or its own code directory, so that their generated code does not share a file name.")
									 : std::string(" If this happened during redefine_problem(), pass a code_dir that differs from the current one.")));
		}
		CCompiler *ccompiler = this->get_ccompiler();
		jit_codes.push_back(new DynamicJITCode(this, ccompiler, dynamic_lib, code_gen, bulkmesh));
		__loaded_jit_libraries[dynamic_lib] = jit_codes.back();
		code_gen->fill_callback_info(jit_codes.back()->functable);
		auto *ft = jit_codes.back()->functable;
		for (unsigned int i = 0; i < ft->numglobal_params; i++)
		{
			//		std::cout << "LINKING GLOBAL PARAM " << i << " of " << functable->numglobal_params << std::endl;
			//		std::cout << "jitcode->get_problem()->get_global_parameter(functable->global_paramindices[i]) << std::endl;
			ft->global_parameters[i] = &(this->get_global_parameter(ft->global_paramindices[i])->value());
		}

		return jit_codes.back();
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
		for (unsigned int i = 0; i < jit_codes.size(); i++)
		{
			numfound += jit_codes[i]->_set_solved_residual(name);
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
		// The dynamic_cast is done once, in the first pass, and the survivors are kept for the second:
		// this runs over every element of every submesh several times per initialisation (and again
		// after every adapt), so the second pass's casts were pure repetition.
		std::vector<BulkElementBase *> bulk_elements;
		for (unsigned nmi = 0; nmi < this->nsub_mesh(); nmi++)
		{
			oomph::Mesh *const msh = mesh_pt(nmi);
			unsigned nelem = msh->nelement();
			bulk_elements.reserve(bulk_elements.size() + nelem);
			for (unsigned n = 0; n < nelem; n++)
			{
				auto el = dynamic_cast<BulkElementBase *>(msh->element_pt(n));
				if (el)
				{
					el->unpin_dummy_values();
					bulk_elements.push_back(el);
				}
			}
		}
		for (BulkElementBase *el : bulk_elements)
		{
			el->pin_dummy_values();
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

	// Builds the permutation for the active layout. Empty mode, or anything this build does not
	// recognise, means "no opinion" and the numbering oomph assigned stands.
	//
	// "reverse" is a TEST layout and nothing else: it is the cheapest permutation that is a genuine
	// bijection and moves essentially every dof, which is exactly what is wanted to prove that the
	// renumbering machinery is transparent to the answer. The layouts that are worth having
	// (nodal-block for AMG, element-block for condensation) are built on Mesh::visit_global_dofs and
	// come next; this one stays as the null hypothesis they are tested against.
	// Defined further down, next to condensation_row_cuts, the other caller that wants a block-aligned
	// row split.
	static std::vector<unsigned> snap_cuts_to_blocks(const std::vector<std::pair<int, int>> &blocks,
													 unsigned nproc, unsigned nd);

	// Glob match, '*' and '?' only, as used on the field names of a dof ordering spec. Written out
	// rather than taken from <fnmatch.h> because that header is POSIX and pyoomph builds on Windows.
	static bool dof_ordering_glob(const char *pat, const char *str)
	{
		const char *star = NULL, *ss = str;
		while (*str)
		{
			if (*pat == '?' || *pat == *str) { pat++; str++; }
			else if (*pat == '*') { star = pat++; ss = str; }
			else if (star) { pat = star + 1; str = ++ss; }
			else return false;
		}
		while (*pat == '*') pat++;
		return !*pat;
	}

	// Builds the permutation for the active layout. Empty mode and no specs means "no opinion", and
	// the numbering oomph assigned stands.
	//
	// "reverse" is a TEST layout and nothing else: the cheapest permutation that is a genuine bijection
	// and moves essentially every dof, which is what is wanted to prove that the renumbering machinery
	// is transparent to the answer. It is the null hypothesis the real layouts are tested against.
	bool Problem::build_dof_permutation(std::vector<long> &perm, unsigned long n)
	{
		if (dof_ordering_mode == "reverse")
		{
			perm.resize(n);
			for (unsigned long i = 0; i < n; i++) perm[i] = (long)(n - 1 - i);
			return true;
		}
		if (dof_ordering_mode != "")
			throw_runtime_error("Unknown dof ordering mode '" + dof_ordering_mode + "'");
		if (dof_ordering_specs.empty()) return false;
		return build_dof_permutation_from_specs(perm, n);
	}

	// Turns the user's field patterns into a permutation.
	//
	// The vocabulary is get_global_field_names() -- "domain/velocity_x", "domain/top/lambda",
	// "domain/coordinate_x" -- because that is the one name for a field that already exists in C++,
	// already covers interface-only fields and nodal positions, and is what petsc_fieldsplit takes. It
	// deliberately does NOT distinguish a field's boundary-restricted dofs from its bulk ones
	// ("domain/left/u" and "domain/u" are both the field "domain/u"): for a layout that is the right
	// granularity, since a block wants a node's fields together whether or not the node is on a
	// boundary.
	bool Problem::build_dof_permutation_from_specs(std::vector<long> &perm, unsigned long n)
	{
		const std::vector<std::string> fnames = this->get_global_field_names();
		const unsigned nf = (unsigned)fnames.size();
		if (!nf) return false;

		// Resolve the patterns once per field, not once per dof. First spec that names a field wins,
		// and within a spec the pattern's position is the field's rank inside a block.
		const int nspec = (int)dof_ordering_specs.size();
		std::vector<int> spec_of_field(nf, -1), rank_of_field(nf, 0);
		std::vector<char> pattern_used(0);
		for (int s = 0; s < nspec; s++)
		{
			const DofOrderingSpec &sp = dof_ordering_specs[s];
			pattern_used.assign(sp.patterns.size(), 0);
			for (unsigned f = 0; f < nf; f++)
			{
				if (spec_of_field[f] >= 0) continue;
				for (unsigned q = 0; q < sp.patterns.size(); q++)
				{
					if (!dof_ordering_glob(sp.patterns[q].c_str(), fnames[f].c_str())) continue;
					spec_of_field[f] = s;
					rank_of_field[f] = (int)q;
					pattern_used[q] = 1;
					break;
				}
			}
			for (unsigned q = 0; q < sp.patterns.size(); q++)
			{
				// A pattern that names nothing is a typo in a field name, and silently ignoring it gives
				// the user a layout that is not the one they asked for while reporting success. The
				// available names are listed because they are not guessable ("domain/coordinate_x" for a
				// mesh position, and a field of an interface carries that interface's full path).
				if (pattern_used[q]) continue;
				std::string avail;
				for (unsigned f = 0; f < nf; f++) avail += (f ? ", " : "") + fnames[f];
				throw_runtime_error("The dof ordering pattern '" + sp.patterns[q] +
									"' matches none of this problem's fields, or only fields already "
									"claimed by an earlier ordering. Available: " + avail);
			}
		}

		// NOT get_dof_to_global_field_index_mapping(): that one sizes its buffer with ndof(), which reads
		// Dof_distribution_pt->nrow() -- and the distribution is rebuilt BELOW this hook, so inside it
		// ndof() still reports the previous numbering's size (zero on the first call). Sizing by n, the
		// length of the dof pointer vector being handed to us, is the only correct thing here. Getting
		// this wrong is silent: the buffer simply came out the wrong length and the layout declined.
		std::vector<int> field_of_dof(n, -1);
		for (unsigned im = 0; im < this->nsub_mesh(); im++)
		{
			pyoomph::Mesh *m = dynamic_cast<pyoomph::Mesh *>(this->mesh_pt(im));
			if (m) m->fill_dof_to_global_field_index_buffer(field_of_dof);
		}

		// Per dof: which spec claims it, and which group of that spec it belongs to. The group is
		// identified by a POINTER (the node, or the element that claimed it first) and then collapsed to
		// the smallest equation number in it, so that groups keep the relative order the mesh gave them
		// and a layout that is already satisfied comes out as the identity.
		std::vector<int> spec_of_dof(n, -1), rank_of_dof(n, 0);
		std::vector<const void *> group_of_dof(n, NULL);
		for (unsigned long e = 0; e < n; e++)
		{
			const int f = field_of_dof[e];
			if (f < 0 || (unsigned)f >= nf) continue;
			spec_of_dof[e] = spec_of_field[f];
			rank_of_dof[e] = rank_of_field[f];
		}

		for (unsigned im = 0; im < this->nsub_mesh(); im++)
		{
			pyoomph::Mesh *m = dynamic_cast<pyoomph::Mesh *>(this->mesh_pt(im));
			if (!m) continue;
			m->visit_global_dofs([&](const Mesh::DofVisit &v)
								 {
									 if (v.eqn < 0 || (unsigned long)v.eqn >= n) return;
									 const int s = spec_of_dof[v.eqn];
									 if (s < 0) return;
									 if (group_of_dof[v.eqn]) return; // first claim wins
									 // A node's dofs group by the node, so its positions and its values
									 // land in one block even though they live in different Data. An
									 // element-local one groups by the element that reported it first,
									 // which for a cell-interior bubble node is the only element there
									 // is - that is what makes a Crouzeix-Raviart block contiguous.
									 group_of_dof[v.eqn] = dof_ordering_specs[s].by_element
															   ? (const void *)v.element
															   : (v.node ? (const void *)v.node : (const void *)v.element);
								 });
		}

		std::map<const void *, long> group_min;
		for (unsigned long e = 0; e < n; e++)
		{
			if (spec_of_dof[e] < 0) continue;
			if (!group_of_dof[e]) { spec_of_dof[e] = -1; continue; } // named, but no mesh reported it
			std::map<const void *, long>::iterator it = group_min.find(group_of_dof[e]);
			if (it == group_min.end()) group_min[group_of_dof[e]] = (long)e;
			else if ((long)e < it->second) it->second = (long)e;
		}

		// The sort key. Unclaimed dofs trail, in their original order: a layout says where the dofs it
		// names go, and inventing a position for the others would make adding one pattern move
		// everything.
		std::vector<unsigned long> order(n);
		for (unsigned long e = 0; e < n; e++) order[e] = e;
		std::vector<long> gmin(n, 0);
		for (unsigned long e = 0; e < n; e++)
			if (spec_of_dof[e] >= 0) gmin[e] = group_min[group_of_dof[e]];
		std::stable_sort(order.begin(), order.end(),
						 [&](unsigned long a, unsigned long b)
						 {
							 const int sa = spec_of_dof[a] < 0 ? nspec : spec_of_dof[a];
							 const int sb = spec_of_dof[b] < 0 ? nspec : spec_of_dof[b];
							 if (sa != sb) return sa < sb;
							 if (sa == nspec) return a < b;
							 if (gmin[a] != gmin[b]) return gmin[a] < gmin[b];
							 if (rank_of_dof[a] != rank_of_dof[b]) return rank_of_dof[a] < rank_of_dof[b];
							 return a < b;
						 });

		perm.assign(n, 0);
		for (unsigned long i = 0; i < n; i++) perm[order[i]] = (long)i;

		// Record the blocks in the numbering just produced. A group's dofs are contiguous there by
		// construction (the sort key is group-major), so one pass over the new order finds the runs.
		// Singletons are skipped: a block of one cannot be cut through.
		dof_ordering_blocks.clear();
		unsigned long i0 = 0;
		while (i0 < n)
		{
			const unsigned long e0 = order[i0];
			unsigned long i1 = i0 + 1;
			if (spec_of_dof[e0] >= 0)
				while (i1 < n && spec_of_dof[order[i1]] == spec_of_dof[e0] &&
					   group_of_dof[order[i1]] == group_of_dof[e0])
					i1++;
			if (i1 - i0 > 1) dof_ordering_blocks.push_back(std::make_pair((int)i0, (int)(i1 - 1)));
			i0 = i1;
		}
		return true;
	}

	// A row split whose cut points fall between the layout's blocks rather than inside them.
	//
	// Only a REPLICATED run needs this. Distributed, each rank's dofs are one contiguous global range
	// by construction (synchronise_eqn_numbers) and the permutation was rank-local, so no block can
	// straddle a rank; the caller hands back the dof distribution there instead.
	std::vector<unsigned> Problem::dof_ordering_row_cuts()
	{
		if (dof_ordering_blocks.empty()) return std::vector<unsigned>();
		if (!Communicator_pt || Communicator_pt->nproc() < 2) return std::vector<unsigned>();
#ifdef OOMPH_HAS_MPI
		// Problem_has_been_distributed is an MPI-only member of oomph::Problem; in a build without
		// MPI there is nothing to distribute, so the replicated path below is the only one anyway.
		if (Problem_has_been_distributed) return std::vector<unsigned>();
#endif
		const unsigned nd = this->ndof();
		if (nd < static_cast<unsigned>(Communicator_pt->nproc())) return std::vector<unsigned>();
		return snap_cuts_to_blocks(dof_ordering_blocks, Communicator_pt->nproc(), nd);
	}

	// Rewrites every equation number this problem owns through perm, and permutes Dof_pt to match.
	//
	// The Data enumeration mirrors synchronise_eqn_numbers()' bump loops rather than the field-aware
	// walk of Mesh::visit_global_dofs, and deliberately so: this has to move EVERY dof, including any
	// no field description reaches, or the numbering would end up self-inconsistent. Same reason it
	// iterates Data::nvalue() rather than a field count.
	void Problem::apply_dof_permutation(const std::vector<long> &perm, oomph::Vector<double *> &dof_pt)
	{
		const unsigned long n = dof_pt.size();
		{
			// NOT under PARANOID, which is off in every normal build (PYOOMPH_PARANOID defaults to OFF).
			// A non-bijective perm silently aliases two dofs onto one row and surfaces much later as a
			// Jacobian that is singular for no visible reason -- precisely the bug a layout under
			// development produces, so the check has to be present in the build people actually use.
			// One O(n) pass and an n-byte array, against a permutation that is already O(n log n) inside
			// an assign_eqn_numbers() whose elemental pass is ~96% of the total; it does not register.
			std::vector<char> seen(n, 0);
			for (unsigned long i = 0; i < n; i++)
			{
				if (perm[i] < 0 || (unsigned long)perm[i] >= n)
					throw_runtime_error("The dof permutation for mode '" + dof_ordering_mode + "' maps " +
										std::to_string(i) + " to " + std::to_string(perm[i]) +
										", which is outside [0," + std::to_string(n) + ").");
				if (seen[perm[i]])
					throw_runtime_error("The dof permutation for mode '" + dof_ordering_mode +
										"' is not a bijection: two dofs map to " + std::to_string(perm[i]) + ".");
				seen[perm[i]] = 1;
			}
		}

		oomph::Vector<double *> permuted(n, NULL);
		for (unsigned long i = 0; i < n; i++) permuted[perm[i]] = dof_pt[i];
		dof_pt = permuted;

		auto remap = [&perm, n](oomph::Data *d)
		{
			const unsigned nval = d->nvalue();
			for (unsigned v = 0; v < nval; v++)
			{
				const long eq = d->eqn_number(v);
				if (eq < 0) continue; // pinned, constrained, halo: not a dof, not in perm
				if ((unsigned long)eq >= n) continue;
				d->eqn_number(v) = perm[eq];
			}
		};

		for (unsigned i = 0; i < this->nglobal_data(); i++) remap(this->global_data_pt(i));

		oomph::Mesh *const gm = this->mesh_pt();
		if (!gm) return;

		const unsigned long nel = gm->nelement();
		for (unsigned long e = 0; e < nel; e++)
		{
			oomph::GeneralisedElement *el = gm->element_pt(e);
			const unsigned nint = el->ninternal_data();
			for (unsigned i = 0; i < nint; i++) remap(el->internal_data_pt(i));
		}

		const unsigned long nnod = gm->nnode();
		for (unsigned long j = 0; j < nnod; j++)
		{
			oomph::Node *nod = gm->node_pt(j);
			// A periodic ("copy") node does not own its equation numbers: make_periodic() points its
			// Eqn_number array at the master's, and master and copy are both in Node_pt. Remapping both
			// would apply perm twice to the same long, i.e. perm[perm[eq]]. Same guard, and the same
			// reason, as the bump loops in synchronise_eqn_numbers().
			if (!nod->is_a_copy()) remap(nod);
			oomph::SolidNode *snod = dynamic_cast<oomph::SolidNode *>(nod);
			if (snod && !snod->position_is_a_copy()) remap(snod->variable_position_pt());
		}
	}

	// See the base declaration in oomph-lib's problem.h for where this is called from and why there.
	void Problem::reorder_global_eqn_numbers(oomph::Vector<double *> &dof_pt)
	{
		// The overwhelmingly common case: no layout asked for, and not even a permutation buffer touched.
		dof_ordering_blocks.clear();
		if (dof_ordering_mode == "" && dof_ordering_specs.empty()) return;
		const unsigned long n = dof_pt.size();
		if (!n) return;
		if (!build_dof_permutation(dof_permutation_buffer, n)) return;
		apply_dof_permutation(dof_permutation_buffer, dof_pt);
	}

	// Performs the standard oomph-lib equation numbering, then lets every InterfaceMesh update its
	// equation remapping (interface dofs referencing bulk equation numbers that may have just changed),
	// and finally, when Dirichlet conditions are enforced by matrix manipulation rather than dof removal,
	// rebuilds the set of globally pinned equation numbers (which depends on the numbering just assigned).
	unsigned long Problem::assign_eqn_numbers(const bool& assign_local_eqn_numbers)
	{
      // Tempting and wrong: this cannot skip the elemental pass when the global numbering comes out
      // unchanged. Mesh::assign_local_eqn_numbers() reaches BulkElementBase::
      // assign_additional_local_eqn_numbers(), which also runs fill_element_info() - the rebuild of the
      // eleminfo struct that the generated code reads (value/coordinate pointers, hang info with its
      // equation numbers). That is not a function of the numbering alone: the interface equation
      // remapping below is computed AFTER it, which is exactly why several call sites here call
      // reapply_boundary_conditions() twice in a row. A fingerprint-guarded skip (equation numbers +
      // Data/element addresses + topology generation, all identical on the second call) therefore left
      // the element info stale and adaptively refined interface problems diverged with an infinite
      // residual. Measured first: the elemental pass is ~96% of an assign_eqn_numbers(). See
      // dev_docs/initialisation_cost.md.
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
	  // The Jacobian sparsity pattern is a function of the equation numbering, which has just been
	  // (re)assigned. This is the central invalidation point: mesh adaptation, remeshing, pinning and
	  // Dirichlet changes all funnel through here. assign_eqn_numbers() is collective, so every MPI
	  // rank bumps the id in lockstep.
	  this->invalidate_jacobian_structure();
#if defined(OOMPH_HAS_MPI) && defined(PARANOID)
	  // Ranks that disagree about the pattern id would disagree about whether to rebuild the matrix,
	  // and rebuilding it is itself collective - so a split decision deadlocks rather than producing a
	  // wrong answer. Check it here, in a place that is already collective, rather than inside
	  // get_jacobian_structure_id(): that one is called from Python and may well be read on a single
	  // rank, where a collective call would be the deadlock it is meant to prevent.
	  if (this->communicator_pt() && this->communicator_pt()->nproc() > 1)
	  {
		  unsigned long mine = this->get_jacobian_structure_id(), lo = 0, hi = 0;
		  MPI_Allreduce(&mine, &lo, 1, MPI_UNSIGNED_LONG, MPI_MIN, this->communicator_pt()->mpi_comm());
		  MPI_Allreduce(&mine, &hi, 1, MPI_UNSIGNED_LONG, MPI_MAX, this->communicator_pt()->mpi_comm());
		  if (lo != hi)
		  {
			  throw_runtime_error("The Jacobian sparsity pattern id diverged across MPI ranks (min " +
								  std::to_string(lo) + ", max " + std::to_string(hi) + "). Some rank "
								  "invalidated the pattern without the others doing so, which would let "
								  "the ranks disagree about whether to rebuild the linear system.");
		  }
	  }
#endif
	  return res;
	}

	// After mesh adaptation: announces that cached element pointers are stale (refinement replaces a
	// leaf by its sons, unrefinement deletes them), re-establishes the dummy-value pinning invariant,
	// and re-runs problem-specific pinning setup.
	//
	// Note that pyoomph's Python Problem overrides this WITHOUT calling back into it, so for anything
	// driven from Python - which is everything - this body does not run. The generation bump is
	// therefore repeated at the top of that override; if you add something here, add it there too.
	void Problem::actions_after_adapt()
	{
		for (unsigned nmi = 0; nmi < this->nsub_mesh(); nmi++)
		{
			if (dynamic_cast<Mesh *>(this->mesh_pt(nmi)))
			{
				dynamic_cast<Mesh *>(this->mesh_pt(nmi))->bump_topology_generation();
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

	// See the class comment in problem.hpp. The dynamic_cast is on the AssemblyHandler, which every
	// tracker derives from alongside AugmentedSparsityProvider.
	Problem::BaseDofDistributionScope::BaseDofDistributionScope(Problem *problem)
	{
		AugmentedSparsityProvider *prov = dynamic_cast<AugmentedSparsityProvider *>(problem->assembly_handler_pt());
		if (prov) Helper = prov->dof_distribution_helper();
		if (Helper) Helper->install_base_distribution();
	}

	Problem::BaseDofDistributionScope::~BaseDofDistributionScope()
	{
		if (Helper) Helper->restore_augmented_distribution();
	}

	void Problem::get_base_dof_distribution_info(unsigned &nrow, unsigned &nrow_local, unsigned &first_row, bool &distributed)
	{
		BaseDofDistributionScope base_scope(this);
		oomph::LinearAlgebraDistribution *dist = this->dof_distribution_pt();
		if (dist == 0)
			throw_runtime_error("The problem has no dof distribution yet, so it has no row layout. Build/initialise the problem first.");
		nrow = dist->nrow();
		nrow_local = dist->nrow_local();
		first_row = dist->first_row();
		distributed = dist->distributed();
	}

	// Assembles (or reuses/reallocates the cached eigen_MassMatrixPt/eigen_JacobianMatrixPt) the mass
	// matrix M and Jacobian J for the generalized eigenproblem J*v = sigma*M*v around the shift sigma_r.
	// Note: Dirichlet-by-matrix-manipulation is not yet supported here (see thrown error below).
	//
	// The BaseDofDistributionScope makes this work while a bifurcation tracker is installed: it is the
	// BASE state's eigenproblem that is wanted then (used to look for secondary/codim-2 bifurcations
	// along a locus), and the base state is exactly what oomph's EigenProblemHandler assembles -- only
	// the row layout had to be put back. Nothing is renumbered and Dof_pt is untouched.
	void Problem::assemble_eigenproblem_matrices(oomph::CRDoubleMatrix *&M, oomph::CRDoubleMatrix *&J, double sigma_r)
	{
		BaseDofDistributionScope base_scope(this);
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

	// oomph::DoubleVector::operator[] indexes the vector's LOCAL rows. On a distributed (MPI) problem the
	// vector holds only nrow_local() doubles, so reading/writing it by GLOBAL equation number runs past the
	// end of the buffer. These two helpers convert between the (possibly distributed) DoubleVector and a
	// globally indexed std::vector; serially they are a plain copy, since the vector is not distributed.
	// (Static members of Problem so that the bifurcation handlers can reuse them, e.g. in get_eigenfunction.)
	void Problem::gather_double_vector_to_global(oomph::DoubleVector &v, std::vector<double> &res)
	{
		if (v.distributed())
		{
			oomph::LinearAlgebraDistribution global_dist(v.distribution_pt()->communicator_pt(), v.nrow(), false);
			v.redistribute(&global_dist);
		}
		res.resize(v.nrow());
		for (unsigned int i = 0; i < v.nrow(); i++)
			res[i] = v[i];
	}

	// Fill v (built on the given, possibly distributed, distribution) from a globally indexed std::vector.
	void Problem::scatter_global_to_double_vector(const std::vector<double> &src, oomph::DoubleVector &v,
											   oomph::LinearAlgebraDistribution *target_dist)
	{
		if (target_dist->distributed())
		{
			// Build globally replicated first, then redistribute onto the target (local) distribution.
			oomph::LinearAlgebraDistribution global_dist(target_dist->communicator_pt(), src.size(), false);
			v.build(&global_dist, 0.0);
			for (unsigned int i = 0; i < src.size(); i++)
				v[i] = src[i];
			v.redistribute(target_dist);
		}
		else
		{
			v.build(target_dist, 0.0);
			for (unsigned int i = 0; i < src.size(); i++)
				v[i] = src[i];
		}
	}

	// Multiplies the residual in place by the deflation factor M(U) (1.0, i.e. a no-op, unless a
	// deflation operator is installed - see get_residual_scale_factor()). A scalar times a row block
	// is correct on any distribution and needs no communication, which is the whole reason deflation
	// can ride on the ordinary assembly rather than the custom-assembler pipeline.
	void Problem::apply_residual_scale_factor(oomph::DoubleVector &residuals)
	{
		if (!residual_scale_hook_active) return;
		const double s = this->get_residual_scale_factor();
		if (s == 1.0) return;
		const unsigned n = residuals.nrow_local();
		double *v = residuals.values_pt();
		for (unsigned i = 0; i < n; i++) v[i] *= s;
	}

	// This rank's block of the dof vector, in the order the dof distribution numbers it. Serially and
	// on a replicated run that is the whole vector; under --distribute it is the owned rows only.
	// get_current_dofs() gathers instead, which is the right contract for its callers but costs
	// O(ndof) per rank - deflation reads this on every residual assembly.
	std::vector<double> Problem::get_local_dof_values()
	{
		const unsigned n = this->GetDofPtr().size();
		std::vector<double> res(n);
		for (unsigned i = 0; i < n; i++) res[i] = *(this->GetDofPtr()[i]);
		return res;
	}

#ifdef OOMPH_HAS_MPI
	// Fresh dof halo scheme for the CURRENT equation numbering. Deletes the previous scheme first:
	// oomph's setup_dof_halo_scheme() plain-overwrites Halo_scheme_pt, which both leaks and -- worse --
	// would leave a scheme built against a stale numbering in place if the allocation pattern hid the
	// leak. Must be called while the DEFAULT assembly handler is installed (the scheme is built from
	// get_my_eqns() of the current handler, and it has to describe the BASE equations).
	void Problem::RebuildDofHaloScheme()
	{
		if (this->Halo_scheme_pt)
		{
			delete this->Halo_scheme_pt;
			this->Halo_scheme_pt = NULL;
			this->Halo_dof_pt.clear();
		}
		this->setup_dof_halo_scheme();
	}
#endif

	// Dof values at history time level t (t=0 is the current time), as a plain std::vector<double>
	std::vector<double> Problem::get_history_dofs(unsigned t)
	{
		std::vector<double> res;
		oomph::DoubleVector dofs;
		if (t == 0)
			this->get_dofs(dofs);
		else
			this->get_dofs(t, dofs);
		gather_double_vector_to_global(dofs, res);
		return res;
	}

	// Current dof values, together with a per-dof flag marking whether the dof is a nodal position dof
	// (i.e. belongs to a node's variable_position_pt(), as opposed to a "physical" field dof); the flag
	// is determined by scanning all nodes of all submeshes for equation numbers coming from their position data.
	std::tuple<std::vector<double>, std::vector<bool>> Problem::get_current_dofs()
	{
		std::vector<double> res;
		std::vector<bool> is_positional(this->ndof(), false);
		oomph::DoubleVector dofs;
		this->get_dofs(dofs);
		gather_double_vector_to_global(dofs, res);

		for (unsigned int ism = 0; ism < this->nsub_mesh(); ism++)
		{
			pyoomph::Mesh *m = dynamic_cast<pyoomph::Mesh *>(this->mesh_pt(ism));
			for (unsigned int in = 0; in < m->nnode(); in++)
			{
				auto *n = static_cast<pyoomph::Node *>(m->node_pt(in));
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
		if (inp.size() != this->ndof())
			throw_runtime_error("Mismatch in dof vector size");
		oomph::DoubleVector dofs;
		// inp is indexed by GLOBAL equation number; the dof distribution may be row-partitioned under MPI.
		scatter_global_to_double_vector(inp, dofs, this->dof_distribution_pt());
		this->set_dofs(dofs);
#ifdef OOMPH_HAS_MPI
		// set_dofs() writes only Dof_pt, i.e. this rank's OWNED dofs; the copies living on halo nodes
		// keep whatever they held before. Every oomph-lib routine that updates the dofs itself (the
		// Newton step, unsteady_newton_solve, arc-length) follows it with this call for exactly that
		// reason -- but set_current_dofs() is reached from Python, so nothing else does it here.
		//
		// Found via the eigenfunction: pushing an eigenvector into the dofs and integrating it over the
		// mesh gave a smaller answer under --distribute than serially, because elements touching a halo
		// node were integrating the stale (base-state) values there.
		//
		// Collective, so set_current_dofs() is now collective too. That is not a new constraint in
		// practice -- it is reached from output, eigenfunction plotting and the solve loop, all of which
		// every rank already runs together -- but a rank-0-only call to it would now hang rather than
		// merely leave the halos wrong.
		if (this->distributed())
			this->synchronise_all_dofs();
#endif
	}


    // oomph-lib's get_dofs(t,...) does not account for the variable_position_pt() dofs of moving nodes
    // (see the analogous comment in set_dofs below) - this override fixes that up afterwards.
    void Problem::get_dofs(const unsigned& t, oomph::DoubleVector& dofs) const
	{
#ifdef OOMPH_HAS_MPI
	  // oomph-lib's own get_dofs(t,...) builds the vector on the dof distribution -- this rank's rows
	  // -- and then indexes it by GLOBAL equation number, which is why it guards itself with a
	  // PARANOID throw that vanishes in pyoomph's default build. Do the same walk here, writing only
	  // the rows this rank owns; a halo copy's equation number falls outside the range and is skipped,
	  // since the rank that owns it writes it.
	  if (this->distributed())
	  {
	    // Reading only, but oomph-lib's global_data_pt()/mesh_pt() accessors have no const overload
	    // (and Global_data_pt is private), so a non-const alias is the only way in from a const method.
	    Problem *self = const_cast<Problem *>(this);
	    dofs.build(this->dof_distribution_pt(), 0.0);
	    const unsigned first_row = this->dof_distribution_pt()->first_row();
	    const unsigned n_row_local = this->dof_distribution_pt()->nrow_local();
	    auto store = [&](int eqn_number, double value)
	    {
	      if (eqn_number >= 0 && (unsigned)eqn_number >= first_row && (unsigned)eqn_number < first_row + n_row_local)
	        dofs[(unsigned)eqn_number - first_row] = value;
	    };
	    for (unsigned i = 0, ni = self->nglobal_data(); i < ni; i++)
	      for (unsigned j = 0, nj = self->global_data_pt(i)->nvalue(); j < nj; j++)
	        store(self->global_data_pt(i)->eqn_number(j), self->global_data_pt(i)->value(t, j));
	    for (unsigned i = 0, ni = self->mesh_pt()->nelement(); i < ni; i++)
	    {
	      oomph::GeneralisedElement *ele_pt = self->mesh_pt()->element_pt(i);
	      for (unsigned j = 0, nj = ele_pt->ninternal_data(); j < nj; j++)
	      {
	        oomph::Data *d_pt = ele_pt->internal_data_pt(j);
	        for (unsigned k = 0, nk = d_pt->nvalue(); k < nk; k++) store(d_pt->eqn_number(k), d_pt->value(t, k));
	      }
	    }
	    for (unsigned i = 0, ni = self->mesh_pt()->nnode(); i < ni; i++)
	    {
	      oomph::Node *n_pt = self->mesh_pt()->node_pt(i);
	      for (unsigned j = 0, nj = n_pt->nvalue(); j < nj; j++) store(n_pt->eqn_number(j), n_pt->value(t, j));
	      // ...and the moving-mesh position dofs oomph-lib forgets, as in the serial branch below.
	      pyoomph::Node *pn_pt = dynamic_cast<pyoomph::Node *>(n_pt);
	      if (!pn_pt) continue;
	      for (unsigned j = 0, nj = pn_pt->variable_position_pt()->nvalue(); j < nj; j++)
	        store(pn_pt->variable_position_pt()->eqn_number(j), pn_pt->variable_position_pt()->value(t, j));
	    }
	    return;
	  }
#endif
      oomph::Problem::get_dofs(t,dofs);
	  //std::cout << "GET HISTORY DOFS " << t << std::endl;
	  for (unsigned i = 0, ni = mesh_pt()->nnode(); i < ni; i++)
      {
       pyoomph::Node* node_pt = static_cast<pyoomph::Node*>(mesh_pt()->node_pt(i));
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
#ifdef OOMPH_HAS_MPI
	  // As in get_dofs(t,...): write only the rows this rank owns, by the same range test, and then
	  // let the halo exchange carry them to the other ranks' copies. Data::add_values_to_vector sends
	  // every time level (it loops to ntstorage()), so synchronise_all_dofs() propagates history
	  // values and not just the current ones -- which is what makes this possible at all.
	  if (this->distributed())
	  {
	    const unsigned first_row = this->dof_distribution_pt()->first_row();
	    const unsigned n_row_local = this->dof_distribution_pt()->nrow_local();
	    auto fetch = [&](int eqn_number, double &out) -> bool
	    {
	      if (eqn_number < 0 || (unsigned)eqn_number < first_row || (unsigned)eqn_number >= first_row + n_row_local) return false;
	      out = dof_pt[(unsigned)eqn_number - first_row];
	      return true;
	    };
	    double v = 0.0;
	    for (unsigned i = 0, ni = nglobal_data(); i < ni; i++)
	      for (unsigned j = 0, nj = global_data_pt(i)->nvalue(); j < nj; j++)
	        if (fetch(global_data_pt(i)->eqn_number(j), v)) global_data_pt(i)->set_value(t, j, v);
	    for (unsigned i = 0, ni = mesh_pt()->nelement(); i < ni; i++)
	    {
	      oomph::GeneralisedElement *ele_pt = mesh_pt()->element_pt(i);
	      for (unsigned j = 0, nj = ele_pt->ninternal_data(); j < nj; j++)
	      {
	        oomph::Data *d_pt = ele_pt->internal_data_pt(j);
	        for (unsigned k = 0, nk = d_pt->nvalue(); k < nk; k++)
	          if (fetch(d_pt->eqn_number(k), v)) d_pt->set_value(t, k, v);
	      }
	    }
	    for (unsigned i = 0, ni = mesh_pt()->nnode(); i < ni; i++)
	    {
	      oomph::Node *n_pt = mesh_pt()->node_pt(i);
	      for (unsigned j = 0, nj = n_pt->nvalue(); j < nj; j++)
	        if (fetch(n_pt->eqn_number(j), v)) n_pt->set_value(t, j, v);
	      pyoomph::Node *pn_pt = dynamic_cast<pyoomph::Node *>(n_pt);
	      if (!pn_pt) continue;
	      for (unsigned j = 0, nj = pn_pt->variable_position_pt()->nvalue(); j < nj; j++)
	        if (fetch(pn_pt->variable_position_pt()->eqn_number(j), v)) pn_pt->variable_position_pt()->set_value(t, j, v);
	    }
	    this->synchronise_all_dofs();
	    return;
	  }
#endif
	 oomph::Problem::set_dofs(t,dof_pt);
	 //std::cout << "SET HISTORY DOFS " << t << std::endl;
	 // But oomph-lib forgot the variable position pt of moving nodes...
     for (unsigned i = 0, ni = mesh_pt()->nnode(); i < ni; i++)
     {
      pyoomph::Node* node_pt = static_cast<pyoomph::Node*>(mesh_pt()->node_pt(i));
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
#ifdef OOMPH_HAS_MPI
	  // As in get_dofs(t,...): write only the rows this rank owns, by the same range test, and then
	  // let the halo exchange carry them to the other ranks' copies. Data::add_values_to_vector sends
	  // every time level (it loops to ntstorage()), so synchronise_all_dofs() propagates history
	  // values and not just the current ones -- which is what makes this possible at all.
	  if (this->distributed())
	  {
	    const unsigned first_row = this->dof_distribution_pt()->first_row();
	    const unsigned n_row_local = this->dof_distribution_pt()->nrow_local();
	    auto fetch = [&](int eqn_number, double &out) -> bool
	    {
	      if (eqn_number < 0 || (unsigned)eqn_number < first_row || (unsigned)eqn_number >= first_row + n_row_local) return false;
	      out = *(dof_pt[(unsigned)eqn_number]);
	      return true;
	    };
	    double v = 0.0;
	    for (unsigned i = 0, ni = nglobal_data(); i < ni; i++)
	      for (unsigned j = 0, nj = global_data_pt(i)->nvalue(); j < nj; j++)
	        if (fetch(global_data_pt(i)->eqn_number(j), v)) global_data_pt(i)->set_value(t, j, v);
	    for (unsigned i = 0, ni = mesh_pt()->nelement(); i < ni; i++)
	    {
	      oomph::GeneralisedElement *ele_pt = mesh_pt()->element_pt(i);
	      for (unsigned j = 0, nj = ele_pt->ninternal_data(); j < nj; j++)
	      {
	        oomph::Data *d_pt = ele_pt->internal_data_pt(j);
	        for (unsigned k = 0, nk = d_pt->nvalue(); k < nk; k++)
	          if (fetch(d_pt->eqn_number(k), v)) d_pt->set_value(t, k, v);
	      }
	    }
	    for (unsigned i = 0, ni = mesh_pt()->nnode(); i < ni; i++)
	    {
	      oomph::Node *n_pt = mesh_pt()->node_pt(i);
	      for (unsigned j = 0, nj = n_pt->nvalue(); j < nj; j++)
	        if (fetch(n_pt->eqn_number(j), v)) n_pt->set_value(t, j, v);
	      pyoomph::Node *pn_pt = dynamic_cast<pyoomph::Node *>(n_pt);
	      if (!pn_pt) continue;
	      for (unsigned j = 0, nj = pn_pt->variable_position_pt()->nvalue(); j < nj; j++)
	        if (fetch(pn_pt->variable_position_pt()->eqn_number(j), v)) pn_pt->variable_position_pt()->set_value(t, j, v);
	    }
	    this->synchronise_all_dofs();
	    return;
	  }
#endif
     oomph::Problem::set_dofs(t,dof_pt);
	 //std::cout << "SET HISTORY DOFS " << t << std::endl;
	 // But oomph-lib forgot the variable position pt of moving nodes...
     for (unsigned i = 0, ni = mesh_pt()->nnode(); i < ni; i++)
     {
      pyoomph::Node* node_pt = static_cast<pyoomph::Node*>(mesh_pt()->node_pt(i));
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
		// 'inp' is indexed by GLOBAL equation number while 'dofs' is built on the dof distribution,
		// i.e. this rank's rows -- so the fill loop below is a scatter, not a copy, exactly as in
		// set_current_dofs(). Writing ndof() entries into it used to overrun the heap by
		// ndof()-nrow_local doubles per call when distributed.
		oomph::DoubleVector dofs;
		if (inp.size() != this->ndof())
		{
			std::ostringstream oss; oss << "Try to set dofs of length " << inp.size() << " while problem has " << this->ndof() << " dofs";
			throw_runtime_error("Mismatch in dof vector size. " + oss.str());
		}
		if (t>=this->time_stepper_pt()->ntstorage()) 
		        throw_runtime_error("Wrong history offset");
		scatter_global_to_double_vector(inp, dofs, this->dof_distribution_pt());
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
						if (static_cast<pyoomph::Node *>(n)->variable_position_pt()->is_pinned(iv))
							res.push_back(static_cast<pyoomph::Node *>(n)->variable_position_pt()->value(iv));
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
	// Before assembling, make each hanging node's own (raw) storage consistent with what its masters
	// say. The elemental (JIT) assembly reads a node's RAW value/position, but a hanging node's raw
	// storage is only a cache of its masters, refreshed as a side effect of an assembly or output pass
	// over the elements that contain it. Anything that does not touch those elements leaves it stale.
	//
	// That bites on ANY MPI run with more than one process, not only a distributed one, because
	// oomph-lib splits the assembly by element across the ranks (First_el_for_assembly /
	// Last_el_plus_one_for_assembly) in both cases. Each rank therefore only ever refreshes the hanging
	// nodes inside its own element range. The two cases need different repairs:
	//
	//  - DISTRIBUTED: collapse_hanging_node_values(). The masters are halo nodes, freshly
	//    value-synchronised at the end of every Newton step; without this a NONLINEAR residual/Jacobian
	//    would be assembled from stale hanging values on every iteration (a single linear step happens
	//    to converge regardless, which is why that bug only showed up in the final values).
	//
	//  - REPLICATED (plain `mpirun`, no --distribute): the whole mesh is present on every rank, but only
	//    the elements this rank assembles got their hanging nodes refreshed -- so the OTHER ranks' region
	//    kept whatever was there before, which for a freshly adapted mesh is zero. This one has to go
	//    through interpolate_hanging_values(), not collapse_hanging_node_values(): it also restores the
	//    hanging POSITIONS, which are equally stale on a moving mesh and which the node-only routine does
	//    not touch.
	//    Found via the spatial error estimator: two ranks handed the identical solution computed
	//    elemental errors differing by 60-92% on every element, because the raw FE flux is built from
	//    nodal values and each rank had correct hanging values only inside its own assembly range. The
	//    ranks then refined different regions and their meshes diverged outright.
	//
	// Must run every iteration, not just once after the solve. No-op on a single process.
	//
	// On its own this made the ranks agree only to ~1e-13. The residue was a SEPARATE defect, and not an
	// MPI one: complete_hanging_nodes copied a std::map<Node*,double> straight into the HangInfo, so the
	// masters were summed in HEAP ADDRESS order and interpolate_hanging_values() returned a different
	// answer on each run of the same binary, in a single process. That is fixed (the masters are now
	// written in mesh node-vector order), and the ranks agree bit-for-bit. The reproducer, the full
	// elimination table -- valgrind, setarch -R, PYTHONHASHSEED all clean -- and the wrong turns are in
	// dev_docs/replicated_mpi_correctness.md §3-§4.
	//
	// Without this sync, 120 hanging nodes sit at ZERO on the ranks that do not assemble them, elemental
	// errors differ by 60-92%, and the meshes diverge outright.
	static inline void sync_hanging_values_if_parallel(pyoomph::Problem *prob)
	{
#ifdef OOMPH_HAS_MPI
		if (!prob->communicator_pt() || prob->communicator_pt()->nproc() < 2) return;
		auto do_mesh = [](oomph::Mesh *mp) {
			pyoomph::Mesh *m = dynamic_cast<pyoomph::Mesh *>(mp);
			if (!m) return;
			if (m->is_mesh_distributed()) m->collapse_hanging_node_values();
			else m->interpolate_hanging_values();
		};
		const unsigned nm = prob->nsub_mesh();
		if (nm == 0) { if (prob->mesh_pt()) do_mesh(prob->mesh_pt()); }
		else for (unsigned i = 0; i < nm; i++) do_mesh(prob->mesh_pt(i));
#endif
	}

	// Ask for the running Newton solve to be abandoned. Cheap and side-effect-free: it records the
	// request, and the next residual evaluation acts on it. See the header for what this replaces.
	void Problem::request_newton_abort(const std::string &reason)
	{
		newton_abort_flag = true;
		newton_abort_reason = reason;
	}

	// Consume a pending abort request, agreeing across ranks first.
	//
	// The agreement is the point. Whatever decides to reject a step -- an interface about to
	// self-intersect, say -- usually only sees it on the ranks holding that part of the mesh, so the
	// request is inherently partition-dependent, while abandoning the solve has to be unanimous. One
	// int through MPI_Allreduce, on a path that is collective anyway.
	bool Problem::consume_newton_abort_request()
	{
		bool aborting = newton_abort_flag;
#ifdef OOMPH_HAS_MPI
		if (Communicator_pt && Communicator_pt->nproc() > 1)
		{
			int mine = aborting ? 1 : 0, any = 0;
			MPI_Allreduce(&mine, &any, 1, MPI_INT, MPI_MAX, Communicator_pt->mpi_comm());
			aborting = (any != 0);
		}
#endif
		if (!aborting) return false;
		if (!newton_abort_reason.empty())
		{
			// Reported rather than carried: oomph::NewtonSolverError has no room for a message.
			std::cout << "Abandoning the Newton solve: " << newton_abort_reason << std::endl;
		}
		newton_abort_flag = false;
		newton_abort_reason.clear();
		return true;
	}

	// --- Inverted elements ---------------------------------------------------------------------
	//
	// Why any of this exists: the detector throws from inside the element loop, and that loop is split
	// by element across the ranks in BOTH MPI modes. The rank holding the folded element left the loop
	// while the others were still inside the assembly's collectives, so `mpirun -n 2` on a folding mesh
	// hung instead of reporting anything - measured on dev_docs/examples/inverted_element_notch.py.
	// Inside this scope the detector records instead of throwing, every rank finishes the loop, and
	// raise_if_any() then reduces and throws on all of them or on none.

	Problem::InvertedElementScope::InvertedElementScope(Problem *p) : prob(p)
	{
		// Nested scopes are possible (get_jacobian assembles residuals too on some paths); only the
		// outermost one owns the flag, so an inner destructor cannot reopen the immediate throw.
		outer = (prob->inverted_element_scope_depth++ == 0);
		if (outer)
		{
			BulkElementBase::inverted_elements_detected = 0;
			BulkElementBase::inverted_element_message.clear();
		}
	}

	Problem::InvertedElementScope::~InvertedElementScope()
	{
		prob->inverted_element_scope_depth--;
	}

	void Problem::InvertedElementScope::raise_if_any()
	{
		if (!outer) return;
		bool seen = (BulkElementBase::inverted_elements_detected > 0);
#ifdef OOMPH_HAS_MPI
		if (prob->communicator_pt() && prob->communicator_pt()->nproc() > 1)
		{
			// MPI_MAX over one int, on a path that is collective anyway - the same shape of agreement
			// consume_newton_abort_request() makes, and for the same reason: the condition is seen by
			// whichever rank happens to hold the element, while the response has to be unanimous.
			int mine = seen ? 1 : 0, any = 0;
			MPI_Allreduce(&mine, &any, 1, MPI_INT, MPI_MAX, prob->communicator_pt()->mpi_comm());
			seen = (any != 0);
		}
#endif
		if (!seen) return;

		// A rank that saw nothing has no message of its own; say so rather than reporting an empty one.
		std::string msg = BulkElementBase::inverted_element_message;
		if (msg.empty()) msg = "An inverted element was detected on another process.";

		BulkElementBase::inverted_elements_detected = 0;
		BulkElementBase::inverted_element_message.clear();

		// One report per assembly that saw one, however many integration points of however many
		// elements were involved.
		prob->inversion_reports_in_this_solve_call++;

		// Escalate once this solve() call has absorbed enough of them that reducing the step is
		// clearly not the answer. See inversion_remesh_threshold in problem.hpp for why the count is
		// per CALL rather than per run of consecutive solves - the run-based proxy the design proposed
		// was measured and does not work.
		//
		// Unanimous by construction: the "seen" flag above is already reduced over the ranks and the
		// counter is incremented on every rank, so every rank reaches the same verdict here without a
		// second collective.
		if (prob->inversion_remesh_threshold > 0 &&
			prob->inversion_reports_in_this_solve_call >= prob->inversion_remesh_threshold)
		{
			std::ostringstream oss;
			oss << "An inverted element has been reported " << prob->inversion_reports_in_this_solve_call
				<< " times while trying to take this step, so reducing the step is not curing it. "
				<< "Remeshing instead." << std::endl
				<< msg;
			throw pyoomph::InvertedElementRemeshRequest(oss.str());
		}
		throw oomph::InvertedElementError(msg, OOMPH_CURRENT_FUNCTION, OOMPH_EXCEPTION_LOCATION);
	}

	void Problem::get_residuals(oomph::DoubleVector &residuals)
	{
		// Before anything else, including the halo sync: an abandoned solve should cost nothing.
		if (consume_newton_abort_request())
		{
			// Raise the error directly rather than returning a residual so large that oomph-lib's own
			// max_residuals test trips. That was the first attempt and it silently did nothing when the
			// request arrived before the FIRST Newton step: the count==1 convergence check compares
			// against the tolerance only, never against max_residuals, so the huge residual was
			// consumed, the step proceeded, and the next check saw an ordinary residual and converged.
			// Throwing does not depend on which check happens to read the residual, nor on the user's
			// max_residuals being set to anything in particular.
			//
			// oomph-lib's steady_newton_solve() wraps the whole Newton loop in catch(NewtonSolverError),
			// so this lands in exactly the handler an ordinary divergence lands in, and callers that
			// already recover from a failed solve need no change. Outside a Newton solve there is
			// nothing to catch it and it surfaces as a generic error -- which is the honest outcome for
			// "abandon the solve" when no solve is running.
			throw oomph::NewtonSolverError(0, std::numeric_limits<double>::max());
		}
		// One assembly pass: the dof vector is fixed for its whole duration, so each element's hanging
		// values need re-deriving at most once (see HangInterpPassScope). The MPI sync below is itself
		// a full interpolate sweep, and it stamps, so its work is then not repeated per element.
		HangInterpPassScope __hang_pass;
		// Defer any inverted-element report until every rank is out of the (collective) element loop.
		InvertedElementScope __inv(this);
		sync_hanging_values_if_parallel(this);
		if (!use_custom_residual_jacobian)
		{
			get_residuals_by_elemental_assembly(residuals);
			__inv.raise_if_any();
			if (!this->dirichlets_by_removing_from_dof_vector)
			{
				if (this->bifurcation_tracking_mode!="") throw_runtime_error("TODO: Cannot remove dirichlet dofs from the residual vector by matrix manipulation when bifurcation tracking is active, since the user-provided residual vector would still contain contributions from the dirichlet dofs, which would be wrong");
				this->remove_dirichlets_by_matrix_manipulation(residuals);
			}
	apply_residual_scale_factor(residuals);
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
		HangInterpPassScope __hang_pass; // see Problem::get_residuals
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
		HangInterpPassScope __hang_pass; // see Problem::get_residuals
		InvertedElementScope __inv(this); // see Problem::get_residuals
		sync_hanging_values_if_parallel(this);
		last_jacobian_was_condensed = false;
		if (!use_custom_residual_jacobian)
		{
			get_jacobian_by_elemental_assembly(residuals, jacobian);
			__inv.raise_if_any();
			if (!this->dirichlets_by_removing_from_dof_vector)
			{
				if (this->bifurcation_tracking_mode!="") throw_runtime_error("TODO: Cannot remove dirichlet dofs from the jacobian matrix by matrix manipulation when bifurcation tracking is active, since the user-provided jacobian matrix would still contain contributions from the dirichlet dofs, which would be wrong");
				this->remove_dirichlets_by_matrix_manipulation(residuals,&jacobian);
			}
			apply_residual_scale_factor(residuals);
			// Static condensation comes LAST, after the Dirichlet manipulation, and that order is forced.
			// Condensation is an exact algebraic elimination of whatever linear system it is given, so it
			// has to be given the FINAL one; remove_dirichlets_by_matrix_manipulation() replaces a
			// Dirichlet row by the identity, zeroes that dof's column everywhere and zeroes its residual,
			// i.e. it is still rewriting the system. Condensing first and manipulating afterwards would
			// eliminate a Dirichlet-constrained dof through its RAW equation and fold that residual into
			// the retained ones -- and the manipulation could no longer reach the eliminated row anyway,
			// since it is gone from the matrix. In this order a Dirichlet dof caught by the selection is
			// simply decoupled by the time the elimination sees it, and comes back out as dx = 0, which is
			// what the full system gives too.
			if (static_condensation_engages_now())
			{
				apply_static_condensation(residuals, jacobian);
				last_jacobian_was_condensed = true;
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
			// The distribution has to be set up first, exactly like the residuals above: the build()
			// overload below writes nrow_local() rows and takes that from the distribution, so an
			// unbuilt one makes it write a full row_start array into a zero-row matrix. Every call
			// coming out of a Newton solve hands in a matrix oomph-lib has already distributed, which
			// is why this went unnoticed; _assemble_residual_jacobian(), i.e. Problem.assemble_jacobian()
			// from Python, passes a fresh CRDoubleMatrix and segfaulted whenever a Python-side custom
			// assembler (a CustomBifurcationTracker) was installed. Note the copy loop still writes
			// info.residuals.size() entries regardless of what the caller's distribution says -- see
			// dev_docs/mpi_augmented_systems.md B2, which is still open and is why this whole branch
			// is refused under MPI.
			if (!jacobian.distribution_built())
			{
				oomph::LinearAlgebraDistribution dist(this->communicator_pt(), info.residuals.size(), false);
				jacobian.build(&dist);
			}
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

	// Toggles this Problem's replace_RJM_by_param_deriv pointer, which, when non-NULL, makes subsequent
	// residual/Jacobian/mass-matrix assembly calls return the derivative w.r.t. the named global
	// parameter instead of the normal residual/Jacobian/mass matrix (used internally e.g. for parameter
	// derivative computations that must reuse the same elemental assembly loop machinery).
	//
	// This is a latch with no scope guard, and it is reachable from Python: an exception raised between
	// the "on" and the "off" call leaves the Problem assembling parameter derivatives. It used to be a
	// process-wide global, so that also left every OTHER Problem doing so.
	void Problem::_replace_RJM_by_param_deriv(std::string name, bool active)
	{
		if (!active)
			replace_RJM_by_param_deriv = NULL;
		else
		{
			if (!global_params_by_name.count(name))
				throw_runtime_error("Cannot replace residuals/jacobian/mass matrix by parameter derivatives for global parameter " + name + ", since it is not present in the problem");
			auto *p = global_params_by_name[name];
			replace_RJM_by_param_deriv = &(p->value());
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

	// Fills oomph-lib's Dof_derivative and Parameter_derivative at the CURRENT solution without taking a
	// step, by the same route a step's end does it: solve J z = dR/dparam, then normalise (dparam/ds,
	// dU/ds) onto the arclength constraint. Wanted wherever a solution was reached by something other
	// than an arclength step - branch switching lands with a plain Newton solve at a prescribed
	// parameter offset - and the direction the continuation would leave in is nevertheless a real
	// quantity that can be asked for. The Jacobian must be non-singular here, i.e. NOT at a bifurcation.
	void Problem::compute_arclength_tangent(const std::string param)
	{
		if (!global_params_by_name.count(param))
			throw_runtime_error("Cannot compute the arclength tangent in the global parameter " + param + ", since it is not present in the problem");
		// oomph sizes these inside arc_length_step_solve_helper, not in calculate_continuation_derivatives,
		// which writes into them regardless. Reached with no step taken yet they are still empty, and the
		// write went out of range - a segfault, not an exception. Same guard as the step's own.
		if (!this->Use_continuation_timestepper)
		{
			const unsigned long ndof_local = this->Dof_distribution_pt->nrow_local();
			if (this->Dof_derivative.size() != ndof_local)
				this->Dof_derivative.resize(ndof_local, 0.0);
			if (this->Dof_current.size() != ndof_local)
				this->Dof_current.resize(ndof_local, 0.0);
		}
		auto *p = global_params_by_name[param];
		this->calculate_continuation_derivatives(&(p->value()));
	}

	// oomph-lib keeps Dof_derivative and Dof_current on the DOF DISTRIBUTION, i.e. this rank's rows
	// only when the problem is distributed. Everything on the Python side of these three functions
	// deals in GLOBAL dof vectors -- get_history_dofs()/set_history_dofs(), which is what
	// remesh_handler_during_continuation() carries the tangent through, and get_current_dofs() --
	// so these gather on the way out and scatter on the way in, and the two ends now agree.
	//
	// They did not. remesh_handler_during_continuation() reads the tangent back with
	// get_history_dofs() (global, ndof long) and handed it to update_dof_vectors_for_continuation(),
	// which required nrow_local: under --distribute that threw "Mismatch in size of ddof and current
	// dof vectors" out of the middle of go_to_param(), so the hanging-droplet tutorial never reached
	// its base state at all. The reverse direction was wrong too and would have written a local-length
	// vector into a global history slot. Serially and under plain mpirun both are the identity, which
	// is why this only ever showed with --distribute.
	static void gather_local_dof_array_to_global(pyoomph::Problem *prob, const oomph::Vector<double> &src,
												 std::vector<double> &res)
	{
		if (src.size() == 0) { res.clear(); return; } // "no continuation state", which callers test for
		oomph::DoubleVector v(prob->dof_distribution_pt(), 0.0);
		const unsigned n = v.nrow_local();
		for (unsigned i = 0; i < n && i < src.size(); i++) v[i] = src[i];
		pyoomph::Problem::gather_double_vector_to_global(v, res);
	}

	// d(dof)/ds (arclength derivative) as a plain std::vector, GLOBALLY indexed on every rank
	std::vector<double> Problem::get_arclength_dof_derivative_vector()
	{
		std::vector<double> res;
		gather_local_dof_array_to_global(this, Dof_derivative, res);
		return res;
	}

	// Dof values at the last continuation step (oomph-lib's internal Dof_current), globally indexed
	std::vector<double> Problem::get_arclength_dof_current_vector()
	{
		std::vector<double> res;
		gather_local_dof_array_to_global(this, Dof_current, res);
		return res;
	}

    // Writes ddof/curr -- GLOBALLY indexed dof-derivative and current-dof vectors, ndof long on every
    // rank -- into oomph-lib's internal Dof_derivative/Dof_current, which hold this rank's rows.
    // See the comment on get_arclength_dof_derivative_vector() for why the boundary is global.
    void Problem::update_dof_vectors_for_continuation(const std::vector<double> &ddof, const std::vector<double> &curr)
    {
		if (ddof.size() != curr.size()) throw_runtime_error("Mismatch in size of ddof and curr");
		const unsigned ndof_local = Dof_distribution_pt->nrow_local();
		if (ddof.size() != this->ndof())
		{
			throw_runtime_error("Mismatch in size of ddof (" + std::to_string(ddof.size()) +
								") and the problem's dof count (" + std::to_string(this->ndof()) +
								"). These vectors are indexed GLOBALLY, as get_history_dofs() and "
								"get_arclength_dof_derivative_vector() return them.");
		}
		if (Dof_derivative.size() != ndof_local) Dof_derivative.resize(ndof_local, 0.0);
		if (Dof_current.size() != ndof_local) Dof_current.resize(ndof_local, 0.0);
		// Take this rank's slice. first_row() is 0 and nrow_local() is ndof unless distributed, so
		// this is the old loop verbatim in every other case.
		const unsigned first = Dof_distribution_pt->first_row();
		for (unsigned i = 0; i < ndof_local; i++)
		{
			Dof_derivative[i] = ddof[first + i];
			Dof_current[i] = curr[first + i];
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
	// Backstop for the Python refusal in Problem.activate_bifurcation_tracking(). oomph-lib's block
	// linear solvers open with
	//     FoldHandler* handler_pt = static_cast<FoldHandler*>(problem_pt->assembly_handler_pt());
	// (assembly_handler.cc) and then call a FoldHandler method on it. pyoomph installs MyFoldHandler /
	// MyHopfHandler, which derive from oomph::AssemblyHandler and not from those classes, so that cast
	// reinterprets an unrelated object and the first member access is undefined behaviour -- measured
	// as a plain SEGV on a serial Bratu fold track, with any linear solver. Throwing here rather than
	// installing the solver turns a crash into a message for any caller that bypasses Python.
	static void refuse_block_solve(const char *what)
	{
		throw_runtime_error(std::string(what) + " with block_solve=true is not supported: oomph-lib's "
							"block linear solvers static_cast the assembly handler to their own "
							"FoldHandler/HopfHandler, which pyoomph's handlers are not. Use the full "
							"augmented solve (block_solve=false).");
	}

    // given starting null-eigenvector guess; if block_solve, additionally switches to the augmented block
    // linear solver that exploits the bordered system's block structure instead of a full dense solve.
    void Problem::activate_my_fold_tracking(double *const &parameter_pt, const oomph::DoubleVector &eigenvector, const bool &block_solve)
	{
		reset_assembly_handler_to_default();
		this->assembly_handler_pt() = new MyFoldHandler(this, parameter_pt, eigenvector);
		if (block_solve) refuse_block_solve("Fold tracking");
	}

	// Same as above, but lets MyFoldHandler determine the initial null-eigenvector itself
	void Problem::activate_my_fold_tracking(double *const &parameter_pt, const bool &block_solve)
	{
		reset_assembly_handler_to_default();
		this->assembly_handler_pt() = new MyFoldHandler(this, parameter_pt);
		if (block_solve) refuse_block_solve("Fold tracking");
	}

	// Installs the Hopf-bifurcation tracking assembly handler (bordered system with complex null vector
	// null_real+i*null_imag and angular frequency omega); optionally with the block Hopf linear solver.
	void Problem::activate_my_hopf_tracking(double *const &parameter_pt, const double &omega, const oomph::DoubleVector &null_real, const oomph::DoubleVector &null_imag, const bool &block_solve)
	{
		reset_assembly_handler_to_default();
		this->assembly_handler_pt() = new MyHopfHandler(this, parameter_pt, omega, null_real, null_imag);
		if (block_solve) refuse_block_solve("Hopf tracking");
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
		// The assembly rebuilds resdv on the distribution oomph-lib picks for the Jacobian, which under
		// mpirun is distributed even when the problem is not, so resdv holds nrow_local() doubles and
		// indexing it over the global dof range would read past the end of the buffer. The caller wants
		// the whole vector.
		std::vector<double> res;
		gather_double_vector_to_global(resdv, res);
		res.resize(this->ndof());
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
			// The "lists" variant filters a second time during compression, when merging duplicate
			// column indices, at a point where the elemental (i,j) is out of scope and the structural
			// mask therefore cannot be consulted. Feeding it explicit zeros there makes it emit the same
			// column index twice. Rather than silently producing an unusable pattern (or a corrupt
			// matrix), refuse the combination - "lists" is neither the default nor the fastest variant.
			if (keep_structural_zeros && prune_structural_zeros_by_field_coupling)
			{
				throw_runtime_error("The 'lists' sparse assembly method cannot be combined with the pruned structural "
									"sparsity pattern. Either choose another assembly method (e.g. the default "
									"'vectors_of_pairs'), or set prune_structural_zeros_by_field_coupling=False.");
			}
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

	// Per-matrix version of oomph-lib's "is this entry small enough to not store?" threshold, called from
	// every sparse assembly routine (serial and distributed) once per candidate entry.
	//
	// oomph-lib filters with "if (fabs(value) > threshold)", and the threshold defaults to 0.0 with a
	// strict comparison, so exact zeros are dropped and the emitted pattern follows the dof values.
	// Returning a NEGATIVE threshold makes that test unconditionally true, which turns the routine into a
	// structural assembly - the pattern becomes the union of all elemental blocks, i.e. a function of the
	// equation numbering alone - without touching a single assembly loop.
	//
	// This is deliberately per matrix, because a multi-matrix assembly builds matrices with very
	// different sparsity. The eigenproblem assembly passes matrix 0 = Jacobian and matrix 1 = mass
	// matrix, and the mass matrix is typically ~3x sparser than the Jacobian (only fields carrying a
	// time derivative contribute to it at all). Giving it the Jacobian's connectivity pattern would
	// inflate it threefold, for no benefit: what makes a stable pattern worth having is that the linear
	// solver can reuse a symbolic factorisation, and the operator it factorises is J (or J - sigma*M,
	// whose pattern is J's, since M's entries are a subset of J's structural pattern). So the mass matrix
	// keeps its own, much tighter pattern unless explicitly asked otherwise.
	double Problem::numerical_zero_for_sparse_assembly(const unsigned &matrix_index) const
	{
		// -1.0 rather than -inf or -eps: any negative value makes the test always true, and a plain
		// -1.0 stays readable in a debugger.
		// When the field-coupling mask is in use it decides which structural zeros to keep, entry by
		// entry, and this threshold must stay at its normal value so that everything the mask does NOT
		// mark is still filtered by value. Blanket-keeping zeros here would make the mask pointless.
		if (prune_structural_zeros_by_field_coupling) return Numerical_zero_for_sparse_assembly;
		if (matrix_index == 0)
		{
			if (keep_structural_zeros) return -1.0;
		}
		else if (keep_structural_zeros_in_secondary_matrices)
		{
			return -1.0;
		}
		return Numerical_zero_for_sparse_assembly;
	}

	// Builds the structural sparsity mask for one element's contribution to matrix matrix_index, from
	// the symbolic field-coupling tables the code generator emits (contributes_to_jacobian /
	// contributes_to_mass_matrix, indexed by contribution class) combined with the element's own
	// local-dof-to-contribution-class map. Returning NULL means "no mask", i.e. fall back to the plain
	// value filter - always a safe answer, at the cost of a pattern that is not value-independent.
	//
	// Everything below is a guard establishing that the local dof indexing the mask would be written in
	// really is the element's own. It is not paranoia: an augmenting assembly handler (fold, Hopf,
	// periodic orbit, ...) presents a larger elemental block over its own augmented dof numbering, for
	// which the element's contribution map says nothing.
	const char *Problem::sparsity_mask_for_element(const unsigned &matrix_index, oomph::GeneralisedElement *const &elem_pt, const unsigned &nvar)
	{
		if (!keep_structural_zeros || !prune_structural_zeros_by_field_coupling) return NULL;
		if (!nvar) return NULL; // Element contributes nothing; there is no block to describe
		// Tier A (no pruning) is expressed through the threshold instead, so no mask is needed there.
		if (matrix_index != MASK_HESSIAN && matrix_index != MASK_MASS && matrix_index > 0 && !dynamic_cast<oomph::EigenProblemHandler *>(this->assembly_handler_pt()))
		{
			// Only the eigenproblem assembly has a known meaning for a secondary matrix (index 1 = mass
			// matrix). For anything else - e.g. the several matrices of a multi-assembly - we have no
			// table to describe it, so do not guess.
			return NULL;
		}
		BulkElementBase *be = dynamic_cast<BulkElementBase *>(elem_pt);
		if (!be) return NULL;
		// The augmented dispatch must come BEFORE the "no active residual" shortcut inside the core. An
		// element with no residual contributes nothing to the BASE Jacobian, so an all-zero mask is right
		// for its raw block -- but a bifurcation handler still writes the border of the augmented block
		// for every element regardless (the fold normalisation row is Phi/Count over the element's dofs,
		// a property of the handler, not of the element's residual). Taking the shortcut first produced
		// an all-zero mask for the whole augmented block and lost exactly those border entries.
		{
			const std::vector<int> &cidx_early = be->get_local_dof_contribution_indices();
			if (cidx_early.size() != nvar)
			{
				return augmented_sparsity_mask_for_element(matrix_index, elem_pt, nvar, (unsigned)cidx_early.size());
			}
		}

		// Which scratch buffer this call writes into. MUST be computed before any use: MASK_HESSIAN and
		// MASK_MASS are huge sentinel values, and indexing the buffer vector by them directly asks for
		// ~4 billion entries -- which is exactly what happened, surfacing as a std::bad_alloc in an
		// ordinary continuation step rather than anywhere near the sparsity code.
		const unsigned scratch_index = (matrix_index == MASK_HESSIAN ? 2u : (matrix_index == MASK_MASS ? 3u : matrix_index));
		if (sparsity_mask_scratch.size() <= scratch_index) sparsity_mask_scratch.resize(scratch_index + 1);
		std::vector<char> &scratch = sparsity_mask_scratch[scratch_index];
		if (!this->field_coupling_mask_for_element(matrix_index, -1, elem_pt, nvar, scratch)) return NULL;
		return scratch.data();
	}


	// See the declaration. Fills `out` with the field-coupling mask of ONE (matrix, residual) pair.
	bool Problem::field_coupling_mask_for_element(unsigned matrix_index, int residual_override, oomph::GeneralisedElement *const &elem_pt, const unsigned &nvar, std::vector<char> &out)
	{
		BulkElementBase *be = dynamic_cast<BulkElementBase *>(elem_pt);
		if (!be) return false;
		const JITFuncSpec_Table_FiniteElement_t *ft = be->get_jit_code()->get_func_table();
		const int resind = (residual_override >= 0 ? residual_override : get_current_res_jac(ft));
		if (resind >= 0 && (unsigned)resind >= ft->num_res_jacs) return false; // Not a residual this code has
		if (resind < 0)
		{
			// This code has no contribution to the residual currently being solved, so its assembly
			// routine returns before writing anything: the element's block is identically zero. An
			// all-zero mask says exactly that. Returning false ("no symbolic description") instead would
			// be equally correct for the OR-filter, but would deny the frozen-sparsity path a complete
			// description of the mesh and force it to give up on the whole problem.
			out.assign((size_t)nvar * nvar, 0);
			return true;
		}
		if (!ft->contributes_to_jacobian || !ft->contributes_to_mass_matrix) return false;
		if (matrix_index == MASK_HESSIAN && !ft->contributes_to_hessian) return false;
		bool **table = (matrix_index == MASK_HESSIAN ? ft->contributes_to_hessian[resind]
													 : (matrix_index == 0 ? ft->contributes_to_jacobian[resind]
																		  : ft->contributes_to_mass_matrix[resind]));
		if (!table) return false;
		const unsigned nclass = ft->contribution_entries_size;
		const std::vector<int> &cidx = be->get_local_dof_contribution_indices();
		if (cidx.size() != nvar) return false;

		out.assign((size_t)nvar * nvar, 0);
		for (unsigned i = 0; i < nvar; i++)
		{
			const int fi = cidx[i];
			char *row = &out[(size_t)i * nvar];
			for (unsigned j = 0; j < nvar; j++)
			{
				const int fj = cidx[j];
				// -2 is a POSITIVE statement: the field takes part in no contribution of this code, so
				// this element writes nothing in its row or column and there is nothing to store. -1 is
				// the absence of information ("not attributed") and must be read as coupled to
				// everything, since under-reporting would drop real entries. Conflating the two put a
				// structural zero on the DIAGONAL of every unclassifiable dof, which on a saddle-point
				// system is what invites MUMPS onto a null pivot (see section 7c).
				if (fi == -2 || fj == -2)
					row[j] = 0;
				else if (fi < 0 || fj < 0 || (unsigned)fi >= nclass || (unsigned)fj >= nclass)
					row[j] = 1;
				else
					row[j] = (table[fi][fj] ? 1 : 0);
			}
			if (matrix_index == 0 && diagonal_entries_are_forced()) row[i] = 1;
		}
		return true;
	}


	// Builds the mask of an AUGMENTED elemental block by tiling the raw ones.
	//
	// A bifurcation-tracking handler presents a block several times larger than the element's dof count
	// and fills it by hand, so the field description cannot describe it -- but it is built out of blocks
	// the field description CAN describe. The handler says which (see AugmentedBlockSpec) and this
	// assembles the result. Returns NULL, so the caller falls back exactly as before, whenever the
	// handler declines or the spec does not fit the block it is describing.
	const char *Problem::augmented_sparsity_mask_for_element(const unsigned &matrix_index, oomph::GeneralisedElement *const &elem_pt, const unsigned &nvar, unsigned raw_nvar)
	{
		// Off by default; set problem._use_frozen_sparsity_for_bifurcation_tracking = True to enable.
		const bool dbg = getenv("PYOOMPH_DBG_AUG") != 0;
		auto decline = [&](const char *why) -> const char * { if (dbg) std::cout << "AUGDECLINE " << why << std::endl; return NULL; };
		if (!use_frozen_sparsity_for_bifurcation_tracking) return decline("switch off");
		if (matrix_index > 0) return decline("matrix_index>0");
		auto *provider = dynamic_cast<AugmentedSparsityProvider *>(this->assembly_handler_pt());
		if (!provider) return decline("handler provides no spec");
		if (!raw_nvar) return decline("raw_nvar==0");
		AugmentedBlockSpec spec;
		if (!provider->get_sparsity_pattern(elem_pt, spec)) return decline("get_sparsity_pattern declined");

		const unsigned ng = spec.n_groups();
		if (!ng) return decline("empty spec");
		// Where each group starts in the augmented numbering, and check the total matches the block.
		std::vector<unsigned> gstart(ng + 1, 0);
		for (unsigned g = 0; g < ng; g++) gstart[g + 1] = gstart[g] + (spec.group_is_scalar[g] ? 1u : raw_nvar);
		if (gstart[ng] != nvar) { if (dbg) std::cout << "AUGDECLINE layout " << gstart[ng] << " vs nvar " << nvar << " raw " << raw_nvar << std::endl; return NULL; }

		BulkElementBase *be = dynamic_cast<BulkElementBase *>(elem_pt);
		if (!be) return decline("not a BulkElementBase");
		const bool have_hessian_code = be->get_jit_code()->get_func_table()->hessian_generated;

		// Collect the distinct (matrix, residual) pairs the spec refers to BEFORE filling any of them, so
		// that the buffer indices stay valid while several are held at once. A tracker can mix residuals
		// within one augmented block -- azimuthal uses the base plus the real and imaginary azimuthal
		// residuals -- so the pair, not the matrix alone, is the key.
		std::vector<std::pair<unsigned, int>> keys;
		auto key_index = [&keys](unsigned mat, int resid) -> int {
			for (size_t k = 0; k < keys.size(); k++)
				if (keys[k].first == mat && keys[k].second == resid) return (int)k;
			keys.push_back(std::make_pair(mat, resid));
			return (int)keys.size() - 1;
		};
		auto matrix_of = [](AugmentedBlockSpec::Kind kd) -> unsigned {
			switch (kd)
			{
			case AugmentedBlockSpec::Jacobian:
			case AugmentedBlockSpec::JacobianT: return 0u;
			case AugmentedBlockSpec::MassMatrix:
			case AugmentedBlockSpec::MassMatrixT: return Problem::MASK_MASS;
			default: return Problem::MASK_HESSIAN; // Hessian / HessianT
			}
		};
		for (unsigned gr = 0; gr < ng; gr++)
		{
			for (unsigned gc = 0; gc < ng; gc++)
			{
				for (const auto &t : spec.terms_at(gr, gc))
				{
					if (t.kind == AugmentedBlockSpec::Empty || t.kind == AugmentedBlockSpec::Dense ||
						t.kind == AugmentedBlockSpec::Diagonal) continue; // Need no coupling table
					if ((t.kind == AugmentedBlockSpec::Hessian || t.kind == AugmentedBlockSpec::HessianT) && !have_hessian_code)
					{
						// No generated Hessian means no coupling table, and the handler finite-differences
						// the second derivatives instead. Refuse the whole augmented pattern rather than
						// guess: the Jacobian bounds an ANALYTIC Hessian, but nothing bounds what a
						// finite-differenced one writes, so the frozen path has no business here. Falling
						// back costs the speedup and nothing else.
						return decline("Hessian needed but none generated");
					}
					key_index(matrix_of(t.kind), t.residual);
				}
			}
		}

		if (augmented_raw_masks.size() < keys.size()) augmented_raw_masks.resize(keys.size());
		for (size_t k = 0; k < keys.size(); k++)
		{
			if (!this->field_coupling_mask_for_element(keys[k].first, keys[k].second, elem_pt, raw_nvar,
													   augmented_raw_masks[k].data))
			{ if (dbg) std::cout << "AUGDECLINE raw mask (matrix " << (int)keys[k].first << ", residual " << keys[k].second << ")" << std::endl; return NULL; }
			augmented_raw_masks[k].matrix = keys[k].first;
			augmented_raw_masks[k].residual = keys[k].second;
		}

		if (sparsity_mask_scratch.size() <= matrix_index) sparsity_mask_scratch.resize(matrix_index + 1);
		std::vector<char> &scratch = sparsity_mask_scratch[matrix_index];
		scratch.assign((size_t)nvar * nvar, 0);

		for (unsigned gr = 0; gr < ng; gr++)
		{
			for (unsigned gc = 0; gc < ng; gc++)
			{
				const std::vector<AugmentedBlockSpec::Term> &terms = spec.terms_at(gr, gc);
				if (terms.empty()) continue;
				const bool scalar_pair = spec.group_is_scalar[gr] || spec.group_is_scalar[gc];
				const unsigned nr = spec.group_is_scalar[gr] ? 1u : raw_nvar;
				const unsigned nc = spec.group_is_scalar[gc] ? 1u : raw_nvar;
				for (const auto &t : terms)
				{
					if (t.kind == AugmentedBlockSpec::Empty) continue;
					// A scalar group has no raw indexing to inherit, so only Dense makes sense there.
					if (scalar_pair && t.kind != AugmentedBlockSpec::Dense) return decline("non-Dense kind on a scalar group");
					if (t.kind == AugmentedBlockSpec::Diagonal)
					{
						// Entry (i,i) of the block only. Used where an identity is written -- an azimuthal
						// row overwritten by a boundary condition, or the periodic orbit's wrap-around
						// term u(nT-1) - u(0), which lands on the DIAGONAL of an OFF-DIAGONAL block. So
						// this is not restricted to gr == gc; it only needs both groups to be raw-sized.
						if (nr == nc)
							for (unsigned i = 0; i < nr; i++) scratch[(size_t)(gstart[gr] + i) * nvar + gstart[gc] + i] = 1;
						continue;
					}
					const std::vector<char> *src = 0;
					const bool transposed = (t.kind == AugmentedBlockSpec::JacobianT ||
											 t.kind == AugmentedBlockSpec::MassMatrixT ||
											 t.kind == AugmentedBlockSpec::HessianT);
					if (t.kind != AugmentedBlockSpec::Dense)
						src = &augmented_raw_masks[key_index(matrix_of(t.kind), t.residual)].data;
					for (unsigned i = 0; i < nr; i++)
					{
						char *row = &scratch[(size_t)(gstart[gr] + i) * nvar + gstart[gc]];
						for (unsigned j = 0; j < nc; j++)
						{
							// OR: several terms can contribute to one block.
							if (src ? (*src)[transposed ? (size_t)j * raw_nvar + i : (size_t)i * raw_nvar + j] : 1)
								row[j] = 1;
						}
					}
				}
			}
		}
		return scratch.data();
	}


	void Problem::set_prune_structural_zeros_by_field_coupling(bool yesno)
	{
		if (yesno == prune_structural_zeros_by_field_coupling) return;
		prune_structural_zeros_by_field_coupling = yesno;
		Sparse_assemble_with_arrays_previous_allocation.resize(0);
		this->invalidate_jacobian_structure();
	}

	void Problem::set_force_jacobian_diagonal_entries(bool yesno)
	{
		const bool before = diagonal_entries_are_forced();
		force_jacobian_diagonal_entries = yesno;
		force_jacobian_diagonal_entries_auto = false; // An explicit answer overrides the solver's
		if (diagonal_entries_are_forced() == before) return;
		Sparse_assemble_with_arrays_previous_allocation.resize(0);
		this->invalidate_jacobian_structure();
	}

	void Problem::set_force_jacobian_diagonal_entries_auto()
	{
		if (force_jacobian_diagonal_entries_auto) return;
		const bool before = diagonal_entries_are_forced();
		force_jacobian_diagonal_entries_auto = true;
		if (diagonal_entries_are_forced() == before) return;
		Sparse_assemble_with_arrays_previous_allocation.resize(0);
		this->invalidate_jacobian_structure();
	}

	// Called from the Python solver layer before assembling. Changing the answer changes the sparsity
	// pattern, so it has to invalidate exactly like an explicit change would -- otherwise a solver that
	// starts needing the diagonal would silently keep assembling without it.
	void Problem::set_solver_requires_explicit_diagonal(bool yesno)
	{
		if (yesno == solver_requires_explicit_diagonal) return;
		const bool before = diagonal_entries_are_forced();
		solver_requires_explicit_diagonal = yesno;
		if (diagonal_entries_are_forced() == before) return;
		Sparse_assemble_with_arrays_previous_allocation.resize(0);
		this->invalidate_jacobian_structure();
	}

	// Turns the structural (value-independent) sparsity pattern on or off for the Jacobian / main
	// matrix. See numerical_zero_for_sparse_assembly() above for how it takes effect, and
	// get_jacobian_structure_id() for what it buys.
	void Problem::set_keep_structural_zeros(bool yesno)
	{
		if (yesno == keep_structural_zeros) return;
		if (yesno && prune_structural_zeros_by_field_coupling && Sparse_assembly_method == Perform_assembly_using_lists)
		{
			throw_runtime_error("The 'lists' sparse assembly method cannot be combined with the pruned structural "
								"sparsity pattern. Either choose another assembly method (e.g. the default "
								"'vectors_of_pairs'), or set prune_structural_zeros_by_field_coupling=False.");
		}
		keep_structural_zeros = yesno;
		// The row-length hints from the previous (differently filtered) assembly are now wrong, and the
		// pattern itself has changed, so nothing derived from it may be reused.
		Sparse_assemble_with_arrays_previous_allocation.resize(0);
		this->invalidate_jacobian_structure();
	}

	// Extends the structural pattern to the secondary matrices of a multi-matrix assembly - in practice
	// the mass matrix of the eigenproblem assembly. Off by default: see the note in
	// numerical_zero_for_sparse_assembly() on why the mass matrix does not want the Jacobian's pattern.
	// Worth turning on only if something downstream needs M to have a stable, reusable pattern of its
	// own (e.g. repeatedly re-assembling M for a parameter sweep and updating values in place).
	void Problem::set_keep_structural_zeros_in_secondary_matrices(bool yesno)
	{
		if (yesno == keep_structural_zeros_in_secondary_matrices) return;
		keep_structural_zeros_in_secondary_matrices = yesno;
		Sparse_assemble_with_arrays_previous_allocation.resize(0);
	}

	// Returns the id of the current Jacobian sparsity pattern, re-validating it first. The explicit
	// bump in assign_eqn_numbers() covers everything that renumbers the equations, but the pattern also
	// depends on state that is changed elsewhere: installing or removing a bifurcation-tracking /
	// periodic-orbit assembly handler (which defines its own eqn_number() over an augmented dof
	// vector), adding augmented dofs, and switching the active residual/Jacobian combination (which
	// changes both the field couplings and which fields get pinned). Rather than trusting every one of
	// those call sites to remember to invalidate, we snapshot them here and compare. Cheap: a pointer
	// compare, two integer compares and a short string compare, once per solve.
	unsigned long Problem::get_jacobian_structure_id()
	{
		if (!keep_structural_zeros) return 0; // Pattern is value-dependent; nothing may be reused
		// Identify the configuration rather than counting changes to it. The explicit bump in
		// assign_eqn_numbers() (via invalidate_jacobian_structure) covers renumbering; everything else the
		// pattern depends on is in this key, so a state nobody remembered to hook still gets its own id --
		// and, crucially, coming BACK to a configuration returns the id it had before, which is what lets
		// a cache survive alternation between two assemblies.
		oomph::AssemblyHandler *ah = this->assembly_handler_pt();
		JacobianStructureKey key(std::string(ah ? typeid(*ah).name() : "none"),
								 (unsigned long)this->ndof(), n_unaugmented_dofs, _solved_residual,
								 use_static_condensation, static_condensation_rules_revision);
		auto it = jacobian_structure_ids.find(key);
		if (it != jacobian_structure_ids.end()) return it->second;
		const unsigned long id = ++next_jacobian_structure_id;
		jacobian_structure_ids[key] = id;
		return id;
	}

	// Elemental-assembly-only timing loop, see the declaration in problem.hpp. Deliberately mirrors the
	// element loop of sparse_assemble_row_or_column_compressed_base_problem() (halo skipping, resizing,
	// get_all_vectors_and_matrices) but throws the results away, so the difference to a full assembly
	// isolates the scatter + CSR compression cost.
	double Problem::benchmark_elemental_assembly(unsigned n_repeat, bool with_jacobian, bool with_mass_matrix)
	{
		if (n_repeat == 0) return 0.0;
		oomph::AssemblyHandler *const assembly_handler_pt = this->assembly_handler_pt();
		const unsigned long n_elements = mesh_pt()->nelement();
		const unsigned n_matrix = (with_jacobian ? (with_mass_matrix ? 2 : 1) : 0);
		oomph::Vector<oomph::Vector<double>> el_residuals(1);
		oomph::Vector<oomph::DenseMatrix<double>> el_jacobian(n_matrix);
		double t_start = oomph::TimingHelpers::timer();
		for (unsigned r = 0; r < n_repeat; r++)
		{
			// One pass PER REPEAT, so each repeat costs what one real assembly sweep costs; one pass
			// around the whole loop would have the first repeat pay for all of them.
			HangInterpPassScope __hang_pass;
			for (unsigned long e = 0; e < n_elements; e++)
			{
				oomph::GeneralisedElement *elem_pt = mesh_pt()->element_pt(e);
#ifdef OOMPH_HAS_MPI
				if (elem_pt->is_halo()) continue;
#endif
				const unsigned nvar = assembly_handler_pt->ndof(elem_pt);
				el_residuals[0].resize(nvar);
				for (unsigned m = 0; m < n_matrix; m++) el_jacobian[m].resize(nvar);
				if (n_matrix)
				{
			assembly_handler_pt->get_all_vectors_and_matrices(elem_pt, el_residuals, el_jacobian);
				}
				else
				{
					assembly_handler_pt->get_residuals(elem_pt, el_residuals[0]);
				}
			}
		}
		return (oomph::TimingHelpers::timer() - t_start) / n_repeat;
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
	void Problem::start_bifurcation_tracking(const std::string param, const std::string typus, const bool &blocksolve, const std::vector<double> &eigenv1, const std::vector<double> &eigenv2, const double &omega, std::map<std::string, std::string> special_residual_forms, const std::string eigenvector_scaling)
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
		// The guesses arrive as full-length arrays, identical on every rank (SLEPc eigenvectors are
		// scattered toAll on the Python side). Build the DoubleVectors NON-distributed: operator[]
		// indexes LOCAL rows, so filling ndof() entries into a vector on the distributed dof
		// distribution would overrun its nrow_local()-sized buffer. The handler constructors index
		// these guesses by GLOBAL equation number and scatter the owned rows themselves.
		oomph::LinearAlgebraDistribution ev_dist(this->communicator_pt(), this->ndof(), false);
		oomph::DoubleVector ev1(&ev_dist);
		for (unsigned i = 0; i < std::min((size_t)eigenv1.size(), (size_t)this->ndof()); i++)
		{
			ev1[i] = eigenv1[i];
		}
		oomph::DoubleVector ev2(&ev_dist);
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

		// Post-construction, like set_global_equations_forced_zero above: rescale the eigenvector
		// guess by its largest entry and move the normalisation constraint's right-hand side with it,
		// so that the eigenvector unknowns stay O(1) on a large problem instead of O(1/sqrt(ndof)).
		if (eigenvector_scaling == "auto")
		{
			auto *handler = this->assembly_handler_pt();
			if (auto *h = dynamic_cast<MyFoldHandler *>(handler)) h->apply_maxabs_normalization();
			else if (auto *h = dynamic_cast<MyHopfHandler *>(handler)) h->apply_maxabs_normalization();
			else if (auto *h = dynamic_cast<MyPitchForkHandler *>(handler)) h->apply_maxabs_normalization();
			else if (auto *h = dynamic_cast<AzimuthalSymmetryBreakingHandler *>(handler)) h->apply_maxabs_normalization();
			else throw_runtime_error("eigenvector_scaling='auto' is not implemented for bifurcation type " + typus);
		}
		else if (eigenvector_scaling != "unit")
			throw_runtime_error("Unknown eigenvector_scaling '" + eigenvector_scaling + "': expected 'unit' or 'auto'");
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
						if (static_cast<pyoomph::Node *>(n)->variable_position_pt()->is_pinned(iv))
							static_cast<pyoomph::Node *>(n)->variable_position_pt()->set_value(t,iv, inp[pos++]);
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
			// Back to pyoomph's tee, not to std::cout: restoring the raw stream used to silently
			// detach oomph-lib's output from the log file (and now from the MPI console filter) for
			// the rest of the run, for every problem that had ever been quiet.
			oomph::oomph_info.stream_pt() = pyoomph::get_console_stream();
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
		// Both 'dir' and the result are indexed by GLOBAL equation number, the same contract as
		// Problem::get_residuals(), which is what the callers pair this with.
		//
		// It used to demand dir.size() == nrow_local() while indexing dir[jG] by a global equation
		// number, size the result by ndof() but never reduce it across ranks, and count halo elements
		// twice. That is correct exactly where nrow_local() == ndof() -- serially, and under a plain
		// replicated mpirun -- and a heap overrun plus a wrong answer under --distribute. Same shape
		// as the Problem::set_history_dofs overrun (see dev_docs/floquet_multipliers.md section 8).
		if (dir.size() != this->ndof())
			throw_runtime_error("Mismatch in size of dir vector and the number of DoFs");

		HangInterpPassScope __hang_pass; // see Problem::get_residuals
		const unsigned long n_elements = mesh_pt()->nelement();

#ifdef OOMPH_HAS_MPI
		if (this->distributed())
		{
			// Owned rows plus halos, so an equation shared with another rank can be accumulated here
			// and summed there; halo ELEMENTS are skipped so nothing is counted twice.
			oomph::DoubleVectorWithHaloEntries res;
			res.build(this->dof_distribution_pt(), 0.0);
			res.build_halo_scheme(this->GetHaloSchemePt());
			for (unsigned int ne = 0; ne < n_elements; ne++)
			{
				BulkElementBase *elem_pt = dynamic_cast<BulkElementBase *>(mesh_pt()->element_pt(ne));
				if (!elem_pt || elem_pt->is_halo()) continue;
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
							res.global_value(iG) += hval * dir[jG] * dir[kG];
						}
					}
				}
			}
			res.sum_all_halo_and_haloed_values();
			std::vector<double> result;
			gather_double_vector_to_global(res, result);
			return result;
		}
#endif

		std::vector<double> result(this->ndof(), 0.0);
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
						result[iG] += hval * dir[jG] * dir[kG];
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
		HangInterpPassScope __hang_pass; // see Problem::get_residuals
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
						// Deliberately the raw threshold, NOT numerical_zero_for_sparse_assembly(): this is
						// a rank-3 tensor, so keeping structural zeros would store every (i,j,k) triple of
						// an element - nvar^3, i.e. ~700k entries per element for 3D Taylor-Hood - where the
						// matrices only pay nvar^2. A stable Hessian pattern buys nothing anyway, since
						// nothing factorises it.
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
							if (std::fabs(value) > numerical_zero_for_sparse_assembly(0))
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
		// The frozen path is tried FIRST, including for periodic orbits. It used to sit behind the
		// orbit branch, which meant the orbit assembly could never be frozen no matter what the handler
		// described -- and that assembly is 64-74% of an orbit solve (see section 7e and
		// tests/benchmarks/bench_periodic_orbit_1d.py). assemble_with_frozen_sparsity declines by itself
		// whenever it cannot apply, so the orbit routine below is still the fallback.
		if (this->assemble_with_frozen_sparsity(column_or_row_index,row_or_column_start,value,nnz,residual,compressed_row_flag))
		{
			// Assembled straight into the preallocated pattern; nothing else to do.
		}
		else
		{
			// Only the frozen route is threaded, so say so once rather than leaving --omp looking as if
			// it had simply bought nothing.
			if (num_threads > 1)
				report_parallel_refusal("this assembly has no frozen sparsity pattern (the periodic-orbit "
										"and map-based routes are not threaded)");
			if (dynamic_cast<PeriodicOrbitHandler*>(this->assembly_handler_pt()))
				sparse_assemble_row_or_column_compressed_for_periodic_orbit(column_or_row_index,row_or_column_start,value,nnz,residual,compressed_row_flag);
			else
				oomph::Problem::sparse_assemble_row_or_column_compressed(column_or_row_index,row_or_column_start,value,nnz,residual,compressed_row_flag);
		}

	}

	void Problem::set_use_frozen_sparsity(bool yesno)
	{
		if (yesno == use_frozen_sparsity) return;
		use_frozen_sparsity = yesno;
		frozen_sparsity_cache.clear();
	}

	// Returns the cache slot holding a usable pattern for (matrix_index, generation), building it if the
	// cache does not already have one. `pinned` holds the slots the current assembly has already taken,
	// which must survive eviction. Returns -1 if the pattern cannot be built at all.
	//
	// Why a cache rather than one slot per matrix index: a workflow can legitimately alternate between
	// two different patterns -- a PETSc preconditioner matrix assembled from another residual, or any
	// multi-residual problem -- and with a single slot each assembly would evict the other's pattern and
	// rebuild, which costs two passes over the mesh plus a sort. That is worse than no cache at all.
	int Problem::acquire_frozen_sparsity(unsigned matrix_index, unsigned long generation, unsigned nd, const std::vector<int> &pinned, unsigned mask_matrix_index_override)
	{
		if (frozen_sparsity_cache_capacity < 1) frozen_sparsity_cache_capacity = 1;
		frozen_sparsity_clock++;

		for (unsigned i = 0; i < frozen_sparsity_cache.size(); i++)
		{
			FrozenSparsity &sp = frozen_sparsity_cache[i];
			if (sp.generation == generation && sp.matrix_index == matrix_index && sp.ndof == nd)
			{
				sp.last_used = frozen_sparsity_clock;
				return (int)i;
			}
		}

		// Miss. Grow the cache if it is still below capacity, else evict the least recently used entry
		// that this assembly is not already using.
		int slot = -1;
		if (frozen_sparsity_cache.size() < frozen_sparsity_cache_capacity)
		{
			frozen_sparsity_cache.push_back(FrozenSparsity());
			slot = (int)frozen_sparsity_cache.size() - 1;
		}
		else
		{
			unsigned long oldest = 0;
			for (unsigned i = 0; i < frozen_sparsity_cache.size(); i++)
			{
				if (std::find(pinned.begin(), pinned.end(), (int)i) != pinned.end()) continue;
				if (slot < 0 || frozen_sparsity_cache[i].last_used < oldest)
				{
					oldest = frozen_sparsity_cache[i].last_used;
					slot = (int)i;
				}
			}
			if (slot < 0)
			{
				// Every entry is in use by this very assembly, i.e. the capacity is below n_matrix.
				// Grow rather than fail; the caller cannot proceed otherwise.
				frozen_sparsity_cache_capacity = (unsigned)frozen_sparsity_cache.size() + 1;
				frozen_sparsity_cache.push_back(FrozenSparsity());
				slot = (int)frozen_sparsity_cache.size() - 1;
			}
		}

		FrozenSparsity &sp = frozen_sparsity_cache[slot];
		const unsigned mask_m = (mask_matrix_index_override == (unsigned)-1 ? matrix_index : mask_matrix_index_override);
		if (!build_frozen_sparsity(mask_m, sp, nd)) { sp.clear(); return -1; }
		sp.generation = generation;
		sp.matrix_index = matrix_index;
		sp.last_used = frozen_sparsity_clock;
		frozen_sparsity_rebuilds++;
		return slot;
	}

	// nnz of the cached pattern for the current pattern id, or 0 if none is held. Diagnostics only, so it
	// deliberately does NOT build one -- asking must not change what is being measured.
	unsigned Problem::get_frozen_sparsity_nnz(unsigned matrix_index)
	{
		const unsigned long gen = this->get_jacobian_structure_id();
		if (!gen) return 0;
		for (const auto &sp : frozen_sparsity_cache)
			if (sp.generation == gen && sp.matrix_index == matrix_index) return sp.nnz();
		return 0;
	}

	// ==================== Static condensation: rule resolution ====================
	// See dev_docs/static_condensation.md. This stage only RESOLVES a declared selection; nothing here
	// is read by the assembly, the solver or the Newton loop yet.

	namespace
	{
		// Where a field name lives in an element's generated code: the space info describing it and its
		// index within that space. Nodal fields (the continuous C1..C2TB spaces) sit in the nodes'
		// value arrays at nodal_offset_basebulk + index; everything else (DL, D0 and the DG spaces)
		// sits in the element's internal Data at internal_offset_new + index.
		struct CondensationFieldLocation
		{
			const JITFuncSpec_Table_FiniteElement_SpaceInfo_t *space = nullptr;
			unsigned index = 0;
			bool is_nodal = false;
		};

		static bool locate_condensation_field(const JITFuncSpec_Table_FiniteElement_t *ft, const std::string &name, CondensationFieldLocation &loc)
		{
			for (unsigned i = 0; i < ft->info_DL.numfields; i++)
				if (name == ft->info_DL.fieldnames[i]) { loc.space = &ft->info_DL; loc.index = i; loc.is_nodal = false; return true; }
			for (unsigned i = 0; i < ft->info_D0.numfields; i++)
				if (name == ft->info_D0.fieldnames[i]) { loc.space = &ft->info_D0; loc.index = i; loc.is_nodal = false; return true; }
			for (unsigned si = 0; si < ft->num_present_dg_spaces; si++)
			{
				const auto *sp = ft->present_dg_spaces[si];
				for (unsigned i = 0; i < sp->numfields_basebulk; i++)
					if (name == sp->fieldnames[i]) { loc.space = sp; loc.index = i; loc.is_nodal = false; return true; }
			}
			for (unsigned si = 0; si < ft->num_present_continuous_spaces; si++)
			{
				const auto *sp = ft->present_continuous_spaces[si];
				for (unsigned i = 0; i < sp->numfields_basebulk; i++)
					if (name == sp->fieldnames[i]) { loc.space = sp; loc.index = i; loc.is_nodal = true; return true; }
			}
			return false;
		}

		// A rule may name a vector field the way the user writes it ("velocity"); the generated code only
		// knows the components ("velocity_x", ...). Exact match wins, so a scalar field whose name happens
		// to be a prefix of nothing is unaffected.
		static std::vector<CondensationFieldLocation> locate_condensation_field_or_components(const JITFuncSpec_Table_FiniteElement_t *ft, const std::string &name)
		{
			std::vector<CondensationFieldLocation> res;
			CondensationFieldLocation loc;
			if (locate_condensation_field(ft, name, loc)) { res.push_back(loc); return res; }
			for (const char *suffix : {"_x", "_y", "_z"})
				if (locate_condensation_field(ft, name + suffix, loc)) res.push_back(loc);
			return res;
		}
	}

	// `part` is "all", "internal", "bubble" or "element_private".
	void Problem::add_static_condensation_rule(Mesh *mesh, const std::string &field, const std::vector<unsigned> &values, const std::string &part)
	{
		StaticCondensationRule::Part p;
		if (part == "all") p = StaticCondensationRule::Part::All;
		else if (part == "internal") p = StaticCondensationRule::Part::Internal;
		else if (part == "bubble") p = StaticCondensationRule::Part::Bubble;
		else if (part == "element_private") p = StaticCondensationRule::Part::ElementPrivate;
		else throw_runtime_error("Unknown static condensation part '" + part + "'. Use one of: all, internal, bubble, element_private");
		if (p != StaticCondensationRule::Part::ElementPrivate && !mesh)
			throw_runtime_error("A static condensation rule for part '" + part + "' needs a mesh");
		if (p == StaticCondensationRule::Part::Bubble && !values.empty())
			throw_runtime_error("A 'bubble' static condensation rule cannot take a value subset: a nodal field has exactly one value per node");
		static_condensation_rules.push_back(StaticCondensationRule(mesh, field, values, p));
		static_condensation_selected.clear();
		static_condensation_n_selected = 0;
		static_condensation_selection_valid = false;
		static_condensation_rules_revision++; // see JacobianStructureKey
		condensation_plan.clear();
	}

	void Problem::clear_static_condensation_rules()
	{
		static_condensation_rules.clear();
		static_condensation_rule_counts.clear();
		static_condensation_selected.clear();
		static_condensation_n_selected = 0;
		static_condensation_selection_valid = false;
		static_condensation_rules_revision++;
		condensation_plan.clear();
	}

	// Resolves every rule to concrete (Data*, value) pairs. Rules union; overlaps are harmless.
	//
	// Deliberately rank-local and communication-free: every decision is taken from data the rank already
	// holds, and a halo element resolves exactly as it does on its owner (same code, same node
	// occurrences within the element, same equation numbers), so a distributed stage can build on this
	// unchanged.
	void Problem::update_static_condensation_selection()
	{
		static_condensation_selected.clear();
		static_condensation_n_selected = 0;
		static_condensation_rule_counts.assign(static_condensation_rules.size(), 0);
		static_condensation_selection_valid = true;
		if (static_condensation_rules.empty()) return;

		// Pinned and constrained values are never selected. A condensed dof must be a genuine unknown of
		// the linear system, and dropping them here rather than later means no downstream stage has to
		// distinguish them. Boundary conditions therefore silently shrink a selection - intended.
		unsigned long *counter = NULL;
		auto select = [&](oomph::Data *d, unsigned v)
		{
			if (!d || v >= d->nvalue()) return;
			if (d->eqn_number(v) < 0) return;
			auto it = static_condensation_selected.find(d);
			if (it == static_condensation_selected.end())
				it = static_condensation_selected.emplace(d, std::vector<bool>(d->nvalue(), false)).first;
			if (!it->second[v]) { it->second[v] = true; static_condensation_n_selected++; }
			(*counter)++;
		};

		// All meshes of the problem, interface meshes included.
		std::vector<oomph::Mesh *> all_meshes;
		const unsigned nsub = this->nsub_mesh();
		if (nsub) { for (unsigned i = 0; i < nsub; i++) all_meshes.push_back(this->mesh_pt(i)); }
		else if (this->mesh_pt()) all_meshes.push_back(this->mesh_pt());

		// The reverse external-data set, built once and only when an element_private rule asks for it:
		// which Data objects some element other than their owner reads. Interface elements adopt the
		// bulk's internal Data exactly this way (InterfaceElementBase::add_required_external_data), and
		// so do interior-facet DG couplings, so this is what "private" has to be measured against.
		std::unordered_set<oomph::Data *> externally_referenced;
		bool need_external_scan = false;
		for (const auto &r : static_condensation_rules)
			if (r.part == StaticCondensationRule::Part::ElementPrivate) need_external_scan = true;
		if (need_external_scan)
		{
			for (oomph::Mesh *m : all_meshes)
			{
				if (!m) continue;
				const unsigned ne = m->nelement();
				for (unsigned ie = 0; ie < ne; ie++)
				{
					oomph::GeneralisedElement *e = m->element_pt(ie);
					if (!e) continue;
					const unsigned nex = e->nexternal_data();
					for (unsigned k = 0; k < nex; k++) externally_referenced.insert(e->external_data_pt(k));
				}
			}
		}

		for (unsigned ir = 0; ir < static_condensation_rules.size(); ir++)
		{
			const StaticCondensationRule &rule = static_condensation_rules[ir];
			counter = &static_condensation_rule_counts[ir];

			if (rule.part == StaticCondensationRule::Part::ElementPrivate)
			{
				for (oomph::Mesh *m : all_meshes)
				{
					// Bulk domains only: an interface element's own internal Data belongs to a facet, not
					// to a cell, and eliminating it is a separate (later) question.
					if (!m || dynamic_cast<InterfaceMesh *>(m)) continue;
					// A rule with a mesh is restricted to that domain (StaticCondensation() added to one
					// domain); one without scans them all. Only the SELECTION is restricted - the
					// externally_referenced scan above stayed problem-wide, since an interface element of
					// any domain may adopt this domain's internal Data and would make it non-private.
					if (rule.mesh && dynamic_cast<Mesh *>(m) != rule.mesh) continue;
					const unsigned ne = m->nelement();
					for (unsigned ie = 0; ie < ne; ie++)
					{
						auto *el = dynamic_cast<BulkElementBase *>(m->element_pt(ie));
						if (!el) continue;
						const unsigned nint = el->ninternal_data();
						for (unsigned k = 0; k < nint; k++)
						{
							oomph::Data *d = el->internal_data_pt(k);
							if (!d || externally_referenced.count(d)) continue;
							for (unsigned v = 0; v < d->nvalue(); v++) select(d, v);
						}
					}
				}
				continue;
			}

			Mesh *m = rule.mesh;
			if (!m) continue;
			const unsigned ne = m->nelement();
			if (!ne) continue; // A rank may legitimately hold no element of this domain

			// Field lookup is per generated code, not per element: all elements of a domain share one,
			// and resolving the name once also means the "unknown field" error is raised once.
			std::map<const JITFuncSpec_Table_FiniteElement_t *, std::vector<CondensationFieldLocation>> located;

			// Bubble nodes are found by how often each node occurs among this mesh's elements: a
			// cell-interior bubble node belongs to exactly one element and lies on no boundary. That also
			// excludes the tet-C2TB face bubbles, which two elements share, without having to name them.
			std::unordered_map<oomph::Node *, unsigned> node_occurrence;
			if (rule.part == StaticCondensationRule::Part::Bubble)
			{
				for (unsigned ie = 0; ie < ne; ie++)
				{
					auto *el = dynamic_cast<BulkElementBase *>(m->element_pt(ie));
					if (!el) continue;
					const unsigned nn = el->nnode();
					for (unsigned n = 0; n < nn; n++) node_occurrence[el->node_pt(n)]++;
				}
			}

			for (unsigned ie = 0; ie < ne; ie++)
			{
				auto *el = dynamic_cast<BulkElementBase *>(m->element_pt(ie));
				if (!el || !el->get_jit_code()) continue;
				const JITFuncSpec_Table_FiniteElement_t *ft = el->get_jit_code()->get_func_table();
				auto found = located.find(ft);
				if (found == located.end())
				{
					std::vector<CondensationFieldLocation> locs = locate_condensation_field_or_components(ft, rule.field);
					if (locs.empty())
						throw_runtime_error("Cannot condense field '" + rule.field + "': no such field on this domain");
					for (const auto &l : locs)
					{
						if (rule.part == StaticCondensationRule::Part::Bubble && !l.is_nodal)
							throw_runtime_error("Field '" + rule.field + "' lives in space '" + std::string(l.space->space_name) + "', which has no nodes - part='bubble' does not apply. Use part='all' to condense the whole elemental field.");
						if (rule.part != StaticCondensationRule::Part::Bubble && l.is_nodal)
							throw_runtime_error("Field '" + rule.field + "' is a continuous nodal field in space '" + std::string(l.space->space_name) + "'; its values are shared between elements and cannot be condensed as a whole. Use part='bubble' to select only the cell-interior bubble nodes.");
					}
					found = located.emplace(ft, locs).first;
				}

				for (const CondensationFieldLocation &l : found->second)
				{
					if (rule.part == StaticCondensationRule::Part::Bubble)
					{
						const std::vector<std::vector<unsigned>> &space_nodes = el->get_nodal_space_index_to_element_index_map();
						const unsigned si = l.space->space_index;
						if (si >= space_nodes.size()) continue;
						for (unsigned i = 0; i < space_nodes[si].size(); i++)
						{
							oomph::Node *nd = el->node_pt(space_nodes[si][i]);
							if (!nd) continue;
							if (node_occurrence[nd] != 1) continue;
							if (nd->is_on_boundary()) continue;
							select(nd, l.space->nodal_offset_basebulk + l.index);
						}
					}
					else
					{
						const unsigned di = l.space->internal_offset_new + l.index;
						if (di >= el->ninternal_data()) continue;
						oomph::Data *d = el->internal_data_pt(di);
						if (!d) continue;
						if (rule.values.empty()) { for (unsigned v = 0; v < d->nvalue(); v++) select(d, v); }
						else { for (unsigned v : rule.values) select(d, v); }
					}
				}
			}
		}
	}

	// The selected dofs as global equation numbers. Sorted, so that the order does not depend on the
	// hash-map iteration order (and hence not on allocation addresses) - the same reason the hanging-node
	// masters had to be ordered, see dev_docs/replicated_mpi_correctness.md.
	std::vector<long> Problem::get_static_condensation_dof_eqns()
	{
		ensure_static_condensation_selection();
		std::vector<long> res;
		res.reserve(static_condensation_n_selected);
		for (const auto &kv : static_condensation_selected)
			for (unsigned v = 0; v < kv.second.size(); v++)
				if (kv.second[v]) res.push_back(kv.first->eqn_number(v));
		std::sort(res.begin(), res.end());
		return res;
	}

	// Names for a few global equations, for error messages. Nothing maps an equation number back to a
	// field directly -- the mapping only exists per element, in get_dof_names() -- so this scans the
	// elements until every requested equation has been found. O(nelement) and it builds an element's
	// whole name list per hit, which is why it is deliberately not usable in a loop.
	std::vector<std::string> Problem::describe_global_dofs(const std::vector<long> &eqns)
	{
		std::vector<std::string> res(eqns.size(), "<unknown dof>");
		if (eqns.empty()) return res;
		std::map<long, unsigned> wanted;
		for (unsigned i = 0; i < eqns.size(); i++) wanted[eqns[i]] = i;
		const unsigned long n_element = mesh_pt()->nelement();
		for (unsigned long e = 0; e < n_element && !wanted.empty(); e++)
		{
			BulkElementBase *be = dynamic_cast<BulkElementBase *>(mesh_pt()->element_pt(e));
			if (!be) continue;
			const unsigned nvar = be->ndof();
			std::vector<std::pair<unsigned, unsigned>> hits; // (local dof, index in eqns)
			for (unsigned i = 0; i < nvar; i++)
			{
				auto it = wanted.find(be->eqn_number(i));
				if (it != wanted.end()) hits.push_back(std::make_pair(i, it->second));
			}
			if (hits.empty()) continue;
			const std::vector<std::string> names = be->get_dof_names();
			const JITFuncSpec_Table_FiniteElement_t *ft = be->get_jit_code()->get_func_table();
			const std::string domain = (ft && ft->domain_name ? std::string(ft->domain_name) : std::string("?"));
			for (const auto &h : hits)
			{
				res[h.second] = domain + "/" + (h.first < names.size() ? names[h.first] : std::string("?"));
				wanted.erase(eqns[h.second]);
			}
		}
		return res;
	}

	namespace
	{
		// A short "field__space__localindex (eqn N)" list for an error message. The dof names are LOCAL
		// to an element, so a handful of samples taken from different elements otherwise reads as five
		// copies of "pressure__DL__1"; the equation number is what tells them apart.
		static std::string format_dof_samples(Problem &prob, const std::vector<long> &eqns, unsigned long total)
		{
			const std::vector<std::string> names = prob.describe_global_dofs(eqns);
			std::string res;
			for (unsigned i = 0; i < names.size(); i++)
				res += (i ? ", " : "") + names[i] + " (eqn " + std::to_string(eqns[i]) + ")";
			if (total > eqns.size()) res += ", ... (" + std::to_string(total) + " in total)";
			return res;
		}
	}

	// ==================== Static condensation: the elimination plan ====================
	// Stage 2 of dev_docs/static_condensation.md. Derives, from the frozen FULL pattern alone,
	// everything the numeric kernel will need: the connected components of the selected dofs, the
	// positions in the assembled value array each of them gathers from, the condensed CSR pattern, and
	// where every entry of the condensed matrix comes from. Nothing here reads a single matrix value,
	// and nothing it produces depends on one.
	bool Problem::build_condensation_plan(CondensationPlan &plan)
	{
		plan.clear();
#ifdef OOMPH_HAS_MPI
		// Any MPI run with more than one rank takes an entirely separate builder, whether the problem is
		// distributed or merely replicated: the Jacobian's ROWS are split across ranks either way, so the
		// rows of a component live on one rank while its E rows may not, and the plan is a question of
		// ownership plus three structural exchanges with every refusal a collective vote
		// (dev_docs/static_condensation.md section 9).
		if (Communicator_pt && Communicator_pt->nproc() > 1) return build_distributed_condensation_plan(plan);
#endif
		if (n_unaugmented_dofs != 0) return false; // Augmented (bifurcation/continuation) system: assembled full
		const unsigned long gen = this->get_jacobian_structure_id();
		if (!gen) return false; // Value-dependent pattern; nothing may be precomputed from it
		this->ensure_static_condensation_selection();
		if (!static_condensation_n_selected) return false; // Nothing selected: not an error, just nothing to do
		const unsigned nd = this->ndof();
		if (!nd) return false;

		// The plan is expressed as positions in the frozen pattern's value array, so the frozen assembly
		// path is a precondition rather than an optimisation. If it declines, condensation declines.
		const std::vector<int> pinned;
		const int fslot = this->acquire_frozen_sparsity(0, gen, nd, pinned);
		if (fslot < 0) return false;
		const std::vector<int> &row_start = frozen_sparsity_cache[fslot].row_start;
		const std::vector<int> &col_index = frozen_sparsity_cache[fslot].column_index;
		plan.full_nnz = frozen_sparsity_cache[fslot].nnz();

		// --- 1. The L set, ordered by global equation number so that everything below is deterministic
		std::vector<std::pair<long, std::pair<oomph::Data *, unsigned>>> sel;
		sel.reserve(static_condensation_n_selected);
		for (const auto &kv : static_condensation_selected)
			for (unsigned v = 0; v < kv.second.size(); v++)
				if (kv.second[v]) sel.push_back(std::make_pair(kv.first->eqn_number(v), std::make_pair(kv.first, v)));
		// By equation number only, and stably: the tie-breaker would otherwise be a Data POINTER, i.e.
		// the allocation order (see dev_docs/replicated_mpi_correctness.md on why that must never decide anything).
		std::stable_sort(sel.begin(), sel.end(), [](const std::pair<long, std::pair<oomph::Data *, unsigned>> &a,
													const std::pair<long, std::pair<oomph::Data *, unsigned>> &b)
						 { return a.first < b.first; });

		plan.is_condensed.assign(nd, 0);
		std::vector<int> l_index(nd, -1); // global eqn -> index in the L arrays, -1 for retained dofs
		std::vector<int> l_eqn;
		std::vector<std::pair<oomph::Data *, unsigned>> l_value;
		for (const auto &s : sel)
		{
			if (s.first < 0 || (unsigned long)s.first >= nd) continue;
			const int eq = (int)s.first;
			if (plan.is_condensed[eq]) continue; // two Data values sharing an equation: keep the first
			plan.is_condensed[eq] = 1;
			l_index[eq] = (int)l_eqn.size();
			l_eqn.push_back(eq);
			l_value.push_back(s.second);
		}
		const unsigned nL = (unsigned)l_eqn.size();
		if (!nL) return false;

		// --- 2. Connected components of the structural L x L coupling, symmetrised
		// (an entry (r,c) couples r and c whichever way round the matrix stores it: A_LL has to be
		// inverted as a block, and reducibility is a property of the undirected graph).
		std::vector<int> uf(nL);
		for (unsigned i = 0; i < nL; i++) uf[i] = (int)i;
		auto uf_find = [&uf](int a) { while (uf[a] != a) { uf[a] = uf[uf[a]]; a = uf[a]; } return a; };
		// While walking the rows anyway, record whether each L dof has a structurally present L column
		// (a nonzero row of A_LL) and appears as an L column of some L row (a nonzero column of A_LL).
		std::vector<char> row_ok(nL, 0), col_ok(nL, 0);
		for (unsigned i = 0; i < nL; i++)
		{
			const int r = l_eqn[i];
			for (int k = row_start[r]; k < row_start[r + 1]; k++)
			{
				const int c = col_index[k];
				if (c < 0 || (unsigned)c >= nd || !plan.is_condensed[c]) continue;
				const int j = l_index[c];
				row_ok[i] = 1;
				col_ok[j] = 1;
				const int ra = uf_find((int)i), rb = uf_find(j);
				if (ra != rb) uf[ra] = rb;
			}
		}
		// Component ids in order of their smallest member, so that a rebuild of an unchanged pattern
		// produces an identical plan.
		std::vector<int> comp_of_l(nL, -1), comp_of_root(nL, -1);
		unsigned ncomp = 0;
		for (unsigned i = 0; i < nL; i++)
		{
			const int r = uf_find((int)i);
			if (comp_of_root[r] < 0) comp_of_root[r] = (int)(ncomp++);
			comp_of_l[i] = comp_of_root[r];
		}
		plan.components.resize(ncomp);
		for (unsigned i = 0; i < nL; i++)
		{
			CondensationComponent &comp = plan.components[comp_of_l[i]];
			comp.L_eqns.push_back(l_eqn[i]); // ascending, since i runs in equation order
			comp.L_values.push_back(l_value[i]);
		}

		// --- 3. Size guard. Each component is inverted as a DENSE block, so a selection whose coupling
		// percolates through the mesh (interior-penalty DG velocities, say) is not a condensation but a
		// second, worse, direct solve. Refuse it with something the user can act on.
		for (unsigned c = 0; c < ncomp; c++)
		{
			const CondensationComponent &comp = plan.components[c];
			if (comp.nL() <= static_condensation_max_component_size) continue;
			std::vector<long> sample;
			for (unsigned i = 0; i < comp.nL() && i < 5; i++) sample.push_back(comp.L_eqns[i]);
			const std::string list = format_dof_samples(*this, sample, comp.nL());
			throw_runtime_error("Static condensation: the selected degrees of freedom do not decompose into small "
								"element-local blocks. One connected component holds " + std::to_string(comp.nL()) +
								" mutually coupled dofs, above the limit static_condensation_max_component_size = " +
								std::to_string(static_condensation_max_component_size) +
								". Each component is inverted as a dense block, so this would cost more than the solve "
								"it is meant to shorten. Sample dofs of the component: " + list +
								". Either restrict the selection to dofs that really are element-local, or raise "
								"problem.static_condensation_max_component_size if the size is genuinely intended. "
								"A DG field whose facet terms couple the two sides directly - an interior-penalty "
								"jump/average formulation - is not element-local and percolates through the whole "
								"mesh; a hybridized (HDG) formulation, where the sides talk only through an unknown "
								"on the facet, is, and condenses into one block per element.");
		}

		// --- 4. Structural invertibility. A_LL restricted to a component must have no structurally
		// empty row and no structurally empty column. This is the check that catches selecting a
		// Crouzeix-Raviart pressure on its own: the continuity rows are weak(div(u), p_test) and contain
		// no pressure at all, so those rows of A_LL are identically zero however the values come out.
		{
			std::vector<long> bad_rows, bad_cols;
			unsigned long n_bad_rows = 0, n_bad_cols = 0;
			for (unsigned i = 0; i < nL; i++)
			{
				if (!row_ok[i]) { n_bad_rows++; if (bad_rows.size() < 4) bad_rows.push_back(l_eqn[i]); }
				if (!col_ok[i]) { n_bad_cols++; if (bad_cols.size() < 4) bad_cols.push_back(l_eqn[i]); }
			}
			if (!bad_rows.empty() || !bad_cols.empty())
			{
				std::string msg = "Static condensation: the selected degrees of freedom cannot be eliminated, because "
								  "the block to be inverted is structurally singular.";
				if (!bad_rows.empty())
					msg += " The equations of these selected dofs contain no selected unknown at all (an empty ROW of "
						   "the block): " + format_dof_samples(*this, bad_rows, n_bad_rows) + ".";
				if (!bad_cols.empty())
					msg += " These selected dofs appear in no selected equation (an empty COLUMN of the block): " +
						   format_dof_samples(*this, bad_cols, n_bad_cols) + ".";
				msg += " A dof can only be eliminated together with an equation that determines it. The classic case is "
					   "selecting a discontinuous (DL) pressure on its own: the continuity equation contains no "
					   "pressure, so it has to be condensed jointly with the bubble velocities that it does determine "
					   "-- and the constant pressure mode has to stay in the global system.";
				throw_runtime_error(msg);
			}
		}

		// --- 5. E_C: every retained dof structurally coupled to the component, by column (it appears in
		// one of the component's equations) or by row (its own equation contains one of the component's
		// unknowns). Both directions are needed: the Schur update touches the rows of A_EL and the
		// columns of A_LE, and interface/facet elements make those two sets genuinely different.
		{
			std::vector<std::vector<int>> E_lists(ncomp);
			for (unsigned i = 0; i < nL; i++) // by column
			{
				const int r = l_eqn[i];
				std::vector<int> &lst = E_lists[comp_of_l[i]];
				for (int k = row_start[r]; k < row_start[r + 1]; k++)
				{
					const int c = col_index[k];
					if (c >= 0 && (unsigned)c < nd && !plan.is_condensed[c]) lst.push_back(c);
				}
			}
			for (unsigned r = 0; r < nd; r++) // by row
			{
				if (plan.is_condensed[r]) continue;
				for (int k = row_start[r]; k < row_start[r + 1]; k++)
				{
					const int c = col_index[k];
					if (c < 0 || (unsigned)c >= nd || !plan.is_condensed[c]) continue;
					E_lists[comp_of_l[l_index[c]]].push_back((int)r);
				}
			}
			for (unsigned c = 0; c < ncomp; c++)
			{
				std::vector<int> &lst = E_lists[c];
				std::sort(lst.begin(), lst.end());
				lst.erase(std::unique(lst.begin(), lst.end()), lst.end());
				plan.components[c].E_eqns.swap(lst);
			}
		}

		// --- 6. Gather slots into the full value array. The columns of a row are ascending, so this is a
		// binary search per pair -- once, here, and never again during a solve.
		auto full_slot = [&](int row, int col) -> int
		{
			const int lo = row_start[row], hi = row_start[row + 1];
			const int *b = col_index.data() + lo, *e = col_index.data() + hi;
			const int *f = std::lower_bound(b, e, col);
			if (f == e || *f != col) return -1;
			return lo + (int)(f - b);
		};
		for (unsigned c = 0; c < ncomp; c++)
		{
			CondensationComponent &comp = plan.components[c];
			const unsigned nl = comp.nL(), ne = comp.nE();
			comp.LL_slots.assign((size_t)nl * nl, -1);
			comp.LE_slots.assign((size_t)nl * ne, -1);
			comp.EL_slots.assign((size_t)ne * nl, -1);
			for (unsigned i = 0; i < nl; i++)
			{
				for (unsigned j = 0; j < nl; j++) comp.LL_slots[(size_t)i * nl + j] = full_slot(comp.L_eqns[i], comp.L_eqns[j]);
				for (unsigned j = 0; j < ne; j++) comp.LE_slots[(size_t)i * ne + j] = full_slot(comp.L_eqns[i], comp.E_eqns[j]);
			}
			for (unsigned i = 0; i < ne; i++)
				for (unsigned j = 0; j < nl; j++) comp.EL_slots[(size_t)i * nl + j] = full_slot(comp.E_eqns[i], comp.L_eqns[j]);
		}

		// --- 7. The condensed pattern: full-size and non-renumbered. A retained row keeps its retained
		// columns and gains the whole E_C x E_C block of every component it takes part in (the fill-in of
		// the Schur complement); a condensed row is replaced by its diagonal alone, which the kernel sets
		// to 1 with a zero right-hand side so the solver returns a zero increment there.
		{
			std::vector<std::vector<int>> cond_cols(nd);
			for (unsigned r = 0; r < nd; r++)
			{
				if (plan.is_condensed[r]) { cond_cols[r].push_back((int)r); continue; }
				for (int k = row_start[r]; k < row_start[r + 1]; k++)
				{
					const int c = col_index[k];
					if (c >= 0 && (unsigned)c < nd && !plan.is_condensed[c]) cond_cols[r].push_back(c);
				}
			}
			for (unsigned c = 0; c < ncomp; c++)
			{
				const std::vector<int> &E = plan.components[c].E_eqns;
				for (size_t i = 0; i < E.size(); i++)
					cond_cols[E[i]].insert(cond_cols[E[i]].end(), E.begin(), E.end());
			}
			plan.cond_row_start.assign(nd + 1, 0);
			for (unsigned r = 0; r < nd; r++)
			{
				std::vector<int> &cc = cond_cols[r];
				std::sort(cc.begin(), cc.end());
				cc.erase(std::unique(cc.begin(), cc.end()), cc.end());
				plan.cond_row_start[r + 1] = plan.cond_row_start[r] + (int)cc.size();
			}
			plan.cond_column_index.resize(plan.cond_row_start[nd]);
			for (unsigned r = 0; r < nd; r++)
				std::copy(cond_cols[r].begin(), cond_cols[r].end(), plan.cond_column_index.begin() + plan.cond_row_start[r]);
		}
		const std::vector<int> &crs = plan.cond_row_start;
		const std::vector<int> &cci = plan.cond_column_index;
		auto cond_slot = [&crs, &cci](int row, int col) -> int
		{
			const int lo = crs[row], hi = crs[row + 1];
			const int *b = cci.data() + lo, *e = cci.data() + hi;
			const int *f = std::lower_bound(b, e, col);
			if (f == e || *f != col) return -1;
			return lo + (int)(f - b);
		};

		// --- 8. Where every entry of the condensed matrix comes from.
		// Pass-through: the (E,E) entries the full pattern already had, copied verbatim. Emitted in
		// full-slot order, so the copy sweeps forward through the full value array.
		for (unsigned r = 0; r < nd; r++)
		{
			if (plan.is_condensed[r]) continue;
			for (int k = row_start[r]; k < row_start[r + 1]; k++)
			{
				const int c = col_index[k];
				if (c < 0 || (unsigned)c >= nd || plan.is_condensed[c]) continue;
				const int cs = cond_slot((int)r, c);
				if (cs < 0) throw_runtime_error("Internal error while building the static condensation plan: a retained "
												"matrix entry has no slot in the condensed pattern");
				plan.passthrough_full_slot.push_back(k);
				plan.passthrough_cond_slot.push_back(cs);
			}
		}
		// The fill-in slots, and the identity rows.
		for (unsigned c = 0; c < ncomp; c++)
		{
			CondensationComponent &comp = plan.components[c];
			const unsigned ne = comp.nE();
			comp.fill_slots.assign((size_t)ne * ne, -1);
			for (unsigned i = 0; i < ne; i++)
				for (unsigned j = 0; j < ne; j++)
				{
					const int cs = cond_slot(comp.E_eqns[i], comp.E_eqns[j]);
					if (cs < 0) throw_runtime_error("Internal error while building the static condensation plan: the "
													"fill-in of a component is not contained in the condensed pattern");
					comp.fill_slots[(size_t)i * ne + j] = cs;
				}
		}
		plan.condensed_eqns = l_eqn;
		plan.L_diagonal_cond_slots.resize(nL);
		for (unsigned i = 0; i < nL; i++) plan.L_diagonal_cond_slots[i] = plan.cond_row_start[l_eqn[i]]; // the row's only entry
		return true;
	}

#ifdef OOMPH_HAS_MPI
	// ==================== Static condensation: the distributed plan ====================
	// dev_docs/static_condensation.md section 9. Everything below rests on two properties of the
	// distributed assembly (structural_assembly.md section 5): a rank's OWNED rows are complete, and for
	// a distributed problem the row distribution IS the dof ownership. Together they say that the rank on
	// which a component's element is non-halo holds A_LL, A_LE and r_L in full and is the only one that
	// can form the Schur operators -- while a neighbour owning one of the E rows holds that row complete,
	// including its A_EL entries, and needs nothing but X and y to apply the update to it.

	// Turns a per-rank refusal into one that every rank throws. An empty message means "no objection
	// here". COLLECTIVE: it must be reached by every rank whatever each of them decided, which is the
	// whole point -- a rank throwing on its own would leave the others in the next collective for ever.
	void Problem::collective_throw(const std::string &msg)
	{
		if (!Communicator_pt || Communicator_pt->nproc() < 2)
		{
			if (!msg.empty()) throw_runtime_error(msg);
			return;
		}
		MPI_Comm comm = Communicator_pt->mpi_comm();
		const int my_rank = (int)Communicator_pt->my_rank();
		int mine = msg.empty() ? -1 : my_rank, root = -1;
		MPI_Allreduce(&mine, &root, 1, MPI_INT, MPI_MAX, comm);
		if (root < 0) return;
		int len = (root == my_rank ? (int)msg.size() : 0);
		MPI_Bcast(&len, 1, MPI_INT, root, comm);
		std::vector<char> buf((size_t)len + 1, 0);
		if (root == my_rank) std::copy(msg.begin(), msg.end(), buf.begin());
		if (len) MPI_Bcast(buf.data(), len, MPI_CHAR, root, comm);
		throw_runtime_error("[rank " + std::to_string(root) + "] " + std::string(buf.data(), (size_t)len));
	}

	namespace
	{
		// One all-to-all exchange of variable-length int payloads: an MPI_Alltoall of the counts so the
		// receivers can allocate, then one message per peer. What this rank "sends to itself" never
		// travels; it is copied, so the caller can treat every peer alike.
		static void condensation_alltoall_ints(MPI_Comm comm, unsigned nproc, unsigned my_rank, int tag,
											   const std::vector<std::vector<int>> &send,
											   std::vector<std::vector<int>> &recv)
		{
			std::vector<int> sn(nproc, 0), rn(nproc, 0);
			for (unsigned p = 0; p < nproc; p++) sn[p] = (int)send[p].size();
			MPI_Alltoall(sn.data(), 1, MPI_INT, rn.data(), 1, MPI_INT, comm);
			recv.assign(nproc, std::vector<int>());
			std::vector<MPI_Request> reqs;
			reqs.reserve(2 * nproc);
			for (unsigned p = 0; p < nproc; p++)
			{
				if (p == my_rank) { recv[p] = send[p]; continue; }
				if (sn[p])
				{
					reqs.push_back(MPI_Request());
					MPI_Isend(const_cast<int *>(send[p].data()), sn[p], MPI_INT, (int)p, tag, comm, &reqs.back());
				}
				if (rn[p])
				{
					recv[p].resize(rn[p]);
					reqs.push_back(MPI_Request());
					MPI_Irecv(recv[p].data(), rn[p], MPI_INT, (int)p, tag, comm, &reqs.back());
				}
			}
			if (!reqs.empty()) MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
		}

		// The same exchange for 64-bit payloads. The interior-facet keys need it: a packed refinement
		// path is 3 bits per level, which leaves an int32 with about ten levels of headroom.
		static void facet_alltoall_longs(MPI_Comm comm, unsigned nproc, unsigned my_rank, int tag,
										 const std::vector<std::vector<long>> &send,
										 std::vector<std::vector<long>> &recv)
		{
			std::vector<int> sn(nproc, 0), rn(nproc, 0);
			for (unsigned p = 0; p < nproc; p++) sn[p] = (int)send[p].size();
			MPI_Alltoall(sn.data(), 1, MPI_INT, rn.data(), 1, MPI_INT, comm);
			recv.assign(nproc, std::vector<long>());
			std::vector<MPI_Request> reqs;
			reqs.reserve(2 * nproc);
			for (unsigned p = 0; p < nproc; p++)
			{
				if (p == my_rank) { recv[p] = send[p]; continue; }
				if (sn[p])
				{
					reqs.push_back(MPI_Request());
					MPI_Isend(const_cast<long *>(send[p].data()), sn[p], MPI_LONG, (int)p, tag, comm, &reqs.back());
				}
				if (rn[p])
				{
					recv[p].resize(rn[p]);
					reqs.push_back(MPI_Request());
					MPI_Irecv(recv[p].data(), rn[p], MPI_LONG, (int)p, tag, comm, &reqs.back());
				}
			}
			if (!reqs.empty()) MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
		}
	}

	// The halo/haloed scheme of the interior-facet skeletons (dev_docs/internal_facet_fields.md).
	//
	// A facet between two bulk elements on different ranks EXISTS on both: the near-side element is
	// non-halo on one of them and a halo on the other, and the facet element inherits that flag from it
	// (oomph's build_face_element). Residual assembly is then already right, because every assembly path
	// skips halo elements. What is missing is the numbering: Mesh::assign_global_eqn_numbers() numbers
	// every element's internal Data with no halo test, and nothing marks a halo facet's Data as halo --
	// so a facet unknown gets numbered once per holder, i.e. two independent copies of what is supposed
	// to be one single-valued trace.
	//
	// Marking it fixes both halves at once, because everything downstream is a loop over submeshes and
	// the skeletons are registered as such: oomph's copy_haloed_eqn_numbers_helper() then sends the
	// owner's equation numbers to the holders, and synchronise_dofs() keeps the VALUES in step, both via
	// add_internal_*_to_vector() on the halo/haloed element lists built here.
	//
	// COLLECTIVE, and one exchange for every skeleton of the problem together. The pairing is by a key
	// that both holders can compute alone -- the near-side element's (root index, refinement path) and
	// the face index, all of which the near-side rule has already made partition-independent -- and both
	// sides SORT by it, so the two lists are index-matched by construction with no second round.
	void Problem::setup_interior_facet_halo_scheme()
	{
		if (!Communicator_pt || Communicator_pt->nproc() < 2 || !Problem_has_been_distributed) return;
		const unsigned nproc = Communicator_pt->nproc(), my_rank = Communicator_pt->my_rank();
		MPI_Comm comm = Communicator_pt->mpi_comm();

		// Collective by construction: the equation tree is replicated, so every rank walks the same
		// submeshes in the same order and agrees on the skeleton indices used as the first key component.
		std::vector<InterfaceMesh *> skels;
		std::string nested;
		const unsigned nsub = this->nsub_mesh();
		for (unsigned im = 0; im < nsub; im++)
		{
			InterfaceMesh *m = dynamic_cast<InterfaceMesh *>(this->mesh_pt(im));
			if (!m || m->get_interface_name() != "_internal_facets_") continue;
			// A skeleton OF A SKELETON ("domain/_internal_facets_/_internal_facets_", which is what
			// setting requires_interior_facet_terms on the facet equations themselves produces) has no
			// partition-independent element numbering to key on: its "bulk" elements are face elements
			// built on the fly, not mesh elements that were numbered before the distribution. Refused
			// rather than skipped, since skipping it would leave its unknowns numbered once per holder.
			if (dynamic_cast<InterfaceMesh *>(m->get_bulk_mesh()))
			{
				nested = m->get_full_domain_path();
				continue;
			}
			skels.push_back(m);
		}
		// Collective: every rank has the same equation tree, so all of them find the same nested
		// skeleton or none does.
		if (!nested.empty())
			throw_runtime_error("The interior-facet skeleton '" + nested + "' is itself attached to an "
								"interior-facet skeleton, which is not supported on a distributed "
								"(--distribute) problem: there is no partition-independent numbering of the "
								"facet elements to identify a facet by. This usually means "
								"requires_interior_facet_terms was set on the equations attached to "
								"'_internal_facets_' rather than on the bulk equations.");
		if (skels.empty()) return;
		for (size_t s = 0; s < skels.size(); s++) skels[s]->clear_halo_element_scheme();

		// Is there anything to pair at all? Between a remesh and the re-distribution that follows it the
		// Problem is still flagged as distributed while the meshes are whole again, so no facet is a halo
		// and there is no scheme to build -- and the base element numbers the keys below need have not
		// been re-assigned yet either. Voted rather than assumed: a rank may hold no halo facet while
		// another does, and this decides whether the exchange below happens at all.
		{
			int mine_halo = 0;
			for (size_t s = 0; s < skels.size() && !mine_halo; s++)
				for (unsigned e = 0; e < skels[s]->nelement(); e++)
					if (skels[s]->element_pt(e)->is_halo()) { mine_halo = 1; break; }
			int any_halo = 0;
			MPI_Allreduce(&mine_halo, &any_halo, 1, MPI_INT, MPI_MAX, comm);
			if (!any_halo) return;
		}

		// key = [skeleton index, root element index, packed refinement path, face index]
		const int KEY = 4;
		std::string err;
		std::map<std::vector<long>, oomph::GeneralisedElement *> mine; // the facets this rank owns
		std::vector<std::vector<long>> to_owner(nproc);
		std::vector<std::vector<oomph::GeneralisedElement *>> my_halo_facet(nproc);
		std::vector<std::vector<std::vector<long>>> my_halo_key(nproc);
		for (size_t s = 0; s < skels.size() && err.empty(); s++)
		{
			const unsigned ne = skels[s]->nelement();
			for (unsigned e = 0; e < ne; e++)
			{
				oomph::FaceElement *fe = dynamic_cast<oomph::FaceElement *>(skels[s]->element_pt(e));
				if (!fe) { err = "Internal error: the interior-facet skeleton holds an element that is not a FaceElement."; break; }
				BulkElementBase *near_el = dynamic_cast<BulkElementBase *>(fe->bulk_element_pt());
				std::vector<long> key(KEY, 0);
				key[0] = (long)s;
				if (!near_el || !Mesh::element_structural_key(near_el, key[1], key[2]))
				{
					err = "Interior-facet fields on a distributed problem need the base element numbers, which are "
						  "assigned before the distribution (Mesh::assign_global_base_element_indices). The mesh "
						  "behind '" + skels[s]->get_full_domain_path() + "' has none, so the two holders of a "
						  "shared facet cannot agree on which facet it is.";
					break;
				}
				key[3] = (long)fe->face_index();
				if (!fe->is_halo())
				{
					if (!mine.insert(std::make_pair(key, skels[s]->element_pt(e))).second)
					{
						err = "Internal error: two interior facets of the same mesh carry the same key, so the "
							  "near-side element and face index do not identify a facet uniquely.";
						break;
					}
				}
				else
				{
					const unsigned owner = (unsigned)fe->non_halo_proc_ID();
					if (owner >= nproc) { err = "Internal error: a halo interior facet names an owner outside the communicator."; break; }
					to_owner[owner].insert(to_owner[owner].end(), key.begin(), key.end());
					my_halo_facet[owner].push_back(skels[s]->element_pt(e));
					my_halo_key[owner].push_back(key);
				}
			}
		}
		// Everything above is per-rank; everything below communicates. Agree first.
		collective_throw(err);

		std::vector<std::vector<long>> recv;
		facet_alltoall_longs(comm, nproc, my_rank, 90, to_owner, recv);

		// The HALO side: this rank's facets owned by p, ordered by key.
		for (unsigned p = 0; p < nproc; p++)
		{
			if (p == my_rank || my_halo_facet[p].empty()) continue;
			std::vector<unsigned> order(my_halo_facet[p].size());
			for (unsigned i = 0; i < order.size(); i++) order[i] = i;
			std::sort(order.begin(), order.end(), [&](unsigned a, unsigned b)
					  { return my_halo_key[p][a] < my_halo_key[p][b]; });
			for (unsigned i = 0; i < order.size(); i++)
			{
				oomph::GeneralisedElement *el = my_halo_facet[p][order[i]];
				const long si = my_halo_key[p][order[i]][0];
				// add_root_halo_element_pt() also (re)sets the element's halo flag, which is already what
				// build_face_element gave it; the internal Data is what actually needs marking.
				skels[(size_t)si]->add_root_halo_element_pt(p, el);
				for (unsigned k = 0; k < el->ninternal_data(); k++) el->internal_data_pt(k)->set_halo(p);
			}
		}

		// The HALOED side: the facets p holds a copy of, in the same key order.
		for (unsigned p = 0; p < nproc; p++)
		{
			if (p == my_rank || recv[p].empty()) continue;
			std::vector<std::vector<long>> keys;
			for (size_t i = 0; i + KEY <= recv[p].size(); i += KEY)
				keys.push_back(std::vector<long>(recv[p].begin() + i, recv[p].begin() + i + KEY));
			std::sort(keys.begin(), keys.end());
			for (size_t i = 0; i < keys.size(); i++)
			{
				std::map<std::vector<long>, oomph::GeneralisedElement *>::iterator f = mine.find(keys[i]);
				if (f == mine.end())
				{
					// The two ranks disagree about which side of the facet is the near one, or about the
					// element numbering itself. Either way the lists would be mis-paired and equation
					// numbers would land on the wrong facet, so refuse rather than proceed.
					err = "Interior-facet fields: rank " + std::to_string(p) + " reports a facet this rank does "
						  "not own, so the two do not agree on which element enumerates it. Facet key (skeleton, "
						  "root element, refinement path, face): " + std::to_string(keys[i][0]) + ", " +
						  std::to_string(keys[i][1]) + ", " + std::to_string(keys[i][2]) + ", " + std::to_string(keys[i][3]) + ".";
					break;
				}
				skels[(size_t)keys[i][0]]->add_root_haloed_element_pt(p, f->second);
			}
			if (!err.empty()) break;
		}
		collective_throw(err);

		// The halo copies are stale at this moment and nothing else will refresh them before they are
		// read. The skeleton was just rebuilt from scratch, and restore_discontinuous_data() refitted
		// each rank's facet values from its OWN partial sample cloud -- so a halo facet holds this rank's
		// fit rather than the owner's. A neighbour's non-halo bulk element reads exactly those values,
		// and so does any output() taken between here and the first Newton step.
		this->synchronise_all_dofs();
	}

	// Row cut points that no element-local condensation block straddles; empty for "no preference".
	//
	// Moves the cut points of a uniform nproc-way row split off the blocks they would otherwise fall
	// inside. blocks must be disjoint and ascending. Returns an empty vector when some block is longer
	// than one rank's share, which cannot be stepped over by moving a cut at all -- only by handing a
	// whole rank's worth of rows to somebody else.
	//
	// Shared by the two callers that want a block-aligned split for different reasons: static
	// condensation, whose blocks are the components it inverts, and the dof ordering, whose blocks are
	// the nodal or elemental blocks the layout built. They were the same fifteen lines.
	static std::vector<unsigned> snap_cuts_to_blocks(const std::vector<std::pair<int, int>> &blocks,
													 unsigned nproc, unsigned nd)
	{
		std::vector<unsigned> cuts;
		const unsigned share = nd / nproc;
		for (size_t i = 0; i < blocks.size(); i++)
			if ((unsigned)(blocks[i].second - blocks[i].first + 1) > share) return cuts;

		cuts.assign(nproc + 1, 0);
		cuts[nproc] = nd;
		size_t bi = 0;
		for (unsigned p = 1; p < nproc; p++)
		{
			unsigned c = (unsigned)((unsigned long)nd * p / nproc);
			if (c < cuts[p - 1]) c = cuts[p - 1];
			while (bi < blocks.size() && (unsigned)blocks[bi].second < c) bi++;
			// One step is enough: the blocks are disjoint, so blocks[bi].second+1 is at most
			// blocks[bi+1].first, which is not strictly inside blocks[bi+1].
			if (bi < blocks.size() && (unsigned)blocks[bi].first < c && c <= (unsigned)blocks[bi].second)
				c = (unsigned)blocks[bi].second + 1;
			cuts[p] = (c > nd ? nd : c);
		}
		return cuts;
	}

	// Only a REPLICATED run needs this. Distributed, oomph renumbers so that each rank's own dofs are
	// one contiguous global range (synchronise_eqn_numbers), so a block of dofs belonging to one
	// element is inside one rank's range by construction. Replicated, the numbering is the serial one
	// and the rows are cut uniformly, which lands a cut inside an element's block essentially always --
	// one per rank boundary, and the plan builder then refuses every such component.
	std::vector<unsigned> Problem::condensation_row_cuts()
	{
		std::vector<unsigned> cuts;
		if (!use_static_condensation) return cuts;
		if (!Communicator_pt || Communicator_pt->nproc() < 2) return cuts;
		if (Problem_has_been_distributed) return cuts;
		const unsigned nproc = Communicator_pt->nproc();
		const unsigned nd = this->ndof();
		if (nd < nproc) return cuts;

		this->ensure_static_condensation_selection();
		std::vector<int> sel;
		for (const auto &kv : static_condensation_selected)
			for (unsigned v = 0; v < kv.second.size(); v++)
			{
				if (!kv.second[v]) continue;
				const long eq = kv.first->eqn_number(v);
				if (eq >= 0) sel.push_back((int)eq);
			}
		if (sel.empty()) return cuts;
		std::sort(sel.begin(), sel.end());
		sel.erase(std::unique(sel.begin(), sel.end()), sel.end());

		// The components, from the SYMBOLIC MASK rather than from mere co-membership of an element. The
		// difference is decisive here: an HDG facet element holds the unknowns of both neighbouring
		// bulk elements, but its residual never differentiates the near side with respect to the far one
		// (that is what "hybridizable" means, and what the @opposite contribution classes express), so
		// the two are NOT in one component. Reading the element's dof list alone would chain every
		// element of the mesh into a single block and give up below.
		oomph::AssemblyHandler *const ah = this->assembly_handler_pt();
		const unsigned long ne = mesh_pt() ? mesh_pt()->nelement() : 0;
		const unsigned nsel = (unsigned)sel.size();
		std::vector<int> uf(nsel);
		for (unsigned i = 0; i < nsel; i++) uf[i] = (int)i;
		std::function<int(int)> uf_find = [&uf](int a) { while (uf[a] != a) { uf[a] = uf[uf[a]]; a = uf[a]; } return a; };
		std::vector<int> local_sel; // index into sel, per elemental dof; -1 for a retained one
		for (unsigned long e = 0; e < ne; e++)
		{
			oomph::GeneralisedElement *el = mesh_pt()->element_pt(e);
			const unsigned nvar = ah->ndof(el);
			if (!nvar) continue;
			local_sel.assign(nvar, -1);
			bool any = false;
			for (unsigned i = 0; i < nvar; i++)
			{
				const int g = (int)ah->eqn_number(el, i);
				const std::vector<int>::iterator f = std::lower_bound(sel.begin(), sel.end(), g);
				if (f != sel.end() && *f == g) { local_sel[i] = (int)(f - sel.begin()); any = true; }
			}
			if (!any) continue;
			const char *mask = this->sparsity_mask_for_element(0, el, nvar);
			if (!mask) return std::vector<unsigned>(); // value-dependent: no informed preference to state
			for (unsigned i = 0; i < nvar; i++)
			{
				if (local_sel[i] < 0) continue;
				for (unsigned j = 0; j < nvar; j++)
				{
					if (local_sel[j] < 0 || j == i) continue;
					if (!mask[(size_t)i * nvar + j]) continue;
					const int ra = uf_find(local_sel[i]), rb = uf_find(local_sel[j]);
					if (ra != rb) uf[ra] = rb;
				}
			}
		}

		// One span per component, then the merge of the overlapping ones: a row index inside no span is
		// one no component crosses, which is all a cut has to avoid.
		std::map<int, std::pair<int, int>> span_of_root;
		for (unsigned i = 0; i < nsel; i++)
		{
			const int r = uf_find((int)i);
			std::map<int, std::pair<int, int>>::iterator it = span_of_root.find(r);
			if (it == span_of_root.end()) span_of_root[r] = std::make_pair(sel[i], sel[i]);
			else { if (sel[i] < it->second.first) it->second.first = sel[i]; if (sel[i] > it->second.second) it->second.second = sel[i]; }
		}
		std::vector<std::pair<int, int>> spans;
		for (const auto &kv : span_of_root) spans.push_back(kv.second);
		if (spans.empty()) return cuts;
		std::sort(spans.begin(), spans.end());
		std::vector<std::pair<int, int>> blocks;
		for (size_t i = 0; i < spans.size(); i++)
		{
			if (!blocks.empty() && spans[i].first <= blocks.back().second)
			{
				if (spans[i].second > blocks.back().second) blocks.back().second = spans[i].second;
			}
			else blocks.push_back(spans[i]);
		}

		// An over-long block means the selection mixes NODAL and element-internal dofs, because oomph
		// numbers all nodal values before any internal ones -- a CR bubble velocity and the DL pressure
		// of the same element then sit hundreds of equations apart and no cut can be moved off them.
		// snap_cuts_to_blocks returns empty there, the uniform split stands, and the plan builder
		// refuses with a message naming the dofs, which is far more use than a silently unbalanced
		// partition. An ElementBlockOrdering on those same fields is what makes the blocks short enough
		// for this to succeed; see dev_docs/dof_ordering.md.
		return snap_cuts_to_blocks(blocks, nproc, nd);
	}

	void Problem::preferred_linear_solver_distribution(oomph::LinearAlgebraDistribution *&dist_pt)
	{
		dist_pt = 0;
		if (!Communicator_pt || Communicator_pt->nproc() < 2) return;
		// Two reasons to want a non-uniform row split, and condensation's is the stronger: it is a
		// CORRECTNESS requirement (a component must be invertible on the rank owning its rows), whereas
		// a dof layout only wants its blocks kept whole so a block preconditioner sees them. So
		// condensation decides whenever it is on, and the layout is consulted only otherwise. Both
		// cannot be served at once in general -- their blocks are different partitions of the same rows.
		const bool for_condensation = use_static_condensation;
		// Deflation dots a DOF-layout vector (grad log M) against the Newton increment the solver
		// returns, so under --distribute the solver's rows have to be the dof rows. oomph's default
		// there is a uniform split, which is a different partition of the same global rows, and the two
		// vectors are then not comparable entry by entry -- with 225 dofs on two ranks the dof blocks
		// are 105/120 and the uniform ones 112/113. Replicated needs nothing: the gradient is the whole
		// vector and the solver's block is a slice of it. Deflation and static condensation are
		// mutually exclusive (Problem.set_deflation_operator refuses the combination), so they cannot
		// both claim this.
		const bool for_deflation = residual_scale_hook_active;
		if (!for_condensation && !for_deflation && dof_ordering_blocks.empty()) return;

		if (Problem_has_been_distributed)
		{
			// Each rank's dofs are already one contiguous global range (synchronise_eqn_numbers) and any
			// dof-ordering permutation was rank-local, so no block straddles a rank: the dof distribution
			// is block-aligned for free, and is what condensation needs besides.
			// Unconditionally, NOT gated on Dist_problem_matrix_distribution: oomph's own default for that
			// enum is Uniform_matrix_distribution (problem.cc, the constructor), not the "Default"
			// heuristic, so consulting it would simply mean never asking for the dof distribution at all.
			if (!for_condensation && !for_deflation) return; // a layout has nothing to add here; leave oomph's choice alone
			dist_pt = new oomph::LinearAlgebraDistribution(this->dof_distribution_pt());
			return;
		}
		// Replicated: deflation alone has no preference (see above), so it must not drag the row split
		// away from oomph's uniform one just by being switched on.
		if (!for_condensation && dof_ordering_blocks.empty()) return;
		const unsigned nproc = Communicator_pt->nproc(), my_rank = Communicator_pt->my_rank();
		std::vector<unsigned> cuts = for_condensation ? condensation_row_cuts() : dof_ordering_row_cuts();
		if (cuts.size() != nproc + 1) return;
		dist_pt = new oomph::LinearAlgebraDistribution(Communicator_pt, cuts[my_rank],
													   cuts[my_rank + 1] - cuts[my_rank], cuts[nproc]);
	}

	bool Problem::build_distributed_condensation_plan(CondensationPlan &plan)
	{
		plan.clear();
		// Serves both MPI modes. Distributed, the mesh is partitioned and the row distribution is the
		// dof distribution (guard 1 below). REPLICATED, every rank holds the whole mesh and the rows are
		// split uniformly; the ownership argument still holds, and more easily -- the requirement is
		// "the rank owning a row holds that dof's Data non-halo", and replicated every rank holds every
		// Data. What is NOT usable replicated is the problem's dof distribution: oomph builds it
		// undistributed there (first_row 0, nrow_local ndof on every rank), so it would name rank 0 as
		// the owner of everything. Row ownership therefore comes from the frozen plan's row_first.
		if (!Communicator_pt || Communicator_pt->nproc() < 2) return false;
		const unsigned nproc = Communicator_pt->nproc();
		const unsigned my_rank = Communicator_pt->my_rank();
		MPI_Comm comm = Communicator_pt->mpi_comm();

		// --- 0. Rank-local declines. Every condition here is a globally uniform property of the problem
		// (the pattern id is MPI_Allreduce-checked for equality in assign_eqn_numbers, and the frozen
		// distributed plan's generation was itself set behind a collective vote), so a rank returning
		// early cannot leave the others waiting in a collective.
		if (n_unaugmented_dofs != 0) return false;
		const unsigned long gen = this->get_jacobian_structure_id();
		if (!gen) return false;
		const unsigned nd = this->ndof();
		if (!nd) return false;
		const DistributedFrozenSparsity &sp = distributed_frozen_sparsity;
		// The plan is a list of positions in the frozen distributed value array, so that assembly is a
		// precondition. Before the first Jacobian of a structure id there is none yet, and the plan is
		// simply built on the next call -- which is where the solve path reaches it anyway.
		if (sp.generation != gen || sp.mats.empty() || sp.nproc != nproc || sp.ndof != nd ||
			sp.row_first.size() != nproc + 1) return false;

		// --- 1. On a DISTRIBUTED problem the matrix distribution has to BE the dof distribution, or the
		// ownership argument fails. oomph picks between Dof_distribution_pt and a uniform one for the
		// Jacobian's rows, switching to uniform as soon as one rank exceeds the uniform share by 10 %
		// (problem.cc, create_new_linear_algebra_distribution). Under a uniform distribution the rank
		// owning a row is not the rank holding the Data, so nothing below holds.
		//
		// Replicated the check is vacuous rather than load-bearing, and would in fact always fail: the
		// dof distribution is undistributed there, so its first_row is 0 on every rank. The property it
		// stands for -- the row owner holds the Data non-halo -- is automatic when every rank holds
		// every Data, so the check is skipped rather than relaxed.
		{
			std::string err;
			const oomph::LinearAlgebraDistribution *dofdist = this->dof_distribution_pt();
			if (Problem_has_been_distributed &&
				(!dofdist || dofdist->first_row() != sp.first_row || dofdist->nrow_local() != sp.nrow_local))
				err = "Static condensation cannot be applied to this distributed run: the Jacobian's row "
					  "distribution is not the problem's dof distribution, so the rank that owns a row is not "
					  "the one that holds the degree of freedom, and a component's block is held by nobody in "
					  "full. pyoomph asks for the dof distribution by itself whenever condensation is on, in "
					  "both places one is chosen (see Problem::preferred_linear_solver_distribution), so "
					  "reaching this message means something overrode that -- a linear solver that lays the "
					  "rows out itself, or an explicit problem.set_dist_problem_matrix_distribution(\"uniform\"). "
					  "Undo that, or switch static condensation off for this run.";
			collective_throw(err); // collective; every rank throws or none does
		}

		const DistributedFrozenSparsity::Mat &mt = sp.mats[0];
		const std::vector<int> &row_start = mt.final_row_start; // nrow_local+1
		const std::vector<int> &col_index = mt.final_col;       // GLOBAL columns, ascending within a row
		const unsigned nrow_local = sp.nrow_local, first_row = sp.first_row;
		plan.full_nnz = col_index.size();
		plan.distributed = true;
		plan.nproc = nproc;
		plan.my_rank = my_rank;
		plan.first_row = first_row;
		plan.nrow_local = nrow_local;

		auto owns = [first_row, nrow_local](int g) { return g >= (int)first_row && g < (int)(first_row + nrow_local); };

		// --- 2. The L dofs THIS rank owns. The selection scan sees halo copies too and therefore also
		// names dofs owned by other ranks; those are their owner's business. Ordered by equation number
		// and stable, exactly as serially, so a rebuild of an unchanged pattern is bit-identical.
		this->ensure_static_condensation_selection();
		std::vector<std::pair<long, std::pair<oomph::Data *, unsigned>>> sel;
		for (const auto &kv : static_condensation_selected)
			for (unsigned v = 0; v < kv.second.size(); v++)
			{
				if (!kv.second[v]) continue;
				const long eq = kv.first->eqn_number(v);
				if (eq >= 0 && owns((int)eq)) sel.push_back(std::make_pair(eq, std::make_pair(kv.first, v)));
			}
		std::stable_sort(sel.begin(), sel.end(), [](const std::pair<long, std::pair<oomph::Data *, unsigned>> &a,
													const std::pair<long, std::pair<oomph::Data *, unsigned>> &b)
						 { return a.first < b.first; });
		std::vector<int> l_index(nrow_local, -1); // local row -> index in the L arrays, -1 for retained
		std::vector<int> l_eqn;
		std::vector<std::pair<oomph::Data *, unsigned>> l_value;
		for (const auto &s : sel)
		{
			const int lr = (int)s.first - (int)first_row;
			if (l_index[lr] >= 0) continue; // two Data values sharing an equation: keep the first
			l_index[lr] = (int)l_eqn.size();
			l_eqn.push_back((int)s.first);
			l_value.push_back(s.second);
		}
		const unsigned nL = (unsigned)l_eqn.size();

		// A rank may legitimately own no condensed dof at all (no element of the domain, or the whole
		// selection sits elsewhere) and still own E rows of somebody else's components, so "nothing
		// selected" is decided globally, once, rather than per rank.
		{
			int mine = (nL ? 1 : 0), any = 0;
			MPI_Allreduce(&mine, &any, 1, MPI_INT, MPI_MAX, comm);
			if (!any) { plan.clear(); return false; }
		}

		// --- 3. Exchange 1: whose columns are condensed?
		// Deciding this from the local halo copy of the Data would be right almost always -- but
		// condense_element_private_dofs() measures "private" against a RANK-LOCAL reverse external-data
		// scan, which a neighbour that does not hold the adopting interface element sees differently.
		// So the owner is asked. The traffic is proportional to the partition boundary.
		std::vector<int> foreign_cols;
		for (size_t k = 0; k < col_index.size(); k++)
			if (!owns(col_index[k])) foreign_cols.push_back(col_index[k]);
		std::sort(foreign_cols.begin(), foreign_cols.end());
		foreign_cols.erase(std::unique(foreign_cols.begin(), foreign_cols.end()), foreign_cols.end());
		std::vector<std::vector<int>> q_send(nproc), q_recv, a_send(nproc), a_recv;
		for (size_t i = 0; i < foreign_cols.size(); i++)
			q_send[sp.rank_of_row((unsigned)foreign_cols[i])].push_back(foreign_cols[i]);
		condensation_alltoall_ints(comm, nproc, my_rank, 80, q_send, q_recv);
		for (unsigned p = 0; p < nproc; p++)
		{
			a_send[p].resize(q_recv[p].size());
			for (size_t i = 0; i < q_recv[p].size(); i++)
			{
				const int lr = q_recv[p][i] - (int)first_row;
				a_send[p][i] = (lr >= 0 && lr < (int)nrow_local && l_index[lr] >= 0) ? 1 : 0;
			}
		}
		condensation_alltoall_ints(comm, nproc, my_rank, 81, a_send, a_recv);
		std::vector<int> foreign_L; // the condensed ones among foreign_cols, ascending
		for (unsigned p = 0; p < nproc; p++)
			for (size_t i = 0; i < a_recv[p].size(); i++)
				if (a_recv[p][i]) foreign_L.push_back(q_send[p][i]);
		std::sort(foreign_L.begin(), foreign_L.end());
		auto is_cond = [&](int g) -> bool
		{
			if (owns(g)) return l_index[g - (int)first_row] >= 0;
			return std::binary_search(foreign_L.begin(), foreign_L.end(), g);
		};

		// --- 4. Components of the owned L block, and the cross-rank refusal.
		std::string err;
		std::vector<int> uf(nL);
		for (unsigned i = 0; i < nL; i++) uf[i] = (int)i;
		auto uf_find = [&uf](int a) { while (uf[a] != a) { uf[a] = uf[uf[a]]; a = uf[a]; } return a; };
		std::vector<char> row_ok(nL, 0), col_ok(nL, 0);
		std::vector<long> straddling;
		for (unsigned i = 0; i < nL && err.empty(); i++)
		{
			const int lr = l_eqn[i] - (int)first_row;
			for (int k = row_start[lr]; k < row_start[lr + 1]; k++)
			{
				const int c = col_index[k];
				if (!is_cond(c)) continue;
				if (!owns(c))
				{
					// The block to be inverted reaches across a partition boundary, so no rank holds it in
					// full. Refusing is the honest answer: serving it would be a distributed dense solve
					// per component, which is not what condensation is for. Detected from at least one
					// side whichever way round the coupling runs.
					// One entry per offending DOF, not per offending column: describe_global_dofs() keys
					// its lookup on the equation number, so a repeated one comes back as "<unknown dof>".
					if (straddling.size() < 4 && (straddling.empty() || straddling.back() != l_eqn[i]))
						straddling.push_back(l_eqn[i]);
					continue;
				}
				const int j = l_index[c - (int)first_row];
				row_ok[i] = 1;
				col_ok[j] = 1;
				const int ra = uf_find((int)i), rb = uf_find(j);
				if (ra != rb) uf[ra] = rb;
			}
		}
		if (!straddling.empty())
		{
			err = "Static condensation: a connected block of selected degrees of freedom is split across MPI "
				  "ranks, so no rank holds it in full and it cannot be eliminated. Sample dofs whose equation "
				  "couples to a selected unknown owned by another rank: " +
				  format_dof_samples(*this, straddling, straddling.size()) + ". ";
			if (Problem_has_been_distributed)
				err += "Static condensation eliminates ELEMENT-LOCAL dofs; a selection that couples across "
					   "element (and hence partition) boundaries -- a DG field coupled across facets, say -- is "
					   "not element-local. Restrict the selection, or switch condensation off for distributed runs.";
			else
				// A replicated run reaches this for a second, quite different reason, and saying "not
				// element-local" there would send the reader looking for a bug in their equations.
				err += "This is a REPLICATED MPI run (mpirun without --distribute): the mesh is not partitioned, "
					   "but the linear system's ROWS still are, and a block can only be eliminated on the rank "
					   "that owns all of its rows. pyoomph moves the row cut points off the blocks where it can "
					   "(Problem::condensation_row_cuts), which works when the selected dofs of an element are "
					   "numbered together -- element-internal values are. It cannot when the selection mixes "
					   "NODAL and element-internal dofs, because oomph-lib numbers every nodal value before any "
					   "internal one, so the two halves of the block sit hundreds of equations apart: a "
					   "Crouzeix-Raviart selection of the bubble VELOCITY (nodal) together with the DL pressure "
					   "(internal) is exactly that. Run with --distribute, where each rank's dofs are renumbered "
					   "contiguously and the question does not arise, or switch condensation off for this run.";
		}
		collective_throw(err);

		std::vector<int> comp_of_l(nL, -1), comp_of_root(nL, -1);
		unsigned ncomp = 0;
		for (unsigned i = 0; i < nL; i++)
		{
			const int r = uf_find((int)i);
			if (comp_of_root[r] < 0) comp_of_root[r] = (int)(ncomp++);
			comp_of_l[i] = comp_of_root[r];
		}
		std::vector<CondensationComponent> owned(ncomp);
		for (unsigned i = 0; i < nL; i++)
		{
			owned[comp_of_l[i]].L_eqns.push_back(l_eqn[i]);
			owned[comp_of_l[i]].L_values.push_back(l_value[i]);
			owned[comp_of_l[i]].owner_rank = (int)my_rank;
		}

		// --- 5. The serial guards, on the owned block, as collective votes.
		for (unsigned c = 0; c < ncomp && err.empty(); c++)
		{
			if (owned[c].nL() <= static_condensation_max_component_size) continue;
			std::vector<long> sample;
			for (unsigned i = 0; i < owned[c].nL() && i < 5; i++) sample.push_back(owned[c].L_eqns[i]);
			err = "Static condensation: the selected degrees of freedom do not decompose into small "
				  "element-local blocks. One connected component holds " + std::to_string(owned[c].nL()) +
				  " mutually coupled dofs, above the limit static_condensation_max_component_size = " +
				  std::to_string(static_condensation_max_component_size) + ". Sample dofs of the component: " +
				  format_dof_samples(*this, sample, owned[c].nL()) +
				  ". Either restrict the selection to dofs that really are element-local, or raise "
				  "problem.static_condensation_max_component_size if the size is genuinely intended.";
		}
		if (err.empty())
		{
			// col_ok is computable from the owned rows alone precisely BECAUSE a component reaching across
			// a rank boundary has already been refused above.
			std::vector<long> bad_rows, bad_cols;
			unsigned long n_bad_rows = 0, n_bad_cols = 0;
			for (unsigned i = 0; i < nL; i++)
			{
				if (!row_ok[i]) { n_bad_rows++; if (bad_rows.size() < 4) bad_rows.push_back(l_eqn[i]); }
				if (!col_ok[i]) { n_bad_cols++; if (bad_cols.size() < 4) bad_cols.push_back(l_eqn[i]); }
			}
			if (!bad_rows.empty() || !bad_cols.empty())
			{
				err = "Static condensation: the selected degrees of freedom cannot be eliminated, because the "
					  "block to be inverted is structurally singular.";
				if (!bad_rows.empty())
					err += " The equations of these selected dofs contain no selected unknown at all (an empty "
						   "ROW of the block): " + format_dof_samples(*this, bad_rows, n_bad_rows) + ".";
				if (!bad_cols.empty())
					err += " These selected dofs appear in no selected equation (an empty COLUMN of the block): " +
						   format_dof_samples(*this, bad_cols, n_bad_cols) + ".";
				err += " A dof can only be eliminated together with an equation that determines it. The classic "
					   "case is selecting a discontinuous (DL) pressure on its own: the continuity equation "
					   "contains no pressure, so it has to be condensed jointly with the bubble velocities that "
					   "it does determine -- and the constant pressure mode has to stay in the global system.";
			}
		}
		collective_throw(err);

		// --- 6. E_C, in two halves. By column (a retained dof appearing in one of the component's own
		// equations) the owner can see for itself; by row (a retained equation containing one of the
		// component's unknowns) it cannot, because that row may be owned by anyone. Both directions are
		// needed and they are genuinely different sets -- that is the reason this design condenses after
		// assembly rather than per element -- so the second half is exchanged (Exchange 2).
		std::vector<std::vector<int>> E_lists(ncomp);
		for (unsigned i = 0; i < nL; i++)
		{
			const int lr = l_eqn[i] - (int)first_row;
			std::vector<int> &lst = E_lists[comp_of_l[i]];
			for (int k = row_start[lr]; k < row_start[lr + 1]; k++)
				if (!is_cond(col_index[k])) lst.push_back(col_index[k]);
		}
		std::vector<std::vector<int>> pair_send(nproc), pair_recv;
		for (unsigned lr = 0; lr < nrow_local; lr++)
		{
			if (l_index[lr] >= 0) continue; // a condensed row is never an E row
			const int r = (int)first_row + (int)lr;
			for (int k = row_start[lr]; k < row_start[lr + 1]; k++)
			{
				const int c = col_index[k];
				if (!is_cond(c)) continue;
				if (owns(c)) E_lists[comp_of_l[l_index[c - (int)first_row]]].push_back(r);
				else
				{
					std::vector<int> &b = pair_send[sp.rank_of_row((unsigned)c)];
					b.push_back(c); b.push_back(r);
				}
			}
		}
		condensation_alltoall_ints(comm, nproc, my_rank, 82, pair_send, pair_recv);
		for (unsigned p = 0; p < nproc; p++)
			for (size_t i = 0; i + 1 < pair_recv[p].size(); i += 2)
			{
				const int c = pair_recv[p][i], r = pair_recv[p][i + 1];
				const int lr = c - (int)first_row;
				// Cannot miss: the sender only ever names columns this rank itself declared condensed in
				// Exchange 1.
				if (lr < 0 || lr >= (int)nrow_local || l_index[lr] < 0)
				{
					err = "Internal error while building the distributed static condensation plan: rank " +
						  std::to_string(p) + " reported a coupling to equation " + std::to_string(c) +
						  ", which this rank does not hold as a condensed dof.";
					continue;
				}
				E_lists[comp_of_l[l_index[lr]]].push_back(r);
			}
		collective_throw(err);
		for (unsigned c = 0; c < ncomp; c++)
		{
			std::sort(E_lists[c].begin(), E_lists[c].end());
			E_lists[c].erase(std::unique(E_lists[c].begin(), E_lists[c].end()), E_lists[c].end());
			owned[c].E_eqns.swap(E_lists[c]);
		}

		// --- 7. Exchange 3: the component descriptors. A rank learns of its participation from this
		// message and not from its own scan -- the by-column direction (its row appears as a COLUMN of one
		// of the owner's L rows) is not visible to it at all. Per component the message carries
		// [nL, nE, L_eqns..., E_eqns...], components ordered by their smallest L equation so that the
		// sender's and the receiver's buffers agree without any further negotiation.
		std::vector<std::vector<int>> desc_send(nproc), desc_recv;
		std::vector<std::vector<int>> peers_of_owned(ncomp);
		for (unsigned c = 0; c < ncomp; c++)
		{
			std::vector<int> peers;
			for (size_t j = 0; j < owned[c].E_eqns.size(); j++)
			{
				const unsigned p = sp.rank_of_row((unsigned)owned[c].E_eqns[j]);
				if (p != my_rank) peers.push_back((int)p);
			}
			std::sort(peers.begin(), peers.end());
			peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
			peers_of_owned[c] = peers;
			for (size_t q = 0; q < peers.size(); q++)
			{
				std::vector<int> &b = desc_send[peers[q]];
				b.push_back((int)owned[c].nL());
				b.push_back((int)owned[c].nE());
				b.insert(b.end(), owned[c].L_eqns.begin(), owned[c].L_eqns.end());
				b.insert(b.end(), owned[c].E_eqns.begin(), owned[c].E_eqns.end());
			}
		}
		condensation_alltoall_ints(comm, nproc, my_rank, 83, desc_send, desc_recv);

		// --- 8. Assemble the component array: the ones this rank owns plus the ones it merely holds an E
		// row of, ordered by smallest L equation.
		std::vector<CondensationComponent> comps;
		std::vector<std::vector<int>> comp_peers; // parallel; only meaningful for owned components
		comps.reserve(ncomp);
		for (unsigned c = 0; c < ncomp; c++) { comps.push_back(std::move(owned[c])); comp_peers.push_back(peers_of_owned[c]); }
		for (unsigned p = 0; p < nproc; p++)
		{
			if (p == my_rank) continue;
			const std::vector<int> &b = desc_recv[p];
			size_t k = 0;
			while (k + 1 < b.size())
			{
				const unsigned nl = (unsigned)b[k], ne = (unsigned)b[k + 1];
				k += 2;
				if (k + nl + ne > b.size()) { err = "Internal error: truncated static condensation descriptor from rank " + std::to_string(p); break; }
				CondensationComponent comp;
				comp.owner_rank = (int)p;
				comp.L_eqns.assign(b.begin() + k, b.begin() + k + nl);
				k += nl;
				comp.E_eqns.assign(b.begin() + k, b.begin() + k + ne);
				k += ne;
				comps.push_back(comp);
				comp_peers.push_back(std::vector<int>());
			}
		}
		collective_throw(err);
		{
			std::vector<unsigned> order(comps.size());
			for (unsigned i = 0; i < order.size(); i++) order[i] = i;
			std::sort(order.begin(), order.end(), [&comps](unsigned a, unsigned b)
					  { return comps[a].L_eqns[0] < comps[b].L_eqns[0]; });
			std::vector<CondensationComponent> sorted;
			std::vector<std::vector<int>> sorted_peers;
			sorted.reserve(comps.size());
			for (unsigned i = 0; i < order.size(); i++) { sorted.push_back(std::move(comps[order[i]])); sorted_peers.push_back(comp_peers[order[i]]); }
			comps.swap(sorted);
			comp_peers.swap(sorted_peers);
		}
		const unsigned nc = (unsigned)comps.size();

		// The operator exchange plan. Both lists come out in ascending smallest-L-equation order because
		// the component array is sorted that way, which is exactly the order Exchange 3 was packed in.
		plan.xy_send_start.assign(nproc + 1, 0);
		plan.xy_recv_start.assign(nproc + 1, 0);
		for (unsigned p = 0; p < nproc; p++)
		{
			for (unsigned i = 0; i < nc; i++)
				if (comps[i].owner_rank == (int)my_rank &&
					std::find(comp_peers[i].begin(), comp_peers[i].end(), (int)p) != comp_peers[i].end())
					plan.xy_send_comp.push_back((int)i);
			plan.xy_send_start[p + 1] = (int)plan.xy_send_comp.size();
			for (unsigned i = 0; i < nc; i++)
				if (comps[i].owner_rank == (int)p && p != my_rank) plan.xy_recv_comp.push_back((int)i);
			plan.xy_recv_start[p + 1] = (int)plan.xy_recv_comp.size();
		}

		// --- 9. my_E: which of a component's E rows this rank owns. This is the only thing that decides
		// what work a rank does for a component, owned or remote.
		for (unsigned i = 0; i < nc; i++)
		{
			CondensationComponent &comp = comps[i];
			for (unsigned j = 0; j < comp.nE(); j++)
				if (owns(comp.E_eqns[j])) comp.my_E.push_back((int)j);
		}

		// --- 10. Reconstruction inputs for the components this rank owns.
		//
		// DISTRIBUTED: dx is distributed too, so only the E rows this rank owns are in it; for the rest
		// the increment is recovered from the local (halo) copy of the VALUE,
		// (u_stored - u_now)/Relaxation_factor, which needs no exchange of dx at all. Such a copy always
		// exists: an E dof is a dof of an element that contains one of the component's L dofs, and that
		// element is non-halo on this very rank.
		//
		// REPLICATED: the problem's dof distribution is not distributed, so newton_solve() redistributes
		// dx to it and every rank gets the whole increment. Every E row can then be read from dx exactly,
		// and the recovery is not needed at all -- E_dx_row holds GLOBAL rows there.
		const bool replicated = !Problem_has_been_distributed;
		plan.dx_is_global = replicated;
		{
			std::vector<int> wanted;
			for (unsigned i = 0; i < nc; i++)
			{
				CondensationComponent &comp = comps[i];
				if (comp.owner_rank != (int)my_rank) continue;
				comp.E_dx_row.assign(comp.nE(), -1);
				comp.E_value_pt.assign(comp.nE(), (double *)NULL);
				comp.E_u_stored.assign(comp.nE(), 0.0);
				for (unsigned j = 0; j < comp.nE(); j++)
				{
					if (replicated) comp.E_dx_row[j] = comp.E_eqns[j];
					else if (owns(comp.E_eqns[j])) comp.E_dx_row[j] = comp.E_eqns[j] - (int)first_row;
					else wanted.push_back(comp.E_eqns[j]);
				}
			}
			std::sort(wanted.begin(), wanted.end());
			wanted.erase(std::unique(wanted.begin(), wanted.end()), wanted.end());
			std::vector<double *> found(wanted.size(), NULL);
			if (!wanted.empty())
			{
				auto take = [&](oomph::Data *d)
				{
					if (!d) return;
					for (unsigned v = 0; v < d->nvalue(); v++)
					{
						const long eq = d->eqn_number(v);
						if (eq < 0) continue;
						const std::vector<int>::iterator f = std::lower_bound(wanted.begin(), wanted.end(), (int)eq);
						if (f != wanted.end() && *f == (int)eq) found[f - wanted.begin()] = d->value_pt(v);
					}
				};
				const unsigned long n_element = mesh_pt() ? mesh_pt()->nelement() : 0;
				for (unsigned long e = 0; e < n_element; e++)
				{
					oomph::GeneralisedElement *el = mesh_pt()->element_pt(e);
					if (!el) continue;
					for (unsigned k = 0; k < el->ninternal_data(); k++) take(el->internal_data_pt(k));
					for (unsigned k = 0; k < el->nexternal_data(); k++) take(el->external_data_pt(k));
					oomph::FiniteElement *fe = dynamic_cast<oomph::FiniteElement *>(el);
					if (!fe) continue;
					for (unsigned n = 0; n < fe->nnode(); n++)
					{
						oomph::Node *nd = fe->node_pt(n);
						take(nd);
						oomph::SolidNode *sn = dynamic_cast<oomph::SolidNode *>(nd);
						if (sn) take(sn->variable_position_pt());
					}
				}
				for (unsigned k = 0; k < this->nglobal_data(); k++) take(this->global_data_pt(k));
				for (size_t i = 0; i < wanted.size(); i++)
					if (!found[i])
					{
						err = "Internal error while building the distributed static condensation plan: no local "
							  "copy of the value carrying global equation " + std::to_string(wanted[i]) +
							  " could be found on this rank, although a condensed component couples to it. The "
							  "reconstruction of the eliminated dofs reads that copy.";
						break;
					}
			}
			collective_throw(err);
			for (unsigned i = 0; i < nc; i++)
			{
				CondensationComponent &comp = comps[i];
				if (comp.owner_rank != (int)my_rank) continue;
				for (unsigned j = 0; j < comp.nE(); j++)
				{
					if (comp.E_dx_row[j] >= 0) continue;
					const std::vector<int>::iterator f = std::lower_bound(wanted.begin(), wanted.end(), comp.E_eqns[j]);
					comp.E_value_pt[j] = found[f - wanted.begin()];
				}
			}
		}

		// --- 10b. Replicated only: where to APPLY a reconstructed increment.
		//
		// A replicated run has no halo synchronisation to lean on -- each rank is a separate copy of the
		// whole problem, kept in step only because every rank applies the same dx to the same dofs. The
		// eliminated dofs are not in dx, and only the owner of a component can reconstruct them, so the
		// increments are allgathered and applied by every rank through this lookup. Every L dof of every
		// component in the plan is in it, owned or not; the value pointer is found by the same scan the E
		// dofs use, which sees every Data on a replicated mesh.
		if (replicated)
		{
			// Built from the whole SELECTION rather than from `comps`: a component owned by another rank
			// whose E rows are also all over there is entirely invisible in this rank's component array,
			// yet its increment still arrives in the allgather and still has to be applied here.
			// Replicated, the selection is identical on every rank and already names the Data, so this is
			// a sort rather than another scan over the mesh.
			std::vector<std::pair<int, double *>> tmp;
			for (const auto &kv : static_condensation_selected)
				for (unsigned v = 0; v < kv.second.size(); v++)
				{
					if (!kv.second[v]) continue;
					const long eq = kv.first->eqn_number(v);
					if (eq >= 0) tmp.push_back(std::make_pair((int)eq, kv.first->value_pt(v)));
				}
			std::stable_sort(tmp.begin(), tmp.end(), [](const std::pair<int, double *> &a, const std::pair<int, double *> &b)
							 { return a.first < b.first; });
			for (size_t i = 0; i < tmp.size(); i++)
			{
				// Two Data values sharing an equation: keep the first, exactly as the L selection does.
				if (!plan.all_L_eqn.empty() && plan.all_L_eqn.back() == tmp[i].first) continue;
				plan.all_L_eqn.push_back(tmp[i].first);
				plan.all_L_value_pt.push_back(tmp[i].second);
			}
		}

		// --- 11. The condensed LOCAL pattern: this rank's owned rows, global column indices. A retained
		// row keeps its retained columns and gains the whole E_C of every component it takes part in; a
		// condensed row keeps its diagonal alone. Identical to the serial construction, restricted to the
		// owned rows -- a fill entry in a column this rank never had is unremarkable, columns are global.
		{
			std::vector<std::vector<int>> cond_cols(nrow_local);
			for (unsigned lr = 0; lr < nrow_local; lr++)
			{
				if (l_index[lr] >= 0) { cond_cols[lr].push_back((int)first_row + (int)lr); continue; }
				for (int k = row_start[lr]; k < row_start[lr + 1]; k++)
					if (!is_cond(col_index[k])) cond_cols[lr].push_back(col_index[k]);
			}
			for (unsigned i = 0; i < nc; i++)
			{
				const CondensationComponent &comp = comps[i];
				for (size_t q = 0; q < comp.my_E.size(); q++)
				{
					const int lr = comp.E_eqns[comp.my_E[q]] - (int)first_row;
					cond_cols[lr].insert(cond_cols[lr].end(), comp.E_eqns.begin(), comp.E_eqns.end());
				}
			}
			plan.cond_row_start.assign(nrow_local + 1, 0);
			for (unsigned lr = 0; lr < nrow_local; lr++)
			{
				std::vector<int> &cc = cond_cols[lr];
				std::sort(cc.begin(), cc.end());
				cc.erase(std::unique(cc.begin(), cc.end()), cc.end());
				plan.cond_row_start[lr + 1] = plan.cond_row_start[lr] + (int)cc.size();
			}
			plan.cond_column_index.resize(plan.cond_row_start[nrow_local]);
			for (unsigned lr = 0; lr < nrow_local; lr++)
				std::copy(cond_cols[lr].begin(), cond_cols[lr].end(), plan.cond_column_index.begin() + plan.cond_row_start[lr]);
		}
		const std::vector<int> &crs = plan.cond_row_start;
		const std::vector<int> &cci = plan.cond_column_index;
		auto full_slot = [&](int grow, int col) -> int
		{
			const int lo = row_start[grow - (int)first_row], hi = row_start[grow - (int)first_row + 1];
			const int *b = col_index.data() + lo, *e = col_index.data() + hi;
			const int *f = std::lower_bound(b, e, col);
			if (f == e || *f != col) return -1;
			return lo + (int)(f - b);
		};
		auto cond_slot = [&](int grow, int col) -> int
		{
			const int lo = crs[grow - (int)first_row], hi = crs[grow - (int)first_row + 1];
			const int *b = cci.data() + lo, *e = cci.data() + hi;
			const int *f = std::lower_bound(b, e, col);
			if (f == e || *f != col) return -1;
			return lo + (int)(f - b);
		};

		// --- 12. The slot lists. LL and LE only on the owner (only it holds the L rows); EL and the fill
		// slots on every participant, sized by the E rows it owns.
		for (unsigned i = 0; i < nc; i++)
		{
			CondensationComponent &comp = comps[i];
			const unsigned nl = comp.nL(), ne = comp.nE(), nmy = (unsigned)comp.my_E.size();
			if (comp.owner_rank == (int)my_rank)
			{
				comp.LL_slots.assign((size_t)nl * nl, -1);
				comp.LE_slots.assign((size_t)nl * ne, -1);
				for (unsigned a = 0; a < nl; a++)
				{
					for (unsigned b = 0; b < nl; b++) comp.LL_slots[(size_t)a * nl + b] = full_slot(comp.L_eqns[a], comp.L_eqns[b]);
					for (unsigned b = 0; b < ne; b++) comp.LE_slots[(size_t)a * ne + b] = full_slot(comp.L_eqns[a], comp.E_eqns[b]);
				}
			}
			comp.EL_slots.assign((size_t)nmy * nl, -1);
			comp.fill_slots.assign((size_t)nmy * ne, -1);
			for (unsigned q = 0; q < nmy; q++)
			{
				const int grow = comp.E_eqns[comp.my_E[q]];
				for (unsigned b = 0; b < nl; b++) comp.EL_slots[(size_t)q * nl + b] = full_slot(grow, comp.L_eqns[b]);
				for (unsigned b = 0; b < ne; b++)
				{
					const int cs = cond_slot(grow, comp.E_eqns[b]);
					if (cs < 0)
					{
						err = "Internal error while building the distributed static condensation plan: the fill-in "
							  "of a component is not contained in the condensed local pattern.";
						break;
					}
					comp.fill_slots[(size_t)q * ne + b] = cs;
				}
			}
		}

		// Pass-through: every (retained, retained) entry of an owned row, copied verbatim. In full-slot
		// order, so the copy sweeps forward through both arrays.
		for (unsigned lr = 0; lr < nrow_local && err.empty(); lr++)
		{
			if (l_index[lr] >= 0) continue;
			for (int k = row_start[lr]; k < row_start[lr + 1]; k++)
			{
				const int c = col_index[k];
				if (is_cond(c)) continue;
				const int cs = cond_slot((int)first_row + (int)lr, c);
				if (cs < 0)
				{
					err = "Internal error while building the distributed static condensation plan: a retained "
						  "matrix entry has no slot in the condensed local pattern.";
					break;
				}
				plan.passthrough_full_slot.push_back(k);
				plan.passthrough_cond_slot.push_back(cs);
			}
		}
		collective_throw(err);

		plan.components.swap(comps);
		plan.condensed_eqns = l_eqn;
		plan.L_diagonal_cond_slots.resize(nL);
		for (unsigned i = 0; i < nL; i++) plan.L_diagonal_cond_slots[i] = crs[l_eqn[i] - (int)first_row]; // the row's only entry
		// Local-row bitmap here, not the global one the serial plan carries: nothing at runtime reads it,
		// and a global one would be ndof bytes per rank for no purpose.
		plan.is_condensed.assign(nrow_local, 0);
		for (unsigned lr = 0; lr < nrow_local; lr++) plan.is_condensed[lr] = (l_index[lr] >= 0);
		return true;
	}
#endif

	// The plan for the current pattern id, rebuilt whenever that id has moved on.
	CondensationPlan *Problem::acquire_condensation_plan()
	{
		const unsigned long gen = this->get_jacobian_structure_id();
		if (!gen) { condensation_plan.clear(); return NULL; }
		const unsigned nd = this->ndof();
		if (condensation_plan.generation == gen && condensation_plan.ndof == nd) return &condensation_plan;
		if (!build_condensation_plan(condensation_plan)) { condensation_plan.clear(); return NULL; }
		condensation_plan.generation = gen;
		condensation_plan.ndof = nd;
		condensation_plan_rebuilds++;
		return &condensation_plan;
	}

	// ==================== Static condensation: the numeric kernel ====================
	namespace
	{
		// Dense LU with partial pivoting on a flat row-major buffer, with a RELATIVE pivot guard.
		// Returns the step at which the pivot was rejected, or -1 on success. `pivots` records the row
		// each step exchanged with, so the same interchanges can be replayed on a right-hand side.
		//
		// oomph-lib's DenseDoubleMatrix::ludecompose() is deliberately NOT used here. DenseLU::factorise()
		// (thirdparty/oomph-lib/include/linear_solver.cc:139) throws only when an entire row is exactly
		// zero and otherwise replaces a vanishing pivot by 1e-20, the Numerical-Recipes trick -- so a
		// singular A_LL comes back as an increment of order 1e20 instead of as an error, which is exactly
		// what the classic "condense the DL constant pressure mode as well" mistake produces. Testing the
		// pivot against the size of the block requires owning the factorisation. Two further reasons:
		// DenseLU news its LU factors and its pivot array on every single factorise, and it back-
		// substitutes one right-hand side at a time through a DoubleVector, while a component needs nE+1
		// of them against the same factors.
		static int dense_lu_factorise(double *A, unsigned n, int *pivots, double pivot_tol)
		{
			double amax = 0.0;
			for (size_t k = 0; k < (size_t)n * n; k++) amax = std::max(amax, std::fabs(A[k]));
			// Relative to the whole block, not to the row: a component mixes bubble-velocity and
			// pressure-gradient equations, whose natural scales differ by orders of magnitude, and a
			// per-row threshold would then declare a perfectly good pivot too small.
			const double thresh = pivot_tol * amax;
			for (unsigned j = 0; j < n; j++)
			{
				unsigned piv = j;
				double best = std::fabs(A[(size_t)j * n + j]);
				for (unsigned i = j + 1; i < n; i++)
				{
					const double v = std::fabs(A[(size_t)i * n + j]);
					if (v > best) { best = v; piv = i; }
				}
				if (best <= thresh) return (int)j;
				pivots[j] = (int)piv;
				if (piv != j)
					for (unsigned k = 0; k < n; k++) std::swap(A[(size_t)j * n + k], A[(size_t)piv * n + k]);
				const double inv = 1.0 / A[(size_t)j * n + j];
				for (unsigned i = j + 1; i < n; i++)
				{
					const double f = A[(size_t)i * n + j] * inv;
					A[(size_t)i * n + j] = f;
					if (f == 0.0) continue;
					for (unsigned k = j + 1; k < n; k++) A[(size_t)i * n + k] -= f * A[(size_t)j * n + k];
				}
			}
			return -1;
		}

		// Solves A X = B in place for all nrhs right-hand sides at once; B is row-major n x nrhs.
		static void dense_lu_solve(const double *LU, unsigned n, const int *pivots, double *B, unsigned nrhs)
		{
			for (unsigned j = 0; j < n; j++)
			{
				const unsigned p = (unsigned)pivots[j];
				if (p != j)
					for (unsigned c = 0; c < nrhs; c++) std::swap(B[(size_t)j * nrhs + c], B[(size_t)p * nrhs + c]);
			}
			for (unsigned i = 1; i < n; i++) // forward substitution, L has a unit diagonal
				for (unsigned k = 0; k < i; k++)
				{
					const double f = LU[(size_t)i * n + k];
					if (f == 0.0) continue;
					for (unsigned c = 0; c < nrhs; c++) B[(size_t)i * nrhs + c] -= f * B[(size_t)k * nrhs + c];
				}
			for (unsigned ii = n; ii-- > 0;) // back substitution
			{
				for (unsigned k = ii + 1; k < n; k++)
				{
					const double f = LU[(size_t)ii * n + k];
					if (f == 0.0) continue;
					for (unsigned c = 0; c < nrhs; c++) B[(size_t)ii * nrhs + c] -= f * B[(size_t)k * nrhs + c];
				}
				const double inv = 1.0 / LU[(size_t)ii * n + ii];
				for (unsigned c = 0; c < nrhs; c++) B[(size_t)ii * nrhs + c] *= inv;
			}
		}
	}

	// Replaces the assembled FULL system by the CONDENSED one, in place: the selected dofs are
	// eliminated component by component through their dense Schur complement, their rows become identity
	// rows with a zero right-hand side (so the solver returns a zero increment there, which stage 4
	// replaces by the reconstructed one), and the matrix is rebuilt on the plan's condensed pattern.
	//
	// Everything the loop needs was precomputed by build_condensation_plan(): this routine performs no
	// column search, allocates nothing per component, and touches the matrix only through slot indices.
	void Problem::apply_static_condensation(oomph::DoubleVector &residuals, oomph::CRDoubleMatrix &jacobian)
	{
		CondensationPlan *plan = this->acquire_condensation_plan();
		if (!plan)
			throw_runtime_error("Internal error: static condensation was applied to an assembly for which no "
								"elimination plan exists. The activation gate must not let this call through.");
#ifdef OOMPH_HAS_MPI
		if (plan->distributed) { apply_static_condensation_distributed(residuals, jacobian); return; }
#endif
		const unsigned nd = plan->ndof;

		// --- 1. Positive engagement check. The plan is a list of positions in a particular value array,
		// so it is only meaningful if the matrix really was assembled on the pattern it was derived from.
		// A silent mismatch would not crash: it would quietly condense the wrong entries, which is the
		// failure mode dev_docs/structural_assembly.md warns about, so the pattern is compared outright.
		if (jacobian.nrow() != nd || residuals.nrow() != nd)
			throw_runtime_error("Static condensation: the assembled system has " + std::to_string(jacobian.nrow()) +
								" rows but the elimination plan was built for " + std::to_string(nd) + ".");
		if ((unsigned long)jacobian.nnz() != plan->full_nnz)
			throw_runtime_error("Static condensation: the assembled Jacobian has " + std::to_string(jacobian.nnz()) +
								" nonzeros, but the elimination plan was built from a pattern with " +
								std::to_string(plan->full_nnz) + ". The frozen-sparsity assembly path must have "
								"declined for this call, and the plan's slot lists do not describe this matrix.");
		{
			const std::vector<int> pinned;
			const int fslot = this->acquire_frozen_sparsity(0, plan->generation, nd, pinned);
			if (fslot < 0)
				throw_runtime_error("Static condensation: the frozen sparsity pattern the elimination plan was built "
									"from is no longer available.");
			const FrozenSparsity &sp = frozen_sparsity_cache[fslot];
			if (std::memcmp(sp.row_start.data(), jacobian.row_start(), (size_t)(nd + 1) * sizeof(int)) != 0 ||
				std::memcmp(sp.column_index.data(), jacobian.column_index(), (size_t)plan->full_nnz * sizeof(int)) != 0)
				throw_runtime_error("Static condensation: the assembled Jacobian does not carry the frozen sparsity "
									"pattern the elimination plan was built from, although it has the same number of "
									"nonzeros. Refusing to condense a matrix whose entries are not where the plan says.");
		}

		const double *full_val = jacobian.value();
		double *r = residuals.values_pt();
		const unsigned ncol = jacobian.ncol();
		const unsigned cnnz = plan->cond_nnz();

		// --- 2. The condensed value array. Freshly allocated, because build_without_copy() below takes
		// ownership of it (and of the pattern arrays) and frees them with the matrix -- the same contract
		// assemble_with_frozen_sparsity() hands its immutable pattern out under.
		std::unique_ptr<double[]> cond_val(new double[cnnz ? cnnz : 1]);
		std::fill(cond_val.get(), cond_val.get() + cnnz, 0.0);
		const size_t n_pass = plan->passthrough_full_slot.size();
		for (size_t k = 0; k < n_pass; k++)
			cond_val[plan->passthrough_cond_slot[k]] = full_val[plan->passthrough_full_slot[k]];

		// --- 3. Eliminate each component. A_LL is block diagonal over the components by construction
		// (they are the connected components of its coupling graph), so the Schur complements simply add
		// up and each block can be inverted on its own.
		std::vector<double> &LL = condensation_scratch_LL;
		std::vector<double> &RHS = condensation_scratch_rhs;
		std::vector<double> &EL = condensation_scratch_EL;
		std::vector<int> &pivots = condensation_scratch_pivots;
		for (unsigned c = 0; c < plan->components.size(); c++)
		{
			CondensationComponent &comp = plan->components[c];
			const unsigned nl = comp.nL(), ne = comp.nE();
			const unsigned nrhs = ne + 1; // the nE columns of A_LE, plus r_L as one more right-hand side

			LL.assign((size_t)nl * nl, 0.0);
			RHS.assign((size_t)nl * nrhs, 0.0);
			EL.assign((size_t)ne * nl, 0.0);
			pivots.resize(nl);
			for (unsigned i = 0; i < nl; i++)
			{
				for (unsigned j = 0; j < nl; j++)
				{
					const int s = comp.LL_slots[(size_t)i * nl + j];
					if (s >= 0) LL[(size_t)i * nl + j] = full_val[s];
				}
				for (unsigned j = 0; j < ne; j++)
				{
					const int s = comp.LE_slots[(size_t)i * ne + j];
					if (s >= 0) RHS[(size_t)i * nrhs + j] = full_val[s];
				}
				RHS[(size_t)i * nrhs + ne] = r[comp.L_eqns[i]];
			}
			for (unsigned i = 0; i < ne; i++)
				for (unsigned j = 0; j < nl; j++)
				{
					const int s = comp.EL_slots[(size_t)i * nl + j];
					if (s >= 0) EL[(size_t)i * nl + j] = full_val[s];
				}

			const int bad = dense_lu_factorise(LL.data(), nl, pivots.data(), 1e-12);
			if (bad >= 0)
			{
				// Structurally the block was fine (build_condensation_plan checked that), so this is a
				// selection that eliminates a dof its own equations do not actually determine.
				std::vector<long> sample;
				for (unsigned i = 0; i < nl && i < 5; i++) sample.push_back(comp.L_eqns[i]);
				throw_runtime_error("Static condensation: the block of selected dofs is numerically singular and cannot "
									"be eliminated. The elimination of a connected component of " + std::to_string(nl) +
									" selected dofs broke down at step " + std::to_string(bad) + ": no pivot above "
									"1e-12 times the size of the block was left. The dofs of this component are: " +
									format_dof_samples(*this, sample, nl) +
									". The block passed the structural check, so the entries are present but cancel; the "
									"selection eliminates a dof that its equations do not determine. The classic case is "
									"taking the CONSTANT mode of a discontinuous (DL) pressure along with the bubble "
									"velocities: the bubble has no coupling to it (the integral of div(u_bubble) against "
									"a constant test function vanishes), so value 0 has to stay a global unknown and only "
									"the gradient modes may be condensed -- values=[1,2] in 2D, [1,2,3] in 3D.");
			}
			dense_lu_solve(LL.data(), nl, pivots.data(), RHS.data(), nrhs);

			// X = A_LL^-1 A_LE and y = A_LL^-1 r_L, kept for the reconstruction after the Newton update:
			// dx_L = y - X dx_E (stage 4).
			comp.X.resize((size_t)nl * ne);
			comp.y.resize(nl);
			for (unsigned i = 0; i < nl; i++)
			{
				for (unsigned j = 0; j < ne; j++) comp.X[(size_t)i * ne + j] = RHS[(size_t)i * nrhs + j];
				comp.y[i] = RHS[(size_t)i * nrhs + ne];
			}
			comp.ops_valid = true;

			// The Schur update. ACCUMULATE: the E_C x E_C fill slots overlap the pass-through entries
			// wherever the full pattern already had that (E,E) coupling, and two components sharing a
			// retained dof write to the same slots as well.
			for (unsigned a = 0; a < ne; a++)
			{
				const double *ELa = &EL[(size_t)a * nl];
				const int *fs = &comp.fill_slots[(size_t)a * ne];
				double racc = 0.0;
				for (unsigned l = 0; l < nl; l++) racc += ELa[l] * comp.y[l];
				r[comp.E_eqns[a]] -= racc;
				for (unsigned l = 0; l < nl; l++)
				{
					const double f = ELa[l];
					if (f == 0.0) continue;
					const double *Xl = &comp.X[(size_t)l * ne];
					for (unsigned b = 0; b < ne; b++) cond_val[fs[b]] -= f * Xl[b];
				}
			}
			// The eliminated equations are replaced by 0 = dx_L. Safe to do here rather than at the end:
			// E sets only ever contain RETAINED dofs, so this can never overwrite another component's r_L.
			for (unsigned i = 0; i < nl; i++) r[comp.L_eqns[i]] = 0.0;
		}

		// --- 4. Identity rows for the eliminated dofs (their row holds nothing but the diagonal).
		for (size_t i = 0; i < plan->L_diagonal_cond_slots.size(); i++) cond_val[plan->L_diagonal_cond_slots[i]] = 1.0;

		// --- 5. Rebuild the matrix on the condensed pattern. The pattern arrays have to be copied out of
		// the plan for the same reason the frozen path copies its own: build_without_copy() adopts them
		// and deletes them when the matrix is cleared, while the plan has to survive for the next Newton
		// step. It also frees the full arrays it currently holds, so nothing may read full_val after this.
		int *crs = new int[nd + 1];
		std::copy(plan->cond_row_start.begin(), plan->cond_row_start.end(), crs);
		int *cci = new int[cnnz ? cnnz : 1];
		std::copy(plan->cond_column_index.begin(), plan->cond_column_index.end(), cci);
		jacobian.build_without_copy(ncol, cnnz, cond_val.release(), cci, crs);
	}

#ifdef OOMPH_HAS_MPI
	// The distributed kernel (dev_docs/static_condensation.md section 9.5). Same algebra as the serial
	// one, with the work split by ownership: the rank owning a component's L rows forms X and y, ships
	// them to the ranks owning its E rows, and every participant applies the Schur update to the E rows
	// it owns. The pivot guard is voted on BEFORE the exchange, so a breakdown throws everywhere instead
	// of leaving the other ranks in MPI_Waitall.
	void Problem::apply_static_condensation_distributed(oomph::DoubleVector &residuals, oomph::CRDoubleMatrix &jacobian)
	{
		CondensationPlan *plan = &condensation_plan;
		const unsigned nproc = plan->nproc, my_rank = plan->my_rank;
		MPI_Comm comm = Communicator_pt->mpi_comm();

		// --- 1. Positive engagement check, on the local row block. Exactly the serial reasoning: the plan
		// is a list of positions in a particular value array, and a silent mismatch would not crash, it
		// would condense the wrong entries.
		if (jacobian.nrow() != plan->ndof || jacobian.nrow_local() != plan->nrow_local ||
			jacobian.first_row() != plan->first_row || residuals.nrow_local() != plan->nrow_local)
			throw_runtime_error("Static condensation: the assembled distributed system does not have the row "
								"block the elimination plan was built for (" + std::to_string(jacobian.nrow_local()) +
								" local rows from " + std::to_string(jacobian.first_row()) + ", plan: " +
								std::to_string(plan->nrow_local) + " from " + std::to_string(plan->first_row) + ").");
		if ((unsigned long)jacobian.nnz() != plan->full_nnz)
			throw_runtime_error("Static condensation: the assembled local Jacobian block has " +
								std::to_string(jacobian.nnz()) + " nonzeros, but the elimination plan was built "
								"from a pattern with " + std::to_string(plan->full_nnz) + ". The frozen "
								"distributed assembly must have declined for this call.");
		{
			const DistributedFrozenSparsity &sp = distributed_frozen_sparsity;
			if (sp.generation != plan->generation || sp.mats.empty())
				throw_runtime_error("Static condensation: the frozen distributed sparsity pattern the elimination "
									"plan was built from is no longer available.");
			const DistributedFrozenSparsity::Mat &mt = sp.mats[0];
			if (std::memcmp(mt.final_row_start.data(), jacobian.row_start(), (size_t)(plan->nrow_local + 1) * sizeof(int)) != 0 ||
				std::memcmp(mt.final_col.data(), jacobian.column_index(), (size_t)plan->full_nnz * sizeof(int)) != 0)
				throw_runtime_error("Static condensation: the assembled local Jacobian block does not carry the "
									"frozen pattern the elimination plan was built from. Refusing to condense a "
									"matrix whose entries are not where the plan says.");
		}

		const double *full_val = jacobian.value();
		double *r = residuals.values_pt();
		const unsigned first_row = plan->first_row;
		const unsigned cnnz = plan->cond_nnz();

		// --- 2. The condensed local value array, and the pass-through copy.
		std::unique_ptr<double[]> cond_val(new double[cnnz ? cnnz : 1]);
		std::fill(cond_val.get(), cond_val.get() + cnnz, 0.0);
		for (size_t k = 0; k < plan->passthrough_full_slot.size(); k++)
			cond_val[plan->passthrough_cond_slot[k]] = full_val[plan->passthrough_full_slot[k]];

		// --- 3. The components this rank owns: gather A_LL, A_LE and r_L (all in owned, hence complete,
		// rows), factorise once and solve for all nE+1 right-hand sides.
		std::vector<double> &LL = condensation_scratch_LL;
		std::vector<double> &RHS = condensation_scratch_rhs;
		std::vector<int> &pivots = condensation_scratch_pivots;
		std::string err;
		for (unsigned c = 0; c < plan->components.size(); c++)
		{
			CondensationComponent &comp = plan->components[c];
			if (comp.owner_rank != (int)my_rank) continue;
			const unsigned nl = comp.nL(), ne = comp.nE(), nrhs = ne + 1;
			LL.assign((size_t)nl * nl, 0.0);
			RHS.assign((size_t)nl * nrhs, 0.0);
			pivots.resize(nl);
			for (unsigned i = 0; i < nl; i++)
			{
				for (unsigned j = 0; j < nl; j++)
				{
					const int s = comp.LL_slots[(size_t)i * nl + j];
					if (s >= 0) LL[(size_t)i * nl + j] = full_val[s];
				}
				for (unsigned j = 0; j < ne; j++)
				{
					const int s = comp.LE_slots[(size_t)i * ne + j];
					if (s >= 0) RHS[(size_t)i * nrhs + j] = full_val[s];
				}
				RHS[(size_t)i * nrhs + ne] = r[comp.L_eqns[i] - (int)first_row];
			}
			const int bad = dense_lu_factorise(LL.data(), nl, pivots.data(), 1e-12);
			if (bad >= 0)
			{
				if (err.empty())
				{
					std::vector<long> sample;
					for (unsigned i = 0; i < nl && i < 5; i++) sample.push_back(comp.L_eqns[i]);
					err = "Static condensation: the block of selected dofs is numerically singular and cannot be "
						  "eliminated. The elimination of a connected component of " + std::to_string(nl) +
						  " selected dofs broke down at step " + std::to_string(bad) + ": no pivot above 1e-12 "
						  "times the size of the block was left. The dofs of this component are: " +
						  format_dof_samples(*this, sample, nl) +
						  ". The block passed the structural check, so the entries are present but cancel; the "
						  "selection eliminates a dof that its equations do not determine. The classic case is "
						  "taking the CONSTANT mode of a discontinuous (DL) pressure along with the bubble "
						  "velocities -- only the gradient modes may be condensed.";
				}
				continue;
			}
			dense_lu_solve(LL.data(), nl, pivots.data(), RHS.data(), nrhs);
			comp.X.resize((size_t)nl * ne);
			comp.y.resize(nl);
			for (unsigned i = 0; i < nl; i++)
			{
				for (unsigned j = 0; j < ne; j++) comp.X[(size_t)i * ne + j] = RHS[(size_t)i * nrhs + j];
				comp.y[i] = RHS[(size_t)i * nrhs + ne];
			}
			comp.ops_valid = true;
			// The state this linearisation belongs to, for the E dofs whose rows another rank owns: the
			// reconstruction recovers their increment as (u_stored - u_now)/Relaxation_factor rather than
			// from dx, which this rank does not have for them. Captured here because the dofs still carry
			// the state the Jacobian was formed at.
			for (unsigned j = 0; j < ne; j++)
				if (comp.E_value_pt[j]) comp.E_u_stored[j] = *(comp.E_value_pt[j]);
		}
		// Collective, and before any point-to-point traffic: a rank throwing on its own here would leave
		// every other rank in the MPI_Waitall below.
		collective_throw(err);

		// --- 4. Ship X and y to the ranks owning this component's E rows. One message per peer, laid out
		// component by component in the order both sides agreed on when the plan was built.
		std::vector<std::vector<double>> xy_send(nproc), xy_recv(nproc);
		std::vector<MPI_Request> reqs;
		reqs.reserve(2 * nproc);
		for (unsigned p = 0; p < nproc; p++)
		{
			if (p == my_rank) continue;
			size_t ns = 0;
			for (int k = plan->xy_send_start[p]; k < plan->xy_send_start[p + 1]; k++)
			{
				const CondensationComponent &comp = plan->components[plan->xy_send_comp[k]];
				ns += (size_t)comp.nL() * comp.nE() + comp.nL();
			}
			if (ns)
			{
				xy_send[p].reserve(ns);
				for (int k = plan->xy_send_start[p]; k < plan->xy_send_start[p + 1]; k++)
				{
					const CondensationComponent &comp = plan->components[plan->xy_send_comp[k]];
					xy_send[p].insert(xy_send[p].end(), comp.X.begin(), comp.X.end());
					xy_send[p].insert(xy_send[p].end(), comp.y.begin(), comp.y.end());
				}
				reqs.push_back(MPI_Request());
				MPI_Isend(xy_send[p].data(), (int)xy_send[p].size(), MPI_DOUBLE, (int)p, 84, comm, &reqs.back());
			}
			size_t nr = 0;
			for (int k = plan->xy_recv_start[p]; k < plan->xy_recv_start[p + 1]; k++)
			{
				const CondensationComponent &comp = plan->components[plan->xy_recv_comp[k]];
				nr += (size_t)comp.nL() * comp.nE() + comp.nL();
			}
			if (nr)
			{
				xy_recv[p].resize(nr);
				reqs.push_back(MPI_Request());
				MPI_Irecv(xy_recv[p].data(), (int)nr, MPI_DOUBLE, (int)p, 84, comm, &reqs.back());
			}
		}

		// The Schur update of one component into the E rows THIS rank owns. Accumulating, for the same two
		// reasons as serially: the fill slots overlap the pass-through entries wherever the full pattern
		// already had that (E,E) coupling, and two components sharing a retained dof write to the same
		// slots.
		std::vector<double> &EL = condensation_scratch_EL;
		auto apply_component = [&](CondensationComponent &comp)
		{
			const unsigned nl = comp.nL(), ne = comp.nE(), nmy = (unsigned)comp.my_E.size();
			if (!nmy) return;
			EL.assign((size_t)nmy * nl, 0.0);
			for (unsigned q = 0; q < nmy; q++)
				for (unsigned j = 0; j < nl; j++)
				{
					const int s = comp.EL_slots[(size_t)q * nl + j];
					if (s >= 0) EL[(size_t)q * nl + j] = full_val[s];
				}
			for (unsigned q = 0; q < nmy; q++)
			{
				const double *ELq = &EL[(size_t)q * nl];
				const int *fs = &comp.fill_slots[(size_t)q * ne];
				double racc = 0.0;
				for (unsigned l = 0; l < nl; l++) racc += ELq[l] * comp.y[l];
				r[comp.E_eqns[comp.my_E[q]] - (int)first_row] -= racc;
				for (unsigned l = 0; l < nl; l++)
				{
					const double f = ELq[l];
					if (f == 0.0) continue;
					const double *Xl = &comp.X[(size_t)l * ne];
					for (unsigned b = 0; b < ne; b++) cond_val[fs[b]] -= f * Xl[b];
				}
			}
		};

		// Our own components need no message; do them while the others are in flight.
		for (unsigned c = 0; c < plan->components.size(); c++)
			if (plan->components[c].owner_rank == (int)my_rank) apply_component(plan->components[c]);

		if (!reqs.empty()) MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

		for (unsigned p = 0; p < nproc; p++)
		{
			if (p == my_rank || xy_recv[p].empty()) continue;
			const double *buf = xy_recv[p].data();
			for (int k = plan->xy_recv_start[p]; k < plan->xy_recv_start[p + 1]; k++)
			{
				CondensationComponent &comp = plan->components[plan->xy_recv_comp[k]];
				const size_t nx = (size_t)comp.nL() * comp.nE();
				comp.X.assign(buf, buf + nx);
				buf += nx;
				comp.y.assign(buf, buf + comp.nL());
				buf += comp.nL();
				// ops_valid stays false: only the owner reconstructs the eliminated values, and it is the
				// only rank that holds them non-halo.
				apply_component(comp);
			}
		}

		// --- 5. The eliminated equations become 0 = dx_L, and their rows the identity. Both live entirely
		// on the owner, whose rows they are.
		for (unsigned c = 0; c < plan->components.size(); c++)
		{
			CondensationComponent &comp = plan->components[c];
			if (comp.owner_rank != (int)my_rank) continue;
			for (unsigned i = 0; i < comp.nL(); i++) r[comp.L_eqns[i] - (int)first_row] = 0.0;
		}
		for (size_t i = 0; i < plan->L_diagonal_cond_slots.size(); i++) cond_val[plan->L_diagonal_cond_slots[i]] = 1.0;

		// --- 6. Rebuild the local block on the condensed pattern; the distribution is untouched by
		// build_without_copy(), which keeps nrow_local and first_row.
		int *crs = new int[plan->nrow_local + 1];
		std::copy(plan->cond_row_start.begin(), plan->cond_row_start.end(), crs);
		int *cci = new int[cnnz ? cnnz : 1];
		std::copy(plan->cond_column_index.begin(), plan->cond_column_index.end(), cci);
		jacobian.build_without_copy(plan->ndof, cnnz, cond_val.release(), cci, crs);
	}
#endif

	// ==================== Static condensation: reconstruction after the Newton update ============
	// The vendored oomph-lib hook (src/thirdparty/INFO_oomph-lib), called from Problem::newton_solve()
	// once the dofs have taken the increment and the halos have been synchronised. The condensed system
	// gave the eliminated dofs an identity row with a zero right-hand side, so dx is zero there and the
	// update loop left them untouched; their real increment is dx_L = y_C - X_C dx_{E_C}, which is what
	// this applies. Stage 4 of dev_docs/static_condensation.md.
	void Problem::actions_after_newton_dof_update(const oomph::DoubleVector &dx)
	{
		// The overwhelmingly common case, and the reason this is the first line: with condensation off
		// (or declined for this assembly) the hook costs one bool test per Newton step.
		if (!last_jacobian_was_condensed) return;

		CondensationPlan &plan = condensation_plan;
		if (!plan.generation)
			throw_runtime_error("Internal error: the last Jacobian was condensed but the elimination plan is "
								"gone, so the eliminated degrees of freedom cannot be reconstructed.");
		if (dx.nrow() != plan.ndof)
			throw_runtime_error("Internal error: the Newton increment has " + std::to_string(dx.nrow()) +
								" rows but static condensation eliminated from a system of " +
								std::to_string(plan.ndof) + ".");
#ifdef OOMPH_HAS_MPI
		if (plan.distributed) { reconstruct_condensed_dofs_distributed(dx); return; }
#endif

		// The ONE place that knows how a global equation number turns into an entry of dx, and the only
		// thing the distributed stage has to replace. Serially dx is not distributed, so the global
		// equation number is the index. Distributed (stage 7), a component's E set may reach into rows
		// this rank does not own, and the plan reconstructs dx_E from the halo-synchronised VALUES,
		// (u_old - u_new)/Relaxation_factor, rather than from dx -- which is why the reconstruction was
		// formulated this way in the first place: no halo exchange of the increment is ever needed.
		const double *dx_pt = dx.values_pt();
		auto dx_of = [dx_pt](int eqn) -> double { return dx_pt[eqn]; };

		// oomph applies Relaxation_factor to the retained increment; the eliminated dofs are part of the
		// same Newton step and must be damped by exactly the same amount, or the reconstruction would
		// describe a different linearised state than the one the retained dofs moved to.
		const double relax = Relaxation_factor;
		for (unsigned c = 0; c < plan.components.size(); c++)
		{
			CondensationComponent &comp = plan.components[c];
			// The operators belong to one linearisation and are consumed by one Newton step. Reaching
			// here twice without an assembly in between would silently apply the same increment again.
			if (!comp.ops_valid)
				throw_runtime_error("Internal error: static condensation was asked to reconstruct the eliminated "
									"degrees of freedom twice from the same Jacobian. The elimination operators of "
									"a component are consumed by the Newton step they were built for.");
			const unsigned nl = comp.nL(), ne = comp.nE();
			for (unsigned k = 0; k < nl; k++)
			{
				const double *Xk = &comp.X[(size_t)k * ne];
				double v = comp.y[k];
				for (unsigned j = 0; j < ne; j++) v -= Xk[j] * dx_of(comp.E_eqns[j]);
				oomph::Data *d = comp.L_values[k].first;
				const unsigned iv = comp.L_values[k].second;
#ifdef PARANOID
				// The selection only ever takes unpinned values, and pinning one changes the equation
				// numbering and hence the pattern id, which rebuilds the plan -- so this can only fire if
				// something pinned a dof without invalidating the structure. Cheap, but not free: it is a
				// pointer chase per eliminated dof, hence PARANOID only.
				if (d->eqn_number(iv) != comp.L_eqns[k])
					throw_runtime_error("Internal error: a static condensation plan outlived its equation numbering "
										"(a condensed value now carries equation " + std::to_string(d->eqn_number(iv)) +
										" instead of " + std::to_string(comp.L_eqns[k]) + ").");
#endif
				*(d->value_pt(iv)) -= relax * v;
			}
			comp.ops_valid = false;
		}
	}

#ifdef OOMPH_HAS_MPI
	// The distributed reconstruction (dev_docs/static_condensation.md section 9.6). Only the rank owning
	// a component reconstructs it -- it is the only one holding those values non-halo -- and the hook
	// therefore ends with one synchronise_all_dofs(), because the halo copies of the values just written
	// are stale until then and a neighbour's non-halo interface or facet element may read exactly them.
	void Problem::reconstruct_condensed_dofs_distributed(const oomph::DoubleVector &dx)
	{
		CondensationPlan &plan = condensation_plan;
		// Replicated, newton_solve() redistributes dx to the problem's dof distribution, which is not
		// distributed there: it comes back whole on every rank, and E_dx_row holds global rows.
		const unsigned want_nrow = plan.dx_is_global ? plan.ndof : plan.nrow_local;
		const unsigned want_first = plan.dx_is_global ? 0u : plan.first_row;
		if (dx.nrow_local() != want_nrow || dx.first_row() != want_first)
			throw_runtime_error("Internal error: the Newton increment's row block (" +
								std::to_string(dx.nrow_local()) + " rows from " + std::to_string(dx.first_row()) +
								") is not the one static condensation eliminated from (" +
								std::to_string(want_nrow) + " from " + std::to_string(want_first) + ").");
		const double *dx_pt = dx.values_pt();
		const double relax = Relaxation_factor;
		// A zero relaxation factor moves no dof at all, so there is nothing to reconstruct -- and the
		// value-difference recovery below would be 0/0.
		const bool do_work = (relax != 0.0);
		std::vector<int> pairs_eqn;      // replicated only, see the allgather after the loop
		std::vector<double> pairs_delta;
		for (unsigned c = 0; c < plan.components.size(); c++)
		{
			CondensationComponent &comp = plan.components[c];
			if (comp.owner_rank != (int)plan.my_rank) continue;
			if (!comp.ops_valid)
				throw_runtime_error("Internal error: static condensation was asked to reconstruct the eliminated "
									"degrees of freedom twice from the same Jacobian. The elimination operators of "
									"a component are consumed by the Newton step they were built for.");
			comp.ops_valid = false;
			if (!do_work) continue;
			const unsigned nl = comp.nL(), ne = comp.nE();
			// dx_E, without any exchange of dx: taken straight from dx where this rank owns the row
			// (exact), and recovered from the halo-synchronised VALUE otherwise. synchronise_all_dofs()
			// ran a few lines above in newton_solve(), and it copies haloed node values AND haloed
			// elements' internal data, so the local copy equals the owner's at both instants and their
			// difference is exactly Relaxation_factor times the increment.
			std::vector<double> &dxE = condensation_scratch_dxE;
			dxE.resize(ne);
			for (unsigned j = 0; j < ne; j++)
				dxE[j] = (comp.E_dx_row[j] >= 0 ? dx_pt[comp.E_dx_row[j]]
												: (comp.E_u_stored[j] - *(comp.E_value_pt[j])) / relax);
			for (unsigned k = 0; k < nl; k++)
			{
				const double *Xk = &comp.X[(size_t)k * ne];
				double v = comp.y[k];
				for (unsigned j = 0; j < ne; j++) v -= Xk[j] * dxE[j];
				oomph::Data *d = comp.L_values[k].first;
				const unsigned iv = comp.L_values[k].second;
#ifdef PARANOID
				if (d->eqn_number(iv) != comp.L_eqns[k])
					throw_runtime_error("Internal error: a static condensation plan outlived its equation numbering "
										"(a condensed value now carries equation " + std::to_string(d->eqn_number(iv)) +
										" instead of " + std::to_string(comp.L_eqns[k]) + ").");
#endif
				// Replicated, this is recorded and allgathered instead of applied: every rank must end up
				// with the identical dof vector, and only this rank can compute this number.
				if (plan.dx_is_global) { pairs_eqn.push_back(comp.L_eqns[k]); pairs_delta.push_back(relax * v); }
				else *(d->value_pt(iv)) -= relax * v;
			}
		}
		if (plan.dx_is_global)
		{
			// One MPI_Allgatherv of (equation, increment) pairs -- nL doubles in total, not the
			// nL*(nE+1) that shipping X and y would cost. Then EVERY rank applies EVERY pair, its own
			// included, so the result is bit-identical everywhere by construction rather than by
			// everyone happening to compute the same rounding.
			const unsigned nproc = Communicator_pt->nproc();
			MPI_Comm comm = Communicator_pt->mpi_comm();
			int mine = (int)pairs_eqn.size();
			std::vector<int> counts(nproc, 0), displs(nproc + 1, 0);
			MPI_Allgather(&mine, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
			for (unsigned p = 0; p < nproc; p++) displs[p + 1] = displs[p] + counts[p];
			std::vector<int> all_eqn(displs[nproc]);
			std::vector<double> all_delta(displs[nproc]);
			MPI_Allgatherv(pairs_eqn.data(), mine, MPI_INT, all_eqn.data(), counts.data(), displs.data(), MPI_INT, comm);
			MPI_Allgatherv(pairs_delta.data(), mine, MPI_DOUBLE, all_delta.data(), counts.data(), displs.data(), MPI_DOUBLE, comm);
			for (size_t i = 0; i < all_eqn.size(); i++)
			{
				const std::vector<int>::const_iterator f =
					std::lower_bound(plan.all_L_eqn.begin(), plan.all_L_eqn.end(), all_eqn[i]);
				if (f == plan.all_L_eqn.end() || *f != all_eqn[i])
					throw_runtime_error("Internal error: rank " + std::to_string(plan.my_rank) + " received a "
										"reconstructed increment for equation " + std::to_string(all_eqn[i]) +
										", which its static condensation plan does not list as condensed.");
				*(plan.all_L_value_pt[f - plan.all_L_eqn.begin()]) -= all_delta[i];
			}
			return; // no halos to synchronise on a replicated run
		}
		// Collective, and reached by every rank because last_jacobian_was_condensed is globally uniform
		// (see the engagement gate). Deliberately NOT conditioned on this rank having reconstructed
		// anything: that is a per-rank quantity, and a rank skipping a collective is exactly the deadlock
		// this whole design is written to avoid.
		this->synchronise_all_dofs();
	}
#endif

	// Whether this get_jacobian() call must hand back the condensed system. See
	// dev_docs/static_condensation.md; the refusals throw and the declines are silent on purpose.
	bool Problem::static_condensation_engages_now()
	{
		if (!use_static_condensation) return false;
		// Only inside a Newton solve entered through one of the pyoomph wrappers, because only there does
		// the reconstruction hook run afterwards. The debug lever is the test-only exception: it hands
		// back the condensed system from a bare get_jacobian() so the algebra can be refereed at a fixed
		// state, with nothing updating any dof.
		if (!inside_flagged_newton_solve && !_debug_force_condensed_assembly) return false;

		// Refusals: combinations the user has explicitly asked for that cannot be served. Every condition
		// tested here is globally uniform, so under MPI these throw on all ranks at once rather than
		// leaving some of them in the next collective (dev_docs/static_condensation.md section 9.4).
#ifdef OOMPH_HAS_MPI
		// Both MPI modes are served. What they share is that the Jacobian's rows are split across the
		// ranks, which is what the whole distributed plan is expressed in terms of -- and that plan is a
		// list of positions in the frozen distributed assembly's value array, so that assembly is a
		// prerequisite in either mode.
		if (Communicator_pt && Communicator_pt->nproc() > 1 &&
			!(use_frozen_sparsity && use_frozen_distributed_sparsity && keep_structural_zeros))
			throw_runtime_error("Static condensation under MPI requires the frozen distributed assembly: the "
								"elimination plan is a list of positions in its value array. Leave "
								"problem.use_frozen_sparsity, problem.use_frozen_distributed_sparsity and "
								"problem.keep_structural_zeros on, or switch static condensation off.");
#endif
		if (this->jacobian_reuse_is_enabled())
			throw_runtime_error("Static condensation cannot be combined with Jacobian reuse: the eliminated dofs are "
								"reconstructed from operators belonging to the linearisation the matrix was built at, "
								"and a reused Jacobian is by definition an older one. Disable one of the two.");
		if (this->globally_convergent_newton_method_is_enabled())
			throw_runtime_error("Static condensation cannot be combined with the globally convergent (line search) "
								"Newton method: the line search rescales the increment of the retained dofs after the "
								"solve, while the eliminated ones would be reconstructed from the unscaled one. Disable "
								"one of the two.");

		// Declines: assemblies that legitimately need the full system. Silent, because nothing was asked
		// for that cannot be given -- these routes are not the flagged Newton solve at all.
		oomph::AssemblyHandler *ah = this->assembly_handler_pt();
		if (!ah || typeid(*ah) != typeid(oomph::AssemblyHandler)) return false; // bifurcation tracking, eigen, ...
		if (n_unaugmented_dofs != 0) return false;                             // augmented (continuation) system
		return this->acquire_condensation_plan() != NULL;                      // nothing selected / no usable pattern
	}

	// Builds the CSR pattern and the elemental scatter map for one matrix of the assembly, from the same
	// symbolic masks the ordinary assembly consults (sparsity_mask_for_element). Two passes over the
	// elements: the first collects the column indices each row can ever hold, the second records, for
	// every position of every elemental block, which slot of the value array it lands in.
	//
	// Returns false - and leaves sp cleared - if any element has no symbolic mask. That is not a
	// failure: it means the pattern for this problem is only knowable from the values, so the caller
	// must fall back to oomph-lib's assembly.
	// The Jacobian and mass-matrix masks OR'd together with their transposes.
	//
	// A multi-assembly does not only build derivatives of the residual with respect to the dofs; it can
	// also be asked for TRANSPOSED Hessian-vector products, which bifurcation tracking on a left
	// eigenvector does. The transpose of a matrix does not live on that matrix's pattern unless the
	// pattern happens to be symmetric, so a single Jacobian mask is not a superset of everything the
	// routine writes -- which is exactly how the Lorenz Hopf case ended up with (H^T V)[0][2] nonzero
	// while the Jacobian has no (0,2) at all (row 0 is sigma*(y-x), so it cannot depend on z).
	//
	// Symmetrising covers both orientations at once, which keeps the "one pattern serves every matrix"
	// structure that makes this path fast. The price is the entries of J^T that J lacks; on a symmetric
	// coupling graph that is nothing, and this is the augmented-system path, which was the slowest in
	// the codebase before it was frozen.
	const char *Problem::symmetrised_union_mask(oomph::GeneralisedElement *elem_pt, unsigned nvar)
	{
		const char *mj = this->sparsity_mask_for_element(0, elem_pt, nvar);
		if (!mj) return 0; // No symbolic description: the caller falls back, as with any single mask
		const char *mm = this->sparsity_mask_for_element(1, elem_pt, nvar);
		const size_t n = (size_t)nvar * nvar;
		union_mask_scratch.assign(n, 0);
		for (unsigned i = 0; i < nvar; i++)
		{
			for (unsigned j = 0; j < nvar; j++)
			{
				const size_t ij = (size_t)i * nvar + j, ji = (size_t)j * nvar + i;
				if (mj[ij] || mj[ji] || (mm && (mm[ij] || mm[ji]))) union_mask_scratch[ij] = 1;
			}
		}
		return union_mask_scratch.data();
	}


	bool Problem::build_frozen_sparsity(unsigned mask_matrix_index, FrozenSparsity &sp, unsigned nd_override)
	{
		sp.clear();
		const unsigned long n_element = mesh_pt()->nelement();
		const unsigned nd = (nd_override ? nd_override : this->ndof());
		const unsigned matrix_index = mask_matrix_index;
		oomph::AssemblyHandler *const assembly_handler_pt = this->assembly_handler_pt();

		// Pass 1: which columns can appear in each row, and how much scatter space each element needs.
		std::vector<std::vector<int>> cols_of_row(nd);
		for (unsigned long e = 0; e < n_element; e++)
		{
			oomph::GeneralisedElement *elem_pt = mesh_pt()->element_pt(e);
			const unsigned nvar = assembly_handler_pt->ndof(elem_pt);
			const char *mask = (matrix_index == MASK_UNION_SYMMETRIC)
								   ? this->symmetrised_union_mask(elem_pt, nvar)
								   : this->sparsity_mask_for_element(matrix_index, elem_pt, nvar);
			if (!mask && nvar)
			{
				sp.clear();
				return false; // No symbolic description of this element: the pattern is value-dependent
			}
			for (unsigned i = 0; i < nvar; i++)
			{
				const int row = (int)assembly_handler_pt->eqn_number(elem_pt, i);
				if (row < 0 || (unsigned)row >= nd) continue;
				for (unsigned j = 0; j < nvar; j++)
				{
					if (!mask[(size_t)i * nvar + j]) continue;
					const int col = (int)assembly_handler_pt->eqn_number(elem_pt, j);
					if (col < 0) continue;
					cols_of_row[row].push_back(col);
				}
			}
		}

		// Compress the per-row column lists into CSR.
		sp.row_start.assign(nd + 1, 0);
		for (unsigned r = 0; r < nd; r++)
		{
			std::vector<int> &c = cols_of_row[r];
			std::sort(c.begin(), c.end());
			c.erase(std::unique(c.begin(), c.end()), c.end());
			sp.row_start[r + 1] = sp.row_start[r] + (int)c.size();
		}
		sp.column_index.resize(sp.row_start[nd]);
		for (unsigned r = 0; r < nd; r++)
		{
			std::copy(cols_of_row[r].begin(), cols_of_row[r].end(), sp.column_index.begin() + sp.row_start[r]);
		}

		// Pass 2: resolve each contributing position of each elemental block to its slot in the value
		// array, and store the pairs sorted by slot so that assembly writes sweep forward through the
		// value array instead of hopping between rows.
		sp.element_offset.assign(n_element + 1, 0);
		sp.scatter_source.clear();
		sp.scatter_slot.clear();
		sp.scatter_source.reserve(sp.column_index.size());
		sp.scatter_slot.reserve(sp.column_index.size());
		std::vector<std::pair<int, int>> pairs; // (slot, source offset) for one element
		for (unsigned long e = 0; e < n_element; e++)
		{
			oomph::GeneralisedElement *elem_pt = mesh_pt()->element_pt(e);
			const unsigned nvar = assembly_handler_pt->ndof(elem_pt);
			pairs.clear();
			if (nvar)
			{
				const char *mask = (matrix_index == MASK_UNION_SYMMETRIC)
									   ? this->symmetrised_union_mask(elem_pt, nvar)
									   : this->sparsity_mask_for_element(matrix_index, elem_pt, nvar);
				if (!mask) { sp.clear(); return false; }
				for (unsigned i = 0; i < nvar; i++)
				{
					const int row = (int)assembly_handler_pt->eqn_number(elem_pt, i);
					if (row < 0 || (unsigned)row >= nd) continue;
					const int *row_cols = &sp.column_index[sp.row_start[row]];
					const int row_len = sp.row_start[row + 1] - sp.row_start[row];
					for (unsigned j = 0; j < nvar; j++)
					{
						if (!mask[(size_t)i * nvar + j]) continue;
						const int col = (int)assembly_handler_pt->eqn_number(elem_pt, j);
						if (col < 0) continue;
						const int *found = std::lower_bound(row_cols, row_cols + row_len, col);
						// Cannot miss: pass 1 put exactly this (row,col) into the list.
						pairs.push_back(std::make_pair(sp.row_start[row] + (int)(found - row_cols),
													   (int)((size_t)i * nvar + j)));
					}
				}
				std::sort(pairs.begin(), pairs.end());
			}
			for (size_t k = 0; k < pairs.size(); k++)
			{
				sp.scatter_slot.push_back(pairs[k].first);
				sp.scatter_source.push_back(pairs[k].second);
			}
			sp.element_offset[e + 1] = (int)sp.scatter_slot.size();
		}

		sp.ndof = nd;
		sp.nelement = n_element;
		return true;
	}

	// Fast assembly: no container, no sort, no compression pass, no allocation of the index arrays from
	// scratch - just zero the value array and add each element's block into it through the precomputed
	// scatter map. Returns false, having touched nothing, whenever the frozen route does not apply.
	bool Problem::assemble_with_frozen_sparsity(oomph::Vector<int*>& column_or_row_index, oomph::Vector<int*>& row_or_column_start, oomph::Vector<double*>& value, oomph::Vector<unsigned>& nnz, oomph::Vector<double*>& residuals, bool compressed_row_flag)
	{
		if (!use_frozen_sparsity || !keep_structural_zeros) return false;
		if (!compressed_row_flag) return false; // Compressed-column output is not worth a second scatter map
		if (n_unaugmented_dofs != 0) return false; // Augmented system: handled by the base-problem routine
#ifdef OOMPH_HAS_MPI
		// The distributed assembly is a different routine with its own row distribution and inter-rank
		// exchange; freezing that is Phase 2b.
		if (Problem_has_been_distributed) return false;
		if (Communicator_pt && Communicator_pt->nproc() > 1) return false;
#endif
		const unsigned long gen = this->get_jacobian_structure_id();
		if (!gen) return false; // Pattern is value-dependent

		const unsigned n_matrix = column_or_row_index.size();
		const unsigned n_vector = residuals.size();
		const unsigned nd = this->ndof();
		std::vector<int> slots(n_matrix, -1);
		for (unsigned m = 0; m < n_matrix; m++)
		{
			slots[m] = this->acquire_frozen_sparsity(m, gen, nd, slots);
			if (slots[m] < 0) return false; // Not describable: fall back to oomph-lib's assembly
		}

		// Hand out the pattern. These arrays must be freshly allocated every time even though their
		// contents never change: CRDoubleMatrix::build_without_copy() takes ownership and frees them.
		for (unsigned m = 0; m < n_matrix; m++)
		{
			const FrozenSparsity &sp = frozen_sparsity_cache[slots[m]];
			nnz[m] = sp.nnz();
			row_or_column_start[m] = new int[nd + 1];
			std::copy(sp.row_start.begin(), sp.row_start.end(), row_or_column_start[m]);
			column_or_row_index[m] = new int[nnz[m] ? nnz[m] : 1];
			std::copy(sp.column_index.begin(), sp.column_index.end(), column_or_row_index[m]);
			value[m] = new double[nnz[m] ? nnz[m] : 1];
			std::fill(value[m], value[m] + nnz[m], 0.0);
		}
		for (unsigned v = 0; v < n_vector; v++)
		{
			residuals[v] = new double[nd ? nd : 1];
			std::fill(residuals[v], residuals[v] + nd, 0.0);
		}

		oomph::AssemblyHandler *const assembly_handler_pt = this->assembly_handler_pt();
		const unsigned long n_element = mesh_pt()->nelement();
		// Threaded element loop, if one was asked for and nothing rules it out. It fills exactly the
		// same arrays with exactly the same bits (dev_docs/openmp_assembly.md); the serial loop below is
		// left untouched and is what runs whenever --omp was not given, which is the default.
		if (this->try_parallel_frozen_assembly(slots, value, residuals, 0, n_element))
		{
			// The frozen-fill statistics below still want the finished value arrays, so fall through to
			// them rather than returning here.
		}
		else
		{
			oomph::Vector<oomph::Vector<double>> el_residuals(n_vector);
			oomph::Vector<oomph::DenseMatrix<double>> el_jacobian(n_matrix);
			for (unsigned long e = 0; e < n_element; e++)
			{
				oomph::GeneralisedElement *elem_pt = mesh_pt()->element_pt(e);
				const unsigned nvar = assembly_handler_pt->ndof(elem_pt);
				if (!nvar) continue;
				for (unsigned v = 0; v < n_vector; v++) el_residuals[v].resize(nvar);
				for (unsigned m = 0; m < n_matrix; m++) el_jacobian[m].resize(nvar);
				assembly_handler_pt->get_all_vectors_and_matrices(elem_pt, el_residuals, el_jacobian);

				for (unsigned i = 0; i < nvar; i++)
				{
					const unsigned eqn_number = assembly_handler_pt->eqn_number(elem_pt, i);
					for (unsigned v = 0; v < n_vector; v++) residuals[v][eqn_number] += el_residuals[v][i];
				}
				for (unsigned m = 0; m < n_matrix; m++)
				{
					const FrozenSparsity &sp = frozen_sparsity_cache[slots[m]];
					const int lo = sp.element_offset[e], hi = sp.element_offset[e + 1];
					const int *slot = &sp.scatter_slot[0];
					const int *src = &sp.scatter_source[0];
					double *val = value[m];
					const double *flat = &el_jacobian[m](0, 0); // Row-major nvar*nvar block
					for (int k = lo; k < hi; k++) val[slot[k]] += flat[src[k]];
					if (verify_frozen_sparsity)
					{
						// The compact scatter never looks at the positions the pattern omits, so nothing would
						// notice if the symbolic mask under-reported and a real entry landed there -- a
						// silently truncated Jacobian. Count the nonzeros of the whole block and of the part
						// actually scattered: equal iff everything omitted was exactly zero. Counting rather
						// than summing on purpose -- a sum over the block and a sum over the slot-sorted
						// subset differ in rounding, so a magnitude comparison reports spurious mismatches.
						// O(nvar^2) reads of a block already in L1, against O(nvar^2) cache-missing writes for
						// the scatter itself.
						unsigned all = 0, taken = 0;
						const size_t nblock = (size_t)nvar * nvar;
						for (size_t q = 0; q < nblock; q++) all += (flat[q] != 0.0);
						for (int k = lo; k < hi; k++) taken += (flat[src[k]] != 0.0);
						if (all > taken)
						{
							// Name the offending positions: which part of the block escaped decides what is
							// wrong. On an AUGMENTED (bifurcation-tracking) block the row and column say which
							// sub-block of the handler's spec is at fault, which "element 898" alone does not.
							std::string where;
							unsigned shown = 0;
							std::vector<bool> covered(nblock, false);
							for (int k = lo; k < hi; k++) covered[src[k]] = true;
							for (size_t q = 0; q < nblock && shown < 6; q++)
							{
								if (flat[q] == 0.0 || covered[q]) continue;
								where += (shown ? ", " : "") + std::string("(") + std::to_string(q / nvar) + "," +
										 std::to_string(q % nvar) + ")=" + std::to_string(flat[q]);
								shown++;
							}
							throw_runtime_error("A Jacobian entry appeared outside the symbolic sparsity pattern (matrix " +
												std::to_string(m) + ", element " + std::to_string(e) + ", nvar " +
												std::to_string(nvar) + "). Positions missing from the pattern: " + where +
												". Silently dropping them would truncate the Jacobian, so assembly is "
												"refused. Workaround: set problem.use_frozen_sparsity=False.");
						}
					}
				}
			}
		}
		// Diagnostic breakdown, off unless PYOOMPH_FROZEN_FILL_BREAKDOWN is set: attribute every scattered
		// entry to the (row class, column class) pair it came from and count how many were zero. That
		// distinguishes the two reasons a frozen pattern carries zeros, which want opposite fixes: an
		// entire (class,class) block reading 100% zero means the symbolic coupling table over-marks and
		// can be tightened, while zeros spread thinly through blocks that are mostly nonzero are terms
		// that are symbolically real but numerically vanish at these parameter values -- inherent to a
		// pattern that must stay valid when the parameters change.
		if (getenv("PYOOMPH_FROZEN_FILL_BREAKDOWN"))
		{
			for (unsigned long e = 0; e < n_element; e++)
			{
				oomph::GeneralisedElement *elem_pt = mesh_pt()->element_pt(e);
				BulkElementBase *be = dynamic_cast<BulkElementBase *>(elem_pt);
				if (!be) continue;
				const unsigned nvar = assembly_handler_pt->ndof(elem_pt);
				if (!nvar) continue;
				const JITFuncSpec_Table_FiniteElement_t *ft = be->get_jit_code()->get_func_table();
				const std::vector<int> &cidx = be->get_local_dof_contribution_indices();
				if (cidx.size() != nvar) continue;
				for (unsigned m = 0; m < n_matrix; m++)
				{
					const FrozenSparsity &sp = frozen_sparsity_cache[slots[m]];
					const int lo = sp.element_offset[e], hi = sp.element_offset[e + 1];
					for (int k = lo; k < hi; k++)
					{
						const int src = sp.scatter_source[k];
						const int ci = cidx[src / nvar], cj = cidx[src % nvar];
						// Unattributed/none dofs are labelled with the CODE they belong to: knowing which
						// element type fails to attribute its dofs is the whole point of the breakdown.
						auto name = [&](int c) -> std::string {
							if (c < 0)
							{
								std::string who = (ft->domain_name ? std::string(ft->domain_name) : std::string("?"));
								return who + (c == -2 ? " <contributes to nothing>" : " <unattributed>");
							}
							if (ft->contribution_names && (unsigned)c < ft->contribution_entries_size)
								return std::string(ft->contribution_names[c]);
							return std::string("<out of range>");
						};
						auto &cell = frozen_fill_breakdown[std::make_pair(name(ci), name(cj))];
						cell.first++;
						if (value[m][sp.scatter_slot[k]] == 0.0) cell.second++;
					}
				}
			}
		}

		// Price of freezing: how many of the slots the pattern reserved actually ended up zero. The
		// pattern must be a superset of the nonzeros to be reusable at all, so this is never 0 in
		// general; a large fraction means the symbolic description is far looser than the matrix and
		// the solver is carrying stored zeros for nothing. One pass over values still in cache.
		for (unsigned m = 0; m < n_matrix; m++)
		{
			const double *val = value[m];
			unsigned long long zeros = 0;
			for (unsigned k = 0; k < nnz[m]; k++) zeros += (val[k] == 0.0);
			frozen_sparsity_stored_entries += nnz[m];
			frozen_sparsity_zero_entries += zeros;
			if (frozen_sparsity_stored_by_matrix.size() <= m)
			{
				frozen_sparsity_stored_by_matrix.resize(m + 1, 0);
				frozen_sparsity_zero_by_matrix.resize(m + 1, 0);
			}
			frozen_sparsity_stored_by_matrix[m] += nnz[m];
			frozen_sparsity_zero_by_matrix[m] += zeros;
		}
		return true;
	}

	// Defined OUTSIDE the OOMPH_HAS_MPI block below: problem.hpp declares it unconditionally and the
	// nanobind binding exposes it either way, so leaving it in there left _pyoomph_core.so with an
	// undefined symbol in every MPI-less build. The body's own guard is what makes it MPI-aware.
	// The elements this rank assembles, exactly as oomph::Problem::parallel_sparse_assemble() computes
	// them. Distributed: all of them, with the halo flag doing the selection. Replicated: the slice
	// oomph-lib handed this rank, which it may re-tune from measured elemental timings -- which is why
	// the range is recorded in the plans and compared before one is reused.
	void Problem::get_assembly_element_range(unsigned long &el_lo, unsigned long &el_hi_plus_one) const
	{
		el_lo = 0;
		el_hi_plus_one = const_cast<Problem *>(this)->mesh_pt()->nelement();
#ifdef OOMPH_HAS_MPI
		if (!Problem_has_been_distributed && Communicator_pt)
		{
			const unsigned r = Communicator_pt->my_rank();
			if (r < First_el_for_assembly.size())
			{
				el_lo = First_el_for_assembly[r];
				el_hi_plus_one = Last_el_plus_one_for_assembly[r];
			}
		}
#endif
	}

#ifdef OOMPH_HAS_MPI

	// Phase 2b. See the comment on DistributedFrozenSparsity for what this replaces and why.
	//
	// Builds the entire distributed plan. Everything up to "exchange the pattern" is local and mirrors
	// build_frozen_sparsity(), only with my_eqns as the row set instead of 0..ndof-1; after that one
	// round of communication tells each rank which (row, column) pairs will arrive from where, which is
	// all it needs to lay out the final CSR and precompute the merge.
	// The row/equation half of a distributed assembly plan, shared by the matrix and residual-only
	// paths. COLLECTIVE, and it returns the same verdict on every rank.
	//
	// Nothing here depends on a matrix, on the symbolic mask or on keep_structural_zeros -- only on the
	// equation numbering and the mesh partition. That is what lets get_residuals() use it even when the
	// pattern machinery does not apply at all.
	bool Problem::build_distributed_residual_plan(const oomph::LinearAlgebraDistribution* const& dist_pt, unsigned n_vector, DistributedResidualPlan &rp)
	{
		rp.clear();
		if (!Communicator_pt || !dist_pt) return false;

		const unsigned nproc = Communicator_pt->nproc();
		const unsigned my_rank = Communicator_pt->my_rank();
		MPI_Comm comm = Communicator_pt->mpi_comm();
		oomph::AssemblyHandler *const assembly_handler_pt = this->assembly_handler_pt();
		const unsigned long n_element = mesh_pt()->nelement();
		unsigned long el_lo, el_hi_plus_one;
		this->get_assembly_element_range(el_lo, el_hi_plus_one);

		// Exactly oomph-lib's own set, so the partition below matches what its routine would do.
		oomph::Vector<unsigned> my_eqns;
		if (el_hi_plus_one > el_lo) this->get_my_eqns(assembly_handler_pt, el_lo, el_hi_plus_one - 1, my_eqns);
		const unsigned my_n_eqn = my_eqns.size();
		rp.my_eqns.assign(my_eqns.begin(), my_eqns.end());
		const unsigned *eq0 = rp.my_eqns.data();

		// ---- send plan ----
		// my_eqns is sorted and the target distribution hands each rank a contiguous global range, so
		// the rows for rank p form a single run. oomph-lib assumes the same; check it rather than trust
		// it, because a non-monotonic distribution would silently send the wrong rows.
		bool ok = true;
		rp.send_row_start.assign(nproc + 1, 0);
		{
			std::vector<int> n_eqn_for_proc(nproc, 0);
			unsigned prev_p = 0;
			for (unsigned i = 0; i < my_n_eqn && ok; i++)
			{
				const unsigned p = dist_pt->rank_of_global_row(rp.my_eqns[i]);
				if (p >= nproc || (i && p < prev_p)) { ok = false; break; }
				prev_p = p;
				n_eqn_for_proc[p]++;
			}
			for (unsigned p = 0; p < nproc; p++) rp.send_row_start[p + 1] = rp.send_row_start[p] + n_eqn_for_proc[p];
		}

		// Everything above can fail per rank; everything below is collective. Agree first, so that a
		// rank which bailed cannot leave the others waiting in MPI_Alltoall.
		{
			int mine = ok ? 1 : 0, all = 0;
			MPI_Allreduce(&mine, &all, 1, MPI_INT, MPI_MIN, comm);
			if (!all) { rp.clear(); return false; }
		}

		// ---- exchange the equation numbers ----
		std::vector<int> send_neq(nproc, 0), recv_neq(nproc, 0);
		for (unsigned p = 0; p < nproc; p++) send_neq[p] = rp.send_row_start[p + 1] - rp.send_row_start[p];
		MPI_Alltoall(send_neq.data(), 1, MPI_INT, recv_neq.data(), 1, MPI_INT, comm);

		std::vector<std::vector<int>> send_buf(nproc), recv_buf(nproc);
		std::vector<MPI_Request> reqs;
		for (unsigned p = 0; p < nproc; p++)
		{
			if (p == my_rank) continue;
			if (send_neq[p])
			{
				send_buf[p].reserve(send_neq[p]);
				for (int i = rp.send_row_start[p]; i < rp.send_row_start[p + 1]; i++) send_buf[p].push_back((int)rp.my_eqns[i]);
				reqs.push_back(MPI_Request());
				MPI_Isend(send_buf[p].data(), send_neq[p], MPI_INT, p, 70, comm, &reqs.back());
			}
			if (recv_neq[p])
			{
				recv_buf[p].resize(recv_neq[p]);
				reqs.push_back(MPI_Request());
				MPI_Irecv(recv_buf[p].data(), recv_neq[p], MPI_INT, p, 70, comm, &reqs.back());
			}
		}
		if (!reqs.empty()) MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

		rp.nrow_local = dist_pt->nrow_local();
		rp.first_row = dist_pt->first_row();
		rp.recv_eq_start.assign(nproc + 1, 0);
		for (unsigned p = 0; p < nproc; p++) rp.recv_eq_start[p + 1] = rp.recv_eq_start[p] + recv_neq[p];
		rp.recv_eq_row.resize(rp.recv_eq_start[nproc]);
		for (unsigned p = 0; p < nproc; p++)
		{
			for (int k = 0; k < recv_neq[p]; k++)
			{
				// Self "arrives" without travelling.
				const int g = (p == my_rank) ? (int)rp.my_eqns[rp.send_row_start[my_rank] + k] : recv_buf[p][k];
				const int row = g - (int)rp.first_row;
				if (row < 0 || row >= (int)rp.nrow_local) { ok = false; break; }
				rp.recv_eq_row[rp.recv_eq_start[p] + k] = row;
			}
		}

		// ---- elemental dof -> row in my_eqns ----
		// Precomputed because it is the whole of what a frozen residual assembly does locally: without
		// it every dof of every element bisects my_eqns again on every assembly.
		rp.res_element_offset.assign(n_element + 1, 0);
		rp.res_slot.reserve(my_n_eqn);
		for (unsigned long e = 0; e < n_element; e++)
		{
			oomph::GeneralisedElement *elem_pt = mesh_pt()->element_pt(e);
			// Outside this rank's range the element is never evaluated, so it needs no slots. Indexed
			// by the GLOBAL element number all the same, so the assembly can address it directly.
			const bool mine = (e >= el_lo && e < el_hi_plus_one);
			const unsigned nvar = (!mine || elem_pt->is_halo()) ? 0 : assembly_handler_pt->ndof(elem_pt);
			for (unsigned i = 0; i < nvar; i++)
			{
				const unsigned g = assembly_handler_pt->eqn_number(elem_pt, i);
				const unsigned *f = std::lower_bound(eq0, eq0 + my_n_eqn, g);
				// -1 for a dof this rank does not own a row for; the scatter skips it. Stored rather
				// than compacted away because res_slot is indexed by the elemental dof number i.
				rp.res_slot.push_back((f == eq0 + my_n_eqn || *f != g) ? -1 : (int)(f - eq0));
			}
			rp.res_element_offset[e + 1] = (int)rp.res_slot.size();
		}

		// The post-exchange check above is per-rank again, so vote once more before anyone commits.
		{
			int mine = ok ? 1 : 0, all = 0;
			MPI_Allreduce(&mine, &all, 1, MPI_INT, MPI_MIN, comm);
			if (!all) { rp.clear(); return false; }
		}

		rp.nproc = nproc;
		rp.my_rank = my_rank;
		rp.nelement = n_element;
		rp.el_lo = el_lo;
		rp.el_hi_plus_one = el_hi_plus_one;
		rp.ndof = this->ndof();
		rp.n_vector = n_vector;
		return true;
	}


	bool Problem::build_distributed_frozen_sparsity(const oomph::LinearAlgebraDistribution* const& dist_pt, unsigned n_matrix, unsigned n_vector, DistributedFrozenSparsity &sp)
	{
		sp.clear();
		if (!Communicator_pt || !dist_pt) return false;
		const unsigned nproc = Communicator_pt->nproc();
		const unsigned my_rank = Communicator_pt->my_rank();
		MPI_Comm comm = Communicator_pt->mpi_comm();
		oomph::AssemblyHandler *const assembly_handler_pt = this->assembly_handler_pt();
		const unsigned long n_element = mesh_pt()->nelement();

		// Both modes oomph-lib's routine serves are frozen here. A REPLICATED problem (mpirun without
		// --distribute, every rank holding the whole mesh) differs only in which elements this rank
		// evaluates: a slice of the element list rather than the non-halo ones. That slice is not a
		// function of the equation numbering -- oomph-lib re-tunes it from measured elemental timings --
		// so a plan built for one slice would silently describe the wrong part of the mesh after a
		// re-tune. Hence it is recorded here and compared before the plan is reused, rather than being a
		// reason to refuse the mode.
		unsigned long el_lo, el_hi_plus_one;
		this->get_assembly_element_range(el_lo, el_hi_plus_one);

		// Which equations this rank contributes to, who owns each of them, and where each incoming one
		// lands. Shared with the residual-only plan so there is one implementation of it; collective,
		// and it returns the same verdict on every rank.
		DistributedResidualPlan rp;
		if (!build_distributed_residual_plan(dist_pt, n_vector, rp)) { sp.clear(); return false; }
		sp.my_eqns = rp.my_eqns;
		sp.send_row_start = rp.send_row_start;
		sp.recv_eq_start = rp.recv_eq_start;
		sp.recv_eq_row = rp.recv_eq_row;
		sp.nrow_local = rp.nrow_local;
		sp.first_row = rp.first_row;
		const unsigned my_n_eqn = sp.my_eqns.size();

		// A global equation -> its row in my_eqns. my_eqns is sorted, so this is a bisection; the
		// global numbering is far too sparse here for a dense lookup table to be worth it.
		const unsigned *eq0 = sp.my_eqns.data();

		// ---- local pattern, per matrix: CSR over my_eqns rows with global column indices ----
		// A rank that cannot describe its own share RECORDS that and carries on to the vote below; it
		// must not return from here, because build_distributed_residual_plan() above has already
		// communicated and everything after the vote does too. That is not hypothetical: a replicated
		// run hands the elements out by range, so a rank can end up with NO elements at all (one ODE
		// element over two ranks) and therefore never ask for a mask, while the rank that holds the
		// element bails on a missing one -- one rank returning to oomph-lib's assembly while the other
		// enters MPI_Alltoall is a hung job, and that is what a periodic-orbit solve under a plain
		// mpirun used to do.
		bool local_ok = true;
		sp.mats.resize(n_matrix);
		for (unsigned m = 0; m < n_matrix && local_ok; m++)
		{
			DistributedFrozenSparsity::Mat &mt = sp.mats[m];
			std::vector<std::vector<int>> cols_of_row(my_n_eqn);
			for (unsigned long e = el_lo; e < el_hi_plus_one; e++)
			{
				oomph::GeneralisedElement *elem_pt = mesh_pt()->element_pt(e);
				if (elem_pt->is_halo()) continue;
				const unsigned nvar = assembly_handler_pt->ndof(elem_pt);
				if (!nvar) continue;
				const char *mask = this->sparsity_mask_for_element(m, elem_pt, nvar);
				if (!mask) { local_ok = false; break; } // Value-dependent: cannot be frozen
				for (unsigned i = 0; i < nvar; i++)
				{
					const unsigned g = assembly_handler_pt->eqn_number(elem_pt, i);
					const unsigned *f = std::lower_bound(eq0, eq0 + my_n_eqn, g);
					if (f == eq0 + my_n_eqn || *f != g) continue;
					std::vector<int> &row_cols = cols_of_row[f - eq0];
					for (unsigned j = 0; j < nvar; j++)
					{
						if (!mask[(size_t)i * nvar + j]) continue;
						row_cols.push_back((int)assembly_handler_pt->eqn_number(elem_pt, j));
					}
				}
			}
			if (!local_ok) break; // nothing below would be consistent, and the vote is about to refuse
			mt.local_row_start.assign(my_n_eqn + 1, 0);
			for (unsigned r = 0; r < my_n_eqn; r++)
			{
				std::vector<int> &c = cols_of_row[r];
				std::sort(c.begin(), c.end());
				c.erase(std::unique(c.begin(), c.end()), c.end());
				mt.local_row_start[r + 1] = mt.local_row_start[r] + (int)c.size();
			}
			mt.local_col.resize(mt.local_row_start[my_n_eqn]);
			for (unsigned r = 0; r < my_n_eqn; r++)
			{
				std::copy(cols_of_row[r].begin(), cols_of_row[r].end(), mt.local_col.begin() + mt.local_row_start[r]);
			}

			// The scatter map, slot-sorted per element for the same cache reason as the serial one.
			mt.element_offset.assign(n_element + 1, 0);
			mt.scatter_source.reserve(mt.local_col.size());
			mt.scatter_slot.reserve(mt.local_col.size());
			std::vector<std::pair<int, int>> pairs;
			for (unsigned long e = 0; e < n_element; e++)
			{
				pairs.clear();
				if (e >= el_lo && e < el_hi_plus_one)
				{
					oomph::GeneralisedElement *elem_pt = mesh_pt()->element_pt(e);
					const unsigned nvar = elem_pt->is_halo() ? 0 : assembly_handler_pt->ndof(elem_pt);
					if (nvar)
					{
						const char *mask = this->sparsity_mask_for_element(m, elem_pt, nvar);
						if (!mask) { local_ok = false; break; }
						for (unsigned i = 0; i < nvar; i++)
						{
							const unsigned g = assembly_handler_pt->eqn_number(elem_pt, i);
							const unsigned *f = std::lower_bound(eq0, eq0 + my_n_eqn, g);
							if (f == eq0 + my_n_eqn || *f != g) continue;
							const int row = (int)(f - eq0);
							const int *row_cols = mt.local_col.data() + mt.local_row_start[row];
							const int row_len = mt.local_row_start[row + 1] - mt.local_row_start[row];
							for (unsigned j = 0; j < nvar; j++)
							{
								if (!mask[(size_t)i * nvar + j]) continue;
								const int col = (int)assembly_handler_pt->eqn_number(elem_pt, j);
								const int *found = std::lower_bound(row_cols, row_cols + row_len, col);
								pairs.push_back(std::make_pair(mt.local_row_start[row] + (int)(found - row_cols),
															   (int)((size_t)i * nvar + j)));
							}
						}
						std::sort(pairs.begin(), pairs.end());
					}
				}
				for (size_t k = 0; k < pairs.size(); k++)
				{
					mt.scatter_slot.push_back(pairs[k].first);
					mt.scatter_source.push_back(pairs[k].second);
				}
				mt.element_offset[e + 1] = (int)mt.scatter_slot.size();
			}
		}

		// ---- per-matrix send plan ----
		// The rows for rank p are one contiguous run of my_eqns (established by the row plan), so each
		// matrix's values for rank p are one contiguous run of its local value array too. Skipped when
		// this rank already knows it cannot describe its share: the arrays it would index are then only
		// partly built.
		for (unsigned m = 0; m < n_matrix && local_ok; m++)
		{
			DistributedFrozenSparsity::Mat &mt = sp.mats[m];
			mt.send_nz_start.assign(nproc + 1, 0);
			for (unsigned p = 0; p < nproc; p++)
			{
				mt.send_nz_start[p + 1] = mt.local_row_start[sp.send_row_start[p + 1]];
			}
		}

		// ---- can the local half be described at all? ----
		// Every bail-out above is per-rank -- a missing symbolic mask, a distribution that is not
		// contiguous by rank -- and everything below is collective. So the answer is agreed here rather
		// than discovered afterwards by a rank left waiting.
		{
			int ok = local_ok ? 1 : 0, all_ok = 0;
			MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, comm);
			if (!all_ok) { sp.clear(); return false; }
		}

		// ---- exchange the pattern ----
		// One message per peer, laid out as
		//     [ per matrix: row lengths ] [ per matrix: column indices ]
		// preceded by an all-to-all of the nonzero counts so the receivers can allocate. The equation
		// numbers are NOT here -- the row plan above already exchanged them, and how many rows arrive
		// from rank p is known from it.
		const unsigned stride = n_matrix; // nnz per matrix
		std::vector<int> send_counts(nproc * stride, 0), recv_counts(nproc * stride, 0);
		std::vector<int> recv_neq(nproc, 0), send_neq(nproc, 0);
		for (unsigned p = 0; p < nproc; p++)
		{
			send_neq[p] = sp.send_row_start[p + 1] - sp.send_row_start[p];
			recv_neq[p] = sp.recv_eq_start[p + 1] - sp.recv_eq_start[p];
			for (unsigned m = 0; m < n_matrix; m++)
			{
				send_counts[p * stride + m] = sp.mats[m].send_nz_start[p + 1] - sp.mats[m].send_nz_start[p];
			}
		}
		MPI_Alltoall(send_counts.data(), stride, MPI_INT, recv_counts.data(), stride, MPI_INT, comm);

		std::vector<std::vector<int>> send_buf(nproc), recv_buf(nproc);
		std::vector<MPI_Request> reqs;
		for (unsigned p = 0; p < nproc; p++)
		{
			if (p == my_rank) continue;
			size_t total = (size_t)send_neq[p] * n_matrix;
			for (unsigned m = 0; m < n_matrix; m++) total += send_counts[p * stride + m];
			if (total)
			{
				send_buf[p].reserve(total);
				for (unsigned m = 0; m < n_matrix; m++)
				{
					const DistributedFrozenSparsity::Mat &mt = sp.mats[m];
					for (int i = sp.send_row_start[p]; i < sp.send_row_start[p + 1]; i++)
					{
						send_buf[p].push_back(mt.local_row_start[i + 1] - mt.local_row_start[i]);
					}
				}
				for (unsigned m = 0; m < n_matrix; m++)
				{
					const DistributedFrozenSparsity::Mat &mt = sp.mats[m];
					send_buf[p].insert(send_buf[p].end(), mt.local_col.begin() + mt.send_nz_start[p],
									   mt.local_col.begin() + mt.send_nz_start[p + 1]);
				}
				reqs.push_back(MPI_Request());
				MPI_Isend(send_buf[p].data(), (int)send_buf[p].size(), MPI_INT, p, 71, comm, &reqs.back());
			}
			size_t rtotal = (size_t)recv_neq[p] * n_matrix;
			for (unsigned m = 0; m < n_matrix; m++) rtotal += recv_counts[p * stride + m];
			if (rtotal)
			{
				recv_buf[p].resize(rtotal);
				reqs.push_back(MPI_Request());
				MPI_Irecv(recv_buf[p].data(), (int)rtotal, MPI_INT, p, 71, comm, &reqs.back());
			}
		}
		// The self entries of recv_counts came from the Alltoall, which includes the diagonal, so what
		// this rank "receives" from itself is already recorded; it simply never travels.
		if (!reqs.empty()) MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

		// ---- lay out the final CSR and the merge permutation ----
		// Per rank, a view of the row lengths and columns it contributed. For self these point straight
		// into our own arrays instead of a copy.
		std::vector<const int*> in_len(nproc, 0), in_col(nproc, 0);
		std::vector<std::vector<int>> self_len(n_matrix);
		for (unsigned m = 0; m < n_matrix; m++)
		{
			const DistributedFrozenSparsity::Mat &mt = sp.mats[m];
			for (int i = sp.send_row_start[my_rank]; i < sp.send_row_start[my_rank + 1]; i++)
			{
				self_len[m].push_back(mt.local_row_start[i + 1] - mt.local_row_start[i]);
			}
		}

		for (unsigned m = 0; m < n_matrix; m++)
		{
			DistributedFrozenSparsity::Mat &mt = sp.mats[m];
			// Where each rank's row lengths and columns sit inside its message.
			for (unsigned p = 0; p < nproc; p++)
			{
				if (!recv_neq[p]) { in_len[p] = 0; in_col[p] = 0; continue; }
				if (p == my_rank)
				{
					in_len[p] = self_len[m].data();
					in_col[p] = mt.local_col.data() + mt.send_nz_start[my_rank];
					continue;
				}
				size_t off = (size_t)recv_neq[p] * m; // past earlier matrices' row lengths
				in_len[p] = recv_buf[p].data() + off;
				off = (size_t)recv_neq[p] * n_matrix;
				for (unsigned q = 0; q < m; q++) off += recv_counts[p * stride + q];
				in_col[p] = recv_buf[p].data() + off;
			}

			// Gather the columns of every local row from every contributing rank, then sort and unique
			// them. Unlike oomph-lib's merge this is O(nnz log nnz) once rather than O(row_nnz^2) on
			// every assembly, and it leaves the columns ascending within a row.
			std::vector<std::vector<int>> cols_of_row(sp.nrow_local);
			std::vector<int> nz_cursor(nproc, 0);
			for (unsigned p = 0; p < nproc; p++)
			{
				int c = 0;
				for (int k = 0; k < recv_neq[p]; k++)
				{
					const int row = sp.recv_eq_row[sp.recv_eq_start[p] + k];
					const int len = in_len[p][k];
					cols_of_row[row].insert(cols_of_row[row].end(), in_col[p] + c, in_col[p] + c + len);
					c += len;
				}
				nz_cursor[p] = c;
			}
			mt.final_row_start.assign(sp.nrow_local + 1, 0);
			for (unsigned r = 0; r < sp.nrow_local; r++)
			{
				std::vector<int> &c = cols_of_row[r];
				std::sort(c.begin(), c.end());
				c.erase(std::unique(c.begin(), c.end()), c.end());
				mt.final_row_start[r + 1] = mt.final_row_start[r] + (int)c.size();
			}
			mt.final_col.resize(mt.final_row_start[sp.nrow_local]);
			for (unsigned r = 0; r < sp.nrow_local; r++)
			{
				std::copy(cols_of_row[r].begin(), cols_of_row[r].end(), mt.final_col.begin() + mt.final_row_start[r]);
			}

			// The permutation: incoming nonzero -> its slot in the final value array.
			mt.recv_nz_start.assign(nproc + 1, 0);
			for (unsigned p = 0; p < nproc; p++) mt.recv_nz_start[p + 1] = mt.recv_nz_start[p] + nz_cursor[p];
			mt.merge_perm.resize(mt.recv_nz_start[nproc]);
			for (unsigned p = 0; p < nproc; p++)
			{
				int c = 0;
				for (int k = 0; k < recv_neq[p]; k++)
				{
					const int row = sp.recv_eq_row[sp.recv_eq_start[p] + k];
					const int len = in_len[p][k];
					const int *row_cols = &mt.final_col[mt.final_row_start[row]];
					const int row_len = mt.final_row_start[row + 1] - mt.final_row_start[row];
					for (int l = 0; l < len; l++)
					{
						const int *found = std::lower_bound(row_cols, row_cols + row_len, in_col[p][c + l]);
						// Cannot miss: the union above was taken over exactly these columns.
						mt.merge_perm[mt.recv_nz_start[p] + c + l] = mt.final_row_start[row] + (int)(found - row_cols);
					}
					c += len;
				}
			}
		}

		// Who owns which rows, taken from the distribution while it is still in hand. Local knowledge:
		// LinearAlgebraDistribution carries every rank's first_row, so this costs no communication.
		sp.row_first.resize(nproc + 1);
		for (unsigned p = 0; p < nproc; p++) sp.row_first[p] = dist_pt->first_row(p);
		sp.row_first[nproc] = dist_pt->nrow();

		sp.nproc = nproc;
		sp.my_rank = my_rank;
		sp.ndof = this->ndof();
		sp.nelement = n_element;
		sp.el_lo = el_lo;
		sp.el_hi_plus_one = el_hi_plus_one;
		sp.n_matrix = n_matrix;
		sp.n_vector = n_vector;
		return true;
	}


	// The per-assembly distributed fast path: evaluate elements into a frozen local CSR, ship the
	// values (nothing else) to their owners, and scatter-add them into the final one.
	bool Problem::assemble_distributed_with_frozen_sparsity(const oomph::LinearAlgebraDistribution* const& dist_pt, oomph::Vector<int*>& column_or_row_index, oomph::Vector<int*>& row_or_column_start, oomph::Vector<double*>& value, oomph::Vector<unsigned>& nnz, oomph::Vector<double*>& residuals)
	{
		if (!use_frozen_distributed_sparsity || !use_frozen_sparsity || !keep_structural_zeros) return false;
		if (n_unaugmented_dofs != 0) return false;
		if (!Communicator_pt || !dist_pt) return false;
		const unsigned n_matrix = column_or_row_index.size();
		const unsigned n_vector = residuals.size();
		if (!n_matrix) return false; // Residual-only assembly: nothing to gain, and one fewer path to get wrong
		// A ParallelResidualsHandler assembles residuals only; it has no business here.
		if (dynamic_cast<oomph::ParallelResidualsHandler*>(this->assembly_handler_pt())) return false;

		const unsigned long gen = this->get_jacobian_structure_id();
		if (!gen) return false;

		unsigned long cur_el_lo, cur_el_hi_plus_one;
		this->get_assembly_element_range(cur_el_lo, cur_el_hi_plus_one);

		DistributedFrozenSparsity &sp = distributed_frozen_sparsity;
		// Everything from here on is collective, so every branch has to be taken by every rank or the
		// ones that took it hang waiting for the ones that did not. Two things could diverge, and both
		// are put to a vote: whether the plan needs rebuilding (nrow_local, first_row and nelement are
		// per-rank quantities), and whether the rebuild succeeded.
		{
			// el_lo/el_hi_plus_one are in here because on a REPLICATED run they are not a function of
			// the equation numbering: oomph-lib re-tunes the slice from measured elemental timings
			// whenever its own routine runs (recompute_load_balanced_assembly). Reusing a plan across
			// such a re-tune would assemble one slice of the mesh through the scatter map of another --
			// a wrong Jacobian AND a wrong residual, so Newton would converge to a wrong state rather
			// than fail. Distributed they are constant and the comparison costs nothing.
			int need = (sp.generation != gen || sp.n_matrix != n_matrix || sp.n_vector != n_vector ||
						sp.nrow_local != dist_pt->nrow_local() || sp.first_row != dist_pt->first_row() ||
						sp.nelement != mesh_pt()->nelement() ||
						sp.el_lo != cur_el_lo || sp.el_hi_plus_one != cur_el_hi_plus_one) ? 1 : 0;
			int any_need = 0;
			MPI_Allreduce(&need, &any_need, 1, MPI_INT, MPI_MAX, Communicator_pt->mpi_comm());
			if (any_need)
			{
				DistributedFrozenSparsity fresh;
				// Collective, and it agrees internally on everything it can decide before communicating.
				// What is left is the handful of checks it can only make once the pattern has arrived.
				int ok = build_distributed_frozen_sparsity(dist_pt, n_matrix, n_vector, fresh) ? 1 : 0;
				int all_ok = 0;
				MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, Communicator_pt->mpi_comm());
				if (!all_ok) { sp.clear(); return false; }
				fresh.generation = gen;
				sp = fresh;
				distributed_frozen_rebuilds++;
			}
		}

		const unsigned nproc = sp.nproc, my_rank = sp.my_rank;
		MPI_Comm comm = Communicator_pt->mpi_comm();
		const unsigned my_n_eqn = sp.my_eqns.size();

		// ---- local stage ----
		std::vector<std::vector<double>> local_values(n_matrix);
		for (unsigned m = 0; m < n_matrix; m++) local_values[m].assign(sp.mats[m].local_col.size(), 0.0);
		std::vector<std::vector<double>> local_res(n_vector);
		for (unsigned v = 0; v < n_vector; v++) local_res[v].assign(my_n_eqn, 0.0);

		oomph::AssemblyHandler *const assembly_handler_pt = this->assembly_handler_pt();
		// From the plan, not recomputed: the two agree by the freshness vote above, and taking them from
		// the plan is what guarantees the scatter map and the element loop describe the same slice.
		const unsigned long el_lo = sp.el_lo, el_hi_plus_one = sp.el_hi_plus_one;
		// my_eqns is sorted, so a row index for a global equation is a bisection. Residuals need it;
		// the matrix entries do not, they go through the scatter map.
		const unsigned *eq0 = sp.my_eqns.data();

		std::string violation; // empty = this rank has no objection, see the vote after the loop
		// Threaded local stage (dev_docs/openmp_assembly.md): hybrid MPI+OpenMP, i.e. threads INSIDE
		// each rank. Only this loop is threaded; the exchange below stays on the calling thread, so
		// MPI_THREAD_FUNNELED is all that is required of the MPI build.
		if (!this->try_parallel_distributed_assembly(sp, n_matrix, n_vector, local_values, local_res, violation))
		{
			oomph::Vector<oomph::Vector<double>> el_residuals(n_vector);
			oomph::Vector<oomph::DenseMatrix<double>> el_jacobian(n_matrix);
			for (unsigned long e = el_lo; e < el_hi_plus_one; e++)
			{
				oomph::GeneralisedElement *elem_pt = mesh_pt()->element_pt(e);
				if (elem_pt->is_halo()) continue;
				const unsigned nvar = assembly_handler_pt->ndof(elem_pt);
				if (!nvar) continue;
				for (unsigned v = 0; v < n_vector; v++) el_residuals[v].resize(nvar);
				for (unsigned m = 0; m < n_matrix; m++) el_jacobian[m].resize(nvar);
				assembly_handler_pt->get_all_vectors_and_matrices(elem_pt, el_residuals, el_jacobian);

				for (unsigned i = 0; i < nvar; i++)
				{
					const unsigned g = assembly_handler_pt->eqn_number(elem_pt, i);
					const unsigned *found = std::lower_bound(eq0, eq0 + my_n_eqn, g);
					if (found == eq0 + my_n_eqn || *found != g) continue;
					const size_t row = found - eq0;
					for (unsigned v = 0; v < n_vector; v++) local_res[v][row] += el_residuals[v][i];
				}
				for (unsigned m = 0; m < n_matrix; m++)
				{
					const DistributedFrozenSparsity::Mat &mt = sp.mats[m];
					const int lo = mt.element_offset[e], hi = mt.element_offset[e + 1];
					const int *slot = mt.scatter_slot.data();
					const int *src = mt.scatter_source.data();
					double *val = local_values[m].data();
					const double *flat = &el_jacobian[m](0, 0);
					for (int k = lo; k < hi; k++) val[slot[k]] += flat[src[k]];
					if (verify_frozen_sparsity && violation.empty())
					{
						// Same guard as the serial path: the compact scatter cannot notice an entry that fell
						// outside the pattern, so count instead of trusting.
						unsigned all = 0, taken = 0;
						const size_t nblock = (size_t)nvar * nvar;
						for (size_t q = 0; q < nblock; q++) all += (flat[q] != 0.0);
						for (int k = lo; k < hi; k++) taken += (flat[src[k]] != 0.0);
						if (all > taken)
						{
							// RECORDED, not thrown: the value exchange below is collective, so a rank throwing
							// here on its own would leave every other rank waiting for a message it will never
							// send - a hung job instead of a failed one. The offending element is this rank's
							// own, so only it can see the violation.
							violation = "A Jacobian entry appeared outside the symbolic sparsity pattern during "
										"distributed assembly (matrix " + std::to_string(m) + ", element " +
										std::to_string(e) + "). Workaround: set problem.use_frozen_sparsity=False.";
						}
					}
				}
			}
		}

		// Every rank reaches this, whatever it found, so it is safe to throw from here: the exchange
		// below is not entered by anybody.
		if (verify_frozen_sparsity) collective_throw(violation);

		// ---- exchange values ----
		// One message per peer: [ residuals, vector by vector ] [ values, matrix by matrix ]. The
		// sources are separate arrays so this is packed, but it is n_vector+n_matrix memcpys of
		// contiguous runs -- no gathering, no index computation, because the send slices are contiguous.
		std::vector<std::vector<double>> dsend(nproc), drecv(nproc);
		std::vector<MPI_Request> reqs;
		reqs.reserve(2 * nproc);
		for (unsigned p = 0; p < nproc; p++)
		{
			if (p == my_rank) continue;
			const int neq = sp.send_row_start[p + 1] - sp.send_row_start[p];
			size_t total = (size_t)neq * n_vector;
			for (unsigned m = 0; m < n_matrix; m++) total += sp.mats[m].send_nz_start[p + 1] - sp.mats[m].send_nz_start[p];
			if (total)
			{
				dsend[p].reserve(total);
				for (unsigned v = 0; v < n_vector; v++)
				{
					dsend[p].insert(dsend[p].end(), local_res[v].begin() + sp.send_row_start[p],
									local_res[v].begin() + sp.send_row_start[p + 1]);
				}
				for (unsigned m = 0; m < n_matrix; m++)
				{
					dsend[p].insert(dsend[p].end(), local_values[m].begin() + sp.mats[m].send_nz_start[p],
									local_values[m].begin() + sp.mats[m].send_nz_start[p + 1]);
				}
				reqs.push_back(MPI_Request());
				MPI_Isend(dsend[p].data(), (int)dsend[p].size(), MPI_DOUBLE, p, 72, comm, &reqs.back());
			}
			const int rneq = sp.recv_eq_start[p + 1] - sp.recv_eq_start[p];
			size_t rtotal = (size_t)rneq * n_vector;
			for (unsigned m = 0; m < n_matrix; m++) rtotal += sp.mats[m].recv_nz_start[p + 1] - sp.mats[m].recv_nz_start[p];
			if (rtotal)
			{
				drecv[p].resize(rtotal);
				reqs.push_back(MPI_Request());
				MPI_Irecv(drecv[p].data(), (int)rtotal, MPI_DOUBLE, p, 72, comm, &reqs.back());
			}
		}

		// ---- hand out the answer and merge ----
		// Freshly allocated every time: CRDoubleMatrix::build_without_copy() takes ownership.
		for (unsigned m = 0; m < n_matrix; m++)
		{
			const DistributedFrozenSparsity::Mat &mt = sp.mats[m];
			nnz[m] = (unsigned)mt.final_col.size();
			row_or_column_start[m] = new int[sp.nrow_local + 1];
			std::copy(mt.final_row_start.begin(), mt.final_row_start.end(), row_or_column_start[m]);
			column_or_row_index[m] = new int[nnz[m] ? nnz[m] : 1];
			std::copy(mt.final_col.begin(), mt.final_col.end(), column_or_row_index[m]);
			value[m] = new double[nnz[m] ? nnz[m] : 1];
			std::fill(value[m], value[m] + nnz[m], 0.0);
		}
		for (unsigned v = 0; v < n_vector; v++)
		{
			residuals[v] = new double[sp.nrow_local ? sp.nrow_local : 1];
			std::fill(residuals[v], residuals[v] + sp.nrow_local, 0.0);
		}

		// Our own contribution needs no message; do it while the others are in flight.
		{
			const unsigned p = my_rank;
			for (unsigned v = 0; v < n_vector; v++)
			{
				for (int k = sp.recv_eq_start[p]; k < sp.recv_eq_start[p + 1]; k++)
				{
					residuals[v][sp.recv_eq_row[k]] += local_res[v][sp.send_row_start[p] + (k - sp.recv_eq_start[p])];
				}
			}
			for (unsigned m = 0; m < n_matrix; m++)
			{
				const DistributedFrozenSparsity::Mat &mt = sp.mats[m];
				const int lo = mt.recv_nz_start[p], hi = mt.recv_nz_start[p + 1];
				const double *src = local_values[m].data() + mt.send_nz_start[p];
				double *dst = value[m];
				for (int k = lo; k < hi; k++) dst[mt.merge_perm[k]] += src[k - lo];
			}
		}

		if (!reqs.empty()) MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

		for (unsigned p = 0; p < nproc; p++)
		{
			if (p == my_rank || drecv[p].empty()) continue;
			const double *buf = drecv[p].data();
			for (unsigned v = 0; v < n_vector; v++)
			{
				for (int k = sp.recv_eq_start[p]; k < sp.recv_eq_start[p + 1]; k++)
				{
					residuals[v][sp.recv_eq_row[k]] += *buf++;
				}
			}
			for (unsigned m = 0; m < n_matrix; m++)
			{
				const DistributedFrozenSparsity::Mat &mt = sp.mats[m];
				const int lo = mt.recv_nz_start[p], hi = mt.recv_nz_start[p + 1];
				double *dst = value[m];
				for (int k = lo; k < hi; k++) dst[mt.merge_perm[k]] += *buf++;
			}
		}
		return true;
	}


	// The residual-only distributed assembly.
	//
	// oomph-lib routes get_residuals() under MPI through the whole of parallel_sparse_assemble() with
	// zero matrices: it recomputes my_eqns (a sort and unique over every elemental dof), sets up the
	// array-of-arrays machinery, exchanges equation numbers alongside the values, and merges by
	// bisecting each sender's equation list per row. All of that to sum a vector, once per Newton step.
	// Measured on a 2D cavity it was 26 % of a residual assembly at 2 ranks and 45 % at 4 -- and roughly
	// constant in absolute terms as ranks grow, so it worsens with scale.
	//
	// Frozen it is: zero, scatter through a precomputed slot map, send contiguous slices, scatter-add.
	bool Problem::assemble_distributed_residuals_only(const oomph::LinearAlgebraDistribution* const& dist_pt, oomph::Vector<double*>& residuals)
	{
		if (!use_frozen_distributed_sparsity) return false;
		if (!Communicator_pt || !dist_pt) return false;
		if (n_unaugmented_dofs != 0) return false;
		const unsigned n_vector = residuals.size();
		if (!n_vector) return false;
		unsigned long cur_el_lo, cur_el_hi_plus_one;
		this->get_assembly_element_range(cur_el_lo, cur_el_hi_plus_one);

		// Deliberately NOT get_jacobian_structure_id(): that is 0 whenever the pattern is
		// value-dependent (keep_structural_zeros off), and this plan does not involve a pattern at all.
		// What it needs is only that the equations have not been renumbered, which
		// invalidate_jacobian_structure() -- called from the collective assign_eqn_numbers() -- signals
		// by clearing the plan, and which the shape checks below confirm.
		DistributedResidualPlan &rp = distributed_residual_plan;
		{
			int need = (rp.generation == 0 || rp.n_vector != n_vector || rp.ndof != this->ndof() ||
						rp.nrow_local != dist_pt->nrow_local() || rp.first_row != dist_pt->first_row() ||
						rp.nelement != mesh_pt()->nelement() ||
						rp.el_lo != cur_el_lo || rp.el_hi_plus_one != cur_el_hi_plus_one) ? 1 : 0;
			int any_need = 0;
			MPI_Allreduce(&need, &any_need, 1, MPI_INT, MPI_MAX, Communicator_pt->mpi_comm());
			if (any_need)
			{
				DistributedResidualPlan fresh;
				if (!build_distributed_residual_plan(dist_pt, n_vector, fresh)) { rp.clear(); return false; }
				fresh.generation = 1; // "built"; validity is the shape checks above, not a pattern id
				rp = fresh;
				distributed_residual_rebuilds++;
			}
		}

		const unsigned nproc = rp.nproc, my_rank = rp.my_rank;
		MPI_Comm comm = Communicator_pt->mpi_comm();
		const unsigned my_n_eqn = rp.my_eqns.size();
		oomph::AssemblyHandler *const assembly_handler_pt = this->assembly_handler_pt();

		// ---- local stage ----
		std::vector<std::vector<double>> local_res(n_vector);
		for (unsigned v = 0; v < n_vector; v++) local_res[v].assign(my_n_eqn, 0.0);
		// Threaded local stage: hybrid MPI+OpenMP (dev_docs/openmp_assembly.md). The engine derives the
		// same rows from rp.my_eqns that rp.res_slot holds, so the two agree by construction; the plan
		// is built once and only when a pattern id exists to invalidate it against, which is why this
		// declines - and falls through to the loop below - with keep_structural_zeros off.
		bool threaded = false;
		if (this->parallel_assembly_possible())
		{
			FrozenAssemblyRequest req;
			req.el_lo = rp.el_lo;
			req.el_hi_plus_one = rp.el_hi_plus_one;
			req.n_matrix = 0;
			req.n_vector = n_vector;
			req.n_map = 0;
			for (unsigned v = 0; v < n_vector; v++) req.residual.push_back(local_res[v].data());
			req.res_row_map = rp.my_eqns.data();
			req.res_row_map_n = (unsigned)rp.my_eqns.size();
			threaded = this->parallel_assemble_frozen(req);
		}
		if (!threaded)
		{
			oomph::Vector<oomph::Vector<double>> el_residuals(n_vector);
			oomph::Vector<oomph::DenseMatrix<double>> no_matrices;
			for (unsigned long e = rp.el_lo; e < rp.el_hi_plus_one; e++)
			{
				oomph::GeneralisedElement *elem_pt = mesh_pt()->element_pt(e);
				if (elem_pt->is_halo()) continue;
				const unsigned nvar = assembly_handler_pt->ndof(elem_pt);
				if (!nvar) continue;
				for (unsigned v = 0; v < n_vector; v++) el_residuals[v].resize(nvar);
				assembly_handler_pt->get_all_vectors_and_matrices(elem_pt, el_residuals, no_matrices);
				const int *slot = rp.res_slot.data() + rp.res_element_offset[e];
				for (unsigned i = 0; i < nvar; i++)
				{
					if (slot[i] < 0) continue;
					for (unsigned v = 0; v < n_vector; v++) local_res[v][slot[i]] += el_residuals[v][i];
				}
			}
		}

		// ---- exchange ----
		std::vector<std::vector<double>> dsend(nproc), drecv(nproc);
		std::vector<MPI_Request> reqs;
		reqs.reserve(2 * nproc);
		for (unsigned p = 0; p < nproc; p++)
		{
			if (p == my_rank) continue;
			const int neq = rp.send_row_start[p + 1] - rp.send_row_start[p];
			if (neq)
			{
				dsend[p].reserve((size_t)neq * n_vector);
				for (unsigned v = 0; v < n_vector; v++)
				{
					dsend[p].insert(dsend[p].end(), local_res[v].begin() + rp.send_row_start[p],
									local_res[v].begin() + rp.send_row_start[p + 1]);
				}
				reqs.push_back(MPI_Request());
				MPI_Isend(dsend[p].data(), (int)dsend[p].size(), MPI_DOUBLE, p, 73, comm, &reqs.back());
			}
			const int rneq = rp.recv_eq_start[p + 1] - rp.recv_eq_start[p];
			if (rneq)
			{
				drecv[p].resize((size_t)rneq * n_vector);
				reqs.push_back(MPI_Request());
				MPI_Irecv(drecv[p].data(), (int)drecv[p].size(), MPI_DOUBLE, p, 73, comm, &reqs.back());
			}
		}

		for (unsigned v = 0; v < n_vector; v++)
		{
			residuals[v] = new double[rp.nrow_local ? rp.nrow_local : 1];
			std::fill(residuals[v], residuals[v] + rp.nrow_local, 0.0);
		}
		// Our own contribution needs no message; do it while the others are in flight.
		for (unsigned v = 0; v < n_vector; v++)
		{
			for (int k = rp.recv_eq_start[my_rank]; k < rp.recv_eq_start[my_rank + 1]; k++)
			{
				residuals[v][rp.recv_eq_row[k]] += local_res[v][rp.send_row_start[my_rank] + (k - rp.recv_eq_start[my_rank])];
			}
		}

		if (!reqs.empty()) MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

		for (unsigned p = 0; p < nproc; p++)
		{
			if (p == my_rank || drecv[p].empty()) continue;
			const double *buf = drecv[p].data();
			for (unsigned v = 0; v < n_vector; v++)
			{
				for (int k = rp.recv_eq_start[p]; k < rp.recv_eq_start[p + 1]; k++) residuals[v][rp.recv_eq_row[k]] += *buf++;
			}
		}
		return true;
	}


	// Phase 2b entry point. Try the frozen distributed assembly; otherwise oomph-lib's.
	void Problem::parallel_sparse_assemble(const oomph::LinearAlgebraDistribution* const& dist_pt, oomph::Vector<int*>& column_or_row_index, oomph::Vector<int*>& row_or_column_start, oomph::Vector<double*>& value, oomph::Vector<unsigned>& nnz, oomph::Vector<double*>& residuals)
	{
		// Residual-only (get_residuals under MPI installs a ParallelResidualsHandler and asks for no
		// matrices) goes to its own, much smaller plan. Keeping the two plans separate matters: the
		// handler swap changes the reported pattern id, so sharing one slot would rebuild on every step.
		if (column_or_row_index.empty())
		{
			if (assemble_distributed_residuals_only(dist_pt, residuals)) return;
		}
		else if (assemble_distributed_with_frozen_sparsity(dist_pt, column_or_row_index, row_or_column_start, value, nnz, residuals)) return;
		oomph::Problem::parallel_sparse_assemble(dist_pt, column_or_row_index, row_or_column_start, value, nnz, residuals);
	}

#endif


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

		// Fast path. Every matrix this routine builds is a derivative of the same residual with respect to
		// the dofs -- the Jacobian, its parameter derivative, a Hessian-vector product, and the mass
		// matrix equivalents -- so ONE frozen pattern can serve all of them, with the scatter writing the
		// same slots into several value arrays. That pays off here more than anywhere else: this routine
		// accumulates into a std::map per row, the slowest container in the codebase, and it is what a
		// multi-quantity bifurcation assembly goes through on every Newton step.
		//
		// The pattern is the SYMMETRISED union of the Jacobian and mass-matrix masks, not the Jacobian
		// mask alone. The original version used the latter and was wrong: the routine is also asked for
		// TRANSPOSED Hessian-vector products (bifurcation tracking on a left eigenvector), whose pattern
		// is J^T. On the Lorenz Hopf case that put a nonzero at (0,2) where the Jacobian has no entry at
		// all, and the per-element verification below refused the assembly -- correctly, but only because
		// that check exists. See symmetrised_union_mask().
		if (use_frozen_sparsity && keep_structural_zeros && prune_structural_zeros_by_field_coupling && compressed_row_flag)
		{
			bool parallel = false;
#ifdef OOMPH_HAS_MPI
			if (Problem_has_been_distributed || (Communicator_pt && Communicator_pt->nproc() > 1)) parallel = true;
#endif
			const unsigned long gen = this->get_jacobian_structure_id();
			std::vector<int> pinned;
			const unsigned mask_sel = multiassembly_wants_transposed ? MASK_UNION_SYMMETRIC : 0u;
			const int slot = (parallel || !gen) ? -1
											   : this->acquire_frozen_sparsity(mask_sel, gen, ndof, pinned, mask_sel);
			if (slot >= 0)
			{
				const FrozenSparsity &sp = frozen_sparsity_cache[slot];
				for (unsigned m = 0; m < n_matrix; m++)
				{
					nnz[m] = sp.nnz();
					row_or_column_start[m] = new int[ndof + 1];
					std::copy(sp.row_start.begin(), sp.row_start.end(), row_or_column_start[m]);
					column_or_row_index[m] = new int[nnz[m] ? nnz[m] : 1];
					std::copy(sp.column_index.begin(), sp.column_index.end(), column_or_row_index[m]);
					value[m] = new double[nnz[m] ? nnz[m] : 1];
					std::fill(value[m], value[m] + nnz[m], 0.0);
				}
				for (unsigned v = 0; v < n_vector; v++)
				{
					residuals[v] = new double[ndof ? ndof : 1];
					std::fill(residuals[v], residuals[v] + ndof, 0.0);
				}
				// Threaded element loop (dev_docs/openmp_assembly.md). This is the assembly that a
				// bifurcation-tracking or multi-assembly step goes through, i.e. the one that gains the
				// most from it. One slot serves every matrix here, so the gather index is built once.
				if (this->try_parallel_frozen_assembly(std::vector<int>(n_matrix, slot), value, residuals,
													   el_lo, el_hi + 1))
					return;
				oomph::Vector<oomph::Vector<double>> el_res(n_vector);
				oomph::Vector<oomph::DenseMatrix<double>> el_jac(n_matrix);
				for (unsigned long e = el_lo; e <= el_hi; e++)
				{
					oomph::GeneralisedElement *elem_pt = mesh_pt()->element_pt(e);
#ifdef OOMPH_HAS_MPI
					if (elem_pt->is_halo()) continue;
#endif
					const unsigned nvar = assembly_handler_pt->ndof(elem_pt);
					if (!nvar) continue;
					for (unsigned v = 0; v < n_vector; v++) el_res[v].resize(nvar);
					for (unsigned m = 0; m < n_matrix; m++) el_jac[m].resize(nvar);
					assembly_handler_pt->get_all_vectors_and_matrices(elem_pt, el_res, el_jac);
					for (unsigned i = 0; i < nvar; i++)
					{
						const unsigned eqn = assembly_handler_pt->eqn_number(elem_pt, i);
						for (unsigned v = 0; v < n_vector; v++) residuals[v][eqn] += el_res[v][i];
					}
					const int lo = sp.element_offset[e], hi = sp.element_offset[e + 1];
					const int *slotp = &sp.scatter_slot[0];
					const int *src = &sp.scatter_source[0];
					for (unsigned m = 0; m < n_matrix; m++)
					{
						double *val = value[m];
						const double *flat = &el_jac[m](0, 0);
						for (int k = lo; k < hi; k++) val[slotp[k]] += flat[src[k]];
						if (verify_frozen_sparsity)
						{
							unsigned all = 0, taken = 0;
							const size_t nblock = (size_t)nvar * nvar;
							for (size_t q = 0; q < nblock; q++) all += (flat[q] != 0.0);
							for (int k = lo; k < hi; k++) taken += (flat[src[k]] != 0.0);
							if (all > taken)
							{
								// Name the offending positions. "matrix 4, element 0" alone leaves the reader
								// no way to tell an over-tight mask from a genuinely wider matrix, and the two
								// need opposite fixes.
								std::string where;
								unsigned shown = 0;
								std::vector<bool> covered(nblock, false);
								for (int k = lo; k < hi; k++) covered[src[k]] = true;
								for (size_t q = 0; q < nblock && shown < 6; q++)
								{
									if (flat[q] == 0.0 || covered[q]) continue;
									const unsigned qi = (unsigned)(q / nvar), qj = (unsigned)(q % nvar);
									where += (shown ? ", " : "") + std::string("(") + std::to_string(qi) + "," +
											 std::to_string(qj) + ")=" + std::to_string(flat[q]);
									shown++;
								}
								throw_runtime_error("A multi-assembly entry appeared outside the symbolic sparsity pattern (matrix " +
													std::to_string(m) + " of " + std::to_string(n_matrix) + ", element " +
													std::to_string(e) + ", nvar " + std::to_string(nvar) + "). Positions missing "
													"from the pattern: " + where + ". Refusing rather than truncating the matrix. "
													"Workaround: problem.use_frozen_sparsity=False.");
							}
						}
					}
				}
				return;
			}
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
			oomph::Vector<const char*> sparsity_masks(n_matrix, 0); // Refilled per element, allocated once
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
					// Fetch the structural sparsity masks once per element, not once per entry.
					for (unsigned m = 0; m < n_matrix; m++) sparsity_masks[m] = sparsity_mask_for_element(m, elem_pt, nvar);
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
								// The mask may only ADD entries, never remove them; see sparsity_mask_for_element().
								if ((sparsity_masks[m] && sparsity_masks[m][i * nvar + j]) ||
									std::fabs(value) > numerical_zero_for_sparse_assembly(m))
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

		// Only a transposed product needs the symmetrised pattern (see symmetrised_union_mask). Deciding
		// it from the request list keeps fold, pitchfork and right-eigenvector Hopf tracking on the exact
		// Jacobian pattern, and pays for the transpose only where it is mathematically required.
		multiassembly_wants_transposed = false;
		for (unsigned i = 0; i < what.size(); i++)
		{
			if (what[i].size() >= 11 && what[i].compare(what[i].size() - 11, 11, "_transposed") == 0)
			{
				multiassembly_wants_transposed = true;
				break;
			}
		}
		oomph::AssemblyHandler * old_handler=this->assembly_handler_pt();
		pyoomph::CustomMultiAssembleHandler * new_handler=new pyoomph::CustomMultiAssembleHandler(this,what,contributions,params,hessian_vectors,hessian_vector_indices,return_indices);
	    ndof = this->get_n_unaugmented_dofs();
		this->assembly_handler_pt()=new_handler;
		// Restore the handler, delete it, and put the transposed flag back to its safe default even if
		// the assembly throws -- which it does whenever the sparsity verification refuses an element.
		// Without this a caught error left the Problem with a deleted-or-custom assembly handler still
		// installed, and left the flag saying "no transposed products" for every LATER caller of the
		// base-problem assembly, including ones that have nothing to do with a multi-assembly.
		struct RestoreOnExit
		{
			pyoomph::Problem *p; oomph::AssemblyHandler *old; oomph::AssemblyHandler *own;
			~RestoreOnExit() { p->assembly_handler_pt() = old; delete own; p->multiassembly_wants_transposed = true; }
		} restore_on_exit{this, old_handler, new_handler};
		unsigned nvector=new_handler->n_vector();
		unsigned nmatrix=new_handler->n_matrix();
		column_or_row_index.resize(nmatrix);
		row_or_column_start.resize(nmatrix);
		value.resize(nmatrix);
		nnz.resize(nmatrix);
		residuals.resize(nvector);
		this->sparse_assemble_row_or_column_compressed_base_problem(column_or_row_index,row_or_column_start,value,nnz,residuals,true);
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
	// Strips the "@opposite" marker a per-element contribution class carries when it refers to the far
	// side of an interior facet, so that both sides map onto the one global field they are copies of.
	// Must agree with OPPOSITE_CONTRIBUTION_CLASS_SUFFIX in codegen.cpp and with the adoption in
	// InterfaceElementBase::fill_local_dof_contribution_indices. Records what it stripped, for the
	// report at the end of get_jacobian_information_string().
	static std::string strip_side_split_suffix(const std::string &name, std::set<std::string> &split_classes)
	{
		static const std::string suffix = "@opposite";
		if (name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
		{
			std::string base = name.substr(0, name.size() - suffix.size());
			split_classes.insert(base);
			return base;
		}
		return name;
	}

	// of loaded codes changes (loading new equations) and before equation numbering can rely on the
	// pinning information being up to date.
	//
	// The per-element contribution classes may distinguish the two sides of an interior facet (the
	// "@opposite" suffix, see codegen.cpp); this level deliberately does not. The question answered
	// here - "does residual R couple field A to field B anywhere in the problem" - is side-agnostic by
	// construction, and there is no global unknown a far-side class could stand for. Exposing one
	// would also be actively harmful: such a class typically has an empty Jacobian ROW, since a
	// residual is rarely tested against the opposite side alone, and the empty-row analysis below would
	// then mark a field that does not exist as one that must be pinned. The suffix is therefore
	// stripped on lookup, and only recorded for the structure-file report.
	void Problem::assemble_defined_field_list()
	{
		defined_fields.clear();
		side_split_field_classes.clear();
		defined_fields_to_domain.clear();
		residual_names.clear();
		std::set<std::string> field_set;
		std::set<std::string> res_jac_combis;
		std::map<std::string,unsigned> field_name_to_index;
		
		for (auto * dc : jit_codes)
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
		mass_matrix_contributing_fields.resize(residual_names.size());
		jacobian_block_flags_union.resize(residual_names.size());
		mass_matrix_block_flags_union.resize(residual_names.size());
		for (unsigned i=0;i<residual_names.size();i++)
		{
			residual_contributing_fields[i].resize(defined_fields.size(),false);
			jacobian_contributing_fields[i].resize(defined_fields.size(),std::vector<bool>(defined_fields.size(),false));
			jacobian_contributing_codes[i].resize(defined_fields.size(),std::vector<std::set<DynamicJITCode*>>(defined_fields.size(),std::set<DynamicJITCode*>()));
			residual_contributing_codes[i].resize(defined_fields.size(),std::set<DynamicJITCode*>());
			mass_matrix_contributing_fields[i].assign(defined_fields.size(),std::vector<bool>(defined_fields.size(),false));
			// Start from all-bits-set and AND per contributor; blocks without any contributor are zeroed below
			jacobian_block_flags_union[i].assign(defined_fields.size(),std::vector<unsigned char>(defined_fields.size(),0xFF));
			mass_matrix_block_flags_union[i].assign(defined_fields.size(),std::vector<unsigned char>(defined_fields.size(),0xFF));
			// Go over all bulk codes and check the contributions
			for (auto * dc : jit_codes)
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
					std::string fn=strip_side_split_suffix(ft->contribution_names[j],side_split_field_classes);
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
						std::string fn2=strip_side_split_suffix(ft->contribution_names[k],side_split_field_classes);
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
							jacobian_block_flags_union[i][row_index][col_index] &= (ft->jacobian_block_flags ? ft->jacobian_block_flags[my_i][j][k] : 0);
						}
						if (ft->contributes_to_mass_matrix[my_i][j][k])
						{
							mass_matrix_contributing_fields[i][row_index][col_index]=true;
							mass_matrix_block_flags_union[i][row_index][col_index] &= (ft->mass_matrix_block_flags ? ft->mass_matrix_block_flags[my_i][j][k] : 0);
						}
					}
				}
				
			}
		}
		// Finalize the block-flags union: blocks nothing contributes to drop from all-bits-set to 0,
		// and the pairwise symmetry bits must survive the AND on BOTH mirror entries - a code
		// contributing only to (j,i) breaks the pair relation of the assembled blocks even if every
		// contributor to (i,j) proves it, and shows up as an unset bit on the (j,i) side only.
		for (unsigned i=0;i<residual_names.size();i++)
		{
			for (unsigned j=0;j<defined_fields.size();j++)
			{
				for (unsigned k=0;k<defined_fields.size();k++)
				{
					if (!jacobian_contributing_fields[i][j][k]) jacobian_block_flags_union[i][j][k]=0;
					if (!mass_matrix_contributing_fields[i][j][k]) mass_matrix_block_flags_union[i][j][k]=0;
				}
			}
			const unsigned char pair_bits=JACOBIAN_BLOCK_SYMMETRIC|JACOBIAN_BLOCK_ANTISYMMETRIC;
			for (unsigned j=0;j<defined_fields.size();j++)
			{
				for (unsigned k=j+1;k<defined_fields.size();k++)
				{
					if (jacobian_contributing_fields[i][j][k] && jacobian_contributing_fields[i][k][j])
					{
						unsigned char common=jacobian_block_flags_union[i][j][k]&jacobian_block_flags_union[i][k][j]&pair_bits;
						jacobian_block_flags_union[i][j][k]=(jacobian_block_flags_union[i][j][k]&~pair_bits)|common;
						jacobian_block_flags_union[i][k][j]=(jacobian_block_flags_union[i][k][j]&~pair_bits)|common;
					}
					else
					{
						// Only one side contributes: a nonzero block cannot be the +-transpose of a zero one
						jacobian_block_flags_union[i][j][k]&=~pair_bits;
						jacobian_block_flags_union[i][k][j]&=~pair_bits;
					}
					if (mass_matrix_contributing_fields[i][j][k] && mass_matrix_contributing_fields[i][k][j])
					{
						unsigned char common=mass_matrix_block_flags_union[i][j][k]&mass_matrix_block_flags_union[i][k][j]&pair_bits;
						mass_matrix_block_flags_union[i][j][k]=(mass_matrix_block_flags_union[i][j][k]&~pair_bits)|common;
						mass_matrix_block_flags_union[i][k][j]=(mass_matrix_block_flags_union[i][k][j]&~pair_bits)|common;
					}
					else
					{
						mass_matrix_block_flags_union[i][j][k]&=~pair_bits;
						mass_matrix_block_flags_union[i][k][j]&=~pair_bits;
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
		for (auto * dc : jit_codes)
		{
			auto *ft=dc->get_func_table();
			//std::cout << "Processing element code " << dc->get_file_name() << " has dirichlet fields " << ft->Dirichlet_set_size << std::endl;
			for (unsigned int i=0;i<ft->Dirichlet_set_size;i++)
			{
				std::string dn=ft->Dirichlet_names[i];
				if (dn=="" || dn.find("__EXT_ODE_")==0) continue;
				if (!ft->moving_nodes && (dn=="coordinate_x" || dn=="coordinate_y" || dn=="coordinate_z")) continue;				
				//std::cout << "Looking for a field index for dirichlet field " << dn << std::endl;
				FiniteElementCode *current=dc->get_code_gen();
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
			// Also reused below to head the block-property matrices, which share the column layout
			auto print_column_header=[&]()
			{
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
			};
			print_column_header();
			
			std::vector<std::set<DynamicJITCode*>> listed_domain_contributions;
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
						for  (auto * dc : listed_domain_contributions[found]) if (!dc->get_code_gen()->is_residual_assembly_ignored(residual_names[ri])) ignored_residuals=false;
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
				// Sorted by name: listed_domain_contributions holds sets of *pointers*, so their
				// iteration order follows the addresses and changed between runs under ASLR. The
				// grouping is unaffected, but the dump could not be diffed across runs.
				std::vector<std::string> contrib_names;
				for (auto * dc : listed_domain_contributions[k])
					contrib_names.push_back(dc->get_code_gen()->get_full_domain_name());
				std::sort(contrib_names.begin(), contrib_names.end());
				unsigned int count=contrib_names.size();
				for (auto const &nm : contrib_names)
				{
					ss << nm << (--count ? " & " : "");
				}
				ss << std::endl;
			}
			ss << std::endl;

			// Per-block properties proven at code generation (JACOBIAN_BLOCK_* bits), AND-combined over
			// all contributing codes in assemble_defined_field_list. Purely informative: an unproven
			// property is shown blank, never claimed to be absent.
			auto print_flags_matrix=[&](const std::string &title, const std::vector<std::vector<unsigned char>> &flags, const std::vector<std::vector<bool>> &contribs)
			{
				ss << "\t" << title << " (S: symmetric pair, A: antisymmetric pair, C: constant, c: constant while the timestep is unchanged, .: contribution without proven properties)" << std::endl;
				print_column_header();
				for (unsigned int fi=0;fi<defined_fields.size();fi++)
				{
					if (pin_due_to_empty_jacobian_row_or_col[ri][fi]) continue;
					ss << "  \t" << std::setfill(' ') << std::setw(3) << fi << " | ";
					for (unsigned int fj=0;fj<defined_fields.size();fj++)
					{
						if (pin_due_to_empty_jacobian_row_or_col[ri][fj]) continue;
						if (!contribs[fi][fj])
						{
							ss << "  ";
							continue;
						}
						unsigned char f=flags[fi][fj];
						char sym=(f&JACOBIAN_BLOCK_SYMMETRIC) ? 'S' : ((f&JACOBIAN_BLOCK_ANTISYMMETRIC) ? 'A' : '.');
						char con=(f&JACOBIAN_BLOCK_CONSTANT) ? 'C' : ((f&JACOBIAN_BLOCK_CONSTANT_FIXED_DT) ? 'c' : ' ');
						ss << sym << con;
					}
					ss << "|" << std::endl;
				}
				ss << "\t----|";
				for (unsigned int fi=0;fi<defined_fields.size();fi++)
				{
					if (pin_due_to_empty_jacobian_row_or_col[ri][fi]) continue;
					ss << "--";
				}
				ss << "-|" << std::endl << std::endl;
			};
			print_flags_matrix("Jacobian block properties",jacobian_block_flags_union[ri],jacobian_contributing_fields[ri]);
			bool any_mass=false;
			for (auto &row : mass_matrix_contributing_fields[ri])
				for (auto b : row)
					any_mass=any_mass||b;
			if (any_mass)
				print_flags_matrix("Mass matrix block properties",mass_matrix_block_flags_union[ri],mass_matrix_contributing_fields[ri]);

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
				DynamicJITCode * dc=defined_fields_to_domain[i];
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

		// Informational, not a warning: these fields are ONE global unknown, listed once above, but the
		// elements on an interior facet distinguish the two sides of that facet and can therefore state
		// that the near-side and far-side copies do not couple. That is what makes an HDG system
		// decompose into one static-condensation block per element; see dev_docs/static_condensation.md.
		if (!side_split_field_classes.empty())
		{
			ss << std::endl << "\tSplit by facet side within the elements (one global field each, listed once above): ";
			unsigned int count=side_split_field_classes.size();
			for (const auto &n : side_split_field_classes)
				ss << n << (--count ? ", " : "");
			ss << std::endl;
		}

		return std::make_tuple(ss.str(), all_good);

	}

	// Reduces the per-block proofs of assemble_defined_field_list() to a whole-matrix verdict for the
	// named residual set: the assembled Jacobian (mass matrix) is symmetric iff every contributing
	// block carries JACOBIAN_BLOCK_SYMMETRIC. False means "not proven", never "disproven". Solvers use
	// this to switch to symmetric factorizations, so anything that makes the actually-solved matrix
	// differ from the plain elemental Jacobian must force (false,false) here: bifurcation-tracking /
	// DofAugmentations borders (n_unaugmented_dofs!=0 while augmented), a Python-side custom assembler,
	// and dR/dp replacement. Arclength continuation is fine - it borders algebraically, not in the matrix.
	// All inputs are rank-replicated, so the verdict is deterministic under MPI without a collective.
	std::tuple<bool,bool> Problem::get_proven_matrix_symmetry(const std::string &residual_name)
	{
		if (n_unaugmented_dofs != 0) return std::make_tuple(false, false);
		// The C++ bifurcation trackers do not touch n_unaugmented_dofs - they enlarge the system via an
		// oomph AssemblyHandler instead, so test for a non-default handler as well (same pair of gates
		// as the static-condensation eligibility check).
		oomph::AssemblyHandler *ah = this->assembly_handler_pt();
		if (!ah || typeid(*ah) != typeid(oomph::AssemblyHandler)) return std::make_tuple(false, false);
		if (use_custom_residual_jacobian) return std::make_tuple(false, false);
		if (replace_RJM_by_param_deriv) return std::make_tuple(false, false);
		int ri = -1;
		for (unsigned i = 0; i < residual_names.size(); i++)
			if (residual_names[i] == residual_name) { ri = i; break; }
		if (ri < 0 || jacobian_block_flags_union.size() != residual_names.size()) return std::make_tuple(false, false);
		// Unlike the report above, do NOT skip pin_due_to_empty_jacobian_row_or_col fields: whether
		// their rows are actually removed depends on the remove_dofs_without_jacobian_row argument of
		// _set_solved_residual (the eigenproblem path keeps them), so requiring the proof on every
		// contributing block is the conservative choice - at worst a false negative.
		bool jac_sym = true, mass_sym = true;
		for (unsigned j = 0; j < defined_fields.size(); j++)
		{
			for (unsigned k = 0; k < defined_fields.size(); k++)
			{
				if (jacobian_contributing_fields[ri][j][k] && !(jacobian_block_flags_union[ri][j][k] & JACOBIAN_BLOCK_SYMMETRIC)) jac_sym = false;
				if (mass_matrix_contributing_fields[ri][j][k] && !(mass_matrix_block_flags_union[ri][j][k] & JACOBIAN_BLOCK_SYMMETRIC)) mass_sym = false;
			}
		}
		return std::make_tuple(jac_sym, mass_sym);
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
