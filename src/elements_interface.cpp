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


// InterfaceElementBase: the face element that carries its own fields on the interface between (or
// on the boundary of) bulk domains. Equation remapping onto the attached bulk element(s), the
// external-data registration the JIT code depends on, interface dof creation and interpolation,
// and the interface normal derivatives.

#include "macroelements.hpp"
#include "elements.hpp"
#include "elements_concrete.hpp"
#include "exception.hpp"
#include "problem.hpp"
#include "nodes.hpp"
#include "meshtemplate.hpp"
#include "expressions.hpp"
#include "timestepper.hpp"

namespace pyoomph
{


	bool InterfaceElementBase::fill_hang_info_with_equations_interface(JITShapeInfo_t *shape_info)
	{
		bool res=false;
		auto * ft = codeinst->get_func_table();
		for (unsigned int ispace=0;ispace<ft->num_present_continuous_spaces;ispace++)
		{
			const JITFuncSpec_Table_FiniteElement_SpaceInfo_t * space_info = ft->present_continuous_spaces[ispace];
			unsigned nnode_space=eleminfo.nnode_of_space[space_info->space_index];
			const std::vector<unsigned> & space_node_to_elem_node=this->get_nodal_space_index_to_element_index_map()[space_info->space_index];
			for (unsigned int f = 0; f < space_info->numfields-space_info->numfields_basebulk; f++)
			{
				JITHangInfo_t * hangbuffer=shape_info->hanginfo[space_info->buffer_offset_interf+f];
				for (unsigned int l = 0; l < nnode_space; l++)
				{
					const unsigned l_elem=space_node_to_elem_node[l];
					if (oomph::HangInfo *hang_info_pt = this->hang_info_for_space(space_info, l_elem))
					{
						res = true;
						hangbuffer[l].nummaster = hang_info_pt->nmaster();
						for (unsigned m = 0; m < hang_info_pt->nmaster(); m++)
						{
							hangbuffer[l].masters[m].weight = hang_info_pt->master_weight(m);
							hangbuffer[l].masters[m].local_eqn = this->local_interface_hang_eqn( space_info->interface_dof_indices[f] , hang_info_pt->master_node_pt(m));
						}
					}
					else
					{
						hangbuffer[l].nummaster = 0;
					}
				}
			}
		}
		return res;
	}

	// Stage-3 scan counterpart of fill_hang_info_with_equations_interface() AND of the interface half
	// of fill_additional_hang_buffer_data() below: true exactly when either of them would report a
	// hang, without writing a buffer.
	bool InterfaceElementBase::scan_hang_interface_fields() const
	{
		auto *ft = codeinst->get_func_table();
		for (unsigned ispace = 0; ispace < ft->num_present_continuous_spaces; ispace++)
		{
			const JITFuncSpec_Table_FiniteElement_SpaceInfo_t *space_info = ft->present_continuous_spaces[ispace];
			if (space_info->numfields == space_info->numfields_basebulk)
				continue; // no interface-only fields in this space
			const std::vector<unsigned> &s2e = this->get_nodal_space_index_to_element_index_map()[space_info->space_index];
			const unsigned nnode_space = eleminfo.nnode_of_space[space_info->space_index];
			for (unsigned l = 0; l < nnode_space; l++)
			{
				if (this->hang_info_for_space(space_info, s2e[l]))
					return true;
				if (!has_additional_dof_constraints)
					continue;
				Node *n = static_cast<Node *>(node_pt(s2e[l]));
				if (!n)
					continue;
				for (const AdditionalDofConstrainingInfo *info = n->get_additional_dof_constraints(); info; info = info->next)
				{
					if (info->mode != INTERFACE_DOF_CONSTRAIN_TO_C1)
						continue;
					for (unsigned j = space_info->numfields_basebulk; j < space_info->numfields; j++)
						if (space_info->interface_dof_indices[j - space_info->numfields_basebulk] == info->index)
							return true;
				}
			}
		}
		return false;
	}

	// Scan counterpart of interpolate_hang_values_at_interface(): a hanging node in a space carrying
	// interface-only fields, or a dummy-value map entry for such a space.
	bool InterfaceElementBase::scan_interface_has_something_to_interpolate() const
	{
		auto *ft = codeinst->get_func_table();
		const std::vector<std::vector<std::vector<unsigned>>> &dummy_map = this->get_dummy_value_interpolation_map();
		for (unsigned ispace = 0; ispace < ft->num_present_continuous_spaces; ispace++)
		{
			const JITFuncSpec_Table_FiniteElement_SpaceInfo_t *space_info = ft->present_continuous_spaces[ispace];
			if (space_info->numfields <= space_info->numfields_basebulk)
				continue;
			if (!dummy_map[space_info->space_index].empty())
				return true;
			const std::vector<unsigned> &s2e = this->get_nodal_space_index_to_element_index_map()[space_info->space_index];
			for (unsigned l = 0; l < eleminfo.nnode_of_space[space_info->space_index]; l++)
				if (node_pt(s2e[l])->is_hanging(space_info->hangindex))
					return true;
		}
		return false;
	}

	// The recursion clause of the predicate: fill_hang_info_with_equations() below returns true for
	// ANY interface element that pulls in a bulk or opposite element, because it then writes the
	// local-equation REMAP into these very buffers. Skipping that fill would hand the generated code
	// another element's equation numbers, so the predicate has to keep reporting "hanging" here.
	bool InterfaceElementBase::hang_fill_would_report_hang(const JITFuncSpec_RequiredShapes_FiniteElement_t &required)
	{
		if (required.bulk_shapes || required.opposite_shapes)
			return true;
		return BulkElementBase::hang_fill_would_report_hang(required);
	}

	// Interface-field counterpart to BulkElementBase::fill_additional_hang_buffer_data: additionally
	// sets up the synthetic hanging scheme for interface-only dofs locally reduced to C1 via
	// add_additional_dof_constraint (mode INTERFACE_DOF_CONSTRAIN_TO_C1); "index" there is the
	// interface id, exactly as used by Node::additional_value_index and interface_dof_indices below.
	// Mirrors the chaining done in the base-bulk version: if a C1 corner master is itself hanging (its
	// own hangbuffer entry, already filled by fill_hang_info_with_equations_interface, has masters),
	// we inline that master's own masters (scaled by our weight) instead of referencing the hanging
	// node's own (non-existent) local equation directly; otherwise we use its direct nodal local eqn.
	bool InterfaceElementBase::fill_additional_hang_buffer_data(JITShapeInfo_t *shape_info)
	{
		bool res = BulkElementBase::fill_additional_hang_buffer_data(shape_info);
		if (!has_additional_dof_constraints)
			return res;
		auto * ft = codeinst->get_func_table();
		const std::vector<std::vector<unsigned>> &c1_dummy_map = this->get_dummy_value_interpolation_map()[SPACE_INDEX_C1];
		auto get_c1_masters = [&c1_dummy_map](unsigned l_elem) -> const std::vector<unsigned> *
		{
			for (const std::vector<unsigned> &entry : c1_dummy_map)
			{
				if (entry[0] == l_elem)
					return &entry;
			}
			return NULL;
		};
		bool has_C1_hanging= ft->continuous_spaces[SPACE_INDEX_C1].numfields_basebulk>0 || ft->continuous_spaces[SPACE_INDEX_C1TB].numfields_basebulk>0;

		for (unsigned int ispace = 0; ispace < ft->num_present_continuous_spaces; ispace++)
		{
			const JITFuncSpec_Table_FiniteElement_SpaceInfo_t * space_info = ft->present_continuous_spaces[ispace];
			if (space_info->numfields == space_info->numfields_basebulk)
				continue; // No interface fields for this space
			const std::vector<unsigned> & space_node_to_elem_node = this->get_nodal_space_index_to_element_index_map()[space_info->space_index];
			const std::vector<int> & elem_node_to_space_node = this->get_element_index_to_nodal_space_index_map()[space_info->space_index];
			unsigned nnode_space = eleminfo.nnode_of_space[space_info->space_index];
			for (unsigned int l = 0; l < nnode_space; l++)
			{
				const unsigned l_elem = space_node_to_elem_node[l];
				Node *n = static_cast<Node *>(node_pt(l_elem));
				for (const AdditionalDofConstrainingInfo *info = n->get_additional_dof_constraints(); info != NULL; info = info->next)
				{
					if (info->mode != INTERFACE_DOF_CONSTRAIN_TO_C1)
						continue;
					int f = -1;
					for (unsigned int j = space_info->numfields_basebulk; j < space_info->numfields; j++)
					{
						if (space_info->interface_dof_indices[j - space_info->numfields_basebulk] == info->index)
						{
							f = j - space_info->numfields_basebulk;
							break;
						}
					}
					if (f < 0)
						continue; // this interface id is not assigned within this space

					JITHangInfo_t * hangbuffer = shape_info->hanginfo[space_info->buffer_offset_interf + f];
					if (has_C1_hanging && n->is_hanging(ft->continuous_spaces[SPACE_INDEX_C1].hangindex))
					{
						// Genuinely hanging on the C1 space, not a C1-corner-averaged dummy node -> use the real hanging masters/weights instead of a plain average
						oomph::HangInfo *hang_info_pt = n->hanging_pt(ft->continuous_spaces[SPACE_INDEX_C1].hangindex);
						hangbuffer[l].nummaster = hang_info_pt->nmaster();
						for (unsigned m = 0; m < hang_info_pt->nmaster(); m++)
						{
							hangbuffer[l].masters[m].weight = hang_info_pt->master_weight(m);
							hangbuffer[l].masters[m].local_eqn = this->local_interface_hang_eqn(info->index, hang_info_pt->master_node_pt(m));
						}
						res = true;
						continue;
					}

					const std::vector<unsigned> *entry = get_c1_masters(l_elem);
					if (!entry)
					{
						throw_runtime_error("An interface dof constrained to C1 is only allowed on nodes which do not belong to the C1 space");
					}
					unsigned nmaster = entry->size() - 1;
					hangbuffer[l].nummaster = 0;
					for (unsigned m = 0; m < nmaster; m++)
					{
						unsigned l2 = elem_node_to_space_node[(*entry)[m + 1]];
						if (hangbuffer[l2].nummaster == 0)
						{
							pyoomph::BoundaryNode *master_bn = dynamic_cast<pyoomph::BoundaryNode *>(node_pt((*entry)[m + 1]));
							if (!master_bn)
								throw_runtime_error("This should be a boundary node here");
							unsigned master_value_index = master_bn->index_of_first_value_assigned_by_face_element(info->index);
							hangbuffer[l].masters[hangbuffer[l].nummaster].weight = 1.0 / (double)nmaster;
							hangbuffer[l].masters[hangbuffer[l].nummaster].local_eqn = this->nodal_local_eqn((*entry)[m + 1], master_value_index);
							hangbuffer[l].nummaster++;
						}
						else
						{
							for (int m2 = 0; m2 < hangbuffer[l2].nummaster; m2++)
							{
								hangbuffer[l].masters[hangbuffer[l].nummaster].weight = hangbuffer[l2].masters[m2].weight / (double)nmaster;
								hangbuffer[l].masters[hangbuffer[l].nummaster].local_eqn = hangbuffer[l2].masters[m2].local_eqn;
								hangbuffer[l].nummaster++;
							}
						}
					}
					res = true;
				}
			}
		}
		return res;
	}

	// An interface or facet element works on dofs that belong to OTHER elements: its own bulk element,
	// and on a DG interior facet also the opposite interface element and the opposite's bulk. The base
	// walk only knows about this element's own nodes and internal data, so all of those stayed "not
	// attributed" (-1), which the sparsity mask has to read conservatively as "coupled to everything".
	// That is not academic: on the DG convection-diffusion tutorial 33% of the frozen pattern was
	// stored zeros, and the breakdown put every single one of them in a row or column of an
	// unattributed facet dof, while the properly attributed domain/c block was exactly tight.
	//
	// Each source element already knows its own attribution, and update_equation_remapping_from_element
	// has already built the map from its local equation numbers into ours -- the same map the generated
	// code uses to address this external data. So the attribution is simply adopted through it.
	// Contribution class indices are per-code, so they are translated by NAME: both codes call the same
	// class "<domain>/<field>", which is exactly how the coupling tables are keyed.
	void InterfaceElementBase::fill_local_dof_contribution_indices(std::vector<int> &dest)
	{
		BulkElementBase::fill_local_dof_contribution_indices(dest);

		const JITFuncSpec_Table_FiniteElement_t *ft = codeinst->get_func_table();
		if (!ft || !ft->contribution_names || !ft->contribution_entries_size) return;
		std::map<std::string, int> my_class;
		for (unsigned c = 0; c < ft->contribution_entries_size; c++)
			if (ft->contribution_names[c]) my_class[std::string(ft->contribution_names[c])] = (int)c;

		// A source element never distinguishes sides - its own dofs are, from its point of view, simply
		// its own - so when we adopt from the FAR side of the facet we must ask for that side's class
		// ourselves. Our code may have split it off as "<name>@opposite" precisely so that it can state
		// that the two sides do not couple; if it did not split (a continuous field, or a code that did
		// not classify), the plain name is the right answer and is conservative.
		static const std::string opposite_suffix = "@opposite";
		auto adopt = [&](BulkElementBase *src, const std::vector<int> &eqn_map, bool opposite)
		{
			if (!src || eqn_map.empty() || src == this) return;
			const JITFuncSpec_Table_FiniteElement_t *sft = src->get_code_instance()->get_func_table();
			if (!sft || !sft->contribution_names) return;
			const std::vector<int> &sidx = src->get_local_dof_contribution_indices();
			const size_t n = std::min(eqn_map.size(), sidx.size());
			for (size_t sl = 0; sl < n; sl++)
			{
				const int my_le = eqn_map[sl];
				if (my_le < 0 || (size_t)my_le >= dest.size()) continue;
				if (dest[my_le] != -1) continue; // Already attributed by the walk above; leave it alone
				const int sc = sidx[sl];
				if (sc == -2) { dest[my_le] = -2; continue; } // "Part of no contribution" carries over
				if (sc < 0 || (unsigned)sc >= sft->contribution_entries_size) continue;
				const char *nm = sft->contribution_names[sc];
				if (!nm) continue;
				std::string name(nm);
				if (opposite)
				{
					// Never compose the marker: the opposite-side dummy shares our jitcode, so its own
					// class names can already carry it. Leaving such a dof unattributed (-1) makes it
					// couple to everything, which is the safe direction.
					if (name.size() > opposite_suffix.size() &&
						name.compare(name.size() - opposite_suffix.size(), opposite_suffix.size(), opposite_suffix) == 0)
						continue;
					auto op = my_class.find(name + opposite_suffix);
					if (op != my_class.end()) { dest[my_le] = op->second; continue; }
				}
				auto it = my_class.find(name);
				if (it != my_class.end()) dest[my_le] = it->second;
			}
		};

		adopt(dynamic_cast<BulkElementBase *>(this->bulk_element_pt()), bulk_eqn_map, false);
		if (opposite_side && !Is_internal_facet_opposite_dummy)
		{
			// A facet whose two sides are the same element has no far side to speak of, and the
			// disjointness the split relies on would not hold: adopt it as near-side throughout.
			const bool self_facet = (opposite_side->bulk_element_pt() == this->bulk_element_pt());
			adopt(opposite_side, opp_interf_eqn_map, !self_facet);
			adopt(dynamic_cast<BulkElementBase *>(opposite_side->bulk_element_pt()), opp_bulk_eqn_map, !self_facet);
		}
	}


	int InterfaceElementBase::local_interface_hang_eqn(unsigned int interface_dof_index, oomph::Node * master_node) const
    {
	  pyoomph::BoundaryNode *const master_bound_node = dynamic_cast<pyoomph::BoundaryNode *>(master_node);
	  if (!master_bound_node)
	  {
		throw_runtime_error("InterfaceElementBase::local_interface_hang_eqn: master_node is not a pyoomph::BoundaryNode");
	  }
      auto it = Local_interface_hang_eqn.find(interface_dof_index);
      if (it == Local_interface_hang_eqn.end())
	  {
        throw_runtime_error("InterfaceElementBase::local_interface_hang_eqn: interface_dof_index not found in Local_interface_hang_eqn");
	  }
      auto it2 = it->second.find(dynamic_cast<pyoomph::BoundaryNode *>(master_bound_node));
      if (it2 == it->second.end())
	  {
		throw_runtime_error("InterfaceElementBase::local_interface_hang_eqn: master_node not found in Local_interface_hang_eqn for the given interface_dof_index");
	  }
      return it2->second;
    }

	void InterfaceElementBase::fill_in_jacobian_from_nodal_by_fd(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian) 
	{
		HangInterpPassSuspension __no_pass; // the interface-only perturbation loop below re-enters get_residuals too
		BulkElementBase::fill_in_jacobian_from_nodal_by_fd(residuals, jacobian);
		const unsigned n_node = this->nnode();
		if (n_node == 0) return;
		this->update_before_nodal_fd();
		auto *ft=codeinst->get_func_table();
		const std::vector<std::vector<unsigned>> & space_node_to_elem_node_map=this->get_nodal_space_index_to_element_index_map();
		oomph::Vector<double> newres(residuals.size());
		for (unsigned int ispace=0;ispace<ft->num_present_continuous_spaces;ispace++)
		{
			auto *space_info=ft->present_continuous_spaces[ispace];
			if (space_info->numfields-space_info->numfields_basebulk)
			{
				for (unsigned n = 0; n < eleminfo.nnode_of_space[space_info->space_index]; n++)
				{
					oomph::Node *const local_node_pt = this->node_pt(space_node_to_elem_node_map[space_info->space_index][n]);
					pyoomph::BoundaryNode *const bound_node_pt = dynamic_cast<pyoomph::BoundaryNode *>(local_node_pt);
					if (!bound_node_pt)
					{
						throw_runtime_error("InterfaceElementBase::fill_in_jacobian_from_nodal_by_fd: node is not a pyoomph::BoundaryNode");
					}
					for (unsigned i = 0; i < space_info->numfields-space_info->numfields_basebulk; i++)
					{
						unsigned interface_dof_id=space_info->interface_dof_indices[i];
						if (local_node_pt->is_hanging(space_info->hangindex) == false)
						{
							unsigned val_index=bound_node_pt->index_of_first_value_assigned_by_face_element(interface_dof_id);
							int local_unknown = this->nodal_local_eqn(space_node_to_elem_node_map[space_info->space_index][n], val_index);
							if (local_unknown >= 0)
							{
								double *const value_pt = local_node_pt->value_pt(val_index);
								const double old_var = *value_pt;
								*value_pt += this->Default_fd_jacobian_step;
								this->update_in_nodal_fd(val_index);
								this->get_residuals(newres);
								for (unsigned m = 0; m < this->ndof(); m++)
								{
									jacobian(m, local_unknown) = (newres[m] - residuals[m]) / this->Default_fd_jacobian_step;
								}
								*value_pt = old_var;
								this->reset_in_nodal_fd(val_index);
							}
						}
						else
						{
							oomph::HangInfo *hang_info_pt = local_node_pt->hanging_pt(space_info->hangindex);
							const unsigned n_master = hang_info_pt->nmaster();
							for (unsigned m = 0; m < n_master; m++)
							{
								oomph::Node *const master_node_pt = hang_info_pt->master_node_pt(m);	
								pyoomph::BoundaryNode *const master_bound_node_pt = dynamic_cast<pyoomph::BoundaryNode *>(master_node_pt);								
								int local_unknown =  this->local_interface_hang_eqn(interface_dof_id, master_bound_node_pt);				
								unsigned val_index=master_bound_node_pt->index_of_first_value_assigned_by_face_element(interface_dof_id);
								if (local_unknown >= 0)
								{
									double *const value_pt = master_node_pt->value_pt(val_index);
									const double old_var = *value_pt;
									*value_pt += this->Default_fd_jacobian_step;
									this->update_in_nodal_fd(val_index);
									this->get_residuals(newres);
									for (unsigned mm = 0; mm < this->ndof(); mm++)
									{	
										jacobian(mm, local_unknown) = (newres[mm] - residuals[mm]) / this->Default_fd_jacobian_step;
									}
									*value_pt = old_var;
									this->reset_in_nodal_fd(val_index);
								}
							}
						}
					}
				}
			}
		}		
	}

	//////////////////////////////
	// InterfaceElementBase implements FaceElements attached to a bulk element (and optionally to an "opposite"
	// interface element across an internal facet), which carry their own additional degrees of freedom
	// (surface fields) while also needing access to the bulk (and opposite bulk/interface) element's degrees
	// of freedom, e.g. to evaluate normal derivatives. See update_equation_remapping() below for how this
	// access is implemented via a "fake hanging node" trick.

	// Builds a map from source_elem's local equation numbers to this element's local equation numbers, by
	// matching global equation numbers (used to let interface residuals/Jacobians reference bulk/opposite
	// element dofs that are already present as external data of this element).
	void InterfaceElementBase::update_equation_remapping_from_element(BulkElementBase *source_elem,const JITFuncSpec_RequiredShapes_FiniteElement_t *,std::vector<int> &eqn_map,int)
	{
		////////////// SIMPLEST APPROACH //////////////////
		eqn_map.clear();
		eqn_map.resize(source_elem->ndof(), -666); // Magic for not used/found
		std::map<unsigned,int> my_global_to_local;
		for (unsigned int i_my_local=0;i_my_local<this->ndof();i_my_local++)
		{
			int my_global_eq=this->eqn_number(i_my_local);
			if (my_global_eq>=0)
			{
				my_global_to_local[my_global_eq]=i_my_local;
			}
		}
		for (unsigned int i_source_local=0;i_source_local<source_elem->ndof();i_source_local++)
		{
			int source_global_eq=source_elem->eqn_number(i_source_local);
			if (source_global_eq>=0)
			{
				auto it=my_global_to_local.find(source_global_eq);
				if (it!=my_global_to_local.end())
				{
					eqn_map[i_source_local]=it->second;
				}
			}
		}
	}

	// For interface fields defined on hanging nodes, copies/interpolates the additional (interface-only) dof
	// values from the master nodes onto the hanging node itself, and likewise patches "dummy" nodes (nodes
	// whose interface value is not an independent dof but an average of other nodes, per
	// get_dummy_value_interpolation_map()) so that both hold consistent interpolated values for output/restart.
	void InterfaceElementBase::interpolate_hang_values_at_interface()
	{
		if (__measure_skip_hang_fills)
			return; // measurement lever only, see __measure_skip_hang_fills
		auto * ft=this->get_code_instance()->get_func_table();
		const std::vector<std::vector<std::vector<unsigned>>> & dummy_value_interpolation_map=this->get_dummy_value_interpolation_map();
		for (unsigned int ispace=0;ispace<ft->num_present_continuous_spaces;ispace++)
		{
			auto space_info=ft->present_continuous_spaces[ispace];
			const std::vector<unsigned> & get_nodal_space_index_to_element_index=this->get_nodal_space_index_to_element_index_map()[space_info->space_index];
			unsigned nnode=eleminfo.nnode_of_space[space_info->space_index];
			int hangindex=space_info->hangindex;
			// Only the values this INTERFACE adds to a node are interpolated here; the base-bulk ones are
			// already handled by the bulk element's own interpolate_hang_values. Without this early exit the
			// loop below insists on a BoundaryNode even when it would then copy nothing, which is wrong on
			// the interior-facet skeleton: its nodes are ordinary interior pyoomph::Nodes, and a hanging one
			// (2:1 facet after adaptation) tripped the "dest_bn is not a BoundaryNode" throw.
			if (space_info->numfields<=space_info->numfields_basebulk) continue;
			for (unsigned int inode=0;inode<nnode;inode++)
			{
				pyoomph::Node * node=static_cast<pyoomph::Node*>(this->node_pt(get_nodal_space_index_to_element_index[inode]));
				if (node->is_hanging(hangindex))
				{					
					pyoomph::BoundaryNode * dest_bn=dynamic_cast<pyoomph::BoundaryNode*>(node);
					if (!dest_bn) throw_runtime_error("dest_bn is not a BoundaryNode");
					oomph::HangInfo * hang_info=node->hanging_pt(hangindex);
					for (unsigned int field_index=0;field_index<space_info->numfields-space_info->numfields_basebulk;field_index++)
					{
					   unsigned add_field_index=space_info->interface_dof_indices[field_index];
					   unsigned dest_value_index=dest_bn->index_of_first_value_assigned_by_face_element(add_field_index);	
					   for (unsigned int t=0;t<node->ntstorage();t++)
					   {
						   double val=0.0;
						   for (unsigned int m=0;m<hang_info->nmaster();m++)
						   {
								pyoomph::Node * master_node=static_cast<pyoomph::Node*>(hang_info->master_node_pt(m));
								BoundaryNode * boundnode=dynamic_cast<BoundaryNode*>(master_node);
								if (!boundnode) throw_runtime_error("master_node is not a BoundaryNode");
								unsigned master_value_index=boundnode->index_of_first_value_assigned_by_face_element(add_field_index);
								val+=master_node->value(t,master_value_index)*hang_info->master_weight(m);
						   }
						   node->value_pt(dest_value_index)[t]=val;					   
					   }
					}
				
				}
			}

			const std::vector<std::vector<unsigned>> & dummy_value_interpolation=dummy_value_interpolation_map[space_info->space_index];
			if (!dummy_value_interpolation.empty() && space_info->numfields-space_info->numfields_basebulk>0)
			{
				for (unsigned int idummy=0;idummy<dummy_value_interpolation.size();idummy++)
				{
					pyoomph::Node * dummynode=static_cast<pyoomph::Node*>(this->node_pt(dummy_value_interpolation[idummy][0]));
					pyoomph::BoundaryNode * dest_boundnode=dynamic_cast<pyoomph::BoundaryNode*>(dummynode);
					if (!dest_boundnode) throw_runtime_error("dummynode is not a BoundaryNode");
					for (unsigned int t=0;t<dummynode->ntstorage();t++)
					{
						for (unsigned int field_index=0;field_index<space_info->numfields-space_info->numfields_basebulk;field_index++)
						{
					   	   unsigned add_field_index=space_info->interface_dof_indices[field_index];
						   double val=0.0;
						   for (unsigned int m=1;m<dummy_value_interpolation[idummy].size();m++)
						   {
							  pyoomph::Node * master_node=static_cast<pyoomph::Node*>(this->node_pt(dummy_value_interpolation[idummy][m]));
							  pyoomph::BoundaryNode * boundnode=dynamic_cast<pyoomph::BoundaryNode*>(master_node);
							  if (!boundnode) throw_runtime_error("master_node is not a BoundaryNode");
							  unsigned master_index=boundnode->index_of_first_value_assigned_by_face_element(add_field_index);
							  val+=master_node->value(t, master_index);
						   }
						   val/=(dummy_value_interpolation[idummy].size()-1.0);						   
						   unsigned dest_index=dest_boundnode->index_of_first_value_assigned_by_face_element(add_field_index);						   
						   dummynode->value_pt(dest_index)[t]=val;
						}
					}

				}
			}
		}

		// User-added additional interface dof constraints (INTERFACE_DOF_CONSTRAIN_TO_C1): mirrors
		// BulkElementBase::interpolate_hang_values' handling of CONTINUOUS_BASE_DOF_CONSTRAIN_TO_C1 -
		// these nodes are plain pinned dofs (see pin_dummy_values), not genuinely hanging, so push the
		// same C1-corner-averaged value used as the equation-assembly target in
		// fill_additional_hang_buffer_data into the raw storage for consistent output/restart.
		if (has_additional_dof_constraints)
		{
			const std::vector<std::vector<unsigned>> &c1_dummy_map = this->get_dummy_value_interpolation_map()[SPACE_INDEX_C1];
			auto get_c1_masters = [&c1_dummy_map](unsigned l_elem) -> const std::vector<unsigned> *
			{
				for (const std::vector<unsigned> &entry : c1_dummy_map)
				{
					if (entry[0] == l_elem)
						return &entry;
				}
				return NULL;
			};
			bool has_C1_hanging= ft->continuous_spaces[SPACE_INDEX_C1].numfields_basebulk>0 || ft->continuous_spaces[SPACE_INDEX_C1TB].numfields_basebulk>0;

			for (unsigned int l_elem = 0; l_elem < this->nnode(); l_elem++)
			{
				pyoomph::Node *n = static_cast<pyoomph::Node *>(node_pt(l_elem));
				pyoomph::BoundaryNode *dest_bn = dynamic_cast<pyoomph::BoundaryNode *>(n);
				bool hangs_on_C1 = has_C1_hanging && n->is_hanging(ft->continuous_spaces[SPACE_INDEX_C1].hangindex);
				for (const AdditionalDofConstrainingInfo *info = n->get_additional_dof_constraints(); info != NULL; info = info->next)
				{
					if (info->mode != INTERFACE_DOF_CONSTRAIN_TO_C1)
						continue;
					if (!dest_bn)
						throw_runtime_error("This should be a boundary node here");
					int dest_index = dest_bn->index_of_first_value_assigned_by_face_element(info->index);
					if (dest_index < 0)
						continue; // this interface id is not assigned on this node

					unsigned ntstorage = n->ntstorage();
					if (hangs_on_C1)
					{
						// Genuinely hanging on the C1 space, not a C1-corner-averaged dummy node -> use the real hanging masters/weights instead of a plain average
						oomph::HangInfo *hang_info_pt = n->hanging_pt(ft->continuous_spaces[SPACE_INDEX_C1].hangindex);
						for (unsigned t = 0; t < ntstorage; t++)
						{
							double val = 0.0;
							for (unsigned m = 0; m < hang_info_pt->nmaster(); m++)
							{
								pyoomph::BoundaryNode *master_bn = dynamic_cast<pyoomph::BoundaryNode *>(hang_info_pt->master_node_pt(m));
								if (!master_bn)
									throw_runtime_error("This should be a boundary node here");
								int master_index = master_bn->index_of_first_value_assigned_by_face_element(info->index);
								val += hang_info_pt->master_weight(m) * master_bn->value(t, master_index);
							}
							n->value_pt((unsigned)dest_index)[t] = val;
						}
						continue;
					}

					const std::vector<unsigned> *entry = get_c1_masters(l_elem);
					if (!entry)
					{
						throw_runtime_error("An interface dof constrained to C1 is only allowed on nodes which do not belong to the C1 space");
					}
					unsigned nmaster = entry->size() - 1;
					for (unsigned t = 0; t < ntstorage; t++)
					{
						double val = 0.0;
						for (unsigned m = 1; m < entry->size(); m++)
						{
							pyoomph::BoundaryNode *master_bn = dynamic_cast<pyoomph::BoundaryNode *>(node_pt((*entry)[m]));
							if (!master_bn)
								throw_runtime_error("This should be a boundary node here");
							int master_index = master_bn->index_of_first_value_assigned_by_face_element(info->index);
							val += master_bn->value(t, master_index);
						}
						n->value_pt((unsigned)dest_index)[t] = val / (double)nmaster;
					}
				}
			}
		}
	}


	// For one continuous field space, assigns local equation numbers for the additional interface-only dofs
	// living on hanging nodes of this element: each hanging node's interface dof is constrained to its
	// hanging-node masters, so instead of a true local dof, this builds (per master node) a map from master
	// node to either a new local equation number (registered as an external dof) or Data::Is_pinned. The
	// resulting per-field maps are stored in add_interf_local_hang_eqs for the generated code's hang_buffer trick.
	void InterfaceElementBase::assign_hanging_additional_interface_local_equations_for_space(const bool &store_local_dof_pt, JITFuncSpec_Table_FiniteElement_SpaceInfo_t * space)
	{
		unsigned local_eqn_number = ndof();
      	std::deque<unsigned long> global_eqn_number_queue;
		unsigned addfields=space->numfields-space->numfields_basebulk;
		if (addfields)
		{
			const std::vector<unsigned> & space_node_to_elem_map=this->get_nodal_space_index_to_element_index_map()[space->space_index];
			std::vector<std::map<oomph::Node*, bool>> local_eqn_number_done(addfields, std::map<oomph::Node*, bool>());									
			for (unsigned int inode=0;inode<eleminfo.nnode_of_space[space->space_index];inode++)
			{
				pyoomph::Node * node=static_cast<pyoomph::Node*>(this->node_pt(space_node_to_elem_map[inode]));
				if (node->is_hanging(space->hangindex))
				{
					oomph::HangInfo * hang_info=node->hanging_pt(space->hangindex);
					for (unsigned int m=0;m<hang_info->nmaster();m++)
					{
						pyoomph::Node * master_node=static_cast<pyoomph::Node*>(hang_info->master_node_pt(m));
						pyoomph::BoundaryNode * boundnode=dynamic_cast<pyoomph::BoundaryNode*>(master_node);
						if (!boundnode) throw_runtime_error("master_node is not a BoundaryNode");

						unsigned local_node_index = this->nnode();                	
                		for (unsigned n1 = 0; n1 < this->nnode(); n1++)
                		{                  
                  			if (master_node == node_pt(n1))
                  			{
                    			local_node_index = n1;
                    			break;
                  			}
                		}
                		if (local_node_index < this->nnode())
                		{             
							for (unsigned int ifield=0;ifield<addfields;ifield++)
							{
								unsigned interface_id=space->interface_dof_indices[ifield];
								unsigned master_value_index=boundnode->index_of_first_value_assigned_by_face_element(interface_id);     
                  				Local_interface_hang_eqn[interface_id][boundnode]  =nodal_local_eqn(local_node_index, master_value_index);
                			}
						}
						else
						{						
							for (unsigned int ifield=0;ifield<addfields;ifield++)
							{
								unsigned interface_id=space->interface_dof_indices[ifield];
								unsigned master_value_index=boundnode->index_of_first_value_assigned_by_face_element(interface_id);
								long eqn_number = master_node->eqn_number(master_value_index);							
								if (eqn_number >= 0)
								{	
									if (Local_interface_hang_eqn[ifield].find(boundnode) == Local_interface_hang_eqn[ifield].end())
									{																			
										Local_interface_hang_eqn[interface_id][boundnode] = local_eqn_number;																		
										global_eqn_number_queue.push_back(eqn_number);								
										if (store_local_dof_pt)
										{
											GeneralisedElement::Dof_pt_deque.push_back(master_node->value_pt(master_value_index));
										}									
										local_eqn_number++;
									}
								}
								else
								{
									Local_interface_hang_eqn[interface_id][boundnode] = oomph::Data::Is_pinned;
								}
							}
						}
					}					
				}
			}
		}

		if (!global_eqn_number_queue.empty())
		{
	  		add_global_eqn_numbers(global_eqn_number_queue,GeneralisedElement::Dof_pt_deque);
      		if (store_local_dof_pt)
      		{
        		std::deque<double*>().swap(GeneralisedElement::Dof_pt_deque);
      		}
		}            	
	
	}


	void InterfaceElementBase::assign_hanging_additional_interface_local_equations(const bool &store_local_dof_pt)
	{
		Local_interface_hang_eqn.clear();
		
		auto * ft=this->get_code_instance()->get_func_table();
		for (unsigned int ispace=0;ispace<ft->num_present_continuous_spaces;ispace++)
		{
			auto space_info=ft->present_continuous_spaces[ispace];
			assign_hanging_additional_interface_local_equations_for_space(store_local_dof_pt,space_info);
		}
	}


	void InterfaceElementBase::update_equation_remapping()
	{		
		/*
			Interface elements store their own equations (defined on this element) intrisically. They are filled in the eleminfo
			If you want to access bulk element data, e.g. if you need a normal bulk gradient, the bulk degrees are added as external data elsewhere			
			In the generated code, we work with local equations of the element, i.e. 0..(N_dof_elem-1). These include the external data
			However, the generated code somehow must know how a local equation from the bulk element is mapped to the local equation of the interface element
			We use a trick in the generated here and use the hang_buffer, which is usually used for hanging nodes, i.e. degrees of freedom which are constrainted by a neighboring coarser element in adaptive solves (for continuity of the solution)
			The generated code just accesses the local equations of the bulk element, but then we fake a hanging node, which just hangs on a single master, which is the corresponding local equation (of the external data) in the interface element
		*/

		const JITFuncSpec_Table_FiniteElement_t *functable = this->codeinst->get_func_table();
		// Must be the very set the attachment used: this hands out local equation numbers OF the attached
		// external data, so remapping from a wider set would resolve dofs the element does not carry.
		const JITFuncSpec_RequiredShapes_FiniteElement_t &attach_req = *attachment_required_shapes(functable);
		update_equation_remapping_from_element(dynamic_cast<BulkElementBase *>(this->bulk_element_pt()),attach_req.bulk_shapes , bulk_eqn_map,1);	
		if (attach_req.bulk_shapes && attach_req.bulk_shapes->bulk_shapes)
		{
			update_equation_remapping_from_element(dynamic_cast<BulkElementBase *>(dynamic_cast<InterfaceElementBase *>(this->bulk_element_pt())->bulk_element_pt()),attach_req.bulk_shapes->bulk_shapes,  bulk_bulk_eqn_map,2);
		}
		if (attach_req.opposite_shapes && !is_internal_facet_opposite_dummy())
		{
			if (!opposite_side)
			{
				throw_runtime_error("Missing opposite element");
			}
			if (!this->is_internal_facet_opposite_dummy())
			{		
				update_equation_remapping_from_element(opposite_side,attach_req.opposite_shapes,  opp_interf_eqn_map,-1);
				if (attach_req.opposite_shapes->bulk_shapes)
				{
					if (!opposite_side->bulk_element_pt())
					{
						throw_runtime_error("Missing opposite bulk element");
					}			
					update_equation_remapping_from_element(dynamic_cast<BulkElementBase *>(opposite_side->bulk_element_pt()),attach_req.opposite_shapes->bulk_shapes , opp_bulk_eqn_map,-2);
				}
			}
		}
	}



	// The following four get_DG_* methods implement access to Discontinuous-Galerkin (DG) field data/dofs
	// through an interface element: fields with fieldindex below numfields_basebulk/numfields_bulk live on
	// the attached bulk element (accessed via external data / the bulk element's own DG indexing), while the
	// remaining ("interface-only") DG fields are genuinely owned by this interface element as internal data.

	// Index into the merged per-element JIT shape/value buffer for a DG field: base-bulk fields are placed at
	// buffer_offset_basebulk, interface-only DG fields after them at buffer_offset_interf.
	unsigned InterfaceElementBase::get_DG_buffer_index(const unsigned &space_index,const unsigned &fieldindex)
    {
		auto * ft=this->get_code_instance()->get_func_table();
		auto & space_info=ft->dg_spaces[space_index];

		if (fieldindex<space_info.numfields_basebulk)
		{
			return space_info.buffer_offset_basebulk+ fieldindex;
		}
		else
		{
			return  space_info.buffer_offset_interf +(fieldindex-space_info.numfields_basebulk);
		}
    }

	// Maps a local node index of this interface element to the corresponding DG node index: for base-bulk
	// fields this resolves through the bulk element's own node numbering (since the field is really owned
	// there); for interface-only DG fields the local node index is used directly.
	unsigned InterfaceElementBase::get_DG_node_index(const unsigned &space_index,const unsigned &fieldindex,const unsigned &nodeindex) const
	{
		auto * ft=this->codeinst->get_func_table();
		auto & space_info=ft->dg_spaces[space_index];
		if (fieldindex>=space_info.numfields_bulk) 	return nodeindex;
		else
		{
			int pnodeindex=this->get_nodal_space_index_to_element_index_map()[space_info.space_index][nodeindex];
			if (pnodeindex<0) throw_runtime_error("Strange");
			pnodeindex=this->bulk_node_number(pnodeindex);			
			BulkElementBase* be=dynamic_cast<BulkElementBase*>(this->bulk_element_pt());
			return be->get_DG_node_index(space_info.space_index, fieldindex, be->get_element_index_to_nodal_space_index_map()[space_info.space_index][pnodeindex]);
		}
	}

	// Returns the Data object holding a DG field's values: interface-only DG fields are internal data of this
	// element, base-bulk DG fields are accessed as external data (the bulk element's own data, registered
	// as external data of this interface element).
	oomph::Data * InterfaceElementBase::get_DG_nodal_data(const unsigned & space_index,const unsigned & fieldindex )
	{
		auto * ft=this->codeinst->get_func_table();
		auto & space_info=ft->dg_spaces[space_index];
		if (fieldindex>=space_info.numfields_bulk) return this->internal_data_pt(space_info.internal_offset_new+(fieldindex-space_info.numfields_bulk));
		else
		{
			return this->external_data_pt(space_info.external_offset_bulk +fieldindex);			
		}
	}

	// Local equation number for a DG field's dof, mirroring get_DG_nodal_data: internal dof for interface-only
	// DG fields, external dof (resolved through get_DG_node_index) for base-bulk DG fields.
	int InterfaceElementBase::get_DG_local_equation(const unsigned &space_index,const unsigned &fieldindex,const unsigned & nodeindex)
	{
		auto * ft=this->codeinst->get_func_table();
		auto space_info=ft->dg_spaces[space_index];
		if (fieldindex>=space_info.numfields_bulk) return this->internal_local_eqn(space_info.internal_offset_new+(fieldindex-space_info.numfields_bulk),nodeindex);
		else
		{
			return this->external_local_eqn(space_info.external_offset_bulk +fieldindex,this->get_DG_node_index(space_info.space_index, fieldindex,nodeindex));			
		}
	}


   // Python/generated-code accessor for the equation-remapping vectors built by update_equation_remapping().
   std::vector<int> InterfaceElementBase::get_attached_element_equation_mapping(const std::string & which)
   {
    if (which=="bulk") return bulk_eqn_map;
    else if (which=="opposite_interface") return  opp_interf_eqn_map;
    else if (which=="opposite_bulk") return opp_bulk_eqn_map;
    else if (which=="bulk_bulk") return bulk_bulk_eqn_map;
    else throw_runtime_error("Unknown map "+which);
   }
   
	// Resolves a field name to a nodal value index, first trying the bulk field lookup, then falling back to
	// searching this element's own (interface-only) continuous fields, returning the boundary-node value
	// index assigned to that field by this face element.
	int InterfaceElementBase::get_nodal_index_by_name(oomph::Node *n, std::string fieldname)
	{
		int bres = BulkElementBase::get_nodal_index_by_name(n, fieldname);
		if (bres >= 0)
			return bres;
		// Interface fields
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();

		for (unsigned int si=0;si<functable->num_present_continuous_spaces;si++)
		{
			auto * space_info=functable->present_continuous_spaces[si];
			for (unsigned int j = 0; j < space_info->numfields - space_info->numfields_basebulk; j++)
			{
				std::string intername = space_info->fieldnames[space_info->numfields_basebulk + j];
				if (intername == fieldname)
				{
					unsigned interf_id = space_info->interface_dof_indices[j];
					return dynamic_cast<oomph::BoundaryNodeBase *>(n)->index_of_first_value_assigned_by_face_element(interf_id);
				}
			}
		}
		return -1;
	}

	// Interface-specific counterpart to BulkElementBase::fill_element_info(): populates the eleminfo buffer
	// (value pointers and local equation numbers, consumed by the JIT-generated residual/Jacobian code) for
	// the additional interface-only continuous fields, DG interface fields, DL (elemental discontinuous) and
	// D0 fields that live on this interface element itself, on top of what the base class already filled in
	// for the shared/bulk fields.
	void InterfaceElementBase::fill_element_info_interface_part(bool without_equations)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();

		const std::vector<std::vector<unsigned>> & space_to_elem_index = this->get_nodal_space_index_to_element_index_map();

		for (unsigned int si=0;si<functable->num_present_continuous_spaces;si++)
		{
			auto * space_info=functable->present_continuous_spaces[si];			
			for (unsigned int i = 0; i < eleminfo.nnode_of_space[space_info->space_index]; i++)
			{
				unsigned i_el = space_to_elem_index[space_info->space_index][i];
				for (unsigned int j = 0; j < space_info->numfields - space_info->numfields_basebulk; j++)
				{
					unsigned node_index = j + space_info->buffer_offset_interf;
					unsigned interf_id = space_info->interface_dof_indices[j];
					unsigned valindex = dynamic_cast<oomph::BoundaryNodeBase *>(this->node_pt(i_el))->index_of_first_value_assigned_by_face_element(interf_id);
					eleminfo.nodal_data[i][node_index] = node_pt(i_el)->value_pt(valindex);
					if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] = this->nodal_local_eqn(i_el, valindex);
				}
			}
		}

		for (unsigned int si=0;si<functable->num_present_dg_spaces;si++)
		{
			auto * space_info=functable->present_dg_spaces[si];			
			for (unsigned int j=0;j<space_info->numfields-space_info->numfields_basebulk;j++)
			{
				unsigned node_index = j + space_info->buffer_offset_interf;
				oomph::Data * data=this->get_DG_nodal_data(space_info->space_index, space_info->numfields_basebulk+j);
				for (unsigned int i=0;i<eleminfo.nnode_of_space[space_info->space_index];i++)
				{
					unsigned valindex=this->get_DG_node_index(space_info->space_index, space_info->numfields_basebulk+j, i);
					eleminfo.nodal_data[i][node_index] = data->value_pt(valindex);
					if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] = this->get_DG_local_equation(space_info->space_index, space_info->numfields_basebulk+j, i);
				}
			}
		}


		for (unsigned int i = 0; i < eleminfo.nnode_DL; i++)
		{
			for (unsigned int j = 0; j < functable->info_DL.numfields; j++)
			{
				unsigned node_index = j + functable->info_DL.buffer_offset_basebulk;
				eleminfo.nodal_data[i][node_index] = internal_data_pt(functable->info_DL.internal_offset_new + j)->value_pt(i);
				if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] = this->internal_local_eqn(functable->info_DL.internal_offset_new + j, i);
			}
		}

		//	if (functable->info_D0.numfields)
		//	{
		// throw_runtime_error("TODO: D0 interface fields "+std::to_string(local_field_offset));
		for (unsigned int j = 0; j < functable->info_D0.numfields; j++)
		{
			unsigned node_index = j + functable->info_D0.buffer_offset_basebulk;
			eleminfo.nodal_data[0][node_index] = internal_data_pt(functable->info_D0.internal_offset_new + j)->value_pt(0);
			if (!without_equations) eleminfo.nodal_local_eqn[0][node_index] = this->internal_local_eqn(functable->info_D0.internal_offset_new + j, 0);
		}
		//	}
		/* ///NOTE: EXT DATA SHOULD BE ALWAYS AT THE END AT THE MOMENT
		local_field_offset+=functable->info_D0.numfields;
		for (unsigned int j=0;j<functable->numfields_ED0;j++)
		{
		   unsigned node_index=j+local_field_offset;
			std::cout << "INTEF NODE INDEX oF " << functable->fieldnames_ED0[i] << " IS " << node_index << std::endl;
			if (!codeinst->linked_external_data[j].data) throw_runtime_error("Element has an external data contribution, which is not assigned: "+std::string(functable->fieldnames_ED0[j]));
			int extdata_i=codeinst->linked_external_data[j].elemental_index;
			if (extdata_i>=(int)this->nexternal_data())  throw_runtime_error("Somehow the external data array was not done well when trying to index data: "+std::string(functable->fieldnames_ED0[i])+"  ext_data_index is "+std::to_string(extdata_i)+", but only "+std::to_string((int)this->nexternal_data())+" ext data slots present");
			int value_i=codeinst->linked_external_data[j].value_index;
			if (value_i<0 || value_i>=(int)this->external_data_pt(extdata_i)->nvalue())  throw_runtime_error("Somehow the external data array was not done, i.e. wrong value index, well when trying to index data: "+std::string(functable->fieldnames_ED0[j])+" at value "+std::to_string(value_i));
			 eleminfo.nodal_data[0][node_index]=this->external_data_pt(extdata_i)->value_pt(value_i);
			 eleminfo.nodal_local_eqn[0][node_index]=this->external_local_eqn(extdata_i,value_i);
		}
		local_field_offset+=functable->numfields_ED0;
	*/
	}
	
	
  // --- Quad2dFaceOrientation: opposite-side matching for quadrilateral face elements -------------
  //
  // The 8 symmetries of the square, written as maps of the face element's own local coordinate
  // s in [-1,1]^2. Index 0..3 are the identity/mirrors, 4..7 the transposed (orientation-reversing
  // resp. rotated) ones. Two faces of two different bulk elements share their geometry but not
  // their local coordinate system, so exactly one of these carries s on one side to the s of the
  // same physical point on the other.
  oomph::Vector<double> Quad2dFaceOrientation::map_s(int orientation, const oomph::Vector<double> &s)
  {
    oomph::Vector<double> res(2);
    switch (orientation)
    {
      case 0: res[0] =  s[0]; res[1] =  s[1]; break;
      case 1: res[0] = -s[0]; res[1] =  s[1]; break;
      case 2: res[0] =  s[0]; res[1] = -s[1]; break;
      case 3: res[0] = -s[0]; res[1] = -s[1]; break;
      case 4: res[0] =  s[1]; res[1] =  s[0]; break;
      case 5: res[0] = -s[1]; res[1] =  s[0]; break;
      case 6: res[0] =  s[1]; res[1] = -s[0]; break;
      case 7: res[0] = -s[1]; res[1] = -s[0]; break;
      default: throw_runtime_error("Invalid quad face orientation " + std::to_string(orientation));
    }
    return res;
  }

  std::vector<int> Quad2dFaceOrientation::node_index_map(int orientation, unsigned nnode_1d)
  {
    if (nnode_1d < 2) throw_runtime_error("Quad face elements have at least 2 nodes per direction");
    const double h = 2.0 / (nnode_1d - 1.0);
    std::vector<int> res(nnode_1d * nnode_1d, -1);
    for (unsigned j = 0; j < nnode_1d; j++)
    {
      for (unsigned i = 0; i < nnode_1d; i++)
      {
        oomph::Vector<double> s(2);
        s[0] = -1.0 + h * i;
        s[1] = -1.0 + h * j;
        oomph::Vector<double> so = map_s(orientation, s);
        // The symmetries permute the tensor grid exactly, so rounding here is only removing the
        // representation error of -1+h*i, never a genuine mismatch.
        int io = static_cast<int>(std::lround((so[0] + 1.0) / h));
        int jo = static_cast<int>(std::lround((so[1] + 1.0) / h));
        res[i + nnode_1d * j] = io + nnode_1d * jo;
      }
    }
    return res;
  }

  void Quad2dFaceOrientation::analyze(const oomph::FiniteElement *self, const oomph::FiniteElement *opposite, const std::vector<double> &offset, unsigned nnode_1d, int &orientation, std::vector<int> &node_index)
  {
    if (!opposite) throw_runtime_error("No opposite side set");
    if (opposite->dim() != 2) throw_runtime_error("Can only connect a 2d InterfaceElement to a 2d InterfaceElement");
    if (self->nvertex_node() != opposite->nvertex_node()) throw_runtime_error("Can only connect InterfaceElements with same number of vertex nodes");
    // For nnode_1d==2 the node map IS the vertex permutation (oomph numbers a quad's vertex nodes
    // in the same tensor order 00,10,01,11), so the candidates are generated instead of tabulated.
    std::vector<double> pdists(8, 0.0);
    for (unsigned int i = 0; i < self->nvertex_node(); i++)
    {
      pyoomph::Node *nthis = static_cast<pyoomph::Node *>(self->vertex_node_pt(i));
      for (int p = 0; p < 8; p++)
      {
        std::vector<int> perm = node_index_map(p, 2);
        pyoomph::Node *nopp = static_cast<pyoomph::Node *>(opposite->vertex_node_pt(perm[i]));
        for (unsigned int k = 0; k < std::min(nthis->ndim(), nopp->ndim()); k++)
          pdists[p] += (nthis->x(k) - nopp->x(k) + offset[k]) * (nthis->x(k) - nopp->x(k) + offset[k]);
      }
    }
    double best_dist = pdists[0];
    orientation = 0;
    for (int p = 1; p < 8; p++)
    {
      if (pdists[p] < best_dist)
      {
        best_dist = pdists[p];
        orientation = p;
      }
    }
    if (best_dist > 1e-14) throw_runtime_error("Vertex nodes are not matching here");
    node_index = node_index_map(orientation, nnode_1d);
  }

  // Finds the local coordinate s on this (surface) element whose interpolated position best matches the
  // given global point x, by first prescreening over the integration knots for a good starting guess, then
  // Newton-iterating (with finite-difference Jacobian) on the residual "tangential displacement dot direction",
  // i.e. driving x(s) towards x. Currently only implemented for 1d elements (edim==1); higher-dim solve is a TODO.
  oomph::Vector<double> InterfaceElementBase::optimize_s_to_match_x(const oomph::Vector<double> & x)
  {
   unsigned edim=this->dim();
   unsigned ndim=this->nodal_dimension();
   unsigned nnode=this->nnode();
   if (ndim!=x.size()) throw_runtime_error("Mismatching size: "+std::to_string(ndim)+" vs. "+std::to_string(x.size()));
   
   // Prescreen via the integration knots
   double best_dist=1e20;
   oomph::Vector<double> current_s;
   for (unsigned ipt = 0; ipt < integral_pt()->nweight(); ipt++)
	{
		oomph::Vector<double> s(edim),xtest(ndim,0.0);
		for (unsigned int i = 0; i < this->dim(); i++) s[i] = integral_pt()->knot(ipt, i);
		this->interpolated_x(s,xtest);
      double dist=0.0;
      for (unsigned k=0;k<x.size();k++) dist+=pow(xtest[k]-x[k],2);
      if (dist<best_dist) { best_dist=dist; current_s=s;}
   }
   
   auto get_residual_at_s=[&](oomph::Vector<double> s)->oomph::Vector<double>
   {
      oomph::Vector<double> xtest(ndim,0.0),R(edim,0.0);
      this->interpolated_x(s,xtest);
		oomph::DenseMatrix<double> interpolated_dxds(edim,ndim,0.0);
      oomph::Shape psi(nnode);
      oomph::DShape dpsids(nnode,edim);
      this->dshape_local(current_s,psi,dpsids);		
		for(unsigned l=0;l<nnode;l++)
		 {
	    for(unsigned j=0;j<edim;j++)
	     {
	      for(unsigned i=0;i<ndim;i++)
	       {
	        interpolated_dxds(j,i) += this->nodal_position(l,i)*dpsids(l,j);
	       }
	     }
		 }
		       
      for (unsigned int j=0;j<edim;j++)
      {
       for (unsigned int i=0;i<ndim;i++)
       {
        R[j]+=interpolated_dxds(j,i)*(xtest[i]-x[i]);
       }
      }   
      return R;
   };
   
   unsigned max_newton=20;
   double FD_eps=1e-8;
   for (unsigned int step=0;step<max_newton;step++)
   {
     oomph::Vector<double> R=get_residual_at_s(current_s);
     oomph::Vector<double> xtest(ndim,0.0);
     this->interpolated_x(current_s,xtest);     
     double dist=0.0;
     for (unsigned k=0;k<x.size();k++) dist+=pow(xtest[k]-x[k],2);     
     if (dist<1e-16) break;
//     std::cout << "STEP " << step << " DIST " << dist << "  s=" << current_s[0] << "  x " << xtest[0] << " , " << xtest[1] << " DEST " << x[0] << " , " << x[1] << std::endl;
     oomph::DenseDoubleMatrix J(edim,edim,0.0);
     for (unsigned int k=0;k<edim;k++)
     {
       oomph::Vector<double> spert=current_s;
       spert[k]+=FD_eps;
       oomph::Vector<double> R_pert=get_residual_at_s(spert);
       for (unsigned int j=0;j<edim;j++)
       {
         J(j,k)=-(R_pert[j]-R[j])/FD_eps;
       }
     }
     oomph::Vector<double> ds(edim,0.0);
     if (edim==1)
     {
       ds[0]=R[0]/J(0,0);
     }
     else
     {
      throw_runtime_error("Implement");
      // J.solve(R,ds);
     }
     for(unsigned i=0;i<edim;i++) {current_s[i] += ds[i];}     
   }
				
   return current_s;
  }

	// Marks this face element as the dummy standing in for the opposite side of an interior facet.
	// Such an element is constructed with the same jitcode as the real facet element (and hence
	// allocates the same element-owned DG/DL/D0 internal Data in allocate_discontinous_fields), but it
	// is never add_element_pt'd anywhere, so Problem::assign_eqn_numbers never visits that Data: it
	// keeps oomph-lib's "unclassified" equation number forever. Pinning it here turns those slots into
	// honestly pinned values, which is what the local-eqn path in assign_additional_local_eqn_numbers
	// and every consumer of internal_local_eqn() expect; leaving them unclassified relied on
	// unclassified and pinned happening to be treated alike. Done here rather than in
	// generate_interface_elements because this runs exactly once per dummy, right after construction,
	// and needs nothing but the element itself.
	void InterfaceElementBase::set_as_internal_facet_opposite_dummy()
	{
		Is_internal_facet_opposite_dummy = true;
		// Only allocate_discontinous_fields() ever adds internal Data to a pyoomph element, so this is
		// exactly the DG-new/DL/D0 storage of the interface level; the bulk element's dofs are untouched.
		for (unsigned int i = 0; i < this->ninternal_data(); i++)
		{
			oomph::Data *d = this->internal_data_pt(i);
			for (unsigned int v = 0; v < d->nvalue(); v++)
				d->pin(v);
		}
	}

	// Allocates the "additional values" (interface-only dofs) on this element's boundary nodes for every
	// interface field of every continuous space present, using oomph-lib's BoundaryNodeBase machinery so
	// several interface elements sharing a node can share/independently own the corresponding slots. Newly
	// allocated (not previously present) dofs are optionally seeded via interpolate_newly_constructed_additional_dof.
	void InterfaceElementBase::add_interface_dofs()
	{
		auto *ft = codeinst->get_func_table();

		// Continuous interface-only fields are stored as oomph-lib "additional values" on the element's
		// nodes, which only BoundaryNodes provide. The interior-facet skeleton is built from ordinary
		// interior nodes (plain pyoomph::Node), so the dynamic_cast<BoundaryNode*> in the loop below
		// returns NULL and dereferencing it segfaulted. Equations._internal_define_scalar_field already
		// refuses such a declaration; this is the backstop in case one reaches the C++ side otherwise.
		if (std::string(ft->domain_name) == "_internal_facets_")
		{
			for (unsigned int i = 0; i < ft->num_present_continuous_spaces; i++)
			{
				const JITFuncSpec_Table_FiniteElement_SpaceInfo_t *space_info = ft->present_continuous_spaces[i];
				if (space_info->numfields > space_info->numfields_bulk)
				{
					throw_runtime_error("Continuous fields on the interior-facet skeleton '_internal_facets_' are not supported (field '" + std::string(space_info->fieldnames[space_info->numfields_bulk]) + "' on space " + std::string(space_info->space_name) + "). Use a discontinuous facet space instead, i.e. D0, DL, D1, D1TB, D2 or D2TB.");
				}
			}
		}

		for (unsigned int i=0;i<ft->num_present_continuous_spaces;i++)
		{
			const JITFuncSpec_Table_FiniteElement_SpaceInfo_t * space_info = ft->present_continuous_spaces[i];
			for (unsigned int i=space_info->numfields_bulk;i<space_info->numfields;i++)
			{
				std::string fieldname = space_info->fieldnames[i];
				
				unsigned value_index=space_info->interface_dof_indices[i-space_info->numfields_basebulk];
				// TODO: Can be removed once we are sure that the interface dof indices are always correct
				unsigned value_index1 = codeinst->resolve_interface_dof_id(fieldname);
				if (value_index1!=value_index) throw_runtime_error("Mismatch between resolved interface dof id and space info index for field "+fieldname+" "+std::to_string(value_index1)+" vs. "+std::to_string(value_index));
				
				oomph::Vector<unsigned> additional_data_values(eleminfo.nnode, 0);
				bool add_values = false;
				std::vector<bool> already_allocated;
				for (unsigned l = 0; l < eleminfo.nnode; ++l)
				{
					additional_data_values[l] = 1;
					already_allocated.push_back(dynamic_cast<BoundaryNode*>(this->node_pt(l))->has_additional_dof(value_index));
					add_values = true;
				}
				if (add_values)
				{
					this->add_additional_values(additional_data_values, value_index);
				   for (unsigned l = 0; l < eleminfo.nnode; ++l)
				   {
					  if (additional_data_values[l] && !already_allocated[l] && interpolate_new_interface_dofs) this->interpolate_newly_constructed_additional_dof(l,value_index,space_info->space_name);
					}				
				}
			}
		}
		
	}

	// Seeds the value of a freshly-allocated interface dof (on node lnode) by interpolating it from the
	// corresponding field on the father bulk element (used e.g. when a mesh is refined/adapted and new
	// interface nodes appear that need a sensible initial value rather than zero).
	void InterfaceElementBase::interpolate_newly_constructed_additional_dof(const unsigned & lnode,const  unsigned & valindex,const std::string & space)
	{
	   //TODO: Co-dim >=2 interpolation!
	   BulkElementBase *blk =dynamic_cast<BulkElementBase *>(this->Bulk_element_pt);
	   BulkElementBase *father = dynamic_cast<BulkElementBase *>(blk->father_element_pt());
	   if (father)
	   {
		  	  unsigned myvalindex = dynamic_cast<oomph::BoundaryNodeBase *>(this->node_pt(lnode))->index_of_first_value_assigned_by_face_element(valindex);
	        oomph::Vector<double> my_s,s_bulk,sfather;
	        oomph::Node * bulknode=NULL;
	        oomph::Node * mynode=this->node_pt(lnode);
	        for (unsigned int ln=0;ln<blk->nnode();ln++)
	        {
	         if (blk->node_pt(ln)==mynode)
	         {
	           bulknode=blk->node_pt(ln);
	         }
	        }
			  if (!bulknode)
			  {
			    throw_runtime_error("Cannot find bulk node ");
			  }
			  int lblk=blk->get_node_number(bulknode);									        
			  blk->get_nodal_s_in_father(lblk, sfather);
			  oomph::Shape psi;
			  std::vector<pyoomph::BoundaryNode *> src_nodes;
			  std::vector<unsigned> src_val_inds;
			  std::vector<double> weights;
			  if (space=="C1")
			  {
				  psi.resize(father->get_eleminfo()->nnode_of_space[SPACE_INDEX_C1]);
		  		  father->shape_at_s_C1(sfather, psi);				 
		  		  for (unsigned int lf=0;lf<psi.nindex1();lf++) 
		  		  {
		  		   if (abs(psi[lf])>1e-9)
		  		   {
		  		    unsigned fnode_index=father->get_nodal_space_index_to_element_index_map()[SPACE_INDEX_C1][lf];
		  		    pyoomph::BoundaryNode * bn=dynamic_cast<pyoomph::BoundaryNode *>(father->node_pt(fnode_index));
		  		    if (!bn) continue;
		  		    if (!bn->has_additional_dof(valindex)) continue;
		  		    src_nodes.push_back(bn);
		  		    if (!src_nodes.back())	    {  		     throw_runtime_error("Found node is not a boundary node");	    }
		  		    src_val_inds.push_back(src_nodes.back()->index_of_first_value_assigned_by_face_element(valindex));
		  		    weights.push_back(psi[lf]);
		  		   }
		  		  }
			  }
			  else if (space=="C1TB")
			  {
				  psi.resize(father->get_eleminfo()->nnode_of_space[SPACE_INDEX_C1TB]);
		  		  father->shape_at_s_C1TB(sfather, psi);				 
		  		  for (unsigned int lf=0;lf<psi.nindex1();lf++) 
		  		  {
		  		   if (abs(psi[lf])>1e-9)
		  		   {
		  		    unsigned fnode_index=father->get_nodal_space_index_to_element_index_map()[SPACE_INDEX_C1TB][lf];
		  		    pyoomph::BoundaryNode * bn=dynamic_cast<pyoomph::BoundaryNode *>(father->node_pt(fnode_index));
		  		    if (!bn) continue;
		  		    if (!bn->has_additional_dof(valindex)) continue;
		  		    src_nodes.push_back(bn);
		  		    if (!src_nodes.back())	    {  		     throw_runtime_error("Found node is not a boundary node");	    }
		  		    src_val_inds.push_back(src_nodes.back()->index_of_first_value_assigned_by_face_element(valindex));
		  		    weights.push_back(psi[lf]);
		  		   }
		  		  }
			  }			  
           else if (space=="C2")
 			  {
				  psi.resize(father->get_eleminfo()->nnode_of_space[SPACE_INDEX_C2]);
		  		  father->shape_at_s_C2(sfather, psi);				 
		  		  for (unsigned int lf=0;lf<psi.nindex1();lf++) 
		  		  {
		  		   if (abs(psi[lf])>1e-9)
		  		   {
		  		    unsigned fnode_index=father->get_nodal_space_index_to_element_index_map()[SPACE_INDEX_C2][lf];
		  		    pyoomph::BoundaryNode * bn=dynamic_cast<pyoomph::BoundaryNode *>(father->node_pt(fnode_index));
		  		    if (!bn) continue;
		  		    if (!bn->has_additional_dof(valindex)) continue;
		  		    src_nodes.push_back(bn);
		  		    if (!src_nodes.back())	    {  		     throw_runtime_error("Found node is not a boundary node");	    }
		  		    src_val_inds.push_back(src_nodes.back()->index_of_first_value_assigned_by_face_element(valindex));
		  		    weights.push_back(psi[lf]);
		  		   }
		  		  }
			  }			  
			  else if (space=="C2TB")
 			  {
				  psi.resize(father->get_eleminfo()->nnode_of_space[SPACE_INDEX_C2TB]);
		  		  father->shape_at_s_C2TB(sfather, psi);				 
		  		  for (unsigned int lf=0;lf<psi.nindex1();lf++) 
		  		  {
		  		   if (abs(psi[lf])>1e-9)
		  		   {
		  		    unsigned fnode_index=father->get_nodal_space_index_to_element_index_map()[SPACE_INDEX_C2TB][lf];
		  		    pyoomph::BoundaryNode * bn=dynamic_cast<pyoomph::BoundaryNode *>(father->node_pt(fnode_index));
		  		    if (!bn) continue;
		  		    if (!bn->has_additional_dof(valindex)) continue;
		  		    src_nodes.push_back(bn);
		  		    if (!src_nodes.back())	    {  		     throw_runtime_error("Found node is not a boundary node");	    }
		  		    src_val_inds.push_back(src_nodes.back()->index_of_first_value_assigned_by_face_element(valindex));
		  		    weights.push_back(psi[lf]);
		  		   }
		  		  }
			  }					  
			  else 
			  {
					 throw_runtime_error("Cannot interpolate interface fields on space '"+space+"' yet");
			  }		

           if (weights.size())
           {
              double renom=0;
              for (unsigned int i=0;i<weights.size();i++) renom+=weights[i];
              for (unsigned int i=0;i<weights.size();i++) weights[i]/=renom;
              
		        for (unsigned t = 0; t < mynode->ntstorage(); t++)
				  {
					  double val=0;
					  for (unsigned int i=0;i<src_nodes.size();i++)
		           {
		             val+=src_nodes[i]->value_pt(src_val_inds[i])[t]*weights[i];            
		           }			     
					  mynode->set_value(t,myvalindex,val);
				  }
			  }

	        
	   }	
   }
   
	// Called by oomph-lib while finite-differencing an external datum (during Jacobian assembly): re-interpolates
	// this element's hanging-node values so that perturbed external master values are propagated correctly.
	void InterfaceElementBase::update_in_external_fd(const unsigned &)
	{
		this->interpolate_hang_values();
	}

	// Registers 'data' as external data of this element unless it is already accessible some other way
	// (as a node/its variable position, an already-added external datum, or - which should not normally
	// happen - internal data); returns true if it was already present so callers can skip re-adding it.
	bool InterfaceElementBase::add_required_ext_data(oomph::Data *data, bool is_geometric)
	{
		for (unsigned int k = 0; k < this->nnode(); k++)
		{
			if (data == this->node_pt(k))
			{
			//	std::cout << "  ALREADY PART OF THE ELEMENT AT NODE INDEX " << k << std::endl;		
				return true;
			}
			if (data==static_cast<pyoomph::Node *>(this->node_pt(k))->variable_position_pt())
			{
				return true;
			}
		}; // Nodes can be the same
		const std::vector<std::vector<unsigned>> & space_to_elem_index = this->get_nodal_space_index_to_element_index_map();
		for (unsigned int si=0;si<this->get_code_instance()->get_func_table()->num_present_continuous_spaces;si++)
		{
			auto * space_info=this->get_code_instance()->get_func_table()->present_continuous_spaces[si];
			// hangindex, NOT space_index: a hanging scheme is stored per nodal VALUE SLOT, and
			// Node::is_hanging(i) reads Hanging_pt[i+1] out of an array of size nvalue()+1 without any
			// bounds check. Asking a node that carries a single value (e.g. a bulk domain with just one
			// C2 field, whose hangindex is 0) for the C2 SPACE index 1 reads one past the end, and the
			// garbage there is dereferenced as a HangInfo* below. That segfaulted every rebuild of an
			// adapted interior-facet skeleton with 2:1 hanging nodes. hangindex==-1 means "hangs with the
			// dominant space", which is exactly what is_hanging(-1)/hanging_pt(-1) (geometric) answer.
			for (unsigned int j=0;j<eleminfo.nnode_of_space[space_info->space_index];j++)
			{
				auto *nod_pt = static_cast<pyoomph::Node *>(this->node_pt(space_to_elem_index[space_info->space_index][j]));
				if (nod_pt->is_hanging(space_info->hangindex)) // TODO: In principle, it can also hang elsewhere, i.e. on another index!
				{
					oomph::HangInfo *const hang_pt = nod_pt->hanging_pt(space_info->hangindex);
					const unsigned nmaster = hang_pt->nmaster();
					for (unsigned m = 0; m < nmaster; m++)
					{
						auto *const master_nod_pt = static_cast<pyoomph::Node *>(hang_pt->master_node_pt(m));
						if (data==master_nod_pt) return true;
						if (data==master_nod_pt->variable_position_pt()) return true;
					}
				}
			}
		}
		
		for (unsigned int k = 0; k < this->nexternal_data(); k++)
		{
			if (data == this->external_data_pt(k))
			{
			//	std::cout << "  ALREADY ADDED AS EXTERNAL DATA AT INDEX INDEX " << k << std::endl;		
				return true;
			}
		}; // Present as internal data (should not really happen)
		for (unsigned int k = 0; k < this->ninternal_data(); k++)
		{
			if (data == this->internal_data_pt(k))
			{
				std::cout << " DATA ALREADY ADDED AS INTERNAL DATA AT INDEX INDEX " << k << " (this should actually not really happen, please report)" << std::endl;		
				return true;
			}
		}; // External data already added		
		for (unsigned int k = 0; k < this->nnode(); k++)
		{
			if (data == static_cast<pyoomph::Node *>(this->node_pt(k))->variable_position_pt())
			{
			//	std::cout << "  IS ALREADY VARIABLE POSITION AT INDEX " << k << std::endl;		
				return true;
			}
		}; // External data already added
		
		unsigned index = this->add_external_data(data, false);
	//	std::cout << "  ADDING AT INDEX " << index << std::endl;		
		if (index >= external_data_is_geometric.size())
			external_data_is_geometric.resize(index + 1, false);
		external_data_is_geometric[index] = is_geometric;
		return false;
	}

	// Registers the bulk element's base-bulk DG field data as external data of this interface element, so
	// the generated interface code can read (but not directly own) those DG values/fluxes.
	void InterfaceElementBase::add_DG_external_data()
	{
      auto *ft=this->codeinst->get_func_table();
	  BulkElementBase * blk=dynamic_cast<BulkElementBase *>(this->bulk_element_pt());
	  for (unsigned int si=0;si<ft->num_present_dg_spaces;si++)
	  {
		  auto * space_info=ft->present_dg_spaces[si];
		  // Deliberately NOT gated on whether anything requires this space, although a Dirichlet-condition
		  // element (which requires no shape at all) does carry half its dofs this way: the bulk DG data is
		  // addressed POSITIONALLY, as external_data_pt(external_offset_bulk + fieldindex), and
		  // fill_element_info marshals every present DG space unconditionally. Skipping one attachment
		  // shifts every later index and reads past the end of the external data - it segfaults rather than
		  // producing a wrong number. Narrowing this needs the DG marshalling to learn which spaces were
		  // attached, which is a separate change from the external-data split.
		  for (unsigned i=0;i<space_info->numfields_bulk;i++)
		  {
			// DG field values, never positions. The flag has to be recorded here rather than left to
			// the resize() in add_required_external_data: that only FILLS new slots, so a code with DG
			// spaces but no ED0 and no bulk/opposite requirements would leave the vector shorter than
			// nexternal_data() and trip the size check in fill_in_jacobian_from_lagragian_by_fd.
			const unsigned index = this->add_external_data(blk->get_DG_nodal_data(space_info->space_index, i));
			if (index >= external_data_is_geometric.size())
				external_data_is_geometric.resize(index + 1, false);
			external_data_is_geometric[index] = false;
		  }
	  }	  
	}

	// Registers, as external data of this interface element, everything from_elem (the bulk element, or the
	// opposite element/its bulk element) that the generated interface code actually needs according to
	// 'required': nodal positions (if the mesh moves, so their derivatives can be finite-differenced),
	// nodal values for each continuous/DG space whose shape functions are required, and DL/D0 internal data.
	// Hanging nodes are resolved to their master nodes so the true independent dofs are what gets added.
	void InterfaceElementBase::add_required_external_data(JITFuncSpec_RequiredShapes_FiniteElement_t *required, BulkElementBase *from_elem)
	{
		external_data_is_geometric.resize(this->nexternal_data(), false); // Fill with the ED0 fields
		// std::cout << "EX DA " << this->nexternal_data() << std::endl;
		DynamicBulkElementInstance *fcodeinst = from_elem->get_code_instance();
		auto *fft = fcodeinst->get_func_table();		

		if (fft->moving_nodes)
		{
			// Isn't this overkill? For normal psi's we don't use it at all....
			// Should be enough to check for the dx_... and for Pos and normal			
			//if (required->dx_psi_C2TB || required->psi_C2TB || required->dX_psi_C2TB || required->dx_psi_C2 || required->psi_C2 || required->dX_psi_C2 || required->dx_psi_C1 || required->psi_C1TB || required->dx_psi_C1TB || required->dX_psi_C1TB || required->psi_C1 || required->dX_psi_C1 || required->psi_Pos || required->psi_DL || required->dx_psi_DL || required->dX_psi_DL || required->psi_D0) 
			bool require_dx_psi=false;
			for (unsigned int i=0;i<fft->num_present_continuous_spaces;i++)
			{
				auto * space_info=fft->present_continuous_spaces[i];			
				require_dx_psi|=required->continuous_spaces[space_info->space_index].dx_psi;
			}
			// Pos.dx_psi/Pos.dX_psi are part of the condition for the same reason as in
			// fill_hang_info_with_equations: without them, an interface term that only uses gradients of the
			// bulk position test function does not get the bulk nodes' positions registered, so the equation
			// remapping cannot resolve them and the contribution is silently lost.
			if (require_dx_psi ||   required->Pos.psi || required->Pos.dx_psi || required->Pos.dX_psi || required->DL.dx_psi || required->normal || required->elemsize_Eulerian || required->elemsize_Eulerian_cartesian)
			{
				// Add required geometric external data to be finite differenced
				
				for (unsigned int j = 0; j < from_elem->get_eleminfo()->nnode; j++)
				{
					auto *nod_pt = static_cast<pyoomph::Node *>(from_elem->node_pt(j));
					if (nod_pt->is_hanging())
					{						
						oomph::HangInfo *const hang_pt = nod_pt->hanging_pt();
						const unsigned nmaster = hang_pt->nmaster();
						for (unsigned m = 0; m < nmaster; m++)
						{
							auto *const master_nod_pt = static_cast<pyoomph::Node *>(hang_pt->master_node_pt(m));
							this->add_required_ext_data(master_nod_pt->variable_position_pt(), true);
						}
					}
					else
					{						
						this->add_required_ext_data(nod_pt->variable_position_pt(), true);
					}
				}
			}
		}

		int hanging_index = -1;
		const std::vector<std::vector<unsigned>> & from_space_to_elem_index = from_elem->get_nodal_space_index_to_element_index_map();

		for (unsigned int si=0;si<fft->num_present_continuous_spaces;si++)
		{
			auto * space_info=fft->present_continuous_spaces[si];
			if (required->continuous_spaces[space_info->space_index].dx_psi || required->continuous_spaces[space_info->space_index].psi || required->continuous_spaces[space_info->space_index].dX_psi)
			{
				hanging_index = space_info->hangindex;
				for (unsigned int j = 0; j < from_elem->get_eleminfo()->nnode_of_space[space_info->space_index]; j++)
				{
					auto *nod_pt = from_elem->node_pt(from_space_to_elem_index[space_info->space_index][j]);
					if (nod_pt->is_hanging(hanging_index))
					{			
						oomph::HangInfo *const hang_pt = nod_pt->hanging_pt(hanging_index);
						const unsigned nmaster = hang_pt->nmaster();
						for (unsigned m = 0; m < nmaster; m++)
						{
							auto *const master_nod_pt = hang_pt->master_node_pt(m);
							this->add_required_ext_data(master_nod_pt, false);
						}
					}
					else
					{
						this->add_required_ext_data(nod_pt, false);
					}									
				}
			}
		}

		for (unsigned int si=0;si<fft->num_present_dg_spaces;si++)
		{
			auto * space_info=fft->present_dg_spaces[si];
			if (required->continuous_spaces[space_info->space_index].dx_psi || required->continuous_spaces[space_info->space_index].psi || required->continuous_spaces[space_info->space_index].dX_psi)
			{
				for (unsigned int fiDG=0;fiDG<space_info->numfields;fiDG++)
				{
				  this->add_required_ext_data(from_elem->get_DG_nodal_data(space_info->space_index, fiDG),false);
				}			
			}
		}
		

		// std::cout << " AT REQ " << codeinst->get_code()->get_file_name() << " FROM " << fcodeinst->get_code()->get_file_name() << " USE DL " << (required->psi_DL || required->DL.dx_psi || required->DL.dX_psi) << std::endl;
		if (required->DL.psi || required->DL.dx_psi || required->DL.dX_psi)
		{

			for (unsigned int j = 0; j < fft->info_DL.numfields; j++)
			{
				auto *id_pt = from_elem->internal_data_pt(fft->info_DL.internal_offset_new+j);
				this->add_required_ext_data(id_pt, false);
			}
		}

		if (required->D0.psi)
		{
			for (unsigned int j = 0; j < fft->info_D0.numfields; j++)
			{
				auto *id_pt = from_elem->internal_data_pt(fft->info_D0.internal_offset_new + j);
				this->add_required_ext_data(id_pt, false);
			}
		}
	}

	/**
	 * Calculate first (and potentially second derivatives) of the normal calculated via oomph::FaceElement::outer_unit_normal(...)
	 * with respect to moving mesh positions at local coordinate s[elem_dim].
	 *
	 * dnormal_dcoord[i:nodal_dim][l:num_bulk_nodes][j:nodaldim] must return the derivative of the i-th normal coordinate with respect to the j-th position coordinate x^l_j of the l-th node of the bulk element (i.e. the parent element where the interface is attached to)
	 *
	 * if !=NULL, d2normal_dcoord2[i:nodal_dim][l:num_bulk_nodes][j:nodaldim][k:num_bulk_nodes][m:nodaldim] must return the second derivatives of the i-th normal component wrt. x^l_j and x^k_m
	 *
	 * @param s the local coordinate in the element
	 * @param dnormal_dcoord first derivatives with respect to coordinate positions (to be calculated)
	 * @param d2normal_dcoord2 second derivatives with respect to coordinate positions (to be calculated if d2normal_dcoord2!=NULL)
	 */

	void InterfaceElementBase::get_dnormal_dcoords_at_s(const oomph::Vector<double> &s, double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT dnormal_dcoord, double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT d2normal_dcoord2) const
	{   
		
		bool new_vers = dim()!=2; // Fall back to old code for this case

		if (new_vers){

         // Required quantities.
		const unsigned element_dim = dim();
		const unsigned spatial_dim = nodal_dimension();
		const unsigned n_node_bulk = Bulk_element_pt->nnode();
        const unsigned n_node = this->nnode();
		double nlen;
		int nsize = spatial_dim;
		if (element_dim==1) nsize=3;
		oomph::Vector<double> normal(nsize, 0.0); 
        oomph::RankThreeTensor<double> dndxli(nsize, n_node_bulk, spatial_dim, 0.0);;
		oomph::RankFiveTensor<double> d2ndx2li(nsize, n_node_bulk, spatial_dim, n_node_bulk, spatial_dim, 0.0);;
		
        // Initialise final result dnormal_dcoord.
        // dnormal_dcoord[i][l][j][m][k] = dn_i/dx_j^l / norm(n) + n_i * dnorm(n)/dx_j^l. 
        for (unsigned int i = 0; i < spatial_dim; i++)
		{
			for (unsigned l = 0; l < n_node_bulk; l++)
			{
				for (unsigned int j = 0; j < spatial_dim; j++)
				{
					dnormal_dcoord[i][l][j] = 0.0;
				}
			}
		}

		if (d2normal_dcoord2)
		{ // Initialize if required.
			for (unsigned int i = 0; i < spatial_dim; i++)
			{
				for (unsigned int l = 0; l < n_node_bulk; l++)
				{
					for (unsigned int j = 0; j < spatial_dim; j++)
					{
						for (unsigned int m = 0; m < n_node_bulk; m++)
						{
							for (unsigned int k = 0; k < spatial_dim; k++)
							{
								d2normal_dcoord2[i][l][j][m][k] = 0.0;
							}
						}
					}
				}
			}
		}


        // To obtain dnormal_dcoord, we first need to find dn_i/dx_j^l, which changes 
        // according to the spatial dimension. dnorm(n)/dx_j^l will be a function of 
        // dn_i/dx_j^l, but it does not explicitly depend on the spatial dimension.

		if (element_dim==0)
		{   
            // Required quantities
			oomph::Vector<double> s_bulk(1);
			this->get_local_coordinate_in_bulk(s, s_bulk);
			oomph::Shape psi(n_node_bulk);
			oomph::DShape dpsids(n_node_bulk, 1);
            oomph::DenseMatrix<double> interpolated_dxds(1, spatial_dim, 0.0);
			Bulk_element_pt->dshape_local(s_bulk, psi, dpsids);
			
            for (unsigned l = 0; l < n_node_bulk; l++)
			{
				for (unsigned i = 0; i < spatial_dim; i++)
				{   
                    // In 1D, the normal is simply the tangent to the surface.
					interpolated_dxds(0, i) += Bulk_element_pt->nodal_position_gen(l, 0, i) * dpsids(l, 0);
                    normal[i] = interpolated_dxds(0,i);
				}
			}

            for (unsigned int i = 0; i < spatial_dim; i++)
			{
				for (unsigned l = 0; l < n_node_bulk; l++)
				{
					for (unsigned int j = 0; j < spatial_dim; j++)
					{
                        // First order derivative of non normalised normal: dnorm(n)/dx_k^l
                        dndxli(i, l, j) = this->normal_sign() * (i == j ? 1 : 0) * dpsids(l, 0);

                        if (d2normal_dcoord2)
							{
							for (unsigned m = 0; m < n_node_bulk; m++)
								{
								for (unsigned int k = 0; k < spatial_dim; k++)
										{
                                            // Second order derivative for non normalized norm.
                                            // In this case, it is 0 since there is no x-dependency on any term.
										    d2ndx2li(i, l, j, m, k) = 0.0;
                                        }
                                }
                            }
                    }
                }

            }
		}
		else if (element_dim==1)
		{	
			// Required quantities
			oomph::Vector<double> s_bulk(2);
			this->get_local_coordinate_in_bulk(s, s_bulk);
			oomph::Shape psi(n_node_bulk);
			oomph::DShape dpsids(n_node_bulk, 2);
            oomph::DenseMatrix<double> interpolated_dxds(2, spatial_dim, 0.0);
            oomph::RankFourTensor<double> dinterpolated_dxds(2, spatial_dim, n_node_bulk, spatial_dim, 0.0);
			Bulk_element_pt->dshape_local(s_bulk, psi, dpsids);

            // For later calculations, tangent of bulk.
			for (unsigned l = 0; l < n_node_bulk; l++)
			{
				for (unsigned j = 0; j < 2; j++)
				{
					for (unsigned i = 0; i < spatial_dim; i++)
					{
						interpolated_dxds(j, i) += Bulk_element_pt->nodal_position_gen(l, 0, i) * dpsids(l, j);
					}
				}
			}

            // Derivative of tangent of bulk wrt coordinate.
			for (unsigned int l = 0; l < n_node_bulk; l++)
			{
				for (unsigned int k = 0; k < spatial_dim; k++)
				{
					for (unsigned j = 0; j < 2; j++)
					{
						for (unsigned i = 0; i < spatial_dim; i++)
						{
							dinterpolated_dxds(j, i, l, k) += dpsids(l, j) * (k == i ? 1 : 0);
						}
					}
				}
			}

            // Initialise tangent, interior tangent to line vectors.
			oomph::Vector<double> t(3, 0.0), T(3, 0.0);
            // Initialise derivative of bulk local coordinate wrt line local coordinate.
			oomph::DenseMatrix<double> dsbulk_dsface(2, 1, 0.0);
            // Initialise interior direction to obtain normal.
			unsigned interior_direction = 0;
            // Obtain interior direction and second vector on plain to obtain 
            // the normal through cross product.
			this->get_ds_bulk_ds_face(s, dsbulk_dsface, interior_direction);

            // Tangent and interior tangent.
			for (unsigned i = 0; i < spatial_dim; i++)
			{
				t[i] = interpolated_dxds(0, i) * dsbulk_dsface(0, 0) + interpolated_dxds(1, i) * dsbulk_dsface(1, 0);
				T[i] = interpolated_dxds(interior_direction, i);
			}


			for (unsigned i = 0; i < spatial_dim; i++)
			{
				for (unsigned int p = 0; p < spatial_dim; p++)
				{   
                    // Calculate normal by the cross product t x t x T.
					normal[i] += this->normal_sign() * (t[p] * T[p] * t[i] - t[p] * t[p] * T[i]); // bac-cab rule
					for (unsigned int l = 0; l < n_node_bulk; l++)
					{

						for (unsigned int j = 0; j < spatial_dim; j++)
						{	

							// Derivatives of t_i and t_p with respect to x_j^l. t_p is need for an additional loop within the calculations.
							double dti_jl = dsbulk_dsface(0, 0) * dinterpolated_dxds(0, i, l, j) + dsbulk_dsface(1, 0) * dinterpolated_dxds(1, i, l, j);
							double dtp_jl = dsbulk_dsface(0, 0) * dinterpolated_dxds(0, p, l, j) + dsbulk_dsface(1, 0) * dinterpolated_dxds(1, p, l, j);
							double dTi_jl = dinterpolated_dxds(interior_direction, i, l, j);
							double dTp_jl = dinterpolated_dxds(interior_direction, p, l, j);
							
							// Derivative of n_i wrt x_j^l
							dndxli(i, l, j) += this->normal_sign() * (dtp_jl * T[p] * t[i] + t[p] * dTp_jl * t[i] + t[p] * T[p] * dti_jl - 2 * dtp_jl * t[p] * T[i] - t[p] * t[p] * dTi_jl); // bac-cab rule


							// Second derivative dx_m^p(dndxli). 
							// Note that dx_m^p(dti) = dx_m^p(dTi) = 0, 
							// since the term dinterpolated_dxds() is independent on x.
							if (d2normal_dcoord2) {
					
								for (unsigned int m = 0; m < n_node_bulk; m++){

									for (unsigned int k = 0; k < spatial_dim; k++){

										// Derivatives of t_i and t_j with respect to x_m^p. 
										double dti_km = dsbulk_dsface(0, 0) * dinterpolated_dxds(0, i, m, k) + dsbulk_dsface(1, 0) * dinterpolated_dxds(1, i, m, k);
										double dtp_km = dsbulk_dsface(0, 0) * dinterpolated_dxds(0, p, m, k) + dsbulk_dsface(1, 0) * dinterpolated_dxds(1, p, m, k);
										double dTi_km = dinterpolated_dxds(interior_direction, i, m, k);
										double dTp_km = dinterpolated_dxds(interior_direction, p, m, k);

										// Second order derivative for non normalized norm.
										d2ndx2li(i, l, j, m, k) += this->normal_sign() * (dtp_jl * dTp_km * t[i] + dtp_jl * T[p] * dti_km + dtp_km * dTp_jl * t[i] + t[p] * dTp_jl * dti_km + dtp_km * T[p] * dti_jl + t[p] * dTp_km * dti_jl - 2 * (dtp_jl * dtp_km * T[i] + dtp_jl * t[p] * dTi_km + dtp_km * t[p] * dTi_jl));
									}

								}

							}

						}
					}
				}
			}

		}

		else
		{
            // Required quantities.
			oomph::Shape psi(n_node);
			oomph::DShape dpsids(n_node, 2);
            oomph::DenseMatrix<double> interpolated_dxds(2, spatial_dim, 0.0);
            oomph::RankFourTensor<double> dinterpolated_dxds(2, spatial_dim, n_node_bulk, spatial_dim, 0.0);
			this->dshape_local(s, psi, dpsids);

			// Tangents depend on the interface only.
			for (unsigned l = 0; l < n_node; l++)
			{
				for (unsigned j = 0; j < 2; j++)
				{
					for (unsigned i = 0; i < spatial_dim; i++)
					{
						interpolated_dxds(j,i) += this->nodal_position_gen(l, 0, i) * dpsids(l, j);
					}
				}
			}

            // Get epsilon function to use for cross product.
			oomph::RankThreeTensor<double> EpsilonIJK(3, 3, 3, 0.0);
			EpsilonIJK(0, 1, 2) = 1;
			EpsilonIJK(0, 2, 1) = -1;
			EpsilonIJK(1, 2, 0) = 1;
			EpsilonIJK(1, 0, 2) = -1;
			EpsilonIJK(2, 0, 1) = 1;
			EpsilonIJK(2, 1, 0) = -1;

            // Normal calculation.
			for (unsigned int i = 0; i < spatial_dim; i++)
			{
				for (unsigned int j = 0; j < spatial_dim; j++)
				{
					for (unsigned int k = 0; k < spatial_dim; k++)
					{
						normal[i] += this->normal_sign() * EpsilonIJK(i, j, k) * interpolated_dxds(0,j) * interpolated_dxds(1,k);
					}
				}
			}

            // Derivative of bulk tangent wrt coordinate
			for (unsigned int l = 0; l < n_node_bulk; l++)
			{
				for (unsigned int k = 0; k < spatial_dim; k++)
				{
					for (unsigned j = 0; j < 2; j++)
					{
						for (unsigned i = 0; i < spatial_dim; i++)
						{
							dinterpolated_dxds(j, i, l, k) += dpsids(l, j) * (k == i ? 1 : 0);
						}
					}
				}
			}   

            
			for (unsigned int i = 0; i < 3; i++)
			{
				for (unsigned int l = 0; l < n_node; l++)
				{
					for (unsigned int j = 0; j < spatial_dim; j++)
					{
						for (unsigned int p = 0; p < spatial_dim; p++)
						{
							for (unsigned int q = 0; q < spatial_dim; q++)
							{   
                                // Derivative of n_i wrt x_j^l
                                dndxli(i, l, j) += this->normal_sign() * EpsilonIJK(i, j, q) * (dinterpolated_dxds(0, j, l, p) * interpolated_dxds(1,q) + interpolated_dxds(0,p) * dinterpolated_dxds(1, j, l, q));
							
                                if (d2normal_dcoord2)
                                {
                                    for (unsigned int m = 0; m < n_node_bulk; m++)
                                        {
                                            for (unsigned int k = 0; k < spatial_dim; k++)
                                            {
                                                // Second order derivative for non normalized norm.    
                                                d2ndx2li(i, l, j, m, k) += this->normal_sign() * EpsilonIJK(i, j, q) * (dinterpolated_dxds(0, j, l, p) * dinterpolated_dxds(1, q, l, k) + dinterpolated_dxds(0, p, l, k) * dinterpolated_dxds(1, j, l, q));
                                            }
                                        }
                                }
                            }
						}
					}
				}
			}

		}
	

        
        //=========================================================================//
        // Here starts the common calculations, independent of the element's dimension.
        
		// Norm of normal vector.
		nlen = 0.0;
        for (unsigned int i = 0; i < spatial_dim; i++)
            nlen += normal[i] * normal[i];
        nlen = sqrt(nlen);

        // Loop through all dimensions of normal vector.
		for (unsigned i = 0; i < spatial_dim; i++)
		{	
			// Loop through all nodes in element.
			for (unsigned int l = 0; l < n_node_bulk; l++)
			{	
				// Loop through all dimensions of coordinates to fill up shape info.
				for (unsigned int j = 0; j < spatial_dim; j++)
				{	
					
                    double crosssum_lj = 0.0;
					// Cross sum.
					for (unsigned int p = 0; p < spatial_dim; p++)
						{crosssum_lj += normal[p] * dndxli(p, l, j);}

					// First order derivative of normalised normal.
					dnormal_dcoord[i][l][j] = dndxli(i, l, j) / nlen - normal[i] * crosssum_lj / (nlen * nlen * nlen);

					if (d2normal_dcoord2)
					{   
						for (unsigned int m = 0; m < n_node_bulk; m++)
						{
							for (unsigned int k = 0; k < spatial_dim; k++){
							
							double crosssum_mk = 0.0;
                            double dcrosssum = 0.0;
                            double d2crosssum = 0.0;

							// Other quantities for calculations
							for (unsigned int p = 0; p < spatial_dim; p++)
							{crosssum_mk += normal[p] * dndxli(p, m, k);
                            dcrosssum += dndxli(p, l, j) * dndxli(p, m, k);
							d2crosssum += normal[p] * d2ndx2li(p, l, j, m, k);}

							
							// Second order derivative of normalised normal.
							d2normal_dcoord2[i][l][j][m][k] = d2ndx2li(i,l,j,m,k) / nlen + (normal[i] * (3 / (nlen * nlen) * crosssum_lj * crosssum_mk - dcrosssum - d2crosssum) - crosssum_mk * dndxli(i,l,j) - dndxli(i,m,k) * crosssum_lj) / (nlen * nlen * nlen);
								}
							}
						}
					}
				}
			}
			
			/*
         if (d2normal_dcoord2)
			{   
				//Check whether it is symmetric //TODO: Remove
				// Also check the FD case
				double d2nodal_FD[spatial_dim][n_node_bulk][spatial_dim][n_node_bulk][spatial_dim];
				double *** dnormal_dcoord0;//[spatial_dim][n_node_bulk][spatial_dim];
				double *** dnormal_dcoord1;//[spatial_dim][n_node_bulk][spatial_dim];				
				dnormal_dcoord0=(double***)std::calloc(spatial_dim,sizeof(double**)); //TODO: Careful: Not free'd!
				dnormal_dcoord1=(double***)std::calloc(spatial_dim,sizeof(double**));				
				for (unsigned i = 0; i < spatial_dim; i++)				
				{
				   dnormal_dcoord0[i]=(double**)std::calloc(n_node_bulk,sizeof(double*));
				   dnormal_dcoord1[i]=(double**)std::calloc(n_node_bulk,sizeof(double*));				
					for (unsigned int l = 0; l < n_node_bulk; l++)
					{
				     dnormal_dcoord0[i][l]=(double*)std::calloc(spatial_dim,sizeof(double));
				     dnormal_dcoord1[i][l]=(double*)std::calloc(spatial_dim,sizeof(double));					  
					}
				}
				this->get_dnormal_dcoords_at_s(s, dnormal_dcoord0, NULL);				
				double FD_eps=1e-8;
				for (unsigned i = 0; i < spatial_dim; i++)
				{				
					for (unsigned int l = 0; l < n_node_bulk; l++)
					{						
						for (unsigned int j = 0; j < spatial_dim; j++)
						{	
							for (unsigned int lp = 0; lp < n_node_bulk; lp++)
							{						
								for (unsigned int jp = 0; jp < spatial_dim; jp++)
								{	
								   double old=static_cast<pyoomph::Node*>(Bulk_element_pt->node_pt(lp))->variable_position_pt()->value(jp);
								   static_cast<pyoomph::Node*>(Bulk_element_pt->node_pt(lp))->variable_position_pt()->set_value(jp,old+FD_eps);								   
               				this->get_dnormal_dcoords_at_s(s, dnormal_dcoord1, NULL);	
               				d2nodal_FD[i][l][j][lp][jp]= (dnormal_dcoord1[i][l][j]-dnormal_dcoord0[i][l][j])/FD_eps;
								   static_cast<pyoomph::Node*>(Bulk_element_pt->node_pt(lp))->variable_position_pt()->set_value(jp,old);								                  				
								}
							}
						}
					}
				}				
				for (unsigned i = 0; i < spatial_dim; i++)
				{				
					for (unsigned int l = 0; l < n_node_bulk; l++)
					{						
						for (unsigned int j = 0; j < spatial_dim; j++)
						{	
							for (unsigned int lp = 0; lp < n_node_bulk; lp++)
							{						
								for (unsigned int jp = 0; jp < spatial_dim; jp++)
								{	
								  double val1=d2normal_dcoord2[i][l][j][lp][jp];
								  double val2=d2normal_dcoord2[i][lp][jp][l][j];							  
								  if (std::fabs(val1-val2)>1e-6)
								  {
									std::cout << "NORMAL SECOND DERIV NOT SYMMETRIC! : "<<i << "  "<< l << "  "<< j  << "  "<< lp  << "  "<< jp << " : " << val1 << " and " << val2 << std::endl;
								  }
								  double val3=d2nodal_FD[i][l][j][lp][jp];
								  if (std::fabs(val1-val3)>1e-8)
								  {
									std::cout << "NORMAL SECOND DERIV NOT MATCHING WITH FD! : "<<i << "  "<< l << "  "<< j  << "  "<< lp  << "  "<< jp << " : " << val1 << " and " << val3 << std::endl;
								  }								  
								}
							}					
						
						}
					}
				}
			}
			*/
		} 
		
		//===============================================================================================================================================//
		//===============================================================================================================================================//
		//===============================================================================================================================================//
		//===============================================================================================================================================//
		//===============================================================Old code========================================================================//

		
		else 
		
		
		//===============================================================================================================================================//
		//===============================================================================================================================================//
		//===============================================================================================================================================//
		//===============================================================================================================================================//
		//===============================================================================================================================================//
		
		
		{
		const unsigned element_dim = dim();
		const unsigned spatial_dim = nodal_dimension();
		const unsigned n_node_bulk = Bulk_element_pt->nnode();
		for (unsigned int i = 0; i < spatial_dim; i++)
		{
			for (unsigned l = 0; l < n_node_bulk; l++)
			{
				for (unsigned int j = 0; j < spatial_dim; j++)
				{
					dnormal_dcoord[i][l][j] = 0.0;
				}
			}
		}

		if (d2normal_dcoord2)
		{ // Initialize if required
			for (unsigned int i = 0; i < spatial_dim; i++)
			{
				for (unsigned int l = 0; l < n_node_bulk; l++)
				{
					for (unsigned int j = 0; j < spatial_dim; j++)
					{
						for (unsigned int k = 0; k < n_node_bulk; k++)
						{
							for (unsigned int m = 0; m < spatial_dim; m++)
							{
								d2normal_dcoord2[i][l][j][k][m] = 0.0;
							}
						}
					}
				}
			}
		}

		switch (element_dim)
		{
		case 0:
		{
			oomph::Vector<double> s_bulk(1);
			this->get_local_coordinate_in_bulk(s, s_bulk);
			oomph::Shape psi(n_node_bulk);
			oomph::DShape dpsids(n_node_bulk, 1);
			Bulk_element_pt->dshape_local(s_bulk, psi, dpsids);
			oomph::DenseMatrix<double> interpolated_dxds(1, spatial_dim, 0.0);
			for (unsigned l = 0; l < n_node_bulk; l++)
			{
				for (unsigned i = 0; i < spatial_dim; i++)
				{
					interpolated_dxds(0, i) += Bulk_element_pt->nodal_position_gen(l, 0, i) * dpsids(l, 0);
				}
			}
			double l = 0.0;
			for (unsigned int i = 0; i < spatial_dim; i++)
				l += interpolated_dxds(0, i) * interpolated_dxds(0, i);
			l = sqrt(l); // Normal length
			double denom = this->normal_sign() / (l * l * l);
			for (unsigned int i = 0; i < spatial_dim; i++)
			{
				for (unsigned coord_node = 0; coord_node < n_node_bulk; coord_node++)
				{
					for (unsigned int j = 0; j < spatial_dim; j++)
					{
						dnormal_dcoord[i][coord_node][j] = denom * (l * l * (i == j ? 1 : 0) - interpolated_dxds(0, i) * interpolated_dxds(0, j)) * dpsids(coord_node, 0);
					}
				}
			}

			if (d2normal_dcoord2)
			{
				throw_runtime_error("Implement second order moving mesh coordinate derivatives of the normal here");
			}
		}
		break;

		case 1:
		{

			oomph::Vector<double> s_bulk(2);
			this->get_local_coordinate_in_bulk(s, s_bulk);
			oomph::Shape psi(n_node_bulk);
			oomph::DShape dpsids(n_node_bulk, 2);
			Bulk_element_pt->dshape_local(s_bulk, psi, dpsids);
			oomph::DenseMatrix<double> interpolated_dxds(2, spatial_dim, 0.0);

			for (unsigned l = 0; l < n_node_bulk; l++)
			{
				for (unsigned j = 0; j < 2; j++)
				{
					for (unsigned i = 0; i < spatial_dim; i++)
					{
						interpolated_dxds(j, i) += Bulk_element_pt->nodal_position_gen(l, 0, i) * dpsids(l, j);
					}
				}
			}

			oomph::RankFourTensor<double> dinterpolated_dxds(2, spatial_dim, n_node_bulk, spatial_dim, 0.0);
			for (unsigned int xl = 0; xl < n_node_bulk; xl++)
			{
				for (unsigned int xi = 0; xi < spatial_dim; xi++)
				{
					for (unsigned j = 0; j < 2; j++)
					{
						for (unsigned i = 0; i < spatial_dim; i++)
						{
							dinterpolated_dxds(j, i, xl, xi) += dpsids(xl, j) * (xi == i ? 1 : 0);
						}
					}
				}
			}

			oomph::Vector<double> t(3, 0.0), T(3, 0.0), normal(3, 0.0);
			oomph::DenseMatrix<double> dsbulk_dsface(2, 1, 0.0);
			unsigned interior_direction = 0;
			this->get_ds_bulk_ds_face(s, dsbulk_dsface, interior_direction);
			oomph::RankThreeTensor<double> dndxli(3, n_node_bulk, spatial_dim, 0.0);
			for (unsigned i = 0; i < spatial_dim; i++)
			{
				// d_interpolated_dxds_dX_km(j,i)=dpsids(k,j) if j==i
				t[i] = interpolated_dxds(0, i) * dsbulk_dsface(0, 0) + interpolated_dxds(1, i) * dsbulk_dsface(1, 0);
				T[i] = interpolated_dxds(interior_direction, i);
			}
			for (unsigned i = 0; i < spatial_dim; i++)
			{
				for (unsigned int j = 0; j < spatial_dim; j++)
				{
					normal[i] += this->normal_sign() * (t[j] * T[j] * t[i] - t[j] * t[j] * T[i]); // bac-cab rule
					for (unsigned int l = 0; l < n_node_bulk; l++)
					{
						for (unsigned int k = 0; k < spatial_dim; k++)
						{
							double dti = dsbulk_dsface(0, 0) * dinterpolated_dxds(0, i, l, k) + dsbulk_dsface(1, 0) * dinterpolated_dxds(1, i, l, k);
							double dtj = dsbulk_dsface(0, 0) * dinterpolated_dxds(0, j, l, k) + dsbulk_dsface(1, 0) * dinterpolated_dxds(1, j, l, k);
							double dTi = dinterpolated_dxds(interior_direction, i, l, k);
							double dTj = dinterpolated_dxds(interior_direction, j, l, k);
							//           std::cout << "dTi("<<i<<","<<l<<","<<k<<")= " << dTi << " vs " << fd_test << std::endl;
							//           if (fabs(dTi-fd_test)>1e-2) throw_runtime_error("Something is wrong");
							dndxli(i, l, k) += this->normal_sign() * (dtj * T[j] * t[i] + t[j] * dTj * t[i] + t[j] * T[j] * dti - dtj * t[j] * T[i] - t[j] * dtj * T[i] - t[j] * t[j] * dTi); // bac-cab rule
						}
					}
				}
			}
			double nleng = 0.0;
			for (unsigned int i = 0; i < spatial_dim; i++)
				nleng += normal[i] * normal[i];
			nleng = sqrt(nleng);
			for (unsigned i = 0; i < spatial_dim; i++)
			{
				for (unsigned int l = 0; l < n_node_bulk; l++)
				{
					for (unsigned int k = 0; k < spatial_dim; k++)
					{
						double crosssum = 0.0;
						for (unsigned int j = 0; j < spatial_dim; j++)
							crosssum += normal[j] * dndxli(j, l, k);
						dnormal_dcoord[i][l][k] = dndxli(i, l, k) / nleng - normal[i] / (nleng * nleng * nleng) * crosssum;
					}
				}
			}

			if (d2normal_dcoord2)
			{
				throw_runtime_error("Implement second order moving mesh coordinate derivatives of the normal here");
			}
		}

		break;

		case 2:
		{

			const unsigned n_node = this->nnode();

			oomph::Shape psi(n_node);
			oomph::DShape dpsids(n_node, 2);
			this->dshape_local(s, psi, dpsids);
			oomph::Vector<oomph::Vector<double>> interpolated_dxds(2, oomph::Vector<double>(3, 0));
			oomph::RankFourTensor<double> dinterpolated_dxds(2, spatial_dim, n_node, spatial_dim, 0.0);

			// Tangents depend on the interface only
			for (unsigned l = 0; l < n_node; l++)
			{
				for (unsigned j = 0; j < 2; j++)
				{
					for (unsigned i = 0; i < 3; i++)
					{
						interpolated_dxds[j][i] += this->nodal_position_gen(l, 0, i) * dpsids(l, j);
					}
				}
			}

			oomph::RankThreeTensor<double> EpsilonIJK(3, 3, 3, 0.0);
			EpsilonIJK(0, 1, 2) = 1;
			EpsilonIJK(0, 2, 1) = -1;
			EpsilonIJK(1, 2, 0) = 1;
			EpsilonIJK(1, 0, 2) = -1;
			EpsilonIJK(2, 0, 1) = 1;
			EpsilonIJK(2, 1, 0) = -1;

			oomph::Vector<double> normal(3, 0.0); // Non-normalized normal
			for (unsigned int i = 0; i < 3; i++)
			{
				for (unsigned int j = 0; j < 3; j++)
				{
					for (unsigned int k = 0; k < 3; k++)
					{
						normal[i] += this->normal_sign() * EpsilonIJK(i, j, k) * interpolated_dxds[0][j] * interpolated_dxds[1][k];
					}
				}
			}

			for (unsigned int xl = 0; xl < n_node; xl++)
			{
				for (unsigned int xi = 0; xi < 3; xi++)
				{
					for (unsigned j = 0; j < 2; j++)
					{
						for (unsigned i = 0; i < 3; i++)
						{
							dinterpolated_dxds(j, i, xl, xi) += dpsids(xl, j) * (xi == i ? 1 : 0);
						}
					}
				}
			}

			oomph::RankThreeTensor<double> dndxlm(3, n_node, 3, 0.0);
			for (unsigned int i = 0; i < 3; i++)
			{
				for (unsigned int l = 0; l < n_node; l++)
				{
					for (unsigned int m = 0; m < 3; m++)
					{
						for (unsigned int j = 0; j < 3; j++)
						{
							for (unsigned int k = 0; k < 3; k++)
							{
								dndxlm(i, l, m) += this->normal_sign() * EpsilonIJK(i, j, k) * (dinterpolated_dxds(0, m, l, j) * interpolated_dxds[1][k] + interpolated_dxds[0][j] * dinterpolated_dxds(1, m, l, k));
							}
						}
					}
				}
			}

			double nleng = 0.0;
			for (unsigned int i = 0; i < 3; i++)
				nleng += normal[i] * normal[i];
			nleng = sqrt(nleng);
			// However, since in 2d cases, the normal might depend on the pure bulk positions, we have to calc the derivatives for the bulk nodes, although may of them are zero
			for (unsigned i = 0; i < 3; i++)
			{
				for (unsigned int l = 0; l < n_node; l++)
				{
					unsigned l_bulk = this->bulk_node_number(l);
					for (unsigned int k = 0; k < 3; k++)
					{
						double crosssum = 0.0;
						for (unsigned int j = 0; j < 3; j++)
							crosssum += normal[j] * dndxlm(j, l, k);
						dnormal_dcoord[i][l_bulk][k] = dndxlm(i, l, k) / nleng - normal[i] / (nleng * nleng * nleng) * crosssum;
					}
				}
			}

			if (d2normal_dcoord2)
			{
				throw_runtime_error("Implement second order moving mesh coordinate derivatives of the normal here");
			}
		}
		break;
		}
		}
	}



	// After letting the base class finite-difference the Lagrangian (solid-mechanics) contributions from this
	// element's own nodal positions, additionally finite-differences the Jacobian columns belonging to this
	// element's *external* geometric data (e.g. bulk/opposite-element nodal positions registered via
	// add_required_ext_data), since those degrees of freedom are not covered by the base class's own nodes.
	void InterfaceElementBase::fill_in_jacobian_from_lagragian_by_fd(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian)
	{
		HangInterpPassSuspension __no_pass; // see BulkElementBase::fill_in_jacobian_from_nodal_by_fd
		BulkElementBase::fill_in_jacobian_from_lagragian_by_fd(residuals, jacobian);
		const unsigned n_node = this->nnode();
		if (n_node == 0)
		{
			return;
		}

		//  const unsigned n_position_type = this->nnodal_position_type();
		//  const unsigned nodal_dim = this->nodal_dimension();
		const unsigned n_dof = this->ndof();
		oomph::Vector<double> newres(n_dof);
		const double fd_step = this->Default_fd_jacobian_step;
		int local_unknown = 0;

		if (this->nexternal_data() > external_data_is_geometric.size())
		{
			throw_runtime_error("Something wrong here: " + std::to_string(this->nexternal_data()) + " external data vs " + std::to_string(external_data_is_geometric.size()));
		}
		for (unsigned int ed = 0; ed < this->nexternal_data(); ed++)
		{
			// Only the GEOMETRIC external data (the bulk/opposite nodes' variable_position_pt, added
			// with is_geometric=true). This routine exists to patch in the position columns the
			// generated code did not produce because fd_position_jacobian is set; the value columns of
			// the attached elements come out of the generated code analytically, through the equation
			// remapping, so finite-differencing them here only overwrote correct entries with less
			// accurate ones.
			if (!external_data_is_geometric[ed])
				continue;
			oomph::Data *data = this->external_data_pt(ed);
			for (unsigned int i = 0; i < data->nvalue(); i++)
			{
				local_unknown = this->external_local_eqn(ed, i);
				if (local_unknown >= 0)
				{
					double *const value_pt = data->value_pt(i);
					const double old_var = *value_pt;
					*value_pt += fd_step;
					get_residuals(newres);
					for (unsigned m = 0; m < n_dof; m++)
					{
						jacobian(m, local_unknown) = (newres[m] - residuals[m]) / fd_step;
					}
					*value_pt = old_var;
				}
			}
		}
	}

	
	// The following methods (fill_shape_info_at_s, prepare_shape_buffer_for_integration,
	// set_remaining_shapes_appropriately, fill_hang_info_with_equations) all follow the same recursive pattern:
	// after doing this element's own part (via the BulkElementBase implementation), they additionally recurse
	// into whichever of bulk_element_pt()/opposite_side (and their own bulk elements, one level deeper) are
	// actually required by the generated code (required.bulk_shapes / required.opposite_shapes), filling the
	// corresponding nested shape_info->bulk_shapeinfo / shape_info->opposite_shapeinfo sub-structures so the
	// JIT code can evaluate shape functions/derivatives on those attached elements too.

	// Evaluate this element's own shape functions/Jacobian at local coordinate s, then recurse into the
	// required bulk/opposite (and their bulk) elements' shape info, evaluated at the corresponding local coordinates.
	double InterfaceElementBase::fill_shape_info_at_s(const oomph::Vector<double> &s, const unsigned int &index, const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, double &JLagr, unsigned int flag, oomph::DenseMatrix<double> *dxds,unsigned history_index) const
	{
		double JEulerian=BulkElementBase::fill_shape_info_at_s(s, index, required, shape_info, JLagr, flag, dxds,history_index);

		if (history_index>0)
		{
			return JEulerian; // Make it simple here
		}
		if (required.bulk_shapes)
		{
			oomph::Vector<double> sbulk = this->local_coordinate_in_bulk(s);
			double JLagrBulk;
			dynamic_cast<BulkElementBase *>(this->bulk_element_pt())->fill_shape_info_at_s(sbulk, index, *(required.bulk_shapes), shape_info->bulk_shapeinfo, JLagrBulk, flag, NULL, history_index);
			if (required.bulk_shapes->bulk_shapes)
			{
				InterfaceElementBase *bulk_as_inter = dynamic_cast<InterfaceElementBase *>(this->bulk_element_pt());
				oomph::Vector<double> sbulkbulk = bulk_as_inter->local_coordinate_in_bulk(sbulk);
				double JLagrBulkBulk;
				dynamic_cast<BulkElementBase *>(bulk_as_inter->bulk_element_pt())->fill_shape_info_at_s(sbulkbulk, index, *(required.bulk_shapes->bulk_shapes), shape_info->bulk_shapeinfo->bulk_shapeinfo, JLagrBulkBulk, flag, NULL, history_index);
			}
		}
		if (required.opposite_shapes)
		{
			if (!opposite_side)
			{
				throw_runtime_error("The interface element requires the opposite side to be set!");
			}
			oomph::Vector<double> sopp = this->local_coordinate_in_opposite_side(s);
			double JLagrOpp;
			opposite_side->fill_shape_info_at_s(sopp, index, *(required.opposite_shapes), shape_info->opposite_shapeinfo, JLagrOpp, flag, NULL, history_index);
			if (required.opposite_shapes->bulk_shapes)
			{
				oomph::Vector<double> sopp_blk = opposite_side->local_coordinate_in_bulk(sopp);
				double JLagrOppBlk;
				// std::cout << "FILLING OPPBLK HERE " << index << std::endl;
				dynamic_cast<BulkElementBase *>(opposite_side->bulk_element_pt())->fill_shape_info_at_s(sopp_blk, index, *(required.opposite_shapes->bulk_shapes), shape_info->opposite_shapeinfo->bulk_shapeinfo, JLagrOppBlk, flag, NULL, history_index);
			}
		}

		return this->J_eulerian(s); // TODO: This likely can be just set to JEulerian from above
	}

	// Ensures hanging-node values on this element and (if required) the attached bulk/opposite elements are
	// up to date before integration, then delegates the rest to the base class.
	void InterfaceElementBase::prepare_shape_buffer_for_integration(const JITFuncSpec_RequiredShapes_FiniteElement_t &required_shapes, unsigned int flag)
	{
		{
		// Timed in its own slot: these are re-interpolations of NEIGHBOURING elements, i.e. work that
		// each of those elements also does when it is assembled itself.
		HangFillTimeScope __hftime(HANGFILL_SLOT_NEIGHBOUR);
		if (required_shapes.bulk_shapes)
		{
			dynamic_cast<BulkElementBase *>(this->bulk_element_pt())->interpolate_hang_values(); // TODO: This might be put somewhere else
			if (required_shapes.bulk_shapes->bulk_shapes)
			{
				dynamic_cast<BulkElementBase *>(dynamic_cast<InterfaceElementBase *>(this->bulk_element_pt())->bulk_element_pt())->interpolate_hang_values(); // TODO: This might be put somewhere else
			}
		}
		if (required_shapes.opposite_shapes)
		{
			if (!opposite_side)
			{
				throw_runtime_error("The interface element requires the opposite site to be set!");
			}
			this->opposite_side->interpolate_hang_values(); // TODO: This might be put somewhere else
			if (required_shapes.opposite_shapes->bulk_shapes)
			{
				dynamic_cast<BulkElementBase *>(this->opposite_side->bulk_element_pt())->interpolate_hang_values(); // TODO: This might be put somewhere else
			}
		}
		}

		BulkElementBase::prepare_shape_buffer_for_integration(required_shapes, flag);
	}

	// See the comment above fill_shape_info_at_s for the general recursive bulk/opposite delegation pattern.
	void InterfaceElementBase::set_remaining_shapes_appropriately(JITShapeInfo_t *shape_info, const JITFuncSpec_RequiredShapes_FiniteElement_t &required_shapes)
	{
		BulkElementBase::set_remaining_shapes_appropriately(shape_info, required_shapes);
		if (required_shapes.bulk_shapes)
		{
			dynamic_cast<BulkElementBase *>(this->bulk_element_pt())->set_remaining_shapes_appropriately(shape_info->bulk_shapeinfo, *(required_shapes.bulk_shapes));
			if (required_shapes.bulk_shapes->bulk_shapes)
			{
				dynamic_cast<BulkElementBase *>(dynamic_cast<InterfaceElementBase *>(this->bulk_element_pt())->bulk_element_pt())->set_remaining_shapes_appropriately(shape_info->bulk_shapeinfo->bulk_shapeinfo, *(required_shapes.bulk_shapes->bulk_shapes));
			}
		}
		if (required_shapes.opposite_shapes)
		{
			this->opposite_side->set_remaining_shapes_appropriately(shape_info->opposite_shapeinfo, *(required_shapes.opposite_shapes));
			if (required_shapes.opposite_shapes->bulk_shapes)
			{
				dynamic_cast<BulkElementBase *>(this->opposite_side->bulk_element_pt())->set_remaining_shapes_appropriately(shape_info->opposite_shapeinfo->bulk_shapeinfo, *(required_shapes.opposite_shapes->bulk_shapes));
			}
		}
	}

	// Builds the hanging-node equation info (used by the JIT-generated code to assemble residuals/Jacobians
	// correctly across hanging nodes) for the required bulk/opposite/their-bulk elements, and additionally
	// (when eqn_remap is not already provided from outside) fills in this element's own eqn-remapping arrays
	// (bulk_eqn_map, bulk_bulk_eqn_map, opp_interf_eqn_map, opp_bulk_eqn_map) built by update_equation_remapping().
	bool InterfaceElementBase::fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap)
	{
		HangFillTimeScope __hftime(HANGFILL_SLOT_FILL);
		//	Bulk is setup elsewhere
		bool ret_bulk = false;

		if (required.bulk_shapes)
		{
			// We need to fill the hang info of the bulk
			BulkElementBase *blk = dynamic_cast<BulkElementBase *>(this->bulk_element_pt());
			try
			{
				blk->fill_hang_info_with_equations(*(required.bulk_shapes), shape_info->bulk_shapeinfo, (eqn_remap ? NULL : &(bulk_eqn_map[0])));
			}
			catch (...)
			{
				std::cerr << "AT PERFORMING BULK EQ REMAPPING OF INTERFACE ELEMENT with code " << this->codeinst->get_code()->get_file_name() << std::endl;
				throw;
			}
			// Now perform the mapping
			ret_bulk = true;
			if (required.bulk_shapes->bulk_shapes)
			{
				BulkElementBase *blkblk = dynamic_cast<BulkElementBase *>(dynamic_cast<InterfaceElementBase *>(blk)->bulk_element_pt());
				try
				{
					blkblk->fill_hang_info_with_equations(*(required.bulk_shapes->bulk_shapes), shape_info->bulk_shapeinfo->bulk_shapeinfo, (eqn_remap ? NULL : &(bulk_bulk_eqn_map[0])));
				}
				catch (...)
				{
					std::cerr << "AT PERFORMING BULK EQ REMAPPING OF INTERFACE ELEMENT with code " << this->codeinst->get_code()->get_file_name() << std::endl;
					throw;
				}
			}
		}

		//	for (unsigned int ii=0;ii<opp_interf_eqn_map.size();ii++)  std::cout << "   " << ii << "  " << opp_interf_eqn_map[ii] << std::endl;
		if (required.opposite_shapes)
		{

			// We need to fill the hang info of the bulk
			InterfaceElementBase *opp = this->opposite_side;
			try
			{
				//std::cout << "Filling opposite hang info for interface element " << this << " with opposite " << opp << std::endl;
				//for (unsigned int i=0;i<opp_interf_eqn_map.size();i++)  std::cout << "   " << i << "  " << opp_interf_eqn_map[i] << " and eq remap is " << eqn_remap <<std::endl;
				opp->fill_hang_info_with_equations(*(required.opposite_shapes), shape_info->opposite_shapeinfo, (eqn_remap ? NULL : &(opp_interf_eqn_map[0])));
			}
			catch (...)
			{
				std::cerr << "AT PERFORMING OPPOSING INTERGACE EQ REMAPPING OF INTERFACE ELEMENT with code " << this->codeinst->get_code()->get_file_name() << std::endl;
				throw;
			}

			if (required.opposite_shapes->bulk_shapes)
			{
				// We need to fill the hang info of the bulk
				BulkElementBase *oppblk = dynamic_cast<BulkElementBase *>(opp->bulk_element_pt());
				try
				{
					oppblk->fill_hang_info_with_equations(*(required.opposite_shapes->bulk_shapes), shape_info->opposite_shapeinfo->bulk_shapeinfo, (eqn_remap ? NULL : &(opp_bulk_eqn_map[0])));
				}
				catch (...)
				{
					std::cerr << "AT PERFORMING OPPOSING BULK EQ REMAPPING OF INTERFACE ELEMENT with code " << this->codeinst->get_code()->get_file_name() << std::endl;
					throw;
				}
			}
			// Now perform the mapping
			ret_bulk = true;
		}

		return ret_bulk;
	}

	// Currently a no-op override (the actual external-data setup happens via add_required_external_data /
	// add_DG_external_data elsewhere); kept as an explicit override with the old approach commented out below
	// for reference, since blindly calling the base class here would flush already-set-up external data.
	// Deliberately a no-op, NOT BulkElementBase::ensure_external_data(): that one flushes the whole
	// external-data list and re-adds only the ED0 entries, which on an interface element would drop the
	// bulk/opposite data added at construction (and leave external_data_is_geometric describing a list
	// that no longer exists).
	void InterfaceElementBase::ensure_external_data()
	{
		/*   BulkElementBase::ensure_external_data(); //This would flush the storage...
		   external_data_is_geometric.resize(this->nexternal_data(),false);
			const JITFuncSpec_Table_FiniteElement_t * functable=this->codeinst->get_func_table();
			if (functable->shapes_required_ResJac.bulk_shapes) add_required_external_data(functable->shapes_required_ResJac.bulk_shapes,dynamic_cast<BulkElementBase*>(this->bulk_element_pt())); //TODO:

			if (functable->shapes_required_ResJac.opposite_shapes)
			{
			  if (!opposite_side) {throw_runtime_error("This element requires an opposite side of the interface to be set");}
			  add_required_external_data(functable->shapes_required_ResJac.opposite_shapes,opposite_side); //TODO:
			}
			*/
	}

	// Debug helper: builds a human-readable name for every local equation of this element, starting from the
	// base class's bulk-field names, then filling in interface-only field names, and finally (for equations
	// still unresolved, i.e. dofs actually owned by the bulk/opposite/opposite-bulk element but shared via the
	// equation-remapping mechanism) tagging them by matching global equation numbers against those elements'
	// own get_dof_names(), so e.g. "@BULK:..." / "@OPPSIDE:..." / "@OPPBLK:..." names show where a dof really lives.
	std::vector<std::string> InterfaceElementBase::get_dof_names(bool not_a_root_call)
	{
		// const JITFuncSpec_Table_FiniteElement_t * functable=codeinst->get_func_table();
		std::vector<std::string> res = BulkElementBase::get_dof_names(not_a_root_call);

		const JITFuncSpec_Table_FiniteElement_t *functable = this->codeinst->get_func_table();


		for (unsigned int si=0;si<functable->num_present_continuous_spaces;si++)
		{
			auto space_info=functable->present_continuous_spaces[si];
			for (unsigned int i = 0; i < eleminfo.nnode_of_space[space_info->space_index]; i++)
			{
				for (unsigned int j = 0; j < space_info->numfields - space_info->numfields_basebulk; j++)
				{
					unsigned node_index = j + space_info->buffer_offset_interf; // TODO: This index right?
					int leq = eleminfo.nodal_local_eqn[i][node_index];
					if (leq >= 0 && res[leq] == "<unknown>")
					{
						res[leq] = "IFIELD_" + std::string(space_info->fieldnames[space_info->numfields_basebulk + j]) + "__" + std::string(space_info->space_name) + "__" + std::to_string(i); // TODO: Interhangs?
					}
				}
			}
		}

		BulkElementBase *be = dynamic_cast<BulkElementBase *>(this->bulk_element_pt());
		std::vector<std::string> bres = be->get_dof_names(not_a_root_call);
		for (unsigned int i = 0; i < bres.size(); i++)
		{
			// Try to resolve the equation for the bulk
			int iglob = be->eqn_number(i);
			if (iglob >= 0)
			{
				// Now see if we also have that number
				for (unsigned int j = 0; j < this->ndof(); j++)
				{
					int jglob = this->eqn_number(j);
					if (iglob == jglob)
					{
						if (res[j] == "<unknown>")
						{
							res[j] = "@BULK:" + bres[i];
						}
					}
				}
			}
		}

		BulkElementBase *opp = this->opposite_side;
		if (opp && !not_a_root_call)
		{
			std::vector<std::string> ores = opp->get_dof_names(true);
			for (unsigned int i = 0; i < ores.size(); i++)
			{
				int iglob = opp->eqn_number(i);
				if (iglob >= 0)
				{
					for (unsigned int j = 0; j < this->ndof(); j++)
					{
						int jglob = this->eqn_number(j);
						if (iglob == jglob)
						{
							if (res[j] == "<unknown>")
							{
								res[j] = "@OPPSIDE:" + ores[i];
							}
						}
					}
				}
			}
			InterfaceElementBase *iopp = opp->as_interface_element();
			if (iopp)
			{
				BulkElementBase *oppblk = dynamic_cast<BulkElementBase *>(iopp->bulk_element_pt());
				std::vector<std::string> obres = oppblk->get_dof_names(not_a_root_call);
				for (unsigned int i = 0; i < obres.size(); i++)
				{
					int iglob = oppblk->eqn_number(i);
					if (iglob >= 0)
					{
						for (unsigned int j = 0; j < this->ndof(); j++)
						{
							int jglob = this->eqn_number(j);
							if (iglob == jglob)
							{
								if (res[j] == "<unknown>")
								{
									res[j] = "@OPPBLK:" + obres[i];
								}
							}
						}
					}
				}
			}
		}

		return res;
	}

	

	// Interface-field counterpart to BulkElementBase::pin_dummy_values() (see the comment near line 478 for
	// what a "dummy value" is): pins the interface-only dof at nodes where it is not an independent dof but
	// interpolated/averaged from others (per get_dummy_value_interpolation_map()), and constrains it at
	// hanging nodes so it follows the corresponding master nodes' interface dofs.
	void InterfaceElementBase::pin_dummy_values()
	{
		BulkElementBase::pin_dummy_values();
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();

		const std::vector<std::vector<unsigned>> & space_nodes_to_element_nodes=this->get_nodal_space_index_to_element_index_map();
		const std::vector<std::vector<std::vector<unsigned>>> & dummy_interpolation_mapping=this->get_dummy_value_interpolation_map();
		for (unsigned int space_index=0;space_index<functable->num_present_continuous_spaces;space_index++)
		{
			
			auto *space_info=functable->present_continuous_spaces[space_index];
			if (space_info->numfields==space_info->numfields_basebulk) continue; // No interface fields for this space
			// Pin all dummy values for this space
			const std::vector<std::vector<unsigned>> & dummies=dummy_interpolation_mapping[space_info->space_index];
			for (const std::vector<unsigned> &dummy_entry : dummies)
			{
				pyoomph::BoundaryNode * bn=dynamic_cast<pyoomph::BoundaryNode*>(this->node_pt(dummy_entry[0]));
				if (!bn) throw_runtime_error("This should be a boundary node here");
				for (unsigned int fi=space_info->numfields_basebulk;fi<space_info->numfields;fi++)
				{					
					//std::cout << "Pinning dummy value for space " << space_info->space_name << " at node " << dummy_entry[0] << " for field " << fi << " with index " << space_info->interface_dof_indices[fi-space_info->numfields_basebulk] << " and name " << space_info->fieldnames[space_info->numfields_basebulk+fi] << " value index " << bn->index_of_first_value_assigned_by_face_element(space_info->interface_dof_indices[fi-space_info->numfields_basebulk]) << std::endl;
					bn->pin(bn->index_of_first_value_assigned_by_face_element(space_info->interface_dof_indices[fi-space_info->numfields_basebulk]));
				}				
			}
			// Check whether non-dummy values are hanging, and if so, constrain them
			for (unsigned int ni : space_nodes_to_element_nodes[space_info->space_index])
			{
				pyoomph::BoundaryNode * bn=dynamic_cast<pyoomph::BoundaryNode*>(this->node_pt(ni));
				if (!bn) throw_runtime_error("This should be a boundary node here");
				if (bn->is_hanging(space_info->hangindex))
				{
					for (unsigned int fi=space_info->numfields_basebulk;fi<space_info->numfields;fi++)
					{
						//std::cout << "Cosntrainting for space " << space_info->space_name << " at node " << ni << " for field " << fi << " with index " << space_info->interface_dof_indices[fi-space_info->numfields_basebulk] << " and name " << space_info->fieldnames[space_info->numfields_basebulk+fi] << std::endl;
						this->node_pt(ni)->constrain(bn->index_of_first_value_assigned_by_face_element(space_info->interface_dof_indices[fi-space_info->numfields_basebulk]));
					}
				}
			}
		}
	}

	// User-added additional dof constraints (see NodeWithFieldIndicesBase::add_additional_dof_constraint):
	// an INTERFACE_DOF_CONSTRAIN_TO_C1 entry pins the additional (interface-only) dof assigned by this
	// interface's field with the given interface id, locally reducing it to C1. Just as for the base
	// fields, this is only allowed on nodes that do not carry an independent C1 dof themselves.
	void InterfaceElementBase::setup_additional_dof_constraints()
	{
		BulkElementBase::setup_additional_dof_constraints();
		auto *functable = codeinst->get_func_table();

		const std::vector<int> &elem_to_C1 = this->get_element_index_to_nodal_space_index_map()[SPACE_INDEX_C1];
		bool has_C1_fields=functable->continuous_spaces[SPACE_INDEX_C1].numfields_basebulk>0 || functable->continuous_spaces[SPACE_INDEX_C1TB].numfields_basebulk>0 ;
		for (unsigned int l = 0; l < nnode(); l++)
		{
			Node *n = static_cast<Node *>(node_pt(l));
			pyoomph::BoundaryNode *bn = dynamic_cast<pyoomph::BoundaryNode *>(n);
			// Interface elements on the interior-facet skeleton sit on ordinary interior nodes, which
			// are not BoundaryNodes: dereferencing bn below segfaulted as soon as the bulk had any
			// C1/C1TB field. Such a node cannot carry an interface-only dof in the first place, so it
			// has nothing to constrain.
			if (!bn)
			{
				if (n->get_additional_dof_constraints())
					throw_runtime_error("An additional interface dof constraint was registered on a node which is not a BoundaryNode. Interface-only dofs cannot be stored there.");
				continue;
			}
			bool hangs_on_C1=has_C1_fields && bn->is_hanging(functable->continuous_spaces[SPACE_INDEX_C1].hangindex) && this->refinement_level()>0	 ;
			for (const AdditionalDofConstrainingInfo *info = n->get_additional_dof_constraints(); info != NULL; )
			{
				const AdditionalDofConstrainingInfo *next_info = info->next;
				if (info->mode == INTERFACE_DOF_CONSTRAIN_TO_C1)
				{
					if (elem_to_C1[l] >= 0 && !hangs_on_C1)
					{
						 throw_runtime_error("Cannot constrain interface dof to C1 on a C1 node.\n\
											  This can happen in adaptive problems without any C1 or C1TB fields in the bulk.\n\
											  Add a ScalarField(\"_dummyC1\",space=\"C1\")+DirichletBC(_dummyC1=0) to the bulk domain to avoid this.");
					}

					int vindex = bn->index_of_first_value_assigned_by_face_element(info->index);
					if (vindex<0) throw_runtime_error("Interface DOF index not found here");
					n->pin((unsigned)vindex);
					has_additional_dof_constraints = true;


				}
				info=next_info; // Move to the next constraint in the list (since we might have removed the current one)
			}
		}

	}

	// Interface-field counterpart to the base class: additionally registers any pinned interface-only dofs
	// as Dirichlet dofs in 'info' (used when temporarily unpinning Dirichlet dofs for direct matrix manipulation,
	// e.g. eigenproblems, and later restoring them).
	void InterfaceElementBase::unpin_Dirichlet_dofs_for_matrix_manipulation(DirichletMatrixManipulationInfo & info)
	{
		BulkElementBase::unpin_Dirichlet_dofs_for_matrix_manipulation(info);
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();


		const std::vector<std::vector<unsigned>> & space_node_to_element_map=this->get_nodal_space_index_to_element_index_map();
		for (unsigned int i_space=0;i_space<functable->num_present_continuous_spaces;i_space++)
		{
			auto *space_info=functable->present_continuous_spaces[i_space];
			if (space_info->numfields==space_info->numfields_basebulk) continue; // No interface-only fields on this space, so the loop body would do nothing - and on the interior-facet skeleton the nodes are not BoundaryNodes, which used to make the check below throw
			for (unsigned ni : space_node_to_element_map[space_info->space_index])
			{
				pyoomph::BoundaryNode * bn=dynamic_cast<pyoomph::BoundaryNode *>(this->node_pt(ni));
				if (!bn) throw_runtime_error("This should be a boundary node here");
				for (unsigned int i = 0;i<space_info->numfields-space_info->numfields_basebulk; i++)
				{
				  unsigned value_index=bn->index_of_first_value_assigned_by_face_element(space_info->interface_dof_indices[i]);
				  if (this->node_pt(ni)->is_pinned(value_index)) info.add_dirichlet_dof(this->node_pt(ni),value_index);
				}
			}
		}
	}


	
   // Evaluates an interface-only field (identified by its interface dof index ifindex) at local coordinate s,
   // using the shape functions of the given nodal space ("C1"/"C1TB"/"C2"/"C2TB"), at history/time index t.
   double InterfaceElementBase::get_interpolated_interface_field(const oomph::Vector<double> &s,  const unsigned & ifindex,const std::string & space,const unsigned &t) const
   {
		double res=0.0;		
		oomph::Shape psi;
      std::vector<unsigned> node_index;		
	  const std::vector<std::vector<unsigned>> & space_nodes_to_element_nodes=this->get_nodal_space_index_to_element_index_map();
		if (space=="C2TB")
		{
		  psi.resize(eleminfo.nnode_of_space[SPACE_INDEX_C2TB]);
		  node_index.resize(psi.nindex1());
  		  this->shape_at_s_C2TB(s, psi);				 
  		  for (unsigned int i=0;i<node_index.size();i++) node_index[i]=space_nodes_to_element_nodes[SPACE_INDEX_C2TB][i];
		}
		else if (space=="C2")
		{
		  psi.resize(eleminfo.nnode_of_space[SPACE_INDEX_C2]);
		  node_index.resize(psi.nindex1());
  		  this->shape_at_s_C2(s, psi);				 
  		  for (unsigned int i=0;i<node_index.size();i++) node_index[i]=space_nodes_to_element_nodes[SPACE_INDEX_C2][i];
		}
		else if (space=="C1TB")
		{
		  psi.resize(eleminfo.nnode_of_space[SPACE_INDEX_C1TB]);
		  node_index.resize(psi.nindex1());
  		  this->shape_at_s_C1TB(s, psi);				 
  		  for (unsigned int i=0;i<node_index.size();i++) node_index[i]=space_nodes_to_element_nodes[SPACE_INDEX_C1TB][i];
		}		
		else if (space=="C1")
		{
		  psi.resize(eleminfo.nnode_of_space[SPACE_INDEX_C1]);
		  node_index.resize(psi.nindex1());
  		  this->shape_at_s_C1(s, psi);				 
  		  for (unsigned int i=0;i<node_index.size();i++) node_index[i]=space_nodes_to_element_nodes[SPACE_INDEX_C1][i];
		}
		else 
		{
		 throw_runtime_error("Cannot interpolate interface fields on space '"+space+"' yet");
		}						

		for (unsigned int l = 0; l < psi.nindex1(); l++)
		{
		   oomph::Node * n=this->node_pt(node_index[l]);
		   unsigned fi=dynamic_cast<oomph::BoundaryNodeBase *>(n)->index_of_first_value_assigned_by_face_element(ifindex);
			res += psi[l] * n->value(t, fi);
		}		
		
		return res;
		
   }
   
	// If the opposite side is only a placeholder "dummy" element (used on internal facets where the true
	// opposite element has no dofs of its own yet), makes sure its local equation numbers get assigned first,
	// then lets the base classes assign this element's own additional (hanging/interface) equations.
	void InterfaceElementBase::assign_additional_local_eqn_numbers()
	{
		if (opposite_side && opposite_side->is_internal_facet_opposite_dummy() && !(opposite_side->ndof()))
		{

		  opposite_side->assign_local_eqn_numbers(true);
		}
		BulkElementBase::assign_additional_local_eqn_numbers();
		oomph::FaceElement::assign_additional_local_eqn_numbers();
	}

	bool InterfaceElementBase::interpolate_new_interface_dofs=true;
}
