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


// Degrees of freedom: local equation numbering, the dof-contribution index tables consumed by the
// frozen-sparsity path, human-readable dof names, and the pinning machinery (dummy values,
// additional constraints, and the Dirichlet unpinning used for matrix manipulation).

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

	// Unpins every position and value dof on this element's nodes and internal data (positions,
	// then nodal values, then internal data values), undoing the effect of pin_dummy_values(). Used
	// e.g. before a full re-pinning pass so pinning state is rebuilt consistently from scratch.
	void BulkElementBase::unpin_dummy_values() // C1 fields on C2 elements have dummy values on only C2 nodes, which needs to be pinned
	{
		// One pass, one cast. This used to be two passes over the same nodes with a dynamic_cast per
		// (node, coordinate direction); the routine runs once per element per Problem::setup_pinning(),
		// so on a 1M-dof mesh that was several million redundant casts and a second sweep through the
		// same (scattered, cache-cold) Node objects. The two passes were independent - unpinning
		// positions and unpinning values touch disjoint storage - so merging them changes nothing.
		const unsigned ndim_nodal = this->nodal_dimension();
		for (unsigned int l = 0; l < nnode(); l++)
		{
			oomph::Node *const on = node_pt(l);
			Node *const pn = dynamic_cast<Node *>(on);
			for (unsigned int i = 0; i < ndim_nodal; i++)
			{
				pn->unpin_position(i);
			}
			on->unconstrain_positions();
			const unsigned nval = on->nvalue();
			for (unsigned int i = 0; i < nval; i++)
			{
				on->unpin(i);
			}
		}

		for (unsigned int d = 0; d < this->ninternal_data(); d++)
		{
			for (unsigned int v = 0; v < this->internal_data_pt(d)->nvalue(); v++)
			{
				this->internal_data_pt(d)->unpin(v);
			}
		}

		has_additional_dof_constraints = false;
	}

	// Pins nodal position dofs that must not be free equations (all positions if the mesh does not
	// move at all; only the hanging ones, via constrain_positions(), if it does - since a hanging
	// position is determined by its masters, not by its own equation), and pins every "dummy" field
	// value (a lower-order field's value at a node that only exists for a higher-order geometric
	// space, see get_dummy_value_interpolation_map) so it never gets its own equation; if a node
	// carrying a real (non-dummy) value happens to be hanging in that space, its value is instead
	// constrained (tied to its masters) rather than pinned.
	void BulkElementBase::pin_dummy_values()
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();


		if (!functable->moving_nodes)
		{
			const unsigned ndim_nodal = this->nodal_dimension();
			for (unsigned int l = 0; l < nnode(); l++)
			{
				Node *const pn = dynamic_cast<Node *>(node_pt(l)); // hoisted out of the direction loop
				for (unsigned int i = 0; i < ndim_nodal; i++)
				{
					pn->pin_position(i);
				}
			}
		}
		else
		{
			for (unsigned int l = 0; l < nnode(); l++)
			{
				if (this->node_pt(l)->is_hanging())
				{
					this->node_pt(l)->constrain_positions();
				}				
			}
		}


		const std::vector<std::vector<unsigned>> & space_nodes_to_element_nodes=this->get_nodal_space_index_to_element_index_map();
		const std::vector<std::vector<std::vector<unsigned>>> & dummy_interpolation_mapping=this->get_dummy_value_interpolation_map();
		for (unsigned int space_index=0;space_index<functable->num_present_continuous_spaces;space_index++)
		{
			auto *space_info=functable->present_continuous_spaces[space_index];
			// Pin all dummy values for this space
			const std::vector<std::vector<unsigned>> & dummies=dummy_interpolation_mapping[space_info->space_index];
			for (const std::vector<unsigned> &dummy_entry : dummies)
			{
				for (unsigned int fi=0;fi<space_info->numfields_basebulk;fi++)
				{
					this->node_pt(dummy_entry[0])->pin(space_info->nodal_offset_basebulk+fi);
				}				
			}
			// Check whether non-dummy values are hanging, and if so, constrain them
			for (unsigned int ni : space_nodes_to_element_nodes[space_info->space_index])
			{
				if (this->node_pt(ni)->is_hanging(space_info->hangindex))
				{
					for (unsigned int fi=0;fi<space_info->numfields_basebulk;fi++)
					{
						this->node_pt(ni)->constrain(space_info->nodal_offset_basebulk+fi);
					}
				}
			}
		}

	}

	// User-added additional dof constraints (see NodeWithFieldIndicesBase::add_additional_dof_constraint):
	void BulkElementBase::setup_additional_dof_constraints()
	{
		// Nothing below does anything unless some node of this element actually carries a constraint:
		// both loops test get_additional_dof_constraints() before acting. The c1_corner_lookup map,
		// however, was built unconditionally - one std::map with its heap allocations per element, i.e.
		// 0.85 s over the 250k elements of a 1M-dof problem that has no additional constraints at all.
		{
			bool any = false;
			for (unsigned int l = 0; l < nnode(); l++)
			{
				Node *n = dynamic_cast<Node *>(node_pt(l));
				if (n && n->get_additional_dof_constraints()) { any = true; break; }
			}
			if (!any) return;
		}
		auto * functable = codeinst->get_func_table();
		const std::vector<int> & elem_to_C1_map = this->get_element_index_to_nodal_space_index_map()[SPACE_INDEX_C1];
		bool has_C1_fields=functable->continuous_spaces[SPACE_INDEX_C1].numfields_basebulk>0 || functable->continuous_spaces[SPACE_INDEX_C1TB].numfields_basebulk>0;
		// Lookup from a non-vertex node's local index to its element C1-corner local indices (the same
		// map used below for c1_constraint_corners). Used to give a constrained *hanging* value its own
		// genuine, oomph-registered linear value-slot hang (see the CONTINUOUS_BASE_DOF_CONSTRAIN_TO_C1
		// branch): pinning such a node would leave its value-slot hang masters unregistered, so the
		// assembled Jacobian would not match the residual on adaptively (un)refined simplex meshes.
		std::map<unsigned, std::vector<unsigned>> c1_corner_lookup;
		{
			const std::vector<std::vector<unsigned>> & c1_dummy_map0 = this->get_dummy_value_interpolation_map()[SPACE_INDEX_C1];
			for (const std::vector<unsigned> & entry : c1_dummy_map0)
				if (entry.size() >= 2) c1_corner_lookup[entry[0]] = std::vector<unsigned>(entry.begin() + 1, entry.end());
		}
		for (unsigned int l = 0; l < nnode(); l++)
		{
			Node *n = dynamic_cast<Node *>(node_pt(l));
			bool is_hanging_on_C1 = elem_to_C1_map[l]>=0 && has_C1_fields && n->is_hanging(functable->continuous_spaces[SPACE_INDEX_C1].hangindex) && this->refinement_level()>0;
			for (const AdditionalDofConstrainingInfo *info = n->get_additional_dof_constraints(); info != NULL; )
			{
				// Capture "next" before the body runs: a removal below can delete "info" itself,
				// so info->next must not be dereferenced afterwards (use-after-free).
				const AdditionalDofConstrainingInfo *next = info->next;
				if (info->mode == CONTINUOUS_BASE_DOF_CONSTRAIN_TO_C1)
				{
					if (info->index >= this->ncont_interpolated_values())
					{
						throw_runtime_error("Cannot enforce a degration to C1 on a base dof index larger than the number of base dofs on this element: index="+std::to_string(info->index)+" vs. ncont_interpolated_values()="+std::to_string(this->ncont_interpolated_values()));
					}
					// A constrained node being a C1 VERTEX of THIS element is not an error, and must not
					// abort: at a 2:1 interface the coarse element's constrained mid-node is a genuine
					// (non-hanging) vertex of each finer neighbour, and in 3d also of the sons created at a
					// father's face/volume centre. The block below already handles that case correctly by
					// doing nothing here -- the hang is installed, identically, by the element(s) where the
					// node IS a non-vertex (see the comment there). The previous guard demanded the node
					// hang on the C1 slot instead, which is true in 2d (a coarse edge-mid node's C1 value is
					// the mean of the two coarse corners) but not in 3d, so any 3d mesh with a 2:1 interface
					// threw. Bisection showed it aborted for plain bricks too, in every configuration except
					// the single one the existing 3d test happens to use.
					//
					// What IS a genuine misuse is asking to degrade to a C1 space that does not exist, which
					// is what this message has always described -- so that is what it now tests.
					if (elem_to_C1_map[l]>=0 && !has_C1_fields)
					{
						throw_runtime_error("Cannot enforce a degration to C1 on a C1 vertex node.\n\
							 This can happen in adaptive problems without any C1 or C1TB fields present in the bulk mesh.\n\
							 Add a ScalarField(\"_dummyC1\",space=\"C1\")+DirichletBC(_dummyC1=0) to the bulk domain to avoid this.");
					}

					// Pinning a constrained value leaves its C1-corner masters unregistered by oomph's
					// hanging machinery. That is harmless on a conforming mesh, but once refinement makes
					// the node (or one of its corner masters) hang, the assembly-time flatten can no longer
					// resolve those masters to real free dofs -- a pinned master is simply dropped -- so the
					// assembled Jacobian's redistribution no longer matches the residual (wrong Jacobian ->
					// slow/failed Newton on adaptively (un)refined simplex meshes). Instead give the value
					// its own genuine linear value-slot HangInfo on the element's C1 corner nodes: oomph
					// then registers these masters (and resolves any master that is itself hanging via
					// complete_hanging_nodes), exactly as RefineableQElement::setup_hang_for_value does for
					// the C1 fields on quads -- which keeps it MPI-consistent.
					//
					// A constrained node is a non-vertex node (edge-/face-mid) of the element(s) that added
					// the constraint, so its C1 corners are found here via the dummy-interpolation map. The
					// SAME node may also be a C1 vertex of a finer neighbouring element (a 2:1 coarse
					// mid-node): there it is not in that element's map, so we do nothing -- the hang is set
					// (identically) by the element(s) where it is a non-vertex. Never pin such a node: a pin
					// from the vertex-side element would clobber the value-slot hang (order-dependently, so
					// especially harmful under MPI) and drop it from its masters' redistribution. Constraint
					// markers are cleared+reapplied every assign (Mesh::clear/apply_additional_dof_constraints),
					// so every currently-constrained node is a non-vertex of at least one live element and
					// therefore always receives its hang.
					std::map<unsigned, std::vector<unsigned>>::const_iterator cit = c1_corner_lookup.find(l);
					if (cit != c1_corner_lookup.end() && !cit->second.empty())
					{
						const std::vector<unsigned> & corners = cit->second;
						oomph::HangInfo *hang = new oomph::HangInfo(corners.size());
						for (unsigned m = 0; m < corners.size(); m++)
							hang->set_master_node_pt(m, node_pt(corners[m]), 1.0 / (double)corners.size());
						n->set_hanging_pt(hang, (int)info->index);
					}
					has_additional_dof_constraints = true;
				}
				else if (info->mode == POSITION_CONSTRAIN_TO_C1)
				{
					if (info->index >= this->nodal_dimension())
					{
						throw_runtime_error("Cannot enforce a degration to C1 on a coordinate index larger than the nodal dimension of this element");
					}
					// Same relaxation as the field branch: a constrained node that is a C1 VERTEX of THIS
					// element is the ordinary 2:1 situation, not an error. This was held back until the
					// position redistribution actually worked at a T-junction; with
					// register_c1_constraint_position_masters() supplying the missing registrations it does,
					// so the guard now tests only the genuine misuse it describes.
					if (elem_to_C1_map[l]>=0 && !has_C1_fields)
					{
						throw_runtime_error("Cannot enforce a degration to C1 on a C1 vertex node\n\
							 This can happen in adaptive problems without any C1 or C1TB fields present in the bulk mesh.\n\
							 Add a ScalarField(\"_dummyC1\",space=\"C1\")+DirichletBC(_dummyC1=0) to the bulk domain to avoid this.");
					}

					n->pin_position(info->index);
					has_additional_dof_constraints = true;
				}
				info = next;
			}
		}

		// Precompute, for every constrained node that is a non-vertex node of THIS element, its
		// immediate C1-corner expansion (the element's C1 vertex nodes, equal weights). Stored on the
		// node so that flatten_hang_for_value() can recursively expand it into real free dofs even when
		// the node is later encountered as a hang master from a neighbouring element (where it may be a
		// C1 vertex, so its own corners are not locally derivable). See
		// NodeWithFieldIndicesBase::c1_constraint_corners and dev_docs/adaptive_refinement.md section 3.
		const std::vector<std::vector<unsigned>> & c1_dummy_map = this->get_dummy_value_interpolation_map()[SPACE_INDEX_C1];
		for (const std::vector<unsigned> & entry : c1_dummy_map)
		{
			if (entry.size() < 2) continue;
			Node *tgt = dynamic_cast<Node *>(node_pt(entry[0]));
			if (!tgt || tgt->get_additional_dof_constraints() == NULL) continue;
			const unsigned nc = entry.size() - 1;
			tgt->c1_constraint_corners.clear();
			for (unsigned m = 1; m < entry.size(); m++)
				tgt->c1_constraint_corners.push_back(std::make_pair(node_pt(entry[m]), 1.0 / (double)nc));
		}
	}

	// Scans every position and field dof on this element (mesh positions, continuous-space fields,
	// DG/DL/D0 discontinuous fields) and, for every one that is currently pinned (a Dirichlet
	// boundary condition), records it in "info" via add_dirichlet_dof so that a linear-algebra
	// backend can later temporarily unpin and directly manipulate those rows/columns (e.g. to
	// enforce the constraint by row replacement rather than by pinning). Despite the name, this
	// function only *collects* the Dirichlet dofs here; the actual unpinning happens elsewhere using
	// the recorded info.
	void BulkElementBase::unpin_Dirichlet_dofs_for_matrix_manipulation(DirichletMatrixManipulationInfo & info)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();


		// TODO: Check if the entire field is pinned



		if (functable->moving_nodes)
		{
			for (unsigned int l = 0; l < nnode(); l++)
			{
				oomph::Data * x=dynamic_cast<Node *>(this->node_pt(l))->variable_position_pt();
				for (unsigned int i = 0; i < this->nodal_dimension(); i++)
				{
				  if (x->is_pinned(i)) info.add_dirichlet_dof(x,i);
				}
			}
		}

		const std::vector<std::vector<unsigned>> & space_node_to_element_map=this->get_nodal_space_index_to_element_index_map();
		for (unsigned int i_space=0;i_space<functable->num_present_continuous_spaces;i_space++)
		{
			auto *space_info=functable->present_continuous_spaces[i_space];
			for (unsigned ni : space_node_to_element_map[space_info->space_index])
			{
				for (unsigned int i = 0;i<space_info->numfields_basebulk; i++)
				{
				  if (this->node_pt(ni)->is_pinned(i+space_info->nodal_offset_basebulk)) info.add_dirichlet_dof(this->node_pt(ni),i+space_info->nodal_offset_basebulk);
				}
			}
		}

		for (unsigned int i_space=0;i_space<functable->num_present_dg_spaces;i_space++)
		{
			auto *space_info=functable->present_dg_spaces[i_space];
			for (unsigned int i = 0; i < space_info->numfields_basebulk; i++)
			{
				for (unsigned int v = 0; v < this->internal_data_pt(space_info->internal_offset_new + i)->nvalue(); v++)
				{
					if (this->internal_data_pt(space_info->internal_offset_new + i)->is_pinned(v)) info.add_dirichlet_dof(this->internal_data_pt(space_info->internal_offset_new + i),v);
				}
			}
		}

		for (unsigned int i = 0; i < functable->info_DL.numfields; i++)
		{
				for (unsigned int v = 0; v < this->internal_data_pt(functable->info_DL.internal_offset_new + i)->nvalue(); v++)
				{	
					if (this->internal_data_pt(functable->info_DL.internal_offset_new + i)->is_pinned(v)) info.add_dirichlet_dof(this->internal_data_pt(functable->info_DL.internal_offset_new + i),v);
				}
		}	
		for (unsigned int i = 0; i < functable->info_D0.numfields; i++)
		{
				for (unsigned int v = 0; v < this->internal_data_pt(functable->info_D0.internal_offset_new + i)->nvalue(); v++)
				{	
					if (this->internal_data_pt(functable->info_D0.internal_offset_new + i)->is_pinned(v)) info.add_dirichlet_dof(this->internal_data_pt(functable->info_D0.internal_offset_new + i),v);
				}
		}
		
	}

	// Small helper for the contribution-index map below: assign, but never downgrade an already
	// attributed dof back to "unknown", and ignore slots the code did not emit an index for.
	static inline void set_contrib(std::vector<int> &dest, int local_eqn, const int *table, unsigned slot)
	{
		// The bounds check is not belt-and-braces: an element with no unknowns at all (every dof pinned,
		// or a halo element) still carries eleminfo local-equation numbers from whenever it last had
		// some, and those are >= 0. Writing them into a dest sized by the CURRENT ndof() then runs off
		// the end. Seen as a segfault in distributed 3D adaptive meshes.
		if (local_eqn < 0 || !table) return;
		if ((size_t)local_eqn >= dest.size()) return;
		if (dest[local_eqn] < 0) dest[local_eqn] = table[slot];
	}

	// See the declaration in elements.hpp. Walks the same field/space order as get_dof_names(), but
	// records which CONTRIBUTION each local dof belongs to instead of a printable name. A hanging dof
	// is attributed to the field of the constrained node, which is what its master dofs contribute to.
	void BulkElementBase::fill_local_dof_contribution_indices(std::vector<int> &dest)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();

		// Positions (nodal coordinates), non-hanging then hanging
		const unsigned npos = std::min(functable->nodal_dim, functable->info_Pos.numfields);
		for (unsigned int i = 0; i < eleminfo.nnode; i++)
		{
			for (unsigned int j = 0; j < npos; j++)
			{
				if (!this->node_pt(i)->is_hanging())
				{
					set_contrib(dest, eleminfo.pos_local_eqn[i][j], functable->info_Pos.field_contribution_index, j);
				}
				else
				{
					oomph::HangInfo *hang_info_pt = this->node_pt(i)->hanging_pt();
					for (unsigned m = 0; m < hang_info_pt->nmaster(); m++)
					{
						oomph::DenseMatrix<int> pos_eqn = this->local_position_hang_eqn(hang_info_pt->master_node_pt(m));
						set_contrib(dest, pos_eqn(0, j), functable->info_Pos.field_contribution_index, j);
					}
				}
			}
		}

		// Continuous spaces (C2TB/C2/C1TB/C1), non-hanging and hanging
		const std::vector<std::vector<unsigned>> &space_node_to_elem_node_map = this->get_nodal_space_index_to_element_index_map();
		for (unsigned int si = 0; si < functable->num_present_continuous_spaces; si++)
		{
			auto *space_info = functable->present_continuous_spaces[si];
			for (unsigned int i = 0; i < eleminfo.nnode_of_space[space_info->space_index]; i++)
			{
				unsigned elem_node_index = space_node_to_elem_node_map[space_info->space_index][i];
				for (unsigned int j = 0; j < space_info->numfields_basebulk; j++)
				{
					unsigned val_index = j + space_info->nodal_offset_basebulk;
					if (!this->node_pt(elem_node_index)->is_hanging(space_info->hangindex))
					{
						set_contrib(dest, eleminfo.nodal_local_eqn[i][val_index], space_info->field_contribution_index, j);
					}
					else
					{
						oomph::HangInfo *hang_info_pt = this->node_pt(elem_node_index)->hanging_pt(space_info->hangindex);
						for (unsigned m = 0; m < hang_info_pt->nmaster(); m++)
						{
							set_contrib(dest, this->local_hang_eqn(hang_info_pt->master_node_pt(m), val_index),
										space_info->field_contribution_index, j);
						}
					}
				}
			}
		}

		// Discontinuous (DG) spaces
		for (unsigned int i_space = 0; i_space < functable->num_present_dg_spaces; i_space++)
		{
			auto *space_info = functable->present_dg_spaces[i_space];
			for (unsigned int j = 0; j < space_info->numfields; j++)
				for (unsigned int i = 0; i < eleminfo.nnode_of_space[space_info->space_index]; i++)
					set_contrib(dest, this->get_DG_local_equation(space_info->space_index, j, i),
								space_info->field_contribution_index, j);
		}

		// Interface-only fields of the continuous spaces (the numfields-numfields_basebulk part). These
		// are extra values that an interface element adds to nodes it shares with the bulk, addressed
		// through interface_dof_indices rather than the nodal offset, which is why the basebulk loop
		// above cannot reach them. Without this they stayed "not attributed", i.e. assumed coupled to
		// everything -- including themselves, which put a structural zero on their diagonal.
		for (unsigned int si = 0; si < functable->num_present_continuous_spaces; si++)
		{
			auto *space_info = functable->present_continuous_spaces[si];
			const unsigned n_interf = space_info->numfields - space_info->numfields_basebulk;
			if (!n_interf || !space_info->interface_dof_indices) continue;
			for (unsigned int i = 0; i < eleminfo.nnode_of_space[space_info->space_index]; i++)
			{
				const unsigned elem_node_index = space_node_to_elem_node_map[space_info->space_index][i];
				oomph::Node *nod_pt = this->node_pt(elem_node_index);
				auto *bnod_pt = dynamic_cast<pyoomph::BoundaryNode *>(nod_pt);
				if (!bnod_pt) continue; // Not an interface node: nothing of ours here
				for (unsigned int f = 0; f < n_interf; f++)
				{
					const unsigned interface_dof_id = space_info->interface_dof_indices[f];
					const unsigned slot = space_info->numfields_basebulk + f; // field_contribution_index is parallel to fieldnames
					if (!nod_pt->is_hanging(space_info->hangindex))
					{
						const unsigned val_index = bnod_pt->index_of_first_value_assigned_by_face_element(interface_dof_id);
						set_contrib(dest, this->nodal_local_eqn(elem_node_index, val_index),
									space_info->field_contribution_index, slot);
					}
					else if (auto *ielem = dynamic_cast<InterfaceElementBase *>(this))
					{
						// Resolving a hanging interface dof to its masters is an interface-element
						// operation. On a bulk element these are somebody else's dofs anyway, and the
						// pass at the end of this function marks them as such.
						oomph::HangInfo *const hang_info_pt = nod_pt->hanging_pt(space_info->hangindex);
						for (unsigned m = 0; m < hang_info_pt->nmaster(); m++)
						{
							set_contrib(dest, ielem->local_interface_hang_eqn(interface_dof_id, hang_info_pt->master_node_pt(m)),
										space_info->field_contribution_index, slot);
						}
					}
				}
			}
		}

		// Element-local spaces: discontinuous Lagrange, piecewise constant, external ODE
		for (unsigned int i = 0; i < eleminfo.nnode_DL; i++)
			for (unsigned int j = 0; j < functable->info_DL.numfields; j++)
				set_contrib(dest, eleminfo.nodal_local_eqn[i][j + functable->info_DL.buffer_offset_basebulk],
							functable->info_DL.field_contribution_index, j);
		for (unsigned int j = 0; j < functable->info_D0.numfields; j++)
			set_contrib(dest, eleminfo.nodal_local_eqn[0][j + functable->info_D0.buffer_offset_basebulk],
						functable->info_D0.field_contribution_index, j);
		for (unsigned int j = 0; j < functable->info_ED0.numfields; j++)
			set_contrib(dest, eleminfo.nodal_local_eqn[0][j + functable->info_ED0.buffer_offset_basebulk],
						functable->info_ED0.field_contribution_index, j);

		// Values that some OTHER code added to our nodes -- an interface element's extra dofs seen from
		// the bulk element, or a second interface's seen from the first. They sit past
		// ncont_interpolated_values(), which is exactly how get_dof_names() identifies them (it labels
		// them "<added interface dof>"). This element has no field for them, hence no residual and no
		// Jacobian entry: their row and column of its block are empty, which is a positive statement
		// and not the absence of one, so they get -2 rather than being left "not attributed".
		//
		// Anything the walk above already attributed is left alone: this only fills in dofs still at -1.
		{
			const unsigned ncont = this->ncont_interpolated_values();
			for (unsigned int l = 0; l < this->nnode(); l++)
			{
				oomph::Node *nod_pt = this->node_pt(l);
				for (unsigned int n = ncont; n < nod_pt->nvalue(); n++)
				{
					const int le = this->nodal_local_eqn(l, n);
					if (le >= 0 && (size_t)le < dest.size() && dest[le] == -1) dest[le] = -2;
				}
			}
		}
	}

	const std::vector<int> &BulkElementBase::get_local_dof_contribution_indices()
	{
		if (!local_dof_contribution_indices_valid)
		{
			if (!this->ndof())
			{
				// Nothing to attribute, and the walk below would read stale eleminfo entries.
				local_dof_contribution_indices.clear();
				local_dof_contribution_indices_valid = true;
				return local_dof_contribution_indices;
			}
			// -1 everywhere to begin with: anything the walk below does not attribute stays "unknown",
			// which downstream must read as "assume coupled to everything" (see elements.hpp).
			local_dof_contribution_indices.assign(this->ndof(), -1);
			this->fill_local_dof_contribution_indices(local_dof_contribution_indices);
			local_dof_contribution_indices_valid = true;
		}
		return local_dof_contribution_indices;
	}

	// Builds a human-readable name for each local dof/equation of this element, used for
	// debugging (e.g. printing which equation a Jacobian row/column corresponds to). Walks
	// through every field/space in the same order fill_element_info() uses (non-hanging Pos,
	// hanging Pos via masters, then continuous spaces, DG, DL, D0, ED0, ...), so the produced
	// names line up with the residual/Jacobian ordering.
	std::vector<std::string> BulkElementBase::get_dof_names(bool)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		std::vector<std::string> res(this->ndof(), "<unknown>");

		// First nonhanging pos
		for (unsigned int i = 0; i < eleminfo.nnode; i++)
		{
			for (unsigned int j = 0; j < std::min(functable->nodal_dim, functable->info_Pos.numfields); j++)
			{
				if (!this->node_pt(i)->is_hanging())
				{
					if (eleminfo.pos_local_eqn[i][j] >= 0)
						res[eleminfo.pos_local_eqn[i][j]] = std::string(functable->info_Pos.fieldnames[j]) + "__Pos__" + std::to_string(i);
				}
			}
		}
		// Now hanging pos
		for (unsigned int i = 0; i < eleminfo.nnode; i++)
		{
			for (unsigned int j = 0; j < std::min(functable->nodal_dim, functable->info_Pos.numfields); j++)
			{
				if (this->node_pt(i)->is_hanging())
				{
					oomph::HangInfo *hang_info_pt = this->node_pt(i)->hanging_pt();
					const unsigned n_master = hang_info_pt->nmaster();
					for (unsigned m = 0; m < n_master; m++)
					{
						oomph::Node *const master_node_pt = hang_info_pt->master_node_pt(m);
						oomph::DenseMatrix<int> Position_local_eqn_at_node = this->local_position_hang_eqn(master_node_pt);
						int local_unknown = Position_local_eqn_at_node(0, j);
						if (local_unknown >= 0)
						{
							if (res[local_unknown] == "<unknown>")
							{
								res[local_unknown] = "__HANGING__" + std::to_string(m) + "__of__" + std::string(functable->info_Pos.fieldnames[j]) + "__Pos__" + std::to_string(i);
							}
						}
					}
				}
			}
		}

		
		const std::vector<std::vector<unsigned>> & space_node_to_elem_node_map=this->get_nodal_space_index_to_element_index_map();
		for (unsigned int si=0;si<functable->num_present_continuous_spaces;si++)
		{
			auto * space_info=functable->present_continuous_spaces[si];
			for (unsigned int i = 0; i < eleminfo.nnode_of_space[space_info->space_index]; i++)
			{
				unsigned elem_node_index=space_node_to_elem_node_map[space_info->space_index][i];
				for (unsigned int j = 0; j < space_info->numfields_basebulk; j++)
				{					
					if (!this->node_pt(elem_node_index)->is_hanging(space_info->hangindex))
					{
						unsigned val_index = j + space_info->nodal_offset_basebulk;
						if (eleminfo.nodal_local_eqn[i][val_index] >= 0)
							res[eleminfo.nodal_local_eqn[i][val_index]] = std::string(space_info->fieldnames[j]) + "__" + space_info->space_name + "__" + std::to_string(i);
					}
					else
					{
						oomph::HangInfo *hang_info_pt = this->node_pt(elem_node_index)->hanging_pt(space_info->hangindex);
						const unsigned n_master = hang_info_pt->nmaster();
						for (unsigned m = 0; m < n_master; m++)
						{
							int local_unknown = this->local_hang_eqn(hang_info_pt->master_node_pt(m), j+ space_info->nodal_offset_basebulk);
							if (local_unknown >= 0)
							{
								if (res[local_unknown] == "<unknown>")
								{
									res[local_unknown] = "__HANGING__" + std::to_string(m) + "__of__" + std::string(space_info->fieldnames[j]) + "__" + space_info->space_name + "__" + std::to_string(i);
								}
							}
						}
					}
				}
			}
		}


		//  Additional interface dofs ( will be added by the overridden method )

		for (unsigned int i_space=0;i_space<functable->num_present_dg_spaces;i_space++)
		{
			auto * space_info=functable->present_dg_spaces[i_space];
			for (unsigned int j = 0; j < space_info->numfields; j++)
			{
				for (unsigned int i = 0; i < eleminfo.nnode_of_space[space_info->space_index]; i++)
				{
				int loc_eq=this->get_DG_local_equation(space_info->space_index, j,i);
				if (loc_eq >= 0) res[loc_eq] = std::string(space_info->fieldnames[j]) + "__"+std::string(space_info->space_name)+"__" + std::to_string(i);
				}
			}
		}


		for (unsigned int i = 0; i < eleminfo.nnode_DL; i++)
		{
			for (unsigned int j = 0; j < functable->info_DL.numfields; j++)
			{
				unsigned node_index = j + functable->info_DL.buffer_offset_basebulk;
				if (eleminfo.nodal_local_eqn[i][node_index] >= 0)
					res[eleminfo.nodal_local_eqn[i][node_index]] = std::string(functable->info_DL.fieldnames[j]) + "__DL__" + std::to_string(i);
			}
		}

		for (unsigned int j = 0; j < functable->info_D0.numfields; j++)
		{
			unsigned node_index = j + functable->info_D0.buffer_offset_basebulk;
			if (eleminfo.nodal_local_eqn[0][node_index] >= 0)
				res[eleminfo.nodal_local_eqn[0][node_index]] = std::string(functable->info_D0.fieldnames[j]) + "__D0";
		}
		
		for (unsigned int j = 0; j < functable->info_ED0.numfields; j++)
		{
			unsigned node_index = j + functable->info_ED0.buffer_offset_basebulk;
			if (eleminfo.nodal_local_eqn[0][node_index] >= 0)
				res[eleminfo.nodal_local_eqn[0][node_index]] = std::string(functable->info_ED0.fieldnames[j]) + "__ExternalODE";
		}		

		if (!dynamic_cast<InterfaceElementBase *>(this))
		{
			// Check if we have unknown fields. It should not happen at the end
			for (unsigned int i = 0; i < res.size(); i++)
			{
				if (res[i] == "<unknown>")
				{
					std::stringstream oss;
					oss << "Cannot find a DoF name for local " << i << ", global " << this->eqn_number(i);
					// Now try to check what it is
					for (unsigned int l = 0; l < this->nnode(); l++)
					{
						for (unsigned int n = 0; n < this->node_pt(l)->nvalue(); n++)
						{
							if (this->node_pt(l)->eqn_number(n) == (long int)this->eqn_number(i))
							{
								if (n >= ncont_interpolated_values() && this->node_pt(l)->is_on_boundary())
								{
									res[i] = "<added interface dof>";
								}
								else
								{
									oss << ", which corresponds to nodal value " << n << " of " << (this->node_pt(l)->is_on_boundary() ? "boundary " : "") << "node " << l;
								}
							}
						}
					}
					for (unsigned int l = 0; l < this->ninternal_data(); l++)
					{
						for (unsigned int n = 0; n < this->internal_data_pt(l)->nvalue(); n++)
						{
							if (this->internal_data_pt(l)->eqn_number(n) == (long int)this->eqn_number(i))
							{
								oss << ", which corresponds to internal data value " << n << " of internal data " << l;
							}
						}
					}
					for (unsigned int l = 0; l < this->nexternal_data(); l++)
					{
						for (unsigned int n = 0; n < this->external_data_pt(l)->nvalue(); n++)
						{
							if (this->external_data_pt(l)->eqn_number(n) == (long int)this->eqn_number(i))
							{
								oss << ", which corresponds to external data value " << n << " of external data " << l;
							}
						}
					}					

					// throw_runtime_error(oss.str());
					if (res[i] == "<unknown>")
					{
						std::cerr << oss.str() << std::endl;
					}
				}
			}
		}

		return res;
	}
	

	// Debug helper: assembles residuals/Jacobian for this element and returns them alongside
	// human-readable dof names, for inspection from Python.
	void BulkElementBase::get_debug_jacobian_info(oomph::Vector<double> &R, oomph::DenseMatrix<double> &J, std::vector<std::string> &dofnames)
	{
		dofnames = get_dof_names();
		R.resize(this->ndof(), 0);
		J.resize(this->ndof(), this->ndof(), 0);
		this->fill_in_contribution_to_jacobian(R, J);
	}

	void BulkElementBase::assign_additional_local_eqn_numbers()
	{
		this->RefineableSolidElement::assign_additional_local_eqn_numbers();
		this->register_c1_constraint_position_masters();
		// ConstrainFieldsToC1Space/ConstrainPositionsToC1Space ("additional dof constraints") locally
		// reduce a higher-order field/position to a C1/linear interpolation. This is now composed with
		// oomph-lib's native geometric hanging-node scheme (T-junctions on adaptively refined meshes):
		// a constrained dof, and any genuine hang that references a constrained master, are flattened
		// into real free leaf dofs in fill_hang_info_with_equations_{basebulk,for_pos} via
		// flatten_hang_for_value / flatten_hang_for_position (using each constrained node's precomputed
		// c1_constraint_corners). No separate equation numbers are required, because the flattened leaf
		// dofs are exactly the coarse vertices that oomph-lib already registered as hang masters. See
		// dev_docs/adaptive_refinement.md section 3.
		// std::cout << "ABOUT TO FILL ELEMINFO" << std::endl;
		fill_element_info();
		if (this->nnode())
		{
			oomph::TimeStepper *tstepper =  this->node_pt(0)->time_stepper_pt();		
			for (unsigned int i = 0; i < this->ninternal_data(); i++)
			{
				this->internal_data_pt(i)->set_time_stepper(tstepper, true);
			}
	   }
	}
}
