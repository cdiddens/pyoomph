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


// The hanging-node engine: hang-info accessors, the mixed-space (C1-in-C2) and mixed-quad hang
// construction, the flattened hang buffers handed to the JIT code, fill_hang_info_with_equations,
// and the Delaunay tesselation used to emit hanging elements for numpy/paraview output.

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


	// For every local node l, if it is a hanging node (its Lagrangian/Eulerian position is
	// constrained by master nodes rather than free, as happens on non-conforming refined meshes),
	// records its master nodes' weights and their local equation numbers for the *position* degrees
	// of freedom into shape_info->hanginfo_Pos[l]. This lets the generated ALE/solid-mechanics code
	// correctly distribute a hanging node's position-Jacobian contributions to its masters. Returns
	// true if any node in the element is hanging.
	// See dev_docs/adaptive_refinement.md. Central resolution of "which HangInfo governs
	// continuous space `space_info` at element-local node l_elem". Returns NULL if the node does not
	// hang in that space. This is the single seam that later phases will redirect to pyoomph-owned
	// named HangInfo pointers; for now it faithfully reproduces the per-space hangindex convention.
	oomph::HangInfo *BulkElementBase::hang_info_for_space(const JITFuncSpec_Table_FiniteElement_SpaceInfo_t *space_info, unsigned l_elem) const
	{
		const int hangindex = space_info->hangindex;
		oomph::Node *const &n = node_pt(l_elem);
		if (!n->is_hanging(hangindex))
			return NULL;
		return n->hanging_pt(hangindex);
	}

	bool BulkElementBase::node_hangs_in_space(const JITFuncSpec_Table_FiniteElement_SpaceInfo_t *space_info, unsigned l_elem) const
	{
		return node_pt(l_elem)->is_hanging(space_info->hangindex);
	}

	// Geometric (positional) hanging info for element-local node l_elem, or NULL if the node's
	// position does not hang. Mirrors the argument-less oomph-lib accessors (the info_Pos slot).
	oomph::HangInfo *BulkElementBase::hang_info_for_position(unsigned l_elem) const
	{
		oomph::Node *const &n = node_pt(l_elem);
		if (!n->is_hanging())
			return NULL;
		return n->hanging_pt();
	}

	bool BulkElementBase::node_is_c1_constrained_for_value(oomph::Node *n, unsigned v) const
	{
		Node *pn = dynamic_cast<Node *>(n);
		if (!pn) return false;
		for (const AdditionalDofConstrainingInfo *info = pn->get_additional_dof_constraints(); info != NULL; info = info->next)
			if (info->mode == CONTINUOUS_BASE_DOF_CONSTRAIN_TO_C1 && info->index == v)
				return true;
		return false;
	}

	bool BulkElementBase::node_is_c1_constrained_for_position(oomph::Node *n, unsigned i) const
	{
		Node *pn = dynamic_cast<Node *>(n);
		if (!pn) return false;
		for (const AdditionalDofConstrainingInfo *info = pn->get_additional_dof_constraints(); info != NULL; info = info->next)
			if (info->mode == POSITION_CONSTRAIN_TO_C1 && info->index == i)
				return true;
		return false;
	}

	int BulkElementBase::leaf_local_eqn_for_value(oomph::Node *n, unsigned v)
	{
		const unsigned nn = this->nnode();
		for (unsigned l = 0; l < nn; l++)
			if (node_pt(l) == n)
				return this->nodal_local_eqn(l, v);
		// External free master: its value was registered by oomph's assign_hanging_local_eqn_numbers
		// (as a master of some node genuinely hanging in v), so it has a local hang equation number.
		return this->local_hang_eqn(n, v);
	}

	// Read a node's local position-hang equation number, failing loudly if the node was never registered.
	//
	// oomph's local_position_hang_eqn() looks the node up in Local_position_hang_eqn with
	// std::map::operator[], which DEFAULT-CONSTRUCTS an empty DenseMatrix<int> when the node is absent.
	// Indexing that empty matrix is undefined behaviour; in practice it yields a junk equation number, which
	// surfaces much later and far from the cause as the generated code's "local_eqn < eleminfo->ndof"
	// assertion (dev_docs/adaptive_refinement.md §9.4). Registration happens in oomph's
	// assign_hanging_local_eqn_numbers, which walks only the GEOMETRIC hang of this element's OWN nodes, so
	// a node reached by any other route -- notably the C1 position-constraint redistribution, whose corners
	// may resolve onto coarse-side vertices of a 2:1 neighbour -- is legitimately absent, and that absence
	// is the bug. Naming the node here turns an unattributable abort into a diagnosis.
	// (The lookup itself inserts the empty entry as a side effect; harmless, since oomph clears the whole
	// map at the start of every assign_all_generic_local_eqn_numbers.)
	int BulkElementBase::position_hang_eqn_or_throw(oomph::Node *n, unsigned i, const std::string &context)
	{
		oomph::DenseMatrix<int> &pe = this->local_position_hang_eqn(n);
		if (pe.nrow() == 0 || i >= pe.ncol())
		{
			std::ostringstream oss;
			oss << "Position hang equation requested for a node that is not registered as a position-hang "
				   "master of this element (" << context << ").\n"
				   "Node at (";
			for (unsigned d = 0; d < n->ndim(); d++) oss << (d ? ", " : "") << n->x(d);
			oss << "), coordinate direction " << i << ".\n"
				   "oomph-lib registers position-hang masters only from the GEOMETRIC hang of this element's "
				   "own nodes, so this node was reached by some other route -- e.g. the C1 position-constraint "
				   "redistribution resolving onto a coarse-side vertex across a 2:1 interface. Without the "
				   "registration there is no local equation number for it, so the contribution cannot be "
				   "assembled. See dev_docs/adaptive_refinement.md §9.4.";
			throw_runtime_error(oss.str());
		}
		return pe(0, i);
	}

	int BulkElementBase::leaf_local_eqn_for_position(oomph::Node *n, unsigned i)
	{
		const unsigned nn = this->nnode();
		for (unsigned l = 0; l < nn; l++)
			if (node_pt(l) == n)
				return eleminfo.pos_local_eqn[l][i]; // as populated in fill_element_info / used by the JIT code
		return this->position_hang_eqn_or_throw(n, i, "flattening a hanging/constrained position");
	}

	// See the block comment on the declarations in elements.hpp. A value id addresses the continuous fields
	// in space order -- C2 and C2TB first, then C1 -- so anything at or beyond the end of the C2 block is a
	// C1 field and must be interpolated with the linear basis on the corner vertices.
	bool BulkElementBase::interpolation_value_is_C1(const int &value_id) const
	{
		if (value_id < 0) return false; // geometry/position: always the geometric basis
		const JITFuncSpec_Table_FiniteElement_t *ft = codeinst->get_func_table();
		return value_id >= static_cast<int>(ft->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk +
											ft->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk);
	}

	unsigned BulkElementBase::generic_ninterpolating_node(const int &value_id)
	{
		if (!this->interpolation_value_is_C1(value_id)) return this->nnode();
		return static_cast<unsigned>(this->get_nodal_space_index_to_element_index_map()[SPACE_INDEX_C1].size());
	}

	oomph::Node *BulkElementBase::generic_interpolating_node_pt(const unsigned &n, const int &value_id)
	{
		if (!this->interpolation_value_is_C1(value_id)) return this->node_pt(n);
		return this->node_pt(this->get_nodal_space_index_to_element_index_map()[SPACE_INDEX_C1][n]);
	}

	void BulkElementBase::generic_interpolating_basis(const oomph::Vector<double> &s, oomph::Shape &psi, const int &value_id) const
	{
		if (!this->interpolation_value_is_C1(value_id))
			this->shape(s, psi);
		else
			this->shape_at_s_C1(s, psi);
	}

	oomph::Node *BulkElementBase::generic_get_interpolating_node_at_local_coordinate(const oomph::Vector<double> &s, const int &value_id)
	{
		if (!this->interpolation_value_is_C1(value_id)) return this->get_node_at_local_coordinate(s);
		// Shape-agnostic: a C1 node sits at a vertex, so just compare local coordinates. This avoids
		// per-shape index arithmetic (which for a pyramid or a wedge is not a tensor product at all).
		const std::vector<unsigned> &c1 = this->get_nodal_space_index_to_element_index_map()[SPACE_INDEX_C1];
		const unsigned nd = this->dim();
		oomph::Vector<double> sl(nd);
		for (unsigned k = 0; k < c1.size(); k++)
		{
			this->local_coordinate_of_node(c1[k], sl);
			bool same = true;
			for (unsigned d = 0; d < nd && same; d++)
				same = (std::abs(sl[d] - s[d]) < 1e-10);
			if (same) return this->node_pt(c1[k]);
		}
		return 0;
	}

	bool BulkElementBase::mixed_hang_node_at(oomph::Node *X, oomph::RefineableElement *nb_re, const oomph::Vector<double> &s_nb, const int &value_id)
	{
		if (!X || !nb_re) return false;
		// Neighbour already owns an interpolating node here -> X is shared/real, not hanging.
		if (nb_re->get_interpolating_node_at_local_coordinate(s_nb, value_id) == X) return false;
		const unsigned nmax = nb_re->ninterpolating_node(value_id);
		oomph::Shape psi(nmax);
		nb_re->interpolating_basis(s_nb, psi, value_id);
		unsigned nmaster = 0;
		for (unsigned m = 0; m < nmax; m++)
			if (std::abs(psi[m]) > 1e-12) nmaster++;
		if (nmaster == 0) return false;
		// Cycle guard (as in the triangle/tet helpers): skip if a master transitively hangs back on X.
		std::function<bool(oomph::Node *, oomph::Node *, int, int)> reaches = [&](oomph::Node *from, oomph::Node *to, int slot, int depth) -> bool {
			if (from == to) return true;
			if (depth > 30 || !from->is_hanging(slot)) return false;
			oomph::HangInfo *h = from->hanging_pt(slot);
			for (unsigned k = 0; k < h->nmaster(); k++)
				if (reaches(h->master_node_pt(k), to, slot, depth + 1)) return true;
			return false;
		};
		for (unsigned m = 0; m < nmax; m++)
			if (std::abs(psi[m]) > 1e-12 && reaches(nb_re->interpolating_node_pt(m, value_id), X, value_id, 0)) return false;
		oomph::HangInfo *hang = new oomph::HangInfo(nmaster);
		unsigned mm = 0;
		for (unsigned m = 0; m < nmax; m++)
			if (std::abs(psi[m]) > 1e-12)
			{
				hang->set_master_node_pt(mm, nb_re->interpolating_node_pt(m, value_id), psi[m]);
				mm++;
			}
		X->set_hanging_pt(hang, value_id);
		return true;
	}

	bool BulkElementBase::mixed_hang_edge_node(oomph::Node *X, oomph::Node *Pb, oomph::Node *Qb, double t, oomph::RefineableElement *nb_root, const int &value_id)
	{
		oomph::FiniteElement *nb_fe = dynamic_cast<oomph::FiniteElement *>(nb_root);
		if (!nb_fe || !X || !Pb || !Qb) return false;
		const int iP = nb_fe->get_node_number(Pb), iQ = nb_fe->get_node_number(Qb);
		if (iP < 0 || iQ < 0) return false; // shared root-edge corners not both present in the neighbour ROOT
		const unsigned nd = nb_fe->dim();
		oomph::Vector<double> sP(nd), sQ(nd), s_root(nd);
		nb_fe->local_coordinate_of_node(iP, sP);
		nb_fe->local_coordinate_of_node(iQ, sQ);
		for (unsigned d = 0; d < nd; d++) s_root[d] = (1.0 - t) * sP[d] + t * sQ[d];
		// Descend the neighbour ROOT to the LEAF containing this shared-edge point (topological), then hang X
		// on that leaf's interpolating_basis. Handles a coarser neighbour whether it is the UNREFINED root or
		// a REFINED (>=1 level) leaf (a single-level 2:1 where the coarse side itself sits one level down).
		oomph::Vector<double> s_leaf(nd);
		oomph::RefineableElement *leaf = nullptr;
		if (oomph::RefineableTElement<2> *tri = dynamic_cast<oomph::RefineableTElement<2> *>(nb_root))
			leaf = tri->leaf_at_root_coordinate(s_root, s_leaf);
		else if (BulkElementBase *q = dynamic_cast<BulkElementBase *>(nb_root))
			leaf = q->quad_leaf_at_root_coordinate(s_root, s_leaf);
		if (!leaf) return false;
		// Hang X only on a STRICTLY COARSER neighbour leaf: an equal-level leaf shares X (node-sharing) and a
		// finer leaf hangs on THIS element from its own side. The shared-node check in mixed_hang_node_at
		// also catches the equal case, but the level guard makes the finer-neighbour case explicit + robust.
		oomph::RefineableElement *this_re = dynamic_cast<oomph::RefineableElement *>(this);
		if (this_re && leaf->refinement_level() >= this_re->refinement_level()) return false;
		return this->mixed_hang_node_at(X, leaf, s_leaf, value_id);
	}

	void BulkElementBase::mixed_quad_edge_hang(const int &value_id, const int &my_edge, oomph::RefineableElement *coarse_nb)
	{
		using namespace oomph::QuadTreeNames;
		oomph::RefineableElement *re = dynamic_cast<oomph::RefineableElement *>(this);
		if (!re || !re->tree_pt() || !coarse_nb) return;
		oomph::FiniteElement *root_fe = dynamic_cast<oomph::FiniteElement *>(re->tree_pt()->root_pt()->object_pt());
		if (!root_fe) return;
		const unsigned n = root_fe->nnode_1d();
		// The quad's son boxes are axis-aligned (no rotation), so my_edge keeps its compass direction all
		// the way to the root; the shared coarse edge's corner nodes are the root quad's corners on my_edge.
		oomph::Node *Pb = 0, *Qb = 0; // Pb at edge-fraction t=0, Qb at t=1
		switch (my_edge)
		{
			case S: Pb = root_fe->node_pt(0);           Qb = root_fe->node_pt(n - 1);       break;
			case N: Pb = root_fe->node_pt(n * (n - 1)); Qb = root_fe->node_pt(n * n - 1);   break;
			case W: Pb = root_fe->node_pt(0);           Qb = root_fe->node_pt(n * (n - 1)); break;
			case E: Pb = root_fe->node_pt(n - 1);       Qb = root_fe->node_pt(n * n - 1);   break;
			default: return;
		}
		const unsigned n_p = re->ninterpolating_node_1d(value_id);
		for (unsigned i0 = 0; i0 < n_p; i0++)
		{
			oomph::Vector<double> s_leaf(2);
			oomph::Node *X = 0;
			switch (my_edge)
			{
				case S: s_leaf[0] = -1.0 + 2.0 * re->local_one_d_fraction_of_interpolating_node(i0, 0, value_id); s_leaf[1] = -1.0; X = re->interpolating_node_pt(i0, value_id); break;
				case N: s_leaf[0] = -1.0 + 2.0 * re->local_one_d_fraction_of_interpolating_node(i0, 0, value_id); s_leaf[1] = 1.0;  X = re->interpolating_node_pt(i0 + n_p * (n_p - 1), value_id); break;
				case W: s_leaf[1] = -1.0 + 2.0 * re->local_one_d_fraction_of_interpolating_node(i0, 1, value_id); s_leaf[0] = -1.0; X = re->interpolating_node_pt(n_p * i0, value_id); break;
				case E: s_leaf[1] = -1.0 + 2.0 * re->local_one_d_fraction_of_interpolating_node(i0, 1, value_id); s_leaf[0] = 1.0;  X = re->interpolating_node_pt(n_p - 1 + n_p * i0, value_id); break;
				default: return;
			}
			if (!X) continue;
			// Ascend to the root frame by composing the axis-aligned son boxes (son s in [-1,1] maps to
			// s_father = offset(son_type) + 0.5*s, offsets +-0.5 per SW/SE/NW/NE quadrant).
			oomph::Vector<double> s = s_leaf;
			oomph::Tree *tr = re->tree_pt();
			while (tr->father_pt())
			{
				const int st = tr->son_type();
				const double xoff = (st == SE || st == NE) ? 0.5 : -0.5;
				const double yoff = (st == NW || st == NE) ? 0.5 : -0.5;
				s[0] = xoff + 0.5 * s[0];
				s[1] = yoff + 0.5 * s[1];
				tr = tr->father_pt();
			}
			const double t = (my_edge == S || my_edge == N) ? 0.5 * (s[0] + 1.0) : 0.5 * (s[1] + 1.0);
			this->mixed_hang_edge_node(X, Pb, Qb, t, coarse_nb, value_id);
		}
	}

	oomph::RefineableElement *BulkElementBase::quad_leaf_at_root_coordinate(const oomph::Vector<double> &s_root, oomph::Vector<double> &s_leaf)
	{
		using namespace oomph::QuadTreeNames;
		oomph::RefineableElement *re = dynamic_cast<oomph::RefineableElement *>(this);
		if (!re || !re->tree_pt()) return nullptr;
		oomph::Tree *node = re->tree_pt()->root_pt();
		oomph::Vector<double> s = s_root;
		while (!node->is_leaf())
		{
			const bool east = s[0] >= 0.0, north = s[1] >= 0.0; // pick the quadrant/son containing the point
			const int st = east ? (north ? NE : SE) : (north ? NW : SW);
			oomph::Tree *ch = node->son_pt(st);
			if (!ch) break;
			const double xoff = (st == SE || st == NE) ? 0.5 : -0.5;
			const double yoff = (st == NW || st == NE) ? 0.5 : -0.5;
			s[0] = 2.0 * (s[0] - xoff); // inverse of the son box map s_father = off + 0.5*s_son
			s[1] = 2.0 * (s[1] - yoff);
			node = ch;
		}
		s_leaf = s;
		return dynamic_cast<oomph::RefineableElement *>(node->object_pt());
	}

	// Search a quad subtree for a node at the point `s` (in the local frame of the subtree's own root
	// element). The counterpart of RefineableTElement<2>::node_in_subtree_at_local_coordinate, and
	// topological for the same reasons: the descent is the exact axis-aligned son-box map and the test is
	// get_node_at_local_coordinate, which compares LOCAL coordinates. Every level is tested, not just the
	// leaf (whether the point is a node depends on the level), and every son containing the point is
	// followed (a point on a son boundary belongs to both, and only one side may hold the node).
	static oomph::Node *quad_node_in_subtree(oomph::Tree *node, const oomph::Vector<double> &s)
	{
		using namespace oomph::QuadTreeNames;
		if (!node) return nullptr;
		oomph::FiniteElement *fe = dynamic_cast<oomph::FiniteElement *>(node->object_pt());
		oomph::RefineableElement *re = dynamic_cast<oomph::RefineableElement *>(node->object_pt());
		// Skip an element whose nodes are not built yet: its node_pt array is still null, and its built
		// ancestor (tested on the way down) holds the same node anyway.
		if (fe && re && re->nodes_built())
		{
			if (oomph::Node *n = fe->get_node_at_local_coordinate(s)) return n;
		}
		if (node->is_leaf()) return nullptr;
		const double tol = 1e-9;
		for (int st = 0; st < 4; st++)
		{
			oomph::Tree *ch = node->son_pt(st);
			if (!ch) continue;
			const double xoff = (st == SE || st == NE) ? 0.5 : -0.5;
			const double yoff = (st == NW || st == NE) ? 0.5 : -0.5;
			oomph::Vector<double> sc(2);
			sc[0] = 2.0 * (s[0] - xoff); // inverse of the son box map s_father = off + 0.5*s_son
			sc[1] = 2.0 * (s[1] - yoff);
			if (std::fabs(sc[0]) > 1.0 + tol || std::fabs(sc[1]) > 1.0 + tol) continue; // outside this son
			if (oomph::Node *n = quad_node_in_subtree(ch, sc)) return n;
		}
		return nullptr;
	}

	oomph::Node *BulkElementBase::quad_node_at_root_coordinate(const oomph::Vector<double> &s_root)
	{
		oomph::RefineableElement *re = dynamic_cast<oomph::RefineableElement *>(this);
		if (!re || !re->tree_pt()) return nullptr;
		return quad_node_in_subtree(re->tree_pt()->root_pt(), s_root);
	}

	oomph::Node *BulkElementBase::mixed_quad_shared_node(const oomph::Vector<double> &s_fraction)
	{
		using namespace oomph::QuadTreeNames;
		oomph::RefineableElement *re = dynamic_cast<oomph::RefineableElement *>(this);
		oomph::RefineableQElement<2> *qre = dynamic_cast<oomph::RefineableQElement<2> *>(this);
		if (!re || !qre || !re->tree_pt()) return nullptr;
		std::vector<int> edges;
		if (s_fraction[0] == 0.0) edges.push_back(W);
		if (s_fraction[0] == 1.0) edges.push_back(E);
		if (s_fraction[1] == 0.0) edges.push_back(S);
		if (s_fraction[1] == 1.0) edges.push_back(N);
		for (int my_edge : edges)
		{
			oomph::Vector<unsigned> tr(2);
			oomph::Vector<double> slo(2), shi(2);
			int ne, dl;
			bool intree;
			oomph::QuadTree *neigh = qre->quadtree_pt()->gteq_edge_neighbour(my_edge, tr, slo, shi, ne, dl, intree);
			if (!neigh) continue;
			oomph::RefineableTElement<2> *tri_root = dynamic_cast<oomph::RefineableTElement<2> *>(neigh->root_pt()->object_pt());
			if (!tri_root || !tri_root->nodes_built()) continue; // quad neighbour -> oomph base already handled it
			// Ascend THIS quad son to the root frame (axis-aligned son boxes) -> t along the root edge.
			oomph::Vector<double> s = {-1.0 + 2.0 * s_fraction[0], -1.0 + 2.0 * s_fraction[1]};
			oomph::Tree *trn = re->tree_pt();
			while (trn->father_pt())
			{
				const int st = trn->son_type();
				const double xo = (st == SE || st == NE) ? 0.5 : -0.5, yo = (st == NW || st == NE) ? 0.5 : -0.5;
				s[0] = xo + 0.5 * s[0];
				s[1] = yo + 0.5 * s[1];
				trn = trn->father_pt();
			}
			const double t = (my_edge == S || my_edge == N) ? 0.5 * (s[0] + 1.0) : 0.5 * (s[1] + 1.0);
			oomph::FiniteElement *qroot = dynamic_cast<oomph::FiniteElement *>(re->tree_pt()->root_pt()->object_pt());
			const unsigned n = qroot->nnode_1d();
			oomph::Node *Pb = 0, *Qb = 0; // Pb at t=0, Qb at t=1
			switch (my_edge)
			{
				case S: Pb = qroot->node_pt(0);           Qb = qroot->node_pt(n - 1);       break;
				case N: Pb = qroot->node_pt(n * (n - 1)); Qb = qroot->node_pt(n * n - 1);   break;
				case W: Pb = qroot->node_pt(0);           Qb = qroot->node_pt(n * (n - 1)); break;
				case E: Pb = qroot->node_pt(n - 1);       Qb = qroot->node_pt(n * n - 1);   break;
				default: continue;
			}
			oomph::FiniteElement *tri_root_fe = dynamic_cast<oomph::FiniteElement *>(neigh->root_pt()->object_pt());
			const int iP = tri_root_fe->get_node_number(Pb), iQ = tri_root_fe->get_node_number(Qb);
			if (iP < 0 || iQ < 0) continue;
			oomph::Vector<double> sP(2), sQ(2), s_troot(2);
			tri_root_fe->local_coordinate_of_node(iP, sP);
			tri_root_fe->local_coordinate_of_node(iQ, sQ);
			for (int d = 0; d < 2; d++) s_troot[d] = (1.0 - t) * sP[d] + t * sQ[d];
			oomph::Node *nn = tri_root->node_at_root_coordinate(s_troot);
			if (nn) return nn;
		}
		return nullptr;
	}

	// See declaration. Mirrors mixed_quad_shared_node's quad->tri blend but registers, for the tesselated-numpy
	// export, each quad edge node on the coarser neighbour at its topologically-computed coordinate there.
	void BulkElementBase::quad_register_on_coarser_for_numpy(std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes)
	{
		using namespace oomph::QuadTreeNames;
		oomph::RefineableElement *re = dynamic_cast<oomph::RefineableElement *>(this);
		oomph::RefineableQElement<2> *qre = dynamic_cast<oomph::RefineableQElement<2> *>(this);
		if (!re || !qre || !re->tree_pt())
			return;
		oomph::FiniteElement *qroot = dynamic_cast<oomph::FiniteElement *>(re->tree_pt()->root_pt()->object_pt());
		const int edges[4] = {S, N, W, E};
		for (int my_edge : edges)
		{
			oomph::Vector<unsigned> translate_s(2);
			oomph::Vector<double> s_lo(2), s_hi(2);
			int neigh_edge, diff_level;
			bool in_tree;
			oomph::QuadTree *neigh = qre->quadtree_pt()->gteq_edge_neighbour(my_edge, translate_s, s_lo, s_hi, neigh_edge, diff_level, in_tree);
			if (!neigh || diff_level == 0)
				continue; // no strictly-coarser neighbour across this edge
			BulkElementBase *coarse = dynamic_cast<BulkElementBase *>(neigh->object_pt());
			oomph::RefineableTElement<2> *coarse_tri = dynamic_cast<oomph::RefineableTElement<2> *>(neigh->object_pt());
			// Cross-shape (coarser tri) shared-root-edge corner nodes, computed once per edge.
			oomph::RefineableTElement<2> *tri_root = nullptr;
			oomph::FiniteElement *tri_root_fe = nullptr;
			oomph::Node *Pb = nullptr, *Qb = nullptr;
			if (coarse_tri)
			{
				tri_root = dynamic_cast<oomph::RefineableTElement<2> *>(neigh->root_pt()->object_pt());
				tri_root_fe = dynamic_cast<oomph::FiniteElement *>(neigh->root_pt()->object_pt());
				if (!tri_root || !tri_root_fe || !tri_root->nodes_built())
					continue;
				const unsigned n = qroot->nnode_1d();
				switch (my_edge)
				{
				case S: Pb = qroot->node_pt(0);           Qb = qroot->node_pt(n - 1);      break;
				case N: Pb = qroot->node_pt(n * (n - 1)); Qb = qroot->node_pt(n * n - 1);  break;
				case W: Pb = qroot->node_pt(0);           Qb = qroot->node_pt(n * (n - 1)); break;
				case E: Pb = qroot->node_pt(n - 1);       Qb = qroot->node_pt(n * n - 1);  break;
				}
			}
			for (unsigned li = 0; li < this->nnode(); li++)
			{
				oomph::Vector<double> s_node(2);
				this->local_coordinate_of_node(li, s_node);
				oomph::Vector<double> s_frac = {0.5 * (s_node[0] + 1.0), 0.5 * (s_node[1] + 1.0)};
				bool on_edge;
				if (my_edge == S)      on_edge = std::abs(s_frac[1]) < 1e-9;
				else if (my_edge == N) on_edge = std::abs(s_frac[1] - 1.0) < 1e-9;
				else if (my_edge == W) on_edge = std::abs(s_frac[0]) < 1e-9;
				else                   on_edge = std::abs(s_frac[0] - 1.0) < 1e-9;
				if (!on_edge)
					continue;
				if (coarse_tri)
				{
					// Ascend this quad son to the root frame -> parameter t along the shared root edge.
					oomph::Vector<double> s = {s_node[0], s_node[1]};
					oomph::Tree *trn = re->tree_pt();
					while (trn->father_pt())
					{
						const int st = trn->son_type();
						const double xo = (st == SE || st == NE) ? 0.5 : -0.5, yo = (st == NW || st == NE) ? 0.5 : -0.5;
						s[0] = xo + 0.5 * s[0];
						s[1] = yo + 0.5 * s[1];
						trn = trn->father_pt();
					}
					const double t = (my_edge == S || my_edge == N) ? 0.5 * (s[0] + 1.0) : 0.5 * (s[1] + 1.0);
					const int iP = tri_root_fe->get_node_number(Pb), iQ = tri_root_fe->get_node_number(Qb);
					if (iP < 0 || iQ < 0)
						continue;
					oomph::Vector<double> sP(2), sQ(2), s_troot(2), s_leaf(2);
					tri_root_fe->local_coordinate_of_node(iP, sP);
					tri_root_fe->local_coordinate_of_node(iQ, sQ);
					for (int d = 0; d < 2; d++) s_troot[d] = (1.0 - t) * sP[d] + t * sQ[d];
					oomph::RefineableElement *tleaf = tri_root->leaf_at_root_coordinate(s_troot, s_leaf);
					BulkElementBase *tleaf_be = dynamic_cast<BulkElementBase *>(tleaf);
					if (tleaf_be)
						tleaf_be->add_node_from_finer_neighbor_for_tesselated_numpy(s_leaf, this->node_pt(li), add_nodes);
				}
				else if (coarse)
				{
					// Coarser quad: gteq_edge_neighbour's own coordinate mapping into the neighbour frame.
					oomph::Vector<double> s_in_neigh(2);
					for (int i = 0; i < 2; i++)
						s_in_neigh[i] = s_lo[i] + s_frac[translate_s[i]] * (s_hi[i] - s_lo[i]);
					coarse->add_node_from_finer_neighbor_for_tesselated_numpy(s_in_neigh, this->node_pt(li), add_nodes);
				}
			}
		}
	}

	void BulkElementBase::flatten_hang_for_value(oomph::Node *n, unsigned v, double weight, std::map<int, double> &out, int depth)
	{
		if (depth > 32)
			throw_runtime_error("flatten_hang_for_value: recursion too deep - cyclic hang/constraint chain?");
		Node *pn = dynamic_cast<Node *>(n);
		// 1. Locally reduced to C1 for this value: expand via the stored C1-corner average, then recurse
		//    (a corner may itself be constrained and/or hanging).
		if (pn && !pn->c1_constraint_corners.empty() && node_is_c1_constrained_for_value(pn, v))
		{
			for (const auto &cw : pn->c1_constraint_corners)
				flatten_hang_for_value(cw.first, v, weight * cw.second, out, depth + 1);
			return;
		}
		// 2. Genuinely hanging in this value: expand via the oomph hang masters (each recursed, since a
		//    master may be a constrained node that must be flattened further).
		if (n->is_hanging(v))
		{
			oomph::HangInfo *h = n->hanging_pt(v);
			const unsigned nm = h->nmaster();
			for (unsigned m = 0; m < nm; m++)
				flatten_hang_for_value(h->master_node_pt(m), v, weight * h->master_weight(m), out, depth + 1);
			return;
		}
		// 3. Real leaf: a free dof contributes; a genuinely pinned (e.g. Dirichlet) leaf drops out.
		const int le = leaf_local_eqn_for_value(n, v);
		if (le >= 0)
			out[le] += weight;
	}

	void BulkElementBase::flatten_hang_for_position(oomph::Node *n, unsigned i, double weight, std::map<int, double> &out, int depth)
	{
		if (depth > 32)
			throw_runtime_error("flatten_hang_for_position: recursion too deep - cyclic hang/constraint chain?");
		Node *pn = dynamic_cast<Node *>(n);
		if (pn && !pn->c1_constraint_corners.empty() && node_is_c1_constrained_for_position(pn, i))
		{
			for (const auto &cw : pn->c1_constraint_corners)
				flatten_hang_for_position(cw.first, i, weight * cw.second, out, depth + 1);
			return;
		}
		if (n->is_hanging()) // geometric (positional) hang
		{
			oomph::HangInfo *h = n->hanging_pt();
			const unsigned nm = h->nmaster();
			for (unsigned m = 0; m < nm; m++)
				flatten_hang_for_position(h->master_node_pt(m), i, weight * h->master_weight(m), out, depth + 1);
			return;
		}
		const int le = leaf_local_eqn_for_position(n, i);
		if (le >= 0)
			out[le] += weight;
	}

	double BulkElementBase::flattened_value(oomph::Node *n, unsigned v, unsigned t, int depth)
	{
		if (depth > 32)
			throw_runtime_error("flattened_value: recursion too deep - cyclic hang/constraint chain?");
		Node *pn = dynamic_cast<Node *>(n);
		if (pn && !pn->c1_constraint_corners.empty() && node_is_c1_constrained_for_value(pn, v))
		{
			double val = 0.0;
			for (const auto &cw : pn->c1_constraint_corners)
				val += cw.second * flattened_value(cw.first, v, t, depth + 1);
			return val;
		}
		if (n->is_hanging(v))
		{
			oomph::HangInfo *h = n->hanging_pt(v);
			const unsigned nm = h->nmaster();
			double val = 0.0;
			for (unsigned m = 0; m < nm; m++)
				val += h->master_weight(m) * flattened_value(h->master_node_pt(m), v, t, depth + 1);
			return val;
		}
		return n->raw_value(t, v); // real free leaf: its stored value is the current solution
	}

	double BulkElementBase::flattened_position(oomph::Node *n, unsigned i, unsigned t, int depth)
	{
		if (depth > 32)
			throw_runtime_error("flattened_position: recursion too deep - cyclic hang/constraint chain?");
		Node *pn = dynamic_cast<Node *>(n);
		if (pn && !pn->c1_constraint_corners.empty() && node_is_c1_constrained_for_position(pn, i))
		{
			double val = 0.0;
			for (const auto &cw : pn->c1_constraint_corners)
				val += cw.second * flattened_position(cw.first, i, t, depth + 1);
			return val;
		}
		if (n->is_hanging())
		{
			oomph::HangInfo *h = n->hanging_pt();
			const unsigned nm = h->nmaster();
			double val = 0.0;
			for (unsigned m = 0; m < nm; m++)
				val += h->master_weight(m) * flattened_position(h->master_node_pt(m), i, t, depth + 1);
			return val;
		}
		return n->x(t, i); // real free leaf
	}

	// True if none of a genuine hang's master nodes carries any additional dof constraint. Masters are
	// non-hanging by construction, so when this holds the flattened expansion is exactly the plain hang
	// and the hangbuffer can be written directly, skipping the std::map allocation. This keeps
	// conventional (constraint-free) adaptive assembly as fast as before the constraint composition.
	static bool hang_masters_are_unconstrained(oomph::HangInfo *h)
	{
		const unsigned nm = h->nmaster();
		for (unsigned m = 0; m < nm; m++)
		{
			pyoomph::Node *mn = dynamic_cast<pyoomph::Node *>(h->master_node_pt(m));
			if (mn && mn->get_additional_dof_constraints() != NULL)
				return false;
		}
		return true;
	}

	// Write a flattened (local_eqn -> weight) map into a single hangbuffer entry. Near-zero weights are
	// dropped to keep the master list within the fixed MAX_HANG capacity of the JIT buffer.
	static void write_flattened_hangbuffer(JITHangInfo_t &hb, const std::map<int, double> &out)
	{
		const int MAX_HANG_LOCAL = 16; // must match the MAX_HANG used to allocate masters[] in fill_element_info
		hb.nummaster = 0;
		for (const auto &kv : out)
		{
			if (kv.second > -1e-15 && kv.second < 1e-15)
				continue;
			if (hb.nummaster >= MAX_HANG_LOCAL)
				throw_runtime_error("Flattened hanging/constraint expansion exceeds MAX_HANG master slots. "
				                    "This can happen with deep refinement inside a ConstrainFieldsToC1Space region; "
				                    "please report with a reproducing script.");
			hb.masters[hb.nummaster].local_eqn = kv.first;
			hb.masters[hb.nummaster].weight = kv.second;
			hb.nummaster++;
		}
	}

	bool BulkElementBase::fill_hang_info_with_equations_for_pos(JITShapeInfo_t *shape_info)
	{
		bool res = false;
		for (unsigned int f = 0; f < this->nodal_dimension(); f++)
		{
			for (unsigned int l = 0; l < eleminfo.nnode; l++)
			{
				oomph::Node *n = node_pt(l);
				// The position needs redistribution iff the node is geometrically hanging or its position
				// was locally reduced to C1 (ConstrainPositionsToC1Space). A *field-only* constraint on a
				// master node is irrelevant here (position is not constrained), so flatten_hang_for_position
				// keys purely on POSITION_CONSTRAIN_TO_C1 and the geometric hang.
				if (n->is_hanging())
				{
					oomph::HangInfo *h = n->hanging_pt();
					if ((!has_additional_dof_constraints || !this->node_is_c1_constrained_for_position(n, f)) && hang_masters_are_unconstrained(h))
					{
						// Fast path: plain geometric hang, no constraints to compose.
						JITHangInfo_t &hb = shape_info->hanginfo_Pos[f][l];
						hb.nummaster = h->nmaster();
						for (unsigned m = 0; m < h->nmaster(); m++)
						{
							hb.masters[m].weight = h->master_weight(m);
							hb.masters[m].local_eqn = this->position_hang_eqn_or_throw(
								h->master_node_pt(m), f, "geometric position hang master");
						}
					}
					else
					{
						std::map<int, double> out;
						this->flatten_hang_for_position(n, f, 1.0, out, 0);
						write_flattened_hangbuffer(shape_info->hanginfo_Pos[f][l], out);
					}
					res = true;
				}
				else if (has_additional_dof_constraints && this->node_is_c1_constrained_for_position(n, f))
				{
					std::map<int, double> out;
					this->flatten_hang_for_position(n, f, 1.0, out, 0);
					write_flattened_hangbuffer(shape_info->hanginfo_Pos[f][l], out);
					res = true;
				}
				else
				{
					shape_info->hanginfo_Pos[f][l].nummaster = 0;
				}
			}
		}

		return res;
	}

	// Analogous to fill_hang_info_with_equations_for_pos, but for the ordinary field values ("base
	// bulk" fields, i.e. not the interface-only additional fields) of each continuous interpolation
	// space present on this element: for every node hanging in that space, records its masters'
	// weights and local equation numbers into shape_info->hanginfo[buffer_offset_basebulk+f]. Used by
	// the generated residual code to correctly assemble Jacobian contributions of hanging field dofs.
	bool BulkElementBase::fill_hang_info_with_equations_basebulk(JITShapeInfo_t *shape_info)
	{
		bool res=false;
		auto * ft = codeinst->get_func_table();
		for (unsigned int ispace=0;ispace<ft->num_present_continuous_spaces;ispace++)
		{
			const JITFuncSpec_Table_FiniteElement_SpaceInfo_t * space_info = ft->present_continuous_spaces[ispace];
			unsigned nnode_space=eleminfo.nnode_of_space[space_info->space_index];
			const std::vector<unsigned> & space_node_to_elem_node=this->get_nodal_space_index_to_element_index_map()[space_info->space_index];
			for (unsigned int f = 0; f < space_info->numfields_basebulk; f++)
			{
				const unsigned v = f+space_info->nodal_offset_basebulk;
				JITHangInfo_t * hangbuffer=shape_info->hanginfo[space_info->buffer_offset_basebulk+f];
				for (unsigned int l = 0; l < nnode_space; l++)
				{
					const unsigned l_elem=space_node_to_elem_node[l];
					oomph::Node *n = node_pt(l_elem);
					// A node's value needs redistribution iff it is genuinely hanging in this value or it
					// was locally reduced to C1 (a constraint). Both are handled uniformly by flattening
					// into real free leaf dofs (which also composes the two on refined constraint regions).
					if (n->is_hanging(v))
					{
						oomph::HangInfo *h = n->hanging_pt(v);
						// Fast path (the conventional case): the node is not itself constrained and none of
						// its masters is, so the flattened expansion is exactly the plain hang. The
						// node-constraint test is gated on has_additional_dof_constraints so a
						// constraint-free element pays nothing for it.
						if ((!has_additional_dof_constraints || !this->node_is_c1_constrained_for_value(n, v)) && hang_masters_are_unconstrained(h))
						{
							hangbuffer[l].nummaster = h->nmaster();
							for (unsigned m = 0; m < h->nmaster(); m++)
							{
								hangbuffer[l].masters[m].weight = h->master_weight(m);
								hangbuffer[l].masters[m].local_eqn = this->local_hang_eqn(h->master_node_pt(m), v);
							}
						}
						else
						{
							std::map<int, double> out;
							this->flatten_hang_for_value(n, v, 1.0, out, 0);
							write_flattened_hangbuffer(hangbuffer[l], out);
						}
						res = true;
					}
					else if (has_additional_dof_constraints && this->node_is_c1_constrained_for_value(n, v))
					{
						std::map<int, double> out;
						this->flatten_hang_for_value(n, v, 1.0, out, 0);
						write_flattened_hangbuffer(hangbuffer[l], out);
						res = true;
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

	// Sets up a synthetic hanging-node scheme for base-bulk continuous-field dofs that were locally
	// reduced to C1 via NodeWithFieldIndicesBase::add_additional_dof_constraint (mode
	// CONTINUOUS_BASE_DOF_CONSTRAIN_TO_C1): such a dof's node is already pinned (see
	// BulkElementBase::pin_dummy_values), so here it instead gets masters = the surrounding C1 corner
	// nodes of this element (found the same way as for ordinary dummy values, see
	// Dummy_Value_Interpolation_Map[SPACE_INDEX_C1]) with equal weights, and their local equation
	// numbers for the very same field, so the generated code redistributes residual/Jacobian
	// contributions there exactly as it does for genuinely hanging nodes.
	bool BulkElementBase::fill_additional_hang_buffer_data(JITShapeInfo_t *shape_info)
	{
		// The base-bulk field and position C1-constraint handling that used to live here (a one-level
		// C1-corner average with limited chaining) is now performed -- and correctly composed with
		// genuine adaptive hanging -- by fill_hang_info_with_equations_basebulk / _for_pos via
		// flatten_hang_for_value / flatten_hang_for_position. This base version is intentionally a no-op.
		// The InterfaceElementBase override still adds INTERFACE_DOF_CONSTRAIN_TO_C1 handling on top.
		(void)shape_info;
		return false;
	}

	// After hanging-node constraints/master nodes have been set up (or changed, e.g. post-refinement),
	// pushes consistent values into the hanging nodes' own storage: (1) positions, via oomph-lib's
	// position() which already accounts for hanging-node interpolation; (2) field values of every
	// continuous space's base-bulk fields, interpolated from the master nodes; and (3) "dummy"
	// values (e.g. a C1 field's value at a C2-only node, which has no direct equation of its own)
	// by averaging the surrounding real nodes' values, since dummy nodes are never themselves
	// hanging but must still hold sensible values for consistent interpolation/output.
	void BulkElementBase::interpolate_hang_values()
	{

		auto * ft = codeinst->get_func_table();

		// Positions: geometrically hanging nodes, and nodes whose position was locally reduced to C1
		// (ConstrainPositionsToC1Space). Both are flattened down to real free leaf dofs, so the pushed
		// value is order-independent (leaves always hold current data) and consistent with the hangbuffer.
		// Only the current position is pushed, matching the previous behaviour.
		for (unsigned l = 0; l < this->nnode(); l++)
		{
			oomph::Node *n = node_pt(l);
			for (unsigned i = 0; i < n->ndim(); i++)
			{
				if (n->is_hanging() || this->node_is_c1_constrained_for_position(n, i))
					n->x(i) = this->flattened_position(n, i, 0, 0);
			}
		}

		// Base-bulk field values of every continuous space: a node that is genuinely hanging in a value,
		// or whose value was locally reduced to C1 (ConstrainFieldsToC1Space), gets the flattened value
		// pushed into its raw storage for every history level.
		const std::vector<std::vector<unsigned>> & space_node_to_elem_node_map = this->get_nodal_space_index_to_element_index_map();
		const std::vector<std::vector<std::vector<unsigned>>> & dummy_value_interpolation_map = this->get_dummy_value_interpolation_map();
		for (unsigned ispace = 0; ispace < ft->num_present_continuous_spaces; ispace++)
		{
			const JITFuncSpec_Table_FiniteElement_SpaceInfo_t * space_info = ft->present_continuous_spaces[ispace];
			const unsigned nnode_space = eleminfo.nnode_of_space[space_info->space_index];
			const std::vector<unsigned> & space_node_to_elem_node = space_node_to_elem_node_map[space_info->space_index];
			for (unsigned l = 0; l < nnode_space; l++)
			{
				const unsigned l_elem = space_node_to_elem_node[l];
				oomph::Node *n = node_pt(l_elem);
				const unsigned ntstorage = n->ntstorage();
				for (unsigned f = 0; f < space_info->numfields_basebulk; f++)
				{
					const unsigned v = space_info->nodal_offset_basebulk + f;
					if (n->is_hanging(v) || this->node_is_c1_constrained_for_value(n, v))
					{
						for (unsigned t = 0; t < ntstorage; t++)
							n->value_pt(v)[t] = this->flattened_value(n, v, t, 0);
					}
				}
			}
			// Dummy values (e.g. a C1 field stored on C2-only nodes): the average of the C1 corner nodes,
			// each flattened so the average stays correct when a corner itself hangs or is constrained.
			const std::vector<std::vector<unsigned>> & dummy_value_interpolation = dummy_value_interpolation_map[space_info->space_index];
			for (const std::vector<unsigned> & interp : dummy_value_interpolation)
			{
				if (interp.size() < 2) continue;
				const unsigned nmaster = interp.size() - 1;
				oomph::Node *tgt = node_pt(interp[0]);
				const unsigned ntstorage = tgt->ntstorage();
				for (unsigned f = 0; f < space_info->numfields_basebulk; f++)
				{
					const unsigned v = space_info->nodal_offset_basebulk + f;
					for (unsigned t = 0; t < ntstorage; t++)
					{
						double val = 0.0;
						for (unsigned m = 1; m < interp.size(); m++)
							val += this->flattened_value(node_pt(interp[m]), v, t, 0);
						tgt->value_pt(v)[t] = val / (double)nmaster;
					}
				}
			}
		}
	}

	// Looks up field "name" across all continuous interpolation spaces (both ordinary base-bulk
	// fields and, for interface elements, the additional interface-only fields) present on this
	// element, and returns the (Data*, component-index) pair for every node carrying that field.
	// If use_elemental_indices is true, the result is a dense array of length nnode() indexed by
	// local element node index (with (nullptr,-1) for nodes not carrying the field); otherwise it is
	// a compact list over just the nodes of the field's own interpolation space.
	std::vector<std::pair<oomph::Data*,int> > BulkElementBase::get_field_data_list(std::string name,bool use_elemental_indices)
	{
	 auto *ft=codeinst->get_func_table();
	 std::vector<std::pair<oomph::Data*,int> > result;
	 auto find_by_name=[name](char **fnames,unsigned numf)->int {for(unsigned int i=0;i<numf;i++) if (name==std::string(fnames[i])) return i;  return -1;};

	 for (unsigned int ispace=0;ispace<ft->num_present_continuous_spaces;ispace++)
	 {
		const JITFuncSpec_Table_FiniteElement_SpaceInfo_t * space_info = ft->present_continuous_spaces[ispace];
		// Basebulk fields of continuous spaces
		for (unsigned int f=0;f<space_info->numfields_basebulk;f++)
		{
			if (name==std::string(space_info->fieldnames[f]))
			{
				std::vector<unsigned> space_node_to_elem_node=this->get_nodal_space_index_to_element_index_map()[space_info->space_index];
				if (!use_elemental_indices)
				{					
					for (unsigned int i=0;i<eleminfo.nnode_of_space[space_info->space_index];i++)
					{
						result.push_back(std::make_pair(this->node_pt(space_node_to_elem_node[i]),f+space_info->nodal_offset_basebulk));
					}					
				}
				else
				{					
					result.resize(eleminfo.nnode,std::make_pair(nullptr,-1));
					for (unsigned int i=0;i<eleminfo.nnode_of_space[space_info->space_index];i++)
					{
						int nind=space_node_to_elem_node[i];
						result[nind]=std::make_pair(this->node_pt(nind),f+space_info->nodal_offset_basebulk);						
					}
				}
				return result;
			}
		}
		// Additional interface fields of continuous spaces
		for (unsigned int i=space_info->numfields_basebulk;i<space_info->numfields;i++)
		{
			if (name==std::string(space_info->fieldnames[i]))
			{
				unsigned interf_id = space_info->interface_dof_indices[i-space_info->numfields_basebulk];
				std::vector<unsigned> space_node_to_elem_node=this->get_nodal_space_index_to_element_index_map()[space_info->space_index];
				if (!use_elemental_indices)
				{					
					for (unsigned int i=0;i<eleminfo.nnode_of_space[space_info->space_index];i++)
					{
						auto *n=this->node_pt(space_node_to_elem_node[i]);
						pyoomph::BoundaryNode *bn=dynamic_cast<pyoomph::BoundaryNode *>(n);
						if (!bn) throw_runtime_error("Node is not a boundary node, but the field is an interface field");
						result.push_back(std::make_pair(n,bn->index_of_first_value_assigned_by_face_element(interf_id)));
					}					
				}
				else
				{					
					result.resize(eleminfo.nnode,std::make_pair(nullptr,-1));
					for (unsigned int i=0;i<eleminfo.nnode_of_space[space_info->space_index];i++)
					{
						int nind=space_node_to_elem_node[i];
						auto *n=this->node_pt(nind);
						pyoomph::BoundaryNode *bn=dynamic_cast<pyoomph::BoundaryNode *>(n);
						if (!bn) throw_runtime_error("Node is not a boundary node, but the field is an interface field");
						result[nind]=std::make_pair(n,bn->index_of_first_value_assigned_by_face_element(interf_id));						
					}
				}
				return result;
			}
		}
		
	 }

	 // TODO: DG loop here
	 int ind=-1;
	 const std::vector<std::vector<int>> & elem_to_space_index_map=this->get_element_index_to_nodal_space_index_map();
	 for (unsigned int ispace=0;ispace<ft->num_present_dg_spaces;ispace++)
	 {
		auto * space_info = ft->present_dg_spaces[ispace];
		if (space_info->numfields>0 && ((ind=find_by_name(space_info->fieldnames,space_info->numfields))>=0))
		{
			oomph::Data * data=this->get_DG_nodal_data(space_info->space_index, ind);
			if (!use_elemental_indices)
			{
				for (unsigned int i=0;i<eleminfo.nnode_of_space[space_info->space_index];i++) 
				{
					result.push_back(std::make_pair(data,this->get_DG_node_index(space_info->space_index, ind, i)));
				}
			}
			else
			{
				for (unsigned int i=0;i<eleminfo.nnode;i++) 
				{
					int nind=elem_to_space_index_map[space_info->space_index][i];
					if (nind>=0)
					{
						result.push_back(std::make_pair(data,nind));
					}			
					else
					{
						result.push_back(std::make_pair(nullptr,-1));
					}
				}
			}
			return result;
		}
	 }	 

	 if (name=="mesh_x" && this->nodal_dimension()>0)
	 {
		for (unsigned int i=0;i<this->nnode();i++) result.push_back(std::make_pair(dynamic_cast<pyoomph::Node*>(this->node_pt(i))->variable_position_pt(),0));
	 }
	 else if (name=="mesh_y"  && this->nodal_dimension()>1)
	 {
		for (unsigned int i=0;i<this->nnode();i++) result.push_back(std::make_pair(dynamic_cast<pyoomph::Node*>(this->node_pt(i))->variable_position_pt(),1));
	 }
	 else if (name=="mesh_z"  && this->nodal_dimension()>2)
	 {
		for (unsigned int i=0;i<this->nnode();i++) result.push_back(std::make_pair(dynamic_cast<pyoomph::Node*>(this->node_pt(i))->variable_position_pt(),2));
	 }
	 else
	 {
	 	throw_runtime_error("Cannot get data of field "+name);	 
	 }
	 return result;
	}


	// Builds the complete hanging-node bookkeeping (positions, base-bulk continuous fields, and
	// discontinuous fields which are never hanging) for this element, then - if eqn_remap is given -
	// reinterprets/abuses that same hanging-info data structure to additionally carry local
	// equation number *remapping*: this is used when this element's shapes are evaluated on behalf
	// of another element (e.g. an interface element pulling in its attached bulk element's shapes as
	// external data) whose local equation numbering differs from this element's own. For every
	// local dof, if it is not actually hanging it is turned into a trivial "1 master, weight 1"
	// hanging entry pointing at the remapped equation number (eqn_remap[old_local_eqn]); genuinely
	// hanging dofs have each of their masters' equation numbers remapped the same way. A remapped
	// value of -666 signals a missing external dependency and raises a descriptive error, since it
	// means the generated code expects a dependency that was never registered as external data.
	bool BulkElementBase::fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap)
	{
		bool res=this->fill_hang_info_with_equations_for_pos(shape_info); // Potentially only do if required
		res=this->fill_hang_info_with_equations_basebulk(shape_info) || res;
		res=this->fill_hang_info_with_equations_interface(shape_info) || res;
		auto * ft=codeinst->get_func_table();
		// DG spaces, DL and D0 fields can never actually be hanging; their hanginfo slots in the
		// unified per-field buffer are only (ab)used below for the eqn_remap indirection, so start
		// from a clean nummaster=0 state (mirrors what used to be a single flat hanginfo_Discont zero).
		for (unsigned int ispace=0;ispace<ft->num_present_dg_spaces;ispace++)
		{
			auto * space_info = ft->present_dg_spaces[ispace];
			unsigned nnode_space=eleminfo.nnode_of_space[space_info->space_index];
			for (unsigned int f = 0; f < space_info->numfields_basebulk; f++)
			{
				JITHangInfo_t * hangbuffer=shape_info->hanginfo[space_info->buffer_offset_basebulk+f];
				for (unsigned int l = 0; l < nnode_space; l++)
				{
					hangbuffer[l].nummaster=0;
				}
			}
			for (unsigned int f = 0; f < space_info->numfields-space_info->numfields_basebulk; f++)
			{
				JITHangInfo_t * hangbuffer=shape_info->hanginfo[space_info->buffer_offset_interf+f];
				for (unsigned int l = 0; l < nnode_space; l++)
				{
					hangbuffer[l].nummaster=0;
				}
			}
		}
		for (unsigned int f = 0; f < ft->info_DL.numfields; f++)
		{
			JITHangInfo_t * hangbuffer=shape_info->hanginfo[ft->info_DL.buffer_offset_basebulk+f];
			for (unsigned int l = 0; l < eleminfo.nnode_DL; l++)
			{
				hangbuffer[l].nummaster=0;
			}
		}
		for (unsigned int f = 0; f < ft->info_D0.numfields; f++)
		{
			shape_info->hanginfo[ft->info_D0.buffer_offset_basebulk+f][0].nummaster=0;
		}

		// Additionally set up hanging for dofs locally reduced to C1 via add_additional_dof_constraint
		res=this->fill_additional_hang_buffer_data(shape_info) || res;

		if (eqn_remap)
		{
		   // If we access e.g. a bulk element from an interface element, we have to remap the local equations, since the interface element has a different local equation numbering.
		   // This is done via the hanging information, which is abused here to store the remapped local equations.
		   auto * ft=codeinst->get_func_table();
			// If the mesh moves, we have to setup the mapping in the hanging scheme
			if (ft->moving_nodes)
			{
				bool require_dx_terms=false;
				for (unsigned int i_space=0;i_space<ft->num_present_continuous_spaces;i_space++)
				{
					if (required.continuous_spaces[ft->present_continuous_spaces[i_space]->space_index].dx_psi)
					{
						require_dx_terms=true;
						break;
					}
				}
				// Pos.dx_psi/Pos.dX_psi must be included: an interface term acting on gradients of the bulk
				// position test function (e.g. InterfaceMeshStiffening) touches bulk_eleminfo->pos_local_eqn
				// without ever requiring the bare Pos.psi, and used to end up with unremapped local equations.
				if (require_dx_terms ||   required.Pos.psi || required.Pos.dx_psi || required.Pos.dX_psi || required.DL.dx_psi || required.normal || required.elemsize_Eulerian || required.elemsize_Eulerian_cartesian)
				{

					unsigned nfields = this->nodal_dimension();

					for (unsigned int f = 0; f < nfields; f++)
					{
						for (unsigned int l = 0; l < eleminfo.nnode; l++)
						{
							if (!shape_info->hanginfo_Pos[f][l].nummaster)
							{
								// NON HANGING -> Set the hanging to 1 node, which is just the remapped equation
								shape_info->hanginfo_Pos[f][l].nummaster = 1;
								shape_info->hanginfo_Pos[f][l].masters[0].weight = 1.0;
								if (eleminfo.pos_local_eqn[l][f] >= 0)
								{
									shape_info->hanginfo_Pos[f][l].masters[0].local_eqn = eleminfo.pos_local_eqn[l][f];
								}
								else
								{
									shape_info->hanginfo_Pos[f][l].masters[0].local_eqn = -1;
								}
							}
							// Now remap the local equations to the interface element numbering
							for (int m = 0; m < shape_info->hanginfo_Pos[f][l].nummaster; m++)
							{
								if (shape_info->hanginfo_Pos[f][l].masters[m].local_eqn >= 0)
								{
									shape_info->hanginfo_Pos[f][l].masters[m].local_eqn = eqn_remap[shape_info->hanginfo_Pos[f][l].masters[m].local_eqn];
									if (shape_info->hanginfo_Pos[f][l].masters[m].local_eqn == -666)
									{
										std::ostringstream oss;
										oss << this;
										throw_runtime_error("MISSING EXTERNAL POS DEPENDENCY ON ELEM PTR: " + oss.str() + "\nThis is part of the Lagrangian field index " + std::to_string(f) + " of " + std::to_string(nfields) + " at node " + std::to_string(l)+ " of "+std::to_string(eleminfo.nnode)+"\n"+"This can happen when you add additional fields or residualsfrom both sides of a single interface. Please only add fields and residuals from one side.");
									}
								}
							}
						}
					}
				}
			}

			//std::cout << "HERE IN REMAPPING" << std::endl;

			for (unsigned int i_space=0;i_space<ft->num_present_continuous_spaces;i_space++)
			{
				//std::cout << " IN SPACE " << i_space << std::endl;
				//std::cout << " REQUIRES CONTINUOUS SHAPE FUNCTIONS: " << (required.continuous_spaces[i_space].psi || required.continuous_spaces[i_space].dx_psi || required.continuous_spaces[i_space].dX_psi) << std::endl;
				//std::cout << " NUMBER OF FIELDS: " << ft->continuous_spaces[i_space].numfields << std::endl;
				const JITFuncSpec_Table_FiniteElement_SpaceInfo_t * space_info = ft->present_continuous_spaces[i_space];
				if ((required.continuous_spaces[space_info->space_index].psi || required.continuous_spaces[space_info->space_index].dx_psi || required.continuous_spaces[space_info->space_index].dX_psi) && space_info->numfields>0)
				{
					//std::cout << " IN SPACE " << i_space << " REQUIRES CONTINUOUS SHAPE FUNCTIONS" << std::endl;

					unsigned nnode_space=eleminfo.nnode_of_space[space_info->space_index];
					//std::cout << " IN SPACE " << i_space << " REQUIRES CONTINUOUS SHAPE FUNCTIONS, NODES: " << nnode_space << std::endl;
					for (unsigned int f = 0; f < space_info->numfields_basebulk; f++)
					{
						unsigned foffs=f+ space_info->buffer_offset_basebulk;
						JITHangInfo_t * hangbuffer=shape_info->hanginfo[foffs];
						for (unsigned int l = 0; l < nnode_space; l++)
						{
							if (!hangbuffer[l].nummaster)
							{
								// NON HANGING -> Set the hanging to 1 node, which is just the remapped equation
								hangbuffer[l].nummaster = 1;
								hangbuffer[l].masters[0].weight = 1.0;
								if (eleminfo.nodal_local_eqn[l][foffs] >= 0)
								{
									hangbuffer[l].masters[0].local_eqn = eleminfo.nodal_local_eqn[l][foffs];
								}
								else
								{
									hangbuffer[l].masters[0].local_eqn = -1;
								}
							}

							for (int m = 0; m < hangbuffer[l].nummaster; m++)
							{
								if (hangbuffer[l].masters[m].local_eqn >= 0)
								{
									hangbuffer[l].masters[m].local_eqn = eqn_remap[hangbuffer[l].masters[m].local_eqn];
									if (hangbuffer[l].masters[m].local_eqn == -666)
									{
										// Name the FIELD. Whether this is a real defect or an over-strict check
										// turns entirely on whether the interface residual actually uses this
										// field, and a bare "space C2, index 0" gives the reader no way to tell.
										std::ostringstream oss;
										oss << "field '" << (space_info->fieldnames && space_info->fieldnames[foffs] ? space_info->fieldnames[foffs] : "?")
											<< "' on space '" << space_info->space_name << "'"
											<< " (node " << l << ", master " << m << " of " << hangbuffer[l].nummaster
											<< ", field index " << f << " of " << space_info->numfields_basebulk << ")";
										throw_runtime_error("MISSING EXTERNAL DEPENDENCY: " + oss.str() + ". The opposite/bulk element "
															"holds this dof, but it is not registered as external data of this interface "
															"element, so its equation cannot be remapped. Element " +
															std::string(this->get_code_instance() && this->get_code_instance()->get_code()
																			? this->get_code_instance()->get_code()->get_file_name() : "?"));
									}
								}
							}
						}
					}
					for (unsigned int f = 0; f < space_info->numfields-space_info->numfields_basebulk; f++)
					{
						unsigned foffs=f+ space_info->buffer_offset_interf;
						JITHangInfo_t * hangbuffer=shape_info->hanginfo[foffs];
						for (unsigned int l = 0; l < nnode_space; l++)
						{
							if (!hangbuffer[l].nummaster)
							{
								// NON HANGING -> Set the hanging to 1 node, which is just the remapped equation
								hangbuffer[l].nummaster = 1;
								hangbuffer[l].masters[0].weight = 1.0;
								if (eleminfo.nodal_local_eqn[l][foffs] >= 0)
								{
									hangbuffer[l].masters[0].local_eqn = eleminfo.nodal_local_eqn[l][foffs];
								}
								else
								{
									hangbuffer[l].masters[0].local_eqn = -1;
								}
							}

							for (int m = 0; m < hangbuffer[l].nummaster; m++)
							{
								if (hangbuffer[l].masters[m].local_eqn >= 0)
								{
									hangbuffer[l].masters[m].local_eqn = eqn_remap[hangbuffer[l].masters[m].local_eqn];
									if (hangbuffer[l].masters[m].local_eqn == -666)
									{
										std::ostringstream oss;
										oss << this;
										oss << " node: " << l << ", master " << m << " of " << hangbuffer[l].nummaster  << ", index " << f << ", " << foffs << " of " << space_info->numfields-space_info->numfields_basebulk;
										throw_runtime_error("MISSING EXTERNAL DEPENDENCY ON SPACE '"+std::string(space_info->space_name)+"' ON ELEM PTR: " + oss.str());
									}
								}
							}
						}
					}
				}
			}


			// TODO: DG loop
			for (unsigned int i_space=0;i_space<codeinst->get_func_table()->num_present_dg_spaces;i_space++)
			{
				const JITFuncSpec_Table_FiniteElement_SpaceInfo_t * space_info = ft->present_dg_spaces[i_space];
				if (space_info->numfields && (required.continuous_spaces[space_info->space_index].psi || required.continuous_spaces[space_info->space_index].dx_psi || required.continuous_spaces[space_info->space_index].dX_psi) && space_info->numfields>0)
				{
					unsigned nnode_space=eleminfo.nnode_of_space[space_info->space_index];
					for (unsigned int f = 0; f < space_info->numfields_basebulk; f++)
					{
						unsigned foffs=f + space_info->buffer_offset_basebulk;
						JITHangInfo_t * hangbuffer=shape_info->hanginfo[foffs];
						for (unsigned int l = 0; l < nnode_space; l++)
						{
							if (!hangbuffer[l].nummaster)
							{
								// NON HANGING -> HANGING WITH WEIGHT 1 for external element data
								hangbuffer[l].nummaster = 1;
								hangbuffer[l].masters[0].weight = 1.0;
							}
							int eq=eleminfo.nodal_local_eqn[l][foffs];
							if (eq >= 0)
							{
								eq=eqn_remap[eq];
								if (eq==-666)
								{
										std::ostringstream oss;
										oss << this;
										throw_runtime_error("MISSING EXTERNAL "+std::string(space_info->space_name)+" DEPENDENCY ON ELEM PTR: " + oss.str());
								}
								hangbuffer[l].masters[0].local_eqn = eq ;
							}
							else
							{
								hangbuffer[l].masters[0].local_eqn = -1;
							}
						}
					}
					for (unsigned int f = 0; f < space_info->numfields-space_info->numfields_basebulk; f++)
					{
						unsigned foffs=f + space_info->buffer_offset_interf;
						JITHangInfo_t * hangbuffer=shape_info->hanginfo[foffs];
						for (unsigned int l = 0; l < nnode_space; l++)
						{
							if (!hangbuffer[l].nummaster)
							{
								// NON HANGING -> HANGING WITH WEIGHT 1 for external element data
								hangbuffer[l].nummaster = 1;
								hangbuffer[l].masters[0].weight = 1.0;
							}
							int eq=eleminfo.nodal_local_eqn[l][foffs];
							if (eq >= 0)
							{
								eq=eqn_remap[eq];
								if (eq==-666)
								{
										std::ostringstream oss;
										oss << this;
										throw_runtime_error("MISSING EXTERNAL " + std::string(space_info->space_name) + " DEPENDENCY ON ELEM PTR: " + oss.str());
								}
								hangbuffer[l].masters[0].local_eqn = eq ;
							}
							else
							{
								hangbuffer[l].masters[0].local_eqn = -1;
							}
						}
					}
				}
			}



			if (codeinst->get_func_table()->info_DL.numfields && (required.DL.dx_psi || required.DL.psi || required.DL.dX_psi))
			{
				for (unsigned int f = 0; f < codeinst->get_func_table()->info_DL.numfields; f++)
				{
					unsigned foffs = f + ft->info_DL.buffer_offset_basebulk;
					JITHangInfo_t * hangbuffer=shape_info->hanginfo[foffs];
					for (unsigned int l = 0; l < eleminfo.nnode_DL; l++)
					{
						if (!hangbuffer[l].nummaster)
						{
							// NON HANGING -> HANGING WITH WEIGHT 1 for external element data
							hangbuffer[l].nummaster = 1;
							hangbuffer[l].masters[0].weight = 1.0;
						}
						int eq=eleminfo.nodal_local_eqn[l][foffs];
						if (eq >= 0)
						{
						   eq=eqn_remap[eq];
						   if (eq==-666)
						   {
								std::ostringstream oss;
								oss << this;
								throw_runtime_error("MISSING EXTERNAL DL DEPENDENCY ON ELEM PTR: " + oss.str());
						   }
							hangbuffer[l].masters[0].local_eqn = eq ;
						}
						else
						{
							hangbuffer[l].masters[0].local_eqn = -1;
						}
					}
				}
			}


			if (codeinst->get_func_table()->info_D0.numfields && (required.D0.psi))
			{
				for (unsigned int f = 0; f < codeinst->get_func_table()->info_D0.numfields; f++)
				{
					unsigned foffs = f + ft->info_D0.buffer_offset_basebulk;
					JITHangInfo_t * hangbuffer=shape_info->hanginfo[foffs];
					if (!hangbuffer[0].nummaster)
					{
						// NON HANGING -> HANGING WITH WEIGHT 1 for external element data
						hangbuffer[0].nummaster = 1;
						hangbuffer[0].masters[0].weight = 1.0;
					}
					int eq=eleminfo.nodal_local_eqn[0][foffs];
					if (eq >= 0)
					{
					   eq=eqn_remap[eq];
					   if (eq==-666)
					   {
							std::ostringstream oss;
							oss << this;
							throw_runtime_error("MISSING EXTERNAL D0 DEPENDENCY ON ELEM PTR: " + oss.str());
					   }
						hangbuffer[0].masters[0].local_eqn = eq;
					}
					else
					{
						hangbuffer[0].masters[0].local_eqn = -1;
					}
				}
			}
		}
		return res;
	}

	/*
	void BulkElementBase::assign_all_generic_local_eqn_numbers(const bool &store_local_dof_pt)
	{
	 std::cout << "IN  assign_all_generic_local_eqn_numbers " << std::endl;
	 this->RefineableSolidElement::assign_all_generic_local_eqn_numbers(store_local_dof_pt);
	 std::cout << "DOING SOLID " << std::endl;
	 this->RefineableSolidElement::assign_solid_local_eqn_numbers(store_local_dof_pt);
	}
	*/
	// After oomph-lib has assigned local equation numbers (nodal, internal, external, position),
	// rebuilds the JIT eleminfo buffer (which caches those numbers) and makes sure all internal
	// data shares the nodes' timestepper, since internal data may be created before the element
	// is fully connected to the mesh's timestepping.
	// Collect every node that flatten_hang_for_position() would eventually resolve to as a LEAF, starting
	// from n. Mirrors that function's recursion exactly (C1 position constraint -> stored corners; geometric
	// hang -> masters; otherwise a leaf), but records nodes instead of equation numbers.
	void BulkElementBase::collect_position_leaf_nodes(oomph::Node *n, std::set<oomph::Node *> &out, int depth)
	{
		if (!n || depth > 32) return;
		Node *pn = dynamic_cast<Node *>(n);
		if (pn && !pn->c1_constraint_corners.empty())
		{
			bool constrained = false;
			for (unsigned d = 0; d < this->nodal_dimension(); d++)
				if (this->node_is_c1_constrained_for_position(pn, d)) { constrained = true; break; }
			if (constrained)
			{
				for (const auto &cw : pn->c1_constraint_corners)
					collect_position_leaf_nodes(cw.first, out, depth + 1);
				return;
			}
		}
		if (n->is_hanging())
		{
			oomph::HangInfo *h = n->hanging_pt();
			for (unsigned m = 0; m < h->nmaster(); m++)
				collect_position_leaf_nodes(h->master_node_pt(m), out, depth + 1);
			return;
		}
		out.insert(n);
	}

	// Register, as position-hang masters of THIS element, the leaf nodes that the C1 position constraint
	// redistributes onto but oomph-lib does not know about.
	//
	// The comment that used to sit in assign_additional_local_eqn_numbers claimed "no separate equation
	// numbers are required, because the flattened leaf dofs are exactly the coarse vertices that oomph-lib
	// already registered as hang masters". That is FALSE for positions. oomph registers position-hang masters
	// by walking the GEOMETRIC hang of this element's own nodes only, whereas a constrained node's
	// c1_constraint_corners are written by whichever element sees that node as a NON-vertex -- which may be a
	// neighbour. The corners can therefore be vertices of a neighbouring element that this element never
	// registers, and reading their local equation number then returns garbage (the section 4.14 abort; since
	// the previous commit, a named diagnostic instead).
	//
	// Runs from assign_additional_local_eqn_numbers(), which oomph documents as being called after ALL other
	// local equation numbering, so Local_position_hang_eqn is already populated and ndof() is final.
	void BulkElementBase::register_c1_constraint_position_masters()
	{
		if (!has_additional_dof_constraints) return;
		const unsigned nd = this->nodal_dimension();
		const unsigned nn = this->nnode();

		std::set<oomph::Node *> leaves;
		for (unsigned l = 0; l < nn; l++)
		{
			Node *n = dynamic_cast<Node *>(node_pt(l));
			if (!n) continue;
			bool constrained = false;
			for (unsigned d = 0; d < nd; d++)
				if (this->node_is_c1_constrained_for_position(n, d)) { constrained = true; break; }
			if (constrained) collect_position_leaf_nodes(n, leaves, 0);
		}
		if (leaves.empty()) return;

		std::deque<unsigned long> new_eqs;
		unsigned local_eqn_number = this->ndof();
		for (std::set<oomph::Node *>::iterator it = leaves.begin(); it != leaves.end(); ++it)
		{
			oomph::Node *n = *it;
			bool is_local = false;
			for (unsigned l = 0; l < nn; l++)
				if (node_pt(l) == n) { is_local = true; break; }
			if (is_local) continue; // resolved through eleminfo.pos_local_eqn instead
			oomph::DenseMatrix<int> &pe = this->local_position_hang_eqn(n);
			if (pe.nrow() > 0) continue; // oomph already registered it
			oomph::SolidNode *sn = dynamic_cast<oomph::SolidNode *>(n);
			if (!sn) continue;
			pe.resize(1, nd);
			for (unsigned k = 0; k < nd; k++)
			{
				long e = sn->position_eqn_number(0, k);
				if (e >= 0) { new_eqs.push_back((unsigned long)e); pe(0, k) = (int)local_eqn_number++; }
				else pe(0, k) = -1; // pinned
			}
		}
		// Empty dof-pointer deque: add_global_eqn_numbers() skips that block entirely, which matches the
		// store_local_dof_pt==false case. These extra masters therefore do not appear in Dof_pt, which is
		// only used for optional local-dof caching, not for assembly.
		if (!new_eqs.empty()) this->add_global_eqn_numbers(new_eqs, std::deque<double *>());
	}

	// Hook for subclasses to add element-type-specific hanging-node setup after the generic
	// oomph-lib hanging node machinery has run; no-op by default.
	void BulkElementBase::further_setup_hanging_nodes()
	{
		// std::cout << "FURTHER SETUP HANG" << std::endl;
	}

	// --- Shared tesselated-numpy hanging-node helpers (used by both the quad and tri paths) ---

	// Record a finer neighbour's hanging node n at its (topologically computed) LOCAL coordinate s_coarse in
	// THIS element. The receiving edge is found in REFERENCE space: which corner-pair's reference segment
	// s_coarse lies on. This is exact and curvature-independent (no physical positions, no locate_zeta) -- the
	// straight/bowed shape of the physical edge is irrelevant because reference edges are always straight.
	void BulkElementBase::tess_register_hanging_node(const oomph::Vector<double> &s_coarse, oomph::Node *n, const std::vector<std::pair<unsigned, unsigned>> &edge_corner_pairs, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes)
	{
		// Same unsupported case as inform_coarser_neighbors_for_tesselated_numpy()/tess_inform_coarser_tri()
		// (see their comments): an InterfaceElementBase is never registered for tesselated-numpy export, so
		// this->_numpy_index is not a valid index into add_nodes here. Reaching this point for an interface
		// element (e.g. a corner element sitting on an interface-of-an-interface, such as a free surface's
		// intersection with an axis of symmetry) indexed add_nodes out of bounds instead of raising - silent
		// heap corruption that surfaced much later as an unrelated-looking std::bad_alloc whose own exception
		// unwinding then segfaulted. Raise here, consistently with the sibling guards, until tesselating
		// interface meshes is actually supported.
		if (dynamic_cast<InterfaceElementBase *>(this))
			throw_runtime_error("Cannot yet tesselate interface meshes [will fail in connecting hanging nodes and have to go via the parent mesh]");
		for (unsigned ni = 0; ni < this->nnode(); ni++)
		{
			if (this->node_pt(ni) == n)
				return; // already my node
		}
		for (unsigned e = 0; e < edge_corner_pairs.size(); e++)
		{
			oomph::Vector<double> sa, sb;
			this->local_coordinate_of_node(edge_corner_pairs[e].first, sa);
			this->local_coordinate_of_node(edge_corner_pairs[e].second, sb);
			double AB2 = 0.0, An = 0.0;
			for (unsigned d = 0; d < 2; d++)
			{
				double ab = sb[d] - sa[d];
				AB2 += ab * ab;
				An += (s_coarse[d] - sa[d]) * ab;
			}
			if (AB2 < 1e-30)
				continue;
			double t = An / AB2;
			if (t < 1e-9 || t > 1.0 - 1e-9)
				continue; // beyond/at the reference endpoints
			double perp2 = 0.0;
			for (unsigned d = 0; d < 2; d++)
			{
				double p = (s_coarse[d] - sa[d]) - t * (sb[d] - sa[d]);
				perp2 += p * p;
			}
			if (perp2 > 1e-12 * AB2)
				continue; // not on this reference edge
			unsigned myindex = this->_numpy_index;
			if (add_nodes[myindex].empty())
				add_nodes[myindex].resize(this->nedges());
			add_nodes[myindex][e].insert(n);
			this->_tess_hang_scoord[n] = s_coarse;
			return;
		}
	}

	// Triangulate this element for the tesselated-numpy export, including the finer-neighbour hanging nodes
	// registered on its edges. The element's boundary is the convex polygon of its corner nodes plus, along
	// each edge, that edge's own mid node(s) and any registered hanging nodes (each placed by a linear blend
	// along the edge). We build that ordered boundary loop and fan-triangulate it - NOT via Delaunay, because
	// with pure-tri multi-level hanging an edge can carry >=3 interior points that are exactly collinear, a
	// degeneracy delaunator mis-handles (it may chain a long edge past a collinear midpoint => T-junction).
	// A tri has at most one interior node (the C1TB/C2TB centroid); with it we fan from the centroid to every
	// boundary segment, without it we fan from boundary vertex 0. `triangles` is filled with flat CCW index
	// triples where each index is an own-node local index (< nnode()) or nnode()+running in add_nodes edge
	// order - exactly the indexing Mesh::to_numpy expects.
	void BulkElementBase::tess_hanging_delaunay(const std::vector<std::pair<unsigned, unsigned>> &edge_corner_pairs, const std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes, std::vector<unsigned> &triangles) const
	{
		triangles.clear();
		const unsigned nn = this->nnode();
		const unsigned dim = this->nodal_dimension();
		// All triangulation topology is done in LOCAL (reference-element) coordinates, where every edge is a
		// straight segment and the edge-mid / centroid nodes sit at their exact reference positions. This is
		// essential for CURVED (C2/higher-geometry) elements, whose nodes are not collinear/planar in physical
		// space; using physical positions there would misclassify edge nodes and produce an invalid mesh.
		std::vector<oomph::Vector<double>> sloc(nn);
		for (unsigned i = 0; i < nn; i++)
			this->local_coordinate_of_node(i, sloc[i]);
		// Corner local indices (edge endpoints).
		std::set<unsigned> corners;
		for (auto &ec : edge_corner_pairs)
		{
			corners.insert(ec.first);
			corners.insert(ec.second);
		}
		// A boundary point carries its OUTPUT index, its parameter along the edge it lies on, and its LOCAL position.
		struct BPt
		{
			unsigned outidx;
			double param;
			double x, y;
		};
		std::vector<std::vector<BPt>> edge_interior(edge_corner_pairs.size()); // non-corner points per edge, unsorted
		std::vector<unsigned> interior_pts;                                     // own nodes not on any edge (centroid)
		// Classify this element's own non-corner nodes onto an edge (by collinearity in LOCAL space) or as interior.
		for (unsigned i = 0; i < nn; i++)
		{
			if (corners.count(i))
				continue;
			bool placed = false;
			for (unsigned e = 0; e < edge_corner_pairs.size(); e++)
			{
				const oomph::Vector<double> &sA = sloc[edge_corner_pairs[e].first];
				const oomph::Vector<double> &sB = sloc[edge_corner_pairs[e].second];
				double AB2 = 0.0, An = 0.0;
				for (unsigned d = 0; d < 2; d++)
				{
					double ab = sB[d] - sA[d];
					AB2 += ab * ab;
					An += (sloc[i][d] - sA[d]) * ab;
				}
				if (AB2 < 1e-30)
					continue;
				double lam = An / AB2;
				if (lam < 1e-9 || lam > 1.0 - 1e-9)
					continue;
				double perp2 = 0.0;
				for (unsigned d = 0; d < 2; d++)
				{
					double p = (sloc[i][d] - sA[d]) - lam * (sB[d] - sA[d]);
					perp2 += p * p;
				}
				if (perp2 > 1e-10 * AB2)
					continue;
				edge_interior[e].push_back({i, lam, sloc[i][0], sloc[i][1]});
				placed = true;
				break;
			}
			if (!placed)
				interior_pts.push_back(i);
		}
		// Registered finer-neighbour hanging nodes: output index nn+running in add_nodes edge order. Each node's
		// LOCAL position in this element was computed TOPOLOGICALLY by the finer neighbour and stored in
		// _tess_hang_scoord; the ordering parameter is its projection onto the reference edge.
		const std::vector<std::set<oomph::Node *>> &mine = add_nodes[this->_numpy_index];
		unsigned running = 0;
		for (unsigned e = 0; e < edge_corner_pairs.size(); e++)
		{
			if (e >= mine.size())
				break;
			const oomph::Vector<double> &sA = sloc[edge_corner_pairs[e].first];
			const oomph::Vector<double> &sB = sloc[edge_corner_pairs[e].second];
			double AB2 = 0.0;
			for (unsigned d = 0; d < 2; d++)
				AB2 += (sB[d] - sA[d]) * (sB[d] - sA[d]);
			for (oomph::Node *n : mine[e])
			{
				auto it = this->_tess_hang_scoord.find(n);
				oomph::Vector<double> sc = (it != this->_tess_hang_scoord.end()) ? it->second : oomph::Vector<double>(2, 0.0);
				double lam = 0.5;
				if (AB2 > 1e-30)
				{
					double An = 0.0;
					for (unsigned d = 0; d < 2; d++)
						An += (sc[d] - sA[d]) * (sB[d] - sA[d]);
					lam = An / AB2;
				}
				edge_interior[e].push_back({nn + running, lam, sc[0], sc[1]});
				running++;
			}
		}
		// Build the ordered boundary loop: for each edge, its start corner then its interior points sorted by
		// parameter. Edges are taken in edge_corner_pairs order, which for the tri corner layout is CCW.
		std::vector<BPt> loop;
		for (unsigned e = 0; e < edge_corner_pairs.size(); e++)
		{
			unsigned cidx = edge_corner_pairs[e].first;
			loop.push_back({cidx, 0.0, sloc[cidx][0], sloc[cidx][1]});
			std::sort(edge_interior[e].begin(), edge_interior[e].end(), [](const BPt &a, const BPt &b)
					  { return a.param < b.param; });
			for (auto &bp : edge_interior[e])
				loop.push_back(bp);
		}
		if (loop.size() < 3)
			return;
		if (!interior_pts.empty())
		{
			// A genuine interior node (the C1TB/C2TB centroid): fan from it around the whole boundary loop.
			unsigned c = interior_pts[0];
			for (unsigned i = 0; i < loop.size(); i++)
			{
				triangles.push_back(c);
				triangles.push_back(loop[i].outidx);
				triangles.push_back(loop[(i + 1) % loop.size()].outidx);
			}
			return;
		}
		// No interior node: triangulate the convex boundary polygon by ear clipping. A plain fan (or clipping
		// the first non-degenerate corner) fails here because an edge can carry several EXACTLY collinear
		// interior points (own C2 mid + several finer-neighbour hanging nodes under multi-level tri hanging):
		// such a chord would skip the points between its endpoints, re-introducing a T-junction. So a triple
		// (a,b,c) is a valid ear only if it is non-degenerate AND no other loop vertex lies inside it or on
		// its base a-c. That keeps every boundary point as a triangle vertex with no spanning chords.
		std::vector<BPt> poly = loop;
		double area2max = 0.0;
		for (unsigned i = 1; i + 1 < poly.size(); i++)
		{
			double a2 = std::abs((poly[i].x - poly[0].x) * (poly[i + 1].y - poly[0].y) - (poly[i + 1].x - poly[0].x) * (poly[i].y - poly[0].y));
			if (a2 > area2max)
				area2max = a2;
		}
		double area_eps = 1e-9 * area2max;
		while (poly.size() > 3)
		{
			unsigned n2 = poly.size();
			bool clipped = false;
			for (unsigned i = 0; i < n2; i++)
			{
				const BPt &a = poly[(i + n2 - 1) % n2];
				const BPt &b = poly[i];
				const BPt &c = poly[(i + 1) % n2];
				double area2 = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
				if (area2 <= area_eps)
					continue; // degenerate/reflex (loop is CCW)
				// Reject if any other loop vertex lies inside triangle (a,b,c) or on its base a-c: for a CCW
				// triangle a point is inside-or-on when all three edge cross products are >= -eps.
				bool empty = true;
				for (unsigned j = 0; j < n2 && empty; j++)
				{
					const BPt &p = poly[j];
					if (&p == &a || &p == &b || &p == &c)
						continue;
					double e1 = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
					double e2 = (c.x - b.x) * (p.y - b.y) - (c.y - b.y) * (p.x - b.x);
					double e3 = (a.x - c.x) * (p.y - c.y) - (a.y - c.y) * (p.x - c.x);
					if (e1 >= -area_eps && e2 >= -area_eps && e3 >= -area_eps)
						empty = false;
				}
				if (!empty)
					continue;
				triangles.push_back(a.outidx);
				triangles.push_back(b.outidx);
				triangles.push_back(c.outidx);
				poly.erase(poly.begin() + i);
				clipped = true;
				break;
			}
			if (!clipped)
				break; // safety: no valid ear found (should not happen for a real convex element)
		}
		if (poly.size() == 3)
		{
			triangles.push_back(poly[0].outidx);
			triangles.push_back(poly[1].outidx);
			triangles.push_back(poly[2].outidx);
		}
	}

	// Tri-native counterpart of BulkElementQuad2dC1::inform_coarser_neighbors_for_tesselated_numpy(): for
	// each of this triangle's 3 edges, use the topological tri_edge_neighbour finder to detect a strictly
	// coarser neighbour (a coarser tri, or - across a mixed quad/tri interface - a coarser quad) and
	// register this triangle's edge nodes with that coarser neighbour.
	void BulkElementBase::tess_inform_coarser_tri(const std::vector<std::pair<unsigned, unsigned>> &edge_corner_pairs, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes)
	{
		(void)edge_corner_pairs;
		if (dynamic_cast<InterfaceElementBase *>(this))
			throw_runtime_error("Cannot yet tesselate interface meshes [will fail in connecting hanging nodes and have to go via the parent mesh]");
		oomph::RefineableTElement<2> *re = dynamic_cast<oomph::RefineableTElement<2> *>(this);
		if (!re)
			return;
		// Fully topological: RefineableTElement<2> finds the coarser edge neighbours via tri_edge_neighbour and
		// computes each edge node's coordinate in the coarse neighbour with the tree affine/cross-shape maps.
		re->tess_register_on_coarser_for_numpy(add_nodes);
	}
}
