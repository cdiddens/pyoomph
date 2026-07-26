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



#include "mesh.hpp"
#include "nodes.hpp"
#include "meshtemplate.hpp"
#include "problem.hpp"
#include "elements.hpp"
#include "mesh3d.hpp"
#include <array>

#include "Telements.h"
// #include "unstructured_two_d_mesh_geometry_base.h"

#include "exception.hpp"

namespace pyoomph
{


	// True only if every element in the mesh is a brick (h-refinement via an OcTreeForest is only
	// implemented for pure-brick meshes). Also issues a one-off warning (per mesh) if adaptive
	// refinement was requested but the mesh contains non-brick (e.g. tetrahedral) elements.
	bool TemplatedMeshBase3d::refinement_possible()
	{
		// h-refinement via the OcTree hierarchy is implemented for pure-brick meshes and (branch
		// mixed_adapt) pure-tetrahedron meshes (a tet refines 1->8, reusing the OcTree's 8-son
		// bookkeeping, with geometric node-sharing/hanging). Mixed brick+tet is not yet supported.
		bool allQ = true, allT = true;
		for (unsigned int i = 0; i < this->nelement(); i++)
		{
			bool is_brick = (dynamic_cast<oomph::BrickElementBase *>(this->element_pt(i)) != NULL);
			bool is_tet = (dynamic_cast<oomph::TElementBase *>(this->element_pt(i)) != NULL);
			allQ = allQ && is_brick;
			allT = allT && is_tet;
		}
		if (allQ || allT)
		{
			return true;
		}
		else
		{
			if (this->max_refinement_level() && !issued_tri_refinement_warning && !this->problem->is_quiet())
			{
				std::cerr << "WARNING: Mixed-element 3d mesh -> cannot be adaptive yet. Requires cross-shape facet neighbouring for mixed meshes" << std::endl;
				issued_tri_refinement_warning = true;
			}
			return false;
		}
	}

	namespace
	{
		// Resolve a (possibly hanging in `slot`) node into a weighted sum over REAL (non-hanging in that
		// slot) leaf nodes, accumulated into `out`. Recurses through hanging masters, so a hang whose
		// masters are themselves hanging flattens to real leaves; the master chain is acyclic (masters are
		// always coarser) so this terminates. Used by the tree-route flatten, per slot (-1, C2, C1).
		void resolve_hang_to_real_slot(oomph::Node *X, double w, std::map<oomph::Node *, double> &out, int slot)
		{
			if (!X->is_hanging(slot)) { out[X] += w; return; }
			oomph::HangInfo *h = X->hanging_pt(slot);
			for (unsigned m = 0; m < h->nmaster(); m++)
				resolve_hang_to_real_slot(h->master_node_pt(m), w * h->master_weight(m), out, slot);
		}
	}

	// Install hanging nodes for tetrahedral meshes after (non-uniform) refinement, in four passes:
	//   1. FACE-interior nodes (a >1-level jump puts nodes strictly inside a coarse tet face) hang
	//      barycentrically on that C1 face's corners -- done first so they bind to real corners.
	//   2. EDGE-interior nodes hang on the coarse edge {P,Q} (linear C1 / quadratic C2), coarsest
	//      edge first and skipping edges with a hanging endpoint.
	//   3. FLATTEN: any hanging node whose master is itself hanging (a chain, arising at >1-level
	//      jumps) is re-expressed directly over real (non-hanging) leaf nodes, so the assembled
	//      Jacobian is exact without relying on assembly-time flattening.
	// Combined with the 2:1 balancing pass (enforce_refinement_balance, run just before this), the
	// linear residual reaches machine zero for BOTH C1 and C2 tets under arbitrary refinement --
	// uniform, single-level, error-driven, and abrupt >1-level RefineToLevel jumps. (Balancing keeps
	// >1-level jumps rare so C2 face-interior nodes -- not handled here -- essentially do not occur;
	// the flatten pass makes whatever hanging remains exact.)
	void TemplatedMeshBase3d::post_adapt_setup_hanging_nodes()
	{
		bool has_tet = false;
		for (unsigned int ie = 0; ie < this->nelement() && !has_tet; ie++)
			if (dynamic_cast<oomph::TElementBase *>(this->element_pt(ie))) has_tet = true;
		if (!has_tet) return; // brick meshes: oomph-lib already handled hanging

		// Validate the tet-native FACE-neighbour finder against geometry (env PYOOMPH_TET_FACENB_CHECK).
		// For every leaf tet face: the tree-found >=-sized neighbour must contain all 3 face vertex nodes
		// (equal neighbour shares the exact face; a coarser one has them as its own vertices/edge-mids).
		// Also count faces the tree leaves as null vs the geometric interior faces (facet adjacency).
		if (getenv("PYOOMPH_TET_FACENB_CHECK"))
		{
			FacetAdjacencyMap fadj = this->build_facet_adjacency();
			int nface = 0, nfound = 0, nnull = 0, nbadnode = 0, ncross = 0, ninterior_missed = 0;
			int ncoarser = 0, ncoarser_bad = 0;
			for (unsigned int ie = 0; ie < this->nelement(); ie++)
			{
				oomph::TElementBase *te = dynamic_cast<oomph::TElementBase *>(this->element_pt(ie));
				oomph::RefineableTElement<3> *re = dynamic_cast<oomph::RefineableTElement<3> *>(this->element_pt(ie));
				if (!te || !re || !re->tree_pt() || !re->tree_pt()->is_leaf()) continue;
				for (int f = 0; f < 4; f++)
				{
					oomph::Node *fn[3]; { int k = 0; for (int v = 0; v < 4; v++) if (v != f) fn[k++] = te->vertex_node_pt(v); }
					std::set<pyoomph::Node *> fkey; for (int i = 0; i < 3; i++) fkey.insert(dynamic_cast<pyoomph::Node *>(fn[i]));
					auto ait = fadj.find(fkey);
					const bool geo_interior = (ait != fadj.end() && ait->second.size() > 1);
					nface++;
					oomph::RefineableTElement<3>::TetFaceNeighbour nb = re->tet_face_neighbour(f);
					if (nb.el)
					{
						nfound++;
						oomph::FiniteElement *nfe = dynamic_cast<oomph::FiniteElement *>(nb.el);
						// Geometric containment: each of the leaf face's 3 vertex POSITIONS must lie inside the
						// neighbour (equal -> on the shared face; coarser -> on the coarse face, where the fine
						// node is NOT a node of the coarse element -- that is what hanging IS). locate_zeta is
						// the right check; node-membership would falsely flag every correct coarse neighbour.
						bool bad = false;
						for (int i = 0; i < 3; i++)
						{
							oomph::Vector<double> xp(3), sloc(3);
							for (int d = 0; d < 3; d++) xp[d] = fn[i]->x(d);
							oomph::GeomObject *go = 0;
							nfe->locate_zeta(xp, go, sloc);
							if (go == 0) { bad = true; break; }
						}
						if (bad) nbadnode++;
						if (nb.diff_level < 0) { ncoarser++; if (bad) ncoarser_bad++; } // the hanging-relevant case
					}
					else
					{
						nnull++;
						if (nb.cross_root) ncross++;
						else if (geo_interior) ninterior_missed++; // same-root interior face the tree failed to resolve
					}
				}
			}
			std::fprintf(stderr, "[tet-facenb] leaf-faces=%d found=%d(bad-node=%d) null=%d(cross-root=%d, same-root-null=%d) | COARSER(hang-relevant)=%d bad=%d\n",
				nface, nfound, nbadnode, nnull, ncross, ninterior_missed, ncoarser, ncoarser_bad);
		}

		for (unsigned int in = 0; in < this->nnode(); in++) this->node_pt(in)->set_nonhanging();

		// A C1(TB) field on a C2-coordinate tet mesh (e.g. 3D Taylor-Hood: C2 velocity + C1 pressure)
		// owns a SEPARATE value-hang slot (continuous_spaces[...].hangindex >= 0), distinct from the
		// geometric slot -1 the C2 fields hang on. Such a field hangs LINEARLY on the coarse edge
		// corners {P,Q} -- even at the coarse edge mid-node M, whose velocity is a real dof but whose
		// pressure the coarse element does not carry. Collect those separate slots (empty for pure-C1
		// meshes, where hangindex == -1 and the geometric slot already covers the field).
		std::vector<int> c1_hang_slots;
		// When the dominant (position/-1) space is C2TB, the -1 hang carries the BUBBLE-ENRICHED trace,
		// which on a tet face differs from the plain C2 trace. A plain C2 field therefore owns its own
		// hang slot (continuous_spaces[C2].hangindex >= 0) and must NOT read -1. C2 only ever hangs on
		// EDGES (it has no face-interior nodes -- those are C2TB bubbles), where the bubble vanishes so
		// the hang equals -1's; we mirror the -1 EDGE hang onto this slot below. -1 if C2 shares the slot.
		int c2_hang_slot = -1;
		if (this->codeinst)
		{
			auto *ft = this->codeinst->get_func_table();
			for (int sp : {SPACE_INDEX_C1TB, SPACE_INDEX_C1})
			{
				int h = ft->continuous_spaces[sp].hangindex;
				if (h >= 0) c1_hang_slots.push_back(h);
			}
			c2_hang_slot = ft->continuous_spaces[SPACE_INDEX_C2].hangindex;
		}

		// -------- PER-ELEMENT TOPOLOGICAL HANGING (the tet tree route) --------
		// The 3d analogue of the triangle tree route: install hanging per element via tet_hang_face /
		// tet_hang_edge (OcTree neighbour finders + the exact affine map + interpolating_basis). This
		// replaced the geometric position/facet-adjacency passes (deleted; validated machine-zero equal
		// against them across C1/C2/TH/CR, uniform/2:1/multi-level/error-based/unrefinement).
		{
			// Valid hang slots only: value_id -1 (geometric) plus any separate C1/C2 slot that is a real
			// value index (< ncont). continuous_spaces[...].hangindex can be a placeholder >= ncont on a mesh
			// with no such field (e.g. C1 hangindex on a pure-C2 mesh); the geometric pass tolerates that via
			// its per-node nvalue guards, but the per-element helpers must not iterate an out-of-range slot.
			const int ncont = this->nelement() ? (int)dynamic_cast<oomph::RefineableElement *>(this->element_pt(0))->ncont_interpolated_values() : 0;
			std::vector<int> slots, valid_c1_slots;
			slots.push_back(-1);
			if (c2_hang_slot >= 0 && c2_hang_slot < ncont) slots.push_back(c2_hang_slot);
			for (int s : c1_hang_slots)
				if (s >= 0 && s < ncont && std::find(slots.begin(), slots.end(), s) == slots.end())
				{ slots.push_back(s); valid_c1_slots.push_back(s); }
			for (unsigned int ie = 0; ie < this->nelement(); ie++)
			{
				oomph::RefineableTElement<3> *re = dynamic_cast<oomph::RefineableTElement<3> *>(this->element_pt(ie));
				if (!re || !re->tree_pt() || !re->tree_pt()->is_leaf()) continue;
				for (int slot : slots)
				{
					for (int f = 0; f < 4; f++) re->tet_hang_face(slot, f);
					for (int e = 0; e < 6; e++) re->tet_hang_edge(slot, e);
				}
			}
			// C1(TB) mid-node rule (mirror BulkElementTri2dC2::further_setup_hanging_nodes): every C2 edge-mid
			// node carrying a separate C1 slot hangs 0.5/0.5 on its two edge-corner nodes -- the C1-on-C2 rule.
			// This also constrains the STALE pressure slot left on a former fine corner that becomes a plain
			// conforming C2 mid-edge after UNREFINEMENT (the edge pass only hangs across a coarser neighbour,
			// so it misses this conforming case). Nodes hung above (2:1) are skipped via is_hanging.
			for (int slot : valid_c1_slots)
			{
				for (unsigned int ie = 0; ie < this->nelement(); ie++)
				{
					oomph::FiniteElement *fe = dynamic_cast<oomph::FiniteElement *>(this->element_pt(ie));
					oomph::RefineableElement *re = dynamic_cast<oomph::RefineableElement *>(this->element_pt(ie));
					if (!fe || !re || !re->tree_pt() || !re->tree_pt()->is_leaf() || fe->nnode() < 10) continue;
					oomph::Vector<double> sm(3);
					static const double VC[4][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}};
					for (unsigned em = 4; em < 10; em++) // the 6 edge-mid nodes of a (C2) tet
					{
						oomph::Node *M = fe->node_pt(em);
						if ((int)M->nvalue() <= slot || M->is_hanging(slot)) continue; // no stale slot / already hung (2:1)
						fe->local_coordinate_of_node(em, sm);
						int ca = -1, cb = -1; // the two corner vertices this edge-mid bisects
						for (int v = 0; v < 4; v++)
							for (int w = v + 1; w < 4; w++)
							{
								bool match = true;
								for (int d = 0; d < 3; d++)
									if (std::abs(0.5 * (VC[v][d] + VC[w][d]) - sm[d]) > 1e-9) { match = false; break; }
								if (match) { ca = v; cb = w; }
							}
						if (ca < 0) continue;
						oomph::Node *A = fe->node_pt(ca), *B = fe->node_pt(cb);
						if ((int)A->nvalue() <= slot || (int)B->nvalue() <= slot) continue;
						oomph::HangInfo *hang = new oomph::HangInfo(2);
						hang->set_master_node_pt(0, A, 0.5);
						hang->set_master_node_pt(1, B, 0.5);
						M->set_hanging_pt(hang, slot);
					}
				}
			}
			// Flatten hanging chains on every slot (a master that is itself hanging -> resolve to real
			// leaves), so the assembled Jacobian is exact. Two-phase (read all, then reinstall) per slot.
			for (int slot : slots)
			{
				std::map<oomph::Node *, std::map<oomph::Node *, double>> resolved;
				for (unsigned int in = 0; in < this->nnode(); in++)
				{
					oomph::Node *X = this->node_pt(in);
					if (!X->is_hanging(slot)) continue;
					oomph::HangInfo *h = X->hanging_pt(slot);
					bool chained = false;
					for (unsigned m = 0; m < h->nmaster(); m++)
						if (h->master_node_pt(m)->is_hanging(slot)) { chained = true; break; }
					if (!chained) continue;
					std::map<oomph::Node *, double> flat;
					for (unsigned m = 0; m < h->nmaster(); m++)
						resolve_hang_to_real_slot(h->master_node_pt(m), h->master_weight(m), flat, slot);
					resolved[X] = flat;
				}
				for (const auto &kv : resolved)
				{
					oomph::HangInfo *nh = new oomph::HangInfo(kv.second.size());
					unsigned i = 0;
					for (const auto &mw : kv.second) { nh->set_master_node_pt(i, mw.first, mw.second); i++; }
					kv.first->set_hanging_pt(nh, slot);
				}
			}
			return;
		}
	}

	// Enforce 2:1 refinement balancing for tetrahedral meshes (see header). Iteratively refine any
	// leaf tet that has a node at the quarter point (t=0.25 or 0.75) of one of its edges -- which
	// means a neighbour is >=2 refinement levels finer than it -- until no such tet remains. Each
	// refinement round uses oomph-lib's refine_selected_elements (full, correct rebuild).
	void TemplatedMeshBase3d::enforce_refinement_balance()
	{
		bool has_tet = false;
		for (unsigned int ie = 0; ie < this->nelement() && !has_tet; ie++)
			if (dynamic_cast<oomph::TElementBase *>(this->element_pt(ie))) has_tet = true;
		if (!has_tet) return; // brick/hex meshes: oomph-lib's tree bounds the level difference itself

		const double scale = 1e8;
		const int max_rounds = 40; // safety bound; convergence is bounded by max_refinement_level()
		for (int round = 0; round < max_rounds; round++)
		{
			// Quantized set of all node positions, for O(1) "does a node exist here?" lookups.
			std::set<std::array<long long, 3>> positions;
			for (unsigned int in = 0; in < this->nnode(); in++)
			{
				oomph::Node *n = this->node_pt(in);
				positions.insert({(long long)std::llround(n->x(0) * scale), (long long)std::llround(n->x(1) * scale), (long long)std::llround(n->x(2) * scale)});
			}

			oomph::Vector<unsigned> to_refine;
			for (unsigned int ie = 0; ie < this->nelement(); ie++)
			{
				oomph::TElementBase *tet = dynamic_cast<oomph::TElementBase *>(this->element_pt(ie));
				if (!tet) continue;
				oomph::RefineableElement *re = dynamic_cast<oomph::RefineableElement *>(this->element_pt(ie));
				if (re && re->refinement_level() >= this->max_refinement_level()) continue; // cannot refine further
				oomph::Node *v[4];
				for (unsigned k = 0; k < 4; k++) v[k] = tet->vertex_node_pt(k);
				// The edge-fraction whose presence signals a neighbour >=2 levels finer. A 1-level
				// neighbour subdivides the edge at its midpoint (t=1/2) and, for order p>=2 (e.g. C2),
				// additionally places the sub-edges' own mid-nodes at t=1/4, 3/4 -- so the finest node a
				// 2:1-allowed neighbour produces is at t=1/2^nnode_1d. A node at that fraction therefore
				// means the neighbour is >=2 levels finer. C1 (nnode_1d=2): 1/4; C2 (nnode_1d=3): 1/8.
				const double frac = 1.0 / (double)(1u << tet->nnode_1d());
				const double fracs[2] = {frac, 1.0 - frac};
				bool imbalanced = false;
				for (unsigned a = 0; a < 4 && !imbalanced; a++)
					for (unsigned b = a + 1; b < 4 && !imbalanced; b++)
					{
						for (int fi = 0; fi < 2; fi++)
						{
							std::array<long long, 3> q;
							for (int d = 0; d < 3; d++)
								q[d] = (long long)std::llround((v[a]->x(d) + fracs[fi] * (v[b]->x(d) - v[a]->x(d))) * scale);
							if (positions.count(q)) { imbalanced = true; break; }
						}
					}
				if (imbalanced) to_refine.push_back(ie);
			}
			if (to_refine.size() == 0) break;
			this->refine_selected_elements(to_refine); // flags + full adapt_mesh rebuild
		}
	}

	// Overload used by the generic (oomph-lib) interface: discards the documentation output.
	void TemplatedMeshBase3d::setup_boundary_element_info()
	{
		std::ostringstream oss;
		setup_boundary_element_info(oss);
	}

	// Build Boundary_element_pt / Face_index_at_boundary for a (possibly mixed brick/tet) 3d mesh.
	// For adaptive, pure-brick meshes, delegates to the generic facet-based TemplatedMeshBase
	// implementation (identication_of_boundary_elements_by_facets), which is robust under refinement.
	// Otherwise (non-adaptive or mixed meshes), uses the brick- and tet-specific helpers below.
	void TemplatedMeshBase3d::setup_boundary_element_info(std::ostream &outfile)
	{

		unsigned nbound = nboundary();

		Boundary_element_pt.clear();
		Face_index_at_boundary.clear();
		Boundary_element_pt.resize(nbound);
		Face_index_at_boundary.resize(nbound);

		if (identication_of_boundary_elements_by_facets)
        {
         if (is_adaptation_enabled() &&refinement_possible() )
         {
          identication_of_boundary_elements_by_facets=false; // For adaptive meshes, we find the facets conventionally, but for non-adaptive meshes we can use the facet information from the mesh template which is always accurate, even at mixed corners
         }
        }
        if (identication_of_boundary_elements_by_facets)
        {
         TemplatedMeshBase::setup_boundary_element_info(outfile);
		 Lookup_for_elements_next_boundary_is_setup = true;
         return;
        }

		setup_boundary_element_info_bricks(outfile);
		setup_boundary_element_info_tris(outfile);
		Lookup_for_elements_next_boundary_is_setup = true;
	}

	// Legacy (non-facet-based) boundary-element detection for the brick elements of the mesh: for every
	// brick element, tabulate which of its nodes lie on which boundaries, then use that per-node
	// boundary-membership pattern (via boundary_identifier, counting how many of a face's 4 corner
	// nodes agree on a given +/-x/y/z face indicator) to work out which local face(s) of the element
	// coincide with a boundary. Skips non-brick elements (handled separately by the tet counterpart).
	void TemplatedMeshBase3d::setup_boundary_element_info_bricks(std::ostream &outfile)
	{
		bool doc = false;
		if (outfile)
			doc = true;
		unsigned nbound = nboundary();
		if (doc)
		{
			outfile << "The number of boundaries is " << nbound << "\n";
		}
		Boundary_element_pt.clear();
		Face_index_at_boundary.clear();
		Boundary_element_pt.resize(nbound);
		Face_index_at_boundary.resize(nbound);
		oomph::Vector<oomph::Vector<oomph::FiniteElement *>> vector_of_boundary_element_pt;
		vector_of_boundary_element_pt.resize(nbound);
		// For each (boundary, element) pair, collects one signed direction indicator
		// per corner node of that element which lies on the boundary; a face is only
		// confirmed once all 4 corners belonging to it contribute the same indicator.
		oomph::MapMatrixMixed<unsigned, oomph::FiniteElement *, oomph::Vector<int> *> boundary_identifier;
		oomph::Vector<oomph::Vector<int> *> tmp_vect_pt;

		unsigned nel = nelement();
		for (unsigned e = 0; e < nel; e++)
		{
			oomph::FiniteElement *fe_pt = finite_element_pt(e);
			if (!dynamic_cast<oomph::BrickElementBase *>(fe_pt))
			{
				continue;
			} // Don't do this on tris
			if (doc)
				outfile << "Element: " << e << " " << fe_pt << std::endl;
			unsigned nnode_1d = fe_pt->nnode_1d();
			// Loop over all nodes of the element in the 3d tensor-product node ordering
			for (unsigned i0 = 0; i0 < nnode_1d; i0++)
			{
				for (unsigned i1 = 0; i1 < nnode_1d; i1++)
				{
					for (unsigned i2 = 0; i2 < nnode_1d; i2++)
					{
						unsigned j = i0 + i1 * nnode_1d + i2 * nnode_1d * nnode_1d;
						std::set<unsigned> *boundaries_pt = 0;
						fe_pt->node_pt(j)->get_boundaries_pt(boundaries_pt);
						if (boundaries_pt != 0)
						{
							for (std::set<unsigned>::iterator it = boundaries_pt->begin(); it != boundaries_pt->end(); ++it)
							{
								unsigned boundary_id = *it;
								oomph::Vector<oomph::FiniteElement *>::iterator b_el_it =
									std::find(vector_of_boundary_element_pt[*it].begin(),
											  vector_of_boundary_element_pt[*it].end(),
											  fe_pt);

								if (b_el_it == vector_of_boundary_element_pt[*it].end())
								{
									vector_of_boundary_element_pt[*it].push_back(fe_pt);
								}

								if (boundary_identifier(boundary_id, fe_pt) == 0)
								{
									oomph::Vector<int> *tmp_pt = new oomph::Vector<int>;
									tmp_vect_pt.push_back(tmp_pt);
									boundary_identifier(boundary_id, fe_pt) = tmp_pt;
								}

								// Only corner nodes (vertices of the brick) can identify a whole face;
								// each corner belongs to exactly 3 of the 6 faces, so push one signed
								// indicator per relevant local direction (+/-1 for x, +/-2 for y, +/-3 for z).
								if (((i0 == 0) || (i0 == nnode_1d - 1)) && ((i1 == 0) || (i1 == nnode_1d - 1)) && ((i2 == 0) || (i2 == nnode_1d - 1)))
								{
									(*boundary_identifier(boundary_id, fe_pt)).push_back(1 * (2 * i0 / (nnode_1d - 1) - 1));
									(*boundary_identifier(boundary_id, fe_pt)).push_back(2 * (2 * i1 / (nnode_1d - 1) - 1));
									(*boundary_identifier(boundary_id, fe_pt)).push_back(3 * (2 * i2 / (nnode_1d - 1) - 1));
								}
							}
						}
					}
				}
			}
		}

		// For each element found to touch a boundary, tally how many corner nodes
		// contributed each possible face indicator; an indicator that was contributed
		// by all 4 corners of a face confirms that face lies entirely on the boundary.
		for (unsigned i = 0; i < nbound; i++)
		{
			// Loop over elements on given boundary
			typedef oomph::Vector<oomph::FiniteElement *>::iterator IT;
			for (IT it = vector_of_boundary_element_pt[i].begin();
				 it != vector_of_boundary_element_pt[i].end();
				 it++)
			{
				oomph::FiniteElement *fe_pt = (*it);
				std::map<int, unsigned> count;
				for (int ii = 0; ii < 3; ii++)
				{
					for (int sign = -1; sign < 3; sign += 2)
					{
						count[(ii + 1) * sign] = 0;
					}
				}

				unsigned n_indicators = (*boundary_identifier(i, fe_pt)).size();
				for (unsigned j = 0; j < n_indicators; j++)
				{
					count[(*boundary_identifier(i, fe_pt))[j]]++;
				}

				int indicator = -10;

				// A face has 4 corners, so an indicator seen exactly 4 times means
				// that whole face lies on boundary i; record it as a boundary face.
				for (int ii = 0; ii < 3; ii++)
				{
					for (int sign = -1; sign < 3; sign += 2)
					{
						if (count[(ii + 1) * sign] == 4)
						{
							indicator = (ii + 1) * sign;
							Boundary_element_pt[i].push_back(*it);
							Face_index_at_boundary[i].push_back(indicator);
						}
					}
				}
			}
		}

		// Free the per-(boundary,element) indicator vectors allocated above via
		// boundary_identifier (they were collected in tmp_vect_pt since the map
		// itself does not own them).
		unsigned n = tmp_vect_pt.size();
		for (unsigned i = 0; i < n; i++)
		{
			delete tmp_vect_pt[i];
		}
	}

	// Legacy (non-facet-based) boundary-element detection for the tetrahedral elements of the mesh
	// (despite the name "tris", these are the 4-noded tets of a 3d mesh, analogous to the triangle
	// case in 2d). For each tet, each of its 4 faces is checked by intersecting the boundary-index
	// sets of its 3 corner nodes: if all three nodes share exactly one common boundary, that face is
	// recorded as lying on that boundary (a face shared by more than one boundary triggers a warning,
	// as it indicates a degenerate/too-coarse mesh).
	void TemplatedMeshBase3d::setup_boundary_element_info_tris(std::ostream &)
	{
		unsigned nel = nelement();
		unsigned nbound = nboundary();
		oomph::Vector<oomph::Vector<oomph::FiniteElement *>> vector_of_boundary_element_pt;
		vector_of_boundary_element_pt.resize(nbound);
		// Matrix map for working out the fixed face for elements on boundary
		oomph::MapMatrixMixed<unsigned, oomph::FiniteElement *, int> face_identifier;
		oomph::Vector<std::set<unsigned> *> boundaries_pt(4, 0);

		for (unsigned e = 0; e < nel; e++)
		{
			// Get pointer to element
			oomph::FiniteElement *fe_pt = finite_element_pt(e);
			if (!dynamic_cast<oomph::TElementBase *>(fe_pt))
				continue; // Only on triangles
			// Only include 3D elements! Some meshes contain interface elements too.
			if (fe_pt->dim() == 3)
			{
				for (unsigned i = 0; i < 4; i++)
				{
					fe_pt->node_pt(i)->get_boundaries_pt(boundaries_pt[i]);
				}
				oomph::Vector<std::set<unsigned>> face(4);

				// Face 3 connnects points 0, 1 and 2
				if (boundaries_pt[0] && boundaries_pt[1] && boundaries_pt[2])
				{
					std::set<unsigned> aux;

					std::set_intersection(boundaries_pt[0]->begin(), boundaries_pt[0]->end(),
										  boundaries_pt[1]->begin(), boundaries_pt[1]->end(),
										  std::insert_iterator<std::set<unsigned>>(
											  aux, aux.begin()));

					std::set_intersection(aux.begin(), aux.end(),
										  boundaries_pt[2]->begin(), boundaries_pt[2]->end(),
										  std::insert_iterator<std::set<unsigned>>(
											  face[3], face[3].begin()));
				}

				if (boundaries_pt[0] && boundaries_pt[1] && boundaries_pt[3])
				{
					std::set<unsigned> aux;

					std::set_intersection(boundaries_pt[0]->begin(), boundaries_pt[0]->end(),
										  boundaries_pt[1]->begin(), boundaries_pt[1]->end(),
										  std::insert_iterator<std::set<unsigned>>(
											  aux, aux.begin()));

					std::set_intersection(aux.begin(), aux.end(),
										  boundaries_pt[3]->begin(), boundaries_pt[3]->end(),
										  std::insert_iterator<std::set<unsigned>>(
											  face[2], face[2].begin()));
				}

				// Face 1 connects points 0, 2 and 3
				if (boundaries_pt[0] && boundaries_pt[2] && boundaries_pt[3])
				{
					std::set<unsigned> aux;

					std::set_intersection(boundaries_pt[0]->begin(), boundaries_pt[0]->end(),
										  boundaries_pt[2]->begin(), boundaries_pt[2]->end(),
										  std::insert_iterator<std::set<unsigned>>(
											  aux, aux.begin()));

					std::set_intersection(aux.begin(), aux.end(),
										  boundaries_pt[3]->begin(), boundaries_pt[3]->end(),
										  std::insert_iterator<std::set<unsigned>>(
											  face[1], face[1].begin()));
				}

				// Face 0 connects points 1, 2 and 3
				if (boundaries_pt[1] && boundaries_pt[2] && boundaries_pt[3])
				{
					std::set<unsigned> aux;

					std::set_intersection(boundaries_pt[1]->begin(), boundaries_pt[1]->end(),
										  boundaries_pt[2]->begin(), boundaries_pt[2]->end(),
										  std::insert_iterator<std::set<unsigned>>(
											  aux, aux.begin()));

					std::set_intersection(aux.begin(), aux.end(),
										  boundaries_pt[3]->begin(), boundaries_pt[3]->end(),
										  std::insert_iterator<std::set<unsigned>>(
											  face[0], face[0].begin()));
				}

				// We now know whether any faces lay on the boundaries
				for (unsigned i = 0; i < 4; i++)
				{
					// How many boundaries are there
					unsigned count = 0;

					// The number of the boundary
					int boundary = -1;

					// Loop over all the members of the set and add to the count
					// and set the boundary
					for (std::set<unsigned>::iterator it = face[i].begin();
						 it != face[i].end(); ++it)
					{
						++count;
						boundary = *it;
					}

					// If we're on more than one boundary, this is weird, so die
					if (count > 1)
					{
						std::ostringstream error_stream;
						fe_pt->output(error_stream);
						error_stream << "Face " << i << " is on " << count << " boundaries.\n";
						error_stream << "This is rather strange.\n";
						error_stream << "Your mesh may be too coarse or your tet mesh\n";
						error_stream << "may be screwed up. I'm skipping the automated\n";
						error_stream << "setup of the elements next to the boundaries\n";
						error_stream << "lookup schemes.\n";
						oomph::OomphLibWarning(
							error_stream.str(),
							OOMPH_CURRENT_FUNCTION,
							OOMPH_EXCEPTION_LOCATION);
					}

					// If we have a boundary then add this to the appropriate set
					if (boundary >= 0)
					{

						// Does the pointer already exits in the vector
						oomph::Vector<oomph::FiniteElement *>::iterator b_el_it =
							std::find(vector_of_boundary_element_pt[static_cast<unsigned>(boundary)].begin(),
									  vector_of_boundary_element_pt[static_cast<unsigned>(boundary)].end(),
									  fe_pt);

						// Only insert if we have not found it (i.e. got to the end)
						if (b_el_it == vector_of_boundary_element_pt[static_cast<unsigned>(boundary)].end())
						{
							vector_of_boundary_element_pt[static_cast<unsigned>(boundary)].push_back(fe_pt);
						}

						// Also set the fixed face
						face_identifier(static_cast<unsigned>(boundary), fe_pt) = i;
					}
				}

				// Now we set the pointers to the boundary sets to zero
				for (unsigned i = 0; i < 4; i++)
				{
					boundaries_pt[i] = 0;
				}
			}
		}

		// Now copy everything across into permanent arrays
		//-------------------------------------------------

		// Loop over boundaries
		//---------------------
		for (unsigned i = 0; i < nbound; i++)
		{
			// Number of elements on this boundary (currently stored in a set)
			unsigned nel = vector_of_boundary_element_pt[i].size();
			unsigned e_count = Face_index_at_boundary[i].size();
			Face_index_at_boundary[i].resize(e_count + nel);

			typedef oomph::Vector<oomph::FiniteElement *>::iterator IT;
			for (IT it = vector_of_boundary_element_pt[i].begin();
				 it != vector_of_boundary_element_pt[i].end();
				 it++)
			{
				// Recover pointer to element
				oomph::FiniteElement *fe_pt = *it;

				// Add to permanent storage
				Boundary_element_pt[i].push_back(fe_pt);

				Face_index_at_boundary[i][e_count] = face_identifier(i, fe_pt);

				// Increment counter
				e_count++;
			}
		}
	}

}
