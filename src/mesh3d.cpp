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
		bool allQ = true, allT = true, allW = true, allP = true;
		for (unsigned int i = 0; i < this->nelement(); i++)
		{
			bool is_brick = (dynamic_cast<oomph::BrickElementBase *>(this->element_pt(i)) != NULL);
			bool is_tet = (dynamic_cast<oomph::TElementBase *>(this->element_pt(i)) != NULL);
			bool is_wedge = (dynamic_cast<oomph::RefineableWedgeElement *>(this->element_pt(i)) != NULL);
			bool is_pyramid = (dynamic_cast<oomph::RefineablePyramidElement *>(this->element_pt(i)) != NULL);
			allQ = allQ && is_brick;
			allT = allT && is_tet;
			allW = allW && is_wedge;
			allP = allP && is_pyramid;
		}
		// Pure-brick, pure-tet, or (branch mixed_adapt) pure-wedge meshes refine 1->8 via the OcTree; a
		// wedge refines its triangular cross-section 1->4 and its extrusion 1->2 (shape-closed). A pure-
		// pyramid mesh refines with HETEROGENEOUS offspring (1 pyramid -> 6 pyramids + 4 tets), so its
		// leaves become a mix of pyramids and tets after the first level.
		if (allQ || allT || allW || allP)
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

	// Tetrahedral hanging nodes are now installed PER ELEMENT by oomph's adapt loop, fully topologically:
	// RefineableTElement<3>::setup_hanging_nodes (geometric slot -1) and setup_hang_for_value (the separate
	// C1/C2 value slots, driven from BulkElementTetra3dC2::further_setup_hanging_nodes, which also applies the
	// C1(TB) mid-node rule) -- each via tet_hang_face / tet_hang_edge (OcTree neighbour finders + the exact
	// affine map + interpolating_basis). oomph's complete_hanging_nodes then flattens the recursive master
	// chains (an earlier explicit mesh-level re-flatten here was not only redundant once the install moved
	// into the hooks -- it ran AFTER complete_hanging_nodes and could corrupt an already-correct scheme).
	// So this pass has nothing left to do for tets; it replaced the former mesh-level geometric facet-
	// adjacency passes. Being per-element, the scheme now also covers refinement paths that bypass this
	// pyoomph pass entirely -- refine_selected_elements / custom_adapt (the tet analogue of the 2d fix).
	void TemplatedMeshBase3d::post_adapt_setup_hanging_nodes()
	{
		// --- Pyramid forest 2:1 hanging (M3c), CROSS-SHAPE. -----------------------------------------------
		// A pyramid refines into 6 sub-pyramids + 4 tets, so after non-uniform refinement the leaves are a
		// MIX of pyramids and tets. Same generative, position-based idea as the wedge branch below (NOT
		// locate_zeta): for every leaf element and every face, walk the face's fine lattice (step
		// h=0.5/(nnode_1d-1)), map each point to physical x via interpolated_x, look the node up by position,
		// and hang it on THIS (coarse) element's interpolating_basis via mixed_hang_node_at. Shape-agnostic on
		// the finer side (it only needs the node's position); the coarse side may be a pyramid (4 tri + 1 quad
		// face) or a tet (4 tri faces). The pyramid's triangular faces meet at the apex (s2=1) where the
		// 1/(1-s2) shape is singular, so lattice points with a non-finite image are skipped (only the apex
		// VERTEX is affected, never a hang node, which sits at s2<=0.5).
		{
			bool has_pyr = false, all_pyr_or_tet = true;
			for (unsigned int ie = 0; ie < this->nelement(); ie++)
			{
				const bool isp = (dynamic_cast<oomph::RefineablePyramidElement *>(this->element_pt(ie)) != nullptr);
				const bool ist = (dynamic_cast<oomph::TElementBase *>(this->element_pt(ie)) != nullptr);
				has_pyr = has_pyr || isp;
				all_pyr_or_tet = all_pyr_or_tet && (isp || ist);
			}
			if (has_pyr && all_pyr_or_tet)
			{
				// Face tables (corner local coords). Pyramid vertices 0(0,0,0)1(1,0,0)2(1,1,0)3(0,1,0)4(0,0,1)
				// apex; tet vertices 0(0,0,0)1(1,0,0)2(0,1,0)3(0,0,1). Matching get_vertex_nodes_of_face.
				struct PFace { bool quad; int nc; double c[4][3]; };
				static const PFace PYR_FACES[5] = {
					{false, 3, {{0, 0, 0}, {1, 0, 0}, {0, 0, 1}, {0, 0, 0}}}, // 0 tri s1=0
					{false, 3, {{1, 0, 0}, {1, 1, 0}, {0, 0, 1}, {0, 0, 0}}}, // 1 tri s0=1-s2
					{false, 3, {{1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}}}, // 2 tri s1=1-s2
					{false, 3, {{0, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}}}, // 3 tri s0=0
					{true,  4, {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}}};// 4 quad base s2=0
				static const PFace TET_FACES[4] = {
					{false, 3, {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}}}, // 0
					{false, 3, {{0, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}}}, // 1
					{false, 3, {{0, 0, 0}, {1, 0, 0}, {0, 0, 1}, {0, 0, 0}}}, // 2
					{false, 3, {{1, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 0, 0}}}};// 3

				for (unsigned int in = 0; in < this->nnode(); in++) this->node_pt(in)->set_nonhanging();
				const double scale = 1e8;
				std::map<std::array<long long, 3>, oomph::Node *> node_at;
				for (unsigned int in = 0; in < this->nnode(); in++)
				{
					oomph::Node *n = this->node_pt(in);
					node_at[{(long long)std::llround(n->x(0) * scale), (long long)std::llround(n->x(1) * scale), (long long)std::llround(n->x(2) * scale)}] = n;
				}

				const int ncont = this->nelement() ? (int)dynamic_cast<oomph::RefineableElement *>(this->element_pt(0))->ncont_interpolated_values() : 0;
				for (unsigned int ie = 0; ie < this->nelement(); ie++)
				{
					BulkElementBase *be = dynamic_cast<BulkElementBase *>(this->element_pt(ie));
					oomph::RefineableElement *re = dynamic_cast<oomph::RefineableElement *>(this->element_pt(ie));
					oomph::FiniteElement *fe = dynamic_cast<oomph::FiniteElement *>(this->element_pt(ie));
					if (!be || !re || !fe || !re->tree_pt() || !re->tree_pt()->is_leaf()) continue;
					const bool is_pyr = (dynamic_cast<oomph::RefineablePyramidElement *>(this->element_pt(ie)) != nullptr);
					const PFace *FACES = is_pyr ? PYR_FACES : TET_FACES;
					const int nfaces = is_pyr ? 5 : 4;
					const unsigned n1d = fe->nnode_1d();
					const int steps = 2 * ((int)n1d - 1);
					if (steps < 1) continue;
					const double h = 1.0 / steps;
					for (int f = 0; f < nfaces; f++)
					{
						const PFace &F = FACES[f];
						for (int iu = 0; iu <= steps; iu++)
							for (int iv = 0; iv <= steps; iv++)
							{
								const double u = iu * h, v = iv * h;
								if (!F.quad && u + v > 1.0 + 1e-12) continue; // tri face: barycentric u+v<=1
								oomph::Vector<double> s(3), x(3);
								for (int d = 0; d < 3; d++)
								{
									if (F.quad)
										s[d] = (1 - u) * (1 - v) * F.c[0][d] + u * (1 - v) * F.c[1][d] + u * v * F.c[2][d] + (1 - u) * v * F.c[3][d];
									else
										s[d] = F.c[0][d] + u * (F.c[1][d] - F.c[0][d]) + v * (F.c[2][d] - F.c[0][d]);
								}
								be->interpolated_x(s, x);
								if (!std::isfinite(x[0]) || !std::isfinite(x[1]) || !std::isfinite(x[2])) continue; // apex (s2=1)
								std::array<long long, 3> key = {(long long)std::llround(x[0] * scale), (long long)std::llround(x[1] * scale), (long long)std::llround(x[2] * scale)};
								auto it = node_at.find(key);
								if (it == node_at.end()) continue;
								oomph::Node *H = it->second;
								be->mixed_hang_node_at(H, re, s, -1);
								for (int val = 0; val < ncont; val++) be->mixed_hang_node_at(H, re, s, val);
							}
					}
				}
				return;
			}
		}

		// --- Wedge (triangular-prism) 2:1 hanging (M2). ---------------------------------------------------
		// Wedges do not (yet) have a topological OcTree neighbour finder, so -- like the tet route did before
		// its per-element migration -- non-uniform (2:1) wedge hanging is installed by this mesh-level pass.
		// It is NOT locate_zeta: for each wedge and each of its 5 faces we visit the FIXED local coordinates
		// where a one-level-finer neighbour places nodes, find the node sitting there by position, and -- if
		// it is not one of this element's own interpolating nodes -- hang it on this element's
		// interpolating_basis at that exactly-known local coordinate via mixed_hang_node_at. The candidate
		// coordinates are GENERATED from each face's fine lattice at step h = 0.5/(nnode_1d-1): a 2:1-finer
		// neighbour refines the face 1->4 and places its order-p nodes at that spacing (C1 -> 0.5, i.e. edge
		// midpoints + quad centre; C2 -> 0.25, i.e. additionally the sub-face quarter points). Coordinates
		// that coincide with this element's own nodes are skipped by mixed_hang_node_at. Combined with
		// enforce_refinement_balance (2:1), the masters are the coarse element's real nodes -> no flattening.
		bool has_wedge = false, all_wedge = true;
		for (unsigned int ie = 0; ie < this->nelement(); ie++)
		{
			bool w = (dynamic_cast<oomph::RefineableWedgeElement *>(this->element_pt(ie)) != nullptr);
			has_wedge = has_wedge || w;
			all_wedge = all_wedge && w;
		}
		if (!has_wedge || !all_wedge)
			return; // tets: per-element hooks; bricks: oomph; mixed: not supported yet

		// The 5 wedge faces as {is_quad, 3 or 4 corner local coords}. (s0,s1) is the triangular cross-section,
		// s2 the extrusion. Faces: 0=bottom tri (s2=0), 1=top tri (s2=1), 2=quad s0=0, 3=quad s1=0,
		// 4=quad hypotenuse s0+s1=1 (see BulkElementWedge3dC1::get_vertex_nodes_of_face).
		struct WFace { bool quad; double c[4][3]; };
		static const WFace FACES[5] = {
			{false, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 0}}},                 // 0 bottom tri
			{false, {{0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {0, 0, 0}}},                 // 1 top tri
			{true,  {{0, 0, 0}, {0, 1, 0}, {0, 1, 1}, {0, 0, 1}}},                 // 2 quad s0=0
			{true,  {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}}},                 // 3 quad s1=0
			{true,  {{1, 0, 0}, {0, 1, 0}, {0, 1, 1}, {1, 0, 1}}}};                // 4 quad hypotenuse

		// Reset, then build a position -> node lookup over the whole mesh.
		for (unsigned int in = 0; in < this->nnode(); in++) this->node_pt(in)->set_nonhanging();
		const double scale = 1e8;
		std::map<std::array<long long, 3>, oomph::Node *> node_at;
		for (unsigned int in = 0; in < this->nnode(); in++)
		{
			oomph::Node *n = this->node_pt(in);
			node_at[{(long long)std::llround(n->x(0) * scale), (long long)std::llround(n->x(1) * scale), (long long)std::llround(n->x(2) * scale)}] = n;
		}

		const int ncont = this->nelement() ? (int)dynamic_cast<oomph::RefineableElement *>(this->element_pt(0))->ncont_interpolated_values() : 0;
		for (unsigned int ie = 0; ie < this->nelement(); ie++)
		{
			BulkElementBase *be = dynamic_cast<BulkElementBase *>(this->element_pt(ie));
			oomph::RefineableElement *re = dynamic_cast<oomph::RefineableElement *>(this->element_pt(ie));
			oomph::FiniteElement *fe = dynamic_cast<oomph::FiniteElement *>(this->element_pt(ie));
			if (!be || !re || !fe || !re->tree_pt() || !re->tree_pt()->is_leaf()) continue;
			const unsigned n1d = fe->nnode_1d();
			const int steps = 2 * ((int)n1d - 1); // lattice divisions per face direction (C1:2, C2:4)
			if (steps < 1) continue;
			const double h = 1.0 / steps;
			for (int f = 0; f < 5; f++)
			{
				const WFace &F = FACES[f];
				for (int iu = 0; iu <= steps; iu++)
					for (int iv = 0; iv <= steps; iv++)
					{
						const double u = iu * h, v = iv * h;
						if (!F.quad && u + v > 1.0 + 1e-12) continue; // triangular face: barycentric u+v<=1
						oomph::Vector<double> s(3), x(3);
						for (int d = 0; d < 3; d++)
						{
							if (F.quad) // bilinear blend of the 4 corners
								s[d] = (1 - u) * (1 - v) * F.c[0][d] + u * (1 - v) * F.c[1][d] + u * v * F.c[2][d] + (1 - u) * v * F.c[3][d];
							else        // triangular: c0 + u(c1-c0) + v(c2-c0)
								s[d] = F.c[0][d] + u * (F.c[1][d] - F.c[0][d]) + v * (F.c[2][d] - F.c[0][d]);
						}
						be->interpolated_x(s, x);
						std::array<long long, 3> key = {(long long)std::llround(x[0] * scale), (long long)std::llround(x[1] * scale), (long long)std::llround(x[2] * scale)};
						std::map<std::array<long long, 3>, oomph::Node *>::iterator it = node_at.find(key);
						if (it == node_at.end()) continue; // no node here -> conforming/boundary, nothing to hang
						oomph::Node *H = it->second;
						be->mixed_hang_node_at(H, re, s, -1); // skips H if it is this element's own node
						for (int val = 0; val < ncont; val++) be->mixed_hang_node_at(H, re, s, val);
					}
			}
		}
	}

	// Enforce 2:1 refinement balancing for tetrahedral meshes (see header). Iteratively refine any
	// leaf tet that has a node at the quarter point (t=0.25 or 0.75) of one of its edges -- which
	// means a neighbour is >=2 refinement levels finer than it -- until no such tet remains. Each
	// refinement round uses oomph-lib's refine_selected_elements (full, correct rebuild).
	void TemplatedMeshBase3d::enforce_refinement_balance()
	{
		bool has_simplexlike = false;
		for (unsigned int ie = 0; ie < this->nelement() && !has_simplexlike; ie++)
			if (dynamic_cast<oomph::TElementBase *>(this->element_pt(ie)) || dynamic_cast<oomph::RefineableWedgeElement *>(this->element_pt(ie)) ||
				dynamic_cast<oomph::RefineablePyramidElement *>(this->element_pt(ie)))
				has_simplexlike = true;
		if (!has_simplexlike) return; // brick/hex meshes: oomph-lib's tree bounds the level difference itself

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
				oomph::RefineableWedgeElement *wedge = dynamic_cast<oomph::RefineableWedgeElement *>(this->element_pt(ie));
				oomph::RefineablePyramidElement *pyr = dynamic_cast<oomph::RefineablePyramidElement *>(this->element_pt(ie));
				oomph::FiniteElement *fe = dynamic_cast<oomph::FiniteElement *>(this->element_pt(ie));
				if ((!tet && !wedge && !pyr) || !fe) continue;
				oomph::RefineableElement *re = dynamic_cast<oomph::RefineableElement *>(this->element_pt(ie));
				if (re && re->refinement_level() >= this->max_refinement_level()) continue; // cannot refine further
				// This element's edges as vertex-index pairs: a tet's 6, a wedge's 9 (2 tri caps + 3
				// verticals), or a pyramid's 8 (4 base + 4 lateral to the apex). Only genuine edges (not face
				// diagonals), so a node at their 1/2^nnode_1d quarter-point signals a neighbour >=2 levels
				// finer (see the C1:1/4, C2:1/8 note below).
				static const int TET_E[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
				static const int WEDGE_E[9][2] = {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}, {0, 3}, {1, 4}, {2, 5}};
				static const int PYR_E[8][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}, {1, 4}, {2, 4}, {3, 4}};
				const int(*E)[2] = tet ? TET_E : (wedge ? WEDGE_E : PYR_E);
				const int nE = tet ? 6 : (wedge ? 9 : 8);
				// The edge-fraction whose presence signals a neighbour >=2 levels finer. A 1-level
				// neighbour subdivides the edge at its midpoint (t=1/2) and, for order p>=2 (e.g. C2),
				// additionally places the sub-edges' own mid-nodes at t=1/4, 3/4 -- so the finest node a
				// 2:1-allowed neighbour produces is at t=1/2^nnode_1d. A node at that fraction therefore
				// means the neighbour is >=2 levels finer. C1 (nnode_1d=2): 1/4; C2 (nnode_1d=3): 1/8.
				const double frac = 1.0 / (double)(1u << fe->nnode_1d());
				const double fracs[2] = {frac, 1.0 - frac};
				bool imbalanced = false;
				for (int e = 0; e < nE && !imbalanced; e++)
				{
					oomph::Node *va = fe->vertex_node_pt(E[e][0]);
					oomph::Node *vb = fe->vertex_node_pt(E[e][1]);
					for (int fi = 0; fi < 2; fi++)
					{
						std::array<long long, 3> q;
						for (int d = 0; d < 3; d++)
							q[d] = (long long)std::llround((va->x(d) + fracs[fi] * (vb->x(d) - va->x(d))) * scale);
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
	// Normally the single, shape-neutral identification from the per-element face boundary tags,
	// which is exact on arbitrarily refined meshes; only untagged meshes use the brick- and
	// tet-specific legacy helpers below.
	void TemplatedMeshBase3d::setup_boundary_element_info(std::ostream &outfile)
	{

		unsigned nbound = nboundary();

		Boundary_element_pt.clear();
		Face_index_at_boundary.clear();
		Boundary_element_pt.resize(nbound);
		Face_index_at_boundary.resize(nbound);

		// Unified, shape-neutral identification from the per-element face boundary tags; see
		// TemplatedMeshBase::setup_boundary_element_info_from_face_tags. Meshes without tags fall
		// back to the legacy per-shape reconstruction below.
		if (face_boundary_tags_valid)
		{
			setup_boundary_element_info_from_face_tags();
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
