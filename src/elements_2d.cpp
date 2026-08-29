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


// Concrete 2-d bulk elements: quads (C1/C2) and triangles (C1/C2, with and without bubble
// enrichment). Shape functions, interpolating-node overrides for the mixed C1-in-C2 spaces,
// numpy/outline export and the static space-index tables.

#include "macroelements.hpp"
#include "elements.hpp"
#include "elements_concrete.hpp"
#include "exception.hpp"
#include "problem.hpp"
#include "nodes.hpp"
#include "meshtemplate.hpp"
#include "expressions.hpp"
#include "thirdparty/delaunator.hpp" // only TU that may include it: it has non-inline definitions
#include "timestepper.hpp"

namespace pyoomph
{

	void BulkElementQuad2dC1::quad_hang_helper(const int &value_id, const int &my_edge, std::ofstream &output_hangfile)
	{
		oomph::Vector<unsigned> translate_s(2);
		oomph::Vector<double> s_lo(2), s_hi(2);
		int neigh_edge, diff_level;
		bool in_tree;
		oomph::QuadTree *neigh = quadtree_pt()->gteq_edge_neighbour(my_edge, translate_s, s_lo, s_hi, neigh_edge, diff_level, in_tree);
		if (neigh && dynamic_cast<oomph::TElementBase *>(neigh->object_pt()))
		{
			// Cross-shape tri neighbour: hang on the correct tri LEAF (found by descending the tri ROOT per
			// node), which is coarser-or-equal. Pass the tri ROOT; the level guard in mixed_hang_edge_node
			// hangs only where the tri leaf is strictly coarser (equal is shared, finer hangs from its side).
			this->mixed_quad_edge_hang(value_id, my_edge, dynamic_cast<oomph::RefineableElement *>(neigh->root_pt()->object_pt()));
			return;
		}
		oomph::RefineableQElement<2>::quad_hang_helper(value_id, my_edge, output_hangfile);
	}

	void BulkElementQuad2dC2::quad_hang_helper(const int &value_id, const int &my_edge, std::ofstream &output_hangfile)
	{
		oomph::Vector<unsigned> translate_s(2);
		oomph::Vector<double> s_lo(2), s_hi(2);
		int neigh_edge, diff_level;
		bool in_tree;
		oomph::QuadTree *neigh = quadtree_pt()->gteq_edge_neighbour(my_edge, translate_s, s_lo, s_hi, neigh_edge, diff_level, in_tree);
		if (neigh && dynamic_cast<oomph::TElementBase *>(neigh->object_pt()))
		{
			// Cross-shape tri neighbour: hang on the correct tri LEAF (found by descending the tri ROOT per
			// node), which is coarser-or-equal. Pass the tri ROOT; the level guard in mixed_hang_edge_node
			// hangs only where the tri leaf is strictly coarser (equal is shared, finer hangs from its side).
			this->mixed_quad_edge_hang(value_id, my_edge, dynamic_cast<oomph::RefineableElement *>(neigh->root_pt()->object_pt()));
			return;
		}
		oomph::RefineableQElement<2>::quad_hang_helper(value_id, my_edge, output_hangfile);
	}

	// Quad build() overrides. oomph's RefineableSolidQElement<2>::build calls the plain quad build --
	// which does place new nodes through the macro map -- and then overwrites every node with the FE
	// interpolation, so on a curved boundary the macro element had no effect at all and the geometry was
	// only ever repaired afterwards by Problem.map_nodes_on_macro_elements(), which the runtime adapt()
	// path never calls. Re-apply the macro positions here, once that build has returned.
	void BulkElementQuad2dC1::build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt, bool &was_already_built, std::ofstream &new_nodes_file)
	{
		oomph::RefineableSolidQElement<2>::build(mesh_pt, new_node_pt, was_already_built, new_nodes_file);
		if (was_already_built) return;
		this->reapply_macro_element_positions();
	}

	void BulkElementQuad2dC2::build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt, bool &was_already_built, std::ofstream &new_nodes_file)
	{
		oomph::RefineableSolidQElement<2>::build(mesh_pt, new_node_pt, was_already_built, new_nodes_file);
		if (was_already_built) return;
		this->reapply_macro_element_positions();
	}

	

	////////////////////////////

	// BulkElementQuad2dC1: bilinear (4-node) quadrilateral element. C1/C1TB use all 4 corner
	// nodes; DL is a 3-value (constant + 2 gradient components) discontinuous linear
	// representation, not tied to any actual node.
	BulkElementQuad2dC1::BulkElementQuad2dC1()
	{
		eleminfo.elem_ptr = static_cast<BulkElementBase *>(this);
		eleminfo.nnode = 4;
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 4;
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = 4;
		eleminfo.nnode_DL = 3;
		eleminfo.nodal_dim = jitcode->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}


	void BulkElementQuad2dC1::shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		psi[2] = s[1];
	}

	void BulkElementQuad2dC1::dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		psi[2] = s[1];
		dpsi(0, 0) = 0.0;
		dpsi(1, 0) = 1.0;
		dpsi(2, 0) = 0.0;
		dpsi(0, 1) = 0.0;
		dpsi(1, 1) = 0.0;
		dpsi(2, 1) = 1.0;
	}

	// Called (by a finer neighbor, via inform_coarser_neighbors_for_tesselated_numpy) to register
	// an extra hanging node `n` sitting on this (coarser) element's edge `edge`, so that when this
	// element is tesselated for numpy/plotting export, it can insert extra triangles that connect
	// to the finer neighbor's edge nodes instead of leaving a T-junction gap in the visualization.
	void BulkElementQuad2dC1::add_node_from_finer_neighbor_for_tesselated_numpy(const oomph::Vector<double> &s_coarse, oomph::Node *n, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes)
	{
		// edgedir 0=S(0,1) 1=N(2,3) 2=W(0,2) 3=E(1,3), matching the corner_pairs used in the fill below.
		this->tess_register_hanging_node(s_coarse, n, {{0, 1}, {2, 3}, {0, 2}, {1, 3}}, add_nodes);
	}

	// For every edge of this element, checks (via gteq_edge_neighbour) whether the neighbouring element is on
	// a coarser refinement level and, if so, registers this element's edge nodes with that coarser neighbour --
	// at each node's TOPOLOGICALLY-computed coordinate in the neighbour -- so the numpy tesselation of the
	// coarser element can be split to avoid a visible crack at the junction.
	void BulkElementQuad2dC1::inform_coarser_neighbors_for_tesselated_numpy(std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes)
	{
		if (this->as_interface_element())
			throw_runtime_error("Cannot yet tesselate interface meshes [will fail in connecting hanging nodes and have to go via the parent mesh");
		this->quad_register_on_coarser_for_numpy(add_nodes);
	}

	// Number of sub-elements (nsubdiv) this quad is tesselated into for numpy/plotting export, and
	// the number of vertices per sub-element (returned value): if not tesselating into triangles,
	// it is exported as a single quad (4 indices); if tesselating into triangles, the base case is
	// 2 triangles, plus one extra triangle for every hanging node contributed by finer neighbors
	// on this element's edges (see inform_coarser_neighbors_for_tesselated_numpy()).
	int BulkElementQuad2dC1::get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
	{
		if (tesselate_tri)
		{
			// Check my neighbors whether they are on a finer level. If so, we need to add more triangles here
			if (add_nodes[this->_numpy_index].empty())
			{
				nsubdiv = 2;
			}
			else
			{
				unsigned tricnt = 0;
				for (unsigned int dir = 0; dir < 4; dir++)
				{
					tricnt += add_nodes[this->_numpy_index][dir].size();
				}
				nsubdiv = 2 + tricnt;
			}
			return 3;
		}
		else
		{
			nsubdiv = 1;
			return 4;
		}
	}

	// Fills in the node indices of sub-element isubelem. Without triangle tesselation, exports the
	// quad as-is (indices 0,1,2,3). With tesselation and no hanging edge nodes to worry about, uses
	// a fixed 2-triangle split (0,1,2 and 2,1,3). If hanging nodes from finer neighbors were
	// registered on this element's edges, instead builds a local coordinate for every added node
	// (by linear interpolation along the edge it sits on) and re-triangulates the whole node set
	// with a Delaunay triangulation, so the boundary triangles line up exactly with the finer
	// neighbor's edge subdivisions (avoiding visible gaps/T-junctions in the exported mesh).
	void BulkElementQuad2dC1::fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
	{
		if (tesselate_tri)
		{
			if (add_nodes[this->_numpy_index].empty())
			{
				if (!isubelem)
				{
					indices[0] = 0;
					indices[1] = 1;
					indices[2] = 2;
				}
				else
				{
					indices[0] = 2;
					indices[1] = 1;
					indices[2] = 3;
				}
			}
			else
			{

				int cnt = this->nnode();
				for (unsigned int d = 0; d < 4; d++)
				{
					cnt += add_nodes[this->_numpy_index][d].size();
					/*		   for (auto * n : add_nodes[this->_numpy_index][d])
								{
								  cnt++;
								}*/
				}
				// Now find the s coordinates of all nodes
				std::vector<oomph::Vector<double>> scoords(cnt);
				for (unsigned int i = 0; i < this->nnode(); i++)
				{
					this->local_coordinate_of_node(i, scoords[i]);
				}
				cnt = this->nnode();
				for (unsigned int d = 0; d < 4; d++)
				{
					for (auto *n : add_nodes[this->_numpy_index][d])
					{
						// The hanging node's LOCAL coordinate in this element was computed topologically by the
						// finer neighbour and stored in _tess_hang_scoord (exact, no physical blend).
						auto it = this->_tess_hang_scoord.find(n);
						scoords[cnt] = (it != this->_tess_hang_scoord.end()) ? it->second : oomph::Vector<double>(2, 0.0);
						cnt++;
					}
				}
				std::vector<double> incoords(2 * scoords.size());
				for (unsigned int i = 0; i < scoords.size(); i++)
				{
					incoords[2 * i] = scoords[i][0];
					incoords[2 * i + 1] = scoords[i][1];
				}
				delaunator::Delaunator d(incoords);
				//		 std::cout <<"ELEMET GOT THE NODAL PAIRS " << d.triangles.size()/3 << " and index " << isubelem << std::endl;
				indices[0] = d.triangles[3 * isubelem];
				indices[2] = d.triangles[3 * isubelem + 1];
				indices[1] = d.triangles[3 * isubelem + 2];
			}
		}
		else
		{
			indices[0] = 0;
			indices[1] = 1;
			indices[2] = 2;
			indices[3] = 3;
		}
	}

	std::vector<double> BulkElementQuad2dC1::get_outline(bool lagrangian)
	{
		std::vector<double> res(4 * this->nodal_dimension());
		unsigned offs = 0;
		for (unsigned int i = 0; i < this->nodal_dimension(); i++)
		{
		   if (lagrangian)
		   {
			res[0 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(0))->xi(i);
			res[1 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(1))->xi(i);
			res[2 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(3))->xi(i);
			res[3 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(2))->xi(i);
			}		   
		   else
		   {
			res[0 + offs] = this->node_pt(0)->x(i);
			res[1 + offs] = this->node_pt(1)->x(i);
			res[2 + offs] = this->node_pt(3)->x(i);
			res[3 + offs] = this->node_pt(2)->x(i);
			}
			offs += 4;
		}
		return res;
	}

	// Returns the i-th node on the given face of a 2-node-per-side (bilinear) quad; face_index
	// follows oomph-lib's convention: -1/+1 = "west"/"east" side (varying second index), -2/+2 =
	// "south"/"north" side (varying first index).
	oomph::Node *BulkElementQuad2dC1::boundary_node_pt(const int &face_index, const unsigned int i)
	{
		const unsigned nn1d = 2;
		if (face_index == -1)
		{
			return this->node_pt(i * nn1d);
		}
		else if (face_index == +1)
		{
			return this->node_pt(nn1d * i + nn1d - 1);
		}
		else if (face_index == -2)
		{
			return this->node_pt(i);
		}
		else if (face_index == +2)
		{
			return this->node_pt(nn1d * (nn1d - 1) + i);
		}
		else
		{
			std::string err = "Face index should be in {-1, +1, -2, +2}.";
			throw oomph::OomphLibError(err, OOMPH_EXCEPTION_LOCATION, OOMPH_CURRENT_FUNCTION);
		}
	}

	// Same idea as BulkElementLine1dC1::get_nodal_s_in_father() above, but for quadtree
	// refinement: maps node l's local coordinate to the father's local coordinate based on which
	// quadrant (SW/SE/NE/NW) this son occupies.
	void BulkElementQuad2dC1::get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather)
	{
		using namespace oomph::QuadTreeNames;
		sfather.resize(2, 0.0);
		int son_type = Tree_pt->son_type();

		oomph::Vector<double> s_lo(2);
		oomph::Vector<double> s_hi(2);
		oomph::Vector<double> s(2);
		oomph::Vector<double> x(2);
		switch (son_type)
		{
		case SW:
			s_lo[0] = -1.0;
			s_hi[0] = 0.0;
			s_lo[1] = -1.0;
			s_hi[1] = 0.0;
			break;

		case SE:
			s_lo[0] = 0.0;
			s_hi[0] = 1.0;
			s_lo[1] = -1.0;
			s_hi[1] = 0.0;
			break;

		case NE:
			s_lo[0] = 0.0;
			s_hi[0] = 1.0;
			s_lo[1] = 0.0;
			s_hi[1] = 1.0;
			break;

		case NW:
			s_lo[0] = -1.0;
			s_hi[0] = 0.0;
			s_lo[1] = 0.0;
			s_hi[1] = 1.0;
			break;
		}

		//   unsigned jnod=0;
		oomph::Vector<double> x_small(2);
		oomph::Vector<double> x_large(2);

		oomph::Vector<double> s_fraction(2);
		unsigned n_p = nnode_1d();
		unsigned i1 = l / n_p;
		unsigned i0 = l - n_p * i1;
		s_fraction[0] = local_one_d_fraction_of_node(i0, 0);
		sfather[0] = s_lo[0] + (s_hi[0] - s_lo[0]) * s_fraction[0];
		s_fraction[1] = local_one_d_fraction_of_node(i1, 1);
		sfather[1] = s_lo[1] + (s_hi[1] - s_lo[1]) * s_fraction[1];
	}

	////////////////////////////
	

	// BulkElementQuad2dC2: biquadratic (9-node) quadrilateral. C2/C2TB use all 9 nodes; C1/C1TB
	// use only the 4 corner nodes (the classic Q2/Q1 "Taylor-Hood" pairing, e.g. velocity/
	// pressure); DL is again a 3-value discontinuous linear representation.
	BulkElementQuad2dC2::BulkElementQuad2dC2()
	{
		eleminfo.elem_ptr = static_cast<BulkElementBase *>(this);
		// std::cout << "SETTING ELEM PTR " <<  eleminfo.elem_ptr << std::endl;
		eleminfo.nnode = 9;
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 4;
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = 4;
		eleminfo.nnode_of_space[SPACE_INDEX_C2] = 9;		
		eleminfo.nnode_of_space[SPACE_INDEX_C2TB] = 9;
		eleminfo.nnode_DL = 3;
		eleminfo.nodal_dim = jitcode->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}

	void BulkElementQuad2dC2::get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather)
	{
		using namespace oomph::QuadTreeNames;
		sfather.resize(2, 0.0);
		int son_type = Tree_pt->son_type();

		oomph::Vector<double> s_lo(2);
		oomph::Vector<double> s_hi(2);
		oomph::Vector<double> s(2);
		oomph::Vector<double> x(2);
		switch (son_type)
		{
		case SW:
			s_lo[0] = -1.0;
			s_hi[0] = 0.0;
			s_lo[1] = -1.0;
			s_hi[1] = 0.0;
			break;

		case SE:
			s_lo[0] = 0.0;
			s_hi[0] = 1.0;
			s_lo[1] = -1.0;
			s_hi[1] = 0.0;
			break;

		case NE:
			s_lo[0] = 0.0;
			s_hi[0] = 1.0;
			s_lo[1] = 0.0;
			s_hi[1] = 1.0;
			break;

		case NW:
			s_lo[0] = -1.0;
			s_hi[0] = 0.0;
			s_lo[1] = 0.0;
			s_hi[1] = 1.0;
			break;
		}

		//   unsigned jnod=0;
		oomph::Vector<double> x_small(2);
		oomph::Vector<double> x_large(2);

		oomph::Vector<double> s_fraction(2);
		unsigned n_p = nnode_1d();
		unsigned i1 = l / n_p;
		unsigned i0 = l - n_p * i1;
		s_fraction[0] = local_one_d_fraction_of_node(i0, 0);
		sfather[0] = s_lo[0] + (s_hi[0] - s_lo[0]) * s_fraction[0];
		s_fraction[1] = local_one_d_fraction_of_node(i1, 1);
		sfather[1] = s_lo[1] + (s_hi[1] - s_lo[1]) * s_fraction[1];
	}

	// Local coordinate of son node l in the father element for the 1->4 triangle split. This is the
	// same s_in_parent map used in RefineableTElement<2>::build (SW/SE/NE/NW son geometry): corner
	// nodes 0-2 depend on son_type, the C2 mid-edge nodes 3-5 are midpoints of the son corners (in
	// father coords), and the bubble node (l==3 for a 4-node C1TB, l==6 for a 7-node C2TB) is the son
	// centroid. Used by BulkElementBase::further_build() to sample father DG/axisymmetric data at the
	// son nodes.
	static void tri2d_nodal_s_in_father(int son_type, unsigned n_node,
	                                    const unsigned &l, oomph::Vector<double> &sfather)
	{
		using namespace oomph::QuadTreeNames;
		double c[3][2]; // son corner vertices, in father local coordinates
		switch (son_type)
		{
		case SW:
			c[0][0] = 0.5; c[0][1] = 0.0; c[1][0] = 0.0; c[1][1] = 0.5; c[2][0] = 0.0; c[2][1] = 0.0;
			break;
		case SE:
			c[0][0] = 1.0; c[0][1] = 0.0; c[1][0] = 0.5; c[1][1] = 0.5; c[2][0] = 0.5; c[2][1] = 0.0;
			break;
		case NE:
			c[0][0] = 0.5; c[0][1] = 0.0; c[1][0] = 0.5; c[1][1] = 0.5; c[2][0] = 0.0; c[2][1] = 0.5;
			break;
		case NW:
			c[0][0] = 0.5; c[0][1] = 0.5; c[1][0] = 0.0; c[1][1] = 1.0; c[2][0] = 0.0; c[2][1] = 0.5;
			break;
		default:
			throw_runtime_error("Unexpected triangle son_type in get_nodal_s_in_father");
		}
		sfather.resize(2, 0.0);
		if (l < 3)
		{
			sfather[0] = c[l][0];
			sfather[1] = c[l][1];
		}
		else if (n_node == 4) // C1TB: node 3 is the centroid bubble
		{
			for (unsigned i = 0; i < 2; i++) sfather[i] = (c[0][i] + c[1][i] + c[2][i]) / 3.0;
		}
		else // C2 / C2TB: nodes 3-5 mid-edges, node 6 centroid bubble
		{
			if (l == 3)
				for (unsigned i = 0; i < 2; i++) sfather[i] = 0.5 * (c[0][i] + c[1][i]);
			else if (l == 4)
				for (unsigned i = 0; i < 2; i++) sfather[i] = 0.5 * (c[1][i] + c[2][i]);
			else if (l == 5)
				for (unsigned i = 0; i < 2; i++) sfather[i] = 0.5 * (c[2][i] + c[0][i]);
			else // l == 6 (C2TB bubble)
				for (unsigned i = 0; i < 2; i++) sfather[i] = (c[0][i] + c[1][i] + c[2][i]) / 3.0;
		}
	}

	void BulkElementTri2dC1::get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather)
	{
		tri2d_nodal_s_in_father(Tree_pt->son_type(), this->nnode(), l, sfather);
	}

	void BulkElementTri2dC2::get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather)
	{
		tri2d_nodal_s_in_father(Tree_pt->son_type(), this->nnode(), l, sfather);
	}

	// For a C2 (biquadratic, 9-node) node index n that is not itself a C1 corner node (i.e. an
	// edge-midside node 1/3/5/7 or the center node 4), returns the C1 corner nodes that
	// geometrically "support" it (the corners of the edge or the whole element it sits at the
	// midpoint of); used to interpolate/constrain C1-only data at those positions. Corner nodes
	// (0,2,6,8) are not "supported" by others and yield an empty vector.
	void BulkElementQuad2dC2::get_supporting_C1_nodes_of_C2_node(const unsigned &n, std::vector<oomph::Node *> &support)
	{
		if (n == 4)
			support = {this->node_pt(0), this->node_pt(2), this->node_pt(6), this->node_pt(8)};
		else if (n == 1)
			support = {this->node_pt(0), this->node_pt(2)};
		else if (n == 3)
			support = {this->node_pt(0), this->node_pt(6)};
		else if (n == 5)
			support = {this->node_pt(2), this->node_pt(8)};
		else if (n == 7)
			support = {this->node_pt(6), this->node_pt(8)};
		else
			support.clear();
	}

	// If C1/C1TB fields are present alongside C2/C2TB ones, their nodal values (stored at the
	// same corner nodes, but at higher value indices, right after the C2/C2TB fields) need their
	// own hanging-node constraints set up too, since a corner node may hang for the C1
	// representation even where the C2 representation does not (or uses different masters).
	void BulkElementQuad2dC2::further_setup_hanging_nodes()
	{
		BulkElementBase::further_setup_hanging_nodes();
		auto *ft = jitcode->get_func_table();
		const int c2_hang = ft->continuous_spaces[SPACE_INDEX_C2].hangindex;
		const bool has_c1 = ft->continuous_spaces[SPACE_INDEX_C1].numfields_basebulk || ft->continuous_spaces[SPACE_INDEX_C1TB].numfields_basebulk;
		if (has_c1 || c2_hang >= 0)
		{
			const unsigned nC2TB = ft->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk;
			const unsigned nC2 = nC2TB + ft->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk;
			// C2 has its OWN (non-geometric) hang slot when the dominant space is C2TB (its enriched -1
			// trace differs from plain C2 -- on tet faces). Its values [nC2TB,nC2) then need the plain hang
			// on their own slot; otherwise only the C1(TB) values [nC2,ncont) hang separately.
			const unsigned start = (c2_hang >= 0) ? nC2TB : nC2;
			for (unsigned i = start; i < ncont_interpolated_values(); i++)
			{
				this->setup_hang_for_value(i);
			}
		}
	}




	// oomph-lib's generic "two co-located spaces" (here C2 vs C1) hook: for value indices
	// belonging to the C1/C1TB fields, the interpolating node is one of the 4 corner nodes
	// (mapped from the flattened index n via get_nodal_space_index_to_element_index_map); for
	// C2/C2TB fields, node n itself is the interpolating node.
	oomph::Node *BulkElementQuad2dC2::interpolating_node_pt(const unsigned &n, const int &value_id)
	{
		if (value_id >= static_cast<int>(jitcode->get_func_table()->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk + jitcode->get_func_table()->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk))
		{
			return this->node_pt(this->get_nodal_space_index_to_element_index_map()[SPACE_INDEX_C1][n]);
		}
		else
		{
			return this->node_pt(n);
		}
	}

	// Companion to interpolating_node_pt(): the 1-d local-coordinate fraction (0 or 1) of the
	// n1d-th interpolating node along direction i, for the C1 space (corner nodes only sit at the
	// element edges); for the C2 space, delegates to the standard quadratic 1-d fraction.
	double BulkElementQuad2dC2::local_one_d_fraction_of_interpolating_node(const unsigned &n1d, const unsigned &i, const int &value_id)
	{
		if (value_id >= static_cast<int>(jitcode->get_func_table()->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk + jitcode->get_func_table()->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk))
		{
			// The C1 nodes are just located on the boundaries at 0 or 1
			return double(n1d);
		}
		else
		{
			return this->local_one_d_fraction_of_node(n1d, i);
		}
	}

	// Companion to interpolating_node_pt(): finds the C1 corner node located exactly at local
	// coordinate s (returns NULL if s does not coincide with a corner, within tolerance); for
	// C2/C2TB fields, any of the 9 geometric nodes may coincide, so delegates to the generic
	// get_node_at_local_coordinate().
	oomph::Node *BulkElementQuad2dC2::get_interpolating_node_at_local_coordinate(const oomph::Vector<double> &s, const int &value_id)
	{
		if (value_id >= static_cast<int>(jitcode->get_func_table()->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk + jitcode->get_func_table()->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk))
		{
			unsigned total_index = 0;
			unsigned NNODE_1D = 2;
			oomph::Vector<int> index(this->dim());
			for (unsigned i = 0; i < this->dim(); i++)
			{
				if (s[i] == -1.0)
				{
					index[i] = 0;
				}
				else if (s[i] == 1.0)
				{
					index[i] = NNODE_1D - 1;
				}
				else
				{
					double float_index = 0.5 * (1.0 + s[i]) * (NNODE_1D - 1);
					index[i] = int(float_index);
					double excess = float_index - index[i];
					if ((excess > FiniteElement::Node_location_tolerance) && ((1.0 - excess) > FiniteElement::Node_location_tolerance))
					{
						return 0;
					}
				}
				total_index += index[i] * static_cast<unsigned>(pow(static_cast<float>(NNODE_1D), static_cast<int>(i)));
			}
			// If we've got here we have a node, so let's return a pointer to it
			return this->node_pt(this->get_nodal_space_index_to_element_index_map()[SPACE_INDEX_C1][total_index]);			
		}
		// Otherwise velocity nodes are the same as pressure nodes
		else
		{
			return this->get_node_at_local_coordinate(s);
		}
	}

	/// \short The number of 1d pressure nodes is 2, the number of 1d velocity
	/// nodes is the same as the number of 1d geometric nodes.
	unsigned BulkElementQuad2dC2::ninterpolating_node_1d(const int &value_id)
	{
		if (value_id >= static_cast<int>(jitcode->get_func_table()->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk + jitcode->get_func_table()->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk))
		{
			return 2;
		}
		else
		{
			return this->nnode_1d();
		}
	}

	/// \short The number of pressure nodes is 2^DIM. The number of
	/// velocity nodes is the same as the number of geometric nodes.
	unsigned BulkElementQuad2dC2::ninterpolating_node(const int &value_id)
	{
		if (value_id >= static_cast<int>(jitcode->get_func_table()->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk + jitcode->get_func_table()->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk))
		{
			return 4;
		}
		else
		{
			return this->nnode();
		}
	}

	void BulkElementQuad2dC2::shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		psi[2] = s[1];
	}

	void BulkElementQuad2dC2::dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		psi[2] = s[1];
		dpsi(0, 0) = 0.0;
		dpsi(1, 0) = 1.0;
		dpsi(2, 0) = 0.0;
		dpsi(0, 1) = 0.0;
		dpsi(1, 1) = 0.0;
		dpsi(2, 1) = 1.0;
	}

	void BulkElementQuad2dC2::shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		double psi1[2], psi2[2];
		oomph::OneDimLagrange::shape<2>(s[0], psi1);
		oomph::OneDimLagrange::shape<2>(s[1], psi2);
		for (unsigned i = 0; i < 2; i++)
		{
			for (unsigned j = 0; j < 2; j++)
			{
				psi[2 * i + j] = psi2[i] * psi1[j];
			}
		}
	}

	void BulkElementQuad2dC2::dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		double psi1[2], psi2[2];
		double dpsi1[2], dpsi2[2];
		oomph::OneDimLagrange::shape<2>(s[0], psi1);
		oomph::OneDimLagrange::shape<2>(s[1], psi2);
		oomph::OneDimLagrange::dshape<2>(s[0], dpsi1);
		oomph::OneDimLagrange::dshape<2>(s[1], dpsi2);
		for (unsigned i = 0; i < 2; i++)
		{
			for (unsigned j = 0; j < 2; j++)
			{
				psi[2 * i + j] = psi2[i] * psi1[j];
				dpsi(2 * i + j, 0) = psi2[i] * dpsi1[j];
				dpsi(2 * i + j, 1) = dpsi2[i] * psi1[j];
			}
		}
	}

	// Evaluates the shape functions of whichever space (C1 or C2/geometric) `value_id` belongs to,
	// completing the "two co-located spaces" interpolation interface used by oomph-lib.
	void BulkElementQuad2dC2::interpolating_basis(const oomph::Vector<double> &s, oomph::Shape &psi, const int &value_id) const
	{
		if (value_id >= static_cast<int>(jitcode->get_func_table()->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk + jitcode->get_func_table()->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk))
		{
			return this->shape_at_s_C1(s, psi);
		}
		else
		{
			return this->shape(s, psi);
		}
	}

	// Same purpose as BulkElementQuad2dC1::add_node_from_finer_neighbor_for_tesselated_numpy() above.
	void BulkElementQuad2dC2::add_node_from_finer_neighbor_for_tesselated_numpy(const oomph::Vector<double> &s_coarse, oomph::Node *n, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes)
	{
		// edgedir 0=S(0,2) 1=N(6,8) 2=W(0,6) 3=E(2,8), matching the `circum_data` edge indices in the fill.
		this->tess_register_hanging_node(s_coarse, n, {{0, 2}, {6, 8}, {0, 6}, {2, 8}}, add_nodes);
	}

	// Same purpose as BulkElementQuad2dC1::inform_coarser_neighbors_for_tesselated_numpy() above; the shared
	// quad_register_on_coarser_for_numpy walks all 9 nodes so the mid-side nodes are registered too.
	void BulkElementQuad2dC2::inform_coarser_neighbors_for_tesselated_numpy(std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes)
	{
		if (this->as_interface_element())
			throw_runtime_error("Cannot yet tesselate interface meshes [will fail in connecting hanging nodes and have to go via the parent mesh");
		this->quad_register_on_coarser_for_numpy(add_nodes);
	}

	// Same purpose as BulkElementQuad2dC1::get_num_numpy_elemental_indices() above; the base
	// (no-hanging-neighbor) triangle fan of a 9-node biquadratic quad needs 8 triangles (built
	// around the center node) instead of 2.
	int BulkElementQuad2dC2::get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
	{
		if (tesselate_tri)
		{
			// Check my neighbors whether they are on a finer level. If so, we need to add more triangles here
			if (add_nodes[this->_numpy_index].empty())
			{
				nsubdiv = 8;
			}
			else
			{
				unsigned tricnt = 0;
				for (unsigned int dir = 0; dir < 4; dir++)
				{
					tricnt += add_nodes[this->_numpy_index][dir].size();
				}
				nsubdiv = 8 + tricnt;
			}
			return 3;
		}
		else
		{
			nsubdiv = 1;
			return 9;
		}
	}
	// Same purpose as BulkElementQuad2dC1::fill_element_nodal_indices_for_numpy() above, adapted
	// to the 9-node biquadratic quad (base case: 8 triangles fanned around the center node 4).
	void BulkElementQuad2dC2::fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
	{
		if (tesselate_tri)
		{
			if (add_nodes[this->_numpy_index].empty())
			{
				indices[2] = 4;
				if (isubelem == 0)
				{
					indices[0] = 0;
					indices[1] = 1;
				}
				else if (isubelem == 1)
				{
					indices[0] = 1;
					indices[1] = 2;
				}
				else if (isubelem == 2)
				{
					indices[0] = 2;
					indices[1] = 5;
				}
				else if (isubelem == 3)
				{
					indices[0] = 5;
					indices[1] = 8;
				}
				else if (isubelem == 4)
				{
					indices[0] = 8;
					indices[1] = 7;
				}
				else if (isubelem == 5)
				{
					indices[0] = 7;
					indices[1] = 6;
				}
				else if (isubelem == 6)
				{
					indices[0] = 6;
					indices[1] = 3;
				}
				else
				{
					indices[0] = 3;
					indices[1] = 0;
				}
			}
			else
			{
				indices[2] = 4;
				// Now add all nodes along the south direction
				std::map<oomph::Node *, int> add_indices;
				for (unsigned int i = 0; i < this->nnode(); i++)
					add_indices[this->node_pt(i)] = i;
				int cnt = this->nnode();
				for (unsigned int d = 0; d < 4; d++)
				{
					for (auto *n : add_nodes[this->_numpy_index][d])
					{
						add_indices[n] = cnt++;
					}
				}

				// Now create a sorted node list -> 0,1,2,3,..8, but with the additional nodal information
				std::vector<int> circular_nodemap;

				std::vector<std::vector<int>> circum_data = {{0, 0, 1}, {2, 3, 5}, {8, 1, 7}, {6, 2, 3}}; // Data storing corner start node, direction node and L2-only node along the corner

				for (auto &side : circum_data)
				{
					int start_corner = side[0];
					int edgeindex = side[1];
					int L2node_along = side[2];
					circular_nodemap.push_back(start_corner); // Start at node 0
					// Order the edge's nodes by their LOCAL-coordinate distance from the start corner (own nodes
					// from local_coordinate_of_node, hanging nodes from _tess_hang_scoord -- topological, exact).
					oomph::Vector<double> sc(2);
					this->local_coordinate_of_node(start_corner, sc);
					std::map<double, oomph::Node *> sorted;
					for (auto *n : add_nodes[this->_numpy_index][edgeindex])
					{
						auto it = this->_tess_hang_scoord.find(n);
						oomph::Vector<double> sn = (it != this->_tess_hang_scoord.end()) ? it->second : oomph::Vector<double>(2, 0.0);
						double dist = (sn[0] - sc[0]) * (sn[0] - sc[0]) + (sn[1] - sc[1]) * (sn[1] - sc[1]);
						sorted[dist] = n;
					}
					oomph::Vector<double> sL(2);
					this->local_coordinate_of_node(L2node_along, sL);
					double dist = (sL[0] - sc[0]) * (sL[0] - sc[0]) + (sL[1] - sc[1]) * (sL[1] - sc[1]);
					sorted[dist] = this->node_pt(L2node_along);
					for (auto &entry : sorted)
						circular_nodemap.push_back(add_indices[entry.second]);
				}
				indices[0] = circular_nodemap[isubelem];
				indices[1] = circular_nodemap[(isubelem + 1) % circular_nodemap.size()];
			}
		}
		else
		{
			indices[0] = 0;
			indices[1] = 1;
			indices[2] = 2;
			indices[3] = 3;
			indices[4] = 4;
			indices[5] = 5;
			indices[6] = 6;
			indices[7] = 7;
			indices[8] = 8;
		}
	}

	std::vector<double> BulkElementQuad2dC2::get_outline(bool lagrangian)
	{
		std::vector<double> res(8 * this->nodal_dimension());
		unsigned offs = 0;
		for (unsigned int i = 0; i < this->nodal_dimension(); i++)
		{
		   if (lagrangian)
		   {
			res[0 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(0))->xi(i);
			res[1 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(1))->xi(i);
			res[2 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(2))->xi(i);
			res[3 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(5))->xi(i);
			res[4 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(8))->xi(i);
			res[5 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(7))->xi(i);
			res[6 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(6))->xi(i);
			res[7 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(3))->xi(i);
			}		   
		   else
		   {
			res[0 + offs] = this->node_pt(0)->x(i);
			res[1 + offs] = this->node_pt(1)->x(i);
			res[2 + offs] = this->node_pt(2)->x(i);
			res[3 + offs] = this->node_pt(5)->x(i);
			res[4 + offs] = this->node_pt(8)->x(i);
			res[5 + offs] = this->node_pt(7)->x(i);
			res[6 + offs] = this->node_pt(6)->x(i);
			res[7 + offs] = this->node_pt(3)->x(i);
			}
			offs += 8;
		}
		return res;
	}

	oomph::Node *BulkElementQuad2dC2::boundary_node_pt(const int &face_index, const unsigned int i)
	{
		const unsigned nn1d = 3;
		if (face_index == -1)
		{
			return this->node_pt(i * nn1d);
		}
		else if (face_index == +1)
		{
			return this->node_pt(nn1d * i + nn1d - 1);
		}
		else if (face_index == -2)
		{
			return this->node_pt(i);
		}
		else if (face_index == +2)
		{
			return this->node_pt(nn1d * (nn1d - 1) + i);
		}
		else
		{
			std::string err = "Face index should be in {-1, +1, -2, +2}.";
			throw oomph::OomphLibError(err, OOMPH_EXCEPTION_LOCATION, OOMPH_CURRENT_FUNCTION);
		}
	}

	//////////////////////////////

	// BulkElementTri2dC1: linear (3-node) triangle, optionally bubble-enriched with a 4th
	// (centroid) node for the C1TB space (used e.g. for MINI-element-type stabilization); C1 and
	// DL always use only the 3 corner nodes / a constant+linear representation.
	BulkElementTri2dC1::BulkElementTri2dC1(bool has_bubble)
	{
		eleminfo.elem_ptr = static_cast<BulkElementBase *>(this);
		eleminfo.nnode = (has_bubble ? 4 : 3);
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = (has_bubble ? 4 : 3);
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 3;
		eleminfo.nnode_DL = 3;
		eleminfo.nodal_dim = jitcode->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}

	oomph::Node *BulkElementTri2dC1::boundary_node_pt(const int &face_index, const unsigned int i)
	{
		return this->node_pt(this->get_bulk_node_number(face_index, i));
	}



	void BulkElementTri2dC1::shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		psi[2] = s[1];
	}

	void BulkElementTri2dC1::dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		psi[2] = s[1];
		dpsi(0, 0) = 0.0;
		dpsi(1, 0) = 1.0;
		dpsi(2, 0) = 0.0;
		dpsi(0, 1) = 0.0;
		dpsi(1, 1) = 0.0;
		dpsi(2, 1) = 1.0;
	}

	// Tri hanging-node tesselation (see the quad counterparts). The three tri edges as corner-node pairs;
	// shared by C1/C1TB (and, with more nodes per edge, C2/C2TB via the same edge corners).
	static const std::vector<std::pair<unsigned, unsigned>> Tri_edge_corner_pairs = {{0, 1}, {1, 2}, {2, 0}};

	void BulkElementTri2dC1::add_node_from_finer_neighbor_for_tesselated_numpy(const oomph::Vector<double> &s_coarse, oomph::Node *n, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes)
	{
		this->tess_register_hanging_node(s_coarse, n, Tri_edge_corner_pairs, add_nodes);
	}

	void BulkElementTri2dC1::inform_coarser_neighbors_for_tesselated_numpy(std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes)
	{
		this->tess_inform_coarser_tri(Tri_edge_corner_pairs, add_nodes);
	}

	int BulkElementTri2dC1::get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
	{
		if (tesselate_tri)
		{
			std::vector<unsigned> tris;
			this->tess_hanging_delaunay(Tri_edge_corner_pairs, add_nodes, tris);
			nsubdiv = tris.size() / 3;
			return 3;
		}
		nsubdiv = 1;
		return 3;
	}

	void BulkElementTri2dC1::fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
	{
		if (tesselate_tri)
		{
			std::vector<unsigned> tris;
			this->tess_hanging_delaunay(Tri_edge_corner_pairs, add_nodes, tris);
			indices[0] = tris[3 * isubelem];
			indices[2] = tris[3 * isubelem + 1];
			indices[1] = tris[3 * isubelem + 2];
			return;
		}
		indices[0] = 0;
		indices[1] = 1;
		indices[2] = 2;
	}

	std::vector<double> BulkElementTri2dC1::get_outline(bool lagrangian)
	{
		std::vector<double> res(3 * this->nodal_dimension());
		unsigned offs = 0;
		for (unsigned int i = 0; i < this->nodal_dimension(); i++)
		{
			if (lagrangian)
			{
			res[0 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(0))->xi(i);
			res[1 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(1))->xi(i);
			res[2 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(2))->xi(i);			
			}
			else
			{
			res[0 + offs] = this->node_pt(0)->x(i);
			res[1 + offs] = this->node_pt(1)->x(i);
			res[2 + offs] = this->node_pt(2)->x(i);
			}
			offs += 3;
		}
		return res;
	}

	

	//////////////////////////////

	// BulkElementTri2dC2: quadratic (6-node) triangle, optionally bubble-enriched with a 7th
	// (centroid) node for C2TB. C2/C2TB use all 6 (or 7) nodes; C1/C1TB use only the 3 (or 4)
	// corner/bubble nodes; DL is a constant+linear discontinuous representation.
	BulkElementTri2dC2::BulkElementTri2dC2(bool with_bubble)
	{
		eleminfo.elem_ptr = static_cast<BulkElementBase *>(this);
		eleminfo.nnode = 6;
		eleminfo.nnode_of_space[SPACE_INDEX_C2TB] = (with_bubble ? 7:6); // Must be done here! DG field allocation would otherwise alloc only 6 for D2TB!
		eleminfo.nnode_of_space[SPACE_INDEX_C2] = 6;
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = (with_bubble ? 4 : 3);		
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 3;
		eleminfo.nnode_DL = 3;
		eleminfo.nodal_dim = jitcode->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}

    // Creates a son of the correct concrete type (bubble-enriched BulkElementTri2dC2TB, or plain
    // BulkElementTri2dC2) matching this element, for dynamic_split()/mesh refinement.
    BulkElementBase * BulkElementTri2dC2::create_son_instance() const
	    {
      BulkElementBase::JITCodeScope __jit_scope1(jitcode);
      // A C2TB (bubble-enriched) father must spawn a genuine BulkElementTri2dC2TB son: the bubble
      // needs a real 7th (centroid) node slot and the 7-node nodal-space map. A plain
      // BulkElementTri2dC2(true) only bumps nnode_of_space[C2TB] to 7 while keeping 6 oomph node
      // slots and the 6-node map -> fill_element_info reads past the map end and segfaults.
      BulkElementTri2dC2 *res;
      if (dynamic_cast<const BulkElementTri2dC2TB*>(this) != nullptr) res = new BulkElementTri2dC2TB();
      else res = new BulkElementTri2dC2(false);
      res->jitcode = jitcode;
      return res;
    }


	void BulkElementTri2dC2::get_supporting_C1_nodes_of_C2_node(const unsigned &n, std::vector<oomph::Node *> &support)
	{
		if (n == 3)
			support = {this->node_pt(0), this->node_pt(1)};
		else if (n == 4)
			support = {this->node_pt(1), this->node_pt(2)};
		else if (n == 5)
			support = {this->node_pt(2), this->node_pt(0)};
		else
			support.clear();
	}

	

	oomph::Node *BulkElementTri2dC2::boundary_node_pt(const int &face_index, const unsigned int i)
	{
		return this->node_pt(this->get_bulk_node_number(face_index, i));
	}

   
	void BulkElementTri2dC2::shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		psi[0] = s[0];
		psi[1] = s[1];
		psi[2] = 1.0 - s[0] - s[1];
	}

	// --- oomph "interpolating node" facilities for mixed-order (C1-on-C2) hanging on triangles ---
	// A value_id belongs to a C1/C1TB field iff it is >= the number of C2/C2TB base-bulk fields.
	namespace
	{
		inline bool tri_value_is_C1(DynamicJITCode *jitcode, const int &value_id)
		{
			auto *ft = jitcode->get_func_table();
			return value_id >= static_cast<int>(ft->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk + ft->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk);
		}
	}

	// For a C1 field the interpolating nodes are the 3 corner vertices; for a C2 field they are the
	// element's geometric nodes. Set up the hanging scheme for every C1 value id (mirrors the quad).
	void BulkElementTri2dC2::further_setup_hanging_nodes()
	{
		BulkElementBase::further_setup_hanging_nodes();
		auto *ft = jitcode->get_func_table();
		const int c2_hang = ft->continuous_spaces[SPACE_INDEX_C2].hangindex;
		const bool has_c1 = ft->continuous_spaces[SPACE_INDEX_C1].numfields_basebulk || ft->continuous_spaces[SPACE_INDEX_C1TB].numfields_basebulk;
		if (has_c1 || c2_hang >= 0)
		{
			const unsigned nC2TB = ft->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk;
			const unsigned nC2 = nC2TB + ft->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk;
			// C2 has its OWN hang slot when the dominant space is C2TB (see BulkElementQuad2dC2). In 2D the
			// C2TB bubble is interior (vanishes on edges), so this slot is just a mirror of -1, but it keeps
			// the value-slot layout consistent with 3d where it genuinely differs on tet faces.
			const unsigned start = (c2_hang >= 0) ? nC2TB : nC2;
			for (unsigned i = start; i < ncont_interpolated_values(); i++)
			{
				this->setup_hang_for_value(i);
			}
			// Hang each C2 mid-edge node's separate C1(TB) value slot linearly on its two edge-corner nodes.
			// This is the C1-on-C2 rule (a mid-edge pressure IS the mean of its edge corners) and, crucially,
			// it constrains the STALE pressure slot oomph leaves behind on a former fine corner once its
			// region is UNREFINED and the node becomes a plain conforming C2 mid-edge: setup_hang_for_value
			// only hangs nodes across a coarser (2:1) neighbour, so without this the stale slot is a free,
			// unassembled dof (wrong Jacobian / a "degrade to C1 on a vertex" throw). A node carries the slot
			// only if it was ever a corner, so ordinary mid-edge nodes (which never store the pressure) are
			// skipped by the nvalue guard; genuine 2:1 mid-nodes were already hung above and are skipped via
			// is_hanging. Weights 0.5/0.5 are exact for the straight edge that a C2 mid-edge bisects.
			if (this->nnode() >= 6)
			{
				std::vector<int> c1_slots;
				for (int sp : {SPACE_INDEX_C1TB, SPACE_INDEX_C1})
				{ int h = ft->continuous_spaces[sp].hangindex; if (h >= 0) c1_slots.push_back(h); }
				static const unsigned edge_ends[3][2] = {{0, 1}, {1, 2}, {2, 0}}; // C2 mid nodes 3,4,5 bisect these corner pairs
				for (unsigned e = 0; e < 3; e++)
				{
					oomph::Node *M = this->node_pt(3 + e);
					oomph::Node *A = this->node_pt(edge_ends[e][0]);
					oomph::Node *B = this->node_pt(edge_ends[e][1]);
					for (int slot : c1_slots)
					{
						if ((int)M->nvalue() <= slot || M->is_hanging(slot)) continue;        // no stale slot, or already hung (2:1)
						if ((int)A->nvalue() <= slot || (int)B->nvalue() <= slot) continue;    // masters must carry the dof
						oomph::HangInfo *hang = new oomph::HangInfo(2);
						hang->set_master_node_pt(0, A, 0.5);
						hang->set_master_node_pt(1, B, 0.5);
						M->set_hanging_pt(hang, slot);
					}
				}
			}
		}
	}

	oomph::Node *BulkElementTri2dC2::interpolating_node_pt(const unsigned &n, const int &value_id)
	{
		if (tri_value_is_C1(jitcode, value_id))
			return this->node_pt(this->get_nodal_space_index_to_element_index_map()[SPACE_INDEX_C1][n]);
		return this->node_pt(n);
	}

	// Triangles are not a tensor product, so there is no genuine 1d fraction; the tri hang helper does
	// not use this (it works off the barycentric edge parametrisation). Kept for interface compatibility:
	// C1 corner nodes sit at the ends (0 or 1); C2 delegates to the standard node fraction.
	double BulkElementTri2dC2::local_one_d_fraction_of_interpolating_node(const unsigned &n1d, const unsigned &i, const int &value_id)
	{
		if (tri_value_is_C1(jitcode, value_id))
			return double(n1d);
		return this->local_one_d_fraction_of_node(n1d, i);
	}

	oomph::Node *BulkElementTri2dC2::get_interpolating_node_at_local_coordinate(const oomph::Vector<double> &s, const int &value_id)
	{
		if (tri_value_is_C1(jitcode, value_id))
		{
			// C1 corners: node 0 at s=(1,0), node 1 at s=(0,1), node 2 at s=(0,0) (cf. shape_at_s_C1).
			const double tol = oomph::FiniteElement::Node_location_tolerance;
			const std::vector<unsigned> &c1map = this->get_nodal_space_index_to_element_index_map()[SPACE_INDEX_C1];
			if (std::abs(s[0] - 1.0) < tol && std::abs(s[1]) < tol) return this->node_pt(c1map[0]);
			if (std::abs(s[0]) < tol && std::abs(s[1] - 1.0) < tol) return this->node_pt(c1map[1]);
			if (std::abs(s[0]) < tol && std::abs(s[1]) < tol) return this->node_pt(c1map[2]);
			return 0;
		}
		return this->get_node_at_local_coordinate(s);
	}

	unsigned BulkElementTri2dC2::ninterpolating_node_1d(const int &value_id)
	{
		if (tri_value_is_C1(jitcode, value_id)) return 2;
		return this->nnode_1d();
	}

	unsigned BulkElementTri2dC2::ninterpolating_node(const int &value_id)
	{
		if (tri_value_is_C1(jitcode, value_id)) return 3;
		return this->nnode();
	}

	void BulkElementTri2dC2::interpolating_basis(const oomph::Vector<double> &s, oomph::Shape &psi, const int &value_id) const
	{
		if (tri_value_is_C1(jitcode, value_id))
			return this->shape_at_s_C1(s, psi);
		return this->shape(s, psi);
	}

	void BulkElementTri2dC2::dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = s[0];
		psi[1] = s[1];
		psi[2] = 1.0 - s[0] - s[1];
		dpsi(0, 0) = 1.0;
		dpsi(0, 1) = 0.0;
		dpsi(1, 0) = 0.0;
		dpsi(1, 1) = 1.0;
		dpsi(2, 0) = -1.0;
		dpsi(2, 1) = -1.0;
		/*
		 double s_2=1.0-s[0]-s[1];
		 psi[0] = 2.0*s[0]*(s[0]-0.5);
		 psi[1] = 2.0*s[1]*(s[1]-0.5);
		 psi[2] = 2.0*s_2 *(s_2 -0.5);
		 psi[3] = 4.0*s[0]*s[1];
		 psi[4] = 4.0*s[1]*s_2;
		 psi[5] = 4.0*s_2*s[0];

		 dpsi(0,0) = 4.0*s[0]-1.0;
		 dpsi(0,1) = 0.0;
		 dpsi(1,0) = 0.0;
		 dpsi(1,1) = 4.0*s[1]-1.0;
		 dpsi(2,0) = 2.0*(2.0*s[0]-1.5+2.0*s[1]);
		 dpsi(2,1) = 2.0*(2.0*s[0]-1.5+2.0*s[1]);
		 dpsi(3,0) = 4.0*s[1];
		 dpsi(3,1) = 4.0*s[0];
		 dpsi(4,0) = -4.0*s[1];
		 dpsi(4,1) = 4.0*(1.0-s[0]-2.0*s[1]);
		 dpsi(5,0) = 4.0*(1.0-2.0*s[0]-s[1]);
		 dpsi(5,1) = -4.0*s[0];*/
	}

	void BulkElementTri2dC2::shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		psi[2] = s[1];
	}

	void BulkElementTri2dC2::dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		psi[2] = s[1];
		dpsi(0, 0) = 0.0;
		dpsi(1, 0) = 1.0;
		dpsi(2, 0) = 0.0;
		dpsi(0, 1) = 0.0;
		dpsi(1, 1) = 0.0;
		dpsi(2, 1) = 1.0;
	}

	void BulkElementTri2dC2::add_node_from_finer_neighbor_for_tesselated_numpy(const oomph::Vector<double> &s_coarse, oomph::Node *n, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes)
	{
		this->tess_register_hanging_node(s_coarse, n, Tri_edge_corner_pairs, add_nodes);
	}

	void BulkElementTri2dC2::inform_coarser_neighbors_for_tesselated_numpy(std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes)
	{
		this->tess_inform_coarser_tri(Tri_edge_corner_pairs, add_nodes);
	}

	int BulkElementTri2dC2::get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
	{
		if (tesselate_tri)
		{
			std::vector<unsigned> tris;
			this->tess_hanging_delaunay(Tri_edge_corner_pairs, add_nodes, tris);
			nsubdiv = tris.size() / 3;
			return 3;
		}
		nsubdiv = 1;
		return 6;
	}

	void BulkElementTri2dC2::fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
	{
		if (tesselate_tri)
		{
			std::vector<unsigned> tris;
			this->tess_hanging_delaunay(Tri_edge_corner_pairs, add_nodes, tris);
			indices[0] = tris[3 * isubelem];
			indices[2] = tris[3 * isubelem + 1];
			indices[1] = tris[3 * isubelem + 2];
			return;
		}
		indices[0] = 0;
		indices[1] = 1;
		indices[2] = 2;
		indices[3] = 3;
		indices[4] = 4;
		indices[5] = 5;
	}

	std::vector<double> BulkElementTri2dC2::get_outline(bool lagrangian)
	{
		std::vector<double> res(6 * this->nodal_dimension());
		unsigned offs = 0;
		for (unsigned int i = 0; i < this->nodal_dimension(); i++)
		{
			if (lagrangian)
			{
			res[0 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(0))->xi(i);
			res[1 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(3))->xi(i);
			res[2 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(1))->xi(i);
			res[3 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(4))->xi(i);
			res[4 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(2))->xi(i);
			res[5 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(5))->xi(i);			
			}
			else
			{
			res[0 + offs] = this->node_pt(0)->x(i);
			res[1 + offs] = this->node_pt(3)->x(i);
			res[2 + offs] = this->node_pt(1)->x(i);
			res[3 + offs] = this->node_pt(4)->x(i);
			res[4 + offs] = this->node_pt(2)->x(i);
			res[5 + offs] = this->node_pt(5)->x(i);
			}			
			offs += 6;
		}
		return res;
	}

   //////////////////////////////
	
   // BulkElementTri2dC1TB: the actual bubble-enriched (MINI-element style) linear triangle, with
   // node 3 the centroid "bubble" node. Its own geometry/shape() overrides the base Q1-per-corner
   // shape functions with the cubic-bubble MINI-element basis (barycentric coordinate minus 9x
   // the cubic bubble x*y*z for the corners, and 27x the bubble for the enrichment function),
   // used e.g. for LBB-stable low-order Stokes discretizations.
   BulkElementTri2dC1TB::BulkElementTri2dC1TB()  : BulkElementTri2dC1(true)
   {
		eleminfo.elem_ptr = static_cast<BulkElementBase *>(this);   
      eleminfo.nnode=4;
      eleminfo.nnode_of_space[SPACE_INDEX_C1]=3;
      eleminfo.nnode_of_space[SPACE_INDEX_C1TB]=4;
      eleminfo.nnode_DL=3;
      eleminfo.nodal_dim = jitcode->get_func_table()->nodal_dim;
		this->set_n_node(eleminfo.nnode);
		this->set_nodal_dimension(eleminfo.nodal_dim);
		this->set_integration_scheme(&Default_enriched_integration_scheme);      
   }
   
   
   void BulkElementTri2dC1TB::shape(const oomph::Vector<double> &s, oomph::Shape &psi) const
   {
      const double x=s[0];
      const double y=s[1];
      const double z=1-x-y;
      const double bubble=x*y*z;
      psi[0] = x-9*bubble;
		psi[1] = y-9*bubble;
		psi[2] = z-9*bubble;
		psi[3]=27.0*bubble;
   }
   void BulkElementTri2dC1TB::dshape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids) const
   {
      const double x=s[0];
      const double y=s[1];
      const double z=1-x-y;
      const double bubble=x*y*z;
      psi[0] = x-9.0*bubble;
		psi[1] = y-9.0*bubble;
		psi[2] = z-9.0*bubble;
		psi[3]=27.0*bubble;
		const double dbubble_dx=y*(z-x);
		const double dbubble_dy=x*(z-y);		
  	   dpsids(0, 0) = 1.0-9.0*dbubble_dx;
		dpsids(0, 1) = -9.0*dbubble_dy;
		dpsids(1, 0) = -9.0*dbubble_dx;
		dpsids(1, 1) = 1.0-9.0*dbubble_dy;
		dpsids(2, 0) = -1.0-9.0*dbubble_dx;
		dpsids(2, 1) = -1.0-9.0*dbubble_dy;		
      dpsids(3, 0) = 27.0*y*(-2*x - y + 1);
		dpsids(3, 1) = 27*x*(-x - 2*y + 1);	
   }
    
    void BulkElementTri2dC1TB::shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const
    {
      psi[0] = s[0];
		psi[1] = s[1];
		psi[2] = 1.0 - s[0] - s[1];
    }
    
   void BulkElementTri2dC1TB::dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
   {
      psi[0] = s[0];
		psi[1] = s[1];
		psi[2] = 1.0 - s[0] - s[1];
		dpsi(0, 0) = 1.0;
		dpsi(0, 1) = 0.0;
		dpsi(1, 0) = 0.0;
		dpsi(1, 1) = 1.0;
		dpsi(2, 0) = -1.0;
		dpsi(2, 1) = -1.0;
   }
   
   // Local coordinates of the 3 corners plus the centroid (node 3, the bubble node).
   void BulkElementTri2dC1TB::local_coordinate_of_node(const unsigned &j, oomph::Vector<double> &s) const
   {
	s.resize(2);
    switch (j)
	{
		case 0:
			s[0] = 1.0;
			s[1] = 0.0;
			break;
		case 1:
			s[0] = 0.0;
			s[1] = 1.0;
			break;
		case 2:
			s[0] = 0.0;
			s[1] = 0.0;
			break;
		case 3:
			s[0] = 1.0 / 3.0;
			s[1] = 1.0 / 3.0;
			break;
		default:
			throw std::out_of_range("Invalid node index");
	}
   }
   
   // Inherits add_node_from_finer_neighbor_for_tesselated_numpy + inform_coarser_neighbors_for_tesselated_numpy
   // from BulkElementTri2dC1. The centroid (node 3) is an interior point, so the hanging-node Delaunay path
   // over all 4 nodes reproduces the 3-triangle centroid fan when there are no hanging nodes.
   int BulkElementTri2dC1TB::get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
   {
      if (tesselate_tri)
      {
         std::vector<unsigned> tris;
         this->tess_hanging_delaunay(Tri_edge_corner_pairs, add_nodes, tris);
         nsubdiv = tris.size() / 3;
         return 3;
      }
      nsubdiv = 1;
      return 4;
   }

   void BulkElementTri2dC1TB::fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
   {
      if (tesselate_tri)
		{
			std::vector<unsigned> tris;
			this->tess_hanging_delaunay(Tri_edge_corner_pairs, add_nodes, tris);
			indices[0] = tris[3 * isubelem];
			indices[2] = tris[3 * isubelem + 1];
			indices[1] = tris[3 * isubelem + 2];
			return;
		}
		else
		{
			indices[0] = 0;
			indices[1] = 1;
			indices[2] = 2;
			indices[3] = 3;
		}
   }
	///////////////////////////////

	// BulkElementTri2dC2TB: quadratic triangle with an additional cubic-bubble enrichment of the
	// C1TB space at the corners plus centroid (node 6) -- same MINI-style bubble as
	// BulkElementTri2dC1TB above, layered on top of the quadratic C2 geometry/fields.
	BulkElementTri2dC2TB::BulkElementTri2dC2TB() : BulkElementTri2dC2(true)
	{
		eleminfo.elem_ptr = static_cast<BulkElementBase *>(this);
		eleminfo.nnode = 7;
		eleminfo.nnode_of_space[SPACE_INDEX_C2TB] = 7;
		eleminfo.nnode_of_space[SPACE_INDEX_C2] = 6;
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = 4;		
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 3;
		eleminfo.nnode_DL = 3;
		eleminfo.nodal_dim = jitcode->get_func_table()->nodal_dim;
		this->set_n_node(eleminfo.nnode);
		this->set_nodal_dimension(eleminfo.nodal_dim);
		this->set_integration_scheme(&Default_enriched_integration_scheme);
	}

   void BulkElementTri2dC2TB::shape_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi) const 
   {
      const double x=s[0];
      const double y=s[1];
      const double z=1-x-y;
      const double bubble=x*y*z;
      psi[0] = x-9*bubble;
		psi[1] = y-9*bubble;
		psi[2] = z-9*bubble;
		psi[3]=27.0*bubble;

   }
   void BulkElementTri2dC2TB::dshape_local_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const 
   {
      const double x=s[0];
      const double y=s[1];
      const double z=1-x-y;
      const double bubble=x*y*z;
      psi[0] = x-9.0*bubble;
		psi[1] = y-9.0*bubble;
		psi[2] = z-9.0*bubble;
		psi[3]=27.0*bubble;
		const double dbubble_dx=y*(z-x);
		const double dbubble_dy=x*(z-y);		
  	   dpsi(0, 0) = 1.0-9.0*dbubble_dx;
		dpsi(0, 1) = -9.0*dbubble_dy;
		dpsi(1, 0) = -9.0*dbubble_dx;
		dpsi(1, 1) = 1.0-9.0*dbubble_dy;
		dpsi(2, 0) = -1.0-9.0*dbubble_dx;
		dpsi(2, 1) = -1.0-9.0*dbubble_dy;		
      dpsi(3, 0) = 27.0*y*(-2*x - y + 1);
		dpsi(3, 1) = 27*x*(-x - 2*y + 1);		
   }
    


	// Inherits add_node/inform_coarser from BulkElementTri2dC2. When hanging nodes are present the Delaunay
	// path triangulates all 7 nodes (6 boundary + centroid node 6), giving a finer split than the plain
	// 3-triangle centroid fan used in the no-hanging base case; both are valid coverings for plotting.
	int BulkElementTri2dC2TB::get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
	{
		if (tesselate_tri)
		{
			std::vector<unsigned> tris;
			this->tess_hanging_delaunay(Tri_edge_corner_pairs, add_nodes, tris);
			nsubdiv = tris.size() / 3;
			return 3;
		}
		nsubdiv = 1;
		return 7;
	}

	void BulkElementTri2dC2TB::fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
	{
		if (tesselate_tri)
		{
			std::vector<unsigned> tris;
			this->tess_hanging_delaunay(Tri_edge_corner_pairs, add_nodes, tris);
			indices[0] = tris[3 * isubelem];
			indices[2] = tris[3 * isubelem + 1];
			indices[1] = tris[3 * isubelem + 2];
			return;
		}
		indices[0] = 0;
		indices[1] = 1;
		indices[2] = 2;
		indices[3] = 3;
		indices[4] = 4;
		indices[5] = 5;
		indices[6] = 6;
	}

	// const unsigned BulkElementTri2dC2TB::Central_node_on_face[3] = {4,5,3};
	oomph::TBubbleEnrichedGauss<2, 3> BulkElementTri2dC1TB::Default_enriched_integration_scheme;
	oomph::TBubbleEnrichedGauss<2, 3> BulkElementTri2dC2TB::Default_enriched_integration_scheme;

	const std::vector<std::vector<unsigned>> BulkElementQuad2dC1::Nodal_Space_Index_To_Element_Index_Map={
		{}, // C2TB 
		{}, // C2
		{0,1,2,3}, // C1TB
		{0,1,2,3}  // C1
	};

	const std::vector<std::vector<unsigned>> BulkElementQuad2dC2::Nodal_Space_Index_To_Element_Index_Map={
		{0,1,2,3,4,5,6,7,8}, // C2TB 
		{0,1,2,3,4,5,6,7,8}, // C2
		{0,2,6,8}, // C1TB
		{0,2,6,8}  // C1
	};

	const std::vector<std::vector<unsigned>> BulkElementTri2dC1::Nodal_Space_Index_To_Element_Index_Map={
		{}, // C2TB 
		{}, // C2
		{0,1,2}, // C1TB
		{0,1,2}  // C1
	};

	const std::vector<std::vector<unsigned>> BulkElementTri2dC1TB::Nodal_Space_Index_To_Element_Index_Map={
		{}, // C2TB 
		{}, // C2
		{0,1,2,3}, // C1TB
		{0,1,2}  // C1
	};

	const std::vector<std::vector<unsigned>> BulkElementTri2dC2::Nodal_Space_Index_To_Element_Index_Map={
		{}, // C2TB 
		{0,1,2,3,4,5}, // C2
		{}, // C1TB
		{0,1,2}  // C1
	};

	const std::vector<std::vector<unsigned>> BulkElementTri2dC2TB::Nodal_Space_Index_To_Element_Index_Map={
		{0,1,2,3,4,5,6}, // C2TB 
		{0,1,2,3,4,5}, // C2
		{0,1,2,6}, // C1TB
		{0,1,2}  // C1
	};


	const std::vector<std::vector<int>> BulkElementQuad2dC1::Element_Index_To_Nodal_Space_Index_Map={
		{}, // C2TB
		{}, // C2
		{0,1,2,3}, // C1TB
		{0,1,2,3}  // C1
	};

	const std::vector<std::vector<int>> BulkElementQuad2dC2::Element_Index_To_Nodal_Space_Index_Map={
		{0,1,2,3,4,5,6,7,8}, // C2TB
		{0,1,2,3,4,5,6,7,8}, // C2
		{0,-1,1,-1,-1,-1,2,-1,3}, // C1TB
		{0,-1,1,-1,-1,-1,2,-1,3}  // C1
	};

	const std::vector<std::vector<int>> BulkElementTri2dC1::Element_Index_To_Nodal_Space_Index_Map={
		{}, // C2TB
		{}, // C2
		{0,1,2}, // C1TB
		{0,1,2}  // C1
	};

	const std::vector<std::vector<int>> BulkElementTri2dC1TB::Element_Index_To_Nodal_Space_Index_Map={
		{}, // C2TB
		{}, // C2
		{0,1,2,3}, // C1TB
		{0,1,2,-1}  // C1
	};

	const std::vector<std::vector<int>> BulkElementTri2dC2::Element_Index_To_Nodal_Space_Index_Map={
		{}, // C2TB
		{0,1,2,3,4,5}, // C2
		{}, // C1TB
		{0,1,2,-1,-1,-1}  // C1
	};

	const std::vector<std::vector<int>> BulkElementTri2dC2TB::Element_Index_To_Nodal_Space_Index_Map={
		{0,1,2,3,4,5,6}, // C2TB
		{0,1,2,3,4,5,-1}, // C2
		{0,1,2,-1,-1,-1,3}, // C1TB
		{0,1,2,-1,-1,-1,-1}  // C1
	};
	const std::vector<unsigned> BulkElementQuad2dC1::Non_Vertex_Node_Indices={};
	const std::vector<unsigned> BulkElementQuad2dC2::Non_Vertex_Node_Indices={1,3,4,5,7};
	const std::vector<unsigned> BulkElementTri2dC1::Non_Vertex_Node_Indices={};
	const std::vector<unsigned> BulkElementTri2dC1TB::Non_Vertex_Node_Indices={3};
	const std::vector<unsigned> BulkElementTri2dC2::Non_Vertex_Node_Indices={3,4,5};
	const std::vector<unsigned> BulkElementTri2dC2TB::Non_Vertex_Node_Indices={3,4,5,6};

	const std::vector<int> BulkElementQuad2dC1::Possible_Face_Indices={-2,-1,1,2};
	const std::vector<int> BulkElementQuad2dC2::Possible_Face_Indices={-2,-1,1,2};

	const std::vector<int> BulkElementTri2dC1::Possible_Face_Indices={0,1,2};
	const std::vector<int> BulkElementTri2dC2::Possible_Face_Indices={0,1,2};
	

	std::vector<pyoomph::Node*> BulkElementQuad2dC1::get_vertex_nodes_of_face(const int &face) const
	{	  
	  	if (face==-2) { return {static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(1))};}
 		else if (face==-1) { return {static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(2))};}
 		else if (face==1) { return {static_cast<pyoomph::Node*>(this->node_pt(1)),static_cast<pyoomph::Node*>(this->node_pt(3))};}
 		else if (face==2) { return {static_cast<pyoomph::Node*>(this->node_pt(2)),static_cast<pyoomph::Node*>(this->node_pt(3))};}
		else throw_runtime_error("Invalid face index for quadrilateral element");
	}

	std::vector<pyoomph::Node*> BulkElementQuad2dC2::get_vertex_nodes_of_face(const int &face) const
	{	  	  
	  if (face==-2) { return {static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(2))};}
      else if (face==-1) { return {static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(6))};}
      else if (face==1) { return {static_cast<pyoomph::Node*>(this->node_pt(2)),static_cast<pyoomph::Node*>(this->node_pt(8))};}
      else if (face==2) { return {static_cast<pyoomph::Node*>(this->node_pt(6)),static_cast<pyoomph::Node*>(this->node_pt(8))};}
	  else throw_runtime_error("Invalid face index for quadrilateral element");
	}

	std::vector<pyoomph::Node*> BulkElementTri2dC1::get_vertex_nodes_of_face(const int &face) const
	{	  
	  if (face==0) { return {static_cast<pyoomph::Node*>(this->node_pt(2)),static_cast<pyoomph::Node*>(this->node_pt(1))};}
      else if (face==1) { return {static_cast<pyoomph::Node*>(this->node_pt(2)),static_cast<pyoomph::Node*>(this->node_pt(0))};}
      else if (face==2) { return {static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(1))};}
	  else throw_runtime_error("Invalid face index for triangular element");
	}

	std::vector<pyoomph::Node*> BulkElementTri2dC2::get_vertex_nodes_of_face(const int &face) const
	{	  
	  if (face==0) { return {static_cast<pyoomph::Node*>(this->node_pt(2)),static_cast<pyoomph::Node*>(this->node_pt(1))};}
      else if (face==1) { return {static_cast<pyoomph::Node*>(this->node_pt(2)),static_cast<pyoomph::Node*>(this->node_pt(0))};}
      else if (face==2) { return {static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(1))};}
	  else throw_runtime_error("Invalid face index for triangular element");	  
	}

	//oomph::FaceElement * construct_face_element(DynamicJITCode *interface_jitcode, int face_index) override;

	// construct_face_element() implementations: create the appropriate InterfaceElement<...> FaceElement type
	// for a given bulk element and face index (e.g. a quad's face is a line, a tet's face is a triangle);
	// most bulk types have a single, fixed face-element type, but pyramids/wedges mix triangular and
	// quadrilateral faces and therefore dispatch on face_index below.
	oomph::FaceElement * BulkElementQuad2dC2::construct_face_element(DynamicJITCode *interface_jitcode, int face_index) { return new InterfaceElementLine1dC2(interface_jitcode, this, face_index); }
	oomph::FaceElement * BulkElementQuad2dC1::construct_face_element(DynamicJITCode *interface_jitcode, int face_index) { return new InterfaceElementLine1dC1(interface_jitcode, this, face_index); }
    oomph::FaceElement * BulkElementTri2dC1::construct_face_element(DynamicJITCode *interface_jitcode, int face_index) { return new InterfaceTElementLine1dC1(interface_jitcode, this, face_index); }
	oomph::FaceElement * BulkElementTri2dC2::construct_face_element(DynamicJITCode *interface_jitcode, int face_index) { return new InterfaceTElementLine1dC2(interface_jitcode, this, face_index); }
}
