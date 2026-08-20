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


// Concrete 3-d bulk elements: bricks (C1/C2), tets (C1/C2, with and without bubble enrichment),
// wedges and pyramids. Shape functions, interpolating-node overrides for the mixed C1-in-C2
// spaces, numpy/outline export and the static space-index tables.

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

	// Tet son of a pyramid: a BulkElementTetra3dC1 bound to the same physics (codeinst) as this pyramid.
	BulkElementBase *BulkElementPyramid3dC1::create_tet_son_instance() const
	{
		BulkElementBase::__CurrentCodeInstance = codeinst;
		auto res = new BulkElementTetra3dC1();
		res->set_code_instance(codeinst); // res is a tet, not a pyramid -> use the public setter
		BulkElementBase::__CurrentCodeInstance = NULL;
		return res;
	}

	const RefinementPattern *BulkElementPyramid3dC1::refinement_pattern() const
	{
		return PyramidMixedRefinementPattern::instance();
	}

	// Tet son of a C2 pyramid: a BulkElementTetra3dC2 bound to the same physics (codeinst) as this pyramid.
	BulkElementBase *BulkElementPyramid3dC2::create_tet_son_instance() const
	{
		BulkElementBase::__CurrentCodeInstance = codeinst;
		auto res = new BulkElementTetra3dC2();
		res->set_code_instance(codeinst); // res is a tet, not a pyramid -> use the public setter
		BulkElementBase::__CurrentCodeInstance = NULL;
		return res;
	}

	const RefinementPattern *BulkElementPyramid3dC2::refinement_pattern() const
	{
		return PyramidMixedRefinementPattern::instance();
	}

	// Brick build() overrides: in a MIXED forest go through the registry (build_as_brick_son) so a brick shares
	// interface nodes with adjacent pyramids/wedges (and other bricks); otherwise keep oomph-lib's native
	// octree build for pure-brick meshes (unchanged, fully validated).
	void BulkElementBrick3dC1::build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt, bool &was_already_built, std::ofstream &new_nodes_file)
	{
		if (oomph::RefineablePyramidElement::Mixed_forest_active)
		{
			if (this->nodes_built()) { was_already_built = true; return; }
			was_already_built = false;
			this->build_as_brick_son(mesh_pt, new_node_pt);
		}
		else
		{
			oomph::RefineableSolidQElement<3>::build(mesh_pt, new_node_pt, was_already_built, new_nodes_file);
			if (was_already_built) return;
		}
		// Same FE overwrite as the 2d solid build; see BulkElementQuad2dC1::build.
		this->reapply_macro_element_positions();
	}

	void BulkElementBrick3dC2::build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt, bool &was_already_built, std::ofstream &new_nodes_file)
	{
		if (oomph::RefineablePyramidElement::Mixed_forest_active)
		{
			if (this->nodes_built()) { was_already_built = true; return; }
			was_already_built = false;
			this->build_as_brick_son(mesh_pt, new_node_pt);
		}
		else
		{
			oomph::RefineableSolidQElement<3>::build(mesh_pt, new_node_pt, was_already_built, new_nodes_file);
			if (was_already_built) return;
		}
		this->reapply_macro_element_positions();
	}

	// 3d counterpart of tri2d_nodal_s_in_father. The tet 1->8 split already has its son->father map as
	// RefineableTElement<3>::son_to_father_local (the very one RefineableTElement<3>::build uses to place
	// the son's nodes), so this only has to feed the son's own local coordinate through it. It is
	// barycentric-affine and therefore covers all 8 sons, the 4 inner (octahedron) ones included.
	static void tet3d_nodal_s_in_father(oomph::RefineableTElement<3> *son, const unsigned int &l, oomph::Vector<double> &sfather)
	{
		oomph::Tree *tree = son->tree_pt();
		if (!tree || !tree->father_pt()) throw_runtime_error("get_nodal_s_in_father on a tet without a father");
		// A tet can also be a son of a PYRAMID (the mixed red split), and that ancestry is not the 1->8
		// tet map -- refuse rather than return a plausible-looking wrong coordinate.
		if (!dynamic_cast<oomph::RefineableTElement<3> *>(tree->father_pt()->object_pt()))
			throw_runtime_error("get_nodal_s_in_father on a tet whose father is not a tet");
		oomph::Vector<double> s_son;
		son->local_coordinate_of_node(l, s_son);
		oomph::RefineableTElement<3>::son_to_father_local(s_son, tree->son_type(), sfather);
	}

	void BulkElementTetra3dC1::get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather)
	{
		tet3d_nodal_s_in_father(this, l, sfather);
	}

	void BulkElementTetra3dC2::get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather)
	{
		tet3d_nodal_s_in_father(this, l, sfather);
	}

	//////////////////////////////

	// BulkElementBrick3dC1: trilinear (8-node) hexahedral element, the 3-d analogue of
	// BulkElementQuad2dC1. DL uses a 4-value (constant + 3 gradient components) representation.
	BulkElementBrick3dC1::BulkElementBrick3dC1()
	{
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 8;
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 8;
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = 8;
		eleminfo.nnode_DL = 4;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}


	// Octree analogue of BulkElementQuad2dC1::get_nodal_s_in_father(): maps node l's local
	// coordinate into the father's, based on which octant this son occupies.
	void BulkElementBrick3dC1::get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather)
	{
	   // TODO: Check whether this is correct
		using namespace oomph::OcTreeNames;
		sfather.resize(3, 0.0);
		int son_type = Tree_pt->son_type();

		oomph::Vector<int> s_lo(3);
		oomph::Vector<int> s_hi(3);
		oomph::Vector<double> s(3);
		oomph::Vector<double> x(3);
      s_lo = octree_pt()->Direction_to_vector[son_type];
      for (unsigned i = 0; i < 3; i++)
      {
        s_lo[i] = (s_lo[i] + 1) / 2 - 1;
      }      
      for (unsigned i = 0; i < 3; i++)
      {
        s_hi[i] = s_lo[i] + 1;
      }		

		oomph::Vector<double> x_small(3);
		oomph::Vector<double> x_large(3);

		oomph::Vector<double> s_fraction(3);
		unsigned n_p = nnode_1d();
		unsigned i2 = l / (n_p*n_p);		
		unsigned i1 = (l - i2*n_p*n_p) / n_p;
		unsigned i0 = l - n_p * i1- n_p*n_p * i2;
		s_fraction[0] = local_one_d_fraction_of_node(i0, 0);
		sfather[0] = s_lo[0] + (s_hi[0] - s_lo[0]) * s_fraction[0];
		s_fraction[1] = local_one_d_fraction_of_node(i1, 1);
		sfather[1] = s_lo[1] + (s_hi[1] - s_lo[1]) * s_fraction[1];
		s_fraction[2] = local_one_d_fraction_of_node(i2, 3);
		sfather[2] = s_lo[2] + (s_hi[2] - s_lo[2]) * s_fraction[2];		
	}




	void BulkElementBrick3dC1::shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		psi[2] = s[1];
		psi[3] = s[2];
	}

	void BulkElementBrick3dC1::dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		psi[2] = s[1];
		psi[3] = s[2];
		dpsi(0, 0) = 0.0;
		dpsi(1, 0) = 1.0;
		dpsi(2, 0) = 0.0;
		dpsi(3, 0) = 0.0;
		dpsi(0, 1) = 0.0;
		dpsi(1, 1) = 0.0;
		dpsi(2, 1) = 1.0;
		dpsi(3, 1) = 0.0;
		dpsi(0, 2) = 0.0;
		dpsi(1, 2) = 0.0;
		dpsi(2, 2) = 0.0;
		dpsi(3, 2) = 1.0;
	}

	void BulkElementBrick3dC1::fill_element_nodal_indices_for_numpy(int *indices, unsigned, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &) const
	{
		if (tesselate_tri)
		{
			throw_runtime_error("Cannot tesselate 3d to tri yet");
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
		}
	}

	std::vector<double> BulkElementBrick3dC1::get_outline(bool)
	{
		std::vector<double> res(0);
		throw_runtime_error("Cannot get outline from 3d elements yet");
		return res;
	}

	////////////////////////////
	

	BulkElementBrick3dC2::BulkElementBrick3dC2()
	{
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 27;
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 8;
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = 8;		
		eleminfo.nnode_of_space[SPACE_INDEX_C2] = 27;
		eleminfo.nnode_of_space[SPACE_INDEX_C2TB] = 27;
		eleminfo.nnode_DL = 4;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}

	void BulkElementBrick3dC2::get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather)
	{
	   // TODO: Check whether this is correct
		using namespace oomph::OcTreeNames;
		sfather.resize(3, 0.0);
		int son_type = Tree_pt->son_type();

		oomph::Vector<int> s_lo(3);
		oomph::Vector<int> s_hi(3);
		oomph::Vector<double> s(3);
		oomph::Vector<double> x(3);
      s_lo = octree_pt()->Direction_to_vector[son_type];
      for (unsigned i = 0; i < 3; i++)
      {
        s_lo[i] = (s_lo[i] + 1) / 2 - 1;
      }      
      for (unsigned i = 0; i < 3; i++)
      {
        s_hi[i] = s_lo[i] + 1;
      }		

		oomph::Vector<double> x_small(3);
		oomph::Vector<double> x_large(3);

		oomph::Vector<double> s_fraction(3);
		unsigned n_p = nnode_1d();
		unsigned i2 = l / (n_p*n_p);		
		unsigned i1 = (l - i2*n_p*n_p) / n_p;
		unsigned i0 = l - n_p * i1- n_p*n_p * i2;
		s_fraction[0] = local_one_d_fraction_of_node(i0, 0);
		sfather[0] = s_lo[0] + (s_hi[0] - s_lo[0]) * s_fraction[0];
		s_fraction[1] = local_one_d_fraction_of_node(i1, 1);
		sfather[1] = s_lo[1] + (s_hi[1] - s_lo[1]) * s_fraction[1];
		s_fraction[2] = local_one_d_fraction_of_node(i2, 3);
		sfather[2] = s_lo[2] + (s_hi[2] - s_lo[2]) * s_fraction[2];		
	}


	void BulkElementBrick3dC2::further_setup_hanging_nodes()
	{

		BulkElementBase::further_setup_hanging_nodes();
		auto *ft = codeinst->get_func_table();
		const int c2_hang = ft->continuous_spaces[SPACE_INDEX_C2].hangindex;
		const bool has_c1 = ft->continuous_spaces[SPACE_INDEX_C1].numfields_basebulk || ft->continuous_spaces[SPACE_INDEX_C1TB].numfields_basebulk;
		if (has_c1 || c2_hang >= 0)
		{
			const unsigned nC2TB = ft->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk;
			const unsigned nC2 = nC2TB + ft->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk;
			// C2 gets its own hang slot when the dominant space is C2TB (see BulkElementQuad2dC2). On a hex
			// the C2TB bubble is cell-interior (vanishes on faces), so oomph's generic setup_hang_for_value
			// installs the same quadratic hang on the C2 slot as on -1 -- a harmless mirror.
			const unsigned start = (c2_hang >= 0) ? nC2TB : nC2;
			for (unsigned int i = start; i < ncont_interpolated_values(); i++)
			{
				this->setup_hang_for_value(i);
			}
		}
	}


	oomph::Node *BulkElementBrick3dC2::interpolating_node_pt(const unsigned &n, const int &value_id)
	{
		if (value_id >= static_cast<int>(codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk + codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk))
		{
			return this->node_pt(this->get_nodal_space_index_to_element_index_map()[SPACE_INDEX_C1][n]);
		}
		else
		{
			return this->node_pt(n);
		}
	}

	double BulkElementBrick3dC2::local_one_d_fraction_of_interpolating_node(const unsigned &n1d, const unsigned &i, const int &value_id)
	{
		if (value_id >= static_cast<int>(codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk + codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk))
		{
			// The C1 nodes are just located on the boundaries at 0 or 1
			return double(n1d);
		}
		else
		{
			return this->local_one_d_fraction_of_node(n1d, i);
		}
	}

	oomph::Node *BulkElementBrick3dC2::get_interpolating_node_at_local_coordinate(const oomph::Vector<double> &s, const int &value_id)
	{
		// TODO: Checl this
		if (value_id >= static_cast<int>(codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk + codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk))
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
	unsigned BulkElementBrick3dC2::ninterpolating_node_1d(const int &value_id)
	{
		if (value_id >= static_cast<int>(codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk + codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk))
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
	unsigned BulkElementBrick3dC2::ninterpolating_node(const int &value_id)
	{
		if (value_id >= static_cast<int>(codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk + codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk))
		{
			return 8;
		}
		else
		{
			return this->nnode();
		}
	}

	void BulkElementBrick3dC2::shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		psi[2] = s[1];
		psi[3] = s[2];
	}

	void BulkElementBrick3dC2::dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		psi[2] = s[1];
		psi[3] = s[2];
		dpsi(0, 0) = 0.0;
		dpsi(1, 0) = 1.0;
		dpsi(2, 0) = 0.0;
		dpsi(3, 0) = 0.0;
		dpsi(0, 1) = 0.0;
		dpsi(1, 1) = 0.0;
		dpsi(2, 1) = 1.0;
		dpsi(3, 1) = 0.0;
		dpsi(0, 2) = 0.0;
		dpsi(1, 2) = 0.0;
		dpsi(2, 2) = 0.0;
		dpsi(3, 2) = 1.0;
	}

	void BulkElementBrick3dC2::shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		double psi1[2], psi2[2], psi3[2];
		oomph::OneDimLagrange::shape<2>(s[0], psi1);
		oomph::OneDimLagrange::shape<2>(s[1], psi2);
		oomph::OneDimLagrange::shape<2>(s[2], psi3);
		for (unsigned i = 0; i < 2; i++)
		{
			for (unsigned j = 0; j < 2; j++)
			{
				for (unsigned k = 0; k < 2; k++)
				{
					/*Multiply the three 1D functions together to get the 3D function*/
					psi[4 * i + 2 * j + k] = psi3[i] * psi2[j] * psi1[k];
				}
			}
		}
	}

	void BulkElementBrick3dC2::dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		double psi1[2], psi2[2], psi3[2];
		double dpsi1[2], dpsi2[2], dpsi3[2];
		oomph::OneDimLagrange::shape<2>(s[0], psi1);
		oomph::OneDimLagrange::shape<2>(s[1], psi2);
		oomph::OneDimLagrange::shape<2>(s[2], psi3);
		oomph::OneDimLagrange::dshape<2>(s[0], dpsi1);
		oomph::OneDimLagrange::dshape<2>(s[1], dpsi2);
		oomph::OneDimLagrange::dshape<2>(s[2], dpsi3);

		// TODO: Check this!
		for (unsigned i = 0; i < 2; i++)
		{
			for (unsigned j = 0; j < 2; j++)
			{
				for (unsigned k = 0; k < 2; k++)
				{
					unsigned ind = 4 * i + 2 * j + k;
					psi[ind] = psi3[i] * psi2[j] * psi1[k];
					dpsi(ind, 0) = psi3[i] * psi2[j] * dpsi1[k];
					dpsi(ind, 1) = psi3[i] * dpsi2[j] * psi1[k];
					dpsi(ind, 2) = dpsi3[i] * psi2[j] * psi1[k];
				}
			}
		}
	}

	void BulkElementBrick3dC2::interpolating_basis(const oomph::Vector<double> &s, oomph::Shape &psi, const int &value_id) const
	{
		if (value_id >= static_cast<int>(codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk + codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk))
		{
			return this->shape_at_s_C1(s, psi);
		}
		else
		{
			return this->shape(s, psi);
		}
	}

	void BulkElementBrick3dC2::fill_element_nodal_indices_for_numpy(int *indices, unsigned, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &) const
	{
		if (tesselate_tri)
		{
			throw_runtime_error("Tesselation not implemented in 3d");
		}
		else
		{
			for (unsigned int i = 0; i < 27; i++)
				indices[i] = i;
		}
	}

	std::vector<double> BulkElementBrick3dC2::get_outline(bool)
	{
		std::vector<double> res(27 * this->nodal_dimension());
		throw_runtime_error("Outline not implemented for 3d");
		return res;
	}

	////////////////////////////////

	BulkElementTetra3dC1::BulkElementTetra3dC1()
	{
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 4;
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 4;
		eleminfo.nnode_DL = 4;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}

	

	void BulkElementTetra3dC1::shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		psi[2] = s[1];
		psi[3] = s[2];
	}

	void BulkElementTetra3dC1::dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		psi[2] = s[1];
		psi[3] = s[2];
		dpsi(0, 0) = 0.0;
		dpsi(1, 0) = 1.0;
		dpsi(2, 0) = 0.0;
		dpsi(3, 0) = 0.0;
		dpsi(0, 1) = 0.0;
		dpsi(1, 1) = 0.0;
		dpsi(2, 1) = 1.0;
		dpsi(3, 1) = 0.0;
		dpsi(0, 2) = 0.0;
		dpsi(1, 2) = 0.0;
		dpsi(2, 2) = 0.0;
		dpsi(3, 2) = 1.0;
	}

	void BulkElementTetra3dC1::fill_element_nodal_indices_for_numpy(int *indices, unsigned, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &) const
	{
		if (tesselate_tri)
		{
			throw_runtime_error("Cannot tesselate 3d to tri yet");
		}
		else
		{
			indices[0] = 0;
			indices[1] = 1;
			indices[2] = 2;
			indices[3] = 3;
		}
	}

	std::vector<double> BulkElementTetra3dC1::get_outline(bool)
	{
		std::vector<double> res(0);
		throw_runtime_error("Cannot get outline from 3d elements yet");
		return res;
	}



	////////////////////////////////

	BulkElementTetra3dC1TB::BulkElementTetra3dC1TB()
	{
		unsigned n_node = this->nnode();
        this->set_n_node(n_node +1);      
        this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 3, 2,true));
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 5;
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 4;
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = 5;
		eleminfo.nnode_DL = 4;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}


    
   
	void BulkElementTetra3dC1TB::shape(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		const double s4 = 1.0 - s[0] - s[1] - s[2];
		const double b = 256.0 * s[0] * s[1] * s[2] * s4;

		psi[0] = s[0] - 0.25 * b;
		psi[1] = s[1] - 0.25 * b;
		psi[2] = s[2] - 0.25 * b;
		psi[3] = s4 - 0.25 * b;
		psi[4] = b;
	}

	void BulkElementTetra3dC1TB::dshape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{		
		const double s4 = 1.0 - s[0] - s[1] - s[2];
		const double b = 256.0 * s[0] * s[1] * s[2] * s4;

		psi[0] = s[0] - 0.25 * b;
		psi[1] = s[1] - 0.25 * b;
		psi[2] = s[2] - 0.25 * b;
		psi[3] = s4 - 0.25 * b;
		psi[4] = b;

		const double db_ds1 = 256.0 * s[1] * s[2] * (s4 - s[0]);
		const double db_ds2 = 256.0 * s[0] * s[2] * (s4 - s[1]);
		const double db_ds3 = 256.0 * s[0] * s[1] * (s4 - s[2]);

		dpsi(0, 0) =  1.0 - 0.25 * db_ds1;
		dpsi(0, 1) =       - 0.25 * db_ds2;
		dpsi(0, 2) =       - 0.25 * db_ds3;

		dpsi(1, 0) =       - 0.25 * db_ds1;
		dpsi(1, 1) =  1.0 - 0.25 * db_ds2;
		dpsi(1, 2) =       - 0.25 * db_ds3;

		dpsi(2, 0) =       - 0.25 * db_ds1;
		dpsi(2, 1) =       - 0.25 * db_ds2;
		dpsi(2, 2) =  1.0 - 0.25 * db_ds3;

		dpsi(3, 0) = -1.0 - 0.25 * db_ds1;
		dpsi(3, 1) = -1.0 - 0.25 * db_ds2;
		dpsi(3, 2) = -1.0 - 0.25 * db_ds3;

		dpsi(4, 0) = db_ds1;
		dpsi(4, 1) = db_ds2;
		dpsi(4, 2) = db_ds3;
	}

    void BulkElementTetra3dC1TB::shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		psi[0] = s[0];
		psi[1] = s[1];
		psi[2] = s[2];
		psi[3] = 1.0 - s[0] - s[1] - s[2];
	}
    
    void BulkElementTetra3dC1TB::dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = s[0];
		psi[1] = s[1];
		psi[2] = s[2];
		psi[3] = 1.0 - s[0] - s[1] - s[2];

		dpsi(0, 0) = 1.0;
		dpsi(0, 1) = 0.0;
		dpsi(0, 2) = 0.0;
		dpsi(1, 0) = 0.0;
		dpsi(1, 1) = 1.0;
		dpsi(1, 2) = 0.0;
		dpsi(2, 0) = 0.0;
		dpsi(2, 1) = 0.0;
		dpsi(2, 2) = 1.0;
		dpsi(3, 0) = -1.0;
		dpsi(3, 1) = -1.0;
		dpsi(3, 2) = -1.0;
	}    
	
    // Numpy/VTK export: 3d elements cannot yet be tesselated into triangles, so just pass the local node indices through unchanged.
    void BulkElementTetra3dC1TB::fill_element_nodal_indices_for_numpy(int *indices, unsigned, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &) const
	{
		if (tesselate_tri)
		{
			throw_runtime_error("Tesselation not implemented in 3d");
		}
		else
		{
			for (unsigned int i = 0; i < this->nnode(); i++)
				indices[i] = i;
		}
	}

	// Local coordinates of the 4 linear (corner) nodes plus the interior bubble node (index 4, at the centroid).
	void BulkElementTetra3dC1TB::local_coordinate_of_node(const unsigned &j, oomph::Vector<double> &s) const
	{
		// The resize is part of the contract, not a precaution: FiniteElement::get_node_at_local_coordinate
		// passes a DEFAULT-CONSTRUCTED (i.e. empty) Vector and relies on this function to size it, so
		// writing s[0..2] without it is an out-of-bounds write on an empty std::vector. That is what made
		// every refinement of a C1TB tet segfault -- oomph's own TBubbleEnrichedElementShape<3,3> and
		// pyoomph's 2d BulkElementTri2dC1TB both resize first, which is why only this one shape crashed.
		s.resize(3);
		if (j==0)
		{
			s[0]=1.0; s[1]=0.0; s[2]=0.0;
		}
		else if (j==1)
		{
			s[0]=0.0; s[1]=1.0; s[2]=0.0;
		}
		else if (j==2)
		{
			s[0]=0.0; s[1]=0.0; s[2]=1.0;
		}
		else if (j==3)
		{
			s[0]=0.0; s[1]=0.0; s[2]=0.0;
		}
		else if (j==4)
		{
			s[0]=0.25; s[1]=0.25; s[2]=0.25;
		}
		else throw_runtime_error("Invalid node index " + std::to_string(j) + " for a C1TB tetrahedron (it has 5 nodes)");
	}


	////////////////////////////////
	// 10-node quadratic (2nd order) tetrahedron. When constructed with has_bubble=true, this instance is
	// used as the base of the 15-node bubble-enriched BulkElementTetra3dC2TB variant, which adds a further
	// interior enrichment field on top (see below); the node counts of the C2TB/C1TB spaces are set up
	// accordingly here even though the bubble node itself is only added by the TB subclass.

	BulkElementTetra3dC2::BulkElementTetra3dC2(bool has_bubble)
	{
		eleminfo.elem_ptr = this;
		eleminfo.nnode = (has_bubble ? 15 : 10);
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 4;
		eleminfo.nnode_of_space[SPACE_INDEX_C2TB] = (has_bubble ? 15 : 10);
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = (has_bubble ? 5 : 4);
		eleminfo.nnode_of_space[SPACE_INDEX_C2] = 10;
		eleminfo.nnode_DL = 4;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}

   // Refinement creates a plain (non-bubble) son by default unless this element is actually the TB variant,
   // in which case the son must keep the bubble-enriched node layout too.
   BulkElementBase *BulkElementTetra3dC2::create_son_instance() const
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      // A C2TB (bubble-enriched) father must spawn a genuine BulkElementTetra3dC2TB son: the bubble
      // needs real 11th-15th (4 face-centroid + 1 volume-centroid) node slots and the 15-node nodal
      // space map. A plain BulkElementTetra3dC2(true) only bumps nnode_of_space[C2TB] to 15 while
      // keeping 10 oomph node slots and the 10-node map -> local_coordinate_of_node(10..) throws.
      BulkElementTetra3dC2 *res;
      if (dynamic_cast<const BulkElementTetra3dC2TB*>(this) != nullptr) res = new BulkElementTetra3dC2TB();
      else res = new BulkElementTetra3dC2(false);
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }


	// No extra hanging-node bookkeeping is required beyond the base class; the commented-out block is left
	// as a reminder of how the embedded C1 sub-space would be hung off the C2 nodes if ever needed here
	// (this is already done for the analogous 2d Quad/Tri elements).
	void BulkElementTetra3dC2::further_setup_hanging_nodes()
	{
		BulkElementBase::further_setup_hanging_nodes();
		// 3d analogue of BulkElementTri2dC2::further_setup_hanging_nodes. The geometric slot -1 is done in
		// RefineableTElement<3>::setup_hanging_nodes; here we drive the separate C1/C2 value slots (their hang
		// on a coarser neighbour) and then apply the C1(TB) mid-node rule. Runs before oomph's
		// complete_hanging_nodes, so recursive master chains are flattened there.
		auto *ft = codeinst->get_func_table();
		const int c2_hang = ft->continuous_spaces[SPACE_INDEX_C2].hangindex;
		const bool has_c1 = ft->continuous_spaces[SPACE_INDEX_C1].numfields_basebulk || ft->continuous_spaces[SPACE_INDEX_C1TB].numfields_basebulk;
		if (!(has_c1 || c2_hang >= 0))
			return;
		const unsigned nC2TB = ft->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk;
		const unsigned nC2 = nC2TB + ft->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk;
		// C2 owns its own hang slot only when the dominant space is C2TB (its enriched face trace differs
		// from the plain C2 one on a tet face); otherwise C2 shares the -1 geometric slot done above.
		const unsigned start = (c2_hang >= 0) ? nC2TB : nC2;
		for (unsigned i = start; i < ncont_interpolated_values(); i++)
			this->setup_hang_for_value(i);
		// C1(TB) mid-node rule: every C2 edge-mid node carrying a separate C1 slot hangs 0.5/0.5 on its two
		// edge-corner nodes (the C1-on-C2 rule; also constrains the STALE pressure slot left on a former fine
		// corner that becomes a plain conforming C2 mid-edge after UNREFINEMENT). Nodes already hung across a
		// 2:1 neighbour (by the edge/face passes above) are skipped via is_hanging; ordinary mid-edge nodes
		// that never carried the slot are skipped via the nvalue guard. The edge-mid <-> corner-pair mapping
		// is read off the local coordinates (mid = midpoint of its two corner vertices).
		if (this->nnode() < 10)
			return;
		std::vector<int> c1_slots;
		for (int sp : {SPACE_INDEX_C1TB, SPACE_INDEX_C1})
		{
			int h = ft->continuous_spaces[sp].hangindex;
			if (h >= 0)
				c1_slots.push_back(h);
		}
		static const double VC[4][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}};
		oomph::Vector<double> sm(3);
		for (unsigned em = 4; em < 10; em++) // the 6 edge-mid nodes of a (C2) tet
		{
			oomph::Node *M = this->node_pt(em);
			this->local_coordinate_of_node(em, sm);
			int ca = -1, cb = -1;
			for (int v = 0; v < 4; v++)
				for (int w = v + 1; w < 4; w++)
				{
					bool match = true;
					for (int d = 0; d < 3; d++)
						if (std::abs(0.5 * (VC[v][d] + VC[w][d]) - sm[d]) > 1e-9) { match = false; break; }
					if (match) { ca = v; cb = w; }
				}
			if (ca < 0)
				continue;
			oomph::Node *A = this->node_pt(ca), *B = this->node_pt(cb);
			for (int slot : c1_slots)
			{
				if ((int)M->nvalue() <= slot || M->is_hanging(slot)) continue;     // no stale slot, or already hung (2:1)
				if ((int)A->nvalue() <= slot || (int)B->nvalue() <= slot) continue; // masters must carry the dof
				oomph::HangInfo *hang = new oomph::HangInfo(2);
				hang->set_master_node_pt(0, A, 0.5);
				hang->set_master_node_pt(1, B, 0.5);
				M->set_hanging_pt(hang, slot);
			}
		}
	}

	// Discontinuous (elemental, non-nodal) linear shape functions {1, s0, s1, s2} for the DL space of this element.
	void BulkElementTetra3dC2::shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		psi[2] = s[1];
		psi[3] = s[2];
	}

	void BulkElementTetra3dC2::dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		psi[2] = s[1];
		psi[3] = s[2];
		dpsi(0, 0) = 0.0;
		dpsi(1, 0) = 1.0;
		dpsi(2, 0) = 0.0;
		dpsi(3, 0) = 0.0;
		dpsi(0, 1) = 0.0;
		dpsi(1, 1) = 0.0;
		dpsi(2, 1) = 1.0;
		dpsi(3, 1) = 0.0;
		dpsi(0, 2) = 0.0;
		dpsi(1, 2) = 0.0;
		dpsi(2, 2) = 0.0;
		dpsi(3, 2) = 1.0;
	}

	// Linear (C1) shape functions in barycentric-like coordinates, for the C1 sub-space embedded in the
	// corner nodes of this quadratic element (used when an additional linear field lives on the same tet).
	void BulkElementTetra3dC2::shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		psi[0] = s[0];
		psi[1] = s[1];
		psi[2] = s[2];
		psi[3] = 1.0 - s[0] - s[1] - s[2];
	}

	void BulkElementTetra3dC2::dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = s[0];
		psi[1] = s[1];
		psi[2] = s[2];
		psi[3] = 1.0 - s[0] - s[1] - s[2];

		// Derivatives
		dpsi(0, 0) = 1.0;
		dpsi(0, 1) = 0.0;
		dpsi(0, 2) = 0.0;

		dpsi(1, 0) = 0.0;
		dpsi(1, 1) = 1.0;
		dpsi(1, 2) = 0.0;

		dpsi(2, 0) = 0.0;
		dpsi(2, 1) = 0.0;
		dpsi(2, 2) = 1.0;

		dpsi(3, 0) = -1.0;
		dpsi(3, 1) = -1.0;
		dpsi(3, 2) = -1.0;
	}

	// C1TB (5-node, bubble-enriched linear) shape functions embedded in this C2TB element; identical formulas
	// to BulkElementTetra3dC1TB::shape/dshape_local, duplicated here as the C1TB sub-space of the quadratic bubble element.
	void BulkElementTetra3dC2TB::shape_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		const double s4 = 1.0 - s[0] - s[1] - s[2];
		const double b = 256.0 * s[0] * s[1] * s[2] * s4;

		psi[0] = s[0] - 0.25 * b;
		psi[1] = s[1] - 0.25 * b;
		psi[2] = s[2] - 0.25 * b;
		psi[3] = s4 - 0.25 * b;
		psi[4] = b;
	}

	void BulkElementTetra3dC2TB::dshape_local_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{		
		const double s4 = 1.0 - s[0] - s[1] - s[2];
		const double b = 256.0 * s[0] * s[1] * s[2] * s4;

		psi[0] = s[0] - 0.25 * b;
		psi[1] = s[1] - 0.25 * b;
		psi[2] = s[2] - 0.25 * b;
		psi[3] = s4 - 0.25 * b;
		psi[4] = b;

		const double db_ds1 = 256.0 * s[1] * s[2] * (s4 - s[0]);
		const double db_ds2 = 256.0 * s[0] * s[2] * (s4 - s[1]);
		const double db_ds3 = 256.0 * s[0] * s[1] * (s4 - s[2]);

		dpsi(0, 0) =  1.0 - 0.25 * db_ds1;
		dpsi(0, 1) =       - 0.25 * db_ds2;
		dpsi(0, 2) =       - 0.25 * db_ds3;

		dpsi(1, 0) =       - 0.25 * db_ds1;
		dpsi(1, 1) =  1.0 - 0.25 * db_ds2;
		dpsi(1, 2) =       - 0.25 * db_ds3;

		dpsi(2, 0) =       - 0.25 * db_ds1;
		dpsi(2, 1) =       - 0.25 * db_ds2;
		dpsi(2, 2) =  1.0 - 0.25 * db_ds3;

		dpsi(3, 0) = -1.0 - 0.25 * db_ds1;
		dpsi(3, 1) = -1.0 - 0.25 * db_ds2;
		dpsi(3, 2) = -1.0 - 0.25 * db_ds3;

		dpsi(4, 0) = db_ds1;
		dpsi(4, 1) = db_ds2;
		dpsi(4, 2) = db_ds3;
	}

	// Selects the shape functions to use for interpolating a given field: fields beyond the quadratic
	// (C2/C2TB) base-bulk fields are interpolated with the embedded linear (C1) shape functions on the
	// corner nodes, all others use the regular quadratic shape functions (mirrors BulkElementQuad2dC2::interpolating_basis).
	void BulkElementTetra3dC2::interpolating_basis(const oomph::Vector<double> &s, oomph::Shape &psi, const int &value_id) const
	{
		if (value_id >= static_cast<int>(codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk + codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk))
		{
			return this->shape_at_s_C1(s, psi);
		}
		else
		{
			return this->shape(s, psi);
		}
	}

	// Numpy/VTK export: 3d elements cannot yet be tesselated into triangles, so just pass the local node indices through unchanged.
	void BulkElementTetra3dC2::fill_element_nodal_indices_for_numpy(int *indices, unsigned, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &) const
	{
		if (tesselate_tri)
		{
			throw_runtime_error("Tesselation not implemented in 3d");
		}
		else
		{
			for (unsigned int i = 0; i < this->nnode(); i++)
				indices[i] = i;
		}
	}

	// Not yet implemented: outlines (used for plotting the element boundary) are only supported for 2d elements.
	std::vector<double> BulkElementTetra3dC2::get_outline(bool)
	{
		std::vector<double> res(10 * this->nodal_dimension());
		throw_runtime_error("Outline not implemented for 3d");
		return res;
	}

	///////////////////////////////
	// 15-node bubble-enriched quadratic tetrahedron: adds a single interior bubble node (index 14) to the
	// 10-node BulkElementTetra3dC2, using the enriched integration scheme below.

	BulkElementTetra3dC2TB::BulkElementTetra3dC2TB() : BulkElementTetra3dC2(true)
	{
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 15;
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 4;
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = 5;		
		eleminfo.nnode_of_space[SPACE_INDEX_C2TB] = 15;
		eleminfo.nnode_of_space[SPACE_INDEX_C2] = 10;
		eleminfo.nnode_DL = 4;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_n_node(eleminfo.nnode);
		this->set_nodal_dimension(eleminfo.nodal_dim);
		this->set_integration_scheme(&Default_enriched_integration_scheme);
	}


	 // Beyond the standard 6 face nodes built by the base class, a bubble-enriched face element also needs
	 // access to the face-centre node of the corresponding tet face (the node lying on the 2d bubble/enrichment
	 // sub-space of that triangular face); Central_node_on_face maps each local face index to that bulk node.
	 void BulkElementTetra3dC2TB::build_face_element(const int& face_index, oomph::FaceElement* face_element_pt)
	{
		BulkElementTetra3dC2::build_face_element(face_index, face_element_pt);
		face_element_pt->nbulk_value_resize(7);
		face_element_pt->bulk_node_number_resize(7);
		// So the faces are
		// 0 : s_0 fixed
		// 1 : s_1 fixed
		// 2 : s_2 fixed
		// 3 : sloping face
		unsigned bulk_number = Central_node_on_face[face_index];
		face_element_pt->node_pt(6) = node_pt(bulk_number);
		face_element_pt->bulk_node_number(6) = bulk_number;
		face_element_pt->nbulk_value(6) =required_nvalue(bulk_number);
   }


    ///////////////////////////////
    // 6-node linear wedge/triangular-prism element (2 triangular + 3 quadrilateral faces).
    BulkElementWedge3dC1::BulkElementWedge3dC1()
	{
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 6;
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 6;
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = 0;		
		eleminfo.nnode_of_space[SPACE_INDEX_C2TB] = 0;
		eleminfo.nnode_of_space[SPACE_INDEX_C2] = 0;
		eleminfo.nnode_DL = 4;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_n_node(eleminfo.nnode);
		this->set_nodal_dimension(eleminfo.nodal_dim);	
		allocate_discontinous_fields();	
	}


    // Discontinuous (elemental, non-nodal) linear shape functions {1, s0, s1, s2} for the DL space of this element.
    void BulkElementWedge3dC1::shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		psi[2] = s[1];
		psi[3] = s[2];
	}

	void BulkElementWedge3dC1::dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		psi[2] = s[1];
		psi[3] = s[2];
		dpsi(0, 0) = 0.0;
		dpsi(1, 0) = 1.0;
		dpsi(2, 0) = 0.0;
		dpsi(3, 0) = 0.0;
		dpsi(0, 1) = 0.0;
		dpsi(1, 1) = 0.0;
		dpsi(2, 1) = 1.0;
		dpsi(3, 1) = 0.0;
		dpsi(0, 2) = 0.0;
		dpsi(1, 2) = 0.0;
		dpsi(2, 2) = 0.0;
		dpsi(3, 2) = 1.0;
	}

	// Numpy/VTK export: 3d elements cannot yet be tesselated into triangles, so just pass the local node indices through unchanged.
	void BulkElementWedge3dC1::fill_element_nodal_indices_for_numpy(int *indices, unsigned, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &) const
	{
		if (tesselate_tri)
		{
			throw_runtime_error("Tesselation not implemented in 3d");
		}
		else
		{
			for (unsigned int i = 0; i < this->nnode(); i++)
				indices[i] = i;
		}
	}

	// Not yet implemented: outlines (used for plotting the element boundary) are only supported for 2d elements.
	std::vector<double> BulkElementWedge3dC1::get_outline(bool)
	{
		std::vector<double> res(0);
		throw_runtime_error("Cannot get outline from 3d elements yet");
		return res;
	}

	////////////////////////////////
	// 5-node linear pyramid element (1 quadrilateral base + 4 triangular faces).

	BulkElementPyramid3dC1::BulkElementPyramid3dC1()
	{
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 5;
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 5;
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = 0;		
		eleminfo.nnode_of_space[SPACE_INDEX_C2TB] = 0;
		eleminfo.nnode_of_space[SPACE_INDEX_C2] = 0;
		eleminfo.nnode_DL = 4;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_n_node(eleminfo.nnode);
		this->set_nodal_dimension(eleminfo.nodal_dim);	
		allocate_discontinous_fields();	
	}


    // Discontinuous (elemental, non-nodal) linear shape functions {1, s0, s1, s2} for the DL space of this element.
    void BulkElementPyramid3dC1::shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		psi[2] = s[1];
		psi[3] = s[2];
	}

	void BulkElementPyramid3dC1::dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		psi[2] = s[1];
		psi[3] = s[2];
		dpsi(0, 0) = 0.0;
		dpsi(1, 0) = 1.0;
		dpsi(2, 0) = 0.0;
		dpsi(3, 0) = 0.0;
		dpsi(0, 1) = 0.0;
		dpsi(1, 1) = 0.0;
		dpsi(2, 1) = 1.0;
		dpsi(3, 1) = 0.0;
		dpsi(0, 2) = 0.0;
		dpsi(1, 2) = 0.0;
		dpsi(2, 2) = 0.0;
		dpsi(3, 2) = 1.0;
	}

	// Numpy/VTK export: 3d elements cannot yet be tesselated into triangles, so just pass the local node indices through unchanged.
	void BulkElementPyramid3dC1::fill_element_nodal_indices_for_numpy(int *indices, unsigned, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &) const
	{
		if (tesselate_tri)
		{
			throw_runtime_error("Tesselation not implemented in 3d");
		}
		else
		{
			for (unsigned int i = 0; i < this->nnode(); i++)
				indices[i] = i;
		}
	}

	// Not yet implemented: outlines (used for plotting the element boundary) are only supported for 2d elements.
	std::vector<double> BulkElementPyramid3dC1::get_outline(bool)
	{
		std::vector<double> res(0);
		throw_runtime_error("Cannot get outline from 3d elements yet");
		return res;
	}

	///////////////////////////////
	// 18-node quadratic wedge/prism element (the C2 counterpart of BulkElementWedge3dC1).

    BulkElementWedge3dC2::BulkElementWedge3dC2()
	{
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 18;
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 6; 
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = 0;		
		eleminfo.nnode_of_space[SPACE_INDEX_C2TB] = 0;
		eleminfo.nnode_of_space[SPACE_INDEX_C2] = 18;
		eleminfo.nnode_DL = 4;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_n_node(eleminfo.nnode);
		this->set_nodal_dimension(eleminfo.nodal_dim);	
		allocate_discontinous_fields();	
	}


    // Discontinuous (elemental, non-nodal) linear shape functions {1, s0, s1, s2} for the DL space of this element.
    void BulkElementWedge3dC2::shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		psi[2] = s[1];
		psi[3] = s[2];
	}

	void BulkElementWedge3dC2::dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		psi[2] = s[1];
		psi[3] = s[2];
		dpsi(0, 0) = 0.0;
		dpsi(1, 0) = 1.0;
		dpsi(2, 0) = 0.0;
		dpsi(3, 0) = 0.0;
		dpsi(0, 1) = 0.0;
		dpsi(1, 1) = 0.0;
		dpsi(2, 1) = 1.0;
		dpsi(3, 1) = 0.0;
		dpsi(0, 2) = 0.0;
		dpsi(1, 2) = 0.0;
		dpsi(2, 2) = 0.0;
		dpsi(3, 2) = 1.0;
	}

	// Numpy/VTK export: 3d elements cannot yet be tesselated into triangles, so just pass the local node indices through unchanged.
	void BulkElementWedge3dC2::fill_element_nodal_indices_for_numpy(int *indices, unsigned, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &) const
	{
		if (tesselate_tri)
		{
			throw_runtime_error("Tesselation not implemented in 3d");
		}
		else
		{
			for (unsigned int i = 0; i < this->nnode(); i++)
				indices[i] = i;
		}
	}

	// Not yet implemented: outlines (used for plotting the element boundary) are only supported for 2d elements.
	std::vector<double> BulkElementWedge3dC2::get_outline(bool)
	{
		std::vector<double> res(0);
		throw_runtime_error("Cannot get outline from 3d elements yet");
		return res;
	}

    BulkElementPyramid3dC2::BulkElementPyramid3dC2() 
	{
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 14;
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 5; 
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = 0;		
		eleminfo.nnode_of_space[SPACE_INDEX_C2TB] = 0;
		eleminfo.nnode_of_space[SPACE_INDEX_C2] = 14;
		eleminfo.nnode_DL = 4;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_n_node(eleminfo.nnode);
		this->set_nodal_dimension(eleminfo.nodal_dim);	
		allocate_discontinous_fields();	
	}


    void BulkElementPyramid3dC2::shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		psi[2] = s[1];
		psi[3] = s[2];
	}

	void BulkElementPyramid3dC2::dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		psi[2] = s[1];
		psi[3] = s[2];
		dpsi(0, 0) = 0.0;
		dpsi(1, 0) = 1.0;
		dpsi(2, 0) = 0.0;
		dpsi(3, 0) = 0.0;
		dpsi(0, 1) = 0.0;
		dpsi(1, 1) = 0.0;
		dpsi(2, 1) = 1.0;
		dpsi(3, 1) = 0.0;
		dpsi(0, 2) = 0.0;
		dpsi(1, 2) = 0.0;
		dpsi(2, 2) = 0.0;
		dpsi(3, 2) = 1.0;
	}

	void BulkElementPyramid3dC2::fill_element_nodal_indices_for_numpy(int *indices, unsigned, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &) const
	{
		if (tesselate_tri)
		{
			throw_runtime_error("Tesselation not implemented in 3d");
		}
		else
		{
			for (unsigned int i = 0; i < this->nnode(); i++)
				indices[i] = i;
		}
	}

	std::vector<double> BulkElementPyramid3dC2::get_outline(bool)
	{
		std::vector<double> res(0);
		throw_runtime_error("Cannot get outline from 3d elements yet");
		return res;
	}
	oomph::TBubbleEnrichedGauss<3, 3> BulkElementTetra3dC2TB::Default_enriched_integration_scheme;
	// Local index of the face bubble node of each face: face 0 is s_0 fixed, 1 is s_1, 2 is s_2 and 3
	// is the sloping face. Read by both build_face_element() and node_index_on_face().
	const std::vector<unsigned> BulkElementTetra3dC2TB::Central_node_on_face{13, 12, 10, 11};

	const std::vector<std::vector<unsigned>> BulkElementBrick3dC1::Nodal_Space_Index_To_Element_Index_Map={
		{}, // C2TB 
		{}, // C2
		{0,1,2,3,4,5,6,7}, // C1TB
		{0,1,2,3,4,5,6,7}  // C1
	};

	const std::vector<std::vector<unsigned>> BulkElementBrick3dC2::Nodal_Space_Index_To_Element_Index_Map={
		{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26}, // C2TB 
		{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26}, // C2
		{0, 2, 6, 8, 18, 20, 24, 26}, // C1TB
		{0, 2, 6, 8, 18, 20, 24, 26}  // C1
	};

	const std::vector<std::vector<unsigned>> BulkElementTetra3dC1::Nodal_Space_Index_To_Element_Index_Map={
		{}, // C2TB 
		{}, // C2
		{}, // C1TB
		{0,1,2,3}  // C1
	};
	

	const std::vector<std::vector<unsigned>> BulkElementTetra3dC1TB::Nodal_Space_Index_To_Element_Index_Map={
		{}, // C2TB 
		{}, // C2
		{0,1,2,3,4}, // C1TB
		{0,1,2,3}  // C1
	};

	const std::vector<std::vector<unsigned>> BulkElementTetra3dC2::Nodal_Space_Index_To_Element_Index_Map={
		{}, // C2TB 
		{0,1,2,3,4,5,6,7,8,9}, // C2
		{}, // C1TB
		{0,1,2,3}  // C1
	};

	const std::vector<std::vector<unsigned>> BulkElementTetra3dC2TB::Nodal_Space_Index_To_Element_Index_Map={
		{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14}, // C2TB 
		{0,1,2,3,4,5,6,7,8,9}, // C2
		{0,1,2,3,14}, // C1TB
		{0,1,2,3}  // C1
	};

	const std::vector<std::vector<unsigned>> BulkElementWedge3dC1::Nodal_Space_Index_To_Element_Index_Map={
		{}, // C2TB 
		{}, // C2
		{}, // C1TB
		{0,1,2,3,4,5}  // C1
	};

	const std::vector<std::vector<unsigned>> BulkElementWedge3dC2::Nodal_Space_Index_To_Element_Index_Map={
		{}, // C2TB 
		{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17}, // C2
		{}, // C1TB
		{0,1,2,12,13,14}  // C1
	};

	const std::vector<std::vector<unsigned>> BulkElementPyramid3dC1::Nodal_Space_Index_To_Element_Index_Map={
		{}, // C2TB 
		{}, // C2
		{}, // C1TB
		{0,1,2,3,4}  // C1
	};

	const std::vector<std::vector<unsigned>> BulkElementPyramid3dC2::Nodal_Space_Index_To_Element_Index_Map={
		{}, // C2TB 
		{0,1,2,3,4,5,6,7,8,9,10,11,12,13}, // C2
		{}, // C1TB
		{0,1,2,3,4}  // C1
	};

	const std::vector<std::vector<int>> BulkElementBrick3dC1::Element_Index_To_Nodal_Space_Index_Map={
		{}, // C2TB
		{}, // C2
		{0,1,2,3,4,5,6,7}, // C1TB
		{0,1,2,3,4,5,6,7}  // C1
	};

	const std::vector<std::vector<int>> BulkElementBrick3dC2::Element_Index_To_Nodal_Space_Index_Map={
		{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26}, // C2TB
		{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26}, // C2
		{0,-1,1,-1,-1,-1,2,-1,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,4,-1,5,-1,-1,-1,6,-1,7}, // C1TB
		{0,-1,1,-1,-1,-1,2,-1,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,4,-1,5,-1,-1,-1,6,-1,7}  // C1
	};

	const std::vector<std::vector<int>> BulkElementTetra3dC1::Element_Index_To_Nodal_Space_Index_Map={
		{}, // C2TB
		{}, // C2
		{}, // C1TB
		{0,1,2,3}  // C1
	};	

	const std::vector<std::vector<int>> BulkElementTetra3dC1TB::Element_Index_To_Nodal_Space_Index_Map={
		{}, // C2TB
		{}, // C2
		{0,1,2,3,4}, // C1TB
		{0,1,2,3,-1}  // C1
	};

	const std::vector<std::vector<int>> BulkElementTetra3dC2::Element_Index_To_Nodal_Space_Index_Map={
		{}, // C2TB
		{0,1,2,3,4,5,6,7,8,9}, // C2
		{}, // C1TB
		{0,1,2,3,-1,-1,-1,-1,-1,-1}  // C1
	};

	const std::vector<std::vector<int>> BulkElementTetra3dC2TB::Element_Index_To_Nodal_Space_Index_Map={
		{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14}, // C2TB
		{0,1,2,3,4,5,6,7,8,9,-1,-1,-1,-1,-1}, // C2
		{0,1,2,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,4}, // C1TB
		{0,1,2,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}  // C1
	};


	const std::vector<std::vector<int>> BulkElementWedge3dC1::Element_Index_To_Nodal_Space_Index_Map={
		{}, // C2TB
		{}, // C2
		{}, // C1TB
		{0,1,2,3,4,5}  // C1
	};

	const std::vector<std::vector<int>> BulkElementWedge3dC2::Element_Index_To_Nodal_Space_Index_Map={
		{}, // C2TB
		{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17}, // C2
		{}, // C1TB
		{0,1,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,3,4,5,-1,-1,-1}  // C1
	};


	const std::vector<std::vector<int>> BulkElementPyramid3dC1::Element_Index_To_Nodal_Space_Index_Map={
		{}, // C2TB
		{}, // C2
		{}, // C1TB
		{0,1,2,3,4}  // C1
	};

	const std::vector<std::vector<int>> BulkElementPyramid3dC2::Element_Index_To_Nodal_Space_Index_Map={
		{}, // C2TB
		{0,1,2,3,4,5,6,7,8,9,10,11,12,13}, // C2
		{}, // C1TB
		{0,1,2,3,4,-1,-1,-1,-1,-1,-1,-1,-1,-1}  // C1
	};
	const std::vector<unsigned> BulkElementBrick3dC1::Non_Vertex_Node_Indices={};
	const std::vector<unsigned> BulkElementBrick3dC2::Non_Vertex_Node_Indices={1,3,4,5,7,9,10,11,12,13,14,15,16,17,19,21,22,23,25};
	const std::vector<unsigned> BulkElementTetra3dC1::Non_Vertex_Node_Indices={};
	const std::vector<unsigned> BulkElementTetra3dC1TB::Non_Vertex_Node_Indices={4};
	const std::vector<unsigned> BulkElementTetra3dC2::Non_Vertex_Node_Indices={4,5,6,7,8,9};
	const std::vector<unsigned> BulkElementTetra3dC2TB::Non_Vertex_Node_Indices={4,5,6,7,8,9,10,11,12,13,14};
	const std::vector<unsigned> BulkElementWedge3dC1::Non_Vertex_Node_Indices={};
	const std::vector<unsigned> BulkElementWedge3dC2::Non_Vertex_Node_Indices={3,4,5,6,7,8,9,10,11,15,16,17}; 
	const std::vector<unsigned> BulkElementPyramid3dC1::Non_Vertex_Node_Indices={};
	const std::vector<unsigned> BulkElementPyramid3dC2::Non_Vertex_Node_Indices={5,6,7,8,9,10,11,12,13};

	const std::vector<int> BulkElementBrick3dC1::Possible_Face_Indices={-3,-2,-1,1,2,3};
	const std::vector<int> BulkElementBrick3dC2::Possible_Face_Indices={-3,-2,-1,1,2,3};

	const std::vector<int> BulkElementTetra3dC1::Possible_Face_Indices={0,1,2,3};
	const std::vector<int> BulkElementTetra3dC2::Possible_Face_Indices={0,1,2,3};

	const std::vector<int> BulkElementWedge3dC1::Possible_Face_Indices={0,1,2,3,4};
	const std::vector<int> BulkElementWedge3dC2::Possible_Face_Indices={0,1,2,3,4};
	
	const std::vector<int> BulkElementPyramid3dC1::Possible_Face_Indices={0,1,2,3,4};
	const std::vector<int> BulkElementPyramid3dC2::Possible_Face_Indices={0,1,2,3,4};

	std::vector<pyoomph::Node*> BulkElementBrick3dC1::get_vertex_nodes_of_face(const int &face) const
	{	  
	  	if (face==-3) { return {static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(1)),static_cast<pyoomph::Node*>(this->node_pt(2)),static_cast<pyoomph::Node*>(this->node_pt(3))};}
 		else if (face==-2) { return {static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(1)),static_cast<pyoomph::Node*>(this->node_pt(4)),static_cast<pyoomph::Node*>(this->node_pt(5))};}
 		else if (face==-1) { return {static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(2)),static_cast<pyoomph::Node*>(this->node_pt(4)),static_cast<pyoomph::Node*>(this->node_pt(6))};}
 		else if (face==1) { return {static_cast<pyoomph::Node*>(this->node_pt(1)),static_cast<pyoomph::Node*>(this->node_pt(3)),static_cast<pyoomph::Node*>(this->node_pt(5)),static_cast<pyoomph::Node*>(this->node_pt(7))};}
 		else if (face==2) { return {static_cast<pyoomph::Node*>(this->node_pt(2)),static_cast<pyoomph::Node*>(this->node_pt(3)),static_cast<pyoomph::Node*>(this->node_pt(6)),static_cast<pyoomph::Node*>(this->node_pt(7))};}
 		else if (face==3) { return {static_cast<pyoomph::Node*>(this->node_pt(4)),static_cast<pyoomph::Node*>(this->node_pt(5)),static_cast<pyoomph::Node*>(this->node_pt(6)),static_cast<pyoomph::Node*>(this->node_pt(7))};}				
		else throw_runtime_error("Invalid face index for brick element");
	}

	std::vector<pyoomph::Node*> BulkElementBrick3dC2::get_vertex_nodes_of_face(const int &face) const
	{	  
	 if (face==3) { return {static_cast<pyoomph::Node*>(this->node_pt(18)),static_cast<pyoomph::Node*>(this->node_pt(20)),static_cast<pyoomph::Node*>(this->node_pt(24)),static_cast<pyoomph::Node*>(this->node_pt(26))};}
     else if (face==-3) { return {static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(2)),static_cast<pyoomph::Node*>(this->node_pt(6)),static_cast<pyoomph::Node*>(this->node_pt(8))};}
     else if (face==-2) { return {static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(2)),static_cast<pyoomph::Node*>(this->node_pt(18)),static_cast<pyoomph::Node*>(this->node_pt(20))};}
 	 else if (face==-1) { return {static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(6)),static_cast<pyoomph::Node*>(this->node_pt(18)),static_cast<pyoomph::Node*>(this->node_pt(24))};}
     else if (face==1) { return {static_cast<pyoomph::Node*>(this->node_pt(2)),static_cast<pyoomph::Node*>(this->node_pt(8)),static_cast<pyoomph::Node*>(this->node_pt(20)),static_cast<pyoomph::Node*>(this->node_pt(26))};}
     else if (face==2) { return {static_cast<pyoomph::Node*>(this->node_pt(6)),static_cast<pyoomph::Node*>(this->node_pt(8)),static_cast<pyoomph::Node*>(this->node_pt(24)),static_cast<pyoomph::Node*>(this->node_pt(26))};}
     else if (face==3) { return {static_cast<pyoomph::Node*>(this->node_pt(18)),static_cast<pyoomph::Node*>(this->node_pt(20)),static_cast<pyoomph::Node*>(this->node_pt(24)),static_cast<pyoomph::Node*>(this->node_pt(26))};}
	 else throw_runtime_error("Invalid face index for brick element");
	}	

	std::vector<pyoomph::Node*> BulkElementTetra3dC1::get_vertex_nodes_of_face(const int &face) const
	{	  
	  if (face==0) { return {static_cast<pyoomph::Node*>(this->node_pt(1)),static_cast<pyoomph::Node*>(this->node_pt(2)),static_cast<pyoomph::Node*>(this->node_pt(3))};}
      else if (face==1) { return {static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(2)),static_cast<pyoomph::Node*>(this->node_pt(3))};}
      else if (face==2) { return {static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(1)),static_cast<pyoomph::Node*>(this->node_pt(3))};}
      else if (face==3) { return {static_cast<pyoomph::Node*>(this->node_pt(1)),static_cast<pyoomph::Node*>(this->node_pt(2)),static_cast<pyoomph::Node*>(this->node_pt(0))};}
	  else throw_runtime_error("Invalid face index for tetrahedral element");
	}

	std::vector<pyoomph::Node*> BulkElementTetra3dC2::get_vertex_nodes_of_face(const int &face) const
	{	  
	  if (face==0) { return {static_cast<pyoomph::Node*>(this->node_pt(1)),static_cast<pyoomph::Node*>(this->node_pt(2)),static_cast<pyoomph::Node*>(this->node_pt(3))};}
      else if (face==1) { return {static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(2)),static_cast<pyoomph::Node*>(this->node_pt(3))};}
      else if (face==2) { return {static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(1)),static_cast<pyoomph::Node*>(this->node_pt(3))};}
      else if (face==3) { return {static_cast<pyoomph::Node*>(this->node_pt(1)),static_cast<pyoomph::Node*>(this->node_pt(2)),static_cast<pyoomph::Node*>(this->node_pt(0))};}
	  else throw_runtime_error("Invalid face index for tetrahedral element");
	}

	std::vector<pyoomph::Node*> BulkElementWedge3dC1::get_vertex_nodes_of_face(const int &face) const
	{	  
	  if (face==0) { return {static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(1)),static_cast<pyoomph::Node*>(this->node_pt(2))};}
      else if (face==1) { return {static_cast<pyoomph::Node*>(this->node_pt(3)),static_cast<pyoomph::Node*>(this->node_pt(4)),static_cast<pyoomph::Node*>(this->node_pt(5))};}
      else if (face==2) { return {static_cast<pyoomph::Node*>(this->node_pt(3)),static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(5)),static_cast<pyoomph::Node*>(this->node_pt(2))};}
      else if (face==3) { return {static_cast<pyoomph::Node*>(this->node_pt(1)),static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(4)),static_cast<pyoomph::Node*>(this->node_pt(3))};}
      else if (face==4) { return {static_cast<pyoomph::Node*>(this->node_pt(1)),static_cast<pyoomph::Node*>(this->node_pt(4)),static_cast<pyoomph::Node*>(this->node_pt(2)),static_cast<pyoomph::Node*>(this->node_pt(5))};}
	  else throw_runtime_error("Invalid face index for wedge element");
	}

	std::vector<pyoomph::Node*> BulkElementPyramid3dC1::get_vertex_nodes_of_face(const int &face) const
	{
		if (face==0) { return {static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(1)),static_cast<pyoomph::Node*>(this->node_pt(4))};}
      else if (face==1) { return {static_cast<pyoomph::Node*>(this->node_pt(1)),static_cast<pyoomph::Node*>(this->node_pt(2)),static_cast<pyoomph::Node*>(this->node_pt(4))};}
      else if (face==2) { return {static_cast<pyoomph::Node*>(this->node_pt(2)),static_cast<pyoomph::Node*>(this->node_pt(3)),static_cast<pyoomph::Node*>(this->node_pt(4))};}
      else if (face==3) { return {static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(4)),static_cast<pyoomph::Node*>(this->node_pt(3))};}
      else if (face==4) { return {static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(3)),static_cast<pyoomph::Node*>(this->node_pt(1)),static_cast<pyoomph::Node*>(this->node_pt(2))};}
	  else throw_runtime_error("Invalid face index for pyramid element");
	}

	std::vector<pyoomph::Node*> BulkElementWedge3dC2::get_vertex_nodes_of_face(const int &face) const
	{
		if      (face==0) { return {static_cast<pyoomph::Node*>(this->node_pt(0)),
									static_cast<pyoomph::Node*>(this->node_pt(1)),
									static_cast<pyoomph::Node*>(this->node_pt(2))}; }
		else if (face==1) { return {static_cast<pyoomph::Node*>(this->node_pt(12)),
									static_cast<pyoomph::Node*>(this->node_pt(13)),
									static_cast<pyoomph::Node*>(this->node_pt(14))}; }
		else if (face==2) { return {static_cast<pyoomph::Node*>(this->node_pt(12)),
									static_cast<pyoomph::Node*>(this->node_pt(0)),
									static_cast<pyoomph::Node*>(this->node_pt(14)),
									static_cast<pyoomph::Node*>(this->node_pt(2))}; }
		else if (face==3) { return {static_cast<pyoomph::Node*>(this->node_pt(1)),
									static_cast<pyoomph::Node*>(this->node_pt(0)),
									static_cast<pyoomph::Node*>(this->node_pt(13)),
									static_cast<pyoomph::Node*>(this->node_pt(12))}; }
		else if (face==4) { return {static_cast<pyoomph::Node*>(this->node_pt(1)),
									static_cast<pyoomph::Node*>(this->node_pt(13)),
									static_cast<pyoomph::Node*>(this->node_pt(2)),
									static_cast<pyoomph::Node*>(this->node_pt(14))}; }
		else throw_runtime_error("Invalid face index for wedge element");
  	}

	std::vector<pyoomph::Node*> BulkElementPyramid3dC2::get_vertex_nodes_of_face(const int &face) const
	{
	  if (face==0) { return {static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(1)),static_cast<pyoomph::Node*>(this->node_pt(4))};}
      else if (face==1) { return {static_cast<pyoomph::Node*>(this->node_pt(1)),static_cast<pyoomph::Node*>(this->node_pt(2)),static_cast<pyoomph::Node*>(this->node_pt(4))};}
      else if (face==2) { return {static_cast<pyoomph::Node*>(this->node_pt(2)),static_cast<pyoomph::Node*>(this->node_pt(3)),static_cast<pyoomph::Node*>(this->node_pt(4))};}
      else if (face==3) { return {static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(4)),static_cast<pyoomph::Node*>(this->node_pt(3))};}
      else if (face==4) { return {static_cast<pyoomph::Node*>(this->node_pt(0)),static_cast<pyoomph::Node*>(this->node_pt(3)),static_cast<pyoomph::Node*>(this->node_pt(1)),static_cast<pyoomph::Node*>(this->node_pt(2))};}
	  else throw_runtime_error("Invalid face index for pyramid element");
	}

	oomph::FaceElement * BulkElementBrick3dC1::construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) { return new InterfaceElementQuad2dC1(jitcode, this, face_index); }
	oomph::FaceElement * BulkElementBrick3dC2::construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) { return new InterfaceElementQuad2dC2(jitcode, this, face_index); }

	oomph::FaceElement * BulkElementTetra3dC1::construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) { return new InterfaceElementTri2dC1(jitcode, this, face_index); }	
	oomph::FaceElement * BulkElementTetra3dC2::construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) { return new InterfaceElementTri2dC2(jitcode, this, face_index); }	
	oomph::FaceElement * BulkElementTetra3dC2TB::construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) { return new InterfaceElementTri2dC2TB(jitcode, this, face_index); }

	// Faces 0 and 1 of the wedge are the two triangular end-caps, faces 2-4 are the three quadrilateral sides.
	oomph::FaceElement * BulkElementWedge3dC1::construct_face_element(DynamicBulkElementInstance *jitcode, int face_index)
	{
		if (face_index<2) return  new InterfaceElementTri2dC1(jitcode, this, face_index);
		else return new InterfaceElementQuad2dC1(jitcode, this, face_index);
	}

	// Face 4 of the pyramid is the quadrilateral base, faces 0-3 are the four triangular sides.
	oomph::FaceElement * BulkElementPyramid3dC1::construct_face_element(DynamicBulkElementInstance *jitcode, int face_index)
	{
		if (face_index==4) return  new InterfaceElementQuad2dC1(jitcode, this, face_index);
		else return new InterfaceElementTri2dC1(jitcode, this, face_index);
	}
	
	oomph::FaceElement * BulkElementWedge3dC2::construct_face_element(DynamicBulkElementInstance *jitcode, int face_index)
	{
		if (face_index<2) return  new InterfaceElementTri2dC2(jitcode, this, face_index);
	    else return new InterfaceElementQuad2dC2(jitcode, this, face_index);					
	}

	oomph::FaceElement * BulkElementPyramid3dC2::construct_face_element(DynamicBulkElementInstance *jitcode, int face_index)
	{
		if (face_index==4) return  new InterfaceElementQuad2dC2(jitcode, this, face_index);
	    else return new InterfaceElementTri2dC2(jitcode, this, face_index);
	}
}
