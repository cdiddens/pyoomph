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


// Concrete 0-d and 1-d bulk elements: the ODE element, the point element, and the C1/C2 line
// elements in both their Q and T flavours. Shape functions, numpy/outline export and the static
// space-index tables; everything generic lives in the BulkElementBase files.

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

	////////////////////////////////

	// BulkElementODE0d represents a plain "ODE" element: no spatial mesh/nodes at all, only D0
	// (element-constant) internal-data fields evolving in time -- used for globally-defined
	// ODEs/scalar quantities that are not associated with any spatial field.
	oomph::PointIntegral BulkElementODE0d::Default_integration_scheme;

	// Sets up a single dummy "node" purely so the generic shape-buffer machinery has something to
	// allocate against; the element has no actual continuous or DL spaces (nnode_of_space and
	// nnode_DL are forced to 0), only D0 fields.
	BulkElementODE0d::BulkElementODE0d(DynamicJITCode *jit_code, oomph::TimeStepper *tstepper) : timestepper(tstepper)
	{
		//std::cout << "CONSTRUCT BULK ODE 0D " << this << std::endl;
		this->jitcode = jit_code;
		eleminfo.elem_ptr = static_cast<BulkElementBase *>(this);
		eleminfo.nnode = 1; // One dummy node... Necessary to create the buffers
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 0;
		eleminfo.nnode_DL = 0;
		eleminfo.nodal_dim = jitcode->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		this->set_integration_scheme(&Default_integration_scheme);
		allocate_discontinous_fields();
		for (unsigned int i = 0; i < this->ninternal_data(); i++)
		{
			this->internal_data_pt(i)->set_time_stepper(timestepper, true);
		}
	}

	BulkElementODE0d::~BulkElementODE0d()
	{
		//std::cout << "DESTRUCT BULK ODE 0D " << this << "  " << dynamic_cast<oomph::GeneralisedElement*>(this) << std::endl;
	}

	// Copies the current D0 field values into a flat buffer for export to numpy.
	void BulkElementODE0d::to_numpy(double *dest)
	{
		unsigned nD0 = jitcode->get_func_table()->info_D0.numfields;
		for (unsigned int i = 0; i < nD0; i++)
			dest[i] = this->internal_data_pt(i)->value(0); // TODO Scaling
	}

	// Trivial: a 0-d element has no spatial extent, so the Jacobian/JLagr are always 1.
	double BulkElementODE0d::fill_shape_info_at_s(const oomph::Vector<double> &, const unsigned int &, const JITFuncSpec_RequiredShapes_FiniteElement_t &, JITShapeInfo_t *, double &JLagr, unsigned int, oomph::DenseMatrix<double> *,unsigned) const
	{
		JLagr = 1.0;
		return 1.0;
	}

	////////////////////////////////

	// PointElement0d: a genuine single-node 0-d spatial element (as opposed to BulkElementODE0d),
	// used e.g. as a point source/sink or a 0-d boundary of a 1-d mesh. All field spaces
	// (C1/C1TB/C2/C2TB/DL) degenerate to a single shape function that is identically 1.
	PointElement0d::PointElement0d()
	{
		eleminfo.elem_ptr = static_cast<BulkElementBase *>(this);
		eleminfo.nnode = 1;
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 1;
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = 1;
		eleminfo.nnode_of_space[SPACE_INDEX_C2] = 1;
		eleminfo.nnode_of_space[SPACE_INDEX_C2TB] = 1;
		eleminfo.nnode_DL = 1;
		eleminfo.nodal_dim = jitcode->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}



	double PointElement0d::invert_jacobian_mapping(const oomph::DenseMatrix<double> &, oomph::DenseMatrix<double> &) const
	{
		return 1.0;
	}

	void PointElement0d::dshape_local(const oomph::Vector<double> &, oomph::Shape &psi, oomph::DShape &dpsids) const
	{
		psi[0] = 1;
		dpsids(0, 0) = 0;
	}

	void PointElement0d::shape_at_s_DL(const oomph::Vector<double> &, oomph::Shape &psi) const
	{
		psi[0] = 1.0;
	}

	void PointElement0d::dshape_local_at_s_DL(const oomph::Vector<double> &, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = 1.0;
		dpsi(0, 0) = 0.0;
	}

	void PointElement0d::shape_at_s_C1(const oomph::Vector<double> &, oomph::Shape &psi) const
	{
		psi[0] = 1.0;
	}
	void PointElement0d::shape_at_s_C2(const oomph::Vector<double> &, oomph::Shape &psi) const
	{
		psi[0] = 1.0;
	}

	void PointElement0d::dshape_local_at_s_C1(const oomph::Vector<double> &, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = 1.0;
		dpsi(0, 0) = 0;
	}

	void PointElement0d::dshape_local_at_s_C2(const oomph::Vector<double> &, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = 1.0;
		dpsi(0, 0) = 0;
	}

	// Writes the local node indices of sub-element `isubelem` (an element may be tesselated into
	// several simple sub-elements/triangles for numpy/plotting export) into `indices`; for a
	// single-node point element there is only one "sub-element" consisting of node 0.
	void PointElement0d::fill_element_nodal_indices_for_numpy(int *indices, unsigned, bool, std::vector<std::vector<std::set<oomph::Node *>>> &) const
	{
	  indices[0]=0;
	}

	// Returns the element's boundary polygon/outline coordinates (Eulerian, or Lagrangian if
	// requested) for plotting; a point element's "outline" is just its single node's position.
	std::vector<double> PointElement0d::get_outline(bool lagrangian)
	{
		std::vector<double> res(this->nodal_dimension());
		// unsigned offs=0;
		for (unsigned int i = 0; i < this->nodal_dimension(); i++)
		{
		   if (lagrangian) res[i] = static_cast<oomph::SolidNode*>(this->node_pt(0))->xi(i);
			else res[i] = this->node_pt(0)->x(i);
		}
		return res;
	}

	///////////////////////

	// BulkElementLine1dC1: linear (2-node) 1-d line element; the C1/C1TB spaces coincide with the
	// nodal (Q1) space, and DL uses the same linear shape functions on its own discontinuous copy.
	BulkElementLine1dC1::BulkElementLine1dC1()
	{
		eleminfo.elem_ptr = static_cast<BulkElementBase *>(this);
		eleminfo.nnode = 2;
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = 2;
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 2;
		eleminfo.nnode_DL = 2;
		eleminfo.nodal_dim = jitcode->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}

	void BulkElementLine1dC1::shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
	}

	void BulkElementLine1dC1::dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		dpsi(0, 0) = 0.0;
		dpsi(1, 0) = 1.0;
	}

	void BulkElementLine1dC1::fill_element_nodal_indices_for_numpy(int *indices, unsigned, bool, std::vector<std::vector<std::set<oomph::Node *>>> &) const
	{
		indices[0] = 0;
		indices[1] = 1;
	}

	std::vector<double> BulkElementLine1dC1::get_outline(bool lagrangian)
	{
		std::vector<double> res(2 * this->nodal_dimension());
		unsigned offs = 0;
		for (unsigned int i = 0; i < this->nodal_dimension(); i++)
		{
		   if (lagrangian)
		   {
			  res[0 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(0))->xi(i);
  			  res[1 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(1))->xi(i);		   
		   }
		   else
		   {
			  res[0 + offs] = this->node_pt(0)->x(i);
  			  res[1 + offs] = this->node_pt(1)->x(i);
  			}
			offs += 2;
		}
		return res;
	}

	
	
	// Maps node l's local coordinate within this (son) element to the corresponding local
	// coordinate in the father element, based on which half [-1,0] ("L") or [0,1] ("R") of the
	// father this son occupies during binary-tree refinement; used by further_build()/
	// rebuild_from_sons() to sample the father's/sons' fields at coincident points.
	void BulkElementLine1dC1::get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather)
	{
		using namespace oomph::BinaryTreeNames;
		sfather.resize(1, 0.0);
		int son_type = Tree_pt->son_type();

		oomph::Vector<double> s_lo(1);
		oomph::Vector<double> s_hi(1);
		oomph::Vector<double> s(1);
		oomph::Vector<double> x(1);
		switch (son_type)
		{
		case L:
			s_lo[0] = -1.0;
			s_hi[0] = 0.0;
			break;

		case R:
			s_lo[0] = 0.0;
			s_hi[0] = 1.0;
			break;

		}

		//   unsigned jnod=0;
		oomph::Vector<double> x_small(1);
		oomph::Vector<double> x_large(1);

		oomph::Vector<double> s_fraction(1);
//		unsigned n_p = nnode_1d();
		s_fraction[0] = local_one_d_fraction_of_node(l, 0);
		sfather[0] = s_lo[0] + (s_hi[0] - s_lo[0]) * s_fraction[0];
	}


	////////////////////////////



	// BulkElementLine1dC2: quadratic (3-node) 1-d line element; C2/C2TB use all 3 nodes
	// (quadratic Lagrange), while C1/C1TB only use the 2 end nodes (linear), and DL is a
	// discontinuous linear copy living on its own 2 "nodes".
	BulkElementLine1dC2::BulkElementLine1dC2()
	{
		eleminfo.elem_ptr = static_cast<BulkElementBase *>(this);
		eleminfo.nnode = 3;
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 2;
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = 2;
		eleminfo.nnode_of_space[SPACE_INDEX_C2] = 3;
		eleminfo.nnode_of_space[SPACE_INDEX_C2TB] = 3;
		eleminfo.nnode_DL = 2;
		eleminfo.nodal_dim = jitcode->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}

    void BulkElementLine1dC2::get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather)
	{
		using namespace oomph::BinaryTreeNames;
		sfather.resize(1, 0.0);
		int son_type = Tree_pt->son_type();

		oomph::Vector<double> s_lo(1);
		oomph::Vector<double> s_hi(1);
		oomph::Vector<double> s(1);
		oomph::Vector<double> x(1);
		switch (son_type)
		{
		case L:
			s_lo[0] = -1.0;
			s_hi[0] = 0.0;
			break;

		case R:
			s_lo[0] = 0.0;
			s_hi[0] = 1.0;
			break;

		}

		//   unsigned jnod=0;
		oomph::Vector<double> x_small(1);
		oomph::Vector<double> x_large(1);

		oomph::Vector<double> s_fraction(1);
//		unsigned n_p = nnode_1d();
		s_fraction[0] = local_one_d_fraction_of_node(l, 0);
		sfather[0] = s_lo[0] + (s_hi[0] - s_lo[0]) * s_fraction[0];
	}
	
	void BulkElementLine1dC2::shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		oomph::OneDimLagrange::shape<2>(s[0], &(psi[0]));
	}

	void BulkElementLine1dC2::dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		oomph::OneDimLagrange::shape<2>(s[0], &(psi[0]));
		double DPsi[2];
		oomph::OneDimLagrange::dshape<2>(s[0], DPsi);
		dpsi(0, 0) = DPsi[0];
		dpsi(1, 0) = DPsi[1];
	}

	void BulkElementLine1dC2::shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
	}

	void BulkElementLine1dC2::dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = 1.0;
		psi[1] = s[0];
		dpsi(0, 0) = 0.0;
		dpsi(1, 0) = 1.0;
	}

	void BulkElementLine1dC2::fill_element_nodal_indices_for_numpy(int *indices, unsigned, bool, std::vector<std::vector<std::set<oomph::Node *>>> &) const
	{
		indices[0] = 0;
		indices[1] = 1;
		indices[2] = 2;
	}

	std::vector<double> BulkElementLine1dC2::get_outline(bool lagrangian)
	{
		std::vector<double> res(3 * this->nodal_dimension());
		unsigned offs = 0;
		for (unsigned int i = 0; i < this->nodal_dimension(); i++)
		{
		   if (lagrangian)
		   {
			res[0 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(0))->xi(i);
			res[1 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(1))->xi(i); // TODO: Check
			res[2 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(2))->xi(i);		   
		   }
		   else
		   {		   
			res[0 + offs] = this->node_pt(0)->x(i);
			res[1 + offs] = this->node_pt(1)->x(i); // TODO: Check
			res[2 + offs] = this->node_pt(2)->x(i);
			}
			offs += 3;
		}
		return res;
	}

	

	////////////////////////////

	// BulkTElementLine1dC1: 1-d line element using the "T" (simplex-style) local coordinate
	// convention s in [0,1] instead of [-1,1] -- these are used as the 1-d edges of triangular/
	// tetrahedral elements. Otherwise the same linear (2-node) element as BulkElementLine1dC1.
	BulkTElementLine1dC1::BulkTElementLine1dC1()
	{
		eleminfo.elem_ptr = static_cast<BulkElementBase *>(this);
		eleminfo.nnode = 2;
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = 2;
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 2;
		eleminfo.nnode_DL = 2;
		eleminfo.nodal_dim = jitcode->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}

	void BulkTElementLine1dC1::shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		psi[0] = 1.0;
		psi[1] = 2 * s[0] - 1;
	}

	void BulkTElementLine1dC1::dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = 1.0;
		psi[1] = 2 * s[0] - 1;
		dpsi(0, 0) = 0.0;
		dpsi(1, 0) = 2.0;
	}

	void BulkTElementLine1dC1::fill_element_nodal_indices_for_numpy(int *indices, unsigned, bool, std::vector<std::vector<std::set<oomph::Node *>>> &) const
	{
		indices[0] = 0;
		indices[1] = 1;
	}

	std::vector<double> BulkTElementLine1dC1::get_outline(bool lagrangian)
	{
		std::vector<double> res(2 * this->nodal_dimension());
		unsigned offs = 0;
		for (unsigned int i = 0; i < this->nodal_dimension(); i++)
		{
		   if (lagrangian)
		   {
			res[0 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(0))->xi(i);
			res[1 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(1))->xi(i);		   
		   }
		   else
		   {
			res[0 + offs] = this->node_pt(0)->x(i);
			res[1 + offs] = this->node_pt(1)->x(i);
			}
			offs += 2;
		}
		return res;
	}

	

	// BulkTElementLine1dC2: quadratic version of BulkTElementLine1dC1, using the T (simplex, s in
	// [0,1]) convention -- the 1-d edge element of quadratic triangular/tetrahedral meshes.
	BulkTElementLine1dC2::BulkTElementLine1dC2()
	{
		eleminfo.elem_ptr = static_cast<BulkElementBase *>(this);
		eleminfo.nnode = 3;
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 2;
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = 2;		
		eleminfo.nnode_of_space[SPACE_INDEX_C2] = 3;
		eleminfo.nnode_of_space[SPACE_INDEX_C2TB] = 3;
		eleminfo.nnode_DL = 2;
		eleminfo.nodal_dim = jitcode->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}

	void BulkTElementLine1dC2::shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		psi[0] = 1.0 - s[0];
		psi[1] = s[0];
	}

	void BulkTElementLine1dC2::dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = 1.0 - s[0];
		psi[1] = s[0];
		dpsi(0, 0) = -1.0;
		dpsi(1, 0) = 1.0;
	}

	void BulkTElementLine1dC2::shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		psi[0] = 1.0;		   // TODO: This good?
		psi[1] = 2 * s[0] - 1; // TODO: This good?
	}

	void BulkTElementLine1dC2::dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = 1.0;		   // TODO: This good?
		psi[1] = 2 * s[0] - 1; // TODO: This good?
		dpsi(0, 0) = 0.0;	   // TODO: This good?
		dpsi(1, 0) = 2.0;	   // TODO: This good?
	}

	void BulkTElementLine1dC2::fill_element_nodal_indices_for_numpy(int *indices, unsigned, bool, std::vector<std::vector<std::set<oomph::Node *>>> &) const
	{
		indices[0] = 0;
		indices[1] = 1;
		indices[2] = 2;
	}

	std::vector<double> BulkTElementLine1dC2::get_outline(bool lagrangian)
	{
		std::vector<double> res(3 * this->nodal_dimension());
		unsigned offs = 0;
		for (unsigned int i = 0; i < this->nodal_dimension(); i++)
		{
		   if (lagrangian)
		   {
			res[0 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(0))->xi(i);
			res[1 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(1))->xi(i); // TODO: Check
			res[2 + offs] = static_cast<oomph::SolidNode*>(this->node_pt(2))->xi(i);
			}
			else
			{
			res[0 + offs] = this->node_pt(0)->x(i);
			res[1 + offs] = this->node_pt(1)->x(i); // TODO: Check
			res[2 + offs] = this->node_pt(2)->x(i);
			}
			
			offs += 3;
		}
		return res;
	}

	const std::vector<std::vector<unsigned>> BulkElementLine1dC1::Nodal_Space_Index_To_Element_Index_Map={
		{}, // C2TB 
		{}, // C2
		{0,1}, // C1TB
		{0,1}  // C1
	};

	const std::vector<std::vector<unsigned>> BulkElementLine1dC2::Nodal_Space_Index_To_Element_Index_Map={
		{0,1,2}, // C2TB 
		{0,1,2}, // C2
		{0,2}, // C1TB
		{0,2}  // C1
	};

	const std::vector<std::vector<unsigned>> BulkTElementLine1dC1::Nodal_Space_Index_To_Element_Index_Map={
		{}, // C2TB 
		{}, // C2
		{0,1}, // C1TB
		{0,1}  // C1
	};

	const std::vector<std::vector<unsigned>> BulkTElementLine1dC2::Nodal_Space_Index_To_Element_Index_Map={
		{0,1,2}, // C2TB 
		{0,1,2}, // C2
		{0,2}, // C1TB
		{0,2}  // C1
	};

	const std::vector<std::vector<unsigned>> BulkElementODE0d::Nodal_Space_Index_To_Element_Index_Map={
		{}, // C2TB 
		{}, // C2
		{}, // C1TB
		{}  // C1
	};

	const std::vector<std::vector<unsigned>> PointElement0d::Nodal_Space_Index_To_Element_Index_Map={
		{0}, // C2TB 
		{0}, // C2
		{0}, // C1TB
		{0}  // C1
	};



	const std::vector<std::vector<int>> BulkElementLine1dC1::Element_Index_To_Nodal_Space_Index_Map={
		{}, // C2TB
		{}, // C2
		{0,1}, // C1TB
		{0,1}  // C1
	};

	const std::vector<std::vector<int>> BulkElementLine1dC2::Element_Index_To_Nodal_Space_Index_Map={
		{0,1,2}, // C2TB
		{0,1,2}, // C2
		{0,-1,1}, // C1TB
		{0,-1,1}  // C1
	};

	const std::vector<std::vector<int>> BulkTElementLine1dC1::Element_Index_To_Nodal_Space_Index_Map={
		{}, // C2TB
		{}, // C2
		{0,1}, // C1TB
		{0,1}  // C1
	};

	const std::vector<std::vector<int>> BulkTElementLine1dC2::Element_Index_To_Nodal_Space_Index_Map={
		{0,1,2}, // C2TB
		{0,1,2}, // C2
		{0,-1,1}, // C1TB
		{0,-1,1}  // C1
	};
	
	const std::vector<std::vector<int>> BulkElementODE0d::Element_Index_To_Nodal_Space_Index_Map={
		{}, // C2TB
		{}, // C2
		{}, // C1TB
		{}  // C1
	};


	const std::vector<std::vector<int>> PointElement0d::Element_Index_To_Nodal_Space_Index_Map={
		{0}, // C2TB
		{0}, // C2
		{0}, // C1TB
		{0}  // C1
	};

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Non_Vertex_Node_Indices[class]: the element-local node indices i for which
	// Element_Index_To_Nodal_Space_Index_Map[3][i] (the C1/vertex space entry) is -1, i.e. the nodes
	// that do not carry a value of the linear (vertex) field space.
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	const std::vector<unsigned> BulkElementLine1dC1::Non_Vertex_Node_Indices={};
	const std::vector<unsigned> BulkElementLine1dC2::Non_Vertex_Node_Indices={1};
	const std::vector<unsigned> BulkTElementLine1dC1::Non_Vertex_Node_Indices={};
	const std::vector<unsigned> BulkTElementLine1dC2::Non_Vertex_Node_Indices={1};
	const std::vector<unsigned> BulkElementODE0d::Non_Vertex_Node_Indices={};
	const std::vector<unsigned> PointElement0d::Non_Vertex_Node_Indices={};
	const std::vector<int> BulkElementODE0d::Possible_Face_Indices={};
	const std::vector<int> PointElement0d::Possible_Face_Indices={};
	const std::vector<int> BulkElementLine1dC1::Possible_Face_Indices={-1,1};
	const std::vector<int> BulkElementLine1dC2::Possible_Face_Indices={-1,1};
	
	const std::vector<int> BulkTElementLine1dC1::Possible_Face_Indices={-1,1};
	const std::vector<int> BulkTElementLine1dC2::Possible_Face_Indices={-1,1};

	// get_vertex_nodes_of_face() implementations below: for each element type and each valid face index
	// (see that type's Possible_Face_Indices table above), return the element's *corner/vertex* nodes
	// bounding that face, in a fixed order (used e.g. to build the face's outline or to identify it geometrically).
	std::vector<pyoomph::Node*> BulkElementLine1dC1::get_vertex_nodes_of_face(const int &face) const
	{
      if (face==-1) return {static_cast<pyoomph::Node*>(this->node_pt(0))};
	  else if (face==1) return {static_cast<pyoomph::Node*>(this->node_pt(1))};	  
	  else throw_runtime_error("Invalid face index for line element");
	}

	std::vector<pyoomph::Node*> BulkElementLine1dC2::get_vertex_nodes_of_face(const int &face) const
	{
	  if (face==-1) return {static_cast<pyoomph::Node*>(this->node_pt(0))};
	  else if (face==1) return {static_cast<pyoomph::Node*>(this->node_pt(2))};
	  else throw_runtime_error("Invalid face index for line element");
	}

	std::vector<pyoomph::Node*> BulkTElementLine1dC1::get_vertex_nodes_of_face(const int &face) const
	{
	  if (face==-1) return {static_cast<pyoomph::Node*>(this->node_pt(0))};
	  else if (face==1) return {static_cast<pyoomph::Node*>(this->node_pt(1))};	  
	  else throw_runtime_error("Invalid face index for line element");
	}

	std::vector<pyoomph::Node*> BulkTElementLine1dC2::get_vertex_nodes_of_face(const int &face) const
	{
	  if (face==-1) return {static_cast<pyoomph::Node*>(this->node_pt(0))};
	  else if (face==1) return {static_cast<pyoomph::Node*>(this->node_pt(2))};
	  else throw_runtime_error("Invalid face index for line element");
	}

	oomph::FaceElement * BulkElementLine1dC1::construct_face_element(DynamicJITCode *interface_jitcode, int face_index) { return new InterfaceElementPoint0d(interface_jitcode, this, face_index); }
	oomph::FaceElement * BulkElementLine1dC2::construct_face_element(DynamicJITCode *interface_jitcode, int face_index) { return new InterfaceElementPoint0d(interface_jitcode, this, face_index); }
	oomph::FaceElement * BulkTElementLine1dC1::construct_face_element(DynamicJITCode *interface_jitcode, int face_index) { return new InterfaceElementPoint0d(interface_jitcode, this, face_index); }
	oomph::FaceElement * BulkTElementLine1dC2::construct_face_element(DynamicJITCode *interface_jitcode, int face_index) { return new InterfaceElementPoint0d(interface_jitcode, this, face_index); }
}
