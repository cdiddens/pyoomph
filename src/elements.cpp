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


#include "elements.hpp"
#include "exception.hpp"
#include "problem.hpp"
#include "nodes.hpp"
#include "meshtemplate.hpp"
#include "expressions.hpp"
#include "thirdparty/delaunator.hpp"
#include "timestepper.hpp"

namespace pyoomph
{
	BulkElementBase *_currently_assembled_element = NULL;
}

extern "C"
{
	double _pyoomph_get_element_size(void *elem_ptr)
	{
		throw_runtime_error("Element size will get problems with the casting");
		pyoomph::BulkElementBase *elem = (pyoomph::BulkElementBase *)elem_ptr;
		return elem->get_element_diam();
	}

	double _pyoomph_invoke_callback(void *functab, int jitindex, double *args, int numargs)
	{
		JITFuncSpec_Table_FiniteElement_t *ft = (JITFuncSpec_Table_FiniteElement_t *)functab;
		pyoomph::CustomMathExpressionBase *expr = (pyoomph::CustomMathExpressionBase *)ft->callback_infos[jitindex].cb_obj; // TODO: This may not be multiple inherited!!
		return expr->_call(args, numargs);
	}
	
	void _pyoomph_invoke_multi_ret(void * functab, int jitindex,int flag,double * arg_list,double * result_list, double * derivative_matrix, int numargs, int numret)
	{
		JITFuncSpec_Table_FiniteElement_t *ft = (JITFuncSpec_Table_FiniteElement_t *)functab;
		pyoomph::CustomMultiReturnExpressionBase *expr = (pyoomph::CustomMultiReturnExpressionBase *)ft->multi_ret_infos[jitindex].cb_obj; // TODO: This may not be multiple inherited!!		
		expr->_call(flag,arg_list, numargs,result_list,numret,derivative_matrix);
	}	

	void _pyoomph_fill_shape_buffer_for_point(unsigned index, JITFuncSpec_RequiredShapes_FiniteElement_t *required, int flag)
	{
		pyoomph::_currently_assembled_element->fill_shape_buffer_for_integration_point(index, *required, flag);
	}
}

namespace pyoomph
{
	double *__replace_RJM_by_param_deriv = NULL;

	size_t __shape_buffer_mem_usage = 0;
	void *counted_calloc(size_t num, size_t size)
	{
		__shape_buffer_mem_usage += num * size;
		return calloc(num, size);
	}

	// my_alloc/my_free/my_alloc_or_free: recursive helpers to allocate (or free) a nested,
	// variable-depth C array (used for the multi-dimensional shape-function buffers inside
	// JITShapeInfo_t, whose depth depends on which derivatives/spaces the generated code needs).
	// The scalar overload is the recursion base case (nothing to do for a non-pointer leaf type).
	template <typename T>
	void my_alloc(T dest) {}

	// Allocates a firstdim-sized array of T, then recurses on each of its elements with the
	// remaining "extra" dimensions (if any), building up a jagged/nested array of the requested depth.
	template <typename T, typename... Extra>
	void my_alloc(T * PYOOMPH_RESTRICT &  dest, size_t firstdim, const Extra &...extra)
	{
		if (!firstdim)
		{
			dest = NULL;
			return;
		}
		dest = (T *)counted_calloc(firstdim, sizeof(T));
		constexpr size_t remaining = sizeof...(extra);
		if (remaining > 0)
		{
			for (size_t c = 0; c < firstdim; c++)
			{
				my_alloc(dest[c], extra...);
			}
		}
	}

	template <typename T>
	void my_free(T dest) {}

	// Mirror of my_alloc: recursively frees a nested array previously built by my_alloc, then frees
	// the top-level pointer itself and nulls it out.
	template <typename T, typename... Extra>
	void my_free(T * PYOOMPH_RESTRICT &  dest, size_t firstdim, const Extra &...extra)
	{
		if (!dest)
			return;
		constexpr size_t remaining = sizeof...(extra);
		if (remaining > 0)
		{
			for (size_t c = 0; c < firstdim; c++)
			{
				my_free(dest[c], extra...);
			}
		}
		free(dest);
		dest = NULL;
	}

	// Convenience wrapper dispatching to my_alloc or my_free depending on "alloc", so call sites can
	// toggle allocation/deallocation of a shape buffer with a single boolean flag.
	template <typename T, typename... Extra>
	void my_alloc_or_free(bool alloc, T * PYOOMPH_RESTRICT &  dest, size_t firstdim, const Extra &...extra)
	{
		if (alloc)
			my_alloc(dest, firstdim, extra...);
		else
			my_free(dest, firstdim, extra...);
	}

	// Selects the per-shape integration-scheme map (see IntegrationSchemeStorage's member maps)
	// matching the requested (triangular/tetrahedral vs. quad/brick, spatial dimension, bubble-enriched) combination.
	std::map<unsigned, oomph::Integral *> &IntegrationSchemeStorage::get_integral_order_map(bool tri, unsigned edim, bool bubble)
	{
		if (tri)
		{
			if (bubble)
			{
				if (edim == 2)
					return T2dTB;
				else if (edim == 3)
					return T3dTB;
				else
					throw_runtime_error("Implement");
			}
			else
			{
				if (edim == 1)
					return T1d;
				else if (edim == 2)
					return T2d;
				else if (edim == 3)
					return T3d;
				else
					throw_runtime_error("Implement");
			}
		}
		else
		{
			if (edim == 1)
				return Q1d;
			else if (edim == 2)
				return Q2d;
			else if (edim == 3)
				return Q3d;
			else if (edim==4)
				return Wedge3d;
			else if (edim==5)
				return Pyramid3d;
			else
				throw_runtime_error("Implement");
		}
		throw_runtime_error("Invalid combination of tri/quad, edim and bubble");
	}

	// Deletes every oomph::Integral owned by "map" and clears it; called for each per-shape map from the destructor.
	void IntegrationSchemeStorage::clean_up_map(std::map<unsigned, oomph::Integral *> &map)
	{
		for (auto m : map)
		{
			delete m.second;
			m.second = NULL;
		}
		map.clear();
	}

	IntegrationSchemeStorage::~IntegrationSchemeStorage()
	{
		clean_up_map(Q1d);
		clean_up_map(Q2d);
		clean_up_map(Q3d);
		clean_up_map(T1d);
		clean_up_map(T2d);
		clean_up_map(T3d);
		clean_up_map(T2dTB);
		clean_up_map(T3dTB);
		clean_up_map(Wedge3d);
		clean_up_map(Pyramid3d);
	}

	// Pre-populates the integration-scheme maps with a fixed set of standard Gauss/TGauss/Wedge/Pyramid
	// quadrature rules at a handful of common orders; get_integration_scheme() falls back to the
	// closest available order above the requested one if an exact match isn't pre-built here.
	IntegrationSchemeStorage::IntegrationSchemeStorage()
	{
		Q1d[2] = new oomph::Gauss<1, 2>();
		Q1d[3] = new oomph::Gauss<1, 3>();
		Q1d[4] = new oomph::Gauss<1, 4>();

		Q2d[2] = new oomph::Gauss<2, 2>();
		Q2d[3] = new oomph::Gauss<2, 3>();
		Q2d[4] = new oomph::Gauss<2, 4>();

		Q3d[2] = new oomph::Gauss<3, 2>();
		Q3d[3] = new oomph::Gauss<3, 3>();
		Q3d[4] = new oomph::Gauss<3, 4>();

		T1d[2] = new oomph::TGauss<1, 2>();
		T1d[3] = new oomph::TGauss<1, 3>();
		T1d[4] = new oomph::TGauss<1, 4>();
		//   T1d[5]=new oomph::TGauss<1,5>(); // IS NOT IMPLEMETED!

		T2d[2] = new oomph::TGauss<2, 2>();
		T2d[3] = new oomph::TGauss<2, 3>();
		T2d[4] = new oomph::TGauss<2, 4>();
		//   T2d[5]=new oomph::TGauss<2,5>();  // Has the wrong weighting factor! element volume is twice as large!
		//  T2d[13]=new oomph::TGauss<2,13>(); // Has the wrong weighting factor! element volume is twice as large!

		T2dTB[3] = new oomph::TBubbleEnrichedGauss<2, 3>();

		T3d[2] = new oomph::TGauss<3, 2>();
		T3d[3] = new oomph::TGauss<3, 3>();
		T3d[5] = new oomph::TGauss<3, 5>();

		//TODO: Having a C1TB here?
		T3dTB[3] = new oomph::TBubbleEnrichedGauss<3, 3>();

		Wedge3d[2] = new oomph::WedgeGaussC1();
		Wedge3d[3] = new oomph::WedgeGaussC2();

		Pyramid3d[2]= new oomph::PyramidGaussC1();
	}

	// Returns the cached scheme for the exact requested order if available; otherwise picks the
	// scheme with the closest order strictly greater than requested (never a lower order, so
	// accuracy is not silently reduced), or the highest available order if none is greater.
	oomph::Integral *IntegrationSchemeStorage::get_integration_scheme(bool tris, unsigned edim, unsigned order, bool bubble)
	{
		std::map<unsigned, oomph::Integral *> &map = this->get_integral_order_map(tris, edim, bubble);
		if (map.count(order))
		{
			// std::cout << "FOR " << (tris ? "TRI" : "QUAD") << " OF DIM " << edim << " ORDER " << order << " WE HAVE " << typeid(map[order]).name();
			return map[order];
		}
		unsigned closestdist = 10000;
		oomph::Integral *ret = NULL;
		unsigned maxorder = 0;
		for (auto mentry : map)
		{
			maxorder = std::max(maxorder, mentry.first);
			if (order < mentry.first)
			{
				if (mentry.first - order < closestdist)
				{
					closestdist = mentry.first - order;
					ret = mentry.second;
				}
			}
		}

		if (ret)
		{
			return ret;
		}
		return map[maxorder];
	}

	JITShapeInfo_t *Default_shape_info_buffer = NULL;
	//JITShapeInfo_t *temp_shape_info_buffer = NULL;
	DynamicBulkElementInstance *BulkElementBase::__CurrentCodeInstance = NULL;
	unsigned BulkElementBase::zeta_time_history = 0;
	unsigned BulkElementBase::zeta_coordinate_type = 0; // 0 means Lagrangian, 1 Eulerian, on co-dimensional meshes it will be the boundary coordinate (if set)
	bool BulkElementBase::use_eigen_error_estimators=false;

	// Element shape-quality measure: ratio of the smallest Jacobian determinant encountered at any
	// integration point to the element's mean Jacobian (i.e. mean element size). A value close to 1
	// indicates a well-shaped, near-uniform element; values close to 0 indicate a nearly degenerate
	// (collapsed/inverted) element.
	double BulkElementBase::get_quality_factor()
	{
		double size = 0.0;
		double weightsum = 0.0;
		double minJ = 1e40;
		for (unsigned ipt = 0; ipt < integral_pt()->nweight(); ipt++)
		{
			oomph::Vector<double> s(this->dim());
			for (unsigned int i = 0; i < this->dim(); i++)
				s[i] = integral_pt()->knot(ipt, i);
			double J = this->J_eulerian(s);
			double w = integral_pt()->weight(ipt);
			weightsum += w;
			size += J * w;
			if (J < minJ)
			{
				minJ = J;
			}
		}		
		return minJ / (size / weightsum);
	}

	// Links this element's refinement tree root to "other"'s along direction mydir/otherdir as a
	// periodic neighbor, by dispatching to the appropriate oomph-lib tree type (QuadTree for 2d,
	// BinaryTree for 1d, OcTree for 3d) and translating pyoomph's signed-integer direction
	// convention (-1/1: W/E, -2/2: S/N, -3/3: D/U) into that tree type's named direction constants.
	// This makes oomph-lib's h-refinement treat the two boundaries as topologically adjacent for
	// hanging-node/neighbor-finding purposes, implementing periodic boundary conditions on refined meshes.
	void BulkElementBase::connect_periodic_tree(BulkElementBase *other, const int &mydir, const int &otherdir)
	{
		oomph::QuadTree *my_qt = dynamic_cast<oomph::QuadTree *>(Tree_pt);
		oomph::BinaryTree *my_bt = dynamic_cast<oomph::BinaryTree *>(Tree_pt);
		oomph::OcTree *my_ot = dynamic_cast<oomph::OcTree *>(Tree_pt);
		oomph::TreeRoot * myroot=NULL;
		oomph::TreeRoot * otherroot=NULL;
		int my_root_dir,other_root_dir;
		if (my_qt)
		{
			using namespace oomph::QuadTreeNames;
			oomph::QuadTree *other_qt = dynamic_cast<oomph::QuadTree *>(other->tree_pt());
			if (!other_qt) throw_runtime_error("Cannot connect a QuadTree with a non-QuadTree for a periodic boundary");
			myroot=my_qt->root_pt(); otherroot=other_qt->root_pt();
			if (mydir==-1) my_root_dir=W;
			else if (mydir==1) my_root_dir=E;
			else if (mydir==-2) my_root_dir=S;
			else if (mydir==2) my_root_dir=N;
			else throw_runtime_error("Invalid direction");
			if (otherdir==-1) other_root_dir=W;
			else if (otherdir==1) other_root_dir=E;
			else if (otherdir==-2) other_root_dir=S;
			else if (otherdir==2) other_root_dir=N;
			else throw_runtime_error("Invalid direction");						
		}
		else if (my_ot)
		{
			using namespace oomph::OcTreeNames;
			oomph::OcTree *other_ot = dynamic_cast<oomph::OcTree *>(other->tree_pt());			
			if (!other_ot) throw_runtime_error("Cannot connect a OcTree with a non-OcTree for a periodic boundary");			
			myroot=my_ot->root_pt(); otherroot=other_ot->root_pt();						
			if (mydir==-1) my_root_dir=L;
			else if (mydir==1) my_root_dir=R;
			else if (mydir==-2) my_root_dir=D;
			else if (mydir==2) my_root_dir=U;			
			else if (mydir==-3) my_root_dir=B;			
			else if (mydir==3) my_root_dir=F;			
			else throw_runtime_error("Invalid direction");
			if (otherdir==-1) other_root_dir=L;
			else if (otherdir==1) other_root_dir=R;
			else if (otherdir==-2) other_root_dir=D;
			else if (otherdir==2) other_root_dir=U;			
			else if (otherdir==-3) other_root_dir=B;			
			else if (otherdir==3) other_root_dir=F;			
			else throw_runtime_error("Invalid direction");			
		}
		else if (my_bt)
		{
			using namespace oomph::BinaryTreeNames;
			oomph::BinaryTree *other_bt = dynamic_cast<oomph::BinaryTree *>(other->tree_pt());
			if (!other_bt) throw_runtime_error("Cannot connect a BinaryTree with a non-BinaryTree for a periodic boundary");
			myroot=my_bt->root_pt(); otherroot=other_bt->root_pt();
			if (mydir==-1) my_root_dir=L;
			else if (mydir==1) my_root_dir=R;			
			else throw_runtime_error("Invalid direction");
			if (otherdir==-1) other_root_dir=L;
			else if (otherdir==1) other_root_dir=R;			
			else throw_runtime_error("Invalid direction");			
		}
		if (myroot && otherroot)
		{
			myroot->set_neighbour_periodic(my_root_dir);
			otherroot->set_neighbour_periodic(other_root_dir);
			myroot->neighbour_pt(my_root_dir)=otherroot;
			otherroot->neighbour_pt(other_root_dir)=myroot;
		}
		
		// Otherwise, we can't do anything
	}


	// For every local node l, if it is a hanging node (its Lagrangian/Eulerian position is
	// constrained by master nodes rather than free, as happens on non-conforming refined meshes),
	// records its master nodes' weights and their local equation numbers for the *position* degrees
	// of freedom into shape_info->hanginfo_Pos[l]. This lets the generated ALE/solid-mechanics code
	// correctly distribute a hanging node's position-Jacobian contributions to its masters. Returns
	// true if any node in the element is hanging.
	bool BulkElementBase::fill_hang_info_with_equations_for_pos(JITShapeInfo_t *shape_info)
	{
		bool res = false;
		for (unsigned int l = 0; l < eleminfo.nnode; l++)
		{
			if (node_pt(l)->is_hanging())
			{
				res = true;
				auto hang_info_pt = node_pt(l)->hanging_pt();				
				shape_info->hanginfo_Pos[l].nummaster = hang_info_pt->nmaster();
				for (unsigned m = 0; m < hang_info_pt->nmaster(); m++)
				{					
					shape_info->hanginfo_Pos[l].masters[m].weight = hang_info_pt->master_weight(m);				
					for (unsigned int f = 0; f < this->nodal_dimension(); f++)
					{						
						oomph::DenseMatrix<int> position_local_eqn_at_node = this->local_position_hang_eqn(hang_info_pt->master_node_pt(m));
						for (unsigned int f = 0; f < this->nodal_dimension(); f++) // TODO: More nnodal_position_type ?
						{
							shape_info->hanginfo_Pos[l].masters[m].local_eqn[f] = position_local_eqn_at_node(0, f);
						}
					}
				}
			}
			else
			{				
				shape_info->hanginfo_Pos[l].nummaster = 0;
			}
		}

		return res;
	}

	// Analogous to fill_hang_info_with_equations_for_pos, but for the ordinary field values ("base
	// bulk" fields, i.e. not the interface-only additional fields) of each continuous interpolation
	// space present on this element: for every node hanging in that space, records its masters'
	// weights and local equation numbers (per field) into shape_info->hanginfo_Cont[space]. Used by
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
			int hangindex=space_info->hangindex;
			JITHangInfo_t * hangbuffer=shape_info->hanginfo_Cont[space_info->space_index];
			for (unsigned int l = 0; l < nnode_space; l++)
			{
				const unsigned l_elem=space_node_to_elem_node[l];
				if (node_pt(l_elem)->is_hanging(hangindex))
				{
					res = true;
					auto hang_info_pt = node_pt(l_elem)->hanging_pt(hangindex);
					hangbuffer[l].nummaster = hang_info_pt->nmaster();				
					for (unsigned m = 0; m < hang_info_pt->nmaster(); m++)
					{
						hangbuffer[l].masters[m].weight = hang_info_pt->master_weight(m);					
						for (unsigned int f = 0; f < space_info->numfields_basebulk; f++)
						{
							hangbuffer[l].masters[m].local_eqn[f+space_info->buffer_offset_basebulk] = this->local_hang_eqn(hang_info_pt->master_node_pt(m), f+space_info->nodal_offset_basebulk);
						}
					}
				}
				else
				{
					hangbuffer[l].nummaster = 0;
				}
			}

		}
		return res;
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

		// First positional hanging
		for (unsigned l = 0; l < this->nnode(); l++)
		{
			pyoomph::Node *n = dynamic_cast<pyoomph::Node *>(this->node_pt(l));
			if (n->is_hanging())
			{
				for (unsigned i = 0; i < n->ndim(); i++)
				{					
					n->x(i) = n->position(i);  // position() considers the hanging
				}
			}
		}

		// Then going over the basebulk fields of all continuous spaces
		auto * ft=codeinst->get_func_table();
		const std::vector<std::vector<unsigned>> & space_node_to_elem_node_map=this->get_nodal_space_index_to_element_index_map();
		const std::vector<std::vector<std::vector<unsigned>>> & dummy_value_interpolation_map=this->get_dummy_value_interpolation_map();
		for (unsigned ispace=0;ispace<ft->num_present_continuous_spaces;ispace++)
		{
			const JITFuncSpec_Table_FiniteElement_SpaceInfo_t * space_info = ft->present_continuous_spaces[ispace];
			unsigned nnode_space=eleminfo.nnode_of_space[space_info->space_index];
			const std::vector<unsigned> & space_node_to_elem_node=space_node_to_elem_node_map[space_info->space_index];
			int hangindex=space_info->hangindex;
			for (unsigned l = 0; l < nnode_space; l++)
			{
				const unsigned l_elem=space_node_to_elem_node[l];
				pyoomph::Node *n = dynamic_cast<pyoomph::Node *>(this->node_pt(l_elem));
				unsigned ntstorage = node_pt(l_elem)->ntstorage();
				if (n->is_hanging(hangindex))
				{
					for (unsigned f = 0; f < space_info->numfields_basebulk; f++)
					{
						for (unsigned t = 0; t < ntstorage; t++)
						{
							node_pt(l_elem)->value_pt(space_info->nodal_offset_basebulk+f)[t] = node_pt(l_elem)->value(t, space_info->nodal_offset_basebulk+f);
						}						
					}
				}
			}
			// If you are e.g. on a C2 element, C1 fields have dummy values on the C2 only nodes. These are filled now by averaging the values of the C1 nodes. This is needed for hanging node interpolation, since the C2 nodes are not hanging, but the C1 nodes are.
			const std::vector<std::vector<unsigned>> & dummy_value_interpolation=dummy_value_interpolation_map[space_info->space_index];
			if (dummy_value_interpolation.size()>0)
			{
				unsigned ntstorage = node_pt(0)->ntstorage();
				for (const std::vector<unsigned> & interp : dummy_value_interpolation)
				{															
					for (unsigned f = 0; f < space_info->numfields_basebulk; f++)
					{
						for (unsigned t = 0; t < ntstorage; t++)
						{
							double val=0.0;
							for (unsigned m=1;m<interp.size();m++)
							{								
								val+=node_pt(interp[m])->value(t, space_info->nodal_offset_basebulk+f);
							}
							node_pt(interp[0])->value_pt(space_info->nodal_offset_basebulk+f)[t] = val/(double)(interp.size()-1.0);
						}						
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
		for (unsigned int l = 0; l < eleminfo.nnode; l++)
		{
			shape_info->hanginfo_Discont[l].nummaster = 0;
		}

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
				if (require_dx_terms ||   required.Pos.psi  || required.DL.dx_psi || required.normal || required.elemsize_Eulerian || required.elemsize_Eulerian_cartesian) 
				{
					
					unsigned nfields = this->nodal_dimension();

					for (unsigned int l = 0; l < eleminfo.nnode; l++)
					{					    
						if (!shape_info->hanginfo_Pos[l].nummaster)
						{
							// NON HANGING -> Set the hanging to 1 node, which is just the remapped equation
							shape_info->hanginfo_Pos[l].nummaster = 1;
							shape_info->hanginfo_Pos[l].masters[0].weight = 1.0;
							for (unsigned int f = 0; f < nfields; f++)
							{
								if (eleminfo.pos_local_eqn[l][f] >= 0)
								{
									shape_info->hanginfo_Pos[l].masters[0].local_eqn[f] = eleminfo.pos_local_eqn[l][f];
								}
								else
								{
									shape_info->hanginfo_Pos[l].masters[0].local_eqn[f] = -1;
								}
							}
						}
						// Now remap the local equations to the interface element numbering
						for (int m = 0; m < shape_info->hanginfo_Pos[l].nummaster; m++)
						{
							for (unsigned int f = 0; f < nfields; f++)
							{
								if (shape_info->hanginfo_Pos[l].masters[m].local_eqn[f] >= 0)
								{
									shape_info->hanginfo_Pos[l].masters[m].local_eqn[f] = eqn_remap[shape_info->hanginfo_Pos[l].masters[m].local_eqn[f]];
									if (shape_info->hanginfo_Pos[l].masters[m].local_eqn[f] == -666)
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
					JITHangInfo_t * hangbuffer=shape_info->hanginfo_Cont[space_info->space_index];
					/*if (space_info.space_index==0) hangbuffer = shape_info->hanginfo_C2TB;
					else if (space_info.space_index==1) hangbuffer = shape_info->hanginfo_C2;
					else if (space_info.space_index==2) hangbuffer = shape_info->hanginfo_C1TB;
					else if (space_info.space_index==3) hangbuffer = shape_info->hanginfo_C1;
					else throw_runtime_error("Invalid space index");*/
					//std::cout << " IN SPACE " << i_space << " REQUIRES CONTINUOUS SHAPE FUNCTIONS, NODES: " << nnode_space << std::endl;
					for (unsigned int l = 0; l < nnode_space; l++)
					{
						if (!hangbuffer[l].nummaster)
						{
							// NON HANGING -> Set the hanging to 1 node, which is just the remapped equation
							hangbuffer[l].nummaster = 1;
							hangbuffer[l].masters[0].weight = 1.0;
							for (unsigned int f = 0; f < space_info->numfields_basebulk; f++)
							{
								if (eleminfo.nodal_local_eqn[l][f + space_info->buffer_offset_basebulk] >= 0)
								{
									hangbuffer[l].masters[0].local_eqn[f + space_info->buffer_offset_basebulk] = eleminfo.nodal_local_eqn[l][f + space_info->buffer_offset_basebulk];
								}
								else
								{
									hangbuffer[l].masters[0].local_eqn[f + space_info->buffer_offset_basebulk] = -1;
								}
							}

							for (unsigned int f = 0; f < space_info->numfields-space_info->numfields_basebulk; f++)
							{
								if (eleminfo.nodal_local_eqn[l][f + space_info->buffer_offset_interf] >= 0)
								{
									hangbuffer[l].masters[0].local_eqn[f+ space_info->buffer_offset_interf] = eleminfo.nodal_local_eqn[l][f + space_info->buffer_offset_interf];							
								}
								else
								{
									hangbuffer[l].masters[0].local_eqn[f+ space_info->buffer_offset_interf] = -1;
								}
							}
						}	
			
						for (int m = 0; m < hangbuffer[l].nummaster; m++)
						{
							for (unsigned int f = 0; f < space_info->numfields_basebulk; f++)
							{
								unsigned foffs=f+ space_info->buffer_offset_basebulk;
								if (hangbuffer[l].masters[m].local_eqn[foffs] >= 0)
								{
									hangbuffer[l].masters[m].local_eqn[foffs] = eqn_remap[hangbuffer[l].masters[m].local_eqn[foffs]];
									if (hangbuffer[l].masters[m].local_eqn[foffs] == -666)
									{
										std::ostringstream oss;
										oss << this;
										oss << " node: " << l << ", master " << m << " of " << hangbuffer[l].nummaster  << ", index " << f << ", " << foffs << " of " << space_info->numfields_basebulk;
										throw_runtime_error("MISSING EXTERNAL DEPENDENCY ON SPACE '"+std::string(space_info->space_name)+"' ON ELEM PTR: " + oss.str());
									}
								}
							}
							for (unsigned int f = 0; f < space_info->numfields-space_info->numfields_basebulk; f++)
							{
								unsigned foffs=f+ space_info->buffer_offset_interf;
								if (hangbuffer[l].masters[m].local_eqn[foffs] >= 0)
								{
									hangbuffer[l].masters[m].local_eqn[foffs] = eqn_remap[hangbuffer[l].masters[m].local_eqn[foffs]];
									if (hangbuffer[l].masters[m].local_eqn[foffs] == -666)
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
					for (unsigned int l = 0; l < nnode_space; l++)
					{
						if (!shape_info->hanginfo_Discont[l].nummaster)
						{
							// NON HANGING -> HANGING WITH WEIGHT 1 for external element data
							shape_info->hanginfo_Discont[l].nummaster = 1;
							shape_info->hanginfo_Discont[l].masters[0].weight = 1.0;
						}
						for (unsigned int f = 0; f < space_info->numfields_basebulk; f++)
						{
							int eq=eleminfo.nodal_local_eqn[l][f + space_info->buffer_offset_basebulk];							
							if (eq >= 0)
							{
								eq=eqn_remap[eq];
								if (eq==-666)
								{
										std::ostringstream oss;
										oss << this;
										throw_runtime_error("MISSING EXTERNAL "+std::string(space_info->space_name)+" DEPENDENCY ON ELEM PTR: " + oss.str());
								}
								shape_info->hanginfo_Discont[l].masters[0].local_eqn[f + space_info->buffer_offset_basebulk] = eq ;
							}
							else
							{
								shape_info->hanginfo_Discont[l].masters[0].local_eqn[f + space_info->buffer_offset_basebulk] = -1;
							}
						}
						for (unsigned int f = 0; f < space_info->numfields-space_info->numfields_basebulk; f++)
						{
							int eq=eleminfo.nodal_local_eqn[l][f + space_info->buffer_offset_interf];					      
							if (eq >= 0)
							{
								eq=eqn_remap[eq];
								if (eq==-666)
								{
										std::ostringstream oss;
										oss << this;
										throw_runtime_error("MISSING EXTERNAL " + std::string(space_info->space_name) + " DEPENDENCY ON ELEM PTR: " + oss.str());
								}
								shape_info->hanginfo_Discont[l].masters[0].local_eqn[f + space_info->buffer_offset_interf] = eq ;
							}
							else
							{
								shape_info->hanginfo_Discont[l].masters[0].local_eqn[f + space_info->buffer_offset_interf] = -1;
							}
						}			
					}
				}
			}



			if (codeinst->get_func_table()->info_DL.numfields && (required.DL.dx_psi || required.DL.psi || required.DL.dX_psi))
			{
				for (unsigned int l = 0; l < eleminfo.nnode_DL; l++)
				{
					if (!shape_info->hanginfo_Discont[l].nummaster)
					{
						// NON HANGING -> HANGING WITH WEIGHT 1 for external element data
						shape_info->hanginfo_Discont[l].nummaster = 1;
						shape_info->hanginfo_Discont[l].masters[0].weight = 1.0;
					}
					for (unsigned int f = 0; f < codeinst->get_func_table()->info_DL.numfields; f++)
					{
					      int eq=eleminfo.nodal_local_eqn[l][f + ft->info_DL.buffer_offset_basebulk];
					      
							if (eq >= 0)
							{
							   eq=eqn_remap[eq];
							   if (eq==-666)
							   {
									std::ostringstream oss;
									oss << this;
									throw_runtime_error("MISSING EXTERNAL DL DEPENDENCY ON ELEM PTR: " + oss.str());
							   }
								shape_info->hanginfo_Discont[l].masters[0].local_eqn[f + ft->info_DL.buffer_offset_basebulk] = eq ;
							}
							else
							{
								shape_info->hanginfo_Discont[l].masters[0].local_eqn[f + ft->info_DL.buffer_offset_basebulk] = -1;
							}
					}
				}
			}


			if (codeinst->get_func_table()->info_D0.numfields && (required.D0.psi))
			{
				if (!shape_info->hanginfo_Discont[0].nummaster)
					{
						// NON HANGING -> HANGING WITH WEIGHT 1 for external element data

						shape_info->hanginfo_Discont[0].nummaster = 1;
						shape_info->hanginfo_Discont[0].masters[0].weight = 1.0;
					}
					for (unsigned int f = 0; f < codeinst->get_func_table()->info_D0.numfields; f++)
					{
					      int eq=eleminfo.nodal_local_eqn[0][f + ft->info_D0.buffer_offset_basebulk];
					      
							if (eq >= 0)
							{
							   eq=eqn_remap[eq];
							   if (eq==-666)
							   {
									std::ostringstream oss;
									oss << this;
									throw_runtime_error("MISSING EXTERNAL D0 DEPENDENCY ON ELEM PTR: " + oss.str());
							   }
								shape_info->hanginfo_Discont[0].masters[0].local_eqn[f + ft->info_D0.buffer_offset_basebulk] = eq;
							}
							else
							{
								shape_info->hanginfo_Discont[0].masters[0].local_eqn[f + ft->info_D0.buffer_offset_basebulk] = -1;
							}
					}
			}
		}
		return res;
	}

	// Default (unimplemented) hook; concrete element types with higher-order interpolation on
	// faces override this to return the boundary node at the given local face index/position.
	oomph::Node *BulkElementBase::boundary_node_pt(const int &face_index, const unsigned int index)
	{
		throw_runtime_error("Implement");
	}

	// Rebuilds this element's external-data list from scratch based on the JIT code instance's
	// linked external data (data shared/globally coupled across elements, e.g. global parameters).
	void BulkElementBase::ensure_external_data()
	{
		this->flush_external_data();
		for (auto &e : codeinst->linked_external_data.get_required_external_data())
		{
			this->add_external_data(e);
		}
	}

	// Analytically differentiates the outer unit normal vector n=dx/ds x .../|...| (line tangent
	// rotated by 90 degrees in 2d, or cross product of the two surface tangents in 3d) with respect
	// to the nodal coordinates, and optionally its second derivative. This is a direct symbolic/
	// algebraic differentiation of the normal formula (not a finite-difference approximation): each
	// case below (1d line normal in 2d, 2d surface normal in 3d) computes d(tangent)/d(node
	// coordinate) via the shape function derivatives, then applies the quotient/product rule for
	// n = t/|t| (and its second derivative) directly in index notation. Used by generated code that
	// differentiates normal-dependent boundary conditions (e.g. surface tension, moving contact
	// lines) with respect to the ALE/solid mesh position.
	void BulkElementBase::get_dnormal_dcoords_at_s(const oomph::Vector<double> &s, double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT dnormal_dcoord, double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT d2normal_dcoord2) const
	{

		unsigned nodal_dim = this->nodal_dimension();
		unsigned eldim = this->dim();

		const unsigned n_node = this->nnode();

		if (nodal_dim == 2 && eldim == 1) // Normal of a line element
		{
			oomph::Shape psi(this->nnode());
			oomph::DShape dpsi(this->nnode(), eldim);
			this->dshape_local(s, psi, dpsi);
			std::vector<double> dxds(nodal_dim, 0);
			for (unsigned int l = 0; l < this->nnode(); l++)
			{
				for (unsigned d = 0; d < nodal_dim; d++)
				{
					dxds[d] += this->nodal_position(l, d) * dpsi(l, 0);
				}
			}
			double denom = 0.0;
			for (unsigned int d = 0; d < nodal_dim; d++)
				denom += dxds[d] * dxds[d];
			if (denom < 1e-20)
				denom = 1;
			double denom_sqr=denom;
			denom = sqrt(denom);
			denom = 1 / (denom * denom * denom);
			for (unsigned int i = 0; i < nodal_dim; i++)
			{
				for (unsigned l = 0; l < n_node; l++)
				{
					for (unsigned int k = 0; k < nodal_dim; k++)
					{						
						dnormal_dcoord[i][l][k] = dpsi(l, 0) * denom * (k == 1 ? -1 : 1) * dxds[i] * dxds[1 - k];
					}
				}
			}
			if (d2normal_dcoord2)
			{			   
				for (unsigned int i = 0; i < nodal_dim; i++)
			   {
					for (unsigned l = 0; l < n_node; l++)
					{
						for (unsigned int j = 0; j < nodal_dim; j++)
						{
							for (unsigned lp = 0; lp < n_node; lp++)
							{
								for (unsigned int jp = 0; jp < nodal_dim; jp++)
								{								 
								 d2normal_dcoord2[i][l][j][lp][jp]=(i==1 ? -1 : 1)*(dpsi(l,0)*dpsi(lp,0))*denom*(( (j==jp && j!=i) ? 3*dxds[1-i] : dxds[1-(j==i ? jp : j)])-3*dxds[1-i]*dxds[j]*dxds[jp]/denom_sqr);
								}												
							}
						}
					}
				}
				
			}
		}
		else if (nodal_dim==3 && eldim==2)
		{
			const unsigned n_node = this->nnode();
			oomph::Shape psi(n_node);
			oomph::DShape dpsids(n_node, 2);
			this->dshape_local(s, psi, dpsids);
			oomph::Vector<oomph::Vector<double>> interpolated_dxds(2, oomph::Vector<double>(3, 0));
			oomph::RankFourTensor<double> dinterpolated_dxds(2, 3, n_node, 3, 0.0);

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
						normal[i] +=  EpsilonIJK(i, j, k) * interpolated_dxds[0][j] * interpolated_dxds[1][k];
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
								dndxlm(i, l, m) +=  EpsilonIJK(i, j, k) * (dinterpolated_dxds(0, m, l, j) * interpolated_dxds[1][k] + interpolated_dxds[0][j] * dinterpolated_dxds(1, m, l, k));
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
					for (unsigned int k = 0; k < 3; k++)
					{
						double crosssum = 0.0;
						for (unsigned int j = 0; j < 3; j++)
							crosssum += normal[j] * dndxlm(j, l, k);
						dnormal_dcoord[i][l][k] = dndxlm(i, l, k) / nleng - normal[i] / (nleng * nleng * nleng) * crosssum;
					}
				}
			}

			if (d2normal_dcoord2)
			{
				throw_runtime_error("Implement second order moving mesh coordinate derivatives of the normal here");
			}

		}
		else if (eldim==0 && nodal_dim==1)
		{ 
           //Actually, this does not mean anything, but we can set the derivative to zero
		   for (unsigned int i = 0; i < nodal_dim; i++)
			{
				for (unsigned l = 0; l < n_node; l++)
				{
					for (unsigned int k = 0; k < nodal_dim; k++)
					{
						dnormal_dcoord[i][l][k] = 0.0;
					}
				}
			}
			if (d2normal_dcoord2)
			{			
				for (unsigned int i = 0; i < nodal_dim; i++)
			   {
					for (unsigned l = 0; l < n_node; l++)
					{
						for (unsigned int j = 0; j < nodal_dim; j++)
						{
							for (unsigned lp = 0; lp < n_node; lp++)
							{
								for (unsigned int jp = 0; jp < nodal_dim; jp++)
								{		
								 d2normal_dcoord2[i][l][j][lp][jp]=0.0;
								}												
							}
						}
					}
				}
			}
		}
		else
		{

			for (unsigned int i = 0; i < nodal_dim; i++)
			{
				for (unsigned l = 0; l < n_node; l++)
				{
					for (unsigned int k = 0; k < nodal_dim; k++)
					{
						dnormal_dcoord[i][l][k] = 0.0;
					}
				}
			}
			std::cerr << "Cannot calculate a dnormal_dcoords for an element of dimension " + std::to_string(eldim) + " embedded in a space of dimension " + std::to_string(nodal_dim) + " yet" << std::endl << std::flush;
			throw_runtime_error("Cannot calculate a dnormal_dcoords for an element of dimension " + std::to_string(eldim) + " embedded in a space of dimension " + std::to_string(nodal_dim) + " yet");
		}
	}

	// Computes the outer unit normal n directly from the local tangent(s) (rotated tangent in 2d, cross
	// product of the two surface tangents in 3d), for the case where this element itself is
	// dimensionally a face (e.g. called directly on an interface element, as opposed to going
	// through oomph-lib's FaceElement::outer_unit_normal). Delegates the derivative computation to
	// get_dnormal_dcoords_at_s if requested.
	void BulkElementBase::get_normal_at_s(const oomph::Vector<double> &s, oomph::Vector<double> &n, double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT* PYOOMPH_RESTRICT dnormal_dcoord, double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT d2normal_dcoord2) const
	{
		unsigned nodal_dim = this->nodal_dimension();
		unsigned eldim = this->dim();

		n.resize(nodal_dim);
		if (nodal_dim == 2 && eldim == 1) // Normal of a line element
		{
			oomph::Shape psi(this->nnode());
			oomph::DShape dpsi(this->nnode(), eldim);
			this->dshape_local(s, psi, dpsi);
			std::vector<double> dxds(nodal_dim, 0);
			for (unsigned int l = 0; l < this->nnode(); l++)
			{
				for (unsigned d = 0; d < nodal_dim; d++)
				{
					dxds[d] += this->nodal_position(l, d) * dpsi(l, 0);
				}
			}
			double l = 0.0;
			for (unsigned int d = 0; d < nodal_dim; d++)
				l += dxds[d] * dxds[d];
			if (l < 1e-20)
				l = 1;
			l = sqrt(l);
			n[0] = -dxds[1] / l;
			n[1] = dxds[0] / l;
		}
		else if (nodal_dim==3 && eldim==2)
		{
			oomph::Shape psi(this->nnode());
			oomph::DShape dpsi(this->nnode(), eldim);
			this->dshape_local(s, psi, dpsi);
			std::vector<double> dxds1(nodal_dim, 0);
			std::vector<double> dxds2(nodal_dim, 0);
			for (unsigned int l = 0; l < this->nnode(); l++)
			{
				for (unsigned d = 0; d < nodal_dim; d++)
				{
					dxds1[d] += this->nodal_position(l, d) * dpsi(l, 0);
					dxds2[d] += this->nodal_position(l, d) * dpsi(l, 1);
				}
			}
			n[0]=dxds1[1]*dxds2[2]-dxds1[2]*dxds2[1];
			n[1]=dxds1[2]*dxds2[0]-dxds1[0]*dxds2[2];
			n[2]=dxds1[0]*dxds2[1]-dxds1[1]*dxds2[0];
			double l = 0.0;
			for (unsigned int d = 0; d < nodal_dim; d++) l += n[d] * n[d];
			if (l < 1e-20)
				l = 1;
			l = sqrt(l);
			for (unsigned int d = 0; d < nodal_dim; d++) n[d] /=l;
		}
		else if (nodal_dim==1 && eldim==0)
		{
			n[0]=1.0; // Makes only partially sense, but for PointMesh with a Cartesian normal mode expansion, it matters
		}
		else
		{
			std::cerr <<("Cannot calculate a normal for an element of dimension " + std::to_string(eldim) + " embedded in a space of dimension " + std::to_string(nodal_dim) + " yet") <<std::endl << std::flush;
			throw_runtime_error("Cannot calculate a normal for an element of dimension " + std::to_string(eldim) + " embedded in a space of dimension " + std::to_string(nodal_dim) + " yet");
		}
		if (dnormal_dcoord)
		{
			this->get_dnormal_dcoords_at_s(s, dnormal_dcoord, d2normal_dcoord2);
		}
	}

	// --- The get_D0/DL/DG_nodal_data / get_D0/DL/DG_buffer_index / get_D0/DL/DG_local_equation
	// family below are simple accessors translating between a discontinuous field's index (D0:
	// element-constant, DL: discontinuous-Lagrange, DG: discontinuous on a given interpolation
	// space) and the underlying oomph-lib Data object / generated-code buffer offset / local
	// equation number, using the offsets recorded in the JIT function table (ft) for this element's
	// generated code.
    oomph::Data *BulkElementBase::get_D0_nodal_data(const unsigned &fieldindex)
    {
		auto * ft=this->get_code_instance()->get_func_table();
		
        return this->internal_data_pt(ft->info_D0.internal_offset_new+fieldindex);
    }

	oomph::Data *BulkElementBase::get_DL_nodal_data(const unsigned &fieldindex)
    {
		auto * ft=this->get_code_instance()->get_func_table();
        return this->internal_data_pt(ft->info_DL.internal_offset_new+fieldindex);
    }

	oomph::Data *BulkElementBase::get_DG_nodal_data(const unsigned &space_index,const unsigned &fieldindex)
	{
		auto * ft=this->get_code_instance()->get_func_table();		
		const JITFuncSpec_Table_FiniteElement_SpaceInfo_t & space_info = ft->dg_spaces[space_index];
		return this->internal_data_pt(space_info.internal_offset_new+fieldindex);
	}

	unsigned BulkElementBase::get_DG_buffer_index(const unsigned &space_index,const unsigned &fieldindex)
	{
		auto * ft=this->get_code_instance()->get_func_table();
		const JITFuncSpec_Table_FiniteElement_SpaceInfo_t & space_info = ft->dg_spaces[space_index];
		return space_info.buffer_offset_basebulk+fieldindex;
	}

	unsigned BulkElementBase::get_DL_buffer_index(const unsigned &fieldindex)
    {
		auto * ft=this->get_code_instance()->get_func_table();
        return ft->info_DL.buffer_offset_basebulk+ fieldindex;
    }

	unsigned BulkElementBase::get_D0_buffer_index(const unsigned &fieldindex)
    {
		auto * ft=this->get_code_instance()->get_func_table();
        return ft->info_D0.buffer_offset_basebulk+ fieldindex;
    }


	int BulkElementBase::get_DG_local_equation(const unsigned &space_index,const unsigned &fieldindex,const unsigned & nodeindex)
	{
		
		auto * ft=this->get_code_instance()->get_func_table();
		const JITFuncSpec_Table_FiniteElement_SpaceInfo_t & space_info = ft->dg_spaces[space_index];
		return this->internal_local_eqn(space_info.internal_offset_new+fieldindex,nodeindex);
	}

    int BulkElementBase::get_DL_local_equation(const unsigned &fieldindex,const unsigned & nodeindex)
	{
	  auto * ft=this->get_code_instance()->get_func_table();
	  return this->internal_local_eqn(ft->info_DL.internal_offset_new+fieldindex,nodeindex);						
	}
    int BulkElementBase::get_D0_local_equation(const unsigned &fieldindex)
	{
	  auto * ft=this->get_code_instance()->get_func_table();
	  return this->internal_local_eqn(ft->info_D0.internal_offset_new+fieldindex,0);								
	}

    // TODO: Use the one defined in doi:10.1016/j.cma.2006.11.013
	// Estimates the element's characteristic size (diameter) from its vertex node positions: the
	// edge length for 1d elements, or an approximation based on the diagonal(s) between opposite
	// vertices for 2d/3d elements. Used e.g. for mesh-quality diagnostics and as a length scale in
	// stabilization terms.
	double BulkElementBase::get_element_diam() const
	{

		// Element size: Choose the max. diagonal
		double h = 0;
		if (this->dim() == 1)
		{
			h = std::fabs(this->vertex_node_pt(1)->x(0) -
						  this->vertex_node_pt(0)->x(0));
		}
		else if (this->dim() == 2)
		{
			h = pow(this->vertex_node_pt(3)->x(0) -
						this->vertex_node_pt(0)->x(0),
					2) +
				pow(this->vertex_node_pt(3)->x(1) -
						this->vertex_node_pt(0)->x(1),
					2);
			double h1 = pow(this->vertex_node_pt(2)->x(0) -
								this->vertex_node_pt(1)->x(0),
							2) +
						pow(this->vertex_node_pt(2)->x(1) -
								this->vertex_node_pt(1)->x(1),
							2);
			if (h1 > h)
				h = h1;
			h = sqrt(h);
		}
		else if (this->dim() == 3)
		{
			// diagonals are from nodes 0-7, 1-6, 2-5, 3-4
			unsigned n1 = 0;
			unsigned n2 = 7;
			for (unsigned i = 0; i < 4; i++)
			{
				double h1 = pow(this->vertex_node_pt(n1)->x(0) -
									this->vertex_node_pt(n2)->x(0),
								2) +
							pow(this->vertex_node_pt(n1)->x(1) -
									this->vertex_node_pt(n2)->x(1),
								2) +
							pow(this->vertex_node_pt(n1)->x(2) -
									this->vertex_node_pt(n2)->x(2),
								2);
				if (h1 > h)
					h = h1;
				n1++;
				n2--;
			}
			h = sqrt(h);
		}
		return h;
	}

	// Maps a local coordinate s of this element (assumed in [-1,1] per direction) into the local
	// coordinate range of the underlying macro element (structured/domain macro-element mesh), or
	// returns an empty vector if this element has no macro element associated (unstructured mesh).
	std::vector<double> BulkElementBase::get_macro_element_coordinate_at_s(oomph::Vector<double> s)
	{
		if (!macro_elem_pt()) return {};
		unsigned el_dim = dim();
		oomph::QElementBase *qelem = dynamic_cast<oomph::QElementBase *>(this);
		if (!qelem) return {};
		std::vector<double> s_macro(el_dim,0);
		for (unsigned i = 0; i < el_dim; i++)
		{
				s_macro[i] = qelem->s_macro_ll(i) + 0.5 * (s[i] + 1.0) * (qelem->s_macro_ur(i) - qelem->s_macro_ll(i));
		}
		return s_macro;
	}

	// For elements built on a macro-element (structured, e.g. curved-boundary) mesh, snaps every
	// node's Eulerian position exactly onto the macro element's geometric mapping (evaluated at that
	// node's local coordinate), so the initial mesh exactly represents the macro-element geometry
	// (e.g. a curved domain boundary) rather than just the polynomial interpolation between corners.
	// Currently only implemented for Q-family (quad/brick) elements; a no-op (TODO) for T-family elements.
	void BulkElementBase::map_nodes_on_macro_element() // Does only work for bulk elems
	{
		if (!macro_elem_pt())
			return;
		unsigned el_dim = dim();
		oomph::Vector<double> s(el_dim);
		oomph::Vector<double> r(el_dim);
		oomph::QElementBase *qelem = dynamic_cast<oomph::QElementBase *>(this);
		if (qelem)
		{
			for (unsigned int ni = 0; ni < this->nnode(); ni++)
			{
				this->local_coordinate_of_node(ni, s);
				oomph::Vector<double> s_macro(el_dim);

				for (unsigned i = 0; i < el_dim; i++)
				{
					s_macro[i] = qelem->s_macro_ll(i) + 0.5 * (s[i] + 1.0) * (qelem->s_macro_ur(i) - qelem->s_macro_ll(i));
				}

				macro_elem_pt()->macro_map(s_macro, r); // TODO: Time loop
				for (unsigned int id = 0; id < r.size(); id++)
					this->node_pt(ni)->x(id) = r[id];
			}
			return;
		}

		oomph::TElementBase *telem = dynamic_cast<oomph::TElementBase *>(this);
		if (telem)
		{
			/* TODO*/
			return;
		}
	}

	// Factory that instantiates the concrete BulkElement* subclass matching a MeshTemplateElement's
	// geometric type index (the same meshio-style codes as get_meshio_type_index()) and the current
	// JIT code instance's "dominant space" (the highest interpolation order actually used by any
	// field, C1/C1TB/C2/C2TB/...). Where the dominant space is lower order than the template
	// element's own geometry (e.g. a template C2 triangle but the code only ever uses C1 fields), a
	// lower-order element is created instead together with "nodemap": the subset (and reordering) of
	// the template element's node indices that should be used to populate the new, lower-order
	// element's nodes.
	BulkElementBase *BulkElementBase::create_from_template(MeshTemplate *mt, MeshTemplateElement *el)
	{
		BulkElementBase *res = NULL;
		std::vector<int> nodemap;
		std::string domspace=std::string(BulkElementBase::__CurrentCodeInstance->get_func_table()->dominant_space);
		if (el->get_geometric_type_index() == 1)
		{
			res = new BulkElementLine1dC1();
		}
		else if (el->get_geometric_type_index() == 2)
		{
			if ( domspace == "C1" || domspace=="C1TB")
			{
				nodemap = {0, 2};
				res = new BulkElementLine1dC1();
			}
			else
			{
			  res = new BulkElementLine1dC2();
			}
		}
		else if (el->get_geometric_type_index() == 3)
		{
		  if (dynamic_cast<MeshTemplateElementTriC1TB*>(el))
		  {
			res = new BulkElementTri2dC1TB();
		  }
		  else
		  {
 		   res = new BulkElementTri2dC1();
		  }
		}
		else if (el->get_geometric_type_index() == 4)
		{
			if (dynamic_cast<MeshTemplateElementTetraC1TB*>(el))
			{
				res = new BulkElementTetra3dC1TB();
			}
			else
			{
				res = new BulkElementTetra3dC1();
			}
		}
		else if (el->get_geometric_type_index() == 44)
			res = new BulkElementTetra3dC1TB();
		else if (el->get_geometric_type_index() == 6)
			res = new BulkElementQuad2dC1();
		else if (el->get_geometric_type_index() == 8)
		{			
			if ( domspace == "C1" || domspace=="C1TB")
			{
				nodemap = {0, 2, 6, 8};
				res = new BulkElementQuad2dC1();
			}
			else
			{
				res = new BulkElementQuad2dC2();
			}
		}
		else if (el->get_geometric_type_index() == 9)
		{
			if (domspace == "C1")
			{
				nodemap = {0, 1, 2};
				res = new BulkElementTri2dC1();
			}
			else if (domspace == "C1TB")
			{
				nodemap = {0, 1, 2,6};
				res = new BulkElementTri2dC1TB();
			}			
			else if (domspace == "C2" || domspace == "")
			{
				res = new BulkElementTri2dC2();
			}
			else
			{
				res = new BulkElementTri2dC2TB();
			}
		}
		else if (el->get_geometric_type_index() == 10) // Tetra C2
		{
			if (domspace == "C1")
			{
				nodemap = {0, 1, 2, 3};
				res = new BulkElementTetra3dC1();
			}
			else if (domspace=="C1TB")
			{
			 if (!dynamic_cast<MeshTemplateElementTetraC2TB*>(el))
			 {
			  throw_runtime_error("Strange: Tetra C1TB element should be created, but the template element is not a C2TB one, which is required for the bubble node in the center");
			 }
			 nodemap = {0, 1, 2, 3,14};
			 res = new BulkElementTetra3dC1TB();
			}
			else if (domspace == "C2")
			{
				res = new BulkElementTetra3dC2();
			}
			else
			{
				res = new BulkElementTetra3dC2TB();
			}
		}
		else if (el->get_geometric_type_index() == 11)
		{
			res = new BulkElementBrick3dC1();
		}
		else if (el->get_geometric_type_index() == 14)
		{
			if (domspace == "C1" || domspace=="C1TB")
			{
				throw_runtime_error("TODO: Restrict nodes");
			}
			else
			{
				res = new BulkElementBrick3dC2();
			}
		}
		else if (el->get_geometric_type_index() == 0)
		{
			res= new PointElement0d();
		}
		else if (el->get_geometric_type_index() == 13)
		{
			if (domspace!="C1") throw_runtime_error("Found a wedge/prism element, which cannot be generalized to the space "+domspace);
			res= new BulkElementWedge3dC1();
		}
		else if (el->get_geometric_type_index() == 26)
		{
			if (domspace == "C1")
			{
			  nodemap = {0, 1, 2, 12, 13, 14};
			  res= new BulkElementWedge3dC1();
			}
			else if (domspace=="C2")
			{
              res= new BulkElementWedge3dC2();
			}
			else
			{
				throw_runtime_error("Found a wedge/prism element, which cannot be generalized to the space "+domspace);				
			}
		}
		else if (el->get_geometric_type_index() == 15)
		{
			if (domspace == "C1")
			{			  
			  res= new BulkElementPyramid3dC1();
			}			
			else
			{
				throw_runtime_error("Pyramids are not implemented yet for space "+domspace);				
			}
		}
		else
		{
			throw_runtime_error("Undefined element type: " + std::to_string(el->get_geometric_type_index()));
		}
		if (el->get_node_indices().size() < res->nnode())
			throw_runtime_error("Too few nodes in the template element: " + std::to_string(el->get_node_indices().size()) + " vs. " + std::to_string(res->nnode()) + " element type: " + std::to_string(el->get_geometric_type_index()) + " , space: " + domspace);
		if (nodemap.empty())
		{
			for (unsigned int i = 0; i < res->nnode(); i++)
			{
				res->node_pt(i) = mt->get_nodes()[el->get_node_indices()[i]]->oomph_node;
				if (!mt->get_nodes()[el->get_node_indices()[i]]->oomph_node)
				{
					throw_runtime_error("Missing a NODE!");
				}
			}
		}
		else
		{
			for (unsigned int i = 0; i < res->nnode(); i++)
			{
				res->node_pt(i) = mt->get_nodes()[el->get_node_indices()[nodemap[i]]]->oomph_node;
			}
		}

		for (unsigned int i = 0; i < res->ninternal_data(); i++)
			res->internal_data_pt(i)->set_time_stepper(res->node_pt(0)->time_stepper_pt(), false);

		
		res->initial_cartesian_nondim_size = res->size();
		res->initial_quality_factor = res->get_quality_factor();

		if (BulkElementBase::__CurrentCodeInstance->get_func_table()->integration_order)
		{
			res->set_integration_order(BulkElementBase::__CurrentCodeInstance->get_func_table()->integration_order);
		}
		return res;
	}

	// Unpins every position and value dof on this element's nodes and internal data (positions,
	// then nodal values, then internal data values), undoing the effect of pin_dummy_values(). Used
	// e.g. before a full re-pinning pass so pinning state is rebuilt consistently from scratch.
	void BulkElementBase::unpin_dummy_values() // C1 fields on C2 elements have dummy values on only C2 nodes, which needs to be pinned
	{
		for (unsigned int l = 0; l < nnode(); l++)
		{
			for (unsigned int i = 0; i < this->nodal_dimension(); i++)
			{
				dynamic_cast<Node *>(node_pt(l))->unpin_position(i);
			}
			this->node_pt(l)->unconstrain_positions();
		}
		
		for (unsigned int l = 0; l < nnode(); l++)
		{

			for (unsigned int i = 0; i < node_pt(l)->nvalue(); i++)
			{
				node_pt(l)->unpin(i); 
			}
		}

		for (unsigned int d = 0; d < this->ninternal_data(); d++)
		{
			for (unsigned int v = 0; v < this->internal_data_pt(d)->nvalue(); v++)
			{
				this->internal_data_pt(d)->unpin(v);
			}
		}
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
			for (unsigned int l = 0; l < nnode(); l++)
			{
				for (unsigned int i = 0; i < this->nodal_dimension(); i++)
				{
					dynamic_cast<Node *>(node_pt(l))->pin_position(i);
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


	// Allocates (do_alloc=true) or frees (do_alloc=false) every array member of a single
	// JITShapeInfo_t buffer used to hold shape-function values/derivatives for one element during
	// residual assembly. Unless FIXED_SIZE_SHAPE_BUFFER is defined, every array is sized generously
	// to fixed upper bounds (MAX_NODES/MAX_NODAL_DIM/MAX_TIME_WEIGHTS/MAX_HANG/MAX_FIELDS) covering
	// the largest supported element type, so the same buffer can be reused across all element types
	// without reallocation; with_analytical_hessian_moving_mesh additionally (de)allocates the
	// second-derivative buffers only needed when computing analytic Hessians on a moving (ALE) mesh.
	void alloc_dealloc_single_shape_buffer(bool do_alloc, JITShapeInfo_t * PYOOMPH_RESTRICT *buff, bool with_analytical_hessian_moving_mesh)
	{
		if (!(*buff))
		{
			if (do_alloc)
			{
				(*buff) = new JITShapeInfo_t;
			}
			else
			{
				return;
			}
		}

#ifndef FIXED_SIZE_SHAPE_BUFFER

		const int MAX_NODES = 27; // Should be max 27 for 3^3 (Brick C2)
		const int MAX_NODAL_DIM = 3;
		const int MAX_TIME_WEIGHTS = 7;
		const int MAX_HANG = 16; // Should be max 3
		const int MAX_FIELDS = 32;

		my_alloc_or_free(do_alloc, (*buff)->t, MAX_TIME_WEIGHTS);
		my_alloc_or_free(do_alloc, (*buff)->dt, MAX_TIME_WEIGHTS);
		my_alloc_or_free(do_alloc, (*buff)->int_pt_weights_d_coords, MAX_NODAL_DIM, MAX_NODES);
		my_alloc_or_free(do_alloc, (*buff)->elemsize_d_coords, MAX_NODAL_DIM, MAX_NODES);
		my_alloc_or_free(do_alloc, (*buff)->elemsize_Cart_d_coords, MAX_NODAL_DIM, MAX_NODES);
		for (unsigned int i = 0; i < NUM_CONTINUOUS_SPACES; i++)
		{
			my_alloc_or_free(do_alloc, (*buff)->d_dx_shape_dcoord[i], MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);			
		}
				
		my_alloc_or_free(do_alloc, (*buff)->d_dx_shape_dcoord_DL, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);
		my_alloc_or_free(do_alloc, (*buff)->d_dshape_dx_tensor, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM, MAX_NODAL_DIM);

		for (unsigned int i = 0; i < NUM_CONTINUOUS_SPACES; i++)
		{
			my_alloc_or_free(do_alloc, (*buff)->shapes[i], MAX_NODES);
			my_alloc_or_free(do_alloc, (*buff)->nodal_shapes[i], MAX_NODES, MAX_NODES);
			my_alloc_or_free(do_alloc, (*buff)->dx_shapes[i], MAX_NODES, MAX_NODAL_DIM);
			my_alloc_or_free(do_alloc, (*buff)->dX_shapes[i], MAX_NODES, MAX_NODAL_DIM);
			my_alloc_or_free(do_alloc, (*buff)->dS_shapes[i], MAX_NODES, MAX_NODAL_DIM);
		}

		my_alloc_or_free(do_alloc, (*buff)->shape_DL, MAX_NODES);
		my_alloc_or_free(do_alloc, (*buff)->nodal_shape_DL, MAX_NODES, MAX_NODES);
		my_alloc_or_free(do_alloc, (*buff)->dx_shape_DL, MAX_NODES, MAX_NODAL_DIM);
		my_alloc_or_free(do_alloc, (*buff)->dX_shape_DL, MAX_NODES, MAX_NODAL_DIM);
		my_alloc_or_free(do_alloc, (*buff)->dS_shape_DL, MAX_NODES, MAX_NODAL_DIM);

		my_alloc_or_free(do_alloc, (*buff)->normal, MAX_NODAL_DIM);

		my_alloc_or_free(do_alloc, (*buff)->timestepper_weights_dt_BDF1, MAX_TIME_WEIGHTS);
		my_alloc_or_free(do_alloc, (*buff)->timestepper_weights_dt_BDF2, MAX_TIME_WEIGHTS);
		my_alloc_or_free(do_alloc, (*buff)->timestepper_weights_dt_Newmark2, MAX_TIME_WEIGHTS);
		my_alloc_or_free(do_alloc, (*buff)->timestepper_weights_d2t_Newmark2, MAX_TIME_WEIGHTS);

		my_alloc_or_free(do_alloc, (*buff)->opposite_node_index, MAX_NODES);
#else
		if (do_alloc)
			__shape_buffer_mem_usage += sizeof(JITShapeInfo_t);
#endif

		my_alloc_or_free(do_alloc, (*buff)->d_normal_dcoord, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);

		if (with_analytical_hessian_moving_mesh || !do_alloc)
		{
			my_alloc_or_free(do_alloc, (*buff)->int_pt_weights_d2_coords, MAX_NODAL_DIM, MAX_NODAL_DIM, MAX_NODES, MAX_NODES);
			my_alloc_or_free(do_alloc, (*buff)->elemsize_d2_coords, MAX_NODAL_DIM, MAX_NODAL_DIM, MAX_NODES, MAX_NODES);
			my_alloc_or_free(do_alloc, (*buff)->elemsize_Cart_d2_coords, MAX_NODAL_DIM, MAX_NODAL_DIM, MAX_NODES, MAX_NODES);	
			for (unsigned int i = 0; i < NUM_CONTINUOUS_SPACES; i++)
			{
				my_alloc_or_free(do_alloc, (*buff)->d2_dx2_shape_dcoord[i], MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);
			}			
			my_alloc_or_free(do_alloc, (*buff)->d2_dx2_shape_dcoord_DL, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);

			my_alloc_or_free(do_alloc, (*buff)->d2_normal_d2coord, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);
		}
		else
		{
			(*buff)->int_pt_weights_d2_coords = NULL;
			(*buff)->elemsize_d2_coords=NULL;
			(*buff)->elemsize_Cart_d2_coords=NULL;
			for (unsigned int i = 0; i < NUM_CONTINUOUS_SPACES; i++)
			{
				(*buff)->d2_dx2_shape_dcoord[i] = NULL;
			}
			(*buff)->d2_dx2_shape_dcoord_DL = NULL;
			(*buff)->d2_normal_d2coord = NULL;
		}

#ifndef FIXED_SIZE_SHAPE_BUFFER
		if (do_alloc)
		{
			my_alloc((*buff)->hanginfo_Pos, MAX_NODES);						

			for (unsigned int si=0;si<NUM_CONTINUOUS_SPACES;si++)
			{
				my_alloc((*buff)->hanginfo_Cont[si], MAX_NODES);
			}
			
			for (unsigned int l = 0; l < MAX_NODES; l++)
			{
				for (unsigned int si=0;si<NUM_CONTINUOUS_SPACES;si++)
				{
					my_alloc((*buff)->hanginfo_Cont[si][l].masters, MAX_HANG);
				}				
				my_alloc((*buff)->hanginfo_Pos[l].masters, MAX_HANG);
				for (unsigned int f = 0; f < MAX_HANG; f++)
				{
					for (unsigned int si=0;si<NUM_CONTINUOUS_SPACES;si++)					
					{
						my_alloc((*buff)->hanginfo_Cont[si][l].masters[f].local_eqn, MAX_FIELDS);	
					}					
					my_alloc((*buff)->hanginfo_Pos[l].masters[f].local_eqn, MAX_FIELDS);
				}
			}

			// Cannot hang, used only for local equation remapping
			my_alloc((*buff)->hanginfo_Discont, MAX_NODES);
			for (unsigned int l = 0; l < MAX_NODES; l++)
			{
				my_alloc((*buff)->hanginfo_Discont[l].masters, 1);
				my_alloc((*buff)->hanginfo_Discont[l].masters[0].local_eqn, MAX_FIELDS);
			}
		}
		else
		{
			for (unsigned int l = 0; l < MAX_NODES; l++)
			{
				for (unsigned int f = 0; f < MAX_HANG; f++)
				{
					for (unsigned int si=0;si<NUM_CONTINUOUS_SPACES;si++)
					{
						my_free((*buff)->hanginfo_Cont[si][l].masters[f].local_eqn, MAX_FIELDS);
					}					
					my_free((*buff)->hanginfo_Pos[l].masters[f].local_eqn, MAX_FIELDS);
				}
				
				for (unsigned int si=0;si<NUM_CONTINUOUS_SPACES;si++)
				{
					my_free((*buff)->hanginfo_Cont[si][l].masters, MAX_HANG);
				}				
				my_free((*buff)->hanginfo_Pos[l].masters, MAX_HANG);
				

			}
			for (unsigned int si=0;si<NUM_CONTINUOUS_SPACES;si++)
			{
				my_free((*buff)->hanginfo_Cont[si], MAX_NODES);
			}			
			my_free((*buff)->hanginfo_Pos, MAX_NODES);

			for (unsigned int l = 0; l < MAX_NODES; l++)
			{
				my_free((*buff)->hanginfo_Discont[l].masters[0].local_eqn, MAX_FIELDS);
				my_free((*buff)->hanginfo_Discont[l].masters, 1);
			}
			my_free((*buff)->hanginfo_Discont, MAX_NODES);

		}
#endif

		if (do_alloc)
		{
			(*buff)->bulk_shapeinfo = NULL;
			(*buff)->opposite_shapeinfo = NULL;
		}
		else
		{
			if ((*buff)->bulk_shapeinfo)
			{
				alloc_dealloc_single_shape_buffer(false, &((*buff)->bulk_shapeinfo), with_analytical_hessian_moving_mesh);
				delete (*buff)->bulk_shapeinfo;
			}
			if ((*buff)->opposite_shapeinfo)
			{
				alloc_dealloc_single_shape_buffer(false, &((*buff)->opposite_shapeinfo), with_analytical_hessian_moving_mesh);
				delete (*buff)->opposite_shapeinfo;
			}
			// delete *buff; //XXX: The main default shape buffer is not deallocated by default! Otherwise, reallocation does not work since DefaultShapeBuffer will be different than BulkElementBase::shape_buffer
		}
	}

	// Allocates (or frees) a full chain of shape buffers: the buffer itself, plus its
	// bulk_shapeinfo and opposite_shapeinfo (used by FaceElements to access the bulk/opposite
	// element's shape info), and one level further (bulk-of-opposite, bulk-of-bulk) since those
	// are also dereferenced when assembling flux/interface contributions.
	void alloc_dealloc_all_shape_buffers(bool do_alloc, JITShapeInfo_t **buff, bool with_analytical_hessian_moving_mesh)
	{
		if (do_alloc)
		{
			__shape_buffer_mem_usage = 0;
			alloc_dealloc_single_shape_buffer(true, buff, with_analytical_hessian_moving_mesh);
			alloc_dealloc_single_shape_buffer(true, &((*buff)->bulk_shapeinfo), with_analytical_hessian_moving_mesh);
			alloc_dealloc_single_shape_buffer(true, &((*buff)->opposite_shapeinfo), with_analytical_hessian_moving_mesh);
			alloc_dealloc_single_shape_buffer(true, &((*buff)->opposite_shapeinfo->bulk_shapeinfo), with_analytical_hessian_moving_mesh);
			alloc_dealloc_single_shape_buffer(true, &((*buff)->bulk_shapeinfo->bulk_shapeinfo), with_analytical_hessian_moving_mesh);
		//	std::cout << "Allocated " << __shape_buffer_mem_usage / (1024.0 * 1024.0) << " MB for the shape buffer" << std::endl;
		}
		else
		{
			// Deallocation of bulk_shapeinfo and opposite_shapeinfo is done in alloc_dealloc_single_shape_buffer
			alloc_dealloc_single_shape_buffer(false, buff, with_analytical_hessian_moving_mesh);
		}
	}

	// Constructs the element and lazily allocates the process-wide Default_shape_info_buffer
	// (shared by all elements of the same "shape", since only one element is assembled at a time).
	// If a previously allocated buffer lacks the second-derivative (Hessian) storage but the
	// current code instance requires it, the buffer is reallocated with that storage included.
	BulkElementBase::BulkElementBase()
	{
		memset(&eleminfo, 0, sizeof(eleminfo));

		codeinst = BulkElementBase::__CurrentCodeInstance;
		if (!codeinst)
		{
			throw_runtime_error("Element generated without jit code");
		}

		bool require_moving_hessian_buffer = this->codeinst->get_func_table()->hessian_generated && this->codeinst->get_func_table()->moving_nodes;
		//     std::cout << "SHAPE BUFFER INFO " << Default_shape_info_buffer << "  REQUI " << require_moving_hessian_buffer << std::endl;
		//     if (Default_shape_info_buffer) std::cout << Default_shape_info_buffer->int_pt_weights_d2_coords << std::endl;
		if (!Default_shape_info_buffer)
		{
			alloc_dealloc_all_shape_buffers(true, &Default_shape_info_buffer, require_moving_hessian_buffer);
			//alloc_dealloc_all_shape_buffers(true, &temp_shape_info_buffer, require_moving_hessian_buffer);
		}
		else if (require_moving_hessian_buffer && Default_shape_info_buffer->int_pt_weights_d2_coords == NULL)
		{
			alloc_dealloc_all_shape_buffers(false, &Default_shape_info_buffer, require_moving_hessian_buffer);
			alloc_dealloc_all_shape_buffers(true, &Default_shape_info_buffer, require_moving_hessian_buffer);

			//alloc_dealloc_all_shape_buffers(false, &temp_shape_info_buffer, require_moving_hessian_buffer);
			//alloc_dealloc_all_shape_buffers(true, &temp_shape_info_buffer, require_moving_hessian_buffer);
		}
		shape_info = Default_shape_info_buffer;

		this->set_nlagrangian_and_ndim(this->codeinst->get_func_table()->lagr_dim, this->codeinst->get_func_table()->nodal_dim);
		this->ensure_external_data();
	}

	// Frees all buffers set up by fill_element_info() (nodal coordinate/data/local-eqn arrays).
	// Safe to call multiple times; a no-op unless fill_element_info() actually allocated anything.
	void BulkElementBase::free_element_info()
	{
		if (!eleminfo.alloced)
			return;
		for (unsigned int i = 0; i < eleminfo.nnode; i++)
		{
			if (eleminfo.nodal_data[i])
			{
				free(eleminfo.nodal_data[i]);
				eleminfo.nodal_data[i] = NULL;
			}
			if (eleminfo.nodal_local_eqn[i])
			{
				free(eleminfo.nodal_local_eqn[i]);
				eleminfo.nodal_local_eqn[i] = NULL;
			}
			if (eleminfo.pos_local_eqn[i])
			{
				free(eleminfo.pos_local_eqn[i]);
				eleminfo.pos_local_eqn[i] = NULL;
			}
			if (eleminfo.nodal_coords[i])
			{
				// Only these are allocated, the rest is allocated in the nodes
				for (unsigned int j = eleminfo.nodal_dim + codeinst->get_func_table()->lagr_dim+this->dim() ; j < this->codeinst->get_func_table()->info_Pos.numfields; j++)
				{
					
					if (eleminfo.nodal_coords[i][j]) delete eleminfo.nodal_coords[i][j];
				}				
				free(eleminfo.nodal_coords[i]);
				eleminfo.nodal_coords[i] = NULL;
			}
		}
		//std::cout << "NODAL COORDS DEALLOCATED FOR " << this << std::endl;
		if (eleminfo.nodal_coords)
		{
			free(eleminfo.nodal_coords);
			eleminfo.nodal_coords = NULL;
		}
		if (eleminfo.nodal_data)
		{
			free(eleminfo.nodal_data);
			eleminfo.nodal_data = NULL;
		}
		if (eleminfo.nodal_local_eqn)
		{
			free(eleminfo.nodal_local_eqn);
			eleminfo.nodal_local_eqn = NULL;
		}
		if (eleminfo.pos_local_eqn)
		{
			free(eleminfo.pos_local_eqn);
			eleminfo.pos_local_eqn = NULL;
		}
		//  if (eleminfo.global_parameters) {free(eleminfo.global_parameters); eleminfo.global_parameters=NULL;}
		// if (eleminfo.nullified_residual_dof) {free(eleminfo.nullified_residual_dof); eleminfo.nullified_residual_dof=NULL;}
		eleminfo.alloced = false;
	}

	BulkElementBase::~BulkElementBase()
	{
		free_element_info();
	}

	// Extra multiplicative factor for the integration measure in curvilinear coordinate systems
	// (e.g. axisymmetric r-factor), evaluated by JIT-generated code if the code table provides it;
	// 1.0 (Cartesian, no extra factor) otherwise.
	double BulkElementBase::geometric_jacobian(const oomph::Vector<double> &x)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		if (functable->GeometricJacobian)
		{
			return functable->GeometricJacobian(&eleminfo, &(x[0]));
		}
		else
			return 1.0;
	}

	// Builds the eleminfo struct (JITElementInfo_t), the flat table of pointers to nodal
	// coordinates/data/local-equation-numbers that JIT-compiled residual/Jacobian code reads
	// directly, in a layout that mirrors the field ordering the code generator assumed
	// (continuous spaces, then DG spaces, then DL, D0, and finally external ED0 fields).
	// Called every time equation numbers or nodal data pointers may have changed (e.g. after
	// mesh adaption or equation numbering) since it caches raw pointers, not indices.
	void BulkElementBase::fill_element_info(bool without_equations)
	{
		free_element_info();

		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();

		/*eleminfo.nodal_coords = (double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT )malloc(eleminfo.nnode * sizeof(double **));		
		eleminfo.nodal_data = (double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT )calloc(eleminfo.nnode, sizeof(double **));
		eleminfo.nodal_local_eqn = (int * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT )calloc(eleminfo.nnode, sizeof(int *));
		eleminfo.pos_local_eqn = (int * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT )calloc(eleminfo.nnode, sizeof(int *));*/
		eleminfo.nodal_coords = (double ***)malloc(eleminfo.nnode * sizeof(double **));		
		eleminfo.nodal_data = (double ***)calloc(eleminfo.nnode, sizeof(double **));
		eleminfo.nodal_local_eqn = (int **)calloc(eleminfo.nnode, sizeof(int *));
		eleminfo.pos_local_eqn = (int **)calloc(eleminfo.nnode, sizeof(int *));

		// Global numfields . That might waste some memory, but it it necessary for having all aligned (in particular for additional interface fields)
		// TODO: Maybe split at least the D0 + ED0 to another storage
		unsigned numfields = 0;

		for (unsigned int i_space=0;i_space<functable->num_present_continuous_spaces;i_space++)
		{
			numfields += functable->present_continuous_spaces[i_space]->numfields;
		}

		for (unsigned int i_space=0;i_space<functable->num_present_dg_spaces;i_space++)
		{
			if (eleminfo.nnode_of_space[functable->present_dg_spaces[i_space]->space_index])
			{
				numfields += functable->present_dg_spaces[i_space]->numfields;
			}
		}
		if (eleminfo.nnode_DL)
			numfields += functable->info_DL.numfields;
		numfields += functable->info_D0.numfields + functable->info_ED0.numfields;
		for (unsigned int i = 0; i < eleminfo.nnode; i++)
		{
			oomph::Vector<double> snode(this->dim(),0.0);
			if (this->dim()>0)
			{
				this->local_coordinate_of_node(i,snode);
			}
						
			
			eleminfo.nodal_coords[i] = (double **)calloc(functable->info_Pos.numfields, sizeof(double *));
			
			for (unsigned int j = 0; j < eleminfo.nodal_dim; j++)
				eleminfo.nodal_coords[i][j] = dynamic_cast<Node *>(node_pt(i))->variable_position_pt()->value_pt(j);
			for (unsigned int j = 0; j < functable->lagr_dim; j++)
				eleminfo.nodal_coords[i][eleminfo.nodal_dim + j] = &(dynamic_cast<Node *>(node_pt(i))->xi(j));
			for (unsigned int j = 0; j < this->dim(); j++)
				eleminfo.nodal_coords[i][eleminfo.nodal_dim + functable->lagr_dim +j] = new double(snode[j]); // Local coordinate buffer

			// FaceElements additionally expose boundary (zeta) coordinates per node, appended after
			// Eulerian/Lagrangian/local-coordinate slots; these are newly allocated doubles (owned
			// here, freed in free_element_info) since Node does not store them contiguously.
			if (dynamic_cast<oomph::FaceElement*>(this))
			{
				unsigned zeta_offset=eleminfo.nodal_dim + functable->lagr_dim+this->dim();
				for (unsigned int j=zeta_offset;j<functable->info_Pos.numfields;j++)
				{
					double zeta=0.0;
					
					if ( dynamic_cast<oomph::BoundaryNodeBase*>(this->node_pt(i))  && this->node_pt(i)->boundary_coordinates_have_been_set_up()) 
					{
						oomph::Mesh * themesh=this->get_code_instance()->get_bulk_mesh();
						while (dynamic_cast<pyoomph::InterfaceMesh*>(themesh))
						{
							themesh = dynamic_cast<pyoomph::InterfaceMesh*>(themesh)->get_bulk_mesh();
						}					
						if (dynamic_cast<pyoomph::Mesh*>(themesh)->is_boundary_coordinate_defined(dynamic_cast<oomph::FaceElement*>(this)->boundary_number_in_bulk_mesh()))
						{
							
							zeta= this->zeta_nodal(i,0,j-zeta_offset);							
						}
					}
					eleminfo.nodal_coords[i][j] =   new double(zeta); // zeta coordinate buffer
				}
			}

			eleminfo.nodal_data[i] = (double **)calloc(numfields, sizeof(double *));
			eleminfo.nodal_local_eqn[i] = (int *)calloc(numfields, sizeof(int));
			for (unsigned int j = 0; j < numfields; j++)
				eleminfo.nodal_local_eqn[i][j] = -1;
			eleminfo.pos_local_eqn[i] = (int *)calloc(eleminfo.nodal_dim, sizeof(int));
			for (unsigned int j = 0; j < eleminfo.nodal_dim; j++)
				eleminfo.pos_local_eqn[i][j] = -1;
		}
		
		
		if (!without_equations)
		{
			for (unsigned int i = 0; i < eleminfo.nnode; i++)
			{
				for (unsigned int j = 0; j < eleminfo.nodal_dim; j++)
				{
					if (dynamic_cast<pyoomph::Node *>(this->node_pt(i))->is_hanging())
					{
						eleminfo.pos_local_eqn[i][j] = -2; //->constrain
					}
					else
					{
						eleminfo.pos_local_eqn[i][j] = this->position_local_eqn(i, 0, j);
					}
				}
			}
		}
				
		
		const std::vector<std::vector<unsigned>> & space_to_element_node_index =this->get_nodal_space_index_to_element_index_map();

		unsigned local_field_offset = 0;
		for (unsigned int i_space=0;i_space<functable->num_present_continuous_spaces;i_space++)
		{
			auto *space_info=functable->present_continuous_spaces[i_space];
			for (unsigned int i = 0; i < eleminfo.nnode_of_space[space_info->space_index]; i++)
			{
				unsigned i_el = space_to_element_node_index[space_info->space_index][i];
				for (unsigned int j = 0; j < functable->present_continuous_spaces[i_space]->numfields_basebulk; j++)
				{
					unsigned value_index = j + space_info->nodal_offset_basebulk;
					eleminfo.nodal_data[i][value_index] = node_pt(i_el)->value_pt(value_index); // Warning: value_pt does not work for hanging nodes! Will be changed if necessary
					if (!without_equations) eleminfo.nodal_local_eqn[i][value_index] = this->nodal_local_eqn(i_el, value_index);
				}
			}
			local_field_offset += functable->present_continuous_spaces[i_space]->numfields_basebulk;			
		}


		
		
		for (unsigned int i_space=0;i_space<functable->num_present_dg_spaces;i_space++)
		{
			auto *space_info=functable->present_dg_spaces[i_space];
			for (unsigned int i = 0; i < eleminfo.nnode_of_space[space_info->space_index]; i++)
			{
				for (unsigned int j = 0; j < space_info->numfields_basebulk; j++)
				{
					unsigned node_index = j + local_field_offset; // TODO: Use a better way to get the global field index for DG fields
					eleminfo.nodal_data[i][node_index] =  this->get_DG_nodal_data(space_info->space_index, j)->value_pt(this->get_DG_node_index(space_info->space_index, j,i)); 
					if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] =  this->get_DG_local_equation(space_info->space_index,j, i);
				}
			}
			local_field_offset += space_info->numfields_basebulk;
		}

		

						
				
		// Elemental (non-continuous) fields				
		// For interface elements,  there is a gap here for indexing. Fill be filled later		
		local_field_offset=0;
		for (unsigned int si=0;si<functable->num_present_continuous_spaces;si++)
		{
			if (eleminfo.nnode_of_space[functable->present_continuous_spaces[si]->space_index])
			{
				local_field_offset += functable->present_continuous_spaces[si]->numfields;
			}
		}
		for (unsigned int si=0;si<functable->num_present_dg_spaces;si++)
		{
			if (eleminfo.nnode_of_space[functable->present_dg_spaces[si]->space_index])
			{
				local_field_offset += functable->present_dg_spaces[si]->numfields;
			}
		}
      

		for (unsigned int i = 0; i < eleminfo.nnode_DL; i++)
		{
			for (unsigned int j = 0; j < functable->info_DL.numfields; j++)
			{
				unsigned node_index = j + local_field_offset;
				eleminfo.nodal_data[i][node_index] = this->get_DL_nodal_data( j)->value_pt(i);
				if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] = this->get_DL_local_equation(j, i);
			}
		}

		local_field_offset += functable->info_DL.numfields;		

		for (unsigned int j = 0; j < functable->info_D0.numfields; j++)
		{
			unsigned node_index = j + local_field_offset;
			eleminfo.nodal_data[0][node_index] = this->get_D0_nodal_data(j)->value_pt(0);
			if (!without_equations) eleminfo.nodal_local_eqn[0][node_index] = this->get_D0_local_equation(j);
		}

		local_field_offset = 0;
		for (unsigned int i_space=0;i_space<functable->num_present_continuous_spaces;i_space++)
		{
			auto *space_info=functable->present_continuous_spaces[i_space];
			local_field_offset += space_info->numfields;
		}
		for (unsigned int i_space=0;i_space<functable->num_present_dg_spaces;i_space++)
		{
			auto *space_info=functable->present_dg_spaces[i_space];
			local_field_offset += space_info->numfields;
		}
		local_field_offset+=functable->info_DL.numfields+functable->info_D0.numfields;

		// Create the information for the external dofs
		for (unsigned int i = 0; i < functable->info_ED0.numfields; i++)
		{

			unsigned node_index = i + local_field_offset;

			if (!without_equations)
			{
				//		std::cout << "NODE INDEX oF " << functable->fieldnames_ED0[i] << " IS " << node_index << std::endl;
				if (!codeinst->linked_external_data[i].data)
					throw_runtime_error("Element has an external data contribution, which is not assigned: " + std::string(functable->info_ED0.fieldnames[i]));
				int extdata_i = codeinst->linked_external_data[i].elemental_index+functable->info_ED0.external_offset_bulk;
				if (extdata_i >= (int)this->nexternal_data())
					throw_runtime_error("Somehow the external data array was not done well when trying to index data: " + std::string(functable->info_ED0.fieldnames[i]) + "  ext_data_index is " + std::to_string(extdata_i) + ", but only " + std::to_string((int)this->nexternal_data()) + " ext data slots present. Happened in " + codeinst->get_code()->get_file_name());
				int value_i = codeinst->linked_external_data[i].value_index;
				if (value_i < 0 || value_i >= (int)this->external_data_pt(extdata_i)->nvalue())
					throw_runtime_error("Somehow the external data array was not done, i.e. wrong value index, well when trying to index data: " + std::string(functable->info_ED0.fieldnames[i]) + " at value " + std::to_string(value_i));
				eleminfo.nodal_data[0][node_index] = this->external_data_pt(extdata_i)->value_pt(value_i); // This is a bit an issue. You cannot access this data if you don't need equations to be linked 
				eleminfo.nodal_local_eqn[0][node_index] = this->external_local_eqn(extdata_i, value_i);


			}
		}

		//	eleminfo.global_parameters=(double**)calloc(functable->numglobal_params,sizeof(double*));

		eleminfo.ndof = this->ndof();
		eleminfo.alloced = true;

		// Checking the nullified dofs
		/*
		for (unsigned int l=0;l<this->nnode();l++)
		{
		  BoundaryNode *bn=dynamic_cast<BoundaryNode*>(this->node_pt(l));
		  if (bn)
		  {
		   if (bn->nullified_dofs.count(codeinst))
		   {
			 if (!eleminfo.nullified_residual_dof) eleminfo.nullified_residual_dof=(bool*)calloc(eleminfo.ndof,sizeof(bool));
			 for (int i : bn->nullified_dofs[codeinst])
			 {
			   if (i<0)
			   {
				i=-i-1;
				i=this->position_local_eqn(l,0,i);
				if (i>=0) eleminfo.nullified_residual_dof[i]=true;
			   }
			   else
			   {
				this->nodal_local_eqn(l,i);
			   }
			 }
		   }
		  }
		}
		*/
	}

	// Convenience overload that fills the element's own (default) shape_info buffer; see the
	// shape_info-taking overload below for the actual work.
	double BulkElementBase::fill_shape_info_at_s(const oomph::Vector<double> &s, const unsigned int &index, const JITFuncSpec_RequiredShapes_FiniteElement_t &required, double &JLagr, unsigned int flag, oomph::DenseMatrix<double> *dxds,unsigned history_index) const
	{
		return fill_shape_info_at_s(s, index, required, this->shape_info, JLagr, flag, dxds,history_index);
	}

	/**
	 * When the mesh moves, we must fill in additional buffer arrays in the shape_info for the Jacobian.
	 *
	 * `interpolated_t` stores the tangent vectors, i.e.
	 *     interpolated_t(j:element_dim,i:nodal_dim)=sum_[l:numnodes] ( x^l_i * dpsi^l/ds_j )
	 *
	 * `dpsids_Element` stores the local shape derivatives (element space, i.e. max FE space in the element, C2TB/C2/C1)
	 *     dpsids_Element(l:nnode,j:element_dim )= dpsi^l/ds_j
	 *
	 * `det_Eulerian`=sqrt(det(g_{ab})) with the metric tensor g_{ab}= g(a:element_dim, b:element_dim) = sum_[i:nodal_dim] ( interpolated_t(a, i) * interpolated_t(b, i) )
	 *
	 * `aup` is the inverse of the metric tensor, i.e. g^{ab}
	 *
	 * `DXdshape_il_jb' is the resulting rank-4-tensor DXdshape(i:nodal_dim,l:numnodes,j:nodal_dim,b:element_dim).
	 *    It must return d(g^{ab}g_{a,j})/d(x_i^l) (summed over a[element_dim]) with the inverse metric tensor g^{ab} and the tangent g_{a,j}=interpolated_t(a,j)
	 *
	 * @param shape_info The destination shape information buffer
	 * @param interpolated_t local tangent vectors of the element at the integration index
	 * @param dpsids_Element stores the local shape derivatives with respect to the intrinsic coordinate s
	 * @param det_Eulerian stores the determinant of the transformation from intrinsic coordinate s to Eulerian coordinate x
	 * @param aup inverse of the metric tensor
	 * @param require_hessian indicates whether we require second order derivatives
	 * @param DXdshape_il_jb rank-4-tensor which is returned
	 */

	void BulkElementBase::fill_shape_info_at_s_dNodalPos_helper(JITShapeInfo_t *shape_info, const unsigned &index, const oomph::DenseMatrix<double> &interpolated_t, const oomph::DShape &dpsids_Element, const double det_Eulerian, const oomph::DenseMatrix<double> &aup, bool require_hessian, oomph::RankFourTensor<double> &DXdshape_il_jb,RankSixTensor * D2X2_dshape) const
	{
		unsigned el_dim = this->dim();
		unsigned n_dim = this->nodal_dimension();
		unsigned n_node = this->nnode();


		// The spatial integral contribution `dx` of the Gauss-Legendre is given by dx=det_Eulerian*integral_pt()->weight(index);
		// In particular, you get the size (length/area/volume) of the element by summing dx over all Gauss-Legendre integration points
		// If the mesh moves, dx depends on the coordinates x^l_i and we require the derivatives of dx with respect to the coordinate dofs x^l_i, i.e. i-th coordinate component of the l-th node in the element
		double dshape_dx[n_dim][n_node];
		for (unsigned l = 0; l < n_node; l++)
		{
			for (unsigned i = 0; i < n_dim; i++)
			{
				// Variable to store the information of the derivative of shape function wrt x coordinates:
				// dpsi^l/dx_i = sum_b^eldim( sum_a^eldim( g^ab * dpsi^l/ds^a * t_bi ) ). This will be used
				// for Hessian calculation.
				dshape_dx[i][l] = 0.0;

				// Store information for the derivative of dx with respect to the x coordinates:
				// d/dx^l_i(dx).
				shape_info->int_pt_weights_d_coords[i][l] = 0.0;

				for (unsigned a = 0; a < el_dim; a++)
				{
					for (unsigned b = 0; b < el_dim; b++)
					{
						dshape_dx[i][l] += aup(a, b) * dpsids_Element(l, b) * interpolated_t(a, i);
					}
				}

				// This derivative expands into:
				// sum_b^eldim( sum_a^eldim( g^ab * dpsi^l/ds^a * t_bi ) ) * sqrt(det(g^ab)) * weight(index), or simply:
				// dshape_dx[i][l] * det_Eulerian * sqrt(det(g^ab)) * weight(index).
				shape_info->int_pt_weights_d_coords[i][l] = dshape_dx[i][l] * det_Eulerian * integral_pt()->weight(index);
			}
		}
		
		// Helper tensors
		// T^{l}_{gdj}=T[l][g][d][j]
		double T[n_node][el_dim][el_dim][n_dim];
		// G^{lab}_j=G[l][a][b][j]
		double G[n_node][el_dim][el_dim][n_dim];
		
		//Fill the T tensor
		for (unsigned int l=0;l<n_node;l++)
		{
		 for (unsigned int c=0;c<el_dim;c++)
		 {
		  for (unsigned int d=0;d<el_dim;d++)
		  {
			for (unsigned int j=0;j<n_dim;j++)
			{
			  T[l][c][d][j]=dpsids_Element(l,c)*interpolated_t(d, j)+dpsids_Element(l,d)*interpolated_t(c, j);
			}
		  }
		 }
		}
		
		//Fill the G tensor
		for (unsigned int l=0;l<n_node;l++)
		{
		 for (unsigned int a=0;a<el_dim;a++)
		 {
		  for (unsigned int b=0;b<el_dim;b++)
		  {
			for (unsigned int j=0;j<n_dim;j++)
			{
			  double Gval=0.0;
			  for (unsigned int c=0;c<el_dim;c++)
			  {
			   for (unsigned int d=0;d<el_dim;d++)
			   {
			     Gval-=aup(a,c)*T[l][c][d][j]*aup(d,b);
			   }
			  }
			  G[l][a][b][j]=Gval;
			}
		  }
		 }
		}
		


		for (unsigned i = 0; i < n_dim; i++)
		{
			for (unsigned l = 0; l < n_node; l++)
			{				
				for (unsigned j = 0; j < n_dim; j++)
				{					
						for (unsigned b = 0; b < el_dim; b++)
						{
							DXdshape_il_jb(i, l, j, b) = 0.0;
							for (unsigned a = 0; a < el_dim; a++)
							{
								if (i == j)
									DXdshape_il_jb(i, l, j, b) += aup(a, b) * dpsids_Element(l, a);		
								DXdshape_il_jb(i, l, j, b) += interpolated_t(a, j) * G[l][a][b][i]; // d(g^{ab})/d(X_i^l);
							}
						}					
				}
				
			}
		}

		if (require_hessian)
		{
		   // Fill the E tensor. Note: D in the document is accessed as follows:
		   //  	$D^{lb}_{ij}=DXdshape_il_jb(j,l,i,b)
		   
		   //fill E^{ll'beta}_{ijj'}=E_hess[i][beta][l][l'][j][j']
		   for (unsigned int i=0;i<n_dim;i++) 
		   {
		     for (unsigned int b=0;b<el_dim;b++)
		     {
		       for (unsigned int l=0;l<n_node;l++)
		       {		       
		        for (unsigned int lp=0;lp<n_node;lp++)
		        {
		          for (unsigned int j=0;j<n_dim;j++)
		          {
		            for (unsigned int jp=0;jp<n_dim;jp++)
		            {
		             double Eval=0.0;
		             // First term: -D^{l'c}_{ij'}T^l_{cdj}*g^{db}
		             // and third term: -g^{ac}T^l_{cdj}*G^{l'db}_j*t_{a,i}
		             for (unsigned int c=0;c<el_dim;c++)
		             {
		              for (unsigned int d=0;d<el_dim;d++)
		              {
		               double asum=0.0;
		               for (unsigned int a=0;a<el_dim;a++)
		               {
		                asum+=aup(a,c)*G[lp][d][b][jp]*interpolated_t(a,i);
		               }
		               Eval-=T[l][c][d][j]*(DXdshape_il_jb(jp,lp,i,c)*aup(d,b) + asum);
		              }
		             }
		             // Second term, only if j=jp:
		             if (j==jp)
		             {
		               for (unsigned int c=0;c<el_dim;c++)
		               {
		                for (unsigned int d=0;d<el_dim;d++)
		                {
		                 for (unsigned int a=0;a<el_dim;a++)
		                 {
		                  Eval-=aup(a,c)*(dpsids_Element(l, c)*dpsids_Element(lp, d)+dpsids_Element(lp, c)*dpsids_Element(l, d))*aup(d,b)*interpolated_t(a,i);
		                 }
		                }
		               }
		             }
		             // Last term, only if i==j
		             if (i==j)
		             {
		              for (unsigned int a=0;a<el_dim;a++)
		              {
		                Eval+=G[lp][a][b][jp]*dpsids_Element(l,a);
		              }
		             }
		             
		             (*D2X2_dshape)(i,b,l,lp,j,jp)=Eval;

		            }
		          }
		        }
		       
		       }
		       /*
		       // Test whether it is symmetric - it should be and apparently is
		       for (unsigned int l=0;l<n_node;l++)
		       {
		        for (unsigned int lp=0;lp<n_node;lp++)
		        {		       		       		     		          
		          for (unsigned int j=0;j<n_dim;j++)
		          {
		            for (unsigned int jp=0;jp<n_dim;jp++)
		            {
		              double E1=(*D2X2_dshape)(i,b,l,lp,j,jp);
		              double E2=(*D2X2_dshape)(i,b,lp,l,jp,j);
		              double diff=E1-E2;
		              if (diff*diff>1e-6)
		              {
		                std::cout << "E["<<i<<"]["<<b<<"]  ["<<l<<"]["<<lp<<"]["<<j<<"]["<<jp<<"] = " <<E1 << " and " << E2 << "for (l,j)<->(l',j') " << std::endl;		              
		              }
		            }
		            
		          }
		        }
		       }
		       */
		     }
		   }


			// Variable to store the second derivatives of shape function wrt to coordinates, i.e.,
			// D_dshape_Dcoords[i][l][j][k] = d/dx_i^l(dpsi^k/dx_j). This can be developed into:
			// sum_b^eldim( dpsi^k/ds^b * DXdshape_il_jb(i, l, j, b) ). Used for Hessian purposes.
			double D_dshape_Dcoords[n_dim][n_node][n_dim][n_node];
			for (unsigned int i = 0; i < n_dim; i++)
			{
				for (unsigned int l = 0; l < n_node; l++)
				{
					for (unsigned int j = 0; j < n_dim; j++)
					{
						for (unsigned int k = 0; k < n_node; k++)
						{
							D_dshape_Dcoords[i][l][j][k] = 0.0;
							for (unsigned int b = 0; b < el_dim; b++)
							{
								D_dshape_Dcoords[i][l][j][k] += dpsids_Element(l, b) * DXdshape_il_jb(j, k, i, b);
								//			           std::cout << "ACCU " << i <<  " " << l << "  " << j << "  " << k << "  " << D_dshape_Dcoords[i][l][j][k] <<std::endl;
							}
						}
					}
				}
			}

			for (unsigned i = 0; i < n_dim; i++)
			{

				for (unsigned j = 0; j < n_dim; j++)
				{

					for (unsigned l = 0; l < n_node; l++)
					{

						for (unsigned k = 0; k < n_node; k++)
						{

							// The derivative of dshape_dx[i][l] * det_Eulerian * sqrt(det(g^ab))
							// wrt to the coordinates x_j^m should then be given by, applying the chain rule:
							// det_Eulerian * D_dshape_Dcoords[i][l][j][k] + (det_Eulerian * dshape_dx[j][k]) * dshape_dx[i][l],
							// where the quantities in paranthesis on the last term corresponds to the derivative of det_Eulerian
							// wrt the coordinates.
							shape_info->int_pt_weights_d2_coords[i][j][l][k] = integral_pt()->weight(index) * det_Eulerian * (dshape_dx[i][l] * dshape_dx[j][k] + D_dshape_Dcoords[i][l][j][k]);
						}
					}
				}
			}
		}
	}

	// Determinant of the Lagrangian (reference/undeformed) metric tensor at local coordinate s,
	// i.e. the analogue of the usual Eulerian Jacobian but built from the Lagrangian nodal
	// positions xi() instead of the (possibly moving) Eulerian positions. Used to integrate over
	// the reference configuration, e.g. for Lagrangian element-size or elasticity formulations.
	double BulkElementBase::J_Lagrangian(const oomph::Vector<double> &s)
	{
		unsigned el_dim = this->dim();
		unsigned n_node = this->nnode();
		unsigned n_lagr = this->nlagrangian();

		//std::cout << "NLAGR " << n_lagr << "  " << el_dim << std::endl;
		oomph::Shape psi_Element(n_node);
		oomph::DShape dpsids_Element(n_node, std::max((unsigned int)1, el_dim));
		this->dshape_local(s, psi_Element, dpsids_Element);
		oomph::DenseMatrix<double> interpolated_T(el_dim, n_lagr, 0.0);
		for (unsigned l = 0; l < n_node; l++)
		{
			for (unsigned i = 0; i < n_lagr; i++)
			{
				for (unsigned j = 0; j < el_dim; j++)
				{
					// interpolated_T(j,i) += dynamic_cast<pyoomph::Node*>(this->node_pt(l))->xi(i)*dpsids_Element(l,j);
					interpolated_T(j, i) += this->raw_lagrangian_position_gen(l, 0, i) * dpsids_Element(l, j);
				}
			}
		}

		if (el_dim == 1)
		{
			double a11 = 0.0;
			for (unsigned int i = 0; i < n_lagr; i++)
				a11 += interpolated_T(0, i) * interpolated_T(0, i);
			return sqrt(a11);
		}
		else if (el_dim == 2)
		{
			double amet[2][2];
			for (unsigned al = 0; al < 2; al++)
			{
				for (unsigned be = 0; be < 2; be++)
				{
					amet[al][be] = 0.0;
					for (unsigned i = 0; i < n_lagr; i++)
					{
						amet[al][be] += interpolated_T(al, i) * interpolated_T(be, i);
					}
				}
			}
			double det_a = amet[0][0] * amet[1][1] - amet[0][1] * amet[1][0];
			return sqrt(det_a);
		}
		else if (el_dim == 0)
		{
			return 1;
		}
		else if (el_dim == 3)
		{

			double amet[3][3];
			for (unsigned al = 0; al < 3; al++)
			{
				for (unsigned be = 0; be < 3; be++)
				{
					amet[al][be] = 0.0;
					for (unsigned i = 0; i < n_lagr; i++)
					{
						amet[al][be] += interpolated_T(al, i) * interpolated_T(be, i);
					}
				}
			}
			double det_a = amet[0][0] * amet[1][1] * amet[2][2] + amet[0][1] * amet[1][2] * amet[2][0] + amet[0][2] * amet[1][0] * amet[2][1] - amet[0][0] * amet[1][2] * amet[2][1] - amet[0][1] * amet[1][0] * amet[2][2] - amet[0][2] * amet[1][1] * amet[2][0];
			return sqrt(det_a);
		}
		else
		{
			throw_runtime_error("Implement for this dimension");
			return 1;
		}

		return 1;
	}
	
	
	// Computes element-size related quantities requested via `required` (Eulerian/Lagrangian
	// element size, in both the "physical" and Cartesian-only sense, i.e. without any extra
	// geometric_jacobian/JacobianForElementSize weighting) by integrating over all knots, and,
	// if the mesh moves (flag!=0), also their derivatives with respect to nodal coordinates
	// (and, if flag indicates a Hessian is required, second derivatives). These are needed
	// because element-size expressions in the generated code are not assembled point-wise like
	// normal residuals but require a pre-integrated scalar (and its coordinate sensitivities).
	void BulkElementBase::fill_shape_info_element_sizes(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info,unsigned flag) const
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		bool require_hessian = flag > 2;
		bool require_dxdshape = (flag && functable->moving_nodes && (!functable->fd_position_jacobian)); //&& (required.dx_psi_C2 || required.dx_psi_C1 || required.dx_psi_DL)			
		bool require_dx_elemsize=require_dxdshape && (required.elemsize_Eulerian ||  required.elemsize_Eulerian_cartesian);
		if (require_dx_elemsize)
		{
		 // Fill the derivative buffer
		 for (unsigned int i=0;i<this->nodal_dimension();i++)
		 {
		  	for (unsigned int l=0;l<this->nnode();l++)
		   {
		    shape_info->elemsize_Cart_d_coords[i][l]=0.0;
		    shape_info->elemsize_d_coords[i][l]=0.0;		    
		    if (require_hessian)
		    {
				for (unsigned int j=0;j<this->nodal_dimension();j++)
			 	{
			  		for (unsigned int m=0;m<this->nnode();m++)
					{
		      		shape_info->elemsize_d2_coords[i][j][l][m]=0.0;
  		      		shape_info->elemsize_Cart_d2_coords[i][j][l][m]=0.0;		    		    
  		      	}
  		      }
  		    }
		   }
		 }
		 JITFuncSpec_RequiredShapes_FiniteElement_t req_dummy;
		 memset(&req_dummy,0,sizeof(JITFuncSpec_RequiredShapes_FiniteElement_t));
		 req_dummy.Pos.psi=req_dummy.Pos.dx_psi=true; // Calculate these
		 double JLagr;
       for(unsigned ipt_for_esize=0;ipt_for_esize<integral_pt()->nweight();ipt_for_esize++)		 
       {  
          //double w = integral_pt()->weight(ipt_for_esize);
          oomph::Vector<double> s_for_esize(this->dim());
          for (unsigned int _i = 0; _i < this->dim(); _i++)	s_for_esize[_i] = integral_pt()->knot(ipt_for_esize, _i);
          this->fill_shape_info_at_s(s_for_esize,0,req_dummy, JLagr, flag,NULL,0); // TODO: Potentially other history indices here
          oomph::Vector<double> x_for_esize(this->nodal_dimension(),0.0);
          std::vector<double> dJdx(this->nodal_dimension(),0.0);
          std::vector<double> d2Jdx2(this->nodal_dimension()*this->nodal_dimension(),0.0);   
          double J=1.0;       
          if (required.elemsize_Eulerian)
          {          
            this->interpolated_x(s_for_esize,x_for_esize);
            J=functable->JacobianForElementSize(&eleminfo, &(x_for_esize[0]));
            if (functable->JacobianForElementSizeSpatialDerivative && flag) 
            {
              functable->JacobianForElementSizeSpatialDerivative(&eleminfo, &(x_for_esize[0]),&(dJdx[0]));
              if (functable->JacobianForElementSizeSecondSpatialDerivative && require_hessian) 
              {
                functable->JacobianForElementSizeSecondSpatialDerivative(&eleminfo, &(x_for_esize[0]),&(d2Jdx2[0]));              
              }
            }
          }                      
			 for (unsigned int i=0;i<this->nodal_dimension();i++)
			 {
			  	for (unsigned int l=0;l<this->nnode();l++)
				{
				 shape_info->elemsize_Cart_d_coords[i][l]+=shape_info->int_pt_weights_d_coords[i][l];
				 if (required.elemsize_Eulerian)
				 {
				   shape_info->elemsize_d_coords[i][l]+=shape_info->int_pt_weights_d_coords[i][l]*J;				  
				   shape_info->elemsize_d_coords[i][l]+=shape_info->int_pt_weight[0]*dJdx[i]*shape_info->shape_Pos[l];
				 }
				 if (require_hessian)
				 {
					for (unsigned int j=0;j<this->nodal_dimension();j++)
				 	{
				  		for (unsigned int m=0;m<this->nnode();m++)
						{
	  		      		shape_info->elemsize_Cart_d2_coords[i][j][l][m]+=shape_info->int_pt_weights_d2_coords[i][j][l][m];		    		    
	  		      		if (required.elemsize_Eulerian)
				         {
					   		shape_info->elemsize_d2_coords[i][j][l][m]+=shape_info->int_pt_weights_d2_coords[i][j][l][m]*J;
					   		shape_info->elemsize_d2_coords[i][j][l][m]+=shape_info->int_pt_weight[0]*d2Jdx2[i*this->nodal_dimension()+j]*shape_info->shape_Pos[l]*shape_info->shape_Pos[m];  
					   		shape_info->elemsize_d2_coords[i][j][l][m]+=shape_info->int_pt_weights_d_coords[i][l]*dJdx[j]*shape_info->shape_Pos[m];
					   		
				         }
	  		      	}
	  		      }
	  		    }				 
				}
			 }          
       }
		}
		
		
		
		
		if (required.elemsize_Eulerian || required.elemsize_Lagrangian)
		{
        //TODO: A bit redundant to do this for each integration point -> Move it in some other routine
		  shape_info->elemsize_Eulerian=0.0;
		  shape_info->elemsize_Lagrangian=0.0;		  
        for(unsigned ipt_for_esize=0;ipt_for_esize<integral_pt()->nweight();ipt_for_esize++)
        {
          double w = integral_pt()->weight(ipt_for_esize);
          oomph::Vector<double> s_for_esize(this->dim());
          for (unsigned int _i = 0; _i < this->dim(); _i++)	s_for_esize[_i] = integral_pt()->knot(ipt_for_esize, _i);
          oomph::Vector<double> x_for_esize(this->nodal_dimension(),0.0);
          if (required.elemsize_Eulerian)
          {          
            this->interpolated_x(s_for_esize,x_for_esize);
            double J = J_eulerian_at_knot(ipt_for_esize);            
            shape_info->elemsize_Eulerian += w*J*functable->JacobianForElementSize(&eleminfo, &(x_for_esize[0]));
          }
          if (required.elemsize_Lagrangian)
          {
            this->interpolated_xi(s_for_esize,x_for_esize);
            double J = J_lagrangian_at_knot(ipt_for_esize);            
            shape_info->elemsize_Lagrangian += w*J*functable->JacobianForElementSize(&eleminfo, &(x_for_esize[0]));          
          }
        }
		}
		if (required.elemsize_Eulerian_cartesian || required.elemsize_Lagrangian_cartesian)
		{
		  shape_info->elemsize_Eulerian_cartesian=0.0;
		  shape_info->elemsize_Lagrangian_cartesian=0.0;		  
        for(unsigned ipt_for_esize=0;ipt_for_esize<integral_pt()->nweight();ipt_for_esize++)
        {
          double w = integral_pt()->weight(ipt_for_esize);
          oomph::Vector<double> s_for_esize(this->dim());
          for (unsigned int _i = 0; _i < this->dim(); _i++)	s_for_esize[_i] = integral_pt()->knot(ipt_for_esize, _i);
          oomph::Vector<double> x_for_esize(this->nodal_dimension(),0.0);
          if (required.elemsize_Eulerian_cartesian)
          {          
            this->interpolated_x(s_for_esize,x_for_esize);
            double J = J_eulerian_at_knot(ipt_for_esize);            
            shape_info->elemsize_Eulerian_cartesian += w*J;
          }
          if (required.elemsize_Lagrangian_cartesian)
          {
            this->interpolated_xi(s_for_esize,x_for_esize);
            double J = J_lagrangian_at_knot(ipt_for_esize);            
            shape_info->elemsize_Lagrangian_cartesian += w*J;          
          }
        }
		}	
		
		if ( dynamic_cast<const InterfaceElementBase *>(this))
		{
			if (required.bulk_shapes)
			{
			 const BulkElementBase *bel = dynamic_cast<const BulkElementBase *>(dynamic_cast<const InterfaceElementBase *>(this)->bulk_element_pt());
			 bel->fill_shape_info_element_sizes(*(required.bulk_shapes),shape_info->bulk_shapeinfo,flag);		 
			}
			
			if (required.opposite_shapes)
			{
			 const BulkElementBase *opp = dynamic_cast<const BulkElementBase *>(dynamic_cast<const InterfaceElementBase *>(this)->get_opposite_side());
			 opp->fill_shape_info_element_sizes(*(required.opposite_shapes),shape_info->opposite_shapeinfo,flag);		 
			}
	   }	   
	}

	// Central shape-function evaluator: at the given local coordinate s, fills the shape_info
	// buffer with the values and physical (x/X) derivatives of every field space present in the
	// element (C2TB, C2, C1TB, C1, DL, ...), plus (if the mesh moves) the sensitivities of those
	// derivatives with respect to nodal coordinates, and (if a Hessian is requested) the second
	// such sensitivities -- everything the JIT-generated residual/Jacobian/Hessian code needs at
	// this integration/evaluation point. Returns the Eulerian Jacobian determinant det_Eulerian
	// and, via the JLagr reference parameter, the Lagrangian one.
	//
	// Strategy: first build the (inverse) metric tensor from the tangent vectors, both in the
	// Eulerian (interpolated_t) and Lagrangian (interpolated_T) configuration -- separately for
	// el_dim 0/1/2/3, since the metric determinant/inverse formulas differ by dimension -- along
	// with gab_gai[b][i] = g^{ab} g_{a,i}, the contraction used below to turn local-coordinate
	// shape derivatives dpsi/ds into physical ones dpsi/dx. If the mesh can move, also computes
	// DXdshape_il_jb = d(g^{ab} g_{a,j})/d(x_i^l) (and, for Hessians, D2X2_dshape) via
	// fill_shape_info_at_s_dNodalPos_helper(), which are then reused for every field space below.
	// Then, for each present field space, evaluates its local shape functions/derivatives
	// (dshape_local_at_s_XXX) and combines them with gab_gai (and DXdshape_il_jb) to fill the
	// corresponding shape_info->shape_XXX / dx_shape_XXX / dX_shape_XXX / dS_shape_XXX /
	// d_dx_shape_dcoord_XXX / d2_dx2_shape_dcoord_XXX arrays -- but only if that space is
	// actually required (per `required`) to avoid needless work.
	double BulkElementBase::fill_shape_info_at_s(const oomph::Vector<double> &s, const unsigned int &index, const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, double &JLagr, unsigned int flag, oomph::DenseMatrix<double> *dxds,unsigned history_index) const
	{
		bool require_hessian = flag > 2;

		unsigned el_dim = this->dim();
		unsigned n_dim = this->nodal_dimension();
		unsigned n_node = this->nnode();
		unsigned n_lagr = this->nlagrangian();

		double det_Eulerian;

		oomph::DenseMatrix<double> interpolated_t(el_dim, n_dim, 0.0); // Tangents
		oomph::DenseMatrix<double> interpolated_T(el_dim, n_lagr, 0.0);
		oomph::Shape psi_Element(n_node);
		oomph::DShape dpsids_Element(n_node, std::max((unsigned int)1, el_dim));
		this->dshape_local(s, psi_Element, dpsids_Element);
		for (unsigned l = 0; l < n_node; l++)
		{
			for (unsigned i = 0; i < n_dim; i++)
			{
				for (unsigned j = 0; j < el_dim; j++)
				{
					interpolated_t(j, i) += this->nodal_position(history_index, l, i) * dpsids_Element(l, j);
				}
			}
			for (unsigned i = 0; i < n_lagr; i++)
			{
				for (unsigned j = 0; j < el_dim; j++)
				{					
					interpolated_T(j, i) += this->raw_lagrangian_position_gen(l, 0, i) * dpsids_Element(l, j);
				}
			}
		}

		if (dxds)
			*dxds = interpolated_t;

		double gab_gai[el_dim][n_dim];		// stores [g^{ab} g_a]_i . First index is b second i
		double gab_gai_Lagr[el_dim][n_dim]; // stores [g^{ab} g_a]_i . First index is b second i

		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();

		bool require_dxdshape = (flag && functable->moving_nodes && (!functable->fd_position_jacobian)); //&& (required.dx_psi_C2 || required.dx_psi_C1 || required.dx_psi_DL)
		// XXX: The last condition may not be used, since even dx depends on the coordinates
		// TODO: Add a flag, whether we have a dx contribution in the residuals. If so, we always need it for moving nodes. If not (e.g. pure Lagrangian dX), we can skip it	
 
		
		oomph::RankFourTensor<double> DXdshape_il_jb; //[n_dim][n_node][n_dim][el_dim]; //this is d(g^{ab}g_{a,j})/d(x_i^l) //TODO: This could lead to stack problems due to size
      RankSixTensor * D2X2_dshape=NULL;
      if (require_hessian && require_dxdshape)
      {
        D2X2_dshape=new RankSixTensor(n_dim,el_dim,n_node,n_node,n_dim,n_dim);
      }
		if (el_dim == 1)
		{
			double a11 = 0.0;
			for (unsigned int i = 0; i < n_dim; i++)
				a11 += interpolated_t(0, i) * interpolated_t(0, i);
			for (unsigned int i = 0; i < n_dim; i++)
				gab_gai[0][i] = interpolated_t(0, i) / a11;
			det_Eulerian = sqrt(a11);

			// TODO: Only calc d(dx)/dcoords if necessary
			if (require_dxdshape)
			{
				DXdshape_il_jb.resize(n_dim, n_node, n_dim, el_dim, 0.0); // this is d(g^{ab}g_{a,j})/d(x_i^l)
				oomph::DenseMatrix<double> aup(1, 1, 1.0 / a11);
				this->fill_shape_info_at_s_dNodalPos_helper(shape_info, index, interpolated_t, dpsids_Element, det_Eulerian, aup, require_hessian, DXdshape_il_jb,D2X2_dshape);
			}

			a11 = 0.0;
			for (unsigned int i = 0; i < n_lagr; i++)
				a11 += interpolated_T(0, i) * interpolated_T(0, i);
			for (unsigned int i = 0; i < n_lagr; i++)
				gab_gai_Lagr[0][i] = interpolated_T(0, i) / a11;
			JLagr = sqrt(a11);
		}
		else if (el_dim == 2)
		{
			double amet[2][2];
			for (unsigned al = 0; al < 2; al++)
			{
				for (unsigned be = 0; be < 2; be++)
				{
					amet[al][be] = 0.0;
					for (unsigned i = 0; i < n_dim; i++)
					{
						amet[al][be] += interpolated_t(al, i) * interpolated_t(be, i);
					}
				}
			}
			double det_a = amet[0][0] * amet[1][1] - amet[0][1] * amet[1][0];
			oomph::DenseMatrix<double> aup(2, 2);
			aup(0, 0) = amet[1][1] / det_a;
			aup(0, 1) = -amet[0][1] / det_a;
			aup(1, 0) = -amet[1][0] / det_a;
			aup(1, 1) = amet[0][0] / det_a;

			for (unsigned int b = 0; b < 2; b++)
			{
				for (unsigned int i = 0; i < n_dim; i++)
				{
					gab_gai[b][i] = aup(0, b) * interpolated_t(0, i) + aup(1, b) * interpolated_t(1, i);
				}
			}
			det_Eulerian = sqrt(det_a);

			// TODO: Only calc d(dx)/dcoords if necessary
			if (require_dxdshape)
			{
				DXdshape_il_jb.resize(n_dim, n_node, n_dim, el_dim, 0.0); // this is d(g^{ab}g_{a,j})/d(x_i^l)
				this->fill_shape_info_at_s_dNodalPos_helper(shape_info, index, interpolated_t, dpsids_Element, det_Eulerian, aup, require_hessian, DXdshape_il_jb,D2X2_dshape);
			}

			// Lagr
			for (unsigned al = 0; al < 2; al++)
			{
				for (unsigned be = 0; be < 2; be++)
				{
					amet[al][be] = 0.0;
					for (unsigned i = 0; i < n_lagr; i++)
					{
						amet[al][be] += interpolated_T(al, i) * interpolated_T(be, i);
					}
				}
			}
			det_a = amet[0][0] * amet[1][1] - amet[0][1] * amet[1][0];
			aup(0, 0) = amet[1][1] / det_a;
			aup(0, 1) = -amet[0][1] / det_a;
			aup(1, 0) = -amet[1][0] / det_a;
			aup(1, 1) = amet[0][0] / det_a;

			for (unsigned int b = 0; b < 2; b++)
			{
				for (unsigned int i = 0; i < n_lagr; i++)
				{
					gab_gai_Lagr[b][i] = aup(0, b) * interpolated_T(0, i) + aup(1, b) * interpolated_T(1, i);
				}
			}
			JLagr = sqrt(det_a);
		}
		else if (el_dim == 0)
		{
			det_Eulerian = 1.0;
			JLagr = 1.0;			
			for (unsigned int ispace=0;ispace<NUM_CONTINUOUS_SPACES;ispace++)
			{
				for (unsigned l = 0; l < eleminfo.nnode_of_space[ispace]; l++)
				{
					shape_info->shapes[ispace][l] = 1.0;
					for (unsigned int i = 0; i < n_dim; i++)					
						shape_info->dx_shapes[ispace][l][i] = 0.0;
					for (unsigned int i = 0; i < n_lagr; i++)
						shape_info->dX_shapes[ispace][l][i] = 0.0;
					for (unsigned int i = 0; i < el_dim; i++)
						shape_info->dS_shapes[ispace][l][i] = 0.0;
				}
			}			
			for (unsigned l = 0; l < eleminfo.nnode_DL; l++)
			{
				shape_info->shape_DL[l] = 1.0;
				for (unsigned int i = 0; i < n_dim; i++)
					shape_info->dx_shape_DL[l][i] = 0.0;
				for (unsigned int i = 0; i < n_lagr; i++)
					shape_info->dX_shape_DL[l][i] = 0.0;
				for (unsigned int i = 0; i < el_dim; i++)
					shape_info->dS_shape_DL[l][i] = 0.0;
			}
			for (unsigned l = 0; l < n_node; l++)
			{
				for (unsigned i = 0; i < n_dim; i++)
				{
					shape_info->int_pt_weights_d_coords[i][l] = 0.0;
				}
			}
		}
		else if (el_dim == 3)
		{

			double amet[3][3];
			for (unsigned al = 0; al < 3; al++)
			{
				for (unsigned be = 0; be < 3; be++)
				{
					amet[al][be] = 0.0;
					for (unsigned i = 0; i < n_dim; i++)
					{
						amet[al][be] += interpolated_t(al, i) * interpolated_t(be, i);
					}
				}
			}
			double det_a = amet[0][0] * amet[1][1] * amet[2][2] + amet[0][1] * amet[1][2] * amet[2][0] + amet[0][2] * amet[1][0] * amet[2][1] - amet[0][0] * amet[1][2] * amet[2][1] - amet[0][1] * amet[1][0] * amet[2][2] - amet[0][2] * amet[1][1] * amet[2][0];

			oomph::DenseMatrix<double> aup(3, 3);
			aup(0, 0) = (amet[1][1] * amet[2][2] - amet[1][2] * amet[2][1]) / det_a;
			aup(0, 1) = -(amet[0][1] * amet[2][2] - amet[0][2] * amet[2][1]) / det_a;
			aup(0, 2) = (amet[0][1] * amet[1][2] - amet[0][2] * amet[1][1]) / det_a;
			aup(1, 0) = -(amet[1][0] * amet[2][2] - amet[1][2] * amet[2][0]) / det_a;
			aup(1, 1) = (amet[0][0] * amet[2][2] - amet[0][2] * amet[2][0]) / det_a;
			aup(1, 2) = -(amet[0][0] * amet[1][2] - amet[0][2] * amet[1][0]) / det_a;
			aup(2, 0) = (amet[1][0] * amet[2][1] - amet[1][1] * amet[2][0]) / det_a;
			aup(2, 1) = -(amet[0][0] * amet[2][1] - amet[0][1] * amet[2][0]) / det_a;
			aup(2, 2) = (amet[0][0] * amet[1][1] - amet[0][1] * amet[1][0]) / det_a;

			for (unsigned int b = 0; b < 3; b++)
			{
				for (unsigned int i = 0; i < n_dim; i++)
				{
					gab_gai[b][i] = aup(0, b) * interpolated_t(0, i) + aup(1, b) * interpolated_t(1, i) + aup(2, b) * interpolated_t(2, i);
				}
			}
			det_Eulerian = sqrt(det_a);

			// TODO: Only calc d(dx)/dcoords if necessary
			if (require_dxdshape)
			{
				DXdshape_il_jb.resize(n_dim, n_node, n_dim, el_dim, 0.0); // this is d(g^{ab}g_{a,j})/d(x_i^l)
				this->fill_shape_info_at_s_dNodalPos_helper(shape_info, index, interpolated_t, dpsids_Element, det_Eulerian, aup, require_hessian, DXdshape_il_jb,D2X2_dshape);
			}

			// Lagr
			for (unsigned al = 0; al < 3; al++)
			{
				for (unsigned be = 0; be < 3; be++)
				{
					amet[al][be] = 0.0;
					for (unsigned i = 0; i < n_lagr; i++)
					{
						amet[al][be] += interpolated_T(al, i) * interpolated_T(be, i);
					}
				}
			}
			det_a = amet[0][0] * amet[1][1] * amet[2][2] + amet[0][1] * amet[1][2] * amet[2][0] + amet[0][2] * amet[1][0] * amet[2][1] - amet[0][0] * amet[1][2] * amet[2][1] - amet[0][1] * amet[1][0] * amet[2][2] - amet[0][2] * amet[1][1] * amet[2][0];
			aup(0, 0) = (amet[1][1] * amet[2][2] - amet[1][2] * amet[2][1]) / det_a;
			aup(0, 1) = -(amet[0][1] * amet[2][2] - amet[0][2] * amet[2][1]) / det_a;
			aup(0, 2) = (amet[0][1] * amet[1][2] - amet[0][2] * amet[1][1]) / det_a;
			aup(1, 0) = -(amet[1][0] * amet[2][2] - amet[1][2] * amet[2][0]) / det_a;
			aup(1, 1) = (amet[0][0] * amet[2][2] - amet[0][2] * amet[2][0]) / det_a;
			aup(1, 2) = -(amet[0][0] * amet[1][2] - amet[0][2] * amet[1][0]) / det_a;
			aup(2, 0) = (amet[1][0] * amet[2][1] - amet[1][1] * amet[2][0]) / det_a;
			aup(2, 1) = -(amet[0][0] * amet[2][1] - amet[0][1] * amet[2][0]) / det_a;
			aup(2, 2) = (amet[0][0] * amet[1][1] - amet[0][1] * amet[1][0]) / det_a;

			for (unsigned int b = 0; b < 3; b++)
			{
				for (unsigned int i = 0; i < n_lagr; i++)
				{
					gab_gai_Lagr[b][i] = aup(0, b) * interpolated_T(0, i) + aup(1, b) * interpolated_T(1, i) + aup(2, b) * interpolated_T(2, i);
				}
			}
			JLagr = sqrt(det_a);
		}
		else
		{
			throw_runtime_error("Implement for this dimension");
		}

		// Now to the parts for the spaces				
		// A space's shapes are needed either because fields living in that space were explicitly
		// requested, or because it is the "dominant" (i.e. geometry-defining, Pos) space of this
		// element and Pos-shapes were requested.		

		for (unsigned int ispace=0;ispace<NUM_CONTINUOUS_SPACES;ispace++)
		{
			bool req = required.continuous_spaces[ispace].dx_psi || required.continuous_spaces[ispace].psi;
			req |= eleminfo.nnode_of_space[ispace] && (required.Pos.psi || required.Pos.dx_psi || required.Pos.dX_psi || required.continuous_spaces[ispace].dX_psi) && ((!strcmp(functable->dominant_space, functable->continuous_spaces[ispace].space_name)) || ((!strcmp(functable->dominant_space, "")) && !eleminfo.nnode_of_space[ispace]));

			if (req)
			{
				oomph::Shape psi(eleminfo.nnode_of_space[ispace]);
				oomph::DShape dpsids(eleminfo.nnode_of_space[ispace], std::max((unsigned int)1, el_dim));
				this->dshape_local_of_space(ispace, s, psi, dpsids);				
				for (unsigned l = 0; l < eleminfo.nnode_of_space[ispace]; l++)
				{
					shape_info->shapes[ispace][l] = psi[l];
					for (unsigned int i = 0; i < n_dim; i++)
					{
						shape_info->dx_shapes[ispace][l][i] = 0.0;
						for (unsigned b = 0; b < el_dim; b++)
						{
							shape_info->dx_shapes[ispace][l][i] += gab_gai[b][i] * dpsids(l, b);
						}
					}

					for (unsigned int i=0; i < this->dim();i++) shape_info->dS_shapes[ispace][l][i] =  dpsids(l, i);

					for (unsigned int i = 0; i < n_lagr; i++)
					{
						shape_info->dX_shapes[ispace][l][i] = 0.0;
						for (unsigned b = 0; b < el_dim; b++)
						{
							shape_info->dX_shapes[ispace][l][i] += gab_gai_Lagr[b][i] * dpsids(l, b);
						}
					}
					// TODO: Only if neccessary!
					if (require_dxdshape)
					{
						for (unsigned int i = 0; i < n_dim; i++)
						{
							for (unsigned l2 = 0; l2 < eleminfo.nnode; l2++)
							{
								for (unsigned int i2 = 0; i2 < n_dim; i2++)
								{
									shape_info->d_dx_shape_dcoord[ispace][l][i][l2][i2] = 0.0;
									for (unsigned int b = 0; b < el_dim; b++)
									{
										shape_info->d_dx_shape_dcoord[ispace][l][i][l2][i2] += DXdshape_il_jb(i2, l2, i, b) * dpsids(l, b); // TODO: Also for all other shapes (C1, DL)
									}
								}
							}
						}
						if (require_hessian)
						{
						for (unsigned int i = 0; i < n_dim; i++)
							{
								for (unsigned lel= 0; lel < eleminfo.nnode; lel++)
								{
								for (unsigned lel2= 0; lel2 < eleminfo.nnode; lel2++)
								{
									for (unsigned int j = 0; j < n_dim; j++)
									{
									for (unsigned int j2 = 0; j2 < n_dim; j2++)
									{
										shape_info->d2_dx2_shape_dcoord[ispace][l][i][lel][j][lel2][j2] = 0.0;
										for (unsigned int b = 0; b < el_dim; b++)
										{
											shape_info->d2_dx2_shape_dcoord[ispace][l][i][lel][j][lel2][j2] += (*D2X2_dshape)(i,b,lel,lel2,j,j2) * dpsids(l, b);
										}
									}
									}
								}
							}
						}
						}
					}
				}
			}
		}
		

		// Same pattern as above, for the discontinuous-Lagrange (DL) space (no "dominant space"
		// fallback since DL fields are never used to represent the geometry).
		if (required.DL.dx_psi || required.DL.psi)
		{
			oomph::Shape psi(eleminfo.nnode_DL);
			oomph::DShape dpsids(eleminfo.nnode_DL, std::max((unsigned int)1, el_dim));
			this->dshape_local_at_s_DL(s, psi, dpsids);
			for (unsigned l = 0; l < eleminfo.nnode_DL; l++)
			{
				shape_info->shape_DL[l] = psi[l];
				for (unsigned int i = 0; i < n_dim; i++)
				{
					shape_info->dx_shape_DL[l][i] = 0.0;
					for (unsigned b = 0; b < el_dim; b++)
					{
						shape_info->dx_shape_DL[l][i] += gab_gai[b][i] * dpsids(l, b);
					}
				}

				for (unsigned int i=0; i < this->dim();i++) shape_info->dS_shape_DL[l][i] =  dpsids(l, i);

				for (unsigned int i = 0; i < n_lagr; i++)
				{
					shape_info->dX_shape_DL[l][i] = 0.0;
					for (unsigned b = 0; b < el_dim; b++)
					{
						shape_info->dX_shape_DL[l][i] += gab_gai_Lagr[b][i] * dpsids(l, b);
					}
				}
				if (require_dxdshape)
				{
					for (unsigned int i = 0; i < n_dim; i++)
					{
						for (unsigned l2 = 0; l2 < eleminfo.nnode; l2++)
						{
							for (unsigned int i2 = 0; i2 < n_dim; i2++)
							{
								shape_info->d_dx_shape_dcoord_DL[l][i][l2][i2] = 0.0;
								for (unsigned int b = 0; b < el_dim; b++)
								{
									shape_info->d_dx_shape_dcoord_DL[l][i][l2][i2] += DXdshape_il_jb(i2, l2, i, b) * dpsids(l, b);
								}
							}
						}
					}
					if (require_hessian)
					{
					  for (unsigned int i = 0; i < n_dim; i++)
						{
							for (unsigned lel= 0; lel < eleminfo.nnode; lel++)
							{
							 for (unsigned lel2= 0; lel2 < eleminfo.nnode; lel2++)
							 {
								for (unsigned int j = 0; j < n_dim; j++)
								{
								 for (unsigned int j2 = 0; j2 < n_dim; j2++)
								 {
									shape_info->d2_dx2_shape_dcoord_DL[l][i][lel][j][lel2][j2] = 0.0;
									for (unsigned int b = 0; b < el_dim; b++)
									{
										shape_info->d2_dx2_shape_dcoord_DL[l][i][lel][j][lel2][j2] += (*D2X2_dshape)(i,b,lel,lel2,j,j2) * dpsids(l, b);
									}
								 }
								}
							}
						  }
					  }
					}
				}
			}
		}

		if (required.normal) // TODO: Better normal
		{
			oomph::Vector<double> unit_normal(this->nodal_dimension());
			this->get_normal_at_s(s, unit_normal, (require_dxdshape ? shape_info->d_normal_dcoord : NULL), ((require_hessian && require_dxdshape) ? shape_info->d2_normal_d2coord : NULL));
			for (unsigned int i = 0; i < nodal_dimension(); i++)
				shape_info->normal[i] = unit_normal[i];
		}
		
		if (D2X2_dshape) delete D2X2_dshape;

		return det_Eulerian;
	}


	// Points the generic "Pos" shape pointers (shape_Pos, dx_shape_Pos, ...) to whichever concrete
	// space (C2TB/C2/C1TB/C1) actually represents the element's geometry, so JIT code that reads
	// shape_info->shape_Pos etc. does not need to know which space is dominant. Falls back down
	// the priority chain C2TB -> C2 -> C1TB -> C1 depending on which nodes/spaces are present.
	void BulkElementBase::set_remaining_shapes_appropriately(JITShapeInfo_t *shape_info, const JITFuncSpec_RequiredShapes_FiniteElement_t &required_shapes)
	{
		bool required_C2TB = required_shapes.continuous_spaces[SPACE_INDEX_C2TB].dx_psi || required_shapes.continuous_spaces[SPACE_INDEX_C2TB].psi;
		required_C2TB |= eleminfo.nnode_of_space[SPACE_INDEX_C2TB] && (required_shapes.Pos.psi || required_shapes.Pos.dx_psi || required_shapes.Pos.dX_psi || required_shapes.continuous_spaces[SPACE_INDEX_C2TB].dX_psi) && (!strcmp(this->codeinst->get_func_table()->dominant_space, "C2TB"));
		
		bool required_C1TB = required_shapes.continuous_spaces[SPACE_INDEX_C1TB].dx_psi || required_shapes.continuous_spaces[SPACE_INDEX_C1TB].psi;		
		required_C1TB |= eleminfo.nnode_of_space[SPACE_INDEX_C1TB] && (required_shapes.Pos.psi || required_shapes.Pos.dx_psi || required_shapes.Pos.dX_psi || required_shapes.continuous_spaces[SPACE_INDEX_C1TB].dX_psi) && (!strcmp(this->codeinst->get_func_table()->dominant_space, "C1TB"));
		
		if (required_C2TB)
		{
			shape_info->shape_Pos = shape_info->shapes[SPACE_INDEX_C2TB];
			shape_info->dx_shape_Pos = shape_info->dx_shapes[SPACE_INDEX_C2TB];
			shape_info->dX_shape_Pos = shape_info->dX_shapes[SPACE_INDEX_C2TB];
			shape_info->dS_shape_Pos = shape_info->dS_shapes[SPACE_INDEX_C2TB];
			shape_info->d_dx_shape_dcoord_Pos = shape_info->d_dx_shape_dcoord[SPACE_INDEX_C2TB];
			shape_info->d2_dx2_shape_dcoord_Pos=shape_info->d2_dx2_shape_dcoord[SPACE_INDEX_C2TB];
		}
		else if (this->eleminfo.nnode_of_space[SPACE_INDEX_C2])
		{
			shape_info->shape_Pos = shape_info->shapes[SPACE_INDEX_C2];
			shape_info->dx_shape_Pos = shape_info->dx_shapes[SPACE_INDEX_C2];
			shape_info->dX_shape_Pos = shape_info->dX_shapes[SPACE_INDEX_C2];
			shape_info->dS_shape_Pos = shape_info->dS_shapes[SPACE_INDEX_C2];
			shape_info->d_dx_shape_dcoord_Pos = shape_info->d_dx_shape_dcoord[SPACE_INDEX_C2];
			shape_info->d2_dx2_shape_dcoord_Pos=shape_info->d2_dx2_shape_dcoord[SPACE_INDEX_C2];
		}
		else if (required_C1TB)
		{
			shape_info->shape_Pos = shape_info->shapes[SPACE_INDEX_C1TB];
			shape_info->dx_shape_Pos = shape_info->dx_shapes[SPACE_INDEX_C1TB];
			shape_info->dX_shape_Pos = shape_info->dX_shapes[SPACE_INDEX_C1TB];
			shape_info->dS_shape_Pos = shape_info->dS_shapes[SPACE_INDEX_C1TB];
			shape_info->d_dx_shape_dcoord_Pos = shape_info->d_dx_shape_dcoord[SPACE_INDEX_C1TB];		
			shape_info->d2_dx2_shape_dcoord_Pos=shape_info->d2_dx2_shape_dcoord[SPACE_INDEX_C1TB];
		}		
		else
		{
		  
			shape_info->shape_Pos = shape_info->shapes[SPACE_INDEX_C1];
			shape_info->dx_shape_Pos = shape_info->dx_shapes[SPACE_INDEX_C1];
			shape_info->dX_shape_Pos = shape_info->dX_shapes[SPACE_INDEX_C1];
			shape_info->dS_shape_Pos = shape_info->dS_shapes[SPACE_INDEX_C1];
			shape_info->d_dx_shape_dcoord_Pos = shape_info->d_dx_shape_dcoord[SPACE_INDEX_C1];
			shape_info->d2_dx2_shape_dcoord_Pos=shape_info->d2_dx2_shape_dcoord[SPACE_INDEX_C1];
		}
	}

	// Fills shape_info for a single Gauss integration point ipt: evaluates fill_shape_info_at_s()
	// at that point's local coordinate (and, if history-weighted time-integrals are required,
	// also at the previous one or two history configurations, to get their Jacobians), and stores
	// the combined integration weights (weight * Jacobian) used by JIT code to form dx/dX/etc.
	void BulkElementBase::fill_shape_buffer_for_integration_point(unsigned ipt, const JITFuncSpec_RequiredShapes_FiniteElement_t &required_shapes, unsigned int flag)
	{
		oomph::Vector<double> s(this->dim());
		for (unsigned int i = 0; i < this->dim(); i++)
			s[i] = integral_pt()->knot(ipt, i);
        double w = integral_pt()->weight(ipt);
		
		if (required_shapes.history_integral_dx1 || required_shapes.history_integral_dx2)
		{			
			JITFuncSpec_RequiredShapes_FiniteElement_t simplified_required_shapes;
			memset(&simplified_required_shapes, 0, sizeof(JITFuncSpec_RequiredShapes_FiniteElement_t));			
			double JLagr_dummy;			
			if (required_shapes.history_integral_dx1)
			{
			  double Jhistory = fill_shape_info_at_s(s, ipt, simplified_required_shapes,  JLagr_dummy, 0,NULL,1);
			  shape_info->int_pt_weight[1] = w * Jhistory;
			}
			if (required_shapes.history_integral_dx2)
			{
			  double Jhistory = fill_shape_info_at_s(s, ipt, simplified_required_shapes, JLagr_dummy, 0,NULL,2);
			  shape_info->int_pt_weight[2] = w * Jhistory;
			}
		}

		double JLagr;
		double J = fill_shape_info_at_s(s, ipt, required_shapes, JLagr, flag);
		
		shape_info->int_pt_weight_unity= w;
		shape_info->int_pt_weight[0] = w * J;
		shape_info->int_pt_weight_Lagrangian = w * JLagr;
		
	}

	// One-time (per residual/Jacobian assembly, not per integration point) setup of shape_info:
	// caches the number of integration points, the current time values/timesteps, and the
	// per-history-value BDF1/BDF2/Newmark2 weights used by JIT code to form time derivatives
	// (degrading to lower-order weights while too few unsteady steps have been taken yet), then
	// resolves the "Pos" shape aliases and computes element-size related quantities. Must be
	// called before fill_shape_buffer_for_integration_point() is used for the individual points.
	void BulkElementBase::prepare_shape_buffer_for_integration(const JITFuncSpec_RequiredShapes_FiniteElement_t &required_shapes, unsigned int flag)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		shape_info->n_int_pt = integral_pt()->nweight();

		const oomph::TimeStepper *tstepper = (this->nnode() ? this->node_pt(0)->time_stepper_pt() : this->internal_data_pt(0)->time_stepper_pt());
		if (tstepper->is_steady())
		{
			shape_info->timestepper_ntstorage = 0;
			for (unsigned int i = 0; i < tstepper->ntstorage(); i++)
			{
				shape_info->timestepper_weights_dt_BDF1[i] = 0;
				shape_info->timestepper_weights_dt_BDF2[i] = 0;
				shape_info->timestepper_weights_dt_Newmark2[i] = 0;
				if (functable->max_dt_order > 1)
					shape_info->timestepper_weights_d2t_Newmark2[i] = 0;
			}
			shape_info->timestepper_weights_dt_BDF2_degr = shape_info->timestepper_weights_dt_BDF2;
			shape_info->timestepper_weights_dt_Newmark2_degr = shape_info->timestepper_weights_dt_Newmark2;
		}
		else
		{
			shape_info->timestepper_ntstorage = tstepper->ntstorage();
			const MultiTimeStepper *mtstepper = dynamic_cast<const MultiTimeStepper *>(tstepper);
			if (mtstepper)
			{
				for (unsigned int i = 0; i < shape_info->timestepper_ntstorage; i++)
				{
					shape_info->timestepper_weights_dt_BDF1[i] = mtstepper->weightBDF1(1, i);
					shape_info->timestepper_weights_dt_BDF2[i] = mtstepper->weightBDF2(1, i);
					shape_info->timestepper_weights_dt_Newmark2[i] = mtstepper->weightNewmark2(1, i);
					if (functable->max_dt_order > 1)
						shape_info->timestepper_weights_d2t_Newmark2[i] = mtstepper->weightNewmark2(2, i);
				}
				unsigned unsteady_steps_done = mtstepper->get_num_unsteady_steps_done();
				if (unsteady_steps_done == 0)
				{
					shape_info->timestepper_weights_dt_BDF2_degr = shape_info->timestepper_weights_dt_BDF1;
					shape_info->timestepper_weights_dt_Newmark2_degr = shape_info->timestepper_weights_dt_BDF1;
				}				
				else
				{
					shape_info->timestepper_weights_dt_BDF2_degr = shape_info->timestepper_weights_dt_BDF2;
					shape_info->timestepper_weights_dt_Newmark2_degr = shape_info->timestepper_weights_dt_Newmark2;
				}
			}
			else
			{
				throw_runtime_error("Only the MultiTimeStepper is allowed");
			}
		}
		for (unsigned int tt = 0; tt < tstepper->time_pt()->ndt(); tt++)
		{
			shape_info->t[tt] = tstepper->time_pt()->time(tt);
			shape_info->dt[tt] = tstepper->time_pt()->dt(tt);
		}

		set_remaining_shapes_appropriately(shape_info, required_shapes);

		_currently_assembled_element = this;
		
      // Should be fine here!
      this->fill_shape_info_element_sizes(required_shapes,shape_info,flag);
		
	}

	// Evaluates a user-defined "integral expression" (an expression integrated over the element);
	// the actual loop over integration points happens inside the JIT-generated
	// EvalIntegralExpression, which reads the prepared shape_info buffer.
	double BulkElementBase::eval_integral_expression(unsigned index)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		if (index >= functable->numintegral_expressions)
			throw_runtime_error("Cannot evaluate integral expression at too large index " + std::to_string(index));		
		this->interpolate_hang_values();
		prepare_shape_buffer_for_integration(functable->shapes_required_IntegralExprs, 0);
		return functable->EvalIntegralExpression(&eleminfo, this->shape_info, index);
	}

	// Evaluates a user-defined "local expression" (a pointwise, non-integrated expression) at a
	// given node's local coordinate.
	double BulkElementBase::eval_local_expression_at_node(unsigned index, unsigned node_index)
	{
		oomph::Vector<double> s;
		this->local_coordinate_of_node(node_index, s);
		return eval_local_expression_at_s(index, s);
	}

	// Evaluates a user-defined "local expression" at an arbitrary local coordinate s.
	double BulkElementBase::eval_local_expression_at_s(unsigned index, const oomph::Vector<double> &s)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		if (index >= functable->numlocal_expressions)
			throw_runtime_error("Cannot evaluate local expression at too large index " + std::to_string(index));
		
		this->interpolate_hang_values();

		double JLagr;
		this->fill_shape_info_at_s(s, 0, codeinst->get_func_table()->shapes_required_LocalExprs, JLagr, 0);
		this->prepare_shape_buffer_for_integration(codeinst->get_func_table()->shapes_required_LocalExprs, 0);
//		set_remaining_shapes_appropriately(shape_info, codeinst->get_func_table()->shapes_required_LocalExprs);
      _currently_assembled_element = this;
	    //std::cout << "CALLING EVAL LOCAL EXPRESSION  " << this << " ELEMINFO " << &eleminfo << std::endl;
		return functable->EvalLocalExpression(&eleminfo, this->shape_info, index);
	}

	// Evaluates a user-defined "extremum expression" (used e.g. to track min/max of some field)
	// at a given node's local coordinate.
	double BulkElementBase::eval_extremum_expression_at_node(unsigned index, unsigned node_index)
	{
		oomph::Vector<double> s;
		this->local_coordinate_of_node(node_index, s);
		return eval_extremum_expression_at_s(index, s);
	}

	// Evaluates a user-defined "extremum expression" at an arbitrary local coordinate s.
	double BulkElementBase::eval_extremum_expression_at_s(unsigned index, const oomph::Vector<double> &s)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		if (index >= functable->numextremum_expressions)
			throw_runtime_error("Cannot evaluate extremum expression at too large index " + std::to_string(index));
		
		this->interpolate_hang_values();

		double JLagr;
		this->fill_shape_info_at_s(s, 0, codeinst->get_func_table()->shapes_required_ExtremumExprs, JLagr, 0);
		this->prepare_shape_buffer_for_integration(codeinst->get_func_table()->shapes_required_ExtremumExprs, 0);
      _currently_assembled_element = this;	    
		return functable->EvalExtremumExpression(&eleminfo, this->shape_info, index);
	}	

	// Evaluates a user-defined tracer-advection velocity (given in physical/Eulerian coordinates
	// by the JIT code) and converts it into a local-coordinate velocity svelo by inverting the
	// Jacobian dx/ds, so that a tracer particle can be advected in local coordinate space (used
	// e.g. to track material points through mesh motion without leaving the element frame).
	bool BulkElementBase::eval_tracer_advection_in_s_space(unsigned index, double time_frac, const oomph::Vector<double> &s, oomph::Vector<double> &svelo)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		if (index >= functable->numtracer_advections)
			throw_runtime_error("Cannot evaluate tracer advection at too large index " + std::to_string(index));
		this->interpolate_hang_values();

		double JLagr;
		oomph::DenseMatrix<double> *dxds_ptr = new oomph::DenseMatrix<double>(s.size(), s.size(), 0.0);
		this->fill_shape_info_at_s(s, 0, codeinst->get_func_table()->shapes_required_TracerAdvection, JLagr, 0, dxds_ptr);
		oomph::DenseMatrix<double> &dxds = *dxds_ptr;
		set_remaining_shapes_appropriately(shape_info, codeinst->get_func_table()->shapes_required_TracerAdvection);

		oomph::Vector<double> xvelo(s.size(), 0.0);
      _currently_assembled_element = this;
		functable->EvalTracerAdvection(&eleminfo, this->shape_info, index, time_frac, &(xvelo[0]));

		// Now calculate the dxds-Inverse
		if (dxds.nrow() == 1 && dxds.ncol() == 1)
		{
			svelo.resize(1, 0.0);
			svelo[0] = 1 / dxds(0, 0) * xvelo[0];
		}
		else if (dxds.nrow() == 2 && dxds.ncol() == 2)
		{
			double det_a = dxds(0, 0) * dxds(1, 1) - dxds(0, 1) * dxds(1, 0);
			oomph::DenseMatrix<double> dsdx2d(2, 2, 0.0);
			dsdx2d(0, 0) = dxds(1, 1) / det_a;
			dsdx2d(0, 1) = -dxds(0, 1) / det_a;
			dsdx2d(1, 0) = -dxds(1, 0) / det_a;
			dsdx2d(1, 1) = dxds(0, 0) / det_a;
			svelo.resize(2, 0.0);
			svelo[0] = 0.0;
			svelo[1] = 0.0;
			for (unsigned int i = 0; i < 2; i++)
				for (unsigned int j = 0; j < 2; j++)
					svelo[j] += dsdx2d(i, j) * xvelo[i];
		}
		else if (dxds.nrow() == 3 && dxds.ncol() == 3)
		{
			double det_a = dxds(0, 0) * dxds(1, 1) * dxds(2, 2) + dxds(0, 1) * dxds(1, 2) * dxds(2, 0) + dxds(0, 2) * dxds(1, 0) * dxds(2, 1) - dxds(0, 0) * dxds(1, 2) * dxds(2, 1) - dxds(0, 1) * dxds(1, 0) * dxds(2, 2) - dxds(0, 2) * dxds(1, 1) * dxds(2, 0);

			oomph::DenseMatrix<double> dsdx(3, 3, 0.0);
			dsdx(0, 0) = (dxds(1, 1) * dxds(2, 2) - dxds(1, 2) * dxds(2, 1)) / det_a;
			dsdx(0, 1) = -(dxds(0, 1) * dxds(2, 2) - dxds(0, 2) * dxds(2, 1)) / det_a;
			dsdx(0, 2) = (dxds(0, 1) * dxds(1, 2) - dxds(0, 2) * dxds(1, 1)) / det_a;
			dsdx(1, 0) = -(dxds(1, 0) * dxds(2, 2) - dxds(1, 2) * dxds(2, 0)) / det_a;
			dsdx(1, 1) = (dxds(0, 0) * dxds(2, 2) - dxds(0, 2) * dxds(2, 0)) / det_a;
			dsdx(1, 2) = -(dxds(0, 0) * dxds(1, 2) - dxds(0, 2) * dxds(1, 0)) / det_a;
			dsdx(2, 0) = (dxds(1, 0) * dxds(2, 1) - dxds(1, 1) * dxds(2, 0)) / det_a;
			dsdx(2, 1) = -(dxds(0, 0) * dxds(2, 1) - dxds(0, 1) * dxds(2, 0)) / det_a;
			dsdx(2, 2) = (dxds(0, 0) * dxds(1, 1) - dxds(0, 1) * dxds(1, 0)) / det_a;
			svelo.resize(3, 0.0);
			svelo[0] = 0.0;
			svelo[1] = 0.0;
			svelo[2] = 0.0;
			for (unsigned int i = 0; i < 3; i++)
				for (unsigned int j = 0; j < 3; j++)
					svelo[j] += dsdx(i, j) * xvelo[i];
		}
		else
		{
			throw_runtime_error("Cannot do this here");
		}

		delete dxds_ptr;

		return true;
	}

	// Multi-assembly: performs several residual/Jacobian/mass-matrix/Hessian-vector-product/
	// parameter-derivative contributions (described by `info`, one entry per "contribution",
	// e.g. bulk + several attached interfaces) for this element in a single pass, sharing one
	// shape_info fill per integration point instead of recomputing shapes from scratch for each
	// contribution separately. First merges the RequiredShapes of all requested contributions
	// (so the shape buffer is filled generously enough for all of them at once) and determines
	// the maximum needed shapeflag (0=residuals,1=+Jacobian,2=+mass matrix,3=+Hessian). Then
	// loops over integration points (or just once, if the code was compiled with a private,
	// non-shared shape buffer per contribution) and, for each requested contribution, calls the
	// appropriate JIT-generated residual/Jacobian/mass-matrix function, parameter-derivative
	// function, and/or Hessian-vector-product function.
	void BulkElementBase::get_multi_assembly(std::vector<SinglePassMultiAssembleInfo> &info)
	{
		JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		JITFuncSpec_RequiredShapes_FiniteElement_t *required_shapes = (JITFuncSpec_RequiredShapes_FiniteElement_t *)std::calloc(1, sizeof(JITFuncSpec_RequiredShapes_FiniteElement_t));
		int shapeflag = -1;
		// std::cout << "MERGED ASSEMBLY " << std::endl;
		// First pass: merge required shapes across all contributions and figure out the highest
		// shapeflag (residuals/Jacobian/mass-matrix/Hessian) needed overall.
		for (auto &inf : info)
		{
			if (inf.contribution < 0)
				continue;
			bool resjac_merged = false;
			if (inf.residuals || inf.jacobian || inf.mass_matrix)
			{
				resjac_merged = true;
				if (functable->fd_jacobian || functable->fd_position_jacobian)
					throw_runtime_error("Multi-assembly does not work with fd_jacobian or fd_position_jacobian");
				//    std::cout << "  MERGED ResJac " << inf.contribution << std::endl;
				RequiredShapes_merge(&functable->shapes_required_ResJac[inf.contribution], required_shapes);
			}
			if (inf.residuals)
				shapeflag = 0;
			if (inf.jacobian && shapeflag < 1)
				shapeflag = 1;
			if (inf.mass_matrix && shapeflag < 2)
				shapeflag = 2;
			if (inf.hessians.size())
			{
				RequiredShapes_merge(&functable->shapes_required_Hessian[inf.contribution], required_shapes);
				shapeflag = 3;
				//    std::cout << "  MERGED HEssian " << inf.contribution << std::endl;
			}
			if (functable->ParameterDerivative)
			{
				for (auto &pdiff : inf.dparams)
				{
					unsigned global_param_index = codeinst->get_problem()->resolve_parameter_value_ptr(pdiff.parameter);
					int paramindex = -1;
					for (unsigned int i = 0; i < functable->numglobal_params; i++)
					{
						if (functable->global_paramindices[i] == global_param_index)
						{
							paramindex = i;
							break;
						}
					}
					if (paramindex >= 0)
					{
						if (functable->ParameterDerivative[inf.contribution] && functable->ParameterDerivative[inf.contribution][paramindex])
						{
							if (!resjac_merged && (pdiff.dRdparam || pdiff.dJdparam || pdiff.dMdparam))
							{
								resjac_merged = true;
								if (functable->fd_jacobian || functable->fd_position_jacobian)
									throw_runtime_error("Multi-assembly does not work with fd_jacobian or fd_position_jacobian");
								//            std::cout << "  MERGED dParamResJac " << inf.contribution <<"   " << paramindex << "  " << pdiff.parameter << std::endl;
								RequiredShapes_merge(&functable->shapes_required_ResJac[inf.contribution], required_shapes);
							}
							if (pdiff.dMdparam && shapeflag < 2)
								shapeflag = 2;
							else if (pdiff.dJdparam && shapeflag < 1)
								shapeflag = 1;
							else if (pdiff.dRdparam && shapeflag < 0)
								shapeflag = 0;
						}
					}
				}
			}
		}
		// std::cout << " SHAPEFLAG " << shapeflag << std::endl;
		if (shapeflag < 0)
		{
			RequiredShapes_free(required_shapes);
			return; // Nothing to assemble at all
		}

		// This is the only benefit of this approach: We only have to do this once!
		this->fill_hang_info_with_equations(*required_shapes, this->shape_info, NULL);
		this->interpolate_hang_values();
		prepare_shape_buffer_for_integration(*required_shapes, shapeflag);

		bool has_hang = true; // Assuming always hanging at the moment
      bool shared_multi_assemble=functable->use_shared_shape_buffer_during_multi_assemble;
      functable->during_shared_multi_assembling=shared_multi_assemble;
      unsigned n_int_pt=(shared_multi_assemble ? this->shape_info->n_int_pt : 1);
      for (unsigned int i_int_pt=0;i_int_pt<n_int_pt;i_int_pt++)
      {

			for (auto &inf : info)
			{
				if (inf.contribution < 0)
					continue;
				// Fill the shape buffer once per integration point, shared across all contributions
				// (only if the code table allows sharing; otherwise each contribution's function
				// call internally handles filling the buffer for all points itself).
				if (shared_multi_assemble)
				{
				  this->fill_shape_buffer_for_integration_point(i_int_pt,*required_shapes,shapeflag);
				}

				// Base contribution: residuals / Jacobian / mass matrix
				if (inf.residuals || inf.jacobian || inf.mass_matrix)
				{
					JITFuncSpec_ResidualAndJacobian_FiniteElement func;
					const oomph::TimeStepper *tstepper = (this->nnode() ? this->node_pt(0)->time_stepper_pt() : this->internal_data_pt(0)->time_stepper_pt());

					if (tstepper->is_steady())
					{
						func = functable->ResidualAndJacobianSteady[inf.contribution];
					}
					else
					{
						if (!has_hang)
							func = functable->ResidualAndJacobian_NoHang[inf.contribution];
						else
							func = functable->ResidualAndJacobian[inf.contribution];
					}
					if (func)
					{
						if (inf.mass_matrix) // residuals, Jacobian, Mass matrix
						{
							if (!inf.jacobian || !inf.residuals)
								throw_runtime_error("Cannot multiassemble a mass matrix without setting Jacobian and residual (possibly dummies)");
							shape_info->jacobian_size = inf.jacobian->nrow();
							shape_info->mass_matrix_size = inf.mass_matrix->nrow();
							//             std::cout << " AEESMBLE RJM " << inf.contribution << std::endl;
							func(&eleminfo, shape_info, &(((*inf.residuals)[0])), &(inf.jacobian->entry(0, 0)), &(inf.mass_matrix->entry(0, 0)), 2);
						}
						else if (inf.jacobian) // residuals, Jacobian
						{
							if (!inf.residuals)
								throw_runtime_error("Cannot multiassemble a Jacobian without setting residual (possibly dummy)");
							//             std::cout << " AEESMBLE RJ " << inf.contribution << std::endl;
							shape_info->jacobian_size = inf.jacobian->nrow();
							func(&eleminfo, shape_info, &(((*inf.residuals)[0])), &(inf.jacobian->entry(0, 0)), NULL, 1);
						}
						else if (inf.residuals)
						{
							//             std::cout << " AEESMBLE R " << inf.contribution << std::endl;
							func(&eleminfo, shape_info, &(((*inf.residuals)[0])), NULL, NULL, 0);
						}
					}
				}

				// Parameter derivatives
				if (functable->ParameterDerivative)
				{
					for (auto &pinf : inf.dparams)
					{
						if (!functable->ParameterDerivative[inf.contribution])
							continue;
						unsigned global_param_index = codeinst->get_problem()->resolve_parameter_value_ptr(pinf.parameter);
						int paramindex = -1;
						for (unsigned int i = 0; i < functable->numglobal_params; i++)
						{
							if (functable->global_paramindices[i] == global_param_index)
							{
								paramindex = i;
								break;
							}
						}
						if (paramindex < 0)
							continue;
						if (!functable->ParameterDerivative[inf.contribution][paramindex])
							continue;
						if (pinf.dMdparam) // residuals, Jacobian, Mass matrix
						{
							if (!pinf.dJdparam || !pinf.dRdparam)
								throw_runtime_error("Cannot multiassemble a mass matrix without setting Jacobian and residual (possibly dummies). Happens in parameter derivative");
							//             std::cout << " AEESMBLE PARAMDERIV RJM " << inf.contribution << "  " << paramindex << "  " << pinf.parameter << std::endl;
							shape_info->jacobian_size = pinf.dJdparam->nrow();
							shape_info->mass_matrix_size = pinf.dMdparam->nrow();
							functable->ParameterDerivative[inf.contribution][paramindex](&eleminfo, shape_info, &(((*pinf.dRdparam)[0])), &(pinf.dJdparam->entry(0, 0)), &(pinf.dMdparam->entry(0, 0)), 2);
						}
						else if (pinf.dJdparam) // residuals, Jacobian
						{
							if (!pinf.dRdparam)
								throw_runtime_error("Cannot multiassemble a Jacobian without setting residual (possibly dummy). Happens in parameter derivative");
							//             std::cout << " AEESMBLE PARAMDERIV RJ " << inf.contribution << "  " << paramindex << "  " << pinf.parameter << std::endl;
							shape_info->jacobian_size = pinf.dJdparam->nrow();
							functable->ParameterDerivative[inf.contribution][paramindex](&eleminfo, shape_info, &(((*pinf.dRdparam)[0])), &(pinf.dJdparam->entry(0, 0)), NULL, 1);
						}
						else if (pinf.dRdparam)
						{
							//             std::cout << " AEESMBLE PARAMDERIV R " << inf.contribution << "  " << paramindex << "  " << pinf.parameter << std::endl;
							functable->ParameterDerivative[inf.contribution][paramindex](&eleminfo, shape_info, &(((*pinf.dRdparam)[0])), NULL, NULL, 0);
						}
					}
				}

				// Hessians
				if (inf.hessians.size())
				{
					if (!functable->hessian_generated)
						throw_runtime_error("You want to calculate Hessian contributions, but analytical Hessian were not set. Please call problem.setup_for_stability_analysis(analytic_hessian=True) before just-in-time compilation");

					for (auto &hinf : inf.hessians)
					{
						if (!functable->HessianVectorProduct || !functable->HessianVectorProduct[inf.contribution])
							continue;
						if (!hinf.M_Hessian && !hinf.J_Hessian)
							continue;
						if (hinf.M_Hessian && !hinf.J_Hessian)
							throw_runtime_error("You want to calculate Hessian mass contributions, but you must set a potentially dummy Hessian Jacobian.");
						unsigned n_var = hinf.Y.size();
						unsigned n_vec = hinf.J_Hessian->ncol();
						if (n_var%n_vec!=0) throw_runtime_error("Y and Hessian must fulfill #Y modulo ncol(H) =0. Thereby, you can assembly multiple vectors products at once");
						shape_info->jacobian_size = n_vec;
						n_vec=n_var/n_vec;
						//std::cout << "NVEC " << n_vec << std::endl;
						if (hinf.M_Hessian)
						{
							//             std::cout << " AEESMBLE HESS JM " << inf.contribution << "  " << &hinf.Y << "  " <<  std::endl;
							functable->HessianVectorProduct[inf.contribution](&eleminfo, shape_info, &hinf.Y[0], &(hinf.M_Hessian->entry(0, 0)), &(hinf.J_Hessian->entry(0, 0)), n_vec, (hinf.transposed ? 5:  2));
						}
						else
						{
							//            std::cout << " AEESMBLE HESS J " << inf.contribution << "  " << &hinf.Y << "  " <<  std::endl;
							functable->HessianVectorProduct[inf.contribution](&eleminfo, shape_info, &hinf.Y[0], NULL, &(hinf.J_Hessian->entry(0, 0)), n_vec, (hinf.transposed ? 4:  1));
						}
					}
				}
			}
		}
      functable->during_shared_multi_assembling=false;
		RequiredShapes_free(required_shapes);
	}

	///\short Compute the derivatives of the
	/// residuals with respect to a parameter
	/// Flag=1 (or 0): do (or don't) compute the Jacobian as well.
	/// Flag=2: Fill in mass matrix too.
	void BulkElementBase::fill_in_generic_dresidual_contribution_jit(double *const &parameter_pt, oomph::Vector<double> &dres_dparam, oomph::DenseMatrix<double> &djac_dparam, oomph::DenseMatrix<double> &dmass_matrix_dparam, unsigned flag)
	{

		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		if (functable->current_res_jac < 0)
			return;
		if (!functable->ParameterDerivative)
			return;
		if (!functable->ParameterDerivative[functable->current_res_jac])
			return;
		unsigned global_param_index = codeinst->get_problem()->resolve_parameter_value_ptr(parameter_pt);
		int paramindex = -1;
		for (unsigned int i = 0; i < functable->numglobal_params; i++)
		{
			if (functable->global_paramindices[i] == global_param_index)
			{
				paramindex = i;
				break;
			}
		}
		if (paramindex < 0)
			return; // Nothing to do -> Element does not depend on this parameter
		if (!functable->ParameterDerivative[functable->current_res_jac][paramindex])
			return;
		this->fill_hang_info_with_equations(functable->shapes_required_ResJac[functable->current_res_jac], this->shape_info, NULL);
		this->interpolate_hang_values(); // XXX This should be moved to somewhere else, after each update of any values
		prepare_shape_buffer_for_integration(functable->shapes_required_ResJac[functable->current_res_jac], flag);
		shape_info->jacobian_size = djac_dparam.nrow();
		shape_info->mass_matrix_size = dmass_matrix_dparam.nrow();

		if (!functable->ParameterDerivative[functable->current_res_jac][paramindex])
			return;
		if (flag)
		{
			if (flag >= 2) // residuals, Jacobian, Mass matrix
			{
				functable->ParameterDerivative[functable->current_res_jac][paramindex](&eleminfo, shape_info, &(dres_dparam[0]), &(djac_dparam.entry(0, 0)), &(dmass_matrix_dparam.entry(0, 0)), flag);
			}
			else // residuals, Jacobian
			{
				functable->ParameterDerivative[functable->current_res_jac][paramindex](&eleminfo, shape_info, &(dres_dparam[0]), &(djac_dparam.entry(0, 0)), NULL, flag);
			}
		}
		else // Only residuals
		{
			functable->ParameterDerivative[functable->current_res_jac][paramindex](&eleminfo, shape_info, &(dres_dparam[0]), NULL, NULL, flag);
		}
	}

	// The main JIT-driven residual/Jacobian/mass-matrix assembly entry point, called from
	// oomph-lib's fill_in_contribution_to_* machinery (flag: 0=residuals only, 1=+Jacobian,
	// >=2=+mass matrix). Two special redirections take priority over normal assembly: if
	// __replace_RJM_by_param_deriv is set (used while computing derivatives w.r.t. a parameter
	// via finite differences elsewhere), delegates to fill_in_generic_dresidual_contribution_jit
	// instead; if enable_zeta_projection is set, assembles the (unrelated) zeta-projection
	// residuals instead of the physical equations. Otherwise prepares the shape buffer, resolves
	// hanging-node equation info, and calls the appropriate JIT-generated ResidualAndJacobian
	// function (steady/unsteady, hanging/non-hanging variant).
	void BulkElementBase::fill_in_generic_residual_contribution_jit(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian, oomph::DenseMatrix<double> &mass_matrix, unsigned flag)
	{
		if (__replace_RJM_by_param_deriv)
		{
			fill_in_generic_dresidual_contribution_jit(__replace_RJM_by_param_deriv, residuals, jacobian, mass_matrix, flag);
			return;
		}

		if (this->enable_zeta_projection)
		{
			residuals_for_zeta_projection(residuals, jacobian, flag);
			this->enable_zeta_projection=false;
			return;
		}

		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		if (functable->current_res_jac < 0)
			return;
		if (!this->ndof())
			return;

		if (!functable->ResidualAndJacobian[functable->current_res_jac])
			return;
		prepare_shape_buffer_for_integration(functable->shapes_required_ResJac[functable->current_res_jac], flag);
		shape_info->jacobian_size = jacobian.nrow();
		shape_info->mass_matrix_size = mass_matrix.nrow();
		bool has_hang = this->fill_hang_info_with_equations(functable->shapes_required_ResJac[functable->current_res_jac], this->shape_info, NULL);
		has_hang = true;				 // ASSUME ALWAYS HANGING!
		this->interpolate_hang_values(); // XXX This should be moved to somewhere else, after each update of any values
		// std::cout << "RESIDUAL LENGTH  " << residuals.size() << "  " << this->nexternal_data() <<  std::endl;

		JITFuncSpec_ResidualAndJacobian_FiniteElement func;
		const oomph::TimeStepper *tstepper = (this->nnode() ? this->node_pt(0)->time_stepper_pt() : this->internal_data_pt(0)->time_stepper_pt());
		if (tstepper->is_steady())
		{
			func = functable->ResidualAndJacobianSteady[functable->current_res_jac];
		}
		else
		{
			if (!has_hang)
				func = functable->ResidualAndJacobian_NoHang[functable->current_res_jac];
			else
				func = functable->ResidualAndJacobian[functable->current_res_jac];
		}

		if (flag)
		{
			if (flag >= 2) // residuals, Jacobian, Mass matrix
			{
				/* for (unsigned int i=0;i<mass_matrix.nrow();i++)
					 for (unsigned int j=0;j<mass_matrix.nrow();j++) if (mass_matrix(i,j)!=0.0) std::cout << "PREE " << mass_matrix(i,j) << std::endl ;
			*/
				func(&eleminfo, shape_info, &(residuals[0]), &(jacobian.entry(0, 0)), &(mass_matrix.entry(0, 0)), flag);
				/* for (unsigned int i=0;i<mass_matrix.nrow();i++)
					 for (unsigned int j=0;j<mass_matrix.nrow();j++) if (std::fabs(mass_matrix(i,j))>100.0) std::cout << "POST " << mass_matrix(i,j) << std::endl ;*/
			}
			else // residuals, Jacobian
			{
				func(&eleminfo, shape_info, &(residuals[0]), &(jacobian.entry(0, 0)), NULL, flag);
			}
		}
		else // Only residuals
		{
			func(&eleminfo, shape_info, &(residuals[0]), NULL, NULL, flag);
		}
		/*
		 std::cout << "C JACO " << std::endl;
		 for (unsigned int i=0;i<jacobian.nrow();i++)
		 {
			 for (unsigned int j=0;j<jacobian.ncol();j++) std::cout << "\t" << jacobian.entry(i,j) ;
			std::cout << std::endl;
		 }
		*/
	}

	// Debug helper: compares the analytical Hessian-vector product (Y,C) -> product against a
	// finite-difference approximation obtained by perturbing each dof and re-assembling the
	// Jacobian, to validate the JIT-generated analytical Hessian code. Not used in production
	// assembly; intended to be called interactively/from Python when debugging.
	void BulkElementBase::debug_hessian(std::vector<double> Y, std::vector<std::vector<double>> C, double epsilon)
	{
		if (Y.size() != this->ndof())
			throw_runtime_error("Y vector is wrong in size " + std::to_string(Y.size()) + "  vs.  " + std::to_string(this->ndof()));
		if (!C.size())
			throw_runtime_error("Empty C matrix");
		oomph::Vector<double> Ys(Y.size());
		for (unsigned int i = 0; i < Ys.size(); i++)
			Ys[i] = Y[i];
		oomph::DenseMatrix<double> Cs(C.size(), C[0].size());
		for (unsigned int iv = 0; iv < C.size(); iv++)
		{
			if (C[iv].size() != this->ndof())
				throw_runtime_error("C vector entry " + std::to_string(iv) + " has wrong size");
			for (unsigned int id = 0; id < this->ndof(); id++)
				Cs(iv, id) = C[iv][id];
		}
		oomph::DenseMatrix<double> anaprod(C.size(), this->ndof(), 0.0);
		this->fill_in_contribution_to_hessian_vector_products(Ys, Cs, anaprod);

		// Now FDing it
		oomph::DenseMatrix<double> fdprod(C.size(), this->ndof(), 0.0);
		oomph::Vector<double> dummy_res(this->ndof());
		oomph::DenseMatrix<double> jac_base(this->ndof());
		this->get_jacobian(dummy_res, jac_base);
		oomph::Vector<double *> dof_pt;
		this->dof_pt_vector(dof_pt);
		oomph::Vector<double> dofbackup(dof_pt.size());
		for (unsigned int i = 0; i < dof_pt.size(); i++)
			dofbackup[i] = *(dof_pt[i]);

		////////////////////

		const double FD_step = 1.0e-7;

		// We can now construct our multipliers
		// Prepare to scale
		double dof_length = 0.0;
		oomph::Vector<double> C_length(C.size(), 0.0);

		for (unsigned n = 0; n < this->ndof(); n++)
		{
			if (std::fabs(dofbackup[n]) > dof_length)
			{
				dof_length = std::fabs(dofbackup[n]);
			}
		}

		// C is assumed to have the same distribution as the dofs
		for (unsigned i = 0; i < C.size(); i++)
		{
			for (unsigned n = 0; n < this->ndof(); n++)
			{
				if (std::fabs(C[i][n]) > C_length[i])
				{
					C_length[i] = std::fabs(C[i][n]);
				}
			}
		}
		///////////////////////////////7
		// Form the multipliers
		oomph::Vector<double> C_mult(C.size(), 0.0);
		for (unsigned i = 0; i < C.size(); i++)
		{
			C_mult[i] = dof_length / C_length[i];
			C_mult[i] += FD_step;
			C_mult[i] *= FD_step;
		}

		for (unsigned i = 0; i < C.size(); i++)
		{
			for (unsigned n = 0; n < this->ndof(); n++)
			{
				*dof_pt[n] += C_mult[i] * C[i][n];
			}
			oomph::DenseMatrix<double> jac_C(this->ndof());
			this->get_jacobian(dummy_res, jac_C);
			for (unsigned n = 0; n < this->ndof(); n++)
			{
				*(dof_pt[n]) = dofbackup[n];
			}

			for (unsigned n = 0; n < this->ndof(); n++)
			{
				double prod_c = 0.0;
				for (unsigned m = 0; m < this->ndof(); m++)
				{
					prod_c += (jac_C(n, m) - jac_base(n, m)) * Y[m];
				}
				fdprod(i, n) += prod_c / C_mult[i];
			}
		}

		for (unsigned int iv = 0; iv < C.size(); iv++)
		{
			bool Cheader_written = false;
			for (unsigned int id = 0; id < this->ndof(); id++)
			{
				if (epsilon <= 0 || std::fabs(fdprod(iv, id) - anaprod(iv, id)) > epsilon)
				{
					if (!Cheader_written)
					{
						std::cout << "  FOR C VECTOR " << iv << " : ";
						for (unsigned k = 0; k < C[iv].size(); k++)
							std::cout << C[iv][k] << "  ";
						std::cout << std::endl;
						Cheader_written = true;
					}
					std::cout << "     COMPONENT " << id << " : FD: " << fdprod(iv, id) << " ANA: " << anaprod(iv, id) << " DELTA: " << std::fabs(fdprod(iv, id) - anaprod(iv, id)) << std::endl;
				}
			}
		}
	}

	// Assembles the full (dense, flattened as ndof x ndof*ndof) Hessian tensor d^2R/dU^2 by
	// calling fill_in_generic_hessian with a dummy Y=0 vector (so no vector-product contraction
	// happens) and flag=3 (request full Hessian assembly rather than a vector product).
	void BulkElementBase::assemble_hessian_tensor(oomph::DenseMatrix<double> &hbuffer)
	{
		oomph::DenseMatrix<double> dummy(this->ndof(),this->ndof()*this->ndof(),0.0);// For the mass matrix
		fill_in_generic_hessian(oomph::Vector<double>(this->ndof(),0.0), dummy, hbuffer, 3);
	}

	// Assembles both the residual Hessian (d^2R/dU^2) and the mass-matrix Hessian (d^2M/dU^2) as
	// full rank-3 tensors directly via the JIT-generated HessianVectorProduct function (called
	// with a NULL Y so it fills the full tensors instead of contracting with a vector). Requires
	// the code to have been JIT-compiled with analytic Hessians enabled.
   void BulkElementBase::assemble_hessian_and_mass_hessian(oomph::RankThreeTensor<double> & hbuffer,oomph::RankThreeTensor<double> & mbuffer)
   {
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		if (functable->current_res_jac < 0)
			return;
		if (!this->ndof())
			return;
		if (!functable->hessian_generated)
			throw_runtime_error("Tried to calculate an analytical Hessian, but the corresponding C code was not generated. Please call setup_for_stability_analysis(analytic_hessian=True) of the Problem instance before initialization of the Problem.");
		if (!functable->HessianVectorProduct[functable->current_res_jac])
			return;

      hbuffer.resize(this->ndof(),this->ndof(),this->ndof());
      hbuffer.initialise(0.0);
      mbuffer.resize(this->ndof(),this->ndof(),this->ndof());
      mbuffer.initialise(0.0);      
		prepare_shape_buffer_for_integration(functable->shapes_required_Hessian[functable->current_res_jac], 3);
		shape_info->jacobian_size = this->ndof();
		this->fill_hang_info_with_equations(functable->shapes_required_Hessian[functable->current_res_jac], this->shape_info, NULL);
		this->interpolate_hang_values(); // This should be done elsewhere
		JITFuncSpec_HessianVectorProduct_FiniteElement func = functable->HessianVectorProduct[functable->current_res_jac];

		func(&eleminfo, shape_info, NULL, &(mbuffer(0, 0,0)), &(hbuffer(0, 0,0)), 1, 3);		   
   }
   
	// Fill up the vector pair to connect integration points with respective integration points of old mesh.
	void BulkElementBase::prepare_zeta_interpolation(oomph::MeshAsGeomObject *mesh_as_geom){

		// Enable projection
		this->enable_zeta_projection=true;

		// Number of integration points.
      	const unsigned n_intpt = integral_pt()->nweight();

		// Element's dimension
		const unsigned int dim = this->dim();

		// Allocate storage for local coordinates
      	oomph::Vector<double> s(dim);

		// Loop through integration points.
		for (unsigned int ipt=0; ipt<n_intpt; ipt++){

			// Initialise vector of coordinates at integration point.
			oomph::Vector<double> zeta(dim, 0.0);

			// Local coordinates of the integration points.
			for (unsigned int i=0; i<dim; i++){
				s[i] = integral_pt()->knot(ipt,i);
				}

			// Coordinates of the integration points.
			FiniteElement::interpolated_zeta(s, zeta);
	
			// Local coordinates of the source element in base mesh.
			oomph::Vector<double> old_s(zeta.size(), 0.5 * (this->s_min() + this->s_max()));
			
			// Source element in base mesh.
			BulkElementBase *src_elem = NULL;

			// Geometrical object into which the source element will be stored.
			oomph::GeomObject *res_go = NULL;
			
			// Use locate_zeta function to identify the element of the base mesh 
			// as geometrical object in which the interpolated_x coordinate is at and
			// the local s coordinate corresponding to it.
			mesh_as_geom->locate_zeta(zeta, res_go, old_s, false);

			// Cast the element from Geometrical object into BulkElementMesh.
        	src_elem = dynamic_cast<BulkElementBase *>(res_go);

			// Update vector pair.
			this->coords_oldmesh[ipt].first=src_elem;
			this->coords_oldmesh[ipt].second=old_s;
		}
	};

	
	// Residuals passed to fill_in_generic_residual_contribution_jit for solving projection of coordinates and fields.
	void BulkElementBase::residuals_for_zeta_projection(oomph::Vector<double>& residuals, oomph::DenseMatrix<double>& jacobian, const unsigned& do_fill_jacobian){
		
		// Store element in variable.
		FiniteElement *elem = dynamic_cast<FiniteElement *>(this);

		// Element's dimension.
		unsigned dim = elem->dim();

		// Local coordinates.
		oomph::Vector<double> s(dim,0.0);

		// Number of nodes.
		unsigned n_node = this->nnode();
		// Number of positional dofs.
      	const unsigned n_position_type = this->nnodal_position_type();
		// Set the value of n_intpt.
		const unsigned n_intpt = integral_pt()->nweight();

		// Get projection time.
		unsigned t =this->projection_time;

		// Create a field map to loop through all fields in element.
		auto *code_instance = this->get_code_instance();
		auto *func_table = code_instance->get_func_table();
		std::vector<int> field_map;
		for (unsigned int si=0;si<func_table->num_present_dg_spaces;si++)
		{
			if (func_table->dg_spaces[si].numfields)
		    {
		     throw_runtime_error("Cannot interpolate discontinuous fields yet");
		    }
		}		
		for (unsigned int si=0;si<func_table->num_present_continuous_spaces;si++)
		{
			if (func_table->continuous_spaces[si].numfields-func_table->continuous_spaces[si].numfields_basebulk)
		    {
		     throw_runtime_error("Cannot interpolate interface fields yet");
		    }
		}
		

		
		field_map.resize(ncont_interpolated_values());
		for (unsigned int i = 0; i < field_map.size(); i++){field_map[i] = i;}

		// Loop over integration points.
		for (unsigned ipt=0; ipt<n_intpt;ipt++){

			// Get local coordinates at integration point.
			for(unsigned i=0;i<dim;i++){s[i] = integral_pt()->knot(ipt, i);}

			// Old element pointer.
			BulkElementBase *old_elem = coords_oldmesh[ipt].first;
			oomph::Vector<double> old_s = coords_oldmesh[ipt].second;

			// Shape functions.
			oomph::Shape psi(n_node,n_position_type);
			this->shape(s,psi);

			// Jacobian of mapping from local to global coordinates.
            double J = this->J_eulerian(s);

			// Get weight at ipt.
			double w = integral_pt()->weight(ipt);

			// Premultiply weights with Jacobian.
			double W = w * J;

			// Current position at current mesh.
			oomph::Vector<double> interpolated_zeta_curr(dim,0.0);
			oomph::Vector<double> interpolated_x_curr(dim,0.0);
			this->interpolated_zeta(s, interpolated_zeta_curr);
			this->interpolated_x(0, s, interpolated_x_curr);

			// Position in old element.
			oomph::Vector<double> interpolated_zeta_old(dim,0.0);
			oomph::Vector<double> interpolated_x_old(dim,0.0);
			old_elem->interpolated_zeta(old_s, interpolated_zeta_old);
			old_elem->interpolated_x(t, old_s, interpolated_x_old);

			// Initialise local equation and local unknown.
			int local_eqn=0;
			int local_unknown=0;

			// Loop through nodes.
			for(unsigned l=0;l<n_node;l++){
				
				// Loop through position types.
				for(unsigned k = 0; k < n_position_type; k++){

					//======= Fill residuals for coordinates =========//

					// Add the residuals for each coordinate's dimension. 
					for(unsigned i=0; i<dim; i++){

						// Get coordinate's equation number. 
							local_eqn = this->position_local_eqn(l, k, i);
						
							// If it is a degree of freedom.
							if(local_eqn >= 0){

								// For projection times>0, we project the x-coordinates for history values.
								// Otherwise, we use the zeta coordinates.
								if(t==0){							
									// Add residuals for zeta.
									residuals[local_eqn]+=(interpolated_zeta_curr[i]-interpolated_zeta_old[i]) * psi(l, k) * W;
								}
								else{
									// Add residuals for zeta.
									residuals[local_eqn]+=(interpolated_x_curr[i]-interpolated_x_old[i]) * psi(l, k) * W;
								}
							
						}

						// Get coordinate's equation number. 
						local_eqn = this->position_local_eqn(l, k, i);
						
						// If it is a degree of freedom.
						if(local_eqn >= 0){
							
							// Add residuals.
							residuals[local_eqn]+=(interpolated_zeta_curr[i]-interpolated_zeta_old[i]) * psi(l, k) * W;
						}

						// Calculate the jacobian
						if (do_fill_jacobian == 1)
						{
							for (unsigned l2 = 0; l2 < n_node; l2++)
							{
								// Loop over position dofs
								for (unsigned k2 = 0; k2 < n_position_type; k2++)
								{

									local_unknown = this->position_local_eqn(l2, k2, i);

									if (local_unknown >= 0)
									{
										//Add Jacobian
										jacobian(local_eqn, local_unknown) += psi(l2, k2) * psi(l, k) * W;
									}
								}	
							}
						}
					}
				}  // End of residuals for coordinates.

				
				//======= Fill residuals for fields =========//

				// Get interpolated values for current mesh.
				oomph::Vector<double> interpolated_values_curr;
				this->get_interpolated_values(0, s, interpolated_values_curr);

				// Get interpolated values for old mesh.
				oomph::Vector<double> interpolated_values_old;
				old_elem->get_interpolated_values(t, s, interpolated_values_old);

				// Loop through every field.
				for(unsigned field=0; field<field_map.size(); field++){
					
					// Get local equation number.
					local_eqn = elem->nodal_local_eqn(l, field);

					// If it is a degree of freedom.
					if(local_eqn >= 0){
						
						// Add residuals.
						residuals[local_eqn]+=(interpolated_values_curr[field]-interpolated_values_old[field]) * psi(l) * W;

					}

					// Calculate the jacobian
					if (do_fill_jacobian == 1)
					{
						for (unsigned l2 = 0; l2 < n_node; l2++)
						{
							// Loop over position dofs
							local_unknown = elem->nodal_local_eqn(l, field);

							if (local_unknown >= 0)
							{	
								//Add Jacobian
								jacobian(local_eqn, local_unknown) += psi(l2) * psi(l) * W;
							}	
						}
					}
				}
			}
		}
	}
   

	// Shared implementation behind fill_in_contribution_to_hessian_vector_products() and
	// assemble_hessian_tensor()/assemble_hessian_and_mass_hessian(): calls the JIT-generated
	// HessianVectorProduct function to contract the residual Hessian d^2R/dU^2 with the given
	// eigenvector Y and a set of directions C (flag selects vector-product vs. full-tensor mode,
	// see the JITFuncSpec_HessianVectorProduct_FiniteElement calling convention).
	void BulkElementBase::fill_in_generic_hessian(oomph::Vector<double> const &Y, oomph::DenseMatrix<double> &C, oomph::DenseMatrix<double> &product, unsigned flag)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		if (functable->current_res_jac < 0)
			return;
		if (!this->ndof())
			return;
		if (!functable->hessian_generated)
			throw_runtime_error("Tried to calculate an analytical Hessian, but the corresponding C code was not generated. Please call setup_for_stability_analysis(analytic_hessian=True) of the Problem instance before initialization of the Problem.");
		if (!functable->HessianVectorProduct[functable->current_res_jac])
			return;

		unsigned n_vec = C.nrow();
		unsigned n_var = Y.size();
		if (flag == 3)
			n_var = product.nrow();

		prepare_shape_buffer_for_integration(functable->shapes_required_Hessian[functable->current_res_jac], 3);
		shape_info->jacobian_size = n_var; // Storing the number of dofs now
										   //& shape_info->mass_matrix_size=n_vec; // Won't be used, but storing the numbers of vects
		// bool has_hang =
		this->fill_hang_info_with_equations(functable->shapes_required_Hessian[functable->current_res_jac], this->shape_info, NULL);
		// bool has_hang = true;				 // ASSUME ALWAYS HANGING!
		this->interpolate_hang_values(); // XXX This should be moved to somewhere else, after each update of any values
		JITFuncSpec_HessianVectorProduct_FiniteElement func = functable->HessianVectorProduct[functable->current_res_jac];

		// const double * Cs=&(const_cast<oomph::DenseMatrix<double>*>(&C)->entry(0,0)); // XXX: Dirty hack, but otherwise not possibility to call this
		func(&eleminfo, shape_info, &(Y[0]), &(C.entry(0, 0)), &(product.entry(0, 0)), n_vec, flag);
	}

	// oomph-lib hook: computes product = sum_dofs Hessian(Y) * C^T, i.e. the Hessian-vector
	// products needed for e.g. bifurcation tracking / stability analysis. Delegates to
	// fill_in_generic_hessian in plain (non-tensor) vector-product mode.
	void BulkElementBase::fill_in_contribution_to_hessian_vector_products(oomph::Vector<double> const &Y, oomph::DenseMatrix<double> const &C, oomph::DenseMatrix<double> &product)
	{
		oomph::DenseMatrix<double> Ccopy = C;
		this->fill_in_generic_hessian(Y, Ccopy, product, 0);
	}

	// Builds a human-readable name for each local dof/equation of this element, used for
	// debugging (e.g. printing which equation a Jacobian row/column corresponds to). Walks
	// through every field/space in the same order fill_element_info() uses (non-hanging Pos,
	// hanging Pos via masters, then continuous spaces, DG, DL, D0, ED0, ...), so the produced
	// names line up with the residual/Jacobian ordering.
	std::vector<std::string> BulkElementBase::get_dof_names(bool not_a_root_call)
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

	// Debug helper: recomputes the Jacobian via the (base oomph-lib) finite-difference path and
	// compares it entry-by-entry to the already-assembled analytical (JIT) `jacobian`, printing
	// every entry that differs by more than diff_eps (with dof names) to help track down bugs in
	// generated residual/Jacobian code. Optionally aborts (stop_on_jacobian_difference) with a
	// detailed dump of the element's dof/equation-number bookkeeping.
	void BulkElementBase::debug_analytical_jacobian(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian, double diff_eps)
	{
		oomph::Vector<double> fd_residuals(residuals.size(), 0.0);
//		std::cout << "DB NDOF " << this->ndof() << std::endl  << std::flush;
//		std::cout << "   J" << jacobian.nrow() << " x " << jacobian.ncol() << std::endl << std::flush;
		oomph::DenseMatrix<double> fd_jacobian(jacobian.nrow(), jacobian.ncol(), 0.0);
		if (codeinst->get_func_table()->missing_residual_assembly[codeinst->get_func_table()->current_res_jac])
		{
		    throw_runtime_error("The Jacobian of the residual "+std::string(codeinst->get_func_table()->res_jac_names[codeinst->get_func_table()->current_res_jac])+" cannot be calculated by finite differences, since the residual is not calculated at all.");
		}
		this->RefineableSolidElement::fill_in_contribution_to_jacobian(fd_residuals, fd_jacobian);
		//	this->fill_in_jacobian_from_lagragian_by_fd(fd_residuals,fd_jacobian);
		std::vector<std::string> dofnames = get_dof_names();
		bool header_written = false;
		for (unsigned int i = 0; i < jacobian.nrow(); i++)
		{
			for (unsigned int j = 0; j < jacobian.ncol(); j++)
			{
				double diff = fd_jacobian(i, j) - jacobian(i, j);
				diff = fabs(diff);
				if (diff > diff_eps)
				{
					if (!header_written)
					{
						std::cout << "DIFFERENCES IN JACOBIAN ndof=" << this->ndof() << std::endl;
						std::cout << "#I\tJ\tDOF_i\tDOF_j\tDIFF\tJana\tJfd\tRana_i\tRfd_i\tRana_j\tRfd_j" << std::endl;
						header_written = true;
					}
					std::cout << i << "\t" << j << "\t" << dofnames[i] << "\t" << dofnames[j] << "\t" << diff << "\t" << jacobian(i, j) << "\t" << fd_jacobian(i, j) << "\t" << residuals[i] << "\t" << fd_residuals[i] << "\t" << residuals[j] << "\t" << fd_residuals[j] << std::endl;
				}
			}
		}
		if (header_written && codeinst->get_func_table()->stop_on_jacobian_difference)
		{
			const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
			// Now a very detailed list:
			std::cout << "DOF LIST" << std::endl;
			for (unsigned int i = 0; i < dofnames.size(); i++)
			{
				std::cout << "\t" << i << "\t" << dofnames[i] << std::endl;
			}
			std::cout << "NODAL VALUE EQ BUFFER" << std::endl;
			for (unsigned int l = 0; l < this->nnode(); l++)
			{
				std::cout << "\t" << l;
				for (unsigned int i = 0; i < this->node_pt(l)->nvalue(); i++)
				{
					std::cout << "\t" << eleminfo.nodal_local_eqn[l][i];
				}
				std::cout << std::endl;
			}
			std::cout << "POS VALUE EQ BUFFER" << std::endl;
			for (unsigned int l = 0; l < this->nnode(); l++)
			{
				std::cout << "\t" << l;
				for (unsigned int i = 0; i < dynamic_cast<Node *>(this->node_pt(l))->variable_position_pt()->nvalue(); i++)
				{
					std::cout << "\t" << eleminfo.pos_local_eqn[l][i];
				}
				std::cout << "\t@\t";
				for (unsigned int i = 0; i < dynamic_cast<Node *>(this->node_pt(l))->variable_position_pt()->nvalue(); i++)
				{
					std::cout << "\t" << dynamic_cast<Node *>(this->node_pt(l))->variable_position_pt()->value(i);
				}
				std::cout << std::endl;
			}
			std::cout << "HANG INFO POS" << std::endl;
			for (unsigned int l = 0; l < this->nnode(); l++)
			{
				std::cout << "\t" << l << " nmst: " << shape_info->hanginfo_Pos[l].nummaster;
				for (int m = 0; m < shape_info->hanginfo_Pos[l].nummaster; m++)
				{
					std::cout << "\t\t weight:" << shape_info->hanginfo_Pos[l].masters[m].weight << "\t";
					for (unsigned int j = 0; j < eleminfo.nodal_dim; j++)
						std::cout << "\t" << shape_info->hanginfo_Pos[l].masters[m].local_eqn[j];
				}
				std::cout << std::endl;
			}

			for (unsigned int si=0;si<NUM_CONTINUOUS_SPACES;si++)
			{
				if (eleminfo.nnode_of_space[si])
				{
					std::cout << "HANG INFO " << functable->continuous_spaces[si].space_name << std::endl;
					for (unsigned int l = 0; l < eleminfo.nnode_of_space[si]; l++)
					{
						std::cout << "\t" << l << " nmst: " << shape_info->hanginfo_Cont[si][l].nummaster;
						for (int m = 0; m < shape_info->hanginfo_Cont[si][l].nummaster; m++)
						{
							std::cout << "\t\t weight:" << shape_info->hanginfo_Cont[si][l].masters[m].weight << "\t";
							for (unsigned int j = 0; j < functable->continuous_spaces[si].numfields; j++)
								std::cout << "\t" << shape_info->hanginfo_Cont[si][l].masters[m].local_eqn[j];
						}
						std::cout << std::endl;
					}
				}
			}

			if (functable->shapes_required_ResJac[functable->current_res_jac].bulk_shapes && dynamic_cast<InterfaceElementBase *>(this))
			{
				BulkElementBase *bel = dynamic_cast<BulkElementBase *>(dynamic_cast<InterfaceElementBase *>(this)->bulk_element_pt());
				std::cout << "BULK HANG INFO POS" << std::endl;
				for (unsigned int l = 0; l < bel->nnode(); l++)
				{
					std::cout << "\t" << l << " nmst: " << shape_info->bulk_shapeinfo->hanginfo_Pos[l].nummaster;
					for (int m = 0; m < shape_info->bulk_shapeinfo->hanginfo_Pos[l].nummaster; m++)
					{
						std::cout << "\t\t weight:" << shape_info->bulk_shapeinfo->hanginfo_Pos[l].masters[m].weight << "\t";
						for (unsigned int j = 0; j < bel->eleminfo.nodal_dim; j++)
							std::cout << "\t" << shape_info->bulk_shapeinfo->hanginfo_Pos[l].masters[m].local_eqn[j];
					}
					std::cout << std::endl;
				}

				for (unsigned int si=0;si<NUM_CONTINUOUS_SPACES;si++)
				{
					if (bel->eleminfo.nnode_of_space[si])
					{
						std::cout << "BULK HANG INFO " << functable->continuous_spaces[si].space_name << std::endl;
						for (unsigned int l = 0; l < bel->eleminfo.nnode_of_space[si]; l++)
						{
							std::cout << "\t" << l << " nmst: " << shape_info->bulk_shapeinfo->hanginfo_Cont[si][l].nummaster;
							for (int m = 0; m < shape_info->bulk_shapeinfo->hanginfo_Cont[si][l].nummaster; m++)
							{
								std::cout << "\t\t weight:" << shape_info->bulk_shapeinfo->hanginfo_Cont[si][l].masters[m].weight << "\t";
								for (unsigned int j = 0; j < functable->continuous_spaces[si].numfields; j++)
									std::cout << "\t" << shape_info->bulk_shapeinfo->hanginfo_Cont[si][l].masters[m].local_eqn[j];
							}
							std::cout << std::endl;
						}
					}
				}
			}

			InterfaceElementBase *ie = dynamic_cast<InterfaceElementBase *>(this);
			std::string prefix = "";
			while (ie)
			{
				prefix = prefix + "BULK_PARENT:";
				BulkElementBase *be = dynamic_cast<BulkElementBase *>(ie->bulk_element_pt());
				std::vector<std::string> pdofnames = be->get_dof_names();
				std::cout << "DOFS FOR " << prefix << std::endl;
				for (unsigned int i = 0; i < pdofnames.size(); i++)
				{
					std::cout << "\t" << i << "\t" << pdofnames[i] << std::endl;
				}
				ie = dynamic_cast<InterfaceElementBase *>(be);
			}

			throw_runtime_error("Mismatch in Jacobian in code: " + this->codeinst->get_code()->get_file_name());
		}
	}

	// Fills in the Jacobian columns corresponding to nodal position (Lagrangian/geometric) dofs
	// by finite-differencing the residuals with respect to each nodal coordinate in turn (used
	// when the JIT-generated code cannot or does not provide the position-Jacobian analytically,
	// e.g. moving-mesh problems with fd_position_jacobian set). Columns belonging to non-position
	// ("Lagrangian") dofs are left untouched here (is_lagrangian_dof is currently unused/disabled
	// via the commented-out block, so effectively every dof's row is updated for every perturbed
	// position dof). Perturbs one nodal coordinate at a time (looping master nodes for hanging
	// nodes), re-evaluates the residuals, and forms the FD column, restoring the coordinate
	// afterwards.
	void BulkElementBase::fill_in_jacobian_from_lagragian_by_fd(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian)
	{
		const unsigned n_node = this->nnode();
		if (n_node == 0)
		{
			return;
		}

		// Test if this is a complete finite difference loop
		//  const JITFuncSpec_Table_FiniteElement_t * functable=codeinst->get_func_table();

		update_before_solid_position_fd();
		const unsigned n_position_type = this->nnodal_position_type();
		const unsigned nodal_dim = this->nodal_dimension();
		const unsigned n_dof = this->ndof();
		oomph::Vector<double> newres(n_dof);
		const double fd_step = this->Default_fd_jacobian_step;
		int local_unknown = 0;

		std::vector<bool> is_lagrangian_dof(this->ndof(), false);

		/*
		  for(unsigned l=0;l<n_node;l++)
		   {
			oomph::Node* const local_node_pt = this->node_pt(l);
			if(local_node_pt->is_hanging()==false)
			 {
			  for(unsigned k=0;k<n_position_type;k++)
			   {
				for(unsigned i=0;i<nodal_dim;i++)
				 {
				  local_unknown = this->position_local_eqn(l,k,i);
				  if(local_unknown >= 0)
				   {
					 is_lagrangian_dof[local_unknown]=true;
				   }
				 }
			   }
			 }
			 else
			 {
			  oomph::HangInfo* hang_info_pt = local_node_pt->hanging_pt();
			  const unsigned n_master = hang_info_pt->nmaster();
			  for(unsigned m=0;m<n_master;m++)
			   {
				oomph::Node* const master_node_pt = hang_info_pt->master_node_pt(m);
				oomph::DenseMatrix<int> Position_local_eqn_at_node = this->local_position_hang_eqn(master_node_pt);
				for(unsigned k=0;k<n_position_type;k++)
				 {
				  for(unsigned i=0;i<nodal_dim;i++)
				   {
					local_unknown = Position_local_eqn_at_node(k,i);
					if(local_unknown >= 0)
					 {
						 is_lagrangian_dof[local_unknown]=true;
					 }
				   }
				 }
			   }

			 }
			}      //TODO: Bulk element external position data

		*/
		for (unsigned l = 0; l < n_node; l++)
		{
			oomph::Node *const local_node_pt = this->node_pt(l);
			if (local_node_pt->is_hanging() == false)
			{
				for (unsigned k = 0; k < n_position_type; k++)
				{
					for (unsigned i = 0; i < nodal_dim; i++)
					{
						local_unknown = this->position_local_eqn(l, k, i);
						if (local_unknown >= 0)
						{
							double *const value_pt = &(local_node_pt->x_gen(k, i));
							const double old_var = *value_pt;
							*value_pt += fd_step;
							//            local_node_pt->perform_auxiliary_node_update_fct();
							update_in_solid_position_fd(l);
							get_residuals(newres);
							for (unsigned m = 0; m < n_dof; m++)
							{
								if (!is_lagrangian_dof[m])
								{
									// std::cout << "PERTURBED RESIDUALS " << l << "  " << k << "  " << i << "  at m " << m << " is " << (newres[m] - residuals[m])/fd_step << " WRITING TO (" << m << ", " << local_unknown << ")" << std::endl;
									jacobian(m, local_unknown) = (newres[m] - residuals[m]) / fd_step;
								}
							}
							*value_pt = old_var;
							// local_node_pt->perform_auxiliary_node_update_fct();
							// reset_in_solid_position_fd(l);
						}
					}
				}
			}
			// Otherwise it's a hanging node
			else
			{
				oomph::HangInfo *hang_info_pt = local_node_pt->hanging_pt();
				const unsigned n_master = hang_info_pt->nmaster();
				for (unsigned m = 0; m < n_master; m++)
				{
					oomph::Node *const master_node_pt = hang_info_pt->master_node_pt(m);
					oomph::DenseMatrix<int> Position_local_eqn_at_node = this->local_position_hang_eqn(master_node_pt);
					for (unsigned k = 0; k < n_position_type; k++)
					{
						for (unsigned i = 0; i < nodal_dim; i++)
						{
							local_unknown = Position_local_eqn_at_node(k, i);
							if (local_unknown >= 0)
							{
								double *const value_pt = &(master_node_pt->x_gen(k, i));
								const double old_var = *value_pt;
								*value_pt += fd_step;
								 master_node_pt->perform_auxiliary_node_update_fct();
								update_in_solid_position_fd(l);
								get_residuals(newres);

								for (unsigned m = 0; m < n_dof; m++)
								{
									if (!is_lagrangian_dof[m])
										jacobian(m, local_unknown) = (newres[m] - residuals[m]) / fd_step;
								}

								*value_pt = old_var;
								// master_node_pt->perform_auxiliary_node_update_fct();
								// reset_in_solid_position_fd(l);
							}
						}
					}
				}
			} // End of hanging node case
		}	  // End of loop over nodes
     reset_after_solid_position_fd();
		this->interpolate_hang_values();
	}
	
	
	void BulkElementBase::update_in_solid_position_fd(const unsigned &i) // For FD with element_sizes, we have to update the element size buffer
	{
	 const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
	 if (functable->moving_nodes && (functable->shapes_required_ResJac[functable->current_res_jac].elemsize_Eulerian_cartesian || functable->shapes_required_ResJac[functable->current_res_jac].elemsize_Eulerian))
	 {
//	  std::cout << "UPDATE CALL" << std::endl;
	  this->fill_shape_info_element_sizes(functable->shapes_required_ResJac[functable->current_res_jac],shape_info,0);
	 }
	}

	// oomph-lib hook for residual + Jacobian assembly. Normally delegates to the JIT-generated
	// analytical assembly; if the code table requests position-Jacobian entries by finite
	// differences (fd_position_jacobian, for moving-mesh problems), those columns are patched in
	// afterwards via fill_in_jacobian_from_lagragian_by_fd(). If debug_jacobian_epsilon is set,
	// cross-checks the analytical Jacobian against a full FD one. If fd_jacobian is set for the
	// whole element (rather than just positions), falls back entirely to oomph-lib's generic FD
	// Jacobian.
	void BulkElementBase::fill_in_contribution_to_jacobian(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		if (!functable->fd_jacobian)
		{
			fill_in_generic_residual_contribution_jit(residuals, jacobian, oomph::GeneralisedElement::Dummy_matrix, 1);
			if (functable->moving_nodes && functable->fd_position_jacobian)
			{
				this->fill_in_jacobian_from_lagragian_by_fd(residuals, jacobian);
			}

			if (functable->debug_jacobian_epsilon != 0.0 && functable->current_res_jac>=0)
				debug_analytical_jacobian(residuals, jacobian, functable->debug_jacobian_epsilon);
		}
		else
		{
		   if (functable->current_res_jac<0) return;
		   if (functable->missing_residual_assembly[functable->current_res_jac])
		   {
		    throw_runtime_error("The Jacobian of the residual "+std::string(functable->res_jac_names[functable->current_res_jac])+" cannot be calculated by finite differences, since the residual is not calculated at all.");
		   }
			this->RefineableSolidElement::fill_in_contribution_to_jacobian(residuals, jacobian);
		}

		/*
			 std::vector<std::string> dofnames=get_dof_names();
			 for (unsigned int i=0;i<jacobian.nrow();i++)
			 {
			   double minv=0;
			   double maxv=0;
			   for (unsigned int j=0;j<jacobian.ncol();j++)
			   {
				if (jacobian(i,j)<minv) minv=jacobian(i,j);
				if (jacobian(i,j)>maxv) maxv=jacobian(i,j);
			   }
			   if (minv==0 && maxv==0)
			   {
				std::cout << "EMPTY JACOBIAN CONTRIBTUION IN ROW " << i << " corresponding to eq " << this->eqn_number(i) << "  which is " << dofnames[i] << std::endl;
				std::cout << "ALL DOFS ARE " << std::endl;
				for (unsigned int k=0;k<dofnames.size();k++) std::cout << "  " << k << "  " << dofnames[k] << std::endl;
				std::cout << "HANGING INFO " << std::endl;
				for (unsigned l=0;l<this->nnode();l++)
				{
				 if (this->node_pt(l)->is_hanging())
				 {
					  oomph::HangInfo* hang_info_pt = this->node_pt(l)->hanging_pt();
						const unsigned n_master = hang_info_pt->nmaster();
						std::cout << "  " << l << " master " << n_master << "  :  " ;
						for(unsigned m=0;m<n_master;m++)
						 {
						  oomph::Node* const master_node_pt = hang_info_pt->master_node_pt(m);
						  oomph::DenseMatrix<int> Position_local_eqn_at_node = this->local_position_hang_eqn(master_node_pt);
						 for(unsigned ii=0;ii<this->node_pt(l)->ndim();ii++)
						 {
							  std::cout << " " << Position_local_eqn_at_node(0,ii);
						  }
						 }
						 std::cout << std::endl;
				 }
				 else
				 {
					std::cout << "  " << l << " not hanging, eqs for direction: ";
						 for(unsigned ii=0;ii<this->node_pt(l)->ndim();ii++)
						 {
							  std::cout << " " << this->position_local_eqn(l,0,ii);
						  }
					std::cout << std::endl;
				 }
				}
				std::cout << "  N  EXTERNAL " << this->nexternal_data() << std::endl;
			   }
			 }
			*/
	}

	// oomph-lib hook for residual + Jacobian + mass-matrix assembly (used e.g. for eigenproblems).
	// FD mass matrices are not implemented (the fd_jacobian branch below is dead code, left in
	// for reference, and would throw before reaching it); the normal path always calls the
	// JIT-generated assembly with flag=2 (residuals + Jacobian + mass matrix).
	void BulkElementBase::fill_in_contribution_to_jacobian_and_mass_matrix(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian, oomph::DenseMatrix<double> &mass_matrix)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		if (functable->fd_jacobian)
		{
			throw_runtime_error("FD Mass matrix not implemented");
			//WARNING: This takes the analytic mass matrix
			codeinst->get_func_table()->fd_jacobian=false;
			fill_in_generic_residual_contribution_jit(residuals, jacobian, mass_matrix, 2);
			codeinst->get_func_table()->fd_jacobian=true;
			residuals.initialise(0.0);
			jacobian.initialise(0.0);
			fill_in_generic_residual_contribution_jit(residuals, jacobian, mass_matrix, 1);
			
		}
		fill_in_generic_residual_contribution_jit(residuals, jacobian, mass_matrix, 2);
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
	void BulkElementBase::assign_additional_local_eqn_numbers()
	{
		this->RefineableSolidElement::assign_additional_local_eqn_numbers();
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

	// Number of nodal values required at every node: same for all nodes of this element type
	// (the total number of "base bulk" field values, i.e. all continuous-space fields the code
	// generator laid out for this element, regardless of which node index n is asked about).
	unsigned BulkElementBase::required_nvalue(const unsigned &n) const
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		return functable->total_num_fields_basebulk;
	}

	// oomph-lib node-construction hooks (plain / with explicit timestepper, interior / boundary):
	// create a pyoomph::Node (or BoundaryNode) with the right Lagrangian/nodal dimensions and
	// enough values to hold every base-bulk field.
	oomph::Node *BulkElementBase::construct_node(const unsigned &n)
	{
		unsigned ntot = this->required_nvalue(n);
		//	 std::cout << "NLAGR " <<  this->nlagrangian() << "  " << this->nnodal_lagrangian_type() << std::endl;
		node_pt(n) = new Node(this->nlagrangian(), this->nnodal_lagrangian_type(), this->nodal_dimension(), this->nnodal_position_type(), ntot);
		return node_pt(n);
	}

	oomph::Node *BulkElementBase::construct_node(const unsigned &n, oomph::TimeStepper *const &time_stepper_pt)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		unsigned ntot = required_nvalue(n);
		//		 		 std::cout << "NLAGR " <<  this->nlagrangian() << "  " << this->nnodal_lagrangian_type() << std::endl;
		node_pt(n) = new Node(time_stepper_pt, this->nlagrangian(), this->nnodal_lagrangian_type(), this->nodal_dimension(), this->nnodal_position_type(), ntot);
		return node_pt(n);
	}

	oomph::Node *BulkElementBase::construct_boundary_node(const unsigned &n)
	{	
		unsigned ntot = required_nvalue(n);
		node_pt(n) = new BoundaryNode(this->nlagrangian(), this->nnodal_lagrangian_type(), this->nodal_dimension(), this->nnodal_position_type(), ntot);
		return node_pt(n);
	}

	oomph::Node *BulkElementBase::construct_boundary_node(const unsigned &n, oomph::TimeStepper *const &time_stepper_pt)
	{		
		unsigned ntot = required_nvalue(n);
		node_pt(n) = new BoundaryNode(time_stepper_pt, this->nlagrangian(), this->nnodal_lagrangian_type(), this->nodal_dimension(), this->nnodal_position_type(), ntot);
		return node_pt(n);
	}


	// Number of continuously-interpolated nodal values per node (same quantity as
	// required_nvalue(), exposed under the name oomph-lib's projection/interpolation code uses).
	unsigned BulkElementBase::ncont_interpolated_values() const
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		return functable->total_num_fields_basebulk;
	}

	oomph::Vector<double> BulkElementBase::get_midpoint_s() // Set s=[0.5*(smin+smax), ... ] (but modified e.g. for tris)
	{
		return oomph::Vector<double>(this->dim(), 0.5 * (this->s_min() + this->s_max()));
	}

	// Evaluates a "local expression" at the element's midpoint local coordinate.
	double BulkElementBase::eval_local_expression_at_midpoint(unsigned index)
	{
		oomph::Vector<double> s = this->get_midpoint_s();
		return eval_local_expression_at_s(index, s);
	}

	// Creates a brand-new (not connected to the mesh's node array) Node at local coordinate s,
	// with its Lagrangian/Eulerian coordinates and history values obtained by interpolating this
	// element's own fields there. Used e.g. to sample the solution at an arbitrary point without
	// requiring an actual mesh node to exist there.
    pyoomph::Node *BulkElementBase::create_interpolated_node(const oomph::Vector<double> &s,bool as_boundary_node)
    {
		if (this->nnode()==0) return 0;
		pyoomph::Node *res;
		if (as_boundary_node)
		{
		 	res= new pyoomph::BoundaryNode(this->node_pt(0)->time_stepper_pt(),this->nlagrangian(), this->nnodal_lagrangian_type(), this->nodal_dimension(), this->nnodal_position_type(), this->required_nvalue(0));
		}
		else
		{
			res= new pyoomph::Node(this->node_pt(0)->time_stepper_pt(),this->nlagrangian(), this->nnodal_lagrangian_type(), this->nodal_dimension(), this->nnodal_position_type(), this->required_nvalue(0));	
		}
		

		oomph::Vector<double> xibuff(this->lagrangian_dimension(),0.0);
		this->interpolated_xi(s,xibuff);	
		for (unsigned i = 0; i < this->lagrangian_dimension(); i++) res->xi(i) = xibuff[i];

		for (unsigned ti = 0; ti < res->time_stepper_pt()->ntstorage(); ti++)
		{
			oomph::Vector<double> xbuff(this->nodal_dimension(),0.0);
			this->interpolated_x(ti,s,xbuff);	
			for (unsigned i = 0; i < this->nodal_dimension(); i++)
				res->x(ti, i) = xbuff[i];

			oomph::Vector<double> vbuff(res->nvalue(),0.0);
			this->get_interpolated_values(ti,s,vbuff);
			for (unsigned int i=0;i<vbuff.size();i++) res->set_value(ti,i,vbuff[i]);
			
		}
        
		return res;
    }

    oomph::Vector<double> BulkElementBase::get_Eulerian_midpoint_from_local_coordinate() // Set s=[0.5*(smin+smax), ... ] and evaluate the position
	{
		oomph::Vector<double> res(this->nodal_dimension(), 0.0);
		if (this->nnode() == 1)
		{
			for (unsigned int i = 0; i < this->nodal_dimension(); i++)
				res[i] = this->node_pt(0)->x(i);
			return res;
		}
		oomph::Vector<double> s = this->get_midpoint_s();
		this->interpolated_x(s, res);
		return res;
	}

	oomph::Vector<double> BulkElementBase::get_Lagrangian_midpoint_from_local_coordinate() // Set s=[0.5*(smin+smax), ... ] and evaluate the position
	{
		oomph::Vector<double> res(this->nlagrangian(), 0.0);
		if (this->nnode() == 1)
		{
			for (unsigned int i = 0; i < this->nlagrangian(); i++)
				res[i] = dynamic_cast<pyoomph::Node *>(this->node_pt(0))->xi(i);
			return res;
		}
		oomph::Vector<double> s = this->get_midpoint_s();
	  oomph::Shape psi(this->nnode());
	  this->shape(s,psi);
	  const unsigned n_lagrangian = dynamic_cast<pyoomph::Node *>(this->node_pt(0))->nlagrangian();
	  for(unsigned i=0;i<n_lagrangian;i++)
		{
		 res[i] = 0.0;
		 for(unsigned l=0;l<this->nnode();l++) 
		  {
		     res[i] += lagrangian_position_gen(l,0,i)*psi(l);		   
		  }
		}
				
		return res;
	}

	// Called by oomph-lib's tree-based mesh refinement before a son element is otherwise set up:
	// makes sure the new (son) element has a code instance, inheriting it from its father since
	// sons are constructed generically without going through the normal Python-driven creation.
	void BulkElementBase::pre_build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt)
	{
		if (!this->codeinst)
		{
			BulkElementBase *cast_father_element_pt = dynamic_cast<BulkElementBase *>(this->father_element_pt());
			if (!cast_father_element_pt)
			{
				throw_runtime_error("Trying to build an element without a code instance during pre_build...");
			}
			else
				this->codeinst = cast_father_element_pt->codeinst;
		}
	}

	// Called by oomph-lib's tree-based mesh refinement right after a son element has been created
	// from a father (bisection/quadsection/octsection) and its nodes set up, to transfer this
	// element type's own data down to the son: initial element size (halved/quartered/eighthed
	// depending on binary/quad/octree), non-nodal coordinates for elements whose nodal dimension
	// exceeds their local dimension, and every non-continuous field storage (DG, DL, D0) which
	// oomph-lib's generic node-based interpolation cannot handle. DG and constant/D0 fields are
	// just evaluated at the son's local coordinate mapped into the father; DL storage (assumed to
	// hold a constant + linear-gradient representation, one value plus one gradient component per
	// spatial direction) is transferred using the standard oomph-lib "restrict to child octant"
	// formulas (value +/- 0.5*gradient per split direction, gradient halved).
	void BulkElementBase::further_build()
	{

		if (!this->tree_pt()->father_pt())
		{
			throw_runtime_error("Try to split an element, but found not father...");
			this->ensure_external_data();
			return;
		}
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		BulkElementBase *father = dynamic_cast<BulkElementBase *>(this->tree_pt()->father_pt()->object_pt());
		if (!father)
			throw_runtime_error("Try to split an element, but found not father...");

		oomph::QuadTree *quadtree_pt = dynamic_cast<oomph::QuadTree *>(Tree_pt);
		oomph::BinaryTree *binarytree_pt = dynamic_cast<oomph::BinaryTree *>(Tree_pt);
		oomph::OcTree *octree_pt = dynamic_cast<oomph::OcTree *>(Tree_pt);
		int nsons = this->tree_pt()->father_pt()->nsons();

		this->set_nlagrangian_and_ndim(this->codeinst->get_func_table()->lagr_dim, this->codeinst->get_func_table()->nodal_dim);

		for (unsigned int i = 0; i < ninternal_data(); i++)
			internal_data_pt(i)->set_time_stepper(node_pt(0)->time_stepper_pt(), false);


		if (binarytree_pt)
		{
			initial_cartesian_nondim_size = 0.5 * father->initial_cartesian_nondim_size;			
		}
		else if (quadtree_pt)
		{
			initial_cartesian_nondim_size = 0.25 * father->initial_cartesian_nondim_size;
		}
		else if (octree_pt)
		{
			initial_cartesian_nondim_size = 0.125 * father->initial_cartesian_nondim_size;
		}

		for (unsigned t = 0; t < node_pt(0)->time_stepper_pt()->ntstorage(); t++)
		{
			if (this->nodal_dimension() != this->dim())
			{
				for (unsigned int l = 0; l < this->nnode(); l++)
				{
					// We need to map the nodes correctly
					oomph::Vector<double> sfather;
					this->get_nodal_s_in_father(l, sfather);
					oomph::Vector<double> x_prev(this->nodal_dimension());
					BulkElementBase *father_el_pt = dynamic_cast<BulkElementBase *>(tree_pt()->father_pt()->object_pt());
					father_el_pt->get_x(t, sfather, x_prev);
					for (unsigned int i = this->dim(); i < this->nodal_dimension(); i++)
					{
						//  std::cout << "BUILD NODE " << l << " has position " << i << " of " << this->node_pt(l)->x(t,i) << " at time index "  << t << std::endl;
						this->node_pt(l)->x(t, i) = x_prev[i]; // TODO: Also lagrangian?
					}
				}
			}

			const std::vector<std::vector<unsigned>> &space_node_to_elem = this->get_nodal_space_index_to_element_index_map();
			//DG 	
			for (unsigned int i_space=0;i_space<functable->num_present_dg_spaces;i_space++)
			{
				auto * space_info=functable->present_dg_spaces[i_space];
				for (unsigned l=0;l<this->get_eleminfo()->nnode_of_space[space_info->space_index];l++)
				{
					oomph::Vector<double> sfather,father_data;				
					this->get_nodal_s_in_father(space_node_to_elem[space_info->space_index][l], sfather);
					father->get_DG_fields_at_s(space_info->space_index,t,sfather,father_data);
					for (unsigned iindex=0;iindex<space_info->numfields;iindex++)
					{
						this->internal_data_pt(space_info->internal_offset_new+iindex)->set_value(t,l,father_data[iindex]);
					}
				}
			}


			//DL and D0
			unsigned DL_offset=functable->info_DL.internal_offset_new;
			for (unsigned int iindex = DL_offset; iindex < DL_offset+functable->info_DL.numfields; iindex++)
			{
				if (binarytree_pt)
				{
					using namespace oomph::BinaryTreeNames;
					int son_type = binarytree_pt->son_type();
					if (son_type == L)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) - 0.5 * father->internal_data_pt(iindex)->value(t, 1));
					else if (son_type == R)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(0) + 0.5 * father->internal_data_pt(iindex)->value(t, 1));
					for (unsigned j = 1; j < internal_data_pt(iindex)->nvalue(); j++)
						internal_data_pt(iindex)->set_value(t, j, father->internal_data_pt(iindex)->value(t, j) / 2);
				}
				else if (quadtree_pt)
				{
					using namespace oomph::QuadTreeNames;
					int son_type = quadtree_pt->son_type();
					for (unsigned j = 1; j < internal_data_pt(iindex)->nvalue(); j++)
						internal_data_pt(iindex)->set_value(t, j, father->internal_data_pt(iindex)->value(t, j) / 2);
					double sx = 0.5 * father->internal_data_pt(iindex)->value(t, 1);
					double sy = 0.5 * father->internal_data_pt(iindex)->value(t, 2);
					if (son_type == SW)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) - sx - sy);
					else if (son_type == NW)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) - sx + sy);
					else if (son_type == SE)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) + sx - sy);
					else if (son_type == NE)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) + sx + sy);
				}
				else if (octree_pt)
				{
					using namespace oomph::OcTreeNames;
					int son_type = octree_pt->son_type();
					for (unsigned j = 1; j < internal_data_pt(iindex)->nvalue(); j++)
						internal_data_pt(iindex)->set_value(t, j, father->internal_data_pt(iindex)->value(t, j) / 2);
					double sx = 0.5 * father->internal_data_pt(iindex)->value(t, 1);
					double sy = 0.5 * father->internal_data_pt(iindex)->value(t, 2);
					double sz = 0.5 * father->internal_data_pt(iindex)->value(t, 3);

					if (son_type == LDB)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) - sx - sy - sz);
					else if (son_type == RDB)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) + sx - sy - sz);
					else if (son_type == LUB)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) - sx + sy - sz);
					else if (son_type == RUB)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) + sx + sy - sz);
					else if (son_type == LDF)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) - sx - sy + sz);
					else if (son_type == RDF)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) + sx - sy + sz);
					else if (son_type == LUF)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) - sx + sy + sz);
					else if (son_type == RUF)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) + sx + sy + sz);					
				}
				else
					internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0)); // TODO: Correct interpolation here, i.e. e.g for Triangle and 3d
			}

			unsigned iD0=0;
			for (unsigned int iindex = functable->info_DL.numfields+DL_offset; iindex < DL_offset+functable->info_DL.numfields + functable->info_D0.numfields; iindex++) // D0 fields
			{
				double factor = 1;
				if (functable->discontinuous_refinement_exponents[functable->info_D0.buffer_offset_basebulk + iD0] != 0.0) // TODO: Consider on DL as well
				{
					factor = pow(nsons, -functable->discontinuous_refinement_exponents[functable->info_D0.buffer_offset_basebulk + iD0]);
				}
				internal_data_pt(iindex)->set_value(t, 0, factor * father->internal_data_pt(iindex)->value(t, 0));
				iD0++;
			}
		}
		this->set_integration_scheme(father->integral_pt());
		this->ensure_external_data();
	}

	// Called by oomph-lib's tree-based mesh refinement when four/two/eight son elements are
	// merged back into their father during unrefinement: reconstructs the father's non-nodal
	// field storage (DG, DL, D0) from the sons' data, since oomph-lib's generic node-based
	// unrefinement cannot handle these. DG fields are averaged at coincident node locations
	// (accumulated then divided by the number of contributing sons); DL fields are rebuilt as
	// value + gradient (average of son values, and centered-difference slopes between sons on
	// opposite sides) -- noted as non-conservative and ignoring axisymmetric weighting; D0
	// fields are simple averages (optionally re-scaled by discontinuous_refinement_exponents for
	// fields that should scale differently under mesh coarsening, e.g. densities vs. totals).
	// TODO: Split this into the particular elements
	void BulkElementBase::rebuild_from_sons(oomph::Mesh *&mesh_pt)
	{

		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		// Quad tree
		oomph::QuadTree *quadtree_pt = dynamic_cast<oomph::QuadTree *>(Tree_pt);
		oomph::BinaryTree *binarytree_pt = dynamic_cast<oomph::BinaryTree *>(Tree_pt);
		oomph::OcTree *octree_pt = dynamic_cast<oomph::OcTree *>(Tree_pt);
		if (functable->integration_order)
		{
			this->set_integration_order(functable->integration_order);
		}

		// DG fields	
		const std::vector<std::vector<unsigned>> &this_space_node_to_elem = this->get_nodal_space_index_to_element_index_map();	
		const std::vector<std::vector<int>> & this_elem_to_space_nodal_index = this->get_element_index_to_nodal_space_index_map();

		for (unsigned int i_space=0;i_space<functable->num_present_dg_spaces;i_space++)
		{
			auto * space_info=functable->present_dg_spaces[i_space];
			for (unsigned t = 0; t < node_pt(0)->time_stepper_pt()->ntstorage(); t++)
			{
				if (space_info->numfields_new)
				{
					const unsigned Nn=this->get_eleminfo()->nnode_of_space[space_info->space_index];
					std::vector<double> denom(Nn,0.0);
					for (unsigned int iindex = space_info->internal_offset_new; iindex < space_info->internal_offset_new+space_info->numfields_new; iindex++) 
					{
						for (unsigned int in=0;in<Nn;in++) this->internal_data_pt(iindex)->set_value(t,in,0.0); //Set to 0
					}
					for (unsigned ison = 0; ison < this->required_nsons(); ison++)
					{
						BulkElementBase* son=dynamic_cast<BulkElementBase*>(Tree_pt->son_pt(ison)->object_pt());
						const std::vector<unsigned> &son_space_node_to_elem = son->get_nodal_space_index_to_element_index_map()[space_info->space_index];
						for (unsigned int in=0;in<Nn;in++)
						{
							oomph::Vector<double> s;				
							son->get_nodal_s_in_father(son_space_node_to_elem[in], s);
							oomph::Node * my_node=this->get_node_at_local_coordinate(s);
							if (my_node)
							{
								int nn=this->get_node_number(my_node);
								if (nn>=0)
								{
									nn=this_elem_to_space_nodal_index[space_info->space_index][nn];
									if (nn>=0)
									{
										for (unsigned int iindex = space_info->internal_offset_new; iindex < space_info->internal_offset_new+space_info->numfields_new; iindex++) 
										{
											double sonval=son->internal_data_pt(iindex)->value(t, in);
											this->internal_data_pt(iindex)->set_value(t,nn,this->internal_data_pt(iindex)->value(t,nn)+sonval); //Accumulate the son values
										}
										denom[nn]+=1.0;
									}
								}
							}
						}
					}
					for (unsigned int iindex = space_info->internal_offset_new; iindex < space_info->internal_offset_new+space_info->numfields_new; iindex++) 
					{
						for (unsigned int in=0;in<Nn;in++) 
						{
							if (denom[in]<0.1) throw_runtime_error("Should not happen");
							this->internal_data_pt(iindex)->set_value(t,in,this->internal_data_pt(iindex)->value(t,in)/denom[in]); 
						}
					}
				}
			}
		}

		// DL and D0 fields and initial size
		if (quadtree_pt)
		{
			using namespace oomph::QuadTreeNames;
			for (unsigned t = 0; t < node_pt(0)->time_stepper_pt()->ntstorage(); t++)
			{
				for (unsigned int iindex = functable->info_DL.internal_offset_new; iindex < functable->info_DL.internal_offset_new+functable->info_DL.numfields; iindex++) // DL fields
				{
					// XXX TODO: Allow for other interpolation methods. In particular, this does not conserve (which does not matter for e.g. pressure) and does not consider axisymmetry
					double av = 0.0;
					for (unsigned ison = 0; ison < 4; ison++)
					{
						av += quadtree_pt->son_pt(ison)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					}
					internal_data_pt(iindex)->set_value(t, 0, 0.25 * av);
					double slope1 = quadtree_pt->son_pt(SE)->object_pt()->internal_data_pt(iindex)->value(t, 0) - quadtree_pt->son_pt(SW)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					double slope2 = quadtree_pt->son_pt(NE)->object_pt()->internal_data_pt(iindex)->value(t, 0) - quadtree_pt->son_pt(NW)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					internal_data_pt(iindex)->set_value(t, 1, 0.5 * (slope1 + slope2));
					slope1 = quadtree_pt->son_pt(NE)->object_pt()->internal_data_pt(iindex)->value(t, 0) - quadtree_pt->son_pt(SE)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					slope2 = quadtree_pt->son_pt(NW)->object_pt()->internal_data_pt(iindex)->value(t, 0) - quadtree_pt->son_pt(SW)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					internal_data_pt(iindex)->set_value(t, 2, 0.5 * (slope1 + slope2));
				}
				for (unsigned int iindex = functable->info_D0.internal_offset_new; iindex < functable->info_D0.internal_offset_new + functable->info_D0.numfields; iindex++) // D0 fields
				{
					// XXX TODO: Allow for other interpolation methods. In particular, this does not conserve (which does not matter for e.g. pressure) and does not consider axisymmetry
					// TODO: Time history loop!
					double av = 0.0;
					for (unsigned ison = 0; ison < 4; ison++)
					{
						av += quadtree_pt->son_pt(ison)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					}
					double avg_factor = 0.25;
					if (functable->discontinuous_refinement_exponents[functable->info_D0.buffer_offset_basebulk + iindex-functable->info_D0.internal_offset_new] != 0.0) // TODO: Consider on DL as well
					{
						avg_factor = avg_factor * pow(avg_factor, -functable->discontinuous_refinement_exponents[functable->info_D0.buffer_offset_basebulk + iindex-functable->info_D0.internal_offset_new]);
					}
					internal_data_pt(iindex)->set_value(t, 0, avg_factor * av);
				}
			}
			initial_cartesian_nondim_size = 0;
			for (unsigned ison = 0; ison < 4; ison++)
			{
				initial_cartesian_nondim_size += dynamic_cast<BulkElementBase *>(quadtree_pt->son_pt(ison)->object_pt())->initial_cartesian_nondim_size;
			}
			// std::cout << "REBUILT FROM SONS " << dynamic_cast<oomph::RefineableElement*>(this)->macro_elem_pt() << std::endl;
		}
		else if (binarytree_pt)
		{
			using namespace oomph::BinaryTreeNames;
			for (unsigned t = 0; t < node_pt(0)->time_stepper_pt()->ntstorage(); t++)
			{
				for (unsigned int iindex = functable->info_DL.internal_offset_new; iindex < functable->info_DL.internal_offset_new+functable->info_DL.numfields; iindex++) // DL fields
				{
					// XXX TODO: Allow for other interpolation methods. In particular, this does not conserve (which does not matter for e.g. pressure) and does not consider axisymmetry
					double av = 0.0;
					for (unsigned ison = 0; ison < 2; ison++)
					{
						av += binarytree_pt->son_pt(ison)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					}
					internal_data_pt(iindex)->set_value(t, 0, 0.5 * av);
					double slope1 = binarytree_pt->son_pt(R)->object_pt()->internal_data_pt(iindex)->value(t, 0) - binarytree_pt->son_pt(L)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					internal_data_pt(iindex)->set_value(t, 1, slope1);
				}
				for (unsigned int iindex = functable->info_D0.internal_offset_new; iindex < functable->info_D0.internal_offset_new + functable->info_D0.numfields; iindex++) // D0 fields
				{
					// XXX TODO: Allow for other interpolation methods. In particular, this does not conserve (which does not matter for e.g. pressure) and does not consider axisymmetry
					double av = 0.0;
					for (unsigned ison = 0; ison < 2; ison++)
					{
						av += binarytree_pt->son_pt(ison)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					}
					double avg_factor = 0.5;
					if (functable->discontinuous_refinement_exponents[functable->info_D0.buffer_offset_basebulk + iindex-functable->info_D0.internal_offset_new] != 0.0) // TODO: Consider on DL as well
					{
						avg_factor = avg_factor * pow(avg_factor, -functable->discontinuous_refinement_exponents[functable->info_D0.buffer_offset_basebulk + iindex-functable->info_D0.internal_offset_new]);
					}
					internal_data_pt(iindex)->set_value(t, 0, avg_factor * av);
				}
			}
			initial_cartesian_nondim_size = 0;
			for (unsigned ison = 0; ison < 2; ison++)
			{
				initial_cartesian_nondim_size += dynamic_cast<BulkElementBase *>(binarytree_pt->son_pt(ison)->object_pt())->initial_cartesian_nondim_size;
			}
		}
		else if (octree_pt)
		{
			using namespace oomph::OcTreeNames;
			for (unsigned t = 0; t < node_pt(0)->time_stepper_pt()->ntstorage(); t++)
			{
				for (unsigned int iindex = functable->info_DL.internal_offset_new; iindex < functable->info_DL.internal_offset_new+functable->info_DL.numfields; iindex++) // DL fields
				{
					// XXX TODO: Allow for other interpolation methods. In particular, this does not conserve (which does not matter for e.g. pressure) and does not consider axisymmetry
					double av = 0.0;
					for (unsigned ison = 0; ison < 8; ison++)
					{
						av += octree_pt->son_pt(ison)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					}
					internal_data_pt(iindex)->set_value(t, 0, 0.125 * av);

					double slope1 = octree_pt->son_pt(RDB)->object_pt()->internal_data_pt(iindex)->value(t, 0) - octree_pt->son_pt(LDB)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					double slope2 = octree_pt->son_pt(RUB)->object_pt()->internal_data_pt(iindex)->value(t, 0) - octree_pt->son_pt(LUB)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					double slope3 = octree_pt->son_pt(RDF)->object_pt()->internal_data_pt(iindex)->value(t, 0) - octree_pt->son_pt(LDF)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					double slope4 = octree_pt->son_pt(RUF)->object_pt()->internal_data_pt(iindex)->value(t, 0) - octree_pt->son_pt(LUF)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					internal_data_pt(iindex)->set_value(t, 1, 0.25 * (slope1 + slope2 + slope3 + slope4));

					slope1 = octree_pt->son_pt(LUB)->object_pt()->internal_data_pt(iindex)->value(t, 0) - octree_pt->son_pt(LDB)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					slope2 = octree_pt->son_pt(RUB)->object_pt()->internal_data_pt(iindex)->value(t, 0) - octree_pt->son_pt(RDB)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					slope3 = octree_pt->son_pt(LUF)->object_pt()->internal_data_pt(iindex)->value(t, 0) - octree_pt->son_pt(LDF)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					slope4 = octree_pt->son_pt(RUF)->object_pt()->internal_data_pt(iindex)->value(t, 0) - octree_pt->son_pt(RDF)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					internal_data_pt(iindex)->set_value(t, 2, 0.25 * (slope1 + slope2 + slope3 + slope4));

					slope1 = octree_pt->son_pt(LDF)->object_pt()->internal_data_pt(iindex)->value(t, 0) - octree_pt->son_pt(LDB)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					slope2 = octree_pt->son_pt(RDF)->object_pt()->internal_data_pt(iindex)->value(t, 0) - octree_pt->son_pt(RDB)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					slope3 = octree_pt->son_pt(LUF)->object_pt()->internal_data_pt(iindex)->value(t, 0) - octree_pt->son_pt(LUB)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					slope4 = octree_pt->son_pt(RUF)->object_pt()->internal_data_pt(iindex)->value(t, 0) - octree_pt->son_pt(RUB)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					internal_data_pt(iindex)->set_value(t, 3, 0.25 * (slope1 + slope2 + slope3 + slope4));
				}
				for (unsigned int iindex = functable->info_D0.internal_offset_new; iindex < functable->info_D0.internal_offset_new + functable->info_D0.numfields; iindex++) // D0 fields
				{
					// XXX TODO: Allow for other interpolation methods. In particular, this does not conserve (which does not matter for e.g. pressure) and does not consider axisymmetry
					// TODO: Time history loop!
					double av = 0.0;
					for (unsigned ison = 0; ison < 8; ison++)
					{
						av += octree_pt->son_pt(ison)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					}
					double avg_factor = 0.125;
					if (functable->discontinuous_refinement_exponents[functable->info_D0.buffer_offset_basebulk + iindex-functable->info_D0.internal_offset_new] != 0.0) // TODO: Consider on DL as well
					{
						avg_factor = avg_factor * pow(avg_factor, -functable->discontinuous_refinement_exponents[functable->info_D0.buffer_offset_basebulk + iindex-functable->info_D0.internal_offset_new]);
					}
					internal_data_pt(iindex)->set_value(t, 0, avg_factor * av);
				}
			}
			initial_cartesian_nondim_size = 0;
			for (unsigned ison = 0; ison < 8; ison++)
			{
				initial_cartesian_nondim_size += dynamic_cast<BulkElementBase *>(octree_pt->son_pt(ison)->object_pt())->initial_cartesian_nondim_size;
			}
		}

		else
			throw_runtime_error("IMPLEMENT");
	}

	// Paraview output hooks inherited from oomph-lib are not used by pyoomph (output goes through
	// its own VTU/plotting machinery instead); left unimplemented on purpose.
	std::string BulkElementBase::scalar_name_paraview(const unsigned &i) const
	{
		throw_runtime_error("NOT IMPLEMENTED");
	}

	// Looks up the nodal value index of a base-bulk continuous field by its generated name.
	// Returns -1 if no such field exists (e.g. it lives in a non-nodal/discontinuous space).
	int BulkElementBase::get_nodal_index_by_name(oomph::Node *n, std::string fieldname)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();

		for (unsigned int si=0;si<functable->num_present_continuous_spaces;si++)
		{
			for (unsigned i = 0; i < functable->present_continuous_spaces[si]->numfields_basebulk; i++)
			{
				if (std::string(functable->present_continuous_spaces[si]->fieldnames[i]) == fieldname) return i+functable->present_continuous_spaces[si]->nodal_offset_basebulk;
			}
		}		
		return -1;
	}

	unsigned BulkElementBase::nscalar_paraview() const
	{
		throw_runtime_error("NOT IMPLEMENTED");
	}

	void BulkElementBase::scalar_value_paraview(std::ofstream &file_out, const unsigned &i, const unsigned &nplot) const
	{
		throw_runtime_error("NOT IMPLEMENTED");
	}



	// Default fallback: elements without bubble enrichment treat the "TB" (bubble-enriched)
	// spaces C1TB/C2TB as identical to the plain C1/C2 spaces; subclasses that actually have
	// bubble functions override these.
	void BulkElementBase::shape_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		//  if (this->has_bubble()) throw_runtime_error("Implement for bubble-enriched elements");
		this->shape_at_s_C1(s, psi);
	}

	void BulkElementBase::shape_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		//  if (this->has_bubble()) throw_runtime_error("Implement for bubble-enriched elements");
		this->shape_at_s_C2(s, psi);
	}

	void BulkElementBase::dshape_local_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		//  if (this->has_bubble()) throw_runtime_error("Implement for bubble-enriched elements");
		this->dshape_local_at_s_C2(s, psi, dpsi);
	}

	void BulkElementBase::dshape_local_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		//  if (this->has_bubble()) throw_runtime_error("Implement for bubble-enriched elements");
		this->dshape_local_at_s_C1(s, psi, dpsi);
	}

	// Interpolates the discontinuous-Lagrange (DL) fields at local coordinate s from their
	// per-node internal-data storage (the DL internal data entries are stored contiguously right
	// after the DG spaces' internal data, hence dg_offset).
	void BulkElementBase::get_interpolated_fields_DL(const oomph::Vector<double> &s, std::vector<double> &res, const unsigned &t) const
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		res.resize(functable->info_DL.numfields);
		oomph::Shape psi(eleminfo.nnode_DL);
		this->shape_at_s_DL(s, psi);
		unsigned dg_offset=0;
		for (unsigned int i_space=0;i_space<functable->num_present_dg_spaces;i_space++)
		{
			auto * space_info=functable->present_dg_spaces[i_space];
			dg_offset+=space_info->numfields_new;
		}		
		for (unsigned int fi = 0; fi < functable->info_DL.numfields; fi++)
		{
			res[fi] = 0.0;
			for (unsigned int l = 0; l < eleminfo.nnode_DL; l++)
			{
				res[fi] += psi[l] * this->internal_data_pt(dg_offset+fi)->value(t, l); // TODO: Better direct access of the buffer offset
			}
		}
	}

	// Returns the (spatially constant) D0 field values; D0 storage is single-valued per element,
	// so the local coordinate s is not actually needed for interpolation.
	void BulkElementBase::get_interpolated_fields_D0(const oomph::Vector<double> &s, std::vector<double> &res, const unsigned &t) const
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		res.resize(functable->info_D0.numfields);
		unsigned dg_offset=0;
		for (unsigned int i_space=0;i_space<functable->num_present_dg_spaces;i_space++)
		{
			auto * space_info=functable->present_dg_spaces[i_space];
			dg_offset+=space_info->numfields_new;
		}		
		for (unsigned int fi = 0; fi < functable->info_D0.numfields; fi++)
		{
			res[fi] = this->internal_data_pt(functable->info_DL.numfields + dg_offset+fi)->value(t, 0);  // TODO: Better direct access of the buffer offset
		}
	}

	// oomph-lib hook: interpolates all continuous (nodal, base-bulk) field values at local
	// coordinate s and history index t, in the same field ordering used throughout eleminfo.
	void BulkElementBase::get_interpolated_values(const unsigned &t, const oomph::Vector<double> &s, oomph::Vector<double> &values)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		const std::vector<std::vector<unsigned>> & node_to_element_index = this->get_nodal_space_index_to_element_index_map();
		values.resize(ncont_interpolated_values(),0.0);
		unsigned index=0;
		for (unsigned int si=0;si<functable->num_present_continuous_spaces;si++)
		{
			auto *space_info=functable->present_continuous_spaces[si];
			oomph::Shape psi(eleminfo.nnode_of_space[space_info->space_index]);
			this->shape_of_space(space_info->space_index, s, psi);			
			for (unsigned int fi = 0; fi < space_info->numfields_basebulk; fi++)
			{
				values[index] = 0.0;
				for (unsigned int l = 0; l < eleminfo.nnode_of_space[space_info->space_index]; l++)
				{
					values[index] += psi[l] * this->node_pt(node_to_element_index[space_info->space_index][l])->value(t, fi+ space_info->nodal_offset_basebulk);
				}
				index++;
			}		
		}
	}

	// Interpolates all discontinuous fields (DL followed by D0) at once, concatenating the results
	// of get_interpolated_fields_DL() and get_interpolated_fields_D0().
	void BulkElementBase::get_interpolated_discontinuous_values(const unsigned &t, const oomph::Vector<double> &s, oomph::Vector<double> &values)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		oomph::Vector<double> resDL;
		oomph::Vector<double> resD0;
		if (functable->info_DL.numfields)
			this->get_interpolated_fields_DL(s, resDL, t);
		if (functable->info_D0.numfields)
			this->get_interpolated_fields_D0(s, resD0, t);
		values.resize(resDL.size() + resD0.size());
		for (unsigned int i = 0; i < resDL.size(); i++)
		{
			values[i] = resDL[i];
		}
		for (unsigned int i = 0; i < resD0.size(); i++)
		{
			values[i + resDL.size()] = resD0[i];
		}
	}

	// Plain-text oomph-lib output is not used by pyoomph (see scalar_name_paraview above).
	void BulkElementBase::output(std::ostream &outfile, const unsigned &nplot)
	{
		throw_runtime_error("Not implemented");
	}

	// Number of flux components used by the Z2 error estimator (for adaptive refinement); a
	// separate, typically smaller, set of fluxes is used when estimating errors for eigenvectors
	// (use_eigen_error_estimators), since eigenmodes usually need different quantities than the
	// primary solution to drive mesh refinement.
	unsigned BulkElementBase::num_Z2_flux_terms()
	{
		if (BulkElementBase::use_eigen_error_estimators) return codeinst->get_func_table()->num_Z2_flux_terms_for_eigen;
		else return codeinst->get_func_table()->num_Z2_flux_terms;
	}

	// Evaluates the Z2-error-estimator flux vector at local coordinate s via the JIT-generated
	// GetZ2Fluxes (or GetZ2FluxesForEigen) function, used by oomph-lib's Z2 error estimator to
	// drive adaptive mesh refinement.
	void BulkElementBase::get_Z2_flux(const oomph::Vector<double> &s, oomph::Vector<double> &flux)
	{
		bool has_fluxes=(BulkElementBase::use_eigen_error_estimators ? codeinst->get_func_table()->GetZ2FluxesForEigen : codeinst->get_func_table()->GetZ2Fluxes );
		if (has_fluxes)
		{
			this->interpolate_hang_values(); // XXX This should be moved to somewhere else, after each update of any values
			this->prepare_shape_buffer_for_integration(codeinst->get_func_table()->shapes_required_Z2Fluxes, 0);
			double JLagr;
			this->fill_shape_info_at_s(s, 0, codeinst->get_func_table()->shapes_required_Z2Fluxes, JLagr, 0);
			this->set_remaining_shapes_appropriately(shape_info,codeinst->get_func_table()->shapes_required_Z2Fluxes);
			if (BulkElementBase::use_eigen_error_estimators)
			{
				codeinst->get_func_table()->GetZ2FluxesForEigen(&eleminfo, shape_info, &(flux[0]));
			}
			else
			{
				codeinst->get_func_table()->GetZ2Fluxes(&eleminfo, shape_info, &(flux[0]));
			}
		}
	}

	// Hook for subclasses to add element-type-specific hanging-node setup after the generic
	// oomph-lib hanging node machinery has run; no-op by default.
	void BulkElementBase::further_setup_hanging_nodes()
	{
		// std::cout << "FURTHER SETUP HANG" << std::endl;
	}

	// oomph-lib refinement hook: creates the son elements (via the type-specific
	// create_son_instance()) and initializes their refinement level and initial size fraction;
	// the sons are then filled in by pre_build()/further_build() as the mesh machinery proceeds.
	void BulkElementBase::dynamic_split(oomph::Vector<BulkElementBase *> &son_pt) const
	{
		// std::cout << "DYN SPLIT " << std::endl;
		int son_refine_level = Refine_level + 1;
		unsigned n_sons = required_nsons();
		son_pt.resize(n_sons);
		for (unsigned i = 0; i < n_sons; i++)
		{
			// std::cout << "C SON INST" << std::endl;
			son_pt[i] = this->create_son_instance();
			// std::cout << "SET REF" << std::endl;
			son_pt[i]->set_refinement_level(son_refine_level);
			son_pt[i]->initial_cartesian_nondim_size = this->initial_cartesian_nondim_size / ((double)n_sons);
		}
	}

   // Total number of DG (discontinuous-Galerkin, per-node-but-not-shared) fields across all
   // present DG spaces; base_bulk_only restricts the count to fields defined in the bulk element
   // itself, excluding additional fields only present on interfaces.
   unsigned BulkElementBase::num_DG_fields(bool base_bulk_only)
   {
    auto *ft=codeinst->get_func_table();
    if (base_bulk_only)
    {
	  unsigned cnt=0;
	  for (unsigned int i=0;i<ft->num_present_dg_spaces;i++)
	  {
		cnt+=ft->present_dg_spaces[i]->numfields_basebulk;
	  }
	  return cnt;      
    }
    else
    {
	  unsigned cnt=0;
	  for (unsigned int i=0;i<ft->num_present_dg_spaces;i++)
	  {
		cnt+=ft->present_dg_spaces[i]->numfields;
	  }
	  return cnt;      
    }
   }
   
   // Interpolates all fields of one DG space at local coordinate s. Fields inherited from a
   // parent/bulk element ("external", the first `nexternal` of them) are read from external
   // data; genuinely new fields defined at this level are read from internal data.
   void BulkElementBase::get_DG_fields_at_s(unsigned int space_index, unsigned history_index,const oomph::Vector<double> &s, oomph::Vector<double> &result) const
   {
	auto *ft=codeinst->get_func_table();
	auto & space_info=ft->dg_spaces[space_index];
	result.resize(space_info.numfields);
     for (unsigned int i=0;i<space_info.numfields;i++) result[i]=0.0;
     oomph::Shape psi(eleminfo.nnode_of_space[space_info.space_index]);
	 this->shape_of_space(space_info.space_index, s, psi);     
     unsigned nexternal=space_info.numfields-space_info.numfields_new;
     for (unsigned int i=0;i<nexternal;i++)
     {
      for (unsigned int l=0;l<eleminfo.nnode_of_space[space_info.space_index];l++) 
      {
       result[i]+=external_data_pt(space_info.external_offset_bulk+i)->value(history_index,this->get_DG_node_index(space_info.space_index, i,l))*psi[l];
      } 
     }
     for (unsigned int i=nexternal;i<space_info.numfields;i++)
     {
      for (unsigned int l=0;l<eleminfo.nnode_of_space[space_info.space_index];l++) 
      {
       result[i]+=internal_data_pt(space_info.internal_offset_new+i-nexternal)->value(history_index,l)*psi[l];
      }
     }
   }

	// Allocates the oomph::Data objects backing this element's non-nodal field storage: one
	// internal Data (sized to the number of nodes in that space) per newly-defined DG field, one
	// per DL field (shared across all DL "nodes" of the element), and one single-value internal
	// Data per D0 field (element-constant). Must be called once the eleminfo node counts per
	// space are known, and before fill_element_info()/further_build() try to access this storage.
	void BulkElementBase::allocate_discontinous_fields()
	{
	   // DG Fields.
		//Only add the fields directly added in this dimension. Parent degrees will be external data	   
		for (unsigned int i_space=0;i_space<codeinst->get_func_table()->num_present_dg_spaces;i_space++)
		{
			auto * space_info=codeinst->get_func_table()->present_dg_spaces[i_space];
			if (eleminfo.nnode_of_space[space_info->space_index] > 0)
			{
				for (unsigned int fi = 0; fi < space_info->numfields_new; fi++)
				{
					this->add_internal_data(new oomph::Data(eleminfo.nnode_of_space[space_info->space_index]), false);
				}
			}
		}
		
			
		if (eleminfo.nnode_DL > 0)
		{
			for (unsigned int fi = 0; fi < codeinst->get_func_table()->info_DL.numfields; fi++)
			{
				this->add_internal_data(new oomph::Data(eleminfo.nnode_DL), false);
				//		          std::cout << "  AFTER DL " << fi << "  " << this->ninternal_data() << " INT DATA" << std::endl <<std::flush;

			}
		}

		for (unsigned int fi = 0; fi < codeinst->get_func_table()->info_D0.numfields; fi++)
		{
			this->add_internal_data(new oomph::Data(1), false);
		}

//          std::cout << "ALLOCATED " << this->ninternal_data() << " INT DATA" << std::endl <<std::flush;
		
	}

	////////////////////////////////

	// BulkElementODE0d represents a plain "ODE" element: no spatial mesh/nodes at all, only D0
	// (element-constant) internal-data fields evolving in time -- used for globally-defined
	// ODEs/scalar quantities that are not associated with any spatial field.
	oomph::PointIntegral BulkElementODE0d::Default_integration_scheme;

	// Sets up a single dummy "node" purely so the generic shape-buffer machinery has something to
	// allocate against; the element has no actual continuous or DL spaces (nnode_of_space and
	// nnode_DL are forced to 0), only D0 fields.
	BulkElementODE0d::BulkElementODE0d(DynamicBulkElementInstance *code_inst, oomph::TimeStepper *tstepper) : timestepper(tstepper)
	{
		//std::cout << "CONSTRUCT BULK ODE 0D " << this << std::endl;
		this->codeinst = code_inst;
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 1; // One dummy node... Necessary to create the buffers
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 0;
		eleminfo.nnode_DL = 0;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
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
		unsigned nD0 = codeinst->get_func_table()->info_D0.numfields;
		for (unsigned int i = 0; i < nD0; i++)
			dest[i] = this->internal_data_pt(i)->value(0); // TODO Scaling
	}

	// Trivial: a 0-d element has no spatial extent, so the Jacobian/JLagr are always 1.
	double BulkElementODE0d::fill_shape_info_at_s(const oomph::Vector<double> &s, const unsigned int &index, const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, double &JLagr, unsigned int flag, oomph::DenseMatrix<double> *dxds,unsigned history_index) const
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
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 1;
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 1;
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = 1;
		eleminfo.nnode_of_space[SPACE_INDEX_C2] = 1;
		eleminfo.nnode_of_space[SPACE_INDEX_C2TB] = 1;
		eleminfo.nnode_DL = 1;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}



	double PointElement0d::invert_jacobian_mapping(const oomph::DenseMatrix<double> &jacobian, oomph::DenseMatrix<double> &inverse_jacobian) const
	{
		return 1.0;
	}

	void PointElement0d::dshape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids) const
	{
		psi[0] = 1;
		dpsids(0, 0) = 0;
	}

	void PointElement0d::shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		psi[0] = 1.0;
	}

	void PointElement0d::dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = 1.0;
		dpsi(0, 0) = 0.0;
	}

	void PointElement0d::shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		psi[0] = 1.0;
	}
	void PointElement0d::shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		psi[0] = 1.0;
	}

	void PointElement0d::dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = 1.0;
		dpsi(0, 0) = 0;
	}

	void PointElement0d::dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		psi[0] = 1.0;
		dpsi(0, 0) = 0;
	}

	// Writes the local node indices of sub-element `isubelem` (an element may be tesselated into
	// several simple sub-elements/triangles for numpy/plotting export) into `indices`; for a
	// single-node point element there is only one "sub-element" consisting of node 0.
	void PointElement0d::fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
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
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 2;
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = 2;
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 2;
		eleminfo.nnode_DL = 2;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}

	// Given a point s and a direction ds inside the element, finds how far (in units of ds) one
	// can move before leaving the local-coordinate range [-1,1], along with the outward normal
	// snormal of the exited face and its distance sdistance from the origin. Used by line-search/
	// tracing algorithms (e.g. Newton's method for locate_zeta, or particle tracing) that must
	// stay within the valid local coordinate domain of the element.
	double BulkElementLine1dC1::factor_when_local_coordinate_becomes_invalid(const oomph::Vector<double> &s, const oomph::Vector<double> &ds, oomph::Vector<double> &snormal, double &sdistance)
	{
		if (abs(ds[0]) < 1e-20)
			return 1e20;
		if (ds[0] > 0)
		{
			snormal.resize(1);
			snormal[0] = 1;
			sdistance = this->s_max();
			return (this->s_max() - s[0]) / ds[0];
		}
		else
		{
			snormal.resize(1);
			snormal[0] = -1;
			sdistance = -this->s_min();
			return (this->s_min() - s[0]) / ds[0];
		}
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

	void BulkElementLine1dC1::fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
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
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 3;
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 2;
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = 2;
		eleminfo.nnode_of_space[SPACE_INDEX_C2] = 3;
		eleminfo.nnode_of_space[SPACE_INDEX_C2TB] = 3;
		eleminfo.nnode_DL = 2;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
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
	
	// Same as BulkElementLine1dC1's version above, for the [-1,1]-parametrized quadratic line.
	double BulkElementLine1dC2::factor_when_local_coordinate_becomes_invalid(const oomph::Vector<double> &s, const oomph::Vector<double> &ds, oomph::Vector<double> &snormal, double &sdistance)
	{
		if (abs(ds[0]) < 1e-20)
			return 1e20;
		if (ds[0] > 0)
		{
			snormal.resize(1);
			snormal[0] = 1;
			sdistance = this->s_max();
			return (this->s_max() - s[0]) / ds[0];
		}
		else
		{
			snormal.resize(1);
			snormal[0] = -1;
			sdistance = -this->s_min();
			return (this->s_min() - s[0]) / ds[0];
		}
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

	void BulkElementLine1dC2::fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
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
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 2;
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = 2;
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 2;
		eleminfo.nnode_DL = 2;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
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

	void BulkTElementLine1dC1::fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
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
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 3;
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 2;
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = 2;		
		eleminfo.nnode_of_space[SPACE_INDEX_C2] = 3;
		eleminfo.nnode_of_space[SPACE_INDEX_C2TB] = 3;
		eleminfo.nnode_DL = 2;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
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

	void BulkTElementLine1dC2::fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
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

	

	////////////////////////////

	// BulkElementQuad2dC1: bilinear (4-node) quadrilateral element. C1/C1TB use all 4 corner
	// nodes; DL is a 3-value (constant + 2 gradient components) discontinuous linear
	// representation, not tied to any actual node.
	BulkElementQuad2dC1::BulkElementQuad2dC1()
	{
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 4;
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 4;
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = 4;
		eleminfo.nnode_DL = 3;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
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
	void BulkElementQuad2dC1::add_node_from_finer_neighbor_for_tesselated_numpy(int edge, oomph::Node *n, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes)
	{
		using namespace oomph::QuadTreeNames;
		// Check if we have already the node
		for (unsigned ni = 0; ni < this->nnode(); ni++)
		{
			if (this->node_pt(ni) == n)
				return; // Discard existing nodes
		}

		unsigned myindex = this->_numpy_index;
		if (add_nodes[myindex].empty())
		{
			add_nodes[myindex].resize(this->nedges());
		}
		int edgedir;
		if (edge == S)
			edgedir = 0;
		else if (edge == N)
			edgedir = 1;
		else if (edge == W)
			edgedir = 2;
		else if (edge == E)
			edgedir = 3;
		else
			throw std::runtime_error("Should not end up here");
		add_nodes[myindex][edgedir].insert(n);
	}

	// For every edge of this element, checks whether the neighboring element is on a coarser
	// refinement level (i.e. this element's edge is a hanging/T-junction edge from the
	// neighbor's point of view) and, if so, registers this element's edge nodes with that
	// coarser neighbor via add_node_from_finer_neighbor_for_tesselated_numpy(), so the numpy
	// tesselation of the coarser element can be split to avoid a visible crack at the junction.
	void BulkElementQuad2dC1::inform_coarser_neighbors_for_tesselated_numpy(std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes)
	{
		using namespace oomph::QuadTreeNames;

		if (dynamic_cast<InterfaceElementBase *>(this))
		{
			throw_runtime_error("Cannot yet tesselate interface meshes [will fail in connecting hanging nodes and have to go via the parent mesh");
		}

		oomph::Vector<int> edges(4);
		edges[0] = S;
		edges[1] = N;
		edges[2] = W;
		edges[3] = E;
		// Check my neighbors whether they are on a finer level. If so, we need to add more triangles here

		for (unsigned edge_counter = 0; edge_counter < 4; edge_counter++)
		{
			oomph::Vector<unsigned> translate_s(2);
			oomph::Vector<double> s(2), s_lo_neigh(2), s_hi_neigh(2), s_fraction(2);
			int neigh_edge, diff_level;
			bool in_neighbouring_tree;
			// Find pointer to neighbour in this direction
			oomph::QuadTree *neigh_pt;
			neigh_pt = quadtree_pt()->gteq_edge_neighbour(edges[edge_counter], translate_s, s_lo_neigh, s_hi_neigh, neigh_edge, diff_level, in_neighbouring_tree);
			if ((neigh_pt != 0) && diff_level != 0)
			{
				BulkElementBase *coarse_neigh = dynamic_cast<BulkElementBase *>(neigh_pt->object_pt());
				// Iterate along the nodes of this boundary
				std::vector<unsigned> local_nodes;
				if (edge_counter == 0)
					local_nodes = {0, 1};
				else if (edge_counter == 1)
					local_nodes = {2, 3};
				else if (edge_counter == 2)
					local_nodes = {0, 2};
				else
					local_nodes = {1, 3};
				for (auto lni : local_nodes)
				{
					coarse_neigh->add_node_from_finer_neighbor_for_tesselated_numpy(neigh_edge, this->node_pt(lni), add_nodes);
				}
			}
		}
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
				std::vector<int> corner_pairs = {0, 1, 2, 3, 0, 2, 1, 3};
				for (unsigned int d = 0; d < 4; d++)
				{
					for (auto *n : add_nodes[this->_numpy_index][d])
					{
						scoords[cnt].resize(2);
						// Now resolve the local coordinate by blending between the local coordinate (works, since elements are linear)
						double dist1 = 0.0;
						double dist2 = 0.0;
						for (unsigned int i = 0; i < this->nodal_dimension(); i++)
						{
							dist1 += (n->x(i) - this->node_pt(corner_pairs[2 * d])->x(i)) * (n->x(i) - this->node_pt(corner_pairs[2 * d])->x(i));
							dist2 += (n->x(i) - this->node_pt(corner_pairs[2 * d + 1])->x(i)) * (n->x(i) - this->node_pt(corner_pairs[2 * d + 1])->x(i));
						}
						dist1 = sqrt(dist1);
						dist2 = sqrt(dist2);
						double lambda = dist1 / (dist1 + dist2);
						scoords[cnt][0] = scoords[corner_pairs[2 * d]][0] * (1 - lambda) + scoords[corner_pairs[2 * d + 1]][0] * lambda;
						scoords[cnt][1] = scoords[corner_pairs[2 * d]][1] * (1 - lambda) + scoords[corner_pairs[2 * d + 1]][1] * lambda;
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

	// Shared implementation of factor_when_local_coordinate_becomes_invalid() for all 2-d
	// quadrilateral elements (parametrized on [-1,1]^2): finds which of the two local-coordinate
	// directions is exited first when moving from s along ds, and returns the corresponding
	// step factor and exit-face normal/distance.
	double QUAD2d_factor_when_local_coordinate_becomes_invalid(const oomph::Vector<double> &s, const oomph::Vector<double> &ds, oomph::Vector<double> &snormal, double &sdistance)
	{
		double f0, f1;
		double dsn = ds[0] * ds[0] + ds[1] * ds[1];
		dsn = sqrt(dsn);
		snormal.resize(2);
		if (dsn < 1e-20)
			return 1e20;
		dsn = 1 / dsn;

		if (abs(ds[0] * dsn) < 1e-16)
			f0 = 1e20;
		else if (ds[0] > 0)
			f0 = (1 - s[0]) / ds[0];
		else
			f0 = (-1 - s[0]) / ds[0];

		if (abs(ds[1] * dsn) < 1e-16)
			f1 = 1e20;
		else if (ds[1] > 0)
			f1 = (1 - s[1]) / ds[1];
		else
			f1 = (-1 - s[1]) / ds[1];

		sdistance = 1.0;
		if (f0 < f1)
		{
			snormal[1] = 0;
			snormal[0] = (ds[0] > 0 ? 1 : -1);
		}
		else
		{
			snormal[0] = 0;
			snormal[1] = (ds[1] > 0 ? 1 : -1);
		}

		return std::min(f0, f1);
	}

	double BulkElementQuad2dC1::factor_when_local_coordinate_becomes_invalid(const oomph::Vector<double> &s, const oomph::Vector<double> &ds, oomph::Vector<double> &snormal, double &sdistance)
	{
		return QUAD2d_factor_when_local_coordinate_becomes_invalid(s, ds, snormal, sdistance);
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
		eleminfo.elem_ptr = this;
		// std::cout << "SETTING ELEM PTR " <<  eleminfo.elem_ptr << std::endl;
		eleminfo.nnode = 9;
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 4;
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = 4;
		eleminfo.nnode_of_space[SPACE_INDEX_C2] = 9;		
		eleminfo.nnode_of_space[SPACE_INDEX_C2TB] = 9;
		eleminfo.nnode_DL = 3;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}

	double BulkElementQuad2dC2::factor_when_local_coordinate_becomes_invalid(const oomph::Vector<double> &s, const oomph::Vector<double> &ds, oomph::Vector<double> &snormal, double &sdistance)
	{
		return QUAD2d_factor_when_local_coordinate_becomes_invalid(s, ds, snormal, sdistance);
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
		if (codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C1].numfields_basebulk || codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C1TB].numfields_basebulk)
		{
			unsigned int nC2=codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk+codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk;
			for (unsigned int i = nC2; i < ncont_interpolated_values(); i++)
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
		if (value_id >= static_cast<int>(codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk + codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk))
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

	// Companion to interpolating_node_pt(): finds the C1 corner node located exactly at local
	// coordinate s (returns NULL if s does not coincide with a corner, within tolerance); for
	// C2/C2TB fields, any of the 9 geometric nodes may coincide, so delegates to the generic
	// get_node_at_local_coordinate().
	oomph::Node *BulkElementQuad2dC2::get_interpolating_node_at_local_coordinate(const oomph::Vector<double> &s, const int &value_id)
	{
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
	unsigned BulkElementQuad2dC2::ninterpolating_node_1d(const int &value_id)
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
	unsigned BulkElementQuad2dC2::ninterpolating_node(const int &value_id)
	{
		if (value_id >= static_cast<int>(codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk + codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk))
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
		if (value_id >= static_cast<int>(codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk + codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk))
		{
			return this->shape_at_s_C1(s, psi);
		}
		else
		{
			return this->shape(s, psi);
		}
	}

	// Same purpose as BulkElementQuad2dC1::add_node_from_finer_neighbor_for_tesselated_numpy()
	// above.
	void BulkElementQuad2dC2::add_node_from_finer_neighbor_for_tesselated_numpy(int edge, oomph::Node *n, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes)
	{
		using namespace oomph::QuadTreeNames;
		// Check if we have already the node
		for (unsigned ni = 0; ni < this->nnode(); ni++)
		{
			if (this->node_pt(ni) == n)
				return; // Discard existing nodes
		}

		unsigned myindex = this->_numpy_index;
		if (add_nodes[myindex].empty())
		{
			add_nodes[myindex].resize(this->nedges());
		}
		int edgedir;
		if (edge == S)
			edgedir = 0;
		else if (edge == N)
			edgedir = 1;
		else if (edge == W)
			edgedir = 2;
		else if (edge == E)
			edgedir = 3;
		else
			throw std::runtime_error("Should not end up here");
		add_nodes[myindex][edgedir].insert(n);
	}

	// Same purpose as BulkElementQuad2dC1::inform_coarser_neighbors_for_tesselated_numpy() above,
	// but registering the 3 edge nodes (2 corners + midside) of each of this 9-node element's
	// sides.
	void BulkElementQuad2dC2::inform_coarser_neighbors_for_tesselated_numpy(std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes)
	{
		using namespace oomph::QuadTreeNames;

		if (dynamic_cast<InterfaceElementBase *>(this))
		{
			throw_runtime_error("Cannot yet tesselate interface meshes [will fail in connecting hanging nodes and have to go via the parent mesh");
		}

		oomph::Vector<int> edges(4);
		edges[0] = S;
		edges[1] = N;
		edges[2] = W;
		edges[3] = E;
		// Check my neighbors whether they are on a finer level. If so, we need to add more triangles here
		for (unsigned edge_counter = 0; edge_counter < 4; edge_counter++)
		{
			oomph::Vector<unsigned> translate_s(2);
			oomph::Vector<double> s(2), s_lo_neigh(2), s_hi_neigh(2), s_fraction(2);
			int neigh_edge, diff_level;
			bool in_neighbouring_tree;
			// Find pointer to neighbour in this direction
			oomph::QuadTree *neigh_pt;
			neigh_pt = quadtree_pt()->gteq_edge_neighbour(edges[edge_counter], translate_s, s_lo_neigh, s_hi_neigh, neigh_edge, diff_level, in_neighbouring_tree);
			if ((neigh_pt != 0) && diff_level != 0)
			{
				BulkElementBase *coarse_neigh = dynamic_cast<BulkElementBase *>(neigh_pt->object_pt());
				// Iterate along the nodes of this boundary
				std::vector<unsigned> local_nodes;
				if (edge_counter == 0)
					local_nodes = {0, 1, 2};
				else if (edge_counter == 1)
					local_nodes = {6, 7, 8};
				else if (edge_counter == 2)
					local_nodes = {0, 3, 6};
				else
					local_nodes = {2, 5, 8};
				for (auto lni : local_nodes)
				{
					coarse_neigh->add_node_from_finer_neighbor_for_tesselated_numpy(neigh_edge, this->node_pt(lni), add_nodes);
				}
			}
		}
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
					std::map<double, oomph::Node *> sorted;
					for (auto *n : add_nodes[this->_numpy_index][edgeindex])
					{
						double dist = 0.0;
						for (unsigned int i = 0; i < this->nodal_dimension(); i++)
							dist += (n->x(i) - this->node_pt(start_corner)->x(i)) * (n->x(i) - this->node_pt(start_corner)->x(i));
						sorted[dist] = n;
					}
					double dist = 0.0;
					for (unsigned int i = 0; i < this->nodal_dimension(); i++)
						dist += (this->node_pt(L2node_along)->x(i) - this->node_pt(start_corner)->x(i)) * (this->node_pt(L2node_along)->x(i) - this->node_pt(start_corner)->x(i));
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
		eleminfo.elem_ptr = this;
		eleminfo.nnode = (has_bubble ? 4 : 3);
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = (has_bubble ? 4 : 3);
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 3;
		eleminfo.nnode_DL = 3;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}

	// Shared implementation of factor_when_local_coordinate_becomes_invalid() for all 2-d
	// triangular elements: the reference triangle has local coordinates s0,s1 in [0,1] with
	// s0+s1<=1, so there are three possible exit edges (s0=0, s1=0, s0+s1=1); works out, from the
	// signs/magnitudes of ds, which edge is hit first and returns the corresponding step factor
	// and outward normal/distance.
	double TRI2d_factor_when_local_coordinate_becomes_invalid(const oomph::Vector<double> &s, const oomph::Vector<double> &ds, oomph::Vector<double> &snormal, double &sdistance)
	{
		snormal.resize(2);
		if (abs(ds[0]) < 1e-20 && abs(ds[1]) < 1e-20)
			return 1e20;

		if (ds[0] < 0 && ds[1] < 0) // Can only hit s0 or s1 axis
		{
			if (-s[0] / ds[0] < -s[1] / ds[1])
			{
				snormal[0] = -1;
				snormal[1] = 0;
				sdistance = 0;
				return -s[0] / ds[0];
			}
			else
			{
				snormal[0] = 0;
				snormal[1] = -1;
				sdistance = 0;
				return -s[1] / ds[1];
			}
		}
		else if (ds[0] > 0 && ds[1] > 0) // Can only hit s2 axis
		{
			sdistance = snormal[0] = snormal[1] = 1 / sqrt(2.0);
			return (1.0 - (s[0] + s[1])) / (ds[0] + ds[1]);
		}
		else if (ds[0] > 0)
		{
			if (abs(ds[1]) < 1e-20)
			{
				sdistance = snormal[0] = snormal[1] = 1 / sqrt(2.0);
				return (1.0 - (s[0] + s[1])) / (ds[0] + ds[1]);
			}
			else if (ds[1] <= -ds[0])
			{
				snormal[0] = 0;
				snormal[1] = -1;
				sdistance = 0;
				return -s[1] / ds[1];
			}
			else
			{
				double l1 = (1.0 - (s[0] + s[1])) / (ds[0] + ds[1]);
				double l2 = -s[1] / ds[1];
				if (l1 < l2)
				{
					sdistance = snormal[0] = snormal[1] = 1 / sqrt(2.0);
					return l1;
				}
				else
				{
					snormal[0] = 0;
					snormal[1] = -1;
					sdistance = 0;
					return l2;
				}
			}
		}
		else
		{
			if (abs(ds[0]) < 1e-20)
			{
				sdistance = snormal[0] = snormal[1] = 1 / sqrt(2.0);
				return (1.0 - (s[0] + s[1])) / (ds[0] + ds[1]);
			}
			else if (ds[0] <= -ds[1])
			{
				snormal[0] = -1;
				snormal[1] = 0;
				sdistance = 0;
				return -s[0] / ds[0];
			}
			else
			{
				double l1 = (1.0 - (s[0] + s[1])) / (ds[0] + ds[1]);
				double l2 = (0 - s[0]) / ds[0];
				if (l1 < l2)
				{
					sdistance = snormal[0] = snormal[1] = 1 / sqrt(2.0);
					return l1;
				}
				else
				{
					snormal[0] = -1;
					snormal[1] = 0;
					sdistance = 0;
					return l2;
				}
			}
		}
	}

	double BulkElementTri2dC1::factor_when_local_coordinate_becomes_invalid(const oomph::Vector<double> &s, const oomph::Vector<double> &ds, oomph::Vector<double> &snormal, double &sdistance)
	{
		return TRI2d_factor_when_local_coordinate_becomes_invalid(s, ds, snormal, sdistance);
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

	void BulkElementTri2dC1::fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
	{
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
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 6;
		eleminfo.nnode_of_space[SPACE_INDEX_C2TB] = (with_bubble ? 7:6); // Must be done here! DG field allocation would otherwise alloc only 6 for D2TB!
		eleminfo.nnode_of_space[SPACE_INDEX_C2] = 6;
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = (with_bubble ? 4 : 3);		
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 3;
		eleminfo.nnode_DL = 3;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}

    // Creates a son of the correct concrete type (bubble-enriched BulkElementTri2dC2TB, or plain
    // BulkElementTri2dC2) matching this element, for dynamic_split()/mesh refinement.
    BulkElementBase * BulkElementTri2dC2::create_son_instance() const
	    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementTri2dC2(dynamic_cast<const BulkElementTri2dC2TB*>(this)!=nullptr);
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }


	double BulkElementTri2dC2::factor_when_local_coordinate_becomes_invalid(const oomph::Vector<double> &s, const oomph::Vector<double> &ds, oomph::Vector<double> &snormal, double &sdistance)
	{
		return TRI2d_factor_when_local_coordinate_becomes_invalid(s, ds, snormal, sdistance);
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

	void BulkElementTri2dC2::fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
	{
		if (tesselate_tri)
		{
			if (isubelem == 0)
			{
				indices[0] = 0;
				indices[1] = 3;
				indices[2] = 5;
			}
			else if (isubelem == 1)
			{
				indices[0] = 1;
				indices[1] = 4;
				indices[2] = 3;
			}
			else if (isubelem == 2)
			{
				indices[0] = 2;
				indices[1] = 5;
				indices[2] = 4;
			}
			else
			{
				indices[0] = 3;
				indices[1] = 4;
				indices[2] = 5;
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
		}
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
		eleminfo.elem_ptr = this;   
      eleminfo.nnode=4;
      eleminfo.nnode_of_space[SPACE_INDEX_C1]=3;
      eleminfo.nnode_of_space[SPACE_INDEX_C1TB]=4;
      eleminfo.nnode_DL=3;
      eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
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
   
   void BulkElementTri2dC1TB::fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
   {
      if (tesselate_tri)
		{
			if (isubelem == 0)
			{
				indices[0] = 0;
				indices[1] = 1;
				indices[2] = 3;
			}
			else if (isubelem == 1)
			{
				indices[0] = 1;
				indices[1] = 2;
				indices[2] = 3;
			}
			else if (isubelem == 2)
			{
				indices[0] = 2;
				indices[1] = 0;
				indices[2] = 3;
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
	///////////////////////////////

	// BulkElementTri2dC2TB: quadratic triangle with an additional cubic-bubble enrichment of the
	// C1TB space at the corners plus centroid (node 6) -- same MINI-style bubble as
	// BulkElementTri2dC1TB above, layered on top of the quadratic C2 geometry/fields.
	BulkElementTri2dC2TB::BulkElementTri2dC2TB() : BulkElementTri2dC2(true)
	{
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 7;
		eleminfo.nnode_of_space[SPACE_INDEX_C2TB] = 7;
		eleminfo.nnode_of_space[SPACE_INDEX_C2] = 6;
		eleminfo.nnode_of_space[SPACE_INDEX_C1TB] = 4;		
		eleminfo.nnode_of_space[SPACE_INDEX_C1] = 3;
		eleminfo.nnode_DL = 3;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
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
    


	void BulkElementTri2dC2TB::fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
	{
		if (tesselate_tri)
		{
			if (isubelem == 0)
			{
				indices[0] = 0;
				indices[1] = 1;
				indices[2] = 6;
			}
			else if (isubelem == 1)
			{
				indices[0] = 1;
				indices[1] = 2;
				indices[2] = 6;
			}
			else if (isubelem == 2)
			{
				indices[0] = 2;
				indices[1] = 0;
				indices[2] = 6;
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
		}
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

	void BulkElementBrick3dC1::fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
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

	std::vector<double> BulkElementBrick3dC1::get_outline(bool lagrangian)
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
		if (codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C1].numfields_basebulk || codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C1TB].numfields_basebulk)
		{			
			unsigned int nC2=codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C2TB].numfields_basebulk+codeinst->get_func_table()->continuous_spaces[SPACE_INDEX_C2].numfields_basebulk;
			for (unsigned int i = nC2; i < ncont_interpolated_values(); i++)
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

	void BulkElementBrick3dC2::fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
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

	std::vector<double> BulkElementBrick3dC2::get_outline(bool lagrangian)
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

	void BulkElementTetra3dC1::fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
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

	std::vector<double> BulkElementTetra3dC1::get_outline(bool lagrangian)
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
    void BulkElementTetra3dC1TB::fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
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
      auto res = new BulkElementTetra3dC2(dynamic_cast<const BulkElementTetra3dC2TB*>(this)!=nullptr);
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
		/*if (codeinst->get_func_table()->numfields_C1_basebulk)
		{
			for (unsigned int i = codeinst->get_func_table()->numfields_C2_basebulk; i < codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C1_basebulk; i++)
				this->setup_hang_for_value(i);
		}*/
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
	void BulkElementTetra3dC2::fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
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
	std::vector<double> BulkElementTetra3dC2::get_outline(bool lagrangian)
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
		std::vector<int> Central_node_on_face{13, 12, 10, 11};
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
	void BulkElementWedge3dC1::fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
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
	std::vector<double> BulkElementWedge3dC1::get_outline(bool lagrangian)
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
	void BulkElementPyramid3dC1::fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
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
	std::vector<double> BulkElementPyramid3dC1::get_outline(bool lagrangian)
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
	void BulkElementWedge3dC2::fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
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
	std::vector<double> BulkElementWedge3dC2::get_outline(bool lagrangian)
	{
		std::vector<double> res(0);
		throw_runtime_error("Cannot get outline from 3d elements yet");
		return res;
	}



	///////////////////////////////
	// RefineableSolidLineElement::build is oomph-lib's own pattern (mirrored here for the 1d solid/Lagrangian
	// case, which isn't provided out of the box) for constructing a refined son element: it copies/interpolates
	// the father element's nodal (Eulerian and Lagrangian) positions and history values onto the son's nodes,
	// selecting the left or right half of the father's local coordinate range depending on son_type.

	void RefineableSolidLineElement::build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt,
										   bool &was_already_built,
										   std::ofstream &new_nodes_file)
	{
		using namespace oomph::BinaryTreeNames;
		oomph::RefineableQElement<1>::build(mesh_pt, new_node_pt, was_already_built, new_nodes_file);
		if (was_already_built)
			return;
		int son_type = Tree_pt->son_type();
		RefineableSolidLineElement *father_el_pt = dynamic_cast<RefineableSolidLineElement *>(Tree_pt->father_pt()->object_pt());
#ifdef PARANOID
		if (static_cast<oomph::SolidNode *>(father_el_pt->node_pt(0))->nlagrangian_type() != 1)
		{
			throw oomph::OomphLibError(
				"We can't handle generalised nodal positions (yet).\n",
				OOMPH_CURRENT_FUNCTION,
				OOMPH_EXCEPTION_LOCATION);
		}
#endif

		oomph::Vector<double> s_left(1);
		oomph::Vector<double> s_right(1);

		oomph::Vector<double> s(1);
		oomph::Vector<double> xi(1);
		oomph::Vector<double> xi_fe(1);
		oomph::Vector<double> x(1);
		oomph::Vector<double> x_fe(1);

		// In order to set up the vertex coordinates we need to know which
		// type of son the current element is
		switch (son_type)
		{
		case L:
			s_left[0] = -1.0;
			s_right[0] = 0.0;
			break;

		case R:
			s_left[0] = 0.0;
			s_right[0] = 1.0;
			break;
		}

		// Pass the undeformed macro element onto the son
		//  hierher why can I read this?
		if (father_el_pt->undeformed_macro_elem_pt() != 0)
		{
			throw_runtime_error("TODO: Check this");
			Undeformed_macro_elem_pt = father_el_pt->undeformed_macro_elem_pt();
			s_macro_ll(0) = father_el_pt->s_macro_ll(0) + 0.5 * (s_left[0] + 1.0) * (father_el_pt->s_macro_ur(0) - father_el_pt->s_macro_ll(0));
			s_macro_ur(0) = father_el_pt->s_macro_ll(0) + 0.5 * (s_right[0] + 1.0) * (father_el_pt->s_macro_ur(0) - father_el_pt->s_macro_ll(0));
		}

		unsigned n = 0;
		unsigned n_p = nnode_1d();
		for (unsigned i0 = 0; i0 < n_p; i0++)
		{
			s[0] = s_left[0] + (s_right[0] - s_left[0]) * double(i0) / double(n_p - 1);
			n = i0;
			father_el_pt->get_x_and_xi(s, x_fe, x, xi_fe, xi);
			oomph::SolidNode *elastic_node_pt = static_cast<oomph::SolidNode *>(node_pt(n));
			elastic_node_pt->x(0) = x_fe[0];
			if (Use_undeformed_macro_element_for_new_lagrangian_coords)
			{
				elastic_node_pt->xi(0) = xi[0];
			}
			else
			{
				elastic_node_pt->xi(0) = xi_fe[0];
			}
			oomph::TimeStepper *time_stepper_pt = father_el_pt->node_pt(0)->time_stepper_pt();
			unsigned ntstorage = time_stepper_pt->ntstorage();
			if (ntstorage != 1)
			{
				for (unsigned t = 1; t < ntstorage; t++)
				{
					elastic_node_pt->x(t, 0) = father_el_pt->interpolated_x(t, s, 0);
				}
			}
		}

		this->set_integration_scheme(father_el_pt->integral_pt());
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
	void InterfaceElementBase::update_equation_remapping_from_element(BulkElementBase *source_elem,const JITFuncSpec_RequiredShapes_FiniteElement_t *required_shapes,std::vector<int> &eqn_map,int bulk_indicator)
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
		auto * ft=this->get_code_instance()->get_func_table();
		const std::vector<std::vector<std::vector<unsigned>>> & dummy_value_interpolation_map=this->get_dummy_value_interpolation_map();
		for (unsigned int ispace=0;ispace<ft->num_present_continuous_spaces;ispace++)
		{
			auto space_info=ft->present_continuous_spaces[ispace];
			const std::vector<unsigned> & get_nodal_space_index_to_element_index=this->get_nodal_space_index_to_element_index_map()[space_info->space_index];
			unsigned nnode=eleminfo.nnode_of_space[space_info->space_index];
			int hangindex=space_info->hangindex;
			for (unsigned int inode=0;inode<nnode;inode++)
			{
				pyoomph::Node * node=dynamic_cast<pyoomph::Node*>(this->node_pt(get_nodal_space_index_to_element_index[inode]));
				if (node->is_hanging(hangindex))
				{					
					throw_runtime_error("Hanging node ");
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
								pyoomph::Node * master_node=dynamic_cast<pyoomph::Node*>(hang_info->master_node_pt(m));
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
					pyoomph::Node * dummynode=dynamic_cast<pyoomph::Node*>(this->node_pt(dummy_value_interpolation[idummy][0]));
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
							  pyoomph::Node * master_node=dynamic_cast<pyoomph::Node*>(this->node_pt(dummy_value_interpolation[idummy][m]));
							  pyoomph::BoundaryNode * boundnode=dynamic_cast<pyoomph::BoundaryNode*>(master_node);
							  if (!boundnode) throw_runtime_error("master_node is not a BoundaryNode");
							  unsigned master_index=boundnode->index_of_first_value_assigned_by_face_element(add_field_index);
							  val+=master_node->value(t, master_index);
						   }
						   val/=(dummy_value_interpolation[idummy].size()-1.0);
						   //if (t==0) std::cout << " Patching interface node "<<  dummy_value_interpolation[idummy][0] <<"  of space " << space_info->space_name << " field " << field_index << " " << space_info->fieldnames[field_index+space_info->numfields_basebulk] << " at time " << t << " with value " << val << std::endl;
						   unsigned dest_index=dest_boundnode->index_of_first_value_assigned_by_face_element(add_field_index);
						   //std::cout << " Patching interface node "<<  dummy_value_interpolation[idummy][0] <<"  of space " << space_info->space_name << " field " << field_index << " " << space_info->fieldnames[field_index+space_info->numfields_basebulk] << " at time " << t << " with value " << val << "field index" << add_field_index << "value index" << dest_index << " is pinned " << dummynode->is_pinned(dest_index) << std::endl;
						   dummynode->value_pt(dest_index)[t]=val;
						}
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
	void InterfaceElementBase::assign_hanging_additional_interface_local_equations_for_space(const bool &store_local_dof_pt,unsigned addfields,unsigned basebulk_offset,unsigned nnode, int hangindex,  char * fieldnames[], unsigned (BulkElementBase::*node_index_to_element)(const unsigned &) const,  std::map<Node*, int> *& add_interf_local_hang_eqs)
	{
		bool has_hanging_interface_dofs=false;
		unsigned local_eqn_number = ndof();      
      	std::deque<unsigned long> global_eqn_number_queue;

		if (addfields)
		{
			std::vector<std::map<oomph::Node*, bool>> local_eqn_number_done(addfields, std::map<oomph::Node*, bool>());

			if (add_interf_local_hang_eqs) delete[] add_interf_local_hang_eqs;
			add_interf_local_hang_eqs=new std::map<Node*, int>[addfields];

			std::vector<unsigned> add_field_indices(addfields, 0);
			for (unsigned int ifield=0;ifield < addfields;ifield++)
			{
				add_field_indices[ifield]=this->codeinst->resolve_interface_dof_id(fieldnames[basebulk_offset+ifield]);
			}
			for (unsigned int inode=0;inode<nnode;inode++)
			{
				pyoomph::Node * node=dynamic_cast<pyoomph::Node*>(this->node_pt((this->*node_index_to_element)(inode)));
				if (node->is_hanging(hangindex))
				{
					oomph::HangInfo * hang_info=node->hanging_pt(hangindex);
					for (unsigned int m=0;m<hang_info->nmaster();m++)
					{
						pyoomph::Node * master_node=dynamic_cast<pyoomph::Node*>(hang_info->master_node_pt(m));
						BoundaryNode * boundnode=dynamic_cast<BoundaryNode*>(master_node);
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
								unsigned master_value_index=boundnode->index_of_first_value_assigned_by_face_element(add_field_indices[ifield]);     
                  				add_interf_local_hang_eqs[ifield][master_node]  =nodal_local_eqn(local_node_index, master_value_index);
								has_hanging_interface_dofs=true;
                			}
						}
						else
						{						
							for (unsigned int ifield=0;ifield<addfields;ifield++)
							{
								unsigned master_value_index=boundnode->index_of_first_value_assigned_by_face_element(add_field_indices[ifield]);
								long eqn_number = master_node->eqn_number(master_value_index);							
								if (eqn_number >= 0)
								{	
									if (add_interf_local_hang_eqs[ifield].find(master_node) == add_interf_local_hang_eqs[ifield].end())
									{										
										add_interf_local_hang_eqs[ifield][master_node] = local_eqn_number;																		
										global_eqn_number_queue.push_back(eqn_number);								
										if (store_local_dof_pt)
										{
											GeneralisedElement::Dof_pt_deque.push_back(master_node->value_pt(master_value_index));
										}									
										local_eqn_number++;
										has_hanging_interface_dofs=true;
									}
								}
								else
								{
									add_interf_local_hang_eqs[ifield][master_node] = oomph::Data::Is_pinned;
								}
							}
						}
					}					
				}
			}
		}

		if (!global_eqn_number_queue.empty())
		{
			has_hanging_interface_dofs=true;
	  		add_global_eqn_numbers(global_eqn_number_queue,GeneralisedElement::Dof_pt_deque);      
      		if (store_local_dof_pt)
      		{
        		std::deque<double*>().swap(GeneralisedElement::Dof_pt_deque);
      		}
		}
      
      	if (!has_hanging_interface_dofs)
      	{
        	delete[] add_interf_local_hang_eqs;
        	add_interf_local_hang_eqs = 0;
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
		update_equation_remapping_from_element(dynamic_cast<BulkElementBase *>(this->bulk_element_pt()),functable->merged_required_shapes.bulk_shapes , bulk_eqn_map,1);	
		if (functable->merged_required_shapes.bulk_shapes && functable->merged_required_shapes.bulk_shapes->bulk_shapes)
		{
			update_equation_remapping_from_element(dynamic_cast<BulkElementBase *>(dynamic_cast<InterfaceElementBase *>(this->bulk_element_pt())->bulk_element_pt()),functable->merged_required_shapes.bulk_shapes->bulk_shapes,  bulk_bulk_eqn_map,2);
		}
		if (functable->merged_required_shapes.opposite_shapes && !is_internal_facet_opposite_dummy())
		{
			if (!dynamic_cast<InterfaceElementBase *>(opposite_side))
			{
				throw_runtime_error("Missing opposite element");
			}
			if (!this->is_internal_facet_opposite_dummy())
			{		
				update_equation_remapping_from_element(opposite_side,functable->merged_required_shapes.opposite_shapes,  opp_interf_eqn_map,-1);
				if (functable->merged_required_shapes.opposite_shapes->bulk_shapes)
				{
					if (!dynamic_cast<InterfaceElementBase *>(opposite_side)->bulk_element_pt())
					{
						throw_runtime_error("Missing opposite bulk element");
					}			
					update_equation_remapping_from_element(dynamic_cast<BulkElementBase *>(dynamic_cast<InterfaceElementBase *>(opposite_side)->bulk_element_pt()),functable->merged_required_shapes.opposite_shapes->bulk_shapes , opp_bulk_eqn_map,-2);
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

	// Allocates the "additional values" (interface-only dofs) on this element's boundary nodes for every
	// interface field of every continuous space present, using oomph-lib's BoundaryNodeBase machinery so
	// several interface elements sharing a node can share/independently own the corresponding slots. Newly
	// allocated (not previously present) dofs are optionally seeded via interpolate_newly_constructed_additional_dof.
	void InterfaceElementBase::add_interface_dofs()
	{
		if (false && std::string(this->codeinst->get_code()->get_func_table()->domain_name)!="_internal_facets_")
		{
			for (unsigned l = 0; l < eleminfo.nnode; ++l)
			{			
				if (!dynamic_cast<BoundaryNode*>(this->node_pt(l))) throw_runtime_error("Interface element has a node which is not a BoundaryNode. This can happen in meshes when you have sharp corners in a boundary. Happened in "+this->codeinst->get_code()->get_file_name());
			}
		}
		
		auto *ft = codeinst->get_func_table();
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
			const std::vector<std::vector<unsigned>> & father_space_to_elem_index = father->get_nodal_space_index_to_element_index_map();
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
	void InterfaceElementBase::update_in_external_fd(const unsigned &i)
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
			if (data==dynamic_cast<pyoomph::Node *>(this->node_pt(k))->variable_position_pt())
			{
				return true;
			}
		}; // Nodes can be the same
		const std::vector<std::vector<unsigned>> & space_to_elem_index = this->get_nodal_space_index_to_element_index_map();
		for (unsigned int si=0;si<this->get_code_instance()->get_func_table()->num_present_continuous_spaces;si++)
		{
			auto * space_info=this->get_code_instance()->get_func_table()->present_continuous_spaces[si];
			
			for (unsigned int j=0;j<eleminfo.nnode_of_space[space_info->space_index];j++)
			{
				auto *nod_pt = dynamic_cast<pyoomph::Node *>(this->node_pt(space_to_elem_index[space_info->space_index][j]));
				if (nod_pt->is_hanging(space_info->space_index)) // TODO: In principle, it can also hang elsewhere, i.e. on another index!
				{
					oomph::HangInfo *const hang_pt = nod_pt->hanging_pt(space_info->space_index);
					const unsigned nmaster = hang_pt->nmaster();
					for (unsigned m = 0; m < nmaster; m++)
					{
						auto *const master_nod_pt = dynamic_cast<pyoomph::Node *>(hang_pt->master_node_pt(m));
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
			if (data == dynamic_cast<pyoomph::Node *>(this->node_pt(k))->variable_position_pt())
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
		  for (unsigned i=0;i<space_info->numfields_bulk;i++)
		  {
			this->add_external_data(blk->get_DG_nodal_data(space_info->space_index, i));
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
			if (require_dx_psi ||   required->Pos.psi  || required->DL.dx_psi || required->normal || required->elemsize_Eulerian || required->elemsize_Eulerian_cartesian) 
			{
				// Add required geometric external data to be finite differenced
				
				for (unsigned int j = 0; j < from_elem->get_eleminfo()->nnode; j++)
				{
					auto *nod_pt = dynamic_cast<pyoomph::Node *>(from_elem->node_pt(j));
					if (nod_pt->is_hanging())
					{						
						oomph::HangInfo *const hang_pt = nod_pt->hanging_pt();
						const unsigned nmaster = hang_pt->nmaster();
						for (unsigned m = 0; m < nmaster; m++)
						{
							auto *const master_nod_pt = dynamic_cast<pyoomph::Node *>(hang_pt->master_node_pt(m));
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
								   double old=dynamic_cast<pyoomph::Node*>(Bulk_element_pt->node_pt(lp))->variable_position_pt()->value(jp);
								   dynamic_cast<pyoomph::Node*>(Bulk_element_pt->node_pt(lp))->variable_position_pt()->set_value(jp,old+FD_eps);								   
               				this->get_dnormal_dcoords_at_s(s, dnormal_dcoord1, NULL);	
               				d2nodal_FD[i][l][j][lp][jp]= (dnormal_dcoord1[i][l][j]-dnormal_dcoord0[i][l][j])/FD_eps;
								   dynamic_cast<pyoomph::Node*>(Bulk_element_pt->node_pt(lp))->variable_position_pt()->set_value(jp,old);								                  				
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
			// TODO: Only geometric data!
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
			dynamic_cast<BulkElementBase *>(this->bulk_element_pt())->fill_shape_info_at_s(sbulk, index, *(required.bulk_shapes), shape_info->bulk_shapeinfo, JLagrBulk, flag);
			if (required.bulk_shapes->bulk_shapes)
			{
				InterfaceElementBase *bulk_as_inter = dynamic_cast<InterfaceElementBase *>(this->bulk_element_pt());
				oomph::Vector<double> sbulkbulk = bulk_as_inter->local_coordinate_in_bulk(sbulk);
				double JLagrBulkBulk;
				dynamic_cast<BulkElementBase *>(bulk_as_inter->bulk_element_pt())->fill_shape_info_at_s(sbulkbulk, index, *(required.bulk_shapes->bulk_shapes), shape_info->bulk_shapeinfo->bulk_shapeinfo, JLagrBulkBulk, flag);
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
			dynamic_cast<InterfaceElementBase *>(opposite_side)->fill_shape_info_at_s(sopp, index, *(required.opposite_shapes), shape_info->opposite_shapeinfo, JLagrOpp, flag);
			if (required.opposite_shapes->bulk_shapes)
			{
				oomph::Vector<double> sopp_blk = dynamic_cast<InterfaceElementBase *>(opposite_side)->local_coordinate_in_bulk(sopp);
				double JLagrOppBlk;
				// std::cout << "FILLING OPPBLK HERE " << index << std::endl;
				dynamic_cast<BulkElementBase *>(dynamic_cast<InterfaceElementBase *>(opposite_side)->bulk_element_pt())->fill_shape_info_at_s(sopp_blk, index, *(required.opposite_shapes->bulk_shapes), shape_info->opposite_shapeinfo->bulk_shapeinfo, JLagrOppBlk, flag);
			}
		}

		return this->J_eulerian(s); // TODO: This likely can be just set to JEulerian from above
	}

	// Ensures hanging-node values on this element and (if required) the attached bulk/opposite elements are
	// up to date before integration, then delegates the rest to the base class.
	void InterfaceElementBase::prepare_shape_buffer_for_integration(const JITFuncSpec_RequiredShapes_FiniteElement_t &required_shapes, unsigned int flag)
	{
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
			dynamic_cast<InterfaceElementBase *>(this->opposite_side)->interpolate_hang_values(); // TODO: This might be put somewhere else
			if (required_shapes.opposite_shapes->bulk_shapes)
			{
				dynamic_cast<BulkElementBase *>(dynamic_cast<InterfaceElementBase *>(this->opposite_side)->bulk_element_pt())->interpolate_hang_values(); // TODO: This might be put somewhere else
			}
			this->fill_opposite_node_indices(shape_info);
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
			dynamic_cast<InterfaceElementBase *>(this->opposite_side)->set_remaining_shapes_appropriately(shape_info->opposite_shapeinfo, *(required_shapes.opposite_shapes));
			if (required_shapes.opposite_shapes->bulk_shapes)
			{
				dynamic_cast<BulkElementBase *>(dynamic_cast<InterfaceElementBase *>(this->opposite_side)->bulk_element_pt())->set_remaining_shapes_appropriately(shape_info->opposite_shapeinfo->bulk_shapeinfo, *(required_shapes.opposite_shapes->bulk_shapes));
			}
		}
	}

	// Builds the hanging-node equation info (used by the JIT-generated code to assemble residuals/Jacobians
	// correctly across hanging nodes) for the required bulk/opposite/their-bulk elements, and additionally
	// (when eqn_remap is not already provided from outside) fills in this element's own eqn-remapping arrays
	// (bulk_eqn_map, bulk_bulk_eqn_map, opp_interf_eqn_map, opp_bulk_eqn_map) built by update_equation_remapping().
	bool InterfaceElementBase::fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap)
	{
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
			InterfaceElementBase *opp = dynamic_cast<InterfaceElementBase *>(this->opposite_side);
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
			InterfaceElementBase *iopp = dynamic_cast<InterfaceElementBase *>(opp);
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
		if (opposite_side && dynamic_cast<InterfaceElementBase *>(opposite_side)->is_internal_facet_opposite_dummy() && !(opposite_side->ndof()))
		{

		  dynamic_cast<InterfaceElementBase *>(opposite_side)->assign_local_eqn_numbers(true);
		}
		BulkElementBase::assign_additional_local_eqn_numbers();
		oomph::FaceElement::assign_additional_local_eqn_numbers();
	}

	// Global cache of pre-built oomph-lib integration schemes (Gauss rules etc.), shared across element instances.
	IntegrationSchemeStorage integration_scheme_storage;

	// const unsigned BulkElementTri2dC2TB::Central_node_on_face[3] = {4,5,3};
	oomph::TBubbleEnrichedGauss<2, 3> BulkElementTri2dC1TB::Default_enriched_integration_scheme;
	oomph::TBubbleEnrichedGauss<2, 3> BulkElementTri2dC2TB::Default_enriched_integration_scheme;
	oomph::TBubbleEnrichedGauss<3, 3> BulkElementTetra3dC2TB::Default_enriched_integration_scheme;

	bool InterfaceElementBase::interpolate_new_interface_dofs=true;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Static per-element-class lookup tables used throughout this file and by the JIT-generated code to
	// translate between the different node numbering schemes:
	//  - Dummy_Value_Interpolation_Map[space]: for each "dummy" node of that space (a node that isn't an
	//    independent dof of the space, e.g. a C1 dummy node on a C2-only element), the list of node indices
	//    {dummy_node, source_node_1, source_node_2, ...} whose average defines its (non-independent) value.
	//  - Nodal_Space_Index_To_Element_Index_Map[space]: maps the local node index within a given field space
	//    (C1/C1TB/C2/C2TB, i.e. "the n-th node that carries this space's dofs") to the local node index within
	//    the full element node numbering.
	//  - Element_Index_To_Nodal_Space_Index_Map[space]: the inverse mapping (element-local node index -> index
	//    within that space's own node numbering, or -1 if that node does not carry dofs of that space).
	//  - Possible_Face_Indices: the valid face_index values that can be passed to construct_face_element()/
	//    boundary_node_pt() etc. for that element type (e.g. {-2,-1,1,2} for the 4 faces of a quad).
	// These tables are all empty ({}) for spaces the given element type does not support.
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	const std::vector<std::vector<std::vector<unsigned>>> BulkElementBase::Dummy_Value_Interpolation_Map=
	{
		{}, // C2TB
		{}, // C2
		{}, // C1TB
		{}  // C1
	};

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

	const std::vector<std::vector<std::vector<unsigned>>> BulkElementLine1dC2::Dummy_Value_Interpolation_Map={
		{}, // C2TB 
		{}, // C2
		{{1,0,2}}, // C1TB
		{{1,0,2}}  // C1
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

	const std::vector<std::vector<std::vector<unsigned>>> BulkTElementLine1dC2::Dummy_Value_Interpolation_Map={
		{}, // C2TB 
		{}, // C2
		{{1,0,2}}, // C1TB
		{{1,0,2}}  // C1
	};

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

	const std::vector<std::vector<std::vector<unsigned>>> BulkElementQuad2dC2::Dummy_Value_Interpolation_Map={
		{}, // C2TB 
		{}, // C2
		{{1, 0,2}, {3,0,6}, {5,2,8}, {7,6,8}, {4,1,3,5,7} }, // C1TB
		{{1, 0,2}, {3,0,6}, {5,2,8}, {7,6,8}, {4,1,3,5,7} }  // C1
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

	const std::vector<std::vector<std::vector<unsigned>>> BulkElementTri2dC1TB::Dummy_Value_Interpolation_Map={
		{}, // C2TB 
		{}, // C2
		{ }, // C1TB
		{{3, 0,1,2} }  // C1
	};

	const std::vector<std::vector<unsigned>> BulkElementTri2dC2::Nodal_Space_Index_To_Element_Index_Map={
		{}, // C2TB 
		{0,1,2,3,4,5}, // C2
		{}, // C1TB
		{0,1,2}  // C1
	};

	const std::vector<std::vector<std::vector<unsigned>>> BulkElementTri2dC2::Dummy_Value_Interpolation_Map={
		{}, // C2TB 
		{}, // C2
		{}, // C1TB
		{{3, 0,1}, {4, 1,2}, {5, 0, 2} }  // C1		
	};

	const std::vector<std::vector<unsigned>> BulkElementTri2dC2TB::Nodal_Space_Index_To_Element_Index_Map={
		{0,1,2,3,4,5,6}, // C2TB 
		{0,1,2,3,4,5}, // C2
		{0,1,2,6}, // C1TB
		{0,1,2}  // C1
	};

	const std::vector<std::vector<std::vector<unsigned>>> BulkElementTri2dC2TB::Dummy_Value_Interpolation_Map={
		{}, // C2TB 
		{{6, 0, 1, 2}}, // C2
		{{3, 0,1}, {4, 1,2}, {5, 0, 2}}, // C1TB
		{{3, 0,1}, {4, 1,2}, {5, 0, 2}, {6, 0, 1, 2} }  // C1		
	};

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

	const std::vector<std::vector<std::vector<unsigned>>> BulkElementBrick3dC2::Dummy_Value_Interpolation_Map={
		{}, // C2TB 
		{}, // C2
		{{1, 0, 2}, {3, 0, 6}, {5, 2, 8}, {7, 6, 8}, {4, 0,2,6,8},
		 {19, 18, 20}, {21, 18, 24}, {23, 20, 26}, {25, 24, 26}, {22, 18,20,24,26},
		 {9, 0, 18}, {11, 2, 20}, {15, 6, 24}, {17, 8, 26}, {10, 0,2,18,20}, {12, 0,6,18,24},  {16, 6,8,24,26}, {14, 2,8,20,26},
		 {13, 0,2,6,8,18,20,24,26}},  // C1TB		
		{{1, 0, 2}, {3, 0, 6}, {5, 2, 8}, {7, 6, 8}, {4, 0,2,6,8},
		 {19, 18, 20}, {21, 18, 24}, {23, 20, 26}, {25, 24, 26}, {22, 18,20,24,26},
		 {9, 0, 18}, {11, 2, 20}, {15, 6, 24}, {17, 8, 26}, {10, 0,2,18,20}, {12, 0,6,18,24},  {16, 6,8,24,26}, {14, 2,8,20,26},
		 {13, 0,2,6,8,18,20,24,26}}  // C1		
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

	const std::vector<std::vector<std::vector<unsigned>>> BulkElementTetra3dC1TB::Dummy_Value_Interpolation_Map={
		{}, // C2TB 
		{}, // C2
		{}, // C1TB
		{{4, 0,1,2,3}}  // C1
	};

	const std::vector<std::vector<unsigned>> BulkElementTetra3dC2::Nodal_Space_Index_To_Element_Index_Map={
		{}, // C2TB 
		{0,1,2,3,4,5,6,7,8,9}, // C2
		{}, // C1TB
		{0,1,2,3}  // C1
	};

	const std::vector<std::vector<std::vector<unsigned>>> BulkElementTetra3dC2::Dummy_Value_Interpolation_Map=
	{
		{}, // C2TB 
		{}, // C2
		{}, // C1TB
		{{4,0,1},{5,0,2},{6,0,3},{7,1,2},{8,2,3},{9,1,3}}  // C1		
	};

	const std::vector<std::vector<unsigned>> BulkElementTetra3dC2TB::Nodal_Space_Index_To_Element_Index_Map={
		{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14}, // C2TB 
		{0,1,2,3,4,5,6,7,8,9}, // C2
		{0,1,2,3,14}, // C1TB
		{0,1,2,3}  // C1
	};

	const std::vector<std::vector<std::vector<unsigned>>> BulkElementTetra3dC2TB::Dummy_Value_Interpolation_Map=
	{
		{}, // C2TB 
		{{10,0,1,3},{11,0,1,2},{12,0,2,3},{13,1,2,3},{14,0,1,2,3}}, // C2
		{{4,0,1},{5,0,2},{6,0,3},{7,1,2},{8,2,3},{9,1,3},{10,0,1,3},{11,0,1,2},{12,0,2,3},{13,1,2,3}}, // C1TB
		{{4,0,1},{5,0,2},{6,0,3},{7,1,2},{8,2,3},{9,1,3},{10,0,1,3},{11,0,1,2},{12,0,2,3},{13,1,2,3},{14,0,1,2,3}}  // C1		
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

	const std::vector<std::vector<std::vector<unsigned>>> BulkElementWedge3dC2::Dummy_Value_Interpolation_Map=
	{
		{}, // C2TB 
		{}, // C2
		{}, // C1TB
		{{3,0,1},{4,1,2},{5,0,2},{15,12,13},{16,12,14},{17,13,14},{6,0,12},{7,1,13},{8,2,14},{9,3,15},{10,4,16},{11,5,17}}  // C1		
	};

	const std::vector<std::vector<unsigned>> BulkElementPyramid3dC1::Nodal_Space_Index_To_Element_Index_Map={
		{}, // C2TB 
		{}, // C2
		{}, // C1TB
		{0,1,2,3,4}  // C1
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
		{0,1,2}  // C1
	};

	const std::vector<std::vector<int>> BulkElementTri2dC2::Element_Index_To_Nodal_Space_Index_Map={
		{}, // C2TB
		{0,1,2,3,4,5}, // C2
		{}, // C1TB
		{0,1,2}  // C1
	};

	const std::vector<std::vector<int>> BulkElementTri2dC2TB::Element_Index_To_Nodal_Space_Index_Map={
		{0,1,2,3,4,5,6}, // C2TB
		{0,1,2,3,4,5}, // C2
		{0,1,2,-1,-1,-1,3}, // C1TB
		{0,1,2}  // C1
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
		{0,1,2,3}  // C1
	};

	const std::vector<std::vector<int>> BulkElementTetra3dC2::Element_Index_To_Nodal_Space_Index_Map={
		{}, // C2TB
		{0,1,2,3,4,5,6,7,8,9}, // C2
		{}, // C1TB
		{0,1,2,3}  // C1
	};

	const std::vector<std::vector<int>> BulkElementTetra3dC2TB::Element_Index_To_Nodal_Space_Index_Map={
		{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14}, // C2TB
		{0,1,2,3,4,5,6,7,8,9}, // C2
		{0,1,2,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,4}, // C1TB
		{0,1,2,3}  // C1
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
		{0,1,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,3,4,5}  // C1
	};


	const std::vector<std::vector<int>> BulkElementPyramid3dC1::Element_Index_To_Nodal_Space_Index_Map={
		{}, // C2TB
		{}, // C2
		{}, // C1TB
		{0,1,2,3,4}  // C1
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

	const std::vector<int> BulkElementODE0d::Possible_Face_Indices={};
	const std::vector<int> PointElement0d::Possible_Face_Indices={};
	const std::vector<int> BulkElementLine1dC1::Possible_Face_Indices={-1,1};
	const std::vector<int> BulkElementLine1dC2::Possible_Face_Indices={-1,1};
	
	const std::vector<int> BulkTElementLine1dC1::Possible_Face_Indices={-1,1};
	const std::vector<int> BulkTElementLine1dC2::Possible_Face_Indices={-1,1};

	const std::vector<int> BulkElementQuad2dC1::Possible_Face_Indices={-2,-1,1,2};
	const std::vector<int> BulkElementQuad2dC2::Possible_Face_Indices={-2,-1,1,2};

	const std::vector<int> BulkElementTri2dC1::Possible_Face_Indices={0,1,2};
	const std::vector<int> BulkElementTri2dC2::Possible_Face_Indices={0,1,2};

	const std::vector<int> BulkElementBrick3dC1::Possible_Face_Indices={-3,-2,-1,1,2,3};
	const std::vector<int> BulkElementBrick3dC2::Possible_Face_Indices={-3,-2,-1,1,2,3};

	const std::vector<int> BulkElementTetra3dC1::Possible_Face_Indices={0,1,2,3};
	const std::vector<int> BulkElementTetra3dC2::Possible_Face_Indices={0,1,2,3};

	const std::vector<int> BulkElementWedge3dC1::Possible_Face_Indices={0,1,2,3,4};
	const std::vector<int> BulkElementWedge3dC2::Possible_Face_Indices={0,1,2,3,4};
	
	const std::vector<int> BulkElementPyramid3dC1::Possible_Face_Indices={0,1,2,3,4};

	// get_vertex_nodes_of_face() implementations below: for each element type and each valid face index
	// (see that type's Possible_Face_Indices table above), return the element's *corner/vertex* nodes
	// bounding that face, in a fixed order (used e.g. to build the face's outline or to identify it geometrically).
	std::vector<pyoomph::Node*> BulkElementLine1dC1::get_vertex_nodes_of_face(const int &face) const
	{
      if (face==-1) return {dynamic_cast<pyoomph::Node*>(this->node_pt(0))};
	  else if (face==1) return {dynamic_cast<pyoomph::Node*>(this->node_pt(1))};	  
	  else throw_runtime_error("Invalid face index for line element");
	}

	std::vector<pyoomph::Node*> BulkElementLine1dC2::get_vertex_nodes_of_face(const int &face) const
	{
	  if (face==-1) return {dynamic_cast<pyoomph::Node*>(this->node_pt(0))};
	  else if (face==1) return {dynamic_cast<pyoomph::Node*>(this->node_pt(2))};
	  else throw_runtime_error("Invalid face index for line element");
	}

	std::vector<pyoomph::Node*> BulkTElementLine1dC1::get_vertex_nodes_of_face(const int &face) const
	{
	  if (face==-1) return {dynamic_cast<pyoomph::Node*>(this->node_pt(0))};
	  else if (face==1) return {dynamic_cast<pyoomph::Node*>(this->node_pt(1))};	  
	  else throw_runtime_error("Invalid face index for line element");
	}

	std::vector<pyoomph::Node*> BulkTElementLine1dC2::get_vertex_nodes_of_face(const int &face) const
	{
	  if (face==-1) return {dynamic_cast<pyoomph::Node*>(this->node_pt(0))};
	  else if (face==1) return {dynamic_cast<pyoomph::Node*>(this->node_pt(2))};
	  else throw_runtime_error("Invalid face index for line element");
	}

	// Developer-only utility (unused by the running code, invoked manually while adding a new element type):
	// prints, to stdout, ready-to-paste C++ source for the "if (face==...) return {...};" body of a new
	// get_vertex_nodes_of_face() override, by inspecting which of the element's face-local nodes are vertex nodes.
	void help_me_with_the_facets(const BulkElementBase *elem,int face_index)
	{

	  unsigned nnode_face;
	  if (dynamic_cast<const BulkElementTetra3dC1*>(elem)) nnode_face=3;
	  else if (dynamic_cast<const BulkElementTetra3dC2TB*>(elem)) nnode_face=7;
	  else if (dynamic_cast<const BulkElementTetra3dC2*>(elem)) nnode_face=6;
	  else if (dynamic_cast<const BulkElementWedge3dC1*>(elem)) {nnode_face=(face_index<2 ? 3 : 4);}
	  else nnode_face=elem->nnode_on_face();	  
	  std::set<oomph::Node*> vertex_nodes;
	 // std::cout << "ELEMENT TYPE " << typeid(*elem).name() << " FACE INDEX " << face_index << " NODES ON FACE " << nnode_face << std::endl;
	  //std::cout << " NNODE " << elem->nnode() << " NVERTEX NODE " << elem->nvertex_node() << std::endl;
	  for (unsigned int i=0;i<elem->nvertex_node();i++)
	  {
	    vertex_nodes.insert(elem->vertex_node_pt(i));
	  }
	  //std::cout << "VERTEX NODES ARE " <<  vertex_nodes.size() << std::endl;
	  std::cout << " if (face=="<<face_index<<") { return {";
	  bool comma=false;
	  for (unsigned i = 0; i < nnode_face; i++)
      {      
        unsigned bulk_number = elem->get_bulk_node_number(face_index, i);
		if (vertex_nodes.count(elem->node_pt(bulk_number)))
		{
			//std::cout << typeid(*elem).name() << " FACE " << face_index << " NODE " << i << " BULK NUMBER " << bulk_number << " IS A VERTEX NODE " << std::endl;
			if (comma) std::cout << ",";
			else comma=true;
			std::cout << "dynamic_cast<pyoomph::Node*>(this->node_pt(" << bulk_number << "))";
		}
	  }     
	  std::cout << "};}" << std::endl; 
    }
	

	std::vector<pyoomph::Node*> BulkElementQuad2dC1::get_vertex_nodes_of_face(const int &face) const
	{	  
	  	if (face==-2) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(0)),dynamic_cast<pyoomph::Node*>(this->node_pt(1))};}
 		else if (face==-1) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(0)),dynamic_cast<pyoomph::Node*>(this->node_pt(2))};}
 		else if (face==1) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(1)),dynamic_cast<pyoomph::Node*>(this->node_pt(3))};}
 		else if (face==2) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(2)),dynamic_cast<pyoomph::Node*>(this->node_pt(3))};}
		else throw_runtime_error("Invalid face index for quadrilateral element");
	}

	std::vector<pyoomph::Node*> BulkElementQuad2dC2::get_vertex_nodes_of_face(const int &face) const
	{	  	  
	  if (face==-2) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(0)),dynamic_cast<pyoomph::Node*>(this->node_pt(2))};}
      else if (face==-1) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(0)),dynamic_cast<pyoomph::Node*>(this->node_pt(6))};}
      else if (face==1) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(2)),dynamic_cast<pyoomph::Node*>(this->node_pt(8))};}
      else if (face==2) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(6)),dynamic_cast<pyoomph::Node*>(this->node_pt(8))};}
	  else throw_runtime_error("Invalid face index for quadrilateral element");
	}

	std::vector<pyoomph::Node*> BulkElementTri2dC1::get_vertex_nodes_of_face(const int &face) const
	{	  
	  if (face==0) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(2)),dynamic_cast<pyoomph::Node*>(this->node_pt(1))};}
      else if (face==1) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(2)),dynamic_cast<pyoomph::Node*>(this->node_pt(0))};}
      else if (face==2) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(0)),dynamic_cast<pyoomph::Node*>(this->node_pt(1))};}
	  else throw_runtime_error("Invalid face index for triangular element");
	}

	std::vector<pyoomph::Node*> BulkElementTri2dC2::get_vertex_nodes_of_face(const int &face) const
	{	  
	  if (face==0) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(2)),dynamic_cast<pyoomph::Node*>(this->node_pt(1))};}
      else if (face==1) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(2)),dynamic_cast<pyoomph::Node*>(this->node_pt(0))};}
      else if (face==2) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(0)),dynamic_cast<pyoomph::Node*>(this->node_pt(1))};}
	  else throw_runtime_error("Invalid face index for triangular element");	  
	}

	std::vector<pyoomph::Node*> BulkElementBrick3dC1::get_vertex_nodes_of_face(const int &face) const
	{	  
	  	if (face==-3) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(0)),dynamic_cast<pyoomph::Node*>(this->node_pt(1)),dynamic_cast<pyoomph::Node*>(this->node_pt(2)),dynamic_cast<pyoomph::Node*>(this->node_pt(3))};}
 		else if (face==-2) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(0)),dynamic_cast<pyoomph::Node*>(this->node_pt(1)),dynamic_cast<pyoomph::Node*>(this->node_pt(4)),dynamic_cast<pyoomph::Node*>(this->node_pt(5))};}
 		else if (face==-1) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(0)),dynamic_cast<pyoomph::Node*>(this->node_pt(2)),dynamic_cast<pyoomph::Node*>(this->node_pt(4)),dynamic_cast<pyoomph::Node*>(this->node_pt(6))};}
 		else if (face==1) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(1)),dynamic_cast<pyoomph::Node*>(this->node_pt(3)),dynamic_cast<pyoomph::Node*>(this->node_pt(5)),dynamic_cast<pyoomph::Node*>(this->node_pt(7))};}
 		else if (face==2) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(2)),dynamic_cast<pyoomph::Node*>(this->node_pt(3)),dynamic_cast<pyoomph::Node*>(this->node_pt(6)),dynamic_cast<pyoomph::Node*>(this->node_pt(7))};}
 		else if (face==3) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(4)),dynamic_cast<pyoomph::Node*>(this->node_pt(5)),dynamic_cast<pyoomph::Node*>(this->node_pt(6)),dynamic_cast<pyoomph::Node*>(this->node_pt(7))};}				
		else throw_runtime_error("Invalid face index for brick element");
	}

	std::vector<pyoomph::Node*> BulkElementBrick3dC2::get_vertex_nodes_of_face(const int &face) const
	{	  
	 if (face==3) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(18)),dynamic_cast<pyoomph::Node*>(this->node_pt(20)),dynamic_cast<pyoomph::Node*>(this->node_pt(24)),dynamic_cast<pyoomph::Node*>(this->node_pt(26))};}
     else if (face==-3) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(0)),dynamic_cast<pyoomph::Node*>(this->node_pt(2)),dynamic_cast<pyoomph::Node*>(this->node_pt(6)),dynamic_cast<pyoomph::Node*>(this->node_pt(8))};}
     else if (face==-2) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(0)),dynamic_cast<pyoomph::Node*>(this->node_pt(2)),dynamic_cast<pyoomph::Node*>(this->node_pt(18)),dynamic_cast<pyoomph::Node*>(this->node_pt(20))};}
 	 else if (face==-1) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(0)),dynamic_cast<pyoomph::Node*>(this->node_pt(6)),dynamic_cast<pyoomph::Node*>(this->node_pt(18)),dynamic_cast<pyoomph::Node*>(this->node_pt(24))};}
     else if (face==1) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(2)),dynamic_cast<pyoomph::Node*>(this->node_pt(8)),dynamic_cast<pyoomph::Node*>(this->node_pt(20)),dynamic_cast<pyoomph::Node*>(this->node_pt(26))};}
     else if (face==2) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(6)),dynamic_cast<pyoomph::Node*>(this->node_pt(8)),dynamic_cast<pyoomph::Node*>(this->node_pt(24)),dynamic_cast<pyoomph::Node*>(this->node_pt(26))};}
     else if (face==3) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(18)),dynamic_cast<pyoomph::Node*>(this->node_pt(20)),dynamic_cast<pyoomph::Node*>(this->node_pt(24)),dynamic_cast<pyoomph::Node*>(this->node_pt(26))};}
	 else throw_runtime_error("Invalid face index for brick element");
	}	

	std::vector<pyoomph::Node*> BulkElementTetra3dC1::get_vertex_nodes_of_face(const int &face) const
	{	  
	  if (face==0) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(1)),dynamic_cast<pyoomph::Node*>(this->node_pt(2)),dynamic_cast<pyoomph::Node*>(this->node_pt(3))};}
      else if (face==1) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(0)),dynamic_cast<pyoomph::Node*>(this->node_pt(2)),dynamic_cast<pyoomph::Node*>(this->node_pt(3))};}
      else if (face==2) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(0)),dynamic_cast<pyoomph::Node*>(this->node_pt(1)),dynamic_cast<pyoomph::Node*>(this->node_pt(3))};}
      else if (face==3) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(1)),dynamic_cast<pyoomph::Node*>(this->node_pt(2)),dynamic_cast<pyoomph::Node*>(this->node_pt(0))};}
	  else throw_runtime_error("Invalid face index for tetrahedral element");
	}

	std::vector<pyoomph::Node*> BulkElementTetra3dC2::get_vertex_nodes_of_face(const int &face) const
	{	  
	  if (face==0) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(1)),dynamic_cast<pyoomph::Node*>(this->node_pt(2)),dynamic_cast<pyoomph::Node*>(this->node_pt(3))};}
      else if (face==1) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(0)),dynamic_cast<pyoomph::Node*>(this->node_pt(2)),dynamic_cast<pyoomph::Node*>(this->node_pt(3))};}
      else if (face==2) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(0)),dynamic_cast<pyoomph::Node*>(this->node_pt(1)),dynamic_cast<pyoomph::Node*>(this->node_pt(3))};}
      else if (face==3) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(1)),dynamic_cast<pyoomph::Node*>(this->node_pt(2)),dynamic_cast<pyoomph::Node*>(this->node_pt(0))};}
	  else throw_runtime_error("Invalid face index for tetrahedral element");
	}

	std::vector<pyoomph::Node*> BulkElementWedge3dC1::get_vertex_nodes_of_face(const int &face) const
	{	  
	  if (face==0) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(0)),dynamic_cast<pyoomph::Node*>(this->node_pt(1)),dynamic_cast<pyoomph::Node*>(this->node_pt(2))};}
      else if (face==1) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(3)),dynamic_cast<pyoomph::Node*>(this->node_pt(4)),dynamic_cast<pyoomph::Node*>(this->node_pt(5))};}
      else if (face==2) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(3)),dynamic_cast<pyoomph::Node*>(this->node_pt(0)),dynamic_cast<pyoomph::Node*>(this->node_pt(5)),dynamic_cast<pyoomph::Node*>(this->node_pt(2))};}
      else if (face==3) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(1)),dynamic_cast<pyoomph::Node*>(this->node_pt(0)),dynamic_cast<pyoomph::Node*>(this->node_pt(4)),dynamic_cast<pyoomph::Node*>(this->node_pt(3))};}
      else if (face==4) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(1)),dynamic_cast<pyoomph::Node*>(this->node_pt(4)),dynamic_cast<pyoomph::Node*>(this->node_pt(2)),dynamic_cast<pyoomph::Node*>(this->node_pt(5))};}
	  else throw_runtime_error("Invalid face index for wedge element");
	}

	std::vector<pyoomph::Node*> BulkElementPyramid3dC1::get_vertex_nodes_of_face(const int &face) const
	{
		if (face==0) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(0)),dynamic_cast<pyoomph::Node*>(this->node_pt(1)),dynamic_cast<pyoomph::Node*>(this->node_pt(4))};}
      else if (face==1) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(1)),dynamic_cast<pyoomph::Node*>(this->node_pt(2)),dynamic_cast<pyoomph::Node*>(this->node_pt(4))};}
      else if (face==2) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(2)),dynamic_cast<pyoomph::Node*>(this->node_pt(3)),dynamic_cast<pyoomph::Node*>(this->node_pt(4))};}
      else if (face==3) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(0)),dynamic_cast<pyoomph::Node*>(this->node_pt(4)),dynamic_cast<pyoomph::Node*>(this->node_pt(3))};}
      else if (face==4) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(0)),dynamic_cast<pyoomph::Node*>(this->node_pt(3)),dynamic_cast<pyoomph::Node*>(this->node_pt(1)),dynamic_cast<pyoomph::Node*>(this->node_pt(2))};}
	  else throw_runtime_error("Invalid face index for pyramid element");
	}

	std::vector<pyoomph::Node*> BulkElementWedge3dC2::get_vertex_nodes_of_face(const int &face) const
	{
		if      (face==0) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(0)),
									dynamic_cast<pyoomph::Node*>(this->node_pt(1)),
									dynamic_cast<pyoomph::Node*>(this->node_pt(2))}; }
		else if (face==1) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(12)),
									dynamic_cast<pyoomph::Node*>(this->node_pt(13)),
									dynamic_cast<pyoomph::Node*>(this->node_pt(14))}; }
		else if (face==2) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(12)),
									dynamic_cast<pyoomph::Node*>(this->node_pt(0)),
									dynamic_cast<pyoomph::Node*>(this->node_pt(14)),
									dynamic_cast<pyoomph::Node*>(this->node_pt(2))}; }
		else if (face==3) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(1)),
									dynamic_cast<pyoomph::Node*>(this->node_pt(0)),
									dynamic_cast<pyoomph::Node*>(this->node_pt(13)),
									dynamic_cast<pyoomph::Node*>(this->node_pt(12))}; }
		else if (face==4) { return {dynamic_cast<pyoomph::Node*>(this->node_pt(1)),
									dynamic_cast<pyoomph::Node*>(this->node_pt(13)),
									dynamic_cast<pyoomph::Node*>(this->node_pt(2)),
									dynamic_cast<pyoomph::Node*>(this->node_pt(14))}; }
		else throw_runtime_error("Invalid face index for wedge element");
  	}

	//oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;

	// construct_face_element() implementations: create the appropriate InterfaceElement<...> FaceElement type
	// for a given bulk element and face index (e.g. a quad's face is a line, a tet's face is a triangle);
	// most bulk types have a single, fixed face-element type, but pyramids/wedges mix triangular and
	// quadrilateral faces and therefore dispatch on face_index below.
	oomph::FaceElement * BulkElementQuad2dC2::construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) { return new InterfaceElementLine1dC2(jitcode, this, face_index); }
	oomph::FaceElement * BulkElementQuad2dC1::construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) { return new InterfaceElementLine1dC1(jitcode, this, face_index); }
    oomph::FaceElement * BulkElementTri2dC1::construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) { return new InterfaceTElementLine1dC1(jitcode, this, face_index); }
	oomph::FaceElement * BulkElementTri2dC2::construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) { return new InterfaceTElementLine1dC2(jitcode, this, face_index); }

	oomph::FaceElement * BulkElementLine1dC1::construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) { return new InterfaceElementPoint0d(jitcode, this, face_index); }
	oomph::FaceElement * BulkElementLine1dC2::construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) { return new InterfaceElementPoint0d(jitcode, this, face_index); }
	oomph::FaceElement * BulkTElementLine1dC1::construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) { return new InterfaceElementPoint0d(jitcode, this, face_index); }
	oomph::FaceElement * BulkTElementLine1dC2::construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) { return new InterfaceElementPoint0d(jitcode, this, face_index); }

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
		// throw_runtime_error("TODO: Constructing face elements for pyramid elements with C1 interpolation on the face. The different face types depend on the face index");
	}
	
	oomph::FaceElement * BulkElementWedge3dC2::construct_face_element(DynamicBulkElementInstance *jitcode, int face_index)
	{
		if (face_index<2) return  new InterfaceElementTri2dC2(jitcode, this, face_index);
	    else return new InterfaceElementQuad2dC2(jitcode, this, face_index);					
	}
    

}
