/*================================================================================
pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
Copyright (C) 2021-2025  Christian Diddens & Duarte Rocha

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

The authors may be contacted at c.diddens@utwente.nl and d.rocha@utwente.nl

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

	template <typename T>
	void my_alloc(T dest) {}

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

	template <typename T, typename... Extra>
	void my_alloc_or_free(bool alloc, T * PYOOMPH_RESTRICT &  dest, size_t firstdim, const Extra &...extra)
	{
		if (alloc)
			my_alloc(dest, firstdim, extra...);
		else
			my_free(dest, firstdim, extra...);
	}

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
				else
					return T3d;
			}
		}
		else
		{
			if (edim == 1)
				return Q1d;
			else if (edim == 2)
				return Q2d;
			else
				return Q3d;
		}
	}

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
	}

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

		T3dTB[3] = new oomph::TBubbleEnrichedGauss<3, 3>();
	}

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
			// std::cout << "FOR " << (tris ? "TRI" : "QUAD") << " OF DIM " << edim << " ORDER " << order << " WE SELECTED " << typeid(ret).name();
			return ret;
		}
		// std::cout << "FOR " << (tris ? "TRI" : "QUAD") << " OF DIM " << edim << " ORDER " << order << " WE SELECTED MAX ORDER " << maxorder << " IE " << typeid(map[maxorder]).name();
		return map[maxorder];
	}

	JITShapeInfo_t *Default_shape_info_buffer = NULL;
	DynamicBulkElementInstance *BulkElementBase::__CurrentCodeInstance = NULL;
	unsigned BulkElementBase::zeta_time_history = 0;
	unsigned BulkElementBase::zeta_coordinate_type = 0; // 0 means Lagrangian, 1 Eulerian, on co-dimensional meshes it will be the boundary coordinate (if set)
	bool BulkElementBase::use_eigen_error_estimators=false;

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
		//  std::cout << "BLA " << size/weightsum << "  " << minJ << std::endl;
		return minJ / (size / weightsum);
	}

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

	unsigned BulkElementBase::ndof_types() const
	{
		auto *ft = codeinst->get_func_table();
		unsigned res = ft->numfields_C2TB + ft->numfields_C2 + ft->numfields_C1 + ft->numfields_D1 + ft->numfields_D2 +ft->numfields_D2TB+ft->numfields_D1TB+  ft->numfields_DL + ft->numfields_D0 ; // TODO: Optionally, allow for splitting of the DL space
		if (ft->moving_nodes)
		{
			res += this->nodal_dimension(); // position dofs
		}
		return res;
	}

	void BulkElementBase::fill_dofs_to_dirichlet_set_indices(std::vector<unsigned> &indices)
	{
		throw_runtime_error("Implement");
	}
	
	void BulkElementBase::get_dof_numbers_for_unknowns(std::list<std::pair<unsigned long, unsigned>> &dof_lookup_list) const
	{

		auto *ft = codeinst->get_func_table();
		//   unsigned n_node = this->nnode();
		std::pair<unsigned, unsigned> dof_lookup;
		unsigned cnt = 0;
	   for (unsigned f = 0; f < ft->numfields_C2TB_basebulk; f++)
		{
			unsigned nind = f;
			for (unsigned l = 0; l < eleminfo.nnode_C2TB; l++)
			{
				int local_eqn_number = this->nodal_local_eqn(this->get_node_index_C2TB_to_element(l), nind);
				if (local_eqn_number >= 0)
				{
					dof_lookup.first = this->eqn_number(local_eqn_number);
					dof_lookup.second = cnt;
					dof_lookup_list.push_front(dof_lookup);
				}
			}
			cnt++;
		}
				
		for (unsigned f = 0; f < ft->numfields_C2_basebulk; f++)
		{
			unsigned nind = f;
			for (unsigned l = 0; l < eleminfo.nnode_C2; l++)
			{
				int local_eqn_number = this->nodal_local_eqn(this->get_node_index_C2_to_element(l), nind);
				if (local_eqn_number >= 0)
				{
					dof_lookup.first = this->eqn_number(local_eqn_number);
					dof_lookup.second = cnt;
					dof_lookup_list.push_front(dof_lookup);
				}
			}
			cnt++;
		}

		for (unsigned f = 0; f < ft->numfields_C1_basebulk; f++)
		{
			unsigned nind = f + ft->numfields_C2_basebulk;
			for (unsigned l = 0; l < this->eleminfo.nnode_C2; l++)
			{
				int local_eqn_number = this->nodal_local_eqn(this->get_node_index_C1_to_element(l), nind);
				if (local_eqn_number >= 0)
				{
					dof_lookup.first = this->eqn_number(local_eqn_number);
					dof_lookup.second = cnt;
					dof_lookup_list.push_front(dof_lookup);
				}
			}
			cnt++;
		}
		
      // TODO: DG				
		// TODO: Interface stuff

		for (unsigned f = 0; f < ft->numfields_DL; f++)
		{
			unsigned nind = f;
			for (unsigned l = 0; l < eleminfo.nnode_DL; l++)
			{
				int local_eqn_number = this->internal_local_eqn(nind, l);
				if (local_eqn_number >= 0)
				{
					dof_lookup.first = this->eqn_number(local_eqn_number);
					dof_lookup.second = cnt;
					dof_lookup_list.push_front(dof_lookup);
				}
			}
			cnt++;
		}

		for (unsigned f = 0; f < ft->numfields_D0; f++)
		{
			unsigned nind = f + ft->numfields_DL;
			int local_eqn_number = this->internal_local_eqn(nind, 0);
			if (local_eqn_number >= 0)
			{
				dof_lookup.first = this->eqn_number(local_eqn_number);
				dof_lookup.second = cnt;
				dof_lookup_list.push_front(dof_lookup);
			}
			cnt++;
		}
	}

	void BulkElementBase::interpolate_hang_values()
	{

		for (unsigned l = 0; l < this->nnode(); l++)
		{
			pyoomph::Node *n = dynamic_cast<pyoomph::Node *>(this->node_pt(l));
			if (n->is_hanging())
			{
				for (unsigned i = 0; i < n->ndim(); i++)
				{
					//         	double old=n->x(i);
					n->x(i) = n->position(i); // Use the hang values to interpolate
											  /*
											  if (old!=n->x(i))
											  {
												  std::cout << "CHANGE in coordinade " << l << "  " << i << "   " << old-n->x(i) << std::endl;
											  }
											  */
				}
			}
		}
	}
	
	std::vector<std::pair<oomph::Data*,int> > BulkElementBase::get_field_data_list(std::string name,bool use_elemental_indices)
	{
	 auto *ft=codeinst->get_func_table();
	 std::vector<std::pair<oomph::Data*,int> > result;
	 auto find_by_name=[name](char **fnames,unsigned numf)->int {for(unsigned int i=0;i<numf;i++) if (name==std::string(fnames[i])) return i;  return -1;};
	 int ind;
	 if (ft->numfields_C2TB && ((ind=find_by_name(ft->fieldnames_C2TB,ft->numfields_C2TB))>=0))
	 {
		if (ind<(int)ft->numfields_C2TB_basebulk)
		{
			if (!use_elemental_indices)
			{
			  for (unsigned int i=0;i<eleminfo.nnode_C2TB;i++) result.push_back(std::make_pair(this->node_pt(this->get_node_index_C2TB_to_element(i)),ind+ft->nodal_offset_C2TB_basebulk));
			}
			else
			{
			  for (unsigned int i=0;i<eleminfo.nnode;i++) 
			  {
				int nind=this->get_node_index_element_to_C2TB(i);
				if (nind>=0)
				{
					result.push_back(std::make_pair(this->node_pt(i),ind+ft->nodal_offset_C2TB_basebulk));
				}
				else
				{
					result.push_back(std::make_pair(nullptr,-1));
				}
			  }
			}
		}
		else
		{
			unsigned interf_id = codeinst->resolve_interface_dof_id(name);
			if (!use_elemental_indices)
			{
			  for (unsigned int i=0;i<eleminfo.nnode_C2TB;i++) 
			  {
				auto *n=this->node_pt(this->get_node_index_C2TB_to_element(i));
				result.push_back(std::make_pair(n,dynamic_cast<oomph::BoundaryNodeBase *>(n)->index_of_first_value_assigned_by_face_element(interf_id)));
			  }
			}
			else
			{
			  for (unsigned int i=0;i<eleminfo.nnode;i++) 
			  {
				int nind=this->get_node_index_element_to_C2TB(i);
				if (nind>=0)
				{
					auto *n=this->node_pt(i);
					result.push_back(std::make_pair(n,dynamic_cast<oomph::BoundaryNodeBase *>(n)->index_of_first_value_assigned_by_face_element(interf_id)));
				}
				else
				{
					result.push_back(std::make_pair(nullptr,-1));
				}
			  }
			}
		}
	 }
	 else if (ft->numfields_C2 && ((ind=find_by_name(ft->fieldnames_C2,ft->numfields_C2))>=0))
	 {
		if (ind<(int)ft->numfields_C2_basebulk)
		{
			if (!use_elemental_indices)
			{
			  for (unsigned int i=0;i<eleminfo.nnode_C2;i++) result.push_back(std::make_pair(this->node_pt(this->get_node_index_C2_to_element(i)),ind+ft->nodal_offset_C2_basebulk));
			}
			else
			{
			  for (unsigned int i=0;i<eleminfo.nnode;i++) 
			  {
				int nind=this->get_node_index_element_to_C2(i);
				if (nind>=0)
				{
					result.push_back(std::make_pair(this->node_pt(i),ind+ft->nodal_offset_C2_basebulk));
				}
				else
				{
					result.push_back(std::make_pair(nullptr,-1));
				}
			  }
			}
		}
		else
		{
			unsigned interf_id = codeinst->resolve_interface_dof_id(name);
			if (!use_elemental_indices)
			{
			  for (unsigned int i=0;i<eleminfo.nnode_C2;i++) 
			  {
				auto *n=this->node_pt(this->get_node_index_C2_to_element(i));
				result.push_back(std::make_pair(n,dynamic_cast<oomph::BoundaryNodeBase *>(n)->index_of_first_value_assigned_by_face_element(interf_id)));
			  }
			}
			else
			{
			  for (unsigned int i=0;i<eleminfo.nnode;i++) 
			  {
				int nind=this->get_node_index_element_to_C2(i);
				if (nind>=0)
				{
					auto *n=this->node_pt(i);
					result.push_back(std::make_pair(n,dynamic_cast<oomph::BoundaryNodeBase *>(n)->index_of_first_value_assigned_by_face_element(interf_id)));
				}
				else
				{
					result.push_back(std::make_pair(nullptr,-1));
				}
			  }
			}
		}
	 }
    else if (ft->numfields_C1TB && ((ind=find_by_name(ft->fieldnames_C1TB,ft->numfields_C1TB))>=0))
	 {
		if (ind<(int)ft->numfields_C1TB_basebulk)
		{
			if (!use_elemental_indices)
			{
			  for (unsigned int i=0;i<eleminfo.nnode_C1TB;i++) result.push_back(std::make_pair(this->node_pt(this->get_node_index_C1TB_to_element(i)),ind+ft->nodal_offset_C1TB_basebulk));
			}
			else
			{
			  for (unsigned int i=0;i<eleminfo.nnode;i++) 
			  {
				int nind=this->get_node_index_element_to_C1TB(i);
				if (nind>=0)
				{
					result.push_back(std::make_pair(this->node_pt(i),ind+ft->nodal_offset_C1TB_basebulk));
				}
				else
				{
					result.push_back(std::make_pair(nullptr,-1));
				}
			  }
			}
		}
		else
		{
			unsigned interf_id = codeinst->resolve_interface_dof_id(name);
			if (!use_elemental_indices)
			{
			  for (unsigned int i=0;i<eleminfo.nnode_C1TB;i++) 
			  {
				auto *n=this->node_pt(this->get_node_index_C1TB_to_element(i));
				result.push_back(std::make_pair(n,dynamic_cast<oomph::BoundaryNodeBase *>(n)->index_of_first_value_assigned_by_face_element(interf_id)));
			  }
			}
			else
			{
			  for (unsigned int i=0;i<eleminfo.nnode;i++) 
			  {
				int nind=this->get_node_index_element_to_C1TB(i);
				if (nind>=0)
				{
					auto *n=this->node_pt(i);
					result.push_back(std::make_pair(n,dynamic_cast<oomph::BoundaryNodeBase *>(n)->index_of_first_value_assigned_by_face_element(interf_id)));
				}
				else
				{
					result.push_back(std::make_pair(nullptr,-1));
				}
			  }
			}
		}
	 }	 
	 else if (ft->numfields_C1 && ((ind=find_by_name(ft->fieldnames_C1,ft->numfields_C1))>=0))
	 {
		if (ind<(int)ft->numfields_C1_basebulk)
		{
			if (!use_elemental_indices)
			{
			  for (unsigned int i=0;i<eleminfo.nnode_C1;i++) result.push_back(std::make_pair(this->node_pt(this->get_node_index_C1_to_element(i)),ind+ft->nodal_offset_C1_basebulk));
			}
			else
			{
			  for (unsigned int i=0;i<eleminfo.nnode;i++) 
			  {
				int nind=this->get_node_index_element_to_C1(i);
				if (nind>=0)
				{
					result.push_back(std::make_pair(this->node_pt(i),ind+ft->nodal_offset_C1_basebulk));
				}
				else
				{
					result.push_back(std::make_pair(nullptr,-1));
				}
			  }
			}
		}
		else
		{
			unsigned interf_id = codeinst->resolve_interface_dof_id(name);
			if (!use_elemental_indices)
			{
			  for (unsigned int i=0;i<eleminfo.nnode_C1;i++) 
			  {
				auto *n=this->node_pt(this->get_node_index_C1_to_element(i));
				result.push_back(std::make_pair(n,dynamic_cast<oomph::BoundaryNodeBase *>(n)->index_of_first_value_assigned_by_face_element(interf_id)));
			  }
			}
			else
			{
			  for (unsigned int i=0;i<eleminfo.nnode;i++) 
			  {
				int nind=this->get_node_index_element_to_C1(i);
				if (nind>=0)
				{
					auto *n=this->node_pt(i);
					result.push_back(std::make_pair(n,dynamic_cast<oomph::BoundaryNodeBase *>(n)->index_of_first_value_assigned_by_face_element(interf_id)));
				}
				else
				{
					result.push_back(std::make_pair(nullptr,-1));
				}
			  }
			}
		}
	 }
	 else if (ft->numfields_D2TB && ((ind=find_by_name(ft->fieldnames_D2TB,ft->numfields_D2TB))>=0))	 
	 {	 
        oomph::Data * data=this->get_D2TB_nodal_data(ind);
	    if (!use_elemental_indices)
		{
			for (unsigned int i=0;i<eleminfo.nnode_C2TB;i++) 
			{
				result.push_back(std::make_pair(data,this->get_D2TB_node_index(ind,i)));
			}
		}
		else
		{
			for (unsigned int i=0;i<eleminfo.nnode;i++) 
			{
				int nind=this->get_node_index_element_to_C2TB(i);
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
	 }
	 else if (ft->numfields_D2 && ((ind=find_by_name(ft->fieldnames_D2,ft->numfields_D2))>=0))
	 {	 
        oomph::Data * data=this->get_D2_nodal_data(ind);
	    if (!use_elemental_indices)
		{
			for (unsigned int i=0;i<eleminfo.nnode_C2;i++) 
			{
				result.push_back(std::make_pair(data,this->get_D2_node_index(ind,i)));
			}
		}
		else
		{
			for (unsigned int i=0;i<eleminfo.nnode;i++) 
			{
				int nind=this->get_node_index_element_to_C2(i);
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
	 }
	 else if (ft->numfields_D1TB && ((ind=find_by_name(ft->fieldnames_D1TB,ft->numfields_D1TB))>=0))
	 {	 
        oomph::Data * data=this->get_D1TB_nodal_data(ind);
	    if (!use_elemental_indices)
		{
			for (unsigned int i=0;i<eleminfo.nnode_C1TB;i++) 
			{
				result.push_back(std::make_pair(data,this->get_D1TB_node_index(ind,i)));
			}
		}
		else
		{
			for (unsigned int i=0;i<eleminfo.nnode;i++) 
			{
				int nind=this->get_node_index_element_to_C1TB(i);
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
	 }	 
	 else if (ft->numfields_D1 && ((ind=find_by_name(ft->fieldnames_D1,ft->numfields_D1))>=0))
	 {	 
        oomph::Data * data=this->get_D1_nodal_data(ind);
	    if (!use_elemental_indices)
		{
			for (unsigned int i=0;i<eleminfo.nnode_C1;i++) 
			{
				result.push_back(std::make_pair(data,this->get_D1_node_index(ind,i)));
			}
		}
		else
		{
			for (unsigned int i=0;i<eleminfo.nnode;i++) 
			{
				int nind=this->get_node_index_element_to_C1(i);
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
	 }
	 else if (name=="mesh_x" && this->nodal_dimension()>0)
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

	bool BulkElementBase::fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap)
	{
		if (eqn_remap)
		{
		   auto * ft=codeinst->get_func_table();
			// std::cout << "APPlYING REMAP " << std::endl;
			if (ft->moving_nodes)
			{

				unsigned nfields = this->nodal_dimension();

				for (unsigned int l = 0; l < eleminfo.nnode; l++) // Lagrangian part
				{

					//    std::cout << "POSNMASTER " << l <<  shape_info->hanginfo_Pos[l].nummaster << std::endl;
					if (!shape_info->hanginfo_Pos[l].nummaster)
					{
						// NON HANGING -> HANGING WITH WEIGHT 1 for external element data
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
									throw_runtime_error("MISSING EXTERNAL POS DEPENDENCY ON ELEM PTR: " + oss.str() + "\nThis is part of the Lagrangian field index " + std::to_string(f) + " of " + std::to_string(nfields) + " at node " + std::to_string(l));
								}
							}
						}
					}
				}
			}


			if (required.dx_psi_C2TB || required.psi_C2TB || required.dX_psi_C2TB)
			{
				for (unsigned int l = 0; l < eleminfo.nnode_C2TB; l++) // C2TB nodes
				{

					if (!shape_info->hanginfo_C2TB[l].nummaster)
					{
						// NON HANGING -> HANGING WITH WEIGHT 1 for external element data
						shape_info->hanginfo_C2TB[l].nummaster = 1;
						shape_info->hanginfo_C2TB[l].masters[0].weight = 1.0;
						for (unsigned int f = 0; f < ft->numfields_C2TB_basebulk; f++)
						{
							if (eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_C2TB_basebulk] >= 0)
							{
								shape_info->hanginfo_C2TB[l].masters[0].local_eqn[f+ ft->buffer_offset_C2TB_basebulk] = eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_C2TB_basebulk];							
							}
							else
							{
								shape_info->hanginfo_C2TB[l].masters[0].local_eqn[f+ ft->buffer_offset_C2TB_basebulk] = -1;
							}
						}
						for (unsigned int f = 0; f < ft->numfields_C2TB-ft->numfields_C2TB_basebulk; f++)
						{
							if (eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_C2TB_interf] >= 0)
							{
								shape_info->hanginfo_C2TB[l].masters[0].local_eqn[f+ ft->buffer_offset_C2TB_interf] = eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_C2TB_interf];							
							}
							else
							{
								shape_info->hanginfo_C2TB[l].masters[0].local_eqn[f+ ft->buffer_offset_C2TB_interf] = -1;
							}
						}						
					}
					for (int m = 0; m < shape_info->hanginfo_C2TB[l].nummaster; m++)
					{
						for (unsigned int f = 0; f < ft->numfields_C2TB_basebulk; f++)
						{
						    unsigned foffs=f+ ft->buffer_offset_C2TB_basebulk;
							if (shape_info->hanginfo_C2TB[l].masters[m].local_eqn[foffs] >= 0)
							{
							   int oldeq=shape_info->hanginfo_C2TB[l].masters[m].local_eqn[foffs];
/*							   if (shape_info->hanginfo_C2TB[l].masters[m].local_eqn[foffs]>=eqn_remap.size())
							   {
							    throw_runtime_error("Problem remapping C2TB dof, local eq is "+std::to_string(shape_info->hanginfo_C2TB[l].masters[m].local_eqn[foffs])+" but the eqn_remap buffer only has "+std::to_string(eqn_remap.size())+" entri
							   }*/
								shape_info->hanginfo_C2TB[l].masters[m].local_eqn[foffs] = eqn_remap[shape_info->hanginfo_C2TB[l].masters[m].local_eqn[foffs]];								
								if (shape_info->hanginfo_C2TB[l].masters[m].local_eqn[foffs] == -666)
								{
									std::ostringstream oss;
									oss << this;
									oss << " node: " << l << ", master " << m << " of " << shape_info->hanginfo_C2TB[l].nummaster  << ", index " << f << ", " << foffs << " of " << codeinst->get_func_table()->numfields_C2TB_basebulk << std::endl;
									oss << " trying to resolve equation  " << oldeq << " which gave  " << shape_info->hanginfo_C2TB[l].masters[m].local_eqn[foffs] << std::endl;
									oss << " AT File" << codeinst->get_code()->get_file_name();
									throw_runtime_error("MISSING EXTERNAL C2TB DEPENDENCY ON ELEM PTR: " + oss.str());
								}
							}
						}
						for (unsigned int f = 0; f < ft->numfields_C2TB-ft->numfields_C2TB_basebulk; f++)
						{
							unsigned foffs=f+ ft->buffer_offset_C2TB_interf;
							if (shape_info->hanginfo_C2TB[l].masters[m].local_eqn[foffs] >= 0)
							{
								//								std::cout << " MAPPED  " << eqn_remap[shape_info->hanginfo_C1[l].masters[m].local_eqn[f]] << std::endl;
								shape_info->hanginfo_C2TB[l].masters[m].local_eqn[foffs] = eqn_remap[shape_info->hanginfo_C2TB[l].masters[m].local_eqn[foffs]];
								if (shape_info->hanginfo_C2TB[l].masters[m].local_eqn[foffs] == -666)
								{
									std::ostringstream oss;
									oss << this;
									throw_runtime_error("MISSING EXTERNAL ADD_INTERFACE C2TB DEPENDENCY ON ELEM PTR: " + oss.str());
								}
							}
						}	
					}
				}
			}

			if (required.dx_psi_C2 || required.psi_C2 || required.dX_psi_C2)
			{
				for (unsigned int l = 0; l < eleminfo.nnode_C2; l++) // C2 nodes
				{

					//std::cout << "C2NMASTER " << l <<  shape_info->hanginfo_C2[l].nummaster << std::endl;
					if (!shape_info->hanginfo_C2[l].nummaster)
					{
						// NON HANGING -> HANGING WITH WEIGHT 1 for external element data
						shape_info->hanginfo_C2[l].nummaster = 1;
						shape_info->hanginfo_C2[l].masters[0].weight = 1.0;
						for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_C2_basebulk; f++)
						{
							if (eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_C2_basebulk] >= 0)
							{
								shape_info->hanginfo_C2[l].masters[0].local_eqn[f+ ft->buffer_offset_C2_basebulk] = eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_C2_basebulk];
							}
							else
							{
								shape_info->hanginfo_C2[l].masters[0].local_eqn[f+ ft->buffer_offset_C2_basebulk] = -1;
							}
						}
						for (unsigned int f = 0; f < ft->numfields_C2-ft->numfields_C2_basebulk; f++)
						{
							if (eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_C2_interf] >= 0)
							{
								shape_info->hanginfo_C2[l].masters[0].local_eqn[f+ ft->buffer_offset_C2_interf] = eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_C2_interf];							
							}
							else
							{
								shape_info->hanginfo_C2[l].masters[0].local_eqn[f+ ft->buffer_offset_C2_interf] = -1;
							}
						}	
					}
					for (int m = 0; m < shape_info->hanginfo_C2[l].nummaster; m++)
					{
						for (unsigned int f = 0; f < ft->numfields_C2_basebulk; f++)
						{
						   unsigned foffs=f+ ft->buffer_offset_C2_basebulk;
							if (shape_info->hanginfo_C2[l].masters[m].local_eqn[foffs] >= 0)
							{
								int orig_eq=shape_info->hanginfo_C2[l].masters[m].local_eqn[foffs];
								shape_info->hanginfo_C2[l].masters[m].local_eqn[foffs] = eqn_remap[shape_info->hanginfo_C2[l].masters[m].local_eqn[foffs]];
								if (shape_info->hanginfo_C2[l].masters[m].local_eqn[foffs] == -666)
								{
									std::ostringstream oss;
									oss << this;
									oss << " node: " << l << ", master " << m << ", index " << f << ", " << foffs << " of " << ft->numfields_C2_basebulk << " Equation to remap is " << orig_eq << std::endl;
									oss << " AT File  " << codeinst->get_code()->get_file_name();
									throw_runtime_error("MISSING EXTERNAL C2 DEPENDENCY ON ELEM PTR: " + oss.str());
								}
							}
						}
						for (unsigned int f = 0; f < ft->numfields_C2-ft->numfields_C2_basebulk; f++)
						{
							unsigned foffs=f+ ft->buffer_offset_C2_interf;
							if (shape_info->hanginfo_C2[l].masters[m].local_eqn[foffs] >= 0)
							{
								//								std::cout << " MAPPED  " << eqn_remap[shape_info->hanginfo_C1[l].masters[m].local_eqn[f]] << std::endl;
								shape_info->hanginfo_C2[l].masters[m].local_eqn[foffs] = eqn_remap[shape_info->hanginfo_C2[l].masters[m].local_eqn[foffs]];
								if (shape_info->hanginfo_C2[l].masters[m].local_eqn[foffs] == -666)
								{
									std::ostringstream oss;
									oss << this;
									throw_runtime_error("MISSING EXTERNAL ADD_INTERFACE C2 DEPENDENCY ON ELEM PTR: " + oss.str());
								}
							}
						}	
					}
							
				}
			}

         if (required.dx_psi_C1TB || required.psi_C1TB || required.dX_psi_C1TB)
			{
				for (unsigned int l = 0; l < eleminfo.nnode_C1TB; l++) // C1TB nodes
				{

					if (!shape_info->hanginfo_C1TB[l].nummaster)
					{
						// NON HANGING -> HANGING WITH WEIGHT 1 for external element data
						shape_info->hanginfo_C1TB[l].nummaster = 1;
						shape_info->hanginfo_C1TB[l].masters[0].weight = 1.0;
						for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_C1TB_basebulk; f++)
						{
							if (eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_C1TB_basebulk] >= 0)
							{
								shape_info->hanginfo_C1TB[l].masters[0].local_eqn[f + ft->buffer_offset_C1TB_basebulk] = eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_C1TB_basebulk];
							}
							else
							{
								shape_info->hanginfo_C1TB[l].masters[0].local_eqn[f + ft->buffer_offset_C1TB_basebulk] = -1;
							}
						}
						for (unsigned int f = 0; f < ft->numfields_C1TB-ft->numfields_C1TB_basebulk; f++)
						{
							if (eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_C1TB_interf] >= 0)
							{
								shape_info->hanginfo_C1TB[l].masters[0].local_eqn[f+ ft->buffer_offset_C1TB_interf] = eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_C1TB_interf];							
							}
							else
							{
								shape_info->hanginfo_C1TB[l].masters[0].local_eqn[f+ ft->buffer_offset_C1TB_interf] = -1;
							}
						}							
					}
					else
					{
	//									   std::cout << " SHAPEINFO ALREADY HANGING " << l << std::endl;
					}
					for (int m = 0; m < shape_info->hanginfo_C1TB[l].nummaster; m++)
					{
						for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_C1TB_basebulk; f++)
						{
						   unsigned foffs=f+ ft->buffer_offset_C1TB_basebulk;
							if (shape_info->hanginfo_C1TB[l].masters[m].local_eqn[foffs] >= 0)
							{
								//								std::cout << " MAPPED  " << eqn_remap[shape_info->hanginfo_C1[l].masters[m].local_eqn[f]] << std::endl;
								shape_info->hanginfo_C1TB[l].masters[m].local_eqn[foffs] = eqn_remap[shape_info->hanginfo_C1TB[l].masters[m].local_eqn[foffs]];
								if (shape_info->hanginfo_C1TB[l].masters[m].local_eqn[foffs] == -666)
								{
									std::ostringstream oss;
									oss << this;
									oss << " At field " << f << " of " <<  codeinst->get_func_table()->numfields_C1TB_basebulk << " with " << m << " of nummaster " << shape_info->hanginfo_C1TB[l].nummaster << " and l= " << l << " of " << eleminfo.nnode_C1TB << std::endl;
									throw_runtime_error("MISSING EXTERNAL C1TB DEPENDENCY ON ELEM PTR: " + oss.str());
								}
							}
						}
						for (unsigned int f = 0; f < ft->numfields_C1TB-ft->numfields_C1TB_basebulk; f++)
						{
							unsigned foffs=f+ ft->buffer_offset_C1TB_interf;
							if (shape_info->hanginfo_C1TB[l].masters[m].local_eqn[foffs] >= 0)
							{
								//								std::cout << " MAPPED  " << eqn_remap[shape_info->hanginfo_C1[l].masters[m].local_eqn[f]] << std::endl;
								shape_info->hanginfo_C1TB[l].masters[m].local_eqn[foffs] = eqn_remap[shape_info->hanginfo_C1TB[l].masters[m].local_eqn[foffs]];
								if (shape_info->hanginfo_C1TB[l].masters[m].local_eqn[foffs] == -666)
								{
									std::ostringstream oss;
									oss << this;
									throw_runtime_error("MISSING EXTERNAL ADD_INTERFACE C1TB DEPENDENCY ON ELEM PTR: " + oss.str());
								}
							}
						}								
					}
					
					
				}
			}

			if (required.dx_psi_C1 || required.psi_C1 || required.dX_psi_C1)
			{
				for (unsigned int l = 0; l < eleminfo.nnode_C1; l++) // C1 nodes
				{

					if (!shape_info->hanginfo_C1[l].nummaster)
					{
						// NON HANGING -> HANGING WITH WEIGHT 1 for external element data
						shape_info->hanginfo_C1[l].nummaster = 1;
						shape_info->hanginfo_C1[l].masters[0].weight = 1.0;
						for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_C1_basebulk; f++)
						{
							if (eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_C1_basebulk] >= 0)
							{
								shape_info->hanginfo_C1[l].masters[0].local_eqn[f + ft->buffer_offset_C1_basebulk] = eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_C1_basebulk];
							}
							else
							{
								shape_info->hanginfo_C1[l].masters[0].local_eqn[f + ft->buffer_offset_C1_basebulk] = -1;
							}
						}
						for (unsigned int f = 0; f < ft->numfields_C1-ft->numfields_C1_basebulk; f++)
						{
							if (eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_C1_interf] >= 0)
							{
								shape_info->hanginfo_C1[l].masters[0].local_eqn[f+ ft->buffer_offset_C1_interf] = eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_C1_interf];							
							}
							else
							{
								shape_info->hanginfo_C1[l].masters[0].local_eqn[f+ ft->buffer_offset_C1_interf] = -1;
							}
						}							
					}
					for (int m = 0; m < shape_info->hanginfo_C1[l].nummaster; m++)
					{
						for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_C1_basebulk; f++)
						{
						   unsigned foffs=f+ ft->buffer_offset_C1_basebulk;
							if (shape_info->hanginfo_C1[l].masters[m].local_eqn[foffs] >= 0)
							{
								//								std::cout << " MAPPED  " << eqn_remap[shape_info->hanginfo_C1[l].masters[m].local_eqn[f]] << std::endl;
								shape_info->hanginfo_C1[l].masters[m].local_eqn[foffs] = eqn_remap[shape_info->hanginfo_C1[l].masters[m].local_eqn[foffs]];
								if (shape_info->hanginfo_C1[l].masters[m].local_eqn[foffs] == -666)
								{
									std::ostringstream oss;
									oss << this;
									throw_runtime_error("MISSING EXTERNAL C1 DEPENDENCY ON ELEM PTR: " + oss.str());
								}
							}
						}
						for (unsigned int f = 0; f < ft->numfields_C1-ft->numfields_C1_basebulk; f++)
						{
							unsigned foffs=f+ ft->buffer_offset_C1_interf;
							if (shape_info->hanginfo_C1[l].masters[m].local_eqn[foffs] >= 0)
							{
								//								std::cout << " MAPPED  " << eqn_remap[shape_info->hanginfo_C1[l].masters[m].local_eqn[f]] << std::endl;
								shape_info->hanginfo_C1[l].masters[m].local_eqn[foffs] = eqn_remap[shape_info->hanginfo_C1[l].masters[m].local_eqn[foffs]];
								if (shape_info->hanginfo_C1[l].masters[m].local_eqn[foffs] == -666)
								{
									std::ostringstream oss;
									oss << this;
									throw_runtime_error("MISSING EXTERNAL ADD_INTERFACE C1 DEPENDENCY ON ELEM PTR: " + oss.str());
								}
							}
						}								
					}
					
					
				}
			}

			if (codeinst->get_func_table()->numfields_D2TB && (required.dx_psi_C2TB || required.psi_C2TB || required.dX_psi_C2TB))
			{
				for (unsigned int l = 0; l < eleminfo.nnode_C2TB; l++)
				{
					if (!shape_info->hanginfo_Discont[l].nummaster)
					{
						// NON HANGING -> HANGING WITH WEIGHT 1 for external element data
						shape_info->hanginfo_Discont[l].nummaster = 1;
						shape_info->hanginfo_Discont[l].masters[0].weight = 1.0;
					}
					for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_D2TB_basebulk; f++)
					{
					      int eq=eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_D2TB_basebulk];
					      
							if (eq >= 0)
							{
							   eq=eqn_remap[eq];
							   if (eq==-666)
							   {
									std::ostringstream oss;
									oss << this;
									throw_runtime_error("MISSING EXTERNAL D2TB DEPENDENCY ON ELEM PTR: " + oss.str());
							   }
								shape_info->hanginfo_Discont[l].masters[0].local_eqn[f + ft->buffer_offset_D2TB_basebulk] = eq ;
							}
							else
							{
								shape_info->hanginfo_Discont[l].masters[0].local_eqn[f + ft->buffer_offset_D2TB_basebulk] = -1;
							}
					}
					for (unsigned int f = 0; f < ft->numfields_D2TB-ft->numfields_D2TB_basebulk; f++)
					{
					      int eq=eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_D2TB_interf];					      
							if (eq >= 0)
							{
							   eq=eqn_remap[eq];
							   if (eq==-666)
							   {
									std::ostringstream oss;
									oss << this;
									throw_runtime_error("MISSING EXTERNAL D2TB ADD INTERFACE DEPENDENCY ON ELEM PTR: " + oss.str());
							   }
								shape_info->hanginfo_Discont[l].masters[0].local_eqn[f + ft->buffer_offset_D2TB_interf] = eq ;
							}
							else
							{
								shape_info->hanginfo_Discont[l].masters[0].local_eqn[f + ft->buffer_offset_D2TB_interf] = -1;
							}
					}					
				}
			}

			if (codeinst->get_func_table()->numfields_D2 && (required.dx_psi_C2 || required.psi_C2 || required.dX_psi_C2))
			{
				for (unsigned int l = 0; l < eleminfo.nnode_C2; l++)
				{
					if (!shape_info->hanginfo_Discont[l].nummaster)
					{
						// NON HANGING -> HANGING WITH WEIGHT 1 for external element data
						shape_info->hanginfo_Discont[l].nummaster = 1;
						shape_info->hanginfo_Discont[l].masters[0].weight = 1.0;
					}
					for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_D2_basebulk; f++)
					{
					      int eq=eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_D2_basebulk];
					      
							if (eq >= 0)
							{
							   eq=eqn_remap[eq];
							   if (eq==-666)
							   {
									std::ostringstream oss;
									oss << this;
									throw_runtime_error("MISSING EXTERNAL D2 DEPENDENCY ON ELEM PTR: " + oss.str());
							   }
								shape_info->hanginfo_Discont[l].masters[0].local_eqn[f + ft->buffer_offset_D2_basebulk] = eq ;
							}
							else
							{
								shape_info->hanginfo_Discont[l].masters[0].local_eqn[f + ft->buffer_offset_D2_basebulk] = -1;
							}
					}
					for (unsigned int f = 0; f < ft->numfields_D2-ft->numfields_D2_basebulk; f++)
					{
					      int eq=eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_D2_interf];					      
							if (eq >= 0)
							{
							   eq=eqn_remap[eq];
							   if (eq==-666)
							   {
									std::ostringstream oss;
									oss << this;
									throw_runtime_error("MISSING EXTERNAL D2 ADD INTERFACE DEPENDENCY ON ELEM PTR: " + oss.str());
							   }
								shape_info->hanginfo_Discont[l].masters[0].local_eqn[f + ft->buffer_offset_D2_interf] = eq ;
							}
							else
							{
								shape_info->hanginfo_Discont[l].masters[0].local_eqn[f + ft->buffer_offset_D2_interf] = -1;
							}
					}							
				}
			}
			
			if (codeinst->get_func_table()->numfields_D1TB && (required.dx_psi_C1TB || required.psi_C1TB || required.dX_psi_C1TB))
			{
				for (unsigned int l = 0; l < eleminfo.nnode_C1TB; l++)
				{
					if (!shape_info->hanginfo_Discont[l].nummaster)
					{
						// NON HANGING -> HANGING WITH WEIGHT 1 for external element data
						shape_info->hanginfo_Discont[l].nummaster = 1;
						shape_info->hanginfo_Discont[l].masters[0].weight = 1.0;
					}
					for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_D1TB_basebulk; f++)
					{
					      int eq=eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_D1TB_basebulk];
					      
							if (eq >= 0)
							{
							   eq=eqn_remap[eq];
							   if (eq==-666)
							   {
									std::ostringstream oss;
									oss << this;
									throw_runtime_error("MISSING EXTERNAL D1TB DEPENDENCY ON ELEM PTR: " + oss.str());
							   }
								shape_info->hanginfo_Discont[l].masters[0].local_eqn[f + ft->buffer_offset_D1TB_basebulk] = eq ;
							}
							else
							{
								shape_info->hanginfo_Discont[l].masters[0].local_eqn[f + ft->buffer_offset_D1TB_basebulk] = -1;
							}
					}					
					for (unsigned int f = 0; f < ft->numfields_D1TB-ft->numfields_D1TB_basebulk; f++)
					{
					      int eq=eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_D1TB_interf];					      
							if (eq >= 0)
							{
							   eq=eqn_remap[eq];
							   if (eq==-666)
							   {
									std::ostringstream oss;
									oss << this;
									throw_runtime_error("MISSING EXTERNAL D1TB ADD INTERFACE DEPENDENCY ON ELEM PTR: " + oss.str());
							   }
								shape_info->hanginfo_Discont[l].masters[0].local_eqn[f + ft->buffer_offset_D1TB_interf] = eq ;
							}
							else
							{
								shape_info->hanginfo_Discont[l].masters[0].local_eqn[f + ft->buffer_offset_D1TB_interf] = -1;
							}
					}						
				}
			}

			if (codeinst->get_func_table()->numfields_D1 && (required.dx_psi_C1 || required.psi_C1 || required.dX_psi_C1))
			{
				for (unsigned int l = 0; l < eleminfo.nnode_C1; l++)
				{
					if (!shape_info->hanginfo_Discont[l].nummaster)
					{
						// NON HANGING -> HANGING WITH WEIGHT 1 for external element data
						shape_info->hanginfo_Discont[l].nummaster = 1;
						shape_info->hanginfo_Discont[l].masters[0].weight = 1.0;
					}
					for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_D1_basebulk; f++)
					{
					      int eq=eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_D1_basebulk];
					      
							if (eq >= 0)
							{
							   eq=eqn_remap[eq];
							   if (eq==-666)
							   {
									std::ostringstream oss;
									oss << this;
									throw_runtime_error("MISSING EXTERNAL D1 DEPENDENCY ON ELEM PTR: " + oss.str());
							   }
								shape_info->hanginfo_Discont[l].masters[0].local_eqn[f + ft->buffer_offset_D1_basebulk] = eq ;
							}
							else
							{
								shape_info->hanginfo_Discont[l].masters[0].local_eqn[f + ft->buffer_offset_D1_basebulk] = -1;
							}
					}					
					for (unsigned int f = 0; f < ft->numfields_D1-ft->numfields_D1_basebulk; f++)
					{
					      int eq=eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_D1_interf];					      
							if (eq >= 0)
							{
							   eq=eqn_remap[eq];
							   if (eq==-666)
							   {
									std::ostringstream oss;
									oss << this;
									throw_runtime_error("MISSING EXTERNAL D1 ADD INTERFACE DEPENDENCY ON ELEM PTR: " + oss.str());
							   }
								shape_info->hanginfo_Discont[l].masters[0].local_eqn[f + ft->buffer_offset_D1_interf] = eq ;
							}
							else
							{
								shape_info->hanginfo_Discont[l].masters[0].local_eqn[f + ft->buffer_offset_D1_interf] = -1;
							}
					}						
				}
			}

			if (codeinst->get_func_table()->numfields_DL && (required.dx_psi_DL || required.psi_DL || required.dX_psi_DL))
			{
				for (unsigned int l = 0; l < eleminfo.nnode_DL; l++)
				{
					if (!shape_info->hanginfo_Discont[l].nummaster)
					{
						// NON HANGING -> HANGING WITH WEIGHT 1 for external element data
						shape_info->hanginfo_Discont[l].nummaster = 1;
						shape_info->hanginfo_Discont[l].masters[0].weight = 1.0;
					}
					for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_DL; f++)
					{
					      int eq=eleminfo.nodal_local_eqn[l][f + ft->buffer_offset_DL];
					      
							if (eq >= 0)
							{
							   eq=eqn_remap[eq];
							   if (eq==-666)
							   {
									std::ostringstream oss;
									oss << this;
									throw_runtime_error("MISSING EXTERNAL DL DEPENDENCY ON ELEM PTR: " + oss.str());
							   }
								shape_info->hanginfo_Discont[l].masters[0].local_eqn[f + ft->buffer_offset_DL] = eq ;
							}
							else
							{
								shape_info->hanginfo_Discont[l].masters[0].local_eqn[f + ft->buffer_offset_DL] = -1;
							}
					}
				}
			}


			if (codeinst->get_func_table()->numfields_D0 && (required.psi_D0))
			{
				if (!shape_info->hanginfo_Discont[0].nummaster)
					{
						// NON HANGING -> HANGING WITH WEIGHT 1 for external element data

						shape_info->hanginfo_Discont[0].nummaster = 1;
						shape_info->hanginfo_Discont[0].masters[0].weight = 1.0;
					}
					for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_D0; f++)
					{
					      int eq=eleminfo.nodal_local_eqn[0][f + ft->buffer_offset_D0];
					      
							if (eq >= 0)
							{
							   eq=eqn_remap[eq];
							   if (eq==-666)
							   {
									std::ostringstream oss;
									oss << this;
									throw_runtime_error("MISSING EXTERNAL D0 DEPENDENCY ON ELEM PTR: " + oss.str());
							   }
								shape_info->hanginfo_Discont[0].masters[0].local_eqn[f + ft->buffer_offset_D0] = eq;
							}
							else
							{
								shape_info->hanginfo_Discont[0].masters[0].local_eqn[f + ft->buffer_offset_D0] = -1;
							}
					}
			}
			return true;
		}
		return false;
	}

	oomph::Node *BulkElementBase::boundary_node_pt(const int &face_index, const unsigned int index)
	{
		throw_runtime_error("Implement");
	}

	void BulkElementBase::ensure_external_data()
	{
		//     std::cout << "ENSUREING EXTERNAL DATGA of elemdim " << this->dim() << "  WITH " << codeinst->linked_external_data.size() << " ED  @ " << codeinst->get_code()->get_file_name()<<  std::endl;
		this->flush_external_data();
		for (auto &e : codeinst->linked_external_data.get_required_external_data())
		{
			//   std::cout << "ADDING EXTERNAL DATGA " << std::endl;
			this->add_external_data(e);
		}
	}

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
						/* double t0prime=(k==0 ? 1 : 0)*dpsi(l,0);
						 double t1prime=(k==1 ? 1 : 0)*dpsi(l,0);
						 dnormal_dcoord[i][l][k]=-denom*(i==0 ? -1 : 1)*dxds[i]*(dxds[1]*t0prime-dxds[0]*t1prime);
						 */
						dnormal_dcoord[i][l][k] = dpsi(l, 0) * denom * (k == 1 ? -1 : 1) * dxds[i] * dxds[1 - k];
					}
				}
			}
			if (d2normal_dcoord2)
			{
			   //double cross=dxds[0]*dxds[1];
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
								 //d2normal_dcoord2[i][l][j][lp][jp]=(i==1 ? -1: 1)*dpsi(l,0)*dpsi(lp,0)*denom*(dxds[1-jp]-3*cross*dxds[jp]/denom_sqr);
								 d2normal_dcoord2[i][l][j][lp][jp]=(i==1 ? -1 : 1)*(dpsi(l,0)*dpsi(lp,0))*denom*(( (j==jp && j!=i) ? 3*dxds[1-i] : dxds[1-(j==i ? jp : j)])-3*dxds[1-i]*dxds[j]*dxds[jp]/denom_sqr);
								}												
							}
						}
					}
				}
				/*
				//TODO: REMOVE
				double ***dnpert;
				dnpert=(double ***)calloc(nodal_dim,sizeof(double**));
				for (unsigned int i=0;i < nodal_dim; i++) 
				{
				  dnpert[i]=(double **)calloc(n_node,sizeof(double*));
   			  for (unsigned int l=0;l < n_node; l++) 
   			  {
   			   dnpert[i][l]=(double *)calloc(n_node,sizeof(double));
   			  }
				}
				for (unsigned int lp=0;lp<n_node;lp++)
				{
				  for (unsigned jp=0;jp<nodal_dim;jp++)
				  {
  					 pyoomph::Node * nod=dynamic_cast<pyoomph::Node *>(this->node_pt(lp));
					 double old=nod->variable_position_pt()->value(jp);
					 double eps=1e-8;
					 nod->variable_position_pt()->set_value(jp,old+eps);
				    this->get_dnormal_dcoords_at_s(s,dnpert,NULL);
				    std::cout << "FOR LP JP " << lp << "  " << jp << std::endl;
				    for (unsigned int i=0;i<nodal_dim;i++)
				    {
				     for (unsigned int l=0;l<n_node;l++)
				     {
				      for (unsigned int j=0;j<nodal_dim;j++)
				      {
				        double ana=d2normal_dcoord2[i][l][j][lp][jp];
				        double fd=(dnpert[i][l][j]-dnormal_dcoord[i][l][j])/eps;
				        std::cout << "   " << i << "  " << l << "  " << j << " :  "  <<  ana << "  " << fd << "   DIFF " << ana-fd << std::endl; 
				      }
				     }
				    }
				    
					 nod->variable_position_pt()->set_value(jp,old);				   
				  }
				}
				*/ 

//			 std::cerr << "DOES NOT WORK: Hessian here" << std::endl << std::flush;
//			 throw_runtime_error("Hessian here");
			}
		}
		else if (nodal_dim==3 && eldim==2)
		{
/************************/


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








/**************************////



					  
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

    oomph::Data *BulkElementBase::get_D0_nodal_data(const unsigned &fieldindex)
    {
		auto * ft=this->get_code_instance()->get_func_table();
		
        return this->internal_data_pt(ft->internal_offset_D0+fieldindex);
    }

	oomph::Data *BulkElementBase::get_DL_nodal_data(const unsigned &fieldindex)
    {
		auto * ft=this->get_code_instance()->get_func_table();
//		std::cout << "GETTING INTERNAL DATA " << ft->internal_offset_DL+fieldindex <<  " OF " << this->ninternal_data() << std::endl << std::flush;
        return this->internal_data_pt(ft->internal_offset_DL+fieldindex);
    }

    oomph::Data *BulkElementBase::get_D1_nodal_data(const unsigned &fieldindex)
    {
		auto * ft=this->get_code_instance()->get_func_table();
		
        return this->internal_data_pt(ft->internal_offset_D1_new+fieldindex);
    }

	oomph::Data *BulkElementBase::get_D2_nodal_data(const unsigned &fieldindex)
    {
		auto * ft=this->get_code_instance()->get_func_table();
        return this->internal_data_pt(ft->internal_offset_D2_new+fieldindex);
    }

	oomph::Data *BulkElementBase::get_D2TB_nodal_data(const unsigned &fieldindex)
    {
		auto * ft=this->get_code_instance()->get_func_table();
        return this->internal_data_pt(ft->internal_offset_D2TB_new+fieldindex);
    }
    
	oomph::Data *BulkElementBase::get_D1TB_nodal_data(const unsigned &fieldindex)
    {
		auto * ft=this->get_code_instance()->get_func_table();
        return this->internal_data_pt(ft->internal_offset_D1TB_new+fieldindex);
    }    

	unsigned BulkElementBase::get_C2TB_buffer_index(const unsigned &fieldindex)
    {
		auto * ft=this->get_code_instance()->get_func_table();
        return ft->buffer_offset_C2TB_basebulk+ fieldindex;
    }
    
	unsigned BulkElementBase::get_C2_buffer_index(const unsigned &fieldindex)
    {
		auto * ft=this->get_code_instance()->get_func_table();
        return ft->buffer_offset_C2_basebulk+ fieldindex;
    }

	unsigned BulkElementBase::get_C1TB_buffer_index(const unsigned &fieldindex)
    {
		auto * ft=this->get_code_instance()->get_func_table();
        return ft->buffer_offset_C1TB_basebulk+ fieldindex;
    }    

	unsigned BulkElementBase::get_C1_buffer_index(const unsigned &fieldindex)
    {
		auto * ft=this->get_code_instance()->get_func_table();
        return ft->buffer_offset_C1_basebulk+ fieldindex;
    }    

	unsigned BulkElementBase::get_D2TB_buffer_index(const unsigned &fieldindex)
    {
		auto * ft=this->get_code_instance()->get_func_table();
        return ft->buffer_offset_D2TB_basebulk+ fieldindex;
    }

	unsigned BulkElementBase::get_D2_buffer_index(const unsigned &fieldindex)
    {
		auto * ft=this->get_code_instance()->get_func_table();
        return ft->buffer_offset_D2_basebulk+ fieldindex;
    }

    unsigned BulkElementBase::get_D1TB_buffer_index(const unsigned &fieldindex)
    {
		auto * ft=this->get_code_instance()->get_func_table();
        return ft->buffer_offset_D1TB_basebulk+ fieldindex;
    }
    
	unsigned BulkElementBase::get_D1_buffer_index(const unsigned &fieldindex)
    {
		auto * ft=this->get_code_instance()->get_func_table();
        return ft->buffer_offset_D1_basebulk+fieldindex;
    }

	unsigned BulkElementBase::get_DL_buffer_index(const unsigned &fieldindex)
    {
		auto * ft=this->get_code_instance()->get_func_table();
        return ft->buffer_offset_DL+ fieldindex;
    }

	unsigned BulkElementBase::get_D0_buffer_index(const unsigned &fieldindex)
    {
		auto * ft=this->get_code_instance()->get_func_table();
        return ft->buffer_offset_D0+ fieldindex;
    }

	int BulkElementBase::get_C2TB_local_equation(const unsigned &fieldindex,const unsigned & nodeindex,const bool & by_elemental_node_index)
	{
	  auto * ft=this->get_code_instance()->get_func_table();
	  if (by_elemental_node_index) return this->nodal_local_eqn(nodeindex,ft->nodal_offset_C2TB_basebulk+fieldindex);
	  else return this->nodal_local_eqn(this->get_node_index_C2TB_to_element(nodeindex),ft->nodal_offset_C2TB_basebulk+fieldindex);
	}

    int BulkElementBase::get_C2_local_equation(const unsigned &fieldindex,const unsigned & nodeindex,const bool & by_elemental_node_index)
	{
	  auto * ft=this->get_code_instance()->get_func_table();
	  if (by_elemental_node_index) return this->nodal_local_eqn(nodeindex,ft->nodal_offset_C2_basebulk+fieldindex);
	  else return this->nodal_local_eqn(this->get_node_index_C2_to_element(nodeindex),ft->nodal_offset_C2_basebulk+fieldindex);		
	}

    int BulkElementBase::get_C1TB_local_equation(const unsigned &fieldindex,const unsigned & nodeindex,const bool & by_elemental_node_index)    
	{
	  auto * ft=this->get_code_instance()->get_func_table();
	  if (by_elemental_node_index) return this->nodal_local_eqn(nodeindex,ft->nodal_offset_C1TB_basebulk+fieldindex);
	  else return this->nodal_local_eqn(this->get_node_index_C1TB_to_element(nodeindex),ft->nodal_offset_C1TB_basebulk+fieldindex);		
	}
	
    int BulkElementBase::get_C1_local_equation(const unsigned &fieldindex,const unsigned & nodeindex,const bool & by_elemental_node_index)
	{
	  auto * ft=this->get_code_instance()->get_func_table();
	  if (by_elemental_node_index) return this->nodal_local_eqn(nodeindex,ft->nodal_offset_C1_basebulk+fieldindex);
	  else return this->nodal_local_eqn(this->get_node_index_C1_to_element(nodeindex),ft->nodal_offset_C1_basebulk+fieldindex);		
	}

    int BulkElementBase::get_D2TB_local_equation(const unsigned &fieldindex,const unsigned & nodeindex)
	{
	  auto * ft=this->get_code_instance()->get_func_table();
	  return this->internal_local_eqn(ft->internal_offset_D2TB_new+fieldindex,nodeindex);
	}
    int BulkElementBase::get_D1TB_local_equation(const unsigned &fieldindex,const unsigned & nodeindex)
	{
	  auto * ft=this->get_code_instance()->get_func_table();
	  return this->internal_local_eqn(ft->internal_offset_D1TB_new+fieldindex,nodeindex);
	}	
	
    int BulkElementBase::get_D2_local_equation(const unsigned &fieldindex,const unsigned & nodeindex)
	{
	  auto * ft=this->get_code_instance()->get_func_table();
	  return this->internal_local_eqn(ft->internal_offset_D2_new+fieldindex,nodeindex);		
	}
    int BulkElementBase::get_D1_local_equation(const unsigned &fieldindex,const unsigned & nodeindex)
	{
	  auto * ft=this->get_code_instance()->get_func_table();
	  return this->internal_local_eqn(ft->internal_offset_D1_new+fieldindex,nodeindex);				
	}
    int BulkElementBase::get_DL_local_equation(const unsigned &fieldindex,const unsigned & nodeindex)
	{
	  auto * ft=this->get_code_instance()->get_func_table();
	  return this->internal_local_eqn(ft->internal_offset_DL+fieldindex,nodeindex);						
	}
    int BulkElementBase::get_D0_local_equation(const unsigned &fieldindex)
	{
	  auto * ft=this->get_code_instance()->get_func_table();
	  return this->internal_local_eqn(ft->internal_offset_D0+fieldindex,0);								
	}

    // TODO: Use the one defined in doi:10.1016/j.cma.2006.11.013
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
			/* for (unsigned int ni=0;ni<this->nnode();ni++)
			 {
			  this->local_coordinate_of_node(ni,s);
			  oomph::Vector<double> s_macro(el_dim);
			  for(unsigned i=0;i<el_dim;i++)
			  {
				 s_macro[i]=telem->s_macro_ll(i)+0.5*(s[i]+1.0)*(telem->s_macro_ur(i)-telem->s_macro_ll(i));
			  }

			  macro_elem_pt()->macro_map(s_macro,r); //TODO: Time loop
			  for (unsigned int id=0;id<r.size();id++) this->node_pt(ni)->x(id)=r[id];
			 }*/
			return;
		}
	}

	BulkElementBase *BulkElementBase::create_from_template(MeshTemplate *mt, MeshTemplateElement *el)
	{
		BulkElementBase *res = NULL;
		std::vector<int> nodemap;
		std::string domspace=std::string(BulkElementBase::__CurrentCodeInstance->get_func_table()->dominant_space);
		if (el->get_geometric_type_index() == 1)
			res = new BulkElementLine1dC1();
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
			res = new BulkElementTetra3dC1();
		else if (el->get_geometric_type_index() == 6)
			res = new BulkElementQuad2dC1();
		else if (el->get_geometric_type_index() == 8)
		{
			//   std::cout << "DOMSPACE " << BulkElementBase::__CurrentCodeInstance->get_func_table()->dominant_space << std::endl;
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
		else if (el->get_geometric_type_index() == 10)
		{
			if (domspace == "C1")
			{
				nodemap = {0, 1, 2, 3};
				res = new BulkElementTetra3dC1();
			}
			else if (domspace=="C1TB")
			{
			 throw_runtime_error("Implement BulkElementTetra3dC1TB");
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
			res = new BulkElementBrick3dC1();
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
		else
			throw_runtime_error("Undefined element type: " + std::to_string(el->get_geometric_type_index()));

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
//			   std::cout << "I " << i << " " << el->get_node_indices().size() << " SI  " << mt->get_nodes().size() << " NM " << nodemap.size()  << " NMU " << nodemap[i] << std::endl;
				res->node_pt(i) = mt->get_nodes()[el->get_node_indices()[nodemap[i]]]->oomph_node;
			}
		}

		for (unsigned int i = 0; i < res->ninternal_data(); i++)
			res->internal_data_pt(i)->set_time_stepper(res->node_pt(0)->time_stepper_pt(), false);

		/*
		 {
		 const unsigned n_node = res->nnode();
		  const unsigned n_position_type = res->nnodal_position_type();
		  //Find the dimension of the node and element
		  const unsigned n_dim_node = res->nodal_dimension();
		  const unsigned n_dim_element = res->dim();
		  std::cout << "INFO " << n_node << "  " << n_position_type << "  " << n_dim_node << "  " << n_dim_element << std::endl;
		   for(unsigned i=0;i<n_dim_element;i++)
		   {
			for(unsigned j=0;j<n_dim_node;j++)
			 {
			  //Initialise the j-th component of the i-th base vector to zero
			  for(unsigned l=0;l<n_node;l++)
			   {
				for(unsigned k=0;k<n_position_type;k++)
				 {
				   std::cout << "   GETTING " << l << "  " << k << "  " << j << std::endl << std::flush;
				   std::cout << "     " << res->nodal_position_gen(l,k,j) << std::endl << std::flush;
				 }
			   }
			 }
		   }

		 }
		  */
		res->initial_cartesian_nondim_size = res->size();
		res->initial_quality_factor = res->get_quality_factor();

		if (BulkElementBase::__CurrentCodeInstance->get_func_table()->integration_order)
		{
			res->set_integration_order(BulkElementBase::__CurrentCodeInstance->get_func_table()->integration_order);
		}
		return res;
	}

	void BulkElementBase::unpin_dummy_values() // C1 fields on C2 elements have dummy values on only C2 nodes, which needs to be pinned
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();

		for (unsigned int l = 0; l < nnode(); l++)
		{
			for (unsigned int i = 0; i < this->nodal_dimension(); i++)
			{
				dynamic_cast<Node *>(node_pt(l))->unpin_position(i);
			}
			this->node_pt(l)->unconstrain_positions();
		}

		if ((!functable->numfields_C1_basebulk) && (!functable->numfields_C2_basebulk) && (!functable->numfields_C2TB_basebulk) && (!functable->numfields_C1TB_basebulk))
			return; // Nothing to do in this case ///TODO: Check in case of ElasticPVD of C1 only
		for (unsigned int l = 0; l < nnode(); l++)
		{

			for (unsigned int i = 0; i < node_pt(l)->nvalue(); i++)
			{
				// if (!node_pt(l)->is_hanging())
				//{
				node_pt(l)->unpin(i); // After that, the BCs are applied to repin what is necessary
									  //}
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

	void BulkElementBase::pin_dummy_values()
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();

		if (this->eleminfo.nnode_C2 && codeinst->get_func_table()->bulk_position_space_to_C1)
		{
			this->constrain_bulk_position_space_to_C1();
		}

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

	//	if ((!functable->numfields_C2TB_basebulk) && (!functable->numfields_C1_basebulk) && (!functable->numfields_C2_basebulk))
	//		return; // Nothing to do in this case ///TODO: Check in case of ElasticPVD of C1 only
		for (unsigned n = 0; n < nnode(); n++)
		{
		   if (!this->is_node_index_part_of_C2TB(n))
		   {
				for (unsigned int i = 0;i<functable->numfields_C2TB_basebulk; i++)
				{
				  this->node_pt(n)->pin(i+functable->nodal_offset_C2TB_basebulk);
				}
			}
		   if (!this->is_node_index_part_of_C2(n))
		   {
				for (unsigned int i = 0;i<functable->numfields_C2_basebulk; i++)
				{
				  this->node_pt(n)->pin(i+functable->nodal_offset_C2_basebulk);
				}
			}
		   if (!this->is_node_index_part_of_C1TB(n))
		   {
				for (unsigned int i = 0;i<functable->numfields_C1TB_basebulk; i++)
				{
				  this->node_pt(n)->pin(i+functable->nodal_offset_C1TB_basebulk);
				}
			}
		   if (!this->is_node_index_part_of_C1(n))
		   {
				for (unsigned int i = 0;i<functable->numfields_C1_basebulk; i++)
				{
				  this->node_pt(n)->pin(i+functable->nodal_offset_C1_basebulk);
				}
			}									
/*				
			if (!this->is_node_index_part_of_C1(n))
			{
				for (unsigned int i = functable->numfields_C2TB_basebulk + functable->numfields_C2_basebulk; i < functable->numfields_C2TB_basebulk + functable->numfields_C2_basebulk + functable->numfields_C1_basebulk ; i++)
				{
					this->node_pt(n)->pin(i);
				}
				if (!this->is_node_index_part_of_C2(n))
				{
					for (unsigned int i = functable->numfields_C2TB_basebulk; i < functable->numfields_C2TB_basebulk + functable->numfields_C2_basebulk; i++)
					{
						this->node_pt(n)->pin(i);
					}
				}
			}
*/			
			if (functable->numfields_C2TB_basebulk)
			{
				int hanging_index = (codeinst->get_func_table()->bulk_position_space_to_C1 ? 0 : -1);
				if (node_pt(n)->is_hanging(hanging_index))
				{
					for (unsigned int i = 0; i < functable->numfields_C2TB_basebulk; i++)
					{
						node_pt(n)->constrain(functable->nodal_offset_C2TB_basebulk+i);
					}
				}
			}
			if (functable->numfields_C2_basebulk)
			{
				int hanging_index = (codeinst->get_func_table()->bulk_position_space_to_C1 ? 0 : -1);
				if (node_pt(n)->is_hanging(hanging_index))
				{
					for (unsigned int i = 0; i < functable->numfields_C2_basebulk; i++)
					{
						node_pt(n)->constrain(functable->nodal_offset_C2_basebulk + i);
					}
				}
			}
			if (functable->numfields_C1TB_basebulk)
			{
				if ((this->is_node_index_part_of_C1TB(n)) && node_pt(n)->is_hanging(functable->numfields_C2TB_basebulk + functable->numfields_C2_basebulk))
				{
					for (unsigned int i = 0; i < functable->numfields_C1TB_basebulk; i++)
					{
						node_pt(n)->constrain(functable->nodal_offset_C1TB_basebulk + functable->numfields_C2_basebulk + i);
					}
				}
			}
			
			if (functable->numfields_C1_basebulk)
			{
				if ((this->is_node_index_part_of_C1(n)) && node_pt(n)->is_hanging(functable->numfields_C2TB_basebulk + functable->numfields_C2_basebulk))
				{
					for (unsigned int i = 0; i < functable->numfields_C1_basebulk; i++)
					{
						node_pt(n)->constrain(functable->nodal_offset_C1_basebulk + i);
					}
				}
			}
		}
	}

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
		my_alloc_or_free(do_alloc, (*buff)->d_dx_shape_dcoord_C2TB, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);
		my_alloc_or_free(do_alloc, (*buff)->d_dx_shape_dcoord_C2, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);
		my_alloc_or_free(do_alloc, (*buff)->d_dx_shape_dcoord_C1TB, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);
		my_alloc_or_free(do_alloc, (*buff)->d_dx_shape_dcoord_C1, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);
		my_alloc_or_free(do_alloc, (*buff)->d_dx_shape_dcoord_DL, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);
		my_alloc_or_free(do_alloc, (*buff)->d_dshape_dx_tensor, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM, MAX_NODAL_DIM);

		my_alloc_or_free(do_alloc, (*buff)->shape_C2TB, MAX_NODES);
		my_alloc_or_free(do_alloc, (*buff)->nodal_shape_C2TB, MAX_NODES, MAX_NODES);
		my_alloc_or_free(do_alloc, (*buff)->dx_shape_C2TB, MAX_NODES, MAX_NODAL_DIM);
		my_alloc_or_free(do_alloc, (*buff)->dX_shape_C2TB, MAX_NODES, MAX_NODAL_DIM);
		my_alloc_or_free(do_alloc, (*buff)->dS_shape_C2TB, MAX_NODES, MAX_NODAL_DIM);

		my_alloc_or_free(do_alloc, (*buff)->shape_C2, MAX_NODES);
		my_alloc_or_free(do_alloc, (*buff)->nodal_shape_C2, MAX_NODES, MAX_NODES);
		my_alloc_or_free(do_alloc, (*buff)->dx_shape_C2, MAX_NODES, MAX_NODAL_DIM);
		my_alloc_or_free(do_alloc, (*buff)->dX_shape_C2, MAX_NODES, MAX_NODAL_DIM);
		my_alloc_or_free(do_alloc, (*buff)->dS_shape_C2, MAX_NODES, MAX_NODAL_DIM);


		my_alloc_or_free(do_alloc, (*buff)->shape_C1TB, MAX_NODES);
		my_alloc_or_free(do_alloc, (*buff)->nodal_shape_C1TB, MAX_NODES, MAX_NODES);
		my_alloc_or_free(do_alloc, (*buff)->dx_shape_C1TB, MAX_NODES, MAX_NODAL_DIM);
		my_alloc_or_free(do_alloc, (*buff)->dX_shape_C1TB, MAX_NODES, MAX_NODAL_DIM);
		my_alloc_or_free(do_alloc, (*buff)->dS_shape_C1TB, MAX_NODES, MAX_NODAL_DIM);

		my_alloc_or_free(do_alloc, (*buff)->shape_C1, MAX_NODES);
		my_alloc_or_free(do_alloc, (*buff)->nodal_shape_C1, MAX_NODES, MAX_NODES);
		my_alloc_or_free(do_alloc, (*buff)->dx_shape_C1, MAX_NODES, MAX_NODAL_DIM);
		my_alloc_or_free(do_alloc, (*buff)->dX_shape_C1, MAX_NODES, MAX_NODAL_DIM);
		my_alloc_or_free(do_alloc, (*buff)->dS_shape_C1, MAX_NODES, MAX_NODAL_DIM);

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
			my_alloc_or_free(do_alloc, (*buff)->d2_dx2_shape_dcoord_C2TB, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);
			my_alloc_or_free(do_alloc, (*buff)->d2_dx2_shape_dcoord_C2, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);
			my_alloc_or_free(do_alloc, (*buff)->d2_dx2_shape_dcoord_C1TB, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);						
			my_alloc_or_free(do_alloc, (*buff)->d2_dx2_shape_dcoord_C1, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);
			my_alloc_or_free(do_alloc, (*buff)->d2_dx2_shape_dcoord_DL, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);

			my_alloc_or_free(do_alloc, (*buff)->d2_normal_d2coord, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);
		}
		else
		{
			(*buff)->int_pt_weights_d2_coords = NULL;
			(*buff)->elemsize_d2_coords=NULL;
			(*buff)->elemsize_Cart_d2_coords=NULL;
			(*buff)->d2_dx2_shape_dcoord_C2TB = NULL;
			(*buff)->d2_dx2_shape_dcoord_C2 = NULL;
			(*buff)->d2_dx2_shape_dcoord_C1TB = NULL;			
			(*buff)->d2_dx2_shape_dcoord_C1 = NULL;
			(*buff)->d2_dx2_shape_dcoord_DL = NULL;
			(*buff)->d2_normal_d2coord = NULL;
		}

#ifndef FIXED_SIZE_SHAPE_BUFFER
		if (do_alloc)
		{
			my_alloc((*buff)->hanginfo_C2TB, MAX_NODES);
			my_alloc((*buff)->hanginfo_C2, MAX_NODES);
			my_alloc((*buff)->hanginfo_C1TB, MAX_NODES);			
			my_alloc((*buff)->hanginfo_C1, MAX_NODES);
			my_alloc((*buff)->hanginfo_Pos, MAX_NODES);				
			for (unsigned int l = 0; l < MAX_NODES; l++)
			{
				my_alloc((*buff)->hanginfo_C2TB[l].masters, MAX_HANG);
				my_alloc((*buff)->hanginfo_C2[l].masters, MAX_HANG);
				my_alloc((*buff)->hanginfo_C1TB[l].masters, MAX_HANG);				
				my_alloc((*buff)->hanginfo_C1[l].masters, MAX_HANG);
				my_alloc((*buff)->hanginfo_Pos[l].masters, MAX_HANG);
				for (unsigned int f = 0; f < MAX_HANG; f++)
				{
					my_alloc((*buff)->hanginfo_C2TB[l].masters[f].local_eqn, MAX_FIELDS);
					my_alloc((*buff)->hanginfo_C2[l].masters[f].local_eqn, MAX_FIELDS);
					my_alloc((*buff)->hanginfo_C1TB[l].masters[f].local_eqn, MAX_FIELDS);					
					my_alloc((*buff)->hanginfo_C1[l].masters[f].local_eqn, MAX_FIELDS);
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
					my_free((*buff)->hanginfo_C2TB[l].masters[f].local_eqn, MAX_FIELDS);
					my_free((*buff)->hanginfo_C2[l].masters[f].local_eqn, MAX_FIELDS);
					my_free((*buff)->hanginfo_C1TB[l].masters[f].local_eqn, MAX_FIELDS);					
					my_free((*buff)->hanginfo_C1[l].masters[f].local_eqn, MAX_FIELDS);
					my_free((*buff)->hanginfo_Pos[l].masters[f].local_eqn, MAX_FIELDS);
				}
				
				my_free((*buff)->hanginfo_C2TB[l].masters, MAX_HANG);
				my_free((*buff)->hanginfo_C2[l].masters, MAX_HANG);
				my_free((*buff)->hanginfo_C1TB[l].masters, MAX_HANG);				
				my_free((*buff)->hanginfo_C1[l].masters, MAX_HANG);
				my_free((*buff)->hanginfo_Pos[l].masters, MAX_HANG);
				

			}
			my_free((*buff)->hanginfo_C2TB, MAX_NODES);
			my_free((*buff)->hanginfo_C2, MAX_NODES);
			my_free((*buff)->hanginfo_C1TB, MAX_NODES);			
			my_free((*buff)->hanginfo_C1, MAX_NODES);
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

	void alloc_dealloc_all_shape_buffers(bool do_alloc, JITShapeInfo_t **buff, bool with_analytical_hessian_moving_mesh)
	{
		if (do_alloc)
		{
			__shape_buffer_mem_usage = 0;
			alloc_dealloc_single_shape_buffer(true, &Default_shape_info_buffer, with_analytical_hessian_moving_mesh);
			alloc_dealloc_single_shape_buffer(true, &(Default_shape_info_buffer->bulk_shapeinfo), with_analytical_hessian_moving_mesh);
			alloc_dealloc_single_shape_buffer(true, &(Default_shape_info_buffer->opposite_shapeinfo), with_analytical_hessian_moving_mesh);
			alloc_dealloc_single_shape_buffer(true, &(Default_shape_info_buffer->opposite_shapeinfo->bulk_shapeinfo), with_analytical_hessian_moving_mesh);
			alloc_dealloc_single_shape_buffer(true, &(Default_shape_info_buffer->bulk_shapeinfo->bulk_shapeinfo), with_analytical_hessian_moving_mesh);
		//	std::cout << "Allocated " << __shape_buffer_mem_usage / (1024.0 * 1024.0) << " MB for the shape buffer" << std::endl;
		}
		else
		{
			alloc_dealloc_single_shape_buffer(false, &Default_shape_info_buffer, with_analytical_hessian_moving_mesh);
		}
	}

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
		}
		else if (require_moving_hessian_buffer && Default_shape_info_buffer->int_pt_weights_d2_coords == NULL)
		{
			alloc_dealloc_all_shape_buffers(false, &Default_shape_info_buffer, require_moving_hessian_buffer);
			alloc_dealloc_all_shape_buffers(true, &Default_shape_info_buffer, require_moving_hessian_buffer);
		}
		shape_info = Default_shape_info_buffer;

		this->set_nlagrangian_and_ndim(this->codeinst->get_func_table()->lagr_dim, this->codeinst->get_func_table()->nodal_dim);
		this->ensure_external_data();
	}

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
				for (unsigned int j = eleminfo.nodal_dim + codeinst->get_func_table()->lagr_dim+this->dim() ; j < this->codeinst->get_func_table()->numfields_Pos; j++)
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
		if (eleminfo.nnode_C2TB)
			numfields += functable->numfields_C2TB+functable->numfields_D2TB;
		if (eleminfo.nnode_C2)
			numfields += functable->numfields_C2+functable->numfields_D2;
		if (eleminfo.nnode_C1TB)
			numfields += functable->numfields_C1TB+functable->numfields_D1TB;			
		if (eleminfo.nnode_C1)
			numfields += functable->numfields_C1+functable->numfields_D1;
		if (eleminfo.nnode_DL)
			numfields += functable->numfields_DL;
		numfields += functable->numfields_D0 + functable->numfields_ED0;
		for (unsigned int i = 0; i < eleminfo.nnode; i++)
		{
			oomph::Vector<double> snode(this->dim(),0.0);
			if (this->dim()>0)
			{
				this->local_coordinate_of_node(i,snode);
			}
						
			//FiniteElement::interpolated_zeta(snode, zeta);
			
			//eleminfo.nodal_coords[i] = (double **)calloc(eleminfo.nodal_dim + functable->lagr_dim +this->dim(), sizeof(double *));
			eleminfo.nodal_coords[i] = (double **)calloc(functable->numfields_Pos, sizeof(double *));
			
			for (unsigned int j = 0; j < eleminfo.nodal_dim; j++)
				eleminfo.nodal_coords[i][j] = dynamic_cast<Node *>(node_pt(i))->variable_position_pt()->value_pt(j);
			for (unsigned int j = 0; j < functable->lagr_dim; j++)
				eleminfo.nodal_coords[i][eleminfo.nodal_dim + j] = &(dynamic_cast<Node *>(node_pt(i))->xi(j));
			for (unsigned int j = 0; j < this->dim(); j++)
				eleminfo.nodal_coords[i][eleminfo.nodal_dim + functable->lagr_dim +j] = new double(snode[j]); // Local coordinate buffer
				
			if (dynamic_cast<oomph::FaceElement*>(this))
			{
				unsigned zeta_offset=eleminfo.nodal_dim + functable->lagr_dim+this->dim();
				for (unsigned int j=zeta_offset;j<functable->numfields_Pos;j++)
				{
					double zeta=0.0;
					
					if ( dynamic_cast<oomph::BoundaryNodeBase*>(this->node_pt(i))  && this->node_pt(i)->boundary_coordinates_have_been_set_up()) 
					{
						//std::cout << "IN BC " << this->get_code_instance()->get_code()->get_file_name() << " i=" << i << " j=" << j << " zeta_offset=" << zeta_offset << " BULK INDEX " << dynamic_cast<oomph::FaceElement*>(this)->boundary_number_in_bulk_mesh() << std::endl;
						oomph::Mesh * themesh=this->get_code_instance()->get_bulk_mesh();
						while (dynamic_cast<pyoomph::InterfaceMesh*>(themesh))
						{
							themesh = dynamic_cast<pyoomph::InterfaceMesh*>(themesh)->get_bulk_mesh();
						}
						//std::cout << "BULKMESH NELEM " << themesh->nelement() << std::endl;
						// This means that we have some boundary coordinates. But we do not really know whether we indeed have coordinates for this specific boundary
						if (dynamic_cast<pyoomph::Mesh*>(themesh)->is_boundary_coordinate_defined(dynamic_cast<oomph::FaceElement*>(this)->boundary_number_in_bulk_mesh()))
						{
							
							zeta= this->zeta_nodal(i,0,j-zeta_offset);
							//std::cout << "  setting  zeta=" << zeta << std::endl;
						}
					}
					eleminfo.nodal_coords[i][j] =   new double(zeta); // zeta coordinate buffer
				}
			}

			/*			unsigned numfields=0;
			//			numfields+=functable->numfields_Lagr; //Lagrangian everywhere
						if (i<eleminfo.nnode_C2) numfields+=functable->numfields_C2;
						if (i<eleminfo.nnode_C1) numfields+=functable->numfields_C1;
						if (i<eleminfo.nnode_DL) numfields+=functable->numfields_DL;
						if (i<1) numfields+=functable->numfields_D0+functable->numfields_ED0;
						*/
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
		
		unsigned local_field_offset = 0;
		unsigned global_offs = 0;

     if (functable->numfields_C2TB_basebulk)
     {
		for (unsigned int i = 0; i < eleminfo.nnode_C2TB; i++)
		{
			unsigned i_el = this->get_node_index_C2TB_to_element(i);
			for (unsigned int j = 0; j < functable->numfields_C2TB_basebulk; j++)
			{
				unsigned node_index = j + local_field_offset;
				eleminfo.nodal_data[i][node_index] = node_pt(i_el)->value_pt(node_index - global_offs); // Warning: value_pt does not work for hanging nodes! Will be changed if necessary
				if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] = this->nodal_local_eqn(i_el, node_index - global_offs);
			}
		}
	  }
		local_field_offset += functable->numfields_C2TB_basebulk;

		for (unsigned int i = 0; i < eleminfo.nnode_C2; i++)
		{
			unsigned i_el = this->get_node_index_C2_to_element(i);
			for (unsigned int j = 0; j < functable->numfields_C2_basebulk; j++)
			{
				unsigned node_index = j + local_field_offset;
				eleminfo.nodal_data[i][node_index] = node_pt(i_el)->value_pt(node_index - global_offs); // Warning: value_pt does not work for hanging nodes! Will be changed if necessary
				if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] = this->nodal_local_eqn(i_el, node_index - global_offs);
			}
		}

		local_field_offset += functable->numfields_C2_basebulk;

		if (functable->numfields_C1TB_basebulk)
		{

			for (unsigned int i = 0; i < eleminfo.nnode_C1TB; i++)
			{
				unsigned i_el = this->get_node_index_C1TB_to_element(i);
				for (unsigned int j = 0; j < functable->numfields_C1TB_basebulk; j++)
				{
					unsigned node_index = j + local_field_offset;
					eleminfo.nodal_data[i][node_index] = node_pt(i_el)->value_pt(node_index - global_offs); // Warning: value_pt does not work for hanging nodes! Will be changed if necessary
					if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] = this->nodal_local_eqn(i_el, node_index - global_offs);
				}
			}
		}

		local_field_offset += functable->numfields_C1TB_basebulk;
		
		for (unsigned int i = 0; i < eleminfo.nnode_C1; i++)
		{
			unsigned i_el = this->get_node_index_C1_to_element(i);
			for (unsigned int j = 0; j < functable->numfields_C1_basebulk; j++)
			{
				unsigned node_index = j + local_field_offset;
				eleminfo.nodal_data[i][node_index] = node_pt(i_el)->value_pt(node_index - global_offs); // Warning: value_pt does not work for hanging nodes! Will be changed if necessary
				if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] = this->nodal_local_eqn(i_el, node_index - global_offs);
			}
		}
		local_field_offset += functable->numfields_C1_basebulk;


      //DG base bulk fields
      	for (unsigned int i = 0; i < eleminfo.nnode_C2TB; i++)
		{
			for (unsigned int j = 0; j < functable->numfields_D2TB_basebulk; j++)
			{
				unsigned node_index = j + local_field_offset;
//				std::cout << "NODE IDNEX " << node_index  << "  " << local_field_offset << " " << j << ":" << this->get_D2TB_nodal_data(j)->nvalue() << " " << this->get_D2TB_node_index(j,i) << std::endl;
				eleminfo.nodal_data[i][node_index] =  this->get_D2TB_nodal_data(j)->value_pt(this->get_D2TB_node_index(j,i)); 
				if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] =  this->get_D2TB_local_equation(j, i);
			}
		}
		local_field_offset += functable->numfields_D2TB_basebulk;
	

		for (unsigned int i = 0; i < eleminfo.nnode_C2; i++)
		{
			for (unsigned int j = 0; j < functable->numfields_D2_basebulk; j++)
			{
				unsigned node_index = j + local_field_offset;
				eleminfo.nodal_data[i][node_index] = this->get_D2_nodal_data(j)->value_pt(this->get_D2_node_index(j,i)); 
				if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] =  this->get_D2_local_equation(j, i);
			}
		}
		local_field_offset += functable->numfields_D2_basebulk;		

		for (unsigned int i = 0; i < eleminfo.nnode_C1TB; i++)
		{			
			for (unsigned int j = 0; j < functable->numfields_D1TB_basebulk; j++)
			{
				unsigned node_index = j + local_field_offset;
				eleminfo.nodal_data[i][node_index] = this->get_D1TB_nodal_data(j)->value_pt(this->get_D1TB_node_index(j,i)); 
				if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] =  this->get_D1TB_local_equation(j, i);
			}
		}
		local_field_offset += functable->numfields_D1TB_basebulk;		

                    
		for (unsigned int i = 0; i < eleminfo.nnode_C1; i++)
		{			
			for (unsigned int j = 0; j < functable->numfields_D1_basebulk; j++)
			{
				unsigned node_index = j + local_field_offset;
//				std::cout << "D1 data " << i << "," <<j <<"  " << this->get_D1_nodal_data(j) << "  " << this->get_D1_node_index(j,i) << " @ " << functable<< " name " << std::string(functable->domain_name) << std::endl;
	//			std::cout << "   NV " << this->get_D1_nodal_data(j)->nvalue() << "  NED " << this->nexternal_data() << " NI " << this->ninternal_data()<< std::endl;
		//		for (unsigned ied=0;ied<this->nexternal_data();ied++) std::cout << "  ED " << ied << " has " << this->external_data_pt(ied)->nvalue() << std::endl;
				 
			//	std::cout << "   fieldindex " << j << " vs " << functable->numfields_D1_bulk << " offs " << functable->external_offset_D1_bulk <<std::endl;
				eleminfo.nodal_data[i][node_index] = this->get_D1_nodal_data(j)->value_pt(this->get_D1_node_index(j,i)); 
				if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] =  this->get_D1_local_equation(j, i);
			}
		}
		local_field_offset += functable->numfields_D1_basebulk;		
						
				
		// Elemental (non-continuous) fields				
		// For interface elements,  there is a gap here for indexing. Fill be filled later
		local_field_offset = functable->numfields_C1+functable->numfields_C1TB + functable->numfields_C2 + functable->numfields_C2TB+functable->numfields_D1 + functable->numfields_D2 + functable->numfields_D2TB + functable->numfields_D1TB;
      

		for (unsigned int i = 0; i < eleminfo.nnode_DL; i++)
		{
			for (unsigned int j = 0; j < functable->numfields_DL; j++)
			{
				unsigned node_index = j + local_field_offset;
				eleminfo.nodal_data[i][node_index] = this->get_DL_nodal_data( j)->value_pt(i);
				if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] = this->get_DL_local_equation(j, i);
			}
		}

		local_field_offset += functable->numfields_DL;		

		for (unsigned int j = 0; j < functable->numfields_D0; j++)
		{
			unsigned node_index = j + local_field_offset;
			eleminfo.nodal_data[0][node_index] = this->get_D0_nodal_data(j)->value_pt(0);
			if (!without_equations) eleminfo.nodal_local_eqn[0][node_index] = this->get_D0_local_equation(j);
		}

		local_field_offset = functable->numfields_C2TB + functable->numfields_C2 + functable->numfields_C1+ functable->numfields_C1TB +functable->numfields_D2TB + functable->numfields_D2 + functable->numfields_D1TB+ functable->numfields_D1+ functable->numfields_DL + functable->numfields_D0;

		// Create the information for the external dofs
		for (unsigned int i = 0; i < functable->numfields_ED0; i++)
		{

			unsigned node_index = i + local_field_offset;

			if (!without_equations)
			{
				//		std::cout << "NODE INDEX oF " << functable->fieldnames_ED0[i] << " IS " << node_index << std::endl;
				if (!codeinst->linked_external_data[i].data)
					throw_runtime_error("Element has an external data contribution, which is not assigned: " + std::string(functable->fieldnames_ED0[i]));
				int extdata_i = codeinst->linked_external_data[i].elemental_index+functable->external_offset_ED0;
				if (extdata_i >= (int)this->nexternal_data())
					throw_runtime_error("Somehow the external data array was not done well when trying to index data: " + std::string(functable->fieldnames_ED0[i]) + "  ext_data_index is " + std::to_string(extdata_i) + ", but only " + std::to_string((int)this->nexternal_data()) + " ext data slots present. Happened in " + codeinst->get_code()->get_file_name());
				int value_i = codeinst->linked_external_data[i].value_index;
				if (value_i < 0 || value_i >= (int)this->external_data_pt(extdata_i)->nvalue())
					throw_runtime_error("Somehow the external data array was not done, i.e. wrong value index, well when trying to index data: " + std::string(functable->fieldnames_ED0[i]) + " at value " + std::to_string(value_i));
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

	double BulkElementBase::fill_shape_info_at_s(const oomph::Vector<double> &s, const unsigned int &index, const JITFuncSpec_RequiredShapes_FiniteElement_t &required, double &JLagr, unsigned int flag, oomph::DenseMatrix<double> *dxds) const
	{
		return fill_shape_info_at_s(s, index, required, shape_info, JLagr, flag, dxds);
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

	double BulkElementBase::J_Lagrangian(const oomph::Vector<double> &s)
	{
		unsigned el_dim = this->dim();
		unsigned n_node = this->nnode();
		unsigned n_lagr = this->nlagrangian();

		std::cout << "NLAGR " << n_lagr << "  " << el_dim << std::endl;
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
	
	
	void BulkElementBase::fill_shape_info_element_sizes(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info,unsigned flag) const
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();	
		bool require_hessian = flag > 2;
		bool require_dxdshape = (flag && functable->moving_nodes && (!functable->fd_position_jacobian)); //&& (required.dx_psi_C2 || required.dx_psi_C1 || required.dx_psi_DL)			
		bool require_dx_elemsize=require_dxdshape && (required.elemsize_Eulerian_Pos ||  required.elemsize_Eulerian_cartesian_Pos);
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
		 req_dummy.psi_Pos=req_dummy.dx_psi_Pos=true; // Calculate these
		 double JLagr;
       for(unsigned ipt_for_esize=0;ipt_for_esize<integral_pt()->nweight();ipt_for_esize++)		 
       {  
          //double w = integral_pt()->weight(ipt_for_esize);
          oomph::Vector<double> s_for_esize(this->dim());
          for (unsigned int _i = 0; _i < this->dim(); _i++)	s_for_esize[_i] = integral_pt()->knot(ipt_for_esize, _i);
          this->fill_shape_info_at_s(s_for_esize,0,req_dummy, JLagr, flag);
          oomph::Vector<double> x_for_esize(this->nodal_dimension(),0.0);
          std::vector<double> dJdx(this->nodal_dimension(),0.0);
          std::vector<double> d2Jdx2(this->nodal_dimension()*this->nodal_dimension(),0.0);   
          double J=1.0;       
          if (required.elemsize_Eulerian_Pos)
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
				 if (required.elemsize_Eulerian_Pos)
				 {
				   shape_info->elemsize_d_coords[i][l]+=shape_info->int_pt_weights_d_coords[i][l]*J;				  
				   shape_info->elemsize_d_coords[i][l]+=shape_info->int_pt_weight*dJdx[i]*shape_info->shape_Pos[l];
				 }
				 if (require_hessian)
				 {
					for (unsigned int j=0;j<this->nodal_dimension();j++)
				 	{
				  		for (unsigned int m=0;m<this->nnode();m++)
						{
	  		      		shape_info->elemsize_Cart_d2_coords[i][j][l][m]+=shape_info->int_pt_weights_d2_coords[i][j][l][m];		    		    
	  		      		if (required.elemsize_Eulerian_Pos)
				         {
				            //TODO: Check this
					   		shape_info->elemsize_d2_coords[i][j][l][m]+=shape_info->int_pt_weights_d2_coords[i][j][l][m]*J;
					   		shape_info->elemsize_d2_coords[i][j][l][m]+=shape_info->int_pt_weight*d2Jdx2[i*this->nodal_dimension()+j]*shape_info->shape_Pos[l]*shape_info->shape_Pos[m];  
					   		shape_info->elemsize_d2_coords[i][j][l][m]+=shape_info->int_pt_weights_d_coords[i][l]*dJdx[j]*shape_info->shape_Pos[m];
//					   		std::cout << "For i,l = "<< i << "," << l <<  " : dJdx[" << j<< "] = " <<  dJdx[j] << " and shape_Pos["<<m<<"] = "<<shape_info->shape_Pos[m] << std::endl;
//					   		shape_info->elemsize_d2_coords[i][j][l][m]+=shape_info->int_pt_weights_d_coords[j][m]*dJdx[i]*shape_info->shape_Pos[l];    
					   		
				         }
	  		      	}
	  		      }
	  		    }				 
				}
			 }          
       }
		}
		
		/*
		// TODO: REMOVE!
		if (require_hessian && required.elemsize_Eulerian_Pos)
		{
		 //Calc and compare by FD
		 std::vector<std::vector<std::vector<std::vector<double>>>> oldres;
		 std::vector<std::vector<double>> base;		 
		 oldres.resize(this->nodal_dimension());
		 base.resize(this->nodal_dimension());
		 for (unsigned int i=0;i<this->nodal_dimension();i++)
		 {
		   oldres[i].resize(this->nodal_dimension());
		   base[i].resize(this->nnode());		   
			for (unsigned int l=0;l<this->nnode();l++)  base[i][l]=shape_info->elemsize_d_coords[i][l];
			for (unsigned int j=0;j<this->nodal_dimension();j++)
			{		 
		      oldres[i][j].resize(this->nnode());			
				for (unsigned int l=0;l<this->nnode();l++)
				{
			      oldres[i][j][l].resize(this->nnode());			
					for (unsigned int m=0;m<this->nnode();m++)
					{
					  oldres[i][j][l][m]=shape_info->elemsize_d2_coords[i][j][l][m];	
					}
				}
			}
		 }
		 // Now get the FD
		 for (unsigned int i=0;i<this->nodal_dimension();i++)
		 {		   
			for (unsigned int j=0;j<this->nodal_dimension();j++)
			{		 
				for (unsigned int l=0;l<this->nnode();l++)
				{
					for (unsigned int m=0;m<this->nnode();m++)
					{
					  pyoomph::Node * nod=dynamic_cast<pyoomph::Node *>(this->node_pt(m));
					  double old=nod->variable_position_pt()->value(j);
					  double eps=1e-7;
					  nod->variable_position_pt()->set_value(j,old+eps);
					 JITFuncSpec_RequiredShapes_FiniteElement_t req_dummy;
					 memset(&req_dummy,0,sizeof(JITFuncSpec_RequiredShapes_FiniteElement_t));
					 req_dummy.psi_Pos=req_dummy.dx_psi_Pos=true; // Calculate these					  
					 req_dummy.elemsize_Eulerian_Pos=true;   req_dummy.elemsize_Eulerian_cartesian_Pos=true;
					  this->fill_shape_info_element_sizes(req_dummy,shape_info,1);
					  double FD_res=(shape_info->elemsize_d_coords[i][l]-base[i][l])/eps;
//					  if (std::fabs(FD_res)>1e-8 || std::fabs(oldres[i][j][l][m])>1e-8)

					  if (std::fabs(FD_res-oldres[i][j][l][m])>1e-5 )
					  {

					    std::cout << "DIFF " << i << " " << j << " nodes " << l << " " << m << " : " << FD_res << "  " << oldres[i][j][l][m] << std::endl;
					  }
					  
					  nod->variable_position_pt()->set_value(j,old);
					}
				}
			}
		 }		 
		}
		*/
		
		
		if (required.elemsize_Eulerian_Pos || required.elemsize_Lagrangian_Pos)
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
          if (required.elemsize_Eulerian_Pos)
          {          
            this->interpolated_x(s_for_esize,x_for_esize);
            double J = J_eulerian_at_knot(ipt_for_esize);            
            shape_info->elemsize_Eulerian += w*J*functable->JacobianForElementSize(&eleminfo, &(x_for_esize[0]));
          }
          if (required.elemsize_Lagrangian_Pos)
          {
            this->interpolated_xi(s_for_esize,x_for_esize);
            double J = J_lagrangian_at_knot(ipt_for_esize);            
            shape_info->elemsize_Lagrangian += w*J*functable->JacobianForElementSize(&eleminfo, &(x_for_esize[0]));          
          }
        }
		}
		if (required.elemsize_Eulerian_cartesian_Pos || required.elemsize_Lagrangian_cartesian_Pos)
		{
		  shape_info->elemsize_Eulerian_cartesian=0.0;
		  shape_info->elemsize_Lagrangian_cartesian=0.0;		  
        for(unsigned ipt_for_esize=0;ipt_for_esize<integral_pt()->nweight();ipt_for_esize++)
        {
          double w = integral_pt()->weight(ipt_for_esize);
          oomph::Vector<double> s_for_esize(this->dim());
          for (unsigned int _i = 0; _i < this->dim(); _i++)	s_for_esize[_i] = integral_pt()->knot(ipt_for_esize, _i);
          oomph::Vector<double> x_for_esize(this->nodal_dimension(),0.0);
          if (required.elemsize_Eulerian_cartesian_Pos)
          {          
            this->interpolated_x(s_for_esize,x_for_esize);
            double J = J_eulerian_at_knot(ipt_for_esize);            
            shape_info->elemsize_Eulerian_cartesian += w*J;
          }
          if (required.elemsize_Lagrangian_cartesian_Pos)
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

	double BulkElementBase::fill_shape_info_at_s(const oomph::Vector<double> &s, const unsigned int &index, const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, double &JLagr, unsigned int flag, oomph::DenseMatrix<double> *dxds) const
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
					interpolated_t(j, i) += this->nodal_position(l, i) * dpsids_Element(l, j);
				}
			}
			for (unsigned i = 0; i < n_lagr; i++)
			{
				for (unsigned j = 0; j < el_dim; j++)
				{
					// interpolated_T(j,i) += dynamic_cast<pyoomph::Node*>(this->node_pt(l))->xi(i)*dpsids_Element(l,j);
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

		//   if (index==1) std::cout << "REQUIRED " <<  require_dxdshape << " SINCE FLAG IS " << flag << " and moving nodes is " << functable->moving_nodes << " and fd_pos_jac is " << functable->fd_position_jacobian << " and REQ is " << (required.dx_psi_C2 || required.dx_psi_C1 || required.dx_psi_DL) << std::endl;
		//   require_dxdshape=true;
		
 
		
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
			for (unsigned l = 0; l < eleminfo.nnode_C2TB; l++)
			{
				shape_info->shape_C2TB[l] = 1.0;
				for (unsigned int i = 0; i < n_dim; i++)
					shape_info->dx_shape_C2TB[l][i] = 0.0;
				for (unsigned int i = 0; i < n_lagr; i++)
					shape_info->dX_shape_C2TB[l][i] = 0.0;
				for (unsigned int i = 0; i < el_dim; i++)
					shape_info->dS_shape_C2TB[l][i] = 0.0;
			}
			for (unsigned l = 0; l < eleminfo.nnode_C2; l++)
			{
				shape_info->shape_C2[l] = 1.0;
				for (unsigned int i = 0; i < n_dim; i++)
					shape_info->dx_shape_C2[l][i] = 0.0;
				for (unsigned int i = 0; i < n_lagr; i++)
					shape_info->dX_shape_C2[l][i] = 0.0;
				for (unsigned int i = 0; i < el_dim; i++)
					shape_info->dS_shape_C2[l][i] = 0.0;
			}
			for (unsigned l = 0; l < eleminfo.nnode_C1TB; l++)
			{
				shape_info->shape_C1TB[l] = 1.0;
				for (unsigned int i = 0; i < n_dim; i++)
					shape_info->dx_shape_C1TB[l][i] = 0.0;
				for (unsigned int i = 0; i < n_lagr; i++)
					shape_info->dX_shape_C1TB[l][i] = 0.0;
				for (unsigned int i = 0; i < el_dim; i++)
					shape_info->dS_shape_C1TB[l][i] = 0.0;
			}
			for (unsigned l = 0; l < eleminfo.nnode_C1; l++)
			{
				shape_info->shape_C1[l] = 1.0;
				for (unsigned int i = 0; i < n_dim; i++)
					shape_info->dx_shape_C1[l][i] = 0.0;
				for (unsigned int i = 0; i < n_lagr; i++)
					shape_info->dX_shape_C1[l][i] = 0.0;
				for (unsigned int i = 0; i < el_dim; i++)
					shape_info->dS_shape_C1[l][i] = 0.0;
			}
			for (unsigned l = 0; l < eleminfo.nnode_DL; l++)
			{
				shape_info->shape_C2[l] = 1.0;
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
		// TODO: Optimize this
		//  bool C2_required_since_C2TB_is_C2=false;
		//  if (this->has_bubble())
		//  {
		bool required_C2TB = required.dx_psi_C2TB || required.psi_C2TB;
		required_C2TB |= eleminfo.nnode_C2TB && (required.psi_Pos || required.dx_psi_Pos || required.dX_psi_Pos || required.dX_psi_C2TB) && ((!strcmp(functable->dominant_space, "C2TB")) || (!strcmp(functable->dominant_space, "")));
		if (required_C2TB)
		{
			oomph::Shape psi(eleminfo.nnode_C2TB);
			oomph::DShape dpsids(eleminfo.nnode_C2TB, std::max((unsigned int)1, el_dim));
			this->dshape_local_at_s_C2TB(s, psi, dpsids);
			for (unsigned l = 0; l < eleminfo.nnode_C2TB; l++)
			{
				shape_info->shape_C2TB[l] = psi[l];
				for (unsigned int i = 0; i < n_dim; i++)
				{
					shape_info->dx_shape_C2TB[l][i] = 0.0;
					for (unsigned b = 0; b < el_dim; b++)
					{
						shape_info->dx_shape_C2TB[l][i] += gab_gai[b][i] * dpsids(l, b);						
					}
				}
				
				for (unsigned int i=0; i < this->dim();i++) shape_info->dS_shape_C2TB[l][i] =  dpsids(l, i);

				for (unsigned int i = 0; i < n_lagr; i++)
				{
					shape_info->dX_shape_C2TB[l][i] = 0.0;
					for (unsigned b = 0; b < el_dim; b++)
					{
						shape_info->dX_shape_C2TB[l][i] += gab_gai_Lagr[b][i] * dpsids(l, b);
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
								shape_info->d_dx_shape_dcoord_C2TB[l][i][l2][i2] = 0.0;
								for (unsigned int b = 0; b < el_dim; b++)
								{
									shape_info->d_dx_shape_dcoord_C2TB[l][i][l2][i2] += DXdshape_il_jb(i2, l2, i, b) * dpsids(l, b); // TODO: Also for all other shapes (C1, DL)
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
									shape_info->d2_dx2_shape_dcoord_C2TB[l][i][lel][j][lel2][j2] = 0.0;
									for (unsigned int b = 0; b < el_dim; b++)
									{
										shape_info->d2_dx2_shape_dcoord_C2TB[l][i][lel][j][lel2][j2] += (*D2X2_dshape)(i,b,lel,lel2,j,j2) * dpsids(l, b);
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
		//}
		//  else
		//  {
		//    if ((required.dx_psi_C2TB || required.psi_C2TB ) && (eleminfo.nnode_C2TB) )
		//    {
		//      C2_required_since_C2TB_is_C2=true;
		//    }
		//  }
		// C2_required_since_C2TB_is_C2 ||
		bool required_C2 = required.dx_psi_C2 || required.psi_C2;
		required_C2 |= eleminfo.nnode_C2 && (required.psi_Pos || required.dx_psi_Pos || required.dX_psi_Pos || required.dX_psi_C2) && ((!strcmp(functable->dominant_space, "C2")) || ((!strcmp(functable->dominant_space, "")) && !eleminfo.nnode_C2TB));

		if (required_C2)
		{
			oomph::Shape psi(eleminfo.nnode_C2);
			oomph::DShape dpsids(eleminfo.nnode_C2, std::max((unsigned int)1, el_dim));
			this->dshape_local_at_s_C2(s, psi, dpsids);
			for (unsigned l = 0; l < eleminfo.nnode_C2; l++)
			{
				shape_info->shape_C2[l] = psi[l];
				for (unsigned int i = 0; i < n_dim; i++)
				{
					shape_info->dx_shape_C2[l][i] = 0.0;
					for (unsigned b = 0; b < el_dim; b++)
					{
						shape_info->dx_shape_C2[l][i] += gab_gai[b][i] * dpsids(l, b);
					}
				}

				for (unsigned int i=0; i < this->dim();i++) shape_info->dS_shape_C2[l][i] =  dpsids(l, i);

				for (unsigned int i = 0; i < n_lagr; i++)
				{
					shape_info->dX_shape_C2[l][i] = 0.0;
					for (unsigned b = 0; b < el_dim; b++)
					{
						shape_info->dX_shape_C2[l][i] += gab_gai_Lagr[b][i] * dpsids(l, b);
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
								shape_info->d_dx_shape_dcoord_C2[l][i][l2][i2] = 0.0;
								for (unsigned int b = 0; b < el_dim; b++)
								{
									shape_info->d_dx_shape_dcoord_C2[l][i][l2][i2] += DXdshape_il_jb(i2, l2, i, b) * dpsids(l, b); // TODO: Also for all other shapes (C1, DL)
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
									shape_info->d2_dx2_shape_dcoord_C2[l][i][lel][j][lel2][j2] = 0.0;
									for (unsigned int b = 0; b < el_dim; b++)
									{
										shape_info->d2_dx2_shape_dcoord_C2[l][i][lel][j][lel2][j2] += (*D2X2_dshape)(i,b,lel,lel2,j,j2) * dpsids(l, b);
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

		bool required_C1TB = required.dx_psi_C1TB || required.psi_C1TB;
		required_C1TB |= eleminfo.nnode_C1TB && (required.psi_Pos || required.dx_psi_Pos || required.dX_psi_Pos || required.dX_psi_C1TB) && ((!strcmp(functable->dominant_space, "C1TB")) || ((!strcmp(functable->dominant_space, "")) && !eleminfo.nnode_C2TB));

		if (required_C1TB)
		{
			oomph::Shape psi(eleminfo.nnode_C1TB);
//			std::cout << "NNODE C1TB : " << eleminfo.nnode_C1TB << std::endl << std::flush;
			oomph::DShape dpsids(eleminfo.nnode_C1TB, std::max((unsigned int)1, el_dim));
			this->dshape_local_at_s_C1TB(s, psi, dpsids);
			for (unsigned l = 0; l < eleminfo.nnode_C1TB; l++)
			{
				shape_info->shape_C1TB[l] = psi[l];
				for (unsigned int i = 0; i < n_dim; i++)
				{
					shape_info->dx_shape_C1TB[l][i] = 0.0;
					for (unsigned b = 0; b < el_dim; b++)
					{
						shape_info->dx_shape_C1TB[l][i] += gab_gai[b][i] * dpsids(l, b);
					}
				}

				for (unsigned int i=0; i < this->dim();i++) shape_info->dS_shape_C1TB[l][i] =  dpsids(l, i);

				for (unsigned int i = 0; i < n_lagr; i++)
				{
					shape_info->dX_shape_C1TB[l][i] = 0.0;
					for (unsigned b = 0; b < el_dim; b++)
					{
						shape_info->dX_shape_C1TB[l][i] += gab_gai_Lagr[b][i] * dpsids(l, b);
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
								shape_info->d_dx_shape_dcoord_C1TB[l][i][l2][i2] = 0.0;
								for (unsigned int b = 0; b < el_dim; b++)
								{
									shape_info->d_dx_shape_dcoord_C1TB[l][i][l2][i2] += DXdshape_il_jb(i2, l2, i, b) * dpsids(l, b);
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
									shape_info->d2_dx2_shape_dcoord_C1TB[l][i][lel][j][lel2][j2] = 0.0;
									for (unsigned int b = 0; b < el_dim; b++)
									{
										shape_info->d2_dx2_shape_dcoord_C1TB[l][i][lel][j][lel2][j2] += (*D2X2_dshape)(i,b,lel,lel2,j,j2) * dpsids(l, b);
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
		

		bool required_C1 = required.dx_psi_C1 || required.psi_C1;
		required_C1 |= eleminfo.nnode_C1 && (required.psi_Pos || required.dx_psi_Pos || required.dX_psi_Pos || required.dX_psi_C1) && ((!strcmp(functable->dominant_space, "C1")) || ((!strcmp(functable->dominant_space, "")) && !eleminfo.nnode_C2));

		if (required_C1)
		{
			oomph::Shape psi(eleminfo.nnode_C1);
			oomph::DShape dpsids(eleminfo.nnode_C1, std::max((unsigned int)1, el_dim));
			this->dshape_local_at_s_C1(s, psi, dpsids);
			for (unsigned l = 0; l < eleminfo.nnode_C1; l++)
			{
				shape_info->shape_C1[l] = psi[l];
				for (unsigned int i = 0; i < n_dim; i++)
				{
					shape_info->dx_shape_C1[l][i] = 0.0;
					for (unsigned b = 0; b < el_dim; b++)
					{
						shape_info->dx_shape_C1[l][i] += gab_gai[b][i] * dpsids(l, b);
					}
				}

				for (unsigned int i=0; i < this->dim();i++) shape_info->dS_shape_C1[l][i] =  dpsids(l, i);

				for (unsigned int i = 0; i < n_lagr; i++)
				{
					shape_info->dX_shape_C1[l][i] = 0.0;
					for (unsigned b = 0; b < el_dim; b++)
					{
						shape_info->dX_shape_C1[l][i] += gab_gai_Lagr[b][i] * dpsids(l, b);
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
								shape_info->d_dx_shape_dcoord_C1[l][i][l2][i2] = 0.0;
								for (unsigned int b = 0; b < el_dim; b++)
								{
									shape_info->d_dx_shape_dcoord_C1[l][i][l2][i2] += DXdshape_il_jb(i2, l2, i, b) * dpsids(l, b);
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
									shape_info->d2_dx2_shape_dcoord_C1[l][i][lel][j][lel2][j2] = 0.0;
									for (unsigned int b = 0; b < el_dim; b++)
									{
										shape_info->d2_dx2_shape_dcoord_C1[l][i][lel][j][lel2][j2] += (*D2X2_dshape)(i,b,lel,lel2,j,j2) * dpsids(l, b);
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
		if (required.dx_psi_DL || required.psi_DL)
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

		if (required.normal_Pos) // TODO: Better normal
		{
			oomph::Vector<double> unit_normal(this->nodal_dimension());
			this->get_normal_at_s(s, unit_normal, (require_dxdshape ? shape_info->d_normal_dcoord : NULL), ((require_hessian && require_dxdshape) ? shape_info->d2_normal_d2coord : NULL));
			for (unsigned int i = 0; i < nodal_dimension(); i++)
				shape_info->normal[i] = unit_normal[i];
		}
		
		if (D2X2_dshape) delete D2X2_dshape;

		return det_Eulerian;
	}

	/*double BulkElementBase::fill_shape_info_at_s(const oomph::Vector<double> & s,const unsigned int & index,const JITFuncSpec_RequiredShapes_FiniteElement_t & required,JITShapeInfo_t * shape_info,double & JLagr) const
	{
		//Get the elemental stuff
	  oomph::Shape element_psi(eleminfo.nnode);
	  oomph::DShape element_dpsi(eleminfo.nnode,eleminfo.nodal_dim);
	  dshape_local(s,element_psi,element_dpsi);
	  oomph::DenseMatrix<double> inverse_jacobian(eleminfo.nodal_dim);
	  double det;

	  if (required.dx_psi_C2 || required.dx_psi_C1 || required.dx_psi_DL)
	  {
		det = local_to_eulerian_mapping(element_dpsi,inverse_jacobian); //This one causes trouble on interface elements
	  }
	  else
	  {
		det=J_eulerian(s);
	  }

	  if (required.normal)
	  {
		oomph::Vector<double> unit_normal(this->nodal_dimension());
		this->get_normal_at_s(s,unit_normal);
		for (unsigned int i=0;i<nodal_dimension();i++) shape_info->normal[index][i]=unit_normal[i];
	  }



		if (required.dx_psi_C2)
		{
			oomph::Shape psi(eleminfo.nnode_C2);
			oomph::DShape dpsidx(eleminfo.nnode_C2,eleminfo.nodal_dim);
			this->dshape_local_at_s_C2(s,psi,dpsidx);
		  transform_derivatives(inverse_jacobian,dpsidx);
		  for (unsigned l=0;l<eleminfo.nnode_C2;l++)
			{
				shape_info->shape_C2[index][l]=psi[l];
				for (unsigned int i=0;i<eleminfo.nodal_dim;i++) shape_info->dx_shape_C2[index][l][i]=dpsidx(l,i);
			}
		}
		else if (required.psi_C2 || (eleminfo.nnode==eleminfo.nnode_C2 && eleminfo.nnode_C2>0))
		{
			oomph::Shape psi(eleminfo.nnode_C2);
			this->shape_at_s_C2(s,psi);
		  for (unsigned l=0;l<eleminfo.nnode_C2;l++)
			{
				shape_info->shape_C2[index][l]=psi[l];
			}
		}

		if (required.dx_psi_C1 )
		{
			oomph::Shape psi(eleminfo.nnode_C1);
			oomph::DShape dpsidx(eleminfo.nnode_C1,eleminfo.nodal_dim);
			this->dshape_local_at_s_C1(s,psi,dpsidx);
		  transform_derivatives(inverse_jacobian,dpsidx);
		  for (unsigned l=0;l<eleminfo.nnode_C1;l++)
			{
				shape_info->shape_C1[index][l]=psi[l];
				for (unsigned int i=0;i<eleminfo.nodal_dim;i++) shape_info->dx_shape_C1[index][l][i]=dpsidx(l,i);
			}
		}
		else if (required.psi_C1 || (eleminfo.nnode==eleminfo.nnode_C1 && eleminfo.nnode_C1>0))
		{
			oomph::Shape psi(eleminfo.nnode_C1);
			this->shape_at_s_C1(s,psi);
		  for (unsigned l=0;l<eleminfo.nnode_C1;l++)
			{
				shape_info->shape_C1[index][l]=psi[l];
			}
		}


		if (required.dx_psi_DL)
		{
			oomph::Shape psi(eleminfo.nnode_DL);
			oomph::DShape dpsidx(eleminfo.nnode_DL,eleminfo.nodal_dim);
			this->dshape_local_at_s_DL(s,psi,dpsidx);

		  transform_derivatives(inverse_jacobian,dpsidx);
		  for (unsigned l=0;l<eleminfo.nnode_DL;l++)
			{
				shape_info->shape_DL[index][l]=psi[l];
				for (unsigned int i=0;i<eleminfo.nodal_dim;i++) shape_info->dx_shape_DL[index][l][i]=dpsidx(l,i);
			}
		}
		else if (required.psi_DL)
		{
			oomph::Shape psi(eleminfo.nnode_DL);
			this->shape_at_s_DL(s,psi);
		  for (unsigned l=0;l<eleminfo.nnode_DL;l++)
			{
				shape_info->shape_DL[index][l]=psi[l];
			}
		}


		if (required.dx_psi_Lagr || required.psi_Lagr)
		{
		  oomph::Shape element_psi_Lagr(eleminfo.nnode);
		  oomph::DShape element_dpsi_Lagr(eleminfo.nnode,eleminfo.nodal_dim);
		//  for (unsigned int l=0;l<this->nnode();l++) std::cout << "L " << l << " x " << dynamic_cast<Node*>(this->node_pt(l))->xi(0) << "  " << dynamic_cast<Node*>(this->node_pt(l))->xi(1) << std::endl;
		  JLagr=this->dshape_lagrangian(s,element_psi_Lagr,element_dpsi_Lagr);
	//	  std::cout << index <<  "  JLagr " << JLagr << " psi " << element_psi_Lagr[0] <<  "  " << element_dpsi_Lagr(0,0) << std::endl;
		  for (unsigned l=0;l<eleminfo.nnode;l++)
		  {
				shape_info->shape_Lagr[index][l]=element_psi_Lagr[l];
				for (unsigned int i=0;i<eleminfo.nodal_dim;i++) shape_info->dx_shape_Lagr[index][l][i]=element_dpsi_Lagr(l,i);
		  }
		}



	//Manifold surface derivatives
	 if (required.dS_psi_C2 || required.dS_psi_C1 || required.dS_psi_DL)
	 {
		 unsigned el_dim=this->dim();
		 unsigned n_dim=this->nodal_dimension();
		unsigned n_node=this->nnode();

		 //Tangent vectors
		oomph::DenseMatrix<double> interpolated_t(el_dim,n_dim,0.0);
		 oomph::Shape psif(n_node);
		 oomph::DShape dpsifds(n_node,el_dim);
		this->dshape_local(s,psif,dpsifds);
		for(unsigned l=0;l<n_node;l++)
		{
		  for(unsigned i=0;i<n_dim;i++)
		   {
			for(unsigned j=0;j<el_dim;j++)
			 {
			  interpolated_t(j,i) += this->nodal_position(l,i)*dpsifds(l,j);
			 }
		   }
		}

		//Calculate and invert metric tensor depending on the element dimension
		if (el_dim==2)
		{
			  double amet[2][2];
			  for(unsigned al=0;al<2;al++)
				{
				 for(unsigned be=0;be<2;be++)
				  {
					amet[al][be] = 0.0;
					for(unsigned i=0;i<n_dim;i++)
					 {
					  amet[al][be] += interpolated_t(al,i)*interpolated_t(be,i);
					 }
				  }
				}

			  double det_a = amet[0][0]*amet[1][1] - amet[0][1]*amet[1][0];
			  double aup[2][2];
			  aup[0][0] = amet[1][1]/det_a;
			  aup[0][1] = -amet[0][1]/det_a;
			  aup[1][0] = -amet[1][0]/det_a;
			  aup[1][1] = amet[0][0]/det_a;

			  for(unsigned l=0;l<this->eleminfo.nnode_C2;l++)
			  {
				const double dpsi_temp[2] =     {aup[0][0]*dpsifds(l,0) + aup[0][1]*dpsifds(l,1),    aup[1][0]*dpsifds(l,0) + aup[1][1]*dpsifds(l,1)};
				for(unsigned i=0;i<n_dim;i++)
				{
				  shape_info->dS_shape_C2[index][l][i] = dpsi_temp[0]*interpolated_t(0,i)+ dpsi_temp[1]*interpolated_t(1,i);
				}
			  }
			  if (required.dS_psi_C1 || required.dS_psi_DL) throw_runtime_error("IMPLEM ");
		}
		else if (el_dim==1)
		{
		  double a11 = interpolated_t(0,0)*interpolated_t(0,0) +    interpolated_t(0,1)*interpolated_t(0,1);
		  if (required.dS_psi_C2)
		  {
			  for(unsigned l=0;l<this->eleminfo.nnode_C2;l++)
				{
				 for(unsigned i=0;i<n_dim;i++)
				  {
					shape_info->dS_shape_C2[index][l][i] = dpsifds(l,0)*interpolated_t(0,i)/a11;
				  }
				}
		  }
		  if (required.dS_psi_C1 || required.dS_psi_DL) throw_runtime_error("IMPLEM above");
		}
		else
		{
		 throw_runtime_error("implement element dim "+std::to_string(el_dim));
		}



	 }

	 if (required.dS_psi_Lagr)
	 {

		 unsigned el_dim=this->dim();
		 unsigned n_dim=this->nodal_dimension();
		unsigned n_node=this->nnode();

		 //Tangent vectors
		oomph::DenseMatrix<double> interpolated_t(el_dim,n_dim,0.0);
		 oomph::Shape psif(n_node);
		 oomph::DShape dpsifds(n_node,el_dim);
		this->dshape_lagrangian(s,psif,dpsifds);
		for(unsigned l=0;l<n_node;l++)
		{
		  for(unsigned i=0;i<n_dim;i++)
		   {
			for(unsigned j=0;j<el_dim;j++)
			 {
			  interpolated_t(j,i) += dynamic_cast<pyoomph::Node*>(this->node_pt(l))->xi(i)*dpsifds(l,j);
			 }
		   }
		}

		 if (el_dim==1)
		{
		  double a11 = interpolated_t(0,0)*interpolated_t(0,0) +    interpolated_t(0,1)*interpolated_t(0,1);
		  for(unsigned l=0;l<this->eleminfo.nnode_C2;l++)
			{
				 for(unsigned i=0;i<n_dim;i++)
				  {
					shape_info->dS_shape_Lagr[index][l][i] = dpsifds(l,0)*interpolated_t(0,i)/a11;
				  }
			}
		}
		else
		{
		 throw_runtime_error("implement element dim "+std::to_string(el_dim));
		}

	 }



	 return det;
	}*/

	void BulkElementBase::set_remaining_shapes_appropriately(JITShapeInfo_t *shape_info, const JITFuncSpec_RequiredShapes_FiniteElement_t &required_shapes)
	{
		bool required_C2TB = required_shapes.dx_psi_C2TB || required_shapes.psi_C2TB;
		required_C2TB |= eleminfo.nnode_C2TB && (required_shapes.psi_Pos || required_shapes.dx_psi_Pos || required_shapes.dX_psi_Pos || required_shapes.dX_psi_C2TB) && (!strcmp(this->codeinst->get_func_table()->dominant_space, "C2TB"));
		
		bool required_C1TB = required_shapes.dx_psi_C1TB || required_shapes.psi_C1TB;		
		required_C1TB |= eleminfo.nnode_C1TB && (required_shapes.psi_Pos || required_shapes.dx_psi_Pos || required_shapes.dX_psi_Pos || required_shapes.dX_psi_C1TB) && (!strcmp(this->codeinst->get_func_table()->dominant_space, "C1TB"));
		
		if (required_C2TB)
		{
			/*  if (!this->has_bubble())
			  {
			   shape_info->shape_C2TB=shape_info->shape_C2;
			   shape_info->dx_shape_C2TB=shape_info->dx_shape_C2;
			   shape_info->dX_shape_C2TB=shape_info->dX_shape_C2;
			   shape_info->d_dx_shape_dcoord_C2TB=shape_info->d_dx_shape_dcoord_C2;
			  }*/
			shape_info->shape_Pos = shape_info->shape_C2TB;
			shape_info->dx_shape_Pos = shape_info->dx_shape_C2TB;
			shape_info->dX_shape_Pos = shape_info->dX_shape_C2TB;
			shape_info->dS_shape_Pos = shape_info->dS_shape_C2TB;
			shape_info->d_dx_shape_dcoord_Pos = shape_info->d_dx_shape_dcoord_C2TB;
			shape_info->d2_dx2_shape_dcoord_Pos=shape_info->d2_dx2_shape_dcoord_C2TB;
		}
		else if (this->eleminfo.nnode_C2)
		{
			shape_info->shape_Pos = shape_info->shape_C2;
			shape_info->dx_shape_Pos = shape_info->dx_shape_C2;
			shape_info->dX_shape_Pos = shape_info->dX_shape_C2;
			shape_info->dS_shape_Pos = shape_info->dS_shape_C2;
			shape_info->d_dx_shape_dcoord_Pos = shape_info->d_dx_shape_dcoord_C2;
			shape_info->d2_dx2_shape_dcoord_Pos=shape_info->d2_dx2_shape_dcoord_C2;
		}
		else if (required_C1TB)
		{
			shape_info->shape_Pos = shape_info->shape_C1TB;
			shape_info->dx_shape_Pos = shape_info->dx_shape_C1TB;
			shape_info->dX_shape_Pos = shape_info->dX_shape_C1TB;
			shape_info->dS_shape_Pos = shape_info->dS_shape_C1TB;
			shape_info->d_dx_shape_dcoord_Pos = shape_info->d_dx_shape_dcoord_C1TB;		
			shape_info->d2_dx2_shape_dcoord_Pos=shape_info->d2_dx2_shape_dcoord_C1TB;
		}		
		else
		{
		  
			shape_info->shape_Pos = shape_info->shape_C1;
			shape_info->dx_shape_Pos = shape_info->dx_shape_C1;
			shape_info->dX_shape_Pos = shape_info->dX_shape_C1;
			shape_info->dS_shape_Pos = shape_info->dS_shape_C1;
			shape_info->d_dx_shape_dcoord_Pos = shape_info->d_dx_shape_dcoord_C1;
			shape_info->d2_dx2_shape_dcoord_Pos=shape_info->d2_dx2_shape_dcoord_C1;
		}
	}

	void BulkElementBase::fill_shape_buffer_for_integration_point(unsigned ipt, const JITFuncSpec_RequiredShapes_FiniteElement_t &required_shapes, unsigned int flag)
	{
		oomph::Vector<double> s(this->dim());
		for (unsigned int i = 0; i < this->dim(); i++)
			s[i] = integral_pt()->knot(ipt, i);
		double JLagr;
		double J = fill_shape_info_at_s(s, ipt, required_shapes, JLagr, flag);
		double w = integral_pt()->weight(ipt);
		shape_info->int_pt_weight_unity= w;
		shape_info->int_pt_weight = w * J;
		shape_info->int_pt_weight_Lagrangian = w * JLagr;
	}

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
				else if (unsteady_steps_done <= 4)
				{
					shape_info->timestepper_weights_dt_BDF2_degr = shape_info->timestepper_weights_dt_BDF2;
					shape_info->timestepper_weights_dt_Newmark2_degr = shape_info->timestepper_weights_dt_BDF2;
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

	double BulkElementBase::eval_integral_expression(unsigned index)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		if (index >= functable->numintegral_expressions)
			throw_runtime_error("Cannot evaluate integral expression at too large index " + std::to_string(index));
		// this->fill_hang_info_with_equations(functable->shapes_required_IntegralExprs,this->shape_info,NULL);
		this->interpolate_hang_values();
		prepare_shape_buffer_for_integration(functable->shapes_required_IntegralExprs, 0);
		return functable->EvalIntegralExpression(&eleminfo, this->shape_info, index);
	}

	double BulkElementBase::eval_local_expression_at_node(unsigned index, unsigned node_index)
	{
		oomph::Vector<double> s;
		this->local_coordinate_of_node(node_index, s);
		return eval_local_expression_at_s(index, s);
	}

	double BulkElementBase::eval_local_expression_at_s(unsigned index, const oomph::Vector<double> &s)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		if (index >= functable->numlocal_expressions)
			throw_runtime_error("Cannot evaluate local expression at too large index " + std::to_string(index));

		// this->fill_hang_info_with_equations(functable->shapes_required_LocalExprs,this->shape_info,NULL);
		this->interpolate_hang_values();

		double JLagr;
		this->fill_shape_info_at_s(s, 0, codeinst->get_func_table()->shapes_required_LocalExprs, JLagr, 0);
		this->prepare_shape_buffer_for_integration(codeinst->get_func_table()->shapes_required_LocalExprs, 0);
//		set_remaining_shapes_appropriately(shape_info, codeinst->get_func_table()->shapes_required_LocalExprs);
      _currently_assembled_element = this;
	    //std::cout << "CALLING EVAL LOCAL EXPRESSION  " << this << " ELEMINFO " << &eleminfo << std::endl;
		return functable->EvalLocalExpression(&eleminfo, this->shape_info, index);
	}

	double BulkElementBase::eval_extremum_expression_at_node(unsigned index, unsigned node_index)
	{
		oomph::Vector<double> s;
		this->local_coordinate_of_node(node_index, s);
		return eval_extremum_expression_at_s(index, s);
	}

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

	void BulkElementBase::get_multi_assembly(std::vector<SinglePassMultiAssembleInfo> &info)
	{
		JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		JITFuncSpec_RequiredShapes_FiniteElement_t *required_shapes = (JITFuncSpec_RequiredShapes_FiniteElement_t *)std::calloc(1, sizeof(JITFuncSpec_RequiredShapes_FiniteElement_t));
		int shapeflag = -1;
		// std::cout << "MERGED ASSEMBLY " << std::endl;
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
				// Fill the shape buffer once
				if (shared_multi_assemble)
				{
				  this->fill_shape_buffer_for_integration_point(i_int_pt,*required_shapes,shapeflag);
				}

				// Base contribution
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

	void BulkElementBase::assemble_hessian_tensor(oomph::DenseMatrix<double> &hbuffer)
	{
		oomph::DenseMatrix<double> dummy(this->ndof(),this->ndof()*this->ndof(),0.0);// For the mass matrix
		fill_in_generic_hessian(oomph::Vector<double>(this->ndof(),0.0), dummy, hbuffer, 3);
	}

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
		if (func_table->numfields_D2TB || func_table->numfields_D1TB || func_table->numfields_D2 || func_table->numfields_D1 || func_table->numfields_DL || func_table->numfields_D0)
		    {
		     throw_runtime_error("Cannot interpolate discontinuous fields yet");
		    }
		if ((func_table->numfields_C2TB-func_table->numfields_C2TB_basebulk) || (func_table->numfields_C2-func_table->numfields_C2_basebulk) || (func_table->numfields_C1-func_table->numfields_C1_basebulk))
		    {
		     throw_runtime_error("Cannot interpolate interface fields yet");
		    }

		field_map.resize(func_table->numfields_C2TB_basebulk + func_table->numfields_C2_basebulk + func_table->numfields_C1_basebulk);
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

	void BulkElementBase::fill_in_contribution_to_hessian_vector_products(oomph::Vector<double> const &Y, oomph::DenseMatrix<double> const &C, oomph::DenseMatrix<double> &product)
	{
		oomph::DenseMatrix<double> Ccopy = C;
		this->fill_in_generic_hessian(Y, Ccopy, product, 0);
	}

	std::vector<std::string> BulkElementBase::get_dof_names(bool not_a_root_call)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		std::vector<std::string> res(this->ndof(), "<unknown>");

		// First nonhanging pos
		for (unsigned int i = 0; i < eleminfo.nnode; i++)
		{
			for (unsigned int j = 0; j < std::min(functable->nodal_dim, functable->numfields_Pos); j++)
			{
				if (!this->node_pt(i)->is_hanging())
				{
					if (eleminfo.pos_local_eqn[i][j] >= 0)
						res[eleminfo.pos_local_eqn[i][j]] = std::string(functable->fieldnames_Pos[j]) + "__Pos__" + std::to_string(i);
				}
			}
		}
		// Now hanging pos
		for (unsigned int i = 0; i < eleminfo.nnode; i++)
		{
			for (unsigned int j = 0; j < std::min(functable->nodal_dim, functable->numfields_Pos); j++)
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
								res[local_unknown] = "__HANGING__" + std::to_string(m) + "__of__" + std::string(functable->fieldnames_Pos[j]) + "__Pos__" + std::to_string(i);
							}
						}
					}
				}
			}
		}

		// First nonhanging C2TB

		int hanging_index = (codeinst->get_func_table()->bulk_position_space_to_C1 ? 0 : -1);

		for (unsigned int i = 0; i < eleminfo.nnode_C2TB; i++)
		{
			for (unsigned int j = 0; j < functable->numfields_C2TB_basebulk; j++)
			{
				if (!this->node_pt(this->get_node_index_C2TB_to_element(i))->is_hanging(hanging_index))
				{
					unsigned val_index = j + functable->nodal_offset_C2TB_basebulk;
					// std::cout << "IN C2TB " << i << " FIELD " << j << "  HANG " << this->node_pt(this->get_node_index_C2TB_to_element(i))->is_hanging(hanging_index) << " VALIND " << val_index <<   "  EQ "  << eleminfo.nodal_local_eqn[i][val_index] <<  std::endl;
					if (eleminfo.nodal_local_eqn[i][val_index] >= 0)
						res[eleminfo.nodal_local_eqn[i][val_index]] = std::string(functable->fieldnames_C2TB[j]) + "__C2TB__" + std::to_string(i);
				}
			}
		}
		// Now nonhanging C2TB

		for (unsigned int i = 0; i < eleminfo.nnode_C2TB; i++)
		{
			for (unsigned int j = 0; j < functable->numfields_C2TB_basebulk; j++)
			{
				if (this->node_pt(this->get_node_index_C2TB_to_element(i))->is_hanging(hanging_index))
				{
					oomph::HangInfo *hang_info_pt = this->node_pt(this->get_node_index_C2TB_to_element(i))->hanging_pt(hanging_index);
					const unsigned n_master = hang_info_pt->nmaster();
					for (unsigned m = 0; m < n_master; m++)
					{
						int local_unknown = this->local_hang_eqn(hang_info_pt->master_node_pt(m), j+ functable->nodal_offset_C2TB_basebulk);
						if (local_unknown >= 0)
						{
							if (res[local_unknown] == "<unknown>")
							{
								res[local_unknown] = "__HANGING__" + std::to_string(m) + "__of__" + std::string(functable->fieldnames_C2TB[j]) + "__C2TB__" + std::to_string(i);
							}
						}
					}
				}
			}
		}

		// First nonhanging C2

		for (unsigned int i = 0; i < eleminfo.nnode_C2; i++)
		{
			for (unsigned int j = 0; j < functable->numfields_C2_basebulk; j++)
			{
				if (!this->node_pt(this->get_node_index_C2_to_element(i))->is_hanging(hanging_index))
				{
					unsigned val_index = j + functable->buffer_offset_C2_basebulk;
					if (eleminfo.nodal_local_eqn[i][val_index] >= 0)
						res[eleminfo.nodal_local_eqn[i][val_index]] = std::string(functable->fieldnames_C2[j]) + "__C2__" + std::to_string(i);
				}
			}
		}
		// Now hanging C2

		for (unsigned int i = 0; i < eleminfo.nnode_C2; i++)
		{
			for (unsigned int j = 0; j < functable->numfields_C2_basebulk; j++)
			{
				if (this->node_pt(this->get_node_index_C2_to_element(i))->is_hanging(hanging_index))
				{
					oomph::HangInfo *hang_info_pt = this->node_pt(this->get_node_index_C2_to_element(i))->hanging_pt(hanging_index);
					const unsigned n_master = hang_info_pt->nmaster();
					for (unsigned m = 0; m < n_master; m++)
					{
						int local_unknown = this->local_hang_eqn(hang_info_pt->master_node_pt(m), j + functable->nodal_offset_C2_basebulk);
						if (local_unknown >= 0)
						{
							if (res[local_unknown] == "<unknown>")
							{
								res[local_unknown] = "__HANGING__" + std::to_string(m) + "__of__" + std::string(functable->fieldnames_C2[j]) + "__C2__" + std::to_string(i);
							}
						}
					}
				}
			}
		}
		
		
		int C1_hangindex=functable->nodal_offset_C2_basebulk+functable->nodal_offset_C2TB_basebulk;
		for (unsigned int i = 0; i < eleminfo.nnode_C1TB; i++)
		{
			for (unsigned int j = 0; j < functable->numfields_C1TB_basebulk; j++)
			{
				if (!this->node_pt(this->get_node_index_C1TB_to_element(i))->is_hanging(C1_hangindex))
				{
					unsigned node_index = j + functable->buffer_offset_C1TB_basebulk;
					if (eleminfo.nodal_local_eqn[i][node_index] >= 0)
						res[eleminfo.nodal_local_eqn[i][node_index]] = std::string(functable->fieldnames_C1TB[j]) + "__C1TB__" + std::to_string(i);
				}
			}
		}

		for (unsigned int i = 0; i < eleminfo.nnode_C1TB; i++)
		{
			for (unsigned int j = 0; j < functable->numfields_C1TB_basebulk; j++)
			{
				if (this->node_pt(this->get_node_index_C1TB_to_element(i))->is_hanging(C1_hangindex))
				{
					oomph::HangInfo *hang_info_pt = this->node_pt(this->get_node_index_C1TB_to_element(i))->hanging_pt(C1_hangindex);
					const unsigned n_master = hang_info_pt->nmaster();
					for (unsigned m = 0; m < n_master; m++)
					{
						int local_unknown = this->local_hang_eqn(hang_info_pt->master_node_pt(m), j +functable->nodal_offset_C1TB_basebulk);
						if (local_unknown >= 0)
						{
							if (res[local_unknown] == "<unknown>")
							{
								res[local_unknown] = "__HANGING__" + std::to_string(m) + "__of__" + std::string(functable->fieldnames_C1TB[j]) + "__C1TB__" + std::to_string(i);
							}
						}
					}
				}
			}
		}
		for (unsigned int i = 0; i < eleminfo.nnode_C1; i++)
		{
			for (unsigned int j = 0; j < functable->numfields_C1_basebulk; j++)
			{
				if (!this->node_pt(this->get_node_index_C1_to_element(i))->is_hanging(C1_hangindex))
				{
					unsigned node_index = j + functable->buffer_offset_C1_basebulk;
					if (eleminfo.nodal_local_eqn[i][node_index] >= 0)
						res[eleminfo.nodal_local_eqn[i][node_index]] = std::string(functable->fieldnames_C1[j]) + "__C1__" + std::to_string(i);
				}
			}
		}

		for (unsigned int i = 0; i < eleminfo.nnode_C1; i++)
		{
			for (unsigned int j = 0; j < functable->numfields_C1_basebulk; j++)
			{
				if (this->node_pt(this->get_node_index_C1_to_element(i))->is_hanging(C1_hangindex))
				{
					oomph::HangInfo *hang_info_pt = this->node_pt(this->get_node_index_C1_to_element(i))->hanging_pt(C1_hangindex);
					const unsigned n_master = hang_info_pt->nmaster();
					for (unsigned m = 0; m < n_master; m++)
					{
						int local_unknown = this->local_hang_eqn(hang_info_pt->master_node_pt(m), j +functable->nodal_offset_C1_basebulk);
						if (local_unknown >= 0)
						{
							if (res[local_unknown] == "<unknown>")
							{
								res[local_unknown] = "__HANGING__" + std::to_string(m) + "__of__" + std::string(functable->fieldnames_C1[j]) + "__C1__" + std::to_string(i);
							}
						}
					}
				}
			}
		}

		//  Additional interface dofs ( will be added by the overridden method )


		// DG Fields (these should fill  the additional dofs as well
	        for (unsigned int j = 0; j < functable->numfields_D2TB; j++)
		{
			for (unsigned int i = 0; i < eleminfo.nnode_C2TB; i++)
			{
			   int loc_eq=this->get_D2TB_local_equation(j,i);
			   if (loc_eq >= 0) res[loc_eq] = std::string(functable->fieldnames_D2TB[j]) + "__D2TB__" + std::to_string(i);
			}
		}

	        for (unsigned int j = 0; j < functable->numfields_D2; j++)
		{
			for (unsigned int i = 0; i < eleminfo.nnode_C2; i++)
			{
			   int loc_eq=this->get_D2_local_equation(j,i);
			   if (loc_eq >= 0) res[loc_eq] = std::string(functable->fieldnames_D2[j]) + "__D2__" + std::to_string(i);
			}
		}
		
		for (unsigned int j = 0; j < functable->numfields_D1TB; j++)
		{
			for (unsigned int i = 0; i < eleminfo.nnode_C1TB; i++)
			{
			   int loc_eq=this->get_D1TB_local_equation(j,i);
			   if (loc_eq >= 0) res[loc_eq] = std::string(functable->fieldnames_D1TB[j]) + "__D1TB__" + std::to_string(i);
			}
		}

	        for (unsigned int j = 0; j < functable->numfields_D1; j++)
		{
			for (unsigned int i = 0; i < eleminfo.nnode_C1; i++)
			{
			   int loc_eq=this->get_D1_local_equation(j,i);
			   if (loc_eq >= 0) res[loc_eq] = std::string(functable->fieldnames_D1[j]) + "__D1__" + std::to_string(i);
			}
		}



		for (unsigned int i = 0; i < eleminfo.nnode_DL; i++)
		{
			for (unsigned int j = 0; j < functable->numfields_DL; j++)
			{
				unsigned node_index = j + functable->buffer_offset_DL;
				if (eleminfo.nodal_local_eqn[i][node_index] >= 0)
					res[eleminfo.nodal_local_eqn[i][node_index]] = std::string(functable->fieldnames_DL[j]) + "__DL__" + std::to_string(i);
			}
		}

		for (unsigned int j = 0; j < functable->numfields_D0; j++)
		{
			unsigned node_index = j + functable->buffer_offset_D0;
			if (eleminfo.nodal_local_eqn[0][node_index] >= 0)
				res[eleminfo.nodal_local_eqn[0][node_index]] = std::string(functable->fieldnames_D0[j]) + "__D0";
		}
		
		for (unsigned int j = 0; j < functable->numfields_ED0; j++)
		{
			unsigned node_index = j + functable->buffer_offset_ED0;
			if (eleminfo.nodal_local_eqn[0][node_index] >= 0)
				res[eleminfo.nodal_local_eqn[0][node_index]] = std::string(functable->fieldnames_ED0[j]) + "__ExternalODE";
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
								if (n >= functable->numfields_C1TB_basebulk+functable->numfields_C1_basebulk + functable->numfields_C2_basebulk + functable->numfields_C2TB_basebulk && this->node_pt(l)->is_on_boundary())
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
	

	void BulkElementBase::get_debug_jacobian_info(oomph::Vector<double> &R, oomph::DenseMatrix<double> &J, std::vector<std::string> &dofnames)
	{
		dofnames = get_dof_names();
		R.resize(this->ndof(), 0);
		J.resize(this->ndof(), this->ndof(), 0);
		this->fill_in_contribution_to_jacobian(R, J);
	}

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
			if (eleminfo.nnode_C2TB)
			{
				std::cout << "HANG INFO C2TB" << std::endl;
				for (unsigned int l = 0; l < eleminfo.nnode_C2TB; l++)
				{
					std::cout << "\t" << l << " nmst: " << shape_info->hanginfo_C2TB[l].nummaster;
					for (int m = 0; m < shape_info->hanginfo_C2TB[l].nummaster; m++)
					{
						std::cout << "\t\t weight:" << shape_info->hanginfo_C2TB[l].masters[m].weight << "\t";
						for (unsigned int j = 0; j < functable->numfields_C2TB_basebulk; j++)
							std::cout << "\t" << shape_info->hanginfo_C2TB[l].masters[m].local_eqn[j];
					}
					std::cout << std::endl;
				}
			}
			if (eleminfo.nnode_C2)
			{
				std::cout << "HANG INFO C2" << std::endl;
				for (unsigned int l = 0; l < eleminfo.nnode_C2; l++)
				{
					std::cout << "\t" << l << " nmst: " << shape_info->hanginfo_C2[l].nummaster;
					for (int m = 0; m < shape_info->hanginfo_C2[l].nummaster; m++)
					{
						std::cout << "\t\t weight:" << shape_info->hanginfo_C2[l].masters[m].weight << "\t";
						for (unsigned int j = functable->numfields_C2TB_basebulk; j < functable->numfields_C2TB_basebulk + functable->numfields_C2_basebulk; j++)
							std::cout << "\t" << shape_info->hanginfo_C2[l].masters[m].local_eqn[j];
					}
					std::cout << std::endl;
				}
			}
			if (eleminfo.nnode_C1)
			{
				std::cout << "HANG INFO C1" << std::endl;
				for (unsigned int l = 0; l < eleminfo.nnode_C1; l++)
				{
					std::cout << "\t" << l << " nmst: " << shape_info->hanginfo_C1[l].nummaster;
					for (int m = 0; m < shape_info->hanginfo_C1[l].nummaster; m++)
					{
						std::cout << "\t\t weight:" << shape_info->hanginfo_C1[l].masters[m].weight << "\t";
						for (unsigned int j = functable->numfields_C2_basebulk + functable->numfields_C2TB_basebulk; j < functable->numfields_C2TB_basebulk + functable->numfields_C2_basebulk + functable->numfields_C1_basebulk; j++)
							std::cout << "\t" << shape_info->hanginfo_C1[l].masters[m].local_eqn[j];
					}
					std::cout << std::endl;
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
				if (bel->eleminfo.nnode_C2TB)
				{
					std::cout << "BULK HANG INFO C2TB" << std::endl;
					for (unsigned int l = 0; l < bel->eleminfo.nnode_C2TB; l++)
					{
						std::cout << "\t" << l << " nmst: " << shape_info->bulk_shapeinfo->hanginfo_C2TB[l].nummaster;
						for (int m = 0; m < shape_info->bulk_shapeinfo->hanginfo_C2TB[l].nummaster; m++)
						{
							std::cout << "\t\t weight:" << shape_info->bulk_shapeinfo->hanginfo_C2TB[l].masters[m].weight << "\t";
							for (unsigned int j = 0; j < functable->numfields_C2TB_basebulk; j++)
								std::cout << "\t" << shape_info->bulk_shapeinfo->hanginfo_C2TB[l].masters[m].local_eqn[j];
						}
						std::cout << std::endl;
					}
				}
				if (bel->eleminfo.nnode_C2)
				{
					std::cout << "BULK HANG INFO C2" << std::endl;
					for (unsigned int l = 0; l < bel->eleminfo.nnode_C2; l++)
					{
						std::cout << "\t" << l << " nmst: " << shape_info->bulk_shapeinfo->hanginfo_C2[l].nummaster;
						for (int m = 0; m < shape_info->bulk_shapeinfo->hanginfo_C2[l].nummaster; m++)
						{
							std::cout << "\t\t weight:" << shape_info->bulk_shapeinfo->hanginfo_C2[l].masters[m].weight << "\t";
							for (unsigned int j = functable->numfields_C2TB_basebulk; j < functable->numfields_C2TB_basebulk + functable->numfields_C2_basebulk; j++)
								std::cout << "\t" << shape_info->bulk_shapeinfo->hanginfo_C2[l].masters[m].local_eqn[j];
						}
						std::cout << std::endl;
					}
				}
				if (eleminfo.nnode_C1)
				{
					std::cout << "BULK HANG INFO C1" << std::endl;
					for (unsigned int l = 0; l < bel->eleminfo.nnode_C1; l++)
					{
						std::cout << "\t" << l << " nmst: " << shape_info->bulk_shapeinfo->hanginfo_C1[l].nummaster;
						for (int m = 0; m < shape_info->bulk_shapeinfo->hanginfo_C1[l].nummaster; m++)
						{
							std::cout << "\t\t weight:" << shape_info->bulk_shapeinfo->hanginfo_C1[l].masters[m].weight << "\t";
							for (unsigned int j = functable->numfields_C2_basebulk + functable->numfields_C2TB_basebulk; j < functable->numfields_C2TB_basebulk + functable->numfields_C2_basebulk + functable->numfields_C1_basebulk; j++)
								std::cout << "\t" << shape_info->bulk_shapeinfo->hanginfo_C1[l].masters[m].local_eqn[j];
						}
						std::cout << std::endl;
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
	 if (functable->moving_nodes && (functable->shapes_required_ResJac[functable->current_res_jac].elemsize_Eulerian_cartesian_Pos || functable->shapes_required_ResJac[functable->current_res_jac].elemsize_Eulerian_Pos))
	 {
//	  std::cout << "UPDATE CALL" << std::endl;
	  this->fill_shape_info_element_sizes(functable->shapes_required_ResJac[functable->current_res_jac],shape_info,0);
	 }
	}

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

	unsigned BulkElementBase::required_nvalue(const unsigned &n) const
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		return functable->numfields_C2TB_basebulk + functable->numfields_C2_basebulk + functable->numfields_C1TB_basebulk +functable->numfields_C1_basebulk;
	}

	oomph::Node *BulkElementBase::construct_node(const unsigned &n)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		unsigned ntot = functable->numfields_C2TB_basebulk + functable->numfields_C2_basebulk + functable->numfields_C1TB_basebulk + functable->numfields_C1_basebulk;
		//	 std::cout << "NLAGR " <<  this->nlagrangian() << "  " << this->nnodal_lagrangian_type() << std::endl;
		node_pt(n) = new Node(this->nlagrangian(), this->nnodal_lagrangian_type(), this->nodal_dimension(), this->nnodal_position_type(), ntot);
		return node_pt(n);
	}

	oomph::Node *BulkElementBase::construct_node(const unsigned &n, oomph::TimeStepper *const &time_stepper_pt)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		unsigned ntot = functable->numfields_C2TB_basebulk + functable->numfields_C2_basebulk + functable->numfields_C1TB_basebulk+ functable->numfields_C1_basebulk;
		//		 		 std::cout << "NLAGR " <<  this->nlagrangian() << "  " << this->nnodal_lagrangian_type() << std::endl;
		node_pt(n) = new Node(time_stepper_pt, this->nlagrangian(), this->nnodal_lagrangian_type(), this->nodal_dimension(), this->nnodal_position_type(), ntot);
		return node_pt(n);
	}

	oomph::Node *BulkElementBase::construct_boundary_node(const unsigned &n)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		unsigned ntot = functable->numfields_C2TB_basebulk + functable->numfields_C2_basebulk + functable->numfields_C1TB_basebulk+ functable->numfields_C1_basebulk;
		node_pt(n) = new BoundaryNode(this->nlagrangian(), this->nnodal_lagrangian_type(), this->nodal_dimension(), this->nnodal_position_type(), ntot);
		return node_pt(n);
	}

	oomph::Node *BulkElementBase::construct_boundary_node(const unsigned &n, oomph::TimeStepper *const &time_stepper_pt)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		unsigned ntot = functable->numfields_C2TB_basebulk + functable->numfields_C2_basebulk +functable->numfields_C1TB_basebulk+ functable->numfields_C1_basebulk;
		node_pt(n) = new BoundaryNode(time_stepper_pt, this->nlagrangian(), this->nnodal_lagrangian_type(), this->nodal_dimension(), this->nnodal_position_type(), ntot);
		return node_pt(n);
	}

	unsigned BulkElementBase::nadditional_fields_C1()
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		return functable->numfields_C1 - functable->numfields_C1_basebulk;
	}

	unsigned BulkElementBase::nadditional_fields_C2()
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		return functable->numfields_C2 - functable->numfields_C2_basebulk;
	}

	unsigned BulkElementBase::nadditional_fields_C1TB()
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		return functable->numfields_C1TB - functable->numfields_C1TB_basebulk;
	}

	unsigned BulkElementBase::nadditional_fields_C2TB()
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		return functable->numfields_C2TB - functable->numfields_C2TB_basebulk;
	}

	unsigned BulkElementBase::ncont_interpolated_values() const
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		return functable->numfields_C1_basebulk + functable->numfields_C1TB_basebulk + functable->numfields_C2_basebulk + functable->numfields_C2TB_basebulk;
	}

	oomph::Vector<double> BulkElementBase::get_midpoint_s() // Set s=[0.5*(smin+smax), ... ] (but modified e.g. for tris)
	{
		return oomph::Vector<double>(this->dim(), 0.5 * (this->s_min() + this->s_max()));
	}

	double BulkElementBase::eval_local_expression_at_midpoint(unsigned index)
	{
		oomph::Vector<double> s = this->get_midpoint_s();
		return eval_local_expression_at_s(index, s);
	}

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


			//DG 			
			if (functable->numfields_D2TB_new)
			{
				for (unsigned l=0;l<this->get_eleminfo()->nnode_C2TB;l++)
				{
					oomph::Vector<double> sfather,father_data;				
					this->get_nodal_s_in_father(this->get_node_index_C2TB_to_element(l), sfather);
					father->get_D2TB_fields_at_s(t,sfather,father_data);
					for (unsigned iindex=0;iindex<functable->numfields_D2TB_new;iindex++)
					{
						this->internal_data_pt(functable->internal_offset_D2TB_new+iindex)->set_value(t,l,father_data[iindex]);
					}
				}
			}
			if (functable->numfields_D2_new)
			{
				for (unsigned l=0;l<this->get_eleminfo()->nnode_C2;l++)
				{
					oomph::Vector<double> sfather,father_data;				
					this->get_nodal_s_in_father(this->get_node_index_C2_to_element(l), sfather);
					father->get_D2_fields_at_s(t,sfather,father_data);
					for (unsigned iindex=0;iindex<functable->numfields_D2_new;iindex++)
					{
						this->internal_data_pt(functable->internal_offset_D2_new+iindex)->set_value(t,l,father_data[iindex]);
					}
				}
			}
			if (functable->numfields_D1TB_new)
			{
				for (unsigned l=0;l<this->get_eleminfo()->nnode_C1TB;l++)
				{
					oomph::Vector<double> sfather,father_data;				
					this->get_nodal_s_in_father(this->get_node_index_C1TB_to_element(l), sfather);
					father->get_D1TB_fields_at_s(t,sfather,father_data);
					for (unsigned iindex=0;iindex<functable->numfields_D1TB_new;iindex++)
					{
						this->internal_data_pt(functable->internal_offset_D1TB_new+iindex)->set_value(t,l,father_data[iindex]);
					}
				}
			}
			
			if (functable->numfields_D1_new)
			{
				for (unsigned l=0;l<this->get_eleminfo()->nnode_C1;l++)
				{
					oomph::Vector<double> sfather,father_data;				
					this->get_nodal_s_in_father(this->get_node_index_C1_to_element(l), sfather);
					father->get_D1_fields_at_s(t,sfather,father_data);
					for (unsigned iindex=0;iindex<functable->numfields_D1_new;iindex++)
					{
						this->internal_data_pt(functable->internal_offset_D1_new+iindex)->set_value(t,l,father_data[iindex]);
					}
				}
			}


			//DL and D0
			unsigned DL_offset=functable->internal_offset_DL;
			for (unsigned int iindex = DL_offset; iindex < DL_offset+functable->numfields_DL; iindex++)
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
			for (unsigned int iindex = functable->numfields_DL+DL_offset; iindex < DL_offset+functable->numfields_DL + functable->numfields_D0; iindex++) // D0 fields
			{
				double factor = 1;
				if (functable->discontinuous_refinement_exponents[functable->buffer_offset_D0 + iD0] != 0.0) // TODO: Consider on DL as well
				{
					factor = pow(nsons, -functable->discontinuous_refinement_exponents[functable->buffer_offset_D0 + iD0]);
				}
				internal_data_pt(iindex)->set_value(t, 0, factor * father->internal_data_pt(iindex)->value(t, 0));
				iD0++;
			}
		}
		this->set_integration_scheme(father->integral_pt());
		this->ensure_external_data();
	}

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
		if (functable->numfields_D2TB_new || functable->numfields_D2_new || functable->numfields_D1_new || functable->numfields_D1TB_new)
		{
			for (unsigned t = 0; t < node_pt(0)->time_stepper_pt()->ntstorage(); t++)
			{
				if (functable->numfields_D2TB_new)
				{				
					const unsigned Nn=this->get_eleminfo()->nnode_C2TB;
					std::vector<double> denom(Nn,0.0);
					for (unsigned int iindex = functable->internal_offset_D2TB_new; iindex < functable->internal_offset_D2TB_new+functable->numfields_D2TB_new; iindex++) 
					{
						for (unsigned int in=0;in<Nn;in++) this->internal_data_pt(iindex)->set_value(t,in,0.0); //Set to 0
					}
					for (unsigned ison = 0; ison < this->required_nsons(); ison++)
					{
						BulkElementBase* son=dynamic_cast<BulkElementBase*>(Tree_pt->son_pt(ison)->object_pt());
						for (unsigned int in=0;in<Nn;in++)
						{
							oomph::Vector<double> s;				
							son->get_nodal_s_in_father(son->get_node_index_C2TB_to_element(in), s);
							oomph::Node * my_node=this->get_node_at_local_coordinate(s);
							if (my_node)
							{
								int nn=this->get_node_number(my_node);
								if (nn>=0)
								{
									nn=this->get_node_index_element_to_C2TB(nn);
									if (nn>=0)
									{
										for (unsigned int iindex = functable->internal_offset_D2TB_new; iindex < functable->internal_offset_D2TB_new+functable->numfields_D2TB_new; iindex++) 
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
					for (unsigned int iindex = functable->internal_offset_D2TB_new; iindex < functable->internal_offset_D2TB_new+functable->numfields_D2TB_new; iindex++) 
					{
						for (unsigned int in=0;in<Nn;in++) 
						{
							if (denom[in]<0.1) throw_runtime_error("Should not happen");
							this->internal_data_pt(iindex)->set_value(t,in,this->internal_data_pt(iindex)->value(t,in)/denom[in]); 
						}
					}
				}

				if (functable->numfields_D2_new)
				{				
					const unsigned Nn=this->get_eleminfo()->nnode_C2;
					std::vector<double> denom(Nn,0.0);
					for (unsigned int iindex = functable->internal_offset_D2_new; iindex < functable->internal_offset_D2_new+functable->numfields_D2_new; iindex++) 
					{
						for (unsigned int in=0;in<Nn;in++) this->internal_data_pt(iindex)->set_value(t,in,0.0); //Set to 0
					}
					for (unsigned ison = 0; ison < this->required_nsons(); ison++)
					{
						BulkElementBase* son=dynamic_cast<BulkElementBase*>(Tree_pt->son_pt(ison)->object_pt());
						for (unsigned int in=0;in<Nn;in++)
						{
							oomph::Vector<double> s;				
							son->get_nodal_s_in_father(son->get_node_index_C2_to_element(in), s);
							oomph::Node * my_node=this->get_node_at_local_coordinate(s);
							if (my_node)
							{
								int nn=this->get_node_number(my_node);
								if (nn>=0)
								{
									nn=this->get_node_index_element_to_C2(nn);
									if (nn>=0)
									{
										for (unsigned int iindex = functable->internal_offset_D2_new; iindex < functable->internal_offset_D2_new+functable->numfields_D2_new; iindex++) 
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
					for (unsigned int iindex = functable->internal_offset_D2_new; iindex < functable->internal_offset_D2_new+functable->numfields_D2_new; iindex++) 
					{
						for (unsigned int in=0;in<Nn;in++) 
						{
							if (denom[in]<0.1) throw_runtime_error("Should not happen");
							this->internal_data_pt(iindex)->set_value(t,in,this->internal_data_pt(iindex)->value(t,in)/denom[in]); 
						}
					}
				}
				
				if (functable->numfields_D1TB_new)
				{				
					const unsigned Nn=this->get_eleminfo()->nnode_C1TB;
					std::vector<double> denom(Nn,0.0);
					for (unsigned int iindex = functable->internal_offset_D1TB_new; iindex < functable->internal_offset_D1TB_new+functable->numfields_D1TB_new; iindex++) 
					{
						for (unsigned int in=0;in<Nn;in++) this->internal_data_pt(iindex)->set_value(t,in,0.0); //Set to 0
					}
					for (unsigned ison = 0; ison < this->required_nsons(); ison++)
					{
						BulkElementBase* son=dynamic_cast<BulkElementBase*>(Tree_pt->son_pt(ison)->object_pt());
						for (unsigned int in=0;in<Nn;in++)
						{
							oomph::Vector<double> s;				
							son->get_nodal_s_in_father(son->get_node_index_C1TB_to_element(in), s);
							oomph::Node * my_node=this->get_node_at_local_coordinate(s);
//							std::cout << "INFO " << this << " ISON " << ison << "  " << in << " sfather " << s[0] << " MY NODE " << my_node << std::endl;
							if (my_node)
							{
								int nn=this->get_node_number(my_node);
								if (nn>=0)
								{
									nn=this->get_node_index_element_to_C1TB(nn);
									if (nn>=0)
									{
										for (unsigned int iindex = functable->internal_offset_D1TB_new; iindex < functable->internal_offset_D1TB_new+functable->numfields_D1TB_new; iindex++) 
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
					for (unsigned int iindex = functable->internal_offset_D1TB_new; iindex < functable->internal_offset_D1TB_new+functable->numfields_D1TB_new; iindex++) 
					{
						for (unsigned int in=0;in<Nn;in++) 
						{
							if (denom[in]<0.1) throw_runtime_error("Should not happen: at node index "+std::to_string(in)+" of "+std::to_string(Nn));
							this->internal_data_pt(iindex)->set_value(t,in,this->internal_data_pt(iindex)->value(t,in)/denom[in]); 
						}
					}
				}

				if (functable->numfields_D1_new)
				{				
					const unsigned Nn=this->get_eleminfo()->nnode_C1;
					std::vector<double> denom(Nn,0.0);
					for (unsigned int iindex = functable->internal_offset_D1_new; iindex < functable->internal_offset_D1_new+functable->numfields_D1_new; iindex++) 
					{
						for (unsigned int in=0;in<Nn;in++) this->internal_data_pt(iindex)->set_value(t,in,0.0); //Set to 0
					}
					for (unsigned ison = 0; ison < this->required_nsons(); ison++)
					{
						BulkElementBase* son=dynamic_cast<BulkElementBase*>(Tree_pt->son_pt(ison)->object_pt());
						for (unsigned int in=0;in<Nn;in++)
						{
							oomph::Vector<double> s;				
							son->get_nodal_s_in_father(son->get_node_index_C1_to_element(in), s);
							oomph::Node * my_node=this->get_node_at_local_coordinate(s);
//							std::cout << "INFO " << this << " ISON " << ison << "  " << in << " sfather " << s[0] << " MY NODE " << my_node << std::endl;
							if (my_node)
							{
								int nn=this->get_node_number(my_node);
								if (nn>=0)
								{
									nn=this->get_node_index_element_to_C1(nn);
									if (nn>=0)
									{
										for (unsigned int iindex = functable->internal_offset_D1_new; iindex < functable->internal_offset_D1_new+functable->numfields_D1_new; iindex++) 
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
					for (unsigned int iindex = functable->internal_offset_D1_new; iindex < functable->internal_offset_D1_new+functable->numfields_D1_new; iindex++) 
					{
						for (unsigned int in=0;in<Nn;in++) 
						{
							if (denom[in]<0.1) throw_runtime_error("Should not happen: at node index "+std::to_string(in)+" of "+std::to_string(Nn));
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
				for (unsigned int iindex = functable->internal_offset_DL; iindex < functable->internal_offset_DL+functable->numfields_DL; iindex++) // DL fields
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
				for (unsigned int iindex = functable->internal_offset_D0; iindex < functable->internal_offset_D0 + functable->numfields_D0; iindex++) // D0 fields
				{
					// XXX TODO: Allow for other interpolation methods. In particular, this does not conserve (which does not matter for e.g. pressure) and does not consider axisymmetry
					// TODO: Time history loop!
					double av = 0.0;
					for (unsigned ison = 0; ison < 4; ison++)
					{
						av += quadtree_pt->son_pt(ison)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					}
					double avg_factor = 0.25;
					if (functable->discontinuous_refinement_exponents[functable->buffer_offset_D0 + iindex-functable->internal_offset_D0] != 0.0) // TODO: Consider on DL as well
					{
						avg_factor = avg_factor * pow(avg_factor, -functable->discontinuous_refinement_exponents[functable->buffer_offset_D0 + iindex-functable->internal_offset_D0]);
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
				for (unsigned int iindex = functable->internal_offset_DL; iindex < functable->internal_offset_DL+functable->numfields_DL; iindex++) // DL fields
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
				for (unsigned int iindex = functable->internal_offset_D0; iindex < functable->internal_offset_D0 + functable->numfields_D0; iindex++) // D0 fields
				{
					// XXX TODO: Allow for other interpolation methods. In particular, this does not conserve (which does not matter for e.g. pressure) and does not consider axisymmetry
					double av = 0.0;
					for (unsigned ison = 0; ison < 2; ison++)
					{
						av += binarytree_pt->son_pt(ison)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					}
					double avg_factor = 0.5;
					if (functable->discontinuous_refinement_exponents[functable->buffer_offset_D0 + iindex-functable->internal_offset_D0] != 0.0) // TODO: Consider on DL as well
					{
						avg_factor = avg_factor * pow(avg_factor, -functable->discontinuous_refinement_exponents[functable->buffer_offset_D0 + iindex-functable->internal_offset_D0]);
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
				for (unsigned int iindex = functable->internal_offset_DL; iindex < functable->internal_offset_DL+functable->numfields_DL; iindex++) // DL fields
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
				for (unsigned int iindex = functable->internal_offset_D0; iindex < functable->internal_offset_D0 + functable->numfields_D0; iindex++) // D0 fields
				{
					// XXX TODO: Allow for other interpolation methods. In particular, this does not conserve (which does not matter for e.g. pressure) and does not consider axisymmetry
					// TODO: Time history loop!
					double av = 0.0;
					for (unsigned ison = 0; ison < 8; ison++)
					{
						av += octree_pt->son_pt(ison)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					}
					double avg_factor = 0.125;
					if (functable->discontinuous_refinement_exponents[functable->buffer_offset_D0 + iindex-functable->internal_offset_D0] != 0.0) // TODO: Consider on DL as well
					{
						avg_factor = avg_factor * pow(avg_factor, -functable->discontinuous_refinement_exponents[functable->buffer_offset_D0 + iindex-functable->internal_offset_D0]);
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

	std::string BulkElementBase::scalar_name_paraview(const unsigned &i) const
	{
		throw_runtime_error("NOT IMPLEMENTED");
	}

	int BulkElementBase::get_nodal_index_by_name(oomph::Node *n, std::string fieldname)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		
		for (unsigned i = 0; i < functable->numfields_C2TB_basebulk; i++)
			if (std::string(functable->fieldnames_C2TB[i]) == fieldname)
				return i+functable->nodal_offset_C2TB_basebulk;
				
		for (unsigned i = 0; i < functable->numfields_C2_basebulk; i++)
			if (std::string(functable->fieldnames_C2[i]) == fieldname)
				return i +functable->nodal_offset_C2_basebulk;
				
		for (unsigned i = 0; i < functable->numfields_C1TB_basebulk; i++)
			if (std::string(functable->fieldnames_C1TB[i]) == fieldname)
				return i + functable->nodal_offset_C1TB_basebulk;

		for (unsigned i = 0; i < functable->numfields_C1_basebulk; i++)
			if (std::string(functable->fieldnames_C1[i]) == fieldname)
				return i + functable->nodal_offset_C1_basebulk;
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

	void BulkElementBase::get_interpolated_fields_C2TB(const oomph::Vector<double> &s, std::vector<double> &res, const unsigned &t) const
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		res.resize(functable->numfields_C2TB_basebulk);
		oomph::Shape psi(eleminfo.nnode_C2TB);
		this->shape_at_s_C2TB(s, psi);
		for (unsigned int fi = 0; fi < functable->numfields_C2TB_basebulk; fi++)
		{
			res[fi] = 0.0;
			for (unsigned int l = 0; l < eleminfo.nnode_C2TB; l++)
			{
				res[fi] += psi[l] * this->node_pt(this->get_node_index_C2TB_to_element(l))->value(t, fi+ functable->nodal_offset_C2TB_basebulk);
			}
		}
	}

	void BulkElementBase::get_interpolated_fields_C2(const oomph::Vector<double> &s, std::vector<double> &res, const unsigned &t) const
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		res.resize(functable->numfields_C2_basebulk);
		oomph::Shape psi(eleminfo.nnode_C2);
		this->shape_at_s_C2(s, psi);
		for (unsigned int fi = 0; fi < functable->numfields_C2_basebulk; fi++)
		{
			res[fi] = 0.0;
			for (unsigned int l = 0; l < eleminfo.nnode_C2; l++)
			{
				res[fi] += psi[l] * this->node_pt(this->get_node_index_C2_to_element(l))->value(t, fi + functable->nodal_offset_C2_basebulk);
			}
		}
	}

   void BulkElementBase::get_interpolated_fields_C1TB(const oomph::Vector<double> &s, std::vector<double> &res, const unsigned &t) const
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		res.resize(functable->numfields_C1TB_basebulk);
		oomph::Shape psi(eleminfo.nnode_C1TB);
		this->shape_at_s_C1TB(s, psi);
		for (unsigned int fi = 0; fi < functable->numfields_C1TB_basebulk; fi++)
		{
			res[fi] = 0.0;
			for (unsigned int l = 0; l < eleminfo.nnode_C1TB; l++)
			{
				res[fi] += psi[l] * this->node_pt(this->get_node_index_C1TB_to_element(l))->value(t, fi + functable->nodal_offset_C1TB_basebulk);
			}
		}
	}
	
	void BulkElementBase::get_interpolated_fields_C1(const oomph::Vector<double> &s, std::vector<double> &res, const unsigned &t) const
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		res.resize(functable->numfields_C1_basebulk);
		oomph::Shape psi(eleminfo.nnode_C1);
		this->shape_at_s_C1(s, psi);
		for (unsigned int fi = 0; fi < functable->numfields_C1_basebulk; fi++)
		{
			res[fi] = 0.0;
			for (unsigned int l = 0; l < eleminfo.nnode_C1; l++)
			{
				res[fi] += psi[l] * this->node_pt(this->get_node_index_C1_to_element(l))->value(t, fi + functable->nodal_offset_C1_basebulk);
			}
		}
	}

	void BulkElementBase::get_interpolated_fields_DL(const oomph::Vector<double> &s, std::vector<double> &res, const unsigned &t) const
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		res.resize(functable->numfields_DL);
		oomph::Shape psi(eleminfo.nnode_DL);
		this->shape_at_s_DL(s, psi);
		unsigned dg_offset=functable->numfields_D2TB_new+functable->numfields_D2_new+functable->numfields_D1_new+functable->numfields_D1TB_new;
		for (unsigned int fi = 0; fi < functable->numfields_DL; fi++)
		{
			res[fi] = 0.0;
			for (unsigned int l = 0; l < eleminfo.nnode_DL; l++)
			{
				res[fi] += psi[l] * this->internal_data_pt(dg_offset+fi)->value(t, l);
			}
		}
	}

	void BulkElementBase::get_interpolated_fields_D0(const oomph::Vector<double> &s, std::vector<double> &res, const unsigned &t) const
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		res.resize(functable->numfields_D0);
		unsigned dg_offset=functable->numfields_D2TB_new+functable->numfields_D2_new+functable->numfields_D1_new+functable->numfields_D1TB_new;		
		for (unsigned int fi = 0; fi < functable->numfields_D0; fi++)
		{
			res[fi] = this->internal_data_pt(functable->numfields_DL + dg_offset+fi)->value(t, 0);
		}
	}

	void BulkElementBase::get_interpolated_values(const unsigned &t, const oomph::Vector<double> &s, oomph::Vector<double> &values)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		oomph::Vector<double> resC2TB;
		oomph::Vector<double> resC2;
		oomph::Vector<double> resC1TB;		
		oomph::Vector<double> resC1;
		if (functable->numfields_C2TB_basebulk)
			this->get_interpolated_fields_C2TB(s, resC2TB, t);
		if (functable->numfields_C2_basebulk)
			this->get_interpolated_fields_C2(s, resC2, t);
		if (functable->numfields_C1TB_basebulk)
			this->get_interpolated_fields_C1TB(s, resC1TB, t);			
		if (functable->numfields_C1_basebulk)
			this->get_interpolated_fields_C1(s, resC1, t);
		values.resize(resC2TB.size() + resC2.size() + resC1TB.size()+ resC1.size());
		for (unsigned int i = 0; i < resC2TB.size(); i++)
		{
			values[i] = resC2TB[i];
		}
		for (unsigned int i = 0; i < resC2.size(); i++)
		{
			values[i + resC2TB.size()] = resC2[i];
		}
		for (unsigned int i = 0; i < resC1TB.size(); i++)
		{
			values[i + resC2TB.size() + resC2.size()] = resC1TB[i];
		}		
		for (unsigned int i = 0; i < resC1.size(); i++)
		{
			values[i + resC2TB.size() + resC2.size()+resC1TB.size()] = resC1[i];
		}
	}

	void BulkElementBase::get_interpolated_discontinuous_values(const unsigned &t, const oomph::Vector<double> &s, oomph::Vector<double> &values)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		oomph::Vector<double> resDL;
		oomph::Vector<double> resD0;
		if (functable->numfields_DL)
			this->get_interpolated_fields_DL(s, resDL, t);
		if (functable->numfields_D0)
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

	void BulkElementBase::output(std::ostream &outfile, const unsigned &nplot)
	{
		throw_runtime_error("Not implemented");
	}

	unsigned BulkElementBase::num_Z2_flux_terms()
	{
		if (BulkElementBase::use_eigen_error_estimators) return codeinst->get_func_table()->num_Z2_flux_terms_for_eigen;
		else return codeinst->get_func_table()->num_Z2_flux_terms;
	}

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

	void BulkElementBase::further_setup_hanging_nodes()
	{
		// std::cout << "FURTHER SETUP HANG" << std::endl;
	}

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

   unsigned BulkElementBase::num_DG_fields(bool base_bulk_only)
   {
    auto *ft=codeinst->get_func_table();
    if (base_bulk_only)
    {
      return ft->numfields_D2TB_basebulk+ft->numfields_D2_basebulk+ft->numfields_D1TB_basebulk+ft->numfields_D1_basebulk;
    }
    else
    {
      return ft->numfields_D2TB+ft->numfields_D2+ft->numfields_D1TB+ft->numfields_D1;    
    }
   }
   
   void BulkElementBase::get_D1_fields_at_s(unsigned history_index,const oomph::Vector<double> &s, oomph::Vector<double> &result) const
   {
     auto *ft=codeinst->get_func_table();
     result.resize(ft->numfields_D1);
     for (unsigned int i=0;i<ft->numfields_D1;i++) result[i]=0.0;
     oomph::Shape psi(eleminfo.nnode_C1);
     this->shape_at_s_C1(s,psi);
     unsigned nexternal=ft->numfields_D1-ft->numfields_D1_new;
     for (unsigned int i=0;i<nexternal;i++)
     {
      for (unsigned int l=0;l<eleminfo.nnode_C1;l++) 
      {
       result[i]+=external_data_pt(ft->external_offset_D1_bulk+i)->value(history_index,this->get_D1_node_index(i,l))*psi[l];
      } 
     }
     for (unsigned int i=nexternal;i<ft->numfields_D1;i++)
     {
      for (unsigned int l=0;l<eleminfo.nnode_C1;l++) 
      {
       result[i]+=internal_data_pt(ft->internal_offset_D1_new+i-nexternal)->value(history_index,l)*psi[l];
      }
     }
     
   }
   void BulkElementBase::get_D2_fields_at_s(unsigned history_index,const oomph::Vector<double> &s, oomph::Vector<double> &result) const
   {
     auto *ft=codeinst->get_func_table();
     result.resize(ft->numfields_D2);
     for (unsigned int i=0;i<ft->numfields_D2;i++) result[i]=0.0;
     oomph::Shape psi(eleminfo.nnode_C2);
     this->shape_at_s_C2(s,psi);
     unsigned nexternal=ft->numfields_D2-ft->numfields_D2_new;	 
     for (unsigned int i=0;i<nexternal;i++)
     {
	  for (unsigned int l=0;l<eleminfo.nnode_C2;l++) 
      {
       result[i]+=external_data_pt(ft->external_offset_D2_bulk+i)->value(history_index,this->get_D2_node_index(i,l))*psi[l];
      }      
     }
     for (unsigned int i=nexternal;i<ft->numfields_D2;i++)
     {
      for (unsigned int l=0;l<eleminfo.nnode_C2;l++) 
      {
		//std::cout << "ADDING to " <<i << " " << ft->internal_offset_D2_new+i-nexternal << internal_data_pt(ft->internal_offset_D2_new+i-nexternal)->value(history_index,l) << std::endl;
       result[i]+=internal_data_pt(ft->internal_offset_D2_new+i-nexternal)->value(history_index,l)*psi[l];
      }
     }   
   }
   void BulkElementBase::get_D1TB_fields_at_s(unsigned history_index,const oomph::Vector<double> &s, oomph::Vector<double> &result) const
   {
     auto *ft=codeinst->get_func_table();
     result.resize(ft->numfields_D1TB);
     for (unsigned int i=0;i<ft->numfields_D1TB;i++) result[i]=0.0;
     oomph::Shape psi(eleminfo.nnode_C1TB);
     this->shape_at_s_C1TB(s,psi);
     unsigned nexternal=ft->numfields_D1TB-ft->numfields_D1TB_new;
     for (unsigned int i=0;i<nexternal;i++)
     {
      for (unsigned int l=0;l<eleminfo.nnode_C1TB;l++) 
      {
       result[i]+=external_data_pt(ft->external_offset_D1TB_bulk+i)->value(history_index,this->get_D1TB_node_index(i,l))*psi[l];
      }      
     }
     for (unsigned int i=nexternal;i<ft->numfields_D1TB;i++)
     {
      for (unsigned int l=0;l<eleminfo.nnode_C1TB;l++) 
      {
       result[i]+=internal_data_pt(ft->internal_offset_D1TB_new+i-nexternal)->value(history_index,l)*psi[l];
      }
     }      
   }
   void BulkElementBase::get_D2TB_fields_at_s(unsigned history_index,const oomph::Vector<double> &s, oomph::Vector<double> &result) const
   {
     auto *ft=codeinst->get_func_table();
     result.resize(ft->numfields_D2TB);
     for (unsigned int i=0;i<ft->numfields_D2TB;i++) result[i]=0.0;
     oomph::Shape psi(eleminfo.nnode_C2TB);
     this->shape_at_s_C2TB(s,psi);
     unsigned nexternal=ft->numfields_D2TB-ft->numfields_D2TB_new;
     for (unsigned int i=0;i<nexternal;i++)
     {
      for (unsigned int l=0;l<eleminfo.nnode_C2TB;l++) 
      {
       result[i]+=external_data_pt(ft->external_offset_D2TB_bulk+i)->value(history_index,this->get_D2TB_node_index(i,l))*psi[l];
      }      
     }
     for (unsigned int i=nexternal;i<ft->numfields_D2TB;i++)
     {
      for (unsigned int l=0;l<eleminfo.nnode_C2TB;l++) 
      {
       result[i]+=internal_data_pt(ft->internal_offset_D2TB_new+i-nexternal)->value(history_index,l)*psi[l];
      }
     }      
   }   
   
	void BulkElementBase::allocate_discontinous_fields()
	{
	   // DG Fields.
		//Only add the fields directly added in this dimension. Parent degrees will be external data	   
		if (eleminfo.nnode_C2TB > 0)
		{

			for (unsigned int fi = 0; fi < codeinst->get_func_table()->numfields_D2TB_new; fi++)
			{
		//	   std::cout << "ALLOC " << eleminfo.nnode_C2TB << " DDATA for " << this->ninternal_data() << std::endl;
				this->add_internal_data(new oomph::Data(eleminfo.nnode_C2TB), false);
			}
		}
		if (eleminfo.nnode_C2 > 0)
		{
			for (unsigned int fi = 0; fi < codeinst->get_func_table()->numfields_D2_new; fi++)
			{
				this->add_internal_data(new oomph::Data(eleminfo.nnode_C2), false);
				//std::cout << "  AFTER D2 " << fi << "  " << this->ninternal_data() << " INT DATA" << std::endl <<std::flush;
			}
		}
		if (eleminfo.nnode_C1TB > 0)
		{
			for (unsigned int fi = 0; fi < codeinst->get_func_table()->numfields_D1TB_new; fi++)
			{
				this->add_internal_data(new oomph::Data(eleminfo.nnode_C1TB), false);
			}
		}		
		
		if (eleminfo.nnode_C1 > 0)
		{
			for (unsigned int fi = 0; fi < codeinst->get_func_table()->numfields_D1_new; fi++)
			{
				this->add_internal_data(new oomph::Data(eleminfo.nnode_C1), false);
			}
		}		
			
		if (eleminfo.nnode_DL > 0)
		{
			for (unsigned int fi = 0; fi < codeinst->get_func_table()->numfields_DL; fi++)
			{
				this->add_internal_data(new oomph::Data(eleminfo.nnode_DL), false);
				//		          std::cout << "  AFTER DL " << fi << "  " << this->ninternal_data() << " INT DATA" << std::endl <<std::flush;

			}
		}

		for (unsigned int fi = 0; fi < codeinst->get_func_table()->numfields_D0; fi++)
		{
			this->add_internal_data(new oomph::Data(1), false);
		}

//          std::cout << "ALLOCATED " << this->ninternal_data() << " INT DATA" << std::endl <<std::flush;
		
	}

	////////////////////////////////

	oomph::PointIntegral BulkElementODE0d::Default_integration_scheme;

	BulkElementODE0d::BulkElementODE0d(DynamicBulkElementInstance *code_inst, oomph::TimeStepper *tstepper) : timestepper(tstepper)
	{
		this->codeinst = code_inst;
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 1; // One dummy node... Necessary to create the buffers
		eleminfo.nnode_C1 = 0;
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

	void BulkElementODE0d::to_numpy(double *dest)
	{
		unsigned nD0 = codeinst->get_func_table()->numfields_D0;
		for (unsigned int i = 0; i < nD0; i++)
			dest[i] = this->internal_data_pt(i)->value(0); // TODO Scaling
	}

	double BulkElementODE0d::fill_shape_info_at_s(const oomph::Vector<double> &s, const unsigned int &index, const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, double &JLagr, unsigned int flag, oomph::DenseMatrix<double> *dxds) const
	{
		JLagr = 1.0;
		return 1.0;
	}

	////////////////////////////////

	PointElement0d::PointElement0d()
	{
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 1;
		eleminfo.nnode_C1 = 1;
		eleminfo.nnode_C1TB = 1;		
		eleminfo.nnode_C2 = 1;
		eleminfo.nnode_C2TB = 1;
		eleminfo.nnode_DL = 1;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}

	bool PointElement0d::fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap)
	{
		for (unsigned int l = 0; l < eleminfo.nnode; l++) // C2 nodes
		{
			shape_info->hanginfo_Pos[l].nummaster = 0;
		}
		for (unsigned int l = 0; l < eleminfo.nnode_C2TB; l++) // C2 nodes
		{
			shape_info->hanginfo_C2TB[l].nummaster = 0;
		}
		for (unsigned int l = 0; l < eleminfo.nnode_C1TB; l++) // C1TB nodes
		{
			shape_info->hanginfo_C1TB[l].nummaster = 0;
		}		
		for (unsigned int l = 0; l < eleminfo.nnode_C2; l++) // C2 nodes
		{
			shape_info->hanginfo_C2[l].nummaster = 0;
		}
		if (codeinst->get_func_table()->numfields_C1)
		{
			for (unsigned int l = 0; l < eleminfo.nnode_C1; l++) // C1 nodes
			{
				shape_info->hanginfo_C1[l].nummaster = 0;
			}
		}
		/*
		if (codeinst->get_func_table()->numfields_DL)
		{
			for (unsigned int l = 0; l < eleminfo.nnode_DL; l++)
			{
				shape_info->hanginfo_Discont[l].nummaster = 0;
			}
		}
		shape_info->hanginfo_Discont[0].nummaster = 0;
		*/
		for (unsigned int l = 0; l < eleminfo.nnode; l++)
		{
			shape_info->hanginfo_Discont[l].nummaster = 0;
		}


		if (eqn_remap)
		{
			return BulkElementBase::fill_hang_info_with_equations(required, shape_info, eqn_remap);
		}
		else
			return false;
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

	void PointElement0d::fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
	{
	  indices[0]=0;
	}

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

	BulkElementLine1dC1::BulkElementLine1dC1()
	{
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 2;
		eleminfo.nnode_C1TB = 2;		
		eleminfo.nnode_C1 = 2;
		eleminfo.nnode_DL = 2;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}

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

	bool BulkElementLine1dC1::fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap)
	{
		for (unsigned int l = 0; l < eleminfo.nnode_C1; l++) // C1 nodes
		{
			shape_info->hanginfo_C1[l].nummaster = 0;
		}
		for (unsigned int l = 0; l < eleminfo.nnode_C1TB; l++) // C1TB nodes
		{
			shape_info->hanginfo_C1TB[l].nummaster = 0;
		}				
		for (unsigned int l = 0; l < eleminfo.nnode; l++)
		{
			shape_info->hanginfo_Pos[l].nummaster = 0;
		}
/*		for (unsigned int l = 0; l < eleminfo.nnode_DL; l++)
		{
			shape_info->hanginfo_Discont[l].nummaster = 0;
		}
		shape_info->hanginfo_Discont[0].nummaster = 0;
*/
		for (unsigned int l = 0; l < eleminfo.nnode; l++)
		{
			shape_info->hanginfo_Discont[l].nummaster = 0;
		}

		if (eqn_remap)
		{
			return BulkElementBase::fill_hang_info_with_equations(required, shape_info, eqn_remap);
		}
		else
			return false;
	}
	
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

	unsigned int BulkElementLine1dC2::index_C1_to_element[2] = {0, 2};
	int BulkElementLine1dC2::element_index_to_C1[3] = {0,-1,1};
	bool BulkElementLine1dC2::node_only_C2[3] = {false, true, false};

	BulkElementLine1dC2::BulkElementLine1dC2()
	{
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 3;
		eleminfo.nnode_C1 = 2;
		eleminfo.nnode_C1TB = 2;		
		eleminfo.nnode_C2 = 3;
		eleminfo.nnode_C2TB = 3;		
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

	void BulkElementLine1dC2::interpolate_hang_values()
	{
		BulkElementBase::interpolate_hang_values();
		if (codeinst->get_func_table()->numfields_C1_basebulk)
		{
			for (unsigned int i = codeinst->get_func_table()->numfields_C2TB_basebulk + codeinst->get_func_table()->numfields_C2_basebulk; i < codeinst->get_func_table()->numfields_C2TB_basebulk + codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C1_basebulk; i++)
			{
				for (unsigned t = 0; t < node_pt(1)->ntstorage(); t++)
				{
					node_pt(1)->value_pt(i)[t] = 0.5 * (node_pt(0)->value_pt(i)[t] + node_pt(2)->value_pt(i)[t]);
				}
			}
		}
	}

	void BulkElementLine1dC2::interpolate_hang_values_at_interface()
	{
		auto *functable = codeinst->get_func_table();
		unsigned numC1 = functable->numfields_C1 - functable->numfields_C1_basebulk;
		if (numC1)
		{
			for (unsigned int i = 0; i < numC1; i++)
			{
				std::string fieldname = functable->fieldnames_C1[functable->numfields_C1_basebulk + i];
				unsigned interf_id = codeinst->resolve_interface_dof_id(fieldname);
				unsigned valindex0 = dynamic_cast<oomph::BoundaryNodeBase *>(this->node_pt(0))->index_of_first_value_assigned_by_face_element(interf_id);
				unsigned valindex1 = dynamic_cast<oomph::BoundaryNodeBase *>(this->node_pt(1))->index_of_first_value_assigned_by_face_element(interf_id);
				unsigned valindex2 = dynamic_cast<oomph::BoundaryNodeBase *>(this->node_pt(2))->index_of_first_value_assigned_by_face_element(interf_id);
				for (unsigned t = 0; t < node_pt(1)->ntstorage(); t++)
				{
					node_pt(1)->value_pt(valindex1)[t] = 0.5 * (node_pt(0)->value_pt(valindex0)[t] + node_pt(2)->value_pt(valindex2)[t]);
				}
			}
		}
	}

	bool BulkElementLine1dC2::fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap)
	{
		for (unsigned int l = 0; l < eleminfo.nnode; l++) // C2 nodes
		{
			shape_info->hanginfo_Pos[l].nummaster = 0;
			//    std::cout << "REM SHAPEINFO " << l << "  " << shape_info->hanginfo_Pos[l].nummaster << std::endl;
		}
		for (unsigned int l = 0; l < eleminfo.nnode_C2TB; l++) // C2 nodes
		{
			shape_info->hanginfo_C2TB[l].nummaster = 0;
		}
		for (unsigned int l = 0; l < eleminfo.nnode_C2; l++) // C2 nodes
		{
			shape_info->hanginfo_C2[l].nummaster = 0;
		}
		if (codeinst->get_func_table()->numfields_C1)
		{
			for (unsigned int l = 0; l < eleminfo.nnode_C1; l++) // C1 nodes
			{
				shape_info->hanginfo_C1[l].nummaster = 0;
			}
		}
		if (codeinst->get_func_table()->numfields_C1TB)
		{
			for (unsigned int l = 0; l < eleminfo.nnode_C1TB; l++) // C1 nodes
			{
				shape_info->hanginfo_C1TB[l].nummaster = 0;
			}
		}		
/*
		if (codeinst->get_func_table()->numfields_DL)
		{
			for (unsigned int l = 0; l < eleminfo.nnode_DL; l++)
			{
				shape_info->hanginfo_Discont[l].nummaster = 0;
			}
		}
		shape_info->hanginfo_Discont[0].nummaster = 0;
*/
		for (unsigned int l = 0; l < eleminfo.nnode; l++)
		{
			shape_info->hanginfo_Discont[l].nummaster = 0;
		}

		if (eqn_remap)
		{
			return BulkElementBase::fill_hang_info_with_equations(required, shape_info, eqn_remap);
		}
		else
			return false;
	}

	////////////////////////////

	BulkTElementLine1dC1::BulkTElementLine1dC1()
	{
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 2;
		eleminfo.nnode_C1TB = 2;		
		eleminfo.nnode_C1 = 2;
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

	bool BulkTElementLine1dC1::fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap)
	{
		for (unsigned int l = 0; l < eleminfo.nnode_C1; l++) // C1 nodes
		{
			shape_info->hanginfo_C1[l].nummaster = 0;
		}
		if (codeinst->get_func_table()->numfields_C1TB)
		{
			for (unsigned int l = 0; l < eleminfo.nnode_C1TB; l++) // C1 nodes
			{
				shape_info->hanginfo_C1TB[l].nummaster = 0;
			}
		}				
		for (unsigned int l = 0; l < eleminfo.nnode; l++)
		{
			shape_info->hanginfo_Pos[l].nummaster = 0;
		}
		/*
		for (unsigned int l = 0; l < eleminfo.nnode_DL; l++)
		{
			shape_info->hanginfo_Discont[l].nummaster = 0;
		}
		shape_info->hanginfo_Discont[0].nummaster = 0;
		*/
		for (unsigned int l = 0; l < eleminfo.nnode; l++)
		{
			shape_info->hanginfo_Discont[l].nummaster = 0;
		}


		if (eqn_remap)
		{
			return BulkElementBase::fill_hang_info_with_equations(required, shape_info, eqn_remap);
		}
		else
			return false;
	}

	unsigned int BulkTElementLine1dC2::index_C1_to_element[2] = {0, 2};
	int BulkTElementLine1dC2::element_index_to_C1[3] = {0,-1,1};
	bool BulkTElementLine1dC2::node_only_C2[3] = {false, true, false};

	BulkTElementLine1dC2::BulkTElementLine1dC2()
	{
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 3;
		eleminfo.nnode_C1 = 2;
		eleminfo.nnode_C1TB = 2;		
		eleminfo.nnode_C2 = 3;
		eleminfo.nnode_C2TB = 3;
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

	void BulkTElementLine1dC2::interpolate_hang_values()
	{
		BulkElementBase::interpolate_hang_values();
		if (codeinst->get_func_table()->numfields_C1_basebulk)
		{
			for (unsigned int i = codeinst->get_func_table()->numfields_C2TB_basebulk + codeinst->get_func_table()->numfields_C2_basebulk; i < codeinst->get_func_table()->numfields_C2TB_basebulk + codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C1_basebulk; i++)
			{
				for (unsigned t = 0; t < node_pt(1)->ntstorage(); t++)
				{
					node_pt(1)->value_pt(i)[t] = 0.5 * (node_pt(0)->value_pt(i)[t] + node_pt(2)->value_pt(i)[t]);
				}
			}
		}
	}

	void BulkTElementLine1dC2::interpolate_hang_values_at_interface()
	{
		auto *functable = codeinst->get_func_table();
		unsigned numC1 = functable->numfields_C1 - functable->numfields_C1_basebulk;
		if (numC1)
		{
			for (unsigned int i = 0; i < numC1; i++)
			{
				std::string fieldname = functable->fieldnames_C1[functable->numfields_C1_basebulk + i];
				unsigned interf_id = codeinst->resolve_interface_dof_id(fieldname);
				unsigned valindex0 = dynamic_cast<oomph::BoundaryNodeBase *>(this->node_pt(0))->index_of_first_value_assigned_by_face_element(interf_id);
				unsigned valindex1 = dynamic_cast<oomph::BoundaryNodeBase *>(this->node_pt(1))->index_of_first_value_assigned_by_face_element(interf_id);
				unsigned valindex2 = dynamic_cast<oomph::BoundaryNodeBase *>(this->node_pt(2))->index_of_first_value_assigned_by_face_element(interf_id);
				for (unsigned t = 0; t < node_pt(1)->ntstorage(); t++)
				{
					node_pt(1)->value_pt(valindex1)[t] = 0.5 * (node_pt(0)->value_pt(valindex0)[t] + node_pt(2)->value_pt(valindex2)[t]);
				}
			}
		}
	}

	bool BulkTElementLine1dC2::fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap)
	{
		for (unsigned int l = 0; l < eleminfo.nnode; l++) // C2 nodes
		{
			shape_info->hanginfo_Pos[l].nummaster = 0;
			//    std::cout << "REM SHAPEINFO " << l << "  " << shape_info->hanginfo_Pos[l].nummaster << std::endl;
		}
		if (codeinst->get_func_table()->numfields_C1TB)
		{
			for (unsigned int l = 0; l < eleminfo.nnode_C1TB; l++) // C1 nodes
			{
				shape_info->hanginfo_C1TB[l].nummaster = 0;
			}
		}				
		for (unsigned int l = 0; l < eleminfo.nnode_C2TB; l++) // C2 nodes
		{
			shape_info->hanginfo_C2TB[l].nummaster = 0;
		}
		for (unsigned int l = 0; l < eleminfo.nnode_C2; l++) // C2 nodes
		{
			shape_info->hanginfo_C2[l].nummaster = 0;
		}
		if (codeinst->get_func_table()->numfields_C1)
		{
			for (unsigned int l = 0; l < eleminfo.nnode_C1; l++) // C1 nodes
			{
				shape_info->hanginfo_C1[l].nummaster = 0;
			}
		}
		/*

		if (codeinst->get_func_table()->numfields_DL)
		{
			for (unsigned int l = 0; l < eleminfo.nnode_DL; l++)
			{
				shape_info->hanginfo_Discont[l].nummaster = 0;
			}
		}
		shape_info->hanginfo_Discont[0].nummaster = 0;
		*/
		for (unsigned int l = 0; l < eleminfo.nnode; l++)
		{
			shape_info->hanginfo_Discont[l].nummaster = 0;
		}

		if (eqn_remap)
		{
			return BulkElementBase::fill_hang_info_with_equations(required, shape_info, eqn_remap);
		}
		else
			return false;
	}

	////////////////////////////

	////////////////////////////

	BulkElementQuad2dC1::BulkElementQuad2dC1()
	{
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 4;
		eleminfo.nnode_C1 = 4;
		eleminfo.nnode_C1TB = 4;		
		eleminfo.nnode_DL = 3;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}

	bool BulkElementQuad2dC1::fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap)
	{
		bool res = false;
		for (unsigned int l = 0; l < eleminfo.nnode_C1; l++)
		{
			if (node_pt(l)->is_hanging())
			{
				res = true;
				auto hang_info_pt = node_pt(l)->hanging_pt();
				shape_info->hanginfo_C1[l].nummaster = hang_info_pt->nmaster();
				shape_info->hanginfo_C1TB[l].nummaster = hang_info_pt->nmaster();				
				shape_info->hanginfo_Pos[l].nummaster = hang_info_pt->nmaster();
				for (unsigned m = 0; m < hang_info_pt->nmaster(); m++)
				{
					shape_info->hanginfo_C1[l].masters[m].weight = hang_info_pt->master_weight(m);
					shape_info->hanginfo_Pos[l].masters[m].weight = hang_info_pt->master_weight(m);
					for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_C1_basebulk+codeinst->get_func_table()->numfields_C1TB_basebulk; f++)
					{
						shape_info->hanginfo_C1[l].masters[m].local_eqn[f] = this->local_hang_eqn(hang_info_pt->master_node_pt(m), f);
						shape_info->hanginfo_C1TB[l].masters[m].local_eqn[f] = this->local_hang_eqn(hang_info_pt->master_node_pt(m), f);						
					}
					for (unsigned int f = 0; f < this->nodal_dimension(); f++)
					{
						//	      oomph::DenseMatrix<int> position_local_eqn_at_node(this->nnodal_position_type(),this->nodal_dimension());
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
				shape_info->hanginfo_C1[l].nummaster = 0;
				shape_info->hanginfo_C1TB[l].nummaster = 0;				
				shape_info->hanginfo_Pos[l].nummaster = 0;
			}
		}
		/*

		if (codeinst->get_func_table()->numfields_DL)
		{
			for (unsigned int l = 0; l < eleminfo.nnode_DL; l++)
			{
				shape_info->hanginfo_Discont[l].nummaster = 0;
			}
		}
		shape_info->hanginfo_Discont[0].nummaster = 0;
		*/
		for (unsigned int l = 0; l < eleminfo.nnode; l++)
		{
			shape_info->hanginfo_Discont[l].nummaster = 0;
		}

		if (eqn_remap)
		{
			return BulkElementBase::fill_hang_info_with_equations(required, shape_info, eqn_remap);
		}
		else
			return res;
	}

	void BulkElementQuad2dC1::interpolate_hang_values()
	{
		BulkElementBase::interpolate_hang_values();
		for (unsigned int l = 0; l < eleminfo.nnode_C1; l++)
		{
			if (node_pt(l)->is_hanging())
			{
				for (unsigned int i = 0; i < node_pt(l)->nvalue(); i++)
				{
					for (unsigned t = 0; t < node_pt(l)->ntstorage(); t++)
					{
						node_pt(l)->value_pt(i)[t] = node_pt(l)->value(t, i);
					}
				}

				for (unsigned int i = 0; i < node_pt(l)->ndim(); i++)
				{
					for (unsigned t = 0; t < node_pt(l)->ntstorage(); t++)
					{
						dynamic_cast<Node *>(node_pt(l))->variable_position_pt()->set_value(t, i, node_pt(l)->position(t, i));
					}
				}
			}
		}
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

	unsigned int BulkElementQuad2dC2::index_C1_to_element[4] = {0, 2, 6, 8};
	int BulkElementQuad2dC2::element_index_to_C1[9] = {0,-1,1,-1,-1,-1,2,-1,3};
	bool BulkElementQuad2dC2::node_only_C2[9] = {false, true, false, true, true, true, false, true, false};

	BulkElementQuad2dC2::BulkElementQuad2dC2()
	{
		eleminfo.elem_ptr = this;
		// std::cout << "SETTING ELEM PTR " <<  eleminfo.elem_ptr << std::endl;
		eleminfo.nnode = 9;
		eleminfo.nnode_C1 = 4;
		eleminfo.nnode_C1TB = 4;
		eleminfo.nnode_C2 = 9;		
		eleminfo.nnode_C2TB = 9;
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

	void BulkElementBase::constrain_bulk_position_space_to_C1()
	{
		for (unsigned int in = 0; in < this->nnode(); in++)
		{
			Node *n = dynamic_cast<Node *>(this->node_pt(in));
			if (!this->is_node_index_part_of_C1(in) && !n->is_on_boundary())
			{
				std::vector<oomph::Node *> sn;
				this->get_supporting_C1_nodes_of_C2_node(in, sn);
				if (sn.empty())
					throw_runtime_error("Should not happen");
				double w = 1.0 / sn.size();
				std::map<oomph::Node *, double> hangdata;
				for (auto *s : sn)
				{
					if (s->is_hanging(-1))
					{
						oomph::HangInfo *hi = s->hanging_pt(-1);
						for (unsigned int m = 0; m < hi->nmaster(); m++)
						{
							oomph::Node *mn = hi->master_node_pt(m);
							double mw = hi->master_weight(m);
							if (hangdata.count(mn))
								hangdata[mn] += w * mw;
							else
								hangdata[mn] = w * mw;
						}
					}
					else
					{
						if (hangdata.count(s))
							hangdata[s] += w;
						else
							hangdata[s] = w;
					}
				}
				oomph::HangInfo *hang_pt = new oomph::HangInfo(hangdata.size());
				unsigned cnt = 0;
				double hangsum = 0.0;
				for (auto &hd : hangdata)
				{
					hang_pt->set_master_node_pt(cnt, hd.first, hd.second);
					hangsum += hd.second;
					cnt++;
					std::cout << "ADDING HANG INFO " << in << "  " << cnt << " " << hd.first << "  " << hd.second << std::endl;
				}
				std::cout << "HANGSUM " << hangsum << std::endl;
				n->set_hanging_pt(hang_pt, -1);
			}
		}
	}

	void BulkElementQuad2dC2::constrain_bulk_position_space_to_C1()
	{

		BulkElementBase::constrain_bulk_position_space_to_C1();

		for (unsigned int i = 0; i < codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk; i++)
		{
			setup_hang_for_value(i);
		}
	}

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

	void BulkElementQuad2dC2::further_setup_hanging_nodes()
	{

		BulkElementBase::further_setup_hanging_nodes();

		if (codeinst->get_func_table()->numfields_C1_basebulk)
		{
			//   this->setup_hang_for_value(codeinst->get_func_table()->numfields_C2);
			for (unsigned int i = codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk; i < codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk + codeinst->get_func_table()->numfields_C1_basebulk; i++)
				this->setup_hang_for_value(i);
			/* 	 for (unsigned int i=codeinst->get_func_table()->numfields_C2+1;i<codeinst->get_func_table()->numfields_C2+codeinst->get_func_table()->numfields_C1;i++)
			   {
					 for (unsigned int l=0;l<eleminfo.nnode_C1;l++)
				 {
			//			this->node_pt_C1(l)->hanging_pt(i)==this->node_pt_C1(l)->hanging_pt(codeinst->get_func_table()->numfields_C2);
			//		std::cout << "HANING " << l << "  " << node_pt_C1(l) << std::endl;
			//		std::cout << "HANING " << l << "  " << node_pt_C1(l)->hanging_pt(codeinst->get_func_table()->numfields_C2) << std::endl;
			//			 this->node_pt_C1(l)->set_hanging_pt(node_pt_C1(l)->hanging_pt(codeinst->get_func_table()->numfields_C2), i); //Copy the Hang-Info for all other C1 values
				 }
			   }
			*/
		}

		if (codeinst->get_func_table()->bulk_position_space_to_C1)
		{
			this->constrain_bulk_position_space_to_C1();
		}
	}

	bool BulkElementQuad2dC2::fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap)
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

		int hanging_index = (codeinst->get_func_table()->bulk_position_space_to_C1 ? 0 : -1);

		if (codeinst->get_func_table()->numfields_C2_basebulk || codeinst->get_func_table()->numfields_C2TB_basebulk)
		{
			for (unsigned int l = 0; l < eleminfo.nnode_C2; l++) // C2 nodes
			{
				if (node_pt(l)->is_hanging(hanging_index))
				{
					res = true;
					auto hang_info_pt = node_pt(l)->hanging_pt(hanging_index);
					shape_info->hanginfo_C2TB[l].nummaster = shape_info->hanginfo_C2[l].nummaster = hang_info_pt->nmaster();
					for (unsigned m = 0; m < hang_info_pt->nmaster(); m++)
					{
						shape_info->hanginfo_C2TB[l].masters[m].weight = shape_info->hanginfo_C2[l].masters[m].weight = hang_info_pt->master_weight(m);

						for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk; f++)
						{
							shape_info->hanginfo_C2TB[l].masters[m].local_eqn[f] = shape_info->hanginfo_C2[l].masters[m].local_eqn[f] = this->local_hang_eqn(hang_info_pt->master_node_pt(m), f);
						}
					}
				}
				else
				{
					shape_info->hanginfo_C2TB[l].nummaster = shape_info->hanginfo_C2[l].nummaster = 0;
				}
			}
		}

		if (codeinst->get_func_table()->numfields_C1_basebulk || codeinst->get_func_table()->numfields_C1TB_basebulk)
		{
			for (unsigned int l = 0; l < eleminfo.nnode_C1; l++) // C1 nodes
			{
				unsigned nel = get_node_index_C1_to_element(l);
				if (node_pt(nel)->is_hanging(codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk))
				{
					res = true;
					auto hang_info_pt = node_pt(nel)->hanging_pt(codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk);
					shape_info->hanginfo_C1[l].nummaster = hang_info_pt->nmaster();
					shape_info->hanginfo_C1TB[l].nummaster = hang_info_pt->nmaster();					
					for (unsigned m = 0; m < hang_info_pt->nmaster(); m++)
					{
						shape_info->hanginfo_C1[l].masters[m].weight = hang_info_pt->master_weight(m);
						shape_info->hanginfo_C1TB[l].masters[m].weight = hang_info_pt->master_weight(m);						
						for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_C1_basebulk; f++)
						{
							shape_info->hanginfo_C1[l].masters[m].local_eqn[f + codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk] = this->local_hang_eqn(hang_info_pt->master_node_pt(m), f + codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk);
							shape_info->hanginfo_C1TB[l].masters[m].local_eqn[f + codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk] = this->local_hang_eqn(hang_info_pt->master_node_pt(m), f + codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk);							
						}
					}
				}
				else
				{
					shape_info->hanginfo_C1[l].nummaster = 0;
					shape_info->hanginfo_C1TB[l].nummaster = 0;					
				}
			}
		}
		/*

		if (codeinst->get_func_table()->numfields_DL)
		{
			for (unsigned int l = 0; l < eleminfo.nnode_DL; l++)
			{
				shape_info->hanginfo_Discont[l].nummaster = 0;
			}
		}
		shape_info->hanginfo_Discont[0].nummaster = 0;
		*/
		for (unsigned int l = 0; l < eleminfo.nnode; l++)
		{
			shape_info->hanginfo_Discont[l].nummaster = 0;
		}

		if (eqn_remap)
		{
			return BulkElementBase::fill_hang_info_with_equations(required, shape_info, eqn_remap);
		}
		else
			return res;
	}

	void BulkElementQuad2dC2::interpolate_hang_values()
	{
		BulkElementBase::interpolate_hang_values();

		for (unsigned int l = 0; l < eleminfo.nnode; l++)
		{
			if (node_pt(l)->is_hanging())
			{
				for (unsigned int i = 0; i < node_pt(l)->ndim(); i++)
				{
					for (unsigned t = 0; t < node_pt(l)->ntstorage(); t++)
					{
						dynamic_cast<Node *>(node_pt(l))->variable_position_pt()->set_value(t, i, node_pt(l)->position(t, i));
					}
				}
			}
		}

		int hanging_index = (codeinst->get_func_table()->bulk_position_space_to_C1 ? 0 : -1);

		for (unsigned int l = 0; l < eleminfo.nnode_C2; l++)
		{
			if (node_pt(l)->is_hanging(hanging_index))
			{
				for (unsigned int i = 0; i < codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk; i++)
				{
					for (unsigned t = 0; t < node_pt(l)->ntstorage(); t++)
					{
						node_pt(l)->value_pt(i)[t] = node_pt(l)->value(t, i);
					}
				}
			}
		}
		if (codeinst->get_func_table()->numfields_C1_basebulk || codeinst->get_func_table()->numfields_C1TB_basebulk)
		{
			for (unsigned int l_C1 = 0; l_C1 < eleminfo.nnode_C1; l_C1++)
			{
				unsigned l = get_node_index_C1_to_element(l_C1);
				if (node_pt(l)->is_hanging(codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk))
				{
					// std::cout << "C1 hang" << std::endl;
					for (unsigned int i = codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk; i < codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk + codeinst->get_func_table()->numfields_C1_basebulk+ codeinst->get_func_table()->numfields_C1TB_basebulk; i++)
					{
						for (unsigned t = 0; t < node_pt(l)->ntstorage(); t++)
						{
							node_pt(l)->value_pt(i)[t] = node_pt(l)->value(t, i); // Does this really work here?
						}
					}
				}
			}
			// Now we still need to handle the dummy pinned dofs, which are not considered so far
			for (unsigned int i = codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk; i < codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C1_basebulk+ codeinst->get_func_table()->numfields_C1TB_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk; i++)
			{
				for (unsigned t = 0; t < node_pt(0)->ntstorage(); t++)
				{
					node_pt(1)->value_pt(i)[t] = 0.5 * (node_pt(0)->value(t, i) + node_pt(2)->value(t, i));
					node_pt(3)->value_pt(i)[t] = 0.5 * (node_pt(0)->value(t, i) + node_pt(6)->value(t, i));
					node_pt(5)->value_pt(i)[t] = 0.5 * (node_pt(2)->value(t, i) + node_pt(8)->value(t, i));
					node_pt(7)->value_pt(i)[t] = 0.5 * (node_pt(6)->value(t, i) + node_pt(8)->value(t, i));
					node_pt(4)->value_pt(i)[t] = 0.25 * (node_pt(0)->value(t, i) + node_pt(2)->value(t, i) + node_pt(6)->value(t, i) + node_pt(8)->value(t, i));
				}
			}
		}

		if (codeinst->get_func_table()->bulk_position_space_to_C1)
		{
			for (unsigned int l = 0; l < eleminfo.nnode; l++)
			{
				Node *n = dynamic_cast<Node *>(this->node_pt(l));
				if (!this->is_node_index_part_of_C1(l) && !n->is_on_boundary())
				{
					std::vector<oomph::Node *> sn;
					this->get_supporting_C1_nodes_of_C2_node(l, sn);
					if (sn.empty())
						continue;
					double w = 1.0 / sn.size();
					for (unsigned t = 0; t < node_pt(0)->ntstorage(); t++)
					{
						for (unsigned int i = 0; i < this->nodal_dimension(); i++)
						{
							double res = 0.0;
							for (auto *s : sn)
							{
								res += s->position(t, i);
							}
							dynamic_cast<Node *>(node_pt(l))->variable_position_pt()->set_value(t, i, res * w);
						}
					}
				}
			}
		}
	}

	void BulkElementQuad2dC2::interpolate_hang_values_at_interface()
	{
		auto *functable = codeinst->get_func_table();
		unsigned numC1 = functable->numfields_C1 - functable->numfields_C1_basebulk;
		// TODO: HANGING!
		if (numC1)
		{
			for (unsigned int i = 0; i < numC1; i++)
			{
				std::string fieldname = functable->fieldnames_C1[functable->numfields_C1_basebulk + i];
				unsigned interf_id = codeinst->resolve_interface_dof_id(fieldname);

				std::vector<int> dirs{-1, 1, -2, 2};
				for (int dir : dirs)
				{
					unsigned valindex0 = dynamic_cast<oomph::BoundaryNodeBase *>(this->boundary_node_pt(dir, 0))->index_of_first_value_assigned_by_face_element(interf_id);
					unsigned valindex1 = dynamic_cast<oomph::BoundaryNodeBase *>(this->boundary_node_pt(dir, 1))->index_of_first_value_assigned_by_face_element(interf_id);
					unsigned valindex2 = dynamic_cast<oomph::BoundaryNodeBase *>(this->boundary_node_pt(dir, 2))->index_of_first_value_assigned_by_face_element(interf_id);
					for (unsigned t = 0; t < boundary_node_pt(dir, 1)->ntstorage(); t++)
					{
						boundary_node_pt(dir, 1)->value_pt(valindex1)[t] = 0.5 * (boundary_node_pt(dir, 0)->value_pt(valindex0)[t] + boundary_node_pt(dir, 2)->value_pt(valindex2)[t]);
					}
				}
				// And the central node
				unsigned valindex0 = dynamic_cast<oomph::BoundaryNodeBase *>(this->node_pt(0))->index_of_first_value_assigned_by_face_element(interf_id);
				unsigned valindex2 = dynamic_cast<oomph::BoundaryNodeBase *>(this->node_pt(2))->index_of_first_value_assigned_by_face_element(interf_id);
				unsigned valindex4 = dynamic_cast<oomph::BoundaryNodeBase *>(this->node_pt(4))->index_of_first_value_assigned_by_face_element(interf_id);
				unsigned valindex6 = dynamic_cast<oomph::BoundaryNodeBase *>(this->node_pt(6))->index_of_first_value_assigned_by_face_element(interf_id);
				unsigned valindex8 = dynamic_cast<oomph::BoundaryNodeBase *>(this->node_pt(8))->index_of_first_value_assigned_by_face_element(interf_id);

				for (unsigned t = 0; t < node_pt(4)->ntstorage(); t++)
				{
					node_pt(4)->value_pt(valindex4)[t] = 0.25 * (node_pt(0)->value_pt(valindex0)[t] + node_pt(2)->value_pt(valindex2)[t] + node_pt(6)->value_pt(valindex6)[t] + node_pt(8)->value_pt(valindex8)[t]);
				}
			}
		}
	}

	oomph::Node *BulkElementQuad2dC2::interpolating_node_pt(const unsigned &n, const int &value_id)
	{
		if (value_id >= static_cast<int>(codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk))
		{
			return this->node_pt_C1(n);
		}
		else
		{
			return this->node_pt(n);
		}
	}

	double BulkElementQuad2dC2::local_one_d_fraction_of_interpolating_node(const unsigned &n1d, const unsigned &i, const int &value_id)
	{
		if (value_id >= static_cast<int>(codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk))
		{
			// The C1 nodes are just located on the boundaries at 0 or 1
			return double(n1d);
		}
		else
		{
			return this->local_one_d_fraction_of_node(n1d, i);
		}
	}

	oomph::Node *BulkElementQuad2dC2::get_interpolating_node_at_local_coordinate(const oomph::Vector<double> &s, const int &value_id)
	{
		if (value_id >= static_cast<int>(codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk))
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
			return this->node_pt_C1(total_index);
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
		if (value_id >= static_cast<int>(codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk))
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
		if (value_id >= static_cast<int>(codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk))
		{
			return static_cast<unsigned>(pow(2.0, static_cast<int>(this->dim())));
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

	void BulkElementQuad2dC2::interpolating_basis(const oomph::Vector<double> &s, oomph::Shape &psi, const int &value_id) const
	{
		if (value_id >= static_cast<int>(codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk))
		{
			return this->shape_at_s_C1(s, psi);
		}
		else
		{
			return this->shape(s, psi);
		}
	}

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

	BulkElementTri2dC1::BulkElementTri2dC1(bool has_bubble)
	{
		eleminfo.elem_ptr = this;
		eleminfo.nnode = (has_bubble ? 4 : 3);
		eleminfo.nnode_C1TB = (has_bubble ? 4 : 0);
		eleminfo.nnode_C1 = 3;
		eleminfo.nnode_DL = 3;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}

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

	bool BulkElementTri2dC1::fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap)
	{
		bool res = false;
		for (unsigned int l = 0; l < eleminfo.nnode_C1; l++)
		{
			if (node_pt(l)->is_hanging())
			{
				res = true;
				auto hang_info_pt = node_pt(l)->hanging_pt();
				shape_info->hanginfo_C1[l].nummaster = hang_info_pt->nmaster();
				shape_info->hanginfo_Pos[l].nummaster = hang_info_pt->nmaster();
				for (unsigned m = 0; m < hang_info_pt->nmaster(); m++)
				{
					shape_info->hanginfo_C1[l].masters[m].weight = hang_info_pt->master_weight(m);
					for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_C1_basebulk; f++)
					{
						shape_info->hanginfo_C1[l].masters[m].local_eqn[codeinst->get_func_table()->buffer_offset_C1_basebulk+f] = this->local_hang_eqn(hang_info_pt->master_node_pt(m), f);
					}
					

					for (unsigned int f = 0; f < this->nodal_dimension(); f++)
					{
						//	      oomph::DenseMatrix<int> position_local_eqn_at_node(this->nnodal_position_type(),this->nodal_dimension());
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
				shape_info->hanginfo_C1[l].nummaster = 0;
				shape_info->hanginfo_Pos[l].nummaster = 0;
			}
		}
		/*

		if (codeinst->get_func_table()->numfields_DL)
		{
			for (unsigned int l = 0; l < eleminfo.nnode_DL; l++)
			{
				shape_info->hanginfo_Discont[l].nummaster = 0;
			}
		}
		shape_info->hanginfo_Discont[0].nummaster = 0;
		*/
		for (unsigned int l = 0; l < eleminfo.nnode; l++)
		{
			shape_info->hanginfo_Discont[l].nummaster = 0;
		}

		if (eqn_remap)
		{
			return BulkElementBase::fill_hang_info_with_equations(required, shape_info, eqn_remap);
		}
		else
			return res;
	}

	void BulkElementTri2dC1::interpolate_hang_values()
	{
		BulkElementBase::interpolate_hang_values();
		for (unsigned int l = 0; l < eleminfo.nnode_C1; l++)
		{
			if (node_pt(l)->is_hanging())
			{
				for (unsigned int i = 0; i < node_pt(l)->nvalue(); i++)
				{
					for (unsigned t = 0; t < node_pt(l)->ntstorage(); t++)
					{
						node_pt(l)->value_pt(i)[t] = node_pt(l)->value(t, i);
					}
				}
				for (unsigned int i = 0; i < node_pt(l)->ndim(); i++)
				{
					for (unsigned t = 0; t < node_pt(l)->ntstorage(); t++)
					{
						dynamic_cast<Node *>(node_pt(l))->variable_position_pt()->set_value(t, i, node_pt(l)->position(t, i));
					}
				}
			}
		}
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

	BulkElementTri2dC2::BulkElementTri2dC2(bool with_bubble)
	{
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 6;
		eleminfo.nnode_C2TB = (with_bubble ? 7:6); // Must be done here! DG field allocation would otherwise alloc only 6 for D2TB!
		eleminfo.nnode_C2 = 6;
		eleminfo.nnode_C1TB = (with_bubble ? 4 : 3);		
		eleminfo.nnode_C1 = 3;
		eleminfo.nnode_DL = 3;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}

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

	void BulkElementTri2dC2::constrain_bulk_position_space_to_C1()
	{
		BulkElementBase::constrain_bulk_position_space_to_C1();
		for (unsigned int ni = 0; ni < eleminfo.nnode; ni++)
		{
			for (unsigned int i = 0; i < codeinst->get_func_table()->numfields_C2_basebulk; i++)
			{
				this->node_pt(ni)->set_hanging_pt(NULL, i);
			}
		}
		for (unsigned int i = 0; i < codeinst->get_func_table()->numfields_C2_basebulk; i++)
		{
			// setup_hang_for_value(i); //TODO: Activate this!
		}
	}

	oomph::Node *BulkElementTri2dC2::boundary_node_pt(const int &face_index, const unsigned int i)
	{
		return this->node_pt(this->get_bulk_node_number(face_index, i));
	}


   
	
	bool BulkElementTri2dC2::fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap)
	{
		bool res = false;

		for (unsigned int l = 0; l < eleminfo.nnode; l++) // Pos nodes
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

		int hanging_index = (codeinst->get_func_table()->bulk_position_space_to_C1 ? 0 : -1);
		for (unsigned int l = 0; l < eleminfo.nnode_C2; l++) // C2 nodes
		{
			if (node_pt(l)->is_hanging(hanging_index))
			{
				res = true;
				auto hang_info_pt = node_pt(l)->hanging_pt(hanging_index);
				shape_info->hanginfo_C2[l].nummaster = hang_info_pt->nmaster();
				for (unsigned m = 0; m < hang_info_pt->nmaster(); m++)
				{
					shape_info->hanginfo_C2[l].masters[m].weight = hang_info_pt->master_weight(m);
					for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_C2_basebulk; f++)
					{
						shape_info->hanginfo_C2[l].masters[m].local_eqn[f] = this->local_hang_eqn(hang_info_pt->master_node_pt(m), f);
					}
				}
			}
			else
			{
				shape_info->hanginfo_C2[l].nummaster = 0;
			}
		}

		if (codeinst->get_func_table()->numfields_C1_basebulk)
		{
			for (unsigned int l = 0; l < eleminfo.nnode_C1; l++) // C1 nodes
			{
				unsigned nel = get_node_index_C1_to_element(l);
				if (node_pt(nel)->is_hanging(codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk))
				{
					res = true;
					auto hang_info_pt = node_pt(nel)->hanging_pt(codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk);
					shape_info->hanginfo_C1[l].nummaster = hang_info_pt->nmaster();
					for (unsigned m = 0; m < hang_info_pt->nmaster(); m++)
					{
						shape_info->hanginfo_C1[l].masters[m].weight = hang_info_pt->master_weight(m);
						for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_C1_basebulk; f++)
						{
							shape_info->hanginfo_C1[l].masters[m].local_eqn[f] = this->local_hang_eqn(hang_info_pt->master_node_pt(m), f + codeinst->get_func_table()->numfields_C2_basebulk);
						}
					}
				}
				else
				{
					shape_info->hanginfo_C1[l].nummaster = 0;
				}
			}
		}

		/*
		if (codeinst->get_func_table()->numfields_DL)
		{
			for (unsigned int l = 0; l < eleminfo.nnode_DL; l++)
			{
				shape_info->hanginfo_Discont[l].nummaster = 0;
			}
		}
		shape_info->hanginfo_Discont[0].nummaster = 0;
		*/
		for (unsigned int l = 0; l < eleminfo.nnode; l++)
		{
			shape_info->hanginfo_Discont[l].nummaster = 0;
		}

		if (eqn_remap)
		{
			return BulkElementBase::fill_hang_info_with_equations(required, shape_info, eqn_remap);
		}
		else
			return res;
	}

	void BulkElementTri2dC2::interpolate_hang_values()
	{
		BulkElementBase::interpolate_hang_values();
		for (unsigned int l = 0; l < eleminfo.nnode; l++)
		{
			if (node_pt(l)->is_hanging())
			{
				for (unsigned int i = 0; i < node_pt(l)->ndim(); i++)
				{
					for (unsigned t = 0; t < node_pt(l)->ntstorage(); t++)
					{
						dynamic_cast<Node *>(node_pt(l))->variable_position_pt()->set_value(t, i, node_pt(l)->position(t, i));
					}
				}
			}
		}
		int hanging_index = (codeinst->get_func_table()->bulk_position_space_to_C1 ? 0 : -1);
		for (unsigned int l = 0; l < eleminfo.nnode_C2; l++)
		{
			if (node_pt(l)->is_hanging(hanging_index))
			{
				for (unsigned int i = 0; i < codeinst->get_func_table()->numfields_C2_basebulk; i++)
				{
					for (unsigned t = 0; t < node_pt(l)->ntstorage(); t++)
					{
						node_pt(l)->value_pt(i)[t] = node_pt(l)->value(t, i);
					}
				}
			}
		}
		if (codeinst->get_func_table()->numfields_C1_basebulk || codeinst->get_func_table()->numfields_C1TB_basebulk)
		{
			for (unsigned int l_C1 = 0; l_C1 < eleminfo.nnode_C1; l_C1++)
			{
				unsigned l = get_node_index_C1_to_element(l_C1);
				if (node_pt(l)->is_hanging(codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk))
				{
					// std::cout << "C1 hang" << std::endl;
					for (unsigned int i = codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk; i < codeinst->get_func_table()->numfields_C2TB_basebulk + codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C1_basebulk+codeinst->get_func_table()->numfields_C1TB_basebulk; i++)
					{
						for (unsigned t = 0; t < node_pt(l)->ntstorage(); t++)
						{
							node_pt(l)->value_pt(i)[t] = node_pt(l)->value(t, i); // Does this really work here?
						}
					}
				}
			}
			for (unsigned int i = codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk; i < codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk + codeinst->get_func_table()->numfields_C1_basebulk+codeinst->get_func_table()->numfields_C1TB_basebulk; i++)
			{
				for (unsigned t = 0; t < node_pt(0)->ntstorage(); t++)
				{
					node_pt(3)->value_pt(i)[t] = 0.5 * (node_pt(0)->value(t, i) + node_pt(1)->value(t, i));
					node_pt(4)->value_pt(i)[t] = 0.5 * (node_pt(1)->value(t, i) + node_pt(2)->value(t, i));
					node_pt(5)->value_pt(i)[t] = 0.5 * (node_pt(2)->value(t, i) + node_pt(0)->value(t, i));
				}
			}
		}
	}

	void BulkElementTri2dC2::interpolate_hang_values_at_interface()
	{
		auto *functable = codeinst->get_func_table();
		unsigned numC1 = functable->numfields_C1 - functable->numfields_C1_basebulk;
		// TODO: HANGING!
		if (numC1)
		{
			for (unsigned int i = 0; i < numC1; i++)
			{
				std::string fieldname = functable->fieldnames_C1[functable->numfields_C1_basebulk + i];
				unsigned interf_id = codeinst->resolve_interface_dof_id(fieldname);

				std::vector<int> dirs{0, 1, 2};
				for (int dir : dirs)
				{
					unsigned valindex0 = dynamic_cast<oomph::BoundaryNodeBase *>(this->boundary_node_pt(dir, 0))->index_of_first_value_assigned_by_face_element(interf_id);
					unsigned valindex1 = dynamic_cast<oomph::BoundaryNodeBase *>(this->boundary_node_pt(dir, 1))->index_of_first_value_assigned_by_face_element(interf_id);
					unsigned valindex2 = dynamic_cast<oomph::BoundaryNodeBase *>(this->boundary_node_pt(dir, 2))->index_of_first_value_assigned_by_face_element(interf_id);
					for (unsigned t = 0; t < boundary_node_pt(dir, 1)->ntstorage(); t++)
					{
						boundary_node_pt(dir, 1)->value_pt(valindex1)[t] = 0.5 * (boundary_node_pt(dir, 0)->value_pt(valindex0)[t] + boundary_node_pt(dir, 2)->value_pt(valindex2)[t]);
					}
				}
			}
		}
	}

	void BulkElementTri2dC2::shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		psi[0] = s[0];
		psi[1] = s[1];
		psi[2] = 1.0 - s[0] - s[1];
		/* double s_2=1.0-s[0]-s[1];
		 psi[0] = 2.0*s[0]*(s[0]-0.5);
		 psi[1] = 2.0*s[1]*(s[1]-0.5);
		 psi[2] = 2.0*s_2 *(s_2 -0.5);
		 psi[3] = 4.0*s[0]*s[1];
		 psi[4] = 4.0*s[1]*s_2;
		 psi[5] = 4.0*s_2*s[0];*/
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
	
   BulkElementTri2dC1TB::BulkElementTri2dC1TB()  : BulkElementTri2dC1(true)
   {
		eleminfo.elem_ptr = this;   
      eleminfo.nnode=4;
      eleminfo.nnode_C1=3;
      eleminfo.nnode_C1TB=4;
      eleminfo.nnode_DL=3;
      eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_n_node(eleminfo.nnode);
		this->set_nodal_dimension(eleminfo.nodal_dim);
		this->set_integration_scheme(&Default_enriched_integration_scheme);      
   }
   void BulkElementTri2dC1TB::interpolate_hang_values()
   {
     BulkElementTri2dC1::interpolate_hang_values();
     auto * ft=codeinst->get_func_table();
	  for (unsigned int f = 0; f < ft->numfields_C1; f++)
	  {
	    unsigned i=f+ft->nodal_offset_C1_basebulk;
 		 for (unsigned t = 0; t < node_pt(0)->ntstorage(); t++)
		 {
				node_pt(3)->value_pt(i)[t] = (node_pt(0)->value(t, i) + node_pt(1)->value(t, i) + node_pt(2)->value(t, i)) / 3.0;
			}
     }
   }
   
   bool BulkElementTri2dC1TB::fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap)
   {
     bool res=false;
     for (unsigned int l = 0; l < eleminfo.nnode_C1TB; l++)
		{
			if (node_pt(l)->is_hanging())
			{
				res = true;
				auto hang_info_pt = node_pt(l)->hanging_pt();
				shape_info->hanginfo_C1TB[l].nummaster = hang_info_pt->nmaster();
				for (unsigned m = 0; m < hang_info_pt->nmaster(); m++)
				{
					shape_info->hanginfo_C1TB[l].masters[m].weight = hang_info_pt->master_weight(m);
					for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_C1TB_basebulk; f++)
					{
						shape_info->hanginfo_C1TB[l].masters[m].local_eqn[codeinst->get_func_table()->buffer_offset_C1TB_basebulk+f] = this->local_hang_eqn(hang_info_pt->master_node_pt(m), f);
					}
					for (unsigned int f = 0; f < this->nodal_dimension(); f++)
					{
						//	      oomph::DenseMatrix<int> position_local_eqn_at_node(this->nnodal_position_type(),this->nodal_dimension());
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
				shape_info->hanginfo_C1TB[l].nummaster = 0;
				shape_info->hanginfo_Pos[l].nummaster = 0;
			}
		}
		
     return BulkElementTri2dC1::fill_hang_info_with_equations(required,shape_info, eqn_remap) || res;
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

	BulkElementTri2dC2TB::BulkElementTri2dC2TB() : BulkElementTri2dC2(true)
	{
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 7;
		eleminfo.nnode_C2TB = 7;
		eleminfo.nnode_C2 = 6;
		eleminfo.nnode_C1TB = 4;		
		eleminfo.nnode_C1 = 3;
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
    
   bool BulkElementTri2dC2TB::fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap)
	{
    	bool res=false;	
    	int hanging_index = (codeinst->get_func_table()->bulk_position_space_to_C1 ? 0 : -1);
		for (unsigned int l = 0; l < eleminfo.nnode_C2TB; l++) // C2TB nodes
		{
			if (node_pt(l)->is_hanging(hanging_index))
			{
				res = true;
				auto hang_info_pt = node_pt(l)->hanging_pt(hanging_index);
				shape_info->hanginfo_C2TB[l].nummaster = hang_info_pt->nmaster();
				for (unsigned m = 0; m < hang_info_pt->nmaster(); m++)
				{
					shape_info->hanginfo_C2TB[l].masters[m].weight = hang_info_pt->master_weight(m);
					for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_C2TB_basebulk; f++)
					{
						shape_info->hanginfo_C2TB[l].masters[m].local_eqn[codeinst->get_func_table()->buffer_offset_C2TB_basebulk+f] = this->local_hang_eqn(hang_info_pt->master_node_pt(m), f);
					}
				}
			}
			else
			{
				shape_info->hanginfo_C2TB[l].nummaster = 0;
			}
		}
		
    	hanging_index = (codeinst->get_func_table()->numfields_C2_basebulk +codeinst->get_func_table()->numfields_C2TB_basebulk);		
    	if (codeinst->get_func_table()->numfields_C1TB_basebulk)
    	{
			for (unsigned int li = 0; li < eleminfo.nnode_C1TB; li++) // C1TB nodes
			{
				unsigned int l=this->get_node_index_C1TB_to_element(li);
				if (node_pt(l)->is_hanging(hanging_index))
				{
					res = true;
					auto hang_info_pt = node_pt(l)->hanging_pt(hanging_index);
					shape_info->hanginfo_C1TB[li].nummaster = hang_info_pt->nmaster();
					for (unsigned m = 0; m < hang_info_pt->nmaster(); m++)
					{
						shape_info->hanginfo_C1TB[li].masters[m].weight = hang_info_pt->master_weight(m);
						for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_C1TB_basebulk; f++)
						{
							shape_info->hanginfo_C1TB[li].masters[m].local_eqn[codeinst->get_func_table()->buffer_offset_C1TB_basebulk+f] = this->local_hang_eqn(hang_info_pt->master_node_pt(m), f);
						}
					}
				}
				else
				{
//				   std::cout << "Setting master to 0 at l=" << l << std::endl;
					shape_info->hanginfo_C1TB[li].nummaster = 0;
				}
			}				
		}
	  return  BulkElementTri2dC2::fill_hang_info_with_equations(required,shape_info,eqn_remap)  || res;
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

	void BulkElementTri2dC2TB::interpolate_hang_values()
	{
//	 return;
		BulkElementTri2dC2::interpolate_hang_values();
		// C2 node at the center
		for (unsigned int i = codeinst->get_func_table()->numfields_C2TB_basebulk; i < codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk ; i++)
		{
			for (unsigned t = 0; t < node_pt(0)->ntstorage(); t++)
			{
				node_pt(6)->value_pt(i)[t] = (node_pt(0)->value(t, i) + node_pt(1)->value(t, i) + node_pt(2)->value(t, i) + node_pt(3)->value(t, i) + node_pt(4)->value(t, i) + node_pt(5)->value(t, i)) / 6.0;
			}
		}
		// C1 nodes at the center
		for (unsigned int i = codeinst->get_func_table()->numfields_C2TB_basebulk+codeinst->get_func_table()->numfields_C2_basebulk+codeinst->get_func_table()->numfields_C1TB_basebulk; i < codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk+ codeinst->get_func_table()->numfields_C1TB_basebulk+ codeinst->get_func_table()->numfields_C1_basebulk ; i++)
		{
			for (unsigned t = 0; t < node_pt(0)->ntstorage(); t++)
			{
				node_pt(6)->value_pt(i)[t] = (node_pt(0)->value(t, i) + node_pt(1)->value(t, i) + node_pt(2)->value(t, i) + node_pt(3)->value(t, i) + node_pt(4)->value(t, i) + node_pt(5)->value(t, i)) / 6.0;
			}
		}		
	}

	//////////////////////////////

	BulkElementBrick3dC1::BulkElementBrick3dC1()
	{
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 8;
		eleminfo.nnode_C1 = 8;
		eleminfo.nnode_DL = 4;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}


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


	bool BulkElementBrick3dC1::fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap)
	{
		bool res = false;
		for (unsigned int l = 0; l < eleminfo.nnode_C1; l++)
		{
			if (node_pt(l)->is_hanging())
			{
				res = true;
				auto hang_info_pt = node_pt(l)->hanging_pt();
				shape_info->hanginfo_C1[l].nummaster = hang_info_pt->nmaster();
				shape_info->hanginfo_Pos[l].nummaster = hang_info_pt->nmaster();
				for (unsigned m = 0; m < hang_info_pt->nmaster(); m++)
				{
					shape_info->hanginfo_C1[l].masters[m].weight = hang_info_pt->master_weight(m);
					shape_info->hanginfo_Pos[l].masters[m].weight = hang_info_pt->master_weight(m);
					for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_C1_basebulk; f++)
					{
						shape_info->hanginfo_C1[l].masters[m].local_eqn[f] = this->local_hang_eqn(hang_info_pt->master_node_pt(m), f);
					}
					for (unsigned int f = 0; f < this->nodal_dimension(); f++)
					{
						//	      oomph::DenseMatrix<int> position_local_eqn_at_node(this->nnodal_position_type(),this->nodal_dimension());
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
				shape_info->hanginfo_C1[l].nummaster = 0;
				shape_info->hanginfo_Pos[l].nummaster = 0;
			}
		}

		/*
		if (codeinst->get_func_table()->numfields_DL)
		{
			for (unsigned int l = 0; l < eleminfo.nnode_DL; l++)
			{
				shape_info->hanginfo_Discont[l].nummaster = 0;
			}
		}
		shape_info->hanginfo_Discont[0].nummaster = 0;
		*/

		if (eqn_remap)
		{
			return BulkElementBase::fill_hang_info_with_equations(required, shape_info, eqn_remap);
		}
		else
			return res;
	}

	void BulkElementBrick3dC1::interpolate_hang_values()
	{
		BulkElementBase::interpolate_hang_values();
		for (unsigned int l = 0; l < eleminfo.nnode_C1; l++)
		{
			if (node_pt(l)->is_hanging())
			{
				for (unsigned int i = 0; i < node_pt(l)->nvalue(); i++)
				{
					for (unsigned t = 0; t < node_pt(l)->ntstorage(); t++)
					{
						node_pt(l)->value_pt(i)[t] = node_pt(l)->value(t, i);
					}
				}

				for (unsigned int i = 0; i < node_pt(l)->ndim(); i++)
				{
					for (unsigned t = 0; t < node_pt(l)->ntstorage(); t++)
					{
						dynamic_cast<Node *>(node_pt(l))->variable_position_pt()->set_value(t, i, node_pt(l)->position(t, i));
					}
				}
			}
		}
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

	unsigned int BulkElementBrick3dC2::index_C1_to_element[8] = {0, 2, 6, 8, 18, 20, 24, 26};
	int BulkElementBrick3dC2::element_index_to_C1[27]={0,-1,1,-1,-1,-1,2,-1,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,4,-1,5,-1,-1,-1,6,-1,7};
	bool BulkElementBrick3dC2::node_only_C2[27] = {false, true, false, true, true, true, false, true, false, true, true, true, true, true, true, true, true, true, false, true, false, true, true, true, false, true, false};

	BulkElementBrick3dC2::BulkElementBrick3dC2()
	{
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 27;
		eleminfo.nnode_C1 = 8;
		eleminfo.nnode_C1TB = 8;		
		eleminfo.nnode_C2 = 27;
		eleminfo.nnode_C2TB = 27;
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
		if (codeinst->get_func_table()->numfields_C1_basebulk)
		{
			//   this->setup_hang_for_value(codeinst->get_func_table()->numfields_C2);
			for (unsigned int i = codeinst->get_func_table()->numfields_C2TB_basebulk + codeinst->get_func_table()->numfields_C2_basebulk; i < codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C1_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk; i++)
				this->setup_hang_for_value(i);
			/* 	 for (unsigned int i=codeinst->get_func_table()->numfields_C2+1;i<codeinst->get_func_table()->numfields_C2+codeinst->get_func_table()->numfields_C1;i++)
			   {
					 for (unsigned int l=0;l<eleminfo.nnode_C1;l++)
				 {
			//			this->node_pt_C1(l)->hanging_pt(i)==this->node_pt_C1(l)->hanging_pt(codeinst->get_func_table()->numfields_C2);
			//		std::cout << "HANING " << l << "  " << node_pt_C1(l) << std::endl;
			//		std::cout << "HANING " << l << "  " << node_pt_C1(l)->hanging_pt(codeinst->get_func_table()->numfields_C2) << std::endl;
			//			 this->node_pt_C1(l)->set_hanging_pt(node_pt_C1(l)->hanging_pt(codeinst->get_func_table()->numfields_C2), i); //Copy the Hang-Info for all other C1 values
				 }
			   }
			*/
		}
	}

	bool BulkElementBrick3dC2::fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap)
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

		int hanging_index = (codeinst->get_func_table()->bulk_position_space_to_C1 ? 0 : -1);

		if (codeinst->get_func_table()->numfields_C2_basebulk || codeinst->get_func_table()->numfields_C2TB_basebulk)
		{
			for (unsigned int l = 0; l < eleminfo.nnode_C2; l++) // C2 nodes
			{
				if (node_pt(l)->is_hanging(hanging_index))
				{
					res = true;
					auto hang_info_pt = node_pt(l)->hanging_pt(hanging_index);
					shape_info->hanginfo_C2TB[l].nummaster = shape_info->hanginfo_C2[l].nummaster = hang_info_pt->nmaster();
					for (unsigned m = 0; m < hang_info_pt->nmaster(); m++)
					{
						shape_info->hanginfo_C2TB[l].masters[m].weight = shape_info->hanginfo_C2[l].masters[m].weight = hang_info_pt->master_weight(m);

						for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk; f++)
						{
							shape_info->hanginfo_C2TB[l].masters[m].local_eqn[f] = shape_info->hanginfo_C2[l].masters[m].local_eqn[f] = this->local_hang_eqn(hang_info_pt->master_node_pt(m), f);
						}
					}
				}
				else
				{
					shape_info->hanginfo_C2TB[l].nummaster = shape_info->hanginfo_C2[l].nummaster = 0;
				}
			}
		}

		if (codeinst->get_func_table()->numfields_C1_basebulk || codeinst->get_func_table()->numfields_C1TB_basebulk)
		{
			for (unsigned int l = 0; l < eleminfo.nnode_C1; l++) // C1 nodes
			{
				unsigned nel = get_node_index_C1_to_element(l);
				if (node_pt(nel)->is_hanging(codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk))
				{
					res = true;
					auto hang_info_pt = node_pt(nel)->hanging_pt(codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk);
					shape_info->hanginfo_C1[l].nummaster = hang_info_pt->nmaster();
					shape_info->hanginfo_C1TB[l].nummaster = hang_info_pt->nmaster();					
					for (unsigned m = 0; m < hang_info_pt->nmaster(); m++)
					{
						shape_info->hanginfo_C1[l].masters[m].weight = hang_info_pt->master_weight(m);
						shape_info->hanginfo_C1TB[l].masters[m].weight = hang_info_pt->master_weight(m);						
						for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_C1_basebulk; f++)
						{
							shape_info->hanginfo_C1[l].masters[m].local_eqn[f + codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk] = this->local_hang_eqn(hang_info_pt->master_node_pt(m), f + codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk);
							shape_info->hanginfo_C1TB[l].masters[m].local_eqn[f + codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk] = this->local_hang_eqn(hang_info_pt->master_node_pt(m), f + codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk);							
						}
					}
				}
				else
				{
					shape_info->hanginfo_C1[l].nummaster = 0;
					shape_info->hanginfo_C1TB[l].nummaster = 0;					
				}
			}
		}
		/*

		if (codeinst->get_func_table()->numfields_DL)
		{
			for (unsigned int l = 0; l < eleminfo.nnode_DL; l++)
			{
				shape_info->hanginfo_Discont[l].nummaster = 0;
			}
		}
		shape_info->hanginfo_Discont[0].nummaster = 0;
		*/
		for (unsigned int l = 0; l < eleminfo.nnode; l++)
		{
			shape_info->hanginfo_Discont[l].nummaster = 0;
		}

		if (eqn_remap)
		{
			return BulkElementBase::fill_hang_info_with_equations(required, shape_info, eqn_remap);
		}
		else
			return res;
	}

	void BulkElementBrick3dC2::interpolate_hang_values()
	{
		BulkElementBase::interpolate_hang_values();
		for (unsigned int l = 0; l < eleminfo.nnode; l++)
		{
			if (node_pt(l)->is_hanging())
			{
				for (unsigned int i = 0; i < node_pt(l)->ndim(); i++)
				{
					for (unsigned t = 0; t < node_pt(l)->ntstorage(); t++)
					{
						dynamic_cast<Node *>(node_pt(l))->variable_position_pt()->set_value(t, i, node_pt(l)->position(t, i));
					}
				}
			}
		}
		int hanging_index = (codeinst->get_func_table()->bulk_position_space_to_C1 ? 0 : -1);
		for (unsigned int l = 0; l < eleminfo.nnode_C2; l++)
		{
			if (node_pt(l)->is_hanging(hanging_index))
			{
				for (unsigned int i = 0; i < codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk; i++)
				{
					for (unsigned t = 0; t < node_pt(l)->ntstorage(); t++)
					{
						node_pt(l)->value_pt(i)[t] = node_pt(l)->value(t, i);
					}
				}
			}
		}
		if (codeinst->get_func_table()->numfields_C1_basebulk)
		{
			for (unsigned int l_C1 = 0; l_C1 < eleminfo.nnode_C1; l_C1++)
			{
				unsigned l = get_node_index_C1_to_element(l_C1);
				if (node_pt(l)->is_hanging(codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk))
				{
					// std::cout << "C1 hang" << std::endl;
					for (unsigned int i = codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk; i < codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk + codeinst->get_func_table()->numfields_C1_basebulk; i++)
					{
						for (unsigned t = 0; t < node_pt(l)->ntstorage(); t++)
						{
							node_pt(l)->value_pt(i)[t] = node_pt(l)->value(t, i); // Does this really work here?
						}
					}
				}
			}
			for (unsigned int i = codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk; i < codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C1_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk; i++)
			{
				for (unsigned t = 0; t < node_pt(0)->ntstorage(); t++)
				{
					// Bottom
					unsigned offs = 0;
					node_pt(offs + 1)->value_pt(i)[t] = 0.5 * (node_pt(offs + 0)->value(t, i) + node_pt(offs + 2)->value(t, i));
					node_pt(offs + 3)->value_pt(i)[t] = 0.5 * (node_pt(offs + 0)->value(t, i) + node_pt(offs + 6)->value(t, i));
					node_pt(offs + 5)->value_pt(i)[t] = 0.5 * (node_pt(offs + 2)->value(t, i) + node_pt(offs + 8)->value(t, i));
					node_pt(offs + 7)->value_pt(i)[t] = 0.5 * (node_pt(offs + 6)->value(t, i) + node_pt(offs + 8)->value(t, i));
					node_pt(offs + 4)->value_pt(i)[t] = 0.25 * (node_pt(offs + 0)->value(t, i) + node_pt(offs + 2)->value(t, i) + node_pt(offs + 6)->value(t, i) + node_pt(offs + 8)->value(t, i));

					// Top
					offs = 18;
					node_pt(offs + 1)->value_pt(i)[t] = 0.5 * (node_pt(offs + 0)->value(t, i) + node_pt(offs + 2)->value(t, i));
					node_pt(offs + 3)->value_pt(i)[t] = 0.5 * (node_pt(offs + 0)->value(t, i) + node_pt(offs + 6)->value(t, i));
					node_pt(offs + 5)->value_pt(i)[t] = 0.5 * (node_pt(offs + 2)->value(t, i) + node_pt(offs + 8)->value(t, i));
					node_pt(offs + 7)->value_pt(i)[t] = 0.5 * (node_pt(offs + 6)->value(t, i) + node_pt(offs + 8)->value(t, i));
					node_pt(offs + 4)->value_pt(i)[t] = 0.25 * (node_pt(offs + 0)->value(t, i) + node_pt(offs + 2)->value(t, i) + node_pt(offs + 6)->value(t, i) + node_pt(offs + 8)->value(t, i));

					// Central nodes along vertical edge lines
					offs = 0;
					node_pt(offs + 9)->value_pt(i)[t] = 0.5 * (node_pt(offs + 0)->value(t, i) + node_pt(offs + 18)->value(t, i));
					offs = 2;
					node_pt(offs + 9)->value_pt(i)[t] = 0.5 * (node_pt(offs + 0)->value(t, i) + node_pt(offs + 18)->value(t, i));
					offs = 6;
					node_pt(offs + 9)->value_pt(i)[t] = 0.5 * (node_pt(offs + 0)->value(t, i) + node_pt(offs + 18)->value(t, i));
					offs = 8;
					node_pt(offs + 9)->value_pt(i)[t] = 0.5 * (node_pt(offs + 0)->value(t, i) + node_pt(offs + 18)->value(t, i));

					node_pt(10)->value_pt(i)[t] = 0.25 * (node_pt(0)->value(t, i) + node_pt(2)->value(t, i) + node_pt(18)->value(t, i) + node_pt(20)->value(t, i));
					node_pt(12)->value_pt(i)[t] = 0.25 * (node_pt(0)->value(t, i) + node_pt(6)->value(t, i) + node_pt(18)->value(t, i) + node_pt(24)->value(t, i));
					node_pt(16)->value_pt(i)[t] = 0.25 * (node_pt(6)->value(t, i) + node_pt(26)->value(t, i) + node_pt(8)->value(t, i) + node_pt(24)->value(t, i));
					node_pt(14)->value_pt(i)[t] = 0.25 * (node_pt(2)->value(t, i) + node_pt(8)->value(t, i) + node_pt(20)->value(t, i) + node_pt(26)->value(t, i));

					node_pt(13)->value_pt(i)[t] = 0.125 * (node_pt(0)->value(t, i) + node_pt(2)->value(t, i) + node_pt(6)->value(t, i) + node_pt(8)->value(t, i) + node_pt(18)->value(t, i) + node_pt(20)->value(t, i) + node_pt(24)->value(t, i) + node_pt(26)->value(t, i));
				}
			}
		}
	}

	oomph::Node *BulkElementBrick3dC2::interpolating_node_pt(const unsigned &n, const int &value_id)
	{
		if (value_id >= static_cast<int>(codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk))
		{
			return this->node_pt_C1(n);
		}
		else
		{
			return this->node_pt(n);
		}
	}

	double BulkElementBrick3dC2::local_one_d_fraction_of_interpolating_node(const unsigned &n1d, const unsigned &i, const int &value_id)
	{
		if (value_id >= static_cast<int>(codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk))
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
		if (value_id >= static_cast<int>(codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk))
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
			return this->node_pt_C1(total_index);
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
		if (value_id >= static_cast<int>(codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk))
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
		if (value_id >= static_cast<int>(codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk))
		{
			return static_cast<unsigned>(pow(2.0, static_cast<int>(this->dim())));
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
		if (value_id >= static_cast<int>(codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk))
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
		eleminfo.nnode_C1 = 4;
		eleminfo.nnode_DL = 4;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}

	bool BulkElementTetra3dC1::fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap)
	{
		bool res = false;
		for (unsigned int l = 0; l < eleminfo.nnode_C1; l++)
		{
			if (node_pt(l)->is_hanging())
			{
				res = true;
				auto hang_info_pt = node_pt(l)->hanging_pt();
				shape_info->hanginfo_C1[l].nummaster = hang_info_pt->nmaster();
				shape_info->hanginfo_Pos[l].nummaster = hang_info_pt->nmaster();
				for (unsigned m = 0; m < hang_info_pt->nmaster(); m++)
				{
					shape_info->hanginfo_C1[l].masters[m].weight = hang_info_pt->master_weight(m);
					shape_info->hanginfo_Pos[l].masters[m].weight = hang_info_pt->master_weight(m);
					for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_C1_basebulk; f++)
					{
						shape_info->hanginfo_C1[l].masters[m].local_eqn[f] = this->local_hang_eqn(hang_info_pt->master_node_pt(m), f);
					}
					for (unsigned int f = 0; f < this->nodal_dimension(); f++)
					{
						//	      oomph::DenseMatrix<int> position_local_eqn_at_node(this->nnodal_position_type(),this->nodal_dimension());
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
				shape_info->hanginfo_C1[l].nummaster = 0;
				shape_info->hanginfo_Pos[l].nummaster = 0;
			}
		}

		/*
		if (codeinst->get_func_table()->numfields_DL)
		{
			for (unsigned int l = 0; l < eleminfo.nnode_DL; l++)
			{
				shape_info->hanginfo_Discont[l].nummaster = 0;
			}
		}
		shape_info->hanginfo_Discont[0].nummaster = 0;
		*/
		for (unsigned int l = 0; l < eleminfo.nnode; l++)
		{
			shape_info->hanginfo_Discont[l].nummaster = 0;
		}

		if (eqn_remap)
		{
			return BulkElementBase::fill_hang_info_with_equations(required, shape_info, eqn_remap);
		}
		else
			return res;
	}

	void BulkElementTetra3dC1::interpolate_hang_values()
	{
		BulkElementBase::interpolate_hang_values();
		for (unsigned int l = 0; l < eleminfo.nnode_C1; l++)
		{
			if (node_pt(l)->is_hanging())
			{
				for (unsigned int i = 0; i < node_pt(l)->nvalue(); i++)
				{
					for (unsigned t = 0; t < node_pt(l)->ntstorage(); t++)
					{
						node_pt(l)->value_pt(i)[t] = node_pt(l)->value(t, i);
					}
				}

				for (unsigned int i = 0; i < node_pt(l)->ndim(); i++)
				{
					for (unsigned t = 0; t < node_pt(l)->ntstorage(); t++)
					{
						dynamic_cast<Node *>(node_pt(l))->variable_position_pt()->set_value(t, i, node_pt(l)->position(t, i));
					}
				}
			}
		}
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

	unsigned int BulkElementTetra3dC2::index_C1_to_element[4] = {0, 1, 2, 3};
	int BulkElementTetra3dC2::element_index_to_C1[15]={0,1,2,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
	bool BulkElementTetra3dC2::node_only_C2[15] = {false, false, false, false, true, true, true, true, true, true, true, true, true, true, true};

	BulkElementTetra3dC2::BulkElementTetra3dC2(bool has_bubble)
	{
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 10;
		eleminfo.nnode_C1 = 4;
		eleminfo.nnode_C2TB = (has_bubble ? 15 : 10);
		eleminfo.nnode_C1TB = (has_bubble ? 5 : 4);		
		eleminfo.nnode_C2 = 10;
		eleminfo.nnode_DL = 4;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_nodal_dimension(eleminfo.nodal_dim);
		allocate_discontinous_fields();
	}

   BulkElementBase *BulkElementTetra3dC2::create_son_instance() const	
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementTetra3dC2(dynamic_cast<const BulkElementTetra3dC2TB*>(this)!=nullptr);
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }


	void BulkElementTetra3dC2::further_setup_hanging_nodes()
	{

		BulkElementBase::further_setup_hanging_nodes();
		if (codeinst->get_func_table()->numfields_C1_basebulk)
		{
			for (unsigned int i = codeinst->get_func_table()->numfields_C2_basebulk; i < codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C1_basebulk; i++)
				this->setup_hang_for_value(i);
		}
	}

	bool BulkElementTetra3dC2::fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap)
	{
		bool res = false;
		int hanging_index = (codeinst->get_func_table()->bulk_position_space_to_C1 ? 0 : -1);
		for (unsigned int l = 0; l < eleminfo.nnode_C2; l++) // C2 nodes
		{
			if (node_pt(l)->is_hanging(hanging_index))
			{
				res = true;
				auto hang_info_pt = node_pt(l)->hanging_pt(hanging_index);
				shape_info->hanginfo_C2[l].nummaster = hang_info_pt->nmaster();
				shape_info->hanginfo_Pos[l].nummaster = hang_info_pt->nmaster();
				for (unsigned m = 0; m < hang_info_pt->nmaster(); m++)
				{
					shape_info->hanginfo_C2[l].masters[m].weight = hang_info_pt->master_weight(m);
					for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_C2_basebulk; f++)
					{
						shape_info->hanginfo_C2[l].masters[m].local_eqn[f] = this->local_hang_eqn(hang_info_pt->master_node_pt(m), f);
					}

					for (unsigned int f = 0; f < this->nodal_dimension(); f++)
					{
						//	      oomph::DenseMatrix<int> position_local_eqn_at_node(this->nnodal_position_type(),this->nodal_dimension());
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
				shape_info->hanginfo_C2[l].nummaster = 0;
				shape_info->hanginfo_Pos[l].nummaster = 0;
			}
		}

		if (codeinst->get_func_table()->numfields_C1_basebulk)
		{
			for (unsigned int l = 0; l < eleminfo.nnode_C1; l++) // C1 nodes
			{
				unsigned nel = get_node_index_C1_to_element(l);
				if (node_pt(nel)->is_hanging(codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk))
				{
					res = true;
					auto hang_info_pt = node_pt(nel)->hanging_pt(codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk);
					shape_info->hanginfo_C1[l].nummaster = hang_info_pt->nmaster();
					for (unsigned m = 0; m < hang_info_pt->nmaster(); m++)
					{
						shape_info->hanginfo_C1[l].masters[m].weight = hang_info_pt->master_weight(m);
						for (unsigned int f = 0; f < codeinst->get_func_table()->numfields_C1_basebulk; f++)
						{
							shape_info->hanginfo_C1[l].masters[m].local_eqn[f + codeinst->get_func_table()->numfields_C2_basebulk] = this->local_hang_eqn(hang_info_pt->master_node_pt(m), f + codeinst->get_func_table()->numfields_C2_basebulk);
						}
					}
				}
				else
				{
					shape_info->hanginfo_C1[l].nummaster = 0;
				}
			}
		}
		/*
		if (codeinst->get_func_table()->numfields_DL)
		{
			for (unsigned int l = 0; l < eleminfo.nnode_DL; l++)
			{
				shape_info->hanginfo_Discont[l].nummaster = 0;
			}
		}
		shape_info->hanginfo_Discont[0].nummaster = 0;
		*/
		for (unsigned int l = 0; l < eleminfo.nnode; l++)
		{
			shape_info->hanginfo_Discont[l].nummaster = 0;
		}

		if (eqn_remap)
		{
			return BulkElementBase::fill_hang_info_with_equations(required, shape_info, eqn_remap);
		}
		else
			return res;
	}

	void BulkElementTetra3dC2::interpolate_hang_values()
	{
		BulkElementBase::interpolate_hang_values();
		for (unsigned int l = 0; l < eleminfo.nnode; l++)
		{
			if (node_pt(l)->is_hanging())
			{
				for (unsigned int i = 0; i < node_pt(l)->ndim(); i++)
				{
					for (unsigned t = 0; t < node_pt(l)->ntstorage(); t++)
					{
						dynamic_cast<Node *>(node_pt(l))->variable_position_pt()->set_value(t, i, node_pt(l)->position(t, i));
					}
				}
			}
		}
		int hanging_index = (codeinst->get_func_table()->bulk_position_space_to_C1 ? 0 : -1);
		for (unsigned int l = 0; l < eleminfo.nnode_C2; l++)
		{
			if (node_pt(l)->is_hanging(hanging_index))
			{
				for (unsigned int i = 0; i < codeinst->get_func_table()->numfields_C2_basebulk; i++)
				{
					for (unsigned t = 0; t < node_pt(l)->ntstorage(); t++)
					{
						node_pt(l)->value_pt(i)[t] = node_pt(l)->value(t, i);
					}
				}
			}
		}
		if (codeinst->get_func_table()->numfields_C1_basebulk)
		{
			for (unsigned int l_C1 = 0; l_C1 < eleminfo.nnode_C1; l_C1++)
			{
				unsigned l = get_node_index_C1_to_element(l_C1);
				if (node_pt(l)->is_hanging(codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk))
				{
					// std::cout << "C1 hang" << std::endl;
					for (unsigned int i = codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk; i < codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk + codeinst->get_func_table()->numfields_C1_basebulk; i++)
					{
						for (unsigned t = 0; t < node_pt(l)->ntstorage(); t++)
						{
							node_pt(l)->value_pt(i)[t] = node_pt(l)->value(t, i); // Does this really work here?
						}
					}
				}
			}
			for (unsigned int i = codeinst->get_func_table()->numfields_C2_basebulk; i < codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C1_basebulk; i++)
			{
				for (unsigned t = 0; t < node_pt(0)->ntstorage(); t++)
				{
					node_pt(4)->value_pt(i)[t] = 0.5 * (node_pt(0)->value(t, i) + node_pt(1)->value(t, i));
					node_pt(5)->value_pt(i)[t] = 0.5 * (node_pt(0)->value(t, i) + node_pt(2)->value(t, i));
					node_pt(6)->value_pt(i)[t] = 0.5 * (node_pt(0)->value(t, i) + node_pt(3)->value(t, i));
					node_pt(7)->value_pt(i)[t] = 0.5 * (node_pt(1)->value(t, i) + node_pt(2)->value(t, i));
					node_pt(8)->value_pt(i)[t] = 0.5 * (node_pt(2)->value(t, i) + node_pt(3)->value(t, i));
					node_pt(9)->value_pt(i)[t] = 0.5 * (node_pt(1)->value(t, i) + node_pt(3)->value(t, i));
				}
			}
		}
	}

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

	void BulkElementTetra3dC2::interpolating_basis(const oomph::Vector<double> &s, oomph::Shape &psi, const int &value_id) const
	{
		if (value_id >= static_cast<int>(codeinst->get_func_table()->numfields_C2_basebulk))
		{
			return this->shape_at_s_C1(s, psi);
		}
		else
		{
			return this->shape(s, psi);
		}
	}

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

	std::vector<double> BulkElementTetra3dC2::get_outline(bool lagrangian)
	{
		std::vector<double> res(10 * this->nodal_dimension());
		throw_runtime_error("Outline not implemented for 3d");
		return res;
	}

	///////////////////////////////

	BulkElementTetra3dC2TB::BulkElementTetra3dC2TB() : BulkElementTetra3dC2(true)
	{
		eleminfo.elem_ptr = this;
		eleminfo.nnode = 15;
		eleminfo.nnode_C1 = 4;
		eleminfo.nnode_C1TB = 5;		
		eleminfo.nnode_C2TB = 15;
		eleminfo.nnode_C2 = 10;
		eleminfo.nnode_DL = 4;
		eleminfo.nodal_dim = codeinst->get_func_table()->nodal_dim;
		this->set_n_node(eleminfo.nnode);
		this->set_nodal_dimension(eleminfo.nodal_dim);
		this->set_integration_scheme(&Default_enriched_integration_scheme);
	}

	void BulkElementTetra3dC2TB::interpolate_hang_values()
	{
		BulkElementTetra3dC2::interpolate_hang_values();
		for (unsigned int i = codeinst->get_func_table()->numfields_C2TB_basebulk; i < codeinst->get_func_table()->numfields_C2_basebulk + codeinst->get_func_table()->numfields_C2TB_basebulk; i++)
		{
			for (unsigned t = 0; t < node_pt(0)->ntstorage(); t++)
			{
				node_pt(10)->value_pt(i)[t] = (node_pt(0)->value(t, i) + node_pt(1)->value(t, i) + node_pt(3)->value(t, i)) / 3.0;
				node_pt(11)->value_pt(i)[t] = (node_pt(0)->value(t, i) + node_pt(1)->value(t, i) + node_pt(2)->value(t, i)) / 3.0;
				node_pt(12)->value_pt(i)[t] = (node_pt(0)->value(t, i) + node_pt(2)->value(t, i) + node_pt(3)->value(t, i)) / 3.0;
				node_pt(13)->value_pt(i)[t] = (node_pt(1)->value(t, i) + node_pt(2)->value(t, i) + node_pt(3)->value(t, i)) / 3.0;

				node_pt(14)->value_pt(i)[t] = (node_pt(0)->value(t, i) + node_pt(1)->value(t, i) + node_pt(2)->value(t, i) + node_pt(3)->value(t, i)) / 4.0; // TODO Possibly also consider C2 contribs?
			}
		}
	}

	///////////////////////////////

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



	unsigned InterfaceElementBase::get_C2TB_buffer_index(const unsigned &fieldindex)
    {
		auto * ft=this->get_code_instance()->get_func_table();
	if (fieldindex<ft->numfields_C2TB_basebulk)
	{
          return ft->buffer_offset_C2TB_basebulk+ fieldindex;
        }
        else
        {
         return  ft->buffer_offset_C2TB_interf +(fieldindex-ft->buffer_offset_C2TB_basebulk);
        }
    }

	unsigned InterfaceElementBase::get_C2_buffer_index(const unsigned &fieldindex)
    {
		auto * ft=this->get_code_instance()->get_func_table();
	if (fieldindex<ft->numfields_C2_basebulk)
	{
          return ft->buffer_offset_C2_basebulk+ fieldindex;
        }
        else
        {
         return  ft->buffer_offset_C2_interf +(fieldindex-ft->buffer_offset_C2_basebulk);
        }    
        }
        

	unsigned InterfaceElementBase::get_C1TB_buffer_index(const unsigned &fieldindex)
    {
		auto * ft=this->get_code_instance()->get_func_table();
	if (fieldindex<ft->numfields_C1TB_basebulk)
	{
          return ft->buffer_offset_C1TB_basebulk+ fieldindex;
        }
        else
        {
         return  ft->buffer_offset_C1TB_interf +(fieldindex-ft->buffer_offset_C1TB_basebulk);
        }
    }        

	unsigned InterfaceElementBase::get_C1_buffer_index(const unsigned &fieldindex)
    {
			auto * ft=this->get_code_instance()->get_func_table();
	if (fieldindex<ft->numfields_C1_basebulk)
	{
          return ft->buffer_offset_C1_basebulk+ fieldindex;
        }
        else
        {
         return  ft->buffer_offset_C1_interf +(fieldindex-ft->buffer_offset_C1_basebulk);
        }
    }

	unsigned InterfaceElementBase::get_D2TB_buffer_index(const unsigned &fieldindex)
    {
         auto * ft=this->get_code_instance()->get_func_table();
	if (fieldindex<ft->numfields_D2TB_basebulk)
	{
          return ft->buffer_offset_D2TB_basebulk+ fieldindex;
        }
        else
        {
         return  ft->buffer_offset_D2TB_interf +(fieldindex-ft->numfields_D2TB_basebulk);
        }
    }

	unsigned InterfaceElementBase::get_D2_buffer_index(const unsigned &fieldindex)
    {
	auto * ft=this->get_code_instance()->get_func_table();
	if (fieldindex<ft->numfields_D2_basebulk)
	{
          return ft->buffer_offset_D2_basebulk+ fieldindex;
        }
        else
        {
         return  ft->buffer_offset_D2_interf +(fieldindex-ft->numfields_D2_basebulk);
        }
     }

	unsigned InterfaceElementBase::get_D1TB_buffer_index(const unsigned &fieldindex)
    {
         auto * ft=this->get_code_instance()->get_func_table();
	if (fieldindex<ft->numfields_D1TB_basebulk)
	{
          return ft->buffer_offset_D1TB_basebulk+ fieldindex;
        }
        else
        {
         return  ft->buffer_offset_D1TB_interf +(fieldindex-ft->numfields_D1TB_basebulk);
        }
    }

	unsigned InterfaceElementBase::get_D1_buffer_index(const unsigned &fieldindex)
    {
			auto * ft=this->get_code_instance()->get_func_table();
	if (fieldindex<ft->numfields_D1_basebulk)
	{
          return ft->buffer_offset_D1_basebulk+ fieldindex;
        }
        else
        {
         return  ft->buffer_offset_D1_interf +(fieldindex-ft->numfields_D1_basebulk);
        }
    }
    
    unsigned InterfaceElementBase::get_D2TB_node_index(const unsigned &fieldindex,const unsigned &nodeindex) const 
	{
		auto * ft=this->codeinst->get_func_table();
		if (fieldindex>=ft->numfields_D2TB_bulk) 	return nodeindex;
		else
		{
			int pnodeindex=this->get_node_index_C2TB_to_element(nodeindex);
			if (pnodeindex<0) throw_runtime_error("Strange");
			pnodeindex=this->bulk_node_number(pnodeindex);			
			BulkElementBase* be=dynamic_cast<BulkElementBase*>(this->bulk_element_pt());
			return be->get_D2TB_node_index(fieldindex,be->get_node_index_element_to_C2TB(pnodeindex));
		}
	}
    unsigned InterfaceElementBase::get_D2_node_index(const unsigned &fieldindex,const unsigned &nodeindex) const 
	{
		auto * ft=this->codeinst->get_func_table();
		if (fieldindex>=ft->numfields_D2_bulk) 	return nodeindex;
		else
		{
			int pnodeindex=this->get_node_index_C2_to_element(nodeindex);
			if (pnodeindex<0) throw_runtime_error("Strange");
			pnodeindex=this->bulk_node_number(pnodeindex);
			BulkElementBase* be=dynamic_cast<BulkElementBase*>(this->bulk_element_pt());
			return be->get_D2_node_index(fieldindex,be->get_node_index_element_to_C2(pnodeindex));
		}
	}

  unsigned InterfaceElementBase::get_D1TB_node_index(const unsigned &fieldindex,const unsigned &nodeindex) const 
	{
		auto * ft=this->codeinst->get_func_table();
		if (fieldindex>=ft->numfields_D1TB_bulk) 	return nodeindex;
		else
		{
			int pnodeindex=this->get_node_index_C1TB_to_element(nodeindex);
			if (pnodeindex<0) throw_runtime_error("Strange");
			pnodeindex=this->bulk_node_number(pnodeindex);			
			BulkElementBase* be=dynamic_cast<BulkElementBase*>(this->bulk_element_pt());
			return be->get_D1TB_node_index(fieldindex,be->get_node_index_element_to_C1TB(pnodeindex));
		}
	}
	
    unsigned InterfaceElementBase::get_D1_node_index(const unsigned &fieldindex,const unsigned &nodeindex) const 
	{
		auto * ft=this->codeinst->get_func_table();
		if (fieldindex>=ft->numfields_D1_bulk) 	return nodeindex;
		else
		{			
			int pnodeindex=this->get_node_index_C1_to_element(nodeindex);
			if (pnodeindex<0) throw_runtime_error("Strange");
			pnodeindex=this->bulk_node_number(pnodeindex);
			BulkElementBase* be=dynamic_cast<BulkElementBase*>(this->bulk_element_pt());
			return be->get_D1_node_index(fieldindex,be->get_node_index_element_to_C1(pnodeindex));
		}		
	}

    oomph::Data * InterfaceElementBase::get_D1_nodal_data(const unsigned & fieldindex )
	{
		auto * ft=this->codeinst->get_func_table();
		if (fieldindex>=ft->numfields_D1_bulk) return this->internal_data_pt(ft->internal_offset_D1_new+(fieldindex-ft->numfields_D1_bulk));
		else
		{
			return this->external_data_pt(ft->external_offset_D1_bulk +fieldindex);			
		}
	}
    oomph::Data * InterfaceElementBase::get_D2_nodal_data(const unsigned & fieldindex )
	{
		auto * ft=this->codeinst->get_func_table();
		if (fieldindex>=ft->numfields_D2_bulk) 
		{
			//std::cout << "RETURNING INT DATA " << ft->internal_offset_D2_new+(fieldindex-ft->numfields_D2_bulk) << std::endl;
			return this->internal_data_pt(ft->internal_offset_D2_new+(fieldindex-ft->numfields_D2_bulk));
		}
		else
		{
			if (ft->external_offset_D2_bulk +fieldindex>=this->nexternal_data()) throw_runtime_error("External data for discontinuous fields not well allocated");
			return this->external_data_pt(ft->external_offset_D2_bulk +fieldindex);			
		}
	}
    oomph::Data * InterfaceElementBase::get_D2TB_nodal_data(const unsigned & fieldindex )
	{
		auto * ft=this->codeinst->get_func_table();
		if (fieldindex>=ft->numfields_D2TB_bulk) return this->internal_data_pt(ft->internal_offset_D2TB_new+(fieldindex-ft->numfields_D2TB_bulk));
		else
		{
			return this->external_data_pt(ft->external_offset_D2TB_bulk +fieldindex);			
		}
	}
	
   oomph::Data * InterfaceElementBase::get_D1TB_nodal_data(const unsigned & fieldindex )
	{
		auto * ft=this->codeinst->get_func_table();
		if (fieldindex>=ft->numfields_D1TB_bulk) return this->internal_data_pt(ft->internal_offset_D1TB_new+(fieldindex-ft->numfields_D1TB_bulk));
		else
		{
			return this->external_data_pt(ft->external_offset_D1TB_bulk +fieldindex);			
		}
	}
		
    int InterfaceElementBase::get_D2TB_local_equation(const unsigned &fieldindex,const unsigned & nodeindex)
	{
		auto * ft=this->codeinst->get_func_table();
		if (fieldindex>=ft->numfields_D2TB_bulk) return this->internal_local_eqn(ft->internal_offset_D2TB_new+(fieldindex-ft->numfields_D2TB_bulk),nodeindex);
		else
		{
			return this->external_local_eqn(ft->external_offset_D2TB_bulk +fieldindex,this->get_D2TB_node_index(fieldindex,nodeindex));			
		}
	}
	
  int InterfaceElementBase::get_D1TB_local_equation(const unsigned &fieldindex,const unsigned & nodeindex)
	{
		auto * ft=this->codeinst->get_func_table();
		if (fieldindex>=ft->numfields_D1TB_bulk) return this->internal_local_eqn(ft->internal_offset_D1TB_new+(fieldindex-ft->numfields_D1TB_bulk),nodeindex);
		else
		{
			return this->external_local_eqn(ft->external_offset_D1TB_bulk +fieldindex,this->get_D1TB_node_index(fieldindex,nodeindex));			
		}
	}	
    int InterfaceElementBase::get_D2_local_equation(const unsigned &fieldindex,const unsigned & nodeindex)
	{
		auto * ft=this->codeinst->get_func_table();
		if (fieldindex>=ft->numfields_D2_bulk) return this->internal_local_eqn(ft->internal_offset_D2_new+(fieldindex-ft->numfields_D2_bulk),nodeindex);
		else
		{
			return this->external_local_eqn(ft->external_offset_D2_bulk +fieldindex,this->get_D2_node_index(fieldindex,nodeindex));			
		}
	}
    int InterfaceElementBase::get_D1_local_equation(const unsigned &fieldindex,const unsigned & nodeindex)
	{
		auto * ft=this->codeinst->get_func_table();
		if (fieldindex>=ft->numfields_D1_bulk) return this->internal_local_eqn(ft->internal_offset_D1_new+(fieldindex-ft->numfields_D1_bulk),nodeindex);
		else
		{
			return this->external_local_eqn(ft->external_offset_D1_bulk +fieldindex,this->get_D1_node_index(fieldindex,nodeindex));			
		}
	}

   std::vector<int> InterfaceElementBase::get_attached_element_equation_mapping(const std::string & which)
   {
    if (which=="bulk") return bulk_eqn_map;
    else if (which=="opposite_interface") return  opp_interf_eqn_map;
    else if (which=="opposite_bulk") return opp_bulk_eqn_map;
    else if (which=="bulk_bulk") return bulk_bulk_eqn_map;
    else throw_runtime_error("Unknown map "+which);
   }
   
	int InterfaceElementBase::get_nodal_index_by_name(oomph::Node *n, std::string fieldname)
	{
		int bres = BulkElementBase::get_nodal_index_by_name(n, fieldname);
		if (bres >= 0)
			return bres;
		// Interface fields
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();

		for (unsigned int j = 0; j < functable->numfields_C2TB - functable->numfields_C2TB_basebulk; j++)
		{
			std::string intername = functable->fieldnames_C2TB[functable->numfields_C2TB_basebulk + j];
			if (intername == fieldname)
			{
				unsigned interf_id = codeinst->resolve_interface_dof_id(intername);
				return dynamic_cast<oomph::BoundaryNodeBase *>(n)->index_of_first_value_assigned_by_face_element(interf_id);
			}
		}


		for (unsigned int j = 0; j < functable->numfields_C2 - functable->numfields_C2_basebulk; j++)
		{
			std::string intername = functable->fieldnames_C2[functable->numfields_C2_basebulk + j];
			if (intername == fieldname)
			{
				unsigned interf_id = codeinst->resolve_interface_dof_id(intername);
				return dynamic_cast<oomph::BoundaryNodeBase *>(n)->index_of_first_value_assigned_by_face_element(interf_id);
			}
		}


		for (unsigned int j = 0; j < functable->numfields_C1TB - functable->numfields_C1TB_basebulk; j++)
		{
			std::string intername = functable->fieldnames_C1[functable->numfields_C1TB_basebulk + j];
			if (intername == fieldname)
			{
				unsigned interf_id = codeinst->resolve_interface_dof_id(intername);
				return dynamic_cast<oomph::BoundaryNodeBase *>(n)->index_of_first_value_assigned_by_face_element(interf_id);
			}
		}

		for (unsigned int j = 0; j < functable->numfields_C1 - functable->numfields_C1_basebulk; j++)
		{
			std::string intername = functable->fieldnames_C1[functable->numfields_C1_basebulk + j];
			if (intername == fieldname)
			{
				unsigned interf_id = codeinst->resolve_interface_dof_id(intername);
				return dynamic_cast<oomph::BoundaryNodeBase *>(n)->index_of_first_value_assigned_by_face_element(interf_id);
			}
		}


		return -1;
	}

	void InterfaceElementBase::fill_element_info_interface_part(bool without_equations)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		// Obtain the offset where the interface additional contiuous fields start

		std::vector<unsigned> interface_ids_C2TB(functable->numfields_C2TB - functable->numfields_C2TB_basebulk);
		for (unsigned int j = 0; j < functable->numfields_C2TB - functable->numfields_C2TB_basebulk; j++)
		{
			std::string fieldname = functable->fieldnames_C2TB[functable->numfields_C2TB_basebulk + j];
			unsigned interf_id = codeinst->resolve_interface_dof_id(fieldname);
			interface_ids_C2TB[j] = interf_id;
		}

		for (unsigned int i = 0; i < eleminfo.nnode_C2TB; i++)
		{
			unsigned i_el = this->get_node_index_C2TB_to_element(i);
			for (unsigned int j = 0; j < functable->numfields_C2TB - functable->numfields_C2TB_basebulk; j++)
			{
				unsigned node_index = j + functable->buffer_offset_C2TB_interf; // TODO: This index right?
				unsigned interf_id = interface_ids_C2TB[j];
				unsigned valindex = dynamic_cast<oomph::BoundaryNodeBase *>(this->node_pt(i_el))->index_of_first_value_assigned_by_face_element(interf_id);
				// std::cout << "NOO " << i << "  " << node_index << "  " << i_el << "   " << valindex << std::endl;
				eleminfo.nodal_data[i][node_index] = node_pt(i_el)->value_pt(valindex);
				if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] = this->nodal_local_eqn(i_el, valindex);
			}
		}

		std::vector<unsigned> interface_ids_C2(functable->numfields_C2 - functable->numfields_C2_basebulk);
		for (unsigned int j = 0; j < functable->numfields_C2 - functable->numfields_C2_basebulk; j++)
		{
			std::string fieldname = functable->fieldnames_C2[functable->numfields_C2_basebulk + j];
			unsigned interf_id = codeinst->resolve_interface_dof_id(fieldname);
			interface_ids_C2[j] = interf_id;
		}

		for (unsigned int i = 0; i < eleminfo.nnode_C2; i++)
		{
			unsigned i_el = this->get_node_index_C2_to_element(i);
			for (unsigned int j = 0; j < functable->numfields_C2 - functable->numfields_C2_basebulk; j++)
			{
				unsigned node_index = j + functable->buffer_offset_C2_interf; // TODO: This index right?
				unsigned interf_id = interface_ids_C2[j];
				unsigned valindex = dynamic_cast<oomph::BoundaryNodeBase *>(this->node_pt(i_el))->index_of_first_value_assigned_by_face_element(interf_id);
				// std::cout << "NOO " << i << "  " << node_index << "  " << i_el << "   " << valindex << std::endl;
				eleminfo.nodal_data[i][node_index] = node_pt(i_el)->value_pt(valindex);
				if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] = this->nodal_local_eqn(i_el, valindex);
			}
		}

	   std::vector<unsigned> interface_ids_C1TB(functable->numfields_C1TB - functable->numfields_C1TB_basebulk);
		for (unsigned int j = 0; j < functable->numfields_C1TB - functable->numfields_C1TB_basebulk; j++)
		{
			std::string fieldname = functable->fieldnames_C1TB[functable->numfields_C1TB_basebulk + j];
			unsigned interf_id = codeinst->resolve_interface_dof_id(fieldname);
			interface_ids_C1TB[j] = interf_id;
		}
		if (functable->numfields_C1TB)
		{
		  for (unsigned int i = 0; i < eleminfo.nnode_C1TB; i++)
		  {
			  unsigned i_el = this->get_node_index_C1TB_to_element(i);
			  for (unsigned int j = 0; j < functable->numfields_C1TB - functable->numfields_C1TB_basebulk; j++)
			  {
				  unsigned node_index = j + functable->buffer_offset_C1TB_interf;
				  unsigned interf_id = interface_ids_C1TB[j];
				  unsigned valindex = dynamic_cast<oomph::BoundaryNodeBase *>(this->node_pt(i_el))->index_of_first_value_assigned_by_face_element(interf_id);
				  eleminfo.nodal_data[i][node_index] = node_pt(i_el)->value_pt(valindex);
				  if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] = this->nodal_local_eqn(i_el, valindex);
			  }
		  }
		}


		std::vector<unsigned> interface_ids_C1(functable->numfields_C1 - functable->numfields_C1_basebulk);
		for (unsigned int j = 0; j < functable->numfields_C1 - functable->numfields_C1_basebulk; j++)
		{
			std::string fieldname = functable->fieldnames_C1[functable->numfields_C1_basebulk + j];
			unsigned interf_id = codeinst->resolve_interface_dof_id(fieldname);
			interface_ids_C1[j] = interf_id;
		}
		for (unsigned int i = 0; i < eleminfo.nnode_C1; i++)
		{
			unsigned i_el = this->get_node_index_C1_to_element(i);
			for (unsigned int j = 0; j < functable->numfields_C1 - functable->numfields_C1_basebulk; j++)
			{
				unsigned node_index = j + functable->buffer_offset_C1_interf;
				unsigned interf_id = interface_ids_C1[j];
				unsigned valindex = dynamic_cast<oomph::BoundaryNodeBase *>(this->node_pt(i_el))->index_of_first_value_assigned_by_face_element(interf_id);
				eleminfo.nodal_data[i][node_index] = node_pt(i_el)->value_pt(valindex);
				if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] = this->nodal_local_eqn(i_el, valindex);
			}
		}

		// DG fields
		for (unsigned int j=0;j<functable->numfields_D2TB-functable->numfields_D2TB_basebulk;j++)
		{
			unsigned node_index = j + functable->buffer_offset_D2TB_interf;
			oomph::Data * data=this->get_D2TB_nodal_data(functable->numfields_D2TB_basebulk+j);
			for (unsigned int i=0;i<eleminfo.nnode_C2TB;i++)
			{
				unsigned valindex=this->get_D2TB_node_index(functable->numfields_D2TB_basebulk+j,i);
				eleminfo.nodal_data[i][node_index] = data->value_pt(valindex);
				if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] = this->get_D2TB_local_equation(functable->numfields_D2TB_basebulk+j, i);
			}
		}
		for (unsigned int j=0;j<functable->numfields_D2-functable->numfields_D2_basebulk;j++)
		{
			unsigned node_index = j + functable->buffer_offset_D2_interf;
			oomph::Data * data=this->get_D2_nodal_data(functable->numfields_D2_basebulk+j);
			for (unsigned int i=0;i<eleminfo.nnode_C2;i++)
			{
				unsigned valindex=this->get_D2_node_index(functable->numfields_D2_basebulk+j,i);
				eleminfo.nodal_data[i][node_index] = data->value_pt(valindex);
				if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] = this->get_D2_local_equation(functable->numfields_D2_basebulk+j, i);
			}
		}
		for (unsigned int j=0;j<functable->numfields_D1TB-functable->numfields_D1TB_basebulk;j++)
		{
			unsigned node_index = j + functable->buffer_offset_D1TB_interf;
			oomph::Data * data=this->get_D1TB_nodal_data(functable->numfields_D1TB_basebulk+j);
			for (unsigned int i=0;i<eleminfo.nnode_C1TB;i++)
			{
				unsigned valindex=this->get_D1TB_node_index(functable->numfields_D1TB_basebulk+j,i);
				eleminfo.nodal_data[i][node_index] = data->value_pt(valindex);
				if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] = this->get_D1TB_local_equation(functable->numfields_D1TB_basebulk+j, i);
			}
		}		
		for (unsigned int j=0;j<functable->numfields_D1-functable->numfields_D1_basebulk;j++)
		{
			unsigned node_index = j + functable->buffer_offset_D1_interf;
			oomph::Data * data=this->get_D1_nodal_data(functable->numfields_D1_basebulk+j);
			for (unsigned int i=0;i<eleminfo.nnode_C1;i++)
			{
				unsigned valindex=this->get_D1_node_index(functable->numfields_D1_basebulk+j,i);
				eleminfo.nodal_data[i][node_index] = data->value_pt(valindex);
				if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] = this->get_D1_local_equation(functable->numfields_D1_basebulk+j, i);
			}
		}


		for (unsigned int i = 0; i < eleminfo.nnode_DL; i++)
		{
			for (unsigned int j = 0; j < functable->numfields_DL; j++)
			{
				unsigned node_index = j + functable->buffer_offset_DL;
				eleminfo.nodal_data[i][node_index] = internal_data_pt(functable->internal_offset_DL + j)->value_pt(i);
				if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] = this->internal_local_eqn(functable->internal_offset_DL + j, i);
			}
		}

		//	if (functable->numfields_D0)
		//	{
		// throw_runtime_error("TODO: D0 interface fields "+std::to_string(local_field_offset));
		for (unsigned int j = 0; j < functable->numfields_D0; j++)
		{
			unsigned node_index = j + functable->buffer_offset_D0;
			eleminfo.nodal_data[0][node_index] = internal_data_pt(functable->internal_offset_D0 + j)->value_pt(0);
			if (!without_equations) eleminfo.nodal_local_eqn[0][node_index] = this->internal_local_eqn(functable->internal_offset_D0 + j, 0);
		}
		//	}
		/* ///NOTE: EXT DATA SHOULD BE ALWAYS AT THE END AT THE MOMENT
		local_field_offset+=functable->numfields_D0;
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
		for (unsigned i = ft->numfields_C2TB_bulk; i < ft->numfields_C2TB; i++)
		{
			std::string fieldname = ft->fieldnames_C2TB[i];
			unsigned value_index = codeinst->resolve_interface_dof_id(fieldname);
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
				  if (additional_data_values[l] && !already_allocated[l] && interpolate_new_interface_dofs) this->interpolate_newly_constructed_additional_dof(l,value_index,"C2TB");
				}				
			}
		}

		for (unsigned i = ft->numfields_C2_bulk; i < ft->numfields_C2; i++)
		{
			std::string fieldname = ft->fieldnames_C2[i];
			unsigned value_index = codeinst->resolve_interface_dof_id(fieldname);
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
				  if (additional_data_values[l && !already_allocated[l]] && interpolate_new_interface_dofs) this->interpolate_newly_constructed_additional_dof(l,value_index,"C2");
				}				
			}
		}

		for (unsigned i = ft->numfields_C1_bulk; i < ft->numfields_C1; i++)
		{
			std::string fieldname = ft->fieldnames_C1[i];
			unsigned value_index = codeinst->resolve_interface_dof_id(fieldname);
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
				  if (additional_data_values[l] && !already_allocated[l] && interpolate_new_interface_dofs) this->interpolate_newly_constructed_additional_dof(l,value_index,"C1");
				}
			}
		}
	}

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
				  psi.resize(father->get_eleminfo()->nnode_C1);
		  		  father->shape_at_s_C1(sfather, psi);				 
		  		  for (unsigned int lf=0;lf<psi.nindex1();lf++) 
		  		  {
		  		   if (abs(psi[lf])>1e-9)
		  		   {
		  		    unsigned fnode_index=father->get_node_index_C1_to_element(lf);
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
				  psi.resize(father->get_eleminfo()->nnode_C1TB);
		  		  father->shape_at_s_C1TB(sfather, psi);				 
		  		  for (unsigned int lf=0;lf<psi.nindex1();lf++) 
		  		  {
		  		   if (abs(psi[lf])>1e-9)
		  		   {
		  		    unsigned fnode_index=father->get_node_index_C1TB_to_element(lf);
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
				  psi.resize(father->get_eleminfo()->nnode_C2);
		  		  father->shape_at_s_C2(sfather, psi);				 
		  		  for (unsigned int lf=0;lf<psi.nindex1();lf++) 
		  		  {
		  		   if (abs(psi[lf])>1e-9)
		  		   {
		  		    unsigned fnode_index=father->get_node_index_C2_to_element(lf);
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
				  psi.resize(father->get_eleminfo()->nnode_C2TB);
		  		  father->shape_at_s_C2TB(sfather, psi);				 
		  		  for (unsigned int lf=0;lf<psi.nindex1();lf++) 
		  		  {
		  		   if (abs(psi[lf])>1e-9)
		  		   {
		  		    unsigned fnode_index=father->get_node_index_C2TB_to_element(lf);
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
   
	void InterfaceElementBase::update_in_external_fd(const unsigned &i)
	{
		this->interpolate_hang_values();
	}

	bool InterfaceElementBase::add_required_ext_data(oomph::Data *data, bool is_geometric)
	{
		for (unsigned int k = 0; k < this->nnode(); k++)
		{
			if (data == this->node_pt(k))
			{
			//	std::cout << "  ALREADY PART OF THE ELEMENT AT NODE INDEX " << k << std::endl;		
				return true;
			}
		}; // Nodes can be the same
		if (dynamic_cast<pyoomph::Node *>(data))
		{
			for (unsigned int j=0;j<this->nnode();j++)	
			{
				auto *nod_pt = dynamic_cast<pyoomph::Node *>(this->node_pt(j));
				if (nod_pt->is_hanging())
				{
					oomph::HangInfo *const hang_pt = nod_pt->hanging_pt();
					const unsigned nmaster = hang_pt->nmaster();
					for (unsigned m = 0; m < nmaster; m++)
					{
						auto *const master_nod_pt = dynamic_cast<pyoomph::Node *>(hang_pt->master_node_pt(m));
						if (data==master_nod_pt) return true;
					}
				}
			}
			auto *fft=dynamic_cast<BulkElementBase*>(this)->get_code_instance()->get_func_table();
			if (fft->numfields_C1 || fft->numfields_C1TB)
			{
				int hang_index = (this->get_eleminfo()->nnode_C2 ? fft->numfields_C2TB_basebulk+fft->numfields_C2_basebulk : -1);
				for (unsigned int j = 0; j < this->get_eleminfo()->nnode_C1; j++)
				{
					auto *nod_pt = this->node_pt(this->get_node_index_C1_to_element(j));
					if (nod_pt->is_hanging(hang_index))
					{
							oomph::HangInfo *const hang_pt = nod_pt->hanging_pt(hang_index);
							const unsigned nmaster = hang_pt->nmaster();
							for (unsigned m = 0; m < nmaster; m++)
							{
								auto *const master_nod_pt = dynamic_cast<pyoomph::Node *>(hang_pt->master_node_pt(m));
								if (data==master_nod_pt) return true;
							}
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

	void InterfaceElementBase::add_DG_external_data()
	{	  
      auto *ft=this->codeinst->get_func_table();
	  BulkElementBase * blk=dynamic_cast<BulkElementBase *>(this->bulk_element_pt());
	  for (unsigned i=0;i<ft->numfields_D2TB_bulk;i++)
	  {
		this->add_external_data(blk->get_D2TB_nodal_data(i));
	  }
	  for (unsigned i=0;i<ft->numfields_D2_bulk;i++)
	  {
		this->add_external_data(blk->get_D2_nodal_data(i));
	  }
	  for (unsigned i=0;i<ft->numfields_D1TB_bulk;i++)
	  {
		this->add_external_data(blk->get_D1TB_nodal_data(i));
	  }	  
	  for (unsigned i=0;i<ft->numfields_D1_bulk;i++)
	  {
		this->add_external_data(blk->get_D1_nodal_data(i));
	  }
	}

	void InterfaceElementBase::add_required_external_data(JITFuncSpec_RequiredShapes_FiniteElement_t *required, BulkElementBase *from_elem)
	{
		external_data_is_geometric.resize(this->nexternal_data(), false); // Fill with the ED0 fields
		// std::cout << "EX DA " << this->nexternal_data() << std::endl;
		DynamicBulkElementInstance *fcodeinst = from_elem->get_code_instance();
		auto *fft = fcodeinst->get_func_table();		

		if (fft->moving_nodes)
		{
			if (required->dx_psi_C2TB || required->psi_C2TB || required->dX_psi_C2TB || required->dx_psi_C2 || required->psi_C2 || required->dX_psi_C2 || required->dx_psi_C1 || required->psi_C1TB || required->dx_psi_C1TB || required->dX_psi_C1TB || required->psi_C1 || required->dX_psi_C1 || required->psi_Pos || required->psi_DL || required->dx_psi_DL || required->dX_psi_DL)
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

		int hanging_index = (fft->bulk_position_space_to_C1 ? 0 : -1);
		if (required->dx_psi_C2TB || required->psi_C2TB || required->dX_psi_C2TB)
		{
			for (unsigned int j = 0; j < from_elem->get_eleminfo()->nnode_C2TB; j++)
			{
				auto *nod_pt = from_elem->node_pt(from_elem->get_node_index_C2TB_to_element(j));
		//		std::cout << "ADDING C2TB EXTERNAL " << j << " NODE " << nod_pt << " " << this << "  " << from_elem <<  std::endl;
				/*		for(unsigned i=0;i<nod_pt->nvalue();i++)
				 *	{*/
				if (nod_pt->is_hanging(hanging_index))
				{
			//	std::cout << "   HANGING " << j << " " << this << "  " << from_elem <<  std::endl;				
					//				std::cout << "HHHHHHHHHHHHHHHHHAAAAAAAAAAAANG " << i << "  FROM ELEM " << from_elem<< std::endl;
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
			// DG fields (from bulk they are already external data, but from opposite interfaces and bulk, they are not)
			for (unsigned int fiDG=0;fiDG<fft->numfields_D2TB;fiDG++)
			{
			  this->add_required_ext_data(from_elem->get_D2TB_nodal_data(fiDG),false);
			}			
		}
		if (required->dx_psi_C2 || required->psi_C2 || required->dX_psi_C2)
		{
			for (unsigned int j = 0; j < from_elem->get_eleminfo()->nnode_C2; j++)
			{
				auto *nod_pt = from_elem->node_pt(from_elem->get_node_index_C2_to_element(j));
				/*		for(unsigned i=0;i<nod_pt->nvalue();i++)
				 *	{*/
				if (nod_pt->is_hanging(hanging_index))
				{
					//				std::cout << "HHHHHHHHHHHHHHHHHAAAAAAAAAAAANG " << i << "  FROM ELEM " << from_elem<< std::endl;
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
				//		}
			}
			// DG fields (from bulk they are already external data, but from opposite interfaces and bulk, they are not)
			for (unsigned int fiDG=0;fiDG<fft->numfields_D2;fiDG++)
			{
			  this->add_required_ext_data(from_elem->get_D2_nodal_data(fiDG),false);
			}							
		}


		if (required->dx_psi_C1TB || required->psi_C1TB || required->dX_psi_C1TB) // C1 < C2, so nothing to do
		{
			int hang_index = (from_elem->get_eleminfo()->nnode_C2 ? fft->nodal_offset_C1_basebulk : -1); // Hangs also like C1!
			// std::cout << "   HANG INDEX " <<  hang_index << "  NNODE C1TB " << from_elem->get_eleminfo()->nnode_C1TB << std::endl;
//						std::cout << "REQ AND FROM ELEM HAs " << from_elem->get_eleminfo()->nnode_C1TB << std::endl;			
			for (unsigned int j = 0; j < from_elem->get_eleminfo()->nnode_C1TB; j++)
			{
				auto *nod_pt = from_elem->node_pt(from_elem->get_node_index_C1TB_to_element(j));
				//	std::cout << "      MAPPOING " << j << "  " << from_elem->get_node_index_C1_to_element(j) << "  HANING " << nod_pt->is_hanging(hang_index) << std::endl;
				if (nod_pt->is_hanging(hang_index))
				{
					oomph::HangInfo *const hang_pt = nod_pt->hanging_pt(hang_index);
					const unsigned nmaster = hang_pt->nmaster();
					for (unsigned m = 0; m < nmaster; m++)
					{
						auto *const master_nod_pt = hang_pt->master_node_pt(m);
						this->add_required_ext_data(master_nod_pt, false);
//						std::cout << "ADDING HANGING C1TB NODE " << j << std::endl;
					}
				}
				else
				{
					//			   std::cout << "    ADDING EXTERNAL NODE " << nod_pt << std::endl;
					this->add_required_ext_data(nod_pt, false);
//						std::cout << "ADDING NONHANGING C1TB NODE " << j << std::endl;					
				}
			}
			// DG fields (from bulk they are already external data, but from opposite interfaces and bulk, they are not)
			for (unsigned int fiDG=0;fiDG<fft->numfields_D1TB;fiDG++)
			{
			  this->add_required_ext_data(from_elem->get_D1TB_nodal_data(fiDG),false);
			}				
		}
		
		// std::cout << " CODE " << codeinst->get_code()->get_file_name() << "  DEP ON " << fcodeinst->get_code()->get_file_name() << "  REQ C1 " << required->psi_C1 << std::endl;
		if (required->dx_psi_C1 || required->psi_C1 || required->dX_psi_C1) // C1 < C2, so nothing to do
		{
			int hang_index = (from_elem->get_eleminfo()->nnode_C2 ? fft->nodal_offset_C1_basebulk : -1);
			// std::cout << "   HANG INDEX " <<  hang_index << "  NNODE C1 " << from_elem->get_eleminfo()->nnode_C1 << std::endl;
			for (unsigned int j = 0; j < from_elem->get_eleminfo()->nnode_C1; j++)
			{
				auto *nod_pt = from_elem->node_pt(from_elem->get_node_index_C1_to_element(j));
				//	std::cout << "      MAPPOING " << j << "  " << from_elem->get_node_index_C1_to_element(j) << "  HANING " << nod_pt->is_hanging(hang_index) << std::endl;
				if (nod_pt->is_hanging(hang_index))
				{
					oomph::HangInfo *const hang_pt = nod_pt->hanging_pt(hang_index);
					const unsigned nmaster = hang_pt->nmaster();
					for (unsigned m = 0; m < nmaster; m++)
					{
						auto *const master_nod_pt = hang_pt->master_node_pt(m);
						this->add_required_ext_data(master_nod_pt, false);
					}
				}
				else
				{
					//			   std::cout << "    ADDING EXTERNAL NODE " << nod_pt << std::endl;
					this->add_required_ext_data(nod_pt, false);
				}
			}
			// DG fields (from bulk they are already external data, but from opposite interfaces and bulk, they are not)
			for (unsigned int fiDG=0;fiDG<fft->numfields_D1;fiDG++)
			{
			  this->add_required_ext_data(from_elem->get_D1_nodal_data(fiDG),false);
			}				
		}

		// std::cout << " AT REQ " << codeinst->get_code()->get_file_name() << " FROM " << fcodeinst->get_code()->get_file_name() << " USE DL " << (required->psi_DL || required->dx_psi_DL || required->dX_psi_DL) << std::endl;
		if (required->psi_DL || required->dx_psi_DL || required->dX_psi_DL)
		{

			for (unsigned int j = 0; j < fft->numfields_DL; j++)
			{
				auto *id_pt = from_elem->internal_data_pt(fft->internal_offset_DL+j);
				this->add_required_ext_data(id_pt, false);
			}
		}

		if (required->psi_D0)
		{
			for (unsigned int j = 0; j < fft->numfields_D0; j++)
			{
				auto *id_pt = from_elem->internal_data_pt(fft->internal_offset_D0 + j);
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

	int InterfaceElementBase::resolve_local_equation_for_external_contributions(long int globeq, BulkElementBase *from_elem, std::string *info)
	{
		if (globeq < 0)
			return -1;
		for (unsigned iloc = 0; iloc < this->ndof(); iloc++)
		{
			long int iglob = this->eqn_number(iloc);
			if (iglob == globeq)
				return iloc;
		}

		{
			std::ostringstream oss;
			oss << "CANNOT RESOLVE EXTERNAL GLOBAL EQUATION NUMBER " << globeq << " in " << codeinst->get_code()->get_file_name();
			if (from_elem)
				oss << " FROM ELEM " << from_elem << " which is in domain " << from_elem->get_code_instance()->get_code()->get_file_name();
			if (info)
				oss << "INFOSTR: " << (*info);
			oss << "THE ELEMENT ITSELF " << this << " HAS THE " << this->ndof() << " EQUATIONS " << std::endl;
			auto dofnames = this->get_dof_names();
			for (unsigned iloc = 0; iloc < this->ndof(); iloc++)
			{
				long int iglob = this->eqn_number(iloc);
				oss << "   " << iloc << "  " << iglob << "  " << dofnames[iloc] << std::endl;
			}
			dofnames = from_elem->get_dof_names();
			oss << "THE SOURCE ELEMENT " << from_elem << " HAS THE " << from_elem->ndof() << " EQUATIONS " << std::endl;
			for (unsigned iloc = 0; iloc < from_elem->ndof(); iloc++)
			{
				long int iglob = from_elem->eqn_number(iloc);
				oss << "   " << iloc << "  " << iglob << "  " << dofnames[iloc] << std::endl;
			}
			throw_runtime_error(oss.str());
		}
		return -1;
	}

	double InterfaceElementBase::fill_shape_info_at_s(const oomph::Vector<double> &s, const unsigned int &index, const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, double &JLagr, unsigned int flag, oomph::DenseMatrix<double> *dxds) const
	{
		BulkElementBase::fill_shape_info_at_s(s, index, required, shape_info, JLagr, flag, dxds);

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

		return this->J_eulerian(s);
	}

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

	bool InterfaceElementBase::fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap)
	{
		//	bool ret_self=BulkElementBase::fill_hang_info_with_equations(required,shape_info,eqn_remap);
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

	std::vector<std::string> InterfaceElementBase::get_dof_names(bool not_a_root_call)
	{
		// const JITFuncSpec_Table_FiniteElement_t * functable=codeinst->get_func_table();
		std::vector<std::string> res = BulkElementBase::get_dof_names(not_a_root_call);

		const JITFuncSpec_Table_FiniteElement_t *functable = this->codeinst->get_func_table();
		for (unsigned int i = 0; i < eleminfo.nnode_C2TB; i++)
		{
			for (unsigned int j = 0; j < functable->numfields_C2TB - functable->numfields_C2TB_basebulk; j++)
			{
				unsigned node_index = j + functable->buffer_offset_C2TB_interf; // TODO: This index right?
				int leq = eleminfo.nodal_local_eqn[i][node_index];
				if (leq >= 0 && res[leq] == "<unknown>")
				{
					res[leq] = "IFIELD_" + std::string(functable->fieldnames_C2TB[functable->numfields_C2TB_basebulk + j]) + "__C2__" + std::to_string(i); // TODO: Interhangs?
				}
			}
		}

		for (unsigned int i = 0; i < eleminfo.nnode_C2; i++)
		{
			for (unsigned int j = 0; j < functable->numfields_C2 - functable->numfields_C2_basebulk; j++)
			{
				unsigned node_index = j + functable->buffer_offset_C2_interf; // TODO: This index right?
				int leq = eleminfo.nodal_local_eqn[i][node_index];
				if (leq >= 0 && res[leq] == "<unknown>")
				{
					res[leq] = "IFIELD_" + std::string(functable->fieldnames_C2[functable->numfields_C2_basebulk + j]) + "__C2__" + std::to_string(i); // TODO: Interhangs?
				}
			}
		}
		for (unsigned int i = 0; i < eleminfo.nnode_C1TB; i++)
		{
			for (unsigned int j = 0; j < functable->numfields_C1TB - functable->numfields_C1TB_basebulk; j++)
			{
				unsigned node_index = j + functable->buffer_offset_C1TB_interf;
				int leq = eleminfo.nodal_local_eqn[i][node_index];
				if (leq >= 0 && res[leq] == "<unknown>")
				{
					res[leq] = "IFIELD_" + std::string(functable->fieldnames_C1TB[functable->numfields_C1TB_basebulk + j]) + "__C1TB__" + std::to_string(i); // TODO: Interhangs?
				}
			}
		}

		
		for (unsigned int i = 0; i < eleminfo.nnode_C1; i++)
		{
			for (unsigned int j = 0; j < functable->numfields_C1 - functable->numfields_C1_basebulk; j++)
			{
				unsigned node_index = j + functable->buffer_offset_C1_interf;
				int leq = eleminfo.nodal_local_eqn[i][node_index];
				if (leq >= 0 && res[leq] == "<unknown>")
				{
					res[leq] = "IFIELD_" + std::string(functable->fieldnames_C1[functable->numfields_C1_basebulk + j]) + "__C1__" + std::to_string(i); // TODO: Interhangs?
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

	void InterfaceElementBase::unpin_dummy_values() // C1 fields on C2 elements have dummy values on only C2 nodes, which needs to be pinned
	{
		// return;
		BulkElementBase::unpin_dummy_values();

		for (unsigned int l = 0; l < nnode(); l++)
		{
			for (unsigned int i = 0; i < node_pt(l)->nvalue(); i++)
				node_pt(l)->unpin(i); // After that, the BCs are applied to repin what is necessary
		}
	}

	void InterfaceElementBase::pin_dummy_values()
	{
		// return;
		BulkElementBase::pin_dummy_values();
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();

		// TODO: Alloc these in advance
		std::vector<unsigned> interface_ids_C1(functable->numfields_C1 - functable->numfields_C1_basebulk);
		for (unsigned int j = 0; j < functable->numfields_C1 - functable->numfields_C1_basebulk; j++)
		{
			std::string fieldname = functable->fieldnames_C1[functable->numfields_C1_basebulk + j];
			unsigned interf_id = codeinst->resolve_interface_dof_id(fieldname);
			interface_ids_C1[j] = interf_id;
		}
		std::vector<unsigned> interface_ids_C2(functable->numfields_C2 - functable->numfields_C2_basebulk);
		for (unsigned int j = 0; j < functable->numfields_C2 - functable->numfields_C2_basebulk; j++)
		{
			std::string fieldname = functable->fieldnames_C2[functable->numfields_C2_basebulk + j];
			unsigned interf_id = codeinst->resolve_interface_dof_id(fieldname);
			interface_ids_C2[j] = interf_id;
		}


		for (unsigned int i = 0; i < eleminfo.nnode; i++)
		{
			if (!this->is_node_index_part_of_C1(i))
			{
				for (unsigned int j = 0; j < functable->numfields_C1 - functable->numfields_C1_basebulk; j++)
				{
					unsigned interf_id = interface_ids_C1[j];
					unsigned valindex = dynamic_cast<oomph::BoundaryNodeBase *>(this->node_pt(i))->index_of_first_value_assigned_by_face_element(interf_id);
					this->node_pt(i)->pin(valindex); // Constrained
				}
				if (!this->is_node_index_part_of_C2(i))
				{
					for (unsigned int j = 0; j < functable->numfields_C2 - functable->numfields_C2_basebulk; j++)
					{
						unsigned interf_id = interface_ids_C2[j];
						unsigned valindex = dynamic_cast<oomph::BoundaryNodeBase *>(this->node_pt(i))->index_of_first_value_assigned_by_face_element(interf_id);
						this->node_pt(i)->pin(valindex); // Constrained
					}
				}
			}
		}
	}

	void InterfaceElementBase::assign_additional_local_eqn_numbers_from_elem(const JITFuncSpec_RequiredShapes_FiniteElement_t *required, BulkElementBase *from_elem, std::vector<int> &eq_map)
	{
		eq_map.clear();

		if (required)
		{
			if (!from_elem)
			{
				throw_runtime_error("Trying to assign required local_eqn_numbers from an element which is not set");
			}

			if (!from_elem->ndof())
			{
				from_elem->assign_local_eqn_numbers(true);
			}

			eq_map.resize(from_elem->ndof(), -666); // Magic for not used/found
			const JITFuncSpec_Table_FiniteElement_t *functable = from_elem->get_code_instance()->get_func_table();
			if (functable->moving_nodes)
			{
				for (unsigned int l = 0; l < from_elem->get_eleminfo()->nnode; l++)
				{
					auto *n = from_elem->node_pt(l);
					auto *vp = dynamic_cast<pyoomph::Node *>(n)->variable_position_pt();
					if (n->is_hanging())
					{
						oomph::HangInfo *const hang_pt = n->hanging_pt();
						const unsigned nmaster = hang_pt->nmaster();
						for (unsigned m = 0; m < nmaster; m++)
						{
							oomph::Node *const master_node_pt = hang_pt->master_node_pt(m);
							oomph::DenseMatrix<int> Position_local_eqn_at_node = from_elem->local_position_hang_eqn(master_node_pt);
							unsigned n_position_type = 1;
							for (unsigned k = 0; k < n_position_type; k++)
							{
								for (unsigned i = 0; i < vp->nvalue(); i++)
								{
									int parent_no = Position_local_eqn_at_node(k, i);
									if (parent_no >= 0)
									{
										//				         std::cout << "HANG VAR POS " << l << "  " << k << "  " << "  " << m << "  " <<i <<  "   " << dynamic_cast<pyoomph::Node*>(master_node_pt)->variable_position_pt()->eqn_number(i) << "  " << parent_no << std::endl;
										std::string info = "VARIABLE POSITION HANG";
										int my_no = resolve_local_equation_for_external_contributions(dynamic_cast<pyoomph::Node *>(master_node_pt)->variable_position_pt()->eqn_number(i), from_elem, &info);
										eq_map[parent_no] = my_no;
									}
								}
							}
						}

						//			 	throw_runtime_error("Hanging bulk Lagrange");
					}
					else
					{
						for (unsigned int k = 0; k < vp->nvalue(); k++)
						{
							int parent_no = from_elem->position_local_eqn(l, 0, k);
							//				   std::cout << "FROM ELEM " << from_elem << "( nnode " << from_elem->nnode() << ") NONHANG VAR POS " << l << "  " << k << "  " << "  " <<  "   " << vp->eqn_number(k) << std::endl;
							std::string info = "VARIABLE POSITION";
							int my_no = resolve_local_equation_for_external_contributions(vp->eqn_number(k), from_elem, &info);
							//			  	 	std::cout << "DONE" << std::endl;
							if (parent_no >= 0)
							{
								eq_map[parent_no] = my_no;
							}
						}
					}
				}
				/*      }
						 else
						 {
								throw_runtime_error("Adding also bulk Lagrange for C1");
						 }
				*/
			}

			int hanging_index = (functable->bulk_position_space_to_C1 ? 0 : -1);

			if (required->dx_psi_C2TB || required->psi_C2TB || required->dX_psi_C2TB)
			{
				for (unsigned int j = 0; j < from_elem->get_eleminfo()->nnode_C2TB; j++)
				{
				   unsigned el_n_index=from_elem->get_node_index_C2TB_to_element(j);
					auto *n = from_elem->node_pt(el_n_index);
					//		  	 for (unsigned int k=0;k<n->nvalue();k++)
					for (unsigned int k = 0; k < functable->numfields_C2TB_basebulk; k++)
					{
						if (n->is_hanging(hanging_index))
						{
							//						std::cout << "SETTING HANHG " << k << "  FROM ELEM " << from_elem<< std::endl;
							oomph::HangInfo *const hang_pt = n->hanging_pt(hanging_index);
							const unsigned nmaster = hang_pt->nmaster();
							for (unsigned m = 0; m < nmaster; m++)
							{
								auto *const master_nod_pt = hang_pt->master_node_pt(m);
								int parent_no = from_elem->local_hang_eqn(master_nod_pt, functable->nodal_offset_C2TB_basebulk+k);
								//			  	 	std::cout << "HANG C2 " << j << "  " << "  " << k << "  " << m << master_nod_pt->eqn_number(k) << std::endl;
								std::string info = "C2TB HANGIG";
								int my_no = resolve_local_equation_for_external_contributions(master_nod_pt->eqn_number(functable->nodal_offset_C2TB_basebulk+k), from_elem, &info);
								if (parent_no >= 0)
								{
									eq_map[parent_no] = my_no;
								}
							}
						}
						else
						{
							int parent_no = from_elem->nodal_local_eqn(el_n_index, functable->nodal_offset_C2TB_basebulk+k);
							//  		  	 	   std::cout << "C2 " << j << "  " << "  " << k << "  " << n->eqn_number(k) << std::endl;
							std::string info = "C2TB";
							int my_no = resolve_local_equation_for_external_contributions(n->eqn_number(functable->nodal_offset_C2TB_basebulk+k), from_elem, &info);
							// std::cout << "DONE" << std::endl;
							if (parent_no >= 0)
							{
								eq_map[parent_no] = my_no;
							}
						}
					}

					for (unsigned int k = 0; k < functable->numfields_C2TB-functable->numfields_C2TB_basebulk; k++)
					{
						std::string fieldname = functable->fieldnames_C2TB[functable->numfields_C2TB_basebulk + k];
						unsigned interf_id = codeinst->resolve_interface_dof_id(fieldname);					
						if (n->is_hanging(functable->nodal_offset_C2TB_basebulk))
						{
							throw_runtime_error("TODO: Hanging nodes on interfaces for equation remapping");
						}
						else
						{	
							unsigned valindex = dynamic_cast<oomph::BoundaryNodeBase *>(n)->index_of_first_value_assigned_by_face_element(interf_id);						
							int parent_no = from_elem->nodal_local_eqn(el_n_index, valindex);
							std::string info = "C2TB";
							int my_no = resolve_local_equation_for_external_contributions(n->eqn_number(valindex), from_elem, &info);
							if (parent_no >= 0)
							{
								eq_map[parent_no] = my_no;
							}
						}
					}
										//DG fields
					for (unsigned int fiDG = 0; fiDG < functable->numfields_D2TB; fiDG++)					
					{										   
						int parent_no = from_elem->get_D2TB_local_equation(fiDG,j);						
						std::string info = "D2TB";
						oomph::Data * DGdata=from_elem->get_D2TB_nodal_data(fiDG);
						int DG_value_index=from_elem->get_D2TB_node_index(fiDG,j);
						int my_no=resolve_local_equation_for_external_contributions(DGdata->eqn_number(DG_value_index), from_elem, &info);						
						if (parent_no >= 0)
						{
							eq_map[parent_no] = my_no;
						}
					}
				}				
			}

			if (required->dx_psi_C2 || required->psi_C2 || required->dX_psi_C2)
			{
				for (unsigned int j = 0; j < from_elem->get_eleminfo()->nnode_C2; j++)
				{
				   unsigned el_n_index=from_elem->get_node_index_C2_to_element(j);
					auto *n = from_elem->node_pt(el_n_index);
					//		  	 for (unsigned int k=0;k<n->nvalue();k++)
					for (unsigned int k = 0; k < functable->numfields_C2_basebulk; k++)
					{
						if (n->is_hanging(hanging_index))
						{
							//						std::cout << "SETTING HANHG " << k << "  FROM ELEM " << from_elem<< std::endl;
							oomph::HangInfo *const hang_pt = n->hanging_pt(hanging_index);
							const unsigned nmaster = hang_pt->nmaster();
							for (unsigned m = 0; m < nmaster; m++)
							{
								auto *const master_nod_pt = hang_pt->master_node_pt(m);
								int parent_no = from_elem->local_hang_eqn(master_nod_pt, functable->nodal_offset_C2_basebulk+k);
								//			  	 	std::cout << "HANG C2 " << j << "  " << "  " << k << "  " << m << master_nod_pt->eqn_number(k) << std::endl;
								std::string info = "C2 HANGIG";
								int my_no = resolve_local_equation_for_external_contributions(master_nod_pt->eqn_number(functable->nodal_offset_C2_basebulk+k), from_elem, &info);
								if (parent_no >= 0)
								{
									eq_map[parent_no] = my_no;
								}
							}
						}
						else
						{
							int parent_no = from_elem->nodal_local_eqn(el_n_index, functable->nodal_offset_C2_basebulk+k);
							  		  	 	   //std::cout << "C2 " << j << "  " << "  " << k << "  " << n->eqn_number(k) << std::endl;
							std::string info = "C2";
							int my_no = resolve_local_equation_for_external_contributions(n->eqn_number(functable->nodal_offset_C2_basebulk+k), from_elem, &info);
							// std::cout << "DONE" << std::endl;
							if (parent_no >= 0)
							{
								eq_map[parent_no] = my_no;
							}
						}
					}

					for (unsigned int k = 0; k < functable->numfields_C2-functable->numfields_C2_basebulk; k++)
					{
						std::string fieldname = functable->fieldnames_C2[functable->numfields_C2_basebulk + k];
						unsigned interf_id = codeinst->resolve_interface_dof_id(fieldname);					
						if (n->is_hanging(functable->nodal_offset_C2_basebulk))
						{
							throw_runtime_error("TODO: Hanging nodes on interfaces for equation remapping");
						}
						else
						{	
							unsigned valindex = dynamic_cast<oomph::BoundaryNodeBase *>(n)->index_of_first_value_assigned_by_face_element(interf_id);						
							int parent_no = from_elem->nodal_local_eqn(el_n_index, valindex);
							std::string info = "C2";
							//std::cout << "RESOLVING FOR INTERFACE FIELD " << fieldname << " AT NODE " << j << " OF " << from_elem->get_eleminfo()->nnode_C2 << " WITH PARENT NO " << parent_no << std::endl;
							//std::cout << "    WHICH HAS GLOBAL EQN " << n->eqn_number(valindex) << std::endl;
							int my_no = resolve_local_equation_for_external_contributions(n->eqn_number(valindex), from_elem, &info);
							if (parent_no >= 0)
							{
								eq_map[parent_no] = my_no;
							}
						}
					}
										
					//DG fields
					for (unsigned int fiDG = 0; fiDG < functable->numfields_D2; fiDG++)					
					{										   
						int parent_no = from_elem->get_D2_local_equation(fiDG,j);						
						std::string info = "D2";
						oomph::Data * DGdata=from_elem->get_D2_nodal_data(fiDG);
						int DG_value_index=from_elem->get_D2_node_index(fiDG,j);
						int my_no=resolve_local_equation_for_external_contributions(DGdata->eqn_number(DG_value_index), from_elem, &info);						
						if (parent_no >= 0)
						{
							eq_map[parent_no] = my_no;
						}
					}
				}
			}
			
			
			int hangind_C1_C1TB=functable->nodal_offset_C2_basebulk+functable->nodal_offset_C2TB_basebulk;

			
			if (required->dx_psi_C1TB || required->psi_C1TB || required->dX_psi_C1TB)
			{
	//		   std::cout << "FILLING EQMAP FOR C1TB" << std::endl;

				for (unsigned int j = 0; j < from_elem->get_eleminfo()->nnode_C1TB; j++)
				{
				   unsigned el_n_index=from_elem->get_node_index_C1TB_to_element(j);
//				   std::cout << "  C1TB NODE " << j << " OF " << from_elem->get_eleminfo()->nnode_C1TB << " IS " << el_n_index << std::endl;
					auto *n = from_elem->node_pt(el_n_index);
					for (unsigned int k = 0; k < functable->numfields_C1TB_basebulk; k++)
					{
		//		   std::cout << "    C1TB FIELD " << k << " OF " << functable->numfields_C1TB_basebulk << std::endl;					
						if (n->is_hanging(hangind_C1_C1TB)) // Hangs on C1
						{

							//						std::cout << "SETTING HANHG " << k << "  FROM ELEM " << from_elem<< std::endl;
							oomph::HangInfo *const hang_pt = n->hanging_pt(hangind_C1_C1TB);
							const unsigned nmaster = hang_pt->nmaster();
			//								   std::cout << "    HANGS WITH " << nmaster  << std::endl;											
							for (unsigned m = 0; m < nmaster; m++)
							{
								auto *const master_nod_pt = hang_pt->master_node_pt(m);
								int parent_no = from_elem->local_hang_eqn(master_nod_pt, functable->nodal_offset_C1TB_basebulk + k);
								//			  	 	std::cout << "HANG C1 " << j << "  " << "  " << k << "  " << m << master_nod_pt->eqn_number(k) << std::endl;
								std::string info = "C1TB HANG";
								int my_no = resolve_local_equation_for_external_contributions(master_nod_pt->eqn_number(functable->nodal_offset_C1TB_basebulk + k), from_elem, &info);
								if (parent_no >= 0)
								{
									eq_map[parent_no] = my_no;
								}
							}
						}
						else
						{
					//						   std::cout << "    DOES NOT HANG " << std::endl;																	
							int parent_no = from_elem->nodal_local_eqn(el_n_index, functable->nodal_offset_C1TB_basebulk + k);
							std::string info = "C1TB";
							int my_no = resolve_local_equation_for_external_contributions(n->eqn_number(functable->nodal_offset_C1TB_basebulk + k), from_elem, &info);
						//					   std::cout << "    OWN EQ " << my_no  << " PARENT NOT " << parent_no <<std::endl;																								
							if (parent_no >= 0)
							{
								eq_map[parent_no] = my_no;
							}
						}
					}
					
					for (unsigned int k = 0; k < functable->numfields_C1TB-functable->numfields_C1TB_basebulk; k++)
					{
						std::string fieldname = functable->fieldnames_C1TB[functable->numfields_C1TB_basebulk + k];
						unsigned interf_id = codeinst->resolve_interface_dof_id(fieldname);					
						if (n->is_hanging(hangind_C1_C1TB))
						{
							throw_runtime_error("TODO: Hanging nodes on interfaces for equation remapping");
						}
						else
						{	
							unsigned valindex = dynamic_cast<oomph::BoundaryNodeBase *>(n)->index_of_first_value_assigned_by_face_element(interf_id);						
							int parent_no = from_elem->nodal_local_eqn(el_n_index, valindex);
							std::string info = "C1TB";
							int my_no = resolve_local_equation_for_external_contributions(n->eqn_number(valindex), from_elem, &info);
							if (parent_no >= 0)
							{
								eq_map[parent_no] = my_no;
							}
						}
					}

										//DG fields
					for (unsigned int fiDG = 0; fiDG < functable->numfields_D1TB; fiDG++)					
					{										   
						int parent_no = from_elem->get_D1TB_local_equation(fiDG,j);						
						std::string info = "D1TB";
						oomph::Data * DGdata=from_elem->get_D1TB_nodal_data(fiDG);
						int DG_value_index=from_elem->get_D1TB_node_index(fiDG,j);
						int my_no=resolve_local_equation_for_external_contributions(DGdata->eqn_number(DG_value_index), from_elem, &info);						
						if (parent_no >= 0)
						{
							eq_map[parent_no] = my_no;
						}
					}
				}
			}
			

			if (required->dx_psi_C1 || required->psi_C1 || required->dX_psi_C1)
			{
				for (unsigned int j = 0; j < from_elem->get_eleminfo()->nnode_C1; j++)
				{
				   unsigned el_n_index=from_elem->get_node_index_C1_to_element(j);
					auto *n = from_elem->node_pt(el_n_index);
					for (unsigned int k = 0; k < functable->numfields_C1_basebulk; k++)
					{
						if (n->is_hanging(hangind_C1_C1TB))
						{
							//						std::cout << "SETTING HANHG " << k << "  FROM ELEM " << from_elem<< std::endl;
							oomph::HangInfo *const hang_pt = n->hanging_pt(hangind_C1_C1TB);
							const unsigned nmaster = hang_pt->nmaster();
							for (unsigned m = 0; m < nmaster; m++)
							{
								auto *const master_nod_pt = hang_pt->master_node_pt(m);
								int parent_no = from_elem->local_hang_eqn(master_nod_pt, functable->nodal_offset_C1_basebulk + k);
								//			  	 	std::cout << "HANG C1 " << j << "  " << "  " << k << "  " << m << master_nod_pt->eqn_number(k) << std::endl;
								std::string info = "C1 HANG";
								int my_no = resolve_local_equation_for_external_contributions(master_nod_pt->eqn_number(functable->nodal_offset_C1_basebulk + k), from_elem, &info);
								if (parent_no >= 0)
								{
									eq_map[parent_no] = my_no;
								}
							}
						}
						else
						{
							int parent_no = from_elem->nodal_local_eqn(el_n_index, functable->nodal_offset_C1_basebulk + k);
							std::string info = "C1";
							int my_no = resolve_local_equation_for_external_contributions(n->eqn_number(functable->nodal_offset_C1_basebulk + k), from_elem, &info);
							if (parent_no >= 0)
							{
								eq_map[parent_no] = my_no;
							}
						}
					}
					
					for (unsigned int k = 0; k < functable->numfields_C1-functable->numfields_C1_basebulk; k++)
					{
						std::string fieldname = functable->fieldnames_C1[functable->numfields_C1_basebulk + k];
						unsigned interf_id = codeinst->resolve_interface_dof_id(fieldname);					
						if (n->is_hanging(hangind_C1_C1TB))
						{
							throw_runtime_error("TODO: Hanging nodes on interfaces for equation remapping");
						}
						else
						{	
							unsigned valindex = dynamic_cast<oomph::BoundaryNodeBase *>(n)->index_of_first_value_assigned_by_face_element(interf_id);						
							int parent_no = from_elem->nodal_local_eqn(el_n_index, valindex);
							std::string info = "C1";
							int my_no = resolve_local_equation_for_external_contributions(n->eqn_number(valindex), from_elem, &info);
							if (parent_no >= 0)
							{
								eq_map[parent_no] = my_no;
							}
						}
					}

										//DG fields
					for (unsigned int fiDG = 0; fiDG < functable->numfields_D1; fiDG++)					
					{										   
						int parent_no = from_elem->get_D1_local_equation(fiDG,j);						
						std::string info = "D1";
						oomph::Data * DGdata=from_elem->get_D1_nodal_data(fiDG);
						int DG_value_index=from_elem->get_D1_node_index(fiDG,j);
						int my_no=resolve_local_equation_for_external_contributions(DGdata->eqn_number(DG_value_index), from_elem, &info);						
						if (parent_no >= 0)
						{
							eq_map[parent_no] = my_no;
						}
					}
				}
			}

			if (required->psi_DL || required->dx_psi_DL || required->dX_psi_DL)
			{
				unsigned ndl = functable->numfields_DL;

				for (unsigned int k = 0; k < ndl; k++)
				{
					auto *n = from_elem->internal_data_pt(functable->internal_offset_DL+k);
					for (unsigned int j = 0; j < n->nvalue(); j++)
					{
						int parent_no = from_elem->get_internal_local_eqn(functable->internal_offset_DL+k, j);
						std::string info = "DL";
						int my_no = resolve_local_equation_for_external_contributions(n->eqn_number(j), from_elem, &info);
						if (parent_no >= 0)
						{
							eq_map[parent_no] = my_no;
						}
					}
				}
			}

			if (required->psi_D0)
			{
				unsigned nd0 = functable->numfields_D0;
				for (unsigned int j = 0; j < 1; j++)
				{
					for (unsigned int k = 0; k < nd0; k++)
					{
						auto *n = from_elem->internal_data_pt(functable->internal_offset_D0 + k);
						int parent_no = from_elem->get_internal_local_eqn(functable->internal_offset_D0 + k, j);
						std::string info = "D0";
						int my_no = resolve_local_equation_for_external_contributions(n->eqn_number(j), from_elem, &info);
						if (parent_no >= 0)
						{
							eq_map[parent_no] = my_no;
						}
					}
				}
			}
		}
	}

   double InterfaceElementBase::get_interpolated_interface_field(const oomph::Vector<double> &s,  const unsigned & ifindex,const std::string & space,const unsigned &t) const
   {
		double res=0.0;		
		oomph::Shape psi;
      std::vector<unsigned> node_index;		
		if (space=="C2TB")
		{
		  psi.resize(eleminfo.nnode_C2TB);
		  node_index.resize(psi.nindex1());
  		  this->shape_at_s_C2TB(s, psi);				 
  		  for (unsigned int i=0;i<node_index.size();i++) node_index[i]=this->get_node_index_C2TB_to_element(i);
		}
		else if (space=="C2")
		{
		  psi.resize(eleminfo.nnode_C2);
		  node_index.resize(psi.nindex1());
  		  this->shape_at_s_C2(s, psi);				 
  		  for (unsigned int i=0;i<node_index.size();i++) node_index[i]=this->get_node_index_C2_to_element(i);
		}
		else if (space=="C1TB")
		{
		  psi.resize(eleminfo.nnode_C1TB);
		  node_index.resize(psi.nindex1());
  		  this->shape_at_s_C1TB(s, psi);				 
  		  for (unsigned int i=0;i<node_index.size();i++) node_index[i]=this->get_node_index_C1TB_to_element(i);
		}		
		else if (space=="C1")
		{
		  psi.resize(eleminfo.nnode_C1);
		  node_index.resize(psi.nindex1());
  		  this->shape_at_s_C1(s, psi);				 
  		  for (unsigned int i=0;i<node_index.size();i++) node_index[i]=this->get_node_index_C1_to_element(i);
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
   
	void InterfaceElementBase::assign_additional_local_eqn_numbers()
	{
		// return;
		BulkElementBase::assign_additional_local_eqn_numbers();
		oomph::FaceElement::assign_additional_local_eqn_numbers();
		
		if (opposite_side && dynamic_cast<InterfaceElementBase *>(opposite_side)->is_internal_facet_opposite_dummy() && !(opposite_side->ndof()))
		{

		  dynamic_cast<InterfaceElementBase *>(opposite_side)->assign_local_eqn_numbers(true);
		}

		const JITFuncSpec_Table_FiniteElement_t *functable = this->codeinst->get_func_table();
		// std::cout << "ADDING BULK ELEMENT DOFS, THIS NNODE " << this->nnode() <<" BULK " << dynamic_cast<BulkElementBase*>(this->bulk_element_pt()) << " BULK NNODE " << this->bulk_element_pt()->nnode() <<std::endl;
		assign_additional_local_eqn_numbers_from_elem(functable->merged_required_shapes.bulk_shapes, dynamic_cast<BulkElementBase *>(this->bulk_element_pt()), bulk_eqn_map);
	//	for (unsigned int i=0;i<bulk_eqn_map.size();i++) {
	//	 std::cout << "BULK EQUATION MAP " << i << "  " << bulk_eqn_map[i] << std::endl;
	//	}
		if (functable->merged_required_shapes.bulk_shapes && functable->merged_required_shapes.bulk_shapes->bulk_shapes)
		{
			assign_additional_local_eqn_numbers_from_elem(functable->merged_required_shapes.bulk_shapes->bulk_shapes, dynamic_cast<BulkElementBase *>(dynamic_cast<InterfaceElementBase *>(this->bulk_element_pt())->bulk_element_pt()), bulk_bulk_eqn_map);
		}
		if (functable->merged_required_shapes.opposite_shapes && !is_internal_facet_opposite_dummy())
		{
			if (!dynamic_cast<InterfaceElementBase *>(opposite_side))
			{
				throw_runtime_error("Missing opposite element");
			}
			if (!this->is_internal_facet_opposite_dummy())
			{
				//   std::cout << "ADDING OPPOSITE INTERFACE DOFS , THIS NNODE " << this->nnode() <<" OPP" <<dynamic_cast<InterfaceElementBase*>(opposite_side)  <<   " OPP NNODE " << dynamic_cast<InterfaceElementBase*>(opposite_side)->nnode() <<std::endl;
				assign_additional_local_eqn_numbers_from_elem(functable->merged_required_shapes.opposite_shapes, opposite_side, opp_interf_eqn_map);
				if (functable->merged_required_shapes.opposite_shapes->bulk_shapes)
				{
					if (!dynamic_cast<InterfaceElementBase *>(opposite_side)->bulk_element_pt())
					{
						throw_runtime_error("Missing opposite bulk element");
					}
					//        std::cout << "ADDING OPPOSITE BULK DOFS " << dynamic_cast<InterfaceElementBase*>(opposite_side)->bulk_element_pt() << std::endl;
					assign_additional_local_eqn_numbers_from_elem(functable->merged_required_shapes.opposite_shapes->bulk_shapes, dynamic_cast<BulkElementBase *>(dynamic_cast<InterfaceElementBase *>(opposite_side)->bulk_element_pt()), opp_bulk_eqn_map);
				}
			}
		}
	}

	IntegrationSchemeStorage integration_scheme_storage;

	// const unsigned BulkElementTri2dC2TB::Central_node_on_face[3] = {4,5,3};
	oomph::TBubbleEnrichedGauss<2, 3> BulkElementTri2dC1TB::Default_enriched_integration_scheme;	
	oomph::TBubbleEnrichedGauss<2, 3> BulkElementTri2dC2TB::Default_enriched_integration_scheme;
	oomph::TBubbleEnrichedGauss<3, 3> BulkElementTetra3dC2TB::Default_enriched_integration_scheme;
	
	bool InterfaceElementBase::interpolate_new_interface_dofs=true;

}
