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


#include "mesh.hpp"
#include "exception.hpp"
#include <cassert>
#include <functional>

#include "elements.hpp"
#include "problem.hpp"
#include "expressions.hpp"
#include <cln/float.h>
#include "codegen.hpp"
#include "kdtree.hpp"

#include "timestepper.hpp"

#include "missing_masters.h"
#include "missing_masters.hpp"

using namespace oomph;

namespace pyoomph
{

  typedef double (*InitialConditionFctPt)(const double &t);

  int Mesh::get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nelem, bool discontinuous) // Gets the number of required elemental indices
  {
    unsigned nelement = this->nelement();

    for (unsigned int ne = 0; ne < nelement; ne++)
    {
      dynamic_cast<BulkElementBase *>(this->element_pt(ne))->_numpy_index = ne;
    }
    std::vector<std::vector<std::set<oomph::Node *>>> additional_elemental_tri_nodes(nelement);
    if (tesselate_tri && !discontinuous)
    {
      unsigned milev = 0, malev = 0;
      oomph::TreeBasedRefineableMeshBase *tbself = dynamic_cast<oomph::TreeBasedRefineableMeshBase *>(this);
      if (tbself)
        tbself->get_refinement_levels(milev, malev);
      if (milev < malev)
      {
        for (unsigned int ne = 0; ne < nelement; ne++)
        {
          dynamic_cast<BulkElementBase *>(this->element_pt(ne))->inform_coarser_neighbors_for_tesselated_numpy(additional_elemental_tri_nodes);
        }
      }
    }

    int res = 0;
    nelem = 0;
    for (unsigned int ne = 0; ne < nelement; ne++)
    {
      BulkElementBase *be = dynamic_cast<BulkElementBase *>(this->element_pt(ne));
      unsigned nsubelem = 0;
      res = std::max(res, be->get_num_numpy_elemental_indices(tesselate_tri, nsubelem, additional_elemental_tri_nodes));
      nelem += nsubelem;
    }
    return res;
  }

  MeshKDTree *Mesh::get_lagrangian_kdtree()
  {
    if (!lagrangian_kdtree)
    {
      lagrangian_kdtree = new MeshKDTree(this, true, 0);
    }
    return lagrangian_kdtree;
  }

  void Mesh::invalidate_lagrangian_kdtree()
  {
    if (lagrangian_kdtree)
      delete lagrangian_kdtree;
    lagrangian_kdtree = NULL;
  }
  Mesh::~Mesh()
  {
    if (lagrangian_kdtree)
      delete lagrangian_kdtree;
    lagrangian_kdtree = NULL;
  }

  void Mesh::check_integrity()
  {
    if (dynamic_cast<oomph::TreeBasedRefineableMeshBase *>(this) && dynamic_cast<oomph::TreeBasedRefineableMeshBase *>(this)->forest_pt())
    {

      oomph::DocInfo docinfo;
      docinfo.disable_doc();
      dynamic_cast<oomph::TreeBasedRefineableMeshBase *>(this)->forest_pt()->check_all_neighbours(docinfo);
    }
    
  }

  unsigned Mesh::count_nnode(bool discontinuous)
  {
    if (!discontinuous)
      return this->nnode();
    else
    {
      unsigned res = 0;
      for (unsigned ie = 0; ie < this->nelement(); ie++)
        res += dynamic_cast<oomph::FiniteElement *>(this->element_pt(ie))->nnode();
      return res;
    }
  }
  void Mesh::_setup_information_from_old_mesh(Mesh *old)
  {
    for (unsigned int i = 0; i < std::min(this->dirichlet_active.size(), old->dirichlet_active.size()); i++)
    {
      this->dirichlet_active[i] = old->dirichlet_active[i];
    }
  }

  void Mesh::boundary_coordinates_bool(unsigned boundary_index)
  {
    Boundary_coordinate_exists[boundary_index] = true;
  }

  bool Mesh::is_boundary_coordinate_defined(unsigned boundary_index)
  {
    return boundary_index < Boundary_coordinate_exists.size() && Boundary_coordinate_exists[boundary_index];
  }

  void Mesh::_save_state(std::vector<double> &meshdata)
  {
    bool old_ordering = true;
    oomph::Vector<oomph::Node *> nodes;
    this->get_node_reordering(nodes, old_ordering);

    meshdata.clear();
    for (auto nii : nodes)
    {
      pyoomph::Node *n = dynamic_cast<pyoomph::Node *>(nii);
      unsigned ntstor = n->ntstorage();
      for (unsigned int iv = 0; iv < n->ndim(); iv++)
      {
        for (unsigned int ti = 0; ti < ntstor; ti++)
        {
          meshdata.push_back(n->variable_position_pt()->value(ti, iv));
        }
      }
      for (unsigned int iv = 0; iv < n->nlagrangian(); iv++)
      {
        meshdata.push_back(n->xi(iv));
      }
      for (unsigned int iv = 0; iv < n->nvalue(); iv++)
      {
        for (unsigned int ti = 0; ti < ntstor; ti++)
        {
          meshdata.push_back(n->value(ti, iv));
        }
      }
    }

    for (unsigned int ie = 0; ie < this->nelement(); ie++)
    {
      BulkElementBase *e = dynamic_cast<BulkElementBase *>(this->element_pt(ie));
      for (unsigned int ied = 0; ied < e->ninternal_data(); ied++)
      {
        for (unsigned int iv = 0; iv < e->internal_data_pt(ied)->nvalue(); iv++)
        {
          for (unsigned int t = 0; t < e->internal_data_pt(ied)->ntstorage(); t++)
          {
            meshdata.push_back(e->internal_data_pt(ied)->value(t, iv));
          }
        }
      }
      meshdata.push_back(e->initial_cartesian_nondim_size);
      meshdata.push_back(e->initial_quality_factor);
    }
  }

  void Mesh::_load_state(const std::vector<double> &meshdata)
  {
    size_t s = 0;
    bool old_ordering = true;
    oomph::Vector<oomph::Node *> nodes;
    this->get_node_reordering(nodes, old_ordering);

    //for (unsigned nii = 0; nii < this->nnode(); nii++)
    //{
    //  pyoomph::Node *n = dynamic_cast<pyoomph::Node *>(this->node_pt(nii));
    for (auto * nn : nodes)
    {
      pyoomph::Node *n = dynamic_cast<pyoomph::Node *>(nn);
      unsigned ntstor = n->ntstorage();
      for (unsigned int iv = 0; iv < n->ndim(); iv++)
      {
        for (unsigned int ti = 0; ti < ntstor; ti++)
        {
          n->variable_position_pt()->set_value(ti, iv, meshdata[s++]);
        }
      }
      for (unsigned int iv = 0; iv < n->nlagrangian(); iv++)
      {
        n->xi(iv) = meshdata[s++];
      }
      for (unsigned int iv = 0; iv < n->nvalue(); iv++)
      {
        for (unsigned int ti = 0; ti < ntstor; ti++)
        {
          n->set_value(ti, iv, meshdata[s++]);
        }
      }
    }
    for (unsigned int ie = 0; ie < this->nelement(); ie++)
    {
      BulkElementBase *e = dynamic_cast<BulkElementBase *>(this->element_pt(ie));
      for (unsigned int ied = 0; ied < e->ninternal_data(); ied++)
      {
        for (unsigned int iv = 0; iv < e->internal_data_pt(ied)->nvalue(); iv++)
        {
          for (unsigned int t = 0; t < e->internal_data_pt(ied)->ntstorage(); t++)
          {
            e->internal_data_pt(ied)->set_value(t, iv, meshdata[s++]);
          }
        }
      }
      e->initial_cartesian_nondim_size = meshdata[s++];
      e->initial_quality_factor = meshdata[s++];
    }
  }

  // Find elements that do not share a facet with the boundary
  void Mesh::enlarge_elemental_error_max_override_to_only_nodal_connected_elems(unsigned bind)
  {
    std::set<pyoomph::BulkElementBase *> elems_with_boundnodes;
    for (unsigned int ie = 0; ie < this->nelement(); ie++)
    {
      pyoomph::BulkElementBase *el = dynamic_cast<pyoomph::BulkElementBase *>(this->element_pt(ie));
      if (!el)
        continue;
      for (unsigned int in = 0; in < el->nnode(); in++)
      {
        if (el->node_pt(in)->is_on_boundary(bind))
        {
          elems_with_boundnodes.insert(el);
          break;
        }
      }
    }
    std::map<oomph::Node *, std::vector<pyoomph::BulkElementBase *>> facet_elems_at_node;
    // Remove the elements which share a facet with the boundary
    for (unsigned int bi = 0; bi < this->nboundary_element(bind); bi++)
    {
      pyoomph::BulkElementBase *el = dynamic_cast<pyoomph::BulkElementBase *>(this->boundary_element_pt(bind, bi));
      if (!el)
        continue;
      elems_with_boundnodes.erase(el);
      for (unsigned int in = 0; in < el->nvertex_node(); in++)
      {
        oomph::Node *vn = el->vertex_node_pt(in);
        if (vn->is_on_boundary(bind))
        {
          if (!facet_elems_at_node.count(vn))
          {
            facet_elems_at_node[vn] = {el};
          }
          else
          {
            facet_elems_at_node[vn].push_back(el);
          }
        }
      }
    }

    for (pyoomph::BulkElementBase *el : elems_with_boundnodes)
    {
      for (unsigned int in = 0; in < el->nvertex_node(); in++)
      {
        oomph::Node *vn = el->vertex_node_pt(in);
        if (vn->is_on_boundary(bind) && facet_elems_at_node.count(vn))
        {
          for (pyoomph::BulkElementBase *f_el : facet_elems_at_node[vn])
          {
            el->elemental_error_max_override = std::max(el->elemental_error_max_override, f_el->elemental_error_max_override);
          }
        }
      }
    }
  }

  void Mesh::ensure_halos_for_periodic_boundaries()
  {
#ifdef OOMPH_HAS_MPI
    // if (!this->is_mesh_distributed()) return;
    for (unsigned int ib = 0; ib < this->nboundary(); ib++)
    {
      unsigned nbe = this->nboundary_element(ib);
      //	std::cout << "NBE IS " << nbe << std::endl;
      for (unsigned int ie = 0; ie < nbe; ie++)
      {
        auto *be = dynamic_cast<BulkElementBase *>(this->boundary_element_pt(ib, ie));
        //		std::cout << "BE IS " << be << std::endl;
        for (unsigned int in = 0; in < be->nnode(); in++)
        {
          auto *n = be->node_pt(in);
          //			std::cout << "N IS " << n << std::endl;
          if (n->is_on_boundary(ib) && n->is_a_copy())
          {
            if (n->nvalue() > 0 || (dynamic_cast<pyoomph::Node*>(n)->variable_position_pt()->nvalue() > 0 && dynamic_cast<pyoomph::Node*>(n)->variable_position_pt()->is_a_copy()))
            {
              throw_runtime_error("Distributed parallel with copied nodes (i.e. PeriodicBC) does not work with nodal degrees of freedom. Either use pure DG or implement a periodic boundary condition by Lagrange multipliers");
            }
            std::cout << "FOUND ELEM NODE: " << ib << "  " << ie << "  " << in << "  iscpy " << n->is_a_copy() << std::endl;
            auto *master = n->copied_node_pt();
            std::cout << "MASTER NODE " << master << std::endl;
            for (unsigned int ib2 = 0; ib2 < this->nboundary(); ib2++)
            {
              if (master->is_on_boundary(ib2))
              {
                unsigned nbe2 = this->nboundary_element(ib2);
                for (unsigned int ie2 = 0; ie2 < nbe2; ie2++)
                {
                  auto *be2 = dynamic_cast<BulkElementBase *>(this->boundary_element_pt(ib2, ie2));
                  if (be2->get_node_number(master) != -1)
                  {
                    be2->set_must_be_kept_as_halo();
                    be->set_must_be_kept_as_halo();
                    break;
                  }
                }
                break;
              }
            }
          }
        }
      }
    }
#endif
  }

  std::vector<std::string> Mesh::list_integral_functions()
  {
    unsigned nelement = this->nelement();
    if (!nelement)
      return std::vector<std::string>();
    auto *cg = dynamic_cast<BulkElementBase *>(this->element_pt(0))->get_code_instance()->get_element_class();
    return cg->get_integral_expressions();
  }

  std::vector<std::string> Mesh::list_local_expressions()
  {
    unsigned nelement = this->nelement();
    if (!nelement)
      return std::vector<std::string>();
    auto *cg = dynamic_cast<BulkElementBase *>(this->element_pt(0))->get_code_instance()->get_element_class();
    return cg->get_local_expressions();
  }

  double improve_extremum_by_newton(BulkElementBase *be, unsigned index, oomph::Vector<double> &s)
  {    
    double val=be->eval_extremum_expression_at_s(index,s); //Update value
    if (!be->dim()) return val; // Cannot optimize a point
    oomph::Vector<double> grad(be->dim());
    oomph::Vector<double> spert(s.size());
    oomph::DenseDoubleMatrix hess(be->dim(), be->dim());
    unsigned maxiter = 10;
    double eps=1e-8;
    for (unsigned int it = 0; it < maxiter; it++)
    {
      
      for (unsigned int i =0;i<s.size();i++)  spert[i]=s[i];
      for (unsigned int i=0;i<s.size();i++)
      {        
        spert[i]=s[i]+eps;
        double valpert=be->eval_extremum_expression_at_s(index,spert);
        spert[i]=s[i]-eps;
        double valpertm=be->eval_extremum_expression_at_s(index,spert);
        grad[i]=(valpert-valpertm)/(2*eps);
        spert[i]=s[i];
        hess(i,i)=(valpert-2*val+valpertm)/(eps*eps);           
        for (unsigned int j=i+1;j<s.size();j++)
        {          
            spert[i]+=eps;
            spert[j]+=eps;
            double fpp=be->eval_extremum_expression_at_s(index,spert);
            spert[j]=s[j]-eps;
            double fpm=be->eval_extremum_expression_at_s(index,spert);
            spert[i]=s[i]-eps;
            double fmm=be->eval_extremum_expression_at_s(index,spert);
            spert[j]=s[j]+eps;
            double fmp=be->eval_extremum_expression_at_s(index,spert);
            spert[i]=s[i];
            spert[j]=s[j];            
            hess(i,j)=hess(j,i)=1.0/(4*eps*eps)*(fpp-fpm-fmp+fmm);                   
        }
      }
      try
      {
        hess.solve(grad);
      }        
      catch (oomph::OomphLibError& error)
      {        
        error.disable_error_message();
#ifdef PARANOID
      oomph::oomph_info << "Error in linear solve for improving extremum!" << std::endl;                   
#endif
      }
    // Add the correction to the local coordinates
      std::vector<double> old_s(s.size());
      for (unsigned i = 0; i < s.size(); i++)
      {
        //std::cout << "  CHANGING s" << i << "from "  << s[i] << " by  " << grad[i] << std::endl;
        s[i] -= grad[i];
      }
      if (!be->local_coord_is_valid(s))
      {
        //std::cout << " PUSHING  S BACK" << std::endl;
        be->move_local_coord_back_into_element(s);
      }
              
      //std::cout << "CHANGED VALUE FROM " << val << " TO " ;
      val=be->eval_extremum_expression_at_s(index,s); //Update value
      //std::cout << val << std::endl;
    }
    return val;
  }
  

  GiNaC::ex Mesh::evaluate_extremum(std::string name,int sign,BulkElementBase *& extreme_element,oomph::Vector<double> &extreme_local_coords,unsigned flags)
  {
    unsigned nelement = this->nelement();
    if (!nelement)
    {
      extreme_element=NULL;
      return 0;
    }
    int index = dynamic_cast<BulkElementBase *>(this->element_pt(0))->get_code_instance()->get_extremum_function_index(name);
    if (index < 0) throw_runtime_error("Extremum function " + name + " not defined on this mesh");
    // Get some reference to start with
    extreme_element=dynamic_cast<BulkElementBase *>(this->element_pt(0));
    extreme_local_coords.resize(extreme_element->dim());
    extreme_element->local_coordinate_of_node(0,extreme_local_coords);
    double extreme_value = sign*extreme_element->eval_extremum_expression_at_s(index,extreme_local_coords);
    
    // Go for a Gauss-Legendre sampling (not sure whether it is the best idea)
    for (unsigned int ne = 0; ne < nelement; ne++)
    {
      //std::cout << "ITERATING " << ne << " OF " << nelement  << std::endl << std::flush;
      BulkElementBase *be = dynamic_cast<BulkElementBase *>(this->element_pt(ne));
      unsigned nintpt = be->integral_pt()->nweight();
      for (unsigned int ipt = 0; ipt < nintpt; ipt++)
      {
        
        oomph::Vector<double> s(be->dim());
        for (unsigned int i = 0; i < be->dim(); i++)
        {
          s[i] = be->integral_pt()->knot(ipt, i);
        }
        double val = sign*be->eval_extremum_expression_at_s(index,s);
        if (val > extreme_value)
        {
          extreme_value = val;
          extreme_element=be;
          extreme_local_coords=s;
        }
      }
      // And also sample the nodes directly
      for (unsigned int in = 0; in < be->nnode(); in++)
      {
        oomph::Vector<double> s;
        be->local_coordinate_of_node(in,s);
        double val = sign*be->eval_extremum_expression_at_s(index,s);
        if (val > extreme_value)
        {
          extreme_value = val;
          extreme_element=be;
          extreme_local_coords=s;
        }
      }
    }

    extreme_value *= sign;
    
    // TODO: Improve the extremum by a Newton or gradient descent or something. Mind leaving the element and mind the boundary
    //std::cout << "ENTERING IMPROVEMENT  " << extreme_local_coords.size() << std::endl;
    double improved=improve_extremum_by_newton(extreme_element,index,extreme_local_coords);
    if (sign*improved<sign*extreme_value)
    {
      //std::cout << "IMPROVEMENT FAILED " << improved << "  " << extreme_value << std::endl;
    }
    else
    {
      //std::cout << "IMPROVEMENT SUCCEEDED " << improved << "  " << extreme_value << std::endl;
      extreme_value=improved;
    }
    

    
    if (flags & 1)
    {
      GiNaC::ex factor_and_unit = dynamic_cast<BulkElementBase *>(this->element_pt(0))->get_code_instance()->get_element_class()->get_extremum_expression_unit_factor(name);
      return factor_and_unit*extreme_value;
      
    }
    else
    {
      return extreme_value;
    }
    
    
  }

  GiNaC::ex Mesh::evaluate_integral_function(std::string name)
  {
    unsigned nelement = this->nelement();
    if (!nelement)
      return 0;
    int index = dynamic_cast<BulkElementBase *>(this->element_pt(0))->get_code_instance()->get_integral_function_index(name);
    if (index < 0)
      throw_runtime_error("Integral function " + name + " not defined on this mesh");
    double res = 0.0;
    bool distributed = this->is_mesh_distributed();
    for (unsigned int ne = 0; ne < nelement; ne++)
    {
#ifdef OOMPH_HAS_MPI
      if (this->element_pt(ne)->is_halo())
        continue;
#endif
      BulkElementBase *be = dynamic_cast<BulkElementBase *>(this->element_pt(ne));
      res += be->eval_integral_expression(index);
    }
#ifdef OOMPH_HAS_MPI
    if (distributed)
    {
      double sum = 0;
      MPI_Allreduce(&res, &sum, 1, MPI_DOUBLE, MPI_SUM, this->communicator_pt()->mpi_comm());
      res = sum;
    }
#endif
    GiNaC::ex factor_and_unit = dynamic_cast<BulkElementBase *>(this->element_pt(0))->get_code_instance()->get_element_class()->get_integral_expression_unit_factor(name);
    return factor_and_unit * res;
  }

  void Mesh::ensure_external_data()
  {
    unsigned nelement = this->nelement();
    for (unsigned int ne = 0; ne < nelement; ne++)
    {
      BulkElementBase *be = dynamic_cast<BulkElementBase *>(this->element_pt(ne));
      be->ensure_external_data();
    }
  }

  void Mesh::generate_interface_elements(std::string intername, Mesh *imesh, DynamicBulkElementInstance *jitcode)
  {
    unsigned bind, nbe;
    bool internal_facets;
    if (intername == "_internal_facets_")
    {
      internal_facets = true;
    }
    else
    {
      bind = this->get_boundary_index(intername);
      internal_facets = false;
      nbe = this->nboundary_element(bind);
    }

    BulkElementBase::__CurrentCodeInstance = jitcode;
    dynamic_cast<InterfaceMesh *>(imesh)->set_rebuild_information(this, intername, jitcode);

    unsigned n_element = imesh->nelement();
    for (unsigned e = 0; e < n_element; e++)
    {
      delete imesh->element_pt(e);
    }
    imesh->flush_element_and_node_storage(); //TODO: This keeps the nodes alive
    for (unsigned i = 0; i < dynamic_cast<InterfaceMesh *>(imesh)->opposite_interior_facets.size(); i++)
      delete dynamic_cast<InterfaceMesh *>(imesh)->opposite_interior_facets[i];
    dynamic_cast<InterfaceMesh *>(imesh)->opposite_interior_facets.clear();

    int restriction_index = -1;
    for (unsigned int i = 0; i < jitcode->get_func_table()->numlocal_expressions; i++)
    {
      if (std::string(jitcode->get_func_table()->local_expressions_names[i]) == "__interface_constraint")
      {
        restriction_index = i;
        break;
      }
    }

    std::vector<BulkElementBase *> internal_elements, opposite_elements;
    std::vector<int> internal_face_dir, opposite_face_dir, opposite_already_at_index;
    if (internal_facets)
    {
      this->fill_internal_facet_buffers(internal_elements, internal_face_dir, opposite_elements, opposite_face_dir, opposite_already_at_index);
      nbe = internal_elements.size();
    }

    auto gen_face_elem = [jitcode,internal_facets](BulkElementBase *be, int fi)->oomph::FaceElement *
    {
      oomph::FaceElement *fe = NULL;
      if (dynamic_cast<BulkElementQuad2dC2 *>(be))
      {
        fe = new InterfaceElementLine1dC2(jitcode, be, fi);
      }
      else if (dynamic_cast<BulkElementQuad2dC1 *>(be))
      {
        fe = new InterfaceElementLine1dC1(jitcode, be, fi);
      }

      // TODO: Tris? Are they different from Quads regarding the interface elements
      else if (dynamic_cast<BulkElementTri2dC2 *>(be))
      {
        //      std::cout << "TRID 2d " << be << "  FI "  << fi << std::endl;
        fe = new InterfaceTElementLine1dC2(jitcode, be, fi);
      }
      else if (dynamic_cast<BulkElementTri2dC1 *>(be))
      {
        fe = new InterfaceTElementLine1dC1(jitcode, be, fi);
      }

      else if (dynamic_cast<BulkElementLine1dC1 *>(be) || dynamic_cast<BulkElementLine1dC2 *>(be))
      {
        fe = new InterfaceElementPoint0d(jitcode, be, fi);
      }
      else if (dynamic_cast<BulkTElementLine1dC1 *>(be) || dynamic_cast<BulkTElementLine1dC2 *>(be))
      {
        fe = new InterfaceElementPoint0d(jitcode, be, fi); // TODO: IS this right for tris?
      }
      else if (dynamic_cast<BulkElementBrick3dC1 *>(be))
      {
        fe = new InterfaceElementQuad2dC1(jitcode, be, fi);
      }
      else if (dynamic_cast<BulkElementBrick3dC2 *>(be))
      {
        fe = new InterfaceElementQuad2dC2(jitcode, be, fi);
      }
      else if (dynamic_cast<BulkElementTetra3dC1 *>(be))
      {
        fe = new InterfaceElementTri2dC1(jitcode, be, fi);
      }
      else if (dynamic_cast<BulkElementTetra3dC2 *>(be))
      {
        fe = new InterfaceElementTri2dC2(jitcode, be, fi);
      }
      else
        throw_runtime_error("Implement interface element generation for this elementtype");
      if (jitcode->get_func_table()->integration_order)
      {
        dynamic_cast<BulkElementBase *>(fe)->set_integration_order(jitcode->get_func_table()->integration_order);
      }

      if (!internal_facets)
      {
        for (unsigned int in=0;in<fe->nnode();in++)
        {
          oomph::Node *n = fe->node_pt(in);
          if (!dynamic_cast<oomph::BoundaryNodeBase*>(n)) 
          {
            //std::cout << "ERROR: Node " << n << " in interface element " << fe << " is not a boundary node!" << std::endl;
            return NULL; // Do not create such elements...
          }
        }
      }

      return fe;
    };

    std::vector<oomph::FaceElement *> generated_opposite_face_elems;

    for (unsigned int ei = 0; ei < nbe; ei++)
    {
      BulkElementBase *be;
      int fi;
      if (internal_facets)
      {
        be = internal_elements[ei];
        fi = internal_face_dir[ei];
      }
      else
      {
        be = dynamic_cast<BulkElementBase *>(this->boundary_element_pt(bind, ei));
        fi = this->face_index_at_boundary(bind, ei);
      }
      oomph::FaceElement *fe = gen_face_elem(be, fi);
      if (!fe) continue;

      if (restriction_index >= 0)
      {
        //std::cout << "RESTRALL " << dynamic_cast<BulkElementBase *>(fe)->get_eleminfo()->alloced << std::endl;
        if (!be->get_eleminfo()->alloced) 
        {
          be->fill_element_info(true);
        }
        if (be->get_eleminfo()->alloced)
        {
          if (!dynamic_cast<BulkElementBase *>(fe)->get_eleminfo()->alloced) 
          {
            dynamic_cast<BulkElementBase *>(fe)->fill_element_info(true);
          }
          //std::cout << "RESTR " << dynamic_cast<BulkElementBase *>(fe)->get_eleminfo()->bulk_eleminfo << " ELEMINFO " << be->get_eleminfo() << " NODAL COORDS " << be->get_eleminfo()->nodal_coords << std::endl;
          double restriction = dynamic_cast<BulkElementBase *>(fe)->eval_local_expression_at_midpoint(restriction_index);
          if (restriction <= 0)
          {
            delete fe;
            continue;
          }
        }
      }

      if (!internal_facets)
      {
        fe->set_boundary_number_in_bulk_mesh(bind);
      }
      else
      {
        oomph::FaceElement *ofe;
        if (opposite_already_at_index[ei] >= 0)
        {
          ofe = generated_opposite_face_elems[opposite_already_at_index[ei]]; // Reuse the opposite face elements if multiple smaller elements share this one
        }
        else
        {
          ofe = gen_face_elem(opposite_elements[ei], opposite_face_dir[ei]);          
          dynamic_cast<InterfaceElementBase *>(ofe)->set_as_internal_facet_opposite_dummy();
        }
        generated_opposite_face_elems.push_back(ofe);
        dynamic_cast<InterfaceMesh *>(imesh)->opposite_interior_facets.push_back(ofe);
        dynamic_cast<InterfaceElementBase *>(fe)->set_opposite_interface_element(dynamic_cast<BulkElementBase *>(ofe),std::vector<double>());
      }

      imesh->add_element_pt(fe);
    }
    dynamic_cast<InterfaceMesh *>(imesh)->set_rebuild_information(this, intername, jitcode);
    dynamic_cast<InterfaceMesh *>(imesh)->setup_boundary_information(this);
    BulkElementBase::__CurrentCodeInstance = NULL;
  }

  std::map<std::string, std::string> Mesh::get_field_information() // first: names, second: list of spaces (C2,C1,DL,D0), but also (../C2 etc for elements defined on bulk domains)
  {
    if (!this->nelement())
      return std::map<std::string, std::string>();
    auto *el = dynamic_cast<BulkElementBase *>(this->element_pt(0));
    auto *ft = el->get_code_instance()->get_func_table();
    // auto *ci = el->get_code_instance();

    std::map<std::string, std::string> res;
    for (unsigned int i = 0; i < ft->numfields_DL; i++)
    {
      res[ft->fieldnames_DL[i]] = "DL";
    }
    for (unsigned int i = 0; i < ft->numfields_D0; i++)
    {
      res[ft->fieldnames_D0[i]] = "D0";
    }

    Mesh *current = this;
    std::string prefix = "";
    while (current)
    {
      auto *cel = dynamic_cast<BulkElementBase *>(current->element_pt(0));
      auto *cft = cel->get_code_instance()->get_func_table();
      if (!dynamic_cast<InterfaceMesh *>(current))
      {
        for (unsigned int i = 0; i < cft->numfields_C2TB_basebulk; i++)
        {
          res[cft->fieldnames_C2TB[i]] = prefix + "C2TB";
        }
        for (unsigned int i = 0; i < cft->numfields_C2_basebulk; i++)
        {
          res[cft->fieldnames_C2[i]] = prefix + "C2";
        }
        for (unsigned int i = 0; i < cft->numfields_C1TB_basebulk; i++)
        {
          res[cft->fieldnames_C1TB[i]] = prefix + "C1TB";
        }
        for (unsigned int i = 0; i < cft->numfields_C1_basebulk; i++)
        {
          res[cft->fieldnames_C1[i]] = prefix + "C1";
        }

        for (unsigned int i = 0; i < cft->numfields_D2TB_basebulk; i++)
        {
          res[cft->fieldnames_D2TB[i]] = prefix + "D2TB";
        }
        for (unsigned int i = 0; i < cft->numfields_D2_basebulk; i++)
        {
          res[cft->fieldnames_D2[i]] = prefix + "D2";
        }
        for (unsigned int i = 0; i < cft->numfields_D1_basebulk; i++)
        {
          res[cft->fieldnames_D1[i]] = prefix + "D1";
        }

        if (cft->moving_nodes)
        {
          for (unsigned int i = 0; i < cel->nodal_dimension(); i++)
          {
            std::vector<std::string> suffix = {"x", "y", "z"};
            res["mesh_" + suffix[i]] = prefix + std::string(cft->dominant_space);
          }
        }
        current = NULL;
      }
      else
      {
        for (unsigned int i = cft->numfields_C2TB_bulk; i < cft->numfields_C2TB; i++)
        {
          res[cft->fieldnames_C2TB[i]] = prefix + "C2TB";
        }
        for (unsigned int i = cft->numfields_C2_bulk; i < cft->numfields_C2; i++)
        {
          res[cft->fieldnames_C2[i]] = prefix + "C2";
        }
        for (unsigned int i = cft->numfields_C1TB_bulk; i < cft->numfields_C1TB; i++)
        {
          res[cft->fieldnames_C1TB[i]] = prefix + "C1TB";
        }
        for (unsigned int i = cft->numfields_C1_bulk; i < cft->numfields_C1; i++)
        {
          res[cft->fieldnames_C1[i]] = prefix + "C1";
        }

        for (unsigned int i = cft->numfields_D2TB_bulk; i < cft->numfields_D2TB; i++)
        {
          res[cft->fieldnames_D2TB[i]] = prefix + "D2TB";
        }
        for (unsigned int i = cft->numfields_D2_bulk; i < cft->numfields_D2; i++)
        {
          res[cft->fieldnames_D2[i]] = prefix + "D2";
        }
        for (unsigned int i = cft->numfields_D1_bulk; i < cft->numfields_D1; i++)
        {
          res[cft->fieldnames_D1[i]] = prefix + "D1";
        }

        current = dynamic_cast<InterfaceMesh *>(current)->get_bulk_mesh();
        prefix = "../" + prefix;
      }
    }

    return res;
  }

  void Mesh::pin_all_my_dofs(std::set<std::string> only_dofs, std::set<std::string> ignore_dofs, std::set<unsigned> ignore_continuous_at_interfaces)
  {
    // throw_runtime_error("Implement");
    if (!this->nelement())
      return;
    auto *el = dynamic_cast<BulkElementBase *>(this->element_pt(0));
    auto *ft = el->get_code_instance()->get_func_table();
    auto *ci = el->get_code_instance();
    auto mustpin = [&](std::string name)
    {
      if (only_dofs.empty())
      {
        return !ignore_dofs.count(name);
      }
      else
      {
        return only_dofs.count(name) && (!ignore_dofs.count(name));
      }
    };

    std::set<unsigned> posindices;
    std::vector<std::string> dir_suffix = {"x", "y", "z"};
    for (unsigned int i = 0; i < el->nodal_dimension(); i++)
    {
      if (mustpin("mesh_" + dir_suffix[i]))
        posindices.insert(i);
    }
    std::set<unsigned> valindices;
    for (unsigned int i = 0; i < ft->numfields_C2TB_basebulk; i++)
    {
      if (mustpin(ft->fieldnames_C2TB[i]))
        valindices.insert(i);
    }
    for (unsigned int i = 0; i < ft->numfields_C2_basebulk; i++)
    {
      if (mustpin(ft->fieldnames_C2[i]))
        valindices.insert(i + ft->numfields_C2TB_basebulk);
    }
    for (unsigned int i = 0; i < ft->numfields_C1TB_basebulk; i++)
    {
      if (mustpin(ft->fieldnames_C1TB[i]))
        valindices.insert(i + ft->numfields_C2TB_basebulk + ft->numfields_C2_basebulk);
    }
    for (unsigned int i = 0; i < ft->numfields_C1_basebulk; i++)
    {
      if (mustpin(ft->fieldnames_C1[i]))
        valindices.insert(i + ft->numfields_C2TB_basebulk + ft->numfields_C2_basebulk + ft->numfields_C1TB_basebulk);
    }
    std::set<unsigned> add_indices;
    for (unsigned int i = ft->numfields_C2TB_basebulk; i < ft->numfields_C2TB; i++)
    {
      if (mustpin(ft->fieldnames_C2TB[i]))
        add_indices.insert(ci->resolve_interface_dof_id(ft->fieldnames_C2TB[i]));
    }
    for (unsigned int i = ft->numfields_C2_basebulk; i < ft->numfields_C2; i++)
    {
      if (mustpin(ft->fieldnames_C2[i]))
        add_indices.insert(ci->resolve_interface_dof_id(ft->fieldnames_C2[i]));
    }
    for (unsigned int i = ft->numfields_C1TB_basebulk; i < ft->numfields_C1TB; i++)
    {
      if (mustpin(ft->fieldnames_C1TB[i]))
        add_indices.insert(ci->resolve_interface_dof_id(ft->fieldnames_C1TB[i]));
    }
    for (unsigned int i = ft->numfields_C1_basebulk; i < ft->numfields_C1; i++)
    {
      if (mustpin(ft->fieldnames_C1[i]))
        add_indices.insert(ci->resolve_interface_dof_id(ft->fieldnames_C1[i]));
    }

    std::set<unsigned> D2TBindices;
    for (unsigned int i = 0; i < ft->numfields_D2TB; i++)
    {
      if (mustpin(ft->fieldnames_D2TB[i]))
        D2TBindices.insert(i);
    }
    std::set<unsigned> D2indices;
    for (unsigned int i = 0; i < ft->numfields_D2; i++)
    {
      if (mustpin(ft->fieldnames_D2[i]))
        D2indices.insert(i);
    }
    std::set<unsigned> D1indices;
    for (unsigned int i = 0; i < ft->numfields_D1; i++)
    {
      if (mustpin(ft->fieldnames_D1[i]))
        D1indices.insert(i);
    }

    std::set<unsigned> DLindices;
    for (unsigned int i = ft->numfields_DL; i < ft->numfields_DL; i++)
    {
      if (mustpin(ft->fieldnames_DL[i]))
        DLindices.insert(i);
    }
    std::set<unsigned> D0indices;
    for (unsigned int i = 0; i < ft->numfields_D0; i++)
    {
      if (mustpin(ft->fieldnames_D0[i]))
      {
        D0indices.insert(i);
      }
    }

    for (unsigned ie = 0; ie < this->nelement(); ie++)
    {
      auto *el = dynamic_cast<BulkElementBase *>(this->element_pt(ie));
      // Conti fields
      for (unsigned int in = 0; in < el->nnode(); in++)
      {
        pyoomph::Node *n = dynamic_cast<pyoomph::Node *>(el->node_pt(in));
        for (unsigned ind : posindices)
          n->variable_position_pt()->pin(ind);
        for (unsigned ind : valindices)
        {
          bool must_pin_me = true;
          for (auto b : ignore_continuous_at_interfaces)
          {
            if (n->is_on_boundary(b))
            {
              must_pin_me = false;
              break;
            }
          }
          if (must_pin_me)
            n->pin(ind);
        }
        for (unsigned ind : add_indices)
        {
          bool must_pin_me = true;
          for (auto b : ignore_continuous_at_interfaces)
          {
            if (n->is_on_boundary(b))
            {
              must_pin_me = false;
              break;
            }
          }
          if (!must_pin_me)
            continue;
          int find = n->additional_value_index(ind);
          if (find < 0)
            throw_runtime_error("Missing additional entry in this node");
          n->pin(find);
        }
      }

      // DG fields
      for (unsigned ind : D2TBindices)
      {
        oomph::Data *dgdata = el->get_D2TB_nodal_data(ind);
        for (unsigned ni = 0; ni < el->get_eleminfo()->nnode_C2TB; ni++)
        {
          bool must_pin_me = true;
          oomph::Node *n = el->node_pt(el->get_node_index_C2TB_to_element(ni));
          for (auto b : ignore_continuous_at_interfaces)
          {
            if (n->is_on_boundary(b))
            {
              must_pin_me = false;
              break;
            }
          }
          if (!must_pin_me)
            continue;
          dgdata->pin(el->get_D2TB_node_index(ind, ni));
        }
      }

      for (unsigned ind : D2indices)
      {
        oomph::Data *dgdata = el->get_D2_nodal_data(ind);
        for (unsigned ni = 0; ni < el->get_eleminfo()->nnode_C2; ni++)
        {
          bool must_pin_me = true;
          oomph::Node *n = el->node_pt(el->get_node_index_C2_to_element(ni));
          for (auto b : ignore_continuous_at_interfaces)
          {
            if (n->is_on_boundary(b))
            {
              must_pin_me = false;
              break;
            }
          }
          if (!must_pin_me)
            continue;
          dgdata->pin(el->get_D2_node_index(ind, ni));
        }
      }

      for (unsigned ind : D1indices)
      {
        oomph::Data *dgdata = el->get_D1_nodal_data(ind);
        for (unsigned ni = 0; ni < el->get_eleminfo()->nnode_C1; ni++)
        {
          bool must_pin_me = true;
          oomph::Node *n = el->node_pt(el->get_node_index_C1_to_element(ni));
          for (auto b : ignore_continuous_at_interfaces)
          {
            if (n->is_on_boundary(b))
            {
              must_pin_me = false;
              break;
            }
          }
          if (!must_pin_me)
            continue;
          dgdata->pin(el->get_D1_node_index(ind, ni));
        }
      }

      for (unsigned ind : D0indices)
      {
        el->internal_data_pt(ind + ft->internal_offset_D0)->pin(0);
      }
      for (unsigned ind : DLindices)
      {
        for (unsigned v = 0; v < el->internal_data_pt(ind + ft->internal_offset_DL)->nvalue(); v++)
          el->internal_data_pt(ind + ft->internal_offset_DL)->pin(v);
      }
    }
  }

  void Mesh::fill_dof_types(int *typarr)
  {
    throw_runtime_error("Implement");
  }

  void Mesh::fill_node_map(std::map<oomph::Node *, unsigned> &nodemap)
  {
    for (unsigned int i = 0; i < this->nnode(); i++)
    {
      nodemap[this->node_pt(i)] = i;
    }
  }

  std::vector<oomph::Node *> Mesh::fill_reversed_node_map(bool discontinuous)
  {
    std::vector<oomph::Node *> result;
    result.reserve(this->nnode());
    if (discontinuous)
    {
      for (unsigned int ei = 0; ei < this->nelement(); ei++)
      {
        oomph::FiniteElement *el = dynamic_cast<oomph::FiniteElement *>(this->element_pt(ei));
        for (unsigned int en = 0; en < el->nnode(); en++)
        {
          result.push_back(el->node_pt(en));
        }
      }
    }
    else
    {
      for (unsigned int i = 0; i < this->nnode(); i++)
      {
        result.push_back(this->node_pt(i));
      }
    }
    return result;
  }

  unsigned Mesh::resolve_interface_dof_id(std::string n)
  {
    if (!interface_dof_ids.count(n))
    {
      interface_dof_ids[n] = interface_dof_ids.size();
    }
    return interface_dof_ids[n];
  }

  int Mesh::has_interface_dof_id(std::string n)
  {
    if (!interface_dof_ids.count(n))
    {
      return -1;
    }
    return interface_dof_ids[n];
  }

  void Mesh::_set_problem(Problem *p, DynamicBulkElementInstance *code)
  {
    problem = p;
    #ifdef OOMPH_HAS_MPI
    //This only for distributed meshes
    /*if (p->is_distributed())
    {
      this->set_communicator_pt(p->communicator_pt());
    }*/
    #endif
    codeinst = code;
    if (code && dirichlet_active.empty())
    {
      dirichlet_active.resize(code->get_func_table()->Dirichlet_set_size, false);
      for (unsigned int i = 0; i < code->get_func_table()->Dirichlet_set_size; i++)
      {
        //    std::cout << "SETTING " << code->get_code()->get_file_name() << " INDEX " << i << " to "  << (code->get_func_table()->Dirichlet_set[i] ? "true" : "false") << std::endl;
        dirichlet_active[i] = code->get_func_table()->Dirichlet_set[i];
      }
    }
  }

  std::vector<std::vector<double>> Mesh::get_values_at_zetas(const std::vector<std::vector<double>> &zetas, std::vector<bool> &masked_lines, bool with_scales)
  {
    auto *el = dynamic_cast<BulkElementBase *>(this->element_pt(0));
    auto *ft = el->get_code_instance()->get_func_table();
    unsigned numfields = el->nodal_dimension() + ft->numfields_C2TB + ft->numfields_C2 + ft->numfields_C1TB + ft->numfields_C1 + ft->numfields_DL + ft->numfields_D0;
    std::vector<std::vector<double>> result(zetas.size(), std::vector<double>(numfields, 0.0));

    double spatial_scale = (with_scales && output_scales.count("spatial") ? output_scales["spatial"] : 1.0);
    std::vector<double> scales(numfields, 1.0);
    if (with_scales)
    {
      for (auto &fi : el->get_code_instance()->get_nodal_field_indices())
      {
        scales[fi.second] = (output_scales.count(fi.first) ? output_scales[fi.first] : 1.0);
      }
      for (auto &fi : el->get_code_instance()->get_elemental_field_indices())
      {
        scales[ft->numfields_C2TB + ft->numfields_C2 + ft->numfields_C1TB + ft->numfields_C1 + fi.second] = (output_scales.count(fi.first) ? output_scales[fi.first] : 1.0);
      }
    }

    masked_lines.resize(zetas.size(), false);
    oomph::MeshAsGeomObject MaGO(this);
    for (unsigned int zi = 0; zi < zetas.size(); zi++)
    {
      oomph::Vector<double> zet(zetas[zi].size());
      for (unsigned int j = 0; j < zetas[zi].size(); j++)
      {
        zet[j] = zetas[zi][j];
      }
      zet.resize(el->dim(), 0.0);

      oomph::GeomObject *res_go = NULL;
      oomph::Vector<double> s(el->dim(), 1.0 / 3.0);
      MaGO.locate_zeta(zet, res_go, s, false);
      BulkElementBase *srcelem = dynamic_cast<BulkElementBase *>(res_go);
      if (!srcelem)
        masked_lines[zi] = true;
      else
      {
        masked_lines[zi] = false;
        std::vector<double> C2, C1, DL, D0;
        oomph::Vector<double> xpos(el->nodal_dimension(), 0.0);
        srcelem->interpolated_x(0, s, xpos);
        srcelem->get_interpolated_fields_C2(s, C2, 0);
        srcelem->get_interpolated_fields_C1(s, C1, 0);
        srcelem->get_interpolated_fields_DL(s, DL, 0);
        srcelem->get_interpolated_fields_D0(s, D0, 0);
        //    result[zi].resize(xpos.size()+C2.size()+C1.size());
        for (unsigned int j = 0; j < xpos.size(); j++)
          result[zi][j] = spatial_scale * xpos[j];
        for (unsigned int j = 0; j < C2.size(); j++)
          result[zi][xpos.size() + j] = scales[j] * C2[j];
        for (unsigned int j = 0; j < C1.size(); j++)
          result[zi][xpos.size() + C2.size() + j] = scales[j + C2.size()] * C1[j];
        for (unsigned int j = 0; j < DL.size(); j++)
          result[zi][xpos.size() + C2.size() + C1.size() + j] = scales[j + C2.size() + C1.size()] * DL[j];
        for (unsigned int j = 0; j < D0.size(); j++)
          result[zi][xpos.size() + C2.size() + C1.size() + DL.size() + j] = scales[j + C2.size() + C1.size() + DL.size()] * D0[j];
      }
    }

    return result;
  }

  std::vector<double> Mesh::evaluate_local_expression_at_nodes(unsigned index, bool nondimensional, bool discontinuous)
  {
    std::map<oomph::Node *, unsigned> nodemap;
    this->fill_node_map(nodemap);
    std::vector<double> res(nodemap.size(), 0.0);
    std::vector<double> denom(nodemap.size(), 0.0);

    unsigned cnt = 0;
    for (unsigned int ne = 0; ne < this->nelement(); ne++)
    {
      BulkElementBase *e = dynamic_cast<BulkElementBase *>(this->element_pt(ne));
      for (unsigned int nn = 0; nn < e->nnode(); nn++)
      {
        double add = e->eval_local_expression_at_node(index, nn);
        // if (denom[nindex]>0.2)  std::cout << " REEVAL EXPR. NEW " << add << "  OLD " << res[nindex]/denom[nindex] << " based on " << denom[nindex]  << std::endl;
        if (discontinuous)
        {
          if (cnt >= res.size())
          {
            res.push_back(add);
          }
          else
          {
            res[cnt] = add;
          }
          cnt++;
        }
        else
        {
          unsigned nindex = nodemap[e->node_pt(nn)];
          res[nindex] += add;
          denom[nindex] += 1.0;
        }
      }
    }
    // normalize
    std::string exprname = this->list_local_expressions()[index];
    double scale = (output_scales.count(exprname) && (!nondimensional) ? output_scales[exprname] : 1.0);

    if (discontinuous)
    {
      for (unsigned int ni = 0; ni < res.size(); ni++)
      {
        res[ni] *= scale;
      }
    }
    else
    {
      for (unsigned int ni = 0; ni < nodemap.size(); ni++)
      {
        if (denom[ni] > 0)
          res[ni] *= scale / denom[ni];
      }
    }
    return res;
  }

  void Mesh::to_numpy(double *xbuffer, int *eleminds, unsigned elemstride, int *elemtypes, bool tesselate_tri, bool nondimensional, double *D0_data, double *DL_data, unsigned history_index, bool discontinuous)
  {
    // unsigned nnode=this->count_nnode();
    pyoomph::Node *node0 = this->get_some_node();
    unsigned nodal_dim = node0->ndim();
    BulkElementBase *be = dynamic_cast<BulkElementBase *>(this->element_pt(0));
    unsigned nlagrangian = node0->nlagrangian();
    unsigned nelement = this->nelement();
    auto *ft = be->get_code_instance()->get_func_table();
    for (unsigned int i = 0; i < nelement; i++)
    {
      dynamic_cast<BulkElementBase *>(this->element_pt(i))->interpolate_hang_values();
    }
    // pyoomph::DynamicBulkElementInstance * ci=be->get_code_instance();
    unsigned ncontfields = be->ncont_interpolated_values();
    unsigned nDGfields = (be ? be->num_DG_fields(false) : 0);
    unsigned nDGfields_basebulk = (be ? be->num_DG_fields(true) : 0);

    unsigned naddC1 = be->nadditional_fields_C1();
    unsigned naddC1TB = be->nadditional_fields_C1TB();
    unsigned naddC2 = be->nadditional_fields_C2();
    unsigned naddC2TB = be->nadditional_fields_C2TB();
    unsigned naddD1 = ft->numfields_D1 - ft->numfields_D1_basebulk;
    unsigned naddD1TB = ft->numfields_D1TB - ft->numfields_D1TB_basebulk;
    unsigned naddD2 = ft->numfields_D2 - ft->numfields_D2_basebulk;
    unsigned naddD2TB = ft->numfields_D2TB - ft->numfields_D2TB_basebulk;

    //    std::cout << "MESHOUT " << nDGfields << "  " << nDGfields_basebulk << "   " << naddD1 << "  " << naddD2 << "  " << naddD2TB << "   consistenccy "  << nDGfields-(nDGfields_basebulk+naddD1+naddD2+naddD2TB) << std::endl;

    unsigned nnormal = 0;
    if (be->nodal_dimension() == be->dim() + 1) // TODO: Also >= ? But what is e.g. the normal of a curved line element in 3d space? Is it the tangent?
    //													XXX MAKE SURE TO ADJUST IT ALSO IN pybind -> to_numpy and in python/output/generic.py indicated by "TODO must agree with the C code"
    {
      nnormal = be->nodal_dimension();
    }

    unsigned contstride = nodal_dim + nlagrangian + ncontfields + nDGfields + naddC1 + naddC1TB + naddC2 + naddC2TB + nnormal;
    double spatial_scale = (output_scales.count("spatial") && (!nondimensional) ? output_scales["spatial"] : 1.0);
    std::vector<double> nodal_scales(ncontfields + nDGfields + naddC1 + naddC1TB + naddC2 + naddC2TB + nnormal, 1.0);
    for (auto &fi : be->get_code_instance()->get_nodal_field_indices())
    {
      nodal_scales[fi.second] = (output_scales.count(fi.first) && (!nondimensional) ? output_scales[fi.first] : 1.0);
    }
    std::vector<int> add_C2TB(naddC2TB);
    std::vector<double> add_C2TB_scales(naddC2TB, 1.0);
    for (unsigned int i = 0; i < naddC2TB; i++)
    {
      std::string fn = ft->fieldnames_C2TB[i + ft->numfields_C2TB_basebulk];
      add_C2TB[i] = this->has_interface_dof_id(fn);
      if (add_C2TB[i] < 0)
        throw_runtime_error("Something is wrong with the interface field " + fn);
      add_C2TB_scales[i] = (output_scales.count(fn) && (!nondimensional) ? output_scales[fn] : 1.0);
    }

    std::vector<int> add_C2(naddC2);
    std::vector<double> add_C2_scales(naddC2, 1.0);
    for (unsigned int i = 0; i < naddC2; i++)
    {
      std::string fn = ft->fieldnames_C2[i + ft->numfields_C2_basebulk];
      add_C2[i] = this->has_interface_dof_id(fn);
      if (add_C2[i] < 0)
        throw_runtime_error("Something is wrong with the interface field " + fn);
      add_C2_scales[i] = (output_scales.count(fn) && (!nondimensional) ? output_scales[fn] : 1.0);
    }
    std::vector<int> add_C1(naddC1);
    std::vector<double> add_C1_scales(naddC1, 1.0);
    for (unsigned int i = 0; i < naddC1; i++)
    {
      std::string fn = ft->fieldnames_C1[i + ft->numfields_C1_basebulk];
      add_C1[i] = this->has_interface_dof_id(fn);
      if (add_C1[i] < 0)
        throw_runtime_error("Something is wrong with the interface field " + fn);
      add_C1_scales[i] = (output_scales.count(fn) && (!nondimensional) ? output_scales[fn] : 1.0);
    }
    std::vector<int> add_C1TB(naddC1TB);
    std::vector<double> add_C1TB_scales(naddC1TB, 1.0);
    for (unsigned int i = 0; i < naddC1TB; i++)
    {
      std::string fn = ft->fieldnames_C1TB[i + ft->numfields_C1TB_basebulk];
      add_C1TB[i] = this->has_interface_dof_id(fn);
      if (add_C1TB[i] < 0)
        throw_runtime_error("Something is wrong with the interface field " + fn);
      add_C1TB_scales[i] = (output_scales.count(fn) && (!nondimensional) ? output_scales[fn] : 1.0);
    }

    std::map<oomph::Node *, unsigned> nodemap;
    this->fill_node_map(nodemap);
    std::vector<oomph::Node *> rev_nodemap = this->fill_reversed_node_map(discontinuous);

    std::vector<double> D2TB_scales(ft->numfields_D2TB, 1.0);
    for (unsigned int i = 0; i < ft->numfields_D2TB; i++)
    {
      std::string fn = ft->fieldnames_D2TB[i];
      D2TB_scales[i] = (output_scales.count(fn) && (!nondimensional) ? output_scales[fn] : 1.0);
    }
    std::vector<double> D2_scales(ft->numfields_D2, 1.0);
    for (unsigned int i = 0; i < ft->numfields_D2; i++)
    {
      std::string fn = ft->fieldnames_D2[i];
      D2_scales[i] = (output_scales.count(fn) && (!nondimensional) ? output_scales[fn] : 1.0);
    }
    std::vector<double> D1TB_scales(ft->numfields_D1TB, 1.0);
    for (unsigned int i = 0; i < ft->numfields_D1TB; i++)
    {
      std::string fn = ft->fieldnames_D1TB[i];
      D1TB_scales[i] = (output_scales.count(fn) && (!nondimensional) ? output_scales[fn] : 1.0);
    }
    std::vector<double> D1_scales(ft->numfields_D1, 1.0);
    for (unsigned int i = 0; i < ft->numfields_D1; i++)
    {
      std::string fn = ft->fieldnames_D1[i];
      D1_scales[i] = (output_scales.count(fn) && (!nondimensional) ? output_scales[fn] : 1.0);
    }

    for (unsigned int ni = 0; ni < rev_nodemap.size(); ni++)
    {

      pyoomph::Node *node = dynamic_cast<pyoomph::Node *>(rev_nodemap[ni]);
      for (unsigned nd = 0; nd < nodal_dim; nd++)
      {
        xbuffer[ni * contstride + nd] = node->position(history_index, nd) * spatial_scale;
      }
      for (unsigned nd = 0; nd < nlagrangian; nd++)
      {
        xbuffer[ni * contstride + nd + nodal_dim] = node->xi(nd) * spatial_scale;
      }

      for (unsigned nd = 0; nd < ncontfields; nd++)
      {
        xbuffer[ni * contstride + nd + nodal_dim + nlagrangian] = node->value(history_index, nd) * nodal_scales[nd];
      }

      for (unsigned nd = 0; nd < naddC2TB; nd++)
      {
        int ind = node->additional_value_index(add_C2TB[nd]);
        if (ind < 0)
          throw_runtime_error("Missing additional entry in this node");
        xbuffer[ni * contstride + nd + ncontfields + nDGfields_basebulk + nodal_dim + nlagrangian] = node->value(history_index, ind) * add_C2TB_scales[nd];
      }

      for (unsigned nd = 0; nd < naddC2; nd++)
      {
        int ind = node->additional_value_index(add_C2[nd]);
        if (ind < 0)
          throw_runtime_error("Missing additional entry in this node");
        xbuffer[ni * contstride + nd + ncontfields + nDGfields_basebulk + naddC2TB + nodal_dim + nlagrangian] = node->value(history_index, ind) * add_C2_scales[nd];
      }
      for (unsigned nd = 0; nd < naddC1TB; nd++)
      {
        int ind = node->additional_value_index(add_C1TB[nd]);
        if (ind < 0)
          throw_runtime_error("Missing additional entry in this node");
        xbuffer[ni * contstride + nd + ncontfields + nDGfields_basebulk + naddC2TB + naddC2 + nodal_dim + nlagrangian] = node->value(history_index, ind) * add_C1TB_scales[nd];
      }
      for (unsigned nd = 0; nd < naddC1; nd++)
      {
        int ind = node->additional_value_index(add_C1[nd]);
        if (ind < 0)
          throw_runtime_error("Missing additional entry in this node");
        xbuffer[ni * contstride + nd + ncontfields + nDGfields_basebulk + naddC2TB + naddC2 + naddC1TB + nodal_dim + nlagrangian] = node->value(history_index, ind) * add_C1_scales[nd];
      }
    }

    // DG fields and normals by averaging contributions
    if (nnormal || nDGfields)
    {
      unsigned interface_DG_fields_offset = nDGfields_basebulk + naddC2TB + naddC2 + naddC1TB + naddC1;
      if (!discontinuous)
      {
        // Fill be zero
        for (unsigned int ni = 0; ni < rev_nodemap.size(); ni++)
        {
          if (nnormal)
          {
            for (unsigned nd = 0; nd < be->nodal_dimension(); nd++)
            {
              xbuffer[ni * contstride + nd + ncontfields + nDGfields + naddC2TB + naddC2 + naddC1TB + naddC1 + nodal_dim + nlagrangian] = 0.0;
            }
          }
          for (unsigned nd = 0; nd < ft->numfields_D2TB_basebulk + ft->numfields_D2_basebulk + ft->numfields_D1_basebulk; nd++)
          {
            xbuffer[ni * contstride + nd + ncontfields + nodal_dim + nlagrangian] = 0.0;
          }
          for (unsigned nd = 0; nd < naddD2TB + naddD2 + naddD1TB + naddD1; nd++)
          {
            xbuffer[ni * contstride + nd + ncontfields + nodal_dim + nlagrangian + interface_DG_fields_offset] = 0.0;
          }
        }
        std::vector<double> dg_denom(nodemap.size());
        for (unsigned int ne = 0; ne < nelement; ne++)
        {
          BulkElementBase *e = dynamic_cast<BulkElementBase *>(this->element_pt(ne));
          for (unsigned int nn = 0; nn < e->nnode(); nn++)
          {
            oomph::Node *n = e->node_pt(nn);
            dg_denom[nodemap[n]]++;
            oomph::Vector<double> sn(e->dim());
            e->local_coordinate_of_node(nn, sn);
            if (nnormal)
            {
              oomph::Vector<double> normal(be->nodal_dimension());
              e->get_normal_at_s(sn, normal, NULL, NULL);
              for (unsigned nd = 0; nd < normal.size(); nd++)
              {
                xbuffer[nodemap[n] * contstride + nd + ncontfields + nDGfields + naddC2TB + naddC2 + naddC1TB + naddC1 + nodal_dim + nlagrangian] += normal[nd];
              }
            }
            if (ft->numfields_D2TB)
            {
              oomph::Vector<double> DGdata;
              e->get_D2TB_fields_at_s(history_index, sn, DGdata);
              for (unsigned nd = 0; nd < ft->numfields_D2TB; nd++)
              {
                unsigned offs = (nd < ft->numfields_D2TB_basebulk ? 0 : interface_DG_fields_offset - ft->numfields_D2TB_basebulk);
                xbuffer[nodemap[n] * contstride + nd + ncontfields + nodal_dim + nlagrangian + offs] += DGdata[nd] * D2TB_scales[nd];
              }
            }
            if (ft->numfields_D2)
            {
              oomph::Vector<double> DGdata;
              e->get_D2_fields_at_s(history_index, sn, DGdata);
              for (unsigned nd = 0; nd < ft->numfields_D2; nd++)
              {
                unsigned offs = (nd < ft->numfields_D2_basebulk ? ft->numfields_D2TB_basebulk : interface_DG_fields_offset + naddD2TB - ft->numfields_D2_basebulk);
                xbuffer[nodemap[n] * contstride + nd + ncontfields + nodal_dim + nlagrangian + offs] += DGdata[nd] * D2_scales[nd];
              }
            }
            if (ft->numfields_D1TB)
            {
              oomph::Vector<double> DGdata;
              e->get_D1TB_fields_at_s(history_index, sn, DGdata);
              for (unsigned nd = 0; nd < ft->numfields_D1TB; nd++)
              {
                unsigned offs = (nd < ft->numfields_D1TB_basebulk ? ft->numfields_D2TB_basebulk + ft->numfields_D2_basebulk : interface_DG_fields_offset + naddD2TB + naddD2 - ft->numfields_D1TB_basebulk);
                xbuffer[nodemap[n] * contstride + nd + ncontfields + nodal_dim + nlagrangian + offs] += DGdata[nd] * D1TB_scales[nd];
              }
            }

            if (ft->numfields_D1)
            {
              oomph::Vector<double> DGdata;
              e->get_D1_fields_at_s(history_index, sn, DGdata);
              for (unsigned nd = 0; nd < ft->numfields_D1; nd++)
              {
                unsigned offs = (nd < ft->numfields_D1_basebulk ? ft->numfields_D2TB_basebulk + ft->numfields_D2_basebulk + ft->numfields_D1TB_basebulk : interface_DG_fields_offset + naddD2TB + naddD2 + naddD1TB - ft->numfields_D1_basebulk);
                xbuffer[nodemap[n] * contstride + nd + ncontfields + nodal_dim + nlagrangian + offs] += DGdata[nd] * D1_scales[nd];
              }
            }
          }
        }
        if (nnormal)
        {
          // normalize
          for (unsigned int ni = 0; ni < nodemap.size(); ni++)
          {
            double nl = 0.0;
            for (unsigned nd = 0; nd < be->nodal_dimension(); nd++)
            {
              double nc = xbuffer[ni * contstride + nd + ncontfields + nDGfields + naddC2TB + naddC2 + naddC1TB + naddC1 + nodal_dim + nlagrangian];
              nl += nc * nc;
            }
            if (nl < 1e-40)
              nl = 0;
            else
              nl = 1.0 / sqrt(nl);
            for (unsigned nd = 0; nd < be->nodal_dimension(); nd++)
            {
              xbuffer[ni * contstride + nd + ncontfields + naddC2TB + nDGfields + naddC2 + naddC1TB + naddC1 + nodal_dim + nlagrangian] *= nl;
            }
          }
        }
        for (unsigned int ni = 0; ni < nodemap.size(); ni++)
        {
          for (unsigned nd = 0; nd < ft->numfields_D2TB_basebulk + ft->numfields_D2_basebulk + ft->numfields_D1_basebulk; nd++)
          {
            xbuffer[ni * contstride + nd + ncontfields + nodal_dim + nlagrangian] /= dg_denom[ni];
          }
          for (unsigned nd = 0; nd < naddD1 + naddD2 + naddD1TB + naddD2TB; nd++)
          {
            xbuffer[ni * contstride + nd + ncontfields + nodal_dim + nlagrangian + interface_DG_fields_offset] /= dg_denom[ni];
          }
        }
      }
      // Discontinuous mode
      else
      {
        unsigned cnt = 0;
        for (unsigned int ne = 0; ne < nelement; ne++)
        {
          BulkElementBase *e = dynamic_cast<BulkElementBase *>(this->element_pt(ne));
          for (unsigned int nn = 0; nn < e->nnode(); nn++)
          {
            oomph::Vector<double> sn(e->dim());
            e->local_coordinate_of_node(nn, sn);
            if (nnormal)
            {
              oomph::Vector<double> normal(be->nodal_dimension());
              e->get_normal_at_s(sn, normal, NULL, NULL);
              for (unsigned nd = 0; nd < normal.size(); nd++)
              {
                xbuffer[cnt * contstride + nd + ncontfields + nDGfields + naddC2TB + naddC2 + naddC1TB + naddC1 + nodal_dim + nlagrangian] = normal[nd];
              }
            }
            if (ft->numfields_D2TB)
            {
              oomph::Vector<double> DGdata;
              e->get_D2TB_fields_at_s(history_index, sn, DGdata);
              for (unsigned nd = 0; nd < ft->numfields_D2TB; nd++)
              {
                unsigned offs = (nd < ft->numfields_D2TB_basebulk ? 0 : interface_DG_fields_offset - ft->numfields_D2TB_basebulk);
                xbuffer[cnt * contstride + nd + ncontfields + offs + nodal_dim + nlagrangian] = DGdata[nd] * D2TB_scales[nd];
              }
            }
            if (ft->numfields_D2)
            {
              oomph::Vector<double> DGdata;
              e->get_D2_fields_at_s(history_index, sn, DGdata);
              for (unsigned nd = 0; nd < ft->numfields_D2; nd++)
              {
                unsigned offs = (nd < ft->numfields_D2_basebulk ? ft->numfields_D2TB_basebulk : interface_DG_fields_offset + naddD2TB - ft->numfields_D2_basebulk);
                //              std::cout << "WRITING D2 " << nd << " value " <<DGdata[nd] <<" to " <<  nd + ncontfields +nodal_dim + nlagrangian +offs << std::endl;
                xbuffer[cnt * contstride + nd + ncontfields + nodal_dim + nlagrangian + offs] = DGdata[nd] * D2_scales[nd];
              }
            }
            if (ft->numfields_D1TB)
            {
              oomph::Vector<double> DGdata;
              e->get_D1TB_fields_at_s(history_index, sn, DGdata);
              for (unsigned nd = 0; nd < ft->numfields_D1TB; nd++)
              {
                unsigned offs = (nd < ft->numfields_D1TB_basebulk ? ft->numfields_D2TB_basebulk + ft->numfields_D2_basebulk : interface_DG_fields_offset + naddD2TB - ft->numfields_D2_basebulk);
                //              std::cout << "WRITING D2 " << nd << " value " <<DGdata[nd] <<" to " <<  nd + ncontfields +nodal_dim + nlagrangian +offs << std::endl;
                xbuffer[cnt * contstride + nd + ncontfields + nodal_dim + nlagrangian + offs] = DGdata[nd] * D1TB_scales[nd];
              }
            }

            if (ft->numfields_D1)
            {
              oomph::Vector<double> DGdata;
              e->get_D1_fields_at_s(history_index, sn, DGdata);
              for (unsigned nd = 0; nd < ft->numfields_D1; nd++)
              {
                unsigned offs = (nd < ft->numfields_D1_basebulk ? ft->numfields_D2TB_basebulk + ft->numfields_D2_basebulk + ft->numfields_D1TB_basebulk : interface_DG_fields_offset + naddD2TB + naddD2 + naddD1TB - ft->numfields_D1_basebulk);
                //                         std::cout << "WRITING D1 " << nd << " VALUE " <<DGdata[nd] <<" to " <<  nd + ncontfields +nodal_dim + nlagrangian +offs << std::endl;
                xbuffer[cnt * contstride + nd + ncontfields + nodal_dim + nlagrangian + offs] = DGdata[nd] * D1_scales[nd];
              }
            }
            cnt++;
          }
        }
      }
    }

    unsigned current_subelem = 0;
    unsigned numD0 = be->get_code_instance()->get_func_table()->numfields_D0;
    unsigned numDL = be->get_code_instance()->get_func_table()->numfields_DL;

    std::vector<double> D_scales(numDL + numD0, 1.0);
    for (auto &fi : be->get_code_instance()->get_elemental_field_indices())
    {
      D_scales[fi.second] = (output_scales.count(fi.first) && (!nondimensional) ? output_scales[fi.first] : 1.0);
    }

    unsigned DL_stride = (be->dim() + 1);

    // Additional nodes due to different refinements:
    for (unsigned int ne = 0; ne < nelement; ne++)
    {
      dynamic_cast<BulkElementBase *>(this->element_pt(ne))->_numpy_index = ne;
    }
    std::vector<std::vector<std::set<oomph::Node *>>> additional_elemental_tri_nodes(nelement);
    if (tesselate_tri && !discontinuous)
    {
      //      if (discontinuous) throw_runtime_error("Cannot make use tesselate_tri and discontinuous together for Mesh::to_numpy yet");
      unsigned milev = 0, malev = 0;
      oomph::TreeBasedRefineableMeshBase *tbself = dynamic_cast<oomph::TreeBasedRefineableMeshBase *>(this);
      if (tbself)
        tbself->get_refinement_levels(milev, malev);
      if (milev < malev)
      {
        for (unsigned int ne = 0; ne < nelement; ne++)
        {
          dynamic_cast<BulkElementBase *>(this->element_pt(ne))->inform_coarser_neighbors_for_tesselated_numpy(additional_elemental_tri_nodes);
        }
      }
    }

    unsigned ncnt = 0;
    for (unsigned int ne = 0; ne < nelement; ne++)
    {
      BulkElementBase *be = dynamic_cast<BulkElementBase *>(this->element_pt(ne));
      elemtypes[ne] = be->get_meshio_type_index();
      unsigned nsubelem = 0;

      unsigned nindices = be->get_num_numpy_elemental_indices(tesselate_tri, nsubelem, additional_elemental_tri_nodes); // nindices
      std::vector<unsigned> local_ni_to_elemindex;
      for (unsigned isubelem = 0; isubelem < nsubelem; isubelem++)
      {
        // TODO: This could be reworked: Write all subelements simultaneously => Better performance for the cases where e.g. a split is done
        //		Or: Alternatively: Store the splitting in a global variable, since they are read directly
        be->fill_element_nodal_indices_for_numpy(&(eleminds[elemstride * current_subelem]), isubelem, tesselate_tri, additional_elemental_tri_nodes);
        std::vector<unsigned> local_nindices;
        unsigned int index = 0;
        for (unsigned iind = 0; iind < nindices; iind++)
        {
          oomph::Node *thenode = NULL;
          index = eleminds[elemstride * current_subelem + iind];
          if (index < be->nnode())
          {
            local_nindices.push_back(index);
            thenode = be->node_pt(index);
          }
          else // It must be an addtional node inserted from a finer element
          {
            if (discontinuous)
              throw_runtime_error("Should not end up here:  index of subelem: " + std::to_string(isubelem) + " node index: " + std::to_string(index) + " nelem:" + std::to_string(be->nnode()) + "  current index:" + std::to_string(iind) + "  nindices:" + std::to_string(nindices));
            unsigned cnt = be->nnode();
            for (unsigned int d = 0; d < additional_elemental_tri_nodes[ne].size(); d++)
            {
              for (auto *addnode : additional_elemental_tri_nodes[ne][d])
              {
                if (cnt == index)
                {
                  thenode = addnode;
                  break;
                }
                cnt++;
              }
              if (thenode)
                break;
            }
          }
          eleminds[elemstride * current_subelem + iind] = (discontinuous ? ncnt + index : nodemap[thenode]); // XXX This won't work for discontinuous and tesselate_tri
        }
        // Clear the rest of the buffer to -1
        for (unsigned int iind = nindices; iind < elemstride; iind++)
          eleminds[elemstride * current_subelem + iind] = -1;

        if (!discontinuous)
        {
          std::vector<double> elemental_D0(numD0);
          std::vector<double> elemental_DL(numDL * DL_stride);
          for (unsigned iDL = 0; iDL < numDL; iDL++)
          {
            for (unsigned int i = 0; i < DL_stride; i++)
            {
              elemental_DL[iDL * DL_stride + i] = be->internal_data_pt(iDL)->value(history_index, i) * D_scales[iDL];
            }
          }
          for (unsigned iD0 = 0; iD0 < numD0; iD0++)
          {
            elemental_D0[iD0] = be->internal_data_pt(numDL + iD0)->value(history_index, 0) * D_scales[numDL + iD0];
          }
          for (unsigned iD0 = 0; iD0 < numD0; iD0++)
          {
            *D0_data = elemental_D0[iD0];
            D0_data++;
          }

          for (unsigned iDL = 0; iDL < numDL; iDL++)
          {
            for (unsigned int i = 0; i < DL_stride; i++)
            {
              *DL_data = elemental_DL[iDL * DL_stride + i];
              DL_data++;
            }
          }
        }
        else
        {
          // if (nsubelem!=1) throw_runtime_error("Should not have nsubelem!=1 ("+std::to_string(nsubelem)+") here");
          for (unsigned int in = 0; in < local_nindices.size(); in++)
          {
            oomph::Vector<double> Dvalues;
            oomph::Vector<double> snodal(be->dim());
            be->local_coordinate_of_node(local_nindices[in], snodal);
            be->get_interpolated_discontinuous_values(history_index, snodal, Dvalues);
            unsigned nindex = eleminds[elemstride * current_subelem + in];
            for (unsigned iDL = 0; iDL < numDL; iDL++)
            {
              DL_data[nindex * numDL + iDL] = Dvalues[iDL] * D_scales[iDL];
            }
            for (unsigned iD0 = 0; iD0 < numD0; iD0++)
            {
              D0_data[nindex * numD0 + iD0] = Dvalues[numDL + iD0] * D_scales[numDL + iD0];
              //             *D0_data = Dvalues[numDL+iD0] * D_scales[numDL+iD0];
              //             D0_data++;
            }
          }
        }
        current_subelem++;
      }
      if (discontinuous)
        ncnt += be->nnode();
    }

    // Apply the scaling
  }

  double Mesh::get_temporal_error_norm_contribution()
  {
    if (!this->nelement())
      return 0.0;
    BulkElementBase *be = dynamic_cast<BulkElementBase *>(this->element_pt(0));
    DynamicBulkElementInstance *ci = be->get_code_instance();
    auto *ft = ci->get_func_table();
    if (!ft->has_temporal_estimators)
      return 0.0;
    double res = 0.0;
    double denom = 0.0;
    unsigned nnode = this->nnode();
    unsigned numcontifields = ft->numfields_C2TB_basebulk + ft->numfields_C2_basebulk + ft->numfields_C1TB_basebulk + ft->numfields_C1_basebulk; // TODO: Interface time errors
    for (unsigned int i = 0; i < numcontifields; i++)
    {
      if (ft->temporal_error_scales[i] == 0.0)
        continue;
      for (unsigned n = 0; n < nnode; n++)
      {
        if (!this->node_pt(n)->is_pinned(i))
        {
          double nodal_err = this->node_pt(n)->time_stepper_pt()->temporal_error_in_value(this->node_pt(n), i);
          res += nodal_err * nodal_err * ft->temporal_error_scales[i];
          denom += 1.0;
        }
      }
    }
    for (unsigned int i = 0; i < ft->numfields_DL; i++)
    {
      if (ft->temporal_error_scales[i + ft->numfields_C2TB + ft->numfields_C2 + ft->numfields_C1TB + ft->numfields_C1] == 0.0)
        continue;
      for (unsigned int j = 0; j < this->nelement(); j++)
      {
        BulkElementBase *be = dynamic_cast<BulkElementBase *>(this->element_pt(j));
        oomph::Data *d = be->internal_data_pt(i);
        for (unsigned int v = 0; v < d->nvalue(); v++)
        {
          double derr = d->time_stepper_pt()->temporal_error_in_value(d, v);
          res += derr * derr * ft->temporal_error_scales[i + ft->numfields_C2TB + ft->numfields_C2 + ft->numfields_C1TB + ft->numfields_C1];
          denom += 1.0;
        }
      }
    }
    for (unsigned int i = 0; i < ft->numfields_D0; i++)
    {
      if (ft->temporal_error_scales[i + ft->numfields_C2TB + ft->numfields_C2 + ft->numfields_C1 + ft->numfields_C1TB + ft->numfields_DL] == 0.0)
        continue;
      for (unsigned int j = 0; j < this->nelement(); j++)
      {
        BulkElementBase *be = dynamic_cast<BulkElementBase *>(this->element_pt(j));
        oomph::Data *d = be->internal_data_pt(i + ft->numfields_DL);
        double derr = d->time_stepper_pt()->temporal_error_in_value(d, 0);
        res += derr * derr * ft->temporal_error_scales[i + ft->numfields_C2TB + ft->numfields_C2 + ft->numfields_C1TB + ft->numfields_C1 + ft->numfields_DL];
        denom += 1.0;
      }
    }
    //	std::cout << " RESDENOM " << res << " " << denom << std::endl;
    // TODO: Discont
    if (denom == 0)
      return 0.0;
    return res / denom;
  }

  void Mesh::set_lagrangian_nodal_coordinates()
  {
    std::cout << "Setting Lagrangian nodal coordinates for all nodes in mesh" << std::endl;
    unsigned long n_node = nnode();
    for (unsigned n = 0; n < n_node; n++)
    {
      Node *node_pt = static_cast<Node *>(Node_pt[n]);
      unsigned n_lagrangian = node_pt->nlagrangian();
      unsigned n_lagrangian_type = node_pt->nlagrangian_type();
      for (unsigned k = 0; k < n_lagrangian_type; k++)
      {
        for (unsigned j = 0; j < n_lagrangian; j++)
        {
          node_pt->xi_gen(k, j) = node_pt->x_gen(k, j);
        }
      }
    }
  }

  // Activate debug to check if function is correct
  void Mesh::activate_duarte_debug()
  {
    Mesh::duarte_debug = true;
  };

  //
  void Mesh::prepare_zeta_interpolation(Mesh *oldmesh)
  {

    // Mesh as a geometric object.
    oomph::MeshAsGeomObject mesh_as_geom(oldmesh);

    // Number of elements.
    const unsigned nelem = this->nelement();

    // Loop through all elements.
    for (unsigned el = 0; el < nelem; el++)
    {

      // Current element
      BulkElementBase *curr_el = dynamic_cast<BulkElementBase *>(this->element_pt(el));

      // Number of integration points.
      unsigned int n_intpt = curr_el->integral_pt()->nweight();

      // Dimension of element
      const unsigned int dim = curr_el->dim();

      // Resize and initialise vector pair.
      curr_el->coords_oldmesh.resize(n_intpt);
      for (unsigned ipt = 0; ipt < n_intpt; ipt++)
      {
        curr_el->coords_oldmesh[ipt].first = NULL;
        curr_el->coords_oldmesh[ipt].second.resize(dim, 0.0);
      }

      // Fill the vector pair.
      curr_el->prepare_zeta_interpolation(&mesh_as_geom);
    }

    // Debug purposes! Ignore and delete when everything is fixed.
    if (this->duarte_debug)
    {

      // Loop through all elements.
      for (unsigned el = 0; el < nelem; el++)
      {

        // Current element
        BulkElementBase *curr_el = dynamic_cast<BulkElementBase *>(this->element_pt(el));

        // Number of integration points.
        unsigned int n_intpt = curr_el->integral_pt()->nweight();

        // Dimension of element
        const unsigned int dim = curr_el->dim();

        // Loop through ipts
        for (unsigned ipt = 0; ipt < n_intpt; ipt++)
        {
          // Get zeta-values of vector pair
          BulkElementBase *old_elem = curr_el->coords_oldmesh[ipt].first;
          oomph::Vector<double> s = curr_el->coords_oldmesh[ipt].second;
          oomph::Vector<double> zeta(dim, 0.0);
          old_elem->interpolated_zeta(s, zeta);

          // Get zeta-values of old mesh
          oomph::Vector<double> curr_zeta(dim, 0.0);
          oomph::Vector<double> curr_s(dim, 0.0);
          for (unsigned i = 0; i < dim; i++)
          {
            curr_s[i] = curr_el->integral_pt()->knot(ipt, i);
          }
          curr_el->interpolated_zeta(curr_s, curr_zeta);

          // Compare the local coordinates of the old mesh on integration points with the ones given by the vector pair.
          for (unsigned i = 0; i < dim; i++)
          {
            std::cout << "OLD MESH: " << curr_zeta[i] << "\t" << "NEW MESH: " << zeta[i] << "\t" << "DELTA: " << zeta[i] - curr_zeta[i] << "\n";
          }
        }
      }
    }
  }

  // Update time level for each element.
  void Mesh::set_time_level_for_projection(unsigned time_level)
  {

    // Number of elements.
    const unsigned nelem = this->nelement();

    // Loop through all elements.
    for (unsigned el = 0; el < nelem; el++)
    {

      // Current element
      BulkElementBase *curr_el = dynamic_cast<BulkElementBase *>(this->element_pt(el));

      // Update projection time.
      curr_el->projection_time = time_level;
    }
  }

  void Mesh::prepare_interpolation()
  {
    if (!this->interpolated_lagrangian_coordinates_at_remeshing) this->set_lagrangian_nodal_coordinates();
  }

  // This only works in max. 2d well
  void Mesh::nodal_interpolate_along_boundary(Mesh *old, int bind, int oldbind, Mesh *imesh, Mesh *oldimesh, double boundary_max_dist)
  {
    //std::cout << "Nodal interpolation along boundary " << bind << std::endl;
    // Bulk field mapping
    BulkElementBase *my_be0 = dynamic_cast<BulkElementBase *>(this->element_pt(0));
    BulkElementBase *from_be0 = dynamic_cast<BulkElementBase *>(old->element_pt(0));
    auto *my_ci = my_be0->get_code_instance();
    auto *from_ci = from_be0->get_code_instance();
    auto *my_ft = my_ci->get_func_table();
    auto *from_ft = from_ci->get_func_table();
    std::vector<int> field_map;
    field_map.resize(my_ft->numfields_C2TB_basebulk + my_ft->numfields_C2_basebulk + my_ft->numfields_C1TB_basebulk + my_ft->numfields_C1_basebulk);
    if (my_ci != from_ci)
    {
      if (my_be0->dim() != from_be0->dim())
      {
        throw_runtime_error("Cannot interpolate meshes of different element dimension");
      }
      if (my_be0->nodal_dimension() != from_be0->nodal_dimension())
      {
        throw_runtime_error("Cannot interpolate meshes of different nodal dimension");
      }
      for (unsigned int i = 0; i < field_map.size(); i++)
      {
        field_map[i] = -1;
        // Iterate over the fields and find the same name
        std::string name2find;
        if (i < my_ft->numfields_C2TB_basebulk)
          name2find = my_ft->fieldnames_C2TB[i];
        else if (i < my_ft->numfields_C2TB_basebulk + my_ft->numfields_C2_basebulk)
          name2find = my_ft->fieldnames_C2[i - my_ft->numfields_C2TB_basebulk];
        else if (i < my_ft->numfields_C2TB_basebulk + my_ft->numfields_C2_basebulk + my_ft->numfields_C1TB_basebulk)
          name2find = my_ft->fieldnames_C1TB[i - my_ft->numfields_C2TB_basebulk - my_ft->numfields_C2_basebulk];
        else
          name2find = my_ft->fieldnames_C1[i - my_ft->numfields_C2_basebulk - my_ft->numfields_C2TB_basebulk - my_ft->numfields_C1TB_basebulk];
        for (unsigned int j = 0; j < from_ft->numfields_C2TB_basebulk; j++)
        {
          if (std::string(from_ft->fieldnames_C2TB[j]) == name2find)
          {
            field_map[i] = j;
            break;
          }
        }
        if (field_map[i] < 0)
        {
          for (unsigned int j = 0; j < from_ft->numfields_C2_basebulk; j++)
          {
            if (std::string(from_ft->fieldnames_C2[j]) == name2find)
            {
              field_map[i] = j + from_ft->numfields_C2TB_basebulk;
              break;
            }
          }
          if (field_map[i] < 0)
          {
            for (unsigned int j = 0; j < from_ft->numfields_C1TB_basebulk; j++)
            {
              if (std::string(from_ft->fieldnames_C1TB[j]) == name2find)
              {
                field_map[i] = j + from_ft->numfields_C2_basebulk + from_ft->numfields_C2TB_basebulk;
                break;
              }
            }
            if (field_map[i] < 0)
            {
              for (unsigned int j = 0; j < from_ft->numfields_C1_basebulk; j++)
              {
                if (std::string(from_ft->fieldnames_C1[j]) == name2find)
                {
                  field_map[i] = j + from_ft->numfields_C2_basebulk + from_ft->numfields_C2TB_basebulk + from_ft->numfields_C1TB_basebulk;
                  break;
                }
              }
            }
          }
        }
      }
    }
    else
    {
      for (unsigned int i = 0; i < field_map.size(); i++)
      {
        field_map[i] = i;
      } // Identity
    }

    // Mapping of additional interface fields
    BulkElementBase *my_fe0 = NULL;
    if (imesh && imesh->nelement())
      my_fe0 = dynamic_cast<BulkElementBase *>(imesh->element_pt(0));
    BulkElementBase *from_fe0 = NULL;
    if (oldimesh && oldimesh->nelement())
      from_fe0 = dynamic_cast<BulkElementBase *>(oldimesh->element_pt(0));
    auto *my_fci = (my_fe0 ? my_fe0->get_code_instance() : NULL);
    auto *from_fci = (from_fe0 ? from_fe0->get_code_instance() : NULL);
    auto *my_fft = (my_fci ? my_fci->get_func_table() : NULL);
    auto *from_fft = (from_fci ? from_fci->get_func_table() : NULL);

    if (my_fft->numfields_D2TB || my_fft->numfields_D2 || my_fft->numfields_D1 || my_fft->numfields_DL || my_fft->numfields_D0)
    {
      std::ostringstream oss;
      oss << "At interface: " << this->domainname << " - Number of discontinuous fields -  D2TB: " << my_fft->numfields_D2TB << " D2: " << my_fft->numfields_D2 << " D1: " << my_fft->numfields_D1 << " DL: " << my_fft->numfields_DL << " D0: " << my_fft->numfields_D0;
      throw_runtime_error("Cannot interpolate discontinuous fields at interfaces yet: " + oss.str());
    }

    std::map<unsigned, unsigned> inter_field_map;

    if (my_fft && from_fft)
    {
      std::map<unsigned, std::string> my_interface_dofs;
      for (unsigned int i = 0; i < my_fft->numfields_C2 - my_fft->numfields_C2_basebulk; i++)
      {
        std::string name2find = my_fft->fieldnames_C2[my_fft->numfields_C2_basebulk + i];
        my_interface_dofs[my_fci->resolve_interface_dof_id(name2find)] = name2find;
      }
      for (unsigned int i = 0; i < my_fft->numfields_C1 - my_fft->numfields_C1_basebulk; i++)
      {
        std::string name2find = my_fft->fieldnames_C1[my_fft->numfields_C1_basebulk + i];
        my_interface_dofs[my_fci->resolve_interface_dof_id(name2find)] = name2find;
      }
      std::map<std::string, unsigned> from_interface_dofs;
      for (unsigned int i = 0; i < from_fft->numfields_C2 - from_fft->numfields_C2_basebulk; i++)
      {
        std::string name2find = from_fft->fieldnames_C2[from_fft->numfields_C2_basebulk + i];
        from_interface_dofs[name2find] = from_fci->resolve_interface_dof_id(name2find);
      }
      for (unsigned int i = 0; i < from_fft->numfields_C1 - from_fft->numfields_C1_basebulk; i++)
      {
        std::cout << "STARTING LOOP " << this->domainname << "  " << i << std::endl << std::flush;
        std::string name2find = from_fft->fieldnames_C1[from_fft->numfields_C1_basebulk + i];
        std::cout << "RESOLVING " << name2find << std::endl << std::flush;
        from_interface_dofs[name2find] = from_fci->resolve_interface_dof_id(name2find);
      }
      for (auto my : my_interface_dofs)
      {
        if (from_interface_dofs.count(my.second))
        {
          inter_field_map[my.first] = from_interface_dofs[my.second]; // Map interface field index
        }
      }
    }

    std::vector<oomph::Node *> newnodes, oldnodes;
    // Fill the node buffers
    if (this->nboundary_node(bind)) // Works only if codim 1 wrt. bulk mesh
    {
      newnodes.reserve(this->nboundary_node(bind));
      for (unsigned in = 0; in < this->nboundary_node(bind); in++)
      {
        newnodes.push_back(this->boundary_node_pt(bind, in));
      }
    }
    else // Now this is more complicated: We only have boundary elements defined, codim 2 or higher
    {
      std::set<oomph::Node *> uniquenodes;
      for (unsigned ie = 0; ie < this->nboundary_element(bind); ie++)
      {
        pyoomph::BulkElementBase *be = dynamic_cast<pyoomph::BulkElementBase *>(this->boundary_element_pt(bind, ie));
        for (unsigned in = 0; in < be->nnode(); in++)
        {
          if (be->node_pt(in)->is_on_boundary(bind))
          {
            uniquenodes.insert(be->node_pt(in));
          }
        }
      }
      for (auto *n : uniquenodes)
      {
        newnodes.push_back(n);
      }
    }

    if (old->nboundary_node(oldbind)) // Works only if codim 1 wrt. bulk mesh
    {
      oldnodes.reserve(old->nboundary_node(oldbind));
      for (unsigned in = 0; in < old->nboundary_node(oldbind); in++)
      {
        oldnodes.push_back(old->boundary_node_pt(oldbind, in));
      }
    }
    else // Now this is more complicated: We only have boundary elements defined, codim 2 or higher
    {
      std::set<oomph::Node *> uniquenodes;
      for (unsigned ie = 0; ie < old->nboundary_element(oldbind); ie++)
      {
        pyoomph::BulkElementBase *be = dynamic_cast<pyoomph::BulkElementBase *>(old->boundary_element_pt(oldbind, ie));
        for (unsigned in = 0; in < be->nnode(); in++)
        {
          if (be->node_pt(in)->is_on_boundary(oldbind))
          {
            uniquenodes.insert(be->node_pt(in));
          }
        }
      }
      for (auto *n : uniquenodes)
      {
        oldnodes.push_back(n);
      }
    }

    for (unsigned in = 0; in < newnodes.size(); in++)
    {
      oomph::Node *n = newnodes[in];
      oomph::Vector<double> xn = n->position();

      double mindist = 1e40;
      oomph::Node *bestnode = NULL;
      for (unsigned im = 0; im < oldnodes.size(); im++)
      {
        oomph::Node *m = oldnodes[im];
        //     std::cerr << "VALS " << m->nvalue() << " vs " <<  n->nvalue() << std::endl;
        //     if (m->nvalue()<n->nvalue()) continue; //Only take the nodes with the same amount of values //TODO: Also check whether the nodes are on the exact same boundaries
        double dist = 0;
        for (unsigned di = 0; di < xn.size(); di++)
          dist += (xn[di] - m->position()[di]) * (xn[di] - m->position()[di]);
        if (dist < mindist)
        {
          bestnode = m;
          mindist = dist;
        }
      }
      if (mindist > 1.0)
      {
        bestnode = NULL;
        mindist = 1e40;
      }
      if (!bestnode)
      {
        std::cerr << "Cannot find a matching boundary node for " << xn[0] << ", " << xn[1] << " NUMOLD " << old->nboundary_node(oldbind) << std::endl;
        std::cerr << "NUMVALS " << n->nvalue() << " BUT FOUND ";
        for (unsigned im = 0; im < oldnodes.size(); im++)
        {
          oomph::Node *m = oldnodes[im];
          std::cerr << " " << m->nvalue();
        }
        std::cerr << std::endl;
        for (unsigned im = 0; im < oldnodes.size(); im++)
        {
          oomph::Node *m = oldnodes[im];
          //			  if (m->nvalue()<n->nvalue()) continue;
          double dist = 0;
          for (unsigned di = 0; di < xn.size(); di++)
            dist += (xn[di] - m->position()[di]) * (xn[di] - m->position()[di]);
          if (dist < mindist)
          {
            bestnode = m;
            mindist = dist;
          }
        }
        //			std::cerr << "BESTDIST " << mindist << "  at " << xn[0] << ", " << xn[1] <<std::endl;
        //      continue;
      } // TODO
      if (boundary_max_dist > 0 && sqrt(mindist) > boundary_max_dist)
      {
        std::cerr << "Cannot find a boundary node within the distance of " << boundary_max_dist << "  at " << xn[0] << ", " << xn[1] << std::endl;
        continue;
        // TODO
      }

      double mindist2 = 1e40;
      oomph::Node *bestnode2 = NULL;
      for (unsigned im = 0; im < oldnodes.size(); im++)
      {
        oomph::Node *m = oldnodes[im];
        //		 if (n->nvalue()!=m->nvalue()) continue;
        if (m == bestnode)
          continue;
        double dist = 0;
        for (unsigned di = 0; di < xn.size(); di++)
          dist += (xn[di] - m->position()[di]) * (xn[di] - m->position()[di]);
        if (dist < mindist2)
        {
          mindist2 = dist;
          bestnode2 = m;
        }
      }
      if (!bestnode2)
      {
        mindist2 = mindist;
        bestnode2 = bestnode;
      }
      //			std::cerr << "	BESTDIST1 " << mindist << "  BESTDIST2 " << mindist2 << "  at " << xn[0] << ", " << xn[1] <<std::endl;
      mindist = sqrt(mindist);
      mindist2 = sqrt(mindist2);
      double lambda1 = (mindist > 1e-20 ? mindist2 / (mindist + mindist2) : 1);
      double lambda2 = (mindist > 1e-20 ? mindist / (mindist + mindist2) : 0);

      oomph::BoundaryNodeBase *bestbnode = dynamic_cast<oomph::BoundaryNodeBase *>(bestnode);
      oomph::BoundaryNodeBase *bestbnode2 = dynamic_cast<oomph::BoundaryNodeBase *>(bestnode2);
      oomph::BoundaryNodeBase *bnode = dynamic_cast<oomph::BoundaryNodeBase *>(n);
      //     std::cout << "   NODE AT " << n->x(0) << " " << n->x(1) << "   at " << lambda1 << " times " << bestnode->x(0) << "," << bestnode->x(1) << "   and  "  << lambda2 << " times " << bestnode2->x(0) << "," << bestnode2->x(1) << std::endl;
      if (!bestbnode || !bestbnode2 || !bnode)
      {
        throw_runtime_error("Found a node on a boundary that is not a boundary node");
      }

      //   oomph::Vector<double> xm=bestnode->position();
      for (unsigned int time_ind = 0; time_ind < n->time_stepper_pt()->ntstorage(); time_ind++)
      {
        for (unsigned vi = 0; vi < field_map.size(); vi++)
        { // Do not interpolate lagrange multipiers
          //          std::cerr << "SETTING VALUE " << xm[0] << "," << xm[1]  << " :  " << time_ind << "  " << vi <<"  -> " << bestnode->value(time_ind,vi) << std::endl;
          if (field_map[vi] >= 0)
          {
            n->set_value(time_ind, vi, bestnode->value(time_ind, field_map[vi]) * lambda1 + bestnode2->value(time_ind, field_map[vi]) * lambda2);
          }
        }
        for (auto interfield : inter_field_map)
        {
          // std::cout << "SIZES  " <<newnodes.size() << " OLD "<< oldnodes.size() << std::endl << std::flush ;
          // std::cout << "DEST  " <<bnode << " @ "<< interfield.first << " NV " << n->nvalue() << " X " << n->x(0) << ", " << n->x(1)<< std::endl << std::flush ;
          int dest_i = bnode->index_of_first_value_assigned_by_face_element(interfield.first);
          // std::cout << "SRC1  " <<bestbnode << " @ "<< interfield.second << " NV " << bestnode->nvalue() << " X " << bestnode->x(0) << ", " << bestnode->x(1) <<std::endl << std::flush ;
          int src_i1 = bestbnode->index_of_first_value_assigned_by_face_element(interfield.second);
          // std::cout << "SRC2  " <<bestbnode2 << " @ "<< interfield.second << std::endl << std::flush ;
          int src_i2 = bestbnode2->index_of_first_value_assigned_by_face_element(interfield.second);
          n->set_value(time_ind, dest_i, bestnode->value(time_ind, src_i1) * lambda1 + bestnode2->value(time_ind, src_i2) * lambda2);
        }
      }

      for (unsigned int time_ind = 1; time_ind < n->time_stepper_pt()->ntstorage(); time_ind++)
      {
        for (unsigned i = 0; i < xn.size(); i++)
          n->x(time_ind, i) = bestnode->x(time_ind, i) * lambda1 + bestnode2->x(time_ind, i) * lambda2;
      }

      if (this->interpolated_lagrangian_coordinates_at_remeshing) // Interpolate also the Lagrangian coordinates
        {
          if (dynamic_cast<pyoomph::Node*>(n)->nlagrangian()!=dynamic_cast<pyoomph::Node*>(bestnode)->nlagrangian())
          {
            throw_runtime_error("Cannot interpolate Lagrangian coordinates if the number of Lagrangian nodes is different");
          }                    
          for (unsigned int i = 0; i < dynamic_cast<pyoomph::Node*>(n)->nlagrangian(); i++)
          {            
            double xl=dynamic_cast<pyoomph::Node*>(bestnode)->lagrangian_position(i)*lambda1+dynamic_cast<pyoomph::Node*>(bestnode2)->lagrangian_position(i)*lambda2;
            //std::cout << "SETTING LAGRANGIAN COORDINATE " << i << " from " << dynamic_cast<pyoomph::Node*>(n)->xi(i) << " to "  << xl << std::endl;
            dynamic_cast<pyoomph::Node*>(n)->xi(i)=xl;
          }
        }
    }
  }

  void Mesh::nodal_interpolate_from(Mesh *from, int boundary_index)
  {
    this->interpolated_lagrangian_coordinates_at_remeshing=from->interpolated_lagrangian_coordinates_at_remeshing;
    auto old_setting=BulkElementBase::zeta_coordinate_type;
    if (this->interpolated_lagrangian_coordinates_at_remeshing) BulkElementBase::zeta_coordinate_type=1;
    if (!this->nelement() || !from->nelement())
      return;
    BulkElementBase *my_be0 = dynamic_cast<BulkElementBase *>(this->element_pt(0));
    BulkElementBase *from_be0 = dynamic_cast<BulkElementBase *>(from->element_pt(0));
    auto *my_ci = my_be0->get_code_instance();
    auto *from_ci = from_be0->get_code_instance();
    auto *my_ft = my_ci->get_func_table();
    auto *from_ft = from_ci->get_func_table();
    std::vector<int> field_map;
    std::vector<int> field_map_D0;

    if (my_ft->numfields_D2TB || my_ft->numfields_D2 || my_ft->numfields_D1)
    {
      throw_runtime_error("Cannot interpolate DG fields at interfaces yet");
    }

    field_map.resize(my_ft->numfields_C2TB_basebulk + my_ft->numfields_C2_basebulk + my_ft->numfields_C1_basebulk + my_ft->numfields_C1TB_basebulk);

    if (my_ci != from_ci)
    {
      if (my_be0->dim() != from_be0->dim())
      {
        throw_runtime_error("Cannot interpolate meshes of different element dimension");
      }
      if (my_be0->nodal_dimension() != from_be0->nodal_dimension())
      {
        throw_runtime_error("Cannot interpolate meshes of different nodal dimension");
      }
      for (unsigned int i = 0; i < field_map.size(); i++)
      {
        field_map[i] = -1;
        // Iterate over the fields and find the same name

        std::string name2find;
        if (i < my_ft->numfields_C2TB_basebulk)
          name2find = my_ft->fieldnames_C2TB[i];
        else if (i < my_ft->numfields_C2TB_basebulk + my_ft->numfields_C2_basebulk)
          name2find = my_ft->fieldnames_C2[i - my_ft->numfields_C2TB_basebulk];
        else if (i < my_ft->numfields_C2TB_basebulk + my_ft->numfields_C2_basebulk + my_ft->numfields_C1TB_basebulk)
          name2find = my_ft->fieldnames_C2[i - my_ft->numfields_C2TB_basebulk - my_ft->numfields_C2_basebulk];
        else
          name2find = my_ft->fieldnames_C1[i - my_ft->numfields_C2_basebulk - my_ft->numfields_C2TB_basebulk - my_ft->numfields_C1TB_basebulk];
        for (unsigned int j = 0; j < from_ft->numfields_C2TB_basebulk; j++)
        {
          if (std::string(from_ft->fieldnames_C2TB[j]) == name2find)
          {
            field_map[i] = j;
            break;
          }
        }
        if (field_map[i] < 0)
        {
          for (unsigned int j = 0; j < from_ft->numfields_C2_basebulk; j++)
          {
            if (std::string(from_ft->fieldnames_C2[j]) == name2find)
            {
              field_map[i] = j + from_ft->numfields_C2TB_basebulk;
              break;
            }
          }
          if (field_map[i] < 0)
          {
            for (unsigned int j = 0; j < from_ft->numfields_C1TB_basebulk; j++)
            {
              if (std::string(from_ft->fieldnames_C1TB[j]) == name2find)
              {
                field_map[i] = j + from_ft->numfields_C2_basebulk + from_ft->numfields_C2TB_basebulk;
                break;
              }
            }
            if (field_map[i] < 0)
            {
              for (unsigned int j = 0; j < from_ft->numfields_C1_basebulk; j++)
              {
                if (std::string(from_ft->fieldnames_C1[j]) == name2find)
                {
                  field_map[i] = j + from_ft->numfields_C2_basebulk + from_ft->numfields_C2TB_basebulk + from_ft->numfields_C1TB_basebulk;
                  break;
                }
              }
            }
          }
        }
      }
    }
    else
    {
      for (unsigned int i = 0; i < field_map.size(); i++)
      {
        field_map[i] = i;
      } // Identity
    }

    std::map<unsigned, unsigned> inter_field_map;
    std::map<unsigned, std::string> old_inter_field_space;

    if (my_ft && from_ft)
    {
      std::map<unsigned, std::string> my_interface_dofs;
      for (unsigned int i = 0; i < my_ft->numfields_C2 - my_ft->numfields_C2_basebulk; i++)
      {
        std::string name2find = my_ft->fieldnames_C2[my_ft->numfields_C2_basebulk + i];
        my_interface_dofs[my_ci->resolve_interface_dof_id(name2find)] = name2find;
      }
      for (unsigned int i = 0; i < my_ft->numfields_C1 - my_ft->numfields_C1_basebulk; i++)
      {
        std::string name2find = my_ft->fieldnames_C1[my_ft->numfields_C1_basebulk + i];
        my_interface_dofs[my_ci->resolve_interface_dof_id(name2find)] = name2find;
      }
      std::map<std::string, unsigned> from_interface_dofs;
      std::map<unsigned, std::string> from_interface_spaces;
      for (unsigned int i = 0; i < from_ft->numfields_C2 - from_ft->numfields_C2_basebulk; i++)
      {
        std::string name2find = from_ft->fieldnames_C2[from_ft->numfields_C2_basebulk + i];
        from_interface_dofs[name2find] = from_ci->resolve_interface_dof_id(name2find);
        from_interface_spaces[from_interface_dofs[name2find]] = "C2";
      }
      for (unsigned int i = 0; i < from_ft->numfields_C1 - from_ft->numfields_C1_basebulk; i++)
      {
        std::string name2find = from_ft->fieldnames_C1[from_ft->numfields_C1_basebulk + i];
        from_interface_dofs[name2find] = from_ci->resolve_interface_dof_id(name2find);
        from_interface_spaces[from_interface_dofs[name2find]] = "C1";
      }
      for (auto my : my_interface_dofs)
      {
        if (from_interface_dofs.count(my.second))
        {
          inter_field_map[my.first] = from_interface_dofs[my.second]; // Map interface field index
          old_inter_field_space[my.first] = from_interface_spaces[inter_field_map[my.first]];
        }
      }
    }

    std::cout << "INTERPOLATING FROM " << from << std::endl;
    oomph::MeshAsGeomObject MaGO(from);

    std::set<oomph::Node *> completed_nodes;
    std::set<oomph::Node *> missing_nodes;

    for (unsigned int ie = 0; ie < this->nelement(); ie++)
    {
      BulkElementBase *deste = dynamic_cast<BulkElementBase *>(this->element_pt(ie));
      for (unsigned int ine = 0; ine < deste->nnode(); ine++)
      {
        oomph::Node *n = deste->node_pt(ine);
        if (n->is_on_boundary() && boundary_index < 0)
          continue; // Skip the boundary nodes here
        if (completed_nodes.count(n) || missing_nodes.count(n))
          continue;

        oomph::Vector<double> xnode = n->position();
        if (dynamic_cast<InterfaceMesh *>(this) && boundary_index >= 0)
        {
          xnode.resize(deste->dim());
          //          std::cout << "BOUNDARY INDEX " << boundary_index << "  xnode " << xnode.size() << std::endl;
          n->get_coordinates_on_boundary(boundary_index, xnode);
          //          std::cout << "  XNODE " << xnode[0] << std::endl;
        }
        oomph::Vector<double> s(xnode.size(), 0.5 * (deste->s_min() + deste->s_max()));
        oomph::GeomObject *resgo = 0;
        BulkElementBase *srcelem = NULL;

        MaGO.locate_zeta(xnode, resgo, s, false);
        srcelem = dynamic_cast<BulkElementBase *>(resgo);
        if (!srcelem)
        {
          if (boundary_index<0) std::cerr << "MISSING_BULKONLY_ELEM_AT\t" << xnode[0] << "\t" << xnode[1] << "  " << completed_nodes.size() * 100.0 / this->nnode() << " % done  | resgo " << resgo  << std::endl;
          missing_nodes.insert(n);
          continue;
        }

        std::vector<double> shift(deste->nodal_dimension(), 0.0);
        for (unsigned int i = 0; i < deste->nodal_dimension(); i++)
        {
          shift[i] = n->x(i) - srcelem->interpolated_x(s, i);
          //         std::cout << "SHIFT " << i << "  " << shift[i] << " WITH BOUND IND " << boundary_index << std::endl;
        }
        for (unsigned int i = 0; i < deste->nodal_dimension(); i++)
        {
          for (unsigned int time_ind = 1; time_ind < n->position_time_stepper_pt()->ntstorage(); time_ind++)
          {
            n->x(time_ind, i) = srcelem->interpolated_x(time_ind, s, i) + shift[i];
          }
        }

        if (this->interpolated_lagrangian_coordinates_at_remeshing) // Interpolate also the Lagrangian coordinates
        {
          if (srcelem->nlagrangian()!=deste->nlagrangian())
          {
            throw_runtime_error("Cannot interpolate Lagrangian coordinates if the number of Lagrangian nodes is different");
          }                    
          for (unsigned int i = 0; i < srcelem->nlagrangian(); i++)
          {            
            double xl=srcelem->interpolated_xi(s,i);
            //std::cout << "SETTING LAGRANGIAN COORDINATE " << i << " from " << dynamic_cast<pyoomph::Node*>(n)->xi(i) << " to "  << xl << std::endl;
            dynamic_cast<pyoomph::Node*>(n)->xi(i)=xl;
          }
        }

        for (unsigned int time_ind = 0; time_ind < n->time_stepper_pt()->ntstorage(); time_ind++)
        {
          oomph::Vector<double> vals;
          srcelem->get_interpolated_values(time_ind, s, vals);
          for (unsigned int vi = 0; vi < vals.size(); vi++)
          {
            if (field_map[vi] >= 0)
            {
              n->set_value(time_ind, vi, vals[field_map[vi]]);
            }
          }

          for (auto interfield : inter_field_map)
          {
            int dest_i = dynamic_cast<oomph::BoundaryNodeBase *>(n)->index_of_first_value_assigned_by_face_element(interfield.first);
            double val = dynamic_cast<pyoomph::InterfaceElementBase *>(srcelem)->get_interpolated_interface_field(s, interfield.second, old_inter_field_space[interfield.first], time_ind);
            n->set_value(time_ind, dest_i, val);
          }
        }

        completed_nodes.insert(n);
      }
      // TODO: Internal data
      if (my_ft->numfields_DL || my_ft->numfields_D0)
      {
        auto *ts = deste->internal_data_pt(0)->time_stepper_pt();
        // Find the elem in the center
        oomph::Vector<double> dmpt = deste->get_Eulerian_midpoint_from_local_coordinate(); // TODO: Lagrangian?

        oomph::Vector<double> s(dmpt.size(), 0.5 * (deste->s_min() + deste->s_max()));
        oomph::GeomObject *resgo = 0;
        BulkElementBase *srcelem = NULL;

        MaGO.locate_zeta(dmpt, resgo, s, false);
        srcelem = dynamic_cast<BulkElementBase *>(resgo);
        if (!srcelem)
        {
          if (boundary_index<0) std::cerr << "MISSING_BULKONLY_ELEM_AT\t" << dmpt[0] << "\t" << dmpt[1] << "  INTERNAL CENTER " << ie * 100.0 / this->nelement() << " % done  | resgo " << resgo << std::endl;
          continue;
        }
        // Interpolate all D0 fields
        if (my_ft->numfields_D0 != from_ft->numfields_D0)
        {
          throw_runtime_error("TODO: Field mapping if D0 spaces are different"); // TODO: Field mapping
        }
        if (my_ft->numfields_DL != from_ft->numfields_DL)
        {
          throw_runtime_error("TODO: Field mapping if DL spaces are different"); // TODO: Field mapping
        }

        for (unsigned int time_ind = 0; time_ind < ts->ntstorage(); time_ind++)
        {
          if (my_ft->numfields_D0)
          {
            oomph::Vector<double> vals;
            srcelem->get_interpolated_fields_D0(s, vals, time_ind);
            for (unsigned int vi = 0; vi < vals.size(); vi++)
            {
              deste->internal_data_pt(my_ft->numfields_DL + vi)->set_value(time_ind, 0, vals[vi]); // TODO: Field mapping
            }
          }
          if (my_ft->numfields_DL)
          {
            oomph::Vector<double> vals;
            srcelem->get_interpolated_fields_DL(s, vals, time_ind);
            for (unsigned int vi = 0; vi < vals.size(); vi++)
            {
              deste->internal_data_pt(vi)->set_value(time_ind, 0, vals[vi]); // TODO: Field mapping
              for (unsigned int j = 1; j < deste->internal_data_pt(vi)->nvalue(); j++)
              {
                deste->internal_data_pt(vi)->set_value(time_ind, j, 0); // TODO: Field mapping, slopes!
              }
            }
          }
        }

        // throw_runtime_error("TODO: DL data interpolation");
      }
    }

    // Handle the nodes which where not found by nearest nodes
    for (oomph::Node *n : missing_nodes)
    {
      if (completed_nodes.count(n))
        continue;
      oomph::Vector<double> xnode = n->position();
      if (boundary_index<0) std::cerr << "FOUND UNTREATED BULK NODE AT\t" << xnode[0] << "\t" << xnode[1] << std::endl;
      double mindist = 1e40;
      oomph::Node *bestnode = NULL;
      for (unsigned int mi = 0; mi < from->nnode(); mi++)
      {
        oomph::Node *m = from->node_pt(mi);
        oomph::Vector<double> xm = m->position();
        double dist = 0;
        for (unsigned di = 0; di < xm.size(); di++)
          dist += (xnode[di] - xm[di]) * (xnode[di] - xm[di]);
        if (dist < mindist)
        {
          mindist = dist;
          bestnode = m;
        }
      }
      if (bestnode)
      {
        double mindist2 = 1e40;
        oomph::Node *bestnode2 = NULL;
        for (unsigned int mi = 0; mi < from->nnode(); mi++)
        {
          oomph::Node *m = from->node_pt(mi);
          if (m == bestnode)
            continue;
          oomph::Vector<double> xm = m->position();
          double dist = 0;
          for (unsigned di = 0; di < xm.size(); di++)
            dist += (xnode[di] - xm[di]) * (xnode[di] - xm[di]);
          if (dist < mindist2)
          {
            mindist2 = dist;
            bestnode2 = m;
          }
        }
        if (!bestnode2)
        {
          mindist2 = mindist;
          bestnode2 = bestnode;
        }
        mindist = sqrt(mindist);
        mindist2 = sqrt(mindist2);
        double lambda1 = (mindist > 1e-20 ? mindist2 / (mindist + mindist2) : 1);
        double lambda2 = (mindist > 1e-20 ? mindist / (mindist + mindist2) : 0);
        oomph::Vector<double> xm = bestnode->position();
        for (unsigned int time_ind = 0; time_ind < n->time_stepper_pt()->ntstorage(); time_ind++)
        {
          for (unsigned vi = 0; vi < std::min((unsigned int)field_map.size(),n->nvalue()); vi++)
          {
            if (field_map[vi] >= 0)
            {
              n->set_value(time_ind, vi, bestnode->value(time_ind, field_map[vi]) * lambda1 + bestnode2->value(time_ind, field_map[vi]) * lambda2);
            }
          }
        }

        for (unsigned int time_ind = 1; time_ind < n->position_time_stepper_pt()->ntstorage(); time_ind++)
        {
          for (unsigned i = 0; i < xm.size(); i++)
            n->x(time_ind, i) = bestnode->x(time_ind, i) * lambda1 + bestnode2->x(time_ind, i) * lambda2;
        }

        if (this->interpolated_lagrangian_coordinates_at_remeshing) // Interpolate also the Lagrangian coordinates
        {
          if (dynamic_cast<pyoomph::Node*>(n)->nlagrangian()!=dynamic_cast<pyoomph::Node*>(bestnode)->nlagrangian())
          {
            throw_runtime_error("Cannot interpolate Lagrangian coordinates if the number of Lagrangian nodes is different");
          }                    
          for (unsigned int i = 0; i < dynamic_cast<pyoomph::Node*>(n)->nlagrangian(); i++)
          {            
            double xl=dynamic_cast<pyoomph::Node*>(bestnode)->lagrangian_position(i)*lambda1+dynamic_cast<pyoomph::Node*>(bestnode2)->lagrangian_position(i)*lambda2;
            //std::cout << "SETTING LAGRANGIAN COORDINATE " << i << " from " << dynamic_cast<pyoomph::Node*>(n)->xi(i) << " to "  << xl << std::endl;
            dynamic_cast<pyoomph::Node*>(n)->xi(i)=xl;
          }
        }

        completed_nodes.insert(n);
      }
      else
      {
        std::cerr << "  NOT EVEN FOUND A NEAREST NODE" << std::endl;
      }
    }

    BulkElementBase::zeta_coordinate_type=old_setting;
  }

  std::vector<pyoomph::Node*> Mesh::add_interpolated_nodes_at(const std::vector<std::vector<double> > & coords,bool all_as_boundary_nodes)
  {
    std::vector<pyoomph::Node*> res;
    oomph::MeshAsGeomObject MaGO(this);
    pyoomph::BulkElementBase* el0=dynamic_cast<pyoomph::BulkElementBase*>(this->element_pt(0));
    pyoomph::Node * n0=dynamic_cast<pyoomph::Node*>(el0->node_pt(0));
    for ( const auto  & coord : coords)
    {
      oomph::GeomObject *res_go = NULL;
      oomph::Vector<double> zet(coord.size());
      for (unsigned i=0;i<coord.size();i++) zet[i]=coord[i];
      oomph::Vector<double> s(el0->dim(), 1.0 / 3.0);
      MaGO.locate_zeta(zet, res_go, s, false);
      BulkElementBase *srcelem = dynamic_cast<BulkElementBase *>(res_go);

      pyoomph::Node *newnode;
      if (all_as_boundary_nodes)
      {
        newnode= new pyoomph::BoundaryNode(n0->time_stepper_pt(),el0->nlagrangian(), el0->nnodal_lagrangian_type(), el0->nodal_dimension(), el0->nnodal_position_type(), el0->required_nvalue(0));
      }
      else
      {
        newnode= new pyoomph::Node(n0->time_stepper_pt(),el0->nlagrangian(), el0->nnodal_lagrangian_type(), el0->nodal_dimension(), el0->nnodal_position_type(), el0->required_nvalue(0));	
      }

      for (unsigned i=0;i<coord.size();i++) newnode->x(i)=coord[i]; // Can't do a lot here
      if (srcelem)
      {
        for (unsigned int time_ind = 0; time_ind < n0->time_stepper_pt()->ntstorage(); time_ind++)
        {
          oomph::Vector<double> vals;
          srcelem->get_interpolated_values(time_ind, s, vals);
          for (unsigned int vi = 0; vi < std::min((unsigned)vals.size(),newnode->nvalue()); vi++)
          {
              newnode->set_value(time_ind, vi, vals[vi]);           
          }
          for (unsigned int i = 0; i < newnode->ndim(); i++)
          {
            newnode->x(time_ind, i) = srcelem->interpolated_x(time_ind, s, i);
          }
        }
        
            /*for (unsigned int time_ind = 1; time_ind < n0->position_time_stepper_pt()->ntstorage(); time_ind++)
            {
           
            }*/
        
        
      }
      

      

      res.push_back(newnode);
    }
    return res;
  }

  void Mesh::set_output_scale(std::string fname, GiNaC::ex s, DynamicBulkElementInstance *_code)
  {
    if (!_code)
    {
      BulkElementBase *be = dynamic_cast<BulkElementBase *>(this->element_pt(0));
      _code = be->get_code_instance();
    }
    GiNaC::ex fscale = _code->get_element_class()->get_scaling(fname);
    GiNaC::ex scale = fscale / s;
    // Expand the scale (to remove any scale factors)
    scale = _code->get_element_class()->expand_placeholders(scale, "OutputScale", true);
    scale = pyoomph::expressions::replace_global_params_by_current_values(scale);
    try
    {
      GiNaC::numeric num = GiNaC::ex_to<GiNaC::numeric>(scale);
      this->output_scales[fname] = num.to_double();
    }
    catch (const std::runtime_error &error)
    {
      std::ostringstream oss;
      oss << fscale << " vs " << s;
      //   oss << " CONV " << GiNaC::ex_to<GiNaC::numeric>(scale) << std::endl;//  << "  " << GiNaC::ex_to<GiNaC::numeric>(scale).to_double() << std::endl;
      throw std::runtime_error("Cannot set the output scale of '" + fname + "' since the dimensions are not matching : " + oss.str());
    }
  }

  void Mesh::describe_global_dofs(std::vector<int> &doftype, std::vector<std::string> &typnames)
  {
    typnames.clear();
    if (!this->nelement())
      return;
    doftype.resize(problem->ndof(), -1);
    BulkElementBase *be = dynamic_cast<BulkElementBase *>(this->element_pt(0));
    DynamicBulkElementInstance *ci = be->get_code_instance();

    auto *ft = ci->get_func_table();

    // TODO: Can't this be just copied from the functable Dirichlet_names ?
    if (ft->Dirichlet_set_size >= 3)
    {
      typnames.reserve(ft->Dirichlet_set_size - 3);
      for (unsigned i = 3; i < ft->Dirichlet_set_size; i++)
        typnames.push_back(ft->Dirichlet_names[i]);
    }

    std::vector<int> idof_C2TB, idof_C2, idof_C1, idof_C1TB;
    for (unsigned int f = ft->numfields_C2TB_basebulk; f < ft->numfields_C2TB; f++)
    {
      idof_C2TB.push_back(ci->resolve_interface_dof_id(ft->fieldnames_C2TB[f]));
    }
    for (unsigned int f = ft->numfields_C2_basebulk; f < ft->numfields_C2; f++)
    {
      idof_C2.push_back(ci->resolve_interface_dof_id(ft->fieldnames_C2[f]));
    }
    for (unsigned int f = ft->numfields_C1TB_basebulk; f < ft->numfields_C1TB; f++)
    {
      idof_C1TB.push_back(ci->resolve_interface_dof_id(ft->fieldnames_C1TB[f]));
    }
    for (unsigned int f = ft->numfields_C1_basebulk; f < ft->numfields_C1; f++)
    {
      idof_C1.push_back(ci->resolve_interface_dof_id(ft->fieldnames_C1[f]));
    }

    unsigned moving_node_offset = typnames.size();
    if (ft->moving_nodes)
    {
      if (ft->nodal_dim > 0)
        typnames.push_back("mesh_x");
      if (ft->nodal_dim > 1)
        typnames.push_back("mesh_y");
      if (ft->nodal_dim > 2)
        typnames.push_back("mesh_z");
    }
    unsigned int num_bulk_nodal = ft->numfields_C2TB_basebulk + ft->numfields_C2_basebulk + ft->numfields_C1TB_basebulk + ft->numfields_C1_basebulk;

    for (unsigned int ne = 0; ne < this->nelement(); ne++)
    {
      BulkElementBase *e = dynamic_cast<BulkElementBase *>(this->element_pt(ne));
      for (unsigned nn = 0; nn < e->nnode(); nn++)
      {
        Node *n = dynamic_cast<Node *>(e->node_pt(nn));
        for (unsigned int nv = 0; nv < num_bulk_nodal; nv++)
        {
          if (n->eqn_number(nv) >= 0)
          {
            doftype[n->eqn_number(nv)] = nv;
          }
        }
        oomph::BoundaryNodeBase *bn = dynamic_cast<oomph::BoundaryNodeBase *>(n);
        if (bn)
        {
          for (unsigned int f = 0; f < idof_C2TB.size(); f++)
          {
            int nv = bn->index_of_first_value_assigned_by_face_element(idof_C2TB[f]);
            if (n->eqn_number(nv) >= 0)
            {
              doftype[n->eqn_number(nv)] = ft->buffer_offset_C2TB_interf + f;
            }
          }
          for (unsigned int f = 0; f < idof_C2.size(); f++)
          {
            int nv = bn->index_of_first_value_assigned_by_face_element(idof_C2[f]);
            if (n->eqn_number(nv) >= 0)
            {
              doftype[n->eqn_number(nv)] = ft->buffer_offset_C2_interf + f;
            }
          }
          for (unsigned int f = 0; f < idof_C1TB.size(); f++)
          {
            int nv = bn->index_of_first_value_assigned_by_face_element(idof_C1TB[f]);
            if (n->eqn_number(nv) >= 0)
            {
              doftype[n->eqn_number(nv)] = ft->buffer_offset_C1TB_interf + f;
            }
          }
          for (unsigned int f = 0; f < idof_C1.size(); f++)
          {
            int nv = bn->index_of_first_value_assigned_by_face_element(idof_C1[f]);
            if (n->eqn_number(nv) >= 0)
            {
              doftype[n->eqn_number(nv)] = ft->buffer_offset_C1_interf + f;
            }
          }
        }
      }

      for (unsigned nf = 0; nf < ft->numfields_D2TB; nf++)
      {
        for (unsigned int ni = 0; ni < e->get_eleminfo()->nnode_C2TB; ni++)
        {
          int eqn_no = e->get_D2TB_nodal_data(nf)->eqn_number(e->get_D2TB_node_index(nf, ni));
          if (eqn_no >= 0)
          {
            doftype[eqn_no] = (nf < ft->numfields_D2TB_basebulk ? ft->buffer_offset_D2TB_basebulk : ft->buffer_offset_D2TB_interf - ft->numfields_D2TB_basebulk) + nf;
          }
        }
      }
      for (unsigned nf = 0; nf < ft->numfields_D2; nf++)
      {
        for (unsigned int ni = 0; ni < e->get_eleminfo()->nnode_C2; ni++)
        {
          int eqn_no = e->get_D2_nodal_data(nf)->eqn_number(e->get_D2_node_index(nf, ni));
          if (eqn_no >= 0)
          {
            doftype[eqn_no] = (nf < ft->numfields_D2_basebulk ? ft->buffer_offset_D2_basebulk : ft->buffer_offset_D2_interf - ft->numfields_D2_basebulk) + nf;
          }
        }
      }
      for (unsigned nf = 0; nf < ft->numfields_D1TB; nf++)
      {
        for (unsigned int ni = 0; ni < e->get_eleminfo()->nnode_C1TB; ni++)
        {
          int eqn_no = e->get_D1TB_nodal_data(nf)->eqn_number(e->get_D1TB_node_index(nf, ni));
          if (eqn_no >= 0)
          {
            doftype[eqn_no] = (nf < ft->numfields_D1TB_basebulk ? ft->buffer_offset_D1TB_basebulk : ft->buffer_offset_D1TB_interf - ft->numfields_D1TB_basebulk) + nf;
          }
        }
      }

      for (unsigned nf = 0; nf < ft->numfields_D1; nf++)
      {
        for (unsigned int ni = 0; ni < e->get_eleminfo()->nnode_C1; ni++)
        {
          int eqn_no = e->get_D1_nodal_data(nf)->eqn_number(e->get_D1_node_index(nf, ni));
          if (eqn_no >= 0)
          {
            doftype[eqn_no] = (nf < ft->numfields_D1_basebulk ? ft->buffer_offset_D1_basebulk : ft->buffer_offset_D1_interf - ft->numfields_D1_basebulk) + nf;
          }
        }
      }
      for (unsigned nid = 0; nid < ft->numfields_DL; nid++)
      {
        auto *idp = e->internal_data_pt(ft->internal_offset_DL + nid);
        for (unsigned int nv = 0; nv < idp->nvalue(); nv++)
        {
          if (idp->eqn_number(nv) >= 0)
          {
            doftype[idp->eqn_number(nv)] = ft->buffer_offset_DL + nid;
          }
        }
      }
      for (unsigned nid = 0; nid < ft->numfields_D0; nid++)
      {
        auto *idp = e->internal_data_pt(ft->internal_offset_D0 + nid);
        for (unsigned int nv = 0; nv < idp->nvalue(); nv++)
        {
          if (idp->eqn_number(nv) >= 0)
          {
            doftype[idp->eqn_number(nv)] = ft->buffer_offset_D0 + nid;
          }
        }
      }

      if (ft->moving_nodes)
      {
        for (unsigned nn = 0; nn < e->nnode(); nn++)
        {
          Node *n = dynamic_cast<Node *>(e->node_pt(nn));
          for (unsigned int nv = 0; nv < ft->nodal_dim; nv++)
          {
            if (n->variable_position_pt()->eqn_number(nv) >= 0)
            {
              doftype[n->variable_position_pt()->eqn_number(nv)] = moving_node_offset + nv;
            }
          }
        }
      }
    }
  }

  pyoomph::Node *Mesh::resolve_copy_master(pyoomph::Node *cpy)
  {
    if (copied_masters.count(cpy))
      return copied_masters[cpy];
    return NULL;
  }

  void Mesh::store_copy_master(pyoomph::Node *cpy, pyoomph::Node *mst)
  {
    copied_masters[cpy] = mst;
  }

  double Mesh::get_output_scale(std::string fname)
  {
    if (this->output_scales.count(fname))
      return this->output_scales[fname];
    else
      return 1.0;
  }

  void Mesh::set_initial_condition(std::string fieldname, GiNaC::ex expression)
  {
    if ((!this->nnode()) || (!this->nelement()))
      return;
    BulkElementBase *be = dynamic_cast<BulkElementBase *>(this->element_pt(0));
    DynamicBulkElementInstance *ci = be->get_code_instance();
    int i = ci->get_nodal_field_index(fieldname);
    if (i < 0)
    {
      i = ci->get_discontinuous_field_index(fieldname);
      if (i < 0)
      {
        throw_runtime_error("Cannot set initial condition of unknown field '" + fieldname + "'");
      }
    }

    ReplaceFieldsToNonDimFields repl(ci->get_element_class(), "InitialCondition");
    initial_conditions[fieldname] = 0 + repl(expression) / ci->get_element_class()->get_scaling(fieldname);
    // Test if the initial condition is nondimensional and has no free parameters
    auto *n = this->node_pt(0);
    GiNaC::lst subslist;
    subslist.append(pyoomph::expressions::x == n->x(0));
    if (n->ndim() > 1)
    {
      subslist.append(pyoomph::expressions::y == n->x(1));
      if (n->ndim() > 2)
      {
        subslist.append(pyoomph::expressions::z == n->x(2));
      }
    }
    auto *ts = n->time_stepper_pt();
    auto *Time_pt = ts->time_pt();
    subslist.append(pyoomph::expressions::t == Time_pt->time());
    GiNaC::ex subst = initial_conditions[fieldname].subs(subslist);
    try
    {
      subst = subst.evalf();
      GiNaC::numeric num = GiNaC::ex_to<GiNaC::numeric>(subst);
    }
    catch (const std::runtime_error &error)
    {
      std::ostringstream oss;
      oss << subst;
      throw std::runtime_error("Cannot evaluate the following initial condition, since it has unknown variables or units in it: " + oss.str());
    }
    // Simplify the expression by setting all units to unity
    GiNaC::lst sublist;
    for (auto &bu : base_units)
    {
      sublist.append(bu.second == 1);
    }
    initial_conditions[fieldname] = initial_conditions[fieldname].subs(sublist);

    std::cout << "Mesh Initial Condition: " << fieldname << std::endl
              << initial_conditions[fieldname] << std::endl;
  }

  void Generic_SetInitialCondition(BulkElementBase *elempt, oomph::Data *data, DynamicBulkElementInstance *ci, int fieldindex, unsigned valindex, double *x_buffer, double *x_lagr, double *normal, bool use_identity, bool resetting_first_step, unsigned icindex)
  {
    auto *ts = data->time_stepper_pt();
    auto *Time_pt = ts->time_pt();
    for (unsigned t = 0; t < Time_pt->ndt(); t++)
    {
      double time_local = Time_pt->time(t);
      double default_val = 0.0; // data->value(t,valindex)
      if (use_identity)
      {
        if (fieldindex < 0 && t == 0 && resetting_first_step)
        {
          default_val = data->value(1, valindex); // Positions from previous step!
        }
        else
        {
          default_val = data->value(t, valindex);
        }
      }
      /*		if (use_identity) {
           std::cout << "POS INIT COND " << t << "  " << default_val << std::endl;
          }*/
      double val = ci->get_func_table()->InitialConditionFunc[icindex](elempt->get_eleminfo(), fieldindex, x_buffer, x_lagr, normal, time_local, 0, default_val);
      //	std::cout  << "INIT COND " << t << "  " << val << std::endl;
      data->set_value(t, valindex, val);
    }

    if (dynamic_cast<oomph::Newmark<2> *>(ts) || dynamic_cast<oomph::NewmarkBDF<2> *>(ts) || dynamic_cast<pyoomph::MultiTimeStepper *>(ts))
    {
      //		std::cout << "NEWMARK" << std::endl;
      unsigned NSTEPS = 2; // TODO: Also NSTEPS=1
      //		if (dynamic_cast<oomph::NewmarkBDF<2>*>(ts)) throw_runtime_error("Cannot set initial condition for NewmarkBDF2 yet");

      pyoomph::MultiTimeStepper *mts = dynamic_cast<pyoomph::MultiTimeStepper *>(ts);
      double U = data->value(0, valindex);
      double U0 = data->value(1, valindex);
      double time_local = Time_pt->time(0);
      double default_val = 0.0;
      //		if (use_identity) default_val=data->value(t,valindex);
      double Udot = ci->get_func_table()->InitialConditionFunc[icindex](elempt->get_eleminfo(), fieldindex, x_buffer, x_lagr, normal, time_local, 1, default_val);    // TODO: Better default value
      double Udotdot = ci->get_func_table()->InitialConditionFunc[icindex](elempt->get_eleminfo(), fieldindex, x_buffer, x_lagr, normal, time_local, 2, default_val); // TODO: Better default value
      //	  std::cout  << "GOT TV " <<  U << "  " << U0 << "  " << Udot << "  " << Udotdot << std::endl;
      oomph::Vector<double> vect(2);
      vect[0] = Udotdot - (mts ? mts->weightNewmark2(2, 0) : ts->weight(2, 0)) * U - (mts ? mts->weightNewmark2(2, 1) : ts->weight(2, 1)) * U0;
      vect[1] = Udot - (mts ? mts->weightNewmark2(1, 0) : ts->weight(1, 0)) * U - (mts ? mts->weightNewmark2(1, 1) : ts->weight(1, 1)) * U0;
      //  std::cout  << "VECT  " <<  vect[0] << "  " << vect[1] <<std::endl;
      oomph::DenseDoubleMatrix matrix(2, 2);

      matrix(0, 0) = (mts ? mts->weightNewmark2(2, NSTEPS + 1) : ts->weight(2, NSTEPS + 1));
      matrix(0, 1) = (mts ? mts->weightNewmark2(2, NSTEPS + 2) : ts->weight(2, NSTEPS + 2));
      matrix(1, 0) = (mts ? mts->weightNewmark2(1, NSTEPS + 1) : ts->weight(1, NSTEPS + 1));
      ;
      matrix(1, 1) = (mts ? mts->weightNewmark2(1, NSTEPS + 2) : ts->weight(1, NSTEPS + 2));
      ;
      // std::cout << "MAT " << matrix(0,0) << "  " << matrix(0,1) << "  |  " << matrix(1,0) << "   " << matrix(1,1) << std::endl;
      if (fabs(matrix(0, 0) * matrix(1, 1) - matrix(1, 0) * matrix(0, 1)) > 1e-14)
      {
        try
        {
          matrix.solve(vect);
          data->set_value(NSTEPS + 1, valindex, vect[0]); // TODO Slopes for DL fields
          data->set_value(NSTEPS + 2, valindex, vect[1]);
        }
        catch (const std::runtime_error &error)
        {
        }
      }
    }
  }

  void Mesh::setup_initial_conditions(bool resetting_first_step, std::string ic_name)
  {
    //  std::cout << "CALLED SET IC  "  << ic_name << std::endl;
    double x_buffer[3] = {0, 0, 0};
    double x_lagr[3] = {0, 0, 0};
    double normal[3] = {0, 0, 0};
    if (!this->nelement())
      return;

    auto *el = dynamic_cast<BulkElementBase *>(this->element_pt(0));
    auto *ft = el->get_code_instance()->get_func_table();
    unsigned nodal_dim = el->nodal_dimension();
    unsigned eldim = el->dim();
    // Precalculate the normals, they might be relevant
    std::map<pyoomph::Node *, oomph::Vector<double>> nodal_normals;
    if (this->nnode() && nodal_dim == eldim + 1)
    {
      for (unsigned ie = 0; ie < this->nelement(); ie++)
      {
        auto *ele = dynamic_cast<BulkElementBase *>(this->element_pt(ie));
        for (unsigned int in = 0; in < ele->nnode(); in++)
        {
          pyoomph::Node *nodept = dynamic_cast<pyoomph::Node *>(ele->node_pt(in));
          oomph::Vector<double> s(eldim);
          ele->local_coordinate_of_node(in, s);
          oomph::Vector<double> n(nodal_dim);
          ele->get_normal_at_s(s, n, nullptr, nullptr);
          if (!nodal_normals.count(nodept))
          {
            nodal_normals[nodept] = n;
          }
          else
          {
            for (unsigned int id = 0; id < nodal_dim; id++)
              nodal_normals[nodept][id] += n[id];
          }
        }
      }
      for (auto &nn : nodal_normals)
      {
        double sqrl = 0.0;
        for (unsigned int id = 0; id < nodal_dim; id++)
          sqrl += nn.second[id] * nn.second[id];
        sqrl = 1.0 / sqrt(sqrl);
        for (unsigned int id = 0; id < nodal_dim; id++)
          nn.second[id] *= sqrl;
      }
    }

    int ic_index = -1;
    for (unsigned int i = 0; i < ft->num_ICs; i++)
    {
      if (std::string(ft->IC_names[i]) == ic_name)
      {
        ic_index = i;
        break;
      }
    }
    if (ic_index < 0)
      return;
    // std::cout << "IC SETTING " << el->get_code_instance()->get_func_table()->numfields_C2 << "  " << el->get_code_instance()->get_func_table()->numfields_C1 << "  NNODE " << this->nnode() << std::endl;
    // First set the coordinates
    for (unsigned int ni = 0; ni < this->nnode(); ni++)
    {
      pyoomph::Node *nodept = dynamic_cast<pyoomph::Node *>(this->node_pt(ni));
      for (unsigned int i = 0; i < nodept->ndim(); i++)
        x_buffer[i] = nodept->x((resetting_first_step ? 1 : 0), i);
      for (unsigned int i = 0; i < nodept->nlagrangian(); i++)
        x_lagr[i] = nodept->xi(i);
      if (nodal_normals.count(nodept))
      {
        for (unsigned int i = 0; i < nodal_dim; i++)
        {
          normal[i] = nodal_normals[nodept][i];
        }
      }
      else
      {
        for (unsigned int i = 0; i < 3; i++)
          normal[i] = 0;
      }

      for (unsigned int d = 0; d < nodept->ndim(); d++)
      {
        int valindex = -1 - d;
        Generic_SetInitialCondition(el, nodept->variable_position_pt(), el->get_code_instance(), valindex, d, x_buffer, x_lagr, normal, true, resetting_first_step, ic_index);
      }
    }

    for (unsigned int ni = 0; ni < this->nnode(); ni++)
    {
      pyoomph::Node *nodept = dynamic_cast<pyoomph::Node *>(this->node_pt(ni));
      for (unsigned int i = 0; i < nodept->ndim(); i++)
        x_buffer[i] = nodept->x(i);
      for (unsigned int i = 0; i < nodept->nlagrangian(); i++)
        x_lagr[i] = nodept->xi(i);
      if (nodal_normals.count(nodept))
      {
        for (unsigned int i = 0; i < nodal_dim; i++)
        {
          normal[i] = nodal_normals[nodept][i];
        }
      }
      else
      {
        for (unsigned int i = 0; i < 3; i++)
          normal[i] = 0;
      }
      unsigned offset = 0;
      for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C2TB_basebulk; fieldindex++)
      {
        Generic_SetInitialCondition(el, nodept, el->get_code_instance(), fieldindex + offset, fieldindex + offset, x_buffer, x_lagr, normal, true, false, ic_index);
      }
      offset += ft->numfields_C2TB_basebulk;
      for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C2_basebulk; fieldindex++)
      {
        Generic_SetInitialCondition(el, nodept, el->get_code_instance(), fieldindex + offset, fieldindex + offset, x_buffer, x_lagr, normal, true, false, ic_index);
      }
      offset += ft->numfields_C2_basebulk;
      for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C1TB_basebulk; fieldindex++)
      {
        Generic_SetInitialCondition(el, nodept, el->get_code_instance(), fieldindex + offset, fieldindex + offset, x_buffer, x_lagr, normal, true, false, ic_index);
      }
      offset += ft->numfields_C1TB_basebulk;
      for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C1_basebulk; fieldindex++)
      {
        Generic_SetInitialCondition(el, nodept, el->get_code_instance(), fieldindex + offset, fieldindex + offset, x_buffer, x_lagr, normal, true, false, ic_index);
      }
    }

    if (ft->numfields_D2TB || ft->numfields_D2 || ft->numfields_D1TB || ft->numfields_D1)
    {
      for (unsigned int ei = 0; ei < this->nelement(); ei++)
      {
        auto *el = dynamic_cast<BulkElementBase *>(this->element_pt(ei));
        if (ft->numfields_D2TB)
        {
          for (unsigned int ni = 0; ni < el->get_eleminfo()->nnode_C2TB; ni++)
          {
            pyoomph::Node *nodept = dynamic_cast<pyoomph::Node *>(el->node_pt(el->get_node_index_C2TB_to_element(ni)));
            for (unsigned int i = 0; i < nodept->ndim(); i++)
              x_buffer[i] = nodept->x(i);
            for (unsigned int i = 0; i < nodept->nlagrangian(); i++)
              x_lagr[i] = nodept->xi(i);
            for (unsigned int fieldindex = 0; fieldindex < ft->numfields_D2TB; fieldindex++)
            {
              Generic_SetInitialCondition(el, el->get_D2TB_nodal_data(fieldindex), el->get_code_instance(), el->get_D2TB_buffer_index(fieldindex), el->get_D2TB_node_index(fieldindex, ni), x_buffer, x_lagr, normal, true, false, ic_index);
            }
          }
        }
        if (ft->numfields_D2)
        {
          for (unsigned int ni = 0; ni < el->get_eleminfo()->nnode_C2; ni++)
          {
            pyoomph::Node *nodept = dynamic_cast<pyoomph::Node *>(el->node_pt(el->get_node_index_C2_to_element(ni)));
            for (unsigned int i = 0; i < nodept->ndim(); i++)
              x_buffer[i] = nodept->x(i);
            for (unsigned int i = 0; i < nodept->nlagrangian(); i++)
              x_lagr[i] = nodept->xi(i);
            for (unsigned int fieldindex = 0; fieldindex < ft->numfields_D2; fieldindex++)
            {
              Generic_SetInitialCondition(el, el->get_D2_nodal_data(fieldindex), el->get_code_instance(), el->get_D2_buffer_index(fieldindex), el->get_D2_node_index(fieldindex, ni), x_buffer, x_lagr, normal, true, false, ic_index);
            }
          }
        }
        if (ft->numfields_D1TB)
        {
          for (unsigned int ni = 0; ni < el->get_eleminfo()->nnode_C1TB; ni++)
          {
            pyoomph::Node *nodept = dynamic_cast<pyoomph::Node *>(el->node_pt(el->get_node_index_C1TB_to_element(ni)));
            for (unsigned int i = 0; i < nodept->ndim(); i++)
              x_buffer[i] = nodept->x(i);
            for (unsigned int i = 0; i < nodept->nlagrangian(); i++)
              x_lagr[i] = nodept->xi(i);
            for (unsigned int fieldindex = 0; fieldindex < ft->numfields_D1TB; fieldindex++)
            {
              Generic_SetInitialCondition(el, el->get_D1TB_nodal_data(fieldindex), el->get_code_instance(), el->get_D1TB_buffer_index(fieldindex), el->get_D1TB_node_index(fieldindex, ni), x_buffer, x_lagr, normal, true, false, ic_index);
            }
          }
        }
        if (ft->numfields_D1)
        {
          for (unsigned int ni = 0; ni < el->get_eleminfo()->nnode_C1; ni++)
          {
            pyoomph::Node *nodept = dynamic_cast<pyoomph::Node *>(el->node_pt(el->get_node_index_C1_to_element(ni)));
            for (unsigned int i = 0; i < nodept->ndim(); i++)
              x_buffer[i] = nodept->x(i);
            for (unsigned int i = 0; i < nodept->nlagrangian(); i++)
              x_lagr[i] = nodept->xi(i);
            for (unsigned int fieldindex = 0; fieldindex < ft->numfields_D1; fieldindex++)
            {
              Generic_SetInitialCondition(el, el->get_D1_nodal_data(fieldindex), el->get_code_instance(), el->get_D1_buffer_index(fieldindex), el->get_D1_node_index(fieldindex, ni), x_buffer, x_lagr, normal, true, false, ic_index);
            }
          }
        }
      }
    }

    if (!this->nnode()) // This happens for interface meshes. Here, we also can eval the normal
    {
      for (unsigned int ei = 0; ei < this->nelement(); ei++)
      {
        auto *el = dynamic_cast<BulkElementBase *>(this->element_pt(ei));
        auto *iel = dynamic_cast<InterfaceElementBase *>(this->element_pt(ei));
        for (unsigned int ni = 0; ni < el->nnode(); ni++)
        {
          normal[0] = normal[1] = normal[2] = 0.0;
          if (iel)
          {
            oomph::Vector<double> sinter(iel->ndim(), 0.0);
            iel->local_coordinate_of_node(ni, sinter);
            oomph::Vector<double> nbuff(iel->nodal_dimension(), 0.0);
            iel->get_normal_at_s(sinter, nbuff, NULL, NULL);
            for (unsigned int jnormd = 0; jnormd < iel->nodal_dimension(); jnormd++)
              normal[jnormd] = nbuff[jnormd];
          }
          pyoomph::Node *nodept = dynamic_cast<pyoomph::Node *>(el->node_pt(ni));
          for (unsigned int i = 0; i < nodept->ndim(); i++)
            x_buffer[i] = nodept->x(i);
          for (unsigned int i = 0; i < nodept->nlagrangian(); i++)
            x_lagr[i] = nodept->xi(i);

          for (unsigned int d = 0; d < nodept->ndim(); d++)
          {
            int valindex = -1 - d;
            Generic_SetInitialCondition(el, nodept->variable_position_pt(), el->get_code_instance(), valindex, d, x_buffer, x_lagr, normal, true, resetting_first_step, ic_index);
          }

          for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C2TB_basebulk; fieldindex++)
          {
            Generic_SetInitialCondition(el, nodept, el->get_code_instance(), fieldindex + ft->buffer_offset_C2TB_basebulk, fieldindex + +ft->buffer_offset_C2TB_basebulk, x_buffer, x_lagr, normal, true, false, ic_index);
          }
          for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C2_basebulk; fieldindex++)
          {
            Generic_SetInitialCondition(el, nodept, el->get_code_instance(), fieldindex + ft->buffer_offset_C2_basebulk, fieldindex + ft->buffer_offset_C2_basebulk, x_buffer, x_lagr, normal, true, false, ic_index);
          }
          for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C1TB_basebulk; fieldindex++)
          {
            Generic_SetInitialCondition(el, nodept, el->get_code_instance(), fieldindex + ft->buffer_offset_C1TB_basebulk, fieldindex + ft->buffer_offset_C1TB_basebulk, x_buffer, x_lagr, normal, true, false, ic_index);
          }
          for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C1_basebulk; fieldindex++)
          {
            Generic_SetInitialCondition(el, nodept, el->get_code_instance(), fieldindex + ft->buffer_offset_C1_basebulk, fieldindex + ft->buffer_offset_C1_basebulk, x_buffer, x_lagr, normal, true, false, ic_index);
          }

          for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C2TB - ft->numfields_C2TB_basebulk; fieldindex++)
          {
            std::string fieldname = ft->fieldnames_C2TB[ft->numfields_C2TB_basebulk + fieldindex];
            unsigned interf_id = el->get_code_instance()->resolve_interface_dof_id(fieldname);
            unsigned valindex = dynamic_cast<oomph::BoundaryNodeBase *>(nodept)->index_of_first_value_assigned_by_face_element(interf_id);
            Generic_SetInitialCondition(el, nodept, el->get_code_instance(), fieldindex + ft->buffer_offset_C2TB_interf, valindex, x_buffer, x_lagr, normal, true, false, ic_index);
          }
          for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C2 - ft->numfields_C2_basebulk; fieldindex++)
          {
            std::string fieldname = ft->fieldnames_C2[ft->numfields_C2_basebulk + fieldindex];
            unsigned interf_id = el->get_code_instance()->resolve_interface_dof_id(fieldname);
            unsigned valindex = dynamic_cast<oomph::BoundaryNodeBase *>(nodept)->index_of_first_value_assigned_by_face_element(interf_id);
            Generic_SetInitialCondition(el, nodept, el->get_code_instance(), fieldindex + ft->buffer_offset_C2_interf, valindex, x_buffer, x_lagr, normal, true, false, ic_index);
          }
          for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C1TB - ft->numfields_C1TB_basebulk; fieldindex++)
          {
            std::string fieldname = ft->fieldnames_C1TB[ft->numfields_C1TB_basebulk + fieldindex];
            unsigned interf_id = el->get_code_instance()->resolve_interface_dof_id(fieldname);
            unsigned valindex = dynamic_cast<oomph::BoundaryNodeBase *>(nodept)->index_of_first_value_assigned_by_face_element(interf_id);
            Generic_SetInitialCondition(el, nodept, el->get_code_instance(), fieldindex + ft->buffer_offset_C1TB_interf, valindex, x_buffer, x_lagr, normal, true, false, ic_index);
          }

          for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C1 - ft->numfields_C1_basebulk; fieldindex++)
          {
            std::string fieldname = ft->fieldnames_C1[ft->numfields_C1_basebulk + fieldindex];
            unsigned interf_id = el->get_code_instance()->resolve_interface_dof_id(fieldname);
            unsigned valindex = dynamic_cast<oomph::BoundaryNodeBase *>(nodept)->index_of_first_value_assigned_by_face_element(interf_id);
            Generic_SetInitialCondition(el, nodept, el->get_code_instance(), fieldindex + ft->buffer_offset_C1_interf, valindex, x_buffer, x_lagr, normal, true, false, ic_index);
          }
        }
      }
    }

    for (unsigned int ei = 0; ei < this->nelement(); ei++)
    {
      auto *el = dynamic_cast<BulkElementBase *>(this->element_pt(ei));
      oomph::Vector<double> xcenter = el->get_Eulerian_midpoint_from_local_coordinate();
      oomph::Vector<double> xlagr = el->get_Lagrangian_midpoint_from_local_coordinate();
      for (unsigned int i = 0; i < xcenter.size(); i++)
        x_buffer[i] = xcenter[i];
      for (unsigned int i = 0; i < xlagr.size(); i++)
        x_lagr[i] = xlagr[i];

      for (unsigned int fieldindex = 0; fieldindex < el->get_code_instance()->get_func_table()->numfields_DL; fieldindex++)
      {
        oomph::Vector<double> np(el->nodal_dimension(), 0.0);
        oomph::Vector<double> np_lagr(el->nodal_dimension(), 0.0);
        oomph::Vector<double> s(el->dim(), 0.5 * (el->s_min() + el->s_max()));
        for (unsigned int j = 0; j < s.size(); j++)
        {
          double old = s[j];
          s[j] = el->s_min();
          el->interpolated_x(s, np);
          el->interpolated_xi(s, np_lagr);
          for (unsigned int i = 0; i < xcenter.size(); i++)
            x_buffer[i] = np[i];
          for (unsigned int i = 0; i < xlagr.size(); i++)
            x_lagr[i] = np_lagr[i];
          Generic_SetInitialCondition(el, this->element_pt(ei)->internal_data_pt(fieldindex + ft->internal_offset_DL), el->get_code_instance(), fieldindex + ft->buffer_offset_DL, 0, x_buffer, x_lagr, normal, false, false, ic_index);

          auto *ts = this->element_pt(ei)->internal_data_pt(fieldindex + ft->internal_offset_DL)->time_stepper_pt();
          oomph::Vector<double> vmin(ts->ntstorage());
          for (unsigned t = 0; t < vmin.size(); t++)
            vmin[t] = this->element_pt(ei)->internal_data_pt(fieldindex + ft->internal_offset_DL)->value(t, 0);

          s[j] = el->s_max();
          el->interpolated_x(s, np);
          el->interpolated_xi(s, np_lagr);
          for (unsigned int i = 0; i < xcenter.size(); i++)
            x_buffer[i] = np[i];
          for (unsigned int i = 0; i < xlagr.size(); i++)
            x_lagr[i] = np_lagr[i];
          Generic_SetInitialCondition(el, this->element_pt(ei)->internal_data_pt(fieldindex + ft->internal_offset_DL), el->get_code_instance(), fieldindex + ft->buffer_offset_DL, 0, x_buffer, x_lagr, normal, false, false, ic_index);
          oomph::Vector<double> vmax(ts->ntstorage());
          for (unsigned t = 0; t < vmax.size(); t++)
            vmax[t] = this->element_pt(ei)->internal_data_pt(fieldindex + ft->internal_offset_DL)->value(t, 0);

          double denom = el->s_max() - el->s_min();
          for (unsigned t = 0; t < vmax.size(); t++)
            this->element_pt(ei)->internal_data_pt(fieldindex + ft->internal_offset_DL)->set_value(t, j + 1, (vmax[t] - vmin[t]) / denom);

          s[j] = old;
        }

        Generic_SetInitialCondition(el, this->element_pt(ei)->internal_data_pt(fieldindex + ft->internal_offset_DL), el->get_code_instance(), fieldindex + ft->buffer_offset_DL, 0, x_buffer, x_lagr, normal, false, false, ic_index);
      }

      for (unsigned int i = 0; i < xcenter.size(); i++)
        x_buffer[i] = xcenter[i];
      for (unsigned int i = 0; i < xlagr.size(); i++)
        x_lagr[i] = xlagr[i];
      for (unsigned int fieldindex = 0; fieldindex < el->get_code_instance()->get_func_table()->numfields_D0; fieldindex++)
      {
        //        std::cout << "d0 ic " << x_lagr[0] << "  " << x_lagr[1] << "  " << xlagr[0] << "  " << xlagr[1] << std::endl;
        Generic_SetInitialCondition(el, this->element_pt(ei)->internal_data_pt(fieldindex + ft->internal_offset_D0), el->get_code_instance(), fieldindex + ft->buffer_offset_D0, 0, x_buffer, x_lagr, normal, false, false, ic_index);
      }
    }
  }

  void Generic_SetDirichletCondition(BulkElementBase *elempt, oomph::Data *data, DynamicBulkElementInstance *ci, int fieldindex, unsigned valindex, double *x_buffer, double *x_lagr, double *normal, bool only_update_vals)
  {
    auto *ts = data->time_stepper_pt();
    auto *Time_pt = ts->time_pt();
    for (unsigned t = 0; t < Time_pt->ndt(); t++)
    {
      double time_local = Time_pt->time(t);
      double default_val = 0.0; // data->value(t,valindex)
      default_val = data->value(t, valindex);
      double val = ci->get_func_table()->DirichletConditionFunc(elempt->get_eleminfo(), fieldindex, x_buffer, x_lagr, normal, time_local, default_val);
      data->set_value(t, valindex, val);
      if (!only_update_vals)
        data->pin(valindex);
    }
  }

  void Mesh::set_dirichlet_active(std::string name, bool active)
  {
    int index = -1;
    if (name == "mesh_x")
      name = "coordinate_x";
    if (name == "mesh_y")
      name = "coordinate_y";
    if (name == "mesh_z")
      name = "coordinate_z";
    JITFuncSpec_Table_FiniteElement_t *ft;
    if (!this->nelement())
    {
      if (!codeinst)
        throw_runtime_error("Cannot toggle a Dirichlet active without elements or JIT code instance.")
            ft = codeinst->get_func_table();
    }
    else
    {
      auto *el = dynamic_cast<BulkElementBase *>(this->element_pt(0));
      ft = el->get_code_instance()->get_func_table();
    }

    for (unsigned int i = 0; i < ft->Dirichlet_set_size; i++)
    {
      if (ft->Dirichlet_names[i] && std::string(ft->Dirichlet_names[i]) == name)
      {
        index = i;
        break;
      }
    }
    if (index == -1)
      throw_runtime_error("Cannot set a Dirichlet condition active or not for an unknown field " + name);
    std::cout << "TOGGLING DIRICHLET ACTIVE AT INDEX " << index << " TO " << active << " CORESPONDING TO " << name <<std::endl;
    dirichlet_active[index] = active;
  }

  bool Mesh::get_dirichlet_active(std::string name)
  {
    int index = -1;
    if (name == "mesh_x")
      name = "coordinate_x";
    if (name == "mesh_y")
      name = "coordinate_y";
    if (name == "mesh_z")
      name = "coordinate_z";
    DynamicBulkElementInstance *ci=this->codeinst;
    if (!ci)
    {	
    	auto *el = dynamic_cast<BulkElementBase *>(this->element_pt(0));
    	ci=el->get_code_instance();
    }
    auto *ft = ci->get_func_table();
    for (unsigned int i = 0; i < ft->Dirichlet_set_size; i++)
    {
      if (ft->Dirichlet_names[i] && std::string(ft->Dirichlet_names[i]) == name)
      {
        index = i;
        break;
      }
    }
    if (index == -1)
      throw_runtime_error("Cannot get whether a Dirichlet condition is active or not for an unknown field " + name);
    return dirichlet_active[index];
  }

  unsigned Mesh::get_nodal_dimension()
  {
    if (!this->nnode())
    {
      if (this->nelement())
      {
        return dynamic_cast<BulkElementBase *>(this->element_pt(0))->nodal_dimension();
      }
      else
      {
        return 0;
      }
    }
    return this->node_pt(0)->ndim();
  }

  int Mesh::get_element_dimension()
  {
    if (!this->nelement())
      return -1;
    return dynamic_cast<BulkElementBase *>(this->element_pt(0))->dim();
  }

  void Mesh::pin_noncontributing_dofs()
  {
    if (!this->nelement()) return;
    if (!this->problem->has_empty_jacobian_rows_marked()) return;
    auto *el0 = dynamic_cast<BulkElementBase *>(this->element_pt(0));
    auto *ft = el0->get_code_instance()->get_func_table();
    int Doffset = 3;
    for (unsigned int ei = 0; ei < this->nelement(); ei++)
    {
      auto *el = dynamic_cast<BulkElementBase *>(this->element_pt(ei));
      for (unsigned int ni=0;ni<el->nnode();ni++)
      {
        pyoomph::Node *nodept = dynamic_cast<pyoomph::Node *>(el->node_pt(ni));
        // Handle moving mesh dofs
        for (unsigned int d = 0; d < nodept->ndim(); d++)
        {
          int valindex = -1 - d;                    
          if (problem->is_field_removed_from_dofs_due_to_missing_jacobian_row(ft->dirichlet_field_index_to_global_field_index[valindex + Doffset]))
          {
            nodept->variable_position_pt()->pin(d);
          }
        }

        // Handling continuous bulk dofs
        for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C2TB_basebulk+ft->numfields_C2_basebulk+ft->numfields_C1TB_basebulk+ft->numfields_C1_basebulk; fieldindex++)
        {
          if (problem->is_field_removed_from_dofs_due_to_missing_jacobian_row(ft->dirichlet_field_index_to_global_field_index[fieldindex  + Doffset]))
          {
            nodept->pin(fieldindex);            
          }
        }                
        unsigned offset = 0;
      }
      // Handling discontinuous dofs
      if (ft->numfields_D2TB || ft->numfields_D2 || ft->numfields_D1TB || ft->numfields_D1)
      {
        for (unsigned int fieldindex = 0; fieldindex < ft->numfields_D2TB; fieldindex++)
        {
          unsigned bindex = el->get_D2TB_buffer_index(fieldindex);
          if (problem->is_field_removed_from_dofs_due_to_missing_jacobian_row(ft->dirichlet_field_index_to_global_field_index[Doffset + bindex]))
          {
            oomph::Data *data = el->get_D2TB_nodal_data(fieldindex);
            for (unsigned int nj = 0; nj < data->nvalue(); nj++) data->pin(nj);            
          }
        }
        for (unsigned int fieldindex = 0; fieldindex < ft->numfields_D2; fieldindex++)
        {
          unsigned bindex = el->get_D2_buffer_index(fieldindex);
          if (problem->is_field_removed_from_dofs_due_to_missing_jacobian_row(ft->dirichlet_field_index_to_global_field_index[Doffset + bindex]))
          {
            oomph::Data *data = el->get_D2_nodal_data(fieldindex);
            for (unsigned int nj = 0; nj < data->nvalue(); nj++) data->pin(nj);            
          }
        }
        for (unsigned int fieldindex = 0; fieldindex < ft->numfields_D1TB; fieldindex++)
        {
          unsigned bindex = el->get_D1TB_buffer_index(fieldindex);
          if (problem->is_field_removed_from_dofs_due_to_missing_jacobian_row(ft->dirichlet_field_index_to_global_field_index[Doffset + bindex]))
          {
            oomph::Data *data = el->get_D1TB_nodal_data(fieldindex);
            for (unsigned int nj = 0; nj < data->nvalue(); nj++) data->pin(nj);            
          }
        }
        for (unsigned int fieldindex = 0; fieldindex < ft->numfields_D1; fieldindex++)
        {
          unsigned bindex = el->get_D1_buffer_index(fieldindex);
          if (problem->is_field_removed_from_dofs_due_to_missing_jacobian_row(ft->dirichlet_field_index_to_global_field_index[Doffset + bindex]))
          {
            oomph::Data *data = el->get_D1_nodal_data(fieldindex);
            for (unsigned int nj = 0; nj < data->nvalue(); nj++) data->pin(nj);            
          }
        }        
      }
      // Handling interface dofs
      if (dynamic_cast<InterfaceElementBase*>(el))
      {
          for (unsigned int ni=0;ni<el->nnode();ni++)
          {
            pyoomph::Node *nodept = dynamic_cast<pyoomph::Node *>(el->node_pt(ni));
            for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C2TB - ft->numfields_C2TB_basebulk; fieldindex++)
            {
              std::string fieldname = ft->fieldnames_C2TB[ft->numfields_C2TB_basebulk + fieldindex];
              unsigned interf_id = el->get_code_instance()->resolve_interface_dof_id(fieldname);              
              if (problem->is_field_removed_from_dofs_due_to_missing_jacobian_row(ft->dirichlet_field_index_to_global_field_index[fieldindex + ft->buffer_offset_C2TB_interf + Doffset]))
              {
                nodept->pin(dynamic_cast<oomph::BoundaryNodeBase *>(nodept)->index_of_first_value_assigned_by_face_element(interf_id));
              }
            }
            for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C2 - ft->numfields_C2_basebulk; fieldindex++)
            {
              std::string fieldname = ft->fieldnames_C2[ft->numfields_C2_basebulk + fieldindex];
              unsigned interf_id = el->get_code_instance()->resolve_interface_dof_id(fieldname);              
              if (problem->is_field_removed_from_dofs_due_to_missing_jacobian_row(ft->dirichlet_field_index_to_global_field_index[fieldindex + ft->buffer_offset_C2_interf + Doffset]))
              {
                nodept->pin(dynamic_cast<oomph::BoundaryNodeBase *>(nodept)->index_of_first_value_assigned_by_face_element(interf_id));
              }
            }
            for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C1TB - ft->numfields_C1TB_basebulk; fieldindex++)
            {
              std::string fieldname = ft->fieldnames_C1TB[ft->numfields_C1TB_basebulk + fieldindex];
              unsigned interf_id = el->get_code_instance()->resolve_interface_dof_id(fieldname);              
              if (problem->is_field_removed_from_dofs_due_to_missing_jacobian_row(ft->dirichlet_field_index_to_global_field_index[fieldindex + ft->buffer_offset_C1TB_interf + Doffset]))
              {
                nodept->pin(dynamic_cast<oomph::BoundaryNodeBase *>(nodept)->index_of_first_value_assigned_by_face_element(interf_id));
              }
            }
            for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C1 - ft->numfields_C1_basebulk; fieldindex++)
            {
              std::string fieldname = ft->fieldnames_C1[ft->numfields_C1_basebulk + fieldindex];
              unsigned interf_id = el->get_code_instance()->resolve_interface_dof_id(fieldname);              
              if (problem->is_field_removed_from_dofs_due_to_missing_jacobian_row(ft->dirichlet_field_index_to_global_field_index[fieldindex + ft->buffer_offset_C1_interf + Doffset]))
              {
                nodept->pin(dynamic_cast<oomph::BoundaryNodeBase *>(nodept)->index_of_first_value_assigned_by_face_element(interf_id));
              }
            }
          }
      }
      // Handling elemental dofs
      for (unsigned int fieldindex = 0; fieldindex < ft->numfields_DL; fieldindex++)
      {
        if (problem->is_field_removed_from_dofs_due_to_missing_jacobian_row(ft->dirichlet_field_index_to_global_field_index[fieldindex + ft->buffer_offset_DL + Doffset]))
        {
         oomph::Data *data=this->element_pt(ei)->internal_data_pt(ft->internal_offset_DL + fieldindex);
         for (unsigned int nj = 0; nj < data->nvalue(); nj++) data->pin(nj);
        }
      }
      for (unsigned int fieldindex = 0; fieldindex < ft->numfields_D0; fieldindex++)
      {
        if (problem->is_field_removed_from_dofs_due_to_missing_jacobian_row(ft->dirichlet_field_index_to_global_field_index[fieldindex + ft->buffer_offset_D0 + Doffset]))
        {
         oomph::Data *data=this->element_pt(ei)->internal_data_pt(ft->internal_offset_D0 + fieldindex);
         for (unsigned int nj = 0; nj < data->nvalue(); nj++) data->pin(nj);
        }
      }
    }
  }

  void Mesh::setup_Dirichlet_conditions(bool only_update_vals)
  {
    double x_buffer[3] = {0, 0, 0};
    double x_lagr[3] = {0, 0, 0};
    double normal[3] = {0, 0, 0};
    if (!this->nelement())
      return;
    auto *el = dynamic_cast<BulkElementBase *>(this->element_pt(0));
    auto *ft = el->get_code_instance()->get_func_table();
    int Doffset = 3;

    for (unsigned int ni = 0; ni < this->nnode(); ni++)
    {
      pyoomph::Node *nodept = dynamic_cast<pyoomph::Node *>(this->node_pt(ni));
      for (unsigned int i = 0; i < nodept->ndim(); i++)
        x_buffer[i] = nodept->x(i);
      for (unsigned int i = 0; i < nodept->nlagrangian(); i++)
        x_lagr[i] = nodept->xi(i);

      for (unsigned int d = 0; d < nodept->ndim(); d++)
      {
        int valindex = -1 - d;
        if (dirichlet_active[valindex + Doffset])
        {
          Generic_SetDirichletCondition(el, nodept->variable_position_pt(), el->get_code_instance(), valindex, d, x_buffer, x_lagr, normal, only_update_vals);
        }
      }
    }

    for (unsigned int ni = 0; ni < this->nnode(); ni++)
    {
      pyoomph::Node *nodept = dynamic_cast<pyoomph::Node *>(this->node_pt(ni));
      for (unsigned int i = 0; i < nodept->ndim(); i++)
        x_buffer[i] = nodept->x(i);
      for (unsigned int i = 0; i < nodept->nlagrangian(); i++)
        x_lagr[i] = nodept->xi(i);

      unsigned offset = 0;
      for (unsigned int fieldindex = 0; fieldindex < el->get_code_instance()->get_func_table()->numfields_C2TB_basebulk; fieldindex++)
      {
        if (dirichlet_active[fieldindex + offset + Doffset])
        {
          Generic_SetDirichletCondition(el, nodept, el->get_code_instance(), fieldindex + offset, fieldindex + offset, x_buffer, x_lagr, normal, only_update_vals);
        }
      }
      offset += el->get_code_instance()->get_func_table()->numfields_C2TB_basebulk;
      for (unsigned int fieldindex = 0; fieldindex < el->get_code_instance()->get_func_table()->numfields_C2_basebulk; fieldindex++)
      {
        if (dirichlet_active[fieldindex + offset + Doffset])
        {
          Generic_SetDirichletCondition(el, nodept, el->get_code_instance(), fieldindex + offset, fieldindex + offset, x_buffer, x_lagr, normal, only_update_vals);
        }
      }
      offset += el->get_code_instance()->get_func_table()->numfields_C2_basebulk;
      for (unsigned int fieldindex = 0; fieldindex < el->get_code_instance()->get_func_table()->numfields_C1TB_basebulk; fieldindex++)
      {
        if (dirichlet_active[fieldindex + offset + Doffset])
        {
          Generic_SetDirichletCondition(el, nodept, el->get_code_instance(), fieldindex + offset, fieldindex + offset, x_buffer, x_lagr, normal, only_update_vals);
        }
      }
      offset += el->get_code_instance()->get_func_table()->numfields_C1TB_basebulk;
      for (unsigned int fieldindex = 0; fieldindex < el->get_code_instance()->get_func_table()->numfields_C1_basebulk; fieldindex++)
      {
        if (dirichlet_active[fieldindex + offset + Doffset])
        {
          Generic_SetDirichletCondition(el, nodept, el->get_code_instance(), fieldindex + offset, fieldindex + offset, x_buffer, x_lagr, normal, only_update_vals);
        }
      }
    }

    if (ft->numfields_D2TB || ft->numfields_D2 || ft->numfields_D1TB || ft->numfields_D1)
    {
      for (unsigned int ei = 0; ei < this->nelement(); ei++)
      {
        auto *el = dynamic_cast<BulkElementBase *>(this->element_pt(ei));
        if (ft->numfields_D2TB)
        {
          for (unsigned int ni = 0; ni < el->get_eleminfo()->nnode_C2TB; ni++)
          {
            pyoomph::Node *nodept = dynamic_cast<pyoomph::Node *>(el->node_pt(el->get_node_index_C2TB_to_element(ni)));
            for (unsigned int i = 0; i < nodept->ndim(); i++)
              x_buffer[i] = nodept->x(i);
            for (unsigned int i = 0; i < nodept->nlagrangian(); i++)
              x_lagr[i] = nodept->xi(i);
            for (unsigned int fieldindex = 0; fieldindex < ft->numfields_D2TB; fieldindex++)
            {
              unsigned bindex = el->get_D2TB_buffer_index(fieldindex);
              if (dirichlet_active[bindex + Doffset])
              {
                Generic_SetDirichletCondition(el, el->get_D2TB_nodal_data(fieldindex), el->get_code_instance(), bindex, el->get_D2TB_node_index(fieldindex, ni), x_buffer, x_lagr, normal, only_update_vals);
              }
            }
          }
        }

        if (ft->numfields_D2)
        {
          for (unsigned int ni = 0; ni < el->get_eleminfo()->nnode_C2; ni++)
          {
            pyoomph::Node *nodept = dynamic_cast<pyoomph::Node *>(el->node_pt(el->get_node_index_C2_to_element(ni)));
            for (unsigned int i = 0; i < nodept->ndim(); i++)
              x_buffer[i] = nodept->x(i);
            for (unsigned int i = 0; i < nodept->nlagrangian(); i++)
              x_lagr[i] = nodept->xi(i);
            for (unsigned int fieldindex = 0; fieldindex < ft->numfields_D2; fieldindex++)
            {
              unsigned bindex = el->get_D2_buffer_index(fieldindex);
              if (dirichlet_active[bindex + Doffset])
              {
                Generic_SetDirichletCondition(el, el->get_D2_nodal_data(fieldindex), el->get_code_instance(), bindex, el->get_D2_node_index(fieldindex, ni), x_buffer, x_lagr, normal, only_update_vals);
              }
            }
          }
        }

        if (ft->numfields_D1TB)
        {
          for (unsigned int ni = 0; ni < el->get_eleminfo()->nnode_C1TB; ni++)
          {
            pyoomph::Node *nodept = dynamic_cast<pyoomph::Node *>(el->node_pt(el->get_node_index_C1TB_to_element(ni)));
            for (unsigned int i = 0; i < nodept->ndim(); i++)
              x_buffer[i] = nodept->x(i);
            for (unsigned int i = 0; i < nodept->nlagrangian(); i++)
              x_lagr[i] = nodept->xi(i);
            for (unsigned int fieldindex = 0; fieldindex < ft->numfields_D1TB; fieldindex++)
            {
              unsigned bindex = el->get_D1TB_buffer_index(fieldindex);
              if (dirichlet_active[bindex + Doffset])
              {
                //                std::cout << "APPLYING DBC ON D1 " << std::string(ft->fieldnames_D1[fieldindex]) << " DBC INDEX " << bindex + Doffset << " ELEM HAS DIM " << el->dim() << std::endl;
                Generic_SetDirichletCondition(el, el->get_D1TB_nodal_data(fieldindex), el->get_code_instance(), bindex, el->get_D1TB_node_index(fieldindex, ni), x_buffer, x_lagr, normal, only_update_vals);
                //                std::cout << "CORRESPONDING DATA HAS BEEN SET TO " << el->get_D1_nodal_data(fieldindex)->value(el->get_D1_node_index(fieldindex,ni)) << std::endl;
              }
            }
          }
        }

        if (ft->numfields_D1)
        {
          for (unsigned int ni = 0; ni < el->get_eleminfo()->nnode_C1; ni++)
          {
            pyoomph::Node *nodept = dynamic_cast<pyoomph::Node *>(el->node_pt(el->get_node_index_C1_to_element(ni)));
            for (unsigned int i = 0; i < nodept->ndim(); i++)
              x_buffer[i] = nodept->x(i);
            for (unsigned int i = 0; i < nodept->nlagrangian(); i++)
              x_lagr[i] = nodept->xi(i);
            for (unsigned int fieldindex = 0; fieldindex < ft->numfields_D1; fieldindex++)
            {
              unsigned bindex = el->get_D1_buffer_index(fieldindex);
              if (dirichlet_active[bindex + Doffset])
              {
                //                std::cout << "APPLYING DBC ON D1 " << std::string(ft->fieldnames_D1[fieldindex]) << " DBC INDEX " << bindex + Doffset << " ELEM HAS DIM " << el->dim() << std::endl;
                Generic_SetDirichletCondition(el, el->get_D1_nodal_data(fieldindex), el->get_code_instance(), bindex, el->get_D1_node_index(fieldindex, ni), x_buffer, x_lagr, normal, only_update_vals);
                //                std::cout << "CORRESPONDING DATA HAS BEEN SET TO " << el->get_D1_nodal_data(fieldindex)->value(el->get_D1_node_index(fieldindex,ni)) << std::endl;
              }
            }
          }
        }
      }
    }

    if (!this->nnode()) // This happens for interface meshes. Here, we also can access the normal, since we do it on an elemental basis
    {
      for (unsigned int ei = 0; ei < this->nelement(); ei++)
      {
        auto *el = dynamic_cast<BulkElementBase *>(this->element_pt(ei));
        auto *iel = dynamic_cast<InterfaceElementBase *>(el);
        for (unsigned int ni = 0; ni < el->nnode(); ni++)
        {
          normal[0] = normal[1] = normal[2] = 0.0;
          if (iel)
          {
            oomph::Vector<double> sinter(iel->ndim(), 0.0);
            iel->local_coordinate_of_node(ni, sinter);
            oomph::Vector<double> nbuff(iel->nodal_dimension(), 0.0);
            iel->get_normal_at_s(sinter, nbuff, NULL, NULL);
            for (unsigned int jnormd = 0; jnormd < iel->nodal_dimension(); jnormd++)
              normal[jnormd] = nbuff[jnormd];
          }

          pyoomph::Node *nodept = dynamic_cast<pyoomph::Node *>(el->node_pt(ni));
          for (unsigned int i = 0; i < nodept->ndim(); i++)
            x_buffer[i] = nodept->x(i);
          for (unsigned int i = 0; i < nodept->nlagrangian(); i++)
            x_lagr[i] = nodept->xi(i);
          for (unsigned int d = 0; d < nodept->ndim(); d++)
          {
            int valindex = -1 - d;
            if (dirichlet_active[valindex + Doffset])
            {
              Generic_SetDirichletCondition(el, nodept->variable_position_pt(), el->get_code_instance(), valindex, d, x_buffer, x_lagr, normal, only_update_vals);
            }
          }

          for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C2TB_basebulk; fieldindex++)
          {
            if (dirichlet_active[fieldindex + ft->buffer_offset_C2TB_basebulk + Doffset])
            {
              Generic_SetDirichletCondition(el, nodept, el->get_code_instance(), fieldindex + ft->buffer_offset_C2TB_basebulk, fieldindex + ft->buffer_offset_C2TB_basebulk, x_buffer, x_lagr, normal, only_update_vals);
            }
          }

          for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C2_basebulk; fieldindex++)
          {
            if (dirichlet_active[fieldindex + ft->buffer_offset_C2_basebulk + Doffset])
            {
              Generic_SetDirichletCondition(el, nodept, el->get_code_instance(), fieldindex + ft->buffer_offset_C2_basebulk, fieldindex + ft->buffer_offset_C2_basebulk, x_buffer, x_lagr, normal, only_update_vals);
            }
          }
          for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C1TB_basebulk; fieldindex++)
          {
            if (dirichlet_active[fieldindex + ft->buffer_offset_C1TB_basebulk + Doffset])
            {
              Generic_SetDirichletCondition(el, nodept, el->get_code_instance(), fieldindex + ft->buffer_offset_C1TB_basebulk, fieldindex + ft->buffer_offset_C1TB_basebulk, x_buffer, x_lagr, normal, only_update_vals);
            }
          }

          for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C1_basebulk; fieldindex++)
          {
            if (dirichlet_active[fieldindex + ft->buffer_offset_C1_basebulk + Doffset])
            {
              Generic_SetDirichletCondition(el, nodept, el->get_code_instance(), fieldindex + ft->buffer_offset_C1_basebulk, fieldindex + ft->buffer_offset_C1_basebulk, x_buffer, x_lagr, normal, only_update_vals);
            }
          }

          for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C2TB - ft->numfields_C2TB_basebulk; fieldindex++)
          {
            std::string fieldname = ft->fieldnames_C2TB[ft->numfields_C2TB_basebulk + fieldindex];
            unsigned interf_id = el->get_code_instance()->resolve_interface_dof_id(fieldname);
            unsigned valindex = dynamic_cast<oomph::BoundaryNodeBase *>(nodept)->index_of_first_value_assigned_by_face_element(interf_id);
            if (dirichlet_active[fieldindex + ft->buffer_offset_C2TB_interf + Doffset])
            {
              Generic_SetDirichletCondition(el, nodept, el->get_code_instance(), fieldindex + ft->buffer_offset_C2TB_interf, valindex, x_buffer, x_lagr, normal, only_update_vals);
            }
          }

          for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C2 - ft->numfields_C2_basebulk; fieldindex++)
          {
            std::string fieldname = ft->fieldnames_C2[ft->numfields_C2_basebulk + fieldindex];
            unsigned interf_id = el->get_code_instance()->resolve_interface_dof_id(fieldname);
            unsigned valindex = dynamic_cast<oomph::BoundaryNodeBase *>(nodept)->index_of_first_value_assigned_by_face_element(interf_id);
            if (dirichlet_active[fieldindex + ft->buffer_offset_C2_interf + Doffset])
            {
              Generic_SetDirichletCondition(el, nodept, el->get_code_instance(), fieldindex + ft->buffer_offset_C2_interf, valindex, x_buffer, x_lagr, normal, only_update_vals);
            }
          }
          for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C1TB - ft->numfields_C1TB_basebulk; fieldindex++)
          {
            std::string fieldname = ft->fieldnames_C1TB[ft->numfields_C1TB_basebulk + fieldindex];
            unsigned interf_id = el->get_code_instance()->resolve_interface_dof_id(fieldname);
            unsigned valindex = dynamic_cast<oomph::BoundaryNodeBase *>(nodept)->index_of_first_value_assigned_by_face_element(interf_id);
            if (dirichlet_active[fieldindex + ft->buffer_offset_C1TB_interf + Doffset])
            {
              Generic_SetDirichletCondition(el, nodept, el->get_code_instance(), fieldindex + ft->buffer_offset_C1TB_interf, valindex, x_buffer, x_lagr, normal, only_update_vals);
            }
          }

          for (unsigned int fieldindex = 0; fieldindex < ft->numfields_C1 - ft->numfields_C1_basebulk; fieldindex++)
          {
            std::string fieldname = ft->fieldnames_C1[ft->numfields_C1_basebulk + fieldindex];
            unsigned interf_id = el->get_code_instance()->resolve_interface_dof_id(fieldname);
            unsigned valindex = dynamic_cast<oomph::BoundaryNodeBase *>(nodept)->index_of_first_value_assigned_by_face_element(interf_id);
            if (dirichlet_active[fieldindex + ft->buffer_offset_C1_interf + Doffset])
            {
              Generic_SetDirichletCondition(el, nodept, el->get_code_instance(), fieldindex + ft->buffer_offset_C1_interf, valindex, x_buffer, x_lagr, normal, only_update_vals);
            }
          }
        }
      }
    }

    for (unsigned int ei = 0; ei < this->nelement(); ei++)
    {
      auto *el = dynamic_cast<BulkElementBase *>(this->element_pt(ei));
      oomph::Vector<double> xcenter = el->get_Eulerian_midpoint_from_local_coordinate();
      oomph::Vector<double> xlagr = el->get_Lagrangian_midpoint_from_local_coordinate();
      for (unsigned int i = 0; i < xcenter.size(); i++)
        x_buffer[i] = xcenter[i];
      for (unsigned int i = 0; i < xlagr.size(); i++)
        x_lagr[i] = xlagr[i];

      for (unsigned int fieldindex = 0; fieldindex < ft->numfields_DL; fieldindex++)
      {
        oomph::Vector<double> np(el->nodal_dimension(), 0.0);
        oomph::Vector<double> np_lagr(el->nodal_dimension(), 0.0);
        oomph::Vector<double> s(el->dim(), 0.5 * (el->s_min() + el->s_max()));
        for (unsigned int j = 0; j < s.size(); j++)
        {
          double old = s[j];
          s[j] = el->s_min();
          el->interpolated_x(s, np);
          el->interpolated_xi(s, np_lagr);
          for (unsigned int i = 0; i < xcenter.size(); i++)
            x_buffer[i] = np[i];
          for (unsigned int i = 0; i < xlagr.size(); i++)
            x_lagr[i] = np_lagr[i];
          if (dirichlet_active[fieldindex + ft->buffer_offset_DL + Doffset])
          {
            Generic_SetDirichletCondition(el, this->element_pt(ei)->internal_data_pt(ft->internal_offset_DL + fieldindex), el->get_code_instance(), fieldindex + ft->buffer_offset_DL, 0, x_buffer, x_lagr, normal, only_update_vals);

            auto *ts = this->element_pt(ei)->internal_data_pt(ft->internal_offset_DL + fieldindex)->time_stepper_pt();
            oomph::Vector<double> vmin(ts->ntstorage());
            for (unsigned t = 0; t < vmin.size(); t++)
              vmin[t] = this->element_pt(ei)->internal_data_pt(ft->internal_offset_DL + fieldindex)->value(t, 0);

            s[j] = el->s_max();
            el->interpolated_x(s, np);
            el->interpolated_xi(s, np_lagr);
            for (unsigned int i = 0; i < xcenter.size(); i++)
              x_buffer[i] = np[i];
            for (unsigned int i = 0; i < xlagr.size(); i++)
              x_lagr[i] = np_lagr[i];
            if (dirichlet_active[fieldindex + ft->buffer_offset_DL + Doffset])
            {
              Generic_SetDirichletCondition(el, this->element_pt(ei)->internal_data_pt(ft->internal_offset_DL + fieldindex), el->get_code_instance(), fieldindex + ft->buffer_offset_DL, 0, x_buffer, x_lagr, normal, only_update_vals);
            }
            oomph::Vector<double> vmax(ts->ntstorage());
            for (unsigned t = 0; t < vmax.size(); t++)
              vmax[t] = this->element_pt(ei)->internal_data_pt(ft->internal_offset_DL + fieldindex)->value(t, 0);

            double denom = el->s_max() - el->s_min();
            for (unsigned t = 0; t < vmax.size(); t++)
              this->element_pt(ei)->internal_data_pt(ft->internal_offset_DL + fieldindex)->set_value(t, j + 1, (vmax[t] - vmin[t]) / denom);
          }
          s[j] = old;
        }
        if (dirichlet_active[fieldindex + ft->buffer_offset_DL + Doffset])
        {
          Generic_SetDirichletCondition(el, this->element_pt(ei)->internal_data_pt(ft->internal_offset_DL + fieldindex), el->get_code_instance(), fieldindex + ft->buffer_offset_DL, 0, x_buffer, x_lagr, normal, only_update_vals);
        }
      }

      for (unsigned int i = 0; i < xcenter.size(); i++)
        x_buffer[i] = xcenter[i];
      for (unsigned int i = 0; i < xlagr.size(); i++)
        x_lagr[i] = xlagr[i];
      for (unsigned int fieldindex = 0; fieldindex < ft->numfields_D0; fieldindex++)
      {
        if (dirichlet_active[fieldindex + ft->buffer_offset_D0 + Doffset])
        {
          Generic_SetDirichletCondition(el, this->element_pt(ei)->internal_data_pt(ft->internal_offset_D0 + fieldindex), el->get_code_instance(), fieldindex + ft->buffer_offset_D0, 0, x_buffer, x_lagr, normal, only_update_vals);
        }
      }
    }
  }

  void ODEStorageMesh::setup_initial_conditions(bool resetting_first_step, std::string ic_name)
  {
    double x_buffer[3] = {0, 0, 0};
    double normal[3] = {0, 0, 0};
    for (unsigned int ei = 0; ei < this->nelement(); ei++)
    {
      int ic_index = -1;
      auto *ode = dynamic_cast<BulkElementODE0d *>(this->element_pt(ei));
      auto *ft = ode->get_code_instance()->get_func_table();
      for (unsigned int i = 0; i < ft->num_ICs; i++)
      {
        if (std::string(ft->IC_names[i]) == ic_name)
        {
          ic_index = i;
          break;
        }
      }
      if (ic_index < 0)
        continue;

      for (unsigned int fieldindex = 0; fieldindex < ode->get_code_instance()->get_func_table()->numfields_D0; fieldindex++)
      {
        Generic_SetInitialCondition(ode, ode->internal_data_pt(fieldindex), ode->get_code_instance(), fieldindex, 0, x_buffer, x_buffer, normal, false, false, ic_index);
      }
    }
  }

  void ODEStorageMesh::setup_Dirichlet_conditions(bool only_update_vals)
  {
    double x_buffer[3] = {0, 0, 0};
    double normal[3] = {0, 0, 0};
    unsigned Doffset = 3;
    for (unsigned int ei = 0; ei < this->nelement(); ei++)
    {
      auto *ode = dynamic_cast<BulkElementODE0d *>(this->element_pt(ei));
      for (unsigned int fieldindex = 0; fieldindex < ode->get_code_instance()->get_func_table()->numfields_D0; fieldindex++)
      {
        if (dirichlet_active[fieldindex + Doffset])
        {
          Generic_SetDirichletCondition(ode, ode->internal_data_pt(fieldindex), ode->get_code_instance(), fieldindex, 0, x_buffer, x_buffer, normal, only_update_vals);
        }
        else if (!only_update_vals)
        {
          ode->internal_data_pt(fieldindex)->unpin(0);
        }
      }
    }
  }

  unsigned ODEStorageMesh::add_ODE(std::string name, oomph::GeneralisedElement *ode)
  {
    unsigned res = this->nelement();
    if (name_to_index.count(name))
      throw_runtime_error("ODE with name " + name + " already added");
    this->add_element_pt(ode);
    name_to_index[name] = res;
    return res;
  }

  oomph::GeneralisedElement *ODEStorageMesh::get_ODE(std::string name)
  {
    if (!name_to_index.count(name))
      throw_runtime_error("ODE with name " + name + " not present");
    return this->element_pt(name_to_index[name]);
  }

  double ODEStorageMesh::get_temporal_error_norm_contribution()
  {
    if (!this->nelement())
      return 0.0;
    double res = 0.0;
    double denom = 0.0;
    for (unsigned int i = 0; i < this->nelement(); i++)
    {
      auto *ode = dynamic_cast<BulkElementBase *>(this->element_pt(i));
      DynamicBulkElementInstance *ci = ode->get_code_instance();
      auto *ft = ci->get_func_table();
      if (!ft->has_temporal_estimators)
        continue;
      unsigned numvars = ft->numfields_D0;
      for (unsigned int j = 0; j < numvars; j++)
      {
        if (ft->temporal_error_scales[j] == 0.0)
          continue;
        if (!ode->internal_data_pt(j)->is_pinned(0))
        {
          double nodal_err = ode->internal_data_pt(j)->time_stepper_pt()->temporal_error_in_value(ode->internal_data_pt(j), 0);
          res += nodal_err * nodal_err * ft->temporal_error_scales[i];
          denom += 1.0;
        }
      }
    }

    if (denom == 0)
      return 0.0;
    return res / denom;
  }

  InterfaceMesh::InterfaceMesh() : Mesh(), code(NULL), bulkmesh(NULL)
  {
    this->disable_adaptation();
    //		this->spatial_error_estimator_pt() = new DummyErrorEstimator();
  }
  InterfaceMesh::~InterfaceMesh()
  {
    for (unsigned i = 0; i < opposite_interior_facets.size(); i++)
      delete opposite_interior_facets[i];
    opposite_interior_facets.clear();
    //	 if (this->spatial_error_estimator_pt()) delete this->spatial_error_estimator_pt();
  }

  void InterfaceMesh::update_zeta_in_buffer()
  {
    for (unsigned int ie = 0; ie < this->nelement(); ie++)
    {
      BulkElementBase *be = dynamic_cast<BulkElementBase *>(this->element_pt(ie));
      DynamicBulkElementInstance *ci = be->get_code_instance();
      auto *functable = ci->get_func_table();
      auto &eleminfo = *be->get_eleminfo();
      unsigned offset_zeta=eleminfo.nodal_dim + functable->lagr_dim +be->dim(); // This is the offset for the zeta coordinate in the nodal data ( first Eulerian, then Lagrangian, then local coords. Finally zeta coords)
      auto *ft = ci->get_func_table();
      oomph::Vector<double> zeta(be->dim(), 0.0);
      oomph::Vector<double> sinter(be->dim(), 0.0);
      for (unsigned int in=0;in<be->nnode();in++)
      {                        
        for (unsigned int iz=0;iz<be->dim();iz++)
        {
          //std::cout << "SETTING ZETA " << in << "  " << iz << " to ["<< offset_zeta+iz <<"]" << be->zeta_nodal(in,0,iz) << std::endl;
          *(be->get_eleminfo()->nodal_coords[in][offset_zeta+iz])=be->zeta_nodal(in,0,iz);
        }        
      }
    }
  }

  void InterfaceMesh::fill_internal_facet_buffers(std::vector<BulkElementBase *> &internal_elements, std::vector<int> &internal_face_dir, std::vector<BulkElementBase *> &opposite_elements, std::vector<int> &opposite_face_dir, std::vector<int> &opposite_already_at_index)
  {
    internal_elements.clear();
    internal_face_dir.clear();
    opposite_elements.clear();
    opposite_face_dir.clear();
    opposite_already_at_index.clear();
    std::map<oomph::Node *, std::pair<BulkElementBase *, int>> nodemap;
    std::set<oomph::Node *> completed_nodes;
    for (unsigned int ie = 0; ie < this->nelement(); ie++)
    {
      BulkElementBase *be = dynamic_cast<BulkElementBase *>(this->element_pt(ie));
      if (be->dim() != 1)
        throw_runtime_error("DG on interfaces only works for 1d interfaces on 2d meshes at the moment");
      for (unsigned int ivn = 0; ivn < be->nvertex_node(); ivn++)
      {
        oomph::Node *npt = be->vertex_node_pt(ivn);
        if (npt->is_a_copy())
        {
          //       std::cout << "IS A COPY  " << npt << " -> " <<  npt->copied_node_pt() << std::endl;
          npt = npt->copied_node_pt();
        }
        if (!nodemap.count(npt))
        {
          if (completed_nodes.count(npt))
            throw_runtime_error("STRANGE, node already completed!");
          nodemap[npt] = std::make_pair(be, (ivn == 0 ? -1 : 1));
        }
        else
        {
          internal_elements.push_back(be);
          internal_face_dir.push_back((ivn == 0 ? -1 : 1));
          opposite_elements.push_back(nodemap[npt].first);
          opposite_face_dir.push_back(nodemap[npt].second);
          opposite_already_at_index.push_back(-1);
          completed_nodes.insert(npt);
        }
      }
    }
  }

  double InterfaceMesh::get_temporal_error_norm_contribution()
  {
    if (!this->nelement())
      return 0.0;
    BulkElementBase *be = dynamic_cast<BulkElementBase *>(this->element_pt(0));
    DynamicBulkElementInstance *ci = be->get_code_instance();
    auto *ft = ci->get_func_table();
    if (!ft->has_temporal_estimators)
      return 0.0;
    double res = 0.0;
    double denom = 0.0;

    std::vector<unsigned> add_indices_C1, add_indices_C2;
    for (unsigned int i = ft->numfields_C2_basebulk; i < ft->numfields_C2; i++)
    {
      add_indices_C2.push_back(ci->resolve_interface_dof_id(ft->fieldnames_C2[i]));
    }
    for (unsigned int i = ft->numfields_C1_basebulk; i < ft->numfields_C1; i++)
    {
      add_indices_C1.push_back(ci->resolve_interface_dof_id(ft->fieldnames_C1[i]));
    }

    std::set<pyoomph::Node *> handled_nodes;
    for (unsigned int ie = 0; ie < this->nelement(); ie++)
    {
      be = dynamic_cast<BulkElementBase *>(this->element_pt(ie));
      // C1 Nodes
      for (unsigned int in = 0; in < be->get_eleminfo()->nnode_C1; in++)
      {
        pyoomph::Node *n = dynamic_cast<pyoomph::Node *>(be->node_pt_C1(in));
        if (handled_nodes.count(n))
          continue;
        for (unsigned int j = 0; j < ft->numfields_C2_basebulk; j++)
        {
          if (ft->temporal_error_scales[j] == 0.0)
            continue;
          double nodal_err = n->time_stepper_pt()->temporal_error_in_value(n, j);
          res += nodal_err * nodal_err * ft->temporal_error_scales[j];
          denom += 1.0;
        }
        for (unsigned int j = 0; j < ft->numfields_C1_basebulk; j++)
        {
          if (ft->temporal_error_scales[j + ft->numfields_C2_basebulk] == 0.0)
            continue;
          double nodal_err = n->time_stepper_pt()->temporal_error_in_value(n, j + ft->numfields_C2_basebulk);
          res += nodal_err * nodal_err * ft->temporal_error_scales[j + ft->numfields_C2_basebulk];
          denom += 1.0;
        }
        pyoomph::BoundaryNode *bn = dynamic_cast<pyoomph::BoundaryNode *>(n);
        if (!bn)
          continue; // Should not happen
        // Now potential interface fields
        for (unsigned int j = 0; j < add_indices_C2.size(); j++)
        {
          if (ft->temporal_error_scales[j + ft->numfields_C2_basebulk + ft->numfields_C1_basebulk] == 0.0)
            continue;
          double nodal_err = bn->time_stepper_pt()->temporal_error_in_value(bn, bn->index_of_first_value_assigned_by_face_element(add_indices_C2[j]));
          res += nodal_err * nodal_err * ft->temporal_error_scales[j + ft->numfields_C2_basebulk + ft->numfields_C1_basebulk];
          denom += 1.0;
        }
        for (unsigned int j = 0; j < add_indices_C1.size(); j++)
        {
          if (ft->temporal_error_scales[j + ft->numfields_C2 + ft->numfields_C1_basebulk] == 0.0)
            continue;
          double nodal_err = bn->time_stepper_pt()->temporal_error_in_value(bn, bn->index_of_first_value_assigned_by_face_element(add_indices_C1[j]));
          res += nodal_err * nodal_err * ft->temporal_error_scales[j + ft->numfields_C2 + ft->numfields_C1_basebulk];
          denom += 1.0;
        }
        handled_nodes.insert(n);
      }
      // C2 Nodes, C1 are already handled now
      for (unsigned int in = 0; in < be->nnode(); in++)
      {
        pyoomph::Node *n = dynamic_cast<pyoomph::Node *>(be->node_pt(in));
        if (handled_nodes.count(n))
          continue;
        for (unsigned int j = 0; j < ft->numfields_C2_basebulk; j++)
        {
          if (ft->temporal_error_scales[j] == 0.0)
            continue;
          double nodal_err = n->time_stepper_pt()->temporal_error_in_value(n, j);
          res += nodal_err * nodal_err * ft->temporal_error_scales[j];
          denom += 1.0;
        }
        pyoomph::BoundaryNode *bn = dynamic_cast<pyoomph::BoundaryNode *>(n);
        if (!bn)
          continue; // Should not happen
        // Now potential interface fields
        for (unsigned int j = 0; j < add_indices_C2.size(); j++)
        {
          if (ft->temporal_error_scales[j + ft->numfields_C2_basebulk + ft->numfields_C1_basebulk] == 0.0)
            continue;
          double nodal_err = bn->time_stepper_pt()->temporal_error_in_value(bn, bn->index_of_first_value_assigned_by_face_element(add_indices_C2[j]));
          res += nodal_err * nodal_err * ft->temporal_error_scales[j + ft->numfields_C2_basebulk + ft->numfields_C1_basebulk];
          denom += 1.0;
        }
        handled_nodes.insert(n);
      }
    }

    for (unsigned int i = 0; i < ft->numfields_DL; i++)
    {
      if (ft->temporal_error_scales[i + ft->numfields_C2 + ft->numfields_C1] == 0.0)
        continue;
      for (unsigned int j = 0; j < this->nelement(); j++)
      {
        BulkElementBase *be = dynamic_cast<BulkElementBase *>(this->element_pt(j));
        oomph::Data *d = be->internal_data_pt(i);
        for (unsigned int v = 0; v < d->nvalue(); v++)
        {
          double derr = d->time_stepper_pt()->temporal_error_in_value(d, v);
          res += derr * derr * ft->temporal_error_scales[i + ft->numfields_C2 + ft->numfields_C1];
          denom += 1.0;
        }
      }
    }
    for (unsigned int i = 0; i < ft->numfields_D0; i++)
    {
      if (ft->temporal_error_scales[i + ft->numfields_C2 + ft->numfields_C1 + ft->numfields_DL] == 0.0)
        continue;
      for (unsigned int j = 0; j < this->nelement(); j++)
      {
        BulkElementBase *be = dynamic_cast<BulkElementBase *>(this->element_pt(j));
        oomph::Data *d = be->internal_data_pt(i + ft->numfields_DL);
        double derr = d->time_stepper_pt()->temporal_error_in_value(d, 0);
        res += derr * derr * ft->temporal_error_scales[i + ft->numfields_C2 + ft->numfields_C1 + ft->numfields_DL];
        denom += 1.0;
      }
    }
    //	std::cout << " RESDENOM " << res << " " << denom << std::endl;
    // TODO: Discont
    if (denom == 0)
      return 0.0;
    return res / denom;
  }

  unsigned InterfaceMesh::get_nodal_dimension()
  {
    unsigned np = Mesh::get_nodal_dimension();
    if (np)
      return np;
    if (code)
    {
      auto *ft = code->get_func_table();
      return ft->nodal_dim;
    }
    else if (bulkmesh)
    {
      np = bulkmesh->get_nodal_dimension();
    }
    return np;
  }

  int InterfaceMesh::get_element_dimension()
  {
    int np = Mesh::get_element_dimension();
    if (np >= 0)
      return np;
    else if (bulkmesh)
    {
      np = bulkmesh->get_element_dimension();
      np--;
      if (np < -1)
        np = -1;
    }
    return np;
  }

  void InterfaceMesh::fill_node_map(std::map<oomph::Node *, unsigned> &nodemap)
  {
    unsigned cnt = 0;
    for (unsigned int i = 0; i < this->nelement(); i++)
    {
      oomph::FiniteElement *FE = dynamic_cast<oomph::FiniteElement *>(this->element_pt(i));
      for (unsigned int j = 0; j < FE->nnode(); j++)
      {
        if (!nodemap.count(FE->node_pt(j)))
        {
          nodemap[FE->node_pt(j)] = cnt++;
        }
      }
    }
  }

  std::vector<oomph::Node *> InterfaceMesh::fill_reversed_node_map(bool discontinuous)
  {
    std::vector<oomph::Node *> result;
    std::set<oomph::Node *> handled;
    for (unsigned int i = 0; i < this->nelement(); i++)
    {
      oomph::FiniteElement *FE = dynamic_cast<oomph::FiniteElement *>(this->element_pt(i));
      for (unsigned int j = 0; j < FE->nnode(); j++)
      {
        if (discontinuous || (!handled.count(FE->node_pt(j))))
        {
          result.push_back(FE->node_pt(j));
          if (!discontinuous)
            handled.insert(FE->node_pt(j));
        }
      }
    }
    return result;
  }

  unsigned InterfaceMesh::count_nnode(bool discontinuous)
  {
    if (!discontinuous)
    {
      std::map<oomph::Node *, bool> counted;
      for (unsigned int i = 0; i < this->nelement(); i++)
      {
        oomph::FiniteElement *FE = dynamic_cast<oomph::FiniteElement *>(this->element_pt(i));
        for (unsigned int j = 0; j < FE->nnode(); j++)
        {
          counted[FE->node_pt(j)] = true;
        }
      }
      return counted.size();
    }
    else
    {
      unsigned res = 0;
      for (unsigned ie = 0; ie < this->nelement(); ie++)
        res += dynamic_cast<oomph::FiniteElement *>(this->element_pt(ie))->nnode();
      return res;
    }
  }

  void InterfaceMesh::nullify_selected_bulk_dofs()
  {
    throw_runtime_error("Nullified dofs are deactivated for now... Never used so far");
    /*
    if (!bulkmesh || !bulkmesh->nelement()) return;
    unsigned n_element = this->nelement();
    if (!n_element) return;
    auto * for_ci=dynamic_cast<BulkElementBase*>(bulkmesh->element_pt(0))->get_code_instance(); //Code instance to nullify the dofs
    auto * my_ci=dynamic_cast<BulkElementBase*>(this->element_pt(0))->get_code_instance(); //My code instance to nullify the dofs
    for (auto index : my_ci->nullify_bulk_residuals)
    {
      for(unsigned e=0;e<n_element;e++)
      {
       InterfaceElementBase * ielem=dynamic_cast<InterfaceElementBase*>(this->element_pt(e));
       for (unsigned int ni=0;ni<ielem->nnode();ni++)
       {
        auto * bn=dynamic_cast<BoundaryNode*>(ielem->node_pt(ni));
        if (!bn->nullified_dofs.count(for_ci)) bn->nullified_dofs[for_ci]=std::set<int>();
        bn->nullified_dofs[for_ci].insert(index);
       }
      }
    }
    */
  }

  void InterfaceMesh::clear_before_adapt()
  {
    unsigned n_element = this->nelement();
    for (unsigned e = 0; e < n_element; e++)
    {
      delete this->element_pt(e);
    }
    this->flush_element_and_node_storage();
    std::set<oomph::FiniteElement *> delete_opposite_interior_facets;
    for (unsigned i = 0; i < opposite_interior_facets.size(); i++)
    {
      if (opposite_interior_facets[i] && !delete_opposite_interior_facets.count(opposite_interior_facets[i]))
      {
        delete_opposite_interior_facets.insert(opposite_interior_facets[i]);
        delete opposite_interior_facets[i];
      }
    }
    opposite_interior_facets.clear();
    this->invalidate_lagrangian_kdtree();
  }

  void InterfaceMesh::setup_boundary_information1d(pyoomph::Mesh *parent, const std::set<unsigned> &possible_bounds)
  {
    // const unsigned n_bound = nboundary();
    oomph::MapMatrixMixed<unsigned, oomph::FiniteElement *, oomph::Vector<int> *> boundary_identifier;
    const unsigned n_element = nelement();

    /*		    std::cout << "ITNEF " << this->interfacename << std::endl;
                  std::cout << "POSS BOUNDS " ;
            for (auto boundary : possible_bounds) {std::cout << "  " << boundary ;}
            std::cout << std::endl;*/

    for (unsigned e = 0; e < n_element; e++)
    {
      oomph::FiniteElement *fe_pt = finite_element_pt(e);
      if (fe_pt->dim() == 1)
      {
        const unsigned n_node = fe_pt->nnode_1d();
        for (unsigned n = 0; n < n_node; n++)
        {
          std::set<unsigned> *boundaries_pt = 0;
          fe_pt->node_pt(n)->get_boundaries_pt(boundaries_pt);
          if (boundaries_pt != 0)
          {
            std::set<unsigned> mybounds;

            /*		    std::cout << "  ON BOUNDS " ;
                    for (auto boundary : *boundaries_pt) {std::cout << "  " << boundary ;}
                    std::cout << std::endl;*/

            std::set_intersection(boundaries_pt->begin(), boundaries_pt->end(), possible_bounds.begin(), possible_bounds.end(), std::inserter(mybounds, mybounds.begin()));

            /*		    std::cout << "  INTERSECT " ;
                    for (auto boundary : mybounds) {std::cout << "  " << boundary ;}
                    std::cout << std::endl;*/

            for (auto boundary : mybounds)
            {
              Boundary_element_pt[boundary].push_back(fe_pt);
              Face_index_at_boundary[boundary].push_back((n == 0 ? -1 : 1));
            }
          }
        }
      }
    }
  }

  void InterfaceMesh::setup_boundary_information2d(pyoomph::Mesh *parent, const std::set<unsigned> &possible_bounds)
  {
    // const unsigned n_bound = nboundary();
    oomph::MapMatrixMixed<unsigned, oomph::FiniteElement *, oomph::Vector<int> *> boundary_identifier;
    const unsigned n_element = nelement();

    /*		    std::cout << "ITNEF " << this->interfacename << std::endl;
                  std::cout << "POSS BOUNDS " ;
            for (auto boundary : possible_bounds) {std::cout << "  " << boundary ;}
            std::cout << std::endl;*/

    for (unsigned e = 0; e < n_element; e++)
    {
      BulkElementBase *fe_pt = dynamic_cast<BulkElementBase *>(finite_element_pt(e));
      if (!fe_pt)
        continue;
      if (fe_pt->dim() == 2)
      {
        const unsigned n_node = fe_pt->nnode_1d();
        if (dynamic_cast<BulkElementQuad2dC2 *>(fe_pt) || dynamic_cast<BulkElementQuad2dC1 *>(fe_pt))
        {
          std::vector<int> bound_dirs{-1, 1, -2, 2};
          for (int dir : bound_dirs)
          {
            std::set<unsigned> mybounds = possible_bounds;
            for (unsigned n = 0; n < n_node; n++)
            {
              std::set<unsigned> *boundaries_pt = 0;
              fe_pt->boundary_node_pt(dir, n)->get_boundaries_pt(boundaries_pt);
              if (boundaries_pt != 0)
              {
                std::set<unsigned> newinter;
                std::set_intersection(boundaries_pt->begin(), boundaries_pt->end(), mybounds.begin(), mybounds.end(), std::inserter(newinter, newinter.begin()));
                mybounds = newinter;
              }
              else
              {
                mybounds.clear();
                break;
              }
            }

            for (auto boundary : mybounds)
            {
              Boundary_element_pt[boundary].push_back(fe_pt);
              Face_index_at_boundary[boundary].push_back(dir);
            }
          }
        }
        else if (dynamic_cast<BulkElementTri2dC2 *>(fe_pt) || dynamic_cast<BulkElementTri2dC1 *>(fe_pt))
        {
          std::vector<int> bound_dirs{0, 1, 2};
          for (int dir : bound_dirs)
          {
            std::set<unsigned> mybounds = possible_bounds;
            for (unsigned n = 0; n < n_node; n++)
            {
              std::set<unsigned> *boundaries_pt = 0;
              fe_pt->boundary_node_pt(dir, n)->get_boundaries_pt(boundaries_pt);
              if (boundaries_pt != 0)
              {
                std::set<unsigned> newinter;
                std::set_intersection(boundaries_pt->begin(), boundaries_pt->end(), mybounds.begin(), mybounds.end(), std::inserter(newinter, newinter.begin()));
                mybounds = newinter;
              }
              else
              {
                mybounds.clear();
                break;
              }
            }

            for (auto boundary : mybounds)
            {
              Boundary_element_pt[boundary].push_back(fe_pt);
              Face_index_at_boundary[boundary].push_back(dir);
            }
          }
        }
        else
        {
          throw_runtime_error("Unknown element type found here");
        }
      }
    }
  }

  void InterfaceMesh::setup_boundary_information(pyoomph::Mesh *parent)
  {
    boundary_names = parent->get_boundary_names(); // Just make a copy of it. However, not all will be non-empty
    this->set_nboundary(boundary_names.size());
    const unsigned n_bound = nboundary();
    // Wipe/allocate storage for arrays
    Boundary_element_pt.clear();
    Face_index_at_boundary.clear();
    Boundary_element_pt.resize(n_bound);
    Face_index_at_boundary.resize(n_bound);
    // std::cout << "SETTING UP BOUNDARY INFO FOR " << interfacename << std::endl;
    // Find out the boundaries that are shared with the parent
    InterfaceMesh *imesh = this;
    // Mesh *root;
    std::set<std::string> to_rem_names;
    // Find the root mesh and mark all interface names to be removed from the boundary look-up
    while (imesh)
    {
      // root = imesh->bulkmesh;
      to_rem_names.insert(imesh->interfacename);
      imesh = dynamic_cast<InterfaceMesh *>(imesh->bulkmesh);
    }
    std::set<unsigned> to_rem_inds;
    for (auto n : to_rem_names)
    {
      for (unsigned int j = 0; j < boundary_names.size(); j++)
      {
        if (boundary_names[j] == n)
        {
          to_rem_inds.insert(j);
          break;
        }
      }
    }
    std::set<unsigned> possible_bounds;
    for (unsigned int el = 0; el < this->nelement(); el++)
    {
      oomph::FiniteElement *elem = dynamic_cast<oomph::FiniteElement *>(this->element_pt(el));
      for (unsigned int ni = 0; ni < elem->nnode(); ni++)
      {
        std::set<unsigned> *boundaries_pt = 0;
        elem->node_pt(ni)->get_boundaries_pt(boundaries_pt);
        if (boundaries_pt)
        {
          std::set<unsigned> mybounds;
          std::set_difference(boundaries_pt->begin(), boundaries_pt->end(), to_rem_inds.begin(), to_rem_inds.end(), std::inserter(mybounds, mybounds.begin()));
          for (auto bi : mybounds)
          {
            // Found (possibly) a real boundary
            possible_bounds.insert(bi);
          }
        }
      }
    }

    if (!possible_bounds.empty() && this->nelement())
    {
      /* for (auto bi :possible_bounds)
       {
        std::cout << "IN INTERFACE " << interfacename << "  WE COULD HAVE " << boundary_names[bi] << std::endl;
       }*/
      unsigned dim = dynamic_cast<oomph::FiniteElement *>(this->element_pt(0))->dim();
      if (dim == 1)
      {
        this->setup_boundary_information1d(parent, possible_bounds);
      }
      else if (dim == 0)
      {
        // Makes no sense... Or, probably it does.. when you have e.g. two contact angles and you want to add only on one side... //TODO
      }
      else if (dim == 2)
      {
        this->setup_boundary_information2d(parent, possible_bounds);
      }
      else
      {
        throw_runtime_error("Cannot do this for dimension " + std::to_string(dim) + " yet");
      }
    }

    Lookup_for_elements_next_boundary_is_setup = true;
  }

  void InterfaceMesh::rebuild_after_adapt()
  {
    if (code)
    {
      auto *ft = code->get_func_table();
      if (ft->numfields_D2TB_new || ft->numfields_D2_new || ft->numfields_D1_new || ft->numfields_DL) // || ft->numfields_D0
      {
        throw_runtime_error("Cannot adapt yet when having discontinuous fields added at an interface. Make sure to set Problem.max_refinement_level=0 and/or Problem.initial_adaption_steps=0. Will be hopefully implemented soon.");
      }
    }
    if (!bulkmesh)
    {
      std::ostringstream err_info;      
      err_info<<"Code: "<<code;
      if (code)
      {
        err_info<<", Func table: "<<code->get_func_table();
        err_info<<", Func table name: "<<code->get_func_table()->domain_name;
      }
      throw_runtime_error("bulkmesh was not set, code: "+err_info.str());
    }

    bulkmesh->generate_interface_elements(interfacename, this, code);
    // this->nullify_selected_bulk_dofs();
    this->invalidate_lagrangian_kdtree();
  }

  void InterfaceMesh::set_rebuild_information(Mesh *_bulkmesh, std::string intername, DynamicBulkElementInstance *jitcode)
  {
    bulkmesh = _bulkmesh;
    interfacename = intername;
    code = jitcode;
  }

  void InterfaceMesh::connect_interface_elements_by_kdtree(InterfaceMesh *other)
  {
    if (!this->nelement() || !other->nelement())
      return;
    std::map<std::set<int>, BulkElementBase *> nodes_to_elemB;

    unsigned ndimB = dynamic_cast<BulkElementBase *>(other->element_pt(0))->nodal_dimension();
    unsigned ndimA = dynamic_cast<BulkElementBase *>(this->element_pt(0))->nodal_dimension();
    KDTree treeB(ndimB);

    for (unsigned int ieB = 0; ieB < other->nelement(); ieB++)
    {
      BulkElementBase *eB = dynamic_cast<BulkElementBase *>(other->element_pt(ieB));
      std::set<int> indices;
      //    std::cout << "INDICES B " ;
      for (unsigned int inB = 0; inB < eB->nvertex_node(); inB++)
      {
        int ind;
        oomph::Node *nB = eB->vertex_node_pt(inB);
        if (ndimB == 3)
          ind = treeB.add_point_if_not_present(nB->x(0), nB->x(1), nB->x(2));
        else if (ndimB == 2)
          ind = treeB.add_point_if_not_present(nB->x(0), nB->x(1));
        else
          ind = treeB.add_point_if_not_present(nB->x(0));
        indices.insert(ind);
        //  std::cout << ind << "  " ;
      }
      //  std::cout << std::endl;
      nodes_to_elemB[indices] = eB;
    }

    for (unsigned int ieA = 0; ieA < this->nelement(); ieA++)
    {
      BulkElementBase *eA = dynamic_cast<BulkElementBase *>(this->element_pt(ieA));
      std::set<int> indices;
      //  std::cout << "INDICES A " ;
      for (unsigned int inA = 0; inA < eA->nvertex_node(); inA++)
      {
        int ind;
        oomph::Node *nA = eA->vertex_node_pt(inA);
        if (ndimA == 3)
          ind = treeB.point_present(nA->x(0), nA->x(1), nA->x(2));
        else if (ndimA == 2)
          ind = treeB.point_present(nA->x(0), nA->x(1));
        else
          ind = treeB.point_present(nA->x(0));
        if (ind < 0)
        {
          std::string posstring="";
          for (unsigned int inda=0;inda<ndimA;inda++) posstring+=std::to_string(nA->x(inda))+(inda+1<ndimA ? "," : "");
          throw_runtime_error("Cannot locate opposite node at x=("+posstring+")");
        }
        indices.insert(ind);
        //  std::cout << ind << "  " ;
      }
      //  std::cout << std::endl;
      if (!nodes_to_elemB.count(indices))
      {
        throw_runtime_error("Cannot locate opposite element");
      }
      BulkElementBase *eB = nodes_to_elemB[indices];

      InterfaceElementBase *iA = dynamic_cast<InterfaceElementBase *>(eA);
      InterfaceElementBase *iB = dynamic_cast<InterfaceElementBase *>(eB);
      iA->set_opposite_interface_element(iB,this->opposite_offset_vector);
      iB->set_opposite_interface_element(iA,this->reversed_opposite_offset_vector);
    }
  }

  void InterfaceMesh::set_opposite_interface_offset_vector(const std::vector<double> & offset)
  {
    this->opposite_offset_vector=offset;
    this->reversed_opposite_offset_vector=offset;
    for (unsigned int i=0;i<this->reversed_opposite_offset_vector.size();i++) this->reversed_opposite_offset_vector[i]=-this->reversed_opposite_offset_vector[i];
  }

  ///////
  /*
  BulkNodeIterator::iterator::iterator(Mesh *m) : msh(m), pos(0), access(NodeAccess(m->get_problem())) { access.current_node=dynamic_cast<pyoomph::NodeWithFieldIndicesBase*>(msh->node_pt(pos));}
  BulkNodeIterator::iterator::iterator(Mesh *m,unsigned p) :msh(m), pos(p), access(NodeAccess(m->get_problem())) { access.current_node=dynamic_cast<pyoomph::NodeWithFieldIndicesBase*>(msh->node_pt(pos));}
  BulkNodeIterator::iterator::iterator(Mesh *m,unsigned p,bool Only_For_End) :msh(m), pos(p), access(NodeAccess(m->get_problem())) { access.current_node=NULL;}
  BulkNodeIterator::iterator & BulkNodeIterator::iterator::operator++() { access.current_node=dynamic_cast<pyoomph::NodeWithFieldIndicesBase*>(msh->node_pt(++pos)); return *this; }
  BulkNodeIterator::iterator BulkNodeIterator::begin() { return {mesh}; }
  BulkNodeIterator::iterator BulkNodeIterator::end() { return {mesh,mesh->nnode(),true}; }
  */

  void DynamicTree::dynamic_split_if_required()
  {
    if (Object_pt->to_be_refined())
    {
      oomph::Vector<BulkElementBase *> new_elements_pt;
      auto *beb = dynamic_cast<BulkElementBase *>(Object_pt);
      beb->dynamic_split(new_elements_pt);
      unsigned n_sons = new_elements_pt.size();
      Son_pt.resize(n_sons);
      DynamicTree *father_pt = this;
      for (unsigned i_son = 0; i_son < n_sons; i_son++)
      {
        Son_pt[i_son] = construct_son(new_elements_pt[i_son], father_pt, i_son);
        Son_pt[i_son]->object_pt()->initial_setup();
        // std::cout << "CONSTRUCTED SONS FATHER " << dynamic_cast<BulkElementBase*>(Son_pt[i_son]->object_pt())->father_element_pt() << "   " << father_pt << std::endl;
        // dynamic_cast<BulkElementBase*>(Son_pt[i_son]->object_pt())->father_element_pt()=father_pt;
      }
    }
  }

  ////////////////

  unsigned TemplatedMeshBase::add_new_element(pyoomph::BulkElementBase *new_el, std::vector<pyoomph::Node *> nodes)
  {
    unsigned res = Element_pt.size();
    Element_pt.push_back(new_el);
    for (unsigned int i = 0; i < new_el->nnode(); i++)
    {
      new_el->node_pt(i) = nodes[i];
    }

    for (unsigned int i = 0; i < new_el->ninternal_data(); i++)
    {
      new_el->internal_data_pt(i)->set_time_stepper(nodes[0]->time_stepper_pt(), false);
    }
    new_el->initial_cartesian_nondim_size = new_el->size();
    new_el->initial_quality_factor = new_el->get_quality_factor();

    if (BulkElementBase::__CurrentCodeInstance->get_func_table()->integration_order)
    {
      new_el->set_integration_order(BulkElementBase::__CurrentCodeInstance->get_func_table()->integration_order);
    }
    return res;
  }

#ifdef OOMPH_HAS_MPI

  void TemplatedMeshBase::additional_synchronise_hanging_nodes(const unsigned &ncont_interpolated_values)
  {
    // Check if additional synchronisation of hanging nodes is disabled
    if (is_additional_synchronisation_of_hanging_nodes_disabled() == true)
    {
      return;
    }

    // This provides all the node-adding helper functions required to reconstruct
    // the missing halo master nodes on this processor
    using namespace Missing_masters_functions;

    double t_start = 0.0;
    double t_end = 0.0;
    if (oomph::Global_timings::Doc_comprehensive_timings)
    {
      t_start = oomph::TimingHelpers::timer();
    }

    // Store number of processors and current process
    MPI_Status status;
    int n_proc = Comm_pt->nproc();
    int my_rank = Comm_pt->my_rank();

#ifdef PARANOID
    // Paranoid check to make sure nothing else is using the
    // external storage. This will need to be changed at some
    // point if we are to use non-uniformly spaced nodes in
    // multi-domain problems.
    bool err = false;
    // Print out external storage
    for (int d = 0; d < n_proc; d++)
    {
      if (d != my_rank)
      {
        // Check to see if external storage is being used by anybody else
        if (nexternal_haloed_node(d) != 0)
        {
          err = true;
          oomph::oomph_info << "Processor " << my_rank << "'s external haloed nodes with processor " << d << " are:" << std::endl;
          for (unsigned i = 0; i < nexternal_haloed_node(d); i++)
          {
            oomph::oomph_info << "external_haloed_node_pt(" << d << "," << i << ") = " << external_haloed_node_pt(d, i) << std::endl;
            oomph::oomph_info << "x = ( " << external_haloed_node_pt(d, i)->x(0) << " , " << external_haloed_node_pt(d, i)->x(1) << " )" << std::endl;
          }
        }
      }
    }
    for (int d = 0; d < n_proc; d++)
    {
      if (d != my_rank)
      {
        // Check to see if external storage is being used by anybody else
        if (nexternal_halo_node(d) != 0)
        {
          err = true;
          oomph::oomph_info << "Processor " << my_rank << "'s external halo nodes with processor " << d << " are:" << std::endl;
          for (unsigned i = 0; i < nexternal_halo_node(d); i++)
          {
            oomph::oomph_info << "external_halo_node_pt(" << d << "," << i << ") = " << external_halo_node_pt(d, i) << std::endl;
            oomph::oomph_info << "x = ( " << external_halo_node_pt(d, i)->x(0) << " , " << external_halo_node_pt(d, i)->x(1) << " )" << std::endl;
          }
        }
      }
    }
    if (err)
    {
      std::ostringstream err_stream;
      err_stream << "There are already some nodes in the external storage"
                 << std::endl
                 << "for this mesh. This bit assumes that nothing else"
                 << std::endl
                 << "uses this storage (for now).";
      throw OomphLibError(
          err_stream.str(),
          OOMPH_CURRENT_FUNCTION,
          OOMPH_EXCEPTION_LOCATION);
    }
#endif

    // Compare the halo and haloed nodes for discrepancies in hanging status

    // Storage for the hanging status of halo/haloed nodes on elements
    oomph::Vector<oomph::Vector<int>> haloed_hanging(n_proc);
    oomph::Vector<oomph::Vector<int>> halo_hanging(n_proc);

    // Storage for the haloed nodes with discrepancies in their hanging status
    // with each processor
    oomph::Vector<std::map<oomph::Node *, unsigned>>
        haloed_hanging_node_with_discrepancy_pt(n_proc);

    if (oomph::Global_timings::Doc_comprehensive_timings)
    {
      t_start = oomph::TimingHelpers::timer();
    }

    // Store number of continuosly interpolated values as int
    int ncont_inter_values = ncont_interpolated_values;

    // Loop over processes: Each processor checks that is haloed nodes
    // with proc d have consistent hanging stats with halo counterparts.
    for (int d = 0; d < n_proc; d++)
    {

      // No halo with self: Setup hang info for my haloed nodes with proc d
      // then get ready to receive halo info from processor d.
      if (d != my_rank)
      {

        // Loop over haloed nodes
        unsigned nh = nhaloed_node(d);
        for (unsigned j = 0; j < nh; j++)
        {
          // Get node
          oomph::Node *nod_pt = haloed_node_pt(d, j);

          // Loop over the hanging status for each interpolated variable
          // (and the geometry)
          for (int icont = -1; icont < ncont_inter_values; icont++)
          {
            // Store the hanging status of this haloed node
            if (nod_pt->is_hanging(icont))
            {
              unsigned n_master = nod_pt->hanging_pt(icont)->nmaster();
              haloed_hanging[d].push_back(n_master);
            }
            else
            {
              haloed_hanging[d].push_back(0);
            }
          }
        }

        // Receive the hanging status information from the corresponding process
        unsigned count_haloed = haloed_hanging[d].size();

#ifdef PARANOID
        // Check that number of halo and haloed data match
        unsigned tmp = 0;
        MPI_Recv(&tmp, 1, MPI_UNSIGNED, d, 0, Comm_pt->mpi_comm(), &status);
        if (tmp != count_haloed)
        {
          std::ostringstream error_stream;
          error_stream << "Number of halo data, " << tmp
                       << ", does not match number of haloed data, "
                       << count_haloed << std::endl;
          throw oomph::OomphLibError(
              error_stream.str(),
              OOMPH_CURRENT_FUNCTION,
              OOMPH_EXCEPTION_LOCATION);
        }
#endif

        // Get the data (if any)
        if (count_haloed != 0)
        {
          halo_hanging[d].resize(count_haloed);
          MPI_Recv(&halo_hanging[d][0], count_haloed, MPI_INT, d, 0,
                   Comm_pt->mpi_comm(), &status);
        }
      }
      else // d==my_rank, i.e. current process: Send halo hanging status
           // to process dd where it's received (see above) and compared
           // and compared against the hang status of the haloed nodes
      {
        for (int dd = 0; dd < n_proc; dd++)
        {
          // No halo with yourself
          if (dd != d)
          {

            // Storage for halo hanging status and counter
            oomph::Vector<int> local_halo_hanging;

            // Loop over halo nodes
            unsigned nh = nhalo_node(dd);
            for (unsigned j = 0; j < nh; j++)
            {
              // Get node
              oomph::Node *nod_pt = halo_node_pt(dd, j);

              // Loop over the hanging status for each interpolated variable
              // (and the geometry)
              for (int icont = -1; icont < ncont_inter_values; icont++)
              {
                // Store hanging status of halo node
                if (nod_pt->is_hanging(icont))
                {
                  unsigned n_master = nod_pt->hanging_pt(icont)->nmaster();
                  local_halo_hanging.push_back(n_master);
                }
                else
                {
                  local_halo_hanging.push_back(0);
                }
              }
            }

            // Send the information to the relevant process
            unsigned count_halo = local_halo_hanging.size();

#ifdef PARANOID
            // Check that number of halo and haloed data match
            MPI_Send(&count_halo, 1, MPI_UNSIGNED, dd, 0, Comm_pt->mpi_comm());
#endif

            // Send data (if any)
            if (count_halo != 0)
            {
              MPI_Send(&local_halo_hanging[0], count_halo, MPI_INT,
                       dd, 0, Comm_pt->mpi_comm());
            }
          }
        }
      }
    }

    if (oomph::Global_timings::Doc_comprehensive_timings)
    {
      t_end = oomph::TimingHelpers::timer();
      oomph::oomph_info << "Time for first all-to-all in additional_synchronise_hanging_nodes(): "
                        << t_end - t_start << std::endl;
      t_start = oomph::TimingHelpers::timer();
    }

    // Now compare equivalent halo and haloed vectors to find discrepancies.
    // It is possible that a master node may not be on either process involved
    // in the halo-haloed scheme; to work round this, we use the shared_node
    // storage scheme, which stores all nodes that are on each pair of processors
    // in the same order on each of the two processors

    // Loop over domains: Each processor checks consistency of hang status
    // of its haloed nodes with proc d against the halo counterpart. Haloed
    // wins if there are any discrepancies.
    for (int d = 0; d < n_proc; d++)
    {
      // No halo with yourself
      if (d != my_rank)
      {
        // Counter for traversing haloed data
        unsigned count = 0;

        // Loop over haloed nodes
        unsigned nh = nhaloed_node(d);
        for (unsigned j = 0; j < nh; j++)
        {
          // Get node
          oomph::Node *nod_pt = haloed_node_pt(d, j);

          // Loop over the hanging status for each interpolated variable
          // (and the geometry)
          for (int icont = -1; icont < ncont_inter_values; icont++)
          {
            // Compare hanging status of halo/haloed counterpart structure

            // Haloed is is hanging and haloed has different number
            // of master nodes (which includes none in which case it isn't
            // hanging)
            if ((haloed_hanging[d][count] > 0) &&
                (haloed_hanging[d][count] != halo_hanging[d][count]))
            {
              // Store this node so it can be synchronised later
              haloed_hanging_node_with_discrepancy_pt[d].insert(
                  std::pair<oomph::Node *, unsigned>(nod_pt, d));
            }
            // Increment counter for number of haloed data
            count++;
          } // end of loop over icont
        } // end of loop over haloed nodes
      }
    } // end loop over all processors

    // Populate external halo(ed) node storage with master nodes of halo(ed)
    // nodes

    // Loop over domains: Each processor checks consistency of hang status
    // of its haloed nodes with proc d against the halo counterpart. Haloed
    // wins if there are any discrepancies.
    for (int d = 0; d < n_proc; d++)
    {
      // No halo with yourself
      if (d != my_rank)
      {
        // Now add haloed master nodes to external storage
        //===============================================

        // Storage for data to be sent
        oomph::Vector<unsigned> send_unsigneds(0);
        oomph::Vector<double> send_doubles(0);

        // Count number of haloed nonmaster nodes for halo process
        unsigned nhaloed_nonmaster_nodes_processed = 0;
        oomph::Vector<unsigned> haloed_nonmaster_node_index(0);

        // Loop over hanging halo nodes with discrepancies
        std::map<oomph::Node *, unsigned>::iterator j;
        for (j = haloed_hanging_node_with_discrepancy_pt[d].begin(); j != haloed_hanging_node_with_discrepancy_pt[d].end(); j++)
        {
          oomph::Node *nod_pt = (*j).first;
          // Find index of this haloed node in the halo storage of processor d
          //(But find in shared node storage in case it is actually haloed on
          // another processor which we don't know about)
          std::vector<oomph::Node *>::iterator it = std::find(Shared_node_pt[d].begin(),
                                                              Shared_node_pt[d].end(),
                                                              nod_pt);
          if (it != Shared_node_pt[d].end())
          {
            // Tell other processor to create this node
            // send_unsigneds.push_back(1);
            nhaloed_nonmaster_nodes_processed++;

            // Tell the other processor where to find this node in its halo node
            // storage
            unsigned index = it - Shared_node_pt[d].begin();
            haloed_nonmaster_node_index.push_back(index);

            // Tell this processor that this node is really a haloed node
            // This also packages up the data which needs to be sent to the
            // processor on which the halo equivalent node lives
            recursively_add_masters_of_external_haloed_node(d, nod_pt, this, ncont_inter_values,
                                                            send_unsigneds, send_doubles);
          }
          else
          {
            throw oomph::OomphLibError(
                "Haloed node not found in haloed node storage",
                OOMPH_CURRENT_FUNCTION,
                OOMPH_EXCEPTION_LOCATION);
          }
        }

        // How much data needs to be sent?
        unsigned send_unsigneds_count = send_unsigneds.size();
        unsigned send_doubles_count = send_doubles.size();

        // Send ammount of data
        MPI_Send(&send_unsigneds_count, 1, MPI_UNSIGNED, d, 0, Comm_pt->mpi_comm());
        MPI_Send(&send_doubles_count, 1, MPI_UNSIGNED, d, 1, Comm_pt->mpi_comm());

        // Send to halo process the number of haloed nodes we processed
        MPI_Send(&nhaloed_nonmaster_nodes_processed, 1, MPI_UNSIGNED, d, 2,
                 Comm_pt->mpi_comm());
        if (nhaloed_nonmaster_nodes_processed > 0)
        {
          MPI_Send(&haloed_nonmaster_node_index[0],
                   nhaloed_nonmaster_nodes_processed, MPI_UNSIGNED, d, 3,
                   Comm_pt->mpi_comm());
        }

        // Send data about external halo nodes
        if (send_unsigneds_count > 0)
        {
          // Only send if there is anything to send
          MPI_Send(&send_unsigneds[0], send_unsigneds_count, MPI_UNSIGNED, d, 4,
                   Comm_pt->mpi_comm());
        }
        if (send_doubles_count > 0)
        {
          // Only send if there is anything to send
          MPI_Send(&send_doubles[0], send_doubles_count, MPI_DOUBLE, d, 5,
                   Comm_pt->mpi_comm());
        }
      }
      else // (d==my_rank), current process
      {
        // Now construct and add halo versions of master nodes to external storage
        //=======================================================================

        // Loop over processors to get data
        for (int dd = 0; dd < n_proc; dd++)
        {
          // Don't talk to yourself
          if (dd != d)
          {
            // How much data to be received
            unsigned nrecv_unsigneds = 0;
            unsigned nrecv_doubles = 0;
            MPI_Recv(&nrecv_unsigneds, 1, MPI_UNSIGNED, dd, 0,
                     Comm_pt->mpi_comm(), &status);
            MPI_Recv(&nrecv_doubles, 1, MPI_UNSIGNED, dd, 1,
                     Comm_pt->mpi_comm(), &status);

            // Get from haloed process the number of halo nodes we need to process
            unsigned nhalo_nonmaster_nodes_to_process = 0;
            MPI_Recv(&nhalo_nonmaster_nodes_to_process, 1, MPI_UNSIGNED, dd, 2,
                     Comm_pt->mpi_comm(), &status);
            oomph::Vector<unsigned> halo_nonmaster_node_index(
                nhalo_nonmaster_nodes_to_process);
            if (nhalo_nonmaster_nodes_to_process != 0)
            {
              MPI_Recv(&halo_nonmaster_node_index[0],
                       nhalo_nonmaster_nodes_to_process, MPI_UNSIGNED, dd, 3,
                       Comm_pt->mpi_comm(), &status);
            }

            // Storage for data to be received
            oomph::Vector<unsigned> recv_unsigneds(nrecv_unsigneds);
            oomph::Vector<double> recv_doubles(nrecv_doubles);

            // Receive data about external haloed equivalent nodes
            if (nrecv_unsigneds > 0)
            {
              // Only send if there is anything to send
              MPI_Recv(&recv_unsigneds[0], nrecv_unsigneds, MPI_UNSIGNED, dd, 4,
                       Comm_pt->mpi_comm(), &status);
            }
            if (nrecv_doubles > 0)
            {
              // Only send if there is anything to send
              MPI_Recv(&recv_doubles[0], nrecv_doubles, MPI_DOUBLE, dd, 5,
                       Comm_pt->mpi_comm(), &status);
            }

            // Counters for flat packed data counters
            unsigned recv_unsigneds_count = 0;
            unsigned recv_doubles_count = 0;

            // Loop over halo nodes with discrepancies in their hanging status
            for (unsigned j = 0; j < nhalo_nonmaster_nodes_to_process; j++)
            {
              // Get pointer to halo nonmaster node which needs processing
              //(But given index is its index in the shared storage)
              oomph::Node *nod_pt = shared_node_pt(dd, halo_nonmaster_node_index[j]);

#ifdef PARANOID
              // Check if we have a MacroElementNodeUpdateNode
              if (dynamic_cast<oomph::MacroElementNodeUpdateNode *>(nod_pt))
              {
                // BENFLAG: The construction of missing master nodes for
                //          MacroElementNodeUpdateNodes does not work as expected.
                //          They require MacroElementNodeUpdateElements to be
                //          created for the missing halo nodes which will be
                //          added. It behaves as expected until duplicate nodes
                //          are pruned at the problem level.
                std::ostringstream err_stream;
                err_stream << "This currently doesn't work for"
                           << std::endl
                           << "MacroElementNodeUpdateNodes because these require"
                           << std::endl
                           << "MacroElementNodeUpdateElements to be created for"
                           << std::endl
                           << "the missing halo nodes which will be added"
                           << std::endl;
                throw oomph::OomphLibError(err_stream.str(),
                                           OOMPH_CURRENT_FUNCTION,
                                           OOMPH_EXCEPTION_LOCATION);
                // OomphLibWarning(err_stream.str(),
                //                 OOMPH_CURRENT_FUNCTION,
                //                 OOMPH_EXCEPTION_LOCATION);
              }
#endif

              // Construct copy of node and add to external halo node storage.
              unsigned loc_p = (unsigned)dd;
              unsigned node_index;
              recursively_add_masters_of_external_halo_node_to_storage<BulkElementBase>(nod_pt, this, loc_p, node_index, ncont_inter_values,
                                                                                        recv_unsigneds_count, recv_unsigneds,
                                                                                        recv_doubles_count, recv_doubles);
            }

          } // end of dd!=d
        } // end of second loop over all processors
      }
    } // end loop over all processors

    if (oomph::Global_timings::Doc_comprehensive_timings)
    {
      t_end = oomph::TimingHelpers::timer();
      oomph::oomph_info
          << "Time for second all-to-all in additional_synchronise_hanging_nodes() "
          << t_end - t_start << std::endl;
      t_start = oomph::TimingHelpers::timer();
    }

    // Populate external halo(ed) node storage with master nodes of halo(ed)
    // nodes [end]

    // Count how many external halo/haloed nodes are added
    unsigned external_halo_count = 0;
    unsigned external_haloed_count = 0;

    // Flag to test whether we attampt to add any duplicate haloed nodes to the
    // shared storage -- if this is the case then we have duplicate halo nodes
    // on another processor but with different pointers and the shared scheme
    // will not be set up correctly
    bool duplicate_haloed_node_exists = false;

    // Loop over all the processors and add the shared nodes
    for (int d = 0; d < n_proc; d++)
    {

      // map of bools for whether the (external) node has been shared,
      // initialised to 0 (false) for each domain d
      std::map<oomph::Node *, bool> node_shared;

      // For all domains lower than the current domain: Do halos first
      // then haloed, to ensure correct order in lookup scheme from
      // the other side
      if (d < my_rank)
      {
        // Do external halo nodes
        unsigned nexternal_halo_nod = nexternal_halo_node(d);
        for (unsigned j = 0; j < nexternal_halo_nod; j++)
        {
          oomph::Node *nod_pt = external_halo_node_pt(d, j);

          // Add it as a shared node from current domain
          if (!node_shared[nod_pt])
          {
            this->add_shared_node_pt(d, nod_pt);
            node_shared[nod_pt] = true;
            external_halo_count++;
          }

        } // end loop over nodes

        // Do external haloed nodes
        unsigned nexternal_haloed_nod = nexternal_haloed_node(d);
        for (unsigned j = 0; j < nexternal_haloed_nod; j++)
        {
          oomph::Node *nod_pt = external_haloed_node_pt(d, j);

          // Add it as a shared node from current domain
          if (!node_shared[nod_pt])
          {
            this->add_shared_node_pt(d, nod_pt);
            node_shared[nod_pt] = true;
            external_haloed_count++;
          }
          else
          {
            duplicate_haloed_node_exists = true;
          }

        } // end loop over nodes
      }

      // If the domain is bigger than the current rank: Do haloed first
      // then halo, to ensure correct order in lookup scheme from
      // the other side
      if (d > my_rank)
      {
        // Do external haloed nodes
        unsigned nexternal_haloed_nod = nexternal_haloed_node(d);
        for (unsigned j = 0; j < nexternal_haloed_nod; j++)
        {
          oomph::Node *nod_pt = external_haloed_node_pt(d, j);

          // Add it as a shared node from current domain
          if (!node_shared[nod_pt])
          {
            this->add_shared_node_pt(d, nod_pt);
            node_shared[nod_pt] = true;
            external_haloed_count++;
          }
          else
          {
            duplicate_haloed_node_exists = true;
          }

        } // end loop over nodes

        // Do external halo nodes
        unsigned nexternal_halo_nod = nexternal_halo_node(d);
        for (unsigned j = 0; j < nexternal_halo_nod; j++)
        {
          oomph::Node *nod_pt = external_halo_node_pt(d, j);

          // Add it as a shared node from current domain
          if (!node_shared[nod_pt])
          {
            this->add_shared_node_pt(d, nod_pt);
            node_shared[nod_pt] = true;
            external_halo_count++;
          }

        } // end loop over nodes

      } // end if (d ...)

    } // end loop over processes

    // Say how many external halo/haloed nodes were added
    oomph::oomph_info << "INFO: " << external_halo_count
                      << " external halo nodes and"
                      << std::endl;
    oomph::oomph_info << "INFO: " << external_haloed_count
                      << " external haloed nodes were added to the shared node scheme"
                      << std::endl;

    // If we added duplicate haloed nodes, throw an error
    if (duplicate_haloed_node_exists)
    {
      // This problem should now be avoided because we are using existing
      // communication methods to locate nodes in this case. The error used
      // to arise as follows:
      //// Let my_rank==A. If this has happened then it means that
      //// duplicate haloed nodes exist on another processor (B). This
      //// problem arises if a master of a haloed node with a discrepancy
      //// is haloed with a different processor (C). A copy is constructed
      //// in the external halo storage on processor (B) because that node
      //// is not found in the (internal) haloed storage on (A) with (B)
      //// but that node already exists on processor (B) in the (internal)
      //// halo storage with processor (C). Thus two copies of this master
      //// node now exist on processor (B).

      std::ostringstream err_stream;
      err_stream << "Duplicate halo nodes exist on another processor!"
                 << std::endl
                 << "(See source code for more detailed explanation)"
                 << std::endl;

      throw oomph::OomphLibError(
          err_stream.str(),
          OOMPH_CURRENT_FUNCTION,
          OOMPH_EXCEPTION_LOCATION);
    }

    if (oomph::Global_timings::Doc_comprehensive_timings)
    {
      t_end = oomph::TimingHelpers::timer();
      oomph::oomph_info
          << "Time for identification of shared nodes in additional_synchronise_hanging_nodes(): "
          << t_end - t_start << std::endl;
    }
  }

#endif

  ////////////////

  MeshKDTree::MeshKDTree(pyoomph::Mesh *mesh, bool use_lagrangian, unsigned time_index) : lagrangian(use_lagrangian), tindex(time_index), tree(NULL)
  {
    std::map<pyoomph::Node *, unsigned> nodeinds;
    BulkElementBase::zeta_time_history = time_index;
    BulkElementBase::zeta_coordinate_type = (use_lagrangian ? 0 : 1);
    max_search_radius = 0.0;
    unsigned int dim = 0;
    std::vector<double> coords;
    for (unsigned int ie = 0; ie < mesh->nelement(); ie++)
    {
      auto *elpt = dynamic_cast<BulkElementBase *>(mesh->element_pt(ie));
      for (unsigned int in = 0; in < elpt->nnode(); in++)
      {
        pyoomph::Node *n = dynamic_cast<pyoomph::Node *>(elpt->node_pt(in));
        unsigned int index;
        if (nodeinds.count(n))
        {
          index = nodeinds[n];
        }
        else
        {
          index = nodes_by_index.size();
          nodeinds[n] = index;
          nodes_by_index.push_back(n);
          if (!dim)
            dim = (use_lagrangian ? n->nlagrangian() : n->ndim());
          for (unsigned int d = 0; d < dim; d++)
            coords.push_back(elpt->zeta_nodal(in, 0, d));
        }
        if (!nodes_to_elem.count(n))
          nodes_to_elem[n] = std::set<pyoomph::BulkElementBase *>();
        nodes_to_elem[n].insert(elpt);

        for (unsigned int jn = in + 1; jn < elpt->nnode(); jn++)
        {
          double nndist = 0.0;
          for (unsigned int d = 0; d < dim; d++)
          {
            double zdist = elpt->zeta_nodal(in, 0, d) - elpt->zeta_nodal(jn, 0, d);
            nndist += zdist * zdist;
          }
          if (nndist > max_search_radius)
            max_search_radius = nndist;
        }
      }
    }
    max_search_radius = sqrt(max_search_radius) * 10;
    BulkElementBase::zeta_time_history = 0;
    BulkElementBase::zeta_coordinate_type = 0;
    tree = new KDTree(coords, dim);
  }

  pyoomph::Node *MeshKDTree::find_node(const oomph::Vector<double> &coord, double *distreturn)
  {
    int index = -1;
    if (coord.size() == 1)
      index = tree->nearest_point(coord[0], 0.0, 0.0, distreturn);
    else if (coord.size() == 2)
      index = tree->nearest_point(coord[0], coord[1], 0.0, distreturn);
    else if (coord.size() == 3)
      index = tree->nearest_point(coord[0], coord[1], coord[2], distreturn);

    if (index < 0)
      return NULL;

    return nodes_by_index[index];
  }

  pyoomph::BulkElementBase *MeshKDTree::find_element(oomph::Vector<double> zeta, oomph::Vector<double> &sreturn)
  {
    // std::cout << "ENTERINF FIND ELEMENT " << std::endl;
    BulkElementBase::zeta_time_history = tindex;
    BulkElementBase::zeta_coordinate_type = (lagrangian ? 0 : 1);

    std::set<BulkElementBase *> processed_elems;
    std::set<pyoomph::Node *> processed_nodes;

    sreturn.resize(zeta.size(), 0.0);

    // First, process the nearest node and all attached elements
    pyoomph::Node *n = this->find_node(zeta);
    if (!n)
    {
      BulkElementBase::zeta_time_history = 0;
      BulkElementBase::zeta_coordinate_type = 0;
      return NULL;
    }

    oomph::GeomObject *ret = NULL;
    for (auto &e : nodes_to_elem[n])
    {
      //  std::cout << " TRY TO FIND ZETA " << zeta[0] << "  " << zeta[1] << " BASED ON NODE POS " << n->x(0) << "  " <<  n->x(1) << std::endl;
      oomph::Vector<double> zeta_in = zeta; // Why ever... But without I had issues
                                            //  std::cout << " BEF ZETA " << zeta[0] << "  " << zeta[1] << "  IN "  << zeta_in[0] << "  " << zeta_in[1] << std::endl;
      e->locate_zeta(zeta_in, ret, sreturn);
      //  std::cout << " AFTER ZETA " << zeta[0] << "  " << zeta[1] << "   IN "  << zeta_in[0] << "  " << zeta_in[1] << "  " << ret << std::endl;
      if (ret)
      {
        BulkElementBase::zeta_time_history = 0;
        BulkElementBase::zeta_coordinate_type = 0;
        return e;
      }
      processed_elems.insert(e);
    }
    processed_nodes.insert(n);

    // Then, go by increasing radius
    double rad = max_search_radius;
    double x = zeta[0], y = 0.0, z = 0.0;
    if (zeta.size() >= 2)
    {
      y = zeta[1];
      if (zeta.size() >= 3)
      {
        z = zeta[2];
      }
    }
    std::vector<std::pair<uint32_t, double>> search_res = tree->radius_search(rad, x, y, z);

    for (auto &sr : search_res)
    {
      //     std::cout << "ITERATING OVER SERACH RES " << sr.first << "  " << sr.second << std::endl;
      pyoomph::Node *n = nodes_by_index[sr.first];
      //     std::cout << " TRY TO FIND ZETA " << zeta[0] << "  " << zeta[1] << " BASED ON NODE POS " << n->x(0) << "  " <<  n->x(1) << std::endl;
      if (processed_nodes.count(n))
        continue;
      for (auto &e : nodes_to_elem[n])
      {
        if (processed_elems.count(e))
          continue;
        oomph::Vector<double> zeta_in = zeta; // Why ever... But without I had issues
        e->locate_zeta(zeta_in, ret, sreturn);
        if (ret)
        {
          BulkElementBase::zeta_time_history = 0;
          BulkElementBase::zeta_coordinate_type = 0;
          return e;
        }
        processed_elems.insert(e);
      }
      processed_nodes.insert(n);
    }

    BulkElementBase::zeta_time_history = 0;
    BulkElementBase::zeta_coordinate_type = 0;
    return NULL;
  }

}
