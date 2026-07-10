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
#include "mesh2d.hpp"

#include "Telements.h"
#include "unstructured_two_d_mesh_geometry_base.h"
#include <array>
namespace pyoomph
{

  DynamicQuadTreeForest::DynamicQuadTreeForest(oomph::Vector<oomph::TreeRoot *> &trees_pt) : oomph::QuadTreeForest(trees_pt, true)
  {
#ifdef LEAK_CHECK
    LeakCheckNames::QuadTreeForest_build += 1;
#endif

    // Don't setup neighbours etc. if forest is empty
    if (trees_pt.size() == 0)
    {
      return;
    }

    using namespace oomph::QuadTreeNames;

    // Setup the neighbours
    find_neighbours();

    // Construct the rotation scheme, note that all neighbour pointers must
    // be set before the constructor is called
    construct_north_equivalents();
  }

  void DynamicQuadTreeForest::find_neighbours()
  {
    using namespace oomph::QuadTreeNames;

    unsigned numtrees = ntree();
    unsigned n = 0; // to store nnode1d
    if (numtrees > 0)
    {
      n = Trees_pt[0]->object_pt()->nnode_1d();
    }
    else
    {
      throw oomph::OomphLibError(
          "Trying to setup the neighbour scheme for an empty forest\n",
          OOMPH_CURRENT_FUNCTION,
          OOMPH_EXCEPTION_LOCATION);
    }

    // Find potentially connected trees by identifying
    // those whose associated elements share a common vertex node
    std::map<oomph::Node *, std::set<unsigned>> tree_assoc_with_vertex_node;

    // Loop over all trees
    for (unsigned i = 0; i < numtrees; i++)
    {
      if (dynamic_cast<oomph::QuadElementBase *>(Trees_pt[i]->object_pt()))
      {
        for (unsigned j = 0; j < 4; j++)
        {
          oomph::Node *nod_pt = dynamic_cast<oomph::QuadElementBase *>(Trees_pt[i]->object_pt())->vertex_node_pt(j);
          tree_assoc_with_vertex_node[nod_pt].insert(i);
        }
      }
      else if (dynamic_cast<oomph::TElementBase *>(Trees_pt[i]->object_pt()))
      {
        for (unsigned j = 0; j < 3; j++)
        {
          oomph::Node *nod_pt = dynamic_cast<oomph::TElementBase *>(Trees_pt[i]->object_pt())->vertex_node_pt(j);
          tree_assoc_with_vertex_node[nod_pt].insert(i);
        }
      }
      else
      {
        throw_runtime_error("Strange element in tree forest");
      }
    }

    // For each tree we store a set of potentially neighbouring trees
    // i.e. trees that share at least one node
    oomph::Vector<std::set<unsigned>> potentially_neighb_tree(numtrees);

    // Loop over vertex nodes
    for (std::map<oomph::Node *, std::set<unsigned>>::iterator it = tree_assoc_with_vertex_node.begin(); it != tree_assoc_with_vertex_node.end(); it++)
    {
      // Loop over connected elements twice
      for (std::set<unsigned>::iterator it_el1 = it->second.begin(); it_el1 != it->second.end(); it_el1++)
      {
        unsigned i = (*it_el1);
        for (std::set<unsigned>::iterator it_el2 = it->second.begin(); it_el2 != it->second.end(); it_el2++)
        {
          unsigned j = (*it_el2);
          // These two elements are potentially connected
          if (i != j)
          {
            potentially_neighb_tree[i].insert(j);
          }
        }
      }
    }

    // Loop over all trees
    for (unsigned i = 0; i < numtrees; i++)
    {
      // Loop over their potential neighbours
      for (std::set<unsigned>::iterator it = potentially_neighb_tree[i].begin(); it != potentially_neighb_tree[i].end(); it++)
      {
        unsigned j = (*it);
        if (dynamic_cast<oomph::QuadElementBase *>(Trees_pt[i]->object_pt()))
        {
          if (dynamic_cast<oomph::QuadElementBase *>(Trees_pt[j]->object_pt()))
          {
            // is it the Northern neighbour ?
            bool is_N_neighbour =
                ((Trees_pt[j]->object_pt()->get_node_number(
                      Trees_pt[i]->object_pt()->node_pt(n * (n - 1))) != -1) &&
                 (Trees_pt[j]->object_pt()->get_node_number(
                      Trees_pt[i]->object_pt()->node_pt(n * n - 1)) != -1));

            // is it the Southern neighbour ?
            bool is_S_neighbour =
                ((Trees_pt[j]->object_pt()->get_node_number(
                      Trees_pt[i]->object_pt()->node_pt(0)) != -1) &&
                 (Trees_pt[j]->object_pt()->get_node_number(
                      Trees_pt[i]->object_pt()->node_pt(n - 1)) != -1));

            // is it the Eastern neighbour ?
            bool is_E_neighbour =
                ((Trees_pt[j]->object_pt()->get_node_number(
                      Trees_pt[i]->object_pt()->node_pt(n - 1)) != -1) &&
                 (Trees_pt[j]->object_pt()->get_node_number(
                      Trees_pt[i]->object_pt()->node_pt(n * n - 1)) != -1));

            // is it the Western neighbour ?
            bool is_W_neighbour =
                ((Trees_pt[j]->object_pt()->get_node_number(
                      Trees_pt[i]->object_pt()->node_pt(0)) != -1) &&
                 (Trees_pt[j]->object_pt()->get_node_number(
                      Trees_pt[i]->object_pt()->node_pt(n * (n - 1))) != -1));

            if (is_N_neighbour)
              Trees_pt[i]->neighbour_pt(N) = Trees_pt[j];
            if (is_S_neighbour)
              Trees_pt[i]->neighbour_pt(S) = Trees_pt[j];
            if (is_E_neighbour)
              Trees_pt[i]->neighbour_pt(E) = Trees_pt[j];
            if (is_W_neighbour)
              Trees_pt[i]->neighbour_pt(W) = Trees_pt[j];
          }
          else
          {
            throw_runtime_error("Check tri neighbors of quads");
          }
        }
        else
        {
          throw_runtime_error("Check neighbors of tris");
        }
      }
    }
  }

  bool TemplatedMeshBase2d::refinement_possible()
    {
      bool allquads = true;
      for (unsigned int i = 0; i < this->nelement(); i++)
      {
        allquads = allquads && (dynamic_cast<oomph::QuadElementBase *>(this->element_pt(i)) != NULL);
      }
      if (allquads)
      {
        return true;
      }
      else
      {
        if (this->max_refinement_level() && issued_tri_refinement_warning==false && !this->problem->is_quiet())
        {
          std::cout << "WARNING: Found a tri or something in the mesh "<< this->domainname << " -> cannot be adaptive right now. Requires to implement a good tree for mixed meshes" << std::endl;
          issued_tri_refinement_warning = true;
        }
        return false;
      }
    }

  void TemplatedMeshBase2d::setup_tree_forest()
    {
      if (refinement_possible())  setup_quadtree_forest();      
      else {        
        if (issued_tri_refinement_warning==false && !this->problem->is_quiet())
        {
          std::cout << "WARNING: Found a tri or something in the mesh "<< this->domainname << " -> cannot be adaptive right now. Requires to implement a good tree for mixed meshes" << std::endl;
          issued_tri_refinement_warning = true;
        }
        this->disable_adaptation();
      }
    }

  void TemplatedMeshBase2d::setup_boundary_element_info()
  {
    std::ostringstream oss;
    setup_boundary_element_info(oss);
  }

  unsigned TemplatedMeshBase2d::add_tri_C1(Node* & n1, Node* & n2, Node* & n3)
  {
    BulkElementBase::__CurrentCodeInstance = codeinst;
    unsigned res=this->add_new_element(new BulkElementTri2dC1(),{n1,n2,n3});
    BulkElementBase::__CurrentCodeInstance = NULL;    
    return res;
  }

  unsigned TemplatedMeshBase2d::add_tri_C1TB(Node* & n1, Node* & n2, Node* & n3, Node* & n4)
  {
    BulkElementBase::__CurrentCodeInstance = codeinst;
    if (!n4)
    {
      auto * functable=codeinst->get_func_table();
      unsigned ntot = functable->total_num_fields_basebulk;
      n4=new pyoomph::BoundaryNode(n1->time_stepper_pt(),n1->nlagrangian(), n1->nlagrangian_type(), n1->ndim(), n1->nposition_type(), ntot);
      for (unsigned t=0;t<n1->time_stepper_pt()->ntstorage();t++)
      {
        for (unsigned ni=0;ni<n1->ndim();ni++)
        {
          n4->variable_position_pt()->set_value(t,ni,(n1->x(t,ni)+n2->x(t,ni)+n3->x(t,ni))/3.0);
        }        
        for (unsigned ni=0;ni<ntot;ni++)
        {
          n4->set_value(t,ni,(n1->value(t,ni)+n2->value(t,ni)+n3->value(t,ni))/3.0);
        }        
      }
      this->add_node_pt(n4);
    }
    unsigned res=this->add_new_element(new BulkElementTri2dC1TB(),{n1,n2,n3,n4});
    BulkElementBase::__CurrentCodeInstance = NULL;    
    return res;
  }  

  void TemplatedMeshBase2d::setup_boundary_element_info(std::ostream &outfile)
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

    setup_boundary_element_info_quads(outfile);
    setup_boundary_element_info_tris(outfile);

    // Now process the boundary elements once more    
    for (unsigned int bi=0;bi<Boundary_element_pt.size();bi++)
    {
      std::vector<unsigned> remove;
      for (unsigned int i=0;i<Boundary_element_pt[bi].size();i++)
      {
        oomph::FiniteElement *fe_pt=Boundary_element_pt[bi][i];
        // TODO: Check whether all nodes are indeed on the boundary
        for (unsigned int j=0;j<fe_pt->nnode_on_face();j++)
        {          
          oomph::Node *n=fe_pt->node_pt(fe_pt->get_bulk_node_number(Face_index_at_boundary[bi][i], j));
          if (n->is_on_boundary(bi))
          {
          }
          else
          {
            remove.push_back(i);
            break;
          }
        }
      }
      // Remove elements that are not actually on the boundary
      // TODO: This is a bit of a hack, we might want to consider:
      //   Pure refineable QuadMesh -> just call the normal quad setup 
      //   Mixed Tri/Quad mesh -> can't be refined -> take facets from the MeshTemplate and add them as boundary elements by hand (might be faster and for sure more robust)
      //   however, the add_nodes_to_boundary method of the mesh template should be add_facet_to_boundary -> facet identification
      //   Deprecate the add_node_to_boundary method
      for (unsigned int i=remove.size();i>0;i--)
      {
        Boundary_element_pt[bi].erase(Boundary_element_pt[bi].begin()+remove[i-1]);
        Face_index_at_boundary[bi].erase(Face_index_at_boundary[bi].begin()+remove[i-1]);
      }
    }

    Lookup_for_elements_next_boundary_is_setup = true;
  }

  void TemplatedMeshBase2d::setup_boundary_element_info_quads(std::ostream &outfile)
  {
    unsigned nbound = nboundary();

    oomph::Vector<oomph::Vector<oomph::FiniteElement *>> vector_of_boundary_element_pt;
    vector_of_boundary_element_pt.resize(nbound);

    // Matrix map for working out the fixed local coord for elements on boundary
    oomph::MapMatrixMixed<unsigned, oomph::FiniteElement *, oomph::Vector<int> *>
        boundary_identifier;

    // Loop over elements
    //-------------------
    unsigned nel = nelement();
    for (unsigned e = 0; e < nel; e++)
    {
      // Get pointer to element
      oomph::FiniteElement *fe_pt = finite_element_pt(e);
      if (!dynamic_cast<oomph::QuadElementBase *>(fe_pt))
        continue; // Don't do this on tris
      if (fe_pt->dim() == 2)
      {
        // Loop over the element's nodes and find out which boundaries they're on
        // ----------------------------------------------------------------------
        unsigned nnode_1d = fe_pt->nnode_1d();

        // Loop over nodes in order
        for (unsigned i0 = 0; i0 < nnode_1d; i0++)
        {
          for (unsigned i1 = 0; i1 < nnode_1d; i1++)
          {
            // Local node number
            unsigned j = i0 + i1 * nnode_1d;

            // Get pointer to vector of boundaries that this
            // node lives on
            std::set<unsigned> *boundaries_pt = 0;
            fe_pt->node_pt(j)->get_boundaries_pt(boundaries_pt);

            // If the node lives on some boundaries....
            if (boundaries_pt != 0)
            {
              // Loop over boundaries
              // unsigned nbound=(*boundaries_pt).size();
              for (std::set<unsigned>::iterator it = boundaries_pt->begin();
                   it != boundaries_pt->end(); ++it)
              {
                // Add pointer to finite element to vector for the appropriate
                // boundary

                // Does the pointer already exits in the vector
                oomph::Vector<oomph::FiniteElement *>::iterator b_el_it =
                    std::find(vector_of_boundary_element_pt[*it].begin(),
                              vector_of_boundary_element_pt[*it].end(),
                              fe_pt);
                // Only insert if we have not found it (i.e. got to the end)
                if (b_el_it == vector_of_boundary_element_pt[*it].end())
                {
                  vector_of_boundary_element_pt[*it].push_back(fe_pt);
                }

                // For the current element/boundary combination, create
                // a vector that stores an indicator which element boundaries
                // the node is located (boundary_identifier=-/+1 for nodes
                // on the left/right boundary; boundary_identifier=-/+2 for nodes
                // on the lower/upper boundary. We determine these indices
                // for all corner nodes of the element and add them to a vector
                // to a vector. This allows us to decide which face of the element
                // coincides with the boundary since the (quad!) element must
                // have exactly two corner nodes on the boundary.
                if (boundary_identifier(*it, fe_pt) == 0)
                {
                  boundary_identifier(*it, fe_pt) = new oomph::Vector<int>;
                }

                // Are we at a corner node?
                if (((i0 == 0) || (i0 == nnode_1d - 1)) && ((i1 == 0) || (i1 == nnode_1d - 1)))
                {
                  // Create index to represent position relative to s_0
                  (*boundary_identifier(*it, fe_pt)).push_back(1 * (2 * i0 / (nnode_1d - 1) - 1));

                  // Create index to represent position relative to s_1
                  (*boundary_identifier(*it, fe_pt)).push_back(2 * (2 * i1 / (nnode_1d - 1) - 1));
                }
              }
            }
            // else
            //  {
            //   oomph_info << "...does not live on any boundaries " << std::endl;
            //  }
          }
        }
      }
    }

    for (unsigned i = 0; i < nbound; i++)
    {
      typedef oomph::Vector<oomph::FiniteElement *>::iterator IT;
      for (IT it = vector_of_boundary_element_pt[i].begin();
           it != vector_of_boundary_element_pt[i].end();
           it++)
      {
        // Recover pointer to element
        oomph::FiniteElement *fe_pt = *it;

        // Initialise count for boundary identiers (-2,-1,1,2)
        std::map<int, unsigned> count;

        // Loop over coordinates
        for (unsigned ii = 0; ii < 2; ii++)
        {
          // Loop over upper/lower end of coordinates
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
        delete boundary_identifier(i, fe_pt);
        int indicator = -10;
        for (unsigned ii = 0; ii < 2; ii++)
        {
          for (int sign = -1; sign < 3; sign += 2)
          {
            if (count[(ii + 1) * sign] == 2)
            {
              indicator = (ii + 1) * sign;
              Boundary_element_pt[i].push_back(*it);
              Face_index_at_boundary[i].push_back(indicator);
            }
          }
        }
      }
    }
  }

  void TemplatedMeshBase2d::setup_interior_boundary_elements(unsigned bindex)
  {
    unsigned nel = nelement();
    for (unsigned int ie = 0; ie < nel; ie++)
    {
      oomph::FiniteElement *fe_pt = finite_element_pt(ie);
      if (!dynamic_cast<oomph::TElementBase *>(fe_pt))
        continue; // Only on triangles
      if (fe_pt->node_pt(0)->is_on_boundary(bindex) && fe_pt->node_pt(1)->is_on_boundary(bindex))
      {
        Boundary_element_pt[bindex].push_back(fe_pt);
        Face_index_at_boundary[bindex].push_back(2);
      }
      if (fe_pt->node_pt(0)->is_on_boundary(bindex) && fe_pt->node_pt(2)->is_on_boundary(bindex))
      {
        Boundary_element_pt[bindex].push_back(fe_pt);
        Face_index_at_boundary[bindex].push_back(1);
      }
      if (fe_pt->node_pt(1)->is_on_boundary(bindex) && fe_pt->node_pt(2)->is_on_boundary(bindex))
      {
        Boundary_element_pt[bindex].push_back(fe_pt);
        Face_index_at_boundary[bindex].push_back(0);
      }
    }
  }

  void TemplatedMeshBase2d::setup_boundary_element_info_tris(std::ostream &outfile2)
  {
    std::ostream &outfile = std::cout;
    bool doc = false;
    unsigned nbound = nboundary();

    oomph::Vector<oomph::Vector<oomph::FiniteElement *>> vector_of_boundary_element_pt;
    vector_of_boundary_element_pt.resize(nbound);

    // Matrix map for working out the fixed face for elements on boundary
    oomph::MapMatrixMixed<unsigned, oomph::FiniteElement *, int> face_identifier;

    // Loop over elements
    //-------------------
    unsigned nel = nelement();

    // Get pointer to vector of boundaries that the
    // node lives on
    oomph::Vector<std::set<unsigned> *> boundaries_pt(3, 0);

    // Data needed to deal with edges through the
    // interior of the domain
    std::map<oomph::Edge, unsigned> edge_count;
    std::map<oomph::Edge, oomph::TriangleBoundaryHelper::BCInfo> edge_bcinfo;
    std::map<oomph::Edge, oomph::TriangleBoundaryHelper::BCInfo> face_info;
    oomph::MapMatrixMixed<unsigned, oomph::FiniteElement *, int> face_count;
    oomph::Vector<unsigned> bonus(nbound);

    std::map<oomph::Edge, oomph::Vector<oomph::TriangleBoundaryHelper::BCInfo>> edge_internal_bnd;

    for (unsigned e = 0; e < nel; e++)
    {
      // Get pointer to element
      oomph::FiniteElement *fe_pt = finite_element_pt(e);
      if (!dynamic_cast<oomph::TElementBase *>(fe_pt))
      {
        continue; // Only on triangles
      }
      if (doc)
      {
        outfile << "Element: " << e << " " << fe_pt << std::endl;
      }
      if (fe_pt->dim() == 2)
      {
        // Loop over the element's nodes and find out which boundaries they're on
        // ----------------------------------------------------------------------

        // We need only loop over the corner nodes
        for (unsigned i = 0; i < 3; i++)
        {
          fe_pt->node_pt(i)->get_boundaries_pt(boundaries_pt[i]);
          if (doc)
          {
            if (boundaries_pt[i])
            {
              outfile << "  Side : " << i << ": ";
              for (auto &b : *(boundaries_pt[i]))
                outfile << b << "  ";
              outfile << std::endl;
            }
          }
        }

        // Find the common boundaries of each edge
        oomph::Vector<std::set<unsigned>> edge_boundary(3);

        // Edge 0 connects points 1 and 2
        //-----------------------------

        if (boundaries_pt[1] && boundaries_pt[2])
        {
          // Create the corresponding edge
          oomph::Edge edge0(fe_pt->node_pt(1), fe_pt->node_pt(2));

          // Update infos about this edge
          oomph::TriangleBoundaryHelper::BCInfo info;
          info.Face_id = 0;
          info.FE_pt = fe_pt;

          std::set_intersection(boundaries_pt[1]->begin(), boundaries_pt[1]->end(),
                                boundaries_pt[2]->begin(), boundaries_pt[2]->end(),
                                std::insert_iterator<std::set<unsigned>>(
                                    edge_boundary[0], edge_boundary[0].begin()));
          std::set<unsigned>::iterator it0 = edge_boundary[0].begin();

          // Edge does exist:
          if (edge_boundary[0].size() > 0)
          {
            info.Boundary = *it0;

            // How many times this edge has been visited
            edge_count[edge0]++;

            // Update edge_bcinfo
            edge_bcinfo.insert(std::make_pair(edge0, info));

            // ... and also update the info associated with internal bnd
            edge_internal_bnd[edge0].push_back(info);
          }
        }

        // Edge 1 connects points 0 and 2
        //-----------------------------

        if (boundaries_pt[0] && boundaries_pt[2])
        {
          std::set_intersection(boundaries_pt[0]->begin(), boundaries_pt[0]->end(),
                                boundaries_pt[2]->begin(), boundaries_pt[2]->end(),
                                std::insert_iterator<std::set<unsigned>>(
                                    edge_boundary[1], edge_boundary[1].begin()));

          // Create the corresponding edge
          oomph::Edge edge1(fe_pt->node_pt(0), fe_pt->node_pt(2));

          // Update infos about this edge
          oomph::TriangleBoundaryHelper::BCInfo info;
          info.Face_id = 1;
          info.FE_pt = fe_pt;
          std::set<unsigned>::iterator it1 = edge_boundary[1].begin();

          // Edge does exist:
          if (edge_boundary[1].size() > 0)
          {
            info.Boundary = *it1;

            // How many times this edge has been visited
            edge_count[edge1]++;

            // Update edge_bcinfo
            edge_bcinfo.insert(std::make_pair(edge1, info));

            // ... and also update the info associated with internal bnd
            edge_internal_bnd[edge1].push_back(info);
          }
        }

        // Edge 2 connects points 0 and 1
        //-----------------------------

        if (boundaries_pt[0] && boundaries_pt[1])
        {
          std::set_intersection(boundaries_pt[0]->begin(), boundaries_pt[0]->end(),
                                boundaries_pt[1]->begin(), boundaries_pt[1]->end(),
                                std::insert_iterator<std::set<unsigned>>(
                                    edge_boundary[2], edge_boundary[2].begin()));

          // Create the corresponding edge
          oomph::Edge edge2(fe_pt->node_pt(0), fe_pt->node_pt(1));

          // Update infos about this edge
          oomph::TriangleBoundaryHelper::BCInfo info;
          info.Face_id = 2;
          info.FE_pt = fe_pt;
          std::set<unsigned>::iterator it2 = edge_boundary[2].begin();

          // Edge does exist:
          if (edge_boundary[2].size() > 0)
          {
            info.Boundary = *it2;

            // How many times this edge has been visited
            edge_count[edge2]++;

            // Update edge_bcinfo
            edge_bcinfo.insert(std::make_pair(edge2, info));

            // ... and also update the info associated with internal bnd
            edge_internal_bnd[edge2].push_back(info);
          }
        }

#ifdef PARANOID

        // Check if edge is associated with multiple boundaries

        // We now know whether any edges lay on the boundaries
        for (unsigned i = 0; i < 3; i++)
        {
          // How many boundaries are there
          unsigned count = 0;

          // Loop over all the members of the set and add to the count
          // and set the boundary
          for (std::set<unsigned>::iterator it = edge_boundary[i].begin();
               it != edge_boundary[i].end(); ++it)
          {
            ++count;
          }

          // If we're on more than one boundary, this is weird, so die
          if (count > 1)
          {
            std::ostringstream error_stream;
            error_stream << "Edge " << i << " is located on " << count << " boundaries.\n";
            error_stream << "This is rather strange, so I'm going to die\n";
            throw oomph::OomphLibError(
                error_stream.str(),
                OOMPH_CURRENT_FUNCTION,
                OOMPH_EXCEPTION_LOCATION);
          }
        }

#endif

        // Now we set the pointers to the boundary sets to zero
        for (unsigned i = 0; i < 3; i++)
        {
          boundaries_pt[i] = 0;
        }
      }
    } // end of loop over all elements

    // Loop over all edges that are located on a boundary
    typedef std::map<oomph::Edge, oomph::TriangleBoundaryHelper::BCInfo>::iterator ITE;
    for (ITE it = edge_bcinfo.begin();
         it != edge_bcinfo.end();
         it++)
    {
      oomph::Edge current_edge = it->first;
      unsigned bound = it->second.Boundary;

      // If the edge has been visited only once
      if (edge_count[current_edge] == 1) // TODO: This is important for some boundarys along a corner |_\. However, it prevents from adding internal boundaries
      {
        // Count the edges that are on the same element and on the same boundary
        face_count(static_cast<unsigned>(bound), it->second.FE_pt) =
            face_count(static_cast<unsigned>(bound), it->second.FE_pt) + 1;

        // If such edges exist, let store the corresponding element
        if (face_count(bound, it->second.FE_pt) > 1)
        {
          // Update edge's infos
          oomph::TriangleBoundaryHelper::BCInfo info;
          info.Face_id = it->second.Face_id;
          info.FE_pt = it->second.FE_pt;
          info.Boundary = it->second.Boundary;

          // Add it to FIinfo, that stores infos of problematic elements
          face_info.insert(std::make_pair(current_edge, info));

          // How many edges on which boundary have to be added
          bonus[bound]++;
        }
        else
        {
          // Add element and face to the appropriate vectors
          //  Does the pointer already exits in the vector
          oomph::Vector<oomph::FiniteElement *>::iterator b_el_it =
              std::find(vector_of_boundary_element_pt[static_cast<unsigned>(bound)].begin(),
                        vector_of_boundary_element_pt[static_cast<unsigned>(bound)].end(),
                        it->second.FE_pt);

          // Only insert if we have not found it (i.e. got to the end)
          if (b_el_it == vector_of_boundary_element_pt[static_cast<unsigned>(bound)].end())
          {
            vector_of_boundary_element_pt[static_cast<unsigned>(bound)].push_back(it->second.FE_pt);
          }

          // set_of_boundary_element_pt[static_cast<unsigned>(bound)].insert(
          //  it->second.FE_pt);
          face_identifier(static_cast<unsigned>(bound), it->second.FE_pt) =
              it->second.Face_id;
        }
      }

    } // End of "adding-boundaries"-loop

    // In mixed meshes, you can have a quad attached to a triangle at the sharp boundary corner. #> 
    // This would then consider the following line as boundary as well: #|>
    // So we must remove all quad boundaries again

    // Now copy everything across into permanent arrays
    //-------------------------------------------------

    // Loop over boundaries
    for (unsigned i = 0; i < nbound; i++)
    {

      // Fill the boundaries of the quads
      std::vector<std::set<oomph::Node*>> quad_boundary_nodes;
      for (unsigned int iq=0;iq<Boundary_element_pt[i].size();iq++)
      {
        oomph::QuadElementBase *qeb=dynamic_cast<oomph::QuadElementBase *>(Boundary_element_pt[i][iq]);
        
        std::vector<unsigned> qboundinds;
        int side=Face_index_at_boundary[i][iq];
        if (side==-2) { qboundinds={0,1};}
        else if (side==2) { qboundinds={2,3};}
        else if (side==-1) { qboundinds={0,2};}
        else if (side==1) { qboundinds={1,3};}
        else throw_runtime_error("Unknown side for quad boundary");
        std::set<oomph::Node*> qboundnodes;
        for (unsigned vni : qboundinds)
        {
          if (!dynamic_cast<oomph::BoundaryNodeBase*>(qeb->vertex_node_pt(vni))) throw_runtime_error("Quad boundary node is not a boundary node: side "+std::to_string(side) );
          if (!dynamic_cast<oomph::BoundaryNodeBase*>(qeb->vertex_node_pt(vni))->is_on_boundary(i)) throw_runtime_error("Quad boundary node is not on the required boundary: side "+std::to_string(side));
          {
            qboundnodes.insert(qeb->vertex_node_pt(vni));
          }
        }
        quad_boundary_nodes.push_back(qboundnodes);
        
      }

      // Number of elements on this boundary that have to be added
      // in addition to other elements
      unsigned bonus1 = bonus[i];

      // Number of elements on this boundary
      unsigned nel = vector_of_boundary_element_pt[i].size() + bonus1;

      // Allocate storage for the coordinate identifiers

      unsigned e_count = Face_index_at_boundary[i].size();
      Face_index_at_boundary[i].resize(e_count + nel); // Resize: Hence, this may only be called after the quads!

      typedef oomph::Vector<oomph::FiniteElement *>::iterator IT;
      for (IT it = vector_of_boundary_element_pt[i].begin();
           it != vector_of_boundary_element_pt[i].end();
           it++)
      {
        // Recover pointer to element
        oomph::FiniteElement *fe_pt = *it;

        // Find the vertex nodes
        int side=face_identifier(i, fe_pt);
        oomph::BoundaryNodeBase *n0 = dynamic_cast<oomph::BoundaryNodeBase*>(fe_pt->vertex_node_pt((side+2)%3));
        oomph::BoundaryNodeBase *n1 = dynamic_cast<oomph::BoundaryNodeBase*>(fe_pt->vertex_node_pt((side+1)%3));
        if (!n0 || !n1)
        {
          throw_runtime_error("Boundary element does not have boundary nodes on the required side: " + std::to_string(side));//+"  "+std::to_string((long unsigned)n0)+"  "+std::to_string((long unsigned)n1));
        }
        if (!n0->is_on_boundary(i) || !n1->is_on_boundary(i))
        {
          throw_runtime_error("Boundary element does not have boundary nodes on the required boundary: " + std::to_string(i));
        }
        std::set<oomph::Node*> tboundnodes{dynamic_cast<oomph::Node*>(n0), dynamic_cast<oomph::Node*>(n1)};
        bool found=false;
        for (unsigned int iq=0;iq<quad_boundary_nodes.size();iq++)
        {
          if (quad_boundary_nodes[iq]==tboundnodes)
          {
            //std::cout << "Found quad boundary element on boundary " << i << " with nodes " << n0 << " and " << n1 << std::endl;
            quad_boundary_nodes.erase(quad_boundary_nodes.begin()+iq); // Remove this quad boundary element
            found=true;
            break; // And do not add the triangle boundary element
          }
        }
        if (found)
        {
          continue; // We have found a quad boundary element, so do not add the triangle boundary element
        }

        // Add to permanent storage
        Boundary_element_pt[i].push_back(fe_pt);

        Face_index_at_boundary[i][e_count] = face_identifier(i, fe_pt);

        

        // Increment counter
        e_count++;
      }
      // We add the elements that have two or more edges on this boundary
      for (ITE itt = face_info.begin(); itt != face_info.end(); itt++)
      {
        if (itt->second.Boundary == i)
        {

            // Find the vertex nodes
          int side=itt->second.Face_id;
          oomph::BoundaryNodeBase *n0 = dynamic_cast<oomph::BoundaryNodeBase*>(itt->second.FE_pt->vertex_node_pt((side+2)%3));
          oomph::BoundaryNodeBase *n1 = dynamic_cast<oomph::BoundaryNodeBase*>(itt->second.FE_pt->vertex_node_pt((side+1)%3));
          if (!n0 || !n1)
          {
            throw_runtime_error("Boundary element does not have boundary nodes on the required side")//;: " + std::to_string(side)+"  "+std::to_string((long unsigned)n0)+"  "+std::to_string((long unsigned)n1));
          }
          if (!n0->is_on_boundary(i) || !n1->is_on_boundary(i))
          {
            throw_runtime_error("Boundary element does not have boundary nodes on the required boundary: " + std::to_string(i));
          }
          std::set<oomph::Node*> tboundnodes{dynamic_cast<oomph::Node*>(n0), dynamic_cast<oomph::Node*>(n1)};
          bool found=false;
          for (unsigned int iq=0;iq<quad_boundary_nodes.size();iq++)
          {
            if (quad_boundary_nodes[iq]==tboundnodes)
            {
              //std::cout << "SECOND Found quad boundary element on boundary " << i << " with nodes " << n0 << " and " << n1 << std::endl;
              quad_boundary_nodes.erase(quad_boundary_nodes.begin()+iq); // Remove this quad boundary element
              found=true;
              break; // And do not add the triangle boundary element
            }
          }
          if (found)
          {
            continue; // We have found a quad boundary element, so do not add the triangle boundary element
          }

          // Add to permanent storage
          Boundary_element_pt[i].push_back(itt->second.FE_pt);

          Face_index_at_boundary[i][e_count] = itt->second.Face_id;

          e_count++;
        }
      }

    } // End of loop over boundaries

    // Doc?
    //-----
    if (doc)
    {
      // Loop over boundaries
      for (unsigned i = 0; i < nbound; i++)
      {
        unsigned nel = Boundary_element_pt[i].size();
        outfile << "Boundary: " << i
                << " is adjacent to " << nel
                << " elements" << std::endl;

        // Loop over elements on given boundary
        for (unsigned e = 0; e < nel; e++)
        {
          oomph::FiniteElement *fe_pt = Boundary_element_pt[i][e];
          outfile << "Boundary element:" << fe_pt
                  << " Face index of boundary is "
                  << Face_index_at_boundary[i][e] << std::endl;
        }
      }
    }
  }



  void TemplatedMeshBase2d::fill_internal_facet_buffers(std::vector<BulkElementBase*> & internal_elements, std::vector<int> & internal_face_dir,std::vector<BulkElementBase*> & opposite_elements,std::vector<int> & opposite_face_dir,std::vector<int> & opposite_already_at_index)
  {
    using namespace oomph::QuadTreeNames;
    internal_elements.clear();
    internal_face_dir.clear();
    opposite_elements.clear();
    opposite_face_dir.clear();
    opposite_already_at_index.clear();
    
    oomph::TreeBasedRefineableMeshBase *tbself = dynamic_cast<oomph::TreeBasedRefineableMeshBase *>(this);
    if (tbself)
    {
        unsigned milev = 0, malev = 0;
        tbself->get_refinement_levels(milev, malev);
        if (milev < malev) // We must do it entirely differntly when we have hanging nodes
        {
			oomph::Vector<int> edges(4);
			edges[0] = S;
			edges[1] = N;
			edges[2] = W;
			edges[3] = E;      
			std::map<int,int> revedges; for (unsigned int i=0;i<edges.size();i++) revedges[edges[i]]=i;
			std::vector<int> edge_to_face_dir={-2,2,-1,1};
			std::map<BulkElementBase*,std::array<int,4>> constructed_facets;
			
          for (unsigned int ie=0;ie<this->nelement();ie++)
          {
            BulkElementBase* be=dynamic_cast<BulkElementBase*>(this->element_pt(ie));
            oomph::RefineableQElement<2>* beq=dynamic_cast<oomph::RefineableQElement<2>*>(be);
            if (!beq) 
            {
             throw_runtime_error("Mixed meshes here");
            }
            for (unsigned edge_counter = 0; edge_counter < 4; edge_counter++)
				{
					oomph::Vector<unsigned> translate_s(2);
					oomph::Vector<double> s(2), s_lo_neigh(2), s_hi_neigh(2), s_fraction(2);
					int neigh_edge, diff_level;
					bool in_neighbouring_tree;
					oomph::QuadTree *neigh_pt;
					neigh_pt = beq->quadtree_pt()->gteq_edge_neighbour(edges[edge_counter], translate_s, s_lo_neigh, s_hi_neigh, neigh_edge, diff_level, in_neighbouring_tree);
					if (neigh_pt != nullptr && neigh_pt->is_leaf())
					{
					 
					 BulkElementBase * adj=dynamic_cast<BulkElementBase *>(neigh_pt->object_pt());

					 if (!constructed_facets.count(be)) constructed_facets[be]={-1,-1,-1,-1};
					 if (!constructed_facets.count(adj)) constructed_facets[adj]={-1,-1,-1,-1};
					 int neigh_edge_counter=revedges[neigh_edge];

                oomph::Vector<double> mp1=be->get_Eulerian_midpoint_from_local_coordinate();
				    oomph::Vector<double> mp2=adj->get_Eulerian_midpoint_from_local_coordinate();
//					 std::cout << "NEIGHT PT " << ie << "  : " << be << "("<<mp1[0] << ", " << mp1[1]<<") , " << edge_counter << "  and  " << adj << "("<<mp2[0] << ", " << mp2[1]<<") ,  " << neigh_edge_counter << "  DL " << diff_level << std::endl;
					 					 
					 if (diff_level==0)
					 {
					   if (constructed_facets[adj][neigh_edge_counter]<0)
					   {
							if (constructed_facets[be][edge_counter]>=0) 
		               {
		                  throw_runtime_error("Facet was already constructed before, this should not happen");
		               }
		               constructed_facets[be][edge_counter]=opposite_elements.size();
		               constructed_facets[adj][neigh_edge_counter]=opposite_elements.size();		               
		               internal_elements.push_back(be);
		               internal_face_dir.push_back(edge_to_face_dir[edge_counter]);		               
		               opposite_elements.push_back(adj);
		               opposite_face_dir.push_back(edge_to_face_dir[neigh_edge_counter]);
                     opposite_already_at_index.push_back(-1);       		               
					   }
					   else if (constructed_facets[be][edge_counter]<0) 
					   {
                   throw_runtime_error("Facet should be actually there, but is not. This should not happen");
					   }
					   else
					   {
//					    std::cout << "  SKIPPED " << std::endl;
					   }
					 }
					 else
					 {
					 		if (constructed_facets[be][edge_counter]>=0) 
		               {
		                  throw_runtime_error("Facet was already constructed before, this should not happen");
		               }
		               int already_constructed=-1;
		               if (constructed_facets[adj][neigh_edge_counter]>=0)
					      {
					       already_constructed=constructed_facets[adj][neigh_edge_counter];
					      }
					      else
					      {
					        constructed_facets[adj][neigh_edge_counter]=opposite_elements.size();		               		               
					      }
		               constructed_facets[be][edge_counter]=opposite_elements.size();		               
		               internal_elements.push_back(be);
		               internal_face_dir.push_back(edge_to_face_dir[edge_counter]);		               
		               opposite_elements.push_back(adj);
		               opposite_face_dir.push_back(edge_to_face_dir[neigh_edge_counter]);
		               opposite_already_at_index.push_back(already_constructed);
					 }
					}
            }          
          }
        std::cout << "Number of internal facets to be constructed: " << internal_elements.size() << std::endl;
          return;
        }
    }
        

    std::map<std::pair<oomph::Node*,oomph::Node*>,std::vector<std::pair<BulkElementBase*,int>>> nodemap;
    for (unsigned int ie=0;ie<this->nelement();ie++)
    {     
     BulkElementBase* be=dynamic_cast<BulkElementBase*>(this->element_pt(ie));
     std::set<oomph::Node*> elem_vertices;
     for (unsigned int ivn=0;ivn<be->nvertex_node();ivn++) elem_vertices.insert(be->vertex_node_pt(ivn));
     std::vector<int> face_dirs;
     if (elem_vertices.size()==3) //Triangular element
     {
       face_dirs={0,1,2};
     }
     else if (elem_vertices.size()==4) //Quad element
     {
       face_dirs={1,-1,2,-2};
     }
     else 
     {
      throw_runtime_error("Should not enter here");
     }
     
     for (auto face_dir : face_dirs)
     {
       unsigned nnode_on_face=be->nnode_on_face();
       std::set<oomph::Node*> face_nodes;
       for (unsigned int nf=0;nf<nnode_on_face;nf++)
       {
         face_nodes.insert(be->node_pt(be->get_bulk_node_number(face_dir,nf)));
       }
       std::vector<oomph::Node*> face_verts;
       std::set_intersection(elem_vertices.begin(), elem_vertices.end(), face_nodes.begin(), face_nodes.end(), std::back_inserter(face_verts));
       if (face_verts.size()!=2)
       {
        throw_runtime_error("Expected 2 vertex nodes on face, but got "+std::to_string(face_verts.size()));
       }
       for (unsigned int i=0;i<face_verts.size();i++) if (face_verts[i]->is_a_copy()) face_verts[i]=face_verts[i]->copied_node_pt();
       std::sort(face_verts.begin(),face_verts.end());
       std::pair<oomph::Node*,oomph::Node*> key={face_verts[0],face_verts[1]};
       if (!nodemap.count(key))
       {
        nodemap[key]={std::make_pair(be,face_dir)};
       }
       else
       {
        nodemap[key].push_back(std::make_pair(be,face_dir));
       }
     }
    }
    
    for (const auto & nm : nodemap)
    {
     if (nm.second.size()>2)
     {
      throw_runtime_error("Found a facet with "+std::to_string(nm.second.size())+" attached elements");
     }
     else if (nm.second.size()==2)
     {
       internal_elements.push_back(nm.second[0].first);
       internal_face_dir.push_back(nm.second[0].second);
       opposite_elements.push_back(nm.second[1].first);
       opposite_face_dir.push_back(nm.second[1].second);      
       opposite_already_at_index.push_back(-1);       
     }
    }
    
    std::cout << "Number of internal facets to be constructed: " << internal_elements.size() << std::endl;
    
  }
  
  
}
