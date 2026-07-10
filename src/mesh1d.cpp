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
#include "mesh1d.hpp"

namespace pyoomph
{

  void TemplatedMeshBase1d::setup_boundary_element_info()
  {
    std::ostringstream oss;
    setup_boundary_element_info(oss);
  }

  void TemplatedMeshBase1d::setup_boundary_element_info(std::ostream &outfile)
  {
    // Initialise documentation flag
    bool doc = false;

    // Set this to true if an open file has been passed to the function
    if (outfile)
    {
      doc = true;
    }

    // Determine number of boundaries in mesh
    const unsigned n_bound = nboundary();

    // Wipe/allocate storage for arrays
    Boundary_element_pt.clear();
    Face_index_at_boundary.clear();
    Boundary_element_pt.resize(n_bound);
    Face_index_at_boundary.resize(n_bound);

    // Matrix map for working out the fixed local coord for elements on boundary
    oomph::MapMatrixMixed<unsigned, oomph::FiniteElement *, oomph::Vector<int> *>
        boundary_identifier;

    // Determine number of elements in the mesh
    const unsigned n_element = nelement();

    // Loop over elements
    for (unsigned e = 0; e < n_element; e++)
    {
      // Get pointer to element
      oomph::FiniteElement *fe_pt = finite_element_pt(e);

      // Output information to output file
      if (doc)
      {
        outfile << "Element: " << e << " " << fe_pt << std::endl;
      }

      // Only include 1D elements! Some meshes contain interface elements too.
      if (fe_pt->dim() == 1)
      {
        // Loop over the element's nodes and find out which boundaries they're on
        // ----------------------------------------------------------------------

        // Determine number of nodes in the element
        const unsigned n_node = fe_pt->nnode_1d();

        // Loop over nodes in order
        for (unsigned n = 0; n < n_node; n++)
        {
          // Allocate storage for pointer to set of boundaries that node lives on
          std::set<unsigned> *boundaries_pt = 0;

          // Get pointer to vector of boundaries that this node lives on
          fe_pt->node_pt(n)->get_boundaries_pt(boundaries_pt);

          // If the node lives on some boundaries....
          if (boundaries_pt != 0)
          {
            // Determine number of boundaries which node lives on
            const unsigned n_bound_node_lives_on = (*boundaries_pt).size();

            // Throw an error if the node lives on more than one boundary
            if (n_bound_node_lives_on > 1)
            {
              std::string error_message =
                  "In a 1D mesh a node shouldn't be able to live on more than\n";
              error_message +=
                  "one boundary, yet this node claims to.";

              throw oomph::OomphLibError(error_message,
                                         OOMPH_CURRENT_FUNCTION,
                                         OOMPH_EXCEPTION_LOCATION);
            }
            // If the node lives on just one boundary
            else if (n_bound_node_lives_on == 1)
            {
              // Determine which boundary the node lives on
              const std::set<unsigned>::iterator boundary = boundaries_pt->begin();

              // In 1D if an element has any nodes on a boundary then it must
              // be a boundary element. This means that (unlike in the 2D and
              // 3D cases) we can immediately add this element to permanent
              // storage.
              Boundary_element_pt[*boundary].push_back(fe_pt);

              // Record information required for FaceElements.
              // `Face_index_at_boundary' = -/+1 for nodes on the left/right
              // boundary. This allows us to decide which edge of the element
              // coincides with the boundary since the line element must have
              // precisely one vertex node on the boundary.

              // Are we at the left-hand vertex node? (left face)
              if (n == 0)
              {
                Face_index_at_boundary[*boundary].push_back(-1);
              }

              // Are we at the right-hand vertex node? (right face)
              else if (n == n_node - 1)
              {
                Face_index_at_boundary[*boundary].push_back(1);
              }
            }
          } // End of if node lives on some boundaries

        } // End of loop over nodes
      }
    } // End of loop over elements

#ifdef PARANOID

    // Check each boundary only has precisely one element next to it
    // -------------------------------------------------------------
    // Only if not distributed
#ifdef OOMPH_HAS_MPI
    if (Comm_pt == 0)
#endif
    {
      // Loop over boundaries
      for (unsigned b = 0; b < n_bound; b++)
      {
        // Evaluate number of elements adjacent to boundary b
        const unsigned n_element = Boundary_element_pt[b].size();

        switch (n_element)
        {
          // Boundary b has no adjacent elements
        case 0:
        {
          std::ostringstream error_stream;
          error_stream << "Boundary " << b << " has no element adjacent to it\n";
          throw oomph::OomphLibError(error_stream.str(),
                                     OOMPH_CURRENT_FUNCTION,
                                     OOMPH_EXCEPTION_LOCATION);
          break;
        }
        // Boundary b has one adjacent element (this is good!)
        case 1:
          break;

          // Boundary b has more than one adjacent element
        default:
        {
          std::ostringstream error_stream;
          error_stream << "Boundary " << b << " has " << n_element
                       << " elements adjacent to it.\n"
                       << "This shouldn't occur in a 1D mesh.\n";
          throw oomph::OomphLibError(error_stream.str(),
                                     OOMPH_CURRENT_FUNCTION,
                                     OOMPH_EXCEPTION_LOCATION);
          break;
        }
        } // End of switch

        // Because each boundary should only have one element adjacent to it,
        // each `Face_index_at_boundary[b]' should be of size one.

        const unsigned face_index_at_boundary_size = Face_index_at_boundary[b].size();

        if (face_index_at_boundary_size != 1)
        {
          std::ostringstream error_stream;
          error_stream
              << "Face_index_at_boundary[" << b << "] has size"
              << face_index_at_boundary_size
              << " which does not make sense.\n"
              << "In a 1D mesh its size should always be one since only\n"
              << "one element can be adjacent to any particular boundary";
          throw oomph::OomphLibError(error_stream.str(),
                                     OOMPH_CURRENT_FUNCTION,
                                     OOMPH_EXCEPTION_LOCATION);
        }
      } // End of loop over boundaries
    }
#endif
    Lookup_for_elements_next_boundary_is_setup = true;
  }
  
  void TemplatedMeshBase1d::fill_internal_facet_buffers(std::vector<BulkElementBase*> & internal_elements, std::vector<int> & internal_face_dir,std::vector<BulkElementBase*> & opposite_elements,std::vector<int> & opposite_face_dir,std::vector<int> & opposite_already_at_index)
  {
    internal_elements.clear();
    internal_face_dir.clear();
    opposite_elements.clear();
    opposite_face_dir.clear();
    opposite_already_at_index.clear();
    std::map<oomph::Node*,std::pair<BulkElementBase*,int>> nodemap;
    std::set<oomph::Node*> completed_nodes;
    for (unsigned int ie=0;ie<this->nelement();ie++)
    {     
     BulkElementBase* be=dynamic_cast<BulkElementBase*>(this->element_pt(ie));
     for (unsigned int ivn=0;ivn<be->nvertex_node();ivn++)
     {
      oomph::Node * npt=be->vertex_node_pt(ivn);
      if (npt->is_a_copy()) 
      {
//       std::cout << "IS A COPY  " << npt << " -> " <<  npt->copied_node_pt() << std::endl;
       npt=npt->copied_node_pt();
      }
      if (!nodemap.count(npt))
      {      
        if (completed_nodes.count(npt)) throw_runtime_error("STRANGE, node already completed!");
        nodemap[npt]=std::make_pair(be,(ivn==0 ? -1 : 1));
      }
      else
      {
       internal_elements.push_back(be);
       internal_face_dir.push_back((ivn==0 ? -1 : 1));
       opposite_elements.push_back(nodemap[npt].first);
       opposite_face_dir.push_back(nodemap[npt].second);       
       opposite_already_at_index.push_back(-1);
       completed_nodes.insert(npt);
      }
     }
   }
//   std::cout << "NUMINTERAL " << internal_elements.size() << std::endl;
  }

}
