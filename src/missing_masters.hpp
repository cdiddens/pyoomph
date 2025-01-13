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


/*******************************
 This file is an adapted version of missing_masters.template.cc from oomph-lib
********************************/

#pragma once

// Oomph-lib headers
#include "geom_objects.h"
#include "problem.h"
#include "shape.h"

#include "mesh.h"
#include "mesh_as_geometric_object.h"
#include "algebraic_elements.h"
#include "macro_element_node_update_element.h"
#include "Qelements.h"
#include "element_with_external_element.h"
#include "missing_masters.h"

namespace oomph
{

  //// Templated helper functions for missing master methods,

#ifdef OOMPH_HAS_MPI

  //============start of add_external_halo_node_to_storage===============
  /// Helper function to add external halo nodes, including any masters,
  /// based on information received from the haloed process
  //=========================================================================
  template <class EXT_ELEMENT>
  void Missing_masters_functions::add_external_halo_node_to_storage(Node *&new_nod_pt, Mesh *const &mesh_pt, unsigned &loc_p,
                                                                    unsigned &node_index, FiniteElement *const &new_el_pt,
                                                                    int &n_cont_inter_values,
                                                                    unsigned &counter_for_recv_unsigneds,
                                                                    Vector<unsigned> &recv_unsigneds,
                                                                    unsigned &counter_for_recv_doubles,
                                                                    Vector<double> &recv_doubles)
  {
    // Add the external halo node if required
    add_external_halo_node_helper(new_nod_pt, mesh_pt, loc_p,
                                  node_index, new_el_pt,
                                  n_cont_inter_values,
                                  counter_for_recv_unsigneds,
                                  recv_unsigneds,
                                  counter_for_recv_doubles,
                                  recv_doubles);

    // Recursively add masters
    recursively_add_masters_of_external_halo_node_to_storage<EXT_ELEMENT>(new_nod_pt, mesh_pt, loc_p,
                                                                          node_index,
                                                                          n_cont_inter_values,
                                                                          counter_for_recv_unsigneds,
                                                                          recv_unsigneds,
                                                                          counter_for_recv_doubles,
                                                                          recv_doubles);
  }

  //========================================================================
  /// Recursively add masters of external halo nodes (and their masters, etc)
  /// based on information received from the haloed process
  //=========================================================================
  template <class EXT_ELEMENT>
  void Missing_masters_functions::
      recursively_add_masters_of_external_halo_node_to_storage(Node *&new_nod_pt, Mesh *const &mesh_pt, unsigned &loc_p,
                                                               unsigned &node_index,
                                                               int &n_cont_inter_values,
                                                               unsigned &counter_for_recv_unsigneds,
                                                               Vector<unsigned> &recv_unsigneds,
                                                               unsigned &counter_for_recv_doubles,
                                                               Vector<double> &recv_doubles)
  {

    for (int i_cont = -1; i_cont < n_cont_inter_values; i_cont++)
    {
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
      oomph_info
          << "Rec:" << counter_for_recv_unsigneds
          << " Boolean to indicate that continuously interpolated variable i_cont "
          << i_cont << " is hanging "
          << recv_unsigneds[counter_for_recv_unsigneds]
          << std::endl;
#endif
      if (recv_unsigneds[counter_for_recv_unsigneds++] == 1)
      {
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
        oomph_info
            << "Rec:" << counter_for_recv_unsigneds
            << "  Number of master nodes "
            << recv_unsigneds[counter_for_recv_unsigneds]
            << std::endl;
#endif
        unsigned n_master = recv_unsigneds
            [counter_for_recv_unsigneds++];

        // Setup new HangInfo
        HangInfo *hang_pt = new HangInfo(n_master);
        for (unsigned m = 0; m < n_master; m++)
        {
          Node *master_nod_pt = 0;
          // Get the master node (creating and adding it if required)
          add_external_halo_master_node_helper<EXT_ELEMENT>(master_nod_pt, new_nod_pt, mesh_pt, loc_p,
                                                            n_cont_inter_values,
                                                            counter_for_recv_unsigneds,
                                                            recv_unsigneds,
                                                            counter_for_recv_doubles,
                                                            recv_doubles);

          // Get the weight and set the HangInfo
          double master_weight = recv_doubles
              [counter_for_recv_doubles++];
          hang_pt->set_master_node_pt(m, master_nod_pt, master_weight);

          // Recursively add masters of master
          recursively_add_masters_of_external_halo_node_to_storage<EXT_ELEMENT>(master_nod_pt, mesh_pt, loc_p,
                                                                                node_index,
                                                                                n_cont_inter_values,
                                                                                counter_for_recv_unsigneds,
                                                                                recv_unsigneds,
                                                                                counter_for_recv_doubles,
                                                                                recv_doubles);
        }
        new_nod_pt->set_hanging_pt(hang_pt, i_cont);
      }
    } // end loop over continous interpolated values
  }

  //========================================================================
  /// Helper function to add external halo node that is a master
  //========================================================================
  template <class EXT_ELEMENT>
  void Missing_masters_functions::add_external_halo_master_node_helper(Node *&new_master_nod_pt, Node *&new_nod_pt, Mesh *const &mesh_pt,
                                                                       unsigned &loc_p, int &ncont_inter_values,
                                                                       unsigned &counter_for_recv_unsigneds,
                                                                       Vector<unsigned> &recv_unsigneds,
                                                                       unsigned &counter_for_recv_doubles,
                                                                       Vector<double> &recv_doubles)
  {
    // Given the node and the external mesh, and received information
    // about them from process loc_p, construct them on the current process
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
    oomph_info
        << "Rec:" << counter_for_recv_unsigneds
        << "  Boolean to trigger construction of new external halo master node "
        << recv_unsigneds[counter_for_recv_unsigneds]
        << std::endl;
#endif
    if (recv_unsigneds[counter_for_recv_unsigneds++] == 1)
    {
      // Construct a new node based upon sent information
      construct_new_external_halo_master_node_helper<EXT_ELEMENT>(new_master_nod_pt, new_nod_pt, loc_p, mesh_pt,
                                                                  counter_for_recv_unsigneds,
                                                                  recv_unsigneds,
                                                                  counter_for_recv_doubles,
                                                                  recv_doubles);
    }
    else
    {
      // Need to check which storage we should copy this halo node from
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
      oomph_info << "Rec:" << counter_for_recv_unsigneds
                 << "  Existing external halo node was found externally (0) or internally (1): "
                 << recv_unsigneds[counter_for_recv_unsigneds]
                 << std::endl;
#endif
      unsigned node_found_internally = recv_unsigneds[counter_for_recv_unsigneds++];
      if (node_found_internally)
      {
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
        oomph_info
            << "Rec:" << counter_for_recv_unsigneds
            << "  index of existing (internal) halo master node "
            << recv_unsigneds[counter_for_recv_unsigneds]
            << std::endl;
#endif
        // Copy node from received location
        new_master_nod_pt = mesh_pt->shared_node_pt(loc_p, recv_unsigneds[counter_for_recv_unsigneds++]);
      }
      else
      {
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
        oomph_info
            << "Rec:" << counter_for_recv_unsigneds
            << "  index of existing external halo master node "
            << recv_unsigneds[counter_for_recv_unsigneds]
            << std::endl;
#endif
        // Copy node from received location
        new_master_nod_pt = mesh_pt->external_halo_node_pt(loc_p, recv_unsigneds[counter_for_recv_unsigneds++]);
      }
    }
  }

  //======start of construct_new_external_halo_master_node_helper===========
  /// Helper function which constructs a new external halo master node
  /// with the required information sent from the haloed process
  //========================================================================
  template <class EXT_ELEMENT>
  void Missing_masters_functions::construct_new_external_halo_master_node_helper(Node *&new_master_nod_pt, Node *&nod_pt, unsigned &loc_p,
                                                                                 Mesh *const &mesh_pt,
                                                                                 unsigned &counter_for_recv_unsigneds,
                                                                                 Vector<unsigned> &recv_unsigneds,
                                                                                 unsigned &counter_for_recv_doubles,
                                                                                 Vector<double> &recv_doubles)
  {
    // First three sent numbers are dimension, position type and nvalue
    // (to be used in Node constructors)
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
    oomph_info
        << "Rec:" << counter_for_recv_unsigneds
        << "  ndim for external halo master node "
        << recv_unsigneds[counter_for_recv_unsigneds]
        << std::endl;
#endif
    unsigned n_dim = recv_unsigneds[counter_for_recv_unsigneds++];
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
    oomph_info
        << "Rec:" << counter_for_recv_unsigneds
        << "  nposition type for external halo master node "
        << recv_unsigneds[counter_for_recv_unsigneds]
        << std::endl;
#endif
    unsigned n_position_type = recv_unsigneds
        [counter_for_recv_unsigneds++];
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
    oomph_info
        << "Rec:" << counter_for_recv_unsigneds
        << "  nvalue for external halo master node "
        << recv_unsigneds[counter_for_recv_unsigneds]
        << std::endl;
#endif
    unsigned n_value = recv_unsigneds
        [counter_for_recv_unsigneds++];
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
    oomph_info
        << "Rec:" << counter_for_recv_unsigneds
        << "  non-halo processor ID for external halo master node "
        << recv_unsigneds[counter_for_recv_unsigneds]
        << std::endl;
#endif
    unsigned non_halo_proc_ID = recv_unsigneds
        [counter_for_recv_unsigneds++];

    // If it's a solid node also receive the lagrangian dimension and pos type
    SolidNode *solid_nod_pt = dynamic_cast<SolidNode *>(nod_pt);
    unsigned n_lag_dim;
    unsigned n_lag_type;
    if (solid_nod_pt != 0)
    {
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
      oomph_info
          << "Rec:" << counter_for_recv_unsigneds
          << "  nlagrdim for external halo master solid node "
          << recv_unsigneds[counter_for_recv_unsigneds]
          << std::endl;
#endif
      n_lag_dim = recv_unsigneds[counter_for_recv_unsigneds++];
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
      oomph_info
          << "Rec:" << counter_for_recv_unsigneds
          << "  nlagrtype for external halo master solid node "
          << recv_unsigneds[counter_for_recv_unsigneds]
          << std::endl;
#endif
      n_lag_type = recv_unsigneds[counter_for_recv_unsigneds++];
    }

    // Null TimeStepper for now
    TimeStepper *time_stepper_pt = 0;
    // Default number of previous values to 1
    unsigned n_prev = 1;

    // Just take timestepper from a node
    // Let's use first node of first element since this must exist
    time_stepper_pt = mesh_pt->finite_element_pt(0)->node_pt(0)->time_stepper_pt();

    // Is the node for which the master is required Algebraic, Macro or Solid?
    AlgebraicNode *alg_nod_pt = dynamic_cast<AlgebraicNode *>(nod_pt);
    MacroElementNodeUpdateNode *macro_nod_pt =
        dynamic_cast<MacroElementNodeUpdateNode *>(nod_pt);

    // What type of node was the node for which we are constructing a master?
    if (alg_nod_pt != 0)
    {
      // The master node should also be algebraic
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
      oomph_info
          << "Rec:" << counter_for_recv_unsigneds
          << "  Boolean for algebraic boundary node "
          << recv_unsigneds[counter_for_recv_unsigneds]
          << std::endl;
#endif
      // If this master node's haloed copy is on a boundary then
      // it needs to be on the same boundary here
      if (recv_unsigneds[counter_for_recv_unsigneds++] == 1)
      {
        // Create a new BoundaryNode (not attached to an element)
        if (time_stepper_pt != 0)
        {
          new_master_nod_pt = new BoundaryNode<AlgebraicNode>(time_stepper_pt, n_dim, n_position_type, n_value);
        }
        else
        {
          new_master_nod_pt = new BoundaryNode<AlgebraicNode>(n_dim, n_position_type, n_value);
        }

        // How many boundaries does the algebraic master node live on?
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
        oomph_info << "Rec:" << counter_for_recv_unsigneds
                   << " Number of boundaries the algebraic master node is on: "
                   << recv_unsigneds[counter_for_recv_unsigneds]
                   << std::endl;
#endif
        unsigned nb = recv_unsigneds[counter_for_recv_unsigneds++];
        for (unsigned i = 0; i < nb; i++)
        {
          // Boundary number
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
          oomph_info << "Rec:" << counter_for_recv_unsigneds
                     << "  Algebraic master node is on boundary "
                     << recv_unsigneds[counter_for_recv_unsigneds]
                     << std::endl;
#endif
          unsigned i_bnd =
              recv_unsigneds[counter_for_recv_unsigneds++];
          mesh_pt->add_boundary_node(i_bnd, new_master_nod_pt);
        }

        // Do we have additional values created by face elements?
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
        oomph_info
            << "Rec:" << counter_for_recv_unsigneds << " "
            << "Number of additional values created by face element "
            << "for master node "
            << recv_unsigneds[counter_for_recv_unsigneds]
            << std::endl;
#endif
        unsigned n_entry = recv_unsigneds[counter_for_recv_unsigneds++];
        if (n_entry > 0)
        {
          // Create storage, if it doesn't already exist, for the map
          // that will contain the position of the first entry of
          // this face element's additional values,
          BoundaryNodeBase *bnew_master_nod_pt =
              dynamic_cast<BoundaryNodeBase *>(new_master_nod_pt);
#ifdef PARANOID
          if (bnew_master_nod_pt == 0)
          {
            throw OomphLibError(
                "Failed to cast new node to boundary node\n",
                OOMPH_CURRENT_FUNCTION,
                OOMPH_EXCEPTION_LOCATION);
          }
#endif
          if (bnew_master_nod_pt->index_of_first_value_assigned_by_face_element_pt() == 0)
          {
            bnew_master_nod_pt->index_of_first_value_assigned_by_face_element_pt() =
                new std::map<unsigned, unsigned>;
          }

          // Get pointer to the map of indices associated with
          // additional values created by face elements
          std::map<unsigned, unsigned> *map_pt =
              bnew_master_nod_pt->index_of_first_value_assigned_by_face_element_pt();

          // Loop over number of entries in map
          for (unsigned i = 0; i < n_entry; i++)
          {
            // Read out pairs...

#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
            oomph_info << "Rec:" << counter_for_recv_unsigneds
                       << " Key of map entry for master node"
                       << recv_unsigneds[counter_for_recv_unsigneds]
                       << std::endl;
#endif
            unsigned first = recv_unsigneds[counter_for_recv_unsigneds++];

#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
            oomph_info << "Rec:" << counter_for_recv_unsigneds
                       << " Value of map entry for master node"
                       << recv_unsigneds[counter_for_recv_unsigneds]
                       << std::endl;
#endif
            unsigned second = recv_unsigneds[counter_for_recv_unsigneds++];

            // ...and assign
            (*map_pt)[first] = second;
          }
        }
      }
      else
      {
        // Create node (not attached to any element)
        if (time_stepper_pt != 0)
        {
          new_master_nod_pt = new AlgebraicNode(time_stepper_pt, n_dim, n_position_type, n_value);
        }
        else
        {
          new_master_nod_pt = new AlgebraicNode(n_dim, n_position_type, n_value);
        }
      }

      // Add this as an external halo node BEFORE considering node update!
      mesh_pt->add_external_halo_node_pt(loc_p, new_master_nod_pt);

      // The external mesh is itself Algebraic...
      AlgebraicMesh *alg_mesh_pt = dynamic_cast<AlgebraicMesh *>(mesh_pt);

#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
      oomph_info
          << "Rec:" << counter_for_recv_unsigneds
          << "  algebraic node update id for master node "
          << recv_unsigneds[counter_for_recv_unsigneds]
          << std::endl;
#endif
      /// The first entry of All_unsigned_values is the default node update id
      unsigned update_id = recv_unsigneds
          [counter_for_recv_unsigneds++];

      // Setup algebraic node update info for this new node
      Vector<double> ref_value;

#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
      oomph_info
          << "Rec:" << counter_for_recv_unsigneds
          << "  algebraic node number of ref values for master node "
          << recv_unsigneds[counter_for_recv_unsigneds]
          << std::endl;
#endif
      // The size of this vector is in the next entry
      unsigned n_ref_val = recv_unsigneds
          [counter_for_recv_unsigneds++];

      // The reference values are in the subsequent entries of All_double_values
      ref_value.resize(n_ref_val);
      for (unsigned i_ref = 0; i_ref < n_ref_val; i_ref++)
      {
        ref_value[i_ref] = recv_doubles
            [counter_for_recv_doubles++];
      }

      // Also require a Vector of geometric objects
      Vector<GeomObject *> geom_object_pt;

#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
      oomph_info
          << "Rec:" << counter_for_recv_unsigneds
          << "  algebraic node number of geom objects for master node "
          << recv_unsigneds[counter_for_recv_unsigneds]
          << std::endl;
#endif

      // The size of this vector is in the next entry of All_unsigned_values
      unsigned n_geom_obj = recv_unsigneds
          [counter_for_recv_unsigneds++];

      // The remaining indices are in the rest of
      // All_alg_nodal_info
      geom_object_pt.resize(n_geom_obj);
      for (unsigned i_geom = 0; i_geom < n_geom_obj; i_geom++)
      {
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
        oomph_info
            << "Rec:" << counter_for_recv_unsigneds
            << "  algebraic node: " << i_geom << "-th out of "
            << n_geom_obj << "-th geom index "
            << recv_unsigneds[counter_for_recv_unsigneds]
            << std::endl;
#endif
        unsigned geom_index = recv_unsigneds
            [counter_for_recv_unsigneds++];

        // This index indicates which (if any) of the AlgebraicMesh's
        // stored geometric objects should be used
        geom_object_pt[i_geom] = alg_mesh_pt->geom_object_list_pt(geom_index);
      }

      AlgebraicNode *alg_master_nod_pt =
          dynamic_cast<AlgebraicNode *>(new_master_nod_pt);

      /// ... so for the specified update_id, call
      /// add_node_update_info
      alg_master_nod_pt->add_node_update_info(update_id, alg_mesh_pt, geom_object_pt, ref_value);

      /// Now call update_node_update
      alg_mesh_pt->update_node_update(alg_master_nod_pt);
    }
    else if (macro_nod_pt != 0)
    {
      // The master node should also be a macro node
      // If this master node's haloed copy is on a boundary then
      // it needs to be on the same boundary here
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
      oomph_info << "Rec:" << counter_for_recv_unsigneds
                 << "  Boolean for master algebraic node is boundary node "
                 << recv_unsigneds[counter_for_recv_unsigneds]
                 << std::endl;
#endif
      if (recv_unsigneds[counter_for_recv_unsigneds++] == 1)
      {
        // Create a new BoundaryNode (not attached to an element)
        if (time_stepper_pt != 0)
        {
          new_master_nod_pt = new BoundaryNode<MacroElementNodeUpdateNode>(time_stepper_pt, n_dim, n_position_type, n_value);
        }
        else
        {
          new_master_nod_pt = new BoundaryNode<MacroElementNodeUpdateNode>(n_dim, n_position_type, n_value);
        }

        // How many boundaries does the macro element master node live on?
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
        oomph_info
            << "Rec:" << counter_for_recv_unsigneds
            << " Number of boundaries the macro element master node is on: "
            << recv_unsigneds[counter_for_recv_unsigneds]
            << std::endl;
#endif
        unsigned nb = recv_unsigneds[counter_for_recv_unsigneds++];
        for (unsigned i = 0; i < nb; i++)
        {
          // Boundary number
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
          oomph_info << "Rec:" << counter_for_recv_unsigneds
                     << "  Macro element master node is on boundary "
                     << recv_unsigneds[counter_for_recv_unsigneds]
                     << std::endl;
#endif
          unsigned i_bnd =
              recv_unsigneds[counter_for_recv_unsigneds++];
          mesh_pt->add_boundary_node(i_bnd, new_master_nod_pt);
        }

        // Do we have additional values created by face elements?
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
        oomph_info
            << "Rec:" << counter_for_recv_unsigneds
            << " Number of additional values created by face element "
            << "for macro element master node "
            << recv_unsigneds[counter_for_recv_unsigneds]
            << std::endl;
#endif
        unsigned n_entry = recv_unsigneds[counter_for_recv_unsigneds++];
        if (n_entry > 0)
        {
          // Create storage, if it doesn't already exist, for the map
          // that will contain the position of the first entry of
          // this face element's additional values,
          BoundaryNodeBase *bnew_master_nod_pt =
              dynamic_cast<BoundaryNodeBase *>(new_master_nod_pt);
#ifdef PARANOID
          if (bnew_master_nod_pt == 0)
          {
            throw OomphLibError(
                "Failed to cast new node to boundary node\n",
                OOMPH_CURRENT_FUNCTION,
                OOMPH_EXCEPTION_LOCATION);
          }
#endif
          if (bnew_master_nod_pt->index_of_first_value_assigned_by_face_element_pt() == 0)
          {
            bnew_master_nod_pt->index_of_first_value_assigned_by_face_element_pt() =
                new std::map<unsigned, unsigned>;
          }

          // Get pointer to the map of indices associated with
          // additional values created by face elements
          std::map<unsigned, unsigned> *map_pt =
              bnew_master_nod_pt->index_of_first_value_assigned_by_face_element_pt();

          // Loop over number of entries in map
          for (unsigned i = 0; i < n_entry; i++)
          {
            // Read out pairs...

#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
            oomph_info << "Rec:" << counter_for_recv_unsigneds
                       << " Key of map entry for macro element master node"
                       << recv_unsigneds[counter_for_recv_unsigneds]
                       << std::endl;
#endif
            unsigned first = recv_unsigneds[counter_for_recv_unsigneds++];

#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
            oomph_info << "Rec:" << counter_for_recv_unsigneds
                       << " Value of map entry for macro element master node"
                       << recv_unsigneds[counter_for_recv_unsigneds]
                       << std::endl;
#endif
            unsigned second = recv_unsigneds[counter_for_recv_unsigneds++];

            // ...and assign
            (*map_pt)[first] = second;
          }
        }
      }
      else
      {
        // Create node (not attached to any element)
        if (time_stepper_pt != 0)
        {
          new_master_nod_pt = new MacroElementNodeUpdateNode(time_stepper_pt, n_dim, n_position_type, n_value);
        }
        else
        {
          new_master_nod_pt = new MacroElementNodeUpdateNode(n_dim, n_position_type, n_value);
        }
      }

      // Add this as an external halo node
      mesh_pt->add_external_halo_node_pt(loc_p, new_master_nod_pt);
      oomph_info << "Added external halo master node:" << new_master_nod_pt << " at [ " << new_master_nod_pt->x(0) << ", " << new_master_nod_pt->x(1) << " ]" << std::endl;

      // Create a new node update element for this master node if required
      FiniteElement *new_node_update_f_el_pt = 0;
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
      oomph_info << "Rec:" << counter_for_recv_unsigneds
                 << "  Bool: need new external halo element "
                 << recv_unsigneds[counter_for_recv_unsigneds]
                 << std::endl;
#endif
      if (recv_unsigneds[counter_for_recv_unsigneds++] == 1)
      {
        // Issue warning about adding a macro element to the external storage
        std::ostringstream warn_stream;
        warn_stream << "You are adding a MacroElementNodeUpdate element to the\n"
                    << "external storage. This functionality is still being\n"
                    << "developed and may cause problems later on, say during\n"
                    << "Problem::remove_duplicate_data().";
        OomphLibWarning(
            warn_stream.str(),
            "Missing_masters_functions::construct_new_external_halo_master_node_helper()",
            OOMPH_EXCEPTION_LOCATION);

        //      GeneralisedElement* new_node_update_el_pt = new EXT_ELEMENT;
        GeneralisedElement *new_node_update_el_pt = NULL;
        throw_runtime_error("IMPLEMENT");

        // Add external halo element to this mesh
        mesh_pt->add_external_halo_element_pt(
            loc_p, new_node_update_el_pt);

        // Cast to finite element
        new_node_update_f_el_pt =
            dynamic_cast<FiniteElement *>(new_node_update_el_pt);

        // Need number of interpolated values if Refineable
        int n_cont_inter_values;
        if (dynamic_cast<RefineableElement *>(new_node_update_f_el_pt) != 0)
        {
          n_cont_inter_values = dynamic_cast<RefineableElement *>(new_node_update_f_el_pt)->ncont_interpolated_values();
        }
        else
        {
          n_cont_inter_values = -1;
        }
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
        oomph_info << "Rec:" << counter_for_recv_unsigneds
                   << "  Bool: we have a macro element mesh "
                   << recv_unsigneds[counter_for_recv_unsigneds]
                   << std::endl;
#endif
        // If we're using macro elements to update,
        if (recv_unsigneds[counter_for_recv_unsigneds++] == 1)
        {
          // Set the macro element
          MacroElementNodeUpdateMesh *macro_mesh_pt =
              dynamic_cast<MacroElementNodeUpdateMesh *>(mesh_pt);

#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
          oomph_info << "Rec:" << counter_for_recv_unsigneds
                     << "  Number of macro element "
                     << recv_unsigneds[counter_for_recv_unsigneds]
                     << std::endl;
#endif
          unsigned macro_el_num =
              recv_unsigneds[counter_for_recv_unsigneds++];
          new_node_update_f_el_pt->set_macro_elem_pt(macro_mesh_pt->macro_domain_pt()->macro_element_pt(macro_el_num));

          // we need to receive
          // the lower left and upper right coordinates of the macro
          QElementBase *q_el_pt =
              dynamic_cast<QElementBase *>(new_node_update_f_el_pt);
          if (q_el_pt != 0)
          {
            unsigned el_dim = q_el_pt->dim();
            for (unsigned i_dim = 0; i_dim < el_dim; i_dim++)
            {
              q_el_pt->s_macro_ll(i_dim) = recv_doubles
                  [counter_for_recv_doubles++];
              q_el_pt->s_macro_ur(i_dim) = recv_doubles
                  [counter_for_recv_doubles++];
            }
          }
          else // Throw an error
          {
            std::ostringstream error_stream;
            error_stream << "You are using a MacroElement node update\n"
                         << "in a case with non-QElements. This has not\n"
                         << "yet been implemented.\n";
            throw OomphLibError(error_stream.str(),
                                OOMPH_CURRENT_FUNCTION,
                                OOMPH_EXCEPTION_LOCATION);
          }
        }

        // Check if haloed version was p-refineable
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
        oomph_info << "Rec:" << counter_for_recv_unsigneds
                   << "  Element was p-refineable "
                   << recv_unsigneds[counter_for_recv_unsigneds]
                   << std::endl;
#endif
        unsigned el_was_p_refineable =
            recv_unsigneds[counter_for_recv_unsigneds++];
        if (el_was_p_refineable)
        {
          // Check created element is p-refineable
          PRefineableElement *p_refineable_el_pt =
              dynamic_cast<PRefineableElement *>(new_node_update_f_el_pt);
          if (p_refineable_el_pt != 0)
          {
            // Recieve p-order
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
            oomph_info << "Rec:" << counter_for_recv_unsigneds
                       << "  p-order: "
                       << recv_unsigneds[counter_for_recv_unsigneds]
                       << std::endl;
#endif
            unsigned p_order =
                recv_unsigneds[counter_for_recv_unsigneds++];

            // Do initial setup with original element as the clone's adopted father
            p_refineable_el_pt->initial_setup(0, p_order);
            // BENFLAG:
            oomph_info << "New node update element: " << new_node_update_el_pt << " (p-order = " << p_order << ")" << std::endl;
          }
          else
          {
            std::ostringstream error_stream;
            error_stream << "Created MacroElement node update element is not p-refineable\n"
                         << "but the haloed version is.\n";
            throw OomphLibError(error_stream.str(),
                                "Missing_masters_functions::construct_new_external_halo_master_...()",
                                OOMPH_EXCEPTION_LOCATION);
          }
        }

        unsigned n_node = new_node_update_f_el_pt->nnode();
        for (unsigned j = 0; j < n_node; j++)
        {
          Node *new_nod_pt = 0;
          add_external_halo_node_to_storage<EXT_ELEMENT>(new_nod_pt, mesh_pt, loc_p, j, new_node_update_f_el_pt,
                                                         n_cont_inter_values,
                                                         counter_for_recv_unsigneds,
                                                         recv_unsigneds,
                                                         counter_for_recv_doubles,
                                                         recv_doubles);
          // BENFLAG:
          oomph_info << "Added node " << new_nod_pt << " at [ " << new_nod_pt->x(0) << ", " << new_nod_pt->x(1) << " ]" << std::endl;
        }

        // BENFLAG:
        oomph_info << "New node update element: " << new_node_update_f_el_pt << " (nnode_1d = " << new_node_update_f_el_pt->nnode_1d() << ")" << std::endl;
      }
      else // The node update element exists already
      {
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
        oomph_info << "Rec:" << counter_for_recv_unsigneds
                   << "  Found internally? "
                   << recv_unsigneds[counter_for_recv_unsigneds]
                   << std::endl;
#endif
        unsigned found_internally = recv_unsigneds[counter_for_recv_unsigneds++];
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
        oomph_info << "Rec:" << counter_for_recv_unsigneds
                   << "  Number of already existing external halo element "
                   << recv_unsigneds[counter_for_recv_unsigneds]
                   << std::endl;
#endif
        unsigned halo_element_index = recv_unsigneds[counter_for_recv_unsigneds++];
        if (found_internally != 0)
        {
          new_node_update_f_el_pt = dynamic_cast<FiniteElement *>(
              (mesh_pt->halo_element_pt(loc_p))[halo_element_index]);
          // BENFLAG:
          oomph_info << "Existing node update element: " << new_node_update_f_el_pt << " (nnode_1d = " << new_node_update_f_el_pt->nnode_1d() << ")" << std::endl;
          oomph_info << "on proc " << loc_p << " at (internal) index " << halo_element_index << std::endl;

          //        //BENFLAG: Also add halo element to external storage
          //        oomph_info << "Adding to external halo storage..." << std::endl;
          //        GeneralisedElement* g_el_pt = dynamic_cast<GeneralisedElement*>(new_node_update_f_el_pt);
          //        mesh_pt->add_external_halo_element_pt(
          //         loc_p,g_el_pt);
          //
          //        // Check if also found externally
          // #ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
          //        oomph_info << "Rec:" << counter_for_recv_unsigneds
          //                   << "  Found externally too? "
          //                   << recv_unsigneds[counter_for_recv_unsigneds]
          //                   << std::endl;
          // #endif
          //        unsigned found_externally_too = recv_unsigneds[counter_for_recv_unsigneds++];
          //        std::cout << "received found_externally_too = " << found_externally_too << std::endl;
          //        if(found_externally_too==1234)
          //         {
          // #ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
          //          oomph_info << "Rec:" << counter_for_recv_unsigneds
          //                     << "  Number of already existing external halo element "
          //                     << recv_unsigneds[counter_for_recv_unsigneds]
          //                     << std::endl;
          // #endif
          //          unsigned ext_version_halo_element_index = recv_unsigneds[counter_for_recv_unsigneds++];
          //          std::cout << "received ext_version_halo_element_index = " << ext_version_halo_element_index << std::endl;
          //
          //          FiniteElement* ext_version_pt = dynamic_cast<FiniteElement*>(
          //           (mesh_pt->halo_element_pt(loc_p))[ext_version_halo_element_index]);
          //          //BENFLAG:
          //          oomph_info << "Existing node update element: " << ext_version_pt << " (nnode_1d = " << ext_version_pt->nnode_1d() << ")" << std::endl;
          //          oomph_info << "on proc " << loc_p << " is also at (external) index " << ext_version_halo_element_index << std::endl;
          //          for(unsigned j=0; j<ext_version_pt->nnode(); j++)
          //           {
          //            oomph_info << ext_version_pt->node_pt(j) << " at [ " << ext_version_pt->node_pt(j)->x(0) << ", " << ext_version_pt->node_pt(j)->x(1) << " ]" << std::endl;
          //           }
          //         }
        }
        else
        {
          new_node_update_f_el_pt = dynamic_cast<FiniteElement *>(
              mesh_pt->external_halo_element_pt(loc_p, halo_element_index));
          // BENFLAG:
          oomph_info << "Existing node update element: " << new_node_update_f_el_pt << " (nnode_1d = " << new_node_update_f_el_pt->nnode_1d() << ")" << std::endl;
          oomph_info << "on proc " << loc_p << " at (external) index " << recv_unsigneds[counter_for_recv_unsigneds - 1] << std::endl;
          // oomph_info << "...and doesn't exist in the external storage." << std::endl;
        }
      }

      // Remaining required information to create functioning
      // MacroElementNodeUpdateNode...

      // Get the required geom objects for the node update
      // from the mesh
      Vector<GeomObject *> geom_object_vector_pt;
      MacroElementNodeUpdateMesh *macro_mesh_pt =
          dynamic_cast<MacroElementNodeUpdateMesh *>(mesh_pt);
      geom_object_vector_pt = macro_mesh_pt->geom_object_vector_pt();

      // Cast to MacroElementNodeUpdateNode
      MacroElementNodeUpdateNode *macro_master_nod_pt =
          dynamic_cast<MacroElementNodeUpdateNode *>(new_master_nod_pt);

      // Set all required information - node update element,
      // local coordinate in this element, and then set node update info
      macro_master_nod_pt->node_update_element_pt() =
          new_node_update_f_el_pt;

      ////print out nodes
      // std::cout << "nodes are:" << std::endl;
      // for(unsigned j=0; j<new_node_update_f_el_pt->nnode(); j++)
      //  {
      //   std::cout << new_node_update_f_el_pt->node_pt(j) << " at [ " << new_node_update_f_el_pt->node_pt(j)->x(0) << ", " << new_node_update_f_el_pt->node_pt(j)->x(1) << " ]" << std::endl;
      //   //std::cout << new_node_update_f_el_pt->node_pt(j) << std::endl;
      //  }
      // std::cout << "should include: " << macro_master_nod_pt << " at [ " << macro_master_nod_pt->x(0) << ", " << macro_master_nod_pt->x(1) << " ]" << std::endl;

      // Need to get the local node index of the macro_master_nod_pt
      unsigned local_node_index = 0;
      // std::cout << "before: " << local_node_index << std::endl;
      unsigned n_node = new_node_update_f_el_pt->nnode();
      for (unsigned j = 0; j < n_node; j++)
      {
        if (macro_master_nod_pt == new_node_update_f_el_pt->node_pt(j))
        {
          // std::cout << "Node " << macro_master_nod_pt << " found at index " << j << " in update element." << std::endl;
          local_node_index = j;
          break;
        }
        // BENFLAG:
        if (j == n_node - 1)
        {
          //// Check if sons...
          // RefineableElement* ref_el_pt = dynamic_cast<RefineableElement*>(new_node_update_f_el_pt);
          // if(ref_el_pt->tree_pt()->nsons()!=0)
          //  {
          //   std::cout << "update el has sons!" << std::endl;
          //  }
          // else
          //  {
          //   std::cout << "No sons." << std::endl;
          //  }

          // oomph_info << "Node not found in update element!" << std::endl;
          throw OomphLibError(
              "Node not found in update element!",
              "Missing_masters_functions::construct_new_external_halo_master_node_helper()",
              OOMPH_EXCEPTION_LOCATION);
        }
      }
      // std::cout << "after: " << local_node_index << std::endl;

      Vector<double> s_in_macro_node_update_element;
      new_node_update_f_el_pt->local_coordinate_of_node(local_node_index, s_in_macro_node_update_element);

      macro_master_nod_pt->set_node_update_info(new_node_update_f_el_pt, s_in_macro_node_update_element,
                                                geom_object_vector_pt);
    }
    else if (solid_nod_pt != 0)
    {
      // The master node should also be a SolidNode
      // If this node was on a boundary then it needs to
      // be on the same boundary here
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
      oomph_info << "Rec:" << counter_for_recv_unsigneds
                 << "  Bool master is a boundary (solid) node "
                 << recv_unsigneds[counter_for_recv_unsigneds]
                 << std::endl;
#endif
      if (recv_unsigneds[counter_for_recv_unsigneds++] == 1)
      {
        // Construct a new boundary node
        if (time_stepper_pt != 0)
        {
          new_master_nod_pt = new BoundaryNode<SolidNode>(time_stepper_pt, n_lag_dim, n_lag_type, n_dim, n_position_type, n_value);
        }
        else
        {
          new_master_nod_pt = new BoundaryNode<SolidNode>(n_lag_dim, n_lag_type, n_dim, n_position_type, n_value);
        }

        // How many boundaries does the macro element master node live on?
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
        oomph_info
            << "Rec:" << counter_for_recv_unsigneds
            << " Number of boundaries the solid master node is on: "
            << recv_unsigneds[counter_for_recv_unsigneds]
            << std::endl;
#endif
        unsigned nb = recv_unsigneds[counter_for_recv_unsigneds++];
        for (unsigned i = 0; i < nb; i++)
        {
          // Boundary number
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
          oomph_info << "Rec:" << counter_for_recv_unsigneds
                     << " Solid master node is on boundary "
                     << recv_unsigneds[counter_for_recv_unsigneds]
                     << std::endl;
#endif
          unsigned i_bnd =
              recv_unsigneds[counter_for_recv_unsigneds++];
          mesh_pt->add_boundary_node(i_bnd, new_master_nod_pt);
        }

        // Do we have additional values created by face elements?
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
        oomph_info
            << "Rec:" << counter_for_recv_unsigneds
            << " Number of additional values created by face element "
            << "for solid master node "
            << recv_unsigneds[counter_for_recv_unsigneds]
            << std::endl;
#endif
        unsigned n_entry = recv_unsigneds[counter_for_recv_unsigneds++];
        if (n_entry > 0)
        {
          // Create storage, if it doesn't already exist, for the map
          // that will contain the position of the first entry of
          // this face element's additional values,
          BoundaryNodeBase *bnew_master_nod_pt =
              dynamic_cast<BoundaryNodeBase *>(new_master_nod_pt);
#ifdef PARANOID
          if (bnew_master_nod_pt == 0)
          {
            throw OomphLibError(
                "Failed to cast new node to boundary node\n",
                OOMPH_CURRENT_FUNCTION,
                OOMPH_EXCEPTION_LOCATION);
          }
#endif
          if (bnew_master_nod_pt->index_of_first_value_assigned_by_face_element_pt() == 0)
          {
            bnew_master_nod_pt->index_of_first_value_assigned_by_face_element_pt() =
                new std::map<unsigned, unsigned>;
          }

          // Get pointer to the map of indices associated with
          // additional values created by face elements
          std::map<unsigned, unsigned> *map_pt =
              bnew_master_nod_pt->index_of_first_value_assigned_by_face_element_pt();

          // Loop over number of entries in map
          for (unsigned i = 0; i < n_entry; i++)
          {
            // Read out pairs...

#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
            oomph_info << "Rec:" << counter_for_recv_unsigneds
                       << " Key of map entry for solid master node"
                       << recv_unsigneds[counter_for_recv_unsigneds]
                       << std::endl;
#endif
            unsigned first = recv_unsigneds[counter_for_recv_unsigneds++];

#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
            oomph_info << "Rec:" << counter_for_recv_unsigneds
                       << " Value of map entry for solid master node"
                       << recv_unsigneds[counter_for_recv_unsigneds]
                       << std::endl;
#endif
            unsigned second = recv_unsigneds[counter_for_recv_unsigneds++];

            // ...and assign
            (*map_pt)[first] = second;
          }
        }
      }
      else
      {
        // Construct an ordinary (non-boundary) node
        if (time_stepper_pt != 0)
        {
          new_master_nod_pt = new SolidNode(time_stepper_pt, n_lag_dim, n_lag_type, n_dim, n_position_type, n_value);
        }
        else
        {
          new_master_nod_pt = new SolidNode(n_lag_dim, n_lag_type, n_dim, n_position_type, n_value);
        }
      }

      // Add this as an external halo node
      mesh_pt->add_external_halo_node_pt(loc_p, new_master_nod_pt);

      // Copy across particular info required for SolidNode
      // NOTE: Are there any problems with additional values for SolidNodes?
      SolidNode *solid_master_nod_pt = dynamic_cast<SolidNode *>(new_master_nod_pt);
      unsigned n_solid_val = solid_master_nod_pt->variable_position_pt()->nvalue();
      for (unsigned i_val = 0; i_val < n_solid_val; i_val++)
      {
        for (unsigned t = 0; t < n_prev; t++)
        {
          solid_master_nod_pt->variable_position_pt()->set_value(t, i_val,
                                                                 recv_doubles[counter_for_recv_doubles++]);
        }
      }
    }
    else // Just an ordinary node!
    {
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
      oomph_info << "Rec:" << counter_for_recv_unsigneds
                 << "  Bool node is on boundary "
                 << recv_unsigneds[counter_for_recv_unsigneds]
                 << std::endl;
#endif

      // If this node was on a boundary then it needs to
      // be on the same boundary here
      if (recv_unsigneds[counter_for_recv_unsigneds++] == 1)
      {
        // Construct a new boundary node
        if (time_stepper_pt != 0)
        {
          new_master_nod_pt = new BoundaryNode<Node>(time_stepper_pt, n_dim, n_position_type, n_value);
        }
        else
        {
          new_master_nod_pt = new BoundaryNode<Node>(n_dim, n_position_type, n_value);
        }

        // How many boundaries does the master node live on?
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
        oomph_info << "Rec:" << counter_for_recv_unsigneds
                   << " Number of boundaries the master node is on: "
                   << recv_unsigneds[counter_for_recv_unsigneds]
                   << std::endl;
#endif
        unsigned nb = recv_unsigneds[counter_for_recv_unsigneds++];
        for (unsigned i = 0; i < nb; i++)
        {
          // Boundary number
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
          oomph_info << "Rec:" << counter_for_recv_unsigneds
                     << "  Master node is on boundary "
                     << recv_unsigneds[counter_for_recv_unsigneds]
                     << std::endl;
#endif
          unsigned i_bnd =
              recv_unsigneds[counter_for_recv_unsigneds++];
          mesh_pt->add_boundary_node(i_bnd, new_master_nod_pt);
        }

        // Do we have additional values created by face elements?
#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
        oomph_info
            << "Rec:" << counter_for_recv_unsigneds
            << " Number of additional values created by face element "
            << "for master node "
            << recv_unsigneds[counter_for_recv_unsigneds]
            << std::endl;
#endif
        unsigned n_entry = recv_unsigneds[counter_for_recv_unsigneds++];
        if (n_entry > 0)
        {
          // Create storage, if it doesn't already exist, for the map
          // that will contain the position of the first entry of
          // this face element's additional values,
          BoundaryNodeBase *bnew_master_nod_pt =
              dynamic_cast<BoundaryNodeBase *>(new_master_nod_pt);
#ifdef PARANOID
          if (bnew_master_nod_pt == 0)
          {
            throw OomphLibError(
                "Failed to cast new node to boundary node\n",
                OOMPH_CURRENT_FUNCTION,
                OOMPH_EXCEPTION_LOCATION);
          }
#endif
          if (bnew_master_nod_pt->index_of_first_value_assigned_by_face_element_pt() == 0)
          {
            bnew_master_nod_pt->index_of_first_value_assigned_by_face_element_pt() =
                new std::map<unsigned, unsigned>;
          }

          // Get pointer to the map of indices associated with
          // additional values created by face elements
          std::map<unsigned, unsigned> *map_pt =
              bnew_master_nod_pt->index_of_first_value_assigned_by_face_element_pt();

          // Loop over number of entries in map
          for (unsigned i = 0; i < n_entry; i++)
          {
            // Read out pairs...

#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
            oomph_info << "Rec:" << counter_for_recv_unsigneds
                       << " Key of map entry for master node"
                       << recv_unsigneds[counter_for_recv_unsigneds]
                       << std::endl;
#endif
            unsigned first = recv_unsigneds[counter_for_recv_unsigneds++];

#ifdef ANNOTATE_MISSING_MASTERS_COMMUNICATION
            oomph_info << "Rec:" << counter_for_recv_unsigneds
                       << " Value of map entry for master node"
                       << recv_unsigneds[counter_for_recv_unsigneds]
                       << std::endl;
#endif
            unsigned second = recv_unsigneds[counter_for_recv_unsigneds++];

            // ...and assign
            (*map_pt)[first] = second;
          }
        }
      }
      else
      {
        // Construct an ordinary (non-boundary) node
        if (time_stepper_pt != 0)
        {
          new_master_nod_pt = new Node(time_stepper_pt, n_dim, n_position_type, n_value);
        }
        else
        {
          new_master_nod_pt = new Node(n_dim, n_position_type, n_value);
        }
      }

      // Add this as an external halo node
      mesh_pt->add_external_halo_node_pt(loc_p, new_master_nod_pt);
    }

    // Remaining info received for all node types
    // Get copied history values
    //  unsigned n_val=new_master_nod_pt->nvalue();
    for (unsigned i_val = 0; i_val < n_value; i_val++)
    {
      for (unsigned t = 0; t < n_prev; t++)
      {
        new_master_nod_pt->set_value(t, i_val, recv_doubles[counter_for_recv_doubles++]);
      }
    }

    // Get copied history values for positions
    unsigned n_nod_dim = new_master_nod_pt->ndim();
    for (unsigned idim = 0; idim < n_nod_dim; idim++)
    {
      for (unsigned t = 0; t < n_prev; t++)
      {
        // Copy to coordinate
        new_master_nod_pt->x(t, idim) = recv_doubles
            [counter_for_recv_doubles++];
      }
    }

    // Assign correct non-halo processor ID
    new_master_nod_pt->set_halo(non_halo_proc_ID);
  }

#endif

}
