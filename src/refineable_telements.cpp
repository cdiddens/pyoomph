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

#include "refineable_telements.hpp"
#include "exception.hpp"
namespace oomph
{

  //==================================================================
  /// Setup static matrix for coincidence between son nodal points and
  /// father boundaries:
  ///
  /// Father_boundd[nnode_1d](nnode_son,son_type)={SW/SE/NW/NE/S/E/N/W/OMEGA}
  ///
  /// so that node nnode_son in element of type son_type lies
  /// on boundary/vertex Father_boundd[nnode_1d](nnode_son,son_type) in its
  /// father element. If the node doesn't lie on a boundary
  /// the value is OMEGA.
  //==================================================================
  void RefineableTElement<1>::setup_father_bounds()
  {
    throw_runtime_error("Implement");
  }

  //==================================================================
  /// Determine Vector of boundary conditions along the element's boundary
  /// (or vertex) bound (S/W/N/E/SW/SE/NW/NE).
  ///
  /// This function assumes that the same boundary condition is applied
  /// along the entire length of an element's edge (of course, the
  /// vertices combine the boundary conditions of their two adjacent edges
  /// in the most restrictive combination. Hence, if we're at a vertex,
  /// we apply the most restrictive boundary condition of the
  /// two adjacent edges. If we're on an edge (in its proper interior),
  /// we apply the least restrictive boundary condition of all nodes
  /// along the edge.
  ///
  /// Usual convention:
  ///   - bound_cons[ival]=0 if value ival on this boundary is free
  ///   - bound_cons[ival]=1 if value ival on this boundary is pinned
  //==================================================================
  void RefineableTElement<1>::get_bcs(int bound, Vector<int> &bound_cons) const
  {
    throw_runtime_error("Implement");
  }

  //==================================================================
  /// Determine Vector of boundary conditions along the element's
  /// edge (S/N/W/E) -- BC is the least restrictive combination
  /// of all the nodes on this edge
  ///
  /// Usual convention:
  ///   - bound_cons[ival]=0 if value ival on this boundary is free
  ///   - bound_cons[ival]=1 if value ival on this boundary is pinned
  //==================================================================
  void RefineableTElement<1>::get_edge_bcs(const int &edge, Vector<int> &bound_cons) const
  {
    throw_runtime_error("Implement");
  }

  //==================================================================
  /// Given an element edge/vertex, return a set that contains
  /// all the (mesh-)boundary numbers that this element edge/vertex
  /// lives on.
  ///
  /// For proper edges, the boundary is the one (if any) that is shared by
  /// both vertex nodes). For vertex nodes, we just return their
  /// boundaries.
  //==================================================================
  void RefineableTElement<1>::get_boundaries(const int &edge,
                                             std::set<unsigned> &boundary) const
  {
    throw_runtime_error("Implement");
  }

  //===================================================================
  /// Return the value of the intrinsic boundary coordinate interpolated
  /// along the edge (S/W/N/E)
  //===================================================================
  void RefineableTElement<1>::
      interpolated_zeta_on_edge(const unsigned &boundary,
                                const int &edge, const Vector<double> &s,
                                Vector<double> &zeta)
  {
    throw_runtime_error("Implement");
  }

  //===================================================================
  /// If a neighbouring element has already created a node at
  /// a position corresponding to the local fractional position within the
  /// present element, s_fraction, return
  /// a pointer to that node. If not, return NULL (0). If the node is
  /// periodic the flag is_periodic will be true
  //===================================================================
  Node *RefineableTElement<1>::
      node_created_by_neighbour(const Vector<double> &s_fraction,
                                bool &is_periodic)
  {
    throw_runtime_error("Implement");
    return 0;
  }

  //==================================================================
  /// Build the element by doing the following:
  /// - Give it nodal positions (by establishing the pointers to its
  ///   nodes)
  /// - In the process create new nodes where required (i.e. if they
  ///   don't exist in father element or have already been created
  ///   while building new neighbour elements). Node building
  ///   involves the following steps:
  ///   - Get nodal position from father element.
  ///   - Establish the time-history of the newly created nodal point
  ///     (its coordinates and the previous values) consistent with
  ///     the father's history.
  ///   - Determine the boundary conditions of the nodes (newly
  ///     created nodes can only lie on the interior of any
  ///     edges of the father element -- this makes it possible to
  ///     to figure out what their bc should be...)
  ///   - Add node to the mesh's stoarge scheme for the boundary nodes.
  ///   - Add the new node to the mesh itself
  ///   - Doc newly created nodes in "new_nodes.dat" stored in the directory
  ///     of the DocInfo object (only if it's open!)
  /// - Finally, excute the element-specific further_build()
  ///   (empty by default -- must be overloaded for specific elements).
  ///   This deals with any build operations that are not included
  ///   in the generic process outlined above. For instance, in
  ///   Crouzeix Raviart elements we need to initialise the internal
  ///   pressure values in manner consistent with the pressure
  ///   distribution in the father element.
  //==================================================================
  void RefineableTElement<1>::build(Mesh *&mesh_pt,
                                    Vector<Node *> &new_node_pt,
                                    bool &was_already_built,
                                    std::ofstream &new_nodes_file)
  {
    throw_runtime_error("Implement");
  }

  //====================================================================
  ///  Print corner nodes, use colour (default "BLACK")
  //====================================================================
  void RefineableTElement<1>::output_corners(std::ostream &outfile,
                                             const std::string &colour) const
  {
    throw_runtime_error("Implement");
  }

  //====================================================================
  /// Set up all hanging nodes. If we are documenting the output then
  /// open the output files and pass the open files to the helper function
  //====================================================================
  void RefineableTElement<1>::setup_hanging_nodes(Vector<std::ofstream *>
                                                      &output_stream)
  {
    throw_runtime_error("Implement");
  }

  //================================================================
  /// Internal function that sets up the hanging node scheme for
  /// a particular continuously interpolated value
  //===============================================================
  void RefineableTElement<1>::setup_hang_for_value(const int &value_id)
  {
    throw_runtime_error("Implement");
  }

  //=================================================================
  /// Internal function to set up the hanging nodes on a particular
  /// edge of the element
  //=================================================================
  void RefineableTElement<1>::
      quad_hang_helper(const int &value_id,
                       const int &my_edge, std::ofstream &output_hangfile)
  {
    throw_runtime_error("Implement");
  }

  //=================================================================
  /// Check inter-element continuity of
  /// - nodal positions
  /// - (nodally) interpolated function values
  //====================================================================
  // template<unsigned NNODE_1D>
  void RefineableTElement<1>::check_integrity(double &max_error)
  {

    throw_runtime_error("Implement");
  }

  //========================================================================
  /// Static matrix for coincidence between son nodal points and
  /// father boundaries
  ///
  //========================================================================
  std::map<unsigned, DenseMatrix<int>> RefineableTElement<1>::Father_bound;

  //==================================================================
  /// Setup static matrix for coincidence between son nodal points and
  /// father boundaries:
  ///
  /// Father_boundd[nnode_1d](nnode_son,son_type)={SW/SE/NW/NE/S/E/N/W/OMEGA}
  ///
  /// so that node nnode_son in element of type son_type lies
  /// on boundary/vertex Father_boundd[nnode_1d](nnode_son,son_type) in its
  /// father element. If the node doesn't lie on a boundary
  /// the value is OMEGA.
  //==================================================================
  void RefineableTElement<2>::setup_father_bounds()
  {
    using namespace QuadTreeNames;

    // Find the number of nodes along a 1D edge
    unsigned n_p = nnode_1d();
    unsigned nnode = this->nnode();
    // Allocate space for the boundary information
    if (nnode == 3)
    {
      Father_bound[n_p].resize(3, 4);
    }
    else if (nnode == 6)
    {
      Father_bound[n_p].resize(6, 4);
    }
    else
    {
      throw_runtime_error("Implement");
    }

    // Initialise: By default points are not on the boundary
    for (unsigned n = 0; n < nnode; n++)
    {
      for (unsigned ison = 0; ison < 4; ison++)
      {
        Father_bound[n_p](n, ison) = Tree::OMEGA;
      }
    }

    // Southwest son
    Father_bound[n_p](0, SW) = S;
    Father_bound[n_p](1, SW) = W;
    Father_bound[n_p](2, SW) = SW;
    if (nnode > 3)
    {
      Father_bound[n_p](4, SW) = W;
      Father_bound[n_p](5, SW) = S;
    }

    // Northwest son
    //--------------
    Father_bound[n_p](0, NW) = E;
    Father_bound[n_p](1, NW) = NW;
    Father_bound[n_p](2, NW) = W;
    if (nnode > 3)
    {
      Father_bound[n_p](3, NW) = E;
      Father_bound[n_p](4, NW) = W;
    }

    // Northeast son (actually the center)
    //--------------
    Father_bound[n_p](0, NE) = S;
    Father_bound[n_p](1, NE) = E;
    Father_bound[n_p](2, NE) = W;

    // Southeast son
    //--------------
    Father_bound[n_p](0, SE) = SE;
    Father_bound[n_p](1, SE) = E;
    Father_bound[n_p](2, SE) = S;
    if (nnode > 3)
    {
      Father_bound[n_p](3, SE) = E;
      Father_bound[n_p](5, SE) = S;
    }
  }

  //==================================================================
  /// Determine Vector of boundary conditions along the element's boundary
  /// (or vertex) bound (S/W/N/E/SW/SE/NW/NE).
  ///
  /// This function assumes that the same boundary condition is applied
  /// along the entire length of an element's edge (of course, the
  /// vertices combine the boundary conditions of their two adjacent edges
  /// in the most restrictive combination. Hence, if we're at a vertex,
  /// we apply the most restrictive boundary condition of the
  /// two adjacent edges. If we're on an edge (in its proper interior),
  /// we apply the least restrictive boundary condition of all nodes
  /// along the edge.
  ///
  /// Usual convention:
  ///   - bound_cons[ival]=0 if value ival on this boundary is free
  ///   - bound_cons[ival]=1 if value ival on this boundary is pinned
  //==================================================================
  void RefineableTElement<2>::get_bcs(int bound, Vector<int> &bound_cons) const
  {
    throw_runtime_error("Implement");
  }

  //==================================================================
  /// Determine Vector of boundary conditions along the element's
  /// edge (S/N/W/E) -- BC is the least restrictive combination
  /// of all the nodes on this edge
  ///
  /// Usual convention:
  ///   - bound_cons[ival]=0 if value ival on this boundary is free
  ///   - bound_cons[ival]=1 if value ival on this boundary is pinned
  //==================================================================
  void RefineableTElement<2>::get_edge_bcs(const int &edge, Vector<int> &bound_cons) const
  {
    throw_runtime_error("Implement");
  }

  //==================================================================
  /// Given an element edge/vertex, return a set that contains
  /// all the (mesh-)boundary numbers that this element edge/vertex
  /// lives on.
  ///
  /// For proper edges, the boundary is the one (if any) that is shared by
  /// both vertex nodes). For vertex nodes, we just return their
  /// boundaries.
  //==================================================================
  void RefineableTElement<2>::get_boundaries(const int &edge,
                                             std::set<unsigned> &boundary) const
  {
    throw_runtime_error("Implement");
  }

  //===================================================================
  /// Return the value of the intrinsic boundary coordinate interpolated
  /// along the edge (S/W/N/E)
  //===================================================================
  void RefineableTElement<2>::
      interpolated_zeta_on_edge(const unsigned &boundary,
                                const int &edge, const Vector<double> &s,
                                Vector<double> &zeta)
  {
    throw_runtime_error("Implement");
  }

  //===================================================================
  /// If a neighbouring element has already created a node at
  /// a position corresponding to the local fractional position within the
  /// present element, s_fraction, return
  /// a pointer to that node. If not, return NULL (0). If the node is
  /// periodic the flag is_periodic will be true
  //===================================================================
  Node *RefineableTElement<2>::
      node_created_by_neighbour(const Vector<double> &s_fraction,
                                bool &is_periodic)
  {
    throw_runtime_error("Implement");
    return 0;
  }

  //==================================================================
  /// Build the element by doing the following:
  /// - Give it nodal positions (by establishing the pointers to its
  ///   nodes)
  /// - In the process create new nodes where required (i.e. if they
  ///   don't exist in father element or have already been created
  ///   while building new neighbour elements). Node building
  ///   involves the following steps:
  ///   - Get nodal position from father element.
  ///   - Establish the time-history of the newly created nodal point
  ///     (its coordinates and the previous values) consistent with
  ///     the father's history.
  ///   - Determine the boundary conditions of the nodes (newly
  ///     created nodes can only lie on the interior of any
  ///     edges of the father element -- this makes it possible to
  ///     to figure out what their bc should be...)
  ///   - Add node to the mesh's stoarge scheme for the boundary nodes.
  ///   - Add the new node to the mesh itself
  ///   - Doc newly created nodes in "new_nodes.dat" stored in the directory
  ///     of the DocInfo object (only if it's open!)
  /// - Finally, excute the element-specific further_build()
  ///   (empty by default -- must be overloaded for specific elements).
  ///   This deals with any build operations that are not included
  ///   in the generic process outlined above. For instance, in
  ///   Crouzeix Raviart elements we need to initialise the internal
  ///   pressure values in manner consistent with the pressure
  ///   distribution in the father element.
  //==================================================================
  void RefineableTElement<2>::build(Mesh *&mesh_pt,
                                    Vector<Node *> &new_node_pt,
                                    bool &was_already_built,
                                    std::ofstream &new_nodes_file)
  {
    using namespace QuadTreeNames;
    unsigned n_p = nnode_1d();
    unsigned n_node = this->nnode();

    if (Father_bound[n_p].nrow() == 0)
    {
      setup_father_bounds();
    }
    QuadTree *father_pt = dynamic_cast<QuadTree *>(quadtree_pt()->father_pt());
    int son_type = Tree_pt->son_type();
    if (!nodes_built())
    {
#ifdef PARANOID
      if (father_pt == 0)
      {
        std::string error_message =
            "Something fishy here: I have no father and yet \n";
        error_message +=
            "I have no nodes. Who has created me then?!\n";

        throw OomphLibError(error_message,
                            OOMPH_CURRENT_FUNCTION,
                            OOMPH_EXCEPTION_LOCATION);
      }
#endif

      was_already_built = false;
      RefineableTElement<2> *father_el_pt = dynamic_cast<RefineableTElement<2> *>(father_pt->object_pt());
      TimeStepper *time_stepper_pt = father_el_pt->node_pt(0)->time_stepper_pt();

      unsigned ntstorage = time_stepper_pt->ntstorage();

      if (father_el_pt->node_pt(0)->nposition_type() != 1)
      {
        throw OomphLibError("Can't handle generalised nodal positions (yet).", OOMPH_CURRENT_FUNCTION, OOMPH_EXCEPTION_LOCATION);
      }

      //   Vector<double> s_lo(2);
      //   Vector<double> s_hi(2);
      Vector<Vector<double>> s_in_parent(n_node, Vector<double>(2));
      Vector<Vector<double>> s_in_son(n_node, Vector<double>(2));

      if (n_node != 3 && n_node != 6)
      {
        throw_runtime_error("Implement");
      }

      s_in_son[0][0] = 1.0;
      s_in_son[0][1] = 0.0;
      s_in_son[1][0] = 0.0;
      s_in_son[1][1] = 1.0;
      s_in_son[2][0] = 0.0;
      s_in_son[2][1] = 0.0;
      if (n_node > 3)
      {
        s_in_son[3][0] = 0.5;
        s_in_son[3][1] = 0.5;
        s_in_son[4][0] = 0.0;
        s_in_son[4][1] = 0.5;
        s_in_son[5][0] = 0.5;
        s_in_son[5][1] = 0.0;
        if (n_node > 6)
        {
          throw_runtime_error("Implement");
        }
      }

      // Setup vertex coordinates in father element:
      //--------------------------------------------
      switch (son_type)
      {
      case SW:
        s_in_parent[0][0] = 0.5;
        s_in_parent[0][1] = 0.0;
        s_in_parent[1][0] = 0.0;
        s_in_parent[1][1] = 0.5;
        s_in_parent[2][0] = 0.0;
        s_in_parent[2][1] = 0.0;

        /*if (n_node>3)
        {
         s_in_parent[3][0]=0.25;
         s_in_parent[3][1]=0.25;
         s_in_parent[4][0]=0.0;
         s_in_parent[4][1]=0.25;
         s_in_parent[5][0]=0.25;
         s_in_parent[5][1]=0.0;
        }*/
        break;

      case SE:
        s_in_parent[0][0] = 1;
        s_in_parent[0][1] = 0.0;
        s_in_parent[1][0] = 0.5;
        s_in_parent[1][1] = 0.5;
        s_in_parent[2][0] = 0.5;
        s_in_parent[2][1] = 0.0;

        /*if (n_node>3)
        {
         s_in_parent[3][0]=0.75;
         s_in_parent[3][1]=0.25;
         s_in_parent[4][0]=0.5;
         s_in_parent[4][1]=0.25;
         s_in_parent[5][0]=0.75;
         s_in_parent[5][1]=0.0;
        }*/
        break;

      case NE:
        s_in_parent[0][0] = 0.5;
        s_in_parent[0][1] = 0.0;
        s_in_parent[1][0] = 0.5;
        s_in_parent[1][1] = 0.5;
        s_in_parent[2][0] = 0.0;
        s_in_parent[2][1] = 0.5;

        /*if (n_node>3)
        {
         s_in_parent[3][0]=0.5;
         s_in_parent[3][1]=0.25;
         s_in_parent[4][0]=0.25;
         s_in_parent[4][1]=0.5;
         s_in_parent[5][0]=0.25;
         s_in_parent[5][1]=0.25;
        }*/
        break;

      case NW:
        s_in_parent[0][0] = 0.5;
        s_in_parent[0][1] = 0.5;
        s_in_parent[1][0] = 0.0;
        s_in_parent[1][1] = 1.0;
        s_in_parent[2][0] = 0.0;
        s_in_parent[2][1] = 0.5;

        break;
      }

      if (n_node > 3)
      {
        for (unsigned int i = 0; i < 2; i++)
        {
          s_in_parent[3][i] = 0.5 * (s_in_parent[0][i] + s_in_parent[1][i]);
          s_in_parent[4][i] = 0.5 * (s_in_parent[1][i] + s_in_parent[2][i]);
          s_in_parent[5][i] = 0.5 * (s_in_parent[2][i] + s_in_parent[0][i]);
        }
        if (n_node > 6)
        {
          throw_runtime_error("Impplement");
        }
      }

      if (father_el_pt->Macro_elem_pt != 0)
      {
        set_macro_elem_pt(father_el_pt->Macro_elem_pt);
        for (unsigned i = 0; i < 2; i++)
        {
          throw_runtime_error("MACRO ELEM");
          // s_macro_ll(i)=      father_el_pt->s_macro_ll(i)+0.5*(s_lo[i]+1.0)*(father_el_pt->s_macro_ur(i)-father_el_pt->s_macro_ll(i));
          // s_macro_ur(i)=      father_el_pt->s_macro_ll(i)+0.5*(s_hi[i]+1.0)*(father_el_pt->s_macro_ur(i)-father_el_pt->s_macro_ll(i));
        }
      }

      // If the father element hasn't been generated yet, we're stuck...
      if (father_el_pt->node_pt(0) == 0)
      {
        throw OomphLibError("Trouble: father_el_pt->node_pt(0)==0\n Can't build son element!\n", OOMPH_CURRENT_FUNCTION, OOMPH_EXCEPTION_LOCATION);
      }
      else
      {
        Vector<double> x_small(2);
        Vector<double> x_large(2);

        for (unsigned i = 0; i < n_node; i++)
        {
          {
            bool node_done = false;
            Vector<double> s = s_in_parent[i];
            Vector<double> s_fraction = s_in_son[i];

            Node *created_node_pt = father_el_pt->get_node_at_local_coordinate(s);

            // Does this node already exist in father element?
            //------------------------------------------------
            if (created_node_pt != 0)
            {
              node_pt(i) = created_node_pt;
              for (unsigned t = 0; t < ntstorage; t++)
              {
                Vector<double> prev_values;
                father_el_pt->get_interpolated_values(t, s, prev_values);
                unsigned n_val_at_node = created_node_pt->nvalue();
                unsigned n_val_from_function = prev_values.size();
                unsigned n_var = n_val_at_node < n_val_from_function ? n_val_at_node : n_val_from_function;
                for (unsigned k = 0; k < n_var; k++)
                {
                  created_node_pt->set_value(t, k, prev_values[k]);
                }
              }

              // Node has been created by copy
              node_done = true;
            }
            // Node does not exist in father element but might already
            //--------------------------------------------------------
            // have been created by neighbouring elements
            //-------------------------------------------
            else
            {
              // Was the node created by one of its neighbours
              // Whether or not the node lies on an edge can be calculated
              // by from the fractional position
              bool is_periodic = false;
              ;
              created_node_pt = node_created_by_neighbour(s_fraction, is_periodic);

              // If the node was so created, assign the pointers
              if (created_node_pt != 0)
              {
                // If the node is periodic
                if (is_periodic)
                {
                  // Now the node must be on a boundary, but we don't know which
                  // one
                  // The returned created_node_pt is actually the neighbouring
                  // periodic node
                  Node *neighbour_node_pt = created_node_pt;

                  // Determine the edge on which the new node will live
                  int father_bound = Father_bound[n_p](i, son_type);

                  // Storage for the set of Mesh boundaries on which the
                  // appropriate father edge lives.
                  // [New nodes should always be mid-edge nodes in father
                  // and therefore only live on one boundary but just to
                  // play it safe...]
                  std::set<unsigned> boundaries;
                  // Only get the boundaries if we are at the edge of
                  // an element. Nodes in the centre of an element cannot be
                  // on Mesh boundaries
                  if (father_bound != Tree::OMEGA)
                  {
                    father_el_pt->get_boundaries(father_bound, boundaries);
                  }

#ifdef PARANOID
                  // Case where a new node lives on more than one boundary
                  //  seems fishy enough to flag
                  if (boundaries.size() > 1)
                  {
                    throw OomphLibError(
                        "boundaries.size()!=1 seems a bit strange..\n",
                        OOMPH_CURRENT_FUNCTION,
                        OOMPH_EXCEPTION_LOCATION);
                  }

                  // Case when there are no boundaries, we are in big trouble
                  if (boundaries.size() == 0)
                  {
                    std::ostringstream error_stream;
                    error_stream
                        << "Periodic node is not on a boundary...\n"
                        << "Coordinates: "
                        << created_node_pt->x(0) << " "
                        << created_node_pt->x(1) << "\n";
                    throw OomphLibError(
                        error_stream.str(),
                        OOMPH_CURRENT_FUNCTION,
                        OOMPH_EXCEPTION_LOCATION);
                  }
#endif

                  // Create node and set the pointer to it from the element
                  created_node_pt = construct_boundary_node(i, time_stepper_pt);
                  // Make the node periodic from the neighbour
                  created_node_pt->make_periodic(neighbour_node_pt);
                  // Add to vector of new nodes
                  new_node_pt.push_back(created_node_pt);

                  // Loop over # of history values
                  for (unsigned t = 0; t < ntstorage; t++)
                  {
                    Vector<double> x_prev(2);
                    father_el_pt->get_x(t, s, x_prev);
                    // Set previous positions of the new node
                    for (unsigned i = 0; i < 2; i++)
                    {
                      created_node_pt->x(t, i) = x_prev[i];
                    }
                  }

                  // Next, we Update the boundary lookup schemes
                  // Loop over the boundaries stored in the set
                  for (std::set<unsigned>::iterator it = boundaries.begin(); it != boundaries.end(); ++it)
                  {
                    // Add the node to the boundary
                    mesh_pt->add_boundary_node(*it, created_node_pt);

                    // If we have set an intrinsic coordinate on this
                    // mesh boundary then it must also be interpolated on
                    // the new node
                    // Now interpolate the intrinsic boundary coordinate
                    if (mesh_pt->boundary_coordinate_exists(*it) == true)
                    {
                      Vector<double> zeta(1);
                      father_el_pt->interpolated_zeta_on_edge(*it,
                                                              father_bound,
                                                              s, zeta);

                      created_node_pt->set_coordinates_on_boundary(*it, zeta);
                    }
                  }

                  // Make sure that we add the node to the mesh
                  mesh_pt->add_node_pt(created_node_pt);
                } // End of periodic case
                // Otherwise the node is not periodic, so just set the
                // pointer to the neighbours node
                else
                {
                  node_pt(i) = created_node_pt;
                }
                // Node has been created
                node_done = true;
              }
              // Node does not exist in neighbour element but might already
              //-----------------------------------------------------------
              // have been created by a son of a neighbouring element
              //-----------------------------------------------------
              else
              {
                // Was the node created by one of its neighbours' sons
                // Whether or not the node lies on an edge can be calculated
                // by from the fractional position
                bool is_periodic = false;
                ;
                created_node_pt = node_created_by_son_of_neighbour(s_fraction, is_periodic);

                // If the node was so created, assign the pointers
                if (created_node_pt != 0)
                {
                  // If the node is periodic
                  if (is_periodic)
                  {
                    // Now the node must be on a boundary, but we don't know which
                    // one
                    // The returned created_node_pt is actually the neighbouring
                    // periodic node
                    Node *neighbour_node_pt = created_node_pt;

                    // Determine the edge on which the new node will live
                    int father_bound = Father_bound[n_p](i, son_type);

                    // Storage for the set of Mesh boundaries on which the
                    // appropriate father edge lives.
                    // [New nodes should always be mid-edge nodes in father
                    // and therefore only live on one boundary but just to
                    // play it safe...]
                    std::set<unsigned> boundaries;
                    // Only get the boundaries if we are at the edge of
                    // an element. Nodes in the centre of an element cannot be
                    // on Mesh boundaries
                    if (father_bound != Tree::OMEGA)
                    {
                      father_el_pt->get_boundaries(father_bound, boundaries);
                    }

#ifdef PARANOID
                    // Case where a new node lives on more than one boundary
                    //  seems fishy enough to flag
                    if (boundaries.size() > 1)
                    {
                      throw OomphLibError(
                          "boundaries.size()!=1 seems a bit strange..\n",
                          OOMPH_CURRENT_FUNCTION,
                          OOMPH_EXCEPTION_LOCATION);
                    }

                    // Case when there are no boundaries, we are in big trouble
                    if (boundaries.size() == 0)
                    {
                      std::ostringstream error_stream;
                      error_stream
                          << "Periodic node is not on a boundary...\n"
                          << "Coordinates: "
                          << created_node_pt->x(0) << " "
                          << created_node_pt->x(1) << "\n";
                      throw OomphLibError(
                          error_stream.str(),
                          OOMPH_CURRENT_FUNCTION,
                          OOMPH_EXCEPTION_LOCATION);
                    }
#endif

                    // Create node and set the pointer to it from the element
                    created_node_pt =
                        construct_boundary_node(i, time_stepper_pt);
                    // Make the node periodic from the neighbour
                    created_node_pt->make_periodic(neighbour_node_pt);
                    // Add to vector of new nodes
                    new_node_pt.push_back(created_node_pt);

                    // Loop over # of history values
                    for (unsigned t = 0; t < ntstorage; t++)
                    {
                      // Get position from father element -- this uses the macro
                      // element representation if appropriate. If the node
                      // turns out to be a hanging node later on, then
                      // its position gets adjusted in line with its
                      // hanging node interpolation.
                      Vector<double> x_prev(2);
                      father_el_pt->get_x(t, s, x_prev);
                      // Set previous positions of the new node
                      for (unsigned i = 0; i < 2; i++)
                      {
                        created_node_pt->x(t, i) = x_prev[i];
                      }
                    }

                    // Next, we Update the boundary lookup schemes
                    // Loop over the boundaries stored in the set
                    for (std::set<unsigned>::iterator it = boundaries.begin();
                         it != boundaries.end(); ++it)
                    {
                      // Add the node to the boundary
                      mesh_pt->add_boundary_node(*it, created_node_pt);

                      // If we have set an intrinsic coordinate on this
                      // mesh boundary then it must also be interpolated on
                      // the new node
                      // Now interpolate the intrinsic boundary coordinate
                      if (mesh_pt->boundary_coordinate_exists(*it) == true)
                      {
                        Vector<double> zeta(1);
                        father_el_pt->interpolated_zeta_on_edge(*it,
                                                                father_bound,
                                                                s, zeta);

                        created_node_pt->set_coordinates_on_boundary(*it, zeta);
                      }
                    }

                    // Make sure that we add the node to the mesh
                    mesh_pt->add_node_pt(created_node_pt);
                  } // End of periodic case
                  // Otherwise the node is not periodic, so just set the
                  // pointer to the neighbours node
                  else
                  {
                    node_pt(i) = created_node_pt;
                  }
                  // Node has been created
                  node_done = true;
                } // Node does not exist in son of neighbouring element
              }   // Node does not exist in neighbouring element
            }     // Node does not exist in father element

            // Node has not been built anywhere ---> build it here
            if (!node_done)
            {
              // Firstly, we need to determine whether or not a node lies
              // on the boundary before building it, because
              // we actually assign a different type of node on boundaries.

              // The node can only be on a Mesh boundary if it
              // lives on an edge that is shared with an edge of its
              // father element; i.e. it is not created inside the father element
              // Determine the edge on which the new node will live
              int father_bound = Father_bound[n_p](i, son_type);

              // Storage for the set of Mesh boundaries on which the
              // appropriate father edge lives.
              // [New nodes should always be mid-edge nodes in father
              // and therefore only live on one boundary but just to
              // play it safe...]
              std::set<unsigned> boundaries;
              // Only get the boundaries if we are at the edge of
              // an element. Nodes in the centre of an element cannot be
              // on Mesh boundaries
              if (father_bound != Tree::OMEGA)
              {
                father_el_pt->get_boundaries(father_bound, boundaries);
              }

#ifdef PARANOID
              // Case where a new node lives on more than one boundary
              //  seems fishy enough to flag
              if (boundaries.size() > 1)
              {
                throw OomphLibError(
                    "boundaries.size()!=1 seems a bit strange..\n",
                    OOMPH_CURRENT_FUNCTION,
                    OOMPH_EXCEPTION_LOCATION);
              }
#endif

              // If the node lives on a mesh boundary,
              // then we need to create a boundary node
              if (boundaries.size() > 0)
              {
                // Create node and set the pointer to it from the element
                created_node_pt = construct_boundary_node(i, time_stepper_pt);
                // Add to vector of new nodes
                new_node_pt.push_back(created_node_pt);

                // Now we need to work out whether to pin the values at
                // the new node based on the boundary conditions applied at
                // its Mesh boundary

                // Get the boundary conditions from the father
                Vector<int> bound_cons(ncont_interpolated_values());
                father_el_pt->get_bcs(father_bound, bound_cons);

                // Loop over the values and pin, if necessary
                unsigned n_value = created_node_pt->nvalue();
                for (unsigned k = 0; k < n_value; k++)
                {
                  if (bound_cons[k])
                  {
                    created_node_pt->pin(k);
                  }
                }

                // Solid node? If so, deal with the positional boundary
                // conditions:

                /* //PROBABLY NOT REQUIRED FOR PYOOMPH
                SolidNode* solid_node_pt = dynamic_cast<SolidNode*>(created_node_pt);
                if (solid_node_pt!=0)
                 {
                  //Get the positional boundary conditions from the father:
                  unsigned n_dim = created_node_pt->ndim();
                  Vector<int> solid_bound_cons(n_dim);
                  RefineableSolidTElement<2>* father_solid_el_pt=dynamic_cast<RefineableSolidTElement<2>*>(father_el_pt);
   #ifdef PARANOID
                  if (father_solid_el_pt==0)
                   {
                    std::string error_message =
                     "We have a SolidNode outside a refineable SolidElement\n";
                    error_message +=
                     "during mesh refinement -- this doesn't make sense";

                    throw OomphLibError(error_message,
                                        OOMPH_CURRENT_FUNCTION,
                                        OOMPH_EXCEPTION_LOCATION);
                   }
   #endif
                  father_solid_el_pt->
                   get_solid_bcs(father_bound,solid_bound_cons);

                  //Loop over the positions and pin, if necessary
                  for(unsigned k=0;k<n_dim;k++)
                   {
                    if (solid_bound_cons[k]) {solid_node_pt->pin_position(k);}
                   }
                 } //End of if solid_node_pt
                 */

                // Next, we Update the boundary lookup schemes
                // Loop over the boundaries stored in the set
                for (std::set<unsigned>::iterator it = boundaries.begin();
                     it != boundaries.end(); ++it)
                {
                  // Add the node to the boundary
                  mesh_pt->add_boundary_node(*it, created_node_pt);

                  // If we have set an intrinsic coordinate on this
                  // mesh boundary then it must also be interpolated on
                  // the new node
                  // Now interpolate the intrinsic boundary coordinate
                  if (mesh_pt->boundary_coordinate_exists(*it) == true)
                  {
                    Vector<double> zeta(1);
                    father_el_pt->interpolated_zeta_on_edge(*it,
                                                            father_bound,
                                                            s, zeta);

                    created_node_pt->set_coordinates_on_boundary(*it, zeta);
                  }
                }
              }
              // Otherwise the node is not on a Mesh boundary and
              // we create a normal "bulk" node
              else
              {
                // Create node and set the pointer to it from the element
                created_node_pt = construct_node(i, time_stepper_pt);
                // Add to vector of new nodes
                new_node_pt.push_back(created_node_pt);
              }

              // Now we set the position and values at the newly created node

              // In the first instance use macro element or FE representation
              // to create past and present nodal positions.
              // (THIS STEP SHOULD NOT BE SKIPPED FOR ALGEBRAIC
              // ELEMENTS AS NOT ALL OF THEM NECESSARILY IMPLEMENT
              // NONTRIVIAL NODE UPDATE FUNCTIONS. CALLING
              // THE NODE UPDATE FOR SUCH ELEMENTS/NODES WILL LEAVE
              // THEIR NODAL POSITIONS WHERE THEY WERE (THIS IS APPROPRIATE
              // ONCE THEY HAVE BEEN GIVEN POSITIONS) BUT WILL
              // NOT ASSIGN SENSIBLE INITIAL POSITONS!

              // Loop over # of history values
              for (unsigned t = 0; t < ntstorage; t++)
              {
                // Get position from father element -- this uses the macro
                // element representation if appropriate. If the node
                // turns out to be a hanging node later on, then
                // its position gets adjusted in line with its
                // hanging node interpolation.
                Vector<double> x_prev(2);
                father_el_pt->get_x(t, s, x_prev);

                // Set previous positions of the new node
                for (unsigned i = 0; i < 2; i++)
                {
                  created_node_pt->x(t, i) = x_prev[i];
                }
              }

              // Loop over all history values
              for (unsigned t = 0; t < ntstorage; t++)
              {
                // Get values from father element
                // Note: get_interpolated_values() sets Vector size itself.
                Vector<double> prev_values;
                father_el_pt->get_interpolated_values(t, s, prev_values);
                // Initialise the values at the new node
                unsigned n_value = created_node_pt->nvalue();
                for (unsigned k = 0; k < n_value; k++)
                {
                  created_node_pt->set_value(t, k, prev_values[k]);
                }
              }

              // Add new node to mesh
              mesh_pt->add_node_pt(created_node_pt);

            } // End of case when we build the node ourselves

            // Check if the element is an algebraic element
            AlgebraicElementBase *alg_el_pt =
                dynamic_cast<AlgebraicElementBase *>(this);

            // If the element is an algebraic element, setup
            // node position (past and present) from algebraic node update
            // function. This over-writes previous assingments that
            // were made based on the macro-element/FE representation.
            // NOTE: YES, THIS NEEDS TO BE CALLED REPEATEDLY IF THE
            // NODE IS MEMBER OF MULTIPLE ELEMENTS: THEY ALL ASSIGN
            // THE SAME NODAL POSITIONS BUT WE NEED TO ADD THE REMESH
            // INFO FOR *ALL* ROOT ELEMENTS!
            if (alg_el_pt != 0)
            {
              // Build algebraic node update info for new node
              // This sets up the node update data for all node update
              // functions that are shared by all nodes in the father
              // element
              alg_el_pt->setup_algebraic_node_update(node_pt(i), s,
                                                     father_el_pt);
            }

            // If we have built the node and we are documenting our progress
            // write the (hopefully consistent position) to  the outputfile
            if ((!node_done) && (new_nodes_file.is_open()))
            {
              new_nodes_file << node_pt(i)->x(0) << " "
                             << node_pt(i)->x(1) << std::endl;
            }

          } // End of vertical loop over nodes in element

        } // End of horizontal loop over nodes in element

        // If the element is a MacroElementNodeUpdateElement, set
        // the update parameters for the current element's nodes --
        // all this needs is the vector of (pointers to the)
        // geometric objects that affect the MacroElement-based
        // node update -- this is the same as that in the father element

        /* // PROBABLY NOT REQUIRED FOR PYOOMPH
        MacroElementNodeUpdateElementBase* father_m_el_pt=dynamic_cast<
         MacroElementNodeUpdateElementBase*>(father_el_pt);
        if (father_m_el_pt!=0)
         {
          // Get vector of geometric objects from father (construct vector
          // via copy operation)
          Vector<GeomObject*> geom_object_pt(father_m_el_pt->geom_object_pt());

          // Cast current element to MacroElementNodeUpdateElement:
          MacroElementNodeUpdateElementBase* m_el_pt=dynamic_cast<
           MacroElementNodeUpdateElementBase*>(this);

   #ifdef PARANOID
          if (m_el_pt==0)
           {
            std::string error_message =
             "Failed to cast to MacroElementNodeUpdateElementBase*\n";
            error_message +=
             "Strange -- if the father is a MacroElementNodeUpdateElement\n";
             error_message += "the son should be too....\n";

            throw OomphLibError(error_message,
                                OOMPH_CURRENT_FUNCTION,
                                OOMPH_EXCEPTION_LOCATION);
           }
   #endif
          // Build update info by passing vector of geometric objects:
          // This sets the current element to be the update element
          // for all of the element's nodes -- this is reversed
          // if the element is ever un-refined in the father element's
          // rebuild_from_sons() function which overwrites this
          // assignment to avoid nasty segmentation faults that occur
          // when a node tries to update itself via an element that no
          // longer exists...
          m_el_pt->set_node_update_info(geom_object_pt);
         }*/

#ifdef OOMPH_HAS_MPI
        // Pass on non-halo proc id
        Non_halo_proc_ID =
            tree_pt()->father_pt()->object_pt()->non_halo_proc_ID();
#endif

        // Is it an ElementWithMovingNodes?
        ElementWithMovingNodes *aux_el_pt =
            dynamic_cast<ElementWithMovingNodes *>(this);

        // Pass down the information re the method for the evaluation
        // of the shape derivatives
        if (aux_el_pt != 0)
        {
          ElementWithMovingNodes *aux_father_el_pt =
              dynamic_cast<ElementWithMovingNodes *>(father_el_pt);

#ifdef PARANOID
          if (aux_father_el_pt == 0)
          {
            std::string error_message =
                "Failed to cast to ElementWithMovingNodes*\n";
            error_message +=
                "Strange -- if the son is a ElementWithMovingNodes\n";
            error_message += "the father should be too....\n";

            throw OomphLibError(error_message,
                                OOMPH_CURRENT_FUNCTION,
                                OOMPH_EXCEPTION_LOCATION);
          }
#endif

          // If evaluating the residuals by finite differences in the father
          // continue to do so in the child
          if (aux_father_el_pt
                  ->are_dresidual_dnodal_coordinates_always_evaluated_by_fd())
          {
            aux_el_pt->enable_always_evaluate_dresidual_dnodal_coordinates_by_fd();
          }

          aux_el_pt->method_for_shape_derivs() =
              aux_father_el_pt->method_for_shape_derivs();

          // If bypassing the evaluation of fill_in_jacobian_from_geometric_data
          // continue to do so
          if (aux_father_el_pt
                  ->is_fill_in_jacobian_from_geometric_data_bypassed())
          {
            aux_el_pt->enable_bypass_fill_in_jacobian_from_geometric_data();
          }
        }

        // Now do further build (if any)
        further_build();

      } // Sanity check: Father element has been generated
    }
    // Element has already been built
    else
    {
      was_already_built = true;
    }
  }

  //====================================================================
  ///  Print corner nodes, use colour (default "BLACK")
  //====================================================================
  void RefineableTElement<2>::output_corners(std::ostream &outfile,
                                             const std::string &colour) const
  {
    throw_runtime_error("Implement");
  }

  //====================================================================
  /// Set up all hanging nodes. If we are documenting the output then
  /// open the output files and pass the open files to the helper function
  //====================================================================
  void RefineableTElement<2>::setup_hanging_nodes(Vector<std::ofstream *>
                                                      &output_stream)
  {
    throw_runtime_error("Implement");
  }

  //================================================================
  /// Internal function that sets up the hanging node scheme for
  /// a particular continuously interpolated value
  //===============================================================
  void RefineableTElement<2>::setup_hang_for_value(const int &value_id)
  {
    throw_runtime_error("Implement");
  }

  //=================================================================
  /// Internal function to set up the hanging nodes on a particular
  /// edge of the element
  //=================================================================
  void RefineableTElement<2>::
      quad_hang_helper(const int &value_id,
                       const int &my_edge, std::ofstream &output_hangfile)
  {
    throw_runtime_error("Implement");
  }

  //=================================================================
  /// Check inter-element continuity of
  /// - nodal positions
  /// - (nodally) interpolated function values
  //====================================================================
  // template<unsigned NNODE_1D>
  void RefineableTElement<2>::check_integrity(double &max_error)
  {

    throw_runtime_error("Implement");
  }

  //========================================================================
  /// Static matrix for coincidence between son nodal points and
  /// father boundaries
  ///
  //========================================================================
  std::map<unsigned, DenseMatrix<int>> RefineableTElement<2>::Father_bound;

  //==================================================================
  /// Setup static matrix for coincidence between son nodal points and
  /// father boundaries:
  ///
  /// Father_boundd[nnode_1d](nnode_son,son_type)={SW/SE/NW/NE/S/E/N/W/OMEGA}
  ///
  /// so that node nnode_son in element of type son_type lies
  /// on boundary/vertex Father_boundd[nnode_1d](nnode_son,son_type) in its
  /// father element. If the node doesn't lie on a boundary
  /// the value is OMEGA.
  //==================================================================
  void RefineableTElement<3>::setup_father_bounds()
  {
    throw_runtime_error("Implement");
  }

  //==================================================================
  /// Determine Vector of boundary conditions along the element's boundary
  /// (or vertex) bound (S/W/N/E/SW/SE/NW/NE).
  ///
  /// This function assumes that the same boundary condition is applied
  /// along the entire length of an element's edge (of course, the
  /// vertices combine the boundary conditions of their two adjacent edges
  /// in the most restrictive combination. Hence, if we're at a vertex,
  /// we apply the most restrictive boundary condition of the
  /// two adjacent edges. If we're on an edge (in its proper interior),
  /// we apply the least restrictive boundary condition of all nodes
  /// along the edge.
  ///
  /// Usual convention:
  ///   - bound_cons[ival]=0 if value ival on this boundary is free
  ///   - bound_cons[ival]=1 if value ival on this boundary is pinned
  //==================================================================
  void RefineableTElement<3>::get_bcs(int bound, Vector<int> &bound_cons) const
  {
    throw_runtime_error("Implement");
  }

  //==================================================================
  /// Determine Vector of boundary conditions along the element's
  /// edge (S/N/W/E) -- BC is the least restrictive combination
  /// of all the nodes on this edge
  ///
  /// Usual convention:
  ///   - bound_cons[ival]=0 if value ival on this boundary is free
  ///   - bound_cons[ival]=1 if value ival on this boundary is pinned
  //==================================================================
  void RefineableTElement<3>::get_edge_bcs(const int &edge, Vector<int> &bound_cons) const
  {
    throw_runtime_error("Implement");
  }

  //==================================================================
  /// Given an element edge/vertex, return a set that contains
  /// all the (mesh-)boundary numbers that this element edge/vertex
  /// lives on.
  ///
  /// For proper edges, the boundary is the one (if any) that is shared by
  /// both vertex nodes). For vertex nodes, we just return their
  /// boundaries.
  //==================================================================
  void RefineableTElement<3>::get_boundaries(const int &edge,
                                             std::set<unsigned> &boundary) const
  {
    throw_runtime_error("Implement");
  }

  //===================================================================
  /// Return the value of the intrinsic boundary coordinate interpolated
  /// along the edge (S/W/N/E)
  //===================================================================
  void RefineableTElement<3>::
      interpolated_zeta_on_edge(const unsigned &boundary,
                                const int &edge, const Vector<double> &s,
                                Vector<double> &zeta)
  {
    throw_runtime_error("Implement");
  }

  //===================================================================
  /// If a neighbouring element has already created a node at
  /// a position corresponding to the local fractional position within the
  /// present element, s_fraction, return
  /// a pointer to that node. If not, return NULL (0). If the node is
  /// periodic the flag is_periodic will be true
  //===================================================================
  Node *RefineableTElement<3>::
      node_created_by_neighbour(const Vector<double> &s_fraction,
                                bool &is_periodic)
  {
    throw_runtime_error("Implement");
    return 0;
  }

  //==================================================================
  /// Build the element by doing the following:
  /// - Give it nodal positions (by establishing the pointers to its
  ///   nodes)
  /// - In the process create new nodes where required (i.e. if they
  ///   don't exist in father element or have already been created
  ///   while building new neighbour elements). Node building
  ///   involves the following steps:
  ///   - Get nodal position from father element.
  ///   - Establish the time-history of the newly created nodal point
  ///     (its coordinates and the previous values) consistent with
  ///     the father's history.
  ///   - Determine the boundary conditions of the nodes (newly
  ///     created nodes can only lie on the interior of any
  ///     edges of the father element -- this makes it possible to
  ///     to figure out what their bc should be...)
  ///   - Add node to the mesh's stoarge scheme for the boundary nodes.
  ///   - Add the new node to the mesh itself
  ///   - Doc newly created nodes in "new_nodes.dat" stored in the directory
  ///     of the DocInfo object (only if it's open!)
  /// - Finally, excute the element-specific further_build()
  ///   (empty by default -- must be overloaded for specific elements).
  ///   This deals with any build operations that are not included
  ///   in the generic process outlined above. For instance, in
  ///   Crouzeix Raviart elements we need to initialise the internal
  ///   pressure values in manner consistent with the pressure
  ///   distribution in the father element.
  //==================================================================
  void RefineableTElement<3>::build(Mesh *&mesh_pt,
                                    Vector<Node *> &new_node_pt,
                                    bool &was_already_built,
                                    std::ofstream &new_nodes_file)
  {
    throw_runtime_error("Implement");
  }

  //====================================================================
  ///  Print corner nodes, use colour (default "BLACK")
  //====================================================================
  void RefineableTElement<3>::output_corners(std::ostream &outfile,
                                             const std::string &colour) const
  {
    throw_runtime_error("Implement");
  }

  //====================================================================
  /// Set up all hanging nodes. If we are documenting the output then
  /// open the output files and pass the open files to the helper function
  //====================================================================
  void RefineableTElement<3>::setup_hanging_nodes(Vector<std::ofstream *>
                                                      &output_stream)
  {
    throw_runtime_error("Implement");
  }

  //================================================================
  /// Internal function that sets up the hanging node scheme for
  /// a particular continuously interpolated value
  //===============================================================
  void RefineableTElement<3>::setup_hang_for_value(const int &value_id)
  {
    throw_runtime_error("Implement");
  }

  //=================================================================
  /// Internal function to set up the hanging nodes on a particular
  /// edge of the element
  //=================================================================
  void RefineableTElement<3>::
      quad_hang_helper(const int &value_id,
                       const int &my_edge, std::ofstream &output_hangfile)
  {
    throw_runtime_error("Implement");
  }

  //=================================================================
  /// Check inter-element continuity of
  /// - nodal positions
  /// - (nodally) interpolated function values
  //====================================================================
  // template<unsigned NNODE_1D>
  void RefineableTElement<3>::check_integrity(double &max_error)
  {

    throw_runtime_error("Implement");
  }

  //========================================================================
  /// Static matrix for coincidence between son nodal points and
  /// father boundaries
  ///
  //========================================================================
  std::map<unsigned, DenseMatrix<int>> RefineableTElement<3>::Father_bound;

}
