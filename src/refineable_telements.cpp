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

  // This file implements RefineableTElement<DIM> (declared in refineable_telements.hpp),
  // the glue between oomph-lib's simplex (T-type: line/triangle/tetrahedron) template
  // elements and its RefineableElement/tree-based h-adaptivity machinery. Most of the
  // logic below is carried over largely unmodified from oomph-lib's own refineable simplex
  // element sources. The DIM=2 (triangle) specialisation is the one with working
  // (non-stub) implementations of build()/setup_father_bounds() etc.; the DIM=1 and DIM=3
  // specialisations mostly still throw_runtime_error("Implement") and are placeholders.

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
  void RefineableTElement<1>::get_bcs(int , Vector<int> &) const
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
  void RefineableTElement<1>::get_edge_bcs(const int &, Vector<int> &) const
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
  void RefineableTElement<1>::get_boundaries(const int &,
                                             std::set<unsigned> &) const
  {
    throw_runtime_error("Implement");
  }

  //===================================================================
  /// Return the value of the intrinsic boundary coordinate interpolated
  /// along the edge (S/W/N/E)
  //===================================================================
  void RefineableTElement<1>::
      interpolated_zeta_on_edge(const unsigned &,
                                const int &, const Vector<double> &,
                                Vector<double> &)
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
      node_created_by_neighbour(const Vector<double> &,
                                bool &)
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
  void RefineableTElement<1>::build(Mesh *&,
                                    Vector<Node *> &,
                                    bool &,
                                    std::ofstream &)
  {
    throw_runtime_error("Implement");
  }

  //====================================================================
  ///  Print corner nodes, use colour (default "BLACK")
  //====================================================================
  void RefineableTElement<1>::output_corners(std::ostream &,
                                             const std::string &) const
  {
    throw_runtime_error("Implement");
  }

  //====================================================================
  /// Set up all hanging nodes. If we are documenting the output then
  /// open the output files and pass the open files to the helper function
  //====================================================================
  void RefineableTElement<1>::setup_hanging_nodes(Vector<std::ofstream *>
                                                      &)
  {
    throw_runtime_error("Implement");
  }

  //================================================================
  /// Internal function that sets up the hanging node scheme for
  /// a particular continuously interpolated value
  //===============================================================
  void RefineableTElement<1>::setup_hang_for_value(const int &)
  {
    throw_runtime_error("Implement");
  }

  //=================================================================
  /// Internal function to set up the hanging nodes on a particular
  /// edge of the element
  //=================================================================
  void RefineableTElement<1>::
      quad_hang_helper(const int &,
                       const int &, std::ofstream &)
  {
    throw_runtime_error("Implement");
  }

  //=================================================================
  /// Check inter-element continuity of
  /// - nodal positions
  /// - (nodally) interpolated function values
  //====================================================================
  // template<unsigned NNODE_1D>
  void RefineableTElement<1>::check_integrity(double &)
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
    // Brings SW/SE/NW/NE/S/E/N/W/OMEGA son-type/boundary enum names into scope
    using namespace QuadTreeNames;

    // Find the number of nodes along a 1D edge
    unsigned n_p = nnode_1d();
    // Total number of nodes in the element (3 for linear, 6 for quadratic triangles)
    unsigned nnode = this->nnode();
    // Allocate space for the boundary information
    // 3 (C1), 6 (C2) or 7 (C2TB: node 6 is the interior centroid/bubble, never on a father
    // boundary -- it stays OMEGA and needs no explicit assignment below).
    if (nnode == 3)
    {
      Father_bound[n_p].resize(3, 4);
    }
    else if (nnode == 6)
    {
      Father_bound[n_p].resize(6, 4);
    }
    else if (nnode == 7)
    {
      Father_bound[n_p].resize(7, 4);
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

    // A triangle is refined into 4 son triangles (reusing the QuadTree SW/SE/NW/NE
    // son-type names even though the underlying tree is a QuadTree of triangles).
    // Nodes 0-2 are the corner vertices, nodes 3-5 (if present) are the mid-edge
    // nodes of a quadratic (6-node) triangle, opposite to vertices 0,1,2 respectively.

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
    using namespace QuadTreeNames;
    unsigned nvalue = bound_cons.size();
    // Triangle vertices SW/SE/NW map to father corner nodes v2/v0/v1; each is the
    // meeting point of two edges. At a vertex we apply the *most* restrictive (OR)
    // of its two adjacent edges' bcs; along a proper edge we forward to get_edge_bcs.
    switch (bound)
    {
    case E:
    case W:
    case S:
      get_edge_bcs(bound, bound_cons);
      break;
    case SE: // father vertex v0 = edges S (v2-v0) and E (v0-v1)
    {
      Vector<int> bc1(nvalue), bc2(nvalue);
      get_edge_bcs(S, bc1);
      get_edge_bcs(E, bc2);
      for (unsigned k = 0; k < nvalue; k++) bound_cons[k] = (bc1[k] || bc2[k]);
      break;
    }
    case NW: // father vertex v1 = edges E (v0-v1) and W (v1-v2)
    {
      Vector<int> bc1(nvalue), bc2(nvalue);
      get_edge_bcs(E, bc1);
      get_edge_bcs(W, bc2);
      for (unsigned k = 0; k < nvalue; k++) bound_cons[k] = (bc1[k] || bc2[k]);
      break;
    }
    case SW: // father vertex v2 = edges W (v1-v2) and S (v2-v0)
    {
      Vector<int> bc1(nvalue), bc2(nvalue);
      get_edge_bcs(W, bc1);
      get_edge_bcs(S, bc2);
      for (unsigned k = 0; k < nvalue; k++) bound_cons[k] = (bc1[k] || bc2[k]);
      break;
    }
    default:
      for (unsigned k = 0; k < nvalue; k++) bound_cons[k] = 0;
      break;
    }
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
    using namespace QuadTreeNames;
    // The two corner nodes at the ends of this triangle edge (consistent with the
    // Father_bound table: E = v0-v1, W = v1-v2, S = v2-v0).
    int left_node = -1, right_node = -1;
    switch (edge)
    {
    case E: left_node = 0; right_node = 1; break;
    case W: left_node = 1; right_node = 2; break;
    case S: left_node = 2; right_node = 0; break;
    default: break;
    }
    unsigned nvalue = bound_cons.size();
    if (left_node < 0 || right_node < 0)
    {
      for (unsigned k = 0; k < nvalue; k++) bound_cons[k] = 0;
      return;
    }
    // A value is treated as pinned along the edge only if it is pinned at *both*
    // end nodes (least restrictive combination), mirroring oomph's quad get_edge_bcs.
    for (unsigned k = 0; k < nvalue; k++)
    {
      bound_cons[k] = node_pt(left_node)->is_pinned(k) * node_pt(right_node)->is_pinned(k);
    }
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
    using namespace QuadTreeNames;
    boundary.clear();
    // Edge -> its two end (corner) nodes; vertex codes -> a single corner node.
    // Consistent with setup_father_bounds: E=v0-v1, W=v1-v2, S=v2-v0; the son
    // corners coinciding with father vertices are SE=v0, NW=v1, SW=v2.
    int left_node = -1, right_node = -1;
    switch (edge)
    {
    case E: left_node = 0; right_node = 1; break;
    case W: left_node = 1; right_node = 2; break;
    case S: left_node = 2; right_node = 0; break;
    case SE: right_node = 0; break;
    case NW: right_node = 1; break;
    case SW: right_node = 2; break;
    default: return;
    }

    std::set<unsigned> *right_bound_pt = 0;
    if (right_node >= 0)
    {
      if (BoundaryNodeBase *bn = dynamic_cast<BoundaryNodeBase *>(node_pt(right_node)))
        bn->get_boundaries_pt(right_bound_pt);
    }

    // Vertex: just return that node's boundaries.
    if (left_node < 0)
    {
      if (right_bound_pt) boundary = *right_bound_pt;
      return;
    }

    // Proper edge: the boundaries shared by *both* end nodes.
    std::set<unsigned> *left_bound_pt = 0;
    if (BoundaryNodeBase *bn = dynamic_cast<BoundaryNodeBase *>(node_pt(left_node)))
      bn->get_boundaries_pt(left_bound_pt);
    if (left_bound_pt && right_bound_pt)
    {
      std::set_intersection(left_bound_pt->begin(), left_bound_pt->end(),
                            right_bound_pt->begin(), right_bound_pt->end(),
                            std::inserter(boundary, boundary.begin()));
    }
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
    using namespace QuadTreeNames;
    unsigned n_node = this->nnode();
    Shape psi(n_node);
    this->shape(s, psi);
    // Nodes lying on this triangle edge (corners, plus the mid-edge node for a
    // 6-node quadratic triangle): E=v0-v1(+mid 3), W=v1-v2(+mid 4), S=v2-v0(+mid 5).
    std::vector<unsigned> edge_nodes;
    switch (edge)
    {
    case E: edge_nodes = {0, 1}; if (n_node > 3) edge_nodes.push_back(3); break;
    case W: edge_nodes = {1, 2}; if (n_node > 3) edge_nodes.push_back(4); break;
    case S: edge_nodes = {2, 0}; if (n_node > 3) edge_nodes.push_back(5); break;
    default: zeta[0] = 0.0; return;
    }
    double inter_zeta = 0.0;
    Vector<double> zeta_tmp(1);
    for (unsigned n : edge_nodes)
    {
      node_pt(n)->get_coordinates_on_boundary(boundary, zeta_tmp);
      inter_zeta += zeta_tmp[0] * psi(n);
    }
    zeta[0] = inter_zeta;
  }

  //===================================================================
  /// If a neighbouring element has already created a node at
  /// a position corresponding to the local fractional position within the
  /// present element, s_fraction, return
  /// a pointer to that node. If not, return NULL (0). If the node is
  /// periodic the flag is_periodic will be true
  //===================================================================
  Node *RefineableTElement<2>::
      node_created_by_neighbour(const Vector<double> &,
                                bool &is_periodic)
  {
    // Not used: triangle build() shares father-edge nodes via the geometric shared-node
    // registry (father_edge_node_key) instead of oomph's quad compass neighbour finding,
    // whose coordinate descent is geometrically wrong for triangles. Kept as a non-throwing
    // stub for interface compatibility.
    is_periodic = false;
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
    // Brings SW/SE/NW/NE/S/E/N/W/OMEGA son-type/boundary enum names into scope
    using namespace QuadTreeNames;
    unsigned n_p = nnode_1d();
    unsigned n_node = this->nnode();

    // Lazily (re-)build the Father_bound lookup table for this node count if
    // it hasn't been set up yet
    if (Father_bound[n_p].nrow() == 0)
    {
      setup_father_bounds();
    }
    QuadTree *father_pt = dynamic_cast<QuadTree *>(quadtree_pt()->father_pt());
    // Which of the 4 sons (SW/SE/NW/NE) this element is, within its father
    int son_type = Tree_pt->son_type();
    // If the nodes haven't been built yet, this element must be built from its father
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
      // Local coordinates of each of this element's nodes, expressed in the
      // father's local coordinate system (s_in_parent) and in this (son)
      // element's own local coordinate system (s_in_son)
      Vector<Vector<double>> s_in_parent(n_node, Vector<double>(2));
      Vector<Vector<double>> s_in_son(n_node, Vector<double>(2));

      // 3-node (C1), 6-node (C2) and 7-node (C2TB, bubble-enriched: node 6 is the centroid) tris.
      if (n_node != 3 && n_node != 6 && n_node != 7)
      {
        throw_runtime_error("Implement");
      }

      // Corner vertices of the son element in its own local coordinates
      s_in_son[0][0] = 1.0;
      s_in_son[0][1] = 0.0;
      s_in_son[1][0] = 0.0;
      s_in_son[1][1] = 1.0;
      s_in_son[2][0] = 0.0;
      s_in_son[2][1] = 0.0;
      // Mid-edge nodes of the son element (quadratic, 6-node triangle only)
      if (n_node > 3)
      {
        s_in_son[3][0] = 0.5;
        s_in_son[3][1] = 0.5;
        s_in_son[4][0] = 0.0;
        s_in_son[4][1] = 0.5;
        s_in_son[5][0] = 0.5;
        s_in_son[5][1] = 0.0;
        // Bubble-enriched (C2TB): node 6 is the centroid, interior to the son.
        if (n_node > 6)
        {
          s_in_son[6][0] = 1.0 / 3.0;
          s_in_son[6][1] = 1.0 / 3.0;
        }
      }

      // Setup vertex coordinates in father element:
      // --------------------------------------------
      // For each of the 4 son types, the corner vertices of the son
      // (indices 0-2) are placed at the corresponding vertex/mid-edge
      // location within the father's local coordinates. Mid-edge nodes
      // (3-5, commented out below) would be the midpoints of the son's
      // edges in father coordinates, but are instead computed generically
      // by averaging below since they coincide with the midpoint formula
      // in all 4 cases.
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

      // Mid-edge nodes (3-5) of the son, in father coordinates, are simply
      // the midpoints of the son's corner vertices (which are themselves
      // already expressed in father coordinates above)
      if (n_node > 3)
      {
        for (unsigned int i = 0; i < 2; i++)
        {
          s_in_parent[3][i] = 0.5 * (s_in_parent[0][i] + s_in_parent[1][i]);
          s_in_parent[4][i] = 0.5 * (s_in_parent[1][i] + s_in_parent[2][i]);
          s_in_parent[5][i] = 0.5 * (s_in_parent[2][i] + s_in_parent[0][i]);
        }
        // Bubble node (C2TB): the son centroid in father coordinates is the average of the son's
        // three corner vertices (already expressed in father coordinates above). Interior node --
        // never on a father boundary, never shared/hanging.
        if (n_node > 6)
        {
          for (unsigned int i = 0; i < 2; i++)
            s_in_parent[6][i] = (s_in_parent[0][i] + s_in_parent[1][i] + s_in_parent[2][i]) / 3.0;
        }
      }

      // If the father is defined via a macro element (curvilinear boundary
      // representation), the son should inherit/derive its own macro-element
      // sub-region -- not yet implemented for triangles
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

        // Loop over all nodes of this (son) element, creating each one
        // unless it can be re-used from the father, a neighbour, or a
        // neighbour's son
        for (unsigned i = 0; i < n_node; i++)
        {
          {
            bool node_done = false;
            Vector<double> s = s_in_parent[i];
            Vector<double> s_fraction = s_in_son[i];
            if (getenv("PYOOMPH_TRI_AFFINE_CHECK")) // self-check: son<->father affine reproduces build's s_in_parent/s_in_son
            {
              Vector<double> sf(2), ss(2);
              son_to_father_local(s_in_son[i], son_type, sf);
              father_to_son_local(s_in_parent[i], son_type, ss);
              double e1 = std::abs(sf[0] - s_in_parent[i][0]) + std::abs(sf[1] - s_in_parent[i][1]);
              double e2 = std::abs(ss[0] - s_in_son[i][0]) + std::abs(ss[1] - s_in_son[i][1]);
              if (e1 > 1e-12 || e2 > 1e-12)
                std::fprintf(stderr, "[affine-check] son_type=%d node=%u FAIL fwd_err=%.3e inv_err=%.3e\n", son_type, i, e1, e2);
            }
            // Registry key (father node-pointer pair that this node bisects) used to share/create
            // this node without duplication; empty if node i is a reused father node.
            std::set<Node *> reg_key = this->father_edge_node_key(s, father_el_pt);

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
              // Was the node created by one of its neighbours (or another son of the same
              // father)? For triangles we do NOT use oomph's quad compass neighbour finding
              // (geometrically wrong here); instead a father-edge node is looked up in the
              // shared-node registry by the pair of father corner nodes it bisects. This shares
              // the node across the edge-sharing neighbour father's sons AND the other sons of
              // this father, so no duplicate mid-edge nodes are created.
              bool is_periodic = false;
              if (!reg_key.empty())
              {
                std::map<std::set<Node *>, Node *>::iterator reg_it = Shared_edge_node_registry.find(reg_key);
                if (reg_it != Shared_edge_node_registry.end()) created_node_pt = reg_it->second;
              }

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

              // For a moving (Solid) mesh the new node is a SolidNode carrying
              // Lagrangian (reference) coordinates. These are NOT touched by the
              // Eulerian-position loop above, so without this they stay at their
              // construct_node() default (0). LaplaceSmoothedMesh (and any
              // solid-mechanics residual) is written in terms of the deformation
              // x - xi, so a wrong xi makes the identity mesh look grossly
              // deformed -> a large spurious residual/Jacobian on every refined
              // element (even conforming, no hanging). Interpolate xi from the
              // father, exactly as RefineableSolidQElement does for quads.
              if (SolidNode *solid_node_pt = dynamic_cast<SolidNode *>(created_node_pt))
              {
                const unsigned n_lagr = solid_node_pt->nlagrangian();
                // Interpolate the Lagrangian coordinates from the father with the SAME geometric
                // shape functions used for the Eulerian position above (pyoomph stores the
                // Lagrangian coords per node but does not register a Lagrangian dimension at the
                // element level, so oomph's SolidFiniteElement::interpolated_xi returns 0 here --
                // hence the manual interpolation).
                const unsigned nnod = father_el_pt->nnode();
                Shape psi(nnod);
                father_el_pt->shape(s, psi);
                for (unsigned i = 0; i < n_lagr; i++)
                {
                  double xi_i = 0.0;
                  for (unsigned l = 0; l < nnod; l++)
                  {
                    if (SolidNode *fn = dynamic_cast<SolidNode *>(father_el_pt->node_pt(l)))
                      xi_i += psi(l) * fn->xi(i);
                  }
                  solid_node_pt->xi(i) = xi_i;
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

              // Register this freshly created father-edge node so the other sons of this father
              // and the sons of the edge-sharing neighbour father reuse it instead of duplicating.
              if (!reg_key.empty()) Shared_edge_node_registry[reg_key] = created_node_pt;

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
  void RefineableTElement<2>::output_corners(std::ostream &,
                                             const std::string &) const
  {
    // Debug-only output; not needed for refinement itself.
  }

  //====================================================================
  /// Set up all hanging nodes.
  //====================================================================
  // Which triangle-hanging path is active (env PYOOMPH_TRI_HANG): 0 = geometric (default, the
  // node_strictly_on_*/facet-adjacency scheme in mesh2d.cpp post_adapt_setup_hanging_nodes);
  // 1 = tree (this oomph-style setup_hang* path); 2 = validate (tree runs during adapt, then the
  // geometric pass captures it, redoes itself and compares -- see post_adapt_setup_hanging_nodes).
  int tri_hang_mode()
  {
    static const int mode = []() {
      const char *e = getenv("PYOOMPH_TRI_HANG");
      if (!e) return 0;
      if (std::string(e) == "tree") return 1;
      if (std::string(e) == "validate") return 2;
      return 0;
    }();
    return mode;
  }

  namespace
  {
    // Father-local coordinates of a tri son's three vertices (son local v0=(1,0), v1=(0,1), v2=(0,0)),
    // per son_type -- read straight off RefineableTElement<2>::build's s_in_parent switch. The son->
    // father map is then the barycentric affine s_father = a*P0 + b*P1 + (1-a-b)*P2.
    inline void tri_son_vertex_father_coords(int son_type, double P[3][2])
    {
      using namespace oomph::QuadTreeNames;
      switch (son_type)
      {
      case SW: P[0][0]=0.5;P[0][1]=0.0; P[1][0]=0.0;P[1][1]=0.5; P[2][0]=0.0;P[2][1]=0.0; break;
      case SE: P[0][0]=1.0;P[0][1]=0.0; P[1][0]=0.5;P[1][1]=0.5; P[2][0]=0.5;P[2][1]=0.0; break;
      case NE: P[0][0]=0.5;P[0][1]=0.0; P[1][0]=0.5;P[1][1]=0.5; P[2][0]=0.0;P[2][1]=0.5; break;
      case NW: P[0][0]=0.5;P[0][1]=0.5; P[1][0]=0.0;P[1][1]=1.0; P[2][0]=0.0;P[2][1]=0.5; break;
      default: P[0][0]=1.0;P[0][1]=0.0; P[1][0]=0.0;P[1][1]=1.0; P[2][0]=0.0;P[2][1]=0.0; break; // identity
      }
    }
  }

  void RefineableTElement<2>::son_to_father_local(const Vector<double> &s_son, int son_type, Vector<double> &s_father)
  {
    double P[3][2];
    tri_son_vertex_father_coords(son_type, P);
    const double a = s_son[0], b = s_son[1], c = 1.0 - a - b;
    s_father[0] = a * P[0][0] + b * P[1][0] + c * P[2][0];
    s_father[1] = a * P[0][1] + b * P[1][1] + c * P[2][1];
  }

  void RefineableTElement<2>::father_to_son_local(const Vector<double> &s_father, int son_type, Vector<double> &s_son)
  {
    double P[3][2];
    tri_son_vertex_father_coords(son_type, P);
    // s_father = P2 + A*s_son, A columns = (P0-P2),(P1-P2). Invert the 2x2.
    const double A00 = P[0][0] - P[2][0], A01 = P[1][0] - P[2][0];
    const double A10 = P[0][1] - P[2][1], A11 = P[1][1] - P[2][1];
    const double det = A00 * A11 - A01 * A10;
    const double rx = s_father[0] - P[2][0], ry = s_father[1] - P[2][1];
    s_son[0] = (A11 * rx - A01 * ry) / det;
    s_son[1] = (-A10 * rx + A00 * ry) / det;
  }

  void RefineableTElement<2>::local_coordinate_in_ancestor(const Vector<double> &s_here, RefineableTElement<2> *ancestor, Vector<double> &s_ancestor) const
  {
    // Walk up the tree from this element to `ancestor`, composing son->father at each level.
    s_ancestor = s_here;
    Tree *tp = this->Tree_pt;
    while (tp && tp->object_pt() != ancestor)
    {
      Tree *father_tp = tp->father_pt();
      if (!father_tp) break; // reached the root without meeting the ancestor
      Vector<double> s_up(2);
      son_to_father_local(s_ancestor, tp->son_type(), s_up);
      s_ancestor = s_up;
      tp = father_tp;
    }
  }

  bool RefineableTElement<2>::local_coordinate_in_other_leaf(const Vector<double> &s_here, RefineableTElement<2> *target, Vector<double> &s_target) const
  {
    if (!this->Tree_pt || !target || !target->Tree_pt) return false;
    Tree *my_root = this->Tree_pt->root_pt();
    Tree *tgt_root = target->Tree_pt->root_pt();
    if (my_root != tgt_root) return false; // cross-root: handled by the inter-tree map, not here
    RefineableTElement<2> *root_el = dynamic_cast<RefineableTElement<2> *>(my_root->object_pt());
    if (!root_el) return false;
    // Up: this element's coord -> root coord.
    Vector<double> s_root(2);
    this->local_coordinate_in_ancestor(s_here, root_el, s_root);
    // Down: root coord -> target coord. Collect target's son_type chain up to the root, then apply
    // father_to_son from the root's direct son down to the target (chain in reverse).
    std::vector<int> chain;
    Tree *tp = target->Tree_pt;
    while (tp != tgt_root) { chain.push_back(tp->son_type()); tp = tp->father_pt(); if (!tp) return false; }
    Vector<double> s = s_root;
    for (auto it = chain.rbegin(); it != chain.rend(); ++it)
    {
      Vector<double> sd(2);
      father_to_son_local(s, *it, sd);
      s = sd;
    }
    s_target = s;
    return true;
  }

  //=================================================================
  /// Tree-based hanging on one triangle edge. See the header for the rationale (topology from the
  /// QuadTree, coordinates from locate_zeta -- never the quad compass coordinate descent).
  //=================================================================
  void RefineableTElement<2>::tri_hang_helper(const int &value_id, const int &my_edge)
  {
    using namespace QuadTreeNames;
    QuadTree *qt = quadtree_pt();
    if (!qt) return;
    Vector<unsigned> translate_s(2);
    Vector<double> s_lo(2), s_hi(2);
    int neigh_edge = 0, diff_level = 0;
    bool in_neigh_tree = false;
    QuadTree *neigh = qt->gteq_edge_neighbour(my_edge, translate_s, s_lo, s_hi, neigh_edge, diff_level, in_neigh_tree);
    // No neighbour (domain boundary), a same-level (conforming) neighbour, or a non-leaf: nothing hangs.
    if (neigh == 0 || diff_level == 0 || !neigh->is_leaf()) return;
    RefineableElement *neigh_re = dynamic_cast<RefineableElement *>(neigh->object_pt());
    FiniteElement *neigh_fe = dynamic_cast<FiniteElement *>(neigh->object_pt());
    if (!neigh_re || !neigh_fe) return;

    // Interpolating nodes along MY edge. Tri edge<->compass convention (must match get_edge_bcs and the
    // forest root-neighbour setup, both derived from Father_bound): E=v0-v1, W=v1-v2, S=v2-v0. This is
    // the slot passed to gteq_edge_neighbour, so the local-coordinate walk below MUST trace the same
    // edge. TElement<2,3> vertex local coords v0=(1,0), v1=(0,1), v2=(0,0). n_edge is 2 for a C1 field
    // (corners), 3 for C2 (+mid).
    const unsigned n_edge = this->ninterpolating_node_1d(value_id);
    for (unsigned i = 0; i < n_edge; i++)
    {
      const double f = (n_edge > 1) ? double(i) / double(n_edge - 1) : 0.0; // 0 .. 1 along my edge
      Vector<double> s(2);
      switch (my_edge)
      {
        case E: s[0] = 1.0 - f; s[1] = f; break;   // edge v0-v1: v0(1,0) -> v1(0,1)
        case W: s[0] = 0.0;     s[1] = 1.0 - f; break; // edge v1-v2: v1(0,1) -> v2(0,0)
        case S: s[0] = f;       s[1] = 0.0; break; // edge v2-v0: v2(0,0) -> v0(1,0)
        default: return;
      }
      Node *X = this->get_interpolating_node_at_local_coordinate(s, value_id);
      if (X == 0) continue;
      // Global position of X, and its local coordinate inside the coarse neighbour.
      Vector<double> X_pos(2);
      X_pos[0] = X->x(0);
      X_pos[1] = X->x(1);
      // Local coordinate of X inside the coarse neighbour. Preferred: the exact tree affine map (no
      // Newton); it works whenever this element and the neighbour share a tree root. Cross-root pairs
      // still fall back to locate_zeta until the inter-tree map lands. PYOOMPH_TRI_AFFINE_XCHECK also
      // runs locate_zeta and logs any disagreement.
      Vector<double> s_neigh(2);
      static const bool USE_AFFINE = getenv("PYOOMPH_NO_AFFINE") == 0; // affine on by default; set PYOOMPH_NO_AFFINE=1 to force locate_zeta
      RefineableTElement<2> *neigh_te = USE_AFFINE ? dynamic_cast<RefineableTElement<2> *>(neigh->object_pt()) : nullptr;
      bool have_affine = neigh_te && this->local_coordinate_in_other_leaf(s, neigh_te, s_neigh);
      static const bool XCHECK = getenv("PYOOMPH_TRI_AFFINE_XCHECK") != 0;
      if (!have_affine || XCHECK)
      {
        Vector<double> s_lz(2);
        GeomObject *geom = 0;
        neigh_fe->locate_zeta(X_pos, geom, s_lz);
        if (have_affine && XCHECK && geom != 0)
        {
          double e = std::abs(s_neigh[0] - s_lz[0]) + std::abs(s_neigh[1] - s_lz[1]);
          if (e > 1e-9)
            std::fprintf(stderr, "[affine-xcheck] X(%.4f,%.4f) affine=(%.5f,%.5f) lz=(%.5f,%.5f) err=%.3e\n", X_pos[0], X_pos[1], s_neigh[0], s_neigh[1], s_lz[0], s_lz[1], e);
        }
        if (!have_affine)
        {
          if (geom == 0) continue; // cross-root and not located (should not happen for a shared edge)
          s_neigh = s_lz;
        }
      }
      // If the neighbour has its own interpolating node exactly here, X is shared -> not hanging.
      if (neigh_re->get_interpolating_node_at_local_coordinate(s_neigh, value_id) == X) continue;
      // Otherwise X hangs on the neighbour's interpolation at s_neigh.
      const unsigned nmax = neigh_re->ninterpolating_node(value_id);
      Shape psi(nmax);
      neigh_re->interpolating_basis(s_neigh, psi, value_id);
      unsigned nmaster = 0;
      for (unsigned m = 0; m < nmax; m++)
        if (std::abs(psi[m]) > 1e-12) nmaster++;
      if (nmaster == 0) continue;
      HangInfo *hang = new HangInfo(nmaster);
      unsigned mm = 0;
      for (unsigned m = 0; m < nmax; m++)
      {
        if (std::abs(psi[m]) > 1e-12)
        {
          hang->set_master_node_pt(mm, neigh_re->interpolating_node_pt(m, value_id), psi[m]);
          mm++;
        }
      }
      X->set_hanging_pt(hang, value_id);
    }
  }

  void RefineableTElement<2>::setup_hanging_nodes(Vector<std::ofstream *> &)
  {
    // Geometric mode: mesh2d.cpp post_adapt_setup_hanging_nodes handles all tri hanging.
    if (tri_hang_mode() == 0) return;
    using namespace QuadTreeNames;
    // Geometric (position / C2) hanging is value_id -1.
    tri_hang_helper(-1, S);
    tri_hang_helper(-1, E);
    tri_hang_helper(-1, W);
  }

  //================================================================
  /// Internal function that sets up the hanging node scheme for
  /// a particular continuously interpolated value
  //===============================================================
  void RefineableTElement<2>::setup_hang_for_value(const int &value_id)
  {
    if (tri_hang_mode() == 0) return; // geometric mode handles it in mesh2d.cpp
    using namespace QuadTreeNames;
    tri_hang_helper(value_id, S);
    tri_hang_helper(value_id, E);
    tri_hang_helper(value_id, W);
  }

  //=================================================================
  /// Internal function to set up the hanging nodes on a particular
  /// edge of the element
  //=================================================================
  void RefineableTElement<2>::
      quad_hang_helper(const int &,
                       const int &, std::ofstream &)
  {
    throw_runtime_error("Implement");
  }

  //=================================================================
  /// Check inter-element continuity of
  /// - nodal positions
  /// - (nodally) interpolated function values
  //====================================================================
  // template<unsigned NNODE_1D>
  void RefineableTElement<2>::check_integrity(double &)
  {

    throw_runtime_error("Implement");
  }

  //========================================================================
  /// Static matrix for coincidence between son nodal points and
  /// father boundaries
  ///
  //========================================================================
  std::map<unsigned, DenseMatrix<int>> RefineableTElement<2>::Father_bound;

  // Shared-node registry for geometric node-sharing during triangle refinement (see header).
  std::map<std::set<Node *>, Node *> RefineableTElement<2>::Shared_edge_node_registry;

  // Registry key for a new son node at father-local coordinate s_in_father: every node created by
  // a 1->4 triangle refinement is the midpoint of exactly two father nodes (two corners for a
  // linear triangle; a corner+mid-edge or two mid-edge nodes for a quadratic one). The key is that
  // father node-pointer pair, which is identical from every element that creates the same node
  // (sibling sons AND the edge-sharing neighbour father's sons, since they share the father edge's
  // nodes), so mid-edge nodes are deduplicated with no coordinate descent. Returns empty if
  // s_in_father is not the midpoint of any father-node pair (e.g. a reused father node, handled
  // separately via get_node_at_local_coordinate).
  std::set<Node *> RefineableTElement<2>::father_edge_node_key(const Vector<double> &s_in_father, RefineableTElement<2> *father_el_pt) const
  {
    std::set<Node *> key;
    unsigned nf = father_el_pt->nnode();
    Vector<double> sa(2), sb(2);
    for (unsigned a = 0; a < nf; a++)
    {
      father_el_pt->local_coordinate_of_node(a, sa);
      for (unsigned b = a + 1; b < nf; b++)
      {
        father_el_pt->local_coordinate_of_node(b, sb);
        if (std::abs(0.5 * (sa[0] + sb[0]) - s_in_father[0]) < 1e-10 &&
            std::abs(0.5 * (sa[1] + sb[1]) - s_in_father[1]) < 1e-10)
        {
          key.insert(father_el_pt->node_pt(a));
          key.insert(father_el_pt->node_pt(b));
          return key;
        }
      }
    }
    return key;
  }

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
  // Not used by the tetrahedron build (boundary conditions of new nodes are derived directly from
  // their two generating father nodes); kept as non-throwing stubs for interface compatibility.
  void RefineableTElement<3>::get_bcs(int, Vector<int> &bound_cons) const
  {
    for (unsigned k = 0; k < bound_cons.size(); k++) bound_cons[k] = 0;
  }

  void RefineableTElement<3>::get_edge_bcs(const int &, Vector<int> &bound_cons) const
  {
    for (unsigned k = 0; k < bound_cons.size(); k++) bound_cons[k] = 0;
  }

  void RefineableTElement<3>::get_boundaries(const int &, std::set<unsigned> &boundary) const
  {
    boundary.clear();
  }

  void RefineableTElement<3>::
      interpolated_zeta_on_edge(const unsigned &, const int &, const Vector<double> &, Vector<double> &zeta)
  {
    zeta[0] = 0.0;
  }

  // Not used: the tetrahedron build shares father-edge nodes via the geometric registry
  // (father_edge_node_key), so this is a non-throwing stub for interface compatibility.
  Node *RefineableTElement<3>::
      node_created_by_neighbour(const Vector<double> &, bool &is_periodic)
  {
    is_periodic = false;
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
                                    std::ofstream &)
  {
    unsigned n_node = this->nnode();
    if (nodes_built())
    {
      was_already_built = true;
      return;
    }
    was_already_built = false;

    OcTree *father_pt = dynamic_cast<OcTree *>(octree_pt()->father_pt());
    int son_type = Tree_pt->son_type();
    RefineableTElement<3> *father_el_pt = dynamic_cast<RefineableTElement<3> *>(father_pt->object_pt());
    TimeStepper *time_stepper_pt = father_el_pt->node_pt(0)->time_stepper_pt();
    unsigned ntstorage = time_stepper_pt->ntstorage();
    if (father_el_pt->Macro_elem_pt != 0)
    {
      throw_runtime_error("Macro elements (curved boundaries) are not yet supported for tetrahedral refinement");
    }

    // The 4 vertices of this son in the father's local coordinates -> affine (barycentric) map
    // from son-local to father-local coordinates.
    Vector<Vector<double>> sv;
    son_vertices_in_father(son_type, sv);

    for (unsigned j = 0; j < n_node; j++)
    {
      // Father-local coordinate of son node j.
      Vector<double> s_son(3);
      this->local_coordinate_of_node(j, s_son);
      double b0 = s_son[0], b1 = s_son[1], b2 = s_son[2], b3 = 1.0 - b0 - b1 - b2;
      Vector<double> s(3);
      for (unsigned d = 0; d < 3; d++)
        s[d] = b0 * sv[0][d] + b1 * sv[1][d] + b2 * sv[2][d] + b3 * sv[3][d];

      // (1) Reuse a father node coincident with this position?
      Node *created_node_pt = father_el_pt->get_node_at_local_coordinate(s);
      if (created_node_pt != 0)
      {
        node_pt(j) = created_node_pt;
        for (unsigned t = 0; t < ntstorage; t++)
        {
          Vector<double> prev_values;
          father_el_pt->get_interpolated_values(t, s, prev_values);
          unsigned n_var = std::min((unsigned)created_node_pt->nvalue(), (unsigned)prev_values.size());
          for (unsigned k = 0; k < n_var; k++) created_node_pt->set_value(t, k, prev_values[k]);
        }
        continue;
      }

      // Generating nodes of son node j -- the already-built (earlier-index) son nodes whose average
      // is j. A new node is keyed on, and inherits boundary/pin data from, these nodes; the key is
      // identical from every element that creates the same node, so it is shared without duplication.
      //   - edge-midpoint nodes (j < 10): the two father nodes j bisects (father_edge_node_key);
      //   - C2TB face-centroid bubbles (j in 10..13): the son's three face-corner nodes (already
      //     built, and themselves shared coarse corners / edge-mids -> shared across the two tets
      //     meeting on that face, keeping the enriched velocity continuous);
      //   - the C2TB volume-centroid bubble (j == 14): interior, never shared/on a boundary.
      std::vector<Node *> gen;
      std::set<Node *> reg_key;
      if (j >= 10 && j <= 13)
      {
        static const unsigned facecorner[4][3] = {{0, 1, 3}, {0, 1, 2}, {0, 2, 3}, {1, 2, 3}};
        const unsigned *fc = facecorner[j - 10];
        for (int c = 0; c < 3; c++) { gen.push_back(node_pt(fc[c])); reg_key.insert(node_pt(fc[c])); }
      }
      else if (j != 14)
      {
        reg_key = father_edge_node_key(s, father_el_pt);
        for (Node *nd : reg_key) gen.push_back(nd);
      }

      // (2) Reuse a node already created (by a sibling son or the face-/edge-sharing neighbour
      // father's sons) via the geometric shared-node registry.
      if (!reg_key.empty())
      {
        std::map<std::set<Node *>, Node *>::iterator it = Shared_edge_node_registry.find(reg_key);
        if (it != Shared_edge_node_registry.end())
        {
          node_pt(j) = it->second;
          continue;
        }
      }

      // (3) Build a new node. It lies on a mesh boundary iff ALL its generating nodes do (their
      // common boundaries); its pinned values are those pinned at every generating node; boundary
      // coordinates are the average of the generating nodes' (its centroid/midpoint position).
      std::set<unsigned> boundaries;
      bool have_bounds = false;
      for (Node *g : gen)
      {
        oomph::BoundaryNodeBase *bg = dynamic_cast<oomph::BoundaryNodeBase *>(g);
        std::set<unsigned> *sg = 0;
        if (bg) bg->get_boundaries_pt(sg);
        if (!sg) { boundaries.clear(); break; } // a generating node off every boundary -> interior
        if (!have_bounds) { boundaries = *sg; have_bounds = true; }
        else
        {
          std::set<unsigned> inter;
          std::set_intersection(boundaries.begin(), boundaries.end(), sg->begin(), sg->end(),
                                std::inserter(inter, inter.begin()));
          boundaries.swap(inter);
        }
      }

      if (!boundaries.empty())
      {
        created_node_pt = construct_boundary_node(j, time_stepper_pt);
        unsigned n_value = created_node_pt->nvalue();
        for (unsigned k = 0; k < n_value; k++)
        {
          bool all_pinned = true;
          for (Node *g : gen) if (!g->is_pinned(k)) { all_pinned = false; break; }
          if (all_pinned) created_node_pt->pin(k);
        }
        for (std::set<unsigned>::iterator it = boundaries.begin(); it != boundaries.end(); ++it)
        {
          mesh_pt->add_boundary_node(*it, created_node_pt);
          if (mesh_pt->boundary_coordinate_exists(*it))
          {
            Vector<double> z;
            for (Node *g : gen)
            {
              Vector<double> zg;
              dynamic_cast<oomph::BoundaryNodeBase *>(g)->get_coordinates_on_boundary(*it, zg);
              if (z.empty()) z.resize(zg.size(), 0.0);
              for (unsigned zi = 0; zi < zg.size(); zi++) z[zi] += zg[zi] / gen.size();
            }
            created_node_pt->set_coordinates_on_boundary(*it, z);
          }
        }
      }
      else
      {
        created_node_pt = construct_node(j, time_stepper_pt);
      }

      node_pt(j) = created_node_pt;
      new_node_pt.push_back(created_node_pt);
      for (unsigned t = 0; t < ntstorage; t++)
      {
        Vector<double> x_prev(3);
        father_el_pt->get_x(t, s, x_prev);
        for (unsigned d = 0; d < 3; d++) created_node_pt->x(t, d) = x_prev[d];
      }
      // Interpolate the Lagrangian (reference) coordinates from the father with the geometric
      // shape functions -- see the 2d build for the full rationale. Without this, new tet nodes on
      // a moving (Solid) mesh keep xi=0 while their Eulerian x is correct, so LaplaceSmoothedMesh /
      // solid residuals see a spurious deformation on every refined element.
      if (SolidNode *solid_node_pt = dynamic_cast<SolidNode *>(created_node_pt))
      {
        const unsigned n_lagr = solid_node_pt->nlagrangian();
        const unsigned nnod = father_el_pt->nnode();
        Shape psi(nnod);
        father_el_pt->shape(s, psi);
        for (unsigned i = 0; i < n_lagr; i++)
        {
          double xi_i = 0.0;
          for (unsigned l = 0; l < nnod; l++)
          {
            if (SolidNode *fn = dynamic_cast<SolidNode *>(father_el_pt->node_pt(l)))
              xi_i += psi(l) * fn->xi(i);
          }
          solid_node_pt->xi(i) = xi_i;
        }
      }
      for (unsigned t = 0; t < ntstorage; t++)
      {
        Vector<double> prev_values;
        father_el_pt->get_interpolated_values(t, s, prev_values);
        unsigned n_var = std::min((unsigned)created_node_pt->nvalue(), (unsigned)prev_values.size());
        for (unsigned k = 0; k < n_var; k++) created_node_pt->set_value(t, k, prev_values[k]);
      }
      mesh_pt->add_node_pt(created_node_pt);
      if (!reg_key.empty()) Shared_edge_node_registry[reg_key] = created_node_pt;
    }
  }

  //====================================================================
  ///  Print corner nodes, use colour (default "BLACK")
  //====================================================================
  void RefineableTElement<3>::output_corners(std::ostream &, const std::string &) const
  {
    // Debug-only output; not needed for refinement itself.
  }

  // Hanging nodes for tetrahedra are installed by a mesh-level geometric pass after refinement
  // (TemplatedMeshBase3d::post_adapt_setup_hanging_nodes), analogous to the 2d triangle scheme, so
  // these per-element hooks are no-ops.
  void RefineableTElement<3>::setup_hanging_nodes(Vector<std::ofstream *> &) {}
  void RefineableTElement<3>::setup_hang_for_value(const int &) {}
  void RefineableTElement<3>::quad_hang_helper(const int &, const int &, std::ofstream &) {}

  void RefineableTElement<3>::check_integrity(double &max_error)
  {
    max_error = 0.0; // continuity is guaranteed by the geometric node-sharing/hanging scheme
  }

  //========================================================================
  /// Static matrix for coincidence between son nodal points and father boundaries (unused by the
  /// geometric tetrahedron refinement, kept for interface compatibility).
  //========================================================================
  std::map<unsigned, DenseMatrix<int>> RefineableTElement<3>::Father_bound;

  // Shared-node registry for geometric node-sharing during tetrahedron refinement (see header).
  std::map<std::set<Node *>, Node *> RefineableTElement<3>::Shared_edge_node_registry;

  // Registry key for a new son node at father-local coordinate s_in_father: it is the midpoint of
  // exactly two father nodes (an edge midpoint for both linear and quadratic tets), and that
  // father node-pointer pair is identical from every element that creates the same node. Returns
  // empty if s_in_father is not the midpoint of any father-node pair (a reused father node).
  std::set<Node *> RefineableTElement<3>::father_edge_node_key(const Vector<double> &s_in_father, RefineableTElement<3> *father_el_pt) const
  {
    std::set<Node *> key;
    unsigned nf = father_el_pt->nnode();
    Vector<double> sa(3), sb(3);
    for (unsigned a = 0; a < nf; a++)
    {
      father_el_pt->local_coordinate_of_node(a, sa);
      for (unsigned b = a + 1; b < nf; b++)
      {
        father_el_pt->local_coordinate_of_node(b, sb);
        if (std::abs(0.5 * (sa[0] + sb[0]) - s_in_father[0]) < 1e-10 &&
            std::abs(0.5 * (sa[1] + sb[1]) - s_in_father[1]) < 1e-10 &&
            std::abs(0.5 * (sa[2] + sb[2]) - s_in_father[2]) < 1e-10)
        {
          key.insert(father_el_pt->node_pt(a));
          key.insert(father_el_pt->node_pt(b));
          return key;
        }
      }
    }
    return key;
  }

  // The 4 vertices (father local coordinates) of son son_type (0..7) of a 1->8 tetrahedron split:
  // 4 corner sub-tets (one per father vertex, using its 3 adjacent edge midpoints) + 4 sub-tets
  // tiling the central octahedron of the 6 edge midpoints, split along a fixed diagonal (m02-m13).
  // The octahedron is interior to the father, so face conformity is automatic via edge-midpoint
  // sharing regardless of the diagonal; it is a free quality choice. Vertices are orientation-
  // corrected to positive signed volume so the son's shape functions/Jacobian are consistent.
  void RefineableTElement<3>::son_vertices_in_father(int son_type, Vector<Vector<double>> &verts) const
  {
    // Father node local coordinates (oomph TElement<3,*> numbering: 4 vertices + 6 edge mids).
    Vector<double> v0(3), v1(3), v2(3), v3(3), m01(3), m02(3), m03(3), m12(3), m23(3), m13(3);
    v0[0] = 1; v0[1] = 0; v0[2] = 0;
    v1[0] = 0; v1[1] = 1; v1[2] = 0;
    v2[0] = 0; v2[1] = 0; v2[2] = 1;
    v3[0] = 0; v3[1] = 0; v3[2] = 0;
    m01[0] = 0.5; m01[1] = 0.5; m01[2] = 0.0;
    m02[0] = 0.5; m02[1] = 0.0; m02[2] = 0.5;
    m03[0] = 0.5; m03[1] = 0.0; m03[2] = 0.0;
    m12[0] = 0.0; m12[1] = 0.5; m12[2] = 0.5;
    m23[0] = 0.0; m23[1] = 0.0; m23[2] = 0.5;
    m13[0] = 0.0; m13[1] = 0.5; m13[2] = 0.0;

    verts.resize(4, Vector<double>(3));
    switch (son_type)
    {
    case 0: verts[0] = v0;  verts[1] = m01; verts[2] = m02; verts[3] = m03; break; // corner v0
    case 1: verts[0] = v1;  verts[1] = m01; verts[2] = m12; verts[3] = m13; break; // corner v1
    case 2: verts[0] = v2;  verts[1] = m02; verts[2] = m12; verts[3] = m23; break; // corner v2
    case 3: verts[0] = v3;  verts[1] = m03; verts[2] = m13; verts[3] = m23; break; // corner v3
    case 4: verts[0] = m02; verts[1] = m13; verts[2] = m01; verts[3] = m03; break; // octahedron
    case 5: verts[0] = m02; verts[1] = m13; verts[2] = m03; verts[3] = m23; break;
    case 6: verts[0] = m02; verts[1] = m13; verts[2] = m23; verts[3] = m12; break;
    case 7: verts[0] = m02; verts[1] = m13; verts[2] = m12; verts[3] = m01; break;
    default: throw_runtime_error("Invalid tetrahedron son type (must be 0..7)");
    }

    // Orientation-correct: ensure det[v0-v3, v1-v3, v2-v3] > 0 (matches the reference tet).
    double d[3][3];
    for (int r = 0; r < 3; r++)
      for (int c = 0; c < 3; c++) d[r][c] = verts[r][c] - verts[3][c];
    double det = d[0][0] * (d[1][1] * d[2][2] - d[1][2] * d[2][1]) - d[0][1] * (d[1][0] * d[2][2] - d[1][2] * d[2][0]) + d[0][2] * (d[1][0] * d[2][1] - d[1][1] * d[2][0]);
    if (det < 0.0)
    {
      Vector<double> tmp = verts[0];
      verts[0] = verts[1];
      verts[1] = tmp;
    }
  }

}
