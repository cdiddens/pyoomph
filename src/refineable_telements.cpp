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
#include "elements.hpp" // for pyoomph::BulkElementBase::mixed_hang_edge_node (cross-shape mixed-mesh hanging)
#include "exception.hpp"
#include <functional>
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

      // If the father is defined via a macro element (curvilinear boundary representation), the son
      // inherits it along with its own region of the macro reference domain. A triangular son is not
      // an axis-aligned sub-box of its father, so oomph's s_macro_ll/s_macro_ur cannot express this;
      // the son's three vertices in father coordinates -- s_in_parent[0..2], already computed above --
      // can, and are what BulkElementBase::macro_coordinate_from_local interpolates. New nodes below
      // are positioned by father_el_pt->get_x(t, s, ...), which routes through it.
      if (father_el_pt->Macro_elem_pt != 0)
      {
        pyoomph::BulkElementBase *son_be = dynamic_cast<pyoomph::BulkElementBase *>(this);
        pyoomph::BulkElementBase *father_be = dynamic_cast<pyoomph::BulkElementBase *>(father_el_pt);
        if (son_be && father_be)
        {
          std::vector<std::vector<double>> son_vertices(3, std::vector<double>(2, 0.0));
          for (unsigned int v = 0; v < 3; v++)
            for (unsigned int i = 0; i < 2; i++) son_vertices[v][i] = s_in_parent[v][i];
          son_be->inherit_macro_element_from_father(father_be, son_vertices);
        }
        else
        {
          set_macro_elem_pt(father_el_pt->Macro_elem_pt);
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
              // First, the oomph-quad way: reuse a node a NEIGHBOUR already holds at this point, found via
              // the tri tree (which persists across adaptation rounds). This is what closes the cross-round
              // gap -- during transient re-adaptation a son built this round must reuse the node a
              // neighbour built in an EARLIER round, else the shared vertex is duplicated and the moving
              // mesh tears apart. The search covers this element's whole tree root plus the adjacent
              // root(s) and does not care what LEVEL the holder is at or whether it is still a leaf, so it
              // also covers a neighbour that is finer than this element, and one that was split earlier in
              // this very round. It is purely topological -- no node is ever identified by its position.
              created_node_pt = this->node_created_by_neighbour(s_fraction);
              // Then the per-round registry, which dedupes nodes built within THIS round against a father
              // node-pointer pair: sibling sons, and sons of the edge-sharing neighbour father, that the
              // tree search above cannot see yet because neither side has built its nodes.
              if (created_node_pt == 0 && !reg_key.empty())
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
  // Triangle hanging is done per-element here (setup_hanging_nodes / setup_hang_for_value ->
  // tri_hang_helper), inside oomph's adapt_mesh, exactly as RefineableQElement does for quads. The
  // legacy mesh-level geometric facet-adjacency pass (mesh2d.cpp post_adapt_setup_hanging_nodes) was
  // removed once this tree route was validated to match it (machine-zero) across the full tri suite.

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

  namespace
  {
    // Symbolic "father points": a tri son's vertex always lands either on a father VERTEX (FV0/FV1/FV2)
    // or on a father EDGE MIDPOINT (FmS/FmE/FmW), per son_type. Read straight off build()'s s_in_parent.
    enum FatherPoint { FV0 = 0, FV1 = 1, FV2 = 2, FmS = 3, FmE = 4, FmW = 5 };

    // son_vertex_fp[son_type][local_vertex 0..2] -> FatherPoint. son_type is QuadTreeNames SW/SE/NW/NE = 0/1/2/3.
    inline int son_vertex_fp(int son_type, int v)
    {
      using namespace oomph::QuadTreeNames;
      static const int tab[4][3] = {
          /*SW*/ {FmS, FmW, FV2},
          /*SE*/ {FV0, FmE, FmS},
          /*NW*/ {FmE, FV1, FmW},
          /*NE*/ {FmS, FmE, FmW}};
      if (son_type < 0 || son_type > 3) return -1;
      return tab[son_type][v];
    }

    // The three father points on each father edge (S = v2-v0, E = v0-v1, W = v1-v2).
    inline bool fp_on_edge(int fp, int edge)
    {
      using namespace oomph::QuadTreeNames;
      switch (edge)
      {
      case S: return fp == FV2 || fp == FmS || fp == FV0;
      case E: return fp == FV0 || fp == FmE || fp == FV1;
      case W: return fp == FV1 || fp == FmW || fp == FV2;
      default: return false;
      }
    }

    // If both father points lie on ONE father edge, return that edge (S/E/W); else -1 (edge is interior).
    inline int father_edge_of_pair(int fpa, int fpb)
    {
      using namespace oomph::QuadTreeNames;
      for (int e : {S, E, W})
        if (fp_on_edge(fpa, e) && fp_on_edge(fpb, e)) return e;
      return -1;
    }

    // Local vertex indices bounding a triangle edge (E=v0-v1, W=v1-v2, S=v2-v0). Ordered so a=param 0, b=param 1.
    inline void edge_local_vertices(int edge, int &a, int &b)
    {
      using namespace oomph::QuadTreeNames;
      switch (edge)
      {
      case E: a = 0; b = 1; break;
      case W: a = 1; b = 2; break;
      case S: a = 2; b = 0; break;
      default: a = -1; b = -1; break;
      }
    }

    // Inverse of edge_local_vertices: the edge (S/E/W) bounded by the unordered vertex pair {a,b}, else -1.
    inline int edge_from_local_vertices(int a, int b)
    {
      using namespace oomph::QuadTreeNames;
      int lo = a < b ? a : b, hi = a < b ? b : a;
      if (lo == 0 && hi == 1) return E;
      if (lo == 1 && hi == 2) return W;
      if (lo == 0 && hi == 2) return S;
      return -1;
    }

    // An interior father edge connects two father-edge-midpoints and is shared by NE and exactly one
    // corner son. Given the current son_type and the two (midpoint) father points, return the sibling.
    inline int interior_sibling(int son_type, int fpa, int fpb)
    {
      using namespace oomph::QuadTreeNames;
      int m0 = fpa < fpb ? fpa : fpb, m1 = fpa < fpb ? fpb : fpa;
      int corner;
      if (m0 == FmS && m1 == FmW) corner = SW;      // {FmS,FmW}
      else if (m0 == FmS && m1 == FmE) corner = SE; // {FmS,FmE}
      else if (m0 == FmE && m1 == FmW) corner = NW; // {FmE,FmW}
      else return -1;
      return (son_type == NE) ? corner : NE;
    }

    // Which son covers the given HALF (0 = param [0,0.5], 1 = [0.5,1]) of a father boundary edge. Only
    // corner (like-oriented) sons touch a father boundary edge, and the edge direction is preserved.
    inline int son_on_edge_half(int edge, int half)
    {
      using namespace oomph::QuadTreeNames;
      switch (edge)
      {
      case S: return half == 0 ? SW : SE; // S: v2..mid = SW, mid..v0 = SE
      case E: return half == 0 ? SE : NW; // E: v0..mid = SE, mid..v1 = NW
      case W: return half == 0 ? NW : SW; // W: v1..mid = NW, mid..v2 = SW
      default: return -1;
      }
    }

    // Local coordinate of the point at parameter t (0..1) along edge (S/E/W) in a tri's local frame.
    inline void point_on_edge(int edge, double t, oomph::Vector<double> &s)
    {
      using namespace oomph::QuadTreeNames;
      switch (edge)
      {
      case S: s[0] = t;       s[1] = 0.0;     break; // v2(0,0)->v0(1,0)
      case E: s[0] = 1.0 - t; s[1] = t;       break; // v0(1,0)->v1(0,1)
      case W: s[0] = 0.0;     s[1] = 1.0 - t; break; // v1(0,1)->v2(0,0)
      default: s[0] = 0.0; s[1] = 0.0; break;
      }
    }

    // Parameter t (0..1) of a local coordinate lying on edge (S/E/W).
    inline double param_on_edge(int edge, const oomph::Vector<double> &s)
    {
      using namespace oomph::QuadTreeNames;
      switch (edge)
      {
      case S: return s[0];         // (t,0)
      case E: return s[1];         // (1-t,t)
      case W: return 1.0 - s[1];   // (0,1-t)
      default: return 0.0;
      }
    }
  } // namespace

  bool RefineableTElement<2>::root_coord_to_leaf(const Vector<double> &s_root, RefineableTElement<2> *leaf, Vector<double> &s_leaf)
  {
    if (!leaf || !leaf->Tree_pt) return false;
    Tree *root = leaf->Tree_pt->root_pt();
    std::vector<int> chain; // son_types from just-below-root down to leaf
    Tree *tp = leaf->Tree_pt;
    while (tp != root) { chain.push_back(tp->son_type()); tp = tp->father_pt(); if (!tp) return false; }
    Vector<double> s = s_root;
    for (auto it = chain.rbegin(); it != chain.rend(); ++it)
    {
      Vector<double> sd(2);
      father_to_son_local(s, *it, sd);
      s = sd;
    }
    s_leaf = s;
    return true;
  }

  // See declaration. Walks a subtree, testing every BUILT element for a node at the given local point and
  // recursing into every son that contains it. Nothing here looks at a physical position: containment is a
  // barycentric test on the affine son<->father map, and the node test compares LOCAL coordinates.
  Node *RefineableTElement<2>::node_in_subtree_at_local_coordinate(Tree *node, const Vector<double> &s)
  {
    if (!node) return 0;
    FiniteElement *fe = dynamic_cast<FiniteElement *>(node->object_pt());
    RefineableElement *re = dynamic_cast<RefineableElement *>(node->object_pt());
    // An element whose nodes are not built yet has a null node_pt array -- skip it (its already-built
    // ancestor was tested on the way down and holds the same node).
    if (fe && re && re->nodes_built())
    {
      if (Node *n = fe->get_node_at_local_coordinate(s)) return n;
    }
    if (node->is_leaf()) return 0;
    const double tol = 1e-9;
    for (int c = 0; c < 4; c++) // tri 1->4 sons, son_type 0..3
    {
      Tree *ch = node->son_pt(c);
      if (!ch) continue;
      Vector<double> sc(2);
      father_to_son_local(s, c, sc);
      const double b = 1.0 - sc[0] - sc[1];
      if (sc[0] < -tol || sc[1] < -tol || b < -tol) continue; // point is outside this son
      // No early break: a point on the boundary between two sons is contained in both, and only one of
      // them may hold the node (the other side can be coarser).
      if (Node *n = node_in_subtree_at_local_coordinate(ch, sc)) return n;
    }
    return 0;
  }

  Node *RefineableTElement<2>::node_created_by_neighbour(const Vector<double> &s_son) const
  {
    using namespace QuadTreeNames;
    if (!this->Tree_pt) return 0;
    TreeRoot *root = this->Tree_pt->root_pt();
    RefineableTElement<2> *my_root = dynamic_cast<RefineableTElement<2> *>(root->object_pt());
    if (!my_root) return 0;

    // The point, once and for all, in this element's own ROOT frame. Exact (the affine son/father maps are
    // exact by construction, refinement being defined in local coordinates), and it is the frame in which
    // both the same-root search and the cross-root hop are expressed.
    Vector<double> s_root(2);
    this->local_coordinate_in_ancestor(s_son, my_root, s_root);

    // (1) Anything inside my own root: siblings, cousins, uncles -- at any level, leaf or not.
    if (Node *n = node_in_subtree_at_local_coordinate(root, s_root)) return n;

    // (2) Root edges the point lies on (S=v2-v0 at y=0, W=v1-v2 at x=0, E=v0-v1 at x+y=1). A point strictly
    // inside the root has none and is done; an edge point has one; a root vertex has two.
    const double tol = 1e-12;
    int es[2], ne = 0;
    if (std::abs(s_root[1]) < tol) es[ne++] = S;
    if (std::abs(s_root[0]) < tol) es[ne++] = W;
    if (std::abs(s_root[0] + s_root[1] - 1.0) < tol) es[ne++] = E;
    for (int k = 0; k < ne; k++)
    {
      const int my_D = es[k];
      TreeRoot *nr = root->neighbour_pt(my_D);
      if (!nr) continue; // domain boundary
      const double t = param_on_edge(my_D, s_root);

      RefineableTElement<2> *nr_root_el = dynamic_cast<RefineableTElement<2> *>(nr->object_pt());
      if (nr_root_el)
      {
        // Tri neighbour root: match the shared edge by its two VERTEX NODE POINTERS (pure topology), which
        // also tells us whether the neighbour parametrises the edge in the opposite sense.
        int ea, eb;
        edge_local_vertices(my_D, ea, eb);
        Node *na = my_root->vertex_node_pt(ea);
        Node *nbv = my_root->vertex_node_pt(eb);
        int ca = -1, cb = -1;
        for (int j = 0; j < 3; j++)
        {
          Node *vj = nr_root_el->vertex_node_pt(j);
          if (vj == na) ca = j;
          if (vj == nbv) cb = j;
        }
        if (ca < 0 || cb < 0) continue; // roots not actually edge-sharing (should not happen)
        const int nr_D = edge_from_local_vertices(ca, cb);
        if (nr_D < 0) continue;
        int nca, ncb;
        edge_local_vertices(nr_D, nca, ncb);
        const double tn = (nca != ca) ? 1.0 - t : t;
        Vector<double> s_nrroot(2);
        point_on_edge(nr_D, tn, s_nrroot);
        if (Node *n = node_in_subtree_at_local_coordinate(nr, s_nrroot)) return n;
      }
      else
      {
        // Cross-SHAPE neighbour root (a QUAD at a mixed quad+tri interface): the tri son tables do not
        // apply, so the shared-edge parameter is carried over by blending the quad-root LOCAL coordinates
        // of the two shared corner NODES -- again node pointers and local coordinates only.
        pyoomph::BulkElementBase *qroot = dynamic_cast<pyoomph::BulkElementBase *>(nr->object_pt());
        FiniteElement *qroot_fe = dynamic_cast<FiniteElement *>(nr->object_pt());
        RefineableElement *qroot_re = dynamic_cast<RefineableElement *>(nr->object_pt());
        if (!qroot || !qroot_fe || !qroot_re || !qroot_re->nodes_built()) continue;
        int ea, eb;
        edge_local_vertices(my_D, ea, eb);
        const int iP = qroot_fe->get_node_number(my_root->vertex_node_pt(ea)); // t=0
        const int iQ = qroot_fe->get_node_number(my_root->vertex_node_pt(eb)); // t=1
        if (iP < 0 || iQ < 0) continue;
        Vector<double> sP(2), sQ(2), s_qroot(2);
        qroot_fe->local_coordinate_of_node(iP, sP);
        qroot_fe->local_coordinate_of_node(iQ, sQ);
        for (int d = 0; d < 2; d++) s_qroot[d] = (1.0 - t) * sP[d] + t * sQ[d];
        if (Node *n = qroot->quad_node_at_root_coordinate(s_qroot)) return n;
      }
    }
    return 0;
  }

  // See declaration. The coordinate machinery mirrors node_created_by_neighbour() (which computes exactly
  // "this node's local coordinate in the edge neighbour"), but here we REGISTER our edge nodes on a strictly
  // coarser neighbour for the tesselated-numpy hanging-node export instead of reusing a coincident node.
  void RefineableTElement<2>::tess_register_on_coarser_for_numpy(std::vector<std::vector<std::set<Node *>>> &add_nodes)
  {
    using namespace QuadTreeNames;
    pyoomph::BulkElementBase *me = dynamic_cast<pyoomph::BulkElementBase *>(this);
    if (!me || !this->Tree_pt)
      return;
    RefineableTElement<2> *my_root = dynamic_cast<RefineableTElement<2> *>(this->Tree_pt->root_pt()->object_pt());
    const int compass[3] = {S, E, W};
    for (int ci = 0; ci < 3; ci++)
    {
      TriEdgeNeighbour nb = this->tri_edge_neighbour(compass[ci]);
      // A coarser tri neighbour (nb.el, diff_level<0) or ANY cross-root quad neighbour (cross_shape_root):
      // we descend the quad to the leaf at our edge point and let the pointer-identity + reference-edge tests
      // in the coarse element's add_node discard equal/finer (2:1-balanced interface) neighbours.
      RefineableTElement<2> *coarse_tri = (nb.el && nb.diff_level < 0 && nb.el != this) ? nb.el : nullptr;
      pyoomph::BulkElementBase *qroot = nullptr;
      FiniteElement *qroot_fe = nullptr;
      if (!coarse_tri && nb.cross_shape_root)
      {
        qroot = dynamic_cast<pyoomph::BulkElementBase *>(nb.cross_shape_root);
        qroot_fe = dynamic_cast<FiniteElement *>(nb.cross_shape_root);
      }
      if (!coarse_tri && !qroot)
        continue;
      for (unsigned li = 0; li < me->nnode(); li++)
      {
        Vector<double> s_here(2);
        me->local_coordinate_of_node(li, s_here);
        bool on_edge;
        if (compass[ci] == S)
          on_edge = std::abs(s_here[1]) < 1e-9;
        else if (compass[ci] == W)
          on_edge = std::abs(s_here[0]) < 1e-9;
        else // E
          on_edge = std::abs(s_here[0] + s_here[1] - 1.0) < 1e-9;
        if (!on_edge)
          continue;
        Vector<double> s_coarse(2);
        if (coarse_tri)
        {
          bool ok;
          if (!nb.cross_root)
            ok = this->local_coordinate_in_other_leaf(s_here, coarse_tri, s_coarse);
          else
          {
            if (!my_root)
              continue;
            Vector<double> s_root(2), s_nrroot(2);
            this->local_coordinate_in_ancestor(s_here, my_root, s_root);
            double t = param_on_edge(nb.my_edge_dir, s_root);
            if (nb.reversed)
              t = 1.0 - t;
            point_on_edge(nb.nr_edge_dir, t, s_nrroot);
            ok = root_coord_to_leaf(s_nrroot, coarse_tri, s_coarse);
          }
          if (ok)
            dynamic_cast<pyoomph::BulkElementBase *>(coarse_tri)->add_node_from_finer_neighbor_for_tesselated_numpy(s_coarse, me->node_pt(li), add_nodes);
        }
        else // cross-shape quad: blend our shared-root-edge parameter into the quad ROOT frame, descend to leaf
        {
          if (!my_root || !qroot_fe || !nb.cross_shape_root->nodes_built())
            continue;
          int ea, eb;
          edge_local_vertices(nb.my_edge_dir, ea, eb);
          Node *Pb = my_root->vertex_node_pt(ea);
          Node *Qb = my_root->vertex_node_pt(eb);
          Vector<double> s_root(2);
          this->local_coordinate_in_ancestor(s_here, my_root, s_root);
          const double t = param_on_edge(nb.my_edge_dir, s_root);
          const int iP = qroot_fe->get_node_number(Pb), iQ = qroot_fe->get_node_number(Qb);
          if (iP < 0 || iQ < 0)
            continue;
          Vector<double> sP(2), sQ(2), s_qroot(2);
          qroot_fe->local_coordinate_of_node(iP, sP);
          qroot_fe->local_coordinate_of_node(iQ, sQ);
          for (int d = 0; d < 2; d++)
            s_qroot[d] = (1.0 - t) * sP[d] + t * sQ[d];
          RefineableElement *qleaf = qroot->quad_leaf_at_root_coordinate(s_qroot, s_coarse);
          pyoomph::BulkElementBase *qleaf_be = dynamic_cast<pyoomph::BulkElementBase *>(qleaf);
          if (qleaf_be)
            qleaf_be->add_node_from_finer_neighbor_for_tesselated_numpy(s_coarse, me->node_pt(li), add_nodes);
        }
      }
    }
  }

  RefineableElement *RefineableTElement<2>::leaf_at_root_coordinate(const Vector<double> &s_root, Vector<double> &s_leaf) const
  {
    if (!this->Tree_pt) return 0;
    Tree *node = this->Tree_pt->root_pt();
    Vector<double> s = s_root;
    while (!node->is_leaf())
    {
      int best = -1;
      double best_min = -1e30;
      Vector<double> best_s(2);
      for (int c = 0; c < 4; c++) // tri 1->4 sons, son_type 0..3
      {
        Tree *ch = node->son_pt(c);
        if (!ch) continue;
        Vector<double> sc(2);
        father_to_son_local(s, c, sc);
        const double b = 1.0 - sc[0] - sc[1];
        const double mn = std::min(std::min(sc[0], sc[1]), b); // most-interior son contains the point
        if (mn > best_min) { best_min = mn; best = c; best_s = sc; }
      }
      if (best < 0 || best_min < -1e-9) break;
      node = node->son_pt(best);
      s = best_s;
    }
    s_leaf = s;
    return dynamic_cast<RefineableElement *>(node->object_pt());
  }

  Node *RefineableTElement<2>::node_at_root_coordinate(const Vector<double> &s_root) const
  {
    // Search the whole subtree, not just the leaf the point falls in: the holder of the node can sit at
    // any level (see node_in_subtree_at_local_coordinate), and on a son boundary it can be on either side.
    if (!this->Tree_pt) return 0;
    return node_in_subtree_at_local_coordinate(this->Tree_pt->root_pt(), s_root);
  }

  //=================================================================
  /// Tri-native topological neighbour finder. See the header comment on TriEdgeNeighbour /
  /// tri_edge_neighbour for the algorithm (son_type ascent + father-point edge tracking).
  //=================================================================
  RefineableTElement<2>::TriEdgeNeighbour RefineableTElement<2>::tri_edge_neighbour(int my_edge) const
  {
    using namespace QuadTreeNames;
    TriEdgeNeighbour res;
    if (!this->Tree_pt) return res;

    // --- Ascent: track the edge as a (local vertex a,b) pair, climbing while it stays on a father edge. ---
    int ea, eb;
    edge_local_vertices(my_edge, ea, eb);
    if (ea < 0) return res;
    const Tree *cur = this->Tree_pt;
    int climbs = 0;
    while (true)
    {
      Tree *fath = cur->father_pt();
      if (!fath) break; // reached the root with the edge on the boundary -> cross-root case below
      const int st = cur->son_type();
      const int fpa = son_vertex_fp(st, ea), fpb = son_vertex_fp(st, eb);
      const int D = father_edge_of_pair(fpa, fpb);
      if (D < 0)
      {
        // Edge is interior to `fath`: the >=-sized neighbour is the sibling son sharing this edge.
        const int sib_st = interior_sibling(st, fpa, fpb);
        if (sib_st < 0) return res;
        Tree *sib = fath->son_pt(sib_st);
        if (!sib) return res;
        // Return the sibling if it is a leaf: climbs==0 => EQUAL-sized neighbour (diff_level 0), climbs>=1
        // => strictly coarser (diff_level<0). The hang helper acts only on diff_level<0, so returning the
        // equal neighbour is behaviour-preserving there. A non-leaf sibling is equal-or-finer -> leave
        // res.el null. (Node-sharing no longer comes through here: it needs holders at any level, leaf or
        // not, and does its own subtree search -- see node_created_by_neighbour.)
        if (sib->is_leaf() || climbs == 0)
        {
          // Leaf sibling (or the immediate equal sibling): return it directly.
          if (sib->is_leaf())
          {
            res.el = dynamic_cast<RefineableTElement<2> *>(sib->object_pt());
            res.diff_level = -climbs;
          }
          return res;
        }
        // Non-leaf sibling: the equal (or finer/coarser) neighbour is a DESCENDANT of sib. Descend into
        // sib toward THIS element's edge position, exactly as oomph's gteq_edge_neighbour descends through
        // its recursion (its node_created_by_son_of_neighbour is a no-op for h-refinement). Without this,
        // an equal same-root COUSIN (this and it refined in different rounds) is never found -> its shared
        // C2 mid-node is duplicated -> a velocity/position jump on adaptive (moving) meshes.
        //
        // The interior shared edge is {fpa,fpb} (father points) in fath's frame. Find sib's local edge that
        // carries the same father points, and this element's edge interval [t0,t1] along that shared edge.
        static const double FPC[6][2] = {{1, 0}, {0, 1}, {0, 0}, {0.5, 0}, {0.5, 0.5}, {0, 0.5}}; // FV0,FV1,FV2,FmS,FmE,FmW
        RefineableTElement<2> *fath_el = dynamic_cast<RefineableTElement<2> *>(fath->object_pt());
        if (!fath_el) return res;
        int sib_edge = -1;
        bool sib_rev = false;
        for (int e : {S, E, W})
        {
          int a2, b2;
          edge_local_vertices(e, a2, b2);
          const int p0 = son_vertex_fp(sib_st, a2), p1 = son_vertex_fp(sib_st, b2);
          if (p0 == fpa && p1 == fpb) { sib_edge = e; sib_rev = false; break; }
          if (p0 == fpb && p1 == fpa) { sib_edge = e; sib_rev = true; break; }
        }
        if (sib_edge < 0) return res;
        // THIS element's own edge endpoints, mapped into fath's frame, then a parameter along {fpa,fpb}.
        int oa, ob;
        edge_local_vertices(my_edge, oa, ob);
        Vector<double> se0(2), se1(2), f0(2), f1(2);
        point_on_edge(my_edge, 0.0, se0);
        point_on_edge(my_edge, 1.0, se1);
        this->local_coordinate_in_ancestor(se0, fath_el, f0);
        this->local_coordinate_in_ancestor(se1, fath_el, f1);
        const double A0 = FPC[fpa][0], A1 = FPC[fpa][1], B0 = FPC[fpb][0], B1 = FPC[fpb][1];
        const double L2 = (B0 - A0) * (B0 - A0) + (B1 - A1) * (B1 - A1);
        if (L2 < 1e-30) return res;
        double t0 = ((f0[0] - A0) * (B0 - A0) + (f0[1] - A1) * (B1 - A1)) / L2;
        double t1 = ((f1[0] - A0) * (B0 - A0) + (f1[1] - A1) * (B1 - A1)) / L2;
        if (sib_rev) { t0 = 1.0 - t0; t1 = 1.0 - t1; } // sib parametrises the shared edge in the opposite sense
        if (t0 > t1) std::swap(t0, t1);
        // Descend into sib toward [t0,t1] along sib_edge, to at most THIS element's depth (climbs).
        Tree *node = sib;
        int dep = 0;
        double lo = 0.0, hi = 1.0;
        const double tol = 1e-9;
        while (true)
        {
          if (node->is_leaf()) break;
          if (dep >= climbs) break;
          const double mid = 0.5 * (lo + hi);
          const double a = (t0 - lo) / (hi - lo), b = (t1 - lo) / (hi - lo);
          int half;
          if (b <= 0.5 + tol) half = 0;
          else if (a >= 0.5 - tol) half = 1;
          else break; // straddles the midpoint: equal-or-finer on the sib side -> no coarser neighbour
          Tree *child = node->son_pt(son_on_edge_half(sib_edge, half));
          if (!child) break;
          node = child;
          if (half == 0) hi = mid; else lo = mid;
          dep++;
        }
        RefineableTElement<2> *cnb = dynamic_cast<RefineableTElement<2> *>(node->object_pt());
        if (cnb && node->is_leaf()) { res.el = cnb; res.diff_level = dep - climbs; } // same-root -> cross_root stays false
        return res;
      }
      // Edge lies on `fath`'s D edge -> climb, re-expressing the edge in the father's vertices.
      cur = fath;
      edge_local_vertices(D, ea, eb);
      climbs++;
    }

    // --- Cross-root: `cur` is the root; the edge is its boundary edge my_D. ---
    const int my_D = edge_from_local_vertices(ea, eb);
    if (my_D < 0) return res;
    TreeRoot *root = cur->root_pt();
    TreeRoot *nr = root->neighbour_pt(my_D);
    if (!nr) return res; // domain boundary: no neighbour
    RefineableTElement<2> *my_root_el = dynamic_cast<RefineableTElement<2> *>(root->object_pt());
    if (!my_root_el) return res;
    RefineableTElement<2> *nr_root_el = dynamic_cast<RefineableTElement<2> *>(nr->object_pt());
    if (!nr_root_el)
    {
      // Cross-SHAPE neighbour root (a QUAD at a mixed quad+tri interface): the tri son-descent tables do
      // not apply. Expose the quad ROOT + shared edge direction for both node-sharing (descend the quad
      // tree to the leaf at the shared-edge point) and hanging. For HANGING we additionally need the quad
      // to be strictly coarser, i.e. the UNREFINED quad root (a leaf): a refined quad hangs on the tri from
      // its own quad_hang_helper. Coordinates are bridged shape-agnostically via my_edge_dir.
      res.cross_shape_root = dynamic_cast<oomph::RefineableElement *>(nr->object_pt());
      res.cross_root = true;
      res.my_edge_dir = my_D;
      if (nr->is_leaf())
      {
        res.cross_shape_el = res.cross_shape_root; // strictly coarser quad -> this tri hangs on it
        res.diff_level = -climbs;
      }
      return res;
    }

    // Topological shared-edge correspondence: my root edge's two vertex nodes -> the neighbour root's
    // vertex indices, giving the neighbour's edge direction and whether the parametrisation is reversed.
    Node *na = my_root_el->vertex_node_pt(ea);
    Node *nb = my_root_el->vertex_node_pt(eb);
    int ca = -1, cb = -1;
    for (int j = 0; j < 3; j++)
    {
      Node *vj = nr_root_el->vertex_node_pt(j);
      if (vj == na) ca = j;
      if (vj == nb) cb = j;
    }
    if (ca < 0 || cb < 0) return res; // roots not actually edge-sharing (should not happen)
    const int nr_D = edge_from_local_vertices(ca, cb);
    if (nr_D < 0) return res;
    int nca, ncb;
    edge_local_vertices(nr_D, nca, ncb); // neighbour edge param: 0 at nca, 1 at ncb
    const bool reversed = (nca != ca);   // my param-0 node (na<->ca) vs neighbour param-0 node (nca)

    // L's edge interval [t0,t1] along my root edge (via the exact affine map), then in the neighbour frame.
    Vector<double> sa(2), sb(2), ra(2), rb(2);
    edge_local_vertices(my_edge, ea, eb); // this element's own local vertices for my_edge
    point_on_edge(my_edge, 0.0, sa);      // = vertex ea in this element
    point_on_edge(my_edge, 1.0, sb);      // = vertex eb
    this->local_coordinate_in_ancestor(sa, my_root_el, ra);
    this->local_coordinate_in_ancestor(sb, my_root_el, rb);
    double t0 = param_on_edge(my_D, ra), t1 = param_on_edge(my_D, rb);
    if (reversed) { t0 = 1.0 - t0; t1 = 1.0 - t1; }
    if (t0 > t1) std::swap(t0, t1);

    // Descend into the neighbour root along nr_D toward [t0,t1], stopping at the >=-sized element.
    Tree *node = nr;
    int dep = 0;
    double lo = 0.0, hi = 1.0;
    const double tol = 1e-9;
    while (true)
    {
      if (node->is_leaf()) break;    // spans [lo,hi] >= L's edge -> this is the >= neighbour
      if (dep >= climbs) break;      // do not descend finer than L
      const double mid = 0.5 * (lo + hi);
      const double a = (t0 - lo) / (hi - lo), b = (t1 - lo) / (hi - lo); // interval in node's [0,1] param
      int half;
      if (b <= 0.5 + tol) half = 0;
      else if (a >= 0.5 - tol) half = 1;
      else break; // straddles the midpoint: the neighbour side is equal-or-finer here -> no coarse neighbour
      const int cst = son_on_edge_half(nr_D, half);
      Tree *child = node->son_pt(cst);
      if (!child) break;
      node = child;
      if (half == 0) hi = mid; else lo = mid;
      dep++;
    }
    RefineableTElement<2> *nb_el = dynamic_cast<RefineableTElement<2> *>(node->object_pt());
    if (!nb_el || !node->is_leaf()) return res; // only a leaf can be a hang master; non-leaf => equal/finer
    res.el = nb_el;
    res.diff_level = dep - climbs;
    res.cross_root = true;
    res.my_edge_dir = my_D;
    res.nr_edge_dir = nr_D;
    res.reversed = reversed;
    return res;
  }

  //=================================================================
  /// Tree-based hanging on one triangle edge. See the header for the rationale (topology from the
  /// QuadTree son_type descent, coordinates from the exact affine map -- never locate_zeta nor the
  /// quad compass coordinate descent).
  //=================================================================
  void RefineableTElement<2>::tri_hang_helper(const int &value_id, const int &my_edge)
  {
    using namespace QuadTreeNames;
    // Tri-native topological neighbour search (son_type ascent + father-point edge tracking). Returns the
    // >=-sized LEAF neighbour across my_edge; we hang only when it is STRICTLY COARSER (diff_level < 0).
    // A finer neighbour hangs on THIS element from its own side, so skipping it here loses nothing.
    TriEdgeNeighbour nb = this->tri_edge_neighbour(my_edge);
    // Cross-SHAPE coarser neighbour (a QUAD at a mixed quad+tri interface): hang each of THIS tri edge's
    // interpolating nodes on the coarse quad via the shape-agnostic primitive -- the shared root-edge corner
    // nodes + the neighbour's own interpolating_basis (the tri affine edge map does not apply to a quad).
    if (nb.cross_shape_root != nullptr)
    {
      // Hang each of THIS tri edge's interpolating nodes on the coarse QUAD: mixed_hang_edge_node maps the
      // node's shared-edge parameter into the quad ROOT (blend of the shared corner nodes) and descends to
      // the correct quad LEAF (coarser-or-equal); the level guard there hangs only where the quad leaf is
      // strictly coarser (equal is shared, finer hangs from its own side). Handles a refined-but-coarser quad.
      RefineableTElement<2> *my_root_el = dynamic_cast<RefineableTElement<2> *>(this->Tree_pt->root_pt()->object_pt());
      pyoomph::BulkElementBase *self = dynamic_cast<pyoomph::BulkElementBase *>(this);
      if (!my_root_el || !self) return;
      int ea, eb;
      edge_local_vertices(nb.my_edge_dir, ea, eb);
      Node *Pb = my_root_el->vertex_node_pt(ea); // shared root-edge corner at t=0
      Node *Qb = my_root_el->vertex_node_pt(eb); // at t=1
      const unsigned n_edge = this->ninterpolating_node_1d(value_id);
      for (unsigned i = 0; i < n_edge; i++)
      {
        const double f = (n_edge > 1) ? double(i) / double(n_edge - 1) : 0.0;
        Vector<double> s(2);
        switch (my_edge)
        {
          case E: s[0] = 1.0 - f; s[1] = f; break;
          case W: s[0] = 0.0; s[1] = 1.0 - f; break;
          case S: s[0] = f; s[1] = 0.0; break;
          default: return;
        }
        Node *X = this->get_interpolating_node_at_local_coordinate(s, value_id);
        if (!X) continue;
        Vector<double> s_root(2);
        this->local_coordinate_in_ancestor(s, my_root_el, s_root);
        const double t = param_on_edge(nb.my_edge_dir, s_root);
        self->mixed_hang_edge_node(X, Pb, Qb, t, nb.cross_shape_root, value_id);
      }
      return;
    }
    if (nb.el == 0 || nb.diff_level >= 0) return;
    RefineableElement *neigh_re = dynamic_cast<RefineableElement *>(nb.el);
    FiniteElement *neigh_fe = dynamic_cast<FiniteElement *>(nb.el);
    if (!neigh_re || !neigh_fe) return;

    // For a cross-root neighbour we bridge coordinates through the two root frames (this root -> shared
    // edge parameter -> neighbour root -> neighbour leaf), all with the exact affine map. Fetch the roots.
    RefineableTElement<2> *my_root_el = nullptr, *nr_root_el = nullptr;
    if (nb.cross_root)
    {
      my_root_el = dynamic_cast<RefineableTElement<2> *>(this->Tree_pt->root_pt()->object_pt());
      nr_root_el = dynamic_cast<RefineableTElement<2> *>(nb.el->Tree_pt->root_pt()->object_pt());
      if (!my_root_el || !nr_root_el) return;
    }

    // Interpolating nodes along MY edge. Tri edge<->compass convention (must match get_edge_bcs and the
    // forest root-neighbour setup, both derived from Father_bound): E=v0-v1, W=v1-v2, S=v2-v0. This is
    // the slot passed to tri_edge_neighbour, so the local-coordinate walk below MUST trace the same edge.
    // TElement<2,3> vertex local coords v0=(1,0), v1=(0,1), v2=(0,0). n_edge is 2 for a C1 field (corners),
    // 3 for C2 (+mid).
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
      // Local coordinate of X inside the coarse neighbour, via the exact affine tree map (no locate_zeta,
      // no Newton -- exact even on curved meshes because refinement is defined in local coordinates).
      Vector<double> s_neigh(2);
      bool have_neigh;
      if (!nb.cross_root)
      {
        have_neigh = this->local_coordinate_in_other_leaf(s, nb.el, s_neigh); // same root: up to root, down to nb
      }
      else
      {
        // Cross root: this leaf -> my root -> shared-edge parameter -> neighbour root edge -> neighbour leaf.
        Vector<double> s_root(2), s_nrroot(2);
        this->local_coordinate_in_ancestor(s, my_root_el, s_root);
        double t = param_on_edge(nb.my_edge_dir, s_root);
        if (nb.reversed) t = 1.0 - t;
        point_on_edge(nb.nr_edge_dir, t, s_nrroot);
        have_neigh = root_coord_to_leaf(s_nrroot, nb.el, s_neigh);
      }
      if (!have_neigh) continue;
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
      // Cycle guard: would any master (transitively) hang back on X? If so this hang would create a
      // mutual/cyclic dependency (infinite recursion in oomph's complete_hanging_nodes). Detect and skip.
      std::function<bool(Node *, Node *, int, int)> reaches = [&](Node *from, Node *to, int slot, int depth) -> bool {
        if (from == to) return true;
        if (depth > 30 || !from->is_hanging(slot)) return false;
        HangInfo *h = from->hanging_pt(slot);
        for (unsigned k = 0; k < h->nmaster(); k++) if (reaches(h->master_node_pt(k), to, slot, depth + 1)) return true;
        return false;
      };
      bool cyclic = false;
      for (unsigned m = 0; m < nmax && !cyclic; m++)
        if (std::abs(psi[m]) > 1e-12 && reaches(neigh_re->interpolating_node_pt(m, value_id), X, value_id, 0)) cyclic = true;
      if (cyclic)
        continue;
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
    using namespace QuadTreeNames;
    // Geometry / position / C2 hanging is value_id -1 (the shared geometric slot).
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

  // Position -> node snapshot for the cross-round (finer-neighbour) node-sharing fallback (see header).
  std::map<std::string, Node *> RefineableTElement<2>::Existing_node_by_position;

  std::string RefineableTElement<2>::position_key(const double *x, unsigned dim)
  {
    char buf[128];
    if (dim >= 3) std::snprintf(buf, sizeof(buf), "%.12g|%.12g|%.12g", x[0], x[1], x[2]);
    else if (dim == 2) std::snprintf(buf, sizeof(buf), "%.12g|%.12g", x[0], x[1]);
    else std::snprintf(buf, sizeof(buf), "%.12g", x[0]);
    return std::string(buf);
  }

  void RefineableTElement<2>::register_existing_node_position(Node *n)
  {
    if (!n) return;
    const unsigned dim = n->ndim();
    double x[3] = {0, 0, 0};
    for (unsigned i = 0; i < dim && i < 3; i++) x[i] = n->x(i);
    Existing_node_by_position[position_key(x, dim)] = n;
  }

  // Shared (2d + 3d) lookup: the start-of-round snapshot is stored on RefineableTElement<2> and reused by
  // both the triangle and tetrahedron build() fallbacks (mesh.hpp registers every node here, 2d or 3d).
  Node *RefineableTElement<2>::find_existing_node_at_position(const double *x, unsigned dim)
  {
    if (Existing_node_by_position.empty()) return 0;
    std::map<std::string, Node *>::iterator it = Existing_node_by_position.find(position_key(x, dim));
    return (it != Existing_node_by_position.end()) ? it->second : 0;
  }

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

    // A tet can be a son of a PYRAMID (the 4 tet children of the pyramid red split). In that case the
    // father is not a tet, so the tet-in-tet affine map below does not apply -- route to the pyramid's
    // generic C1 son builder, which handles the mixed offspring uniformly.
    if (dynamic_cast<oomph::RefineablePyramidElement *>(father_pt->object_pt()))
    {
      dynamic_cast<pyoomph::BulkElementBase *>(this)->build_as_pyramid_son(mesh_pt, new_node_pt);
      return;
    }

    RefineableTElement<3> *father_el_pt = dynamic_cast<RefineableTElement<3> *>(father_pt->object_pt());
    TimeStepper *time_stepper_pt = father_el_pt->node_pt(0)->time_stepper_pt();
    unsigned ntstorage = time_stepper_pt->ntstorage();

#ifdef OOMPH_HAS_MPI
    // Propagate the halo-ownership tag from father to son, exactly as RefineableTElement<2>::build does.
    // Without it a refined tet son keeps the default Non_halo_proc_ID, so under MPI the distributed
    // classify_halo_and_haloed_nodes misclassifies its nodes and the per-rank halo/haloed node lists drift
    // apart -- which leaks an unmatched message in Mesh::resize_halo_nodes and then crashes/deadlocks the
    // downstream halo-node synchronisation. (Pure-serial builds ignore this field.)
    Non_halo_proc_ID = father_el_pt->non_halo_proc_ID();
#endif

    // The 4 vertices of this son in the father's local coordinates -> affine (barycentric) map
    // from son-local to father-local coordinates.
    Vector<Vector<double>> sv;
    son_vertices_in_father(son_type, sv);

    // If the father sits on a curved macro element, hand it down together with this son's region of
    // the macro reference domain -- which is exactly the four vertices just computed. New nodes below
    // take their positions from father_el_pt->get_x(), which routes through it.
    if (father_el_pt->Macro_elem_pt != 0)
    {
      pyoomph::BulkElementBase *son_be = dynamic_cast<pyoomph::BulkElementBase *>(this);
      pyoomph::BulkElementBase *father_be = dynamic_cast<pyoomph::BulkElementBase *>(father_el_pt);
      if (son_be && father_be)
      {
        std::vector<std::vector<double>> son_vertices(sv.size(), std::vector<double>(3, 0.0));
        for (unsigned int v = 0; v < sv.size(); v++)
          for (unsigned int i = 0; i < 3; i++) son_vertices[v][i] = sv[v][i];
        son_be->inherit_macro_element_from_father(father_be, son_vertices);
      }
      else
      {
        set_macro_elem_pt(father_el_pt->Macro_elem_pt);
      }
    }

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

      // (1.5) Topological: reuse a node an already-built NEIGHBOUR holds at this point, found via the
      // OcTree (which persists across adaptation rounds). This closes the cross-round gap -- a neighbour
      // built in an EARLIER round -- so a moving tet mesh does not tear at a refine/coarsen interface
      // (the 3d analogue of the triangle node_created_by_neighbour). It searches this element's whole tree
      // root and then out through root face neighbours, so the holder may be at any level, need not be a
      // leaf, and may share only an edge with this element. The per-round registry below only dedupes
      // nodes built within THIS round. Only for genuine shared nodes.
      if (!reg_key.empty())
      {
        Node *nbn = this->node_created_by_neighbour(s_son);
        if (nbn) { node_pt(j) = nbn; continue; }
      }

      // (2) Reuse a node already created (by a sibling son or the face-/edge-sharing neighbour father's sons)
      // via the geometric shared-node registry. In a PYRAMID forest use the pyramid forest's registry
      // instead of the tet-only one, so a tet-of-pyramid and an adjacent sub-pyramid share their interface
      // face nodes. That registry's key is the (father node, rounded father-shape weight) PAIRS -- the same
      // weight-augmented key build_as_pyramid_son builds -- so a shared triangular face node gets identical
      // pairs from either side (the pyramid and tet face traces are both the standard quadratic on the shared
      // face), while for C2 two distinct interior points on one father edge no longer collide. Topological ->
      // MPI-safe. Outside a pyramid forest the bare father-node SET key into the tet-only registry suffices.
      // Route into the shared pyramid registry when this tet is a son of a pyramid (in_pyramid_forest) OR the
      // mesh is a mixed 3d forest (a tet adjacent to a wedge/pyramid): both need the weight-augmented key so a
      // shared triangular face node is keyed identically from either side.
      const bool pyr_forest = in_pyramid_forest() || oomph::RefineablePyramidElement::Mixed_forest_active;
      oomph::RefineablePyramidElement::SharedNodeKey pyr_key;
      if (pyr_forest && !reg_key.empty())
      {
        Shape fpsi(father_el_pt->nnode());
        father_el_pt->shape(s, fpsi);
        for (unsigned l = 0; l < father_el_pt->nnode(); l++)
          if (fpsi(l) > 1e-6)
            pyr_key.insert(std::make_pair(father_el_pt->node_pt(l), (long long)std::llround(fpsi(l) * 1e6)));
      }
      if (!reg_key.empty())
      {
        if (pyr_forest)
        {
          std::map<oomph::RefineablePyramidElement::SharedNodeKey, Node *>::iterator it =
              oomph::RefineablePyramidElement::Shared_node_registry.find(pyr_key);
          if (it != oomph::RefineablePyramidElement::Shared_node_registry.end()) { node_pt(j) = it->second; continue; }
        }
        else
        {
          std::map<std::set<Node *>, Node *>::iterator it = Shared_edge_node_registry.find(reg_key);
          if (it != Shared_edge_node_registry.end()) { node_pt(j) = it->second; continue; }
        }
      }

      // (2b) Fallback for a PYRAMID-ROOTED (mixed 3d) forest only: reuse a coincident node that already
      // existed at the START of this refinement round -- notably one built by a FINER neighbour in an
      // EARLIER round. The per-round registry above only dedupes within THIS round, so without this a
      // shared node is DUPLICATED under transient re-adaptation and a moving mixed mesh tears apart at the
      // refine/coarsen interface. The new node sits at the average of its generating nodes, which is where
      // the existing coincident node also sits.
      //
      // A pure tet forest does NOT come here: step (1.5) above now finds the holder topologically at any
      // level and across edges as well as faces. Identifying a node by its POSITION is only as good as the
      // positions are, and a hanging node's stored position is a cache of its masters that anything writing
      // the dof vector from outside the Newton solver leaves stale -- which is exactly how adapting on an
      // eigenfunction used to tear a 2d mesh. The remaining mixed-3d shapes (pyramid/wedge/brick-as-son)
      // still need the same treatment as the tets got here.
      //
      // "Mixed" has to be tested with the same predicate the registry above uses, not just
      // in_pyramid_forest(): in a three-way (tet+wedge+pyramid) forest a tet can be rooted in a WEDGE, and
      // its neighbours across a face can be of any shape -- neither the tet tree walk nor its root hops
      // reach those, so the fallback is still load-bearing there.
      const bool mixed_3d_forest = in_pyramid_forest() || oomph::RefineablePyramidElement::Mixed_forest_active;
      if (mixed_3d_forest && !reg_key.empty() && !gen.empty())
      {
        const unsigned dim = gen[0]->ndim();
        double x_new[3] = {0.0, 0.0, 0.0};
        for (Node *g : gen)
          for (unsigned i = 0; i < dim && i < 3; i++) x_new[i] += g->x(i);
        for (unsigned i = 0; i < dim && i < 3; i++) x_new[i] /= double(gen.size());
        Node *ex = RefineableTElement<2>::find_existing_node_at_position(x_new, dim);
        if (ex) { node_pt(j) = ex; continue; }
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
      if (!reg_key.empty())
      {
        if (pyr_forest) oomph::RefineablePyramidElement::Shared_node_registry[pyr_key] = created_node_pt;
        else Shared_edge_node_registry[reg_key] = created_node_pt;
      }
    }
  }

  //====================================================================
  ///  Print corner nodes, use colour (default "BLACK")
  //====================================================================
  void RefineableTElement<3>::output_corners(std::ostream &, const std::string &) const
  {
    // Debug-only output; not needed for refinement itself.
  }

  // Per-element hanging setup, driven by oomph's adapt loop (setup_hanging_nodes + further_setup_hanging_nodes
  // per element, then complete_hanging_nodes) -- the 3d analogue of RefineableTElement<2>. The geometric slot
  // -1 (position/C2/C2TB) is done here; the separate C1/C2 value slots are driven from the tet element's
  // further_setup_hanging_nodes (which calls setup_hang_for_value). Each helper uses the OcTree neighbour
  // finders + the exact affine map + interpolating_basis -- fully topological. A per-element install (vs the
  // former mesh-level pass) means refine_selected_elements / custom_adapt get hanging too, and it composes
  // with oomph's complete_hanging_nodes (which flattens recursive master chains).
  // A tet that is a son of a PYRAMID (the 4 tet children of the pyramid red split) lives in a pyramid-rooted
  // forest. Its son_type (6..9) and the tree ancestry above it are NOT the tet-in-tet map, so the tet
  // neighbour/hanging tree-walk (son_vertices_in_father, son_to_father_local) does not apply. Uniform pyramid
  // refinement is conforming (node-sharing via the pyramid registry), so there is nothing to hang -- skip.
  // (Non-uniform pyramid hanging will need a dedicated cross-shape scheme, a later milestone.)
  bool RefineableTElement<3>::in_pyramid_forest() const
  {
    return this->Tree_pt && this->Tree_pt->root_pt() &&
           dynamic_cast<oomph::RefineablePyramidElement *>(this->Tree_pt->root_pt()->object_pt()) != 0;
  }
  void RefineableTElement<3>::setup_hanging_nodes(Vector<std::ofstream *> &)
  {
    if (in_pyramid_forest()) return;
    for (int f = 0; f < 4; f++) tet_hang_face(-1, f);
    for (int e = 0; e < 6; e++) tet_hang_edge(-1, e);
  }
  void RefineableTElement<3>::setup_hang_for_value(const int &value_id)
  {
    if (in_pyramid_forest()) return;
    for (int f = 0; f < 4; f++) tet_hang_face(value_id, f);
    for (int e = 0; e < 6; e++) tet_hang_edge(value_id, e);
  }
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
  void RefineableTElement<3>::son_vertices_in_father(int son_type, Vector<Vector<double>> &verts)
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

  void RefineableTElement<3>::son_to_father_local(const Vector<double> &s_son, int son_type, Vector<double> &s_father)
  {
    Vector<Vector<double>> sv;
    son_vertices_in_father(son_type, sv);
    const double b0 = s_son[0], b1 = s_son[1], b2 = s_son[2], b3 = 1.0 - b0 - b1 - b2;
    s_father.resize(3);
    for (unsigned d = 0; d < 3; d++)
      s_father[d] = b0 * sv[0][d] + b1 * sv[1][d] + b2 * sv[2][d] + b3 * sv[3][d];
  }

  void RefineableTElement<3>::father_to_son_local(const Vector<double> &s_father, int son_type, Vector<double> &s_son)
  {
    // s_father = P3 + A * s_son, A columns = (P0-P3),(P1-P3),(P2-P3). Invert the 3x3.
    Vector<Vector<double>> sv;
    son_vertices_in_father(son_type, sv);
    const double A[3][3] = {
        {sv[0][0] - sv[3][0], sv[1][0] - sv[3][0], sv[2][0] - sv[3][0]},
        {sv[0][1] - sv[3][1], sv[1][1] - sv[3][1], sv[2][1] - sv[3][1]},
        {sv[0][2] - sv[3][2], sv[1][2] - sv[3][2], sv[2][2] - sv[3][2]}};
    const double r0 = s_father[0] - sv[3][0], r1 = s_father[1] - sv[3][1], r2 = s_father[2] - sv[3][2];
    const double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    // Inverse via the adjugate (cofactor) matrix / det.
    const double inv[3][3] = {
        {(A[1][1] * A[2][2] - A[1][2] * A[2][1]) / det, (A[0][2] * A[2][1] - A[0][1] * A[2][2]) / det, (A[0][1] * A[1][2] - A[0][2] * A[1][1]) / det},
        {(A[1][2] * A[2][0] - A[1][0] * A[2][2]) / det, (A[0][0] * A[2][2] - A[0][2] * A[2][0]) / det, (A[0][2] * A[1][0] - A[0][0] * A[1][2]) / det},
        {(A[1][0] * A[2][1] - A[1][1] * A[2][0]) / det, (A[0][1] * A[2][0] - A[0][0] * A[2][1]) / det, (A[0][0] * A[1][1] - A[0][1] * A[1][0]) / det}};
    s_son.resize(3);
    s_son[0] = inv[0][0] * r0 + inv[0][1] * r1 + inv[0][2] * r2;
    s_son[1] = inv[1][0] * r0 + inv[1][1] * r1 + inv[1][2] * r2;
    s_son[2] = inv[2][0] * r0 + inv[2][1] * r1 + inv[2][2] * r2;
  }

  void RefineableTElement<3>::local_coordinate_in_ancestor(const Vector<double> &s_here, RefineableTElement<3> *ancestor, Vector<double> &s_ancestor) const
  {
    s_ancestor = s_here;
    Tree *tp = this->Tree_pt;
    while (tp && tp->object_pt() != ancestor)
    {
      Tree *father_tp = tp->father_pt();
      if (!father_tp) break; // reached the root without meeting the ancestor
      Vector<double> s_up(3);
      son_to_father_local(s_ancestor, tp->son_type(), s_up);
      s_ancestor = s_up;
      tp = father_tp;
    }
  }

  bool RefineableTElement<3>::root_coord_to_leaf(const Vector<double> &s_root, RefineableTElement<3> *leaf, Vector<double> &s_leaf)
  {
    if (!leaf || !leaf->Tree_pt) return false;
    Tree *root = leaf->Tree_pt->root_pt();
    std::vector<int> chain; // son_types from just-below-root down to leaf
    Tree *tp = leaf->Tree_pt;
    while (tp != root) { chain.push_back(tp->son_type()); tp = tp->father_pt(); if (!tp) return false; }
    Vector<double> s = s_root;
    for (auto it = chain.rbegin(); it != chain.rend(); ++it)
    {
      Vector<double> sd(3);
      father_to_son_local(s, *it, sd);
      s = sd;
    }
    s_leaf = s;
    return true;
  }

  bool RefineableTElement<3>::local_coordinate_in_other_leaf(const Vector<double> &s_here, RefineableTElement<3> *target, Vector<double> &s_target) const
  {
    if (!this->Tree_pt || !target || !target->Tree_pt) return false;
    Tree *my_root = this->Tree_pt->root_pt();
    if (my_root != target->Tree_pt->root_pt()) return false; // cross-root handled by the inter-tree map
    RefineableTElement<3> *root_el = dynamic_cast<RefineableTElement<3> *>(my_root->object_pt());
    if (!root_el) return false;
    Vector<double> s_root(3);
    this->local_coordinate_in_ancestor(s_here, root_el, s_root);
    return root_coord_to_leaf(s_root, target, s_target);
  }

  namespace
  {
    // Father "points" of a 1->8 tet split: 4 father vertices + 6 father edge-midpoints. Every son vertex
    // is one of these (C1: sons are built from the 4 corners + 6 edge-mids). Ids: FV0..3 = 0..3;
    // FM01,FM02,FM03,FM12,FM13,FM23 = 4..9.
    inline int tet_fp_id(const oomph::Vector<double> &c)
    {
      static const double P[10][3] = {
          {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0},                     // FV0..3
          {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0.5, 0, 0},                      // FM01,FM02,FM03
          {0, 0.5, 0.5}, {0, 0.5, 0}, {0, 0, 0.5}};                       // FM12,FM13,FM23
      for (int i = 0; i < 10; i++)
        if (std::abs(c[0] - P[i][0]) + std::abs(c[1] - P[i][1]) + std::abs(c[2] - P[i][2]) < 1e-9) return i;
      return -1;
    }

    // The 6 father points on each father face (face f is opposite father vertex f).
    inline bool tet_fp_on_face(int fp, int f)
    {
      static const int FACE[4][6] = {
          {1, 2, 3, 7, 8, 9},  // face 0 (opp v0): v1,v2,v3,m12,m13,m23
          {0, 2, 3, 5, 6, 9},  // face 1 (opp v1): v0,v2,v3,m02,m03,m23
          {0, 1, 3, 4, 6, 8},  // face 2 (opp v2): v0,v1,v3,m01,m03,m13
          {0, 1, 2, 4, 5, 7}}; // face 3 (opp v3): v0,v1,v2,m01,m02,m12
      for (int k = 0; k < 6; k++) if (FACE[f][k] == fp) return true;
      return false;
    }

    // Which father face (0..3) contains all three father points, or -1 if the triple is INTERIOR.
    inline int tet_father_face_of_triple(int a, int b, int c)
    {
      for (int f = 0; f < 4; f++)
        if (tet_fp_on_face(a, f) && tet_fp_on_face(b, f) && tet_fp_on_face(c, f)) return f;
      return -1;
    }

    // Father-point ids of son son_type's four LOCAL faces (face i is opposite local vertex i).
    inline void tet_son_face_fps(int son_type, int faceFP[4][3])
    {
      oomph::Vector<oomph::Vector<double>> sv;
      oomph::RefineableTElement<3>::son_vertices_in_father(son_type, sv);
      int vfp[4];
      for (int v = 0; v < 4; v++) vfp[v] = tet_fp_id(sv[v]);
      for (int f = 0; f < 4; f++) { int k = 0; for (int v = 0; v < 4; v++) if (v != f) faceFP[f][k++] = vfp[v]; }
    }

    // The sibling son (0..7, != st) that shares the interior face whose father-point set is {t0,t1,t2}.
    inline int tet_sibling_across_face(int st, int t0, int t1, int t2)
    {
      std::set<int> T{t0, t1, t2};
      for (int s = 0; s < 8; s++)
      {
        if (s == st) continue;
        int fp[4][3];
        tet_son_face_fps(s, fp);
        for (int f = 0; f < 4; f++)
          if (std::set<int>{fp[f][0], fp[f][1], fp[f][2]} == T) return s;
      }
      return -1;
    }

    // Reference-tet vertex local coordinates (v0..v3) and the centroid of face f (opposite vertex f).
    inline void tet_face_centroid(int f, oomph::Vector<double> &s_c)
    {
      static const double V[4][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}};
      s_c.resize(3);
      s_c[0] = s_c[1] = s_c[2] = 0.0;
      for (int v = 0; v < 4; v++)
        if (v != f)
          for (int d = 0; d < 3; d++) s_c[d] += V[v][d] / 3.0;
    }

    // The 6 local edges of a tet as vertex-index pairs (consistent indexing for edge tracking).
    static const int TET_EDGE_V[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
    // Father-point triples of the 6 father edges: [FVi, FMij, FVj] (fp ids from tet_fp_id).
    static const int TET_FATHER_EDGE_FP[6][3] = {
        {0, 4, 1}, {0, 5, 2}, {0, 6, 3}, {1, 7, 2}, {1, 8, 3}, {2, 9, 3}};
    // Which father edge (0..5) carries BOTH father points fpa,fpb (a sub-segment of it), or -1 if the pair
    // is not collinear on any father edge (the edge went interior to the father).
    inline int tet_father_edge_of_pair(int fpa, int fpb)
    {
      for (int e = 0; e < 6; e++)
      {
        const int *t = TET_FATHER_EDGE_FP[e];
        bool ha = (fpa == t[0] || fpa == t[1] || fpa == t[2]);
        bool hb = (fpb == t[0] || fpb == t[1] || fpb == t[2]);
        if (ha && hb) return e;
      }
      return -1;
    }

    // Descend into `start`'s subtree toward the point s (given in start's OWN local frame), following at
    // each level the son that CONTAINS the point (largest minimum barycentric = most interior, so shared
    // son faces are unambiguous for an interior target). Stops at a leaf or after max_extra levels. Writes
    // the point in the reached node's frame + the number of levels descended, and returns the node. The 3d
    // analogue of the tri interval-bisection descent, but by point-containment (a face region is 2d).
    inline oomph::Tree *tet_descend_to_point(oomph::Tree *start, oomph::Vector<double> s, int max_extra,
                                             oomph::Vector<double> &s_out, int &depth_out)
    {
      oomph::Tree *node = start;
      int d = 0;
      while (!node->is_leaf() && d < max_extra)
      {
        int best = -1;
        double best_min = -1e30;
        oomph::Vector<double> best_s(3);
        for (int c = 0; c < 8; c++)
        {
          if (!node->son_pt(c)) continue;
          oomph::Vector<double> sc(3);
          oomph::RefineableTElement<3>::father_to_son_local(s, c, sc);
          const double b3 = 1.0 - sc[0] - sc[1] - sc[2];
          const double mn = std::min(std::min(sc[0], sc[1]), std::min(sc[2], b3));
          if (mn > best_min) { best_min = mn; best = c; best_s = sc; }
        }
        if (best < 0 || best_min < -1e-9) break; // point not inside any son (should not happen for interior s)
        node = node->son_pt(best);
        s = best_s;
        d++;
      }
      s_out = s;
      depth_out = d;
      return node;
    }
  } // namespace

  RefineableTElement<3>::TetFaceNeighbour RefineableTElement<3>::tet_face_neighbour(int my_face) const
  {
    TetFaceNeighbour res;
    if (!this->Tree_pt) return res;
    // Track the face as its 3 local vertex indices (the vertices != my_face).
    int fv[3]; { int k = 0; for (int v = 0; v < 4; v++) if (v != my_face) fv[k++] = v; }
    const Tree *cur = this->Tree_pt;
    int climbs = 0;
    while (true)
    {
      Tree *fath = cur->father_pt();
      if (!fath) break; // reached the root with the face on the boundary -> cross-root (later step)
      const int st = cur->son_type();
      Vector<Vector<double>> sv;
      son_vertices_in_father(st, sv);
      int fp[3];
      for (int i = 0; i < 3; i++) fp[i] = tet_fp_id(sv[fv[i]]);
      const int F = tet_father_face_of_triple(fp[0], fp[1], fp[2]);
      if (F < 0)
      {
        // Face is interior to `fath`: the >=-sized neighbour is the sibling son sharing it (or a
        // descendant of it if that sibling is refined).
        const int sib_st = tet_sibling_across_face(st, fp[0], fp[1], fp[2]);
        if (sib_st < 0) return res;
        Tree *sib = fath->son_pt(sib_st);
        if (!sib) return res;
        if (sib->is_leaf() || climbs == 0)
        {
          // Leaf sibling (or the immediate equal sibling): return it directly. climbs==0 => equal
          // (diff 0); climbs>=1 => strictly coarser (diff<0, this hangs on it). A non-leaf immediate
          // sibling is equal-or-finer -> leave res.el null (hang path acts only on diff_level<0).
          if (sib->is_leaf())
          {
            res.el = dynamic_cast<RefineableTElement<3> *>(sib->object_pt());
            res.diff_level = -climbs;
          }
          return res;
        }
        // Non-leaf sibling: the equal (or finer/coarser) neighbour is a DESCENDANT of sib -- descend into
        // it toward THIS element's face position, exactly as the 2d tri path descends into a non-leaf
        // sibling. Without this an equal same-root COUSIN (this and it refined in different rounds) is never
        // found, so its shared face node is duplicated. Target = this element's my_face centroid (strictly
        // interior to the shared face), mapped up to fath's frame then into sib's own frame.
        Vector<double> s_c(3), s_fath(3), s_sib(3), s_leaf(3);
        tet_face_centroid(my_face, s_c);
        this->local_coordinate_in_ancestor(s_c, dynamic_cast<RefineableTElement<3> *>(fath->object_pt()), s_fath);
        father_to_son_local(s_fath, sib_st, s_sib);
        int dep = 0;
        Tree *leaf = tet_descend_to_point(sib, s_sib, climbs, s_leaf, dep);
        if (leaf && leaf->is_leaf())
        {
          res.el = dynamic_cast<RefineableTElement<3> *>(leaf->object_pt());
          res.diff_level = dep - climbs; // <0 coarser, 0 equal
        }
        return res;
      }
      // Face lies on `fath`'s face F -> climb, re-expressing it as the father's 3 vertices != F.
      cur = fath;
      { int k = 0; for (int v = 0; v < 4; v++) if (v != F) fv[k++] = v; }
      climbs++;
    }

    // --- Cross-root: `cur` is the root; the face is its boundary face opposite the vertex not in fv[]. ---
    const int my_root_face = 6 - fv[0] - fv[1] - fv[2]; // the vertex (0..3) NOT among fv[]
    using namespace oomph::OcTreeNames;
    static const int FACE_SLOT[4] = {L, R, D, U};
    TreeRoot *root = cur->root_pt();
    Tree *nr = root->neighbour_pt(FACE_SLOT[my_root_face]);
    if (!nr) return res; // domain boundary: no neighbour
    RefineableTElement<3> *my_root_el = dynamic_cast<RefineableTElement<3> *>(root->object_pt());
    RefineableTElement<3> *nr_root_el = dynamic_cast<RefineableTElement<3> *>(nr->object_pt());
    if (!my_root_el || !nr_root_el) return res;

    // Topological shared-face correspondence: each shared corner NODE (my root vertex fv[i]) -> its vertex
    // index in the neighbour root. A point on the shared face keeps its barycentric weights on these corners.
    for (int i = 0; i < 3; i++)
    {
      Node *nd = my_root_el->vertex_node_pt(fv[i]);
      for (int j = 0; j < 4; j++)
        if (nr_root_el->vertex_node_pt(j) == nd) { res.corner_map[fv[i]] = j; break; }
      if (res.corner_map[fv[i]] < 0) return res; // roots not actually face-sharing (should not happen)
    }

    // This element's my_face centroid -> my root frame -> (barycentric relabel) -> neighbour root frame.
    Vector<double> s_c(3), s_root(3);
    tet_face_centroid(my_face, s_c);
    this->local_coordinate_in_ancestor(s_c, my_root_el, s_root);
    const double b[4] = {s_root[0], s_root[1], s_root[2], 1.0 - s_root[0] - s_root[1] - s_root[2]};
    double nb_bary[4] = {0, 0, 0, 0};
    for (int d = 0; d < 4; d++)
      if (res.corner_map[d] >= 0) nb_bary[res.corner_map[d]] = b[d];
    Vector<double> s_nrroot(3);
    s_nrroot[0] = nb_bary[0]; s_nrroot[1] = nb_bary[1]; s_nrroot[2] = nb_bary[2];

    // Descend into the neighbour root toward that point, to at most THIS element's depth (climbs).
    Vector<double> s_leaf(3);
    int dep = 0;
    Tree *leaf = tet_descend_to_point(nr, s_nrroot, climbs, s_leaf, dep);
    if (!leaf || !leaf->is_leaf()) return res; // only a leaf can be a hang master; non-leaf => equal/finer
    res.el = dynamic_cast<RefineableTElement<3> *>(leaf->object_pt());
    res.diff_level = dep - climbs;
    res.cross_root = true;
    return res;
  }

  //=================================================================
  /// See declaration. Walks a tet subtree, testing every BUILT element for a node at the given local
  /// point and recursing into every son that contains it. Containment is a barycentric test on the exact
  /// affine son/father map and the node test compares LOCAL coordinates -- no positions anywhere.
  //=================================================================
  Node *RefineableTElement<3>::node_in_subtree_at_local_coordinate(Tree *node, const Vector<double> &s)
  {
    if (!node) return 0;
    FiniteElement *fe = dynamic_cast<FiniteElement *>(node->object_pt());
    RefineableElement *re = dynamic_cast<RefineableElement *>(node->object_pt());
    // An element whose nodes are not built yet has a null node_pt array -- skip it; its already-built
    // ancestor was tested on the way down and holds the same node.
    if (fe && re && re->nodes_built())
    {
      if (Node *n = fe->get_node_at_local_coordinate(s)) return n;
    }
    if (node->is_leaf()) return 0;
    const double tol = 1e-9;
    for (int c = 0; c < 8; c++) // tet 1->8 sons (4 corner tets + the 4 from the inner octahedron)
    {
      Tree *ch = node->son_pt(c);
      if (!ch) continue;
      Vector<double> sc(3);
      father_to_son_local(s, c, sc);
      const double b3 = 1.0 - sc[0] - sc[1] - sc[2];
      if (sc[0] < -tol || sc[1] < -tol || sc[2] < -tol || b3 < -tol) continue; // outside this son
      // No early break: a point on a face/edge shared by several sons lies in all of them, and only one
      // of them may hold the node (the others can be coarser).
      if (Node *n = node_in_subtree_at_local_coordinate(ch, sc)) return n;
    }
    return 0;
  }

  namespace
  {
    // Relabel a point given by its barycentric weights in tet root A's frame into root B's frame, by
    // matching the corners that CARRY WEIGHT via their NODE POINTERS. Corners with zero weight need not be
    // shared, which is exactly what lets a point on a shared EDGE hop between roots that have only that
    // edge in common. Returns false if a weight-carrying corner is not a corner of B.
    inline bool tet_point_into_other_root(oomph::RefineableTElement<3> *A, const oomph::Vector<double> &sA,
                                          oomph::RefineableTElement<3> *B, oomph::Vector<double> &sB)
    {
      if (!A || !B) return false;
      const double b[4] = {sA[0], sA[1], sA[2], 1.0 - sA[0] - sA[1] - sA[2]};
      double nb[4] = {0.0, 0.0, 0.0, 0.0};
      for (int d = 0; d < 4; d++)
      {
        if (std::abs(b[d]) < 1e-12) continue; // corner carries no weight -> irrelevant to this point
        oomph::Node *nd = A->vertex_node_pt(d);
        int j = -1;
        for (int k = 0; k < 4; k++)
          if (B->vertex_node_pt(k) == nd) { j = k; break; }
        if (j < 0) return false;
        nb[j] = b[d];
      }
      sB[0] = nb[0]; sB[1] = nb[1]; sB[2] = nb[2];
      return true;
    }
  } // namespace

  Node *RefineableTElement<3>::node_created_by_neighbour(const Vector<double> &s_son) const
  {
    using namespace oomph::OcTreeNames;
    if (!this->Tree_pt) return 0;
    // In a pyramid-rooted forest the tet tree-walk (son_vertices_in_father) does not apply: the ancestry
    // above a tet son of a pyramid is not the tet-in-tet map (son_type 6..9). Cross-parent node sharing
    // there is handled by the father-node registry instead, so defer to it.
    if (in_pyramid_forest()) return 0;

    TreeRoot *root = this->Tree_pt->root_pt();
    RefineableTElement<3> *my_root = dynamic_cast<RefineableTElement<3> *>(root->object_pt());
    if (!my_root) return 0;
    // The point in my own ROOT frame, once (the affine son/father maps are exact -- refinement is defined
    // in local coordinates), which is the frame every hop below starts from.
    Vector<double> s_root(3);
    this->local_coordinate_in_ancestor(s_son, my_root, s_root);

    // Breadth-first over roots: my own first, then out through root FACE neighbours, carrying the point
    // along by its barycentric weights on the shared corner NODES. Transitive, so a point on a root EDGE
    // reaches the whole fan of roots around that edge -- including those sharing no face with this one.
    static const int FACE_SLOT[4] = {L, R, D, U};
    std::vector<std::pair<Tree *, Vector<double>>> queue;
    std::set<Tree *> seen;
    queue.push_back(std::make_pair(static_cast<Tree *>(root), s_root));
    seen.insert(root);
    for (unsigned qi = 0; qi < queue.size(); qi++)
    {
      Tree *rt = queue[qi].first;
      const Vector<double> s = queue[qi].second; // by value: the push_back below can reallocate `queue`
      if (Node *n = node_in_subtree_at_local_coordinate(rt, s)) return n;
      RefineableTElement<3> *rt_el = dynamic_cast<RefineableTElement<3> *>(rt->object_pt());
      if (!rt_el) continue;
      const double b[4] = {s[0], s[1], s[2], 1.0 - s[0] - s[1] - s[2]};
      for (int f = 0; f < 4; f++)
      {
        if (std::abs(b[f]) > 1e-12) continue; // the point is not on root face f (opposite vertex f)
        Tree *nr = rt->root_pt()->neighbour_pt(FACE_SLOT[f]);
        if (!nr || seen.count(nr)) continue;
        RefineableTElement<3> *nr_el = dynamic_cast<RefineableTElement<3> *>(nr->object_pt());
        if (!nr_el) continue; // a foreign shape (mixed forest) -- handled by the registry, not here
        Vector<double> s_nr(3);
        if (!tet_point_into_other_root(rt_el, s, nr_el, s_nr)) continue;
        seen.insert(nr);
        queue.push_back(std::make_pair(nr, s_nr));
      }
    }
    return 0;
  }

  //=================================================================
  /// Map this element's local coordinate s into the coarse face-neighbour nb (affine same-root / corner-map
  /// relabel cross-root). Small shared helper for the face hang + node share. Returns false if not mappable.
  //=================================================================
  namespace
  {
    bool tet_map_into_face_neighbour(const RefineableTElement<3> *self,
                                     const RefineableTElement<3>::TetFaceNeighbour &nb,
                                     RefineableTElement<3> *my_root,
                                     const Vector<double> &s, Vector<double> &s_neigh)
    {
      if (!nb.cross_root)
        return self->local_coordinate_in_other_leaf(s, nb.el, s_neigh);
      if (!my_root) return false;
      Vector<double> s_root(3);
      self->local_coordinate_in_ancestor(s, my_root, s_root);
      const double br[4] = {s_root[0], s_root[1], s_root[2], 1.0 - s_root[0] - s_root[1] - s_root[2]};
      double nbb[4] = {0, 0, 0, 0};
      for (int d = 0; d < 4; d++)
        if (nb.corner_map[d] >= 0) nbb[nb.corner_map[d]] = br[d];
      Vector<double> s_nr(3);
      s_nr[0] = nbb[0]; s_nr[1] = nbb[1]; s_nr[2] = nbb[2];
      return RefineableTElement<3>::root_coord_to_leaf(s_nr, nb.el, s_neigh);
    }
  }

  //=================================================================
  /// Per-element FACE hanging. See the header. Mirrors the 2d tri_hang_helper but for a triangular face,
  /// using the element's own interpolating_basis so C1/C2/C2TB (enriched) traces are handled uniformly.
  //=================================================================
  void RefineableTElement<3>::tet_hang_face(const int &value_id, int my_face)
  {
    TetFaceNeighbour nb = this->tet_face_neighbour(my_face);
    if (nb.el == 0 || nb.diff_level >= 0) return; // only hang on a STRICTLY COARSER neighbour
    RefineableElement *neigh_re = dynamic_cast<RefineableElement *>(nb.el);
    if (!neigh_re) return;
    RefineableTElement<3> *my_root = nullptr;
    if (nb.cross_root)
    {
      my_root = dynamic_cast<RefineableTElement<3> *>(this->Tree_pt->root_pt()->object_pt());
      if (!my_root) return;
    }
    const unsigned nn = this->nnode();
    for (unsigned m = 0; m < nn; m++)
    {
      Vector<double> s(3);
      this->local_coordinate_of_node(m, s);
      const double bary[4] = {s[0], s[1], s[2], 1.0 - s[0] - s[1] - s[2]};
      if (std::abs(bary[my_face]) > 1e-10) continue; // node m not on this face
      Node *X = this->get_interpolating_node_at_local_coordinate(s, value_id);
      if (X == 0) continue;
      if (value_id >= 0 && (int)X->nvalue() <= value_id) continue; // node carries no dof in this separate slot
      if (X->is_hanging(value_id)) continue;                       // already constrained (another facet)
      Vector<double> s_neigh(3);
      if (!tet_map_into_face_neighbour(this, nb, my_root, s, s_neigh)) continue;
      // If the neighbour has its own interpolating node exactly here, X is shared -> not hanging.
      if (neigh_re->get_interpolating_node_at_local_coordinate(s_neigh, value_id) == X) continue;
      const unsigned nmax = neigh_re->ninterpolating_node(value_id);
      Shape psi(nmax);
      neigh_re->interpolating_basis(s_neigh, psi, value_id);
      unsigned nmaster = 0;
      for (unsigned k = 0; k < nmax; k++)
        if (std::abs(psi[k]) > 1e-12) nmaster++;
      if (nmaster == 0) continue;
      // Cycle guard (as in the 2d helper): skip if a master transitively hangs back on X.
      std::function<bool(Node *, Node *, int, int)> reaches = [&](Node *from, Node *to, int slot, int depth) -> bool {
        if (from == to) return true;
        if (depth > 30 || !from->is_hanging(slot)) return false;
        HangInfo *h = from->hanging_pt(slot);
        for (unsigned k = 0; k < h->nmaster(); k++)
          if (reaches(h->master_node_pt(k), to, slot, depth + 1)) return true;
        return false;
      };
      bool cyclic = false;
      for (unsigned k = 0; k < nmax && !cyclic; k++)
        if (std::abs(psi[k]) > 1e-12 && reaches(neigh_re->interpolating_node_pt(k, value_id), X, value_id, 0)) cyclic = true;
      if (cyclic) continue;
      HangInfo *hang = new HangInfo(nmaster);
      unsigned mm = 0;
      for (unsigned k = 0; k < nmax; k++)
        if (std::abs(psi[k]) > 1e-12)
        {
          hang->set_master_node_pt(mm, neigh_re->interpolating_node_pt(k, value_id), psi[k]);
          mm++;
        }
      X->set_hanging_pt(hang, value_id);
    }
  }

  //=================================================================
  /// Coarser-or-equal LEAF sharing this element's edge my_edge. OcTree ascent to the coarse edge (tracked
  /// as a vertex-index pair, climbing while it stays on a father edge) then a face-neighbour ring check.
  //=================================================================
  RefineableTElement<3>::TetEdgeNeighbour RefineableTElement<3>::tet_edge_neighbour(int my_edge) const
  {
    TetEdgeNeighbour res;
    if (!this->Tree_pt || my_edge < 0 || my_edge > 5) return res;
    int a = TET_EDGE_V[my_edge][0], b = TET_EDGE_V[my_edge][1];
    const Tree *cur = this->Tree_pt;
    int climbs = 0;
    while (true)
    {
      Tree *fath = cur->father_pt();
      if (!fath) break; // reached the root: cross-root coarse edge (handled via the ring below from `cur`)
      const int st = cur->son_type();
      Vector<Vector<double>> sv;
      son_vertices_in_father(st, sv);
      const int fpa = tet_fp_id(sv[a]), fpb = tet_fp_id(sv[b]);
      const int FE = tet_father_edge_of_pair(fpa, fpb);
      if (FE < 0) break; // edge interior to fath -> the coarse edge is (a,b) at `cur`'s level
      cur = fath;
      // Re-express the edge as the father-edge's two father VERTICES (indices 0..3) for the next climb.
      a = TET_EDGE_V[FE][0];
      b = TET_EDGE_V[FE][1];
      climbs++;
    }
    if (climbs == 0) return res; // a full edge of this element (or a refinement-diagonal): never hangs

    // Coarse edge (a,b) lives on `cur` (this element's ancestor, climbs levels up). Its endpoint NODES:
    RefineableTElement<3> *cur_el = dynamic_cast<RefineableTElement<3> *>(cur->object_pt());
    FiniteElement *cur_fe = dynamic_cast<FiniteElement *>(cur->object_pt());
    if (!cur_el || !cur_fe) return res;
    res.P = cur_el->vertex_node_pt(a);
    res.Q = cur_el->vertex_node_pt(b);
    if (!res.P || !res.Q) return res;
    // Coarse edge mid node (quadratic spaces): the node at the (a,b) midpoint local coordinate of cur.
    if (cur_fe->nnode_1d() > 2)
    {
      static const double VC[4][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}};
      Vector<double> smid(3);
      for (int d = 0; d < 3; d++) smid[d] = 0.5 * (VC[a][d] + VC[b][d]);
      res.M = cur_fe->get_node_at_local_coordinate(smid);
    }

    // Confirm a coarser LEAF actually shares edge (P,Q): check cur's two faces incident to the edge (a
    // standard 2:1 coarse neighbour is face-adjacent to cur across the edge). If found, this node hangs.
    for (int f = 0; f < 4; f++)
    {
      if (f == a || f == b) continue; // faces NOT incident to edge (a,b) are opposite a or b
      TetFaceNeighbour fn = cur_el->tet_face_neighbour(f);
      if (!fn.el || fn.diff_level > 0) continue; // finer or none
      FiniteElement *fe = dynamic_cast<FiniteElement *>(fn.el);
      if (!fe) continue;
      if (fe->get_node_number(res.P) != -1 && fe->get_node_number(res.Q) != -1)
      {
        res.el = fn.el;
        res.diff_level = fn.diff_level - climbs; // relative to THIS element
        return res;
      }
    }
    return res;
  }

  namespace
  {
    // Install the edge-interpolation hang for node X (slot value_id) at parameter t along coarse edge P->Q:
    // linear (1-t,t) on {P,Q}, or quadratic on {P,M,Q} for a quadratic space (the bubble vanishes on edges,
    // so this is the exact trace). Skips masters the node cannot carry (nvalue guard for separate slots).
    void install_tet_edge_hang(Node *X, int value_id, Node *P, Node *Q, Node *M, double t, bool quadratic)
    {
      const int slot = value_id;
      if (slot >= 0)
      {
        if ((int)X->nvalue() <= slot || (int)P->nvalue() <= slot || (int)Q->nvalue() <= slot) return;
        if (quadratic && M && (int)M->nvalue() <= slot) quadratic = false; // M carries no dof here -> linear
      }
      if (!quadratic || !M)
      {
        HangInfo *h = new HangInfo(2);
        h->set_master_node_pt(0, P, 1.0 - t);
        h->set_master_node_pt(1, Q, t);
        X->set_hanging_pt(h, value_id);
      }
      else
      {
        HangInfo *h = new HangInfo(3);
        h->set_master_node_pt(0, P, 2.0 * (t - 0.5) * (t - 1.0));
        h->set_master_node_pt(1, M, 4.0 * t * (1.0 - t));
        h->set_master_node_pt(2, Q, 2.0 * t * (t - 0.5));
        X->set_hanging_pt(h, value_id);
      }
    }
  }

  //=================================================================
  /// Per-element EDGE hanging. Hang this element's interpolating nodes strictly inside a coarser tet edge
  /// {P,Q}(,M) on that edge's interpolation (linear for a linear space, quadratic for C2/C2TB where the
  /// bubble vanishes on edges). The coarse edge nodes come topologically from tet_edge_neighbour; the
  /// parameter t along the (straight) coarse edge is the exact projection of X onto P->Q.
  //=================================================================
  void RefineableTElement<3>::tet_hang_edge(const int &value_id, int my_edge)
  {
    TetEdgeNeighbour nb = this->tet_edge_neighbour(my_edge);
    if (!nb.el || nb.diff_level >= 0 || !nb.P || !nb.Q) return; // only hang on a strictly coarser edge
    if (nb.P->is_hanging(value_id) || nb.Q->is_hanging(value_id)) return; // coarse masters must be real
    RefineableElement *neigh_re = dynamic_cast<RefineableElement *>(nb.el);
    if (!neigh_re) return;
    // Whether THIS value's space is quadratic along the edge (C2/C2TB: masters {P,M,Q}) or linear (C1
    // pressure: masters {P,Q}, and the edge-mid M itself is an ordinary slave). Keyed on the value_id's
    // own 1d interpolation order in the coarse element -- NOT the mesh's dominant space (a linear pressure
    // on a C2 mesh must still hang linearly).
    const bool quadratic = nb.M && neigh_re->ninterpolating_node_1d(value_id) > 2;
    const int va = TET_EDGE_V[my_edge][0], vb = TET_EDGE_V[my_edge][1];
    double den = 0.0;
    for (int d = 0; d < 3; d++) { const double e = nb.Q->x(d) - nb.P->x(d); den += e * e; }
    if (den < 1e-30) return;
    const unsigned nn = this->nnode();
    for (unsigned m = 0; m < nn; m++)
    {
      Vector<double> s(3);
      this->local_coordinate_of_node(m, s);
      // On this element's edge (va,vb)? barycentric of the two OFF-edge vertices must both be ~0.
      const double bary[4] = {s[0], s[1], s[2], 1.0 - s[0] - s[1] - s[2]};
      bool on_edge = true;
      for (int v = 0; v < 4; v++)
        if (v != va && v != vb && std::abs(bary[v]) > 1e-10) { on_edge = false; break; }
      if (!on_edge) continue;
      Node *X = this->get_interpolating_node_at_local_coordinate(s, value_id);
      if (X == 0 || X == nb.P || X == nb.Q) continue;
      if (quadratic && X == nb.M) continue;  // M is a real quadratic master (only for a quadratic value space)
      if (value_id >= 0 && (int)X->nvalue() <= value_id) continue; // node carries no dof in this separate slot
      if (X->is_hanging(value_id)) continue;                       // already constrained (e.g. by the face pass)
      double num = 0.0;
      for (int d = 0; d < 3; d++) num += (X->x(d) - nb.P->x(d)) * (nb.Q->x(d) - nb.P->x(d));
      const double t = num / den;
      if (t < 1e-7 || t > 1.0 - 1e-7) continue; // at an endpoint (real coarse node) -> not hanging
      install_tet_edge_hang(X, value_id, nb.P, nb.Q, nb.M, t, quadratic);
    }
  }

}
