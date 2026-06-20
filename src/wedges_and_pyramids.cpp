#include "wedges_and_pyramids.hpp"

namespace oomph
{ 

const double WedgeGaussC1::Knot[6][3] =
{
  {1.0/6.0, 1.0/6.0, (1.0-1.0/sqrt(3.0))/2.0},
  {2.0/3.0, 1.0/6.0, (1.0-1.0/sqrt(3.0))/2.0},
  {1.0/6.0, 2.0/3.0, (1.0-1.0/sqrt(3.0))/2.0},

  {1.0/6.0, 1.0/6.0, (1.0+1.0/sqrt(3.0))/2.0},
  {2.0/3.0, 1.0/6.0, (1.0+1.0/sqrt(3.0))/2.0},
  {1.0/6.0, 2.0/3.0, (1.0+1.0/sqrt(3.0))/2.0}
};

const double WedgeGaussC1::Weight[6] =
{
  1.0/12.0,
  1.0/12.0,
  1.0/12.0,
  1.0/12.0,
  1.0/12.0,
  1.0/12.0
};




 void RefineableWedgeElement::setup_father_bounds()
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
  void RefineableWedgeElement::get_bcs(int bound, Vector<int> &bound_cons) const
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
  void RefineableWedgeElement::get_edge_bcs(const int &edge, Vector<int> &bound_cons) const
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
  void RefineableWedgeElement::get_boundaries(const int &edge,
                                             std::set<unsigned> &boundary) const
  {
    throw_runtime_error("Implement");
  }

  //===================================================================
  /// Return the value of the intrinsic boundary coordinate interpolated
  /// along the edge (S/W/N/E)
  //===================================================================
  void RefineableWedgeElement::
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
  Node *RefineableWedgeElement::
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
  void RefineableWedgeElement::build(Mesh *&mesh_pt,
                                    Vector<Node *> &new_node_pt,
                                    bool &was_already_built,
                                    std::ofstream &new_nodes_file)
  {
    throw_runtime_error("Implement");
  }

  //====================================================================
  ///  Print corner nodes, use colour (default "BLACK")
  //====================================================================
  void RefineableWedgeElement::output_corners(std::ostream &outfile,
                                             const std::string &colour) const
  {
    throw_runtime_error("Implement");
  }

  //====================================================================
  /// Set up all hanging nodes. If we are documenting the output then
  /// open the output files and pass the open files to the helper function
  //====================================================================
  void RefineableWedgeElement::setup_hanging_nodes(Vector<std::ofstream *>
                                                      &output_stream)
  {
    throw_runtime_error("Implement");
  }

  //================================================================
  /// Internal function that sets up the hanging node scheme for
  /// a particular continuously interpolated value
  //===============================================================
  void RefineableWedgeElement::setup_hang_for_value(const int &value_id)
  {
    throw_runtime_error("Implement");
  }

  //=================================================================
  /// Internal function to set up the hanging nodes on a particular
  /// edge of the element
  //=================================================================
  void RefineableWedgeElement::
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
  void RefineableWedgeElement::check_integrity(double &max_error)
  {

    throw_runtime_error("Implement");
  }

  //========================================================================
  /// Static matrix for coincidence between son nodal points and
  /// father boundaries
  ///
  //========================================================================
  std::map<unsigned, DenseMatrix<int>> RefineableWedgeElement::Father_bound;





  WedgeGaussC1  WedgeElementC1::Default_integration_scheme;
}