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


/*
Dummy stuff to merge TElements with RefineableElements
*/
#pragma once
#include "oomph_lib.hpp"
#include "Telements.h"
#include "exception.hpp"

namespace oomph
{

  template <unsigned DIM>
  class RefineableTElement
  {
  public:
    RefineableTElement() {}
  };

  template <>
  class RefineableTElement<1> : public virtual RefineableElement,
                                public virtual TElementBase
  {

  public:
    /// \short Shorthand for pointer to an argument-free void member
    /// function of the refineable element
    typedef void (RefineableTElement<1>::*VoidMemberFctPt)();

    /// Constructor: Pass refinement level (default 0 = root)
    RefineableTElement() : RefineableElement()
    {
    }

    /// Broken copy constructor
    RefineableTElement(const RefineableTElement<1> &)
    {
      BrokenCopy::broken_copy("RefineableTElement<1>");
    }

    ~RefineableTElement() override
    {
    }

    unsigned required_nsons() const override { return 2; }

    // Return the node already created by a neighbouring element at fractional local
    // position s_fraction (if any), so it can be shared rather than duplicated; see .cpp.
    virtual Node *node_created_by_neighbour(const Vector<double> &s_fraction, bool &is_periodic);

    // As above, but checking sons of neighbours (not yet implemented; always returns 0 here).
    virtual Node *node_created_by_son_of_neighbour(const Vector<double> &, bool &)
    {
      return 0;
    }

    // Build this (son) element from its father during refinement: establish node pointers,
    // creating shared/new nodes as needed, and set up boundary/periodicity info; see .cpp.
    void build(Mesh *&mesh_pt, Vector<Node *> &new_node_pt, bool &was_already_built, std::ofstream &new_nodes_file) override;

    // Check inter-element continuity of nodal positions and interpolated values.
    void check_integrity(double &max_error) override;

    // Debug output of the element's corner node positions.
    void output_corners(std::ostream &outfile, const std::string &colour) const;

    BinaryTree *binarytree_pt() { return dynamic_cast<BinaryTree *>(Tree_pt); }

    BinaryTree *binarytree_pt() const { return dynamic_cast<BinaryTree *>(Tree_pt); }

    // Set up all hanging nodes of this element (opens/passes debug output streams if given).
    void setup_hanging_nodes(Vector<std::ofstream *> &output_stream) override;

    // Element-type-specific part of the hanging node setup; must be overridden by concrete elements.
    void further_setup_hanging_nodes() override = 0;

  protected:
    // Static lookup table (keyed by nnode_1d) encoding, for each son type and local node,
    // which boundary/vertex of the father element that son node coincides with (see .cpp).
    static std::map<unsigned, DenseMatrix<int>> Father_bound;

    // Populate Father_bound for this element's node count (called lazily on first use).
    void setup_father_bounds();

    // Boundary conditions along a whole element edge (least restrictive combination of its nodes).
    void get_edge_bcs(const int &edge, Vector<int> &bound_cons) const;

  public:
    // Mesh-boundary indices that a given local edge/vertex lies on.
    void get_boundaries(const int &edge, std::set<unsigned> &boundaries) const;

    // Boundary conditions at an edge/vertex, combining adjacent-edge BCs at vertices.
    void get_bcs(int bound, Vector<int> &bound_cons) const;
    // Intrinsic boundary coordinate interpolated to local position s along an edge.
    void interpolated_zeta_on_edge(const unsigned &boundary, const int &edge, const Vector<double> &s, Vector<double> &zeta);

  protected:
    // Set up the hanging-node scheme for one continuously-interpolated value.
    void setup_hang_for_value(const int &value_id);

    // Set up hanging nodes on one particular edge of the element for a given value.
    virtual void quad_hang_helper(const int &value_id, const int &my_edge, std::ofstream &output_hangfile);
  };

  template <>
  class RefineableTElement<2> : public virtual RefineableElement,
                                public virtual TElementBase
  {

  public:
    /// \short Shorthand for pointer to an argument-free void member
    /// function of the refineable element
    typedef void (RefineableTElement<2>::*VoidMemberFctPt)();

    /// Constructor: Pass refinement level (default 0 = root)
    RefineableTElement() : RefineableElement()
    {
    }

    /// Broken copy constructor
    RefineableTElement(const RefineableTElement<2> &)
    {
      BrokenCopy::broken_copy("RefineableTElement<2>");
    }

    ~RefineableTElement() override
    {
    }

    unsigned required_nsons() const override { return 4; }

    // Return the node already created by a neighbouring element at fractional local
    // position s_fraction (if any), so it can be shared rather than duplicated; see .cpp.
    virtual Node *node_created_by_neighbour(const Vector<double> &s_fraction, bool &is_periodic);

    // As above, but checking sons of neighbours (not yet implemented; always returns 0 here).
    virtual Node *node_created_by_son_of_neighbour(const Vector<double> &, bool &)
    {
      return 0;
    }

    // Build this (son) element from its father during refinement: establish node pointers,
    // creating shared/new nodes as needed, and set up boundary/periodicity info; see .cpp.
    void build(Mesh *&mesh_pt, Vector<Node *> &new_node_pt, bool &was_already_built, std::ofstream &new_nodes_file) override;

    // Check inter-element continuity of nodal positions and interpolated values.
    void check_integrity(double &max_error) override;

    // Debug output of the element's corner node positions.
    void output_corners(std::ostream &outfile, const std::string &colour) const;

    QuadTree *quadtree_pt() { return dynamic_cast<QuadTree *>(Tree_pt); }

    QuadTree *quadtree_pt() const { return dynamic_cast<QuadTree *>(Tree_pt); }

    // Set up all hanging nodes of this element (opens/passes debug output streams if given).
    void setup_hanging_nodes(Vector<std::ofstream *> &output_stream) override;

    // Element-type-specific part of the hanging node setup; must be overridden by concrete elements.
    void further_setup_hanging_nodes() override = 0;

  protected:
    // Static lookup table (keyed by nnode_1d) encoding, for each son type and local node,
    // which boundary/vertex of the father element that son node coincides with (see .cpp).
    static std::map<unsigned, DenseMatrix<int>> Father_bound;

    // Populate Father_bound for this element's node count (called lazily on first use).
    void setup_father_bounds();

    // Boundary conditions along a whole element edge (least restrictive combination of its nodes).
    void get_edge_bcs(const int &edge, Vector<int> &bound_cons) const;

  public:
    // Mesh-boundary indices that a given local edge/vertex lies on.
    void get_boundaries(const int &edge, std::set<unsigned> &boundaries) const;

    // Boundary conditions at an edge/vertex, combining adjacent-edge BCs at vertices.
    void get_bcs(int bound, Vector<int> &bound_cons) const;
    // Intrinsic boundary coordinate interpolated to local position s along an edge.
    void interpolated_zeta_on_edge(const unsigned &boundary, const int &edge, const Vector<double> &s, Vector<double> &zeta);

  protected:
    // Set up the hanging-node scheme for one continuously-interpolated value.
    void setup_hang_for_value(const int &value_id);

    // Set up hanging nodes on one particular edge of the element for a given value.
    virtual void quad_hang_helper(const int &value_id, const int &my_edge, std::ofstream &output_hangfile);

    // Tree-based hanging on one triangle edge (my_edge = S/E/W for tri edges 0-1/1-2/2-0). Finds the
    // coarser neighbour via the QuadTree (gteq_edge_neighbour, topology only), then hangs each of this
    // element's interpolating edge nodes that is interior to the neighbour's edge onto the neighbour's
    // interpolating basis -- coordinates via locate_zeta (curvature-robust), not the quad compass
    // coordinate descent (geometrically wrong for triangles).
    void tri_hang_helper(const int &value_id, const int &my_edge);

    // Result of the tri-native topological neighbour search (see tri_edge_neighbour).
    struct TriEdgeNeighbour
    {
      RefineableTElement<2> *el = nullptr; // >=-sized LEAF neighbour across the queried edge (or null)
      int diff_level = 0;                  // level(neighbour)-level(this) along the edge; <0 => strictly coarser => this hangs
      bool cross_root = false;             // true if the neighbour lives in an adjacent tree root
      int my_edge_dir = -1;                // this element's edge direction expressed in ITS root frame (S/E/W)
      int nr_edge_dir = -1;                // the shared edge's direction in the NEIGHBOUR's root frame (S/E/W)
      bool reversed = false;               // whether the two roots parametrise the shared edge in opposite senses
    };

    // Tri-native topological neighbour finder (the "oomph way": son_type ascent, NOT the geometric
    // node_strictly_on_* scheme, NOT locate_zeta). Tracks the queried edge as a pair of tree-node local
    // vertex indices and, at each level, maps them to father points (a father vertex or a father edge
    // midpoint) via the build()'s s_in_parent data; the edge is interior to the father exactly when the
    // two father points lie on different father edges, at which moment the >=-sized neighbour is simply
    // the sibling son -- so NO quad Reflect/Rotate/Is_adjacent tables are needed and the inverted middle
    // (NE) son is handled correctly. If the ascent reaches the root with the edge still on the boundary,
    // the neighbour lives in the adjacent root (found topologically from shared vertex nodes) and we
    // descend into it along the shared edge. Returns the >=-sized leaf neighbour + the data needed to
    // bridge coordinates across roots with the affine map (never locate_zeta).
    TriEdgeNeighbour tri_edge_neighbour(int my_edge) const;
    // Map a coordinate given in `leaf`'s ROOT-element local frame down to `leaf`'s own local frame, by
    // composing father_to_son_local along leaf's son_type chain (used for cross-root coordinate bridging).
    static bool root_coord_to_leaf(const Vector<double> &s_root, RefineableTElement<2> *leaf, Vector<double> &s_leaf);

    // Affine map of a local coordinate between a son element (of the given son_type) and its father,
    // s_father = A*s_son + b (and its inverse). Unlike oomph's quad box representation
    // (s_lo/s_hi/translate_s), a full 2x2 A correctly represents ALL four triangle sons including the
    // inverted middle (NE) son. These compose up/down the tree to map coordinates across refinement
    // levels with no locate_zeta. son_type is a QuadTreeNames SW/SE/NE/NW value.
    static void son_to_father_local(const Vector<double> &s_son, int son_type, Vector<double> &s_father);
    static void father_to_son_local(const Vector<double> &s_father, int son_type, Vector<double> &s_son);
    // Map a local coordinate in THIS (leaf) element up to the local coordinate system of `ancestor`
    // (an ancestor element of this one in the same tree), by composing son_to_father_local up the tree.
    void local_coordinate_in_ancestor(const Vector<double> &s_here, RefineableTElement<2> *ancestor, Vector<double> &s_ancestor) const;
    // Map a local coordinate in THIS element into the local coordinates of `target` when both share the
    // SAME tree root (up to the common root, then down): compose son_to_father up + father_to_son down.
    // Returns false if the two elements are in different tree roots (cross-root handled elsewhere).
    bool local_coordinate_in_other_leaf(const Vector<double> &s_here, RefineableTElement<2> *target, Vector<double> &s_target) const;

  public:
    // Clear the shared-node registry (call once before each refinement round). See below.
    static void clear_shared_edge_node_registry() { Shared_edge_node_registry.clear(); }

  protected:
    // --- Geometric node-sharing during triangle refinement (Phase 2, branch mixed_adapt) ---
    // Instead of oomph's quad compass neighbour finding (geometrically wrong for triangles), a
    // newly created son node lying on a father edge is keyed by the (unordered) pair of that
    // father edge's two corner nodes. That key is identical for every element touching the edge
    // (both sons of the same father and sons of the edge-sharing neighbour father), so a shared
    // registry lets them all reuse a single node instead of duplicating it. Valid for linear
    // (3-node) triangles, where each father edge spawns exactly one new (mid-edge) node. Cleared
    // at the start of each refinement round by TemplatedMeshBase::split_elements_if_required.
    static std::map<std::set<Node *>, Node *> Shared_edge_node_registry;
    // Registry key (father node-pointer pair that this son node bisects) for a new son node at
    // father-local coordinate s_in_father; empty if it is not the midpoint of a father-node pair.
    std::set<Node *> father_edge_node_key(const Vector<double> &s_in_father, RefineableTElement<2> *father_el_pt) const;
  };

  template <>
  class RefineableTElement<3> : public virtual RefineableElement,
                                public virtual TElementBase
  {

  public:
    /// \short Shorthand for pointer to an argument-free void member
    /// function of the refineable element
    typedef void (RefineableTElement<3>::*VoidMemberFctPt)();

    /// Constructor: Pass refinement level (default 0 = root)
    RefineableTElement() : RefineableElement()
    {
    }

    /// Broken copy constructor
    RefineableTElement(const RefineableTElement<3> &)
    {
      BrokenCopy::broken_copy("RefineableTElement<3>");
    }

    ~RefineableTElement() override
    {
    }

    unsigned required_nsons() const override { return 8; } // a tetrahedron refines 1->8

    // Clear the shared-node registry (call once before each refinement round). See below.
    static void clear_shared_edge_node_registry() { Shared_edge_node_registry.clear(); }

    // Return the node already created by a neighbouring element at fractional local
    // position s_fraction (if any), so it can be shared rather than duplicated; see .cpp.
    virtual Node *node_created_by_neighbour(const Vector<double> &s_fraction, bool &is_periodic);

    // As above, but checking sons of neighbours (not yet implemented; always returns 0 here).
    virtual Node *node_created_by_son_of_neighbour(const Vector<double> &, bool &)
    {
      return 0;
    }

    // Build this (son) element from its father during refinement: establish node pointers,
    // creating shared/new nodes as needed, and set up boundary/periodicity info; see .cpp.
    void build(Mesh *&mesh_pt, Vector<Node *> &new_node_pt, bool &was_already_built, std::ofstream &new_nodes_file) override;

    // Check inter-element continuity of nodal positions and interpolated values.
    void check_integrity(double &max_error) override;

    // Debug output of the element's corner node positions.
    void output_corners(std::ostream &outfile, const std::string &colour) const;

    OcTree *octree_pt() { return dynamic_cast<OcTree *>(Tree_pt); }

    OcTree *octree_pt() const { return dynamic_cast<OcTree *>(Tree_pt); }

    // Set up all hanging nodes of this element (opens/passes debug output streams if given).
    void setup_hanging_nodes(Vector<std::ofstream *> &output_stream) override;

    // Element-type-specific part of the hanging node setup; must be overridden by concrete elements.
    void further_setup_hanging_nodes() override = 0;

  protected:
    // Static lookup table (keyed by nnode_1d) encoding, for each son type and local node,
    // which boundary/vertex of the father element that son node coincides with (see .cpp).
    static std::map<unsigned, DenseMatrix<int>> Father_bound;

    // Populate Father_bound for this element's node count (called lazily on first use).
    void setup_father_bounds();

    // Boundary conditions along a whole element edge (least restrictive combination of its nodes).
    void get_edge_bcs(const int &edge, Vector<int> &bound_cons) const;

  public:
    // Mesh-boundary indices that a given local edge/vertex lies on.
    void get_boundaries(const int &edge, std::set<unsigned> &boundaries) const;

    // Boundary conditions at an edge/vertex, combining adjacent-edge BCs at vertices.
    void get_bcs(int bound, Vector<int> &bound_cons) const;
    // Intrinsic boundary coordinate interpolated to local position s along an edge.
    void interpolated_zeta_on_edge(const unsigned &boundary, const int &edge, const Vector<double> &s, Vector<double> &zeta);

  protected:
    // Set up the hanging-node scheme for one continuously-interpolated value.
    void setup_hang_for_value(const int &value_id);

    // Set up hanging nodes on one particular edge of the element for a given value.
    virtual void quad_hang_helper(const int &value_id, const int &my_edge, std::ofstream &output_hangfile);

    // --- Geometric node-sharing during tetrahedron refinement (Phase 4, branch mixed_adapt) ---
    // Direct 3D analog of the 2D triangle scheme: every node a 1->8 tet refinement creates is the
    // midpoint of two father nodes (an edge midpoint), so it is keyed by that father node-pointer
    // pair, identical from every element that creates it (sibling sons and the face-sharing
    // neighbour father's sons), giving duplicate-free node-sharing with no coordinate descent.
    // Cleared each refinement round by TemplatedMeshBase::split_elements_if_required.
    static std::map<std::set<Node *>, Node *> Shared_edge_node_registry;
    // The father node-pointer pair that a son node at father-local coordinate s_in_father bisects
    // (empty if it is a reused father node).
    std::set<Node *> father_edge_node_key(const Vector<double> &s_in_father, RefineableTElement<3> *father_el_pt) const;
    // The 4 vertices (in father local coordinates) of son number son_type (0..7) of a 1->8 split.
    void son_vertices_in_father(int son_type, Vector<Vector<double>> &verts) const;
  };

}
