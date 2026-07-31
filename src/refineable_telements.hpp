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

  public:
    // Result of the tri-native topological neighbour search (see tri_edge_neighbour).
    struct TriEdgeNeighbour
    {
      RefineableTElement<2> *el = nullptr; // >=-sized LEAF TRIANGLE neighbour across the queried edge (or null)
      int diff_level = 0;                  // level(neighbour)-level(this) along the edge; <0 => strictly coarser => this hangs
      bool cross_root = false;             // true if the neighbour lives in an adjacent tree root
      int my_edge_dir = -1;                // this element's edge direction expressed in ITS root frame (S/E/W)
      int nr_edge_dir = -1;                // the shared edge's direction in the NEIGHBOUR's root frame (S/E/W)
      bool reversed = false;               // whether the two roots parametrise the shared edge in opposite senses
      // Cross-SHAPE coarser neighbour (a QUAD at a mixed quad+tri interface): stored separately because it is
      // not a RefineableTElement<2>. Set only for a coarser (unrefined) cross-root quad; `el` stays null.
      // The tri hang path then uses my_edge_dir + the shape-agnostic BulkElementBase::mixed_hang_edge_node.
      oomph::RefineableElement *cross_shape_el = nullptr;
      // The neighbour QUAD's ROOT element (set whenever the cross-root neighbour is a quad, refined or not).
      // Node-sharing needs this (to descend the quad tree to the leaf at the shared-edge point) even when the
      // quad is equal/finer (not a coarse leaf, so cross_shape_el is null).
      oomph::RefineableElement *cross_shape_root = nullptr;
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

    // Tesselated-numpy export: for each of this triangle's 3 edges, find a strictly coarser edge neighbour
    // (a coarser tri, or a coarser quad across a mixed interface) via tri_edge_neighbour, compute each of
    // this element's edge nodes' LOCAL coordinate in that coarse neighbour TOPOLOGICALLY (the same affine
    // son/father maps, cross-root corner correspondence and cross-shape quad-root blend the hang path uses --
    // no physical positions, no locate_zeta), and register it there. The coarse element then places the node
    // by its reference-space edge/parameter, so the result is exact even on curved elements.
    void tess_register_on_coarser_for_numpy(std::vector<std::vector<std::set<Node *>>> &add_nodes);

  protected:
    // Reuse a node that a NEIGHBOUR element has already built, located via the tri tree (which persists
    // across adaptation rounds), like oomph's RefineableQElement::node_created_by_neighbour for quads.
    // This closes the cross-round gap of the per-round Shared_edge_node_registry: during transient
    // re-adaptation a son built this round must reuse a node an already-built neighbour holds at the same
    // topological point (else the shared vertex is duplicated and the moving mesh tears apart). s_son is
    // node i's local coordinate in THIS (son) element. Returns the shared node, or 0 if none.
    //
    // The search is over the WHOLE of this element's own tree root plus, if the point lies on a root edge,
    // the adjacent root(s) -- not just the >=-sized leaf neighbour that tri_edge_neighbour returns. That
    // matters because the holder of the node can be at any level and need not be a leaf: a neighbour that
    // was split earlier in this same round is not a leaf any more but still owns its nodes, and a neighbour
    // FINER than this element holds the point as a vertex several levels down. Both cases used to fall
    // through to a coordinate-based fallback. See node_in_subtree_at_local_coordinate.
    Node *node_created_by_neighbour(const Vector<double> &s_son) const;
    // Search a whole subtree for a node sitting at the point `s` (given in the local frame of the subtree's
    // OWN root element), and return it, or 0. Purely topological: it recurses into every son that contains
    // the point (`father_to_son_local` + a barycentric containment test, so a point on a son boundary
    // follows BOTH sons), and asks each element on the way for a node at the corresponding local coordinate
    // via get_node_at_local_coordinate -- which compares LOCAL coordinates, never physical positions.
    // Elements are tested at every level, not only at the leaves, because whether the point is a node
    // depends on the level: it is a mid-edge node of the element at this element's own level and a vertex
    // of that element's sons. Elements whose nodes are not built yet are skipped (their node_pt array is
    // still null) -- their built ancestor is tested first and holds the same node.
    static Node *node_in_subtree_at_local_coordinate(Tree *subtree, const Vector<double> &s);
    // Given a coordinate in THIS TREE's ROOT-element local frame, return the node sitting there, or null --
    // node_in_subtree_at_local_coordinate over this element's whole tree (so the holder may be at any
    // level, not only in the leaf the point falls in, and on a son boundary may be on either side). Used
    // for cross-shape (mixed quad+tri) node-sharing: a quad finds the node an adjacent refined tri already
    // built. Call on the root. Public because the quad node-sharing path
    // (BulkElementBase::mixed_quad_shared_node) calls it.
  public:
    Node *node_at_root_coordinate(const Vector<double> &s_root) const;
    // Descend to the LEAF containing a root-frame coordinate, returning it and the coordinate in its own
    // frame (for cross-shape HANGING: the quad hangs on this tri leaf's interpolating_basis at s_leaf).
    // Same topological point-in-simplex descent (father_to_son_local, no geometry); null if not resolvable.
    RefineableElement *leaf_at_root_coordinate(const Vector<double> &s_root, Vector<double> &s_leaf) const;

  protected:
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

  public:
    // The 4 vertices (in father local coordinates) of son number son_type (0..7) of a 1->8 split.
    static void son_vertices_in_father(int son_type, Vector<Vector<double>> &verts);

    // --- son<->father 3x3 affine coordinate map (the 3d analogue of the RefineableTElement<2> map) ---
    // A tet son (of the given son_type 0..7 of a 1->8 split) maps to its father by the barycentric affine
    // s_father = sum_i b_i(s_son) * son_vertex_father_coord_i. A full 3x3 A correctly represents ALL 8 sons,
    // including the 4 inner (octahedron) sons. These compose up/down the OcTree to map local coordinates
    // across refinement levels with NO locate_zeta -- the machinery the tet tree hang/neighbour path needs.
    static void son_to_father_local(const Vector<double> &s_son, int son_type, Vector<double> &s_father);
    static void father_to_son_local(const Vector<double> &s_father, int son_type, Vector<double> &s_son);
    // Map a local coordinate in THIS (leaf) element up to the local frame of `ancestor` (an ancestor in
    // the same tree), composing son_to_father_local up the tree.
    void local_coordinate_in_ancestor(const Vector<double> &s_here, RefineableTElement<3> *ancestor, Vector<double> &s_ancestor) const;
    // Map a local coordinate in THIS element into the local frame of `target`, both in the SAME tree root
    // (up to the common root, then down). Returns false if they are in different roots.
    bool local_coordinate_in_other_leaf(const Vector<double> &s_here, RefineableTElement<3> *target, Vector<double> &s_target) const;
    // Map a coordinate given in `leaf`'s ROOT-element frame down to `leaf`'s own frame (father_to_son chain).
    static bool root_coord_to_leaf(const Vector<double> &s_root, RefineableTElement<3> *leaf, Vector<double> &s_leaf);

    // Result of the tet-native topological neighbour search (see tet_face_neighbour). Same shape as the 2d
    // TriEdgeNeighbour but for a triangular FACE of a tetrahedron.
    struct TetFaceNeighbour
    {
      RefineableTElement<3> *el = nullptr; // >=-sized LEAF neighbour across the queried face (or null)
      int diff_level = 0;                  // level(neighbour)-level(this); <0 => strictly coarser => this hangs
      bool cross_root = false;             // neighbour lives in an adjacent tree root
      // Cross-root coordinate bridging: my ROOT vertex index -> the NEIGHBOUR ROOT vertex index, for the
      // three shared face corners (the fourth, opposite the shared face on each side, is -1). A point on the
      // shared face keeps its barycentric weights on the corresponding corners, so this permutation maps it
      // from my root frame into the neighbour root frame (the 3d analogue of the tri my/nr_edge_dir+reversed).
      int corner_map[4] = {-1, -1, -1, -1};
    };
    // Tet-native topological FACE-neighbour finder (the octree analogue of the 2d tri_edge_neighbour):
    // OcTree son_type ascent + father-point tracking. Tracks the queried face (my_face in 0..3, opposite
    // local vertex my_face) as a triple of tree-node local vertex indices; at each level maps them to
    // father points (a father vertex or a father edge-midpoint) via son_vertices_in_father; the face is
    // interior to the father exactly when its three father points do not all lie on one father face, at
    // which moment the >=-sized neighbour is the sibling son sharing that face -- so no oomph OcTree
    // Reflect/Rotate tables and the octahedron inner sons are handled naturally. (Same-root only for now;
    // cross-root + the descent land in a later step.)
    TetFaceNeighbour tet_face_neighbour(int my_face) const;
    // Reuse a node an already-built NEIGHBOUR has created, located via the OcTree (which persists across
    // adaptation rounds), the 3d analogue of RefineableTElement<2>::node_created_by_neighbour. Closes the
    // cross-round gap of the per-round Shared_edge_node_registry: during transient re-adaptation a son built
    // this round must reuse the node a neighbour built in an earlier round, else the shared node is
    // duplicated and a moving tet mesh tears apart. s_son is node j's local coordinate in THIS (son)
    // element. Same signature clash note as 2d: this hides the oomph base
    // node_created_by_neighbour(s_fraction,is_periodic).
    //
    // Purely topological, and complete for a pure tet forest: it searches this element's WHOLE tree root
    // (so the holder may sit at any level, need not be a leaf, and may be reached across an edge rather
    // than a face) and then walks out through root FACE neighbours, hopping barycentric weights from
    // corner NODE to corner NODE. The walk is transitive, which is what covers a node on a root EDGE: the
    // roots in the fan around that edge are reached by chaining face hops, even though most of them share
    // no face with this one. Returns 0 in a pyramid-rooted (mixed) forest -- see in_pyramid_forest().
    Node *node_created_by_neighbour(const Vector<double> &s_son) const;
    // Search a whole tet subtree for a node at the point `s` (in the local frame of the subtree's own root
    // element). Exactly RefineableTElement<2>::node_in_subtree_at_local_coordinate one dimension up: it
    // recurses into every son containing the point (father_to_son_local + a barycentric containment test,
    // so a point on a shared son face follows all of them) and asks every BUILT element on the way for a
    // node at the corresponding LOCAL coordinate. No physical position is ever compared.
    static Node *node_in_subtree_at_local_coordinate(Tree *subtree, const Vector<double> &s);

    // --- Per-element topological hanging (the tet tree route; 3d analogue of RefineableTElement<2>::
    // tri_hang_helper) --- Hang this element's interpolating nodes for `value_id` that lie on face
    // my_face (0..3) onto a STRICTLY COARSER face-neighbour, using the exact affine map for coordinates and
    // the oomph interpolating_basis for weights (so C1/C2/C2TB enriched traces are all handled by the
    // element's own facilities, never hand-written formulas). A finer neighbour hangs from its own side.
    void tet_hang_face(const int &value_id, int my_face);
    // True iff this tet's forest root is a pyramid (i.e. this is a tet son of the pyramid red split); the
    // tet-in-tet neighbour/hanging tree-walk does not apply then. See the .cpp.
    bool in_pyramid_forest() const;
    // Hang this element's interpolating nodes for `value_id` strictly inside a coarser tet EDGE. The coarse
    // edge {P,Q} (+ mid M for a quadratic space) and this node's parameter along it come from the OcTree
    // ascent + affine map (exact); a coarser leaf actually sharing {P,Q} is confirmed via tet_edge_neighbour.
    void tet_hang_edge(const int &value_id, int my_edge);
    // Coarser-or-equal LEAF sharing this element's edge my_edge (0..5), found by ascending the OcTree to the
    // coarse edge then walking the face-neighbour ring around it. Fills the coarse edge's endpoint/mid NODES
    // and this element's parameter interval, so tet_hang_edge can build the edge-interpolation hang directly.
    struct TetEdgeNeighbour
    {
      RefineableTElement<3> *el = nullptr; // a coarser LEAF sharing the coarse edge (null => none => no hang)
      int diff_level = 0;                  // level(neighbour)-level(this); <0 => strictly coarser => this hangs
      Node *P = nullptr;                   // coarse edge endpoint nodes (real, non-hanging)
      Node *Q = nullptr;
      Node *M = nullptr;                   // coarse edge mid node (quadratic spaces only; else null)
    };
    TetEdgeNeighbour tet_edge_neighbour(int my_edge) const;
  };

}
