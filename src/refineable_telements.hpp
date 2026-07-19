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

    unsigned required_nsons() const override
    {
      throw_runtime_error("TODO");
      return 4;
    }

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
  };

}
