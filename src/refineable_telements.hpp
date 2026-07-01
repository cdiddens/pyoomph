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

The authors may be contacted at c.diddens@utwente.nl and d.rocha@utwente.nl

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
    RefineableTElement(const RefineableTElement<1> &dummy)
    {
      BrokenCopy::broken_copy("RefineableTElement<1>");
    }

    virtual ~RefineableTElement()
    {
    }

    unsigned required_nsons() const { return 2; }

    virtual Node *node_created_by_neighbour(const Vector<double> &s_fraction, bool &is_periodic);

    virtual Node *node_created_by_son_of_neighbour(const Vector<double> &s_fraction, bool &is_periodic)
    {
      return 0;
    }

    virtual void build(Mesh *&mesh_pt, Vector<Node *> &new_node_pt, bool &was_already_built, std::ofstream &new_nodes_file);

    void check_integrity(double &max_error);

    void output_corners(std::ostream &outfile, const std::string &colour) const;

    BinaryTree *binarytree_pt() { return dynamic_cast<BinaryTree *>(Tree_pt); }

    BinaryTree *binarytree_pt() const { return dynamic_cast<BinaryTree *>(Tree_pt); }

    void setup_hanging_nodes(Vector<std::ofstream *> &output_stream);

    virtual void further_setup_hanging_nodes() = 0;

  protected:
    static std::map<unsigned, DenseMatrix<int>> Father_bound;

    void setup_father_bounds();

    void get_edge_bcs(const int &edge, Vector<int> &bound_cons) const;

  public:
    void get_boundaries(const int &edge, std::set<unsigned> &boundaries) const;

    void get_bcs(int bound, Vector<int> &bound_cons) const;
    void interpolated_zeta_on_edge(const unsigned &boundary, const int &edge, const Vector<double> &s, Vector<double> &zeta);

  protected:
    void setup_hang_for_value(const int &value_id);

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
    RefineableTElement(const RefineableTElement<2> &dummy)
    {
      BrokenCopy::broken_copy("RefineableTElement<2>");
    }

    virtual ~RefineableTElement()
    {
    }

    unsigned required_nsons() const { return 4; }

    virtual Node *node_created_by_neighbour(const Vector<double> &s_fraction, bool &is_periodic);

    virtual Node *node_created_by_son_of_neighbour(const Vector<double> &s_fraction, bool &is_periodic)
    {
      return 0;
    }

    virtual void build(Mesh *&mesh_pt, Vector<Node *> &new_node_pt, bool &was_already_built, std::ofstream &new_nodes_file);

    void check_integrity(double &max_error);

    void output_corners(std::ostream &outfile, const std::string &colour) const;

    QuadTree *quadtree_pt() { return dynamic_cast<QuadTree *>(Tree_pt); }

    QuadTree *quadtree_pt() const { return dynamic_cast<QuadTree *>(Tree_pt); }

    void setup_hanging_nodes(Vector<std::ofstream *> &output_stream);

    virtual void further_setup_hanging_nodes() = 0;

  protected:
    static std::map<unsigned, DenseMatrix<int>> Father_bound;

    void setup_father_bounds();

    void get_edge_bcs(const int &edge, Vector<int> &bound_cons) const;

  public:
    void get_boundaries(const int &edge, std::set<unsigned> &boundaries) const;

    void get_bcs(int bound, Vector<int> &bound_cons) const;
    void interpolated_zeta_on_edge(const unsigned &boundary, const int &edge, const Vector<double> &s, Vector<double> &zeta);

  protected:
    void setup_hang_for_value(const int &value_id);

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
    RefineableTElement(const RefineableTElement<3> &dummy)
    {
      BrokenCopy::broken_copy("RefineableTElement<3>");
    }

    virtual ~RefineableTElement()
    {
    }

    unsigned required_nsons() const
    {
      throw_runtime_error("TODO");
      return 4;
    }

    virtual Node *node_created_by_neighbour(const Vector<double> &s_fraction, bool &is_periodic);

    virtual Node *node_created_by_son_of_neighbour(const Vector<double> &s_fraction, bool &is_periodic)
    {
      return 0;
    }

    virtual void build(Mesh *&mesh_pt, Vector<Node *> &new_node_pt, bool &was_already_built, std::ofstream &new_nodes_file);

    void check_integrity(double &max_error);

    void output_corners(std::ostream &outfile, const std::string &colour) const;

    OcTree *octree_pt() { return dynamic_cast<OcTree *>(Tree_pt); }

    OcTree *octree_pt() const { return dynamic_cast<OcTree *>(Tree_pt); }

    void setup_hanging_nodes(Vector<std::ofstream *> &output_stream);

    virtual void further_setup_hanging_nodes() = 0;

  protected:
    static std::map<unsigned, DenseMatrix<int>> Father_bound;

    void setup_father_bounds();

    void get_edge_bcs(const int &edge, Vector<int> &bound_cons) const;

  public:
    void get_boundaries(const int &edge, std::set<unsigned> &boundaries) const;

    void get_bcs(int bound, Vector<int> &bound_cons) const;
    void interpolated_zeta_on_edge(const unsigned &boundary, const int &edge, const Vector<double> &s, Vector<double> &zeta);

  protected:
    void setup_hang_for_value(const int &value_id);

    virtual void quad_hang_helper(const int &value_id, const int &my_edge, std::ofstream &output_hangfile);
  };

}
