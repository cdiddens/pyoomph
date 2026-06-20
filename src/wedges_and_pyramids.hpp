#pragma once

#include "oomph_lib.hpp"
#include "exception.hpp"
// Wedges and pyramids are not supported by oomph-lib, but the basic Element classes are defined in the oomph namespace here,
// they are adjusted for pyoomph in the elements.{cpp,hpp} files.
namespace oomph
{


class WedgeGaussC1 : public Integral
  {
  private:    
    static const unsigned Npts = 6;    
    static const double Knot[6][3], Weight[6];
  public:    
    WedgeGaussC1(){};
    WedgeGaussC1(const WedgeGaussC1& dummy) = delete;
    void operator=(const WedgeGaussC1&) = delete;
    unsigned nweight() const    {   return Npts;   }
    double knot(const unsigned& i, const unsigned& j) const   {    return Knot[i][j];  }
    double weight(const unsigned& i) const   {   return Weight[i];   }
 };


// No need to template these classes, wedges only exist in 3d, and the number of nodes along a line is either 2 or 3
class WedgeElementBase :  public virtual FiniteElement
{
  public:
    void build_face_element(const int& face_index,FaceElement* face_element_pt) override;
    CoordinateMappingFctPt face_to_bulk_coordinate_fct_pt(const int& face_index) const override;
    BulkCoordinateDerivativesFctPt bulk_coordinate_derivatives_fct_pt(const int& face_index) const override ;
    int face_outer_unit_normal_sign(const int&) const override;
    double s_min() const    { return 0.0; }
    double s_max() const    { return 1.0; }
    unsigned nvertex_node() const { return 6; }
    Node* vertex_node_pt(const unsigned& j) const
    {           
      if (j > 5)
      {
          std::ostringstream error_message;
          error_message  << "Element only has six vertex nodes; called with node number " << j << std::endl;
          throw OomphLibError(error_message.str(),OOMPH_CURRENT_FUNCTION,OOMPH_EXCEPTION_LOCATION);
      }
      return node_pt(j);
    }
    unsigned nnode_on_face() const override { throw_runtime_error("nnode_on_face cannot be implemented for a Wedge, damn."); }
    unsigned nnode_on_face_by_index(const int& face_index) const  { return (face_index<2) ? 3 : 4; }
};

class RefineableWedgeElement : public virtual RefineableElement, public virtual WedgeElementBase
  {

  public:
    /// \short Shorthand for pointer to an argument-free void member
    /// function of the refineable element
    typedef void (RefineableWedgeElement::*VoidMemberFctPt)();

    /// Constructor: Pass refinement level (default 0 = root)
    RefineableWedgeElement() : RefineableElement()
    {
    }

    /// Broken copy constructor
    RefineableWedgeElement(const RefineableWedgeElement &dummy)
    {
      BrokenCopy::broken_copy("RefineableWedgeElement");
    }

    virtual ~RefineableWedgeElement()
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


class WedgeElementC1 :  public virtual RefineableWedgeElement
{
 // A first order wedge element 
 // Nodes are ordered as follows:
 /*
 Index : Local coordinates (s0,s1,s2)
   0: (0,0,0)
   1: (1,0,0)
   2: (0,1,0)

   3: (0,0,1)
   4: (1,0,1)
   5: (0,1,1)


              5 o
               /\
              /  \
             /    \
            / [1]  \
         3 o--------o 4
           |  [3]   |
   [2]     |  2 o   |    [4]
           |   /\   |
           |  /  \  |
           | /    \ |
           |/  [0] \|
         0 o--------o 1

    s[i] runs from 0 to 1, but with the constraint that s[0]+s[1] <= 1

    facets are numbered as follows:
    facet 0: s[2] = 0, nodes 0,1,2
    facet 1: s[2] = 1, nodes 3,4,5
    facet 2: s[0] = 0, nodes 0,2,3,5
    facet 3: s[1] = 0, nodes 0,1,3,4
    facet 4: s[0]+s[1] = 1, nodes 1,2,4,5
 */

 private:
    static WedgeGaussC1 Default_integration_scheme;
 public:    
    unsigned nnode_1d() const { return 2;}

    WedgeElementC1() : WedgeElementBase() 
    {
        set_n_node(6);
        set_dimension(3);
        set_integration_scheme(&Default_integration_scheme);
    }
    WedgeElementC1(const WedgeElementC1&) = delete;
    ~WedgeElementC1() {}

    unsigned int get_bulk_node_number(const int & face_index, const unsigned int& i) const override;
        
    inline void shape(const Vector<double>& s, Shape& psi) const
    {
        const double s0 = s[0];
        const double s1 = s[1];
        const double s2 = s[2];
        const double l1 = 1.0 - s0 - s1;
        psi[0] = l1 * (1.0 - s2);
        psi[1] = s0 * (1.0 - s2);
        psi[2] = s1 * (1.0 - s2);
        psi[3] = l1 * s2;
        psi[4] = s0 * s2;
        psi[5] = s1 * s2;
    }

    
    inline void dshape_local(const Vector<double>& s,Shape& psi,DShape& dpsids) const
    {
        const double s0 = s[0];
        const double s1 = s[1];
        const double s2 = s[2];
        const double l1 = 1.0 - s0 - s1;

        // Shape functions
        psi[0] = l1 * (1.0 - s2);
        psi[1] = s0 * (1.0 - s2);
        psi[2] = s1 * (1.0 - s2);

        psi[3] = l1 * s2;
        psi[4] = s0 * s2;
        psi[5] = s1 * s2;

        // Derivatives wrt s0
        dpsids(0,0) = -(1.0 - s2);
        dpsids(1,0) =  (1.0 - s2);
        dpsids(2,0) =  0.0;
        dpsids(3,0) = -s2;
        dpsids(4,0) =  s2;
        dpsids(5,0) =  0.0;

        // Derivatives wrt s1
        dpsids(0,1) = -(1.0 - s2);
        dpsids(1,1) =  0.0;
        dpsids(2,1) =  (1.0 - s2);
        dpsids(3,1) = -s2;
        dpsids(4,1) =  0.0;
        dpsids(5,1) =  s2;

        // Derivatives wrt s2
        dpsids(0,2) = -l1;
        dpsids(1,2) = -s0;
        dpsids(2,2) = -s1;
        dpsids(3,2) =  l1;
        dpsids(4,2) =  s0;
        dpsids(5,2) =  s1;
    }

    inline void local_coordinate_of_node(const unsigned& j,Vector<double>& s) const
    {
      if (j==0) { s[0] = 0.0; s[1] = 0.0; s[2] = 0.0;}
      else if (j==1) { s[0] = 1.0; s[1] = 0.0; s[2] = 0.0;}
      else if (j==2) { s[0] = 0.0; s[1] = 1.0; s[2] = 0.0;}
      else if (j==3) { s[0] = 0.0; s[1] = 0.0; s[2] = 1.0;}
      else if (j==4) { s[0] = 1.0; s[1] = 0.0; s[2] = 1.0;}
      else if (j==5) { s[0] = 0.0; s[1] = 1.0; s[2] = 1.0;}
      else
      {
          std::ostringstream error_message;
          error_message  << "Element only has six nodes; called with node number " << j << std::endl;
          throw OomphLibError(error_message.str(),OOMPH_CURRENT_FUNCTION,OOMPH_EXCEPTION_LOCATION);
      }
    }


};


}