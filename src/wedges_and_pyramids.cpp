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





  


  //////////////////

  WedgeGaussC1  WedgeElementC1::Default_integration_scheme;

  unsigned int WedgeElementC1::get_bulk_node_number(const int & face_index, const unsigned int& i) const
  {
    if (face_index==0 && i<3) { return i; }
    else if (face_index==1 && i<3) { return i+3; }
    else if (face_index==2) {  // 3 0 5 2
        switch (i)
        {
            case 0: return 3;
            case 1: return 0;
            case 2: return 5;
            case 3: return 2;
            default: throw_runtime_error("Invalid node index for face");
        }
    }
    else if (face_index==3) {  // 1 0 4 3
        switch (i)
        {
            case 0: return 1;
            case 1: return 0;
            case 2: return 4;
            case 3: return 3;
            default: throw_runtime_error("Invalid node index for face");
        }
    }
    else if (face_index==4) {  // 1 4 2 5
        switch (i)
        {
            case 0: return 1;
            case 1: return 4;
            case 2: return 2;
            case 3: return 5;
            default: throw_runtime_error("Invalid node index for face");
        }
    }
    
    throw_runtime_error("Invalid node or face index for wedge element "+std::to_string(face_index)+", "+std::to_string(i));
    return 0;
  }

  void WedgeElementBase::build_face_element(const int& face_index,FaceElement* face_element_pt)
  {    
    face_element_pt->set_nodal_dimension(nodal_dimension());   
    face_element_pt->bulk_element_pt() = this;

#ifdef OOMPH_HAS_MPI    
    face_element_pt->set_halo(Non_halo_proc_ID);
#endif    
    face_element_pt->face_index() = face_index;
    const unsigned nnode_face = nnode_on_face_by_index(face_index);
    
    face_element_pt->face_to_bulk_coordinate_fct_pt() = face_to_bulk_coordinate_fct_pt(face_index);    
    face_element_pt->bulk_coordinate_derivatives_fct_pt() = bulk_coordinate_derivatives_fct_pt(face_index);    
    face_element_pt->nbulk_value_resize(nnode_face);    
    face_element_pt->bulk_node_number_resize(nnode_face);
    
    for (unsigned i = 0; i < nnode_face; i++)
    {
      unsigned bulk_number = get_bulk_node_number(face_index, i);     
      face_element_pt->node_pt(i) = node_pt(bulk_number);
      face_element_pt->bulk_node_number(i) = bulk_number;      
      face_element_pt->nbulk_value(i) = required_nvalue(bulk_number);
    }    
    face_element_pt->normal_sign() = face_outer_unit_normal_sign(face_index);
  }

  

  namespace WedgeElementFaceToBulkCoordinates
  {    
    void face0(const Vector<double>& s, Vector<double>& s_bulk)
    {
        s_bulk[0] = s[0];
        s_bulk[1] = s[1];
        s_bulk[2] = 0.0;
    }
   
    void face1(const Vector<double>& s, Vector<double>& s_bulk)
    {        
        s_bulk[0] = s[0];
        s_bulk[1] = s[1];
        s_bulk[2] = 1.0;
    }
    
    void face2(const Vector<double>& s, Vector<double>& s_bulk)
    {
        throw_runtime_error("Implement");
    }
    
    void face3(const Vector<double>& s, Vector<double>& s_bulk)
    {
        throw_runtime_error("Implement");    
    }
    
    void face4(const Vector<double>& s, Vector<double>& s_bulk)
    {
        throw_runtime_error("Implement");
    }
    
  } 

  namespace WedgeElementBulkCoordinateDerivatives
  {    
    void faces0(const Vector<double>& s,DenseMatrix<double>& dsbulk_dsface,unsigned& interior_direction)
    {
        throw_runtime_error("Implement");
    }
    
    void faces1(const Vector<double>& s,DenseMatrix<double>& dsbulk_dsface,unsigned& interior_direction)
    {
        throw_runtime_error("Implement");
    }

    void faces2(const Vector<double>& s,DenseMatrix<double>& dsbulk_dsface,unsigned& interior_direction)
    {
        throw_runtime_error("Implement");
    }

    void faces3(const Vector<double>& s,DenseMatrix<double>& dsbulk_dsface,unsigned& interior_direction)
    {
        throw_runtime_error("Implement");
    }

    void faces4(const Vector<double>& s,DenseMatrix<double>& dsbulk_dsface,unsigned& interior_direction)
    {
        throw_runtime_error("Implement");
    }
  } 


  

  CoordinateMappingFctPt WedgeElementBase::face_to_bulk_coordinate_fct_pt(const int& face_index) const
    {
      if (face_index == 0)
      {
        return &WedgeElementFaceToBulkCoordinates::face0;
      }
      else if (face_index == 1)
      {
        return &WedgeElementFaceToBulkCoordinates::face1;
      }
      else if (face_index == 2)
      {
        return &WedgeElementFaceToBulkCoordinates::face2;
      }
      else if (face_index == 3)
      {
        return &WedgeElementFaceToBulkCoordinates::face3;
      }
      else if (face_index == 4)
      {
        return &WedgeElementFaceToBulkCoordinates::face4;
      }      
      {
        std::string err = "Face index should be in {0..4}.";
        throw OomphLibError(
          err, OOMPH_EXCEPTION_LOCATION, OOMPH_CURRENT_FUNCTION);
      }
    }

    /// Get a pointer to the derivative of the mapping from face to bulk
    /// coordinates.
    BulkCoordinateDerivativesFctPt WedgeElementBase::bulk_coordinate_derivatives_fct_pt(const int& face_index) const
    {
      if (face_index == 0)
      {
        return &WedgeElementBulkCoordinateDerivatives::faces0;
      }
      else if (face_index == 1)
      {
        return &WedgeElementBulkCoordinateDerivatives::faces1;
      }
      else if (face_index == 2)
      {
        return &WedgeElementBulkCoordinateDerivatives::faces2;
      }
      else if (face_index == 3)
      {
        return &WedgeElementBulkCoordinateDerivatives::faces3;
      }
      else if (face_index == 4)
      {
        return &WedgeElementBulkCoordinateDerivatives::faces4;
      }      
      else
      {
        std::string err = "Face index should be in {0..4}.";
        throw OomphLibError(
          err, OOMPH_EXCEPTION_LOCATION, OOMPH_CURRENT_FUNCTION);
      }
    }

    int WedgeElementBase::face_outer_unit_normal_sign(const int&) const
    {
        return 1; // This is a placeholder. The actual sign will depend on the face orientation and the convention used for the normal vector.
    }
}