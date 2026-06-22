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


const double WedgeGaussC2::Knot[18][3] =
{
  // ── Orbit A, triangle point 0: (s0,s1) = (a1, a1) ──────────────────────
  {0.445948490915965, 0.445948490915965, 0.112701665379258},  //  0
  {0.445948490915965, 0.445948490915965, 0.500000000000000},  //  1
  {0.445948490915965, 0.445948490915965, 0.887298334620742},  //  2
  // ── Orbit A, triangle point 1: (s0,s1) = (a1, 1-2a1) ───────────────────
  {0.445948490915965, 0.108103018168070, 0.112701665379258},  //  3
  {0.445948490915965, 0.108103018168070, 0.500000000000000},  //  4
  {0.445948490915965, 0.108103018168070, 0.887298334620742},  //  5
  // ── Orbit A, triangle point 2: (s0,s1) = (1-2a1, a1) ───────────────────
  {0.108103018168070, 0.445948490915965, 0.112701665379258},  //  6
  {0.108103018168070, 0.445948490915965, 0.500000000000000},  //  7
  {0.108103018168070, 0.445948490915965, 0.887298334620742},  //  8
  // ── Orbit B, triangle point 0: (s0,s1) = (a2, a2) ──────────────────────
  {0.091576213509771, 0.091576213509771, 0.112701665379258},  //  9
  {0.091576213509771, 0.091576213509771, 0.500000000000000},  // 10
  {0.091576213509771, 0.091576213509771, 0.887298334620742},  // 11
  // ── Orbit B, triangle point 1: (s0,s1) = (a2, 1-2a2) ───────────────────
  {0.091576213509771, 0.816847572980458, 0.112701665379258},  // 12
  {0.091576213509771, 0.816847572980458, 0.500000000000000},  // 13
  {0.091576213509771, 0.816847572980458, 0.887298334620742},  // 14
  // ── Orbit B, triangle point 2: (s0,s1) = (1-2a2, a2) ───────────────────
  {0.816847572980458, 0.091576213509771, 0.112701665379258},  // 15
  {0.816847572980458, 0.091576213509771, 0.500000000000000},  // 16
  {0.816847572980458, 0.091576213509771, 0.887298334620742},  // 17
};

// Weight[i] = triangle_weight * s2_weight.
//
//   Orbit A triangle weight per point : 0.223381589678011 / 2
//   Orbit B triangle weight per point : 0.109951743655322 / 2
//
//   s2 weights (3-point GL on [0,1]) : 5/18, 4/9, 5/18
//
//   So within each orbit the three weights repeat as (wt*5/18, wt*4/9, wt*5/18).
const double WedgeGaussC2::Weight[18] =
{
  // Orbit A (triangle weight = 0.111690794839005 per point)
  0.031025220788613,  //  0  wA * 5/18
  0.049640353261780,  //  1  wA * 4/9
  0.031025220788613,  //  2  wA * 5/18
  0.031025220788613,  //  3  wA * 5/18
  0.049640353261780,  //  4  wA * 4/9
  0.031025220788613,  //  5  wA * 5/18
  0.031025220788613,  //  6  wA * 5/18
  0.049640353261780,  //  7  wA * 4/9
  0.031025220788613,  //  8  wA * 5/18
  // Orbit B (triangle weight = 0.054975871827661 per point)
  0.015271075507684,  //  9  wB * 5/18
  0.024433720812294,  // 10  wB * 4/9
  0.015271075507684,  // 11  wB * 5/18
  0.015271075507684,  // 12  wB * 5/18
  0.024433720812294,  // 13  wB * 4/9
  0.015271075507684,  // 14  wB * 5/18
  0.015271075507684,  // 15  wB * 5/18
  0.024433720812294,  // 16  wB * 4/9
  0.015271075507684,  // 17  wB * 5/18
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
    if (face_index==0) 
    { 
      switch (i)
      {
        case 0: return 2;
        case 1: return 1;
        case 2: return 0;
        default: throw_runtime_error("Invalid node index for face");        
      }      
    }
    else if (face_index==1) 
    {
        switch (i)
        {
            case 0: return 3;
            case 1: return 4;
            case 2: return 5;
            default: throw_runtime_error("Invalid node index for face");
        }
    }
    else if (face_index==2) {  // 3 0 5 2
        switch (i)
        {
            case 0: return 0;
            case 1: return 3;
            case 2: return 2;
            case 3: return 5;
            default: throw_runtime_error("Invalid node index for face");
        }
    }
    else if (face_index==3) {  // 1 0 4 3
        switch (i)
        {
            case 0: return 0;
            case 1: return 1;
            case 2: return 3;
            case 3: return 4;
            default: throw_runtime_error("Invalid node index for face");
        }
    }
    else if (face_index==4) {  // 1 4 2 5
        switch (i)
        {
            case 0: return 4;
            case 1: return 1;
            case 2: return 5;
            case 3: return 2;
            default: throw_runtime_error("Invalid node index for face");
        }
    }
    
    throw_runtime_error("Invalid node or face index for wedge element "+std::to_string(face_index)+", "+std::to_string(i));
    return 0;
  }



///////////

WedgeGaussC2  WedgeElementC2::Default_integration_scheme;

 unsigned WedgeElementC2::get_bulk_node_number(const int& face_index,const unsigned int& i) const
    {
        // ---- Face 0 : s2 = 0, 6-node triangular facet ----
        // Reversed winding (outward normal = -s2 direction).
        // Corners: 2, 1, 0.
        // Edge midpoints in the same winding:
        //   mid(2,1) = node 5,  mid(1,0) = node 3,  mid(0,2) = node 4.
        if (face_index == 0)
        {
            switch (i)
            {
                case 0: return 2;
                case 1: return 1;
                case 2: return 0;
                case 3: return 5;   // midpoint of edge 1–2
                case 4: return 3;   // midpoint of edge 0–1
                case 5: return 4;   // midpoint of edge 0–2
                default: throw_runtime_error("Invalid node index for face 0");
            }
        }

        // ---- Face 1 : s2 = 1, 6-node triangular facet ----
        // Forward winding (outward normal = +s2 direction).
        // Matches C1 ordering 3,4,5 extended to layer-2 nodes.
        else if (face_index == 1)
        {
            switch (i)
            {              
                case 0: return 12;  // corner (0,0,1)
                case 1: return 13;  // corner (1,0,1)
                case 2: return 14;  // corner (0,1,1)                
                case 3: return 15;  // midpoint of edge 12–14 // 15,16,17, 
                case 4: return 17;  // midpoint of edge 13–14
                case 5: return 16;  // midpoint of edge 12–13
                default: throw_runtime_error("Invalid node index for face 1");
            }
        }

        // ---- Face 2 : s0 = 0, 9-node quadrilateral facet ----
        // Parametric coords on this face: (s1, s2).
        else if (face_index == 2)
        {            
            switch (i)
            {
                case 0: return 0;   // (s1=0,   s2=0  )  corner                
                case 1: return 6;   // (s1=0,   s2=1/2)  s2-mid of i=0,1                                
                case 2: return 12;  // (s1=0,   s2=1  )  corner                                
                case 3: return 4;   // (s1=1/2, s2=0  )  s1-mid of i=0,2
                case 4: return 10;  // (s1=1/2, s2=1/2)  centre
                case 5: return 16;  // (s1=1/2, s2=1  )  s1-mid of i=1,3
                case 6: return 2;   // (s1=1,   s2=0  )  corner                                
                case 7: return 8;   // (s1=1,   s2=1/2)  s2-mid of i=2,3                                
                case 8: return 14;  // (s1=1,   s2=1  )  corner                
                default: throw_runtime_error("Invalid node index for face 2");
            }
        }

        // ---- Face 3 : s1 = 0, 9-node quadrilateral facet ----
        // Parametric coords on this face: (s0, s2).
        else if (face_index == 3)
        {
            switch (i)
            {
                case 0: return 0;   // (s0=0,   s2=0  )  corner
                case 1: return 3;   // (s0=1/2, s2=0  )  s0-mid of i=0,2
                case 2: return 1;   // (s0=1,   s2=0  )  corner
                case 3: return 6;   // (s0=0,   s2=1/2)  s2-mid of i=0,1
                case 4: return 9;   // (s0=1/2, s2=1/2)  centre
                case 5: return 7;   // (s0=1,   s2=1/2)  s2-mid of i=2,3
                case 6: return 12;  // (s0=0,   s2=1  )  corner
                case 7: return 15;  // (s0=1/2, s2=1  )  s0-mid of i=1,3
                case 8: return 13;  // (s0=1,   s2=1  )  corner                
                default: throw_runtime_error("Invalid node index for face 3");
            }
        }

        // ---- Face 4 : s0+s1 = 1, 9-node quadrilateral facet ----
        // Parametric coord t runs along the hypotenuse:
        //   t=0 at (s0=1, s1=0),  t=1 at (s0=0, s1=1).
        else if (face_index == 4)
        {
            switch (i)
            {
                case 0: return 13;  // (t=0,   s2=1  )  corner                
                case 1: return 7;   // (t=0,   s2=1/2)  s2-mid of i=0,1
                case 2: return 1;   // (t=0,   s2=0  )  corner
                case 3: return 17;  // (t=1/2, s2=1  )  t-mid  of i=1,3
                case 4: return 11;  // (t=1/2, s2=1/2)  centre                
                case 5: return 5;   // (t=1/2, s2=0  )  t-mid  of i=0,2
                case 6: return 14;  // (t=1,   s2=1  )  corner                
                case 7: return 8;   // (t=1,   s2=1/2)  s2-mid of i=2,3                
                case 8: return 2;   // (t=1,   s2=0  )  corner                                
                default: throw_runtime_error("Invalid node index for face 4");
            }
        }

        throw_runtime_error("Invalid face index for wedge element: "
                            + std::to_string(face_index));
        return 0;
    }

///////////

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

//////////////////////
    // Consistency check: make sure that the order of the get_bulk_node_number matches the order of the nodes in the face element (requires the facet->bulk mapping to be correct, see below)
    /*
    std::vector<oomph::Vector<double>> xface_buffer;
    std::vector<oomph::Vector<double>> xbulk_buffer;
    for (unsigned int i=0; i<nnode_face; i++)
    {
      oomph::Vector<double> sface(face_element_pt->dim(),0.0);
      oomph::Vector<double> xface(face_element_pt->nodal_dimension(),0.0);
      face_element_pt->local_coordinate_of_node(i,sface);
      face_element_pt->face_to_bulk_coordinate_fct_pt()(sface,xface);
      xface_buffer.push_back(xface);
      unsigned bulk_number = face_element_pt->bulk_node_number(i);
      oomph::Vector<double> s_bulk(this->dim(),0.0);
      this->local_coordinate_of_node(bulk_number,s_bulk);      
      xbulk_buffer.push_back(s_bulk);
    }
    //Find the permutation so that xface_buffer[i] matches xbulk_buffer[perm[i]]
    std::vector<unsigned int> perm(nnode_face);
    for (unsigned int i=0; i<nnode_face; i++)    {
      bool found_match=false;
      for (unsigned int j=0; j<nnode_face; j++)      {
        double dist=0.0;
        for (unsigned int k=0; k<xface_buffer[i].size(); k++)        {
          dist+=(xface_buffer[i][k]-xbulk_buffer[j][k])*(xface_buffer[i][k]-xbulk_buffer[j][k]);
        }
        if (dist<1e-8)        {
          perm[i]=j;
          found_match=true;
          break;
        }      }
      if (!found_match)      {
        std::ostringstream error_message;
        error_message << "Inconsistency in face to bulk node mapping for face " << face_index << ": no match found for face node " << i;
        throw OomphLibError(error_message.str(), OOMPH_CURRENT_FUNCTION, OOMPH_EXCEPTION_LOCATION);
      }
    }
    // Check that the permutation is the identity (i.e. that the order of the face nodes matches the order of the bulk nodes as given by get_bulk_node_number)
    for (unsigned int i=0; i<nnode_face; i++)    {
      if (perm[i]!=i)      {
        std::ostringstream error_message;
        std::cout << "Permutation for face " << face_index << ": ";
        for (unsigned int j=0; j<nnode_face; j++)        {
          std::cout << perm[j] << " ";        }
        std::cout << std::endl;
        error_message << "Inconsistency in face to bulk node mapping for face " << face_index << ": face node " << i << " matches bulk node " << perm[i] << " but should match bulk node " << i;
        throw OomphLibError(error_message.str(), OOMPH_CURRENT_FUNCTION, OOMPH_EXCEPTION_LOCATION);
      }
    }

*/
//////////////////////
    // Consistency check: make sure that the facet -> bulk is correct
    /*
    std::vector<std::vector<double>> sface_test={ {1.0,0.0},{0.0,1.0},{0.2,0.2},{0.2,0.3}, {0.1,0.4}, {0.7,0.1} }; // For triangular faces
    std::ostringstream ai_request;
    ai_request << "I need a affine mapping from a 2d vector sface to a 3d vector s_bulk, which should realize the following mapping:" << std::endl;
    bool has_fails=false;
    for (const auto& sface_ : sface_test)
    {
      Vector<double> sface(sface_.size());
      for (unsigned i = 0; i < sface_.size(); i++)      {
        sface[i] = sface_[i];
      }
      Vector<double> s_bulk=face_element_pt->local_coordinate_in_bulk(sface);
      Vector<double> xface(3);
      Vector<double> x_bulk(3);
      face_element_pt->FiniteElement::interpolated_x(sface,xface);
      this->interpolated_x(s_bulk,x_bulk);
      std::cout << "Testing face to bulk coordinate mapping for face " << face_index << " and sface = [" << sface[0] << ", " << sface[1] << "]" << std::endl;
      std::cout << "Mapped to bulk coordinates s_bulk = [" << s_bulk[0] << ", " << s_bulk[1] << ", " << s_bulk[2] << "]" << std::endl;
      std::cout << "Interpolated x at face coordinates: [" << xface[0] << ", " << xface[1] << ", " << xface[2] << "]" << std::endl;
      std::cout << "Interpolated x at bulk coordinates: [" << x_bulk[0] << ", " << x_bulk[1] << ", " << x_bulk[2] << "]" << std::endl;
      
      // This only works if the bulk element has the same eulerian coordinates at the local bulk coordinates sbulk
      ai_request << "[" << sface[0] << ", " << sface[1] << "] ->  [" << xface[0] << ", " << xface[1] << ", " << xface[2] << "]" << std::endl;
      for (unsigned i = 0; i < 3; i++)
      {
        if (std::abs(xface[i]-x_bulk[i])>1e-8)
        {
          std::ostringstream error_message;
          error_message << "Inconsistency in face to bulk coordinate mapping for face " << face_index
                        << ": xface[" << i << "] = " << xface[i] << " but x_bulk[" << i << "] = " << x_bulk[i];
          //throw OomphLibError(error_message.str(), OOMPH_CURRENT_FUNCTION, OOMPH_EXCEPTION_LOCATION);
          std::cout << "WARNING: " << error_message.str() << std::endl;
          has_fails=true;
        }
      }
    }
    if (has_fails)
    {
      std::cout << "FACE INDEX " << face_index << ": AI Request for affine mapping from face to bulk coordinates:" << std::endl;
      std::cout << ai_request.str() << std::endl;
    }
    */
  }

  

  namespace WedgeElementFaceToBulkCoordinates
  {    
    void face0(const Vector<double>& s, Vector<double>& s_bulk)
    {  
        s_bulk[0] = s[1];
        s_bulk[1] = s[0];
        s_bulk[2] = 0.0;
    }
   
    void face1(const Vector<double>& s, Vector<double>& s_bulk)
    {        
        s_bulk[0] = s[1];
        s_bulk[1] = 1.0-s[0]-s[1];
        s_bulk[2] = 1.0;
    }
    
    void face2(const Vector<double>& s, Vector<double>& s_bulk)
    {
        s_bulk[0] = 0.0;
        s_bulk[1] = (s[1]+1.0)/2.0;
        s_bulk[2] = (s[0]+1.0)/2.0; // Map from Quad coordinates here
    }
    
    void face3(const Vector<double>& s, Vector<double>& s_bulk)
    {      
        s_bulk[0] = (s[0]+1.0)/2.0; // Map from Quad coordinates here
        s_bulk[1] = 0;
        s_bulk[2] = (s[1]+1.0)/2.0; // Map from Quad coordinates here
    }
    
    void face4(const Vector<double>& s, Vector<double>& s_bulk)
    {
        s_bulk[0] = (1-s[1])/2.0; // Map from Quad coordinates here
        s_bulk[1] = (1.0+s[1])/2.0; // Map from Quad coordinates here
        s_bulk[2] = (1.0-s[0])/2.0; // Map from Quad coordinates here
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

    int WedgeElementBase::face_outer_unit_normal_sign(const int& face_index) const
    {                
        return 1;
    }
}