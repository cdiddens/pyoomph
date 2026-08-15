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


// Declarations of the concrete element types: every bulk element from the 0-d ODE element up to the
// 3-d wedges and pyramids, and the interface (face) element instantiated for each of them.
//
// These live apart from elements.hpp because most consumers of the element hierarchy only ever name
// BulkElementBase and InterfaceElementBase -- only the mesh/meshtemplate code and the element
// implementations themselves need the concrete types. Include this header only if you name one.

#pragma once
#include "elements.hpp"

namespace pyoomph
{
  // Concrete zero-dimensional "bulk" element combining BulkElementBase (JIT residual assembly,
  // dof/equation bookkeeping) with ODEElementBase (no spatial nodes). Represents a single ODE
  // "point" degree of freedom set governed by generated residual code, e.g. used for globally
  // coupled ODEs that are not tied to any mesh geometry.
  class BulkElementODE0d :  public virtual BulkElementBase, public virtual ODEElementBase
  {
  protected:
    //	virtual void fill_element_info(); //TODO simplify this
    oomph::TimeStepper *timestepper;
    static oomph::PointIntegral Default_integration_scheme;
    static const std::vector<int> Possible_Face_Indices;
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;    
    bool fill_hang_info_with_equations_for_pos(JITShapeInfo_t *) override {return false;} // An ODE never has positions
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
    static const std::vector<unsigned> Non_Vertex_Node_Indices;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<unsigned> & non_vertex_node_indices() const override {return Non_Vertex_Node_Indices;}
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *, int ) override {throw_runtime_error("ODE Elements do not have faces"); return NULL;}
    const std::vector<int> & get_possible_face_indices() const override { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & ) const override { return std::vector<pyoomph::Node*>(); }
    int nedges() const override { return 0; }
    double fill_shape_info_at_s(const oomph::Vector<double> &s, const unsigned int &index, const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, double &JLagr, unsigned int flag, oomph::DenseMatrix<double> *dxds = NULL, unsigned history_index=0) const override;
    unsigned get_meshio_type_index() const override { return 0; }
    void dshape_local_at_s_C1(const oomph::Vector<double> &, oomph::Shape &, oomph::DShape &) const override { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &, oomph::Shape &, oomph::DShape &) const override { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_C2TB(const oomph::Vector<double> &, oomph::Shape &, oomph::DShape &) const override { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &, oomph::Shape &, oomph::DShape &) const override { throw_runtime_error("Makes no sense"); }
    
    unsigned nrecovery_order() override { return 0; }
    unsigned nvertex_node() const override { return 0; }
    oomph::Node *vertex_node_pt(const unsigned &) const override { return NULL; }
    void further_setup_hanging_nodes() override {};
    void to_numpy(double *dest);
    void shape(const oomph::Vector<double> &, oomph::Shape &) const override {}
    void build(oomph::Mesh *&, oomph::Vector<oomph::Node *> &, bool &, std::ofstream &) override {}
    void check_integrity(double &max_error) override { max_error = 0; }
    BulkElementBase *create_son_instance() const override { return NULL; }
    void shape_at_s_C1(const oomph::Vector<double> &, oomph::Shape &) const override {}
    void shape_at_s_C2(const oomph::Vector<double> &, oomph::Shape &) const override {}
    void shape_at_s_C2TB(const oomph::Vector<double> &, oomph::Shape &) const override {}
    void shape_at_s_DL(const oomph::Vector<double> &, oomph::Shape &) const override {}
    int get_num_numpy_elemental_indices(bool , unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &) const override
    {
      nsubdiv = 0;
      return 0;
    }
    void fill_element_nodal_indices_for_numpy(int *, unsigned, bool, std::vector<std::vector<std::set<oomph::Node *>>> &) const override {}

    BulkElementODE0d(DynamicBulkElementInstance *code_inst, oomph::TimeStepper *tstepper);
    ~BulkElementODE0d() override;

    // Factory that sets the __CurrentCodeInstance "side channel" (see BulkElementBase) around
    // construction so the new element picks up the right JIT code instance, then clears it again.
    static BulkElementODE0d *construct_new(DynamicBulkElementInstance *code_inst, oomph::TimeStepper *tstepper)
    {
      BulkElementBase::__CurrentCodeInstance = code_inst;
      BulkElementODE0d *res = new BulkElementODE0d(code_inst, tstepper);
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    double get_quality_factor() override { return 1.0; }

    void set_integration_order(unsigned int ) override {}
  };

  // Oomph-libs RefineableSolidQElement<1> needs to be adjusted, since it is marked as broken in the constructor
  // (oomph-lib does not ship a working 1d refineable solid Q-element directly; this class recombines
  // the pieces - refineable 1d geometry, solid-mechanics position dofs, Q-element macro-element
  // support - manually to get a working 1d refineable solid line element).
  class RefineableSolidLineElement : public virtual oomph::RefineableQElement<1>, public virtual oomph::RefineableSolidElement, public virtual oomph::QSolidElementBase
  {
  public:
    RefineableSolidLineElement() : oomph::RefineableQElement<1>(), oomph::RefineableSolidElement()
    {
    }

    /// Broken copy constructor
    RefineableSolidLineElement(const RefineableSolidLineElement &)
    {
      oomph::BrokenCopy::broken_copy("RefineableSolidLineElement");
    }

    ~RefineableSolidLineElement() override {}

    void set_macro_elem_pt(oomph::MacroElement *macro_elem_pt) override
    {
      oomph::QSolidElementBase::set_macro_elem_pt(macro_elem_pt);
    }

    void set_macro_elem_pt(oomph::MacroElement *macro_elem_pt, oomph::MacroElement *undeformed_macro_elem_pt) override
    {
      oomph::QSolidElementBase::set_macro_elem_pt(macro_elem_pt, undeformed_macro_elem_pt);
    }

    void get_jacobian(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian) override
    {
      oomph::RefineableSolidElement::get_jacobian(residuals, jacobian);
    }

    void build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt,
               bool &was_already_built,
               std::ofstream &new_nodes_file) override;
  };

  // --- The following classes (BulkElement*, up to PointElement0d) are the concrete geometric bulk
  // element types: one per (element shape x interpolation order) combination that pyoomph supports
  // (1d line / 2d quad / 2d triangle / 3d brick / 3d tetrahedron / 3d wedge / 3d pyramid, each in
  // C1 = linear/bilinear/trilinear and, where applicable, C2 = quadratic/biquadratic/bubble-enriched
  // "TB" variants). They all follow the same pattern, illustrated here for BulkElementLine1dC1:
  //  - Possible_Face_Indices / Nodal_Space_Index_To_Element_Index_Map /
  //    Element_Index_To_Nodal_Space_Index_Map / Dummy_Value_Interpolation_Map: static lookup
  //    tables (defined in elements.cpp) describing the element's face numbering and how the
  //    different interpolation spaces (C1/C2/C1TB/C2TB/DL/D0) map onto its local dof/node indices;
  //    these feed the generic bookkeeping in BulkElementBase.
  //  - shape_at_s_XX / dshape_local_at_s_XX: delegate to the underlying oomph-lib geometric
  //    element's shape() / dshape_local() for the spaces the element actually supports, and throw
  //    for spaces that make no sense for this element (e.g. a C1 line element has no C2 space).
  //  - get_meshio_type_index(): numeric cell-type code (see the Meshio type table earlier in this
  //    file) used when exporting the mesh to meshio-compatible formats.
  //  - get_num_numpy_elemental_indices() / fill_element_nodal_indices_for_numpy(): describe how the
  //    element tessellates into simple (triangle/line/tet) sub-cells for numpy/vtk-style export.
  //  - create_son_instance(): factory for a fresh element of the same type, used during refinement.
  //  - construct_face_element(): builds the matching Interface*Element* face element (see below).
  // Only functions with non-obvious behaviour are commented individually in the later classes of
  // this family to avoid repeating this same explanation for every element type.
  //
  // 1d line element, linear (C1) Lagrange interpolation, refineable + moving-mesh (solid) capable.
  class BulkElementLine1dC1 : public virtual BulkElementBase,
                              public virtual oomph::QElement<1, 2>,
                              public virtual RefineableSolidLineElement
  {
  protected:
    static const std::vector<int> Possible_Face_Indices;
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
    static const std::vector<unsigned> Non_Vertex_Node_Indices;
  public:
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<unsigned> & non_vertex_node_indices() const override {return Non_Vertex_Node_Indices;}
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    const std::vector<int> & get_possible_face_indices() const override { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
    int nedges() const override { return 2; }
    BulkElementLine1dC1();    
    unsigned get_meshio_type_index() const override { return 1; }
    void check_integrity(double &max_error) override { max_error = 0; } // TODO throw_runtime_error("IMPLEMENT");

    

    void output(std::ostream &outfile, const unsigned &n_plot) override { BulkElementBase::output(outfile, n_plot); }
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const override { this->shape(s, psi); }
    void shape_at_s_C2(const oomph::Vector<double> &, oomph::Shape &) const override { throw_runtime_error("Makes no sense"); }
    void shape_at_s_C2TB(const oomph::Vector<double> &, oomph::Shape &) const override { throw_runtime_error("Makes no sense"); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const override;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &, oomph::Shape &, oomph::DShape &) const override { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_C2TB(const oomph::Vector<double> &, oomph::Shape &, oomph::DShape &) const override { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    // Second local derivatives: C1 is this element's own (QElement<1,2>) space.
    void d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override { this->d2shape_local_pyoomph(s, psi, dpsi, d2psi); }
    bool supports_second_spatial_derivatives(std::string &) const override { return true; }

    unsigned nrecovery_order() override { return 1; }
    unsigned nvertex_node() const override { return oomph::QElement<1, 2>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const override { return QElement<1, 2>::vertex_node_pt(j); }
    // void further_setup_hanging_nodes() {BulkElementBase::further_setup_hanging_nodes();};
    void further_setup_hanging_nodes() override {};
    std::vector<double> get_outline(bool lagrangian) override;
    BulkElementBase *create_son_instance() const override
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementLine1dC1();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }

    void pre_build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt) override
    {
      BulkElementBase::pre_build(mesh_pt, new_node_pt);
      oomph::RefineableQElement<1>::pre_build(mesh_pt, new_node_pt);
    }

    int get_num_numpy_elemental_indices(bool , unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &) const override
    {
      nsubdiv = 1;
      return 2;
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;
    void set_integration_order(unsigned int order) override { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 1, order)); }
    void get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather) override;
  };

  // 1d line element, quadratic (C2) Lagrange interpolation (plus a C1 dummy sub-space), refineable + solid.
  class BulkElementLine1dC2 : public virtual BulkElementBase, public virtual oomph::QElement<1, 3>, public virtual RefineableSolidLineElement
  {
  protected:    
    static const std::vector<int> Possible_Face_Indices;
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<std::vector<unsigned>>> Dummy_Value_Interpolation_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
    static const std::vector<unsigned> Non_Vertex_Node_Indices;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<unsigned> & non_vertex_node_indices() const override {return Non_Vertex_Node_Indices;}
    const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const override {return Dummy_Value_Interpolation_Map;}
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    const std::vector<int> & get_possible_face_indices() const override { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
    int nedges() const override { return 2; }
    unsigned get_meshio_type_index() const override { return 2; }
    BulkElementLine1dC2();    
    void check_integrity(double &max_error) override { max_error = 0; } // TODO

    

    
    
    void output(std::ostream &outfile, const unsigned &n_plot) override { BulkElementBase::output(outfile, n_plot); }

    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const override;
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const override { this->shape(s, psi); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const override;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    // Second local derivatives: C2 is this element's own (QElement<1,3>) space.
    void d2shape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override { this->d2shape_local_pyoomph(s, psi, dpsi, d2psi); }
    void d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override;
    bool supports_second_spatial_derivatives(std::string &) const override { return true; }
    int get_num_numpy_elemental_indices(bool , unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &) const override
    {
      nsubdiv = 1;
      return 3;
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;
    std::vector<double> get_outline(bool lagrangian) override;
    unsigned nrecovery_order() override { return 2; }
    unsigned nvertex_node() const override { return oomph::QElement<1, 3>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const override { return QElement<1, 3>::vertex_node_pt(j); }
    // void further_setup_hanging_nodes() {BulkElementBase::further_setup_hanging_nodes();};
    void further_setup_hanging_nodes() override {};
    BulkElementBase *create_son_instance() const override
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementLine1dC2();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    void set_integration_order(unsigned int order) override { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 1, order)); }
    void get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather) override;
  };

  // TRIANGULAR LINE ELEMENTS

  // 1d simplex ("T") line element, linear (C1) interpolation; the T-family uses barycentric-style
  // local coordinates and simplex refinement rules instead of the Q-family's tensor-product ones.
  class BulkTElementLine1dC1 : public virtual BulkElementBase, public virtual oomph::TElement<1, 2>, public virtual oomph::RefineableTElement<1>
  {
  protected:
    static const std::vector<int> Possible_Face_Indices;
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
    static const std::vector<unsigned> Non_Vertex_Node_Indices;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<unsigned> & non_vertex_node_indices() const override {return Non_Vertex_Node_Indices;}
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    const std::vector<int> & get_possible_face_indices() const override { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
    int nedges() const override { return 2; }
    // A face of a 1d simplex is a single point. oomph::TElement<1,*> implements neither
    // nnode_on_face() nor get_bulk_node_number(), so both hooks have to be supplied here.
    unsigned nnode_on_face_by_index(const int & ) const override { return 1; }
    unsigned node_index_on_face(const int & face_index, const unsigned & ) const override { return (face_index == -1 ? 0 : 1); }
    BulkTElementLine1dC1();
    unsigned get_meshio_type_index() const override { return 1; }
    void check_integrity(double &max_error) override { max_error = 0; } // TODO throw_runtime_error("IMPLEMENT");

    
    void output(std::ostream &outfile, const unsigned &n_plot) override { BulkElementBase::output(outfile, n_plot); }
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const override { this->shape(s, psi); }
    void shape_at_s_C2(const oomph::Vector<double> &, oomph::Shape &) const override { throw_runtime_error("Makes no sense"); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const override;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &, oomph::Shape &, oomph::DShape &) const override { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    // Second local derivatives: C1 is this element's own (TElement<1,2>) space.
    void d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override { this->d2shape_local_pyoomph(s, psi, dpsi, d2psi); }
    bool supports_second_spatial_derivatives(std::string &) const override { return true; }

    unsigned nrecovery_order() override { return 1; }
    unsigned nvertex_node() const override { return oomph::TElement<1, 2>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const override { return TElement<1, 2>::vertex_node_pt(j); }
    // void further_setup_hanging_nodes() {BulkElementBase::further_setup_hanging_nodes();};
    void further_setup_hanging_nodes() override {};
    std::vector<double> get_outline(bool lagrangian) override;
    BulkElementBase *create_son_instance() const override
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkTElementLine1dC1();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }

    void pre_build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt) override
    {
      BulkElementBase::pre_build(mesh_pt, new_node_pt);
      oomph::RefineableTElement<1>::pre_build(mesh_pt, new_node_pt);
    }

    int get_num_numpy_elemental_indices(bool , unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &) const override
    {
      nsubdiv = 1;
      return 2;
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;
    void set_integration_order(unsigned int order) override { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 1, order)); }
  };

  // 1d simplex ("T") line element, quadratic (C2) interpolation.
  class BulkTElementLine1dC2 : public virtual BulkElementBase, public virtual oomph::TElement<1, 3>, public virtual oomph::RefineableTElement<1>
  {
  protected:    
    static const std::vector<int> Possible_Face_Indices;
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<std::vector<unsigned>>> Dummy_Value_Interpolation_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
    static const std::vector<unsigned> Non_Vertex_Node_Indices;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<unsigned> & non_vertex_node_indices() const override {return Non_Vertex_Node_Indices;}
    const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const override {return Dummy_Value_Interpolation_Map;}
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    const std::vector<int> & get_possible_face_indices() const override { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
    int nedges() const override { return 2; }
    // As for the C1 line, but the far vertex is node 2 (node 1 is the mid-side node).
    unsigned nnode_on_face_by_index(const int & ) const override { return 1; }
    unsigned node_index_on_face(const int & face_index, const unsigned & ) const override { return (face_index == -1 ? 0 : 2); }
    unsigned get_meshio_type_index() const override { return 2; }
    BulkTElementLine1dC2();
    void check_integrity(double &max_error) override { max_error = 0; } // TODO

    
    
    
    void output(std::ostream &outfile, const unsigned &n_plot) override { BulkElementBase::output(outfile, n_plot); }

    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const override;
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const override { this->shape(s, psi); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const override;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    // Second local derivatives: C2 is this element's own (TElement<1,3>) space.
    void d2shape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override { this->d2shape_local_pyoomph(s, psi, dpsi, d2psi); }
    void d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override;
    bool supports_second_spatial_derivatives(std::string &) const override { return true; }
    int get_num_numpy_elemental_indices(bool , unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &) const override
    {
      nsubdiv = 1;
      return 3;
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;
    std::vector<double> get_outline(bool lagrangian) override;
    unsigned nrecovery_order() override { return 2; }
    unsigned nvertex_node() const override { return oomph::TElement<1, 3>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const override { return TElement<1, 3>::vertex_node_pt(j); }
    // void further_setup_hanging_nodes() {BulkElementBase::further_setup_hanging_nodes();};
    void further_setup_hanging_nodes() override {};
    BulkElementBase *create_son_instance() const override
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkTElementLine1dC2();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    void set_integration_order(unsigned int order) override { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 1, order)); }
  };

  // 2d quadrilateral element, bilinear (C1) interpolation, refineable + solid.
  class BulkElementQuad2dC1 : public virtual BulkElementBase, public virtual oomph::QElement<2, 2>, public virtual oomph::RefineableSolidQElement<2>
  {
  protected:
    static const std::vector<int> Possible_Face_Indices;
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
    static const std::vector<unsigned> Non_Vertex_Node_Indices;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<unsigned> & non_vertex_node_indices() const override {return Non_Vertex_Node_Indices;}
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    const std::vector<int> & get_possible_face_indices() const override { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
    int nedges() const override { return 4; }
    BulkElementQuad2dC1();    
    unsigned get_meshio_type_index() const override { return 6; }

    void check_integrity(double &max_error) override { max_error = 0; } // TODO

    

    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const override { this->shape(s, psi); }
    void shape_at_s_C2(const oomph::Vector<double> &, oomph::Shape &) const override { throw_runtime_error("Makes no sense"); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const override;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &, oomph::Shape &, oomph::DShape &) const override { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    // Second local derivatives: C1 is this element's own (QElement<2,2>) space.
    void d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override { this->d2shape_local_pyoomph(s, psi, dpsi, d2psi); }
    bool supports_second_spatial_derivatives(std::string &) const override { return true; }

    void add_node_from_finer_neighbor_for_tesselated_numpy(const oomph::Vector<double> &s_coarse, oomph::Node *n, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) override;
    void inform_coarser_neighbors_for_tesselated_numpy(std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) override;
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;
    oomph::Node *boundary_node_pt(const int &face_index, const unsigned int index) override;
    std::vector<double> get_outline(bool lagrangian) override;
    unsigned nrecovery_order() override { return 1; }
    void output(std::ostream &outfile, const unsigned &n_plot) override { BulkElementBase::output(outfile, n_plot); }
    // See BulkElementBase::reapply_macro_element_positions: oomph's solid build throws the
    // macro-element positions away, so re-apply them once it has finished.
    void build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt, bool &was_already_built, std::ofstream &new_nodes_file) override;
    unsigned nvertex_node() const override { return oomph::QElement<2, 2>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const override { return QElement<2, 2>::vertex_node_pt(j); }
    void further_setup_hanging_nodes() override { BulkElementBase::further_setup_hanging_nodes(); } // There can't be any problem here, since it is all isoparametric
    // Override to handle a cross-shape (triangular) coarse neighbour at a mixed quad+tri interface, which
    // oomph's QuadTree gteq_edge_neighbour compass math cannot (it would hang on wrong masters -> a cyclic
    // hang). Quad<->quad edges fall through to the base implementation.
    void quad_hang_helper(const int &value_id, const int &my_edge, std::ofstream &output_hangfile) override;
    // Override to also share a coincident interface node with an adjacent TRI (mixed mesh); quad<->quad
    // sharing falls through to the oomph base. Prevents duplicate interface nodes when both sides refine.
    oomph::Node *node_created_by_neighbour(const oomph::Vector<double> &s_fraction, bool &is_periodic) override
    {
      oomph::Node *n = oomph::RefineableQElement<2>::node_created_by_neighbour(s_fraction, is_periodic);
      if (n) return n;
      return this->mixed_quad_shared_node(s_fraction);
    }
    BulkElementBase *create_son_instance() const override
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementQuad2dC1();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    void get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather) override;
    void set_integration_order(unsigned int order) override { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 2, order)); }
  };

  // 2d quadrilateral element, biquadratic (C2) interpolation (plus a C1 dummy sub-space); also
  // implements the interpolating_node_pt/interpolating_basis family needed for non-isoparametric
  // hanging-node interpolation on refined meshes.
  class BulkElementQuad2dC2 : public virtual BulkElementBase, public virtual oomph::QElement<2, 3>, public virtual oomph::RefineableSolidQElement<2>
  {
  protected:  
    static const std::vector<int> Possible_Face_Indices;
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<std::vector<unsigned>>> Dummy_Value_Interpolation_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
    static const std::vector<unsigned> Non_Vertex_Node_Indices;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<unsigned> & non_vertex_node_indices() const override {return Non_Vertex_Node_Indices;}
    const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const override {return Dummy_Value_Interpolation_Map;}  
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    const std::vector<int> & get_possible_face_indices() const override { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
    void get_supporting_C1_nodes_of_C2_node(const unsigned &n, std::vector<oomph::Node *> &support) override;
    int nedges() const override { return 4; }
    BulkElementQuad2dC2();
    unsigned get_meshio_type_index() const override { return 8; }
    

    void check_integrity(double &max_error) override { max_error = 0; } // TODO
    std::vector<double> get_outline(bool lagrangian) override;
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const override;
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const override { this->shape(s, psi); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const override;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    // Second local derivatives: C2 is this element's own (QElement<2,3>) space.
    void d2shape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override { this->d2shape_local_pyoomph(s, psi, dpsi, d2psi); }
    void d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override;
    bool supports_second_spatial_derivatives(std::string &) const override { return true; }

    oomph::Node *boundary_node_pt(const int &face_index, const unsigned int index) override;
    

    
    

    void add_node_from_finer_neighbor_for_tesselated_numpy(const oomph::Vector<double> &s_coarse, oomph::Node *n, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) override;
    void inform_coarser_neighbors_for_tesselated_numpy(std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) override;
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;

    unsigned nrecovery_order() override { return 2; }
    void output(std::ostream &outfile, const unsigned &n_plot) override { BulkElementBase::output(outfile, n_plot); }
    // See BulkElementBase::reapply_macro_element_positions: oomph's solid build throws the
    // macro-element positions away, so re-apply them once it has finished.
    void build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt, bool &was_already_built, std::ofstream &new_nodes_file) override;
    unsigned nvertex_node() const override { return oomph::QElement<2, 3>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const override { return QElement<2, 3>::vertex_node_pt(j); }

    void further_setup_hanging_nodes() override;
    // See BulkElementQuad2dC1: handle a cross-shape (tri) coarse neighbour at a mixed interface.
    void quad_hang_helper(const int &value_id, const int &my_edge, std::ofstream &output_hangfile) override;
    // See BulkElementQuad2dC1: share a coincident interface node with an adjacent tri (mixed mesh).
    oomph::Node *node_created_by_neighbour(const oomph::Vector<double> &s_fraction, bool &is_periodic) override
    {
      oomph::Node *n = oomph::RefineableQElement<2>::node_created_by_neighbour(s_fraction, is_periodic);
      if (n) return n;
      return this->mixed_quad_shared_node(s_fraction);
    }
    oomph::Node *interpolating_node_pt(const unsigned &n, const int &value_id) override;
    double local_one_d_fraction_of_interpolating_node(const unsigned &n1d, const unsigned &i, const int &value_id) override;
    oomph::Node *get_interpolating_node_at_local_coordinate(const oomph::Vector<double> &s, const int &value_id) override;
    unsigned ninterpolating_node_1d(const int &value_id) override;
    unsigned ninterpolating_node(const int &value_id) override;
    void interpolating_basis(const oomph::Vector<double> &s, oomph::Shape &psi, const int &value_id) const override;
    BulkElementBase *create_son_instance() const override
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementQuad2dC2();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    void get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather) override;
    void set_integration_order(unsigned int order) override { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 2, order)); }
  };

  // 2d triangular element, linear (C1) interpolation.
  class BulkElementTri2dC1 : public virtual BulkElementBase, public virtual oomph::TElement<2, 2>, public virtual oomph::RefineableTElement<2>
  {
  protected:
    static const std::vector<int> Possible_Face_Indices;
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
    static const std::vector<unsigned> Non_Vertex_Node_Indices;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<unsigned> & non_vertex_node_indices() const override {return Non_Vertex_Node_Indices;}
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    const std::vector<int> & get_possible_face_indices() const override { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
    oomph::Node *boundary_node_pt(const int &face_index, const unsigned int index) override;
    int nedges() const override { return 3; }
    unsigned nnode_on_face() const override { return 2; }
    // Maps son node l to its local coordinate in the father, based on the son_type of the 1->4
    // triangle split. Mirrors the s_in_parent map in RefineableTElement<2>::build; handles the
    // 3-node (C1), 4-node (C1TB bubble), 6-node (C2) and 7-node (C2TB) tri layouts via nnode().
    void get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather) override;
    BulkElementTri2dC1(bool has_bubble = false);
    unsigned get_meshio_type_index() const override { return 3; }
    void check_integrity(double &max_error) override { max_error = 0; } // TODO
    
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const override { this->shape(s, psi); }
    void shape_at_s_C2(const oomph::Vector<double> &, oomph::Shape &) const override { throw_runtime_error("Makes no sense"); }
    void shape_at_s_DL(const oomph::Vector<double> &, oomph::Shape &) const override;
    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &, oomph::Shape &, oomph::DShape &) const override { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    // Second local derivatives: C1 is this element's own (TElement<2,2>) space.
    void d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override { this->d2shape_local_pyoomph(s, psi, dpsi, d2psi); }
    bool supports_second_spatial_derivatives(std::string &) const override { return true; }
    void add_node_from_finer_neighbor_for_tesselated_numpy(const oomph::Vector<double> &s_coarse, oomph::Node *n, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) override;
    void inform_coarser_neighbors_for_tesselated_numpy(std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) override;
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;
    std::vector<double> get_outline(bool lagrangian) override;
    unsigned nrecovery_order() override { return 1; }
    void output(std::ostream &outfile, const unsigned &n_plot) override { BulkElementBase::output(outfile, n_plot); }
    // TElementBase leaves FiniteElement's macro-element hook broken, so route it to the generic
    // vertex-coordinate implementation. (The Q family must NOT do this: QElementBase already provides
    // a final overrider via its s_macro_ll/ur box, and a second one reachable from the same class
    // would be ambiguous.)
    void get_x_from_macro_element(const oomph::Vector<double> &s, oomph::Vector<double> &x) const override
    { this->get_x_from_generic_macro_element(0, s, x); }
    void get_x_from_macro_element(const unsigned &t, const oomph::Vector<double> &s, oomph::Vector<double> &x) override
    { this->get_x_from_generic_macro_element(t, s, x); }
    unsigned nvertex_node() const override { return oomph::TElement<2, 2>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const override { return TElement<2, 2>::vertex_node_pt(j); }
    void further_setup_hanging_nodes() override { BulkElementBase::further_setup_hanging_nodes(); } // There can't be any problem here, since it is all isoparametric
    BulkElementBase *create_son_instance() const override
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementTri2dC1();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    void set_integration_order(unsigned int order) override { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 2, order)); }
    oomph::Vector<double> get_midpoint_s() override { return oomph::Vector<double>(this->dim(), 1.0 / 3.0); }
  };

  // 2d triangular element, linear (C1) interpolation enriched with a cubic interior "bubble"
  // function (C1TB space); shape()/dshape_local() here implement the enriched basis, overriding the
  // plain linear shape() inherited from BulkElementTri2dC1.
  class BulkElementTri2dC1TB : public virtual BulkElementTri2dC1
  {
  private:
    static oomph::TBubbleEnrichedGauss<2, 3> Default_enriched_integration_scheme; // Don't know which scheme is best here
    //  static const unsigned Central_node_on_face[3];
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<std::vector<unsigned>>> Dummy_Value_Interpolation_Map;    
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
    static const std::vector<unsigned> Non_Vertex_Node_Indices;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<unsigned> & non_vertex_node_indices() const override {return Non_Vertex_Node_Indices;}
    const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const override {return Dummy_Value_Interpolation_Map;}  
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    BulkElementTri2dC1TB();
    
    void shape(const oomph::Vector<double> &s, oomph::Shape &psi) const override;
    void dshape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids) const override;

    
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const override;
    void shape_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi) const override { this->shape(s, psi); }
    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    void dshape_local_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override { this->dshape_local(s, psi, dpsi); }    
    // The MINI bubble enrichment is pyoomph's own (see elements_2d.cpp), so oomph-lib's
    // TElement<2,2>::d2shape_local would silently ignore the bubble node.
    void d2shape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids, oomph::DShape &d2psids) const override;
    void d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override;
    void d2shape_local_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override { this->d2shape_local_pyoomph(s, psi, dpsi, d2psi); }
    bool supports_second_spatial_derivatives(std::string &) const override { return true; }

    void local_coordinate_of_node(const unsigned &j, oomph::Vector<double> &s) const override;
    bool has_bubble() const override { return true; }
    unsigned get_meshio_type_index() const override { return 66; } // Just some otherwise unused value here
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;

    // A C1TB (MINI) father must spawn genuine C1TB sons. Without this override it inherits
    // BulkElementTri2dC1's factory, which makes plain 3-node C1 sons: the bubble silently
    // disappears from every refined element (MINI degenerates to unstabilised P1-P1), and the
    // father's centroid node, no longer used by any leaf, is deleted as obsolete -- leaving the
    // father with a null node pointer that segfaults as soon as the sons are merged back.
    // Same failure mode as the C2TB case, see BulkElementTri2dC2::create_son_instance.
    BulkElementBase *create_son_instance() const override
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementTri2dC1TB();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }

    void set_integration_order(unsigned int order) override { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 2, order, true)); }
  };

  class BulkElementTri2dC2TB;
  // 2d triangular element, quadratic (C2) interpolation (plus a C1 dummy sub-space); also provides
  // the interpolating_node_pt/interpolating_basis machinery for hanging-node interpolation.
  class BulkElementTri2dC2 : public virtual BulkElementBase, public virtual oomph::TElement<2, 3>, public virtual oomph::RefineableTElement<2>
  {
  protected:    

    static const std::vector<int> Possible_Face_Indices;
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<std::vector<unsigned>>> Dummy_Value_Interpolation_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
    static const std::vector<unsigned> Non_Vertex_Node_Indices;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<unsigned> & non_vertex_node_indices() const override {return Non_Vertex_Node_Indices;}
    const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const override {return Dummy_Value_Interpolation_Map;}  
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    const std::vector<int> & get_possible_face_indices() const override { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;    
    void get_supporting_C1_nodes_of_C2_node(const unsigned &n, std::vector<oomph::Node *> &support) override;
    oomph::Node *boundary_node_pt(const int &face_index, const unsigned int index) override;
    int nedges() const override { return 3; }
    // See BulkElementTri2dC1::get_nodal_s_in_father. Same map; handles 6/7-node (C2/C2TB) via nnode().
    void get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather) override;
    unsigned nnode_on_face() const override { return 3; }
    BulkElementTri2dC2(bool with_bubble = false);    
    unsigned get_meshio_type_index() const override { return 9; }
    
    void check_integrity(double &max_error) override { max_error = 0; } // TODO
    
    
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const override;
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const override { this->shape(s, psi); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const override;
    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    // Second local derivatives: C2 is this element's own (TElement<2,3>) space.
    void d2shape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override { this->d2shape_local_pyoomph(s, psi, dpsi, d2psi); }
    void d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override;
    bool supports_second_spatial_derivatives(std::string &) const override { return true; }
    void add_node_from_finer_neighbor_for_tesselated_numpy(const oomph::Vector<double> &s_coarse, oomph::Node *n, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) override;
    void inform_coarser_neighbors_for_tesselated_numpy(std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) override;
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;
    std::vector<double> get_outline(bool lagrangian) override;
    unsigned nrecovery_order() override { return 2; }
    void output(std::ostream &outfile, const unsigned &n_plot) override { BulkElementBase::output(outfile, n_plot); }
    // TElementBase leaves FiniteElement's macro-element hook broken, so route it to the generic
    // vertex-coordinate implementation. (The Q family must NOT do this: QElementBase already provides
    // a final overrider via its s_macro_ll/ur box, and a second one reachable from the same class
    // would be ambiguous.)
    void get_x_from_macro_element(const oomph::Vector<double> &s, oomph::Vector<double> &x) const override
    { this->get_x_from_generic_macro_element(0, s, x); }
    void get_x_from_macro_element(const unsigned &t, const oomph::Vector<double> &s, oomph::Vector<double> &x) override
    { this->get_x_from_generic_macro_element(t, s, x); }
    unsigned nvertex_node() const override { return oomph::TElement<2, 3>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const override { return TElement<2, 3>::vertex_node_pt(j); }
    // oomph "interpolating node" facilities for mixed-order (C1-on-C2) hanging, analogous to
    // BulkElementQuad2dC2. For C1/C1TB value ids the interpolating nodes are the 3 corner vertices with
    // the linear (barycentric) basis; for C2/C2TB value ids they are the geometric nodes with the
    // quadratic basis. Used by RefineableTElement<2>::setup_hang_for_value / tri_hang_helper.
    void further_setup_hanging_nodes() override;
    oomph::Node *interpolating_node_pt(const unsigned &n, const int &value_id) override;
    double local_one_d_fraction_of_interpolating_node(const unsigned &n1d, const unsigned &i, const int &value_id) override;
    oomph::Node *get_interpolating_node_at_local_coordinate(const oomph::Vector<double> &s, const int &value_id) override;
    unsigned ninterpolating_node_1d(const int &value_id) override;
    unsigned ninterpolating_node(const int &value_id) override;
    void interpolating_basis(const oomph::Vector<double> &s, oomph::Shape &psi, const int &value_id) const override;
    BulkElementBase *create_son_instance() const override;
    void set_integration_order(unsigned int order) override { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 2, order)); }
    oomph::Vector<double> get_midpoint_s() override { return oomph::Vector<double>(this->dim(), 1.0 / 3.0); }
  };

  // 2d triangular element, quadratic interpolation enriched with an interior bubble function
  // (C2TB space); combines BulkElementTri2dC2's quadratic fields with oomph-lib's
  // TBubbleEnrichedElementShape for the enriched geometry/shape functions.
  class BulkElementTri2dC2TB : public virtual BulkElementTri2dC2, public oomph::TBubbleEnrichedElementShape<2, 3>
  {
  private:
    static oomph::TBubbleEnrichedGauss<2, 3> Default_enriched_integration_scheme;
    //  static const unsigned Central_node_on_face[3];
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<std::vector<unsigned>>> Dummy_Value_Interpolation_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
    static const std::vector<unsigned> Non_Vertex_Node_Indices;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<unsigned> & non_vertex_node_indices() const override {return Non_Vertex_Node_Indices;}
    const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const override {return Dummy_Value_Interpolation_Map;}  
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    BulkElementTri2dC2TB();
    
    void shape_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi) const override;
    void dshape_local_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const override { BulkElementTri2dC2::shape(s, psi); }
    void shape_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi) const override { oomph::TBubbleEnrichedElementShape<2, 3>::shape(s, psi); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override { BulkElementTri2dC2::dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override { oomph::TBubbleEnrichedElementShape<2, 3>::dshape_local(s, psi, dpsi); }
    
    // oomph-lib does implement the enriched second derivatives; the override is needed because the
    // inherited BulkElementTri2dC2 one would return the unenriched TElement<2,3> basis.
    inline void d2shape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids, oomph::DShape &d2psids) const override { oomph::TBubbleEnrichedElementShape<2, 3>::d2shape_local(s, psi, dpsids, d2psids); }
    void d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override;
    void d2shape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override;
    void d2shape_local_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override;
    void d2shape_local_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override { this->d2shape_local_pyoomph(s, psi, dpsi, d2psi); }
    bool supports_second_spatial_derivatives(std::string &) const override { return true; }
    inline void shape(const oomph::Vector<double> &s, oomph::Shape &psi) const override { oomph::TBubbleEnrichedElementShape<2, 3>::shape(s, psi); }
    inline void dshape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids) const override { oomph::TBubbleEnrichedElementShape<2, 3>::dshape_local(s, psi, dpsids); }
    inline void local_coordinate_of_node(const unsigned &j, oomph::Vector<double> &s) const override { oomph::TBubbleEnrichedElementShape<2, 3>::local_coordinate_of_node(j, s); }
    bool has_bubble() const override { return true; }
    unsigned get_meshio_type_index() const override { return 99; } // Just some otherwise unused value here
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;

    void set_integration_order(unsigned int order) override { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 2, order, true)); }
  };

  // 3d brick (hexahedral) element, trilinear (C1) interpolation, refineable + solid.
  class BulkElementBrick3dC1 : public virtual BulkElementBase, public virtual oomph::QElement<3, 2>, public virtual oomph::RefineableSolidQElement<3>
  {
  protected:
    static const std::vector<int> Possible_Face_Indices;
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
    static const std::vector<unsigned> Non_Vertex_Node_Indices;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<unsigned> & non_vertex_node_indices() const override {return Non_Vertex_Node_Indices;}
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    const std::vector<int> & get_possible_face_indices() const override { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
    int nedges() const override { return 8; }
    BulkElementBrick3dC1();    
    unsigned get_meshio_type_index() const override { return 11; }


    void check_integrity(double &max_error) override { max_error = 0; } // TODO

    

    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const override { this->shape(s, psi); }
    void shape_at_s_C2(const oomph::Vector<double> &, oomph::Shape &) const override { throw_runtime_error("Makes no sense"); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const override;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &, oomph::Shape &, oomph::DShape &) const override { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    // Second local derivatives: C1 is this element's own (QElement<3,2>) space.
    void d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override { this->d2shape_local_pyoomph(s, psi, dpsi, d2psi); }
    bool supports_second_spatial_derivatives(std::string &) const override { return true; }

    void get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather) override;
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &) const override
    {
      if (tesselate_tri)
      {
        throw_runtime_error("Tesselation of 3d not possible");
      }
      else
      {
        nsubdiv = 1;
        return 8;
      }
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;
    std::vector<double> get_outline(bool lagrangian) override;
    unsigned nrecovery_order() override { return 1; }
    void output(std::ostream &outfile, const unsigned &n_plot) override { BulkElementBase::output(outfile, n_plot); }
    unsigned nvertex_node() const override { return oomph::QElement<3, 2>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const override { return QElement<3, 2>::vertex_node_pt(j); }
    void further_setup_hanging_nodes() override { BulkElementBase::further_setup_hanging_nodes(); } // There can't be any problem here, since it is all isoparametric
    BulkElementBase *create_son_instance() const override
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementBrick3dC1();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    void set_integration_order(unsigned int order) override { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 3, order)); }
    // In a MIXED forest, share interface nodes via the registry (build_as_brick_son); otherwise oomph's native
    // octree build. Defined in elements.cpp (needs RefineablePyramidElement::Mixed_forest_active).
    void build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt, bool &was_already_built, std::ofstream &new_nodes_file) override;
  };

  // 3d brick element, triquadratic (C2) interpolation (plus a C1 dummy sub-space).
  class BulkElementBrick3dC2 : public virtual BulkElementBase, public virtual oomph::QElement<3, 3>, public virtual oomph::RefineableSolidQElement<3>
  {
  protected:
    static const std::vector<int> Possible_Face_Indices;
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<std::vector<unsigned>>> Dummy_Value_Interpolation_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
    static const std::vector<unsigned> Non_Vertex_Node_Indices;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<unsigned> & non_vertex_node_indices() const override {return Non_Vertex_Node_Indices;}
    const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const override {return Dummy_Value_Interpolation_Map;}  
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    const std::vector<int> & get_possible_face_indices() const override { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
    int nedges() const override { return 8; }
    BulkElementBrick3dC2();
    unsigned get_meshio_type_index() const override { return 14; }
    

    void check_integrity(double &max_error) override { max_error = 0; } // TODO
    std::vector<double> get_outline(bool lagrangian) override;
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const override;
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const override { this->shape(s, psi); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const override;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    // Second local derivatives: C2 is this element's own (QElement<3,3>) space.
    void d2shape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override { this->d2shape_local_pyoomph(s, psi, dpsi, d2psi); }
    void d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override;
    bool supports_second_spatial_derivatives(std::string &) const override { return true; }

    

    
    

    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &) const override
    {
      if (tesselate_tri)
      {
        throw_runtime_error("Tesselation of 3d not possible");
      }
      else
      {
        nsubdiv = 1;
        return 27;
      }
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;

    unsigned nrecovery_order() override { return 2; }
    void output(std::ostream &outfile, const unsigned &n_plot) override { BulkElementBase::output(outfile, n_plot); }
    unsigned nvertex_node() const override { return oomph::QElement<3, 3>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const override { return QElement<3, 3>::vertex_node_pt(j); }

    void further_setup_hanging_nodes() override;
    oomph::Node *interpolating_node_pt(const unsigned &n, const int &value_id) override;
    double local_one_d_fraction_of_interpolating_node(const unsigned &n1d, const unsigned &i, const int &value_id) override;
    oomph::Node *get_interpolating_node_at_local_coordinate(const oomph::Vector<double> &s, const int &value_id) override;
    unsigned ninterpolating_node_1d(const int &value_id) override;
    unsigned ninterpolating_node(const int &value_id) override;
    void interpolating_basis(const oomph::Vector<double> &s, oomph::Shape &psi, const int &value_id) const override;
    void get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather) override;
    BulkElementBase *create_son_instance() const override
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementBrick3dC2();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    void set_integration_order(unsigned int order) override { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 3, order)); }
    // In a MIXED forest, share interface nodes via the registry (build_as_brick_son); otherwise oomph's native
    // octree build. Defined in elements.cpp (needs RefineablePyramidElement::Mixed_forest_active).
    void build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt, bool &was_already_built, std::ofstream &new_nodes_file) override;
  };

  // 3d tetrahedral element, linear (C1) interpolation.
  class BulkElementTetra3dC1 : public virtual BulkElementBase, public virtual oomph::TElement<3, 2>, public virtual oomph::RefineableTElement<3>
  {
  protected:
     static const std::vector<int> Possible_Face_Indices;
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;    
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
    static const std::vector<unsigned> Non_Vertex_Node_Indices;
  public:    
    // See BulkElementTri2dC1::get_nodal_s_in_father, 3d counterpart: the son's own local coordinate
    // mapped up through the barycentric affine son->father map of the 1->8 tet split.
    void get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather) override;
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<unsigned> & non_vertex_node_indices() const override {return Non_Vertex_Node_Indices;}
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    const std::vector<int> & get_possible_face_indices() const override { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
    int nedges() const override { return 6; }
    // oomph::TElement<3,*> supplies get_bulk_node_number() (via Node_on_face) but not nnode_on_face().
    unsigned nnode_on_face_by_index(const int & ) const override { return 3; }
    BulkElementTetra3dC1();
    unsigned get_meshio_type_index() const override { return 4; }

    void check_integrity(double &max_error) override { max_error = 0; } // TODO

    
    
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const override { this->shape(s, psi); }
    void shape_at_s_C2(const oomph::Vector<double> &, oomph::Shape &) const override { throw_runtime_error("Makes no sense"); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const override;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &, oomph::Shape &, oomph::DShape &) const override { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    // Second local derivatives: C1 is this element's own (TElement<3,2>) space.
    void d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override { this->d2shape_local_pyoomph(s, psi, dpsi, d2psi); }
    bool supports_second_spatial_derivatives(std::string &) const override { return true; }

    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &) const override
    {
      if (tesselate_tri)
      {
        throw_runtime_error("Tesselation of 3d not possible");
      }
      else
      {
        nsubdiv = 1;
        return 4;
      }
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;
    std::vector<double> get_outline(bool lagrangian) override;
    unsigned nrecovery_order() override { return 1; }
    void output(std::ostream &outfile, const unsigned &n_plot) override { BulkElementBase::output(outfile, n_plot); }
    // As for the triangles: TElementBase leaves this virtual broken, so route it to the generic
    // vertex-coordinate implementation.
    void get_x_from_macro_element(const oomph::Vector<double> &s, oomph::Vector<double> &x) const override
    { this->get_x_from_generic_macro_element(0, s, x); }
    void get_x_from_macro_element(const unsigned &t, const oomph::Vector<double> &s, oomph::Vector<double> &x) override
    { this->get_x_from_generic_macro_element(t, s, x); }
    unsigned nvertex_node() const override { return oomph::TElement<3, 2>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const override { return TElement<3, 2>::vertex_node_pt(j); }
    void further_setup_hanging_nodes() override { BulkElementBase::further_setup_hanging_nodes(); } // There can't be any problem here, since it is all isoparametric
    BulkElementBase *create_son_instance() const override
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementTetra3dC1();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    void set_integration_order(unsigned int order) override { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 3, order)); }
    oomph::Vector<double> get_midpoint_s() override { return oomph::Vector<double>(this->dim(), 1.0 / 3.0); }
  };

  // 3d tetrahedral element, linear interpolation enriched with a quartic interior bubble function (C1TB space).
  class BulkElementTetra3dC1TB : public virtual BulkElementTetra3dC1
  {
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<std::vector<unsigned>>> Dummy_Value_Interpolation_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
    static const std::vector<unsigned> Non_Vertex_Node_Indices;
  public:
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<unsigned> & non_vertex_node_indices() const override {return Non_Vertex_Node_Indices;}
    const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const override {return Dummy_Value_Interpolation_Map;}
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    BulkElementTetra3dC1TB();
    unsigned get_meshio_type_index() const override { return 44; }
    void shape(const oomph::Vector<double> &s, oomph::Shape &psi) const override;
    void dshape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids) const override;
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const override;
    void shape_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi) const override { this->shape(s, psi); }
    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    void dshape_local_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override { this->dshape_local(s, psi, dpsi); }
    // Without this override the inherited oomph::TElement<3,2>::d2shape_local would be used, which
    // knows nothing about the 5th (bubble) node and would silently return wrong numbers.
    void d2shape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids, oomph::DShape &d2psids) const override;
    void d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override;
    void d2shape_local_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override { this->d2shape_local_pyoomph(s, psi, dpsi, d2psi); }
    bool supports_second_spatial_derivatives(std::string &) const override { return true; }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;
    void local_coordinate_of_node(const unsigned &j, oomph::Vector<double> &s) const override;
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &) const override
    {
      if (tesselate_tri)
      {
        throw_runtime_error("Tesselation of 3d not possible");
      }
      else
      {
        nsubdiv = 1;
        return 5;
      }
    }
    BulkElementBase *create_son_instance() const override
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementTetra3dC1TB();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    void set_integration_order(unsigned int order) override { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 3, order,true)); }
  };

  class BulkElementTetra3dC2TB;
  // 3d tetrahedral element, quadratic (C2) interpolation (plus a C1 dummy sub-space).
  class BulkElementTetra3dC2 : public virtual BulkElementBase, public virtual oomph::TElement<3, 3>, public virtual oomph::RefineableTElement<3>
  {
  protected:
    static const std::vector<int> Possible_Face_Indices;
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<std::vector<unsigned>>> Dummy_Value_Interpolation_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
    static const std::vector<unsigned> Non_Vertex_Node_Indices;
  public:    
    // See BulkElementTri2dC1::get_nodal_s_in_father, 3d counterpart: the son's own local coordinate
    // mapped up through the barycentric affine son->father map of the 1->8 tet split.
    void get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather) override;
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<unsigned> & non_vertex_node_indices() const override {return Non_Vertex_Node_Indices;}
    const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const override {return Dummy_Value_Interpolation_Map;}  
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    const std::vector<int> & get_possible_face_indices() const override { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
    int nedges() const override { return 6; }
    // 3 vertices + 3 mid-edge nodes per face; Node_on_face is [4][6] here.
    unsigned nnode_on_face_by_index(const int & ) const override { return 6; }
    BulkElementTetra3dC2(bool has_bubble = false);
    unsigned get_meshio_type_index() const override { return 10; }
    

    void check_integrity(double &max_error) override { max_error = 0; } // TODO
    std::vector<double> get_outline(bool lagrangian) override;
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const override;
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const override { this->shape(s, psi); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const override;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    // Second local derivatives: C2 is this element's own (TElement<3,3>) space.
    void d2shape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override { this->d2shape_local_pyoomph(s, psi, dpsi, d2psi); }
    void d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override;
    bool supports_second_spatial_derivatives(std::string &) const override { return true; }

    

    
    

    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &) const override
    {
      if (tesselate_tri)
      {
        throw_runtime_error("Tesselation of 3d not possible");
      }
      else
      {
        nsubdiv = 1;
        return 10;
      }
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;

    unsigned nrecovery_order() override { return 2; }
    void output(std::ostream &outfile, const unsigned &n_plot) override { BulkElementBase::output(outfile, n_plot); }
    // As for the triangles: TElementBase leaves this virtual broken, so route it to the generic
    // vertex-coordinate implementation.
    void get_x_from_macro_element(const oomph::Vector<double> &s, oomph::Vector<double> &x) const override
    { this->get_x_from_generic_macro_element(0, s, x); }
    void get_x_from_macro_element(const unsigned &t, const oomph::Vector<double> &s, oomph::Vector<double> &x) override
    { this->get_x_from_generic_macro_element(t, s, x); }
    unsigned nvertex_node() const override { return oomph::TElement<3, 3>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const override { return TElement<3, 3>::vertex_node_pt(j); }

    void further_setup_hanging_nodes() override;
    // interpolating_basis() alone used to be overridden here, which left ninterpolating_node() at oomph's
    // isoparametric default (nnode()==10) while the C1 basis only writes 4 entries -- so callers read 6
    // UNINITIALISED doubles from the Shape array (oomph::Shape allocates with new double[N]). It happened
    // to work because a TElement numbers its vertices first, so the garbage entries were usually rejected
    // by the |psi|>1e-12 master test; that is luck, not correctness. The four hooks now agree with each
    // other. See BulkElementBase::interpolation_value_is_C1.
    unsigned ninterpolating_node(const int &value_id) override { return this->generic_ninterpolating_node(value_id); }
    oomph::Node *interpolating_node_pt(const unsigned &n, const int &value_id) override { return this->generic_interpolating_node_pt(n, value_id); }
    oomph::Node *get_interpolating_node_at_local_coordinate(const oomph::Vector<double> &s, const int &value_id) override { return this->generic_get_interpolating_node_at_local_coordinate(s, value_id); }
    void interpolating_basis(const oomph::Vector<double> &s, oomph::Shape &psi, const int &value_id) const override;
    BulkElementBase *create_son_instance() const override;
    void set_integration_order(unsigned int order) override { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 3, order)); }
    oomph::Vector<double> get_midpoint_s() override { return oomph::Vector<double>(this->dim(), 1.0 / 3.0); }
  };

  // 3d tetrahedral element, quadratic interpolation enriched with an interior bubble function (C2TB space).
  class BulkElementTetra3dC2TB : public virtual BulkElementTetra3dC2, public oomph::TBubbleEnrichedElementShape<3, 3>
  {
  private:
    static oomph::TBubbleEnrichedGauss<3, 3> Default_enriched_integration_scheme;
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<std::vector<unsigned>>> Dummy_Value_Interpolation_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
    static const std::vector<unsigned> Non_Vertex_Node_Indices;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<unsigned> & non_vertex_node_indices() const override {return Non_Vertex_Node_Indices;}
    const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const override {return Dummy_Value_Interpolation_Map;}  
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    // The 15-node enriched tet adds one bubble node per face (10..13) plus one cell-interior bubble
    // (14). The face bubbles live OUTSIDE oomph::TElement<3,3>::Node_on_face, which is only [4][6],
    // so build_face_element() has to wire them in by hand -- and so does node_index_on_face() below.
    // Getting this wrong would make the boundary-membership repair strip a genuine membership, since
    // these nodes are created with boundary_possible=true (meshtemplate.cpp).
    static const std::vector<unsigned> Central_node_on_face;
    unsigned nnode_on_face_by_index(const int & ) const override { return 7; }
    unsigned node_index_on_face(const int & face_index, const unsigned & i) const override
      { return (i < 6 ? BulkElementTetra3dC2::node_index_on_face(face_index, i) : Central_node_on_face[face_index]); }
    BulkElementTetra3dC2TB();
    unsigned get_meshio_type_index() const override { return 100; } // Just some otherwise unused value here
    
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &) const override
    {
      if (tesselate_tri)
      {
        throw_runtime_error("Tesselation of 3d not possible");
      }
      else
      {
        nsubdiv = 1;
        return 15;
      }
    }
    //   void fill_element_nodal_indices_for_numpy(int *indices,unsigned isubelem,bool tesselate_tri,std::vector<std::vector<std::set<oomph::Node*>>> & add_nodes) const;

    
    
    
    void shape_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi) const override;    
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const override { BulkElementTetra3dC2::shape(s, psi); }
    void shape_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi) const override { oomph::TBubbleEnrichedElementShape<3, 3>::shape(s, psi); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override { BulkElementTetra3dC2::dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    void dshape_local_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override { oomph::TBubbleEnrichedElementShape<3, 3>::dshape_local(s, psi, dpsi); }
    // As for the 2d bubble-enriched triangle above.
    inline void d2shape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids, oomph::DShape &d2psids) const override { oomph::TBubbleEnrichedElementShape<3, 3>::d2shape_local(s, psi, dpsids, d2psids); }
    void d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override;
    void d2shape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override;
    void d2shape_local_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override;
    void d2shape_local_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override { this->d2shape_local_pyoomph(s, psi, dpsi, d2psi); }
    bool supports_second_spatial_derivatives(std::string &) const override { return true; }
    inline void shape(const oomph::Vector<double> &s, oomph::Shape &psi) const override { oomph::TBubbleEnrichedElementShape<3, 3>::shape(s, psi); }
    inline void dshape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids) const override { oomph::TBubbleEnrichedElementShape<3, 3>::dshape_local(s, psi, dpsids); }
    inline void local_coordinate_of_node(const unsigned &j, oomph::Vector<double> &s) const override { oomph::TBubbleEnrichedElementShape<3, 3>::local_coordinate_of_node(j, s); }
    bool has_bubble() const override { return true; }
    void set_integration_order(unsigned int order) override { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 3, order, true)); }
    void build_face_element(const int& face_index, oomph::FaceElement* face_element_pt) override;
  };

  // 3d wedge (triangular prism) element, linear (C1) interpolation.
  class BulkElementWedge3dC1 : public virtual BulkElementBase, public virtual oomph::WedgeElementC1
  {
    protected:
      static const std::vector<int> Possible_Face_Indices;
      static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
    static const std::vector<unsigned> Non_Vertex_Node_Indices;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<unsigned> & non_vertex_node_indices() const override {return Non_Vertex_Node_Indices;}
      const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
      oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
      const std::vector<int> & get_possible_face_indices() const override { return Possible_Face_Indices; }
      std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
      BulkElementWedge3dC1();
      // oomph::WedgeElementC1 already declares nnode_on_face_by_index() with this exact signature, and
      // so does BulkElementBase -- two independent bases, hence two vtable slots. Without this single
      // derived declaration (which legally overrides both) a call through a BulkElementBase* would
      // reach BulkElementBase's default, i.e. the throwing nnode_on_face(). Same below for the other
      // three wedge/pyramid classes.
      unsigned nnode_on_face_by_index(const int & face_index) const override { return oomph::WedgeElementC1::nnode_on_face_by_index(face_index); }
      int nedges() const override { throw_runtime_error("Not implemented"); }
      unsigned get_meshio_type_index() const override { return 13; }      
      void shape(const oomph::Vector<double> &s, oomph::Shape &psi) const override {oomph::WedgeElementC1::shape(s, psi); }
      // WedgeElementBase/PyramidElementBase leave FiniteElement's macro-element hook broken, as the
      // T family does; route it to the generic vertex-coordinate implementation.
      void get_x_from_macro_element(const oomph::Vector<double> &s, oomph::Vector<double> &x) const override
      { this->get_x_from_generic_macro_element(0, s, x); }
      void get_x_from_macro_element(const unsigned &t, const oomph::Vector<double> &s, oomph::Vector<double> &x) override
      { this->get_x_from_generic_macro_element(t, s, x); }
      void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const override { this->shape(s, psi); }
      void shape_at_s_C2(const oomph::Vector<double> &, oomph::Shape &) const override { throw_runtime_error("Makes no sense"); }
      void shape_at_s_DL(const oomph::Vector<double> &, oomph::Shape &) const override;
      void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override { this->dshape_local(s, psi, dpsi); }
      void dshape_local_at_s_C2(const oomph::Vector<double> &, oomph::Shape &, oomph::DShape &) const override { throw_runtime_error("Makes no sense"); }
      void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
      // Second local derivatives: C1 is this element's own space.
      void d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override { this->d2shape_local_pyoomph(s, psi, dpsi, d2psi); }
      bool supports_second_spatial_derivatives(std::string &) const override { return true; }
      
      int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &) const override
      {
          if (tesselate_tri)
          {
            throw_runtime_error("Tesselation of 3d not possible");
          }
          else
          {
            nsubdiv = 1;
            return 6;
          }
      }
      void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;
      std::vector<double> get_outline(bool lagrangian) override;
      unsigned nrecovery_order() override { return 1; }
      void output(std::ostream &outfile, const unsigned &n_plot) override { BulkElementBase::output(outfile, n_plot); }      
      unsigned nvertex_node() const override { return oomph::WedgeElementC1::nvertex_node(); }
      oomph::Node *vertex_node_pt(const unsigned &j) const override { return WedgeElementC1::vertex_node_pt(j); }      
      void further_setup_hanging_nodes() override { BulkElementBase::further_setup_hanging_nodes(); } // There can't be any problem here, since it is all isoparametric
      BulkElementBase *create_son_instance() const override
      {
          BulkElementBase::__CurrentCodeInstance = codeinst;
          auto res = new BulkElementWedge3dC1();
          res->codeinst = codeinst;
          BulkElementBase::__CurrentCodeInstance = NULL;
          return res;
      }
      void set_integration_order(unsigned int order) override { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 4, order)); }
      oomph::Vector<double> get_midpoint_s() override { oomph::Vector<double> res(this->dim(), 1.0 / 3.0); res[2]=0.5; return res; }
  };


  // 3d pyramid element, linear (C1) interpolation.
  class BulkElementPyramid3dC1 : public virtual BulkElementBase, public virtual oomph::PyramidElementC1
  {
    protected:
      static const std::vector<int> Possible_Face_Indices;
      static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
    static const std::vector<unsigned> Non_Vertex_Node_Indices;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<unsigned> & non_vertex_node_indices() const override {return Non_Vertex_Node_Indices;}
      const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
      oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
      const std::vector<int> & get_possible_face_indices() const override { return Possible_Face_Indices; }
      std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
      BulkElementPyramid3dC1();
      unsigned nnode_on_face_by_index(const int & face_index) const override { return oomph::PyramidElementC1::nnode_on_face_by_index(face_index); }
      int nedges() const override { throw_runtime_error("Not implemented"); } // No need tom implement this now
      unsigned get_meshio_type_index() const override { return 15; }      
      void shape(const oomph::Vector<double> &s, oomph::Shape &psi) const override {oomph::PyramidElementC1::shape(s, psi); }
      // WedgeElementBase/PyramidElementBase leave FiniteElement's macro-element hook broken, as the
      // T family does; route it to the generic vertex-coordinate implementation.
      void get_x_from_macro_element(const oomph::Vector<double> &s, oomph::Vector<double> &x) const override
      { this->get_x_from_generic_macro_element(0, s, x); }
      void get_x_from_macro_element(const unsigned &t, const oomph::Vector<double> &s, oomph::Vector<double> &x) override
      { this->get_x_from_generic_macro_element(t, s, x); }
      void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const override { this->shape(s, psi); }
      void shape_at_s_C2(const oomph::Vector<double> &, oomph::Shape &) const override { throw_runtime_error("Makes no sense"); }
      void shape_at_s_DL(const oomph::Vector<double> &, oomph::Shape &) const override;
      void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override { this->dshape_local(s, psi, dpsi); }
      void dshape_local_at_s_C2(const oomph::Vector<double> &, oomph::Shape &, oomph::DShape &) const override { throw_runtime_error("Makes no sense"); }
      void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;      
      // Second local derivatives: C1 is this element's own space.
      void d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override { this->d2shape_local_pyoomph(s, psi, dpsi, d2psi); }
      bool supports_second_spatial_derivatives(std::string &) const override { return true; }
      int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &) const override
      {
          if (tesselate_tri)
          {
            throw_runtime_error("Tesselation of 3d not possible");
          }
          else
          {
            nsubdiv = 1;
            return 5;
          }
      }
      void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;
      std::vector<double> get_outline(bool lagrangian) override;
      unsigned nrecovery_order() override { return 1; }
      void output(std::ostream &outfile, const unsigned &n_plot) override { BulkElementBase::output(outfile, n_plot); }      
      unsigned nvertex_node() const override { return oomph::PyramidElementC1::nvertex_node(); }
      oomph::Node *vertex_node_pt(const unsigned &j) const override { return PyramidElementC1::vertex_node_pt(j); }      
      void further_setup_hanging_nodes() override { BulkElementBase::further_setup_hanging_nodes(); } // There can't be any problem here, since it is all isoparametric
      BulkElementBase *create_son_instance() const override
      {
          BulkElementBase::__CurrentCodeInstance = codeinst;
          auto res = new BulkElementPyramid3dC1();
          res->codeinst = codeinst;
          BulkElementBase::__CurrentCodeInstance = NULL;
          return res;
      }
      // Factory for a TETRAHEDRAL son of the same physics (same codeinst) -- used by
      // PyramidMixedRefinementPattern for the 4 tet children of the 6-pyramid+4-tet red split.
      BulkElementBase *create_tet_son_instance() const;
      // Mixed (pyramid -> 6 pyramids + 4 tets) split scheme; see PyramidMixedRefinementPattern.
      const RefinementPattern *refinement_pattern() const override;
      void set_integration_order(unsigned int order) override { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 5, order)); }
      oomph::Vector<double> get_midpoint_s() override { oomph::Vector<double> res(this->dim(), 0.375); res[2]=0.25; return res; }
  };


  // 3d wedge element, quadratic (C2) interpolation (plus a C1 dummy sub-space).
  class BulkElementWedge3dC2 : public virtual BulkElementBase, public virtual oomph::WedgeElementC2
  {
    protected:
      static const std::vector<int> Possible_Face_Indices;      
      static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
      static const std::vector<std::vector<std::vector<unsigned>>> Dummy_Value_Interpolation_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
    static const std::vector<unsigned> Non_Vertex_Node_Indices;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<unsigned> & non_vertex_node_indices() const override {return Non_Vertex_Node_Indices;}
      const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const override {return Dummy_Value_Interpolation_Map;}    
      const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
      oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
      const std::vector<int> & get_possible_face_indices() const override { return Possible_Face_Indices; }
      std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
      BulkElementWedge3dC2();
      unsigned nnode_on_face_by_index(const int & face_index) const override { return oomph::WedgeElementC2::nnode_on_face_by_index(face_index); }
      int nedges() const override { throw_runtime_error("Not implemented"); }
      unsigned get_meshio_type_index() const override { return 26; }
      void shape(const oomph::Vector<double> &s, oomph::Shape &psi) const override {oomph::WedgeElementC2::shape(s, psi); }
      void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const override { oomph::WedgeElementShapeC1::shape(s, psi); }
      void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const override { this->shape(s, psi); }      
      void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const override;
      void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override { this->dshape_local(s, psi, dpsi); }
      void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override { oomph::WedgeElementShapeC1::dshape_local(s, psi, dpsi); }
      void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
      // Second local derivatives: C2 is this element's own space; C1 comes from the linear shape class.
      void d2shape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override { this->d2shape_local_pyoomph(s, psi, dpsi, d2psi); }
      void d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override;
      bool supports_second_spatial_derivatives(std::string &) const override { return true; }
      
      int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &) const override
      {
          if (tesselate_tri)
          {
            throw_runtime_error("Tesselation of 3d not possible");
          }
          else
          {
            nsubdiv = 1;
            return 18;
          }
      }
      void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;
      std::vector<double> get_outline(bool lagrangian) override;
      unsigned nrecovery_order() override { return 1; }
      void output(std::ostream &outfile, const unsigned &n_plot) override { BulkElementBase::output(outfile, n_plot); }      
      unsigned nvertex_node() const override { return oomph::WedgeElementC2::nvertex_node(); }
      oomph::Node *vertex_node_pt(const unsigned &j) const override { return WedgeElementC2::vertex_node_pt(j); }      
      void further_setup_hanging_nodes() override { BulkElementBase::further_setup_hanging_nodes(); }
      // A C1 field on this C2-geometry element must hang on the LINEAR basis over the corner vertices,
      // not on the quadratic geometric basis over all nodes -- see BulkElementBase::interpolation_value_is_C1.
      // (The former comment here, "there can't be any problem since it is all isoparametric", only holds
      // while every field shares the geometric space.)
      unsigned ninterpolating_node(const int &value_id) override { return this->generic_ninterpolating_node(value_id); }
      oomph::Node *interpolating_node_pt(const unsigned &n, const int &value_id) override { return this->generic_interpolating_node_pt(n, value_id); }
      void interpolating_basis(const oomph::Vector<double> &s, oomph::Shape &psi, const int &value_id) const override { this->generic_interpolating_basis(s, psi, value_id); }
      oomph::Node *get_interpolating_node_at_local_coordinate(const oomph::Vector<double> &s, const int &value_id) override { return this->generic_get_interpolating_node_at_local_coordinate(s, value_id); }
      BulkElementBase *create_son_instance() const override
      {
          BulkElementBase::__CurrentCodeInstance = codeinst;
          auto res = new BulkElementWedge3dC2();
          res->codeinst = codeinst;
          BulkElementBase::__CurrentCodeInstance = NULL;
          return res;
      }
      void set_integration_order(unsigned int order) override { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 4, order)); }
      oomph::Vector<double> get_midpoint_s() override { oomph::Vector<double> res(this->dim(), 1.0 / 3.0); res[2]=0.5; return res; }            
  };

  // Maxim: Add BulkElementPyramid3dC2
  // class BulkElementPyramid3dC2 : public virtual BulkElementBase, public virtual oomph::PyramidElementC2
  // {
  //   protected:
  //     static const std::vector<int> Possible_Face_Indices;
  //     static int element_index_to_C1[14];
  //     static bool node_only_C2[14]; // TODO Including the C2TBs
  //     static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
  //   public:
  //     const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
  //     oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
  //     virtual const std::vector<int> & get_possible_face_indices() const { return Possible_Face_Indices; }
  //     std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
  //     BulkElementPyramid3dC2();
  //     void interpolate_hang_values() override;
  //     int nedges() const { throw_runtime_error("Not implemented"); }
  //     virtual unsigned get_meshio_type_index() const { return 27; }
  //     bool fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap);
  //     void shape(const oomph::Vector<double> &s, oomph::Shape &psi) const {oomph::PyramidElementC2::shape(s, psi); }
  //     void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const { oomph::PyramidElementShapeC1::shape(s, psi); }
  //     void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }      
  //     void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;
  //     void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
  //     void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { oomph::PyramidElementShapeC1::dshape_local(s, psi, dpsi); }
  //     void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
  //     unsigned int get_node_index_C1_to_element(const unsigned int &i) const { return i; } // Same as for C1
  //     unsigned int get_node_index_C2_to_element(const unsigned int &i) const { return i; }
  //     int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
  //     {
  //         if (tesselate_tri)
  //         {
  //           throw_runtime_error("Tesselation of 3d not possible");
  //         }
  //         else
  //         {
  //           nsubdiv = 1;
  //           return 14;
  //         }
  //     }
  //     void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
  //     virtual std::vector<double> get_outline(bool lagrangian);
  //     unsigned nrecovery_order() { return 1; }
  //     void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }      
  //     unsigned nvertex_node() const { return oomph::PyramidElementC2::nvertex_node(); }
  //     oomph::Node *vertex_node_pt(const unsigned &j) const { return PyramidElementC2::vertex_node_pt(j); }      
  //     void further_setup_hanging_nodes() { BulkElementBase::further_setup_hanging_nodes(); } // There can't be any problem here, since it is all isoparametric
  //     virtual BulkElementBase *create_son_instance() const
  //     {
  //         BulkElementBase::__CurrentCodeInstance = codeinst;
  //         auto res = new BulkElementPyramid3dC2();
  //         res->codeinst = codeinst;
  //         BulkElementBase::__CurrentCodeInstance = NULL;
  //         return res;
  //     }
  //     virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 4, order)); }
  //     oomph::Vector<double> get_midpoint_s() override { oomph::Vector<double> res(this->dim(), 1.0 / 3.0); res[2]=0.5; return res; }
  //     bool is_node_index_part_of_C1(const unsigned &n) override { return !node_only_C2[n]; }
  //     int get_node_index_element_to_C1(const unsigned int &i) const override { return element_index_to_C1[i]; }
  // };

  class BulkElementPyramid3dC2 : public virtual BulkElementBase, public virtual oomph::PyramidElementC2
  {
    protected:
      static const std::vector<int> Possible_Face_Indices;      
      static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
      static const std::vector<std::vector<std::vector<unsigned>>> Dummy_Value_Interpolation_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
    static const std::vector<unsigned> Non_Vertex_Node_Indices;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<unsigned> & non_vertex_node_indices() const override {return Non_Vertex_Node_Indices;}
      const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const override {return Dummy_Value_Interpolation_Map;}    
      const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
      oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
      const std::vector<int> & get_possible_face_indices() const override { return Possible_Face_Indices; }
      std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
      BulkElementPyramid3dC2();
      unsigned nnode_on_face_by_index(const int & face_index) const override { return oomph::PyramidElementC2::nnode_on_face_by_index(face_index); }
      int nedges() const override { throw_runtime_error("Not implemented"); }
      unsigned get_meshio_type_index() const override { return 27; }      
      void shape(const oomph::Vector<double> &s, oomph::Shape &psi) const override {oomph::PyramidElementC2::shape(s, psi); }
      void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const override { oomph::PyramidElementShapeC1::shape(s, psi); }
      void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const override { this->shape(s, psi); }      
      void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const override;
      void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override { this->dshape_local(s, psi, dpsi); }
      void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override { oomph::PyramidElementShapeC1::dshape_local(s, psi, dpsi); }
      void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
      // Second local derivatives: C2 is this element's own space; C1 comes from the linear shape class.
      void d2shape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override { this->d2shape_local_pyoomph(s, psi, dpsi, d2psi); }
      void d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const override;
      bool supports_second_spatial_derivatives(std::string &) const override { return true; }
      
      int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &) const override
      {
          if (tesselate_tri)
          {
            throw_runtime_error("Tesselation of 3d not possible");
          }
          else
          {
            nsubdiv = 1;
            return 14;
          }
      }
      void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;
      std::vector<double> get_outline(bool lagrangian) override;
      unsigned nrecovery_order() override { return 1; }
      void output(std::ostream &outfile, const unsigned &n_plot) override { BulkElementBase::output(outfile, n_plot); }      
      unsigned nvertex_node() const override { return oomph::PyramidElementC2::nvertex_node(); }
      oomph::Node *vertex_node_pt(const unsigned &j) const override { return PyramidElementC2::vertex_node_pt(j); }      
      void further_setup_hanging_nodes() override { BulkElementBase::further_setup_hanging_nodes(); }
      // A C1 field on this C2-geometry element must hang on the LINEAR basis over the corner vertices,
      // not on the quadratic geometric basis over all nodes -- see BulkElementBase::interpolation_value_is_C1.
      // (The former comment here, "there can't be any problem since it is all isoparametric", only holds
      // while every field shares the geometric space.)
      unsigned ninterpolating_node(const int &value_id) override { return this->generic_ninterpolating_node(value_id); }
      oomph::Node *interpolating_node_pt(const unsigned &n, const int &value_id) override { return this->generic_interpolating_node_pt(n, value_id); }
      void interpolating_basis(const oomph::Vector<double> &s, oomph::Shape &psi, const int &value_id) const override { this->generic_interpolating_basis(s, psi, value_id); }
      oomph::Node *get_interpolating_node_at_local_coordinate(const oomph::Vector<double> &s, const int &value_id) override { return this->generic_get_interpolating_node_at_local_coordinate(s, value_id); }
      BulkElementBase *create_son_instance() const override
      {
          BulkElementBase::__CurrentCodeInstance = codeinst;
          auto res = new BulkElementPyramid3dC2();
          res->codeinst = codeinst;
          BulkElementBase::__CurrentCodeInstance = NULL;
          return res;
      }
      // Mixed red-split of a C2 pyramid: 6 sub-pyramids (create_son_instance) + 4 tets. The tet son is a
      // C2 tet bound to the same physics, mirroring BulkElementPyramid3dC1::create_tet_son_instance.
      BulkElementBase *create_tet_son_instance() const;
      const RefinementPattern *refinement_pattern() const override;
      void set_integration_order(unsigned int order) override { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 5, order)); }
      oomph::Vector<double> get_midpoint_s() override { oomph::Vector<double> res(this->dim(), 0.375); res[2]=0.25; return res; }
  };

  // 0-dimensional spatial point element (a single node, no extent) - used e.g. as the "face
  // element" attached to the endpoint of a 1d line element, or standalone for point-sampled
  // physics. Unlike ODEElementBase/BulkElementODE0d, this does have one spatial node and a (trivial)
  // geometric mapping.
  class PointElement0d : public virtual BulkElementBase, public virtual oomph::PointElement
  {
  protected:
    static const std::vector<int> Possible_Face_Indices;
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
    static const std::vector<unsigned> Non_Vertex_Node_Indices;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<unsigned> & non_vertex_node_indices() const override {return Non_Vertex_Node_Indices;}
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *, int ) override {throw_runtime_error("A point element has no faces");}
    const std::vector<int> & get_possible_face_indices() const override { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & ) const override {return std::vector<pyoomph::Node*>{}; }
    int nedges() const override { return 0; }
    PointElement0d();
    unsigned get_meshio_type_index() const override { return 0; }
    void dshape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids) const override;
    double invert_jacobian_mapping(const oomph::DenseMatrix<double> &jacobian, oomph::DenseMatrix<double> &inverse_jacobian) const override;
    void build(oomph::Mesh *&, oomph::Vector<oomph::Node *> &, bool &, std::ofstream &) override {}
    void check_integrity(double &max_error) override { max_error = 0; }
    
    void output(std::ostream &outfile, const unsigned &n_plot) override { BulkElementBase::output(outfile, n_plot); }
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const override;
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const override;
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const override;
    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    unsigned nrecovery_order() override { return 1; }
    unsigned nvertex_node() const override { return 1; }
    oomph::Node *vertex_node_pt(const unsigned &j) const override { return node_pt(j); }
    void further_setup_hanging_nodes() override {};
    BulkElementBase *create_son_instance() const override
    {
      throw_runtime_error("Makes no sense");
      return NULL;
    }
    void pre_build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt) override
    {
      BulkElementBase::pre_build(mesh_pt, new_node_pt);
    }
    int get_num_numpy_elemental_indices(bool , unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &) const override
    {
      nsubdiv = 1;
      return 1;
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const override;
    std::vector<double> get_outline(bool lagrangian) override;   
    double get_quality_factor() override { return 1.0; }
    double s_min() const override
    {
      return 0.0;
    }

    double s_max() const override
    {
      return 0.0;
    }
    void set_integration_order(unsigned int ) override {}
  };

  // --- The following classes instantiate InterfaceElement<BASE> for each bulk element type,
  // adding the geometry-specific opposite-side matching (analyze_opposite_orientation) and local
  // coordinate transfer (local_coordinate_in_opposite_side) described in InterfaceElementBase's
  // class comment above. Individual functions are commented only where the logic is non-obvious;
  // the general approach (match vertex nodes by nearest distance under all admissible
  // permutations/orientations, then derive the coordinate transform from the chosen orientation) is
  // the same across all of them.

  // Point (0d) interface element, attached to the endpoint of a 1d line bulk element.
  class InterfaceElementPoint0d : public virtual InterfaceElement<PointElement0d>
  {
  protected:
  public:
    InterfaceElementPoint0d(DynamicBulkElementInstance *jitcode, FiniteElement *const &bulk_el_pt, const int &face_index) : InterfaceElement<PointElement0d>(jitcode, bulk_el_pt, face_index)
    {
    }
    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const override
    {
      return s;
    }
    // A point has no orientation ambiguity; just checks the opposite side is also a point element.
    void analyze_opposite_orientation(const std::vector<double> & ) override
    {
      if (!dynamic_cast<InterfaceElementPoint0d *>(opposite_side))
      {
        throw_runtime_error("Can only connect an InterfaceElementPoint0d to an InterfaceElementPoint0d");
      }
      opposite_orientation = 0; // Does not matter anyhow
      opposite_node_index.resize(1, 0);
    }
  };

  // Line (1d) interface element on a Q-family (quadrilateral/brick) bulk element's C1 face.
  class InterfaceElementLine1dC1 : public InterfaceElement<BulkElementLine1dC1>
  {
  protected:
    bool partial_opposite_internal_facet;
    double partial_opposite_s_at_smin, partial_opposite_s_at_smax;

  public:
    InterfaceElementLine1dC1(DynamicBulkElementInstance *jitcode, FiniteElement *const &bulk_el_pt, const int &face_index) : InterfaceElement<BulkElementLine1dC1>(jitcode, bulk_el_pt, face_index), partial_opposite_internal_facet(false)
    {
    }

    pyoomph::Node *opposite_node_pt(unsigned int i) override
    {
      if (partial_opposite_internal_facet)
        throw_runtime_error("opposite_node_pt not allowed in internal facets with partial overlap with the opposite side");
      return InterfaceElement<BulkElementLine1dC1>::opposite_node_pt(i);
    }

    // Matches this element's 2 vertex (endpoint) nodes against the opposite side's, trying both
    // orientations (0: same order, 1: swapped) and picking whichever gives the smaller total
    // squared distance. If neither matches within tolerance, falls back to treating this as a
    // partially-overlapping "internal facet" pairing (only allowed if the opposite side has been
    // marked as such), where the two elements do not share coincident nodes and coordinates must
    // instead be mapped continuously via optimize_s_to_match_x at the endpoints.
    void analyze_opposite_orientation(const std::vector<double> & offset) override
    {
      if (opposite_side->dim() != 1)
      {
        throw_runtime_error("Can only connect a 1d InterfaceElement to a 1d InterfaceElement");
      }
      if (this->nvertex_node() != opposite_side->nvertex_node())
      {
        throw_runtime_error("Can only connect InterfaceElements with same number of vertex nodes");
      }

      double dist0 = 0.0;
      double dist1 = 0.0;
      for (unsigned int i = 0; i < this->nvertex_node(); i++)
      {
        pyoomph::Node *nthis = dynamic_cast<pyoomph::Node *>(this->vertex_node_pt(i));
        pyoomph::Node *nopp = dynamic_cast<pyoomph::Node *>(opposite_side->vertex_node_pt(i));
        for (unsigned int k = 0; k < std::min(nthis->ndim(), nopp->ndim()); k++)
          dist0 += (nthis->x(k) - nopp->x(k)+offset[k]) * (nthis->x(k) - nopp->x(k)+offset[k]);
        nopp = dynamic_cast<pyoomph::Node *>(opposite_side->vertex_node_pt(1 - i));
        for (unsigned int k = 0; k < std::min(nthis->ndim(), nopp->ndim()); k++)
          dist1 += (nthis->x(k) - nopp->x(k)+offset[k]) * (nthis->x(k) - nopp->x(k)+offset[k]);
      }
      opposite_orientation = (dist0 < dist1 ? 0 : 1);
      /*      if (dynamic_cast<BulkTElementLine1dC1*>(opposite_side))
            {
             std::cout << "FOUND TRI OPPOSITE TO QUAD " << dist0 << "   " << dist1 << std::endl;
            }*/
      if ((dist0 < dist1 ? dist0 : dist1) > 1e-14)
      {
        if (!opposite_side->is_internal_facet_opposite_dummy())
        {
          throw_runtime_error("Vertex nodes are not matching here. This is only allowed for internal facets");
        }
        partial_opposite_internal_facet = true;
        oomph::Vector<double> x_at_smin(this->nodal_dimension(), 0.0), x_at_smax(this->nodal_dimension(), 0.0);
        this->interpolated_x(oomph::Vector<double>(1, this->s_min()), x_at_smin);
        this->interpolated_x(oomph::Vector<double>(1, this->s_max()), x_at_smax);
        partial_opposite_s_at_smin = opposite_side->optimize_s_to_match_x(x_at_smin)[0];
        partial_opposite_s_at_smax = opposite_side->optimize_s_to_match_x(x_at_smax)[0];
      }
      else
      {
        opposite_node_index.resize(2);
        if (opposite_side->nnode() == 2)
        {
          if (!opposite_orientation)
          {
            opposite_node_index[0] = 0;
            opposite_node_index[1] = 1;
          }
          else
          {
            opposite_node_index[0] = 1;
            opposite_node_index[1] = 0;
          }
        }
        else if (opposite_side->nnode() == 3)
        {
          if (!opposite_orientation)
          {
            opposite_node_index[0] = 0;
            opposite_node_index[1] = 2;
          }
          else
          {
            opposite_node_index[0] = 2;
            opposite_node_index[1] = 0;
          }
        }
        else
        {
          throw_runtime_error("Should not happen");
        }
      }
    }

    // Maps local coordinate s to the opposite side's local coordinate. Three cases: (1) a partial
    // internal-facet overlap, where s is linearly re-parametrized between the pre-computed opposite
    // endpoint coordinates; (2) opposite side is a T-element (simplex local coordinate range
    // [0,1] rather than [-1,1]), rescaling and optionally flipping s accordingly; (3) opposite side
    // is the same Q-family type (range [-1,1]), where only a sign flip is needed for the swapped orientation.
    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const override
    {
      if (partial_opposite_internal_facet)
      {
        double srel = (s[0] - this->s_min()) / (this->s_max() - this->s_min());
        srel = partial_opposite_s_at_smin + (partial_opposite_s_at_smax - partial_opposite_s_at_smin) * srel;
        return oomph::Vector<double>(1, srel);
      }
      else if (dynamic_cast<BulkTElementLine1dC1 *>(opposite_side) || dynamic_cast<BulkTElementLine1dC2 *>(opposite_side))
      {
        if (opposite_orientation)
        {
          oomph::Vector<double> res = s;
          res[0] = (1 - res[0]) * 0.5;
          //          std::cout << "INFO OPPOSITE " << this->interpolated_x(s,0) << " vs " << opposite_side->interpolated_x(res,0) << "  s  " << s[0] << " vs " << res[0] <<  std::endl;
          return res;
        }
        else
        {
          oomph::Vector<double> res = s;
          res[0] = (res[0] + 1) * 0.5;
          //          std::cout << "INFO NONOPPOSITE " << this->interpolated_x(s,0) << " vs " << opposite_side->interpolated_x(res,0) <<  "  s  " << s[0] << " vs " << res[0] <<std::endl;
          return res;
        }
      }
      else if (dynamic_cast<BulkElementLine1dC1 *>(opposite_side) || dynamic_cast<BulkElementLine1dC2 *>(opposite_side))
      {
        if (opposite_orientation)
        {
          oomph::Vector<double> res = s;
          res[0] = -res[0];
          return res;
        }
        else
        {
          return s;
        }
      }
      else
      {
        throw_runtime_error("TODO");
      }
    }
  };

  // Line interface element on a Q-family bulk element's C2 face.
  class InterfaceElementLine1dC2 : public InterfaceElement<BulkElementLine1dC2>
  {
  protected:
    bool partial_opposite_internal_facet;
    double partial_opposite_s_at_smin, partial_opposite_s_at_smax;

  public:
    InterfaceElementLine1dC2(DynamicBulkElementInstance *jitcode, FiniteElement *const &bulk_el_pt, const int &face_index) : InterfaceElement<BulkElementLine1dC2>(jitcode, bulk_el_pt, face_index), partial_opposite_internal_facet(false)
    {
    }

    /*   inline void assign_nodal_local_eqn_numbers(const bool &store_local_dof_pt)
      {
       oomph::SolidFiniteElement::assign_nodal_local_eqn_numbers(store_local_dof_pt);
    //   assign_hanging_local_eqn_numbers(store_local_dof_pt);
    //	 fill_element_info();
      }*/

    pyoomph::Node *opposite_node_pt(unsigned int i) override
    {
      if (partial_opposite_internal_facet)
        throw_runtime_error("opposite_node_pt not allowed in internal facets with partial overlap with the opposite side");
      return InterfaceElement<BulkElementLine1dC2>::opposite_node_pt(i);
    }

    //  void further_setup_hanging_nodes() {} //TODO: REM
    void analyze_opposite_orientation(const std::vector<double> & offset) override
    {
      if (opposite_side->dim() != 1)
      {
        throw_runtime_error("Can only connect a 1d InterfaceElement to a 1d InterfaceElement");
      }
      if (this->nvertex_node() != opposite_side->nvertex_node())
      {
        throw_runtime_error("Can only connect InterfaceElements with same number of vertex nodes");
      }

      double dist0 = 0.0;
      double dist1 = 0.0;
      pyoomph::Node *nopp0 = dynamic_cast<pyoomph::Node *>(opposite_side->vertex_node_pt(0));
      pyoomph::Node *nopp1 = dynamic_cast<pyoomph::Node *>(opposite_side->vertex_node_pt(1));
      pyoomph::Node *nthis0 = dynamic_cast<pyoomph::Node *>(this->vertex_node_pt(0));
      pyoomph::Node *nthis1 = dynamic_cast<pyoomph::Node *>(this->vertex_node_pt(1));            
      for (unsigned int k = 0; k < std::min(nthis0->ndim(), nopp0->ndim()); k++)
        dist0 += (nthis0->x(k) - nopp0->x(k)+offset[k]) * (nthis0->x(k) - nopp0->x(k)+offset[k]);
      for (unsigned int k = 0; k < std::min(nthis0->ndim(), nopp0->ndim()); k++)
        dist0 += (nthis1->x(k) - nopp1->x(k)+offset[k]) * (nthis1->x(k) - nopp1->x(k)+offset[k]);
      for (unsigned int k = 0; k < std::min(nthis0->ndim(), nopp0->ndim()); k++)
        dist1 += (nthis1->x(k) - nopp0->x(k)+offset[k]) * (nthis1->x(k) - nopp0->x(k)+offset[k]);
      for (unsigned int k = 0; k < std::min(nthis0->ndim(), nopp0->ndim()); k++)
        dist1 += (nthis0->x(k) - nopp1->x(k)+offset[k]) * (nthis0->x(k) - nopp1->x(k)+offset[k]);
      opposite_orientation = (dist0 < dist1 ? 0 : 1);
      if ((dist0 < dist1 ? dist0 : dist1) > 1e-14)
      {
        if (!opposite_side->is_internal_facet_opposite_dummy())
        {
          throw_runtime_error("Vertex nodes are not matching here. This is only allowed for internal facets");
        }
        partial_opposite_internal_facet = true;
        oomph::Vector<double> x_at_smin(this->nodal_dimension(), 0.0), x_at_smax(this->nodal_dimension(), 0.0);
        this->interpolated_x(oomph::Vector<double>(1, this->s_min()), x_at_smin);
        this->interpolated_x(oomph::Vector<double>(1, this->s_max()), x_at_smax);
        partial_opposite_s_at_smin = opposite_side->optimize_s_to_match_x(x_at_smin)[0];
        partial_opposite_s_at_smax = opposite_side->optimize_s_to_match_x(x_at_smax)[0];
      }
      opposite_node_index.resize(3);
      if (opposite_side->nnode() == 3)
      {
        if (!opposite_orientation)
        {
          opposite_node_index[0] = 0;
          opposite_node_index[1] = 1;
          opposite_node_index[2] = 2;
        }
        else
        {
          opposite_node_index[0] = 2;
          opposite_node_index[1] = 1;
          opposite_node_index[2] = 0;
        }
      }
      else if (opposite_side->nnode() == 2)
      {
        if (!opposite_orientation)
        {
          opposite_node_index[0] = 0;
          opposite_node_index[1] = -1;
          opposite_node_index[2] = 1;
        }
        else
        {
          opposite_node_index[0] = 1;
          opposite_node_index[1] = -1;
          opposite_node_index[2] = 0;
        }
      }
      else
      {
        throw_runtime_error("Should not happen");
      }
      //    std::cout << "DISTS ARE " << dist0 << "  " << dist1 << " OPP ORIENT " << opposite_orientation << std::endl;
    }

    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const override
    {
      if (partial_opposite_internal_facet)
      {
        double srel = (s[0] - this->s_min()) / (this->s_max() - this->s_min());
        srel = partial_opposite_s_at_smin + (partial_opposite_s_at_smax - partial_opposite_s_at_smin) * srel;
        return oomph::Vector<double>(1, srel);
      }
      else if (dynamic_cast<BulkTElementLine1dC1 *>(opposite_side) || dynamic_cast<BulkTElementLine1dC2 *>(opposite_side))
      {
        if (opposite_orientation)
        {
          oomph::Vector<double> res = s;
          res[0] = (1 - res[0]) * 0.5;
          return res;
        }
        else
        {
          oomph::Vector<double> res = s;
          res[0] = (res[0] + 1) * 0.5;
          return res;
        }
      }
      else if (dynamic_cast<BulkElementLine1dC1 *>(opposite_side) || dynamic_cast<BulkElementLine1dC2 *>(opposite_side))
      {
        if (opposite_orientation)
        {
          oomph::Vector<double> res = s;
          res[0] = -res[0];
          return res;
        }
        else
        {
          return s;
        }
      }
      else
      {
        throw_runtime_error("TODO");
      }
    }
  };

  // Line interface element on a T-family (triangular/tetrahedral) bulk element's C1 face.
  class InterfaceTElementLine1dC1 : public InterfaceElement<BulkTElementLine1dC1>
  {
  protected:
  public:
    InterfaceTElementLine1dC1(DynamicBulkElementInstance *jitcode, FiniteElement *const &bulk_el_pt, const int &face_index) : InterfaceElement<BulkTElementLine1dC1>(jitcode, bulk_el_pt, face_index)
    {
    }
    void analyze_opposite_orientation(const std::vector<double> & offset) override
    {
      if (opposite_side->dim() != 1)
      {
        throw_runtime_error("Can only connect a 1d InterfaceElement to a 1d InterfaceElement");
      }
      if (this->nvertex_node() != opposite_side->nvertex_node())
      {
        throw_runtime_error("Can only connect InterfaceElements with same number of vertex nodes");
      }

      double dist0 = 0.0;
      double dist1 = 0.0;
      for (unsigned int i = 0; i < this->nvertex_node(); i++)
      {
        pyoomph::Node *nthis = dynamic_cast<pyoomph::Node *>(this->vertex_node_pt(i));
        /*        for (unsigned int j = 0; j < opposite_side->nvertex_node(); j++)
                {*/
        pyoomph::Node *nopp = dynamic_cast<pyoomph::Node *>(opposite_side->vertex_node_pt(i));
        for (unsigned int k = 0; k < std::min(nthis->ndim(), nopp->ndim()); k++)
          dist0 += (nthis->x(k) - nopp->x(k)+offset[k]) * (nthis->x(k) - nopp->x(k)+offset[k]);
        nopp = dynamic_cast<pyoomph::Node *>(opposite_side->vertex_node_pt(1 - i));
        for (unsigned int k = 0; k < std::min(nthis->ndim(), nopp->ndim()); k++)
          dist1 += (nthis->x(k) - nopp->x(k)+offset[k]) * (nthis->x(k) - nopp->x(k)+offset[k]);
        //        }
      }
      if ((dist0 < dist1 ? dist0 : dist1) > 1e-14)
      {
        throw_runtime_error("Vertex nodes are not matching here");
      }
      opposite_orientation = (dist0 < dist1 ? 0 : 1);
      //      std::cout << "DISTS " << dist0 << "  " << dist1 << std::endl;
      opposite_node_index.resize(2);

      if (opposite_side->nnode() == 2)
      {
        if (!opposite_orientation)
        {
          opposite_node_index[0] = 0;
          opposite_node_index[1] = 1;
        }
        else
        {
          if (dynamic_cast<BulkElementLine1dC1 *>(opposite_side) || dynamic_cast<BulkElementLine1dC2 *>(opposite_side))
          {
            opposite_node_index[0] = 0;
            opposite_node_index[1] = 1;
          }
          else
          {
            opposite_node_index[0] = 1;
            opposite_node_index[1] = 0;
          }
        }
      }
      else if (opposite_side->nnode() == 3)
      {
        if (!opposite_orientation)
        {
          opposite_node_index[0] = 0;
          opposite_node_index[1] = 2;
        }
        else
        {
          opposite_node_index[0] = 2;
          opposite_node_index[1] = 0;
        }
      }
      else
      {
        throw_runtime_error("Should not happen");
      }
    }

    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const override
    {
      if (dynamic_cast<BulkTElementLine1dC1 *>(opposite_side) || dynamic_cast<BulkTElementLine1dC2 *>(opposite_side))
      {
        //        std::cout << "LC IN OPP " << s[0] << " : " << opposite_orientation << std::endl;
        if (opposite_orientation)
        {
          oomph::Vector<double> res = s;
          res[0] = 1 - res[0];
          return res;
        }
        else
        {
          return s;
        }
      }
      else if (dynamic_cast<BulkElementLine1dC1 *>(opposite_side) || dynamic_cast<BulkElementLine1dC2 *>(opposite_side))
      {
        if (opposite_orientation)
        {
          oomph::Vector<double> res = s;
          res[0] = 2 * (res[0] - 0.5);

          oomph::Vector<double> mycoord(2, 0);
          oomph::Vector<double> ocoord(2, 0);
          this->interpolated_x(s, mycoord);
          opposite_side->interpolated_x(res, ocoord);
          //   std::cout << "S CALC : " << s[0] << " " << res[0] << "  COORDS " << mycoord[0] << " , " << ocoord[0] << "    " << mycoord[1] << " , " << ocoord[1] <<std::endl;

          return res;
        }
        else
        {
          oomph::Vector<double> res = s;
          res[0] = -2 * (res[0] - 0.5);
          return res;
        }
      }
      else
      {
        throw_runtime_error("TODO");
      }
    }
  };

  // Line interface element on a T-family bulk element's C2 face.
  class InterfaceTElementLine1dC2 : public InterfaceElement<BulkTElementLine1dC2>
  {
  protected:
  public:
    InterfaceTElementLine1dC2(DynamicBulkElementInstance *jitcode, FiniteElement *const &bulk_el_pt, const int &face_index) : InterfaceElement<BulkTElementLine1dC2>(jitcode, bulk_el_pt, face_index)
    {
    }

    /*   inline void assign_nodal_local_eqn_numbers(const bool &store_local_dof_pt)
      {
       oomph::SolidFiniteElement::assign_nodal_local_eqn_numbers(store_local_dof_pt);
    //   assign_hanging_local_eqn_numbers(store_local_dof_pt);
    //	 fill_element_info();
      }*/

    //  void further_setup_hanging_nodes() {} //TODO: REM
    void analyze_opposite_orientation(const std::vector<double> & offset) override
    {
      if (opposite_side->dim() != 1)
      {
        throw_runtime_error("Can only connect a 1d InterfaceElement to a 1d InterfaceElement");
      }
      if (this->nvertex_node() != opposite_side->nvertex_node())
      {
        throw_runtime_error("Can only connect InterfaceElements with same number of vertex nodes");
      }

      double dist0 = 0.0;
      double dist1 = 0.0;
      pyoomph::Node *nopp0 = dynamic_cast<pyoomph::Node *>(opposite_side->vertex_node_pt(0));
      pyoomph::Node *nopp1 = dynamic_cast<pyoomph::Node *>(opposite_side->vertex_node_pt(1));
      pyoomph::Node *nthis0 = dynamic_cast<pyoomph::Node *>(this->vertex_node_pt(0));
      pyoomph::Node *nthis1 = dynamic_cast<pyoomph::Node *>(this->vertex_node_pt(1));
      for (unsigned int k = 0; k < std::min(nthis0->ndim(), nopp0->ndim()); k++)
        dist0 += (nthis0->x(k) - nopp0->x(k)+offset[k]) * (nthis0->x(k) - nopp0->x(k)+offset[k]);
      for (unsigned int k = 0; k < std::min(nthis0->ndim(), nopp0->ndim()); k++)
        dist0 += (nthis1->x(k) - nopp1->x(k)+offset[k]) * (nthis1->x(k) - nopp1->x(k)+offset[k]);
      for (unsigned int k = 0; k < std::min(nthis0->ndim(), nopp0->ndim()); k++)
        dist1 += (nthis1->x(k) - nopp0->x(k)+offset[k]) * (nthis1->x(k) - nopp0->x(k)+offset[k]);
      for (unsigned int k = 0; k < std::min(nthis0->ndim(), nopp0->ndim()); k++)
        dist1 += (nthis0->x(k) - nopp1->x(k)+offset[k]) * (nthis0->x(k) - nopp1->x(k)+offset[k]);
      opposite_orientation = (dist0 < dist1 ? 0 : 1);
      if ((dist0 < dist1 ? dist0 : dist1) > 1e-14)
      {
        throw_runtime_error("Vertex nodes are not matching here");
      }
      opposite_node_index.resize(3);
      if (opposite_side->nnode() == 3)
      {
        if (!opposite_orientation)
        {
          opposite_node_index[0] = 0;
          opposite_node_index[1] = 1;
          opposite_node_index[2] = 2;
        }
        else
        {
          opposite_node_index[0] = 2;
          opposite_node_index[1] = 1;
          opposite_node_index[2] = 0;
        }
      }
      else if (opposite_side->nnode() == 2)
      {
        if (!opposite_orientation)
        {
          opposite_node_index[0] = 0;
          opposite_node_index[1] = -1;
          opposite_node_index[2] = 1;
        }
        else
        {
          opposite_node_index[0] = 1;
          opposite_node_index[1] = -1;
          opposite_node_index[2] = 0;
        }
      }
      else
      {
        throw_runtime_error("Should not happen");
      }
      //    std::cout << "DISTS ARE " << dist0 << "  " << dist1 << " OPP ORIENT " << opposite_orientation << std::endl;
    }

    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const override
    {
      if (dynamic_cast<BulkTElementLine1dC1 *>(opposite_side) || dynamic_cast<BulkTElementLine1dC2 *>(opposite_side))
      {
        //        std::cout << "LC IN OPP " << s[0] << " : " << opposite_orientation << std::endl;
        if (opposite_orientation)
        {
          oomph::Vector<double> res = s;
          res[0] = 1 - res[0];
          return res;
        }
        else
        {
          return s;
        }
      }
      else if (dynamic_cast<BulkElementLine1dC1 *>(opposite_side) || dynamic_cast<BulkElementLine1dC2 *>(opposite_side))
      {
        if (opposite_orientation)
        {
          oomph::Vector<double> res = s;
          res[0] = -2 * (res[0] - 0.5);
          return res;
        }
        else
        {
          oomph::Vector<double> res = s;
          res[0] = 2 * (res[0] - 0.5);
          return res;
        }
      }
      else
      {
        throw_runtime_error("TODO");
      }
    }
  };

  // Opposite-side matching for QUADRILATERAL face elements (a brick's face, a wedge's side, a
  // pyramid's base). The analogue of the 6 vertex permutations InterfaceElementTri2d* enumerates,
  // but for the 8 symmetries of the square. Shared by the C1 (2x2 nodes) and C2 (3x3 nodes)
  // variants, and derived rather than tabulated: a quad face element is a tensor-product element,
  // so local node i+nnode_1d*j sits at s=(-1+2i/(nnode_1d-1), -1+2j/(nnode_1d-1)) and the node
  // correspondence follows from the coordinate map alone. That is what keeps the C2 case honest -
  // the 6-node-triangle counterpart above needed hand-determined special cases.
  struct Quad2dFaceOrientation
  {
    // Local coordinate on the opposite side under symmetry `orientation` (0..7).
    static oomph::Vector<double> map_s(int orientation, const oomph::Vector<double> &s);
    // Local node index on the opposite side for each local node of a nnode_1d x nnode_1d face.
    // For nnode_1d==2 this is exactly the vertex permutation of the symmetry.
    static std::vector<int> node_index_map(int orientation, unsigned nnode_1d);
    // Picks the symmetry whose vertex correspondence matches the two faces geometrically (with the
    // periodic `offset` applied) and fills orientation/node_index accordingly.
    static void analyze(const oomph::FiniteElement *self, const oomph::FiniteElement *opposite, const std::vector<double> &offset, unsigned nnode_1d, int &orientation, std::vector<int> &node_index);
  };

  // Quadrilateral (2d) interface element on a brick bulk element's C1 face.
  class InterfaceElementQuad2dC1 : public InterfaceElement<BulkElementQuad2dC1>
  {
  protected:
  //  std::map<Node*, int>* add_interf_local_hang_eqs_C1, *add_interf_local_hang_eqs_C1TB;
  public:
    InterfaceElementQuad2dC1(DynamicBulkElementInstance *jitcode, FiniteElement *const &bulk_el_pt, const int &face_index) : InterfaceElement<BulkElementQuad2dC1>(jitcode, bulk_el_pt, face_index)//, add_interf_local_hang_eqs_C1(NULL), add_interf_local_hang_eqs_C1TB(NULL)
    {
    }
    //void assign_hanging_additional_interface_local_equations(const bool &store_local_dof_pt) override;
    /*~InterfaceElementQuad2dC1() override
    {
      if (add_interf_local_hang_eqs_C1)
        delete[] add_interf_local_hang_eqs_C1;
      if (add_interf_local_hang_eqs_C1TB)
        delete[] add_interf_local_hang_eqs_C1TB;
    }*/

    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const override
    {
      return Quad2dFaceOrientation::map_s(opposite_orientation, s);
    }

    void analyze_opposite_orientation(const std::vector<double> & offset) override
    {
      Quad2dFaceOrientation::analyze(this, opposite_side, offset, 2, opposite_orientation, opposite_node_index);
    }

  };

  // Quadrilateral interface element on a brick bulk element's C2 face.
  class InterfaceElementQuad2dC2 : public InterfaceElement<BulkElementQuad2dC2>
  {
  protected:
    //std::map<Node*, int>* add_interf_local_hang_eqs_C1, *add_interf_local_hang_eqs_C1TB,* add_interf_local_hang_eqs_C2, *add_interf_local_hang_eqs_C2TB;
  public:
    InterfaceElementQuad2dC2(DynamicBulkElementInstance *jitcode, FiniteElement *const &bulk_el_pt, const int &face_index) : InterfaceElement<BulkElementQuad2dC2>(jitcode, bulk_el_pt, face_index)//, add_interf_local_hang_eqs_C1(NULL), add_interf_local_hang_eqs_C1TB(NULL), add_interf_local_hang_eqs_C2(NULL), add_interf_local_hang_eqs_C2TB(NULL)
    {
    }
    
    /*~InterfaceElementQuad2dC2() override
    {
      if (add_interf_local_hang_eqs_C1)
        delete[] add_interf_local_hang_eqs_C1;
      if (add_interf_local_hang_eqs_C1TB)
        delete[] add_interf_local_hang_eqs_C1TB;
      if (add_interf_local_hang_eqs_C2)
        delete[] add_interf_local_hang_eqs_C2;
      if (add_interf_local_hang_eqs_C2TB)
        delete[] add_interf_local_hang_eqs_C2TB;
    }*/

    //void assign_hanging_additional_interface_local_equations(const bool &store_local_dof_pt) ;

    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const override
    {
      return Quad2dFaceOrientation::map_s(opposite_orientation, s);
    }

    void analyze_opposite_orientation(const std::vector<double> & offset) override
    {
      Quad2dFaceOrientation::analyze(this, opposite_side, offset, 3, opposite_orientation, opposite_node_index);
    }

  };

  // Triangular (2d) interface element on a tetrahedral bulk element's C1 face. Unlike the 1d line
  // interface elements above (which only have 2 possible orientations), a triangular face has 6
  // possible vertex permutations (3 rotations x 2 reflections); opposite_orientation therefore
  // indexes into the fixed permutation list "perms" below rather than being a plain 0/1 flag.
  class InterfaceElementTri2dC1 : public InterfaceElement<BulkElementTri2dC1>
  {
  protected:
  public:
    InterfaceElementTri2dC1(DynamicBulkElementInstance *jitcode, FiniteElement *const &bulk_el_pt, const int &face_index) : InterfaceElement<BulkElementTri2dC1>(jitcode, bulk_el_pt, face_index)
    {
    }

    // Applies the vertex permutation "opposite_orientation" (chosen by analyze_opposite_orientation
    // below) to the barycentric-style local coordinate s to obtain the corresponding coordinate on
    // the opposite side.
    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const override
    {
      oomph::Vector<double> res = s;
      if (opposite_orientation == 0)
      {
        res[0] = s[0];
        res[1] = s[1];
      }
      else if (opposite_orientation == 1)
      {
        res[0] = s[0];
        res[1] = 1 - s[0] - s[1];
      }
      else if (opposite_orientation == 2)
      {
        res[0] = s[1];
        res[1] = s[0];
      }
      else if (opposite_orientation == 3)
      {
        res[0] = 1 - s[0] - s[1];
        res[1] = s[0];
      }
      else if (opposite_orientation == 4)
      {
        res[0] = s[1];
        res[1] = 1 - s[0] - s[1];
      }
      else
      {
        res[0] = 1 - s[0] - s[1];
        res[1] = s[1];
      }
      return res;
    }

    // Tries all 6 vertex permutations ("perms") of the opposite side's vertex nodes against this
    // element's own, computing the total squared coordinate distance for each (with the periodic
    // "offset" applied), and picks the permutation with the smallest distance as opposite_orientation.
    void analyze_opposite_orientation(const std::vector<double> & offset) override
    {
      if (opposite_side->dim() != 2)
      {
        throw_runtime_error("Can only connect a 2d InterfaceElement to a 2d InterfaceElement");
      }
      if (this->nvertex_node() != opposite_side->nvertex_node())
      {
        throw_runtime_error("Can only connect InterfaceElements with same number of vertex nodes");
      }
      std::vector<std::vector<int>> perms = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
      std::vector<double> pdists(perms.size(), 0.0);
      for (unsigned int i = 0; i < this->nvertex_node(); i++)
      {
        pyoomph::Node *nthis = dynamic_cast<pyoomph::Node *>(this->vertex_node_pt(i));
        for (unsigned int p = 0; p < perms.size(); p++)
        {
          pyoomph::Node *nopp = dynamic_cast<pyoomph::Node *>(opposite_side->vertex_node_pt(perms[p][i]));
          for (unsigned int k = 0; k < std::min(nthis->ndim(), nopp->ndim()); k++)
            pdists[p] += (nthis->x(k) - nopp->x(k)+offset[k]) * (nthis->x(k) - nopp->x(k)+offset[k]);
        }
      }
      double best_dist = pdists[0];
      opposite_orientation = 0;
      for (unsigned int p = 1; p < perms.size(); p++)
      {
        if (pdists[p] < best_dist)
        {
          best_dist = pdists[p];
          opposite_orientation = p;
        }
      }
      if (best_dist > 1e-14)
      {
        throw_runtime_error("Vertex nodes are not matching here");
      }
      opposite_node_index = perms[opposite_orientation]; // Making use of the fact that also for C2 opposite elements, the vertex nodes are at 0,1,2
    }
  };

  // Triangular interface element on a tetrahedral bulk element's C2 face; also fills in the
  // opposite-side indices of the 3 edge-midside nodes (indices 3-5) once the vertex permutation is known.
  class InterfaceElementTri2dC2 : public InterfaceElement<BulkElementTri2dC2>
  {
  protected:
  public:
    InterfaceElementTri2dC2(DynamicBulkElementInstance *jitcode, FiniteElement *const &bulk_el_pt, const int &face_index) : InterfaceElement<BulkElementTri2dC2>(jitcode, bulk_el_pt, face_index)
    {
    }

    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const override
    {
      oomph::Vector<double> res = s;
      if (opposite_orientation == 0)
      {
        res[0] = s[0];
        res[1] = s[1];
      }
      else if (opposite_orientation == 1)
      {
        res[0] = s[0];
        res[1] = 1 - s[0] - s[1];
      }
      else if (opposite_orientation == 2)
      {
        res[0] = s[1];
        res[1] = s[0];
      }
      else if (opposite_orientation == 3)
      {
        res[0] = 1 - s[0] - s[1];
        res[1] = s[0];
      }
      else if (opposite_orientation == 4)
      {
        res[0] = s[1];
        res[1] = 1 - s[0] - s[1];
      }
      else
      {
        res[0] = 1 - s[0] - s[1];
        res[1] = s[1];
      }
      return res;
     
    }

    // Same vertex-permutation matching as InterfaceElementTri2dC1::analyze_opposite_orientation,
    // then additionally derives the opposite-side indices of the 3 mid-edge nodes (local indices
    // 3-5) from the chosen vertex permutation, based on oomph-lib's fixed edge-to-midnode numbering
    // convention for 6-node triangles (the explicit per-permutation cases below were determined by
    // matching that convention).
    void analyze_opposite_orientation(const std::vector<double> & offset) override
    {
      if (opposite_side->dim() != 2)
      {
        throw_runtime_error("Can only connect a 2d InterfaceElement to a 2d InterfaceElement");
      }
      if (this->nvertex_node() != opposite_side->nvertex_node())
      {
        throw_runtime_error("Can only connect InterfaceElements with same number of vertex nodes");
      }
      std::vector<std::vector<int>> perms = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
      std::vector<double> pdists(perms.size(), 0.0);
      for (unsigned int i = 0; i < this->nvertex_node(); i++)
      {
        pyoomph::Node *nthis = dynamic_cast<pyoomph::Node *>(this->vertex_node_pt(i));
        for (unsigned int p = 0; p < perms.size(); p++)
        {
          pyoomph::Node *nopp = dynamic_cast<pyoomph::Node *>(opposite_side->vertex_node_pt(perms[p][i]));
          for (unsigned int k = 0; k < std::min(nthis->ndim(), nopp->ndim()); k++)
            pdists[p] += (nthis->x(k) - nopp->x(k)+offset[k]) * (nthis->x(k) - nopp->x(k)+offset[k]);
        }
      }
      double best_dist = pdists[0];
      opposite_orientation = 0;
      for (unsigned int p = 1; p < perms.size(); p++)
      {
        if (pdists[p] < best_dist)
        {
          best_dist = pdists[p];
          opposite_orientation = p;
        }
      }
      if (best_dist > 1e-14)
      {
        throw_runtime_error("Vertex nodes are not matching here");
      }
      opposite_node_index = perms[opposite_orientation];
      opposite_node_index.resize(6, -1);
      if (opposite_side->nnode() > 3)
      {
        if (opposite_orientation == 1)
        { // 3 5 4
          opposite_node_index[3] = 5;
          opposite_node_index[4] = 4;
          opposite_node_index[5] = 3;
        }
        else if (opposite_orientation == 2)
        { // 4 3 5
          opposite_node_index[3] = 3;
          opposite_node_index[4] = 5;
          opposite_node_index[5] = 4;
        }
        else if (opposite_orientation == 5)
        { // 5 4 3, 4 5 3, 3 5 4, 5 3 4,
          opposite_node_index[3] = 4;
          opposite_node_index[4] = 3;
          opposite_node_index[5] = 5;
        }
        else
        {
          for (unsigned int k = 3; k < 6; k++)
          {
            opposite_node_index[k] = opposite_node_index[k - 3] + 3; // Seem to work
          }
        }
      }
    }
  };



  // Triangular interface element on a tetrahedral bulk element's C2TB (bubble-enriched) face;
  // additionally maps the single interior bubble node (local index 6) directly to index 6 on the
  // opposite side, since that node is always numbered last irrespective of orientation.
  class InterfaceElementTri2dC2TB : public InterfaceElement<BulkElementTri2dC2TB>
  {
  protected:
  public:
    InterfaceElementTri2dC2TB(DynamicBulkElementInstance *jitcode, FiniteElement *const &bulk_el_pt, const int &face_index) : InterfaceElement<BulkElementTri2dC2TB>(jitcode, bulk_el_pt, face_index)
    {
    }

    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const override
    {
      oomph::Vector<double> res = s;
      if (opposite_orientation == 0)
      {
        res[0] = s[0];
        res[1] = s[1];
      }
      else if (opposite_orientation == 1)
      {
        res[0] = s[0];
        res[1] = 1 - s[0] - s[1];
      }
      else if (opposite_orientation == 2)
      {
        res[0] = s[1];
        res[1] = s[0];
      }
      else if (opposite_orientation == 3)
      {
        res[0] = 1 - s[0] - s[1];
        res[1] = s[0];
      }
      else if (opposite_orientation == 4)
      {
        res[0] = s[1];
        res[1] = 1 - s[0] - s[1];
      }
      else
      {
        res[0] = 1 - s[0] - s[1];
        res[1] = s[1];
      }
      return res;
     
    }

    void analyze_opposite_orientation(const std::vector<double> & offset) override
    {
      if (opposite_side->dim() != 2)
      {
        throw_runtime_error("Can only connect a 2d InterfaceElement to a 2d InterfaceElement");
      }
      if (this->nvertex_node() != opposite_side->nvertex_node())
      {
        throw_runtime_error("Can only connect InterfaceElements with same number of vertex nodes");
      }
      std::vector<std::vector<int>> perms = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
      std::vector<double> pdists(perms.size(), 0.0);
      for (unsigned int i = 0; i < this->nvertex_node(); i++)
      {
        pyoomph::Node *nthis = dynamic_cast<pyoomph::Node *>(this->vertex_node_pt(i));
        for (unsigned int p = 0; p < perms.size(); p++)
        {
          pyoomph::Node *nopp = dynamic_cast<pyoomph::Node *>(opposite_side->vertex_node_pt(perms[p][i]));
          for (unsigned int k = 0; k < std::min(nthis->ndim(), nopp->ndim()); k++)
            pdists[p] += (nthis->x(k) - nopp->x(k)+offset[k]) * (nthis->x(k) - nopp->x(k)+offset[k]);
        }
      }
      double best_dist = pdists[0];
      opposite_orientation = 0;
      for (unsigned int p = 1; p < perms.size(); p++)
      {
        if (pdists[p] < best_dist)
        {
          best_dist = pdists[p];
          opposite_orientation = p;
        }
      }
      if (best_dist > 1e-14)
      {
        throw_runtime_error("Vertex nodes are not matching here");
      }
      opposite_node_index = perms[opposite_orientation];
      opposite_node_index.resize(7, -1);      
      if (opposite_side->nnode() > 3)
      {
        if (opposite_side->nnode() > 6)
        {
          opposite_node_index[6] = 6; // The center node is always 6 in the opposite element, so we can directly set it here
        }
        if (opposite_orientation == 1)
        { // 3 5 4
          opposite_node_index[3] = 5;
          opposite_node_index[4] = 4;
          opposite_node_index[5] = 3;          
        }
        else if (opposite_orientation == 2)
        { // 4 3 5
          opposite_node_index[3] = 3;
          opposite_node_index[4] = 5;
          opposite_node_index[5] = 4;
        }
        else if (opposite_orientation == 5)
        { // 5 4 3, 4 5 3, 3 5 4, 5 3 4,
          opposite_node_index[3] = 4;
          opposite_node_index[4] = 3;
          opposite_node_index[5] = 5;
        }
        else
        {
          for (unsigned int k = 3; k < 6; k++)
          {
            opposite_node_index[k] = opposite_node_index[k - 3] + 3; // Seem to work
          }
        }
      }
    }
  };
}
