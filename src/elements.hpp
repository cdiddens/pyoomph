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


#pragma once
#include "exception.hpp"
#include "jitbridge.h"

#include "oomph_lib.hpp"

#include "refineable_brick_element.h"

#include "refineable_telements.hpp"
#include "wedges_and_pyramids.hpp"
#include "problem.hpp"

#include "mesh_as_geometric_object.h"

// #include "meshtemplate.hpp"

extern "C"
{
  double _pyoomph_get_element_size(void *);
  double _pyoomph_invoke_callback(void *, int, double *, int);
  void _pyoomph_invoke_multi_ret(void *, int, int, double *, double *, double *, int, int); // Index, flag,args,returns,derivative matrix, nargs,nret
  void _pyoomph_fill_shape_buffer_for_point(unsigned, JITFuncSpec_RequiredShapes_FiniteElement_t *, int);
}

namespace pyoomph
{

  // Required for the Hessian nodal derivatives of second order
  // Dense rank-6 tensor with flat storage (row-major, index n6 varies fastest), used to hold
  // d^2(nodal position)/d(dof_i)d(dof_j) type second derivatives needed when assembling the
  // Hessian (second derivative) contributions of the generated residuals w.r.t. two degrees of freedom.
  class RankSixTensor
  {
  protected:
    unsigned int n1, n2, n3, n4, n5, n6;
    std::vector<double> data;

  public:
    // Allocates storage for an n1 x n2 x n3 x n4 x n5 x n6 tensor, zero-initialized.
    RankSixTensor(unsigned int _n1, unsigned int _n2, unsigned int _n3, unsigned int _n4, unsigned int _n5, unsigned int _n6) : n1(_n1), n2(_n2), n3(_n3), n4(_n4), n5(_n5), n6(_n6), data(_n1 * _n2 * _n3 * _n4 * _n5 * _n6) {}

    // Element access (read/write) at the given 6 indices, using the flattened row-major layout.
    inline double &operator()(const unsigned long &i, const unsigned long &j, const unsigned long &k, const unsigned long &l, const unsigned long &m, const unsigned long &n)
    {
      return data[n6 * (n5 * (n4 * (n3 * (n2 * i + j) + k) + l) + m) + n];
    }

    // Element access (read-only) at the given 6 indices.
    inline double operator()(const unsigned long &i, const unsigned long &j, const unsigned long &k, const unsigned long &l, const unsigned long &m, const unsigned long &n) const
    {
      return data[n6 * (n5 * (n4 * (n3 * (n2 * i + j) + k) + l) + m) + n];
    }
  };

  class BulkElementBase;

  // One requested Hessian-vector-product contribution to be evaluated during a combined
  // ("single pass") assembly sweep: Y is the vector to contract the Hessian with, and
  // J_Hessian/M_Hessian are the destination matrices for the Jacobian- and mass-matrix-Hessian
  // contractions respectively (either may be NULL if not required). "transposed" selects whether
  // the contraction is done with the transposed Hessian (relevant since the Hessian is symmetric
  // only in certain index pairs).
  class SinglePassMultiAssembleHessianInfo
  {
  public:
    oomph::Vector<double> &Y;
    oomph::DenseMatrix<double> *M_Hessian;
    oomph::DenseMatrix<double> *J_Hessian;
    bool transposed;
    SinglePassMultiAssembleHessianInfo(oomph::Vector<double> &_Y, oomph::DenseMatrix<double> *J, oomph::DenseMatrix<double> *M, bool _transposed=false) : Y(_Y), M_Hessian(M), J_Hessian(J), transposed(_transposed) {}
  };

  // One requested parameter-derivative contribution to be evaluated during a combined assembly
  // sweep: holds the pointer to the parameter being differentiated with respect to, and the
  // destination residual/Jacobian/mass-matrix derivative buffers (Jacobian/mass may be NULL).
  class SinglePassMultiAssembleDParamInfo
  {
  public:
    double *const &parameter;
    oomph::Vector<double> *dRdparam;
    oomph::DenseMatrix<double> *dMdparam;
    oomph::DenseMatrix<double> *dJdparam;
    SinglePassMultiAssembleDParamInfo(double *const &param, oomph::Vector<double> *dres, oomph::DenseMatrix<double> *dJ = NULL, oomph::DenseMatrix<double> *dM = NULL) : parameter(param), dRdparam(dres), dMdparam(dM), dJdparam(dJ) {}
  };

  // Bundles everything needed for one "contribution" (a residual/Jacobian/mass-matrix triple,
  // plus zero or more requested Hessian-vector-products and parameter derivatives) that should be
  // filled in during a single combined assembly loop over an element. This lets several different
  // global assembly requests (e.g. for eigenproblems, bifurcation tracking, or multiple linear
  // systems) share one evaluation of the shape functions and generated residual code per element,
  // instead of looping over the element separately for each. See BulkElementBase::get_multi_assembly.
  class SinglePassMultiAssembleInfo
  {
  protected:
    friend class BulkElementBase;
    std::vector<SinglePassMultiAssembleHessianInfo> hessians;
    std::vector<SinglePassMultiAssembleDParamInfo> dparams;

  public:
    int contribution = 0;
    oomph::Vector<double> *residuals = NULL;
    oomph::DenseMatrix<double> *jacobian = NULL;
    oomph::DenseMatrix<double> *mass_matrix = NULL;

    // Registers an additional Hessian-vector-product to be computed in the same assembly pass.
    void add_hessian(oomph::Vector<double> &_Y, oomph::DenseMatrix<double> *J, oomph::DenseMatrix<double> *M = NULL,bool transposed=false)
    {
      hessians.push_back(SinglePassMultiAssembleHessianInfo(_Y, J, M,transposed));
    }

    SinglePassMultiAssembleHessianInfo & get_hessian(unsigned int i) { return  hessians[i]; }

    // Registers an additional parameter-derivative contribution to be computed in the same assembly pass.
    void add_param_deriv(double *const &param, oomph::Vector<double> *dres, oomph::DenseMatrix<double> *dJ = NULL, oomph::DenseMatrix<double> *dM = NULL)
    {
      dparams.push_back(SinglePassMultiAssembleDParamInfo(param, dres, dJ, dM));
    }
    SinglePassMultiAssembleInfo(int contrib, oomph::Vector<double> *res, oomph::DenseMatrix<double> *J, oomph::DenseMatrix<double> *M = NULL) : contribution(contrib), residuals(res), jacobian(J), mass_matrix(M) {}
  };

  // Empty tag/junction class: gives pyoomph's element hierarchy its own root distinct from plain
  // oomph::GeneralisedElement, so that virtual inheritance below can disambiguate the diamond
  // between oomph-lib's element classes and pyoomph's own mixins.
  class ElementBase : public virtual oomph::GeneralisedElement
  {
  };

  // Another virtual-inheritance junction class: combines oomph-lib's refineable solid element
  // (moving/ALE mesh + h-refinement) and Z2 flux-recovery error estimator interfaces into a single
  // base that all pyoomph finite elements derive from.
  class FiniteElementBase : public virtual ElementBase, public virtual oomph::RefineableSolidElement, public virtual oomph::ElementWithZ2ErrorEstimator
  {
  public:
  };

  /*Meshio type indices
  0 : vertex
  1 : line
  2 : line3
  3 : triangle
  4 : triangle6
  5 : triangle7
  6 : quad
  7 : quad8 (not intended to be implemented)
  8 : quad9

  */

  // Central cache/owner of oomph-lib integration (quadrature) rule objects, keyed by element shape
  // (quad/tri-like Q/T family, per spatial dimension, with or without bubble enrichment) and
  // requested integration order. Elements request their integration scheme via
  // get_integration_scheme() instead of constructing oomph::Integral objects themselves, so that
  // a given (shape, dimension, order) combination is only ever allocated once and shared across all
  // elements of that kind.
  class IntegrationSchemeStorage
  {
  protected:
    std::map<unsigned, oomph::Integral *> Q1d;
    std::map<unsigned, oomph::Integral *> T1d;
    std::map<unsigned, oomph::Integral *> Q2d;
    std::map<unsigned, oomph::Integral *> T2d;
    std::map<unsigned, oomph::Integral *> T2dTB;
    std::map<unsigned, oomph::Integral *> Q3d;
    std::map<unsigned, oomph::Integral *> T3d;
    std::map<unsigned, oomph::Integral *> T3dTB;
    std::map<unsigned, oomph::Integral *> Wedge3d;
    std::map<unsigned, oomph::Integral *> Pyramid3d;
    // Selects which of the per-shape maps above stores integration schemes for the given
    // (triangular/tetrahedral vs. quad/brick, element dimension, bubble-enriched) combination.
    std::map<unsigned, oomph::Integral *> &get_integral_order_map(bool tri, unsigned edim, bool bubble);
    // Deletes all oomph::Integral objects owned by one of the per-shape maps.
    void clean_up_map(std::map<unsigned, oomph::Integral *> &map);

  public:
    IntegrationSchemeStorage();
    virtual ~IntegrationSchemeStorage();
    // Returns the (lazily constructed, cached) integration scheme for the given element shape
    // (tris=true for simplex/T-elements, false for Q-elements), spatial dimension edim, and
    // integration order; bubble selects the enriched scheme variant used for bubble functions.
    oomph::Integral *get_integration_scheme(bool tris, unsigned edim, unsigned order, bool bubble = false);
  };

  extern IntegrationSchemeStorage integration_scheme_storage;

  class MeshTemplate;
  class MeshTemplateElement;
  class DynamicBulkElementInstance;
  class Problem;
  // The central base class for all pyoomph "bulk" finite elements (as opposed to face/interface
  // elements, see InterfaceElementBase below). A concrete element type (e.g. BulkElementTri2dC2)
  // combines this class with the appropriate oomph-lib geometric element (shape functions,
  // refinement rules) via virtual inheritance.
  //
  // BulkElementBase does *not* itself know the governing equations: the actual residuals, Jacobian
  // and (optionally) Hessian and mass matrix are produced by C code that is generated from the
  // user's symbolic (GiNaC) weak-form expressions, compiled at runtime, and reached through the
  // JIT function table stored in the associated DynamicBulkElementInstance (codeinst). This class
  // provides the glue: it evaluates shape functions/derivatives at integration points and fills a
  // JITShapeInfo_t buffer, maps nodal/internal/external data to local equation numbers (including
  // hanging-node constraints from mesh refinement and "dummy" values used for mixed-order
  // interpolation of discontinuous fields), and then calls into the generated code
  // (fill_in_generic_residual_contribution_jit / fill_in_generic_dresidual_contribution_jit /
  // fill_in_generic_hessian) once per integration point to accumulate the element's contribution to
  // the global residual/Jacobian/mass-matrix/Hessian.
  class BulkElementBase : public virtual FiniteElementBase
  {
  protected:
    DynamicBulkElementInstance *codeinst;

    JITElementInfo_t eleminfo;
    JITShapeInfo_t *shape_info;

    // Releases/resets the JITElementInfo_t buffers owned by this element (nodal/external/internal
    // data pointers etc. handed to the generated code).
    void free_element_info();

    // Allocates internal/external Data for fields stored discontinuously per element (D0: constant,
    // DL: discontinuous-Lagrange, DG: discontinuous on a sub-space) rather than as ordinary nodal data.
    virtual void allocate_discontinous_fields();
    // Allocates/sizes the shape-function value/derivative buffers inside shape_info according to
    // which shapes (psi, dpsi, hang info, ...) the generated code actually requires, before the
    // integration loop starts.
    virtual void prepare_shape_buffer_for_integration(const JITFuncSpec_RequiredShapes_FiniteElement_t &required_shapes, unsigned int flag);
    // Fills in element-size-related entries (e.g. element diameter) of the shape buffer that are
    // needed by the generated code but do not depend on the integration point.
    virtual void fill_shape_info_element_sizes(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, unsigned flag) const;
    // Evaluates shape functions, their derivatives, and the (Lagrangian/Eulerian) Jacobian of the
    // geometric mapping at local coordinate s, storing the results into the internal shape_info
    // buffer (the overload below writes into a caller-supplied buffer instead). "index" selects
    // which set of required-shapes/JIT function table entry this call corresponds to (bulk element
    // itself vs. an attached interface, see overrides in InterfaceElementBase). Returns the
    // Eulerian Jacobian determinant of the mapping, used as the integration weight factor.
    virtual double fill_shape_info_at_s(const oomph::Vector<double> &s, const unsigned int &index, const JITFuncSpec_RequiredShapes_FiniteElement_t &required, double &JLagr, unsigned int flag, oomph::DenseMatrix<double> *dxds = NULL, unsigned history_index=0) const;
    // Helper for fill_shape_info_at_s: computes the derivatives of the (mapped) shape functions
    // with respect to the nodal Eulerian positions (dshape/dX), and optionally their second
    // derivatives (D2X2_dshape, a RankSixTensor), which are required for ALE-moving-mesh Jacobian
    // and Hessian contributions.
    virtual void fill_shape_info_at_s_dNodalPos_helper(JITShapeInfo_t *shape_info, const unsigned &index, const oomph::DenseMatrix<double> &interpolated_t, const oomph::DShape &dpsids_Element, const double det_Eulerian, const oomph::DenseMatrix<double> &aup, bool require_hessian, oomph::RankFourTensor<double> &DXdshape_il_jb, RankSixTensor *D2X2_dshape) const;
    // Finite-difference fallback for the Jacobian contribution from the Lagrangian (undeformed
    // solid) position degrees of freedom, used where an analytic derivative is not available.
    virtual void fill_in_jacobian_from_lagragian_by_fd(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian);
    // Computes the derivative of the (interface/boundary) outer unit normal with respect to the
    // nodal coordinates (and optionally its second derivative), required by generated code that
    // differentiates normal-dependent boundary conditions w.r.t. the moving mesh position.
    virtual void get_dnormal_dcoords_at_s(const oomph::Vector<double> &s, double *  PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT dnormal_dcoord, double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT d2normal_dcoord2) const;
    void update_in_solid_position_fd(const unsigned &i) override; // For FD with element_sizes, we have to update the element size buffer
    // Sets up the local-equation-number bookkeeping for hanging nodes on the Lagrangian/Eulerian
    // position degrees of freedom (as opposed to field values), needed on refined (non-conforming)
    // meshes with ALE/solid mechanics where the mesh position itself is a degree of freedom.
    virtual bool fill_hang_info_with_equations_for_pos(JITShapeInfo_t *shape_info);
    // Sets up hanging-node local-equation-number bookkeeping for the "base bulk" fields (i.e. not
    // the additional interface-only fields added by InterfaceElementBase, which extends this).
    virtual bool fill_hang_info_with_equations_basebulk(JITShapeInfo_t *shape_info);
    // Additional interface-only hanging-node bookkeeping, used by InterfaceElementBase to handle
    // the extra fields that exist only on the interface element and not on the bulk element.
    virtual bool fill_hang_info_with_equations_interface(JITShapeInfo_t *shape_info) {return false;}
    static const std::vector<std::vector<std::vector<unsigned>>> Dummy_Value_Interpolation_Map;
  public:
    // Maps a "dummy value" (a value slot that exists only to keep a lower-order field's nodal
    // layout consistent with a higher-order geometric element, e.g. C1 fields living on a subset of
    // a C2 element's nodes) to the real interpolation nodes/weights used to fill it in. Overridden
    // by element types that actually have such dummy values.
    virtual const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const {return Dummy_Value_Interpolation_Map;}
    // Per-concrete-element-type static tables (defined in the .cpp for each Bulk*Element* class)
    // that translate between "nodal space index" (which interpolation space, e.g. C1/C2/C1TB/C2TB,
    // a node belongs to) and the linear "element index" ordering used by the generated code's
    // field/equation numbering.
    virtual const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const=0;
    virtual const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const=0;
    unsigned _numpy_index;
    double initial_cartesian_nondim_size = 0.0;
    double initial_quality_factor = 0.0;
    // Factory for the FaceElement (interface element) attached to a given face/edge of this bulk
    // element; concrete element types override this to return the matching Interface*Element* type.
    virtual oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) {throw_runtime_error(std::string("Specify the face element constructor for the element type ")+typeid(*this).name()); return NULL;}
    virtual const std::vector<int> & get_possible_face_indices() const=0;
    virtual  std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const=0;
    // Evaluates and stores (into the shape_info buffer) the shape function values/derivatives
    // required at one integration point, as requested by "required_shapes" (a bitmask-like struct
    // generated alongside the JIT code, describing exactly which shapes the weak form needs).
    virtual void fill_shape_buffer_for_integration_point(unsigned ipt, const JITFuncSpec_RequiredShapes_FiniteElement_t &required_shapes, unsigned int flag);
    virtual void set_remaining_shapes_appropriately(JITShapeInfo_t *shape_info, const JITFuncSpec_RequiredShapes_FiniteElement_t &required_shapes);
    // (Re)builds the JITElementInfo_t/JITShapeInfo_t bookkeeping (nodal/internal/external data
    // pointers, equation-number maps, hanging-node info) that the generated code relies on to
    // access this element's degrees of freedom. Must be called whenever the element's data layout
    // or equation numbering changes (e.g. after mesh refinement). If without_equations is true,
    // only the data layout is set up, skipping the (more expensive) equation-numbering part - used
    // when only field values, not residuals/Jacobians, are needed (e.g. plain evaluation/output).
    virtual void fill_element_info(bool without_equations=false);
    virtual void describe_my_dofs(std::ostream &os, const std::string &in) { this->describe_local_dofs(os, in); }
    // Jacobian determinant of the Lagrangian (undeformed, solid-mechanics reference) mapping at
    // local coordinate s, as opposed to the usual (Eulerian) geometric Jacobian.
    virtual double J_Lagrangian(const oomph::Vector<double> &s);
    virtual int get_internal_local_eqn(unsigned idindex, unsigned vindex) { return this->internal_local_eqn(idindex, vindex); }
    virtual int get_external_local_eqn(unsigned idindex, unsigned vindex) { return this->external_local_eqn(idindex, vindex); }
    // Public wrapper around get_dnormal_dcoords_at_s: computes the outer unit normal n at s, and
    // (if the output pointers are non-NULL) its first and second derivatives w.r.t. the nodal
    // coordinates, for use by generated code implementing normal-dependent boundary conditions.
    virtual void get_normal_at_s(const oomph::Vector<double> &s, oomph::Vector<double> &n, double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT dnormal_dcoord, double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT d2normal_dcoord2) const;

    // Discontinuous fields are stored as internal_data, on interfaces possibly also on external_data
    virtual oomph::Data *get_D0_nodal_data(const unsigned &fieldindex);
    virtual oomph::Data *get_DL_nodal_data(const unsigned &fieldindex);    
    virtual oomph::Data *get_DG_nodal_data(const unsigned &space_index,const unsigned &fieldindex);

    // Indices to the nodal buffer of the code generation
    
    virtual unsigned get_DG_buffer_index(const unsigned &space_index, const unsigned &fieldindex);    
    virtual unsigned get_DL_buffer_index(const unsigned &fieldindex);
    virtual unsigned get_D0_buffer_index(const unsigned &fieldindex);

    // Parent elements may have more nodal data entries than the interfaces. These functions cast a interface nodal index to the nodal index of the defining element
    virtual unsigned get_DG_node_index(const unsigned &space_index, const unsigned &fieldindex, const unsigned &nodeindex) const { return nodeindex; }
    virtual int get_DG_local_equation(const unsigned &space_index, const unsigned &fieldindex, const unsigned &nodeindex);
    
    virtual int get_DL_local_equation(const unsigned &fieldindex, const unsigned &nodeindex);
    virtual int get_D0_local_equation(const unsigned &fieldindex);

    virtual void get_DG_fields_at_s(unsigned space_index,unsigned history_index, const oomph::Vector<double> &s, oomph::Vector<double> &result) const;
    virtual int nedges() const = 0;
    virtual void add_node_from_finer_neighbor_for_tesselated_numpy(int edge, oomph::Node *n, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) {}
    virtual void inform_coarser_neighbors_for_tesselated_numpy(std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) {}
    // Re-derives the values at hanging nodes from their constraining (master) nodes after
    // refinement, for all history (time-level) slots. Used by discontinuous-field bookkeeping in
    // addition to oomph-lib's own hanging-node value interpolation.
    virtual void interpolate_hang_values();
    virtual unsigned num_DG_fields(bool base_bulk_only);
    // Core hanging-node handling: given the required shapes, fills in shape_info's hanging-node
    // equation/weight information (which local dofs are actually hanging, their master nodes and
    // weights) and, via eqn_remap, remaps local equation numbers so that hanging dofs are properly
    // eliminated/redistributed to their masters during residual/Jacobian assembly on non-conforming
    // (refined) meshes. Returns true if any hanging node was found in this element.
    virtual bool fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap);
    // Overload of fill_shape_info_at_s that writes directly into a caller-supplied shape_info buffer
    // (rather than the element's own), used e.g. when an interface element evaluates shapes of its
    // attached bulk element.
    virtual double fill_shape_info_at_s(const oomph::Vector<double> &s, const unsigned int &index, const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, double &JLagr, unsigned int flag, oomph::DenseMatrix<double> *dxds = NULL, unsigned history_index=0) const;
    virtual unsigned get_meshio_type_index() const = 0;
    // For macro-element-based (structured) meshes: projects/attaches nodes to their position on the
    // underlying macro element geometry, used e.g. for curved boundary representation.
    virtual void map_nodes_on_macro_element();
    // Assembles the full dense Hessian (second derivative of the residuals w.r.t. two degrees of
    // freedom) of this element into hbuffer, by calling the generated Hessian code at each
    // integration point. Used for bifurcation-tracking / Hessian-based solvers where the explicit
    // (dense) second derivative tensor, rather than just Hessian-vector products, is required.
    virtual void assemble_hessian_tensor(oomph::DenseMatrix<double> &hbuffer);
    // Same as assemble_hessian_tensor, but also assembles the corresponding Hessian of the mass
    // matrix (needed e.g. for parametrized eigenvalue/stability problems where the mass matrix
    // itself depends on the solution).
    virtual void assemble_hessian_and_mass_hessian(oomph::RankThreeTensor<double> &hbuffer, oomph::RankThreeTensor<double> &mbuffer);
    // Taking the old mesh, map an element with the local coordinates associated to each integration point of the new mesh.
    virtual void prepare_zeta_interpolation(oomph::MeshAsGeomObject *mesh_as_geom);
    // Enable projection
    bool enable_zeta_projection = false;
    // Initialise vector to store.
    std::vector<std::pair<pyoomph::BulkElementBase *, oomph::Vector<double>>> coords_oldmesh;
    // Fill in residuals for projection.
    virtual void residuals_for_zeta_projection(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian, const unsigned &do_fill_jacobian);
    // Assign projection time to variable.
    unsigned projection_time = 0;
    const JITElementInfo_t *get_eleminfo() const { return &eleminfo; }
    JITElementInfo_t *get_eleminfo() { return &eleminfo; }
    double get_element_diam() const;
    virtual std::vector<double> get_macro_element_coordinate_at_s(oomph::Vector<double> s);
    DynamicBulkElementInstance *get_code_instance() { return codeinst; }
    const DynamicBulkElementInstance *const get_code_instance() const { return codeinst; }

    // Global "current code instance" used to pass the DynamicBulkElementInstance through
    // oomph-lib's mesh/element construction machinery (e.g. Mesh::build, refinement son-element
    // creation), which offers no direct way to pass extra constructor arguments. Set immediately
    // before creating a new element instance of a given code, and read (then typically cleared) by
    // that element's constructor/create_son_instance.
    static DynamicBulkElementInstance *__CurrentCodeInstance; // Really annoying, but no other way to pass it through the entire mesh stur

    static unsigned zeta_time_history;    // Index in time for zeta. Only Eulerian
    static unsigned zeta_coordinate_type; // 0: Lagrangian, 1: Eulerian -- On interfaces usually boundary coordinate
    static bool use_eigen_error_estimators;

    // The "boundary coordinate" zeta used e.g. for mesh-to-mesh projection, taken to be either the
    // Lagrangian (reference/undeformed) or Eulerian (current) nodal position depending on the
    // static zeta_coordinate_type flag.
    double zeta_nodal(const unsigned &n, const unsigned &k, const unsigned &i) const
    {
      if (!zeta_coordinate_type)
        return lagrangian_position_gen(n, k, i);
      else
      {
        return nodal_position_gen(zeta_time_history, n, k, i);
      }
    }

    BulkElementBase();
    // Factory used when building a mesh from a MeshTemplate: constructs the concrete
    // BulkElementBase-derived instance matching the given template element's shape.
    static BulkElementBase *create_from_template(MeshTemplate *mt, MeshTemplateElement *el);

    virtual void ensure_external_data();

    // Connects this element (typically on a periodic mesh boundary) to the corresponding element
    // "other" on the opposite periodic boundary, along direction mydir/otherdir, so that periodic
    // degrees of freedom can be identified/coupled.
    virtual void connect_periodic_tree(BulkElementBase *other, const int &mydir, const int &otherdir);

    virtual std::vector<std::string> get_dof_names(bool not_a_root_call = false);
    // Compares the analytically assembled Jacobian (from fill_in_generic_residual_contribution_jit)
    // against a finite-difference approximation with step diff_eps, for debugging generated code.
    virtual void debug_analytical_jacobian(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian, double diff_eps);
    // Overrides oomph-lib's RefineableElement::fill_in_jacobian_from_nodal_by_fd (used by
    // debug_analytical_jacobian's generic-FD fallback). oomph-lib's version treats every nodal
    // value with node_pt->is_hanging(i)==true as governed by RefineableElement::Local_hang_eqn,
    // which is only ever sized/filled for i<ncont_interpolated_values() base-bulk fields. On
    // interface elements, nodes additionally carry interface-only values at indices
    // i>=ncont_interpolated_values(); those are geometrically hanging (the node's position is
    // hanging) but have no corresponding Local_hang_eqn entry, so calling into it indexes out of
    // bounds. This override treats all such added interface dofs as non-hanging nodal dofs instead.
    void fill_in_jacobian_from_nodal_by_fd(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian) override;
    // The core residual/Jacobian/mass-matrix assembly routine: loops over integration points,
    // evaluates the required shape functions via fill_shape_buffer_for_integration_point, and calls
    // the JIT-generated residual code with the filled shape_info buffer, accumulating into
    // residuals/jacobian/mass_matrix according to flag (0: residuals only, 1: +Jacobian, 2:
    // +Jacobian and mass matrix). This is the single most important function tying the symbolic
    // weak form to the oomph-lib assembly loop.
    virtual void fill_in_generic_residual_contribution_jit(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian, oomph::DenseMatrix<double> &mass_matrix, unsigned flag);

    ///\short Compute the derivatives of the
    /// residuals with respect to a parameter
    /// Flag=1 (or 0): do (or don't) compute the Jacobian as well.
    /// Flag=2: Fill in mass matrix too.
    virtual void fill_in_generic_dresidual_contribution_jit(double *const &parameter_pt, oomph::Vector<double> &dres_dparam, oomph::DenseMatrix<double> &djac_dparam, oomph::DenseMatrix<double> &dmass_matrix_dparam, unsigned flag);
    // Combined-assembly entry point: given a list of requested contributions (see
    // SinglePassMultiAssembleInfo), evaluates the shape functions once per integration point and
    // fills in all requested residual/Jacobian/mass-matrix/Hessian/parameter-derivative buffers in
    // a single loop, instead of the caller looping over the element once per contribution.
    virtual void get_multi_assembly(std::vector<SinglePassMultiAssembleInfo> &info);

    // Thin wrappers around fill_in_generic_residual_contribution_jit/fill_in_generic_dresidual_contribution_jit
    // that adapt to oomph-lib's expected GeneralisedElement virtual function signatures (residuals
    // only / +Jacobian / +Jacobian and mass matrix, and the parameter-derivative equivalents).
    void fill_in_contribution_to_residuals(oomph::Vector<double> &residuals)
    {
      fill_in_generic_residual_contribution_jit(residuals, oomph::GeneralisedElement::Dummy_matrix, oomph::GeneralisedElement::Dummy_matrix, 0);
    }
    void fill_in_contribution_to_jacobian(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian);
    void fill_in_contribution_to_jacobian_and_mass_matrix(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian, oomph::DenseMatrix<double> &mass_matrix);

    void fill_in_contribution_to_dresiduals_dparameter(double *const &parameter_pt, oomph::Vector<double> &dres_dparam)
    {
      fill_in_generic_dresidual_contribution_jit(parameter_pt, dres_dparam, oomph::GeneralisedElement::Dummy_matrix, oomph::GeneralisedElement::Dummy_matrix, 0);
    }

    void fill_in_contribution_to_djacobian_dparameter(double *const &parameter_pt, oomph::Vector<double> &dres_dparam, oomph::DenseMatrix<double> &djac_dparam)
    {
      fill_in_generic_dresidual_contribution_jit(parameter_pt, dres_dparam, djac_dparam, oomph::GeneralisedElement::Dummy_matrix, 1);
    }

    void fill_in_contribution_to_djacobian_and_dmass_matrix_dparameter(double *const &parameter_pt, oomph::Vector<double> &dres_dparam, oomph::DenseMatrix<double> &djac_dparam, oomph::DenseMatrix<double> &dmass_matrix_dparam)
    {
      fill_in_generic_dresidual_contribution_jit(parameter_pt, dres_dparam, djac_dparam, dmass_matrix_dparam, 2);
    }

    // Computes the Hessian-vector product contribution C^T * H * Y (H being this element's
    // residual Hessian) directly, without ever forming the dense Hessian tensor - used by
    // eigenvalue/bifurcation solvers that only need the action of the Hessian.
    void fill_in_contribution_to_hessian_vector_products(oomph::Vector<double> const &Y, oomph::DenseMatrix<double> const &C, oomph::DenseMatrix<double> &product);
    // Shared implementation behind fill_in_contribution_to_hessian_vector_products and the
    // multi-assembly Hessian requests: loops over integration points and accumulates the
    // contraction of the generated second-derivative code with Y/C into product.
    void fill_in_generic_hessian(oomph::Vector<double> const &Y, oomph::DenseMatrix<double> &C, oomph::DenseMatrix<double> &product, unsigned flag);

    // Evaluators for user-defined symbolic (GiNaC-generated) expressions attached to this element:
    // integral expressions (integrated over the element), local expressions evaluated at a given
    // local coordinate / node / element midpoint, and "extremum" expressions (min/max-type
    // quantities) at a coordinate or node.
    double eval_integral_expression(unsigned index);
    double eval_local_expression_at_s(unsigned index, const oomph::Vector<double> &s);
    double eval_local_expression_at_node(unsigned index, unsigned node_index);
    double eval_local_expression_at_midpoint(unsigned index);
    double eval_extremum_expression_at_s(unsigned index, const oomph::Vector<double> &s);
    double eval_extremum_expression_at_node(unsigned index, unsigned node_index);


    // Creates a new node at local coordinate s, interpolating its position/field values from this
    // element, optionally flagged as lying on a mesh boundary. Used e.g. when constructing
    // additional sample points (not part of the original mesh) for post-processing/projection.
    pyoomph::Node * create_interpolated_node(const oomph::Vector<double> & s,bool as_boundary_node);

    // Evaluates a user-defined "tracer" advection velocity field at local coordinate s and time
    // fraction time_frac (for particle tracing / streamline integration); returns false if
    // evaluation is not possible (e.g. s outside the element).
    bool eval_tracer_advection_in_s_space(unsigned index, double time_frac, const oomph::Vector<double> &s, oomph::Vector<double> &svelo);

    //  void assign_local_eqn_numbers(const bool &store_local_dof_pt);
    // Assigns local equation numbers to the "additional" degrees of freedom introduced beyond
    // oomph-lib's standard nodal/internal/external data handling (e.g. interface-only dofs); called
    // as part of the element's local equation numbering pass.
    void assign_additional_local_eqn_numbers();
    //  virtual void assign_all_generic_local_eqn_numbers(const bool &store_local_dof_pt);

    virtual ~BulkElementBase();


    // Creates a new, empty instance of the same concrete element type as `this` (same JIT code
    // instance), used when a mesh element is split into sons during h-refinement.
    virtual BulkElementBase *create_son_instance() const = 0;
    unsigned ncont_interpolated_values() const;
    virtual unsigned required_nvalue(const unsigned &n) const;

    // Evaluate the shape functions of the given interpolation space (C1: linear/bilinear, C2:
    // quadratic/biquadratic, DL: discontinuous-Lagrange, and below C1TB/C2TB: bubble-enriched
    // variants) at local coordinate s. Each concrete element type implements these according to its
    // own geometric shape; "makes no sense" errors are thrown for spaces an element type does not
    // support (e.g. a bilinear element has no C2 space).
    virtual void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const = 0;
    virtual void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const = 0;
    virtual void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const = 0;
    virtual void shape_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    virtual void shape_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi) const;

    // Dispatches to shape_at_s_C2TB/C2/C1TB/C1 based on a numeric space index (0..3), used by
    // generated code that addresses interpolation spaces generically by index rather than by name.
    inline void shape_of_space(const unsigned &space_index, const oomph::Vector<double> &s, oomph::Shape &psi) const
    {
      switch (space_index)
      {
      case 0:
        this->shape_at_s_C2TB(s, psi); break;
      case 1:
        this->shape_at_s_C2(s, psi); break;
      case 2:
        this->shape_at_s_C1TB(s, psi); break;
      case 3:
        this->shape_at_s_C1(s, psi); break;
      default:
        throw_runtime_error("Invalid space index " + std::to_string(space_index));
      }
    
    }

    virtual int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const = 0;
    virtual void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const = 0;

    // Local-coordinate derivatives of the shape functions for each interpolation space; analogous
    // to the shape_at_s_* family above, but returning dpsi/ds as well.
    virtual void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const = 0;
    virtual void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const = 0;
    virtual void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const = 0;
    virtual void dshape_local_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    virtual void dshape_local_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;

    // Dispatches to dshape_local_at_s_C2TB/C2/C1TB/C1 based on a numeric space index, mirroring shape_of_space.
    inline void dshape_local_of_space(const unsigned &space_index, const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
    {
      switch (space_index)
      {
      case 0:
        this->dshape_local_at_s_C2TB(s, psi, dpsi); break;
      case 1:
        this->dshape_local_at_s_C2(s, psi, dpsi); break;
      case 2:
        this->dshape_local_at_s_C1TB(s, psi, dpsi); break;
      case 3:
        this->dshape_local_at_s_C1(s, psi, dpsi); break;
      default:
        throw_runtime_error("Invalid space index " + std::to_string(space_index));
      }
    
    }
    

    

    // Construct node n of pyoomph's own Node type (rather than plain oomph::Node), so that pyoomph's
    // additional per-node bookkeeping (e.g. discontinuous-field data) is available; the
    // TimeStepper-taking overloads additionally register a specific time-stepping scheme for that node.
    virtual oomph::Node *construct_node(const unsigned &n);
    virtual oomph::Node *construct_node(const unsigned &n, oomph::TimeStepper *const &time_stepper_pt);
    virtual oomph::Node *construct_boundary_node(const unsigned &n);
    virtual oomph::Node *construct_boundary_node(const unsigned &n, oomph::TimeStepper *const &time_stepper_pt);
    virtual oomph::Node *boundary_node_pt(const int &face_index, const unsigned int index);


    // For a C2 (quadratic) node, returns the C1 (linear) nodes that geometrically "support" it
    // (i.e. whose linear interpolation would reproduce its position) - used to interpolate
    // dummy/lower-order field values living only on the C1 sub-mesh.
    virtual void get_supporting_C1_nodes_of_C2_node(const unsigned &n, std::vector<oomph::Node *> &support) { throw_runtime_error("Implement"); }

    // Evaluate the discontinuous-Lagrange (DL) resp. element-constant (D0) fields at local
    // coordinate s and history/time index t, writing all field values into res.
    void get_interpolated_fields_DL(const oomph::Vector<double> &s, std::vector<double> &res, const unsigned &t = 0) const;
    void get_interpolated_fields_D0(const oomph::Vector<double> &s, std::vector<double> &res, const unsigned &t = 0) const;

    virtual oomph::Vector<double> get_midpoint_s();                        // Set s=[0.5*(smin+smax), ... ] (but modified e.g. for tris)
    oomph::Vector<double> get_Eulerian_midpoint_from_local_coordinate();   // Set s=[0.5*(smin+smax), ... ] and evaluate the position
    oomph::Vector<double> get_Lagrangian_midpoint_from_local_coordinate(); // Set s=[0.5*(smin+smax), ... ] and evaluate the position

    void get_interpolated_values(const unsigned &t, const oomph::Vector<double> &s, oomph::Vector<double> &values);
    void get_interpolated_values(const oomph::Vector<double> &s, oomph::Vector<double> &values) { get_interpolated_values(0, s, values); }
    void get_interpolated_discontinuous_values(const unsigned &t, const oomph::Vector<double> &s, oomph::Vector<double> &values);
    void get_interpolated_discontinuous_values(const oomph::Vector<double> &s, oomph::Vector<double> &values) { get_interpolated_discontinuous_values(0, s, values); }
    void output(std::ostream &outfile, const unsigned &n_plot);

    virtual std::vector<double> get_outline(bool lagrangian) { return std::vector<double>(0); }
    // Number of independent flux quantities used by oomph-lib's Z2 error estimator for this
    // element type (drives the size of get_Z2_flux's output).
    unsigned num_Z2_flux_terms();
    // Evaluates the Z2-recovery flux vector (typically the gradient of the dominant field) at local
    // coordinate s, used by the Z2 error estimator to drive adaptive mesh refinement.
    void get_Z2_flux(const oomph::Vector<double> &s, oomph::Vector<double> &flux);
    // After h-refinement has split this element into sons and then possibly un-refined again,
    // rebuilds this element's data from the (surviving) son elements' data.
    void rebuild_from_sons(oomph::Mesh *&mesh_pt);
    // Finishes constructing a newly created element (called after pre_build/nodes are set up):
    // allocates discontinuous field data and performs other setup that requires the full node set
    // to already be in place.
    void further_build();
    // For a son element created during refinement, returns the local coordinate sfather of local
    // node l as seen in its father element (used to interpolate values from father to son node l).
    // Each concrete geometric element type must implement this according to its own son-numbering scheme.
    virtual void get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather) { throw_runtime_error("Implement"); }
    // Sets up as much of a new element as possible before all of its nodes exist yet (used during
    // mesh refinement/construction, where nodes are shared with adjacent elements and must not be
    // duplicated); new_node_pt accumulates nodes that had to be freshly constructed.
    void pre_build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt);

    unsigned nscalar_paraview() const;
    void scalar_value_paraview(std::ofstream &file_out, const unsigned &i, const unsigned &nplot) const;
    std::string scalar_name_paraview(const unsigned &i) const;
    // Additional hanging-node setup beyond oomph-lib's default (e.g. for non-isoparametric spaces
    // where hanging-node constraints differ from the geometric element's own); overridden by
    // concrete element types, many of which are pure isoparametric and can simply delegate to
    // oomph-lib's default via BulkElementBase::further_setup_hanging_nodes().
    virtual void further_setup_hanging_nodes();

    virtual int get_nodal_index_by_name(oomph::Node *n, std::string fieldname);


    /*
     inline void assign_nodal_local_eqn_numbers(const bool &store_local_dof_pt)
      {
       oomph::RefineableSolidElement::assign_nodal_local_eqn_numbers(store_local_dof_pt);
    //   assign_hanging_local_eqn_numbers(store_local_dof_pt);
    //	 fill_element_info();
      }
    */

    // After oomph-lib assigns local equation numbers for plain nodal data, additionally assigns
    // local equation numbers for hanging-node constraint equations (on refined/non-conforming meshes).
    inline void assign_nodal_local_eqn_numbers(const bool &store_local_dof_pt)
    {
      FiniteElement::assign_nodal_local_eqn_numbers(store_local_dof_pt);
      assign_hanging_local_eqn_numbers(store_local_dof_pt);
      //	 fill_element_info();
    }

    // Pins (fixes to zero contribution) resp. unpins the "dummy" data slots used to keep a
    // lower-order field's nodal data layout consistent across an element with higher-order
    // geometry (see get_dummy_value_interpolation_map); dummy dofs must stay pinned during normal
    // assembly since they carry no independent physical meaning.
    virtual void unpin_dummy_values();
    virtual void pin_dummy_values();
    // Temporarily unpins Dirichlet-constrained dofs so that a linear-algebra backend can directly
    // manipulate rows/columns for these dofs (e.g. when assembling with the constraint substituted
    // out); info records what needs to be restored afterwards.
    virtual void unpin_Dirichlet_dofs_for_matrix_manipulation(DirichletMatrixManipulationInfo & info);

    // Splits this element into its refinement "sons" and returns pointers to the newly created son
    // elements in son_pt, without altering the mesh's element list (unlike a normal adaptive
    // refinement step) - used e.g. for one-off geometric subdivision/export.
    void dynamic_split(oomph::Vector<BulkElementBase *> &son_pt) const;

    // Geometric (non-solid) Jacobian determinant of the mapping from local to given global
    // coordinates x, used by oomph-lib's locate_zeta / point-location machinery.
    double geometric_jacobian(const oomph::Vector<double> &x) override;

    // Debug helper: assembles the residual vector R and Jacobian matrix J together with a
    // human-readable name for each degree of freedom (dofnames), for inspection/printing.
    void get_debug_jacobian_info(oomph::Vector<double> &R, oomph::DenseMatrix<double> &J, std::vector<std::string> &dofnames);
    double elemental_error_max_override;

    // A measure of element shape quality (e.g. for mesh-quality-based remeshing triggers); default
    // implementation and overrides differ by geometric element type.
    virtual double get_quality_factor();

    // Given that following direction ds from local coordinate s would leave the element's valid
    // local-coordinate domain, computes the largest scale factor for which s+factor*ds is still (on
    // the boundary of) the valid domain, along with the corresponding boundary normal snormal and
    // remaining "overshoot" distance sdistance. Used e.g. when integrating particle/tracer paths
    // that cross element boundaries. Must be implemented per concrete geometric element type.
    virtual double factor_when_local_coordinate_becomes_invalid(const oomph::Vector<double> &s, const oomph::Vector<double> &ds, oomph::Vector<double> &snormal, double &sdistance) { throw_runtime_error("Implement for the specific element"); }

    // Sets the integration (quadrature) order/scheme to use for this element; each concrete element
    // type maps "order" to the appropriate IntegrationSchemeStorage lookup for its shape.
    virtual void set_integration_order(unsigned int order) { throw_runtime_error("Implement"); }

    virtual bool has_bubble() const { return false; } // If not, C2TB is the same space as C2

    // Debug helper analogous to debug_analytical_jacobian, but for the Hessian: compares the
    // analytic Hessian-vector product fill_in_generic_hessian(Y, C, ...) against a finite-difference
    // approximation with step epsilon.
    virtual void debug_hessian(std::vector<double> Y, std::vector<std::vector<double>> C, double epsilon);
    // Looks up all (Data*, index) pairs holding the field called "name" on this element (nodal,
    // internal or external data as appropriate); use_elemental_indices selects whether the returned
    // index is the element-local field index or the raw Data-object component index.
    virtual std::vector<std::pair<oomph::Data *, int>> get_field_data_list(std::string name, bool use_elemental_indices);
  };


  // Base class for "ODE elements": zero-dimensional pyoomph elements with no spatial nodes at all,
  // used to represent plain ODEs / globally-coupled degrees of freedom (e.g. scalar ODEs, global
  // parameters evolving in time) within the same JIT/residual-assembly framework as spatial
  // finite elements. Since it has no nodes, shape functions and node-related queries are trivial/no-ops.
  class ODEElementBase : public virtual oomph::FiniteElement
  {
  public:
    /// Constructor
    ODEElementBase()
    {
      this->set_n_node(0);
    }

    /// Broken copy constructor
    ODEElementBase(const ODEElementBase&) = delete;

    /// Calculate the geometric shape functions at local coordinate s
    void shape(const oomph::Vector<double>& s, oomph::Shape& psi) const {}

    void local_coordinate_of_node(const unsigned& j, oomph::Vector<double>& s) const
    {
      s.resize(0);
    }
    
  };


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
    bool fill_hang_info_with_equations_for_pos(JITShapeInfo_t *shape_info) override {return false;} // An ODE never has positions
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override {throw_runtime_error("ODE Elements do not have faces"); return NULL;}
    virtual const std::vector<int> & get_possible_face_indices() const { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override { return std::vector<pyoomph::Node*>(); }
    int nedges() const { return 0; }
    virtual double fill_shape_info_at_s(const oomph::Vector<double> &s, const unsigned int &index, const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, double &JLagr, unsigned int flag, oomph::DenseMatrix<double> *dxds = NULL, unsigned history_index=0) const;
    virtual unsigned get_meshio_type_index() const { return 0; }
    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { throw_runtime_error("Makes no sense"); }
    
    unsigned nrecovery_order() { return 0; }
    unsigned nvertex_node() const { return 0; }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return NULL; }
    void further_setup_hanging_nodes() {};
    void to_numpy(double *dest);
    void shape(const oomph::Vector<double> &s, oomph::Shape &psi) const {}
    void build(oomph::Mesh *&, oomph::Vector<oomph::Node *> &, bool &, std::ofstream &) {}
    void check_integrity(double &max_error) { max_error = 0; }
    virtual BulkElementBase *create_son_instance() const { return NULL; }
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const {}
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const {}
    void shape_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi) const {}
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const {}
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      nsubdiv = 0;
      return 0;
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const { isubelem = 0; }

    BulkElementODE0d(DynamicBulkElementInstance *code_inst, oomph::TimeStepper *tstepper);
    virtual ~BulkElementODE0d();

    // Factory that sets the __CurrentCodeInstance "side channel" (see BulkElementBase) around
    // construction so the new element picks up the right JIT code instance, then clears it again.
    static BulkElementODE0d *construct_new(DynamicBulkElementInstance *code_inst, oomph::TimeStepper *tstepper)
    {
      BulkElementBase::__CurrentCodeInstance = code_inst;
      BulkElementODE0d *res = new BulkElementODE0d(code_inst, tstepper);
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    virtual double get_quality_factor() { return 1.0; }

    virtual void set_integration_order(unsigned int order) {}
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
    RefineableSolidLineElement(const RefineableSolidLineElement &dummy)
    {
      oomph::BrokenCopy::broken_copy("RefineableSolidLineElement");
    }

    virtual ~RefineableSolidLineElement() {}

    void set_macro_elem_pt(oomph::MacroElement *macro_elem_pt)
    {
      oomph::QSolidElementBase::set_macro_elem_pt(macro_elem_pt);
    }

    void set_macro_elem_pt(oomph::MacroElement *macro_elem_pt, oomph::MacroElement *undeformed_macro_elem_pt)
    {
      oomph::QSolidElementBase::set_macro_elem_pt(macro_elem_pt, undeformed_macro_elem_pt);
    }

    void get_jacobian(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian)
    {
      oomph::RefineableSolidElement::get_jacobian(residuals, jacobian);
    }

    void build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt,
               bool &was_already_built,
               std::ofstream &new_nodes_file);
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
  public:
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    virtual const std::vector<int> & get_possible_face_indices() const { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
    int nedges() const { return 2; }
    BulkElementLine1dC1();    
    virtual unsigned get_meshio_type_index() const { return 1; }
    void check_integrity(double &max_error) { max_error = 0; } // TODO throw_runtime_error("IMPLEMENT");

    

    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { throw_runtime_error("Makes no sense"); }
    void shape_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi) const { throw_runtime_error("Makes no sense"); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;

    unsigned nrecovery_order() { return 1; }
    unsigned nvertex_node() const { return oomph::QElement<1, 2>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return QElement<1, 2>::vertex_node_pt(j); }
    // void further_setup_hanging_nodes() {BulkElementBase::further_setup_hanging_nodes();};
    void further_setup_hanging_nodes() {};
    virtual std::vector<double> get_outline(bool lagrangian);
    virtual BulkElementBase *create_son_instance() const
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementLine1dC1();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }

    void pre_build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt)
    {
      BulkElementBase::pre_build(mesh_pt, new_node_pt);
      oomph::RefineableQElement<1>::pre_build(mesh_pt, new_node_pt);
    }

    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      nsubdiv = 1;
      return 2;
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    double factor_when_local_coordinate_becomes_invalid(const oomph::Vector<double> &s, const oomph::Vector<double> &ds, oomph::Vector<double> &snormal, double &sdistance);
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 1, order)); }
    virtual void get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather);
  };

  // 1d line element, quadratic (C2) Lagrange interpolation (plus a C1 dummy sub-space), refineable + solid.
  class BulkElementLine1dC2 : public virtual BulkElementBase, public virtual oomph::QElement<1, 3>, public virtual RefineableSolidLineElement
  {
  protected:    
    static const std::vector<int> Possible_Face_Indices;
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<std::vector<unsigned>>> Dummy_Value_Interpolation_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const override {return Dummy_Value_Interpolation_Map;}
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    virtual const std::vector<int> & get_possible_face_indices() const { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
    int nedges() const { return 2; }
    virtual unsigned get_meshio_type_index() const { return 2; }
    BulkElementLine1dC2();    
    void check_integrity(double &max_error) { max_error = 0; } // TODO

    

    
    
    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }

    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      nsubdiv = 1;
      return 3;
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    virtual std::vector<double> get_outline(bool lagrangian);
    unsigned nrecovery_order() { return 2; }
    unsigned nvertex_node() const { return oomph::QElement<1, 3>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return QElement<1, 3>::vertex_node_pt(j); }
    // void further_setup_hanging_nodes() {BulkElementBase::further_setup_hanging_nodes();};
    void further_setup_hanging_nodes() {};
    virtual BulkElementBase *create_son_instance() const
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementLine1dC2();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    virtual double factor_when_local_coordinate_becomes_invalid(const oomph::Vector<double> &s, const oomph::Vector<double> &ds, oomph::Vector<double> &snormal, double &sdistance);
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 1, order)); }
    virtual void get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather);
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
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    virtual const std::vector<int> & get_possible_face_indices() const { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
    int nedges() const { return 2; }
    BulkTElementLine1dC1();    
    virtual unsigned get_meshio_type_index() const { return 1; }
    void check_integrity(double &max_error) { max_error = 0; } // TODO throw_runtime_error("IMPLEMENT");

    
    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { throw_runtime_error("Makes no sense"); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;

    unsigned nrecovery_order() { return 1; }
    unsigned nvertex_node() const { return oomph::TElement<1, 2>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return TElement<1, 2>::vertex_node_pt(j); }
    // void further_setup_hanging_nodes() {BulkElementBase::further_setup_hanging_nodes();};
    void further_setup_hanging_nodes() {};
    virtual std::vector<double> get_outline(bool lagrangian);
    virtual BulkElementBase *create_son_instance() const
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkTElementLine1dC1();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }

    void pre_build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt)
    {
      BulkElementBase::pre_build(mesh_pt, new_node_pt);
      oomph::RefineableTElement<1>::pre_build(mesh_pt, new_node_pt);
    }

    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      nsubdiv = 1;
      return 2;
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 1, order)); }
  };

  // 1d simplex ("T") line element, quadratic (C2) interpolation.
  class BulkTElementLine1dC2 : public virtual BulkElementBase, public virtual oomph::TElement<1, 3>, public virtual oomph::RefineableTElement<1>
  {
  protected:    
    static const std::vector<int> Possible_Face_Indices;
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<std::vector<unsigned>>> Dummy_Value_Interpolation_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const override {return Dummy_Value_Interpolation_Map;}
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    virtual const std::vector<int> & get_possible_face_indices() const { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
    int nedges() const { return 2; }
    virtual unsigned get_meshio_type_index() const { return 2; }
    BulkTElementLine1dC2();    
    void check_integrity(double &max_error) { max_error = 0; } // TODO

    
    
    
    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }

    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      nsubdiv = 1;
      return 3;
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    virtual std::vector<double> get_outline(bool lagrangian);
    unsigned nrecovery_order() { return 2; }
    unsigned nvertex_node() const { return oomph::TElement<1, 3>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return TElement<1, 3>::vertex_node_pt(j); }
    // void further_setup_hanging_nodes() {BulkElementBase::further_setup_hanging_nodes();};
    void further_setup_hanging_nodes() {};
    virtual BulkElementBase *create_son_instance() const
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkTElementLine1dC2();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 1, order)); }
  };

  // 2d quadrilateral element, bilinear (C1) interpolation, refineable + solid.
  class BulkElementQuad2dC1 : public virtual BulkElementBase, public virtual oomph::QElement<2, 2>, public virtual oomph::RefineableSolidQElement<2>
  {
  protected:
    static const std::vector<int> Possible_Face_Indices;
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    virtual const std::vector<int> & get_possible_face_indices() const { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
    int nedges() const { return 4; }
    BulkElementQuad2dC1();    
    virtual unsigned get_meshio_type_index() const { return 6; }

    void check_integrity(double &max_error) { max_error = 0; } // TODO

    

    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { throw_runtime_error("Makes no sense"); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;

    void add_node_from_finer_neighbor_for_tesselated_numpy(int edge, oomph::Node *n, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes);
    void inform_coarser_neighbors_for_tesselated_numpy(std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes);
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    virtual oomph::Node *boundary_node_pt(const int &face_index, const unsigned int index);
    virtual std::vector<double> get_outline(bool lagrangian);
    unsigned nrecovery_order() { return 1; }
    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }
    unsigned nvertex_node() const { return oomph::QElement<2, 2>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return QElement<2, 2>::vertex_node_pt(j); }
    void further_setup_hanging_nodes() { BulkElementBase::further_setup_hanging_nodes(); } // There can't be any problem here, since it is all isoparametric
    virtual BulkElementBase *create_son_instance() const
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementQuad2dC1();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    virtual void get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather);
    virtual double factor_when_local_coordinate_becomes_invalid(const oomph::Vector<double> &s, const oomph::Vector<double> &ds, oomph::Vector<double> &snormal, double &sdistance);
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 2, order)); }
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
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const override {return Dummy_Value_Interpolation_Map;}  
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    virtual const std::vector<int> & get_possible_face_indices() const { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
    virtual void get_supporting_C1_nodes_of_C2_node(const unsigned &n, std::vector<oomph::Node *> &support);
    int nedges() const { return 4; }
    BulkElementQuad2dC2();
    virtual unsigned get_meshio_type_index() const { return 8; }
    

    void check_integrity(double &max_error) { max_error = 0; } // TODO
    virtual std::vector<double> get_outline(bool lagrangian);
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;

    virtual oomph::Node *boundary_node_pt(const int &face_index, const unsigned int index);
    

    
    

    void add_node_from_finer_neighbor_for_tesselated_numpy(int edge, oomph::Node *n, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes);
    void inform_coarser_neighbors_for_tesselated_numpy(std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes);
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;

    unsigned nrecovery_order() { return 2; }
    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }
    unsigned nvertex_node() const { return oomph::QElement<2, 3>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return QElement<2, 3>::vertex_node_pt(j); }

    void further_setup_hanging_nodes();
    oomph::Node *interpolating_node_pt(const unsigned &n, const int &value_id);
    double local_one_d_fraction_of_interpolating_node(const unsigned &n1d, const unsigned &i, const int &value_id);
    oomph::Node *get_interpolating_node_at_local_coordinate(const oomph::Vector<double> &s, const int &value_id);
    unsigned ninterpolating_node_1d(const int &value_id);
    unsigned ninterpolating_node(const int &value_id);
    void interpolating_basis(const oomph::Vector<double> &s, oomph::Shape &psi, const int &value_id) const;
    virtual BulkElementBase *create_son_instance() const
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementQuad2dC2();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    virtual void get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather);
    virtual double factor_when_local_coordinate_becomes_invalid(const oomph::Vector<double> &s, const oomph::Vector<double> &ds, oomph::Vector<double> &snormal, double &sdistance);
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 2, order)); }
  };

  // 2d triangular element, linear (C1) interpolation.
  class BulkElementTri2dC1 : public virtual BulkElementBase, public virtual oomph::TElement<2, 2>, public virtual oomph::RefineableTElement<2>
  {
  protected:
    static const std::vector<int> Possible_Face_Indices;
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    virtual const std::vector<int> & get_possible_face_indices() const { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
    virtual oomph::Node *boundary_node_pt(const int &face_index, const unsigned int index);
    int nedges() const { return 3; }
    unsigned nnode_on_face() const override { return 2; }
    BulkElementTri2dC1(bool has_bubble = false);    
    virtual unsigned get_meshio_type_index() const { return 3; }
    void check_integrity(double &max_error) { max_error = 0; } // TODO
    
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { throw_runtime_error("Makes no sense"); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      nsubdiv = 1;
      return 3;
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    virtual std::vector<double> get_outline(bool lagrangian);
    unsigned nrecovery_order() { return 1; }
    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }
    unsigned nvertex_node() const { return oomph::TElement<2, 2>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return TElement<2, 2>::vertex_node_pt(j); }
    void further_setup_hanging_nodes() { BulkElementBase::further_setup_hanging_nodes(); } // There can't be any problem here, since it is all isoparametric
    virtual BulkElementBase *create_son_instance() const
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementTri2dC1();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    virtual double factor_when_local_coordinate_becomes_invalid(const oomph::Vector<double> &s, const oomph::Vector<double> &ds, oomph::Vector<double> &snormal, double &sdistance);
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 2, order)); }
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
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const override {return Dummy_Value_Interpolation_Map;}  
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    BulkElementTri2dC1TB();
    
    void shape(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void dshape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids) const;

    
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void shape_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    void dshape_local_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }    
    inline void d2shape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids, oomph::DShape &d2psids) const { throw_runtime_error("Implement"); }

    void local_coordinate_of_node(const unsigned &j, oomph::Vector<double> &s) const;
    bool has_bubble() const { return true; }
    virtual unsigned get_meshio_type_index() const { return 66; } // Just some otherwise unused value here
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      if (tesselate_tri)
      {
        nsubdiv = 3;
        return 3;
      }
      else
      {
        nsubdiv = 1;
        return 4;
      }
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;

    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 2, order, true)); }
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
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const override {return Dummy_Value_Interpolation_Map;}  
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    virtual const std::vector<int> & get_possible_face_indices() const { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;    
    virtual void get_supporting_C1_nodes_of_C2_node(const unsigned &n, std::vector<oomph::Node *> &support);
    virtual oomph::Node *boundary_node_pt(const int &face_index, const unsigned int index);
    int nedges() const { return 3; }
    unsigned nnode_on_face() const override { return 3; }
    BulkElementTri2dC2(bool with_bubble = false);    
    virtual unsigned get_meshio_type_index() const { return 9; }
    
    void check_integrity(double &max_error) { max_error = 0; } // TODO
    
    
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      if (tesselate_tri)
      {
        nsubdiv = 4;
        return 3;
      }
      else
      {
        nsubdiv = 1;
        return 6;
      }
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    virtual std::vector<double> get_outline(bool lagrangian);
    unsigned nrecovery_order() { return 2; }
    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }
    unsigned nvertex_node() const { return oomph::TElement<2, 3>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return TElement<2, 3>::vertex_node_pt(j); }
    void further_setup_hanging_nodes() { BulkElementBase::further_setup_hanging_nodes(); } // There can't be any problem here, since it is all isoparametric
    virtual BulkElementBase *create_son_instance() const;
    virtual double factor_when_local_coordinate_becomes_invalid(const oomph::Vector<double> &s, const oomph::Vector<double> &ds, oomph::Vector<double> &snormal, double &sdistance);
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 2, order)); }
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
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const override {return Dummy_Value_Interpolation_Map;}  
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    BulkElementTri2dC2TB();
    
    void shape_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi) const override;
    void dshape_local_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { BulkElementTri2dC2::shape(s, psi); }
    void shape_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi) const { oomph::TBubbleEnrichedElementShape<2, 3>::shape(s, psi); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { BulkElementTri2dC2::dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { oomph::TBubbleEnrichedElementShape<2, 3>::dshape_local(s, psi, dpsi); }
    
    inline void d2shape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids, oomph::DShape &d2psids) const { throw_runtime_error("Implement"); }
    inline void shape(const oomph::Vector<double> &s, oomph::Shape &psi) const { oomph::TBubbleEnrichedElementShape<2, 3>::shape(s, psi); }
    inline void dshape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids) const { oomph::TBubbleEnrichedElementShape<2, 3>::dshape_local(s, psi, dpsids); }
    inline void local_coordinate_of_node(const unsigned &j, oomph::Vector<double> &s) const { oomph::TBubbleEnrichedElementShape<2, 3>::local_coordinate_of_node(j, s); }
    bool has_bubble() const { return true; }
    virtual unsigned get_meshio_type_index() const { return 99; } // Just some otherwise unused value here
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      if (tesselate_tri)
      {
        nsubdiv = 3;
        return 3;
      }
      else
      {
        nsubdiv = 1;
        return 7;
      }
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;

    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 2, order, true)); }
  };

  // 3d brick (hexahedral) element, trilinear (C1) interpolation, refineable + solid.
  class BulkElementBrick3dC1 : public virtual BulkElementBase, public virtual oomph::QElement<3, 2>, public virtual oomph::RefineableSolidQElement<3>
  {
  protected:
    static const std::vector<int> Possible_Face_Indices;
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    virtual const std::vector<int> & get_possible_face_indices() const { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
    int nedges() const { return 8; }
    BulkElementBrick3dC1();    
    virtual unsigned get_meshio_type_index() const { return 11; }


    void check_integrity(double &max_error) { max_error = 0; } // TODO

    

    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { throw_runtime_error("Makes no sense"); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;

    void get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather);
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
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
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    virtual std::vector<double> get_outline(bool lagrangian);
    unsigned nrecovery_order() { return 1; }
    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }
    unsigned nvertex_node() const { return oomph::QElement<3, 2>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return QElement<3, 2>::vertex_node_pt(j); }
    void further_setup_hanging_nodes() { BulkElementBase::further_setup_hanging_nodes(); } // There can't be any problem here, since it is all isoparametric
    virtual BulkElementBase *create_son_instance() const
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementBrick3dC1();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 3, order)); }
  };

  // 3d brick element, triquadratic (C2) interpolation (plus a C1 dummy sub-space).
  class BulkElementBrick3dC2 : public virtual BulkElementBase, public virtual oomph::QElement<3, 3>, public virtual oomph::RefineableSolidQElement<3>
  {
  protected:
    static const std::vector<int> Possible_Face_Indices;
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<std::vector<unsigned>>> Dummy_Value_Interpolation_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const override {return Dummy_Value_Interpolation_Map;}  
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    virtual const std::vector<int> & get_possible_face_indices() const { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
    int nedges() const { return 8; }
    BulkElementBrick3dC2();
    virtual unsigned get_meshio_type_index() const { return 14; }
    

    void check_integrity(double &max_error) { max_error = 0; } // TODO
    virtual std::vector<double> get_outline(bool lagrangian);
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;

    

    
    

    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
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
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;

    unsigned nrecovery_order() { return 2; }
    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }
    unsigned nvertex_node() const { return oomph::QElement<3, 3>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return QElement<3, 3>::vertex_node_pt(j); }

    void further_setup_hanging_nodes();
    oomph::Node *interpolating_node_pt(const unsigned &n, const int &value_id);
    double local_one_d_fraction_of_interpolating_node(const unsigned &n1d, const unsigned &i, const int &value_id);
    oomph::Node *get_interpolating_node_at_local_coordinate(const oomph::Vector<double> &s, const int &value_id);
    unsigned ninterpolating_node_1d(const int &value_id);
    unsigned ninterpolating_node(const int &value_id);
    void interpolating_basis(const oomph::Vector<double> &s, oomph::Shape &psi, const int &value_id) const;
    void get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather);
    virtual BulkElementBase *create_son_instance() const
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementBrick3dC2();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 3, order)); }
  };

  // 3d tetrahedral element, linear (C1) interpolation.
  class BulkElementTetra3dC1 : public virtual BulkElementBase, public virtual oomph::TElement<3, 2>, public virtual oomph::RefineableTElement<3>
  {
  protected:
     static const std::vector<int> Possible_Face_Indices;
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;    
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    virtual const std::vector<int> & get_possible_face_indices() const { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
    int nedges() const { return 6; }
    BulkElementTetra3dC1();    
    virtual unsigned get_meshio_type_index() const { return 4; }

    void check_integrity(double &max_error) { max_error = 0; } // TODO

    
    
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { throw_runtime_error("Makes no sense"); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;

    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
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
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    virtual std::vector<double> get_outline(bool lagrangian);
    unsigned nrecovery_order() { return 1; }
    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }
    unsigned nvertex_node() const { return oomph::TElement<3, 2>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return TElement<3, 2>::vertex_node_pt(j); }
    void further_setup_hanging_nodes() { BulkElementBase::further_setup_hanging_nodes(); } // There can't be any problem here, since it is all isoparametric
    virtual BulkElementBase *create_son_instance() const
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementTetra3dC1();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 3, order)); }
    oomph::Vector<double> get_midpoint_s() override { return oomph::Vector<double>(this->dim(), 1.0 / 3.0); }
  };

  // 3d tetrahedral element, linear interpolation enriched with a quartic interior bubble function (C1TB space).
  class BulkElementTetra3dC1TB : public virtual BulkElementTetra3dC1
  {
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<std::vector<unsigned>>> Dummy_Value_Interpolation_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
  public:
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const override {return Dummy_Value_Interpolation_Map;}
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    BulkElementTetra3dC1TB();
    virtual unsigned get_meshio_type_index() const { return 44; }
    void shape(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void dshape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids) const;
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void shape_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    void dshape_local_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    void local_coordinate_of_node(const unsigned &j, oomph::Vector<double> &s) const;
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
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
    virtual BulkElementBase *create_son_instance() const
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementTetra3dC1TB();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 3, order,true)); }
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
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const override {return Dummy_Value_Interpolation_Map;}  
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    virtual const std::vector<int> & get_possible_face_indices() const { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
    int nedges() const { return 6; }
    BulkElementTetra3dC2(bool has_bubble = false);
    virtual unsigned get_meshio_type_index() const { return 10; }
    

    void check_integrity(double &max_error) { max_error = 0; } // TODO
    virtual std::vector<double> get_outline(bool lagrangian);
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;

    

    
    

    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
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
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;

    unsigned nrecovery_order() { return 2; }
    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }
    unsigned nvertex_node() const { return oomph::TElement<3, 3>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return TElement<3, 3>::vertex_node_pt(j); }

    void further_setup_hanging_nodes();
    // TODO: For refinement!
    // oomph::Node* interpolating_node_pt(const unsigned &n,const int &value_id);
    // double local_one_d_fraction_of_interpolating_node(const unsigned &n1d,const unsigned &i,const int &value_id);
    // oomph::Node* get_interpolating_node_at_local_coordinate(const oomph::Vector<double> &s,const int &value_id); //TO be done
    // unsigned ninterpolating_node_1d(const int &value_id);
    // unsigned ninterpolating_node(const int &value_id);
    void interpolating_basis(const oomph::Vector<double> &s, oomph::Shape &psi, const int &value_id) const;
    virtual BulkElementBase *create_son_instance() const;
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 3, order)); }
    oomph::Vector<double> get_midpoint_s() override { return oomph::Vector<double>(this->dim(), 1.0 / 3.0); }
  };

  // 3d tetrahedral element, quadratic interpolation enriched with an interior bubble function (C2TB space).
  class BulkElementTetra3dC2TB : public virtual BulkElementTetra3dC2, public oomph::TBubbleEnrichedElementShape<3, 3>
  {
  private:
    static oomph::TBubbleEnrichedGauss<3, 3> Default_enriched_integration_scheme;
    //  static const unsigned Central_node_on_face[3];
    static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<std::vector<unsigned>>> Dummy_Value_Interpolation_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const override {return Dummy_Value_Interpolation_Map;}  
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
    BulkElementTetra3dC2TB();    
    virtual unsigned get_meshio_type_index() const { return 100; } // Just some otherwise unused value here
    
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
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

    
    
    
    void shape_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi) const;    
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { BulkElementTetra3dC2::shape(s, psi); }
    void shape_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi) const { oomph::TBubbleEnrichedElementShape<3, 3>::shape(s, psi); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { BulkElementTetra3dC2::dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    void dshape_local_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { oomph::TBubbleEnrichedElementShape<3, 3>::dshape_local(s, psi, dpsi); }
    inline void d2shape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids, oomph::DShape &d2psids) const { throw_runtime_error("Implement"); }
    inline void shape(const oomph::Vector<double> &s, oomph::Shape &psi) const { oomph::TBubbleEnrichedElementShape<3, 3>::shape(s, psi); }
    inline void dshape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids) const { oomph::TBubbleEnrichedElementShape<3, 3>::dshape_local(s, psi, dpsids); }
    inline void local_coordinate_of_node(const unsigned &j, oomph::Vector<double> &s) const { oomph::TBubbleEnrichedElementShape<3, 3>::local_coordinate_of_node(j, s); }
    bool has_bubble() const { return true; }
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 3, order, true)); }
    void build_face_element(const int& face_index, oomph::FaceElement* face_element_pt) override;
  };

  // 3d wedge (triangular prism) element, linear (C1) interpolation.
  class BulkElementWedge3dC1 : public virtual BulkElementBase, public virtual oomph::WedgeElementC1
  {
    protected:
      static const std::vector<int> Possible_Face_Indices;
      static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
      const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
      oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
      virtual const std::vector<int> & get_possible_face_indices() const { return Possible_Face_Indices; }
      std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
      BulkElementWedge3dC1();
      int nedges() const { throw_runtime_error("Not implemented"); }
      virtual unsigned get_meshio_type_index() const { return 13; }      
      void shape(const oomph::Vector<double> &s, oomph::Shape &psi) const {oomph::WedgeElementC1::shape(s, psi); }
      void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
      void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { throw_runtime_error("Makes no sense"); }
      void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;
      void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
      void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { throw_runtime_error("Makes no sense"); }
      void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
      
      int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
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
      void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
      virtual std::vector<double> get_outline(bool lagrangian);
      unsigned nrecovery_order() { return 1; }
      void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }      
      unsigned nvertex_node() const { return oomph::WedgeElementC1::nvertex_node(); }
      oomph::Node *vertex_node_pt(const unsigned &j) const { return WedgeElementC1::vertex_node_pt(j); }      
      void further_setup_hanging_nodes() { BulkElementBase::further_setup_hanging_nodes(); } // There can't be any problem here, since it is all isoparametric
      virtual BulkElementBase *create_son_instance() const
      {
          BulkElementBase::__CurrentCodeInstance = codeinst;
          auto res = new BulkElementWedge3dC1();
          res->codeinst = codeinst;
          BulkElementBase::__CurrentCodeInstance = NULL;
          return res;
      }
      virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 4, order)); }
      oomph::Vector<double> get_midpoint_s() override { oomph::Vector<double> res(this->dim(), 1.0 / 3.0); res[2]=0.5; return res; }
  };


  // 3d pyramid element, linear (C1) interpolation.
  class BulkElementPyramid3dC1 : public virtual BulkElementBase, public virtual oomph::PyramidElementC1
  {
    protected:
      static const std::vector<int> Possible_Face_Indices;
      static const std::vector<std::vector<unsigned>> Nodal_Space_Index_To_Element_Index_Map;
    static const std::vector<std::vector<int>> Element_Index_To_Nodal_Space_Index_Map;
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
      const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
      oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
      virtual const std::vector<int> & get_possible_face_indices() const { return Possible_Face_Indices; }
      std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
      BulkElementPyramid3dC1();
      int nedges() const { throw_runtime_error("Not implemented"); } // No need tom implement this now
      virtual unsigned get_meshio_type_index() const { return 15; }      
      void shape(const oomph::Vector<double> &s, oomph::Shape &psi) const {oomph::PyramidElementC1::shape(s, psi); }
      void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
      void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { throw_runtime_error("Makes no sense"); }
      void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;
      void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
      void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { throw_runtime_error("Makes no sense"); }
      void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;      
      int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
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
      void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
      virtual std::vector<double> get_outline(bool lagrangian);
      unsigned nrecovery_order() { return 1; }
      void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }      
      unsigned nvertex_node() const { return oomph::PyramidElementC1::nvertex_node(); }
      oomph::Node *vertex_node_pt(const unsigned &j) const { return PyramidElementC1::vertex_node_pt(j); }      
      void further_setup_hanging_nodes() { BulkElementBase::further_setup_hanging_nodes(); } // There can't be any problem here, since it is all isoparametric
      virtual BulkElementBase *create_son_instance() const
      {
          BulkElementBase::__CurrentCodeInstance = codeinst;
          auto res = new BulkElementPyramid3dC1();
          res->codeinst = codeinst;
          BulkElementBase::__CurrentCodeInstance = NULL;
          return res;
      }
      virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 5, order)); }
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
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
      const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const override {return Dummy_Value_Interpolation_Map;}    
      const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
      oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
      virtual const std::vector<int> & get_possible_face_indices() const { return Possible_Face_Indices; }
      std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
      BulkElementWedge3dC2();      
      int nedges() const { throw_runtime_error("Not implemented"); }
      virtual unsigned get_meshio_type_index() const { return 26; }      
      void shape(const oomph::Vector<double> &s, oomph::Shape &psi) const {oomph::WedgeElementC2::shape(s, psi); }
      void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const { oomph::WedgeElementShapeC1::shape(s, psi); }
      void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }      
      void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;
      void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
      void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { oomph::WedgeElementShapeC1::dshape_local(s, psi, dpsi); }
      void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
      
      int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
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
      void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
      virtual std::vector<double> get_outline(bool lagrangian);
      unsigned nrecovery_order() { return 1; }
      void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }      
      unsigned nvertex_node() const { return oomph::WedgeElementC2::nvertex_node(); }
      oomph::Node *vertex_node_pt(const unsigned &j) const { return WedgeElementC2::vertex_node_pt(j); }      
      void further_setup_hanging_nodes() { BulkElementBase::further_setup_hanging_nodes(); } // There can't be any problem here, since it is all isoparametric
      virtual BulkElementBase *create_son_instance() const
      {
          BulkElementBase::__CurrentCodeInstance = codeinst;
          auto res = new BulkElementWedge3dC2();
          res->codeinst = codeinst;
          BulkElementBase::__CurrentCodeInstance = NULL;
          return res;
      }
      virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 4, order)); }
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
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
      const std::vector<std::vector<std::vector<unsigned>>> & get_dummy_value_interpolation_map() const override {return Dummy_Value_Interpolation_Map;}    
      const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
      oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override;
      virtual const std::vector<int> & get_possible_face_indices() const { return Possible_Face_Indices; }
      std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override;
      BulkElementPyramid3dC2();      
      int nedges() const { throw_runtime_error("Not implemented"); }
      virtual unsigned get_meshio_type_index() const { return 27; }      
      void shape(const oomph::Vector<double> &s, oomph::Shape &psi) const {oomph::PyramidElementC2::shape(s, psi); }
      void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const { oomph::PyramidElementShapeC1::shape(s, psi); }
      void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }      
      void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;
      void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
      void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { oomph::PyramidElementShapeC1::dshape_local(s, psi, dpsi); }
      void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
      
      int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
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
      void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
      virtual std::vector<double> get_outline(bool lagrangian);
      unsigned nrecovery_order() { return 1; }
      void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }      
      unsigned nvertex_node() const { return oomph::PyramidElementC2::nvertex_node(); }
      oomph::Node *vertex_node_pt(const unsigned &j) const { return PyramidElementC2::vertex_node_pt(j); }      
      void further_setup_hanging_nodes() { BulkElementBase::further_setup_hanging_nodes(); } // There can't be any problem here, since it is all isoparametric
      virtual BulkElementBase *create_son_instance() const
      {
          BulkElementBase::__CurrentCodeInstance = codeinst;
          auto res = new BulkElementPyramid3dC2();
          res->codeinst = codeinst;
          BulkElementBase::__CurrentCodeInstance = NULL;
          return res;
      }
      virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 5, order)); }
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
  public:    
    const std::vector<std::vector<int>> & get_element_index_to_nodal_space_index_map() const override {return Element_Index_To_Nodal_Space_Index_Map;}
    const std::vector<std::vector<unsigned>> & get_nodal_space_index_to_element_index_map() const override {return Nodal_Space_Index_To_Element_Index_Map;}
    oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *jitcode, int face_index) override {throw_runtime_error("A point element has no faces");}
    virtual const std::vector<int> & get_possible_face_indices() const { return Possible_Face_Indices; }
    std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const override {return std::vector<pyoomph::Node*>{}; }
    int nedges() const { return 0; }
    PointElement0d();
    virtual unsigned get_meshio_type_index() const { return 0; }
    virtual void dshape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids) const;
    virtual double invert_jacobian_mapping(const oomph::DenseMatrix<double> &jacobian, oomph::DenseMatrix<double> &inverse_jacobian) const;
    void build(oomph::Mesh *&, oomph::Vector<oomph::Node *> &, bool &, std::ofstream &) {}
    void check_integrity(double &max_error) { max_error = 0; }
    
    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    unsigned nrecovery_order() { return 1; }
    unsigned nvertex_node() const { return 1; }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return node_pt(j); }
    void further_setup_hanging_nodes() {};
    virtual BulkElementBase *create_son_instance() const
    {
      throw_runtime_error("Makes no sense");
      return NULL;
    }
    void pre_build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt)
    {
      BulkElementBase::pre_build(mesh_pt, new_node_pt);
    }
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      nsubdiv = 1;
      return 1;
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    virtual std::vector<double> get_outline(bool lagrangian);   
    virtual double get_quality_factor() { return 1.0; }
    virtual double s_min() const
    {
      return 0.0;
    }

    virtual double s_max() const
    {
      return 0.0;
    }
    virtual void set_integration_order(unsigned int order) {}
  };

  /////////////////////////////

  // Base class for all "interface" (face/boundary) elements: elements living on a face of a bulk
  // element (an oomph-lib FaceElement) that additionally carry their own JIT-generated residual
  // contributions (surface integrals: boundary conditions, interface physics, fluxes, ...) on top
  // of what their attached bulk element provides. An interface element can optionally be connected
  // to an "opposite side" interface element (e.g. the matching interface element on the other side
  // of an internal facet, or on a periodic/two-domain coupling), whose fields/coordinates it can
  // then access as external data - the opposite_side/opposite_node_index/opposite_orientation
  // members and analyze_opposite_orientation()/local_coordinate_in_opposite_side() (implemented per
  // concrete Interface*Element* subclass below) handle matching up local node/coordinate
  // conventions between the two potentially differently-oriented/refined element types.
  class InterfaceElementBase : public virtual BulkElementBase, public virtual oomph::SolidFaceElement
  {
  protected:
    InterfaceElementBase *opposite_side;
    bool Is_internal_facet_opposite_dummy;
    std::vector<int> opposite_node_index;
    int opposite_orientation;
    std::vector<int> bulk_eqn_map, opp_interf_eqn_map, opp_bulk_eqn_map, bulk_bulk_eqn_map;
    std::vector<bool> external_data_is_geometric;


    // Mapping for the additional interface dof ID to a map of master node to local equation number for the hanging-node constraints of that dof
    std::map<unsigned,std::map<pyoomph::BoundaryNode*, int>> Local_interface_hang_eqn;

    // Re-derives hanging-node values for the interface-only additional fields (in addition to what
    // BulkElementBase::interpolate_hang_values already does for the inherited bulk fields).
    virtual void interpolate_hang_values_at_interface();
    // Assigns local equation numbers for the hanging-node constraints of one interpolation space's
    // "additional" (interface-only) fields; addfields/basebulk_offset/nnode/hangindex/fieldnames
    // describe which fields and how many nodes are involved, node_index_to_element maps a node to
    // its element-local index for this space, and add_interf_local_hang_eqs caches already-assigned
    // hanging equation numbers per master node to avoid duplicating equations.
    virtual void assign_hanging_additional_interface_local_equations_for_space(const bool &store_local_dof_pt,JITFuncSpec_Table_FiniteElement_SpaceInfo_t * space);
    // Rebuilds the mapping from a source element's (bulk_indicator selects which "role": this
    // element's own bulk element, the opposite interface element, or the opposite's bulk element)
    // local equation numbers into this interface element's local equation numbers, as stored in
    // eqn_map; needed because generated code addresses external data uniformly regardless of which
    // element it actually lives on.
    virtual void update_equation_remapping_from_element(BulkElementBase *source_elem,const JITFuncSpec_RequiredShapes_FiniteElement_t *required_shapes,std::vector<int> &eqn_map,int bulk_indicator);
    virtual void update_in_external_fd(const unsigned &i);
    // Registers "data" (a Data object, e.g. from the bulk or opposite element) as required external
    // data of this element if not already present; is_geometric marks it as a solid/ALE position
    // dof (relevant for how its Jacobian contribution is computed). Returns true if newly added.
    virtual bool add_required_ext_data(oomph::Data *data, bool is_geometric);
    // Walks the "required_shapes" description generated alongside the JIT code and adds all Data
    // (nodal/internal/external) of from_elem that this interface element's generated residual code
    // needs to access as external data.
    virtual void add_required_external_data(JITFuncSpec_RequiredShapes_FiniteElement_t *required, BulkElementBase *from_elem);
    virtual void prepare_shape_buffer_for_integration(const JITFuncSpec_RequiredShapes_FiniteElement_t &required_shapes, unsigned int flag);
    double fill_shape_info_at_s(const oomph::Vector<double> &s, const unsigned int &index, const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, double &JLagr, unsigned int flag, oomph::DenseMatrix<double> *dxds = NULL, unsigned history_index=0) const;
    virtual bool fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap);
    // Additional interface-only hanging-node bookkeeping, used by InterfaceElementBase to handle
    // the extra fields that exist only on the interface element and not on the bulk element.
    bool fill_hang_info_with_equations_interface(JITShapeInfo_t *shape_info) override;
    virtual void ensure_external_data();
    virtual void assign_additional_local_eqn_numbers();
    virtual void fill_in_jacobian_from_lagragian_by_fd(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian);
    // Allocates the additional (interface-only, beyond the inherited bulk) field dofs on this
    // element's nodes/internal data, based on the interface's own JIT function table.
    virtual void add_interface_dofs();
    // Interface-specific part of fill_element_info: rebuilds the JITElementInfo_t/JITShapeInfo_t
    // bookkeeping for the additional interface fields and the bulk_eqn_map/opposite-side equation
    // maps, complementing BulkElementBase::fill_element_info for the inherited bulk part.
    virtual void fill_element_info_interface_part(bool without_equations=false);
    virtual std::vector<std::string> get_dof_names(bool not_a_root_call = false);
    virtual void get_dnormal_dcoords_at_s(const oomph::Vector<double> &s, double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT dnormal_dcoord, double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT d2normal_dcoord2) const;

    // Maps a local coordinate s on this interface element to the corresponding local coordinate on
    // the opposite_side interface element, accounting for possibly different node/edge orientation
    // and (for non-conforming "internal facet" pairings) different parametrization ranges. Must be
    // implemented per concrete Interface*Element* subclass, since the mapping depends on the face
    // geometry (line/triangle/quad) and node ordering conventions.
    virtual oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const { throw_runtime_error("Implement"); }
    virtual void fill_opposite_node_indices(JITShapeInfo_t *shape_info)
    {
      for (unsigned int i = 0; i < opposite_node_index.size(); i++)
      {
        shape_info->opposite_node_index[i] = opposite_node_index[i];
      }
    }
    // Determines opposite_orientation and opposite_node_index by matching this element's vertex
    // nodes to those of opposite_side (within a small tolerance, allowing for the "offset" vector
    // e.g. on periodic domains), so that fields on the two sides can be looked up consistently.
    // Must be implemented per concrete Interface*Element* subclass (the matching logic - which
    // permutations of nodes to try - depends on the face's geometric shape).
    virtual void analyze_opposite_orientation(const std::vector<double> & offset) { throw_runtime_error("Implement"); }
    // Adds the discontinuous-Galerkin (DG) field data of the attached bulk element as external data
    // of this interface element, so generated interface code can access DG fields of the bulk domain.
    virtual void add_DG_external_data();
    // Initializes a newly created additional-dof value (at local node lnode, value index valindex,
    // in the given interpolation space) by interpolating from already-existing data, used when new
    // interface dofs are created (e.g. after mesh refinement) and need sensible initial values.
    virtual void interpolate_newly_constructed_additional_dof(const unsigned &lnode, const unsigned &valindex, const std::string &space);


    virtual void assign_hanging_additional_interface_local_equations(const bool &store_local_dof_pt);
    // After the base class assigns local equation numbers for the inherited bulk nodal data,
    // additionally assigns local equation numbers for the interface-only additional fields' hanging-node constraints.
    inline void assign_nodal_local_eqn_numbers(const bool &store_local_dof_pt)
    {
      BulkElementBase::assign_nodal_local_eqn_numbers(store_local_dof_pt);
      assign_hanging_additional_interface_local_equations(store_local_dof_pt);
    }
  public:
    InterfaceElementBase() : opposite_side(NULL), Is_internal_facet_opposite_dummy(false) {}

    virtual int local_interface_hang_eqn(unsigned int interface_dof_index, oomph::Node * master_node) const;  
    void fill_in_jacobian_from_nodal_by_fd(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian) override;
    static bool interpolate_new_interface_dofs;
    // Public entry point to refresh all the eqn_map bookkeeping (bulk_eqn_map,
    // opp_interf_eqn_map, opp_bulk_eqn_map, bulk_bulk_eqn_map) after equation numbers have changed
    // (e.g. following mesh refinement or re-numbering), by calling
    // update_equation_remapping_from_element for each relevant source element.
    virtual void update_equation_remapping();
    virtual void set_remaining_shapes_appropriately(JITShapeInfo_t *shape_info, const JITFuncSpec_RequiredShapes_FiniteElement_t &required_shapes);
    void pin_dummy_values() override;
    void unpin_Dirichlet_dofs_for_matrix_manipulation(DirichletMatrixManipulationInfo & info) override;

    void set_as_internal_facet_opposite_dummy() { Is_internal_facet_opposite_dummy = true; }
    bool is_internal_facet_opposite_dummy() const { return Is_internal_facet_opposite_dummy; }

    // Returns the local-to-global (or local-to-local, depending on "which": "bulk"/"opposite_interface"/...)
    // equation number mapping for the requested attached element role, for introspection/debugging.
    std::vector<int> get_attached_element_equation_mapping(const std::string &which);
    // Connects this interface element to _opposite_side as its opposite-side partner (see class
    // comment above), wiring up the required external data for merged/opposite shape requirements
    // from the JIT function table and determining the node/orientation correspondence (with an
    // optional periodic "offset" applied before matching coordinates).
    void set_opposite_interface_element(BulkElementBase *_opposite_side,std::vector<double>  offset)
    {
      if (_opposite_side && !dynamic_cast<InterfaceElementBase *>(_opposite_side))
      {
        throw_runtime_error("Can only set an Interface Element as the opposite side of and interface element");
      }
      opposite_side = dynamic_cast<InterfaceElementBase *>(_opposite_side);
      const JITFuncSpec_Table_FiniteElement_t *functable = this->codeinst->get_func_table();

      if (functable->merged_required_shapes.opposite_shapes)
      {
        // std::cout << "INTERFACE ELEM MERGED " << functable->merged_required_shapes.opposite_shapes->psi_D0 << std::endl;
        add_required_external_data(functable->merged_required_shapes.opposite_shapes, dynamic_cast<BulkElementBase *>(opposite_side));
        if (functable->merged_required_shapes.opposite_shapes->bulk_shapes)
        {
          //        std::cout << "INTERFACE ELEM MERGED BULK " <<  functable->merged_required_shapes.opposite_shapes->bulk_shapes->psi_D0 << std::endl;
          add_required_external_data(functable->merged_required_shapes.opposite_shapes->bulk_shapes, dynamic_cast<BulkElementBase *>(dynamic_cast<InterfaceElementBase *>(opposite_side)->bulk_element_pt()));
        }
      }

      this->eleminfo.opposite_eleminfo = &(opposite_side->eleminfo);
      std::vector<double> offs=offset;
      for (unsigned int i=offset.size();i<this->nodal_dimension();i++) offs.push_back(0.0);
      this->analyze_opposite_orientation(offs);
    }

    double zeta_nodal(const unsigned &n, const unsigned &k, const unsigned &i) const
    {
      return oomph::FaceElement::zeta_nodal(n, k, i);
    }

    // Finds the local coordinate s on this element whose Eulerian position best matches the given
    // global coordinate x (a local Newton/optimization search), used e.g. to locate the opposite-side
    // local coordinate for partially-overlapping ("internal facet") interface pairings.
    virtual oomph::Vector<double> optimize_s_to_match_x(const oomph::Vector<double> &x);

    // Returns the node on the opposite-side interface element corresponding to local node i of this
    // element, or NULL if there is no opposite side or no corresponding node (e.g. for a
    // lower-to-higher order mismatch).
    virtual pyoomph::Node *opposite_node_pt(unsigned int i)
    {
      if (!opposite_side || opposite_node_index[i] < 0)
        return NULL;
      return dynamic_cast<pyoomph::Node *>(opposite_side->node_pt(opposite_node_index[i]));
    }
    InterfaceElementBase *get_opposite_side() { return opposite_side; }
    const InterfaceElementBase *get_opposite_side() const { return opposite_side; }

    virtual int get_nodal_index_by_name(oomph::Node *n, std::string fieldname);
    // Evaluates the interface-only field "ifindex" (in the given interpolation space) at local
    // coordinate s and history index t, by interpolating from the interface's additional dof data.
    virtual double get_interpolated_interface_field(const oomph::Vector<double> &s, const unsigned &ifindex, const std::string &space, const unsigned &t = 0) const;

    unsigned get_DG_buffer_index(const unsigned &space_index, const unsigned &fieldindex) override;    
    unsigned get_DG_node_index(const unsigned & space_index, const unsigned &fieldindex, const unsigned &nodeindex) const override;
    oomph::Data *get_DG_nodal_data(const unsigned & space_index,const unsigned &fieldindex) override;
    int get_DG_local_equation(const unsigned &space_index, const unsigned &fieldindex, const unsigned &nodeindex) override;
    
  };

  // Generic template that turns any bulk element type BASE (e.g. BulkElementTri2dC1) into the
  // corresponding face/interface element, by combining BASE (as the FaceElement's own geometric
  // shape, since oomph-lib builds a FaceElement's geometry from its bulk element's face) with
  // InterfaceElementBase (the JIT-driven interface residual machinery). Virtual functions that
  // exist on both bases (hanging-node handling, hanging-value interpolation) are combined by
  // calling both parents. Concrete Interface*Element* classes below instantiate this template for
  // each bulk element type and add the geometry-specific opposite-side matching logic.
  template <class BASE>
  class InterfaceElement : public virtual BASE, public virtual InterfaceElementBase
  {
  protected:
    virtual bool fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap)
    {
      bool res1 = BASE::fill_hang_info_with_equations(required, shape_info, eqn_remap);
      bool res2 = InterfaceElementBase::fill_hang_info_with_equations(required, shape_info, eqn_remap);
      return res1 || res2;
    }


    virtual void interpolate_hang_values()
    {
      BASE::interpolate_hang_values();
      this->interpolate_hang_values_at_interface();
    }

  public:
    double zeta_nodal(const unsigned &n, const unsigned &k, const unsigned &i) const
    {
      return oomph::FaceElement::zeta_nodal(n, k, i);
    }

    virtual void fill_element_info(bool without_equations=false)
    {
      BASE::fill_element_info(without_equations);
      this->fill_element_info_interface_part(without_equations);
      if (this->nnode())
      {
        oomph::TimeStepper *tstepper = this->node_pt(0)->time_stepper_pt();
        for (unsigned int i = 0; i < this->ninternal_data(); i++)
        {
          this->internal_data_pt(i)->set_time_stepper(tstepper, true);
        }
      }
    }

    // Builds this face element from face "face_index" of bulk_el_pt (which must be built from the
    // JIT code instance jitcode): sets up the shared geometry via oomph-lib's build_face_element,
    // wires up the interface's own dofs and required external data (including the bulk element's
    // data, and - if the interface's dominant space is higher order than the bulk's - rejects the
    // combination since bulk fields could not represent the interface's higher-order dofs).
    InterfaceElement(DynamicBulkElementInstance *jitcode, FiniteElement *const &bulk_el_pt, const int &face_index)
    {
      bulk_el_pt->build_face_element(face_index, this);
      this->codeinst = jitcode;
      this->eleminfo.bulk_eleminfo = dynamic_cast<BulkElementBase *>(bulk_el_pt)->get_eleminfo();
      this->add_interface_dofs();
      const JITFuncSpec_Table_FiniteElement_t *functable = this->get_code_instance()->get_func_table();

      const JITFuncSpec_Table_FiniteElement_t *bfunctable = dynamic_cast<BulkElementBase *>(bulk_el_pt)->get_code_instance()->get_func_table();
      
      if (std::string(functable->dominant_space) == "C2")
      {      
        if (std::string(bfunctable->dominant_space) == "C1")
        {
          throw_runtime_error("Cannot attach an interface element with C2 fields to a parent domain with max. C1 space");
        }
      }
      //      std::cout << "ADDING INTERFACE ELEM EXTERNAL DATA " << this->nexternal_data() << std::endl;
      this->flush_external_data();
      //      std::cout << "FLUSING EXTERNAL DATA " << this->nexternal_data() << std::endl;
      this->add_DG_external_data();
      //      std::cout << "DONE ADDING INTERFACE ELEM DG DATA " << this->nexternal_data() << std::endl;

      for (auto &e : this->codeinst->get_linked_external_data().get_required_external_data())
      {
        //        std:: cout << "ADDING ED0 " << std::endl;
        this->add_required_ext_data(e, false);
      }
      //      std::cout << "DONE ADDING INTERFACE ELEM ED0 DATA " << this->nexternal_data() << std::endl;

      if (functable->merged_required_shapes.bulk_shapes)
      {
        //	  std::cout << "ADDING BULK EXT DATA" << std::endl;
        add_required_external_data(functable->merged_required_shapes.bulk_shapes, dynamic_cast<BulkElementBase *>(bulk_el_pt)); // TODO: Also the others? (is it necessary e.g. spatial integration of the stress along interface)
        if (functable->merged_required_shapes.bulk_shapes->bulk_shapes)
        {
          InterfaceElementBase *ip = dynamic_cast<InterfaceElementBase *>(bulk_el_pt);
          add_required_external_data(functable->merged_required_shapes.bulk_shapes->bulk_shapes, dynamic_cast<BulkElementBase *>(ip->bulk_element_pt()));
        }
      }
    }

    virtual void get_normal_at_s(const oomph::Vector<double> &s, oomph::Vector<double> &n, double *  PYOOMPH_RESTRICT  *  PYOOMPH_RESTRICT *  PYOOMPH_RESTRICT  dnormal_dcoord, double *  PYOOMPH_RESTRICT  *  PYOOMPH_RESTRICT  *  PYOOMPH_RESTRICT  *  PYOOMPH_RESTRICT *  PYOOMPH_RESTRICT d2normal_dcoord2) const
    {
      this->outer_unit_normal(s, n);
      if (dnormal_dcoord)
      {
        this->get_dnormal_dcoords_at_s(s, dnormal_dcoord, d2normal_dcoord2);
      }
    }
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
    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const
    {
      return s;
    }
    // A point has no orientation ambiguity; just checks the opposite side is also a point element.
    void analyze_opposite_orientation(const std::vector<double> & offset)
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

    virtual pyoomph::Node *opposite_node_pt(unsigned int i)
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
    void analyze_opposite_orientation(const std::vector<double> & offset)
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
    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const
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

    virtual pyoomph::Node *opposite_node_pt(unsigned int i)
    {
      if (partial_opposite_internal_facet)
        throw_runtime_error("opposite_node_pt not allowed in internal facets with partial overlap with the opposite side");
      return InterfaceElement<BulkElementLine1dC2>::opposite_node_pt(i);
    }

    //  void further_setup_hanging_nodes() {} //TODO: REM
    void analyze_opposite_orientation(const std::vector<double> & offset)
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

    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const
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
    void analyze_opposite_orientation(const std::vector<double> & offset)
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

    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const
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
    void analyze_opposite_orientation(const std::vector<double> & offset)
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

    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const
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

  // Quadrilateral (2d) interface element on a brick bulk element's C1 face; does not override
  // analyze_opposite_orientation/local_coordinate_in_opposite_side (2d face-to-face matching for
  // quad faces is presently not implemented beyond the base class's default).
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
    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const
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
    void analyze_opposite_orientation(const std::vector<double> & offset)
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

    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const
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
    void analyze_opposite_orientation(const std::vector<double> & offset)
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

    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const
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

    void analyze_opposite_orientation(const std::vector<double> & offset)
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


  extern double *__replace_RJM_by_param_deriv;
}
