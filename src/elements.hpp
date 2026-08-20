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
#include <chrono>
#include "exception.hpp"
#include "jitbridge.h"

#include "oomph_lib.hpp"

#include "refineable_brick_element.h"

#include "refineable_telements.hpp"
#include "wedges_and_pyramids.hpp"
#include "refinement_pattern.hpp"
#include "problem.hpp"


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

  // --- Assembly-overhead campaign, stage 0 diagnostics ---
  //
  // Three measurement-only levers around the per-element hang bookkeeping, which runs in full for
  // every element on every assembly (the outlook item of dev_docs/code_generation.md 9.4.14).
  //
  // PYOOMPH_MEASURE_SKIP_HANG_FILLS turns fill_hang_info_with_equations and interpolate_hang_values
  // into early returns, to put a ceiling on what removing them could ever save. MEASUREMENT ONLY and
  // unsound in general - the generated code then reads a stale hangbuffer. It is bitwise-safe exactly
  // on meshes with no hanging nodes, no additional dof constraints and no dummy-value maps, which is
  // the configuration the ceiling is measured in. The attached-element eqn_remap channel is
  // deliberately NOT skipped: fill_hang_info_with_equations abuses the very same buffers for it
  // (codegen.cpp, "REMAP channel"), and skipping that yields garbage local equation numbers - a
  // crash rather than a stale-but-plausible number.
  extern const bool __measure_skip_hang_fills;

  // PYOOMPH_REPORT_HANG_FILL_TIME accumulates wall time and call counts of both fills and of the
  // interface neighbour re-interpolation, reported at exit. Same motivation as
  // PYOOMPH_REPORT_NOHANG_DISPATCH: a share that was never measured is not thereby a small share.
  extern const bool __report_hang_fill_time;
  enum HangFillTimeSlot
  {
    HANGFILL_SLOT_FILL = 0,            // fill_hang_info_with_equations (incl. its bulk/opposite recursion)
    HANGFILL_SLOT_INTERP = 1,          // interpolate_hang_values on the element being assembled
    HANGFILL_SLOT_NEIGHBOUR = 2,       // the same, re-run on the attached bulk/opposite elements
    HANGFILL_NUM_SLOTS = 3
  };
  void __hang_fill_time_add(int slot, double seconds);
  extern unsigned __hang_fill_time_depth;

  // RAII accumulator. Only the OUTERMOST scope is timed: the fills recurse (interface -> bulk ->
  // bulk's bulk) and the InterfaceElement template calls both of its parents, so accumulating per
  // entry would count the same nanoseconds several times over. The neighbour block is entered at
  // depth 0 and therefore lands in its own slot rather than in the element's own.
  class HangFillTimeScope
  {
    int slot;
    std::chrono::steady_clock::time_point t0;

  public:
    explicit HangFillTimeScope(int s) : slot(-1)
    {
      if (!__report_hang_fill_time || __hang_fill_time_depth++)
        return;
      slot = s;
      t0 = std::chrono::steady_clock::now();
    }
    ~HangFillTimeScope()
    {
      if (!__report_hang_fill_time)
        return;
      __hang_fill_time_depth--;
      if (slot < 0)
        return;
      __hang_fill_time_add(slot, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
  };

  // --- Assembly-overhead campaign, stage 3: hang bookkeeping caches ---
  //
  // PYOOMPH_DISABLE_HANG_FILL_CACHE restores the pre-stage-3 behaviour entirely (fill always run,
  // interpolate always run, no classification, no pass dedupe), so an A/B comparison of the whole
  // stage can be made with ONE binary - the pattern PYOOMPH_DISABLE_NOHANG_DISPATCH established.
  extern const bool __disable_hang_fill_cache;
  // PYOOMPH_REPORT_HANG_FILL_CACHE prints, at exit, how often each of the two mechanisms engaged. A
  // "no difference" measurement with zero engagement proves nothing, so the counters are reported
  // alongside every benchmark.
  extern const bool __report_hang_fill_cache;
  // PYOOMPH_PARANOID_HANG_FILL_CACHE is the self-test: it runs the full fill on every element whose
  // fill was skipped and aborts if its return value disagrees with the cheap predicate, then poisons
  // the hang buffers of skipped elements so that a NoHang body that DOES read one is caught; and it
  // checks the "nothing to interpolate" classification against the nodes themselves.
  extern const bool __paranoid_hang_fill_cache;
  enum HangFillCacheCounter
  {
    HANGCACHE_FILL_SKIPPED = 0,
    HANGCACHE_FILL_RUN,
    HANGCACHE_INTERP_SKIPPED_BY_CLASS,
    HANGCACHE_INTERP_SKIPPED_BY_STAMP,
    HANGCACHE_INTERP_RUN,
    HANGCACHE_NUM_COUNTERS
  };
  void __hang_fill_cache_count(int which);

  // Id of the global assembly sweep currently running, or 0 for "none". interpolate_hang_values()
  // re-derives values that are constant for the whole sweep, so within one pass the second and
  // further calls for the same element (an interface re-interpolating its bulk neighbour, then that
  // bulk element assembling itself, then the interface on its other side) are redundant. 0 is the
  // safe default: every caller that is NOT inside a sweep interpolates unconditionally.
  extern unsigned long __hang_interp_pass;
  extern unsigned long __hang_interp_pass_counter;

  // Opens a new pass for its lifetime. Nesting is fine (the previous id is restored).
  class HangInterpPassScope
  {
    unsigned long prev;

  public:
    HangInterpPassScope() : prev(__hang_interp_pass) { __hang_interp_pass = ++__hang_interp_pass_counter; }
    ~HangInterpPassScope() { __hang_interp_pass = prev; }
  };

  // Closes the pass for its lifetime, i.e. restores unconditional interpolation. Wrapped around
  // every finite-difference loop: those perturb a dof and re-enter get_residuals per perturbation,
  // so the hang values genuinely change within one enclosing sweep. On exit a FRESH pass id is
  // opened rather than the old one restored, because the FD loop left interpolated values belonging
  // to the last perturbed state in the nodes of this element and its neighbours; a stale stamp would
  // keep them.
  class HangInterpPassSuspension
  {
    unsigned long prev;

  public:
    HangInterpPassSuspension() : prev(__hang_interp_pass) { __hang_interp_pass = 0; }
    ~HangInterpPassSuspension() { __hang_interp_pass = (prev ? ++__hang_interp_pass_counter : 0); }
  };

  class BulkElementBase;
  // Depth-guarded gate in front of interpolate_hang_values(). Only the OUTERMOST call may skip: the
  // InterfaceElement override calls its bulk base, and gating there a second time would run the
  // interface half without the bulk half.
  class HangInterpGate
  {
    BulkElementBase *el;
    bool outer;

  public:
    explicit HangInterpGate(BulkElementBase *e);
    ~HangInterpGate();
    bool skip();
  };

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
  class InterfaceElementBase;
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

    // Set by pin_dummy_values (and its InterfaceElementBase override) whenever at least one node of
    // this element carries an additional dof constraint (see
    // NodeWithFieldIndicesBase::add_additional_dof_constraint), reset by unpin_dummy_values. Lets
    // fill_additional_hang_buffer_data/interpolate_hang_values skip their (otherwise per-node)
    // additional-dof-constraint loops entirely for the common case where none are present.
    bool has_additional_dof_constraints = false;

    // Releases/resets the JITElementInfo_t buffers owned by this element (nodal/external/internal
    // data pointers etc. handed to the generated code).
    void free_element_info();

  public:
    // Which of oomph-lib's geometric shape families this element belongs to. The four 3d families are
    // mutually exclusive (a wedge/pyramid is neither a BrickElementBase nor a TElementBase), as are
    // quad and simplex in 2d.
    enum ElementFamily
    {
      EF_UNKNOWN = 0, // not yet determined - never returned by element_family()
      EF_QUAD,        // oomph::QuadElementBase - 2d quadrilateral
      EF_SIMPLEX,     // oomph::TElementBase - triangle or tetrahedron
      EF_BRICK,       // oomph::BrickElementBase - 3d hexahedron
      EF_WEDGE,       // oomph::RefineableWedgeElement
      EF_PYRAMID,     // oomph::RefineablePyramidElement
      EF_OTHER        // none of the above (a line element, a point/ODE element, ...)
    };

    // Cached: determining the family means dynamic_cast-ing across the virtual-inheritance diamond,
    // and the mesh-level refinement_possible() asks it once per element on every call (several times
    // per adaptation). An element never changes shape, so one cast per element object is enough.
    ElementFamily element_family() const;

    // Is this element an InterfaceElementBase? Overridden there to return `this`; the base returns
    // NULL. Every element is a BulkElementBase, so this replaces dynamic_cast<InterfaceElementBase*>
    // at every call site that already holds one. That cast is NOT cheap: the whole element hierarchy
    // is joined by virtual inheritance (see ElementBase/FiniteElementBase above), so it takes
    // libstdc++'s __vmi_class_type_info graph walk - measured at ~1900 cycles a call, and
    // fill_shape_info_element_sizes asks the question once per element per integration sweep.
    virtual InterfaceElementBase *as_interface_element() { return NULL; }
    const InterfaceElementBase *as_interface_element() const { return const_cast<BulkElementBase *>(this)->as_interface_element(); }

  protected:
    // See element_family(). EF_UNKNOWN until first asked.
    mutable ElementFamily element_family_cache = EF_UNKNOWN;

    // Allocates internal/external Data for fields stored discontinuously per element (D0: constant,
    // DL: discontinuous-Lagrange, DG: discontinuous on a sub-space) rather than as ordinary nodal data.
    virtual void allocate_discontinous_fields();
    // Allocates/sizes the shape-function value/derivative buffers inside shape_info according to
    // which shapes (psi, dpsi, hang info, ...) the generated code actually requires, before the
    // integration loop starts.
    virtual void prepare_shape_buffer_for_integration(const JITFuncSpec_RequiredShapes_FiniteElement_t &required_shapes, unsigned int flag);
    // Fills in element-size-related entries (e.g. element diameter) of the shape buffer that are
    // needed by the generated code but do not depend on the integration point.
    virtual void fill_shape_info_element_sizes(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, unsigned flag, unsigned history_index = 0) const;
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

    // Nodal-coordinate sensitivities of the ingredients of the second spatial derivative, i.e. of
    // M_i^(b) = g^{ab} t_{a,i} and of Q[i][b][c] = dM_i^(b)/ds_c (see fill_shape_info_at_s). Both are
    // explicit algebraic functions of the nodal positions, and everything they are built from
    // (Psi_{m,a}, Psi_{m,ac}, t, X_{i,ab}, g^{ab}) is already available at that point. Written for
    // the general metric form, so it is valid with a codimension as well.
    //
    // Outputs are flat arrays with the index helpers documented at the definition:
    //   dM_dX  [i_dim][b_eldim][m_node][p_dim]
    //   dQ_dX  [i_dim][b_eldim][c_eldim][m_node][p_dim]
    //   d2M_dXdX [i_dim][b_eldim][m_node][p_dim][m2_node][p2_dim]           (only if non-NULL)
    //   d2Q_dXdX [i_dim][b_eldim][c_eldim][m_node][p_dim][m2_node][p2_dim]   (only if non-NULL)
    void fill_d2x_dNodalPos_helper(unsigned n_node, unsigned n_dim, unsigned el_dim,
                                   const oomph::DenseMatrix<double> &interpolated_t,
                                   const oomph::DShape &dpsids_Element, const oomph::DShape &d2psids_Element,
                                   const oomph::DenseMatrix<double> &aup, const double (*Xkab)[MAX_N2DERIV],
                                   const double (*dgab_ds)[3][3],
                                   std::vector<double> &dM_dX, std::vector<double> &dQ_dX,
                                   std::vector<double> *d2M_dXdX = NULL, std::vector<double> *d2Q_dXdX = NULL) const;
    // Finite-difference fallback for the Jacobian contribution from the Lagrangian (undeformed
    // solid) position degrees of freedom, used where an analytic derivative is not available.
    virtual void fill_in_jacobian_from_lagragian_by_fd(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian);
    // Computes the derivative of the (interface/boundary) outer unit normal with respect to the
    // nodal coordinates (and optionally its second derivative), required by generated code that
    // differentiates normal-dependent boundary conditions w.r.t. the moving mesh position.
    virtual void get_dnormal_dcoords_at_s(const oomph::Vector<double> &s, double *  PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT dnormal_dcoord, double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT d2normal_dcoord2) const;
    void update_in_solid_position_fd(const unsigned &i) override; // For FD with element_sizes, we have to update the element size buffer
    // --- Named hanging-node accessors (see dev_docs/adaptive_refinement.md) ---
    // Single, auditable point where pyoomph maps an interpolation space to the oomph-lib HangInfo
    // that governs it at a given element-local node (or NULL if that node does not hang in that
    // space). Today these just resolve the per-space `hangindex` convention that codegen writes into
    // the func table (info_Pos.hangindex==-1 for the geometric slot; C2TB/C2 share it; C1TB/C1 either
    // share it or use their own value slot). Later phases will redirect these to pyoomph-owned named
    // HangInfo pointers (geometric / C2(TB) / C1(TB), plus interface/constraint variants) without
    // having to touch the assembly call sites that go through here.
    oomph::HangInfo *hang_info_for_space(const JITFuncSpec_Table_FiniteElement_SpaceInfo_t *space_info, unsigned l_elem) const;
    // Status-only companion of hang_info_for_space: does element-local node l_elem hang in this
    // space? Equivalent to hang_info_for_space(...) != NULL but avoids implying a HangInfo is wanted.
    bool node_hangs_in_space(const JITFuncSpec_Table_FiniteElement_SpaceInfo_t *space_info, unsigned l_elem) const;
    // Geometric (positional) hanging for the given element-local node, i.e. the info_Pos slot.
    oomph::HangInfo *hang_info_for_position(unsigned l_elem) const;

    // --- Flattened hang/constraint composition (see dev_docs/adaptive_refinement.md section 3) ---
    // Does node n carry a CONTINUOUS_BASE_DOF_CONSTRAIN_TO_C1 constraint for value index v?
    bool node_is_c1_constrained_for_value(oomph::Node *n, unsigned v) const;
    // Does node n carry a POSITION_CONSTRAIN_TO_C1 constraint for coordinate index i?
    bool node_is_c1_constrained_for_position(oomph::Node *n, unsigned i) const;
    // Local equation number of a genuine free leaf node's value v as seen by this element: its nodal
    // local eqn if it is one of this element's nodes, else its (hang-registered) local_hang_eqn.
    int leaf_local_eqn_for_value(oomph::Node *n, unsigned v);
    // As above but for the leaf's coordinate i (position dof).
    int leaf_local_eqn_for_position(oomph::Node *n, unsigned i);
    // Read a node's local position-hang equation number, throwing a diagnostic naming the node if it was
    // never registered as a position-hang master of this element (oomph's accessor would silently return an
    // empty matrix and hence a junk index). See the implementation comment.
    int position_hang_eqn_or_throw(oomph::Node *n, unsigned i, const std::string &context);
    // Leaf nodes that flatten_hang_for_position() would resolve to, starting from n (same recursion).
    void collect_position_leaf_nodes(oomph::Node *n, std::set<oomph::Node *> &out, int depth);
    // Register the C1-position-constraint leaves that oomph does not know are masters of this element.
    void register_c1_constraint_position_masters();

    // --- Cross-shape (mixed-mesh) hanging (topological, no geometry) ---
  public:
    // When true, fill_shape_buffer_for_integration_point() raises oomph::InvertedElementError as soon
    // as the SIGNED determinant of the Eulerian mapping dx/ds at an integration point is not strictly
    // positive, i.e. the element has turned inside out or collapsed. Note this is deliberately not the
    // J that the same function turns into dx: that one is sqrt(det(g_ab)), which is what lets pyoomph
    // integrate over elements of lower dimension than the nodal space, and which is non-negative by
    // construction and blind to orientation. Consequently the check only applies where the mapping is
    // square (element dimension == nodal dimension); interface elements have no orientation to lose
    // and are skipped. The adaptive time stepper and the arclength continuation in oomph-lib catch the
    // exception and retry with a smaller step, which is the whole point: on a moving mesh an inverted
    // element is normally a symptom of too large a step, not of an ill-posed problem. Global rather
    // than per-Problem, following use_eigen_error_estimators and interpolate_new_interface_dofs; off
    // by default, since without a catching solver an inversion would turn a survivable garbage step
    // into an abort.
    static bool detect_inverted_elements;

    // Hang node X (one of THIS fine element's edge interpolating nodes for value_id) on the strictly
    // COARSER neighbour nb_re of a DIFFERENT shape, given X's fraction t along the shared coarse edge
    // whose real corner nodes are Pb (t=0) and Qb (t=1). The neighbour-local coordinate is the affine
    // blend (1-t)*s(Pb in nb) + t*s(Qb in nb) of the neighbour's local coords of the shared corner nodes
    // (pure topology -- node indices + local coords, no positions/locate_zeta) and the master weights come
    // from the neighbour's interpolating_basis. Shape-agnostic; the 3D analogue blends 3/4 face-corner
    // nodes barycentrically. Skips X if the neighbour already owns an interpolating node there, and guards
    // cyclic hangs. Public because the triangle hang path (RefineableTElement<2>) calls it too. Returns
    // true iff a hang was installed.
    bool mixed_hang_edge_node(oomph::Node *X, oomph::Node *Pb, oomph::Node *Qb, double t, oomph::RefineableElement *nb_re, const int &value_id);
    // Given a coordinate in THIS QUAD's ROOT-element local frame ([-1,1]^2), descend the QuadTree to the
    // leaf that contains it (topological axis-aligned son-box descent, no geometry) and return that leaf's
    // node at the coordinate, or null. The quad counterpart of RefineableTElement<2>::node_at_root_coordinate,
    // used for cross-shape node-sharing (a tri finds the coincident node an adjacent refined quad built).
    oomph::Node *quad_node_at_root_coordinate(const oomph::Vector<double> &s_root);
    // As above but returns the LEAF element and the coordinate in its own frame (for cross-shape HANGING:
    // a tri hangs on this quad leaf's interpolating_basis at s_leaf). null if not resolvable.
    oomph::RefineableElement *quad_leaf_at_root_coordinate(const oomph::Vector<double> &s_root, oomph::Vector<double> &s_leaf);
    // Hang core: hang node X on neighbour nb_re's interpolating_basis at nb-local coordinate s_nb (shared
    // skip + cycle guard). The shape/dimension-agnostic primitive shared by all cross-shape hang paths.
    bool mixed_hang_node_at(oomph::Node *X, oomph::RefineableElement *nb_re, const oomph::Vector<double> &s_nb, const int &value_id);

    // --- Generic per-value ("interpolating node") hooks for MIXED-ORDER spaces --------------------------
    // oomph-lib's defaults are isoparametric: ninterpolating_node()==nnode() and interpolating_basis()==
    // shape(). That is wrong the moment a C1 field lives on a C2-geometry element -- a hanging constraint
    // for such a field must be built from the LINEAR basis on the corner vertices, not from the quadratic
    // basis on all nodes -- and it is silently wrong, because the mismatch also leaves the tail of the
    // Shape array UNINITIALISED (oomph::Shape allocates with new double[N], which does not zero).
    // BulkElementBrick3dC2 / BulkElementQuad2dC2 / BulkElementTri2dC2 each hand-roll these overrides; the
    // helpers below give the same behaviour shape-agnostically, so the remaining C2 element families
    // (tet, wedge, pyramid) can share one implementation instead of repeating index arithmetic per shape.
    // value_id < 0 means the geometry/position, which always uses the geometric basis.
    bool interpolation_value_is_C1(const int &value_id) const;
    unsigned generic_ninterpolating_node(const int &value_id);
    oomph::Node *generic_interpolating_node_pt(const unsigned &n, const int &value_id);
    void generic_interpolating_basis(const oomph::Vector<double> &s, oomph::Shape &psi, const int &value_id) const;
    oomph::Node *generic_get_interpolating_node_at_local_coordinate(const oomph::Vector<double> &s, const int &value_id);
    // Tesselated-numpy export, quad fine side: for each of this quad's 4 edges use gteq_edge_neighbour to find
    // a strictly coarser neighbour and register this quad's edge nodes on it, computing each node's coordinate
    // in the coarse neighbour TOPOLOGICALLY -- via gteq_edge_neighbour's own translate_s/s_lo/s_hi mapping for
    // a coarser QUAD, or (mixed interface) the shared-root-edge blend + leaf_at_root_coordinate for a coarser
    // TRI. No physical positions / locate_zeta. Shared by BulkElementQuad2dC1/C2::inform_coarser_*.
    void quad_register_on_coarser_for_numpy(std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes);

  protected:
    // Local face index -> mesh boundary indices this face lies on. See the accessors
    // get_face_boundaries()/set_face_boundaries() below for what this is and how it is maintained.
    // Empty for the (vast majority of) elements that do not touch any boundary.
    std::map<short, std::vector<unsigned>> face_boundaries;
    // Cross-shape node-sharing on THIS quad's side: for the son node at fractional coordinate s_fraction on
    // an edge shared with a TRI, map topologically into the tri ROOT frame and descend to the tri leaf. The
    // quad node_created_by_neighbour override calls this after oomph's own quad<->quad sharing returns null.
    oomph::Node *mixed_quad_shared_node(const oomph::Vector<double> &s_fraction);
    // Called from the overridden quad_hang_helper when the edge my_edge (N/S/E/W) has a strictly coarser
    // cross-shape (triangular) neighbour coarse_nb: enumerate this quad's edge interpolating nodes for
    // value_id, ascend each to the quad's root frame (axis-aligned son boxes -> the edge keeps its compass
    // direction, so the root-edge corners are the coarse edge corners Pb,Qb), get its fraction t along that
    // edge, and hang it via mixed_hang_edge_node.
    void mixed_quad_edge_hang(const int &value_id, const int &my_edge, oomph::RefineableElement *coarse_nb);
    // Flatten the value v of node n into a weighted sum over real free leaf dofs, accumulating
    // local_eqn -> weight into `out`. Composes genuine oomph hanging (n->hanging_pt(v)) with C1
    // dof-constraints (a constrained node is expanded via its stored c1_constraint_corners, then each
    // corner recursed), so ConstrainFieldsToC1Space works on adaptively refined meshes. depth guards
    // against runaway recursion (cyclic hang/constraint chains).
    void flatten_hang_for_value(oomph::Node *n, unsigned v, double weight, std::map<int, double> &out, int depth);
    // Position (geometric-hang) counterpart of flatten_hang_for_value for coordinate index i.
    void flatten_hang_for_position(oomph::Node *n, unsigned i, double weight, std::map<int, double> &out, int depth);
    // Value counterparts of the flatten_* routines used by interpolate_hang_values: instead of local
    // equation numbers they return the interpolated *value* (resp. coordinate) of node n at time level
    // t, by recursing (constrained -> C1 corners; hanging -> masters) down to real free leaf dofs whose
    // raw stored values are always current. This makes the pushed-back hanging/dummy/constrained
    // storage order-independent and consistent with the hangbuffer flattening.
    double flattened_value(oomph::Node *n, unsigned v, unsigned t, int depth);
    double flattened_position(oomph::Node *n, unsigned i, unsigned t, int depth);

    // Sets up the local-equation-number bookkeeping for hanging nodes on the Lagrangian/Eulerian
    // position degrees of freedom (as opposed to field values), needed on refined (non-conforming)
    // meshes with ALE/solid mechanics where the mesh position itself is a degree of freedom.
    virtual bool fill_hang_info_with_equations_for_pos(JITShapeInfo_t *shape_info);
    // Sets up hanging-node local-equation-number bookkeeping for the "base bulk" fields (i.e. not
    // the additional interface-only fields added by InterfaceElementBase, which extends this).
    virtual bool fill_hang_info_with_equations_basebulk(JITShapeInfo_t *shape_info);
    // Additional interface-only hanging-node bookkeeping, used by InterfaceElementBase to handle
    // the extra fields that exist only on the interface element and not on the bulk element.
    virtual bool fill_hang_info_with_equations_interface(JITShapeInfo_t *) {return false;}
    // Sets up a synthetic hanging-node scheme (masters + weights + local equation numbers) for the
    // base-bulk continuous-field dofs that were locally reduced to C1 via
    // NodeWithFieldIndicesBase::add_additional_dof_constraint (mode CONTINUOUS_BASE_DOF_CONSTRAIN_TO_C1),
    // so generated code redistributes their residual/Jacobian contributions to the C1 corner nodes.
    // Overridden by InterfaceElementBase to additionally handle INTERFACE_DOF_CONSTRAIN_TO_C1.
    virtual bool fill_additional_hang_buffer_data(JITShapeInfo_t *shape_info);

    // --- Stage 3: cached per-element hang classification ---
    //
    // The shape buffer is shared between all elements of a code (Default_shape_info_buffer), so the
    // hang TABLES cannot be cached there. What can be cached is the DECISION: whether anything in
    // this element hangs at all, and whether interpolate_hang_values() would write anything. Both
    // depend only on the mesh topology, the hang schemes and the dof constraints - never on dof
    // VALUES, which is why caching them is sound while caching any result would not be.
    //
    // Invalidated in fill_element_info(), next to local_dof_contribution_indices_valid: every
    // adapt/remesh/pin/constraint change reaches it through assign_eqn_numbers().
    enum HangStateBits
    {
      HANGSTATE_HAS_HANG = 1u,               // fill_hang_info_with_equations() would return true
      HANGSTATE_NOTHING_TO_INTERPOLATE = 2u  // interpolate_hang_values() would write nothing
    };
    unsigned char hang_state = 0;
    bool hang_state_valid = false;
    // Id of the assembly pass in which this element's hang values were last interpolated; 0 = never.
    unsigned long last_hang_interp_pass = 0;
    // Computed lazily on FIRST USE, deliberately not eagerly in fill_element_info(): the interface
    // elements' equation remap vectors are rebuilt after it (problem.cpp, "rebuild the interface
    // remapping"), so a classification taken there would be based on a half-built state.
    unsigned char get_hang_state();
    // The scan itself. Mirrors, branch for branch, what the corresponding fill/interpolate would do;
    // virtual for the same reason the fills are (the interface element adds its own fields).
    virtual unsigned char compute_hang_state() const;
    // The interface-only halves of the scan, so the base version composes exactly as the fills do.
    virtual bool scan_hang_interface_fields() const { return false; }
    virtual bool scan_interface_has_something_to_interpolate() const { return false; }

  public:
    // What fill_hang_info_with_equations(required, ..., NULL) WOULD return, without doing the fill.
    // The skip in elements_assembly.cpp needs the answer before the fill, which is why this exists
    // at all (today has_hang IS the fill's return value).
    virtual bool hang_fill_would_report_hang(const JITFuncSpec_RequiredShapes_FiniteElement_t &required);
    // PYOOMPH_PARANOID_HANG_FILL_CACHE only: writes a NaN weight into every hang buffer slot the
    // fill would have written, so that a supposedly hang-free body that still reads one is caught.
    virtual void poison_hang_info(JITShapeInfo_t *shape_info);
    friend class HangInterpGate;

  protected:
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
    // Element-local node indices that do not carry a C1 (vertex/linear) dof, i.e. the
    // indices i for which get_element_index_to_nodal_space_index_map()[3][i]==-1.
    virtual const std::vector<unsigned> & non_vertex_node_indices() const=0;
    unsigned _numpy_index;
    // Index of this element in the UNDISTRIBUTED base mesh, assigned before Problem::distribute() and
    // carried along by the element itself (oomph backs the element objects up and re-adds them rather
    // than recreating them). Only meaningful on root elements; -1 until assigned. Together with the
    // tree path this addresses an element independently of the partition, which is what lets state
    // files be written and read back on any number of processes. See dev_docs/distributed_state_files.md
    long global_base_index = -1;
    // Transient (per tesselated-numpy pass, cleared by Mesh::to_numpy / get_num_numpy_elemental_indices):
    // for each finer-neighbour hanging node registered on THIS element, its LOCAL coordinate in this element,
    // computed TOPOLOGICALLY by the finer neighbour from the tree neighbour finder (no physical geometry, no
    // locate_zeta). Drives edge/parameter placement in the tesselation and is exact on curved elements.
    std::map<oomph::Node *, oomph::Vector<double>> _tess_hang_scoord;
    double initial_cartesian_nondim_size = 0.0;
    double initial_quality_factor = 0.0;
    // Factory for the FaceElement (interface element) attached to a given face/edge of this bulk
    // element; concrete element types override this to return the matching Interface*Element* type.
    virtual oomph::FaceElement * construct_face_element(DynamicBulkElementInstance *, int ) {throw_runtime_error(std::string("Specify the face element constructor for the element type ")+typeid(*this).name()); return NULL;}
    virtual const std::vector<int> & get_possible_face_indices() const=0;
    virtual  std::vector<pyoomph::Node*> get_vertex_nodes_of_face(const int & face_index) const=0;

    // ALL nodes of a local face -- vertices AND face-interior ones (C2 mid-side/face-centre nodes,
    // the C2TB tet's face bubble): exactly the set build_face_element() wires into the face element,
    // and therefore exactly the nodes an interface mesh on that face owns.
    // get_vertex_nodes_of_face() above is deliberately vertices-only (it keys the facet map) and must
    // NOT be used to decide nodal boundary membership -- dropping a genuine mid-side node's
    // membership in repair_boundary_node_membership_from_face_tags() would be a correctness bug.
    std::vector<oomph::Node *> get_all_nodes_of_face(const int & face_index) const;
    // oomph's nnode_on_face() is face-index independent, which no wedge or pyramid can answer (their
    // facets are a mix of triangles and quads, hence the throws in wedges_and_pyramids.hpp), and
    // TElement<1,*>/TElement<3,*> never implement it at all. Same story for get_bulk_node_number: it
    // is missing on TElement<1,*>, and on the C2TB tet the face bubble lives OUTSIDE Node_on_face and
    // is wired in by hand (BulkElementTetra3dC2TB::Central_node_on_face). Hence these two hooks.
    virtual unsigned nnode_on_face_by_index(const int & face_index) const { return this->nnode_on_face(); }
    virtual unsigned node_index_on_face(const int & face_index, const unsigned & i) const { return this->get_bulk_node_number(face_index, i); }

    // --- Per-face boundary tags -------------------------------------------------------------
    // Which mesh boundaries each of this element's local faces lies on. This is the single source
    // of truth for TemplatedMeshBase::setup_boundary_element_info_from_face_tags(), replacing the
    // old per-shape reconstruction from nodal boundary membership (which cannot distinguish a
    // genuine boundary face from an interior face all of whose vertices happen to sit on the same
    // boundary -- the "third edge" false positive of a corner triangle, and its quad analogue in a
    // channel whose opposite walls share a boundary name).
    //
    // The tags are seeded once from the MeshTemplate's facet records
    // (TemplatedMeshBase::seed_face_boundaries_from_facets) and are then propagated FORWARD onto
    // the son elements at every split (BulkElementBase::dynamic_split, via face_index_in_father).
    // Because they live on the elements themselves rather than in a mesh-level map keyed by the
    // roots' node sets, they survive re-rooting/pruning of the tree forest (see
    // TemplatedMeshBase2d::setup_quadtree_forest, which deletes all ancestors below the coarsest
    // common refinement level) and need no bookkeeping on unrefinement (the father element still
    // carries the tags it was given when it was built).
    //
    // Key is the LOCAL FACE INDEX, i.e. one of the values in get_possible_face_indices() -- these
    // can be negative (quads/bricks use +/-1, +/-2, +/-3), hence the signed key type. Only faces
    // that actually lie on a boundary get an entry, so the map is empty for interior elements.
    const std::vector<unsigned> * get_face_boundaries(const int & face_index) const;
    void set_face_boundaries(const int & face_index, const std::vector<unsigned> & boundaries);
    void clear_face_boundaries() { face_boundaries.clear(); }
    const std::map<short, std::vector<unsigned>> & get_all_face_boundaries() const { return face_boundaries; }

    // Returned by face_index_in_father() when the son face lies in the father's interior, i.e. it
    // is not part of any father face and therefore inherits no boundary tag.
    static const int FACE_INTERIOR_IN_FATHER = -1000;

    // Given a local face index of THIS element (a son) and the son type (= son index, which is what
    // DynamicTree::dynamic_split_if_required passes to construct_son), return the local face index
    // of the father element that this face is a part of, or FACE_INTERIOR_IN_FATHER. Depends only on
    // the element shape and the split scheme, not on the polynomial order, so the default
    // implementation dispatches on dim()/shape; see elements.cpp. father_el is the parent element, needed
    // only for HETEROGENEOUS splits (a pyramid -> pyramids + tets) where the son shape alone does not reveal
    // the father shape -- a tet son of a pyramid is not a tet son of a tet; pass it from dynamic_split.
    virtual int face_index_in_father(const int & my_face_index, const unsigned & son_type, const BulkElementBase *father_el = nullptr) const;
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
    virtual void get_normal_at_s(const oomph::Vector<double> &s, oomph::Vector<double> &n, double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT dnormal_dcoord, double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT d2normal_dcoord2, unsigned history_index = 0) const;

    // The node index set in which the normal's coordinate sensitivities are expressed. For an
    // interface element that is the PARENT (bulk) element's nodes, not this element's own - see
    // InterfaceElementBase::get_dnormal_dcoords_at_s, which builds them from
    // Bulk_element_pt->dshape_local. The generated code loops over the same set, because
    // mark_further_required_fields marks "psi" on the bulk's position space. Quantities that are
    // naturally indexed by this element's own nodes (the metric, the shape functions) therefore have
    // to be scattered through normal_coord_node() when they are combined with them.
    virtual unsigned n_normal_coord_nodes() const { return this->nnode(); }
    virtual unsigned normal_coord_node(unsigned l) const { return l; }

    // PYOOMPH_REPORT_EXT_DATA sampling hook (defined in elements_assembly.cpp). A member because
    // external_local_eqn() is protected in oomph-lib. No-op unless the lever is set.
    void __sample_ext_data_stats();

    // PYOOMPH_POISON_UNREQUIRED: fill every shape buffer family that `required` does NOT ask for with
    // signalling NaN, so that generated code reading a buffer nobody flagged produces NaN instead of
    // a plausible stale number. Diagnostic only; a no-op unless the lever is set. `required` is the
    // struct actually PASSED to the fill, not functable->merged_required_shapes - the point is to
    // catch a per-pass under-request that the merge would hide. Never touches hanginfo / the equation
    // remap tables. Defined in elements_shapeinfo.cpp.
    virtual void poison_unrequired_shapes(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *si, bool element_level) const;

    // The requirement set that decides which external data an element attaches and, in lockstep, which
    // equations the interface remapping hands out. Normally assembly_required_shapes; the env lever
    // PYOOMPH_DISABLE_ASSEMBLY_EXTDATA_SPLIT restores the old behaviour of attaching from the full
    // merge, for A/B-ing the split on one and the same binary. Defined in elements.cpp.
    static const JITFuncSpec_RequiredShapes_FiniteElement_t *attachment_required_shapes(const JITFuncSpec_Table_FiniteElement_t *ft);

    // Which shape-function families of ONE interpolation space a fill pass has to produce. They are
    // filled individually: a psi-only space (by far the most common combination) does not pay for the
    // two gradient contractions.
    //
    // That split is only sound because the flags are complete. They were not: codegen.cpp excludes
    // "derived" shape expansions (the bare basis functions of the Jacobian COLUMNS) from the set it
    // marks required shapes from, so shapes_required_Hessian carried psi alone for a lid-driven
    // cavity whose HessianVectorProduct body dereferences dx_shapes. All-or-nothing filling hid it
    // completely - any one flag pulled in all four families. The generator now marks those bases from
    // a separate set (__all_Hessian_shapeexps_for_shapeflags); poison mode is what keeps it honest.
    struct RequiredShapeFamilies
    {
      bool psi = false;  // shapes[]
      bool dx = false;   // dx_shapes[] (Eulerian gradient), all three history slots
      bool dX = false;   // dX_shapes[] (Lagrangian gradient) AND dS_shapes[]: the local-coordinate
                         // derivative has no flag of its own and is marked as dX_psi by the code
                         // generator (D1XBasisFunctionLocalCoord derives from D1XBasisFunctionLagr).
      bool d2x = false;  // d2x_shapes[], d2S_shapes[] and d_d2x_shape_dcoord[]
      bool dcoord = false; // d_dx_shape_dcoord[]: the rank-4 sensitivity of this space's Eulerian
                         // gradient to the nodal positions. Only ever set on a moving mesh, and only
                         // for spaces whose gradient an assembled entry really differentiates - it is
                         // the most expensive single family of a moving-mesh Jacobian fill.
      bool any() const { return psi || dx || dX || d2x; }
    };
    // The single place that decides the above. fill_shape_info_at_s, set_remaining_shapes_appropriately
    // and poison_unrequired_shapes all key on it; they used to spell the predicate out three times and
    // had to be kept in lockstep by hand.
    RequiredShapeFamilies required_shape_families(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, unsigned ispace) const;
    // The DL twin. No "dominant space" clause: DL fields never represent the geometry.
    static RequiredShapeFamilies required_shape_families_DL(const JITFuncSpec_RequiredShapes_FiniteElement_t &required);

    // Discontinuous fields are stored as internal_data, on interfaces possibly also on external_data
    virtual oomph::Data *get_D0_nodal_data(const unsigned &fieldindex);
    virtual oomph::Data *get_DL_nodal_data(const unsigned &fieldindex);    
    virtual oomph::Data *get_DG_nodal_data(const unsigned &space_index,const unsigned &fieldindex);

    // Indices to the nodal buffer of the code generation
    
    virtual unsigned get_DG_buffer_index(const unsigned &space_index, const unsigned &fieldindex);    
    virtual unsigned get_DL_buffer_index(const unsigned &fieldindex);
    virtual unsigned get_D0_buffer_index(const unsigned &fieldindex);

    // Parent elements may have more nodal data entries than the interfaces. These functions cast a interface nodal index to the nodal index of the defining element
    virtual unsigned get_DG_node_index(const unsigned &, const unsigned &, const unsigned &nodeindex) const { return nodeindex; }
    virtual int get_DG_local_equation(const unsigned &space_index, const unsigned &fieldindex, const unsigned &nodeindex);
    
    virtual int get_DL_local_equation(const unsigned &fieldindex, const unsigned &nodeindex);
    virtual int get_D0_local_equation(const unsigned &fieldindex);

    virtual void get_DG_fields_at_s(unsigned space_index,unsigned history_index, const oomph::Vector<double> &s, oomph::Vector<double> &result) const;
    virtual int nedges() const = 0;
    // Called by a finer neighbour to register its hanging node `n` on THIS (coarser) element for the
    // tesselated-numpy export. `s_coarse` is the node's LOCAL coordinate in THIS element, computed
    // TOPOLOGICALLY by the finer neighbour from the tree neighbour finder (gteq_edge_neighbour /
    // tri_edge_neighbour coordinate facilities), NOT from physical positions. The default records it via
    // tess_register_hanging_node; every shape uses that (edge determined in reference space).
    virtual void add_node_from_finer_neighbor_for_tesselated_numpy(const oomph::Vector<double> &, oomph::Node *, std::vector<std::vector<std::set<oomph::Node *>>> &) {}
    virtual void inform_coarser_neighbors_for_tesselated_numpy(std::vector<std::vector<std::set<oomph::Node *>>> &) {}
    // --- Shared tesselated-numpy hanging-node helpers (used by both quad and tri paths) ---
    // Record a finer neighbour's hanging node n at local coordinate s_coarse in THIS element: find which of
    // this element's edges s_coarse lies on IN REFERENCE SPACE (exact, curvature-independent), bucket n on
    // that edge in add_nodes, and store s_coarse in _tess_hang_scoord. No-op if n is already my node or
    // s_coarse is on no edge. edge_corner_pairs are local corner-node index pairs (one per edge).
    void tess_register_hanging_node(const oomph::Vector<double> &s_coarse, oomph::Node *n, const std::vector<std::pair<unsigned, unsigned>> &edge_corner_pairs, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes);
    // Triangulate this element (corners + own mids + finer-neighbour hanging nodes) for the tesselated-numpy
    // export, entirely in LOCAL (reference) coordinates: own nodes from local_coordinate_of_node, hanging
    // nodes from _tess_hang_scoord (topological). Fills `triangles` (flat CCW index triples; each index an
    // own-node local index, or nnode()+running in add_nodes edge order — matching Mesh::to_numpy).
    void tess_hanging_delaunay(const std::vector<std::pair<unsigned, unsigned>> &edge_corner_pairs, const std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes, std::vector<unsigned> &triangles) const;
    // Tri-native counterpart of BulkElementQuad2dC1::inform_coarser_neighbors_for_tesselated_numpy: for each
    // of this triangle's 3 edges, use tri_edge_neighbour to find a strictly coarser neighbour (tri, or quad
    // across a mixed interface), compute each edge node's coordinate in that coarse neighbour topologically,
    // and register it. Shared by all four tri variants. (Implemented on RefineableTElement<2>.)
    void tess_inform_coarser_tri(const std::vector<std::pair<unsigned, unsigned>> &edge_corner_pairs, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes);
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
    // Re-apply the macro map to this element's nodes after a build that discarded it (see the
    // definition: oomph's solid build overwrites node positions with the FE interpolation).
    void reapply_macro_element_positions();
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
    // Where this element's C1 vertices sit in the reference domain of the macro element it is attached
    // to. Empty means "this is the macro element's own root element", i.e. the identity map.
    //
    // This generalises oomph's s_macro_ll/s_macro_ur, which record the same thing as an axis-aligned
    // box and therefore only work for the Q family: a red-refined triangle son is not an axis-aligned
    // sub-box of its father, and in a mixed forest a son need not even have its father's shape. Vertex
    // coordinates carry all of those cases, and reduce to the box for quads and bricks.
    std::vector<std::vector<double>> Macro_element_vertex_s;
    // Map a local coordinate of this element into the macro element's reference domain, by
    // interpolating Macro_element_vertex_s with this element's own C1 shape functions.
    std::vector<double> macro_coordinate_from_local(const oomph::Vector<double> &s) const;
    // Take over the father's macro element, recording where this son lies in the macro reference
    // domain. `son_vertices_in_father` gives this son's C1 vertices in the father's local coordinates.
    void inherit_macro_element_from_father(BulkElementBase *father_pt, const std::vector<std::vector<double>> &son_vertices_in_father);
    // Generic get_x_from_macro_element. Not an override: oomph::QElementBase already overrides that
    // virtual for the Q family (via the s_macro_ll/ur box, which stays correct), and having a second
    // final overrider reachable from the same class would be ambiguous. The simplex/mixed classes,
    // whose bases leave the virtual broken, forward to this.
    void get_x_from_generic_macro_element(const unsigned &t, const oomph::Vector<double> &s, oomph::Vector<double> &x) const;
    // True if this element's nodes should be snapped onto the macro geometry. False on a moving
    // (ALE) mesh, where the Eulerian position is an unknown and forcing it would fight the solve --
    // there the macro element only drives the initial configuration, exactly as it does today. See
    // dev_docs/macro_elements.md 7.
    bool macro_element_may_set_positions() const;
    // Evaluate the macro-element geometry itself: the Eulerian position this element's macro element
    // maps the local coordinate s to. Empty if the element has no macro element. Unlike
    // get_macro_element_coordinate_at_s (which stops at the macro-element coordinate) this returns a
    // position, which is what lets a test sample the curved geometry over the whole reference domain
    // -- interior included -- rather than only where nodes happen to sit.
    virtual std::vector<double> get_macro_element_position_at_s(oomph::Vector<double> s);
    DynamicBulkElementInstance *get_code_instance() { return codeinst; }
    const DynamicBulkElementInstance *get_code_instance() const { return codeinst; }
    // Bind this element to a physics (codeinst). Needed when a factory creates a son of a DIFFERENT element
    // type than the parent (e.g. a pyramid's tet son), where the parent cannot touch the son's protected
    // codeinst directly. Same-type factories set codeinst inline.
    void set_code_instance(DynamicBulkElementInstance *c) { codeinst = c; }

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
    double zeta_nodal(const unsigned &n, const unsigned &k, const unsigned &i) const override
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

    // For each of this element's local dofs, the index into the code's contribution_names -- i.e. the
    // row/column class that contributes_to_jacobian / contributes_to_mass_matrix are indexed by. Lets
    // the sparsity machinery decide which entries of the dense elemental block can ever be nonzero,
    // without evaluating anything. Cached; rebuilt after fill_element_info() (i.e. after every local
    // equation renumbering).
    // -1 means "could not be attributed to a field" and must be read CONSERVATIVELY, as "assume this
    // dof couples to everything". Under-reporting a coupling is safe (the entry then falls back to the
    // value filter), whereas wrongly claiming a dof is decoupled would silently truncate the Jacobian.
    const std::vector<int> &get_local_dof_contribution_indices();

  protected:
    std::vector<int> local_dof_contribution_indices;
    bool local_dof_contribution_indices_valid = false;
    virtual void fill_local_dof_contribution_indices(std::vector<int> &dest); // Overridden by InterfaceElementBase for the dofs it borrows from other elements

  public:
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
    void fill_in_contribution_to_residuals(oomph::Vector<double> &residuals) override
    {
      fill_in_generic_residual_contribution_jit(residuals, oomph::GeneralisedElement::Dummy_matrix, oomph::GeneralisedElement::Dummy_matrix, 0);
    }
    void fill_in_contribution_to_jacobian(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian) override;
    void fill_in_contribution_to_jacobian_and_mass_matrix(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian, oomph::DenseMatrix<double> &mass_matrix) override;

    void fill_in_contribution_to_dresiduals_dparameter(double *const &parameter_pt, oomph::Vector<double> &dres_dparam) override
    {
      fill_in_generic_dresidual_contribution_jit(parameter_pt, dres_dparam, oomph::GeneralisedElement::Dummy_matrix, oomph::GeneralisedElement::Dummy_matrix, 0);
    }

    void fill_in_contribution_to_djacobian_dparameter(double *const &parameter_pt, oomph::Vector<double> &dres_dparam, oomph::DenseMatrix<double> &djac_dparam) override
    {
      fill_in_generic_dresidual_contribution_jit(parameter_pt, dres_dparam, djac_dparam, oomph::GeneralisedElement::Dummy_matrix, 1);
    }

    void fill_in_contribution_to_djacobian_and_dmass_matrix_dparameter(double *const &parameter_pt, oomph::Vector<double> &dres_dparam, oomph::DenseMatrix<double> &djac_dparam, oomph::DenseMatrix<double> &dmass_matrix_dparam) override
    {
      fill_in_generic_dresidual_contribution_jit(parameter_pt, dres_dparam, djac_dparam, dmass_matrix_dparam, 2);
    }

    // Computes the Hessian-vector product contribution C^T * H * Y (H being this element's
    // residual Hessian) directly, without ever forming the dense Hessian tensor - used by
    // eigenvalue/bifurcation solvers that only need the action of the Hessian.
    void fill_in_contribution_to_hessian_vector_products(oomph::Vector<double> const &Y, oomph::DenseMatrix<double> const &C, oomph::DenseMatrix<double> &product) override;
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

    // --- Tracer particle advection (see tracers.hpp) ---------------------------------------------

    // Geometry of the time-interpolated mesh configuration at local coordinate s.
    //
    // The configuration within a timestep is defined as X(s,tau) = sum_k w_k X^k(s), a Lagrange
    // interpolation in time of the NODAL POSITIONS between the stored history levels k. Because the
    // shape functions do not depend on tau, both the Jacobian and the configuration velocity are
    // then exact rather than approximated: J(tau) = sum_k w_k J^k and dX/dtau = sum_k w'_k X^k.
    // That exactness is what makes a tracer sitting in a moving mesh with zero advection velocity
    // stay put; the old code took J at history level 0 for the whole step irrespective of tau.
    //
    //   w / dwdtau : nlevel Lagrange weights and their derivatives w.r.t. tau (tau in [0,1], where
    //                tau = 1 is history level 0). Any of the three outputs may be null.
    //   J          : J(a,i) = dX_i/ds_a, so el_dim rows by nodal_dim columns - NOT square on an
    //                interface element, where the caller wants the pseudo-inverse.
    //   dXdtau     : per unit tau, not per unit time.
    void tracer_geometry_at_s(const oomph::Vector<double> &s, unsigned nlevel, const double *w, const double *dwdtau,
                              oomph::Vector<double> *x, oomph::DenseMatrix<double> *J,
                              oomph::Vector<double> *dXdtau) const;

    // Per-element preparation (timestepper weights, time levels, element sizes), valid for as long
    // as a particle stays in this element. Must precede eval_tracer_advection_at_s.
    void tracer_prepare_element();

    // Evaluates registered tracer-advection field `index` at local coordinate s, in physical
    // (Eulerian) components. One index per tracer name per history level; the caller blends them
    // with the same weights it passed to tracer_geometry_at_s. xvelo is sized 3 - see the .cpp.
    void eval_tracer_advection_at_s(unsigned index, const oomph::Vector<double> &s, oomph::Vector<double> &xvelo);

    //  void assign_local_eqn_numbers(const bool &store_local_dof_pt);
    // Assigns local equation numbers to the "additional" degrees of freedom introduced beyond
    // oomph-lib's standard nodal/internal/external data handling (e.g. interface-only dofs); called
    // as part of the element's local equation numbering pass.
    void assign_additional_local_eqn_numbers() override;
    //  virtual void assign_all_generic_local_eqn_numbers(const bool &store_local_dof_pt);

    ~BulkElementBase() override;


    // Creates a new, empty instance of the same concrete element type as `this` (same JIT code
    // instance), used when a mesh element is split into sons during h-refinement.
    virtual BulkElementBase *create_son_instance() const = 0;
    unsigned ncont_interpolated_values() const override;
    unsigned required_nvalue(const unsigned &n) const override;

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

    // Local-coordinate SECOND derivatives d2psi/(ds_a ds_b) of the shape functions, indexed as
    // d2psi(l, PYOOMPH_D2_SLOT(a,b)) - i.e. the full square 3x3 slot layout defined in jitbridge.h,
    // NOT oomph-lib's dimension-dependent N2deriv packing. d2psi must therefore be sized
    // (nnode, MAX_N2DERIV), and both the (a,b) and the (b,a) slot are written.
    //
    // The defaults throw. supports_second_spatial_derivatives() reports the same information without
    // raising, so an unsupported element/space combination can be rejected at problem setup rather
    // than mid-assembly.
    virtual void d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const;
    virtual void d2shape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const;
    virtual void d2shape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const;
    virtual void d2shape_local_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const;
    virtual void d2shape_local_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const;

    // Mirrors dshape_local_of_space.
    inline void d2shape_local_of_space(const unsigned &space_index, const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const
    {
      switch (space_index)
      {
      case 0:
        this->d2shape_local_at_s_C2TB(s, psi, dpsi, d2psi); break;
      case 1:
        this->d2shape_local_at_s_C2(s, psi, dpsi, d2psi); break;
      case 2:
        this->d2shape_local_at_s_C1TB(s, psi, dpsi, d2psi); break;
      case 3:
        this->d2shape_local_at_s_C1(s, psi, dpsi, d2psi); break;
      default:
        throw_runtime_error("Invalid space index " + std::to_string(space_index));
      }
    }

    // Second derivatives of the element's OWN (geometry/Pos) shape functions. Needed for the
    // X_{k,ab} = sum_l X^l_k d2Psi_l/(ds_a ds_b) term that converts local into spatial second
    // derivatives, so it is required whenever any second derivative is requested at all.
    //
    // The default is an adapter around oomph::FiniteElement::d2shape_local performing the
    // N2deriv -> PYOOMPH_D2_SLOT remap; keeping that remap here, in exactly one place, is what stops
    // the 2D packing ({00,11,01}) and the 3D packing ({00,11,22,01,02,12}) from being confused.
    virtual void d2shape_local_pyoomph(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const;

    // Helper for the above: remaps a d2psids filled by oomph-lib in N2deriv packing for element
    // dimension el_dim into the pyoomph slot layout, writing both (a,b) and (b,a).
    static void remap_oomph_d2shape_packing(unsigned el_dim, unsigned nnode, const oomph::DShape &d2psids_oomph, oomph::DShape &d2psi);

    // Clears all MAX_N2DERIV slots of the first nnode entries. Used by the spaces whose local second
    // derivatives vanish identically (linear simplex bases, DL).
    static inline void zero_d2shape(oomph::DShape &d2psi, unsigned nnode)
    {
      for (unsigned l = 0; l < nnode; l++)
        for (unsigned k = 0; k < MAX_N2DERIV; k++) d2psi(l, k) = 0.0;
    }

    // Whether this element can produce second spatial derivatives at all, i.e. whether it has a real
    // d2shape_local for its geometry AND for each of its interpolation spaces. Defaults to false and
    // is overridden by the concrete element classes that implement them, so that a newly added
    // element type is rejected loudly rather than silently returning garbage.
    virtual bool supports_second_spatial_derivatives(std::string &why) const;


    

    // Construct node n of pyoomph's own Node type (rather than plain oomph::Node), so that pyoomph's
    // additional per-node bookkeeping (e.g. discontinuous-field data) is available; the
    // TimeStepper-taking overloads additionally register a specific time-stepping scheme for that node.
    oomph::Node *construct_node(const unsigned &n) override;
    oomph::Node *construct_node(const unsigned &n, oomph::TimeStepper *const &time_stepper_pt) override;
    oomph::Node *construct_boundary_node(const unsigned &n) override;
    oomph::Node *construct_boundary_node(const unsigned &n, oomph::TimeStepper *const &time_stepper_pt) override;
    virtual oomph::Node *boundary_node_pt(const int &face_index, const unsigned int index);


    // For a C2 (quadratic) node, returns the C1 (linear) nodes that geometrically "support" it
    // (i.e. whose linear interpolation would reproduce its position) - used to interpolate
    // dummy/lower-order field values living only on the C1 sub-mesh.
    virtual void get_supporting_C1_nodes_of_C2_node(const unsigned &, std::vector<oomph::Node *> &) { throw_runtime_error("Implement"); }

    // Evaluate the discontinuous-Lagrange (DL) resp. element-constant (D0) fields at local
    // coordinate s and history/time index t, writing all field values into res.
    void get_interpolated_fields_DL(const oomph::Vector<double> &s, std::vector<double> &res, const unsigned &t = 0) const;
    void get_interpolated_fields_D0(const oomph::Vector<double> &s, std::vector<double> &res, const unsigned &t = 0) const;

    virtual oomph::Vector<double> get_midpoint_s();                        // Set s=[0.5*(smin+smax), ... ] (but modified e.g. for tris)
    oomph::Vector<double> get_Eulerian_midpoint_from_local_coordinate();   // Set s=[0.5*(smin+smax), ... ] and evaluate the position
    oomph::Vector<double> get_Lagrangian_midpoint_from_local_coordinate(); // Set s=[0.5*(smin+smax), ... ] and evaluate the position

    void get_interpolated_values(const unsigned &t, const oomph::Vector<double> &s, oomph::Vector<double> &values) override;
    void get_interpolated_values(const oomph::Vector<double> &s, oomph::Vector<double> &values) override { get_interpolated_values(0, s, values); }
    void get_interpolated_discontinuous_values(const unsigned &t, const oomph::Vector<double> &s, oomph::Vector<double> &values);
    void get_interpolated_discontinuous_values(const oomph::Vector<double> &s, oomph::Vector<double> &values) { get_interpolated_discontinuous_values(0, s, values); }
    void output(std::ostream &outfile, const unsigned &n_plot) override;

    virtual std::vector<double> get_outline(bool ) { return std::vector<double>(0); }
    // Number of independent flux quantities used by oomph-lib's Z2 error estimator for this
    // element type (drives the size of get_Z2_flux's output).
    unsigned num_Z2_flux_terms() override;
    // Compound-flux grouping: how the flux terms are partitioned into groups that are normalised
    // independently (and then combined by taking the maximum). One group unless the generated code
    // says otherwise. See dev_docs/spatial_error_estimators.md.
    unsigned ncompound_fluxes() override;
    void get_Z2_compound_flux_indices(oomph::Vector<unsigned> &flux_index) override;
    // Per-group normalisation exponent (1 = relative, 0 = absolute) and weight, read by
    // LagrZ2ErrorEstimator after the mesh-global norms have been assembled.
    virtual double Z2_compound_flux_normalize_relative(const unsigned &g);
    virtual double Z2_compound_flux_weight(const unsigned &g);
    // Evaluates the Z2-recovery flux vector (typically the gradient of the dominant field) at local
    // coordinate s, used by the Z2 error estimator to drive adaptive mesh refinement.
    void get_Z2_flux(const oomph::Vector<double> &s, oomph::Vector<double> &flux) override;
    // After h-refinement has split this element into sons and then possibly un-refined again,
    // rebuilds this element's data from the (surviving) son elements' data.
    void rebuild_from_sons(oomph::Mesh *&mesh_pt) override;
    // Finishes constructing a newly created element (called after pre_build/nodes are set up):
    // allocates discontinuous field data and performs other setup that requires the full node set
    // to already be in place.
    void further_build() override;
    // For a son element created during refinement, returns the local coordinate sfather of local
    // node l as seen in its father element (used to interpolate values from father to son node l).
    // Each concrete geometric element type must implement this according to its own son-numbering scheme.
    virtual void get_nodal_s_in_father(const unsigned int &, oomph::Vector<double> &) { throw_runtime_error("Implement"); }
    // Inverse of the above for a SIMPLEX son of a simplex father: given a father-local coordinate,
    // returns this son's own local coordinate for the same point, or false if the point lies
    // outside this son. See the .cpp; used by restore_orphaned_interior_nodes.
    bool son_local_from_father_simplex(const oomph::Vector<double> &s_father, oomph::Vector<double> &s_son);
    // Rebuilds any of this (father) element's node slots that h-refinement orphaned and adapt_mesh
    // then deleted -- i.e. interior bubble nodes that no son inherited. Called at the start of
    // rebuild_from_sons(), while the sons are still alive to restrict from.
    void restore_orphaned_interior_nodes(oomph::Mesh *&mesh_pt);
    // Sets up as much of a new element as possible before all of its nodes exist yet (used during
    // mesh refinement/construction, where nodes are shared with adjacent elements and must not be
    // duplicated); new_node_pt accumulates nodes that had to be freshly constructed.
    void pre_build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt) override;

    unsigned nscalar_paraview() const override;
    void scalar_value_paraview(std::ofstream &file_out, const unsigned &i, const unsigned &nplot) const override;
    std::string scalar_name_paraview(const unsigned &i) const override;
    // Additional hanging-node setup beyond oomph-lib's default (e.g. for non-isoparametric spaces
    // where hanging-node constraints differ from the geometric element's own); overridden by
    // concrete element types, many of which are pure isoparametric and can simply delegate to
    // oomph-lib's default via BulkElementBase::further_setup_hanging_nodes().
    void further_setup_hanging_nodes() override;

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
    inline void assign_nodal_local_eqn_numbers(const bool &store_local_dof_pt) override
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

    	// User-added additional dof constraints (see NodeWithFieldIndicesBase::add_additional_dof_constraint):		
    virtual void setup_additional_dof_constraints();
    // Temporarily unpins Dirichlet-constrained dofs so that a linear-algebra backend can directly
    // manipulate rows/columns for these dofs (e.g. when assembling with the constraint substituted
    // out); info records what needs to be restored afterwards.
    virtual void unpin_Dirichlet_dofs_for_matrix_manipulation(DirichletMatrixManipulationInfo & info);

    // Splits this element into its refinement "sons" and returns pointers to the newly created son
    // elements in son_pt, without altering the mesh's element list (unlike a normal adaptive
    // refinement step) - used e.g. for one-off geometric subdivision/export. Delegates the actual
    // choice of son count/types to refinement_pattern().
    void dynamic_split(oomph::Vector<BulkElementBase *> &son_pt) const;

    // The split scheme currently used to refine this element. Default: isotropic subdivision into
    // required_nsons() sons of the same type (IsotropicSameTypeRefinementPattern). Override (or make
    // configurable) to support anisotropic / heterogeneous splits; see
    // dev_docs/adaptive_refinement.md.
    virtual const RefinementPattern *refinement_pattern() const;

    // Fill this element as a C1 son of a PYRAMID father (mixed 6-pyramid + 4-tet red split). Works whether
    // `this` is a sub-pyramid or a tet: for a C1 son each node IS a vertex, so its father-local coordinate is
    // oomph::RefineablePyramidElement::son_vertices_in_father[son_type][j] -- no son-shape evaluation (which
    // would hit the pyramid's 1/(1-s2) apex singularity). Reuses coincident father nodes and shares new
    // edge/face nodes across all 10 sons (and adjacent fathers) via the pyramid registry. Called from
    // RefineablePyramidElement::build() and, when a tet's father is a pyramid, from RefineableTElement<3>::
    // build(). Lives here (not on the pyramid) so it can call the son's protected construct_node().
    void build_as_pyramid_son(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt);

    // Fill this element as a son of a BRICK father, sharing interface nodes through the mixed-forest registry
    // (RefineablePyramidElement::Shared_node_registry). Used only when RefineablePyramidElement::
    // Mixed_forest_active -- i.e. a mixed 3d mesh with a brick and a pyramid/wedge -- so that a brick and an
    // adjacent pyramid/wedge sharing a QUAD face (hex face <-> pyramid base / wedge side) key that face's nodes
    // on the same (father node, rounded father-shape weight) pairs and reuse one shared node. The son node's
    // father-local coordinate is get_nodal_s_in_father(j) (the octree 1->8 affine map). Pure-brick meshes keep
    // oomph-lib's native RefineableQElement<3>::build (octree compass node reuse) untouched. Called from the
    // brick elements' build() override; lives here so it can call the son's protected construct_node().
    void build_as_brick_son(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt);

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

    // Sets the integration (quadrature) order/scheme to use for this element; each concrete element
    // type maps "order" to the appropriate IntegrationSchemeStorage lookup for its shape.
    virtual void set_integration_order(unsigned int ) { throw_runtime_error("Implement"); }

    virtual bool has_bubble() const { return false; } // If not, C2TB is the same space as C2

    // Debug helper analogous to debug_analytical_jacobian, but for the Hessian: compares the
    // analytic Hessian-vector product fill_in_generic_hessian(Y, C, ...) against a finite-difference
    // approximation with step epsilon.
    // Compares the analytical Hessian-vector products against finite differences of the (analytical)
    // Jacobian and returns the largest absolute discrepancy, so a caller can assert on it. Entries
    // exceeding epsilon are additionally printed; pass epsilon<=0 to print everything.
    virtual double debug_hessian(std::vector<double> Y, std::vector<std::vector<double>> C, double epsilon);
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
    void shape(const oomph::Vector<double>& , oomph::Shape& ) const override {}

    void local_coordinate_of_node(const unsigned& , oomph::Vector<double>& s) const override
    {
      s.resize(0);
    }
    
  };


  // BulkElementBase is implemented across several translation units, split by responsibility because
  // one file of 15k lines was unnavigable. Where to look for a member definition:
  //
  //   elements.cpp             construction, create_from_template, fill/free_element_info,
  //                            shape-buffer allocation, macro elements, D0/DL/DG accessors, output
  //   elements_shapeinfo.cpp   fill_shape_info_at_s and everything that fills the JIT shape buffer
  //   elements_assembly.cpp    residual/Jacobian/mass/Hessian assembly, FD fallbacks, debug compares
  //   elements_dofs.cpp        local equation numbering, dof names/indices, pinning and constraints
  //   elements_hanging.cpp     the hanging-node engine and the hang buffers handed to the JIT code
  //   elements_adapt.cpp       h-adaptivity: build/rebuild, son<->father maps, refinement patterns
  //   elements_geometry.cpp    normals and their nodal-coordinate derivatives, diameter, faces
  //   elements_interface.cpp   InterfaceElementBase
  //   elements_0d1d.cpp        concrete 0-d and 1-d elements   (declared in elements_concrete.hpp)
  //   elements_2d.cpp          concrete 2-d elements                        "
  //   elements_3d.cpp          concrete 3-d elements                        "


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
  public:
    using BulkElementBase::as_interface_element; // keep the const overload visible past the override
    InterfaceElementBase *as_interface_element() override { return this; }

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
    void update_in_external_fd(const unsigned &i) override;
    // Registers "data" (a Data object, e.g. from the bulk or opposite element) as required external
    // data of this element if not already present; is_geometric marks it as a solid/ALE position
    // dof (relevant for how its Jacobian contribution is computed). Returns true if newly added.
    virtual bool add_required_ext_data(oomph::Data *data, bool is_geometric);
    // Walks the "required_shapes" description generated alongside the JIT code and adds all Data
    // (nodal/internal/external) of from_elem that this interface element's generated residual code
    // needs to access as external data.
    virtual void add_required_external_data(JITFuncSpec_RequiredShapes_FiniteElement_t *required, BulkElementBase *from_elem);
    void prepare_shape_buffer_for_integration(const JITFuncSpec_RequiredShapes_FiniteElement_t &required_shapes, unsigned int flag) override;
    double fill_shape_info_at_s(const oomph::Vector<double> &s, const unsigned int &index, const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, double &JLagr, unsigned int flag, oomph::DenseMatrix<double> *dxds = NULL, unsigned history_index=0) const override;
    bool fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap) override;
    // Additionally poisons the bulk/opposite sub-buffers that this pass does NOT recurse into, i.e.
    // exactly those an unflagged reader could still reach through shapeinfo->bulk_shapeinfo.
    void poison_unrequired_shapes(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *si, bool element_level) const override;
    // Additional interface-only hanging-node bookkeeping, used by InterfaceElementBase to handle
    // the extra fields that exist only on the interface element and not on the bulk element.
    bool fill_hang_info_with_equations_interface(JITShapeInfo_t *shape_info) override;
    // Additionally handles INTERFACE_DOF_CONSTRAIN_TO_C1 additional dof constraints (on top of the
    // CONTINUOUS_BASE_DOF_CONSTRAIN_TO_C1 ones already handled by the base class implementation).
    bool fill_additional_hang_buffer_data(JITShapeInfo_t *shape_info) override;
    // Stage-3 scan counterparts of the two overrides above, plus the recursion clause: the fill
    // reports "hanging" for ANY interface element that pulls in a bulk or opposite element, because
    // it then abuses the hang buffers as the local-equation remap channel. That is exactly why the
    // fill of such an element must never be skipped, and keeping the clause here rather than at the
    // call sites keeps the predicate and the fill in one place.
    bool scan_hang_interface_fields() const override;
    bool scan_interface_has_something_to_interpolate() const override;
    bool hang_fill_would_report_hang(const JITFuncSpec_RequiredShapes_FiniteElement_t &required) override;
    void ensure_external_data() override;
    void assign_additional_local_eqn_numbers() override;
    void fill_in_jacobian_from_lagragian_by_fd(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian) override;
    // Allocates the additional (interface-only, beyond the inherited bulk) field dofs on this
    // element's nodes/internal data, based on the interface's own JIT function table.
    virtual void add_interface_dofs();
    // Interface-specific part of fill_element_info: rebuilds the JITElementInfo_t/JITShapeInfo_t
    // bookkeeping for the additional interface fields and the bulk_eqn_map/opposite-side equation
    // maps, complementing BulkElementBase::fill_element_info for the inherited bulk part.
    virtual void fill_element_info_interface_part(bool without_equations=false);
    std::vector<std::string> get_dof_names(bool not_a_root_call = false) override;
    // Additionally attributes the dofs this element BORROWS from other elements (its own bulk, the
    // opposite interface element, the opposite's bulk), which the base walk cannot see.
    void fill_local_dof_contribution_indices(std::vector<int> &dest) override;
    void get_dnormal_dcoords_at_s(const oomph::Vector<double> &s, double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT dnormal_dcoord, double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT d2normal_dcoord2) const override;
    unsigned n_normal_coord_nodes() const override { return this->bulk_element_pt()->nnode(); }
    unsigned normal_coord_node(unsigned l) const override { return const_cast<InterfaceElementBase *>(this)->bulk_node_number(l); }

    // Maps a local coordinate s on this interface element to the corresponding local coordinate on
    // the opposite_side interface element, accounting for possibly different node/edge orientation
    // and (for non-conforming "internal facet" pairings) different parametrization ranges. Must be
    // implemented per concrete Interface*Element* subclass, since the mapping depends on the face
    // geometry (line/triangle/quad) and node ordering conventions.
    virtual oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &) const { throw_runtime_error("Implement"); }
    // Determines opposite_orientation and opposite_node_index by matching this element's vertex
    // nodes to those of opposite_side (within a small tolerance, allowing for the "offset" vector
    // e.g. on periodic domains), so that fields on the two sides can be looked up consistently.
    // Must be implemented per concrete Interface*Element* subclass (the matching logic - which
    // permutations of nodes to try - depends on the face's geometric shape).
    virtual void analyze_opposite_orientation(const std::vector<double> & ) { throw_runtime_error("Implement"); }
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
    inline void assign_nodal_local_eqn_numbers(const bool &store_local_dof_pt) override
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
    void set_remaining_shapes_appropriately(JITShapeInfo_t *shape_info, const JITFuncSpec_RequiredShapes_FiniteElement_t &required_shapes) override;
    void pin_dummy_values() override;
    // User-added additional dof constraints (see NodeWithFieldIndicesBase::add_additional_dof_constraint):		
    void setup_additional_dof_constraints() override;
    void unpin_Dirichlet_dofs_for_matrix_manipulation(DirichletMatrixManipulationInfo & info) override;

    // Flags this element as the never-assembled placeholder on the far side of an interior facet and
    // pins the element-owned (DG/DL/D0) storage it nevertheless allocated - see elements.cpp.
    void set_as_internal_facet_opposite_dummy();
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
      if (_opposite_side && !_opposite_side->as_interface_element())
      {
        throw_runtime_error("Can only set an Interface Element as the opposite side of and interface element");
      }
      opposite_side = (_opposite_side ? _opposite_side->as_interface_element() : NULL);
      const JITFuncSpec_Table_FiniteElement_t *functable = this->codeinst->get_func_table();
      // Attachment reads the assembled requirements only - see attachment_required_shapes.
      const JITFuncSpec_RequiredShapes_FiniteElement_t &attach_req = *attachment_required_shapes(functable);

      if (attach_req.opposite_shapes)
      {
        // std::cout << "INTERFACE ELEM MERGED " << attach_req.opposite_shapes->psi_D0 << std::endl;
        add_required_external_data(attach_req.opposite_shapes, opposite_side);
        if (attach_req.opposite_shapes->bulk_shapes)
        {
          //        std::cout << "INTERFACE ELEM MERGED BULK " <<  attach_req.opposite_shapes->bulk_shapes->psi_D0 << std::endl;
          auto *opp_blk = dynamic_cast<BulkElementBase *>(opposite_side->bulk_element_pt());
          add_required_external_data(attach_req.opposite_shapes->bulk_shapes, opp_blk);
          // ...and register the same data on the OPPOSITE element as well.
          //
          // Reaching the opposite side's bulk goes through the opposite INTERFACE element: its shape
          // info is filled by remapping the bulk's local equations through that element's own
          // bulk_eqn_map, which can only resolve dofs that are among ITS dofs. That map is built from
          // the opposite element's own requirements, and those need not mention the bulk at all -- the
          // requirement lives here, on the side that wrote the expression.
          //
          // A droplet/gas interface makes it concrete: writing grad(c) of the gas domain in a droplet
          // interface condition needs the gas bulk element's C2 dofs, but the gas-side interface
          // element may only carry a Dirichlet condition on c, which needs no bulk shapes whatsoever.
          // Without this the remap yields "not found" for exactly those dofs.
          opposite_side->add_required_external_data(attach_req.opposite_shapes->bulk_shapes, opp_blk);
        }
      }

      this->eleminfo.opposite_eleminfo = &(opposite_side->eleminfo);
      std::vector<double> offs=offset;
      for (unsigned int i=offset.size();i<this->nodal_dimension();i++) offs.push_back(0.0);
      this->analyze_opposite_orientation(offs);
    }

    double zeta_nodal(const unsigned &n, const unsigned &k, const unsigned &i) const override
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
      return static_cast<pyoomph::Node *>(opposite_side->node_pt(opposite_node_index[i]));
    }
    InterfaceElementBase *get_opposite_side() { return opposite_side; }
    const InterfaceElementBase *get_opposite_side() const { return opposite_side; }

    int get_nodal_index_by_name(oomph::Node *n, std::string fieldname) override;
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
    bool fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap) override
    {
      HangFillTimeScope __hftime(HANGFILL_SLOT_FILL);
      bool res1 = BASE::fill_hang_info_with_equations(required, shape_info, eqn_remap);
      bool res2 = InterfaceElementBase::fill_hang_info_with_equations(required, shape_info, eqn_remap);
      return res1 || res2;
    }


    void interpolate_hang_values() override
    {
      HangFillTimeScope __hftime(HANGFILL_SLOT_INTERP);
      HangInterpGate __gate(this);
      if (__gate.skip())
        return;
      BASE::interpolate_hang_values();
      this->interpolate_hang_values_at_interface();
    }

  public:
    double zeta_nodal(const unsigned &n, const unsigned &k, const unsigned &i) const override
    {
      return oomph::FaceElement::zeta_nodal(n, k, i);
    }

    void fill_element_info(bool without_equations=false) override
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
      // The geometric flags describe external_data_pt() BY INDEX, so they have to go with the list they
      // describe. (Today the only flush is this one, on a freshly constructed element whose vector is
      // still empty, so nothing is currently stale - but the flag became load-bearing when
      // fill_in_jacobian_from_lagragian_by_fd started honouring it, and a stale "geometric" entry there
      // is a silently wrong Jacobian column.)
      this->external_data_is_geometric.clear();
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

      // Attachment reads the assembled requirements only - see attachment_required_shapes.
      const JITFuncSpec_RequiredShapes_FiniteElement_t &attach_req = *attachment_required_shapes(functable);
      if (attach_req.bulk_shapes)
      {
        //	  std::cout << "ADDING BULK EXT DATA" << std::endl;
        add_required_external_data(attach_req.bulk_shapes, dynamic_cast<BulkElementBase *>(bulk_el_pt)); // TODO: Also the others? (is it necessary e.g. spatial integration of the stress along interface)
        if (attach_req.bulk_shapes->bulk_shapes)
        {
          InterfaceElementBase *ip = dynamic_cast<InterfaceElementBase *>(bulk_el_pt);
          add_required_external_data(attach_req.bulk_shapes->bulk_shapes, dynamic_cast<BulkElementBase *>(ip->bulk_element_pt()));
        }
      }
    }

    void get_normal_at_s(const oomph::Vector<double> &s, oomph::Vector<double> &n, double *  PYOOMPH_RESTRICT  *  PYOOMPH_RESTRICT *  PYOOMPH_RESTRICT  dnormal_dcoord, double *  PYOOMPH_RESTRICT  *  PYOOMPH_RESTRICT  *  PYOOMPH_RESTRICT  *  PYOOMPH_RESTRICT *  PYOOMPH_RESTRICT d2normal_dcoord2, unsigned history_index = 0) const override
    {
      this->outer_unit_normal(s, n);
      if (history_index > 0)
      {
        // outer_unit_normal() only knows the current nodal positions, so the normal of a previous
        // configuration is built from that configuration's tangents instead. Those give the normal
        // only up to a sign; the orientation is taken from the current outer normal, which is safe
        // because the mesh motion over one step cannot turn the element inside out (an inverted
        // element is detected and rejected elsewhere).
        oomph::Vector<double> n_hist(n.size(), 0.0);
        this->BulkElementBase::get_normal_at_s(s, n_hist, NULL, NULL, history_index);
        double dot = 0.0;
        for (unsigned d = 0; d < n.size(); d++) dot += n_hist[d] * n[d];
        double sign = (dot < 0.0 ? -1.0 : 1.0);
        for (unsigned d = 0; d < n.size(); d++) n[d] = sign * n_hist[d];
      }
      if (dnormal_dcoord)
      {
        this->get_dnormal_dcoords_at_s(s, dnormal_dcoord, d2normal_dcoord2);
      }
    }
  };



  extern double *__replace_RJM_by_param_deriv;

  // Set by the element that is currently having its shape buffer filled, read back by the extern "C"
  // trampoline the generated code calls to refill that buffer at an integration point. Declared here
  // because setter (elements_shapeinfo.cpp) and reader (elements.cpp) are no longer the same TU.
  extern BulkElementBase *_currently_assembled_element;
}
