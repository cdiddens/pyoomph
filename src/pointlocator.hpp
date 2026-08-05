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

// Point location and field evaluation on a source mesh, for mesh-to-mesh transfer.
//
// This replaces every use of oomph::MeshAsGeomObject::locate_zeta in pyoomph (remeshing
// interpolation, the zeta-based interface interpolation, the projection solve's integration-point
// mapping, get_values_at_zetas, add_interpolated_nodes_at) and, once complete, MeshKDTree as well.
// See dev_docs/mesh_point_locator.md for the design rationale; the short version:
//
//  * All those call sites ask the same question - "given a target point in some coordinate space,
//    which source element contains it, and where?" - and differ only in the coordinate space
//    (Eulerian/Lagrangian/boundary zeta), in whether the source elements have the same dimension as
//    that space (square Newton inversion) or one less (overdetermined -> closest-point projection),
//    and in whether the space is periodic. Zeta is therefore not a separate mechanism, just one
//    space; a closed interface loop is that space being periodic; a 2d surface in 3d is the
//    codimension-1 case, which needs no chart at all.
//
//  * Location and evaluation are deliberately SPLIT. Locating is the expensive part and is done
//    once; evaluation is cheap and is repeated (the projection solve evaluates once per time level
//    against the same locations). Under MPI this split is what keeps the cost down: the routing
//    schedule is derived during location and reused verbatim by every evaluation, so each
//    evaluation is one MPI_Alltoallv of doubles with nothing to re-derive.
//
// Two rules exist purely so the MPI layer can be added later without touching any call site, and
// both must be respected by new code even though nothing is distributed yet:
//
//  1. A located point is identified by an opaque LocationHandle, never by a source element pointer
//    that the caller stores. Under distribution the source element may live on another rank, where
//    a pointer means nothing. (The projection solve's coords_oldmesh currently stores a raw
//    BulkElementBase* - that is exactly the pattern being replaced.)
//
//  2. Everything a consumer will need is requested up front through EvalRequest and pulled into a
//    local buffer BEFORE it is used. A residual that reaches back into the source element while
//    assembling cannot work across ranks, and a consumer that discovers it needs one more quantity
//    afterwards forces a second communication round trip.

#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include "kdtree.hpp"
// pyoomph::Node is a typedef of a template instantiation (nodes.hpp), not a class, so it cannot be
// forward declared - the header has to be pulled in.
#include "nodes.hpp"

namespace pyoomph
{
  class Mesh;
  class BulkElementBase;

  // Which coordinate space the query points and the source mesh index live in. This is the single
  // knob that turns the bulk interpolation problem into the interface one: the source elements and
  // the search structure are the same, only the coordinate attached to each node changes.
  enum class LocatorSpace
  {
    Eulerian,    // nodal x at the requested time level - ordinary bulk remeshing interpolation
    Lagrangian,  // nodal xi - used when interpolated_lagrangian_coordinates_at_remeshing is set
    BoundaryZeta // the intrinsic boundary coordinate of one boundary (see LocatorSetup::boundary_index)
  };

  // How a query point is matched to a source element once candidates have been found.
  enum class LocatorMode
  {
    Invert,  // space dimension == element dimension: solve x(s) = x_query for s
    Project  // source is codimension 1 in the space: minimise |x(s) - x_query| over s
  };

  struct LocatorSetup
  {
    LocatorSpace space = LocatorSpace::Eulerian;
    unsigned time_index = 0; // history level the source coordinates are taken at

    // Only for LocatorSpace::BoundaryZeta: which boundary of the source mesh's bulk mesh the zeta
    // values belong to.
    int boundary_index = -1;

    // Per-component period of the coordinate space, 0.0 meaning "not periodic". A closed interface
    // loop parameterised by arclength sets period[0] to the loop length (or 1.0 when normalised).
    // The seam element then runs from the last node's zeta to the first node's zeta + period, and
    // is only invertible because the locator unwraps any drop larger than half a period. Without
    // this the seam element spans the whole zeta range and matches essentially any query - the
    // silent failure that motivated this class.
    std::vector<double> period;

    // How far outside its reference domain a local coordinate may land and still count as inside,
    // in LOCAL coordinate units (which are O(1), so this is a fraction of an element).
    //
    // It must not be at machine epsilon. A query that sits exactly on an element's edge - the end of
    // a boundary, a node shared with the neighbouring element - inverts to s = 1 + a few ulp as
    // often as to s = 1 - a few ulp, and rejecting the first means the point is reported unlocatable
    // for a purely numerical reason. At 1e-8 the geometric slack is 1e-8 of an element, far below
    // anything physical, and a point genuinely outside is still outside by orders of magnitude.
    double inside_tolerance = 1e-8;

    // Reject a Project match whose residual offset exceeds this multiple of the local source
    // element size. Guards against a near-touching interface matching the wrong sheet, which a pure
    // nearest-point search cannot distinguish. Ignored for Invert.
    double max_projection_offset_factor = 0.5;
  };

  // What a consumer wants evaluated at each located point. A bitfield rather than separate calls
  // because under MPI every extra request after the fact is another round trip - see rule 2 above.
  struct EvalRequest
  {
    bool continuous_fields = false; // nodal (C1/C2/...) fields
    bool D0_fields = false;         // element-constant discontinuous fields
    bool DL_fields = false;         // discontinuous-Lagrange fields
    bool DG_fields = false;         // per-space discontinuous fields
    bool position = false;          // interpolated_x
    bool lagrangian = false;        // interpolated xi
    bool zeta = false;              // interpolated_zeta

    // Time-history levels to evaluate, in the order they will appear in the result. Empty means
    // level 0 only.
    std::vector<unsigned> time_levels;

    // Maps this consumer's continuous field indices onto the source mesh's, for the case where the
    // two meshes were generated from different JIT code instances. Empty means identity. Negative
    // entries mark fields absent from the source, which are left untouched in the result.
    std::vector<int> field_map;
  };

  // Opaque identity of one located query point. Deliberately carries no element pointer: `owner` is
  // the rank that holds the source element (always 0 in serial) and `slot` indexes that rank's own
  // location table. Resolving a handle to a BulkElementBase* is possible only on its owner, through
  // LocationSet, and only for local handles.
  struct LocationHandle
  {
    int owner = -1; // -1 marks a point that could not be located anywhere
    unsigned slot = 0;

    bool is_located() const { return owner >= 0; }
  };

  // Result of locating a batch of points, and the handle through which they are evaluated.
  //
  // Owns the routing schedule: which query belongs to which rank, in what order the values come
  // back, and the send/receive counts of the collective. That schedule is built once here and
  // reused by every evaluate() call, which is why repeated evaluation (ten history levels of the
  // projection solve) costs one MPI_Alltoallv each rather than a fresh point-location pass.
  //
  // In serial the schedule degenerates to "every point is local" and evaluate() never touches MPI,
  // so there is no serial cost to carrying it.
  class LocationSet
  {
    friend class MeshPointLocator;

  protected:
    const class MeshPointLocator *locator = nullptr;
    unsigned npoint = 0;

    std::vector<LocationHandle> handles;

    // Local part of the table: for each slot owned by this rank, the source element and the local
    // coordinate found in it. Never exposed as a pointer to callers - see rule 1.
    std::vector<BulkElementBase *> local_elements;
    std::vector<double> local_s;     // npoint_local * element_dim, structure-of-arrays
    unsigned local_s_stride = 0;

    // For Project matches: the residual distance |x(s) - x_query|, kept so callers can report how
    // far a transferred value actually travelled instead of silently accepting it.
    std::vector<double> local_offset;

    // --- routing schedule, unused in serial ---
    std::vector<int> send_counts, send_displs, recv_counts, recv_displs;
    std::vector<unsigned> gather_permutation; // remote reply order -> query order

    // How the candidate set was arrived at, per point. Only for reporting - a locate that mostly
    // falls through to the widening search is a sign that the seeding is not working, which is
    // otherwise invisible because the answer is correct either way.
    unsigned n_by_walk = 0, n_by_nearest_node = 0, n_by_widening = 0;
    // How many points the cheap single-Newton pass could not place, so that the expensive
    // multi-start pass had to run. A single Newton on a strongly deformed curved element is not
    // guaranteed to converge, so this is the number that says whether that matters in practice.
    unsigned n_needing_multistart = 0;

  public:
    unsigned size() const { return npoint; }
    const std::vector<LocationHandle> &get_handles() const { return handles; }

    // Resolve query `i` to the source element and local coordinate it was matched to. Only valid
    // for a handle owned by this rank, which in serial is every located handle; returns false
    // otherwise (unlocated, or owned elsewhere). This is the ONLY route from a handle back to an
    // element, and callers must not store what it returns beyond the current operation - see rule 1
    // in the file header.
    bool resolve_local(unsigned i, BulkElementBase *&element, std::vector<double> &s) const;

    // "37 walked, 4 by nearest node, 1 widened, 0 unlocated" - for the migration's A/B reporting.
    std::string search_statistics() const;

    // Number of query points that could not be located on any rank. Callers are expected to check
    // this and report rather than fall through to a nearest-node guess unannounced.
    unsigned n_unlocated() const;

    // Largest Project offset over all located points, for diagnostics. Zero for Invert.
    double max_projection_offset() const;
    // Perpendicular offset recorded for one query; 0 for an exact inversion, -1 if unlocated.
    double offset_of(unsigned i) const;

    // Evaluate `what` at every located point. Returns a flat buffer with values_per_point()
    // entries per query, in query order; unlocated points are left at zero.
    // One collective under MPI, regardless of how many fields and time levels are requested.
    //
    // Layout per point: the requested time levels outermost, and within one level the blocks in the
    // order the EvalRequest declares them - continuous, DL, D0, DG, position, lagrangian, zeta.
    // Fixed rather than "whatever was asked for first", so a consumer can index the result without
    // consulting the request it sent.
    std::vector<double> evaluate(const EvalRequest &what) const;

    // Entries per point that evaluate() will produce for `what`.
    unsigned values_per_point(const EvalRequest &what) const;
  };

  // The index over one source mesh in one coordinate space.
  //
  // Built once and cached on the mesh (a fresh oomph::MeshAsGeomObject per call site is a large
  // part of what makes the current path slow). Backed by the already-vendored nanoflann tree in
  // kdtree.hpp - oomph-lib's equivalent acceleration is CGAL-only and switched off in this build,
  // so nothing is lost by not depending on it.
  class MeshPointLocator
  {
  protected:
    Mesh *source = nullptr;
    LocatorSetup setup;
    LocatorMode mode = LocatorMode::Invert;
    unsigned space_dim = 0;   // dimension of the coordinate space
    unsigned element_dim = 0; // dimension of the source elements

    KDTree *tree = nullptr;
    std::vector<pyoomph::Node *> nodes_by_index;
    std::map<pyoomph::Node *, unsigned> node_lookup; // inverse of nodes_by_index, for the walk

    // Node -> element adjacency as CSR rather than map<Node*,set<Element*>>: at ~1e6 queries the
    // container lookups of the latter dominate the actual geometry.
    std::vector<unsigned> node_elem_offsets;
    std::vector<BulkElementBase *> node_elem_entries;
    std::map<BulkElementBase *, unsigned> element_index;
    std::vector<BulkElementBase *> elements_by_index;

    // Per-element axis-aligned bounding box in the coordinate space, used to reject candidates
    // before any Newton solve is attempted.
    std::vector<double> element_bbox_min, element_bbox_max;

    // Precomputed affine inverse for straight-sided simplices, so their inversion is a matrix
    // multiply giving barycentric coordinates rather than an iteration. Zero for elements that
    // genuinely need Newton (curved geometry, quads/hexes).
    // How an element's geometric map can be inverted. Determined once, per element.
    enum class GeomKind : unsigned char
    {
      General = 0,  // Newton, but seeded from the affine fit below when there is one
      Affine = 1,   // exact, one matrix multiply
      Bilinear2d = 2 // exact, one quadratic - a straight-edged but non-parallelogram 2d quad
    };
    // Which reference domain a local coordinate has to land in to count as inside.
    enum class RefDomain : unsigned char
    {
      Unknown = 0, // no containment test available: Newton only, though still affinely seeded
      Simplex = 1, // s_i >= 0 and sum s_i <= 1
      Box = 2,     // s_min <= s_i <= s_max
      Prism = 3,   // wedge: s0,s1 >= 0, s0+s1 <= 1, s2 in [s_min,s_max]
      Pyramid = 4  // s2 in [s_min,s_max], s0 and s1 in [0, 1-s2]
    };
    std::vector<GeomKind> element_geom_kind;
    std::vector<RefDomain> element_ref_domain;

    // x(s) = X0 + D * u, where D's columns are the edge vectors X_k - X0 of a chosen affinely
    // independent node set and u is the coordinate in that basis. Storing D^-1 turns the query into
    // u = D^-1 (x - X0) and then s = s0 + Sdiff * u. Going through u rather than s directly keeps
    // the code independent of each element family's local coordinate convention, which is not what
    // one would guess - a 2d T-element puts node 0 at s=(1,0) and node 2 at the origin.
    //
    // Stored for EVERY element whose D is invertible, not only the exactly-affine ones: where the
    // map is not affine this is its best affine fit, which is a far better Newton starting point
    // than the element centre for a distorted element.
    // D itself is stored too, not only its inverse: in Project mode the residual x - (X0 + D u) is
    // the perpendicular offset from the element, which is both the accept/reject criterion and the
    // way competing candidate elements are ranked.
    std::vector<double> affine_basis;   // D, space_dim x element_dim per element, row-major
    // In Invert mode this is D^-1; in Project mode the least-squares pseudo-inverse
    // (D^T D)^-1 D^T, which is the same thing when D happens to be square.
    std::vector<double> affine_inverse; // element_dim x space_dim per element, row-major
    std::vector<double> affine_origin;  // X0, space_dim per element
    std::vector<double> affine_s0;      // local coordinate of the origin node, element_dim per element
    std::vector<double> affine_sdiff;   // columns s_k - s0, element_dim x element_dim per element
    std::vector<bool> has_affine_fit;

    // Bilinear coefficients for GeomKind::Bilinear2d, as x(s,t) = b0 + b1 s + b2 t + b3 s t.
    // 4 * space_dim per element.
    std::vector<double> bilinear_coeffs;

    unsigned n_affine_elements = 0, n_bilinear_elements = 0;

    bool build_affine_fit_for(BulkElementBase *e, unsigned slot);
    bool is_exactly_affine(BulkElementBase *e, unsigned slot) const;
    bool build_bilinear_for(BulkElementBase *e, unsigned slot);
    bool inside_reference_domain(unsigned slot, BulkElementBase *e, const double *s) const;
    // Move s to the nearest point of the reference domain. Needed for projection: when the
    // unconstrained closest point lies outside the element, the true closest point is on its
    // boundary, and clamping each step is what keeps the iteration there.
    void clamp_to_reference_domain(unsigned slot, BulkElementBase *e, double *s) const;
    // Damped, domain-clamped Gauss-Newton for the codimension-1 case: minimise |x(s) - x| over s.
    // Returns the achieved perpendicular offset (NOT relative), which the caller compares against
    // the local element size.
    double project_local_coordinate(BulkElementBase *e, unsigned slot, const double *x, double *s) const;
    // Representative size of one element in the coordinate space, for scaling tolerances.
    double element_size(unsigned slot) const;
    // Newton-refines s until |x(s) - x| is at machine precision; returns the achieved relative
    // residual so a caller can tell convergence from a candidate element that simply does not contain x.
    double polish_local_coordinate(BulkElementBase *e, const double *x, double *s) const;

    // Slack added to every element bounding box before it is used to reject a candidate, as a
    // fraction of the box diagonal. Curved elements bulge outside the box spanned by their nodes,
    // and a query legitimately sitting in that bulge must not be rejected.
    double bbox_slack = 0.05;

    void build_index();
    void build_element_boxes();
    void build_affine_inverses();

    bool bbox_contains(unsigned element_slot, const double *x) const;
    // Coordinate of a node in the locator's space.
    //
    // Deliberately NOT BulkElementBase::zeta_nodal for the position spaces. On a bulk element that
    // function returns exactly xi or x depending on two static flags, but InterfaceElementBase
    // overrides it to FaceElement::zeta_nodal (elements.hpp:2761), which returns the intrinsic
    // BOUNDARY coordinate instead - fewer components than the nodal dimension, so asking it for
    // component 2 of a face in 3d reads out of range. Reading the node directly is both correct on
    // an interface mesh and independent of the static flags.
    double nodal_coordinate(BulkElementBase *e, unsigned n, unsigned d) const;
    // Shift v by whole periods until it is the branch nearest `ref`. No-op when not periodic.
    double unwrap(double v, double ref, unsigned d) const;
    // A node's coordinate as this ELEMENT sees it: unwrapped onto the branch of the element's own
    // node 0. This is what makes a closed loop work. In canonical form the seam element runs from
    // z_last back to 0 and so spans the entire range, matching any query; unwrapped it runs from
    // z_last to z_first + period, which is monotone and invertible like every other element.
    double element_nodal_coordinate(BulkElementBase *e, unsigned n, unsigned d) const;

    // Wrap a query coordinate into the principal period, for periodic spaces. No-op otherwise.
    void wrap_into_period(double *x) const;

    // Try to match `x` against one element. Dispatches on `mode`; returns false if the point is
    // outside (or, for Project, farther than the offset guard allows).
    bool try_element(BulkElementBase *e, const double *x, double *s_out, double *offset_out,
                     bool allow_multistart) const;

  public:
    MeshPointLocator(Mesh *source_mesh, const LocatorSetup &setup);
    virtual ~MeshPointLocator();

    Mesh *get_source_mesh() const { return source; }
    const LocatorSetup &get_setup() const { return setup; }
    LocatorMode get_mode() const { return mode; }
    unsigned get_space_dim() const { return space_dim; }
    unsigned get_element_dim() const { return element_dim; }
    // "812/814 affine, 2 bilinear" - how many source elements avoid Newton entirely. A low fraction
    // on a simplex or structured mesh means the straightness test is rejecting elements it should
    // not, which costs an order of magnitude per query without being otherwise visible.
    std::string affine_fraction() const;

    // Locate `npoint` query points given as a flat array of npoint*space_dim coordinates.
    //
    // `hint_groups`, if given, marks which queries belong together (typically all integration
    // points of one target element) so the walk can be seeded from the previous match. It is a
    // pure optimisation - the result does not depend on it.
    LocationSet locate_batch(const std::vector<double> &coords, unsigned npoint,
                             const std::vector<unsigned> *hint_groups = nullptr) const;

    // Convenience composition of locate_batch + LocationSet::evaluate, for one-shot callers that
    // will not evaluate again. Consumers that evaluate repeatedly (the projection solve) must keep
    // the LocationSet instead, or they pay for point location once per time level.
    std::vector<double> locate_and_evaluate(const std::vector<double> &coords, unsigned npoint,
                                            const EvalRequest &what) const;
  };

}
