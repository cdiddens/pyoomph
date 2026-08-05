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

  public:
    unsigned size() const { return npoint; }
    const std::vector<LocationHandle> &get_handles() const { return handles; }

    // Number of query points that could not be located on any rank. Callers are expected to check
    // this and report rather than fall through to a nearest-node guess unannounced.
    unsigned n_unlocated() const;

    // Largest Project offset over all located points, for diagnostics. Zero for Invert.
    double max_projection_offset() const;

    // Evaluate `what` at every located point. Returns a flat structure-of-arrays buffer with
    // values_per_point() entries per query, in query order; unlocated points are left at zero.
    // One collective under MPI, regardless of how many fields and time levels are requested.
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
    // multiply giving barycentric coordinates rather than an iteration. Empty for elements that
    // genuinely need Newton (curved geometry, quads/hexes).
    std::vector<double> affine_inverse;
    std::vector<bool> element_is_affine;

    // Face-neighbour adjacency, for walking from the previously matched element to the next query
    // instead of returning to the tree. Query points of one target element are clustered, so this
    // turns roughly n_intpt tree searches per element into one.
    std::vector<unsigned> elem_neighbour_offsets;
    std::vector<unsigned> elem_neighbour_entries;

    void build_index();
    void build_element_boxes();
    void build_affine_inverses();
    void build_neighbours();

    // Wrap a query coordinate into the principal period, for periodic spaces. No-op otherwise.
    void wrap_into_period(double *x) const;

    // Try to match `x` against one element. Dispatches on `mode`; returns false if the point is
    // outside (or, for Project, farther than the offset guard allows).
    bool try_element(BulkElementBase *e, const double *x, double *s_out, double *offset_out) const;

  public:
    MeshPointLocator(Mesh *source_mesh, const LocatorSetup &setup);
    virtual ~MeshPointLocator();

    Mesh *get_source_mesh() const { return source; }
    const LocatorSetup &get_setup() const { return setup; }
    LocatorMode get_mode() const { return mode; }

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
