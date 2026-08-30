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

// Conforming refinement across coupled domain interfaces.
//
// Two bulk domains built from the same MeshTemplate can share a geometric interface, their nodes
// there being distinct objects tied together weakly (ConnectFieldsAtInterface / ConnectMeshAtInterface).
// The two sides are paired up element-for-element by InterfaceMesh::connect_interface_elements_by_kdtree,
// which demands an exact bijection of vertex-position sets. oomph-lib adapts meshes individually, so
// nothing makes the two sides refine the same facets -- and the moment they disagree, that matcher
// throws ("Cannot locate opposite element").
//
// This module states and restores the invariant the matcher needs:
//
//   INTERFACE CONFORMITY. For a declared opposite-interface connection (meshA, boundary bA) <->
//   (meshB, boundary bB) with offset t: the set of boundary facets of meshA on bA, translated by t,
//   equals the set of boundary facets of meshB on bB, facet for facet.
//
// Two properties of that formulation matter:
//
//  * It is about FACETS, not refinement levels. Two domains may carry different
//    _initial_uniform_refinement_level, so equal refinement_level() is neither necessary nor
//    sufficient; facet identity is what the matcher actually consumes.
//
//  * It can be checked and repaired WITHOUT ANY INTERFACE ELEMENT EXISTING. Interface meshes are
//    destroyed by clear_before_adapt() and only rebuilt afterwards, which is what makes this problem
//    awkward -- so nothing here ever looks at one. Boundary facets are read off the bulk mesh from
//    nboundary_element()/face_index_at_boundary(), the same source Mesh::generate_interface_elements
//    itself uses, and adapt() keeps that current via setup_boundary_element_info.
//
// See dev_docs/interface_refinement_coupling.md.

#pragma once

#include "oomph_lib.hpp"
#include <array>
#include <set>
#include <string>
#include <vector>

namespace pyoomph
{
  class Mesh;

  // One declared opposite-interface connection, expressed on the BULK meshes and their boundary
  // NAMES -- both of which are permanent, unlike the interface meshes that get torn down on adapt.
  struct CoupledInterfacePair
  {
    Mesh *meshA = nullptr;
    std::string bnameA;
    Mesh *meshB = nullptr;
    std::string bnameB;
    // Constant offset relating the two sides (periodic/translated interfaces): x_A + offset == x_B.
    // Empty or all-zero for the usual coincident case.
    std::vector<double> offset;
  };

  // Quantised position, the shape- and rank-independent key used throughout. The scale matches both
  // TemplatedMeshBase3d::enforce_refinement_balance and the KDTree epsilon (1e-8) used by the
  // opposite-element matcher, so a pair considered matched here is a pair that matcher will find.
  typedef std::array<long long, 3> InterfacePosKey;
  extern const double INTERFACE_COUPLING_KEY_SCALE;

  // The two globally-reduced sets describing one side of one coupled interface. Everything the
  // conformity test needs is in here; see the test itself in refinement_coupling.cpp.
  struct InterfaceSideFacets
  {
    // Leaf boundary facets, each as its SORTED vertex-position keys (2 for a line, 3 for a triangle,
    // 4 for a quadrilateral). Globally reduced under MPI.
    std::set<std::vector<InterfacePosKey>> facets;
    // Union of all those vertex keys. Globally reduced under MPI.
    std::set<InterfacePosKey> vertices;
    // This rank's own facets: (element index in mesh->element_pt(), face index, sorted vertex keys).
    // Includes halo copies, deliberately: every rank must reach the same verdict about every element
    // it holds, or the halo layer drifts out of step with its owner.
    std::vector<std::pair<std::pair<unsigned, int>, std::vector<InterfacePosKey>>> local;
    // Per facet, the pending adaptation decision of the element behind it, globally reduced:
    // bit 0 = the element is flagged for refinement
    // bit 1 = its father is flagged for unrefinement
    // bit 2 = it COULD be refined (refinement enabled and not already at max_refinement_level)
    // Only filled by collect_interface_side_with_flags; empty otherwise.
    std::map<std::vector<InterfacePosKey>, int> flags;
  };

  // Bit meanings of InterfaceSideFacets::flags.
  enum InterfaceFacetFlag
  {
    IFACET_TO_BE_REFINED = 1,
    IFACET_SONS_TO_BE_UNREFINED = 2,
    IFACET_CAN_REFINE = 4
  };

  // Gather one side. Silently yields an empty result if the mesh is null or carries no such boundary
  // (a side may legitimately be absent, e.g. when only dummy equations were created for it).
  // `topological` selects the node identity: the cross-domain topological id (see
  // pyoomph::Node::interface_topological_id) when both sides carry a complete set and the connection is
  // coincident, the quantised Eulerian position otherwise. The two sides of one pair must always be
  // collected with the same choice.
  void collect_interface_side(Mesh *m, const std::string &bname, const std::vector<double> &offset,
                              InterfaceSideFacets &out, bool topological);

  // Report how many of this rank's facets are non-conforming, globally summed so every rank agrees.
  // mode: 0 silent, 1 report to stdout, 2 throw. `when` labels the call site in the message.
  unsigned check_interface_conformity(const std::vector<CoupledInterfacePair> &pairs,
                                      const std::string &when, int mode);

  // Reconcile the per-element refine/unrefine FLAGS across every coupled interface, in the gap between
  // TemplatedMeshBase::select_for_adaptation and ::execute_adaptation -- after both meshes have decided
  // and before either has acted. This is the exact step: it works on the decision itself rather than on
  // the error that produced it, so it is not defeated by the case that defeats any error-level
  // comparison (a father whose unrefinement is vetoed by a son that does not touch the interface).
  //
  // Two directions with opposite monotonicity, run in this order and each to a fixed point:
  //   1. unrefinement -- DESELECT a father whose partner's father is not being unrefined. Only ever
  //      deselects: an unrefinement cannot be manufactured, since it needs unanimity among sons we do
  //      not control. Monotone downwards.
  //   2. refinement -- SELECT an element whose partner is selected. Monotone upwards.
  // They cannot oscillate against each other: selecting a refinement never creates a new unrefinement
  // selection, and deselecting an unrefinement never creates a new refinement selection.
  //
  // Returns the number of flag changes made (globally summed).
  unsigned harmonise_adapt_selection(const std::vector<CoupledInterfacePair> &pairs,
                                     unsigned max_rounds = 40);

  // Restore conformity by refining the coarser side of every coupled interface, interleaved with each
  // mesh's own 2:1 balancing (which itself refines further, and can therefore break conformity again),
  // to a joint fixed point. Refinement only, never unrefinement, so it terminates: levels are bounded
  // by max_refinement_level(). Returns the number of elements refined to restore FACET conformity.
  //
  // The same fixed point also closes the vertex-connected balance (see the implementation): an element
  // that touches a coupled interface only at a VERTEX carries no facet, so nothing above can see it,
  // and a refinement forced onto its facet-carrying neighbours would leave it arbitrarily coarser.
  // Those refinements are counted separately, into *n_vertex_balance when given, because they are a
  // grading closure inside one mesh and not a conformity repair -- and unlike a repair they are not
  // lossy, so a non-zero count is not the warning sign that a non-zero return value is.
  unsigned enforce_interface_conformity(const std::vector<CoupledInterfacePair> &pairs,
                                        unsigned max_rounds = 40, unsigned *n_vertex_balance = NULL);
}
