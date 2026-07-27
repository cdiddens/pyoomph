#include "wedges_and_pyramids.hpp"
#include "elements.hpp" // for the complete pyoomph::BulkElementBase (RefineablePyramidElement::build delegates to build_as_pyramid_son)

// This file implements the geometry (Gauss integration rules, shape functions, face/node
// numbering, and refinement glue) for the wedge/prism and pyramid element types that
// oomph-lib itself does not support; see wedges_and_pyramids.hpp for the class overview.
namespace oomph
{

// 6-point rule for the linear (C1) wedge: the tensor product of a 3-point symmetric rule
// exact for linear functions on the triangular (s0,s1) cross-section and a 2-point
// Gauss-Legendre rule (at (1 -+ 1/sqrt(3))/2) along the extrusion direction s2.
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

// 12-point Stroud "conical product" rule for the linear pyramid: a 2x2 Gauss-Legendre grid
// in (s0,s1) collapsed conically towards the apex, combined with a 3-point Gauss rule along
// s2 (whose weights are scaled by (1-z)^2 below to account for the shrinking cross-section).
const double PyramidGaussC1::Knot[12][3] =
{
    // Stroud conical based 12 point rule
    // z = 0.5*(1 - sqrt(3/5))
    { 0.5*(1-1/sqrt(3)), 0.5*(1-1/sqrt(3)), 0.5*(1-sqrt(3.0/5.0)) },
    { 0.5*(1+1/sqrt(3)), 0.5*(1-1/sqrt(3)), 0.5*(1-sqrt(3.0/5.0)) },
    { 0.5*(1-1/sqrt(3)), 0.5*(1+1/sqrt(3)), 0.5*(1-sqrt(3.0/5.0)) },
    { 0.5*(1+1/sqrt(3)), 0.5*(1+1/sqrt(3)), 0.5*(1-sqrt(3.0/5.0)) },

    // z = 0.5
    { 0.5*(1-1/sqrt(3)), 0.5*(1-1/sqrt(3)), 0.5 },
    { 0.5*(1+1/sqrt(3)), 0.5*(1-1/sqrt(3)), 0.5 },
    { 0.5*(1-1/sqrt(3)), 0.5*(1+1/sqrt(3)), 0.5 },
    { 0.5*(1+1/sqrt(3)), 0.5*(1+1/sqrt(3)), 0.5 },

    // z = 0.5*(1 + sqrt(3/5))
    { 0.5*(1-1/sqrt(3)), 0.5*(1-1/sqrt(3)), 0.5*(1+sqrt(3.0/5.0)) },
    { 0.5*(1+1/sqrt(3)), 0.5*(1-1/sqrt(3)), 0.5*(1+sqrt(3.0/5.0)) },
    { 0.5*(1-1/sqrt(3)), 0.5*(1+1/sqrt(3)), 0.5*(1+sqrt(3.0/5.0)) },
    { 0.5*(1+1/sqrt(3)), 0.5*(1+1/sqrt(3)), 0.5*(1+sqrt(3.0/5.0))}
};

const double PyramidGaussC1::Weight[12] =
{
    // --- z0 = 0.5*(1 - sqrt(3/5)) ---
    0.25 * (5.0/18.0) * pow(1.0 - 0.5*(1-sqrt(3.0/5.0)),2),  // (1-z0)^2
    0.25 * (5.0/18.0) * pow(1.0 - 0.5*(1-sqrt(3.0/5.0)),2),
    0.25 * (5.0/18.0) * pow(1.0 - 0.5*(1-sqrt(3.0/5.0)),2),
    0.25 * (5.0/18.0) * pow(1.0 - 0.5*(1-sqrt(3.0/5.0)),2),

    // --- z1 = 0.5 ---
    0.25 * (8.0/18.0) * 0.25,  // (1-z1)^2
    0.25 * (8.0/18.0) * 0.25,
    0.25 * (8.0/18.0) * 0.25,
    0.25 * (8.0/18.0) * 0.25,

    // --- z2 = 0.5*(1 + sqrt(3/5)) ---
    0.25 * (5.0/18.0) * pow(1.0 - 0.5*(1+sqrt(3.0/5.0)),2),  // (1-z2)^2
    0.25 * (5.0/18.0) * pow(1.0 - 0.5*(1+sqrt(3.0/5.0)),2),
    0.25 * (5.0/18.0) * pow(1.0 - 0.5*(1+sqrt(3.0/5.0)),2),
    0.25 * (5.0/18.0) * pow(1.0 - 0.5*(1+sqrt(3.0/5.0)),2)
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

const double PyramidGaussC2::Knot[27][3] =
{
  { 0.104475117303480, 0.104475117303480, 0.072994024073149 },
  { 0.104475117303480, 0.463502987963425, 0.072994024073149 },
  { 0.104475117303480, 0.822530858623369, 0.072994024073149 },
  { 0.463502987963425, 0.104475117303480, 0.072994024073149 },
  { 0.463502987963425, 0.463502987963425, 0.072994024073149 },
  { 0.463502987963425, 0.822530858623369, 0.072994024073149 },
  { 0.822530858623369, 0.104475117303480, 0.072994024073149 },
  { 0.822530858623369, 0.463502987963425, 0.072994024073149 },
  { 0.822530858623369, 0.822530858623369, 0.072994024073149 },
  { 0.073593763053861, 0.073593763053861, 0.347003766038351 },
  { 0.073593763053861, 0.326498116980824, 0.347003766038351 },
  { 0.073593763053861, 0.579402470907786, 0.347003766038351 },
  { 0.326498116980824, 0.073593763053861, 0.347003766038351 },
  { 0.326498116980824, 0.326498116980824, 0.347003766038351 },
  { 0.326498116980824, 0.579402470907786, 0.347003766038351 },
  { 0.579402470907786, 0.073593763053861, 0.347003766038351 },
  { 0.579402470907786, 0.326498116980824, 0.347003766038351 },
  { 0.579402470907786, 0.579402470907786, 0.347003766038351 },
  { 0.033246742228767, 0.033246742228767, 0.705002209888498 },
  { 0.033246742228767, 0.147498895055750, 0.705002209888498 },
  { 0.033246742228767, 0.261751047882734, 0.705002209888498 },
  { 0.147498895055750, 0.033246742228767, 0.705002209888498 },
  { 0.147498895055750, 0.147498895055750, 0.705002209888498 },
  { 0.147498895055750, 0.261751047882734, 0.705002209888498 },
  { 0.261751047882734, 0.033246742228767, 0.705002209888498 },
  { 0.261751047882734, 0.147498895055750, 0.705002209888498 },
  { 0.261751047882734, 0.261751047882734, 0.705002209888498 },
};

const double PyramidGaussC2::Weight[27] =
{
  0.012124719217969,
  0.019399550748751,
  0.012124719217969,
  0.019399550748751,
  0.031039281198002,
  0.019399550748751,
  0.012124719217969,
  0.019399550748751,
  0.012124719217969,
  0.011284434356471,
  0.018055094970353,
  0.011284434356471,
  0.018055094970353,
  0.028888151952566,
  0.018055094970353,
  0.011284434356471,
  0.018055094970353,
  0.011284434356471,
  0.002311011034612,
  0.003697617655380,
  0.002311011034612,
  0.003697617655380,
  0.005916188248608,
  0.003697617655380,
  0.002311011034612,
  0.003697617655380,
  0.002311011034612,
};

 // The 6 vertices of son `son_type` in the father's local coordinates. son_type % 4 selects the cross-
 // section sub-triangle (0/1/2 = the three corner sub-tris, 3 = the inverted middle one), son_type / 4 the
 // extrusion half (0 = s2 in [0,0.5], 1 = [0.5,1]). Bottom tri = verts 0,1,2 (at zlo); top tri = 3,4,5 (zhi),
 // matching the wedge node layout. Sub-triangle vertices are listed in the father tri's (CCW) winding so no
 // son is inverted.
 void RefineableWedgeElement::son_vertices_in_father(int son_type, Vector<Vector<double>> &verts)
  {
    const double P0[2] = {0.0, 0.0}, P1[2] = {1.0, 0.0}, P2[2] = {0.0, 1.0};
    const double M01[2] = {0.5, 0.0}, M12[2] = {0.5, 0.5}, M20[2] = {0.0, 0.5};
    const double *subtri[4][3] = {
        {P0, M01, M20},  // corner v0
        {M01, P1, M12},  // corner v1
        {M20, M12, P2},  // corner v2
        {M01, M12, M20}, // inverted middle
    };
    const int tri = son_type % 4;
    const int zhalf = son_type / 4;
    const double zlo = 0.5 * zhalf, zhi = 0.5 * zhalf + 0.5;
    verts.resize(6, Vector<double>(3));
    for (int k = 0; k < 3; k++)
    {
      verts[k][0] = subtri[tri][k][0];   verts[k][1] = subtri[tri][k][1];   verts[k][2] = zlo;
      verts[3 + k][0] = subtri[tri][k][0]; verts[3 + k][1] = subtri[tri][k][1]; verts[3 + k][2] = zhi;
    }
  }

 // Not used by the geometric wedge refinement (boundary conditions of new nodes are derived directly from
 // their generating father nodes in build()); kept as a non-throwing stub for interface compatibility.
 void RefineableWedgeElement::setup_father_bounds()
  {
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
  void RefineableWedgeElement::get_bcs(int , Vector<int> &bound_cons) const
  {
    for (unsigned k = 0; k < bound_cons.size(); k++) bound_cons[k] = 0;
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
  void RefineableWedgeElement::get_edge_bcs(const int &, Vector<int> &bound_cons) const
  {
    for (unsigned k = 0; k < bound_cons.size(); k++) bound_cons[k] = 0;
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
  void RefineableWedgeElement::get_boundaries(const int &,
                                             std::set<unsigned> &boundary) const
  {
    boundary.clear();
  }

  //===================================================================
  /// Return the value of the intrinsic boundary coordinate interpolated
  /// along the edge (S/W/N/E)
  //===================================================================
  void RefineableWedgeElement::
      interpolated_zeta_on_edge(const unsigned &,
                                const int &, const Vector<double> &,
                                Vector<double> &zeta)
  {
    if (zeta.size() > 0) zeta[0] = 0.0;
  }

  //===================================================================
  /// If a neighbouring element has already created a node at
  /// a position corresponding to the local fractional position within the
  /// present element, s_fraction, return
  /// a pointer to that node. If not, return NULL (0). If the node is
  /// periodic the flag is_periodic will be true
  //===================================================================
  // Not used by the geometric wedge refinement (shared nodes are found via the father-node-keyed registry in
  // build()); a non-throwing stub for interface compatibility.
  Node *RefineableWedgeElement::
      node_created_by_neighbour(const Vector<double> &,
                                bool &is_periodic)
  {
    is_periodic = false;
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
  // Build this son wedge from its father (uniform 1->8 refinement). Mirrors RefineableTElement<3>::build:
  // the son->father coordinate map is the wedge C1 shape evaluated on the son's 6 vertices in father
  // coordinates (an affine map, since each sub-wedge is an affine image of the father); a new node's
  // "generating" father nodes are those with nonzero father shape at its father coordinate (the father
  // vertices it is the average of), which double as the shared-node registry key and the source of its
  // boundary/pin data. Fully geometric/topological -- no macro elements / locate_zeta.
  void RefineableWedgeElement::build(Mesh *&mesh_pt,
                                    Vector<Node *> &new_node_pt,
                                    bool &was_already_built,
                                    std::ofstream &)
  {
    const unsigned n_node = this->nnode();
    if (nodes_built()) { was_already_built = true; return; }
    was_already_built = false;

    OcTree *father_octree = dynamic_cast<OcTree *>(octree_pt()->father_pt());
    const int son_type = Tree_pt->son_type();
    FiniteElement *father_el_pt = dynamic_cast<FiniteElement *>(father_octree->object_pt());
    RefineableElement *father_re = dynamic_cast<RefineableElement *>(father_el_pt);
    TimeStepper *time_stepper_pt = father_el_pt->node_pt(0)->time_stepper_pt();
    const unsigned ntstorage = time_stepper_pt->ntstorage();
    if (father_el_pt->macro_elem_pt() != 0)
      throw_runtime_error("Macro elements (curved boundaries) are not yet supported for wedge refinement");

    Vector<Vector<double>> sv;
    son_vertices_in_father(son_type, sv);
    const unsigned nfath = father_el_pt->nnode();

    for (unsigned j = 0; j < n_node; j++)
    {
      // Son node j's coordinate in the father: wedge C1 shape at s_son, dotted with the 6 son vertices.
      Vector<double> s_son(3);
      this->local_coordinate_of_node(j, s_son);
      const double l1 = 1.0 - s_son[0] - s_son[1];
      const double w[6] = {l1 * (1 - s_son[2]), s_son[0] * (1 - s_son[2]), s_son[1] * (1 - s_son[2]),
                           l1 * s_son[2], s_son[0] * s_son[2], s_son[1] * s_son[2]};
      Vector<double> s(3, 0.0);
      for (int k = 0; k < 6; k++)
        for (int d = 0; d < 3; d++) s[d] += w[k] * sv[k][d];

      // (1) Reuse a father node coincident with this position.
      Node *created_node_pt = father_el_pt->get_node_at_local_coordinate(s);
      if (created_node_pt != 0)
      {
        node_pt(j) = created_node_pt;
        for (unsigned t = 0; t < ntstorage; t++)
        {
          Vector<double> prev;
          father_re->get_interpolated_values(t, s, prev);
          const unsigned nv = std::min((unsigned)created_node_pt->nvalue(), (unsigned)prev.size());
          for (unsigned k = 0; k < nv; k++) created_node_pt->set_value(t, k, prev[k]);
        }
        continue;
      }

      // Generating father nodes = those with POSITIVE father shape at s. Using positive weights (not just
      // nonzero) is essential for C2: a quadratic edge shape at the 1/4-point is {corner0:+0.375, mid:+0.75,
      // corner1:-0.125} and at the 3/4-point {corner0:-0.125, mid:+0.75, corner1:+0.375}, so keying on the
      // positive nodes gives {corner0,mid} vs {mid,corner1} -- distinct keys, whereas including the small
      // negative far-corner lobe would collide the two edge nodes onto one key (tearing the mesh). For C1
      // (linear, all weights >=0) this reduces to the corner/edge-mid/face-corner set as before.
      Shape psi(nfath);
      father_el_pt->shape(s, psi);
      std::vector<Node *> gen;
      SharedNodeKey reg_key;
      for (unsigned l = 0; l < nfath; l++)
        if (psi(l) > 1e-6)
        {
          gen.push_back(father_el_pt->node_pt(l));
          // Round the weight so bit-level FP differences between two adjacent fathers evaluating the same
          // shared-face point still collapse, while genuinely different interior positions stay distinct.
          reg_key.insert(std::make_pair(father_el_pt->node_pt(l), (long long)std::llround(psi(l) * 1e6)));
        }

      // (2) Reuse a node an already-built element created this round (keyed on the same generating
      // (node,weight) pairs; adjacent fathers produce identical pairs for a shared face/edge node, so the
      // key -- and hence the node -- is shared, but distinct interior points get distinct keys).
      if (!reg_key.empty())
      {
        std::map<SharedNodeKey, Node *>::iterator it = Shared_node_registry.find(reg_key);
        if (it != Shared_node_registry.end()) { node_pt(j) = it->second; continue; }
      }

      // (3) Build a new node. It lies on a mesh boundary iff ALL its generating nodes share one; pinned
      // values are those pinned at every generating node; boundary coordinates the generating-node average.
      std::set<unsigned> boundaries;
      bool have_bounds = false;
      for (Node *g : gen)
      {
        BoundaryNodeBase *bg = dynamic_cast<BoundaryNodeBase *>(g);
        std::set<unsigned> *sg = 0;
        if (bg) bg->get_boundaries_pt(sg);
        if (!sg) { boundaries.clear(); break; }
        if (!have_bounds) { boundaries = *sg; have_bounds = true; }
        else
        {
          std::set<unsigned> inter;
          std::set_intersection(boundaries.begin(), boundaries.end(), sg->begin(), sg->end(), std::inserter(inter, inter.begin()));
          boundaries.swap(inter);
        }
      }

      if (!boundaries.empty())
      {
        created_node_pt = construct_boundary_node(j, time_stepper_pt);
        const unsigned nval = created_node_pt->nvalue();
        for (unsigned k = 0; k < nval; k++)
        {
          bool all_pinned = true;
          for (Node *g : gen) if (!g->is_pinned(k)) { all_pinned = false; break; }
          if (all_pinned) created_node_pt->pin(k);
        }
        for (std::set<unsigned>::iterator it = boundaries.begin(); it != boundaries.end(); ++it)
        {
          mesh_pt->add_boundary_node(*it, created_node_pt);
          if (mesh_pt->boundary_coordinate_exists(*it))
          {
            Vector<double> z;
            for (Node *g : gen)
            {
              Vector<double> zg;
              dynamic_cast<BoundaryNodeBase *>(g)->get_coordinates_on_boundary(*it, zg);
              if (z.empty()) z.resize(zg.size(), 0.0);
              for (unsigned zi = 0; zi < zg.size(); zi++) z[zi] += zg[zi] / gen.size();
            }
            created_node_pt->set_coordinates_on_boundary(*it, z);
          }
        }
      }
      else
      {
        created_node_pt = construct_node(j, time_stepper_pt);
      }

      node_pt(j) = created_node_pt;
      new_node_pt.push_back(created_node_pt);
      for (unsigned t = 0; t < ntstorage; t++)
      {
        Vector<double> xp(3);
        father_el_pt->get_x(t, s, xp);
        for (int d = 0; d < 3; d++) created_node_pt->x(t, d) = xp[d];
      }
      // Interpolate Lagrangian (reference) coordinates for SolidNodes (moving mesh), like the tet build.
      if (SolidNode *sn = dynamic_cast<SolidNode *>(created_node_pt))
      {
        const unsigned nl = sn->nlagrangian();
        for (unsigned i = 0; i < nl; i++)
        {
          double xi = 0.0;
          for (unsigned l = 0; l < nfath; l++)
            if (SolidNode *fn = dynamic_cast<SolidNode *>(father_el_pt->node_pt(l))) xi += psi(l) * fn->xi(i);
          sn->xi(i) = xi;
        }
      }
      for (unsigned t = 0; t < ntstorage; t++)
      {
        Vector<double> prev;
        father_re->get_interpolated_values(t, s, prev);
        const unsigned nv = std::min((unsigned)created_node_pt->nvalue(), (unsigned)prev.size());
        for (unsigned k = 0; k < nv; k++) created_node_pt->set_value(t, k, prev[k]);
      }
      mesh_pt->add_node_pt(created_node_pt);
      if (!reg_key.empty()) Shared_node_registry[reg_key] = created_node_pt;
    }
  }

  //====================================================================
  ///  Print corner nodes, use colour (default "BLACK")
  //====================================================================
  void RefineableWedgeElement::output_corners(std::ostream &,
                                             const std::string &) const
  {
    // Debug-only output; not needed for refinement itself.
  }

  //====================================================================
  /// Set up all hanging nodes. If we are documenting the output then
  /// open the output files and pass the open files to the helper function
  //====================================================================
  // Uniform (1->8) wedge refinement keeps the mesh conforming, so there are no hanging nodes to set up yet.
  // Non-uniform (2:1) wedge hanging is a later milestone (needs a wedge face/edge neighbour finder).
  void RefineableWedgeElement::setup_hanging_nodes(Vector<std::ofstream *>
                                                      &)
  {
  }

  //================================================================
  /// Internal function that sets up the hanging node scheme for
  /// a particular continuously interpolated value
  //===============================================================
  void RefineableWedgeElement::setup_hang_for_value(const int &)
  {
    // No hanging under uniform refinement (see setup_hanging_nodes).
  }

  //=================================================================
  /// Internal function to set up the hanging nodes on a particular
  /// edge of the element
  //=================================================================
  void RefineableWedgeElement::
      quad_hang_helper(const int &,
                       const int &, std::ofstream &)
  {
    // No hanging under uniform refinement (see setup_hanging_nodes).
  }

  //=================================================================
  /// Check inter-element continuity of
  /// - nodal positions
  /// - (nodally) interpolated function values
  //====================================================================
  // template<unsigned NNODE_1D>
  void RefineableWedgeElement::check_integrity(double &max_error)
  {
    max_error = 0.0; // continuity is guaranteed by the geometric node-sharing scheme
  }

  //========================================================================
  /// Static matrix for coincidence between son nodal points and father boundaries
  /// (unused by the geometric wedge refinement; kept for interface compatibility).
  //========================================================================
  std::map<unsigned, DenseMatrix<int>> RefineableWedgeElement::Father_bound;

  // Per-round shared-node registry (see header): nodes created on a father edge/face, keyed by the set of
  // father nodes they are the average of. Cleared each refinement round in mesh.hpp.
  std::map<RefineableWedgeElement::SharedNodeKey, Node *> RefineableWedgeElement::Shared_node_registry;




 void RefineablePyramidElement::setup_father_bounds()
  {
  }

  // ================================================================================================
  //  Pyramid "red" refinement: 1 pyramid -> 6 sub-pyramids + 4 tetrahedra (mixed offspring).
  //
  //  Father local vertices (see PyramidElementC1 node numbering):
  //    A=v0=(0,0,0)  B=v1=(1,0,0)  C=v2=(1,1,0)  D=v3=(0,1,0)   base square (s2=0),   E=v4=(0,0,1) apex.
  //  The father map is linear along every edge and bilinear on the base, so a LOCAL midpoint equals the
  //  PHYSICAL midpoint -- new son vertices are just midpoints of father-local coordinates:
  //    base edge mids   : mAB=(1/2,0,0)   mBC=(1,1/2,0)  mCD=(1/2,1,0)  mDA=(0,1/2,0)
  //    base centre      : O =(1/2,1/2,0)
  //    lateral edge mids: mAE=(0,0,1/2)   mBE=(1/2,0,1/2) mCE=(1/2,1/2,1/2) mDE=(0,1/2,1/2)
  //  The 6 sub-pyramids (base 0-3 then apex 4) and 4 tets tile the father exactly (validated by a
  //  conserved-volume + machine-zero manufactured-solution test). Windings are chosen for a positive
  //  Jacobian (the inverted centre pyramid son 5 reverses its base order relative to the top pyramid son 4).
  // ================================================================================================
  void RefineablePyramidElement::son_vertices_in_father(int son_type, Vector<Vector<double>> &verts)
  {
    auto V = [](double a, double b, double c) { Vector<double> v(3); v[0] = a; v[1] = b; v[2] = c; return v; };
    const Vector<double> A = V(0, 0, 0), B = V(1, 0, 0), C = V(1, 1, 0), D = V(0, 1, 0), E = V(0, 0, 1);
    const Vector<double> mAB = V(0.5, 0, 0), mBC = V(1, 0.5, 0), mCD = V(0.5, 1, 0), mDA = V(0, 0.5, 0);
    const Vector<double> O = V(0.5, 0.5, 0);
    const Vector<double> mAE = V(0, 0, 0.5), mBE = V(0.5, 0, 0.5), mCE = V(0.5, 0.5, 0.5), mDE = V(0, 0.5, 0.5);

    verts.clear();
    switch (son_type)
    {
      // --- 6 sub-pyramids (base quad {0,1,2,3} + apex {4}) ---
      case 0: verts = {A, mAB, O, mDA, mAE};   break; // corner A
      case 1: verts = {mAB, B, mBC, O, mBE};   break; // corner B
      case 2: verts = {O, mBC, C, mCD, mCE};   break; // corner C
      case 3: verts = {mDA, O, mCD, D, mDE};   break; // corner D
      case 4: verts = {mAE, mBE, mCE, mDE, E}; break; // top pyramid (apex = father apex E)
      case 5: verts = {mAE, mDE, mCE, mBE, O}; break; // inverted centre pyramid (apex = base centre O; base reversed)
      // --- 4 tetrahedra (vertices {0,1,2,3}) filling the gaps along the base edges ---
      case 6: verts = {mAB, mAE, mBE, O};      break; // along edge AB
      case 7: verts = {mBC, mBE, mCE, O};      break; // along edge BC
      case 8: verts = {mCD, mCE, mDE, O};      break; // along edge CD
      case 9: verts = {mDA, mDE, mAE, O};      break; // along edge DA
      default: throw_runtime_error("pyramid son_type out of range [0,10)");
    }
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
  void RefineablePyramidElement::get_bcs(int , Vector<int> &bound_cons) const
  {
    for (unsigned k = 0; k < bound_cons.size(); k++) bound_cons[k] = 0;
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
  void RefineablePyramidElement::get_edge_bcs(const int &, Vector<int> &bound_cons) const
  {
    for (unsigned k = 0; k < bound_cons.size(); k++) bound_cons[k] = 0;
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
  void RefineablePyramidElement::get_boundaries(const int &,
                                             std::set<unsigned> &boundary) const
  {
    boundary.clear();
  }

  //===================================================================
  /// Return the value of the intrinsic boundary coordinate interpolated
  /// along the edge (S/W/N/E)
  //===================================================================
  void RefineablePyramidElement::
      interpolated_zeta_on_edge(const unsigned &,
                                const int &, const Vector<double> &,
                                Vector<double> &zeta)
  {
    if (zeta.size() > 0) zeta[0] = 0.0;
  }

  //===================================================================
  /// If a neighbouring element has already created a node at
  /// a position corresponding to the local fractional position within the
  /// present element, s_fraction, return
  /// a pointer to that node. If not, return NULL (0). If the node is
  /// periodic the flag is_periodic will be true
  //===================================================================
  // Not used by the geometric pyramid refinement (shared nodes are found via the father-node-keyed registry
  // in BulkElementBase::build_as_pyramid_son); kept as a non-throwing stub for interface compatibility.
  Node *RefineablePyramidElement::
      node_created_by_neighbour(const Vector<double> &,
                                bool &)
  {
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
  void RefineablePyramidElement::build(Mesh *&mesh_pt,
                                    Vector<Node *> &new_node_pt,
                                    bool &was_already_built,
                                    std::ofstream &)
  {
    if (nodes_built()) { was_already_built = true; return; }
    was_already_built = false;
    dynamic_cast<pyoomph::BulkElementBase *>(this)->build_as_pyramid_son(mesh_pt, new_node_pt);
  }

  //====================================================================
  ///  Print corner nodes, use colour (default "BLACK")
  //====================================================================
  void RefineablePyramidElement::output_corners(std::ostream &,
                                             const std::string &) const
  {
  }

  //====================================================================
  /// Set up all hanging nodes. If we are documenting the output then
  /// open the output files and pass the open files to the helper function
  //====================================================================
  void RefineablePyramidElement::setup_hanging_nodes(Vector<std::ofstream *>
                                                      &)
  {
  }

  //================================================================
  /// Internal function that sets up the hanging node scheme for
  /// a particular continuously interpolated value
  //===============================================================
  void RefineablePyramidElement::setup_hang_for_value(const int &)
  {
  }

  //=================================================================
  /// Internal function to set up the hanging nodes on a particular
  /// edge of the element
  //=================================================================
  void RefineablePyramidElement::
      quad_hang_helper(const int &,
                       const int &, std::ofstream &)
  {
  }

  //=================================================================
  /// Check inter-element continuity of
  /// - nodal positions
  /// - (nodally) interpolated function values
  //====================================================================
  // template<unsigned NNODE_1D>
  void RefineablePyramidElement::check_integrity(double &max_error)
  {
    max_error = 0.0;
  }

  //========================================================================
  /// Static matrix for coincidence between son nodal points and
  /// father boundaries
  ///
  //========================================================================
  std::map<unsigned, DenseMatrix<int>> RefineablePyramidElement::Father_bound;

  // Per-round father-node-keyed shared-node registry for the whole mixed pyramid forest (see header): the
  // pyramid-son build AND the tet-son build (in a pyramid forest) both key on shared father Node pointers, so
  // a node on a pyramid<->tet shared face is created once. Topological -> MPI-safe.
  std::map<std::set<Node *>, Node *> RefineablePyramidElement::Shared_node_registry;

  


  //////////////////

  WedgeGaussC1  WedgeElementC1::Default_integration_scheme;

  // Map local face-node index i (on facet face_index, see the node/facet sketch in the
  // WedgeElementC1 class comment) to the bulk element's node index.
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

PyramidGaussC1  PyramidElementC1::Default_integration_scheme;

  // Map local face-node index i (on facet face_index, see the node/facet sketch in the
  // PyramidElementC1 class comment) to the bulk element's node index.
  unsigned int PyramidElementC1::get_bulk_node_number(const int & face_index, const unsigned int& i) const
  {
        if (face_index==0) 
    { 
      switch (i)
      {
        case 0: return 0;
        case 1: return 1;
        case 2: return 4;
        default: throw_runtime_error("Invalid node index for face");        
      }      
    }
    else if (face_index==1) 
    {
        switch (i)
        {
            case 0: return 1;
            case 1: return 2;
            case 2: return 4;
            default: throw_runtime_error("Invalid node index for face");
        }
    }
    else if (face_index==2) {
        switch (i)
        {
            case 0: return 2;
            case 1: return 3;
            case 2: return 4;
            default: throw_runtime_error("Invalid node index for face");
        }
    }
    else if (face_index==3) {
        switch (i)
        {
            case 0: return 0;
            case 1: return 4;
            case 2: return 3;
            default: throw_runtime_error("Invalid node index for face");
        }
    }
    else if (face_index==4) {
        switch (i)
        {
            case 0: return 0;
            case 1: return 3;
            case 2: return 1;
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

PyramidGaussC2  PyramidElementC2::Default_integration_scheme;

 unsigned PyramidElementC2::get_bulk_node_number(const int& face_index,const unsigned int& i) const
    {
        if (face_index == 0)
        {
            switch (i)
            {
                case 0: return 0;
                case 1: return 1;
                case 2: return 4;
                case 3: return 5; 
                case 4: return 10;  
                case 5: return 9;   
                default: throw_runtime_error("Invalid node index for face 0");
            }
        }
        else if (face_index == 1)
        {
            switch (i)
            {              
                case 0: return 1;  
                case 1: return 2;  
                case 2: return 4;            
                case 3: return 6;  
                case 4: return 11;  
                case 5: return 10; 
                default: throw_runtime_error("Invalid node index for face 1");
            }
        }
        else if (face_index == 2)
        {
            switch (i)
            {              
                case 0: return 2;  
                case 1: return 3;  
                case 2: return 4;            
                case 3: return 7;  
                case 4: return 12;  
                case 5: return 11; 
                default: throw_runtime_error("Invalid node index for face 2");
            }
        }

        else if (face_index == 3)
        {
            switch (i)
            {              
                case 0: return 0;  
                case 1: return 4;  
                case 2: return 3;            
                case 3: return 9;  
                case 4: return 12;  
                case 5: return 8; 
                default: throw_runtime_error("Invalid node index for face 3");
            }
        }

        else if (face_index == 4)
        {
            switch (i)
            {
                case 0: return 0;                 
                case 1: return 8;   
                case 2: return 3;   
                case 3: return 5;  
                case 4: return 13;           
                case 5: return 7;   
                case 6: return 1; 
                case 7: return 6;
                case 8: return 2;
                default: throw_runtime_error("Invalid node index for face 4");
            }
        }

        throw_runtime_error("Invalid face index for pyramid element: "
                            + std::to_string(face_index));
        return 0;
    }

///////////

  // Populate a FaceElement representing local facet `face_index` of this bulk wedge
  // element: wire up its nodes (via get_bulk_node_number()), the face-to-bulk coordinate
  // mapping, and the outward normal sign, so the FaceElement can be used for e.g. boundary
  // integrals/conditions. The commented-out block below is a self-consistency check (not
  // normally compiled in) verifying that get_bulk_node_number()'s ordering actually matches
  // the local node ordering implied by face_to_bulk_coordinate_fct_pt().
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

  

  // Per-facet affine maps from a facet's own 2d local coordinate s to the wedge's 3d bulk
  // local coordinate s_bulk (facet numbering/geometry as sketched in the WedgeElementC1
  // class comment: faces 0/1 are the triangular s2=const end-caps in (s0,s1)-like
  // coordinates, faces 2-4 are quadrilateral side faces parametrized by a standard
  // [-1,1]x[-1,1] quad coordinate s that gets rescaled to the relevant [0,1] bulk range.
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

  // Derivatives d(s_bulk)/d(s_face) of the maps above, plus the bulk direction that is
  // "interior" (perpendicular to the facet) -- needed e.g. for evaluating bulk shape
  // function derivatives from a FaceElement. Not yet implemented for wedges.
  namespace WedgeElementBulkCoordinateDerivatives
  {
    void faces0(const Vector<double>& ,DenseMatrix<double>& ,unsigned& )
    {
        throw_runtime_error("Implement");
    }

    void faces1(const Vector<double>& ,DenseMatrix<double>& ,unsigned& )
    {
        throw_runtime_error("Implement");
    }

    void faces2(const Vector<double>& ,DenseMatrix<double>& ,unsigned& )
    {
        throw_runtime_error("Implement");
    }

    void faces3(const Vector<double>& ,DenseMatrix<double>& ,unsigned& )
    {
        throw_runtime_error("Implement");
    }

    void faces4(const Vector<double>& ,DenseMatrix<double>& ,unsigned& )
    {
        throw_runtime_error("Implement");
    }
  }




  // Dispatch to the appropriate WedgeElementFaceToBulkCoordinates::faceN function pointer for face_index in {0,...,4}.
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
      else
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

    // Sign to apply to the geometrically-computed face normal to make it point outward;
    // always +1 here (i.e. the face node orderings above are already chosen consistently).
    int WedgeElementBase::face_outer_unit_normal_sign(const int& ) const
    {
        return 1;
    }


//////////////////////////////////
  // Pyramid counterpart of WedgeElementBase::build_face_element(); see that function's comment.
  void PyramidElementBase::build_face_element(const int& face_index,FaceElement* face_element_pt)
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

  

  // Per-facet affine/rational maps from a facet's own 2d local coordinate s to the
  // pyramid's 3d bulk local coordinate s_bulk; faces 0-3 are the triangular side faces
  // meeting at the apex, face 4 is the quadrilateral base (see the PyramidElementC1 class
  // comment for the facet/node sketch).
  namespace PyramidElementFaceToBulkCoordinates
  {
    void face0(const Vector<double>& s, Vector<double>& s_bulk)
    {
        s_bulk[0] = s[1];
        s_bulk[1] = 0.0;
        s_bulk[2] = 1-s[0]-s[1];
    }

    void face1(const Vector<double>& s, Vector<double>& s_bulk)
    {
        s_bulk[0] = s[0]+s[1];
        s_bulk[1] = s[1];
        s_bulk[2] = 1-s[0]-s[1];
    }

    void face2(const Vector<double>& s, Vector<double>& s_bulk)
    {
        s_bulk[0] = s[0];
        s_bulk[1] = s[1]+s[0];
        s_bulk[2] = 1-s[0]-s[1];
    }

    void face3(const Vector<double>& s, Vector<double>& s_bulk)
    {
        s_bulk[0] = 0.0;
        s_bulk[1] = 1-s[0]-s[1];
        s_bulk[2] = s[1];
    }

    void face4(const Vector<double>& s, Vector<double>& s_bulk)
    {
        s_bulk[0] = (s[1]+1.0)/2.0;
        s_bulk[1] = (s[0]+1.0)/2.0;
        s_bulk[2] = 0.0;
    }
  } 

  // Derivatives of the maps above; see WedgeElementBulkCoordinateDerivatives. Not yet implemented for pyramids.
  namespace PyramidElementBulkCoordinateDerivatives
  {
    void faces0(const Vector<double>& ,DenseMatrix<double>& ,unsigned& )
    {
        throw_runtime_error("Implement");
    }

    void faces1(const Vector<double>& ,DenseMatrix<double>& ,unsigned& )
    {
        throw_runtime_error("Implement");
    }

    void faces2(const Vector<double>& ,DenseMatrix<double>& ,unsigned& )
    {
        throw_runtime_error("Implement");
    }

    void faces3(const Vector<double>& ,DenseMatrix<double>& ,unsigned& )
    {
        throw_runtime_error("Implement");
    }

    void faces4(const Vector<double>& ,DenseMatrix<double>& ,unsigned& )
    {
        throw_runtime_error("Implement");
    }
  } 

 

  // Dispatch to the appropriate PyramidElementFaceToBulkCoordinates::faceN function pointer for face_index in {0,...,4}.
  CoordinateMappingFctPt PyramidElementBase::face_to_bulk_coordinate_fct_pt(const int& face_index) const
    {
      if (face_index == 0)
      {
        return &PyramidElementFaceToBulkCoordinates::face0;
      }
      else if (face_index == 1)
      {
        return &PyramidElementFaceToBulkCoordinates::face1;
      }
      else if (face_index == 2)
      {
        return &PyramidElementFaceToBulkCoordinates::face2;
      }
      else if (face_index == 3)
      {
        return &PyramidElementFaceToBulkCoordinates::face3;
      }
      else if (face_index == 4)
      {
        return &PyramidElementFaceToBulkCoordinates::face4;
      }
      else
      {
        std::string err = "Face index should be in {0..4}.";
        throw OomphLibError(
          err, OOMPH_EXCEPTION_LOCATION, OOMPH_CURRENT_FUNCTION);
      }
    }

    BulkCoordinateDerivativesFctPt PyramidElementBase::bulk_coordinate_derivatives_fct_pt(const int& face_index) const
    {
      if (face_index == 0)
      {
        return &PyramidElementBulkCoordinateDerivatives::faces0;
      }
      else if (face_index == 1)
      {
        return &PyramidElementBulkCoordinateDerivatives::faces1;
      }
      else if (face_index == 2)
      {
        return &PyramidElementBulkCoordinateDerivatives::faces2;
      }
      else if (face_index == 3)
      {
        return &PyramidElementBulkCoordinateDerivatives::faces3;
      }
      else if (face_index == 4)
      {
        return &PyramidElementBulkCoordinateDerivatives::faces4;
      }      
      else
      {
        std::string err = "Face index should be in {0..4}.";
        throw OomphLibError(
          err, OOMPH_EXCEPTION_LOCATION, OOMPH_CURRENT_FUNCTION);
      }
    }

    // See WedgeElementBase::face_outer_unit_normal_sign(): always +1, the face node orderings above are already outward-consistent.
    int PyramidElementBase::face_outer_unit_normal_sign(const int& ) const
    {
      return 1;
    }

}