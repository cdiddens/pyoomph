# Boundary-node membership: over-marking after refinement, and the repair

Status: **implemented and on by default.** After an adapt a node could be marked as lying on a mesh
boundary it is not on. Nodal membership is now reconciled against the per-element face tags after every
adapt, with a cross-rank push so the ranks cannot diverge.

The part worth reading if you touch this again is §4: **a hanging node is on the boundary without being
a node of any tagged face.** Stripping those is not a harmless mislabel — it makes a later refinement
create a plain `Node` where a `BoundaryNode` was needed, which cannot be undone. That was caught by
`test_tet_refinement.py`, not by anything written for this work.

## 1. The defect

A new node gets the boundaries **shared by all its generating nodes**. Two nodes can share a boundary
label without the edge between them lying on that boundary.

* 2d triangles: the two end nodes of the father edge, `RefineableTElement<2>::get_boundaries`.
* 3d tets: the intersection over all generating nodes. Wedges: `src/wedges_and_pyramids.cpp`. Pyramid
  and brick sons: `src/elements.cpp`.
* oomph's own `RefineableQElement<2>/<3>::get_boundaries` use the identical end-node rule, so native
  quad/brick refinement over-marks the same way.
* The mesh template's intermediate-node rule does too, so the **unrefined** mesh can already be wrong
  before any adapt happens.

**The trigger is precise: an element with two or more faces on the same boundary.** Its remaining faces
then have all their vertices on that boundary, so every one of their edges is mislabelled, and each
mislabelled edge seeds more at the next refinement. Measured as (nodes marked) → (of those, on no facet
of that boundary):

```
                                          nref=1        nref=2        nref=3
tet, 2 of 4 faces on the boundary        1 of 10      10 of 35      84 of 165   (51%)
tet, 4 of 4 faces on the boundary        0 of 10       1 of 35      35 of 165   (21%)
```

So where it bites it is not marginal. But on every mesh anyone would actually write it does not bite at
all — **0 spurious nodes** across a triangular disc, a spherical octant of bricks, a tetrahedral ball, a
rectangular quad/tri mesh, a full circular disc and an unstructured gmsh tetrahedral ball, each at one,
two and three refinements, up to 64512 elements. The reason is structural: an element normally meets a
given boundary in a single face, so no interior edge has both ends on it. Two faces of one element on
the *same* boundary needs the boundary to wrap around the element — a one-element-thick shell or slot.
Reachable, but not what the shipped meshes do.

Consequences are bounded, which is why this survived so long: boundary *elements* come from the face
tags and are correct, and Dirichlet conditions are applied through those face elements. Only code that
iterates nodes *by boundary index* sees the wrong answer — user post-processing, boundary coordinates,
and tests.

## 2. Why the truth is already available

Since the face-tag work (see [adaptive_refinement.md](adaptive_refinement.md) §6) every element holds
`std::map<short, std::vector<unsigned>> face_boundaries`, seeded once from the template's facets by
`seed_face_boundaries_from_facets()`, inherited exactly at every split by `dynamic_split()` via
`face_index_in_father()`, and consumed by `setup_boundary_element_info_from_face_tags()`.

So the true node set of a boundary is the union of the nodes of its tagged faces. The repair computes
that and drops everything else.

The fix cannot be in the inheritance rule itself: an element cannot tell whether its face is on a
boundary from its own nodes — that is the circularity the rule exists to work around — so it would have
to consult the mesh's boundary-element information, which is what is rebuilt *after* an adapt. Hence a
post-adapt repair.

## 3. A complete face-node accessor

`get_vertex_nodes_of_face()` returns vertices only (it keys the `facets` map) and is not enough:
dropping a genuine mid-side node's membership would be a correctness bug, so the truth set has to be
built from *all* nodes of a face. `BulkElementBase::get_all_nodes_of_face(face_index)` is that
accessor, over two virtuals `nnode_on_face_by_index` / `node_index_on_face`.

There was no uniform accessor, and the gaps were not where one would guess:

* `nnode_on_face()` throws for wedges and pyramids, because their facets are mixed tri/quad; those
  classes carry `nnode_on_face_by_index(face_index)` instead.
* `oomph::TElement<1,*>` and `TElement<3,*>` declare neither `nnode_on_face()` nor
  `get_bulk_node_number()`.
* **`BulkElementTetra3dC2TB` has a face node that `get_bulk_node_number` cannot return.**
  `build_face_element` wires it in by hand from `Central_node_on_face{13, 12, 10, 11}`, because
  `TElement<3,3>::Node_on_face` is only `[4][6]`. Those face bubbles are created with
  `boundary_possible=true`, so they genuinely carry boundary membership. **This is the one way the
  repair could strip a real membership, and it is why the `missing` counter ships enabled.**

The cell-interior bubbles of `Tri2dC1TB`, `Tri2dC2TB` and `Tetra3dC1TB` are created with
`boundary_possible=false` and must stay out of the face sets.

**The four wedge/pyramid forwarding overrides are not optional.** `oomph::WedgeElementC1` declares a
virtual with the identical signature and `BulkElementWedge3dC1` inherits from both it and
`BulkElementBase`; without an explicit override in the derived class the two are separate vtable slots
and a call through a `BulkElementBase*` reaches the throwing default. One derived declaration legally
overrides both bases.

## 4. The repair, and the hanging-node exemption

`repair_boundary_node_membership_from_face_tags()` (`src/mesh.cpp`):

1. Bail out unless `face_boundary_tags_valid && nboundary()`.
2. **Truth set.** For every element — halo elements *included*, they are evidence — and every
   `(face, boundaries)` pair, insert all of `get_all_nodes_of_face(face)` into `truth[b]`.
3. **Decidable set.** The nodes of every non-halo element; in serial, every node. This is provably safe:
   oomph retains a foreign element as a root halo iff it shares a node with an element of my domain, and
   keeps it through pruning on the same criterion. So *if a node lies in at least one non-halo element on
   a rank, every element incident to that node is present on that rank* — and the rank's truth set for it
   is complete. Element `is_halo()` is propagated to sons, so it is reliable at this point. Node
   `is_halo()` is **not** — see §5.
4. **Per boundary, one pass** over `Boundary_node_pt[b]`, rebuilding the vector wholesale rather than
   calling `Mesh::remove_boundary_node`, which is O(n) per call and would make the sweep quadratic. Drop
   a node iff `!n->is_obsolete() && decidable(n) && !truth[b].count(n)`. Skipping obsolete nodes keeps
   `prune_dead_nodes` (which deletes an obsolete node only once it is on no boundary) out of the picture.
5. `if (bn->is_on_boundary(b)) bn->remove_from_boundary(b);`. **The guard is mandatory.** PARANOID is off
   by default, so the "not on this boundary" check inside `BoundaryNodeBase::remove_from_boundary` is
   compiled out and `nodes.cc` dereferences a possibly-null `Boundaries_pt` — a segfault, not an
   exception.

Also called at the end of `seed_face_boundaries_from_facets()`, so the template-level over-marking is
fixed once on the unrefined, pre-distribution mesh.

### 4.1 Hanging nodes must be exempt — the defect this nearly shipped

`truth[b]` is a union of the **node lists** of tagged faces, and that is not the same set as "the nodes
lying on boundary b". A **hanging** node sits inside a coarser element's facet without being one of its
nodes, so it belongs to no tagged face while being every bit as much on the boundary. In 3d it does not
even need the coarse element to be its neighbour: an edge of a boundary facet is shared by several
elements, only some of which have a face on that boundary, and if the one that refines the edge is an
interior element then the new midpoint lies on the boundary and appears in nobody's tagged face list.

Removing such a membership is **irreversible**, and it fails somewhere else entirely: when the coarse
element is eventually refined, its new node's class is chosen from its generating nodes' memberships
(`BoundaryNode` vs plain `Node`, and a plain `Node` can never become a boundary node afterwards), so the
node is born plain and the interface mesh dies with "Node ... is not a boundary node" from
`Mesh::generate_interface_elements`.

Measured on `test_tet_refinement.py::test_tet_node_sharing_ignores_node_positions`, a non-uniformly
adapted `TetCubeMesh(N=3)` that also unrefines: four nodes — all four `is_hanging()` and correctly marked
one step earlier — lost their memberships at one adapt, and the interface rebuild threw at the next. So
the repair skips any node with `is_hanging()` (`may_drop_boundary_membership()`), and the checker exempts
them too, so a mesh the repair deliberately left alone does not read as broken.

That is conservative in one direction: a genuinely spurious mark on a hanging node survives, which is
exactly the behaviour there was before any of this, and it is cleaned up by the next adapt at which the
node stops hanging. The configurations the repair exists for refine uniformly and have no hanging nodes.

### 4.2 Remove only; never add

The inheritance rule is an intersection, so the marked set is always a superset of the truth and
under-marking cannot arise from refinement. Adding is also structurally impossible:
`src/missing_masters.hpp` chooses the node *class* from the sender's flag, so a node that arrived as a
plain `Node` can never acquire membership, and `Mesh::add_boundary_node` throws on one. A
non-`BoundaryNodeBase` cannot be in `Boundary_node_pt[b]` and cannot report `is_on_boundary(b)`, so it
never becomes a removal candidate. The `missing` half of
`check_boundary_node_membership_against_face_tags()` is therefore a *diagnostic*: a nonzero value means
seeding or the §3 tables are wrong, and it should surface rather than be patched over.

## 5. The two hooks, and why their placement is load-bearing

`setup_boundary_element_info()` is called from oomph's `TreeBasedRefineableMeshBase::adapt_mesh` — the
single choke point for `adapt`, `refine_uniformly` and `refine_selected_elements`. Two no-op virtuals
were added to `oomph::Mesh` (`//FOR PYOOMPH`, recorded in `INFO_oomph-lib`):
`reconcile_boundary_node_membership_locally()` and
`reconcile_boundary_node_membership_across_processes()`.

* **Local pass** immediately after the `setup_boundary_element_info()` call, *outside* the
  `Lookup_for_elements_next_boundary_is_setup` block — the repair needs the face tags, not
  `Boundary_element_pt`. It must precede the 3d macro repositioning of hanging boundary nodes, which
  reads node membership and would otherwise snap a spurious node onto a boundary it is not on.
* **Collective push** after the closing `} // End if (this->nelement()>0)` and before the
  `classify_halo_and_haloed_nodes` block.

Both halves of that second placement **deadlock** rather than misbehave if got wrong:

1. Everything in that region is inside `if (this->nelement() > 0)`, whose own comment says it is there
   because "in a distributed problem with multiple meshes ... a particular process may not have any
   elements on a particular submesh". A collective placed inside that block hangs on such a rank.
2. `classify_halo_and_haloed_nodes` runs *after* both hooks, so `Halo_node_pt` / `Haloed_node_pt` /
   `Shared_node_pt` and `Node::is_halo()` are stale for every node this adapt created. Only the
   **element** halo lists are usable — `Mesh::halo_element_pt(p)` recomputes the leaves from the root
   halo elements on every call.

### 5.1 The cross-rank push

For each domain `d != my_rank`, walk `haloed_element_pt(d)` and pack, for every node of every haloed
element that appears in `Pending_boundary_membership_removals`, the triple `(index in the list, local
node index, boundary)`; send the count, then the triples. The receiver walks `halo_element_pt(dd)` in the
matching order and applies the same removals. Reuses the loop structure of
`check_halo_element_consistency` — the same `d`/`dd` pairing and the same "both sides walk their lists in
the same order" contract.

Why it is complete: a node that a rank cannot decide lies only in halo elements there, hence is a node of
some halo element, hence receives the owner's decision; and two ranks that both find it decidable both
see every element incident to it and compute the same answer. Traffic is zero whenever the repair is a
no-op, which is every realistic mesh.

**It is not optional extra work.** Nothing else exchanges boundary membership between ranks, so a
local-only repair would silently diverge, and `InterfaceMesh::setup_boundary_information2d` would then
build different numbers of corner elements on the halo and haloed copies.

`TemplatedMeshBase1d` overrides both hooks as documented no-ops: 1d rebuilds `Boundary_element_pt` *from*
node membership and never reads the tags, so the tags are not the 1d authority and applying the repair
there would be circular.

## 6. Default-on, with a flag as the escape hatch

`repair_boundary_node_membership` (read/write bool, default `True`) and
`check_boundary_node_membership()` → `(spurious, missing)` on all three `TemplatedMeshBase{1,2,3}d`,
mirroring the existing `identication_of_boundary_elements_by_facets` property.

The §1 measurement is the argument: on every realistic mesh the truth set equals the marked set, so the
repair provably removes nothing — a no-op that can be demonstrated rather than argued. Behind an opt-in
flag nobody would enable it and the strict xfail could not be flipped, which is the point of doing this
at all.

## 7. Verification

* **The three strict xfails now pass.** `test_curved_boundaries.py::test_boundary_node_membership_is_repaired_when_a_boundary_wraps_an_element`
  keeps the `[("halftet",1,1), ("halftet",3,84), ("singletet",3,35)]` parametrisation with the old count
  quoted in the failure message — "84 spurious" describes a regression far better than "not 0". Marked
  counts after the repair are exactly the old ones minus the spurious: 165−84 = 81, 165−35 = 130. The
  companion test on realistic meshes asserts set *equality* in both directions, which is what catches an
  incomplete face-node enumeration.
* **Every family, every space, both directions.** `test_boundary_element_identification.py` sweeps
  hex/tet/wedge/pyramid and the `all_four` mixed layout over C1, C2, C1TB and C2TB at 0–2 refinements,
  asserting `check_boundary_node_membership() == (0, 0)` *and* the same comparison against the
  **interface meshes** instead of the tags. The second is the load-bearing check: an interface mesh is
  assembled by `build_face_element()`, the routine that knows about the oddities, so it is an oracle
  independent of the §3 tables — a self-consistent check against the same tables could not catch a wrong
  one. That the C2TB tet is really covered was checked rather than assumed: its interface element has 7
  nodes, and an unrefined tet slab has 11 bulk nodes marked on a wall against 9 for C2 — the two extra
  face bubbles. Had `nnode_on_face_by_index()` returned 6, those would have shown up as `missing`.
* **MPI.** `test_mpi_boundary_membership.py` at 2 and 4 ranks on `tests/slab_mesh.py`, refining once
  **after** distributing (the initial refinement happens before distribution, where every element is
  local and the push has nothing to do). Per rank `(0, 0)`; globally, the union over ranks of marked
  positions against the union of the interface meshes' positions (neither is conclusive on one rank,
  since a halo-only node can sit on a facet owned elsewhere); plus an assertion that no two ranks
  disagree about a node they both hold.
* **The MPI test was verified to have teeth.** With the cross-rank push commented out it fails as
  intended — "tet/side: 3 nodes marked on no facet anywhere" at 2 ranks, 14 at 4, and the agreement check
  reporting `node at 0.9375,0.0625,0.25 is on ['wall'] on rank 0 but on ['side', 'wall'] on rank 1`. That
  divergence is invisible to every other check in the codebase. On the 4-rank hex slab, 55 of the 180
  nodes a rank holds are undecidable there, so the push is not a theoretical concern.

`tests/slab_mesh.py` is the defect made big enough to distribute: an N×N×1 slab of bricks or tets whose
top and bottom faces share the boundary name "wall", so every element has two faces on "wall", all eight
of its nodes are on "wall", and its four vertical edges have both ends on "wall" without lying on it.
Serial with the repair off:

```
                nref=1        nref=2
brick slab     21 spurious   135 spurious
tet slab       31 spurious   303 spurious
```

and zero with it on, in both directions. The brick numbers matter on their own: those come from oomph's
own `RefineableQElement<3>::get_boundaries`, so this is not only a pyoomph-simplex problem.

Two failures found on the way were **pre-existing and unrelated**, both confirmed by rebuilding without
this work and reproducing them identically:

* refining a **C1TB tet** segfaulted on the first `refine_uniformly` of any tet mesh.
  `BulkElementTetra3dC1TB::local_coordinate_of_node` wrote `s[0..2]` without resizing `s`, and
  `FiniteElement::get_node_at_local_coordinate` hands it a default-constructed, i.e. empty, `Vector` to
  size itself — an out-of-bounds write on an empty `std::vector`. Every other shape resizes first, so
  this one had simply been missed, and nothing had ever refined a C1TB tet. Fixed;
  `test_uniform_tet_refinement_conforming` now sweeps all four tet spaces.
* `test_adaptive_3d_campaign.py::test_ale_moving_mesh[levels0-hex_pyr]` stalled at
  `max|residual| = 8.583e-09` against a 1e-11 tolerance, deterministically, on an unrefined mesh. Not a
  Jacobian problem: the same system with SuperLU or UMFPACK gives 5.6e-15. See
  [linear_solvers.md](linear_solvers.md) for why that one case is named in `_EXACT_SOLVER_CASES` rather
  than a blanket policy being applied.

## 8. Residual risks

1. **Irreversibility** (§4.1). A wrong removal surfaces much later and far from its cause, as "is not a
   boundary node" out of `generate_interface_elements`. The first thing to suspect on such a throw is a
   node that is on the boundary without being on a tagged face.
2. **Boundary coordinates.** `remove_from_boundary` deletes the node's zeta for that boundary but never
   frees `Boundary_coordinates_pt` when the map empties, unlike its handling of `Boundaries_pt`. So
   `boundary_coordinates_have_been_set_up()` stays true on a node with no coordinates left. Only spurious
   zetas are lost, and `pyoomph/meshes/zeta.py` re-sets zeta on every interface-element node after each
   adapt — a candidate `//FOR PYOOMPH` tidy-up, not a blocker.
3. **Periodic corner detection.** `pyoomph/meshes/bcs.py` uses `len(n.get_boundary_indices()) >= 2` to
   identify corner nodes and *raises* on a master/slave mismatch. Removing a spurious second membership
   flips that branch — only on a mesh that already had spurious marks, but it turns a silent wrong answer
   into an exception.
4. **Refinement patterns move on pathological meshes.**
   `enlarge_elemental_error_max_override_to_only_nodal_connected_elems`,
   `ensure_halos_for_periodic_boundaries`, `Mesh::node_is_in_scope` (remesh interpolation skips any
   `is_on_boundary()` node in the bulk pass) and `nodal_interpolate_along_boundary` all read membership.
   Expect numbers to move on the halftet/singletet family and nowhere else.
5. **Corner elements of interface meshes.** `InterfaceMesh::setup_boundary_information2d` intersects the
   same node sets to find boundary-of-boundary corners, so smaller sets mean fewer corner elements. A
   per-rank difference there changes the element count between halo and haloed copies — which is why §5.1
   cannot be skipped and why the MPI test must check cross-rank *agreement*, not just correctness.
