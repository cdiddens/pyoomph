# Adaptive refinement: hanging nodes, mixed element shapes, and what it took to make them correct

This is the reference for pyoomph's h-adaptivity: how a hanging node is represented, how the
refinement engine was generalised from pure quad/hex to every element shape pyoomph has, and the
defects that shaped the code along the way. It merges three earlier documents (the hanging-node
redesign of branch `new_hanging`, the mixed-adaptive-mesh engine notes of branch `mixed_adapt`, and
the validation campaign that closed them out).

Everything described here is implemented and validated unless a section says otherwise; §11 is the
list of what is not.

---

## 1. Where things stand

| shape family | mechanism | state |
|---|---|---|
| 1D lines | `BinaryTree` hanging-node refinement | works |
| 2D quads | `QuadTree` (oomph-lib's own compass machinery) | works |
| 2D triangles | pyoomph's topological tri tree route (§5.1) | works, C1/C2/TH/CR |
| 2D mixed quad+tri | cross-shape node-sharing + hanging + interface balancing (§5.3) | works |
| 3D hexes | `OcTree` | works |
| 3D tets | topological tet tree route (§5.2) | works, C1/C2/TH/CR |
| 3D wedges, pyramids, mixed | weight-augmented shared-node registry + mesh-level hang pass | works, C1/C2 |

Serial and under `mpirun -n N --distribute`, on non-adaptive, uniformly refined and two-level
non-uniform (2:1) meshes, with mixed continuous spaces (Taylor-Hood, Crouzeix-Raviart), C1-space
constraints on fields *and* positions, ALE/moving meshes and Neumann boundary conditions.

The gate is `refinement_possible()`: 2D accepts any mesh of quads and/or triangles, 3D accepts any
mesh of bricks, tets, wedges and pyramids. When it is false, `setup_tree_forest()` calls
`disable_adaptation()` and the mesh gets "adaptivity" only through full Gmsh remeshing plus field
interpolation (`pyoomph/meshes/remesher.py`).

---

## 2. oomph-lib's hanging-node machinery, and the constraints it imposes

### 2.1 `HangInfo` and `Node::Hanging_pt`

`oomph::HangInfo` (`nodes.h`) stores one hanging quantity as a linear combination of *master* nodes,
`q = sum_m w_m q(master_m)`. Masters are by construction never themselves hanging. Copy construction
is deleted, so a `HangInfo` is always heap-owned by exactly one owner.

`Node` holds `HangInfo** Hanging_pt`, an array of size `nvalue()+1`. The signed logical index `i`
maps to physical slot `i+1`:

* `i == -1` → slot 0 → **geometric/positional** hanging (the node position);
* `i >= 0` → slot `i+1` → hanging of continuously-interpolated **value** `i`.

`set_hanging_pt(hang, -1)` records which value slots currently *alias* slot 0, installs the new
geometric `HangInfo`, then re-points every aliasing slot back at slot 0. By default all values
therefore follow the geometry; the documented exception is Taylor-Hood pressure, given its own slot
via `further_setup_hanging_nodes()`.

**Three properties of this are load-bearing for everything below.**

1. **The accessors are non-virtual and inlined.** `is_hanging`, `hanging_pt`, `value`, `position` all
   read `Hanging_pt` directly, and oomph's refinement, equation numbering, error estimation, output,
   mesh-to-mesh projection and MPI hang synchronisation go through them. Overriding them on a
   pyoomph-derived `Node` would not change what oomph-lib sees. **Therefore `Node::Hanging_pt` stays
   the single source of truth**, and pyoomph's contribution is a management/naming layer deciding
   which `HangInfo` each slot points at — not a parallel representation. (`Do not invent a hang
   representation only pyoomph understands` is also what keeps MPI working: oomph's
   `synchronise_hanging_nodes` and the `missing_masters` machinery serialise straight out of
   `Hanging_pt`.)
2. **The deletion rule recognises exactly one shared pointer.** `set_nonhanging()` and `~Node()` free
   a value slot iff `Hanging_pt[ival] != Hanging_pt[0]`. Any *other* `HangInfo` referenced by two or
   more value slots is deleted once per slot → double free. So: C2/C2TB slots may alias slot 0
   (free-safe), but a non-geometric `HangInfo*` must never be installed into two slots.
3. **Every value index of a hanging space needs a non-null slot**, or oomph's
   `assign_hanging_local_eqn_numbers` will not populate `Local_hang_eqn[value_index]` for a field
   pyoomph later reads.

Constraints are enforced by **condensation** — the slave gets no equation and its influence folds
into the master columns weighted by `w_m` — not by Lagrange multipliers. Positions use a separate map,
`RefineableSolidElement::Local_position_hang_eqn` (per master, an `(n_position_type x dim)` block).

### 2.2 How pyoomph maps spaces onto slots

Each space carries a `hangindex`, set in codegen: Position, C2TB and C2 get `-1` (share slot 0); C1TB
and C1 get `-1` when the coordinate space is linear, else a **separate value index**
`nC2TB_basebulk + nC2_basebulk`. So pyoomph runs a de-facto two-pointer scheme — geometric+C2 fused,
plus a distinct C1 pointer on quadratic-coordinate elements.

`BulkElementBase::hang_info_for_space` (`src/elements.cpp`) is the single seam that resolves "which
`HangInfo` governs space S at element-local node l"; `fill_hang_info_with_equations_{for_pos,basebulk,
interface}` consume it to fill the JIT hangbuffers.

Interface-added dofs live *beyond* `ncont_interpolated_values()` and use pyoomph's own
`Local_interface_hang_eqn` plus an override of `fill_in_jacobian_from_nodal_by_fd`, because oomph's
version would index `Local_hang_eqn` out of bounds for them.

---

## 3. C1-space constraints composed with genuine hanging

`ConstrainFieldsToC1Space` / `ConstrainPositionsToC1Space` degrade a field (or the mesh position) to
the C1 space by expressing every non-vertex dof over the element's C1 corners. Historically this
*threw* if combined with genuine hanging. Making the two compose is what the flattening machinery
below exists for, and it is the section the C++ comments point at.

### 3.1 Why a purely element-local patch cannot work

The first attempt kept the composition local: route a C1-constrained field through the C1 hang
instead of its native C2 hang and rely on one-level chaining. On a two-level adaptively refined mesh
the uniform case converged and the genuinely non-conforming case diverged. Instrumentation pinned it:

* for a constrained mid-edge node at a T-junction, `hanging_pt(C1.hangindex)` returns the **quadratic
  three-master geometric hang** (weights like `(-0.125, 0.375, 0.75)`), not a linear C1 hang. The
  `0.75` master is the coarse edge's mid-node, whose value the constraint *pins*, so its contribution
  is silently dropped and only `0.375 - 0.125` of the value survives;
* expanding that pinned master `cm = 0.5(c1+c2)` needs `cm`'s *home coarse element's* C1 corners. From
  the neighbouring fine element `cm` is a C1 vertex, so its corners are not derivable there at all.
  The information is inherently cross-element;
* equivalently, pinning `cm` severs the sensitivity chain `M1 -> cm -> (c1,c2)`.

So the constraint has to be materialised as a **flattened, globally visible map on the node**, computed
once and read by every element that uses the node as a master.

### 3.2 The flattening, as implemented

Entirely inside pyoomph's JIT hangbuffer fill; oomph's `Hanging_pt`, pin state and equation numbering
are untouched.

* **Per-node stored expansion.** `NodeWithFieldIndicesBase::c1_constraint_corners` holds each
  constrained node's immediate C1-corner expansion (equal weights). It is computed in
  `BulkElementBase::setup_additional_dof_constraints` — in the element that *can* see those corners —
  so it is available when the node is later reached as a master from a neighbour. Recomputed after
  every adapt.
* **Recursive flatten.** `flatten_hang_for_value` / `flatten_hang_for_position` expand a dof into a
  weighted sum over real free leaf dofs: constrained node → its stored corners (recursed); genuinely
  hanging node → `hanging_pt(v)` masters (recursed); otherwise a real leaf, whose local eqn is
  `nodal_local_eqn` if it is one of this element's nodes, else the hang-registered
  `local_hang_eqn` / `local_position_hang_eqn`.
* **Why (mostly) no new equation numbers are needed.** The leaves reached are the coarse edge/face
  vertices oomph already registered as masters of the genuinely hanging non-vertex nodes on the same
  edge/face. The one native master the constraint drops is the coarse mid-node it pins. *Positions are
  the exception* — see §9.4.
* **`interpolate_hang_values`** uses the same flattening (`flattened_value` / `flattened_position`),
  so the raw values pushed into hanging/dummy/pinned storage for assembly input, output and restart
  are order-independent and consistent with the hangbuffer. Unlike the hangbuffer it keeps pinned
  leaves, since Dirichlet data does contribute to the interpolated value.
* **Cost.** `fill_hang_info_with_equations_{basebulk,for_pos}` take a fast path when the node is not
  itself constrained and none of its masters is (`hang_masters_are_unconstrained`), writing the plain
  hang with no map. Gated on the element's `has_additional_dof_constraints`, so a constraint-free
  problem hits exactly the old code path. Measured on adaptive Poisson (~6.8k dofs, many hanging
  nodes, min-of-5 assembly): 21.05 ms before vs 21.1 ms after, peak RSS identical.

Two Python wiring bugs had made the *interface* variant a silent no-op and are fixed:
`ConstrainFieldsToC1Space.before_assigning_equations_preorder` called the binding with its arguments
swapped (`(index, mode)` for `(mode, index)`, harmless only at index 0), and
`Problem.reapply_boundary_conditions` never recursed into `_interfacemeshes`, so an interface
element's `setup_additional_dof_constraints` was never called.

Validated by the linear residual oracle on `tests/test_constrained_adaptivity.py`: both constraint
kinds, `where`-restricted variants, 2D and 3D, two-level refinement plus a Neumann face element, a
two-domain mutual interface coupling (saddle-point conditioning limits the residual to ~1e-12 there,
so correctness was confirmed by an analytic-vs-FD Jacobian check, max diff 4.5e-9), and mixed spaces
where a partially constrained C2 field coexists with a native C1 field.

---

## 4. The refinement engine: what was reusable, and what had to be replaced

oomph-lib's refinement stack splits cleanly in two.

**Shape-neutral primitives — fully reusable.** `oomph::Tree` stores sons in an arbitrary-length
`Vector<Tree*>`; `nsons()` is `Son_pt.size()` and the son-name enums are "simply aliases for ints".
`Tree::split_if_required<ELEMENT>` reads `n_sons = new_elements_pt.size()`, not a hard-coded `2^dim`.
`HangInfo` is a generic (masters, weights) list. `complete_hanging_nodes_recursively` folds
hanging-on-hanging chains into genuine masters, merges duplicates, drops tiny weights and checks the
weights sum to 1 — which is how 3D edge-hangs already worked even though `oc_hang_helper` only handles
faces. And pyoomph's `DynamicTree::dynamic_split_if_required` already delegates son creation to the
element and sizes `Son_pt` to the returned count, so **variable son counts and heterogeneous son types
are structurally supported already**.

**The hard-wired concrete layer — not reusable.** Son numbering assumes tensor-product bisection;
`required_nsons()` returns literal 4/8; neighbour finding is compass-based (the `QuadTreeNames` /
`OcTreeNames` enums and the static rotation/reflection tables presume 4 edges / 6 faces + 12 edges of
a box); `quad_hang_helper` / `oc_hang_helper` use hard-coded `n_p`-based index formulas for a
`[-1,1]^dim` tensor-product Lagrange layout; `TreeRoot::Neighbour_pt` is keyed by compass direction.

Both new requirements — mixed simplex/pyramid shapes, and variable/anisotropic split schemes — break
exactly that concrete layer while leaving the primitives intact. So the architecture is **one generic
engine over the shape-neutral primitives**, not N shape-specific trees imitating `QuadTree`'s compass
model (which would hard-code one split scheme per shape and reproduce the rigidity being escaped).

`RefinementPattern` (`src/refinement_pattern.hpp`) is the descriptor, keyed by *(element type ×
scheme)*: child count, each child's element type (heterogeneous allowed), each child's affine
embedding into the parent reference element, and each child facet → parent facet map.
`BulkElementBase::dynamic_split` routes through `refinement_pattern()`; the default
`IsotropicSameTypeRefinementPattern` reproduces `required_nsons()` + `create_son_instance()` exactly.
The facet→parent map currently lives outside the pattern as
`BulkElementBase::face_index_in_father(my_face, son_type)`; it must move onto the pattern when
heterogeneous and anisotropic schemes arrive, since the son's shape may then differ from the father's.

`TemplatedMeshBase::build_facet_adjacency()` provides the shape-neutral neighbour primitive, keyed by
the shared vertex-node set of a facet (`std::map<std::set<Node*>, std::vector<unsigned>>`), validated
manifold by `tests/test_facet_adjacency.py`.

### 4.1 One decision worth restating: hanging nodes, not conforming refinement

Red-green/bisection conforming refinement would discard `HangInfo`,
`complete_hanging_nodes_recursively`, the MPI sync and the flattening work above, and would need its
own closure-reversion engine. The anisotropic-splits requirement is what tips "one generic engine"
firmly over "per-shape compass trees" — the latter cannot express variable splits at all.

---

## 5. The topological tree route

The route that landed is **purely topological**: neighbours, node identity and hang weights all come
from tree structure and local coordinates, never from comparing physical positions. An earlier
*geometric* generation (facet adjacency + `node_strictly_on_segment` collinearity + `locate_zeta`) was
built, validated, and then deleted; §10.3 records why, because the failure modes are instructive.

### 5.1 2D triangles

`src/refineable_telements.{cpp,hpp}`.

* **Affine map.** `son_to_father_local` / `father_to_son_local` /
  `local_coordinate_in_ancestor` / `local_coordinate_in_other_leaf` / `root_coord_to_leaf`: a full 2x2
  `A = [P0-P2, P1-P2]` represents **all four** triangle sons *including the inverted middle son*,
  which oomph's quad box (`s_lo`/`s_hi`/`translate_s`) cannot. Exact even on curved meshes, since
  refinement is defined in local coordinates — so it replaces `locate_zeta` with zero off-edge leakage
  and no Newton solve.
* **`tri_edge_neighbour(my_edge)`.** A son_type ascent tracking the edge as a pair of *father points*
  (a father vertex or a father edge-mid). The edge is interior to the father exactly when its two
  father points are not on one father edge, at which moment the ≥-neighbour is simply the sibling son —
  so no Reflect/Rotate tables, and the inverted middle son is handled naturally. Cross-root goes via
  the forest `neighbour_pt` plus a shared-vertex-node correspondence and an affine-interval descent.
  It also descends into a *non-leaf* sibling toward this element's edge, so an equal same-root cousin
  refined in a different round is found.
  > oomph's own `RefineableTElement<2>::gteq_edge_neighbour` is unusable here: the QuadTree descent
  > assumes quad son geometry, and the triangle split's inverted middle son makes it report `dl=0`
  > (a same-level neighbour) for genuine 2:1 interface edges — it misses ~2/3 of tri interfaces
  > (32 tree slots vs 96 geometric, measured by a cross-validator; never *wrong* masters, just absent).
* **`tri_hang_helper`** hangs each interpolating node on the ≥-coarser neighbour's
  `interpolating_basis` (generic over C1/C2/C2TB), with a cycle guard.
  `BulkElementTri2dC2::further_setup_hanging_nodes` installs the separate C1(TB)/C2 value-slot hangs.
* **Bubble-enriched elements.** `BulkElementTri2dC2::create_son_instance` returns a genuine
  `BulkElementTri2dC2TB` for C2TB fathers — the old `BulkElementTri2dC2(true)` bumped
  `nnode_of_space[C2TB]` to 7 while keeping 6 oomph node slots and the 6-node nodal-space map, so
  `fill_element_info` ran off the map and segfaulted. The 7th (centroid) node is interior, never on a
  boundary and never hanging; the central son reuses the father's centroid.

### 5.2 3D tetrahedra

The same recipe a dimension up. A linear tet refines 1→8 (4 corner sub-tets + 4 tiling the interior
octahedron along a fixed diagonal — a free choice, since the octahedron never touches a shared face,
so conformity holds through edge-midpoint sharing).

* **Affine 3x3 map** and **`tet_face_neighbour`** (track a face as a father-point triple; interior ⇔
  not all on one father face ⇒ sibling; cross-root through a tet-aware
  `DynamicOcTreeForest::find_neighbours` populating root face `neighbour_pt` from shared 3-corner-node
  faces, plus a corner correspondence and a point-containment descent).
* **Hanging** via `tet_hang_face` (+ `interpolating_basis`) and `tet_hang_edge` (an OcTree ascent to
  the coarse edge), with separate C1/C2 value slots and per-slot chain flattening.
* **2:1 balancing** (`enforce_refinement_balance`, run in `adapt` before the hang pass) iteratively
  refines any leaf tet with a node at `t = 1/2^nnode_1d` of an edge — meaning a neighbour is ≥2 levels
  finer — to a fixed point. The fraction is **order-aware** (C1: 1/4, C2: 1/8), because a 1-level
  neighbour already puts C2 sub-edge mid-nodes at 1/4: a naive quarter-point test false-positives every
  refined C2 tet and explodes the mesh.
* **Hang-chain flattening**: a hanging node whose master is itself hanging is re-expressed over real
  leaf nodes, so the assembled `HangInfo` has only real masters. Assembly-time flattening alone left
  ~1e-9 for tets.
* **3D Crouzeix-Raviart** needs more: the 15-node C2TB tet has 4 face-centroid bubbles + 1 volume
  bubble. A face bubble is keyed on its three son face-corner nodes so both tets meeting on a face
  build one node; the volume bubble is interior. Unlike C2/TH, C2TB produces face-*interior* fine nodes
  already at a single-level 2:1 interface, so the face pass hangs them on the coarse face's **7** nodes
  (3 corners + 3 edge-mids + face bubble) via the enriched triangle shape — which is exactly the trace
  of the 3D enriched tet shape on a face. The coarse face's own 7 nodes are excluded as slaves, or the
  face bubble hangs on itself and the flatten recurses forever.
* **Gotcha:** `continuous_spaces[…].hangindex` can be a *placeholder ≥ ncont* on a mesh without that
  field (e.g. a C1 hangindex on a pure-C2 mesh). The deleted geometric pass tolerated it via per-node
  `nvalue()` guards; the per-element helpers must filter slots to valid value ids
  (`slot == -1 || 0 <= slot < ncont`) or they bus-fault.

### 5.3 Mixed quad+tri in 2D

One domain may hold both shapes and refine through the shared QuadTree forest, with **no geometry and
no `locate_zeta`** at the interface.

* **Forest.** `DynamicQuadTreeForest::find_neighbours` resolves quad↔tri root neighbours (the
  neighbour tests already key only on shared corner nodes); `check_all_neighbours` skips oomph's
  quad-coordinate self-test whenever any tri is present.
* **Cross-shape core.** `BulkElementBase::mixed_hang_edge_node` / `mixed_hang_node_at`: a point at
  fraction `t` along the shared edge maps to the neighbour by the affine blend `(1-t)s(Pb) + t s(Qb)`
  of the neighbour's local coordinates of the shared root-edge corner *nodes*, then descends the
  neighbour root to the coarse leaf and hangs on that leaf's `interpolating_basis`, with a
  strictly-coarser level guard. The fine-side ascent to `t` is per-shape.
* **Hanging wiring.** The virtual `quad_hang_helper` is overridden on `BulkElementQuad2dC1/C2` so a tri
  neighbour routes to the cross-shape path and quad↔quad falls through to oomph. This is why oomph's
  compass `gteq_edge_neighbour` — meaningless for a tri neighbour — no longer produces the cyclic hang
  (stack overflow in `complete_hanging_nodes`) it did before.
* **Node-sharing** is the crux: a quad and a tri refined to the same level must *share* the coincident
  interface node. The virtual `node_created_by_neighbour` is overridden on the quad and extended on the
  tri — find the neighbour root, map the point into its frame, descend to the leaf, then
  `get_node_at_local_coordinate`. Uniform-refine interface duplicates: 8 → 0.
* **Interface 2:1 balancing** refines the coarser element wherever a *different-shape* node sits at the
  `t = 1/2^nnode_1d` edge fraction. Restricted to cross-shape so pure-tri multi-level is not
  over-refined.

3D mixed cannot be done this way: a hex face is a quad and a tet face is a triangle, so **hex↔tet
cannot share a facet at all**. Wedges and pyramids are the required transition elements, and the mixed
3D families accordingly share nodes through the weight-augmented `Shared_node_registry` (§7) and hang
from the mesh-level pass rather than through octree neighbour pointers.

---

## 6. Boundary identification by per-element face tags

Two boundary mechanisms used to coexist: per-node membership (oomph's, propagated to mid-edge nodes
during `build()`) and pyoomph's per-facet `facets` map. Reconstructing boundary *elements* from nodal
membership cannot distinguish a genuine boundary face from an **interior** face all of whose vertices
happen to lie on one and the same boundary. Both shapes have such a configuration: a triangle with two
edges on boundary `b` has its third, interior edge with both ends on `b`; a one-element-wide channel
whose opposite walls share a name has all four quad corners on `b`, so all four faces match. Measured
before the fix: a 2-triangle unit square with one shared boundary name gave 3/5/9 spurious boundary
faces at levels 1/2/3; the quad channel gave 8 spurious faces out of 16.

**Rejected: making the `facets` map dynamic**, either by propagating parent facets to child facets or
by keeping the map at root level and ascending at query time. Both are unsound because **ancestry does
not survive**: `TemplatedMeshBase2d::setup_quadtree_forest` (and its 1D/3D counterparts) flattens an
existing forest to the globally coarsest common refinement level and `delete`s every element below it,
re-rooting at `min_ref`; oomph's MPI `prune_halo_elements_and_nodes` does the same.

**Landed: forward propagation onto the elements themselves.** `BulkElementBase::face_boundaries` is a
`std::map<short, std::vector<unsigned>>` from local face index (signed: quads and bricks use ±1, ±2,
±3) to the boundaries that face lies on. Empty for interior elements; ~48 B plus ~112 B per tagged
face, i.e. ~2% of the ~2.16 kB marginal memory of a C1 triangle.

* **Seed** — `seed_face_boundaries_from_facets()` runs once on the unrefined mesh, right after
  `setup_facets_from_template`, where the template's facet topology is exact by construction.
* **Inherit** — `dynamic_split` copies each father face's tag onto the son faces that are part of it,
  via `face_index_in_father(my_face, son_type)`. It depends only on shape + split scheme, never on
  polynomial order, so it is one dispatch rather than an override per element class. Tets evaluate it
  from `son_vertices_in_father` against the father's four faces rather than from a table, so it cannot
  drift from the split code — which matters because the octahedron sons are orientation-corrected and
  their local vertex numbering is not obvious. Wedges and pyramids throw rather than silently dropping
  tags.
* **Query** — `setup_boundary_element_info_from_face_tags()`: one loop over elements × tagged faces,
  identical for every shape and dimension. No nodal membership is consulted, so neither false positive
  can occur.

Because the tags travel with the elements, re-rooting and halo pruning are non-events and unrefinement
needs no bookkeeping — the father still holds the tags it was given when it was built. The per-shape
`setup_boundary_element_info_{quads,tris,bricks}` routines survive only as the fallback for meshes that
never receive tags. `tests/test_boundary_element_identification.py` checks in both directions (no
interior face registered, no genuine boundary face missed) against a geometric oracle; the two
adversarial meshes go 9→0 and 8→0.

**Still open here:** `RefineableTElement<2>::get_boundaries` still marks a new mid-edge node as on `b`
whenever both edge ends are, so the spurious third edge's midpoint still joins the boundary *node* list.
Nothing in the identification path reads that and `DirichletBC` is unaffected (it goes via boundary
elements), but `nboundary_node`/`boundary_node_pt` still report them. Characterised and repaired
separately — see [boundary_node_membership.md](boundary_node_membership.md).

---

## 7. Node identity is topological, never positional

**The rule for this engine: a node is never identified by comparing physical positions.** Only tree
topology — the son_type ascent, the exact affine son/father maps, shared vertex/face-corner node
*pointers* across roots, and `get_node_at_local_coordinate`, which compares *local* coordinates.
Positions are an output of the mesh, not an index into it, and for hanging nodes they are not even
reliable (§9.5).

The engine passed through a phase where a start-of-round position snapshot was the fallback, and that
fallback was where real bugs hid. `node_created_by_neighbour` used to ask `tri_edge_neighbour` for the
one ≥-sized *leaf* neighbour across each son edge, which returns nothing in two common situations: the
neighbour is finer (the descent straddles), or the neighbour is at the right level but is no longer a
leaf — which happens whenever it was split earlier in the *same* round, since
`split_elements_if_required` creates every son tree before any `build()` runs. The 3D version had a
third gap: it only looked across faces, so a node shared only along an edge was never found.

What replaces it is `node_in_subtree_at_local_coordinate` (2D and 3D): recurse into every son that
*contains* the point (`father_to_son_local` + a barycentric containment test, so a point on a son
boundary follows all of them) and ask every element on the way — not just the leaf, since whether the
point is a node depends on the level — skipping elements whose nodes are not built yet (their built
ancestor is tested first and holds the same node). `node_created_by_neighbour` then searches this
element's whole tree root and hops outwards: in 2D to the root(s) across the edge the point lies on, in
3D breadth-first through root *face* neighbours, carrying the point by its barycentric weights on
shared corner node pointers. The 3D walk is transitive, which is what covers a node on a root *edge*:
the fan of roots around it is reached by chaining face hops even though most share no face with the
starting root.

The mixed 3D families (wedge, pyramid, brick-as-son, tet in a mixed forest) do not share by a tree walk
but by the `(father node, rounded father-shape weight)` key of
`RefineablePyramidElement::Shared_node_registry`, which is already topological — only its *lifetime*
was the problem, since it is cleared every round and the cross-round half was being done by position.
A node now remembers the key it was born with (`pyoomph::Node::refinement_generating_key`) and
`Mesh::split_elements_if_required` rebuilds a `Shared_node_snapshot` from the live mesh each round.
Both sides of a facet compute the same key, because both compute it from their own element at the level
where the node is born and those two elements share the facet's nodes — including the cross-level case,
since a node on a facet between a level-L and a level-(L+1) element was born when the finer side's
father, also at level L, split. Rebuilding from the live mesh rather than keeping a registry alive is
what makes the pointers safe: every node in the list is alive, and so are its generating nodes, because
unrefinement removes the finer nodes before the coarser ones they were built from.

`RefineableTElement<2>::Existing_node_by_position` and its `position_key` helpers are deleted, so **no
build identifies a node by its position in any dimension or shape.** The invariant is stated directly
by `test_{triangle,tet,mixed}_..._node_sharing_ignores_node_positions`: displace every hanging node's
stored position before each adapt (exactly what a stale cache looks like) over a refine/unrefine
sequence keyed on the *Lagrangian* midpoint, and require the mesh topology to come out bit-identical.

Two things worth remembering from building it:

* Gating the tet fallback on `in_pyramid_forest()` alone is **not** enough: in a three-way
  (tet+wedge+pyramid) forest a tet can be rooted in a wedge and have neighbours of any shape across a
  face, which the tet tree walk does not reach. The right predicate is the one the shared-node registry
  already uses, `in_pyramid_forest() || RefineablePyramidElement::Mixed_forest_active`.
* The 3D root walk held a `const Vector<double>&` into the work queue while pushing to it; the
  reallocation left a dangling reference and segfaulted on the first multi-root hop. Copy the point out
  of the queue.

---

## 8. Distributed adaptivity

Distributed adaptive *solves* work and are machine-zero for every 2D and 3D family, uniform, non-uniform
(2:1 cross-shape hanging) and multi-level, linear and nonlinear. Three fixes got them there, plus one
diagnostic that should be reached for first next time.

### 8.1 Halo ownership must be propagated to refined sons

`RefineableTElement<2>::build` propagates the halo-ownership tag to each son under `OOMPH_HAS_MPI`
(`Non_halo_proc_ID = father->non_halo_proc_ID()`), which the distributed
`classify_halo_and_haloed_nodes` uses to decide which rank owns each new node. The 3D builds
(`RefineableTElement<3>::build`, `build_as_pyramid_son`, `build_as_brick_son`, the wedge build) never
did. So refined 3D sons kept the default tag, `classify` mis-owned their new interface nodes, and the
per-rank halo/haloed node lists drifted apart, violating the invariant
`rankA.nhaloed_node(B) == rankB.nhalo_node(A)` that `resize_halo_nodes`,
`synchronise_hanging_nodes` and `additional_synchronise_hanging_nodes` all assume (and only *check*
under `#ifdef PARANOID`, compiled out in the opt build). That leaked an unmatched header message in
`Mesh::resize_halo_nodes`, which then poisoned the tag-0 halo-node sync: a SEGV in
`synchronise_nonhanging_nodes`, a deadlock in `synchronise_hanging_nodes`.

Symptom-patching the three exchanges to drain cleanly under asymmetry was tried and **reverted**: it
made the residual oracle pass on a *doubled* mesh (∫u came out ~2x the serial value — halo elements
double-counted) while C2 deep refinement still diverged. A self-consistent solve on a mis-classified
mesh is not a fix.

One genuine oomph-lib bug was fixed on the way:
`TreeBasedRefineableMeshBase::synchronise_nonhanging_nodes` compared the node's *current* position
`nod_pt->x(dir)` against `x_exp = get_x(t, s)` interpolated at *history level t*. Simplex meshes carry
`ntstorage() > 1` position storage with only t=0/1 initialised, so every non-vertex node was falsely
flagged as "position differs" (dmax ~0.5, not round-off), emitting ~250 spurious halo records per rank.
Reading `nod_pt->x(t, dir)` makes the comparison like-for-like and the send count drops to 0.

### 8.2 Refinement criteria must be synchronised, not just the estimator's errors

The one that took four rounds. On a distributed pure-tet non-uniform mesh, rank 1 installed 80 hanging
constraints that existed neither serially nor on rank 0, `ndof` came out 1573 against 1460 serially, and
the first Newton step gave `inf`.

**Cause:** oomph-lib's Z2 estimator carefully synchronises the errors *it* computes from haloed to halo
elements — but pyoomph then applies its own per-element **error overrides** on top (`RefineToLevel`,
`RefineMaxElementSize`, `RefineAccordingToElement`, the interface-driven
`_override_bulk_errors_where_necessary`, any user `calculate_error_overrides`), and those ran
rank-locally, *after* the synchronisation. Six geometrically identical elements ended up with
`err=0.1` (`must_refine`) on the owner and `err=0.00055` (`may_not_unrefine`) on the halo copy. Rank 0
refined them, rank 1 did not; from that point `Mesh::halo_element_pt()`/`haloed_element_pt()` — which
build their vectors by walking the *leaves* of the same roots' trees — were misaligned, so every
subsequent halo exchange was silently off, and six elements' worth of disagreement amplified into 258
geometry mismatches.

Fixed by `TemplatedMeshBase::synchronise_elemental_errors()`, called from `TemplatedMeshBase::adapt()`
on the final error vector just before it is handed to oomph-lib. There is exactly one call site into
`Mesh::adapt()`, so structurally no criterion can bypass it.

Which criteria were affected is not about *which* criterion but about *where it is stated*: a criterion
evaluated on the bulk mesh reads only the element's own geometry and level, so a halo copy necessarily
agrees with its owner; a criterion stated on an **interface** mesh reaches the bulk through
`_override_bulk_errors_where_necessary`, and a rank holds halo copies of bulk elements *without*
holding the interface elements that would override their error, so the override never arrives. Same
argument puts `_enlarge_elemental_error_max_override_to_only_nodal_connected_elems` in the affected
class.

**The lesson, and the tool it produced.** The consistency check that would have found this on day one
already existed upstream, inside `TreeBasedRefineableMeshBase::adapt()`'s `#ifdef PARANOID`, complete
with the message *"This is most likely because the error estimator has not assigned the same errors to
halo and haloed elements — it ought to!"*. pyoomph builds with `PYOOMPH_PARANOID=OFF`, so it never ran.
**When a distributed run misbehaves, look for the vendored PARANOID checks first.** That one now has a
pyoomph home: `check_halo_element_consistency()` (`src/mesh.cpp`), armed with
`PYOOMPH_CHECK_HALO_CONSISTENCY=1|throw` and off by default, runs at three points in every adapt (after
the error synchronisation, after refinement, after 2:1 balancing). The verdict is `MPI_Allreduce`d
before anyone throws — only the process that *owns* a contested element can see the disagreement, so an
un-agreed throw would be asymmetric, which is the exact failure mode the check exists to prevent. With
`synchronise_elemental_errors()` disabled it names the six offending elements by position and fails the
run in ~10 s instead of deadlocking.

Two more diagnostic lessons from the same hunt: comparing only what you suspect is wrong (flags,
hanging sets, levels) tells you nothing once the two lists being compared have themselves drifted out
of correspondence — carry an independent identity (the element centroid) in every diagnostic exchange.
And four rounds chased the symptom *downstream* (value collapse → equation numbering → hang
installation → stale halo elements), each a layer too low; the error vector, the actual input to the
whole process, was the last thing checked and the first thing wrong.

`TemplatedMeshBase3d::enforce_refinement_balance()` was made globally consistent in the same campaign
(it unions the selection across processes by quantized centroid and terminates on the global set being
empty, so every rank enters the collective `refine_selected_elements()` together). That is a real latent
inconsistency and worth keeping, but it fixed nothing on its own.

### 8.3 Hanging nodes' raw values go stale across the halo

pyoomph reads a node's **raw** value (`oomph::Data::value`, not the hanging-aware `Node::value` that
interpolates from masters on the fly) both in the JIT elemental assembly and for output. On a
distributed mesh the masters are halo nodes value-synced only at each Newton step's end, so a hanging
node's raw storage is left stale: wrong hanging values across the halo boundary and, for *nonlinear*
problems, a residual and Jacobian assembled from stale values every iteration. A single linear step
converges regardless, which is why it only showed in the final values.

`Mesh::collapse_hanging_node_values()` writes each hanging node's master-interpolated value into its own
raw storage, called from the start of `Problem::get_residuals` **and** `Problem::get_jacobian` (helper
`sync_hanging_values_if_distributed`), so it runs before every assembly, when the masters are current.
Guarded to `is_mesh_distributed()`.

### 8.4 Three bindings that read a distributed vector by global index

Not adaptivity bugs, but they destroyed the oracle that every adaptivity test relies on, so nothing
distributed could be measured until they were fixed. `oomph::DoubleVector::operator[]` is documented as
"[] access function to the **(local)** values"; under `--distribute` the vector holds only
`nrow_local()` doubles while these loops read `ndof()` of them. `get_residuals()` on a 4x4 quad Poisson
at `-n 2` returned 3.1e-17 (accidentally fine) / 9.96e+148 / `nan` at 0/1/2 refinement levels while the
solution itself was correct. `Problem::get_history_dofs()` and `get_current_dofs()` read out of bounds
the same way, and `set_current_dofs()` **wrote** out of bounds.

Fixed with `double_vector_to_global_std_vector()` (`src/nanobind/problem.cpp`) and the pair
`gather_double_vector_to_global` / `scatter_global_to_double_vector` (`src/problem.cpp`), which
redistribute to a globally replicated distribution before reading. This changes those APIs under MPI
from "silently wrong" to "globally replicated on every rank" (noted in `CHANGELOG.md`); the Jacobian
from `_assemble_residual_jacobian` deliberately stays process-local CSR. The bifurcation-handler
`get_eigenfunction` bindings index the same way and are *not* fixed — see §11.

`create_pressure_fixation()` had the matching defect on the Python side: `self.mesh.element_pt(0)` is
the *rank-local* element 0, so each rank pinned a different pressure node and the global system was
inconsistently constrained (SEGV at `-n 2`). All three fixation classes now select globally — Taylor-Hood
by the lexicographically smallest coordinate among the vertex nodes of **non-halo** elements, agreed by
`allreduce(MPI.MIN)`, with every rank pinning whichever local copy matches *including halo copies*
(a halo node must mirror the owner's pinned state or the per-rank dof counts diverge); Crouzeix-Raviart
and Scott-Vogelius by the smallest element centroid, where a rank that does not own the winner pins
nothing, since an element-internal dof exists only on its owner. Two details: selection happens in
`apply()`, not `setup()`, because refinement and distribution both rebuild the node/element objects and
`setup()` is not re-run after adaptation; and the no-candidate marker in the reduction is `(1,)` against
tagged candidates `(0, key)`, not an inf-filled tuple, because a rank with no local elements does not
know `ndim` and `(inf,) < (inf, inf)` is true.

---

## 9. Defects worth remembering

These explain why particular code looks the way it does.

### 9.1 The wedge and pyramid C1-corner tables were wrong without any adaptivity

Coupled C2+C1 Poisson, Taylor-Hood Stokes and ALE all failed under non-uniform refinement on wedges,
pyramids and every mixed layout while passing on bricks and tets, and passing on all families under
uniform refinement. Three separate bugs behind one symptom:

1. **The wedge and pyramid C2 elements never overrode the per-value interpolation hooks.** oomph-lib's
   defaults are isoparametric — `ninterpolating_node()` returns `nnode()` and `interpolating_basis()`
   returns `shape()` — so a C1 field's hanging constraint was built from the **quadratic geometric
   basis over all 18 (wedge) / 14 (pyramid) nodes** instead of the linear basis over the corner
   vertices. Their `further_setup_hanging_nodes` even carried the comment *"there can't be any problem
   here, since it is all isoparametric"*, which was precisely the wrong assumption. `BulkElementBase`
   now provides these shape-agnostically (`interpolation_value_is_C1`, `generic_ninterpolating_node`,
   `generic_interpolating_node_pt`, `generic_interpolating_basis`,
   `generic_get_interpolating_node_at_local_coordinate`).
   The same helpers went onto `BulkElementTetra3dC2`, which overrode `interpolating_basis` *alone*:
   that left `ninterpolating_node()` at 10 while the C1 basis writes only 4 entries, so callers read six
   **uninitialised** doubles (`oomph::Shape` allocates with `new double[N]`, which does not zero). It
   happened to work because a `TElement` numbers its vertices first, so the garbage was usually rejected
   by the `|psi| > 1e-12` master test — luck, not correctness.
2. **The pyramid's C1-corner table was wrong for the base centre.** `Dummy_Value_Interpolation_Map`
   listed node 13 (the base quad centre) as `{13, 0, 2}` — the mean of one diagonal — where the bilinear
   base requires all four corners. Now `{13, 0, 1, 2, 3}`.
3. **The wedge's C1-corner table had two entries swapped.** Bottom-layer edge mids 3, 4, 5 were listed
   over corner pairs (0,1), (1,2), (0,2), but node 4 sits at the 0–2 midpoint and node 5 at the 1–2
   midpoint. The top layer already had the right pattern.

Bugs 2 and 3 were found by the Green identity ∫v = ∫u² (§10.1), which read 0.29 on pyramids and 0.089 on
wedges *on a non-adaptive mesh*, separating "the constraint is wrong" from "the hanging is wrong". The
tables were then verified against actual nodal positions — on an affine element geometry and a C1 field
interpolate identically, so each listed corner set must average to its target node's position. That
check found bug 3 after two rounds of reading the table by eye had missed it.

### 9.2 The C1-constraint vertex guard was 2D-shaped

`ConstrainFieldsToC1Space` threw `Cannot enforce a degration to C1 on a C1 vertex node` under
non-uniform 3D refinement on **all** families, bricks included — pre-existing, not a regression, and
passing only in the one narrow configuration the then-existing test happened to use (16 of 16
combinations of mesh / boundary conditions / band wall / live-vs-pinned C1 field failed except that one).

The code below the guard already handles the vertex case correctly: `c1_corner_lookup.find(l)` misses
for a node this element sees as a C1 vertex, so nothing is installed here and the hang comes identically
from the element(s) where the node *is* a non-vertex. The guard aborted before that code could run. It
demanded the node hang on the C1 slot, which holds in 2D (a coarse edge-mid node's C1 value *is* the
mean of the two coarse corners) but not in 3D, where a father's face-centre and volume-centre nodes also
become sons' vertices. The guard now tests the condition its own error message describes — asking to
degrade to a C1 space that does not exist (`!has_C1_fields`).

### 9.3 The field constraint installs a registered hang; pinning does not

`ConstrainFieldsToC1Space` originally **pinned** each constrained non-vertex value and expanded it via
`c1_constraint_corners` only in the assembly flatten. A pinned dof is not processed by oomph's hanging
machinery, so its C1-corner masters are never registered — and when a constrained mid-node is itself a
master of a finer node's hang, the flatten drops it (pinned leaf) instead of resolving it through to the
real coarse corners. Symptom: correct solution, wrong Jacobian, geometric Newton hitting the adaptive
cap. Quads never hit it because `RefineableQElement::setup_hang_for_value` gives every value a genuine
registered hang; simplex `setup_hang_for_value` is a no-op.

`BulkElementBase::setup_additional_dof_constraints` now gives each constrained non-vertex value a
genuine linear value-slot `HangInfo` on the element's C1 corner nodes. oomph then registers those
masters and resolves any master that is itself hanging via `complete_hanging_nodes` — exactly the quad
behaviour, and MPI-consistent. It is order-independent: a node that is a C1 vertex of a finer neighbour
(a 2:1 coarse mid) is skipped there and gets its identical hang from the element where it is a
non-vertex. It is **never pinned** — a pin from the vertex side would clobber the hang,
order-dependently, which is fatal under MPI.

### 9.4 Positions need their hang masters registered explicitly

`ConstrainPositionsToC1Space` worked on non-adaptive and uniformly refined meshes of every 2D and 3D
family, and aborted the moment 2:1 hanging appeared, with `Assertion local_eqn < eleminfo->ndof` in the
generated residual code.

`assign_additional_local_eqn_numbers` used to claim *"No separate equation numbers are required, because
the flattened leaf dofs are exactly the coarse vertices that oomph-lib already registered as hang
masters."* **That is false for positions.** oomph registers position-hang masters by walking the
geometric hang of an element's *own* nodes only. But a constrained node's `c1_constraint_corners` are
written by whichever element sees that node as a non-vertex — which may be a **neighbour**. So the
corners can be vertices of a neighbouring element that this element never registers.

`register_c1_constraint_position_masters()` (called from `assign_additional_local_eqn_numbers`, which
oomph documents as running after all other numbering, so `Local_position_hang_eqn` is populated and
`ndof()` is final) walks each constrained node's position redistribution with the same recursion as
`flatten_hang_for_position`, collects the leaves, and registers any that are neither nodes of this
element nor already known, allocating local equation numbers via `add_global_eqn_numbers`.

Two dead ends, kept because they cost real time:

* **Installing the constraint as a geometric hang** (`set_hanging_pt(hang, -1)`) so oomph's own
  `assign_hanging_local_eqn_numbers` would register the masters. The stated objection — that it "couples
  the dominant C2 values" — is *not* correct: `Node::is_hanging(i)` for `i >= 0` reads `Hanging_pt[i+1]`
  with no fallback to the geometric slot. The real obstacle is `set_hanging_pt(hang, -1)` itself:
  before overwriting slot 0 it re-points every value slot holding the *same* pointer at the new hang,
  and when a node hangs in nothing every slot is null *including* the geometric one, so a naive call
  makes every field hang on the C1 corners. That is surmountable (snapshot, install, restore) and was
  implemented and verified — and it still fails: the global dof count collapses (2D quad base mesh: 139
  with `pin_position`, 59 with the hang) and Newton diverges to ~1e+54 from the first step, on the
  *non-adaptive* mesh. Installing refinement-style geometric hangs on an unrefined mesh is not something
  oomph's dof accounting expects.
* The abort itself was unattributable because `RefineableElement::local_position_hang_eqn` reads the map
  with `std::map::operator[]`, which **default-constructs an empty `DenseMatrix<int>`** for an
  unregistered node; indexing that is undefined behaviour and in practice returns a junk equation number
  that surfaces much later. `BulkElementBase::position_hang_eqn_or_throw` now checks membership and names
  the offending node, which is what turned the abort into a diagnosis and pointed straight at the missing
  registration.

### 9.5 Two ways stale hanging state corrupted the mesh

* **`refine_eigenfunction` tore the mesh (serial).** `Problem._adapt_with_interfacial_errors`, when
  adapting on an eigenfunction, writes the eigenfunction into the dof vector, runs the Z2 estimator, and
  restores the base dofs with `set_all_values_at_current_time`. Evaluating an element calls
  `interpolate_hang_values()`, which pushes the *perturbed* master interpolation into each hanging node's
  raw storage; the restore puts the masters back, but a hanging node has no dof, so its raw **position**
  stays where the eigenfunction put it. Measured on an azimuthal-stability run: 364 hanging nodes, max
  position error 6.8e-3 of the domain scale surviving the restore. Refinement then ran on that mesh and
  the position-snapshot node sharing (§7) failed to recognise the displaced nodes, building 14 duplicate
  pairs; `load_state` of the resulting dump died with `Expected 5988 in state file, but read 6002`.
  `Mesh::interpolate_hanging_values()` re-establishes the invariant right after the restore. Unlike
  `collapse_hanging_node_values` (§8.3) it covers nodal *positions*, which is what node sharing depended
  on. Both fixes are kept even though §7 makes the tear impossible on its own: the stale hanging state was
  wrong on its own account, since it is also what new nodes inherit their values from.
* **Refined tri/tet nodes were missing their interpolated Lagrangian coordinates.**
  `RefineableTElement<2>::build` and `<3>::build` set every new node's Eulerian position `x` and its
  field values but **never set the Lagrangian coordinates `xi`**, so new nodes kept `xi = 0` while `x`
  sat at the correct midpoint (1752 nodes with `x != xi`, worst discrepancy 0.5; quads had 0).
  `LaplaceSmoothedMesh` and any solid residual are written in terms of `x - xi`, so the identity mesh
  looked grossly deformed: a spurious mesh residual of 27.5, present even for *conforming*
  `refine_uniformly` with zero hanging nodes. That drove the mesh to move and blew the conditioning to
  cond ≈ 2e8, on which MKL Pardiso non-deterministically produced a garbage factorisation
  (`||J·dx+r|| ~ 1e20`) — which is why an earlier investigation mis-attributed the whole thing to
  "Pardiso fragility". Both builds now interpolate `xi` from the father with the same geometric shape
  functions used for `x`, guarded on `dynamic_cast<SolidNode*>`. (oomph's
  `SolidFiniteElement::interpolated_xi` returns 0 here because pyoomph carries the Lagrangian
  coordinates per node but registers no Lagrangian dimension at the element level, hence the manual
  interpolation; this mirrors what `RefineableSolidQElement` does for quads via `get_x_and_xi`.) After
  the fix: identity-mesh residual 17.5 → 1e-16, `x != xi` count 1752 → 0, one-step convergence, cond
  2e6 (on par with quads).
  > An FD Jacobian check cannot catch this: the Jacobian is self-consistent with the *spurious* residual.
  > FD verifies dJ/dx, not that the residual is physically zero.

### 9.6 oomph's quad neighbour lookup adopted triangle nodes at a mixed interface

Reported as "the final output step has tears in the mesh" on a gmsh mesh with a `Quads=1` boundary layer
inside a triangular mesh. A single `adapt()` moved two nodes that already existed — coarse triangle
mid-side nodes, by 0.19 and 0.086 domain units — and folded eight elements.

`BulkElementQuad2dC1/C2::node_created_by_neighbour` calls `oomph::RefineableQElement<2>::node_created_by_neighbour`
first and only falls back to the cross-shape `mixed_quad_shared_node` when the base declines. The base
maps the son node's fractional position into the edge neighbour with the **quad box map** and asks it
`get_node_at_local_coordinate`. At a mixed interface the neighbour is a triangle, for which that
coordinate means nothing — but a triangle's node coordinates live in `[0,1]^2` and a quad's in
`[-1,1]^2`, so the frames overlap and the lookup **matches a triangle node by accident**: a quad son's
E-edge mid node at `s_fraction=(1,0.5)` maps to `s=(1,0)`, which is the triangle's vertex 0.

Why it tears rather than merely mis-shares: the quad son adopts that node, so the triangle side creates
its own node at the position the quad's should have had (a coincident duplicate), and the adopted node —
a real, non-hanging node of the *coarse* triangle — becomes an edge-mid of the fine quad, hangs on the
coarse tri leaf, and is dragged onto the quad's edge by `interpolate_hang_values()`. Every triangle
owning that node folds. The visible damage is geometric; the cause is purely topological, and the final
mesh no longer shows the node as hanging.

Fixed in the vendored copy (`//FOR PYOOMPH` in `refineable_quad_element.cc`, recorded in
`INFO_oomph-lib`): skip a neighbour that is not a `RefineableQElement<2>`. Pure-quad meshes are
unaffected; upstream oomph-lib has no mixed quad+tri forests.

**Why the existing tests missed it:** whether the collision happens depends on `neigh_edge`/`translate_s`,
i.e. on how the two roots happen to be oriented. A sweep over 128 variants of the hand-built
`MixedRectMesh` (quad corner rotation × triangle split × refinement order × two sizes) produced **zero**
accidental matches. An unstructured gmsh mesh produces all of them — which is why the regression test is
a gmsh boundary-layer mesh asserting that refinement moves no pre-existing node, leaves no coincident
pair and folds no element (before: displacement 0.11–0.18, 5–20 coincident pairs, 49–180 folded elements,
and uniform refinement segfaulting outright).

The same trap exists in `RefineableQElement<3>::node_created_by_neighbour` and is currently unreachable
(a mixed 3D forest gets no octree neighbour pointers at all, and a brick inside a mixed forest refines
through `build_as_brick_son`). Verified rather than assumed — an instrumented build counted zero non-brick
neighbours reaching that function across the whole 3D campaign — and guarded anyway, because the trap
springs the moment a topological cross-shape 3D neighbour finder starts populating those pointers.

### 9.7 Two smaller ones

* **`DynamicOcTreeForest::check_all_neighbours` inspected only `Trees_pt[0]`** when deciding whether to
  skip oomph's brick compass self-test. In a mixed forest whose first root is a brick, the guard did not
  fire and the self-test ran on a forest for which `find_neighbours()` had deliberately set no neighbour
  pointers, reporting a bogus `Max. error in octree neighbour finding: 1.24373 is too big` or running
  away into an OOM. Now scans every tree, as the 2D version already did.
* **A refined father can end up in `Mesh::Element_pt`.** `Tree::~Tree` deletes `Object_pt` only for
  father nodes; leaves live in `Element_pt` and are deleted by `~Mesh`. The invariant is that after adapt
  `Element_pt` holds only leaves. An exception thrown *mid-`adapt_mesh`* — after splitting, before the
  `stick_leaves_into_vector` rebuild — leaves refined fathers in `Element_pt`, which are then freed twice
  (`double free or corruption`, cascading heap corruption). The fix is always to stop the abort rather
  than to catch it. Debugging note: the tree holds `RefineableElement*` and `Element_pt` holds
  `FiniteElement*` — different values under multiple inheritance, so identity checks need
  `dynamic_cast<void*>`, which is unusable on the already-freed side.

---

## 10. Oracles, and diagnostic techniques that paid off

### 10.1 Four layers, because "it converged" is not "it is correct"

The single most useful lesson of the validation campaign: the prediction *"serially, everything already
works, so these are codification tasks"* held in 2D and was wrong in 3D, and what broke the deadlock was
oracles that can fail on a **non-adaptive** mesh.

1. **`max|residual|` ≈ 0.**
2. **One Newton step removes the whole residual.** These problems are linear, so an exact analytic
   Jacobian goes from O(1e-2) to machine zero in one step. Expressed as the *ratio* `conv[1]/conv[0]`,
   not as an iteration count — iteration counts are tolerance-dependent (Crouzeix-Raviart lands at 1.9e-8
   after a perfect first step and takes a cosmetic second one purely because the Newton tolerance is
   1e-8). Calibrated: the worst non-CR ratio across the whole 2D matrix is 1.5e-13, so the bound is 1e-10
   — three orders of headroom and eight or more below what an inconsistent Jacobian gives. This is what
   catches a constrained or hanging dof that was pinned instead of given a registered hang, and under MPI
   it additionally exercises §8.3.
3. **Cross-discretisation agreement.** The same physical problem on quads, on two triangle splits and on
   the mixed mesh must give the same global angular momentum, *and refinement must bring them together* —
   the spread drops from 3% at level 0 to 0.2% at two-level. A tear at the quad↔tri interface leaves free
   nodes and shifts the integral instead of converging it, which no residual check would notice.
4. **Exact discrete identities.** `ndof(constrained) < ndof(unconstrained-on-top) < ndof(baseline)` on
   every mesh and level, so a silently inert constraint cannot pass. And the **Green identity ∫v = ∫u²**,
   which holds *exactly* once `u` is restricted to `v`'s space (verified to ~5e-18 on every mesh and
   refinement state, and asserted *not* to hold for the unconstrained baseline). This is the one that
   found §9.1.

Two calibration findings worth keeping. **A test can be a no-op without looking like one:** with `u`
Dirichlet on all four walls, the top-edge C2 mid-node values are pinned anyway, so
`UnconstrainFieldsFromC1Space @ "top"` restores nothing. The cases therefore give `u` and `v` a natural
(zero-flux) condition on `top` — which also makes the Green identity's boundary terms drop, so the same
choice buys both. And **do not assert on an observable that is zero by symmetry:** ∫u_x and ∫u_y vanish
identically for this forcing on this box, so the first MPI version compared round-off against round-off.

**The reset-and-resolve oracle** is the sharpest variant: zero all dofs on the final adapted mesh and
solve. A linear problem reaches machine zero in *one* Newton step iff the Jacobian is exact. It is what
diagnosed the unrefinement defects, where the observable symptom (a nodal pressure fixation "failing")
was several steps downstream of the wrong Jacobian.

### 10.2 FD oracles need central differences and an eps sweep

A forward-difference FD Jacobian check reported the tri ALE coupling `d[field]/d[mesh_position]` as
inexact on adapted meshes, off by 200–1400, and only for hanging refinement. **It was a false positive.**
The tri's *relative* forward-difference error (~8e-4) is identical to the quad's and scales **linearly
with eps** (3738 → 376 → 37.7 → 3.8 as eps goes 1e-6 → 1e-9) — the textbook signature of O(eps·f″)
truncation. Central difference at eps=1e-7 matches analytic to ~6.5e-7 relative. The tri only looked
worse in absolute terms because refinement plus compression produces a small distorted element and hence
a stiff entry (analytic ≈ 4.6e5 vs the quad's ≈ 55). That misdiagnosis stood for a long time and sent a
whole investigation the wrong way.

So: **always use central differences with a relative tolerance for ALE Jacobian verification.**

And when an FD "inexactness" survives that, **sweep eps**. A genuine structural gap looks completely
different: analytic ≡ 0.0 while the finite difference is a *constant* ≈ 0.28 across eps 1e-5…1e-10. A
non-shrinking FD against an exact zero is a missing sparsity entry, not truncation.

### 10.3 Why the geometric hanging route was abandoned

The first working tri/mixed implementation found the coarse neighbour by facet adjacency and selected the
fine nodes by geometric collinearity (`node_strictly_on_segment`, tolerance `d2 <= 1e-14*len2`). It was
correct on straight meshes and reproduced oomph's quad hanging to machine precision, but it failed in
three distinct ways that are all the same mistake:

* **Curved edges.** On a gmsh-generated C2 mesh a fine vertex on a 2:1 coarse edge sits on the *curve*,
  off the straight chord by `d2/len2 ≈ 4e-6` — 1428 rejections on one live problem. A rejected node is
  never registered as hanging, so its value came from the dummy-value interpolation in the *residual*
  path while the *Jacobian* had no matching redistribution: residual depends on the coarse corners,
  Jacobian does not.
* **Compressed meshes.** Under aggressive mesh compression the collinearity scan captured nodes across
  refinement levels, forming two overlapping "coarse" edges on a straight boundary so that A hung on B and
  B hung on A — a mutual hang cycle.
* **Coincident nodes.** At a T-junction two distinct coincident nodes, each the C2 mid of a different
  coarse element, hung on each other. Pointer-equality and basis-weight-≈1 shared-node tests each fixed
  one case and broke the other.

Each was patched (a curvature-aware quadratic-edge test, a physical-distance shared-node test, a flatten
depth guard turning cycles into clean errors instead of stack overflows) and each patch exposed the next.
The topological route of §5 removes the whole class: refinement is *defined* in local coordinates, so a
topological answer is exact on curved, moved and compressed meshes alike. A `validate` mode
cross-checking the two routes node-by-node is what proved the tree route a correct superset before the
geometric one was deleted.

### 10.4 One physics conclusion, so it is not re-derived

Adaptive refinement chasing a genuine singularity does not converge, and no adaptivity setting fixes it.
On a pinned evaporating droplet, tri refinement drives `h` small at the triple point, the slip-regularised
contact-line constraint block (slip velocity ↔ kinematic BC ↔ pressure datum ↔ mesh-connection
multipliers) goes ill-conditioned with coefficients ~ μ/(L_slip·h), and the interpolated guess falls
outside Newton's basin. Established, with each alternative explanation ruled out by measurement: the
Jacobian after re-adapt is FD-exact; it is not unrefinement (disabling coarsening still blows up); it is
not sliver elements (max aspect ratio 1.63, zero elements above 10); the Z2 error peak at the triple point
is legitimate and *quads estimate it higher*, so it is not a tri-estimator artifact; a global
`max_refinement_level` cap does not help; and a larger slip length makes it **worse**, because there are
two singularities at the pinned triple point — the hydrodynamic slip one *and* an evaporative-flux one —
and the slip length regularises only the first. The lever, if one is wanted, is a region-specific
refinement cap or minimum element size at the contact line, or a block-preconditioned solve for that
constraint block. The adaptation machinery is not the problem. A positive control (a flat evaporating
film with the same free-surface constraint but no singular triple point) re-adapts through a full
transient without trouble.

---

## 11. What is open

* **Anisotropic / variable split schemes.** `RefinementPattern` was built for them but only the isotropic
  same-type pattern exists. What is additionally needed: scheme *selection* (default isotropic, then a
  user override, then a directional error estimator — a per-direction `LagrZ2ErrorEstimator` or a
  recovered-Hessian anisotropy indicator); a **direction-aware balance rule**, since an x-split
  neighbouring a y-split produces hanging configurations the uniform 2:1 rule does not describe; and
  `rebuild_from_sons` knowing which pattern produced the sons in order to invert it (it is pervasively
  `QuadTreeNames`-specific below its DG loop). The hang machinery itself is already scheme-agnostic.
* **3D mixed across a hex↔tet boundary** needs the transition cell (§5.3); non-uniform wedge/pyramid
  hanging still comes from the mesh-level pass rather than a topological cross-shape face/edge neighbour
  finder.
* **Tet per-element hanging inside `adapt_mesh`.** The tet driver runs in `post_adapt` (a mesh loop), not
  in `adapt_mesh`'s per-element hooks, because the chain flatten inherently needs an all-elements-done
  post-pass. Consequence: a `refine_selected` / `custom_adapt` hanging gap.
* **Distributed eigensolves.** `--distribute` with SLEPc segfaults in oomph's
  `Problem::parallel_sparse_assemble` reached via `get_eigenproblem_matrices`, *including on a
  non-adaptive mesh with zero hanging nodes* — a general distributed-eigen-assembly defect, not an
  adaptivity one. Enabling them is: (1) fix that crash; (2) extend the §8.3 hanging-value collapse to
  `assemble_eigenproblem_matrices` and to the eigenvector→node output path; (3) fix the
  `get_eigenfunction` bindings, which index a distributed `Vector<DoubleVector>` the way §8.4 describes.
* **Curvilinear macro-element mapping for simplices/wedges/pyramids** where curved boundaries meet
  refinement — see [macro_elements.md](macro_elements.md).
* **Coverage that was chosen, not blocked.** The refinement-criterion axis is swept distributed in 2D
  only. Everything in the campaign runs on purpose-built box meshes. No run above 4 ranks, and no mesh
  large enough for partitioning to produce genuinely thin halo layers.

---

## 12. Tests

| file | what |
|---|---|
| `test_triangle_refinement.py`, `test_tet_refinement.py` | the per-shape tree routes, C1/C2/TH/CR, uniform / 2:1 / multi-level / error-driven / unrefinement |
| `test_mixed_mesh.py` | 2D quad+tri, including the gmsh boundary-layer regression of §9.6 |
| `test_mixed_3d.py`, `test_adaptive_3d_campaign.py` | the 3D families and the 11-layout `MixedBoxMesh3D` campaign |
| `test_adaptive_2d_campaign.py` | nine equation systems × four discretisations × three refinement states |
| `test_constrained_adaptivity.py` | C1 field and position constraints composed with hanging (§3) |
| `test_boundary_element_identification.py`, `test_facet_adjacency.py` | §6 and the facet-adjacency primitive |
| `test_mpi_adaptivity.py` | the same campaign cases under `mpirun -n 2` and `-n 4` |

`tests/box_cases.py` and `tests/box_cases_3d.py` are the single definition of each campaign, imported by
the serial tests, the MPI harness's in-process reference *and* the `mpirun` worker — so the serial and
distributed campaigns cannot drift apart.

Three things the MPI harness had to get right:

* **Nested `mpirun` dies silently.** Importing pyoomph calls `MPI_Init`, so the pytest process is itself a
  singleton MPI job owning an Open MPI session directory under `TMPDIR`. A nested `mpirun` collides with
  it and exits 1 with *no stdout and no stderr at all*. Give the child its own `TMPDIR`. This cannot be
  dodged by import discipline — any other test module in the same pytest process has already imported
  pyoomph.
* **Compare only partition-independent quantities:** `max|residual|` (globally replicated since §8.4),
  global `ndof`, and integral observables (`Mesh::evaluate_integral_function` skips halo elements and
  `MPI_Allreduce`-sums). *Not* `nelement()` (per-rank, includes halos) or nodal values. The integral is
  what catches a **wrong field**, which a residual check alone would pass. Two independent oracles are
  applied: distributed-vs-serial, and cross-rank agreement.
* **Negative-control the oracle.** Perturbing the reference field integral by 1e-8 relative is caught, and
  so is a single-dof difference in `ndof`.

3D sizing: base 2×2×2 cells with uniform level 1 plus a level-2 or level-3 band, i.e. 1–6k elements,
1–6 s and well under 1 GB per case — which is also safe for a 2-rank MPI run. The 3D MPI matrices sweep a
representative five of the 11 layouts (every pure family plus `all_four`, which carries bricks, tets,
wedges, pyramids *and* the brick-to-tet transition cells in one mesh); `neumann` keeps all 11, since
boundary-facet propagation under refinement is the most shape-dependent part. The serial 3D campaign
sweeps all 11 exhaustively, so nothing is uncovered — the distributed campaign only has to show that
partitioning does not break what serial already proved.
