# Mixed adaptive meshes (branch `mixed_adapt`)

Status: **design + phased implementation, just started**. Author: development notes.

This document describes how pyoomph currently performs adaptive (hanging-node)
mesh refinement, why that machinery is restricted to pure quad (2D) / pure hex
(3D) / line (1D) meshes, and the plan to generalise it to **mixed adaptive
meshes**:

* 2D: pure triangular and mixed triangular + quad meshes,
* 3D: tetrahedral and mixed meshes of hexahedra, tetrahedra, wedges (prisms) and
  pyramids,

with, as an additional requirement, support for **multiple splitting schemes**
per element (e.g. splitting a quad into *two* quads in either direction instead
of four; analogous anisotropic / bisection splits for the other shapes).

It builds directly on the hanging-node work described in
[`hanging_nodes_redesign.md`](hanging_nodes_redesign.md), whose flattening-based
constraint machinery is now on `main`. All file/line references are to the state
of the tree at the time of writing.

---

## 1. Present state

| Case | Mechanism today | Status |
|---|---|---|
| 1D lines | `BinaryTree` hanging-node refinement | works |
| 2D pure quad | `QuadTree` (`DynamicQuadTreeForest`) hanging-node refinement | works |
| 3D pure hex | `OcTree` hanging-node refinement | works |
| 2D triangle / mixed | none — `refinement_possible()` false → full Gmsh **remesh** | no h-adaptivity |
| 3D tet / wedge / pyramid / mixed | none — `refinement_possible()` false (brick-only) → remesh only | no h-adaptivity |

The gate is `refinement_possible()`:

* 1D: always true (`src/mesh1d.hpp:86`).
* 2D: true only if **every** element casts to `oomph::QuadElementBase`
  (`src/mesh2d.cpp:201-221`); a triangle anywhere warns *"Found a tri or
  something in the mesh … cannot be adaptive right now. Requires to implement a
  good tree for mixed meshes"* and disables adaptation.
* 3D: true only if **every** element is a brick/hex (`src/mesh3d.cpp:43-63`).

When the gate is false, `setup_tree_forest()` calls `disable_adaptation()`
(`src/mesh2d.cpp:225-236`, `src/mesh3d.hpp:113`), and simplex/mixed meshes obtain
"adaptivity" only through full Gmsh **remeshing** + field interpolation
(`pyoomph/meshes/remesher.py`, driven from `Problem.force_remesh`,
`pyoomph/generic/problem.py:5977`).

### 1.1 Dormant scaffolding that already exists

* `RefineableTElement<1,2,3>` (`src/refineable_telements.{hpp,cpp}`): derive from
  `oomph::RefineableElement` + `oomph::TElementBase`. Only the **DIM=2**
  `setup_father_bounds` (`refineable_telements.cpp:234-308`) and a partial
  `build()` (`:416-1229`) are real; **all** hanging-node methods
  (`setup_hanging_nodes`, `setup_hang_for_value`, `quad_hang_helper`) are
  `throw_runtime_error("Implement")` in every dimension
  (`:181/190/201`, `:1247/1256/1267`, `:1435/1444/1455`). It piggy-backs a
  `QuadTree` for a 1→4 triangle split, reusing the SW/SE/NW/NE son names
  (`:266-267`). DIM=1 and DIM=3 are entirely stubbed. The partial DIM=2 `build()`
  also throws on a father macro element (`:602 "MACRO ELEM"`).
* `Refineable{Wedge,Pyramid}Element` (`src/wedges_and_pyramids.{hpp,cpp}`): satisfy
  the `RefineableElement` interface but every refinement hook throws
  (`wedges_and_pyramids.cpp:201-548`). `required_nsons()` throws `"TODO"`.
* Local (non-tree) triangle subdivision helpers `add_tri_C1` / `add_tri_C1TB`
  (`src/mesh2d.cpp:246-283`) exist for mesh generation, not tree refinement.

### 1.2 Known gaps in the mixed elements themselves

* Wedges/pyramids have **no macro-element / curvilinear** mapping — attaching a
  curved facet throws *"MacroElements not implement for this element type"*
  (`src/meshtemplate.cpp:~2561`). Tets are in the same `else` branch.
* Wedge/pyramid face `bulk_coordinate_derivatives` (`d s_bulk / d s_face`, needed
  for some FaceElement Jacobians) `throw("Implement")`
  (`src/wedges_and_pyramids.cpp:956-980, 1131-1155`); only the forward
  face→bulk coordinate maps work.

---

## 2. The decisive architectural fact

oomph-lib's refinement stack splits into two layers.

### 2.1 Shape-neutral primitives (fully reusable)

* `oomph::Tree` stores sons in an **arbitrary-length** `Vector<Tree*>`; `nsons()`
  is `Son_pt.size()`, `son_type()` is a plain `int`, and the son-name enums are
  "simply aliases for ints" (`tree.h:99-132`).
* The split driver `Tree::split_if_required<ELEMENT>` reads
  `n_sons = new_elements_pt.size()` — **not** a hard-coded `2^dim` — and builds
  each son via the pure-virtual `construct_son(object, father, son_type=i)`
  (`tree.template.cc:100-120`).
* `oomph::HangInfo` is a generic *(master nodes, weights)* list
  (`nodes.h:741`); nothing about it is shape-specific.
* `complete_hanging_nodes_recursively` (`refineable_mesh.cc:2214`) folds
  hanging-on-hanging chains into genuine (non-hanging) masters, merges duplicate
  masters, drops tiny weights, and checks weights sum to 1. This is how 3D
  edge-hangs already work even though `oc_hang_helper` only handles faces.
* pyoomph's own `DynamicTree::dynamic_split_if_required` (`src/mesh.cpp:4710`)
  already delegates son creation to the element
  (`beb->dynamic_split(new_elements_pt)`), sizes `Son_pt` to the returned count,
  and constructs one tree son per returned element — i.e. **variable son count
  and heterogeneous son types are already structurally supported.**

### 2.2 Hard-wired concrete layer (not reusable as-is)

* Son numbering and son→sub-box maps assume tensor-product bisection
  (`refineable_quad_element.cc:711-740`; `OcTree::Direction_to_vector`).
* `required_nsons()` returns literal `4` / `8`
  (`refineable_quad_element.h:106`, `refineable_brick_element.h:102`).
* Neighbour finding is compass-based: the `QuadTreeNames` / `OcTreeNames` enums
  and the static rotation/reflection tables (`quadtree.h:255-292`,
  `octree.h:47-77`) presume 4 edges / 6 faces + 12 edges of a box.
* `quad_hang_helper` / `oc_hang_helper` enumerate a fixed 4 edges / 6 faces and
  use hard-coded `n_p`-based node-index formulas for a `[-1,1]^dim`
  tensor-product Lagrange layout (`refineable_quad_element.cc:1522-1828`,
  `refineable_brick_element.cc:2142-2508`).
* `TreeRoot::Neighbour_pt` is a `map<int,TreeRoot*>` keyed by compass direction
  (`tree.h:323-398`).

**Both** new requirements — mixed simplex/pyramid shapes **and** variable /
anisotropic split schemes — break exactly this concrete layer, in the same way,
while leaving §2.1 intact.

---

## 3. Chosen architecture: one generic refinement engine

Rather than writing N shape-specific tree subclasses (`TriTree`, `TetTree`,
`WedgeTree`, …) that imitate `QuadTree`'s compass model — which would hard-code
one split scheme per shape and reproduce the rigidity we are trying to escape —
we build a **single pyoomph-native generic refinement engine** on top of the
shape-neutral primitives of §2.1. Three abstractions carry it.

### 3.1 `RefinementPattern` descriptor

Keyed by *(element type × scheme)*. Declares, for one parent element:

* the **child count** and each child's **element type** (heterogeneous allowed —
  a pyramid red-splits into pyramids + tets; a quad "1→2" split yields 2 quads;
  a wedge into wedges);
* each child's **affine embedding** into the parent reference element
  (child-ref → parent-ref), used to place/create son nodes and to compose
  coordinate maps for hanging;
* each child **facet → parent facet (or interior)** map, used for boundary-facet
  propagation (§4.3) and neighbour matching.

This replaces `required_nsons()` + `create_son_instance()` inside
`BulkElementBase::dynamic_split` (`src/elements.cpp:6632`) and the per-shape
`Father_bound` tables. A pattern is a static, per-element-type table (plus, for
anisotropic schemes, a selected variant); the element exposes *which* patterns it
supports and the engine picks one (default isotropic; user-directed or
error-driven otherwise, §5).

### 3.2 Facet-based neighbour finding

Replace the compass-keyed `TreeRoot::Neighbour_pt` with adjacency keyed by the
**shared vertex-node set of a facet**. This is shape-, scheme-, and
dimension-neutral, and it already exists in pyoomph as the `facets` map
(`std::map<std::set<pyoomph::Node*>, std::vector<unsigned>>`, `src/mesh.hpp:351`,
built by `setup_facets_from_template`, `src/mesh.cpp:5492`). The same key type
handles a quad face abutting two triangle faces, or a 1→2 split abutting a 1→4
split, uniformly. Neighbour relations are looked up by the current facet
vertex-node set at each tree level, so they follow refinement instead of a
static compass table.

### 3.3 Geometric (not index-formula) hang helper

For a facet shared with a **coarser** neighbour, for each fine interpolating node
on the facet:

1. map the node's position into the neighbour's reference coordinates (compose
   the child→parent embeddings of §3.1 up the neighbour's ancestry, or locate it
   geometrically);
2. evaluate the neighbour's shape functions there;
3. `HangInfo::set_master_node_pt(master, weight)` for each neighbour node with the
   shape-function value as weight;
4. install via `Node::set_hanging_pt(hang, value_id)`.

Because it works from geometry + shape functions, it is agnostic to how *either*
side was split (isotropic, anisotropic, different shape). Then hand off to
oomph's `complete_hanging_nodes_recursively` for chain resolution. This is the
same idea as `quad_hang_helper` minus the hard-coded compass/index assumptions,
and it composes with the flattening-based C1-constraint machinery already on
`main` (see `hanging_nodes_redesign.md` §5.5.2): flattening reads
`hanging_pt(v)` masters recursively, so genuine simplex/mixed hangs feed it
unchanged.

A generalised **balance / closure rule** (the analogue of 2:1) bounds the
hanging depth across a facet. Multi-level hangs are supported by
`complete_hanging_nodes_recursively`, but cost conditioning, so the default is to
enforce a one-level (2:1-style) balance by propagating refinement to neighbours.

---

## 4. How the four known concerns map onto this design

### 4.1 Hanging nodes assigned correctly
Handled by §3.3 + `complete_hanging_nodes_recursively`. Truth stays in
`Node::Hanging_pt` / `Local_hang_eqn`, so error estimation, output, projection,
assembly (incl. the flattening constraint path) and MPI sync all keep reading the
same source (the load-bearing decision of the hanging redesign).

### 4.2 MPI
Keeping truth in `Hanging_pt` means oomph's `synchronise_hanging_nodes` +
pyoomph's `additional_synchronise_hanging_nodes` (`src/mesh.cpp:4769`) continue
to reconcile halo masters. Facet-based neighbour finding (§3.2) must be made
halo-aware, but **no new hang representation is invented** — the explicit warning
of the hanging redesign (§5.7).

### 4.3 Facet boundary marking under refinement (the "third edge" problem)
Two boundary mechanisms coexist: per-node membership (oomph, propagated to
mid-edge nodes during `build()`) and pyoomph's per-facet `facets` map
(`src/mesh.hpp:351`). The facet map fixes the false positive where a first-order
triangle with two edges on the same boundary also marks its third (interior)
edge (`src/mesh2d.cpp:604-634`), by requiring an **exact facet vertex-node-set
match** (`src/mesh.cpp:5429-5473`). But the map is a **one-shot snapshot** of the
coarse template and is therefore *disabled* whenever adaptation is on
(`src/mesh2d.cpp:303-315`, `src/mesh3d.cpp:86-98`), falling back to the node-based
reconstruction — which reintroduces the third-edge false positive.

**Fix enabled by the engine:** make the `facets` map **dynamic**. On split,
propagate each parent boundary facet to its child facets (via §3.1's
child-facet→parent-facet map, inserting the new mid/inner nodes into the key); on
unrefine, remove the child facets and restore the parent. Exact facet identity is
then preserved through refinement, eliminating *both* the third-edge false
positive *and* the stale-snapshot problem, for every shape — so the facet scheme
no longer needs disabling under adaptation.

### 4.4 Mixed offspring (refinement not shape-closed)
Already structurally supported (§2.1): `dynamic_split_if_required` builds one
tree son per returned element irrespective of type. `RefinementPattern` emits the
mixed child types (pyramid → pyramids + tets being the driving case);
`construct_son` reads each child's type. This is why pyramids are scheduled last —
they exercise the heterogeneous-son path.

---

## 5. The multiple-splitting-scheme requirement

`dynamic_split` (`src/elements.cpp:6632`) currently hard-codes
`n_sons = required_nsons()` and `create_son_instance()` (same type). The
`RefinementPattern` abstraction (§3.1) replaces both, so variable son count and
heterogeneous types drop out for free at the tree layer (§2.1). What the
*anisotropic* case additionally needs:

* **Scheme selection.** A parent may support several patterns (quad → 4; quad →
  2 in x; quad → 2 in y; simplex bisection; …). Initially: default isotropic,
  with a user-directed override. Then a **directional error estimator** —
  extending `LagrZ2ErrorEstimator` (`src/lagr_error_estimator.cpp`) to yield a
  per-direction error, or a recovered-Hessian anisotropy indicator — to *choose*
  the pattern automatically.
* **Generalised balance rule.** The 2:1 closure must account for direction: an
  x-split neighbouring a y-split, or a 1→2 neighbouring a 1→4, produces hanging
  configurations the uniform rule does not describe. The geometric hang helper
  (§3.3) already handles the *assembly* of such hangs; the balance rule governs
  *how far* to let them go before forcing neighbour refinement.
* **Unrefinement.** `rebuild_from_sons` (`src/elements.hpp:690`) must know which
  pattern produced the sons to invert it; the pattern id is stored on the tree
  node / element.

Because §3.2–§3.3 are geometric and facet-based, they are already
scheme-agnostic; anisotropy is mostly an *authoring* (patterns) + *selection*
(error) + *balance* concern, not a rewrite of the hang machinery.

---

## 6. Implementation phases

1. **Generic engine skeleton — no behaviour change. [DONE]** Introduce
   `RefinementPattern` + a facet-adjacency builder; re-express the existing
   quad/hex/line refinement through them. Validate at parity against
   `tests/test_adaptivity.py` and `tests/test_constrained_adaptivity.py`. De-risks
   the abstraction on the known-good path before any new shape.
   * `src/refinement_pattern.hpp`: `RefinementPattern` +
     `IsotropicSameTypeRefinementPattern`; `BulkElementBase::dynamic_split` routes
     through `refinement_pattern()` (default reproduces `required_nsons()` +
     `create_son_instance()` exactly). No behaviour change (10/10 adaptivity tests,
     quadtree neighbour error 0).
   * `TemplatedMeshBase::build_facet_adjacency()` + `facet_adjacency_summary()`
     (bound on all three dims): the shape-neutral neighbour primitive, validated
     manifold on quad/tri/brick meshes by `tests/test_facet_adjacency.py`.
   * *Not* done here (deliberately): oomph's quad/hex neighbour finding is left in
     place — the facet-adjacency builder is added alongside it as scaffolding for
     Phase 2, not a rip-and-replace (that would be high-risk on the known-good
     path). `rebuild_from_sons` generalisation deferred to Phase 7 (the method is
     pervasively `QuadTreeNames`-specific below its DG loop).
2. **2D triangles, isotropic 1→4.** First new shape; reuse `RefineableTElement<2>`
   `Father_bound` / `build`. Establish the generic hang helper, the dynamic facet
   map (§4.3), and the closure rule. Oracle: linear-residual → machine zero on an
   adapted triangular mesh (as in the hanging redesign tests).
   * **Phase 2a [DONE] — uniform (conforming) triangle refinement.** `refinement_possible()`
     now accepts pure-triangle meshes; a linear (C1) triangle refines 1→4 reusing the
     QuadTree hierarchy, but node-sharing is **geometric**, not via oomph's compass
     descent: a new son node on a father edge is keyed by the pair of father corner
     nodes it bisects (`RefineableTElement<2>::Shared_edge_node_registry`), a key
     identical from every element touching the edge (sibling sons + the edge-sharing
     neighbour father's sons). This confirmed the compass descent is unusable for
     triangles (the NW son touches father edges E+W not N+W; the centre son shares no
     full father edge). Build helpers (`get_boundaries`/`get_bcs`/`get_edge_bcs`/
     `interpolated_zeta_on_edge`) implemented; `find_neighbours` sets triangle root
     neighbours by shared edges; `check_all_neighbours` skips the quad self-test for
     triangle forests. Validated: 8→32→128 elements, conforming (facet-adjacency
     manifold), Poisson solve + `IntegralObservables` converge; `tests/test_triangle_refinement.py`.
   * **Phase 2b [DONE] — non-conforming (hanging-node) triangle refinement.** After
     oomph's `adapt()` returns, `TemplatedMeshBase::adapt` calls the new virtual
     `post_adapt_setup_hanging_nodes()` (overridden in `TemplatedMeshBase2d`). A hanging
     node lies strictly in the interior of a coarser neighbour's edge `{P,Q}` — that edge
     appears in the facet adjacency as an interior facet incident on exactly one element.
     Each node collinear strictly between `P` and `Q` is constrained to the linear
     interpolation of `P,Q` (`HangInfo` weights `(1-t), t`). No coordinate descent, no 2:1
     assumption; hanging masters that are themselves hanging are resolved by the
     assembly-time flattening from the hanging redesign. Runs before equation numbering.
     **Validated** with the linear-residual oracle (residual → machine zero): non-uniform
     `RefineToLevel` on a boundary (3 splits × 2 boundaries, ~1e-16) and genuine Z2
     error-driven adaptive refinement (~6.6e-16).
   * **Phase 2c [DONE] — quadratic (C2) triangles.** Node-sharing generalised: every node a
     1→4 refinement creates is the midpoint of exactly two father nodes (two corners for C1;
     a corner+mid-edge or two mid-edge nodes for C2), so `father_edge_node_key` keys on that
     father node-pointer pair (disambiguates the two edge quarter-points + interior nodes a
     C2 father edge spawns; reduces to the C1 corner-pair rule). Hanging generalised:
     `post_adapt_setup_hanging_nodes` reads the coarse element's edge order — a C2 coarse edge
     interpolates quadratically through `{P,M,Q}` (M = mid-node at t=0.5, real/shared), so its
     interior fine nodes hang on `{P,M,Q}` with quadratic Lagrange weights; C1 edges keep the
     2-master linear weights. **Validated** (C1+C2, machine-zero oracle): uniform, non-uniform,
     Z2-adaptive, and a deep level-1→4 jump (hanging-on-hanging via assembly-time flattening).
     Full suite 33/33.
   * **Phase 2d [DONE] — mixed continuous spaces on one triangle mesh (Taylor-Hood).** A C1(TB)
     field living on a C2-coordinate mesh (e.g. the Taylor-Hood pressure: C2 velocity + C1
     pressure) owns a **separate value-hang slot** — `continuous_spaces[SPACE_INDEX_C1(TB)].hangindex`
     is `>= 0` (codegen sets it to `numfields_basebulk[C2TB]+numfields_basebulk[C2]`), distinct from
     the geometric slot `-1` the C2 fields hang on. The C2 hang (quadratic on `{P,M,Q}`, slot `-1`)
     aliases *every* value slot onto slot 0, which is **wrong for the linear pressure**. Fix in
     `post_adapt_setup_hanging_nodes`: after the geometric hang, for each such separate slot install
     an extra **linear** hang on the coarse edge corners `{P,Q}` (weights `1-t, t`) for every interior
     node carrying that dof — including the coarse edge **mid-node M**, whose velocity is a real dof
     (not hanging) but whose pressure the coarse element does not carry and so must hang. Nodes without
     the dof (C2-only edge midpoints, `nvalue() <= slot`) are skipped. Ordering (velocity `-1` first,
     then the C1 slot) is double-free-safe: `Node::set_hanging_pt` only deletes a value slot when it
     differs from the geometric pointer, and the aliased slot equals it. **Validated** (machine-zero
     residual oracle) — lid-driven cavity Stokes, non-uniform refinement, splits left/right/crossed
     (`test_taylor_hood_triangle_cavity_residual_oracle`). Without the fix the (linear) Stokes Newton
     step does not converge (residual ~0.68, stalling), with it → ~3e-15.
3. **2D mixed quad + tri**, incl. quad-face/two-tri-face hangs and boundary
   facets across shape changes.
   * **[INVESTIGATED — reverted, not landed]** The hard case: quads share nodes /
     hang via oomph-lib's box (QuadTree) machinery, triangles via the geometric
     registry + post-adapt pass, and the two clash at a quad-tri interface. Findings:
     (a) the geometric hang pass **reproduces oomph's quad hanging to machine
     precision** on pure-quad meshes (so hanging *can* be unified); (b) a full
     geometric quad *build* is blocked — oomph's quad build handles curved-boundary
     **macro elements** (e.g. `CircularMesh`) that the geometric build does not; (c) at
     the interface, quad and triangle each create a *coincident duplicate* mid-edge
     node. Pruning the duplicate corrupts oomph's tree-based adapt (it retains nodes
     for unrefinement → `reorder_nodes` crash), so instead the duplicates were **tied**
     (a weight-1 hang slaving one to a representative). This gave a correct, conforming
     mesh for **uniform** refinement (converges to the analytic solution) and for a few
     adapt cycles, **but** robust multi-cycle error-driven adaptivity fails: on ~the 3rd
     cycle the geometric quad hangs from the previous cycle collide with oomph's own
     quad adaptation (inconsistent Jacobian → slow Newton; earlier variants crashed in
     `reorder_nodes`). Clearing hangs before oomph's adapt and various tie/representative
     rules did not fully resolve it. **Conclusion:** mixed needs a genuine reconciliation
     of the two adaptation mechanisms across cycles (or a geometric quad build that also
     supports macro elements), not a post-hoc weld. Reverted to keep pure-shape
     adaptivity solid; the approach and dead-ends are recorded here for the next attempt.
4. **3D tets → wedges → pyramids** (pyramid last: forces mixed offspring). Fill
   the wedge/pyramid `bulk_coordinate_derivatives` gaps (§1.2) as needed for face
   Jacobians.
   * **Phase 4 (tets) [DONE for C1]** — pure-tet meshes now h-refine, the 2D triangle
     recipe lifted a dimension. A linear tet refines 1→8 on the OcTree hierarchy (8 sons =
     4 corner sub-tets + 4 tiling the interior octahedron along a fixed diagonal — a free
     choice, since the octahedron never touches a shared face so conformity holds via
     edge-midpoint sharing). `RefineableTElement<3>::build` was written from scratch
     (barycentric son→father map; registry keyed by the bisected father-node pair, 3D
     `father_edge_node_key`; new-node boundary/BC/coords derived directly from the two
     father nodes, so no `Father_bound` table). Forest: `octree.h/.cc` got the 2D quadtree
     treatment (skip-init ctor flag + virtual `find_neighbours`); new `DynamicOcTreeForest`
     skips brick neighbouring/self-test for tets, delegates for bricks. Hanging
     (`TemplatedMeshBase3d::post_adapt_setup_hanging_nodes`): edge-interior nodes hang on the
     coarse edge (linear C1 / quadratic C2), and for >1-level jumps face-interior nodes hang
     barycentrically on the coarse C1 face. The face-interior pass runs BEFORE the edge pass
     and the edge pass skips edges with a hanging endpoint, so a face-interior node binds to
     its coarse face's real corners rather than a fine sub-edge with a hanging endpoint.
     Two further passes make **arbitrary** refinement machine-zero: (i) **2:1 balancing**
     (`enforce_refinement_balance`, run in `adapt` before the hang pass) iteratively refines any
     leaf tet with a node at `t = 1/2^nnode_1d` of an edge — meaning a neighbour is ≥2 levels
     finer — via `refine_selected_elements`, to a fixed point. The fraction is order-aware (C1:
     1/4, C2: 1/8), since a 1-level neighbour already puts C2 sub-edge mid-nodes at 1/4 — a naive
     quarter-point test would false-positive every refined C2 tet and explode the mesh. (ii)
     **Hang-chain flattening**: a hanging node whose master is itself hanging is re-expressed over
     real (non-hanging) leaf nodes, so the assembled `HangInfo` has only real masters and the
     Jacobian is exact (assembly-time flattening left ~1e-9 for tets). Balancing keeps the mesh
     2:1 so the unhandled C2 face-interior case does not arise.
     **Validated** (machine-zero oracle) for **C1 and C2** tets: uniform (6→48→384, manifold),
     single-level 2:1, Z2 error-driven adaptivity, **and abrupt >1-level `RefineToLevel` jumps**
     (`tests/test_tet_refinement.py`, parametrised over C1/C2; 13/13). C2 tet face-interior
     hanging is still not handled directly (balancing avoids it). Wedges/pyramids: TODO.
   * **[DONE] — 3D Taylor-Hood (mixed C2 velocity / C1 pressure tets).** Same separate-C1-slot fix as
     2D Phase 2d, in `TemplatedMeshBase3d::post_adapt_setup_hanging_nodes`: after the geometric hang,
     for each separate C1(TB) slot install a linear hang on the coarse edge corners `{P,Q}` for every
     interior node carrying that dof (including the coarse mid-node). The edge pass skips edges with a
     hanging endpoint, so the C1 masters `{P,Q}` are always real -> no C1 flattening needed. Single-
     level 2:1 tets produce only edge-interior hangs (face-interior only at >1-level jumps), so the
     edge pass suffices. **Validated** (machine-zero) — cavity Stokes, uniform / single-level 2:1 /
     Z2 error-driven (`test_taylor_hood_tet_cavity_*`); residual ~1e-16 (was ~1e-2, non-convergent).
   * **[DONE] — Crouzeix-Raviart (C2TB bubble velocity / DL discontinuous pressure) triangles.**
     Refinement of bubble-enriched elements: (1) `BulkElementTri2dC2::create_son_instance` now returns
     a genuine `BulkElementTri2dC2TB` for C2TB fathers (the old `BulkElementTri2dC2(true)` bumped
     `nnode_of_space[C2TB]` to 7 while keeping 6 oomph node slots + the 6-node nodal-space map, so
     `fill_element_info` ran off the map and segfaulted); (2) `RefineableTElement<2>::build` +
     `setup_father_bounds` handle the 7th (centroid) node -- interior, never on a boundary/shared/
     hanging; the central son reuses the father's centroid node, the other three get fresh ones. The
     DL pressure is element-internal (allocated per son, never hangs); the C2TB velocity hangs
     quadratically on edges exactly like C2 (bubble vanishes on edges). **Validated** (machine-zero,
     looser ~1e-7 tol for CR's poorer conditioning) — cavity Stokes, uniform / single-level 2:1 /
     Z2 error-driven (`test_crouzeix_raviart_triangle_*`). ~~KNOWN GAP: abrupt >1-level jumps on very
     coarse CR meshes (e.g. N=2, level 1→3) diverge~~ **[RESOLVED — this was a Pardiso artifact, see
     §4.1: with SuperLU these deep jumps converge to machine zero (1.4e-13); the hanging was correct
     all along.]**
   * **[DONE] — 3D Crouzeix-Raviart (C2TB bubble velocity / DL pressure tets).** The 3D C2TB tet has
     15 nodes: 10 C2 (4 corners + 6 edge-mids) + **4 face-centroid bubbles + 1 volume-centroid
     bubble**. Fixes: (1) `BulkElementTetra3dC2::create_son_instance` returns a real
     `BulkElementTetra3dC2TB` for C2TB fathers (else `local_coordinate_of_node(10..)` throws --
     15-node map vs 10 oomph slots). (2) `RefineableTElement<3>::build` shares the 4 face bubbles
     across face-adjacent tets: a face bubble (son nodes 10-13) is keyed on its **three son face-
     corner nodes** (already shared coarse corners / edge-mids), so both tets meeting on a face build
     one node -> continuous enriched velocity; the volume bubble (node 14) is interior/never shared;
     boundary/pin data generalised from the 2-node edge case to N generating nodes. (3) UNLIKE
     C2/TH, C2TB produces face-INTERIOR fine nodes (sub-face bubbles + inner-edge-mids) already at a
     single-level 2:1 interface, so `post_adapt_setup_hanging_nodes`'s face pass now hangs them on
     the coarse face's **7 nodes** (3 corners + 3 edge-mids + face-bubble) via the enriched triangle
     shape -- which is exactly the trace of the 3D enriched tet shape on a face (verified: corner
     `(2L-1)L+3·L0L1L2`, mid `4LiLj-12·L0L1L2`, bubble `27·L0L1L2`). The coarse face's own 7 nodes
     are excluded as slaves (the face-bubble would otherwise hang on itself -> infinite flatten
     recursion); the coincident fine central sub-face bubble is a different pointer and stays
     constrained. **Validated** for uniform (machine-zero), single-level 2:1 and Z2 error-driven
     (`test_crouzeix_raviart_tet_cavity_*`). NOTE: 3D CR is inherently 2-Newton-step and poorly
     conditioned (residual floor ~1e-11 even unrefined), so the oracle uses the FINAL residual with a
     ~1e-7 tolerance, not the 1-step value. **[Superseded — see §4.1: this "poor conditioning" is a
     Pardiso artifact, not intrinsic; with SuperLU it is 1-step machine-zero.]**

   * **§4.1 — Linear-solver sensitivity of the bubble/DL-pressure Stokes systems (IMPORTANT).**
     The "CR is inherently 2-Newton-step / poorly conditioned (~1e-11 floor)" and "2D CR abrupt
     >1-level jumps diverge" caveats above are **NOT** properties of the elements or the hanging
     scheme -- they are artifacts of the **default linear solver, Pardiso**, on these Stokes
     saddle-point + hanging-constraint matrices (bubble-enriched velocity, discontinuous DL
     pressure). Switching to **SuperLU** (`problem.set_linear_solver("superlu")`, SciPy's serial
     direct solver) makes **every** CR/TH case -- 2D and 3D, uniform / single-level 2:1 / Z2
     error-driven, AND the previously "diverging" 2D deep jumps (N=2, level 1→3 and 1→4) --
     converge in a **single Newton step to true machine zero** (~1e-13 residual). Head-to-head on
     the 3D CR 2:1 cavity: Pardiso 1.9e-10 (2 steps), SuperLU 1.2e-12 (1 step).

     Consequences: (a) the exact-Jacobian residual oracle actually **passes at 1 step** for all the
     bubble/DL cases once the linear solve is exact -- i.e. the enriched C2TB face-hanging (and all
     the other hangs) are provably **correct**, not merely approximately right; (b) the "known gap"
     for 2D CR deep jumps is a solver issue, not a hanging bug -- SuperLU handles them fine;
     (c) Pardiso is losing accuracy / diverging on these indefinite saddle-point matrices (suspect
     matrix-type / pivoting handling for the DL-pressure + hanging block structure) -- a solver-side
     item, tracked separately from this branch. The committed tests still run on the default (Pardiso)
     with looser ~1e-7 tolerances so they pass on the shipped default; a SuperLU + tight-1-step
     variant would be a strictly stronger oracle (not yet added).

   * **§4.2 — Unrefinement (coarsening) of mixed-space (Taylor-Hood / Crouzeix-Raviart) meshes
     [DONE].** Pure C1/C2 Poisson coarsening already worked; the mixed-space Stokes cases failed
     (slow-diverging Newton / hit the adaptive Newton iteration cap) specifically when Z2 adaptivity
     *unrefined* into a non-conforming state. `rebuild_from_sons` field reconstruction was fine
     (conforming coarsening was machine-zero); the bug was in the C1-pressure hanging on the coarsened
     mesh, and had two parts, both fixed in `TemplatedMeshBase2d::post_adapt_setup_hanging_nodes`:
     (1) **Stale-pressure edge-mids.** A node that was a fine corner (so carries a C1 pressure value
     slot) can, once its region is coarsened, become a plain C2 edge-mid of a *conforming* coarse
     element. oomph does not shrink the node, so its pressure slot survives but no element assembles
     it -> an unconstrained free dof. Fix: any C1-carrying node that is not a vertex of any element is
     hung linearly (0.5,0.5) on the two endpoints of the coarse element edge it bisects (removes the
     free dof and is the physically correct value).
     (2) **Unflattened C1-pressure chains.** Scattered (error-driven) refinement makes hanging nodes
     whose masters are themselves hanging. The 2D pass had relied on oomph's *assembly-time* flattening
     -- which resolves the geometric (velocity) slot but NOT the separate C1 pressure value slot, so
     the pressure hang was expressed over hanging masters -> wrong Jacobian -> multi-step Newton. Fix:
     explicit two-phase chain flattening (as the 3D tet pass does) now run for EVERY slot (geometric -1
     and each C1 slot), resolving all masters to real leaf nodes. (The 3D edge pass already skips
     hanging-endpoint edges, so its C1 masters are always real -> 3D TH needed no change.)
     Diagnosis was decisive via the *reset-and-resolve* oracle: zero all dofs on the final adapted
     mesh and solve -- a linear problem reaches machine zero in ONE Newton step iff the Jacobian is
     exact. Before: multi-step; after: 1 step (~1e-13 SuperLU, ~1e-15 Pardiso). The nodal
     pressure-fixation "failure" was only a symptom (the wrong Jacobian's slow convergence tripped the
     Newton iteration cap); with the exact Jacobian it converges in one step. **Validated** for 2D TH,
     2D CR and 3D TH unrefinement (`test_stokes_triangle_unrefinement_residual_oracle`,
     `test_taylor_hood_tet_unrefinement_residual_oracle`). Works on BOTH Pardiso and SuperLU -- once
     the Jacobian is exact, Pardiso solves these fine too (cf. §4.1's Pardiso concerns were about
     accuracy on a *wrong*/ill-conditioned system, not these now-exact ones).

   * **§4.3 — `ConstrainFieldsToC1Space` × unrefinement [DONE].** The explicit "degrade a C2 field to
     C1" feature (`set_additional_dof_constraint`, distinct from the C1-field hanging) also broke under
     error-driven coarsening -- but only on simplex meshes; quads were fine. Symptom: correct solution
     but a **wrong Jacobian** (FD-confirmed wrong redistribution weights on the constrained field's
     rows) -> geometric (slow) Newton, hitting the adaptive-Newton cap. Root cause: the mechanism
     **pinned** each constrained non-vertex value and expanded it via `c1_constraint_corners` only in
     the assembly flatten. A pinned dof is not processed by oomph's hanging machinery, so its C1-corner
     masters are **never registered** -- and when a constrained mid-node is itself a master of a finer
     node's hang, the flatten drops it (pinned leaf) instead of resolving it through to the real coarse
     corners. Quads never hit this because `RefineableQElement::setup_hang_for_value` gives every value
     a genuine registered hang; simplex `setup_hang_for_value` is a no-op (mesh-level geometric pass
     instead), and that pass doesn't touch the *constrained* value slot. Fix (in
     `BulkElementBase::setup_additional_dof_constraints`): give each constrained non-vertex value a
     **genuine linear value-slot `HangInfo`** on the element's C1 corner nodes instead of pinning.
     oomph then registers those masters and resolves any master that is itself hanging via
     `complete_hanging_nodes` -- exactly the quad behaviour, and MPI-consistent. Order-independent: a
     node that is a C1 vertex of a finer neighbour (a 2:1 coarse mid) is simply skipped there and gets
     its (identical) hang from the element(s) where it is a non-vertex; it is **never pinned** (a pin
     from the vertex side would clobber the hang, order-dependently -- bad under MPI). Constraint
     markers are cleared+reapplied each assign, so every live constrained node is a non-vertex of some
     element and always receives its hang. **Validated** (reset-and-resolve + FD-Jacobian oracle) for
     2D/3D C2-Poisson-constrained-to-C1 under refine+unrefine, on Pardiso and SuperLU (1-step machine
     zero); quad `ConstrainFieldsToC1Space` unchanged; full suite green
     (`test_constrained_field_unrefinement`).

   * **§4.4 — MPI distributed adaptivity (`--petsc_mumps --distribute`) [2D DONE; 3D tet BLOCKED].**
     Validated distributed (real METIS-partitioned meshes, MUMPS since Pardiso is not MPI-capable)
     via the reset/resolve residual oracle on `get_last_residual_convergence()` (NOT `get_residuals()`,
     which returns garbage for non-owned dofs under MPI). **2D machine-zero, all cases:** triangle
     Poisson C1 (3-rank final_res 6.9e-16, integral 0.4425747) and C2 (1.2e-15); Stokes TH
     (final_res 1.3e-15, ke matches serial) and CR (1.5e-15). **3D tets: serial machine-zero, but
     distributed refinement fails** — root-caused to two distinct bugs in oomph-lib's distributed
     halo synchronisation, exposed because simplex meshes carry `ntstorage()>1` position storage with
     only t=0/1 initialised (t>=2 left at zero), which 2D happens to tolerate and 3D does not:
     - **Bug A [FIXED].** `TreeBasedRefineableMeshBase::synchronise_nonhanging_nodes`
       (`refineable_mesh.cc`) compared the node's **current** position `nod_pt->x(dir)` against
       `x_exp = get_x(t, s)` interpolated at **history level t**. With history t>=2 = 0, every
       non-vertex node was falsely flagged as "position differs" (dmax ~0.5, not roundoff), emitting
       ~250 spurious halo records per rank. For 3D tets the resulting send/recv unsigned stream
       desynced by one entry -> one rank throws `recv_unsigneds_count != recv_unsigneds_index`
       (`refineable_mesh.cc:3777`) while its peer blocks in `MPI_Recv` -> **deadlock**. 2D triangles
       hit the same spurious flagging but did not desync, so they "worked". Fix: read `nod_pt->x(t, dir)`
       so the comparison is like-for-like at each history level; conforming nodes are then never
       flagged and nothing is sent (verified: send count 0). This is a genuine oomph-lib bug; the
       one-line fix is regression-clean (2D Poisson C1/C2 + Stokes TH/CR distributed still machine-zero,
       serial unchanged since the function is MPI-only).
     - **Bug B [OPEN, blocks 3D tet distributed — ROOT CAUSE IDENTIFIED].** With Bug A fixed a deeper,
       fundamental fault surfaces. Traced end-to-end (an `LD_PRELOAD` PMPI `MPI_Send` shim with
       `backtrace()` pinned the sender; `MPI_Iprobe`/checkpoint instrumentation the rest, since the
       sandbox blocks live gdb/ptrace):
       * **Symptom chain.** A stray tag-0 message is left unmatched during refinement and later grabbed
         by an unrelated tag-0 recv: `synchronise_nonhanging_nodes` misreads it as a count -> resizes to
         garbage -> **segfault**; if that path is retagged/fixed, `synchronise_hanging_nodes` chokes ->
         **deadlock**. The sender is **`Mesh::resize_halo_nodes()`** (called from within base
         `Mesh::classify_halo_and_haloed_nodes`), whose sender guards data transmission on the local
         **haloed** count while the receiver guards on the local **halo** count.
       * **Root cause (the real bug).** After 3D-tet refinement the halo/haloed **node** classification
         is **asymmetric across ranks**: measured directly, `rank0.nhaloed_node(1)=249` but
         `rank1.nhalo_node(0)=0` (while the other direction `rank1.nhaloed_node(0)=480 == rank0.nhalo_node(1)=480`
         is fine). The invariant `rankA.nhaloed_node(B) == rankB.nhalo_node(A)` — which
         `resize_halo_nodes`, `synchronise_hanging_nodes` and `additional_synchronise_hanging_nodes`
         all assume (they only *check* it under `#ifdef PARANOID`, compiled out in the opt build) — is
         **violated**. Mechanism: a node newly created on a shared boundary during refinement is not
         registered symmetrically in the distributed shared-node scheme, so **both** ranks end up
         believing they own it (`processor_in_charge` disagreement) and **both** put it in their
         *haloed* list, **neither** in the *halo* list. oomph's quad/hex distributed refinement keeps
         this consistent via the Quad/OcTree neighbour structure; pyoomph's tet refinement shares new
         nodes **geometrically** (bypassing that structure, §3.2/§3.3) and never reconciles the new
         nodes into `Shared_node_pt` / the halo classification.
       * **Why symptom-patching is NOT a fix (tested, then reverted).** Making the three exchanges drain
         cleanly under asymmetry (always exchange the header/count; for asymmetric pairs set
         `halo_hanging=haloed_hanging` so no spurious hang discrepancy corrupts pyoomph's geometric
         hanging) made **C1 uniform + error-based distributed run to machine-zero** and **C2 survive 2
         adapt cycles** — but it is papering over a **mis-classified mesh**: the observable
         `∫u` comes out **~2x** the serial value for distributed 3D tet C1 (`0.083` serial vs `0.166`
         2-rank — halo elements double-counted), whereas 2D distributed matches serial **exactly**
         (`0.4425747`). So the residual oracle passes (self-consistent solve) but on a **doubled/wrong
         mesh**; C2 deep refinement (level 3) still diverges to `inf`/NaN. Conclusion: the fix must be
         at the **root** — register newly-created shared tet nodes into the distributed shared-node /
         halo scheme so `classify_halo_and_haloed_nodes` produces **symmetric** halo/haloed node lists.
         Relevant machinery: `Mesh::synchronise_shared_nodes`, `additional_setup_shared_node_scheme`
         (`problem.cc`), and the halo/haloed node population in `Mesh::classify_halo_and_haloed_nodes`.
     - **Net:** 2D distributed adaptivity (all element types) is production-ready and correct
       (integrals match serial). 3D tet distributed adaptivity is blocked on Bug B — a genuine
       distributed-tet-mesh-construction defect (asymmetric halo/haloed node classification), not a
       messaging bug; symptom-patching yields silently wrong meshes and is intentionally NOT committed.
       Serial 3D tet adaptivity is unaffected and remains machine-zero.

   * **§4.5 — Complex tri-adaptive problems (curved / moving / multi-domain) [OPEN, serial].**
     Reported crash: `pinned_water_droplet_tri_adapt.py` (multi-domain droplet-on-substrate,
     `mesh_mode="tris"`, moving mesh, curved surface, `spatial_adapt`) aborts serially with
     `double free or corruption` during `initialise()`. **Regression introduced by this branch**: on
     `main`, triangle meshes are not tree-refineable so adaptation falls back to gmsh **remeshing**;
     Phase 2 made triangle meshes tree-refineable and the droplet's initial adaptation now takes the
     tree path. Several **distinct** foundational gaps are exposed once tree-based tri refinement runs on
     a mesh with FaceElement/InterfaceMesh layers, a moving mesh and curved (macro-element) boundaries.
     Established by valgrind + a leak-test (skip `delete Object_pt` in `Tree::~Tree` → the abort turns
     into a clean error) + targeted instrumentation:
     - **(A) The crash [FIXED, commit c0df5e2].** `Tree::~Tree` (`tree.cc:132-139`) deletes `Object_pt`
       only for **father** nodes (`nsons>0`); leaf elements live in `Mesh::Element_pt` and are deleted by
       `Mesh::~Mesh`. Invariant: after adapt `Element_pt` holds **only leaves**. Valgrind + a
       `Tree::~Tree` leak-test showed a refined **father was ALSO in `Element_pt`** (freed by the tree,
       then re-deleted by `Mesh::~Mesh` (mesh.cc:678) → use-after-free → cascading heap corruption).
       Root cause: `adapt_mesh` was **aborted mid-way by an exception** (the `get_nodal_s_in_father`
       "Implement" throw below, and/or MACRO ELEM), *after* splitting the 37 elements but *before* the
       `Element_pt` leaf-rebuild (`stick_leaves_into_vector`), leaving exactly 37 refined fathers stuck in
       `Element_pt` (confirmed by an `nsons()>0` count in the `~TemplatedMeshBase2d` body: `fathers=37`).
       The fix is to stop the abort: implement the missing tri method (B') so adapt completes and rebuilds
       `Element_pt` cleanly. (Base-pointer subtlety for future debuggers: the tree holds
       `RefineableElement*`, `Element_pt` holds `FiniteElement*` — different values under multiple
       inheritance; identity checks need `dynamic_cast<void*>`, unusable on the already-freed side.)
     - **(B') `get_nodal_s_in_father` unimplemented for tri/tet [FIXED, commit c0df5e2].** Only
       Line/Quad/Brick overrode it; tris threw "Implement" whenever `further_build()` sampled father data
       at son nodes — for **DG/discontinuous fields** or `nodal_dimension != dim`. Basic tri Poisson/Stokes
       tests never hit that branch; the droplet (axisymmetric, moving mesh, interface DG fields) does.
       Implemented for `BulkElementTri2dC1/C2` using the same son→father map as
       `RefineableTElement<2>::build`. **`initialise()` now completes** (with macro elements off). Verified
       machine-zero on the CR/TH triangle residual-oracle tests (which exercise the DG branch).
     - **(B) `MACRO ELEM` unimplemented [DEFERRED — disable instead].** `RefineableTElement<2>::build`
       (`refineable_telements.cpp:736-745`) throws "MACRO ELEM" on a triangle father with a macro element
       (curved boundary). Per project decision, **deactivate macro elements** on such meshes
       (`use_macro_elements=False` on the GmshTemplate) rather than implementing curved tri refinement.
       With macro elements off the throw does not occur.
     - **(C) A latent double-delete of reused opposite interior-facet elements [FIXED, commit 5309fee].**
       `opposite_interior_facets` held duplicate pointers on 2:1 non-conforming inter-domain facets.
     - **(D) Position-hang ALE Jacobian on adapted moving-mesh tris [NOT A BUG — the analytic ALE Jacobian
       is EXACT for both quads and tris; the earlier "inexact" finding was an FD-oracle artifact].**
       An earlier forward-difference FD Jacobian check reported the tri ALE coupling
       `d[field]/d[mesh_position]` as inexact on adapted (hanging) tri meshes (off by ~200-1400) and only
       for hanging (non-conforming) refinement. **This was a false positive caused by forward-difference
       truncation on stiff entries, not a real Jacobian defect.** Definitive minimal test
       (`scratchpad/ale_rect.py`: `LaplaceSmoothedMesh` + `PoissonEquation(source=f(x), space="C2")` on a
       compressed rectangle, prescribed/error-based refinement, quad **and** tri):
       * `hanginfo_Pos` is filled **correctly** for both shapes — verified by direct dump: a C2-quadratic
         position hang whose masters are the coarse edge's 3 nodes with the correct quadratic Lagrange
         weights (e.g. tri hang node at edge-param t=0.8 → weights `{-0.12, 0.64, 0.48}`; quad at t=0.75 →
         `{-0.125, 0.75, 0.375}`), valid master local-eqns, `pos_local_eqn=-2` for the hanging node.
       * The codegen redistribution (`PositionFiniteElementSpace::write_generic_RJM_jacobian_contribution`,
         codegen.cpp:1850; via `get_hanginfo_str`→`hanginfo_Pos`) **is** applied for both.
       * **eps sweep + central difference (`scratchpad/eps_sweep.py`) proves the analytic Jacobian is
         correct**: the tri's *relative* forward-difference error (~8e-4) is **identical** to the quad's,
         and it scales **linearly with eps** (3738 → 376 → 37.7 → 3.8 as eps 1e-6→1e-9) — the textbook
         signature of O(eps·f″) truncation. **Central difference at eps=1e-7 matches the analytic to
         ~6.5e-7 relative (tri) / ~5.8e-9 (quad).** The tri only *looked* worse in absolute terms because
         refinement+compression produces a small distorted element → a stiff entry (analytic ≈ 4.6e5 vs
         quad's ≈ 55), so the same relative truncation shows up as a large absolute number.
       * With a **central-difference relative** FD-oracle, **all** cases (quad/tri × uniform/selected/
         error-based-adapt, with hanging nodes) are **exact**.
       **Lesson:** the forward-difference FD-oracle (`move_fd.py`, and earlier droplet checks) gives false
       positives on stiff moving-mesh entries; always use central differences with a relative tolerance for
       ALE Jacobian verification. The core adaptive ALE machinery (position hanging + analytic position
       Jacobian) is correct for tris.
     **Net:** the reported **crash is fixed** and the core adaptive-tri ALE Jacobian is **exact**. The
     droplet's separately-observed slow (×0.66) `presolve_gas_phase()` Newton was previously attributed to
     (D) via the same forward-difference oracle; since (D) is now disproven, that attribution is void and
     the droplet's slow Newton (if it reproduces) must be **re-investigated with a central-difference
     oracle** — likely a different term (interface/opposite-side coupling) or genuine stiffness/conditioning
     from the distorted adapted mesh, **not** the ALE position hang. Core tri refinement is unaffected
     (`tests/test_triangle_refinement.py` 28/28); the coverage gap that let the misdiagnosis stand is the
     absence of a **central-difference** moving-mesh + hanging FD-oracle regression test (for both quads and
     tris) — that should be added.
   - **§4.6 — Refined tri/tet nodes were missing their interpolated Lagrangian coordinates on moving
     meshes [ROOT-CAUSED + FIXED, `src/refineable_telements.cpp`].** Repro
     `pyoomph_runs/Bugs/TriAdapt/adapt_test.py` (`RectangularQuadMesh(split_in_tris="crossed")` +
     `LaplaceSmoothedMesh` + Taylor-Hood Stokes + `RefineToLevel(1)@"top"`; `p.solve(); p.adapt();
     p.solve()`): quads take 1 Newton step post-adapt; **tris took 2 steps or diverged to ~1e20,
     non-deterministically** (valgrind clean).
       * **Root cause:** `RefineableTElement<2>::build` (and `<3>::build`) set every new node's Eulerian
         position `x` (via `father->get_x`) and field values, but **never set the Lagrangian/reference
         coordinates `xi`** — so new tri/tet nodes kept `xi=0` while their `x` sat at the correct geometric
         midpoint (verified: 1752 nodes with `x≠xi`, worst discrepancy 0.5; quads had 0). `LaplaceSmoothedMesh`
         (and any solid residual) is written in terms of the deformation `x − xi`, so the identity mesh looked
         grossly deformed → a **spurious mesh residual** (`27.5` on `mesh_x`; the identity-mesh test residual
         was `17.5` even for *conforming* `refine_uniformly` with **zero** hanging nodes — proving it was not a
         hanging-interface issue). This both (a) drove the mesh to move — activating the weakly-nonlinear
         `bulkforce` coupling → multi-step Newton — and (b) blew the conditioning up to cond ≈2e8 (`smax`
         11670), on which MKL Pardiso non-deterministically produced a garbage factorization (`||J·dx+r||~1e20`).
         Those were **downstream symptoms**, not the disease: scipy SuperLU always solved the (spurious) matrix
         cleanly, which is why the earlier investigation mis-attributed it to "Pardiso fragility."
       * **Fix:** in both build methods, interpolate `xi` from the father with the same geometric shape
         functions used for `x`, guarded on `dynamic_cast<SolidNode*>`. (Note: oomph's
         `SolidFiniteElement::interpolated_xi` returns 0 here because pyoomph carries the Lagrangian coords per
         node but does not register a Lagrangian dimension at the element level — hence the manual shape-based
         interpolation. This mirrors what `RefineableSolidQElement` does for quads via `get_x_and_xi`.)
       * **Verified after fix:** identity-mesh residual `17.5 → 1e-16`; `x≠xi` count `1752 → 0`; post-adapt
         residual `27.5 → 0.37` (genuine interpolation error, same scale as quads); **1-step convergence** with
         umfpack; cond `2e8 → 2.7e6` (now on par with quads); Pardiso robust **12/12** (crossed) and 3/3
         (alternate_left/right); `tests/test_triangle_refinement.py` **28/28**.
       * This is almost certainly the true cause of the earlier "droplet slow-Newton (×0.66)" (§4.5(D)) and
         any other tri-adaptive **moving-mesh** slowness: the FD-oracle there passed because the Jacobian is
         self-consistent with the (spurious) residual — an FD check verifies dJ/dx, not that the residual
         itself is physically zero, so it cannot catch a wrong-`xi` residual. Re-test the droplet on this fix.
       * The coverage gap that hid this: `tests/test_triangle_refinement.py` had no **moving-mesh**
         (`LaplaceSmoothedMesh`/solid) case. Add an identity-mesh regression: pin all boundary positions, refine
         (uniform and 2:1), assert the mesh residual is ~machine-zero for both quads and tris.
   - **§4.7 — C1-field (C1-on-C2) Jacobian missing a coupling at tri 2:1 interfaces [OPEN, deeply localized].**
     After §4.6, the full evaporating-droplet case (`pyoomph_runs/DropletEvapExamples/pinned_water_droplet_tri_adapt.py`,
     `mesh_mode="tris"`) **no longer segfaults**, **quads run end-to-end (exact Jacobian, quadratic Newton, exit 0)**,
     but **tris converge only linearly (~0.30/step)** and can hit the Newton iteration cap. A central-difference
     FD-check (`scratchpad/drop_fd_central.py`) localizes a single dominant defect: the **gas water mass-fraction**
     self-coupling `d[gas/massfrac_water]/d[gas/massfrac_water]` is wrong on tris (quads: fully exact).
       * **It is a genuinely MISSING analytic entry, not an FD artifact.** An eps sweep on the worst entry
         (`scratchpad/drop_epssweep.py`) gives **analytic ≡ 0.0** while the finite difference is a **constant ≈0.28
         across eps 1e-5…1e-10** (a non-shrinking FD vs an exact zero = a structural sparsity gap). Use this eps-sweep
         technique whenever an FD "inexactness" is suspected — forward-diff truncation on stiff entries fooled the
         earlier §4.5(D) investigation.
       * **Localization (via `scratchpad/drop_topo.py`, `drop_perturb.py`, `drop_values.py` + temporary C++ dumps in
         `fill_hang_info_with_equations_basebulk`):** the two coupled dofs are **free C1 massfrac vertices** ~0.02 apart
         in a **2:1 tri interface region near the contact line**; the coupling is **purely massfrac→massfrac** (no
         interface/position/multiplier fields); the two nodes **share no element**; **none of the row-node's C1 vertices
         hang**; perturbing the column dof changes **only its own stored value** (no intermediary node's stored value
         moves); the massfrac space is genuinely **C1** (`nnode_space=3`, vertices only, `space_index=3`).
       * **Ruled out:** dropped-to-(−1) hang masters (instrumented both the fast path and `flatten_hang_for_value`
         — **zero** free masters dropped); macro elements (disabling them changes nothing); the inter-domain
         `droplet_gas` interface coupling (a matched two-domain nonlinear opposite-side FD-oracle is exact; unmatched
         throws "Cannot locate opposite", which the droplet avoids); and ~24 single-domain minimal FD-oracles
         (nonlinear diffusion incl. reciprocal `rho`, subexpressions incl. nested/grad-of-ratio, `define_field_by_substitution`
         for the passive field, advection, mesh-velocity advection, moving mesh Laplace/PseudoElastic, C1, C1-on-C2,
         `time_scheme(BDF2)`, transient step, deep hanging chains) — **all exact**.
       * **CONFIRMED ROOT CAUSE — curved-mesh collinearity failure in the C1-hang detection.** The residual/Jacobian
         inconsistency traces to a single fragile test. pyoomph builds the C1(-on-C2) hanging in `mesh2d.cpp`
         `post_adapt_setup_hanging_nodes` by finding fine-side vertices that lie on a coarse element edge, using the
         **straight-chord** geometric predicate `node_strictly_on_segment` (mesh2d.cpp:242) with tolerance
         `d2 <= 1e-14*len2` — it explicitly assumes straight-sided edges. The droplet gas mesh is **gmsh-generated,
         C2 (curved)**: a fine vertex on a 2:1 coarse edge sits on the *curve*, off the straight chord by
         `d2/len2 ≈ 4e-6…8e-6` (≫ 1e-14). Instrumenting the predicate on the live droplet showed **1428 such
         rejections** (all at t≈0.5). Rejected → the node is never registered as hanging in the C1 value slot →
         its C1 value is instead set by the **dummy-value interpolation** in `interpolate_hang_values`
         (elements.cpp:938-958, the *residual* path) as the average of the coarse-edge corners — but the **Jacobian**
         (`fill_hang_info_with_equations_basebulk`, elements.cpp:665) has **no matching redistribution** (it sees the
         node as a pinned, non-hanging, non-c1-constrained dead dof, `nummaster=0`). Residual depends on the corners
         (incl. `j`); Jacobian doesn't → `analytic[i][j]=0`, FD finite. Quads never hit this: they carry oomph's
         standard mixed-order hanging (`setup_hanging_nodes` via the interpolating-node facilities) which is
         topological, not geometric. `numfields_C2=0` on the gas (a C2-*geometry* mesh whose only field is C1)
         maximizes the exposure.
       * **THE FIX (per code review): give `BulkElementTri2dC2` the oomph "interpolating node" facilities the quad has**
         — `further_setup_hanging_nodes`, `interpolating_node_pt`, `local_one_d_fraction_of_interpolating_node`,
         `get_interpolating_node_at_local_coordinate`, `ninterpolating_node_1d`, `ninterpolating_node`,
         `interpolating_basis` (all keyed on `value_id`; cf. `BulkElementQuad2dC2::*`). With these, oomph's own
         `setup_hanging_nodes` builds the mixed-order (C1-on-C2) hanging **topologically** for tris — no geometric
         collinearity guessing, curvature-robust — and both residual and Jacobian use the same real HangInfo. This
         removes the reliance on `node_strictly_on_segment` + the dummy-value residual path for C1 fields.
         (Interim/defensive alternative if the full port is deferred: make `node_strictly_on_segment` curvature-aware
         by testing against the coarse edge's C2 quadratic P-M-Q instead of the straight chord.)
       * **FIX IMPLEMENTED [mesh2d.cpp].** Rather than porting oomph's full tri tree-hanging (a large Phase-2b
         effort — `RefineableTElement<2>::setup_hanging_nodes/setup_hang_for_value/quad_hang_helper` are all
         stubs, so pyoomph does *all* tri hanging in `post_adapt_setup_hanging_nodes` geometrically), the
         detection was made **curvature-aware**: `node_strictly_on_quadratic_edge(P,M,Q,X,t)` tests a node
         against the coarse element's **quadratic** edge (M = the coarse element's own edge mid-node, found
         topologically via `coarse_edge_mid_node`), used as an **additive fallback** after the straight-chord
         test (so straight-mesh behaviour is unchanged). A depth guard on the flatten `resolve` recursion
         turns any bug-induced hang cycle into a clean error instead of a segfault. Verified: droplet tri
         central-diff FD-check now **exact** (was 15%); first Newton solve **quadratic** (was linear ~0.3);
         **without** re-adaptation the full transient droplet runs to completion with quadratic convergence;
         `tests/test_triangle_refinement.py` 34/34 and the adapt/tet/stokes/constrained suites 66/66 (no
         regression). NOTE: the coarse mid-node must **not** be used directly as the C2-hang master M (it can
         itself be hanging → hang cycle → stack overflow, seen on CR error-adaptivity); keep M = the fine leaf
         node found from the `between` set.
       * **REMAINING (separate) follow-up: transient divergence under *re-adaptation*.** With the (now exact)
         Jacobian, a droplet run *with* `spatial_adapt` during the transient diverges/stalls at the first
         re-adapt step, while the identical run *without* re-adaptation converges quadratically to completion,
         and a central-diff FD-check of the (stationary) Jacobian *after* a re-adapt is exact. So this is NOT a
         hanging/stationary-Jacobian bug; it is a transient issue (BDF2 time-derivative history interpolated
         onto the re-adapted moving mesh) that was previously **masked** by the inexact-massfrac Jacobian
         damping Newton into slow-but-stable convergence. Line-search (`globally_convergent_newton=True`)
         removes the blow-up but the Newton then stalls, confirming a residual transient-Jacobian/history
         inconsistency. Next: central-difference FD-check of the **transient** Jacobian (with dt + history)
         right after a re-adapt.

   - **§4.8 — Tri hanging weights now come from the oomph "interpolating node" facilities (hybrid) [DONE].**
     Follow-up to §4.7: replace the hand-written linear/quadratic Lagrange weight formulas in
     `post_adapt_setup_hanging_nodes` with the same interpolating-node facilities the quad hanging uses
     (`interpolating_basis`, `interpolating_node_pt`, `ninterpolating_node` on `BulkElementTri2dC2`; the
     base `RefineableElement` isoparametric defaults cover plain linear coarse tris). **Design = "hybrid":**
     the *coarse neighbour element* is still found by the robust **geometric facet adjacency** (a facet
     {P,Q} incident on exactly one element is the coarse side), because the pure oomph **tree** route
     `RefineableTElement<2>::gteq_edge_neighbour` is **unreliable for triangle sons** — the QuadTree
     descent (translate_s / s_lo/s_hi / north-equivalent rotation) assumes quad son geometry, and the
     triangle split's **inverted middle son** makes it return `dl=0` (claims a same-level neighbour) for
     genuine 2:1 interface edges, so it misses ~2/3 of the tri interfaces (verified by the cross-validator
     below: 32 tree slots vs 96 geometric, tree ⊂ geo, never wrong masters). The interpolating-node
     facilities the quads use (committed `b566043`) are kept and now *consumed* by the hybrid path.
       * **Local coordinate is analytic, not `locate_zeta`.** The detection already yields `t` (the edge
         parameter, which *equals* the local edge coordinate for both straight and quadratic edges), and
         P,Q are vertices of the coarse element, so `s_c = (1-t)·s(P) + t·s(Q)` lands *exactly* on the
         reference edge. `locate_zeta`'s 2D Newton inversion returned `s` slightly off the edge → leaked
         tiny basis weights onto off-edge coarse nodes (which may themselves hang → spurious masters / hang
         cycles); the analytic `s_c` has zero off-edge leakage.
       * **Shared-node skip is weight-based, not pointer-based.** Skip X when the coarse basis puts weight
         ≈1 on a single master (X sits *at* a coarse interpolating node). Pointer equality
         (`get_interpolating_node_at_local_coordinate == X`) was too strict: at a T-junction **two distinct
         coincident nodes** can each be the C2 mid of a *different* coarse element's edge at the same point;
         pointer-guard let each hang on the other → **2-cycle** (caught by the flatten depth guard on the CR
         error-adaptivity test). The weight-based test skips both, matching the old hand-written "skip the
         mid M" semantics physically.
       * **Cross-validation harness (`PYOOMPH_TRI_HANG=validate`, `tree`).** `post_adapt` can snapshot the
         tree-set hanging (per node, per slot: sorted master position+weight) and diff it against the
         geometric/hybrid hanging, logging every mismatch. This is what isolated the `gteq` tri-son failure
         and proved the hybrid HangInfo is a correct superset. The `tree` path is **retained only as this
         diagnostic** — it is inert by default (`PYOOMPH_TRI_HANG` unset ⇒ mode 0 ⇒ the tree helpers
         `setup_hanging_nodes`/`setup_hang_for_value` return immediately and `post_adapt` runs the hybrid).
       * **Verified:** `tests/test_triangle_refinement.py` 34/34, tet suite 20/20 (no 3D regression), and
         the curved multi-domain droplet central-diff FD-check (`scratchpad/drop_fd_central.py tris`) still
         **exact**.

5. **Variable / anisotropic schemes** (§5): quad 1→2 (both directions), simplex
   bisection; directional error estimator; generalised balance rule.
6. **MPI hardening** across all shapes; distributed adaptivity tests.
7. **Unrefinement / `rebuild_from_sons` per pattern**, and curvilinear
   macro-element mapping for simplices/wedges/pyramids (§1.2) where curved
   boundaries meet refinement.

## 7. Load-bearing decisions (settled for this branch)

* **Hanging-node, not conforming red-green/bisection.** Reuses `HangInfo`,
  `complete_hanging_nodes_recursively`, MPI sync, and the flattening/constraint
  work already landed; matches the "hanging nodes must be assigned correctly"
  requirement. Conforming refinement would discard all of that and need a
  separate closure-reversion engine.
* **One generic engine, not per-shape trees.** The anisotropic requirement is
  what tips this firmly — per-shape compass trees cannot express variable splits.

The biggest latent risk is the geometric hang helper's coordinate composition
across mixed/anisotropic facets (§3.3); the genuinely new engineering is the
`RefinementPattern` catalogue (§3.1) and the balance rule (§3.3/§5).
