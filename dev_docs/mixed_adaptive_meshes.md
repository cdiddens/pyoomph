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
     error-driven adaptive refinement (~6.6e-16). Full suite 24/24. Linear (C1) triangles
     only; C2 needs a finer node-sharing key (Phase 2a registry keys on the two corner
     nodes, which is ambiguous for the multiple new nodes a C2 father edge spawns).
3. **2D mixed quad + tri**, incl. quad-face/two-tri-face hangs and boundary
   facets across shape changes.
4. **3D tets → wedges → pyramids** (pyramid last: forces mixed offspring). Fill
   the wedge/pyramid `bulk_coordinate_derivatives` gaps (§1.2) as needed for face
   Jacobians.
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
