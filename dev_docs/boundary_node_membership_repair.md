# Repairing over-marked boundary-node membership after an adapt

Status: **implemented and on by default.** This was the "own validation pass" that
`macro_elements_generalisation.md` §23.2 asked for when it deliberately left the defect alone. File and
line references are to the tree state at the time of writing (branch `develop`, after `8da42d7`).

The defect itself is characterised and measured in `macro_elements_generalisation.md` §23; that section
is the reference for *how big it is*, this one for what was done about it. Two decisions were settled
up front: the MPI half is done properly with a cross-rank push (§4), and the repair is on by default
with the flag as an escape hatch (§5).

One thing the plan did not anticipate, and it is the part worth reading if you touch this again:
**a hanging node is on the boundary without being a node of any tagged face** (§3.1). Stripping those
is not a harmless mislabel — it makes a later refinement create a plain `Node` where a `BoundaryNode`
was needed, which cannot be undone. That was caught by `test_tet_refinement.py`, not by anything
written for this work.

## 1. The defect, and why the truth is already available

After refinement a node can be marked as lying on a mesh boundary it is not on. The cause is the
inheritance rule: a new node gets the boundaries **shared by all its generating nodes**.

* 2d triangles: the two end nodes of the father edge — `RefineableTElement<2>::get_boundaries`,
  `src/refineable_telements.cpp:409-467`, reached from `src/refineable_telements.cpp:1088-1102`.
* 3d tets: the intersection over all generating nodes, `src/refineable_telements.cpp:2612-2655`.
* Wedges: `src/wedges_and_pyramids.cpp:462-503`. Pyramid and brick sons: `src/elements.cpp:8098-8140`
  and `src/elements.cpp:8269-8330`.
* oomph's own `RefineableQElement<2>/<3>::get_boundaries` use the identical end-node rule, so native
  quad/brick refinement over-marks the same way.
* The mesh template's intermediate-node rule does too — `src/meshtemplate.cpp:1670-1673` (2-parent) and
  `src/meshtemplate.cpp:1707-1714` (3-parent) — so the **unrefined** mesh can already be wrong before
  any adapt happens.

Two nodes can share a boundary label without the edge between them lying on that boundary. The trigger
is an element with **two or more faces on the same boundary**: its remaining faces then have all their
vertices on that boundary, so every one of their edges is mislabelled, and each mislabelled edge seeds
more at the next refinement. §23.1 measured 84 of 165 marked nodes spurious at three refinements for a
tet with 2 of 4 faces on one boundary, and 0 spurious on every realistic mesh tried.

What makes the repair possible is that pyoomph already carries the exact truth, and has since the
face-tag work:

* every element holds `std::map<short, std::vector<unsigned>> face_boundaries` (`src/elements.hpp:375`,
  accessors `:474-477`, rationale `:456-472`);
* seeded once from the template's facets by `seed_face_boundaries_from_facets()`
  (`src/mesh.cpp:7470-7516`);
* inherited **exactly** at every split — `BulkElementBase::dynamic_split()`,
  `src/elements.cpp:7948-7957`, via `face_index_in_father()` (`src/elements.cpp:7680-7845`), which
  covers lines, quads/bricks, tris, tets, wedges and the heterogeneous pyramid split and throws rather
  than silently dropping tags for anything else;
* consumed by `setup_boundary_element_info_from_face_tags()` (`src/mesh.cpp:7537`), which is the path
  taken by `src/mesh2d.cpp:400-405` and `src/mesh3d.cpp:394-399` — both `return` early, so the legacy
  node-membership cleanup at `src/mesh2d.cpp:411-442` does **not** run there.

So the true node set of a boundary is simply the union of the nodes of its tagged faces. The repair is
to compute that and drop everything else.

Note the asymmetry this creates today, and why the defect is survivable in the meantime: boundary
*elements* come from the tags and are correct, Dirichlet conditions are applied through those face
elements, and only code that iterates nodes *by boundary index* sees the wrong answer.

## 2. Stage 1 — a complete face-node accessor

`get_vertex_nodes_of_face()` (`src/elements.hpp:452`) returns vertices only — it keys the `facets` map
— and is not enough here. Dropping a genuine mid-side node's membership would be a correctness bug, so
the truth set has to be built from *all* nodes of a face.

There is no uniform accessor today, and the gaps are not where one would guess:

* `nnode_on_face()` throws for wedges and pyramids (`src/wedges_and_pyramids.hpp:779`, `:919`) because
  their facets are mixed tri/quad; those classes carry `nnode_on_face_by_index(face_index)` instead
  (`:781`, `:921`, `:1402`).
* `oomph::TElement<1,*>` and `TElement<3,*>` declare neither `nnode_on_face()` nor
  `get_bulk_node_number()`.
* **`BulkElementTetra3dC2TB` has a face node that `get_bulk_node_number` cannot return.**
  `build_face_element` wires it in by hand from a function-local
  `std::vector<int> Central_node_on_face{13, 12, 10, 11}` (`src/elements.cpp:11653`), because
  `TElement<3,3>::Node_on_face` is only `[4][6]`. Those face bubbles are created with
  `boundary_possible=true`, so they genuinely carry boundary membership. **This is the one way the
  repair could strip a real membership, and it is the reason the `missing` counter of §3 has to ship
  enabled.**

The developer helper `help_me_with_the_facets()` (`src/elements.cpp:14837-14868`) already encodes the
same per-family table (`C1→3`, `C2TB→7`, `C2→6`, `WedgeC1→face<2?3:4`) — that it had to is the proof
that no uniform accessor exists.

Add to `BulkElementBase`, right after `src/elements.hpp:452`:

```cpp
// All nodes of a local face -- vertices AND face-interior ones (C2 mid-side, brick/wedge face centre,
// the C2TB tet face bubble): exactly the set build_face_element() wires into the face element, and so
// exactly the nodes an interface mesh on that face owns. get_vertex_nodes_of_face() is deliberately
// vertices-only (it keys the `facets` map) and must not be used for boundary membership.
std::vector<oomph::Node *> get_all_nodes_of_face(const int &face_index) const;

virtual unsigned nnode_on_face_by_index(const int &face_index) const { return this->nnode_on_face(); }
virtual unsigned node_index_on_face(const int &face_index, const unsigned &i) const
  { return this->get_bulk_node_number(face_index, i); }
```

`get_all_nodes_of_face` is the loop over those two virtuals; define it near `src/elements.cpp:14804`.
Overrides required:

| class | count | index |
|---|---|---|
| `BulkTElementLine1dC1` / `C2` | 1 | `face==-1 ? 0 : 1` / `: 2` |
| `BulkElementTetra3dC1` / `C2` | 3 / 6 | default |
| `BulkElementTetra3dC2TB` | 7 | `i<6` default, `i==6` → `Central_node_on_face[face_index]` |
| `BulkElementWedge3dC1/C2`, `BulkElementPyramid3dC1/C2` | forward to the `oomph::` base's `nnode_on_face_by_index` | default |

The defaults are already right for `QElement<1/2/3>` (Line/Quad/Brick C1+C2, `Qelements.h:299` gives
`nnode_1d^(dim-1)`) and for `TElement<2>` (Tri2dC1/C2, which already declare 2 and 3 at
`src/elements.hpp:1624`, `:1732`). The cell-interior bubbles of `Tri2dC1TB`, `Tri2dC2TB` and
`Tetra3dC1TB` are created with `boundary_possible=false` (`src/meshtemplate.cpp:349`, `:397`, `:699`)
and must stay out of the face sets. `BulkElementODE0d` and `PointElement0d` have empty
`Possible_Face_Indices`.

The four forwarding overrides are **not** optional. `oomph::WedgeElementC1` already declares a virtual
with the identical signature (`src/wedges_and_pyramids.hpp:781`) and `BulkElementWedge3dC1` inherits
from both it and `BulkElementBase`; without an explicit override in the derived class the two are
separate vtable slots and a call through a `BulkElementBase*` reaches the throwing default. One derived
declaration legally overrides both bases.

While here, hoist `Central_node_on_face` out of `build_face_element` into a static member so the
constant is not written down twice, and cross-reference it from `help_me_with_the_facets()`.

## 3. Stage 2 — the repair itself (local, no communication)

Declare next to `src/mesh.hpp:709`, define after `src/mesh.cpp:7560`:

```cpp
std::pair<unsigned,unsigned> check_boundary_node_membership_against_face_tags() const;  // (spurious, missing)
unsigned repair_boundary_node_membership_from_face_tags();
std::vector<std::pair<oomph::Node*,unsigned>> Pending_boundary_membership_removals;
bool repair_boundary_node_membership = true;
```

1. Bail out unless `face_boundary_tags_valid && nboundary()`.
2. **Truth set.** For every element — halo elements *included*, they are evidence — and every
   `(face, boundaries)` in `get_all_face_boundaries()`, insert all of `get_all_nodes_of_face(face)`
   into `truth[b]` for each `b < nboundary()`.
3. **Decidable set.** The nodes of every element with `!el->is_halo()`; in serial, every node. This is
   provably safe: `mesh.cc:5228-5265` retains a foreign element as a root halo iff it shares a node
   with an element of my domain, and `mesh.cc:5789-5800` keeps it through pruning on the same
   criterion. So *if a node lies in at least one non-halo element on a rank, every element incident to
   that node is present on that rank* — and the rank's truth set for it is complete. Element
   `is_halo()` is propagated to sons (`src/elements.cpp:8003`, `:8245`,
   `src/refineable_telements.cpp:1377`, `:2470`, and the wedge/brick/quad equivalents), so it is
   reliable at this point. Node `is_halo()` is **not** — see §4.
4. **Per boundary, one pass** over `Boundary_node_pt[b]`. It is a protected member of `oomph::Mesh`, so
   rebuild the vector wholesale rather than calling `Mesh::remove_boundary_node`, which is O(n) per
   call (`mesh.cc:233-246`) and would make the sweep quadratic. Drop a node iff
   `!n->is_obsolete() && decidable(n) && !truth[b].count(n)`. Skipping obsolete nodes keeps
   `prune_dead_nodes` (`mesh.cc:993-1081`, which deletes an obsolete node only once it is on no
   boundary) entirely out of the picture.
5. For each dropped node, `if (bn->is_on_boundary(b)) bn->remove_from_boundary(b);`. **The guard is
   mandatory.** PARANOID is off by default (`CMakeLists.txt:52`), so the "not on this boundary" check
   inside `BoundaryNodeBase::remove_from_boundary` is compiled out and `nodes.cc:3076` dereferences a
   possibly-null `Boundaries_pt` — a segfault, not an exception.
6. Record each dropped `(node, boundary)` in `Pending_boundary_membership_removals` for §4.

Also call the repair at the end of `seed_face_boundaries_from_facets()`, so the template-level
over-marking of §1 is fixed once on the unrefined, pre-distribution mesh.

### 3.1 Hanging nodes must be exempt — the defect this work nearly shipped

`truth[b]` is a union of the **node lists** of tagged faces, and that is not the same set as "the nodes
lying on boundary b". A **hanging** node sits inside a coarser element's facet without being one of its
nodes, so it belongs to no tagged face while being every bit as much on the boundary. In 3d it does not
even need the coarse element to be its neighbour: an edge of a boundary facet is shared by several
elements, only some of which have a face on that boundary, and if the one that refines the edge is an
interior element then the new midpoint lies on the boundary and appears in nobody's tagged face list.

Removing such a membership is not a mislabel that the next adapt puts right. It is irreversible, and it
fails somewhere else entirely: when the coarse element is eventually refined, its new node's class is
chosen from its generating nodes' memberships (`BoundaryNode` vs plain `Node`, and a plain `Node` can
never become a boundary node afterwards), so the node is born plain and the interface mesh dies with
"Node ... is not a boundary node" from `Mesh::generate_interface_elements`.

Measured on `test_tet_refinement.py::test_tet_node_sharing_ignores_node_positions`, a non-uniformly
adapted `TetCubeMesh(N=3)` that also unrefines: four nodes — all four `is_hanging()` and correctly
marked one step earlier — lost their memberships at one adapt, and the interface rebuild threw at the
next. So `repair_boundary_node_membership_from_face_tags()` skips any node with `is_hanging()`
(`may_drop_boundary_membership()` in `src/mesh.cpp`), and `check_boundary_node_membership()` exempts
them too, so that a mesh the repair has deliberately left alone does not read as broken.

That is deliberately conservative in one direction: a genuinely spurious mark on a hanging node
survives, which is exactly the behaviour there was before any of this — and it is cleaned up by the
next adapt at which the node stops hanging. The configurations the repair exists for (§6) refine
uniformly and have no hanging nodes at all, so nothing is lost there.

**Remove only; never add.** The inheritance rule is an intersection, so the marked set is always a
superset of the truth and under-marking cannot arise from refinement. Adding is also structurally
impossible: `src/missing_masters.hpp:363-375`, `:563-580`, `:990-1007`, `:1129-1145` choose the node
*class* (`BoundaryNode` vs plain `Node`) from the sender's flag, so a node that arrived as a plain
`Node` can never acquire membership, and `Mesh::add_boundary_node` (`mesh.cc:255`) throws on one. A
non-`BoundaryNodeBase` also cannot be in `Boundary_node_pt[b]` and cannot report `is_on_boundary(b)`,
so it never becomes a removal candidate — a `dynamic_cast` guard is enough. The `missing` half of
`check_boundary_node_membership_against_face_tags()` is therefore a *diagnostic*: a nonzero value means
seeding or the §2 tables are wrong, and it should surface rather than be patched over.

## 4. Stage 3 and 4 — the two hooks, and the cross-rank push

### 4.1 Where the hooks go, and why the placement is load-bearing

`setup_boundary_element_info()` is called from oomph's `TreeBasedRefineableMeshBase::adapt_mesh` at
`src/thirdparty/oomph-lib/include/refineable_mesh.cc:1304` — the single choke point for `adapt`,
`refine_uniformly` and `refine_selected_elements`. Add two no-op virtuals to `oomph::Mesh` (`mesh.h`,
near `classify_halo_and_haloed_nodes` at `:1706`), each with a `//FOR PYOOMPH` comment and an entry in
`src/thirdparty/INFO_oomph-lib`:

```cpp
virtual void reconcile_boundary_node_membership_locally() {}
virtual void reconcile_boundary_node_membership_across_processes() {}
```

* **Local pass** immediately after `refineable_mesh.cc:1305`, *outside* the
  `Lookup_for_elements_next_boundary_is_setup` block — the repair needs the face tags, not
  `Boundary_element_pt`. It must precede the 3d macro repositioning of hanging boundary nodes at
  `:1430-1500`, which reads node membership and would otherwise snap a spurious node onto a boundary it
  is not on.
* **Collective push** after `:1781` (`} // End if (this->nelement()>0)`) and before the
  `classify_halo_and_haloed_nodes` block at `:1784`.

Both halves of that second placement are load-bearing, and getting either wrong **deadlocks** rather
than misbehaving:

1. Everything from `refineable_mesh.cc:965` to `:1781` is inside `if (this->nelement() > 0)`, whose
   comment says explicitly it is there because "in a distributed problem with multiple meshes ... a
   particular process may not have any elements on a particular submesh". A collective placed anywhere
   inside that block hangs on such a rank.
2. `classify_halo_and_haloed_nodes` runs only at `:1790`, i.e. *after* both hooks. So
   `Halo_node_pt`/`Haloed_node_pt`/`Shared_node_pt` and `Node::is_halo()` are stale for every node this
   adapt created. Only the **element** halo lists are usable — `Mesh::halo_element_pt(p)` (`mesh.h:1740`)
   recomputes the leaves from the root halo elements on every call.

### 4.2 The push

`TemplatedMeshBase::reconcile_boundary_node_membership_across_processes()`, guarded by
`#ifdef OOMPH_HAS_MPI` plus `is_mesh_distributed()` and `nproc()>1`. For each domain `d != my_rank`,
walk `haloed_element_pt(d)` and pack, for every node of every haloed element that appears in
`Pending_boundary_membership_removals`, the triple `(index in the list, local node index, boundary)`;
send the count, then the triples. The receiver walks `halo_element_pt(dd)` in the matching order and
applies the same removals, rebuilding `Boundary_node_pt[b]` once at the end and using the same
`is_on_boundary` guard. Then clear the pending list.

Reuse the loop structure of `check_halo_element_consistency` (`src/mesh.cpp:690-800`) — the same `d`/`dd`
pairing and the same "both sides walk their lists in the same order" contract.

Why this is complete: a node that a rank cannot decide lies only in halo elements there, hence is a node
of some halo element, hence receives the owner's decision; and two ranks that both find it decidable
both see every element incident to it and compute the same answer. Traffic is zero whenever the repair
is a no-op, which is every realistic mesh.

`TemplatedMeshBase1d` overrides both hooks as documented no-ops: `src/mesh1d.cpp:46` rebuilds
`Boundary_element_pt` *from* node membership and never reads the tags at all, so the tags are not the 1d
authority and applying the repair there would be circular.

This is not optional extra work. Nothing currently exchanges boundary membership between ranks — the
only cross-rank check, `src/mesh.cpp:690-800`, packs seven doubles of geometry, level, flags and error —
so a local-only repair would silently diverge, and `InterfaceMesh::setup_boundary_information2d`
(`src/mesh.cpp:6452-6490`) would then build different numbers of corner elements on the halo and haloed
copies.

## 5. Stage 5 — Python surface, and why default-on

Mirror the existing `identication_of_boundary_elements_by_facets` property
(`src/nanobind/mesh.cpp:1744`, `:1775`, `:1805`) on all three `TemplatedMeshBase{1,2,3}d`:

* `repair_boundary_node_membership` — read/write bool, **default `True`**;
* `check_boundary_node_membership()` → `(spurious, missing)`, the diagnostic the tests use.

Docstrings go in `src/nanobind/`, never in `pyoomph/_pyoomph_core.pyi`, which the build regenerates.

Default-on, with the flag as an escape hatch rather than an opt-in. §23.1 measured **0 spurious nodes**
on a triangular disc, a brick spherical octant, a tetrahedral ball, a rectangular quad/tri mesh, a full
circular disc and an unstructured gmsh tetrahedral ball, each at one, two and three refinements, up to
64512 elements. On all of those the truth set equals the marked set, so the repair provably removes
nothing — it is a no-op that can be demonstrated rather than argued. Behind an opt-in flag nobody would
enable it and the strict xfail could not be flipped, which is the point of doing this at all.

## 6. Verification, as run

**The three cases that used to be a strict xfail now pass.** `tests/test_curved_boundaries.py`'s
`test_boundary_node_membership_is_repaired_when_a_boundary_wraps_an_element` keeps the
`[("halftet",1,1), ("halftet",3,84), ("singletet",3,35)]` parametrisation, with the old count quoted in
the failure message -- "84 spurious" describes a regression here far better than "not 0". The marked
counts after the repair are exactly the old ones minus the spurious ones: 165-84 = 81 and 165-35 = 130.
The companion test on realistic meshes now asserts set *equality* in both directions rather than
"no spurious", which is what catches an incomplete face-node enumeration.

**Every element family, every space, both directions.** `tests/test_boundary_element_identification.py`
sweeps hex/tet/wedge/pyramid and the `all_four` mixed layout over C1, C2, C1TB and C2TB at 0-2
refinements (C1TB/C2TB for the simplices only -- a wedge or pyramid throws rather than producing an
element), asserting `check_boundary_node_membership() == (0, 0)` *and* the same comparison made
against the INTERFACE MESHES instead of the tags. The second one is the load-bearing check: an
interface mesh is assembled by `build_face_element()`, which is the routine that knows about the
oddities, so it is an oracle independent of the tables in §2 -- a self-consistent check against the
same tables could not catch a wrong one. A wider sweep run while developing covered 103 (family, space,
level) combinations, all `(0, 0)`.

That the C2TB tet is really covered was checked rather than assumed: its interface element has **7**
nodes, and an unrefined tet slab has 11 bulk nodes marked on a wall against 9 for C2 -- the two extra
face bubbles. Had `nnode_on_face_by_index()` returned 6, those would have shown up as `missing`.

**MPI.** `tests/test_mpi_boundary_membership.py` at 2 and 4 ranks, on the slab of `tests/slab_mesh.py`
(§6.1). It refines once **after** distributing, which matters: the initial refinement happens before
distribution, where every element is local and the push has nothing to do. Per rank it asserts
`check_boundary_node_membership() == (0, 0)`; globally it compares the union over ranks of the marked
positions against the union of the interface meshes' positions (neither is conclusive on one rank,
since a halo-only node can sit on a facet owned elsewhere); and separately it asserts that no two ranks
disagree about a node they both hold.

**The MPI test was verified to have teeth.** With the cross-rank push commented out and everything else
unchanged, it fails as intended -- "tet/side: 3 nodes marked on no facet anywhere" at 2 ranks, 14 at 4,
and the agreement check reporting `node at 0.9375,0.0625,0.25 is on ['wall'] on rank 0 but on
['side', 'wall'] on rank 1`. That divergence is invisible to every other check in the codebase, which
is the whole reason §4.2 exists. On the 4-rank hex slab, 55 of the 180 nodes a rank holds are
undecidable there, so the push is not a theoretical concern.

### 6.1 `tests/slab_mesh.py`

The single tetrahedron of `test_curved_boundaries.py` is the defect in its smallest form and cannot be
distributed. `SlabTemplate` is the same configuration made big enough: an N x N x 1 slab of bricks or
tets whose top and bottom faces share the boundary name "wall", so every element has two faces on
"wall", all eight of its nodes are on "wall", and its four vertical edges have both ends on "wall"
without lying on it. Serial, with the repair switched off:

```
                nref=1        nref=2
brick slab     21 spurious   135 spurious
tet slab       31 spurious   303 spurious
```

and zero with it on, in both directions. The brick numbers matter on their own: those come from oomph's
own `RefineableQElement<3>::get_boundaries`, so this is not only a pyoomph-simplex problem.

### 6.2 Suites run

Serial, all passing: `test_boundary_element_identification`, `test_facet_adjacency`,
`test_curved_boundaries`, `test_segment_ordering`, `test_mesh_point_locator`, `test_mixed_3d`,
`test_adaptive_interface_coupling`, `test_tet_refinement`, `test_wedge_refinement`,
`test_pyramid_refinement`, `test_triangle_refinement` (452 passed), and the 2d/3d adaptive campaigns
with `--full`. MPI with `--full`, all passing: `test_mpi_boundary_membership`, `test_mpi_adaptivity`,
`test_mpi_adaptivity_3d`, `test_mpi_curved_boundaries`, `test_mpi_remeshing`, `test_mpi_state_files`,
`test_mpi_interface_coupling`, `test_mpi_error_estimator`, `test_mpi_adaptive_recovery`,
`test_mpi_undistributable`, `test_mpi_global_meshdata`, `test_mpi_structural_assembly`,
`test_mpi_observables`, `test_mpi_tracers`, `test_mpi_rank_zero_failures`, `test_mpi_newton_abort`.

Two failures found on the way were **pre-existing and unrelated**, both confirmed by rebuilding the
tree without this work and reproducing them identically:

* refining a **C1TB tet** segfaulted, on the first `refine_uniformly` of any tet mesh.
  `BulkElementTetra3dC1TB::local_coordinate_of_node` wrote `s[0..2]` without resizing `s`, and
  `FiniteElement::get_node_at_local_coordinate` (`elements.cc:3890`) hands it a default-constructed,
  i.e. empty, `Vector` to size itself -- so it was an out-of-bounds write on an empty `std::vector`.
  Every other shape resizes first (the 2d C1TB triangle, both wedges, both pyramids, and oomph's own
  `TBubbleEnrichedElementShape`, which is why C2TB was fine), so this one shape had simply been
  missed, and nothing had ever refined a C1TB tet. **Fixed**, and
  `test_uniform_tet_refinement_conforming` now sweeps all four tet spaces so that no enrichment goes
  unrefined again.
* `test_adaptive_3d_campaign.py::test_ale_moving_mesh[levels0-hex_pyr]` stalled at
  `max|residual| = 8.583e-09` against a 1e-11 tolerance, deterministically, on an unrefined mesh. Not
  a Jacobian problem: the same assembled system solved with SuperLU or UMFPACK gives 5.6e-15. It is
  MKL Pardiso being imprecise on that one matrix, far below the backward error its own escalation
  triggers on. **Fixed** by naming that single case in the campaign's `_EXACT_SOLVER_CASES`; see
  `dev_docs/pardiso_static_pivoting.md` §8 for the measurements and for why it is one case rather than
  a blanket policy.


## 7. Risks

The first two were the ones that mattered: risk 2 is what actually happened, in the form of §3.1.

1. **Incomplete face-node enumeration strips a genuine membership.** Concentrated on
   `BulkElementTetra3dC2TB` (bubble outside `Node_on_face`), `WedgeElementC2` (9-node quad facets) and
   `PyramidElementC2`. Mitigated by shipping the `missing` counter as an assertion rather than a silent
   loss, and by checking against the interface meshes rather than against the same tables (§6).
2. **Irreversibility.** Once a node's membership is gone, a son built from it is a plain `Node` and can
   never become a boundary node again. A wrong removal therefore surfaces much later, and far from its
   cause, as "is not a boundary node" out of `generate_interface_elements`. §3.1 is one instance of this
   that was found and fixed; anything else of that shape will look the same, so the first thing to
   suspect on such a throw is a node that is on the boundary without being on a tagged face.
3. **Boundary coordinates.** `remove_from_boundary` deletes the node's zeta for that boundary
   (`nodes.cc:3080-3086`) but never frees `Boundary_coordinates_pt` when the map empties, unlike its
   handling of `Boundaries_pt` at `:3088-3092`. So `boundary_coordinates_have_been_set_up()`
   (`nodes.h:2170`) stays true on a node with no coordinates left. Only spurious zetas are lost here and
   `pyoomph/meshes/zeta.py` re-sets zeta on every interface-element node after each adapt, so this is
   worth a line in the code comment and is a candidate `//FOR PYOOMPH` tidy-up, not a blocker.
4. **Periodic corner detection.** `pyoomph/meshes/bcs.py:649-659` uses
   `len(n.get_boundary_indices()) >= 2` to identify corner nodes and *raises* on a master/slave
   mismatch. Removing a spurious second membership flips that branch — only on a mesh that already had
   spurious marks, but it turns a silent wrong answer into an exception.
5. **Refinement patterns move on pathological meshes.**
   `enlarge_elemental_error_max_override_to_only_nodal_connected_elems` (`src/mesh.cpp:1033-1098`),
   `ensure_halos_for_periodic_boundaries` (`:1108-1155`), `Mesh::node_is_in_scope` (`:530-533`, where
   remesh interpolation skips any `is_on_boundary()` node in the bulk pass) and
   `nodal_interpolate_along_boundary` (`:2812-2831`, `:2977-3003`) all read membership. Expect numbers
   to move on the halftet/singletet family and nowhere else.
6. **Corner elements of interface meshes.** `InterfaceMesh::setup_boundary_information2d`
   (`src/mesh.cpp:6452-6490`) intersects the same node sets to find boundary-of-boundary corners, so
   smaller sets mean fewer corner elements. A per-rank difference there changes the element count
   between halo and haloed copies — which is why §4.2 cannot be skipped and why the MPI test must check
   cross-rank *agreement*, not just correctness.
7. **Deadlock if the collective is misplaced** — inside `if (nelement()>0)`, inside the
   `Lookup_for_elements_next_boundary_is_setup` block, or at `:1304`. See §4.1.
