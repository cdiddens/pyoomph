# Initialisation cost: where the seconds go before the first solve

**Status: profiled, then reduced by ~27 %; one further idea tried and rejected; the largest single
item is still open.** All measurements are 2d quadrilaterals — the profiling harness is
dimension-generic, but 1d, 3d and simplices have *not* been run (§7).

The subject is `Problem.initialise()` on a large but otherwise trivial problem: no adaptation, no
distribution, no solve. It cost about 25 µs per degree of freedom, of which roughly 40 % was work
done two to five times over. It now costs about 18.5 µs/dof; what remains is dominated by one thing (§5.2).

---

## 1. The observation

```python
p = Problem()
p += RectangularQuadMesh(size=[1, 1], N=[500, 500])
p += (PoissonEquation(source=1) + DirichletBC(u=0) @ "*") @ "domain"
p.initial_adaption_steps = 0
p.initialise()          # 25 s before, 18.5 s now - and the problem has not been solved
```

250 000 quads, C2, 998 001 dofs. One fresh output directory per run, JIT cache warm, idle machine:

| N | ndof | before | after | µs/dof before → after |
|---|---|---|---|---|
| 100 | 39 601 | 1.34 s | **0.79 s** | 33.8 → 19.9 |
| 200 | 159 201 | 4.24 s | | 26.6 |
| 300 | 358 801 | 9.38 s | **6.5 s** | 26.1 → 18.1 |
| 400 | 638 401 | 16.0 s | | 25.1 |
| 500 | 998 001 | 25.3 s | **18.5 s** | 25.3 → 18.5 |

The relative gain is largest at the small end, where the fixed per-run costs (§4, the JIT cache
prune) dominate. Quote the totals with care: five back-to-back N=500 runs on this machine spread over
19.0-20.4 s (the first is always the fastest, so it is thermal, not warm-up). The per-routine self
times below are stable to a few percent and are the numbers each change was judged on.

**Nothing here was superlinear.** Worth stating, because it ruled out the usual suspects — no
accidental O(n²) node lookup, no quadratic vector growth, no repeated `nnode()` scan inside an inner
loop. What was left was a large constant: about fifteen separate linear sweeps over 250 000 elements /
1 000 000 dofs, several of which ran more than once.

## 2. How to measure it

Both obvious tools fail here, so the technique is worth recording.

- **`perf` is unavailable.** `perf record` on the development machines returns *"Failure to open any
  events for recording"* — `kernel.perf_event_paranoid` is not set permissively and raising it is a
  machine-wide change. Nothing in this document came from a sampling profiler.
- **`cProfile` is blind to nanobind.** It records Python frames and CPython builtins, but a nanobind
  method is neither, so its cost is silently folded into the `tottime` of whatever Python function
  called it. Read naively, cProfile says `reapply_boundary_conditions` spends 2.4 s *in itself* — a
  function whose body is nine lines of loop and one call. The 2.4 s is `assign_eqn_numbers`.

So the entry points are wrapped by hand. A nanobind method inherited by a Python subclass can be
shadowed with `setattr` on that subclass, and the wrapper subtracts the time its own wrapped children
consumed, giving **self** times that sum to the total rather than counting nested calls repeatedly.
[examples/initialisation_profile.py](examples/initialisation_profile.py) is the runnable version; it
accounts for 100 % of the wall time at N=500.

Two traps that cost time on the way, both of which make a routine look *free*:

- **Wrap the concrete mesh class.** `MeshFromTemplateBase` is not in the MRO of
  `MeshFromTemplate1d/2d/3d` — they are siblings that each carry their own copy of the mixin's
  methods (`_install_mixin`), not subclasses. A `setattr` on `MeshFromTemplateBase` succeeds, changes
  nothing, and the wrapped routine reports zero.
- **Do not wrap anything called per element.** Instrumenting `element_pt` (160 000 calls) inflated
  `map_nodes_on_macro_elements` from 0.22 s to 1.65 s. The wrapper overhead *is* the measurement at
  that granularity.

## 3. The breakdown, before and after

N=500, 998 001 dofs, warm JIT cache, self times. "calls" is per `initialise()`.

| routine | before | after | calls (before → after) |
|---|---|---|---|
| `oomph::Problem::assign_eqn_numbers` | 6.7 | 6.4 | 3 → 3 |
| `Mesh::generate_from_template` | 2.9 | 2.8 | 1 |
| `Problem::ensure_dummy_values_to_be_dummy` | 2.7 | 1.0 | 5 → 3 |
| `Mesh::setup_Dirichlet_conditions` (bulk) | 2.3 | **0.0** | 5 → 3 |
| `setup_tree_forest` (oomph-lib `QuadTreeForest`) | 2.2 | 2.2 | 1 |
| `_compile_bulk_equations`, of which `set_element_code` | 1.6 (1.52) | **0.6 (0.62)** | 1 |
| `_link_geometry_and_equations`, of which `define_geometry` 1.1 | 1.2 | 1.1 | 1 |
| `Mesh::check_integrity` → `check_all_neighbours` | 1.2 | 1.2 | 1 |
| `Mesh::setup_initial_conditions` | 1.0 | 0.9 | 1 |
| `Mesh::apply_additional_dof_constraints` | 0.85 | **0.04** | 3 |
| `assign_global_base_element_indices` | 0.36 | 0.35 | 2 |
| `map_nodes_on_macro_elements` | 0.35 | 0.19 | 1 |
| `_compile_interface_equations` (JIT cache prune) | 0.29 | 0.04 | 3 |
| rebuild/build global mesh, `relink_external_data`, rest | 0.9 | 0.9 | |
| **total** | **25.3 s** | **18.5 s** | |

Size dependence, checked against N=100: everything scales with the mesh except
`compile_bulk_element_code` (GiNaC + `dlopen`, ~0.35 s) and `_compile_interface_equations`, which are
flat. Those are the whole of the fixed startup cost — 2 % of the total at N=500, 30 % at N=100.

## 4. What was changed

Each item was measured individually. None of them changes *what* is computed; they stop work that
produced nothing, or stop producing it twice.

**`Mesh::setup_Dirichlet_conditions` — 2.3 s → 0.** [mesh.cpp](../src/mesh.cpp). Every loop filled
the `x`/`xi` buffers, and the elemental one evaluated two element midpoints per element, *before*
testing `dirichlet_active`. On a bulk mesh whose Dirichlet conditions all live on its boundary
(interface) meshes — the common case — every flag is false and the whole routine was a no-op that
took 0.46 s a call. It now returns immediately when no flag is set at all, and each kind of dof
(positions, continuous fields, the interface-mesh node loop with its per-node normal evaluation, and
the elemental DL/D0 loop) is skipped unless something of that kind is active. Nothing outside an
`if (dirichlet_active[...])` branch had a side effect, so this is exactly equivalent. Unpinning a
condition that was switched *off* is not this routine's job — `ensure_dummy_values_to_be_dummy()`
unpins everything on the way in.

**`BulkElementBase::setup_additional_dof_constraints` and `Mesh::apply_additional_dof_constraints`
— 0.85 s → 0.04 s.** [elements.cpp](../src/elements.cpp), [mesh.cpp](../src/mesh.cpp). Both loops in
the element routine already tested `get_additional_dof_constraints()` before acting, but the
`c1_corner_lookup` map was built first — one `std::map` with its allocations per element, for the
overwhelming majority of problems that use no `ConstrainFieldsToC1Space`/`ConstrainPositionsToC1Space`
at all. The element now returns before that if none of its nodes carries a constraint, and the mesh
answers the same question once from its node list (`nnode()` visits instead of one pass per element
over all its nodes). Interface meshes have an empty `Node_pt` and keep taking the element route.

**`Problem::ensure_dummy_values_to_be_dummy` — 0.54 s → 0.33 s a call.**
[problem.cpp](../src/problem.cpp), [elements.cpp](../src/elements.cpp). The two passes are still
needed (dummy-ness is a per-`Data` property shared between elements, so a single pass would see a node
already pinned by its neighbour), but the `dynamic_cast<BulkElementBase*>` is now done once, in the
first pass, and the survivors are kept for the second. Inside `unpin_dummy_values` the two separate
node loops — one unpinning positions, one unpinning values — were merged into one, and the
`dynamic_cast<Node*>` hoisted out of the per-coordinate-direction loop; `pin_dummy_values` got the
same hoist. Same operations, one pass through the (scattered, cache-cold) `Node` objects instead of
two.

**Two duplicated `setup_pinning()` calls — 5 sweeps → 3.**
[problem.py](../pyoomph/generic/problem.py). `reapply_boundary_conditions()` opens with
`setup_pinning()` + `before_assigning_equation_numbers()`, and both `initialise()` and
`set_initial_condition()` did exactly those two calls immediately *before* calling it. Worse, the
extra sweep was the stale one: `reapply_boundary_conditions()` first flushes the additional dof
constraints, so whatever the earlier `setup_pinning()` computed was discarded and rebuilt a few lines
later regardless.

**`Mesh::setup_initial_conditions` — the `ic_index` lookup moved to the top.**
[mesh.cpp](../src/mesh.cpp). The "does this mesh define an IC under this name?" test used to sit
*below* the nodal-normal precomputation, so a mesh that defines nothing under the requested name
still paid a full per-element `get_normal_at_s()` sweep and built the `nodal_normals` map before
bailing out. Invisible on the flat bulk mesh here (its `nodal_dim != eldim + 1`, so no normals are
computed), and worth having on every codimension-1 mesh, which is where that block runs.

**The C1→C2 template conversion — 1.52 s → 0.62 s** (measured directly on `set_element_code`, three
runs each, spread under 0.03 s)**.** [meshtemplate.cpp](../src/meshtemplate.cpp),
[kdtree.cpp](../src/kdtree.cpp). `MeshTemplateElementCollection::set_element_code` upgrades every
template element once the compiled code reports a C2 dominant space, and each upgrade asks
`add_intermediate_node_unique()` for the element's mid-edge, face-centre and cell-centre nodes.
Those resolved to `add_node_unique(x, y, z)` — a k-d tree nearest-neighbour query with a 1e-8
absolute tolerance, followed by an incremental insertion into a dynamic nanoflann tree growing to a
million points. 1.25 M queries and 750 k one-at-a-time insertions for a 250k-quad mesh. See §5 for what this took.

**`Mesh::map_nodes_on_macro_elements` — 0.35 s → 0.19 s.** New mesh-level method
([mesh.cpp](../src/mesh.cpp), bound in [nanobind/mesh.cpp](../src/nanobind/mesh.cpp)); the Python
loop used to cross the nanobind boundary three times per element (`element_pt`, `get_macro_element`,
`map_nodes_on_macro_element`).

**The JIT fingerprint stat storm — 0.33 s per run, at any problem size.**
[jit_cache.py](../pyoomph/generic/jit_cache.py). `_prune_if_needed` and
`_prune_fingerprints_if_needed` ran after *every* store and each `stat()`ed the whole shard tree: with
~21 000 fingerprint entries that was **105 191 `stat()` calls** per `initialise()`, a fixed tax on
every script no matter how small (a quarter of the total at N=100). One run adds a handful of entries,
so pruning that often cannot change the outcome — both now prune on the first store of the process and
then only every 64th, and the fingerprint prune, whose cap is on the entry *count*, gets that count
from `listdir` alone and only builds the stat listing when something actually has to be evicted.

## 5. Intermediate template nodes: identity is topological, not geometric

An intermediate node is defined by *which* mesh entity it is the centre of. The template resolved it
by position instead, and that turned out to be load-bearing in a way worth writing down.

**Step 1, keying by the parent tuple, bought nothing.** A hash map from the sorted parent node
indices to the node index answers a repeat lookup exactly and in O(1). But on a 500×500 quad mesh only
~40 % of the 1.25 M calls are repeats (each interior edge is seen twice, each cell centre once), and
the map insertion on the other 60 % cancelled the saving: 1.52 s → 1.59 s.

**Step 2, dropping the geometric query, gave 1.52 s → 1.03 s**, and **step 3, deferring the k-d tree
insertion, gave 1.03 s → 0.57 s.** The tree is only ever read by `add_node_unique`, so
`KDTree::add_point_deferred()` appends to the point cloud and leaves it unindexed; every query path
(`point_present`, `nearest_point`, `radius_search`, `k_nearest`) calls `index_deferred_points()`
first, which indexes the whole backlog in one `addPoints(start, end)`. nanoflann's dynamic adaptor
rebuilds a sub-index per `addPoints` call, so adding n points as one range costs far less than n
single adds. The deferral is invisible from outside; a dimension upgrade clears it, since the upgrade
constructor re-indexes the entire cloud anyway.

**And step 2 broke every mixed hex/wedge/pyramid mesh** — 40 failures across `test_mixed_3d.py`,
`test_pyramid_refinement.py`. The reason is the interesting part: **the C2 node constructions are not
shape-invariant.** A brick places a side-face centre at the midpoint of the face's two vertical
edge mid-points; a wedge does the same for its quadrilateral faces; a pyramid places its base centre
on the 0–2 diagonal. Only the brick's top and bottom faces, and every edge mid-point everywhere, are
built from the entity's corners. So a brick and the wedge next to it construct the *same physical
node* from different parents, and the 1e-8 coincidence test was the only thing that ever made them
one node. Keying on the placement parents gave it two identities and cracked the mesh.

The fix separates the two roles: **identity from the entity's corner set, position from whatever
parents the element wants to average.** `add_entity_centre_node_unique(key_corners, parents, ...)`
takes both, and the four non-canonical constructions — a brick's four side-face centres and its cell
centre, a wedge's three quadrilateral-face centres, a pyramid's base centre — now pass the face (or
cell) corners as the key while placing the node exactly where they did before. Boundary and domain
membership are inherited over the corner set too, which is what they should always have been: a
face-centre node is on a boundary only if the whole face is.

The k-d tree is still consulted on a map miss, but only when the template contains an element of
order higher than C1 that was added *directly* through the `add_*_C2` API rather than produced by
conversion (`has_predefined_higher_order_elements`). Such elements bring midside nodes that never
passed through `add_intermediate_node_unique()`, so they are not in the map and can only be found
geometrically. Everywhere else the map is complete by construction.

Two things got better beyond the timing. Distinct entities whose centres happen to coincide are no
longer silently welded — the case
`test_mixed_3d.py::test_mixed_node_sharing_ignores_node_positions` exists for. And the periodicity
record still stores the *placement parents*, which is what `set_element_code` matches against when it
links the partner side, so that path is untouched.

## 5.2 Tried and rejected: reusing the elemental local equation numbering

`assign_eqn_numbers` is the largest single item, 6.4 s over three calls, and **96 % of it is the
elemental local pass** — measured directly: `assign_eqn_numbers(False)` (global numbering + `Dof_pt`)
takes 0.09 s at N=500, `assign_eqn_numbers(True)` takes 2.2 s.

The attempt: call the base with `assign_local_eqn_numbers=false`, then decide from a fingerprint
whether the local pass has anything to do. The fingerprint was FNV-1a over everything a local equation
number is derived from — every `Data` and element by address (so a replaced element or reallocated
node shows up), every value by its equation number (so any pin/unpin/constrain, and hence any
renumbering, shows up), plus each mesh's topology generation, `Store_local_dof_pt_in_elements`, and
every element's node/internal/external data. Under MPI the verdict was `MPI_Allreduce`d, since the
local pass is not collective. It worked, and took initialisation from 19.4 s to 16.2 s.

**It is wrong, and `tests/test_tet_refinement.py` says so within seconds** (four failures, Newton
diverging with an infinite residual). `Mesh::assign_local_eqn_numbers` is not a numbering function.
It reaches `BulkElementBase::assign_additional_local_eqn_numbers()`, which also runs
`register_c1_constraint_position_masters()` and **`fill_element_info()`** — the rebuild of the
`eleminfo` struct the generated code reads: value and coordinate pointers, hang info with its equation
numbers. And that is not a function of the equation numbering alone. The interface equation remapping
(`InterfaceMesh::update_equation_remapping`) is computed *after* the numbering, in the same
`assign_eqn_numbers` call, which is precisely why six sites in `problem.py` call
`reapply_boundary_conditions()` **twice in a row** with a comment saying the second call is needed to
set the remapping up correctly. The second call is the one that folds the new remapping into
`eleminfo` — and it is exactly the call a fingerprint match would skip, because by construction
nothing it looks at has changed.

The lesson generalises: **anything that skips `assign_local_eqn_numbers` has to reason about
`fill_element_info`, not about equation numbers.** A correct version would have to fingerprint the
element info's inputs, which include the remapping vectors and whatever else `fill_element_info` will
read in future — a much weaker invariant to hold onto than "the numbering did not change", and one
that fails silently and catastrophically when it slips. The code was reverted; the reasoning is
recorded in a comment at `pyoomph::Problem::assign_eqn_numbers` so the next reader does not re-derive
it.

Reducing the *number* of calls instead was also looked at and not done. In `initialise()` the three
are: `problem.py:3796` (needed before `distribute()`/adaptation), the one inside
`set_initial_condition()` (needed because an initial condition may overwrite a Dirichlet value), and
`problem.py:3899` (after `rebuild_global_mesh_from_list(rebuild=True)`). The third is redundant in the
plain path — nothing between it and the second changes the submesh list — but proving that in general
means proving a negative about `init_output()`, which users override. A guard would have to be
positive evidence that the structure changed, not the absence of evidence that it did not.

## 6. What is left

Ordered by size at N=500. Percentages are of the 18.5 s that remains.

| item | s | note |
|---|---|---|
| `assign_eqn_numbers` × 3 | 6.4 (34 %) | §5.2. Either a `fill_element_info`-aware skip, or a proof that the third call is unnecessary. |
| `Mesh::generate_from_template` | 2.8 (15 %) | Element and node construction, ~11 µs/element. Not yet looked at for waste. |
| `setup_tree_forest` | 2.2 (12 %) | oomph-lib's `QuadTreeForest::find_neighbours`, which resolves each candidate neighbour with `get_node_number()` — a linear scan over the element's 9 nodes, ~8 of them per tree per candidate. Vendored code; a per-element node→index map would fix it. |
| `check_integrity` | 1.2 (6 %) | `check_mesh_integrity` defaults to `"initially"` (`problem.py:675`), so oomph-lib walks the forest verifying the neighbours it has just built, on every run of every script. Deliberately left on: it is a correctness net, not waste. `p.check_mesh_integrity=False` is a one-line opt-out. |
| `define_geometry` | 1.1 (6 %) | Pure Python, 250k elements added one at a time across nanobind. A vectorised bulk-add (coordinate array + connectivity array) would make it nearly free and would serve every structured mesh in `simplemeshes.py`. |
| `ensure_dummy_values_to_be_dummy` × 3 | 1.0 (5 %) | Already halved. Going further means iterating the mesh node list rather than each element's nodes (~2.25× fewer visits on a quad C2 mesh), with an element-loop fallback for meshes with no node list of their own. |
| `setup_initial_conditions` | 0.9 (5 %) | Calls the generated `InitialConditionFunc` once per (node, field, history level) even for fields the IC does not define, where it just returns the default. A per-IC mask of which field indices are actually set would skip it — needs a `jitbridge.h` field, a codegen change and a `FORMAT_VERSION` bump. |

## 7. What has not been checked

- **1d and 3d.** All numbers here are 2d quads. The routines are dimension-generic, so the *shape* of
  the profile should carry over, but the weights will not: a 3d brick has 27 nodes per C2 element
  instead of 9, which multiplies exactly the per-node loops that dominate
  (`ensure_dummy_values_to_be_dummy`), while `setup_tree_forest` moves to `OcTreeForest` with 6
  neighbours per element instead of 4. 1d is the opposite extreme and is worth running mainly because
  it isolates the fixed costs. The harness takes the dimension as its second argument:
  `python3 initialisation_profile.py 40 3`.
- **Triangles and tetrahedra.** `set_element_code`'s conversion path is different for simplices
  (C1→C1TB→C2TB), and the Gmsh/`MeshTemplate` route reaches `generate_from_template` with a different
  element mix.
- **Interface-heavy problems.** This problem has four trivial boundary meshes. A problem with real
  interface equations puts far more into `_compile_interface_equations` and into the `InterfaceMesh`
  branch of `setup_Dirichlet_conditions` — the branch that was given a guard here largely on the
  strength of reading it, since this problem barely exercises it.
- **Whether the same repetition costs anything during a *solve*.** `reapply_boundary_conditions` is
  called from roughly twenty sites in `problem.py`, and six of them call it *twice in a row*
  (`problem.py:5076-5077`, `5399-5400`, `5752-5753`, `5976-5977`, `6093-6094`, `6147-6148`). §5
  establishes that the second call is not redundant — it is what installs the new equation remapping
  — but each pair still costs two full numbering passes on the eigenvalue, bifurcation-tracking and
  continuation paths, which on a large problem is a bigger prize than initialisation.

## 8. Validation

The changes were checked against the four things they must not break, with every suite compared
against a `git stash`ed build of the same tree where anything failed:

| area | suites | result |
|---|---|---|
| everything | the whole non-slow suite, twice (before and after the template-node change) | 1173 passed, 572 skipped, 6 pre-existing failures |
| mixed 3d shapes | `test_mixed_3d`, `test_pyramid_refinement`, `test_wedge_refinement` (the ones §5 broke and then fixed) | pass |
| adaptivity | `test_adaptivity`, `test_adaptive_2d_campaign`, `test_adaptive_3d_campaign`, `test_adaptive_interface_coupling`, `test_constrained_adaptivity`, `test_desired_ndof`, `test_mixed_mesh`, `test_mixed_3d`, `test_triangle_refinement`, `test_tet_refinement` | pass |
| moving meshes | `test_curved_boundaries`, `test_interface_mesh_stiffening`, `test_solid_momentum`, `test_remeshing_leaks` | pass |
| MPI `--distribute` | `test_mpi_adaptivity`, `test_mpi_adaptivity_3d`, `test_mpi_remeshing`, `test_mpi_state_files`, `test_mpi_boundary_membership`, `test_mpi_structural_assembly`, `test_mpi_global_meshdata`, `test_mpi_interface_coupling` (`--full`) | pass except one pre-existing failure |
| initial conditions | `test_state_file_restart`, `test_tracers`, `test_vectorial_conditions` | pass except six pre-existing failures |

Two failures are **pre-existing on `develop` at 4221549** and unrelated to any of this — both were
reproduced on a clean build:

- `test_state_file_restart.py::test_restart_reproduces_the_state_and_the_continuation` (6 cases): the
  residual right after loading a state differs from the residual at write time by ~5e-16 against a
  `tol=0.0` bit-exactness assertion. All three kinds (`transient`, `tempadapt`, `movmesh`).
- `test_mpi_interface_coupling.py::test_moving_mesh_distributed`: `ale-tri_left-12-level` reports
  ndof 2942 on rank 1 versus 2938 serially.
