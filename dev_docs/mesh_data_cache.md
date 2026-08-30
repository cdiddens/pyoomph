# Mesh data cache: typed keys and globally merged (distributed) mesh data

Status: **implemented and tested**, except for operator support (§3.5, phase 2), which still raises.
`tests/test_mpi_global_meshdata.py` (6 tests, `--full`) covers 2/3/4 ranks, the discontinuous path,
the no-`--distribute` case and a repeated (cached) request. File/line references are to the tree
state at the time of writing (branch `develop`, after `7bea507`), i.e. they predate the changes
described here.

Two changes to `pyoomph/meshes/meshdatacache.py`:

1. **Typed cache keys.** `MeshDataCacheStorage._storage` is keyed by a bare `tuple[Any,...]` and
   read back positionally (`k[2]`, `k[6]`). Replace it by an explicit key object.
2. **A `global` flag.** On a distributed problem, ask for the *whole* mesh instead of the local
   partition: the per-rank `Mesh.to_numpy` outputs are gathered and merged into one consistent
   node/element/field set. Needed for plotting (only rank 0 draws), for remeshing boundary
   identification, and for anything else that reasons about mesh *topology* rather than about
   per-node values.

Part 1 is a small refactor and is a prerequisite for part 2 (the flag becomes a key field, and
`clear(only_eigens=True)` breaks silently if a field is inserted into the positional tuple).

---

## 1. What exists today

### 1.1 Data flow

```
Problem.get_cached_mesh_data(msh, ...)            problem.py:821
  -> MeshDataCacheStorage.get_data(...)           meshdatacache.py:449   keyed by the option tuple
       -> MeshDataCache.get_data(msh)             meshdatacache.py:414   keyed by the mesh object
            -> MeshDataCacheEntry(msh, ...)       meshdatacache.py:41    calls msh.to_numpy(...)
```

`MeshDataCacheStorage` holds one `MeshDataCache` per *option set*, each of which holds one
`MeshDataCacheEntry` per *mesh*. `Problem.invalidate_cached_mesh_data(only_eigens)` clears it
whenever values change.

### 1.2 What an entry holds

| attribute | shape / meaning | rank-dependent? |
| --- | --- | --- |
| `nodal_values` | `(nnode, nfields)` | yes — local nodes |
| `nodal_field_inds` | `{name: column}` | no (metadata) |
| `elem_indices` | `(nelem_out, max_indices)`, int | yes |
| `elem_types` | `(nelem_out,)`, int | yes |
| `D0_data` | `(nelem_out, numD0)`, or `(nnode, numD0)` if `discontinuous` | yes |
| `DL_data` | `(nelem_out, numDL, dim+1)`, or `(nnode, numDL)` if `discontinuous` | yes |
| `elemental_field_inds` | `{name: index}`, DL first then D0 | no |
| `merged_eigendata` | `{ev: {"nodal_values": (re, im), "DL_data": ..., "D0_data": ...}}` | yes |
| `nodal_local_exprs` | `{name: (nnode,)}`, **filled lazily** in `get_data` | yes |
| `local_expr_indices` | `{name: index}` | no |
| `vector_fields` | `{vector: [components]}` from the equation tree | no |
| `interface_lines_segs` | lazily built from `elem_indices` | yes |
| `_additional_eigendata` | `{ev: (preRe, preIm, preMerge)}`, set by operators | no |

`nelem_out` is the number of **output** element rows, which is *not* the number of mesh elements:
`Mesh::get_num_numpy_elemental_indices` (`src/mesh.cpp:55`) sums a per-element sub-element count, and
with `tesselate_tri=True` a quad yields 2 triangles (`src/elements.cpp:9406`) — more if hanging
neighbours force extra tesselation triangles. Remember this; see §6.1.

### 1.3 Ordering guarantees we can rely on

* Non-discontinuous: node row `i` is `mesh.node_pt(i)` for a bulk mesh
  (`Mesh::fill_node_map`, `src/mesh.cpp:1434`), and for an `InterfaceMesh` it is first-occurrence
  order over its elements (`src/mesh.cpp:4581`).
* Discontinuous: node rows are the concatenation of each element's own nodes, in element order
  (`fill_reversed_node_map`, `src/mesh.cpp:1445`) — i.e. no node is shared between rows.
* Element output rows are in mesh element order, each mesh element contributing its sub-elements
  consecutively.

### 1.4 What "distributed" means here

* `mpirun` **without** `--distribute`: every rank holds the complete mesh. Local == global; nothing
  to merge. `mesh.is_mesh_distributed()` is False.
* `mpirun` **with** `--distribute`: each rank holds its partition, plus halo copies of its
  neighbours' elements. Elements are owned by exactly one rank (`is_halo()`), nodes on the partition
  interface exist on several ranks and are *non-halo on all of them* — oomph-lib records those in the
  **shared node scheme** (`Shared_node_pt`, `src/thirdparty/oomph-lib/include/mesh.h:111`, built by
  `Mesh::setup_shared_node_scheme`, `include/mesh.cc:2739`), which is exactly "a unique correspondence
  between all nodes on the halo/haloed elements between two processors". This is the hook the merge
  will use (§6.2).

---

## 2. Part 1 — typed cache key (implemented)

### 2.1 The problem

```python
key = (nondimensional, tesselate_tri, eigenvector, eigenmode, history_index,
       with_halos, operator, discontinuous, add_eigen_to_mesh_positions)   # meshdatacache.py:452
...
if k[2] is not None or (k[6] is not None and k[6].depends_on_eigen()):     # meshdatacache.py:437
```

`k[2]`/`k[6]` are `eigenvector`/`operator`. Inserting a field anywhere but at the end silently
changes what `clear(only_eigens=True)` flushes — and "silently" here means eigen output keeps stale
data instead of raising.

### 2.2 The typed key, as built

A frozen dataclass, `MeshDataCacheKey`, with one field per option that changes the *content* of an
entry. Details settled during implementation:

* `MeshDataCacheKey.create()` normalizes any sequence of eigenvector indices to a **sorted** tuple, so
  `[1,0]` and `[0,1]` now hit the same slot (the old `tuple(set(...))` did not guarantee that).
* `MeshDataCache.__init__` keeps its old keyword/positional signature — callers construct it directly,
  e.g. `AxisymmetricReconnection` in `pyoomph/equations/topological_changes.py`, and the positional
  form `MeshDataCache(True, True)` is in use — and builds
  `self.key` from its arguments. The nine option attributes are now read-only properties on the key.
  The storage constructs it as `MeshDataCache(**key.as_kwargs())`, so the storage key and the cache's
  own key cannot drift apart.
* `MeshDataCacheEntry.__init__(msh, key)` takes the key object; it still mirrors the options into the
  attribute names operators and output classes read (`tesselate_tri`, `discontinuous`, `with_halos`, ...).
* `as_kwargs()` deliberately does not use `dataclasses.asdict`, which deep-copies non-dataclass values
  and would therefore clone the operator, breaking its identity.
* **Small behaviour change**: passing a list of eigenvectors with a non-`merge` eigenmode now raises.
  The guard existed (`"Multiple eigenvectors in MeshDataCache only works if eigenmode is set to
  'merge'"`) but was dead, because the storage had already converted the list to a tuple while the
  check only tested `list`/`set`. Such a request used to return plain, non-eigen data.

Verified with a scratch script covering: cache hit identity, `eigenvector=int`, the merge path with a
list of eigenvectors, key-order normalization, the operator path, and
`invalidate_cached_mesh_data(only_eigens=True)` flushing exactly the six eigen-dependent slots of
seven while leaving the plain one.

### 2.4 Notes

* `frozen=True` gives `__hash__`/`__eq__` for free. `operator` is an arbitrary object, hashed by
  identity — two structurally identical operator instances still get separate cache slots. That is
  today's behaviour; keep it, but say so in the docstring.
* `MeshDataCache` currently duplicates all nine options as attributes. Give it a single
  `self.key: MeshDataCacheKey` and let `MeshDataCacheEntry` be constructed from it
  (`MeshDataCacheEntry(msh, key)`), which removes the three-fold argument list repetition at
  `meshdatacache.py:41 / 399 / 449` and makes adding an option a one-line change.
  `MeshDataCacheEntry` keeps its existing attribute names as properties/assignments so external code
  (`cache.tesselate_tri`, `cache.discontinuous`, `cache.with_halos`, used by
  `MeshDataCombineWithEigenfunction.apply`) keeps working.
* `Problem.get_cached_mesh_data` keeps its keyword signature — it is public API and is called from
  tutorials (`docs/source/tutorial/advstab/response/linear_response_drum.py:94`). It just builds a
  `MeshDataCacheKey.create(...)` internally.
* Fix the mistyped `remkeys: list[str]` while there.

Cost: ~1 h, no behaviour change, no rebuild.

---

## 3. Part 2 — the `global` flag

### 3.1 Naming

`global` is a reserved word, so it cannot be a parameter or a dataclass field name. The parameter and
key field are therefore `global_mesh: bool = False`, and the entry attribute is
`MeshDataCacheEntry.is_global`.

### 3.2 Semantics

`get_cached_mesh_data(msh, ..., global_mesh=True)`:

* **Serial, or `mpirun` without `--distribute`** (`not msh.is_mesh_distributed()`): returns exactly
  the same entry the local request returns, on every rank. No communication, no extra cache slot
  content-wise (but still a separate key — see §3.4). This is the case the user described: global and
  local are the same mesh.
* **Distributed**: a **collective** call. Every rank must reach it. Rank 0 gets the merged entry;
  all other ranks get `None`.

So the return type becomes `MeshDataCacheEntry | None`, and only for `global_mesh=True`. Overloads
(`Literal[False]` -> `MeshDataCacheEntry`, `bool` -> `MeshDataCacheEntry | None`) keep the type
checker useful for the 15 existing call sites, which do not pass the flag at all.

Rejected alternative: return an *empty* entry (0 nodes, 0 elements) on the non-root ranks so that
callers need no `None` check. It hides the "you are not the rank that has the data" case and turns
it into an empty plot or an empty boundary list, which is much harder to debug than an
`AttributeError` on `None`. Rejected as well: broadcasting the merged entry to all ranks — the user
explicitly said rank 0 is enough, and it would multiply memory by `nproc` for the exact use case
(plotting) that motivated the feature. If a consumer later needs it everywhere, add
`global_root: int | None = 0` where `None` means allgather.

### 3.3 `global_mesh` and `with_halos`

Mutually exclusive: halos exist only to give each rank the neighbour data it is missing, and the
merged mesh is not missing anything. `global_mesh=True and with_halos=True` raises.

### 3.4 Collective contract and the cache

The dangerous part of putting a collective behind a cache is that **the hit/miss decision must be
identical on every rank**, otherwise rank 0 returns from the cache while the others sit in a gather
that will never be matched. Rules:

* Every rank inserts an entry for the key. Non-root ranks insert `None` (a sentinel object, so that
  "cached None" and "not cached" stay distinguishable).
* The key is built purely from the call arguments, which all ranks pass identically — as long as
  callers obey the contract.
* `invalidate_cached_mesh_data` is already only triggered from collective points (solve, set initial
  condition), so invalidation stays symmetric.
* Callers must not wrap a `global_mesh=True` request in `if get_mpi_rank()==0:` — unless they are
  inside the request scope of §3.4b, which exists precisely because plotting cannot obey this rule.

### 3.4b Rank 0 asking on its own: the request scope

The symmetric contract above is unusable for the case that motivated the feature. A plot definition
is user code that knows nothing about ranks, only rank 0 should draw, and which meshes it will ask
for is not known before it runs — so the other ranks cannot reach the same requests by themselves,
and "every rank runs `define_plot`, only rank 0 saves" would mean rendering the figure `nproc` times
and handling `None` in every plot element.

`run_with_global_mesh_data(problems, func)` in `meshdatamerge.py` solves it the other way round,
modelled on the existing `run_on_rank_zero` (`pyoomph/generic/mpi.py`):

* rank 0 runs `func` and broadcasts `(problem name, mesh full name, key fields)` on every **cache
  miss** for a `global_mesh=True` request;
* the other ranks sit in a serve loop, receive each request and replay the very same call. That is
  what makes the gather line up — and equally anything collective *inside* the local extraction, such
  as `set_eigenfunction_as_dofs` reassigning the dofs for an eigenvector request;
* a `None` broadcast ends the loop; it is sent from a `finally`, so a failure in `func` ends the loop
  too instead of leaving the others waiting, and `mpi_share_root_failure` then re-raises on all ranks.

A cache **hit** on rank 0 broadcasts nothing and needs nothing from the others, which is consistent:
the driver is rank 0's cache state alone. `BasePlotter.plot()` is now a one-line wrapper around
`run_with_global_mesh_data(self._named_problems, self._do_plot)`.

The operator is dropped from the broadcast key (it is refused for global data anyway and would not
survive pickling); requests for meshes of a problem the scope does not know about raise rather than
being sent into the void.

### 3.4c Perturbed requests: the eigendynamics animation

`Problem.create_eigendynamics_animation` draws the base state plus `Re(f*v)` for a complex factor
`f = A*exp(i*m*phi)*exp(lambda*t)` — one frame per phase, and a *mirrored* half of the same frame with
`exp(i*m*(phi+pi))` instead. Serially, `MatplotlibPlotter._get_mesh_data` just perturbs the dofs around
the extraction and restores them.

Under §3.4b that is impossible: the perturbation would be applied by rank 0, inside `func`, and merged
with everybody else's **base** state. Nor can rank 0 do it and tell the others afterwards —
`set_current_dofs` is collective, so one rank calling it alone deadlocks. The animation was therefore
refused on a distributed mesh, and `docs/source/tutorial/plotting/eigendynamics.py` was the last
tutorial that could not run under `--distribute`.

`merge_perturbed_global_mesh_data(msh, key, eigenvector_index, factor)` makes the perturbation part of
the request instead. The request tuple grows one optional element,

    plain:     (problem name, mesh full name, key fields)
    perturbed: (problem name, mesh full name, key fields, (eigenvector index, complex factor))

so a plain request is unchanged on the wire, and the serve loop unpacks by length. Only two scalars
travel: an eigenvector is replicated full length on every rank
(`SlepcEigenSolver._vector_to_global_array`), and `get_current_dofs`/`set_current_dofs` are global and
collective, so every rank reconstructs the identical perturbed state from the index and the factor.

Three properties are load-bearing:

* **Atomic.** Perturb, extract, gather, restore is one request. The restore is in a `finally`, and
  `_merge_on_root` — the only part that fails on rank 0 alone — runs *after* it. No rank can be left
  holding a perturbed state, whatever fails. This is also why the function cannot simply wrap
  `merge_global_mesh_data`.
* **Two invalidations, both inside the request.** `_local_payload` reads the ordinary
  (`global_mesh=False`) slot, which the plotter's field probes have already filled with base-state
  data, so the cache is invalidated *before* the perturbation; and the perturbed values would
  otherwise sit in slots keyed as if they were the base state, so it is invalidated *after* the
  restore too. Both happen on every rank, so the hit/miss symmetry of §3.4 survives.
* **The result is never cached.** The key has no room for the factor: the two mirror halves of one
  frame, and successive frames, would collide. The entry is returned directly. That matches the
  serial cost, which re-extracts per plotted part anyway.

Validation before the first collective is not a style choice here: anything that raises between the
broadcast and the gather leaves rank 0 unwinding while the others wait. `needs_merging`, `with_halos`,
the operator refusal (this path bypasses `MeshDataCacheStorage.get_data`, so §3.3/§3.5 no longer guard
it) and the eigenvector index are all checked up front.

**A serial bug fell out of this.** Local expressions are evaluated lazily, i.e. after the dofs are
restored, so an animated frame drew every ordinary field from the perturbed state and every
`add_local_function` field from the base state. The merged path evaluates them eagerly inside the
perturbed window (§4.1), so the two would have disagreed; `_get_mesh_data` now materialises them before
restoring, and both are right.

The same request scope carries the **tracers** of a plot, for the same reason: a particle belongs to
the process holding its element, so `MatplotLibTracers` -- which reads `mesh.get_tracers()` -- drew
whatever happened to be on the drawing rank, silently, since a scatter plot of a subset looks
perfectly reasonable. `gather_global_tracers` returns a `GatheredTracers` with the same
`get_positions`/`get_tags`/`get_ids`/`get_history` API a collection has, so the drawing code did not
change; the underlying gathers are `MPI_Allgatherv` sorted by particle identity. Trails need the
position history, and the only gather that carries it is the checkpoint one (`_save_state(True)`),
whose packing is decoded there.

`tests/test_mpi_eigendynamics.py` covers it at 2 and 3 ranks: the collective called symmetrically, the
same thing through the request scope (including a plain request afterwards that must return exactly
what the plain request before it did), the whole animation through
`create_eigendynamics_animation`, that no rank keeps a perturbed state, and that a perturbation which
reaches only one rank's dofs is *detectably* different — otherwise the comparison would prove nothing.

### 3.5 Operators

Operators (`MeshDataCombineWithEigenfunction`, `MeshDataRotationalExtrusion`,
`MeshDataCartesianExtrusion`) must run on the *merged* data — a rotational extrusion of a partition
is not a piece of the extrusion of the whole mesh (the `% mod_length` index wrap at
`meshdatacache.py:1324` is over the local node count).

But `MeshDataCombineWithEigenfunction.apply` itself calls `get_cached_mesh_data` twice
(`meshdatacache.py:561-562`). Run on rank 0 only, those nested calls are collectives issued by one
rank: deadlock.

Staging:

* **Phase 1**: `global_mesh=True` with `operator is not None` raises `NotImplementedError`. No
  current call site combines them (plotting passes no operator; `MeshFileOutput`, which does, writes
  one file per rank and a `.pvd` that stitches them, so it does not need merging at all).
* **Phase 2**: add `MeshDataCacheOperatorBase.required_cache_keys(base_key) -> list[MeshDataCacheKey]`
  (default `[]`; `MeshDataCombineWithEigenfunction` returns its Re/Im keys). The global path resolves
  those *collectively, on all ranks*, before merging; the operator then runs on rank 0 and its own
  `get_cached_mesh_data` calls are pure cache hits.

---

## 4. What has to be merged

Let rank `r` contribute node rows `N_r` and element rows `E_r`.

| entry field | merge rule |
| --- | --- |
| `nodal_values` | rows of the *representative* of each global node class, in a deterministic order |
| `elem_indices` | concatenate owned element rows, remap indices, pad to the global max width |
| `elem_types` | concatenate owned element rows |
| `D0_data`, `DL_data` | concatenate the **same** owned element rows (`discontinuous=False`), or the node blocks of owned elements (`discontinuous=True`) |
| `merged_eigendata[ev]["nodal_values"]` | same node mapping |
| `merged_eigendata[ev]["D0_data"/"DL_data"]` | same element mask |
| `nodal_local_exprs` | must be **eagerly** evaluated before the gather, see §4.1 |
| `nodal_field_inds`, `elemental_field_inds`, `local_expr_indices`, `vector_fields` | rank-independent; take rank 0's, but check the others agree and raise if not |
| `interface_lines_segs` | drop; recomputed lazily from the merged `elem_indices` (and it is finally *correct* — per-rank segments are cut at the partition boundary) |
| `mesh`, `operator`, flags | kept as-is; `mesh` is still needed for `get_unit`, code gen and the equation tree, all rank-independent |

### 4.1 Local expressions are the awkward one

`MeshDataCacheEntry.get_data(name)` for a local expression calls
`self.mesh.evaluate_local_expression_at_nodes(...)` lazily (`meshdatacache.py:271-303`), i.e. it
touches the *local* mesh. On a merged rank-0 entry there is no local mesh covering the global node
set, and calling it on rank 0 alone is wrong (it returns the local partition's values, silently
misaligned with the merged node ordering). Worse, the eigen branch calls
`set_eigenfunction_as_dofs`/`set_all_values_at_current_time`, which are collective.

Resolution: at merge time, on every rank, evaluate **all** entries of `local_expr_indices`, gather
them alongside `nodal_values`, and store them in the merged entry's `nodal_local_exprs`. Then set a
flag on the merged entry so `get_data` never falls into the lazy branch — and raises a clear
"not available on a merged mesh data entry" if a name is missing.

Eager evaluation costs one pass per defined local expression even if the caller wants none of them.
Make it controllable: `global_mesh_local_expressions: bool = True` on the key, or simply always eager
in phase 1 and revisit if a tutorial gets noticeably slower. Meshes typically define a handful.

### 4.2 Metadata consistency

`nodal_field_inds` can in principle differ between ranks (the count of additional interface fields is
derived from the element code instance, `src/nanobind/mesh.cpp:1381`). It should not, but a rank whose
partition has *no* elements of the mesh produces nothing at all — `to_numpy` raises "No elements in
mesh" there. So: ranks with `mesh.nelement()==0` contribute nothing and are excluded from the
metadata comparison; a mismatch between two non-empty ranks raises rather than producing an array
whose columns mean different things in different row ranges.

---

## 5. The merge algorithm

### 5.1 Non-discontinuous (`discontinuous=False`)

Per rank:

1. Build the local entry with `with_halos=False` and `operator=None`.
2. `own = ` mask of element rows whose **source mesh element** is not a halo (§6.1).
3. `used = numpy.unique(elem_indices[own])` — only nodes referenced by owned elements travel. Halo-only
   nodes (interior to a neighbour's partition) are dropped here; a node referenced by an owned element
   is by construction non-halo, and appears on every rank that references it.
4. `shared[p] = mesh.get_shared_node_numpy_indices(p)` for every other rank `p` (§6.2).
5. Send to rank 0: `nodal_values[used]`, `used`, `elem_indices[own]` (already remapped into `used`
   positions), `elem_types[own]`, `D0_data[own]`, `DL_data[own]`, the eigen/local-expression arrays,
   and `shared`.

On rank 0:

6. Union–find over `(rank, local_row)` pairs: for each unordered rank pair `(r, p)`, entry `j` of
   `shared_r[p]` and entry `j` of `shared_p[r]` are the same node. Skip `j` where either side is `-1`.
7. One representative per class. Global row order: sort classes by `(min rank in class, local row on
   that rank)`. Deterministic, and for `nproc==1` it reproduces the serial ordering exactly — which
   makes the regression test in §8 a plain array comparison rather than a set comparison.
8. `nodal_values` = the representatives' rows; `elem_indices` = concatenated per-rank rows with local
   indices mapped through `(rank, local_row) -> global row`.
9. Assemble a `MeshDataCacheEntry` from arrays (§7.3).

**Padding.** `elem_indices` has one row per element and `max_indices` columns, where `max_indices` is
the per-rank maximum over element types; ranks with different element types differ in width, and the
unused tail of a row is **uninitialised memory** (`new int[...]` at `src/nanobind/mesh.cpp:1396`,
never zero-filled; consumers read only as many entries as `elem_types` implies). So: pad to the global
max width with zeros, and when remapping, map only entries that are valid local indices
(`0 <= i < nnode_local`) — no element-type-to-node-count table needed. (Such a table does exist
implicitly in `pyoomph/output/meshio.py:139-175`; if one is ever needed, factor that one out rather
than writing a second.)

**Sanity check (cheap, keep on):** after merging, verify that the coordinates of the rows that were
folded together agree to a tight tolerance. If the shared-node matching were ever off by one, this
catches it immediately instead of producing a plausible-looking but scrambled mesh.

### 5.2 Discontinuous (`discontinuous=True`)

No node is shared between rows, so there is nothing to identify: node rows are the per-element node
blocks, in element order. Merging is a concatenation of the blocks belonging to owned elements, with
the element rows concatenated in the same order. No shared-node scheme, no union–find — and the
scheme must not be consulted either, since it is expressed in the *continuous* node numbering, which
does not describe these rows at all.

This is also the one case where dropping halo elements has to drop node rows: each element owns its
block, so a halo element's block is unreferenced once its element rows are gone. It is removed in the
entry itself (§9.7), which also means the node rows have to be renumbered in `elem_indices` — and that
any local expression evaluated later has to be filtered the same way, because the mesh always
evaluates it over all of its rows (§9.8).

### 5.3 Fallback backend: coordinate matching

If the shared-node scheme is unavailable (mesh distributed by something else, or an interface mesh
whose bulk mesh has no scheme), fall back to geometric matching on rank 0: `scipy.spatial.cKDTree`
over the gathered coordinates, `query_pairs(r=tol)` -> union–find, with `tol` relative to the mesh
bounding box (`1e-9 * diagonal`). Correct for ordinary meshes, wrong for meshes that legitimately
carry two distinct nodes at the same position (slits, a torn interface). Therefore: fallback only,
never the default, and it prints once which backend was used.

---

## 6. C++ side: two additions

Both are small and neither changes an existing signature.

### 6.1 Output element row -> source element index

**This fixes a bug that exists today**, independently of the merge (see §9.1). Needed because
`elem_indices` rows are sub-elements, not elements.

```cpp
// pyoomph::Mesh, src/mesh.cpp
// For each element row produced by to_numpy(tesselate_tri, ..., discontinuous), the index of the
// mesh element it was generated from. to_numpy splits e.g. a quad into two triangles when
// tesselate_tri is set (and into more when hanging neighbours require it), so the row count is not
// nelement() and rows cannot be zipped with element_pt().
std::vector<int> get_numpy_element_source_indices(bool tesselate_tri, bool discontinuous);
```

Implement by factoring the counting loop out of `get_num_numpy_elemental_indices`
(`src/mesh.cpp:55-102`) into a helper that optionally fills the vector, so the two can never drift
apart. Note that the counting pass mutates element state (`_numpy_index`, `_tess_hang_scoord`) — it is
idempotent for the same arguments, but the new method must be called with the *same* `tesselate_tri`
/`discontinuous` as the `to_numpy` call it describes.

Alternative considered: return the array as an 8th element of `to_numpy`'s tuple. Always consistent,
but it breaks every full unpack of `to_numpy` (3 sites in `meshdatacache.py`, 1 in
`tests/test_triangle_refinement.py:182` which slices `[:2]` and is safe). A separate method keeps the
public signature stable; the drift risk is handled by the shared helper.

### 6.2 Shared-node row indices

```cpp
// pyoomph::Mesh
// Row indices, in the node ordering to_numpy uses, of the nodes this mesh shares with process p.
// Entry j corresponds to entry j of process p's own list for this rank: oomph-lib builds
// Shared_node_pt in a matched order on both sides (Mesh::setup_shared_node_scheme).
// -1 marks a shared node of the underlying bulk mesh that this mesh does not itself contain, which
// keeps the index-by-index correspondence intact for interface meshes.
virtual std::vector<int> get_shared_node_numpy_indices(unsigned p);
```

* Bulk mesh: `nshared_node(p)` / `shared_node_pt(p, j)` (`include/mesh.h:2109/2122`) looked up through
  `fill_node_map`.
* `InterfaceMesh`: its own `Shared_node_pt` is empty (pyoomph builds interface meshes itself, oomph-lib
  never distributes them). Override to walk the **bulk** mesh's shared list and map each node through
  the interface mesh's own node map, emitting `-1` for nodes it does not contain. Interface elements
  do carry the halo flag of their bulk element (`src/wedges_and_pyramids.cpp:1176`), so element
  ownership needs nothing extra.
* Only meaningful for `discontinuous=False`; the discontinuous path does not call it.

Both go through `mesh_method(...)` in `src/nanobind/mesh.cpp` next to the existing
`is_mesh_distributed` binding (line 1207), returning numpy int arrays. Remember `./build_for_develop.sh`,
not `ninja`.

Both were built as described. A third C++ change turned out to be necessary on the way: `elem_types`
was uninitialised for most rows as soon as an element is split, see §9.6.

---

## 7. Python structure

### 7.1 New module

`pyoomph/meshes/meshdatamerge.py` — `meshdatacache.py` is already 1545 lines, ~1100 of which are the
extrusion operators. The merge lives in its own module and is imported lazily from the storage, so a
serial run never imports mpi4py through this path.

### 7.2 Entry point

```python
class MeshDataCacheStorage:
    def get_data(self, msh, key: MeshDataCacheKey) -> MeshDataCacheEntry | None:
        if not key.global_mesh:
            ...                      # unchanged
        if not msh.is_mesh_distributed() or get_mpi_nproc() <= 1:
            return self.get_data(msh, replace(key, global_mesh=False))   # same data, no comms
        # collective from here on; every rank stores something under `key`
```

### 7.3 Constructing an entry from arrays

`MeshDataCacheEntry.__init__` (`meshdatacache.py:41-112`) *is* the extraction: it calls `to_numpy`,
manipulates eigen dofs, applies the operator. The merged entry needs the same object with arrays
supplied from outside. Split it:

* `__init__(msh, key)` -> keeps today's behaviour, delegating the extraction to `_extract_from_mesh()`.
* `classmethod from_arrays(msh, key, nodal_values, elem_indices, ...) -> MeshDataCacheEntry` ->
  bypasses the extraction, sets `is_global=True`, fills `nodal_local_exprs` eagerly and marks the lazy
  path closed.

Everything downstream (`get_data`, `get_coordinates`, `get_unit`, `get_default_output_fields`,
`get_interface_line_segments`) then works unchanged on a merged entry.

---

## 8. Tests (as built)

`tests/test_mpi_global_meshdata.py` + `tests/mpi_global_meshdata_worker.py`, marked `slow`, 6 tests,
~9 s. The reference is always a serial in-process run of the same worker. Everything compared is
numbering-independent, because the merged data is ordered rank by rank:

* node and element counts (too many nodes = the partition interface was not sewn together, too few =
  distinct nodes were merged — neither is visible in a plot of a smooth field);
* a digest over the sorted node coordinates, and one over each element's sorted set of node
  coordinates. Exact, since coordinates are copied node positions rather than solver output;
* field statistics (sum, sum of squares, max) with a round-off tolerance, including a local
  expression, since those are what would expose values landing on the wrong nodes;
* interface line segments — the purely topological check: per-rank data yields one segment per
  partition, the merged data must yield the single 17-node line the serial run sees;
* the non-root ranks getting `None` with `--distribute`, and *not* getting `None` without it;
* a repeated request, which must come from the cache without leaving the others in a dead gather.

Covered: 2/3/4 ranks, `discontinuous=True`, `mpirun` without `--distribute`, a bulk mesh and an
interface mesh. Measured agreement on the bulk mesh: identical node/element counts, coordinates equal
to 0.0, fields to 5e-15.

Not yet covered, worth adding: an adaptively refined mesh (hanging nodes, so the tesselation adds
sub-triangles), 3d tets, a triangular mesh, and an eigenvector request.

The distributed plotting path was verified by hand (`mpirun -np 3 ... --distribute` produces one plot
file, drawn from the full 289-node/512-element mesh, all ranks returning); it has no automated test
yet because it needs a matplotlib rendering comparison.

---

## 9. Bugs found while reading this code

Independent of the feature; all in `pyoomph/meshes/meshdatacache.py` unless noted. Worth fixing in the
same series because the merge either depends on them or would inherit them.

**9.1 Halo element filtering is wrong whenever `tesselate_tri` splits elements** (lines 82-91):

```python
for i, (ei, et) in enumerate(zip(self.elem_indices, self.elem_types)):
    if msh.element_pt(i).non_halo_proc_ID() < 0:
```

`i` indexes *output* rows, `element_pt(i)` indexes *mesh elements*. With `tesselate_tri=True` (the
default, and what plotting and `RemeshWhen` use) a quad produces 2 rows, so from the second split
element on, the halo decision is read off the wrong element — and past `nelement()` it would raise.
Fixed by §6.1. Affects every distributed run that plots or checks triangulation validity.

**9.2 `D0_data`/`DL_data` are not filtered along with the elements** (same block). They are indexed by
output element row, so after dropping halo rows from `elem_indices`/`elem_types` the elemental fields
belong to different elements than the connectivity does. Same fix.

**9.3 `merged_eigendata` stores the real part twice** (line 74) — **fixed** in part 1:

```python
"D0_data": (real_D0_data, real_D0_data)     # second should be imag_D0_data
```

`imag_D0_data` is computed and discarded, so the imaginary part of a D0 field is silently the real
part for `eigenmode="merge"`.

**9.4 `clear(only_eigens=True)` reads the key positionally** (line 437) — the reason for part 1;
**fixed**, it now asks `key.depends_on_eigen`. `remkeys: list[str]` was also mistyped (keys are
tuples); fixed.

**9.5 Leftover debug print** in `MeshDataCartesianExtrusion.apply` (line 1085):
`print("DONE HRE", base.nodal_field_inds)` on every call — **removed**.

**9.6 `elem_types` was uninitialised memory for most rows whenever `tesselate_tri` splits elements**
(`Mesh::to_numpy`, `src/mesh.cpp`) — **fixed**. It wrote `elemtypes[ne] = be->get_meshio_type_index()`
with `ne` the *mesh element* index, while the rows are sub-elements. On a 3x3 C2 quad mesh:

```
tesselate_tri=False  rows (9, 9)   unique types [8]
tesselate_tri=True   rows (72, 3)  unique types [3, 0, 8, 1153, 29406, -1134559232, 1012137984, ...]
```

i.e. 9 of 72 rows described, the rest whatever was in the buffer — and those 9 carried the parent
Quad9 type although the row holds 3 triangle nodes. The type now belongs to the row: every row of a
split element gets type 3, since only 2d elements are ever split and always into linear triangles.
Nothing without splitting changes. This is why it went unnoticed: the plotters read `elem_indices`
and ignore `elem_types`, and `MeshFileOutput` defaults to `tesselate_tri=False`.

**9.7 `with_halos=False` kept the halo elements' node blocks in the discontinuous case** — **fixed**.
Each element owns its node rows there, so after the halo element rows were dropped their blocks stayed
behind, unreferenced (576 nodes serial vs 882 on 3 ranks in the test problem).

**9.8 Local expressions ignored the halo filtering** — **fixed** (follows from 9.7). They are
evaluated lazily from the mesh, which always covers all of its node rows, so they were one array
length longer than `nodal_values` and thus attached to the wrong nodes. They now go through
`MeshDataCacheEntry._evaluate_local_expression`, which applies the same row filter.

---

## 10. Decisions and what is left

Settled: the parameter is `global_mesh`; the non-root ranks get `None`; all local expressions are
evaluated eagerly at merge time; both plotters (`MatplotlibPlotter`, `PyVistaPlotter`) use
`global_mesh=True` unconditionally, with no constructor knob — rank 0 draws the whole mesh, the others
serve it. Before this, plotting had no MPI handling at all and every rank drew its own partition into
the same file name.

Left open:

1. **Operators** (`MeshDataCombineWithEigenfunction`, the extrusions) still raise `NotImplementedError`
   together with `global_mesh=True`; see §3.5 for the `required_cache_keys` design that resolves it.
2. ~~**Eigendynamics animations**~~ and ~~**tracer plots**~~ DONE (29th August 2026): both are
   requests now, so every rank takes part -- §3.4c. The one thing a distributed tracer plot still
   cannot show is the fading trail of a DEAD particle: those are deliberately not gathered
   (`dev_docs/tracers.md`), so a particle that has left the domain loses its trail at once instead of
   over the history window.
3. **Remeshing** still works on per-rank data; switching `RemeshWhen` and the boundary identification
   to the merged mesh is the next consumer.
4. **Coordinate fallback** (§5.3) is not implemented: every distributed mesh met so far has the shared
   node scheme. Worth adding only if one turns up that does not.
5. The MPI suites (`tests/test_mpi_*.py`) have not been re-run since the halo-filter change.
