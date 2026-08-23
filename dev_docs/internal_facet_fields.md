# Fields on the interior-facet skeleton (`_internal_facets_`)

Status: **implemented and tested, serial and distributed.** Discontinuous fields (`D0`, `DL`, and the
nodal DG spaces `D1`/`D2`/`D1TB`/`D2TB`) can be declared on the `_internal_facets_` domain of any bulk
mesh (1d/2d/3d, all element shapes including mixed), and **every one of them** survives spatial
adaptation (§5), remeshing (§6), MPI `--distribute` (§7.1) and a state file (§8). Refused by design:
continuous spaces on the skeleton (§2), and the parent-space constraints a `Dx` facet field inherits
(§5.4). User docs: `AGENTS_ADVANCED.md` ("Unknowns on the facet skeleton") and the tutorial
`docs/source/tutorial/dg/facetfields.rst`; tests: `tests/test_internal_facet_fields.py` and
`tests/test_mpi_facet_fields.py`.

**Correcting the record.** Until this was written, three guards, this document and two tests all said
that a nodal `Dx` facet field could not be carried across a skeleton rebuild because "there is no
`get_interpolated_fields_Dx()` to sample them with". That was false.
`BulkElementBase::get_DG_fields_at_s(space_index, t, s, result)` (`src/elements.cpp`) has always
interpolated exactly those per-node slots — it is what the bulk father→son transfer in
`further_build()` uses. Only the *name* was missing. Carrying a `Dx` facet field is therefore a
parameterisation of the existing machinery, not new numerics: "the DL basis" becomes "the basis of
space *k*" and "the DL block offset" becomes "space *k*'s `internal_offset_new`".

**The idea.** The skeleton mesh existed long before this work: it is a real `InterfaceMesh` over
all interior element-to-element facets, auto-created when equations set
`requires_interior_facet_terms=True`, but it only ever *assembled* residuals (DG jump/penalty
terms) against bulk dofs. This work lets it *own* unknowns — HDG-style traces, mortar
multipliers — declared exactly like any other field:

```python
class Trace(Equations):
    def define_fields(self):
        self.define_scalar_field("lam","DL")
eqs += Trace() @ "_internal_facets_"
eqs.set_facet_recovery("lam", -dot(var("normal"), avg(grad(var("u")))))  # optional, see §5.3
```

---

## 1. Where the dofs live

A facet field is stored in the face element's own internal `oomph::Data`
(`BulkElementBase::allocate_discontinous_fields`, `src/elements.cpp`), one `Data` per DG space
plus the DL/D0 blocks — the same storage every bulk element uses for its element-local fields.
That is what makes the whole feature cheap: the JIT wiring
(`InterfaceElementBase::fill_element_info_interface_part`) and the field indexing
(`FiniteElementCode::index_fields`, the "interface-only field" branch) predate it unchanged.
Consequences worth stating:

- The dofs are **per facet**. Two facets meeting at a vertex do not share anything there — which
  is what a trace/multiplier space wants, and also why a naive 2d mortar (a `DL` multiplier
  enforcing `jump(u)=0` on *every* facet) is rank deficient: facets meeting at a vertex re-impose
  the same continuity. The 1d hybridised Poisson in the tutorial is the well-posed configuration.
- Equation numbering needed no change: the skeleton `InterfaceMesh` was already a global submesh.

## 2. Why continuous spaces are rejected

Interface-*nodal* dofs use oomph's `BoundaryNodeBase::assign_additional_values_with_face_id`
machinery, and only `pyoomph::BoundaryNode`s carry that map. Interior nodes are plain
`pyoomph::Node` and can never be upgraded afterwards (see
[boundary_node_membership.md](boundary_node_membership.md) §4.1); refinement constructs plain
interior nodes as well, so even `all_nodes_as_boundary_nodes` would only cover the never-refined
mesh. `_internal_define_scalar_field` therefore raises for C-spaces on the skeleton, and the
long-dead `if (false && ...)` guard in `InterfaceElementBase::add_interface_dofs` — which had
anticipated exactly this null-deref — is now a real backstop throw. If shared nodal trace spaces
are ever genuinely needed, the route is: `BoundaryNode` construction for interior nodes when the
functable demands it, namespaced interface dof ids (they are keyed on the bare field name per
bulk mesh), and father-element seeding on refinement. Strictly separate work.

## 3. The opposite dummy carries pinned ballast

`Mesh::generate_interface_elements` builds, for every facet, the mesh-registered "+"-side element
*and* an opposite dummy face element from the same JIT code, so the dummy's constructor allocates
the same internal `Data` — but the dummy is in no mesh and never gets global equation numbers.
`set_as_internal_facet_opposite_dummy()` therefore pins all of it (all time levels): pinned Data
(eqn −2) is what the dummy's local-equation path already tolerated, and unpinned it would either
crash numbering or dangle. Reading a skeleton-owned field through the opposite side (`|-`, or
`jump(...,at_facet=True)`) is rejected at code-generation time — a facet field is single-valued,
so the read is meaningless and would otherwise silently evaluate the pinned zeros. `jump`/`avg`
of *bulk* fields resolve through `+`/`-` to the bulk codes and are untouched. On 2:1 hanging
facets several fine "+"-elements share one coarse dummy (`opposite_already_at_index`); with the
Data pinned that sharing is harmless.

## 4. Facet enumeration

- 1d and 2d keep their dedicated `fill_internal_facet_buffers` (the 2d one owns the quadtree
  hanging-node branch that produces the 2:1 facets).
- 3d (`TemplatedMeshBase3d::fill_internal_facet_buffers`, `src/mesh3d.cpp`) is built on the
  generic `build_facet_adjacency()` (vertex-node-set → incidence list): incidence 2 is an interior
  facet, incidence 1 a boundary facet. **Non-conformity is detected via refinement levels and
  hanging nodes, not incidence** — a 2:1 configuration yields incidence 1 everywhere, so an
  incidence check alone would silently drop facets. Conforming meshes only; uniform refinement is
  fine. `build_facet_adjacency` also resolves `copied_node_pt` so a periodic seam becomes one
  interior facet, and it is the designated replacement for the 2d branch's "Mixed meshes here"
  triangle limitation.
- Quad faces of 3d elements (brick faces, wedge sides, pyramid bases) had no opposite-side
  orientation matching at all; `Quad2dFaceOrientation` (8 symmetries of the square, derived from
  the tensor-grid coordinate map) closes that. Tri faces already worked.

## 5. Spatial adaptation

### 5.1 Snapshot and refit

Every adaptation destroys and regenerates all interface elements, so facet data has no home
across it (see [mesh_point_locator.md](mesh_point_locator.md) §4.5, where this design was
sketched). Every skeleton-owned discontinuous field reuses the interface DL/D0 mechanism:
`clear_before_adapt` samples each element on a local-coordinate lattice (all time levels, Eulerian
positions), `rebuild_after_adapt` locates the cloud on the new skeleton and refits per element.

What is sampled is always one scalar per field per point; what differs is the basis the fit
reconstructs coefficients in, and which internal `Data` they are written to:

| space | sampler | fit basis (`ElementModeFit::build`) | written to |
|---|---|---|---|
| `DL` | `get_interpolated_fields_DL` | `shape_at_s_DL` (selector `-1`) | `info_DL.internal_offset_new + fi` |
| `D0` | `get_interpolated_fields_D0` | none — the value is the mean | `info_D0.internal_offset_new + fi` |
| `D1`/`D2`/`D1TB`/`D2TB` | `get_DG_fields_at_s` | `shape_of_space(k)` (selector `k`) | `space_info->internal_offset_new + fi` |

Only the fields the interface declares **itself** are carried (`[numfields - numfields_new,
numfields)` of each space). The ones inherited from the bulk domain are that bulk element's storage,
reached here through external data, and they travel by the bulk's own father→son route.
`DiscontinuousSnapshot` carries a per-space signature (space index, own field count, node count)
alongside `nDL`/`nD0`, so a snapshot taken against a different set of spaces is recognised as
unusable instead of being read with the wrong stride.

### 5.2 Two numbers that are load-bearing, not cosmetic

- **The sample lattice is shrunk by 0.8 towards the element centre.** Facets created *inside* a
  refined element end **on** the surviving facets; samples sitting exactly on element boundaries
  therefore landed on the wrong facet after coarsening and the fit blended neighbours (a constant
  field came back at 5/6 of its value). This also fixes the same latent contamination for
  ordinary boundary interfaces (a shared node is geometrically in two elements and DL/D0 are
  discontinuous there).
- **`max_projection_offset_factor` is 0.02 for the skeleton** (0.5 for boundary interfaces). The
  skeleton is a non-manifold facet soup; a generous projection tolerance grabs samples from
  unrelated nearby facets. Boundary interfaces keep 0.5 because a curved macro-element boundary
  genuinely moves under refinement.
- **The lattice density follows the widest DG basis present**: `NS = 2*nmode_1d + 1` per direction,
  where `nmode_1d` is 3 for `D2`/`D2TB` and 2 otherwise. It was a fixed 5, chosen so that each SON of
  a refined element still receives the two points per direction a `DL` fit needs. A `D2` field has
  three modes per direction, so at `NS = 5` the fit on a son would be underdetermined — and that does
  not fail loudly, it falls back to a constant and drops the quadratic part of every surviving facet.
  `NS = 2*nmode_1d + 1` keeps the per-son count at `nmode_1d` and reproduces the historical 5 for
  `DL`/`D1`. Whole-element counts were never the problem (a triangular facet keeps 15 of 25 lattice
  points against 7 modes for `D2TB`); it is the per-son count that bites, and `ElementModeFit`'s
  fallback counter is what says whether it worked.

### 5.3 New facets: zero, or recovered

Facets created inside refined bulk elements have no old data. Default: zero, a one-time warning,
and `mesh.get_discontinuous_unrestored_elements()` lists them.
`Equations.set_facet_recovery(field, expr)` registers a local function
`__facet_recovery_<field>` that `restore_discontinuous_data` evaluates on the same lattice and
fits the same way for exactly those elements — the correct HDG answer, since a trace is
determined by the bulk solution anyhow (one solve restores consistency either way).

### 5.4 What stays rejected

Nothing about the *transfer* any more; the remaining restrictions are older and unrelated to it.

- Triangle skeletons adapt under *uniform* refinement (node-based enumeration branch); non-uniform
  triangle/mixed adaptation still throws in the 2d enumerator (pre-existing, §4).
- The **parent-domain space constraints** of `pyoomph/generic/codegen.py`: `D2TB` needs a `C2TB`
  bulk, `D2` a `C2TB`/`C2` one, and `D1TB` on a 2d facet of a 3d `C2`/`C1TB` bulk is refused by name,
  because tetrahedra of those spaces have no face bubble node. These bite more often for a facet
  field than one would expect, because — unlike `DL`/`D0` — declaring a `Dx` field *raises the
  skeleton's dominant coordinate space* (`find_dominant_element_space`,
  `pyoomph/expressions/generic.py`).
- Worth knowing rather than refused: on a **1d facet** (2d bulk) `D1TB ≡ D1` and `D2TB ≡ D2`, since a
  line has no bubble node (`nnode_of_space[C1TB] == nnode_of_space[C1] == 2`).

## 6. Remeshing

### 6.1 Pull, not push

The plan was to push the old skeleton's snapshot onto the new facets, filtered by bulk-element
membership. Measured, that leaves a refining remesh half empty — each old sample lands on exactly
one nearest new facet, and new facets through the interior of an old bulk element are nearest to
nothing: 59 of 132 facets empty on a 2× refinement, still 42 after densifying the cloud 6.6×. So
the direction is flipped, which a remesh (unlike an adaptation) allows because the old mesh is
still alive during `force_remesh`: **each new facet pulls** — its own shrunk lattice is evaluated
in the old skeleton (`InterfaceMesh::interpolate_discontinuous_data_from`, feeding the same
`fit_discontinuous_data` as §5), all history levels.

### 6.2 Topological disambiguation

A pulled value counts only if the answering old facet **separates the old bulk element** the
lattice point locates into (widened by one ring of face neighbours, read off the old skeleton
itself); otherwise nearest-facet matching on a branching non-manifold picks wrong facets — this
is the "locate in the old bulk mesh first" fix predicted in
[mesh_point_locator.md](mesh_point_locator.md) §4.5. Facets with no admissible source fall to the
recovery expression, else zero + warning + `get_discontinuous_unrestored_elements()`. Only an
identical remesh is exact; anything else is O(nearest-facet-distance × gradient) — for true
traces, `set_facet_recovery` is the recommended default.

### 6.3 What the pull evaluates, and what it checks

The old values come out of `MeshPointLocator`'s `EvalRequest`, which knows `DL_fields`, `D0_fields`
and `DG_fields` — the last one interpolating the source element's own internal `Data` over
`shape_of_space`, i.e. the same thing `get_DG_fields_at_s` does on the push side.
`LocationSet::evaluate` writes its blocks in one fixed order (continuous, DL, D0, DG) while the fit
reads a snapshot in the internal-`Data` order `[DG][DL][D0]`, so `interpolate_discontinuous_data_from`
rotates each time level's block once instead of giving either side a convention flag. The two meshes'
DG spaces are matched the same way the `DL`/`D0` field names are: same spaces, same own-field counts,
same node counts, or the transfer is refused rather than read positionally through a different layout.

`ProjectionInternalInterpolator` does not L2-project skeleton fields; it re-applies the same pull
transfer after its projection solve (during which interface meshes assemble their *physical*
residuals — only meshes in projection mode assemble the projection ones — so without the re-run
the facet unknowns would be dragged by their own equations).

Historical note: a `Dx` field used to be refused at the top of `force_remesh`, one level up from the
`rebuild_after_adapt` throw that came before it — throwing from the middle of a remesh left the
Problem with swapped meshes and undestroyed predecessors, which surfaced as a segfault in the *next*
Problem of the same process. Both guards are gone; `InterfaceMesh::get_own_nodal_dg_fields()`, which
was their predicate, stays as a query and is still bound to Python.

## 7. Not built, on purpose

| refused / missing | where the guard sits | note |
|---|---|---|
| continuous spaces on the skeleton | `_internal_define_scalar_field` + backstop in `add_interface_dofs` | route sketched in §2 |
| `Dx` whose parent domain cannot carry it | `pyoomph/generic/codegen.py` | pre-existing and unrelated to the transfer; §5.4 |
| non-conforming 3d, non-uniform 2d-triangle enumeration | 3d/2d enumerators | `build_facet_adjacency` is the designated basis for both |
| 1d-bulk and 3d **remeshing** of skeleton fields | untested, no guard | no `LineMesh` remesher exists; 3d transfer should work but has no test. 1d and 3d under `--distribute` *are* covered (`tests/test_mpi_facet_fields.py`) — it is only the remesh transfer that is not |
| a skeleton of a **2d interface**, i.e. a free surface in 3d | `src/mesh.cpp:5988` | see §7.2 |
| a skeleton of an interface under `--distribute` | `src/problem.cpp:4343` | see §7.2 |

### 7.2 The skeleton of an INTERFACE, not of a bulk

Everything above is about the `_internal_facets_` domain of a *bulk* mesh. An interface can carry one
too — `domain/interface/_internal_facets_` — and that is what an upwind DG surfactant needs
(`dev_docs/surfactant_transport.md` §7). It works, in 2d and axisymmetric, and it had no test or
mention anywhere until then:

* `InterfaceMesh::fill_internal_facet_buffers` (`src/mesh.cpp:5975`) pairs the two interface elements
  sharing a vertex node. A closed curve of N elements gives N facets, an open one N−1 — exterior ends
  are correctly excluded.
* The facet element is `InterfaceElementPoint0d` (`oomph::PointIntegral`, J = 1), the same class a
  contact line uses, and `var("normal")` there is the **in-surface conormal**, not the interface
  normal (which stays reachable as `var("normal", domain="..")`).
* The residual transfer to the child has its own path at `pyoomph/meshes/mesh.py:1986`, the interface
  counterpart of the bulk one in `Problem.compile_meshes`.
* No 2:1 bookkeeping is needed and its absence is not a gap: a facet of a curve is a point, and a
  point is shared by exactly two elements whatever their sizes. Verified on a Z2-adapted interface
  with an element-size ratio of 4.
* **A field owned by the interface is not visible on its own facets.** Unlike a bulk DG field, which
  the facet element carries as external data, an interface-owned field lives in that interface
  element's internal data: `var("G")` unrestricted and `jump(G, at_facet=True)` both fail at code
  generation, while `var("G", domain="+")`, `jump(G)` and `avg(G)` work. Facet terms there must be
  written with `+`/`-` restrictions.

**3d is the missing piece.** `be->dim() != 1` throws. The fix is a `dim()==2` branch built on
`TemplatedMeshBase::build_facet_adjacency()` (`src/mesh.cpp:8814`), which is already shape-neutral and
works off `get_possible_face_indices()` / `get_vertex_nodes_of_face()` — both inherited by
`InterfaceElement*` from their `BulkElement*` base — so it is a smaller job than it looks. The facet
elements it would have to produce (`InterfaceElementLine1d*`) already exist. Expect the same
conforming-meshes-only restriction the 3d bulk enumerator carries.

**MPI is a separate piece of work.** `Problem::setup_interior_facet_halo_scheme` refuses any skeleton
whose bulk is an `InterfaceMesh`, and the reason is structural: the key it pairs facets across ranks
with is the bulk element's `(root global_base_index, refinement path, face index)`, and here those
"bulk" elements are face elements built on the fly rather than mesh elements numbered before the
distribution. A partition-independent key would have to be built from the *interface's* own bulk
element first.

### 7.1 MPI `--distribute`

A facet whose two bulk elements end up on different ranks EXISTS on both. The near-side element is
non-halo on one of them and a halo on the other, and `oomph::FiniteElement::build_face_element` stamps
the facet element with that flag — so residual assembly was already right, since every assembly path
skips halo elements. What was missing was the numbering: `Mesh::assign_global_eqn_numbers()` numbers
every element's internal `Data` with no halo test, and nothing marked a halo facet's `Data` as halo. The
symptom was not a crash but an inflated `ndof` — one independent copy of the trace per holder — and an
answer that still looked plausible for a facet field that does not feed back into the bulk. For one that
does (an HDG trace) the far side's coupling was simply never assembled.

`Problem::setup_interior_facet_halo_scheme()` fixes both halves at once. It gives each skeleton its
`Root_halo_element_pt` / `Root_haloed_element_pt` lists and calls `set_halo(owner)` on the halo side's
internal `Data`; everything downstream is then free, because the skeletons are registered as submeshes
and oomph's `copy_haloed_eqn_numbers_helper()` (numbers) and `synchronise_dofs()` (values) are submesh
loops over exactly those lists.

* **The pairing** is by a key both holders compute alone: the near-side element's
  `(root global_base_index, packed refinement path)` and the face index, all partition-independent
  because the near-side rule itself is (`Mesh::compare_structural_order`, used by the three
  `fill_internal_facet_buffers`). One `MPI_Alltoallv` for every skeleton of the problem together, and
  **both sides sort by the key**, so the two lists are index-matched by construction with no second
  round. A key the owner does not recognise means the ranks disagree about which side is the near one,
  which is refused collectively rather than silently mis-pairing equation numbers.
* **Lifecycle.** Built in `actions_after_adapt` (which covers `actions_after_distribute`, adaptation and
  state-file loading) after the whole `rebuild_after_adapt` loop and before `setup_pinning`, and in
  `load_balance`. Deliberately NOT inside `rebuild_after_adapt`: that is a per-mesh loop that can throw
  per rank, and a collective inside it would deadlock the ranks that did not throw. The lists are dropped
  in `InterfaceMesh::clear_before_adapt` before the elements they point at are deleted —
  `flush_element_and_node_storage()` does not touch them, and a stale entry is a dangling read inside
  `copy_haloed_eqn_numbers_helper`.
* **Values.** The build ends with one `synchronise_all_dofs()`. The skeleton has just been rebuilt and
  `restore_discontinuous_data()` refitted each rank's facet values from its OWN partial sample cloud, so
  a halo facet holds this rank's fit rather than the owner's — which a neighbour's non-halo bulk element,
  and any `output()` taken before the first Newton step, would read.
* **Nodal `Dx` facet spaces need nothing of their own.** `Problem.distribute()` used to refuse them,
  because distributing reaches the skeleton through `actions_after_distribute → actions_after_adapt`
  and the sample-and-refit snapshot only carried `DL`/`D0`. It carries every space now (§5.1), and the
  halo scheme marks whole `Data` objects, so it cannot tell how many values each holds.
* **One offset that is easy to get wrong — and is currently dead.**
  `Mesh::share_interpolation_across_ranks()` pools the values of elements a rank could not place
  itself during a distributed *remesh*. It indexed `internal_data_pt(d)` for `d < nDL + nD0`, which
  is only the DL/D0 block while no DG space is present: `allocate_discontinous_fields` lays the
  element out as `[DG][DL][D0]`. It now covers the whole range,
  `dg_internal_data_offset(ft) + nDL + nD0`.

  That loop cannot run today, and the claim that a `Dx` facet field reaches it is wrong.
  `Mesh::nodal_interpolate_from()` throws *"Cannot interpolate DG fields at interfaces yet"* for any
  mesh carrying a DG, DL **or** D0 field before it ever calls the pooling, so `ndisc` is 0 in every
  run — instrumenting a distributed `D1`-skeleton remesh showed only the bulk mesh and the boundary
  interfaces arriving there, none of them with internal data, and the skeleton never taking that
  route at all (it is rebuilt and refilled by `_transfer_internal_facet_fields`,
  `pyoomph/meshes/interpolator.py`). Two consequences worth knowing:
  * the DL/D0 write-back inside `nodal_interpolate_from` had the *same* off-by-`dg_off` indexing and
    has been corrected to match, so that the two agree the day the gate is lifted;
  * the gate is a *narrowing*. Until `41b438f2` it tested the nodal DG spaces alone, so a DL or D0
    field did survive a remesh; widening it to DL/D0 made that a hard error, and nothing noticed
    because no test covered it. `tests/test_mesh_point_locator.py::test_remeshing_refuses_a_
    discontinuous_field` now pins the refusal for `DL`/`D0`/`D1`/`D2`, so restoring it has to be
    deliberate. (Note also that the refusal leaves the `Problem` half-remeshed: tearing it down
    afterwards aborts the interpreter, which is why that test runs in a subprocess.)

Combined with static condensation ([static_condensation.md](static_condensation.md) §9) this is what
makes HDG work in parallel: the trace lives on the skeleton and the bulk unknowns are eliminated, at 2
and 4 ranks, reproducing the serial answer.

## 8. State files

A state file has to reproduce exactly the state it was written at. Until recently it did not, and not
only for `Dx`: **no interface or skeleton element data was written at all.** `Problem._define_state_file`
looped `self._meshdict` only (asserting `not isinstance(mesh, InterfaceMesh)`), and `InterfaceMesh` has
no `_define_state_file` of its own. What happened on load instead: `_load_state` snapshots the *pre-load*
interface values (`clear_before_adapt`), loads the bulk mesh, and lets
`actions_after_adapt → restore_discontinuous_data` refit that pre-load snapshot onto the loaded
geometry. So the facet values after a load were whatever the in-memory problem happened to hold — for a
fresh reader, zeros — and not what the file recorded. `DL`/`D0` were no better off than `Dx`.

* **The record.** `save_interface_state`/`read_interface_state`/`apply_interface_state` in
  `pyoomph/meshes/meshstate.py`, streaming `Mesh::save_elemental_state`/`load_elemental_state` — which
  are on `pyoomph::Mesh` and therefore already worked on an interface mesh unchanged
  (`ninternal_data() × nvalue() × ntstorage` per element, so every discontinuous space is in it by
  construction, history included).
* **The key** is `InterfaceMesh::get_interface_element_structural_keys()`: three longs per element, the
  bulk element's `(root global_base_index, packed refinement path)` plus the `face_index` there. A face
  element has no refinement tree of its own, so it is addressed through the bulk element it hangs off.
  Same key `Problem::setup_interior_facet_halo_scheme` pairs facets across ranks with, which is what
  makes the record partition-independent — a file written serially loads on any number of ranks.
* **Two-phase load.** The interface elements do not exist in their final form until after
  `actions_after_adapt()`, which is well past the point where the bulk block is read. So the block is
  read into `Problem._pending_interface_states` during `_define_state_file` and applied by
  `Problem._apply_interface_states()` immediately after `actions_after_adapt()` — deliberately *after*
  `restore_discontinuous_data` has run, so the file's values overwrite that refit rather than race it.
  An element whose key is not in the file keeps what the rebuild left it (the refit, or the recovery
  expression), which is the right answer for a mesh refined since the file was written. Saving stays
  single-phase. Dump version 0.1.3 → **0.1.4**; older files still load and simply have no such block.
* **The oracle has to be a load that is never solved.** One solve repairs a trace whose residual
  determines it algebraically, which is exactly why the pre-existing
  `test_state_file_saved_serially_loads_distributed` could not have noticed the missing block: its
  workers solve after loading. The tests added with the block load and stop
  (`test_a_state_file_restores_the_facet_field_without_a_solve` serially, `--state load_nosolve`
  distributed).

## 9. Pre-existing bugs fixed along the way

All found because skeleton fields exercise paths nothing else did; none is skeleton-specific:

- `DirichletBC` on a `DL` field only pinned value slot 0 — the BC constrained the mean and let
  the residual choose the gradient modes (bulk DL fields too).
- `Node::is_hanging(i)` reads `Hanging_pt[i+1]` without bounds check;
  `add_required_ext_data` passed a space index instead of a hang index and dereferenced garbage
  as a `HangInfo*` on 2:1 hanging nodes of a rebuilt skeleton.
- IC/Dirichlet setup took `get_nodal_space_index_to_element_index_map()` from element 0 and
  reused it across a mixed-shape mesh (tri + quad face elements in one `InterfaceMesh`).
- Null-`BoundaryNode` derefs in `setup_additional_dof_constraints` and an unwarranted throw in
  `unpin_Dirichlet_dofs_for_matrix_manipulation` for interior (non-boundary) face-element nodes.
- Snapshot samples on element boundaries contaminating the coarsening refit (§5.2, also affected
  boundary interfaces).

## 10. Test map

`tests/test_internal_facet_fields.py` (125 tests): trace projection exact on
quad/tri/line/brick/tet/wedge/pyr/mixed × DL/D1/D2 (+D0); ndof accounting vs
`facet_adjacency_summary()`; 1d hybridised Poisson (exact for linear, 2nd-order for `sin(πx)`);
opposite-side orientation exactness for the 3d quad symmetries; SIP-DG over all 11 3d layouts at
`DG_alpha = 1`, both against the continuous solution and with a linear manufactured one the `D1`
space reproduces exactly (the sharp form: it is what the tetrahedron winding of
[mesh_construction.md](mesh_construction.md) §6 broke); SIP-DG + skeleton field combined;
adapt cycle (survivor exactness without solving, masked-oracle separation of survivors vs new
facets, history levels, refine→unrefine round trip, 2:1 case, recovery hook, re-pinning/assembly);
remeshing (identical remesh machine-exact incl. history, distorted/coarsened/refined with measured
tolerances, recovery hook, unrestored reporting), each over `DL`/`D0`/`D1`/`D2`; state files loaded
without a solve, with the values compared bit for bit and keyed by the elements' structural keys;
all error-message guards.

`tests/test_mpi_facet_fields.py` (69 tests, `--full`) is the distributed gate. Two layers, because a
broken skeleton does not crash: the skeleton MEASURE and the integral of its NORMAL certify the
enumeration and the orientation without reference to the solution (a duplicated facet inflates the
first, a facet enumerated from the other side flips its contribution to the second), and `ndof` against
the serial reference catches a trace numbered once per holder. On top of that: DG with a linear
manufactured solution stays exact distributed (the flux term does not vanish at the exact solution, so a
mis-enumerated facet moves the answer), `DL`/`D0`/`D1`/`D2` facet unknowns at 2 and 4 ranks, `DL` and
`D1` across an adaptation that really refines, a state file written serially loaded on 2 and 4 ranks
(once with a solve afterwards, once without one at all), and the replicated (`mpirun` without
`--distribute`) mode. `tests/test_mpi_remeshing.py` carries a `DL` and a `D1` facet unknown through a
distributed remesh.

All three enumerators are covered, not only the 2d one they were first written for. 1d is the case to
watch, because its near-side rule is the inverted one (`src/mesh1d.cpp` keeps the *later* of the two
elements, 2d/3d the structurally smaller) and because a 1d facet is a point: the measure degenerates to
the plain count of interior facets and the normal integral to a signed count of which way they face,
so both are integers and a single mis-enumerated facet moves them by a whole unit. `∫n·dx = -(N-1)`
distributed, as serially. 3d uses bricks and stays uniform, since `TemplatedMeshBase3d::fill_internal_facet_buffers`
refuses non-conforming meshes; there `∫nᵢ·dx = N-1` in each direction. Every distributed case also
asserts that the run really was partitioned (`is_distributed()`, and that the rank holds halo elements)
- without that a `--distribute` that quietly did nothing agrees with the serial reference perfectly.

Every 3d element FAMILY is covered as well, not only bricks: all 11 layouts of `tests/box_mesh_3d.py`
(pure tet/wedge/pyramid and the seven mixed ones) run distributed at 2 and 4 ranks with a SIP bulk and a
`DL` facet unknown in the same launch. The mixed layouts are not redundant with the pure ones - they are
where the two sides of one facet belong to DIFFERENT families (a pyramid's quadrilateral base against a
brick face, a tet's triangle against a pyramid's), a pairing no pure mesh produces. 2d covers both
triangulations of the quad mesh, `"tri"` (parallel diagonals) and `"tri_crossed"` (a centre node, four
triangles per cell, diagonals in every direction).

This is what turned up the tetrahedron winding bug: every hand-built tet mesh in `tests/` was wound the
opposite way from oomph's `TElement<3,N>` convention, so tet face normals pointed inwards and SIP-DG was
inconsistent on every tet-bearing mesh. `add_tetra_3d_C1/C2` repair it now; the whole story, including
why only the normals - not volumes or stiffness matrices - could notice, is in
[mesh_construction.md](mesh_construction.md) §6.
