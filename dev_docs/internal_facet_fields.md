# Fields on the interior-facet skeleton (`_internal_facets_`)

Status: **implemented and tested, serial and distributed.** Discontinuous fields (`D0`, `DL`, and the
nodal DG spaces `D1`/`D2`/`D1TB`/`D2TB`) can be declared on the `_internal_facets_` domain of any bulk
mesh (1d/2d/3d, all element shapes including mixed), and `D0`/`DL` survive spatial adaptation,
remeshing and MPI `--distribute` (§7.1). Refused by design: continuous spaces on the skeleton (§2) and
nodal DG under adapt/remesh/distribute (§5.4, §6.3, §7.1). User docs: `AGENTS_ADVANCED.md`
("Unknowns on the facet skeleton") and the tutorial `docs/source/tutorial/dg/facetfields.rst`;
tests: `tests/test_internal_facet_fields.py` and `tests/test_mpi_facet_fields.py`.

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
sketched). Skeleton `D0`/`DL` reuse the interface DL/D0 mechanism: `clear_before_adapt` samples
each element on a local-coordinate lattice (all time levels, Eulerian positions),
`rebuild_after_adapt` locates the cloud on the new skeleton and refits per element
(least squares for DL via `ElementModeFit`, mean for D0).

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

### 5.3 New facets: zero, or recovered

Facets created inside refined bulk elements have no old data. Default: zero, a one-time warning,
and `mesh.get_discontinuous_unrestored_elements()` lists them.
`Equations.set_facet_recovery(field, expr)` registers a local function
`__facet_recovery_<field>` that `restore_discontinuous_data` evaluates on the same lattice and
fits the same way for exactly those elements — the correct HDG answer, since a trace is
determined by the bulk solution anyhow (one solve restores consistency either way).

### 5.4 What stays rejected under adaptation

Nodal DG (`Dx`) skeleton fields: their values live in per-node slots of the internal `Data` and
there is no `get_interpolated_fields_Dx()` to feed the point cloud from; not a small change
(rationale comment at the throw in `rebuild_after_adapt`). Triangle skeletons adapt under
*uniform* refinement (node-based enumeration branch); non-uniform triangle/mixed adaptation still
throws in the 2d enumerator (pre-existing).

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

### 6.3 Guards on this path

`Dx` skeleton fields are refused **at the top of `force_remesh`**: the previous behaviour (throw
from `rebuild_after_adapt` halfway through the remesh) left the Problem with swapped meshes and
undestroyed predecessors, which surfaced as a segfault in the *next* Problem of the same process.
`ProjectionInternalInterpolator` does not L2-project skeleton fields; it re-applies the same pull
transfer after its projection solve (during which interface meshes assemble their *physical*
residuals — only meshes in projection mode assemble the projection ones — so without the re-run
the facet unknowns would be dragged by their own equations).

## 7. Not built, on purpose

| refused / missing | where the guard sits | note |
|---|---|---|
| continuous spaces on the skeleton | `_internal_define_scalar_field` + backstop in `add_interface_dofs` | route sketched in §2 |
| `Dx` under adaptation / remeshing / `--distribute` | `rebuild_after_adapt` / `force_remesh` / `Problem.distribute()` | §5.4, §6.3, §7.1 |
| non-conforming 3d, non-uniform 2d-triangle enumeration | 3d/2d enumerators | `build_facet_adjacency` is the designated basis for both |
| 1d-bulk and 3d remeshing of skeleton fields | untested, no guard | no `LineMesh` remesher exists; 3d transfer should work but has no test |

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
* **Nodal `Dx` facet spaces stay refused**, now in `Problem.distribute()` with a message naming `DL`/`D0`
  as the way out: distributing goes through the adaptation path, which rebuilds every skeleton element,
  and the sample-and-refit snapshot only carries `DL`/`D0` (§5.4).

Combined with static condensation ([static_condensation.md](static_condensation.md) §9) this is what
makes HDG work in parallel: the trace lives on the skeleton and the bulk unknowns are eliminated, at 2
and 4 ranks, reproducing the serial answer.

## 8. Pre-existing bugs fixed along the way

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

## 9. Test map

`tests/test_internal_facet_fields.py` (88 tests): trace projection exact on
quad/tri/line/brick/tet/wedge/pyr/mixed × DL/D1/D2 (+D0); ndof accounting vs
`facet_adjacency_summary()`; 1d hybridised Poisson (exact for linear, 2nd-order for `sin(πx)`);
opposite-side orientation exactness for the 3d quad symmetries; SIP-DG + skeleton field combined;
adapt cycle (survivor exactness without solving, masked-oracle separation of survivors vs new
facets, history levels, refine→unrefine round trip, 2:1 case, recovery hook, re-pinning/assembly);
remeshing (identical remesh machine-exact incl. history, distorted/coarsened/refined with measured
tolerances, recovery hook, unrestored reporting, Dx refusal); all error-message guards.

`tests/test_mpi_facet_fields.py` (18 tests, `--full`) is the distributed gate. Two layers, because a
broken skeleton does not crash: the skeleton MEASURE and the integral of its NORMAL certify the
enumeration and the orientation without reference to the solution (a duplicated facet inflates the
first, a facet enumerated from the other side flips its contribution to the second), and `ndof` against
the serial reference catches a trace numbered once per holder. On top of that: DG with a linear
manufactured solution stays exact distributed (the flux term does not vanish at the exact solution, so a
mis-enumerated facet moves the answer), `DL`/`D0` facet unknowns at 2 and 4 ranks, the same across an
adaptation that really refines, a state file written serially loaded on 2 and 4 ranks, the nodal-`Dx`
refusal, and the replicated (`mpirun` without `--distribute`) mode.
