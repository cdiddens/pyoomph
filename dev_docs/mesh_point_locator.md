# Mesh point location and mesh-to-mesh transfer

Status: **design agreed, implementation started.** `src/pointlocator.hpp` holds the API and
`src/pointlocator.cpp` the index construction; the matching and evaluation paths still throw. Nothing
calls into it yet, so the old facility is unchanged and in use. The old facility is deleted, not kept
as a fallback, once every call site in §3 has been migrated.

This collects five things that look unrelated and are not:

* zeta coordinates break on a closed interface loop,
* there is no usable zeta for a 2d surface embedded in 3d,
* none of it works under `--distribute`,
* zeta has to be assigned by hand,
* and the *bulk* interpolation has the same two structural defects as the interface one.

All five are the same question asked in different coordinate spaces. §2 is the argument for that
claim; everything after it follows from it.

---

## 1. What zeta actually has to satisfy

`pyoomph/meshes/zeta.py` assigns a boundary coordinate per node. That value is not a label - it is fed
to `Mesh::nodal_interpolate_from(from, boundary_index)` (`src/mesh.cpp:2931`), which wraps the **old**
interface mesh in an `oomph::MeshAsGeomObject` and calls `locate_zeta` with the new node's boundary
coordinate as the global coordinate.

So zeta must be a **globally invertible chart** on the interface mesh: single-valued, monotone within
each element, of the same dimension as the elements. Anything else does not throw. It returns the
wrong element.

Three consequences:

**Closed loops.** The seam element runs from zeta ~ 1 back to zeta = 0, so it spans the entire zeta
range and `locate_zeta` inverts it for *any* query in [0,1]. An arbitrary node can be matched against
it and receive values from the opposite side of the loop - not a local error near the seam, a global
one. Refinement then averages zeta over the generating nodes (`src/elements.cpp:8166`, and the
corresponding sites in `src/refineable_telements.cpp`), so a node created inside the seam element gets
zeta ~ 0.5, which is geometrically meaningless. There is no fix within a single-valued coordinate:
a circle needs at least two charts, or an explicit period (§4.3).

Measured on a `CircularMesh`, the two assigners behaved differently, which is worth recording because
it was not the expectation: `AssignZetaCoordinatesByArclength` already **failed** on a closed loop,
because the segment walk returns the start node again at the end and the resulting length mismatch
trips the check at `zeta.py:163` - but it failed with `NODEMAP AND SEGMENT LENGTH MISMATCH`, which
names neither the loop nor the cause. `AssignZetaCoordinatesByEulerianCoordinate` was the one that
silently produced a broken chart. Phase 0 replaced both with an explanatory error.

**Overhangs.** `AssignZetaCoordinatesByEulerianCoordinate` has exactly the same non-invertibility if
the interface folds back along the chosen axis - a circle parameterised by x is precisely this case,
and is why the guard for it is an overlap test rather than a loop test. Its only previous guard was
the degenerate all-equal case (`zeta.py:106`).

**Surfaces in 3d.** A two-component chart on a 2-manifold exists globally only for disc-like patches,
is arbitrary even then, and has no arclength analogue. This is a topological obstruction, not a gap in
the implementation - see §4.2 for why the answer is to stop needing a chart.

---

## 2. The five call sites are one query

| site | locates | in which space | migrated |
| --- | --- | --- | --- |
| `Mesh::nodal_interpolate_from(from,-1)` | every non-boundary node of the new mesh | Lagrangian xi (see below) | yes |
| the same, `boundary_index >= 0` | interface nodes | boundary zeta | yes |
| the same, element-centre pass | one point per element, for DL/D0 | as above | yes |
| `Mesh::prepare_zeta_interpolation` | every **integration point** of every new element | Lagrangian xi | yes |
| `Mesh::add_interpolated_nodes_at` | sampling points | Lagrangian xi | yes |
| `Mesh::get_values_at_zetas` | - | - | **not a call site** |

`get_values_at_zetas` turned out to be dead: its whole body is commented out and it throws
`"Implement"`. It is listed here only so the next reader does not go looking for it.

"Lagrangian xi" rather than Eulerian x is not a typo, and it is the one subtlety in the migration.
`zeta_coordinate_type` defaults to **0 = Lagrangian** (`elements.cpp:291`), so the bulk locate
normally happens in Lagrangian space, moving to Eulerian only when
`interpolated_lagrangian_coordinates_at_remeshing` is set. That works because in the default case
`prepare_interpolation()` has just copied the old mesh's x into its xi
(`set_lagrangian_nodal_coordinates`), so the two spaces coincide; when the Lagrangian coordinates are
themselves interpolated they are no longer a copy and the location has to use x explicitly.

Every one is *given a target point in some coordinate space, return (source element, local
coordinate)*. They differ in exactly three ways:

1. **which space** - Eulerian, Lagrangian, or boundary zeta;
2. **codimension** - space dimension equal to element dimension (invert `x(s) = x_query`), or one
   greater (overdetermined, minimise `|x(s) - x_query|` instead);
3. **periodicity** of the space.

So zeta is not a mechanism, it is one choice of space. A closed loop is that space being periodic. A
2d surface in 3d is the codimension-1 case, which needs no chart at all. And MPI is one routing layer
that all of them inherit rather than three separate pieces of work. That is the whole design.

## 2.1 What the current fallback does

Anything `locate_zeta` fails to find is not an error. It goes to `missing_nodes` and is handled at
`mesh.cpp:3213` by **two full sweeps over every node of the source mesh** - nearest, then
second-nearest - followed by an inverse-distance blend of those two values. That is
O(N_missing x N_from), and the blend is not an interpolation; it is not even linear-exact on a general
mesh. `Mesh::nodal_interpolate_along_boundary` (`mesh.cpp:2585`) uses the identical two-nearest blend
for boundaries, with a hard-coded absolute cutoff `mindist > 1.0` (`mesh.cpp:2795`) that encodes an
assumption about nondimensional scale rather than a local element size.

Under `--distribute` this is the entire problem in one sentence: a point owned by another rank is not
found, falls into the nearest-node blend **over local nodes only**, and produces plausible wrong
values, with at most a stderr line for the bulk case and nothing at all for boundaries.

Remeshing under `--distribute` is not supported today. It is planned, which is why §5 is designed now
and built last.

---

## 3. API contract

`src/pointlocator.hpp`. Two objects:

`MeshPointLocator` - the index over one source mesh in one `LocatorSetup` (space, time level,
boundary index, period, projection guard). Built once and cached on the mesh. Backed by the
already-vendored nanoflann (`src/thirdparty/nanoflann.hpp`, wrapped in `src/kdtree.hpp`).

`LocationSet` - the result of `locate_batch`, and the handle through which `evaluate(EvalRequest)` is
called. It owns the routing schedule.

**Locate and evaluate are split deliberately.** Locating is expensive and happens once; evaluating is
cheap and repeats. The projection solve evaluates ten times (once per history level) against one set
of locations. Under MPI the split is what makes the cost acceptable: the schedule is derived during
location and reused verbatim, so each evaluation is one `MPI_Alltoallv` of doubles with nothing to
re-derive.

Two rules exist purely so §5 can be added later without touching a call site. They must be honoured
by new code even though nothing is distributed yet:

**Rule 1 - no source element pointers escape.** A located point is a `LocationHandle` (owning rank +
slot). Under distribution the source element may be on another rank, where a pointer means nothing.
The projection solve's `coords_oldmesh` currently stores a raw `BulkElementBase*`
(`src/elements.hpp:594`) - that is the pattern being replaced.

**Rule 2 - everything is requested up front and pulled into a local buffer before use.** The
projection residual currently calls `old_elem->interpolated_zeta(...)` and `interpolated_x(t, old_s,
...)` *inside* element assembly (`src/elements.cpp:5615`). That can never be a remote call. A consumer
that discovers afterwards that it also wants, say, a derivative forces a second round trip, which is
why `EvalRequest` is a bitfield with room for quantities not needed yet.

### 3.1 A latent bug the design forces into the open

`BulkElementBase::zeta_nodal` decides which coordinate it reports from two **static** flags,
`zeta_time_history` and `zeta_coordinate_type` (`src/elements.hpp:647`). `MeshKDTree` sets them and
then resets them to **0** rather than to their previous values (`mesh.cpp:6450`, `mesh.cpp:6494`),
which silently discards the setting `nodal_interpolate_from` installed around itself
(`mesh.cpp:2934`). Harmless today only because the two never nest - which this design deliberately
makes them do. `pointlocator.cpp` scopes them through a save/restore guard; the statics themselves
should become parameters when `MeshKDTree` is retired.

---

## 4. What each of the five problems becomes

### 4.1 Bulk interpolation

Same locator, Eulerian space, `Invert` mode. Gains: it works across ranks (§5), it is fast (§6), and
the two-nearest blend stops being the silent catch-all - unlocated points are counted and reported.

### 4.2 Surfaces in 3d - closest-point projection

`Project` mode. For each new interface node, seed from the tree, then Gauss-Newton on the 2 local
coordinates minimising `|x_old(s) - x_new|` (2 unknowns, 3 residual components), with edge/vertex
clamping for queries outside an element. Parameterisation-free, works for any topology, second-order
accurate. It also works in 1d, so it replaces the two-nearest blend as the general interface path.

Guard: reject a match whose residual offset exceeds `max_projection_offset_factor` times the local
element size. Without it a near-touching interface matches the wrong sheet, which a pure nearest-point
search cannot distinguish.

Zeta is then an *override*, not the mechanism, and remains worth keeping for three cases projection
handles badly: near-touching interfaces, interfaces whose shape changed appreciably between the old
and new mesh, and preserving an arclength distribution across kinks.

### 4.3 Closed loops - periodic zeta

Store a period per boundary (`LocatorSetup::period`, 0 meaning non-periodic). The locator wraps the
query into the principal period and unwraps any drop larger than half a period when reading an
element's zeta range. The seam element then covers `[z_last, z_first + P]` and the map is invertible
on the circle. Needs, in addition:

* a closed-loop mode in `AssignZetaCoordinatesByArclength` with a **continuous** seam anchor - the
  intersection with a ray from the loop centroid, interpolated inside the element, not quantised to a
  node, so the old and new mesh agree on it;
* periodic-aware zeta averaging at refinement (`elements.cpp:8166` and `refineable_telements.cpp`).

With §4.2 in place this is optional for many cases: projection handles a circle with no chart.

### 4.4 Automatic assignment

The requirement is weaker than it looks - not a *good* parameterisation, only one the old and new mesh
agree on. A deterministic geometric rule suffices: orient each segment from its lexicographically
smaller endpoint within a tolerance, sort segments by start point, normalise per segment. Segment
endpoints are domain corners and boundary intersections, which survive remeshing. Ties on symmetric
geometries get a warning, not a coin flip. Closed loops additionally need §4.3's seam anchor.

### 4.5 Facet data and HDG

Internal facets *are* real `FaceElement`s (`InterfaceElementBase`) in an `InterfaceMesh` named
`_internal_facets_`, built from `fill_internal_facet_buffers` and paired with an opposite dummy
element (`mesh.cpp:1312`). So the class could carry internal data. The obstacle is elsewhere, and
splits into three problems worth keeping apart:

**(a) Persistence, not interpolation.** `InterfaceMesh::clear_before_adapt` (`mesh.cpp:5021`)
`delete`s every interface element, and `generate_interface_elements` rebuilds from scratch. This
happens on **every mesh adaptation**, not only on remeshing. Facet-owned data dies there long before
an interpolator sees it. `InterfaceMesh::rebuild_after_adapt` (`mesh.cpp:5279`) already refuses
interface DG spaces with `numfields_new > 0` for exactly this reason - but it checks only DG, so
interface DL/D0 data is silently reset instead (§8, Phase 0 item).

**(b) MPI ownership.** An internal facet between bulk elements on two ranks exists on both. oomph
distributes nodes and elements, not facets, so facet-owned data would need its own owner designation
and halo sync, with no existing scaffolding.

**(c) Interpolation, which for HDG should mostly be avoided.** The trace is not independent data - it
is tied to the bulk solution by the local solves and the global trace equation. So: interpolate the
bulk fields with the machinery here, then **recover the trace** on the new skeleton by evaluating the
new bulk solution there, and let one solve restore consistency. No facet-to-facet mapping, no facet
ownership, (a) and (b) both sidestepped.

If genuinely independent facet history is ever needed, the skeleton is a codim-1 mesh and §4.2 almost
applies - except it is not a manifold, it branches at edges, and closest-point projection is ambiguous
exactly there. The fix is topological, not geometric: locate the query in the old **bulk** mesh first
(a query already being made), then consider only the old facets of that element and its face
neighbours.

---

## 5. MPI cost model

Sizing for the design target - the projection solve on a 3d tet mesh at the 200k-dof ceiling: ~1e6
integration-point queries, d = 3, P = 4.

**Locate, once per remesh.** Rank bounding boxes cost `P x nbox x 6` doubles; use a handful of boxes
per rank from a shallow KD split rather than one AABB per rank, because a coarse box sends a point to
three ranks that each run a full failed local search - wasted compute, not bandwidth. Routing the
queries costs `d+1` doubles per point per candidate rank: ~32 MB aggregate, ~10 MB per rank, **one
`Alltoallv`**. The reply is *not* `(elem, s)`: the located pair stays on its owner, and only a
found/not-found bitmap returns, so the origin knows which points need the residue pass. What the
origin keeps is the **schedule** - per query, the owning rank and the slot index there, plus the
collective's counts and displacements.

**Evaluate, repeated.** Only points whose source element is on another rank travel, and after a decent
partition that is the partition-interface layer - O(N^((d-1)/d)), not O(N). Realistically a few
percent: ~3e4 points x ~12 doubles ~ 3 MB per evaluate, ~30 MB across ten history levels. Negligible.

So bandwidth is a non-issue in both phases. What would be fatal is per-point messaging, or re-deriving
the routing per field and per time level. Hence:

1. the schedule is a first-class object, built once, degenerate-local in serial so there is no serial
   cost to carrying it;
2. `evaluate` batches fields x time levels into one buffer - the current per-field, per-time-index
   loops (`mesh.cpp:3183`) would become one collective each if translated literally;
3. structure-of-arrays buffers, so the send buffer is a slice rather than a gather;
4. the residue pass for genuinely unlocated points is one `Allgatherv`, everyone searches, then
   `Allreduce(MIN, rank)` for a unique owner - bounded, and a large residue is a bounding-box bug to
   be reported, not absorbed.

This also gives bulk DG transfer for free: D0/DL live on elements, so a remote source element's values
ride back in the same reply buffer. That removes the reason for the refusal at `mesh.cpp:2961`.

---

## 6. What makes it fast

The workload that sets the target is the projection solve: ~1e5-1e6 locate queries per remesh. The
current per-query cost is dominated by constants, not asymptotics - a bin-array lookup, a Newton solve
per candidate, several `oomph::Vector<double>` heap allocations, `std::map`/`std::set` lookups, and a
`zeta_in = zeta` defensive copy carrying a `// Why ever... But without I had issues` comment
(`mesh.cpp:6503`) that needs root-causing rather than inheriting.

In rough order of payoff:

1. **Locate once, evaluate many** (§3) - the projection solve already caches `(old_elem, old_s)` per
   integration point and reuses it across all ten history levels, so this amortisation exists and must
   be preserved, just through a handle.
2. **Neighbour walk seeded per element.** Integration points inside one element are clustered: one
   tree query for the first, then walk from the previous match for the rest. Turns ~n_intpt tree
   searches per element into one. `locate_batch`'s `hint_groups` argument carries the grouping; it is
   a pure optimisation and the result does not depend on it.
3. **Affine inversion for straight-sided simplices** - barycentric coordinates from a precomputed
   inverse, exact, no iteration. Given how simplex-dominated these meshes are this is probably the
   largest constant-factor win. Curved geometry keeps Newton, seeded from the affine guess.
4. **Per-element AABB prefilter** before any Newton, and CSR node->element adjacency instead of
   `std::map<Node*, std::set<Element*>>`.
5. **Cache the locator on the mesh.** All five call sites currently build a fresh
   `oomph::MeshAsGeomObject`; `get_lagrangian_kdtree()` already establishes the caching pattern.
6. **Zero per-point allocation.**

`MeshKDTree::find_element` additionally falls back to `radius_search(max_search_radius)` where
`max_search_radius` is 10x the largest node-pair distance **anywhere in the mesh** (`mesh.cpp:6449`),
so one coarse element inflates the fallback radius globally and the search degenerates to O(N).
Replace with expanding k-nearest.

No OpenMP. The queries are independent and would thread trivially, but pyoomph's own sources contain
none today and the six items above should make it unnecessary.

---

## 7. The projection solve

`ProjectionInternalInterpolator` (`pyoomph/meshes/interpolator.py:49`) is **scaffolding, not a working
path**: it is referenced nowhere (the default is `InternalInterpolator`, `interpolator.py:153`), and
its `interpolate()` constructs a *fresh empty* `Problem()` and calls `steady_newton_solve()` on it ten
times (`interpolator.py:58`), which cannot be projecting the actual meshes. The C++ half is real -
`prepare_zeta_interpolation` (`elements.cpp:5487`) fills `coords_oldmesh` and the projection residual
(`elements.cpp:5589`) consumes it behind `enable_zeta_projection` - but the driver was never finished.
So "move to a projection-based solve" is *finish it*, and it should be finished on top of the locator
rather than before it.

Three things it must not inherit from the physical problem.

### 7.1 A separate linear solver

pyoomph's linear solver is a per-Problem slot (`_lasolver`, `problem.py:1415`) and eigen solvers
already occupy a second slot (`_eigensolver`), so a third is not a new concept.

The projection system is structurally unrelated to the physical one: a mass matrix, SPD, with a
different sparsity and a different field layout. A PETSc `fieldsplit` configured for a Navier-Stokes
saddle point is at best wasted on it and at worst wrong, because the index sets are built from the
physical problem's dof ordering, which the projection problem does not share. The same goes for a
user-supplied nullspace, a custom PC, or MUMPS options tuned for the real system. This codebase
already has evidence that forcing one solver across structurally different systems breaks things -
`petsc_mumps` collapses `hopf_switch`'s arclength continuation and plain iterative `petsc` fails on
the augmented systems.

So: `Problem.set_projection_linear_solver` / `get_projection_linear_solver`, **defaulting to a robust
direct solver rather than inheriting**. Inheriting is the trap - it silently drags a fieldsplit
configuration into a system it does not fit. The projection solve saves and restores the active solver
around itself so a failure does not leave the Problem holding the wrong one.

### 7.2 The frozen sparsity must be off

`acquire_frozen_sparsity` keys its cache on `(matrix_index, generation, ndof)` where `generation` is
`get_jacobian_structure_id()` (`problem.cpp:3381`). `invalidate_jacobian_structure()` is driven by
`assign_eqn_numbers()`, so a change in *pinning* is caught. A change in **residual** is not: the
projection swaps the element residual behind `enable_zeta_projection` (`elements.cpp:5263`) while the
dof structure is untouched, so the cache would hand back the *physical* problem's pattern for a mass
matrix. Where the projection pattern is a strict subset that only wastes explicit zeros; where it is
not, entries are dropped and the answer is silently wrong.

The cache is already keyed by `matrix_index` for exactly this kind of situation (see the comment about
multi-residual problems at `problem.cpp:2202`), so a distinct index is the principled fix. Disabling
frozen sparsity for the duration via `set_use_frozen_sparsity(false)` and restoring afterwards is the
cheap and safe one, and is what should be done first.

### 7.3 One factorisation, many right-hand sides

The projection residual is **linear** in the unknowns, so its "Newton solve" is a single linear solve
whose exact Jacobian is a mass matrix. That matrix:

* does not depend on which field is being projected,
* does not depend on the history level,
* is identical for all fields sharing a function space,
* is SPD.

The current driver solves once per history level (`interpolator.py:60`) - ten assemblies and ten
factorisations of a matrix that never changes. That is a larger waste than the per-field question.

The right structure is to **group by function space** (C1, C2, C1TB, C2TB, and the position
components), and per group assemble once, factorise once, and solve `nfields_in_group x nlevels`
right-hand sides against that one factorisation. Splitting per space rather than solving one coupled
system matters because the coupled system is block diagonal with identical blocks, which a generic
sparse LU does not exploit: the fill-in and factorisation cost are those of an `nfields x N` system
instead of an `N` one, and that grows superlinearly. The per-space matrix is also SPD, which the
coupled system including position dofs need not be, so Cholesky or CG+AMG become available.

Two riders. Restricting to one group means pinning the rest and re-running `assign_eqn_numbers()` per
group - O(ndof) each, cheap next to a factorisation, and it invalidates the frozen sparsity cache,
which §7.2 wants anyway. And D0/DL need no global solve at all: they are element-local, so their
projection is a per-element mass matrix inverted in place, never touching the linear solver.

Whether the backends expose reuse of a factorisation needs checking - oomph's `LinearSolver::resolve()`
exists but no pyoomph plumbing for it was found. Without it the fallback is one factorisation per
group, which is still far better than one per history level.

---

## 8. Adaptation, and where DG/DL/D0 stand

Spatial adaptation is the same transfer problem with more structure available, and it is currently
covered for the bulk and not at all for interfaces.

**Bulk, refinement:** handled. `BulkElementBase::further_build` (`elements.cpp:7092`) samples the
father's DG fields at each son's nodal local coordinate and reconstructs DL from the father's value
and slope per son type.

**Bulk, unrefinement:** handled. `BulkElementBase::rebuild_from_sons` (`elements.cpp:7257`) averages
the sons' DG values at coincident points and the sons' DL/D0 values. The code documents its own
caveats - `// XXX TODO: ... this does not conserve ... and does not consider axisymmetry` - and the
DL/D0 branches are written per tree type, so whether the simplex refinement path in
`refineable_telements.cpp` is covered needs checking rather than assuming.

**Interfaces and facets: not covered, and confirmed by repro.** `clear_before_adapt` deletes every
interface element, so any interface-owned internal data is destroyed on every adaptation.
`rebuild_after_adapt` guards this for DG spaces by refusing outright, but the guard checks only DG -
and its DG test uses `numfields_new` while `allocate_discontinous_fields` (`elements.cpp:8537`)
allocates DL and D0 by the full `numfields`, so nothing covers them. A D0 field pinned to 7 on an
interface reads back as 0 after `refine_uniformly`, with no error. Phase 0 added a one-time warning
naming the interface and the field count; it warns rather than throws because the common case - a
DL/D0 field its own residual determines algebraically - does recover at the next solve, and only
history-carrying fields are actually wrong. Proper transfer is Phase 3b.

The fix reuses §4.2 rather than adding a mechanism: take a **snapshot** of the discontinuous interface
data before `clear_before_adapt` (evaluation points in the coordinate space plus their values - small,
and serialisable, which is also what a distributed version would need), then restore it after
`rebuild_after_adapt` by locating those points on the new interface mesh and evaluating. The same
snapshot object serves remeshing, so adaptation and remeshing stop being two code paths.

---

## 9. Phases

Ordering rationale: Phase 1 first because everything else sits on it and it is a pure refactor with an
A/B switch. Phase 2 next because it delivers the 3d capability and reduces how much zeta has to carry.
3, 4 and 4b are then small. 5 is largest and last, and is gated on `--distribute` remeshing existing
at all rather than on anything here.

**Phase 0 - diagnostics and guards. DONE.**

* `_find_closed_segments` (`zeta.py`) marks closed loops, and `AssignZetaCoordinatesByArclength`
  refuses them with an explanatory message instead of the length mismatch.
* `_check_zeta_is_invertible` validates both assigners after every assignment. It is a *tiling*
  test - a valid chart's element intervals abut without overlapping, so their widths sum to the
  total span; a seam element or a fold-back re-covers ground and the sum exceeds it. This was
  chosen over "one element is suspiciously wide", which false-positives on a coarse interface, and
  it never trips on disconnected segments because their jump offsets are gaps no element covers.
  Off via `validate_zetas = False`.
* `_refuse_if_distributed` on both assigners.
* `nodal_interpolate_along_boundary` reports how many nodes were matched further than two source
  element lengths away, and `nodal_interpolate_from` reports how many fell through to the blend at
  all - the boundary case used to be entirely silent, since its `cerr` was guarded by
  `boundary_index < 0`.
* The `mindist > 1.0` cutoff is now two old-boundary element lengths. The literal was an absolute
  length in nondimensional units: it never fires on a domain of size 0.01 and always fires on one
  of size 100.
* Interface DL/D0 adaptation reset: confirmed (§8) and warned about.

Verified: both zeta tutorials (`beads_on_string`, `rayleigh_plateau`) still run under `--quick-test`,
`tests/test_mixed_mesh.py` passes, and open boundaries are unaffected by the new guards.

**Phase 1 - locator core, serial. Mostly done.**

All the call sites of §2 are migrated, behind `Mesh.set_use_point_locator()`. Measured on a ~9000-node
axisymmetric triangular remesh, interleaved A,B,A,B, with results **bit-identical** between the two
backends (largest difference 0.000e+00):

| | MeshAsGeomObject | MeshPointLocator |
| --- | --- | --- |
| index build | 52 50 55 57 ms | 15 17 18 18 ms |
| location | 128 156 132 132 ms | 8.6 9.1 8.3 9.5 ms |

The location speedup is ~15x and comes almost entirely from the affine simplex inversion; 4536 of
4615 elements qualify, the 79 exceptions being the curved-boundary elements along the arc. Before
that path existed the same benchmark read 76 ms, i.e. the k-d tree and the single-start Newton alone
bought ~1.7x and the affine inversion the remaining ~9x.

### Which geometries avoid Newton

An element is classified once, at index-build time, into one of three kinds. The classification is
driven entirely by testing **every** node against a predicted position, never by assuming an element
family behaves a certain way, which is what makes it work uniformly across orders and shapes.

| geometry | kind | why |
| --- | --- | --- |
| straight-sided simplex, any order | Affine | a T6 with mid-edge nodes at the midpoints has its quadratic terms cancel identically |
| parallelogram / parallelepiped quad or hex, any order | Affine | the bilinear cross term a3 = (X00 - X10 - X01 + X11)/4 vanishes exactly for a parallelogram - *not* only for a rectangle, so sheared structured meshes qualify too |
| straight-edged 2d quad, non-parallelogram | Bilinear2d | a bilinear map sends the reference square's edges to straight segments, and eliminating one variable leaves a **quadratic** - still a closed form, no iteration |
| curved anything, distorted 3d hex | General | Newton, seeded from the element's best affine fit |

3d is the genuine exception: a trilinear inverse has no closed form and hex faces are ruled surfaces
rather than planes, so a distorted hex has to iterate. Seeding from the affine fit rather than the
element centre is what makes that iteration cheap.

Self-locating a mesh's own integration points, per point:

| | classification | per point |
| --- | --- | --- |
| triangles (T6) | 144/144 affine | 0.079 us |
| quads (Q9), rectangular | 36/36 affine | 0.093 us |
| quads (Q9), trapezoidal | 36 bilinear | 0.117 us |
| hexes (Q27), undistorted | 125/125 affine | 0.079 us |
| hexes (Q27), sheared | 125/125 affine | 0.079 us |
| hexes (Q27), trilinearly distorted | 125 newton | 10.6 us |

Before this classification existed, quads cost 2.3 us/point - so it is a ~25x saving on structured
meshes, bringing them level with simplices.

### Accuracy: the new path is not just faster

Results are bit-identical to `MeshAsGeomObject` wherever the geometry is straight, but on **curved**
elements they now differ, and the new values are the correct ones. oomph's locate_zeta stops at
`Locate_zeta_helpers::Newton_tolerance = 1e-7` on the residual (`elements.cc:1654`), so the local
coordinate it returns was only ever good to about that. Measured on the curved circular mesh:

| test | MeshAsGeomObject | MeshPointLocator |
| --- | --- | --- |
| interpolate at a node, vs that node's own value | 2.1e-08 | 9.2e-17 |
| interpolate a linear field at interior points, vs exact | **9.0e-03** | 1.3e-15 |

The second row is the striking one: a linear field is reproduced exactly by an isoparametric FE
space, so any deviation is pure location error, and the old path was ~1% wrong at arbitrary interior
points of curved elements. The locator polishes the local coordinate to machine precision after the
Newton path (`polish_local_coordinate`), which both fixes that and removes the dependence of the
answer on the starting guess - without it, merely changing the Newton seed moved interpolated values
by ~1e-8.

**Still open in this phase:**

* A distorted 3d hex is two orders of magnitude slower than every other case. If that ever matters,
  the lever is a better seed still, not a closed form - there is none.

**Tracers are explicitly out of scope.** `MeshKDTree` (`mesh.cpp:6395`) stays where it is for now,
with the static-flag reset of §3.1 still in it, and `tracers.cpp` keeps using it. Porting tracers onto
the locator is a **separate campaign for later**, not a loose end of this one.

That includes the consequence: **if refactoring here breaks the tracers, that is accepted.** Do not
hold back a change to the locator, to `MeshKDTree`, or to the zeta statics in order to keep
`tracers.cpp` working, and do not bolt compatibility shims onto the locator for its benefit. The
tracer campaign will pick up whatever state it finds and fix it there. What *is* worth doing is
leaving a note in this file when a change is known to have broken them, so that campaign starts from
a list rather than from a bisect.

Known to be affected so far: nothing yet - `MeshKDTree` is still untouched.
* `add_interpolated_nodes_at` has no in-tree caller and `prepare_zeta_interpolation`'s only caller is
  the unfinished projection interpolator (§7), so neither is covered by the test suite. Both were
  checked directly instead: the former against the other backend point by point, including a point
  outside the mesh; the latter by self-location, where every integration point must be found in the
  element it came from (0 unlocated, both element types).

*Verified:* `tests/test_adaptive_interface_coupling.py`, `test_curved_boundaries.py`,
`test_triangle_refinement.py`, `test_mixed_mesh.py` - 264 passed, 3 xfailed; both zeta tutorials under
`--quick-test`.

**Phase 2 - closest-point projection.** §4.2, 1d and 2d, with the offset guard; primary interface
path; two-nearest blend demoted to last resort with a warning; also the honest bulk fallback for
genuinely-outside points. Extended with the bulk-restricted candidate rule of §4.5 so the skeleton
case works later. *Acceptance:* 3d surface remesh with an analytic surface field, error converging
under refinement.

**Phase 3 - periodic zeta.** §4.3. *Acceptance:* closed-loop remesh with a `sin(2 theta)` surface
field at interpolation order rather than O(1).

**Phase 3b - discontinuous data across adaptation.** §8 - the snapshot/restore, covering interface
DG/DL/D0 for both adaptation and remeshing.

**Phase 4 - automatic zeta assignment.** §4.4.

**Phase 4b - finish the projection solve.** §7 - a real problem instead of the empty `Problem()`, the
locator through `LocationSet`, values pulled into a per-(element, ipt) buffer, the separate projection
linear solver, frozen sparsity off, per-space grouping with one factorisation.

**Phase 5 - MPI routing.** §5. Additive if Phases 1 and 4b honour rules 1 and 2. Testing at 4 ranks
maximum, small problems.

**Phase 6 - facet/HDG data.** Sequenced as: trace recovery from the bulk first (§4.5c, needs only
Phases 1-2 and unblocks HDG remeshing); then facet-data persistence keyed by a partition-independent
facet identity (`mesh.hpp:207` has the ingredients) if genuinely independent facet state is needed;
then facet ownership and halo sync, which follows Phase 5.

**Tracers - a separate campaign, deliberately not scheduled here.** Port `tracers.cpp` off
`MeshKDTree` onto `MeshPointLocator`, then delete `MeshKDTree` and turn the `zeta_*` statics into
parameters. It is listed last because nothing else depends on it and because the phases above are
allowed to break it in passing (see phase 1). Whoever picks it up should expect to repair as well as
port.

---

## 10. What gets deleted

At the end, and not before every call site is migrated:

* the five `oomph::MeshAsGeomObject` uses of §2 and the `#include "mesh_as_geometric_object.h"` in
  `elements.hpp`;
* `MeshKDTree` (`mesh.cpp:6395`) - but only in the later tracer campaign, not here; it is its only
  remaining user, and until then it stays as it is even if this work breaks it (see phase 1);
* the two-nearest blend in `nodal_interpolate_from`'s `missing_nodes` pass (`mesh.cpp:3213`) and in
  `nodal_interpolate_along_boundary` (`mesh.cpp:2585`), which becomes a reported last resort rather
  than the silent default;
* `BulkElementBase::zeta_time_history` / `zeta_coordinate_type` as statics (§3.1);
* `KNNInterpolator` (`interpolator.py:159`), already dead behind `if False:`.
