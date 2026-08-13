# Mesh point location and mesh-to-mesh transfer

Status: **in use by default.** `MeshPointLocator` (`src/pointlocator.{hpp,cpp}`) has replaced
`oomph::MeshAsGeomObject` at every real call site; interfaces without a zeta transfer by closest-point
projection; closed loops have a periodic zeta; and the projection-based interpolator works and
conserves. Open: automatic zeta assignment (§4.4, now of doubtful value), MPI (§5, gated on
`--distribute` remeshing), and MPI facet ownership (§4.5(b); facet fields themselves are done,
see [internal_facet_fields.md](internal_facet_fields.md)).

This collects five things that look unrelated and are not: zeta coordinates break on a closed interface
loop; there is no usable zeta for a 2d surface embedded in 3d; none of it works under `--distribute`;
zeta has to be assigned by hand; and the *bulk* interpolation has the same two structural defects as
the interface one. **All five are the same question asked in different coordinate spaces** — §2 is the
argument, everything after it follows.

References here name files and symbols but deliberately not line numbers: the first draft carried them
and most had rotted before the campaign was over, since the campaign edited the very files it was
describing.

---

## 1. What zeta actually has to satisfy

`pyoomph/meshes/zeta.py` assigns a boundary coordinate per node. That value is not a label — it is fed
to `Mesh::nodal_interpolate_from(from, boundary_index)`, which locates the new node's boundary
coordinate in the old interface mesh. So zeta must be a **globally invertible chart**: single-valued,
monotone within each element, of the same dimension as the elements. Anything else does not throw. It
returns the wrong element.

**Closed loops.** The seam element runs from zeta ≈ 1 back to zeta = 0, so it spans the entire zeta
range and the inversion succeeds for *any* query in [0,1]. An arbitrary node can be matched against it
and receive values from the opposite side of the loop — not a local error near the seam, a global one.
Refinement then averages zeta over the generating nodes, so a node created inside the seam element gets
zeta ≈ 0.5, which is geometrically meaningless. There is no fix within a single-valued coordinate: a
circle needs at least two charts, or an explicit period (§4.3).

Measured on a `CircularMesh`, the two assigners behaved differently, which was not the expectation:
`AssignZetaCoordinatesByArclength` already **failed** on a closed loop, because the segment walk
returns the start node again at the end and the length mismatch trips a check — but it failed with
`NODEMAP AND SEGMENT LENGTH MISMATCH`, which names neither the loop nor the cause.
`AssignZetaCoordinatesByEulerianCoordinate` was the one that silently produced a broken chart. Both now
refuse with an explanatory error.

**Overhangs.** `AssignZetaCoordinatesByEulerianCoordinate` has exactly the same non-invertibility if
the interface folds back along the chosen axis — a circle parameterised by x is precisely this case,
and is why the guard for it is an overlap test rather than a loop test. Its only previous guard was the
degenerate all-equal case.

**Surfaces in 3d.** A two-component chart on a 2-manifold exists globally only for disc-like patches,
is arbitrary even then, and has no arclength analogue. A topological obstruction, not a gap in the
implementation — §4.2 is why the answer is to stop needing a chart.

---

## 2. The five call sites are one query

| site | locates | in which space |
| --- | --- | --- |
| `Mesh::nodal_interpolate_from(from,-1)` | every non-boundary node of the new mesh | Lagrangian xi (see below) |
| the same, `boundary_index >= 0` | interface nodes | boundary zeta |
| the same, element-centre pass | one point per element, for DL/D0 | as above |
| `Mesh::prepare_zeta_interpolation` | every **integration point** of every new element | Eulerian x (§9) |
| `Mesh::add_interpolated_nodes_at` | sampling points | Lagrangian xi |

(`Mesh::get_values_at_zetas` turned out to be dead — its whole body is commented out and it throws
`"Implement"`. Listed only so the next reader does not go looking for it.)

"Lagrangian xi" rather than Eulerian x is not a typo, and it is the one subtlety in the migration.
`zeta_coordinate_type` defaults to **0 = Lagrangian**, so the bulk locate normally happens in
Lagrangian space, moving to Eulerian only when `interpolated_lagrangian_coordinates_at_remeshing` is
set. That works because in the default case `prepare_interpolation()` has just copied the old mesh's x
into its xi, so the two spaces coincide; when the Lagrangian coordinates are themselves interpolated
they are no longer a copy and the location has to use x explicitly.

Every call site is *given a target point in some coordinate space, return (source element, local
coordinate)*. They differ in exactly three ways: **which space** (Eulerian, Lagrangian, boundary zeta);
**codimension** (space dimension equal to element dimension → invert `x(s) = x_query`, or one greater →
overdetermined, minimise `|x(s) − x_query|`); and **periodicity**.

So zeta is not a mechanism, it is one choice of space. A closed loop is that space being periodic. A 2d
surface in 3d is the codimension-1 case, which needs no chart at all. And MPI is one routing layer that
all of them inherit rather than three separate pieces of work. **That is the whole design.**

### 2.1 What the fallback does, and why it is the whole MPI problem in one sentence

Anything the location fails to find is not an error. It goes to `missing_nodes` and is handled by **two
full sweeps over every node of the source mesh** — nearest, then second-nearest — followed by an
inverse-distance blend. That is O(N_missing × N_from), and the blend is not an interpolation; it is not
even linear-exact on a general mesh. `nodal_interpolate_along_boundary` used the identical two-nearest
blend with a hard-coded absolute cutoff `mindist > 1.0` that encoded an assumption about
nondimensional scale rather than a local element size (it never fires on a domain of size 0.01 and
always fires on one of size 100; it is now two old-boundary element lengths).

Under `--distribute`: a point owned by another rank is not found, falls into the nearest-node blend
**over local nodes only**, and produces plausible wrong values, with at most a stderr line for the bulk
case and nothing at all for boundaries.

---

## 3. API contract

`MeshPointLocator` is the index over one source mesh in one `LocatorSetup` (space, time level, boundary
index, period, projection guard), built once and cached on the mesh, backed by the already-vendored
nanoflann. `LocationSet` is the result of `locate_batch` and the handle through which
`evaluate(EvalRequest)` is called; it owns the routing schedule.

`evaluate`/`values_per_point` handle continuous, DL, D0 and DG fields plus position, Lagrangian
coordinates and zeta, at any list of time levels. **Layout per point is fixed** — time levels
outermost, then those blocks in that order — rather than "whatever the request listed first", so a
consumer can index the result without re-reading the request it sent. `field_map` maps the caller's own
continuous field indices onto the source's; a `-1` entry leaves that slot at zero.

Two details worth knowing. A `zeta` request re-enters `ZetaFlagGuard` with the *locator's* setup, so the
coordinates come back in the space the points were located in — without that, a search run in Eulerian
space answers with Lagrangian coordinates. And the DG block reports only `numfields_new` per present
space: those are the element's own internal Data, whereas anything inherited from the bulk is external
data and not the source element's to report.

**Locate and evaluate are split deliberately.** Locating is expensive and happens once; evaluating is
cheap and repeats — the projection solve evaluates ten times (once per history level) against one set of
locations. Under MPI the split is what makes the cost acceptable: the schedule is derived during
location and reused verbatim, so each evaluation is one `MPI_Alltoallv` of doubles with nothing to
re-derive.

**Two rules exist purely so §5 can be added later without touching a call site.** They must be honoured
by new code even though nothing is distributed yet:

1. **No source element pointers escape.** A located point is a `LocationHandle` (owning rank + slot).
   Under distribution the source element may be on another rank, where a pointer means nothing. The
   projection solve's `coords_oldmesh` still stores a raw `BulkElementBase*` — that is the pattern being
   replaced, and it is what MPI will have to route through `LocationSet`.
2. **Everything is requested up front and pulled into a local buffer before use.** The projection
   residual calls `old_elem->interpolated_zeta(...)` and `interpolated_x(t, old_s, ...)` *inside*
   element assembly. That can never be a remote call. A consumer that discovers afterwards that it also
   wants, say, a derivative forces a second round trip, which is why `EvalRequest` is a bitfield with
   room for quantities not needed yet.

### 3.1 A latent bug the design forces into the open

`BulkElementBase::zeta_nodal` decides which coordinate it reports from two **static** flags,
`zeta_time_history` and `zeta_coordinate_type`. `MeshKDTree` set them and then reset them to **0**
rather than to their previous values, silently discarding the setting `nodal_interpolate_from` installed
around itself. Harmless only because the two never nested — which this design deliberately makes them
do. `pointlocator.cpp` scopes them through a save/restore guard. The statics themselves are kept, and
that is not laziness: `interpolated_zeta` is an oomph virtual with no room for a parameter, so these are
how it is told which space and which time level to answer in. Removing them means giving pyoomph's
elements their own non-virtual zeta accessor and rerouting some twenty call sites onto it — a design
change wanting its own campaign.

---

## 4. What each of the five problems becomes

### 4.1 Bulk interpolation

Same locator, Eulerian space, `Invert` mode. Gains: it works across ranks (§5), it is fast (§6), and the
two-nearest blend stops being the silent catch-all — unlocated points are counted and reported.

### 4.2 Surfaces in 3d — closest-point projection

`Project` mode. For each new interface node, seed from the tree, then Gauss-Newton on the local
coordinates minimising `|x_old(s) − x_new|`, with edge/vertex clamping for queries outside an element.
Parameterisation-free, works for any topology, second-order accurate. It also works in 1d, so it
replaces the two-nearest blend as the general interface path.

A flat element needs no iteration at all — the least-squares solve through the pseudo-inverse
`(DᵀD)⁻¹Dᵀ` *is* the exact closest point. Matches beyond `max_projection_offset_factor × element size`
are rejected; without that guard a near-touching interface matches the wrong sheet, which a pure
nearest-point search cannot distinguish.

Zeta is then an *override*, not the mechanism, and remains worth keeping for three cases projection
handles badly: near-touching interfaces, interfaces whose shape changed appreciably between the old and
new mesh, and preserving an arclength distribution across kinks.

Verified by pushing element centroids off the surface along its normal — 16/16 found at offsets 0, +0.01
and −0.03 with the offset reproduced exactly, 0/16 at +0.5 (rejected), for both a 2d surface in 3d and a
1d curve in 2d. Three latent bugs had to be fixed for it:

* **Coordinates are now read from the nodes, not through `zeta_nodal`.** On a bulk element `zeta_nodal`
  returns exactly xi or x depending on the static flags, but `InterfaceElementBase` overrides it to
  `FaceElement::zeta_nodal`, which returns the intrinsic *boundary* coordinate — fewer components than
  the nodal dimension, so asking for component 2 of a face in 3d reads out of range and segfaults.
  Reading `node_pt(n)->x(d)` / `->xi(d)` is correct on both.
* **The bounding-box slack was a fraction of a degenerate box.** A codimension-1 element's box is flat
  in the normal direction, so a slack proportional to its diagonal rejected every off-surface query —
  exactly the queries projection exists for. In Project mode the slack is now at least the distance the
  offset guard would accept.
* **The candidate scan stopped at the first acceptable element.** Correct when inverting — a hit is
  exact — but wrong when projecting, where the answer is the *nearest* element and several are
  acceptable. An on-surface query matched a neighbour whose closest point was half an element away. The
  scan now ranks by offset and keeps the minimum.

`nodal_interpolate_from(from, boundary_index, use_boundary_coordinate)` chooses between the two on its
last argument, and `InternalInterpolator` takes the projection branch whenever no zeta is defined
(`project_on_boundary_without_zeta`, on by default). The improvement over the blend it replaces is not
marginal — transferring an analytic interface field across a remesh of a curved 2d interface:

| boundary path | worst \|transferred − exact\| | mean |
| --- | --- | --- |
| nearest-node blend | 2.100e+00 | 9.630e-01 |
| projection | 1.464e-11 | 2.094e-12 |

The field ranges over about [0.4, 2.1], so the blend was not merely inaccurate — it was losing the
interface field almost entirely. A 2d interface in 3d transfers at 2.2e-16 over 49 nodes.

> **The A/B switch could not express this**, which is worth knowing about any comparison against the old
> backend: asking `MeshAsGeomObject` to locate a 2-component position among 1d face elements walks off
> the end of the coordinate vector and segfaults. Note also that a droplet remesh is a weak test of the
> boundary path — it reproduces the same interface node positions, so the blend and the projection agree
> trivially. The transfer test above is the one that discriminates.

**Still open here:** the offset guard is a fixed fraction of the element size and has not been tuned
against a near-touching interface, which is the case it exists for.

### 4.3 Closed loops — periodic zeta

A per-boundary period (`Mesh::set_boundary_zeta_period`) plumbed into `LocatorSetup`. The locator reads
each element's node coordinates unwrapped onto that element's own branch, so the seam element runs from
`z_last` to `z_first + P` and is monotone like any other, and shifts each query onto the same branch
before testing it. `AssignZetaCoordinatesByArclength` gained a closed-loop mode measuring arclength from
a **continuous** geometric seam — the outermost intersection of the loop with a ray from its centroid,
so two discretisations of the same curve agree on it to O(h²) rather than to one element — with
orientation fixed counter-clockwise by the polygon's signed area.

Two things worth recording from building it:

* **`get_interface_line_segments` must not be used to order a closed loop.** Its walk is written around
  open curves — it looks for a degree-one endpoint and falls back to an arbitrary node when there is
  none — and on a loop it can emit a segment whose last entries are not adjacent. Since zeta *is* the
  accumulated arclength, that inflated the loop length by 1.6× and corrupted the whole parameterisation.
  `_walk_closed_loop` walks the element connectivity instead, which is unambiguous and additionally
  *validates* that the boundary is one single cycle.
* The assignment refuses a loop whose largest step exceeds five times the median rather than
  parameterising it. That check is what surfaced §11.

### 4.4 Automatic assignment

The requirement is weaker than it looks — not a *good* parameterisation, only one the old and new mesh
agree on. A deterministic geometric rule suffices: orient each segment from its lexicographically
smaller endpoint within a tolerance, sort segments by start point, normalise per segment. Segment
endpoints are domain corners and boundary intersections, which survive remeshing. Ties on symmetric
geometries get a warning, not a coin flip. **Of doubtful value now that projection exists** — see §12.

### 4.5 Facet data and HDG

**Since implemented** for every discontinuous skeleton field — (a) via the §7 snapshot/restore, (c) as
`set_facet_recovery` plus a pull-transfer using exactly the bulk-first topological disambiguation
described below; (b) MPI is done too. See [internal_facet_fields.md](internal_facet_fields.md).
The analysis is kept as written because it is the design rationale:

Internal facets *are* real `FaceElement`s in an `InterfaceMesh` named `_internal_facets_`, so the class
could carry internal data. The obstacle is elsewhere, and splits into three problems worth keeping
apart:

**(a) Persistence, not interpolation.** `InterfaceMesh::clear_before_adapt` `delete`s every interface
element and `generate_interface_elements` rebuilds from scratch, on **every mesh adaptation**, not only
on remeshing. Facet-owned data dies there long before an interpolator sees it. (§7 is how that was
closed for interface DL/D0.)

**(b) MPI ownership.** An internal facet between bulk elements on two ranks exists on both. oomph
distributes nodes and elements, not facets, so facet-owned data would need its own owner designation and
halo sync, with no existing scaffolding.

**(c) Interpolation, which for HDG should mostly be avoided.** The trace is not independent data — it is
tied to the bulk solution by the local solves and the global trace equation. So: interpolate the bulk
fields with the machinery here, then **recover the trace** on the new skeleton by evaluating the new bulk
solution there, and let one solve restore consistency. No facet-to-facet mapping, no facet ownership,
(a) and (b) both sidestepped.

If genuinely independent facet history is ever needed, the skeleton is a codim-1 mesh and §4.2 almost
applies — except it is not a manifold, it branches at edges, and closest-point projection is ambiguous
exactly there. The fix is topological, not geometric: locate the query in the old **bulk** mesh first (a
query already being made), then consider only the old facets of that element and its face neighbours.

---

## 5. The MPI design, and why it is affordable

Not implemented; sized for the design target — the projection solve on a 3d tet mesh at the 200k-dof
ceiling: ~1e6 integration-point queries, d = 3, P = 4.

**Locate, once per remesh.** Rank bounding boxes cost `P × nbox × 6` doubles; use a handful of boxes per
rank from a shallow KD split rather than one AABB per rank, because a coarse box sends a point to three
ranks that each run a full failed local search — wasted compute, not bandwidth. Routing the queries
costs `d+1` doubles per point per candidate rank: ~32 MB aggregate, ~10 MB per rank, **one `Alltoallv`**.
The reply is *not* `(elem, s)`: the located pair stays on its owner, and only a found/not-found bitmap
returns. What the origin keeps is the **schedule** — per query, the owning rank and the slot index
there, plus the collective's counts and displacements.

**Evaluate, repeated.** Only points whose source element is on another rank travel, and after a decent
partition that is the partition-interface layer — O(N^((d−1)/d)), not O(N). Realistically a few percent:
~3 MB per evaluate, ~30 MB across ten history levels.

So bandwidth is a non-issue in both phases. What would be fatal is per-point messaging, or re-deriving
the routing per field and per time level. Hence: the schedule is a first-class object, built once,
degenerate-local in serial; `evaluate` batches fields × time levels into one buffer (the current
per-field, per-time-index loops would become one collective each if translated literally);
structure-of-arrays buffers, so the send buffer is a slice rather than a gather; and the residue pass for
genuinely unlocated points is one `Allgatherv`, everyone searches, then `Allreduce(MIN, rank)` for a
unique owner — bounded, and a large residue is a bounding-box bug to be reported, not absorbed.

This also gives bulk DG transfer for free: D0/DL live on elements, so a remote source element's values
ride back in the same reply buffer.

---

## 6. What makes it fast

Measured on a ~9000-node axisymmetric triangular remesh, interleaved A,B,A,B, with results
**bit-identical** between the two backends (largest difference 0.000e+00):

| | MeshAsGeomObject | MeshPointLocator |
| --- | --- | --- |
| index build | 52 50 55 57 ms | 15 17 18 18 ms |
| location | 128 156 132 132 ms | 8.6 9.1 8.3 9.5 ms |

~15× on location, almost entirely from the affine simplex inversion: 4536 of 4615 elements qualify, the
79 exceptions being the curved-boundary elements along the arc. Before that path existed the same
benchmark read 76 ms, i.e. the k-d tree and the single-start Newton alone bought ~1.7× and the affine
inversion the remaining ~9×.

### 6.1 Which geometries avoid Newton

An element is classified once, at index-build time. **The classification is driven entirely by testing
every node against a predicted position, never by assuming an element family behaves a certain way**,
which is what makes it work uniformly across orders and shapes.

| geometry | kind | why |
| --- | --- | --- |
| straight-sided simplex, any order | Affine | a T6 with mid-edge nodes at the midpoints has its quadratic terms cancel identically |
| parallelogram / parallelepiped quad or hex, any order | Affine | the bilinear cross term `a3 = (X00 − X10 − X01 + X11)/4` vanishes exactly for a parallelogram — *not* only for a rectangle, so sheared structured meshes qualify |
| straight-edged 2d quad, non-parallelogram | Bilinear2d | a bilinear map sends the reference square's edges to straight segments, and eliminating one variable leaves a **quadratic** — closed form, no iteration |
| extruded wedge, parallelogram-based pyramid | Affine | same all-nodes test |
| curved anything, distorted 3d hex/wedge/pyramid | General | Newton, seeded from the element's best affine fit |

3d is the genuine exception: a trilinear inverse has no closed form and hex faces are ruled surfaces
rather than planes, so a distorted hex has to iterate. Self-locating a mesh's own integration points:
triangles 0.079 µs/point, rectangular quads 0.093, trapezoidal quads (bilinear) 0.117, undistorted or
sheared hexes 0.079, **trilinearly distorted hexes 10.6 µs**. Before the classification existed, quads
cost 2.3 µs/point — so it is a ~25× saving on structured meshes.

### 6.2 Wedges and pyramids: the old path could not do them at all

Not merely slower — `oomph::MeshAsGeomObject` **throws** on them. Its `locate_zeta` needs two virtuals
that `WedgeElementBase` and `PyramidElementBase` do not implement: `nplot_points()` for the multi-start
grid and `local_coord_is_valid()` for the containment check, both pure-virtual-by-exception on
`FiniteElement`. So mesh-to-mesh interpolation on a wedge or pyramid mesh was impossible, not
inaccurate.

The locator depends on neither: containment comes from the reference domains documented on the element
classes themselves, and where the geometry is not affine it runs its own Newton. Measured on
`tests/box_mesh_3d.py`, interpolating a linear field: 4.4e-16 or better in every regular and
trilinearly distorted wedge / pyramid / mixed case, against a throw on the old path, with nothing
unlocated.

### 6.3 Newton convergence, and why there are two passes

**Single-start Newton on a curved isoparametric map is not guaranteed to converge, and no formulation of
it is**: the map can be non-injective on a tangled element, the Jacobian can be singular inside it, and a
full step can overshoot. Everything below exists because of that, not because it was observed to fail.

* The `General` path runs **two passes**. Pass 1 takes one damped Newton per candidate element; only if
  no candidate accepts the point does pass 2 run the expensive multi-start. So deferring the multi-start
  is a cost optimisation, not a removal of the safety net.
* `polish_local_coordinate` is **damped**: a step that does not reduce the residual is halved, up to ten
  times, and the iteration stops if no descent is found. It returns its achieved residual, and every
  caller treats a large one as "not this element" rather than as an answer.
* Its result is only kept if it converged **and** still lies in the reference domain; otherwise the
  pre-polish coordinate is kept. Without that guard a polish that diverged on a badly deformed element
  would silently replace a good answer with a worse one.
* Wedges and pyramids have their own multi-start: the affine seed first, then on pass 2 the element's
  own node local coordinates plus their centroid, each pulled slightly towards the centroid because
  starting exactly on a vertex of a curved element can sit on a Jacobian degeneracy — the pyramid apex
  in particular.
* `search_statistics()` reports how many points needed pass 2, so this is measurable rather than assumed.

Measured against a deliberately hostile warp (non-polynomial, so nothing is affine or bilinear and the
mid-side nodes are pulled well off their chords), cell size ~1/3:

| warp amplitude | points needing multi-start | unlocated | worst error |
| --- | --- | --- | --- |
| 0.05 – 0.20 | 0 | 0 | ≤ 6.7e-16 |
| 0.30 | 30 (pyramids only) | 0 | ≤ 8.3e-16 |
| 0.42 | 90 wedge / 309 pyramid / 96 mixed | 0 | ≤ 6.7e-16 |
| 0.55 | 173 / 533 / 224 | 9 (mixed) | ≤ 5.0e-16 |

So the seeded single Newton is sufficient well past any deformation a usable mesh would have, the
fallback genuinely earns its place beyond that, and at amplitude 0.55 — a displacement larger than the
cell, i.e. a tangled mesh with no unique preimage — nine points are honestly reported as unlocated
rather than silently given a wrong answer.

### 6.4 Accuracy: the new path is not just faster

Bit-identical to `MeshAsGeomObject` wherever the geometry is straight, but on **curved** elements they
differ, and the new values are the correct ones. oomph's `locate_zeta` stops at
`Locate_zeta_helpers::Newton_tolerance = 1e-7` on the residual, so the local coordinate it returns was
only ever good to about that:

| test | MeshAsGeomObject | MeshPointLocator |
| --- | --- | --- |
| interpolate at a node, vs that node's own value | 2.1e-08 | 9.2e-17 |
| interpolate a linear field at interior points, vs exact | **9.0e-03** | 1.3e-15 |

The second row is the striking one: a linear field is reproduced exactly by an isoparametric FE space, so
any deviation is pure location error, and the old path was ~1 % wrong at arbitrary interior points of
curved elements. `polish_local_coordinate` both fixes that and removes the dependence of the answer on
the starting guess — without it, merely changing the Newton seed moved interpolated values by ~1e-8.

Two things also fell out of retiring the old backend. `MeshKDTree::find_element` used to fall back to a
`radius_search` at 10× the largest node-pair distance **anywhere in the mesh**, so one coarse element
inflated the fallback radius globally and the search degenerated to O(N). And the DL/D0 element-centre
query read `if (locator) ... else MaGO->locate_zeta(...)`, where `locator` is null when no node was in
scope and `MaGO` was null whenever the locator was enabled — an element-wise-only transfer would have
dereferenced null on the default path.

---

## 7. Adaptation, and where DG/DL/D0 stand

**Bulk refinement and unrefinement:** handled. `BulkElementBase::further_build` samples the father's DG
fields at each son's nodal local coordinate and reconstructs DL from the father's value and slope;
`rebuild_from_sons` averages the sons' DG values at coincident points and the sons' DL/D0 values. The
code documents its own caveats — `// XXX TODO: ... this does not conserve ... and does not consider
axisymmetry` — and the DL/D0 branches are written per tree type, so whether the simplex refinement path
is covered needs checking rather than assuming.

**Interface DL/D0: was silently broken, now transferred.** `clear_before_adapt` deletes every interface
element, so any interface-owned internal data went with it. `rebuild_after_adapt` guarded this for DG
spaces by refusing outright, but the guard checked only DG — and its DG test uses `numfields_new` while
`allocate_discontinous_fields` allocates DL and D0 by the full `numfields`, so nothing covered them. A D0
field pinned to 7 on an interface read back as 0 after `refine_uniformly`, with no error. It went
unnoticed for a long time because the common case — a DL/D0 field its own residual determines
algebraically — recovers at the next solve; only history-carrying fields were actually wrong, and they
were wrong silently.

The fix is a **snapshot**, and the reconstruction runs the opposite way round from the obvious sketch:
there is no father/son relation to exploit the way the bulk path does, and by restore time the old
elements are gone, so nothing can be evaluated *on* them.

* `snapshot_discontinuous_data()`, at the top of `clear_before_adapt`, samples each element on a
  5-per-direction lattice in its own local coordinates and stores the Eulerian position and the DL/D0
  values **at every time level**. Five rather than the nodes, because after one refinement each son must
  still get enough points to determine a linear field by itself, and a father's nodes sit on its sons'
  shared edges, where they arbitrate to one son and leave the others empty.
* `restore_discontinuous_data()`, at the end of `rebuild_after_adapt`, locates the whole sample cloud on
  the *new* interface mesh in one batch (Eulerian, so Project mode — an interface is codimension 1 in the
  space its positions live in; adaptation does not move nodes, so an old sample lies on the new interface
  to round-off), then fits per new element: least squares on the DL basis, mean for D0. A singular normal
  matrix — fewer points than DL modes — falls back to the constant.

Both directions work without a special case: refinement is one old element feeding several new ones,
coarsening several feeding one, and least squares over whatever points land inside handles each.
Accuracy: a DL field set exactly to `x` comes back with ~1e-8 RMS error rather than machine zero,
inherited from the local-coordinate round-off of the projection. For scale, a fit that collapsed to a
constant would sit near 7e-2.

`ensure_eleminfo_filled` guards both ends. Freshly generated interface elements have no `eleminfo`, and
both the locator and `shape_at_s_DL` size an `oomph::Shape` out of it — this cost a segfault before the
guard existed.

Skeleton facet fields adapt and remesh since
[internal_facet_fields.md](internal_facet_fields.md) - `D0`/`DL` and the nodal DG spaces alike, the
latter fitted in their own nodal basis rather than the DL modal one.

---

## 8. The projection solve

`ProjectionInternalInterpolator` was scaffolding, not a working path: referenced nowhere, and its
`interpolate()` constructed a *fresh empty* `Problem()` and called `steady_newton_solve()` on it ten
times. The C++ half was real but the driver had never run, so nothing had ever exercised it. Seven
concrete bugs were found in it, listed because each is the kind of thing that survives in code nothing
calls:

* `prepare_zeta_interpolation(new)` was passed the NEW mesh as its source, i.e. it located the new mesh's
  integration points in itself;
* the position residual was added **twice**, once with the t==0 / t>0 distinction and again with the zeta
  form regardless of the level — double the residual against a single Jacobian;
* the field Jacobian indexed the unknown by `l` instead of `l2`, so every entry of a row landed in the
  same column and the assembled matrix was not a mass matrix at all;
* the position Jacobian sat OUTSIDE its `local_eqn >= 0` test, so on any problem with pinned positions it
  wrote `jacobian(-1, ...)`;
* `field` indexes a nodal value but `field_map` is sized by `ncont_interpolated_values()`, which need not
  equal a given node's `nvalue()` in a mixed-space element;
* `enable_zeta_projection` cleared itself on first assembly, so only the first assembly of each element
  used the projection residual and every later one silently reverted to the physics;
* the history copy wrote to the *source* mesh.

The abort that stopped it was `free(): invalid size` — heap corruption, not a bad index caught by a check
— from the field Jacobian being assembled outside its `local_eqn >= 0` test. For any pinned value
`local_eqn` is −1 and `jacobian(-1, ...)` writes before the start of the elemental matrix; the damage only
surfaced later, in an unrelated free inside the sparse assembly. The position block had exactly the same
bug, fixed earlier, which is what made the bisection confusing: skipping the position block did not help
because the field block repeated it. **Finding it needed bisection rather than inspection** — a switch
disabling each block in turn, then within the field block: no source evaluation, no target evaluation, no
Jacobian. Only the last came back clean.

Three more things were needed to get a solve out of it: the source values were read with `s`, this
element's local coordinate, instead of `old_s` (the mapping between the two being the entire point of
`coords_oldmesh`); `old_elem` is NULL for an integration point that could not be located in the old mesh
at all, which a remesh of a curved boundary produces; and the C++ solver callback holds a weakref to "the
current problem" which does not point at ours during remeshing, so the solve failed with "The problem has
not been set yet" before reaching any linear algebra.

**The important fault.** The integration points were located in **Lagrangian** space, because the query
was built with `interpolated_zeta` and zeta defaults to the Lagrangian coordinate. On a freshly remeshed
mesh those are not the positions, so the query named a point that was not the integration point at all:

| | worst integration-point mapping error | points worse than 1e-8 |
| --- | --- | --- |
| via zeta (Lagrangian) | 2.3e-04 | 131 of 1862 |
| via x (Eulerian) | 2.8e-14 | 0 of 1862 |

`prepare_zeta_interpolation` now reports that mapping error under
`Mesh.set_report_interpolation_timing`, because it is the one number that says whether a projection can
possibly be right: the whole scheme rests on (old element, old local coordinate) naming the same physical
point as the integration point it came from.

### 8.1 Positions are frozen, not projected

Solving for the position dofs does not work and has been removed. The integration weights
`W = J_eulerian(s) · w` depend on the very positions being solved for, and the Jacobian assembled here
does not include that dependence — so the iteration is a fixed point, not a Newton method. On a moving
mesh started 3 % from the answer it diverged outright: residuals 1.7e-3, 7.1e-4, 4.7e-2, 1.6e+47, NaN, as
elements inverted. Seeding it with the converged nodal answer did not save it.

Positions are instead given a unit diagonal with zero residual. Nothing is lost: the current positions
are the generator's and already the answer, and the history positions — what the mesh velocity is built
from — come from the nodal transfer at ~6e-13, better than the projection could manage. **The driver must
not copy positions into the history levels afterwards**, or it overwrites those with the present geometry
and flattens the mesh velocity to zero. Freezing them also makes the field system exactly linear, which
is what lets one factorisation serve every field and every history level.

### 8.2 One factorisation, many right-hand sides

The projection residual is linear in the unknowns, so its "Newton solve" is a single linear solve whose
exact Jacobian is a mass matrix — independent of which field is being projected, of the history level,
identical for all fields sharing a function space, and SPD. The driver caches the factorisation across
history levels and refactorises only when the residual stops falling fast enough, which for pinned
positions is never. It also assembles the matrix only when it is about to factorise, calling
`get_residuals()` otherwise.

| | remesh + transfer, 3186 dofs |
| --- | --- |
| nodal | 0.23 s |
| projection, one solve per level | 0.72 s |
| projection, one factorisation | 0.45 s |

**Grouping by function space: measured, and not worth doing.** The argument for it was that the coupled
system is block diagonal with identical blocks and a generic sparse LU does not exploit this, so cost
grows superlinearly in the field count. Measured on a 12.6k-node blob remesh with 1, 2 and 4 decoupled C2
fields, factorisation at 4 fields is **4.8× the single-field cost, not the blowup the argument predicts**
— a fill-reducing ordering does separate the blocks, so the premise was wrong. The 19 % penalty that
remains is under 3 % of the remesh, while assembly was 53 %. What is left, if this is ever revisited: the
residual-only pass is still ~200 ms per call against ~300 ms for a full Jacobian assembly, so the five
convergence checks are the next target. With positions pinned the system is exactly linear and that check
is provably redundant — but it is also what caught a divergence documented in `interpolator.py`, so it
stays until something else guards that case. D0/DL need no global solve at all: they are element-local.

### 8.3 Two things it must not inherit from the physical problem

**A separate linear solver.** The projection system is structurally unrelated to the physical one: a mass
matrix, SPD, with a different sparsity and a different field layout. A PETSc `fieldsplit` configured for
a Navier-Stokes saddle point is at best wasted on it and at worst wrong, because the index sets are built
from the physical problem's dof ordering. The same goes for a user-supplied nullspace, a custom PC, or
MUMPS options tuned for the real system. Hence `Problem.set_projection_linear_solver`, **defaulting to a
robust direct solver rather than inheriting** — inheriting is the trap. The solve saves and restores the
active solver around itself so a failure does not leave the Problem holding the wrong one.

**Frozen sparsity off.** `acquire_frozen_sparsity` keys its cache on `(matrix_index, generation, ndof)`,
and `generation` is driven by `assign_eqn_numbers()`, so a change in *pinning* is caught. A change in
**residual** is not: the projection swaps the element residual behind `enable_zeta_projection` while the
dof structure is untouched, so the cache would hand back the *physical* problem's pattern for a mass
matrix. Where the projection pattern is a strict subset that only wastes explicit zeros; where it is not,
entries are dropped and the answer is silently wrong. A distinct `matrix_index` is the principled fix;
disabling frozen sparsity for the duration is the cheap and safe one, and is what is done.

### 8.4 Accuracy, measured

Across a remesh, for a field the space represents exactly (linear) and one it cannot (a Gaussian bump):

| | integral, relative change | pointwise worst |
| --- | --- | --- |
| nodal, linear | 1.8e-13 | 9.3e-09 |
| **projection, linear** | **1.4e-13** | **5.0e-14** |
| nodal, bump | 3.9e-05 | 3.1e-03 |
| **projection, bump** | **2.6e-05** | 5.8e-03 |

This is what the two methods are supposed to do. On a representable field the projection returns it
essentially exactly. On one that is not, it conserves the integral better than nodal interpolation —
which is the reason to use it — while being pointwise worse, because an L2 projection minimises the
integrated error rather than the maximum one. **Conservation is not incidental**: an exact L2 projection
conserves exactly, since constants lie in the space and Galerkin orthogonality then gives
`∫(u_new − u_old) = 0`. That identity is also the sharpest test of the implementation, and it is what
showed the mapping of §8 was wrong while the pointwise error alone was merely "large-ish".

---

## 9. Choosing the interpolator

`Problem.mesh_interpolator` selects the class used whenever the meshes are rebuilt. It has to be a
Problem setting rather than only an argument because the paths that matter most do not take one: the
remesh handler used during continuation calls `force_remesh()` bare.

**A new simulation does not have to opt in to any of this document except the projection.** The default
is `InternalInterpolator`, and the locator, the projection-based boundary transfer of §4.2, the periodic
zeta of §4.3 and every guard are all in that default path. Only the L2 projection of the bulk fields —
which trades pointwise accuracy for conservation — is opt-in, because it is the one change that is not
simply better.

---

## 10. Interface-only dofs

Interface-only fields (those an `InterfaceEquations` adds on top of the bulk's) go through their own map,
`inter_field_map`, and are evaluated with `get_interpolated_interface_field` rather than the bulk field
machinery, so they are worth checking separately. Two interface fields on different spaces, each stamped
at time level 0 and history level 1, transferred across a remesh:

| | projection | zeta |
| --- | --- | --- |
| C2 field, straight boundary | 2.2e-16 | 1.6e-13 |
| C2 field, curved boundary | 1.5e-11 | 4.8e-10 |
| **C1 field, curved boundary** | **1.4e-03** | **1.4e-03** |

The last row is not a transfer defect, which is why it is worth spelling out: a C1 field on a *curved*
interface is linear between vertex nodes, so it cannot represent a linear function of position along an
arc at all, and the chord-versus-arc gap is ~h²/8 ≈ 6e-4 for this mesh. **The tell is that the number is
identical on both transfer paths and disappears entirely on a straight boundary.**

**One real gap, fixed.** `nodal_interpolate_from`'s `missing_nodes` fallback transferred bulk fields,
position history and Lagrangian coordinates but never `inter_field_map`. An interface node that could not
be located therefore kept the zero it was built with while its bulk fields arrived normally — the
interface field simply vanished on that node, silently.

---

## 11. Two closed-loop defects, both fixed

Neither was caused by this work; both were found by the periodic-zeta guard and both corrupted more than
zeta.

**Remeshing a closed boundary misplaced one node.** After remeshing a domain whose boundary is a single
closed loop, exactly one BULK element carried a mid-side node at the **antipode** of where it belonged —
on the boundary, at the right radius, so nothing downstream noticed, while that element was grossly
distorted. Cause: `Remesher2dBoundaryLineCollection` emitted a closed curve to gmsh as **one spline with
its first point repeated**. Such a spline has a seam, and the element straddling it takes its
second-order node from the average of its endpoints' curve parameters — which at the seam averages t≈1
and t≈0 to t≈0.5, i.e. halfway around the loop. Fixed by emitting two open splines instead. Verified on a
circle (worst intra-element angular step after remesh 3.11 rad → 0.076) and on a 12-gon of straight
lines, so it was never about curved entities.

**`get_interface_line_segments` corrupted closed loops**, in two ways: on completing a loop the walk
appended `currentcurve` to `lines` and **did not reset it** — and `lines` holds a reference, so the next
fragment's nodes were tacked onto a curve that had already been emitted, inflating any arclength computed
from it by 1.6×. And the reverse-direction entry of `inbetween_pts` was filed under the key
`(e[-1], e[1])` instead of `(e[-1], e[0])`, and built from `reversed(...)`, a one-shot iterator — so a
backwards traversal fell back to the forward list and inserted intermediate nodes in the wrong order,
invisible with one intermediate node per element (C2), wrong for any higher-order space.

**And one introduced by this work, for the record.** The seam anchor's ray/edge intersection had its
numerator's sign flipped, so every genuine crossing came out with u in [−1,0], was rejected as "outside
the edge", and the search fell through to the node-quantised fallback. The seam then landed on a different
node in the old and the new mesh and the whole parameterisation was offset by about one element — a
constant 0.048 rad angular shift in the transferred field, uniform enough that only plotting the error
against angle made it obvious.

---

## 12. Where closed-loop zeta stands

| transfer across a remesh of a closed loop | worst | mean |
| --- | --- | --- |
| circle, projection | 2.311e-05 | 9.562e-06 |
| circle, periodic zeta | 8.635e-03 | 5.481e-03 |
| 12-gon, projection | 6.895e-04 | 6.737e-05 |
| 12-gon, periodic zeta | 4.173e-03 | 2.273e-03 |

Periodic zeta works — it went from O(1) (total loss) to O(h²) — but on a closed loop it is **two orders of
magnitude worse than projection**, and that gap is structural rather than a remaining bug: the seam has to
be inferred from the discretised curve, and both the anchor and the arclength along a polyline carry O(h²)
error, whereas projection is limited only by the interpolation order. For comparison, on an OPEN arc —
where the seam is a genuine endpoint and nothing has to be inferred — the same zeta path transfers at
**4.8e-10**.

**So: prefer projection for closed loops.** Keep periodic zeta for the cases projection cannot serve — a
near-touching interface where the closest point is on the wrong sheet, or when arclength semantics are
wanted for their own sake.

---

## 13. Two diagnoses from real scripts

### 13.1 The boundary pass had no boundary

On a two-domain script the transfer warnings read "16 node(s) could not be located" for a boundary that
has two nodes on it. The indices were not swapped, which was the natural suspicion.
`nodal_interpolate_from` only used `boundary_index` to decide whether to *skip* boundary nodes
(`boundary_index < 0`) and which coordinate to query with. With an index ≥ 0 on a BULK mesh it walked
**every node of the mesh**. That branch is taken for a boundary with no interface mesh of its own —
"corners to another domain" — and it runs *after* the interface passes, so it re-did their nodes and, for
the ones it could not locate, overwrote correctly projected values with a nearest-node blend. Restricting
it to `n->is_on_boundary(boundary_index)` removed every warning that script produced.

Two diagnostics changed with it, both because they had been actively misleading:
`INTERPOLATING FROM 0x23d45218` is now
`Interpolating droplet/droplet_gas from droplet/droplet_gas by projection onto the old interface`, which
also states *how* the transfer was done; and the bulk pass and the corner-case pass, which land on the
same mesh and used to print identically (reading as the same work being done twice), are now
`droplet (interior nodes only)` and `droplet/gas_axisymm (boundary nodes only)`.

Worth knowing when reading such a log: an interface transfers **by zeta only if an
`AssignZetaCoordinates*` equation was attached to it**. With none, every interface goes by projection,
which is usually what you want anyway (§12).

### 13.2 When a boundary is not the same boundary any more

Three nodes on `gas/gas_substrate` could not be located. They are one C2 element sitting where
`gas_substrate` meets `gas_substrate_refined`. **The corner node slides, and the remesher puts it back:**

| output step | min x of `gas_substrate` (m) |
| --- | --- |
| three steps before the remesh | 0.0007872699 |
| one before | 0.0008116883 |
| **immediately after remeshing** | 0.0007853982 = π/4 mm exactly |

π/4 is where the geometry puts the point. During the run the node slides along the substrate; the
remesher rebuilds the boundary from the fixed geometric point, so a strip that was
`gas_substrate_refined` before is `gas_substrate` now. **Those nodes have no counterpart on the old
`gas_substrate` at all.**

Two plausible explanations are both wrong, which is why this is worth spelling out. The zeta there *is*
x, re-derived on both meshes immediately before the transfer, so it neither drifts nor stretches — it is
the boundary's extent that changed. And the bulk mesh is not the answer: interfaces carry their own
degrees of freedom, which the bulk knows nothing about.

Three things were implemented and none of them fixes the run: a node that zeta cannot place is retried by
projection onto the old interface before anything falls through to the blend (which rescues nodes just
past the old boundary's end); `LocatorSetup::inside_tolerance` is 1e-8 in local coordinate units rather
than 1e-10, because a query exactly on an element edge inverts to `s = 1 + a few ulp` as readily as to
`1 − a few ulp` and rejecting the first reports a point unlocatable for a purely numerical reason; and
`Problem._debug_remeshing` writes an output immediately before and after every remeshing, which is how
the table above was produced — every earlier attempt at this diagnosis was inference from warnings.

**What actually resolved it: the geometry has to follow the mesh.** Nothing in the library is wrong; the
mesh template asks for a boundary the old mesh does not have. `define_geometry` runs again on every
remesh, so a line like `pr0 = self.point(radius*pi/4, 0, ...)` re-pins the corner to its *initial* place
while the node it corresponds to has moved. Reading its current position from the old mesh instead is
enough, and the run then produces no interpolation warnings at all.

**The general rule: anything in `define_geometry` that a moving mesh can move must be re-read from the old
mesh on a remesh, not recomputed from a constant.** A junction between two *named* boundaries is where
that hurts, because the strip between the old corner and the new one changes which name it belongs to and
interpolation is per-boundary. A pinned point in the *interior* of one boundary is harmless by comparison
— it perturbs where nodes land, but every node still has the same named boundary to be located on.

**The library-side fix was considered and dropped.** Searching the *sibling* interfaces of the same bulk
mesh when the corresponding one cannot place a node would make a genuine modelling mistake — naming the
same stretch of substrate differently before and after — transfer quietly and look correct. Should it ever
be wanted, the obstacle is bookkeeping rather than concept: `field_map` and `inter_field_map` are computed
once between this mesh's code instance and the source mesh's, so a different source interface needs its
own maps.

---

## 14. What was deleted, and what was deliberately kept

`oomph::MeshAsGeomObject` no longer appears anywhere in pyoomph outside comments that explain why the new
code looks as it does. Gone with it: `Mesh::use_point_locator` and every branch it selected;
`BulkElementBase::prepare_zeta_interpolation(oomph::MeshAsGeomObject*)` and the
`#include "mesh_as_geometric_object.h"`; and `KNNInterpolator`, dead behind `if False:` and the module's
only `sklearn` dependency. `MeshKDTree` went with the tracer campaign (see [tracers.md](tracers.md)),
which was its last user; the staleness signal is now `Mesh::bump_topology_generation()`, a counter rather
than the address of a cached object that the invalidation deleted.

**Kept deliberately:** the two-nearest blend in `missing_nodes` and in `nodal_interpolate_along_boundary`
— already demoted from silent default to reported last resort, and reachable now only when a point
genuinely lies outside the old mesh; §13.2 is a case where it is the only thing standing between a moved
boundary and no values at all. And the two statics of §3.1.

Tests: `tests/test_mesh_point_locator.py`, plus `test_adaptive_interface_coupling.py`,
`test_curved_boundaries.py`, `test_triangle_refinement.py`, `test_mixed_mesh.py` as regressions.
`add_interpolated_nodes_at` has no in-tree caller and `prepare_zeta_interpolation`'s only caller is the
projection interpolator, so neither is covered by the suite; both were checked directly instead — the
former against the other backend point by point including a point outside the mesh, the latter by
self-location, where every integration point must be found in the element it came from.
