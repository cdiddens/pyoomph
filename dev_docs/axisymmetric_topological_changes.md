# Axisymmetric pinch-off and coalescence

Status: **works, opt-in per interface, validated on real surface-tension-driven flow.** Attach an
`AxisymmetricReconnection` to a free surface whose bulk mesh comes from a
`TopologicalChangesGmshTemplate` and the interface is allowed to change its topology: a neck thinner
than `rmin` pinches off, two fragments closer than `distmin` merge. Single phase and two phase, serial,
`mpirun` and `--distribute`. The optional dependency is `shapely >= 2.0` (`pip install pyoomph[topology]`).

This replaces `AxisymmetricPinchoffAndCoalescence`, which is deleted. §9 says why the port was not
attempted; §10 lists everything that is still refused or still approximate.

| | |
| --- | --- |
| geometry, no FEM imports | `pyoomph/meshes/axisymm_topology.py` |
| the equation and the template | `pyoomph/equations/topological_changes.py` |
| tests | `tests/test_axisymm_*.py`, `tests/test_mpi_axisymm_reconnection.py`, `tests/test_rayleigh_plateau_pinchoff.py`, `tests/test_droplet_coalescence.py` |

---

## 1. The workflow

Two objects, and one rule about `define_geometry`.

```python
from pyoomph.equations.topological_changes import (AxisymmetricReconnection,
                                                   TopologicalChangesGmshTemplate)

class JetMesh(TopologicalChangesGmshTemplate):
    def define_geometry(self):
        self.mesh_mode = "tris"
        pr = self.get_problem()
        if self.is_first_time():
            zs = numpy.linspace(0.0, pr.L, 61)
            pts = [self.point(1 + pr.amplitude * numpy.cos(pr.k * z), z) for z in zs]
            self.spline(pts, name="interface")
            self.create_lines(pts[0], "bottom", self.point(0, 0), "axisymm",
                              self.point(0, pr.L), "top", pts[-1])
        else:
            rb = self.get_reconnected_boundaries("liquid/interface", "liquid/axisymm")
            for chain in rb.interface_chains:
                self.spline_from_chain(chain, "interface")
            self.lines_from_axis_segments(rb.axis_segments, "axisymm")
            for chain in rb.interface_chains:          # the two symmetry planes, per fragment
                for end, kind in ((0, chain.end_types[0]), (-1, chain.end_types[1])):
                    if kind == "fixed":
                        x, y = chain.points[end]
                        self.line(self.point(0.0, y), self.point(x, y),
                                  name="bottom" if y < 0.5 * pr.L else "top")
        self.plane_surface("bottom", "axisymm", "top", "interface", name="liquid")

...
eqs += NavierStokesFreeSurface(surface_tension=1) @ "interface"
eqs += AxisymmetricReconnection(rmin=0.08) @ "interface"
eqs += RemeshWhen(RemeshingOptions())
```

That is the whole contract, and `tests/axisymm_physics_worker.py` is the runnable version of it
(`JetMesh`, `DropletsMesh`, `TwoPhaseDropletsMesh` — the third adds a gas domain built from the same
template).

**The `else:` branch is one code path, not two.** It is entered for a reconnection *and* for an
ordinary quality remesh, and it is not told which: with no plan pending,
`get_reconnected_boundaries()` wraps the current geometry — the output of
`MeshedMeshTemplate.get_boundary_coordinates()` — into the same `ReconnectedBoundaries` structure.
This was a deliberate API decision. The alternative, a separate hook for the surgery, means the user
writes and maintains a second geometry builder that runs once per event and is therefore never
exercised; here the branch that carries the event is the branch that runs on every remesh.

**What the branch has to be able to cope with** is a *variable number of fragments* and axis spans
that appear and vanish. In the example that is a `for` loop and nothing else. Two further points that
the tests found the hard way:

* `end_types` is what tells a chain end at the wall from a chain end on the axis. After a pinch each
  fragment of the jet above still owns one symmetry plane and has grown one fresh axis cap, so the
  count of "fixed" ends per chain drops from two to one and the code must not assume either.
* `rb.opposite_axis_segments` (only filled when `opposite_axis_name` was given) is the *complement*:
  what the other phase covers is everything the axis was covered by before, by either phase, minus what
  this phase covers now. Building it that way is what makes the gap a pinch just opened appear on the
  gas side without the caller special-casing it.

`spline_from_chain` and `lines_from_axis_segments` are conveniences. `chain.points` /
`chain.suggested_sizes` are public, so dropping to `self.point(..., size=...)` / `self.spline(...)` for
full control over the resolution is always available; `suggested_sizes` are the *old* local element
sizes, with cap points at the plan's cap spacing.

### 1.1 Surgery is not built on `Remesher2d`

`Remesher2d` reconstructs a geometry from the deformed mesh's boundary nodes, and the legacy handler
edited its point and line entries in place. That is the one remeshing path that *cannot* run the user's
`define_geometry`, so the resolution of the mesh after an event was whatever the automatic
reconstruction produced. Delivering the surgery through recreation instead gives the event no special
status at all: it is the ordinary `RemesherViaRecreation` path (`mesh.py:841-844`,
`remesher.py:670`), and the user chooses the sizes.

---

## 2. Detection: morphology on the mirrored cross section

The interface polylines are closed into a half-section and **mirrored about the axis** into a full 2D
cross section `P`; fragment count is polygon count. Detection is then a topology diff of two
morphological operations on `P`.

| | operation | structuring radius | threshold it realises |
| --- | --- | --- | --- |
| pinch-off | opening, `P.buffer(-e).buffer(+e)` | `eps_p = rmin` | minimum interface radius below `rmin` |
| coalescence | closing, `P.buffer(+e).buffer(-e)` | `eps_c = distmin/2` | tip-to-tip axial gap below `distmin` |

**The mirroring supplies the factor two for the opening.** An opening deletes exactly the parts of a
set that cannot contain an open disc of radius `e`. A neck of minimal interface radius `r_w` appears in
the mirrored section as a full-width strip of half-width `r_w`, whose medial axis is the symmetry axis
itself, so the erosion empties it precisely when `r_w < e`. Hence `eps_p = rmin` and **not** `rmin/2`:
writing `rmin/2` there would put the pinch threshold at half the radius the user asked for. The closing
has no such factor, because a gap of length `d` between two fragments is a gap of length `d` in the
mirrored section too, and both sides grow by `e` under the dilation — so `eps_c = distmin/2`.

### 2.1 Why the three-step composite was rejected

`P.buffer(+e).buffer(-2e).buffer(+e)` looks like a symmetric "smooth both ways" operator and is a
common shortcut. It is not an opening. Regrouping it as `((P (+e) (-e)) (-e) (+e))` shows it to be the
**opening of the closing**, i.e. it silently bridges every gap shorter than `2 eps_p` before it opens
anything. That couples the two thresholds: with the composite, `rmin` also acts as a coalescence
distance, and the combined neck-and-gap case becomes ambiguous — a pinch would be re-bridged by its own
detector on the same call. The plain two-step forms commute with the intended thresholds and are what
`tests/test_axisymm_topology.py` pins down.

### 2.2 The two caveats that remain, and what each does

Neither is papered over; both raise.

**(i) The erosion carries the topology, the dilation can undo it.** The dilation that completes the
opening re-glues two eroded components whenever they end up closer than `2 eps_p`, i.e. whenever the
waist is axially *shorter* than the structuring element. The component counts of the erosion and of the
opening are compared and a mismatch raises.

This one has a timing character that only the physics tests exposed, and it is the single most
important thing in this document for anyone using the feature: **a collapsing neck hits it once, every
time.** At the step where the minimum radius first drops below `rmin`, the axial stretch over which the
interface is below `rmin` has only just opened from zero, so it is far shorter than `rmin` — the
erosion separates the section at a point and the dilation glues it straight back. One or two more steps
of thinning make it long enough. Measured on `tests/test_rayleigh_plateau_pinchoff.py`, over `eps_p`
from 0.02 to 0.15 `R0`: exactly one step, at every value tried.

So it raises as `WaistNotYetSeparable`, a type of its own, and `AxisymmetricReconnection` catches
*that type only* and reports no event for this solve — the next one re-detects. Nothing has been
changed at that point, so waiting costs nothing. It does not wait for ever: `max_deferred_events`
(default 25) consecutive deferrals re-raise, because a neck held just below `rmin` by something other
than capillary collapse would otherwise defer indefinitely while the mesh around it degenerates. A
genuinely too-coarse `rmin` fails differently anyway, and quickly — see §10.

**(ii) A pinch gap shorter than `distmin`.** If the axial gap the opening carves out at a waist is
itself shorter than `distmin`, the closing that follows would immediately re-merge the two children.
That is a contradictory parameter choice and it raises. The constructor also warns at set-up time if
`distmin > rmin`.

### 2.3 The gates around the morphology

Two things sit in front of the geometry, and both exist because of a defect in the legacy handler.

* **The mesh-velocity gate** (`check_mesh_motion_direction`) refuses a coalescence whose two tips are
  moving apart. It reads `partial_t(var("mesh_y"), ALE=False)`. The `ALE=False` is essential: under the
  default `ALE="auto"`, `partial_t(var("mesh_y"))` on a moving mesh is `d(mesh_y)/dt - u_mesh.grad(mesh_y)
  = u_y - u_y`, identically zero. The legacy gate compared `0 > 0` and therefore blocked every pinch-off
  and suppressed no coalescence at all. (Related: a position pinned by a `DirichletBC` carries the
  prescribed value in all of its history slots and so reports a mesh velocity of exactly zero — which is
  why the kinematic tests drive the mesh by `PrescribedMovingMesh` velocity, not by position.)
  The gate discards the whole plan rather than editing the offending event out of it: a plan is one
  consistent description of the entire new interface, and dropping it costs at most one step.
* **The overlap guard** (`overlap_reject_factor`, `before_newton_convergence_check`) rejects a Newton
  step that would bring two tips closer than `overlap_reject_factor * distmin`, so an adaptive time
  stepper cuts `dt` instead of letting two fragments pass through each other. It is only armed once some
  gap is below `coalescence_arm_factor * distmin`, because it runs on *every* Newton step.

### 2.4 Nodes a hair past the axis

`_snap_negative_radii_to_axis` pulls interface points with `r < 0` back onto the axis before the chains
are handed to the geometry module, up to a quarter of the local point spacing; further out it raises
and names the node.

This is not defensive tidiness. A free surface that has just retracted into a fresh cap leaves nodes at
a small negative fraction of an element size: the tip node is pinned at `r=0` by the axisymmetry
condition but its neighbours are not, and one Newton step on a strongly curved cap is enough. The
mirrored ring (`_ring_coords`) then puts the forward run of the chain on the far side of its own mirror
image — shapely calls it self-intersecting and the detection ends the run. It showed up in one of three
Rayleigh-Plateau runs at coarse resolution, several steps *after* the event that caused it, i.e. as an
intermittent failure with no visible connection to its cause. The axis is at `r=0` by construction, so a
negative radius carries nothing worth keeping; a node further out than a fraction of an element is a
folded mesh, and turning that into a valid polygon would hand the surgery a geometry that does not
exist.

---

## 3. Building the new interface

### 3.1 Identity away from the event

Only points within `cap_window_factor * eps` (geodesic distance) of the symmetric-difference blobs are
replaced. Everything else keeps its exact coordinates, its size and its zeta — bit-identical, and
untouched fragments are not even volume-corrected. Cap and bridge portions are resampled at
`min(cap_spacing_factor * local_size, 0.7 eps)`, which also bounds the vertex count shapely's round
joins produce.

The last `2 eps` towards an axis tip is reparameterised via `u = x^2`, so the emitted curve meets the
axis exactly perpendicular (`x ~ sqrt(a |z - z_tip|)`) instead of at whatever angle a uniform sample
happens to give.

### 3.2 The volume correction, and what "nearly" means

Revolved volume of a half-section polyline is exact:

```
V = pi * sum_i (r_i^2 + r_i r_j + r_j^2)/3 * (z_j - z_i)     (axis segments contribute 0)
```

Targets: pinch children get the parent's volume split at the waist plane (so the children sum to the
parent exactly); a coalescence target is the sum of the parents. The correction perturbs **only fresh
points** (`origin == -1`) along their outward normals, `p -> p + A w(t) n`, with a smoothstep weight for
caps and `sin^2(pi t)` for bridges, and solves `V(A) = V_target` by `brentq` on `A` in `[-2 eps, 2 eps]`.
The cap weight is full at the tip, whose normal is axial — so the tip *slides along the axis*, which
preserves both `r = 0` and the 90-degree contact.

The result is **exact on the plan polyline**: 2e-16 relative, against 6.5e-4 with the correction
switched off (`tests/test_axisymm_pinchoff_remesh.py`).

"Nearly" refers to what happens next. `define_geometry` turns the plan's points into a Catmull-Rom
spline and the spline bulges outside the chords, so the volume the *rebuilt mesh* carries drifts from
the plan's target by O(h^2 * curvature). That is a property of the recreation path, not of the surgery:
it is the same drift an ordinary quality remesh has. Measured on the kinematic dumbbell (relative drift
per fragment):

| h | drift | C = drift/h^2 |
| --- | --- | --- |
| 0.08 | 3.59e-3 | 0.56 |
| 0.06 | 1.76e-3 | 0.49 |
| 0.04 | 8.61e-4 | 0.54 |
| 0.03 | 4.72e-4 | 0.53 |

so `C ~ 0.55`, and the tests bound it at `max(1e-6, 1.5 h^2)`. `ReconnectedBoundaries.fragment_volumes`
exposes the targets so a user can check for themselves.

§8 has the end-to-end numbers from the physics tests, where the event's share of the volume budget is
compared against the ordinary per-step drift rather than against zero.

---

## 4. Carrying the fields across: the zeta chart

Near the event the old and the new interface are **not the same curve**, so nothing derived from the
geometry alone can map one onto the other there: an arclength chart restarts at a fresh cap, and a
nearest-point projection would happily pair a point just below a pinch with the interface just above it.
The plan, on the other hand, knows which new points *are* old ones and where the fresh ones came from,
and it writes that down as a zeta chart. Installing that chart on both sides turns the transfer of the
interface fields into the ordinary zeta-based one.

**The old chart** is cumulative unnormalised arclength per chain, ascending `z`, with
`segment_jump_offset` inserted between chains — the `AssignZetaCoordinatesByArclength` convention, so
`_check_zeta_is_invertible` accepts it.

**The new chart.** Surviving points keep their old zeta verbatim. A pinch cap's zetas map monotonically
*into the old-chart range of its own half of the removed neck*, so `locate_zeta` on the old mesh finds
them on the correct side of the waist and never transfers material across it. A coalescence bridge maps
its two halves into or near the two old tip ranges, with the jump gap falling between two adjacent
bridge nodes. Per-chain strict monotonicity is asserted.

The cap window's chart follows the **old arclength** rather than a uniform ramp. A uniform ramp over the
`6 eps` window was tried first and shifted the transferred field by about one element; the waist margin
went from 5 % to 1e-4 at the same time.

**Old side and new side are matched differently, on purpose.** The old interface nodes *are* the points
the detection ran on, so their zeta is looked up exactly, by position
(`zeta.assign_zetas_from_position_table`). The new ones are not: the mesh generator produced them from
a spline through the plan's polyline, so they sag off it by O(h^2 * curvature) and only a projection can
place them (`zeta.assign_zetas_by_polyline_projection`, tolerance a fraction of the segment hit, not an
absolute number).

**The chart governs interface-only dofs; the bulk does not go through it.**
`interpolator.zeta_for_interface_fields_only` restricts it that way. Bulk values at interface nodes come
from geometric point location in the old bulk, which is both more accurate and defined for nodes the
chart has nothing sensible to say about. This is the same distinction the codimension-2 corner pass
needed (stage 1): a two-node blend along a boundary must not overwrite an accurate bulk value.

**`bulk_locate_boundaries` and the axis.** A pinch opens a gap in the symmetry axis and a coalescence
closes one, so the fresh axis nodes have no counterpart on the old axis *at all* — the old boundary
either did not reach there, or ran through material that now belongs to the other phase. They do lie in
the old **bulk** of the correct phase, and that is the question worth asking, so the axis name goes into
`interpolator.bulk_locate_boundaries` and its nodes are located in the bulk instead of along the
boundary. The price is whatever interface-only dofs the axis carries; a symmetry boundary normally
carries none. `boundary_max_distances` for the interface/axis corner pair is set to
`2 max(rmin, distmin)` at the same time, nondimensional, because that is the scale over which the corner
legitimately moved.

**A one-off chart.** If the user has no zeta assigner on the interface, the chart is installed for this
transfer only and `boundary_coordinate_bool` is put back down in `after_remeshing` — otherwise the next,
ordinary remesh would find an old mesh claiming a chart the new one does not have and refuse. If the
user *does* have an assigner, `mesh._zeta_chart_overridden` keeps it from re-charting the new mesh from
the geometry in `after_mapping_on_macro_elements`, which runs between this hook and the transfer and
would otherwise read old and new through different parameterisations.

**The opposite phase** is reached through `interpolator.remesh_group`. The interface is a boundary of
both domains, each with its own interface mesh, boundary index and zeta, and each transferred by its own
domain's interpolator — which this equation is never dispatched with, since the hook is called per
domain eqtree. `Problem.force_remesh` / `redefine` therefore build **all** interpolators before
dispatching the hooks, so the group is populated when the first hook runs.

---

## 5. Under `--distribute`

The detection is a statement about the **whole** interface — where its thinnest waist is, which fragments
face each other across which gap. A rank's partition of it is not a piece of that answer but a
different, truncated shape, cut wherever the partition happens to run, and a waist sitting on a partition
boundary is not a corner case: it is what partitioning by element count normally does.

So the data flow is:

```
local mesh data --> merged cache (collective, result on rank 0)
                --> rank 0: _detect_from_data()  [pure numpy in, plan out]
                --> comm.bcast(payload)          [plan + armed flag + old zeta table]
                --> mpi_share_root_failure(...)  [rank 0's refusals become everyone's]
                --> every rank parks the IDENTICAL plan on the template
```

After that nothing needs to communicate: `define_geometry` is already a collective region,
`get_reconnected_boundaries` reads the parked plan (or the equally collective
`get_boundary_coordinates`), and each rank writes the zeta chart onto the nodes it holds. The gate into
the merge is `get_mpi_any(needs_merging(mesh))` — agreed rather than trusted locally, exactly as in
`AssignZetaCoordinatesBase`. Serial and `mpirun` without `--distribute` answer `False` with no
communication at all and take the local path unchanged.

Three details that are load-bearing:

* **The armed flag rides the same broadcast.** It gates a collective guard on every Newton step, so a
  rank that kept a stale `True` while the others went `False` would walk into a broadcast alone and hang.
  It is cleared before every early return, including the one for "no interface found at all", which used
  to be that asymmetry serially.
* **`_interface_and_axis_name` and `_interface_element_size_factor` are reduced over the ranks.** Both
  read a boundary a partition may hold none of. The size factor is the sharper one: a rank with no
  interface elements reports 1 where the others report 2, and since only rank 0's `.msh` becomes the
  mesh, a rank 0 without interface elements would halve the mesh size of the whole problem at every
  remesh.
* **`_redistribute_after_remeshing` maps the nodes back onto their macro elements after
  `distribute()`.** oomph places the halo copies' nodes itself, and on a strongly curved spline that is
  somewhere else — 3 nodes up to 1.4e-2 off, 6 halo mismatches, and the merged mesh data then refusing.
  Pre-existing: a plain remesh of the same peanut breaks the same way (`distributed_remeshing.md` §3.2a).

**Known and rank-count dependent:** the few fresh cap/bridge nodes that no rank can locate (3 of ~400 in
the tests) fall back to a nearest-node blend, and that blend is rank-local. Their values therefore move
with the rank count — sums by up to 2e-4 relative; min, max and counts are exact. Lifting it needs the
distributed point locator of `distributed_remeshing.md` §3.6.

---

## 6. Interpolation defects fixed underneath

Three general defects of `nodal_interpolate_along_boundary` were found by exploration and hit *any*
remesh, fresh caps and corners hardest. They are fixed and tested independently
(`tests/test_boundary_interpolation_fixes.py`):

* `bi_old` was used where `bi_new` was meant, in the no-interface-mesh bulk fallback
  (`interpolator.py`);
* `inter_field_map` in `Mesh::nodal_interpolate_along_boundary` was built from the **bulk** field tables
  gated on the interface tables, so it came out empty and interface-only fields and codimension-2 dofs
  were silently dropped. It is built from the interface meshes' own tables now, with a NULL guard on
  `info_DL` and a `face_value_index()` helper — `operator[]` on the oomph map *inserts* a zero for an
  unknown id, which would write over a bulk field;
* the codimension-2 corner pass overwrote accurate bulk values with a two-node blend; it now passes
  `only_interface_fields=true`.

Deliberate follow-up: under `only_interface_fields` the corner pass still blends position history and
Lagrangian coordinates. That matters only for moving-mesh history at corners.

---

## 7. Tests

| file | what it pins | cost |
| --- | --- | --- |
| `tests/test_axisymm_topology.py` | the geometry module standalone: thresholds, the composite rejection, every refusal, the volume correction | 20 tests, seconds |
| `tests/test_axisymm_reconnection_detect.py` | detection, the velocity gate, the overlap guard | subprocess worker |
| `tests/test_axisymm_pinchoff_remesh.py` | pinch end to end through `define_geometry`; the O(h^2) volume convergence of §3.2 | 2 resolutions |
| `tests/test_axisymm_coalescence_remesh.py` | coalescence end to end | |
| `tests/test_axisymm_twophase_remesh.py` | liquid + gas from one template | |
| `tests/test_axisymm_reconnection_transfer.py` | the zeta chart of §4: fields on both sides, with and without a user assigner, `handle_zeta=False` as the control | 15 tests |
| `tests/test_mpi_axisymm_reconnection.py` | §5: identical plans and digests at 2 and 4 ranks, with and without `--distribute`, against a serial run of the same worker | 18 tests, `slow` |
| `tests/test_rayleigh_plateau_pinchoff.py` | §8: real capillary pinch-off, satellite and all | 7 tests, `slow` |
| `tests/test_droplet_coalescence.py` | §8: real merging droplets, single phase and with a gas box | 13 tests, `slow` |

The two physics modules together are 20 tests in ~65 s (three worker runs, one per scenario).

The workers are `tests/axisymm_reconnection_worker.py` (kinematic) and
`tests/axisymm_physics_worker.py` (Navier-Stokes). One `Problem` per process throughout: a second one
in the same process segfaults in the JIT loader.

**Kinematic on purpose, in the first worker.** What those suites test is the detection, the plan and the
geometry `define_geometry` rebuilds from it. A free-surface flow would add a Newton solve whose outcome
the assertions depend on, and would have to be *tuned* until it pinches — which is exactly the kind of
test that stops being about the code under test. The physics worker exists to answer the complementary
question, which is not "does the surgery produce the right geometry" but "does the solve survive it".

---

## 8. What real physics showed

Nondimensionalised with the unperturbed radius `R0`, the density and the surface tension, so lengths are
in `R0`, time in the inertio-capillary time and the viscosity is the Ohnesorge number.

### 8.1 The three scenarios

| | driving | resolution | dofs | wall time |
| --- | --- | --- | --- | --- |
| Rayleigh-Plateau pinch-off | one wavelength at `k R0 = 0.697`, `a = 0.5`, `Oh = 0.1`, surface tension only | `h_min = 0.04 R0`, 2.5 elements per local radius, `rmin = 0.08 R0` (2 elements) | 2.3k -> 11k | 31 s, 73 accepted steps |
| droplet coalescence | two spheres of radius 0.8 approaching ballistically at 0.25 | `h_min = 0.06 R0`, `distmin = 0.15 R0` | 2.4k -> 12k | 14 s, 24 steps |
| two-phase coalescence | the same, inside a gas box (`ConnectMeshAtInterface` + `ConnectVelocityAtInterface`) | `h_min = 0.07 R0` | 5.3k -> 14.5k | 21 s, 34 steps |

The pinch-off produces a **satellite**: the column does not pinch at one point but at both ends of the
thin filament between the two growing drops, so the plan carries *two* simultaneous pinch events and
leaves three fragments. Not suppressed - it is a harder case than a single waist and it is the only
thing exercising more than one event in one plan.

All three are run **single-threaded** (`OMP_NUM_THREADS=1` plus `General.NumThreads = 1` for Gmsh),
which makes them bit-reproducible. They are not otherwise, and it matters: the transient ends in a
capillary singularity, two runs whose meshes differ in the last bits separate within a few dozen steps,
and the pinch-off test failed about one run in five with three different messages, none of which said
anything about the cause. Pinning the threading is what turned that into a test.

### 8.2 The volume budget

The number that matters is not the end-to-end drift, which the free-surface kinematic condition
produces on every step anyway, but the event's **share** of it. So the volume is recorded per accepted
step together with what happened in that step, and the three kinds of step are compared:

| relative change of one step | RP pinch | coalescence | two-phase |
| --- | --- | --- | --- |
| worst ordinary step | +9.3e-6 | -6.3e-5 | -1.4e-4 |
| worst quality remesh | -8.9e-6 | -8.8e-5 | -3.7e-5 |
| **the event step** | **-1.2e-5** | **-4.0e-4** | **-1.3e-5** |
| whole run | -1.4e-4 (73 steps) | -8.2e-4 (24) | -4.0e-4 (34) |

So the surgery costs about what an ordinary quality remesh of the same mesh costs, and both are one to
two orders of magnitude below the 0.5 % the plan asked for. The coalescence event is the largest of the
three because a bridge is by far the most strongly curved piece of interface either scenario ever
carries, and the O(h^2 * curvature) sag of §3.2 scales with exactly that - geometry, not a defect. (The
two-phase worst ordinary step is its *first* step: the gas starts at rest while the liquid starts
moving, so the connected velocity absorbs an impulsive start. Halving the approach step size removed
it, and it has nothing to do with the event.)

The liquid and gas meshes stayed conforming to **exactly zero** across the event, and liquid + gas
reproduced the fixed box volume to 2e-16 - a sharper statement than the liquid volume alone, since a
surgery could conserve the liquid perfectly and still leave a sliver belonging to neither domain.

### 8.3 What real physics found that the kinematic tests could not

Five things, all of them fixed or accounted for in the sections above. Collected here because each was
invisible to a prescribed interface, and the first four would have hit the first user.

1. **Every collapsing neck ended the run** at the step where its waist first crossed `rmin`, with
   "axially shorter than 2*rmin". A prescribed dumbbell is handed a waist that is already long; a
   *collapsing* one is not. Fixed by giving that refusal its own type and deferring - §2.2(i).
2. **Nodes a hair past the axis** made the mirrored cross section of a *later* detection
   self-intersect, in one of three runs, several steps after the event that caused it. Fixed by §2.4.
3. **The first step past the event must not be a BDF2 step.** A node the surgery created has no
   history: the transfer gives it whatever the old mesh held at that place, which for a fresh cap is
   the middle of a neck that was collapsing at the largest velocity anywhere in the domain. With fixed
   time steps, BDF2 extrapolates through that history and the Newton solve diverges - on the *first*
   post-event step, not two steps later as this section used to say. §8.4 explains why it could not be
   two: the history slots are shifted at the start of every step, so the transferred history reaches
   the scheme only as its level 2, on one step, and has left it entirely by the second. With adaptive
   stepping the run survives by subdividing the step, which is why the numbers here looked healthy.
   One `assign_initial_values_impulsive()` at the event is what makes the continuation work, and it
   has to be done **at every event**, not only the first - a satellite pinches again, and a second
   event with no restart kills the run just like the first would have. It is done by the test worker
   and by the tutorial, not by the handler, because it is a statement about the caller's time stepper.
   **A user integrating through a pinch should do the same.** §8.4 measures the alternative, one step
   with BDF1 weights, at about 270 times more accurate - and less robust here, which is why the
   impulsive one is still what the tutorial and the worker do.
4. **Remeshing after a pinch is not optional.** Continuing on the surgery's own mesh, with quality
   remeshing switched off, failed in 5 runs out of 5 - the two fresh caps retract fast enough to
   degenerate the mesh within a handful of steps. Which also says where the *remaining* post-event
   fragility belongs: to the remeshing path, not to the surgery having left a bad mesh (§10).
5. **A two-phase moving mesh needs `ConnectMeshAtInterface`.** The two domains of one Gmsh template
   share the curve, not the nodes, so a free surface on the liquid side moves only the liquid copy.
   The gas copy drifted 0.07 `R0` away within a few steps, and the failure surfaced as a Gmsh loop that
   could not be closed at the next remesh, because the gas axis no longer met the liquid tip. Not a
   defect of this feature - it is how pyoomph two-phase problems have always been written - but it is
   the first thing to check when a two-phase reconnection misbehaves, and the tutorial-shaped example
   in `tests/axisymm_physics_worker.py` has it.

Two non-findings worth recording, because both cost time:

* **`rmin` cannot be chosen freely.** Scanning `eps_p` from 0.015 to 0.15 `R0` over the whole
  Rayleigh-Plateau transient: everything at or below 0.08 `R0` worked at every step past the
  threshold; 0.10 and 0.12 worked at some steps and failed at others; 0.15 never worked. The failures
  are the splicing ones of §10, not the morphology.
* **A short dumbbell in a gas box does not pinch.** At neck 0.25, half length 1.5 the neck stopped
  thinning at 0.22 and sat there for 400 steps - Rayleigh stability, a correct answer to a badly posed
  question. The two-phase test is a coalescence for that reason as much as for cost.

### 8.4 How to restart BDF2 across the event, measured

The question this section answers is whether a run can stay genuinely *second order in time* across a
topological change, and if so at what price. It can, and the price is one time step with BDF1
weights: an order of magnitude more accurate than the `assign_initial_values_impulsive()` of §8.3 and
second order over three decades of step size. Four ways of repairing the history of the fresh nodes
specifically - the obvious thing to want - were built and measured as well; all of them came out
worse on accuracy, and the code that implemented them was **removed again afterwards**. What follows
is therefore the record of an experiment, not a description of an option in the tree: there is no
knob to turn, and the numbers below are the reason there is none.

The reason the accurate answer is nevertheless *not* what §8.3 recommends is at the end of this
section: on the Rayleigh-Plateau pinch the post-event phase is a cap retraction the mesh only just
resolves, and the dissipation that makes the impulsive restart inaccurate is what keeps that
retraction from folding the mesh.

**The mechanism first, because it makes the rest inevitable.** oomph shifts the history slots at the
start of every step (`BDF::shift_time_values`: `value(2) := value(1)`, `value(1) := value(0)`). So on
the first step after a remesh the scheme sees the transferred level 0 - the current, correct state -
as its level 1, and the transferred level 1 as its level 2; and on the second step both transferred
levels have already been shifted out. **The entire influence of whatever the transfer wrote into the
history is one level-2 slot on one step.** BDF1 weights ignore level 2. Hence one BDF1 step removes
that influence *exactly*, and it cannot help to repair the history instead: nothing is left to repair
after that step, and a single first-order step does not lower the global order anyway (its local
error is `O(dt^2)`, which is the size of the second-order scheme's *global* error).

That is not an argument, it is measured: `eulerian+bdf1` and `freeze_position+bdf1` below reproduced
plain `bdf1` to twelve digits at every step size, and `tests/test_axisymm_transferred_history.py`
still pins the identity - it perturbs the transferred history levels directly (the worker's
`--flatten-history`, a test instrument) and asserts that one BDF1 step makes the difference vanish to
the last bit, and that without that step it does not.

**The instrument.** The Rayleigh-Plateau pinch cannot answer the question. Its usable fixed step past
the event is bounded above by the cap retraction (`2e-4` at the tutorial's resolution, i.e. fifty
times smaller than the step before the event, and *every* restart strategy diverges above it) and
below by a Newton stagnation on the fresh caps' surface-tension residual that sets in around `5e-5` -
the same window taken one step *before* the event converges at every step size, so it is the
post-surgery geometry and not the history. Two usable step sizes is not a convergence study.

So the study is run on a kinematic analogue: the same one-wavelength column and the same handler, but
the geometry is driven by a *prescribed* mesh velocity `u = (-V r exp(-((z-L/2)/w)^2), 0)` that
collapses the waist, and the only unknown field is a scalar transported in the ALE frame with
diffusion. Everything is smooth, Newton converges at any step size, the pre-event trajectory is
identical in every run (fixed steps, so the event lands on a step boundary), quality remeshing is off
for the measured window, and the functional is `∫ c^2 dV` at a common end time `t_event + 0.08`. The
harness is `Scratchpad/bdf2_history_restart/kin.py`.

Error against the finest run (`bdf1` at `dt = 1.5625e-4`), and the order fitted per arm against its
own finest run. The `freeze_*` / `local_impulsive` / `eulerian` arms are the four per-node repairs
described below; they were measured with a temporary implementation that no longer exists:

| restart | `dt=0.02` | `0.01` | `0.005` | `0.0025` | `0.00125` | order |
| --- | --- | --- | --- | --- | --- | --- |
| **one BDF1 step** | **5.9e-5** | **1.5e-5** | **3.7e-6** | **9.3e-7** | **2.3e-7** | **2.00** |
| nothing at all | diverges | diverges | 8.9e-6 | 2.8e-6 | 7.9e-7 | 1.8 |
| `freeze_value` | diverges | diverges | 2.2e-5 | 6.7e-6 | 1.9e-6 | 1.8 |
| `local_impulsive` | 9.4e-4 | 3.5e-4 | 1.2e-4 | 3.5e-5 | 9.7e-6 | 1.4 -> 2.1 |
| `freeze_position` | 1.1e-3 | 3.9e-4 | 1.3e-4 | 3.9e-5 | 1.1e-5 | 1.4 -> 2.1 |
| `eulerian` | 1.1e-3 | 4.0e-4 | 1.3e-4 | 4.0e-5 | 1.1e-5 | 1.4 -> 2.1 |
| `assign_initial_values_impulsive()` | 7.9e-3 | 3.0e-3 | 1.0e-3 | 3.0e-4 | 8.3e-5 | 1.4 -> 2.0 |
| any of the above **+ one BDF1 step** | identical to `bdf1`, to twelve digits | | | | | 2.00 |

Reading it:

* **One BDF1 step wins on both counts here**, accuracy and robustness. It is the only arm that is
  second order over the whole ladder, and it is an order of magnitude more accurate than anything
  else at every step size. (On the real pinch its robustness does not survive - see the end of this
  section. This instrument has no capillary singularity in it.)
* **What the per-node repairs did buy is robustness, and only through the position history.** The arms
  that keep BDF2 on the first step and leave the fresh nodes' *motion* alone - "nothing at all" and
  `freeze_value` - fold the mesh across the symmetry axis at `dt >= 0.01`; the mesh velocity the
  transfer gave the cap simply carries it through `r = 0` on the first step. Freezing that motion
  (`freeze_position`, and the two modes that include it) removed the failure entirely, which is a
  real result and the one thing those modes were good for. It costs a factor of 20 to 35 in accuracy
  against degrading the step instead, and it does not carry over: on the real pinch,
  `--restart-mode none` with `freeze_position` reached the same fourteen post-event steps as
  `--restart-mode none` alone and then died of something else. Not worth keeping, and it was not
  kept.
* **`assign_initial_values_impulsive()` is the same idea done worse.** It also recovers second order
  asymptotically - it is also a single degraded step - but with a constant about 270 times larger
  than the BDF1 step's at `dt = 0.005`, because it throws away the accurate history of *every* node in
  the domain and not just of the handful the surgery created. It is not wrong, it is expensive - and
  §8.3 recommends it anyway, for the reason at the end of this section.
* **Doing nothing is the most accurate of the arms that keep BDF2 on the first step** - more accurate
  than any of the per-node repairs. That surprised, and it should not have: the fresh cap's material
  genuinely *was* moving, so the old material trajectory the transfer hands it is a better estimate of
  where it was one step ago than "it was standing still" is. What doing nothing is not, is robust.
* **The per-node repairs cannot win on accuracy, whatever they are**, for the shift reason above: the
  one history slot they influence is the one slot the BDF1 step discards.

**What the per-node repairs were.** Each of them marked the nodes the surgery had created - as balls
of a couple of `max(rmin, distmin)` around each fresh point of the plan (`NewChain.origin < 0`),
because the new mesh is built by Gmsh from a spline through those points and so has no node in common
with them, and the fresh *bulk* nodes inside a cap are not named anywhere at all - and handed the
region down to `Mesh::nodal_interpolate_from`, which treated a node inside it differently from
everything else. `freeze_position` set `x(t>=1) := x(0)`, `freeze_value` set `u(t>=1) := u(0)`,
`local_impulsive` did both, and `eulerian` froze the position and read the value history at the
node's *current* position in the old mesh's level-`t` configuration - a few Newton steps on
`interpolated_x(t, .)` inside the located element - so that the ALE derivative of a now-stationary
node was a genuine Eulerian `d/dt` at a fixed point. The table says `eulerian` and `freeze_position`
differ by under a per cent, i.e. the Eulerian correction to the value history bought nothing next to
what freezing the motion had already done; that Newton solve on the past configuration was the most
intricate part of the whole apparatus and it paid for nothing. `back-extrapolate the trajectory with
the local velocity` was never a separate mode: for these equations the old mesh moves *with* the
fluid, so the old material velocity at the located point is the fluid velocity there, and that arm is
the "nothing at all" row already.

Since every arm lost to one degraded step, all of this was taken back out again: `Mesh` has no
history-restart state, `nodal_interpolate_from` has no branch on it, and
`AxisymmetricReconnection`/`InternalInterpolator` have no arguments for it. Anyone tempted to build
it again should read the shift argument above first - the slot it can influence is the slot the BDF1
step throws away.

**And why §8.3 still says `assign_initial_values_impulsive()`.** The table above is a statement about
*accuracy*, measured where the time integration is the only thing that can go wrong. The
Rayleigh-Plateau pinch has something else that can: the two fresh caps retract at a speed the mesh
only just resolves, and `tests/axisymm_physics_worker.py --case rayleigh_plateau` completes its 20
post-event steps with the impulsive restart and dies with the BDF1 one - a cap node carried across the
symmetry axis about a dozen steps after the event ("an interface node sits at r = -0.0033, i.e. 0.36
local element sizes on the far side"). The dissipation that costs the impulsive restart its two orders
of magnitude is exactly what keeps that from happening, so it stays the default in the worker, in the
tutorial and in `dev_docs/examples/axisymm_reconnection_minimal.py`. `--restart-mode bdf1` on the
worker reproduces the failure, and `--restart-mode none` the one §8.3 describes.

So the honest summary is: **one BDF1 step is the right restart, and the impulsive one is the safe
one.** Take the BDF1 step when the post-event phase is resolved well enough to survive it - which is
something only the run itself can tell you - and the impulsive one otherwise.

**Where the fresh nodes actually are, per event type.** For a *pinch* the opened geometry is
essentially a subset of the old domain, so the fresh cap nodes locate in the old mesh and the located
branch of `nodal_interpolate_from` governs them - every mode is well defined there. For a
*coalescence* the bridge is built where there was no liquid at all, so its nodes cannot be located
and fall to the two-nearest-node blend instead, which blends the position history as well as the
values - so a fresh bridge node's history is that of the nearest old tip, motion included, and any
future attempt at repairing it would have to cover that branch too (the removed `eulerian` mode had
no old element to read a past configuration from there and degraded to freezing the position). That
is what `tests/test_axisymm_transferred_history.py` measures - deliberately a coalescence, since it
is the harder of the two.

---

## 9. Why the legacy classes were deleted rather than ported

`AxisymmetricPinchoffAndCoalescence` was 791 lines and had no tests and no tutorial.

* **Detection** was a spline fit of `r(arclength)` per interface segment, so it could only see a waist
  *within one* segment, never the merging of two, and the answer depended on the knot placement of the
  fit.
* **Surgery** inserted two hand-placed points per new tip and left the volume to chance.
* **Zeta continuation** was unimplemented — line 565 read "Here the pinch-off and coalescence dynamics
  has to go" — with a per-node `scipy.optimize.minimize` zeta fit elsewhere.
* Real defects: the coalescence gate read the radial instead of the axial mesh velocity (line 456), the
  opposite-domain axis indexing was broken (lines 691-693), and the velocity gate was identically zero
  (§2.3).
* It needed a `Remesher2d` (§1.1).

`DisjunctDomainMarker` is kept, unchanged.

---

## 9a. Wall-attached menisci: `reservoir_depth`

A `"fixed"` chain end is closed synthetically, and the closure decides what the morphology considers
to be liquid. Closing it straight across at the contact height is right for a drop cut by a symmetry
plane - there is nothing behind the contact - and wrong for a nozzle meniscus, where the liquid
continues up the nozzle: what the erosion then sees is the sliver in *front* of the interface. Two
ordinary meniscus shapes break on it, neither of them a topological change:

* **dimpled near the axis** - the sliver is thinner than `2 eps` and erodes away, so the reservoir is
  reported as a vanished fragment (and with `allow_fragment_removal` would be deleted from the mesh),
  or the erosion hollows it into a ring, whose mirrored cross section does not touch the axis at all
  and cannot be represented;
* **crossing the contact height** - the closure cuts the interface and the section self-intersects.

`AxisymmetricReconnection(reservoir_depth=...)` says how deep the liquid goes behind the contact line
and in which axial direction, and `InterfaceChain.reservoir` carries it into the plan. The closure
then runs along the wall by that much and only then across, so the enclosed body is the nozzle rather
than the sliver. It has to be **deeper than any excursion of the interface past the contact line**,
and its **sign cannot be inferred** - the same shape with the two phases swapped needs the other one.

Everything downstream uses the same closure, so no volume has to be corrected for it: `_closed_section`
grows the same stub, `_interface_curve` cuts the stub back off the run at the contact point (it runs at
`x > 0` and would otherwise come back as interface), and `NewChain.reservoir` is inherited from the old
chain that owned the end. Pinned by `tests/test_axisymm_reservoir_closure.py`.

The depth is also what the plan's `axis_spans_inside` uses at a fixed end. A chain end on a wall does
not touch the axis, so the span has to run to where the *closure* meets it - the contact height plus
the depth. Taking the contact height itself leaves the whole nozzle above the meniscus outside every
span, and `define_geometry` then cannot close the liquid's curve loop.

Independently of it, the opening now leaves a band of `2 eps` around a *flat* closure alone. That
costs no detection - an event within `4 eps` of a fixed end is refused anyway - and it is what keeps a
merely dimpled meniscus from being eroded away where no reservoir depth was given.

---

## 9b. Either mesh generator: `TopologicalChangesTemplate`

The surgery is geometry, and the calls `define_geometry` makes to build it - `spline_from_chain`,
`lines_from_axis_segments`, `plane_surface` - are spelled the same way by gmsh and by TQMesh. So the
shared half lives in `_TopologicalChangesMixin` and is composed into one class per backend:
`TopologicalChangesGmshTemplate` and `TopologicalChangesTQMeshTemplate`. `TopologicalChangesTemplate`
is the common type to test with `isinstance`, and what `AxisymmetricReconnection` requires.

*Composed*, not inherited: `MeshTemplate` is a nanobind type and nanobind refuses any class with two
bases, so the mixin's methods are copied into the concrete classes by `_with_topological_changes`,
which also registers them with the (abstract) `TopologicalChangesTemplate`. Each backend supplies only
what it must - an `__init__`, a `_reset` and a `point` in its own signature, routed through the shared
`_snapped_point`/`_register_snapped_point`.

The one place the two genuinely differ is what a chain of points means. **gmsh** takes them as the
geometry and discretises it by size afterwards; **TQMesh** takes them as the boundary edges
themselves. `get_boundary_coordinates` walks the boundary *nodes*, of which a second-order mesh has
two per element, so handing all of them to TQMesh doubles the boundary at every remesh and multiplies
the element count by four - measured on a plain unit square, 40 boundary edges became 80, 160, 320 and
589 elements became 30850 over three quality remeshes. `TopologicalChangesTQMeshTemplate` therefore
resamples: the chain becomes the *control* points of the spline and TQMesh distributes the boundary
along it by the size function, which is fed the plan's per-point sizes for the duration of the call.
With that, four consecutive remeshes of the same blob stay at 1523/1527/1503/1503/1505 elements.
(This is not specific to the surgery: any `TQMeshTemplate` that rebuilds a boundary from
`get_boundary_coordinates` needs `spline(..., resample=True)`, which its class docstring now says.)

`tests/test_axisymm_reconnection_tqmesh.py` runs the detection, the pinch-off and the coalescence
through the TQMesh backend; `tests/axisymm_reconnection_worker.py --backend tqmesh` is how.

---

## 10. Limitations

**Refused, with a clear message.**

| | why |
| --- | --- |
| a cross section that encloses a **hole** (entrapped opposite phase) | the plan has no representation for it; `plane_surface` does support holes, so this is an extension path rather than a wall |
| an event within `4 eps` of a **fixed end** | the synthetic drop to the axis that closes a "fixed" end would be inside the event window |
| a waist **axially shorter** than the structuring element | §2.2(i) — deferred, not fatal, for `max_deferred_events` solves |
| a pinch gap shorter than `distmin` | §2.2(ii): the closing would re-bridge what the opening just opened |
| a bulk template that is not a `TopologicalChangesTemplate` | the surgery is delivered by re-running `define_geometry` |
| a wall-attached interface without `reservoir_depth`, once it dimples or crosses the contact height | §9a - the flat closure is not the body |

**Approximate or absent.**

* **`rmin` too coarse relative to the drop is not diagnosed as such.** It fails, but late and in the
  splicing rather than in the morphology: "the corrected interface is self-intersecting" or "a new
  fragment retains no old interface point; the event windows are too wide". Measured on the
  Rayleigh-Plateau profile at `R0 = 1`, `eps_p <= 0.08 R0` worked at every step past the threshold,
  `eps_p >= 0.10 R0` never worked. A rough rule from that single scan: `rmin` below a tenth of the drop
  radius, and `cap_window_factor * rmin` comfortably shorter than the fragment it sits on.
* **Refining the interface by its own RADIUS is not the answer, and is dangerous.** The obvious
  reaction to a cap the mesh cannot resolve is to make the element size follow the local radius
  (`size = x/k` on the interface). Measured on the printhead: it refines everywhere the interface
  approaches the axis - both caps, the ligament tip, the meniscus tip - and with a low
  `Mesh.MeshSizeMin` it does so without bound. The mesh went from 30 000 to **two million** equations
  in a few steps and the machine ran out of memory. It also destabilises the phase *before* the pinch,
  where the finer elements at an entrained gas wedge stall the Newton solve. Curvature refinement on
  the surgery mesh alone is the cheap, bounded version of the same idea.

* **Continuing far past a pinch is limited by the resolution of the fresh caps, not by the surgery.**
  The Rayleigh-Plateau test asserts twelve accepted steps past the event. It reaches 18 or 19 and then
  stops, always the same way: the satellite filament's two caps retract into a tip that `h_min = 0.04
  R0` cannot resolve, its interface polyline becomes non-simple, and the next detection refuses the
  mirrored cross section as self-intersecting - or, one step further, Newton gives up. No plan is
  involved, no element is inverted at that point, and the volume is still good to 1e-4. Refining until
  it goes away costs more than the whole test does. Practical reading: after a pinch, either give the
  caps a mesh size of their own or stop before they have retracted a few element sizes.

* **A fresh cap needs the mesh to follow its CURVATURE, and to be ALLOWED to.** The cap the surgery
  builds has a radius of about `rmin`, and a second-order triangle whose curved edge wraps around
  something smaller than itself is inverted at an integration point - `det(dx/ds) < 0` right on the
  axis at the cap, at the pinch height. The run then dies on a fold that neither a smaller `dt` nor a
  remesh can undo, since the same geometry rebuilds the same element; with
  `RemeshingOptions(on_inverted_element=True)` it is the "kept folding after 3 remeshes" message. The
  inverted-element report names the position, which is how this was found rather than guessed.

  Two settings, and **both** are needed. Measured on the dumbbell at `rmin = 0.025`, `h = 0.05`:

  | `Mesh.MeshSizeMin` | `MeshSizeFromCurvature` | |
  | --- | --- | --- |
  | none | off | folds |
  | none | 8 | ok |
  | `rmin/2` = 0.0125 | 8, 12, 20 | **folds at every one of them** |
  | `rmin/3` = 0.008 | 8 | folds |
  | `rmin/4` = 0.00625 | 8 | ok |
  | `rmin/5` = 0.005 | 8 | ok, also with a background size field |

  So the floor decides whether gmsh may follow the curvature at all, and it turned out to matter more
  than the curvature setting: at `rmin/2` no amount of curvature refinement helps. **Halving `h`
  everywhere does not help either** - this is curvature, not resolution. The cost of getting it right
  is about 10 % more elements. A background mesh size field does not interfere.

* **Satellite-fragment removal loses volume by definition.** `allow_fragment_removal=True` drops
  fragments the opening deletes entirely; the loss is reported in the event print and in
  `plan.volume_lost_by_removal`.
* **DG / DL / D0 interface fields remain untransferable** across any remesh (`mesh.cpp` refuses them).
  Unchanged limitation, not specific to this feature.
* **`DisjunctDomainMarker` is not MPI-capable.** Its flood fill walks the elements *this rank* holds, so
  each rank numbers the components of its own partition from zero. Unlike the detection this cannot be
  fixed by merging — the marker is written back onto local elements — and it would need the component
  labels reconciled across partition cuts.
* **One handler per template.** The plans are filed on the template keyed by interface name, so two
  interfaces of the same bulk domain each carrying an `AxisymmetricReconnection` would both request the
  same remesh; only the first is consumed. Untested and unsupported.
* **The rank-local nearest-node blend** of §5.
* **2D axisymmetric only.** The whole geometry pipeline is a half-section mirrored about a line.
