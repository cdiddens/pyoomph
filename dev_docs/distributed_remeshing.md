# Remeshing under `--distribute`

Status: **works, with no opt-in.** `mpirun -n N python3 script.py --distribute` remeshes for both
remeshing paths — recreation (`RemesherViaRecreation`, the default on any `MeshedMeshTemplate`) and
the automatic `Remesher2d` — with `InternalInterpolator`, across codimension-2 interfaces (contact
lines, axis points) and on boundaries parameterised by `zeta` by either assigner. What is still
refused is named below; lifting it is §3.6.

The two paths differ enough to be treated separately throughout:

* **recreation** — the user's `define_geometry` runs again and rebuilds the geometry, usually from
  `MeshedMeshTemplate.get_boundary_coordinates`;
* **`Remesher2d`** — the geometry is reconstructed automatically from the deformed mesh's boundary
  nodes.

| refused | why | see |
| --- | --- | --- |
| `ProjectionInternalInterpolator` | its projection integrates the old field at partition-local locations | §3.6 |
| remeshing only *some* domains | the untouched ones stay partitioned, and oomph cannot distribute a mesh twice | §3.2 |
| explicit `num_adapt > 0` | adapting leaves the new mesh non-uniformly refined, which `distribute()` rejects | §3.6 |

The default `num_adapt` (from `max_refinement_level`) is dropped to 0 with a printed note rather than
refused, since the caller did not ask for it by name. `Problem.experimental_distributed_remeshing`
bypasses every refusal, for developing what is left.

**Two behaviour changes to know about**, both explained where they arose:

* `Remesher2d` reconstructs its geometry in a canonical order now, so a **serial** script will not
  reproduce its old node numbering (§3.5). The domain, its named boundaries and the mesh quality are
  unchanged.
* `define_geometry` is a **collective region** on a distributed mesh — do not branch on the rank
  inside it (§3.1).

---

## 1. What used to happen

Kept because it is what the tests exist to prevent coming back. Measured on a quarter disc meshed by
Gmsh, `initialise()` then `force_remesh()`, at 2, 3 and 4 ranks.

* **Recreation at 2 ranks: silently wrong geometry.** `get_boundary_coordinates` asked for the mesh
  data *without* `global_mesh`, so each rank got only its own partition of the interface (rank 0: 19
  points from (0,1) to (0.92,0.38); rank 1: 7 points from there to (1,0)) and described a different
  geometry to Gmsh. `generate_mesh_to_file` writes the `.msh` through `run_on_rank_zero` and every
  rank then loads *that one file*, so **rank 0's truncated wedge became the mesh on all ranks** — the
  reconstructed geometry stopped at 67.5° and closed the domain with a chord. Nothing raised.
* **Recreation at 3 and 4 ranks: deadlock.** A rank whose partition holds no element of the requested
  boundary got `RuntimeError: No elements in mesh. Cannot convert to numpy.` and unwound out of
  `define_geometry` while the others walked into the `mpi_barrier()`/`run_on_rank_zero` pair inside
  `generate_mesh_to_file`. At three ranks it is **rank 0** that owns no interface elements, i.e. the
  merge root and the rank that writes the `.msh` is exactly the one that cannot see the boundary.
  Not a corner case; it is what partitioning a domain by element count normally does.
* **Both: the distribution lost, `ndof` multiplied by `nproc`.** The new mesh was built in full on
  every rank and never distributed, while oomph's `Problem_has_been_distributed` stayed true, so
  `assign_eqn_numbers` counted every locally-owned node once per rank. The sharpest single symptom,
  and it makes a good assertion.
* **Both: the value transfer never crossed a rank boundary.** `MeshPointLocator` contains no MPI code,
  so a quarter of the nodes fell back to nearest-node blending *over local nodes only* — a plausible
  wrong value rather than a failure.
* **`Remesher2d` failed before any of the above**, with `Cannot close line loop for surface domain`:
  its walk over `mesh.boundary_elements(bn)` sees a partition bounded partly by named boundaries and
  partly by the partition cut, and the cut carries no boundary name, so the curves can never close.

---

## 2. What already existed

Most of the reading half is done. `pyoomph/meshes/meshdatamerge.py` merges the per-rank mesh data of a
distributed mesh into one entry on rank 0, **including interface meshes** (through the `-1`-padded
shared-node scheme of `Mesh::get_shared_node_numpy_indices`), and `tests/test_mpi_global_meshdata.py`
already asserts that the merged `get_interface_line_segments()` reproduces the single segment a serial
run sees. `dev_docs/mesh_data_cache.md` §10 lists "remeshing boundary identification on the
merged data" as the one remaining consumer of that machinery; this campaign is that item.

Ranks holding no element of a mesh are handled there too: `_local_payload` returns `None` for them
instead of calling `to_numpy`, so routing the deadlock of §1 through the merge removes that crash at the source.

One structural unknown was worth settling before planning anything: **`Problem::distribute()` can be
called a second time, after a remesh.** Spike, two ranks, `distribute()` invoked by hand on the
replicated post-remesh mesh:

```
after              rank=0 nelem=47 nhalo=0 mesh_distributed=False ndof=422
after_redistribute rank=0 nelem=32 nhalo=9 mesh_distributed=True  ndof=211
after_redistribute rank=1 nelem=32 nhalo=8 mesh_distributed=True  ndof=211
```

Both of oomph-lib's preconditions hold: `Base_mesh_element_pt` and
`Base_mesh_element_number_plus_one` are rebuilt from scratch on every call
(`src/thirdparty/oomph-lib/include/problem.cc:896`), and a freshly generated mesh is uniformly
refined, which is what `Problem::distribute` insists on (`problem.cc:650`).

---

## 3. How it works now

### 3.0 Making the failure visible and local

Nothing here makes remeshing work; it makes the failure visible and local.

* `Problem.force_remesh()` refuses on a distributed problem, with a message naming what each attached
  remesher would get wrong. `Problem.experimental_distributed_remeshing = True` bypasses it, which is
  how the later stages are developed and how anyone who wants the old (wrong) behaviour back gets it.
  The check sits after the "is there anything to remesh at all" test, so a `remesh_if_necessary()`
  that finds nothing to do still returns quietly on a distributed problem.
* `MeshedMeshTemplate._do_define_geometry` agrees across ranks on whether `define_geometry` succeeded
  (`get_mpi_any`, symmetric rather than rooted at 0 - the rank that fails is usually *not* rank 0, see
  §1). A raise on one rank now ends the job with an error on every rank instead of hanging in the
  next collective. This covers the initial mesh generation as well, where the same hang was possible.

  It can only catch exceptions, not hangs: a `define_geometry` whose ranks disagree on how many
  collectives to enter still deadlocks inside them. §3.1 makes `define_geometry` collective, so
  that contract has to be documented there.
* Tests: `tests/test_mpi_remeshing.py` (§4).

### 3.1 Global boundary coordinates (the recreation geometry)

`get_boundary_coordinates` asks for `global_mesh=True`, sorts the segments on rank 0 and
**broadcasts the resulting polylines**. Broadcasting the sorted result rather than the merged entry
keeps the payload to a few hundred points and leaves `meshdatamerge.py` untouched. Serial and
`mpirun` without `--distribute` take the old path verbatim, guarded by `needs_merging()`, so no
communication happens there at all.

Details that are not optional:

* The broadcast lives in `get_boundary_coordinates` and is unconditional. It must *not* rely on the
  request scope of `meshdatamerge.py` §3.4b: a cache **hit** on rank 0 broadcasts nothing there, so a
  repeated call would leave the other ranks in a gather nobody joins.
* Only the nondimensional floats travel. The spatial scale is applied on every rank after the
  broadcast, so a dimensional scaling (a GiNaC expression) never has to be pickled.
* Resolving the mesh name is agreed on across ranks *before* the merge
  (`_resolve_mesh_for_boundary_coordinates`), and rank 0's sorting is agreed on *after* it through
  `mpi_share_root_failure`. Both are sections one rank could otherwise leave alone, straight into a
  collective the others are already in.

Measured on the reproducer: the reconstructed `.geo_unrolled` is now **byte-identical to the serial
one** at 2, 3 and 4 ranks, and no rank rebuilds a truncated arc any more.

**What this costs.** `define_geometry` is now a collective region. Every rank has to reach the same
`get_boundary_coordinates` calls, with the same arguments, in the same order; user code that branches
on the rank inside `define_geometry` deadlocks. This is documented on `MeshedMeshTemplate`.

It also *narrows* what §3.0's guard can catch, which is worth stating plainly because it is a
genuine regression in robustness: before §3.1 the first collective came after `define_geometry`, so
any raise inside it was catchable. Now a rank that raises *before* reaching
`get_boundary_coordinates` never joins its merge, and the others hang inside it - the guard is never
reached. Nothing short of timeouts can catch that, which is exactly why the "do not branch on rank"
contract has to be a contract. The test for the guard therefore injects its failure *after* the
collective (`mpi_remeshing_worker.Disc.fail_on_rank`), which is the case that can be caught: user code
that gets through the collectives and then produces something invalid on one rank.

**What is left over for §3.2.** With the geometry correct, the remaining damage is cleanly
isolated. Every rank now builds the *right* mesh (269 nodes, 60 elements - the same one the serial
remesh produces), and the only thing wrong is that it is replicated:

| ranks | nodes per rank | `ndof` | serial `ndof` |
| --- | --- | --- | --- |
| 2 | 269 | 538 | 269 |
| 3 | 269 | 807 | 269 |
| 4 | 269 | 1076 | 269 |

i.e. exactly `nproc` x the true count, confirming the `nproc` multiplier of §1 at three partition counts rather than one.

### 3.2 Re-distributing after the remesh

`force_remesh()` builds the new mesh replicated on every rank (which is exactly what `initialise()`
does before its own first `distribute()`), transfers the old solution, and then runs
`actions_before_distribute()` / `distribute()` / `actions_after_distribute()` -
`_redistribute_after_remeshing()`, called after `remove_macro_elements()` and before
`actions_after_remeshing()`, so that user code sees the mesh in its final, partitioned state.

Measured: `ndof` after the remesh is now 437 at 1, 2, 3 and 4 ranks, where it used to be
437 x `nproc`, and every rank holds a partition with halos.

Two things had to come with it:

* **The base element numbers.** `Mesh::assign_global_base_element_indices` can only run while the
  mesh is whole, and `BaseMesh._define_state_file`'s lazy assignment deliberately stands down on a
  distributed mesh - so without assigning them here, saving a state after a distributed remesh fails
  with "The mesh has elements without a global base index". Verified by removing the call: it does.
* **Both preconditions refused up front.** oomph's `distribute()` needs a whole mesh, uniformly
  refined. Neither can be checked at the point where it is used: `force_remesh()` has no way back
  once it starts replacing meshes, and raising there leaves the problem half rebuilt - which does not
  even survive interpreter shutdown (it segfaulted at exit while the partial-remesh check still sat
  in `_redistribute_after_remeshing`). Both are therefore decided in
  `_check_distributed_remeshing_scope()`, before the first mesh is touched, and asserted rather than
  checked at the distribution.

What those two refusals are:

* **Remeshing only some domains.** A domain that is not remeshed stays partitioned from before, and
  `Mesh::distribute` builds a partition of a whole mesh rather than re-partitioning a split one. The
  message names the domains left behind.
* **Adapting the new mesh.** `num_adapt > 0` leaves it non-uniformly refined. Unlike at startup
  (`_defer_uneven_initial_refinement`) there is nothing to defer, since the refinement comes from the
  error estimate on the new mesh rather than from a level that was asked for. An **explicit**
  `num_adapt` is refused; the default (`None`, i.e. `max_refinement_level`) is dropped to 0 by
  `_remesh_adaption_steps()` with a printed note, since the caller did not ask for it by name. The
  note is not behind `is_quiet()` - it changes the mesh that comes out.

Running the interpolation replicated is redundant work, but it is correct, it matches the startup
flow, and peak memory is no worse than at startup, where the whole mesh exists on every rank anyway.
Making it scalable, and lifting the adaption refusal, belongs to §3.6.

### 3.3 Cross-rank value transfer

The old mesh is distributed; the new one is replicated and **identically numbered on every rank**,
which is what makes the cheap route work: each rank transfers whatever it can find in its own share
of the old mesh, and `Mesh::share_interpolation_across_ranks` then pools the results - one
`MPI_Allreduce` of the values each rank could fill (pre-multiplied by whether it could) and one of
how many ranks could, so the value is the sum over the ranks that had it divided by their number.

It runs *before* the nearest-node fallback, not after. A node this rank could not place is usually
one that simply lives in another rank's partition, and blending it from local nodes first would
produce a confident wrong value that the pooling could no longer tell from a real one. Nodes rescued
from another rank are moved out of `missing_nodes`, so the blend - and the count it reports - are
left with the nodes that are genuinely outside the old mesh everywhere. The two per-node `cerr`
diagnostics are suppressed on this path for the same reason: they would fire for most of the mesh and
mean nothing.

What travels is exactly what the transfer writes: nodal values at every time level, the position
history, the Lagrangian coordinates when they are interpolated, and the DL/D0 internal data of the
elements (whose element-centre query fails and succeeds independently of the nodes around it).

Two things about the collective were not obvious:

* **The decision to pool has to be collective too.** `nodal_interpolate_from` returns early when
  either mesh has no elements, and on an interface mesh a rank holding no part of it hits exactly
  that - so it skipped the `Allreduce` while the others entered it, and the three-rank run hung. The
  gate is now a Problem-wide flag (`distributed()` and the process count), inside which the local
  answer is `Allreduce`d, and a rank with an empty source still joins the pooling and contributes
  nothing.
* **The buffer layout has to be derived from the destination**, which is the replicated mesh, so
  every rank packs the same lengths in the same order - including the rank that has nothing to say.

Measured against a serial run of the same problem, on merged global mesh data: identical node count
and field extrema, and sums agreeing to ~1e-9 relative at 2, 3 and 4 ranks - the residue being the
projection solve, which runs on a partitioned matrix here and a whole one serially, i.e. it differs
before the transfer even starts. **Zero** nodes fall through to the blend, where a quarter of them
used to.

`ProjectionInternalInterpolator` does not do any of this - its right-hand side integrates the old
field at the new mesh's integration points, which are located the same partition-local way - so it is
refused, see below.

### 3.3a The refusal, narrowed

With §3.1–§3.3 in place, `force_remesh()` on a distributed problem **works** for
`RemesherViaRecreation` together with `InternalInterpolator`, and needs no opt-in. What was refused
from here on is listed in the table at the top; each refusal names itself, and
`Problem.experimental_distributed_remeshing` bypasses all of them.

### 3.4 Codimension-2 interfaces

Contact lines and axis points are ordinary in real scripts - both droplet tutorials have one - and
they are transferred by `nodal_interpolate_along_boundary`, a different mechanism that the §3.3
pooling does not reach: nearest-node matching along the boundary rather than point location.

The difference that matters is that **every rank produces an answer for every node** there. It
matches against the nearest of *its* old boundary nodes, however far away that is, so this is not
"who found it" but "who found it closest". One `MPI_Allreduce` with `MPI_MINLOC` over the match
distances picks the owner - ties broken by the lower rank, which the nodes on a partition boundary
need - and only that rank's blend is kept, through the same
`pool_node_values_across_ranks` that §3.3 uses.

Three things fell out of it:

* A rank can hold **no part of the old corner at all**, which is ordinary under distribution and used
  to be a hard error: with no old nodes to match against, `bestnode` stayed null and the routine
  threw "Found a node on a boundary that is not a boundary node" - on one rank, while the others
  waited in the new collective. Such a rank now offers no match and contributes nothing. Serially an
  empty old boundary is still a real problem and still reports itself.
* The destination node list for a codimension-2 boundary was collected through a `std::set` of node
  **pointers**, whose order differs between processes. The pooling addresses nodes by position in
  that list, so it is now built in element/node order.
* The "matched implausibly far away" warning only means something after the pooling, and its
  threshold (two old interface element lengths) is itself per-rank. Both the count and the threshold
  are now global, and only rank 0 prints them.

Verified against a serial run with equations on `domain/interface/axis`, at 2, 3 and 4 ranks: same
node count, same field extrema, sums agreeing to ~1e-9.

### 3.4a Zeta coordinates

`AssignZetaCoordinatesByArclength` and `AssignZetaCoordinatesByEulerianCoordinate` refused a
distributed mesh outright (`_refuse_if_distributed`), which blocked the two moving-interface tutorials
that use them. Zeta is worth having distributed for its own sake: where it is defined, the transfer
locates through the chart instead of falling back to the nearest-node blend at all.

The two need very different amounts of work, which is the whole point:

* **By Eulerian coordinate** - zeta *is* a nodal coordinate, so a rank assigning it to its own nodes
  produces exactly the serial values and the chart is global by construction. Nothing had to be
  merged; only the "is this degenerate" test spans the interface, and it is now an `Allreduce` (a
  rank whose share is a short stretch, or none, would otherwise call the whole boundary degenerate).
* **By arclength** - a property of the whole curve, so it has to be measured on the whole curve. The
  computation is split from the assignment (`_compute_zetas`), run on the merged interface on rank 0,
  and broadcast as `(position, zeta)` pairs; each rank then assigns to the nodes it holds, matching by
  position through a k-d tree. Addressed by position and not by index deliberately: the merged data
  numbers the points of the whole interface, which says nothing about this rank's node order. Every
  local node *is* one of the merged points, so a match further away than 1e-8 of the mesh extent means
  the table does not describe this interface, and that is an error rather than a nearest neighbour
  worth taking.

`_check_zeta_is_invertible` stays local, and correctly so: it tests that the elements this rank holds
tile their own stretch of zeta without overlapping, which is as meaningful on a partition as on the
whole interface - disconnected stretches only ever leave gaps, never overlaps.

Verified at 2, 3 and 4 ranks against serial, both through the worker (same field statistics as the
other transfer paths) and on `docs/source/tutorial/ale/beads_on_string.py`, which runs to completion
distributed with the same three remeshing events as serially, the ranks in step, and a minimum radius
tracking the serial one to ~5e-5 over the whole transient.

### 3.5 `Remesher2d` from the merged boundaries

The automatic remesher reconstructs the geometry from the boundaries of the mesh it replaces, walking
the local `Node` objects. A partition is bounded partly by named boundaries and partly by the
partition cut, which carries no boundary name at all, so the curves could never close - §1.

`Remesher2dBoundaryLineCollection` now works on `RemeshBoundaryPoint`, a point addressed by its
position rather than a node pointer, and the boundaries are collected in three steps: each rank
lists its share as a point table and edges, one `allgather` puts the shares together, and the copies
of a node that several ranks contributed are fused by position (k-d tree, tolerance far below any
distance between two distinct nodes). Serial takes the same path - the merge is a fixed point for a
single contribution - so there is one code path rather than two.

Three things it turned up:

* **`_ptsizes` was dead.** The plan expected the node-keyed size dictionary to be the hard part.
  Nothing has ever written to it; `get_size_at_point` was its only reader. Removed. The sizes that do
  matter are the element sizes, which are now accumulated onto the point while it is collected
  (`sum sqrt(initial)`, `sum sqrt(current)`, count) and summed across the ranks, so the average is
  over all the elements around the point rather than over this rank's.
* **Halo elements had to be excluded.** oomph's boundary lookup keeps them, and letting a halo
  contribute would give its nodes the same element twice in that average.
* **The order had to become canonical.** `split_into_curves` starts its walk at the first edge it
  meets, which fixes the direction of the curve and with it the order the gmsh points are created
  in - and gmsh meshes a differently numbered geometry differently. Concatenating the ranks'
  contributions makes that order depend on how many ranks there are, and 3 ranks gave a different
  mesh from 2 and 4. The edges are therefore sorted by position.

  **This changes serial results.** The reconstructed geometry is the same domain with the same named
  boundaries, but some lines come out in the opposite direction and the curve loop with the opposite
  orientation, so gmsh produces a slightly different triangulation than before. Nothing about the
  mesh is worse - and a remesh regenerates the mesh anyway - but a `Remesher2d` script will not
  reproduce its old node numbering. The alternative was to keep the element order serially and use
  the canonical one distributed, i.e. to give up on "distributed reproduces serial", which is the
  property everything else here has.

Verified against serial at 2, 3 and 4 ranks through the worker (identical node count, field
statistics agreeing to ~1e-9, and at 3 ranks bit-identical), and on
`docs/source/tutorial/ale/remeshing.py`, which runs distributed with the same three remeshing events
as serially.

#### The bug underneath it

The tutorial deadlocked the first time, and not in the remesher: one rank remeshed while the other
went on to write its output, and the next collective paired the boundary `allgather` with the mesh
file output's - which surfaced as `TypeError: '>' not supported between instances of 'dict' and 'int'`
deep inside `meshio.py`.

`Problem._agree_on_domains_to_remesh` matched the requests against `_meshtemplate_list`. A
`Remesher2d` hands the problem a *new* template (`GmshRemesher2d`, its `get_new_template()`), and
`RemeshWhen` asks by the template its mesh currently carries, so from the first such remesh onwards
the request named something that list did not contain. It fell through to the "keep it rather than
drop it" branch, which preserves exactly the asymmetry the method exists to remove. The candidate
list now also contains the templates the current meshes carry (`_meshdict` is built in the same order
on every rank, so it stays rank-independent), and a request that still cannot be matched says so
instead of quietly desynchronising the run.

This was only reachable once `Remesher2d` could run distributed at all: `tests/test_mpi_rank_zero_failures.py`
covers the agreement, but without `--distribute`.

### 3.6 OPEN: the MPI point locator

The MPI phase of `dev_docs/mesh_point_locator.md`: give `MeshPointLocator` its MPI routing layer. The API
was designed for it (`LocationHandle::owner`, the reusable `LocationSet` schedule, the batched
`evaluate`), and §5 there carries the cost model. It would make remeshing *scale* rather than merely
work, and it is what the two remaining refusals are waiting for.

Scoped but not started, and the scoping is the useful part:

* **The foundation is there.** The locator already keeps a per-element bounding box
  (`element_bbox_min/max`), so the per-rank boxes the routing needs are a union away.
* **An `Allgatherv`-only version is not worth building.** Routing every query to every rank leaves
  each rank searching every query - the same O(N) work §3.3 already does. All of the speed is in
  the bounding-box routing, so a "correctness first, optimise later" split buys nothing here.
* **Nothing is observable until the Python side moves too.** The routing layer's first consumer is a
  transfer from an old distributed mesh to a *new distributed* one, and `force_remesh` currently
  interpolates while the new mesh is still replicated. Both have to land together.
* **The new mesh is replicated for a reason that §3.6 does not remove.** Gmsh generates the whole
  mesh on every rank, exactly as at startup; what §3.6 removes is holding it replicated through
  the *transfer*, not the generation.

The two refusals it would lift are narrow: `num_adapt > 0` (adapting the new mesh) and
`ProjectionInternalInterpolator`. `num_adapt` could also be lifted without any of this - interpolate
replicated as now, distribute, then let the adaption rounds use oomph's own father-to-son
interpolation instead of re-reading the old mesh - at the cost of some accuracy on the nodes the
adaption creates. That is a smaller, self-contained piece if the refusal ever becomes a nuisance.

## 3.7 Validated on real scripts

Every tutorial the campaign touches - the ones that remesh, move their mesh, or parameterise a
boundary by zeta - run under `mpirun -n 2 ... --distribute`:

| tutorial | what it exercises | result |
| --- | --- | --- |
| `ale/remeshing.py` | `Remesher2d`, `RemeshWhen`, `RemeshMeshSize` | 3 remeshing events, as serially |
| `ale/beads_on_string.py` | zeta by arclength and by Eulerian coordinate | 3 events, as serially |
| `ale/rayleigh_plateau.py` | zeta, and an extremum observable driving the time step | 15 events, ranks in step |
| `multidom/evaporating_water_droplet.py` | two coupled domains, contact line, recreation | 1 event, volume as serially |
| `plotting/evaporating_water_droplet.py` | the same, plus plotting | 1 event |
| `advstab/movmesh/hanging_droplet.py` | remeshing plus bifurcation tracking | remeshes, then stops on `Bifurcation tracking is not supported on a distributed (--distribute) problem yet` - a refusal that predates this campaign and has nothing to do with remeshing |

The detail behind the droplet row - the stages above were all built against the synthetic quarter disc. The first end-to-end check on
something real is `docs/source/tutorial/multidom/evaporating_water_droplet.py` - two coupled domains
(droplet and gas) rebuilt by one template, a moving mesh, a contact line (codimension-2), and
remeshing triggered by `RemeshWhen` rather than by an explicit `force_remesh()`. Run at
`default_resolution=0.05` to stay under the 40000-dof MPI guideline (14520 equations, 4.5 s serially),
for the full 100 s, which remeshes once.

The droplet volume, an integral observable over the whole domain, after the remesh:

| | volume at t=100 s |
| --- | --- |
| serial | 3.309166529745143e-11 |
| 2 ranks | 3.3091665297536237e-11 |
| 3 ranks | 3.309166529546143e-11 |
| 4 ranks | 3.309166526057907e-11 |

i.e. agreeing to ~1e-9 relative at four ranks and better below that, over a transient that accumulates
the difference between a partitioned and a whole linear solve. This is also the first exercise of the
two-domain path: both domains are rebuilt by the same template, so the partial-remesh refusal does not
fire, and `distribute()` re-partitions them together with their coupled interface.

---

## 4. Tests

`tests/test_mpi_remeshing.py` + `tests/mpi_remeshing_worker.py`, marked `slow`, 24 tests, ~54 s.
Following the `tests/mpi_*_worker.py` pattern: a serial in-process run as the reference, and
numbering-independent comparisons (node and element counts, a digest over sorted coordinates, field
sums), at 2, 3 and 4 ranks. Everything except the refusals runs **without** the opt-in, i.e. in the
configuration that ships.

What each covers, and why:

* **The boundary every rank rebuilds matches the serial one** — as a digest over the polylines. The
  three-rank case is the one that matters most: rank 0 owning no interface element is what used to
  hang.
* **`is_mesh_distributed()` true and `ndof` equal to the serial count** after the remesh. The `nproc`
  multiplier of §1 makes that a single decisive assertion.
* **Field values after remeshing equal to serial, with zero "could not be located" fallbacks**, for
  each of the three transfer mechanisms (point location, codimension-2 nearest-node, zeta) and for the
  `Remesher2d` path. Sums are weighted by position too, so values landing on the wrong nodes cannot
  cancel out.
* **The collective agreement when one rank raises inside `define_geometry`** — rank 0 or rank 2, since
  the agreement is symmetric.
* **`mpirun` without `--distribute` unchanged** at 2 and 4 ranks. `needs_merging()` already
  short-circuits that case, but it is the regression that would go unnoticed.
* **Both refusals**, on **every** rank.

Every run is under a timeout that fails rather than waits, so a regression in the agreement shows up as
a failed test instead of a stuck suite — it has already caught one (see the end of §3.1). The
partial-remesh test additionally asserts the exact return code, because the bug it pins down was a
segfault *after* a correct error message.

---

## 5. Found on the way, and fixed

Two things surfaced while running the tutorials distributed. Neither has anything to do with
remeshing, and both made a distributed run of an otherwise working script produce nonsense:

* **`ExtremumObservables` was not reduced across the ranks.** `Mesh::evaluate_extremum` samples the
  elements this rank holds, so each rank reported the extremum of its own partition. In
  `docs/source/tutorial/ale/rayleigh_plateau.py` that observable *drives the time step*
  (`dt = 0.1 * r_min`), so the ranks picked different steps and marched to different times - the run
  did not fail, it silently stopped being one simulation.

  `evaluate_maximum`/`evaluate_minimum` are now collective: one `allgather` of the local extremum and
  its position, and `min`/`max` over the gathered list, which settles a tie by the lowest rank and
  does so identically everywhere. A rank holding no element of the mesh contributes -inf or +inf and
  needs no special case.

  The one subtlety is the **unit**. The value used to come back from C++ already dimensional, and a
  rank with no element has nothing to read the unit off - and a GiNaC expression cannot be sent in
  its place. It now comes from `FiniteElementCode::get_extremum_expression_unit_factor`, i.e. from
  the registered expression rather than from the evaluated value, so only the number ever travels.
  Verified equal to the old source exactly, including the decomposition's remainder being 1.
* **`Problem.create_text_file_output` had every rank write the same file.** The rows interleaved
  mid-number: `beads_on_string`'s `minimum.txt` came out as `10.0\t0.8511691410.0\t0.85116914...`.
  Only rank 0 writes now; the other ranks keep a working object whose writes are dropped, so the same
  script runs serially and distributed without asking about the rank.
  `only_on_rank_zero=False` restores the old behaviour for genuinely per-rank rows.

With both, `rayleigh_plateau.py` runs distributed at 2 and 4 ranks: the same 226 steps as serially,
the ranks at identical times, and a trajectory agreeing with the serial one to within that script's
own run-to-run spread (two identical serial runs already diverge at step 148 near pinch-off, by about
as much). Tests: `tests/test_mpi_observables.py` (10 tests, ~18 s).

## 6. Open questions

* `ParametricGmshMeshRemesher2d` subclasses `Remesher2d` but rebuilds its geometry from problem
  parameters alone, so it needs none of §3.5. Untested distributed.
* **Reading a state back** after a distributed remesh. Writing one works and is covered (§3.2
  assigns the base element numbers for exactly that), but `dev_docs/distributed_state_files.md` §7
  says sharded state files cannot be read at all yet, and nothing here has tried a
  write-then-restart across a remesh.
* The **3d** remeshers. Everything above is 2d: `Remesher2d` by name, and the boundary curves the
  recreation path rebuilds are polylines. A 3d surface has no `get_interface_line_segments`
  equivalent, so §3.1 does not carry over as it stands.
* `RemeshWhen`'s criterion is judged per rank and made unanimous afterwards
  (`_agree_on_domains_to_remesh`), so a distributed run remeshes as soon as *any* rank's elements are
  distorted enough. That is the safe direction, but it means a distributed run can remesh more often
  than the serial one it is compared against.
