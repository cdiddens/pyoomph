# Remeshing under `--distribute`

Status: **stages 0 to 3 done.** Remeshing a distributed problem now works for the recreation path
(`RemesherViaRecreation` + `InternalInterpolator`) and needs no opt-in. Stages 4 and 5 are open, and
what they cover is refused by name - see "The refusal, narrowed" below for the list. File and line
references are to the tree state at the time of writing (branch `develop`, after `7b8770a`).

Remeshing a distributed problem did not work at all when this started, and it did not *say* so: at
two ranks it ran to completion and produced a truncated domain, a replicated mesh and an equation
count `nproc` times too large, without an error; at three and four ranks it hung. §1 below is that
starting state, kept because it is what the tests exist to prevent coming back; §3 is what was done
about it, stage by stage.

The two paths differ enough to be treated separately throughout:

* **recreation** - `RemesherViaRecreation`, attached to every `MeshedMeshTemplate` by default. The
  user's `define_geometry` runs again and rebuilds the geometry, usually from
  `MeshedMeshTemplate.get_boundary_coordinates` (`pyoomph/meshes/mesh.py:881`).
* **`Remesher2d`** - the geometry is reconstructed automatically from the deformed mesh's boundary
  nodes (`pyoomph/meshes/remesher.py:328`).

---

## 1. What actually happens

Measured with the reproducer in `pyoomph_runs/Bugs/MPIRemeshing/remesh.py`: a quarter disc meshed by
Gmsh, `p.initialise()` then `p.force_remesh()`, under `mpirun -n {2,3,4} ... --distribute`.

### 1.1 Recreation, 2 ranks: silently wrong geometry

`get_boundary_coordinates` asks `get_cached_mesh_data(name)` **without** `global_mesh`, so each rank
gets only its own partition of `domain/interface`:

```
rank 0   19 points   (0, 1)             ... (0.92388, 0.38268)
rank 1    7 points   (0.92388, 0.38268) ... (1, 0)
```

Each rank then describes a different geometry and runs Gmsh on it. `generate_mesh_to_file`
(`pyoomph/meshes/gmsh.py:303`) writes the `.msh` through `run_on_rank_zero` and every rank afterwards
loads *that one file*, so **rank 0's truncated wedge becomes the mesh on all ranks**. The
reconstructed `.geo_unrolled` stops at 67.5 degrees and `create_lines` closes the domain with a chord
from (0.924, 0.383) to the origin instead of along the substrate. Nothing raises.

### 1.2 Recreation, 3 and 4 ranks: deadlock

A rank whose partition holds no element of the requested boundary reaches `Mesh::to_numpy` with an
empty mesh and gets `RuntimeError: No elements in mesh. Cannot convert to numpy.`
(`src/nanobind/mesh.cpp:1449`). It unwinds out of `define_geometry` while the other ranks walk into
the `mpi_barrier()` / `run_on_rank_zero` pair inside `generate_mesh_to_file`, and the job hangs.

At three ranks it is **rank 0** that owns no interface elements, i.e. the merge root and the rank that
writes the `.msh` is exactly the one that cannot see the boundary. At four ranks it is rank 3. This is
not a corner case; it is what partitioning a domain by element count normally does.

### 1.3 Both: the distribution is lost and `ndof` is multiplied by `nproc`

```
before   rank=0 nelem=39 nhalo=9 mesh_distributed=True  problem_distributed=True  ndof=269
after    rank=0 nelem=47 nhalo=0 mesh_distributed=False problem_distributed=True  ndof=422
after    rank=1 nelem=47 nhalo=0 mesh_distributed=False problem_distributed=True  ndof=422
```

The new mesh is built in full on every rank and never distributed, while oomph-lib's
`Problem_has_been_distributed` stays `true`. `assign_eqn_numbers` therefore counts every locally-owned
node once per rank: 422 = 2 x 211 nodes for a mesh whose true global count is 211. The `nproc`
multiplier is the sharpest single symptom and makes a good assertion for the tests of §4.

### 1.4 Both: the value transfer never crosses a rank boundary

`MeshPointLocator` (`src/pointlocator.cpp`) contains no MPI code - `locate_batch` searches the local
mesh only. Transferring from the old (distributed) mesh to the new (replicated) mesh therefore gives

```
WARNING: interpolating domain: 77 of 236 node(s) could not be located in the old mesh
         and fell back to nearest-node blending instead of proper interpolation
```

and that blend (`Mesh::nodal_interpolate_from`, `src/mesh.cpp:3838`) runs over *local* nodes only. So
those 77 nodes do not keep their value and do not fail either; they get a plausible wrong one. This is
phase 5 of `dev_docs/mesh_point_locator.md`, which that document already gates on "`--distribute`
remeshing existing at all".

### 1.5 `Remesher2d`: fails before any of the above

```
RuntimeError: Cannot close line loop for surface domain.
Loop so far: substrate  -interface
Line list:   substrate  axis  interface
```

`Remesher2d._define_boundaries_for_domain` (`pyoomph/meshes/remesher.py:386`) walks
`mesh.boundary_elements(bn)` on the local mesh. A partition is bounded partly by named boundaries and
partly by the partition cut, and the cut carries no boundary name at all, so the collected curves can
never close. The same method also skips boundaries with `nboundary_element(ind)==0`, which makes
`_meshbounds` differ from rank to rank.

---

## 2. What already exists, and what has to be built

Most of the reading half is done. `pyoomph/meshes/meshdatamerge.py` merges the per-rank mesh data of a
distributed mesh into one entry on rank 0, **including interface meshes** (through the `-1`-padded
shared-node scheme of `Mesh::get_shared_node_numpy_indices`), and `tests/test_mpi_global_meshdata.py`
already asserts that the merged `get_interface_line_segments()` reproduces the single segment a serial
run sees. `dev_docs/mesh_data_cache_global.md` §10 lists "remeshing boundary identification on the
merged data" as the one remaining consumer of that machinery; this campaign is that item.

Ranks holding no element of a mesh are handled there too: `_local_payload` returns `None` for them
instead of calling `to_numpy`, so routing §1.2 through the merge removes that crash at the source.

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

## 3. Stages

### Stage 0 - stop the silent corruption. DONE.

Nothing here makes remeshing work; it makes the failure visible and local.

* `Problem.force_remesh()` refuses on a distributed problem, with a message naming what each attached
  remesher would get wrong. `Problem.experimental_distributed_remeshing = True` bypasses it, which is
  how the later stages are developed and how anyone who wants the old (wrong) behaviour back gets it.
  The check sits after the "is there anything to remesh at all" test, so a `remesh_if_necessary()`
  that finds nothing to do still returns quietly on a distributed problem.
* `MeshedMeshTemplate._do_define_geometry` agrees across ranks on whether `define_geometry` succeeded
  (`get_mpi_any`, symmetric rather than rooted at 0 - the rank that fails is usually *not* rank 0, see
  §1.2). A raise on one rank now ends the job with an error on every rank instead of hanging in the
  next collective. This covers the initial mesh generation as well, where the same hang was possible.

  It can only catch exceptions, not hangs: a `define_geometry` whose ranks disagree on how many
  collectives to enter still deadlocks inside them. Stage 1 makes `define_geometry` collective, so
  that contract has to be documented there.
* Tests: `tests/test_mpi_remeshing.py` (§4).

### Stage 1 - global boundary coordinates (recreation geometry). DONE.

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

It also *narrows* what stage 0's guard can catch, which is worth stating plainly because it is a
genuine regression in robustness: before stage 1 the first collective came after `define_geometry`, so
any raise inside it was catchable. Now a rank that raises *before* reaching
`get_boundary_coordinates` never joins its merge, and the others hang inside it - the guard is never
reached. Nothing short of timeouts can catch that, which is exactly why the "do not branch on rank"
contract has to be a contract. The test for the guard therefore injects its failure *after* the
collective (`mpi_remeshing_worker.Disc.fail_on_rank`), which is the case that can be caught: user code
that gets through the collectives and then produces something invalid on one rank.

**What is left over for stage 2.** With the geometry correct, the remaining damage is cleanly
isolated. Every rank now builds the *right* mesh (269 nodes, 60 elements - the same one the serial
remesh produces), and the only thing wrong is that it is replicated:

| ranks | nodes per rank | `ndof` | serial `ndof` |
| --- | --- | --- | --- |
| 2 | 269 | 538 | 269 |
| 3 | 269 | 807 | 269 |
| 4 | 269 | 1076 | 269 |

i.e. exactly `nproc` x the true count, confirming §1.3 at three partition counts rather than one.

### Stage 2 - re-distribute after remeshing. DONE.

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
Making it scalable, and lifting the adaption refusal, belongs to stage 5.

### Stage 3 - cross-rank value transfer. DONE.

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

### The refusal, narrowed

With stages 1-3 in place, `force_remesh()` on a distributed problem **works** for
`RemesherViaRecreation` together with `InternalInterpolator`, and needs no opt-in. What is refused is
now specific, and each refusal names itself:

| refused | why | lifted by |
| --- | --- | --- |
| `Remesher2d` | rebuilds the geometry from this rank's boundary elements, and the partition cut has no boundary name | stage 4 |
| `ProjectionInternalInterpolator` | its projection integrates the old field at partition-local locations | stage 5 |
| codimension-2 interfaces | transferred by `nodal_interpolate_along_boundary`, whose nearest-node matching is not pooled | see below |
| remeshing only some domains | the untouched domains stay partitioned; oomph cannot distribute a mesh twice | not planned |
| explicit `num_adapt > 0` | leaves the new mesh non-uniformly refined | stage 5 |

The codimension-2 case is the one worth doing next after stage 4: contact lines and axis points are
ordinary in real scripts, and `nodal_interpolate_along_boundary` is a different mechanism from
`nodal_interpolate_from` (nearest-node matching along a boundary rather than point location), so the
pooling does not reach it. Refusing it up front is what keeps such a problem from quietly getting its
corner values from whichever rank happened to hold them.

`Problem.experimental_distributed_remeshing` bypasses all of these.

### Stage 4 - `Remesher2d` from merged data

`Remesher2d` reasons in `_pyoomph.Node` and `OomphGeneralisedElement` objects and has to reason in
merged `MeshDataCacheEntry` arrays instead:

* **curves** come from the merged `get_interface_line_segments()`, already proven to reproduce the
  serial segmentation;
* **`_meshbounds`** becomes the union over ranks, not each rank's locally non-empty boundaries;
* **point sizes** are the awkward part. `get_size_at_point` reads
  `get_initial_cartesian_nondim_size()`, `get_current_cartesian_nondim_size()` and
  `refinement_level()` off the adjacent boundary elements, none of which travel with the merge today;
  they need a small per-boundary-element array in the payload. Deriving the size from the merged
  polyline spacing would be simpler but drops the initial/current averaging that encodes how far the
  mesh has stretched - a behaviour change, so not that;
* **`_ptsizes`**, filled by `EquationTree._setup_remeshing_size`, is keyed by `Node` and has to be
  re-keyed by merged point index;
* `_corner_size_map` comes from the template and is rank-independent - unchanged.

Since every rank already loads rank 0's `.msh`, the natural shape is to reconstruct the geometry on
rank 0 only, from merged data, under stage 0's collective failure guard.

### Stage 5 - the actual fix, later

Phase 5 of `dev_docs/mesh_point_locator.md`: give `MeshPointLocator` its MPI routing layer. The API
was designed for it (`LocationHandle::owner`, the reusable `LocationSet` schedule, the batched
`evaluate`), and §5 there already carries the cost model. Once it exists, stage 2's "replicate the
whole new mesh" step can go, remeshing becomes scalable rather than merely correct, stage 3's
`Allreduce` disappears, and the projection interpolator works distributed for free.

---

## 4. Tests

Following the `tests/mpi_*_worker.py` pattern, with a serial in-process run as the reference and
numbering-independent comparisons (node and element counts, a digest over sorted coordinates, field
sums), at 2, 3 and 4 ranks:

1. the boundary every rank rebuilds matches the serial one (stage 1, **done** - compared as a digest
   over the polylines, against a serial run of the same worker, at 2, 3 and 4 ranks);
2. the three-rank case specifically - rank 0 owning no interface element is what used to hang
   (**done**, covered by the same test: at three ranks rank 0 is the rank with no interface element);
3. `is_mesh_distributed()` true and `ndof` equal to the serial count after the remesh (stage 2,
   **done** - at 2, 3 and 4 ranks; the `nproc` multiplier of §1.3 makes it a single decisive
   assertion), plus the two refusals stage 2 has to make;
4. field values after remeshing equal to serial, with **zero** "could not be located" fallbacks
   (stage 3, **done** - compared through merged global mesh data at 2, 3 and 4 ranks, including sums
   weighted by position so that values landing on the wrong nodes cannot cancel out);
5. the `Remesher2d` path, once stage 4 lands;
6. `mpirun` **without** `--distribute` keeps working unchanged throughout - `needs_merging()` already
   short-circuits that case, but it is the regression that would go unnoticed.

`tests/test_mpi_remeshing.py` + `tests/mpi_remeshing_worker.py`, marked `slow`, 18 tests, ~39 s,
covering stages 0 to 3: the remaining refusals on **every** rank, the unchanged
behaviour of `mpirun` without `--distribute` at 2 and 4 ranks, the collective agreement when one rank
(0 or 2, since the agreement is symmetric) raises inside `define_geometry`, the merged boundary
matching a serial reference run at 2, 3 and 4 ranks, `ndof` and `is_mesh_distributed()` matching the
serial run after the re-partitioning, the transferred field matching it too, and the `Remesher2d`,
partial-remesh, explicit-`num_adapt` and codimension-2 refusals.

Everything except the refusals runs **without** the opt-in, i.e. in the configuration that ships.

Every run is under a timeout that fails rather than waits, so a regression in the agreement shows up
as a failed test instead of a stuck suite - it has already caught one (see the end of stage 1). The
partial-remesh test additionally asserts the exact return code, because the bug it pins down was a
segfault *after* a correct error message.

---

## 5. Open questions

* `ParametricGmshMeshRemesher2d` subclasses `Remesher2d` but rebuilds its geometry from problem
  parameters alone, so it should work with stage 2 only. Worth confirming as the early win.
* Boundaries carrying a `zeta`: `AssignZetaCoordinatesByArclength._refuse_if_distributed`
  (`pyoomph/meshes/zeta.py:211`) already refuses distributed meshes, so such an interface still
  refuses after stage 1. Whether that blocks real scripts is not yet known.
* How remeshing interacts with distributed **state files** has not been looked at;
  `dev_docs/distributed_state_files.md` may already constrain it.
