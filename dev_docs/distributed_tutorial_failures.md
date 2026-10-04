# Tutorial failures under `--mpirun N --distribute`

Status: **fixed.** Five scripts, three causes, six defects - all of them in pyoomph's own Python and
C++ (and one in a tutorial script), none in vendored oomph-lib, and every one of them an instance of
the same mistake: a decision that has to be the same on every rank was taken from a rank-local answer,
in front of a collective.

Found while running the 0.2.2 release checklist's phase 6, which asks for the tutorial scripts under
`--omp N`, `--mpirun 4` and `--mpirun 4 --distribute`. The complete results of that phase, on the
0.2.2 tree, i.e. BEFORE the fixes:

| configuration | result | wall clock |
|---|---|---|
| `--omp 4` | 139 passed, 1 skipped | 75 min |
| `--mpirun 4` (replicated) | 138 passed, 2 skipped | 116 min |
| `--mpirun 4 --distribute` | **133 passed, 5 failed**, 2 skipped | 373 min |

So this was specific to `--distribute`, not to MPI: the same scripts pass with four ranks as long as
every rank holds the whole mesh. (The 373 minutes are mostly the two 2 h timeouts in §5.1.)

These were **pre-existing** failures, not a 0.2.2 regression. `save_interface_state`, `_sorted_records`
and the guard that fired all predate `v0.2.1` (`cbe65852`), and nothing in 0.2.2's 79 commits touches
`Mesh::get_all_refinement_signatures`, the collective structure of `Problem.save_state` or
`_MeshFileOutput.output`. The reason they went unnoticed is coverage, not newness:
`citools/nightly_develop.sh` runs the tutorial pass serially and under `mpirun -n 4` **without**
`--distribute` (`TUTORIAL_MPI_RANKS`), so nothing ran these scripts in this configuration on any
schedule. §7 is still open on that.

Everything below was measured on draugr (the diagnosis) and walhalla (the fixes), 4 ranks.

---

## 0. The one sentence worth keeping

**On a distributed problem, no rank may decide on its own whether to enter a collective, or whether to
fail.** All four defects are that sentence violated:

| § | the rank-local answer that was trusted | what it gated |
|---|---|---|
| 2 | the live tree root's `global_base_index`, which the pruning leaves unset | which map entry a refinement signature lands in |
| 3 | the signature consistency check, which only rank 0 can run | the rest of `_define_state_file`'s gathers |
| 5 | `mesh.nodes()` on the plotting rank | nothing - it just gave an empty `max()` |
| 5.1 | `nref==0 and nunref==0` from `_adapt()`, i.e. this partition's count | `refine_eigenfunction`'s whole re-solve and re-eigensolve |
| 5.1 | `mesh.is_mesh_distributed()`, which an empty partition does not carry | an `MPI_Allgather` in `MeshFileOutput.output` |
| 5.1 | `must_reassign_eqs` from this rank's equations (latent) | `reapply_boundary_conditions` -> `assign_eqn_numbers` |

The codebase already had this rule written down in two places - `Mesh::needs_pooling_across_ranks`
("the gate has to be something every rank answers the same way") and
`InterfaceMesh._resolve_mesh_for_boundary_coordinates` - and both name an interface mesh whose
partition holds none of it as the case that breaks the local answer. §5.1 is exactly that case, one
file further on.

## 1. What failed, and what fixed it

| script | symptom | cause | fix |
|---|---|---|---|
| `Moving_Mesh/droplet_spread_hyperelastic_tangential_shift.py` | hangs forever at `t=8.0 s` | §2 + §3 | `src/mesh.cpp`, `meshstate.py` |
| `Moving_Mesh/droplet_spread_marangoni_and_gravity.py` | `mpi4py.MPI.Exception: MPI_ERR_TRUNCATE` at `t=3.0 s` | §2 + §3 | same |
| `SpatioTemporal_PDEs/moffatt_eddies.py` | `ValueError: max() iterable argument is empty` on rank 0, while plotting | §5 | the tutorial script |
| `Plotting_Interface/rising_bubble.py` | `TIMED OUT after 7200 s` | §5.1 | `generic/problem.py`, `output/generic.py`, `output/meshio.py` |
| `Advanced_Linear_Dynamics/rising_bubble.py` | `TIMED OUT after 7200 s` | §5.1 | same |

The hangs are the dangerous ones, because they are silent and unbounded: all four ranks sit at 100 %
CPU, nothing is written, and there is no timeout. In the run that found the first,
`state_000008.dump` had been created at 0 bytes while `state_000000`-`state_000007` were complete at
~26.8 KB each, and the log ended at `Saving state .../state_000008.dump`. A user would see a job that
never finishes.
## 2. The data defect: every unstamped root collapsed onto `-1`

`Mesh::get_all_refinement_signatures` (`src/mesh.cpp`) keyed its per-root map by `global_base_index`:

```cpp
BulkElementBase *root = root_element_of(this->element_pt(ie));
if (!root) continue;
oomph::Tree *t = (re && re->tree_pt() ? re->tree_pt()->root_pt() : NULL);
auto it = tree_of_root.find(root->global_base_index);
```

`global_base_index` is the **root-only** field, documented in `src/elements.hpp` as "Only meaningful on
root elements; -1 until assigned". That is exactly the field `Mesh::element_structural_key` deliberately
does **not** trust on a distributed mesh - it prefers the stamped, inheritance-propagated
`global_root_index`/`global_root_path` and only falls back to the live root lookup.
`get_all_refinement_signatures` never got that treatment.

### 2.1 Where the unstamped roots come from

From `Problem::distribute()` itself, on a mesh that is **already uniformly refined when it runs**.
`elements.hpp` says so next to `global_root_path`, quoting oomph: distribute() re-roots the forest at
whatever the leaves are when it is called -- "these elements become roots on each of the processors
involved in the distribution". Those new roots never went through
`assign_global_base_element_indices()`, which can only run before the distribution while the mesh is
whole, so their `global_base_index` is -1.

What puts refinement in front of the distribution is `_initial_uniform_refinement_level`, which
`RefineToLevel` raises (`pyoomph/equations/generic.py`) and which `Problem._do_initialise` applies
before it distributes - only the *coupled-domain remainder* is deferred until afterwards
(`_defer_uneven_initial_refinement`). `droplet_spread_marangoni_and_gravity.py` has
`RefineToLevel(self.initial_adaption_steps)` on a four-element `CircularMesh`, and measured on the
pre-fix tree every rank reported **`nroots=1, min=-1`** for `domain`: one re-rooted tree per rank,
all four under the same sentinel.

Refining *after* the distribution does not do it - the trees keep the roots they were distributed
with. Measured on the same pre-fix tree, `RectangularQuadMesh(N=6)` over 4 ranks with four adaptive
rounds reports `nroots=16, nneg=0`, roots 0..35, and writes a perfectly good file. **That** is why the
existing round trips in `tests/test_mpi_state_files.py` passed and the tutorials did not, and it is
the whole content of the new `test_uniformly_prerefined_distributed_mesh_is_addressable`: the same
worker with `RefineToLevel(2)` added reports `min_signature_root == -1` on the pre-fix tree and
`>= 0` after.

The consistency check in `_sorted_records` (`pyoomph/meshes/meshstate.py`) then correctly noticed that
the merged records disagreed:

```
StateFileInconsistency: Two processes describe different refinement trees for root element -1.
A process must see the whole tree of every root it touches (its own elements plus the halo copies
of the others) for the refinement to be storable independently of the partition
```

The `-1` in that message was the tell. It was not two processes disagreeing about a real root; it was
several roots sharing the sentinel. Note that the earlier bulk guard at `_local_contribution`
(`elem_keys[:, 0] < 0`) did **not** fire first - the element keys are fine, because they go through
`element_structural_key` and its stamped path. Only the signature map was broken. Measured separately:
the interface path is clean too, `badkeys=0` on every rank for every interface mesh.

### 2.2 The fix: build the signature from the stamps, and merge by union

Two halves, and both are needed.

`Mesh::get_all_refinement_signatures` no longer walks the live tree at all. It keys by
`element_structural_key` - the same stamped address `get_element_structural_keys()` uses, so the
signature and the element keys can no longer disagree about where an element sits - and builds each
root's tree as the **set of packed paths** of the elements this rank holds plus their ancestors:

```cpp
long r = -1, p = 1;
if (!element_structural_key(this->element_pt(ie), r, p)) continue;
std::set<long> &present = paths_of_root[r];
for (long q = p; q > 0; q /= 8)
  if (!present.insert(q).second) break;   // stop at the first ancestor already there
```

It then emits the same wire format as before (preorder son counts per root, roots ascending), with
`nsons(path)` read off the path set instead of off `Tree::nsons()`. Serially the set is complete and
the output is bit-identical to the old live walk, so **the state file format did not change** and old
files still load.

The second half is in `_sorted_records`. A rank holds the part of a tree its own elements and their
halo copies sit in, and that need not be the whole tree - so requiring the ranks to *agree* was wrong
in the first place, quite apart from the sentinel. The signatures are now decoded to path sets
(`_signature_to_paths`), **unioned** per root, and re-encoded (`_paths_to_signature`). That is correct
by construction: every leaf of the tree is held by some rank, as an owned element or as a halo copy, so
the union of the reported paths is exactly the set a serial run walks. And a union cannot disagree with
itself, so there is nothing left to be inconsistent about - the check is gone rather than relaxed.

The node and element records still go through `_dedup`'s `check`, which compares *values* and is a
different question.

## 3. The structural defect: a rank-local raise inside a collective section

This is the part that turned a reportable inconsistency into a hang, and it was worth fixing on its own
even with §2 fixed, because it would do the same for the next rank-local failure.

`_sorted_records` gathered, and then **only rank 0 kept going**:

```python
gathered = _gather_blocks(...)      # collective: every rank calls it
if gathered is None:
    return None                     # every non-root rank leaves here
...
        raise StateFileInconsistency(...)   # ... so this can only ever raise on rank 0
```

The check therefore ran on rank 0 alone, and raised there. Meanwhile ranks 1..n-1 had already returned
and marched on through the rest of `Problem._define_state_file` - into the interface block, whose
`save_interface_state` calls `_gather_blocks` again, once per interface mesh. Rank 0 unwound out of
`_define_state_file` instead and arrived at the rescue at the end of `save_state`, i.e. in
`mpi_share_any_failure`'s `MPI_Allgather`, while the others were in a later `MPI_Gather`. Measured, with
the raise replaced by a counter: rank 0 executed **12** `save_interface_state` calls where ranks 1, 2
and 3 executed **16** - exactly one round of the four interface meshes short.

Two collectives mismatched gives, depending on sizes and timing, either of the two observed symptoms:

* **hang** - `py-spy` on the four ranks of the first script:

  ```
  ranks A,B: mpi_share_any_failure (generic/mpi.py:251)        <- MPI_Allgather
             save_state (generic/problem.py:10934)
             error = StateFileInconsistency
  ranks C,D: _gather_blocks (meshes/meshstate.py:80)           <- MPI_Gather
             save_interface_state (meshes/meshstate.py:348)
             _define_state_file (generic/problem.py:10722)
             save_state (generic/problem.py:10924)
  ```

* **`MPI_ERR_TRUNCATE`** - the same mismatch, when the `allgather` happens to be matched against a
  `gather` payload of a different size. This is the better outcome of the two only because it stops.

The comment above `mpi_share_any_failure` claimed this could not happen - "a rank that failed would
otherwise leave the others waiting in the next collective - the run would hang instead of reporting the
failure". The net is real but it was in the wrong place: it wraps the section, while the section itself
contains collectives. It can only rescue a failure raised *between* collectives, not one raised before a
collective the other ranks are already committed to.

### 3.1 The fix: `_agree_or_raise`

Not "move the raise". The requirement is that **the decision to fail is collective**: every rank must
learn of the failure at the same point in the collective sequence, or none must.
`meshstate._agree_or_raise(error, distributed, context)` is that point - a thin wrapper over
`mpi_share_any_failure` that no-ops serially - and it is now called:

* in `_local_contribution`, **before** `_reconcile_node_keys`, for the unaddressable-element guard
  (which used to raise in front of that function's `alltoall`);
* inside `_reconcile_node_keys`' round loop, right after the existing `allreduce`, for the shared-node
  scheme mismatch (which used to raise in front of the next round's `alltoall`);
* in `save_mesh_state`, around `_local_contribution` + `_sorted_records`, which is the §3 case itself;
* in `save_interface_state`, both before its gather (for the key guard) and after it (for `_dedup`) -
  which is why the non-root ranks no longer `return` straight after the gather.

`_replay_refinement` got the same treatment for a loop condition rather than a failure: the number of
refinement rounds is now agreed with `_any_rank(...)`, because `refine_selected_elements_by_index` is
collective and the partitions do not run out of work at the same time.

What was **not** done, and must not be: filtering the inconsistent records away to make the symptom go
away. The records are the refinement of the mesh; writing a state file whose refinement is wrong
produces a file that loads without complaint and resumes a different problem. A hang is a bad failure;
a silently wrong state file is a worse one.

## 4. Reproducing

The isolated reproduction used for the measurements, which leaves the checkout alone (the editable
install means editing `pyoomph/` in place would change the behaviour of a tutorial pass running at the
same time):

```bash
# a copy of the package plus the compiled core, so PYTHONPATH can shadow the editable install
cp -r $HOME/code/pyoomph/pyoomph /tmp/iso/pyoomph
cp $HOME/.local/lib/python3.12/site-packages/pyoomph/_pyoomph_core.abi3.so /tmp/iso/pyoomph/

# PYTHONNOUSERSITE keeps the editable-install .pth finder from winning; the user site-packages is
# put back on PYTHONPATH (as a plain directory, so its .pth files are NOT processed) for pygmsh etc.
export PYTHONNOUSERSITE=1
export PYTHONPATH=/tmp/iso:$HOME/.local/lib/python3.12/site-packages:$PETSC_DIR/$PETSC_ARCH_REAL/lib

cd <workdir with the script>
mpirun -n 4 python3 -u droplet_spread_marangoni_and_gravity.py --distribute
```

`droplet_spread_marangoni_and_gravity.py` is the better of the two §2/§3 reproducers: it fails at
`state_000003` (a few minutes) rather than `state_000008`, and it fails rather than hangs, so it needs
no `py-spy`. `moffatt_eddies.py` reproduces §5 in about three minutes. Both rising_bubble scripts need
the **complex** PETSc arch on `PYTHONPATH` (azimuthal stability), and reproduce §5.1 after a handful of
Bond-number steps - about ten minutes, not two hours; the 2 h in §1 is when the harness gave up, not
when the deadlock started.

For a hang, `py-spy dump --pid <each rank>` is the whole diagnosis, and it is worth dumping **every**
rank rather than one: in §5.1 the three ranks with the identical stack were the red herring and the
fourth, which was somewhere else entirely, was the answer.

## 5. The plotting failure was in the tutorial script

`moffatt_eddies.py` failed differently and had nothing to do with state files:

```
RuntimeError: MPI rank 0 failed while plotting (ValueError: max() iterable argument is empty).
```

`MoffattStreamPlotter.local_velocity_range()` walked `mesh.nodes()` of the plotting rank's own mesh and
took `max()` over the nodes within `zoom` of the apex. Under `--distribute` rank 0 holds one partition,
and the deepest zoom (`zoom=0.01`) contains none of it - so the list was empty. It now reads the
**merged** data instead, `get_cached_mesh_data("wedge", global_mesh=True)`, which is answered by the
other ranks from the serve loop `MatplotlibPlotter.plot()` already wraps `define_plot()` in. The
problem's own `reach()` and `elements_per_decade()` were rank-local in the same way - they did not fail,
they just printed one partition's share - and are now reduced (`get_mpi_min`/`get_mpi_sum`, halo copies
excluded).

Here the rescue worked exactly as designed: `mpi_share_root_failure` reported which rank failed and
why, and the job ended instead of hanging - which is the contrast that makes §3 concrete.

### 5.1 Three rank-local decisions in the eigenfunction-adaptation path

**Both** scripts that hung here call `Problem.refine_eigenfunction` under `--distribute`
(`Plotting_Interface/rising_bubble.py` and `Advanced_Linear_Dynamics/rising_bubble.py`, line 229,
`refine_eigenfunction(use_startvector=True)`), and the first diagnosis followed that lead: sampled on
one of them, 40+ minutes with all four ranks at 100 % CPU, and the stack inside **oomph-lib's own**
distributed equation-number synchronisation:

```
PMPI_Alltoall
oomph::Problem::copy_haloed_eqn_numbers_helper
oomph::Problem::synchronise_eqn_numbers      (problem.cc:17220)
oomph::Problem::assign_eqn_numbers           (problem.cc:2374)
  <- reapply_boundary_conditions (problem.py:5568)
  <- actions_before_stationary_solve -> solve -> refine_eigenfunction (problem.py:6754)
```

That reading was wrong, and it was wrong because only *two* of the four ranks had been sampled. Dumping
all four shows **three** ranks in that stack and the fourth somewhere else entirely:

```
ranks 0,1,2: reapply_boundary_conditions (problem.py:5568)   <- assign_eqn_numbers, Alltoall
             solve -> refine_eigenfunction (rising_bubble.py:229)
rank 3:      output (output/meshio.py)                       <- comm.allgather
             _do_output -> _dispatch -> output (rising_bubble.py:235)
```

Stable over minutes. Nothing is wrong inside oomph-lib: three ranks are waiting in a collective that
the fourth is never going to call, because the fourth has already left the solve those three are still
inside. **Dump every rank, not one** - the identical stacks are the ranks that are still together, and
the one that is missing from them is the answer.

Three defects, found in that order, each hiding the next.

#### The one that deadlocked: `nref==0 and nunref==0` is a rank-local verdict

`refine_eigenfunction`:

```python
with self.custom_adapt(True):
    nref,nunref=self.adapt()
    if nref==0 and nunref==0:
        return self.get_last_eigenvalues()[0],self.get_last_eigenvectors()[0]
if resolve_base_state:
    self.solve()                 # <- collective
...
self.solve_eigenproblem(...)     # <- collective
```

`Problem._adapt()` reports what **this rank's partition** refined. A rank whose share is already at the
level the eigenfunction error estimator asks for reports `(0, 0)` while the others report work done -
and returns past the whole re-solve and re-eigensolve. Confirmed by labelling every collective in the
solve path and diffing the per-rank sequences: all four ranks agreed for 538 labelled events, and then
one went to its `MeshFileOutput` while the other three entered `actions_before_stationary_solve`.

The fix is `Problem._agreed_adapt_counts(nref, nunref)`, a `get_mpi_sum` of both counts, applied at all
four places that break on `(0, 0)`: `refine_eigenfunction`, the initial adaption in `_do_initialise`
(which already summed by hand, now folded into the helper) and the two remeshing loops. The initial
adaption having had this exactly right since it was written is the reason the same bug in the other
three went unnoticed.

#### The one underneath it: `is_mesh_distributed()` is not the same on every rank

With the counts agreed, the deadlock moved one collective further on, to the first one after the solve:

```python
if (not mesh.is_mesh_distributed()) and self.mpi_rank>0:
    return
if get_mpi_nproc()>1 and mesh.is_mesh_distributed():
    all_nelement = numpy.array(comm.allgather(mesh.nelement()))   # <- collective
```

A rank whose partition holds no element of a mesh does not carry the flag, so this gate let some ranks
out of `_MeshFileOutput.output` in front of its allgather. The same local flag was asked a second time
further down to choose between a per-rank and a single filename, so a run that did not deadlock could
still have had two ranks writing the same file.

The fix is `_BaseOutputter.mesh_is_partitioned()`: `get_mpi_any(mesh.is_mesh_distributed())`, computed
once before anything in `output()` can return and used for both the gate and the filename. A mesh that
is genuinely replicated (an undistributable one in a distributed problem) answers False everywhere and
still gets its single file, which is why this is not simply `problem.is_distributed()`. Three more
`output()` implementations opened with the identical line and got the identical fix: `_TextOutput`,
`_OutputTxtAlongLine` and `_GridFileOutput` in `pyoomph/output/generic.py`, all three of them in front
of a merge. The comment on the first of them claimed the early return "only lets a rank out when there
is nothing to merge in the first place"; that is the assumption that was false, and it is corrected
there.

#### The latent one: `must_reassign_eqs` is a rank-local verdict too

Fixed on the way past rather than observed failing, because it is the same shape and the consequence is
the same deadlock. `actions_before_stationary_solve` / `..._transient_solve` / `..._eigen_solve` gate
`reapply_boundary_conditions()` - i.e. `assign_eqn_numbers()` and its `Alltoall` - on what
`_before_stationary_or_transient_solve` / `_before_eigen_solve` return, and those walk **this rank's**
equations and flip **this rank's** Dirichlet activation flags. An `AxisymmetryBC` on an interface a rank
holds no element of has nothing to flip and answers False while the others answer True.
`Problem._agreed_must_reassign_eqs` is a `get_mpi_any` over the verdict, at all five call sites. OR
rather than AND: a rank that changed a pinning has to be followed by all of them, and renumbering when
nothing changed only costs time.

`src/thirdparty/oomph-lib` being ~1500 commits behind upstream (see `INFO_oomph-lib`) had nothing to do
with any of this.

## 6. Why this was documented before it was fixed

Recorded so the decision is not re-derived: it was pre-existing (0.2.2 is no worse than 0.2.1 here, so
shipping did not make anything worse and holding would not have made anything better), and §2/§3 live
in the partition-stability machinery, where the failure mode of a wrong fix is a state file that loads
fine and is wrong (§3.1) - not work to do against a release deadline. The complete failure list in §1
is what made the fix a bounded job afterwards.

## 7. Coverage, which is the actual root cause of the surprise

Partly closed. `tests/test_mpi_state_files.py::test_uniformly_prerefined_distributed_mesh_is_addressable`
covers §2: four ranks, `RefineToLevel(2)` so the uniform refinement lands in front of the
distribution, asserting that no rank reports a signature for root `-1` and that the file the ranks
wrote reads back serially. It fails on the pre-fix tree with exactly the message the tutorials
produced, and passes after. The existing round trips do not reach the case, which is why "add a
distributed state-file test" was never the gap; narrowing down which ingredient breaks it was, and
the ingredient is **uniform refinement before the distribution**, not refinement after it.

It does **not** reproduce §3. The collapsed `-1` signatures only *disagree*, and so only reach the
rank-0 raise, when the trees merged under the sentinel have different shapes; with the uniform
pre-refinement above they agree and the file is written (wrongly, but without complaint). Measured at
two, three, four and six adaptive rounds on top: `min_signature_root == -1` every time, no raise. So
§3 is still covered only by reading - the collective structure of `save_mesh_state` /
`save_interface_state` - and a test for it would have to construct ranks whose re-rooted trees differ,
which is what the `go_to_param` continuation in the droplet_spread scripts does and this worker does
not.

Still open:

* a nightly tutorial pass with `--distribute`, or at least a subset of it. The `Moving_Mesh` bundle
  alone would have caught §2/§3; neither rising_bubble is in it, so §5.1 needs
  `Advanced_Linear_Dynamics` too. Budget about six hours for the full pass as it stood - four of which
  were the two timeouts that are now gone.
* no test covers §5.1. Two cases, both easy and neither written here: an adaptive distributed problem
  driven through `refine_eigenfunction` to the point where one partition has nothing left to refine
  (`tests/mpi_eigen_adapt_worker.py` is the place for it, and it adapts only twice today), and a
  `MeshFileOutput` on a mesh that at least one rank holds no element of.
* a guard against the general pattern of §0. `grep` for `is_mesh_distributed()` in a condition, and for
  `raise` between a `_gather_blocks` and a `return None`: those are the two shapes that produced all
  four defects.
