# Replicated problems under `mpirun` (no `--distribute`)

Status: **eight defects found and fixed**; one open question (§6) and one scaling caveat (§8) recorded
rather than solved.

Running without `--distribute` is not "running serially with extra processes". oomph-lib splits the
*assembly* by element and assembles the Jacobian over a **uniform distributed layout** as soon as
`nproc > 1`, whether or not the problem is distributed. Two mistakes follow from that, and every defect
here is an instance of one of them:

* **Indexing a vector by global equation number when it may only hold this rank's rows** (§2). Anything
  that comes back from the solver or the assembly is `nrow_local()` long, not `ndof()` long. Reading past
  it is silent — you get whatever is next in the heap.
* **Letting a rank's answer depend on which rank it is** (§3–§6). A replicated problem is only replicated
  as long as every rank computes the same numbers. Where they stop agreeing, the ranks refine to
  different meshes and the next assembly stitches a Jacobian out of four of them.

The second is the more dangerous: it does not crash where it happens. `rising_bubble.py` died several
solves later on a matrix and a vector of different sizes, and `marangoni_instability.py` did not die at
all — it just stopped converging.

---

## 1. The invariant

Without `--distribute` every rank holds the whole mesh and the whole dof vector; only the assembly and
the linear solve are parallel. Everything else must be a pure function of the replicated state,
evaluated identically on every rank. Two consequences, both of which were violated:

* **Bitwise, not "to within tolerance".** Adaptation decisions are discrete. An elemental error that
  differs in the last bit can flip a refinement, and from then on the ranks are solving different
  problems. There is no tolerance at which "close enough" is enough.
* **Every branch must be decided on replicated data.** A branch on a local count (`nnz_local`,
  `nrow_local`, "did *my* rows contain anything") can send ranks into different collectives. That is a
  deadlock, not a wrong answer. `_agree_on_reuse_structure_distributed` (`pyoomph/solvers/petsc.py`) and
  `_mpi_any_rank` (`pyoomph/solvers/generic.py`) exist for exactly this.

---

## 2. Reading a distributed vector by global index

Four places did this. The symptom differs only in how far the garbage travels before something notices.

**`MyFoldHandler` / `MyHopfHandler`, no-guess constructors** (`src/bifurcation.cpp`). Both derive the
null-vector guess from `x = J^-1 dR/dparam` and then read `x[n]` for `n < Ndof`. Under `mpirun` `x` holds
`nrow_local()` doubles. The guess came out NaN and the augmented Newton reported `Initial Maximum
residuals inf` before taking a step — while converging perfectly serially, which is what made it look
like a bifurcation-tracking bug rather than an indexing one. Now gathered with
`Problem::gather_double_vector_to_global` first. The other 30-odd `n < Ndof` loops in that file operate on
`Phi`/`Psi`/`Y`/`C`, which go through `Dist_helper` and already guard the distributed case; these two were
the only ones fed by a solver.

**`Problem::get_parameter_derivative`** (`src/problem.cpp`). Same shape, in a public API: `resdv.clear()`
un-builds the vector, the assembly rebuilds it on its own (distributed) distribution, and the copy loop
then runs to `ndof()`. Nothing in the failing scripts called it — it was found by grepping for the
pattern after the first two — but anything that did got garbage under `mpirun`.

**`CRDoubleMatrix::redistribute`** — two separate bugs in the same function, both upstream oomph-lib,
both recorded in `src/thirdparty/INFO_oomph-lib`:

* the per-rank nnz offsets were computed by scanning for the rank whose `first_row` equals a running row
  counter. That is well defined only while the `first_row`s are distinct, and they are not as soon as a
  rank owns no rows. Two rows over four ranks is `first_row [0,0,1,1]`, `nrow_local [0,1,0,1]`; the scan
  locked onto rank 0 — empty, so the counter never advanced — for all four steps, `nnz_count` stayed 0,
  the buffers were allocated empty, and rows arriving from other ranks were received *past the end of a
  zero-sized heap block*. The symptom was `corrupted size vs. prev_size` out of an `MPI_Type_free`
  further down, i.e. nowhere near the write.
* the wait for the non-blocking sends was guarded *and sized* by the **receive** count. The two are equal
  only when the two distributions mirror each other. In the same case rank 1 has three sends and one
  receive, so `MPI_Waitall` wrote three statuses into a one-element buffer; and a rank with sends but no
  receives skipped the wait entirely, so `build_without_copy` freed the send buffers while MPI was still
  reading them (`munmap_chunk(): invalid pointer`).

Both are reached from `get_eigenproblem_matrices`, whose non-distributed MPI branch assembles over a
distributed layout and redistributes back to a replicated one — so **any** eigenvalue problem with fewer
equations than ranks aborted. `turing_dispersion.py` is a 0d, 2-dof stability problem: fine on 1 or 2
ranks, dead on 3 or more.

> A replicated-assembly fallback was written for this (assemble locally, slice out this rank's rows)
> before the redistribute bugs were understood. Once they were fixed it was unnecessary — oomph's
> parallel assembly is correct here — and it was removed rather than left in as a second path. Worth
> remembering for the next one of these: **disable the workaround and check whether the real fix already
> covers it.**

**`PETScSolver.solve_serial`** (`pyoomph/solvers/petsc.py`) is the mirror image: it is handed a
*complete* `n x n` CSR (the replicated systems built in Python by `PeriodicDrivingResponse` and the
Lyapunov/Halley utilities — oomph's own solves go through `solve_distributed` as soon as `nproc > 1`) but
built its `Mat` on `COMM_WORLD`, so PETSc read `size=(n,n)` as *this rank's local block*
(`ValueError: size(I) is 809, expected 203` on 808 dofs at 4 ranks). Now `COMM_SELF`, with the KSP created
on the matrix's own communicator and `_structure_nrow_local` reset to `-1` so `solve_distributed` can
never adopt a `COMM_SELF` matrix as its own. Solving the replicated system redundantly per rank is also
what keeps the problem replicated, and it involves no collective, so it cannot deadlock against a rank
that took another branch.

---

## 3. Hanging-node values go stale under element-partitioned assembly

A hanging node's raw value/position storage is a **cache of its masters**, refreshed only as a side effect
of an assembly or output pass over the elements that contain it. oomph splits assembly by element
(`First_el_for_assembly` / `Last_el_plus_one_for_assembly`) whenever `nproc > 1`, distributed or not. So
each rank only ever refreshed the hanging nodes inside its own element range, and on a replicated mesh
the rest kept whatever they held before — which after adaptation is **zero**.

`sync_hanging_values_if_distributed` existed for exactly this staleness but was gated on
`is_mesh_distributed()`. **The trigger is not distribution, it is element-partitioned assembly.** Renamed
to `sync_hanging_values_if_parallel` and gated on `nproc > 1` (`src/problem.cpp`):

- **distributed** → `collapse_hanging_node_values()`, unchanged;
- **replicated** → `interpolate_hanging_values()`, which also restores hanging **positions** — equally
  stale on a moving mesh, and not covered by the node-only routine.

This was found from the far end of a long chain. The rising-bubble azimuthal stability tutorial under
`mpirun -n 2` (no `--distribute`) died in the **linear solver** with
`MatMPIAIJSetPreallocationCSR: Column too large: col 9837 max 9786`. Working backwards:

| what was measured | result |
|---|---|
| the two ranks' global `ndof` at that solve | **9714 vs 9787** — the ranks disagreed |
| the meshes after one `adapt()` | 428 elements and 1891 nodes on both, but 24 nodes each rank did not share |
| the pinning on the 1867 shared nodes | identical — pinning was innocent |
| the solution entering `adapt()` | bitwise identical |
| the elemental errors out of the estimator | **212 of 212 differed, by 60–92%** |

Identical solution in, wildly different error estimates out, hence different refinement, hence different
meshes, hence a linear solver handed inconsistent sizes. Rank 0 refined the bottom of the bubble, rank 1
the top. Measured effect of the fix:

```
hanging-node mismatches   120 of 1007  ->  22 (1 ulp, previously zeros)
elemental error max|diff|   2.289e-03  ->  2.596e-16
mesh after adapt #1   [(9366,428,1891),(9362,428,1891)] -> [(9224,422,1867),(9224,422,1867)] AGREE
mesh after adapt #2   [(18450,881,3739),(12174,554,2431)] -> [(18744,899,3817),(18744,899,3817)] AGREE
bubble under plain mpirun   crashed  ->  runs; eigenvalue matches serial to 4e-12
```

---

## 4. Nondeterministic hanging-master order — a *serial* bug MPI merely exposed

The 22 surviving 1-ulp differences above were not an MPI problem at all:

> `Mesh::interpolate_hanging_values()` returns a different answer each time it is run on the same input,
> **in a single-process serial run.**

```
serial, no mpirun, two runs of the same binary:
  positions before the call : SAME   (bit-identical input mesh)
  positions after the call  : DIFFER (~13 of 2014, by one ulp)
```

So results involving hanging nodes were not reproducible run to run. Usually invisible — but a 1-ulp
change in an elemental error can flip a refinement decision at a threshold, and from there the answer
diverges macroscopically. That is exactly the 60–92% divergence of §3.

**The cause.** `TreeBasedRefineableMeshBase::complete_hanging_nodes` accumulates the flattened master
weights in a `std::map<Node*,double>` and copied that map straight into the `HangInfo`. A map keyed by
`Node*` iterates in **heap address order**, and allocation addresses differ between runs and between
processes:

```
run1:  m0 w=-0.125  m1 w=0.375   m2 w=0.75   ->  0.84541717156758067
run2:  m0 w= 0.75   m1 w=-0.125  m2 w=0.375  ->  0.84541717156758078
```

The same three masters, the same three weights, the same three sub-values, enumerated in a different
order. Fixed in `refineable_mesh.cc` (`//FOR PYOOMPH`, recorded in `INFO_oomph-lib`): the masters are
written in a canonical order — their index in the mesh's node vector, which *is* reproducible — via a
`std::map<Node*,unsigned>` built once per call and a `stable_sort`. Verified: serial two runs
DIFFER → SAME; `mpirun -n 2` rank0 vs rank1 DIFFER → SAME; after a full transient, 0/4807 dofs, 0/3157
nodal values (was 15–22) and 0/2014 positions (was 10–24) differ. 112 regression tests pass, 365 s vs
363 s — no values moved, no measurable cost.

Sorting at the point of use (in `flattened_value`/`flattened_position`) was considered and rejected: it
would have made the sync reproducible while leaving the assembly's own constrained rows still
address-ordered. **The hang scheme has to be canonical at construction.**

### 4.1 The negative results, and why they are worth keeping

A pure function over bit-identical inputs cannot vary, so one of the premises had to be false. Each of
these was tested by experiment, and all of them are reusable:

| candidate | how it was tested | verdict |
|---|---|---|
| stale/partial coverage | dumped every hanging entry the pass writes | all 34 differing entries **are** written |
| the assembly writing afterwards | sync alone, no assembly | sync alone already produces the difference |
| flattening not leaf-only | code: recursion ends at real dofs/pinned data | pure; masters non-hanging by construction |
| element-dependent flattening | `node_is_c1_constrained_for_*` read only node state | element-independent |
| ordering / non-idempotence | ran the sync 1x, 2x, 3x | idempotent — fixed point after one pass |
| the dofs | bitwise compare | 0 of 4807 differ |
| the leaves | classified every differing entry | 0 non-hanging, 0 pinned differ |
| hang weights | dumped them | exact binary fractions (−0.125, 0.375, 0.75) |
| mesh geometry at setup | compared after `initialise()` | 0 of 2014 differ |
| uninitialised / invalid memory | `valgrind memcheck --track-origins` | **0 errors from 0 contexts** |
| pointer/heap-address container order | `setarch -R` (ASLR off) | still differs |
| Python iteration order → build order | `PYTHONHASHSEED=0` | still differs |

The false premise was an unstated one: **"identical inputs" was being read as identical *values*, when
what differed was the *order* they were summed in.** That is also why valgrind was clean (no bad memory),
why `setarch -R` did not help (the ordering is decided by allocation sequence, not by the ASLR base) and
why `PYTHONHASHSEED` did not help (nothing Python-side is involved).

Five wrong turns, each of which cost a build-and-run cycle and each of which looked sound:

- *"The linear solver mis-declares replicated matrices."* It does not; oomph hands it a genuine row split
  even without `--distribute` (71+72 on a 143-dof problem). It was reporting a divergence, not causing one.
- *"The per-node flux container is a pointer-ordered `std::set`."* True in
  `LagrZ2ErrorEstimator::get_element_errors`, and a real latent hazard — but changing it to an
  insertion-ordered vector left the result *bit-identical*, because patch matrices are allocated
  sequentially so set order already equalled insertion order. Reverted rather than kept (see §5 for what
  did land there, and why).
- *"The patch broadcast is incomplete."* It is not: both ranks end with all 286 patches and identical
  per-node contribution counts. (§6 is about something else that broadcast does.)
- *"A second, post-assembly sync will make the last write uniform."* It did not converge (15 → 11 → 17
  across variants, i.e. noise). Reverted.
- *"The divergence is seeded in the refined geometry."* No — the mesh is bit-identical after
  `initialise()`.

**The common thread: everything was searched for as a difference *between ranks*, and correctly found
that nothing differed, because the defect was not between ranks at all.**

---

## 5. The Z2 estimator: ranks disagreed about the error

Separate from §3 and §4, and it survived both. `LagrZ2ErrorEstimator::get_element_errors` split the patch
loop over the ranks when the mesh is *not* distributed, and broadcast the recovered flux coefficients back
so that everyone ends up with every patch.

It does not produce the same numbers on every rank. Measured on `rising_bubble.py` at 4 ranks, at the
`refine_eigenfunction` adaptation, the largest eigen-based elemental error was

| rank | max elemental error |
|---|---|
| 0 | 0.021520880031612715 |
| 1 | 0.021884607913632413 |
| 2 | 0.021878846084001466 |
| 3 | 0.021879279295485769 |

from **bit-identical** dofs, eigenvector and eigenvalue (md5 of each agreed across all four ranks). The
plain, non-eigen elemental errors disagreed too. Downstream:

```
Processor 0:  ---> 222 elements to be refined, and  4 to be unrefined, in total.
Processor 1:  ---> 222 elements to be refined, and 12 to be unrefined, in total.
Processor 2:  ---> 222 elements to be refined, and  4 to be unrefined, in total.
Processor 3:  ---> 222 elements to be refined, and  4 to be unrefined, in total.
```

after which rank 1 had 2038 elements and the others 2046, and the next solve died on
`Mat mat,Vec x: global dim 34840 34802`.

Every rank now computes every patch — `itbegin = 0`, `itend = n_patch` unconditionally — and the broadcast
branch is gone, leaving the path that serial and `--distribute` already used. Verified: at 4 ranks the
eigenvector, eigenvalue, dofs, eigen-based elemental errors and plain elemental errors now hash
identically on all four ranks. This is not obviously slower either: the split it replaces paid one
`MPI_Bcast` per patch — thousands of them on this mesh — to save each rank three quarters of a flux
recovery.

A second change went in alongside: `flux_coeff_pt` keyed each node's patches by the *pointer* to the
coefficient matrix, so the nodal average summed patches in **heap-address order**. It is now keyed by
patch index. That was found first and fixed first, and on its own it changed nothing (§6); it stays
because averaging in allocation order is not a defensible thing for a reproducible error estimate to do,
whatever the allocator happens to hand out.

---

## 6. Open: why the broadcast merge disagreed at all

**This should have been exact, and it is not explained.** Worth writing down, because the same pattern —
compute a share, broadcast, merge — is a natural thing to reach for again.

The merge looks like it ought to be bit-identical on every rank. The patches are enumerated from
`vertex_node_pt`, which `setup_patches` fills in **element order**, not map order, so patch `i` is the
same node on every rank and the `[itbegin, itend)` split partitions the same set. Each patch's
coefficients are computed by exactly one rank and `MPI_Bcast`-ed, so every rank should hold the same
doubles. They are appended in the same order (`iproc` outer, `ipatch` inner) everywhere, and the per-node
average and elemental error loops run over all nodes and all elements on every rank.

Ruled out by measurement rather than by reading:

* **Not the eigenvector or the state.** md5 of the eigenvector, the eigenvalue and the dof vector agreed
  across all four ranks *before* the estimator ran, on the failing adaptation.
* **Not the recovery frames.** `recovery_frame_wanted` returns `nodal_dim != dim`, false for this bulk 2D
  mesh, so the frame-rebuild-on-the-receiver path is not entered.
* **Not the pointer-ordered average.** Keying `flux_coeff_pt` by patch index instead of by pointer left
  the four numbers above *unchanged to every digit printed*. So whatever differs, it differs before the
  averaging step — the coefficients themselves, or which patches a rank ends up holding.
* **Not patch loss.** At 2 ranks both ends held all 286 patches with identical per-node contribution
  counts (§4.1).
* **Not a rounding-level effect.** 0.0215209 against 0.0218846 is a difference in the **third significant
  digit**. Reordered summation of the same values cannot do that.

That last point is what makes it strange rather than merely annoying, and it is why the fix was to delete
the split rather than to make the merge deterministic: the deterministic-ordering change was made *first*,
measured, and found to be a no-op. Something in the broadcast/merge is losing or duplicating patch
contributions, not merely reordering them. Suspects not yet excluded, roughly in order of how much they
would explain:

1. `comm_pt->broadcast(iproc, mattosend)` on a `DenseMatrix<double>` — whether the receiver's dimensions
   come out right for every patch, particularly one whose element count differs from the previous one's.
   (Note §7: that function *did* have a genuine type bug, fixed, which moves the computed errors by ~5e-7
   relative — the right order of magnitude for nothing here, but the function has form.)
2. Patches skipped by the `nelem >= 2` guard: the sender pushes to `vector_of_elements_in_patch_to_send`
   and `..._to_send` in lockstep, but a rank whose share contains a different number of skipped patches
   broadcasts a different `n_patches`, and the receivers' loop bound is the sender's — which is correct,
   but it is the one place the two vectors could get out of step.
3. `elem_num[...]`, the element-pointer-to-index map used to serialise a patch's element list.

Reproducing it needs the old code back; the probe that produced §5's table is in §10.

---

## 7. Also fixed: a broadcast that wrote eight bytes into a four-byte object

`OomphCommunicator::broadcast(const int&, DenseMatrix<double>&)` sent the matrix dimensions with
`MPI_UNSIGNED_LONG` while the locals are `unsigned`. Genuine undefined behaviour on any MPI run that
broadcasts a `DenseMatrix`; it survived because on little-endian x86 the extra bytes are the zero half.
Fixed and recorded in `src/thirdparty/INFO_oomph-lib`. It moves the computed errors by ~5e-7 relative, and
it is **not** the cause of §4.

---

## 8. Scaling caveat: replicated eigenvectors and dof derivatives

Everything above assumes the replicated things stay affordable, and today they do. Three of them are
`ndof`-long on **every** rank:

* eigenvectors — `SlepcEigenSolver` deliberately gathers each eigenpair to full global length with
  `PETSc.Scatter.toAll` (`_vector_to_global_array`, `pyoomph/solvers/petsc.py`), so that
  `set_eigenfunction_as_dofs`, the mesh data cache and the VTK output can go on indexing by global
  equation number;
* `dR/dparameter`, gathered by `get_parameter_derivative` (§2);
* the bifurcation handlers' `Phi`/`Psi`/`Y`/`C` guesses, likewise.

For the problem sizes pyoomph runs today that is the right trade: it costs `neval` vectors, not a matrix,
and it keeps one indexing convention across the whole output stack. **For a genuinely large problem it is
the wrong trade** — every rank paying `ndof` doubles per eigenvector defeats the point of distributing.
The eventual fix is to keep these local and teach the consumers (`set_eigenfunction_as_dofs`, the mesh
data cache, `MeshDataCombineWithEigenfunction`, the eigenfunction-driven error estimation) to work on the
local block with a halo exchange, exactly as the dof vector itself already does. Nothing currently needs
it, so it is recorded rather than started.

`refine_eigenfunction` is already refused on a distributed problem for a related reason
(`_require_non_distributed`): it carries the eigenfunction across the adaptation in history levels 3 and
4, and the history-dof accessors are global-index-only.

---

## 9. The scripts, and the one that still fails

Under `mpirun -n 4`, no `--distribute`:

| script | died of | now |
|---|---|---|
| `turing_dispersion.py` | `redistribute` heap corruption (§2) | passes, all 400 wavenumbers |
| `eigendynamics.py` | Z2 rank divergence (§5) | passes |
| `linear_response_drum.py` | `COMM_WORLD` `Mat` (§2) | passes, 1000 frequencies |
| `kuramoto_sivanshinsky_bifurcation.py` (both copies) | NaN fold guess (§2) | runs, finds the fold, continues |
| `rising_bubble.py` | Z2 rank divergence (§5) | ran 9 Bond-continuation steps, each with a `refine_eigenfunction`, where it used to crash on the first |
| `marangoni_instability.py` (mcflow) | Z2 rank divergence (§5), as a Newton that would not converge | ran well past the timestep it used to fail on |
| `plotting_eigenmodes.py` | NaN fold guess (§2) | guess is sane; still exhausts its 10 fold-Newton iterations — below |

**`plotting_eigenmodes.py` is the one that still fails, and it is not one of these bugs.** It converges in
3 Newton steps serially (`7.0e-4 → 7.0e-6 → 4.0e-9`, fold at gamma = 0.282482) and blows up at step 2
under MPI (`2.9e-3 → 1.9`). Chased to the end:

| quantity | serial | `np=4` |
|---|---|---|
| `norm(dofs)` at gamma = 0.28 | 34.878266889501646 | 34.87826688950166 |
| `norm(dR/dgamma)` | 6.8404799366270108 | 6.8404799366270117 |

The base state agrees to ~1e-15 and the parameter derivative to 16 digits, and both are now identical
across ranks. The fold guess is `x = J^-1 dR/dgamma` evaluated at gamma = 0.28, immediately next to the
fold — deliberately near-singular — and it amplifies those last bits into a measurably different guess,
after which the two runs converge to *different* folds (0.282592 serially, 0.276253 at 4 ranks; the
tutorial only claims "close to 0.28"). **That is conditioning, not a defect.** If the script should be
green under MPI the honest change is its own `max_newton_iterations`, which is a tutorial decision and
was left alone.

---

## 10. Reproducing

The probe that produced §5's table, dropped in place of `problem.refine_eigenfunction(...)` in
`rising_bubble.py`, writes one file per rank; `diff` them.

```python
import hashlib, numpy as _np
from pyoomph.generic.mpi import get_mpi_rank
from pyoomph import _pyoomph as _core
_h = lambda a: hashlib.md5(_np.ascontiguousarray(a).tobytes()).hexdigest()
f = open("probe_%d.txt" % get_mpi_rank(), "w")
f.write("EVEC %s\n" % _h(problem.get_last_eigenvectors()[0]))
f.write("DOFS %s\n" % _h(_np.array(problem.get_current_dofs()[0])))
_core.set_use_eigen_Z2_error_estimators(True)
bk, bp = problem.set_eigenfunction_as_dofs(0, mode="real")
e = _np.array(problem.get_mesh("domain").get_elemental_errors())
f.write("EIGERR %s max=%.20g\n" % (_h(e), e.max()))
problem.set_all_values_at_current_time(bk, bp, False)
_core.set_use_eigen_Z2_error_estimators(False)
f.write("BASEERR %s\n" % _h(_np.array(problem.get_mesh("domain").get_elemental_errors())))
f.close(); raise SystemExit
```

For §4, the reproducer needs no MPI, no solve, no adaptation and no eigensolver: take the problem to
`initialise()`, dump every nodal position, call `mesh._interpolate_hanging_values()`, dump again — and run
the whole thing twice, comparing the second dumps.

For §9's table, `get_parameter_derivative("gamma")` and `get_current_dofs()` after the solve at
gamma = 0.28 are enough; run with and without `mpirun` and compare the norms rather than the hashes, since
the distributed LU legitimately differs in the last bit.

Anything touching eigenvalues needs the complex PETSc on `PYTHONPATH` — see `CLAUDE.md`; the path differs
per machine.
