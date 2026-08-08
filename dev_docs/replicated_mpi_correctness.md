# Replicated problems under `mpirun` (no `--distribute`)

Status: **six defects found and fixed**; one open question (§4) and one scaling caveat (§5) recorded
rather than solved. Written 2026-08-08 on branch `develop`, after `c80731b`. File and line references
are to the tree at the time of writing.

Eight tutorial scripts failed under `mpirun -n 4` without `--distribute`. They are listed in §6 with
what each of them died of. All eight turned out to be instances of two mistakes, made in six places:

* **indexing a vector by global equation number when it may only hold this rank's rows** — §2.
  oomph-lib assembles the Jacobian over a *uniform distributed* layout as soon as `nproc > 1`, even
  when the problem itself is not distributed (`create_new_linear_algebra_distribution`,
  `problem.cc`), so anything that comes back from the solver or the assembly is `nrow_local()` long,
  not `ndof()` long. Reading past it is silent: you get whatever is next in the heap.
* **letting a rank's answer depend on which rank it is** — §3 and §4. A replicated problem is only
  replicated as long as every rank computes the same numbers. Where they stop agreeing, the ranks
  refine to different meshes, and the next assembly stitches a Jacobian out of four of them.

The second is the more dangerous of the two: it does not crash where it happens. `rising_bubble.py`
died several solves later, on a matrix and a vector of different sizes, and
`marangoni_instability.py` (mcflow) did not die at all — it just stopped converging.

---

## 1. The invariant

Without `--distribute` every rank holds the whole mesh and the whole dof vector, and the only thing
that is parallel is the *assembly* (elements are split by index and the contributions merged) and the
*linear solve*. Everything else must be a pure function of the replicated state, evaluated
identically on every rank.

Two consequences worth stating, because both were violated:

* **Bitwise, not "to within tolerance".** Adaptation decisions are discrete. An elemental error that
  differs in the last bit can flip a refinement and from then on the ranks are solving different
  problems. There is no tolerance at which "close enough" is enough.
* **Every branch must be decided on replicated data.** A branch on a local count (`nnz_local`,
  `nrow_local`, "did *my* rows contain anything") can send ranks into different collectives. That is
  a deadlock, not a wrong answer. `_agree_on_reuse_structure_distributed` in
  `pyoomph/solvers/petsc.py` already exists for exactly this reason, and `_mpi_any_rank` in
  `pyoomph/solvers/generic.py` likewise; the fixes below keep to the same rule.

---

## 2. Reading a distributed vector by global index

Four places did this. The symptom differs only in how far the garbage travels before something
notices.

**`MyFoldHandler` / `MyHopfHandler`, no-guess constructors** (`src/bifurcation.cpp:509`, `:1618`).
Both derive the null-vector guess from `x = J^-1 dR/dparam` and then read `x[n]` for `n < Ndof`. On an
mpirun `x` holds `nrow_local()` doubles. The guess came out NaN and the augmented Newton solve
reported `Initial Maximum residuals inf` before taking a single step — while converging perfectly
serially, which is what made it look like a bifurcation-tracking bug rather than an indexing one.
Now gathered with `Problem::gather_double_vector_to_global` first. The other 30-odd `n < Ndof` loops
in that file operate on `Phi`/`Psi`/`Y`/`C`, which are built through `Dist_helper` and already guard
the distributed case with `base_nrow_local()`; these two were the only ones fed by a solver.

**`Problem::get_parameter_derivative`** (`src/problem.cpp:1921`). Same shape, in a public API:
`resdv.clear()` un-builds the vector, the assembly rebuilds it on its own (distributed) distribution,
and the copy loop then runs to `ndof()`. Nothing in the failing scripts called it — it was found by
grepping for the pattern after the first two — but anything that did got garbage under `mpirun`.

**`CRDoubleMatrix::redistribute`** (`src/thirdparty/oomph-lib/include/matrices.cc`) — two separate
bugs in the same function, both upstream oomph-lib, both recorded in `src/thirdparty/INFO_oomph-lib`:

* the per-rank nnz offsets were computed by scanning for the rank whose `first_row` equals a running
  row counter. That is only well defined while the `first_row`s are distinct, and they are not as
  soon as a rank owns no rows. Two rows over four ranks is `first_row [0,0,1,1]`, `nrow_local
  [0,1,0,1]`; the scan locked onto rank 0 — empty, so the counter never advanced — for all four
  steps, `nnz_count` stayed 0, the value and column buffers were allocated empty, and the rows
  arriving from the other ranks were received *past the end of a zero-sized heap block*. The
  symptom was `corrupted size vs. prev_size` out of an `MPI_Type_free` further down, i.e. nowhere
  near the write.
* the wait for the non-blocking sends was guarded *and sized* by the **receive** count. The two are
  equal only when the two distributions mirror each other. In the same 2-rows-on-4-ranks case rank 1
  has three sends and one receive, so `MPI_Waitall` wrote three statuses into a one-element buffer;
  and a rank with sends but no receives skipped the wait entirely, so `build_without_copy` freed the
  send buffers while MPI was still reading them (`munmap_chunk(): invalid pointer`).

Both are reached from `get_eigenproblem_matrices`, whose non-distributed MPI branch assembles over a
distributed layout and redistributes back to a replicated one — so **any** eigenvalue problem with
fewer equations than ranks aborted. `turing_dispersion.py` is a 0d, 2-dof stability problem: fine on
1 or 2 ranks, dead on 3 or more.

A replicated-assembly fallback was written for this (assemble locally, slice out the rows this rank
owns) before the redistribute bugs were understood. Once they were fixed it turned out to be
unnecessary — oomph's parallel assembly is correct here — and it was removed rather than left in as
a second path. That test is worth keeping in mind for the next one of these: **disable the workaround
and check whether the real fix already covers it.**

**`PETScSolver.solve_serial`** (`pyoomph/solvers/petsc.py`) is the mirror image: it is handed a
*complete* `n x n` CSR (the replicated systems built in Python by `PeriodicDrivingResponse` and the
Lyapunov/Halley utilities — oomph's own solves go through `solve_distributed` as soon as `nproc > 1`)
but built its `Mat` on `COMM_WORLD`, so PETSc read `size=(n,n)` as *this rank's local block*
(`ValueError: size(I) is 809, expected 203`, `linear_response_drum.py`, 808 dofs on 4 ranks). Now
`COMM_SELF`, with the KSP created on the matrix's own communicator and `_structure_nrow_local` reset
to `-1` so `solve_distributed` can never adopt a `COMM_SELF` matrix as its own. Solving the
replicated system redundantly per rank is also what keeps the problem replicated, and it involves no
collective, so it cannot deadlock against a rank that took another branch.

---

## 3. The Z2 estimator: ranks disagreed about the error

`LagrZ2ErrorEstimator::get_element_errors` split the patch loop over the ranks when the mesh is
*not* distributed, and broadcast the recovered flux coefficients back so that everyone ends up with
every patch (`src/lagr_error_estimator.cpp`, the `itbegin`/`itend` block and the `if
(!mesh_pt->is_mesh_distributed() && MPI_Helpers::mpi_has_been_initialised())` branch after it).

It does not produce the same numbers on every rank. Measured on `rising_bubble.py` at 4 ranks, at the
`refine_eigenfunction` adaptation, the largest eigen-based elemental error was

| rank | max elemental error |
|---|---|
| 0 | 0.021520880031612715 |
| 1 | 0.021884607913632413 |
| 2 | 0.021878846084001466 |
| 3 | 0.021879279295485769 |

from **bit-identical** dofs, eigenvector and eigenvalue (md5 of each agreed across all four ranks).
The plain, non-eigen elemental errors disagreed too. The consequence downstream:

```
Processor 0:  ---> 222 elements to be refined, and  4 to be unrefined, in total.
Processor 1:  ---> 222 elements to be refined, and 12 to be unrefined, in total.
Processor 2:  ---> 222 elements to be refined, and  4 to be unrefined, in total.
Processor 3:  ---> 222 elements to be refined, and  4 to be unrefined, in total.
```

after which rank 1 had 2038 elements and the others 2046, and the next solve died on `Mat mat,Vec x:
global dim 34840 34802`.

Every rank now computes every patch — `itbegin = 0`, `itend = n_patch` unconditionally — and the
broadcast branch is gone, leaving the path that serial and `--distribute` already used. Verified: at
4 ranks the eigenvector, eigenvalue, dofs, eigen-based elemental errors and plain elemental errors
now hash identically on all four ranks.

This is not obviously slower, either. The split it replaces paid one `MPI_Bcast` per patch — on this
mesh, thousands of them — to save each rank three quarters of a flux recovery.

A second change went in alongside: `flux_coeff_pt` keyed each node's patches by the *pointer* to the
coefficient matrix, so the nodal average summed patches **in heap-address order**. It is now keyed by
patch index. That was found first and fixed first, and on its own it changed nothing (§4); it stays
because averaging in allocation order is not a defensible thing for a reproducible error estimate to
do, whatever the allocator happens to hand out.

---

## 4. Open: why the broadcast merge disagreed at all

**This should have been exact, and it is not explained.** Worth writing down, because the same
pattern (compute a share, broadcast, merge) is a natural thing to reach for again.

The merge looks like it ought to be bit-identical on every rank. The patches are enumerated from
`vertex_node_pt`, which `setup_patches` fills in **element order**, not map order — so patch `i` is
the same node on every rank and the `[itbegin, itend)` split partitions the same set. Each patch's
coefficients are computed by exactly one rank and `MPI_Bcast`-ed, so every rank should hold the same
doubles. They are appended to `vector_of_recovered_flux_coefficient_pt` in the same order
(`iproc` outer, `ipatch` inner) everywhere. The per-node average and the elemental error loops then
run over all nodes and all elements on every rank.

What was ruled out by measurement rather than by reading:

* **Not the eigenvector or the state.** md5 of the eigenvector, the eigenvalue and the dof vector
  agreed across all four ranks *before* the estimator ran, on the failing adaptation.
* **Not the recovery frames.** `recovery_frame_wanted` returns `nodal_dim != dim`, which is false for
  this bulk 2D mesh, so the frame-rebuild-on-the-receiver path is not even entered.
* **Not the pointer-ordered average.** Keying `flux_coeff_pt` by patch index instead of by pointer
  left the four numbers above *unchanged to every digit printed*. So whatever differs, it differs
  before the averaging step — the coefficients themselves, or which patches a rank ends up holding.
* **Not a rounding-level effect.** 0.0215209 against 0.0218846 is a difference in the **third
  significant digit**. Reordered summation of the same values cannot do that.

That last point is what makes it strange rather than merely annoying, and it is why the fix was to
delete the split rather than to make the merge deterministic: the deterministic-ordering change was
made *first*, measured, and found to be a no-op. Something in the broadcast/merge is losing or
duplicating patch contributions, not merely reordering them. Suspects not yet excluded, roughly in
order of how much they would explain:

1. `comm_pt->broadcast(iproc, mattosend)` on a `DenseMatrix<double>` — whether the receiver's
   dimensions come out right for every patch, particularly a patch whose element count differs from
   the previous one's.
2. Patches skipped by the `nelem >= 2` guard: the sender pushes to
   `vector_of_elements_in_patch_to_send` and `..._to_send` in lockstep, but a rank whose share
   contains a different number of skipped patches broadcasts a different `n_patches`, and the
   receivers' loop bound is the sender's — which is correct, but it is the one place the two
   vectors could get out of step.
3. `elem_num[...]`, the element-pointer-to-index map used to serialise a patch's element list.

Reproducing it needs the old code back; the probe that produced the table above is in §7.

---

## 5. Scaling caveat: replicated eigenvectors and dof derivatives

Everything above assumes the replicated things stay affordable, and today they do. Three of them are
`ndof`-long on **every** rank:

* eigenvectors — `SlepcEigenSolver` deliberately gathers each eigenpair to full global length with
  `PETSc.Scatter.toAll` (`_vector_to_global_array`, `pyoomph/solvers/petsc.py`), so that
  `set_eigenfunction_as_dofs`, the mesh data cache and the VTK output can go on indexing by global
  equation number;
* `dR/dparameter`, gathered by `get_parameter_derivative` (§2);
* the bifurcation handlers' `Phi`/`Psi`/`Y`/`C` guesses, likewise.

For the problem sizes pyoomph runs today that is the right trade: it costs `neval` vectors, not a
matrix, and it keeps one indexing convention across the whole output stack. **For a genuinely large
problem it is the wrong trade** — every rank paying `ndof` doubles per eigenvector defeats the point
of distributing in the first place. The eventual fix is to keep these local and teach the consumers
(`set_eigenfunction_as_dofs`, the mesh data cache, `MeshDataCombineWithEigenfunction`, the
eigenfunction-driven error estimation) to work on the local block with a halo exchange, exactly as
the dof vector itself already does. That is a real piece of work and nothing currently needs it, so
it is recorded here rather than started.

`refine_eigenfunction` is already refused on a distributed problem for a related reason
(`_require_non_distributed`, `pyoomph/generic/problem.py`): it carries the eigenfunction across the
adaptation in history levels 3 and 4, and the history-dof accessors are global-index-only.

---

## 6. The scripts, and what is left

Under `mpirun -n 4`, no `--distribute`:

| script | died of | now |
|---|---|---|
| `turing_dispersion.py` | `redistribute` heap corruption (§2) | passes, all 400 wavenumbers |
| `eigendynamics.py` | Z2 rank divergence (§3) | passes |
| `linear_response_drum.py` | `COMM_WORLD` `Mat` (§2) | passes, 1000 frequencies |
| `kuramoto_sivanshinsky_bifurcation.py` (both copies) | NaN fold guess (§2) | runs, finds the fold, continues |
| `rising_bubble.py` | Z2 rank divergence (§3) | ran 9 Bond-continuation steps, each with a `refine_eigenfunction`, where it used to crash on the first |
| `marangoni_instability.py` (mcflow) | Z2 rank divergence (§3), as a Newton that would not converge | ran well past the timestep it used to fail on |
| `plotting_eigenmodes.py` | NaN fold guess (§2) | guess is sane; still exhausts its 10 fold-Newton iterations — below |

**`plotting_eigenmodes.py` is the one that still fails, and it is not one of these bugs.** It
converges in 3 Newton steps serially (`7.0e-4 -> 7.0e-6 -> 4.0e-9`, fold at gamma = 0.282482) and blows
up at step 2 under MPI (`2.9e-3 -> 1.9`). Chased to the end:

| quantity | serial | `np=4` |
|---|---|---|
| `norm(dofs)` at gamma = 0.28 | 34.878266889501646 | 34.87826688950166 |
| `norm(dR/dgamma)` | 6.8404799366270108 | 6.8404799366270117 |

i.e. the base state agrees to ~1e-15 and the parameter derivative to 16 digits, and both are now
identical across ranks. The fold guess is `x = J^-1 dR/dgamma` evaluated at gamma = 0.28, immediately
next to the fold — deliberately near-singular — and it amplifies those last bits into a measurably
different guess, after which the two runs converge to *different* folds (`kuramoto_...` on the same
mesh: 0.282592 serially, 0.276253 at 4 ranks; the tutorial only claims "close to 0.28"). That is
conditioning, not a defect. If the script should be green under MPI the honest change is its own
`max_newton_iterations`, which is a tutorial decision and was left alone.

---

## 7. Reproducing

The probe that produced §3's table, dropped in place of `problem.refine_eigenfunction(...)` in
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

For §6's table, `get_parameter_derivative("gamma")` and `get_current_dofs()` after the solve at
gamma = 0.28 are enough; run the same script with and without `mpirun` and compare the norms rather than
the hashes, since the distributed LU legitimately differs in the last bit.

Anything touching eigenvalues needs the complex PETSc on `PYTHONPATH` — see `CLAUDE.md`, the path
differs per machine.
