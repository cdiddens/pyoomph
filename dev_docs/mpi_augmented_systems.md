# Augmented systems under MPI

Two halves of the same subject, at opposite ends of their lifecycle:

* **The C++ handlers** (`src/bifurcation.cpp`: `MyFoldHandler`, `MyHopfHandler`, `MyPitchForkHandler`,
  `AzimuthalSymmetryBreakingHandler`) — **done**. They used to refuse `--distribute` outright; §1–§6
  record what had to change, what deliberately did not, and what is still refused.
* **The Python custom assembler** (`CustomAssemblyBase` and the `AugmentedAssemblyHandler` family in
  `pyoomph/generic/bifurcation_tools.py`) — **plan only, nothing implemented**. §7–§10.

The two are independent implementations of the same idea, and the second one throws from deep inside
C++ the moment `nproc > 1`.

---

# Part I — the C++ bifurcation handlers (done)

## 1. What was serial about them

Nothing in the *elemental* assembly. Commit 27a7c23, which made tracking survive a plain (replicated)
`mpirun`, had already established that the augmented Jacobian and residual agree with the serial ones to
1e-8 under MPI. What was serial was the global bookkeeping around it:

- **Storage.** `Phi`, `Psi`, `Y`, `C`, the azimuthal eigenvector parts and `Count` were plain
  `Vector<double>` of length `Ndof = problem->ndof()`, which stays the *global* row count when
  distributed, and were indexed inside the element loops by *global* equation number.
- **Dof vector.** Every rank pushed all `Ndof` eigenvector pointers plus every scalar unknown, and the
  augmented distribution was built non-distributed.
- **`Count` and `nelement`.** Both came from loops over `mesh_pt()->nelement()` with no halo filter and
  no reduction, so under distribution `Count` over-counted shared equations and the normalisation
  constant `-1/nelement` no longer summed to `-1` across ranks.
- **Direct `GetDofPtr()[global_eqn]`.** Correct on a replicated dof vector, out of bounds on a
  partitioned one.
- **Destructors** rebuilt the base distribution as non-distributed, which is wrong if the base was
  distributed.

## 2. The shared helper

`AugmentedDofDistributionHelper` (`src/bifurcation.hpp`) holds the dof bookkeeping for all four handlers
as a **member object rather than a base class** — the handlers already derive from
`oomph::AssemblyHandler` plus `AugmentedSparsityProvider`, and a member avoids reordering that.

Two modes, of which the replicated one reproduces the historical behaviour exactly:

| | serial / plain `mpirun` | `--distribute` |
|---|---|---|
| eigenvector dofs pushed | all `Ndof`, every rank | this rank's `nrow_local()` only |
| scalar dofs (param, Omega, Sigma) | every rank | rank 0 only |
| augmented distribution | non-distributed, built in place | `LinearAlgebraDistribution(first_row, n_row_local)`, pointer-swapped in |
| `global_eqn(naive)` | identity | per-rank interleaved translation table |
| `synchronise_scalars` | no-op | `MPI_Bcast` from rank 0 |

The layout is upstream oomph-lib's, from its distributed `PitchForkHandler`: rank `d`'s augmented rows
are its base rows, then — in the handler's naive block order — its rows of each eigenvector block, with
each scalar contributing one row at `d == 0`. Every handler describes itself to `build_augmented_dofs()`
as a list of blocks in its own naive order, so the same table construction serves fold `[u | param | Y]`,
Hopf `[u | Phi | Psi | param | Omega]`, pitchfork `[u | param | Y | Sigma]` and azimuthal
`[u | Re | (Im) | param | (Omega)]`.

**Why the distribution is pointer-swapped rather than rebuilt in place.** The dof halo scheme (and
therefore `Problem::global_dof_pt`) keeps a raw pointer to the distribution object it was built against.
Rebuilding that object in place while an augmented distribution is installed would leave the halo scheme
describing a layout that no longer exists. `Problem::SetDofDistributionPt` exists for this; the base
object is kept alive by the helper and restored on destruction.

## 3. Per-handler changes

- Eigenvector *unknowns* became `oomph::DoubleVectorWithHaloEntries` on the base dof distribution, read
  through `.global_value(g)` — which degrades to `[g]` when not distributed, so one code path serves all
  three modes. `Count` likewise.
- The fixed normalisation and symmetry vectors (`Phi` for fold, `C` for Hopf and pitchfork, pitchfork's
  `Psi`, the azimuthal `normalization_vector`) stayed **fully replicated** `Vector<double>`. They are
  read-only after construction and are set from guesses already identical on every rank, so replicating
  them removes any synchronisation obligation. The cost is `Ndof` doubles per rank per vector, accepted
  deliberately.
- `Nelement` is now a member holding the `MPI_Allreduce`d count of non-halo elements; `Count` is built
  from halo-skipping loops and then `sum_all_halo_and_haloed_values()`.
- `eqn_number()` still computes the naive number and returns `Dist_helper.global_eqn(...)` of it.
- Each handler overrides `AssemblyHandler::synchronise()` — which oomph already calls at the end of
  `Problem::synchronise_all_dofs`, so **no vendored oomph-lib change was needed; verify that stays true**
  — to refresh eigenvector halos and broadcast the rank-0-owned scalars.
- `get_eigenfunction()` keeps its contract (a globally replicated, non-distributed vector on every rank)
  by gathering when distributed, so the nanobind bindings and every Python consumer are unchanged.
- The FD branches perturb base dofs through `Problem::global_dof_pt(elem_pt->eqn_number(n))`. **Note the
  argument is the element's *own* equation number, not the handler's**: `eqn_number()` now returns a
  translated augmented number, and feeding that to `global_dof_pt` would index nonsense. The same applies
  to the azimuthal forced-zero sets, which hold base equation numbers.
- `realign_C_vector()` reads its members directly (owned rows + `MPI_Allreduce`) instead of the
  replicated dof-pointer table. Fold's long-standing `Y[n] = phin/phisqr` — a division by the sum of
  squares rather than the norm — is preserved; the dof entries it read always pointed into `Y` itself, so
  this always was a rescale of `Y` by its own square sum.

A few pre-existing slips were fixed on the way, all of the same kind — `GetDofPtr()[k*Ndof]` used to
reach the bifurcation parameter, where `Parameter_pt` says it directly. The azimuthal one was reading
`GetDofPtr()[3*Ndof]`, which is out of bounds when `has_imaginary_part` is false.

## 4. Still refused when distributed

- **`blocksolve=True`**, and the handlers' block modes (`solve_block_system`, `solve_complex_system`,
  `Block_augmented_J`, …). They rebuild replicated in-place dof vectors, and upstream's own augmented
  block path throws when distributed too.
- **Periodic orbit tracking.** `PeriodicOrbitHandler` temporarily overwrites arbitrary global dofs during
  assembly; a separate project. Its Poincaré-plane constraint also has a pre-existing inconsistency worth
  fixing first: `-d_plane` is not divided by `nelement` while the row is assembled additively per
  element, so the constraint is effectively `n0.x = nelement*d_plane` — a value that would become
  rank-dependent under distribution.
- **`adapt()` and arclength continuation while tracking.** Blocked by the history-dof refusals in
  `Problem::get_dofs(t,...)` / `set_dofs(t,...)`; independent of the handlers.
- **The frozen-sparsity fast path** declines augmented systems under MPI, so tracking falls back to
  oomph's `parallel_sparse_assemble`. Correct, just slower.
- **The no-guess fold and Hopf constructors**, which derive their eigenvector from a serial linear solve
  on a replicated vector. They throw when distributed; the Python side raises the same message earlier,
  with a better traceback. (Those two constructors were also the site of a live out-of-bounds read under
  plain `mpirun` — see [replicated_mpi_correctness.md](replicated_mpi_correctness.md) §2.)

## 5. A dense row and column, by construction

`eqn_number()` maps the last local indices of *every* element onto the scalar rows, so the normalisation
row and the parameter column are dense across all ranks. Under a row-distributed matrix that is one fully
dense row assembled from every rank. Correct, and the same as upstream, but a scaling hot spot — worth
revisiting before anyone runs this on a very large problem.

The scalar rows also have a deliberately **empty diagonal** (see the comments at the
`get_sparsity_pattern` implementations): the normalisation equation does not involve the parameter, so
declaring it dense would manufacture a stored zero on the diagonal, which is what invites MUMPS to plan
an elimination onto a null pivot. If a distributed MUMPS solve ever reports a zero pivot on the bordered
rows, that is the place to look, and its ICNTL knobs are reachable through the PETSc options pyoomph
already exposes.

## 6. `eigenvector_scaling`, and why the default is what it is

`activate_bifurcation_tracking(..., eigenvector_scaling=...)` defaults to `"unit"`, which is bit-for-bit
the historical behaviour: the guess is normalised to unit length and the constraint reads `c.y = 1`.

The problem with that on a large system is **scale**. A unit-length vector over `N` dofs has entries of
order `1/sqrt(N)`, so the eigenvector unknowns are tiny, and so are the constraint row's Jacobian entries
`c_i/Count_i`, while the residual they are weighed against is O(1). `"auto"` normalises the guess by its
**largest entry** instead and sets the constraint's right-hand side to the dot product the rescaled guess
actually has (`Normalization_rhs`, which multiplies the `-1/Nelement` constant at all nine residual
sites). The guess therefore satisfies the constraint exactly, and both the unknowns and the row stay O(1)
however large the problem is. `c` is rescaled along with the eigenvector, since leaving it at unit length
would fix only half the problem. It composes multiplicatively with `set_eigenweight`; the two are
independent knobs.

Measured on the Bratu fold:

| mesh | ndof | `unit` max&#124;y&#124; | `auto` max&#124;y&#124; | critical λ |
|---|---|---|---|---|
| N=8  | 451  | 0.1366  | 1.086 | 6.8082638085 |
| N=24 | 4419 | 0.04554 | 1.086 | 6.8081260834 |

`unit` decays with the mesh, `auto` does not, and the located fold is the same to 1e-10. Only the
eigenfunction's amplitude — always arbitrary — differs.

## 6a. Testing, and what each case is the only cover for

`tests/mpi_bifurcation_worker.py` + `tests/test_mpi_bifurcation_tracking.py` (marked `slow`, so `--full`
is needed): pytest runs serially and launches the worker under `mpirun`, comparing against an in-process
serial run of the same worker. 13 tests: four cases × {`np=2 --distribute`, `np=3 --distribute`, `np=2`
plain}, plus the `auto` scaling. The plain-`mpirun` cases exist to keep commit 27a7c23 fixed, not to
cover the new code.

| case | handler | naive layout | what it pins down |
|---|---|---|---|
| `fold` | `MyFoldHandler` | `[u \| param \| Y]` | one real eigenvector block, one scalar |
| `hopf` | `MyHopfHandler` | `[u \| Phi \| Psi \| param \| Omega]` | two blocks and two scalars — catches a scalar left un-broadcast |
| `pitchfork` | `MyPitchForkHandler` | `[u \| param \| Y \| Sigma]` | a scalar on either side of the block in the naive numbering |
| `azimuthal` | `AzimuthalSymmetryBreakingHandler` | `[u \| Re \| Im \| param \| Omega]` | the axis dofs forced to zero by global equation number |

**The critical parameter is a weaker certificate than it looks**: it is one number every rank reads off
the same converged augmented state, so it survives a great deal. Two things constrain more:

- **`eigfunc_usqr`**, the mesh integral of the squared tracked eigenfunction. Reaching it runs the
  eigenvector back through `set_current_dofs()`, which scatters by *global* equation number, and
  integrates over non-halo elements with an `MPI_Allreduce`. A broken `eqn_number` translation or a
  missing halo synchronise leaves the critical parameter and the eigenvector norm right and moves this.
  It must be taken **after** deactivating tracking: while tracking is active the dofs are the augmented
  ones, whose length and layout differ between serial and distributed, and padding a base-length
  eigenvector into them measures the padding.
- **cross-rank agreement**, since the parameter and Omega live on rank 0's dof vector alone, and
  `evect_len` catches a rank that returned its own row block instead of gathering.

Two ambiguities the assertions must allow for, both intrinsic and both visible in a plain replicated run
as readily as a distributed one: the Hopf frequency comes as ±iω and which one the eigensolver returns is
arbitrary (so the magnitude is compared), and the eigenvector's sign or phase is free (so the
eigenfunction integral is taken from the entrywise |v|).

Verified numbers, serial versus distributed, `N=8` (`N=20` for Hopf):

| case | serial parameter | `np=2 --distribute` | `np=3 --distribute` | eigfunc rel. diff |
|---|---|---|---|---|
| fold | 6.808263808496 | 6.808263808509 | 6.808263808502 | 4e-14 |
| hopf | 1.999999837212 | 1.999999836902 | — | 1.8e-14 |
| pitchfork | 19.739855570997 | 19.739855578031 | — | 2.3e-12 |
| azimuthal | 24.552372532331 | 24.552372532331 | 24.552372532331 | 1e-15 |

The Bratu fold's λ* ≈ 6.808, the Brusselator Hopf's `B = 1+A² = 2` with `ω = ±A = ±1`, and the
reaction-diffusion pitchfork's `λ = 2π² ≈ 19.7392` are known analytically, so these are not merely
self-consistent between the two runs.

**An unrelated SLEPc failure found while writing this.** The first attempt built the azimuthal case on
the tutorial's Rayleigh–Bénard problem. SLEPc fails on it with `error code 73` inside `EPS.solve` — **in
serial**, at every shift tried, with or without the pressure integral constraint, while scipy solves the
same eigenproblem fine. The shipped `axiflow` case of `tests/test_mpi_eigenvalues.py` fails identically.
Nothing to do with bifurcation tracking (no handler is installed during that eigensolve) and it predates
this work, but the Navier–Stokes azimuthal eigensolve is currently broken with complex PETSc on this
machine and is worth its own look. The azimuthal tracking case is built on a reaction-diffusion problem
instead, which SLEPc handles.

## 6b. Open: the moving-mesh droplet segfaults

`docs/source/tutorial/advstab/movmesh/hanging_droplet.py` — fold tracking on a moving mesh with an
interface and Lagrange multipliers — **segfaults on rank 0 during the tracking solve** under
`mpirun -n 2 --distribute`. Serially, and under plain `mpirun`, it is fine (`Bo_c = 2.946075780049` at
`V = 2.0944`). Not diagnosed, and not re-run since — a segfault under MPI takes PETSc's `MPI_ABORT` with
it.

The one observation says the failure is probably **not** in the handler: the distributed run reported
`ndof = 7846` where serial has `15693`, i.e. its *base state* was already a different problem before
tracking was activated. The script gets there through `go_to_param` with
`remesh_handler_during_continuation` followed by `force_remesh()`, so it sits directly on distributed
remeshing, whose stage 5 is deliberately unfinished. The sensible next step is to check whether a
`--distribute` run of that script *without* any bifurcation tracking already produces the wrong `ndof`,
which would settle it without touching the handlers.

Until that is understood, treat moving-mesh problems with remeshing as unvalidated here. The four handler
cases above use fixed meshes.

---

# Part II — the Python custom assembler (plan only)

`CustomAssemblyBase` (`pyoomph/generic/assembly.py`) and the `AugmentedAssemblyHandler` family in
`pyoomph/generic/bifurcation_tools.py` — the fold/pitchfork/Hopf/normal-mode/eigenbranch trackers and the
deflation handler — work only in serial. Under `mpirun`:

    RuntimeError: src/problem.cpp: This likely does not work in parallel

raised from `Problem::sparse_assemble_row_or_column_compressed_base_problem` via
`assemble_multiassembly` and `MultiAssembleRequest.assemble`. Behind that throw sit three more layers
that would each fail in turn.

A throwaway prototype (each rank assembling the full element range, the residual/Jacobian handoff sliced
to the caller's distribution, and a gather/scatter around the deflation solve) got past all of them and
then **hung** — still running Newton steps at a 600 s timeout. So §8 is what is *known* to be wrong, not
necessarily all of it; see B6.

## 7. Where "one rank" is baked in

```
tracker.get_residuals_and_jacobian()          bifurcation_tools.py (FoldTracker, e.g.)
  -> MultiAssembleRequest.assemble()
     -> Problem::assemble_multiassembly       problem.cpp
        installs CustomMultiAssembleHandler, then
        -> sparse_assemble_row_or_column_compressed_base_problem
     <- ONE complete global CSR + global dense vectors, as numpy/scipy
  builds the bordered system in Python (scipy.sparse.block_array)
Problem::get_jacobian / get_residuals
  -> get_custom_residuals_jacobian (Python)
  <- the bordered system, copied into oomph's DoubleVector/CRDoubleMatrix
-> oomph SuperLUSolver -> pyoomph LA backend (PETSc/pardiso/scipy)
```

Three properties of that pipeline are what "serial only" means concretely: the multi-assembly returns the
**whole** system to every caller, indexed `0..ndof-1`; the augmented dofs are appended to `Dof_pt` by
`add_augmented_dofs`, which rebuilds the dof distribution **non-distributed**, and
`DofAugmentations::split` reads the raw `Dof_pt` array; and the trackers do their algebra on global numpy
vectors (`numpy.dot(V, V0)`, `numpy.linalg.norm(V)`, `J@V`, `scipy.sparse.block_array`).

Two MPI regimes have to be kept apart throughout:

* **(a) replicated** — `mpirun -n N` without `--distribute`. Every rank holds the whole mesh and dof
  vector. oomph still splits the *element loop* across ranks and the linear algebra is genuinely
  row-distributed (`SuperLUSolver::solve(Problem*)` builds `LinearAlgebraDistribution(comm, ndof, true)`).
* **(b) distributed** — `--distribute`. The mesh is partitioned, `Dof_pt` holds only this rank's dofs, and
  every global-vector assumption above is false as well.

**Regime (a) is where the win is and where the work should start.** Bifurcation tracking on a real 2D/3D
problem is dominated by the elemental assembly of `J`, `dJ/dp` and the Hessian-vector products, and in (a)
the O(ndof) memory per rank is already implied by the replicated mesh.

## 8. The blockers, each verified against the reproduction

The reproduction is `docs/source/tutorial/temporal/stability/deflated_solve.py` — a one-dof ODE, so no
MPI benefit whatsoever; it is just the smallest thing that fails.

**B1 — the assembly refuses to run.** The element range would otherwise be this rank's slice, and nothing
in the path reduces over the ranks, so lifting the throw without a reduction hands each rank a partial
residual and Jacobian.

**B2 — the residual/Jacobian handoff ignores the distribution it is given.** The custom branches of
`get_residuals`, `get_derivative_wrt_global_parameter` and `get_jacobian` copy `info.residuals.size()`
entries into a `DoubleVector` and call `jacobian.build(n_global, ...)`, but the caller has already built
both on the solver's distribution. `CRDoubleMatrix::build(ncol, ...)` uses `this->nrow_local()`, so on the
one-dof case rank 1 owns no rows and gets `nnz = row_start[0] = 0` with a one-entry value array — which
surfaced as PETSc's `ValueError: size(J) is 1, expected 0`. The `DoubleVector` writes are worse: they run
past the end of the local storage with nothing to catch them. **This is a live memory-corruption bug
independent of everything else here**, and the one part of the plan worth doing on its own merits.

**B3 — the augmented dof distribution disagrees with the solver's.** `add_augmented_dofs` builds the dof
distribution with `distributed=false` while the solver picks a distributed one for the same ndof. In (a)
that is survivable because the dof *values* are replicated; in (b) it is not.

**B4 — no LA backend supports the deflation solve under MPI.** `DeflationAssemblyHandler` is the only
assembler with `has_custom_solve_routine`: its routine wraps three solves of the same Jacobian into a
Sherman-Morrison correction, using global dot products. PETSc refuses it outright in the serial *and* the
distributed path — which also means deflation cannot use PETSc at all, not just under MPI. Pardiso's
distributed path gathers the system onto rank 0 but then passes the **local** `b` into the routine
(against the gathered `b_global` used on the line below it) — a latent wrong-answer bug. scipy/SuperLU is
serial-only.

**B5 — the Python algebra is global.** Every reduction in the trackers (`numpy.dot`,
`numpy.linalg.norm`), every `J@V`, and `split()` reading `Dof_pt` assume one rank owns everything. This
costs nothing in regime (a) — the vectors really are replicated — and is the bulk of the work for (b).

**B6 — unexplained hang.** The prototype completed several Newton solves and then stopped making
progress. Candidates, in order of suspicion: a rank-dependent decision inside the deflation loop
(`Problem.iterate_over_multiple_solutions_by_deflation` retries from random perturbations, and any RNG
not identically seeded on every rank sends the ranks down different control flow), the
Newton-failure/`max_newton_iterations` recovery path taking different branches per rank, and PETSc's
`_ksp_solve_checked` rebuild retry. **Diagnose this before committing to stage 2** — it may be an entirely
separate bug from the assembly, and it decides whether the "identical Python on every rank" assumption
that stages 1–3 rest on actually holds.

## 9. Which parallelism, in which regime

Three ways to make the pipeline parallel, evaluated for regime (a):

**Option A — replicated LA.** Every rank assembles the whole system and solves it redundantly with a
*serial* backend. Rejected: it needs a per-rank serial solver, and the MPI default backend is PETSc/MUMPS
on `COMM_WORLD`, so it requires either `COMM_SELF` PETSc objects or forcing a different backend under
MPI. It also buys no speedup at all and extends to (b) not at all.

> Not to be confused with the gather-to-root solve that the serial backends now use under `mpirun`
> ([linear_solvers.md](linear_solvers.md) §9). That is a different thing: the system is solved **once**,
> on rank 0, not redundantly on all of them, and the element loop stays split. It is a fallback for a
> backend that cannot solve in parallel at all, not a design for the trackers — it does not reduce
> memory per rank and does not extend to (b) either. Option A stays rejected.

**Option B — fully row-distributed.** The multi-assembly returns each rank's row block, the trackers work
on local blocks with allreduces for every inner product, and the border rows/columns live on a designated
rank. This is the endpoint for regime (b) and the only design that reduces memory per rank. It touches
every tracker and needs a distributed bordered solve.

**Option C — parallel assembly, replicated algebra, distributed solve.** Keep the element loop split
across ranks (real parallel work), reduce the CSR/residual over the communicator, `Allgather` so every
rank holds the identical global system, leave all the Python untouched, and slice to the caller's
distribution on the way into the linear solver. O(ndof) memory per rank, and the assembly — the expensive
part for these systems — is the part that parallelises.

**Recommendation: C for regime (a), then B for regime (b).** C is a legitimate endpoint for (a), not a
stopgap: in (a) the mesh is replicated anyway, so replicated vectors cost nothing extra, and the Python
trackers — the user-extensible part, the whole point of `CustomBifurcationTracker` — keep working
unchanged.

## 10. Stages

**Stage 0 — refuse honestly (small, do first).** A `_require_single_rank(...)` guard next to
`_require_non_distributed`, called from `Problem.set_custom_assembler` so the failure names the feature at
the point the user switched it on, instead of arriving five frames deep from `problem.cpp` — or, worse,
not arriving at all (B2 corrupts memory silently). Also fix the pardiso `b`/`b_global` mix-up (B4), a
two-character fix that is wrong today regardless of this plan.

**Stage 1 — make the handoff distribution-correct (B2).** In the three custom branches: read
`first_row()`/`nrow_local()` off the incoming `DoubleVector`/`CRDoubleMatrix`, copy only that row block,
and rebase the CSR row starts (column indices stay global). Build a non-distributed distribution only when
the caller handed in something unbuilt. In serial the block is the whole thing, so nothing changes there.
Prototyped already (kept out of the tree); it turns the PETSc `size(J)` error into a working distributed
solve of a correctly assembled system. Prerequisite for every later stage.

**Stage 2 — parallel multi-assembly for regime (a) (B1, Option C).** In
`sparse_assemble_row_or_column_compressed_base_problem`: keep the `First_el_for_assembly` element slice
(do **not** make every rank walk every element); after the local map-based accumulation, reduce across the
communicator — the cheap correct version is an `Allreduce` on the dense residual vectors plus a merge of
the per-row `std::map`s, and the fast version, once it works, is to build the frozen sparsity pattern
(`build_frozen_sparsity` is a purely local walk over the replicated mesh, so every rank derives the *same*
pattern with no communication) and `Allreduce` the value arrays in place, turning the whole reduction into
one collective per matrix. Leave the load-balance bookkeeping alone — the element split is real here, so
the timings mean what they normally mean. Keep the `Problem_has_been_distributed` throw (that is stage 4).
Validate by asserting the assembled CSR matches the serial one at 1, 2 and 4 ranks — same element set, so
only summation order differs; compare with a tolerance and say so.

**Stage 3 — the deflation solve under MPI (B4).** Deflation needs `solve_Jx_b` several times on the same
factorisation plus global dot products. With the system replicated (stage 2) and the solve
row-distributed, the adapter is: `Allgatherv` the local RHS block, run the routine on global vectors with
a `solve_Jx_b` that scatters into the KSP and gathers the result back, then write this rank's block into
`b`. Roughly 30 lines in `petsc.py::solve_distributed`, plus the same in `solve_serial` so deflation works
with PETSc in serial at all. **The gather/scatter branch must be chosen from the caller** ("am I the
distributed entry point?"), never from `len(b) == n` — on a one-dof, two-rank problem rank 0 owns every
row and rank 1 owns none, so a local test sends the ranks into different branches and hangs the first
collective. Lowest value of all the stages (deflation is used by two tutorial scripts and nothing else),
but it is where B6 will have to be settled.

**Stage 4 — regime (b), `--distribute` (B3, B5, Option B).** Only worth starting once stages 1–3 are in
and the `--distribute` remeshing/adaptivity work it sits on is settled. Sketch: augmented dofs get a
genuine distribution (border rows on the last rank); `split()` returns local blocks; the trackers get a
small reduction API (`self.dot(a,b)`, `self.norm(a)`) so `numpy.dot`/`numpy.linalg.norm` become allreduces
and user-written trackers have one obvious way to stay correct; `block_array` assembles local blocks with
global column indices.

**Validation** across the stages: the two deflation tutorials must report the same solution set as serial
(`x = [0, 1, -1]` at `r=1`); one tracker tutorial per family at 1 vs 2 vs 4 ranks with the same critical
parameter to solver tolerance and the same eigenvector up to sign/normalisation; and a unit test at the
C++ boundary assembling the same augmented system in serial and at N ranks and comparing the CSR — which
is what would have caught B2 immediately.

## 11. Open questions

* **B6** — is the hang a control-flow divergence between ranks, and if so, does anything else in the
  custom-assembler path make rank-dependent decisions? Nothing in §9–§10 survives if the ranks can take
  different Python branches.
* Is Option C's `Allgather` acceptable at the sizes these trackers are actually run at? An augmented
  system is ~2x the base ndof and the collective is O(ndof) per Newton step per matrix; at the 200k-dof
  ceiling that is a few MB per rank per step, which is fine, but it should be measured in-process (see
  `CLAUDE.md` on benchmarking) rather than assumed.
* Should the trackers' reduction API (stage 4) be introduced early, as no-ops in regime (a), so that
  user-written trackers written today keep working when (b) lands?
