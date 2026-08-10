# Linear solvers: reuse contracts, pivoting, and reporting failure

What each backend guarantees when a factorisation is reused, what MKL Pardiso's static pivoting does
when a matrix gets hard, and why a solver that reports its own failure must not kill the run.

Most of this came out of the frozen-sparsity work ([structural_assembly.md](structural_assembly.md)),
which made reuse routine and thereby exposed several long-standing defects: nobody read PETSc's
`KSPConvergedReason`, MKL's error code was written into a discarded temporary, and reusing a KSP
across solves silently froze MUMPS' *value-dependent* analysis.

---

## 1. Where the solver call actually goes

`superlu()` is **not** SuperLU in pyoomph. pyoomph defines C-linkage symbols with SuperLU's exact
signatures (`src/nanobind/solver.cpp`) so that oomph-lib's solver calls land in
`GeneralSolverCallback::solve_la_system_serial(op_flag, n, nnz, nrhs, values, rowind, colptr, b, ldb,
transpose)`, a nanobind trampoline into the Python solver classes. The CSR arrays arrive as zero-copy
numpy views onto oomph-lib's `CRDoubleMatrix` buffers. `op_flag`: **1 = factorise**, **2 =
back-substitute**, 3 = clean up (never reached — the shim leaves `f_factors` at NULL).

---

## 2. Does any backend use the CSR index arrays as scratch?

**No evidence of it, and the design fails safe either way.**

* **PETSc.** `createAIJ(csr=...)` *copies* into PETSc's own AIJ storage, so nothing pyoomph holds is
  aliased. The zero-copy alternative `MatCreateSeqAIJWithArrays` *would* alias — it was tried before
  and abandoned because it silently broke hypre/BoomerAMG. It must stay abandoned: with a frozen
  pattern the aliasing hazard is worse, not better, since the arrays now outlive a single solve.
* **MKL Pardiso.** `a`, `ia`, `ja` are documented inputs; MKL keeps its permutations in its own
  storage, and pyoomph additionally hands Pardiso a private `.copy()`.
* **oomph-lib.** `CRDoubleMatrix::sort_entries()` would reorder in place, but nothing in pyoomph or
  oomph calls it — checked.

The structural point is that this does not rest on the audit being exhaustive: **the Pardiso reuse
path verifies `ia`/`ja` by comparison before reusing**, so anything that did modify them produces a
full refactorisation, not a wrong answer. Any future backend reuse must keep that property. (That
check earned its keep immediately: oomph-lib emits **unsorted column indices per row**, which
`pardisoSolver` sorts in its constructor, so copying freshly assembled values into the sorted value
array would have scrambled them.)

**MPIAIJ, tested rather than assumed** — scribbling over the input arrays after `createAIJ` and
re-reading the matrix on 2 ranks shows no aliasing. It cannot be otherwise: an MPIAIJ matrix splits the
*global* column indices it is handed into diagonal and off-diagonal blocks with its own local
numbering, so it must transform them, and a transform implies a copy.

Two further MPIAIJ points the reuse path depends on:

* **`setValuesCSR` on MPIAIJ is ownership-range relative**, not global-row-indexed. Handing it an `I`
  array of length `nrow_local + 1` writes *this rank's* rows. Verified on 2 ranks: rank 1's update
  landed on global rows 3–5 and left rank 0's untouched. Had it been global-row-indexed, every rank
  above 0 would have silently written into rank 0's rows via the stash — a wrong answer, not a crash.
* **A latent dtype inconsistency, pre-existing, fixed.** The distributed *creation* call passed raw
  arrays while the serial one converted with `.astype(PETSc.IntType, copy=False)`, so on a
  64-bit-index (or complex-scalar) PETSc build the distributed path would have handed int32/float64
  where int64/complex was expected. Every `csr=` in the file now converts. On a matching build each
  conversion is a no-op returning the same view, which is exactly why it had survived.

**Symbolic reuse is confirmed, not inferred.** With `-info`, a `Mat` kept across two solves with only
its values changed emits `MatLUFactorSymbolic_SeqAIJ` **once**: PETSc compares the nonzero state, sees
`SAME_NONZERO_PATTERN`, and calls only `MatLUFactorNumeric`.

**scipy** (`splu`) exposes no symbolic-reuse API, so there is nothing to reuse at the factorisation
level; it still benefits from the frozen assembly and from that path emitting canonically sorted CSR.

**Mac Accelerate** does reuse: `MacAccelerateSparseSolver::refactorize_values_only()` keeps the
symbolic factorisation and calls `SparseRefactor`, refilling the values through a cached CSR→CSC
permutation, while `jacobian_structure_id` is unchanged and after verifying the index arrays as the
Pardiso path does. A failed `SparseRefactor` releases the factorisation and returns false rather than
throwing, so the caller falls back to a full factorisation — reusing a symbolic factorisation fixes
the pivoting, and new values may not tolerate it where a fresh one would have chosen differently.

> **None of the Accelerate code can be compiled on Linux** — it is inside `#ifdef __APPLE__`, so a
> Linux build does not even syntax-check it. `.github/workflows/test_mac_accelerate.yml` (manual
> dispatch, `macos-14`, arm64) runs `citools/check_accelerate_reuse.py`, which checks that the answers
> match an independent solver, that the reuse *actually happened*
> (`num_symbolic_factorizations()` stops growing while `num_numeric_refactorizations()` climbs — **not
> timings**, which is how a false negative survived for a long time elsewhere), and that renumbering
> forces a new symbolic factorisation. Measured: 2 of 3 factorisations became numeric-only with reuse
> on, 0 of 3 with it off, and renumbering took a new symbolic. Wall-clock is *not* measured — a shared
> CI runner is too noisy — so "SparseRefactor is cheaper than SparseFactor" remains the same
> assumption the Pardiso path makes, where it was measured locally and held.
> GitHub only shows the dispatch button for workflows on the default branch, so the file has to reach
> `main` before it can be run.

---

## 3. The forced diagonal belongs to the solver, not the problem

Needing a stored diagonal is a property of the *factorisation*, so `problem.force_jacobian_diagonal_entries`
was the wrong owner. It is now decided by the active linear solver, with the flag surviving as an
override:

* `GenericLinearSystemSolver.requires_explicit_diagonal()` — default `False`.
* `PETSCSolver` answers it **from the options database, not from its class**. Every `*pc_type` option
  is scanned (so a factorisation under a fieldsplit is seen), and a `*pc_factor_mat_solver_type`
  naming an external package (MUMPS, SuperLU, PaStiX, CHOLMOD, UMFPACK, MKL Pardiso, STRUMPACK…)
  answers `False`, because those build their own structure from the CSR. PETSc's own LU/ILU/Cholesky/ICC
  answers `True`. Deciding from the options rather than the class is what makes `petsc_mumps` and a
  hand-configured `-pc_type lu -pc_factor_mat_solver_type mumps` agree.
* `Problem._sync_diagonal_requirement_from_solver()`, called from `actions_before_newton_solve`, pushes
  the answer down before each solve — re-asked every time, because PETSc options can change at any
  point. Changing the answer invalidates the sparsity pattern, so a solver that *starts* needing a
  diagonal cannot keep being handed one assembled without it.

Verified, each in a clean process:

| solver | `requires_explicit_diagonal()` | rows without a diagonal | outcome |
|---|---|---|---|
| `pardiso` | False | 48 | converged |
| `petsc_mumps` | False | 48 | converged |
| `petsc` (PETSc's own LU) | **True** | **0** | fails on a zero pivot |

The last row is the point. PETSc's LU used to reject the matrix outright with
`MatLUFactorSymbolic_SeqAIJ … Matrix is missing diagonal entry 0`; that message is now gone — the
structural obstacle is removed and the diagonal arrives from the assembly. What remains is the
pre-existing numerical limitation of non-pivoting LU on a Taylor-Hood saddle-point system, which
reproduces identically without any of this work. Erring towards `False` is deliberate: an unnecessary
`True` costs a stored zero on every diagonal and perturbs pivoting, whereas a wrong `False` surfaces as
PETSc's own explicit complaint.

**Caveat.** PETSc's options database is global and sticky within a process, so a solver configured
earlier changes what a later one reports. Constructing `petsc_mumps` and then `petsc` in one process
leaves `pc_factor_mat_solver_type=mumps` in the database, and the second solver then both reports
`False` *and* actually runs MUMPS. That is PETSc's design; it is why the table above was measured one
solver per process.

**`MatShift(0.0)` never inserted anything.** The "manually inserted zeros on the diagonal" that the
sparsity work set out to replace was a no-op in PETSc 3.22, tested in all four combinations on a 3×3
matrix with a deliberately missing diagonal, checking the actual stored column indices and whether
PETSc's LU accepts the matrix:

| `NEW_NONZERO_ALLOCATION_ERR` | `shift(0.0)` | row 0 diagonal stored | PETSc LU |
|---|---|---|---|
| False | False | no | "Matrix is missing diagonal entry 0" |
| False | True | no | same |
| True | False | no | same |
| **True** | **True** | **no** | **same** |

`MatShift` with a zero shift returns early regardless of allocation policy.
`PETSCSolver._force_zero_diagonal()` is now a documented no-op.

---

## 4. Stored zeros on a saddle-point diagonal defeat MUMPS' analysis

The sharpest consequence of a value-independent sparsity pattern, and it took a long time to place.

`keep_structural_zeros` turns a **structurally absent** diagonal into one that is **present and exactly
zero**. On a distributed moving-mesh ALE case that happened for 158 rows, and they were not arbitrary:
124 Lagrange multipliers of the interface coupling and 35 `pressure__C1__*` — **saddle-point constraint
rows whose self-coupling is zero by construction**, not by coincidence at this state.

MUMPS chooses its elimination order from the **structure**, in the analysis phase, before it sees a
single value. Those stored zeros invite it to plan an elimination that then hits a zero pivot at
factorisation time — `INFOG(28)` reported 8 null pivots. With null-pivot detection off
(`ICNTL(24) = 0`, the MUMPS default) it divides by them and returns a silently wrong solution. Newton
then *diverges from a nearly converged state*: 3.6e-07 → 2e-05 → 0.03 → 4350 → 1e18, which is the
give-away — a correct Jacobian cannot amplify like that from 1e-7.

**Fix:** `PETSCSolver.use_mumps()` sets `ICNTL(24) = 1` by default, and the SLEPc spectral transform
does the same. It costs nothing on a matrix with no null pivots and only ever replaces a garbage answer
with a usable one. Set through `_SetDefaultPetscOption`, so anyone who configures it explicitly keeps
control. Flip side: a genuinely singular system now yields a pseudo-solution, so it shows up as a
Newton that fails to converge rather than one that explodes.

**Why it took so long, and what to do differently.** Every earlier attribution was wrong:

* the failure is a **knife-edge** — whether MUMPS' ordering reaches a null pivot shifts with tiny
  perturbations. That made it look nondeterministic and produced apparent *batches* of passes and
  failures;
* one such perturbation is absurd but real: putting an **empty directory on `PYTHONPATH`** flipped the
  result reproducibly, 10 out of 10 interleaved runs. Not because pyoomph reads `PYTHONPATH` — it does
  not — but because `problem.py` logs `str(sys.path)`, so a longer path changes an allocation and
  shifts the heap. Setting an unrelated variable of the same length did *not* flip it, which is what
  ruled out "environment size" and pointed at the log line;
* consequently **no single run means anything**, which is exactly how two retracted explanations came
  about. A third ("it's `PYTHONPATH`") was nearly recorded before interleaving disproved it.

What worked was refusing to bisect until there was a *deterministic* failing configuration, then
bisecting the feature flags inside it, injecting them through the case module rather than a
`sitecustomize` on `PYTHONPATH` — because the harness itself was one of the perturbations. From there
the diagnosis is three measurements: the two matrices are identical (all extra entries exactly `0.0`),
the diagonal census, and `INFOG(28)`.

**Loose end.** The field-coupling mask **over-reports**: a correct analysis would not add `(λ, λ)` or
`(p, p)` for these rows at all, since those blocks are structurally absent. The safety argument "a mask
that over-reports merely stores explicit zeros" is now known to be too optimistic — over-reporting on
the diagonal of a saddle-point system is what created this. Tightening it would also cut nnz. Worth
doing, but an optimisation now rather than a correctness fix.

---

## 5. A reused KSP freezes MUMPS' *analysis*, not just its symbolic factorisation

Reported from a user script (an axisymmetric two-phase ALE continuation): runs fine under `pardiso`,
and under `--petsc_mumps` collapses to `Maximum residuals inf` partway through an arclength
continuation, never recovering — the step keeps halving until oomph aborts.

**Nothing to do with the matrix.** Instrumenting `PETSCSolver.solve_serial` gives it in one line per
solve:

| solve | `INFOG(12)` (off-diagonal pivots) | `INFOG(1)` | KSP reason | ‖x‖ |
| --- | --- | --- | --- | --- |
| first, ndof 12474 | 288 | 0 | 4 (converged) | 2.5 |
| a few continuation steps later | 1648 → 1682 → 1776 | 0 | 4 | fine |
| the one that breaks | 2638 | **−9** | **−11** (`DIVERGED_PC_FAILED`) | **inf** |
| every solve after it | 2638 (frozen) | −9 | −11 | inf, in ~30 µs |

`INFOG(1) = -9` is "the main internal real workarray is too small". **KSP reuse is what caused it.**
Keeping the KSP alive across solves is the whole point of `_setup_solver_if_needed`, but PETSc then
sees `SAME_NONZERO_PATTERN` and skips `MatLUFactorSymbolic` — which for MUMPS means skipping the
**analysis phase**, and `use_mumps()` sets `ICNTL(6) = 5`, so that analysis is **value-dependent**: it
fixes a maximum-transversal column permutation and a scaling from the matrix it first saw. A
continuation walks the values away from that matrix, the frozen ordering stops putting large entries on
the diagonal, numerical pivoting grows an order of magnitude, and the fill-in finally exceeds the
workspace the stale analysis predicted.

Three A/B runs isolate it:

| arm | failures | `INFOG(12)` | in-process KSP time (25 solves) |
| --- | --- | --- | --- |
| as shipped (Mat + KSP reused) | fails, unrecoverably | 288 → 2638 | — |
| Mat reused, **KSP rebuilt each solve** | none | ~250, flat | 6.53 s |
| Mat + KSP reused, `ICNTL(14) = 200` | none | ~2900 | 2.43 s |
| no reuse at all | none | ~238, flat | 7.61 s |

The middle row is the diagnosis: rebuilding *only* the KSP, on the very same `Mat`, fixes it. The last
column is why the fix is not "stop reusing" — the stale analysis is worse numerically and still **2.7x
faster** than redoing the analysis every solve, because the analysis phase dominates.

### 5.1 `_ksp_solve_checked`

Replaces the bare `ksp.solve()` in both `solve_serial` and `solve_distributed`:

1. Read `getConvergedReason()`. **Nothing did this before**, which is why the failure presented as an
   `inf` residual rather than an error: on a PC failure PETSc returns immediately and leaves the
   solution vector untouched, and pyoomph handed that straight to Newton.
2. Act only on `DIVERGED_PCSETUP_FAILED` (−11) and `DIVERGED_NANORINF` (−9) — the two that mean the
   returned vector is not a solution of anything. Destroy the KSP and rebuild it (a fresh PC means a
   fresh analysis for the values actually on hand) and solve again. If MUMPS reported one of the
   workspace codes, double `ICNTL(14)` first and keep the larger value for the rest of the run.
3. If the retry also fails, raise, naming the reason and `INFOG(1)`. `raise_on_failed_solve = False`
   downgrades that to a warning.
4. Every *other* negative reason warns and hands the iterate on unchanged, which is what every
   configuration did before the check existed. `DIVERGED_MAX_IT` is the one that matters: an iterative
   KSP stopping on its iteration limit has produced a real, merely inaccurate, answer that Newton is
   often happy with. Retrying there would discard a hypre/GAMG setup to reach the same place, and
   raising would break any hand-configured iterative solver that has always been allowed to return
   early.

Nothing MUMPS-specific is reached unless MUMPS is present *and* in use: `_mumps_infog()` gates on
`PETSc.Sys.hasExternalPackage("mumps")` and then on `pc.getFactorSolverType() == "mumps"`.
`getFactorSolverType()` is what makes the second gate free of side effects — it answers None for a PC
with no factor matrix at all (jacobi, gamg, hypre, none), where `getFactorMatrix()` raises PETSc error
56. Collective under MPI: `KSPSolve` reduces the PC failure across the communicator, so every rank sees
the same reason and takes the same rebuild branch.

On the reported script this turns the run into one recovered failure —
`KSPConvergedReason -11, MUMPS INFOG(1) -9 … retrying with -mat_mumps_icntl_14 40` — after which the
continuation completes and reproduces the `pardiso` trajectory (arclength derivative agreeing to 15
digits), keeping the fast reuse path.

**The general point.** Reusing a factorisation object across solves silently changes *which*
factorisation you get, not just how long it takes to get it. Anything the first call decided from
values — MUMPS' transversal and scaling here — is now decided once, for a matrix that is no longer the
one being solved. Backends whose symbolic phase is purely structural (Pardiso's phase 11) do not have
this failure mode, which is exactly why `pardiso` was unaffected. Worth remembering before extending
reuse to another backend.

### 5.2 SLEPc + MUMPS: escalation only

Asked directly whether the eigensolver needs the same treatment. A third of it; the reasons the other
two thirds do not apply are worth recording.

**The stale analysis cannot happen here.** `SlepcEigenSolver.solve()` builds `J`, `M` and the `EPS`
from scratch on every call and destroys all three at the end, so the ST's KSP, its PC and the MUMPS
instance inside it are new each time. There is nothing cached to go stale.

**The silent garbage cannot happen here either.** SLEPc's `STSetDefaultKSP` calls
`KSPSetErrorIfNotConverged(st->ksp, PETSC_TRUE)`, so a failed solve inside the spectral transform
raises out of `EPSSolve` rather than returning an untouched vector — the opposite of the bare
`KSPSolve` in the linear solver. And PETSc's MUMPS wrapper already names the failure. So no convergence
check was added: the library performs it, and better.

**The escalation is worth having.** A shift-and-invert factorises `J − σM`, commonly denser than `J`
alone, so it exhausts MUMPS' workspace more readily — and pyoomph never sets `st_mat_mumps_icntl_14`
unless `use_mumps()` was handed an explicit `mumps_param14`, which asks the user to have foreseen the
whole thing. `_eps_solve_with_workspace_retry()` doubles it and **rebuilds** the EPS once. Rebuilding,
not re-solving: a PETSc option is consumed when the object it configures is set up, so a raised
`ICNTL(14)` is only seen by a KSP/PC that has not been set up yet, and the spectral transform's has.
The EPS construction moved into a closure for exactly this.

**`INFOG(1) = -19` is not in the escalation list**, in either solver. It means the factorisation
exceeded `ICNTL(23)`, the hard cap on working memory in MB, and doubling `ICNTL(14)` against a cap asks
for more of something already forbidden. The lever is `ICNTL(23)`, which is the user's to set (pyoomph
never does), so both solvers say so and re-raise rather than spending a retry that cannot work.

Verified on the complex PETSc build, `LineMesh(N=20000)`, 80 002 dofs:

| provocation | INFOG(1) | outcome |
| --- | --- | --- |
| none | 0 | unchanged; same eigenvalues as before the patch |
| `st_mat_mumps_icntl_14 = -95` | −9 family | `retrying with -st_mat_mumps_icntl_14 40`, then the correct spectrum |
| `st_mat_mumps_icntl_23 = 1` | −19 | names the cap, re-raises, no retry spent |
| an ST KSP that cannot converge | n/a | re-raised untouched (PETSc error 91), exactly as before |

The MUMPS plumbing both solvers share — the INFOG accessor with its two gates, the `ICNTL(14)`
doubling, and the two error-code lists — is module-level in `pyoomph/solvers/petsc.py` rather than
duplicated.

---

## 6. MKL Pardiso static pivoting

Chased from an intermittent-looking failure while building the heated-cylinder tutorial:

```
PardisoError: MKL Pardiso failed during the solve and iterative refinement: error -4 (zero pivot, ...)
```

### 6.1 It is not intermittent, and it is not one failure

At fixed settings it is fully deterministic — three runs at `desired_ndof=250000` failed identically,
at the same ndof. What looks like intermittency is that **the onset is not monotone in the budget**:
what matters is the particular mesh the `desired_ndof` controller lands on, not how large it is.

| `desired_ndof` | outcome | perturbed pivots (IPARM(14)) |
|---|---|---|
| 80 000 | OK (ndof 77 742) | 0 |
| 150 000 | OK | 0 |
| 175 000 | **Newton stall** | 0 |
| 200 000 | OK | 0 |
| 225 000 | **Pardiso −4** | 2 |
| 250 000 | **Newton stall** | 0 |
| 350 000 | **Pardiso −4** | 65 |

The stalls looked, from the outside, like a physics or continuation problem. They are not.
Instrumenting every solve with its backward error `||Ax-b||_inf / ||b||_inf`:

| run | solves | median backward error | worst |
|---|---|---|---|
| budget 80 000 (healthy) | 12 | 1.1e-09 | 9.9e-07 |
| budget 250 000 (stalling) | 20 | **1.9e-01** | **7.6e+00** |

So in the stalling case Pardiso reports a **perfectly clean factorisation** (`IPARM(14) == 0`),
therefore performs no iterative refinement at all, and returns a solution wrong by order unity. Newton
is fed nonsense. **This silent mode is considerably more dangerous than the `-4`**, which at least
announces itself. Raising `max_newton_iterations` from 10 to 30 changes nothing, which confirms the
Newton iteration is not the problem.

**The matrices are not singular.** UMFPACK — which pivots dynamically with a threshold criterion rather
than statically — solves every one of them; at budget 250 000 it returns ndof 240 172 where Pardiso
stalled at 219 279, simply much slower (206 s against ~35 s). That identifies the cause as MKL's
**static pivoting**: a pivot that comes out too small is perturbed rather than exchanged, and repairing
the damage is left to iterative refinement. On these meshes that repair either fails loudly (`-4`) or is
never attempted because MKL did not think it needed it.

### 6.2 What cures it

| setting | 225 000 | 350 000 |
|---|---|---|
| baseline | −4 | −4 |
| `IPARM(11)=1, IPARM(13)=1` (scaling + matching) | −4 | −4 |
| `IPARM(13)=2` (two-level weighted matching) | **OK** | **OK** |
| `IPARM(10)=8` (perturb doubtful pivots earlier) | OK | −4 |
| `IPARM(10)=20` | Newton stall | −4 |

Scaling and matching are already on by default for `mtype=11`, so asking for them explicitly is a
no-op — worth knowing, because they are the first thing MKL's documentation suggests. `IPARM(13)=2`
alone fixes both `-4` cases and the 175 000 stall but not the 250 000 one; adding `IPARM(10)=8` fixes
that too. The pair is `_ESCALATED_IPARM` in `pyoomph/solvers/pardiso.py`.

**Counter-intuitive detail:** with `IPARM(13)=2` the number of *perturbed* pivots goes **up** sharply
(2 → 194 at budget 225 000) and the solve nonetheless succeeds. The perturbation count is therefore not
a measure of trouble; what matters is whether refinement can repair it. **Do not use IPARM(14) as a
health indicator** — a comment in `solve_checked` used to say the opposite ("no pivot was perturbed, so
the factors are trustworthy") and has been corrected.

### 6.3 The escalation, and the second half of it that had to be withdrawn

`solve_checked` computes the backward error of **every** solve, not only of the ones MKL flags. If it
exceeds `_BACKWARD_ERROR_LIMIT`, or the solve raised, the factorisation is rebuilt once with
`_ESCALATED_IPARM` and the solve retried. `IPARM(13)` is read by the *reordering* phase, so this cannot
reuse the existing analysis — it must go back through phase 12, releasing the handle first or MKL leaks
its workspace.

The threshold is **1e-4**: healthy solves peak at 1e-6 and broken ones sit at 1e0, so anything in
between separates them, and 1e-4 keeps two orders of margin on the side that must not fire — a false
positive costs a full refactorisation. If the escalation does not help, the code **warns and returns
the better of the two answers rather than raising**: before this check existed such a solve was
returned silently, so refusing it outright would convert badly-converging runs elsewhere into crashing
ones. The `-4` path, which had no usable answer to begin with, still re-raises — with the *original*
error, since on a singular matrix the escalated reordering gives up with `-6 (reordering failed)`,
which describes the retry and not the matrix.

**An escalation that did not help must be withdrawn.** The first version carried a successful-looking
escalation forward into `iparm_override` for every later factorisation, on the reasoning that under
spatial adaptivity the mesh only gets harder. That broke three tutorials on the first nightly, because
`_escalated_iparm` was doing two jobs at once: *the escalation has been tried* (a one-shot guard) and
*the escalated settings are in force* (what the carry-over reads). Once set it was never cleared, so a
**single** marginal solve escalated, found the escalation no better, correctly returned the
pre-escalation answer — and left `IPARM(13)=2, IPARM(10)=8` in force for the rest of the run. §6.2
already recorded that these knobs can *cause* a Newton stall where they are not needed.

| tutorial | backward error | after escalating | what happened next |
|---|---|---|---|
| `droplet_spread_marangoni_and_gravity` | 4.4e-04 | 5.3e-04 | every later arclength step stalled at 1.1e-8 instead of 1e-14; `ds` halved below its 1e-10 minimum |
| `two_layer_flow_single_domain` | 1.6e-01 | 2.1e+26 | the next Newton residual was `inf`, at t=0.5 of 50 |
| `eigenbranch_continuation` | 1.7e-04 | 5.9e+08 | ran on for 3 000 more log lines and then diverged |

Worth noting what makes this hard to see from a log: the carry-over fired **silently** whenever the
escalation *succeeded*, and the warning that did print named only the retry's error. In the eigenbranch
log the one `PARDISO WARNING` sits 3 300 lines before the traceback with nothing in between connecting
them. Note also *where* it triggers — the eigenbranch escalation happens on an arclength step that is
already diverging, i.e. the backward error is measured on an iterate the step is about to reject
anyway. The trigger is doing its job; the matrix really is bad. **It was the permanence that was
wrong.**

Now `_escalation_spent` keeps the one-shot guard and `_escalated_iparm` means only *in force right
now*; `_deescalate_pivoting()` restores the saved `IPARM` and refactorises when the escalated solve's
backward error is not an improvement, and `_escalate_pivoting()` also restores them if its own
`factor()` raises, so a retryable `SolverError` hands back a handle that can still be factorised on the
next, smaller step. All three tutorials pass, the escalations still firing at the same solves and being
withdrawn.

### 6.4 A third failure mode, far below the threshold, and not worth escalating

`test_adaptive_3d_campaign`'s `test_ale_moving_mesh[levels0-hex_pyr]` failed at
`max|residual| = 8.583e-09` against an ALE tolerance of 1e-11. It looks like §6.1 and it is not.
Nothing here is near-singular: the escalation triggers on 1e-4, healthy solves peak at 1e-6, and this
solve is orders of magnitude cleaner than either — Pardiso considers it, correctly, a good solve. Nor is
it a bad Jacobian: the same assembled system handed to SuperLU or UMFPACK gives 5.6e-15 from the
identical starting residual. The answer is right; only the solve was imprecise.

Over all 33 ALE cases (11 layouts × 3 refinement states), worst `max|residual|`:

```
Pardiso   4.4e-16 .. 1.1e-14      except 3d-ale-hex_pyr-00 at 8.6e-09
SuperLU   5.3e-15 .. 1.2e-13
```

It is **one** case out of 33, not a property of ALE or of pyramids (the same layout refined gives
7.9e-16 and 1.1e-15) — and Pardiso is normally **one to two orders better than SuperLU**. Switching the
sweep to an "exact" solver to be safe would make 32 cases worse to fix one. So the campaign names the
single case in `_EXACT_SOLVER_CASES` and passes `linear_solver="superlu"` for it. That is a statement
about what the test is for — it certifies the discretisation and the analytic Jacobian by driving a
linear problem to machine zero in one Newton step, which needs the linear solve exact to roundoff — and
not a claim that Pardiso is wrong. 8.6e-9 is a perfectly good answer; it is simply not machine zero,
and this test measures machine zero.

Deliberately **not** done: lowering `_BACKWARD_ERROR_LIMIT` so the §6.3 escalation catches this. §6.3
is the argument against — the escalation is expensive, its knobs can themselves cause a stall where they
are not needed, and the trigger would have to drop by four orders to reach a solve that nothing is
actually wrong with. Relaxing the ALE tolerance from 1e-11 to 1e-8 was considered and rejected for the
same reason the tolerance is 1e-11 in the first place: it is the tightness that makes the case a
Jacobian oracle at all.

### 6.5 Pardiso could not report its own errors either

`run_pardiso()` passed MKL's `error` argument as `byref(c_int(ERR))`. That builds a throwaway ctypes
object from the Python int, so MKL wrote its code into a temporary that was discarded on the next line,
and the `if ERR != 0` below it compared the untouched Python int against zero. **Pardiso has therefore
never been able to report any of its own errors**, for as long as the call has existed — same class of
bug as the KSP reason nobody read, different backend.

Now a named `c_int` whose `.value` is checked against MKL's documented table, raising with the phase
named. The release phase is the deliberate exception: `clear()` runs from `__del__`, where raising only
produces "Exception ignored in `__del__`" and there is nothing to salvage, so it warns.

**Whether a singular matrix provokes an error depends on which solve path it takes.** Through the bare
`solve()` it does not — MKL replaces the tiny pivot and returns success with a huge solution, its
documented static-pivoting behaviour, not a gap in the check. Through `solve_checked()` it does: that
path raises `iparm[7]` to 20 whenever the factorisation reports a perturbed pivot, and the refinement
then fails with `error -4`. Measured on a 6×6 diagonal matrix with one zero pivot: `solve()` returns
`|x|_max = 3e13` and `error 0`; `solve_checked()` raises. Since `solve_checked()` is what both serial
branches call, a singular Jacobian now raises in normal use — which is what made §7 necessary.

### 6.6 Left open

* **The `try_to_reuse_solver` branch is untouched.** It has an accuracy check and a refactorisation
  fallback of its own, but no escalation, so it can still spin on a matrix that needs one. It defaults
  to `False` (and §8 is the argument for leaving it there).
* **Why these meshes at all.** The failures cluster on the *joint*-criterion meshes, i.e. exactly the
  ones where the tracer is left under-resolved. A sharply under-resolved advection-dominated field
  presumably produces the near-singular blocks that defeat static pivoting, but that was not pinned
  down — the tracer's own Péclet number turned out not to matter, because under the joint criterion it
  cannot influence the mesh at all.

---

## 7. A solver that reports its failure must not end the run

Giving the backends a voice (§5, §6.5) created a second problem, and it is the more serious: a solver
that reports a failure **killed the simulation outright**, where before it merely returned nonsense the
caller recovered from.

The recovery paths in oomph-lib catch `NewtonSolverError` and `InvertedElementError`, nothing else:
`adaptive_unsteady_newton_solve` halves `dt` and retries, the arclength continuation scales `Ds` by 2/3
and retries. A Python exception raised inside a solver backend becomes an `nb::python_error`, which
unwinds through the `superlu()` shim, past both handlers, and out of `run()` /
`arclength_continuation()`. And a singular Jacobian is exactly the state adaptivity exists to back away
from:

| what the backend does | adaptive transient | arclength |
| --- | --- | --- |
| returns a huge solution (pre-§6.5 Pardiso) | `TIMESTEP REJECTED`, run completes | — |
| raises — before the fix | **run killed** | **killed at step 4** |
| raises — after the fix | `TIMESTEP REJECTED`, run completes | `STEP REJECTED --- Ds=0.1333`, completes |

The shim now catches `nb::python_error` and, *if the backend declared the failure*, prints the Python
message and traceback and throws `oomph::NewtonSolverError(0, DBL_MAX)`.

### 7.1 The retry is opt-in, via `SolverError`

A backend raises for two quite different reasons and only one is worth retrying. MKL reporting a zero
pivot is a property of *this* Jacobian at *this* step; "your PETSc was not built with MUMPS", "no field
matches `velocity`", a missing MKL runtime or a plain typo in a custom backend will fail identically on
every retry, and treating those as a rejected step would shrink `dt` until it fell under `Minimum_dt`
and then blame the time step — burying the message that says what is actually wrong under fifty
hopeless solves.

`pyoomph.solvers.generic.SolverError` (deriving from `RuntimeError`, so an existing
`except RuntimeError` around a solve keeps working) is the marker:

| backend | raises | on |
| --- | --- | --- |
| `pardiso` | `PardisoError` | any of MKL's documented `error` codes |
| `petsc`, `petsc_mumps` | `PETScSolverError` | a KSP failure that survived the retry with a fresh factorisation |
| `superlu`, `umfpack` | `ScipySolverError` | `splu` reporting "Factor is exactly singular" |
| `accelerate` | `AccelerateSolverError` | `SparseMatrixIsSingular` / `SparseFactorizationFailed` |

Everything else propagates untouched, `KeyboardInterrupt` included — which is also why it needs no
special case of its own. Deliberately *not* converted: PETSc's missing-MUMPS and field-split errors,
Accelerate's `SparseParameterError` and `SparseInternalError`, every "unknown mode" internal check, and
the refusal to gather a custom solve routine onto rank 0 (§9.2). None of those get better with a smaller
time step.

(Pardiso's blanket refusal to run under MPI used to be on this list. It is gone — see §9 — but the
principle it illustrated stands: MKL Pardiso is still not MPI-parallel, the system is merely gathered
onto one rank for it.)

Accelerate is the one that raises from C++, and it is split by exception **type** rather than by parsing
the message: `checkStatus()` throws a `MacAccelerateNumericalFailure` for exactly the two statuses that
describe the matrix, and a translator in the bindings turns that into `AccelerateSolverError`. Renaming
a status string therefore cannot silently stop the retry.

The type is defined in Python rather than created in C++ and exported, so it lives where users find and
subclass it and appears in the stubs; the shim resolves it once, lazily, on the first failed solve. If
it cannot be resolved, nothing counts as retryable — the safe direction, since an exception then
propagates rather than being swallowed.

Two further things are load-bearing, each arrived at the hard way:

* **`NewtonSolverError(0, DBL_MAX)`, not `NewtonSolverError(true)`.** Reporting it *honestly* as a
  linear-solver failure does not work: the one-argument constructor sets `linear_solver_error`, and both
  recovering callers rethrow that as a fatal `OomphLibError` ("ERROR IN THE LINEAR SOLVER") rather than
  rejecting the step. The two-argument constructor leaves the flag false, i.e. reports what an ordinary
  divergence reports, which is what gets the step retried.
* **The distributed shim agrees across the ranks first**, with one `MPI_Allreduce`, exactly as
  `Problem::consume_newton_abort_request()` does. Rejecting the step changes the control flow of every
  rank, so a `NewtonSolverError` on some and not others leaves the rest in the next collective forever;
  a backend that gathers onto one rank sees the failure there alone. It cannot rescue a backend that
  raises halfway through its *own* sequence of collectives — those ranks are already lost.

A `SolverError` that turns out not to be transient does not spin: the retrying callers raise once the
step they are shrinking falls below `Minimum_dt` (1e-12) or `Minimum_ds`. A failure with nothing to
retry it — a stationary solve — stays fatal, which is the honest outcome; `oomph::NewtonSolverError` is
not a `std::exception`, so a nanobind exception translator was added to keep the few paths that let one
escape from reporting "Caught an unknown exception!".

### 7.2 What the Accelerate check establishes, and what it does not

Apple's `SparseFactor` does not return a status for a structurally singular matrix: it **traps the
process** (`SIGTRAP`, exit 133), with no exception, no message and no traceback. The first version of
the check probed it in-process and therefore killed the whole check before it could report anything —
which, since the surviving output ended at check [1], looked exactly like a regression in the solver
path and cost a bisect across four macOS runs to place. `faulthandler` does not cover `SIGTRAP` by
default; registering it explicitly is what finally named the line. (Registering it with `chain=False`
makes the trapping instruction re-execute forever rather than dying — 358 000 stack dumps in one run.)

The probe now runs in a subprocess, so a trap is contained and reported, and it tries `cholesky` as well
as `qr`. That second method is what makes the check worth having:

| method | singular matrix |
| --- | --- |
| `qr` | traps the process (`SIGTRAP`), no status, nothing to classify |
| `cholesky` | returns `SparseFactorizationFailed`, which arrives in Python as `AccelerateSolverError` |

So the translation **is** confirmed end to end by the `cholesky` case. Had the probe only ever used
`qr`, this would have stayed a plausible but untested claim. The singular case is nonetheless reported
best-effort — whether a given Accelerate method reports a status is Accelerate's business. The two
things pyoomph controls (that `AccelerateSolverError` is a `SolverError`, and that a malformed call does
**not** become a retryable one) are hard assertions.

The `cktso` wrapper was removed rather than given the same treatment: it was reachable only by name
through `factory_solver`, nothing else referred to it, and none of its error codes could be classified
without a copy of the library to test against.

### 7.3 The sliver step at the end of a run — never solver-specific

Found while checking that `before_newton_convergence_check` returning `False` takes the same route. It
does — `Problem::request_newton_abort()` has always thrown this very `NewtonSolverError`, and the step
was duly rejected. The run then died anyway:

```
SOLVE CALL timestep=0.1 at t=0.89999999999999991
SOLVE CALL timestep=1.1102341268554029E-16 at t=0.99999999999999989
Oomph-lib ERROR: Tried to reduce dt to 5.55117e-17, less than the minimum dt (1e-12).
```

Once *any* step is rejected the accepted `dt` is not the requested one, so the accumulated time misses
`endtime` by an ulp or two. `Problem.run()` then clamped the next step to the ~1e-16 that was left; at
that `dt` the mass term swamps the Jacobian, Newton cannot converge, oomph-lib halves it repeatedly and
kills the run — at the very end of a simulation that was otherwise finished. This hits every rejection
mechanism (convergence check, inverted element, and the solver route); the solver tests passed only
because their rejection happened to land back on an exact grid.

`run()` now treats a gap eight orders of magnitude below the step it was about to take as arrival
rather than as a time step. No legitimate final step is lost: for one to be skipped, `dt` would have to
have been planned 1e8 times larger than the time actually left.

### 7.4 Tests

`tests/test_solver_failure_recovery.py` (9 tests, ~2 s) asserts on *behaviour*, not on the rejection
message: pytest's `capfd` reads its temp file while oomph-lib's output is still in the C stdio buffer,
so `TIMESTEP REJECTED` only appears once the process exits, long after `readouterr()`. Instead the
transient test compares against an undisturbed reference run (agreeing to 1.5e-6 on a solution of order
0.13 — the rejected step was retried, not skipped or half-applied); the arclength test re-solves at the
parameter it stopped at and checks the dofs do not move (3e-10), i.e. the recovered continuation ended
on the solution branch; a parametrised test pins that the four non-`SolverError` exceptions come out of
`run()` as themselves; one drives the `before_newton_convergence_check` rejection to `endtime`; and one
pins the premise, that `solve_checked` really does raise `error -4` where plain `solve()` returns
`|x| = 3e13`.

---

## 8. Composing the reuse tiers: the numeric tier is not worth having

`PardisoSolver.try_to_reuse_solver` (off by default) sets `iparm[3] = 63` — MKL's preconditioned CGS
with the *previous numerical* factorisation as preconditioner, tolerance 1e-6 — plus `iparm[7]`
iterative-refinement steps, falling back to a fresh factorisation when it stalls. That is a third,
independent tier of reuse:

| tier | reuses | validity |
|---|---|---|
| symbolic | fill-reducing ordering, elimination tree | always, while the pattern holds |
| numeric-as-preconditioner (`try_to_reuse_solver`) | the LU factors themselves | only while the values stay close; checked a posteriori |

The two are now composed: when a numeric reuse stalls, the fallback is a phase-22 refactorisation rather
than a full rebuild, because a failed *numerical* reuse says nothing about the sparsity. That makes
`try_to_reuse_solver` strictly better than it was. It does not make it a good idea.

Six time steps of 3D Taylor-Hood NS (the slowly-varying-Jacobian case the numeric tier is for),
observables identical to 10 digits throughout:

| | `N=6` | `N=9` | full (ph12) | numeric (ph22) | CGS reuse |
|---|---|---|---|---|---|
| no reuse at all | 1.804 s | 7.652 s | 10 | 0 | 0 |
| **symbolic only** | **1.191 s** | **5.19–5.28 s** | 2 | 8 | 0 |
| both tiers | 1.191 s | 5.85–5.91 s | 1 | 1–2 | 9 |

Symbolic reuse alone is worth **−32 %**. Adding the numeric tier is a wash at `N=6` and a **+13 % loss**
at `N=9`, reproduced three times: once the problem is big enough, up to 30 CGS iterations plus a sparse
mat-vec for the residual check cost more than simply redoing the numbers.

So `try_to_reuse_solver` stays **off by default**, and the recommendation is to leave it off — its
premise, that reusing numerical factors beats recomputing them, stopped holding once phase 11 was no
longer part of "recomputing them". **The cheap thing became cheap enough that the clever thing lost.**

---

## 9. A serial backend under `mpirun`: gather onto rank 0

### 9.1 Why it is needed at all

oomph-lib picks the distributed solver path on the **process count**, not on the mesh
(`linear_solver.cc:850`):

```cpp
if (Solver_type == Distributed ||
    (Solver_type == Default && problem_pt->communicator_pt()->nproc() > 1))
```

So under any `mpirun -n N>1`, with or without `--distribute`, `solve_serial` is never reached from a
Newton solve and every serial backend refused: Pardiso raised in its constructor, SuperLU/UMFPACK raised
"cannot solve distributed", Accelerate inherited the base refusal. On a machine without PETSc that made
`mpirun` unusable outright — pyoomph warned about it and then died at the first solve.

`GenericLinearSystemSolver._solve_distributed_on_root` closes that: gather the row-distributed system
onto rank 0, call the backend's own `solve_serial` there, scatter the solution back. Opt-in per backend
via `gathers_to_root_under_mpi`, so a backend that has not thought about MPI keeps getting the refusal
instead of a silent serialisation.

**It does not scale and is not meant to.** Only the element loop stays parallel. `petsc_mumps` remains
the automatic MPI default; this is what makes an explicit `--pardiso`/`--superlu` under `mpirun` work,
and what a machine without PETSc falls back to.

### 9.2 Reusing `solve_serial` is the whole point

Pardiso previously had its own `_solve_distributed`. It duplicated the gather but not the machinery
around it: no symbolic-factorisation reuse, `try_to_reuse_solver` raising `NotImplementedError`, and it
passed the **local** `b` into `custom_solve_routine` next to the line using the gathered one. Calling
`solve_serial` on rank 0 instead inherits all of it. Measured under `mpirun -n 2`, five Newton steps: 1
full factorisation and 4 phase-22 refreshes on rank 0 — identical to serial.

Two preconditions come with the flag, and they are what makes it safe:

* **`solve_serial` must issue no MPI collective.** It runs on rank 0 alone. `custom_solve_routine` is
  the one thing on that path that could, so it is refused up front, on a replicated condition, before
  the first collective.
* **The factorisation slot must not be shared with anything else.** `PeriodicDrivingResponse`, the
  Lyapunov utilities and Halley's method build a *replicated* system in Python and call `solve_serial`
  on **every** rank, into the same slot rank 0's gathered factors live in. They now call
  `_note_external_serial_solve()` first, so a back-substitution landing on the wrong factors fails
  loudly rather than being silently wrong on every rank at once.

### 9.3 The refusal it replaces was describing a bug that had been fixed

`pardiso.py` refused MPI on the grounds that a gathered solve is written back onto only each rank's half
of a replicated dof vector, leaving stale or NaN residuals. That is not what happens.
`Problem::newton_solve` redistributes `dx` onto `Dof_distribution_pt` (`problem.cc:9356`), which is built
**non**-distributed without `--distribute` (`problem.cc:2354`), and `DoubleVector::redistribute` then
takes its `MPI_Allgatherv` branch (`double_vector.cc:358`) — every rank ends up with the whole `dx`.

Verified before any of this was written, with a throwaway gather-to-root subclass and no library edits,
on a 529-dof nonlinear Poisson Newton solve:

| | plain `mpirun` | `--distribute` |
|---|---|---|
| ranks agree with each other | **bitwise** | bitwise |
| matches serial | `‖u‖ = 2.605673453267989` at 1, 2 and 3 ranks | same |

The claim predates the two `CRDoubleMatrix::redistribute` fixes of 8 Aug 2026
(`src/thirdparty/INFO_oomph-lib`). This is the second time that has happened — see the note in
[replicated_mpi_correctness.md](replicated_mpi_correctness.md) §2 about disabling a workaround and
checking whether the real fix already covers it. It did.

### 9.4 The waiting ranks must sleep, not spin

MPI's blocking collectives busy-poll (`mpi_yield_when_idle=0` by default), so N−1 ranks waiting out a
factorisation pin N−1 cores at 100% — starving exactly the OpenMP/MKL threads rank 0 was supposed to
get. Sleeping only inside pyoomph's own waits is not enough either: a rank that returns from Python
immediately enters the `MPI_Allreduce` at the end of `superlu_dist_distributed_matrix`
(`src/nanobind/solver.cpp`) and spins *there*. **Every call must therefore end on a collective all ranks
leave at the same instant**, which is why the outcome agreement is not optional decoration.

`mpi_wait_idle` (`pyoomph/generic/mpi.py`) spins briefly — `Test()` is also what drives progress on a
build without a progress thread — then backs off to `time.sleep` between 0.1 ms and 5 ms. Interleaved
A/B on a 1.5 s stand-in factorisation, `-n 2`:

| wait | idle rank CPU | wall | fraction |
|---|---|---|---|
| polled (`mpi_wait_idle`) | 0.05 s | 3.03 s | **1.5 %** |
| blocking (`req.Wait()`) | 3.01 s | 3.01 s | **100 %** |

Wall time is identical, so the polling costs nothing. There is no timeout: a timeout that fires on one
rank *is* the deadlock it would be reporting. It prints a one-off note after 10 minutes instead, so a
hang is diagnosable from the log rather than silent.

**Freeing the cores only helps if rank 0 may use them.** Open MPI 4.1 maps by core at `-n ≤ 2` and by
socket above it, so at `-n 2` rank 0 is pinned to one CPU and its threads have nowhere to go. pyoomph
cannot change that from inside the process — binding is applied by `mpirun` before `exec`, and
`InitMPI` runs at import of `pyoomph/generic/mpi.py`. So it is detected and reported, not worked around:
the one-time rank-0 notice compares `sched_getaffinity` against `cpu_count` and, when they differ, says
to re-run with `mpirun --bind-to none` and to set the thread count via `problem.set_num_threads(...)`.
Setting `OMPI_MCA_mpi_yield_when_idle` globally was rejected: it would degrade every *other* MPI path,
whose collectives are short and balanced, to buy what the polled waits already deliver.

### 9.5 Failure has two routes, because the C++ layer only has one

`src/nanobind/solver.cpp` `MPI_Allreduce`s a "did this rank report a solver failure" flag right after the
Python call, and its comment says it exists for "a backend that gathers the system onto one rank". So a
**`SolverError`** needs nothing new: rank 0 raises, the other ranks return normally, and all of them come
out with a retryable `NewtonSolverError`. On that path `b` deliberately keeps the right-hand side rather
than a fabricated solution — if anything ever swallowed the `NewtonSolverError`, a bounded wrong step is
far easier to notice than a zero one, which looks like convergence.

Anything **else** cannot use that route: the shim rethrows a non-`SolverError` *before* reaching its
reduce, so rank 0 would unwind while the others sat in it. That case is agreed in Python first
(`_agree_on_gathered_outcome`, one polled `Iallreduce`) and then routed through `mpi_share_any_failure`,
which also covers a *non-root* rank failing — something the C++ mechanism cannot see at all.

### 9.6 Eigenproblems

`ScipyEigenSolver` gets the same treatment under `--distribute`, where each rank assembles only its own
`(nrow_local, n)` block: gather into one square matrix on rank 0, solve, broadcast. It recurses through
`solve()` itself with `custom_J_and_M`, which is already defined to take a whole global matrix, so
ARPACK, the dense fallback, the `ncv` retries and the sorting are reused rather than duplicated. The
matrix manipulators — which `get_J_M_n_and_type` *skips* when distributed, because they rewrite whole
rows of a square matrix — apply on rank 0, where that is what it holds.

No eigensolver pyoomph ships now answers `distributed_possible() == False`. The check stays for backends
defined outside pyoomph, which are the only thing that can still trip it.

### 9.7 Two pre-existing defects this uncovered

* **`factory_solver` could not find `superlu`/`umfpack`.** They live in `pyoomph/solvers/scipy.py`, and
  the lookup only special-cased `petsc_mumps`. They were registered *by accident* — via the SuperLU
  fallback in `pyoomph/__init__.py`, or via `solvers.pardiso`'s `from .scipy import ScipyEigenSolver`.
  Under `mpirun` the cascade picks `petsc_mumps`, so neither happens and an explicit `--superlu` failed
  with `No module named 'pyoomph.solvers.superlu'`.
* **`assemble_jacobian` had never worked under MPI**, for the same underlying reason as this whole
  section: it fed oomph's *local* CSR into a global-shaped `scipy.sparse.csr_matrix`.
