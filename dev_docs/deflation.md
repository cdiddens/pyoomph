# Deflation

Deflated solving (`Problem.iterate_over_multiple_solutions_by_deflation`) and deflated continuation
(`Problem.deflated_continuation`) find several solutions of the same nonlinear problem by multiplying
the residual with a factor that blows up at every solution already found. Farrell, Birkisson & Funke,
[arXiv:1410.5620](https://arxiv.org/pdf/1410.5620); the continuation loop is Farrell, Beentjes &
Birkisson, [arXiv:1603.00809](https://arxiv.org/pdf/1603.00809).

Two things are worth knowing before touching any of it, because both are counter-intuitive and both
are load-bearing:

* **The deflated Newton step is the ordinary Newton step times a scalar.** One linear solve, not
  three.
* **The deflation factor is normalised to its far-field value.** Without that, the factor silently
  loosens Newton's convergence test by a factor of `1/alpha` per known solution.

## 1. The operator

For known solutions `W_i` and `n_i = ||U - W_i||`,

    M(U) = prod_i ( 1/(alpha*n_i^p) + 1 )          # shift_mode="each" (the default)

The residual handed to Newton is `M(U) R(U)`. The **Jacobian is not deflated**; §2 says why it does
not have to be.

`shift_mode` selects where the shift sits, and the three modes are genuinely different operators:

| mode | `M` | far field | near one `W_i`, with others far away |
|---|---|---|---|
| `each` (default) | `prod_i (1/(alpha n_i^p) + 1)` | 1 | blows up |
| `single` | `(prod_i n_i^-p)/alpha + 1` | 1 | **damped** by the distant ones |
| `scaled` | `each` with `alpha^(1/k)` | 1 | blows up, more slowly per factor |

The `single` row is not a defect: one shift outside the whole product means a large `n_j` divides the
pole at `W_i` down. It is why `tests/test_deflation.py` asserts the blow-up with a single known
solution and the far-field limit with ten.

Everything is evaluated in log space (`log1p`, `logaddexp`, `scipy.special.expit`). The running
product overflows within a few tens of known solutions otherwise, and the log form is also what makes
the far-field limit exactly 1 rather than `1+eps` per factor.

`log M` is **clamped** at 300 rather than allowed to reach infinity. An iterate landing exactly on a
known solution would otherwise scale the residual by `inf`, and `inf*0` is `nan` for every dof whose
residual happens to vanish -- and a `nan` residual is not a failed Newton step to oomph-lib, it is a
comparison that is false whichever way it is asked, so the solve neither converges nor gives up. A
merely enormous factor trips `max_residuals` instead, which the drivers already treat as "this
attempt failed, try another perturbation".

## 2. One linear solve, not three

The deflated Jacobian is a rank-one update of `J`:

    G'(u) = M J + R (grad M)^T

The previous implementation inverted it by Sherman-Morrison with **three** calls to `solve_Jx_b`. But
the update vector is a multiple of the right-hand side, so the correction collapses. Writing
`eta = 1/M` and `d = grad eta`, the old routine was

    f = -b/eta ; fsol = solve(f) ; bsol = solve(b)
    numer = solve(f*dot(d,bsol)) ; denom = 1 + eta*dot(d,fsol)
    dx = eta*bsol - eta^2*numer/denom

and since `f = -b/eta` is a scalar multiple of `b`, `fsol = -bsol/eta`, `numer = -bsol*dot(d,bsol)/eta`
and `denom = 1 - dot(d,bsol)`, so the whole thing is

    dx = eta*bsol/(1 - dot(d,bsol))

i.e. **one solve and one dot product**. Equivalently, with `dU = J^-1 R` the ordinary Newton step,

    dU_deflated = dU / (1 + grad(log M) . dU)

which is Farrell's own observation that deflation only rescales the Newton direction. Two
consequences beyond the speed:

* the step depends on `M` only through `grad log M`, which is **invariant under `M -> cM`** -- so
  normalising the factor (§3) leaves every iterate bit-identical;
* under MPI the only global reductions left are the distances and this one dot product, which is what
  let deflation stay off the custom-assembler pipeline entirely (§5).

`tests/test_deflation.py::test_one_solve_equals_the_three_solve_sherman_morrison` keeps the two forms
tied together.

**What the speed actually is.** Measured in-process on a 2D `u_t = laplace(u) + lam u - u^3` at 14161
dofs with Pardiso, interleaved A/B against the old implementation: **66 back-substitutions become 22**
for the same 22 Newton steps and the same converged solution. End to end the two are within noise
(4.0-4.2 s vs 4.0-4.4 s over three passes each), because on a cheap 2D scalar problem with a direct
solver the elemental assembly is ~3.4 s of that and a back-substitution is ~10 ms. The reduction is
therefore worth having where a solve is expensive relative to an assembly -- 3D, a costly constitutive
model, or an iterative KSP, where three solves per Newton step is three times the Krylov work -- and is
close to free elsewhere. It is not the reason to prefer the new form; the MPI reductions and the
convergence test are.

A related defect surfaced while measuring, and is fixed: Pardiso's `op_flag==2` recomputes the backward
error itself for the symmetric mtypes, and it was doing so against the vector the branch RETURNS. With
the rescale applied there that vector is no longer a solution of `J x = b`, so the check condemned the
factorisation on every Newton step -- 21 full factorisations where 1 plus 21 phase-22 refreshes was
right. The raw solve and the Newton step are now kept apart in that branch.

## 3. Why the factor is normalised

oomph-lib's Newton loop converges on `max|residual| < newton_solver_tolerance` (`1e-8` by default),
and the residual it sees is `M R`. The unnormalised factor of the paper tends to `alpha^k` far from
`k` known solutions, so with the default `alpha = 0.1` **the tested residual shrank by ten per known
solution**: after eight branches any starting guess passed the test at once and deflation reported
the perturbed guess as a solution. Short of that it merely eroded the accuracy -- the old pitchfork
tutorial returned `x = -0.99999999`, eight digits where the same solve now gives fifteen.

Normalised, `M >= 1` everywhere and `M -> 1` far away, so deflation can only ever *tighten* the test.
The iterates do not move (§2). `tests/test_deflation.py::test_deflated_residual_never_shrinks` asserts
it through the real assembly, which is where the number oomph compares actually comes from.

## 4. Where it plugs in

Deflation is **not** a `CustomAssemblyBase`. It used to be, and that is what made it serial-only: the
custom assembler routes assembly through
`Problem::sparse_assemble_row_or_column_compressed_base_problem`, which throws for `nproc > 1`
(blockers B1/B2/B3 of [mpi_augmented_systems.md](mpi_augmented_systems.md) §8). Deflation never needed
any of that machinery -- it adds no unknowns and does not modify the Jacobian -- so it now sits on the
ordinary assembly through two small hooks:

| what | where |
|---|---|
| residual `*= M(U)` | `Problem::apply_residual_scale_factor`, called from `Problem::get_residuals` and `Problem::get_jacobian` (`src/problem.cpp`) |
| the virtual it calls | `Problem::get_residual_scale_factor()`, overridden in Python by `Problem.get_residual_scale_factor` |
| gated by | `Problem::residual_scale_hook_active`, a plain bool set from `Problem.set_deflation_operator` |
| step rescale | `GenericLinearSystemSolver._postprocess_newton_step` (`pyoomph/solvers/generic.py`), called from each backend's `op_flag==2` branch |
| the numerics | `DeflationOperator` in `pyoomph/generic/bifurcation_tools.py` |

The bool gate exists because overriding a C++ virtual in Python makes **every** call go through the
nanobind trampoline, and this one sits in every residual assembly of every problem. With the gate the
default costs one bool test.

What that buys, besides MPI: `use_custom_residual_jacobian` stays `False`, so the frozen sparsity
pattern, the proven-symmetry factorisation and the OpenMP assembly all stay available while deflation
is installed; and **every** linear solver works, including PETSc, which used to refuse the custom
solve routine outright and could therefore not be used for deflation even serially.

`DeflationAssemblyHandler` is kept as a thin `AugmentedAssemblyHandler` shell around the same
`DeflationOperator`, for scripts that construct it directly.

Refused combinations. At `set_deflation_operator`:

* **static condensation** -- it reconstructs the dofs it eliminated from the Newton increment, using
  operators built from the unrescaled residual, and deflation rescales that increment afterwards.
* **bifurcation tracking** -- the augmented system has its own residual and its own meaning for the
  increment.

And at `_adapt()` / `force_remesh()`, through `Problem._require_no_deflation`, but only once the
operator actually holds a solution: **adaptation and remeshing**. Every known solution is a dof vector
of the current numbering, which both change in length and in meaning. `add_known_solution`'s length
check would catch the next registration but says nothing about the factor computed from the ones
already stored -- and on a refinement that happens to preserve the dof count, not even that.

## 5. MPI

Two reductions, on **two different layouts**. Getting the layouts confused is the whole difficulty.

| quantity | layout | serial | replicated `mpirun` | `--distribute` |
|---|---|---|---|---|
| `\|\|U-W_i\|\|^2` | the **dof** vector | plain numpy | plain numpy (`Dof_pt` is the whole vector on every rank) | owned rows + `get_mpi_sum` |
| `grad(log M) . dU` | the **solver's** row block | plain numpy | **allreduce** | allreduce |

The second row is the trap `replicated_mpi_correctness.md` warns about: oomph row-splits the linear
algebra as soon as `nproc > 1`, `--distribute` or not, so the increment is a row block even when the
dof vector is replicated. `DeflationOperator.rescale_newton_step` therefore takes `first_row` and
`reduce_dot` from the **caller**, and decides whether its gradient is a local block or the whole
vector from `problem.is_distributed()`, never from a length -- on a small system rank 0 owns every row
and rank 1 owns none, so a length test picks the wrong branch there.

Three things had to be arranged, and each was found by a run that returned a wrong answer rather than
an error:

1. **`Problem::preferred_linear_solver_distribution` now asks for the dof distribution when deflation
   is active and the problem is distributed.** oomph's default there is a *uniform* row split, which
   is a different partition of the same rows: at 225 dofs on two ranks the dof blocks are 105/120 and
   the uniform ones 112/113, and the gradient and the increment are then not comparable entry by
   entry. (Static condensation uses the same hook for its own reason; the two are mutually exclusive.)
2. **The PETSc backend remembers `first_row` from the factorise call.** oomph passes a meaningful
   `first_row` only at `op_flag==1`; at `op_flag==2` it is `0` on every rank. Trusting the argument
   gave every rank the block starting at row 0 -- the deflated search then found only the trivial
   solution, with no error anywhere.
3. **The row counts are checked.** `rescale_newton_step` packs `len(y)` into the same allreduce as the
   dot product and refuses if the blocks do not tile the dof vector. One extra double per Newton step,
   and it turns exactly the failure in (2) into a message.

The gather-to-root fallback (`GenericLinearSystemSolver.solve_gathered_to_root`) used to refuse
deflation outright, because the old routine would have run three solves and two global dot products on
rank 0 while the other ranks waited. After §2 there is nothing to run there: the rescale is applied
**after the scatter**, on each rank's own block.

### Rank-independent control flow

The search is a sequence of random perturbations and Newton solves whose success decides what happens
next, so a rank that draws a different perturbation or disagrees about convergence takes a different
branch and the job deadlocks in the next collective. Both drivers therefore draw from a
`numpy.random.default_rng(random_seed)` (seed `0` by default) instead of the global numpy state.
`mpi_augmented_systems.md` §8 records an unexplained hang (B6) in an earlier prototype and names the
unseeded RNG as the prime suspect; it has not recurred since.

Seeding makes a run reproducible **for a fixed partition**. It cannot make it reproducible across
partitions: the perturbation is drawn in dof-index space and `distribute()` renumbers, so global index
`i` is a different node at `np=2` than at `np=3`. `tests/mpi_deflation_worker.py` drives the search
with the leading eigenvector as well, which is a field and therefore means the same thing however the
mesh was cut up; with that, every configuration finds the whole solution set.

## 6. Driver defects fixed on the way

All in `pyoomph/generic/problem.py`, all found by reading rather than by a failing test -- there was
no test.

* **A re-found solution opened a new branch.** Deflation stops Newton converging *onto* a known
  solution; it does not stop it converging arbitrarily *close* to one, and at a bifurcation point the
  branches meet. Both drivers now compare a converged solution against the ones they already hold
  (`Problem._deflation_solution_is_new`, a norm over the globally gathered dof vector, so no reduction)
  and treat a duplicate as a failed attempt. This is what produced the one-point stub branches in the
  deflated-continuation diagram.
* **The branch-reorder block in `deflated_continuation` was dead code twice over.** `mindist` started
  at the distance from the failed branch's stored dofs to themselves, which is exactly zero because a
  branch that failed to continue was never updated; and the body assigned `cdist = mindist` instead of
  `mindist = cdist`, so it would not have tracked the minimum either. Live, "nearest survivor" alone
  turned out to be the wrong test as well -- on the pitchfork the trivial branch is simply the nearest
  thing to a lower branch that has just died, and taking it over renamed the trivial branch halfway
  through the diagram. A swap now additionally requires the survivor to be nearer the *dead* branch's
  old position than its own.
* **`deflated_solve_by_eigenperturbation` ignored its `eigenindex`** (it always perturbed with
  eigenvector 0) and did not take the real part of a complex eigenvector.
* **`deflated_continuation` did not initialise the problem**, so a parameter defined in
  `define_problem()` was reported as not existing.
* Bare `except:` in the three drivers, which swallowed `KeyboardInterrupt`.

## 7. Cover

| | |
|---|---|
| `tests/test_deflation.py` | 17 serial tests: the closed form against the literal Sherman-Morrison update, `grad log M` against finite differences in all three shift modes, the far-field and near-field limits, the residual-never-shrinks regression, the pitchfork ODE and its deflated continuation, and the two refusals |
| `tests/test_mpi_deflation.py` + `tests/mpi_deflation_worker.py` | `np=2` plain, `np=2 --distribute`, `np=3 --distribute`, both drivers, against an in-process serial run. Marked `slow` (`--full`). A 2D `u_t = laplace(u) + lam u - u^3` whose symmetric pair has the *same* integral of `u^2` and is distinguished only by the signed one |
| the two tutorial scripts | no longer skipped under `--mpirun` in `citools/test_all_tutorial_scripts.py` |

## 6a. alpha has units, and what that costs

`alpha` multiplies `||U-W||^p`, so it is not a dimensionless number: the same physics written in
different units is a different deflation problem. Measured on a pitchfork PDE
`u_t = laplace(u) + lam*u - u^3/S^2`, whose bifurcating branch has amplitude `S`, scanning `lam`
across the bifurcation and counting how many of the three known solutions the scan recovers at each
parameter value:

| solution scale S | alpha | branches found |
|---|---|---|
| 1 | 0.1 (the default) | 21 / 21 |
| 1e-3 | 0.1 | **7 / 21** (only the trivial one) |
| 1e-3 | 1e4 ... 1e5, i.e. `0.1*S^-p` | 21 / 21 |

So the defaults were never wrong, they were only right at one scale. The fix is to give the operator a
**length** to measure distances in, `DeflationOperator.scale`, and let `alpha` be dimensionless in
units of it. With the scale set to the perturbation amplitude the same `alpha` now scores identically
at `S = 1e-3`, `1` and `1e3` - `alpha` between 0.03 and 3 all score 0.95-1.0, which is why the default
stayed at 0.1 rather than moving to Farrell's 1.

Two things about the scale that are easy to get wrong, and one that cannot be fixed:

- **It must be a constant of the operator, not a function of `U`.** A scale recomputed from the
  current iterate is always the right order, which is exactly the problem: it follows Newton away from
  the known solution, `||U-W||/L` stays near 1, and the deflation never decays. Measured, that is
  worse than no scale at all (21/21 became 12/21), and it also invalidates the analytic gradient,
  which is derived with `L` held fixed. `auto_scale_from()` therefore sets it once, when the search
  starts.
- **The perturbation amplitude is the honest choice for it.** It is already the user's statement of
  how far away a different solution might be, and it is the only length available when the state being
  deflated is the trivial one. That collapses two coupled knobs into one.
- **On a state that is exactly zero there is no length in the problem at all.** Nothing can be
  derived, the scale falls back to 1, and the amplitude has to be set by hand - which is worth doing,
  since it is what the whole search is measured in. The GUI says so on the tab rather than leaving it
  to be discovered.

## 7a. In the bifurcation GUI

The Deflation tab wraps both drivers; what it does with them, and why a scanned point is recorded
without a spectrum, is in [bifurcation_loci.md](bifurcation_loci.md) "The Deflation tab".

One thing the GUI needed from the drivers here: both generators now do their teardown in a `finally`.
A caller is free to stop consuming a generator half-way - the Abort button does exactly that - and
plain trailing code after the last `yield` does **not** run when a generator is closed, so the
deflation operator used to stay installed on the problem and quietly deflate every later solve.

## 8. Open

* **The random perturbation is not partition-independent** (§5). Drawing it from a
  `DeterministicRandomField` over the node coordinates instead of from dof indices would make a
  deflated search reproduce across serial, replicated and distributed runs. Worth doing; not needed
  for correctness.
* **`deflated_continuation` cannot connect branch points**, which is intrinsic to the method
  (Farrell 2016) -- the two halves of one arclength branch come out as two branch indices.
* **`dparameter` is not implemented**, so deflation cannot be combined with arclength continuation.
