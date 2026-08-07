# MKL Pardiso static pivoting: one cause, two failures, only one of them audible

Written 2026-08-02 on branch `develop`, chasing the intermittent-looking Pardiso failure seen while
building the heated-cylinder tutorial (`docs/source/tutorial/pde/adapt/cylinder.rst`). The reported
symptom was

```
PardisoError: MKL Pardiso failed during the solve and iterative refinement: error -4 (zero pivot,
numerical factorisation or iterative refinement problem (the matrix is singular or nearly so at this
state)). The solution vector is meaningless, so the solve is aborted here rather than handing it on
as an answer.
```

Everything below was measured. The reproduction driver is the tutorial's own
`HeatedCylinderProblem(criterion="joint")`, whose joint error criterion deliberately leaves an
advection-dominated tracer under-resolved; the only variable is `desired_ndof`.

---

## 1. It is not intermittent, and it is not one failure

The original report described the failure as intermittent. It is not. At fixed settings it is fully
deterministic — three runs at `desired_ndof=250000` failed identically, at the same ndof. What looks
like intermittency is that **the onset is not monotone in the budget**: what matters is the
particular mesh the `desired_ndof` controller happens to land on, not how large it is.

Classifying each run by the actual error rather than by exit status gives:

| `desired_ndof` | outcome | perturbed pivots (IPARM(14)) |
|---|---|---|
| 80 000 | OK (ndof 77 742) | 0 |
| 150 000 | OK | 0 |
| 175 000 | **Newton stall** | 0 |
| 200 000 | OK | 0 |
| 225 000 | **Pardiso −4** | 2 |
| 250 000 | **Newton stall** | 0 |
| 350 000 | **Pardiso −4** | 65 |

The `-4` correlates exactly with perturbed pivots. The other failures reported
`MAXIMUM NUMBER OF ITERATIONS (10) REACHED WITHOUT CONVERGENCE` and looked, from the outside, like a
physics or continuation problem rather than a linear-solver one.

They are not. Instrumenting every solve with its backward error
`||Ax-b||_inf / ||b||_inf` settles it:

| run | solves | median backward error | worst |
|---|---|---|---|
| budget 80 000 (healthy) | 12 | 1.1e-09 | 9.9e-07 |
| budget 250 000 (stalling) | 20 | **1.9e-01** | **7.6e+00** |

So in the stalling case Pardiso reports a **perfectly clean factorisation** (`IPARM(14) == 0`),
therefore performs no iterative refinement at all, and returns a solution that is wrong by order
unity. Newton is being fed nonsense. This silent mode is considerably more dangerous than the `-4`,
which at least announces itself.

Raising `max_newton_iterations` from 10 to 30 changes nothing, which is the direct confirmation that
the Newton iteration is not the problem.

## 2. The matrices are not singular

Whatever MKL's message says, these systems are fine. UMFPACK — which pivots dynamically with a
threshold criterion, rather than statically — solves every one of them. At budget 250 000 it returns
ndof 240 172 where Pardiso stalled at 219 279. It is simply much slower: 206 s against Pardiso's
~35 s.

This is what identifies the cause as MKL's **static pivoting**: a pivot that comes out too small is
perturbed rather than exchanged, and repairing the damage is left to iterative refinement. On these
meshes that repair either fails loudly (`-4`) or is never attempted because MKL did not think it
needed it.

## 3. What cures it

Sweeping MKL's documented remedies at the two `-4` budgets. Keys are the 1-based IPARM numbers that
`pardisoSolver`'s `iparm_override` takes:

| setting | 225 000 | 350 000 |
|---|---|---|
| baseline | −4 | −4 |
| `IPARM(11)=1, IPARM(13)=1` (scaling + matching) | −4 | −4 |
| `IPARM(13)=2` (two-level weighted matching) | **OK** | **OK** |
| `IPARM(10)=8` (perturb doubtful pivots earlier) | OK | −4 |
| `IPARM(10)=20` | Newton stall | −4 |

Scaling and matching are already on by default for `mtype=11`, so asking for them explicitly is a
no-op — worth knowing, because they are the first thing MKL's documentation suggests.

`IPARM(13)=2` alone fixes both `-4` cases and the 175 000 stall, but not the 250 000 stall. Adding
`IPARM(10)=8` fixes that too. The pair is what `_ESCALATED_IPARM` in
[pardiso.py](../pyoomph/solvers/pardiso.py) now contains.

Note the counter-intuitive detail: with `IPARM(13)=2` the number of *perturbed* pivots goes **up**
sharply (2 → 194 at budget 225 000) and the solve nonetheless succeeds. The perturbation count is
therefore not a measure of trouble; what matters is whether refinement can repair it, and after the
better matching it can. Do not use IPARM(14) as a health indicator.

## 4. The fix

Two changes in `pardisoSolver`:

- `solve_checked` now computes the backward error of **every** solve, not only of the ones MKL flags.
  If it exceeds `_BACKWARD_ERROR_LIMIT`, or if the solve raised, the factorisation is rebuilt once
  with `_ESCALATED_IPARM` and the solve retried. `IPARM(13)` is read by the *reordering* phase, so
  this cannot reuse the existing analysis — it must go back through phase 12, releasing the handle
  first or MKL leaks its workspace.
- `PardisoSolver.solve` folds a *kept* escalation into `iparm_override`, so the next factorisation
  starts there. Under spatial adaptivity the mesh only gets harder, so rediscovering the need through
  another failed solve on every refinement would be pure waste. (It originally folded in every spent
  escalation, whether or not it had helped. That is §7.)

The threshold is **1e-4**: healthy solves peak at 1e-6 and broken ones sit at 1e0, so anything in
between separates them, and 1e-4 keeps two orders of margin on the side that must not fire — a false
positive costs a full refactorisation.

If the escalation does not help, the code **warns and returns the better of the two answers rather
than raising**, and (since §7) puts the previous `IPARM` back. Before this check existed such a solve
was returned silently, so refusing it outright would convert badly-converging runs elsewhere into
crashing ones. That is a deliberate asymmetry: the `-4` path, which had no usable answer to begin
with, still re-raises.

### Measured afterwards, all with the default solver and no overrides

- All seven budgets from 80 000 to 350 000 pass.
- Results are **bit-identical** wherever they already worked, and budget 250 000 now reproduces
  UMFPACK's answer exactly (ndof 240 172, `|dofs|` 2.1318565107e+02) roughly six times faster.
- **Zero escalations** on the healthy 80 000 run: the check is inert when nothing is wrong. 11 s
  against 11.7/12.3 s before, so the extra sparse mat-vec per solve is below the noise.
- `tests/test_adaptive_3d_campaign.py --full`: 265 passed, no warnings emitted. That is the
  hex+pyramid ALE box the pre-existing refinement comment in `solve_checked` was written for, and so
  the path most at risk from this change.

## 5. Corrected along the way

The comment on the early return in `solve_checked` read *"IPARM(14): no pivot was perturbed, so the
factors are trustworthy"*. §1 is the counter-example: the worst solves measured came out of
factorisations reporting exactly that. Fixed in place.

## 6. Left open

- **The `try_to_reuse_solver` branch is untouched.** It has an accuracy check and a refactorisation
  fallback of its own, but no escalation, so it can still spin on a matrix that needs one. It
  defaults to `False`, so nothing hits it unless asked.
- **Why these meshes at all.** The failures cluster on the *joint*-criterion meshes, i.e. exactly the
  ones where the tracer is left under-resolved. A sharply under-resolved advection-dominated field
  presumably produces the near-singular blocks that defeat static pivoting, but that was not pinned
  down — the tracer's own Péclet number turned out not to matter, because under the joint criterion
  it cannot influence the mesh at all.
- **The tutorial is unaffected** at its shipped budget of 80 000, so no tutorial change was needed;
  `set_linear_solver("umfpack")` is correctly absent from it.

---

## 7. Follow-up: an escalation that did not help must be withdrawn

Written 2026-08-03, from the first nightly (`citools/nightly_develop.sh`) to run with §4 in it. It
failed three tutorials that had passed the night before — `two_layer_flow_single_domain`,
`droplet_spread_marangoni_and_gravity` and `eigenbranch_continuation` — and all three were the same
defect, in the escalation rather than in the check that triggers it.

### What went wrong

`_escalated_iparm` was doing two jobs at once: *the escalation has been tried* (a one-shot guard, so
it cannot loop) and *the escalated settings are in force* (which is what the `iparm_override`
carry-over reads). Once set, it was never cleared. So a **single** marginal solve escalated, found
the escalation no better, correctly returned the pre-escalation answer — and then left
`IPARM(13)=2, IPARM(10)=8` in force on that handle *and* folded them into `iparm_override` for every
factorisation the problem would ever build afterwards. §3 already recorded that these knobs can cause
a Newton stall where they are not needed (`IPARM(10)=20` did exactly that at budget 250 000); this
made one bad solve impose them on the rest of the run.

The three failures are the same story at three severities, and in each the trigger was a solve barely
over the 1e-4 line:

| tutorial | backward error | after escalating | what happened next |
|---|---|---|---|
| `droplet_spread_marangoni_and_gravity` | 4.4e-04 | 5.3e-04 | every later arclength step stalled at a residual of 1.1e-8 instead of reaching 1e-14, `ds` halved to below its 1e-10 minimum |
| `two_layer_flow_single_domain` | 1.6e-01 | 2.1e+26 | the next Newton step's residual was `inf`, at t=0.5 of 50 |
| `eigenbranch_continuation` | 1.7e-04 | 5.9e+08 | ran on for another 3 000 log lines and then diverged, at the fifth of seven `create_Bond_curve` calls |

Worth noting what makes this hard to see from a log: the carry-over fired **silently** whenever the
escalation *succeeded*, and the warning that did print named only the retry's error. In the
eigenbranch log the one `PARDISO WARNING` sits 3 300 lines before the traceback, with nothing in
between connecting them.

Note also *where* it triggers: the eigenbranch escalation happens on an arclength step that is
already diverging (residual 5e9, `L` = −34), i.e. the backward error is being measured on an iterate
the step is about to reject anyway. The trigger is doing its job; the matrix really is bad. It is the
permanence that was wrong.

### The fix

- The two meanings are now two flags. `_escalation_spent` keeps the one-shot guard; `_escalated_iparm`
  means only *in force right now*, and the carry-over reads that.
- `_deescalate_pivoting()` restores the saved `IPARM` and refactorises when the escalated solve's
  backward error is not an improvement. Costs one extra factorisation in the false-positive case, and
  the settings never leave the solve that asked for them.
- `_escalate_pivoting()` also restores them if its own `factor()` raises, so a retryable
  `SolverError` hands back a handle that can still be factorised on the next, smaller step.
- The `-4` path no longer loses its diagnosis. On a singular matrix the escalated reordering gives up
  with `-6 (reordering failed)`, which describes the retry and not the matrix; the original error is
  what is raised now. `tests/test_solver_failure_recovery.py` had been matching on `error -4` and was
  failing for a *third* reason — MKL 2025.0 refines the perturbed pivot away and returns error 0 with
  a solution of order 1e13, so no `-4` is produced on this machine at all (it failed with
  "DID NOT RAISE" the night before §4 landed). It now asserts what is version-independent: the huge
  solution is refused, as a retryable `SolverError`.

### Measured

- All three tutorials pass, with the complex PETSc the nightly uses. The escalations still fire, at
  the same solves, and are withdrawn: `droplet_spread` 4.4e-04 → one withdrawal → rc 0;
  `two_layer_flow` 1.6e-01 → one withdrawal → runs to t=50; `eigenbranch_continuation` 1.9e-04 and
  1.7e-04 → two withdrawals → rc 0, all seven curves written.
- `tests/test_solver_failure_recovery.py` 9 passed; `test_structural_assembly.py` +
  `test_newton_abort.py` 46 passed.
- Not re-run here: the heated-cylinder budget sweep of §4 (its 250 000-dof cases are over this
  machine's 200 000-dof working limit) and `test_adaptive_3d_campaign --full`. The escalation itself
  is unchanged for the case where it *helps*, which is the case those measured.

---

## 8. A third failure mode, far below the escalation threshold, and not worth escalating

Written 2026-08-07. Found while running `test_adaptive_3d_campaign --full` — the run §7 explicitly
did not do — where `test_ale_moving_mesh[levels0-hex_pyr]` failed at
`max|residual| = 8.583e-09` against the ALE tolerance of 1e-11.

It looks like §1 and it is not. Nothing here is near-singular: the escalation of §4 triggers on a
backward error of 1e-4, healthy solves peak at 1e-6, and this solve is orders of magnitude cleaner
than either — Pardiso considers it, correctly, a good solve. Nor is it a bad Jacobian: the same
assembled system handed to SuperLU or UMFPACK gives `5.6e-15` from the identical starting residual of
`1.185e-01`. The answer is right; only the solve was imprecise.

Measured over all 33 ALE cases of the sweep (11 layouts x 3 refinement states), worst `max|residual|`:

```
Pardiso   4.4e-16 .. 1.1e-14      except 3d-ale-hex_pyr-00 at 8.6e-09
SuperLU   5.3e-15 .. 1.2e-13
```

Two things follow, and the second is the reason this is a section rather than a one-line fix:

* it is **one** case out of 33, not a property of ALE or of pyramids — `3d-ale-hex_pyr-11` and
  `-12`, the same layout refined, are at 7.9e-16 and 1.1e-15;
* Pardiso is normally **one to two orders better than SuperLU**. Switching the sweep to an "exact"
  solver to be safe would make 32 cases worse in order to fix one, and would have moved several from
  1e-16 to 1e-13.

So `test_adaptive_3d_campaign.py` names the single case in `_EXACT_SOLVER_CASES` and passes
`linear_solver="superlu"` for it through `box_cases_3d.solve_case`. That is a statement about what the
test is for — it certifies the discretisation and the analytic Jacobian by driving a linear problem to
machine zero in one Newton step, which needs the linear solve to be exact to roundoff — and not a
claim that Pardiso is wrong here. 8.6e-9 is a perfectly good answer for a solver to return; it is
simply not machine zero, and this test measures machine zero.

Deliberately NOT done: lowering `_BACKWARD_ERROR_LIMIT` so the §4 escalation catches this. §7 is the
argument against — the escalation is expensive, its knobs can themselves cause a Newton stall where
they are not needed, and the trigger would have to drop by four orders to reach a solve that nothing
is actually wrong with. The other option, relaxing the ALE tolerance from 1e-11 to 1e-8, was
considered and rejected for the same reason the tolerance is 1e-11 in the first place: it is the
tightness that makes the case a Jacobian oracle at all.
