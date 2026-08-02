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
- `PardisoSolver.solve` folds a spent escalation into `iparm_override`, so the next factorisation
  starts there. Under spatial adaptivity the mesh only gets harder, so rediscovering the need through
  another failed solve on every refinement would be pure waste.

The threshold is **1e-4**: healthy solves peak at 1e-6 and broken ones sit at 1e0, so anything in
between separates them, and 1e-4 keeps two orders of margin on the side that must not fire — a false
positive costs a full refactorisation.

If the escalation does not help, the code **warns and returns the better of the two answers rather
than raising**. Before this check existed such a solve was returned silently, so refusing it outright
would convert badly-converging runs elsewhere into crashing ones. That is a deliberate asymmetry: the
`-4` path, which had no usable answer to begin with, still re-raises.

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
