# Branch switching, and why seeding oomph's arclength state does not work

`BifurcationController.branch_switch` steps from a bifurcation onto the *other* branch through it. It
used to seed oomph-lib's continuation state and take an arclength step; that quietly did not work, and
this records why, since the same trap catches anything that tries to hand oomph a tangent.

## The failure

On `du/dt = mu*u - u^2` (branches `u = 0` and `u = mu`, crossing transcritically at `mu = 0`), the old
implementation located the bifurcation correctly, classified it correctly as transcritical, recorded two
branch-switch tangents - and then produced a "new branch" sitting at `u = 0`, i.e. the branch it started
on. It jumped to `mu = 1.0` on the first step with `ds = 0.1125`.

Silent, because the trivial branch exists for every `mu`: there is always a solution for Newton to find,
so nothing fails, it just does not switch. Any test has to check *which* branch was reached.

## Why

The two setters take **derivatives**, not increments:

```
Problem::update_dof_vectors_for_continuation(ddof, curr)   ->  Dof_derivative[i] = ddof[i]
Problem::update_param_info_for_continuation(dp, p0)        ->  Parameter_derivative = |dp|
```

and oomph normalises its own tangent, in `calculate_continuation_derivatives_helper`:

```
Parameter_derivative = 1 / sqrt(1 + Theta_squared * chi)
dof_derivative(l)    = -Parameter_derivative * z(l)
```

which is the constraint

    (dparameter/ds)^2 + Theta^2 * |dU/ds|^2 = 1

The old code passed `tangent * ds` to both - an increment. The seeded tangent therefore had norm about
`ds` instead of 1, and the arclength constraint `Delta . d = ds` then asks for a step of length about
`ds/|d|`, i.e. **inflated by 1/ds**. With `ds = 0.1125` that is a factor of nine, which threw the
predictor far past the bifurcation, and Newton settled on the branch that exists everywhere.

So a seeded tangent must be normalised to the constraint above. This is the same fragility recorded in
`bifurcation_loci.md` §4, where seeding a fold's null direction (`dparameter/ds = 0`) instead produced
`Ds = -nan` and made oomph retry for ever.

## What works instead

Predict a point on the new branch from the normal form and do a **regular Newton solve** there - no
arclength state involved. A short distance off the bifurcation the Jacobian is regular again, so the
solve converges, and the prediction decides which branch it converges to. `NormalFormCalculator` supplies
exactly what is needed:

| type | `param_predictor(eps)` | `perturbation_predictor(eps)` |
| --- | --- | --- |
| transcritical | `eps` | `-zeta * 2*b1/b2 * eps` |
| pitchfork | `psign * abs(eps)` | `sign(eps) * zeta * sqrt(abs(6*b1/b3 * eps))` |
| fold | `0` | `zeta * eps` |

So `parameter = p0 + param_predictor(eps)`, `dofs = base + perturbation_predictor(eps)`, then `solve()`.
Both signs of `eps` and a ladder of magnitudes are tried, because which side the new branch lies on is
not known in advance. A fold is refused outright: it has one branch through it, which turns around -
getting off *that* is `leave_locus`.

The `sign(eps)` on the pitchfork's perturbation is not decoration. A pitchfork's two branches sit on the
**same** side of the parameter and differ only in the sign of the amplitude, so with `abs()` on both
predictors the two directions asked for the very same point, one arm was unreachable, and the direction
argument did nothing. `tests/branch_switch_worker.py` now pins that `direction=+1` and `direction=-1`
reach `u = +sqrt(mu)` and `u = -sqrt(mu)`.

### The ladder, and how narrow the window can be

The offset that works can be very sharply defined. On the thin-film branch point this was written for,
40 offsets spanning three decades were tried by hand, and **exactly one landed** on the new branch:

| offset | outcome |
| --- | --- |
| 2.45e-2 … 1.2e-2 | converges back onto the branch we came from |
| **8.4e-3** | **lands on the new branch** |
| 5.9e-3 … 1.4e-3 | Newton diverges |
| 9.9e-4 and below | back onto the old branch again |

Its neighbours a factor 1.43 either side fail, and in opposite ways, so there is no monotonicity to
bisect on. Scanning the amplitude at a fixed offset does not rescue it either - at an offset of 1.2e-2
every amplitude from 0.02 to 0.9 diverges, because the new branch has already turned around by there
and no solution is near any guess.

Hence: a **geometric ladder from `3*offset` downwards over three decades, spaced by `offset_ratio`
(1.25 by default, i.e. 21 rungs)**, stopping at the first success, with `max_attempts` capping the
Newton solves so a large problem cannot spend minutes failing. The earlier
`(eps, eps/10, eps*10)` was far too coarse to find a window like that, and its `eps*10` rung pointed
the wrong way: the prediction is asymptotic in the distance from the bifurcation, so *too long* is the
failure that actually happens.

For a **pitchfork** the mirrored parameter side is tried as well - at each rung, not after the whole
ladder. Which side the two arms are on is the sign of `6*b1/b3`, and `b3` is the least trustworthy
number in the normal form (a third derivative through a nearly singular solve); on the thin-film case
it came out wrong, and running 21 hopeless offsets before looking at the other side is 21 wasted
solves. Found on the mirrored side, `switch_branch` says so - it is the sub- against supercritical
statement.

### Where the offset comes from

`BifurcationController.branch_switch_parameter_offset()`: **the ds you are continuing with**. ds is an
arclength step, so what it buys in the parameter is `ds*|dparameter/ds|`, taken from the tangent
recorded at the point *before* the bifurcation - the one at the bifurcation is gone, see the arrows
section below. The direction likewise defaults to `sign(ds)`. So `+`/`-`/`/` steer a branch switch the
same way they steer a sweep, which is what a user who has just tuned them expects; both are still
overridable per call.

The fallback, when there is no tangent to ask, is what this always used: `branch_switch_offset` (2%) of
the parameter. On a diagram spanning 4% of its parameter that overshoots every branch in it.

### Accepting a landing

Judged by the **amplitude in the critical mode**: `du` points along it, and the branch we came from has
none of it, since the normal form's own splitting puts that branch's tangent in the complement of the
kernel. So project what Newton did onto `du` and require at least a tenth of the predicted amplitude.
Measured at a thin-film branch point: landings on the new branch score 0.9 to 9 times the prediction,
every fall-back onto the old branch scores 1e-4 or less. There is nothing to tune.

This used to be judged against the old branch's arclength tangent instead, which fails twice over. The
prediction is only leading order, so a genuine landing routinely differs from `du` by more than the 50%
that test allowed - and `get_arclength_dof_derivative_vector()` is **empty** unless an arclength
continuation ran in this session, so after a `go_to_param` walk, or on a diagram loaded from disk, every
candidate was rejected out of hand and the switch always reported failure.

## The second half of the bug

Landing correctly is not enough. With the predict-and-solve above, the switch reached `u = mu` exactly -
and the *first continuation step afterwards* went straight back to `u = 0`.

Just off the bifurcation the Jacobian is still nearly singular, so `dU/dparameter` is large and badly
conditioned, and a step of the size the old branch had reached (`ds = 0.1125` against an offset of
`0.02`) overshoots. `branch_switch` therefore sets `ds` from the offset it just took - **a quarter** of
it. The offset that works is often the largest the ladder tried, because the smaller ones fall back onto
the branch we came from, and a branch born at a bifurcation can be short: on the thin-film case,
continuing at the full offset stepped over the new branch's own fold on the very first step and landed
back on the trivial branch, while a quarter of it traced the branch to its turning point. The arclength
step grows it again by itself once the branch is well separated.

## Where it lives

The numerics are on `Problem`, so a plain script can use them without the bifurcation GUI:

```python
problem.solve_eigenproblem(1)
problem.activate_bifurcation_tracking("gamma")
problem.solve()                              # now AT the bifurcation
nf = problem.classify_bifurcation("gamma")   # "fold" / "transcritical" / "pitchfork" / "Hopf"
ds = problem.switch_branch("gamma")          # now on the other branch; None if none was reachable
while ...:
    ds = problem.arclength_continuation("gamma", ds)
```

`switch_branch` deactivates bifurcation tracking (the switch needs the plain system), tries both sides
and a ladder of offset magnitudes, verifies where it landed, and returns a step size for carrying on - a
quarter of the offset it took. It raises for a fold rather than returning None, since that is a statement
about the bifurcation and not a failure to converge. `classify_bifurcation` warns when the critical eigenvalue is
not near zero: the calculation returns "fold" for a regular point, and saying nothing would make that
look like an answer.

`BifurcationController.branch_switch` is now only the diagram half - which point we are at, opening a
branch for the result, and the step size to carry on with.

## Verified

`tests/branch_switch_worker.py`, one process per bifurcation type and per API (a second `Problem` in one
process segfaults in the JIT loader), through the GUI and through `Problem.switch_branch` directly:

- transcritical: lands on and stays on `u = mu` - `(0.020, 0.020), (0.033, 0.033), (0.047, 0.047), ...`,
  max error against the closed form **7.7e-8**
- pitchfork: lands on and stays on `u = sqrt(mu)` - `(0.020, 0.141), (0.040, 0.200), (0.061, 0.247), ...`,
  max error **9.5e-8**, and `direction=+1` / `-1` reach the two arms
- fold (`du/dt = mu - u^2`): stays classified as a fold and refuses to switch. The other half of the
  classification, see below - a measure merely loosened until nothing is a fold any more would pass the
  two tests above.

The first two check every point after the switch, not just the first, because that is where the old
behaviour came apart.

## Fold or branch point: measure the angle, not the number

Which of the two it is comes down to `a = <-dR/dparameter, zeta*>`, zero at a branch point. That number
carries **no scale**: `zeta*` is normalised by `1/<zeta,zeta*>`, so a mode whose left and right null
vectors overlap poorly - which is exactly what a second, nearly critical eigenvalue does - inflates `a`
by that factor alone, and an absolute threshold then reads a branch point as a fold. That is what
reported the thin-film branch point at `beta_mu = 1.2716` as a fold, and a fold has nothing to switch to,
so the branch was unreachable.

What does not move is the **angle**: `a = 0` says `dR/dparameter` lies in the range of the Jacobian, and
`|a| / (|R01| |zeta*|)` says that scale-free. Measured:

| case | `a_rel` |
| --- | --- |
| `du/dt = mu - u^2`, the 1-dof fold | 1.0 |
| Bratu `u'' + lambda e^u = 0`, 200 elements, the fold at `lambda = 3.5138` | 0.94 |
| thin-film branch point, 20002 dofs | 2.7e-5 |

Three decades either side of the 1e-3 threshold, and anything within one decade of it says so rather
than deciding silently. `classify_bifurcation(..., assume="fold"|"branch_point")` overrides the decision
where the caller knows better - and the GUI does know better: a fold is where the parameter turns around,
so both points either side of one lie on the same side of its parameter value, while a branch point sits
strictly *between* the last two continuation points. `BifurcationController._fold_ruled_out_by_the_branch`
passes that on.


## Picking the eigenvalue to track

`BifurcationController.critical_eigenindex` used to answer "nearest the imaginary axis", which is right
on a branch that was already unstable when the sweep arrived - the KS hexdot branch past its fold, where
the transcritical point at `gamma = 0` belongs to the *second* eigenvalue - and wrong immediately after a
crossing. Two modes going unstable within a few thousandths of each other is what a periodic domain does
routinely: the thin-film spectrum one step past the crossing was

    [+0.016507, +0.016484, -0.016089, ...]

where nearest the axis is the **stable** eigenvalue, by 2%. Tracking it asks the fold handler for a
bifurcation the branch has not reached, from a starting guess belonging to a different mode; it diverged
outright.

So the recorded stability of the last two points decides: where the step just made the branch (more)
unstable, take the least unstable of the modes now on the wrong side - its crossing is the one between
those two points. Where the count did not change, the old rule stands.

## Leaving a branch transiently

`transient_leave_branch` perturbs along an eigenfunction and integrates until the solution settles
somewhere else. Two things have to hold, and neither did.

**The step size must stay below the growth time of the mode.** A fully implicit step far above it does
not amplify an unstable mode, it damps one - BDF2's amplification factor tends to zero as `lambda*dt`
grows - and the adaptive stepper walks straight into that, because the solution sits near a stationary
state, the temporal error is tiny, and `dt` doubles every step. Measured on the thin-film branch: `dt`
reached 31000 against a growth time of 60, and the run came back to the very solution it was told to
leave, with the same eigenvalue to six digits, reported as a new branch. Hence `maxstep`.

**The perturbation must be scaled to something meaningful.** It now comes from
`Problem.perturb_by_eigenfunction`, which bisects the amplitude to a target initial residual and fills in
the history dofs for the exponential growth, instead of a fixed `0.1 *` an eigenvector of arbitrary norm
with an impulsive start. That bisection had its own hole: a perturbation big enough to push a field out
of the domain its equations are defined on - a negative film height under a `1/h^6` disjoining pressure,
a negative concentration under a `log` - gives `inf` or `nan`, which compares as neither larger nor
smaller than the target, so both of its loops fell through and the **full** eigenvector was kept. The
transient then died on an infinite residual at its first step. Non-finite now counts as too large.

`tests/transient_leave_worker.py` pins both on `du/dt = mu*u - u^3 - 1e-6/(1-u^2)`, whose `1/(1-u^2)` is
infinite at `|u| = 1`, exactly where a full-amplitude perturbation of the unit eigenvector lands.


## The arrows on the diagram

The black arrow at the current point is `ds * [dparameter/ds, dobservable/ds]`, the direction the next
continuation step will go. **At a located bifurcation there is no such direction to read**: the
normal-form calculation deactivates bifurcation tracking, which resets oomph's continuation state, so
the dof-derivative vector comes back empty and `dparameter/ds` reads back as 1. Filling that in
produced `[1, 0]` - a horizontal arrow of length ds at every located bifurcation, pointing at
increasing parameter for reasons that had nothing to do with the diagram. `_update_tangents` now draws
none rather than one that means nothing.

What is drawn there instead are the **branch-switch arrows**, brown, one per branch through the
bifurcation: for a transcritical the two sides, for a pitchfork both arms (which sit at the same
parameter offset and differ in the sign of the amplitude - drawing one would suggest the other does not
exist). Each is the chord to the point `switch_branch` would predict at the offset it would actually
use, so the arrow shows where switching *aims*; it is stored divided by `|ds|` because the plotter
multiplies by that again, and by `|ds|` rather than `ds` because reversing the direction of travel does
not swap the two branches over. They used to be computed for transcritical points only, from
`dparameter/ds` - which at a bifurcation is the 1 above, and is not a parameter offset in any case.

## Repairing a diagram by hand

Two operations on the model, for when a sweep has gone somewhere the numerics cannot undo:

- **Split** (`split_branch`, `x`): cut a branch in two, the selected point starting the new half. This
  is for the ordinary failure above - a continuation step lands on a different branch and everything
  from there on belongs somewhere else. Nothing is recomputed and no state file is touched; the points
  are the same objects, they just stop being drawn as one curve.
- **Merge** (`merge_branches`, `X`): join the selected branch into the current one. Which of the four
  end-to-end pairings is used is decided by distance in the plotted coordinates, so a branch computed
  in the opposite direction is turned over rather than joined backwards, and the size of the joint is
  reported. Refused across different branch kinds, continuation parameters, or slices of parameter
  space - joining those would produce a curve that is not a section of anything.
