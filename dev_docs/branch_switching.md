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

## The projection has to be SIGNED

The acceptance test above was written `abs(<moved, du>) < 0.1 |du|^2`, and the `abs()` was wrong. A
pitchfork's two arms differ only in the *sign* of the amplitude, so a landing on the arm opposite the
one `direction` asked for passed the test and was reported as that direction's result. `direction` did
pick which point is *predicted*; it just did not constrain which one Newton was allowed to reach.
Dropping the `abs()` costs nothing and makes it a guarantee - nothing becomes unreachable, because
`sides` tries both signs either way.

The `|moved| <= 1e-9` guard above it goes with it: an absolute threshold on a dof vector of arbitrary
scale (a problem whose dofs are O(1e-6) never leaves it), and subsumed by the projection test anyway,
since a landing that did not move has `<moved, du> ~ 0`.

## A Hopf has no second steady branch

`switch_branch` refuses one and points at `Problem.switch_to_hopf_orbit`. It used to die with a bare
`TypeError` from inside a lambda: a Hopf's normal form *does* carry both predictor keys, so the "no
branch predictor" guard could not catch it, and its `perturbation_predictor` takes `(dp, omega*t)` and
returns an absolute state rather than a perturbation. The bifurcation GUI has always routed a Hopf to
`switch_to_orbit` first, so only the plain-script API documented in the docstring ever saw this.

## Pitchfork or transcritical: the same problem, and the zero it cannot normalise

The fold/branch-point decision was made scale-free above. The second decision had the same defect in a
worse form: the test was

    100*|b2/2| < |b3/6|

comparing `b2` against `b3` **directly**, and those carry different powers of `zeta`. With `zeta`
normalised to unit **Euclidean** length its entries are of order `N^-1/2`, so `b1 ~ N^-1`,
`b2 ~ N^-3/2`, `b3 ~ N^-2` and the tested ratio `|b2/b3|` grows like `sqrt(N)`. The fixed factor
`1/300` therefore means something different on every mesh, always pushing the verdict toward
transcritical. Same root cause as the `unit` eigenvector scaling in
[mpi_augmented_systems.md](mpi_augmented_systems.md) §6.

Measured on `u_t = laplace(u) + lam*u - u^2` (transcritical, since `-u^2` is even) on the 1 x 1.05
rectangle, one mesh doubling:

| | N=8, 225 dofs | N=16, 961 dofs | change |
| --- | --- | --- | --- |
| `b1` | 4.1011e-3 | 1.0254e-3 | x0.250 (predicted `N^-1`, 0.234) |
| `b2` | -7.38681e-4 | -9.23491e-5 | x0.125 (predicted `N^-3/2`, 0.113) |
| `b3` | 5.03433e-7 | 2.81798e-8 | x0.0625 (predicted `N^-2`, 0.0548) |
| **`|b2/b3|`, the old test** | **1467** | **3277** | **x2.23** |
| **`b2_rel`, the new one** | **0.8681** | **0.8655** | **-0.3%** |

The replacement is the same angle trick: `b2_rel = |b2| / (|b2v| |zeta*|)`, the cosine between the
quadratic term `b2v = -H(zeta,zeta)` and the left null vector. It is invariant under `zeta -> s*zeta`
and bounded by 1, so `tol_pitchfork = 1e-3` sits three decades below every non-pitchfork measured -
0.868 for the transcritical, 0.815 for the Bratu fold - the margin `tol_fold` already has.

**The zero it cannot normalise.** On the canonical pitchfork `u_t = laplace(u) + lam*u - u^3` at
`u = 0` the elemental Hessian is `0.0` at every quadrature point, so `b2v` is the **exact** zero vector
and the cosine is 0/0. There is no better denominator: anything carrying the same power of `zeta` is
itself an `H(zeta, .)` contraction and vanishes with it. So `norm(b2v) == 0` is taken as its own
answer - "the quadratic form vanishes identically" is a stronger certificate than a threshold - and
only a nonzero one goes through the cosine.

**A behaviour change, declared.** A *slightly imperfect* pitchfork - symmetry broken by an inexact
boundary condition, or by a non-symmetric mesh in a symmetry-carrying problem - has `b2v` small but
nonzero and its cosine with `zeta*` is O(1). The new test calls that transcritical where the old one
called it a pitchfork. Mathematically that is right (an imperfect pitchfork *is* a transcritical), but
it is a change, which is why `assume="pitchfork"` and `assume="transcritical"` were added in the same
edit alongside the existing `assume="fold"`/`"branch_point"`.

Also **printed but not decided on**: the ratio of the two predicted branch amplitudes at the offset
the switch will take. It carries the parameter's units, and asymptotically close to a bifurcation a
quadratic term always beats a cubic one, so it is not a classifier - "is it a pitchfork" is a symmetry
question, not a size one - but it is what a caller wanting to override the verdict needs to see.

## The singular solve, done as a plain solve

`psi01` and `wst` are solves against `L = -J`, which at a branch point is **exactly** singular - that
is what makes it a branch point. They went through `scipy.sparse.linalg.spsolve` with an `lsqr`
fallback on NaN, and the outer `E()` projection removing the kernel component was the only reason the
answer was ever usable.

What happened was worse than failing. On the transcritical PDE case `spsolve` returned a **finite**
vector with no warning, so the NaN test never tripped, the fallback never ran, and `b3` was whatever
that vector projected to. (The fallback, had it run, works to its default `atol=btol=1e-6`.)

`bordered_la_solve` solves

    [[L, zeta], [zeta_star^T, 0]] [psi; s] = [rhs; 0]

instead. The bordered matrix is nonsingular whenever the singularity is simple - `<zeta,zeta*> != 0`,
which the caller already checks - and gives the `zeta_star`-orthogonal solution directly, at the
conditioning of the *bordered* matrix rather than of `L`. `s` is zero by construction (`rhs` is
`E()`-projected, hence orthogonal to `ker(L^T) = zeta_star`, and so is `L psi`), so a nonzero one says
the null vectors do not belong to this `L` and is raised rather than discarded. Measured on the Bratu
fold: `|L psi01 - E(R01)| = 1.2e-10` relative, `|<psi01, zeta*>| = 2e-18`.

`la_solve` stays, and stays correct, on the **Hopf** path: there `J` is regular - the singularity is in
`J +- i omega M` - so `psi001`, `psi110` and `psi200` are ordinary solves. Its `lsqr` fallback is
tightened to `atol=btol=1e-13` there, because a fire means `J` itself is near-singular (a codim-2
point) and a least-squares answer papers over that.

## The finite difference behind b3

There is no third derivative in the code generator, so `b3` needs a central difference of the analytic
Hessian contraction. The step was `fd_eps = 1e-7` applied as `fd_eps*zeta` with no scaling, and both
halves of that were wrong.

*Magnitude.* The differenced quantity is exact to a relative machine epsilon, so cancellation costs
`eps/h` and the central difference truncates at `h^2`: the optimum is `h ~ eps^(1/3) ~ 6e-6`. Swept on
the Bratu fold at N=8, against the value at the floor:

| `fd_eps` | 1e-3 | 1e-4 | 1e-5 | 3e-6 | 1e-6 | 1e-7 | 1e-8 | 1e-9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rel. error in `b3` | 2e-7 | 2e-9 | **1.6e-13** | **1.6e-13** | 2e-11 | 6.8e-10 | 3.8e-9 | 7.8e-8 |

A clean V with its floor at 1e-5 to 3e-6, which is where the default now sits; the old 1e-7 is three
and a half decades above it.

*Scaling.* `direct_scale = 1` was a dead variable. `directfd` is the unit-normalised eigenvector, so
its entries are of order `N^-1/2` and the actual per-dof perturbation was `fd_eps/sqrt(N)` - `7e-10`
on a 20k-dof problem, at or below the roundoff floor of the dofs themselves, and worse the finer the
mesh. `fd_eps` is now relative: the largest dof moves by `fd_eps*|u|_inf`. That is why `b3` was the
least trustworthy number in the normal form.

**What no branch-point test on a trivial branch can show.** Both PDE cases sit on `u = 0` with a
polynomial nonlinearity, and there the Hessian contraction is exactly *affine* along the perturbation
direction, so the difference is exact at every step - `b3` did not move when the step changed by three
decades (identical to 15 digits). The same problems have `dR/dlam = -int(u v)` vanishing identically at
`u = 0`, so `R01` is exactly zero and the bordered solve is handed a zero right-hand side. Neither
defect above is visible on them. That is why `tests/mpi_branch_switch_worker.py` also carries the
**Bratu fold** - nonzero base state, `exp()` nonlinearity - purely for its coefficients.

## The eigenvector's phase

`zeta` was `numpy.real(eigenvector)`. A real eigenvalue's eigenvector is determined only up to a
scalar, and on a **complex** PETSc/SLEPc build that scalar is complex: SLEPc happily returns
`exp(i phi) v`, whose real part is `cos(phi) v` and for an unlucky phase is a vector of roundoff -
then normalised up into a direction the whole normal form is built out of. Nothing checked it.
`_as_real_eigenvector` rotates by the phase of the largest-magnitude entry (well defined for anything
real up to a phase, where the half-angle of `sum(v_i^2)` is degenerate for an isotropic complex
vector) and raises if what is left is not essentially real. Used at all three places that took a real
part: `zeta`, `zeta_star`, and the GUI's diagram-reload path.

Deliberately not routed through either existing rotation. `Problem.rotate_eigenvectors` needs a
*named* set of dofs to fix the phase by, which a normal form has none of.
`rotate_complex_eigenvector_nicely` (`src/bifurcation.cpp`) solves the harder genuinely-complex
problem but is reachable only through an installed Hopf handler, and falls back to the **unrotated**
vector when its own denominator is small - exactly the hole being closed.

## Under MPI

Branch switching works under a plain `mpirun` and under `--distribute`.

What blocked it was **not** `--distribute`. `NormalFormCalculator` took everything it needed from the
Python custom multi-assembly, and that throws from
`Problem::sparse_assemble_row_or_column_compressed_base_problem` for **any** `nproc > 1`; the
Python-side `_require_non_distributed` guard caught only the distributed half, so a replicated
`mpirun` failed five frames deep in C++ instead.

The fix is the move deflation ([deflation.md](deflation.md)) and `get_hopf_lyapunov_coefficient`
([hopf_normal_form.md](hopf_normal_form.md) §4) both made first: leave that pipeline. Each quantity
now comes from an accessor already MPI-safe in both regimes, and every substitution was A/B'd against
the multi-assembly serially, on both a trivial-branch branch point and the Bratu fold:

| was | is now | agreement |
| --- | --- | --- |
| `a.J()` | `assemble_eigenproblem_matrices(0)`, `_allgather_square`d when distributed | **bit-identical** |
| `a.dRdp(p)` | `Problem.get_parameter_derivative(p)` (gathers in C++, analytic) | **bit-identical** |
| `a.dJdU(zeta) @ x` | `0.25*(H(zeta+x) - H(zeta-x))`, `H = get_second_order_directional_derivative` | 4e-16 .. 2.5e-15 |
| `a.dJdp(p) @ zeta` | central difference of `get_parameter_derivative` along `zeta` | 2.3e-11 |

The **signs** are the part worth pinning, and the A/B pins them: a sign error there is a factor `-1`,
not a small discrepancy, and on a one-dof problem the two conventions can coincide in magnitude.
`d2f` in `bifurcation_tools.py` is the *negative* of the raw Hessian contraction, so a polarisation
written in terms of it needs a minus; `Hraw`/`Hpair` are written against the raw one instead.

`dJdp @ zeta` is differenced in the **dofs**, not the parameter - mixed partials commute, so
`(dJdp @ v)_i = d/dt [dR_i/dp at u + t v]` at `t=0`. Two cheap *vector* assemblies rather than two
whole eigen assemblies plus two gathers, and a decade more accurate as well.

Three things beyond the accessors:

* **The pencil is assembled once** and shared with `get_left_eigenvector`, so `L` and the null vectors
  that border it come from one and the same assembly. Under the old code they came from two, and the
  border's consistency rested on the two agreeing. `L_zeta` and `LT_zeta_star` are reported for
  exactly that reason.
* **The left eigenvector is broadcast from rank 0.** `_left_eigenvector_by_scipy` runs ARPACK
  *redundantly* on every rank with no start vector, so nothing makes the ranks agree on its phase,
  sign or last bits - and `zeta_star` is the border of the bordered solve, the projector in `E()`, and
  the vector every coefficient is a projection onto.
* **`switch_branch` agrees on Newton failures across ranks**, with `get_mpi_any` *outside* the `try`.
  Insurance rather than a known bug - oomph's own convergence tests already allreduce - but a
  rank-dependent linear-solver failure, or an exception out of an `actions_*` hook, would send one
  rank to `continue` while the others carried on, and that hangs at the next collective rather than
  failing. Every other `break` and `continue` in the ladder is driven by rank-identical data.

This removes the last non-tracker user of the Python custom assembler, which finally makes stage 0 of
[mpi_augmented_systems.md](mpi_augmented_systems.md) §10 possible: `Problem._require_single_rank`,
called from `set_custom_assembler`.

**Cost.** One element loop became roughly eleven assemblies. That is the right trade and not a
concession: this runs once per bifurcation, not once per Newton step. The single-element-loop version
needs the parallel multi-assembly of `mpi_augmented_systems.md` §10 stage 2, whose real content is B2
and B3, and branch switching needs none of it.

### Verified

`tests/mpi_branch_switch_worker.py` + `tests/test_mpi_branch_switch.py` (18 tests, `slow`, needs
`--full`), on the two PDE problems above at serial / `mpirun -n 4` / `mpirun -n 4 --distribute`, plus
`-n 3 --distribute` and the Bratu fold. The switch and the four continuation steps after it agree to
every digit across all three:

| | serial | `-n 4` | `-n 4 --distribute` |
| --- | --- | --- | --- |
| transcritical, landed `lam` | 19.9515778393 | 19.9515778393 | 19.9515778393 |
| transcritical, `usqr` | 0.6480430008 | 0.6480430008 | 0.6480430008 |
| pitchfork, `uphi` (both arms) | +/-0.3738231699 | +/-0.3738231699 | +/-0.3738231699 |

and the landings sit on the analytic branches (`u ~ B(lam-lam_c)phi` and `u ~ +-A sqrt(lam-lam_c) phi`)
to 0.5% at the offset actually taken, which is what a leading-order prediction is worth.

## Not done

**Branch switching for symmetry-breaking bifurcations** - azimuthal `m != 0`, Cartesian normal mode
`k != 0`. The normal form is computed on the `m = 0` dof vector throughout: `zeta` from
`get_last_eigenvectors()`, the Hessian contraction from the base dofs, `dR/dp` from the base parameter
derivative. For `m != 0` the bifurcating branch lives in a **different function space** - a
three-dimensional solution off an axisymmetric base state - so there is no dof vector on this problem
to switch onto; the `m != 0` mode would have to be reconstructed into a full problem first. Worth
naming because `AzimuthalSymmetryBreakingHandler` itself works under `--distribute`, so a user will
reasonably expect the switch to follow.

**The Hopf normal form's factor of two** - see the comment at `get_normal_form_hopf`'s `bv`. The old
bare TODO now carries measurements against `get_hopf_lyapunov_coefficient`, which computes the same
coefficient independently in Kuznetsov's invariant normalisation (`ga = Re(b)/omega0`). The ratio is
2.000 where the quadratic terms vanish identically, 0.99984 where they dominate and 2.29 in between,
so it is not one factor and cannot be fixed by one - and the old TODO's guess ("the quadratic terms")
is contradicted by the first case, where they are identically zero and `b` is still exactly half.
What is *not* broken is the prediction: `perturbation_predictor` divides by `2*Re(b)` where the normal
form says `Re(b)`, and on the case whose orbit is known in closed form the two errors cancel exactly
(predicted radius `sqrt(|dp/sigma|)`, which is the exact limit cycle). Changing either alone breaks it.

