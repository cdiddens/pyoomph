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
| pitchfork | `psign * abs(eps)` | `zeta * sqrt(abs(6*b1/b3 * eps))` |
| fold | `0` | `zeta * eps` |

So `parameter = p0 + param_predictor(eps)`, `dofs = base + perturbation_predictor(eps)`, then `solve()`.
Both signs of `eps` and a few magnitudes are tried, because which side the new branch lies on is not
known in advance (for a pitchfork only one side has one at all). A fold is refused outright: it has one
branch through it, which turns around - getting off *that* is `leave_locus`.

Landing has to be verified, not assumed. The step is accepted only when the displacement is explained by
the new branch's prediction and is not simply along the tangent of the branch we came from; otherwise
Newton has walked back and the next `eps` is tried.

## The second half of the bug

Landing correctly is not enough. With the predict-and-solve above, the switch reached `u = mu` exactly -
and the *first continuation step afterwards* went straight back to `u = 0`.

Just off the bifurcation the Jacobian is still nearly singular, so `dU/dparameter` is large and badly
conditioned, and a step of the size the old branch had reached (`ds = 0.1125` against an offset of
`0.02`) overshoots. `branch_switch` therefore sets `ds` to the offset it just took. The arclength step
grows it again by itself once the branch is well separated.

## Verified

`tests/branch_switch_worker.py`, one process per bifurcation type (a second `Problem` in one process
segfaults in the JIT loader):

- transcritical: lands on and stays on `u = mu` - `(0.020, 0.020), (0.033, 0.033), (0.047, 0.047), ...`,
  max error against the closed form **7.7e-8**
- pitchfork: lands on and stays on `u = sqrt(mu)` - `(0.020, 0.141), (0.040, 0.200), (0.061, 0.247), ...`,
  max error **9.5e-8**

Both check every point after the switch, not just the first, because that is where the old behaviour
came apart.
