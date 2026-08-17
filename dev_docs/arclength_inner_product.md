# The arclength metric: why the default is mesh-dependent, and how to replace it

`Problem.set_arclength_inner_product()` changes the norm in which arclength continuation measures the
*solution* part of a step. This records what is wrong with the default, why a single scalar is enough to
fix it, and what the fix does not fix.

## The problem, measured

oomph's constraint (`src/thirdparty/oomph-lib/include/problem.cc:10047`) is

    ds = (dp/ds)*dp + theta^2 * (dU/ds).dU

with the tangent normalised by `(dp/ds)^2 + theta^2*|dU/ds|^2 = 1`, i.e.
`|dp/ds| = 1/sqrt(1 + theta^2*chi)` and `chi = |dU/dparameter|^2`.

`chi` is a **sum over degrees of freedom**, not an integral, so it has no continuum limit. Measured on
Bratu (`du/dt = laplace(u) + lam*exp(u)`, fold at `lam = 3.513830719`) over three decades of refinement,
`chi/ndof` is constant to three digits (0.01407, 0.01386, 0.01374, 0.01372, 0.01351, 0.01351): **chi
grows in exact proportion to ndof**. Hence with the default `theta^2 = 1`:

| ndof | chi | \|dp/ds\| |
| --- | --- | --- |
| 39 | 0.549 | 0.804 |
| 999 | 13.7 | 0.261 |
| 9 999 | 135 | 0.0857 |
| 49 999 | 676 | 0.0384 |

The same `ds` buys **21x less parameter movement** at 50 000 dofs than at 39. `theta^2 = 1` is therefore
not a neutral default but a choice that depends on the mesh - and on how the fields are scaled, since
`chi` is a sum of raw dof values.

`Scale_arc_length` hides this by tuning `theta^2` until the parameter takes a fixed share `D` of the
arclength. What it converges to is `theta^2` proportional to `1/ndof` (measured: 1.751 at 39 dofs down to
0.00140 at 50 000, a ratio of 1250 against an ndof ratio of 1282) - it rediscovers the normalisation at
runtime, once per step, from a target that has nothing to do with the physics.

## Why a scalar is enough

A weighted norm cannot in general be written as a scalar times a Euclidean one - but only **one
direction** is ever measured. For the current tangent `v`,

    theta^2 = ||v||_W^2 / ||v||_2^2

makes the Euclidean constraint agree with the W-weighted one *exactly*. It is re-derived before every
step, as the tangent turns, which is why this needs no change to the vendored solver.

## theta^2 cannot be changed alone

Setting `theta^2` without touching the tangent leaves the pair violating
`(dp/ds)^2 + theta^2*|dU/ds|^2 = 1`, and then `ds` is no longer a step length. The direction
`z = -(dU/ds)/(dp/ds) = dU/dparameter` does **not** depend on `theta^2`, so the tangent can be rebuilt
exactly:

    dp/ds_new = sign(dp/ds) / sqrt(1 + theta^2_new * |z|^2)
    dU/ds_new = -dp/ds_new * z

set through `_update_dof_vectors_for_continuation` and `_set_arc_length_parameter_derivative`. Verified:
the invariant holds to 3.8e-15 after every retune, on every mesh tried.

`ds` is then rescaled by `|dp/ds|_before / |dp/ds|_after`, for the reason recorded in
`bifurcation_loci.md` and in `BifurcationController._recast_ds_after_metric_change`: along the tangent
`ds = dparameter/(dp/ds)`, so a changed metric buys a different step for the same number. oomph does this
compensation only on the very first step (`problem.cc:11176-11181`, guarded by `!Arc_length_step_taken`).

**The GUI's own recast must therefore stand aside** when an inner product is set, or the correction is
counted twice. That guard is in `_recast_ds_after_metric_change` and is pinned by a test.

## What it buys

Bratu, parameter movement per unit `ds`, fixed `ds`, over 65x refinement:

| ndof | plain dof sum | `"ndof"` | `"l2"` (mass matrix) |
| --- | --- | --- | --- |
| 99 | 0.6389 | 0.992603 | 0.992617 |
| 399 | 0.3872 | 0.992712 | 0.992716 |
| 1 599 | 0.2070 | 0.992765 | 0.992766 |
| 6 399 | 0.1057 | 0.992789 | 0.992790 |

The default drifts by 6x; both normalised metrics are constant to four digits. `"l2"` reaches
`theta^2*ndof` = 0.998, 0.9995, 0.99987, 0.99997 **without being told about ndof at all** - it converges
to the `1/ndof` scaling from the mass matrix alone, which is independent confirmation of what
`Scale_arc_length` was approximating. Fold traversal is unaffected (13 steps at every resolution).

## Two things it does not fix

**`"l2"` needs a mass matrix.** It is assembled from the `partial_t` terms, so a dof appearing in no time
derivative - pressure in an incompressible flow, a Lagrange multiplier - gets **zero weight**, and a step
that only moves those is invisible to the constraint. `"ndof"` weights every dof equally and is the safe
choice there; a callable is the way to weight per field.

**Mesh-independence is not scale-independence.** Making the norm an integral says nothing about how large
"a unit change in `u`" is against "a unit change in the parameter": on Bratu the settled `dlam/ds` is
0.993, i.e. `ds` is almost entirely a parameter step, because this field's mean-square derivative happens
to be small in these units. That trade-off is inherent to arclength in a mixed space and is exactly what
`theta^2` always encoded. Near a fold it self-corrects, since `|dU/dparameter|` diverges and the parameter
share collapses whatever the scale - which is why the fold traversals above are unharmed.

## The tangent across a mesh adaptation

oomph's own path zero-fills `Dof_derivative` when the dof count changes (`problem.cc:10350`,
`problem.cc:10966`), which would throw the tangent away. pyoomph avoids that by stashing the two
continuation vectors in history slots 5 and 6 before adapting and reading them back afterwards
(`Problem._adapt_with_interfacial_errors`), so oomph's projection interpolates them onto the new mesh.
That part works: measured against a freshly computed tangent on the new mesh, the carried one has
**cos = 0.999998**.

What was missing was the renormalisation. Interpolation preserves the direction but not the length, since
`|dU/ds|^2` is a dof sum: refining 39 -> 79 dofs grew it by sqrt(2) (measured 0.635 -> 0.899, against
sqrt(79/39) = 1.42) and left the constraint at 1.40 instead of 1. Measured on Bratu:

| metric | stride, no adapt | stride, first step after adapt | ratio |
| --- | --- | --- | --- |
| plain dof sum, before the fix | 0.239236 | 0.170431 | 0.712 |
| plain dof sum, after the fix | 0.239236 | 0.201302 | 0.841 |
| `"l2"` | 0.252525 | 0.250425 | 0.992 |

Scaling `(dparameter/ds, dU/ds)` by `1/sqrt((dparameter/ds)^2 + theta^2*|dU/ds|^2)` after the read-back
restores the constraint to 3e-16 and preserves the direction, both components being scaled together.
oomph renormalises at the end of every step anyway, so before the fix it healed itself - after one step
with a 29% wrong stride.

The residual 16% under the plain metric is **not** a carry-over defect: the stride ratio equals the
`dparameter/ds` ratio across the adapt (0.8414 against 0.8440), i.e. the metric direction itself changes
with the mesh, because the same physical direction has different (parameter : solution) proportions in a
dof-sum norm at different resolutions. Under `"l2"` that disappears (0.9917 against 0.9917), which is the
same statement as the rest of this note.

One measurement trap worth recording: `get_residuals()` after a continuation step is **not** a
convergence check. The dofs have moved but the time history has not, so a `partial_t` term contributes
when the residual is evaluated outside a steady solve - 8.2e-03 with no adaptation anywhere, against
2.8e-14 after a plain `solve()`. It looked like the step after an adapt was failing to converge.
