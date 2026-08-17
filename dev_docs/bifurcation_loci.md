# Two-parameter continuation of bifurcations, and the slice a diagram lives in

Notes behind the parameter/locus work in `pyoomph/utils/bifurcation_gui/`. Everything here was
established by probing a two-parameter ODE, `du/dt = mu - u^2 - b*u`, whose stationary states are
`mu = u^2 + b*u` and whose fold sits at `u_c = -b/2`, `mu_c(b) = -b^2/4` - a parabola, so every
number below can be checked against theory rather than against the code.

## A diagram is a slice, and state files move you between slices

`save_state`/`load_state` round-trip **every** global parameter, not just the continued one. Saving
at `a=1, b=1`, moving to `a=4, b=9` and loading the dump back puts you at `a=1, b=1`.

Two consequences for anything that stores solution points:

- A diagram continued in one parameter is only valid at fixed values of the others, and those values
  must be recorded with it. They live in `BifurcationGUISolutionPoint.param_values` and are derived
  per branch by `BifurcationGUISolutionBranch.fixed_parameters()`.
- Loading a point silently moves every other parameter to that point's values. The GUI has to say so
  rather than let the user believe the parameter they just typed is still in force.

Parameters other than the continued one stay **bit-identical** through an arclength step (checked
over 8 steps), so exact comparison would do for slice identity; `SLICE_RTOL` exists only to absorb
values the user typed or reached through `go_to_param`.

## Continuing a bifurcation in a second parameter

`activate_bifurcation_tracking(A, type)` followed by `arclength_continuation(B, ds)` traces the locus
of the bifurcation in the (A,B) plane. This is supported and documented - the docstring of
`activate_bifurcation_tracking` says `arclength_continuation` will track the bifurcation, and
`arclength_continuation` refuses only `B == A`. Two tutorials do it by hand:
`docs/source/tutorial/advstab/movmesh/hanging_droplet.py` writes `Bo_c(V)` and
`docs/source/tutorial/pde/patterns/kuramoto_sivanshinsky_bifurcation.py` writes `gamma_c(delta)`.

It is accurate: six steps along the locus land on the analytic parabola to `2.6e-11`.

Four things bite, in the order you meet them.

### 1. `solve_eigenproblem` raises while tracking is active

`problem.py` refuses it outright ("Cannot calculate eigenvalues/vectors when bifurcation tracking is
active"). A stepping routine that unconditionally solves the eigenproblem after each continuation
step - as `BifurcationController.step` did - dies on the first locus step. There is nothing to solve
anyway: during tracked continuation `get_last_eigenvalues()` is replaced by the single synthetic
value `0 + i*omega`, which is what makes locus points read as bifurcations to code that tests
`eig_value_Re == 0`. Store that value and skip the eigensolve.

### 2. A locus point's dump cannot be loaded into a plain problem

While tracking is active the dof vector is the augmented one (base dofs + eigenvector + parameter),
and `continuation_data_in_states` writes an arclength direction vector of that size. Loading it after
`deactivate_bifurcation_tracking()` fails with

    Mismatching size in the dof direction vector and the actual number of DoFs: 3 vs 1

Load locus points with `ignore_continuation_data=True`. The base solution and all parameters restore
correctly; only the arclength direction is dropped, which `reset_arc_length_parameters()` replaces.

Note this bites *only* points made by locus continuation. A bifurcation located by a plain `solve()`
has no augmented direction vector yet, which is why the pre-existing `locate_bifurcation` path -
which also saves its state while tracking is active - never hit it.

Resuming works: load with `ignore_continuation_data=True`, then
`activate_bifurcation_tracking(A, type, eigenvector=get_last_eigenvectors()[0])` - the dump restores
the critical eigenvector, which is exactly what both tutorials pass as `eigenvector=`. Four further
steps stay on the analytic parabola to `6.9e-12`.

### 3. Changing the continuation parameter wipes a seeded arclength state

`arclength_continuation` calls `reset_arc_length_parameters()` whenever the parameter differs from
the one it was last called with. Anything that seeds `_update_dof_vectors_for_continuation` /
`_update_param_info_for_continuation` before switching parameter has its seed thrown away. Claiming
`_last_arclength_parameter` first avoids the reset - which is what `branch_switch` gets away with
only because it continues in the *same* parameter.

### 4. You cannot arclength-step off a fold

Leaving the locus means resuming an ordinary branch, and that branch passes through the fold, where
the plain Jacobian is singular and there is no regular tangent to compute. Two approaches fail:

- A plain `arclength_continuation(A, ds)` from the fold point: `OomphException`.
- Seeding the fold's own normal-form direction (`param_predictor = 0`, `perturbation_predictor =
  zeta*dp`, i.e. `dparam/ds = 0` along the null vector - what `branch_switch` does at a
  transcritical or pitchfork point): the arclength normalization produces `Ds = -nan` and oomph-lib
  then retries **for ever**, printing `STEP REJECTED --- TRYING AGAIN with Ds=-nan`. It filled a
  200 MB log in six minutes. Do not do this without an iteration cap.

What works is to step off the fold with a regular Newton solve instead of a continuation step:

1. `deactivate_bifurcation_tracking()`, then `reset_arc_length_parameters()`.
2. Capture the dofs **after** deactivating - a copy taken while tracking is active is the augmented
   vector and `set_current_dofs` rejects it ("Mismatch in dof vector size").
3. Offset the parameter off the fold and guess the dofs along the null vector:
   `A = A_c + sign*delta`, `dofs = fold_dofs + hsign*sqrt(delta)*zeta`, then `solve()`.
4. Try the combinations of `sign`, `delta` and `hsign` until one converges to a solution distinct
   from the fold. Only one side of the fold has solutions, and which one is not known a priori.

On the probe this converges on the first combination, lands on the branch to `1e-9`, reproduces the
analytic eigenvalue `-(2u+b)` to six digits, and ordinary continuation then proceeds normally with
the second parameter untouched.

## Other guards worth surfacing early

`arclength_continuation` refuses, while tracking is active, `spatial_adapt > 0` and any
`--distribute`d problem (it needs history dofs, which are not distributed). Locating a bifurcation
with `solve()` does work distributed; continuing it does not.
