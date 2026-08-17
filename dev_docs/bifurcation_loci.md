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

Five things bite, in the order you meet them; §6 then covers eigensolving on a locus in its own
right, since that is where most of the machinery ended up.

### 1. The *critical* eigenvalue on a locus is not solved for - but the rest of the spectrum is

**This section used to say `solve_eigenproblem` refuses outright while tracking. It no longer does**;
see §6 below for what changed and what is still refused.

The critical eigenvalue is still not re-solved. During tracked continuation
`get_last_eigenvalues()` is replaced by the single synthetic value `0 + i*omega`, which is what makes
locus points read as bifurcations to code that tests `eig_value_Re == 0`; re-solving it would turn
the exact zero into a small nonzero number and the point would stop being a bifurcation. Store that
value.

The **rest** of the spectrum is worth having, and is what `BifurcationController.step` now records on
a locus (`_add_locus_state`): a second eigenvalue reaching zero, or a pair crossing the axis, is a
codim-2 point on the curve being followed, and without the spectrum there is no way to see one
coming. Two things about it:

- it needs a **non-zero shift** (`BifurcationController.locus_eigen_shift`, default `0.1`; the
  ordinary `self.shift` defaults to `0`, which is exactly the value that cannot work), and
- it is **non-fatal**: a failed shift-invert factorisation costs that point its spectrum rather than
  aborting a two-parameter sweep that may have been running for hours.

`_sync_tracking_to` no longer takes its re-activation guess from `get_last_eigenvectors()[0]`, since
that is now whatever the secondary eigensolve returned; it asks the handler through
`_get_bifurcation_eigenvector()` instead. **Any other code that reads `get_last_eigenvectors()[0]`
expecting the tracked vector has the same problem** - the handler is the authority while tracking is
active.

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

### 5. A fold is a fold of the Jacobian, not of one parameter

Starting *any* continuation from exactly a bifurcation fails for the same reason as (4), so switching
the continuation parameter while sitting on one fails too. `set_continuation_parameter` warns about
it rather than letting the solver report it five frames deep.

### 6. The base state's eigenproblem, with a tracker installed

This is what §1 above used to refuse. It turned out to be almost pure bookkeeping: oomph-lib's
`Problem::get_eigenproblem_matrices` installs its **own** `EigenProblemHandler` for the duration of
the assembly, and that handler's `ndof`/`eqn_number` delegate straight to the element - so the
*elemental* assembly was already the base one, and the base state still sits in the node data, which
no tracker moves. The only thing that was augmented is the row layout the matrices are built on (and
`Problem::ndof()`, which oomph's own PARANOID block compares them against).

`Problem::BaseDofDistributionScope` (`src/problem.hpp`) restores that layout for the duration of
`assemble_eigenproblem_matrices`, using a reversible form of what the handlers already do on
destruction: an in-place rebuild when replicated, a pointer swap when distributed
(`AugmentedDofDistributionHelper::install_base_distribution` / `restore_augmented_distribution`).
Two details are deliberate:

- **`Dof_pt` is not resized.** The eigen assembly reads element and node data, not the dof vector, so
  leaving it alone keeps every handler dof pointer - and hence the tracked state - valid across the
  eigensolve. Verified: the tracked critical parameter is unchanged to 7e-15 after three eigensolves.
- **The sparse-assembly allocation cache is dropped on both transitions**, since it is a per-row nnz
  table sized to whatever ndof was current when it was filled.

`_get_base_dof_distribution_info()` reports that layout to Python; `_get_dof_distribution_info()`
keeps its old meaning (the augmented one while tracking), which is what
`tests/mpi_bifurcation_worker.py` reads.

**Which modes.** `azimuthal_m` / `normal_mode_k` stay free arguments, independent of the tracked
mode - tracking an `m=1` bifurcation and asking "does the axisymmetric base state fold underneath
this locus?" is one of the more useful checks, and it is an `m=0` eigenproblem. The limit is that
**nothing may be renumbered**: the tracker cached the base equation count and pushed dof pointers
built against the current numbering. The only mode-dependent numbering in the tree is
`AxisymmetryBC._before_eigen_solve`, which releases the strong axis conditions at `m != 0`, so:

| tracking | `m = 0` | `m != 0` |
|---|---|---|
| fold / Hopf / pitchfork | yes | **refused** - the axis pins are on and releasing them renumbers |
| azimuthal / cartesian_normal_mode | yes (axis conditions come back as an `EigenMatrixSetDofsToZero` manipulator) | yes - the pins are already off |

`_before_eigen_solve` does not merely *report* that a renumbering is needed, it has already flipped
the flags by the time it answers, so `actions_before_eigen_solve(must_not_renumber=True)` snapshots
them (`Mesh::get/set_dirichlet_active_flags`) and restores them before refusing. **That restore is
load-bearing, not tidiness**: removing it leaves the axis conditions released while the equation
numbering still has them pinned, and the next eigensolve aborts with SIGABRT rather than returning
anything wrong. `test_refusals` fails with exactly that crash if the restore is taken out, which is
why it asserts on the flags and not only on the exception.

A plain `solve_eigenproblem(n, shift)` with no mode argument assembles at whatever the `azimuthal_m`
/ `normal_mode_k` global parameter currently holds, i.e. **the tracked mode** during azimuthal
tracking. Pass `azimuthal_m=0` for the axisymmetric spectrum. `_solve_normal_mode_eigenproblem` now
restores the mode parameter to what it was on entry rather than hard-coding `0`; the tracker reads
the same parameter when it assembles its own eigen rows, so resetting it to 0 silently retuned the
tracked bifurcation.

**The shift may not be zero, and a zero one is refused.** The tracker has converged the base state
onto the bifurcation, so `lambda = 0` (fold, pitchfork) or `+-i*omega` (Hopf, azimuthal) is an exact
eigenvalue, and `shift=0` - the default of `solve_eigenproblem` - is exactly where shift-invert asks
for a factorisation. SLEPc reports it as a MUMPS zero pivot several frames from the cause.
`shift=None` is refused too, since SLEPc then targets 0 and factorises there anyway.

**Still refused**, and by name: periodic orbit tracking (its augmented dofs are `nT` copies held in
the handler and there is no base distribution kept alive to fall back on - note this used to fall
through the old guard entirely, because `get_bifurcation_tracking_mode()` is `""` for orbits, and
would have assembled on the `nT*Ndof+1` distribution); the Python-side custom augmentation
(`add_augmented_dofs` / `CustomBifurcationTracker`); `set_eigenfunction_as_dofs` while tracking (a
base-length eigenvector would be silently zero-padded into the augmented dof vector, i.e. written
over the tracker's own unknowns); and `refine_eigenfunction`, which adapts.

Covered by `tests/test_eigen_during_tracking.py` (serial, four handlers) and
`tests/test_mpi_bifurcation_tracking.py::test_eigen_during_tracking_*` (12 cases: four handlers x
`np=2 --distribute`, `np=3 --distribute`, `np=2` plain). The assertion that matters in both is the
A/B against the *same state with the tracker removed*: the element contributions are identical either
way, so anything that differs came from the row layout. Under MPI the row blocks are additionally
required to tile `[0, base_ndof)` exactly, which is what says the eigensolver was handed the base
distribution rather than the augmented one.

## How the GUI holds this together

One invariant: **the problem's bifurcation-tracking state always matches the current branch's kind**,
and only `_sync_tracking_to()` and the branch-opening calls (`start_locus`, `leave_locus`) may change
it. `load_pt` calls it on every load, which is what makes clicking between a locus and an ordinary
branch safe - and what the test asserting `get_bifurcation_tracking_mode()` after each load pins.

Everything else follows from that:

- `step()` dispatches on `branch.kind`: a locus step records the critical eigenvalue as the synthetic
  `0 + i*omega` rather than re-solving it, and solves the base state's eigenproblem for the *rest* of
  the spectrum (§1, §6).
- `classify_bifurcations` is suppressed on a locus - every point there has a zero real part, so it
  would run a normal-form calculation per step for an answer already known.
- `_update_tangents` returns early on a locus; the plotted direction comes from `axis_tangent()`,
  which finite-differences the two most recent points in the *current* axes and so needs no solver
  internals. That also made `multistep`'s ds cap work for any pair of axes rather than only for
  parameter-versus-observable.
- `locate_bifurcation` and `branch_switch` refuse while on a locus: the former would activate tracking
  in the continued parameter rather than the tracked one.
- The plotter draws a locus as one brown curve. The stability *segmentation* still has nothing to say
  there: it reads `eig_value_Re`, which is the synthetic zero at every locus point, and would
  alternate the line style from point to point. The eigenvalue pane and `unstable_count()` read
  `eig_values`, which is now the solved spectrum (§1), so those are meaningful on a locus.
- `output_curves` writes each branch in its **own** coordinates, so a locus leads with its parameter
  pair (`b`, `mu_c`) the way the tutorials write `V`, `Bo_c`. Exporting everything in the current view
  labelled the locus with a parameter it does not vary.

## Other guards worth surfacing early

`arclength_continuation` refuses, while tracking is active, `spatial_adapt > 0` and any
`--distribute`d problem (it needs history dofs, which are not distributed). Locating a bifurcation
with `solve()` does work distributed; continuing it does not.
