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

### 1. The *critical* eigenvalue at a tracked point is not solved for - but the rest of the spectrum is

**This section used to say `solve_eigenproblem` refuses outright while tracking. It no longer does**;
see §6 below for what changed and what is still refused.

The critical eigenvalue is still not re-solved. During tracked continuation
`get_last_eigenvalues()` is replaced by the single synthetic value `0 + i*omega`, which is what makes
locus points read as bifurcations to code that tests `eig_value_Re == 0`; re-solving it would turn
the exact zero into a small nonzero number and the point would stop being a bifurcation. Store that
value.

The **rest** of the spectrum is worth having, and `BifurcationController._tracked_spectrum` records
it at **every** point a tracker has converged onto - each step of a locus (`_add_locus_state`) and
each freshly located bifurcation (`locate_bifurcation`) alike. On a locus a second eigenvalue
reaching zero is a codim-2 point on the curve being followed; at a located bifurcation the rest of
the spectrum is what says whether the branch was already unstable there and which mode goes next.
Four things about it:

- it needs a **non-zero shift** (`BifurcationController.tracked_eigen_shift`, default `0.1`, formerly
  `locus_eigen_shift` and still readable under that name; the ordinary `self.shift` defaults to `0`,
  which is exactly the value that cannot work),
- it is **non-fatal**: a failed shift-invert factorisation costs that point its spectrum rather than
  aborting a two-parameter sweep that may have been running for hours,
- the synthetic critical value **replaces** the solved copy of the same eigenvalue rather than being
  listed beside it. Both would show one mode twice, and the solved copy's real part is a
  rounding-error-sized number of arbitrary sign, so a positive one counts the located bifurcation as
  unstable. Which entry it is is decided by **eigenvector overlap**, not by `|lambda - crit|`: at a
  codim-2 point two eigenvalues sit on the axis together, which is precisely the case the spectrum is
  being solved for. A Hopf's partner at `conj(crit)` is a different entry and stays in the list, but
  is snapped onto the axis for the same sign reason. `BifurcationGUISolutionPoint.tracked_eigenindex`
  records which entry the tracked one is; the Points tab marks it with a `*`, and
- `tracked_eigenindex` is set **only when that eigensolve actually ran**, which is also the only way
  to tell a solved spectrum from the fallback on a problem with one dof.

Nothing may re-solve the spectrum at a located bifurcation afterwards: the exact zero is what makes
it a bifurcation everywhere in the diagram and no eigensolve taken from its dump can reproduce it
(no tracker is installed any more, and the state *is* the singularity). `compute_spectrum` refuses
outright and `_store_spectrum` - the stripe scan's way in, which is worth having there - carries the
tracked value across the merge.

`_sync_tracking_to` does not take its re-activation guess from `get_last_eigenvectors()[0]`: although
`_tracked_spectrum` puts the tracked eigenpair back when it is done, anything that eigensolves in
between would leave that pointing at a different mode. It asks the handler through
`_get_bifurcation_eigenvector()` instead. **Any other code that reads `get_last_eigenvectors()[0]`
expecting the tracked vector has the same problem** - the handler is the authority while tracking is
active. The put-back matters for what runs *at* the point: `NormalFormCalculator.get_normal_form`
reads `get_last_eigenvalues()[0]`/`get_last_eigenvectors()[0]`, so a located bifurcation would
otherwise be classified from whichever mode the secondary solve happened to lead with.

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

## A real eigenvalue has one eigenfunction, not two

The field-plot menu offers the eigenfunction as a real and an imaginary part. For a REAL eigenvalue
those are the same function: the eigenvector is fixed only up to a global phase, and a complex PETSc
build has no reason to return that phase as zero, so what comes back is `exp(i*phi)*w` with `w` real
and the two parts are `cos(phi)*w` and `sin(phi)*w`. Since an eigenplot autoscales - it has no scale
in common with the solution - the two panes render as the same picture twice, which is how this was
noticed.

`eigenvector_is_essentially_real()` decides it on the eigenVECTOR, not the eigenvalue. "Is
`Im(lambda)` small" needs a scale to be small against, and near a fold both parts of `lambda` go to
zero together; "are `Re v` and `Im v` parallel" needs none, and it is the question the plot actually
asks - whether there are two functions to draw or one. The imaginary entry is then shown **disabled
and labelled** rather than removed: a menu entry that silently disappears between two refreshes is
harder to understand than one that says why.

## The stripe scan needs a complex PETSc

Scanning a rectangle of the complex plane for every eigenvalue in it is SLEPc's contour-integral
solver (CISS), which integrates along the rectangle's boundary. There is no such thing on a
real-scalar build, and SLEPc says so with `PETSC_ERR_SUP`, i.e. `PETSc.Error(56)`, "no support for
requested operation" - a number that tells the user nothing, and that a stale `eps_type` in the
options database produces as well (see `_apply_eigenvalue_region`), so it cannot even be looked up
unambiguously after the fact. `_require_complex_petsc_for_region()` therefore refuses before the
solve, where the build can be named and the fix stated. Reproduced both ways: the same scan is
`STRIPE OK` against `$PETSC_ARCH_COMPLEX` and `PETSc.Error(56)` against the real arch.

## The Deflation tab

Arclength continuation follows the branch it is on. Deflation ([deflation.md](deflation.md)) is how the
GUI reaches a branch that is not connected to it: it multiplies the residual by a factor that blows up
at every solution already known, so Newton has to converge somewhere else or not at all.

Two commands, both in the Bifurcation menu and on the tab:

- **Deflated solve** (`BifurcationController.deflated_solve`) stands still in parameter space and looks
  for another solution *here*. A solution that is genuinely new opens a **new branch**; the parameter
  is not moved. The set of solutions being avoided is kept between clicks and reset when the parameter
  moves (`_refresh_deflation_known`), so pressing it repeatedly walks through the solutions at one
  parameter value instead of finding the same one again - "Forget known solutions" restarts that.
  Two things it must not skip: taking the operator **off** before the eigensolve and before the point
  is recorded (everything downstream has to see the ordinary residual), and putting the dofs back when
  nothing was found, since every failed attempt left a perturbed state behind.
- **Deflated continuation** (`deflated_continuation`) drives `Problem.deflated_continuation` over a
  parameter scan and maps each `branch_index` it reports onto a new GUI branch. A scan and an arclength
  branch are different objects - a scan steps the parameter and cannot turn a fold - so *every* index
  becomes a new branch, including the continuation of the one we started from, rather than being
  appended to a branch whose ordering it does not respect. Abortable between parameter steps.

The default step is **ds** and the default count is **10**, which is not a coincidence: a fresh
diagram opens on `initial_view_ds_ahead = 10` ds ahead of the first point, so the default scan is
exactly the visible window and ends on its right edge. Tying the increment to the parameter's own
magnitude instead (`0.05*|p|`, as it first was) has nothing to do with either, and on a fresh diagram
a single step of it could already leave the plot - which the tab then reported as a scan cut short
after one step.

A scan stops where the diagram does, in the same spirit as `multistep`, and it takes **two** tests to
mean that, not one:

- `deflated_scan_values()` truncates the value list where the parameter leaves the visible range of
  whichever axis shows it (`scanned_parameter_axis_limits()` - either axis can carry either quantity,
  and the parameter need not be drawn at all). Three conditions took a few tries to get right, and
  each wrong version showed up in the tab as a nonsense step count:
  - the first value **outside** the range is kept, so the scan steps and then notices it has left,
    exactly as multistep does. That also guarantees at least one step;
  - the view has to actually **bound** this scan for the clip to apply: the starting value inside it
    *and* at least one whole step fitting inside it. Both halves matter. A diagram with no points has
    never had its limits set, so the view is still matplotlib's default `(0, 1)` and says nothing
    about a parameter of 20 (that read *"in 0 steps"*); and after ten continuation steps the point
    sits exactly **on** the right edge, because the plotter grew the limits to include it, so there is
    no room ahead and clipping there reported a ten-step scan as a one-step one;
  - a non-finite limit is not a bound. Matplotlib hands one out if a non-finite coordinate ever
    reached `extend_lims`, and comparing against it answers False to everything, which came out of the
    tab as a range of *"nan ... nan"*.

  The view is not a wall in any case - it grows as the scan draws into it - and the loop's own box
  test below is what stops a scan that has genuinely left the diagram. The tab says when the range is
  what shortened the scan, rather than the step count, and it recomputes that line on every canvas
  repaint (`_on_canvas_drawn`), since a navigation-toolbar zoom changes the answer without going
  through `refresh()`.
- The loop also gives up once a whole parameter step has produced nothing inside the axes. A range
  check on the parameter alone cannot see that: on `du/dt = mu - u^2` with the view at `|u| < 1.2`,
  both arms `u = +-sqrt(mu)` are off the plot from `mu > 1.44` while `mu` itself is still deep inside
  its own range. That is exactly what `multistep`'s box test catches. It is judged per parameter step
  rather than per point, because one branch running off the top must not end a scan the others are
  still on.

**Stability is not inferred along a scanned branch.** `propagate_stability()` carries an unstable
count along a branch on the argument that `sign(det J)` flips exactly when an odd number of real
eigenvalues crosses zero - which says nothing about a point that records neither a determinant sign nor
a `dparam_ds`, and a deflated scan records neither. It steps the parameter with no arclength control
and no test function, in precisely the regions where branches appear and disappear, so carrying a
neighbour's count across would not be an inference but an assumption drawn in the same colour as a
measurement. Propagation therefore stops at any point with no test function at all, and everything
beyond it stays unknown from that side. Quick mode is untouched: its points always carry one of the
two. The other beneficiary is the point recorded when an eigensolve returned nothing, which was in the
same position and was being coloured just as confidently.

"Solve eigenproblems during the scan" is **on** by default, like an ordinary continuation step: a
branch drawn without stability is half a bifurcation diagram, and finding a new branch without knowing
whether it is stable rarely answers the question that led to it. Turned off - worth it where the
eigensolves dominate - the points are recorded with `_add_current_state(measured=False)`: no eigensolve
was done, so the point must say it has **no** spectrum rather than carry whatever the problem still
held from an earlier one. Its stability then reads as unknown, exactly like a quick-mode point, and
"Compute the eigenvalues along this branch" fills it in once it is clear which of the branches found
are worth the eigensolves.

Afterwards the scan does two things that are easy to forget and both break the *next* command rather
than itself. It resets the arclength state - the parameter was moved by hand, so oomph's tangent
describes a step that was never taken and an ordinary Step has to start a fresh one. And it **reloads
the last recorded point**: a scan ends on whatever its last attempt left behind, and its last attempt
is by construction a failed one, since the hunt for new solutions is what stops when a deflated solve
stops converging. Without the reload the problem sits on a diverged state that is on no branch, which
is exactly the invariant `current_point` is supposed to carry - and a second Deflated continuation
began its opening solve from there and went straight to inf/nan residuals.

## Periodic orbits

A Hopf bifurcation sheds a periodic orbit, and until now the GUI could only leave one transiently and
watch where the solution ended up. `switch_to_orbit()` steps onto the orbit itself and opens a branch
of kind `"orbit"`, continued in the parameter like any other. `branch_switch()` dispatches to it at a
Hopf, so the one key that means "leave this bifurcation sideways" does the right thing at all three:
a second steady branch at a transcritical or a pitchfork, nothing at a fold, the orbit at a Hopf.

### The analytic Hessian is a prerequisite, not a nicety

`PeriodicOrbitHandler::get_jacobian_*_mode()` throws *"Cannot track periodic orbits without having
analytical Hessian"* (`src/bifurcation.cpp`, four sites). Not just the first Lyapunov coefficient -
the handler's own Jacobian. So there is **no** Hessian-free route to an orbit, and in particular
neither of the two that look like one: the explicit-amplitude form of `switch_to_hopf_orbit` (which
only skips the Lyapunov calculation) and building the guess from the Hopf normal form's
`perturbation_predictor` (which still installs the same handler).

The throw comes out of the first Newton solve, by which point the augmented system is installed and
every later command is working on the wrong problem. `orbit_can_be_started()` therefore checks
`are_hessian_products_calculated_analytically()` **before** anything is installed, and the Orbit tab
shows the answer without waiting to be asked - because the remedy,
`setup_for_stability_analysis(analytic_hessian=True)`, has to be applied before the problem is
initialised and so cannot be tried again in the same session.

`switch_to_hopf_orbit` also does not work under `--distribute` (the orbit itself and its multipliers
do; `dev_docs/floquet_multipliers.md` §10.1), which is refused in the same place.

### The step off the Hopf is the one ds buys

An orbit emerging from a Hopf sits at a parameter offset of `eps**2`, and `switch_to_hopf_orbit` takes
`eps = sqrt(dparam)`. Passing `branch_switch_parameter_offset()` - what one continuation step buys in
the parameter - as `dparam` therefore puts the first orbit exactly one ds away, and the `+`/`-` keys
that scale a sweep scale the step onto the orbit too. Which SIDE of the Hopf it lands on is not a
choice: the orbits exist on one side only, and the first Lyapunov coefficient says which.

`get_init_ds()` gives the signed step to continue with, but it clamps to `5e-10` when the parameter
barely moved, which would leave a sweep that never visibly moves; the sign is kept and the magnitude
taken from the offset in that case. And it is only meaningful on the object `switch_to_hopf_orbit`
returns - the one `activate_periodic_orbit_handler` builds has `emerging_info` full of `None`.

### An orbit point is a whole cycle, recorded as three numbers per observable

The average over the period goes under **the observable's own name**, and the extremes under
`"<name>  [orbit min]"` / `"[orbit max]"`. That way round on purpose: `branch_can_be_plotted` probes a
branch's first point for the current axis, so an average stored under a new name would make the orbit
branch invisible on the very axis its own Hopf sits on. As it is, every axis, tangent, export and
selection path works unchanged and the orbit branch continues the stationary line straight through
the bifurcation, with the band opening out from it.

The average is the exact Gauss-Legendre weighted time integral for mesh integral observables and the
sampled mean for the rest - `evaluate_observable_time_integral` resolves the name before the last `/`
with `get_mesh()`, and an ODE domain is an `ODEStorageMesh`, which is not a mesh. Two traps in that
loop, both of which produce plausible numbers rather than errors:

* the samples are taken with `endpoint=False`, because `s=0` and `s=1` are the same state and
  including both counts one sample twice, biasing the mean by `1/N`;
* the exact integral runs **after** the sampling loop, never inside it: both back the dofs up on the
  handler, and the nested second one throws "the dofs have already been backed up".

The period is an ordinary observable too, `"orbit/T"`, rather than a third kind of axis - everything
that already carries observables then carries it. A stationary point does not have it, so putting it
on an axis drops the steady branches out of the plot, which is exactly what a period-vs-parameter
diagram should show.

### Stability: exponents drive the diagram, multipliers name the bifurcation

`get_floquet_multipliers()` returns the multipliers; what is stored in `eig_values` is
`log(mu)/T`, the Floquet **exponents**, put through `_phys_eig` like any other eigenvalue. The whole
stability machinery is written against a real part, and `Re(log(mu)/T) > 0` is exactly `|mu| > 1`, so
the segmentation, the line styles and the propagation work with no changes at all. The multipliers are
kept alongside in `point.floquet` because the exponent cannot say WHICH bifurcation is approaching: a
real multiplier leaving through `-1` (a period doubling) and a complex pair leaving anywhere else (a
torus) have the same exponent real part.

Three things about the trivial multiplier, which every orbit has by time-translation invariance:

1. It must be removed, but **not** because it would look like a bifurcation. It comes back as
   `1 +- 1e-15`, so its exponent is a tiny number of *either sign* and `eig_value_Re == 0` is False.
   What it does is corrupt the *count*: `measured_unstable_count` counts positive real parts, so the
   branch would flip between stable and unstable at random from one point to the next.
2. Which is also why `unstable_count` is set explicitly from `|mu| > 1 + tol` rather than left to that
   count - the multiplier is the quantity with the clean threshold, and `stability_indicator` prefers
   `unstable_count` whenever it is set.
3. **`ignore_periodic_unity` is not used to remove it.** That argument is a *tolerance*, and it
   removes every multiplier within it of 1 - including the one that matters. Near a Hopf the orbit's
   own multiplier tends to 1 as well: measured on the subcritical Lorenz orbit `1e-4` off its Hopf,
   the trivial multiplier came out at `1+2.1e-9` and the physical one at `1+3.9e-6`, and the default
   tolerance of `1e-5` deleted both - i.e. exactly the number that answers whether the orbit is
   unstable, on the branch where the answer is least obvious. An orbit has *exactly one* trivial
   multiplier, so the GUI asks for all of them and removes the single one nearest 1.

How far that one actually came out from 1 is worth reading: it is the accuracy of the discretization
at this point and nothing else, and it bounds what can be believed about every other multiplier. It
is logged when it exceeds `floquet_unity_tol`. (With an EVEN number of time intervals a DAE's
algebraic directions sit next to `+1` too, which is one more reason not to remove by tolerance.)

And the parity itself: with an ODD number of intervals those algebraic directions land on exactly
`-1`, which is where a period doubling would be (`dev_docs/floquet_multipliers.md`). `_orbit_NT()`
raises the requested count to a multiple of the collocation order **and** to an even number, and says
so in the log. Note `get_num_time_steps()` is `NT+1` in collocation and floquet mode - the handler
appends the end-of-period block - so the interval count is `nT-1`.

`get_floquet_multipliers()` **overwrites** `problem._last_eigenvalues` and `_last_eigenvectors` with
the multipliers. Everything downstream that reads them - `_add_current_state`'s own branch,
`critical_eigenindex`, `_sync_tracking_to`, the eigenfunction panes - would take them for eigenvalues,
so they are snapshotted and put back (`floquet_feeds_eigen_panes` opts out, since on an orbit the
Floquet vectors are the interesting field to plot).

### A collapsed orbit is a converged solution, and looks like one

Continuation can walk an orbit branch back onto the stationary branch it came from - Newton has no
reason not to, an unstable orbit least of all - and what comes back is a perfectly converged solution
that is recorded as an orbit, drawn as an orbit, and is not one. `switch_to_hopf_orbit`'s own collapse
check guards only the FIRST orbit, so `_orbit_has_collapsed` guards every step: every band has zero
width relative to its own observable's scale. Measured on the Lorenz orbit, a genuine one gives `7e-3`
and a collapsed one `1e-11`, so the threshold has three orders of magnitude of daylight on each side.

Not measured on the multipliers, tempting though it is: a collapsed orbit does lose its multiplier at
1, but so does a perfectly good one whose trivial multiplier the discretization has not resolved, and
the two cannot be told apart that way.

The point is recorded and then the step raises - it IS a converged solution, so discarding it is worse
than leaving one that can be looked at and deleted, while continuing from it silently is the one thing
that must not happen. Raising rather than setting the abort flag, which stops the current sweep and
then quietly makes the next Step do nothing.

This is the failure mode the ds-derived step off the Hopf can walk into. The offset is a NORMAL-FORM
step, valid while it is small; on Lorenz a perfectly ordinary `ds = 0.05` buys `0.49` in `rho`, which
lands on a genuine orbit of amplitude 3.9 far from the asymptotic branch, and the step after that
collapses. A smaller parameter step in the Orbit tab is the cure, and the same run with `1e-4` follows
the branch exactly as the `hopf_switch` tutorial does by hand.

### Saving one: the dump is one phase of the cycle

A state dump holds the mesh, i.e. the orbit at `s=0`. The rest of the cycle and the period live in the
handler and go into a companion beside the dump, `state_%06d.orbit.npz`: the `(nT, nbase)` time blocks
in naive time-major order, `T`, and the augmented arclength tangent (which is deliberately kept out of
the dump itself - it has the augmented length there and would be dropped with a note on every reload).
Restoring is `load_state` then `activate_periodic_orbit_handler`, followed by writing the whole
augmented vector back, which is what makes the round trip exact rather than approximately right.

Read the blocks out of the dof vector, **not** by re-sampling the orbit.
`PeriodicOrbit.change_sampling()` is the obvious thing to copy and it is wrong three ways: it samples
with `endpoint=True` so `u_0` is duplicated as `u_1` and the tail of the cycle dropped, `nT` grows by
one on every call in floquet mode, and its `do_solve` is ignored.

A raw dof vector only means anything under one equation numbering, unlike a dump, which is written per
node. So the numbering is fingerprinted from `get_dof_description()` and a mismatch - a mesh
adaptation, a different number of processes - is **refused**: the blocks would otherwise load happily
as a different, entirely plausible orbit. The fingerprint has to be taken while the PLAIN system is
installed; asked under a handler, `get_dof_description()` is sized by the augmented `ndof()` while its
walk fills only the base entries, and it says so with "UNASSIGNED DOF IN DOFLIST".

`orbit_portable` stores `nT` full state dumps instead, one per time point. Partition- and
mesh-independent, at `nT` times the disk; forced under `--distribute`. Both formats are removed by
`_remove_statefile` (one function, three callers) and copied alongside a tagged point's dump.

### What an orbit branch refuses, and why

`get_bifurcation_tracking_mode()` answers `""` under the orbit handler - `start_orbit_tracking` only
swaps the assembly handler - so every guard written against that string alone is blind to an orbit.
`_augmented_system_active()` is the one to use. Two places where the blind version was actively
harmful: `_sync_tracking_to` would leave the handler installed when moving to a stationary point, and
`load_pt` would read a dump into the augmented dof vector.

The same "it is augmented without saying so" is why `_retune_arclength_theta` needed a guard of its
own: `_get_n_unaugmented_dofs()` is 0 under the orbit handler, so with the controller's default `"l2"`
inner product it would assemble the mass matrix on the `nT*Ndof+1` distribution - the very thing
`solve_eigenproblem` refuses there - on the very first orbit step.

Refused on an orbit branch, each with the reason: locating a bifurcation (a bifurcation of an orbit is
a multiplier leaving the unit circle, which no tracker here locates), starting a locus, a transient
departure, a deflated solve or scan, a stripe scan, `new_branch_from_state`, moving or changing a
parameter, and adaptation after a step (it renumbers, which pulls the augmented vector out from under
the handler and invalidates every set of blocks already stored on the branch). `compute_spectrum` does
not refuse - it computes the multipliers instead, which is the same question asked of an orbit.

## Deleting a branch

`delete_branch()` is the one diagram command that reloading cannot undo - it removes the state dumps as
well - so the GUI asks first, names the branch and its point count, and defaults the dialog to Cancel.
Two things it has to get right, both pinned by tests: the **last** branch cannot be deleted (the
problem would be sitting on a solution the diagram does not know about, and every command that reads
`current_point` would work from a stale one), and deleting the branch the problem is **on** has to
*reload* a neighbour's last point rather than merely re-point `current_branch` at it, or the next step
continues from a solution the diagram does not show.

## Switching tracking off belongs where it was switched on

`locate_bifurcation()` activates the tracker, solves the augmented system and records the point. The
solve is the part that can diverge, and the teardown used to sit in one *caller*
(`locate_bifurcation_or_switch`) rather than beside the activation - so a diverged tracking solve left
the augmented system installed, and two of the three entry points (Locate pitchfork, Locate the
bifurcation of the selected eigenvalue) call `locate_bifurcation` directly and had no cleanup at all.

The symptom was thoroughly misleading. Everything afterwards silently solved the augmented problem,
and the next attempt to locate a bifurcation failed in its opening `solve_eigenproblem` with *"a
non-zero shift is required"* - the leftover tracker's own guard (SS6), correct in itself and saying
nothing whatever about the solve that had actually failed one click earlier. The activation and the
`deactivate` are now a try/finally, and the test stubs the solve to raise rather than driving it to
divergence, so it pins the invariant instead of one route to it.

## Other guards worth surfacing early

`arclength_continuation` refuses, while tracking is active, `spatial_adapt > 0` and any
`--distribute`d problem (it needs history dofs, which are not distributed). Locating a bifurcation
with `solve()` does work distributed; continuing it does not.

## Closing the window: a Tk window is never freed unless it is destroyed

Two of Python's reference-cycle blind spots meet in this GUI, and between them they used to keep a
whole `Problem` alive to the end of the process - which nanobind then reports as
`nanobind: leaked N instances!` while the interpreter shuts down.

- The Tcl interpreter holds the callbacks a window registers (menu entries, bindings, the
  `WM_DELETE_WINDOW` protocol handler). The cyclic collector cannot see into it, so a window that is
  never destroyed is never collected either, and neither is anything it references.
- A C++ object's Python references, held through nanobind, are equally invisible to the collector, so
  an object graph that joins a window to a `Problem` is beyond it for good.

Measured on the stub session `tests/gui_close_teardown_worker.py --phase raises` uses: a window built
and left standing costs **61 leaked instances, 19 types and 737 functions**, with no problem in the
process at all - the module-level `Expression`s alone. Destroy it and the count is zero.

`BifurcationTkApp.teardown()` is therefore the one place that ends a session, and every route into it
is guarded: `_on_close` (the close button and the Quit command), `run()`'s `finally` (including the
set-up before `mainloop`, which reads the controller and can raise on a session that has not started),
`BifurcationGUI.start()`'s `finally`, and the constructor's own `except`, which destroys a half-built
window before re-raising. It destroys the root, closes the panes' figures, and drops the window's
references to the controller, the plotter and the panes - so whatever of the window does survive holds
nothing of pyoomph's. `teardown()` is idempotent; the methods that a sweep still calls on the way out
(`log`, `refresh`, `pump`, `_on_status`, `_on_busy`, `_report_error`, `_invoke`'s tail) return early
once it has run, and a sweep that was running when the window closed is sent an abort request.

Two things that are *not* leaks and should not be chased:

- The **facade** keeps the controller, the branches and the diagram plotter after `start()` returns -
  a script is meant to read the diagram afterwards. The figure gets a plain Agg canvas back on
  teardown, so `update_plot()` and `savefig()` still work with no window around.
- The window **object** may survive the teardown, because of the Tcl half above. That is a few
  kilobytes of dead widgets; what matters is that it no longer reaches the problem, which is what
  `tests/test_bifurcation_gui.py::test_closing_the_window_lets_go_of_the_problem` asserts.

The panes are the other half of the story: they are pyplot-managed figures (see `panes.py` on why they
have to be), and matplotlib keeps those in a process-wide registry until they are closed. Dropping a
pane without `plt.close()` leaves the plotter, the mesh data cache, the meshes and the problem
reachable from a module global - the same leak by a different route, which is why the test walks
`Gcf.figs` as well.
