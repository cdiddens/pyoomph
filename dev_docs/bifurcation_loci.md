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
