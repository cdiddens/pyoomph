# Recovering from a failed re-solve after spatial adaptation

Status: **implemented.** Off by default - a problem without an
`AdaptiveResolveRecovery` behaves exactly as before, and that is asserted by
`tests/test_adaptive_resolve_recovery.py::test_without_a_policy_the_failure_is_still_fatal`.

The symptom: on badly conditioned problems, `solve(spatial_adapt=n)` / `run(..., spatial_adapt=n)`
dies during the Newton solve that follows an adaptation. The run ends there. It cannot be caught,
and even if it could, the state left behind is unusable - the pre-adapt mesh no longer exists and
the dofs on the new mesh are whatever the diverging Newton iteration last wrote.

To switch the recovery on:

```python
from pyoomph.generic.adaptive_recovery import AdaptiveResolveRecovery
problem.adaptive_resolve_recovery = AdaptiveResolveRecovery()
```

---

## 1. Why it is fatal without this

pyoomph does not own the adaptation loop. `Problem.solve` hands `spatial_adapt` straight to
oomph-lib as a `max_adapt` argument (`pyoomph/generic/problem.py:6216`, `:6265`, `:6267`) and the
loop runs entirely in C++:

| entry point | loop | `adapt()` at | re-solve at |
| --- | --- | --- | --- |
| `steady_newton_solve(max_adapt)` (`problem.cc:9603`) | `newton_solve(max_adapt)` (`:16446`) | `:16462` | `:16532` |
| `unsteady_newton_solve(dt,max_adapt,first,shift)` (`:16283`) | itself | `:16334` | `:16409` |
| `doubly_adaptive_unsteady_newton_solve_helper` (`:11686`) | itself, then delegates | `:11718` | `:11794` |
| `arc_length_step_solve_helper` (`:10774`) | itself | `:10827` | `:10960` |

(All paths are in `src/thirdparty/oomph-lib/include/problem.cc`. `adapt()` is pyoomph's override,
`src/problem.hpp:756`, which forwards to the Python-overridable `_adapt()`.)

Three independent things have to go wrong, and all three do:

1. **`adapt()` has no undo.** It mutates the trees in place and rebuilds the global mesh. The
   converged pre-adapt solution ceases to exist at that line.

2. **A failed Newton leaves the dofs as garbage.** `newton_solve()` throws `NewtonSolverError`
   mid-iteration and restores nothing. Contrast `adaptive_unsteady_newton_solve`, which backs the
   dofs up before the step *precisely so it can retry* ("we need to backup the existing dofs, in
   case the timestep is [rejected]", `:11350`). The adaptation loops have no such backup, so even
   the freshly *interpolated* post-adapt state - a perfectly serviceable field, and the natural
   starting point for any retry - is destroyed by the attempt that failed.

3. **The error is upgraded to fatal.** Every relevant catch block converts `NewtonSolverError` into
   `OomphLibError` and says so ("Die horribly!!"). The two callers in this codebase that know how to
   recover from a Newton failure - the `dt`-halving in `adaptive_unsteady_newton_solve` and the
   `Ds *= 2/3` in `arc_length_step_solve_helper` - only catch `NewtonSolverError` and
   `InvertedElementError`. This is the same asymmetry described at `src/nanobind/solver.cpp:204`,
   where a solver backend's own error had to be re-reported as `NewtonSolverError(0, DBL_MAX)` for
   exactly this reason. So nothing upstream ever sees it.

Note the ordering in the doubly-adaptive path: the temporal step is *accepted* at `:11699` before
the adapt and the re-solve. By the time the re-solve fails, the one recovery mechanism that exists
(halve `dt`, retry) has already returned and is out of scope.

### 1.1 Why `globally_convergent_newton` does not fix it

A line search only helps when the residual descends towards a nearby root. After an unrefinement
that removes resolution where the solution is stiff, or an interpolation onto a mesh whose Jacobian
is near-singular, the interpolated state is not in the basin at all: the line search stalls at a
tiny step length and reports the same failure. The initial state (or the mesh change) is the
problem, not the step length. See also §7.1 - enabling it in a retry is not merely useless here, it
currently aborts the process.

---

## 2. The one observation the whole design rests on

**The adaptation loops always solve first and adapt afterwards.** Look at any of them: `isolve == 0`
solves, and only `isolve > 0` adapts. So the state that exists immediately before an `adapt()` is
not some intermediate - it is a *converged* solution, or, in the transient loops, a **completed
timestep** (the time reset `time_pt()->time() = initial_time` happens after the adapt, not before).

That turns the hard problem ("reconstruct something usable from wreckage") into an easy one
("remember the good state for the duration of one adaptation"). And it gives the recovery a strategy
that cannot fail: roll back and accept. You do not get the refined mesh, but you keep the solution
you already had, and the run continues.

---

## 3. What was added to oomph-lib

Two empty virtual hooks and one exception class, ~40 lines. Both hooks are no-ops by default, so a
`Problem` that does not override them behaves exactly as before. Recorded in
`src/thirdparty/INFO_oomph-lib` under "3rd August 2026, adaptive-resolve recovery".

```cpp
// problem.h, next to actions_after_adapt()
virtual void adaptive_solve_checkpoint(const unsigned& isolve, const bool& just_adapted) {}
virtual bool recover_from_failed_adaptive_resolve(const bool& linear_solver_error,
                                                  const unsigned& iterations) { return false; }

// problem.h, next to NewtonSolverError
class AdaptiveResolveRecovered : public std::runtime_error { ... };
```

`adaptive_solve_checkpoint` is called at five sites - immediately before and after the `adapt()` in
`newton_solve(max_adapt)` and in `unsteady_newton_solve(dt,max_adapt,...)`, and around the single
`adapt()` in `doubly_adaptive_unsteady_newton_solve_helper`. The "after" call sits as late as
possible, past the convergence break and past the `set_initial_condition()` re-assignment, so the
state it captures is exactly what the re-solve starts from.

`recover_from_failed_adaptive_resolve` is called from the three fatal `catch (NewtonSolverError&)`
blocks, ahead of the existing code:

```cpp
if (recover_from_failed_adaptive_resolve(error.linear_solver_error, error.iterations))
  throw AdaptiveResolveRecovered("...");
... unchanged: report, then throw OomphLibError ...
```

`AdaptiveResolveRecovered` derives from `std::runtime_error` and deliberately **not** from
`OomphLibError` or `NewtonSolverError`: it must fly past every catch block in oomph-lib. In
particular `steady_newton_solve()` wraps `newton_solve(max_adapt)` in `catch (NewtonSolverError&)`,
so an exception of either of those types thrown from the inner loop would simply be converted back
into a fatal error one frame up. Being a type nothing catches also means the hook cannot fire twice
for one failure.

**The arclength path was deliberately left alone.** `arc_length_step_solve_helper` keeps
`Dof_pt`-indexed `dof_current`/`dof_derivative` vectors live across its `adapt()` (`:10952`), so a
rollback there would have to re-establish them; and its non-linear-solver failures already recover
by shrinking `Ds`. Recovering that path is a separate piece of work.

---

## 4. The pyoomph side

`pyoomph::Problem` overrides both hooks and forwards them to Python as `_adaptive_solve_checkpoint`
/ `_recover_from_failed_adaptive_resolve` (`src/problem.hpp`). The `_`-prefixed indirection exists
because oomph-lib's `const unsigned&` / `const bool&` signatures are awkward through the trampoline
- the same reason `adapt()` forwards to `_adapt()`. `NB_TRAMPOLINE(pyoomph::Problem, 16)` had to
become `18`; it was exactly full.

`AdaptiveResolveRecovered` is translated to `pyoomph.generic.adaptive_recovery.SpatialAdaptResolveError`
by the translator in `src/nanobind/solver.cpp`, reusing the lazy `lookup_python_type_once` helper
already there. Its own class rather than the `RuntimeError` that `std::runtime_error` would map to,
because `Problem.solve()`'s retry loop has to tell this apart from every other failure - it is the
one case in which the problem is still usable afterwards.

### 4.1 Two checkpoints

* **Before `adapt()`** - the converged pre-adapt state. The only thing that can bring the old mesh
  back, and the one `accept_unadapted` falls back to.
* **After `adapt()`, before the re-solve** - new mesh, interpolated field, and (transient) the time
  back at the start of the step. This is exactly what the failed Newton destroys, and what a retry
  on the refined mesh needs.

Both are full state snapshots. A dof-only variant for the second one was considered and dropped: it
would not restore time or the history values, which makes it wrong for the transient path, and the
saving does not justify a second code path with a subtly narrower contract.

At most three snapshots live at once (the two above plus the never-overwritten first pre-adapt one),
and only while a `solve(spatial_adapt>0)` with an active policy is running - `end_solve` drops them
in a `finally`.

### 4.2 The snapshots reuse the state-file machinery

`save_state`/`load_state` already round-trip everything needed: refinement pattern, dofs, history
values, pinned values, current time, all `dt`s, continuation data. Two adjustments were made:

* `DumpFile` (`pyoomph/output/states.py:38`) now accepts an already-open binary stream as well as a
  filename - it only ever used `read`/`write`/`seek`/`tell`. `Problem._snapshot_state()` passes an
  `io.BytesIO`, so a snapshot costs no disk and does not disturb `_states/` or `--runmode continue`.
  `close()` does not close a stream it does not own. `save_state`/`load_state` grew a `quiet` flag so
  a snapshot does not narrate itself.
* `save_state`'s "only one rank writes" guard is skipped for streams. That guard exists so that N
  ranks holding the same non-distributed problem do not all write the same *file*; a stream is
  per-rank private, and a snapshot only rank 0 held could not be restored, because every rank has to
  read the whole state back.

For a genuinely distributed problem, `save_state` merges into one partition-independent stream that
only rank 0 ends up holding, so `_snapshot_state` broadcasts it. That `bcast` is the only extra cost
a distributed snapshot has over a serial one.

### 4.3 MPI

Both modes work and are tested (`tests/test_mpi_adaptive_recovery.py`, np=2 and 4, with and without
`--distribute`). They are genuinely different:

* **Without `--distribute`** every rank holds the whole problem, so a snapshot is per-rank private
  and *every* rank has to write a complete one. Two guards had to be relaxed for that, and each was
  correct until an in-memory snapshot existed - see §4.4.
* **With `--distribute`** the mesh is merged onto rank 0 and broadcast back, as above.

The harder question is **agreement**. Everything `handle_failure` does is collective (`load_state` ->
mesh rebuild -> `assign_eqn_numbers`), so it must only run when every rank is in it - and it cannot
check that itself, because an `MPI_Allreduce` placed there *is* the deadlock it would be meant to
prevent: a rank that did not fail never reaches the line. The agreement therefore has to already
exist where the failure is decided, and for three of the four ways in it does:

| failure | agreed because |
| --- | --- |
| `max_residuals` exceeded | `maxres` comes from `DoubleVector::max()`, an `MPI_Allreduce` for a distributed vector (`double_vector.cc:646`) - every rank compares the same number |
| `Max_newton_iterations` reached | the iteration count is identical on every rank |
| pyoomph's abort request | `consume_newton_abort_request()` (`src/problem.cpp:1535`) Allreduces the request *before* throwing, precisely so a rank-local decision becomes a global failure |
| **linear-solver error** | **nothing guarantees it** - a backend may report failure on some ranks only |

So `handle_failure` declines outright when `linear_solver_error` is set on a distributed problem.
That leaves exactly that case as fatal as it was before, which is the safe direction: a hang is far
worse than the error the user already had.

The MPI test injects the failure **from rank 0 only** for this reason - it is the partition-dependent
case, and it passing is what demonstrates that the abort request's Allreduce turns one rank's
decision into every rank's failure. If that agreement ever breaks, the test hangs rather than fails,
so `_run()` has a timeout that reports as an assertion.

### 4.4 Two guards that the MPI tests broke, and why they were right before

Both are cases where an existing rank-0-only shortcut was correct precisely *because* the other ranks
had already been dropped, and stopped being correct once a rank could write a private stream:

* `save_state` chose `DumpFile(fname if rank==0 else os.devnull)`. Fixing only the early return
  above it left every non-zero rank writing its snapshot to `/dev/null`; the restore then failed with
  "Unsupported state file", and rank 0 hung in the next `mpi_barrier()` of the mesh rebuild. Now
  gated on `distributed` as well.
* `save_mesh_state` (`pyoomph/meshes/meshstate.py`) returned early for `rank != 0` unconditionally.
  That is right when the contributions were gathered onto rank 0, but on a non-distributed mesh
  `_sorted_records` hands *every* rank the complete set. The symptom was a 216-byte snapshot on
  rank 1 against 23818 on rank 0. Also gated on `distributed` now.

Neither changes the file path: `save_state` still drops the redundant writers before either point,
which the byte-identical-file assertions in `tests/test_mpi_state_files.py` confirm still hold.

---

## 5. The policy object

`AdaptiveResolveRecovery` (`pyoomph/generic/adaptive_recovery.py`) holds all the decisions. Strategies
are tried in order, each starting from whichever snapshot it needs:

| strategy | starts from | does |
| --- | --- | --- |
| `keep_adapted` | post-adapt | re-solves on the refined mesh with `spatial_adapt=0`. Adapting again from here walks into the same failure. |
| `refine_only` | pre-adapt | re-adapts with unrefinement suppressed, then solves. Unrefinement is the usual culprit: it removes resolution where the solution is stiff. |
| `pseudo_transient` | post-adapt | takes a few pseudo-timesteps to relax into the basin, then the stationary solve. Skipped during a timestep. |
| `accept_unadapted` | first pre-adapt | restores and returns. **Cannot fail** - that state is a converged solution, or a completed timestep. |

Default order: `keep_adapted`, `refine_only`, `accept_unadapted`. Exhausting the list re-raises
`SpatialAdaptResolveError` - with the state consistent, so it can be caught and the problem is still
usable.

`refine_only` suppresses unrefinement through `Problem._suppress_unrefinement()`, which sets
`mesh.min_permitted_error = -1` (errors are non-negative and oomph unrefines on `error < tol`) and
temporarily unsets `desired_ndof`, since that controller recomputes *both* thresholds on every
adaptation and would otherwise overwrite the setting.

### 5.1 Retries, and what must not be retried

Two things had to be got right:

**Shift.** Every snapshot is taken *after* the history values were shifted for this step - oomph-lib
shifts in the `isolve == 0` pass, before any `adapt()`. A restored state must therefore never be
shifted again, so every retry passes `shift_values=False`.

**Not stealing oomph-lib's own recovery.** `_invoke()` clears the snapshots before handing control
back to oomph-lib. A failure that happens *before* this attempt's first `adapt()` is an ordinary
Newton or timestep failure, which `adaptive_unsteady_newton_solve` recovers from perfectly well by
halving `dt`. Without the clear, stale snapshots from the previous attempt would make
`handle_failure` claim that failure too and replace a working recovery with ours. The one exception:
on a retry with nothing adapted yet, `handle_failure` falls all the way back to the first pre-adapt
snapshot rather than declining, because declining there would mean a fatal error despite a good
state being available.

---

## 6. Reentrancy: is it safe to rebuild the mesh from inside the C++ frame?

Yes, and this is now verified rather than assumed. The adaptive loops hold only local scalars across
the call (`isolve`, `n_refined`/`n_unrefined`, `initial_time`, `dt_taken`, `new_dt`) - nothing that
points into the mesh, the dof vector or the distribution - and the `DoubleVector`s of the failed
`newton_solve()` are destroyed during unwinding before the catch block runs. A `load_state` from
inside the catch block, followed by further solves, works.

The exception is `arc_length_step_solve_helper`, which is why §3 leaves it alone.

---

## 7. Two things found on the way, both pre-existing

### 7.1 `globally_convergent_newton` armed a heap overflow (found here, FIXED)

Enabling the globally convergent Newton method for the `keep_adapted` retry crashed the process.
Root-caused with valgrind and fixed in `problem.cc` / `matrices.cc`; see
`src/thirdparty/INFO_oomph-lib`, "globally convergent Newton heap overflow", and
`tests/test_globally_convergent_newton.py`.

In short: `Problem::newton_solve()` switched the linear solver's gradient computation on and nothing
ever switched it off, so every *later* solve kept computing the gradient into a vector that
`multiply_transpose()` only resizes when it is unbuilt - and the first solve after the dof count grew
wrote past the end of it. Any `solve(globally_convergent_newton=True)` followed by a solve that adds
dofs did it; the recovery just happened to be a way to reach that sequence.

Worth recording as a debugging lesson, because it cost two wrong diagnoses: the overflow is silent,
and the crash lands nowhere near the cause and in a *different place on different runs* - inside MKL
Pardiso's allocator on one run, as a null `oomph::Node::position()` during the next residual assembly
on another. Both looked like self-contained bugs in the place they surfaced, and neither was. What
settled it was one valgrind run, which found exactly two errors, both at the real line.

`retry_globally_convergent_newton` still defaults to `False`, now for the original reason only: it is
not a promising cure for this particular failure (§1.1), since the interpolated state is usually
outside the basin rather than merely too far along a good direction.

### 7.2 `load_state` can renumber the dofs

Restoring a state can hand back a different dof *ordering* than was in force when it was written -
`get_current_dofs()` afterwards is a permutation of what it was before. The values are preserved
bit-for-bit (a sorted comparison is exactly equal) and the field is the same field, which is what a
rollback needs, so the recovery is unaffected and the tests compare accordingly.

But it is worth knowing, because a state file also stores **dof-indexed** sections - eigenvectors
and the arclength dof-derivative vector. If the numbering after a load differs from the one at save
time, those land on the wrong dofs. Observed with a snapshot of the unrefined base mesh; a snapshot
of a refined mesh round-tripped in the same order. **Not investigated further here.**

---

## 8. What is not done

* **User-facing documentation.** There is no tutorial chapter on this, so nobody will find it. That,
  plus a general troubleshooting chapter for non-convergence and the undocumented
  `--largest_residuals` flag, is written up in `dev_docs/nonconvergence_diagnostics.md` together with
  the diagnostics that would be worth building (condition-number estimation, Lagrange-multiplier
  consistency checks, and a structural inconsistency analysis on the equation tree - the "Stokes with
  pure Dirichlet boundaries and no pressure constraint" class of problem, which pyoomph could detect
  symbolically at setup time instead of as a divergence later).

* **Recovery from a rank-local linear-solver error under `--distribute`** - declined on purpose,
  §4.3. Making it work needs the agreement established at the point of failure, the way
  `consume_newton_abort_request` does; the natural place is
  `throw_solver_failure_as_newton_error` (`src/nanobind/solver.cpp`), where every rank *is* inside
  the linear solve and an Allreduce would therefore be safe.
* **Arclength continuation** (§3).
* **In-place retry inside the C++ loop.** Everything currently unwinds to Python and retries there,
  which is enough because `run()` drives one `solve()` per step. If it is ever wanted, it is
  additive: give `recover_from_failed_adaptive_resolve` an `int` return (0 fatal, 1
  recovered-and-unwind, 2 retry-this-solve) and wrap the re-solve in a small retry loop. Note that
  `dt` is a `const double&` there, so "retry with a smaller step" cannot be done in the hook at all.
* **A correction to an earlier draft of this document**, which claimed the three Python-side
  adaptation loops (`initialise`'s initial adaption, `force_remesh`, `redefine_problem`) would get
  the same protection "for free". They would not, because none of them solves between adaptations -
  they interpolate or re-assign the initial condition. They were never the failure site.

---

## 9. Tests

`tests/test_adaptive_resolve_recovery.py`, 10 tests, ~4 s. The failure is injected with
`_request_newton_abort()` rather than by hunting for a genuinely ill-conditioned problem: it fires at
a chosen point, reproducibly, and lands in exactly the catch block an ordinary divergence lands in
(`Problem::consume_newton_abort_request` throws `NewtonSolverError(0, DBL_MAX)` for precisely that
reason). What is under test is the control flow, not the conditioning that provokes it in practice.
The problem itself is linear, so a clean run is genuinely clean.

Covered: the default is still fatal; a clean solve is undisturbed; the rollback reproduces the
pre-adapt solution and mesh size; the problem is usable (and adaptable) afterwards; `keep_adapted`
keeps the refined mesh; an empty strategy list raises `SpatialAdaptResolveError` and leaves a finite
state; `active=False` behaves like no policy; the transient path keeps the completed step; and the
snapshot/restore round trip, in memory and through a file.

`tests/test_mpi_adaptive_recovery.py`, 8 tests (np=2 and 4, with and without `--distribute`), ~12 s,
marked `slow` so they need `--full`. The failure is injected from rank 0 only (§4.3). Asserted: the
sabotage fires exactly once on every rank; the rollback takes every rank back to the pre-adapt
`ndof`; every rank ends with the *same* mesh size and the same solution, so a recovery that ran on
some ranks only would be caught rather than merely not hanging; the problem is usable afterwards;
and the snapshot is non-empty and the same size on every rank. `_run()` also asserts that the
distributed half really distributed - without that check it would pass while exercising the
replicated code path, which is a different one in all three of `save_state`, `save_mesh_state` and
`_snapshot_state`.
