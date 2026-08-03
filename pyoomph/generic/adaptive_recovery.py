from __future__ import annotations
#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  @author Maxim de Wildt <m.dewildt@utwente.nl>
#
#  @section LICENSE
#
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC
#  Copyright (C) 2021-2026  Christian Diddens, Duarte Rocha & Maxim de Wildt
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#  The main author may be contacted at c.diddens@utwente.nl
#
# ========================================================================

"""Recovery from a failed Newton solve after a spatial adaptation.

Without this, such a failure ends the run: oomph-lib's adaptation loops convert the
``NewtonSolverError`` into a fatal ``OomphLibError`` and there is nothing left to recover from
anyway, because ``adapt()`` has already replaced the mesh and the diverged Newton has overwritten
the dofs. See ``dev_docs/adaptive_resolve_recovery.md`` for the analysis.

The way out rests on one observation about *when* the adaptation loops adapt: they always solve
first and adapt afterwards, so the state that exists immediately before an ``adapt()`` is a
**converged** one - a completed timestep, or a converged stationary solution, on the mesh as it was.
Snapshotting there costs one state serialisation and turns the fatal case into "we keep the solution
we already had, just not on the refined mesh".
"""

import math

from ..typings import *

if TYPE_CHECKING:
    from .problem import Problem


class SpatialAdaptResolveError(RuntimeError):
    """The Newton solve after a spatial adaptation failed.

    Raised (from the C++ side, see ``pyoomph::AdaptiveResolveRecovered``) *only* once a recovery
    handler has put the problem back into a consistent state, so unlike the fatal error it replaces
    it can be caught and the problem can still be used afterwards.
    """

    def __init__(self, msg: str = "", linear_solver_error: bool = False, iterations: int = 0):
        super().__init__(msg)
        #: Whether the failure came from the linear solver rather than from Newton divergence.
        self.linear_solver_error = linear_solver_error
        #: Newton iterations performed when it gave up.
        self.iterations = iterations


class _Snapshot:
    """One in-memory state file plus the bookkeeping to tell snapshots apart in messages."""

    __slots__ = ("data", "isolve", "just_adapted", "ndof")

    def __init__(self, data: bytes, isolve: int, just_adapted: bool, ndof: int):
        self.data = data
        self.isolve = isolve
        self.just_adapted = just_adapted
        self.ndof = ndof

    def __repr__(self) -> str:
        return ("post-adapt" if self.just_adapted else "pre-adapt") + " state at adaptation step " \
            + str(self.isolve) + " (ndof=" + str(self.ndof) + ")"


class AdaptiveResolveRecovery:
    """Policy for what to do when the Newton solve after a spatial adaptation fails.

    Attach one to a problem to stop such a failure from ending the run::

        problem.adaptive_resolve_recovery = AdaptiveResolveRecovery()

    Set it to ``None`` (the default) to get the old behaviour, in which the failure is fatal.

    The strategies are tried in the order given by :py:attr:`strategies`, each starting from the
    relevant snapshot. ``accept_unadapted`` cannot fail and is therefore what the list should end
    with; without it, exhausting the list re-raises :py:class:`SpatialAdaptResolveError`.

    Args:
        active: Whether to do anything at all. Turning this off leaves the snapshots untaken too, so
            it costs nothing.
        strategies: What to try, in order. See :py:attr:`strategies`.
        max_attempts: Upper bound on retries per solve, independent of how long ``strategies`` is.
            Guards against a strategy that keeps failing in a new way each time.
        retry_newton_relaxation_factor: Newton relaxation for the retries. ``None`` (the default)
            leaves the problem's own setting alone. Note that relaxing costs iterations: if this is
            set and ``retry_max_newton_iterations`` is not, the cap is raised in proportion, because
            otherwise the retry runs out of iterations rather than diverging and the strategy looks
            like it failed when it was only cut short.
        retry_globally_convergent_newton: Whether the retries switch the globally convergent Newton
            method on. **Off by default, deliberately.** It is not a cure for this failure anyway -
            the interpolated state is usually outside the basin, not merely too far along a good
            direction, and a line search then stalls at a tiny step length and reports the same
            failure. And enabling it here has been observed to abort the process outright (SIGSEGV
            inside MKL Pardiso's own teardown, so the whole MPI job dies); that is a separate,
            pre-existing problem with that path, but it is not one to walk into by default.
        retry_max_newton_iterations: Newton iteration cap for retries. ``None`` keeps the problem's,
            unless a relaxation factor is set - see above.
        pseudo_transient_steps: Number of pseudo-timesteps taken by the ``pseudo_transient``
            strategy before it tries the stationary solve again.
        pseudo_transient_dt: Size of those pseudo-timesteps. ``None`` derives one from the problem's
            temporal scale.
        keep_snapshots_in_memory: Keep the snapshots as bytes in memory (the default). If a problem
            is large enough that this hurts, set it to ``False`` and the snapshots go to files in
            the output directory's ``_states/`` instead, under names that are cleaned up afterwards.
        quiet: Suppress the running commentary about what is being restored and retried.
    """

    #: Restore the post-adapt state (new mesh, interpolated field) and retry the solve there with
    #: fallback solver settings, without adapting again.
    STRATEGY_KEEP_ADAPTED = "keep_adapted"
    #: Roll back to the pre-adapt state and adapt again with unrefinement suppressed. Unrefinement
    #: is the usual culprit: it removes resolution exactly where the solution is stiff.
    STRATEGY_REFINE_ONLY = "refine_only"
    #: Restore the post-adapt state and relax into the basin with a few pseudo-timesteps before
    #: retrying the stationary solve. Stationary solves only; ignored for a transient step.
    STRATEGY_PSEUDO_TRANSIENT = "pseudo_transient"
    #: Roll back to the pre-adapt state and accept it. Cannot fail: that state is the converged
    #: solution the loop had before it decided to adapt.
    STRATEGY_ACCEPT_UNADAPTED = "accept_unadapted"

    def __init__(self, *, active: bool = True,
                 strategies: Sequence[str] = (STRATEGY_KEEP_ADAPTED, STRATEGY_REFINE_ONLY, STRATEGY_ACCEPT_UNADAPTED),
                 max_attempts: int = 4,
                 retry_newton_relaxation_factor: float | None = None,
                 retry_globally_convergent_newton: bool = False,
                 retry_max_newton_iterations: int | None = None,
                 pseudo_transient_steps: int = 5,
                 pseudo_transient_dt: ExpressionNumOrNone = None,
                 keep_snapshots_in_memory: bool = True,
                 quiet: bool = False):
        self.active = active
        self.strategies = list(strategies)
        self.max_attempts = max_attempts
        self.retry_newton_relaxation_factor = retry_newton_relaxation_factor
        self.retry_globally_convergent_newton = retry_globally_convergent_newton
        self.retry_max_newton_iterations = retry_max_newton_iterations
        self.pseudo_transient_steps = pseudo_transient_steps
        self.pseudo_transient_dt = pseudo_transient_dt
        self.keep_snapshots_in_memory = keep_snapshots_in_memory
        self.quiet = quiet

        # Per-solve state, all reset by begin_solve.
        self._in_solve = False
        self._transient = False
        self._spatial_adapt = 0
        self._pre: _Snapshot | None = None       # latest state before an adapt()
        self._post: _Snapshot | None = None      # latest state after an adapt(), before the resolve
        self._first_pre: _Snapshot | None = None # the very first one, never overwritten
        self._failures = 0
        #: Number of failures this policy has absorbed over the problem's lifetime. Useful in tests
        #: and as a "was this run clean?" indicator afterwards.
        self.total_failures = 0

    # ------------------------------------------------------------------ helpers

    def _say(self, problem: "Problem", *args: Any) -> None:
        if self.quiet or problem.is_quiet():
            return
        print("ADAPT RECOVERY:", *args)

    def _take(self, problem: "Problem", isolve: int, just_adapted: bool) -> _Snapshot:
        return _Snapshot(problem._snapshot_state(to_memory=self.keep_snapshots_in_memory),
                         isolve, just_adapted, problem.ndof())

    def _restore(self, problem: "Problem", snap: _Snapshot) -> None:
        self._say(problem, "restoring the", repr(snap))
        problem._restore_state(snap.data)

    # ------------------------------------------------------- called from solve()

    def begin_solve(self, problem: "Problem", spatial_adapt: int, transient: bool) -> None:
        self._in_solve = True
        self._transient = transient
        self._spatial_adapt = spatial_adapt
        problem._adapt_recovery_transient = transient
        self._pre = self._post = self._first_pre = None
        self._failures = 0

    def end_solve(self, problem: "Problem") -> None:
        # Dropping the snapshots here is what bounds the memory: at most three states live at once,
        # and only while a solve with spatial_adapt>0 is running.
        self._in_solve = False
        if not self.keep_snapshots_in_memory:
            for snap in (self._pre, self._post, self._first_pre):
                if snap is not None:
                    problem._discard_state_snapshot(snap.data)
        self._pre = self._post = self._first_pre = None

    # --------------------------------------------------- called from the C++ hooks

    def checkpoint(self, problem: "Problem", isolve: int, just_adapted: bool) -> None:
        """Snapshot immediately before (``just_adapted=False``) or after an ``adapt()``."""
        if not self.active or not self._in_solve:
            return
        snap = self._take(problem, isolve, just_adapted)
        if just_adapted:
            self._post = snap
        else:
            self._pre = snap
            if self._first_pre is None:
                # Kept separately because a retry that adapts again overwrites self._pre, and
                # accept_unadapted has to be able to fall all the way back to where the solve
                # started - which is the last state known to be converged.
                self._first_pre = snap

    def handle_failure(self, problem: "Problem", linear_solver_error: bool, iterations: int) -> bool:
        """Make the state consistent again. Returns False to let the failure stay fatal.

        Only the *consistency* is decided here; which snapshot a retry wants is decided later, at
        Python level, where restoring another one is just another load. What matters is that this
        returns having left no garbage behind, because the C++ frames it returns into will unwind
        through code that assumes a usable problem.
        """
        if not self.active or not self._in_solve:
            return False

        # Everything this handler does is COLLECTIVE (_restore_state -> load_state -> mesh rebuild ->
        # assign_eqn_numbers), so it must only run when every rank is here. It cannot check that
        # itself: an Allreduce placed here would be the deadlock it is meant to prevent, since a rank
        # that did not fail never reaches this line at all. So the agreement has to be established
        # where the failure is decided, and for three of the four ways in it already is:
        #
        #   * max_residuals exceeded - maxres comes from DoubleVector::max(), which is an
        #     MPI_Allreduce for a distributed vector (double_vector.cc:646), so every rank compares
        #     the same number and throws together;
        #   * Max_newton_iterations reached - the iteration count is identical on every rank;
        #   * pyoomph's own abort request - Problem::consume_newton_abort_request() Allreduces the
        #     request before throwing, precisely so a rank-local decision becomes a global failure.
        #
        # The fourth, a linear-solver error, is the one with no such guarantee: a backend may report
        # failure on some ranks only. Declining leaves that case exactly as fatal as it was, which is
        # the safe direction - a hang is far worse than the error the user already had.
        if linear_solver_error and problem.is_distributed():
            self._say(problem, "declining to recover from a linear-solver failure on a distributed "
                               "problem: it may have been seen on some ranks only, and recovering "
                               "would deadlock the ranks that did not see it")
            return False

        target = self._post if self._post is not None else self._pre
        if target is None:
            # Nothing was adapted yet in this attempt. On the first attempt that means the failure
            # has nothing to do with adaptivity - it is an ordinary Newton divergence, and
            # oomph-lib's own recovery (adaptive_unsteady_newton_solve halves dt) is both better
            # placed and already there, so stay out of its way and leave it fatal if it is.
            if self._failures == 0:
                return False
            # On a retry it means we cannot localise the failure any further, so fall all the way
            # back instead - that state is still known to be converged.
            target = self._first_pre
            if target is None:
                return False
        self._failures += 1
        self.total_failures += 1
        self._say(problem, "the Newton solve after the adaptation failed"
                  + (" in the linear solver" if linear_solver_error else " after " + str(iterations) + " iterations")
                  + " - restoring a consistent state")
        self._restore(problem, target)
        return True

    # ------------------------------------------------------- the retry loop itself

    def run(self, problem: "Problem", do_solve: Callable[[int, bool], Any],
            spatial_adapt: int, shift_values: bool, transient: bool) -> Any:
        """Run ``do_solve(spatial_adapt, shift_values)``, applying the strategies if it fails.

        ``do_solve`` is the actual oomph-lib call. It is re-invoked on a retry with a possibly
        reduced adaptation level and always with ``shift_values=False``: every snapshot is taken
        *after* the history values were shifted for this step (oomph-lib shifts in the very first
        pass through its adaptation loop, before any adapt()), so a restored state must not be
        shifted a second time.
        """
        self.begin_solve(problem, spatial_adapt, transient)
        try:
            return self._run(problem, do_solve, shift_values)
        finally:
            self.end_solve(problem)

    def _invoke(self, do_solve: Callable[[int, bool], Any], spatial_adapt: int, shift: bool) -> Any:
        # Forget the previous attempt's snapshots before handing control back to oomph-lib. A
        # failure that happens BEFORE this attempt's first adapt() is an ordinary Newton or timestep
        # failure, which oomph-lib recovers from perfectly well on its own (adaptive_unsteady_-
        # newton_solve halves dt and retries). Leaving stale snapshots around would make
        # handle_failure claim that failure too and replace a working recovery with ours.
        self._pre = self._post = None
        return do_solve(spatial_adapt, shift)

    def _run(self, problem: "Problem", do_solve: Callable[[int, bool], Any], shift_values: bool) -> Any:
        try:
            return self._invoke(do_solve, self._spatial_adapt, shift_values)
        except SpatialAdaptResolveError:
            pass

        attempt = 0
        for strategy in self.strategies:
            if attempt >= self.max_attempts:
                self._say(problem, "giving up after", attempt, "attempts")
                break
            attempt += 1
            if strategy == self.STRATEGY_PSEUDO_TRANSIENT and self._transient:
                continue # meaningless during a timestep; the step itself is already transient
            self._say(problem, "attempt", attempt, "- strategy '" + strategy + "'")
            try:
                handled, result = self._apply(problem, strategy, do_solve)
            except SpatialAdaptResolveError:
                continue # this strategy failed the same way; the state is consistent again, move on
            if handled:
                return result
        # Out of strategies. The state is consistent (handle_failure saw to that) but not solved, so
        # the caller has to hear about it.
        raise SpatialAdaptResolveError(
            "The Newton solve after the spatial adaptation failed and none of the recovery "
            "strategies " + str(self.strategies) + " helped. The problem has been restored to a "
            "consistent state, so this can be caught, but the solve did not converge.")

    def _apply(self, problem: "Problem", strategy: str, do_solve: Callable[[int, bool], Any]) -> tuple[bool, Any]:
        if strategy == self.STRATEGY_ACCEPT_UNADAPTED:
            snap = self._first_pre if self._first_pre is not None else self._pre
            if snap is None:
                return False, None
            self._restore(problem, snap)
            self._say(problem, "accepting the solution on the unadapted mesh")
            # Nothing to solve: that snapshot IS a converged solution (or a completed timestep).
            return True, problem._adapt_recovery_unsolved_result()

        if strategy == self.STRATEGY_KEEP_ADAPTED:
            if self._post is None:
                return False, None
            self._restore(problem, self._post)
            with self._retry_solver_settings(problem):
                # spatial_adapt=0: we already have the adapted mesh and adapting again from here
                # would walk into exactly the same failure.
                return True, self._invoke(do_solve, 0, False)

        if strategy == self.STRATEGY_REFINE_ONLY:
            if self._pre is None:
                return False, None
            self._restore(problem, self._pre)
            with problem._suppress_unrefinement():
                with self._retry_solver_settings(problem):
                    return True, self._invoke(do_solve, self._spatial_adapt, False)

        if strategy == self.STRATEGY_PSEUDO_TRANSIENT:
            if self._post is None:
                return False, None
            self._restore(problem, self._post)
            dt = self.pseudo_transient_dt
            if dt is None:
                dt = problem.get_scaling("temporal")
            self._say(problem, "relaxing with", self.pseudo_transient_steps, "pseudo-timesteps of dt =", dt)
            problem.solve(timestep=[dt] * self.pseudo_transient_steps, spatial_adapt=0)
            with self._retry_solver_settings(problem):
                return True, self._invoke(do_solve, 0, False)

        raise ValueError("Unknown adaptive-resolve recovery strategy: " + str(strategy))

    def _retry_solver_settings(self, problem: "Problem"):
        overrides: dict[str, Any] = {}
        relax = self.retry_newton_relaxation_factor
        if relax is not None:
            overrides["newton_relaxation_factor"] = relax
        if self.retry_globally_convergent_newton:
            overrides["globally_convergent_newton"] = True
        if self.retry_max_newton_iterations is not None:
            overrides["max_newton_iterations"] = self.retry_max_newton_iterations
        elif relax is not None and relax > 0:
            # A relaxed Newton takes roughly 1/relax times as many steps for the same progress, so
            # keeping the problem's cap would make the retry run out of iterations rather than fail
            # to converge - reported identically, but for the wrong reason, and fixable only by a
            # setting the user has no reason to suspect.
            overrides["max_newton_iterations"] = int(math.ceil(problem.max_newton_iterations/relax))
        return problem._temporary_newton_settings(**overrides)
