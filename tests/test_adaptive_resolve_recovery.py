#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
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

# Surviving a Newton failure in the re-solve AFTER a spatial adaptation.
#
# Such a failure used to end the run outright, and there was nothing to catch it with: oomph-lib's
# adaptation loops convert the NewtonSolverError into a fatal OomphLibError, and by then adapt() has
# replaced the mesh and the diverged Newton has overwritten the dofs, so even a caller that DID catch
# something had nothing usable left. See dev_docs/adaptive_resolve_recovery.md.
#
# The failure is injected with _request_newton_abort() rather than by looking for a genuinely
# ill-conditioned problem: it fires at a chosen point, reproducibly, and lands in exactly the catch
# block an ordinary divergence lands in (see Problem::consume_newton_abort_request, which throws
# NewtonSolverError(0, DBL_MAX) for precisely that reason). What is under test is the control flow,
# not the conditioning that provokes it in practice.

import numpy
import pytest

from pyoomph import Problem, DirichletBC, MeshFileOutput
from pyoomph.expressions import var, exp
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.generic.adaptive_recovery import AdaptiveResolveRecovery, SpatialAdaptResolveError
from pyoomph.meshes.simplemeshes import RectangularQuadMesh
from pyoomph.equations.generic import SpatialErrorEstimator


class _SabotagedPoisson(Problem):
    """A peaked Poisson problem whose re-solve after the n-th adaptation is made to fail.

    Poisson is linear, so nothing here diverges by itself: every failure the tests see is the one
    injected on purpose, and a clean run is genuinely clean.
    """

    def __init__(self, fail_after_adapt=1, transient=False):
        super().__init__()
        # No adaptation during initialise(), so that every _adapt() the counter sees is one inside a
        # solve() - which is where the loop under test lives. Otherwise the sabotage arms during
        # initialisation and fires on the very first solve, before anything has been snapshotted.
        self.initial_adaption_steps = 0
        #: Which adaptation to sabotage the re-solve of (1 = the first one). None sabotages nothing.
        self.fail_after_adapt = fail_after_adapt
        self.transient = transient
        self.adapt_count = 0
        self._armed = False
        #: ndof at the moment the sabotaged adaptation was about to happen, i.e. what a rollback to
        #: the pre-adapt state has to reproduce.
        self.ndof_before_sabotaged_adapt = None
        self.dofs_before_sabotaged_adapt = None

    def define_problem(self):
        x = var("coordinate")
        eqs = PoissonEquation(source=exp(-((x[0]-0.5)**2+(x[1]-0.5)**2)/0.002))
        eqs += DirichletBC(u=0) @ ["bottom", "top", "left", "right"]
        eqs += SpatialErrorEstimator(u=1)
        eqs += MeshFileOutput()
        self += RectangularQuadMesh(N=8)
        self += eqs @ "domain"

    def _adapt(self):
        # Arm right before the adaptation whose re-solve is to fail, and record what the pre-adapt
        # state looked like so the test can check that a rollback really reproduces it.
        self.adapt_count += 1
        if self.adapt_count == self.fail_after_adapt:
            self.ndof_before_sabotaged_adapt = self.ndof()
            self.dofs_before_sabotaged_adapt = numpy.array(self.get_current_dofs()[0])
            self._armed = True
        return super()._adapt()

    def actions_before_newton_solve(self):
        # Fires once, on the first solve after the armed adaptation - which is exactly the re-solve.
        if self._armed:
            self._armed = False
            self._request_newton_abort("sabotage: pretend the re-solve after the adaptation diverged")
        return super().actions_before_newton_solve()


def _make(tmp_path, **kw):
    p = _SabotagedPoisson(**kw)
    p.set_output_directory(str(tmp_path))
    p.quiet()
    return p


# ---------------------------------------------------------------------------------------------
# Without a policy: unchanged, and still fatal. This is the control - if it ever stops raising, the
# tests below stop proving anything.
# ---------------------------------------------------------------------------------------------

def test_without_a_policy_the_failure_is_still_fatal(tmp_path):
    with _make(tmp_path) as p:
        assert p.adaptive_resolve_recovery is None, "the default must not change behaviour"
        with pytest.raises(Exception):
            p.solve(spatial_adapt=3)


def test_a_clean_adaptive_solve_is_untouched_by_the_policy(tmp_path):
    """The policy must not disturb a solve that never fails."""
    with _make(tmp_path, fail_after_adapt=None) as p:
        p.adaptive_resolve_recovery = AdaptiveResolveRecovery(quiet=True)
        p.solve(spatial_adapt=3)
        assert p.adaptive_resolve_recovery.total_failures == 0
        assert p.adapt_count > 0, "the problem did not adapt at all, so nothing was exercised"
        dofs = numpy.array(p.get_current_dofs()[0])
        assert numpy.all(numpy.isfinite(dofs))


# ---------------------------------------------------------------------------------------------
# With a policy: the run survives and the state is the pre-adapt one, exactly.
# ---------------------------------------------------------------------------------------------

def test_rollback_reproduces_the_pre_adapt_state(tmp_path):
    """accept_unadapted must restore the converged solution the loop had before it adapted."""
    with _make(tmp_path) as p:
        # Only accept_unadapted, so that what is checked is the rollback itself rather than some
        # retry happening to converge.
        p.adaptive_resolve_recovery = AdaptiveResolveRecovery(
            strategies=[AdaptiveResolveRecovery.STRATEGY_ACCEPT_UNADAPTED], quiet=True)
        p.solve(spatial_adapt=3)

        policy = p.adaptive_resolve_recovery
        assert policy.total_failures == 1, "the sabotage did not fire exactly once"
        assert p.ndof_before_sabotaged_adapt is not None
        assert p.ndof() == p.ndof_before_sabotaged_adapt, \
            "the mesh was not rolled back: ndof=%d, expected %d" % (p.ndof(), p.ndof_before_sabotaged_adapt)
        # Bit-for-bit, but compared as a multiset: the state file round-trip stores the values
        # exactly, yet load_state can hand back a different dof NUMBERING than the one in force when
        # it was written (it addresses the mesh structurally and re-runs assign_eqn_numbers on the
        # rebuilt mesh). The field is the same field either way - which is what a rollback has to
        # guarantee - so compare it in the order-independent form rather than pretending the
        # permutation is a defect of the rollback.
        restored = numpy.sort(numpy.array(p.get_current_dofs()[0]))
        expected = numpy.sort(p.dofs_before_sabotaged_adapt)
        assert numpy.array_equal(restored, expected), \
            "the restored solution differs from the pre-adapt one (max |diff| = %.3e)" \
            % numpy.max(numpy.abs(restored-expected))


def test_the_problem_is_still_usable_afterwards(tmp_path):
    """The whole point: a recovered problem can be solved, output and adapted again."""
    with _make(tmp_path) as p:
        p.adaptive_resolve_recovery = AdaptiveResolveRecovery(quiet=True)
        p.solve(spatial_adapt=3)
        ndof_after_recovery = p.ndof()

        # Nothing is armed any more, so this must simply work - including adapting further.
        p.solve(spatial_adapt=2)
        assert p.ndof() >= ndof_after_recovery
        dofs = numpy.array(p.get_current_dofs()[0])
        assert numpy.all(numpy.isfinite(dofs))
        p.output()


def test_keep_adapted_retries_on_the_new_mesh(tmp_path):
    """The default first strategy keeps the refined mesh and just solves it again.

    The injected failure is a one-shot, so the retry converges and the refined mesh survives -
    which is what distinguishes this from the accept_unadapted fallback above.
    """
    with _make(tmp_path) as p:
        p.adaptive_resolve_recovery = AdaptiveResolveRecovery(
            strategies=[AdaptiveResolveRecovery.STRATEGY_KEEP_ADAPTED,
                        AdaptiveResolveRecovery.STRATEGY_ACCEPT_UNADAPTED], quiet=True)
        p.solve(spatial_adapt=3)
        assert p.adaptive_resolve_recovery.total_failures == 1
        assert p.ndof() > p.ndof_before_sabotaged_adapt, \
            "keep_adapted fell back to the coarse mesh (ndof=%d, pre-adapt was %d)" \
            % (p.ndof(), p.ndof_before_sabotaged_adapt)


def test_exhausting_the_strategies_raises_a_catchable_error(tmp_path):
    """With no strategy that can work, the failure is reported - but as something catchable, and
    with the problem left consistent rather than destroyed."""
    with _make(tmp_path) as p:
        p.adaptive_resolve_recovery = AdaptiveResolveRecovery(strategies=[], quiet=True)
        with pytest.raises(SpatialAdaptResolveError):
            p.solve(spatial_adapt=3)
        # handle_failure restored a state on the way out, so the dofs are a solution, not debris.
        dofs = numpy.array(p.get_current_dofs()[0])
        assert numpy.all(numpy.isfinite(dofs))
        assert numpy.max(numpy.abs(dofs)) < 1e3
        # And it is still usable.
        p.solve(spatial_adapt=0)


def test_inactive_policy_behaves_like_no_policy(tmp_path):
    with _make(tmp_path) as p:
        p.adaptive_resolve_recovery = AdaptiveResolveRecovery(active=False, quiet=True)
        with pytest.raises(Exception) as excinfo:
            p.solve(spatial_adapt=3)
        assert not isinstance(excinfo.value, SpatialAdaptResolveError)


# ---------------------------------------------------------------------------------------------
# The transient path, which reaches oomph-lib through a different adaptation loop
# (unsteady_newton_solve(dt, max_adapt, ...) rather than newton_solve(max_adapt)).
# ---------------------------------------------------------------------------------------------

def test_transient_step_survives_and_keeps_the_completed_step(tmp_path):
    """The pre-adapt snapshot of a transient solve is a COMPLETED timestep, so a rollback means the
    step is kept - taken on the coarser mesh - rather than lost."""
    with _make(tmp_path, transient=True) as p:
        p.adaptive_resolve_recovery = AdaptiveResolveRecovery(
            strategies=[AdaptiveResolveRecovery.STRATEGY_ACCEPT_UNADAPTED], quiet=True)
        p.solve(timestep=0.1, spatial_adapt=3)
        assert p.adaptive_resolve_recovery.total_failures == 1
        assert p.get_current_time(as_float=True) == pytest.approx(0.1), \
            "the timestep was not kept: t=%r" % p.get_current_time(as_float=True)
        dofs = numpy.array(p.get_current_dofs()[0])
        assert numpy.all(numpy.isfinite(dofs))


# ---------------------------------------------------------------------------------------------
# The snapshot mechanism itself, independent of any failure.
# ---------------------------------------------------------------------------------------------

def test_snapshot_and_restore_round_trip(tmp_path):
    """_snapshot_state/_restore_state must be an exact round trip, in memory and through a file."""
    for to_memory in (True, False):
        with _make(tmp_path, fail_after_adapt=None) as p:
            p.solve(spatial_adapt=2)
            before_ndof = p.ndof()
            before = numpy.array(p.get_current_dofs()[0])
            snap = p._snapshot_state(to_memory=to_memory)
            assert isinstance(snap, bytes if to_memory else str)

            # Move away from it: adapt further and re-solve, so both the mesh and the dofs differ.
            p.solve(spatial_adapt=2)
            assert p.ndof() != before_ndof, "the problem did not move, so the restore proves nothing"

            p._restore_state(snap)
            assert p.ndof() == before_ndof
            # Order-independent, for the reason spelled out in the rollback test above.
            assert numpy.array_equal(numpy.sort(numpy.array(p.get_current_dofs()[0])), numpy.sort(before))
            p._discard_state_snapshot(snap)


def test_suppress_unrefinement_is_restored(tmp_path):
    with _make(tmp_path, fail_after_adapt=None) as p:
        p.initialise()
        mesh = p.get_mesh("domain")
        original = mesh.min_permitted_error
        with p._suppress_unrefinement():
            assert mesh.min_permitted_error < 0
        assert mesh.min_permitted_error == original
