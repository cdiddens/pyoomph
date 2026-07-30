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

# Abandoning a running Newton solve.
#
# An equation can reject a step from before_newton_convergence_check() -- the real user is
# pyoomph/equations/topological_changes.py, which rejects a step that would make two interfaces
# overlap so that the time-stepper cuts dt and retries.
#
# This used to be implemented by multiplying the whole dof vector by 1e40 and adding noise, so that
# the NEXT residual evaluation would exceed max_residuals and make oomph-lib throw. That worked, but:
#
#   * it destroyed the state. The rejected configuration -- the one you actually want to look at when
#     asking "why was this rejected?" -- was overwritten with 1e40s before anything could see it;
#   * it went through get_current_dofs()/set_current_dofs(), which redistribute and are therefore
#     COLLECTIVE, while the decision to reject is typically only reached on the ranks holding the
#     offending part of the mesh. A rejection seen by some ranks and not others deadlocked;
#   * it paid for a full extra residual assembly, and an O(ndof) gather/scatter, to communicate one bit;
#   * it only worked if 1e40 * dof happened to produce a residual above max_residuals.
#
# What replaces it is a request that the next residual evaluation consumes, agreed across ranks first.
# These tests pin down the part a caller can observe: the solve is still abandoned with the same
# exception, and the dofs survive it.

import numpy
import pytest

from pyoomph import Problem, Equations, DirichletBC
from pyoomph.expressions import var, dot
from pyoomph.equations.navier_stokes import NavierStokesEquations
from pyoomph.meshes.simplemeshes import RectangularQuadMesh


class _RejectAfter(Equations):
    """Rejects every convergence check after the first `after` of them (`after < 0` never rejects)."""

    def __init__(self, after):
        super().__init__()
        self.after = after
        self.n = 0

    def before_newton_convergence_check(self, eqtree):
        self.n += 1
        return self.after < 0 or self.n <= self.after


class _Cavity(Problem):
    def __init__(self, reject_after):
        super().__init__()
        self.reject_after = reject_after

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=6))
        eqs = NavierStokesEquations(dynamic_viscosity=0.05, mass_density=1)
        for b in ["left", "right", "bottom"]:
            eqs += DirichletBC(velocity_x=0, velocity_y=0) @ b
        eqs += DirichletBC(velocity_x=1, velocity_y=0) @ "top"
        eqs += DirichletBC(pressure=0) @ "bottom/left"
        eqs += _RejectAfter(self.reject_after)
        self.add_equations(eqs @ "domain")


def _run(tmp_path, reject_after):
    """Returns (raised_exception_or_None, dofs_afterwards)."""
    p = _Cavity(reject_after)
    with p:
        p.set_output_directory(str(tmp_path))
        p.quiet()
        p.initialise()
        try:
            p.solve()
            err = None
        except Exception as e:  # noqa: BLE001 -- the type is what is under test
            err = e
        return err, numpy.array(p.get_current_dofs()[0])


def test_not_rejecting_still_solves(tmp_path):
    """The control: the machinery must not disturb an ordinary solve."""
    err, dofs = _run(tmp_path, -1)
    assert err is None, "an unrejected solve raised %r" % (err,)
    assert numpy.all(numpy.isfinite(dofs))


@pytest.mark.parametrize("reject_after", [0, 2], ids=["immediately", "after-two-steps"])
def test_rejecting_a_step_abandons_the_solve(tmp_path, reject_after):
    err, dofs = _run(tmp_path, reject_after)
    assert err is not None, "the solve was not abandoned"

    # The dofs are the point of the refactor. The old mechanism left every one of them at ~1e40, so
    # the rejected state could not be inspected, written out, or recovered from; a caller that wanted
    # to reduce dt and retry had to restore from its own backup. They must now be untouched -- which
    # here means finite and of the same order as an ordinary intermediate Newton iterate.
    assert numpy.all(numpy.isfinite(dofs)), "the abandoned solve left non-finite dofs"
    assert numpy.max(numpy.abs(dofs)) < 1e3, \
        "the abandoned solve left dofs of magnitude %.3e -- the state was destroyed, not preserved" \
        % numpy.max(numpy.abs(dofs))


def test_abort_request_is_consumed_and_does_not_leak(tmp_path):
    """A request must fire exactly once, not poison every later solve on the same problem."""
    p = _Cavity(-1)
    with p:
        p.set_output_directory(str(tmp_path))
        p.quiet()
        p.initialise()
        assert not p._newton_abort_requested()

        p._request_newton_abort("test")
        assert p._newton_abort_requested()
        with pytest.raises(Exception):
            p.solve()
        # Consumed by the residual evaluation that acted on it.
        assert not p._newton_abort_requested()

        # And the problem is still usable: no state was destroyed, so this just solves.
        p.solve()
        dofs = numpy.array(p.get_current_dofs()[0])
        assert numpy.all(numpy.isfinite(dofs))
        assert numpy.max(numpy.abs(dofs)) < 1e3
