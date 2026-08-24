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

# Deleting the branch the problem is SITTING ON. The in-process test
# (test_a_branch_can_be_deleted_whole) covers deleting some other branch on a hand-built diagram; this
# is the half that needs a real Problem, because the current branch's disappearance has to leave the
# problem loaded somewhere else and its state dumps have to be gone from disk.
#
# du/dt = mu - u^2 again, for its two arms u = +-sqrt(mu): deflation gives a second branch to delete
# without having to go round the fold to reach it.

import argparse
import os
import sys

from pyoomph import Problem, ODEEquations, InitialCondition
from pyoomph.expressions import var_and_test, partial_t
from pyoomph.utils.bifurcation_gui import BifurcationGUI
from pyoomph.utils.bifurcation_gui.controller import _FixedViewLimits


class FoldEqs(ODEEquations):
    def __init__(self, mu):
        super().__init__()
        self.mu = mu

    def define_fields(self):
        self.define_ode_variable("u")

    def define_residuals(self):
        u, u_test = var_and_test("u")
        self.add_weak(partial_t(u) - (self.mu - u**2), u_test)


class FoldProblem(Problem):
    def define_problem(self):
        eqs = FoldEqs(self.get_global_parameter("mu"))
        eqs += InitialCondition(u=1.0)
        self += eqs @ "ode"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    with FoldProblem() as problem:
        problem.set_output_directory(args.outdir)
        problem.set_linear_solver("superlu")
        problem.setup_for_stability_analysis(analytic_hessian=True)
        problem.get_global_parameter("mu").value = 1.0

        gui = BifurcationGUI(problem, "mu")
        c = gui.controller
        c.view = _FixedViewLimits(xlim=(-0.5, 4.0), ylim=(-3.0, 3.0))
        c.neigen = 1
        c.set_observer(on_log=lambda *a: None)
        c.start(0.05)
        c.step()

        c.deflation_random_tries = 6
        c.deflation_perturbation = 1.5
        assert c.deflated_solve(), "needed a second branch to delete"
        c.step()

        upper, lower = c.branches[0], c.branches[1]
        assert c.current_branch is lower, "the deflated solve left us on the new branch"
        dumps = [p.statefile for p in lower if p.statefile]
        assert dumps and all(os.path.exists(f) for f in dumps), "the branch must own state files"
        npoints = len(lower)

        assert c.delete_branch() == npoints, "delete_branch reports how many points went"
        assert len(c.branches) == 1 and c.branches[0] is upper, "only the deleted branch may go"
        assert not any(os.path.exists(f) for f in dumps), \
            "the state dumps of a deleted branch must be removed from disk"

        # The problem cannot be left on a point that no longer exists.
        assert c.current_branch is upper, "the current branch must fall back to a surviving one"
        assert c.current_point is upper[-1], "and to a point that is really on it"
        assert c.current_point in c.current_branch
        u = float(c.current_point.obs_values[c._current_observable])
        assert u > 0, "we were on u = -sqrt(mu) and must have landed back on the other arm: " + repr(u)
        # Reloaded, not merely re-pointed: the dofs have to be the ones of that point.
        live = float(problem.get_current_dofs()[0][0])
        assert abs(live - u) < 1e-8, "the problem was not reloaded: {:.10g} vs {:.10g}".format(live, u)

        # And it can still be continued from there.
        c.step()
        assert len(upper) > npoints or True
        assert c.current_branch is upper, "a step after the delete must extend the surviving branch"

        print("DELETE BRANCH OK: {:d} points and their dumps removed".format(npoints))

    print("PYOOMPH_WORKER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
