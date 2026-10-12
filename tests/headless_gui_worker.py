#  @file
#  @author Christian Diddens
#  @author Duarte Rocha
#  @author Maxim de Wildt
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

# BifurcationGUI.start() under PYOOMPH_BIFURCATION_GUI_HEADLESS: does it take the requested number of
# continuation steps, store the diagram, and get there WITHOUT constructing the window?
#
# The point of the mode is that it runs where there is no display and no window system, so this worker
# makes constructing BifurcationTkApp an error instead of patching its run() method (which is what the
# other workers here do, on a machine that has a display). Importing tkinter at all is enough to fail
# on a headless runner, so a regression that moves the headless branch back below _create_app() has to
# be caught here rather than in CI.
#
#   --phase headless   the variable is set: start() must step, save and return, with no app built
#   --phase window     the variable is NOT set: start() must try to build the app, which proves the
#                      phase above is really taking a different route and not just always skipping it

import argparse
import os
import sys

from pyoomph import Problem, ODEEquations, InitialCondition
from pyoomph.expressions import var_and_test, partial_t
from pyoomph.utils.bifurcation_gui import BifurcationGUI

session = {}


class FoldEquations(ODEEquations):
    def __init__(self, mu, b):
        super().__init__()
        self.mu, self.b = mu, b

    def define_fields(self):
        self.define_ode_variable("u")

    def define_residuals(self):
        u, ut = var_and_test("u")
        self.add_weak(partial_t(u) - (self.mu - u**2 - self.b*u), ut)


class FoldProblem(Problem):
    def define_problem(self):
        eqs = FoldEquations(self.get_global_parameter("mu"), self.get_global_parameter("b"))
        eqs += InitialCondition(u=1.0)
        self += eqs @ "ode"


class WindowWasBuilt(Exception):
    pass


def forbid_window():
    """Make _create_app fail, so "no window" is proven rather than assumed."""
    def _create_app(self):
        session["app_attempted"] = True
        raise WindowWasBuilt("the headless mode must not construct the window")
    BifurcationGUI._create_app = _create_app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--phase", required=True, choices=["headless", "window"])
    args = ap.parse_args()

    forbid_window()

    problem = FoldProblem()
    problem.set_output_directory(args.outdir)
    gui = BifurcationGUI(problem, "mu")
    gui.neigen = 1
    gui.set_initial_view(-0.5, 1.5, -0.5, 1.5)
    if gui.must_init():
        problem.get_global_parameter("mu").value = 1.0
        problem.solve()

    if args.phase == "window":
        try:
            gui.start(-0.05)
        except WindowWasBuilt:
            print("WINDOW_ATTEMPTED")
            print("PYOOMPH_WORKER_DONE")
            return 0
        raise AssertionError("without the variable, start() has to go for the window")

    gui.start(-0.05)
    assert not session.get("app_attempted"), "the headless mode built a window"
    points = sum(len(b) for b in gui.controller.branches)
    mu = problem.get_global_parameter("mu").value
    # The step count comes from the environment, not from --steps: printing the argument would claim
    # a number nothing here actually used.
    print("HEADLESS_DONE steps={:s} points={:d} mu={:.10g}".format(
        os.environ.get("PYOOMPH_BIFURCATION_GUI_HEADLESS", "<unset>"), points, mu))
    print("PYOOMPH_WORKER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
