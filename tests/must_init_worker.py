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

# BifurcationGUI.must_init: does a stored diagram survive, is the setup skipped, and does the window
# open by itself at the right moment?
#
# One Problem per process (a second segfaults in the JIT loader), and phases against the SAME output
# directory, because surviving from one run to the next is the whole point.
#
#   --phase init     nothing stored: must_init must return True and the window must open at the end of
#                    the "with problem" block
#   --phase reload   a diagram is stored: must_init must return False, nothing may be solved, and the
#                    stored diagram must come back intact
#   --phase raises   the script fails after must_init: the window must NOT open, or it would bury the
#                    traceback the user needs
#
# The window is stood in for rather than shown: run() takes a few steps and saves, which is what a
# session that gets closed leaves behind.

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


def headless(nsteps: int):
    from pyoomph.utils.bifurcation_gui.tkapp import BifurcationTkApp

    def run(self):
        for _ in range(nsteps):
            self.controller.step()
        self.controller.save_all()
        session["opened"] = True
        session["points"] = sum(len(b) for b in self.controller.branches)
        session["mu"] = self.controller.problem.get_global_parameter("mu").value
        print("WINDOW_OPENED points={:d} mu={:.10g}".format(session["points"], session["mu"]))

    orig_init = BifurcationTkApp.__init__

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        self.root.withdraw()          # never on the user's screen, and never a mainloop

    BifurcationTkApp.run = run
    BifurcationTkApp.__init__ = init


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--phase", required=True, choices=["init", "reload", "raises", "nowith"])
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()
    nsteps = args.steps if args.phase == "init" else 0
    headless(nsteps)

    ran = {"body": False}
    failed = None
    must_init = None

    if args.phase == "nowith":
        # No "with problem", so release() is never called and the atexit route has to open the window.
        # Registered BEFORE must_init arms its own handler: atexit runs handlers last-registered-first,
        # so this one reports after the window has been and gone. Asserting here would be pointless -
        # an exception in an atexit handler does not fail the process - hence a marker to grep for.
        import atexit

        def report_after_the_window():
            print("NOWITH opened={:s} points={:d}".format(
                str(bool(session.get("opened"))), session.get("points", -1)))
            print("PYOOMPH_WORKER_DONE")
        atexit.register(report_after_the_window)

        problem = FoldProblem()
        problem.set_output_directory(args.outdir)
        gui = BifurcationGUI(problem, "mu")
        gui.neigen = 1
        gui.set_initial_view(-0.5, 1.5, -0.5, 1.5)
        if gui.must_init(-0.05):
            problem.get_global_parameter("mu").value = 1.0
            problem.get_global_parameter("b").value = 0.4
            problem.solve()
        assert not session.get("opened"), "the window must not open while the script is still running"
        return 0

    try:
        with FoldProblem() as problem:
            problem.set_output_directory(args.outdir)
            gui = BifurcationGUI(problem, "mu")
            gui.neigen = 1
            gui.set_initial_view(-0.5, 1.5, -0.5, 1.5)

            # The API under test. The window opens when this "with problem" block ends, so nothing
            # below the block can inspect the problem - the checks read the recorded session instead.
            must_init = gui.must_init(-0.05)
            if must_init:
                ran["body"] = True
                problem.get_global_parameter("mu").value = 1.0
                problem.get_global_parameter("b").value = 0.4
                problem.solve()

            assert not session.get("opened"), "the window must not open before the block ends"
            if args.phase == "raises":
                raise RuntimeError("deliberate failure after must_init")
    except RuntimeError as e:
        if args.phase != "raises" or "deliberate failure" not in str(e):
            raise
        failed = e

    print("phase={:s} must_init={:s} body_ran={:s} opened={:s}".format(
        args.phase, str(must_init), str(ran["body"]), str(bool(session.get("opened")))))

    if args.phase == "raises":
        assert failed is not None, "the deliberate failure did not propagate"
        assert not session.get("opened"), "the window opened on a script that was raising"
    elif args.phase == "init":
        assert must_init, "an empty output directory must ask for initialisation"
        assert ran["body"], "the setup block must run when there is nothing to load"
        assert session.get("opened"), "the window must open when the 'with problem' block ends"
        assert session["points"] == nsteps + 1, \
            "expected {:d} points, got {:d}".format(nsteps + 1, session["points"])
        assert os.path.isfile(os.path.join(args.outdir, "_bifurcation_gui_data", "state.json")), \
            "the diagram was not saved"
    else:
        # This is what breaks without must_init: initialising under the default "delete" runmode
        # removes state.json, so the diagram is silently lost and this run would ask to initialise
        # again and come back with a single point.
        assert not must_init, "a stored diagram must be found - did initialise() delete it?"
        assert not ran["body"], "the setup block must be skipped when a diagram was loaded"
        assert session.get("opened"), "the window must open when the 'with problem' block ends"
        assert session["points"] == args.steps + 1, \
            "the diagram came back with {:d} points".format(session["points"])
        assert abs(session["mu"] - float(os.environ["PYOOMPH_EXPECT_MU"])) < 1e-9, \
            "the loaded state is not where the first run ended: mu = {:.10g}".format(session["mu"])

    print("PYOOMPH_WORKER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
