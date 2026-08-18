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

# Branch switching, checked against branches known in closed form. Out of process and one bifurcation
# type per invocation, because each needs its own Problem and a second one in the same process
# segfaults in the JIT loader.
#
#   transcritical   du/dt = mu*u - u^2   branches u = 0 and u = mu, crossing at mu = 0
#   pitchfork       du/dt = mu*u - u^3   branches u = 0 and u = +-sqrt(mu), meeting at mu = 0
#   fold            du/dt = mu - u^2     one branch, u = +-sqrt(mu), turning around at mu = 0
#
# The fold is here to pin the OTHER half of the classification: the fold/branch-point decision is made
# on how far dR/dparameter lies out of the range of the Jacobian, measured as an angle so that it does
# not depend on the arbitrary scaling of the left null vector. A fold has to keep coming out as a fold,
# and has to refuse to switch, or the measure has merely been loosened until everything is a branch
# point.
#
# The trivial branch u = 0 exists for every mu in both, which is what makes this a real test: a switch
# that does not work does not fail loudly, it quietly stays on u = 0. Both the switch AND the steps
# after it are therefore checked, since the switch used to land correctly and then be undone by the
# first continuation step.

import argparse
import sys

import numpy

from pyoomph import Problem, ODEEquations, InitialCondition
from pyoomph.expressions import var_and_test, partial_t
from pyoomph.utils.bifurcation_gui import BifurcationGUI
from pyoomph.utils.bifurcation_gui.controller import _FixedViewLimits


def build(kind: str):
    class Eqs(ODEEquations):
        def __init__(self, mu):
            super().__init__()
            self.mu = mu

        def define_fields(self):
            self.define_ode_variable("u")

        def define_residuals(self):
            u, u_test = var_and_test("u")
            rhs = {"transcritical": self.mu*u - u**2,
                   "pitchfork": self.mu*u - u**3,
                   "fold": self.mu - u**2}[kind]
            self.add_weak(partial_t(u) - rhs, u_test)

    class Prob(Problem):
        def define_problem(self):
            eqs = Eqs(self.get_global_parameter("mu"))
            # The fold's branch is u = +-sqrt(mu), which does not pass through u = 0 at all, so it
            # needs a starting point ON the branch rather than the trivial one the others use.
            eqs += InitialCondition(u=1.0 if kind == "fold" else 0.0)
            self += eqs @ "ode"

    return Prob()


def other_branch(kind: str, mu: float):
    """|u| on the branch that is NOT u = 0, or None where it does not exist."""
    if kind == "transcritical":
        return abs(mu)
    return numpy.sqrt(mu) if mu > 0 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["transcritical", "pitchfork", "fold"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--api", default="gui", choices=["gui", "problem"],
                    help="gui: through BifurcationGUI; problem: Problem.switch_branch on its own")
    args = ap.parse_args()

    with build(args.kind) as problem:
        problem.set_output_directory(args.outdir)
        problem.set_linear_solver("superlu")
        problem.setup_for_stability_analysis(analytic_hessian=True)
        problem.get_global_parameter("mu").value = 1.0 if args.kind == "fold" else -0.2

        if args.kind == "fold":
            problem.solve()
            problem.solve_eigenproblem(1)
            problem.activate_bifurcation_tracking("mu")
            problem.solve()
            assert abs(problem.get_global_parameter("mu").value) < 1e-7, "the fold is at mu = 0"
            nf = problem.classify_bifurcation("mu")
            assert nf.get("type") == "fold", "expected a fold, got " + str(nf.get("type"))
            # dR/dmu IS the null direction here, so the measure is at its maximum. A branch point of
            # the same problem size scores 1e-5; the decision is not close.
            assert nf["a_rel"] > 0.5, "a fold must be nowhere near the branch-point end: {:.3g}".format(nf["a_rel"])
            try:
                problem.switch_branch("mu", normal_form=nf, quiet=True)
            except RuntimeError as e:
                assert "fold" in str(e), str(e)
            else:
                raise AssertionError("switching branches at a fold has to be refused")
            print("kind=fold a_rel={:.3g}".format(nf["a_rel"]))
            print("PYOOMPH_WORKER_DONE")
            return 0

        if args.api == "problem":
            # No GUI anywhere: this is the path a plain script takes.
            problem.solve()
            problem.solve_eigenproblem(1)
            problem.activate_bifurcation_tracking("mu")
            problem.solve()
            assert abs(problem.get_global_parameter("mu").value) < 1e-7, "the bifurcation is at mu = 0"
            nf = problem.classify_bifurcation("mu")
            assert nf.get("type") == args.kind, "expected " + args.kind + ", got " + str(nf.get("type"))
            # Saved here so the pitchfork's second arm can be reached from the bifurcation later.
            import io
            at_bifurcation = io.BytesIO()
            problem.save_state(at_bifurcation, quiet=True)

            ds = problem.switch_branch("mu", normal_form=nf, quiet=True)
            assert ds is not None, "switch_branch could not reach the other branch"
            assert problem.get_bifurcation_tracking_mode() == "", "tracking must be off afterwards"
            worst = 0.0
            for _ in range(4):
                ds = problem.arclength_continuation("mu", ds)
                mu = problem.get_global_parameter("mu").value
                u = problem.get_ode("ode").get_value("u", as_float=True)
                expect = other_branch(args.kind, mu)
                if expect is None or abs(mu) < 1e-4:
                    continue
                assert abs(u) > 1e-6, "fell back onto the trivial branch at mu = {:.4g}".format(mu)
                worst = max(worst, abs(abs(u) - expect))
            print("kind={:s} api=problem max_error={:.3e}".format(args.kind, worst))
            assert worst < 1e-6, "not on the analytic other branch: {:.3e}".format(worst)

            if args.kind == "pitchfork":
                # Both arms, u = +sqrt(mu) and u = -sqrt(mu), have to be reachable. They sit on the
                # SAME side of mu, so the direction argument is the only thing that tells them apart -
                # and it did not, because the predictor took the absolute value of its argument and
                # returned the very same point for both. Done last, from the saved state: it does not
                # need bifurcation tracking back (switch_branch only wants the normal form and the
                # solution at the bifurcation), and re-activating it here would have to re-solve an
                # eigenproblem whose Jacobian is exactly singular.
                signs = []
                for direction in (1, -1):
                    at_bifurcation.seek(0)
                    problem.load_state(at_bifurcation, quiet=True)
                    assert problem.switch_branch("mu", normal_form=nf, direction=direction,
                                                 quiet=True) is not None, \
                        "direction {:+d} could not reach an arm of the pitchfork".format(direction)
                    signs.append(numpy.sign(problem.get_ode("ode").get_value("u", as_float=True)))
                assert signs[0] == -signs[1] and 0 not in signs, \
                    "the two directions must reach the two arms of the pitchfork, got " + str(signs)
                print("pitchfork arms reached: {:+g} and {:+g}".format(*signs))

            print("PYOOMPH_WORKER_DONE")
            return 0

        gui = BifurcationGUI(problem, "mu")
        gui.neigen = 1
        gui.classify_bifurcations = True
        c = gui.controller
        c.view = _FixedViewLimits(xlim=(-2.0, 4.0), ylim=(-2.0, 2.0))
        c.start(0.05)
        for _ in range(2):
            c.step()

        c.locate_bifurcation()
        cp = c.current_point
        assert cp is not None
        assert abs(cp.param_value) < 1e-7, "the bifurcation is at mu = 0"
        assert cp.bifurcation_info is not None, "classify_bifurcations must name it"
        assert cp.bifurcation_info.get("type") == args.kind, \
            "expected " + args.kind + ", got " + str(cp.bifurcation_info.get("type"))

        n_before = len(c.branches)
        assert c.branch_switch(), "branch_switch reported failure"
        assert len(c.branches) == n_before + 1, "a new branch must be opened"
        new = c.branches[-1]

        for _ in range(4):
            c.step()

        errors = []
        for p in new:
            expect = other_branch(args.kind, p.param_value)
            if expect is None or abs(p.param_value) < 1e-4:
                continue
            errors.append(abs(abs(p.obs_values["ode/u"]) - expect))
            assert abs(p.obs_values["ode/u"]) > 1e-6, \
                "fell back onto the trivial branch at mu = {:.4g}".format(p.param_value)
        assert errors, "the new branch has no point away from the bifurcation"
        worst = max(errors)
        print("kind={:s} points={:d} max_error={:.3e}".format(args.kind, len(new), worst))
        assert worst < 1e-6, "not on the analytic other branch: max error {:.3e}".format(worst)

    print("PYOOMPH_WORKER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
