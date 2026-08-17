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
            rhs = self.mu * u - u**2 if kind == "transcritical" else self.mu * u - u**3
            self.add_weak(partial_t(u) - rhs, u_test)

    class Prob(Problem):
        def define_problem(self):
            eqs = Eqs(self.get_global_parameter("mu"))
            eqs += InitialCondition(u=0.0)
            self += eqs @ "ode"

    return Prob()


def other_branch(kind: str, mu: float):
    """|u| on the branch that is NOT u = 0, or None where it does not exist."""
    if kind == "transcritical":
        return abs(mu)
    return numpy.sqrt(mu) if mu > 0 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["transcritical", "pitchfork"])
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    with build(args.kind) as problem:
        problem.set_output_directory(args.outdir)
        problem.set_linear_solver("superlu")
        problem.setup_for_stability_analysis(analytic_hessian=True)
        problem.get_global_parameter("mu").value = -0.2

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
