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

# Leaving an unstable branch by integrating in time, and the perturbation it starts from.
#
#   du/dt = mu*u - u^3 - eps/(1 - u^2)
#
# u = 0 (very nearly) is unstable at mu = 0.5, u = +-sqrt(mu) are the stable states it can fall to, and
# the 1/(1-u^2) term - a stand-in for the disjoining pressure of the thin-film problem this came from -
# is INFINITE at |u| = 1. The eigenvector has norm one, so perturbing by all of it lands exactly on that
# singularity, which is the case perturb_by_eigenfunction used to keep: an inf residual is neither
# larger nor smaller than the target it bisects towards, so both of its loops fell through and the full
# amplitude survived. The transient then died on its first step.

import argparse
import os
import sys

import numpy

from pyoomph import Problem, ODEEquations, InitialCondition
from pyoomph.expressions import var_and_test, partial_t
from pyoomph.utils.bifurcation_gui import BifurcationGUI
from pyoomph.utils.bifurcation_gui.controller import _FixedViewLimits

EPS = 1e-6


class Eqs(ODEEquations):
    def __init__(self, mu):
        super().__init__()
        self.mu = mu

    def define_fields(self):
        self.define_ode_variable("u")

    def define_residuals(self):
        u, u_test = var_and_test("u")
        self.add_weak(partial_t(u) - (self.mu*u - u**3 - EPS/(1 - u**2)), u_test)


class Prob(Problem):
    def __init__(self):
        super().__init__()
        self.step_times: list = []

    def define_problem(self):
        eqs = Eqs(self.get_global_parameter("mu"))
        eqs += InitialCondition(u=0.0)
        self += eqs @ "ode"

    def actions_after_transient_solve(self):
        # To measure the steps the run actually took: what makes an unstable mode grow is dt staying
        # below its growth time, and a fully implicit step far above it damps the mode instead.
        self.step_times.append(self.get_current_time(dimensional=False, as_float=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    with Prob() as problem:
        problem.set_output_directory(args.outdir)
        problem.set_linear_solver("superlu")
        problem.setup_for_stability_analysis(analytic_hessian=True)
        problem.get_global_parameter("mu").value = 0.5

        gui = BifurcationGUI(problem, "mu")
        gui.neigen = 1
        c = gui.controller
        c.view = _FixedViewLimits(xlim=(0.0, 1.0), ylim=(-1.0, 1.0))
        c.start(0.02)

        start = problem.get_ode("ode").get_value("u", as_float=True)
        rate = c.current_point.eig_value_Re
        assert rate > 0, "this branch has to be unstable to be worth leaving, got {:.4g}".format(rate)
        assert abs(start) < 1e-3, "expected to start on the unstable state, at u = {:.4g}".format(start)

        # The perturbation on its own first, since it is where the failure was.
        dt = problem.perturb_by_eigenfunction(dt=1.0/rate/20, eigenmode=0)
        residual = float(numpy.amax(numpy.absolute(problem.get_residuals())))
        moved = abs(problem.get_ode("ode").get_value("u", as_float=True) - start)
        print("PERTURBED by {:.4g}, residual {:.4g}, dt {:.4g}".format(moved, residual, float(dt)))
        assert numpy.isfinite(residual), "the perturbation left the domain the equations are defined on"
        assert residual < 0.2, "the perturbation was not scaled down to the target residual: {:.4g}".format(residual)
        assert 0 < moved < 1.0, "and it has to have moved without reaching the singularity: {:.4g}".format(moved)

        # Then the whole manoeuvre, from the branch again.
        c.load_pt(c.current_point)
        n_before = len(c.branches)
        problem.step_times = []
        c.transient_leave_branch(0)
        gaps = [b - a for a, b in zip(problem.step_times, problem.step_times[1:])]
        growth_time = 1.0/rate
        print("TOOK {:d} steps, longest {:.4g} against a growth time of {:.4g}, ending at {:.4g}".format(
            len(problem.step_times), max(gaps) if gaps else float("nan"), growth_time,
            problem.step_times[-1] if problem.step_times else float("nan")))
        # The cap applies WHILE the unstable mode is what is growing, which is what a step longer than
        # the growth time would damp away instead of amplifying. The controller watches the distance
        # from the branch and lets the step grow once that growth has stalled, so the cap is asserted
        # over the first two growth times - the interval over which the stall cannot yet be declared -
        # and not over the nonlinear approach to the new state afterwards.
        early = [b - a for a, b in zip(problem.step_times, problem.step_times[1:]) if b <= 2*growth_time]
        assert early and max(early) <= growth_time/5*1.001, \
            "the time step ran away past the growth time while the mode was growing: {:.4g}".format(max(early))
        # And it has to STOP once the solution has arrived. Integrating the full 100 growth times took
        # 602 steps here, all of them after the solution had reached the state it was going to.
        assert problem.step_times[-1] < 50*growth_time, \
            "the transient did not stop when the solution had settled: ran to {:.4g}".format(problem.step_times[-1])
        assert len(problem.step_times) < 100, \
            "far more steps than the departure needs: {:d}".format(len(problem.step_times))
        landed = problem.get_ode("ode").get_value("u", as_float=True)
        print("LEFT the branch, landing at u = {:+.6g} (expected +-{:.6g})".format(landed, numpy.sqrt(0.5)))
        assert len(c.branches) == n_before + 1, "leaving the branch has to open a new one"
        assert abs(abs(landed) - numpy.sqrt(0.5)) < 1e-3, \
            "did not settle on a stable state: u = {:.6g}".format(landed)
        assert c.current_point.eig_value_Re < 0, "and that state is a stable one"

    print("PYOOMPH_WORKER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
