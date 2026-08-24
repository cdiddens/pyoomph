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

# The Deflation tab's two commands, driven through the controller (no window), against a problem
# whose complete solution set is known:
#
#   du/dt = mu - u^2     ->   u = +-sqrt(mu),  i.e. EXACTLY TWO solutions for every mu > 0
#
# That is what makes the assertions worth making. Arclength continuation started on u = +sqrt(mu)
# never reaches u = -sqrt(mu) without going round the fold at mu = 0; deflation has to find it while
# standing still. And because there are only two, a second deflated solve has to find NOTHING -- a
# deflation that merely perturbs and re-converges would keep "finding" the branch it is on.
#
# Out of process because it needs its own Problem, and a second one in the same process segfaults in
# the JIT loader.

import argparse
import sys

import numpy

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


def u_of(controller, branch):
    """The u values recorded on one branch."""
    return [float(p.obs_values[controller._current_observable]) for p in branch]


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
        logged: list = []
        c.set_observer(on_log=lambda *a: logged.append(" ".join(str(x) for x in a)))
        c.start(0.05)

        start_u = float(c.current_point.obs_values[c._current_observable])
        assert abs(start_u - 1.0) < 1e-8, "the diagram must start on u = +sqrt(mu) = 1: " + repr(start_u)
        nbranches0 = len(c.branches)

        # ---------------------------------------------------- one deflated solve finds the other root
        c.deflation_random_tries = 6
        c.deflation_perturbation = 1.5
        assert c.deflated_solve(), "deflation must find u = -1, the only other solution at mu = 1"
        assert len(c.branches) == nbranches0 + 1, "a solution that is new must open a NEW branch"
        found = float(c.current_point.obs_values[c._current_observable])
        assert abs(found + 1.0) < 1e-6, "expected u = -1, got {:.10g}".format(found)
        assert abs(float(c.get_bifurcation_parameter().value) - 1.0) < 1e-12, \
            "a deflated solve must not move the parameter"
        assert c.deflation_known_count() == 2, "both roots are now known: " + str(c.deflation_known_count())
        # The point must carry a real spectrum: du/dt = mu - u^2 linearises to -2u, so u = -1 is
        # UNSTABLE (eigenvalue +2) while u = +1 is stable. Getting the sign right is what says the
        # eigensolve happened on the undeflated problem after the operator came off.
        assert abs(c.current_point.eig_value_Re - 2.0) < 1e-6, \
            "u = -1 has eigenvalue +2: " + repr(c.current_point.eig_value_Re)

        # ---------------------------------------------------- and there is nothing else to find
        assert not c.deflated_solve(), "mu = 1 has exactly two solutions, so the second search must fail"
        assert len(c.branches) == nbranches0 + 1, "a failed search must not open a branch"
        assert problem.get_deflation_operator() is None, "the operator must come off after a solve"

        # ---------------------------------------------------- forgetting restarts the search
        c.clear_deflation_known_solutions()
        assert c.deflation_known_count() == 0

        # ---------------------------------------------------- the scan stops at the edge of the plot
        # The view is mu in [-0.5, 4], so a scan of 20 steps of 0.5 from mu = 1 has to be cut off at 4
        # rather than march to 11. Like Multistep, which stops when the branch leaves the axes.
        c.deflated_scan_steps = 20
        c.deflated_scan_dparam = 0.5
        assert c.deflated_scan_is_clipped(), "20 steps of 0.5 from mu=1 leave the visible range"
        clipped = c.deflated_scan_values()
        # One value PAST the edge, the way multistep steps and then notices it has left the axes.
        assert clipped[0] == 1.0 and abs(clipped[-1] - 4.5) < 1e-12 and len(clipped) == 8, \
            "expected mu = 1 ... 4.5, got " + repr(clipped)
        c.deflated_scan_dparam = -0.5
        back = c.deflated_scan_values()
        assert abs(back[-1] + 1.0) < 1e-12, \
            "the other direction stops one past the other end: " + repr(back)
        # Zoomed somewhere else entirely, the view says nothing about where this scan should stop.
        c.view = _FixedViewLimits(xlim=(50.0, 60.0), ylim=(-3.0, 3.0))
        c.deflated_scan_dparam = 0.5
        assert len(c.deflated_scan_values()) == 21, "a scan starting outside the view is not clipped"
        assert not c.deflated_scan_is_clipped()
        # Sitting exactly ON the right edge, which is where ten continuation steps leave you: the
        # plotter grew the limits to include the last point. There is no room for a step inside the
        # view, so it bounds nothing and the whole scan runs -- clipping there reported a twenty-step
        # scan as a one-step one.
        mu_now = float(c.get_bifurcation_parameter().value)
        c.view = _FixedViewLimits(xlim=(mu_now - 1.0, mu_now), ylim=(-3.0, 3.0))
        assert len(c.deflated_scan_values()) == 21, "at the edge the view bounds nothing"
        assert not c.deflated_scan_is_clipped()
        c.view = _FixedViewLimits(xlim=(-0.5, 4.0), ylim=(-3.0, 3.0))

        # ---------------------------------------------------- deflated continuation
        c.deflated_scan_steps = 4
        c.deflated_scan_dparam = 0.5           # mu = 1, 1.5, ..., 3, all inside the view
        assert not c.deflated_scan_is_clipped(), "this one must fit, or the counts below mean nothing"
        assert c.deflated_scan_eigensolve is True, "the scan solves eigenproblems unless told not to"
        c.deflated_scan_eigensolve = False
        nb_before = len(c.branches)
        created = c.deflated_continuation()
        assert created >= 2, "both arms u = +-sqrt(mu) exist over the whole range: " + str(created)
        assert len(c.branches) == nb_before + created, "every scanned branch is a new branch"
        assert problem.get_deflation_operator() is None, "the operator must come off after a scan"

        # A scan ends on whatever its last attempt left, and its last attempt is by construction a
        # FAILED deflated solve - that is how the hunt for new solutions terminates - so the dofs are
        # a diverged state on no branch unless the scan puts the last recorded point back. Everything
        # downstream assumes current_point is what the problem holds; before this was fixed, a second
        # Deflated continuation began its opening solve from that garbage and went to inf/nan.
        live = float(problem.get_current_dofs()[0][0])
        mu_now = float(c.get_bifurcation_parameter().value)
        assert abs(live*live - mu_now) < 1e-6, \
            "the scan left the problem on a non-solution: u={:.10g} mu={:.10g}".format(live, mu_now)
        assert c.current_point is not None
        assert abs(live - float(c.current_point.obs_values[c._current_observable])) < 1e-8, \
            "the problem and current_point must be the same state"

        # ... and a second scan therefore just continues from there.
        again = c.deflated_continuation()
        assert again >= 2, "a second scan must work as well as the first: " + str(again)
        live = float(problem.get_current_dofs()[0][0])
        mu_now = float(c.get_bifurcation_parameter().value)
        assert abs(live*live - mu_now) < 1e-6, \
            "the second scan left a non-solution: u={:.10g} mu={:.10g}".format(live, mu_now)

        scanned = c.branches[nb_before:]
        for b in scanned:
            for p in b:
                mu = float(p.param_values["mu"])
                u = float(p.obs_values[c._current_observable])
                assert abs(u*u - mu) < 1e-6, "u^2 = mu on every branch: u={:.10g} mu={:.10g}".format(u, mu)
                # No eigensolve was asked for, so the point must say it has no spectrum rather than
                # carry whatever the problem still held from the last one.
                assert not p.eig_values, "the scan was asked not to eigensolve"
                assert numpy.isnan(p.eig_value_Re), "a point without a spectrum has a NaN eigenvalue"
        both = {round(float(p.obs_values[c._current_observable]), 6) > 0 for b in scanned for p in b}
        assert both == {True, False}, "the scan must find both arms, not one twice"

        # ---------------------------------------------------- with the eigensolve on, they come with one
        c.deflated_scan_eigensolve = True
        c.deflated_scan_steps = 2
        nb_eig = len(c.branches)
        c.deflated_continuation()
        for b in c.branches[nb_eig:]:
            for p in b:
                u = float(p.obs_values[c._current_observable])
                assert p.eig_values, "the scan was asked to eigensolve"
                assert abs(p.eig_value_Re + 2.0*u) < 1e-5, \
                    "d/du (mu - u^2) = -2u: u={:.6g} lambda={:.6g}".format(u, p.eig_value_Re)
        c.deflated_scan_eigensolve = False

        # ---------------------------------------------------- the spectra can be filled in afterwards
        target = scanned[0]
        assert c.compute_spectrum_for_branch(target) == len(target), \
            "every point of the branch needed a spectrum and could get one"
        for p in target:
            u = float(p.obs_values[c._current_observable])
            assert abs(p.eig_value_Re + 2.0*u) < 1e-5, \
                "d/du (mu - u^2) = -2u: u={:.6g} lambda={:.6g}".format(u, p.eig_value_Re)

        # ---------------------------------------------------- and it gives up when the branches leave
        # A range check on the PARAMETER alone would not catch this: mu stays deep inside [-0.5, 20]
        # while both arms u = +-sqrt(mu) leave |u| < 1.2 at mu > 1.44. Multistep's box test does catch
        # it, and so must this.
        c.view = _FixedViewLimits(xlim=(-0.5, 20.0), ylim=(-1.2, 1.2))
        c.load_pt(scanned[0][0])
        c.deflated_scan_steps = 20
        c.deflated_scan_dparam = 0.5
        assert not c.deflated_scan_is_clipped(), "mu itself stays on the plot the whole way"
        nb_before = len(c.branches)
        c.deflated_continuation()
        offplot = c.branches[nb_before:]
        reached = max((float(p.param_values["mu"]) for b in offplot for p in b), default=0.0)
        assert reached < 4.0, "the scan should have stopped once both arms left |u| < 1.2, not " \
                              "reached mu = {:g}".format(reached)
        assert any("left the visible axes" in m for m in logged), \
            "stopping because everything left the plot has to be said"
        c.view = _FixedViewLimits(xlim=(-0.5, 4.0), ylim=(-3.0, 3.0))

        # ---------------------------------------------------- abort stops it and cleans up
        c.request_abort()
        assert c.deflated_continuation() == 0 or True, "an aborted scan returns whatever it managed"
        assert not c.abort_requested, "the abort flag must be consumed by the loop that saw it"
        assert problem.get_deflation_operator() is None, \
            "an aborted scan must not leave the deflation operator installed"
        assert any("aborted" in m.lower() for m in logged), "the abort must be reported"

        print("DEFLATION GUI OK: found u=-1 by deflation, {:d} scanned branches".format(created))

    print("PYOOMPH_WORKER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
