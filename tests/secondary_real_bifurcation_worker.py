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

# Branch switching at a pitchfork or a transcritical whose classification is recomputed AFTER the
# fact - a point loaded from a diagram, or one that was never classified - on a branch that may
# already be unstable.
#
# Two separate defects met here, and both had to go before any of the four combinations below worked:
#
#  1. get_normal_form builds the normal form from eigenpair `eigenindex` of the last eigensolve, and
#     every caller left that at 0. A plain solve sorts by descending real part, so on an unstable
#     branch index 0 is the mode that went unstable EARLIER. See BifurcationController.
#     _located_eigenindex.
#
#  2. That eigensolve was taken at the controller's own shift, which defaults to 0 - and a located
#     REAL bifurcation has an exactly singular Jacobian, so a shift-invert at sigma = 0 factorises
#     it. Measured: the critical eigenvalue came back as +0.36787944 on the stable-branch pitchfork,
#     as -1.2e18 on another, and was simply missing from the spectrum on a third. This one is not
#     specific to an unstable branch at all - the plain pitchfork failed with "the left eigenvector
#     solve landed on the eigenvalue 1.3e-23 instead of the requested 0.368". See
#     BifurcationController._shift_for_an_eigensolve_at_a_bifurcation.
#
# The system is exact in closed form. x' = mu*x - x^3 has a pitchfork at mu = 0 with the two arms at
# x = +-sqrt(mu); x' = mu*x - x^2 a transcritical with the second branch at x = mu. The optional
# extra direction contributes a constant eigenvalue that does not bifurcate: +0.3 (real) or
# 0.3 +- 1.3i (a complex pair), which is what makes the branch already unstable, or -1 for the
# stable-branch control.

"""Does branch switching at a recomputed pitchfork/transcritical take the right mode?"""
import json
import sys

import numpy

from pyoomph import Problem, ODEEquations, InitialCondition
from pyoomph.expressions import var, testfunction, partial_t
from pyoomph.utils.bifurcation_gui import BifurcationGUI
from pyoomph.utils.bifurcation_gui.controller import _FixedViewLimits

_C, _W = 0.3, 1.3


class Eqs(ODEEquations):
    def __init__(self, kind, unstable):
        super().__init__()
        self.kind, self.unstable = kind, unstable

    def define_fields(self):
        self.define_ode_variable("x")
        if self.unstable == "hopf":
            self.define_ode_variable("z1", "z2")
        else:
            self.define_ode_variable("z")

    def define_residuals(self):
        mu = self.get_problem().mu
        x = var("x")
        f = mu*x - x**3 if self.kind == "pitchfork" else mu*x - x**2
        self.add_residual((partial_t(x) - f)*testfunction(x))
        if self.unstable == "hopf":
            z1, z2 = var(["z1", "z2"])
            r2 = z1**2 + z2**2
            self.add_residual((partial_t(z1) - (_C*z1 - _W*z2 - z1*r2))*testfunction(z1))
            self.add_residual((partial_t(z2) - (_W*z1 + _C*z2 - z2*r2))*testfunction(z2))
        else:
            # "real": an unstable real eigenvalue at +0.3. "stable": a stable one at -1, the control.
            c = _C if self.unstable == "real" else -1.0
            z = var("z")
            self.add_residual((partial_t(z) - (c*z - z**3))*testfunction(z))


class Prob(Problem):
    def __init__(self, kind, unstable):
        super().__init__()
        self.kind, self.unstable = kind, unstable
        self.mu = self.define_global_parameter(mu=-0.6)

    def define_problem(self):
        self += Eqs(self.kind, self.unstable) @ "nf"
        ic = {"x": 0.0}
        ic.update({"z1": 0.0, "z2": 0.0} if self.unstable == "hopf" else {"z": 0.0})
        self += InitialCondition(**ic) @ "nf"


# Run under a __main__ guard: see the note in tag_output_worker.py.
def main():
    outdir, kind, unstable = sys.argv[1], sys.argv[2], sys.argv[3]
    problem = Prob(kind, unstable)
    problem.set_output_directory(outdir)
    problem.quiet()
    problem.setup_for_stability_analysis(analytic_hessian=True)
    gui = BifurcationGUI(problem, "mu")
    c = gui.controller
    c.view = _FixedViewLimits(xlim=(-1, 1), ylim=(-1, 1))
    c.neigen = 4
    c.set_initial_observable("nf/x")
    res = {"kind": kind, "unstable": unstable}
    with problem:
        c.start(0.1)
        while problem.mu.value < -0.05:
            c.step()
        c.locate_bifurcation()
        cp = c.current_point
        res["mu"] = float(cp.param_value)
        res["spectrum"] = [[float(numpy.real(v)), float(numpy.imag(v))] for v in (cp.eig_values or [])]
        res["type_while_tracking"] = str((cp.bifurcation_info or {}).get("type"))

        # The case that was broken: no classification to hand, so it is recomputed with the tracker
        # off, from a full eigensolve.
        cp.bifurcation_info = None
        problem.deactivate_bifurcation_tracking()
        problem.solve_eigenproblem(c.neigen, c._shift_for_an_eigensolve_at_a_bifurcation())
        evs = [complex(v) for v in problem.get_last_eigenvalues()]
        res["fresh_spectrum"] = [[v.real, v.imag] for v in evs]
        idx = c._located_eigenindex(cp)
        res["located_eigenvalue"] = [evs[idx].real, evs[idx].imag]

        nbranches = len(c.branches)
        try:
            res["switched"] = bool(c.branch_switch())
        except Exception as e:
            res["switched"] = False
            res["error"] = str(e)
        res["type_after"] = str((cp.bifurcation_info or {}).get("type"))
        res["new_branches"] = len(c.branches) - nbranches
        res["landed_mu"] = float(problem.mu.value)
        res["landed_x"] = float(problem.get_current_dofs()[0][0])
    print("PYOOMPH_REALBIF_RESULT " + json.dumps(res))
    print("PYOOMPH_WORKER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
