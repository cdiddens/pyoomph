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

# A bifurcation on a branch that is ALREADY unstable must be classified from ITS OWN mode.
#
# get_normal_form reads the critical eigenpair out of the last eigensolve at `eigenindex`, and every
# caller left that at 0. While a tracker is installed that is right - the eigensolve holds the tracked
# pair and nothing else - but classifying AFTER THE FACT (a point loaded from a diagram, or a branch
# switch at one that was never classified) solves the whole spectrum, which comes back sorted by
# descending real part. Index 0 is then whatever went unstable EARLIER.
#
# Measured on the system below before the fix: the Hopf at omega = 1.7 was classified from the real
# unstable eigenvalue at +0.3 and came back "pitchfork", so switching onto its orbit refused it with
# "Only a Hopf bifurcation sheds a periodic orbit; this one is pitchfork". Saved and reloaded, that
# wrong classification then produced
#
#   The eigenvector for the saved classification is not real: ... A real bifurcation needs a real
#   null vector; a complex one is a Hopf.
#
# because the eigenvector recovered for it is the Hopf pair the point actually belongs to.
#
# Everything is known in closed form here: the trivial state x = y = z = 0 exists for every mu, the
# Hopf sits at mu = A with frequency W (period 2*pi/W), and the z direction contributes the constant
# real eigenvalue +C, which is what makes the branch unstable before the Hopf is reached.

"""Is a Hopf on an already-unstable branch classified as a Hopf, and can its orbit be started?"""
import json
import sys

import numpy

from pyoomph import Problem, ODEEquations, InitialCondition
from pyoomph.expressions import var, testfunction, partial_t
from pyoomph.utils.bifurcation_gui import BifurcationGUI
from pyoomph.utils.bifurcation_gui.controller import _FixedViewLimits
from pyoomph.utils.bifurcation_gui.model import ORBIT_T_KEY

_A, _W, _C = 1.0, 1.7, 0.3


class Eqs(ODEEquations):
    def define_fields(self):
        self.define_ode_variable("x", "y", "z")

    def define_residuals(self):
        mu = self.get_problem().mu
        x, y, z = var(["x", "y", "z"])
        r2 = x**2 + y**2
        self.add_residual((partial_t(x) - ((mu-_A)*x - _W*y - x*r2))*testfunction(x))
        self.add_residual((partial_t(y) - (_W*x + (mu-_A)*y - y*r2))*testfunction(y))
        self.add_residual((partial_t(z) - (_C*z - z**3))*testfunction(z))


class Prob(Problem):
    def __init__(self):
        super().__init__()
        self.mu = self.define_global_parameter(mu=0.2)

    def define_problem(self):
        self += Eqs() @ "nf"
        self += InitialCondition(x=0, y=0, z=0) @ "nf"


# Run under a __main__ guard: see the note in tag_output_worker.py.
def main():
    problem = Prob()
    problem.set_output_directory(sys.argv[1])
    problem.quiet()
    problem.setup_for_stability_analysis(analytic_hessian=True)
    gui = BifurcationGUI(problem, "mu")
    c = gui.controller
    c.view = _FixedViewLimits(xlim=(0, 2), ylim=(-1.5, 1.5))
    c.neigen = 4
    c.set_initial_observable("nf/x")
    res = {}
    with problem:
        c.start(0.05)
        while problem.mu.value < _A - 0.02:
            c.step()
        c.locate_bifurcation()
        cp = c.current_point
        res["mu_hopf"] = float(cp.param_value)
        res["spectrum"] = [[float(numpy.real(v)), float(numpy.imag(v))] for v in (cp.eig_values or [])]
        res["type_while_tracking"] = str((cp.bifurcation_info or {}).get("type"))

        # The case that was broken: no saved classification, so it is recomputed here - with the
        # tracker off, from the whole spectrum.
        cp.bifurcation_info = None
        problem.deactivate_bifurcation_tracking()
        res["located_eigenindex"] = None
        problem.solve_eigenproblem(c.neigen, c.shift)
        evs = [complex(v) for v in problem.get_last_eigenvalues()]
        res["fresh_spectrum"] = [[v.real, v.imag] for v in evs]
        idx = c._located_eigenindex(cp)
        res["located_eigenindex"] = int(idx)
        res["located_eigenvalue"] = [evs[idx].real, evs[idx].imag]

        try:
            c.switch_to_orbit()
            res["switched"] = True
            res["type_after"] = str((cp.bifurcation_info or {}).get("type"))
            res["omega_after"] = float((cp.bifurcation_info or {}).get("omega", float("nan")))
            res["kind"] = c.current_branch.kind
            res["T"] = float(c.current_point.obs_values[ORBIT_T_KEY])
        except Exception as e:
            res["switched"] = False
            res["error"] = str(e)
        res["T_exact"] = float(2*numpy.pi/_W)
    print("PYOOMPH_SECHOPF_RESULT " + json.dumps(res))
    print("PYOOMPH_WORKER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
