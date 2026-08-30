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

# The script that test_state_file_restart.py's --runmode continue tests interrupt and resume. It is a
# separate process on purpose: "continue" is decided by the command line and by the state files under
# the output directory, so it cannot be exercised from inside a single pytest process.
#
# One driven scalar ODE, integrated by the same run() machinery a PDE would use. No mesh is needed to
# see the effect - what is being compared is the sequence of time steps and the BDF2 history that goes
# with it - and an ODE keeps the run cheap enough to do six of them per test.

import os
import sys

from pyoomph import *
from pyoomph.expressions import *

# Interrupt after this nondimensional time, imitating a killed run. Reported through SystemExit rather
# than an exception so the state files stay exactly as an interrupted run would have left them.
ABORT_AT = float(os.environ.get("PYOOMPH_ABORT_AT", "-1"))
VARIANT = os.environ["PYOOMPH_VARIANT"]


class DrivenOde(ODEEquations):
    def define_fields(self):
        self.define_ode_variable("u")

    def define_residuals(self):
        u, v = var_and_test("u")
        # Driven and nonlinear, so that BDF1 and BDF2 really do disagree and a step sequence that
        # differs from the uninterrupted one shows up in the answer.
        self.add_residual(weak(partial_t(u) + u ** 3 - 0.3 * u - sin(2 * var("time")), v))


class ContinueProblem(Problem):
    def define_problem(self):
        self += (DrivenOde() + InitialCondition(u=1)) @ "osc"

    def actions_after_transient_solve(self):
        if ABORT_AT > 0 and self.get_current_time(as_float=True, dimensional=False) >= ABORT_AT:
            raise SystemExit(7)


def report(problem, tag):
    tp = problem.timestepper.time_pt()
    print("%s u=%.17g t=%.17g steps=%d dts=%s" % (
        tag, problem.get_ode("osc").get_value("u"),
        problem.get_current_time(as_float=True, dimensional=False),
        problem.timestepper.get_num_unsteady_steps_done(),
        ",".join("%.17g" % tp.dt(i) for i in range(tp.ndt()))))


with ContinueProblem() as problem:
    if VARIANT == "fixed":
        # dt=0.037 does not divide outstep=0.1, so every third step is clamped short to land on an
        # output time - which is exactly where the state file is written.
        problem.run(1.0, timestep=0.037, outstep=0.1)
    elif VARIANT == "tempadapt":
        problem.run(1.0, startstep=0.01, temporal_error=1e-4, outstep=0.1, maxstep=0.09)
    elif VARIANT == "numouts":
        problem.run(1.0, timestep=0.037, numouts=7)
    elif VARIANT == "tworuns":
        # Two run statements with different steps. Interrupted between them, the second one must start
        # with its own 0.023 rather than inheriting the first one's 0.037.
        problem.run(0.4, timestep=0.037, outstep=0.1)
        problem.run(1.0, timestep=0.023, outstep=0.1)
    else:
        raise RuntimeError("unknown variant " + VARIANT)
    report(problem, "FINAL")
print("PYOOMPH_WORKER_DONE")
