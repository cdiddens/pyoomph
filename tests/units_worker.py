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

# The bifurcation GUI on a problem that carries physical units.
#
#     du/dt = -u/tau + mu*u^2/(u_scale*tau)      tau = 2 s, u scaled in mm
#     dw/dt = -(w - 0.7*u_scale)/tau             a nonzero steady state, 0.7 mm
#
# At u = 0 the eigenvalue is exactly -1/tau = -0.5 1/s, whatever nondimensionalisation is used. The
# temporal scale is deliberately 10 s and NOT 1 s, so a value reported nondimensionally (-5) cannot be
# confused with one reported in 1/s (-0.5).
#
# Three things used to be wrong and are pinned here:
#   * the GUI could not START on such a problem: float() of a spatial observable throws while it still
#     carries a unit, and the integration measure dx gives EVERY integral observable one,
#   * eigenvalues were nondimensional (-5 instead of -0.5 1/s),
#   * ODE observables came out in SI base units (0.0007) with nothing saying so, while spatial ones had
#     to be hand-nondimensionalised - two different conventions, neither labelled.

import argparse
import os
import sys

import numpy

from pyoomph import Problem, ODEEquations, Equations, InitialCondition
from pyoomph.expressions import var_and_test, var, partial_t
from pyoomph.expressions.units import second, meter, milli
from pyoomph.equations.generic import IntegralObservables
from pyoomph.meshes.simplemeshes import LineMesh
from pyoomph.utils.bifurcation_gui import BifurcationGUI
from pyoomph.utils.bifurcation_gui.controller import _FixedViewLimits

TAU = 2*second
U_SCALE = 1*milli*meter


class Decay(ODEEquations):
    def __init__(self, mu):
        super().__init__()
        self.mu = mu

    def define_fields(self):
        # The test scale makes the weak form dimensionless: the residual carries [u]/[time].
        self.define_ode_variable("u", "w", scale=U_SCALE, testscale=TAU/U_SCALE)

    def define_residuals(self):
        u, ut = var_and_test("u")
        self.add_weak(partial_t(u) + u/TAU - self.mu*u*u/(U_SCALE*TAU), ut)
        w, wt = var_and_test("w")
        self.add_weak(partial_t(w) + (w - 0.7*U_SCALE)/TAU, wt)


class Field(Equations):
    def define_fields(self):
        self.define_scalar_field("h", "C2", scale=U_SCALE, testscale=TAU/U_SCALE)

    def define_residuals(self):
        h, v = var_and_test("h")
        self.add_weak(partial_t(h) + h/TAU, v)


class Prob(Problem):
    def define_problem(self):
        self.set_scaling(temporal=10*second, spatial=1*meter, u=U_SCALE, w=U_SCALE, h=U_SCALE)
        eqs = Decay(self.get_global_parameter("mu"))
        eqs += InitialCondition(u=0.5*U_SCALE, w=0.7*U_SCALE)
        self += eqs @ "ode"
        self.add_mesh(LineMesh(N=10, size=1*meter))
        feqs = Field()
        feqs += InitialCondition(h=0.5*U_SCALE)
        # h_avg is a LENGTH, and _area carries the measure's metres: the natural way to write it, and
        # exactly what used to stop the GUI from starting.
        feqs += IntegralObservables(_area=1, _h_int=var("h"))
        feqs += IntegralObservables(h_avg=lambda _area, _h_int: _h_int/_area)
        self += feqs @ "domain"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    with Prob() as problem:
        problem.set_output_directory(args.outdir)
        problem.set_linear_solver("superlu")
        problem.setup_for_stability_analysis(analytic_hessian=True)
        problem.get_global_parameter("mu").value = 0.0
        problem.quiet()

        gui = BifurcationGUI(problem, "mu")
        gui.neigen = 2
        c = gui.controller
        c.view = _FixedViewLimits(xlim=(-2, 2), ylim=(-2, 2))
        c.start(0.02)          # used to raise: "Cannot convert ... it still carries the physical unit"

        pt = c.current_point
        assert pt is not None
        print("EIGEN {:.6g} {:s}".format(pt.eig_value_Re, c.eigen_unit))
        assert c.eigen_unit == "1/s", "eigenvalues should be rates in 1/s, got '" + c.eigen_unit + "'"
        assert abs(pt.eig_value_Re - (-0.5)) < 1e-6, \
            "eigenvalue should be -1/tau = -0.5 1/s, got {:.6g} (nondimensional would be -5)".format(
                pt.eig_value_Re)

        print("OBS " + " ".join("{:s}={:.6g}[{:s}]".format(k, v, c.observable_unit(k))
                                for k, v in sorted(pt.obs_values.items())))
        assert c.observable_unit("ode/w") == "mm", \
            "expected mm, got '" + c.observable_unit("ode/w") + "'"
        assert abs(pt.obs_values["ode/w"] - 0.7) < 1e-9, \
            "w should read 0.7 mm, got {:.6g} (SI metres would be 7e-4)".format(pt.obs_values["ode/w"])
        assert "domain/h_avg" in pt.obs_values, "the dimensional spatial observable must be usable"
        assert c.observable_unit("domain/h_avg") == "mm", c.observable_unit("domain/h_avg")

        label = c.axis_label(("observable", "ode/w"))
        print("LABEL " + label)
        assert label == "ode/w [mm]", label
        assert c.axis_label(("parameter", "mu")) == "mu", "a global parameter is a plain number"

        c.output_all_observables = True
        c.output_curves()
        odir = problem.get_output_directory(os.path.join(c.data_subdir, "output"))
        written = sorted(os.path.join(dp, f) for dp, _d, fs in os.walk(odir) for f in fs if f.endswith(".txt"))
        assert written, "nothing exported"
        with open(written[0]) as fh:
            head = fh.readline().strip()
        print("HEADER " + head)
        assert "[mm]" in head and "ReEigen [1/s]" in head, head

        # The units have to survive a save/load, since the stored numbers are IN them.
        c.save_all()
        c.load_all()
        assert c.observable_unit("ode/w") == "mm", "the unit did not survive a reload"

    print("PYOOMPH_WORKER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
