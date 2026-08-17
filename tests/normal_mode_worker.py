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

# Normal-mode eigenvalues in the bifurcation GUI, end to end.
#
# Axisymmetric diffusion, -laplace(u) = mu*u, with the azimuthal stability code generated. Under the
# azimuthal expansion the m^2/r^2 term makes the operator more positive, so the eigenvalues fall with
# |m| - the mode dependence is checkable rather than merely present (measured -16.8, -23.6, -35.3 for
# m = 0, 1, 2).
#
# Out of process because it needs its own Problem, and the mesh is deliberately tiny: generating and
# compiling the azimuthal code is what costs time here, not solving.

import argparse
import sys

import numpy

from pyoomph import Problem, Equations, InitialCondition, DirichletBC
from pyoomph.expressions import var_and_test, var, grad, partial_t
from pyoomph.equations.generic import IntegralObservables
from pyoomph.meshes.simplemeshes import RectangularQuadMesh
from pyoomph.utils.bifurcation_gui import BifurcationGUI
from pyoomph.utils.bifurcation_gui.controller import _FixedViewLimits


class Diffusion(Equations):
    def __init__(self, mu):
        super().__init__()
        self.mu = mu

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_weak(partial_t(u), v)
        self.add_weak(grad(u), grad(v))
        self.add_weak(-self.mu*u, v)


class Prob(Problem):
    def define_problem(self):
        self.set_coordinate_system("axisymmetric")
        self.add_mesh(RectangularQuadMesh(size=[1.0, 1.0], N=[4, 4]))
        eqs = Diffusion(self.get_global_parameter("mu"))
        eqs += InitialCondition(u=0)
        for b in ("left", "right", "top", "bottom"):
            eqs += DirichletBC(u=0) @ b
        eqs += IntegralObservables(_vol=1, _u_int=var("u"))
        eqs += IntegralObservables(u_avg=lambda _vol, _u_int: _u_int/_vol)
        self += eqs @ "domain"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    with Prob() as problem:
        problem.set_output_directory(args.outdir)
        problem.set_linear_solver("superlu")
        problem.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
        problem.get_global_parameter("mu").value = 1.0
        problem.quiet()

        gui = BifurcationGUI(problem, "mu")
        gui.neigen = 2
        c = gui.controller
        c.view = _FixedViewLimits(xlim=(-5.0, 40.0), ylim=(-5.0, 5.0))
        c.start(0.5)

        assert c.normal_mode_kind() == "m", repr(c.normal_mode_kind())
        point = c.current_point
        assert point is not None
        # Without a mode list nothing may be recorded as a mode: the problem itself leaves
        # get_last_eigenmodes_m() at None then, and that is the "base state only" sentinel.
        assert point.eig_modes is None, point.eig_modes

        # ---- the scan
        c.normal_modes = [1, 2]
        assert c.compute_spectrum(point, force=True)
        print("MODES " + " ".join("{:g}".format(m) for m in point.eig_modes))
        assert len(point.eig_modes) == len(point.eig_values), "one mode per eigenvalue"
        assert set(int(m) for m in point.eig_modes) == {0, 1, 2}, point.eig_modes
        # The physics: the leading eigenvalue of each mode falls as |m| rises.
        leads = [max(float(numpy.real(v)) for v in point.eigenvalues_of_mode(float(m))) for m in (0, 1, 2)]
        print("LEADS " + " ".join("{:.4g}".format(x) for x in leads))
        assert leads[0] > leads[1] > leads[2], leads

        # ---- per-mode plot axes
        axes = [n for n in c.available_observables if n.startswith("eigen/")]
        assert "eigen/max Re  [m=1]" in axes, axes
        assert c.axis_label(("observable", "eigen/max Re  [m=1]")).startswith("eigen/max Re  [m=1]")
        assert abs(point.obs_values["eigen/max Re  [m=1]"] - leads[1]) < 1e-9

        # ---- the stability toggle. An unstable eigenvalue placed on m=1 must count as unstable only
        # while the modes are being counted; the stored count is kept in step, as the recording code
        # does, since stability_indicator reads it rather than recounting.
        point.eig_values = [0.5+0j] + list(point.eig_values)
        point.eig_modes = [1.0] + list(point.eig_modes)
        point.unstable_count = point.measured_unstable_count()
        assert point.measured_unstable_count(True) == 1
        assert point.measured_unstable_count(False) == 0
        assert point.stability_indicator(include_modes=True) > 0
        assert point.stability_indicator(include_modes=False) < 0

        # ---- staleness: raising the eigenvalue count is what used to recompute nothing
        assert not c.spectrum_is_stale(point)
        gui.neigen = 4
        assert c.spectrum_is_stale(point), "raising neigen must mark the point stale"
        n_before = len(point.eig_values)
        done = c.compute_spectrum_for_branch()      # NOT forced: staleness alone must select it
        print("REFILL {:d} points, n {:d} -> {:d}".format(done, n_before, len(point.eig_values)))
        assert done >= 1, "the ordinary back-fill has to pick up a stale point"
        assert len(point.eig_values) == 4*3, "4 eigenvalues on each of 3 modes"
        assert not c.spectrum_is_stale(point)

        # ---- persistence
        c.save_all()
        c.load_all()
        reloaded = c.branches[0][0]
        assert reloaded.eig_modes is not None
        assert set(int(m) for m in reloaded.eig_modes) == {0, 1, 2}, reloaded.eig_modes
        assert not c.spectrum_is_stale(reloaded), "a reloaded point must not read as stale"
        print("RELOADED {:d} eigenvalues, {:d} modes".format(
            len(reloaded.eig_values), len(set(reloaded.eig_modes))))

    print("PYOOMPH_WORKER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
