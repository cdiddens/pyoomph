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

# ExtremumObservables as bifurcation-diagram axes: the minimum and the maximum of each one, and where
# each of them sits.
#
# The field is chosen so that every answer is known in closed form AND unique:
#
#     h -> sin(2*pi*(x/Lx - shift*mu)) * sin(pi*y/Ly)      on [0,Lx] x [0,Ly]
#
#     max = +1 at (Lx*(0.25 + shift*mu), Ly/2),  min = -1 at (Lx*(0.75 + shift*mu), Ly/2)
#
# Uniqueness is the whole difficulty in testing a POSITION. The obvious cos(2pi*x/Lx)*cos(2pi*y/Ly) has
# five maxima of equal height (four corners and the centre), so it reports a position other than the one
# written down and there is no bug to find - the test would simply be wrong. The phase also moves with
# mu, so the position is a curve along the branch rather than a constant that any indexing mistake would
# still reproduce.
#
# Out of process because it needs its own Problem, following the pattern the rest of the suite uses.

import argparse
import os
import sys

from pyoomph import Problem, Equations, InitialCondition
from pyoomph.expressions import var, var_and_test, partial_t, sin, pi
from pyoomph.equations.generic import ExtremumObservables, IntegralObservables
from pyoomph.meshes.simplemeshes import RectangularQuadMesh
from pyoomph.utils.bifurcation_gui import BifurcationGUI

Lx, Ly = 2.0, 1.0
SHIFT = 0.15
MAXVAL, MAXX, MAXY = "domain/h_extreme  [max, val]", "domain/h_extreme  [max, x]", "domain/h_extreme  [max, y]"
MINVAL, MINX, MINY = "domain/h_extreme  [min, val]", "domain/h_extreme  [min, x]", "domain/h_extreme  [min, y]"


class Drift(Equations):
    def __init__(self, mu):
        super().__init__()
        self.mu = mu

    def define_fields(self):
        self.define_scalar_field("h", "C2")

    def define_residuals(self):
        h, ht = var_and_test("h")
        x, y = var(["coordinate_x", "coordinate_y"])
        target = sin(2*pi*(x/Lx - SHIFT*self.mu))*sin(pi*y/Ly)
        self.add_weak(partial_t(h) - (target - h), ht)


class Prob(Problem):
    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(size=[Lx, Ly], N=[24, 12]))
        eqs = Drift(self.get_global_parameter("mu"))
        eqs += InitialCondition(h=0)
        eqs += ExtremumObservables(h_extreme=var("h"))
        eqs += IntegralObservables(_area=1, h_int=var("h"))
        self += eqs @ "domain"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    with Prob() as problem:
        problem.set_output_directory(args.outdir)
        problem.get_global_parameter("mu").value = 0.0
        gui = BifurcationGUI(problem, "mu")
        gui.neigen = 1
        gui.set_initial_view(-0.2, 1.2, -0.2, 2.2)
        # The diagram should open on this one rather than on the first name in alphabetical order.
        # Written with a single space, which the double space of the extremum names has to tolerate.
        gui.set_initial_observable("domain/h_extreme [min, x]")
        problem.solve()
        gui.controller.start(0.05)
        c = gui.controller

        assert c._current_observable == MINX, \
            "set_initial_observable() did not take: " + str(c._current_observable)
        assert c.y_axis == ("observable", MINX), "the y axis did not follow: " + str(c.y_axis)
        # A name that does not exist must say so, not fall back to an arbitrary observable.
        c._initial_observable = "domain/not_an_observable"
        try:
            c._resolve_initial_observable()
        except ValueError as e:
            assert "not_an_observable" in str(e)
        else:
            raise AssertionError("an unknown initial observable was accepted")
        c._initial_observable = None

        obs = c.evaluate_observables()
        print("axis choices: " + ", ".join(sorted(k for k in obs if "h_extreme" in k)))

        # Every choice must exist, and no z on a 2D mesh.
        for key in (MAXVAL, MAXX, MAXY, MINVAL, MINX, MINY):
            assert key in obs, "missing axis choice " + key
        assert not any(k.endswith("[max, z]") or k.endswith("[min, z]") for k in obs), \
            "a 2D mesh must not offer a z position"

        # Values and positions against the closed form. The tolerance is the discretisation: the
        # extremum is sampled on the mesh and h is the projection of the target, not the target.
        for key, expect in ((MAXVAL, 1.0), (MINVAL, -1.0),
                            (MAXX, Lx*0.25), (MAXY, Ly/2),
                            (MINX, Lx*0.75), (MINY, Ly/2)):
            err = abs(obs[key] - expect)
            print("  {:<30s} {: .6f}  expect {: .6f}  err {:.2e}".format(key, obs[key], expect, err))
            assert err < 3e-2, key + " is off by {:.3e}".format(err)

        # One mesh sweep per extremum, not per axis choice: six choices come from two sweeps.
        assert len(c._extremum_cache) == 2, \
            "expected 2 mesh sweeps, got {:d}".format(len(c._extremum_cache))

        # They have to work as AXIS selections, through the same path the widgets use.
        gui.plotter.initialise_view(c)
        app = gui._create_app()
        gui.app = app
        app.root.withdraw()
        app.run = lambda: None
        app._rebuild_axis_menus()
        offered = list(app.obs_combo["values"])
        # The tag replaces the "[obs]" an ordinary observable gets, rather than being added to it.
        assert MAXX in offered, "the axis menu does not offer '" + MAXX + "', only: " + str(offered)
        assert MAXX + "  [obs]" not in offered, "the extremum tag must replace [obs], not stack with it"
        assert "domain/h_int  [obs]" in offered, "an ordinary observable must still be tagged [obs]"

        app._set_axis("y", app._axis_display(("observable", MAXX)))
        app._set_axis("x", app._axis_display(("observable", MINVAL)))
        assert c.y_axis == ("observable", MAXX), "the y axis did not take: " + str(c.y_axis)
        assert c.x_axis == ("observable", MINVAL), "the x axis did not take: " + str(c.x_axis)

        for _ in range(3):
            app._invoke(app._actions["step"])
        app.refresh()

        print("the maximum drifting with mu:")
        for p in gui.branches[0]:
            expect = Lx*(0.25 + SHIFT*p.param_value)
            got = p.obs_values[MAXX]
            print("  mu={: .4f}  max at x={: .4f}  expect {: .4f}".format(p.param_value, got, expect))
            assert abs(got - expect) < 4e-2, "the peak is not where mu puts it at mu={:.4g}".format(p.param_value)

        xs = [p.obs_values[MAXX] for p in gui.branches[0]]
        assert max(xs) - min(xs) > 0.05, "the peak position should move along the branch: " + str(xs)

        # A name with spaces, a comma and brackets has to survive the export and a save/load.
        c.output_all_observables = True
        c.output_curves()
        odir = problem.get_output_directory(os.path.join(c.data_subdir, "output"))
        written = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(odir) for f in fs if f.endswith(".txt"))
        assert written, "nothing was exported"
        with open(written[0]) as fh:
            head = fh.readline()
        assert MAXX in head and MINVAL in head, "extremum columns missing from the export: " + head

        c.save_all()
        c.load_all()
        assert c.y_axis == ("observable", MAXX), "the axis did not survive save/load: " + str(c.y_axis)

    print("PYOOMPH_WORKER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
