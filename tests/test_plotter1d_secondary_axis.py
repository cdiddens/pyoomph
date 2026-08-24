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

# MatplotlibPlotter1D's secondary y-axis has to sit on the SAME rectangle as the primary one.
#
# twinx() shares the x-axis - that part was always right - but it copies the parent's rectangle once,
# when it is created, and does not follow later set_position() calls. The twin can be created before
# the plotter applies its margins, so it kept matplotlib's default subplot rect and the figure came
# out with two frames a few percent apart: measured (0.125, 0.11, 0.775, 0.77) for the twin against
# (0.1, 0.13, 0.8, 0.82) for the axes everything else was drawn on.
#
# Asserted on the axes rather than on pixels: a rendered comparison would fail for any change of
# style, and what went wrong here is a geometry, not an appearance.

import os

import matplotlib
matplotlib.use("Agg")

from pyoomph import Problem, Equations
from pyoomph.expressions import var_and_test, var, weak, grad, partial_t
from pyoomph.equations.generic import DirichletBC
from pyoomph.meshes.simplemeshes import LineMesh
from pyoomph.output.plotting1d import MatplotlibPlotter1D


class _Diffusion(Equations):
    def define_fields(self):
        self.define_scalar_field("u", "C1")
        self.define_scalar_field("v", "C1")

    def define_residuals(self):
        u, ut = var_and_test("u")
        v, vt = var_and_test("v")
        self.add_weak(partial_t(u), ut).add_weak(grad(u), grad(ut)).add_weak(-1, ut)
        self.add_weak(partial_t(v), vt).add_weak(grad(v), grad(vt)).add_weak(-100, vt)


class _TwoScalePlotter(MatplotlibPlotter1D):
    """u and v on their own y-axes, with margins that are NOT matplotlib's defaults."""

    def define_plot(self):
        self.set_axes(ylabel="u", y2label="v", ymin=0, ymax=1, y2min=0, y2max=100,
                      margins=(0.10, 0.13, 0.90, 0.95), legend=True)
        self.add_plot("domain/u", color="navy", label="u")
        self.add_plot("domain/v", color="darkred", use_y2=True, label="v")


class _P(Problem):
    def define_problem(self):
        self += LineMesh(N=8)
        # Pinned at both ends: without them the steady problem is singular, and this test is about
        # the axes, not about finding something interesting to plot.
        eqs = _Diffusion()
        eqs += DirichletBC(u=0, v=0) @ "left"
        eqs += DirichletBC(u=0, v=0) @ "right"
        self += eqs @ "domain"


def test_the_secondary_y_axis_shares_the_primary_ones_rectangle(tmp_path):
    with _P() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.quiet()
        problem.plotter = _TwoScalePlotter(problem)
        problem.solve()
        problem.plotter.plot()

        axes = problem.plotter.main_axes
        ax, ax2 = axes.ax, axes.ax_y2
        assert ax2 is not None, "use_y2 has to create the secondary axis"

        # The geometry: same rectangle, so there is one frame and not two.
        assert ax2.get_position().bounds == ax.get_position().bounds, \
            "the twin kept its own rectangle: {!r} vs {!r}".format(
                ax2.get_position().bounds, ax.get_position().bounds)
        # ... and it really is the one the plotter asked for, not a default that happens to match.
        left, bottom, right, top = (0.10, 0.13, 0.90, 0.95)
        assert ax.get_position().bounds == (left, bottom, right - left, top - bottom)

        # The x-axis is shared, which is what makes the two curves comparable point by point.
        assert ax in ax2.get_shared_x_axes().get_siblings(ax2)
        assert ax.get_xlim() == ax2.get_xlim()

        # Each axis keeps its own y-range.
        assert ax.get_ylim() == (0.0, 1.0)
        assert ax2.get_ylim() == (0.0, 100.0)

        # Only the right spine of the twin is drawn; the other three would double the parent's box.
        assert ax2.spines["right"].get_visible()
        for side in ("top", "bottom", "left"):
            assert not ax2.spines[side].get_visible(), side + " spine of the twin is drawn twice"

        # And it does produce a figure.
        png = os.path.join(str(tmp_path), "_plots", "plot_00000.png")
        assert os.path.exists(png) and os.path.getsize(png) > 0
