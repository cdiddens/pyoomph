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

import numpy
from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.output.plotting1d import MatplotlibPlotter1D


class DiffusionEquation(Equations):
    def __init__(self, D=0.05):
        super(DiffusionEquation, self).__init__()
        self.D = D

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(partial_t(u), v) + self.D * weak(grad(u), grad(v)))


class DiffusionPlotter(MatplotlibPlotter1D):
    D = 0.05

    def define_plot(self):
        self.background_color = "white"
        self.image_size = [900, 500]
        # A graph, not a spatial map. rangemode_y="grow" unions the y-range over all output steps, so
        # the frames of the resulting movie stay comparable instead of each rescaling to its own peak.
        # legend_position rather than legend=True, because matplotlib's "best" would put the legend
        # under the time label - the two know nothing about each other.
        self.set_axes(xmin=0, xmax=1, rangemode_y="grow", legend_position="upper right",
                      title="a decaying diffusion mode")
        # The abscissa is coordinate_x unless another one is asked for, so a field name is all it takes
        self.add_plot("domain/u", color="navy", linewidth=2, label="$u$")
        # Where the elements are, which a smooth curve otherwise hides completely
        self.add_nodes("domain/u", only_vertex_nodes=True, color="navy", markersize=4)
        self.add_element_borders("domain", color="0.85")
        # exp(-D pi^2 t) sin(pi x) solves this exactly, so it can be drawn straight over the solution
        t = float(self.get_problem().get_current_time(dimensional=False, as_float=True))
        self.add_analytical(lambda x: numpy.exp(-self.D * numpy.pi ** 2 * t) * numpy.sin(numpy.pi * x),
                            color="crimson", linestyle="dotted", label="exact")
        self.add_time_label(position="top left")


class DiffusionProblem(Problem):
    def define_problem(self):
        self.add_mesh(LineMesh(N=20, size=1))
        eqs = DiffusionEquation()
        eqs += InitialCondition(u=sin(pi * var("coordinate_x")))
        eqs += DirichletBC(u=0) @ "left"
        eqs += DirichletBC(u=0) @ "right"
        self.add_equations(eqs @ "domain")


# A one-dimensional domain need not live in one-dimensional space. This one is bent onto a curve in
# the plane, so it has a shape of its own to draw as well.
class BendOntoCurve(Equations):
    def define_fields(self):
        self.activate_coordinates_as_dofs(coordinate_space="C2")

    def define_residuals(self):
        x, y = var(["coordinate_x", "coordinate_y"])
        X = var("lagrangian_x")
        self.add_residual(weak(x - X, testfunction("mesh_x")))
        self.add_residual(weak(y - 0.3 * sin(2 * pi * X), testfunction("mesh_y")))


class CurvePlotter(MatplotlibPlotter1D):
    def define_plot(self):
        self.background_color = "white"
        self.image_size = [900, 500]
        self.aspect_ratio = True  # so that the shape is not distorted
        self.set_axes(grid=True, title="the curve the mesh itself traces")
        cb = self.add_colorbar("u", position="top right", length=0.25, thickness=0.03)
        # A bare domain name means its own (x,y) curve; colorfield says what colours it
        self.add_curve("domain", colorbar=cb, colorfield="u", linewidth=6)


class ArclengthPlotter(MatplotlibPlotter1D):
    def define_plot(self):
        self.background_color = "white"
        self.image_size = [900, 500]
        self.file_trunk = "arclength_{:05d}"
        self.set_axes(grid=True, legend=True, title="the same field, two abscissae")
        self.add_plot("domain/u", label="against $x$", color="navy")
        self.add_plot("domain/u", xaxis="arclength", label="against arclength $s$",
                      color="crimson", linestyle="dashed")


class CurvedProblem(Problem):
    def define_problem(self):
        # nodal_dimension=2 is what gives the line mesh a y coordinate to be bent into
        self.add_mesh(LineMesh(N=20, size=1, nodal_dimension=2))
        eqs = BendOntoCurve() + PoissonEquation(name="u", source=1, space="C2")
        eqs += DirichletBC(u=0) @ "left"
        eqs += DirichletBC(u=0) @ "right"
        self.add_equations(eqs @ "domain")


if __name__ == "__main__":
    with DiffusionProblem() as problem:
        problem.set_output_directory("diffusion")
        problem.plotter = DiffusionPlotter()
        problem.run(2, outstep=0.1, startstep=0.05)

    with CurvedProblem() as problem:
        problem.set_output_directory("curved")
        problem.plotter = [CurvePlotter(), ArclengthPlotter()]
        problem.solve()
        problem.output()
