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

# The shared "sort_along_axis / start_near_point" treatment in pyoomph.meshes.ordering and the
# places that use it: TextFileOutput, AssignZetaCoordinatesByArclength and get_boundary_coordinates.
#
# Most of this goes through TextFileOutput, because the point of those kwargs is a file that can be
# plotted as a curve, so the checks read the written file rather than any internal state. The case
# that matters most is the bent line: ordering is decided on the segment end points only, so the
# written points must follow the curve even where x runs backwards along it. A naive implementation
# that sorts the points themselves passes every other test here and fails that one.

import numpy
import pytest

from pyoomph import Problem, DirichletBC, TextFileOutput
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.expressions.units import meter
from pyoomph.meshes.ordering import sort_line_segments, sort_point_indices
from pyoomph.meshes.simplemeshes import LineMesh, RectangularQuadMesh, CuboidBrickMesh


def _read(tmp_path, trunk, column):
    fname = tmp_path / trunk / (trunk + "_000000.txt")
    with open(fname, "r") as f:
        header = [h.strip() for h in f.readline().lstrip("#").split("\t")]
    data = numpy.atleast_2d(numpy.loadtxt(fname))
    for i, h in enumerate(header):
        if h == column or h.startswith(column + "["):
            return data[:, i]
    raise KeyError(column + " not among " + str(header))


class _LineProblem(Problem):
    """1d bulk mesh with a spatial scale, so the units of start_near_point are exercised too."""

    def define_problem(self):
        self += LineMesh(N=6, size=1 * meter)
        self.set_scaling(spatial=1 * meter)
        eqs = PoissonEquation(source=1, coefficient=1 * meter**2)
        eqs += DirichletBC(u=0) @ "left" + DirichletBC(u=0) @ "right"
        eqs += TextFileOutput(filetrunk="axis", sort_along_axis="x-")
        eqs += TextFileOutput(filetrunk="point", start_near_point=[1 * meter])
        eqs += TextFileOutput(filetrunk="nondim", nondimensional=True, start_near_point=[1 * meter])
        self += eqs @ "domain"


class _PointsProblem(Problem):
    """Both ends of the line share one boundary name, giving a 0d domain of two points."""

    def define_problem(self):
        self += LineMesh(N=6, left_name="ends", right_name="ends")
        eqs = PoissonEquation(source=1)
        eqs += DirichletBC(u=0) @ "ends"
        eqs += TextFileOutput(filetrunk="axis", sort_along_axis="x-") @ "ends"
        eqs += TextFileOutput(filetrunk="point", start_near_point=0.9) @ "ends"
        self += eqs @ "domain"


class _BentLineProblem(Problem):
    """1d line embedded in 2d, bent in bend() so that x is not monotonic along the curve."""

    def define_problem(self):
        self += LineMesh(N=20, nodal_dimension=2)
        eqs = PoissonEquation(source=1)
        eqs += DirichletBC(u=0) @ "left" + DirichletBC(u=0) @ "right"
        eqs += TextFileOutput(filetrunk="axis", sort_along_axis="x-")
        eqs += TextFileOutput(filetrunk="point", start_near_point=[0.0, 0.0])
        self += eqs @ "domain"

    def bend(self):
        for n in self.get_mesh("domain").nodes():
            t = n.x(0)
            n.set_x(1, t)
            n.set_x(0, t + 0.3 * numpy.sin(4 * numpy.pi * t))


class _EdgeProblem(Problem):
    """3d mesh, output on a co-dimension 2 interface, i.e. a line in 3d."""

    def define_problem(self):
        self += CuboidBrickMesh(N=3)
        eqs = PoissonEquation(source=1)
        for b in ["left", "right", "top", "bottom", "front", "back"]:
            eqs += DirichletBC(u=0) @ b
        eqs += TextFileOutput(filetrunk="axis", sort_along_axis="x-") @ "top/back"
        eqs += TextFileOutput(filetrunk="zaxis", sort_along_axis="z+") @ "top/right"
        eqs += TextFileOutput(filetrunk="point", start_near_point=[1, 1, 0]) @ "top/back"
        self += eqs @ "domain"


class _BulkProblem(Problem):
    def define_problem(self):
        self += RectangularQuadMesh(N=3)
        eqs = PoissonEquation(source=1)
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        eqs += TextFileOutput(sort_along_axis="x-")
        self += eqs @ "domain"


def _output(problem, tmp_path):
    problem.set_output_directory(str(tmp_path))
    problem.initialise()
    problem.output()


def test_1d_bulk_mesh_is_written_in_the_requested_order(tmp_path):
    with _LineProblem() as problem:
        _output(problem, tmp_path)
    for trunk in ["axis", "point"]:
        x = _read(tmp_path, trunk, "coordinate_x")
        assert len(x) == 13  # C2 line mesh: 6 elements, 13 nodes, no NaN separator for one segment
        assert numpy.all(numpy.diff(x) < 0), trunk


def test_a_start_near_point_with_units_works_on_nondimensional_output(tmp_path):
    # The cache reports a unit of 1 in nondimensional mode, so the point has to be divided by the
    # domain's spatial scale instead; getting that wrong makes float() throw on a dimensional point.
    with _LineProblem() as problem:
        _output(problem, tmp_path)
    x = _read(tmp_path, "nondim", "coordinate_x")
    assert x[0] == pytest.approx(1.0)
    assert numpy.all(numpy.diff(x) < 0)


def test_0d_point_domain_is_ordered(tmp_path):
    with _PointsProblem() as problem:
        _output(problem, tmp_path)
    for trunk in ["axis", "point"]:
        assert list(_read(tmp_path, trunk, "coordinate_x")) == [1.0, 0.0], trunk


def test_ordering_follows_a_curve_that_runs_backwards(tmp_path):
    with _BentLineProblem() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.initialise()
        problem.bend()
        problem.output()
    x = _read(tmp_path, "axis", "coordinate_x")
    y = _read(tmp_path, "axis", "coordinate_y")
    assert numpy.any(numpy.diff(x) > 0)  # not a plain sort by x
    assert numpy.all(numpy.diff(y) < 0)  # but the curve is walked, reversed as x- asks for
    # The same curve from its (0,0) end: the mesh order already starts there, so nothing is reversed
    assert numpy.all(numpy.diff(_read(tmp_path, "point", "coordinate_y")) > 0)


def test_line_in_3d_is_ordered_along_any_axis(tmp_path):
    with _EdgeProblem() as problem:
        _output(problem, tmp_path)
    assert numpy.all(numpy.diff(_read(tmp_path, "axis", "coordinate_x")) < 0)
    assert numpy.allclose(_read(tmp_path, "axis", "coordinate_y"), 1.0)
    assert numpy.allclose(_read(tmp_path, "axis", "coordinate_z"), 0.0)
    assert numpy.all(numpy.diff(_read(tmp_path, "zaxis", "coordinate_z")) > 0)
    assert numpy.all(numpy.diff(_read(tmp_path, "point", "coordinate_x")) < 0)


def test_contradictory_or_impossible_orderings_are_refused(tmp_path):
    with pytest.raises(RuntimeError, match="not both"):
        TextFileOutput(sort_along_axis="x+", start_near_point=[0, 0])
    with pytest.raises(RuntimeError, match="Unknown sort_along_axis"):
        TextFileOutput(sort_along_axis="X")
    with pytest.raises(RuntimeError, match="Cannot combine"):
        TextFileOutput(sort_along_axis="x+", sort_segments_by=lambda seg, coords: 0.0)
    with pytest.raises(RuntimeError, match="only work on 0d and 1d domains"):
        with _BulkProblem() as problem:
            _output(problem, tmp_path)


def test_the_shared_ordering_orients_and_sorts_segments():
    pts = numpy.array([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 1.0, 1.0]])
    assert sort_line_segments(pts, [[1, 0], [3, 2]], sort_along_axis="x+") == [[0, 1], [2, 3]]
    assert sort_line_segments(pts, [[1, 0], [3, 2]], sort_along_axis="x-") == [[3, 2], [1, 0]]
    # Nearest end of the nearest segment first - this used to be the far end, in both copies of it
    assert sort_line_segments(pts, [[1, 0], [3, 2]], start_near_point=(3.1, 1.0)) == [[3, 2], [1, 0]]
    assert sort_point_indices(pts, sort_along_axis="y-") == [2, 3, 0, 1]
    assert sort_point_indices(pts, start_near_point=[2.0, 1.0]) == [2, 3, 1, 0]
    with pytest.raises(RuntimeError, match="only has 2 coordinate"):
        sort_line_segments(pts, [[0, 1]], sort_along_axis="z+")
    with pytest.raises(RuntimeError, match="has 3 entries"):
        sort_line_segments(pts, [[0, 1]], start_near_point=(1, 2, 3))
