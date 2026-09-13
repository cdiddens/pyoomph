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

# The headers pyoomph writes on top of its text output.
#
# A column name is "name[unit]", and a compound unit used to be written with a SPACE between its
# symbols - "power[kg m^2/s^4]". pyoomph's own reader joins and splits the header on tabs and copes
# with that, but nothing else does: numpy.loadtxt on the header line, awk, a spreadsheet import or a
# plain header.split() all take the space as a column break, so one such name became three columns
# and every name after it lined up against the wrong data. Units written into a file therefore use
# UNIT_SEPARATOR_IN_FILES, while everything meant for a person to read (plot labels, the bifurcation
# GUI) keeps the readable space.
#
# Asserted on the files themselves rather than on unit_to_string, because what matters is that every
# writer passes the flag: test_unit_strings.py covers the function.

import glob
import os

import numpy
import pytest

from pyoomph import Problem, Equations, ODEEquations
from pyoomph.expressions import var, var_and_test, weak, grad, partial_t, scale_factor
from pyoomph.expressions.units import kilogram, meter, second, kelvin, milli
from pyoomph.equations.generic import InitialCondition, IntegralObservables
from pyoomph.output.generic import (TextFileOutput, TextFileOutputAlongLine, ODEFileOutput,
                                    IntegralObservableOutput)
from pyoomph.meshes.simplemeshes import RectangularQuadMesh
from pyoomph.utils.num_text_out import LoadedTextDataFile

#: kg m^2/s^4 on purpose: three symbols, and no derived unit of its own to collapse to (kg m^2/s^3
#: is the watt, which is one symbol and would separate nothing).
U_SCALE = 1 * kilogram * meter ** 2 / second ** 4
#: kg m/(K s^2): a denominator of several symbols too, which is bracketed.
W_SCALE = 1 * kilogram * meter / (second ** 2 * kelvin)


class _Field(Equations):
    def define_fields(self):
        self.define_scalar_field("u", "C2", scale=U_SCALE,
                                 testscale=scale_factor("spatial") ** 2 / U_SCALE)

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(grad(u), grad(v)))


class _Ode(ODEEquations):
    def define_fields(self):
        self.define_ode_variable("w", scale=W_SCALE, testscale=scale_factor("temporal") / W_SCALE)

    def define_residuals(self):
        w, wt = var_and_test("w")
        self.add_residual(weak(partial_t(w), wt))


class _OutputEverything(Problem):
    """Every text writer at once, on a problem whose scales are dimensional."""

    def define_problem(self):
        self.set_scaling(spatial=1 * milli * meter, temporal=1 * second)
        self.add_mesh(RectangularQuadMesh(size=[2 * milli * meter, 1 * milli * meter], N=[3, 2]))
        eqs = _Field() + TextFileOutput()
        eqs += TextFileOutputAlongLine(start=[0, 0.5 * milli * meter],
                                       end=[2 * milli * meter, 0.5 * milli * meter],
                                       N=5, filename="alongline")
        eqs += IntegralObservables(power=var("u")) + IntegralObservableOutput()
        self += eqs @ "domain"
        self += (_Ode() + InitialCondition(w=W_SCALE) + ODEFileOutput()) @ "ode"


def _headers(outdir):
    """The header line of every text file the run wrote, keyed by file name."""
    out = {}
    for path in sorted(glob.glob(os.path.join(outdir, "**", "*.txt"), recursive=True)):
        name = os.path.basename(path)
        if name.startswith("_") or os.sep + "_ccode" + os.sep in path:
            continue     # pyoomph's own logs and generated sources, not data
        with open(path) as f:
            line = f.readline().rstrip("\n")
        if line.startswith("#"):
            out[name] = line
    return out


@pytest.fixture(scope="module")
def written_headers(tmp_path_factory):
    outdir = str(tmp_path_factory.mktemp("text_headers") / "out")
    with _OutputEverything() as problem:
        problem.set_output_directory(outdir)
        problem.quiet()
        problem.run(1 * second, numouts=1, startstep=1 * second)
    headers = _headers(outdir)
    # two writers produce one file per output step, two append to a single file
    assert len(headers) == 6, "expected six data files, got " + repr(sorted(headers))
    return outdir, headers


def test_no_written_header_contains_a_space(written_headers):
    _outdir, headers = written_headers
    for name, line in headers.items():
        # Only the "# " that opens a comment line is allowed, and loadtxt drops that anyway.
        body = line.lstrip("#").strip()
        assert " " not in body, name + " has a space in its header: " + repr(line)


def test_every_writer_states_the_unit_it_was_given(written_headers):
    """Space-free must not mean unit-free: each writer still says what its column is in."""
    _outdir, headers = written_headers
    assert "u[kg*m^2/s^4]" in headers["domain_000001.txt"]
    assert "u[kg*m^2/s^4]" in headers["alongline_000001.txt"]
    assert "power[kg*m^4/s^4]" in headers["domain_IntObsv.txt"]     # integrated over an area
    assert "w[kg*m/(K*s^2)]" in headers["ode.txt"]
    for name in ("domain_IntObsv.txt", "ode.txt"):
        assert "time[s]" in headers[name]


def test_the_columns_survive_a_plain_whitespace_split(written_headers):
    """What the space broke: splitting the header on whitespace has to give one token per column."""
    outdir, headers = written_headers
    for name, line in headers.items():
        path = glob.glob(os.path.join(outdir, "**", name), recursive=True)[0]
        ncols = numpy.loadtxt(path, ndmin=2).shape[1]
        tokens = [t for t in line.lstrip("#").split() if not t.startswith("@")]
        assert len(tokens) == ncols, \
            name + ": header splits into " + str(len(tokens)) + " tokens for " + str(ncols) + " columns"


def test_pyoomphs_own_reader_still_finds_the_columns(written_headers):
    outdir, _headers = written_headers
    data = LoadedTextDataFile(glob.glob(os.path.join(outdir, "**", "ode.txt"), recursive=True)[0])
    assert data.columns == ["time[s]", "w[kg*m/(K*s^2)]"]
    assert data["w"].shape[0] == data.data.shape[0]
