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

# What the generated C source actually computes, for expressions whose *representation* -- not
# whose value -- decides how they are printed.
#
# GiNaC compares and hashes numbers by value: an exact -1 and an inexact -1.0 are equal and hash
# alike. Its C-source printer for products, however, distinguishes them, and does so
# inconsistently: mul::do_print_csrc decides whether to emit "1.0/" or "/" from
# info_flags::negint, which an inexact -1 does not satisfy, but decides whether to leave the
# exponent out altogether from is_equal(-1), which it does. A factor x^(-1.0) is therefore printed
# as a multiplication by x. Nothing rejects it; the generated code is simply wrong, and wrong by a
# factor of x^2.
#
# pyoomph produced such exponents by itself, through the memo in
# ReplaceFieldsToNonDimFields::operator(): keyed by GiNaC's hash and compared with is_equal, it
# handed back the inexact -1.0 of a floating point coefficient when asked to expand the exact -1
# exponent of a reciprocal it had met later. Both halves are fixed, so both are tested here.

import numpy

from pyoomph import Problem, DirichletBC
from pyoomph.expressions import var, matrix, transpose, matproduct, identity_matrix, symbolic_diff
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.equations.viscoelastic import symmetric_2x2_matrix_log
from pyoomph.meshes.simplemeshes import RectangularQuadMesh


class _DirichletValues(Problem):
    """Pins one Poisson field per expression, so the nodal values are the generated code's answer."""

    def __init__(self, expressions):
        super().__init__()
        self.expressions = expressions

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=[1, 4]))
        equations = None
        for name in self.expressions:
            poisson = PoissonEquation(source=1, name=name, space="C2")
            equations = poisson if equations is None else equations + poisson
        equations += DirichletBC(**self.expressions) @ "left"
        equations += DirichletBC(**{name: 0 for name in self.expressions}) @ "right"
        self.add_equations(equations @ "domain")


def _pinned_values(tmp_path, expressions):
    """{name: {y: value}} of the Dirichlet conditions, as the compiled element evaluates them."""
    with _DirichletValues(expressions) as problem:
        problem.set_output_directory(str(tmp_path))
        problem.initialise()
        mesh = problem.get_mesh("domain")
        indices = mesh.get_nodal_field_indices()
        values = {name: {} for name in expressions}
        for node in mesh.nodes():
            if node.x(0) < 1e-9:
                for name in expressions:
                    values[name][round(node.x(1), 6)] = node.value(indices[name])
        return values


def test_inexact_exponent_keeps_its_reciprocal(tmp_path):
    """
    x**(-1.0) is a division, whatever the exponent's internal representation.

    The two expressions here are equal to GiNaC -- their difference is exactly zero -- and differ
    only in whether the exponent is stored as an integer or as a floating point number.
    """
    y = var("coordinate_y")
    expressions = {"exact": 3 / (1 + y), "inexact": 3 * (1 + y) ** (-1.0)}
    assert (expressions["exact"] - expressions["inexact"]).is_zero()
    values = _pinned_values(tmp_path, expressions)
    for position, value in values["exact"].items():
        assert abs(value - 3 / (1 + position)) < 1e-12
        assert abs(values["inexact"][position] - value) < 1e-12


def test_a_float_coefficient_does_not_break_an_unrelated_reciprocal(tmp_path):
    """
    The same expression, written once with an exact and once with a floating point coefficient.

    This is how the defect was actually met: the expression contains a reciprocal whose exponent is
    an exact -1, and a coefficient that is an inexact -1.0 only in the second variant. The expansion
    memo could not tell the two numbers apart and gave the reciprocal the coefficient's
    representation, after which the generated code multiplied where it should have divided. Note
    that a matrix logarithm is not incidental here: it is where the reciprocal comes from.
    """
    y = var("coordinate_y")

    def logarithm(entry):
        L = matrix([[0, entry, 0], [0, 0, 0], [0, 0, 0]])
        C = identity_matrix(3) + (L + transpose(L)) + 2 * matproduct(L, transpose(L))
        return symmetric_2x2_matrix_log(C)[0, 1]

    expressions = {"exact": logarithm(-1 * y), "float_coefficient": logarithm(-1.0 * y),
                   # the same thing again, but with the coefficient coming out of a differentiation
                   # of a profile written with floats, which is what an inflow condition does
                   "differentiated": logarithm(symbolic_diff(0.5 * (1 - y ** 2), "coordinate_y",
                                                             hold_until_codegen=False))}
    values = _pinned_values(tmp_path, expressions)
    for position, value in values["exact"].items():
        # log of [[1+2*w^2, w], [w, 1]] with w=-y, off-diagonal component
        C = numpy.array([[1 + 2 * position ** 2, -position], [-position, 1.0]])
        eigenvalues, eigenvectors = numpy.linalg.eigh(C)
        reference = (eigenvectors @ numpy.diag(numpy.log(eigenvalues)) @ eigenvectors.T)[0, 1]
        assert abs(value - reference) < 1e-12
        for name in ("float_coefficient", "differentiated"):
            assert abs(values[name][position] - value) < 1e-12, name


def test_in_place_arithmetic_with_a_plain_number_stays_real(tmp_path):
    """
    "expr -= 1" must produce the real numeric -1, not the complex -1.0+0.0i.

    __iadd__ and __isub__ used to be bound only against std::complex<double>, and nanobind converts an
    int or a float to that without complaint, so the in-place forms quietly produced complex constants.
    Nothing looked wrong symbolically -- they print as "-1" and compare equal to -1 -- but the C printer
    emitted std::complex<double>(-1.0,0.0) into a real-valued residual and the compiler rejected the
    whole element, so the failure surfaced as "command '/usr/bin/cc' failed" far from its cause. *= and
    /= were unaffected, they always had int/double overloads.

    Met through RadialSymmetricCoordinateSystem, whose gradients and divergence shift the radius with
    "coords[0] -= self.Rcenter": any non-zero Rcenter given as a plain number made the problem
    uncompilable, while wrapping it in Expression() worked.
    """
    y = var("coordinate_y")

    def in_place_minus():
        value = 3 * y
        value -= 1
        return value

    def in_place_plus():
        value = 3 * y
        value += 1
        return value

    expressions = {"binary_minus": 3 * y - 1, "in_place_minus": in_place_minus(),
                   "binary_plus": 3 * y + 1, "in_place_plus": in_place_plus()}
    # Compiling at all is most of the point here; the values then confirm nothing else moved.
    values = _pinned_values(tmp_path, expressions)
    for position in values["binary_minus"]:
        assert abs(values["in_place_minus"][position] - (3 * position - 1)) < 1e-12
        assert abs(values["in_place_plus"][position] - (3 * position + 1)) < 1e-12
        assert values["in_place_minus"][position] == values["binary_minus"][position]
        assert values["in_place_plus"][position] == values["binary_plus"][position]
