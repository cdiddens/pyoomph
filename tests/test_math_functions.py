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

# asinh/acosh/atanh (from GiNaC) and erf/erfc (registered by pyoomph itself, GiNaC has no error function).
#
# All five reach the generated code as a call to the C99 function of the same name, which is why the tcc arm of these
# tests is not redundant: tcc is invoked with -nostdinc, so jitbridge.h declares by hand every libm function the
# generated code may call. A function missing from that list is not a compile error but an implicit int-returning
# declaration - erf and erfc first came back as 1072064102 rather than 0.677..., with nothing reported anywhere.

import math

import numpy
import pytest

from pyoomph import ODEEquations, Problem, _pyoomph
from pyoomph.expressions import acosh, asinh, atanh, erf, erfc, heaviside, imag_part, imaginary_i, real_part, var, var_and_test
from pyoomph.expressions.units import meter, milli
from pyoomph.generic.ccompiler import BaseCCompiler

# shift keeps both the function and its derivative finite at the initial guess 0 as well as at the solution
FUNCTIONS = {"erf": (erf, math.erf, 0.0), "erfc": (erfc, math.erfc, 0.0), "asinh": (asinh, math.asinh, 0.0),
             "acosh": (acosh, math.acosh, 2.0), "atanh": (atanh, math.atanh, 0.0)}
SOLUTION = 0.7

COMPILERS = sorted(set(BaseCCompiler.available_compilers().keys()) & {"tccbox", "system"})


def test_symbolic_evaluation_matches_the_math_module():
    for name, (function, reference, shift) in FUNCTIONS.items():
        for argument in (0.25, 0.7, 0.9):  # atanh keeps us below 1
            value = float(function(argument + shift).evalf())
            assert value == pytest.approx(reference(argument + shift), rel=1e-15), name
    # the two arguments with an exact answer, which must not be turned into a float
    assert erf(0).is_zero()
    assert (erfc(0) - 1).is_zero()
    # what erfc is for: 1-erf(5) is 1.5e-12 of cancellation, erfc(5) is not
    assert float(erfc(5).evalf()) == pytest.approx(math.erfc(5), rel=1e-15)


class _Values(ODEEquations):
    """One variable per function, each written through the unknown s so that a Jacobian entry is generated."""

    def define_fields(self):
        self.define_ode_variable("s")
        for name in FUNCTIONS:
            self.define_ode_variable("u_" + name)

    def define_residuals(self):
        s, vs = var_and_test("s")
        self.add_residual((s - SOLUTION) * vs)
        for name, (function, reference, shift) in FUNCTIONS.items():
            u, v = var_and_test("u_" + name)
            self.add_residual((u - function(s + shift)) * v)


class _ValueProblem(Problem):
    def define_problem(self):
        self.add_equations(_Values() @ "ode")


@pytest.mark.parametrize("compiler", COMPILERS)
def test_generated_code_matches_the_math_module(tmp_path, compiler):
    with _ValueProblem() as problem:
        problem.set_c_compiler(compiler)
        problem.set_output_directory(str(tmp_path))
        problem.solve()
        ode = problem.get_ode("ode")
        for name, (function, reference, shift) in FUNCTIONS.items():
            value = float(ode.get_value("u_" + name))
            assert value == pytest.approx(reference(SOLUTION + shift), rel=1e-14), name


class _Nonlinear(ODEEquations):
    """The functions applied to the unknowns themselves, so their derivatives end up in the Jacobian."""

    def define_fields(self):
        self.define_ode_variable("s")
        for name in FUNCTIONS:
            self.define_ode_variable("u_" + name)

    def define_residuals(self):
        s, vs = var_and_test("s")
        self.add_residual((s - 0.4) * vs)
        for name, (function, reference, shift) in FUNCTIONS.items():
            u, v = var_and_test("u_" + name)
            self.add_residual((function(0.5 * u + 0.3 * s + shift) - 0.2) * v)


class _NonlinearProblem(Problem):
    def define_problem(self):
        self.add_equations(_Nonlinear() @ "ode")


def test_generated_jacobian_matches_finite_differences(tmp_path):
    """erf' = 2/sqrt(pi)*exp(-x^2) and friends, as the compiled element actually differentiates them."""
    with _NonlinearProblem() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.initialise()
        ndof = problem.ndof()
        base = numpy.linspace(-0.25, 0.25, ndof)

        def residual_at(dofs):
            problem.set_current_dofs(list(dofs))
            return numpy.array(problem.get_residuals())

        problem.set_current_dofs(list(base))
        _, jacobian = problem.assemble_jacobian(True)
        jacobian = numpy.array(jacobian.todense())
        step = 1e-6
        for column in range(ndof):
            forward, backward = base.copy(), base.copy()
            forward[column] += step
            backward[column] -= step
            difference = (residual_at(forward) - residual_at(backward)) / (2 * step)
            assert numpy.max(numpy.abs(jacobian[:, column] - difference)) < 1e-8


def test_erf_of_a_real_argument_splits_trivially():
    x = var("x")
    assert (real_part(erf(x)) - erf(x)).is_zero()
    assert imag_part(erf(x)).is_zero()
    assert (real_part(erfc(x)) - erfc(x)).is_zero()
    assert imag_part(erfc(x)).is_zero()


def test_erf_of_a_complex_argument_is_rejected():
    """
    The real/imaginary parts are those of a *real* argument, so a complex one must not pass silently.

    erf(x+i*y) does not split into elementary functions, and the generated code calls the C erf(double), which cannot
    take a complex argument anyway. minimum/maximum/heaviside declare their imaginary part zero unconditionally; here
    an argument that visibly carries the imaginary unit - which is how a normal-mode perturbation would enter - is
    rejected instead.
    """
    x, y = var("x"), var("y")
    for function in (erf, erfc):
        with pytest.raises(RuntimeError, match="only real arguments are supported"):
            imag_part(function(x + imaginary_i() * y))
        with pytest.raises(RuntimeError, match="only real arguments are supported"):
            real_part(function(x + imaginary_i() * y))


def test_the_real_part_of_a_field_free_expression_is_split_right_away():
    """
    real_part()/imag_part() used to stay held whenever the argument contained no field.

    They are held while the argument still carries unresolved placeholders, since a later stage may
    resolve them better. Once it does not, GiNaC can usually do the split outright, and the result
    has to be taken: a held call is neither separable into units - the residual was rejected with
    "The units of ... cannot be separated from the rest at all" - nor printable as C, since there is
    no get_real_part() in the generated code. It reached every expression whose only variable is a
    global parameter, and even a constant such as real_part(3*mm + 20i*mm).
    """
    def split(expression):
        factor, unit, rest, success = _pyoomph.GiNaC_collect_units(expression)
        assert success, "units and rest are not separable in " + str(expression)
        return factor * unit * rest

    a, b = 3 * milli * meter, 20 * milli * meter
    assert (split(real_part(a + imaginary_i() * b)) - a).is_zero()
    assert (split(imag_part(a + imaginary_i() * b)) - b).is_zero()
    # the same with a global parameter, which is what a real residual looks like
    problem = Problem()
    s = problem.get_global_parameter("s").get_symbol()
    assert (split(real_part(s * meter + imaginary_i() * b)) - s * meter).is_zero()
    assert (split(imag_part(s * meter + imaginary_i() * b)) - b).is_zero()
    assert (split(real_part(heaviside(s - 0.5) * (a + imaginary_i() * b))) - heaviside(s - 0.5) * a).is_zero()


def test_an_incomplete_split_is_still_held():
    """A field placeholder cannot be split yet, so the real_part stays for a later stage to resolve."""
    assert "real_part" in str(real_part(var("u")))
