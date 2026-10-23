#  @file
#  @author Christian Diddens
#  @author Duarte Rocha
#  @author Maxim de Wildt
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

# Global parameters that carry a physical unit (Problem.define_global_parameter(..., unit=...)).
#
# The unit is stored on the parameter and multiplied in where the parameter crosses into an
# expression, so a dimensional parameter rides the ordinary nondimensionalisation and the stored
# double stays the raw number the generated code reads and oomph-lib continues on. The unit
# therefore also fixes the SCALE the parameter is expanded with, which is the point of declaring it
# rather than inferring it: with unit=1*pico*liter the generated C carries 1e-15*meter^3, not a
# litre with a 1e12-larger number. test_generated_code_uses_the_declared_unit_not_the_base_unit is
# the one that would catch that confusion.

import numpy
import pytest

from pyoomph import DirichletBC, ODEEquations, Problem, _pyoomph
from pyoomph.expressions import square_root, symbolic_diff, testfunction, var
from pyoomph.expressions.units import (bar, liter, meter, nano, pascal, pi, pico, second)


class _Scalar(ODEEquations):
    """u fixed by one algebraic residual, so the solved value IS what the generated code computed."""

    def __init__(self, rhs):
        super().__init__()
        self.rhs = rhs

    def define_fields(self):
        self.define_ode_variable("u")

    def define_residuals(self):
        u = var("u")
        self.add_residual((u - self.rhs) * testfunction(u))


class _VolumeProblem(Problem):
    """volume declared in picolitres; u is the parameter measured in litres, hence dimensionless."""

    def define_problem(self):
        self.V = self.define_global_parameter(volume=0 * liter, unit=1 * pico * liter)
        self.add_equations(_Scalar(self.V / (1 * liter)) @ "ode")


def _solved_u(problem):
    return problem.get_ode("ode").get_value("u", dimensional=False)


# ---------------------------------------------------------------------------------------------
# the value round trip
# ---------------------------------------------------------------------------------------------

def test_value_round_trips_dimensionally():
    problem = Problem()
    volume = problem.define_global_parameter(volume=0 * liter, unit=1 * pico * liter)
    volume.value = 1.53 * pico * liter
    assert volume.nondim_value == pytest.approx(1.53)
    assert float(volume.value / (pico * liter)) == pytest.approx(1.53)
    assert float(volume.value / liter) == pytest.approx(1.53e-12)
    # the same volume written in another prefix is the same stored number
    volume.value = 3 * nano * liter
    assert volume.nondim_value == pytest.approx(3000.0)


def test_a_plain_number_is_read_as_being_in_the_declared_unit():
    problem = Problem()
    volume = problem.define_global_parameter(volume=0 * liter, unit=1 * pico * liter)
    volume.value = 7
    assert volume.nondim_value == pytest.approx(7.0)
    assert float(volume.value / (pico * liter)) == pytest.approx(7.0)
    assert float(volume) == pytest.approx(7.0)


def test_a_dimensionless_value_may_be_a_whole_expression():
    """collect_base_units leaves pi*sqrt(2) in `rest`, which still has to evaluate to a number."""
    problem = Problem()
    pressure = problem.define_global_parameter(p=0 * pascal, unit=pascal)
    pressure.value = 4.5 * pi * square_root(2) * pascal
    assert pressure.nondim_value == pytest.approx(4.5 * numpy.pi * numpy.sqrt(2))


def test_the_unit_may_carry_its_own_scale_factor():
    """unit=bar, so the stored number is in bars while the value reads back in pascals."""
    problem = Problem()
    pressure = problem.define_global_parameter(p=2 * bar, unit=bar)
    assert pressure.nondim_value == pytest.approx(2.0)
    assert float(pressure.value / pascal) == pytest.approx(2e5)


def test_dimensionalise_and_nondimensionalise_are_inverse():
    problem = Problem()
    volume = problem.define_global_parameter(volume=0 * liter, unit=1 * pico * liter)
    assert volume.nondimensionalise(2 * pico * liter) == pytest.approx(2.0)
    assert float(volume.dimensionalise(2.0) / (pico * liter)) == pytest.approx(2.0)
    # a plain number passes through untouched, in both directions
    assert volume.nondimensionalise(2.0) == pytest.approx(2.0)


# ---------------------------------------------------------------------------------------------
# what is refused
# ---------------------------------------------------------------------------------------------

def test_a_value_of_the_wrong_dimension_is_refused_and_says_which_unit_it_wanted():
    problem = Problem()
    volume = problem.define_global_parameter(volume=0 * liter, unit=1 * pico * liter)
    with pytest.raises(RuntimeError, match="meter"):
        volume.value = 1.53 * pascal
    assert volume.nondim_value == 0.0, "the refused assignment must not have changed anything"


def test_positivity_is_enforced_on_a_dimensional_value_too():
    """The check is on the stored number, which is where the sign lives: base units are positive
    GiNaC symbols, so dividing a dimensional value by the unit cannot change its sign."""
    problem = Problem()
    pressure = problem.define_global_parameter(p=1 * bar, unit=bar)
    pressure.restrict_to_positive_values()
    with pytest.raises(RuntimeError, match="restricted to positive values"):
        pressure.value = -2 * bar
    assert pressure.nondim_value == pytest.approx(1.0), "the refused assignment changed nothing"


def test_a_dimensional_value_on_a_dimensionless_parameter_is_refused():
    problem = Problem()
    reynolds = problem.define_global_parameter(Re=1.0)
    with pytest.raises(RuntimeError, match="dimensionless"):
        reynolds.value = 2 * pascal


def test_a_dimensional_initial_value_without_a_unit_is_refused():
    """The unit also fixes the scale, so it must not depend on whatever the first value happened to be."""
    problem = Problem()
    with pytest.raises(RuntimeError, match="no unit"):
        problem.define_global_parameter(p=100 * pascal)


def test_the_unit_cannot_be_changed():
    problem = Problem()
    volume = problem.define_global_parameter(volume=0 * liter, unit=1 * pico * liter)
    with pytest.raises(RuntimeError, match="cannot be changed"):
        volume._set_unit(1 * nano * liter)
    # the same unit again is not a change
    volume._set_unit(1 * pico * liter)
    assert float(volume.unit / (pico * liter)) == pytest.approx(1.0)


def test_a_unit_cannot_be_given_after_initialisation():
    """The generated code has baked the unit's numerical factor in by then."""
    with _VolumeProblem() as problem:
        problem.initialise()
        with pytest.raises(RuntimeError, match="after the problem has been initialised"):
            problem.define_global_parameter(late=1.0, unit=pascal)


def test_unit_is_a_reserved_keyword_rather_than_a_silent_no_op():
    problem = Problem()
    with pytest.raises(RuntimeError, match="cannot be named 'unit'"):
        problem.define_global_parameter(unit=5)


def test_one_unit_cannot_serve_several_parameters_at_once():
    problem = Problem()
    with pytest.raises(RuntimeError, match="once per parameter"):
        problem.define_global_parameter(a=1, b=2, unit=pascal)


def test_a_unit_has_to_be_a_number_times_base_units():
    problem = Problem()
    parameter = problem.get_global_parameter("weird")
    with pytest.raises(RuntimeError, match="not a plain number times base units"):
        parameter._set_unit(var("x") * meter)


# ---------------------------------------------------------------------------------------------
# the generated code
# ---------------------------------------------------------------------------------------------

def test_generated_code_uses_the_declared_unit_not_the_base_unit(tmp_path):
    """3 picolitres measured in litres is 3e-12. Reading the unit as a litre would give 3.

    The two differ by 1e12, so this is the test that says the `unit` keyword fixes the scale and not
    merely the dimension.
    """
    with _VolumeProblem() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.initialise()
        problem.V.value = 3 * pico * liter
        problem.solve()
        assert _solved_u(problem) == pytest.approx(3e-12, rel=1e-9)


def test_a_wrongly_dimensioned_residual_is_still_rejected(tmp_path):
    """The parameter's unit takes part in the ordinary dimensional analysis, errors included."""

    class _Wrong(Problem):
        def define_problem(self):
            volume = self.define_global_parameter(volume=1 * pico * liter, unit=1 * pico * liter)
            # a volume is not a dimensionless number, and no scale can make this contribution so
            self.add_equations(_Scalar(volume) @ "ode")

    with _Wrong() as problem:
        problem.set_output_directory(str(tmp_path))
        with pytest.raises(Exception, match="dimensionless"):
            problem.initialise()


def test_the_parameter_derivative_is_with_respect_to_the_stored_number(tmp_path):
    """dResidual/dParameter differentiates the bare leaf, so it is dR/d(nondim_value).

    That is what oomph-lib's continuation needs, since it steps that same stored double - so the
    finite difference this is compared against has to step nondim_value too.
    """
    with _VolumeProblem() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.initialise()
        problem.V.value = 3 * pico * liter
        problem.solve()
        analytic = numpy.array(problem.get_parameter_derivative("volume"))
        epsilon = 1e-6
        start = problem.V.nondim_value
        problem.V.nondim_value = start + epsilon
        plus = numpy.array(problem.get_residuals())
        problem.V.nondim_value = start - epsilon
        minus = numpy.array(problem.get_residuals())
        problem.V.nondim_value = start
        difference = (plus - minus) / (2 * epsilon)
        scale = max(numpy.max(numpy.abs(difference)), numpy.max(numpy.abs(analytic)), 1e-30)
        assert numpy.max(numpy.abs(analytic - difference)) / scale < 1e-5


@pytest.mark.parametrize("hold", [False, True])
def test_symbolic_diff_by_a_dimensional_parameter_carries_the_inverse_unit(hold):
    """d/dP of 3*P^2 is 6*P, which for a pressure P means the result is a pressure.

    Both routes are checked because they reach it differently: the held one goes through the by-name
    substitution, which divides a replacement by the unit, and the in-place one does not.
    """
    problem = Problem()
    pressure = problem.define_global_parameter(p=2 * bar, unit=bar)
    derivative = symbolic_diff(3 * (0 + pressure) ** 2, pressure, hold_until_codegen=hold)
    assert (_pyoomph.Expression(derivative) - 6 * (0 + pressure)).is_zero()


def test_substituting_a_parameter_by_name_takes_a_dimensional_value():
    """expr(p=5*bar) means "p is 5 bar", the same reading a field substitution has."""
    problem = Problem()
    pressure = problem.define_global_parameter(p=2 * bar, unit=bar)
    expression = 3 * (0 + pressure)
    assert (_pyoomph.Expression(expression(p=5 * bar)) - 15 * bar).is_zero()


def test_substituting_a_parameter_by_another_parameter_keeps_both_units():
    problem = Problem()
    pressure = problem.define_global_parameter(p=2 * bar, unit=bar)
    other = problem.define_global_parameter(q=3 * pascal, unit=pascal)
    expression = 3 * (0 + pressure)
    assert (_pyoomph.Expression(expression(p=other)) - 3 * (0 + other)).is_zero()


def test_substituting_a_dimensionless_parameter_is_unchanged():
    problem = Problem()
    reynolds = problem.define_global_parameter(Re=1.0)
    expression = 3 * (0 + reynolds)
    assert (_pyoomph.Expression(expression(Re=5)) - 15).is_zero()


# ---------------------------------------------------------------------------------------------
# continuation and state
# ---------------------------------------------------------------------------------------------

def test_go_to_param_accepts_a_dimensional_target(tmp_path):
    with _VolumeProblem() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.initialise()
        problem.V.value = 1 * pico * liter
        problem.solve()
        problem.go_to_param(volume=10 * pico * liter)
        assert problem.V.nondim_value == pytest.approx(10.0)
        assert float(problem.V.value / (pico * liter)) == pytest.approx(10.0)


def test_arclength_continuation_accepts_a_dimensional_step_and_returns_a_number(tmp_path):
    """The step is dimensional on the way in; the returned ds is a plain number, so that the usual
    `ds = problem.arclength_continuation(p, ds, ...)` loop keeps doing arithmetic on numbers."""
    with _VolumeProblem() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.initialise()
        problem.V.value = 1 * pico * liter
        problem.solve()
        step = problem.arclength_continuation("volume", 1 * pico * liter, max_ds=2 * pico * liter)
        assert isinstance(step, float)
        assert problem.V.nondim_value > 1.0
        # and the returned number can be fed straight back in
        problem.arclength_continuation("volume", step)


def test_state_files_hold_the_stored_number_and_round_trip(tmp_path):
    with _VolumeProblem() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.initialise()
        problem.V.value = 2.5 * pico * liter
        problem.solve()
        problem.save_state(str(tmp_path / "s.dump"))
        problem.V.value = 9 * pico * liter
        problem.load_state(str(tmp_path / "s.dump"))
        assert problem.V.nondim_value == pytest.approx(2.5)
        assert float(problem.V.value / (pico * liter)) == pytest.approx(2.5)


# ---------------------------------------------------------------------------------------------
# a dimensionless parameter behaves exactly as it always did
# ---------------------------------------------------------------------------------------------

def test_a_dimensionless_parameter_is_unchanged():
    problem = Problem()
    reynolds = problem.define_global_parameter(Re=100.0)
    assert isinstance(reynolds.value, float)
    assert reynolds.value == 100.0
    assert reynolds.nondim_value == 100.0
    assert float(reynolds) == 100.0
    assert reynolds.unit.is_equal(_pyoomph.Expression(1))
    reynolds.value = 2.5
    assert reynolds.value == 2.5


@pytest.mark.parametrize("unit", [1, pascal])
def test_every_operator_keeps_the_unit(unit):
    """The unit is multiplied in at one boundary, and every operator has to cross it.

    A forgotten one drops the unit silently - the expression is still valid, just wrong - so each
    form is checked against the same expression written with get_symbol() instead.
    """
    problem = Problem()
    parameter = problem.define_global_parameter(p=1.0, unit=unit)
    symbol = parameter.get_symbol()
    other = problem.define_global_parameter(q=1.0, unit=unit)
    other_symbol = other.get_symbol()
    expression = 3 * meter

    cases = [
        (-parameter, -symbol),
        (parameter + other, symbol + other_symbol),
        (parameter - other, symbol - other_symbol),
        (parameter * other, symbol * other_symbol),
        (parameter / other, symbol / other_symbol),
        (parameter + expression, symbol + expression),
        (parameter - expression, symbol - expression),
        (parameter * expression, symbol * expression),
        (parameter / expression, symbol / expression),
        (expression + parameter, expression + symbol),
        (expression - parameter, expression - symbol),
        (expression * parameter, expression * symbol),
        (expression / parameter, expression / symbol),
        (parameter + 2, symbol + 2),
        (parameter - 2, symbol - 2),
        (parameter * 2, symbol * 2),
        (parameter / 2, symbol / 2),
        (2 + parameter, 2 + symbol),
        (2 - parameter, 2 - symbol),
        (2 * parameter, 2 * symbol),
        (2 / parameter, 2 / symbol),
        (parameter + 2.5, symbol + 2.5),
        (parameter - 2.5, symbol - 2.5),
        (parameter * 2.5, symbol * 2.5),
        (parameter / 2.5, symbol / 2.5),
        (2.5 + parameter, 2.5 + symbol),
        (2.5 - parameter, 2.5 - symbol),
        (2.5 * parameter, 2.5 * symbol),
        (2.5 / parameter, 2.5 / symbol),
        (parameter ** 2, symbol ** 2),
        (parameter ** 2.0, symbol ** 2.0),
        (_pyoomph.Expression(parameter), symbol),
    ]
    for index, (actual, expected) in enumerate(cases):
        assert (_pyoomph.Expression(actual) - expected).is_zero(), (
            "case %d: %s != %s" % (index, actual, expected))


def test_a_dimensionless_exponent_base_still_works():
    """2**p and p**2 both exist; only the former makes a dimensional p meaningless, so p is plain here."""
    problem = Problem()
    parameter = problem.define_global_parameter(p=1.0)
    symbol = parameter.get_symbol()
    assert (_pyoomph.Expression(2 ** parameter) - 2 ** symbol).is_zero()
    assert (_pyoomph.Expression(2.0 ** parameter) - 2.0 ** symbol).is_zero()


def test_a_dimensional_parameter_still_gets_the_reciprocal_rewrite(tmp_path):
    """ParameterReciprocals matches power(leaf, negint), and the unit must not hide that.

    By the time it runs the units are stripped, and GiNaC distributes an integer power over a
    product, so (unit*leaf)^-1 still contains leaf^-1 - but that is an argument, not a measurement.
    This asserts the rewrite reaches the residual: _rgp_<i> is DECLARED for every parameter in scope
    whether or not it fires, so the declaration alone would prove nothing.
    """
    import glob
    import os

    class _Reciprocal(Problem):
        def define_problem(self):
            volume = self.define_global_parameter(volume=1 * pico * liter, unit=1 * pico * liter)
            self.add_equations(_Scalar((1 * liter) / volume) @ "ode")

    with _Reciprocal() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.initialise()

    sources = glob.glob(os.path.join(str(tmp_path), "_ccode", "*.c"))
    assert sources, "no generated C was written"
    residuals = [line for path in sources for line in open(path).read().split("\n")
                 if "BEGIN_RESIDUAL" in line]
    assert residuals, "no residual was emitted"
    assert any("_rgp_" in line for line in residuals), (
        "the parameter reciprocal did not reach the residual:\n" + "\n".join(residuals))


def test_the_bifurcation_gui_labels_a_parameter_axis_with_its_unit():
    """A recorded point stores nondim_value, so the label is the declared unit, factor and all.

    1e-15 m^3 is a picolitre and is not a prefixed cubic metre, so the factor is written out rather
    than folded into a prefix. A dimensionless parameter keeps its bare name, which used to be the
    only case - see axis_label.
    """
    from pyoomph.utils.bifurcation_gui.controller import BifurcationController
    from pyoomph.utils.bifurcation_gui.model import parameter_axis

    with _VolumeProblem() as problem:
        problem.initialise()
        problem.define_global_parameter(plain=1.0)
        controller = BifurcationController.__new__(BifurcationController)
        controller.problem = problem
        controller._on_log = lambda *a: None
        assert controller.parameter_unit("volume") == "1e-15 m^3"
        assert controller.parameter_unit("plain") == ""
        assert controller.axis_label(parameter_axis("volume")) == "volume [1e-15 m^3]"
        assert controller.axis_label(parameter_axis("plain")) == "plain"
