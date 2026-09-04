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


# Symbolic comparisons: var("time") < 2*second and friends.
#
# Python's <, <=, > and >= on an Expression return a held RelationalExpression instead of a bool, so
# that the comparison survives until either the expression is called with concrete values or it is
# compiled. conditional(cond, iftrue, iffalse) turns it into piecewise_gt0/piecewise_geq0 of the
# difference of both sides. Python's own ternary "a if cond else b" cannot be used for this: CPython
# always casts the condition to a bool, so the branch would be picked immediately.

import pytest

from pyoomph import ODEEquations, Problem, _pyoomph
from pyoomph.expressions import conditional, imaginary_i, logical_and, logical_not, logical_or, logical_xor, piecewise_geq0, piecewise_gt0, real_part, subexpression, var, var_and_test
from pyoomph.expressions.units import meter, milli, second


def _split(expression):
    """(factor, unit, rest) of the unit split, as the code generation would perform it."""
    factor, unit, rest, success = _pyoomph.GiNaC_collect_units(expression)
    assert success, "units and rest are not separable in " + str(expression)
    return factor, unit, rest


def test_comparison_stays_symbolic():
    """A comparison of two expressions is held, not decided."""
    relation = var("time") < 2 * second
    assert isinstance(relation, _pyoomph.RelationalExpression)
    assert relation.operator_string() == "<"
    assert (relation.rhs - 2 * second).is_zero()
    # ... and so is the conditional built from it
    expression = conditional(relation, 1, 2)
    assert not expression.is_zero()
    assert "piecewise_gt0" in str(expression)


@pytest.mark.parametrize(
    "operator,at_1s,at_2s,at_4s",
    [("<", 1, 2, 2), ("<=", 1, 1, 2), (">", 2, 2, 1), (">=", 2, 1, 1)],
)
def test_substituting_the_condition_selects_the_branch(operator, at_1s, at_2s, at_4s):
    """expression(time=...) resolves the comparison, including the boundary case time==2*second."""
    time, threshold = var("time"), 2 * second
    relation = {"<": time < threshold, "<=": time <= threshold, ">": time > threshold, ">=": time >= threshold}[operator]
    expression = conditional(relation, 1, 2)
    for value, expected in ((1, at_1s), (2, at_2s), (4, at_4s)):
        assert (expression(time=value * second) - expected).is_zero(), operator + " at " + str(value) + "*second"


def test_the_comparison_may_be_written_either_way_round():
    forward = conditional(var("time") < 2 * second, 1, 2)
    reflected = conditional(2 * second > var("time"), 1, 2)
    assert (forward - reflected).is_zero()


def test_units_of_both_sides_must_match():
    """Both sides are subtracted from each other, so a unit mismatch is rejected as in any sum."""
    with pytest.raises(RuntimeError, match="Cannot extract the unit from the condition"):
        _split(conditional(var("t") * second < 2 * meter, 1, 2))


def test_the_unit_of_the_condition_drops_out():
    """As for piecewise_geq0: only the sign of the condition matters, so its unit and scale divide out."""
    factor, unit, rest = _split(conditional(var("t") * second < 2 * second, 1 * milli * meter, 2 * meter))
    assert (factor * unit - 2 * meter).is_zero()
    assert (rest - piecewise_gt0(1 - var("t") / 2, 1 * milli * meter / (2 * meter), 1)).is_zero()


def test_a_decidable_comparison_still_converts_to_a_bool():
    """Existing numeric comparisons keep working, i.e. they may be used in an if statement."""
    assert _pyoomph.Expression(3.0) < 5.0
    assert not (_pyoomph.Expression(3.0) > 5.0)
    assert _pyoomph.Expression(3.0) <= 3.0
    # dimensional, but still decidable, since base units are positive symbols
    assert 1 * second < 2 * second
    assert not (2 * second < 2 * second)


def test_an_undecidable_comparison_refuses_to_become_a_bool():
    with pytest.raises(RuntimeError, match="conditional"):
        bool(var("time") < 2 * second)


def test_conditional_accepts_a_plain_bool_or_expression():
    """A Python bool selects immediately, a bare expression is tested for being >=0."""
    assert (conditional(True, 2 * meter, 3 * meter) - 2 * meter).is_zero()
    assert (conditional(False, 2 * meter, 3 * meter) - 3 * meter).is_zero()
    assert (conditional(var("x") - 1, 2 * meter, 0) - piecewise_geq0(var("x") - 1, 2 * meter, 0)).is_zero()


def test_piecewise_gt0_differs_from_geq0_only_at_zero():
    assert (piecewise_gt0(0, 2 * meter, 3 * meter) - 3 * meter).is_zero()
    assert (piecewise_geq0(0, 2 * meter, 3 * meter) - 2 * meter).is_zero()
    assert (piecewise_gt0(1, 2 * meter, 3 * meter) - 2 * meter).is_zero()
    assert (piecewise_gt0(-1, 2 * meter, 3 * meter) - 3 * meter).is_zero()


class _Compared(ODEEquations):
    """u is 3*mm while the parameter s stays below the threshold, and 20*um once it is above."""

    def __init__(self, threshold_relation):
        super().__init__()
        self.threshold_relation = threshold_relation

    def define_fields(self):
        self.define_ode_variable("u", scale=meter, testscale=1 / meter)

    def define_residuals(self):
        u, v = var_and_test("u")
        branch = conditional(self.threshold_relation, 3 * milli * meter, 20 * milli * meter)
        self.add_residual((u - branch) * v)


class _ComparedProblem(Problem):
    def __init__(self, strict):
        super().__init__()
        self.strict = strict

    def define_problem(self):
        s = self.get_global_parameter("s").get_symbol() * meter
        relation = (s < 0.5 * meter) if self.strict else (s <= 0.5 * meter)
        self.add_equations(_Compared(relation) @ "ode")


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("parameter_value,expected", [(0.1, 3e-3), (0.9, 20e-3)])
def test_the_generated_code_evaluates_the_comparison(tmp_path, strict, parameter_value, expected):
    """
    The comparison must survive into the generated code as a C ternary.

    The threshold is compared against a global parameter, so that the condition is not folded away
    while the expression is still symbolic.
    """
    with _ComparedProblem(strict) as problem:
        problem.set_output_directory(str(tmp_path))
        problem.get_global_parameter("s").value = parameter_value
        problem.solve()
        value = float(problem.get_ode("ode").get_value("u", dimensional=True) / meter)
    assert value == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("strict,expected", [(True, 20e-3), (False, 3e-3)])
def test_the_generated_code_agrees_on_the_boundary(tmp_path, strict, expected):
    """At exactly the threshold, "<" is false and "<=" is true, in the generated code as well."""
    with _ComparedProblem(strict) as problem:
        problem.set_output_directory(str(tmp_path))
        problem.get_global_parameter("s").value = 0.5
        problem.solve()
        value = float(problem.get_ode("ode").get_value("u", dimensional=True) / meter)
    assert value == pytest.approx(expected, rel=1e-12)


def test_identical_branches_collapse():
    """A condition that selects between two equal values is irrelevant and is dropped right away.

    Differentiating a branch-wise constant produces exactly this, and the ternary would otherwise
    survive into the generated code as e.g. "(cond >0 ? 0.0 : 0.0)".
    """
    u = var("u")
    assert (conditional(u < 1, 5 * meter, 5 * meter) - 5 * meter).is_zero()
    assert str(conditional(u < 1, 5 * meter, 5 * meter)).find("piecewise") < 0


def test_real_and_imaginary_part_distribute_over_the_branches():
    """
    Only the sign of the condition matters, so the split simply goes into both branches.

    Without this, real_part() stays wrapped around the held call and the normal-mode analysis cannot
    resolve it, which used to make the whole expression unit-inseparable.
    """
    u = var("u")
    branch = conditional(u < 1, 2 * u + imaginary_i() * u, 3 * u)
    split = real_part(branch)
    assert (split - conditional(u < 1, real_part(2 * u + imaginary_i() * u), real_part(3 * u))).is_zero()
    assert "real_part(piecewise" not in str(split)


class _NewtonBranch(ODEEquations):
    """Branch-linear in u, so an exact Jacobian solves it in a single Newton step."""

    def __init__(self, use_subexpression):
        super().__init__()
        self.use_subexpression = use_subexpression

    def define_fields(self):
        self.define_ode_variable("u", scale=meter, testscale=1 / meter)

    def define_residuals(self):
        u, v = var_and_test("u")
        branch = conditional(u < 0.5 * meter, 0.1 * meter + 0.2 * u, 5 * u - 2 * meter)
        if self.use_subexpression:
            branch = subexpression(branch)
        self.add_residual((u - branch) * v)


class _NewtonProblem(Problem):
    def __init__(self, use_subexpression):
        super().__init__()
        self.use_subexpression = use_subexpression

    def define_problem(self):
        self.add_equations(_NewtonBranch(self.use_subexpression) @ "ode")


@pytest.mark.parametrize("use_subexpression", [False, True])
def test_the_jacobian_branches_along_with_the_residual(tmp_path, use_subexpression):
    """
    The derivative of a conditional is the conditional of the derivatives, i.e. a ternary again.

    The residual is linear within each branch, so an exact Jacobian converges in a single Newton
    step - a Jacobian that missed the branch would not. The same is checked once with the whole
    conditional hidden inside a subexpression, where the derivative becomes a separate ternary for
    d_subexpr_/d_u in the generated code.
    """
    with _NewtonProblem(use_subexpression) as problem:
        problem.set_output_directory(str(tmp_path))
        problem.max_newton_iterations = 1
        problem.solve()
        assert float(problem.get_ode("ode").get_value("u", dimensional=True) / meter) == pytest.approx(0.125, rel=1e-10)
    generated = (tmp_path / "_ccode" / "ode.c").read_text()
    jacobian_lines = [line for line in generated.splitlines() if "BEGIN_JACOBIAN" in line]
    assert jacobian_lines, "no Jacobian was generated"
    if use_subexpression:
        # the ternary sits in the derivative of the subexpression rather than in the Jacobian line itself
        assert any("?" in line and "d_subexpr" in line for line in generated.splitlines())
    else:
        assert any("?" in line for line in jacobian_lines)


def test_equality_is_tested_explicitly_rather_than_with_the_operator():
    """
    == stays an identity comparison, so that expressions remain usable in dicts, sets and "in" tests.

    is_equal() compares the expression trees structurally, while (a-b).is_zero() simplifies first.
    """
    u = var("u")
    assert (2 * u).is_equal(u * 2)
    assert not (2 * u).is_equal(3 * u)
    assert ((2 * u) - (u + u)).is_zero()
    # the operator itself is untouched, i.e. two separate wrappers of the same expression differ
    assert (2 * u) != (u * 2)
    assert len({u, 2 * u}) == 2


# --- combining conditions with ~, &, | and ^ ------------------------------------------------------
#
# Each comparison is evaluated to a 0/1 indicator and those are combined arithmetically (1-p, p*q,
# p+q-p*q, p+q-2*p*q), so that the whole condition stays a single sign test. Only the comparisons are
# combined this way; the two branch values remain a proper ternary and are never both evaluated.


def _truth_table(condition_of, values=(0.5, 1.0, 1.5, 2.0, 2.5)):
    """Which branch conditional(...) picks at each of the given times."""
    expression = conditional(condition_of(var("time")), 1, 0)
    return [int(expression(time=value * second)) for value in values]


def test_and_or_not_and_xor_pick_the_right_branch():
    lower, upper = 1 * second, 2 * second
    assert _truth_table(lambda t: (t > lower) & (t < upper)) == [0, 0, 1, 0, 0]
    assert _truth_table(lambda t: (t < lower) | (t > upper)) == [1, 0, 0, 0, 1]
    assert _truth_table(lambda t: ~(t < lower)) == [0, 1, 1, 1, 1]
    # exactly one of "below 1s" and "below 2s" holds only in between
    assert _truth_table(lambda t: (t < lower) ^ (t < upper)) == [0, 1, 1, 0, 0]
    # double negation, and a combination of combinations
    assert _truth_table(lambda t: ~~(t < lower)) == [1, 0, 0, 0, 0]
    assert _truth_table(lambda t: ((t > lower) & (t < upper)) | (t > 2.4 * second)) == [0, 0, 1, 0, 1]


def test_the_named_helpers_match_the_operators():
    t = var("time")
    lower, upper = 1 * second, 2 * second
    pairs = [
        (logical_and(t > lower, t < upper), (t > lower) & (t < upper)),
        (logical_or(t < lower, t > upper), (t < lower) | (t > upper)),
        (logical_not(t < lower), ~(t < lower)),
        (logical_xor(t < lower, t < upper), (t < lower) ^ (t < upper)),
    ]
    for named, operator in pairs:
        assert (conditional(named, 1, 2) - conditional(operator, 1, 2)).is_zero()
    # and they take more than two operands
    assert _truth_table(lambda t: logical_and(t > lower, t < upper, t > 1.2 * second)) == [0, 0, 1, 0, 0]


def test_conditions_only_combine_with_each_other():
    """A bare expression would be ambiguous, so it has to be turned into a condition explicitly."""
    t = var("time")
    with pytest.raises(TypeError, match="expression>=0"):
        logical_and(t > 1 * second, t)
    with pytest.raises(TypeError):
        (t > 1 * second) & t


def test_a_combined_condition_converts_to_a_bool_when_decidable():
    t = var("time")
    assert (1 * second < 2 * second) & (2 * second < 3 * second)
    assert not ((1 * second < 2 * second) & (3 * second < 2 * second))
    assert (1 * second > 2 * second) | (2 * second < 3 * second)
    assert ~(1 * second > 2 * second)
    # and/or short-circuit as their Python counterparts do, i.e. a decidable false wins over an
    # undecidable second operand
    assert not ((1 * second > 2 * second) & (t < 1 * second))
    assert (1 * second < 2 * second) | (t < 1 * second)


def test_the_error_message_points_at_the_bitwise_operators():
    """not/and/or and chained comparisons all end up in __bool__, which is where they are caught."""
    t = var("time")
    with pytest.raises(RuntimeError, match=r"\(a < b\) & \(b < c\)"):
        bool((t > 1 * second) & (t < 2 * second))
    with pytest.raises(RuntimeError, match="logical_and"):
        1 * second < t < 2 * second


def test_the_indicator_is_one_where_the_condition_holds():
    t = var("time")
    indicator = ((t > 1 * second) & (t < 2 * second)).indicator()
    assert (indicator(time=1.5 * second) - 1).is_zero()
    assert indicator(time=0.5 * second).is_zero()


class _BandBranch(ODEEquations):
    """Branch-linear in u, but selected by a two-sided condition on a global parameter."""

    def define_fields(self):
        self.define_ode_variable("u", scale=meter, testscale=1 / meter)

    def define_residuals(self):
        u, v = var_and_test("u")
        s = self.get_current_code_generator().get_problem().get_global_parameter("s").get_symbol() * meter
        inside = (s > 0.2 * meter) & (s < 0.8 * meter)
        self.add_residual((u - conditional(inside, 0.1 * meter + 0.2 * u, 5 * u - 2 * meter)) * v)


class _BandProblem(Problem):
    def define_problem(self):
        self.add_equations(_BandBranch() @ "ode")


@pytest.mark.parametrize("parameter_value,expected", [(0.5, 0.125), (0.9, 0.5)])
def test_a_combined_condition_reaches_the_generated_code(tmp_path, parameter_value, expected):
    """
    The indicators become ternaries of their own, multiplied into one outer sign test.

    Both branches are linear in u, so an exact Jacobian - which must branch on the very same
    condition - converges in a single Newton step.
    """
    with _BandProblem() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.get_global_parameter("s").value = parameter_value
        problem.max_newton_iterations = 1
        problem.solve()
        assert float(problem.get_ode("ode").get_value("u", dimensional=True) / meter) == pytest.approx(expected, rel=1e-10)
    generated = (tmp_path / "_ccode" / "ode.c").read_text()
    jacobian_lines = [line for line in generated.splitlines() if "BEGIN_JACOBIAN_NOHANG" in line]
    assert jacobian_lines and jacobian_lines[0].count("?") == 3, "expected two indicators plus the branch selection"
