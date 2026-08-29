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

# Units in piecewise_geq0(cond, a, b).
#
# The three arguments have unrelated dimensions: only the sign of cond selects a branch, so its
# unit and (positive) scale divide out, whereas a and b are alternatives of one quantity and must
# therefore agree. Before collect_base_units() knew about the function, any dimensional argument
# was left inside the call by the generic-function fallback and the residual was then rejected with
# "CANNOT SEPARATE UNITS AND REST" - the branch had to be written with heaviside() instead.

import pytest

from pyoomph import ODEEquations, Problem, _pyoomph
from pyoomph.expressions import heaviside, piecewise_geq0, var, var_and_test
from pyoomph.expressions.units import meter, micro, milli, second


def _split(expression):
    """(factor, unit, rest) of the unit split, as the code generation would perform it."""
    factor, unit, rest, success = _pyoomph.GiNaC_collect_units(expression)
    assert success, "units and rest are not separable in " + str(expression)
    return factor, unit, rest


def test_condition_units_drop_out_of_piecewise():
    """The condition may carry any unit, independently of the branches."""
    x = var("x")
    factor, unit, rest = _split(piecewise_geq0(x * second, 1 * milli * meter, 2 * meter))
    assert (factor * unit - 2 * meter).is_zero()
    # normalized against the dominant branch, and with the condition stripped of its second
    assert (rest - piecewise_geq0(x, 1 * milli * meter / (2 * meter), 1)).is_zero()


def test_piecewise_branch_units_are_the_result_units():
    x = var("x")
    factor, unit, rest = _split(piecewise_geq0(x * meter - 0.5 * meter, 2 * second, 3 * second) / meter)
    assert (factor * unit - 3 * second / meter).is_zero()
    assert (rest - piecewise_geq0(x - 0.5, 2 * second / (3 * second), 1)).is_zero()


def test_zero_branch_takes_any_unit():
    """A plain 0 is compatible with every unit, as it is in a sum."""
    x = var("x")
    for expression in (piecewise_geq0(x * meter, 1 * meter, 0), piecewise_geq0(x * meter, 0, 1 * meter)):
        factor, unit, _ = _split(expression)
        assert (factor * unit - meter).is_zero()
    # ... and two zero branches make the whole expression vanish
    _, _, rest = _split(piecewise_geq0(x * meter, 0 * meter, 0 * meter))
    assert rest.is_zero()


def test_piecewise_with_inconsistent_branch_units_is_rejected():
    x = var("x")
    with pytest.raises(RuntimeError, match="Nonmatching units in the two branches"):
        _split(piecewise_geq0(x * meter, 2 * second, 3 * meter))
    # the condition is a sum and must still be consistent within itself
    with pytest.raises(RuntimeError, match="Cannot extract the unit from the condition"):
        _split(piecewise_geq0(x * meter - 1 * second, 2 * meter, 3 * meter))


def test_a_vanishing_condition_takes_the_geq0_branch():
    """
    piecewise_geq0(0,a,b) is a, here as well as in the generated code.

    The symbolic evaluation used to decide the zero case with the strict numeric::is_positive() and
    hence returned b, while the generated C ternary tests ">= 0" and returned a - the same
    expression thus gave two different answers depending on whether it had been folded away before
    code generation or not.
    """
    assert (piecewise_geq0(0, 2 * meter, 3 * meter) - 2 * meter).is_zero()
    assert (piecewise_geq0(0 * second, 2 * meter, 3 * meter) - 2 * meter).is_zero()
    # a dimensional condition stays held until the units are collected, and is decided only then
    factor, unit, rest = _split(piecewise_geq0(-1 * micro * meter, 2 * meter, 3 * meter))
    assert (factor * unit * rest - 3 * meter).is_zero()


class _Branches(ODEEquations):
    """u_pw and u_hv are the same branch, written once with piecewise_geq0 and once with heaviside."""

    def __init__(self, scale, condition):
        super().__init__()
        self.scale = scale
        self.condition = condition

    def define_fields(self):
        for name in ("u_pw", "u_hv"):
            self.define_ode_variable(name, scale=self.scale, testscale=1 / self.scale)

    def define_residuals(self):
        condition = self.condition * meter - 0.5 * meter
        a, b = 3 * milli * meter, 20 * micro * meter
        u, v = var_and_test("u_pw")
        self.add_residual((u - piecewise_geq0(condition, a, b)) * v)
        u, v = var_and_test("u_hv")
        step = heaviside(condition)
        self.add_residual((u - (step * a + (1 - step) * b)) * v)


class _BranchProblem(Problem):
    """`condition` is a number, or None to take it from a global parameter, i.e. only at runtime."""

    def __init__(self, scale, condition):
        super().__init__()
        self.scale = scale
        self.condition = condition

    def define_problem(self):
        condition = self.condition
        if condition is None:
            condition = self.get_global_parameter("s").get_symbol()
        self.add_equations(_Branches(self.scale, condition) @ "ode")


def _solved_branch_values(tmp_path, scale, condition, parameter_value=None):
    with _BranchProblem(scale, condition) as problem:
        problem.set_output_directory(str(tmp_path))
        if parameter_value is not None:
            problem.get_global_parameter("s").value = parameter_value
        problem.solve()
        ode = problem.get_ode("ode")
        return {name: float(ode.get_value(name, dimensional=True) / meter) for name in ("u_pw", "u_hv")}


@pytest.mark.parametrize("condition,expected", [(1.0, 3 * 1e-3), (0.0, 20 * 1e-6)])
@pytest.mark.parametrize("scale", [meter, milli * meter])
def test_compiled_piecewise_matches_heaviside(tmp_path, condition, expected, scale):
    """The generated code must pick the same branch and undo the nondimensionalization it applied."""
    values = _solved_branch_values(tmp_path, scale, condition)
    assert values["u_pw"] == pytest.approx(expected, rel=1e-12)
    assert values["u_hv"] == pytest.approx(expected, rel=1e-12)


def test_compiled_piecewise_agrees_on_a_vanishing_condition(tmp_path):
    """
    The runtime zero case, i.e. the ">= 0" of the C ternary, must agree with the symbolic one.

    The condition comes from a global parameter here, so that it survives to the generated code
    instead of being folded away while the expression is still symbolic.
    """
    values = _solved_branch_values(tmp_path, meter, None, parameter_value=0.5)
    assert values["u_pw"] == pytest.approx(3 * 1e-3, rel=1e-12)
    # heaviside is 1/2 at zero, both symbolically and in the generated code, so it interpolates
    assert values["u_hv"] == pytest.approx(0.5 * (3 * 1e-3 + 20 * 1e-6), rel=1e-12)
