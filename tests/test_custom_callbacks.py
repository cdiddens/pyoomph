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

# A CustomMultiReturnExpression implemented in Python only, i.e. without generate_c_code(), so the
# generated element code cannot inline it and has to call back into Python through
# my_func_table->invoke_multi_ret at every integration point.
#
# That path writes its results into a buffer owned by the C++ side, which is handed to eval() as a
# numpy array. It has to be a VIEW of that buffer. nanobind decides between a view and a copy from
# whether the array has an owner (ndarray_export(): copy = th->owner == nullptr && th->self ==
# nullptr), and with no owner attached the callback filled a copy that nobody ever read back: every
# such callback silently returned zero, and so did every derivative it was asked for. Nothing
# complained -- the residual was simply built from zeros.
#
# The same class WITH a C implementation is used as the control throughout: it exercises the other
# branch of write_code_multi_ret_call, never went through nanobind, and was never affected.

import numpy
import pytest

from pyoomph import Problem, ODEEquations
from pyoomph.expressions import var, var_and_test
from pyoomph.expressions.cb import CustomMultiReturnExpression
from pyoomph.equations.generic import IntegralObservables
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.meshes.simplemeshes import RectangularQuadMesh


class _PythonOnly(CustomMultiReturnExpression):
    """f(a) = (2a, a^2), evaluated in Python. No C code, so the callback is dispatched at runtime."""

    def get_num_returned_scalars(self, nargs):
        return 2

    def eval(self, flag, arg_list, result_list, derivative_matrix):
        result_list[0] = 2 * arg_list[0]
        result_list[1] = arg_list[0] ** 2
        if flag:
            derivative_matrix[0] = 2.0
            derivative_matrix[1] = 2 * arg_list[0]


class _WithCCode(_PythonOnly):
    """The same function, inlined into the generated code instead."""

    def generate_c_code(self):
        return """
        result_list[0] = 2*arg_list[0];
        result_list[1] = arg_list[0]*arg_list[0];
        if (flag) { derivative_matrix[0] = 2.0; derivative_matrix[1] = 2*arg_list[0]; }
        """


class _IntegrateCallbacks(Problem):
    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=1))          # the unit square
        x = var("coordinate_x")
        python_double, python_square = _PythonOnly()(x)
        c_double, c_square = _WithCCode()(x)
        equations = PoissonEquation(source=1)            # only to give the domain a coordinate space
        equations += IntegralObservables(python_double=python_double, python_square=python_square,
                                         c_double=c_double, c_square=c_square)
        self.add_equations(equations @ "domain")


def test_python_callback_results_reach_the_residual(tmp_path):
    """
    Integrals of 2x and x^2 over the unit square, evaluated through both dispatch paths.

    Both callbacks compute the same function, so any difference between the columns is the
    marshalling and nothing else. The exact values are 1 and 1/3, and a callback whose results never
    arrive integrates to exactly zero.
    """
    with _IntegrateCallbacks() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.initialise()
        mesh = problem.get_mesh("domain")
        values = {name: float(mesh.evaluate_observable(name))
                  for name in ("python_double", "python_square", "c_double", "c_square")}
    assert abs(values["c_double"] - 1.0) < 1e-8
    assert abs(values["c_square"] - 1.0 / 3.0) < 1e-8
    assert abs(values["python_double"] - values["c_double"]) < 1e-12
    assert abs(values["python_square"] - values["c_square"]) < 1e-12


class _SquareRootODE(ODEEquations):
    """u^2 = 4, written through a callback, so Newton needs the callback's derivative to get there."""

    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def define_fields(self):
        self.define_ode_variable("u")
        self.set_initial_condition("u", 1.0)

    def define_residuals(self):
        u, u_test = var_and_test("u")
        self.add_weak(self.callback(u)[1] - 4, u_test)   # the second return value is u^2


@pytest.mark.parametrize("with_c_code", [False, True], ids=["python", "c-code"])
def test_python_callback_derivatives_reach_the_jacobian(tmp_path, with_c_code):
    """
    The derivative buffer travels the same way as the result buffer and was equally lost.

    A vanishing derivative here is not a slow Newton but an exactly singular Jacobian, since the
    single unknown's only equation would have no dependence on it left.
    """
    class _Problem(Problem):
        def define_problem(self):
            self.add_equations(_SquareRootODE(_WithCCode() if with_c_code else _PythonOnly()) @ "ode")

    with _Problem() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.initialise()
        problem.solve()
        assert abs(float(problem.get_ode("ode").get_value("u")) - 2.0) < 1e-8
