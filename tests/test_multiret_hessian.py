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

# Second derivatives of CustomMultiReturnExpression, i.e. multi-return callbacks inside an analytic
# Hessian.
#
# A multi-return callback is a black box: pyoomph gets its value and, at runtime, its Jacobian
# w.r.t. the arguments, and builds the chain rule around them symbolically. Until this was added the
# chain rule stopped after one differentiation ("Multi-Return Callbacks can only be derived to the
# first order at the moment"), which put every one of them out of reach of an analytic Hessian and
# so out of reach of every eigenvalue, bifurcation, Floquet and orbit workflow.
#
# What each test here really checks is d2R/du_j du_k, through
# Problem.debug_analytic_hessian_by_fd(): it contracts the analytic Hessian with random vectors and
# compares against a central difference of the analytic Jacobian along one of them. Its noise floor
# on these problems is around 1e-9, so a 1e-5 threshold is four orders of magnitude of headroom -
# and dropping the second-derivative term entirely takes the discrepancy to O(1), which is what
# makes the check worth anything.
#
# The residual has to be nonlinear in the callback's RESULT as well as in its argument, otherwise
# d2/du2 of the callback never enters and the tests pass without exercising anything.

from pyoomph import Problem, Equations
from pyoomph.expressions import var_and_test, weak, grad, subexpression
from pyoomph.expressions.cb import CustomMultiReturnExpression
from pyoomph.meshes.simplemeshes import LineMesh


# =============================================================================================
# Callbacks covering every route to a second derivative
# =============================================================================================

class _PythonOnly(CustomMultiReturnExpression):
    """f(a) = (2a, a^3). No C code, so every call goes back into Python.

    The Jacobian is analytic, the second derivatives come from the default finite-difference
    fallback in eval_second_derivatives().
    """

    def get_num_returned_scalars(self, nargs):
        return 2

    def eval(self, flag, arg_list, result_list, derivative_matrix):
        a = arg_list[0]
        result_list[0] = 2 * a
        result_list[1] = a ** 3
        if flag:
            derivative_matrix[0, 0] = 2.0
            derivative_matrix[1, 0] = 3 * a ** 2


class _PythonAnalyticSecond(_PythonOnly):
    """The same function with the second derivatives written out."""

    def eval_second_derivatives(self, arg_list, result_list, derivative_matrix, second_derivative_tensor):
        self.eval(1, arg_list, result_list, derivative_matrix)
        second_derivative_tensor[0, 0, 0] = 0.0
        second_derivative_tensor[1, 0, 0] = 6 * arg_list[0]


class _CCodeFDSecond(_PythonOnly):
    """The same function in C, with no second-derivative C code at all.

    This is the case every callback shipped in pyoomph is in today, so it is the one that decides
    whether they keep working: the code generator has to append FILL_MULTI_RET_HESSIAN_BY_FD by
    itself. The FD-of-a-Jacobian is only about sqrt(eps) accurate, hence the looser threshold where
    this class is used.
    """

    def generate_c_code(self):
        return """
        result_list[0] = 2*arg_list[0];
        result_list[1] = arg_list[0]*arg_list[0]*arg_list[0];
        if (flag) { derivative_matrix[0] = 2.0; derivative_matrix[1] = 3*arg_list[0]*arg_list[0]; }
        """


class _CCodeAnalyticSecond(_CCodeFDSecond):
    """The same again, with the second derivatives written out in C."""

    def generate_c_code_second_derivatives(self):
        return """
        CURRENT_MULTIRET_FUNCTION(PYOOMPH_MULTIRET_FLAG_DERIVATIVES, arg_list, result_list, derivative_matrix, nargs, nret);
        second_derivative_tensor[0] = 0.0;
        second_derivative_tensor[1] = 6*arg_list[0];
        """


class _SymbolicFirstDerivative(_PythonOnly):
    """Supplies result 0's first derivative in closed form, and leaves result 1's numeric.

    A closed-form first derivative is an ordinary expression, so the code generator differentiates
    it again by itself and never builds a second-derivative node for it. The mixture is the point:
    both routes have to coexist in one Hessian.
    """

    def use_symbolic_derivative(self, arg_list, i_res, j_arg):
        if i_res == 0:
            return 2
        return None


class _SymbolicSecondDerivative(_PythonOnly):
    """Supplies the second derivatives symbolically, so no d2 tensor is needed at runtime."""

    def use_symbolic_second_derivative(self, arg_list, i_res, j_arg, k_arg):
        if i_res == 0:
            return 0
        return 6 * arg_list[0]


class _TwoArgument(CustomMultiReturnExpression):
    """g(a, b) = (a*b, a^2 + b^3), to exercise the mixed j != k entries and their symmetry."""

    def get_num_returned_scalars(self, nargs):
        return 2

    def eval(self, flag, arg_list, result_list, derivative_matrix):
        a, b = arg_list[0], arg_list[1]
        result_list[0] = a * b
        result_list[1] = a ** 2 + b ** 3
        if flag:
            derivative_matrix[0, 0] = b
            derivative_matrix[0, 1] = a
            derivative_matrix[1, 0] = 2 * a
            derivative_matrix[1, 1] = 3 * b ** 2

    def eval_second_derivatives(self, arg_list, result_list, derivative_matrix, second_derivative_tensor):
        self.eval(1, arg_list, result_list, derivative_matrix)
        second_derivative_tensor.fill(0.0)
        second_derivative_tensor[0, 0, 1] = 1.0
        second_derivative_tensor[0, 1, 0] = 1.0
        second_derivative_tensor[1, 0, 0] = 2.0
        second_derivative_tensor[1, 1, 1] = 6 * arg_list[1]


# =============================================================================================
# The harness
# =============================================================================================

def _hessian_fd_check(residual_builder, threshold=1e-5, N=3):
    """Build a 1D nonlinear problem around `residual_builder(u)` and check its analytic Hessian.

    Returns the worst relative discrepancy, so a caller can also assert it is not absurdly small
    for the wrong reason.
    """

    class _Eq(Equations):
        def define_fields(self):
            self.define_scalar_field("u", "C2")

        def define_residuals(self):
            u, v = var_and_test("u")
            self.add_residual(weak(grad(u), grad(v)) + weak(residual_builder(u), v))

    class _P(Problem):
        def define_problem(self):
            self.add_mesh(LineMesh(N=N))
            self.add_equations(_Eq() @ "domain")

    with _P() as p:
        p.setup_for_stability_analysis(analytic_hessian=True)
        p.initialise()
        mesh = p.get_mesh("domain")
        # Anything but a uniform state: at u == const many of the second derivatives are equal and
        # an index mix-up between them would go unnoticed.
        for n in mesh.nodes():
            n.set_value(0, 0.35 + 0.4 * n.x(0))
        return p.debug_analytic_hessian_by_fd(epsilon=threshold)


# =============================================================================================
# Tests
# =============================================================================================

def test_python_callback_second_derivatives_by_fd():
    """The default: no analytic second derivatives anywhere, so eval() is finite-differenced."""
    cb = _PythonOnly()
    _hessian_fd_check(lambda u: cb(u)[1] * cb(u)[0], threshold=1e-4)


def test_python_callback_analytic_second_derivatives():
    cb = _PythonAnalyticSecond()
    _hessian_fd_check(lambda u: cb(u)[1] * cb(u)[0])


def test_c_callback_second_derivatives_by_generated_fd():
    """No second-derivative C code: the generator has to append FILL_MULTI_RET_HESSIAN_BY_FD.

    This is what keeps every generate_c_code() implementation already in pyoomph working under an
    analytic Hessian without being touched.
    """
    cb = _CCodeFDSecond()
    _hessian_fd_check(lambda u: cb(u)[1] * cb(u)[0], threshold=1e-4)


def test_c_callback_analytic_second_derivatives():
    cb = _CCodeAnalyticSecond()
    _hessian_fd_check(lambda u: cb(u)[1] * cb(u)[0])


def test_symbolic_first_derivative_needs_no_second_derivative_node():
    """A closed-form first derivative is differentiated again by GiNaC, not by the callback."""
    cb = _SymbolicFirstDerivative()
    _hessian_fd_check(lambda u: cb(u)[1] * cb(u)[0], threshold=1e-4)


def test_symbolic_second_derivative():
    cb = _SymbolicSecondDerivative()
    _hessian_fd_check(lambda u: cb(u)[1] * cb(u)[0])


def test_two_argument_callback_mixed_second_derivatives():
    """d2f/da db, the off-diagonal entries the index canonicalisation folds together."""
    cb = _TwoArgument()
    _hessian_fd_check(lambda u: cb(u, 1 + u * u)[0] * cb(u, 1 + u * u)[1])


def test_nested_callbacks():
    """B(A(u)): the second derivative is B''(A')^2 + B'A'', so BOTH tensors are read.

    If the inner call's tensor were not emitted the generated C would not compile, so the failure
    mode here is loud rather than silent - but only if the nesting is actually tested.
    """
    inner = _PythonAnalyticSecond()
    outer = _CCodeAnalyticSecond()
    _hessian_fd_check(lambda u: outer(inner(u)[1])[1])


def test_callback_inside_a_subexpression():
    """Subexpressions take a different route through the Hessian generator.

    The outer Hessian index wraps d(body)/d(field) in a fresh nested subexpression, whose own
    first-derivative cache is filled by a differentiation that happens AFTER the multi-return call
    is emitted. That is what decides whether the d2 tensor is declared at all.
    """
    cb = _PythonAnalyticSecond()
    _hessian_fd_check(lambda u: subexpression(cb(u)[1]) * subexpression(cb(u)[0]))


def test_residual_and_jacobian_code_is_untouched(tmp_path):
    """The one property the whole design turns on: nothing outside the Hessian may change.

    The second-derivative machinery lives in a separate generated function
    (multi_ret_ccode_d2_*) reached through a separate function-table entry, so the residual and
    Jacobian keep calling exactly what they always called. Checked on the emitted C rather than
    argued about, because a regression here would be invisible until someone benchmarked it.
    """
    cb = _CCodeAnalyticSecond()

    def _generate(hessian, name):
        class _Eq(Equations):
            def define_fields(self):
                self.define_scalar_field("u", "C2")

            def define_residuals(self):
                u, v = var_and_test("u")
                self.add_residual(weak(grad(u), grad(v)) + weak(cb(u)[1] * cb(u)[0], v))

        class _P(Problem):
            def define_problem(self):
                self.add_mesh(LineMesh(N=2))
                self.add_equations(_Eq() @ "domain")

        with _P() as p:
            p.set_output_directory(str(tmp_path / name))
            if hessian:
                p.setup_for_stability_analysis(analytic_hessian=True)
            p.initialise()
        with open(tmp_path / name / "_ccode" / "domain.c") as f:
            return f.read()

    plain = _generate(False, "plain")
    # Without a Hessian nothing second-order may appear at any call site, and the callback is still
    # invoked with the enclosing routine's own flag variable.
    assert "d2multi_ret_" not in plain
    assert "multi_ret_ccode_d2_0(" not in plain
    assert "multi_ret_ccode_0(flag," in plain

    with_hessian = _generate(True, "hessian")
    # The residual/Jacobian call in the very same file is unchanged...
    assert "multi_ret_ccode_0(flag," in with_hessian
    # ...while the Hessian routine uses the second-derivative entry point with a literal flag. That
    # literal is also the fix for a real bug: the Hessian routine's own `flag` is a Hessian MODE
    # (0..5, and 0 is the ordinary path), so passing it through told the callback not to fill the
    # derivative matrix that the Hessian body then read anyway.
    assert "multi_ret_ccode_d2_0(PYOOMPH_MULTIRET_FLAG_DERIVATIVES|PYOOMPH_MULTIRET_FLAG_SECOND_DERIVATIVES," in with_hessian
    assert "d2multi_ret_0" in with_hessian


def test_second_derivative_tensor_layout(tmp_path):
    """d2multi_ret_<i>[(i_res*nargs + j)*nargs + k], extending the derivative matrix one index deep."""
    cb = _TwoArgument()

    class _Eq(Equations):
        def define_fields(self):
            self.define_scalar_field("u", "C2")

        def define_residuals(self):
            u, v = var_and_test("u")
            r = cb(u, 1 + u * u)
            self.add_residual(weak(grad(u), grad(v)) + weak(r[0] * r[1], v))

    class _P(Problem):
        def define_problem(self):
            self.add_mesh(LineMesh(N=2))
            self.add_equations(_Eq() @ "domain")

    with _P() as p:
        p.set_output_directory(str(tmp_path))
        p.setup_for_stability_analysis(analytic_hessian=True)
        p.initialise()
    with open(tmp_path / "_ccode" / "domain.c") as f:
        code = f.read()
    # nargs == 2, so the strides are 4 and 2 and the array is nret*nargs*nargs == 8 long.
    assert "PYOOMPH_AQUIRE_ARRAY(double,d2multi_ret_0,2*2*2)" in code
    assert "d2multi_ret_0[4*" in code
    # And it is declared above the integration-point loop, not inside it: PYOOMPH_AQUIRE_ARRAY is
    # _alloca on Windows, which a loop never releases, so nret*nargs*nargs doubles per integration
    # point would accumulate across the whole quadrature rule.
    hessian_fn = code.index("HessianVectorProduct0(")
    decl = code.index("PYOOMPH_AQUIRE_ARRAY(double,d2multi_ret_0", hessian_fn)
    loop = code.index("for(unsigned ipt=", hessian_fn)
    assert decl < loop
    # The value and derivative arrays stay inside the loop, exactly as they always were.
    assert code.index("PYOOMPH_AQUIRE_ARRAY(double,multi_ret_0", hessian_fn) > loop


# =============================================================================================
# The callbacks pyoomph itself ships
# =============================================================================================
#
# These are checked at the level of the callback rather than through a Hessian assembly: what a
# shipped class has to get right is its own second-derivative tensor, and comparing that against
# central differences of its own analytic Jacobian isolates exactly that. Whether the surrounding
# chain rule then works is what every test above already covers, on callbacks small enough that a
# wrong answer is obvious.

import numpy
import pytest


def _second_derivatives_against_fd(cb, args, threshold=1e-6):
    """Worst relative difference between the callback's second derivatives and central FD."""
    args = numpy.array(args, dtype=float)
    nargs = len(args)
    nres = cb.get_num_returned_scalars(nargs)
    res = numpy.zeros(nres)
    deriv = numpy.zeros((nres, nargs))
    second = numpy.zeros((nres, nargs, nargs))
    cb.eval_second_derivatives(args.copy(), res, deriv, second)

    # The reference differences the class's own Jacobian, so this checks the second derivatives
    # against the first ones and nothing else.
    h = 1e-6
    ref = numpy.zeros((nres, nargs, nargs))
    scratch = numpy.zeros(nres)
    for k in range(nargs):
        up, dn = args.copy(), args.copy()
        up[k] += h
        dn[k] -= h
        Jp = numpy.zeros((nres, nargs))
        Jm = numpy.zeros((nres, nargs))
        cb.eval(1, up, scratch, Jp)
        cb.eval(1, dn, scratch, Jm)
        ref[:, :, k] = (Jp - Jm) / (2 * h)
    scale = numpy.maximum(1.0, numpy.abs(ref))
    worst = float(numpy.max(numpy.abs(second - ref) / scale))
    # The tensor is symmetric by definition, and the generated code relies on it: MultiRetCallback
    # canonicalises the index pair and reads only the j<=k half, so an asymmetric tensor would make
    # the answer depend on which order the indices happened to come out of the chain rule.
    assert numpy.max(numpy.abs(second - numpy.transpose(second, (0, 2, 1)))) < 1e-12
    assert worst < threshold, "worst relative discrepancy " + str(worst)
    return worst


def _tensor_callbacks():
    from pyoomph.expressions.coordsys import CartesianCoordinateSystem
    from pyoomph.expressions.tensor_funcs import (DiagonalizeSymmetricTensor, InvertMatrix,
                                                  LogConfTensorDecompositionAxisymmetric,
                                                  LogConfTensorDecompositionCartesian2d,
                                                  SymmetricMatrixExponential)
    cart = CartesianCoordinateSystem()
    logconf_args = [0.9, 0.1, 0.1, 1.2, 0.3, -0.2, 0.15, 0.25, 1.1, 0.7]
    return [
        ("InvertMatrix-2-general", InvertMatrix(2, "general"), [1.3, 0.2, 0.15, 2.1]),
        ("InvertMatrix-2-symmetric", InvertMatrix(2, "symmetric"), [2.0, 0.3, 1.7]),
        ("InvertMatrix-3-general", InvertMatrix(3, "general"),
         [2.0, 0.3, 0.1, 0.2, 1.7, 0.25, 0.15, 0.2, 2.4]),
        ("InvertMatrix-3-symmetric", InvertMatrix(3, "symmetric"),
         [2.0, 0.3, 0.1, 1.7, 0.2, 2.4]),
        ("SymmetricMatrixExponential", SymmetricMatrixExponential(cart, 2), [0.4, 0.1, 0.3]),
        ("DiagonalizeSymmetricTensor", DiagonalizeSymmetricTensor(cart, 2), [1.4, 0.25, 0.9]),
        ("LogConfCartesian2d", LogConfTensorDecompositionCartesian2d(), logconf_args),
        ("LogConfAxisymmetric", LogConfTensorDecompositionAxisymmetric(), logconf_args + [0.8]),
    ]


@pytest.mark.parametrize("name,cb,args", _tensor_callbacks(), ids=lambda v: v if isinstance(v, str) else "")
def test_shipped_tensor_callback_second_derivatives(name, cb, args):
    """Every tensor callback pyoomph ships, differentiated twice through its own eval().

    These use second_derivative_mode="ad", i.e. their exact derivatives come from running eval()
    on HyperDual numbers rather than from anything written out by hand.
    """
    _second_derivatives_against_fd(cb, args)


def test_multi_safe_divide_second_derivatives():
    """A/B, whose only nonlinearity is the division: written out analytically."""
    from pyoomph.expressions.utils import MultiSafeDivide
    # A0, A1, B, eps, C0, C1
    _second_derivatives_against_fd(MultiSafeDivide(), [0.7, 1.3, 2.1, 1e-10, 0.0, 0.0])


def test_multi_safe_divide_cutoff_branch_is_linear():
    """Below the cutoff the result is C, which enters linearly: every second derivative is zero."""
    from pyoomph.expressions.utils import MultiSafeDivide
    cb = MultiSafeDivide()
    args = numpy.array([0.7, 1.3, 1e-14, 1e-10, 0.4, 0.5])
    res = numpy.zeros(2)
    deriv = numpy.zeros((2, 6))
    second = numpy.zeros((2, 6, 6))
    cb.eval_second_derivatives(args, res, deriv, second)
    assert numpy.max(numpy.abs(second)) == 0.0


@pytest.mark.parametrize("phi", [-1.4, -0.3, 0.6, 1.7])
def test_cahn_hilliard_potentials_second_derivatives(phi):
    """Both piecewise double-well potentials, in and outside the well."""
    from pyoomph.equations.NSCH import PiecewiseNSCHPotential
    from pyoomph.equations.low_order_NSCH import PiecewiseLowOrderNSCHPotential
    _second_derivatives_against_fd(PiecewiseNSCHPotential(), [phi])
    _second_derivatives_against_fd(PiecewiseLowOrderNSCHPotential(), [phi])


def test_spline_interpolator_second_derivatives():
    """scipy differentiates the piecewise polynomial exactly, twice over."""
    from pyoomph.expressions.interpol import CSplineInterpolator
    x = numpy.linspace(0.0, 3.0, 12)
    data = numpy.stack([x, numpy.sin(x) + 0.3 * x ** 2], axis=1)
    # Away from a knot: a cubic spline's second derivative is only piecewise linear, so exactly at
    # a knot the central difference straddles a kink and compares two different one-sided values.
    _second_derivatives_against_fd(CSplineInterpolator(data), [1.37])


def test_activity_model_second_derivatives():
    """UNIFAC and AIOMFAC, whose Jacobians used to be finite differences themselves.

    That is what makes them the important case: a second derivative differenced out of an already
    differenced Jacobian has no correct digits left (measured: relative error 2.2 on AIOMFAC), so
    these two only became usable in a Hessian once their Jacobians became exact.
    """
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_activity_aiomfac import _glycerol_nacl_mixture
    from pyoomph.materials.activity_electrolyte import AIOMFACElectrolyteMultiReturnExpression

    mr = AIOMFACElectrolyteMultiReturnExpression(_glycerol_nacl_mixture())
    _second_derivatives_against_fd(mr, [0.12, 0.88, 0.6, 0.6, 298.15], threshold=1e-5)


# =============================================================================================
# End to end: the workflow this exists for
# =============================================================================================

def test_fold_tracking_through_a_multi_return_callback(tmp_path):
    """Track a fold in a residual that goes through a multi-return callback.

    This is the case the feature exists for, and it is also the one the Hessian-mode bug corrupted:
    a fold handler reaches HessianVectorProduct through
    fill_in_contribution_to_hessian_vector_products, which passes Hessian mode 0 - and mode 0 used
    to be forwarded to the callback as "do not fill the derivative matrix", while the Hessian body
    read that matrix regardless.

    The problem is u^2 = lam with u^2 supplied by the callback. Its fold is at lam = 0, u = 0.
    Both dispatch paths are exercised: an ODE this small converges to the fold exactly, so a
    wrong Hessian shows up as a failure to converge rather than as a slightly wrong number.
    """
    from pyoomph import ODEEquations

    class _Square(_PythonAnalyticSecond):
        def get_num_returned_scalars(self, nargs):
            return 2

        def eval(self, flag, arg_list, result_list, derivative_matrix):
            a = arg_list[0]
            result_list[0] = a
            result_list[1] = a * a
            if flag:
                derivative_matrix[0, 0] = 1.0
                derivative_matrix[1, 0] = 2 * a

        def eval_second_derivatives(self, arg_list, result_list, derivative_matrix, second_derivative_tensor):
            self.eval(1, arg_list, result_list, derivative_matrix)
            second_derivative_tensor[0, 0, 0] = 0.0
            second_derivative_tensor[1, 0, 0] = 2.0

    class _FoldODE(ODEEquations):
        def define_fields(self):
            self.define_ode_variable("u")
            self.set_initial_condition("u", 1.0)

        def define_residuals(self):
            u, ut = var_and_test("u")
            lam = self.get_current_code_generator().get_problem().get_global_parameter("lam")
            self.add_weak(_Square()(u)[1] - lam, ut)

    class _P(Problem):
        def define_problem(self):
            self.lam = self.get_global_parameter("lam")
            self.add_equations(_FoldODE() @ "ode")

    with _P() as p:
        p.set_output_directory(str(tmp_path))
        p.setup_for_stability_analysis(analytic_hessian=True)
        p.get_global_parameter("lam").value = 1.0
        p.initialise()
        p.solve()
        assert abs(float(p.get_ode("ode").get_value("u")) - 1.0) < 1e-8
        p.activate_bifurcation_tracking("lam", "fold")
        p.solve()
        assert abs(float(p.get_global_parameter("lam").value)) < 1e-7
        assert abs(float(p.get_ode("ode").get_value("u"))) < 1e-6
