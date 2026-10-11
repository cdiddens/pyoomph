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

# The analytic Jacobian that UNIFACMultiReturnExpression emits into its generated C.
#
# A multi-return callback is a black box to GiNaC: pyoomph asks it for its value and, separately,
# for its Jacobian with respect to the arguments, and builds the chain rule around them. Original
# and Dortmund UNIFAC used to answer the second question with FILL_MULTI_RET_JACOBIAN_BY_FD, i.e.
# one extra evaluation of the whole model per argument, accurate to about sqrt(epsilon). They now
# propagate the derivatives alongside the value in a single pass (analytic_c_jacobian), so what is
# emitted has to be checked against the values it is supposed to be the derivative of.
#
# The mole fractions the model is written in are not the arguments: they are clipped away from 0
# and 1 and renormalised when they sum to more than one. That mapping is piecewise, so it is done
# at runtime and its chain rule is a loop at the end of the generated function - which is exactly
# the part a hand-derived Jacobian gets wrong. The tests therefore also visit compositions inside
# the clipped and the renormalised branch.

import numpy
import pytest

from pyoomph import Problem, Equations
from pyoomph.expressions import var, var_and_test, weak, grad
from pyoomph.expressions.units import kelvin
from pyoomph.materials import Mixture, get_pure_liquid
import pyoomph.materials.default_materials  # noqa: F401  (registers water/ethanol)
from pyoomph.materials.activity import UNIFACMultiReturnExpression
from pyoomph.meshes.simplemeshes import LineMesh

MODELS = ("Original", "Dortmund")


def _mixture():
    return Mixture(get_pure_liquid("water") + 0.3 * get_pure_liquid("ethanol"),
                   quantity="mole_fraction")


def _multi_return(model, analytic):
    mr = UNIFACMultiReturnExpression(_mixture(), model, constant_temperature=298.15 * kelvin)
    mr.analytic_c_jacobian = analytic
    return mr


# =============================================================================================
# End to end: the generated C inside an assembled element Jacobian
# =============================================================================================

def _jacobian_fd_check(model, analytic, initial, capfd, epsilon=1e-5):
    """Assemble a residual built on the activity coefficients and cross-check its Jacobian.

    equation_compilation_flags.debug_jacobian_epsilon assembles the analytic elemental Jacobian
    and a finite-difference one side by side and reports the entries that differ. The
    finite-difference one calls the callback with flag=0, i.e. for values only, so this compares
    the emitted Jacobian against the emitted values - through pyoomph's own chain rule.

    The report is written by the C++ side, so it has to be captured at file-descriptor level
    (capfd). contextlib.redirect_stdout only rebinds Python's sys.stdout and would silently see
    nothing at all, which is indistinguishable from a Jacobian that agrees.
    """
    mr = _multi_return(model, analytic)
    gamma_w = mr.get_activity_coefficient("water")
    gamma_e = mr.get_activity_coefficient("ethanol")

    class Eq(Equations):
        def define_fields(self):
            self.define_scalar_field("molefrac_ethanol", "C2")

        def define_residuals(self):
            x, v = var_and_test("molefrac_ethanol")
            # Nonlinear in the callback's results as well as in its argument, so that a wrong
            # derivative cannot cancel out of the Jacobian.
            self.add_residual(weak(grad(x), grad(v)) + weak(gamma_e * gamma_e + x * gamma_w, v))

    class P(Problem):
        def define_problem(self):
            self.add_mesh(LineMesh(N=3))
            self.add_equations(Eq() @ "domain")
            self.equation_compilation_flags.debug_jacobian_epsilon = epsilon

    with P() as p:
        p.initialise()
        mesh = p.get_mesh("domain")
        for n in mesh.nodes():
            n.set_value(0, initial(n.x(0)))
        capfd.readouterr()
        p.assemble_jacobian(with_residual=False)
        return capfd.readouterr().out


@pytest.mark.parametrize("model", MODELS)
def test_the_analytic_c_jacobian_agrees_with_the_values_it_differentiates(model, capfd):
    out = _jacobian_fd_check(model, True, lambda s: 0.15 + 0.5 * s, capfd)
    assert "DIFFERENCES IN JACOBIAN" not in out, out[:4000]


@pytest.mark.parametrize("model", MODELS)
def test_the_clipped_and_renormalised_branches_are_differentiated_too(model, capfd):
    # Below molefraction_limit_epsilon at one end and past 1 at the other, so both runtime branches
    # of the mole-fraction mapping are visited by the same assembly.
    out = _jacobian_fd_check(model, True, lambda s: 1e-12 + 1.4 * s, capfd)
    assert "DIFFERENCES IN JACOBIAN" not in out, out[:4000]


@pytest.mark.parametrize("model", MODELS)
def test_the_finite_difference_emitter_still_works(model, capfd):
    # analytic_c_jacobian=False must keep giving the old, finite-differenced code.
    out = _jacobian_fd_check(model, False, lambda s: 0.15 + 0.5 * s, capfd)
    assert "DIFFERENCES IN JACOBIAN" not in out, out[:4000]


# =============================================================================================
# The emitted derivatives against the exact ones
# =============================================================================================

@pytest.mark.parametrize("model", MODELS)
def test_the_emitted_code_carries_a_derivative_for_every_argument(model):
    """A structural check, so that a silent fallback to the FD macro cannot pass as a success."""
    mr = _multi_return(model, True)
    code = mr.generate_c_code()
    assert "FILL_MULTI_RET_JACOBIAN_BY_FD" not in code
    nargs = len(mr.argument_order)
    nret = len(mr.argument_order_with_passive)
    for i in range(nret * nargs):
        assert "derivative_matrix[" + str(i) + "]" in code
    assert "FILL_MULTI_RET_JACOBIAN_BY_FD" in _multi_return(model, False).generate_c_code()


@pytest.mark.parametrize("model", MODELS)
def test_the_python_reference_is_exact_where_the_mapping_is_the_identity(model):
    """Python eval() has no clipping, so inside (eps, 1-eps) it must reproduce the same numbers.

    This pins the Python path, which is what the generated code was checked against - see
    fill_python_derivatives_by_AD.
    """
    mr = _multi_return(model, True)
    for x in (0.05, 0.3, 0.8):
        args = numpy.array([x])
        res = numpy.zeros(2)
        deriv = numpy.zeros((2, 1))
        mr.fill_python_derivatives_by_AD(args, res, deriv)
        h = 1e-6
        plus = numpy.zeros(2)
        minus = numpy.zeros(2)
        mr.eval(0, numpy.array([x + h]), plus, numpy.zeros((2, 1)))
        mr.eval(0, numpy.array([x - h]), minus, numpy.zeros((2, 1)))
        assert deriv[:, 0] == pytest.approx((plus - minus) / (2 * h), rel=1e-6)
