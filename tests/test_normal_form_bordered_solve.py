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

# NormalFormCalculator.bordered_la_solve, with a NONZERO right-hand side.
#
# This path had no coverage at all. Every existing normal-form and branch-switch case reaches it on
# the trivial branch, where dR/dparameter is identically zero -- so the `nrhs == 0.0` early return
# fires and the bordered system is never built, let alone factorised. Instrumented and counted: zero
# hits across test_normal_form_units, test_normal_form_degenerate_a, test_mpi_branch_switch and
# test_normal_mode_branch_switch.
#
# That matters because L is EXACTLY singular here -- that is what makes a branch point a branch
# point -- and the comment on bordered_la_solve records what the old plain spsolve did with it:
# returned a finite vector with no warning, so the NaN test never tripped and the normal-form
# coefficient was whatever that vector projected to. So the assertions below are on the
# mathematical contract, not on a reference number:
#
#   - the border multiplier s comes out zero, which it must by construction when rhs is
#     E()-projected (the routine raises if it does not, and that raise is what is checked here);
#   - L @ psi reproduces rhs, which is the actual solve;
#   - psi is zeta_star-orthogonal, which is the component the routine promises to return.
#
# L is fabricated rather than assembled: a singular matrix with a known one-dimensional kernel is
# exactly what is needed and an assembled one would have to be coaxed to a branch point first,
# which is what makes this testable without MPI at all.

import os
import sys

import numpy
import pytest
import scipy.sparse

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.generic.bifurcation_tools import NormalFormCalculator


class _TinyProblem(Problem):
    """Somewhere for the calculator to find a linear solver. Its equations are irrelevant here."""

    def define_problem(self):
        self += ODEFile() @ "dummy"


class ODEFile(ODEEquations):
    def define_fields(self):
        self.define_ode_variable("y")

    def define_residuals(self):
        y, yt = var_and_test("y")
        self.add_residual(weak(partial_t(y) + y, yt))


def _singular_system(n=12, seed=3):
    """A symmetric L with a one-dimensional kernel, plus a right-hand side inside its range.

    Built from an eigendecomposition so the kernel is EXACT rather than merely small: a nearly
    singular L would let a plain solve succeed and the test would not distinguish anything.
    """
    rng = numpy.random.default_rng(seed)
    Q, _ = numpy.linalg.qr(rng.standard_normal((n, n)))
    eigs = numpy.concatenate([[0.0], rng.uniform(0.5, 2.0, n - 1)])
    L = Q @ numpy.diag(eigs) @ Q.T
    zeta = Q[:, 0].copy()                  # ker(L)
    zeta_star = Q[:, 0].copy()             # ker(L^T); L is symmetric here
    # A right-hand side in range(L) = ker(L^T)^perp, which is what E() produces at the call sites.
    rhs = L @ rng.standard_normal(n)
    return scipy.sparse.csr_matrix(L), rhs, zeta, zeta_star


@pytest.fixture(scope="module")
def calculator(tmp_path_factory):
    with _TinyProblem() as p:
        p.set_output_directory(str(tmp_path_factory.mktemp("nf_bordered")))
        p.quiet()
        p.initialise()
        yield NormalFormCalculator(p)


def test_the_bordered_solve_inverts_a_singular_L(calculator):
    L, rhs, zeta, zeta_star = _singular_system()
    psi = calculator.bordered_la_solve(L, rhs, zeta, zeta_star)
    assert numpy.all(numpy.isfinite(psi)), psi
    # The solve itself. L is singular, so this is only satisfiable because rhs is in its range.
    residual = numpy.linalg.norm(L @ psi - rhs) / numpy.linalg.norm(rhs)
    assert residual < 1e-9, "L @ psi does not reproduce rhs (relative residual %.3e)" % residual
    # And the component the routine promises: the zeta_star-orthogonal one.
    overlap = abs(float(numpy.dot(zeta_star, psi))) / numpy.linalg.norm(psi)
    assert overlap < 1e-9, "psi has a kernel component of %.3e" % overlap


def test_a_right_hand_side_outside_the_range_is_refused(calculator):
    """The border residual check, which is the routine's own guard against a wrong normal form.

    Adding a kernel-direction component to rhs makes the system inconsistent in exactly the way an
    inconsistent (zeta, zeta_star) pair would. The multiplier s then comes out nonzero and must be
    reported -- silently returning psi there is what would give a plausible wrong coefficient.
    """
    L, rhs, zeta, zeta_star = _singular_system()
    bad = rhs + 0.5 * numpy.linalg.norm(rhs) * zeta_star
    with pytest.raises(RuntimeError, match="border residual"):
        calculator.bordered_la_solve(L, bad, zeta, zeta_star)


def test_a_zero_right_hand_side_is_exactly_zero(calculator):
    """The early return, which is the case every existing test actually takes."""
    L, _, zeta, zeta_star = _singular_system()
    psi = calculator.bordered_la_solve(L, numpy.zeros(L.shape[0]), zeta, zeta_star)
    assert numpy.array_equal(psi, numpy.zeros(L.shape[0]))


def test_a_complex_system_is_refused_by_name(calculator):
    """The real-only contract of the backend route, stated rather than discovered as a cast."""
    L, rhs, zeta, zeta_star = _singular_system()
    with pytest.raises(RuntimeError, match="complex"):
        calculator._solve_bordered_real(L.astype(numpy.complex128), rhs.astype(numpy.complex128))
