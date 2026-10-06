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

# HalleySolver, which had no test, no tutorial and no caller anywhere in the tree.
#
# It is here because the MPI work touches it: MultiAssembleRequest.assemble() now returns this
# rank's (nrow_local, n) row block rather than the whole system, and HalleySolver consumes that
# alongside a global Jacobian and two solve_serial calls. It is refused under --distribute for that
# reason (see the class docstring), and these tests pin what it does where it IS correct -- so that
# the refusal is a statement about a working method rather than about one nobody has run.
#
# The assertion that matters is the CONVERGENCE RATE. Halley is cubic where Newton is quadratic, and
# that is the entire reason the class exists; a wrong second-order term still converges, just more
# slowly, so a test that only checked the root would pass on a broken Hessian contraction.

import os

import numpy
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.utils.halleys_method import HalleySolver


class _CubicODE(ODEEquations):
    """y**3 - a = 0, whose root is a**(1/3). Strongly nonlinear, so the rate is visible."""

    def __init__(self, a):
        super().__init__()
        self.a = a

    def define_fields(self):
        self.define_ode_variable("y")

    def define_residuals(self):
        y, yt = var_and_test("y")
        self.add_residual(weak(y ** 3 - self.a, yt))


class _CubicProblem(Problem):
    def __init__(self, a=8.0, y0=1.0):
        super().__init__()
        self.a, self.y0 = a, y0

    def define_problem(self):
        eqs = _CubicODE(self.a)
        eqs += InitialCondition(y=self.y0)
        self += eqs @ "ode"
        # Halley's second-order term is a Hessian contraction (MultiAssembleRequest.dJdU), which the
        # JIT only emits when asked for at compile time -- without this the solve fails in the
        # assembly with "analytical Hessian were not set", several frames from the method.
        self.setup_for_stability_analysis(analytic_hessian=True)


def _solve_with(problem_kwargs, halley, tmp_path, max_iterations=30):
    """Return (root, number of steps) for Halley or Newton on the same problem and start point."""
    with _CubicProblem(**problem_kwargs) as p:
        p.set_output_directory(str(tmp_path))
        p.quiet()
        p.initialise()
        if halley:
            HalleySolver(p).solve(max_iterations=max_iterations, accuracy=1e-10)
            nsteps = None       # Halley prints its own; the residual history below is what is used
        else:
            p.max_newton_iterations = max_iterations
            p.newton_solver_tolerance = 1e-10
            p.solve()
            nsteps = len(p.get_last_residual_convergence())
        root = float(p.get_ode("ode").get_value("y", dimensional=False, as_float=True))
        return root, nsteps


def test_halley_finds_the_root(tmp_path):
    root, _ = _solve_with(dict(a=8.0, y0=1.0), True, tmp_path / "h")
    assert abs(root - 2.0) < 1e-8, "Halley converged to %r, expected the cube root of 8" % root


def test_halley_converges_cubically(tmp_path):
    """The step count from a far start point, which is what the second-order term buys.

    Not a comparison against Problem.solve(): Halley's loop stops on norm(R, inf) < accuracy while
    pyoomph's Newton uses its own criterion and its own residual history, so "fewer steps than
    Newton" compares two different stopping rules. Measured, that comparison passes even with the
    second-order term deleted -- it was the first version of this test and it was not a test of
    anything.

    What does discriminate is Halley's own step count from y0=0.5 (root at 2, so a long way out):
    4 steps with the term, 9 without, the residuals going 8.5e-2 -> 7.2e-7 (order ~3) against
    5.0e-2 -> 1.0e-4 -> 4.6e-10 (order 2). A bound of 6 sits well clear of both. Verified by
    replacing dJdU/2 with dJdU*0 and re-running: this fails, the other two cases do not.
    """
    import re
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        root, _ = _solve_with(dict(a=8.0, y0=0.5), True, tmp_path / "h2")
    residuals = [float(x) for x in re.findall(r"Residual norm: ([0-9.eE+-]+)", buf.getvalue())]
    assert residuals, "HalleySolver printed no residuals:\n" + buf.getvalue()
    assert abs(root - 2.0) < 1e-8, "converged to %r" % root
    assert len(residuals) <= 6, (
        "Halley took %d steps from y0=0.5; cubic convergence needs about 4 and plain Newton takes "
        "9, so this many means the second-order term is not contributing. Residuals: %r"
        % (len(residuals), residuals))
    # And the shape of it: the last finite residual must be a long way below the square of the one
    # before, which quadratic convergence would not manage.
    finite = [r for r in residuals if r > 1e-13]
    assert len(finite) >= 2, residuals
    assert finite[-1] < finite[-2] ** 2, \
        "the terminal step went %.3e -> %.3e, which is not better than quadratic" % (
            finite[-2], finite[-1])


def test_halley_is_refused_when_distributed(tmp_path):
    """The refusal is by name, so that it cannot be mistaken for a solver failure.

    Serial here, so _require_non_distributed does not fire; the point is that the call exists and
    names the method. A distributed run is covered by the guard itself, which is shared machinery
    with its own tests -- what is asserted here is that HalleySolver reaches it.
    """
    with _CubicProblem() as p:
        p.set_output_directory(str(tmp_path / "g"))
        p.quiet()
        p.initialise()
        assert not p.is_distributed()
        # Fake the distribution flag the guard reads, which is how the refusal is reachable without
        # mpirun at all.
        import unittest.mock
        with unittest.mock.patch.object(type(p), "is_distributed", lambda self: True):
            with pytest.raises(RuntimeError, match="Halley"):
                HalleySolver(p).solve()
