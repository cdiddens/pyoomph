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

# delayed_lambda_expansion(f) holds f symbolically and calls it at CODE GENERATION time, not when the
# expression is built. That is the whole point of it: a residual can refer to a value that is still
# going to change, and picks up whatever it is once the element code is actually generated.
#
# Two properties are worth pinning, because nothing else in the tree uses this function - it is public
# API with no in-tree caller, so a regression here would otherwise surface only in user code:
#
#   * WHEN f runs. Eagerly evaluating it at construction would silently freeze the wrong value, and
#     every test below that checks a number would still pass if the value happened not to change.
#   * That a call does not leak. The binding used to carry a mutual
#     nb::keep_alive<0,1>()+keep_alive<1,0>(): the returned Expression pinned the callable and the
#     callable pinned the Expression, so neither could ever be collected and every single call leaked
#     both, reported at shutdown as "nanobind: leaked N instances" plus one leaked keep_alive record.
#     The leaf is reference-counted now (DelayedPythonCallbackExpansionWrapper, src/expressions.hpp);
#     test_a_delayed_expansion_is_not_leaked guards that it stays that way.

import re
import subprocess
import sys
import textwrap

import pytest

from pyoomph import InitialCondition, ODEEquations, Problem, _pyoomph
from pyoomph.expressions import delayed_lambda_expansion, partial_t, var, var_and_test, weak


class _RelaxToValue(ODEEquations):
    """du/dt = value - u, so the stationary solution is exactly "value"."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def define_fields(self):
        self.define_ode_variable("u")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(partial_t(u) - (self.value - u), v))


class _RelaxProblem(Problem):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def define_problem(self):
        self += (_RelaxToValue(self.value) + InitialCondition(u=0)) @ "ode"


def test_the_callable_is_not_invoked_while_the_expression_is_built():
    """Building the expression must not call f - only code generation may."""
    calls = []

    expression = delayed_lambda_expansion(lambda: (calls.append(1), _pyoomph.Expression(3))[1])
    assert calls == [], "the callable ran at construction time, so the expansion is not delayed at all"

    # Combining it with other expressions must not trigger it either: it is a symbolic leaf.
    _ = 2 * expression + 1
    assert calls == [], "the callable ran while the expression was being combined"


def test_the_expansion_picks_up_the_value_current_at_code_generation():
    """The documented purpose: a value that changes after the residual is written still lands in the code.

    Guards against an eager evaluation that would freeze the value the expression was built with -
    which is why the value is deliberately changed in between, and why the frozen value (7) and the
    generated one (11) are different numbers.
    """
    box = {"value": 7}
    problem = _RelaxProblem(delayed_lambda_expansion(lambda: _pyoomph.Expression(box["value"])))
    problem.set_output_directory("delayed_expansion_codegen")
    problem.quiet()

    box["value"] = 11  # changed AFTER the residual was written, BEFORE the code is generated
    problem.initialise()
    problem.solve()  # stationary solve -> u == the value the generated code was given

    # float(): get_value() hands back an Expression, which does not compare numerically on its own.
    assert float(problem.get_ode("ode").get_value("u")) == pytest.approx(11.0, rel=1e-9), (
        "the generated code used the value from construction time, not from code generation time")


def test_a_non_callable_is_rejected():
    with pytest.raises(RuntimeError, match="must pass a callable"):
        delayed_lambda_expansion(42)  # type: ignore[arg-type]


def test_a_result_that_is_no_expression_is_rejected_when_it_is_expanded():
    """The ValueError is raised from the expansion, i.e. at code generation, not at construction."""
    problem = _RelaxProblem(delayed_lambda_expansion(lambda: object()))  # type: ignore[arg-type,return-value]
    problem.set_output_directory("delayed_expansion_bad_result")
    problem.quiet()
    with pytest.raises(ValueError, match="cannot be converted into an expression"):
        problem.initialise()


# --------------------------------------------------------------------------------------------------
# The leak guard. Run in a subprocess, because the symptom is a message nanobind prints during
# interpreter finalization, long after any assertion inside the test could observe it.

_SCRIPT = textwrap.dedent("""
    from pyoomph import _pyoomph
    from pyoomph.expressions import delayed_lambda_expansion

    for i in range(%d):
        e = delayed_lambda_expansion(lambda: _pyoomph.Expression(i))
        del e
    print("BUILT", flush=True)
""")


def _leaked_instances(output):
    """The count from 'nanobind: leaked N instances!', or 0 if the line is absent."""
    matches = re.findall(r"nanobind: leaked (\d+) instances", output)
    return max((int(m) for m in matches), default=0)


def _run(tmp_path, calls, tag):
    script = tmp_path / ("delayed_%s.py" % tag)
    script.write_text(_SCRIPT % calls)
    proc = subprocess.run([sys.executable, str(script)], cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (
        "the %s run exited %d\n--- stdout ---\n%s\n--- stderr tail ---\n%s"
        % (tag, proc.returncode, proc.stdout[-2000:], proc.stderr[-3000:]))
    assert "BUILT" in proc.stdout, "the %s run never built the expressions" % tag
    return _leaked_instances(proc.stdout + proc.stderr)


def test_a_delayed_expansion_is_not_leaked(tmp_path):
    """Calling it must not add anything that survives to interpreter shutdown.

    Measured against a run that makes no call at all rather than against zero: "from pyoomph import *"
    binds nanobind singletons that some interpreter builds report as leaked no matter what the script
    did (a constant baseline - see the module comment of test_remeshing_leaks.py). The question here is
    only whether delayed_lambda_expansion() adds to it, and whether what it adds grows with the number
    of calls, which is what an every-call leak looks like.
    """
    baseline = _run(tmp_path, 0, "baseline")
    once = _run(tmp_path, 1, "once")
    many = _run(tmp_path, 25, "many")

    assert once <= baseline, (
        "one delayed_lambda_expansion() call leaked %d nanobind instances against a baseline of %d: the "
        "returned Expression and the callable are pinning each other again (a mutual nb::keep_alive on "
        "GiNaC_delayed_expansion, or a leaf reference that is never released)" % (once, baseline))
    assert many <= baseline, (
        "25 delayed_lambda_expansion() calls leaked %d nanobind instances against a baseline of %d, i.e. "
        "%d per call - every call leaks its Expression and its callable" % (many, baseline, many - baseline))
