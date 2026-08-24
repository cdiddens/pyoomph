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

# Deflation, serial. Until this file there was no pytest cover for it at all -- only the two
# tutorial scripts, which the tutorial harness skips under --mpirun. dev_docs/deflation.md has the
# derivations these assertions pin down; the MPI side is tests/test_mpi_deflation.py.

import numpy
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.generic.bifurcation_tools import DeflationOperator


class PitchForkNormalForm(ODEEquations):
    """xdot = r x - x^3: exactly three steady states at r>0, namely 0 and +-sqrt(r)."""

    def __init__(self, r):
        super().__init__()
        self.r = r

    def define_fields(self):
        self.define_ode_variable("x")

    def define_residuals(self):
        x, xt = var_and_test("x")
        self.add_residual((partial_t(x) - (self.r * x - x ** 3)) * xt)


class PitchForkProblem(Problem):
    def __init__(self, r=1.0):
        super().__init__()
        self.r_value = r

    def define_problem(self):
        self.r = self.define_global_parameter(r=self.r_value)
        self += PitchForkNormalForm(r=self.r) @ "pitchfork"


def _operator(alpha=0.1, p=2, shift_mode="each", knowns=()):
    op = DeflationOperator(alpha=alpha, p=p, shift_mode=shift_mode)
    for W in knowns:
        op.add_known_solution(numpy.asarray(W, dtype=float))
    return op


# ---------------------------------------------------------------- the algebra


def test_one_solve_equals_the_three_solve_sherman_morrison():
    """The deflated step used to cost three linear solves; it costs one.

    The deflated Jacobian is J/eta + f d^T with f = -b/eta -- a rank-one update whose update vector is
    a MULTIPLE OF THE RIGHT-HAND SIDE. Sherman-Morrison then collapses: solve(f) is -solve(b)/eta and
    the third solve is solve(b) times a scalar. This reproduces the routine as it stood before, so the
    algebra cannot silently rot; a failure here means the closed form and the literal update have
    parted company, not that either is slow.
    """
    rng = numpy.random.default_rng(0)
    n = 12
    J = rng.normal(size=(n, n))
    Jinv = numpy.linalg.inv(J)
    solve = lambda v: Jinv @ v
    b, d = rng.normal(size=n), rng.normal(size=n)
    eta = 0.37

    f = -b / eta
    fsol, bsol = solve(f), solve(b)
    numer = solve(f * numpy.dot(d, bsol))
    denom = 1 + eta * numpy.dot(d, fsol)
    three_solve = eta * bsol - eta ** 2 * numer / denom

    one_solve = eta * solve(b) / (1 - numpy.dot(d, solve(b)))

    assert numpy.allclose(three_solve, one_solve, rtol=0, atol=1e-12)
    # ... and both really do invert the deflated Jacobian.
    A = J / eta + numpy.outer(f, d)
    assert numpy.amax(numpy.absolute(A @ one_solve - b)) < 1e-10


@pytest.mark.parametrize("shift_mode", ["each", "single", "scaled"])
def test_gradient_of_log_M_matches_finite_differences(shift_mode):
    """grad log M is the whole of the Newton-step rescale, and nothing else checks it.

    "single" is here because it used to raise RuntimeError from _get_eta_prime_single (with dead code
    after the raise) while its _get_eta_prime branch summed where it had to multiply -- i.e. it was
    both unreachable and wrong.
    """
    op = _operator(shift_mode=shift_mode, knowns=([0.0, 0.0], [1.0, 0.0], [0.0, 1.5]))
    U = numpy.array([0.3, 0.7])
    _, G = op.evaluate(need_gradient=True, U=U)
    eps = 1e-6
    fd = numpy.array([(op.evaluate(U=U + eps * e)[0] - op.evaluate(U=U - eps * e)[0]) / (2 * eps)
                      for e in numpy.eye(len(U))])
    assert numpy.allclose(G, fd, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("shift_mode", ["each", "single", "scaled"])
def test_factor_tends_to_one_far_from_every_known_solution(shift_mode):
    """The regression for the defect that made deflation report non-solutions.

    Newton's test is max|M R| < newton_solver_tolerance. The unnormalised factor tends to alpha^k for
    k known solutions, so with the default alpha=0.1 the tested residual shrank by TEN PER KNOWN
    SOLUTION: after eight branches any starting guess passed the 1e-8 test at once. Normalised, M >= 1
    always, so deflation can only ever tighten the test. The Newton iterates are unaffected -- the
    step depends on M only through grad log M, which is invariant under M -> cM.
    """
    knowns = [[float(i), 0.0] for i in range(10)]
    op = _operator(shift_mode=shift_mode, knowns=knowns)
    M_far = numpy.exp(op.evaluate(U=numpy.array([0.0, 1e4]))[0])
    # 1e-7 per known solution at this distance, so 1+1e-6 with ten of them -- not exactly 1, but
    # nowhere near the alpha^10 = 1e-10 the unnormalised formula gives.
    assert M_far == pytest.approx(1.0, abs=1e-4)
    # The unnormalised formula would give 1e-10 here, i.e. it would move the effective Newton
    # tolerance from 1e-8 to 1e2.
    assert M_far > 0.999


@pytest.mark.parametrize("shift_mode", ["each", "single", "scaled"])
def test_factor_blows_up_on_a_known_solution(shift_mode):
    """...and it still blows up where it has to, which is what makes deflation work at all.

    ONE known solution, deliberately: "single" puts a single shift outside the whole product, so the
    distant known solutions damp the near one rather than each contributing a pole of its own. That
    is the defining difference between the modes, not a defect -- but it does mean a "near a known
    solution" assertion has to be made with nothing else in the set.
    """
    op = _operator(shift_mode=shift_mode, knowns=([0.0, 0.0],))
    assert numpy.exp(op.evaluate(U=numpy.array([1e-3, 0.0]))[0]) > 1e4


def test_landing_exactly_on_a_known_solution_stays_finite():
    """An iterate that lands ON a known solution must give a huge factor, not a nan.

    inf*0 is nan for every dof whose residual happens to vanish, and a nan residual is not a failed
    Newton step to oomph-lib -- it is a comparison that is false whichever way it is asked, so the
    solve neither converges nor gives up.
    """
    op = _operator(knowns=([1.0, 2.0],))
    logM, G = op.evaluate(need_gradient=True, U=numpy.array([1.0, 2.0]))
    assert numpy.isfinite(logM) and logM > 100.0
    assert numpy.all(numpy.isfinite(G))


def test_rejects_nonsensical_parameters():
    with pytest.raises(ValueError):
        DeflationOperator(p=0)
    with pytest.raises(ValueError):
        DeflationOperator(p=2.5)  # type:ignore
    with pytest.raises(ValueError):
        DeflationOperator(alpha=0.0)
    with pytest.raises(ValueError):
        DeflationOperator(shift_mode="nope")  # type:ignore


# ---------------------------------------------------------------- end to end


def test_pitchfork_ode_finds_all_three_solutions(tmp_path):
    """The tutorial's own claim: x = 0, +1, -1 at r = 1.

    With the seeded generator this is deterministic, which it was not before -- the search drew from
    the global numpy state, so the tutorial found two or three solutions depending on the run.
    """
    with PitchForkProblem(r=1.0) as p:
        p.set_output_directory(str(tmp_path / "solve"))
        p.quiet()
        sols = [float(s[0]) for s in p.iterate_over_multiple_solutions_by_deflation(
            deflation_alpha=0.1, deflation_p=2, perturbation_amplitude=0.1,
            num_random_tries=2, random_seed=0)]
    # Newton's own accuracy, not the eroded 8 digits the unnormalised factor used to leave behind
    # (it returned -0.99999999 here). The bound is the residual tolerance divided by dR/dx = 2 with a
    # margin, not machine epsilon: the solve stops when the residual does, so the position of a root
    # is only good to about 5e-9, and which backend factorised it - and how strongly the deflation
    # pushed on the way in - moves the last couple of digits.
    assert sorted(sols) == pytest.approx([-1.0, 0.0, 1.0], abs=1e-7)
    for s in sols:
        assert abs(abs(s) - round(abs(s))) < 1e-7


def test_deflation_leaves_no_trace_on_the_problem(tmp_path):
    """The operator must come off cleanly: no residual scaling, no augmented dofs, ever.

    Deflation deliberately does NOT go through set_custom_assembler, so it never appends to Dof_pt
    and never sets use_custom_residual_jacobian -- which is what keeps the frozen sparsity pattern,
    the proven-symmetry fast path and MPI available while it is installed.
    """
    with PitchForkProblem(r=1.0) as p:
        p.set_output_directory(str(tmp_path / "trace"))
        p.quiet()
        p.initialise()
        p.solve()
        ndof_before = p.ndof()
        op = DeflationOperator(alpha=0.1, p=2)
        p.set_deflation_operator(op)
        op.add_known_solution(p.get_current_dofs()[0])
        assert p.residual_scale_hook_active is True
        assert p.use_custom_residual_jacobian is False
        assert p.ndof() == ndof_before
        assert p.get_residual_scale_factor() > 1.0
        p.set_deflation_operator(None)
        assert p.residual_scale_hook_active is False
        assert p.get_residual_scale_factor() == 1.0
        assert p.ndof() == ndof_before


def test_deflated_residual_never_shrinks(tmp_path):
    """The assembled residual with deflation on is at least the plain one, entry by entry.

    This is the same defect as test_factor_tends_to_one..., taken through the real assembly instead
    of the operator alone: it is the number oomph-lib compares against newton_solver_tolerance.
    """
    with PitchForkProblem(r=1.0) as p:
        p.set_output_directory(str(tmp_path / "resid"))
        p.quiet()
        p.initialise()
        p.set_current_dofs(numpy.array([0.3]))
        plain = numpy.amax(numpy.absolute(p.get_residuals()))
        op = DeflationOperator(alpha=0.1, p=2)
        p.set_deflation_operator(op)
        for w in numpy.linspace(10.0, 30.0, 10):   # ten known solutions, all far away
            op.add_known_solution(numpy.array([w]))
        deflated = numpy.amax(numpy.absolute(p.get_residuals()))
    assert plain > 1e-3, "the test state must have a residual worth measuring"
    assert deflated >= plain * (1.0 - 1e-12)


def test_deflated_continuation_traces_the_pitchfork(tmp_path):
    """Deflated continuation over r, straight through the bifurcation at r=0.

    Below it there is one solution and above it three, so the branch bookkeeping -- which branches
    survive a parameter step, which are new, which two got swapped -- is exercised rather than just
    the deflated solve.
    """
    branches = {}
    with PitchForkProblem(r=-1.0) as p:
        p.set_output_directory(str(tmp_path / "conti"))
        p.quiet()
        for branch_index, rvalue, sol in p.deflated_continuation(
                r=numpy.linspace(-1.0, 1.0, 21), perturbation_amplitude=0.5,
                num_random_tries=2, random_seed=0):
            branches.setdefault(branch_index, []).append((float(rvalue), float(sol[0])))
    assert len(branches) == 3, "expected the trivial branch and the symmetric pair, got %r" % (
        {k: len(v) for k, v in branches.items()},)
    finals = sorted(round(v[-1][1], 6) for v in branches.values())
    assert finals == pytest.approx([-1.0, 0.0, 1.0], abs=1e-6)
    # The trivial branch exists for every r; the other two only above the bifurcation.
    assert max(len(v) for v in branches.values()) == 21


def test_assembly_handler_shell_still_works(tmp_path):
    """DeflationAssemblyHandler is kept for scripts that construct it directly.

    It is now a shell around the same DeflationOperator, installed through set_custom_assembler
    rather than set_deflation_operator -- so it goes through the custom-assembler pipeline and is
    serial-only, but it must still find what it always found.
    """
    from pyoomph.generic.bifurcation_tools import DeflationAssemblyHandler
    with PitchForkProblem(r=1.0) as p:
        p.set_output_directory(str(tmp_path / "shell"))
        p.quiet()
        p.initialise()
        p.solve()
        h = DeflationAssemblyHandler(alpha=0.1, p=2)
        p.set_custom_assembler(h)
        h.add_known_solution(p.get_current_dofs()[0])
        assert (h.alpha, h.p, h.shift_mode, len(h.Ws)) == (0.1, 2, "each", 1)
        p.perturb_dofs(numpy.array([0.3]))
        p.solve()
        found = float(p.get_current_dofs()[0][0])
        p.set_custom_assembler(None)
    assert abs(abs(found) - 1.0) < 1e-7, "deflating x=0 must lead to x=+-1, got %r" % found


def test_refuses_adaptation_while_solutions_are_known(tmp_path):
    """Known solutions are dof vectors of the current numbering, which adaptation invalidates."""
    from pyoomph.generic.bifurcation_tools import DeflationOperator
    with PitchForkProblem(r=1.0) as p:
        p.set_output_directory(str(tmp_path / "adapt"))
        p.quiet()
        p.initialise()
        p.solve()
        op = DeflationOperator()
        p.set_deflation_operator(op)
        # An empty operator is harmless, so the refusal must only fire once something is stored.
        p._adapt()
        op.add_known_solution(p.get_current_dofs()[0])
        with pytest.raises(RuntimeError, match="deflation operator holds known solutions"):
            p._adapt()
        op.clear_known_solutions()
        p._adapt()
        p.set_deflation_operator(None)


def test_refuses_static_condensation(tmp_path):
    """Deflation rescales the Newton increment AFTER the solve, and condensation reconstructs the
    dofs it eliminated FROM that increment, using operators built from the unrescaled residual. The
    two cannot both be right, so the combination is refused where the user switches it on."""
    with PitchForkProblem(r=1.0) as p:
        p.set_output_directory(str(tmp_path / "cond"))
        p.quiet()
        p.initialise()
        p.use_static_condensation = True
        with pytest.raises(RuntimeError, match="static condensation"):
            p.set_deflation_operator(DeflationOperator())
