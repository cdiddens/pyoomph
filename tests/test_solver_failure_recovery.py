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

# A linear solver that reports its own failure must not end the run.
#
# The callers that can recover from a failed solve -- the adaptive time stepper, which halves dt, and
# the arclength continuation, which scales Ds by 2/3 -- recognise oomph::NewtonSolverError and nothing
# else. A Python exception raised by a solver backend used to unwind straight past them and out of
# run()/arclength_continuation().
#
# The retry is opt-in, via SolverError: a backend raises it to say "this system could not be solved",
# and only those are reported onwards as a failed Newton solve. Everything else -- a missing file, a
# misconfigured solver, a plain bug -- would fail identically on every retry, so it propagates
# untouched rather than being buried under fifty shrinking steps.
#
# That was harmless only for as long as the backends never raised. MKL Pardiso then learned to report
# its errors, and a singular Jacobian -- the very state adaptivity exists to back away from -- became
# fatal: it used to come back as a huge solution, which tripped max_residuals and was rejected as an
# ordinary divergence. The solver shim (src/nanobind/solver.cpp) now translates such an exception into
# the failure oomph-lib recovers from, and these tests pin down that it does.
#
# The failure is injected rather than provoked with a genuinely singular problem, because what is
# under test is the plumbing between "the backend raised" and "the step was retried", and an injected
# raise exercises exactly that for every backend at once. test_a_singular_matrix_really_does_raise
# below covers the other half -- that Pardiso does raise on a matrix a simulation can walk into.

import numpy
import pytest

from pyoomph import Problem, Equations, DirichletBC
from pyoomph.expressions import var, grad, weak, partial_t, testfunction
from pyoomph.meshes.simplemeshes import LineMesh
from pyoomph.solvers.generic import SolverError


class _NonlinearDiffusion(Equations):
    """u_t = u'' + p*(1+u^2), i.e. nonlinear enough to need several Newton steps per solve."""

    def __init__(self, source):
        super().__init__()
        self.source = source

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var("u"), testfunction("u")
        self.add_residual(weak(partial_t(u), v) + weak(grad(u), grad(v))
                          - weak(self.source * (1 + u * u), v))


class _Diffusion(Problem):
    def __init__(self, with_parameter=False):
        super().__init__()
        self.with_parameter = with_parameter

    def define_problem(self):
        self.add_mesh(LineMesh(N=20))
        source = self.get_global_parameter("p").get_symbol() if self.with_parameter else 1
        eqs = _NonlinearDiffusion(source)
        eqs += DirichletBC(u=0) @ "left"
        eqs += DirichletBC(u=0) @ "right"
        self.add_equations(eqs @ "domain")


class _PretendPardisoError(SolverError):
    """Stands in for pardiso.PardisoError, so these tests do not need an MKL runtime."""


class _FailAt:
    """Makes the linear solver raise on the n-th factorisation, the way Pardiso reports a zero pivot."""

    MESSAGE = "MKL Pardiso failed during the numerical factorisation: error -4 (zero pivot ...)"

    def __init__(self, problem, nth, error=None):
        self.nth = nth
        self.count = 0
        self.error = error if error is not None else _PretendPardisoError(self.MESSAGE)
        solver = problem.get_la_solver()
        original = solver.solve_serial

        def solve_serial(op_flag, *args, **kwargs):
            if op_flag == 1:
                self.count += 1
                if self.count == self.nth:
                    raise self.error
            return original(op_flag, *args, **kwargs)

        solver.solve_serial = solve_serial

    @property
    def fired(self):
        return self.count >= self.nth


def _setup(tmp_path, with_parameter=False):
    problem = _Diffusion(with_parameter)
    problem.set_output_directory(str(tmp_path))
    problem.initialise()
    return problem


# The rejection is announced by oomph-lib on C-level stdout, but asserting on that text is not
# reliable: pytest's capfd reads its temp file while oomph's output is still sitting in the C stdio
# buffer, so the line only appears once the process exits, long after readouterr(). These tests
# therefore check what the recovery is FOR -- the run finishes and the answer is right -- rather than
# what it prints.

def _run_transient(problem, fail_at=None):
    injector = _FailAt(problem, nth=fail_at) if fail_at else None
    with problem:
        problem.run(1.0, startstep=0.01, maxstep=0.1, temporal_error=1e-3, outstep=False)
        return (numpy.array(problem.get_current_dofs()[0]),
                problem.get_current_time(as_float=True), injector)


def test_adaptive_timestep_rejects_a_solver_failure(tmp_path):
    reference, _, _ = _run_transient(_setup(tmp_path / "reference"))
    dofs, time, injector = _run_transient(_setup(tmp_path / "injected"), fail_at=4)

    assert injector.fired, "the injected failure never happened, so nothing was tested"
    assert time == pytest.approx(1.0), "the run did not reach the end time"
    assert numpy.all(numpy.isfinite(dofs)), "the recovered run left non-finite dofs"
    # The rejected step was retried, not skipped or half-applied: reaching t=1 through a different
    # sequence of (smaller) steps must land on the same solution to within the temporal tolerance.
    assert numpy.amax(numpy.abs(dofs - reference)) < 1e-4 * numpy.amax(numpy.abs(reference)), \
        "the recovered run drifted from the undisturbed one by %.3e" \
        % numpy.amax(numpy.abs(dofs - reference))


def test_arclength_continuation_rejects_a_solver_failure(tmp_path):
    problem = _setup(tmp_path, with_parameter=True)
    with problem:
        p = problem.get_global_parameter("p")
        p.value = 0.1
        problem.solve()
        injector = _FailAt(problem, nth=6)
        ds = 0.05
        for _ in range(10):
            ds = problem.arclength_continuation(p, ds, max_ds=0.2)
        reached = p.value
        on_branch = numpy.array(problem.get_current_dofs()[0])
        # A stationary solve at the parameter the continuation stopped at: if the recovered step left
        # a genuine point of the solution branch, this has nothing left to do and the dofs do not move.
        problem.solve()
        settled = numpy.array(problem.get_current_dofs()[0])

    assert injector.fired, "the injected failure never happened, so nothing was tested"
    # Continuation carried on past the failure rather than stopping at it.
    assert reached > 0.5, "the continuation did not get anywhere (p = %.3g)" % reached
    assert numpy.amax(numpy.abs(on_branch - settled)) < 1e-6 * numpy.amax(numpy.abs(on_branch)), \
        "the continuation ended off the solution branch (moved %.3e when re-solved)" \
        % numpy.amax(numpy.abs(on_branch - settled))


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError("no such file: mumps.cnf"),
     RuntimeError("your PETSc installation was not compiled with MUMPS support"),
     TypeError("unsupported operand type(s)"),
     KeyboardInterrupt()],
    ids=["missing-file", "misconfigured-solver", "a-bug", "interrupt"])
def test_only_a_declared_solver_failure_is_retried(tmp_path, error):
    """Anything that is not a SolverError must come out of run() as itself.

    None of these gets better on a smaller step. Retrying them would shrink dt until it fell below
    Minimum_dt and then blame the time step, hiding the message that says what is actually wrong --
    and, for KeyboardInterrupt, would make a long run uninterruptible.
    """
    problem = _setup(tmp_path)
    with problem:
        _FailAt(problem, nth=4, error=error)
        with pytest.raises(type(error)) as caught:
            problem.run(1.0, startstep=0.01, maxstep=0.1, temporal_error=1e-3, outstep=False)
    assert caught.value is error, "the exception was replaced on the way out"


def test_a_rejected_step_does_not_strand_the_run_short_of_endtime(tmp_path):
    """A rejection leaves the accumulated time an ulp short of endtime; run() must not step into that.

    The rejection here comes from before_newton_convergence_check rather than the solver, because the
    defect belongs to run() and every rejection mechanism reaches it. Once the accepted dt differs
    from the requested one, the time grid no longer lands exactly on endtime, and run() used to clamp
    the final step to the ~1e-16 that was left; the Newton solver cannot converge that, oomph-lib
    halves it until it drops below Minimum_dt, and an otherwise finished run died with "Tried to
    reduce dt to 5.55e-17 which is less than the minimum dt".
    """

    class _RejectOnce(Equations):
        def __init__(self, at):
            super().__init__()
            self.at = at
            self.n = 0

        def before_newton_convergence_check(self, eqtree):
            self.n += 1
            return self.n != self.at

    class _Rejecting(_Diffusion):
        def define_problem(self):
            super().define_problem()
            self.add_equations(_RejectOnce(12) @ "domain")

    problem = _Rejecting()
    problem.set_output_directory(str(tmp_path))
    problem.initialise()
    with problem:
        problem.run(1.0, startstep=0.01, maxstep=0.1, temporal_error=1e-3, outstep=False)
        time = problem.get_current_time(as_float=True)
        dofs = numpy.array(problem.get_current_dofs()[0])

    assert time == pytest.approx(1.0), "the run stopped at t = %.17g" % time
    assert numpy.all(numpy.isfinite(dofs))


def test_a_stationary_solve_still_fails(tmp_path):
    """Nothing can retry a stationary solve, so the failure must stay a failure."""
    problem = _setup(tmp_path)
    with problem:
        _FailAt(problem, nth=2)
        with pytest.raises(Exception):
            problem.solve()


def test_a_singular_matrix_really_does_raise():
    """The premise of the above: Pardiso reports, rather than papers over, a singular matrix.

    Only with repair_bad_solves on, which is NOT the default -- the repairs cost a refactorisation
    each and are off unless asked for. Left to itself Pardiso perturbs the tiny pivot and returns
    error 0 with a solution of order 1e13 -- the documented static-pivoting behaviour, and the reason
    this reaches the time stepper as an ordinary divergence when the repairs are off.

    Which MKL error carries the refusal is a version detail and is deliberately not asserted: on MKL
    2025.0 the refined solve still comes back with error 0, the backward error of 1.0 is what
    condemns it, and the escalated refactorisation then fails in its reordering with -6; the -4 in
    phase 33 that this test originally matched on belongs to another MKL. What must hold on all of
    them is that the huge solution is refused, and refused as a retryable SolverError.
    """
    try:
        from pyoomph.solvers.pardiso import pardisoSolver, PardisoError
    except Exception as e:  # no MKL runtime on this machine
        pytest.skip("Pardiso is not available here: %s" % e)
    import scipy.sparse as sp

    assert issubclass(PardisoError, SolverError), \
        "Pardiso's error is not declared retryable, so a singular Jacobian would end the run"

    n = 6
    A = sp.diags([numpy.arange(1.0, n + 1)], [0]).tolil()
    A[3, 3] = 0.0  # structurally present, numerically zero: a singular pivot
    A = sp.csr_matrix(A)
    rhs = numpy.ones(n)

    unrepaired = pardisoSolver(A, mtype=11, repair_bad_solves=False)
    unrepaired.factor()
    assert numpy.amax(numpy.abs(unrepaired.solve(rhs))) > 1e10, \
        "MKL no longer returns a huge solution here; the premise of solve_checked has changed"

    repaired = pardisoSolver(A, mtype=11, repair_bad_solves=True)
    repaired.factor()
    with pytest.raises(PardisoError, match="singular"):
        repaired.solve_checked(rhs)

    # Off, the same solve returns that huge answer instead of raising -- but the backward error is
    # still measured, which is what lets PardisoSolver discard the factorisation behind it.
    tolerated = pardisoSolver(A, mtype=11, repair_bad_solves=False)
    tolerated.factor()
    assert numpy.amax(numpy.abs(tolerated.solve_checked(rhs))) > 1e10
    assert tolerated.last_backward_error is not None and tolerated.last_backward_error > 1e-4, \
        "a solve this wrong was not diagnosed, so the factorisation behind it would be reused"


def test_a_resolve_refactorises_when_the_factorisation_was_discarded():
    """A discarded factorisation must not make the next resolve impossible.

    _invalidate_factorisation() drops the factors of a solve whose backward error came back over the
    limit. With repair_bad_solves off - the default - that is the ONLY thing that happens to it:
    nothing refactorises on the way to the next call. oomph-lib's resolve (op_flag=2) means "same
    matrix, new right-hand side" and expects the factors to still be there, so it asserted instead,
    and arclength continuation - which resolves for its extra right-hand sides - walked straight into
    it. Three tutorials died that way: hopf_switch, droplet_spread_marangoni_and_gravity and
    rising_bubble.

    Driven through solve_serial directly rather than through a continuation that has to be coaxed
    into a bad solve first: what is under test is that a resolve without factors rebuilds them, and
    the invalidation is exactly what the bad solve would have done.
    """
    try:
        from pyoomph.solvers.pardiso import PardisoSolver
    except Exception as e:  # no MKL runtime on this machine
        pytest.skip("Pardiso is not available here: %s" % e)
    import scipy.sparse as sp

    with _Diffusion() as problem:
        problem.set_linear_solver("pardiso")
        problem.quiet()
        problem.initialise()
        solver = problem.get_la_solver()
        if not isinstance(solver, PardisoSolver):
            pytest.skip("the pardiso solver was not selected here")

        n = 5
        A = sp.csr_matrix(sp.diags([numpy.arange(2.0, n + 2.0)], [0]))
        rhs = numpy.arange(1.0, n + 1.0)
        # op_flag=1 is "factorise this matrix"; oomph hands it CSR triples.
        solver.solve_serial(1, n, A.nnz, 1, A.data.copy(), A.indices.copy(), A.indptr.copy(),
                            rhs.copy(), n, 0)
        assert solver._current_pardiso is not None, "the matrix was not factorised at all"

        solver._invalidate_factorisation()
        assert solver._current_pardiso is None, "the invalidation did not drop the factorisation"

        b = rhs.copy()
        # The matrix arguments are not read at op_flag=2 - a resolve uses the factors and _lastA -
        # so what this asserts is precisely that it can still get to a solution without them.
        solver.solve_serial(2, n, A.nnz, 1, A.data.copy(), A.indices.copy(), A.indptr.copy(),
                            b, n, 0)
        expected = rhs / numpy.arange(2.0, n + 2.0)
        assert numpy.allclose(b, expected), \
            "the resolve returned %s instead of %s" % (b, expected)
