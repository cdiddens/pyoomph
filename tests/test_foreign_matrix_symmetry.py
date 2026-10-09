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

# A matrix that is NOT the problem's Jacobian must not inherit the Jacobian's symmetry proof.
#
# The proof behind exploit_proven_symmetry (Problem::get_proven_matrix_symmetry, covered in
# test_symmetric_solver_switch.py) is a statement about the JACOBIAN. Several entry points hand the
# backends a different matrix entirely through the same solve_serial() call the Newton solve uses --
# the Python-built distributed entry points (a Lyapunov operator, a bordered normal-form system,
# anything going through LinearAlgebraBackend.solve), PeriodicDrivingResponse's frequency-response
# pencil, Halley's J-dJdU/2 -- and those used to be routed through the Jacobian's verdict.
#
# That is a WRONG ANSWER, not a missed optimisation, and it is the quietest kind. A symmetric
# factorisation reads only one triangle: Pardiso's mtype -2 takes sp.triu(A) and throws the lower
# triangle away. So on a problem whose Jacobian is proven symmetric, an unsymmetric Python-built
# matrix came back as the solution of its own symmetrisation -- measured at 4.2 % off on the 3x3
# system below, and matching the symmetrised solution to 1.1e-16. Nothing warned and nothing raised.
#
# The expectations here are therefore written as BOTH solutions: the true one, which must come back,
# and the symmetrised one, which must not. A test that only asserted the true solution would still
# pass if some later change symmetrised a matrix that happens to be symmetric anyway, and would give
# no clue what went wrong when it failed. Both are literals, from numpy.linalg.solve on the dense
# matrix, so they are independent of what pyoomph does with it.
#
# The gate is GenericLinearSystemSolver._solving_foreign_matrix(), one flag honoured by
# _use_symmetric_factorisation_now(), so that all four backends that act on the verdict (pardiso
# mtype -2, mumps sym=2, accelerate ldlt_sbk, petsc MAT_SYMMETRIC) are covered at once. The other
# half of the contract matters just as much and is checked below: the Jacobian's own symmetric
# factorisation must survive this, i.e. the fix must not be "stop proving symmetry".

import inspect

import numpy
import pytest
import scipy.sparse

from pyoomph import *
from pyoomph.expressions import *


# ---------------------------------------------------------------------------------------------------
# The systems, and both solutions of each
# ---------------------------------------------------------------------------------------------------

# Unsymmetric, well conditioned, nothing else special about it: the operator a Lyapunov sweep of the
# Lorenz tutorial actually handed over, which is how this was found.
_A = numpy.array([[110.0,     -10.0,     0.0],
                  [ -9.66149, 101.0,    -3.84998],
                  [  5.19497,   3.84998, 102.66667]])
_B = numpy.array([72.400696, 1.878625, 68.95368])
_A_SOLUTION = [0.66788357, 0.10664967, 0.63383224]
_A_SYMMETRISED = [0.66823557, 0.11052165, 0.67577127]

# The shape the bordered/augmented paths produce: a symmetric core (a 1D Laplacian), a parameter
# COLUMN and a normalisation ROW that are not each other's transpose. Unsymmetric by construction
# even where the base Jacobian is symmetric, which is exactly the case this defect bit hardest.
_BORDERED = numpy.array([[ 4.0, -1.0,  0.0, 1.0],
                         [-1.0,  4.0, -1.0, 0.0],
                         [ 0.0, -1.0,  4.0, 0.0],
                         [ 0.0,  0.0,  2.0, 1.0]])
_BORDERED_B = numpy.array([1.0, 2.0, 3.0, 1.0])
_BORDERED_SOLUTION = [0.72222222, 0.92592593, 0.98148148, -0.96296296]
_BORDERED_SYMMETRISED = [0.26829268, 0.80487805, 0.95121951, 0.73170732]


class _ThreeDecoupledODEs(ODEEquations):
    """Three independent dv/dt = -v, so the Jacobian is DIAGONAL and trivially proven symmetric.

    Deliberately trivial: the point is the solver's verdict about this problem, not the problem.
    """

    def define_fields(self):
        self.define_ode_variable("x", "y", "z")

    def define_residuals(self):
        for name in ("x", "y", "z"):
            v, vt = var_and_test(name)
            self.add_residual((partial_t(v) + v) * vt)


class _SymmetricJacobianProblem(Problem):
    def define_problem(self):
        self += _ThreeDecoupledODEs() @ "d"


def _solved():
    """A solved _SymmetricJacobianProblem, so a linear solver exists and has factorised once."""
    p = _SymmetricJacobianProblem()
    p.__enter__()
    p.quiet()
    p.solve()
    return p


# ---------------------------------------------------------------------------------------------------
# The premise
# ---------------------------------------------------------------------------------------------------

def test_the_jacobian_of_this_problem_is_proven_symmetric():
    """Without this every assertion below would pass for the wrong reason."""
    p = _solved()
    try:
        assert p._get_proven_matrix_symmetry("") == (True, True)
        assert p.get_la_solver().exploit_proven_symmetry is True, "the switch under test is off"
        assert p.get_la_solver().last_symmetry_decision is True, \
            "the Newton solve did not take the symmetric path, so this problem cannot show the defect"
    finally:
        p.__exit__(None, None, None)


# ---------------------------------------------------------------------------------------------------
# The defect itself
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("many", [False, True], ids=["one_rhs", "many_rhs"])
def test_a_python_built_unsymmetric_system_is_not_symmetrised(many):
    p = _solved()
    try:
        la = p.get_la_solver()
        mat = scipy.sparse.csr_matrix(_A)
        rhs = numpy.ascontiguousarray(_B)
        if many:
            got = la.solve_python_built_distributed_many(3, 3, 0, mat, [rhs])[0]
        else:
            got = la.solve_python_built_distributed(3, 3, 0, mat, rhs)
        got = numpy.asarray(got, dtype=float)
        assert got == pytest.approx(_A_SOLUTION, abs=1e-8)
        # Said separately and with a message, because this is the fingerprint: a failure that lands
        # here rather than above means the lower triangle was discarded, not that the solve was noisy.
        assert got != pytest.approx(_A_SYMMETRISED, abs=1e-6), \
            "the returned vector solves triu(A)+triu(A).T, i.e. the matrix was symmetrised"
        assert la.last_symmetry_decision is False, la.last_symmetry_decision_reason
        assert "not the problem's Jacobian" in la.last_symmetry_decision_reason
    finally:
        p.__exit__(None, None, None)


def test_a_bordered_system_through_the_backend_is_not_symmetrised():
    """The route the augmented paths take: LinearAlgebraBackend.solve, which is the entry point above
    one level up (NormalFormCalculator._solve_bordered_real, FoldTracker's bordered solves).
    """
    from pyoomph.generic.distributed_la import RowLayout, get_la_backend
    p = _solved()
    try:
        la = p.get_la_solver()
        backend = get_la_backend(p)
        layout = RowLayout.serial(4)
        A = backend.matrix(scipy.sparse.csr_matrix(_BORDERED), layout, 4)
        b = backend.vector(numpy.ascontiguousarray(_BORDERED_B), layout)
        got = numpy.asarray(backend.solve(A, b).local, dtype=float)
        assert got == pytest.approx(_BORDERED_SOLUTION, abs=1e-7)
        assert got != pytest.approx(_BORDERED_SYMMETRISED, abs=1e-6), \
            "the bordered system was factorised symmetrically; its normalisation row was discarded"
        assert la.last_symmetry_decision is False, la.last_symmetry_decision_reason
    finally:
        p.__exit__(None, None, None)


# ---------------------------------------------------------------------------------------------------
# The other half: the Jacobian keeps its symmetric factorisation
# ---------------------------------------------------------------------------------------------------

def test_the_symmetric_path_comes_back_for_the_jacobian_itself():
    """The fix must gate the foreign matrix, not disable the proof.

    The flip back is where this kind of bug hides -- the same place test_symmetric_solver_switch.py
    checks it for a bifurcation tracker being removed -- so the Newton solve AFTER a foreign solve is
    what is asserted, not one before it.
    """
    p = _solved()
    try:
        la = p.get_la_solver()
        la.solve_python_built_distributed(3, 3, 0, scipy.sparse.csr_matrix(_A),
                                         numpy.ascontiguousarray(_B))
        assert la.last_symmetry_decision is False
        assert la._foreign_matrix_solve is False, "the flag outlived the solve it was set for"

        # Perturbed first, or the next solve converges in zero steps and factorises nothing.
        dofs, _ = p.get_current_dofs()
        p.set_current_dofs((numpy.array(dofs) + 0.01).tolist())
        p.solve()
        assert la.last_symmetry_decision is True, la.last_symmetry_decision_reason
        if type(la).__name__ == "PardisoSolver":
            assert la._current_pardiso is not None and la._current_pardiso.mtype == -2
    finally:
        p.__exit__(None, None, None)


def test_the_flag_is_released_when_the_solve_raises():
    """solve_serial() raising is routine (a failed factorisation is a SolverError), and a flag left
    set would silently cost every later Newton solve its symmetric factorisation."""
    p = _solved()
    try:
        la = p.get_la_solver()
        boom = RuntimeError("pretend the factorisation failed")
        with pytest.raises(RuntimeError, match="pretend"):
            with la._solving_foreign_matrix():
                assert la._foreign_matrix_solve is True
                raise boom
        assert la._foreign_matrix_solve is False
        assert la._use_symmetric_factorisation_now() is True
    finally:
        p.__exit__(None, None, None)


def test_the_gate_is_re_entrant():
    """Nested wraps must not release the flag at the inner exit -- a caller that wraps a routine which
    wraps its own solve is the ordinary case (LinearAlgebraBackend.solve inside a utility)."""
    p = _solved()
    try:
        la = p.get_la_solver()
        with la._solving_foreign_matrix():
            with la._solving_foreign_matrix():
                assert la._foreign_matrix_solve is True
            assert la._foreign_matrix_solve is True, "the inner exit released the outer flag"
        assert la._foreign_matrix_solve is False
    finally:
        p.__exit__(None, None, None)


# ---------------------------------------------------------------------------------------------------
# The other call sites, and the one path that must NOT need the gate
# ---------------------------------------------------------------------------------------------------

def _record_op_flag_1(la, store):
    """Record (flag, verdict) at every factorisation the solver is asked for."""
    orig = la.solve_serial

    def wrapped(op_flag, *a, **kw):
        if op_flag == 1:
            store.append((la._foreign_matrix_solve, la._use_symmetric_factorisation_now()))
        return orig(op_flag, *a, **kw)

    la.solve_serial = wrapped


class _DrivenODE(ODEEquations):
    def __init__(self, driving):
        super().__init__()
        self.driving = driving

    def define_fields(self):
        self.define_ode_variable("x")

    def define_residuals(self):
        x, xt = var_and_test("x")
        self.add_residual((partial_t(x) + x - self.driving) * xt)


class _DrivenProblem(Problem):
    def __init__(self):
        super().__init__()
        self.driving = 0

    def define_problem(self):
        self += _DrivenODE(self.driving) @ "o"


def test_the_periodic_driving_pencil_is_solved_as_a_foreign_matrix():
    """PeriodicDrivingResponse's serial branch calls solve_serial() directly with its bordered pencil.

    Checked on the FLAG rather than on a wrong answer, and honestly so: the _DrivingForResponse
    equations this class injects are themselves unsymmetric (dEQ_y/dd = -1 against
    dEQ_yp/ddp = omega**2), so _get_proven_matrix_symmetry comes out False for any problem it is
    attached to and no case is known today in which the verdict would have been True here. The gate
    is still right -- the pencil is foreign whatever the Jacobian turns out to be -- and this pins
    that it is in place, which a verdict assertion could not distinguish.
    """
    from pyoomph.utils.periodic_driving_response import PeriodicDrivingResponse
    p = _DrivenProblem()
    p.__enter__()
    try:
        pdr = PeriodicDrivingResponse(p)
        p.driving = 1.0 * pdr.get_driving_mode()
        p.quiet()
        p.solve()
        seen = []
        _record_op_flag_1(p.get_la_solver(), seen)
        pdr.new_solve_driving_response(omega=0.7)
        assert seen, "the response solve did not factorise anything, so nothing was checked"
        assert all(flag and verdict is False for flag, verdict in seen), seen
    finally:
        p.__exit__(None, None, None)


def test_the_petsc_auxiliary_path_does_not_consult_the_symmetry_verdict():
    """PETSc overrides the Python-built entry points with its own _aux_ Mat/KSP, which is a plain LU
    and never asks about symmetry -- so that path was already correct and needs no gate. Asserted on
    the source, because it is the ABSENCE of a call that is being protected and a behavioural test
    cannot see an absence. petsc4py is not imported, so this runs everywhere.
    """
    from pyoomph.solvers import petsc as petsc_module
    src = inspect.getsource(petsc_module)
    body = src[src.index("def _aux_prepare"):src.index("def _aux_backsolve")]
    for forbidden in ("_use_symmetric_factorisation_now", "_update_symmetry_engagement",
                      "_symmetric_engaged", "_apply_mat_symmetry_option"):
        assert forbidden not in body, \
            ("PETSCSolver._aux_prepare now consults " + forbidden + ". The auxiliary path solves a "
             "matrix that is not the Jacobian, so it must either stay clear of the symmetry verdict "
             "or go through _solving_foreign_matrix().")
