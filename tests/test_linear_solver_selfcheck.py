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

# Does the installed linear solver actually solve?
#
# Written for a failure nobody could place: on macOS the three deflation tests fail with the deflated
# Newton running from a 0.3 perturbation to x = -27 and only the trivial solution being found - on a
# problem with ONE degree of freedom, where the linear solve is a division and no solver's numerical
# quality can be the story. The wheel jobs of 29th August 2026 named the backend
# (MacAccelerateLinearSolver) but nothing established whether it was returning wrong answers or being
# handed a wrong system.
#
# So this asks the solver directly, in the smallest possible terms, and on every platform: build a
# matrix, hand it to whatever backend the problem installed, and compare against numpy. Nothing about
# deflation, nothing about physics - if these fail, the backend is broken for that shape of matrix and
# everything built on it is downstream of that; if they pass, the fault lies in what deflation feeds
# it and this file has narrowed the search by half.
#
# n=1 is deliberately first: the deflation problems ARE one-dimensional, and a 1x1 sparse system is
# exactly the kind of degenerate input a sparse backend is least likely to have been tested on.

"""The installed linear solver, checked against numpy on matrices small enough to verify by hand."""

import numpy
import pytest
import scipy.sparse

from pyoomph import Problem, ODEEquations
from pyoomph.expressions import var, testfunction, partial_t


class _TrivialEqs(ODEEquations):
    """Something for the Problem to be about; the solver is what is under test, not this."""

    def define_fields(self):
        self.define_ode_variable("u")

    def define_residuals(self):
        u = var("u")
        self.add_residual((partial_t(u) + u)*testfunction(u))


class _TrivialProblem(Problem):
    def define_problem(self):
        self += _TrivialEqs() @ "ode"


# (name, matrix). Each is nonsingular and small enough to read.
_MATRICES = {
    "1x1": numpy.array([[2.0]]),
    "1x1_negative": numpy.array([[-0.5]]),
    "2x2_symmetric": numpy.array([[2.0, -1.0], [-1.0, 2.0]]),
    "2x2_nonsymmetric": numpy.array([[1.0, 2.0], [0.0, 3.0]]),
    "3x3_tridiagonal": numpy.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]),
    # A rank-one update on top of a diagonal, which is the SHAPE deflation produces: J/eta + f d^T.
    "3x3_rank_one_update": numpy.eye(3)*3.0 + numpy.outer([1.0, 2.0, -1.0], [0.5, -1.0, 2.0]),
    # Symmetric INDEFINITE, i.e. a saddle point - the shape every interface Lagrange multiplier,
    # constraint and augmented tracker assembles. It is here because this file's first version had
    # nothing of the kind and so did not catch the Accelerate backend factorising exactly such a
    # system with an unpivoted LDL^T: 1600 dofs, symmetric to 2.2e-16, 128 negative eigenvalues, a
    # factorisation reported as successful, and a Newton residual of inf one step later.
    "2x2_saddle": numpy.array([[0.0, 1.0], [1.0, 0.0]]),
    "3x3_saddle": numpy.array([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [1.0, 1.0, 0.0]]),
}


def _solve_through_the_backend(problem, A, b):
    """Factorize and solve A x = b through the problem's own linear solver.

    Returns (x, convention), where convention says whether the backend solved A x = b or A^T x = b -
    reported rather than assumed, because oomph hands its matrix over column-compressed and the
    Python backends read it as CSR, so which of the two a nonsymmetric matrix comes out as is a
    property of the plumbing and not the point of this test.
    """
    n = A.shape[0]
    csr = scipy.sparse.csr_matrix(A)
    csr.sort_indices()
    solver = problem.get_la_solver()
    # The backend picks a symmetric factorization when the PROBLEM's residual is symbolically
    # symmetric (GenericLinearSystemSolver._use_symmetric_factorisation_now), and then reads only one
    # triangle - of ITS matrix, which here is one this test made up rather than the problem's. A
    # nonsymmetric matrix would come back mirrored, failing for a reason that says nothing about the
    # backend. Deflation's own matrix is nonsymmetric too, so this is the mode being tested.
    solver.exploit_proven_symmetry = False
    values = csr.data.astype("float64")
    indices = csr.indices.astype("int32")
    indptr = csr.indptr.astype("int32")
    # op_flag 1 factorizes, op_flag 2 solves in place.
    solver.solve_serial(1, n, int(csr.nnz), 1, values, indices, indptr,
                        numpy.zeros(n, dtype="float64"), n, 0)
    rhs = numpy.array(b, dtype="float64")
    solver.solve_serial(2, n, int(csr.nnz), 1, values, indices, indptr, rhs, n, 0)
    x = numpy.array(rhs, dtype=float)
    residual_normal = float(numpy.amax(numpy.absolute(A @ x - b)))
    residual_transposed = float(numpy.amax(numpy.absolute(A.T @ x - b)))
    convention = "A" if residual_normal <= residual_transposed else "A^T"
    return x, convention, min(residual_normal, residual_transposed)


@pytest.mark.parametrize("name", sorted(_MATRICES))
def test_the_installed_solver_solves(name, tmp_path):
    A = _MATRICES[name]
    n = A.shape[0]
    b = numpy.arange(1.0, n + 1.0)
    with _TrivialProblem() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.quiet()
        problem.initialise()
        backend = type(problem.get_la_solver()).__name__
        x, convention, residual = _solve_through_the_backend(problem, A, b)
    expected = numpy.linalg.solve(A if convention == "A" else A.T, b)
    assert residual < 1e-10, (
        "%s did not solve the %s system: max|Ax-b| = %g with x = %r (expected %r)"
        % (backend, name, residual, x, expected))
    assert numpy.allclose(x, expected, rtol=1e-9, atol=1e-12), (
        "%s returned %r for the %s system, numpy returns %r (solving %s)"
        % (backend, x, name, expected, convention))


def test_the_solver_survives_a_second_matrix_with_the_same_pattern(tmp_path):
    """Re-solve with the SAME sparsity and different values, which is every Newton step.

    The Accelerate backend skips the symbolic factorization when the pattern is unchanged
    (pyoomph/solvers/accelerate.py, reuse_symbolic_factorization) - so "the second solve" is a
    different code path from the first, and it is the one a Newton iteration actually takes. A
    deflated solve changes only the values, which is exactly this case.
    """
    A = _MATRICES["3x3_tridiagonal"]
    B = A * 2.0 + numpy.diag([0.5, -0.25, 0.75])   # same pattern, different numbers
    b = numpy.array([1.0, 2.0, 3.0])
    with _TrivialProblem() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.quiet()
        problem.initialise()
        backend = type(problem.get_la_solver()).__name__
        _, convention, _ = _solve_through_the_backend(problem, A, b)
        x, _, residual = _solve_through_the_backend(problem, B, b)
    expected = numpy.linalg.solve(B if convention == "A" else B.T, b)
    assert residual < 1e-10 and numpy.allclose(x, expected, rtol=1e-9, atol=1e-12), (
        "%s got the second system wrong after one with the same pattern: %r against numpy's %r"
        % (backend, x, expected))
