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

# Correctness gates for the structural (value-independent) Jacobian sparsity pattern, i.e. for
# problem.keep_structural_zeros and problem.jacobian_structure_id. See dev_docs/structural_assembly.md.
#
# What is being protected here:
#
#   By default oomph-lib drops any Jacobian entry that evaluates to exactly zero, so the emitted CSR
#   pattern follows the current degrees of freedom and genuinely does change between Newton steps.
#   keep_structural_zeros makes the pattern a function of the equation numbering alone, and
#   jacobian_structure_id names it so that linear solvers can reuse a symbolic factorisation while it
#   is unchanged. Two failure modes matter, and neither announces itself:
#
#     * the structural pattern MISSING an entry the numerical assembly produces -> silently truncated
#       Jacobian -> Newton converges to the wrong thing (or not at all);
#     * jacobian_structure_id NOT changing when the pattern did -> a solver applies a stale
#       elimination tree to a new pattern -> silently wrong solution, not a crash.
#
#   So the tests below check the superset invariant directly at several dof states, and check that the
#   id moves for every state change that renumbers the system (adaptation, augmentation by bifurcation
#   tracking, switching the active residual) while staying put across things that must NOT invalidate
#   it (Newton steps, arclength continuation steps).

import numpy
import pytest
from scipy.sparse import coo_matrix

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.navier_stokes import NavierStokesEquations
from pyoomph.equations.advection_diffusion import AdvectionDiffusionEquations
from pyoomph.meshes.simplemeshes import RectangularQuadMesh


# ---------------------------------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------------------------------

class _CavityProblem(Problem):
    """Lid-driven cavity. Taylor-Hood, so the pressure-pressure block is structurally zero -- which is
    exactly the case where the numerically-filtered pattern has empty diagonal entries."""

    def __init__(self, N=6, with_scalar=False):
        super().__init__()
        self.N = N
        self.with_scalar = with_scalar

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=self.N))
        eqs = NavierStokesEquations(dynamic_viscosity=0.1, mass_density=1)
        if self.with_scalar:
            # A weakly coupled extra field: c is advected by the flow, but nothing in the flow equations
            # depends on c. The pure connectivity pattern therefore over-allocates the c-vs-flow blocks.
            eqs += AdvectionDiffusionEquations(fieldnames="c", diffusivity=0.05, space="C2")
            eqs += DirichletBC(c=1) @ "left"
        for b in ["left", "right", "bottom"]:
            eqs += DirichletBC(velocity_x=0, velocity_y=0) @ b
        eqs += DirichletBC(velocity_x=1, velocity_y=0) @ "top"
        eqs += DirichletBC(pressure=0) @ "bottom/left"
        self.add_equations(eqs @ "domain")


class _BratuProblem(Problem):
    """-laplace(u) + dt(u) = lam*exp(u): a turning point in lam, for the continuation/fold tests."""

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=6))
        lam = self.get_global_parameter("lam")

        class _Bratu(Equations):
            def define_fields(self):
                self.define_scalar_field("u", "C2")

            def define_residuals(self):
                u, v = var_and_test("u")
                self.add_residual(weak(partial_t(u), v) + weak(grad(u), grad(v)) - weak(lam * exp(u), v))

        eqs = _Bratu()
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        self.add_equations(eqs @ "domain")


# ---------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------

def _connectivity_pattern(problem, meshname="domain"):
    """The pattern implied by element connectivity alone: every (row,col) pair of equation numbers that
    share an element. This is what keep_structural_zeros should produce, computed here independently
    from Python so the test does not just compare the implementation against itself."""
    mesh = problem.get_mesh(meshname)
    rows, cols = [], []
    for e in range(mesh.nelement()):
        el = mesh.element_pt(e)
        eq = numpy.array([el.eqn_number(i) for i in range(el.ndof())])
        eq = eq[eq >= 0]
        rows.append(numpy.repeat(eq, len(eq)))
        cols.append(numpy.tile(eq, len(eq)))
    rows = numpy.concatenate(rows)
    cols = numpy.concatenate(cols)
    n = problem.ndof()
    S = coo_matrix((numpy.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    S.sum_duplicates()
    return S


def _pattern(J):
    return J.indptr.copy(), J.indices.copy()


def _rows_without_diagonal(J):
    return sum(1 for i in range(J.shape[0])
               if not (J.indices[J.indptr[i]:J.indptr[i + 1]] == i).any())


# ---------------------------------------------------------------------------------------------------
# Gate 1 -- the pattern really is value-independent, and really does contain the numerical one
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("with_scalar", [False, True], ids=["navier_stokes", "coupled"])
def test_structural_pattern_is_value_independent(with_scalar):
    with _CavityProblem(with_scalar=with_scalar) as p:
        p.quiet()
        p.initialise()
        p.keep_structural_zeros = True

        J_zero = p.assemble_jacobian(with_residual=False)
        pat = _pattern(J_zero)
        p.solve()
        J_solved = p.assemble_jacobian(with_residual=False)

        assert numpy.array_equal(pat[0], J_solved.indptr)
        assert numpy.array_equal(pat[1], J_solved.indices), \
            "the pattern must not depend on the degrees of freedom"

        # ... and it is genuinely the element-connectivity pattern, computed independently.
        S = _connectivity_pattern(p)
        assert J_solved.nnz == S.nnz
        assert numpy.array_equal(J_solved.indptr, S.indptr)
        assert numpy.array_equal(numpy.sort(J_solved.indices), numpy.sort(S.indices))


def test_default_pattern_does_change_with_the_dofs():
    """The premise of the whole exercise: with the default value filter the pattern is not reusable.

    Note it is not reusable rather than always different -- for the plain cavity it happens to come out
    identical at both states, because the entries the convective term contributes at U=0 are already
    occupied by the viscous term. The coupled problem is the honest witness: the advection term's
    derivative w.r.t. velocity is identically zero while c is uniform, so ~3200 entries only appear once
    the concentration develops a gradient. A solver cannot know in advance which case it is in, which is
    precisely why jacobian_structure_id refuses to advertise a pattern here at all."""
    with _CavityProblem(with_scalar=True) as p:
        p.quiet()
        p.initialise()
        assert p.keep_structural_zeros is False
        assert p.jacobian_structure_id == 0, "no pattern may be advertised while zeros are filtered"
        J_zero = p.assemble_jacobian(with_residual=False)
        p.solve()
        J_solved = p.assemble_jacobian(with_residual=False)
        assert J_solved.nnz > J_zero.nnz
        assert not (numpy.array_equal(J_zero.indptr, J_solved.indptr)
                    and numpy.array_equal(J_zero.indices, J_solved.indices))


@pytest.mark.parametrize("with_scalar", [False, True], ids=["navier_stokes", "coupled"])
def test_structural_pattern_is_a_superset_with_identical_values(with_scalar):
    """The differential test against the value-filtered path: same values everywhere the filtered
    assembly has an entry, and every extra entry is exactly zero."""
    with _CavityProblem(with_scalar=with_scalar) as p:
        p.quiet()
        p.initialise()
        p.solve()

        p.keep_structural_zeros = False
        J_ref = p.assemble_jacobian(with_residual=False)
        p.keep_structural_zeros = True
        J_str = p.assemble_jacobian(with_residual=False)

        assert J_str.nnz >= J_ref.nnz
        D = (J_str - J_ref).tocsr()
        D.eliminate_zeros()
        assert D.nnz == 0, "structural assembly changed a value"
        # Every entry the filtered assembly produced must be present structurally (a missing one would
        # not show up above, since the difference of two sparse matrices is taken on the union).
        missing = ((J_ref != 0).astype(int) - (J_str != 0).astype(int) > 0)
        assert missing.nnz == 0


@pytest.mark.parametrize("with_scalar", [False, True], ids=["navier_stokes", "coupled"])
def test_structural_pattern_contains_the_full_diagonal(with_scalar):
    """The 'add zeros on the diagonal' requirement. Taylor-Hood has an empty pressure-pressure block, so
    the filtered pattern leaves those diagonals out and PETSc's LU refuses the matrix outright."""
    with _CavityProblem(with_scalar=with_scalar) as p:
        p.quiet()
        p.initialise()
        p.solve()

        p.keep_structural_zeros = False
        assert _rows_without_diagonal(p.assemble_jacobian(with_residual=False)) > 0, \
            "expected the filtered pattern to be missing pressure diagonals"
        p.keep_structural_zeros = True
        assert _rows_without_diagonal(p.assemble_jacobian(with_residual=False)) == 0


# ---------------------------------------------------------------------------------------------------
# The mass matrix keeps its own, much tighter pattern
# ---------------------------------------------------------------------------------------------------

def _eigen_matrices(p):
    """(M, J) from the two-matrix eigenproblem assembly. Copies, because the returned arrays are
    zero-copy views onto oomph's buffers and the next assembly reallocates them."""
    n, _, _, Mv, Mc, Mr, _, _, Jv, Jc, Jr = p.assemble_eigenproblem_matrices(0.0)
    from scipy.sparse import csr_matrix
    return (csr_matrix((Mv, Mc, Mr), shape=(n, n)).copy(),
            csr_matrix((Jv, Jc, Jr), shape=(n, n)).copy())


@pytest.mark.parametrize("with_scalar", [False, True], ids=["navier_stokes", "coupled"])
def test_mass_matrix_is_not_inflated_to_the_jacobian_pattern(with_scalar):
    """Only fields carrying a time derivative appear in the mass matrix at all, so M is several times
    sparser than J. Handing it J's connectivity pattern would inflate it for nothing -- what a stable
    pattern buys is symbolic-factorisation reuse, and the operator being factorised is J. So
    keep_structural_zeros must apply to the Jacobian and leave the mass matrix alone."""
    with _CavityProblem(N=8, with_scalar=with_scalar) as p:
        p.quiet()
        p.initialise()
        p.solve()

        p.keep_structural_zeros = False
        M_ref, J_ref = _eigen_matrices(p)
        p.keep_structural_zeros = True
        M_str, J_str = _eigen_matrices(p)

        assert J_str.nnz > J_ref.nnz, "the Jacobian should have gained its structural zeros"
        assert M_str.nnz == M_ref.nnz, "the mass matrix must keep its own pattern"
        assert M_str.nnz < J_str.nnz / 2, "expected the mass matrix to be several times sparser"

        # ... and opting in must actually opt in.
        p.keep_structural_zeros_in_mass_matrix = True
        M_all, J_all = _eigen_matrices(p)
        assert M_all.nnz == J_all.nnz


@pytest.mark.parametrize("with_scalar", [False, True], ids=["navier_stokes", "coupled"])
def test_mass_matrix_entries_lie_inside_the_structural_jacobian_pattern(with_scalar):
    """Why the mixed policy is safe for eigenproblems: a shift-and-invert solve factorises J - sigma*M,
    whose pattern is the union of the two. Since M's entries all sit inside J's structural pattern, that
    union IS J's pattern -- so it is still value-independent and still reusable, even though M itself is
    stored on a tighter pattern."""
    with _CavityProblem(N=8, with_scalar=with_scalar) as p:
        p.quiet()
        p.initialise()
        p.solve()
        p.keep_structural_zeros = True
        M, J = _eigen_matrices(p)
        outside = ((M != 0).astype(int) - (J != 0).astype(int) > 0)
        assert outside.nnz == 0


def test_mass_matrix_values_are_unchanged_by_the_jacobian_policy():
    with _CavityProblem(N=6) as p:
        p.quiet()
        p.initialise()
        p.solve()
        p.keep_structural_zeros = False
        M_ref, _ = _eigen_matrices(p)
        p.keep_structural_zeros = True
        M_str, _ = _eigen_matrices(p)
        D = (M_str - M_ref).tocsr()
        D.eliminate_zeros()
        assert D.nnz == 0


def test_hessian_tensor_is_not_given_structural_zeros():
    """The Hessian is a rank-3 tensor: keeping structural zeros would store every (i,j,k) triple of an
    element, i.e. nvar^3 rather than nvar^2 entries. Nothing factorises it, so a stable pattern buys
    nothing and the blow-up is pure cost. Guard against it being wired into the per-matrix policy by
    accident later."""
    with _CavityProblem(N=4) as p:
        p.quiet()
        p.setup_for_stability_analysis(analytic_hessian=True)
        p.initialise()
        p.solve()
        p.keep_structural_zeros = False
        n_ref = p._assemble_hessian_tensor(False).finalize_for_vector_product()[1][-1]
        p.keep_structural_zeros = True
        n_str = p._assemble_hessian_tensor(False).finalize_for_vector_product()[1][-1]
        assert n_str == n_ref


# ---------------------------------------------------------------------------------------------------
# Gate 2 -- the solution is unchanged
# ---------------------------------------------------------------------------------------------------

def test_newton_solution_is_unchanged():
    ref = None
    for flag in (False, True):
        with _CavityProblem() as p:
            p.quiet()
            p.initialise()
            p.keep_structural_zeros = flag
            p.solve()
            got = numpy.asarray(p.get_history_dofs(0)).copy()
            if ref is None:
                ref = got
            else:
                assert numpy.allclose(ref, got, rtol=0, atol=1e-12)


def test_arclength_continuation_is_unchanged_and_keeps_the_pattern():
    """Arclength uses oomph's bordering algorithm: one factorisation plus a resolve against a second
    right-hand side, with the Jacobian itself untouched. So the pattern must survive a continuation step
    -- if it did not, every step would throw away the symbolic factorisation it is meant to reuse."""
    trajectories = []
    for flag in (False, True):
        with _BratuProblem() as p:
            p.quiet()
            p.initialise()
            p.keep_structural_zeros = flag
            lam = p.get_global_parameter("lam")
            lam.value = 0.5
            p.solve()
            sid_before = p.jacobian_structure_id
            ds, traj = 0.2, []
            for _ in range(6):
                ds = p.arclength_continuation("lam", ds)
                traj.append((float(lam.value), float(numpy.max(p.get_history_dofs(0)))))
            assert p.jacobian_structure_id == sid_before, \
                "arclength continuation must not invalidate the sparsity pattern"
            trajectories.append(traj)
    for a, b in zip(*trajectories):
        assert a == pytest.approx(b, rel=0, abs=1e-9)


# ---------------------------------------------------------------------------------------------------
# Gate 4 -- invalidation coverage. A missed invalidation is a wrong answer, not a crash.
# ---------------------------------------------------------------------------------------------------

def test_structure_id_is_zero_unless_the_pattern_is_usable():
    with _CavityProblem() as p:
        p.quiet()
        p.initialise()
        assert p.jacobian_structure_id == 0
        p.keep_structural_zeros = True
        assert p.jacobian_structure_id != 0
        p.keep_structural_zeros = False
        assert p.jacobian_structure_id == 0


def test_structure_id_survives_newton_steps():
    with _CavityProblem() as p:
        p.quiet()
        p.initialise()
        p.keep_structural_zeros = True
        sid = p.jacobian_structure_id
        p.solve()
        assert p.jacobian_structure_id == sid
        p.solve()
        assert p.jacobian_structure_id == sid


def test_structure_id_changes_on_renumbering():
    with _CavityProblem() as p:
        p.quiet()
        p.initialise()
        p.keep_structural_zeros = True
        sid = p.jacobian_structure_id
        p.assign_eqn_numbers()
        assert p.jacobian_structure_id != sid


def test_structure_id_changes_when_the_active_residual_is_switched():
    """A multi-residual problem has a different Jacobian per residual/Jacobian combination -- different
    field couplings, and different fields pinned for having an empty Jacobian row. Switching between
    them via _set_solved_residual() does not renumber anything, so, like augmentation, this is caught
    only by the lazy re-validation in Problem::get_jacobian_structure_id()."""

    class _TwoResidualProblem(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=4))

            class _Eqs(Equations):
                def define_fields(self):
                    self.define_scalar_field("u", "C2")

                def define_residuals(self):
                    u, v = var_and_test("u")
                    self.add_residual(weak(grad(u), grad(v)) - weak(1, v))
                    # A second, differently-coupled residual living on the same fields.
                    self.add_residual(weak(u, v) - weak(2, v), destination="alt")

            eqs = _Eqs()
            for b in ["left", "right", "top", "bottom"]:
                eqs += DirichletBC(u=0) @ b
            self.add_equations(eqs @ "domain")

    with _TwoResidualProblem() as p:
        p.quiet()
        p.initialise()
        p.keep_structural_zeros = True
        sid_default = p.jacobian_structure_id
        assert p._set_solved_residual("alt", True, True)
        sid_alt = p.jacobian_structure_id
        assert sid_alt != sid_default, "switching the active residual must invalidate the pattern"
        p._set_solved_residual("", False, True)
        assert p.jacobian_structure_id != sid_alt


def test_structure_id_changes_when_the_dof_vector_is_augmented():
    """Bifurcation tracking installs an assembly handler with its own eqn_number() over an augmented dof
    vector, which is a different matrix entirely. Nothing in assign_eqn_numbers() sees that, so this
    relies on the lazy re-validation in Problem::get_jacobian_structure_id()."""
    with _BratuProblem() as p:
        p.quiet()
        p.initialise()
        p.keep_structural_zeros = True
        lam = p.get_global_parameter("lam")
        lam.value = 3.0
        p.solve()
        sid_plain = p.jacobian_structure_id
        ndof_plain = p.ndof()

        p.activate_bifurcation_tracking("lam", "fold")
        p.solve()
        assert p.ndof() > ndof_plain
        sid_augmented = p.jacobian_structure_id
        assert sid_augmented != sid_plain, "augmenting the dof vector must invalidate the pattern"

        p.deactivate_bifurcation_tracking()
        assert p.ndof() == ndof_plain
        assert p.jacobian_structure_id != sid_augmented
