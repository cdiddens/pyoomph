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
from scipy.sparse import coo_matrix, identity

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

@pytest.mark.parametrize("prune", [False, True], ids=["tier_a_connectivity", "tier_b_field_coupling"])
@pytest.mark.parametrize("with_scalar", [False, True], ids=["navier_stokes", "coupled"])
def test_structural_pattern_is_value_independent(prune, with_scalar):
    with _CavityProblem(with_scalar=with_scalar) as p:
        p.quiet()
        p.initialise()
        p.prune_structural_zeros_by_field_coupling = prune
        p.keep_structural_zeros = True

        J_zero = p.assemble_jacobian(with_residual=False)
        pat = _pattern(J_zero)
        p.solve()
        J_solved = p.assemble_jacobian(with_residual=False)

        assert numpy.array_equal(pat[0], J_solved.indptr)
        assert numpy.array_equal(pat[1], J_solved.indices), \
            "the pattern must not depend on the degrees of freedom"

        S = _connectivity_pattern(p)
        if not prune:
            # Tier A: genuinely the element-connectivity pattern, computed independently.
            assert J_solved.nnz == S.nnz
            assert numpy.array_equal(J_solved.indptr, S.indptr)
            assert numpy.array_equal(numpy.sort(J_solved.indices), numpy.sort(S.indices))
        else:
            # Tier B: connectivity pruned to the field pairs that can contribute, so strictly tighter,
            # and still a superset of everything the value-filtered assembly produces.
            assert J_solved.nnz <= S.nnz
            assert ((S > 0).astype(int) - (J_solved != 0).astype(int) > 0).nnz > 0 or J_solved.nnz == S.nnz


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
        p.keep_structural_zeros = False   # the pre-Phase-3 behaviour, now no longer the default
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
def test_full_diagonal_is_available_on_request(with_scalar):
    """The "add zeros on the diagonal" requirement. Taylor-Hood has an empty pressure-pressure block, so
    neither the value-filtered pattern nor the field-pruned one contains those diagonals, and some PETSc
    factorisations refuse such a matrix outright. force_jacobian_diagonal_entries supplies them.

    It is OFF by default: only some PETSc factorisations need it (MUMPS does not, and PETSc can insert
    them itself where it does), and stored zeros are not free -- they change the matrix a direct solver
    sees, hence its pivoting. So this checks the option is available and exact, not that it is on."""
    with _CavityProblem(with_scalar=with_scalar) as p:
        p.quiet()
        p.initialise()
        p.solve()

        assert p.force_jacobian_diagonal_entries is False, "expected the forced diagonal to be opt-in"
        p.keep_structural_zeros = False
        assert _rows_without_diagonal(p.assemble_jacobian(with_residual=False)) > 0, \
            "expected the filtered pattern to be missing pressure diagonals"
        p.keep_structural_zeros = True
        assert _rows_without_diagonal(p.assemble_jacobian(with_residual=False)) > 0, \
            "the pruned pattern should not invent diagonals unless asked"
        p.force_jacobian_diagonal_entries = True
        assert _rows_without_diagonal(p.assemble_jacobian(with_residual=False)) == 0
        # Tier A gets the full diagonal for free: connectivity always contains (i,i).
        p.force_jacobian_diagonal_entries = False
        p.prune_structural_zeros_by_field_coupling = False
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

        assert J_str.nnz >= J_ref.nnz, "the structural pattern must contain the numerical one"
        assert M_str.nnz == M_ref.nnz, "the mass matrix must keep its own pattern"
        assert M_str.nnz < J_str.nnz / 2, "expected the mass matrix to be several times sparser"

        # Under Tier A the mass matrix can be opted in to the Jacobian's connectivity pattern, which is
        # the only way it ever gets a value-independent pattern there. Under Tier B that opt-in is
        # superseded: contributes_to_mass_matrix gives the mass matrix a pattern that is BOTH
        # value-independent and ~3x tighter than the Jacobian's, so there is nothing to opt in to.
        p.prune_structural_zeros_by_field_coupling = False
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
# Phase 3 -- the symbolic field-coupling tables and the per-element dof attribution they are indexed by
# ---------------------------------------------------------------------------------------------------

def _masked_pattern(problem, which, meshname="domain"):
    """The sparsity pattern implied by codegen's symbolic field-coupling table for `which` ("jacobian"
    or "mass"), i.e. element connectivity pruned to the field pairs that can contribute at all."""
    mesh = problem.get_mesh(meshname)
    rows, cols = [], []
    for e in range(mesh.nelement()):
        el = mesh.element_pt(e)
        _names, jac, mass = el._get_contribution_tables()
        table = numpy.array(jac if which == "jacobian" else mass, dtype=bool)
        cidx = numpy.array(el._get_dof_contribution_indices())
        eq = numpy.array([el.eqn_number(i) for i in range(el.ndof())])
        keep = eq >= 0
        eq, ci = eq[keep], cidx[keep]
        R, C = numpy.repeat(eq, len(eq)), numpy.tile(eq, len(eq))
        FR, FC = numpy.repeat(ci, len(ci)), numpy.tile(ci, len(ci))
        # An unattributed dof (-1) must be read conservatively as "coupled to everything", never as
        # "decoupled" -- see BulkElementBase::get_local_dof_contribution_indices.
        sel = (FR < 0) | (FC < 0)
        known = ~sel
        sel[known] = table[FR[known], FC[known]]
        rows.append(R[sel])
        cols.append(C[sel])
    n = problem.ndof()
    S = coo_matrix((numpy.ones(len(numpy.concatenate(rows))),
                    (numpy.concatenate(rows), numpy.concatenate(cols))), shape=(n, n)).tocsr()
    S.sum_duplicates()
    return S


@pytest.mark.parametrize("with_scalar", [False, True], ids=["navier_stokes", "coupled"])
def test_every_local_dof_is_attributed_to_a_field(with_scalar):
    """The mask is only as good as the local dof -> contribution map behind it. An unattributed dof is
    handled safely (assume coupled), but it costs density, so it should not happen for ordinary
    problems -- and if it starts happening, that is worth being told about."""
    with _CavityProblem(N=6, with_scalar=with_scalar) as p:
        p.quiet()
        p.initialise()
        p.solve()
        mesh = p.get_mesh("domain")
        unattributed = sum(1 for e in range(mesh.nelement())
                           for v in mesh.element_pt(e)._get_dof_contribution_indices() if v < 0)
        assert unattributed == 0


@pytest.mark.parametrize("which", ["jacobian", "mass"])
@pytest.mark.parametrize("with_scalar", [False, True], ids=["navier_stokes", "coupled"])
def test_field_coupling_mask_covers_the_numerical_pattern(which, with_scalar):
    """The invariant the whole of Phase 3 rests on: the symbolic table is a SUPERSET of whatever is
    numerically nonzero. If it ever under-reports, the assembled matrix is silently truncated."""
    with _CavityProblem(N=6, with_scalar=with_scalar) as p:
        p.quiet()
        p.initialise()
        p.solve()
        M, J = _eigen_matrices(p)
        numerical = J if which == "jacobian" else M
        masked = _masked_pattern(p, which)
        missing = ((numerical != 0).astype(int) - (masked > 0).astype(int) > 0)
        assert missing.nnz == 0, "%d numerically nonzero entries lie outside the symbolic mask" % missing.nnz


@pytest.mark.parametrize("with_scalar", [False, True], ids=["navier_stokes", "coupled"])
def test_tier_b_assembles_exactly_the_masked_pattern(with_scalar):
    """End to end: with pruning on, what actually comes out of the assembly is the field-masked pattern
    plus the forced diagonal -- nothing more, nothing less -- for BOTH matrices of the eigenproblem
    assembly. The mass matrix is the interesting half: it gets its own tight pattern from
    contributes_to_mass_matrix rather than the Jacobian's."""
    with _CavityProblem(N=6, with_scalar=with_scalar) as p:
        p.quiet()
        p.initialise()
        p.keep_structural_zeros = True   # pruning is the default; the forced diagonal is not
        p.force_jacobian_diagonal_entries = True
        p.solve()
        M, J = _eigen_matrices(p)

        # force_jacobian_diagonal_entries: add the diagonal to the pattern. Done as a matrix sum
        # rather than entry by entry -- assigning into a csr_matrix reallocates the whole thing per
        # entry (and warns about it).
        expect_J = ((_masked_pattern(p, "jacobian") > 0).astype(int)
                    + identity(J.shape[0], dtype=int, format="csr"))
        expect_J.data[:] = 1             # sum, not union: the overlap counted twice
        assert J.nnz == expect_J.nnz
        assert ((J != 0).astype(int) - expect_J > 0).nnz == 0

        expect_M = _masked_pattern(p, "mass")
        assert M.nnz == expect_M.nnz
        assert ((M != 0).astype(int) - (expect_M > 0).astype(int) > 0).nnz == 0
        assert M.nnz < J.nnz / 2, "the mass matrix must not have been given the Jacobian's pattern"


def test_diagonal_requirement_comes_from_the_linear_solver():
    """Needing a stored diagonal is a property of the FACTORISATION, not of the problem, so the answer
    comes from the active solver via requires_explicit_diagonal() -- MUMPS and Pardiso do not need one,
    PETSc's own LU rejects a matrix without one.

    Note PETSc's options database is global and sticky within a process, so a solver configured earlier
    can change what a later one reports; this test therefore only exercises the solvers whose answer is
    unambiguous here, and checks the override machinery rather than the PETSc option parsing."""
    with _CavityProblem(N=6) as p:
        p.quiet()
        p.set_linear_solver("pardiso")
        p.initialise()
        assert p.get_la_solver().requires_explicit_diagonal() is False
        p.solve()
        assert p._force_jacobian_diagonal_entries_is_auto, "should still be taking the solver's answer"
        assert p.force_jacobian_diagonal_entries is False
        assert _rows_without_diagonal(p.assemble_jacobian(with_residual=False)) > 0

        # An explicit setting overrides the solver in either direction...
        p.force_jacobian_diagonal_entries = True
        assert not p._force_jacobian_diagonal_entries_is_auto
        assert _rows_without_diagonal(p.assemble_jacobian(with_residual=False)) == 0
        # ...and can be handed back.
        p._set_force_jacobian_diagonal_entries_auto()
        assert p._force_jacobian_diagonal_entries_is_auto
        assert _rows_without_diagonal(p.assemble_jacobian(with_residual=False)) > 0


def test_a_solver_asking_for_the_diagonal_gets_it():
    """The other half: a solver that answers True must actually receive the diagonal, and the pattern
    must be invalidated when the answer changes -- otherwise a solver that starts needing a diagonal
    would keep being handed a pattern assembled without one."""
    class _Fussy:
        def requires_explicit_diagonal(self):
            return True

    with _CavityProblem(N=6) as p:
        p.quiet()
        p.initialise()
        p.solve()
        assert _rows_without_diagonal(p.assemble_jacobian(with_residual=False)) > 0
        before = p.jacobian_structure_id
        p._set_solver_requires_explicit_diagonal(True)
        assert p.force_jacobian_diagonal_entries is True, "auto mode must follow the solver"
        assert p.jacobian_structure_id != before, "changing the answer changes the pattern"
        assert _rows_without_diagonal(p.assemble_jacobian(with_residual=False)) == 0


def test_forced_diagonal_adds_exactly_the_missing_diagonals():
    with _CavityProblem(N=6) as p:
        p.quiet()
        p.initialise()
        p.keep_structural_zeros = True
        p.solve()
        without = p.assemble_jacobian(with_residual=False)
        missing = _rows_without_diagonal(without)
        # Taylor-Hood has no pressure-pressure coupling, so the pruned pattern has no pressure diagonal.
        assert missing > 0
        p.force_jacobian_diagonal_entries = True
        with_diag = p.assemble_jacobian(with_residual=False)
        assert _rows_without_diagonal(with_diag) == 0
        assert with_diag.nnz == without.nnz + missing, "it should add the missing diagonals and nothing else"

def test_field_coupling_mask_is_tighter_than_connectivity():
    """...and is worth having: on a weakly coupled problem the pure connectivity pattern over-allocates
    by ~37%, which the field-pair mask removes entirely."""
    with _CavityProblem(N=6, with_scalar=True) as p:
        p.quiet()
        p.initialise()
        p.keep_structural_zeros = False   # compare the masks against the purely numerical pattern
        p.solve()
        connectivity = _connectivity_pattern(p)
        masked = _masked_pattern(p, "jacobian")
        assert masked.nnz < connectivity.nnz
        _M, J = _eigen_matrices(p)
        assert masked.nnz == J.nnz, "the field-pair mask should reproduce the numerical pattern exactly here"
        # The mass matrix mask must be tighter still, and a subset of the Jacobian's.
        mass_masked = _masked_pattern(p, "mass")
        assert mass_masked.nnz < masked.nnz / 2
        assert ((mass_masked > 0).astype(int) - (masked > 0).astype(int) > 0).nnz == 0


# ---------------------------------------------------------------------------------------------------
# Phase 2 -- assembling straight into the frozen pattern
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("with_scalar", [False, True], ids=["navier_stokes", "coupled"])
def test_frozen_sparsity_matches_the_container_assembly(with_scalar):
    """The fast path skips the accumulate-and-compress machinery entirely, so it has to be checked
    against it: same pattern, same values, for both matrices of the eigenproblem assembly."""
    ref = None
    for frozen in (False, True):
        with _CavityProblem(N=8, with_scalar=with_scalar) as p:
            p.quiet()
            p.initialise()
            p.use_frozen_sparsity = frozen
            p.solve()
            J = p.assemble_jacobian(with_residual=False)
            M, J_eig = _eigen_matrices(p)
            dofs = numpy.asarray(p.get_history_dofs(0)).copy()
            if ref is None:
                ref = (J.copy(), M.copy(), J_eig.copy(), dofs)
                continue
            J0, M0, E0, d0 = ref
            assert numpy.allclose(dofs, d0, rtol=0, atol=1e-12)
            for got, want, what in ((J, J0, "jacobian"), (M, M0, "mass matrix"), (J_eig, E0, "eigen jacobian")):
                assert got.nnz == want.nnz, "%s: %d vs %d nonzeros" % (what, got.nnz, want.nnz)
                d = (got - want).tocsr()
                d.eliminate_zeros()
                assert d.nnz == 0, "%s differs between the frozen and the container assembly" % what


def test_frozen_sparsity_emits_canonically_sorted_csr():
    """A side benefit worth keeping: oomph's container assembly emits each row's column indices in
    insertion order, whereas the frozen path emits them ascending. Consumers that need sorted indices
    (MKL Pardiso, among others) then get them for free."""
    with _CavityProblem(N=6) as p:
        p.quiet()
        p.initialise()
        p.solve()
        assert p.assemble_jacobian(with_residual=False).has_sorted_indices
        p.use_frozen_sparsity = False
        assert not p.assemble_jacobian(with_residual=False).has_sorted_indices


def test_frozen_sparsity_cache_survives_alternating_assemblies():
    """The pattern cache must hold several patterns at once, and the pattern id must be a FUNCTION of
    the configuration rather than a counter.

    A workflow that alternates between the Newton assembly and the eigenproblem assembly switches
    assembly handler on every step, which is a different pattern -- correctly so. If the id merely
    counted changes, returning to a configuration would produce a NEW id, every lookup would miss, and
    each assembly would re-derive its pattern: two passes over the mesh plus a sort, i.e. slower than
    having no cache at all. The same applies to a PETSc preconditioner matrix assembled from another
    residual (PETSCSolver.assemble_matrix), which is what prompted this.

    Three patterns are involved here -- Newton's Jacobian, and the eigenproblem's Jacobian and mass
    matrix -- so three builds is the floor no matter how many times we go round."""
    with _CavityProblem(N=6) as p:
        p.quiet()
        p.initialise()
        p.solve()
        for _ in range(4):
            p.assemble_jacobian(with_residual=False)
            p.assemble_eigenproblem_matrices(0.0)
        assert p._get_frozen_sparsity_rebuild_count() == 3, \
            "expected one build per distinct pattern; more means the cache is thrashing"


def test_structure_id_is_stable_when_returning_to_a_configuration():
    """The property the cache rests on, checked directly: leaving a configuration and coming back must
    give the same id, or nothing downstream (the pattern cache, Pardiso's symbolic factorisation, a
    PETSc Mat) can be reused across the round trip."""
    with _CavityProblem(N=6) as p:
        p.quiet()
        p.initialise()
        p.solve()
        first = p.jacobian_structure_id
        assert first != 0
        p.assemble_eigenproblem_matrices(0.0)      # installs and removes another assembly handler
        assert p.jacobian_structure_id == first
        p.keep_structural_zeros = False            # a real change: no usable pattern
        assert p.jacobian_structure_id == 0
        p.keep_structural_zeros = True
        # Ids are never recycled after an invalidation, so this must differ from the pre-invalidation one.
        assert p.jacobian_structure_id not in (0, first)


def test_multiassembly_matches_with_and_without_the_frozen_pattern():
    """The multi-quantity assembly behind the Python-level bifurcation trackers goes through
    sparse_assemble_row_or_column_compressed_base_problem, which accumulates into a std::map per row --
    the slowest container in the codebase. Every matrix it builds is a derivative of the same residual
    w.r.t. the dofs, so they all share the Jacobian's pattern and one frozen pattern serves all of them.

    This checks the fast path against the map-based one: same pattern, same values, same converged fold.
    """
    results = []
    for frozen in (False, True):
        with _BratuProblem() as p:
            p.quiet()
            p.setup_for_stability_analysis(analytic_hessian=True)
            p.initialise()
            p.use_frozen_sparsity = frozen
            lam = p.get_global_parameter("lam")
            lam.value = 3.0
            p.solve()
            p.solve_eigenproblem(1)
            from pyoomph.generic.bifurcation_tools import FoldTracker
            tracker = FoldTracker(p, "lam", eigenvector=0)
            p.set_custom_assembler(tracker)
            p.solve()
            # Contract the Hessian with a FIXED vector, not the eigenvector guess. An eigenvector is
            # only defined up to sign and the solver picks either, so the two runs would be handed
            # opposite vectors -- and the Hessian-vector product is linear in it, so dJdU would come
            # back exactly negated and the comparison would report a bug that is not there.
            fixed = numpy.cos(numpy.arange(p.ndof(), dtype=float))
            R, J, dRdp, dJdp, HV = (tracker.start_multiassembly()
                                    .R().J().dRdp("lam").dJdp("lam").dJdU(fixed).assemble())
            results.append((float(lam.value), R.copy(), J.copy(), dRdp.copy(), dJdp.copy(), HV.copy()))

    slow, fast = results
    assert fast[0] == pytest.approx(slow[0], rel=0, abs=1e-9), "the tracked fold moved"
    assert numpy.allclose(fast[1], slow[1], rtol=1e-12, atol=1e-12)
    for a, b, what in ((fast[2], slow[2], "J"), (fast[3], slow[3], "dRdp"),
                       (fast[4], slow[4], "dJdp"), (fast[5], slow[5], "dJdU")):
        if hasattr(a, "nnz"):
            assert a.nnz == b.nnz, "%s: %d vs %d nonzeros" % (what, a.nnz, b.nnz)
            d = abs(a - b)
            scale = max(abs(b).max(), 1e-30)
            assert d.max() <= 1e-12 * scale, \
                "%s differs between the frozen and the map-based multi-assembly by %.3e (scale %.3e)" % (
                    what, d.max(), scale)
        else:
            assert numpy.allclose(a, b, rtol=0, atol=1e-12), what


def test_frozen_sparsity_falls_back_where_it_cannot_apply():
    """It must decline, not misbehave, when the pattern route does not fit -- here an augmented dof
    vector from bifurcation tracking, whose elemental blocks are indexed by the handler's own numbering
    and are not described by the element's field map."""
    with _BratuProblem() as p:
        p.quiet()
        p.initialise()
        lam = p.get_global_parameter("lam")
        lam.value = 3.0
        p.solve()
        ndof_plain = p.ndof()
        p.activate_bifurcation_tracking("lam", "fold")
        p.solve()   # would throw or corrupt if the frozen path tried to handle the augmented system
        assert p.ndof() > ndof_plain
        assert numpy.isfinite(numpy.asarray(p.get_history_dofs(0))).all()
        # Converged onto the turning point, which is what the augmented system is for. (Do not solve the
        # plain problem again here: at a fold the unaugmented Jacobian is singular by definition.)
        assert float(lam.value) == pytest.approx(6.808, abs=0.05)
        p.deactivate_bifurcation_tracking()
        assert p.ndof() == ndof_plain


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
        assert p.keep_structural_zeros is True, "structural zeros are expected to be on by default"
        assert p.jacobian_structure_id != 0
        p.keep_structural_zeros = False
        assert p.jacobian_structure_id == 0, "a value-dependent pattern must not be advertised"
        p.keep_structural_zeros = True
        assert p.jacobian_structure_id != 0


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


def test_structure_id_changes_on_remeshing():
    """Remeshing replaces the mesh wholesale, so nothing about the old pattern survives. It funnels
    through assign_eqn_numbers() like adaptation does, but "should be covered by construction" is not
    evidence, and this is the one invalidation path with no other test.

    Also checks the pattern is still VALID after the remesh, not merely different: a stale
    element -> dof attribution surviving a mesh rebuild would produce a mask that no longer covers the
    numerical pattern, which is the silent-truncation failure mode.

    NOTE: this test used to make the suite print "nanobind: leaked N instances!" at interpreter exit,
    which was never caused by the test or by the structural-sparsity work: a superseded mesh kept its
    _templatemesh reference, closing an uncollectable Problem -> mesh -> template -> remesher -> Problem
    cycle, so any remeshing script leaked its whole Problem. Fixed in _destroy_superseded_mesh() (see
    generic/problem.py); if leak reports ever reappear here, look there first."""
    gmsh = pytest.importorskip("gmsh", reason="remeshing needs gmsh")  # noqa: F841
    from pyoomph.equations.ALE import LaplaceSmoothedMesh
    from pyoomph.meshes.remesher import Remesher2d
    from pyoomph.equations.generic import RemeshWhen, RemeshingOptions, ElementSpace

    class _RemeshProblem(Problem):
        def __init__(self):
            super().__init__()
            self.remeshing = True
            self.remesh_options = RemeshingOptions(max_expansion=1.3, min_expansion=0.7,
                                                   min_quality_decrease=0.2)
            self.remesh_count = 0

        def actions_after_remeshing(self):
            super().actions_after_remeshing()
            self.remesh_count += 1

        def define_problem(self):
            mesh = RectangularQuadMesh(N=6)
            mesh.remesher = Remesher2d(mesh)
            self.add_mesh(mesh)
            eqs = LaplaceSmoothedMesh()
            eqs += ElementSpace("C2")  # the mesh motion alone does not pin down the coordinate space
            eqs += DirichletBC(mesh_x=0, mesh_y=True) @ "left"
            eqs += DirichletBC(mesh_x=True, mesh_y=0) @ "bottom"
            eqs += DirichletBC(mesh_y=1) @ "top"
            xi = var("lagrangian")
            # Stretch the right edge until the elements are distorted enough to trigger a remesh.
            eqs += DirichletBC(mesh_x=1 + 1.5 * xi[1] * var("time")) @ "right"
            eqs += RemeshWhen(self.remesh_options)
            self.add_equations(eqs @ "domain")

    with _RemeshProblem() as p:
        p.quiet()
        p.initialise()
        p.keep_structural_zeros = True
        p.solve()
        sid_before = p.jacobian_structure_id
        assert sid_before != 0

        for _ in range(8):
            p.run(0.5, startstep=0.5, maxstep=0.5, outstep=False, temporal_error=None)
            if p.remesh_count:
                break
        assert p.remesh_count > 0, "no remesh was triggered -- the test would prove nothing"

        assert p.jacobian_structure_id != sid_before, "remeshing must invalidate the sparsity pattern"
        # ...and the rebuilt pattern must still be a superset of what is numerically nonzero.
        J = p.assemble_jacobian(with_residual=False)
        masked = _masked_pattern(p, "jacobian")
        assert ((J != 0).astype(int) - (masked > 0).astype(int) > 0).nnz == 0
        assert _rows_without_diagonal(J) == 0


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
