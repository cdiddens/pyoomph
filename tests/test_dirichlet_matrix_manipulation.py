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

# Correctness gates for problem.apply_Dirichlet_BCs_by_dof_removing = False, i.e. for enforcing
# Dirichlet conditions by MATRIX MANIPULATION rather than by dropping the dof from the system.
#
# The two strategies:
#
#   True (the default)  -- the constrained value is pinned, Data::assign_eqn_numbers skips it, and it
#                          never enters Dof_pt at all. Smaller matrix, but the dof vector has holes in
#                          it wherever a boundary condition sits.
#   False               -- the dof is unpinned again and registered in Problem::dirichlet_info, so it
#                          IS numbered and IS assembled; after assembly
#                          Problem::remove_dirichlets_by_matrix_manipulation replaces its row by an
#                          identity row, zeroes its column elsewhere and zeroes its residual entry.
#                          Bigger matrix, but a dof layout with no holes -- which is what a block
#                          preconditioner (BoomerAMG's strided blocks) needs, and why this exists.
#
# The strategy is an implementation detail of how the constraint is imposed, so the ONLY acceptable
# outcome is that it changes nothing about the answer. That is what is protected here, together with the
# two structural properties the AMG use case actually depends on:
#
#   * the Dirichlet row is exactly (d, 0, ..., 0) with the residual entry zero, so the increment of a
#     constrained dof is exactly zero for ANY non-zero d;
#   * the diagonal entry d is actually PRESENT. It is written as
#     `values[i] = (col == row) ? d : 0.0` while scanning the stored entries of the row, so if the
#     diagonal is not in the sparsity pattern nothing writes it and the row comes out identically zero
#     -- a singular matrix, not a slow one. Whether it is in the pattern was, until this file existed, a
#     property of the LINEAR SOLVER (Problem::diagonal_entries_are_forced() defers to
#     solver_requires_explicit_diagonal, and MUMPS and SuperLU do not ask for one), so the failure
#     needed a field whose own equation carries no self-coupling. A pressure with a DirichletBC on a
#     Taylor-Hood discretisation is exactly that: the continuity equation contains no pressure.
#     test_pressure_dirichlet_row_is_not_structurally_empty is that case.
#
# Before this file there was no test anywhere for apply_Dirichlet_BCs_by_dof_removing=False.
#
# Note on the linear solver: scipy's SuperLU is set explicitly, because the point of several tests is
# what the matrix looks like when the solver does NOT ask for an explicit diagonal, and because a
# solver whose own error sits above the Newton tolerance would blur the equivalence comparisons.

import numpy
import pytest

from pyoomph import *
from pyoomph.equations.navier_stokes import StokesEquations
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.meshes.simplemeshes import RectangularQuadMesh


# ---------------------------------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------------------------------

class _DirichletPoisson(Problem):
    """Poisson with a Dirichlet condition on all four sides. Every constrained dof's own equation has a
    healthy self-coupling (the Laplacian's diagonal), so this exercises the strategy without touching
    the structural-diagonal question."""

    def __init__(self, N=4):
        super().__init__()
        self.N = N

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=self.N))
        eqs = PoissonEquation(source=1, space="C2")
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        self.add_equations(eqs @ "domain")


class _DirichletPressureStokes(Problem):
    """Lid-driven Stokes on Taylor-Hood, with the pressure level fixed by a DirichletBC rather than by
    a Lagrange multiplier.

    The continuity equation contains no pressure, so the constrained pressure dof's own diagonal is
    structurally absent unless something forces it. That is what makes this the zero-row case."""

    def __init__(self, N=3):
        super().__init__()
        self.N = N

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=self.N))
        eqs = StokesEquations(dynamic_viscosity=1, mode="TH")
        eqs += InitialCondition(velocity_x=0, velocity_y=0)
        for b in ["left", "right", "bottom"]:
            eqs += DirichletBC(velocity_x=0, velocity_y=0) @ b
        eqs += DirichletBC(velocity_x=1, velocity_y=0) @ "top"
        eqs += DirichletBC(pressure=0) @ "bottom/left"
        self.add_equations(eqs @ "domain")


# ---------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------

def _build(cls, removing, outdir, force_diagonal=None, **kwargs):
    p = cls(**kwargs)
    p.apply_Dirichlet_BCs_by_dof_removing = removing
    p.set_output_directory(str(outdir))
    p.quiet()
    p.set_linear_solver("superlu")
    if force_diagonal is not None:
        p.force_jacobian_diagonal_entries = force_diagonal
    p.initialise()
    return p


def _classify_rows(J):
    """Split the rows of a CSR Jacobian into (all-zero, single-entry-on-the-diagonal, ordinary).

    "Single entry on the diagonal" is what remove_dirichlets_by_matrix_manipulation leaves behind; its
    VALUE is deliberately not part of the classification, so that the same helper serves both the
    current unit diagonal and a magnitude-matched one."""
    J = J.tocsr()
    zero, constrained, ordinary = [], [], []
    for i in range(J.shape[0]):
        s, e = J.indptr[i], J.indptr[i + 1]
        cols, vals = J.indices[s:e], J.data[s:e]
        nz = numpy.abs(vals) > 0.0
        if not nz.any():
            zero.append(i)
        elif nz.sum() == 1 and cols[nz][0] == i:
            constrained.append(i)
        else:
            ordinary.append(i)
    return zero, constrained, ordinary


def _dof_names(p):
    types, names = p.get_dof_description()
    return [names[t] for t in types]


# ---------------------------------------------------------------------------------------------------
# The strategy must not change the answer
# ---------------------------------------------------------------------------------------------------

def test_poisson_solution_is_independent_of_the_strategy(tmp_path):
    """The same Poisson solution, whether the constrained dofs are removed or kept and manipulated.

    Compared at the NODES rather than in the dof vector, because the two strategies do not have the
    same dof vector at all -- that is the whole difference between them."""
    pT = _build(_DirichletPoisson, True, tmp_path / "rm")
    pT.solve()
    uT = numpy.array([n.value(0) for n in pT.get_mesh("domain").nodes()])

    pF = _build(_DirichletPoisson, False, tmp_path / "mm")
    pF.solve()
    uF = numpy.array([n.value(0) for n in pF.get_mesh("domain").nodes()])

    assert pF.ndof() > pT.ndof(), \
        "the kept Dirichlet dofs should make the system BIGGER (%d vs %d)" % (pF.ndof(), pT.ndof())
    numpy.testing.assert_allclose(uF, uT, rtol=1e-10, atol=1e-12)


def test_stokes_solution_is_independent_of_the_strategy(tmp_path):
    """The same, on the saddle-point problem, and with a constrained PRESSURE among the conditions."""
    pT = _build(_DirichletPressureStokes, True, tmp_path / "rm")
    pT.solve()
    mT = pT.get_mesh("domain")
    vT = numpy.array([[n.value(0), n.value(1)] for n in mT.nodes()])

    pF = _build(_DirichletPressureStokes, False, tmp_path / "mm")
    pF.solve()
    mF = pF.get_mesh("domain")
    vF = numpy.array([[n.value(0), n.value(1)] for n in mF.nodes()])

    assert pF.ndof() > pT.ndof()
    numpy.testing.assert_allclose(vF, vT, rtol=1e-9, atol=1e-11)


# ---------------------------------------------------------------------------------------------------
# What the manipulated rows look like
# ---------------------------------------------------------------------------------------------------

def test_dirichlet_rows_are_a_lone_diagonal_with_a_zero_residual(tmp_path):
    """The structural contract the increment-is-exactly-zero argument rests on.

    A constrained row must hold exactly one non-zero, on its own diagonal, and its residual entry must
    be zero. Then the Newton increment of that dof is exactly 0 for any non-zero diagonal value -- which
    is what lets the value be chosen for conditioning rather than being forced to 1."""
    p = _build(_DirichletPoisson, False, tmp_path)
    r, J = p.assemble_jacobian()
    zero, constrained, ordinary = _classify_rows(J)
    names = _dof_names(p)

    assert not zero, "all-zero rows: %s" % [names[i] for i in zero]
    # Every boundary dof, and only boundary dofs, is constrained here.
    assert {names[i] for i in constrained} == {"domain/left/u", "domain/right/u",
                                              "domain/top/u", "domain/bottom/u"}
    assert len(constrained) + len(ordinary) == J.shape[0]
    numpy.testing.assert_array_equal(numpy.asarray(r)[constrained], 0.0)


def test_dirichlet_columns_are_zeroed_in_the_other_rows(tmp_path):
    """The other half of the manipulation: a constrained dof does not change, so its derivative must not
    appear in anybody else's row either. Without this the identity row alone would still give the right
    increment, but the retained block would be the wrong matrix."""
    p = _build(_DirichletPoisson, False, tmp_path)
    _r, J = p.assemble_jacobian()
    J = J.tocsr()
    _zero, constrained, ordinary = _classify_rows(J)
    cset = set(constrained)
    for i in ordinary:
        s, e = J.indptr[i], J.indptr[i + 1]
        for col, val in zip(J.indices[s:e], J.data[s:e]):
            if col in cset:
                assert val == 0.0, "row %d keeps a non-zero in constrained column %d" % (i, col)


# ---------------------------------------------------------------------------------------------------
# The zero-row case
# ---------------------------------------------------------------------------------------------------

def test_pressure_dirichlet_row_is_not_structurally_empty(tmp_path):
    """A DirichletBC on a field whose own equation has no self-coupling.

    The continuity equation contains no pressure, so nothing puts the constrained pressure dof's
    diagonal into the sparsity pattern by itself, and the linear solver in use (SuperLU) does not ask
    for an explicit diagonal either. The manipulation writes the diagonal only where the pattern
    already has a slot for it, so the row would come out identically zero -- an exactly singular
    matrix. Keeping the dof therefore has to force its own diagonal, independently of the solver."""
    p = _build(_DirichletPressureStokes, False, tmp_path)
    # The guard is the SOLVER's answer, not problem.force_jacobian_diagonal_entries: reading that
    # property gives the effective value, which keeping the Dirichlet dofs is now itself a reason for.
    assert not p.get_la_solver().requires_explicit_diagonal(), \
        "this test is only meaningful while the solver does not ask for an explicit diagonal"
    assert p._force_jacobian_diagonal_entries_is_auto, \
        "and only while nothing has overridden the policy explicitly"
    _r, J = p.assemble_jacobian()
    zero, _constrained, _ordinary = _classify_rows(J)
    names = _dof_names(p)
    assert not zero, \
        "structurally empty Dirichlet row(s): %s -- the matrix is exactly singular" % \
        [names[i] for i in zero]


def test_pressure_dirichlet_problem_solves(tmp_path):
    """The end of the same story: with the row present, the Newton solve converges. Before the diagonal
    was forced, this failed with SuperLU's 'Factor is exactly singular'."""
    p = _build(_DirichletPressureStokes, False, tmp_path)
    p.solve()
    assert len(p.get_last_residual_convergence()) <= 3, \
        "a linear problem should converge in one Newton step: %s" % p.get_last_residual_convergence()


def test_forcing_the_diagonal_explicitly_still_works(tmp_path):
    """The explicit override must not be broken by whatever makes the kept-Dirichlet case force it: the
    two reasons to keep a diagonal are independent and either alone is enough."""
    p = _build(_DirichletPressureStokes, False, tmp_path, force_diagonal=True)
    _r, J = p.assemble_jacobian()
    zero, _c, _o = _classify_rows(J)
    assert not zero
