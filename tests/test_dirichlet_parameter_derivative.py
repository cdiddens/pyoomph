#  @file
#  @author Christian Diddens
#  @author Duarte Rocha
#  @author Maxim de Wildt
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

# dR/dp for a global parameter that sits in a DirichletBC, and the Dirichlet columns of the Jacobian.
#
# A DirichletBC is emitted into ElementalDirichletConditions and never into the residual, so the
# residual's own dR/dp cannot see a parameter in it - for a parameter that appears in nothing else the
# derivative used to be an exact zero, silently (dev_docs/shape_optimization.md 3.1). The missing part is
# the chain rule through the prescribed values U_d = g(p):
#
#     dR_f/dp = dR_f/dp|_U + J_fd dg/dp ,     and, with the dofs kept,  dR_d/dp = -dg/dp .
#
# With apply_Dirichlet_BCs_by_dof_removing=False that is now exact, for both settings of
# keep_Dirichlet_columns. With the default dof removal J_fd is never assembled; that case warns instead.
#
# The reference throughout is the tangent the derivative exists to produce: dU/dp = -J^{-1} dR/dp must
# equal the central difference of two converged solutions. The problems are linear in the solution, so
# the only difference error is that of g itself, O(h^2) on a smooth g.
#
# See dev_docs/dirichlet_parameter_derivative.md.

import numpy
import pytest
from scipy.sparse.linalg import spsolve

from pyoomph import *
from pyoomph.expressions import sin, cos, pi
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.equations.ALE import LaplaceSmoothedMesh
from pyoomph.meshes.simplemeshes import RectangularQuadMesh


FD_STEP = 1e-5
P0 = 1.3


class _BoundaryValuePoisson(Problem):
    """-lap(u) = s on the unit square, u = g(p) on the bottom and 0 elsewhere.

    in_source: the parameter appears in the source as well, so the residual's own dR/dp is not zero and
    the Dirichlet part has to be ADDED to it rather than replace it."""

    def __init__(self, in_source=False, N=4):
        super().__init__()
        self.in_source = in_source
        self.N = N
        self.p = self.get_global_parameter("p")
        self.p.value = P0

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=self.N))
        x = var("coordinate_x")
        eqs = PoissonEquation(source=(self.p if self.in_source else 1), space="C2")
        eqs += DirichletBC(u=sin(self.p * x) * (1 + x)) @ "bottom"
        for b in ["left", "right", "top"]:
            eqs += DirichletBC(u=0) @ b
        self.add_equations(eqs @ "domain")


class _ShapePoisson(Problem):
    """The shape-optimisation case: Poisson on a Laplace-smoothed mesh whose bottom boundary is
    prescribed by a strong position DirichletBC that contains the parameter."""

    def __init__(self, N=4):
        super().__init__()
        self.N = N
        self.p = self.get_global_parameter("p")
        self.p.value = 0.05

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=self.N))
        X = var("lagrangian_x")
        eqs = PoissonEquation(source=1, space="C2")
        eqs += LaplaceSmoothedMesh()
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        eqs += DirichletBC(mesh=var("lagrangian")) @ "top"
        for b in ["left", "right", "bottom"]:
            eqs += DirichletBC(mesh_x=var("lagrangian_x")) @ b
        eqs += DirichletBC(mesh_y=var("lagrangian_y") + self.p * sin(pi * X)) @ "bottom"
        self.add_equations(eqs @ "domain")


class _PrescribedSource(Equations):
    """A discontinuous field c, entirely prescribed by a DirichletBC on the whole domain. Its own
    equation (c = 0) is there only so that the field has a Jacobian row; the condition overrides it."""

    def __init__(self, space):
        super().__init__()
        self.space = space

    def define_fields(self):
        self.define_scalar_field("c", self.space)

    def define_residuals(self):
        c, ctest = var_and_test("c")
        self.add_weak(c, ctest)


class _ElementalFieldPoisson(Problem):
    """-lap(u) = c with c on a DL or D0 space and c = g(p, x, y) prescribed everywhere.

    The DL case is the one with its own code path: the condition is fitted as a midpoint value plus one
    slope per local direction, so the derivative has to be fitted the same way."""

    def __init__(self, space, N=4):
        super().__init__()
        self.space = space
        self.N = N
        self.p = self.get_global_parameter("p")
        self.p.value = P0

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=self.N))
        x, y = var("coordinate_x"), var("coordinate_y")
        eqs = PoissonEquation(source=var("c"), space="C2")
        eqs += _PrescribedSource(self.space)
        eqs += DirichletBC(c=sin(self.p * x) * (1 + y))  # on the domain itself, i.e. every element
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        self.add_equations(eqs @ "domain")


def _build(problem, removing, outdir, keep_columns=False):
    problem.apply_Dirichlet_BCs_by_dof_removing = removing
    problem.keep_Dirichlet_columns = keep_columns
    problem.set_output_directory(str(outdir))
    problem.quiet()
    problem.set_linear_solver("superlu")
    # The shape problem is nonlinear in the mesh, so the difference quotient carries the Newton
    # tolerance divided by the step: 1e-8 / 2e-5 showed up as a 2e-5 relative "error" in the tangent.
    problem.newton_solver_tolerance = 1e-12
    problem.initialise()
    return problem


def _dRdp(problem):
    problem._replace_RJM_by_param_deriv("p", True)
    try:
        return numpy.array(problem.get_residuals()).copy()
    finally:
        problem._replace_RJM_by_param_deriv("p", False)


def _solve_at(problem, p):
    problem.p.value = p
    problem.solve()
    return numpy.array(problem.get_current_dofs()[0]).copy()


def _tangent_and_fd(problem):
    """(-J^{-1} dR/dp, central difference of the converged dofs, dR/dp, J) at P = problem.p.value."""
    p0 = problem.p.value
    _solve_at(problem, p0)
    _, J = problem.assemble_jacobian(with_residual=True)
    d = _dRdp(problem)
    tangent = spsolve(J.tocsc(), -d)
    Up = _solve_at(problem, p0 + FD_STEP)
    Um = _solve_at(problem, p0 - FD_STEP)
    _solve_at(problem, p0)
    return tangent, (Up - Um) / (2 * FD_STEP), d, J


def _identity_rows(J):
    """Rows whose only nonzero VALUE is a 1 on the diagonal, i.e. the manipulated Dirichlet rows.
    By value, not by stored entry: the pattern keeps structural zeros (keep_structural_zeros)."""
    J = J.tocsr()
    rows = []
    for i in range(J.shape[0]):
        s, e = J.indptr[i], J.indptr[i + 1]
        nz = J.data[s:e] != 0.0
        cols = J.indices[s:e][nz]
        if len(cols) == 1 and cols[0] == i and J.data[s:e][nz][0] == 1.0:
            rows.append(i)
    return rows


def _assert_tangent_matches(tangent, fd, rtol=1e-7):
    scale = numpy.max(numpy.abs(fd))
    assert scale > 1e-3, "the parameter does not move the solution at all; the test proves nothing"
    err = numpy.max(numpy.abs(tangent - fd)) / scale
    assert err < rtol, "dU/dp from dR/dp is off by %.3e relative" % err


# ---------------------------------------------------------------------------------------------------
# Exact with dof removal off
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("keep_columns", [False, True])
@pytest.mark.parametrize("in_source", [False, True])
def test_boundary_value_tangent_is_exact(tmp_path, keep_columns, in_source):
    """A parameter in a field DirichletBC, alone or also in the residual, either column setting."""
    p = _build(_BoundaryValuePoisson(in_source=in_source), False, tmp_path, keep_columns)
    tangent, fd, _, _ = _tangent_and_fd(p)
    _assert_tangent_matches(tangent, fd)


@pytest.mark.parametrize("keep_columns", [False, True])
def test_position_dirichlet_tangent_is_exact(tmp_path, keep_columns):
    """The shape case: a strong DirichletBC on mesh_y carries the parameter, so the whole sensitivity
    of the field reaches it through the mesh."""
    p = _build(_ShapePoisson(), False, tmp_path, keep_columns)
    tangent, fd, _, _ = _tangent_and_fd(p)
    _assert_tangent_matches(tangent, fd)


@pytest.mark.parametrize("space", ["DL", "D0"])
def test_elemental_field_tangent_is_exact(tmp_path, space):
    """Elemental fields: D0 takes the generic per-dof path, DL its own slope fit."""
    p = _build(_ElementalFieldPoisson(space), False, tmp_path)
    tangent, fd, _, _ = _tangent_and_fd(p)
    _assert_tangent_matches(tangent, fd)


def test_dirichlet_rows_carry_minus_dg_dp(tmp_path):
    """In the kept-dof system the Dirichlet rows read U_d - g(p), so dR_d/dp is -dg/dp, i.e. the
    tangent moves the prescribed values by exactly dg/dp. Checked against the closed form."""
    p = _build(_BoundaryValuePoisson(), False, tmp_path)
    p.solve()
    d = _dRdp(p)
    eqs = p.get_dof_description()
    bottom = []
    for n in p.get_mesh("domain").boundary_nodes("bottom"):
        if n.x(0) < 1e-12 or n.x(0) > 1 - 1e-12:
            continue  # the corners are shared with u=0 conditions applied later
        bottom.append((n.x(0), n.eqn_number(0)))
    assert bottom
    for x, eq in bottom:
        assert eq >= 0
        expected = -x * numpy.cos(P0 * x) * (1 + x)
        assert abs(d[eq] - expected) < 1e-12, (x, d[eq], expected)


# ---------------------------------------------------------------------------------------------------
# keep_Dirichlet_columns
# ---------------------------------------------------------------------------------------------------

def test_keeping_the_columns_changes_the_matrix_not_the_solution(tmp_path):
    zeroed = _build(_ShapePoisson(), False, tmp_path / "zeroed", keep_columns=False)
    kept = _build(_ShapePoisson(), False, tmp_path / "kept", keep_columns=True)
    Uz = _solve_at(zeroed, 0.05)
    Uk = _solve_at(kept, 0.05)
    numpy.testing.assert_allclose(Uk, Uz, rtol=1e-11, atol=1e-13)

    _, Jz = zeroed.assemble_jacobian(with_residual=True)
    _, Jk = kept.assemble_jacobian(with_residual=True)
    Jz, Jk = Jz.tocsc(), Jk.tocsc()
    dirichlet = _identity_rows(Jz)
    assert len(dirichlet) > 0
    # Dirichlet rows: the identity in both
    assert _identity_rows(Jk) == dirichlet
    # Dirichlet columns: only the diagonal survives the default, the coupling survives with kept columns
    offdiag = lambda J: abs(J[:, dirichlet]).sum() - len(dirichlet)
    assert offdiag(Jz) == 0.0
    assert offdiag(Jk) > 1e-3
    assert not kept._get_proven_matrix_symmetry("")[0]


def test_free_node_gradient_is_the_adjoint_at_the_dirichlet_dofs(tmp_path):
    """With kept columns, J^T lambda = e_k gives dU_k/d(prescribed value) at the Dirichlet entries of
    lambda - the free-node control gradient. Contracted with dg/dp it must reproduce dU_k/dp, which is
    an independent quantity here because the parameter appears in nothing but the condition."""
    p = _build(_BoundaryValuePoisson(), False, tmp_path, keep_columns=True)
    tangent, fd, d, J = _tangent_and_fd(p)
    free = numpy.setdiff1d(numpy.arange(J.shape[0]), _identity_rows(J))
    k = int(free[numpy.argmax(numpy.abs(fd[free]))])  # a free dof: a prescribed one is its own gradient
    e = numpy.zeros(J.shape[0])
    e[k] = 1.0
    lam = spsolve(J.transpose().tocsc(), e)
    # d is [0 ; -dg/dp]: zero in the free rows since p is only in the condition
    dirichlet = numpy.flatnonzero(d)
    assert set(dirichlet) <= set(_identity_rows(J))
    node_gradient = lam[dirichlet]
    dg = -d[dirichlet]
    assert abs(node_gradient @ dg - fd[k]) < 1e-7 * abs(fd[k])

    # Without the columns the same adjoint knows nothing about the prescribed values.
    q = _build(_BoundaryValuePoisson(), False, tmp_path / "zeroed", keep_columns=False)
    _solve_at(q, P0)
    _, Jz = q.assemble_jacobian(with_residual=True)
    lam_z = spsolve(Jz.transpose().tocsc(), e)
    assert numpy.max(numpy.abs(lam_z[dirichlet])) == 0.0


# ---------------------------------------------------------------------------------------------------
# The default dof removal: lost, and said so
# ---------------------------------------------------------------------------------------------------

def test_dof_removal_warns_once_and_says_why(tmp_path, capfd):
    p = _build(_BoundaryValuePoisson(), True, tmp_path)
    p.solve()
    capfd.readouterr()
    d = _dRdp(p)
    out = capfd.readouterr().out
    assert numpy.max(numpy.abs(d)) == 0.0, "nothing in the residual depends on p, so this stays zero"
    assert "global parameter 'p' appears in a DirichletBC" in out
    assert "apply_Dirichlet_BCs_by_dof_removing=False" in out
    _dRdp(p)
    assert "appears in a DirichletBC" not in capfd.readouterr().out, "the warning must be shown once"


def test_no_warning_without_a_dependence(tmp_path, capfd):
    """A parameter in the residual only must not trigger it."""
    class _SourceOnly(_BoundaryValuePoisson):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=self.N))
            eqs = PoissonEquation(source=self.p, space="C2")
            for b in ["left", "right", "top", "bottom"]:
                eqs += DirichletBC(u=0) @ b
            self.add_equations(eqs @ "domain")

    p = _build(_SourceOnly(), True, tmp_path)
    p.solve()
    capfd.readouterr()
    d = _dRdp(p)
    assert numpy.max(numpy.abs(d)) > 0.0
    assert "appears in a DirichletBC" not in capfd.readouterr().out
