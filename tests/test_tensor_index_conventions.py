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

# The index order of grad, div and the contraction operators.
#
# pyoomph stores grad(u)[i,j] = d(u_i)/d(x_j), the Jacobian, and contracts adjacent indices
# everywhere: dot/contract/matproduct all pair the index of a matrix that faces the other operand,
# and div of a rank-2 tensor contracts the second index, which makes it the adjoint of grad. Two of
# those used to be the other way round -- contract contracted the outer index, and div the first
# index -- and the two reversals cancelled for the terms people actually write, which is why nothing
# noticed. Hence this file: none of it was pinned before.
#
# Note that the symbolic route is not available for anything involving grad: a gradient of a field
# stays a held Diff(..., nondimfield(...)) until the code generator resolves it, so
# (lhs - rhs).is_zero() is False even for expressions that are identical. Everything below that
# touches grad therefore goes through a compiled element, projecting the expression onto a C2 field
# and reading the nodal values back.

import pytest

from pyoomph import Problem, Equations, DirichletBC, InitialCondition
from pyoomph.expressions import (var, var_and_test, weak, matrix, vector, grad, div, dyadic, dot,
                                 contract, double_dot, matproduct, transpose, directional_derivative,
                                 Expression)
from pyoomph.expressions.coordsys import AxisymmetricCoordinateSystem
from pyoomph.equations.ALE import PrescribedMovingMesh
from pyoomph.equations.generic import AverageConstraint
from pyoomph.equations.navier_stokes import NavierStokesEquations
from pyoomph.meshes.simplemeshes import RectangularQuadMesh, CuboidBrickMesh

#: A matrix whose every entry identifies its own (row, column), so that a result of [0,10,20] can only
#: be column 0 and [0,1,2] can only be row 0. Any transposed contraction is immediately visible.
ROWCOL = matrix([[Expression(10 * i + j) for j in range(3)] for i in range(3)])
E_X = vector(1, 0, 0)


def _components(expression, n=3):
    """
    The n leading components of a vector-valued expression, as plain floats.

    Not (a-b).is_zero(): is_zero() asks whether the *expression* is the zero expression, not whether it
    is a zero matrix, so it answers False for a difference that evalm() prints as [[0],[0],[0]] as soon
    as either side is an unevaluated product such as 2*[[1],[2],[3]]. (double_dot_eval in the core uses
    is_zero_matrix() for exactly this reason.)
    """
    resolved = expression.evalm()
    return [float(resolved[i]) for i in range(n)]


# ----------------------------------------------------------------------------------------------
# The contraction operators. Pure expressions, no Problem and no compilation needed.
# ----------------------------------------------------------------------------------------------

def test_matrix_vector_contractions_pair_adjacent_indices():
    """
    dot/contract/@ contract the index of the matrix facing the other operand, as matproduct does.

    A.b is A_ij*b_j, i.e. column 0 of ROWCOL against e_x, and a.B is a_j*B_ji, i.e. row 0. The
    reversed convention that contract used to implement swaps these two.
    """
    column, row = [0.0, 10.0, 20.0], [0.0, 1.0, 2.0]
    assert _components(matproduct(ROWCOL, E_X)) == pytest.approx(column)

    assert _components(dot(ROWCOL, E_X)) == pytest.approx(column)   # A.b -> column
    assert _components(dot(E_X, ROWCOL)) == pytest.approx(row)      # a.B -> row

    # contract and @ must agree with dot in every shape they share; they are one implementation.
    assert _components(contract(ROWCOL, E_X)) == pytest.approx(column)
    assert _components(contract(E_X, ROWCOL)) == pytest.approx(row)
    assert _components(ROWCOL @ E_X) == pytest.approx(column)
    assert _components(E_X @ ROWCOL) == pytest.approx(row)

    # a.B is also transpose(B).a, which is the other way to spell it
    assert _components(matproduct(transpose(ROWCOL), E_X)) == pytest.approx(row)


def test_the_two_mixed_orders_are_genuinely_different():
    """Guards the test above against a symmetric matrix making both orders accidentally agree."""
    assert _components(dot(ROWCOL, E_X)) != pytest.approx(_components(dot(E_X, ROWCOL)))


def test_vector_and_matrix_contractions_are_unchanged():
    """Only the mixed case moved: dot products, Frobenius products and scalars are as they were."""
    assert float(dot(vector(1, 2, 3), vector(4, 5, 6))) == pytest.approx(32.0)
    assert float(contract(vector(1, 2, 3), vector(4, 5, 6))) == pytest.approx(32.0)
    # A:B has no index order to get wrong, and contract falls through to it for two matrices
    assert float(double_dot(ROWCOL, ROWCOL)) == pytest.approx(1695.0)
    assert float(contract(ROWCOL, ROWCOL)) == pytest.approx(1695.0)
    # scaling by a scalar
    assert _components(contract(Expression(2), vector(1, 2, 3))) == pytest.approx([2.0, 4.0, 6.0])


def test_dot_between_two_matrices_is_rejected():
    """A*B and A:B are both plausible, so dot refuses rather than picking one."""
    with pytest.raises(RuntimeError, match="ambiguous"):
        dot(ROWCOL, ROWCOL)


def test_an_unpadded_tensor_still_contracts_with_a_padded_vector():
    """
    matrix(..., fill_to_max_vector_dim=False) is 2x2 while vector() always pads to three components.

    The mixed branch tolerates that by summing over the overlap, the same way the vector/vector branch
    and double_dot do, rather than raising on the shape mismatch. The result keeps the tensor's extent,
    so it is a two-component vector, which is why the components are compared individually rather than
    against vector(1,3) -- that would pad back to three and never match.
    """
    small = matrix([[Expression(1), Expression(2)], [Expression(3), Expression(4)]],
                   fill_to_max_vector_dim=False)
    assert _components(dot(small, E_X), n=2) == pytest.approx([1.0, 3.0])


# ----------------------------------------------------------------------------------------------
# grad and div. These need a compiled element, see the note at the top of the file.
# ----------------------------------------------------------------------------------------------

class _Project(Equations):
    """Projects each given scalar expression onto its own C2 field, so the nodal values are its value."""

    def __init__(self, expressions):
        super().__init__()
        self.expressions = expressions

    def define_fields(self):
        for name in self.expressions:
            self.define_scalar_field(name, "C2")

    def define_residuals(self):
        for name, expression in self.expressions.items():
            unknown, test = var_and_test(name)
            self.add_residual(weak(unknown - expression, test))


class _ProjectionProblem(Problem):
    def __init__(self, expressions, coordsys=None, lower_left=None, box=False):
        super().__init__()
        self.expressions = expressions
        self._coordsys = coordsys
        self._lower_left = lower_left
        self._box = box

    def define_problem(self):
        if self._coordsys is not None:
            self.set_coordinate_system(self._coordsys)
        if self._box:
            self.add_mesh(CuboidBrickMesh(N=2))
        else:
            self.add_mesh(RectangularQuadMesh(N=2, lower_left=self._lower_left or [0, 0]))
        self.add_equations(_Project(self.expressions) @ "domain")


def _projected(tmp_path, expressions, coordsys=None, lower_left=None, box=False):
    """{name: max |value| over the nodes} of each expression, as the generated code evaluates it."""
    with _ProjectionProblem(expressions, coordsys, lower_left, box) as problem:
        problem.set_output_directory(str(tmp_path))
        problem.initialise()
        problem.solve()
        mesh = problem.get_mesh("domain")
        indices = mesh.get_nodal_field_indices()
        return {name: max(abs(node.value(indices[name])) for node in mesh.nodes())
                for name in expressions}


def test_div_of_grad_is_the_laplacian(tmp_path):
    """
    div(grad(u)) is the vector Laplacian, which is only true because div contracts the second index.

    With the first-index convention this returned grad(div(u)) instead: for the field below that is
    (0, 2y, 0) rather than (2x, 0, 0), so the x-residual would come out at 2 instead of zero.
    """
    x, y = var(["coordinate_x", "coordinate_y"])
    u = vector(x * y * y, 0, 0)                 # laplacian is (2x, 0, 0); grad(div u) is (0, 2y, 0)
    laplacian = vector(2 * x, 0, 0)
    expressions = {f"residual_{i}": div(grad(u))[i] - laplacian[i] for i in range(3)}
    expressions["magnitude"] = div(grad(u))[0]  # guards against the residuals being trivially zero
    values = _projected(tmp_path, expressions)
    assert values["magnitude"] > 1.0
    for i in range(3):
        assert values[f"residual_{i}"] < 1e-10


# a and b are deliberately neither parallel nor equal, so that dyadic(a,b) is not symmetric and the
# two possible index conventions for div genuinely disagree.
_A_2D = lambda x, y: vector(x * y + 2 * y * y, 3 * x * x, 0)
_B_2D = lambda x, y: vector(5 * y, 7 * x * y, 0)


@pytest.mark.parametrize("coordsys_name", ["cartesian", "axisymmetric"])
def test_divergence_of_a_dyadic_product(tmp_path, coordsys_name):
    """
    div(dyadic(a,b)) == div(b)*a + (b.grad)a, the product rule for the second-index convention.

    Both sides of the reference are built from operators that are trusted independently -- the vector
    divergence and directional_derivative, which is matproduct(grad(a),b) -- so this pins the tensor
    divergence without hand-deriving any curvilinear connection term. In the axisymmetric system it is
    exactly those connection terms that the transposition inside div has to carry through correctly.

    a and b are kept swirl-free here: AxisymmetricCoordinateSystem.vector_gradient hard-zeros the
    azimuthal off-diagonals, so a tensor with T_phi_r != 0 is outside what that coordinate system
    describes at all, which is also why its azimuthal connection term differs in sign from the
    azimuthal-symmetry-breaking one (see the comment on AxisymmetryBreakingCoordinateSystem.
    tensor_divergence).
    """
    x, y = var(["coordinate_x", "coordinate_y"])
    a, b = _A_2D(x, y), _B_2D(x, y)
    if coordsys_name == "cartesian":
        coordsys, lower_left = None, [0, 0]
    else:
        # away from the axis, since the connection terms carry 1/r
        coordsys, lower_left = AxisymmetricCoordinateSystem(), [1, 0]

    lhs = div(dyadic(a, b))
    rhs = div(b) * a + directional_derivative(a, b)
    expressions = {f"residual_{i}": lhs[i] - rhs[i] for i in range(3)}
    expressions["magnitude"] = lhs[0]
    values = _projected(tmp_path, expressions, coordsys=coordsys, lower_left=lower_left)
    assert values["magnitude"] > 1.0
    for i in range(3):
        assert values[f"residual_{i}"] < 1e-11


def test_divergence_of_a_dyadic_product_in_3d(tmp_path):
    """The same identity in 3d, where all three rows of the tensor carry derivatives."""
    x, y, z = var(["coordinate_x", "coordinate_y", "coordinate_z"])
    a = vector(x * y + 2 * z, 3 * x * x * z, 4 * y * z)
    b = vector(5 * y * z, 7 * x * y, 2 * x * z)
    lhs = div(dyadic(a, b))
    rhs = div(b) * a + directional_derivative(a, b)
    expressions = {f"residual_{i}": lhs[i] - rhs[i] for i in range(3)}
    expressions["magnitude"] = lhs[0]
    values = _projected(tmp_path, expressions, box=True)
    assert values["magnitude"] > 1.0
    for i in range(3):
        assert values[f"residual_{i}"] < 1e-11


# ----------------------------------------------------------------------------------------------
# The one place in pyoomph that takes the divergence of a tensor: the conservative ALE momentum flux.
# ----------------------------------------------------------------------------------------------

class _FreeStreamOnAMovingMesh(Problem):
    """
    A uniform free stream on a mesh that is deliberately moved, with GCL Navier-Stokes.

    u = (1,0) with p constant is an exact solution: the viscous stress vanishes for a uniform field,
    div(u) = 0, and the conservative flux term d_j(rho*u_i*(u_j-w_j)) = -rho*u_i*div(w) is exactly
    what d/dt(int rho*u*q) contributes on a moving mesh, so the two cancel.

    The prescribed mesh velocity is NOT harmonic, and that is the whole point. The two possible
    orderings of the flux tensor differ by (-U*d_y w_y, U*d_x w_y), whose curl is U*laplacian(w_y). A
    Laplace-smoothed mesh makes w harmonic, the difference becomes a pure gradient, and the pressure
    field absorbs it -- so with LaplaceSmoothedMesh this test passes either way and proves nothing.
    """

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=3))
        X, Y = var(["lagrangian_x", "lagrangian_y"])
        mesh_velocity = vector(0, 0.3 * (Y * Y + X * Y))   # laplacian(w_y) = 0.6, not zero
        equations = NavierStokesEquations(mass_density=1, dynamic_viscosity=1, GCL=True)
        equations += PrescribedMovingMesh(mesh_velocity)
        equations += DirichletBC(mesh_x=True, mesh_y=True) @ "bottom"
        equations += DirichletBC(mesh_x=True) @ "left"
        equations += DirichletBC(mesh_x=True) @ "right"
        for boundary in ["left", "right", "top", "bottom"]:
            equations += DirichletBC(velocity_x=1, velocity_y=0) @ boundary
        equations += AverageConstraint(pressure=0)
        equations += InitialCondition(velocity_x=1, velocity_y=0)
        self.add_equations(equations @ "domain")


def test_gcl_momentum_flux_preserves_a_free_stream(tmp_path):
    """
    The conservative momentum flux has to be dyadic(u, u-w), i.e. F_ij = rho*u_i*(u_j-w_j).

    Written the other way round the free-stream error plateaus at about 2e-3 however far dt is refined
    (measured: 2.1e-3, 1.9e-3, 1.9e-3 at dt = 0.1, 0.025, 0.0125), whereas this ordering converges away
    (1.6e-4, 1.1e-7, 7.9e-11). The tolerance below sits between the two by orders of magnitude.
    """
    with _FreeStreamOnAMovingMesh() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.initialise()
        problem.run(0.3, startstep=0.0125, outstep=False, temporal_error=None)
        mesh = problem.get_mesh("domain")
        indices = mesh.get_nodal_field_indices()
        error_x = max(abs(node.value(indices["velocity_x"]) - 1.0) for node in mesh.nodes())
        error_y = max(abs(node.value(indices["velocity_y"])) for node in mesh.nodes())
    assert error_x < 1e-8
    assert error_y < 1e-8
