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
from pyoomph.expressions import (var, var_and_test, weak, matrix, vector, grad, div, dyadic, dot, trace,
                                 identity_matrix, partial_t, testfunction,
                                 contract, double_dot, matproduct, transpose, directional_derivative,
                                 Expression)
from pyoomph.expressions.coordsys import AxisymmetricCoordinateSystem
from pyoomph.equations.ALE import PrescribedMovingMesh, BaseMovingMeshEquations
from pyoomph.equations.generic import AverageConstraint
from pyoomph.equations.navier_stokes import NavierStokesEquations
from pyoomph.meshes.simplemeshes import RectangularQuadMesh, CuboidBrickMesh, LineMesh

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
    def __init__(self, expressions, coordsys=None, lower_left=None, box=False, line=False,
                 interface=False):
        super().__init__()
        self.expressions = expressions
        self._coordsys = coordsys
        self._lower_left = lower_left
        self._box = box
        self._line = line
        self._interface = interface

    def define_problem(self):
        if self._coordsys is not None:
            self.set_coordinate_system(self._coordsys)
        if self._box:
            self.add_mesh(CuboidBrickMesh(N=2))
        elif self._line:
            self.add_mesh(LineMesh(N=2, minimum=1))
        else:
            self.add_mesh(RectangularQuadMesh(N=2, lower_left=self._lower_left or [0, 0]))
        if self._interface:
            # On an interface the element dimension is one below the nodal one, which is what selects
            # the surface branches of the differential operators. The bulk still needs one field of its
            # own, or the coordinate space of the domain cannot be deduced.
            self.add_equations((_Project({"bulk_anchor": Expression(0)})
                                + _Project(self.expressions) @ "top") @ "domain")
        else:
            self.add_equations(_Project(self.expressions) @ "domain")


def _projected(tmp_path, expressions, coordsys=None, lower_left=None, box=False, line=False,
               stability=None, interface=False):
    """
    {name: max |value| over the nodes} of each expression, as the generated code evaluates it.

    ``stability`` is "azimuthal" or "cartesian_mode" to swap in the corresponding normal-mode
    coordinate system, which has to happen before initialise().

    Note what this does and does not see: it reads nodal values after a steady solve, i.e. the *base*
    residual. Under a normal-mode coordinate system the mesh-perturbation (mm*) and mode (I*k, I*m)
    terms live entirely in the eigen-residual and are structurally zero here, so this pins index
    patterns and connection terms and nothing else. Use _eigenmode_projected for the rest.
    """
    with _ProjectionProblem(expressions, coordsys, lower_left, box, line, interface) as problem:
        problem.set_output_directory(str(tmp_path))
        if stability == "azimuthal":
            problem.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
        elif stability == "cartesian_mode":
            problem.setup_for_stability_analysis(additional_cartesian_mode=True, analytic_hessian=False)
        problem.initialise()
        problem.solve()
        where = "domain/top" if interface else "domain"
        mesh = problem.get_mesh(where)
        indices = mesh.get_nodal_field_indices()
        return {name: max(abs(node.value(indices[name])) for node in mesh.nodes())
                for name in expressions}


# (name, coordinate system, kwargs for _projected, mesh dimension). The radial meshes and the flipped
# axisymmetric one are placed away from the symmetry axis, since the connection terms carry 1/r.
_ALL_SYSTEMS = [
    ("cartesian2d", None, {}, 2),
    ("cartesian3d", None, {"box": True}, 3),
    ("axisymmetric", AxisymmetricCoordinateSystem(), {"lower_left": [1, 0]}, 2),
    ("axisymmetric_flipped", AxisymmetricCoordinateSystem(use_x_as_symmetry_axis=True),
     {"lower_left": [0, 1]}, 2),
    ("axisymmetric_radial", AxisymmetricCoordinateSystem(), {"line": True}, 1),
]


def _vector_and_scalar(dimension):
    """A vector and a scalar field, referring only to coordinates the mesh in question actually has."""
    x, y, z = var(["coordinate_x", "coordinate_y", "coordinate_z"])
    if dimension == 1:
        return vector(2 * x * x + 3 * x, 0, 0), 3 * x * x + 2 * x + 1
    if dimension == 2:
        return vector(2 * x * x + 3 * y * x, 4 * y * y + x, 0), 3 * x * x + 2 * y * x + y * y + 1
    return (vector(2 * x * x + 3 * y * x, 4 * y * y + x * z, 5 * z * z + y),
            3 * x * x + 2 * y * x + y * y + z * z + 1)


@pytest.mark.parametrize("name,coordsys,kwargs,dimension", _ALL_SYSTEMS,
                         ids=[entry[0] for entry in _ALL_SYSTEMS])
def test_trace_of_grad_is_the_divergence(tmp_path, name, coordsys, kwargs, dimension):
    """trace(grad(u)) == div(u) in every coordinate system, connection terms and all."""
    u, _ = _vector_and_scalar(dimension)
    expressions = {"residual": trace(grad(u)) - div(u), "magnitude": div(u)}
    values = _projected(tmp_path, expressions, coordsys=coordsys, **kwargs)
    assert values["magnitude"] > 1.0
    assert values["residual"] < 1e-11


@pytest.mark.parametrize("name,coordsys,kwargs,dimension", _ALL_SYSTEMS,
                         ids=[entry[0] for entry in _ALL_SYSTEMS])
def test_divergence_of_a_scalar_times_identity_is_its_gradient(tmp_path, name, coordsys, kwargs,
                                                               dimension):
    """
    div(f*I) == grad(f), since d_j(f*delta_ij) = d_i f. This is the pressure part of any stress
    divergence, and a sharp test of the connection terms: in cylindrical coordinates the radial row
    only comes out right because the (T_rr - T_phiphi)/r hoop term cancels for an isotropic tensor.

    It also pins something duller. The identity tensor is the cheapest expression with a nonzero
    out-of-plane component, and the Cartesian tensor divergence used to sum over all three coordinates
    whatever the mesh dimension, so on a two-dimensional mesh this died with "Cannot expand the field
    'coordinate_z'" -- coordinate_z is not a field of the element, so reaching for a derivative that is
    merely zero is not free.
    """
    _, f = _vector_and_scalar(dimension)
    tensor_divergence, gradient = div(f * identity_matrix()), grad(f)
    expressions = {f"residual_{i}": tensor_divergence[i] - gradient[i] for i in range(3)}
    expressions["magnitude"] = gradient[0]
    values = _projected(tmp_path, expressions, coordsys=coordsys, **kwargs)
    assert values["magnitude"] > 1.0
    for i in range(3):
        assert values[f"residual_{i}"] < 1e-11


@pytest.mark.parametrize("name,coordsys,kwargs,dimension", _ALL_SYSTEMS,
                         ids=[entry[0] for entry in _ALL_SYSTEMS])
def test_directional_derivative_of_a_dyadic_product(tmp_path, name, coordsys, kwargs, dimension):
    """
    (d.grad)(a (x) b) == ((d.grad)a) (x) b + a (x) ((d.grad)b), the product rule for the advection
    operator on a tensor.

    This is what material_derivative of a tensor field is built from, and in a curvilinear system it
    carries connection terms of its own, separate from those of the divergence. As with the divergence,
    the reference uses only operators trusted independently -- directional_derivative of a vector, which
    is matproduct(grad(a),b).

    On a radial mesh only the diagonal of a tensor is meaningful anyway (define_tensor_field builds a
    diagonal there), so the dyads below are chosen in-plane; the off-diagonal entries of the
    one-dimensional branch of AxisymmetricCoordinateSystem.directional_tensor_derivative drop the radial
    derivative, but nothing that coordinate system can build reaches them, for the same reason its
    vector_gradient is swirl-free.
    """
    x, y, z = var(["coordinate_x", "coordinate_y", "coordinate_z"])
    if dimension == 1:
        a, b, d = vector(2 * x * x + 1, 0, 0), vector(3 * x + 2, 0, 0), vector(5 * x, 0, 0)
    elif dimension == 2:
        a, b, d = vector(2 * x * x + y, 3 * y * y, 0), vector(3 * x + 2 * y, y * x, 0), vector(5 * y, 7 * x, 0)
    else:
        a = vector(2 * x * x + y, 3 * y * y, z * x)
        b, d = vector(3 * x + 2 * y, y * x, z), vector(5 * y, 7 * x, 2 * z)

    lhs = directional_derivative(dyadic(a, b), d)
    rhs = (dyadic(directional_derivative(a, d), b) + dyadic(a, directional_derivative(b, d))).evalm()
    expressions = {f"residual_{i}{j}": lhs[i, j] - rhs[i, j] for i in range(3) for j in range(3)}
    expressions["magnitude"] = lhs[0, 0]
    values = _projected(tmp_path, expressions, coordsys=coordsys, **kwargs)
    assert values["magnitude"] > 1.0
    for i in range(3):
        for j in range(3):
            assert values[f"residual_{i}{j}"] < 1e-10


def test_divergence_of_a_vector_with_an_out_of_plane_component(tmp_path):
    """
    div(vector(a,b,c)) on a two-dimensional mesh, where the third slot has no coordinate to pair with.

    The Cartesian vector divergence used to sum over the vector's three padded slots rather than the
    mesh's coordinates, so a nonzero third component reached for d/dz and died with "Cannot expand the
    field 'coordinate_z'". The out-of-plane component simply does not contribute in 2d.
    """
    x, y = var(["coordinate_x", "coordinate_y"])
    planar, out_of_plane = vector(3 * x * x, 2 * y * y, 0), vector(3 * x * x, 2 * y * y, 5 * x * y)
    expressions = {"residual": div(out_of_plane) - div(planar), "magnitude": div(planar)}
    values = _projected(tmp_path, expressions)
    assert values["magnitude"] > 1.0
    assert values["residual"] < 1e-11


def test_divergence_of_a_vector_with_swirl_on_a_radial_mesh(tmp_path):
    """
    div(vector(u_r,u_phi,0)) on a one-dimensional radial mesh, where slot 1 is the azimuthal component.

    The axisymmetric vector divergence gated its axial term on arg.nops(), which is 3 for every padded
    vector, so on a radial mesh it reached for d/dy -- a coordinate that mesh does not have. It stayed
    invisible while the azimuthal slot was zero, since d/dy of zero never needs coordinate_y expanded.
    The azimuthal component contributes (1/r)*d_phi(u_phi), which axisymmetry makes zero.
    """
    x = var("coordinate_x")
    radial_only, with_swirl = vector(3 * x * x, 0, 0), vector(3 * x * x, 4 * x + 1, 0)
    expressions = {"residual": div(with_swirl) - div(radial_only), "magnitude": div(radial_only)}
    values = _projected(tmp_path, expressions, coordsys=AxisymmetricCoordinateSystem(), line=True)
    assert values["magnitude"] > 1.0
    assert values["residual"] < 1e-11


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


@pytest.mark.parametrize("coordsys_name", ["cartesian", "axisymmetric", "axisymmetric_flipped"])
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
    elif coordsys_name == "axisymmetric":
        # away from the axis, since the connection terms carry 1/r
        coordsys, lower_left = AxisymmetricCoordinateSystem(), [1, 0]
    else:
        # use_x_as_symmetry_axis, i.e. the exported "axisymmetric_flipped": y is the radial direction
        # here, so it is y that has to stay away from zero. Its rows are a separate branch.
        coordsys = AxisymmetricCoordinateSystem(use_x_as_symmetry_axis=True)
        lower_left = [0, 1]

    lhs = div(dyadic(a, b))
    rhs = div(b) * a + directional_derivative(a, b)
    expressions = {f"residual_{i}": lhs[i] - rhs[i] for i in range(3)}
    expressions["magnitude"] = lhs[0]
    values = _projected(tmp_path, expressions, coordsys=coordsys, lower_left=lower_left)
    assert values["magnitude"] > 1.0
    for i in range(3):
        assert values[f"residual_{i}"] < 1e-11


@pytest.mark.parametrize("case", ["radial_x_azimuthal", "azimuthal_x_radial", "azimuthal_x_azimuthal"])
def test_axisymmetric_tensor_divergence_with_swirl(tmp_path, case):
    """
    The azimuthal row of the axisymmetric tensor divergence, against hand-derived references.

    This row is unreachable from any tensor the axisymmetric system can build from fields:
    define_tensor_field puts the azimuthal component on the diagonal only and vector_gradient hard-zeros
    the azimuthal off-diagonals, since plain axisymmetry has no azimuthal velocity component. So the
    dyadic-identity test above cannot reach it either -- (b.grad)a on the right-hand side would need the
    swirl terms of vector_gradient that are not there -- and the reference has to come from outside.

    Each case below is div(a (x) b) = div(b)*a + (b.grad)a worked out by hand in the orthonormal
    cylindrical frame, using d_phi(anything) = 0 and the frame derivatives d_phi(e_r) = e_phi,
    d_phi(e_phi) = -e_r. Slot order is (r, z, phi), so vector(h,0,0) is radial and vector(0,0,k)
    azimuthal.

    Both forms of this row were wrong before: the 2d one had (T_phir - T_rphi)/r, which is zero for a
    symmetric tensor and the wrong sign otherwise, and the radial one had (2*T_phir - T_rphi)/r, which is
    wrong even for a symmetric tensor. The first case here is what pins the sign -- it has T_rphi != 0
    and T_phir == 0, so the old expression gave exactly minus the right answer.
    """
    r, z = var(["coordinate_x", "coordinate_y"])
    h, k, f, g = 2 * r + 3 * z, 5 * r * z + 1, r * r + 2 * z, 4 * r + z
    if case == "radial_x_azimuthal":
        # div(e_r h (x) e_phi k): div(e_phi k) = 0 and (e_phi k .grad)(e_r h) = h*k/r * e_phi
        tensor, reference = dyadic(vector(h, 0, 0), vector(0, 0, k)), vector(0, 0, h * k / r)
    elif case == "azimuthal_x_radial":
        # div(e_phi f (x) e_r g): div(e_r g) = d_r g + g/r and (e_r g .grad)(e_phi f) = g d_r f * e_phi
        tensor = dyadic(vector(0, 0, f), vector(g, 0, 0))
        reference = vector(0, 0, f * (grad(g)[0] + g / r) + g * grad(f)[0])
    else:
        # div(e_phi f (x) e_phi k): div(e_phi k) = 0 and (e_phi k .grad)(e_phi f) = -f*k/r * e_r
        tensor, reference = dyadic(vector(0, 0, f), vector(0, 0, k)), vector(-f * k / r, 0, 0)

    divergence = div(tensor)
    expressions = {f"residual_{i}": divergence[i] - reference[i] for i in range(3)}
    expressions["magnitude"] = divergence[0] + divergence[1] + divergence[2]
    values = _projected(tmp_path, expressions, coordsys=AxisymmetricCoordinateSystem(),
                        lower_left=[1, 0])
    assert values["magnitude"] > 1.0
    for i in range(3):
        assert values[f"residual_{i}"] < 1e-11


def test_axisymmetric_tensor_divergence_with_swirl_on_a_radial_mesh(tmp_path):
    """
    The same azimuthal row on a one-dimensional radial mesh, which is a separate branch.

    There the components are ordered (r, phi) rather than (r, z, phi), so the azimuthal slot is index 1
    and index 2 is unused. Its connection term used to read (2*T_phir - T_rphi)/r, which is wrong for a
    symmetric tensor too, not only in sign -- and nothing exercised it, so reverting it alone left every
    other test in this file green.
    """
    r = var("coordinate_x")
    h, k = 2 * r + 1, 3 * r * r
    # div(e_r h (x) e_phi k), as above: div(e_phi k) = 0 and (e_phi k .grad)(e_r h) = h*k/r * e_phi
    tensor = dyadic(vector(h, 0, 0), vector(0, k, 0))
    reference = vector(0, h * k / r, 0)
    divergence = div(tensor)
    expressions = {f"residual_{i}": divergence[i] - reference[i] for i in range(3)}
    expressions["magnitude"] = divergence[1]
    values = _projected(tmp_path, expressions, coordsys=AxisymmetricCoordinateSystem(), line=True)
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


# ----------------------------------------------------------------------------------------------
# The normal-mode coordinate systems, used for azimuthal and Cartesian-wavenumber stability
# analysis. These had no coverage of their tensor operations at all, and
# AxisymmetryBreakingCoordinateSystem.tensor_divergence is the one implementation whose indices the
# move to the second-index convention rewrote without a test to catch it.
# ----------------------------------------------------------------------------------------------

def test_azimuthal_stability_coordinate_system_keeps_the_identities(tmp_path):
    """
    With azimuthal stability active the coordinate system becomes AxisymmetryBreaking, whose div, grad
    and tensor_divergence carry the mode number m and the first-order mesh perturbation. All three
    identities must still hold, and the base state has to reduce to the plain axisymmetric answer.

    What this does and does not cover. The projected values are the *base* residual, i.e. the eps -> 0
    limit, so this pins the index pattern and the connection terms of the second-index tensor divergence
    -- which is what the rewrite touched -- and confirms the reduction to AxisymmetricCoordinateSystem.
    It does not reach the m-dependent or eps-dependent terms: those live in the eigen-residual, which a
    projection cannot see. The generated code does differ from the plain axisymmetric run (roughly 53 kB
    against 38 kB), so the mode machinery is genuinely engaged rather than skipped.
    """
    r, z = var(["coordinate_x", "coordinate_y"])
    a = vector(2 * r * r + z * r, 3 * z * z, 0)
    b = vector(3 * r + 2 * z, z * r, 0)
    f = 3 * r * r + 2 * z * r + z * z + 1

    expressions = {"trace_grad_minus_div": trace(grad(a)) - div(a)}
    for i in range(3):
        expressions[f"identity_{i}"] = div(f * identity_matrix())[i] - grad(f)[i]
    lhs, rhs = div(dyadic(a, b)), div(b) * a + directional_derivative(a, b)
    for i in range(3):
        expressions[f"dyadic_{i}"] = lhs[i] - rhs[i]
    expressions["magnitude"] = lhs[0]

    baseline = _projected(tmp_path / "plain", expressions, coordsys=AxisymmetricCoordinateSystem(),
                          lower_left=[1, 0])
    values = _projected(tmp_path / "azimuthal", expressions,
                        coordsys=AxisymmetricCoordinateSystem(), lower_left=[1, 0],
                        stability="azimuthal")
    assert values["magnitude"] > 1.0
    for name, value in values.items():
        if name == "magnitude":
            # the base state must agree with the plain axisymmetric system, not merely be self-consistent
            assert abs(value - baseline[name]) < 1e-10
        else:
            assert value < 1e-10, name


@pytest.mark.parametrize("case", ["axisymmetric", "axisymmetric_flipped", "axisymmetric_radial",
                                  "radial_mesh_radial_direction"])
def test_axisymmetric_directional_tensor_derivative_with_swirl(case, tmp_path):
    """
    The frame-rotation terms of AxisymmetricCoordinateSystem.directional_tensor_derivative.

    Same situation as the swirl tests of tensor_divergence above and for the same reason: every one of
    these terms multiplies an azimuthal off-diagonal, and plain axisymmetry cannot build a tensor that
    has one (dev-doc section 4). So each branch had a wrong entry that nothing could reach, and each
    was found by requiring C(T^transpose) == C(T)^transpose of the rotation, which none satisfied.

    References are hand-derived in the orthonormal frame from d_phi(e_r) = e_phi and
    d_phi(e_phi) = -e_r, one case per branch and per defect:

      axisymmetric               (r,z,phi): (e_phi f (x) e_z . e_phi g).grad = -f*g/r * e_r (x) e_z
                                 -- the code had +T_zphi/r here, i.e. exactly minus this
      axisymmetric_flipped       (z,r,phi): (e_z f (x) e_r . e_phi g).grad = +f*g/r * e_z (x) e_phi
                                 -- the code read the transposed entry, which is zero here
      axisymmetric_radial        (r,phi):   (e_r f (x) e_phi . e_phi g).grad
                                            = f*g/r * (e_phi (x) e_phi - e_r (x) e_r)
                                 -- the code had no azimuthal term on the diagonal at all
      radial_mesh_radial_direction         the same branch along e_r, where the code omitted the
                                            radial derivative on the off-diagonals entirely
    """
    if case == "axisymmetric_flipped":
        # x is the symmetry axis, so the slots are (z, r, phi) and r is the y coordinate
        z, r = var(["coordinate_x", "coordinate_y"])
        f, g = 2 * z + 3 * r, 5 * r * z + 1
        tensor, direction = dyadic(vector(f, 0, 0), vector(0, 1, 0)), vector(0, 0, g)
        reference = [[0, 0, f * g / r], [0, 0, 0], [0, 0, 0]]
        probe = (0, 2)
        kwargs = {"coordsys": AxisymmetricCoordinateSystem(use_x_as_symmetry_axis=True),
                  "lower_left": [0, 1]}
    elif case == "axisymmetric":
        r, z = var(["coordinate_x", "coordinate_y"])
        f, g = 2 * r + 3 * z, 5 * r * z + 1
        tensor, direction = dyadic(vector(0, 0, f), vector(0, 1, 0)), vector(0, 0, g)
        reference = [[0, -f * g / r, 0], [0, 0, 0], [0, 0, 0]]
        probe = (0, 1)
        kwargs = {"coordsys": AxisymmetricCoordinateSystem(), "lower_left": [1, 0]}
    else:
        # a radial mesh, where the components are ordered [r, phi] and slot 2 is unused
        r = var("coordinate_x")
        f, g = 2 * r + 1, 3 * r * r
        tensor = dyadic(vector(f, 0, 0), vector(0, 1, 0))
        if case == "axisymmetric_radial":
            direction = vector(0, g, 0)
            reference = [[-f * g / r, 0, 0], [0, f * g / r, 0], [0, 0, 0]]
            probe = (1, 1)
        else:
            # along e_r instead: the plain radial derivative, which the off-diagonals used to drop
            direction = vector(g, 0, 0)
            reference = [[0, g * grad(f)[0], 0], [0, 0, 0], [0, 0, 0]]
            probe = (0, 1)
        kwargs = {"coordsys": AxisymmetricCoordinateSystem(), "line": True}

    result = directional_derivative(tensor, direction)
    expressions = {f"residual_{i}{j}": result[i, j] - reference[i][j]
                   for i in range(3) for j in range(3)}
    # a single characteristic entry, not the sum: the radial case has two that cancel exactly
    expressions["magnitude"] = result[probe[0], probe[1]]
    values = _projected(tmp_path, expressions, **kwargs)
    assert values["magnitude"] > 1.0
    for name, value in values.items():
        if name != "magnitude":
            assert value < 1e-11, name


def _all_identities(a, b, d, f):
    """
    The identities of the module header for every operator, given four ingredients.

    ``a``, ``b``, ``d`` are vectors and ``f`` a scalar. The dyads are deliberately non-symmetric, or the
    index convention of the rank-2 operators goes untested.
    """
    expressions = {"trace_grad_minus_div": trace(grad(a)) - div(a)}
    for i in range(3):
        expressions[f"identity_{i}"] = div(f * identity_matrix())[i] - grad(f)[i]
    lhs, rhs = div(dyadic(a, b)), div(b) * a + directional_derivative(a, b)
    for i in range(3):
        expressions[f"dyadic_{i}"] = lhs[i] - rhs[i]
    # the product rule for the directional derivative of a tensor, the only in-system reference for
    # directional_tensor_derivative: both sides of it use only grad of a vector, trusted independently
    dtd = directional_derivative(dyadic(a, b), d)
    dref = dyadic(directional_derivative(a, d), b) + dyadic(a, directional_derivative(b, d))
    for i in range(3):
        for j in range(3):
            expressions[f"advect_{i}{j}"] = (dtd - dref)[i, j]
    expressions["magnitude"] = lhs[0]
    expressions["advect_magnitude"] = dtd[0, 0]
    return expressions


def test_azimuthal_stability_coordinate_system_keeps_the_tensor_advection_identity(tmp_path):
    """
    The product rule for AxisymmetryBreakingCoordinateSystem.directional_tensor_derivative.

    That method used to raise outright, which is what kept viscoelastic azimuthal stability analysis
    out. Unlike plain axisymmetry (dev-doc section 5), the reference here is not incomplete in the same
    way as the thing it tests: this class's vector_gradient carries all three components, so
    directional_derivative of a vector -- which routes to matproduct(grad(.), d) -- genuinely reaches
    the swirl entries that the tensor version has to get right.

    Base mode only, as for the divergence above; the eigen-level half is covered by
    test_cartesian_normal_mode_operators_in_an_actual_eigenmode's azimuthal sibling.
    """
    r, z = var(["coordinate_x", "coordinate_y"])
    a = vector(2 * r * r + z * r, 3 * z * z, r * z + 2 * r)
    b = vector(3 * r + 2 * z, z * r, 4 * z + r * r)
    d = vector(r + z, 2 * r * z, 3 * r + z * z)
    f = 3 * r * r + 2 * z * r + z * z + 1

    values = _projected(tmp_path, _all_identities(a, b, d, f),
                        coordsys=AxisymmetricCoordinateSystem(), lower_left=[1, 0],
                        stability="azimuthal")
    assert values["advect_magnitude"] > 1.0
    for name, value in values.items():
        if not name.endswith("magnitude"):
            assert value < 1e-10, name


def test_cartesian_normal_mode_coordinate_system_keeps_the_identities(tmp_path):
    """
    All four operators of CartesianCoordinateSystemWithAdditionalNormalMode, on a 2d and on a 1d mesh.

    Its tensor_divergence and directional_tensor_derivative used to raise, which is what blocked
    NavierStokesEquations(GCL=True) -- it takes the divergence of a momentum flux tensor -- and the
    viscoelastic module from being combined with
    setup_for_stability_analysis(additional_cartesian_mode=True) at all.

    The 1d variant is not redundant: that is where the component order is [x, x_add] rather than
    [x, y], so the additional direction sits in slot 1 and slot 2 is unused. A row/column slip there is
    invisible in 2d.

    Base mode only -- see test_cartesian_normal_mode_operators_in_an_actual_eigenmode for the terms a
    projection cannot see.
    """
    x, y = var(["coordinate_x", "coordinate_y"])
    a = vector(2 * x * x + y * x, 3 * y * y, x * y + 2 * x)
    b = vector(3 * x + 2 * y, y * x, 4 * y + x * x)
    d = vector(x + y, 2 * x * y, 3 * x + y * y)
    f = 3 * x * x + 2 * y * x + y * y + 1
    expressions = _all_identities(a, b, d, f)

    baseline = _projected(tmp_path / "plain", expressions)
    values = _projected(tmp_path / "normal_mode", expressions, stability="cartesian_mode")
    assert values["magnitude"] > 1.0
    for name, value in values.items():
        if name.endswith("magnitude"):
            # the base state must agree with the plain Cartesian system, not merely be self-consistent
            assert abs(value - baseline[name]) < 1e-10, name
        else:
            assert value < 1e-10, name

    # 1d: the [x, x_add] slot order. Only the third vector slot is unused here, so the dyads are built
    # from two-component vectors and the identities reduce accordingly.
    a1 = vector(2 * x * x + 3 * x, x * x, 0)
    b1 = vector(4 * x + 1, 2 * x * x, 0)
    d1 = vector(x + 2, 3 * x, 0)
    f1 = 3 * x * x + 2 * x + 1
    values = _projected(tmp_path / "line", _all_identities(a1, b1, d1, f1), line=True,
                        stability="cartesian_mode")
    assert values["magnitude"] > 1.0
    for name, value in values.items():
        if not name.endswith("magnitude"):
            assert value < 1e-10, name


class _MeshSlavedToADiffusingField(BaseMovingMeshEquations):
    """
    A moving mesh whose displacement follows a diffusing scalar, so that eigenvectors move the mesh.

    The point is to get a first-order mesh perturbation into an eigenvector as cheaply as possible,
    since that is what multiplies every mm* term in the normal-mode coordinate systems. A smoothed mesh
    on its own will not do it: its position equations carry no time derivative, so the eigensolver
    refuses the problem outright with an empty mass matrix, and prescribing the mesh velocity instead
    pins the position dofs, which leaves the perturbation identically zero and every mm* term untested
    while the test still passes.

    So the mass matrix comes from a diffusing field s, and the mesh is slaved to it with two different
    factors. Both mesh components are then excited by the same eigenvector, unlike a smoother, whose
    components do not couple.
    """

    def define_fields(self):
        super().define_fields()
        self.define_scalar_field("s", "C2")

    def define_residuals(self):
        s, s_test = var_and_test("s")
        self.add_residual(weak(partial_t(s), s_test) + weak(grad(s), grad(s_test)))
        X, Y = var(["lagrangian_x", "lagrangian_y"])
        self.add_residual(weak(var("mesh_x") - (X + 3 * s), testfunction("mesh_x")))
        self.add_residual(weak(var("mesh_y") - (Y + 2 * s), testfunction("mesh_y")))


class _MeshEigenmodeProblem(Problem):
    def __init__(self, expressions, coordsys=None, lower_left=None):
        super().__init__()
        self.expressions = expressions
        self._coordsys = coordsys
        self._lower_left = lower_left

    def define_problem(self):
        if self._coordsys is not None:
            self.set_coordinate_system(self._coordsys)
        self.add_mesh(RectangularQuadMesh(N=3, lower_left=self._lower_left or [0, 0]))
        equations = _MeshSlavedToADiffusingField() + _Project(self.expressions)
        for boundary in ("left", "right", "top", "bottom"):
            equations += DirichletBC(s=0) @ boundary
        self.add_equations(equations @ "domain")


def _eigenmode_projected(tmp_path, expressions, stability, k=0.7, n=6, coordsys=None, lower_left=None):
    """
    {name: max |value|} of each projected expression *in the eigenvector*, i.e. its first-order value.

    This is what _projected cannot do. A projection residual weak(unknown - expr, test) has no mass
    matrix row, so its row of the eigenproblem reads d(unknown) - d(expr) = 0: the eigenvector entry for
    that field IS the first-order-in-eps value of the expression at the requested wavenumber. So every
    mm* mesh-perturbation term and every I*k (or I*m) term is live here, which is exactly the half of
    the normal-mode operators that a steady projection leaves structurally zero.

    Both the real and the imaginary part of each eigenvector are read back, because the entry is the
    complex first-order value and the eigensolver is free to return the whole vector multiplied by any
    phase, which can leave either part alone at zero.

    Only expressions that are *identically* zero as expressions can be compared against zero this way.
    A projection of something the C2 space cannot represent exactly leaves a non-zero residual at the
    base state, and the first-order change of the integration measure then feeds back into the
    eigenvector entry. Identity residuals are exactly zero, so they are clean; anything else here is
    only a magnitude check.
    """
    with _MeshEigenmodeProblem(expressions, coordsys, lower_left) as problem:
        problem.set_output_directory(str(tmp_path))
        if stability == "azimuthal":
            problem.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
            mode = {"azimuthal_m": 2}
        else:
            problem.setup_for_stability_analysis(additional_cartesian_mode=True, analytic_hessian=False)
            mode = {"normal_mode_k": k}
        problem.initialise()
        problem.solve()
        problem.solve_eigenproblem(n, **mode)

        mesh = problem.get_mesh("domain")
        indices = mesh.get_nodal_field_indices()
        worst = {name: 0.0 for name in expressions}
        for eigenvector in problem.get_last_eigenvectors():
            for part in (eigenvector.real, eigenvector.imag):
                problem.set_current_dofs(list(part))
                for name in expressions:
                    worst[name] = max(worst[name],
                                      max(abs(node.value(indices[name])) for node in mesh.nodes()))
        return worst


@pytest.mark.parametrize("stability", ["cartesian_mode", "azimuthal"])
def test_normal_mode_tensor_operators_in_an_actual_eigenmode(tmp_path, stability):
    """
    The identities of the module header inside an eigenvector, where the mode terms are alive.

    This is the test the tensor operations of the two normal-mode coordinate systems actually needed.
    Everything else in this file projects a steady solution, i.e. the base residual, in which every
    mm* and every I*k/I*m term is multiplied by zero -- so the older normal-mode tests pinned the index
    pattern and the connection terms and nothing whatsoever of the perturbation.

    The two mesh_perturbation_ entries are the guard that this is not vacuous: they are the first-order
    values of the mesh coordinates themselves, so a run in which the eigenvector happened to leave the
    mesh alone -- which is what happens with PrescribedMovingMesh, whose mesh dofs are pinned -- would
    fail on them rather than pass with every mm* term silently multiplied by nothing. Both components
    are required, since a single relaxation mode of the smoother excites only one.

    Reverting checks: dropping the mm* term from d_dz, or transposing T[i,j] in tensor_divergence, or
    dropping the frame rotation from the azimuthal directional_tensor_derivative, each fails this.
    """
    if stability == "azimuthal":
        coordsys, lower_left = AxisymmetricCoordinateSystem(), [1, 0]
    else:
        coordsys, lower_left = None, None
    x, y = var(["coordinate_x", "coordinate_y"])
    a = vector(2 * x * x + y * x, 3 * y * y, x * y + 2 * x)
    b = vector(3 * x + 2 * y, y * x, 4 * y + x * x)
    d = vector(x + y, 2 * x * y, 3 * x + y * y)
    f = 3 * x * x + 2 * y * x + y * y + 1

    expressions = {name: expression for name, expression in _all_identities(a, b, d, f).items()
                   if not name.endswith("magnitude")}
    expressions["mesh_perturbation_x"] = var("mesh_x")
    expressions["mesh_perturbation_y"] = var("mesh_y")

    values = _eigenmode_projected(tmp_path, expressions, stability,
                                  coordsys=coordsys, lower_left=lower_left)
    assert values["mesh_perturbation_x"] > 1e-3, "the eigenvectors leave mesh_x alone, nothing is tested"
    assert values["mesh_perturbation_y"] > 1e-3, "the eigenvectors leave mesh_y alone, nothing is tested"
    for name, value in values.items():
        if not name.startswith("mesh_perturbation"):
            assert value < 1e-8, name


def _eigen_operator_values(tmp_path, expressions, stability, n=2):
    """
    Per eigenvector, the complex first-order nodal values of every projected expression, phase-fixed.

    Eigenvectors are defined up to a complex factor, so each is divided by its own entry at the node
    where the reference field is largest. That makes the arrays comparable between two runs of the same
    mesh, which is what the k=0 reduction below needs.
    """
    import numpy

    with _MeshEigenmodeProblem(expressions) as problem:
        problem.set_output_directory(str(tmp_path))
        if stability:
            problem.setup_for_stability_analysis(additional_cartesian_mode=True, analytic_hessian=False)
        problem.initialise()
        problem.solve()
        problem.solve_eigenproblem(n, **({"normal_mode_k": 0.0} if stability else {}))

        mesh = problem.get_mesh("domain")
        indices = mesh.get_nodal_field_indices()

        def nodal(part):
            problem.set_current_dofs(list(part))
            return {name: numpy.array([node.value(indices[name]) for node in mesh.nodes()])
                    for name in expressions}

        result = []
        for eigenvalue, eigenvector in zip(problem.get_last_eigenvalues(),
                                           problem.get_last_eigenvectors()):
            real, imag = nodal(eigenvector.real), nodal(eigenvector.imag)
            values = {name: real[name] + 1j * imag[name] for name in expressions}
            phase = values["reference"][int(numpy.argmax(numpy.abs(values["reference"])))]
            result.append((complex(eigenvalue), {n: v / phase for n, v in values.items()}))
        return result


def _slepc_or_scipy_fallback():
    """Whether the eigensolves below would fall back to scipy.

    pyoomph picks its eigensolver by what imports, and says little about it, so a wheel-test
    environment without petsc4py silently exercises the scipy backend instead. Measured on the same
    machine and the same commit: with slepc_mumps the k=0 test below passes, with the eigensolver
    forced to "scipy" it fails by O(1) - the modes come back in a basis this comparison cannot
    undo. A SLEPc without MUMPS was not tried.
    """
    try:
        from petsc4py import PETSc  # type:ignore  # noqa: F401
        import slepc4py  # type:ignore  # noqa: F401
    except Exception:
        return "the eigensolve would fall back to scipy (no petsc4py/slepc4py), whose eigenvectors this comparison cannot phase-fix"
    return None


_EIGEN_SKIP = _slepc_or_scipy_fallback()


@pytest.mark.skipif(_EIGEN_SKIP is not None, reason=str(_EIGEN_SKIP))
def test_cartesian_normal_mode_operators_reduce_to_plain_cartesian_at_k_zero(tmp_path):
    """
    At k=0 the new operators must reproduce what the ordinary eigensolve computes, term by term.

    This is the one check here whose reference is outside the coordinate system. Every other test
    compares the tensor operations against grad and div of the same class, so a defect shared by both
    sides of an identity would cancel and pass. Here the *raw* operator values are compared, not
    identity residuals, against a run of the same problem in plain Cartesian coordinates, where the
    mesh sensitivity of a residual comes from the C++ shape-derivative machinery instead.

    What it reaches, and what it does not. Measured, by zeroing terms and re-running: at k=0 the
    in-plane mesh corrections of d_dx and d_dy drop out of the eigen-residual entirely -- removing them
    leaves every value in this comparison bit-identical -- so what is pinned here is the index pattern
    and the derivative structure of the tensor operations against an independent implementation, not
    their mm* terms. Those are covered by test_normal_mode_tensor_operators_in_an_actual_eigenmode,
    which runs at k != 0 and does fail when they are perturbed. The reason the two are separable is
    presumably that set_ignore_dpsi_coord_diffs_in_jacobian, which problem.py sets for the normal-mode
    contributions, leaves nothing for the corrections to replace once the additional direction is gone;
    that has not been established, only observed.

    Only the leading eigenvalues are used: the higher relaxation modes of this problem come in
    near-degenerate pairs, where the eigenvector basis within the pair is not determined and a
    componentwise comparison is meaningless.
    """
    import numpy

    x, y = var(["coordinate_x", "coordinate_y"])
    a = vector(2 * x * x + y * x, 3 * y * y, x * y + 2 * x)
    b = vector(3 * x + 2 * y, y * x, 4 * y + x * x)
    d = vector(x + y, 2 * x * y, 3 * x + y * y)
    tensor = dyadic(a, b)
    divergence, advected = div(tensor), directional_derivative(tensor, d)

    expressions = {"reference": var("mesh_x")}
    for i in range(3):
        expressions[f"divT_{i}"] = divergence[i]
        for j in range(3):
            expressions[f"advT_{i}{j}"] = advected[i, j]

    plain = _eigen_operator_values(tmp_path / "plain", expressions, stability=False)
    normal_mode = _eigen_operator_values(tmp_path / "k0", expressions, stability=True)

    for (eigenvalue, left), (eigenvalue_k0, right) in zip(plain, normal_mode):
        assert abs(eigenvalue - eigenvalue_k0) < 1e-8 * (1 + abs(eigenvalue))
        scale = max(numpy.max(numpy.abs(left[name])) for name in expressions)
        assert scale > 1.0
        for name in expressions:
            assert numpy.max(numpy.abs(left[name] - right[name])) < 1e-9 * scale, name


def test_normal_mode_tensor_operations_are_refused_on_a_surface(tmp_path):
    """
    The bulk-only restriction, pinned as a restriction.

    Both normal-mode systems implement their tensor operations for ndim==edim only. The surface branches
    of their vector_divergence are of an entirely different shape -- the 2d/1d one is a thirty-product
    expression in local_coordinate_1 containing structurally dead diff(0, s1) factors -- so the rank-2
    surface case is refused rather than guessed at. Replaces the older test that pinned the whole
    operation as unimplemented.
    """
    x, y = var(["coordinate_x", "coordinate_y"])
    f = 3 * x * x + 2 * y * x + 1
    tensor_on_the_interface = div(f * identity_matrix())[0]
    for stability in ("cartesian_mode", "azimuthal"):
        with pytest.raises(RuntimeError, match="bulk mesh"):
            _projected(tmp_path / stability, {"out": tensor_on_the_interface},
                       coordsys=AxisymmetricCoordinateSystem() if stability == "azimuthal" else None,
                       lower_left=[1, 0] if stability == "azimuthal" else None,
                       stability=stability, interface=True)


# ----------------------------------------------------------------------------------------------
# A tensor UNKNOWN in the azimuthally-broken system.
#
# AxisymmetryBreakingCoordinateSystem carried working tensor operations long before it could define a
# tensor field to feed them: it inherited define_tensor_field from plain axisymmetry, which puts the
# azimuthal direction on the diagonal alone and hard-zeros T_rphi/T_zphi -- exactly the components a
# non-axisymmetric mode excites. Everything above drives those operators with tensors built inline
# from coordinates; these two drive them through the component fields of an unknown, which is the
# path an equation takes and the only one that can catch a mismatch between the slot layout
# define_tensor_field hands out and the one the operators assume.
# ----------------------------------------------------------------------------------------------

class _TensorUnknown(Equations):
    """A tensor unknown pinned to a prescribed value, plus scalars projected off its operators."""

    def __init__(self, value, expressions_of):
        super().__init__()
        self.value = value
        self.expressions_of = expressions_of

    def define_fields(self):
        self.define_tensor_field("sig", "C2", symmetric=False)
        for name in self.expressions_of(identity_matrix()):
            self.define_scalar_field(name, "C2")

    def define_residuals(self):
        # var("sig") rather than the components one by one: define_tensor_field registers the tensor
        # as a substituted field, and that is the form an equation writes.
        sig, sigtest = var_and_test("sig")
        self.add_residual(weak(sig - self.value, sigtest))
        for name, expression in self.expressions_of(sig).items():
            unknown, test = var_and_test(name)
            self.add_residual(weak(unknown - expression, test))


class _TensorUnknownProblem(Problem):
    def __init__(self, value, expressions_of):
        super().__init__()
        self.value = value
        self.expressions_of = expressions_of

    def define_problem(self):
        self.set_coordinate_system(AxisymmetricCoordinateSystem())
        # away from the axis, since the connection terms carry 1/r
        self.add_mesh(RectangularQuadMesh(N=2, lower_left=[1, 0]))
        self.add_equations(_TensorUnknown(self.value, self.expressions_of) @ "domain")


def _tensor_unknown_values(tmp_path, value, expressions_of, stability):
    with _TensorUnknownProblem(value, expressions_of) as problem:
        problem.set_output_directory(str(tmp_path))
        if stability:
            problem.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
        problem.initialise()
        problem.solve()
        mesh = problem.get_mesh("domain")
        indices = mesh.get_nodal_field_indices()
        names = list(expressions_of(identity_matrix()).keys())
        return {name: max(abs(node.value(indices[name])) for node in mesh.nodes()) for name in names}


def test_azimuthal_tensor_unknown_has_its_swirl_components(tmp_path):
    """The nine components exist as unknowns, where plain axisymmetry offers five.

    Counting dofs rather than inspecting names, so the test fails if a component is declared but not
    actually allocated."""
    value = dyadic(vector(1, 2, 3), vector(4, 5, 6))
    expressions_of = lambda sig: {}

    def ndof(stability):
        with _TensorUnknownProblem(value, expressions_of) as problem:
            problem.set_output_directory(str(tmp_path / ("azi" if stability else "plain")))
            if stability:
                problem.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
            problem.initialise()
            return problem.ndof(), problem.get_mesh("domain").nnode()

    plain, nnode = ndof(False)
    azimuthal, _ = ndof(True)
    assert plain == 5 * nnode                 # xx, xy, yx, yy and the azimuthal diagonal aa
    assert azimuthal == 9 * nnode             # ... plus xa, ya, ax, ay


def test_azimuthal_tensor_unknown_satisfies_the_dyadic_identity(tmp_path):
    """div(a (x) b) == div(b)*a + (b.grad)a, with the tensor carried by unknowns.

    The same identity the inline tensors above are checked against, so the reference is independent
    of the field definition being tested. Driving it through the component fields is what pins the
    slot layout: define_tensor_field puts the azimuthal direction at slot 2, and tensor_divergence
    and directional_tensor_derivative assume exactly that.
    """
    r, z = var(["coordinate_x", "coordinate_y"])
    # Every entry of dyadic(a,b) is one of r*r, z*z or r*z, i.e. inside the C2 space of the element.
    # That matters: the unknown has to equal the value EXACTLY for the residual below to be
    # identically zero rather than the projection error. A quartic dyadic leaves an O(1) remainder
    # on this mesh and says nothing about the operators.
    a = vector(r, z, r)
    b = vector(z, r, z)

    def expressions_of(sig):
        rhs = div(b) * a + directional_derivative(a, b)
        out = {"magnitude": div(sig)[0]}
        for i in range(3):
            out["dyadic_%d" % i] = div(sig)[i] - rhs[i]
        return out

    values = _tensor_unknown_values(tmp_path / "azimuthal", dyadic(a, b), expressions_of,
                                    stability=True)
    # a real magnitude, or the residuals below are trivially small
    assert values["magnitude"] > 0.5
    for i in range(3):
        assert values["dyadic_%d" % i] < 1e-8, i

    # The same run under plain axisymmetry, whose tensor field has no azimuthal off-diagonals, so it
    # cannot hold this dyadic at all. Without this the test above would pass on a field that simply
    # dropped the swirl: the residual is 1.0 and 2.0 in two of the three components there, not small.
    inherited = _tensor_unknown_values(tmp_path / "plain", dyadic(a, b), expressions_of,
                                       stability=False)
    assert max(inherited["dyadic_%d" % i] for i in range(3)) > 0.1


def test_flipped_symmetry_axis_is_refused_under_azimuthal_stability(tmp_path):
    """use_x_as_symmetry_axis belongs to plain axisymmetry alone.

    The azimuthal normal mode system spells out the (r,z,phi) layout throughout -- the I*m/r factors
    of its derivative operators, the "_phi" component of define_vector_field, the slot order of
    define_tensor_field -- so a flipped axis is not something it can honour. Two ways in, both
    refused: setting the flag on the class, and handing setup_for_stability_analysis a coordinate
    system that already carries it. The latter used to build a fresh unflipped system and drop the
    flag without a word, which changes what the base state means.
    """
    from pyoomph.expressions.coordsys import AxisymmetryBreakingCoordinateSystem

    coordsys = AxisymmetryBreakingCoordinateSystem(Expression(1))
    assert coordsys.use_x_as_symmetry_axis is False
    coordsys.use_x_as_symmetry_axis = False          # what the inherited constructor does
    with pytest.raises(RuntimeError, match="only supported in a plain"):
        coordsys.use_x_as_symmetry_axis = True

    # ... and plain axisymmetry still takes it, or the guard is too wide
    assert AxisymmetricCoordinateSystem(use_x_as_symmetry_axis=True).use_x_as_symmetry_axis is True

    class _Scalar(Equations):
        def define_fields(self):
            self.define_scalar_field("u", "C2")

        def define_residuals(self):
            u, v = var_and_test("u")
            self.add_residual(weak(partial_t(u), v) + weak(grad(u), grad(v)))

    class _FlippedProblem(Problem):
        def define_problem(self):
            self.set_coordinate_system(AxisymmetricCoordinateSystem(use_x_as_symmetry_axis=True))
            self.add_mesh(RectangularQuadMesh(N=2, lower_left=[0, 1]))
            self.add_equations(_Scalar() @ "domain")

    with pytest.raises(RuntimeError, match="use_x_as_symmetry_axis"):
        with _FlippedProblem() as problem:
            problem.set_output_directory(str(tmp_path))
            problem.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
            problem.initialise()


def test_cartesian_error_estimation_survives_the_switch_to_azimuthal_stability(tmp_path):
    """setup_for_stability_analysis builds a fresh coordinate system, so settings have to be carried.

    It used to test the OLD system for the breaking subclass before copying, i.e. it only carried the
    setting over from a system it had installed itself. A problem written with a plain
    AxisymmetricCoordinateSystem -- every real case -- silently lost it.
    """
    class _Scalar(Equations):
        def define_fields(self):
            self.define_scalar_field("u", "C2")

        def define_residuals(self):
            u, v = var_and_test("u")
            self.add_residual(weak(partial_t(u), v) + weak(grad(u), grad(v)))

    class _Prob(Problem):
        def __init__(self, cartesian_error_estimation):
            super().__init__()
            self.cartesian_error_estimation = cartesian_error_estimation

        def define_problem(self):
            self.set_coordinate_system(AxisymmetricCoordinateSystem(
                cartesian_error_estimation=self.cartesian_error_estimation))
            self.add_mesh(RectangularQuadMesh(N=2, lower_left=[1, 0]))
            self.add_equations(_Scalar() @ "domain")

    for setting in (False, True):
        with _Prob(setting) as problem:
            problem.set_output_directory(str(tmp_path / str(setting)))
            problem.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
            problem.initialise()
            assert problem.get_coordinate_system().cartesian_error_estimation is setting
