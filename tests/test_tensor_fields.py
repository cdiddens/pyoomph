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

# define_tensor_field() and the operations that act on what it produces.
#
# Tensor fields went unused for a long time -- pyoomph.equations.viscoelastic is the first consumer
# -- so the paths that only they reach had never been exercised, and two of them assumed a shape they
# do not have. Both are fixed; these are the regression tests.
#
# Note the sizes involved. A symmetric tensor field in 2d Cartesian coordinates gives a 2x2 block
# (three unknowns per node, since the off-diagonal is shared), while pyoomph pads vectors to three
# components and matrices to 3x3 almost everywhere else. It is that mismatch the two defects came
# from, so the tests deliberately use the 2x2 case.

import pytest

from pyoomph import Problem, Equations, DirichletBC
from pyoomph.expressions import var, testfunction, partial_t, weak, grad, dot, matrix, vector
from pyoomph.equations.ALE import LaplaceSmoothedMesh
from pyoomph.equations.generic import SpatialErrorEstimator
from pyoomph.meshes.simplemeshes import RectangularQuadMesh


class _TensorTransport(Equations):
    """A symmetric tensor field carried by a prescribed flow, with no diffusion."""

    def __init__(self, moving_mesh=False, error_estimator=False):
        super().__init__()
        self.moving_mesh = moving_mesh
        self.error_estimator = error_estimator

    def define_fields(self):
        self.define_tensor_field("sigma", "C2", symmetric=True)
        # The wind goes in as a substituted field rather than being written inline as
        # vector(1, 0). Both are the same flow, but a literal constant vector inside
        # dot(..., grad(...)) inside a matrix triggers a pre-existing and NONDETERMINISTIC failure
        # in residual assembly -- "RuntimeError: Not a 32-bit integer: <pointer-sized value>" from
        # _add_residual, on roughly 8 runs in 10, at both 2x2 and 3x3. A coordinate-dependent or
        # field-valued wind never does. That is unrelated to anything tested here, and
        # pyoomph.equations.viscoelastic is not exposed to it because its wind is var("velocity").
        self.define_field_by_substitution("wind", vector(1.0, 0.0), also_on_interface=True)

    def define_residuals(self):
        # Assembled at 2x2 rather than read back through var("sigma"), which returns the tensor
        # padded to 3x3. The unpadded form is what a caller writes when the terms have to line up
        # with something else of the same shape, and it is the shape that used to break partial_t.
        rows = (("xx", "xy"), ("xy", "yy"))
        sigma = matrix([[var("sigma_" + c) for c in row] for row in rows], fill_to_max_vector_dim=False)
        sigma_test = matrix([[testfunction("sigma_" + c) for c in row] for row in rows],
                            fill_to_max_vector_dim=False)
        wind = var("wind")
        # partial_t of the tensor itself, which is the operation that used to fail: its ALE
        # correction is assembled whether or not the mesh moves, and that correction looped over a
        # hardcoded 3x3.
        advection = matrix([[dot(wind, grad(var("sigma_" + c))) for c in row] for row in rows],
                           fill_to_max_vector_dim=False)
        self.add_residual(weak(partial_t(sigma) + advection + sigma, sigma_test))

    def define_scaling(self):
        if self.error_estimator:
            # Naming the tensor field itself, which is the second thing that used to fail.
            self += SpatialErrorEstimator(sigma=1)


class _TensorProblem(Problem):
    def __init__(self, moving_mesh=False, error_estimator=False):
        super().__init__()
        self.moving_mesh = moving_mesh
        self.error_estimator = error_estimator

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=3))
        eqs = _TensorTransport(self.moving_mesh, self.error_estimator)
        if self.moving_mesh:
            # A real moving mesh, so the ALE correction is not merely assembled but actually
            # contributes: the top boundary is driven and the rest of the mesh follows the smoother.
            eqs += LaplaceSmoothedMesh()
            eqs += DirichletBC(mesh_x=True, mesh_y=True) @ "bottom"
            eqs += DirichletBC(mesh_x=True, mesh_y=1 + 0.1 * var("time")) @ "top"
            eqs += DirichletBC(mesh_x=True) @ "left"
            eqs += DirichletBC(mesh_x=True) @ "right"
        self.add_equations(eqs @ "domain")


@pytest.mark.parametrize("moving_mesh", [False, True], ids=["static", "moving"])
def test_partial_t_of_a_two_by_two_tensor_field(tmp_path, moving_mesh):
    """
    partial_t() of a tensor field that is not 3x3.

    BaseCoordinateSystem.directional_tensor_derivative used to allocate a 3x3 result and loop over
    range(3) regardless of the tensor it was given, so this raised
    "matrix::operator(): index out of range" during code generation. The static case matters as much
    as the moving one: partial_t builds the ALE correction either way and only multiplies it by an
    eval_flag(moving_mesh) afterwards, so a fixed mesh does not avoid the code path.
    """
    with _TensorProblem(moving_mesh=moving_mesh) as problem:
        problem.set_output_directory(str(tmp_path))
        problem.initialise()
        problem.run(0.2, startstep=0.1, maxstep=0.1, temporal_error=None, outstep=False)
        assert problem.ndof() > 0


def test_spatial_error_estimator_accepts_a_tensor_field(tmp_path):
    """
    SpatialErrorEstimator(sigma=1) where sigma is a tensor field.

    It takes grad() of whatever it is handed, and grad() of a tensor field is not a vector gradient,
    so vector_gradient walked off the end of the matrix. The components are now expanded into one
    criterion each, all in the same group.
    """
    with _TensorProblem(error_estimator=True) as problem:
        problem.set_output_directory(str(tmp_path))
        problem.initialise()
        problem.solve(spatial_adapt=1)
        assert problem.ndof() > 0
