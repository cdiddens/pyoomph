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

# define_tensor_field() and the operations that act on what it produces.
#
# Tensor fields went unused for a long time -- pyoomph.equations.viscoelastic is the first consumer --
# so the paths only they reach had never been exercised. What is covered here is the combination an
# equation carrying a tensor field actually uses: material_derivative() of the tensor, on a static and
# on a genuinely moving mesh, and a SpatialErrorEstimator naming the field.
#
# One shape convention is worth knowing. Everything here is 3x3, because that is what pyoomph
# produces: vector() pads to three components and matrix() to 3x3, so grad() of a vector field is 3x3
# in every coordinate system, and define_tensor_field hands back the padded form too. A tensor
# assembled UNPADDED -- matrix(..., fill_to_max_vector_dim=False), i.e. 2x2 in 2d Cartesian -- cannot
# be passed to partial_t(): the ALE correction inside it goes through
# CartesianCoordinateSystem.directional_tensor_derivative, which allocates a 3x3 result and loops
# over range(3) whatever it is given, so it raises "matrix::operator(): index out of range". That
# limitation is left standing rather than fixed, because 3x3 is the convention everywhere else and
# the library functions used below both produce and consume that shape. Note it would bite on a
# static mesh too: the ALE correction is assembled either way and only multiplied by an
# eval_flag(moving_mesh) afterwards.

import pytest

from pyoomph import Problem, Equations, DirichletBC
from pyoomph.expressions import var, testfunction, weak, matrix, vector, material_derivative
from pyoomph.equations.ALE import LaplaceSmoothedMesh
from pyoomph.equations.generic import SpatialErrorEstimator
from pyoomph.meshes.simplemeshes import RectangularQuadMesh

#: (row, column) component names of a symmetric 2d tensor field. The off-diagonal is shared, so such
#: a field costs three unknowns per node rather than four.
ROWS = (("xx", "xy"), ("xy", "yy"))


class _TensorTransport(Equations):
    """A symmetric tensor field carried by a prescribed flow, with a relaxation term and no diffusion."""

    def __init__(self, moving_mesh=False, error_estimator=False):
        super().__init__()
        self.moving_mesh = moving_mesh
        self.error_estimator = error_estimator

    def define_fields(self):
        self.define_tensor_field("sigma", "C2", symmetric=True)
        # The wind goes in as a substituted field rather than being written inline as vector(1, 0).
        # Both are the same flow, but a literal constant vector inside a tensor-valued expression
        # triggers a pre-existing and NONDETERMINISTIC failure in residual assembly --
        # "RuntimeError: Not a 32-bit integer: <pointer-sized value>" out of _add_residual, on
        # roughly 8 runs in 10. A coordinate-dependent or field-valued wind never does. That is
        # unrelated to anything tested here.
        self.define_field_by_substitution("wind", vector(1.0, 0.0), also_on_interface=True)

    def define_residuals(self):
        sigma = matrix([[var("sigma_" + c) for c in row] for row in ROWS])
        sigma_test = matrix([[testfunction("sigma_" + c) for c in row] for row in ROWS])
        # dt(sigma) + (u.grad)sigma in one call rather than component by component. This is the path
        # pyoomph.equations.viscoelastic takes, and it reaches both partial_t() and
        # directional_tensor_derivative() with a tensor argument.
        transported = material_derivative(sigma, var("wind"), ALE="auto")
        self.add_residual(weak(transported + sigma, sigma_test))

    def define_scaling(self):
        if self.error_estimator:
            # Naming the tensor field itself, rather than its components one by one.
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
            # A mesh that really moves, so the ALE correction does not merely get assembled but
            # actually contributes: the top boundary is driven and the interior follows the smoother.
            eqs += LaplaceSmoothedMesh()
            eqs += DirichletBC(mesh_x=True, mesh_y=True) @ "bottom"
            eqs += DirichletBC(mesh_x=True, mesh_y=1 + 0.1 * var("time")) @ "top"
            eqs += DirichletBC(mesh_x=True) @ "left"
            eqs += DirichletBC(mesh_x=True) @ "right"
        self.add_equations(eqs @ "domain")


@pytest.mark.parametrize("moving_mesh", [False, True], ids=["static", "moving"])
def test_material_derivative_of_a_tensor_field(tmp_path, moving_mesh):
    """
    material_derivative() of a tensor field, which is how an equation carrying one transports it.

    Both cases matter. The static one still reaches directional_tensor_derivative during code
    generation, because partial_t builds its ALE correction whether or not the mesh moves; the moving
    one additionally makes that correction contribute to the answer.
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
    so vector_gradient used to walk off the end of the matrix. The components are now expanded into
    one criterion each, all in the same group.
    """
    with _TensorProblem(error_estimator=True) as problem:
        problem.set_output_directory(str(tmp_path))
        problem.initialise()
        problem.solve(spatial_adapt=1)
        assert problem.ndof() > 0
