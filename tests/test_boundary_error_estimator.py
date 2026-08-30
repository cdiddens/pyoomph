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

"""Z2 error estimation on meshes with a co-dimension (interfaces).

The Z2 flux recovery fits a polynomial in the first `dim` GLOBAL coordinates, which parametrises an
interface only if the interface happens to be a graph over those axes. A boundary at x=const makes
the normal matrix exactly singular. LagrZ2ErrorEstimator now fits in a patch-local tangent frame
instead; `use_local_recovery_frame_in_codim = False` restores the old behaviour, which is what the
negative tests here switch on to show that the frame is doing the work.

See dev_docs/spatial_error_estimators.md.
"""

import math

import numpy
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.meshes.simplemeshes import CircularMesh, CuboidBrickMesh


class RotatedRectMesh(MeshTemplate):
    """A unit-square quad mesh rotated by ``angle``, with the usual four boundary names.

    Node and element creation order does not depend on the angle, so element i of a rotated mesh is
    the image of element i of the unrotated one. That is what lets the error vectors of two
    rotations be compared entry by entry.
    """

    def __init__(self, N: int = 8, angle: float = 0.0):
        super().__init__()
        self.N = N
        self.angle = angle

    def define_geometry(self):
        N, c, s = self.N, math.cos(self.angle), math.sin(self.angle)
        dom = self.new_domain("domain")

        def node(ix, iy):
            x, y = ix / N, iy / N
            return self.add_node_unique(c * x - s * y, s * x + c * y)

        for ix in range(N):
            for iy in range(N):
                n00, n10 = node(ix, iy), node(ix + 1, iy)
                n01, n11 = node(ix, iy + 1), node(ix + 1, iy + 1)
                dom.add_quad_2d_C1(n00, n10, n01, n11)
                if ix == 0:
                    self.add_facet_to_boundary("left", [n00, n01])
                if ix == N - 1:
                    self.add_facet_to_boundary("right", [n10, n11])
                if iy == 0:
                    self.add_facet_to_boundary("bottom", [n00, n10])
                if iy == N - 1:
                    self.add_facet_to_boundary("top", [n01, n11])


class BoundaryFront(InterfaceEquations):
    """A C2 field on the boundary, L2-projected onto a sharp front in the boundary's arclength.

    The front is written in the boundary's own tangent direction, so the field rotates with the mesh
    and a correct recovery must rotate with it too. A projection rather than a solve keeps the test
    on the estimator instead of on whatever an interface PDE would do.
    """

    def __init__(self, tangent, offset: float = 0.5, width: float = 0.05):
        super().__init__()
        self.tangent, self.offset, self.width = tangent, offset, width

    def define_fields(self):
        self.define_scalar_field("v", "C2")

    def define_residuals(self):
        x = var("coordinate")
        arclen = self.tangent[0] * x[0] + self.tangent[1] * x[1] - self.offset
        self.add_residual(weak(var("v") - tanh(arclen / self.width), testfunction("v")))

    def define_error_estimators(self):
        self.add_spatial_error_estimator(grad(var("v"), nondim=True))


class RotatedProblem(Problem):
    def __init__(self, angle: float, boundary: str):
        super().__init__()
        self.angle, self.boundary = angle, boundary

    def define_problem(self):
        c, s = math.cos(self.angle), math.sin(self.angle)
        tangent = {"bottom": (c, s), "top": (c, s),
                   "left": (-s, c), "right": (-s, c)}[self.boundary]
        self += RotatedRectMesh(N=8, angle=self.angle)
        eqs = PoissonEquation(source=1) + DirichletBC(u=0) @ "bottom"
        eqs += BoundaryFront(tangent) @ self.boundary
        self += eqs @ "domain"


def _interface_errors(angle: float, boundary: str, frame: bool = True):
    with RotatedProblem(angle, boundary) as problem:
        problem.initial_adaption_steps = 0
        problem.solve()
        imesh = problem.get_mesh("domain/" + boundary)
        imesh._error_estimator.use_local_recovery_frame_in_codim = frame
        imesh._enable_adaptation()
        try:
            return numpy.array(imesh.get_elemental_errors())
        finally:
            imesh._disable_adaptation()


def test_vertical_boundary_is_estimable():
    """x=const is the case the global-coordinate recovery cannot parametrise at all."""
    errs = _interface_errors(0.0, "left")
    assert numpy.all(numpy.isfinite(errs))
    assert errs.max() > 0
    # The front sits at the middle of the boundary, so that is where the error must be.
    assert numpy.argmax(errs) in (len(errs) // 2 - 1, len(errs) // 2)


def test_vertical_boundary_needs_the_local_frame():
    """Negative test: without the frame this is the "Singular Matrix" throw it always was."""
    with pytest.raises(RuntimeError):
        _interface_errors(0.0, "left", frame=False)


def test_local_frame_reproduces_the_global_recovery():
    """On a boundary the old code COULD handle, the frame must not change the answer.

    A complete polynomial space is invariant under affine maps, so this is exact mathematics, not a
    tolerance: fitting in the patch frame recovers the same field as fitting in global coordinates.
    """
    with_frame = _interface_errors(0.0, "bottom", frame=True)
    without = _interface_errors(0.0, "bottom", frame=False)
    assert numpy.allclose(with_frame, without, rtol=1e-9, atol=0.0)


@pytest.mark.parametrize("degrees", [30, 90, 137])
def test_rotation_invariance(degrees):
    """Rotating the mesh must not change the estimated error.

    This is the property the global-axis recovery structurally lacks, and it fails for the old code
    at every angle where the boundary stops being a graph over x.
    """
    reference = _interface_errors(0.0, "bottom")
    rotated = _interface_errors(math.radians(degrees), "bottom")
    assert numpy.allclose(rotated, reference, rtol=1e-8, atol=0.0)


def test_all_four_boundaries_agree():
    """The same front on each of the four edges of the square must give the same error vector."""
    reference = _interface_errors(0.0, "bottom")
    for boundary in ["top", "left", "right"]:
        errs = _interface_errors(0.0, boundary)
        assert numpy.allclose(errs, reference, rtol=1e-8, atol=0.0), boundary


class CircumferenceProblem(Problem):
    def define_problem(self):
        self += CircularMesh(radius=1, segments=["NE", "NW", "SW", "SE"])
        eqs = PoissonEquation(source=1) + DirichletBC(u=0) @ "circumference"
        # tangent (1,0): the front is the x=0 meridian, so it cuts the closed curve twice.
        eqs += BoundaryFront((1.0, 0.0), offset=0.0) @ "circumference"
        self += eqs @ "domain"


def test_closed_curved_interface():
    """A closed curve has no global axis that parametrises it, at any rotation."""
    with CircumferenceProblem() as problem:
        problem.initial_adaption_steps = 0
        problem.solve()
        imesh = problem.get_mesh("domain/circumference")
        imesh._enable_adaptation()
        errs = numpy.array(imesh.get_elemental_errors())
        imesh._disable_adaptation()
    assert numpy.all(numpy.isfinite(errs))
    assert errs.min() > 0
    # Localised, not uniform: the front crosses the circle at two points and the estimator has to
    # see that. A degenerate patch would show up here as either a flat vector or a blown-up one.
    assert 2.0 < errs.max() / errs.min() < 1e3


class Box3dProblem(Problem):
    def __init__(self, boundary: str):
        super().__init__()
        self.boundary = boundary

    def define_problem(self):
        self += CuboidBrickMesh(N=4)
        eqs = PoissonEquation(source=1) + DirichletBC(u=0) @ "bottom"
        # A front in y, which varies along both the x=const and the z=const faces.
        eqs += BoundaryFront((0.0, 1.0)) @ self.boundary
        self += eqs @ "domain"


def _face_errors(boundary: str, frame: bool = True):
    with Box3dProblem(boundary) as problem:
        problem.initial_adaption_steps = 0
        problem.solve()
        imesh = problem.get_mesh("domain/" + boundary)
        imesh._error_estimator.use_local_recovery_frame_in_codim = frame
        imesh._enable_adaptation()
        try:
            return numpy.array(imesh.get_elemental_errors())
        finally:
            imesh._disable_adaptation()


def test_3d_face_at_constant_x():
    """A 2D patch embedded in 3D: only the z=const face is a graph over (x,y)."""
    errs = _face_errors("left")
    assert numpy.all(numpy.isfinite(errs))
    assert errs.max() > 0
    # "front" (z=const) is the orientation the old code could already do; both faces contain the
    # direction the front varies in, so they must see the same errors. Sorted, because the face
    # element enumeration follows the bulk element order and so differs between the two faces.
    assert numpy.allclose(numpy.sort(errs), numpy.sort(_face_errors("front")),
                          rtol=1e-8, atol=0.0)


def test_3d_face_at_constant_x_needs_the_local_frame():
    with pytest.raises(RuntimeError):
        _face_errors("left", frame=False)


class BulkProblem(Problem):
    def define_problem(self):
        self += RotatedRectMesh(N=8, angle=0.0)
        x = var("coordinate")
        source = exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.01)
        eqs = PoissonEquation(source=source) + DirichletBC(u=0) @ "bottom"
        eqs += SpatialErrorEstimator(u=1)
        self += eqs @ "domain"


def test_codim0_is_unaffected():
    """The frame is off for bulk meshes by default, and would not change them if it were on.

    The default matters because every baselined refinement decision in the test suite depends on the
    exact bulk error values; the agreement matters because it is the evidence that the frame is a
    change of basis and not a change of estimator.
    """
    def bulk_errors(force_frame):
        with BulkProblem() as problem:
            problem.initial_adaption_steps = 0
            problem.solve()
            mesh = problem.get_mesh("domain")
            assert mesh._error_estimator.force_local_recovery_frame is False
            mesh._error_estimator.force_local_recovery_frame = force_frame
            return numpy.array(mesh.get_elemental_errors())

    plain = bulk_errors(False)
    framed = bulk_errors(True)
    assert plain.max() > 0
    assert numpy.allclose(framed, plain, rtol=0.0, atol=1e-8 * plain.max())


def test_interface_inherits_error_thresholds():
    """An interface mesh used to run on oomph-lib's own defaults instead of the Problem's."""
    with RotatedProblem(0.0, "bottom") as problem:
        problem.initial_adaption_steps = 0
        problem.solve()
        bulk = problem.get_mesh("domain")
        imesh = problem.get_mesh("domain/bottom")
        assert imesh.min_permitted_error == bulk.min_permitted_error
        assert imesh.max_permitted_error == bulk.max_permitted_error
