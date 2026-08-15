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

# Second spatial derivatives of the shape functions, i.e. grad(grad(u)), div(grad(u)) and
# partial_x(u,2).
#
# The trick used throughout is to pick a test function that the space represents EXACTLY, so that the
# expected Hessian is known to round-off rather than only up to an interpolation error. On a
# quadrilateral whose geometry map is bilinear in the local coordinates, any function of degree <= 2
# in each local coordinate (x*y, x^2, ...) lies in the biquadratic C2 space; on a simplex with an
# affine map, any quadratic does. Nodal values are therefore written directly instead of solving.
#
# Note which distortion is admissible. A globally BILINEAR node map keeps every element's geometry map
# bilinear, so the C2 midside and centre nodes stay consistent with it and x*y remains representable.
# Moving the midside nodes freely instead makes the map biquadratic, x*y is then degree 4 in s and no
# longer in the space, and the "exact" expectation is simply wrong -- that is an interpolation error,
# not a bug. Both distortions appear below, with the appropriate test function for each.
#
# The linear-function case on a genuinely curved element is the sharp test of the X_{k,ab} term: for
# w with nodal values equal to a coordinate, the interpolant IS that coordinate, so its discrete
# Hessian must vanish exactly. That only happens because the K K psi_,ab term and the
# -(dpsi/dx_k) K K X_{k,ab} term cancel; dropping the latter leaves K K x_{,ab} != 0.

import math

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.expressions import FiniteElementSpaceEnum
from pyoomph.meshes.simplemeshes import (RectangularQuadMesh, CircularMesh, LineMesh,
                                         CuboidBrickMesh)


class _Carrier(Equations):
    """A field whose nodal values are set by hand; the residual only exists to give it equations."""

    def __init__(self, space: "FiniteElementSpaceEnum" = "C2"):
        super().__init__()
        self.space = space

    def define_fields(self):
        self.define_scalar_field("w", self.space)

    def define_residuals(self):
        w, q = var_and_test("w")
        self.add_residual(weak(w, q))


def _set_nodal(mesh, fn, name="w"):
    idx = mesh.element_pt(0).get_code_instance().get_nodal_field_indices()[name]
    for n in mesh.nodes():
        n.set_value(idx, fn(*[n.x(i) for i in range(n.ndim())]))


def _bilinear_distort(mesh):
    """A globally bilinear node map: keeps each element's geometry map bilinear, but makes
    d2x/(ds0 ds1) nonzero so the X_{k,ab} correction is active."""
    for n in mesh.nodes():
        a, b = n.x(0), n.x(1)
        n.set_x(0, a + 0.30 * a * b)
        n.set_x(1, b - 0.20 * a * b)


def _curved_distort(mesh):
    """Moves every node, midside ones included, so the geometry map becomes genuinely biquadratic."""
    for n in mesh.nodes():
        a, b = n.x(0), n.x(1)
        n.set_x(0, a + 0.23 * a * (1 - a) * (1 + 2 * b))
        n.set_x(1, b + 0.17 * b * (1 - b) * (1 + 3 * a))


def _hessian_observables(dim):
    w = var("w")
    H = grad(grad(w))
    obs = {"lap": div(grad(w)), "vol": 1}
    for i in range(dim):
        for j in range(dim):
            obs["h%d%d" % (i, j)] = H[i, j]
    return IntegralObservables(**obs)


def _measure(mesh_factory, fn, dim, space="C2", distort=None, domain="domain"):
    class P(Problem):
        def define_problem(self):
            self.add_mesh(mesh_factory())
            self.add_equations((_Carrier(space) + _hessian_observables(dim)) @ domain)

    with P() as p:
        p.initialise()
        m = p.get_mesh(domain)
        if distort is not None:
            distort(m)
        _set_nodal(m, fn)
        o = m.evaluate_all_observables()
        vol = float(o["vol"])
        res = {k: float(o[k]) / vol for k in o if k != "vol"}
        return res


def _check(res, dim, exact_h, exact_lap, tol=1e-9):
    for i in range(dim):
        for j in range(dim):
            got = res["h%d%d" % (i, j)]
            assert abs(got - exact_h[i][j]) < tol, \
                "grad(grad(w))[%d,%d] = %.16g, expected %.16g" % (i, j, got, exact_h[i][j])
    assert abs(res["lap"] - exact_lap) < tol, \
        "div(grad(w)) = %.16g, expected %.16g" % (res["lap"], exact_lap)


# ---------------------------------------------------------------------------------------------
# 1d
# ---------------------------------------------------------------------------------------------

def test_second_derivative_line_C2():
    res = _measure(lambda: LineMesh(N=5), lambda x: 3 * x * x - 2 * x, 1)
    _check(res, 1, [[6.0]], 6.0)


def test_partial_x_order_two_matches_grad_grad():
    """partial_x(w,2) is sugar over nested diff and must land on the same machinery."""

    class P(Problem):
        def define_problem(self):
            self.add_mesh(LineMesh(N=5))
            w = var("w")
            eqs = _Carrier() + IntegralObservables(a=partial_x(w, 2), b=grad(grad(w))[0, 0], vol=1)
            self.add_equations(eqs @ "domain")

    with P() as p:
        p.initialise()
        m = p.get_mesh("domain")
        _set_nodal(m, lambda x: 3 * x * x - 2 * x)
        o = m.evaluate_all_observables()
        vol = float(o["vol"])
        assert abs(float(o["a"]) / vol - 6.0) < 1e-9
        assert abs(float(o["a"]) - float(o["b"])) < 1e-12


# ---------------------------------------------------------------------------------------------
# 2d quadrilaterals
# ---------------------------------------------------------------------------------------------

def test_second_derivative_quad_uniform():
    res = _measure(lambda: RectangularQuadMesh(N=4), lambda x, y: x * x + 3 * x * y - 2 * y * y, 2)
    _check(res, 2, [[2.0, 3.0], [3.0, -4.0]], -2.0)


def test_second_derivative_quad_bilinearly_distorted():
    """The discriminating case: the geometry map has a nonzero mixed second derivative, so the
    X_{k,ab} correction contributes, yet x*y is still exactly representable."""
    for fn, hess, lap in [
        (lambda x, y: x * y, [[0.0, 1.0], [1.0, 0.0]], 0.0),
        (lambda x, y: x * x, [[2.0, 0.0], [0.0, 0.0]], 2.0),
        (lambda x, y: y * y, [[0.0, 0.0], [0.0, 2.0]], 2.0),
        (lambda x, y: 3 * x * y - 2 * y * y, [[0.0, 3.0], [3.0, -4.0]], -4.0),
    ]:
        res = _measure(lambda: RectangularQuadMesh(N=3), fn, 2, distort=_bilinear_distort)
        _check(res, 2, hess, lap)


def test_second_derivative_vanishes_for_coordinates_on_curved_mesh():
    """Sharp test of the X_{k,ab} geometry term. On a genuinely curved (biquadratic) element the
    interpolant of a coordinate is that coordinate, so its discrete Hessian must be exactly zero -
    which requires the two terms of the transform to cancel."""
    for fn in (lambda x, y: x, lambda x, y: y, lambda x, y: 2 * x + 5 * y):
        res = _measure(lambda: RectangularQuadMesh(N=3), fn, 2, distort=_curved_distort)
        _check(res, 2, [[0.0, 0.0], [0.0, 0.0]], 0.0, tol=1e-11)


def test_second_derivative_quad_C1():
    """A Q1 space: the pure second derivatives vanish, the mixed one does not."""
    res = _measure(lambda: RectangularQuadMesh(N=4), lambda x, y: x * y, 2, space="C1")
    _check(res, 2, [[0.0, 1.0], [1.0, 0.0]], 0.0)


# ---------------------------------------------------------------------------------------------
# 2d triangles
# ---------------------------------------------------------------------------------------------

def test_second_derivative_triangles():
    def mesh():
        return RectangularQuadMesh(N=4, split_in_tris="crossed")

    res = _measure(mesh, lambda x, y: x * x + 3 * x * y - 2 * y * y, 2)
    _check(res, 2, [[2.0, 3.0], [3.0, -4.0]], -2.0)


# ---------------------------------------------------------------------------------------------
# 3d
# ---------------------------------------------------------------------------------------------

def test_second_derivative_brick():
    res = _measure(lambda: CuboidBrickMesh(N=3),
                   lambda x, y, z: x * x + 3 * x * y - 2 * y * z + z * z, 3)
    _check(res, 3, [[2.0, 3.0, 0.0], [3.0, 0.0, -2.0], [0.0, -2.0, 2.0]], 4.0)


# ---------------------------------------------------------------------------------------------
# Interfaces: div(grad(.)) of a surface field is the Laplace-Beltrami operator
# ---------------------------------------------------------------------------------------------

class _SurfCarrier(Equations):
    def define_fields(self):
        self.define_scalar_field("w", "C2")

    def define_residuals(self):
        w, q = var_and_test("w")
        self.add_residual(weak(w, q))


class _Dummy(Equations):
    def define_fields(self):
        self.define_scalar_field("dummy", "C2")

    def define_residuals(self):
        d, q = var_and_test("dummy")
        self.add_residual(weak(d, q))


def test_surface_laplacian_on_a_straight_interface():
    """1d interface embedded in 2d: el_dim < nodal_dim, so the metric route uses a pseudo-inverse.
    Along the straight edge y=0 the surface Laplacian of x^2 is exactly 2."""

    class P(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=4))
            self.add_equations(_Dummy() @ "domain")
            w = var("w")
            self.add_equations((_SurfCarrier() +
                                IntegralObservables(lap=div(grad(w)), length=1)) @ "domain/bottom")

    with P() as p:
        p.initialise()
        m = p.get_mesh("domain/bottom")
        _set_nodal(m, lambda x, y: x * x)
        o = m.evaluate_all_observables()
        got = float(o["lap"]) / float(o["length"])
        assert abs(got - 2.0) < 1e-10, "surface Laplacian %.16g, expected 2" % got


def test_laplace_beltrami_on_a_curved_interface():
    """On the unit circle the Laplace-Beltrami operator of x is -x. This needs the curvature part of
    the transform - the codimension-free formula would get it wrong - and converges at the second
    order set by the isoparametric (quadratic) approximation of the circle."""

    def run(nref):
        class P(Problem):
            def define_problem(self):
                self.add_mesh(CircularMesh(radius=1.0))
                self.max_refinement_level = 8
                self.add_equations(_Dummy() @ "domain")
                w, x = var("w"), var("coordinate_x")
                self.add_equations((_SurfCarrier() +
                                    IntegralObservables(err=(div(grad(w)) + x) ** 2,
                                                        ref=x ** 2)) @ "domain/circumference")

        with P() as p:
            p.initialise()
            for _ in range(nref):
                p.refine_uniformly()
            m = p.get_mesh("domain/circumference")
            _set_nodal(m, lambda x, y: x)
            o = m.evaluate_all_observables()
            return math.sqrt(float(o["err"]) / float(o["ref"]))

    errs = [run(k) for k in range(3)]
    assert errs[-1] < 1e-2, "Laplace-Beltrami error did not get small: %r" % (errs,)
    for a, b in zip(errs, errs[1:]):
        order = math.log(a / b, 2)
        assert order > 1.7, "expected ~2nd order convergence, got %.2f (%r)" % (order, errs)


# ---------------------------------------------------------------------------------------------
# Moving mesh: the position columns of the Jacobian come from d_d2x_shape_dcoord
# ---------------------------------------------------------------------------------------------

def test_moving_mesh_position_jacobian_of_a_strong_laplacian():
    """debug_jacobian_by_fd_epsilon compares the analytic elemental Jacobian against a finite
    difference one and raises on a mismatch. The mesh is distorted on purpose: on a uniform
    Cartesian mesh enough terms vanish to hide a sign error."""
    from pyoomph.equations.ALE import LaplaceSmoothedMesh

    class Eq(Equations):
        def define_fields(self):
            self.define_scalar_field("u", "C2")

        def define_residuals(self):
            u, v = var_and_test("u")
            self.add_residual(weak(div(grad(u)), v) + weak(1, v))

    class P(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=3))
            eqs = Eq() + LaplaceSmoothedMesh()
            eqs += DirichletBC(u=0) @ "left" + DirichletBC(u=0) @ "right"
            for b in ("top", "bottom", "left", "right"):
                eqs += DirichletBC(mesh_x=True, mesh_y=True) @ b
            self.add_equations(eqs @ "domain")
            self.debug_jacobian_by_fd_epsilon = 1e-7

    import io
    import contextlib
    buf = io.StringIO()
    with P() as p:
        p.initialise()
        _bilinear_distort(p.get_mesh("domain"))
        # Assembling is enough to trigger the comparison; the system itself need not converge.
        # A mismatch is reported on stdout (stop_on_jacobian_difference defaults to false), so the
        # output has to be inspected rather than relying on an exception.
        with contextlib.redirect_stdout(buf):
            p.assemble_jacobian(with_residual=False)
    out = buf.getvalue()
    assert "DIFFERENCES IN JACOBIAN" not in out, \
        "analytic and finite-difference Jacobians disagree:\n" + out[:4000]


# ---------------------------------------------------------------------------------------------
# Things that must be refused rather than answered wrongly
# ---------------------------------------------------------------------------------------------

def test_third_derivative_is_refused():
    class Eq(Equations):
        def define_fields(self):
            self.define_scalar_field("u", "C2")

        def define_residuals(self):
            u, v = var_and_test("u")
            self.add_residual(weak(partial_x(u, 3), v))

    class P(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=2))
            self.add_equations(Eq() @ "domain")

    import pytest
    with pytest.raises(Exception, match="third order"):
        with P() as p:
            p.initialise()


def test_lagrangian_second_derivative_is_refused():
    class Eq(Equations):
        def define_fields(self):
            self.define_scalar_field("u", "C2")

        def define_residuals(self):
            u, v = var_and_test("u")
            self.add_residual(weak(div(grad(grad(u, lagrangian=True), lagrangian=True)), v))

    class P(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=2))
            self.add_equations(Eq() @ "domain")

    import pytest
    with pytest.raises(Exception):
        with P() as p:
            p.initialise()
