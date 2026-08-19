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
    """equation_compilation_flags.debug_jacobian_epsilon compares the analytic elemental Jacobian against a finite
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
            self.equation_compilation_flags.debug_jacobian_epsilon = 1e-7

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
# Moving mesh, analytic Hessian: d2_d2x2_shape_dcoord
#
# Problem.debug_analytic_hessian_by_fd() compares the analytic Hessian-vector products against a
# central difference of the analytic Jacobian. Its noise floor here is around 1e-8; zeroing the
# second-derivative Hessian term on purpose takes the discrepancy to ~1.9, so the check is sensitive
# to the thing it is meant to test by some eight orders of magnitude.
#
# The residual has to be nonlinear in u AND contain a second derivative: for the plain strong-form
# Laplacian the residual is linear, so its Hessian vanishes and nothing is tested. The mesh has to be
# distorted for the same reason as everywhere else here.
# ---------------------------------------------------------------------------------------------

def _hessian_fd_check(mesh_factory, boundaries, dim):
    from pyoomph.equations.ALE import LaplaceSmoothedMesh

    class Eq(Equations):
        def define_fields(self):
            self.define_scalar_field("u", "C2")

        def define_residuals(self):
            u, v = var_and_test("u")
            self.add_residual(weak((1 + u ** 2) * div(grad(u)), v) + weak(1, v))

    class P(Problem):
        def define_problem(self):
            self.add_mesh(mesh_factory())
            eqs = Eq() + LaplaceSmoothedMesh()
            pinned = {"mesh_x": True, "mesh_y": True}
            if dim == 3:
                pinned["mesh_z"] = True
            for b in boundaries:
                eqs += DirichletBC(**pinned) @ b
            eqs += DirichletBC(u=0) @ "left" + DirichletBC(u=0) @ "right"
            self.add_equations(eqs @ "domain")

    with P() as p:
        p.setup_for_stability_analysis(analytic_hessian=True)
        p.initialise()
        m = p.get_mesh("domain")
        for n in m.nodes():
            a, b = n.x(0), n.x(1)
            n.set_x(0, a + 0.30 * a * b)
            n.set_x(1, b - 0.20 * a * b)
        for n in m.nodes():
            n.set_value(0, 0.3 * n.x(0) + 0.7 * n.x(1) ** 2 + 0.2)
        return p.debug_analytic_hessian_by_fd(epsilon=1e-4)


def test_moving_mesh_hessian_quads():
    _hessian_fd_check(lambda: RectangularQuadMesh(N=2), ("top", "bottom", "left", "right"), 2)


def test_moving_mesh_hessian_triangles():
    _hessian_fd_check(lambda: RectangularQuadMesh(N=2, split_in_tris="crossed"),
                      ("top", "bottom", "left", "right"), 2)


def test_moving_mesh_hessian_bricks():
    _hessian_fd_check(lambda: CuboidBrickMesh(N=2),
                      ("top", "bottom", "left", "right", "front", "back"), 3)


# ---------------------------------------------------------------------------------------------
# The 3d element families, through the full pipeline rather than only at the shape-function level.
#
# The local second derivatives of every element type are finite-differenced against dshape_local
# separately (that is where the wedge/pyramid D2Jet arithmetic and the MINI bubbles are pinned down);
# what these check is that they survive the transform, the buffers and the code generator.
#
# A linear function is exact in every one of these spaces, wedges and the rational pyramid included,
# so its discrete Hessian must vanish - which is again the X_{k,ab} cancellation, and on the pyramid
# the geometry map is rational so that term is far from trivial. The quadratic needs C2.
# ---------------------------------------------------------------------------------------------

def _cube_mesh_class(kind):
    import os
    import sys
    testdir = os.path.dirname(os.path.abspath(__file__))
    if testdir not in sys.path:
        sys.path.insert(0, testdir)
    if kind == "tet":
        from test_tet_refinement import TetCubeMesh
        return TetCubeMesh
    if kind == "wedge":
        from test_wedge_refinement import WedgeCubeMesh
        return WedgeCubeMesh
    from test_pyramid_refinement import PyramidCubeMesh
    return PyramidCubeMesh


def _measure_3d(kind, space, fn):
    cls = _cube_mesh_class(kind)

    class P(Problem):
        def define_problem(self):
            self += cls(N=2)
            self.add_equations((_Carrier(space) + _hessian_observables(3)) @ "domain")

    with P() as p:
        p.initialise()
        m = p.get_mesh("domain")
        _set_nodal(m, fn)
        o = m.evaluate_all_observables()
        vol = float(o["vol"])
        return {k: float(o[k]) / vol for k in o if k != "vol"}


import pytest


@pytest.mark.parametrize("kind", ["tet", "wedge", "pyramid"])
@pytest.mark.parametrize("space", ["C1", "C2"])
def test_3d_families_reproduce_a_linear_function(kind, space):
    res = _measure_3d(kind, space, lambda x, y, z: 0.3 * x - 0.7 * y + 1.1 * z + 0.4)
    _check(res, 3, [[0.0] * 3] * 3, 0.0, tol=1e-10)


@pytest.mark.parametrize("kind", ["tet", "wedge", "pyramid"])
def test_3d_families_reproduce_a_quadratic(kind):
    res = _measure_3d(kind, "C2", lambda x, y, z: x * x + 3 * x * y - 2 * y * y + z * z)
    _check(res, 3, [[2.0, 3.0, 0.0], [3.0, -4.0, 0.0], [0.0, 0.0, 2.0]], 0.0)


# ---------------------------------------------------------------------------------------------
# Coordinate systems other than plain Cartesian
#
# grad/div bottom out in diff(., coordinate_x) inside the coordinate system, and the metric factors
# are applied BEFORE the second diff, so the product rule is supposed to generate the extra terms by
# itself. In axisymmetric coordinates that means div(grad(u)) = u_rr + u_zz + u_r/r and
# grad(grad(u))[2,2] = u_r/r. The mesh keeps away from r=0, where the symbolic u_r/r would be
# evaluated as 0/0 (Gauss points never land exactly on the axis, but there is no reason to rely on
# that here).
# ---------------------------------------------------------------------------------------------

def _axisym_measure(fn, distort):
    class P(Problem):
        def define_problem(self):
            self.set_coordinate_system("axisymmetric")
            self.add_mesh(RectangularQuadMesh(N=3, size=[1, 1], lower_left=[1, 0]))
            w = var("w")
            H = grad(grad(w))
            eqs = _Carrier() + IntegralObservables(lap=div(grad(w)), h00=H[0, 0], h11=H[1, 1],
                                                  h22=H[2, 2], vol=1)
            self.add_equations(eqs @ "domain")

    with P() as p:
        p.initialise()
        m = p.get_mesh("domain")
        if distort:
            for n in m.nodes():
                a, b = n.x(0), n.x(1)
                n.set_x(0, a + 0.20 * (a - 1) * b)
                n.set_x(1, b - 0.15 * (a - 1) * b)
        _set_nodal(m, fn)
        o = m.evaluate_all_observables()
        v = float(o["vol"])
        return {k: float(o[k]) / v for k in ("lap", "h00", "h11", "h22")}


def test_axisymmetric_laplacian():
    for distort in (False, True):
        # w = r^2 : u_rr=2, u_zz=0, u_r/r=2  ->  lap 4
        r = _axisym_measure(lambda x, y: x * x, distort)
        assert abs(r["lap"] - 4.0) < 1e-9 and abs(r["h00"] - 2.0) < 1e-9
        assert abs(r["h11"]) < 1e-9 and abs(r["h22"] - 2.0) < 1e-9
        # w = z^2 : only u_zz
        r = _axisym_measure(lambda x, y: y * y, distort)
        assert abs(r["lap"] - 2.0) < 1e-9 and abs(r["h11"] - 2.0) < 1e-9
        assert abs(r["h00"]) < 1e-9 and abs(r["h22"]) < 1e-9
        # a constant has no derivatives at all, metric terms included
        r = _axisym_measure(lambda x, y: 1.0 + 0.0 * x, distort)
        assert max(abs(v) for v in r.values()) < 1e-9


def test_axisymmetric_hoop_term_is_the_whole_laplacian_for_r_times_z():
    """For w = r*z both u_rr and u_zz vanish, so the Laplacian is exactly the hoop term u_r/r, i.e.
    grad(grad(w))[2,2]. On the undistorted mesh the r-weighted average of z/r over
    r in [1,2], z in [0,1] is (1*1/2)/(3/2) = 1/3."""
    r = _axisym_measure(lambda x, y: x * y, False)
    assert abs(r["lap"] - r["h22"]) < 1e-12
    assert abs(r["h00"]) < 1e-9 and abs(r["h11"]) < 1e-9
    assert abs(r["lap"] - 1.0 / 3.0) < 1e-9


# ---------------------------------------------------------------------------------------------
# Local element coordinates
# ---------------------------------------------------------------------------------------------

def test_second_derivative_with_respect_to_local_coordinates():
    """d2S_shapes, i.e. D2XBasisFunctionLocalCoord. On a uniform N x N mesh of [0,1]^2 with local
    coordinates in [-1,1], dx/ds = 1/(2N), so d2w/ds1^2 = w_xx/(2N)^2 and
    d2w/(ds1 ds2) = w_xy/(2N)^2."""
    N = 3

    class P(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=N))
            w = var("w")
            s1, s2 = var("local_coordinate_1"), var("local_coordinate_2")
            eqs = _Carrier() + IntegralObservables(dss=diff(diff(w, s1), s1),
                                                   dst=diff(diff(w, s1), s2), vol=1)
            self.add_equations(eqs @ "domain")

    with P() as p:
        p.initialise()
        m = p.get_mesh("domain")
        _set_nodal(m, lambda x, y: x * x + 3 * x * y)   # w_xx = 2, w_xy = 3
        o = m.evaluate_all_observables()
        v = float(o["vol"])
        scale = 1.0 / (2.0 * N) ** 2
        assert abs(float(o["dss"]) / v - 2.0 * scale) < 1e-12
        assert abs(float(o["dst"]) / v - 3.0 * scale) < 1e-12


# ---------------------------------------------------------------------------------------------
# Surfaces on a MOVING mesh: codimension together with d_d2x_shape_dcoord and its Hessian
# ---------------------------------------------------------------------------------------------

def _moving_interface_problem(hessian):
    from pyoomph.equations.ALE import LaplaceSmoothedMesh

    class Bulk(Equations):
        def define_fields(self):
            self.define_scalar_field("d", "C2")

        def define_residuals(self):
            d, q = var_and_test("d")
            self.add_residual(weak(grad(d), grad(q)) + weak(1, q))

    class Surf(Equations):
        def define_fields(self):
            self.define_scalar_field("u", "C2")

        def define_residuals(self):
            u, v = var_and_test("u")
            self.add_residual(weak((1 + u ** 2) * div(grad(u)), v) + weak(1, v))

    class P(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=3))
            eqs = Bulk() + LaplaceSmoothedMesh()
            for b in ("top", "left", "right"):
                eqs += DirichletBC(mesh_x=True, mesh_y=True) @ b
            eqs += DirichletBC(mesh_x=True) @ "bottom"      # the interface may move vertically
            eqs += DirichletBC(d=0) @ "top"
            self.add_equations(eqs @ "domain")
            self.add_equations(Surf() @ "domain/bottom")
            if not hessian:
                self.equation_compilation_flags.debug_jacobian_epsilon = 1e-5

    p = P()
    if hessian:
        p.setup_for_stability_analysis(analytic_hessian=True)
    p.initialise()
    m = p.get_mesh("domain")
    for n in m.nodes():
        a, b = n.x(0), n.x(1)
        n.set_x(0, a + 0.30 * a * b)
        n.set_x(1, b - 0.20 * a * b + 0.1 * a * (1 - a))   # curves the interface itself
    im = p.get_mesh("domain/bottom")
    _set_nodal(im, lambda x, y: 0.3 + 0.5 * x, name="u")
    return p


def test_surface_second_derivative_position_jacobian():
    import io
    import contextlib
    p = _moving_interface_problem(hessian=False)
    buf = io.StringIO()
    with p:
        with contextlib.redirect_stdout(buf):
            p.assemble_jacobian(with_residual=False)
    assert "DIFFERENCES IN JACOBIAN" not in buf.getvalue(), buf.getvalue()[:4000]


def test_surface_second_derivative_hessian():
    p = _moving_interface_problem(hessian=True)
    with p:
        p.debug_analytic_hessian_by_fd(epsilon=1e-4)


# ---------------------------------------------------------------------------------------------
# The discontinuous (DL) space
# ---------------------------------------------------------------------------------------------

def test_dl_second_derivative_on_a_moving_mesh():
    """A DL basis is affine in the local coordinates, so its local second derivatives vanish and the
    whole physical second derivative comes from the X_{k,ab} term. Checked through the Jacobian,
    since DL dofs are element-internal and cannot simply be written like nodal values."""
    import io
    import contextlib
    from pyoomph.equations.ALE import LaplaceSmoothedMesh

    class Eq(Equations):
        def define_fields(self):
            self.define_scalar_field("p", "DL")

        def define_residuals(self):
            pf, r = var_and_test("p")
            self.add_residual(weak((1 + pf ** 2) * div(grad(pf)), r) + weak(1, r))

    class P(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=3))
            eqs = Eq() + LaplaceSmoothedMesh()
            for b in ("top", "bottom", "left", "right"):
                eqs += DirichletBC(mesh_x=True, mesh_y=True) @ b
            self.add_equations(eqs @ "domain")
            self.equation_compilation_flags.debug_jacobian_epsilon = 1e-5

    buf = io.StringIO()
    with P() as p:
        p.initialise()
        _bilinear_distort(p.get_mesh("domain"))
        with contextlib.redirect_stdout(buf):
            p.assemble_jacobian(with_residual=False)
    assert "DIFFERENCES IN JACOBIAN" not in buf.getvalue(), buf.getvalue()[:4000]


# ---------------------------------------------------------------------------------------------
# Bubble-enriched spaces and hanging nodes
# ---------------------------------------------------------------------------------------------

def test_second_derivative_on_bubble_enriched_triangles():
    res = _measure(lambda: RectangularQuadMesh(N=3, split_in_tris="crossed"),
                   lambda x, y: x * x + 3 * x * y - 2 * y * y, 2, space="C2TB")
    _check(res, 2, [[2.0, 3.0], [3.0, -4.0]], -2.0)


def test_second_derivative_with_hanging_nodes():
    """Adaptive refinement puts hanging nodes into the C2 space; the second derivatives have to go
    through the same constraint machinery as the first ones."""

    class P(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=2))
            self.max_refinement_level = 3
            w = var("w")
            eqs = _Carrier() + _hessian_observables(2) + SpatialErrorEstimator(w=1)
            self.add_equations(eqs @ "domain")

    with P() as p:
        p.initialise()
        p.refine_uniformly()
        m = p.get_mesh("domain")
        _set_nodal(m, lambda x, y: x * x + 3 * x * y - 2 * y * y)
        o = m.evaluate_all_observables()
        vol = float(o["vol"])
        res = {k: float(o[k]) / vol for k in o if k != "vol"}
        _check(res, 2, [[2.0, 3.0], [3.0, -4.0]], -2.0)


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


# ---------------------------------------------------------------------------------------------
# Analytic Hessian of the AZIMUTHAL (m!=0) contributions when a normal is involved
#
# The Hessian of the m!=0 contributions is what azimuthal bifurcation tracking assembles into the
# eigenvector rows of its augmented Jacobian. Deriving it runs the second differentiation under
# expansion mode 0 (the base state) while the first ran under mode 1 (the eigenfunction), and a
# normal that had already been derived once still carries the mode-1 label. Vetoing it on that
# label dropped every d2_normal_d2coord term from the generated HessianVectorProduct of the m!=0
# residual - the first derivatives were all there, so the result looked plausible - and azimuthal
# bifurcation tracking then diverged on any moving mesh with a free surface.
#
# The check compares the augmented Jacobian against a central difference of the augmented residual
# along a base-dof direction. Its noise floor is ~1e-7; with the terms missing it reads 5.0e-3, so
# the check is sensitive to the thing it is meant to test by four orders of magnitude.
#
# No eigensolver is involved: the tracker is handed a synthetic "eigenvector", which is all the
# Hessian blocks need. Only base-dof directions are probed, since perturbing the eigenvector dofs
# from Python does not refresh the handler's internal copies of them.
# ---------------------------------------------------------------------------------------------

def test_azimuthal_hessian_with_normal_on_moving_mesh():
    import numpy
    from pyoomph.equations.ALE import LaplaceSmoothedMesh

    class Eq(Equations):
        def define_fields(self):
            self.define_scalar_field("u", "C2")

        def define_residuals(self):
            u, v = var_and_test("u")
            self.add_residual(weak(grad(u), grad(v)) + weak((1 + u ** 2), v))

    class IFaceEq(Equations):
        # the normal is the whole point: an interface term without one is differentiated correctly
        def define_residuals(self):
            u, v = var_and_test("u")
            A = self.get_current_code_generator().get_problem().get_global_parameter("A")
            self.add_residual(weak(A * u * dot(var("normal"), vector(1, 1)), v))

    class P(Problem):
        def define_problem(self):
            self.A = self.define_global_parameter(A=1.0)
            self.set_coordinate_system("axisymmetric")
            # away from r=0, so nothing here depends on the axis treatment
            self.add_mesh(RectangularQuadMesh(N=2, size=[1, 1], lower_left=[1, 0]))
            eqs = Eq() + LaplaceSmoothedMesh() + IFaceEq() @ "top"
            eqs += DirichletBC(mesh_x=True, mesh_y=True) @ "left"
            eqs += DirichletBC(mesh_x=True, mesh_y=True) @ "bottom"
            eqs += DirichletBC(mesh_x=True, mesh_y=True) @ "right"
            eqs += DirichletBC(u=0) @ "bottom"
            self.add_equations(eqs @ "domain")

    with P() as p:
        p.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=True)
        p.initialise()
        nbase = p.ndof()

        # A generic state: on a perfectly regular mesh several geometric second derivatives vanish
        # identically and a missing term would not show up at all.
        rng = numpy.random.default_rng(7)
        x0 = numpy.array(p.get_current_dofs()[0]) + 0.02 * rng.standard_normal(nbase)
        p.set_current_dofs(x0)

        V = rng.standard_normal(nbase) + 1j * rng.standard_normal(nbase)
        V /= numpy.linalg.norm(V)
        p.activate_bifurcation_tracking("A", bifurcation_type="azimuthal", azimuthal_mode=1,
                                        eigenvector=V, omega=0.3)
        naug = p.ndof()
        xaug = numpy.array(p.get_current_dofs()[0])
        J = p.assemble_jacobian(with_residual=False)

        d = numpy.zeros(naug)
        d[:nbase] = rng.standard_normal(nbase)
        d /= numpy.linalg.norm(d)
        eps = 1e-6
        p.set_current_dofs(xaug + eps * d)
        rp = numpy.array(p.get_residuals())
        p.set_current_dofs(xaug - eps * d)
        rm = numpy.array(p.get_residuals())
        p.set_current_dofs(xaug)

        fd = (rp - rm) / (2 * eps)
        ana = J @ d
        # the eigenvector rows, i.e. everything the Hessian feeds
        lo, hi = nbase, min(3 * nbase, naug)
        rel = numpy.max(numpy.abs(ana[lo:hi] - fd[lo:hi])) / max(numpy.max(numpy.abs(fd[lo:hi])), 1e-30)
        assert rel < 1e-5, "azimuthal Hessian disagrees with a finite difference: rel=%.3e" % rel
