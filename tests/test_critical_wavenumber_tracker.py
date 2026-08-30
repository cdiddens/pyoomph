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

# CriticalWavenumberTracker: the co-dimension-2 point of a Cartesian normal mode instability, where
# Re(lambda)=0 AND dRe(lambda)/dk=0, i.e. the minimum of the neutral curve gamma_c(k).
#
# Two of the tests here have a CLOSED-FORM answer, which is the whole reason they are built on a
# PointMesh: with no spatial discretisation the k-dependence of the operator is purely algebraic, so
# the dispersion relation is the textbook one and the critical point can be written down.
#
#   - Brusselator, a stationary (Turing) instability, so the tracker's REAL branch:
#         B_c = (1 + A/sqrt(d))^2,   k_c = sqrt(A)/d^(1/4)
#     and, since both are closed form in d, arclength-continuing the critical point in d can be
#     checked at every step and not only at the start.
#   - A linear rotating pair carrying an auxiliary field for the k^4 term, so the tracker's COMPLEX
#     branch: lambda(k) = r - (k^2-q0^2)^2 +- I*(w0 + c k^2), hence r_c=0, k_c=q0, omega=w0+c q0^2 and
#     dlambda/dk = +-2 c q0 -- the last one is why the rotation rate is made k-dependent, otherwise
#     the mu unknown would be zero and untested.
#
# The remaining tests are consistency rather than correctness: every column of the augmented Jacobian
# against a central difference of the augmented residual, with the same instrument as
# test_bifurcation_tracker_jacobians.py. Those blocks that need d2J/dU dk, d2J/dgamma dk or d2J/dk2
# are finite-differenced in k inside the tracker, so this is the only thing that says whether they
# land in the right place.
#
# The complex consistency test needs an odd-in-k term (otherwise J_imag does not exist at all) and a
# nonlinearity (otherwise the Hessian blocks are zero and invisible), which no closed-form problem
# provides. It makes its eigenvector up and never calls an eigensolver -- legitimate, because
# residual/Jacobian consistency does not care where the state came from, and it keeps the test off
# the complex PETSc build that a genuinely complex J would otherwise require.

import numpy
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.generic import AxisymmetryBC
from pyoomph.meshes.simplemeshes import PointMesh, LineMesh, RectangularQuadMesh
from pyoomph.generic.bifurcation_tools import CriticalWavenumberTracker, NormalModeBifurcationTracker


# ----------------------------------------------------------------------------------------------
# the finite-difference instrument (same idea as test_bifurcation_tracker_jacobians._fd_all_columns)
# ----------------------------------------------------------------------------------------------

def _fd_all_columns(p, eps=1e-6):
    """Worst relative disagreement between a column of the augmented Jacobian and a central
    difference of the augmented residual, over all columns. Restores the dofs it perturbs."""
    naug = p.ndof()
    x0 = numpy.array(p.get_current_dofs()[0])
    J = numpy.asarray(p.assemble_jacobian(with_residual=False).todense())
    worst, worst_at = 0.0, "none"
    for j in range(naug):
        x = x0.copy(); x[j] += eps
        p.set_current_dofs(x); rp = numpy.array(p.get_residuals())
        x = x0.copy(); x[j] -= eps
        p.set_current_dofs(x); rm = numpy.array(p.get_residuals())
        col = (rp - rm) / (2 * eps)
        scale = max(numpy.max(numpy.abs(col)), numpy.max(numpy.abs(J[:, j])), 1e-30)
        rel = numpy.max(numpy.abs(J[:, j] - col)) / scale
        if rel > worst:
            worst, worst_at = rel, "column %d of %d" % (j, naug)
    p.set_current_dofs(x0)
    return worst, worst_at


def _fd_parameter_derivative(p, param, eps=1e-6):
    """The handler's dResidual/dparameter against a central difference over the parameter. This is
    what arclength continuation of the critical point in a third parameter runs on, and no
    column-by-column check of the Jacobian reaches it."""
    ana = numpy.array(p.get_parameter_derivative(param))
    par = p.get_global_parameter(param)
    p0 = par.value
    par.value = p0 + eps; rp = numpy.array(p.get_residuals())
    par.value = p0 - eps; rm = numpy.array(p.get_residuals())
    par.value = p0
    fd = (rp - rm) / (2 * eps)
    scale = max(numpy.max(numpy.abs(fd)), numpy.max(numpy.abs(ana)), 1e-30)
    return numpy.max(numpy.abs(ana - fd)) / scale


# ----------------------------------------------------------------------------------------------
# Brusselator on a point: a Turing instability with a closed-form critical point
# ----------------------------------------------------------------------------------------------

class _BrusselatorEqs(Equations):
    def __init__(self, A, B, d):
        super().__init__()
        self.A, self.B, self.d = A, B, d

    def define_fields(self):
        self.define_scalar_field("u", "C2")
        self.define_scalar_field("v", "C2")

    def define_residuals(self):
        u, ut = var_and_test("u")
        v, vt = var_and_test("v")
        f = self.A - (self.B + 1) * u + u ** 2 * v
        g = self.B * u - u ** 2 * v
        self.add_weak(partial_t(u) - f, ut).add_weak(grad(u), grad(ut))
        self.add_weak(partial_t(v) - g, vt).add_weak(self.d * grad(v), grad(vt))


class _BrusselatorProblem(Problem):
    def __init__(self, A=2.0, B=2.5, d=8.0):
        super().__init__()
        self.A = self.define_global_parameter(A=A)
        self.B = self.define_global_parameter(B=B)
        self.d = self.define_global_parameter(d=d)

    def define_problem(self):
        self += PointMesh()
        eqs = _BrusselatorEqs(self.A, self.B, self.d)
        eqs += InitialCondition(u=self.A, v=self.B / self.A)
        self += eqs @ "domain"


def _brusselator_exact(A, d):
    return (1 + A / numpy.sqrt(d)) ** 2, numpy.sqrt(A) / d ** 0.25


def _tracked_brusselator(A=2.0, d=8.0, offset=0.15):
    """Start off the critical point in BOTH unknowns, solve, and hand back the installed tracker."""
    Bc, kc = _brusselator_exact(A, d)
    p = _BrusselatorProblem(A=A, B=Bc - offset, d=d)
    p.set_c_compiler("tcc")
    p.quiet()
    p.setup_for_stability_analysis(additional_cartesian_mode=True, analytic_hessian=True)
    p.solve()
    p.solve_eigenproblem(2, normal_mode_k=0.8 * kc)
    tracker = CriticalWavenumberTracker(p, "B", eigenvector=0)
    p.set_custom_assembler(tracker)
    return p, tracker


def test_brusselator_critical_point_is_exact():
    """The stationary branch, against the closed-form Turing threshold.

    A Turing instability is neutral at lambda=0 exactly, so this exercises the smaller (3N+2)
    formulation in which the mass matrix never appears and dlambda/dk is real and pinned to zero.
    """
    A, d = 2.0, 8.0
    Bc, kc = _brusselator_exact(A, d)
    p, tracker = _tracked_brusselator(A=A, d=d)
    with p:
        assert not tracker.has_imag, "the Turing mode is stationary, the real branch must be taken"
        assert p.ndof() == 3 * 2 + 2
        p.solve()
        assert float(p.get_global_parameter("B").value) == pytest.approx(Bc, abs=1e-8)
        assert tracker.get_critical_wavenumber() == pytest.approx(kc, abs=1e-8)
        assert tracker.get_critical_omega() == 0.0


def test_critical_point_continues_in_a_third_parameter():
    """Arclength continuation of the co-dimension-2 point, checked against the exact locus.

    Both B_c(d) and k_c(d) are closed form, so this does not merely check that the continuation runs
    -- every step has a known answer. It is also the only thing that exercises the handler's
    dResidual/dparameter, which is what oomph-lib's bordered continuation solves against.
    """
    A, d0 = 2.0, 8.0
    p, tracker = _tracked_brusselator(A=A, d=d0, offset=0.1)
    with p:
        p.solve()
        ds, d = 0.4, d0
        for _ in range(5):
            ds = p.arclength_continuation("d", ds, max_ds=1.0)
            d = float(p.get_global_parameter("d").value)
            Bc, kc = _brusselator_exact(A, d)
            assert float(p.get_global_parameter("B").value) == pytest.approx(Bc, abs=1e-8), \
                "left the exact locus at d=%g" % d
            assert tracker.get_critical_wavenumber() == pytest.approx(kc, abs=1e-8), \
                "left the exact locus at d=%g" % d
        assert d > d0 + 1.0, "the continuation did not actually move"


def test_augmented_jacobian_is_exact_for_a_stationary_mode():
    p, tracker = _tracked_brusselator()
    with p:
        rng = numpy.random.default_rng(3)
        n = p.ndof()
        p.set_current_dofs(numpy.array(p.get_current_dofs()[0]) + 0.05 * rng.standard_normal(n))
        worst, where = _fd_all_columns(p)
        assert worst < 1e-5, "real-branch augmented Jacobian: %.3e at %s" % (worst, where)
        # "d" is the case that matters: it multiplies the k^2 diffusion term, so d(eigen rows)/dd
        # depends on k and the finite difference in k is actually engaged. "A" enters the reaction
        # terms only, where that contribution is identically zero.
        for param in ("A", "d"):
            rel = _fd_parameter_derivative(p, param)
            assert rel < 1e-5, "dResidual/d%s disagrees with a finite difference: %.3e" % (param, rel)


# ----------------------------------------------------------------------------------------------
# A rotating pair on a point: an oscillatory instability at finite k, also closed form
# ----------------------------------------------------------------------------------------------

_Q0, _W0, _C = 0.9, 0.7, 0.25


class _RotatingPairEqs(Equations):
    """wp,wq carry the factor (k^2-q0^2), so that the (p,q) pair obeys

        dp/dt = (r-(k^2-q0^2)^2) p - (w0+c k^2) q
        dq/dt = (w0+c k^2) p + (r-(k^2-q0^2)^2) q

    i.e. lambda(k) = r - (k^2-q0^2)^2 +- I*(w0 + c k^2). Re(lambda) is stationary in k at k=q0 and
    vanishes there for r=0; the k-dependent rotation rate is what makes dlambda/dk nonzero at the
    critical point, so that the mu unknown carries information.
    """

    def __init__(self, r):
        super().__init__()
        self.r = r

    def define_fields(self):
        for f in ("p", "q", "wp", "wq"):
            self.define_scalar_field(f, "C2")

    def define_residuals(self):
        p, pt = var_and_test("p")
        q, qt = var_and_test("q")
        wp, wpt = var_and_test("wp")
        wq, wqt = var_and_test("wq")
        self.add_weak(wp + _Q0 ** 2 * p, wpt).add_weak(-grad(p), grad(wpt))
        self.add_weak(wq + _Q0 ** 2 * q, wqt).add_weak(-grad(q), grad(wqt))
        self.add_weak(partial_t(p) - self.r * p + _W0 * q - _Q0 ** 2 * wp, pt)
        self.add_weak(grad(wp), grad(pt)).add_weak(_C * grad(q), grad(pt))
        self.add_weak(partial_t(q) - self.r * q - _W0 * p - _Q0 ** 2 * wq, qt)
        self.add_weak(grad(wq), grad(qt)).add_weak(-_C * grad(p), grad(qt))


class _RotatingPairProblem(Problem):
    def __init__(self, r=-0.05):
        super().__init__()
        self.r = self.define_global_parameter(r=r)

    def define_problem(self):
        self += PointMesh()
        self += _RotatingPairEqs(self.r) @ "domain"


def test_oscillatory_critical_point_is_exact():
    """The complex branch, against a dispersion relation known in closed form.

    Checks all four scalar unknowns: the parameter, the wavenumber, omega, and mu=Im(dlambda/dk).
    """
    with _RotatingPairProblem(r=-0.05) as p:
        p.set_c_compiler("tcc")
        p.quiet()
        p.setup_for_stability_analysis(additional_cartesian_mode=True, analytic_hessian=True)
        p.solve()
        # wp,wq carry no time derivative, so the mass matrix is singular and the generalised
        # eigenproblem has spurious eigenvalues at ~1/eps. Keep only the finite ones.
        p.solve_eigenproblem(4, normal_mode_k=0.75, filter=lambda l: abs(l) < 1e3)
        tracker = CriticalWavenumberTracker(p, "r", eigenvector=0)
        assert tracker.has_imag, "the neutral mode is oscillatory, the complex branch must be taken"
        p.set_custom_assembler(tracker)
        assert p.ndof() == 5 * 4 + 4
        p.solve()
        omega = tracker.get_critical_omega()
        assert float(p.get_global_parameter("r").value) == pytest.approx(0.0, abs=1e-8)
        assert tracker.get_critical_wavenumber() == pytest.approx(_Q0, abs=1e-8)
        assert abs(omega) == pytest.approx(_W0 + _C * _Q0 ** 2, abs=1e-8)
        # dlambda/dk is purely imaginary by construction; its sign follows the branch the
        # eigensolver happened to return.
        assert tracker.get_dlambda_dk().real == 0.0
        assert tracker.get_dlambda_dk().imag == pytest.approx(
            numpy.sign(omega) * 2 * _C * _Q0, abs=1e-8)


# ----------------------------------------------------------------------------------------------
# A nonlinear problem with an odd-in-k term, for the complex branch's Jacobian
# ----------------------------------------------------------------------------------------------

class _AdvectionEqs(Equations):
    """Nonlinear advection-diffusion of c by a vector field U.

    Under the Cartesian normal mode expansion U gains a component along the extra direction, so
    dot(U,grad(c)) contains U_z*(I*k*c): a term of ODD order in k, which is the only way the
    imaginary contribution J_imag/M_imag becomes nonzero at all. The mass factor (B + c^2/2) makes
    both dM/dB and dM/dU nonzero, and the products of unknowns make the Hessian blocks nonzero --
    without all three, most of the augmented Jacobian is structurally zero and untested.
    """

    def __init__(self, A, B):
        super().__init__()
        self.A, self.B = A, B

    def define_fields(self):
        self.define_vector_field("U", "C2")
        self.define_scalar_field("c", "C2")

    def define_residuals(self):
        U, Ut = var_and_test("U")
        c, ct = var_and_test("c")
        mass = self.B + 0.5 * c ** 2
        self.add_weak(mass * partial_t(c) + dot(U, grad(c)) - self.A * c + c ** 3, ct)
        self.add_weak(grad(c), grad(ct))
        self.add_weak(mass * partial_t(U) + self.A * U + c * U, Ut)
        self.add_weak(0.5 * grad(U), grad(Ut))


class _AdvectionProblem(Problem):
    def define_problem(self):
        self.A = self.define_global_parameter(A=1.3)
        self.B = self.define_global_parameter(B=1.5)
        self += LineMesh(N=2)
        eqs = _AdvectionEqs(self.A, self.B)
        eqs += InitialCondition(c=0.4)
        eqs += DirichletBC(c=0.2, U_x=0) @ "left"
        self += eqs @ "domain"


def test_augmented_jacobian_is_exact_for_an_oscillatory_mode():
    """Every column of the 5N+4 system, on a problem where none of its blocks is structurally zero.

    No eigensolver is called: the eigenvector guess is made up and pushed into the problem's
    eigendata directly. Residual/Jacobian consistency does not depend on where the state came from,
    and a made-up complex eigenvector keeps this off the complex PETSc build.
    """
    with _AdvectionProblem() as p:
        p.set_c_compiler("tcc")
        p.quiet()
        # A normal-mode tracker with an imaginary contribution asks the multi-assembly for Hessian
        # products whose entries fall outside the Jacobian's symbolic pattern, which the frozen
        # sparsity path refuses. Pre-existing: NormalModeBifurcationTracker is refused identically.
        p.use_frozen_sparsity = False
        p.setup_for_stability_analysis(additional_cartesian_mode=True, analytic_hessian=True)
        p.solve()
        rng = numpy.random.default_rng(11)
        nbase = p.ndof()
        V = rng.standard_normal(nbase) + 1j * rng.standard_normal(nbase)
        V /= numpy.linalg.norm(V)
        p._last_eigenvalues = numpy.array([-0.13 + 0.42j])
        p._last_eigenvectors = numpy.array([V])
        p._last_eigenvalues_k = numpy.array([0.9])
        tracker = CriticalWavenumberTracker(p, "A", eigenvector=0, cartesian_k=0.9)
        assert tracker.has_imag and tracker.has_imag_contribution
        p.set_custom_assembler(tracker)
        assert p.ndof() == 5 * nbase + 4
        n = p.ndof()
        p.set_current_dofs(numpy.array(p.get_current_dofs()[0]) + 0.05 * rng.standard_normal(n))
        worst, where = _fd_all_columns(p)
        assert worst < 1e-5, "complex-branch augmented Jacobian: %.3e at %s" % (worst, where)
        for param in ("A", "B"):
            rel = _fd_parameter_derivative(p, param)
            assert rel < 1e-5, "dResidual/d%s disagrees with a finite difference: %.3e" % (param, rel)


# ----------------------------------------------------------------------------------------------
# Azimuthal modes: the same Brusselator on an annulus, where m plays the role of k
# ----------------------------------------------------------------------------------------------

class _AnnulusProblem(Problem):
    """The Brusselator on an axisymmetric annulus, away from the axis.

    The base state is the same uniform (A, B/A) as on a point -- diffusion does not see a constant --
    but the azimuthal expansion puts m^2/r^2 into the operator, so the Turing instability's preferred
    total wavenumber becomes a preferred m. The neutral curve B_c(m) therefore has an interior
    minimum, at roughly r*sqrt(A)/d^(1/4), which is what makes this a critical-mode problem at all.
    The answer is not closed form (r varies across the annulus, and the radial modes are quantised),
    so it is checked against the ordinary tracker instead.
    """

    def __init__(self, A=2.0, B=2.75, d=8.0):
        super().__init__()
        self.A = self.define_global_parameter(A=A)
        self.B = self.define_global_parameter(B=B)
        self.d = self.define_global_parameter(d=d)

    def define_problem(self):
        self.set_coordinate_system("axisymmetric")
        self += RectangularQuadMesh(N=[4, 1], size=[1.0, 0.3], lower_left=[2.0, 0.0])
        eqs = _BrusselatorEqs(self.A, self.B, self.d)
        eqs += InitialCondition(u=self.A, v=self.B / self.A)
        self += eqs @ "domain"


def _tracked_annulus():
    p = _AnnulusProblem()
    p.set_c_compiler("tcc")
    p.quiet()
    p.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=True)
    p.solve()
    p.solve_eigenproblem(3, azimuthal_m=2)
    tracker = CriticalWavenumberTracker(p, "B", eigenvector=0)
    p.set_custom_assembler(tracker)
    return p, tracker


def test_azimuthal_critical_mode_is_stationary():
    """The critical (B_c, m_c) of an azimuthal instability, verified against the ordinary tracker.

    There is no closed form here, so the check is the defining property instead: run
    NormalModeBifurcationTracker at three fixed REAL m around the answer, and require that B_c(m) has
    its minimum there. The parabola through the three points must have its vertex at m_c, and its
    value at m_c must be the B the co-dimension-2 system converged to.
    """
    p, tracker = _tracked_annulus()
    with p:
        assert tracker.azimuthal and not tracker.has_imag
        assert p.ndof() == 3 * p._get_n_unaugmented_dofs() + 2
        p.solve(max_newton_iterations=40)
        mc = tracker.get_critical_azimuthal_m()
        Bc = float(p.get_global_parameter("B").value)
        assert 1.5 < mc < 3.0, "critical mode %g is nowhere near the expected one" % mc
        with pytest.raises(RuntimeError, match="azimuthal mode"):
            tracker.get_critical_wavenumber()

        p.set_custom_assembler(None)
        dm = 0.05
        B_of_m = {}
        for off in (-dm, 0.0, dm):
            p.get_global_parameter("B").value = Bc
            # No eigensolve in this loop: m is not an integer, and the eigenvector stored by the
            # tracker is a perfectly good guess at a neighbouring m anyway.
            fixed = NormalModeBifurcationTracker(p, "B", eigenvector=0, azimuthal_m=mc + off)
            p.set_custom_assembler(fixed)
            p.solve(max_newton_iterations=40)
            B_of_m[off] = float(p.get_global_parameter("B").value)
            p.set_custom_assembler(None)

        assert B_of_m[0.0] == pytest.approx(Bc, abs=1e-8), \
            "the two trackers disagree about B_c at the same m"
        assert B_of_m[0.0] < B_of_m[-dm] and B_of_m[0.0] < B_of_m[dm], \
            "B_c(m) is not minimal at the tracked m: %r" % B_of_m
        curvature = B_of_m[-dm] - 2 * B_of_m[0.0] + B_of_m[dm]
        assert curvature > 0
        # Vertex of the parabola through the three samples, relative to m_c. Exact only to O(dm^2),
        # hence the loose bound -- but a tracker that stopped anywhere but the minimum would miss by
        # something of order dm or more.
        vertex = dm * (B_of_m[-dm] - B_of_m[dm]) / (2 * curvature)
        assert abs(vertex) < 0.01, "the sampled minimum sits %+.4f away from the tracked m" % vertex


def test_azimuthal_augmented_jacobian_is_exact():
    p, tracker = _tracked_annulus()
    with p:
        rng = numpy.random.default_rng(5)
        n = p.ndof()
        p.set_current_dofs(numpy.array(p.get_current_dofs()[0]) + 0.02 * rng.standard_normal(n))
        worst, where = _fd_all_columns(p)
        assert worst < 1e-5, "azimuthal augmented Jacobian: %.3e at %s" % (worst, where)
        for param in ("A", "d"):
            rel = _fd_parameter_derivative(p, param)
            assert rel < 1e-5, "dResidual/d%s disagrees with a finite difference: %.3e" % (param, rel)


def test_stationary_normal_mode_tracker_jacobian_is_exact():
    """Regression for the ordinary tracker's stationary branch, which the test above depends on.

    That branch is only reached by a normal-mode problem with a stationary neutral mode AND no
    imaginary contribution -- scalar fields only -- and it had four independent defects: no
    dR/dparameter column at all, the eigen residual built from the matrix rather than the
    matrix-vector product, and the Hessian and dJ/dp taken from the base residual instead of the real
    contribution.
    """
    p, tracker = _tracked_annulus()
    with p:
        p.set_custom_assembler(None)
        fixed = NormalModeBifurcationTracker(p, "B", eigenvector=0, azimuthal_m=2.1)
        assert not fixed.has_imag
        p.set_custom_assembler(fixed)
        rng = numpy.random.default_rng(4)
        n = p.ndof()
        p.set_current_dofs(numpy.array(p.get_current_dofs()[0]) + 0.02 * rng.standard_normal(n))
        worst, where = _fd_all_columns(p)
        assert worst < 1e-5, "NormalModeBifurcationTracker (stationary): %.3e at %s" % (worst, where)


# ----------------------------------------------------------------------------------------------
# The axis, which is where a real m stops being a mere relabelling
# ----------------------------------------------------------------------------------------------

class _AxisEqs(Equations):
    """A vector field is essential here: for a SCALAR the axis conditions of |m|=1 and |m|>=2 agree,
    and only the radial/azimuthal components tell the two regimes apart."""

    def __init__(self, G):
        super().__init__()
        self.G = G

    def define_fields(self):
        self.define_vector_field("U", "C1")
        self.define_scalar_field("c", "C1")

    def define_residuals(self):
        U, Ut = var_and_test("U")
        c, ct = var_and_test("c")
        self.add_weak(partial_t(c) - self.G * c + c ** 3 - 1, ct).add_weak(grad(c), grad(ct))
        self.add_weak(partial_t(U) + self.G * U + c * U, Ut).add_weak(0.4 * grad(U), grad(Ut))


class _AxisProblem(Problem):
    def define_problem(self):
        self.G = self.define_global_parameter(G=1.2)
        self.set_coordinate_system("axisymmetric")
        self += RectangularQuadMesh(N=[2, 1], size=[1.0, 0.5])
        eqs = _AxisEqs(self.G)
        eqs += AxisymmetryBC(verbose=False) @ "left"
        eqs += InitialCondition(c=1.0)
        self += eqs @ "domain"


def test_axis_mask_is_frozen_in_the_high_m_regime():
    """A real m has no regime of its own: the axis conditions are a step function of m, and
    _get_forced_zero_dofs_for_eigenproblem truncates before branching. The tracker must therefore take
    the |m|>1 mask no matter where in the reals m starts, and must not accept |m|<=1 at all."""
    with _AxisProblem() as p:
        p.set_c_compiler("tcc")
        p.quiet()
        p.use_frozen_sparsity = False   # see the note on CriticalWavenumberTracker
        p.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=True)
        p.solve()
        # AxisymmetryBC keeps the radial/azimuthal components Dirichlet-pinned for ordinary solving
        # and only frees them for an azimuthal eigenproblem, so the masks are meaningless until that
        # switch has happened. Doing it without an eigensolve keeps this off complex PETSc.
        p.get_global_parameter("azimuthal_m").value = 2
        p.actions_before_eigen_solve()
        ref = {}
        for m in (0, 1, 2, 3):
            names = p._equation_system._get_forced_zero_dofs_for_eigenproblem(p.get_eigen_solver(), m, None)
            ref[m] = set(p.dof_strings_to_global_equations(names))
        assert ref[0] != ref[1] and ref[1] != ref[2] and ref[2] == ref[3], \
            "this problem does not separate the three axis regimes, so it cannot test the freezing"

        rng = numpy.random.default_rng(7)
        nb = p.ndof()
        V = rng.standard_normal(nb) + 1j * rng.standard_normal(nb)
        p._last_eigenvalues = numpy.array([-0.11 + 0.35j])
        p._last_eigenvectors = numpy.array([V / numpy.linalg.norm(V)])
        for mstart in (1.4, 2.4, 3.7):
            t = CriticalWavenumberTracker(p, "G", eigenvector=0, azimuthal_m=mstart)
            assert set(t.eigen_zero_dofs) == ref[2], \
                "m=%g took the m=%d mask" % (mstart, int(mstart))
            assert set(t.base_zero_dofs) == ref[0]

        for bad in (0.0, 0.6, 1.0, -1.0):
            with pytest.raises(RuntimeError, match=r"\|m\|<=1"):
                CriticalWavenumberTracker(p, "G", eigenvector=0, azimuthal_m=bad)


# ----------------------------------------------------------------------------------------------
# refusals
# ----------------------------------------------------------------------------------------------

def test_refuses_to_start_at_zero_wavenumber():
    """The set of dofs forced to zero in the eigenproblem differs at k=0, and the tracker freezes it
    when it is created, so k=0 would silently carry the wrong mask through the whole solve."""
    with _BrusselatorProblem() as p:
        p.set_c_compiler("tcc")
        p.quiet()
        p.setup_for_stability_analysis(additional_cartesian_mode=True, analytic_hessian=True)
        p.solve()
        p.solve_eigenproblem(2, normal_mode_k=0.7)
        with pytest.raises(RuntimeError, match="k=0"):
            CriticalWavenumberTracker(p, "B", eigenvector=0, cartesian_k=0.0)
