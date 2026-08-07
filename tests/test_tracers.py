#  @file
#  Passive tracer particles (dev_docs/tracers.md).
#
#  The advection field is given as an ANALYTIC EXPRESSION of the coordinates, never as a solved
#  field. That is deliberate: it removes the discretisation error entirely, so every assertion here
#  measures the tracer machinery and nothing else. Several of the cases then have an exact answer,
#  and are asserted at machine zero rather than at a tolerance - which is what makes them able to
#  catch the class of defect this rewrite existed to fix, all of which were silent:
#
#   * the mesh Jacobian being taken at the end-of-step configuration for every sub-step,
#   * the ALE term not being blended over the sub-step,
#   * the ALE term not being emitted AT ALL on a mesh without position dofs, so that a mesh moved by
#     hand or by macro elements dragged its tracers with it.
#
#  Three exactness statements recur:
#
#   * In the BULK the ALE term cancels analytically, so a particle in a moving mesh with a zero
#     advection field must not move at all - every Runge-Kutta stage derivative is identically zero,
#     not a cancellation of two rounded terms.
#   * On an INTERFACE the pseudo-inverse projects the field onto the tangent space, so a purely
#     normal advection field moves nothing, and nodes sliding tangentially under a stationary
#     particle move nothing either.
#   * A field that is constant along a particle's own path is integrated exactly by any Runge-Kutta
#     method, so Poiseuille flow gives the analytic straight line to round-off.

import math

import numpy
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.equations.tracers import (TracerParticles, PointSeed, GridSeed, ElementSeed,
                                       RandomSeed, CallableSeed)
from pyoomph.meshes.simplemeshes import RectangularQuadMesh, CuboidBrickMesh, LineMesh


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------

class _TracerProblem(Problem):
    """A rectangle carrying a trivial Poisson dof (so there is something to solve), an analytic
    advection field, and optionally a mesh moved by hand.

    Moving the nodes directly, with no position dofs anywhere, is the point of `motion`: it is the
    configuration for which the old implementation emitted no ALE term at all.
    """

    def __init__(self, advection, seeds, *, motion=None, on_interface=None, N=(8, 4),
                 rtol=1e-10, atol=1e-12, fixed_substeps=0, payloads=None,
                 time_interpolation_order="auto", history_time=None, shear=0.0):
        super().__init__()
        self.advection = advection
        self.seeds = seeds
        self.motion = motion
        self.on_interface = on_interface
        self.N = N
        self._rtol, self._atol = rtol, atol
        self._fixed_substeps = fixed_substeps
        self._payloads = payloads
        self._tio = time_interpolation_order
        self._history_time = history_time
        self.shear = shear
        self._X0 = None

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(size=[4, 2], lower_left=[0, -1], N=list(self.N)))
        eqs = PoissonEquation(source=0) + DirichletBC(u=0) @ "left" + DirichletBC(u=1) @ "right"
        tp = TracerParticles(self.advection, seed=PointSeed(self.seeds),
                             rtol=self._rtol, atol=self._atol,
                             payloads=self._payloads,
                             time_interpolation_order=self._tio,
                             history_time=self._history_time)
        self._tracer_eqs = tp
        eqs += (tp @ self.on_interface) if self.on_interface else tp
        self += eqs @ "domain"

    def _tracer_mesh(self):
        return self.get_mesh("domain/" + self.on_interface if self.on_interface else "domain")

    def tracers(self):
        return self._tracer_mesh().get_tracers()

    def actions_after_newton_solve(self):
        super().actions_after_newton_solve()
        if self._fixed_substeps:
            self.tracers().fixed_substeps = self._fixed_substeps

    def actions_before_newton_solve(self):
        super().actions_before_newton_solve()
        m = self.get_mesh("domain")
        if self._X0 is None:
            self._X0 = numpy.array([[n.x(0), n.x(1)] for n in m.nodes()])
            if self.shear:
                # A shear in x only: the element map stops being affine, while y(s) stays affine per
                # element so a field quadratic in y is still exactly in the FE space.
                for n, X in zip(m.nodes(), self._X0):
                    n.set_x(0, X[0] + self.shear * math.sin(math.pi * X[1]))
                self._X0 = numpy.array([[n.x(0), n.x(1)] for n in m.nodes()])
        if self.motion is None:
            return
        t = float(self.get_current_time(as_float=True, dimensional=False))
        for n, X in zip(m.nodes(), self._X0):
            x, y = self.motion(X[0], X[1], t)
            n.set_x(0, x)
            n.set_x(1, y)


def _run(problem, tag, endtime=1.0, dt=0.05, fixed_substeps=0):
    """Returns (collection, start positions, end positions, elapsed nondimensional time).

    The elapsed time is read back rather than assumed: Problem.run adjusts its last step to land on
    the requested end time and can overshoot it by a few parts in 1e7, which is far above the
    round-off the exact cases are asserted at.
    """
    problem.set_output_directory("_tracer_" + tag)
    problem.quiet()
    problem.initialise()
    tr = problem.tracers()
    if fixed_substeps:
        tr.fixed_substeps = fixed_substeps
    t0 = float(problem.get_current_time(as_float=True, dimensional=False))
    start = tr.get_positions().copy()
    problem.run(endtime, timestep=dt, outstep=False, startstep=dt)
    t1 = float(problem.get_current_time(as_float=True, dimensional=False))
    return tr, start, tr.get_positions().copy(), t1 - t0


# Rigid translation: affine in space, linear in time.
def _translate(x, y, t):
    return (x + 0.3 * t, y + 0.1 * t)


# Non-affine in space and nonlinear in time - the case a frozen end-of-step Jacobian cannot get right.
def _squeeze(x, y, t):
    a = 0.25 * math.sin(1.7 * t)
    return (x * (1 + 0.2 * a) + 0.15 * math.sin(1.1 * t) * y,
            y * (1.0 + a) + 0.1 * math.sin(0.9 * t) * x * y)


def _slide_x(x, y, t):
    return (x + 0.3 * t, y)


def _lift_y(x, y, t):
    return (x, y + 0.1 * t)


POISEUILLE = vector(1 - var("coordinate_y") ** 2, 0)
SEEDS = [[0.5, 0.0], [1.3, 0.4], [2.1, -0.6], [3.2, 0.75]]


# ----------------------------------------------------------------------------------------------
# A - Poiseuille on a static mesh: machine zero
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("shear", [0.0, 0.3], ids=["plain", "sheared"])
def test_poiseuille_static_is_exact(shear):
    """v_y is identically zero and v_x is constant along each particle's own path, so any
    Runge-Kutta method integrates this exactly. Anything above round-off is a defect, and on the
    sheared mesh it additionally says the non-affine element inversion is exact."""
    p = _TracerProblem(POISEUILLE, SEEDS, shear=shear)
    tr, start, end, T = _run(p, "poiseuille_%s" % ("sheared" if shear else "plain"))
    assert tr.nlocal() == len(SEEDS)
    assert numpy.max(numpy.abs(end[:, 1] - start[:, 1])) < 1e-13
    expected_x = start[:, 0] + (1.0 - start[:, 1] ** 2) * T
    assert numpy.max(numpy.abs(end[:, 0] - expected_x)) < 1e-11


# ----------------------------------------------------------------------------------------------
# B - rigid rotation: radius conservation and integrator order
# ----------------------------------------------------------------------------------------------

_ROT_CENTRE = (2.0, 0.0)


def _rotation(omega):
    """Rigid rotation about the centre of the mesh. Linear in the coordinates, so it is represented
    exactly and the only error left is the ODE integrator's."""
    return vector(-omega * (var("coordinate_y") - _ROT_CENTRE[1]),
                  omega * (var("coordinate_x") - _ROT_CENTRE[0]))


def test_rotation_conserves_radius():
    """A linear field, hence exact in the FE space and in the analytic expression alike, so the only
    error is the ODE integrator's."""
    omega = 2 * math.pi  # one revolution per unit time
    seeds = [[_ROT_CENTRE[0] + r * math.cos(a), _ROT_CENTRE[1] + r * math.sin(a)]
             for r, a in [(0.3, 0.0), (0.6, 1.1), (0.8, 2.4)]]
    p = _TracerProblem(_rotation(omega), seeds, N=(16, 8), rtol=1e-11, atol=1e-13)
    tr, start, end, _T = _run(p, "rotation", endtime=1.0, dt=0.02)
    assert tr.nlocal() == len(seeds)
    r0 = numpy.hypot(start[:, 0] - _ROT_CENTRE[0], start[:, 1] - _ROT_CENTRE[1])
    r1 = numpy.hypot(end[:, 0] - _ROT_CENTRE[0], end[:, 1] - _ROT_CENTRE[1])
    assert numpy.max(numpy.abs(r1 - r0)) < 1e-8
    # A full revolution returns every particle to where it started.
    assert numpy.max(numpy.abs(end - start)) < 1e-6


def test_rotation_integrator_is_third_order():
    """The only place the integrator's order is measured, and it is clean because there is no
    spatial error at all to mix in."""
    omega = 2 * math.pi
    seeds = [[_ROT_CENTRE[0] + 0.6, _ROT_CENTRE[1]]]
    errs = []
    for nsub in (2, 4, 8, 16):
        p = _TracerProblem(_rotation(omega), seeds, N=(16, 8), fixed_substeps=nsub)
        _tr, start, end, _T = _run(p, "rotorder_%d" % nsub, endtime=1.0, dt=0.1, fixed_substeps=nsub)
        errs.append(float(numpy.max(numpy.abs(end - start))))
    orders = [math.log(errs[i] / errs[i + 1], 2) for i in range(len(errs) - 1)]
    assert min(orders) > 2.6, "observed orders %s from errors %s" % (orders, errs)


# ----------------------------------------------------------------------------------------------
# C - pure ALE: the mesh moves, the particles do not
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("motion,name", [(_translate, "translate"), (_squeeze, "squeeze")])
def test_moving_mesh_zero_advection_does_not_move_tracers(motion, name):
    """With no position dofs anywhere, so `eval_flag("moving_mesh")` would have been 0 and the old
    code emitted no ALE term at all. Exactly zero is achievable here because the bulk stage
    derivative is identically zero, not a difference of two computed velocities."""
    p = _TracerProblem(vector(0, 0), SEEDS, motion=motion)
    tr, start, end, _T = _run(p, "ale_" + name)
    assert tr.nlocal() == len(SEEDS)
    assert numpy.max(numpy.abs(end - start)) < 1e-13


def test_pure_ale_test_is_not_vacuous():
    """The mesh has to actually move under the particles, by more than an element, or the assertion
    above would hold for a tracer implementation that did nothing at all."""
    def far(x, y, t):
        return (x + 0.8 * t, y + 0.3 * t)

    # Seeds in the middle: the domain slides by more than an element, so a particle near an edge
    # would leave it and the comparison below would be against a different set of particles.
    p = _TracerProblem(vector(0, 0), [[1.8, 0.0], [2.4, 0.2]], motion=far)
    p.set_output_directory("_tracer_ale_vacuity")
    p.quiet()
    p.initialise()
    m = p.get_mesh("domain")
    X0 = numpy.array([[n.x(0), n.x(1)] for n in m.nodes()])
    tr = p.tracers()
    start = tr.get_positions().copy()
    p.run(1.0, timestep=0.05, outstep=False, startstep=0.05)
    X1 = numpy.array([[n.x(0), n.x(1)] for n in m.nodes()])
    element_size = 4.0 / 8
    assert numpy.max(numpy.abs(X1 - X0)) > 1.5 * element_size, \
        "the mesh did not move far enough to matter"
    assert tr.nlocal() == 2, "a particle left the moving domain, so this compares different sets"
    assert numpy.max(numpy.abs(tr.get_positions() - start)) < 1e-13


# ----------------------------------------------------------------------------------------------
# D - the combination: the moving mesh contributes nothing
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("motion,name", [(_translate, "translate"), (_squeeze, "squeeze")])
def test_uniform_advection_on_moving_mesh_matches_static_mesh(motion, name):
    """A spatially uniform field, so the history-level blend of the field is exact whatever the mesh
    does. The mesh motion must then contribute literally nothing - a sharper statement than
    comparing against the analytic answer, because it holds independently of the integrator."""
    seeds = [[0.6, 0.0], [1.2, 0.3], [1.8, -0.35]]
    static = _TracerProblem(vector(0.7, 0.3), seeds)
    _t1, _s1, end_static, _T1 = _run(static, "combo_static_" + name)
    moving = _TracerProblem(vector(0.7, 0.3), seeds, motion=motion)
    _t2, _s2, end_moving, _T2 = _run(moving, "combo_moving_" + name)
    assert len(end_static) == len(seeds) and len(end_moving) == len(seeds)
    assert numpy.max(numpy.abs(end_moving - end_static)) < 1e-11


def test_nonuniform_advection_on_moving_mesh_converges_to_the_static_answer():
    """Poiseuille on a moving mesh, against the same field on a static one.

    This one is NOT exact, and the reason is worth stating: the advection field is blended over the
    nodal time-history levels, which is exactly right for a solved FE field (it is then the field
    with time-interpolated nodal values) but only approximate for an analytic function of the
    coordinates, because sum_k w_k f(x_k) is not f(sum_k w_k x_k) unless f is affine. The error is
    quadratic in the per-step mesh displacement, i.e. the same order as the in-step interpolation of
    the mesh configuration itself - so it converges away with dt rather than being a defect.
    """
    errs = []
    for dt in (0.1, 0.05, 0.025):
        static = _TracerProblem(POISEUILLE, SEEDS)
        _t1, _s1, end_static, _T1 = _run(static, "conv_static_%g" % dt, dt=dt)
        moving = _TracerProblem(POISEUILLE, SEEDS, motion=_squeeze)
        _t2, _s2, end_moving, _T2 = _run(moving, "conv_moving_%g" % dt, dt=dt)
        errs.append(float(numpy.max(numpy.abs(end_moving - end_static))))
    orders = [math.log(errs[i] / errs[i + 1], 2) for i in range(len(errs) - 1)]
    assert min(orders) > 1.7, "observed orders %s from errors %s" % (orders, errs)
    assert errs[-1] < 1e-4


# ----------------------------------------------------------------------------------------------
# E - order of the in-step interpolation of the mesh configuration
# ----------------------------------------------------------------------------------------------

def test_time_interpolation_order_is_honoured():
    """The in-step Lagrange interpolation of the nodal positions, measured where it actually shows.

    Getting a case where it shows at all takes some care, and the two traps are worth recording:

      * in the BULK the stage derivative is v itself, so with v = 0 a particle stays put at any
        interpolation order and the test passes vacuously;
      * a SPATIALLY UNIFORM interface motion telescopes - the integral of dX/dtau over the step is
        X(1) - X(0) whatever the interpolant does in between, as long as it hits both endpoints.

    So the interface has to deform non-uniformly AND the particle has to travel along it, so that
    which part of the interface it co-moves with depends on where it is at each instant. Then a
    motion quadratic in time is reproduced exactly by the quadratic interpolant and not by the
    linear one.
    """
    def wave_quadratic_in_t(x, y, t):
        amp = 0.3 * t * t
        return (x, y + (0.5 * (y + 1.0)) * amp * math.sin(0.9 * x))

    seeds = [[0.8, 1.0], [1.9, 1.0]]

    def run_at(order, dt, tag):
        p = _TracerProblem(vector(1, 0), seeds, motion=wave_quadratic_in_t, on_interface="top",
                           time_interpolation_order=order, N=(24, 4))
        _tr, _start, end, _T = _run(p, tag, endtime=0.6, dt=dt)
        assert len(end) == len(seeds)
        return end

    reference = run_at(2, 0.0125, "tinterp_ref")
    errs = {order: float(numpy.max(numpy.abs(run_at(order, 0.2, "tinterp_%d" % order) - reference)))
            for order in (1, 2)}
    assert errs[1] > 10 * errs[2], \
        "quadratic interpolation was not measurably better (linear %g, quadratic %g)" % (errs[1], errs[2])


# ----------------------------------------------------------------------------------------------
# F - interface confinement
# ----------------------------------------------------------------------------------------------

SURF_SEEDS = [[0.5, 1.0], [1.4, 1.0], [2.3, 1.0]]


def test_interface_tangential_slide_does_not_move_tracers():
    """F1, the sharpest test of the tangential ALE correction: nodes slide along the interface under
    a stationary particle. Getting the correction wrong gives a drift of exactly the slide rate."""
    p = _TracerProblem(vector(0, 0), SURF_SEEDS, motion=_slide_x, on_interface="top")
    tr, start, end, _T = _run(p, "surf_slide")
    assert tr.nlocal() == len(SURF_SEEDS)
    assert numpy.max(numpy.abs(end - start)) < 1e-13


def test_interface_normal_advection_moves_nothing():
    """F2: only the tangential component advects, and on a horizontal interface (0,1) has none."""
    p = _TracerProblem(vector(0, 1), SURF_SEEDS, on_interface="top")
    tr, start, end, _T = _run(p, "surf_normalv")
    assert numpy.max(numpy.abs(end - start)) < 1e-13


def test_interface_tangential_advection_is_independent_of_slide():
    """F3: the answer must not depend on how the nodes happen to be parameterised."""
    ends = []
    for name, motion in (("static", None), ("sliding", _slide_x)):
        p = _TracerProblem(vector(1, 0), SURF_SEEDS, motion=motion, on_interface="top")
        _tr, start, end, T = _run(p, "surf_tang_" + name)
        ends.append(end)
        assert numpy.max(numpy.abs(end[:, 0] - (start[:, 0] + T))) < 1e-11
        assert numpy.max(numpy.abs(end[:, 1] - start[:, 1])) < 1e-13
    assert numpy.max(numpy.abs(ends[0] - ends[1])) < 1e-11


def test_interface_comoves_normally():
    """F5: the interface translates in y, the particle must follow it exactly while its tangential
    position stays put."""
    p = _TracerProblem(vector(0, 0), SURF_SEEDS, motion=_lift_y, on_interface="top")
    tr, start, end, T = _run(p, "surf_lift")
    assert numpy.max(numpy.abs(end[:, 0] - start[:, 0])) < 1e-13
    assert numpy.max(numpy.abs(end[:, 1] - (start[:, 1] + 0.1 * T))) < 1e-12


def test_interface_normal_offset_stays_at_machine_zero_on_a_deforming_interface():
    """F6: the interface both curves and deforms, so nothing about this is exact except the
    constraint itself - the particle must never leave the surface. Measured independently of the
    tracer code, through the point locator's projection offset."""
    amp, omega = 0.15, 2.0

    def wave(x, y, t):
        # Only the top row moves, and it moves in y, so the interface curves and deforms in time.
        return (x, y + (0.5 * (y + 1.0)) * amp * math.sin(math.pi * x) * math.cos(omega * t))

    p = _TracerProblem(vector(0, 0), SURF_SEEDS, motion=wave, on_interface="top", N=(16, 4))
    p.set_output_directory("_tracer_surf_wave")
    p.quiet()
    p.initialise()
    tr = p.tracers()
    imesh = p.get_mesh("domain/top")
    worst = 0.0
    for _ in range(20):
        p.run(p.get_current_time() + 0.05, timestep=0.05, outstep=False, startstep=0.05)
        pos = tr.get_positions()
        located = numpy.array(imesh.locate_points(pos, lagrangian=False), dtype=float)
        assert numpy.all(located[:, 0] > 0.5), "a tracer left the interface mesh entirely"
        worst = max(worst, float(numpy.max(numpy.abs(located[:, 1]))))
    assert worst < 1e-12, "normal offset from the interface grew to %g" % worst


# ----------------------------------------------------------------------------------------------
# G - seeding
# ----------------------------------------------------------------------------------------------

class _HoledProblem(Problem):
    """Two stacked rectangles with a gap between them, so a bounding-box lattice proposes points
    that are in no element at all."""

    def __init__(self, seed):
        super().__init__()
        self.seed = seed

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(size=[1, 1], lower_left=[0, 0], N=[6, 6], name="lower"))
        self.add_mesh(RectangularQuadMesh(size=[1, 1], lower_left=[0, 2], N=[6, 6], name="upper"))
        for d in ("lower", "upper"):
            self += (PoissonEquation(source=0) + DirichletBC(u=0) @ "left"
                     + TracerParticles(vector(0, 0), seed=self.seed)) @ d


def test_grid_seed_rejects_points_outside_the_mesh():
    p = _HoledProblem(GridSeed(0.2, bbox=([0.0, 0.0], [1.0, 3.0]), inset=0.0))
    p.set_output_directory("_tracer_seed_holed")
    p.quiet()
    p.initialise()
    pos = p.get_mesh("lower").get_tracers().get_positions()
    assert len(pos) > 0
    # Every surviving candidate is in the lower block; the ones in the gap and in the upper block
    # were proposed by the bounding box and rejected by the containment check.
    assert numpy.all(pos[:, 1] <= 1.0 + 1e-12)


class _Seed3dProblem(Problem):
    def __init__(self, seed):
        super().__init__()
        self.seed = seed

    def define_problem(self):
        self.add_mesh(CuboidBrickMesh(size=[1, 1, 1], N=4))
        self += (PoissonEquation(source=0) + DirichletBC(u=0) @ "left"
                 + TracerParticles(vector(0, 0, 0), seed=self.seed)) @ "domain"


@pytest.mark.parametrize("seed,name", [(GridSeed(0.3), "grid"),
                                       (ElementSeed(), "element"),
                                       (RandomSeed(50, rng_seed=3), "random")])
def test_seeding_works_in_3d(seed, name):
    """The old implementation raised outright for anything but 2d, both in the seeding and in the
    element-exit test."""
    p = _Seed3dProblem(seed)
    p.set_output_directory("_tracer_seed3d_" + name)
    p.quiet()
    p.initialise()
    tr = p.get_mesh("domain").get_tracers()
    pos = tr.get_positions()
    assert tr.nlocal() > 0
    assert pos.shape[1] == 3
    assert numpy.all(pos >= -1e-12) and numpy.all(pos <= 1.0 + 1e-12)


def test_tracers_work_in_3d():
    """Advection in 3d at all, which the old element-exit predicate made impossible."""
    class P(Problem):
        def define_problem(self):
            self.add_mesh(CuboidBrickMesh(size=[1, 1, 1], N=4))
            adv = vector(1, 0.5, 0.25)
            self += (PoissonEquation(source=0) + DirichletBC(u=0) @ "left"
                     + TracerParticles(adv, seed=PointSeed([[0.1, 0.2, 0.3]]),
                                       rtol=1e-11, atol=1e-13)) @ "domain"
    p = P()
    p.set_output_directory("_tracer_3d")
    p.quiet()
    p.initialise()
    tr = p.get_mesh("domain").get_tracers()
    start = tr.get_positions().copy()
    p.run(0.5, timestep=0.05, outstep=False, startstep=0.05)
    end = tr.get_positions()
    assert tr.nlocal() == 1
    expected = start[0] + numpy.array([1.0, 0.5, 0.25]) * 0.5
    assert numpy.max(numpy.abs(end[0] - expected)) < 1e-11


# ----------------------------------------------------------------------------------------------
# H - the advection fires once per accepted timestep
# ----------------------------------------------------------------------------------------------

def test_advection_happens_once_per_timestep_not_once_per_newton_solve():
    """A stationary solve must not advect at all, and a transient step must advect exactly once
    however many Newton solves it took. The old code hooked into after_newton_solve, so an
    adaptation re-solve or an arclength step moved the particles again."""
    p = _TracerProblem(vector(1, 0), [[0.5, 0.0]])
    p.set_output_directory("_tracer_hooks")
    p.quiet()
    p.initialise()
    tr = p.tracers()
    start = tr.get_positions().copy()

    p.solve()  # stationary
    assert numpy.max(numpy.abs(tr.get_positions() - start)) < 1e-14

    for _ in range(4):
        p.solve(timestep=0.1)
    end = tr.get_positions()
    assert abs(float(end[0, 0]) - (float(start[0, 0]) + 0.4)) < 1e-11

    p.solve()  # stationary again: still no movement
    assert numpy.max(numpy.abs(tr.get_positions() - end)) < 1e-14


# ----------------------------------------------------------------------------------------------
# J - path-integrated payloads
# ----------------------------------------------------------------------------------------------

def test_payload_residence_time_is_exact():
    """dp/dt = 1 integrates to t exactly for any Runge-Kutta method."""
    p = _TracerProblem(POISEUILLE, SEEDS, payloads={"residence": 1})
    tr, _start, _end, T = _run(p, "payload_residence", endtime=0.6, dt=0.05)
    pay = tr.get_payloads()
    assert pay.shape == (len(SEEDS), 1)
    assert numpy.max(numpy.abs(pay[:, 0] - T)) < 1e-11


def test_payload_of_a_field_constant_along_the_path_is_exact():
    """In Poiseuille flow a particle keeps its y, hence its own v_x, so the accumulated speed is
    exactly v_x * t."""
    p = _TracerProblem(POISEUILLE, SEEDS, payloads={"path": 1 - var("coordinate_y") ** 2})
    tr, start, _end, T = _run(p, "payload_speed", endtime=0.6, dt=0.05)
    pay = tr.get_payloads()
    expected = (1.0 - start[:, 1] ** 2) * T
    assert numpy.max(numpy.abs(pay[:, 0] - expected)) < 1e-11


# ----------------------------------------------------------------------------------------------
# History
# ----------------------------------------------------------------------------------------------

def test_position_history_is_recorded_and_windowed():
    p = _TracerProblem(POISEUILLE, SEEDS, history_time=0.2)
    tr, _start, _end, _T = _run(p, "history", endtime=1.0, dt=0.05)
    ids = tr.get_ids()
    hist = tr.get_history(int(ids[0]))
    assert hist.shape[1] == 3  # (t, x, y)
    assert hist.shape[0] >= 2
    tspan = float(hist[-1, 0] - hist[0, 0])
    assert tspan <= 0.2 + 1e-12, "the rolling window was not pruned (span %g)" % tspan
    assert numpy.all(numpy.diff(hist[:, 0]) > 0), "history is not in chronological order"
    # The last sample is the current position.
    assert numpy.max(numpy.abs(hist[-1, 1:] - tr.get_positions()[0])) < 1e-14


def test_history_is_empty_without_a_window():
    p = _TracerProblem(POISEUILLE, SEEDS)
    tr, _start, _end, _T = _run(p, "history_off", endtime=0.2, dt=0.05)
    assert tr.get_history(int(tr.get_ids()[0])).size == 0


# ----------------------------------------------------------------------------------------------
# Identity and removal
# ----------------------------------------------------------------------------------------------

def test_ids_are_unique_stable_and_not_recycled():
    p = _TracerProblem(POISEUILLE, SEEDS)
    p.set_output_directory("_tracer_ids")
    p.quiet()
    p.initialise()
    tr = p.tracers()
    ids = list(tr.get_ids())
    assert len(set(ids)) == len(ids)
    assert tr.remove_tracer(int(ids[1]))
    assert not tr.remove_tracer(int(ids[1]))
    newid = tr.add_tracer([2.0, 0.1])
    assert newid not in ids, "a removed particle's id was handed out again"
    p.run(0.2, timestep=0.05, outstep=False, startstep=0.05)
    after = list(tr.get_ids())
    assert ids[0] in after and ids[1] not in after and newid in after


def test_add_tracer_outside_the_mesh_is_reported_not_silently_placed():
    p = _TracerProblem(POISEUILLE, [[0.5, 0.0]])
    p.set_output_directory("_tracer_outside")
    p.quiet()
    p.initialise()
    tr = p.tracers()
    assert tr.add_tracer([100.0, 100.0]) == 0
    assert tr.nlocal() == 1


# ----------------------------------------------------------------------------------------------
# State files
# ----------------------------------------------------------------------------------------------

def _state_problem(tag):
    p = _TracerProblem(POISEUILLE, SEEDS, payloads={"residence": 1})
    p.set_output_directory("_tracer_" + tag)
    p.quiet()
    return p


def test_state_file_round_trip_preserves_positions_ids_and_payloads(tmp_path):
    """Identities have to survive, not just positions: they are what makes a dump comparable across
    a restart at all, and under MPI what makes the file partition-independent."""
    a = _state_problem("state_save")
    a.initialise()
    a.run(0.3, timestep=0.05, outstep=False, startstep=0.05)
    tr_a = a.tracers()
    pos, ids, pay = (tr_a.get_positions().copy(), list(tr_a.get_ids()), tr_a.get_payloads().copy())
    dump = str(tmp_path / "tracers.dump")
    a.save_state(dump)

    b = _state_problem("state_load")
    b.initialise()
    b.load_state(dump)
    tr_b = b.tracers()
    assert list(tr_b.get_ids()) == ids
    assert numpy.max(numpy.abs(tr_b.get_positions() - pos)) == 0.0
    assert numpy.max(numpy.abs(tr_b.get_payloads() - pay)) == 0.0

    # and the restored particles keep advecting correctly
    b.run(b.get_current_time() + 0.2, timestep=0.05, outstep=False, startstep=0.05)
    a.run(a.get_current_time() + 0.2, timestep=0.05, outstep=False, startstep=0.05)
    assert numpy.max(numpy.abs(tr_b.get_positions() - tr_a.get_positions())) < 1e-12


# ----------------------------------------------------------------------------------------------
# K - handover between two domains sharing an interface
# ----------------------------------------------------------------------------------------------

class _StackedMesh(MeshTemplate):
    """One rectangle split at y = 1 into two domains that SHARE the nodes on the interface.

    Two separate meshes would not do: a transfer interface needs the two sides to be opposite sides
    of one interface, which is a property of the nodes being the same.
    """

    def __init__(self, N=6):
        super().__init__()
        self.N = N

    def define_geometry(self):
        lower = self.new_domain("lower")
        upper = self.new_domain("upper")
        N = self.N
        nodes = {}
        for i in range(N + 1):
            for j in range(2 * N + 1):
                nodes[(i, j)] = self.add_node_unique(i / N, j / N)
        for i in range(N):
            for j in range(2 * N):
                dom = lower if j < N else upper
                dom.add_quad_2d_C1(nodes[(i, j)], nodes[(i + 1, j)], nodes[(i, j + 1)], nodes[(i + 1, j + 1)])
        for i in range(N):
            self.add_facet_to_boundary("interface", [nodes[(i, N)], nodes[(i + 1, N)]])
            self.add_facet_to_boundary("bottom", [nodes[(i, 0)], nodes[(i + 1, 0)]])
            self.add_facet_to_boundary("top", [nodes[(i, 2 * N)], nodes[(i + 1, 2 * N)]])
        for j in range(2 * N):
            self.add_facet_to_boundary("left", [nodes[(0, j)], nodes[(0, j + 1)]])
            self.add_facet_to_boundary("right", [nodes[(N, j)], nodes[(N, j + 1)]])


def _two_domain_counts(with_transfer):
    from pyoomph.equations.tracers import TracerTransferAtInterface

    # All well below the interface, and far enough that 0.8 units of upward travel puts every one
    # of them clearly inside the upper domain rather than on the interface itself.
    seeds = [[0.3, 0.45], [0.6, 0.6], [0.85, 0.3]]

    class P(Problem):
        def define_problem(self):
            self.add_mesh(_StackedMesh())
            adv = vector(0, 1)
            lower = PoissonEquation(source=0) + DirichletBC(u=0) @ "bottom"
            lower += TracerParticles(adv, seed=PointSeed(seeds), rtol=1e-11, atol=1e-13)
            upper = PoissonEquation(source=0) + DirichletBC(u=0) @ "top"
            upper += TracerParticles(adv, seed=None, rtol=1e-11, atol=1e-13)
            if with_transfer:
                lower += TracerTransferAtInterface() @ "interface"
            self += lower @ "lower"
            self += upper @ "upper"

    p = P()
    p.set_output_directory("_tracer_twodomain_" + ("on" if with_transfer else "off"))
    p.quiet()
    p.initialise()
    tr_lo = p.get_mesh("lower").get_tracers()
    tr_up = p.get_mesh("upper").get_tracers()
    start = tr_lo.get_positions().copy()
    t0 = float(p.get_current_time(as_float=True, dimensional=False))
    p.run(0.8, timestep=0.05, outstep=False, startstep=0.05)
    elapsed = float(p.get_current_time(as_float=True, dimensional=False)) - t0
    return start, elapsed, tr_lo, tr_up


def test_tracers_are_handed_over_between_domains():
    """Without the handover a particle reaching the edge of its domain is simply dropped, which is
    also what makes the comparison below meaningful."""
    start, elapsed, tr_lo, tr_up = _two_domain_counts(with_transfer=True)
    assert tr_lo.nlocal() == 0, "particles should all have left the lower domain"
    assert tr_up.nlocal() == len(start), "%d of %d particles survived the handover" % (
        tr_up.nlocal(), len(start))
    end = tr_up.get_positions()
    order = numpy.argsort(end[:, 0])
    expected = start[numpy.argsort(start[:, 0])]
    end = end[order]
    assert numpy.max(numpy.abs(end[:, 0] - expected[:, 0])) < 1e-11
    assert numpy.max(numpy.abs(end[:, 1] - (expected[:, 1] + elapsed))) < 1e-10


def test_without_a_transfer_interface_the_particles_are_dropped():
    """The counterpart of the above, so that the handover test cannot pass by accident."""
    _start, _elapsed, tr_lo, tr_up = _two_domain_counts(with_transfer=False)
    assert tr_lo.nlocal() == 0
    assert tr_up.nlocal() == 0
