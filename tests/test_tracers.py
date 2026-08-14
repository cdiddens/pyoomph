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
from pyoomph.equations.tracers import (TracerParticles, TracerSeedPoints, TracerSeedGrid, TracerSeedElement,
                                       TracerPeriodicBoundaryCondition,
                                       TracerSeedRandom, TracerSeedCallable)
from pyoomph.meshes.simplemeshes import RectangularQuadMesh, CuboidBrickMesh, LineMesh


# The ~70 relative "_tracer_*" output directories below land under pytest's tmp_path, not in the
# repository: see the _output_below_tmp_path fixture in conftest.py.

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
                 time_interpolation_order="auto", history_time=None, shear=0.0, origin_x=0.0):
        super().__init__()
        # Where the domain sits along x. Only ever moved away from the origin to make the rounding
        # noise of the coordinates, which scales with |x|, large enough to matter.
        self.origin_x = origin_x
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
        self.add_mesh(RectangularQuadMesh(size=[4, 2], lower_left=[self.origin_x, -1], N=list(self.N)))
        eqs = PoissonEquation(source=0) + DirichletBC(u=0) @ "left" + DirichletBC(u=1) @ "right"
        tp = TracerParticles(self.advection, seed=TracerSeedPoints(self.seeds),
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


@pytest.mark.parametrize("offset", [0.0, 1.0e3], ids=["at_origin", "far_from_origin"])
def test_no_particle_is_lost_in_the_interior_of_a_moving_mesh(offset):
    """Nothing may be dropped while it is still comfortably inside the mesh.

    The threshold on the Newton step that places a particle in its element used to be a fixed 1e-14
    in reference coordinates. What that iteration can actually reach is the rounding noise of the
    residual, eps*|x|, divided by the element scale |dX/ds| - which for a coarse mesh a few units
    from the origin is already of that order and beyond it further out. The iteration then spent all
    its rounds bouncing on the floor, the placement was reported as failed, and interior particles
    disappeared one by one, the more of them the further from the origin. Hence the offset
    parametrisation: at the origin the old code passes, far from it it loses most of the particles.
    """
    def wobble(x, y, t):
        return (x + 0.05 * math.sin(1.3 * t) * (y + 1.0), y + 0.07 * math.sin(0.9 * t) * math.sin(x - offset))

    # Away from the outflow edge, so a particle leaving through it cannot be mistaken for a loss.
    seeds = [[offset + 0.4 + 0.13 * i, -0.8 + 0.11 * i] for i in range(12)]
    p = _TracerProblem(vector(0.2, 0), seeds, motion=wobble, N=(16, 8), origin_x=offset)
    tr, _start, end, _T = _run(p, "interior_loss_%g" % offset, endtime=1.0, dt=0.05)
    assert len(end) == len(seeds), "%d of %d particles were lost inside the mesh" % (
        len(seeds) - len(end), len(seeds))


def test_a_moved_mesh_invalidates_the_cached_point_locator():
    """A locator freezes the nodal positions it was built from, so caching it on the mesh's topology
    generation alone is not enough: the geometry moves without the topology changing. Adding a
    particle near the moved boundary then went through a locator describing where the mesh used to
    be, and the point was reported as outside the mesh."""
    def rise(x, y, t):
        return (x, y + 0.6 * t)

    p = _TracerProblem(vector(0, 0), [[2.0, 0.0]], motion=rise, N=(8, 4))
    tr, _start, _end, _T = _run(p, "locator_staleness", endtime=1.0, dt=0.25)
    # The mesh now spans y in [-0.4, 1.6]. A point above the ORIGINAL top edge is inside it, and
    # only a locator that noticed the motion can say so.
    assert tr.add_tracer([2.0, 1.4]) != 0, "a point inside the moved mesh was reported as outside"
    assert tr.add_tracer([2.0, 1.8]) == 0, "a point above the moved mesh was accepted"


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


@pytest.mark.parametrize("at_end,speed", [(False, 1e-2), (True, 1.0), (True, 1e-6)],
                         ids=["mid_curve", "at_the_end_fast", "at_the_end_slow"])
def test_an_interface_tracer_at_the_end_of_the_curve_is_pinned_there(at_end, speed):
    """A confined particle cannot leave its interface, only reach the end of it - so it has to be
    pinned there, not dropped and not spun on forever.

    The geometry is the apex of an evaporating droplet in miniature: a curved interface whose
    highest point is also the end of the curve, descending, with a particle sitting exactly on that
    end and a tangential advection pushing it off. The two speeds are the two ways the sub-step
    controller finds out that the particle is stuck, and they need different halves of the fix:

      * fast - the overshoot past the end does not shrink with the sub-step, so h collapses through
        1e-12 in about forty halvings and the "the particle is leaving" branch is reached as it
        always was. Only the pinning is new here.
      * slow - the overshoot IS proportional to the sub-step, so the halved step is accepted, the
        controller grows h straight back because the error estimate is tiny, and tau crawls forward
        at rounding scale. Without the progress watchdog this runs a million sub-steps and then
        raises; it was found at the apex of a real evaporating droplet.

    Either way the particle stays. Dropping it - which is what a bulk particle leaving its domain
    gets - would delete a particle that never left anything. The `mid_curve` case passes whatever
    happens at the ends and is here to show that an ordinary interface particle is untouched.
    """
    def descending_dome(x, y, t):
        # A curved top whose apex sits on x = 0, which is also the end of the interface.
        return (x, y - (0.5 * (y + 1.0)) * (0.3 * t) * math.cos(0.35 * x))

    seeds = [[0.0 if at_end else 1.2, 1.0]]
    p = _TracerProblem(vector(-speed, 0), seeds, motion=descending_dome, on_interface="top",
                       N=(16, 4))
    tag = "surf_end_%s_%g" % ("on" if at_end else "off", speed)
    tr, _start, end, _T = _run(p, tag, endtime=0.2, dt=0.05)

    assert len(end) == 1, "the particle was dropped instead of being kept on the interface"
    if at_end:
        assert tr.get_pins_last_step() >= 1, \
            "nothing was pinned, so this no longer exercises the end of the curve"
        # Pinned means pinned AT the end, not slid back along the interface.
        assert abs(end[0][0]) < 1e-9, "the pinned particle left the end (x = %g)" % end[0][0]
    else:
        assert tr.get_pins_last_step() == 0, "an ordinary interface particle must not be pinned"

    # And it is still ON the interface either way, which is the invariant the whole formulation
    # exists for. Measured through the point locator's projection offset, independently of the
    # tracer code, exactly as the deforming-interface test above does.
    located = numpy.array(p.get_mesh("domain/top").locate_points(end, lagrangian=False), dtype=float)
    assert located[0][0] > 0.5, "the particle is not on the interface mesh at all"
    assert abs(located[0][1]) < 1e-9, "normal offset from the interface is %g" % located[0][1]


def test_every_sub_step_setting_is_reachable_from_python():
    """A runaway that can only be diagnosed by editing C++ is not diagnosable, so every tunable of
    the collection is settable from the equation and stays writable on the collection afterwards."""
    class P(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(size=[4, 2], lower_left=[0, -1], N=[8, 4]))
            eqs = PoissonEquation(source=0) + DirichletBC(u=0) @ "left"
            eqs += TracerParticles(POISEUILLE, seed=TracerSeedPoints(SEEDS),
                                   rtol=1e-7, atol=1e-9, history_capacity=17,
                                   time_interpolation_order=1, fixed_substeps=3,
                                   max_substeps=4321, max_migration_rounds=5,
                                   max_periodic_wraps=2)
            self += eqs @ "domain"

    p = P()
    p.set_output_directory("_tracer_settings")
    p.quiet()
    p.initialise()
    tr = p.get_mesh("domain").get_tracers()
    assert (tr.rtol, tr.atol, tr.history_capacity) == (1e-7, 1e-9, 17)
    assert tr.time_interpolation_order == 1
    assert (tr.fixed_substeps, tr.max_substeps) == (3, 4321)
    assert (tr.max_migration_rounds, tr.max_periodic_wraps) == (5, 2)
    tr.max_substeps = 10
    assert tr.max_substeps == 10


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
    p = _HoledProblem(TracerSeedGrid(0.2, bbox=([0.0, 0.0], [1.0, 3.0]), inset=0.0))
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


@pytest.mark.parametrize("seed,name", [(TracerSeedGrid(0.3), "grid"),
                                       (TracerSeedElement(), "element"),
                                       (TracerSeedRandom(50, rng_seed=3), "random")])
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
                     + TracerParticles(adv, seed=TracerSeedPoints([[0.1, 0.2, 0.3]]),
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


def test_state_file_round_trip_preserves_the_position_history(tmp_path):
    """The rolling history is what the trail plots are drawn from, and it used not to be written at
    all: a restored state came back with every particle in the right place and no trail, which then
    grew back from scratch instead of continuing."""
    def problem(tag):
        p = _TracerProblem(POISEUILLE, SEEDS, history_time=0.4)
        p.set_output_directory("_tracer_" + tag)
        p.quiet()
        return p

    a = problem("hist_save")
    a.initialise()
    a.run(0.3, timestep=0.05, outstep=False, startstep=0.05)
    tr_a = a.tracers()
    ids = list(tr_a.get_ids())
    hist = {int(i): tr_a.get_history(int(i)).copy() for i in ids}
    assert min(h.shape[0] for h in hist.values()) >= 3, "nothing to compare - no history was recorded"
    dump = str(tmp_path / "hist.dump")
    a.save_state(dump)

    b = problem("hist_load")
    b.initialise()
    b.load_state(dump)
    tr_b = b.tracers()
    assert list(tr_b.get_ids()) == ids
    for i in ids:
        got = tr_b.get_history(int(i))
        assert got.shape == hist[int(i)].shape
        assert numpy.max(numpy.abs(got - hist[int(i)])) == 0.0

    # and it continues rather than restarts: the next step appends to what was restored
    n_before = tr_b.get_history(int(ids[0])).shape[0]
    b.run(b.get_current_time() + 0.05, timestep=0.05, outstep=False, startstep=0.05)
    assert tr_b.get_history(int(ids[0])).shape[0] == n_before + 1


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
            lower += TracerParticles(adv, seed=TracerSeedPoints(seeds), rtol=1e-11, atol=1e-13)
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


# ----------------------------------------------------------------------------------------------
# L1 - bulk to its own interface, the evaporating-surface case
# ----------------------------------------------------------------------------------------------

_SWALLOW_SEEDS = [[1.0, 0.5], [2.0, 0.6], [3.0, 0.7]]


def _swallowed_by_the_surface(with_transfer):
    """A boundary that recedes over stationary particles - a free surface losing mass, in miniature.

    The advection field is zero, so the particles do not swim out; the surface comes to them. That
    is the whole point of the transfer: a parcel the surface has caught up with belongs ON it.
    """
    from pyoomph.equations.tracers import TracerTransferToInterface

    def descending_top(x, y, t):
        return (x, y - 0.6 * t * (0.5 * (y + 1.0)))

    class P(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(size=[4, 2], lower_left=[0, -1], N=[8, 4]))
            self._X0 = None
            eqs = PoissonEquation(source=0) + DirichletBC(u=0) @ "left"
            eqs += TracerParticles(vector(0, 0), seed=TracerSeedPoints(_SWALLOW_SEEDS),
                                   rtol=1e-11, atol=1e-13)
            surf = TracerParticles(vector(0, 0), tracer_name="surface", seed=None,
                                   rtol=1e-11, atol=1e-13)
            if with_transfer:
                surf = surf + TracerTransferToInterface()
            eqs += surf @ "top"
            self += eqs @ "domain"

        def actions_before_newton_solve(self):
            super().actions_before_newton_solve()
            m = self.get_mesh("domain")
            if self._X0 is None:
                self._X0 = numpy.array([[n.x(0), n.x(1)] for n in m.nodes()])
            t = float(self.get_current_time(as_float=True, dimensional=False))
            for n, X in zip(m.nodes(), self._X0):
                x, y = descending_top(X[0], X[1], t)
                n.set_x(0, x)
                n.set_x(1, y)

    p = P()
    p.set_output_directory("_tracer_swallow_" + ("on" if with_transfer else "off"))
    p.quiet()
    p.initialise()
    bulk = p.get_mesh("domain").get_tracers()
    surf = p.get_mesh("domain/top").get_tracers("surface")
    ids = list(bulk.get_ids())
    p.run(1.0, timestep=0.05, outstep=False, startstep=0.05)
    return bulk, surf, ids


def test_particles_the_surface_catches_up_with_are_transferred_onto_it():
    """They keep their identity, which is what says they were handed over rather than re-seeded."""
    bulk, surf, ids = _swallowed_by_the_surface(with_transfer=True)
    assert bulk.nlocal() == 0, "the receding surface should have caught up with all of them"
    assert sorted(surf.get_ids()) == sorted(ids), "the particles did not arrive with their identity"
    # And they are on the interface, not merely somewhere near it.
    pos = surf.get_positions()
    assert pos.shape == (len(ids), 2)
    assert numpy.max(numpy.abs(pos[:, 1] - numpy.max(pos[:, 1]))) < 1e-9, \
        "the transferred particles are not on the (flat) interface"


def test_without_the_transfer_the_swallowed_particles_are_dropped():
    """The counterpart, so the test above cannot pass by accident."""
    bulk, surf, _ids = _swallowed_by_the_surface(with_transfer=False)
    assert bulk.nlocal() == 0
    assert surf.nlocal() == 0


# ----------------------------------------------------------------------------------------------
# L2 - a trail outliving its particle
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("history_time", [0.5, None], ids=["with_history", "without_history"])
def test_a_trail_outlives_the_particle_that_drew_it(history_time):
    """A particle that leaves the domain is gone from the simulation, but its trail is not gone from
    the picture: it fades out over the history window instead of blinking out with the marker. With
    no window there is no trail, so the particle is deleted outright as it always was."""
    p = _TracerProblem(POISEUILLE, [[3.9, 0.0]], history_time=history_time)
    p.set_output_directory("_tracer_afterlife_%s" % history_time)
    p.quiet()
    p.initialise()
    tr = p.tracers()
    tid = int(tr.get_ids()[0])

    # v = 1 on the axis, so it is out through x = 4 well within this.
    p.run(0.3, timestep=0.05, outstep=False, startstep=0.05)
    assert tr.nlocal() == 0, "the particle should have left the domain by now"
    assert list(tr.get_ids()) == [], "a dead particle must not appear among the living"
    if history_time is None:
        assert tr.nlocal_dead() == 0, "without a window there is no trail to keep it around for"
        assert tr.get_history(tid).size == 0
        return

    assert tr.nlocal_dead() == 1, "the trail was thrown away with the particle"
    assert [int(i) for i in tr.get_dead_ids()] == [tid]
    hist = tr.get_history(tid)
    assert hist.shape[0] >= 2, "the trail of a dead particle has to still be readable"
    assert hist.shape[1] == 3

    # ... and it really does go away once it has aged out of the window, rather than accumulating.
    p.run(p.get_current_time() + 1.5 * history_time, timestep=0.05, outstep=False, startstep=0.05)
    assert tr.nlocal_dead() == 0, "the faded trail was never forgotten"
    assert tr.get_history(tid).size == 0


# ----------------------------------------------------------------------------------------------
# L - periodic re-injection
# ----------------------------------------------------------------------------------------------

_PERIODIC_L = 4.0
_PERIODIC_SEEDS = [[0.5, 0.2], [1.3, -0.6], [2.1, 0.4], [3.6, 0.8]]


def _periodic_run(tag, with_wrap, endtime=6.0, dt=0.05, both_directions=True):
    """Uniform advection down a channel of length 4, so the analytic answer is (x0 + T) mod 4."""
    class P(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(size=[_PERIODIC_L, 2], lower_left=[0, -1], N=[16, 8]))
            eqs = PoissonEquation(source=0) + DirichletBC(u=0) @ "left"
            eqs += TracerParticles(vector(1, 0), seed=TracerSeedPoints(_PERIODIC_SEEDS),
                                   history_time=0.5, payloads={"residence": 1},
                                   rtol=1e-11, atol=1e-13)
            if with_wrap:
                eqs += TracerPeriodicBoundaryCondition(vector(-_PERIODIC_L, 0),
                                                       both_directions=both_directions) @ "right"
            self += eqs @ "domain"

    p = P()
    p.set_output_directory("_tracer_periodic_" + tag)
    p.quiet()
    p.initialise()
    tr = p.get_mesh("domain").get_tracers()
    ids = list(tr.get_ids())
    t0 = float(p.get_current_time(as_float=True, dimensional=False))
    p.run(endtime, timestep=dt, outstep=False, startstep=dt)
    elapsed = float(p.get_current_time(as_float=True, dimensional=False)) - t0
    return tr, ids, elapsed


def test_periodic_boundary_reinjects_particles_at_the_other_end():
    """A wrapped particle finishes the rest of its timestep from the image rather than stopping at
    the boundary, so the answer stays exact across the jump - which is the whole reason the wrap is
    applied inside the advection rather than as a position fix-up afterwards."""
    tr, ids, elapsed = _periodic_run("wrap", with_wrap=True)
    assert list(tr.get_ids()) == ids, "the wrap must preserve identity, not re-seed particles"
    end = tr.get_positions()
    expect = numpy.array([[(s[0] + elapsed) % _PERIODIC_L, s[1]] for s in _PERIODIC_SEEDS])
    assert numpy.max(numpy.abs(end - expect)) < 1e-11
    # The payload is path-integrated and knows nothing about the jump: every particle has been in
    # the domain for the whole run.
    assert numpy.max(numpy.abs(tr.get_payloads()[:, 0] - elapsed)) < 1e-10


def test_without_a_periodic_boundary_the_particles_leave():
    """The counterpart, so the test above cannot pass by accident on a run where nothing wrapped."""
    tr, _ids, _elapsed = _periodic_run("nowrap", with_wrap=False)
    assert tr.nlocal() == 0


def test_a_periodic_wrap_restarts_the_trail_rather_than_drawing_it_across_the_domain():
    """A trail is a path through the plotted coordinates and a wrapped path is not continuous
    there, so the history starts again at the image instead of keeping the samples from the far
    end - which would draw one line straight back across the whole domain."""
    tr, ids, _elapsed = _periodic_run("trail", with_wrap=True, endtime=3.9)
    longest = 0.0
    for i in ids:
        h = tr.get_history(int(i))
        if len(h) > 1:
            longest = max(longest, float(numpy.max(numpy.abs(numpy.diff(h[:, 1])))))
    assert longest < 0.5 * _PERIODIC_L, "a trail segment spans the domain (%g)" % longest
    assert min(len(tr.get_history(int(i))) for i in ids) < max(len(tr.get_history(int(i))) for i in ids), \
        "no trail was ever restarted, so this check is vacuous"


@pytest.mark.parametrize("both_directions,expected",
                        [(True, [(-_PERIODIC_L, 0.0), (_PERIODIC_L, 0.0)]),
                         (False, [(-_PERIODIC_L, 0.0)])],
                        ids=["both", "one_way"])
def test_the_registered_shifts_are_the_declared_ones_without_duplicates(both_directions, expected):
    """`both_directions` adds the opposite shift, and a shift already registered is dropped - so
    attaching the condition to both ends of a periodic pair, which is the natural thing to write,
    costs nothing."""
    tr, _ids, _elapsed = _periodic_run("shifts_" + ("both" if both_directions else "one"),
                                       with_wrap=True, endtime=0.1,
                                       both_directions=both_directions)
    assert sorted(tuple(w) for w in tr.get_periodic_wraps()) == expected


# ----------------------------------------------------------------------------------------------
# Spatial adaptation
# ----------------------------------------------------------------------------------------------
#
# Refinement replaces a leaf element by its sons; unrefinement DELETES them. A particle holding an
# element pointer across either is holding a pointer the mesh has invalidated, and the failure is
# not one that shows up in the numbers: oomph keeps a refined element's parent alive, so a stale
# pointer into one keeps producing plausible values, while a pointer into a deleted son is
# undefined behaviour that need not crash. So these tests assert the MECHANISM (the collection
# noticed and re-located) as well as the answer.

class _AdaptiveTracerProblem(Problem):
    """A blob that travels to the right, so the refinement pattern keeps moving under the particles
    and elements are created and destroyed on essentially every step."""

    def __init__(self, seeds, advection, on_interface=None, max_level=4):
        super().__init__()
        self.seeds = seeds
        self.advection = advection
        self.on_interface = on_interface
        self.max_level = max_level

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(size=[4, 1], N=[16, 4]))
        eqs = _MovingBlob() + DirichletBC(u=0) @ "left" + DirichletBC(u=0) @ "right"
        eqs += SpatialErrorEstimator(u=1)
        tp = TracerParticles(self.advection, seed=TracerSeedPoints(self.seeds),
                             rtol=1e-11, atol=1e-13)
        eqs += (tp @ self.on_interface) if self.on_interface else tp
        self += eqs @ "domain"

    def tracers(self):
        return self.get_mesh("domain/" + self.on_interface if self.on_interface
                             else "domain").get_tracers()


class _MovingBlob(Equations):
    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        x, y = var("coordinate_x"), var("coordinate_y")
        xc = 0.3 + 0.8 * var("time")
        source = 50 * exp(-200 * ((x - xc) ** 2 + (y - 0.5) ** 2))
        self.add_residual(weak(grad(u), grad(v)) - weak(source, v))


_ADAPT_FIELD = vector(4 * var("coordinate_y") * (1 - var("coordinate_y")), 0)


def test_tracers_survive_refinement_and_unrefinement():
    p = _AdaptiveTracerProblem([[0.3, 0.25], [0.3, 0.5], [0.3, 0.75]], _ADAPT_FIELD)
    p.set_output_directory("_tracer_adapt")
    p.quiet()
    p.max_refinement_level = 4
    p.initialise()
    m = p.get_mesh("domain")
    tr = p.tracers()
    start = tr.get_positions().copy()
    t0 = float(p.get_current_time(as_float=True, dimensional=False))

    refined, unrefined, relocations = False, False, 0
    counts = [m.nelement()]
    for _ in range(12):
        p.solve(timestep=0.05, spatial_adapt=1)
        counts.append(m.nelement())
        if counts[-1] > counts[-2]:
            refined = True
        if counts[-1] < counts[-2]:
            unrefined = True
        # Every particle must have been re-located on a step where the mesh changed, and none on a
        # step where it did not: this is the assertion that fails if the adaptation is not announced.
        if counts[-1] != counts[-2]:
            assert tr.get_relocations_last_step() == tr.nlocal(), \
                "the mesh adapted but %d of %d particles kept their old element" % (
                    tr.nlocal() - tr.get_relocations_last_step(), tr.nlocal())
            relocations += 1
    elapsed = float(p.get_current_time(as_float=True, dimensional=False)) - t0

    assert refined and unrefined, "elements were only %s (counts %s)" % (
        "created" if refined else "destroyed", counts)
    assert relocations >= 5, "only %d adapting steps, too few to mean much" % relocations
    assert tr.nlocal() == len(start), "particles were lost across the adaptation"
    end = tr.get_positions()
    expected_x = start[:, 0] + 4 * start[:, 1] * (1 - start[:, 1]) * elapsed
    assert numpy.max(numpy.abs(end[:, 0] - expected_x)) < 1e-11
    assert numpy.max(numpy.abs(end[:, 1] - start[:, 1])) < 1e-13


def test_interface_tracers_survive_adaptation():
    """The interface mesh is torn down and rebuilt on every adaptation, so its elements are new
    objects rather than merely re-parented ones."""
    p = _AdaptiveTracerProblem([[0.4, 1.0], [1.1, 1.0]], vector(1, 0), on_interface="top")
    p.set_output_directory("_tracer_adapt_interface")
    p.quiet()
    p.max_refinement_level = 4
    p.initialise()
    tr = p.tracers()
    start = tr.get_positions().copy()
    t0 = float(p.get_current_time(as_float=True, dimensional=False))
    for _ in range(10):
        p.solve(timestep=0.05, spatial_adapt=1)
    elapsed = float(p.get_current_time(as_float=True, dimensional=False)) - t0
    assert tr.nlocal() == len(start)
    end = tr.get_positions()
    assert numpy.max(numpy.abs(end[:, 0] - (start[:, 0] + elapsed))) < 1e-10
    assert numpy.max(numpy.abs(end[:, 1] - start[:, 1])) < 1e-12


def test_tracers_survive_adaptation_of_a_curved_boundary():
    """Refining a curved boundary MOVES it: the new nodes are snapped onto the true arc, so the
    domain a particle sits in is not the domain it was located in. A particle near the boundary can
    genuinely end up outside, which is the one case where relocation has to do more than bookkeeping.
    """
    from pyoomph.meshes.simplemeshes import CircularMesh

    class P(Problem):
        def define_problem(self):
            self.add_mesh(CircularMesh(radius=1, segments=["NE", "NW", "SW", "SE"]))
            eqs = PoissonEquation(source=1) + DirichletBC(u=0) @ "circumference"
            eqs += SpatialErrorEstimator(u=1)
            # Seeded close to the arc, where refinement moves the boundary the most.
            seeds = [[0.94 * math.cos(a), 0.94 * math.sin(a)] for a in
                     numpy.linspace(0, 2 * math.pi, 12, endpoint=False)]
            # Slow rotation, so they stay in the annulus the refinement keeps changing.
            eqs += TracerParticles(vector(-var("coordinate_y"), var("coordinate_x")),
                                   seed=TracerSeedPoints(seeds), rtol=1e-11, atol=1e-13)
            self += eqs @ "domain"

    p = P()
    p.set_output_directory("_tracer_adapt_curved")
    p.quiet()
    p.max_refinement_level = 4
    p.initialise()
    tr = p.get_mesh("domain").get_tracers()
    n0 = tr.nlocal()
    assert n0 == 12
    r0 = numpy.hypot(*tr.get_positions().T)
    for _ in range(8):
        p.solve(timestep=0.05, spatial_adapt=1)
    assert tr.nlocal() == n0, "%d of %d particles were lost at the curved boundary" % (
        n0 - tr.nlocal(), n0)
    # Rigid rotation preserves the radius exactly; the boundary moving underneath must not drag them.
    r1 = numpy.hypot(*tr.get_positions().T)
    assert numpy.max(numpy.abs(numpy.sort(r1) - numpy.sort(r0))) < 1e-9


# ----------------------------------------------------------------------------------------------
# Remeshing
# ----------------------------------------------------------------------------------------------
#
# Different from adaptation: remeshing builds an entirely NEW mesh object, sharing no element and no
# node with the old one, and discretising the domain differently. Nothing about the particles'
# element pointers survives, so the collections have to be carried across to the replacement mesh
# and every particle re-located from its stored physical position.

class _RemeshBlob(GmshTemplate):
    """Quarter disc with a curved boundary, rebuilt through a spline over the previous nodes."""

    def define_geometry(self):
        self.default_resolution = 0.12
        p00 = self.point(0, 0)
        if not self.is_remeshing():
            p10, p01 = self.point(1, 0), self.point(0, 1)
            self.circle_arc(p10, p01, center=p00, name="interface")
        else:
            coords = self.get_boundary_coordinates("domain/interface", sort_along_axis="x+")
            pts = [self.point(x, y) for x, y in coords[0]]
            self.spline(pts, name="interface")
            p10, p01 = pts[-1], pts[0]
        self.create_lines(p10, "substrate", p00, "axis", p01)
        self.plane_surface("substrate", "axis", "interface", name="domain")


_REMESH_BULK = [[0.3, 0.2], [0.5, 0.35], [0.2, 0.5]]
_REMESH_SURF = [[math.cos(a), math.sin(a)] for a in (0.4, 0.9, 1.3)]


class _RemeshProblem(Problem):
    def define_problem(self):
        from pyoomph.meshes.remesher import Remesher2d
        m = _RemeshBlob()
        m.remesher = Remesher2d(m)
        self.add_mesh(m)
        eqs = PoissonEquation(source=1) + DirichletBC(u=0) @ "interface"
        eqs += TracerParticles(vector(0, 0), seed=TracerSeedPoints(_REMESH_BULK),
                               tracer_name="bulk", rtol=1e-11, atol=1e-13)
        eqs += TracerParticles(vector(0, 0), seed=TracerSeedPoints(_REMESH_SURF),
                               tracer_name="surf", rtol=1e-11, atol=1e-13) @ "interface"
        self += eqs @ "domain"


def _remesh_problem(tag):
    p = _RemeshProblem()
    p.set_output_directory("_tracer_" + tag)
    p.quiet()
    p.initialise()
    return p


def test_bulk_tracers_survive_a_remeshing_event():
    p = _remesh_problem("remesh_bulk")
    tr = p.get_mesh("domain").get_tracers("bulk")
    before = tr.get_positions().copy()
    assert len(before) == len(_REMESH_BULK)
    p.force_remesh()
    tr = p.get_mesh("domain").get_tracers("bulk")
    assert tr.nlocal() == len(_REMESH_BULK), "particles were lost in the remesh"
    # The domain is unchanged, only its discretisation, so the particles must not move at all.
    assert numpy.max(numpy.abs(tr.get_positions() - before)) < 1e-13


def test_interface_tracers_survive_a_remeshing_event():
    """An interface mesh is rebuilt as a new object rather than replaced in the problem's mesh dict,
    so without carrying the collections across the replacement the particles - and the collection
    itself - simply vanished."""
    p = _remesh_problem("remesh_surf")
    tr = p.get_mesh("domain/interface").get_tracers("surf")
    before = tr.get_positions().copy()
    ids_before = list(tr.get_ids())
    p.force_remesh()
    tr = p.get_mesh("domain/interface").get_tracers("surf", error_on_missing=False)
    assert tr is not None, "the tracer collection did not survive the remesh at all"
    assert tr.nlocal() == len(_REMESH_SURF), "particles were lost in the remesh"
    assert list(tr.get_ids()) == ids_before, "identities were not preserved"
    # The rebuilt boundary is a spline through the old nodes, so it is the same curve to within the
    # spline's own error; the particles are re-projected onto it and must not have travelled along it.
    assert numpy.max(numpy.abs(tr.get_positions() - before)) < 1e-3


def test_interface_tracers_stay_on_the_interface_after_a_remeshing_event():
    """The invariant that matters for an interface particle is not where it is along the surface but
    that it is ON it, and that has to hold immediately after the remesh, not only once the next
    sub-step has re-anchored it."""
    p = _remesh_problem("remesh_surf_offset")
    p.force_remesh()
    imesh = p.get_mesh("domain/interface")
    tr = imesh.get_tracers("surf")
    located = numpy.array(imesh.locate_points(tr.get_positions(), lagrangian=False), dtype=float)
    assert numpy.all(located[:, 0] > 0.5), "a tracer is not in the rebuilt interface mesh at all"
    assert numpy.max(numpy.abs(located[:, 1])) < 1e-12, \
        "normal offset after the remesh: %g" % numpy.max(numpy.abs(located[:, 1]))


def test_tracers_keep_advecting_correctly_after_a_remeshing_event():
    """A remesh in the middle of a run must not disturb the trajectory."""
    class P(Problem):
        def define_problem(self):
            from pyoomph.meshes.remesher import Remesher2d
            m = _RemeshBlob()
            m.remesher = Remesher2d(m)
            self.add_mesh(m)
            eqs = PoissonEquation(source=1) + DirichletBC(u=0) @ "interface"
            # Rigid rotation: the radius is preserved exactly, so any disturbance from the remesh
            # shows up immediately and cannot be absorbed into the answer.
            eqs += TracerParticles(vector(-var("coordinate_y"), var("coordinate_x")),
                                   seed=TracerSeedPoints([[0.55, 0.15], [0.35, 0.4], [0.2, 0.2]]),
                                   rtol=1e-11, atol=1e-13)
            self += eqs @ "domain"

    p = P()
    p.set_output_directory("_tracer_remesh_advect")
    p.quiet()
    p.initialise()
    tr = p.get_mesh("domain").get_tracers()
    r0 = numpy.hypot(*tr.get_positions().T)
    for i in range(6):
        p.solve(timestep=0.02)
        if i == 2:
            p.force_remesh()
            tr = p.get_mesh("domain").get_tracers()
    assert tr.nlocal() == 3, "particles were lost across the remesh"
    r1 = numpy.hypot(*tr.get_positions().T)
    assert numpy.max(numpy.abs(r1 - r0)) < 1e-8, "the remesh disturbed the trajectory"
