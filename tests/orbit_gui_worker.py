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

# Worker for tests/test_bifurcation_gui.py's periodic-orbit tests. One Problem per process, so each
# phase is its own run.
#
# The system is the Hopf normal form in Cartesian coordinates, the same one tests/hopf_lyapunov_worker.py
# uses, because everything the GUI has to get right about an orbit is known in closed form for it:
#
#   m*xdot = mu*x - w*y + sigma*x*(x^2+y^2)
#   m*ydot = w*x + mu*y + sigma*y*(x^2+y^2)
#
# The polar reduction is exact: m*rdot = mu*r + sigma*r^3 and m*thetadot = w. So with sigma = -1 and
# m = w = 1 the Hopf sits at mu = 0, the cycle for mu > 0 has radius exactly sqrt(mu), and its period
# is exactly 2*pi whatever the amplitude. Which pins every number the GUI records:
#
#   x over one cycle:      minimum -sqrt(mu), maximum +sqrt(mu), average 0
#   x^2 + y^2 over it:     constant sqrt(mu)^2 -- a degenerate band, which must still render
#   period:                2*pi, on every point of the branch
#   Floquet multipliers:   the trivial 1 (which must NOT be recorded) and exp(-2*mu*T/m), from
#                          m*d(dr)/dt = (mu + 3*sigma*r^2)*dr = -2*mu*dr
#   Floquet exponent:      log(mu_2)/T = -2*mu/m
#
# so nothing here is checked against the code's own output.

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy

from pyoomph import Problem, ODEEquations, InitialCondition
from pyoomph.expressions import var, testfunction, partial_t
from pyoomph.utils.bifurcation_gui import BifurcationGUI
from pyoomph.utils.bifurcation_gui.controller import _FixedViewLimits
from pyoomph.utils.bifurcation_gui.model import ORBIT_T_KEY, orbit_band_names

_W = 1.0
_SIGMA = -1.0
_M = 1.0


class NormalFormODE(ODEEquations):
    def define_fields(self):
        self.define_ode_variable("x", "y")

    def define_residuals(self):
        x, y = var(["x", "y"])
        mu = self.get_problem().mu
        r2 = x**2 + y**2
        self.add_residual((_M*partial_t(x) - (mu*x - _W*y + _SIGMA*x*r2))*testfunction(x))
        self.add_residual((_M*partial_t(y) - (_W*x + mu*y + _SIGMA*y*r2))*testfunction(y))


class OrbitProblem(Problem):
    def __init__(self, hessian=True):
        super().__init__()
        self.mu = self.define_global_parameter(mu=-0.1)
        self._want_hessian = hessian

    def define_problem(self):
        self += NormalFormODE() @ "nf"
        self += InitialCondition(x=0.0, y=0.0) @ "nf"


def build(args, hessian=True):
    problem = OrbitProblem(hessian=hessian)
    problem.set_output_directory(os.path.join(args.outdir, "out"))
    problem.quiet()
    if hessian:
        problem.setup_for_stability_analysis(analytic_hessian=True)
    gui = BifurcationGUI(problem, "mu")
    c = gui.controller
    c.view = _FixedViewLimits(xlim=(-0.2, 1.0), ylim=(-1.5, 1.5))
    c.neigen = 2
    c.orbit_NT = args.NT
    c.orbit_order = 3
    c.orbit_portable = bool(args.portable)
    c.set_initial_observable("nf/x")
    return problem, gui, c


def _orbit_facts(c, point):
    """Everything the closed form pins, read off one recorded orbit point."""
    lo, hi = orbit_band_names("nf/x")
    mu = float(point.param_value)
    T = float(point.obs_values[ORBIT_T_KEY])
    return {
        "mu": mu,
        "T": T,
        "avg_x": float(point.obs_values["nf/x"]),
        "min_x": float(point.obs_values[lo]),
        "max_x": float(point.obs_values[hi]),
        "nfloquet": len(point.floquet),
        "floquet": [[float(numpy.real(m)), float(numpy.imag(m))] for m in point.floquet],
        "exponents": [[float(numpy.real(v)), float(numpy.imag(v))] for v in point.eig_values],
        "unstable_count": point.unstable_count,
        "nT": int((point.orbit_info or {}).get("nT", 0)),
        "exact_average": bool((point.orbit_info or {}).get("exact_average", False)),
    }


def run_switch(args):
    problem, gui, c = build(args)
    with problem:
        res = {"phase": "switch"}
        c.start(0.02)
        c.locate_bifurcation()
        hopf = c.current_point
        res["mu_hopf"] = float(hopf.param_value)
        res["hopf_type"] = str((hopf.bifurcation_info or {}).get("type"))
        # The step onto the orbit is the one the current ds buys in the parameter.
        res["offset"] = float(c.branch_switch_parameter_offset())

        assert c.branch_switch(), "the branch switch at the Hopf did not take"
        res["kind"] = c.current_branch.kind
        res["mu_first_orbit"] = float(c.current_point.param_value)
        res["points"] = [_orbit_facts(c, c.current_point)]

        for _ in range(args.steps):
            c.step()
            res["points"].append(_orbit_facts(c, c.current_point))

        # Nothing may take the problem's own spectrum for the multipliers: get_floquet_multipliers
        # writes them into _last_eigenvalues, and everything that reads those afterwards would be
        # reading multipliers as eigenvalues.
        before = list(problem.get_last_eigenvalues())
        c._orbit_floquet()
        after = list(problem.get_last_eigenvalues())
        res["eigenvalues_survived_floquet"] = (
            len(before) == len(after) and all(abs(a-b) < 1e-12 for a, b in zip(before, after)))

        # The band of a quantity that is constant over the cycle must be degenerate, not absent.
        lo, hi = orbit_band_names("nf/x")
        res["band_names_registered"] = (lo in c._avail_observables and hi in c._avail_observables
                                        and ORBIT_T_KEY in c._avail_observables)
        res["period_axis_hidden_on_steady"] = _period_axis_hides_steady_branches(c)
        c.save_all()
        c.output_curves()
        res["export_header"] = _first_orbit_export_header(c)
        return res


def _period_axis_hides_steady_branches(c):
    """A stationary point has no period, so a period-vs-parameter plot shows the orbits alone."""
    from pyoomph.utils.bifurcation_gui.model import observable_axis, BRANCH_ORBIT
    c.set_y_axis(observable_axis(ORBIT_T_KEY))
    try:
        return all(c.branch_can_be_plotted(b) == (b.kind == BRANCH_ORBIT) for b in c.branches)
    finally:
        c.set_y_axis(observable_axis("nf/x"))


def _first_orbit_export_header(c):
    import glob
    odir = c.problem.get_output_directory(os.path.join(c.data_subdir, "output"))
    for f in sorted(glob.glob(os.path.join(odir, "*", "*", "*.txt"))):
        head = "".join(open(f).readlines()[:2])
        if "orbit" in head:
            return head
    return ""


def run_near_hopf(args):
    """Very close to the Hopf, where the orbit's OWN multiplier is as near 1 as the trivial one.

    mu = 1e-7 puts the physical multiplier at 1 - 1.3e-6, inside the default tolerance for
    recognising the trivial one. Removing "everything within the tolerance of 1" deletes it - i.e.
    exactly the number that says whether the orbit is stable, on the branch where it is least
    obvious. Exactly one multiplier is trivial, so exactly one may be removed.
    """
    problem, gui, c = build(args)
    with problem:
        res = {"phase": "nearhopf"}
        c.orbit_eps = 1e-7
        c.start(0.02)
        c.locate_bifurcation()
        assert c.branch_switch(), "the branch switch at the Hopf did not take"
        p = c.current_point
        res["mu"] = float(p.param_value)
        res["T"] = float(p.obs_values[ORBIT_T_KEY])
        res["floquet"] = [[float(numpy.real(m)), float(numpy.imag(m))] for m in p.floquet]
        res["trivial_error"] = float(c._orbit_trivial_multiplier_error)
        return res


def run_reload(args):
    """A second process: the diagram on disk must come back as an orbit, and step on."""
    problem, gui, c = build(args)
    with problem:
        res = {"phase": "reload"}
        c.prepare()
        c.load_all()
        orbit_branches = [b for b in c.branches if b.kind == "orbit"]
        assert orbit_branches, "no orbit branch survived the reload"
        b = orbit_branches[0]
        res["npoints"] = len(b)
        c.load_pt(b[-1])
        # The handler has to be back, or "the orbit" is one phase of it and the next step continues a
        # stationary state instead.
        res["handler_installed"] = c._orbit_handler() is not None
        res["T_after_reload"] = float(c.orbit_period())
        res["stored"] = _orbit_facts(c, b[-1])
        # ... and everything computed from it again must agree with what was stored.
        obs, info = c._evaluate_orbit_observables()
        lo, hi = orbit_band_names("nf/x")
        res["recomputed"] = {"T": float(obs[ORBIT_T_KEY]), "min_x": float(obs[lo]),
                             "max_x": float(obs[hi]), "avg_x": float(obs["nf/x"])}
        exps, mults = c._orbit_floquet()
        res["recomputed"]["floquet"] = [[float(numpy.real(m)), float(numpy.imag(m))] for m in mults]
        c.step()
        res["stepped"] = _orbit_facts(c, c.current_point)
        return res


def run_rediscretize(args):
    """Switch off the Hopf with bsplines, then convert the orbit to collocation.

    The reason to want this: bsplines converge more readily on the step off a Hopf, but carry no
    degree of freedom at the end of the period, so the orbit has no Floquet multipliers at all. The
    conversion has to produce the SAME orbit - which is checkable here, since the closed form pins
    the period, the amplitude and the one non-trivial multiplier.
    """
    problem, gui, c = build(args)
    with problem:
        res = {"phase": "rediscretize"}
        c.orbit_mode = "bspline"
        c.start(0.02)
        c.locate_bifurcation()
        assert c.branch_switch(), "the branch switch at the Hopf did not take"
        res["kind"] = c.current_branch.kind

        def facts(tag):
            f = _orbit_facts(c, c.current_point)
            f["mode"] = str((c.current_point.orbit_info or {}).get("mode"))
            res[tag] = f
            return f

        facts("bspline")
        npoints_before = len(c.current_branch)
        res["mu"] = float(c.current_point.param_value)

        c.orbit_mode = "collocation"
        assert c.change_orbit_discretisation(), "the re-discretization reported failure"
        facts("collocation")
        # The same point, not another one: it is the same solution at the same parameter value.
        res["points_unchanged"] = (len(c.current_branch) == npoints_before)
        res["mu_unchanged"] = float(c.current_point.param_value)

        # It continues as an orbit branch afterwards, in the new discretization.
        c.step()
        facts("stepped")

        # ... and the converted point reloads with the cycle it was rewritten with.
        c.save_all()
        c.load_pt(c.current_branch[0])
        res["reloaded_nT"] = int(c._orbit_handler().get_num_time_steps())
        res["reloaded_mode"] = str(c._orbit.mode)
        return res


def run_go_back(args):
    """Going back to an earlier point of an orbit branch, and stepping on from there.

    oomph's FIRST arclength step after a reset is not one: it increments the parameter by the whole of
    ds and only then builds the derivatives. A point that is LOADED has no tangent unless one is put
    back - the state dump of an augmented system cannot carry it - so the step from it degenerated
    into a plain parameter jump of the current ds. Measured before the fix, with ds grown from 0.02 to
    0.63 over three steps: going back to the point the Hopf switch created and stepping took mu from
    0.02 to 0.647, straight off the branch, where the tangent gives 0.064.

    Points reached BY a step keep their tangent in the orbit's sidecar, and this checks that too: the
    restored tangent has to be the one that was in force, and stepping with the same ds has to
    reproduce the point the forward sweep found.
    """
    problem, gui, c = build(args)
    with problem:
        res = {"phase": "goback"}
        c.start(0.02)
        c.locate_bifurcation()
        assert c.branch_switch(), "the branch switch at the Hopf did not take"
        b = c.current_branch

        def arclength():
            return (numpy.asarray(problem.get_arclength_dof_derivative_vector(), dtype=float).copy(),
                    float(problem.get_arc_length_parameter_derivative()),
                    float(problem.get_arc_length_theta_sqr()), float(c._last_ds))

        recorded = {}
        for _ in range(3):
            c.step()
            recorded[len(b)-1] = arclength()
        res["npoints"] = len(b)
        res["ds_after_three_steps"] = float(c._last_ds)
        res["mu"] = [float(p.param_value) for p in b]
        # By reference, not by index: each reload below is followed by a step, and a step that lands
        # BETWEEN two existing points is inserted between them - so b[1] afterwards is not the point
        # b[1] was. (Which is how this test first passed for the wrong reason: before the arclength
        # metric applied to orbits, that step landed far past the end of the branch and was appended.)
        original = list(b)

        # (a) the FIRST point of the branch - the one the switch made, which never had a tangent.
        c.load_pt(original[0])
        d, dp, th, _ds = arclength()
        res["first"] = {"tangent_len": len(d), "ndof": int(problem.ndof()),
                        "dparam_ds": dp, "mu_before": float(problem.mu.value)}
        c.step()
        res["first"]["mu_after"] = float(problem.mu.value)
        res["first"]["ds_used"] = float(res["ds_after_three_steps"])

        # (b) a point reached BY a step: its tangent must come back exactly, and stepping with the ds
        # that was in force must land where the forward sweep landed.
        k = 1
        d0, dp0, th0, ds0 = recorded[k]
        mu_forward = float(original[k+1].param_value)
        c.load_pt(original[k])
        d, dp, th, _ds = arclength()
        res["restored"] = {
            "tangent_len": len(d), "same_length": len(d) == len(d0),
            "max_abs_diff": float(numpy.max(numpy.abs(d-d0))) if len(d) == len(d0) else None,
            "dparam_ds": dp, "dparam_ds_stored": dp0,
            "theta_sqr": th, "theta_sqr_stored": th0}
        c._last_ds = ds0
        c.step()
        res["restored"]["mu_after"] = float(problem.mu.value)
        res["restored"]["mu_forward"] = mu_forward
        return res


def run_arclength_metric(args):
    """What one arclength step buys on an orbit must not depend on how finely time is resolved.

    The arclength constraint is (dparameter/ds)^2 + theta^2*|dU/ds|^2 = 1, and on an orbit dU is the
    WHOLE CYCLE - nT copies of the base dofs. Left at theta^2 = 1 that charges the sum over all of
    them, so dparameter/ds falls off as 1/sqrt(nT*Ndof) and the parameter creeps: measured before the
    fix, 0.164 at nT=12, 0.107 at nT=30, 0.0766 at nT=60, exactly the ratio of the square roots.
    Doubling the time resolution halved the parameter movement per step, for the same ds and the same
    orbit. (The FIRST step after switching hid it: oomph's first step is a plain parameter increment
    of ds, which is why it always looked fine and every one after it did not.)

    The mass-matrix metric now applies to an orbit as the base norm AVERAGED over the cycle, so this
    reports the same numbers whatever --NT is.
    """
    problem, gui, c = build(args)
    with problem:
        res = {"phase": "metric", "NT": int(args.NT)}
        c.start(0.02)
        c.locate_bifurcation()
        assert c.branch_switch(), "the branch switch at the Hopf did not take"
        res["nT"] = int(c._orbit_handler().get_num_time_steps())
        res["ndof"] = int(problem.ndof())
        steps = []
        for _ in range(4):
            before = float(problem.mu.value)
            c.step()
            steps.append({"dmu": float(problem.mu.value) - before,
                          "dparam_ds": float(problem.get_arc_length_parameter_derivative()),
                          "theta_sqr": float(problem.get_arc_length_theta_sqr()),
                          "T": float(c.orbit_period())})
        res["steps"] = steps
        return res


def run_bspline_floquet(args):
    """A B-spline orbit branch, continued with its Floquet multipliers.

    A periodic B-spline basis has no end-of-period degree of freedom, so the orbit Jacobian has no
    seam for the condensation to cut and the orbit has no multipliers of its own. The GUI asks the
    ORBIT rather than the problem, and the orbit answers on a collocation sampling of itself - so
    every point of a bspline branch gets its stability without the branch ever being converted.
    """
    problem, gui, c = build(args)
    with problem:
        res = {"phase": "bsplinefloquet"}
        c.orbit_mode = "bspline"
        c.orbit_order = 3
        c.start(0.02)
        c.locate_bifurcation()
        assert c.branch_switch(), "the branch switch at the Hopf did not take"
        res["kind"] = c.current_branch.kind
        res["installed_mode"] = str(c._orbit.mode)
        res["is_floquet_mode"] = bool(c._orbit_handler().is_floquet_mode())

        pts = []
        for _ in range(3):
            p = c.current_point
            pts.append({"mu": float(p.param_value),
                        "mode": str((p.orbit_info or {}).get("mode")),
                        "nT": int((p.orbit_info or {}).get("nT", 0)),
                        "T": float(p.obs_values[ORBIT_T_KEY]),
                        "floquet": [[float(numpy.real(m)), float(numpy.imag(m))] for m in p.floquet],
                        "unstable_count": p.unstable_count,
                        "exponent": float(p.eig_value_Re)})
            c.step()
        res["points"] = pts
        # Still a B-spline branch afterwards, and still one after a reload.
        res["mode_after"] = str(c._orbit.mode)
        res["ndof_after"] = int(problem.ndof())
        c.save_all()
        c.load_pt(c.current_branch[0])
        res["mode_after_reload"] = str(c._orbit.mode)
        res["nT_after_reload"] = int(c._orbit_handler().get_num_time_steps())
        return res


def run_bad_fingerprint(args):
    """A stored orbit that no longer matches the problem must be refused, not loaded."""
    problem, gui, c = build(args)
    with problem:
        res = {"phase": "fingerprint"}
        c.prepare()
        c.load_all()
        b = [x for x in c.branches if x.kind == "orbit"][0]
        pt = b[-1]
        pt.orbit_info = dict(pt.orbit_info or {})
        pt.orbit_info["fingerprint"] = "not the fingerprint of this problem"
        try:
            c.load_pt(pt)
        except Exception as e:
            res["refused"] = True
            res["message"] = str(e)
            return res
        res["refused"] = False
        return res


def run_no_hessian(args):
    """Without the analytic Hessian there is no route to an orbit at all - say so, and install nothing."""
    problem, gui, c = build(args, hessian=False)
    with problem:
        res = {"phase": "nohessian"}
        c.start(0.02)
        c.locate_bifurcation()
        res["refusal"] = c.orbit_can_be_started() or ""
        try:
            c.switch_to_orbit()
            res["raised"] = False
        except Exception as e:
            res["raised"] = True
            res["message"] = str(e)
        res["handler_installed"] = c._orbit_handler() is not None
        res["kind"] = c.current_branch.kind
        return res


PHASES = {"switch": run_switch, "reload": run_reload, "fingerprint": run_bad_fingerprint,
          "nohessian": run_no_hessian, "nearhopf": run_near_hopf,
          "rediscretize": run_rediscretize, "goback": run_go_back,
          "metric": run_arclength_metric, "bsplinefloquet": run_bspline_floquet}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="orbit_gui_out")
    ap.add_argument("--phase", default="switch", choices=sorted(PHASES))
    ap.add_argument("--NT", type=int, default=30)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--portable", action="store_true")
    args = ap.parse_args()
    try:
        res = PHASES[args.phase](args)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    print("PYOOMPH_ORBIT_RESULT " + json.dumps(res))
    print("PYOOMPH_WORKER_DONE")


if __name__ == "__main__":
    main()
