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

# The bifurcation GUI's numerics had no coverage at all while they lived inside the key handler of a
# matplotlib window: there was no way to reach step(), locate_bifurcation() or the save/load
# round-trip without a display. They now sit in BifurcationController, which knows nothing about
# matplotlib or tkinter, so the whole workflow can be driven from here.
#
# The test problem is du/dt = mu - u^2 - b*u, with TWO global parameters. At b = 0 it is the
# textbook saddle-node: branches u = +-sqrt(mu) meeting at a fold at mu = 0, eigenvalue -2u. In
# general the stationary states are mu = u^2 + b*u and the fold sits where d(mu)/du = 2u + b = 0,
# i.e. at u_c = -b/2 and mu_c(b) = -b^2/4 - a parabolic fold locus, known in closed form. Every
# assertion below is therefore checked against theory rather than against the code's own output,
# which is what makes locate_bifurcation() and (later) the fold-locus continuation testable at all.
#
# The second parameter also exists to pin the slice bookkeeping: a diagram continued in mu is only
# valid at one value of b, and that has to be recorded.
#
# Only ONE Problem is constructed per process: a second one segfaults in the JIT loader, which is
# why the module has a single test function rather than one per aspect.

import json
import os
import shutil
import sys

import pytest

import numpy

from pyoomph import Problem, ODEEquations, InitialCondition
from pyoomph.expressions import var_and_test, partial_t
from pyoomph.utils.bifurcation_gui import BifurcationGUI
from pyoomph.utils.bifurcation_gui.controller import (STATE_VERSION, BifurcationController,
                                                     _FixedViewLimits)


def gui_launch_prefix():
    """Command prefix for a worker that opens a real Tk window.

    Two of the workers below drive BifurcationTkApp, i.e. tk.Tk(), which needs an X display to talk
    to. The nightly runs from cron, where there is none, so without this they fail with "no display
    name and no $DISPLAY environment variable" -- which says nothing about the behaviour under test.

    xvfb-run -a picks a free display number itself and tears the server down when the command exits,
    so there is no lifetime to manage here and no fixed :N to collide with a desktop session or with
    a second copy of the suite. A real DISPLAY is used as-is when there is one, so this changes
    nothing for a developer running the suite from a terminal.

    Skipped rather than failed when there is neither: the tests need something they have not been
    given, which is what skip means. The nightly surfaces SKIPPED lines in its report
    (citools/nightly_develop.sh), so it stays visible instead of quietly passing.
    """
    if os.environ.get("DISPLAY"):
        return []
    xvfb_run = shutil.which("xvfb-run")
    if xvfb_run is None:
        pytest.skip("needs an X display: $DISPLAY is unset and xvfb-run is not installed "
                    "(apt install xvfb)")
    return [xvfb_run, "-a"]


def fold_location(b: float) -> tuple[float, float]:
    """Analytic fold of du/dt = mu - u^2 - b*u : (mu_c, u_c) = (-b^2/4, -b/2)."""
    return -b * b / 4.0, -b / 2.0


class FoldEquations(ODEEquations):
    """du/dt = mu - u^2 - b*u."""

    def __init__(self, mu, b):
        super().__init__()
        self.mu, self.b = mu, b

    def define_fields(self):
        self.define_ode_variable("u")

    def define_residuals(self):
        u, u_test = var_and_test("u")
        self.add_weak(partial_t(u) - (self.mu - u**2 - self.b * u), u_test)


class FoldProblem(Problem):
    def define_problem(self):
        eqs = FoldEquations(self.get_global_parameter("mu"), self.get_global_parameter("b"))
        eqs += InitialCondition(u=1.0)
        self += eqs @ "ode"


def test_bifurcation_gui_controller(tmp_path):
    with FoldProblem() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.get_global_parameter("mu").value = 1.0
        problem.get_global_parameter("b").value = 1.0

        gui = BifurcationGUI(problem, "mu")
        c: BifurcationController = gui.controller
        # No plot in this test, so the controller gets a fixed view rectangle. It has to be large
        # enough to contain the whole sweep, because multistep() stops at its border.
        c.view = _FixedViewLimits(xlim=(-1.0, 2.0), ylim=(-2.0, 2.0))
        c.neigen = 1

        logged: list = []
        c.set_observer(on_log=logged.append)

        # ---------------------------------------------------------- start
        c.start(-0.05)
        assert c.available_observables == ["ode/u"]
        assert c.observable == "ode/u"
        assert len(c.branches) == 1 and len(c.branches[0]) == 1
        p0 = c.current_point
        assert p0 is not None
        assert numpy.isclose(p0.param_value, 1.0)
        # mu = u^2 + b*u with mu = b = 1 gives the golden-ratio root u = (sqrt(5)-1)/2
        u0 = (numpy.sqrt(5.0) - 1.0) / 2.0
        assert numpy.isclose(p0.obs_values["ode/u"], u0, atol=1e-8)
        # The stable branch: eigenvalue -(2u + b) = -sqrt(5)
        assert numpy.isclose(p0.eig_value_Re, -numpy.sqrt(5.0), atol=1e-6)
        assert os.path.isfile(p0.statefile)
        assert p0.param_values == {"mu": 1.0, "b": 1.0}, "all parameters are recorded, not just mu"

        # ---------------------------------------------------------- multistep on a fresh diagram
        # Regression: multistep read the continuation tangent that only step() ever fills in, so on
        # a diagram that had not been stepped yet it raised KeyError - and once that was fixed, a
        # zero-length tangent scaled ds down to nothing and the sweep never terminated.
        c.view = _FixedViewLimits(xlim=(0.99, 1.01), ylim=(0.5, 0.7))  # left by the first step in mu
        c.multistep()
        assert len(c.branches[0]) == 2
        assert c.ds != 0.0
        c.view = _FixedViewLimits(xlim=(-1.0, 2.0), ylim=(-2.0, 2.0))

        # ---------------------------------------------------------- continuation
        for _ in range(6):
            c.step()
        assert len(c.branches[0]) == 8
        for p in c.branches[0]:
            # every point must satisfy the stationary relation mu = u^2 + b*u
            u = p.obs_values["ode/u"]
            assert numpy.isclose(p.param_value, u * u + p.param_values["b"] * u, atol=1e-7)
        # stepping with a negative ds walks towards the fold
        assert c.branches[0][-1].param_value < 1.0 or c.branches[0][0].param_value < 1.0
        # scoords are normalized and ordered by the insertion heuristic
        scoords = [p.scoord for p in c.branches[0]]
        assert scoords == sorted(scoords)

        # ---------------------------------------------------------- the fold, against theory
        c.locate_bifurcation()
        fold = c.current_point
        assert fold is not None
        mu_c, u_c = fold_location(1.0)
        assert numpy.isclose(fold.param_value, mu_c, atol=1e-7), "fold should be at mu=-b^2/4"
        assert numpy.isclose(fold.obs_values["ode/u"], u_c, atol=1e-5)
        assert fold.eig_value_Re == 0, "a located bifurcation is flagged by a zero real part"
        # Bifurcation tracking must not stay active afterwards, or every later solve would be
        # augmented.
        assert not problem.get_bifurcation_tracking_mode()

        # ... and not when the tracking solve FAILS either, which is the case that used to go wrong.
        # The teardown lived in one caller rather than beside the activation, so a diverged tracking
        # solve left the augmented system installed; the next attempt to locate a bifurcation then
        # reported "a non-zero shift is required" from its opening eigensolve - the leftover tracker's
        # message, saying nothing about the solve that actually failed. Two of the three entry points
        # (Locate pitchfork, Locate the bifurcation of the selected eigenvalue) had no cleanup at all.
        # The solve is stubbed rather than driven to divergence, so the invariant is tested and not
        # some particular way of reaching it.
        real_solve = problem.solve
        boom = RuntimeError("the tracking solve diverged")

        def failing_solve(*a, **kw):
            raise boom

        problem.solve = failing_solve   # type:ignore[method-assign]
        # try/except rather than pytest.raises: this function imports pytest locally further down,
        # which makes the name local to the whole function and unbound up here.
        raised = None
        try:
            c.locate_bifurcation()
        except RuntimeError as e:
            raised = e
        finally:
            problem.solve = real_solve  # type:ignore[method-assign]
        assert raised is boom, "the real failure has to reach the caller, not be swallowed"
        assert not problem.get_bifurcation_tracking_mode(), \
            "a failed locate must leave the tracker off, or every later solve is augmented"
        # And the problem is usable again straight away, which is the point of the invariant.
        c.locate_bifurcation()
        assert numpy.isclose(c.current_point.param_value, mu_c, atol=1e-7)
        assert not problem.get_bifurcation_tracking_mode()

        # the stability split now has the fold as its boundary
        segs, stabs = c.branches[0].to_branch_stab_list("ode/u")
        assert len(segs) == len(stabs) and len(segs) >= 1

        # ---------------------------------------------------------- the slice this diagram is in
        branch = c.branches[0]
        assert branch.kind == "solution"
        assert branch.continuation_parameter == "mu"
        assert branch.slice_is_known(), "every point records all global parameters"
        assert branch.fixed_parameters() == {"b": 1.0}, "b is held fixed along a continuation in mu"
        assert branch.slice_is_consistent()
        assert branch.describe_slice() == "b = 1"
        assert c.describe_current_slice() == "b = 1"
        assert set(c.all_parameter_names()) == {"mu", "b"}

        # ---------------------------------------------------------- the whole spectrum is recorded
        assert len(p0.eig_values) == c.neigen, "every computed eigenvalue is kept, not just the leading one"
        assert numpy.isclose(p0.eig_values[0].real, p0.eig_value_Re), "the leading one comes first"
        assert p0.measured_unstable_count() == 0, "the starting point is stable"
        spectra_before = [[complex(v) for v in p.eig_values] for b in c.branches for p in b]

        # ---------------------------------------------------------- axes: either can be anything
        from pyoomph.utils.bifurcation_gui.model import observable_axis, parameter_axis

        assert c.x_axis == parameter_axis("mu"), "x defaults to the continued parameter"
        assert c.y_axis == observable_axis("ode/u"), "y defaults to the current observable"
        assert set(c.available_axes()) == {parameter_axis("mu"), parameter_axis("b"),
                                          observable_axis("ode/u")}
        # A parameter on both axes is the shape a bifurcation locus is drawn in. Here b is constant,
        # so it is a horizontal line - the point is that it plots at all, from the same machinery.
        c.set_x_axis(parameter_axis("b"))
        c.set_y_axis(parameter_axis("mu"))
        assert c.branch_can_be_plotted(branch)
        segs, _ = branch.to_branch_stab_list(c.y_axis, xspec=c.x_axis)
        assert all(numpy.allclose(seg[:, 0], 1.0) for seg in segs), "b is the x column now"
        assert c.axis_range(parameter_axis("b")) == (1.0, 1.0)
        # Setting an observable back on y must also restore _current_observable, which the tangent
        # bookkeeping, the saved state and the facade attribute all key off.
        c.set_x_axis(parameter_axis("mu"))
        c.set_y_axis(observable_axis("ode/u"))
        assert c._current_observable == "ode/u"
        assert c._y_axis is None, "an observable on y is expressed through _current_observable"

        # A branch that never recorded the parameter now on an axis is skipped, not drawn wrong.
        from pyoomph.utils.bifurcation_gui.model import BifurcationGUISolutionBranch
        foreign = BifurcationGUISolutionBranch(kind="solution", continuation_parameter="nonesuch")
        foreign.append(branch[0])
        c.set_x_axis(parameter_axis("nonesuch"))
        assert not c.branch_can_be_plotted(foreign)
        c.set_x_axis(None)
        assert c.x_axis == parameter_axis("mu")

        # ---------------------------------------------------------- tags and export
        c.toggle_point_tag(branch[0], 3)
        assert branch[0].tag == 3
        outdir = c.output_curves()
        assert os.path.isfile(os.path.join(outdir, "tag_03.txt"))
        assert os.path.isfile(os.path.join(outdir, "tag03.dump"))
        # One directory per slice of parameter space, so curves computed at different fixed values
        # cannot land in one folder and be plotted together by accident.
        bdir = os.path.join(outdir, "slice00", "branch000")
        assert any(f.startswith("smoothed_") for f in os.listdir(bdir))
        assert any(f.startswith("bifurcation_") for f in os.listdir(bdir))
        # An exported curve must say what was held fixed, or it cannot be interpreted later.
        exported = next(f for f in os.listdir(bdir) if f.startswith("smoothed_"))
        with open(os.path.join(bdir, exported)) as f:
            head = "".join(line for line in f if line.startswith("#"))
        assert "continued in mu" in head and "fixed: b = 1" in head
        with open(os.path.join(outdir, "tag_03.txt")) as f:
            assert "fixed: b = 1" in f.read()

        # output_all_observables used to pass None down as the observable name, which reached
        # obs_values[None] and raised - the flag never worked. The y axis is now a list of specs.
        c.output_all_observables = True
        try:
            outdir = c.output_curves()
            bdir = os.path.join(outdir, "slice00", "branch000")
            exported = next(f for f in os.listdir(bdir) if f.startswith("smoothed_"))
            with open(os.path.join(bdir, exported)) as f:
                lines = [ln for ln in f if not ln.startswith("#")]
            header = open(os.path.join(bdir, exported)).readline()
            assert "ode/u" in header
            # parameter, every observable, then the two eigenvalue columns
            assert len(lines[0].split()) == 1 + len(c.available_observables) + 2
        finally:
            c.output_all_observables = False

        # ---------------------------------------------------------- save / load round trip
        c.save_all()
        statefile = os.path.join(problem.get_output_directory(c.data_subdir), "state.json")
        assert os.path.isfile(statefile)
        with open(statefile) as f:
            stored = json.load(f)
        assert stored["current_observable"] == "ode/u"
        assert stored["xlim"] == [-1.0, 2.0]
        assert stored["version"] == STATE_VERSION
        assert stored["parameter"] == "mu"
        assert stored["branches"][0]["kind"] == "solution"
        assert stored["branches"][0]["continuation_parameter"] == "mu"
        assert stored["branches"][0]["points"][0]["param_values"] == {"mu": 1.0, "b": 1.0}
        # The spectrum goes into the file as two flat lists, which is what makes it available in the
        # Points tab without reloading a point's state dump.
        first = stored["branches"][0]["points"][0]
        assert len(first["eig_values_Re"]) == c.neigen
        assert len(first["eig_values_Im"]) == c.neigen

        before = [(p.param_value, p.obs_values["ode/u"], p.eig_value_Re, p.scoord, p.tag)
                  for b in c.branches for p in b]

        applied: list = []
        c.load_all(apply_view=applied.append)
        assert applied and applied[0]["statestep"] == stored["statestep"]
        after = [(p.param_value, p.obs_values["ode/u"], p.eig_value_Re, p.scoord, p.tag)
                 for b in c.branches for p in b]
        assert before == after, "reloading must reproduce the diagram exactly"
        assert [[complex(v) for v in p.eig_values] for b in c.branches for p in b] == spectra_before, \
            "the recorded spectra survive the round trip"
        assert c.current_point is not None
        assert numpy.isclose(c.get_bifurcation_parameter().value, c.current_point.param_value)

        # ---------------------------------------------------------- point management
        n_before = len(c.branches[0])
        doomed = c.branches[0][0]
        c.select_point(c.branches[0], doomed)
        c.delete_selected_point()
        assert len(c.branches[0]) == n_before - 1
        assert doomed not in c.branches[0]
        # the dump goes with it, otherwise _states would grow without bound over a long session
        assert not os.path.isfile(doomed.statefile)
        assert os.path.isfile(c.branches[0][0].statefile), "surviving points keep their state"

        # navigation moves the selection, never the loaded state
        loaded = c.current_point
        c.select_relative("first")
        assert c.selected_point is c.branches[0][0]
        c.select_relative("next")
        assert c.selected_point is c.branches[0][1]
        c.select_relative("last")
        assert c.selected_point is c.branches[0][-1]
        assert c.current_point is loaded

        # ---------------------------------------------------------- switching the continuation parameter
        # Step back onto a regular point first: a fold is a fold of the Jacobian, not of one
        # parameter, so continuing in ANY parameter from exactly there has no regular tangent.
        regular = next(p for p in c.branches[0] if p.eig_value_Re != 0)
        c.select_point(c.branches[0], regular)
        c.goto_selected_point()
        assert c.current_point is regular

        n_branches = len(c.branches)
        first = c.branches[0]
        assert c.set_continuation_parameter("b")
        assert c._paramname == "b"
        assert len(c.branches) == n_branches + 1, "a different parameter is a new diagram"
        assert c.x_axis == parameter_axis("b"), "the x axis follows the continued parameter"
        assert c._tangs == {}, "a (dparam, dobs) tangent says nothing about a different parameter"
        new = c.branches[-1]
        assert new.continuation_parameter == "b"
        assert set(new.fixed_parameters()) == {"mu"}, "mu is what is held fixed now"
        # The old branch is a section at a fixed b, so it is not part of the diagram being built now.
        assert not c.branch_is_on_current_slice(first)
        assert c.branch_is_on_current_slice(new)
        assert not c.set_continuation_parameter("b"), "switching to the same parameter is a no-op"

        # Continuing in b must leave mu alone, and the points must still solve the problem.
        mu_before = problem.get_global_parameter("mu").value
        for _ in range(3):
            c.step()
        assert numpy.isclose(problem.get_global_parameter("mu").value, mu_before), "mu is fixed now"
        for p in new:
            u = p.obs_values["ode/u"]
            assert numpy.isclose(p.param_values["mu"], u * u + p.param_values["b"] * u, atol=1e-7)

        # Clicking must never select a point from another slice: loading it would move mu.
        for p in first:
            if c.select_nearest_point(p.param_values["b"], p.obs_values["ode/u"]):
                assert c.branch_is_on_current_slice(c.selected_branch)

        # Moving a parameter that is being held fixed is likewise a new slice.
        c.set_continuation_parameter("mu")
        n_branches = len(c.branches)
        c.set_fixed_parameter("b", 2.0)
        assert len(c.branches) == n_branches + 1
        assert numpy.isclose(problem.get_global_parameter("b").value, 2.0)
        assert c.branches[-1].fixed_parameters() == {"b": 2.0}
        # The fold moved with b, which is the whole point of tracking the slice.
        mu_c2, u_c2 = fold_location(2.0)
        c.locate_bifurcation()
        assert c.current_point is not None
        assert numpy.isclose(c.current_point.param_value, mu_c2, atol=1e-6)
        assert numpy.isclose(c.current_point.obs_values["ode/u"], u_c2, atol=1e-4)

        import pytest
        with pytest.raises(ValueError):
            c.set_fixed_parameter("mu", 0.0)      # that is the continuation parameter
        with pytest.raises(ValueError):
            c.set_continuation_parameter("nope")

        # ------------------------------------------------- following the fold through parameter space
        # mu is adjusted to hold the fold while b is stepped, which traces mu_c(b) = -b^2/4.
        c.set_continuation_parameter("mu")
        regular = next(p for p in c.branches[-1] if p.eig_value_Re != 0)
        c.select_point(c.branches[-1], regular)
        c.goto_selected_point()
        c.ds = -0.05
        c.locate_bifurcation()
        assert c.current_point is not None and c.current_point.eig_value_Re == 0

        c.start_locus(tracked="mu", continue_in="b")
        locus = c.branches[-1]
        assert locus.kind == "locus"
        assert (locus.tracked_parameter, locus.continuation_parameter) == ("mu", "b")
        assert locus.bifurcation_type == "fold"
        assert c.on_locus()
        # A locus varies both parameters, so it is drawn in their plane.
        assert (c.x_axis, c.y_axis) == (parameter_axis("b"), parameter_axis("mu"))
        assert problem.get_bifurcation_tracking_mode() == "fold"

        c.view = _FixedViewLimits(xlim=(-4.0, 4.0), ylim=(-6.0, 6.0))
        c.ds = 0.2
        for _ in range(5):
            c.step()
        assert len(locus) == 6
        for p in locus:
            b_here = p.param_values["b"]
            mu_c, u_c = fold_location(b_here)
            # The whole point: every locus point is the analytic fold for its own b.
            assert numpy.isclose(p.param_values["mu"], mu_c, atol=1e-6), (b_here, p.param_values)
            assert numpy.isclose(p.obs_values["ode/u"], u_c, atol=1e-4)
            # The CRITICAL eigenvalue stays the synthetic 0 + i*omega that tracked continuation
            # reports: re-solving it would turn the exact zero into a small nonzero value and the
            # point would stop reading as a bifurcation.
            assert p.eig_value_Re == 0
            # The rest of the spectrum IS solved for now (the base state's eigenproblem is available
            # while tracking), which is what makes a codim-2 point along a locus visible at all. On
            # this one-dof problem the single eigenvalue is the fold's own, so it must come back at
            # zero -- the certificate that the base state really is at the fold and that the
            # eigenproblem was assembled on the BASE, not the augmented, dof layout.
            assert len(p.eig_values) == c.neigen, p.eig_values
            assert numpy.isclose(numpy.real(p.eig_values[0]), 0.0, atol=1e-6), p.eig_values
        # ...and the eigensolve really ran. Both ways of not running it -- the deliberately non-fatal
        # except (a failed shift-invert must not abort a long two-parameter sweep) and
        # locus_eigen_shift=None -- fall back to recording the synthetic critical value alone, which
        # on this fold is exactly 0 with neigen == 1 and would satisfy both assertions above without
        # a single eigenproblem having been solved. A solved fold eigenvalue is a rounding-error-sized
        # number rather than an exact zero, so at least one point has to differ from its own critical
        # value; the log is checked as well, so a failure says which of the two happened.
        assert not any("Could not solve the eigenproblem" in str(m) for m in logged), logged
        assert any(complex(p.eig_values[0]) != complex(p.eig_value_Re, p.eig_value_Im) for p in locus), \
            [(p.eig_values[0], p.eig_value_Re) for p in locus]
        assert len({round(p.param_values["b"], 6) for p in locus}) == len(locus), "b really moved"
        # No normal form is computed per locus point even with classification on: the type is known.
        assert all(p.bifurcation_info is None for p in locus)

        # ------------------------------------------------- the invariant: tracking follows the branch
        solution_pt = c.branches[0][0]
        c.load_pt(solution_pt)
        assert problem.get_bifurcation_tracking_mode() == "", "an ordinary branch must not be tracked"
        c.load_pt(locus[2])
        assert problem.get_bifurcation_tracking_mode() == "fold", "a locus branch must be tracked"
        assert c.current_branch is locus and c._paramname == "b"
        # Resuming the locus after that round trip must still land on the analytic curve.
        c.step()
        p = c.current_point
        assert p is not None
        assert numpy.isclose(p.param_values["mu"], fold_location(p.param_values["b"])[0], atol=1e-6)

        # locating or switching a bifurcation makes no sense while following one
        with pytest.raises(RuntimeError):
            c.locate_bifurcation()
        with pytest.raises(RuntimeError):
            c.branch_switch()
        with pytest.raises(ValueError):
            c.start_locus(tracked="b", continue_in="b")

        # ------------------------------------------------- dropping back off the locus
        b_at_exit = problem.get_global_parameter("b").value
        n_branches = len(c.branches)
        c.leave_locus()
        assert len(c.branches) == n_branches + 1
        assert not c.on_locus()
        assert problem.get_bifurcation_tracking_mode() == "", "tracking is off again"
        assert c._paramname == "mu", "back to an ordinary continuation in mu"
        assert numpy.isclose(problem.get_global_parameter("b").value, b_at_exit), "b stays where it was"
        left = c.current_point
        assert left is not None
        u = left.obs_values["ode/u"]
        # A regular point of the ordinary branch at that b, not the fold itself.
        assert numpy.isclose(left.param_values["mu"], u * u + b_at_exit * u, atol=1e-7)
        assert left.eig_value_Re != 0
        assert not numpy.isclose(u, fold_location(b_at_exit)[1], atol=1e-6)
        c.step()   # and ordinary continuation works from there

        # ------------------------------------------------- a locus exports in its OWN coordinates
        # Not in whatever the window is showing: the first two columns must be the parameter pair,
        # i.e. the curve the tutorials write by hand as "V  Bo_c".
        outdir = c.output_curves()
        locus_headers = []
        for root, _dirs, files in os.walk(outdir):
            for f in files:
                if not f.endswith(".txt"):
                    continue
                text = open(os.path.join(root, f)).read()
                if "locus of fold bifurcations" in text:
                    locus_headers.append(text.splitlines()[0])
        assert locus_headers, "the locus branch must be exported"
        for h in locus_headers:
            cols = h.lstrip("# ").split()
            assert cols[:2] == ["b", "mu"], h

        # ---------------------------------------------------------- quick mode
        # What a quick step records, and that the leading eigenvalue is NaN rather than a fake zero -
        # a zero would make every quick point look like a located bifurcation.
        problem.set_linear_solver("superlu")
        assert c.determinant_sign_supported(), "superlu reports a determinant sign"
        # Continue on the branch we are already on: the assertions are about what a quick STEP records,
        # not about the branch, and moving a fixed parameter from here would be a continuation of its own.
        quick_branch = c._get_current_branch()
        c.ds = -0.05
        c.set_quick_mode(True)
        assert c.quick_mode and c.quick_mode_detector == "auto"
        for _ in range(5):
            c.step()
        quick_points = [p for p in quick_branch if p.det_sign is not None]
        assert quick_points, "a quick step records a determinant sign"
        for p in quick_points:
            assert p.eig_values == [], "no spectrum is computed in quick mode"
            assert p.eig_value_Re != p.eig_value_Re, "the leading eigenvalue is NaN, not a fake zero"
            assert p.dparam_ds is not None, "the tangent is recorded as the second test function"
        assert not any(p.eig_value_Re == 0 for p in quick_points), "NaN must not read as a bifurcation"

        c.propagate_stability(quick_branch)
        assert any(p.stability_source == "inferred" for p in quick_branch)

        # Back-fill from the state dumps: the inferred stability must agree with what the spectrum says.
        inferred = {id(p): p.unstable_count for p in quick_branch}
        n = c.compute_spectrum_for_branch(quick_branch)
        assert n > 0
        for p in quick_branch:
            assert p.stability_source == "eigen" and p.eig_values
            if inferred[id(p)] is not None:
                assert p.measured_unstable_count() == inferred[id(p)], \
                    "stability inferred from the determinant disagrees with the spectrum"

        c.set_quick_mode(False)

        # ---------------------------------------------------------- abort flag
        c.request_abort()
        assert c.abort_requested
        assert c.step() is None, "an aborted step must not advance the solution"
        c.clear_abort()

        assert logged, "the controller must route its progress messages to the observer"


def test_legacy_state_file_reports_the_slice_as_unknown():
    """A state.json written before the parameters were recorded must still load.

    The parameter that was continued is recoverable - it can only have been the one the GUI was
    constructed with - but what the others were held at is not, and must be reported as unknown
    rather than invented. This is the whole reason for the format version.
    """
    from pyoomph.utils.bifurcation_gui.model import BifurcationGUISolutionBranch

    legacy = {
        "points": [
            {"param_value": 1.0, "obs_value": {"ode/u": 0.5}, "eig_value_Re": -2.0,
             "eig_value_Im": 0.0, "statefile": None, "outstep": 0, "scoord": 0.0, "tangs": {}},
            {"param_value": 0.8, "obs_value": {"ode/u": 0.4}, "eig_value_Re": -1.5,
             "eig_value_Im": 0.0, "statefile": None, "outstep": 1, "scoord": 1.0, "tangs": {}},
        ]
    }
    b = BifurcationGUISolutionBranch.from_dict(legacy, default_continuation_parameter="mu")
    assert b.kind == "solution"
    assert b.continuation_parameter == "mu", "recoverable, so recovered"
    assert not b.slice_is_known(), "the fixed values were never written down"
    assert b.fixed_parameters() == {}
    assert b.describe_slice() == "slice unknown"
    # An empty fixed-parameter dict must not be read as "nothing was held fixed": a diagram that
    # genuinely has no other parameters says so differently.
    modern = BifurcationGUISolutionBranch.from_dict(
        {"points": [dict(legacy["points"][0], param_values={"mu": 1.0})], "kind": "solution",
         "continuation_parameter": "mu"})
    assert modern.slice_is_known()
    assert modern.fixed_parameters() == {}
    assert modern.describe_slice() == "no other parameters"


def test_embedded_plot_hook_leaves_the_figure_open(tmp_path):
    """MatplotlibPlotter.plot_into_current_figure must draw and then get out of the way.

    Deliberately checked with a trivial plot definition and no display: the mechanics that can break
    an embedded pane are the file write, the restoring of file_trunk, and above all the plt.close()
    that _after_plot does for a plot written to file - a closed figure would leave every pane in the
    GUI permanently blank. The real 2D render (mesh, colorbar, set_view) is covered by the GUI smoke
    script, which needs a display.
    """
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pyoomph.output.plotting import MatplotlibPlotter

    class LinePlotter(MatplotlibPlotter):
        def define_plot(self):
            plt.plot([0, 1, 2], [0, 1, 0])

    with FoldProblem() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.initialise()
        pl = LinePlotter()
        pl._problem = problem
        pl._named_problems[""] = problem

        fig = plt.figure(figsize=(3, 2))
        num = fig.number
        plt.figure(num)
        trunk_before = pl.file_trunk
        pl.plot_into_current_figure()

        assert num in plt.get_fignums(), "the figure must NOT be closed - a pane would go blank"
        assert plt.gcf() is fig, "the plot went into the figure we made current"
        assert len(fig.axes) == 1 and len(fig.axes[0].lines) == 1
        assert pl.file_trunk == trunk_before, "file_trunk is restored so normal output still works"
        assert pl._embedded is False
        assert glob.glob(os.path.join(str(tmp_path), "_plots", "*")) == [], "nothing may be written"

        # Re-rendering into the same figure must not accumulate axes.
        fig.clf()
        plt.figure(num)
        pl.plot_into_current_figure()
        assert len(fig.axes) == 1

        # And save() now refuses instead of raising an AttributeError on a None trunk.
        pl.file_trunk = None
        import pytest
        with pytest.raises(RuntimeError):
            pl.save()
        plt.close(num)


def test_eigenfunction_plotter_is_derived_from_the_source():
    """An eigenfunction pane reuses the plot definition rather than asking for it twice."""
    from pyoomph.output.plotting import MatplotlibPlotter
    from pyoomph.utils.bifurcation_gui.panes import derive_eigenfunction_plotter

    class Custom(MatplotlibPlotter):
        # A user's plotter may take its own constructor arguments, which is why the clone is a copy
        # rather than type(source)(...) - that would have to guess this signature.
        def __init__(self, scale, **kwargs):
            super().__init__(**kwargs)
            self.scale = scale

        def define_plot(self):
            pass

    src = Custom(2.5)
    src._range_objects["h"] = object()
    clone = derive_eigenfunction_plotter(src, eigenvector=0, eigenmode="imag")

    assert isinstance(clone, Custom) and clone.scale == 2.5, "subclass and its own state survive"
    assert (clone.eigenvector, clone.eigenmode) == (0, "imag")
    assert clone.file_trunk is None, "a derived plotter never writes files"
    assert clone._range_objects == {} and clone._range_objects is not src._range_objects, \
        "sharing the range objects would tie the panes' colour scales together"
    assert clone._added_parts is not src._added_parts
    assert not clone._initialised
    # The source must be left exactly as the script wrote it.
    assert src.eigenvector is None and src.eigenmode == "abs"
    assert src.file_trunk is not None and "h" in src._range_objects


def test_leaving_a_branch_transiently_reaches_another_solution(tmp_path):
    """Perturb along the unstable eigenfunction, integrate, and land somewhere else.

    Two things had to be true for this to work at all, and neither was. The perturbation is scaled to a
    target residual by bisection, and a residual that comes out inf - which is what a perturbation the
    size of a whole eigenvector does to any equation with a 1/h or a log in it - compared as neither
    larger nor smaller than the target, so the full amplitude survived and the transient died on its
    first step. And the step size has to stay below the growth time of the mode: a fully implicit step
    far above it damps an unstable mode rather than amplifying it, so the adaptive stepper - which sees
    a solution sitting still near a stationary state and keeps doubling dt - marched back to the very
    solution it was told to leave and reported the old branch as a new one.

    The worker checks both, plus that the landing is a stable state and not the unstable one.
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "transient_leave_worker.py")
    proc = subprocess.run([sys.executable, worker, "--outdir", os.path.join(str(tmp_path), "out")],
                          cwd=here, capture_output=True, text=True, timeout=900)
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, "worker failed:\n" + out[-3000:]
    assert "PYOOMPH_WORKER_DONE" in out, "worker did not finish:\n" + out[-3000:]
    assert "LEFT the branch" in out


def test_a_fold_is_not_mistaken_for_a_branch_point(tmp_path):
    """The other half of the classification: a fold must keep coming out as a fold.

    The fold/branch-point decision is how far dR/dparameter lies out of the range of the Jacobian,
    measured as an ANGLE - the projection itself carries no scale, since the left null vector is
    normalised by its overlap with the right one, and a second nearly-critical eigenvalue makes that
    overlap small and the projection correspondingly large. That is what used to report a thin-film
    branch point as a fold. A test that only checks branch points would be passed by a measure that
    has simply been loosened until nothing is a fold any more, hence this one.
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "branch_switch_worker.py")
    proc = subprocess.run([sys.executable, worker, "--kind", "fold", "--api", "problem",
                           "--outdir", os.path.join(str(tmp_path), "out")],
                          cwd=here, capture_output=True, text=True, timeout=900)
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, "worker failed:\n" + out[-3000:]
    assert "PYOOMPH_WORKER_DONE" in out, "worker did not finish:\n" + out[-3000:]
    assert "kind=fold" in out


@pytest.mark.parametrize("api", ["gui", "problem"])
@pytest.mark.parametrize("kind", ["transcritical", "pitchfork"])
def test_branch_switching_lands_on_the_other_branch(kind, api, tmp_path):
    """Switching must reach the OTHER branch and stay on it, checked against closed-form branches.

    Out of process because each bifurcation type needs its own Problem, following the worker pattern the
    rest of the suite uses. Run through the GUI and through Problem.switch_branch on its own, since the
    numerics live on Problem so that a plain script can use them. The trivial branch u = 0 exists for
    every mu in both problems, so a switch that does not work fails silently by staying on it - which is
    exactly how the old implementation failed. It seeded oomph's arclength state with INCREMENTS where the two setters take derivatives, so
    the tangent had norm ~ds instead of 1 and the arclength constraint asked for a step inflated by 1/ds.
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "branch_switch_worker.py")
    proc = subprocess.run([sys.executable, worker, "--kind", kind, "--api", api,
                           "--outdir", os.path.join(str(tmp_path), "out")],
                          cwd=here, capture_output=True, text=True, timeout=900)
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, "worker failed:\n" + out[-3000:]
    assert "PYOOMPH_WORKER_DONE" in out, "worker did not finish:\n" + out[-3000:]
    assert "kind=" + kind in out


@pytest.mark.parametrize("kind", ["transcritical", "pitchfork"])
def test_a_reloaded_diagram_keeps_its_classifications(kind, tmp_path):
    """What a bifurcation IS has to survive being written to a diagram and read back.

    Two processes over one output directory, because that is the situation: the second run of a script
    finds state.json and rebuilds the diagram from it. The classification used to be left out of the
    file altogether, so a reloaded bifurcation came back as an anonymous point - no letter on the
    diagram, no arrows, no way to choose how to leave it, and branch switching that had to recompute
    the whole normal form before it could do anything.

    The null vector is still not in the file - one entry per degree of freedom does not belong in a
    small text file beside the dumps - so the second half of this is the recovery: one eigensolve at the
    point, then the predictors rebuilt from the coefficients that were saved. Landing back on the
    analytic branch afterwards is what says they were rebuilt as the originals and not merely rebuilt.
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "branch_switch_worker.py")
    outdir = os.path.join(str(tmp_path), "out")
    for phase in ("write", "reload"):
        proc = subprocess.run([sys.executable, worker, "--kind", kind, "--api", "gui",
                               "--phase", phase, "--outdir", outdir],
                              cwd=here, capture_output=True, text=True, timeout=900)
        out = (proc.stdout or "") + (proc.stderr or "")
        assert proc.returncode == 0, phase + " worker failed:\n" + out[-3000:]
        assert "PYOOMPH_WORKER_DONE" in out, phase + " worker did not finish:\n" + out[-3000:]
    assert "reloaded max_error" in out


def test_must_init_keeps_the_diagram_and_skips_the_setup(tmp_path):
    """A second run must find its diagram, not redo the walk to the starting solution, and open itself.

    Four invocations, the first two against the same output directory since surviving from one run to
    the next is the entire question. Without must_init the second run loses everything: a script that
    solves before building the GUI has already initialised the problem under the default "delete"
    runmode, and initialising strips every file from every subdirectory of the output directory -
    state.json included, which is the index that references the state dumps.

    The reload run is also required to solve *nothing at all*. That is not just economy: with the setup
    block skipped the problem sits at its raw initial condition, so the initial solve that start() used
    to do unconditionally would be working from a guess the script never intended.

    The window is opened by start(), which the script calls after the guarded block, so a failure in
    there raises where the script can see it. The "raises" phase pins that a script failing between the
    two gets its traceback rather than a window, "nowith" that the common shape without "with
    SomeProblem() as problem" works the same, and "legacy" that the earlier must_init(ds) call - which
    opened the window from an atexit handler, where an exception is printed as "Exception ignored" and
    the process still exits 0 - is rejected with a message naming its replacement.
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "must_init_worker.py")
    outdir = os.path.join(str(tmp_path), "out")

    def run(phase, out_sub="out", expect_mu=None):
        env = dict(os.environ)
        if expect_mu is not None:
            env["PYOOMPH_EXPECT_MU"] = repr(expect_mu)
        proc = subprocess.run(gui_launch_prefix() + [sys.executable, worker, "--phase", phase,
                               "--outdir", os.path.join(str(tmp_path), out_sub)],
                              cwd=here, capture_output=True, text=True, timeout=900, env=env)
        out = (proc.stdout or "") + (proc.stderr or "")
        assert proc.returncode == 0, phase + " failed:\n" + out[-3000:]
        assert "PYOOMPH_WORKER_DONE" in out, phase + " did not finish:\n" + out[-3000:]
        return out

    first = run("init")
    mu = float([ln for ln in first.splitlines() if ln.startswith("WINDOW_OPENED")][0].split("mu=")[1])
    second = run("reload", expect_mu=mu)

    # The problem announces every solve it does, so this is a direct check and not a proxy.
    assert "STATIONARY SOLVE" in first, "the first run has to solve:\n" + first[-2000:]
    assert "STATIONARY SOLVE" not in second, "the reload must not solve:\n" + second[-2000:]
    assert "Continuation Step" not in second, "the reload must not continue:\n" + second[-2000:]

    raised = run("raises", out_sub="raises")
    assert "WINDOW_OPENED" not in raised, "a raising script must not get a window:\n" + raised[-2000:]

    nowith = run("nowith", out_sub="nowith")
    assert "NOWITH opened=True" in nowith, "start() must open the window:\n" + nowith[-2000:]

    legacy = run("legacy", out_sub="legacy")
    assert "LEGACY rejected" in legacy, "must_init(ds) must be rejected:\n" + legacy[-2000:]


def test_extremum_observables_become_axis_choices(tmp_path):
    """Each ExtremumObservables entry offers its minimum, its maximum, and where each of them sits.

    Checked against a field whose extrema are unique and known in closed form, with a phase that moves
    with the parameter so the position is a curve rather than a constant. Uniqueness is the whole
    difficulty in testing a position: cos(2pi*x/Lx)*cos(2pi*y/Ly) has five maxima of equal height, so a
    correct implementation reports a position other than the one written down.

    Out of process, since it needs its own Problem. The worker also pins that six axis choices cost two
    mesh sweeps rather than six, that the "[max, x]" tag replaces the "[obs]" one instead of stacking
    with it, that set_initial_observable() picks which of them the diagram opens on, and that a name
    containing spaces, a comma and brackets survives the CSV export and a save/load round trip.
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "extremum_observable_worker.py")
    proc = subprocess.run(gui_launch_prefix() + [sys.executable, worker,
                           "--outdir", os.path.join(str(tmp_path), "out")],
                          cwd=here, capture_output=True, text=True, timeout=900)
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, "worker failed:\n" + out[-3000:]
    assert "PYOOMPH_WORKER_DONE" in out, "worker did not finish:\n" + out[-3000:]
    assert "domain/h_extreme  [max, x]" in out


def test_ds_is_recast_when_the_arclength_metric_changes():
    """Toggling the arclength scaling must not silently change the stride.

    The constraint is ds = (dp/ds)*dp + theta^2*(dU/ds).dU, so along the tangent ds = dp/(dp/ds) and a
    given ds buys a parameter increment of ds*|dp/ds|. Retuning theta^2 changes |dp/ds| and therefore
    what the same number buys; oomph compensates only on the very first step (problem.cc:11176-11181,
    guarded by !Arc_length_step_taken). Measured on the real solver with a fixed ds either side of the
    toggle, the stride dropped 20% (ratio 0.8016) without this and stays within 0.4% with it.

    A stub, because the arithmetic is the whole content and it must hold exactly.
    """
    from pyoomph.utils.bifurcation_gui.controller import BifurcationController

    class StubProblem:
        def __init__(self):
            self.theta = 1.0
            self.dpds = 0.88
            # A real Problem always has this; the recast steps aside when an inner product is set,
            # because arclength_continuation has then already compensated internally.
            self._arclength_inner_product = None
        def get_arc_length_theta_sqr(self):
            return self.theta
        def get_arc_length_parameter_derivative(self):
            return self.dpds

    logged: list = []
    c = BifurcationController.__new__(BifurcationController)
    c._on_log = logged.append
    c._on_changed = None
    c._on_status = None
    c._on_busy = None
    stub = StubProblem()
    c.problem = stub  # type: ignore[assignment]

    # theta^2 untouched: nothing to compensate, whatever dp/ds does.
    stub.dpds = 0.5
    assert c._recast_ds_after_metric_change(-0.05, 1.0, 0.88) == -0.05

    # Scaling switched on: dp/ds is pinned at sqrt(D)=sqrt(0.5), so ds must grow by 0.88/0.7071.
    stub.theta, stub.dpds = 3.252, 0.7071067811865476
    got = c._recast_ds_after_metric_change(-0.05, 1.0, 0.88)
    assert abs(got - (-0.05*0.88/0.7071067811865476)) < 1e-12, got
    # Sign is the direction of travel and must survive.
    assert got < 0
    assert any("recast" in m for m in logged), "a metric change worth 24% has to be reported"

    # A settled scaled sweep retunes theta^2 every step with dp/ds pinned, so nothing may happen.
    stub.theta, stub.dpds = 2.9, 0.7071067811865476
    assert c._recast_ds_after_metric_change(-0.05, 3.252, 0.7071067811865476) == -0.05

    # A vanishing dp/ds (right at a fold) would divide by zero; leave ds alone instead.
    stub.theta, stub.dpds = 1e-9, 0.0
    assert c._recast_ds_after_metric_change(-0.05, 1.0, 0.88) == -0.05

    # With an inner product set, Problem.arclength_continuation retunes theta^2 and rescales the step
    # itself, so doing it again here would double-count.
    stub._arclength_inner_product = "l2"
    stub.theta, stub.dpds = 3.252, 0.7071067811865476
    assert c._recast_ds_after_metric_change(-0.05, 1.0, 0.88) == -0.05


def test_arclength_proportion_must_be_a_proper_fraction():
    """D outside (0,1) would make oomph divide by D or by 1-D when it retunes theta^2."""
    from pyoomph.utils.bifurcation_gui.controller import BifurcationController

    c = BifurcationController.__new__(BifurcationController)
    c._on_log = lambda *a: None
    c._on_changed = None
    c._on_status = None
    c._on_busy = None
    c.arclength_proportion = 0.5
    c.scale_arc_length = True

    class StubProblem:
        def __init__(self):
            self.seen = None
        def set_arc_length_parameter(self, **kwargs):
            self.seen = kwargs

    c.problem = StubProblem()  # type: ignore[assignment]
    c.set_arclength_proportion(0.9)
    assert c.arclength_proportion == 0.9
    assert c.problem.seen == {"desired_proportion_of_arc_length": 0.9}, "it has to reach the problem"

    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            c.set_arclength_proportion(bad)
    assert c.arclength_proportion == 0.9, "a rejected value must not be kept"


def test_arclength_inner_product_is_mesh_independent(tmp_path):
    """A proper inner product must make ds mean the same thing on every mesh.

    oomph's arclength constraint measures the solution part with the plain Euclidean norm of the dof
    vector - a SUM over unknowns, not an integral, so it has no continuum limit. Measured on Bratu, the
    parameter movement per unit ds under that metric drifts 0.639 -> 0.106 over 65x refinement, i.e. a
    tuned ds silently means something else after a remesh. Both normalised metrics hold it to four
    digits.

    Two resolutions, one process each (a second Problem segfaults in the JIT loader). The worker also
    asserts, per mesh, that theta^2 is exactly 1/ndof for "ndof", that the mass-matrix norm lands on the
    same scaling WITHOUT being told about ndof, and that the constraint
    (dp/ds)^2 + theta^2*|dU/ds|^2 == 1 still holds after every retune - retuning theta^2 without
    renormalising the tangent would break that silently and rob ds of its meaning as a step length.
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "arclength_inner_product_worker.py")

    def run(N):
        proc = subprocess.run([sys.executable, worker, "--N", str(N),
                               "--outdir", os.path.join(str(tmp_path), "N" + str(N))],
                              cwd=here, capture_output=True, text=True, timeout=900)
        out = (proc.stdout or "") + (proc.stderr or "")
        assert proc.returncode == 0, "worker N=" + str(N) + " failed:\n" + out[-3000:]
        assert "PYOOMPH_WORKER_DONE" in out, "worker N=" + str(N) + " did not finish:\n" + out[-3000:]
        strides = {}
        for line in out.splitlines():
            if line.startswith("IP "):
                label = line.split("theta2")[0].split(None, 2)[2].strip()
                strides[label] = float(line.split("dlam/ds")[1].split()[0])
        return strides, out

    coarse, out_c = run(50)
    fine, out_f = run(200)
    assert coarse and fine, "no IP lines parsed"

    for label in ("ndof", "l2 (mass matrix)"):
        rel = abs(fine[label] - coarse[label])/abs(coarse[label])
        assert rel < 1e-3, ("{:s} is not mesh-independent: {:.6f} vs {:.6f} ({:.2%})"
                            .format(label, coarse[label], fine[label], rel))
    # And the point of the exercise: the metric it replaces IS mesh-dependent, so the test would pass
    # vacuously if the two resolutions happened to agree anyway.
    plain = abs(fine["plain dof sum"] - coarse["plain dof sum"])/abs(coarse["plain dof sum"])
    assert plain > 0.3, "the dof-sum metric was expected to drift with ndof, but moved only {:.2%}".format(plain)

    # The fold still has to be traversable in the new metric - a norm change must not cost that.
    for out in (out_c, out_f):
        for line in out.splitlines():
            if line.startswith("FOLD "):
                assert " steps x" not in line, "a fold traversal failed: " + line


def test_arclength_retune_steps_aside_on_an_inconsistent_continuation_state():
    """The retune must not touch a continuation state whose two vectors disagree in size.

    oomph keeps Dof_derivative and Dof_current as independent vectors and does not always resize them
    together (calculate_continuation_derivatives_helper resizes only the derivative), so they can
    legitimately disagree - measured right after locate_bifurcation and after load_pt, the pair is
    (0, 1). Writing back from such a pair throws "Mismatch in size of ddof and curr" out of
    update_dof_vectors_for_continuation, which is how this reached a user: a step taken after locating a
    bifurcation with an inner product active died in the middle of a session.

    A stub, because the whole content is which states are refused; the unbound method is called directly
    so no Problem has to be built.
    """
    from pyoomph.generic.problem import Problem

    class Stub:
        def __init__(self, ddof, cur, n):
            self._arclength_inner_product = "ndof"
            self.ddof, self.cur, self.n = ddof, cur, n
            self.written = []
        def _get_n_unaugmented_dofs(self): return 0
        def is_distributed(self): return False
        def ndof(self): return self.n
        def get_arclength_dof_derivative_vector(self): return numpy.array(self.ddof, dtype=float)
        def get_arclength_dof_current_vector(self): return numpy.array(self.cur, dtype=float)
        def get_arc_length_parameter_derivative(self): return 0.8
        def get_arc_length_theta_sqr(self): return 1.0
        def _update_dof_vectors_for_continuation(self, ddof, cur): self.written.append(("dofs", len(ddof)))
        def _set_arc_length_parameter_derivative(self, dp): self.written.append(("dp", dp))
        def _set_arc_length_theta_sqr(self, th): self.written.append(("theta", th))
        _arclength_weighted_square_norm = Problem._arclength_weighted_square_norm

    # The states that used to crash: one vector empty, or the two of different length.
    for ddof, cur, n in (([], [1.0], 1), ([1.0], [], 1), ([1.0, 2.0], [1.0], 2), ([1.0], [1.0], 5)):
        stub = Stub(ddof, cur, n)
        assert Problem._retune_arclength_theta(stub) == 1.0, str((ddof, cur, n))
        assert not stub.written, "nothing may be written back from an inconsistent state: " + str(stub.written)

    # A consistent state must still be retuned, or the guard would have disabled the feature.
    stub = Stub([3.0, 4.0], [0.0, 0.0], 2)
    factor = Problem._retune_arclength_theta(stub)
    kinds = [w[0] for w in stub.written]
    assert kinds == ["dofs", "dp", "theta"], kinds
    assert factor != 1.0, "a real retune has to recast ds"
    assert abs(dict(stub.written[1:])["theta"] - 0.5) < 1e-12, "'ndof' means theta^2 = 1/ndof exactly"

    # And under MPI it must refuse loudly rather than weight by each rank's share of the dofs.
    stub = Stub([3.0, 4.0], [0.0, 0.0], 2)
    stub.is_distributed = lambda: True
    with pytest.raises(RuntimeError, match="distributed"):
        Problem._retune_arclength_theta(stub)


def test_the_gui_reports_physical_units(tmp_path):
    """A problem with real units must work, and its numbers must say what they are.

    Three things were wrong. The GUI could not START: float() of an observable that still carries a unit
    throws, and since the integration measure dx has length, EVERY spatial integral observable in a
    scaled problem has one. Eigenvalues were nondimensional - with set_scaling(temporal=10*second) a
    true rate of -0.5 1/s was reported as -5. And the two observable paths disagreed: ODE values came
    out in SI base units (0.0007) while spatial ones had to be hand-nondimensionalised, with nothing
    recording either.

    The worker checks against a problem whose answers are known exactly (eigenvalue -1/tau = -0.5 1/s,
    w = 0.7 mm), with the temporal scale deliberately 10 s so nondimensional and physical values cannot
    be confused. Out of process, since it needs its own Problem.
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "units_worker.py")
    proc = subprocess.run([sys.executable, worker, "--outdir", os.path.join(str(tmp_path), "out")],
                          cwd=here, capture_output=True, text=True, timeout=900)
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, "worker failed:\n" + out[-3000:]
    assert "PYOOMPH_WORKER_DONE" in out, "worker did not finish:\n" + out[-3000:]
    assert "EIGEN -0.5 1/s" in out, "the eigenvalue must be a rate in 1/s:\n" + out[-2000:]
    assert "LABEL ode/w [mm]" in out, "axis labels must carry the unit:\n" + out[-2000:]


def test_normal_mode_eigenvalues_are_recorded_and_recomputable(tmp_path):
    """Azimuthal modes end to end: recorded per eigenvalue, plottable, and recomputable.

    The GUI only ever solved the base-state eigenproblem, so on a problem set up with
    setup_for_stability_analysis(azimuthal_stability=True) the modes that decide stability were
    invisible - an axisymmetric state can be perfectly stable to m=0 and unstable to m=1, which is what
    a polygonal hydraulic jump is.

    The worker also pins the recompute path: raising neigen used to recompute nothing at all, because
    compute_spectrum_for_branch skipped every point that already "had" a spectrum.

    Out of process, on the smallest axisymmetric problem that carries an azimuthal eigenproblem -
    generating and compiling the azimuthal code is the cost, not the solving.
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "normal_mode_worker.py")
    proc = subprocess.run([sys.executable, worker, "--outdir", os.path.join(str(tmp_path), "out")],
                          cwd=here, capture_output=True, text=True, timeout=900)
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, "worker failed:\n" + out[-3000:]
    assert "PYOOMPH_WORKER_DONE" in out, "worker did not finish:\n" + out[-3000:]
    assert "MODES 0 1 2 0 1 2" in out, "one mode per eigenvalue:\n" + out[-2000:]
    assert "REFILL 1 points" in out, "a stale point must be picked up:\n" + out[-2000:]


def test_stability_can_count_modes_or_the_base_state_alone():
    """The View toggle has to change the answer, and only where a mode is actually unstable.

    A stub: the whole content is which eigenvalues are counted. stability_indicator reads the STORED
    unstable count rather than recounting (they agree while the spectrum is sorted), so the base-state
    reading has to recount - that is the branch this pins.
    """
    from pyoomph.utils.bifurcation_gui.model import BifurcationGUISolutionPoint

    # Unstable on m=1 only: stable as an axisymmetric state, unstable as a full one.
    p = BifurcationGUISolutionPoint(1.0, {}, 0.5+0j, "", 0,
                                    eig_values=[0.5+0j, -1.0+0j, -2.0+0j],
                                    eig_modes=[1.0, 0.0, 2.0])
    p.unstable_count = p.measured_unstable_count()
    assert p.measured_unstable_count(True) == 1
    assert p.measured_unstable_count(False) == 0
    assert p.stability_indicator(include_modes=True) > 0
    assert p.stability_indicator(include_modes=False) < 0
    assert p.eigenvalues_of_mode(1.0) == [0.5+0j]

    # Without a scan every eigenvalue IS the base state, so the toggle must change nothing.
    q = BifurcationGUISolutionPoint(1.0, {}, 0.5+0j, "", 0, eig_values=[0.5+0j, -1.0+0j])
    q.unstable_count = q.measured_unstable_count()
    assert q.measured_unstable_count(True) == q.measured_unstable_count(False) == 1
    assert q.stability_indicator(include_modes=False) > 0
    assert q.eigenvalues_of_mode(0.0) == [0.5+0j, -1.0+0j]


def test_a_spectrum_is_stale_when_the_settings_changed():
    """Raising the eigenvalue count has to make existing points count as needing recomputation.

    Includes the json round trip, because the settings are compared for equality: a tuple written to
    state.json comes back as a list, and a complex shift cannot be written at all, so both are
    canonicalised by eigen_settings().
    """
    import json

    from pyoomph.utils.bifurcation_gui.controller import BifurcationController
    from pyoomph.utils.bifurcation_gui.model import BifurcationGUISolutionPoint, eigen_settings

    c = BifurcationController.__new__(BifurcationController)
    c.neigen = 4
    c.shift = 0.0
    c.normal_modes = [1, 2]

    class StubProblem:
        def is_normal_mode_stability_set_up(self): return "azimuthal"
        def get_bifurcation_tracking_mode(self): return ""
    c.problem = StubProblem()  # type: ignore[assignment]
    c.on_locus = lambda: False  # type: ignore[method-assign]

    p = BifurcationGUISolutionPoint(1.0, {}, -1+0j, "", 0, eig_values=[-1+0j],
                                    eig_settings=c.current_eigen_settings())
    assert not c.spectrum_is_stale(p)
    c.neigen = 30
    assert c.spectrum_is_stale(p), "a raised eigenvalue count must show up as stale"
    c.neigen = 4
    c.normal_modes = [1, 2, 3]
    assert c.spectrum_is_stale(p), "an added mode must show up as stale"

    # A point from before the settings were recorded is not KNOWN to be stale, and treating it as such
    # would recompute whole diagrams on load.
    legacy = BifurcationGUISolutionPoint(1.0, {}, -1+0j, "", 0, eig_values=[-1+0j])
    assert not c.spectrum_is_stale(legacy)

    # Through real json, with a complex shift.
    settings = eigen_settings(4, 0.5+2j, [0, 1])
    q = BifurcationGUISolutionPoint(1.0, {}, -1+0j, "", 0, eig_values=[-1+0j], eig_settings=settings)
    back = BifurcationGUISolutionPoint.from_dict(json.loads(json.dumps(q.to_state_dict())))
    assert back.eig_settings == settings, back.eig_settings


def _complex_petsc_pythonpath():
    """The lib directory of a COMPLEX PETSc build, or None.

    PYTHONPATH is unset in a non-login shell, so petsc4py is not importable at all there and the real
    build is what a bare import would find - see CLAUDE.md. The arch directory differs per machine, so
    it is searched for rather than pasted; a machine without one skips the test instead of failing it.
    """
    import glob
    import subprocess

    roots = [os.environ.get("PETSC_DIR", ""), os.path.expanduser("~/code/packages")]
    seen = []
    for root in roots:
        if not root:
            continue
        seen += glob.glob(os.path.join(root, "*", "lib", "petsc4py"))
        seen += glob.glob(os.path.join(root, "*", "*", "lib", "petsc4py"))
    for cand in seen:
        libdir = os.path.dirname(cand)
        try:
            out = subprocess.run([sys.executable, "-c",
                                  "from petsc4py import PETSc;import numpy;"
                                  "print(PETSc.ScalarType is numpy.complex128)"],
                                 env={**os.environ, "PYTHONPATH": libdir},
                                 capture_output=True, text=True, timeout=120)
        except Exception:
            continue
        if out.returncode == 0 and out.stdout.strip() == "True":
            return libdir
    return None


def test_the_stripe_scan_finds_a_pair_shift_invert_misses(tmp_path):
    """A region scan must find what a shift-invert solve cannot see, and merge without duplicating.

    pyoomph asks SLEPc for the eigenvalues whose REAL part is nearest the target, so a complex pair
    further from the target in real part is simply not in the answer however many are requested - which
    is how a Hopf goes unnoticed. The worker's spectrum makes that unambiguous: two real modes at -0.01
    and -0.02 and a pair at -0.5+-8i, so a 2-eigenvalue solve returns the reals and nothing else.

    Three things had to be got right and each of them failed first, so each is worth the test: the
    region has to be applied BEFORE setFromOptions (afterwards EPSSolve raises PETSC_ERR_SUP), the
    sticky eps_type/st_type options have to be cleared (krylovschur cannot do Which.ALL on a
    non-Hermitian problem), and the scan needs its own eigenvalue cap (SLEPc bounds a contour solve by
    the requested count, so asking for 2 returned 2 of the 4 in the region).

    Needs a complex PETSc, which a non-login shell does not have on its path; skipped when absent.
    """
    import subprocess

    libdir = _complex_petsc_pythonpath()
    if libdir is None:
        pytest.skip("no complex PETSc build found; the stripe scan needs one (see CLAUDE.md)")
    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "stripe_worker.py")
    proc = subprocess.run([sys.executable, worker, os.path.join(str(tmp_path), "out")],
                          cwd=here, capture_output=True, text=True, timeout=900,
                          env={**os.environ, "PYTHONPATH": libdir})
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, "worker failed:\n" + out[-3000:]
    assert "PYOOMPH_WORKER_DONE" in out, "worker did not finish:\n" + out[-3000:]
    assert "STRIPE OK" in out


def test_outputting_a_tagged_point_writes_its_fields(tmp_path):
    """Tagging a point and outputting it must write plots and VTUs, not only a state dump.

    The redirection is the substance. Problem._change_output_directory tells each output where to write
    by storing the new location relative to the problem's base directory - and _MeshFileOutput was the
    one outputter that never overrode that hook, so VTUs ignored it completely. (The same gap meant
    PeriodicOrbit.output_orbit, which relies on the very same call, was writing orbit VTUs on top of the
    ordinary ones.) The worker therefore uses a problem WITH a MeshFileOutput and asserts a .vtu lands
    in the tag directory; a text-only problem would have passed either way.

    It also asserts the restore: the output directory, the output step counter and the current point all
    have to come back, and no state dump may leak into the diagram's own store.
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "tag_output_worker.py")
    proc = subprocess.run([sys.executable, worker, os.path.join(str(tmp_path), "out")],
                          cwd=here, capture_output=True, text=True, timeout=900)
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, "worker failed:\n" + out[-3000:]
    assert "PYOOMPH_WORKER_DONE" in out, "worker did not finish:\n" + out[-3000:]
    assert "TAGOUT OK" in out


@pytest.mark.parametrize("policy", ["off", "when_needed", "every_n"])
def test_adaptivity_during_continuation(policy, tmp_path):
    """The GUI can remesh/adapt during a sweep, and the arclength metric survives it.

    Two different things sit behind one setting, and only one of them applies to most problems:
    remesh_handler_during_continuation REMESHES and does nothing without a remesher (RemeshWhen), while
    the adaptation passes refine the existing mesh from the problem's SpatialErrorEstimators. The first
    version of this only called the handler, and every policy left ndof at 39 - the test passed while
    proving nothing.

    The assertion that matters is the arclength invariant afterwards. Both paths restore the
    continuation tangent from the history slots, which preserves its direction but not its length, so
    without renormalisation ds silently stops meaning a step length - measured 29% short in the plain
    adapt case. Off must change nothing; the other two must actually adapt.
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "adapt_continuation_worker.py")
    proc = subprocess.run([sys.executable, worker, "--gui-policy", policy,
                           "--outdir", os.path.join(str(tmp_path), "out")],
                          cwd=here, capture_output=True, text=True, timeout=900)
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, policy + " failed:\n" + out[-3000:]
    assert "PYOOMPH_WORKER_DONE" in out, policy + " did not finish:\n" + out[-3000:]
    assert "GUIPOLICY " + policy in out


def test_a_state_saved_on_an_adapted_mesh_restores_its_tangent(tmp_path):
    """Loading a state must apply its continuation tangent AFTER the equations are renumbered.

    _define_state_file reads the meshes but the numbering is rebuilt only after it returns -
    rebuild_global_mesh_from_list, actions_after_adapt and reapply_boundary_conditions all run later. So
    the tangent used to be checked against the OLD dof count, and any state saved on an adapted mesh
    threw "Mismatching size in the dof direction vector" and took the whole reload with it. It is now
    parked and applied afterwards, the same way the interface values already were.

    The assertion is that the tangent is RESTORED, not merely that nothing raises: dropping it would
    also "pass" while quietly losing the continuation direction.
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "stale_tangent_worker.py")
    proc = subprocess.run([sys.executable, worker, os.path.join(str(tmp_path), "out")],
                          cwd=here, capture_output=True, text=True, timeout=900)
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, "worker failed:\n" + out[-3000:]
    assert "STALE TANGENT OK" in out
    assert "Mismatching size" not in out


def test_a_state_dumped_while_tracking_can_be_reloaded(tmp_path):
    """A dump written with a bifurcation tracker active must not make the diagram unloadable.

    locate_bifurcation() activates the tracker, solves, and records the point - which dumps a state -
    and only then deactivates, so the dump carried a continuation tangent of the AUGMENTED length
    (2n+1 for a fold, 3n+2 for a Hopf or azimuthal one). Reloading the diagram later threw

        Mismatching size in the dof direction vector and the actual number of DoFs: 89777 vs 29976

    out of Problem::set_dof_direction_arclength and took the whole reload with it. 89777 = 3*29925+2 is
    an augmented count, not a mesh difference - which is what identified the cause.

    Two guards, and the worker exercises both: an augmented tangent is no longer written (it means
    nothing outside the tracker), and a stored tangent whose length does not match is ignored with a
    note rather than raising - which is what rescues a file written before the first guard existed, or
    one whose mesh has been adapted since. An ordinary dump must still round-trip its tangent, or the
    fix would have quietly disabled the feature.
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "tracked_state_worker.py")
    proc = subprocess.run([sys.executable, worker, os.path.join(str(tmp_path), "out")],
                          cwd=here, capture_output=True, text=True, timeout=900)
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, "worker failed:\n" + out[-3000:]
    assert "PYOOMPH_WORKER_DONE" in out, "worker did not finish:\n" + out[-3000:]
    assert "Mismatching size" not in out
    assert "TRACKED STATE OK" in out


def test_branch_switch_refuses_a_fold():
    """A fold has one branch through it, so there is nothing to switch to.

    Checked on a stub, since the refusal happens before the problem is touched at all - and this is the
    branch of branch_switch that must NOT go wandering off looking for a second branch.
    """
    from pyoomph.utils.bifurcation_gui.controller import BifurcationController
    from pyoomph.utils.bifurcation_gui.model import (BifurcationGUISolutionBranch,
                                                     BifurcationGUISolutionPoint)

    logged: list = []
    c = BifurcationController.__new__(BifurcationController)
    c._on_log = logged.append
    c._on_changed = None
    c._on_status = None
    c._on_busy = None
    branch = BifurcationGUISolutionBranch(kind="solution", continuation_parameter="mu")
    fold = BifurcationGUISolutionPoint(0.0, {"u": 0.0}, 0 + 0j, None, 0)
    fold.bifurcation_info = {"type": "fold"}
    branch.append(fold)
    c.branches = [branch]
    c.current_branch = branch
    c.current_point = fold

    assert c.branch_switch() is False
    assert any("fold" in m for m in logged), "it has to say why"

    # And a bifurcation whose normal form was never computed cannot be switched from either.
    fold.bifurcation_info = None
    with pytest.raises(RuntimeError):
        c.branch_switch()


def test_a_step_off_the_end_extends_the_branch():
    """A continuation step from an end point belongs beyond that end, not wherever it fits best.

    Points are ordered along a branch by a search over every insertion index, scored by the length of
    the resulting path. On a branch that curves back towards its own beginning - an isola about to
    close, a hysteresis loop - the shortest path can be the one with the new point at the OTHER end,
    and the branch then no longer holds the points in the order they were computed. Everything read off
    it goes with that: the stability segments, the splines, and the two-point tangent.

    Where the step came from settles it without any of that geometry, so the origin is now passed in.
    Only outwards, though: reversing ds at an end and stepping back over the branch is a legitimate
    thing to do and really does put the point between two others, which is what the search is for.

    On stubs, since none of this touches the solver.
    """
    from pyoomph.utils.bifurcation_gui.controller import BifurcationController, _FixedViewLimits
    from pyoomph.utils.bifurcation_gui.model import (BifurcationGUISolutionBranch,
                                                     BifurcationGUISolutionPoint,
                                                     AXIS_PARAMETER, AXIS_OBSERVABLE)

    # An arc that comes back round towards its start: the last point sits near the first.
    ISOLA = [(0.0, 0.0), (0.6, 0.5), (1.0, 0.0), (0.6, -0.5), (0.1, -0.4)]

    def build():
        c = BifurcationController.__new__(BifurcationController)
        c._on_log = lambda *a: None
        c._on_changed = None
        c._on_status = None
        c._on_busy = None
        c.view = _FixedViewLimits(xlim=(-2.0, 2.0), ylim=(-2.0, 2.0))
        c._x_axis = (AXIS_PARAMETER, "mu")
        c._y_axis = (AXIS_OBSERVABLE, "u")
        c._current_observable = "u"
        c._paramname = "mu"
        branch = BifurcationGUISolutionBranch(kind="solution", continuation_parameter="mu")
        for i, (mu, u) in enumerate(ISOLA):
            p = BifurcationGUISolutionPoint(mu, {"u": u}, -1 + 0j, None, i, param_values={"mu": mu})
            p.scoord = i / (len(ISOLA) - 1)
            branch.append(p)
        c.branches = [branch]
        c.current_branch = branch
        return c, branch

    def place(new, origin_index, pass_origin=True):
        c, branch = build()
        p = BifurcationGUISolutionPoint(new[0], {"u": new[1]}, -1 + 0j, None, 99,
                                        param_values={"mu": new[0]})
        c.reorder_branch_upon_point_insertion(branch, p,
                                              origin=branch[origin_index] if pass_origin else None)
        return branch.index(p), len(branch)

    # Continuing on from the last point, which here heads back towards the first.
    assert place((-0.05, -0.1), -1) == (5, 6), "a step off the end has to become the new end"
    # Without the origin - every other caller, which inserts rather than continues - the search
    # decides as it always did, and here it closes the loop instead.
    assert place((-0.05, -0.1), -1, pass_origin=False)[0] == 0, \
        "the search itself is unchanged; this is the case the origin exists to settle"
    # The same off the other end.
    assert place((-0.4, 0.3), 0) == (0, 6), "a step off the start has to become the new start"
    # And ds reversed at the end: the step points back along the branch, so it is an insertion.
    assert place((0.35, -0.47), -1)[0] == 4, \
        "stepping back over the branch must still be placed by the search"


def test_test_functions_discriminate_fold_from_branch_point():
    """The detection logic itself, on synthetic points, so it does not depend on a sweep's step size.

    A fold reverses the tangent AND flips the determinant; a pitchfork or transcritical point flips only
    the determinant. That difference is the whole reason two test functions are watched - a dp/ds-only
    quick mode would pass a pitchfork in silence. Both crossings are exercised end to end against
    analytic problems by the scratchpad probes; this pins the decision.
    """
    from pyoomph.utils.bifurcation_gui.controller import BifurcationController
    from pyoomph.utils.bifurcation_gui.model import BifurcationGUISolutionPoint

    logged: list = []
    c = BifurcationController.__new__(BifurcationController)   # no Problem needed for the decision
    c._paramname = "mu"
    c._on_log = logged.append
    c._on_changed = None
    c._on_status = None
    c._on_busy = None

    def pt(mu, det, dp):
        return BifurcationGUISolutionPoint(mu, {"u": 0.0}, None, None, 0, det_sign=det, dparam_ds=dp)

    # nothing changed
    a, b = pt(1.0, 1, +1.0), pt(0.9, 1, +1.0)
    c._detect_bifurcation_between(a, b)
    assert b.detected_bifurcation is None

    # determinant only -> a branch point (pitchfork / transcritical)
    a, b = pt(1.0, 1, +1.0), pt(0.9, -1, +1.0)
    c._detect_bifurcation_between(a, b)
    assert b.detected_bifurcation == "branch_point"

    # tangent reversed as well -> a fold
    a, b = pt(1.0, 1, +1.0), pt(0.9, -1, -1.0)
    c._detect_bifurcation_between(a, b)
    assert b.detected_bifurcation == "fold"

    # a Hopf moves neither, and must NOT be reported - this is the documented blind spot
    a, b = pt(1.0, 1, +1.0), pt(0.9, 1, +1.0)
    c._detect_bifurcation_between(a, b)
    assert b.detected_bifurcation is None, "a Hopf is invisible to both test functions"

    # a missing sign never invents a detection
    a, b = pt(1.0, None, None), pt(0.9, -1, +1.0)
    c._detect_bifurcation_between(a, b)
    assert b.detected_bifurcation is None
    assert any("branch_point" in m or "fold" in m for m in logged), "detections are reported"


def test_a_bifurcation_of_a_second_eigenvalue_does_not_look_stable():
    """Passing a bifurcation must not be assumed to swap stable for unstable.

    The Kuramoto-Sivashinsky hexdot branch is the case: past its fold it already has one unstable
    eigenvalue, and the transcritical point at gamma = 0 belongs to a SECOND one - so what follows is
    MORE unstable, not stable. The segmentation used to flip its notion of stability at every located
    bifurcation, which drew that stretch as a stable branch.
    """
    from pyoomph.utils.bifurcation_gui.model import (BifurcationGUISolutionBranch,
                                                     BifurcationGUISolutionPoint)

    def pt(mu, unstable, at_bifurcation=False):
        # A located bifurcation is flagged by an exactly zero leading real part, as tracking leaves it.
        lead = 0 + 0j if at_bifurcation else (0.5 + 0j if unstable else -0.5 + 0j)
        p = BifurcationGUISolutionPoint(mu, {"u": mu}, lead, None, 0)
        p.unstable_count = unstable
        return p

    b = BifurcationGUISolutionBranch(kind="solution", continuation_parameter="mu")
    for mu, n in [(0.4, 1), (0.3, 1), (0.2, 1)]:      # already unstable, from an earlier fold
        b.append(pt(mu, n))
    b.append(pt(0.1, 1, at_bifurcation=True))          # a second eigenvalue crosses here
    for mu, n in [(0.0, 2), (-0.1, 2), (-0.2, 2)]:     # now two unstable, still not stable
        b.append(pt(mu, n))

    _segs, stabs = b.to_branch_stab_list("u")
    assert True not in stabs, \
        "a branch that is unstable on both sides of a bifurcation must not be drawn stable: " + str(stabs)
    assert False in stabs, "and it must still be drawn as unstable"

    # The ordinary case still behaves: stable, a bifurcation, then unstable.
    b2 = BifurcationGUISolutionBranch(kind="solution", continuation_parameter="mu")
    for mu, n in [(0.4, 0), (0.3, 0)]:
        b2.append(pt(mu, n))
    b2.append(pt(0.2, 0, at_bifurcation=True))
    for mu, n in [(0.1, 1), (0.0, 1)]:
        b2.append(pt(mu, n))
    _segs2, stabs2 = b2.to_branch_stab_list("u")
    assert True in stabs2 and False in stabs2, "the plain case must still show both: " + str(stabs2)
    assert stabs2.index(True) < stabs2.index(False), "stable first, then unstable"


def test_stability_indicator_and_the_inferred_toggle():
    """The segmentation reads stability through one accessor, so quick-mode points can take part.

    For a point with a measured spectrum it must return exactly the leading real part, or an ordinary
    diagram would change - that is the regression this guards.
    """
    from pyoomph.utils.bifurcation_gui.model import (BifurcationGUISolutionBranch,
                                                     BifurcationGUISolutionPoint)

    measured = BifurcationGUISolutionPoint(1.0, {"u": 1.0}, -2.5 + 0j, None, 0)
    assert measured.stability_indicator() == -2.5, "unchanged for a measured point"

    quick = BifurcationGUISolutionPoint(0.9, {"u": 0.9}, None, None, 1, det_sign=1, dparam_ds=1.0)
    assert quick.stability_indicator() != quick.stability_indicator(), "unknown reads as NaN"
    quick.stability_source = "inferred"
    quick.unstable_count = 0
    assert quick.stability_indicator() == -1.0, "inferred stable"
    quick.unstable_count = 2
    assert quick.stability_indicator() == +1.0, "inferred unstable"
    # Distrusting the inference must put it back to unknown rather than guessing.
    assert quick.stability_indicator(trust_inferred=False) != quick.stability_indicator(trust_inferred=False)

    # A branch mixing measured and unknown points yields a neutral segment across the pair, which is
    # the style the plot already uses where stability changes.
    b = BifurcationGUISolutionBranch(kind="solution", continuation_parameter="mu")
    b.append(measured)
    b.append(BifurcationGUISolutionPoint(0.9, {"u": 0.9}, None, None, 1))
    b.append(BifurcationGUISolutionPoint(0.8, {"u": 0.8}, -2.0 + 0j, None, 2))
    _segs, stabs = b.to_branch_stab_list("u")
    assert None in stabs, "a pair with an unknown side cannot claim a stability"


def test_legacy_point_has_no_recorded_spectrum():
    """A state file written before the spectrum was recorded must not look as if it had one.

    Reporting an empty list lets the Points tab say "only the leading eigenvalue was recorded" instead
    of presenting that single value as though it were the whole spectrum.
    """
    from pyoomph.utils.bifurcation_gui.model import BifurcationGUISolutionPoint

    legacy = {"param_value": 1.0, "obs_value": {"u": 0.5}, "eig_value_Re": -2.0, "eig_value_Im": 0.5,
              "statefile": None, "outstep": 0, "scoord": 0.0, "tangs": {}}
    p = BifurcationGUISolutionPoint.from_dict(legacy)
    assert p.eig_values == []
    assert p.measured_unstable_count() == 0
    assert (p.eig_value_Re, p.eig_value_Im) == (-2.0, 0.5), "the leading one is still there"

    # A point that really did record its spectrum round-trips it, unstable count included.
    full = BifurcationGUISolutionPoint(1.0, {"u": 0.5}, -2 + 0.5j, None, 0,
                                       eig_values=[-2 + 0.5j, -2 - 0.5j, 0.3 + 0j])
    assert full.measured_unstable_count() == 1
    back = BifurcationGUISolutionPoint.from_dict(full.to_state_dict())
    assert back.eig_values == [-2 + 0.5j, -2 - 0.5j, 0.3 + 0j]
    assert back.measured_unstable_count() == 1


def test_branch_describes_itself():
    """A branch list has to say what each branch IS, not merely how many points it holds.

    "12 points" cannot be told apart from another diagram in a different parameter or from a curve of
    bifurcations, which is exactly what is needed once several are on screen.
    """
    from pyoomph.utils.bifurcation_gui.model import (BifurcationGUISolutionBranch,
                                                     BifurcationGUISolutionPoint)

    def pt(mu, b, u=0.5):
        return BifurcationGUISolutionPoint(mu, {"u": u}, -1.0, None, 0,
                                           param_values={"mu": mu, "b": b})

    sol = BifurcationGUISolutionBranch(kind="solution", continuation_parameter="mu")
    sol.append(pt(1.0, 0.5))
    assert sol.describe() == "1 point | continued in mu | at b = 0.5"
    sol.append(pt(0.9, 0.5))
    assert sol.describe().startswith("2 points | continued in mu")

    locus = BifurcationGUISolutionBranch(kind="locus", continuation_parameter="b",
                                         tracked_parameter="mu", bifurcation_type="fold")
    locus.append(pt(-0.0625, 0.5))
    locus.append(pt(-0.25, 1.0))
    # Both parameters vary along a locus, so nothing is reported as held fixed.
    assert locus.describe() == "2 points | fold locus: mu tracked, continued in b"

    # A single-parameter problem has nothing to hold fixed and must not claim otherwise.
    only = BifurcationGUISolutionBranch(kind="solution", continuation_parameter="r")
    only.append(BifurcationGUISolutionPoint(1.0, {"x": 1.0}, -2.0, None, 0, param_values={"r": 1.0}))
    assert only.describe() == "1 point | continued in r"

    # A pre-slice state file says so rather than pretending nothing was held fixed.
    legacy = BifurcationGUISolutionBranch.from_dict(
        {"points": [{"param_value": 1.0, "obs_value": {"u": 1.0}, "eig_value_Re": -1.0,
                     "eig_value_Im": 0.0, "statefile": None, "outstep": 0, "scoord": 0.0,
                     "tangs": {}}]}, default_continuation_parameter="mu")
    assert legacy.describe() == "1 point | continued in mu | slice unknown"


def test_slice_detects_a_parameter_that_moved():
    """A branch whose supposedly fixed parameters drift must be flagged, not averaged.

    go_to_param() called from a custom key function mid-branch is exactly how this happens.
    """
    from pyoomph.utils.bifurcation_gui.model import (BifurcationGUISolutionBranch,
                                                     BifurcationGUISolutionPoint)

    b = BifurcationGUISolutionBranch(kind="solution", continuation_parameter="mu")
    b.append(BifurcationGUISolutionPoint(1.0, {"u": 1.0}, -1.0, None, 0,
                                         param_values={"mu": 1.0, "b": 0.5}))
    b.append(BifurcationGUISolutionPoint(0.9, {"u": 0.9}, -1.0, None, 1,
                                         param_values={"mu": 0.9, "b": 0.5}))
    assert b.slice_is_consistent()
    b.append(BifurcationGUISolutionPoint(0.8, {"u": 0.8}, -1.0, None, 2,
                                         param_values={"mu": 0.8, "b": 0.9}))
    assert not b.slice_is_consistent(), "b moved along a branch that claims to hold it fixed"


def test_keymap_defaults_and_rebinding(tmp_path):
    # Pure bookkeeping, no Problem involved.
    from pyoomph.utils.bifurcation_gui.actions import (DEFAULT_KEYMAP, KeyMap, format_accelerator)

    path = str(tmp_path / "keymap.json")
    km = KeyMap(path)
    assert km.get("step") == "space"
    assert km.action_for("shift+space") == "multistep"
    assert km.get("tag_7") == "7"

    # Rebinding takes the accelerator away from whoever held it, so no two commands can share one.
    km.set("step", "b")
    assert km.get("step") == "b"
    assert km.get("locate_bifurcation") is None
    assert km.action_for("b") == "step"
    assert km.save()

    reloaded = KeyMap(path)
    assert reloaded.get("step") == "b"
    assert reloaded.get("locate_bifurcation") is None
    assert reloaded.get("multistep") == DEFAULT_KEYMAP["multistep"], "untouched bindings stay"

    reloaded.reset_to_defaults()
    assert reloaded.get("step") == "space"

    assert format_accelerator("shift+space") == "Shift+Space"
    assert format_accelerator("pagedown") == "PageDown"
    assert format_accelerator(None) == ""


def test_import_order_is_free():
    # pyoomph.output.plotting forces the Agg backend. The old bifurcation GUI went through pyplot
    # and raised a RuntimeError telling the user to reorder their imports when it was imported
    # second; the figure is now built through the object-oriented API, so the order is irrelevant.
    import pyoomph.output.plotting  # noqa: F401
    from pyoomph.utils.bifurcation_gui.plotter import BifurcationDiagramPlotter

    plotter = BifurcationDiagramPlotter()
    assert plotter.figure is not None
    assert plotter.get_xscale() == "linear"
    plotter.set_yscale("log")
    assert plotter.get_yscale() == "log"


def test_critical_eigenindex_takes_the_mode_that_just_crossed():
    """Right after a crossing, the mode to track is the one that crossed - not the nearest to zero.

    Two modes going unstable within a few thousandths of each other is what a periodic domain does
    routinely. On a real thin-film branch the spectrum straight after such a step was
    [+0.016507, +0.016484, -0.016089, ...]: nearest the imaginary axis is the STABLE eigenvalue, by 2%.
    Tracking that one asks the fold handler for a bifurcation the branch has not reached yet, from a
    guess belonging to a different mode, and it diverged.

    Where the unstable count did NOT change the old rule is the right one and is kept: on a branch that
    was already unstable when we arrived, the bifurcation ahead belongs to the eigenvalue closest to
    the axis, which is the KS hexdot case the method was written for.

    A stub, because it is a choice of index and nothing else.
    """
    from pyoomph.utils.bifurcation_gui.controller import BifurcationController

    class StubProblem:
        def __init__(self, evs):
            self.evs = evs
        def get_last_eigenvalues(self):
            return self.evs

    class StubPoint:
        def __init__(self, unstable, param):
            self.unstable_count = unstable
            self.param_value = param

    c = BifurcationController.__new__(BifurcationController)
    prev, cur = StubPoint(0, 1.271), StubPoint(2, 1.279)
    c.current_branch = [prev, cur]
    c.current_point = cur

    c.problem = StubProblem([0.016507, 0.016484, -0.016089, -0.03])  # type: ignore[assignment]
    assert c.critical_eigenindex() == 1, "the least unstable of the two that just crossed"

    # Same spectrum, but the step did not change the stability: nothing crossed here, so the next
    # crossing is the eigenvalue nearest the axis.
    cur.unstable_count = prev.unstable_count = 2
    assert c.critical_eigenindex() == 2

    # Already unstable on arrival and staying that way - the eigenvalue about to cross is not the
    # leading one, which is exactly why this method exists.
    c.problem = StubProblem([0.5, -0.01, -0.2])  # type: ignore[assignment]
    assert c.critical_eigenindex() == 1

    # No history to compare against: fall back to the same rule rather than guess.
    c.current_branch, c.current_point = [], None
    assert c.critical_eigenindex() == 1


def test_a_branch_stepped_through_is_not_a_fold():
    """The branch's own geometry answers fold-or-branch-point, and it outranks the normal form.

    A fold is where the parameter turns around, so the two points either side of one lie on the SAME
    side of its parameter value; a branch point sits strictly between them. The normal form decides the
    same question from the projection of dR/dparameter on the left null vector, which is the least
    reliable number in the calculation when a second eigenvalue is nearly critical - and that is
    precisely the situation in which a bifurcation gets located between two ordinary continuation
    points.
    """
    from pyoomph.utils.bifurcation_gui.controller import BifurcationController

    class StubParam:
        def __init__(self, value):
            self.value = value

    class StubProblem:
        def __init__(self, param, evs):
            self.param = param
            self.evs = evs
        def get_last_eigenvalues(self):
            return self.evs

    class StubPoint:
        def __init__(self, param):
            self.param_value = param
            self.unstable_count = None

    c = BifurcationController.__new__(BifurcationController)
    c.current_branch = [StubPoint(1.270959), StubPoint(1.279080)]
    c.current_point = c.current_branch[-1]
    c.problem = StubProblem(StubParam(1.271597), [0.0])  # type: ignore[assignment]
    c.get_bifurcation_parameter = lambda: c.problem.param  # type: ignore[assignment]

    assert c._fold_ruled_out_by_the_branch() == "branch_point"

    # A fold found beyond the pair says nothing - both points are on one side of it, which is what a
    # fold looks like from a continuation.
    c.problem.param.value = 1.2801
    assert c._fold_ruled_out_by_the_branch() is None

    # Landing on a point rather than between them is not evidence either.
    c.problem.param.value = 1.279080
    assert c._fold_ruled_out_by_the_branch() is None

    # A Hopf also passes straight through the parameter, so the test must not claim anything there.
    c.problem.param.value = 1.271597
    c.problem.evs = [0.0 + 3.2j]
    assert c._fold_ruled_out_by_the_branch() is None


def _diagram_with_two_branches():
    """A controller carrying two straight branches, with no problem and no window behind it."""
    from pyoomph.utils.bifurcation_gui.controller import BifurcationController, _FixedViewLimits
    from pyoomph.utils.bifurcation_gui.model import (BifurcationGUISolutionBranch,
                                                     BifurcationGUISolutionPoint)

    def point(mu, u):
        return BifurcationGUISolutionPoint(mu, {"ode/u": u}, -1.0, "", 0,
                                           param_values={"mu": mu})

    c = BifurcationController.__new__(BifurcationController)
    c._on_changed = c._on_status = c._on_busy = None
    c._on_log = lambda *a: None
    c.view = _FixedViewLimits(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    c._paramname = "mu"
    c._x_axis = c._y_axis = None
    c._current_observable = "ode/u"
    c._avail_observables = ["ode/u"]
    first = BifurcationGUISolutionBranch([point(0.1*i, 0.1*i) for i in range(4)],
                                         continuation_parameter="mu")
    second = BifurcationGUISolutionBranch([point(0.3+0.1*i, 0.3+0.1*i) for i in range(1, 4)],
                                          continuation_parameter="mu")
    c.branches = [first, second]
    c.current_branch, c.current_point = first, first[-1]
    c.selected_branch, c.selected_point = first, first[-1]
    return c, first, second


def test_a_fresh_diagram_opens_on_a_window_derived_from_ds():
    """The starting window along the parameter axis: 10 ds ahead of the first point, 2 ds behind it.

    It used to be a 1e-4 box in both directions, which made the very first thing on the screen -- the
    direction arrow, which is ds long -- five hundred times wider than the axes. Ahead is where Step
    is about to go and roughly what a multistep sweep covers before it stops at the border; a little
    behind so the point is not glued to the edge and Reverse has somewhere to land.

    Across the parameter the tiny box stays: nothing is known about the observable's scale before
    anything has been solved, and the view only ever grows as points arrive.
    """
    from pyoomph.utils.bifurcation_gui.model import observable_axis, parameter_axis

    c, first, _second = _diagram_with_two_branches()
    c._tangs = {}
    fresh = type(first)([first[0]], continuation_parameter="mu")
    c.branches = [fresh]
    c.current_branch, c.current_point = fresh, fresh[0]
    mu = float(fresh[0].param_value)
    u = float(fresh[0].obs_values["ode/u"])

    c._last_ds = 0.05
    assert numpy.allclose(c.initial_view_box(),
                          [mu - 0.1, mu + 0.5, u - 1e-4, u + 1e-4]), c.initial_view_box()
    # Stepping the other way opens the window the other way round, not upside down.
    c._last_ds = -0.05
    assert numpy.allclose(c.initial_view_box(),
                          [mu - 0.5, mu + 0.1, u - 1e-4, u + 1e-4]), c.initial_view_box()
    # It follows the axes rather than assuming the parameter is on x.
    c._last_ds = 0.05
    c._x_axis, c._y_axis = observable_axis("ode/u"), parameter_axis("mu")
    assert numpy.allclose(c.initial_view_box(),
                          [u - 1e-4, u + 1e-4, mu - 0.1, mu + 0.5]), c.initial_view_box()
    c._x_axis = c._y_axis = None
    # A zero ds has no direction and no length, so there is nothing to derive a window from.
    c._last_ds = 0.0
    assert numpy.allclose(c.initial_view_box(),
                          [mu - 1e-4, mu + 1e-4, u - 1e-4, u + 1e-4]), c.initial_view_box()


def test_the_default_deflated_scan_is_the_default_window():
    """The default scan covers exactly what a fresh diagram shows, and reports itself as uncut.

    Three defaults have to agree for that: the increment is ds, the count is 10, and the window opens
    initial_view_ds_ahead = 10 ds ahead of the first point. When they did not -- the increment used to
    be 0.05*|parameter|, which is unrelated to ds and to the window -- one step could already leave
    the plot and the tab read "1 step (cut short by the visible range)" before anything had been done.

    Checked over four parameter magnitudes and both directions, because the failure was scale
    dependent: it only showed up where 0.05*|parameter| was large against ds.
    """
    class _ViewFromBox:
        """Stands in for the plotter: the window initialise_view() would have set."""

        def __init__(self, c):
            self.c = c

        def get_xlim(self):
            b = self.c.initial_view_box()
            return (b[0], b[1])

        def get_ylim(self):
            b = self.c.initial_view_box()
            return (b[2], b[3])

        def get_xscale(self):
            return "linear"

        def get_yscale(self):
            return "linear"

    for param, ds in ((20.0, 0.05), (0.5, 0.01), (-3.0, 0.2), (1000.0, 1.0)):
        for signed in (ds, -ds):
            c, first, _second = _diagram_with_two_branches()
            c._tangs = {}
            c._last_ds = signed
            c.deflated_scan_steps = 10
            c.deflated_scan_dparam = None
            fresh = type(first)([type(first[0])(param, {"ode/u": 0.5}, -1.0, "", 0,
                                                param_values={"mu": param})],
                                continuation_parameter="mu")
            c.branches = [fresh]
            c.current_branch, c.current_point = fresh, fresh[0]
            c.get_bifurcation_parameter = lambda p=param: type("P", (), {"value": p})()
            c.view = _ViewFromBox(c)

            values = c.deflated_scan_values()
            assert len(values) - 1 == 10, \
                "mu0={:g} ds={:g}: {:d} steps".format(param, signed, len(values) - 1)
            assert not c.deflated_scan_is_clipped(), \
                "mu0={:g} ds={:g}: the default scan must fit the default window".format(param, signed)
            assert abs(values[-1] - (param + 10*signed)) < 1e-9*max(1.0, abs(param)), values[-1]
            assert (values[-1] > values[0]) == (signed > 0), "the scan must follow the sign of ds"


def test_a_fresh_diagram_shows_which_way_step_will_go():
    """The continuation arrow on a brand-new diagram, where there is no tangent yet.

    The arrow is drawn from the arclength tangent recorded for the plotted observable, and a diagram
    with one point has none -- so it used to be missing at the moment it is most wanted, before the
    first Step, when the question is exactly "which way?". A first step moves the parameter and, to
    first order, nothing else, so the fallback is a unit vector along whichever axis carries the
    parameter; the plotter multiplies by ds, which draws (ds, 0) on the ordinary diagram.

    The fallback must NOT reach a located bifurcation, where _tangs is emptied on purpose and the
    arrows worth drawing are the departure directions the plotter draws itself.
    """
    from pyoomph.utils.bifurcation_gui.model import observable_axis, parameter_axis

    c, first, second = _diagram_with_two_branches()
    c._tangs = {}
    c._last_ds = 0.05

    # A branch of one point: no tangent, so the parameter axis is the answer.
    fresh = type(first)([first[0]], continuation_parameter="mu")
    c.branches = [fresh]
    c.current_branch, c.current_point = fresh, fresh[0]
    assert numpy.allclose(c.plotted_tangent(), [1.0, 0.0]), c.plotted_tangent()
    # Reversing ds does not flip the VECTOR -- the plotter multiplies by ds and gets the sign there.
    c._last_ds = -0.05
    assert numpy.allclose(c.plotted_tangent(), [1.0, 0.0]), c.plotted_tangent()
    # ... and it follows the axes rather than assuming the parameter is on x.
    c._x_axis, c._y_axis = observable_axis("ode/u"), parameter_axis("mu")
    assert numpy.allclose(c.plotted_tangent(), [0.0, 1.0]), c.plotted_tangent()
    c._x_axis = c._y_axis = None

    # A real tangent always wins over the fallback.
    c._tangs = {"ode/u": numpy.array([0.3, 0.9])}
    assert numpy.allclose(c.plotted_tangent(), [0.3, 0.9])

    # A branch that has points but no tangent is a located bifurcation: no fallback arrow there.
    c._tangs = {}
    c.branches = [first]
    c.current_branch, c.current_point = first, first[-1]
    assert c.plotted_tangent() is None, "the fallback must not reach a bifurcation"


def test_a_branch_can_be_split_at_a_point():
    """Cutting a branch in two, for when a continuation step landed on a different one.

    The selected point starts the new half - that is the one the user identifies, being the first that
    does not belong. Splitting at the first point of a branch would leave nothing behind and has to be
    refused rather than produce an empty branch that every later operation has to guard against.
    """
    c, first, _ = _diagram_with_two_branches()
    c.selected_branch, c.selected_point = first, first[2]

    tail = c.split_branch()
    assert list(first) == [first[0], first[1]], "the old branch keeps everything before the point"
    assert len(tail) == 2 and tail[0] is c.selected_point, "and the point starts the new one"
    assert c.branches.index(tail) == c.branches.index(first)+1, "the new half follows the old one"
    assert tail.continuation_parameter == "mu", "a split half is still the same kind of branch"
    # The problem is still sitting on the point it was, so that half is what a step would extend.
    assert c.current_branch is tail and c.current_point in tail

    c.selected_point = tail[0]
    try:
        c.split_branch()
    except RuntimeError as e:
        assert "first point" in str(e)
    else:
        raise AssertionError("splitting at the first point of a branch has to be refused")


def test_stability_is_not_inferred_across_a_point_with_no_test_function():
    """Propagation needs a test function at every point it walks across, not just at the ends.

    The argument for carrying an unstable count along a branch is that sign(det J) flips exactly when
    an odd number of real eigenvalues crosses zero -- which says nothing at all about a point that has
    no det_sign and no dparam_ds either. A DEFLATED SCAN records exactly such points: it steps the
    parameter with no arclength control and no test function, in precisely the regions where branches
    appear and disappear. Carrying the count across them would not be an inference, it would be an
    assumption drawn in the same colour as a measurement.

    Quick mode is unaffected -- its points always carry one of the two -- and that is asserted here
    too, because the cheap way to "fix" this would be to stop propagating anywhere.
    """
    from pyoomph.utils.bifurcation_gui.controller import BifurcationController
    from pyoomph.utils.bifurcation_gui.model import (BifurcationGUISolutionBranch,
                                                     BifurcationGUISolutionPoint, STABILITY_EIGEN,
                                                     STABILITY_INFERRED, STABILITY_UNKNOWN)

    def point(mu, u, det=None, dparam=None, eig=None):
        return BifurcationGUISolutionPoint(mu, {"ode/u": u}, eig, "", 0,
                                           param_values={"mu": mu},
                                           eig_values=[eig] if eig is not None else None,
                                           det_sign=det, dparam_ds=dparam)

    def propagate(branch):
        c = BifurcationController.__new__(BifurcationController)
        c._on_changed = c._on_status = c._on_busy = None
        c._on_log = lambda *a: None
        c._paramname = "mu"
        c.branches = [branch]
        c.propagate_stability(branch)
        return branch

    # Quick mode: one measured anchor, then determinant signs. Still inferred, still flipped where
    # the sign changes.
    b = propagate(BifurcationGUISolutionBranch(
        [point(0.0, 0.0, eig=-1.0), point(0.1, 0.1, det=1), point(0.2, 0.2, det=-1),
         point(0.3, 0.3, det=-1)]))
    assert [p.stability_source for p in b] == [STABILITY_EIGEN] + [STABILITY_INFERRED]*3
    assert [p.unstable_count for p in b] == [0, 0, 1, 1]

    # "Folds only": dparam_ds and no determinant is still a test function.
    b = propagate(BifurcationGUISolutionBranch(
        [point(0.0, 0.0, eig=-1.0), point(0.1, 0.1, dparam=0.5), point(0.2, 0.2, dparam=0.5)]))
    assert all(p.stability_source == STABILITY_INFERRED for p in b[1:])

    # A deflated scan: neither, so the points stay unknown however close the measured one is.
    b = propagate(BifurcationGUISolutionBranch(
        [point(0.0, 0.0, eig=-1.0), point(0.5, 0.7), point(1.0, 1.0)]))
    assert b[0].stability_source == STABILITY_EIGEN
    assert all(p.stability_source == STABILITY_UNKNOWN and p.unstable_count is None for p in b[1:])

    # The blind spot hides what is beyond it, but only from that side: a point past it is still
    # reached from the anchor on the other end.
    b = propagate(BifurcationGUISolutionBranch(
        [point(0.0, 0.0, eig=-1.0), point(0.5, 0.7), point(1.0, 1.0, det=1), point(1.5, 1.2, eig=2.0)]))
    assert b[1].stability_source == STABILITY_UNKNOWN
    assert b[2].stability_source == STABILITY_INFERRED


def test_a_branch_can_be_deleted_whole():
    """Deleting a branch removes it and nothing else, and the last one is refused.

    The refusal is the part worth pinning: with no branches left the problem would be sitting on a
    solution the diagram does not know about, and every command that reads current_point would be
    working from a stale one. Deleting the branch the problem is ON is the other half and needs a real
    Problem to reload from, so it lives in delete_branch_worker.py.
    """
    c, first, second = _diagram_with_two_branches()
    # The one that is NOT current, so nothing has to be reloaded.
    c.selected_branch, c.selected_point = second, second[0]
    npoints = len(second)
    assert c.delete_branch() == npoints
    assert c.branches == [first], "only the deleted branch may go"
    assert c.current_branch is first and c.current_point is first[-1], \
        "deleting another branch must not move the problem"
    assert c.selected_branch is None and c.selected_point is None, \
        "the selection pointed into the branch that is gone"

    with pytest.raises(RuntimeError, match="last branch"):
        c.delete_branch()
    assert c.branches == [first], "the refusal must leave the diagram untouched"


def test_two_branches_can_be_merged_by_the_ends_that_meet():
    """Joining two branches that are one curve, and refusing the ones that are not.

    Which of the four end-to-end pairings is used is decided by distance in the plotted coordinates,
    so the caller does not have to know which branch was computed in which direction.
    """
    c, first, second = _diagram_with_two_branches()
    c.selected_branch, c.selected_point = second, second[0]

    merged = c.merge_branches()
    assert merged is first, "the branch you are on survives, the other one goes"
    assert len(c.branches) == 1
    assert [round(p.param_value, 6) for p in merged] == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], \
        "first[-1] meets second[0], so they are laid end to end in that order"
    assert all(merged[i].scoord <= merged[i+1].scoord for i in range(len(merged)-1)), \
        "and the arclength coordinates are renormalised over the whole thing"

    # A branch computed the other way round has to be turned over rather than joined backwards.
    c, first, second = _diagram_with_two_branches()
    second.data = list(reversed(second.data))
    c.selected_branch = second
    merged = c.merge_branches()
    assert [round(p.param_value, 6) for p in merged] == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    # Different slices of parameter space are different physical results, not one curve.
    c, first, second = _diagram_with_two_branches()
    for p in second:
        p.param_values = {"mu": p.param_value, "b": 0.5}
    for p in first:
        p.param_values = {"mu": p.param_value, "b": 0.25}
    c.selected_branch = second
    try:
        c.merge_branches()
    except RuntimeError as e:
        assert "slice" in str(e), str(e)
    else:
        raise AssertionError("branches from different slices must not be merged")


def test_the_branch_switch_offset_follows_the_step_size():
    """The offset a branch switch steps off by comes from the ds the user has set.

    ds is an arclength step, so what it buys in the parameter is ds*|dparameter/ds| - and the tangent
    AT a located bifurcation is gone, because the normal-form calculation resets the continuation
    state, so the one recorded at the point before it is what there is to ask. The old fixed 2% of the
    parameter is the fallback, and on a diagram spanning 4% of the parameter it overshoots every
    branch in it.
    """
    from pyoomph.utils.bifurcation_gui.controller import BifurcationController

    class StubParam:
        value = 1.2716

    class StubProblem:
        def get_last_eigenvalues(self):
            return [0.0]

    class StubPoint:
        def __init__(self, tangs):
            self._tangs = tangs
            self.param_value = 1.271
            self.unstable_count = None

    c = BifurcationController.__new__(BifurcationController)
    c.problem = StubProblem()  # type: ignore[assignment]
    c.get_bifurcation_parameter = lambda: StubParam()  # type: ignore[assignment]
    c._current_observable = "obs"
    c.branch_switch_offset = 0.02
    c._last_ds = 0.0173
    prev = StubPoint({"obs": numpy.array([0.7071, 2.3e-4])})
    c.current_branch = [prev, StubPoint({})]
    c.current_point = c.current_branch[-1]

    assert abs(c.branch_switch_parameter_offset() - 0.0173*0.7071) < 1e-12

    # No tangent recorded anywhere: back to the old rule, and it must not be zero.
    c.current_branch = []
    c.current_point = None
    assert abs(c.branch_switch_parameter_offset() - 0.02*1.2716) < 1e-12


def test_arclength_metric_radio_group(tmp_path):
    """The GUI starts in the mass-matrix metric and the menu's dot says which one is in force.

    oomph's dof-sum norm is mesh-dependent - the same ds buys a different step after a refinement -
    so the controller sets the "l2" inner product in its constructor. The three menu entries are one
    radio group whose dot is re-read from the problem after every command, so a failed or
    self-adjusting switch cannot leave it pointing at a metric that is not active.
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "arclength_metric_menu_worker.py")
    proc = subprocess.run(gui_launch_prefix() + [sys.executable, worker],
                          cwd=here, capture_output=True, text=True, timeout=300)
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, "worker failed:\n" + out[-3000:]
    assert "PYOOMPH_WORKER_DONE" in out, "worker did not finish:\n" + out[-3000:]
    assert "ARCLENGTH METRIC OK" in out


def test_deflation_finds_the_branch_arclength_cannot_reach(tmp_path):
    """The Deflation tab's two commands, against a problem whose whole solution set is known.

    du/dt = mu - u^2 has exactly the two solutions u = +-sqrt(mu), and arclength continuation started
    on one of them reaches the other only by going round the fold at mu = 0. Deflation has to find it
    standing still - and, because there are only two, a SECOND deflated solve has to find nothing,
    which is what separates deflation from perturbing and re-converging onto the branch one is on.

    Out of process because it needs its own Problem, following the worker pattern the rest of this
    suite uses; no window is involved, the controller is driven directly.
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "deflation_gui_worker.py")
    proc = subprocess.run([sys.executable, worker, "--outdir", os.path.join(str(tmp_path), "out")],
                          cwd=here, capture_output=True, text=True, timeout=900)
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, "worker failed:\n" + out[-3000:]
    assert "PYOOMPH_WORKER_DONE" in out, "worker did not finish:\n" + out[-3000:]
    assert "DEFLATION GUI OK" in out


def test_deleting_the_branch_the_problem_is_on(tmp_path):
    """The other half of test_a_branch_can_be_deleted_whole, which needs a real Problem.

    Deleting the branch the problem is sitting on has to leave it loaded somewhere else -- not merely
    re-pointed at another branch, the dofs have to be those of the point it landed on, or the next
    step continues from a solution the diagram does not show. The state dumps must be gone too, which
    is what makes this the one diagram command reloading cannot undo.
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "delete_branch_worker.py")
    proc = subprocess.run([sys.executable, worker, "--outdir", os.path.join(str(tmp_path), "out")],
                          cwd=here, capture_output=True, text=True, timeout=900)
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, "worker failed:\n" + out[-3000:]
    assert "PYOOMPH_WORKER_DONE" in out, "worker did not finish:\n" + out[-3000:]
    assert "DELETE BRANCH OK" in out


def test_deflation_tab_settings_round_trip(tmp_path):
    """What is typed into the Deflation tab reaches the controller, and a refresh brings it back.

    The tab's variables only exist once the window is built, i.e. tk.Tk(), so this runs as a worker
    under gui_launch_prefix() like the other window-opening tests. It also pins the two settings whose
    empty value means something other than zero - the random seed (unseeded) and the Newton cap (the
    problem's own) - and that a refused value leaves the old one in place rather than a default.
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "deflation_tab_worker.py")
    proc = subprocess.run(gui_launch_prefix() + [sys.executable, worker],
                          cwd=here, capture_output=True, text=True, timeout=300)
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, "worker failed:\n" + out[-3000:]
    assert "PYOOMPH_WORKER_DONE" in out, "worker did not finish:\n" + out[-3000:]
    assert "DEFLATION TAB OK" in out


def test_every_default_shortcut_reaches_a_command(tmp_path):
    """A key in the default map with no action behind it does nothing, silently.

    _on_key looks the accelerator up, finds no action of that id and returns - which is how "Grab
    selected point" spent its life reachable only from a checkbox in the settings panel. Building the
    window is enough to check this; it needs no problem and is never shown, but it is tk.Tk() all the
    same, hence a worker under gui_launch_prefix() rather than an in-process test.
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "shortcut_worker.py")
    proc = subprocess.run(gui_launch_prefix() + [sys.executable, worker],
                          cwd=here, capture_output=True, text=True, timeout=300)
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, "worker failed:\n" + out[-3000:]
    assert "PYOOMPH_WORKER_DONE" in out, "worker did not finish:\n" + out[-3000:]
    assert "SHORTCUTS OK" in out


def test_closing_the_window_lets_go_of_the_problem(tmp_path):
    """Closing the window must leave nothing of pyoomph behind it.

    The field plots are pyplot-managed figures, so matplotlib's process-wide registry held every
    pane - and with it the plotter, the mesh data cache, the meshes and the Problem - once the window
    was gone. Nothing frees such a graph afterwards: what the window registered with Tcl is held by
    the interpreter, and the references a C++ object holds through nanobind are invisible to the
    cyclic collector, so a graph joining the two is beyond it for good and nanobind reports it as
    leaked instances while the interpreter shuts down.
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "gui_close_teardown_worker.py")
    proc = subprocess.run(gui_launch_prefix() + [sys.executable, worker, "--outdir", str(tmp_path)],
                          cwd=here, capture_output=True, text=True, timeout=600)
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, "worker failed:\n" + out[-3000:]
    assert "PYOOMPH_WORKER_DONE" in out, "worker did not finish:\n" + out[-3000:]
    assert "GUI CLOSE TEARDOWN OK" in out
    assert "nanobind: leaked" not in out, "C++ objects outlived the process:\n" + out[-3000:]


def test_a_window_that_fails_to_open_is_not_left_standing(tmp_path):
    """A window is never freed unless it is destroyed, so a failed set-up has to destroy it.

    Building the window reads the controller (the axis menus do), and a session not far enough along
    for that used to leave the whole window standing behind the exception - 61 leaked nanobind
    instances at exit, measured, for a session that had not even started.
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "gui_close_teardown_worker.py")
    proc = subprocess.run(gui_launch_prefix() + [sys.executable, worker, "--phase", "raises"],
                          cwd=here, capture_output=True, text=True, timeout=300)
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, "worker failed:\n" + out[-3000:]
    assert "PYOOMPH_WORKER_DONE" in out, "worker did not finish:\n" + out[-3000:]
    assert "GUI FAILED-SETUP TEARDOWN OK" in out
    assert "nanobind: leaked" not in out, "C++ objects outlived the process:\n" + out[-3000:]
