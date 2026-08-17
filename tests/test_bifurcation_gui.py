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
import sys

import pytest

import numpy

from pyoomph import Problem, ODEEquations, InitialCondition
from pyoomph.expressions import var_and_test, partial_t
from pyoomph.utils.bifurcation_gui import BifurcationGUI
from pyoomph.utils.bifurcation_gui.controller import (STATE_VERSION, BifurcationController,
                                                     _FixedViewLimits)


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

    A plain "if" has no closing hook, so the window is opened from Problem.release() - which is also
    reached when the block exits through an exception, because Problem.__exit__'s
    isinstance(type,Exception) test compares a class against Exception and never holds. The "raises"
    phase pins that a failing script gets its traceback rather than a window, and "nowith" pins the
    atexit fallback for scripts that never use "with SomeProblem() as problem".
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "must_init_worker.py")
    outdir = os.path.join(str(tmp_path), "out")

    def run(phase, out_sub="out", expect_mu=None):
        env = dict(os.environ)
        if expect_mu is not None:
            env["PYOOMPH_EXPECT_MU"] = repr(expect_mu)
        proc = subprocess.run([sys.executable, worker, "--phase", phase,
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
    assert "NOWITH opened=True" in nowith, "atexit must open the window:\n" + nowith[-2000:]


def test_extremum_observables_become_axis_choices(tmp_path):
    """Each ExtremumObservables entry offers its minimum, its maximum, and where each of them sits.

    Checked against a field whose extrema are unique and known in closed form, with a phase that moves
    with the parameter so the position is a curve rather than a constant. Uniqueness is the whole
    difficulty in testing a position: cos(2pi*x/Lx)*cos(2pi*y/Ly) has five maxima of equal height, so a
    correct implementation reports a position other than the one written down.

    Out of process, since it needs its own Problem. The worker also pins that six axis choices cost two
    mesh sweeps rather than six, that the "[max, x]" tag replaces the "[obs]" one instead of stacking
    with it, and that a name containing spaces, a comma and brackets survives the CSV export and a
    save/load round trip.
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "extremum_observable_worker.py")
    proc = subprocess.run([sys.executable, worker, "--outdir", os.path.join(str(tmp_path), "out")],
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
