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

        before = [(p.param_value, p.obs_values["ode/u"], p.eig_value_Re, p.scoord, p.tag)
                  for b in c.branches for p in b]

        applied: list = []
        c.load_all(apply_view=applied.append)
        assert applied and applied[0]["statestep"] == stored["statestep"]
        after = [(p.param_value, p.obs_values["ode/u"], p.eig_value_Re, p.scoord, p.tag)
                 for b in c.branches for p in b]
        assert before == after, "reloading must reproduce the diagram exactly"
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
