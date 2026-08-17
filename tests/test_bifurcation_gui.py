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
# The test problem is the textbook saddle-node, du/dt = mu - u^2: two stationary branches
# u = +-sqrt(mu) meeting at a fold at mu = 0, with eigenvalue -2u, so the analytic answer for every
# assertion below is known in closed form. That matters most for locate_bifurcation(), which is
# checked against mu = 0 rather than against itself.
#
# Only ONE Problem is constructed per process: a second one segfaults in the JIT loader, which is
# why the module has a single test function rather than one per aspect.

import json
import os

import numpy

from pyoomph import Problem, ODEEquations, InitialCondition
from pyoomph.expressions import var_and_test, partial_t
from pyoomph.utils.bifurcation_gui import BifurcationGUI
from pyoomph.utils.bifurcation_gui.controller import BifurcationController, _FixedViewLimits


class FoldEquations(ODEEquations):
    """du/dt = mu - u^2 : a saddle-node at mu=0, stable branch u=+sqrt(mu)."""

    def __init__(self, mu):
        super().__init__()
        self.mu = mu

    def define_fields(self):
        self.define_ode_variable("u")

    def define_residuals(self):
        u, u_test = var_and_test("u")
        self.add_weak(partial_t(u) - (self.mu - u**2), u_test)


class FoldProblem(Problem):
    def define_problem(self):
        mu = self.get_global_parameter("mu")
        eqs = FoldEquations(mu)
        eqs += InitialCondition(u=1.0)
        self += eqs @ "ode"


def test_bifurcation_gui_controller(tmp_path):
    with FoldProblem() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.get_global_parameter("mu").value = 1.0

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
        assert numpy.isclose(p0.obs_values["ode/u"], 1.0, atol=1e-8)
        # The stable branch: eigenvalue -2u = -2
        assert numpy.isclose(p0.eig_value_Re, -2.0, atol=1e-6)
        assert os.path.isfile(p0.statefile)

        # ---------------------------------------------------------- multistep on a fresh diagram
        # Regression: multistep read the continuation tangent that only step() ever fills in, so on
        # a diagram that had not been stepped yet it raised KeyError - and once that was fixed, a
        # zero-length tangent scaled ds down to nothing and the sweep never terminated.
        c.view = _FixedViewLimits(xlim=(0.99, 1.01), ylim=(0.99, 1.01))  # left by the first step
        c.multistep()
        assert len(c.branches[0]) == 2
        assert c.ds != 0.0
        c.view = _FixedViewLimits(xlim=(-1.0, 2.0), ylim=(-2.0, 2.0))

        # ---------------------------------------------------------- continuation
        for _ in range(6):
            c.step()
        assert len(c.branches[0]) == 8
        for p in c.branches[0]:
            # every point must sit on u = +sqrt(mu)
            assert numpy.isclose(p.obs_values["ode/u"], numpy.sqrt(p.param_value), atol=1e-7)
        # stepping with a negative ds walks towards the fold
        assert c.branches[0][-1].param_value < 1.0 or c.branches[0][0].param_value < 1.0
        # scoords are normalized and ordered by the insertion heuristic
        scoords = [p.scoord for p in c.branches[0]]
        assert scoords == sorted(scoords)

        # ---------------------------------------------------------- the fold, against theory
        c.locate_bifurcation()
        fold = c.current_point
        assert fold is not None
        assert numpy.isclose(fold.param_value, 0.0, atol=1e-7), "fold should be at mu=0"
        assert numpy.isclose(fold.obs_values["ode/u"], 0.0, atol=1e-5)
        assert fold.eig_value_Re == 0, "a located bifurcation is flagged by a zero real part"
        # Bifurcation tracking must not stay active afterwards, or every later solve would be
        # augmented.
        assert not problem.get_bifurcation_tracking_mode()

        # the stability split now has the fold as its boundary
        segs, stabs = c.branches[0].to_branch_stab_list("ode/u")
        assert len(segs) == len(stabs) and len(segs) >= 1

        # ---------------------------------------------------------- tags and export
        c.toggle_point_tag(c.branches[0][0], 3)
        assert c.branches[0][0].tag == 3
        outdir = c.output_curves()
        assert os.path.isfile(os.path.join(outdir, "tag_03.txt"))
        assert os.path.isfile(os.path.join(outdir, "tag03.dump"))
        bdir = os.path.join(outdir, "branch000")
        assert any(f.startswith("smoothed_") for f in os.listdir(bdir))
        assert any(f.startswith("bifurcation_") for f in os.listdir(bdir))

        # ---------------------------------------------------------- save / load round trip
        c.save_all()
        statefile = os.path.join(problem.get_output_directory(c.data_subdir), "state.json")
        assert os.path.isfile(statefile)
        with open(statefile) as f:
            stored = json.load(f)
        assert stored["current_observable"] == "ode/u"
        assert stored["xlim"] == [-1.0, 2.0]

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
