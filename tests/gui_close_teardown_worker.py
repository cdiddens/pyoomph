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

# Closing the window has to let go of the problem.
#
# The field plots are pyplot-managed figures (panes.py explains why they have to be), so matplotlib's
# process-wide registry held every pane - and through it the plotter, the mesh data cache, the meshes
# and the Problem - for the rest of the process once the window was gone. The C++ objects behind them
# were then freed, if at all, during interpreter finalisation, and whatever the shutdown order left
# behind was reported by nanobind as leaked instances.
#
# Reachability from matplotlib's Gcf is what this checks, rather than the leak message itself: the
# message is printed after the interpreter has finished with this process, and whether it appears at
# all depends on finalisation order, which is exactly the fragility the teardown removes.
#
# The "raises" phase covers the other half: a window that is never destroyed at all. Nothing frees
# it - its callbacks are held by the Tcl interpreter, which the cyclic collector cannot see into - so
# everything it references lives to the end of the process. Building the window reads the controller
# (the axis menus do), so a session that is not far enough along for that used to leave a whole
# window standing behind the exception, and 61 leaked instances behind that.
#
# A worker rather than an in-process test: the window is a real tk.Tk() (run under xvfb-run by the
# test), and the problem has to be the only one in its process.

import argparse
import gc
import sys
import types
import weakref

from pyoomph import Problem, Equations, InitialCondition, DirichletBC
from pyoomph.expressions import var_and_test, var, grad, partial_t
from pyoomph.equations.generic import IntegralObservables
from pyoomph.meshes.simplemeshes import RectangularQuadMesh
from pyoomph.output.plotting import MatplotlibPlotter
from pyoomph.utils.bifurcation_gui import BifurcationGUI
from pyoomph.utils.bifurcation_gui.controller import BifurcationController
from pyoomph.utils.bifurcation_gui.plotter import BifurcationDiagramPlotter
from pyoomph.utils.bifurcation_gui.tkapp import BifurcationTkApp


class Diffusion(Equations):
    def __init__(self, mu):
        super().__init__()
        self.mu = mu

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_weak(partial_t(u), v)
        self.add_weak(grad(u), grad(v))
        self.add_weak(-self.mu*u - 1, v)


class Plot(MatplotlibPlotter):
    def define_plot(self):
        self.set_view(-0.05, -0.05, 1.05, 1.05)
        self.add_plot("domain/u", colorbar=self.add_colorbar("u", position="top right"))


class Prob(Problem):
    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(size=[1.0, 1.0], N=[3, 3], name="domain"))
        eqs = Diffusion(self.get_global_parameter("mu"))
        eqs += InitialCondition(u=0)
        eqs += IntegralObservables(uavg=var("u"))
        for b in ("left", "right", "top", "bottom"):
            eqs += DirichletBC(u=0) @ b
        self += eqs @ "domain"


def walk(root, maxdepth=16):
    """Every object reachable from root, without escaping into module globals.

    Functions are followed through their closure cells only, and modules and classes not at all:
    every function defined in a pyoomph module carries that module's globals, so a walk that took
    those edges would "reach" the unit system and the coordinate systems from anything at all.
    """
    seen, stack = {}, [(root, 0)]
    while stack:
        obj, depth = stack.pop()
        if id(obj) in seen or depth > maxdepth:
            continue
        seen[id(obj)] = obj
        module = type(obj).__dict__.get("__module__", "")
        if isinstance(module, str) and module.startswith("pyoomph._pyoomph_core"):
            continue    # a C++ object's referents are not Python's business
        if isinstance(obj, (types.ModuleType, type, types.BuiltinFunctionType)):
            continue
        if isinstance(obj, types.FunctionType):
            referents = list(obj.__closure__ or ())
        elif isinstance(obj, types.MethodType):
            referents = [obj.__self__]
        else:
            try:
                referents = gc.get_referents(obj)
            except Exception:
                referents = []
        for r in referents:
            stack.append((r, depth + 1))
    return seen


def pyoomph_names(reached):
    """The pyoomph types among the reached objects, C++ ones under their nanobind module name."""
    names = set()
    for obj in reached.values():
        module = type(obj).__dict__.get("__module__", "")
        if isinstance(module, str) and module.startswith("pyoomph"):
            names.add(module + "." + type(obj).__name__)
    return names


class StubProblem:
    """What the controller's constructor and the menu builders touch, and nothing else."""

    _runmode = "overwrite"
    write_states = False
    continuation_data_in_states = False
    plotter = None
    _arclength_inner_product = None

    def get_global_parameter_names(self):
        return ["mu"]

    def is_initialised(self):
        return True

    def set_arclength_inner_product(self, kind):
        pass


def phase_raises() -> int:
    """A window whose set-up fails must not be left standing."""
    import tkinter

    app = BifurcationTkApp(BifurcationController(StubProblem(), "mu"), BifurcationDiagramPlotter(),
                           title="never started")
    app.root.withdraw()
    try:
        app.run()    # the axis menus read a controller that was never start()ed
    except Exception:
        pass
    else:
        raise AssertionError("the stub session was supposed to fail on the way into the window")
    assert tkinter._default_root is None, "the window is still standing after run() raised"
    print("GUI FAILED-SETUP TEARDOWN OK: no window is left standing")
    print("PYOOMPH_WORKER_DONE")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="teardown", choices=["teardown", "raises"])
    ap.add_argument("--outdir")
    args = ap.parse_args()

    if args.phase == "raises":
        return phase_raises()
    assert args.outdir is not None, "--outdir is required for the teardown phase"

    app_ref = []

    original_run = BifurcationTkApp.run

    def run(self):
        # Close the window from the event loop, i.e. through exactly the path the window manager's
        # close button takes, once one step has been taken and the pane has drawn the solution.
        app_ref.append(weakref.ref(self))

        def close():
            self.controller.step()
            self.refresh()
            self._refresh_plots()
            assert self.plot_panes.has_any(), "the problem's plotter should have given us a pane"
            self._on_close()

        self.root.after(300, close)
        original_run(self)

    BifurcationTkApp.run = run

    with Prob() as problem:
        problem.set_output_directory(args.outdir)
        problem.plotter = Plot(problem)
        problem.get_global_parameter("mu").value = 1.0
        gui = BifurcationGUI(problem, "mu")
        gui.neigen = 2
        if gui.must_init():
            problem.solve()
        gui.start(0.001)

        import matplotlib.pyplot as plt
        from matplotlib._pylab_helpers import Gcf

        assert plt.get_fignums() == [], \
            "the pane figures are still registered with pyplot: " + repr(plt.get_fignums())
        from_gcf = walk(Gcf.figs)
        assert id(problem) not in from_gcf, "the problem is still reachable from pyplot's registry"
        left = pyoomph_names(from_gcf)
        assert not left, "still reachable from matplotlib's global figure registry: " + repr(sorted(left))
        assert gui.app is None, "the facade still holds the closed window"
        assert gui.controller._on_changed is None, "the controller still calls back into the window"

        # The window object itself may well survive: what it registered with Tcl is held by the
        # interpreter, which the cyclic collector cannot see into. What it must not do is take any of
        # pyoomph with it - that is what turns into leaked nanobind instances at exit, which the test
        # around this worker checks for in the output.
        gc.collect()
        window = app_ref[0]()
        if window is not None:
            # The window object itself may well survive: what it registered with Tcl is held by the
            # interpreter, which the cyclic collector cannot see into, and neither can it see the
            # Python references a C++ object holds through nanobind - so an object graph joining the
            # two is beyond the collector for good. What must not survive with it is any of pyoomph:
            # that is what turns into the leaked instances nanobind reports at exit, which the test
            # around this worker looks for in the output.
            reached = walk(window)
            assert id(problem) not in reached, "the closed window still holds the problem"
            held = {n for n in pyoomph_names(reached) if not n.startswith("pyoomph.utils.bifurcation_gui.")}
            assert not held, "the closed window still holds: " + repr(sorted(held))
            assert not any(n.endswith("BifurcationController") for n in pyoomph_names(reached)), \
                "the closed window still holds the controller"

    print("GUI CLOSE TEARDOWN OK: nothing of the problem outlives the window")
    print("PYOOMPH_WORKER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
