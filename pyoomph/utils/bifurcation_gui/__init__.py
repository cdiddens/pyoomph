from __future__ import annotations
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

"""Interactive construction of bifurcation diagrams.

Typical use is unchanged from the earlier, purely key-driven version::

    from pyoomph.utils.bifurcation_gui import BifurcationGUI

    with MyProblem() as problem:
        problem.setup_for_stability_analysis()
        gui=BifurcationGUI(problem,"my_parameter")
        gui.neigen=30
        gui.classify_bifurcations=True
        if gui.must_init():          # False if a stored diagram is found, which is loaded instead
            problem.solve()          # whatever it takes to reach the first stationary solution
        gui.start(0.001)             # opens the window; the argument is the initial arclength step

:py:meth:`BifurcationGUI.start` now opens a window with menus, a toolbar and side panels instead of
a bare matplotlib figure; all the old keys still work and are listed next to their menu entries.
They can be rebound under Settings -> Keyboard shortcuts.

The parts are separated so that everything except the window can be driven headlessly:

* :py:mod:`~pyoomph.utils.bifurcation_gui.model` - solution points and branches,
* :py:mod:`~pyoomph.utils.bifurcation_gui.controller` - continuation, bifurcation tracking, storage,
* :py:mod:`~pyoomph.utils.bifurcation_gui.plotter` - the matplotlib rendering,
* :py:mod:`~pyoomph.utils.bifurcation_gui.tkapp` - the tkinter/ttk user interface,
* :py:mod:`~pyoomph.utils.bifurcation_gui.actions` - commands and their keyboard shortcuts,
* :py:mod:`~pyoomph.utils.bifurcation_gui.panes` - the problem's own field plots, shown live.

If the problem carries a plotter (or a list of them, which ``Problem.plotter`` accepts), each one is
rendered next to the diagram and re-drawn as you move along a branch, so there is no need to leave the
window to see what the solution looks like. Eigenfunction views are derived from the same plot
definition on demand under *View -> Field plots*.
"""

from ...generic import Problem
from ... import _pyoomph_core as _pyoomph
from ...typings import *

from .model import BifurcationGUISolutionPoint, BifurcationGUISolutionBranch
from .controller import BifurcationController
from .plotter import BifurcationDiagramPlotter


class BifurcationGUI:
    """Builds a bifurcation diagram of ``problem`` in ``parameter``, interactively.

    Attributes that user scripts customize before :py:meth:`start` (``neigen``, ``shift``,
    ``classify_bifurcations``, ``custom_key_functions``, ...) are forwarded to the underlying
    :py:class:`~pyoomph.utils.bifurcation_gui.controller.BifurcationController`.
    """

    def __init__(self,problem:Problem,parameter:"str | _pyoomph.GiNaC_GlobalParam | None"=None) -> None:
        self.controller=BifurcationController(problem,parameter)
        self.plotter=BifurcationDiagramPlotter()
        self.controller.view=self.plotter
        self.app=None
        #: Extra commands, keyed by the shortcut that triggers them. Each is called with this GUI
        #: and also shows up in the "Custom" menu.
        self.custom_key_functions:dict[str,Callable[[BifurcationGUI],None]]={}
        #: Window title; ``None`` picks a default naming the problem's parameter.
        self.title:str | None=None
        #: Reserved for the background-solver executor; the inline one is the only one implemented.
        self.use_solver_thread=False

    # ---------------------------------------------------------------- forwarded state

    @property
    def problem(self)->Problem:
        return self.controller.problem

    @property
    def branches(self)->list[BifurcationGUISolutionBranch]:
        return self.controller.branches

    @property
    def current_branch(self): return self.controller.current_branch
    @property
    def current_point(self): return self.controller.current_point
    @property
    def selected_branch(self): return self.controller.selected_branch
    @property
    def selected_point(self): return self.controller.selected_point

    def _fwd(name:str,doc:str=""):  #type:ignore[misc] # noqa: N805 - a property factory, not a method
        def getter(self): return getattr(self.controller,name)
        def setter(self,value): setattr(self.controller,name,value)
        return property(getter,setter,doc=doc)

    neigen=_fwd("neigen","Number of eigenvalues computed at every solution point.")
    shift=_fwd("shift","Shift handed to the eigensolver.")
    classify_bifurcations=_fwd("classify_bifurcations","Compute the normal form at each located bifurcation, naming it fold/transcritical/pitchfork. On by default; branch switching computes it on demand if it is off.")
    interpolated_splines=_fwd("interpolated_splines","Draw and export spline-interpolated branches instead of the raw polylines.")
    arclength_proportion=_fwd("arclength_proportion","Fraction D of the arclength given to the continued parameter while the scaling is on, i.e. (dparameter/ds)^2 == D after every step. Set it through set_arclength_proportion() to have it reach the problem.")
    output_all_observables=_fwd("output_all_observables","Write all observable values to the output files.")
    data_subdir=_fwd("data_subdir","Subdirectory of the problem's output directory holding states and curves.")
    parameter_range=_fwd("parameter_range","Two values pinning the parameter axis, or an empty list.")
    _out_demo_video=_fwd("_out_demo_video","Save a PNG of every redraw, for making a screencast of the session.")
    _last_ds=_fwd("_last_ds")
    _current_observable=_fwd("_current_observable")
    _avail_observables=_fwd("_avail_observables")
    _mode=_fwd("_mode")
    _state_step=_fwd("_state_step")
    del _fwd

    # ---------------------------------------------------------------- forwarded commands

    def set_initial_view(self,xmin,xmax,ymin,ymax):
        """Window shown when a diagram is started from scratch."""
        self.controller.set_initial_view(xmin,xmax,ymin,ymax)

    def get_bifurcation_parameter(self)->"_pyoomph.GiNaC_GlobalParam":
        return self.controller.get_bifurcation_parameter()

    @property
    def continuation_parameter(self)->str:
        """Name of the parameter being continued."""
        return self.controller._get_paramname_str()

    def all_parameter_names(self)->list[str]:
        """Every global parameter of the problem (only complete after :py:meth:`start`)."""
        return self.controller.all_parameter_names()

    def set_continuation_parameter(self,name:str):
        """Continue in a different parameter from here on, starting a new diagram.

        The existing branches are sections at a fixed value of the parameter now being varied, so
        they are kept as they are and a new branch is opened at the current solution.
        """
        return self.controller.set_continuation_parameter(name)

    def set_fixed_parameter(self,name:str,value:float):
        """Move a parameter that is being held fixed, which begins a new slice."""
        self.controller.set_fixed_parameter(name,value)

    def set_arclength_scaling(self,scale:bool):
        """Retune theta^2 after every step so the parameter keeps a fixed share of the arclength."""
        self.controller.set_arclength_scaling(scale)

    def set_arclength_proportion(self,proportion:float):
        """That share: ``(dparameter/ds)^2 == proportion`` after every step. Strictly between 0 and 1."""
        self.controller.set_arclength_proportion(proportion)

    def start_locus(self,tracked:str,continue_in:str,bifurcation_type:str | None=None):
        """From a located bifurcation, follow it through parameter space.

        ``tracked`` is adjusted to keep the bifurcation condition satisfied while ``continue_in`` is
        stepped, tracing the locus in the (``continue_in``, ``tracked``) plane - the interactive
        equivalent of the ``Bo_c(V)`` and ``gamma_c(delta)`` curves the tutorials build by hand.
        """
        self.controller.start_locus(tracked,continue_in,bifurcation_type)

    def leave_locus(self,continue_in:str | None=None,offset:float | None=None):
        """Step off a bifurcation locus onto an ordinary branch through it."""
        self.controller.leave_locus(continue_in,offset)

    def describe_slice(self)->str:
        """The parameters the current diagram holds fixed, e.g. ``"b = 0.3"``.

        A diagram is a section through parameter space and is only valid at these values.
        """
        return self.controller.describe_current_slice()

    def evaluate_observables(self)->dict[str,float]:
        return self.controller.evaluate_observables()

    def new_branch_from_state(self,statefile):
        self.controller.new_branch_from_state(statefile)

    def load_pt(self,pt):
        self.controller.load_pt(pt)

    def output_curves(self):
        return self.controller.output_curves()

    def toggle_point_tag(self,pt,tag):
        self.controller.toggle_point_tag(pt,tag)

    def step(self,ds=None):
        return self.controller.step(ds)

    def multistep(self):
        self.controller.multistep()

    def locate_bifurcation(self,pitchfork:bool=False):
        self.controller.locate_bifurcation(pitchfork)

    def branch_switch(self):
        self.controller.branch_switch()

    def transient_leave_branch(self,eigenindex=0):
        self.controller.transient_leave_branch(eigenindex)

    def save_all(self):
        self.controller.save_all()

    def load_all(self):
        self.controller.load_all(apply_view=self.plotter.apply_saved_view)

    def update_plot(self,infotext:str | None=None):
        """Redraw. Kept for scripts and custom key functions that called it directly."""
        if self.app is not None:
            self.app.refresh(infotext)
        else:
            self.plotter.draw(self.controller,infotext)

    # ---------------------------------------------------------------- lifecycle

    def start(self,init_ds:float | None=None,initial_max_newton_iterations:int=10):
        """Solve the starting point, open the window and enter the event loop.

        ``init_ds`` is the initial arclength step. Returns when the window is closed. An existing
        diagram in the problem's output directory is reloaded, so a session can be resumed - the same
        diagram whose presence :py:meth:`must_init` reports, and the call belongs *after* the block
        that method guards::

            if gui.must_init():
                problem.solve()
            gui.start(0.001)
        """
        if init_ds is None:
            raise RuntimeError("start() needs the initial arclength step size, e.g. gui.start(0.001)")
        fresh=self.controller.start(init_ds,initial_max_newton_iterations)
        if fresh:
            self.plotter.initialise_view(self.controller)

        app=self._create_app()
        self.app=app

        if not fresh:
            try:
                self.controller.load_all(apply_view=self.plotter.apply_saved_view)
            except Exception as e:
                app.log("Could not reload the stored diagram: "+repr(e))
                # Nothing is loaded and nothing was solved, so there is no point to draw yet. Solve
                # where the script left off, so the window opens on something rather than empty.
                self.controller.start(init_ds,initial_max_newton_iterations,ignore_saved=True)
                self.plotter.initialise_view(self.controller)

        app.run()

    def must_init(self,init_ds:float | None=None)->bool:
        """Prepare the problem for a diagram and say whether it still has to be built by hand.

        Guards the code that walks the problem to its first solution, with :py:meth:`start` after it::

            gui=BifurcationGUI(problem,problem.param_gamma)
            if gui.must_init():
                problem.param_gamma.value=0.24
                problem.set_initial_condition(ic_name="hexdots")
                problem.solve(timestep=10)
                problem.solve()
                problem.go_to_param(gamma=0.28,startstep=0.01)
            gui.start(0.001)

        False means a diagram was found in the output directory, so the block is skipped and
        :py:meth:`start` loads that diagram instead - which on a real problem saves the whole
        transient-plus-go_to_param walk. True means there is nothing to load and the block has to
        produce a starting solution.

        The other half of the job is *when* this runs. It initialises the problem here, after the
        constructor has set the runmode to "overwrite" - initialising under the default "delete"
        runmode removes the stored diagram, which is what made a resumable session impossible; see
        :py:meth:`BifurcationController.prepare`.

        ``init_ds`` exists only to catch the earlier signature, where this method took the step size
        and opened the window by itself once the enclosing ``with problem`` block (or the interpreter)
        ended. That hid every failure inside :py:meth:`start`: an exception raised from an atexit
        handler is printed as "Exception ignored" and leaves the process exiting with status 0.
        """
        if init_ds is not None:
            raise RuntimeError(
                "must_init() no longer takes the initial step size and no longer opens the window by "
                "itself - call start() after the guarded block instead:\n"
                "    gui=BifurcationGUI(problem,\"param\")\n"
                "    if gui.must_init():\n"
                "        ...\n"
                "    gui.start({:g})".format(init_ds))
        self.controller.prepare()
        fresh=not self.controller.has_saved_state()
        if not fresh:
            self.controller.log("Found a stored diagram and will load it, so the initialisation is skipped")
        return fresh

    def _create_app(self):
        try:
            from .tkapp import BifurcationTkApp
        except ImportError as e:
            raise RuntimeError(
                "The bifurcation GUI needs tkinter, which is not available in this Python "
                "installation.\nOn Debian/Ubuntu install it with 'sudo apt install python3-tk', on "
                "Fedora with 'sudo dnf install python3-tkinter'.\nWindows and macOS ship it with "
                "the official Python installers.\nOriginal error: "+str(e)) from e
        title=self.title
        if title is None:
            title="pyoomph bifurcation diagram - "+self.controller._get_paramname_str()
        return BifurcationTkApp(self.controller,self.plotter,facade=self,title=title)


from ...typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
