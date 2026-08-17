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

"""Live field plots beside the bifurcation diagram.

The problem's own :py:class:`~pyoomph.output.plotting.MatplotlibPlotter` objects - the ones that
would otherwise only write PNGs - are rendered into embedded canvases, so the solution at the current
point is on screen next to the diagram.

Each plotter gets a figure of its **own**. That is not a layout preference: a plotter assumes it owns
the whole figure (it sets the margins to fill it, switches the axes off, positions colorbars in
absolute figure coordinates and resizes the figure to enforce its aspect ratio), so plotters sharing
one figure through ``add_subplot`` would fight each other. Separate figures also let each plotter keep
the aspect ratio its ``set_view`` asked for.

The figures here are created **through pyplot**, unlike the diagram's. A plotter drives everything
through ``plt.gcf()``/``plt.gca()``, and only a pyplot-managed figure can be made current
(``plt.figure(bare_figure)`` raises "The passed figure is not managed by pyplot"). Attaching a
``FigureCanvasTkAgg`` afterwards replaces the canvas while the figure stays managed, so
``plt.figure(number)`` then aims every pyplot-global call at the embedded figure. The selected
matplotlib backend is irrelevant because the canvas is supplied here.
"""

import copy
import time
import traceback

import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk #type:ignore[attr-defined]

from ...typings import *


def bind_plotter(plotter,problem):
    """Attach a plotter to its problem if the problem has not done so itself yet.

    A plotter constructed as ``MyPlotter()`` has no problem until something binds it;
    :py:meth:`~pyoomph.generic.problem.Problem.output` does it lazily on the first output, and
    ``get_problem()`` asserts otherwise. Plots here are triggered without going through ``output()``,
    so the same binding has to happen.
    """
    if getattr(plotter,"_problem",None) is None:
        plotter._problem=problem
        plotter._named_problems[""]=problem
    return plotter


def derive_eigenfunction_plotter(source,eigenvector:int=0,eigenmode:str="real"):
    """A copy of ``source`` that draws an eigenfunction instead of the solution.

    An eigenfunction plot is the same plot of the same fields with ``eigenvector``/``eigenmode`` set,
    so the plot definition is reused rather than written a second time.

    Copied rather than re-instantiated: a user's plotter subclass may define its own ``__init__``
    signature, which ``type(source)(...)`` would have to guess. The per-plot state has to be reset
    afterwards - in particular ``_range_objects``, which a shallow copy would otherwise share, tying
    this pane's colour scale to the source's.
    """
    clone=copy.copy(source)
    clone.eigenvector=eigenvector
    clone.eigenmode=eigenmode
    clone._added_parts=[]
    clone._range_objects={}
    clone._initialised=False
    clone._output_step=0
    clone.file_trunk=None      # never write files from a derived plotter
    return clone


class PlotterPane:
    """One embedded field plot: a title, a matplotlib canvas and the plotter that fills it."""

    def __init__(self,master,plotter,title:str,log:Callable[[str],None] | None=None) -> None:
        self.plotter=plotter
        self.title=title
        self._log=log
        self.frame=ttk.Frame(master)
        header=ttk.Frame(self.frame)
        self._header=header
        header.pack(side=tk.TOP,fill=tk.X)
        ttk.Label(header,text=title,padding=(4,2)).pack(side=tk.LEFT)
        self.status_var=tk.StringVar(value="")
        ttk.Label(header,textvariable=self.status_var,padding=(4,2),foreground="#804000").pack(side=tk.RIGHT)

        # Created through pyplot so it can be made current later - see the module docstring.
        self.figure=plt.figure(figsize=(4.0,3.0))
        self._number=self.figure.number
        self.canvas=FigureCanvasTkAgg(self.figure,master=self.frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP,fill=tk.BOTH,expand=True)
        self.nav=NavigationToolbar2Tk(self.canvas,self.frame,pack_toolbar=False)
        self.nav.update()
        self.nav.pack(side=tk.BOTTOM,fill=tk.X)
        self._drawn_once=False

        #: The rectangle the user zoomed/panned to, imposed on every later render. None means "use
        #: whatever the plot definition asks for".
        self._locked_view:tuple[float,float,float,float] | None=None
        self.canvas.mpl_connect("button_release_event",self._on_release)
        # Home means "back to the plot's own view", so it also drops the lock.
        self._nav_home=self.nav.home
        self.nav.home=self._home   #type:ignore[method-assign]
        self._wire_toolbar_buttons()

    def _wire_toolbar_buttons(self):
        """Point the toolbar's Home/Back/Forward at our own handlers.

        Reassigning ``nav.home`` alone is not enough: matplotlib binds each button's command at
        construction time, so the Home BUTTON keeps calling the original method. It would then reset
        the axes while leaving the view lock in place, and the next replot would silently snap back to
        the locked rectangle. Verified against matplotlib 3.10.
        """
        buttons=getattr(self.nav,"_buttons",None)
        wired=False
        if isinstance(buttons,dict):
            home=buttons.get("Home")
            if home is not None:
                home.configure(command=self._home)
                wired=True
            for name in ("Back","Forward"):
                btn=buttons.get(name)
                original=getattr(self.nav,name.lower(),None)
                if btn is not None and original is not None:
                    btn.configure(command=lambda o=original: self._nav_history(o))
        if not wired:
            # Private API that has moved before; without it there must still be a way to unlock.
            ttk.Button(self._header,text="Reset view",command=self._home).pack(side=tk.RIGHT,padx=2)

    def _nav_history(self,original):
        """Back/Forward step between the user's own views, so the new one becomes the locked one."""
        original()
        ax=self._main_axes()
        if ax is not None:
            xl,yl=ax.get_xlim(),ax.get_ylim()
            self._locked_view=(float(xl[0]),float(xl[1]),float(yl[0]),float(yl[1]))
            self.status_var.set("view locked")
            self.refresh()

    # ---------------------------------------------------------------- view locking

    def _main_axes(self):
        """The field axes: created first by the plotter, before any colorbar is added."""
        return self.figure.axes[0] if self.figure.axes else None

    def _on_release(self,_event):
        # nav.mode is truthy only while zoom-rect or pan is armed, so this fires for a deliberate
        # view change and not for an ordinary click.
        if not self.nav.mode:
            return
        ax=self._main_axes()
        if ax is None:
            return
        xl,yl=ax.get_xlim(),ax.get_ylim()
        self._locked_view=(float(xl[0]),float(xl[1]),float(yl[0]),float(yl[1]))
        self.status_var.set("view locked")

    def _impose_locked_view(self)->Callable[[],None]:
        """Make the plot definition's set_view() call return the user's rectangle instead.

        Interception rather than pre-setting ``plotter.xmin..ymax``, because ``define_plot`` normally
        calls ``set_view`` itself on every render and would overwrite anything set beforehand.

        Doing it this way - rather than fixing up the axes limits afterwards - is what makes the colour
        scale follow the zoom: the range of a colorbar is taken from the data *inside* the plotter's
        view (``get_visible_data_range``), so imposing the view early means the range, the aspect
        handling and the drawn field are all computed for the region actually on screen.
        """
        if self._locked_view is None:
            return lambda: None
        xmin,xmax,ymin,ymax=self._locked_view
        original=self.plotter.set_view

        def forced(*_args,**_kwargs):
            return original(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)

        self.plotter.set_view=forced   #type:ignore[method-assign]

        def restore():
            try:
                del self.plotter.set_view
            except AttributeError:
                self.plotter.set_view=original   #type:ignore[method-assign]
        return restore

    def _home(self,*args,**kwargs):
        """Toolbar Home: release the lock and re-render with the plot's own view.

        Re-rendering rather than only resetting the axes, because the colour range is computed from
        the data inside the view - going back to the full view has to recompute it.
        """
        self._locked_view=None
        self.status_var.set("")
        self._nav_home(*args,**kwargs)
        self.refresh()

    def destroy(self):
        try:
            plt.close(self._number)
        except Exception:
            pass
        self.frame.destroy()

    def _message(self,text:str):
        """Say why there is nothing to see, rather than leaving blank axes."""
        self.figure.clf()
        ax=self.figure.add_subplot(1,1,1)
        ax.set_axis_off()
        ax.text(0.5,0.5,text,ha="center",va="center",wrap=True,fontsize="small",color="0.35")
        self.canvas.draw()

    def refresh(self)->bool:
        """Re-render the plot from the problem's current state. Never raises."""
        self.status_var.set("")
        started=time.time()
        try:
            problem=self.plotter.get_problem()
        except Exception:
            problem=None
        if problem is None:
            self._message("No problem attached to this plotter")
            return False
        # An eigenfunction pane has nothing to show until the eigenvectors exist; the plotter itself
        # would just return without drawing, leaving whatever was on the canvas before.
        if self.plotter.eigenvector is not None:
            evs=problem.get_last_eigenvectors()
            if evs is None or len(evs)<=self.plotter.eigenvector:
                self._message("Eigenvector {:d} has not been computed at this point".format(
                    self.plotter.eigenvector))
                self.status_var.set("no eigendata")
                return False

        previous=plt.get_fignums()
        restore_view=self._impose_locked_view()
        try:
            self.figure.clf()
            plt.figure(self._number)           # aim the plotter's plt.gcf()/plt.gca() at this figure
            self.plotter.plot_into_current_figure()
            if self._locked_view is not None:
                # Also applied directly, for a plot definition that never calls set_view at all and
                # would otherwise autoscale back to the full data on each render.
                ax=self._main_axes()
                if ax is not None:
                    ax.set_xlim(self._locked_view[0],self._locked_view[1])
                    ax.set_ylim(self._locked_view[2],self._locked_view[3])
            if getattr(self.plotter,"_has_invalid_triangulation",False):
                self._message("The mesh could not be triangulated for plotting")
                self.status_var.set("invalid triangulation")
                return False
            self.canvas.draw()
            self._drawn_once=True
            if self._log is not None:
                self._log("  plotted {:s} in {:.2f} s".format(self.title,time.time()-started))
            return True
        except Exception as e:
            # A broken plot definition must not take the window down with it.
            self._message("Plotting failed:\n"+str(e))
            self.status_var.set("failed")
            if self._log is not None:
                self._log("*** plotting "+self.title+" failed: "+repr(e))
                for line in traceback.format_exc().splitlines():
                    self._log("    "+line)
            return False
        finally:
            restore_view()
            # A plot definition may create figures of its own; anything it left behind would leak.
            for num in plt.get_fignums():
                if num not in previous and num!=self._number:
                    plt.close(num)


class PlotterPaneSet:
    """The column of field plots: the problem's plotters plus derived eigenfunction panes."""

    def __init__(self,master,controller,log:Callable[[str],None] | None=None) -> None:
        self.controller=controller
        self._log=log
        self.paned=ttk.PanedWindow(master,orient=tk.VERTICAL)
        self.panes:list[PlotterPane]=[]
        #: (source index, eigenvector, eigenmode) of each derived pane currently shown.
        self.eigen_panes:set[tuple[int,int,str]]=set()
        self._build()

    # ---------------------------------------------------------------- discovery

    def source_plotters(self)->list:
        """The problem's plotters. ``Problem.plotter`` may be one, a list, or None."""
        plotters=getattr(self.controller.problem,"plotter",None)
        if plotters is None:
            return []
        if not isinstance(plotters,(list,tuple)):
            plotters=[plotters]
        return [p for p in plotters if p is not None and getattr(p,"active",True)]

    def has_any(self)->bool:
        return len(self.panes)>0

    def _plotter_title(self,plotter,index:int)->str:
        name=type(plotter).__name__
        if plotter.eigenvector is not None:
            return "{:s} - eigenvector {:d} ({:s})".format(name,plotter.eigenvector,plotter.eigenmode)
        return name if len(self.source_plotters())<2 else "{:s} #{:d}".format(name,index)

    def _build(self):
        for p in self.panes:
            p.destroy()
        self.panes=[]
        sources=self.source_plotters()
        for i,src in enumerate(sources):
            # The problem's own plotter object is used as it is, so what is on screen is exactly what
            # the script defined - including its file output when the problem writes one.
            self._add_pane(src,self._plotter_title(src,i))
        for (si,ev,mode) in sorted(self.eigen_panes):
            if si<len(sources):
                clone=derive_eigenfunction_plotter(sources[si],ev,mode)
                self._add_pane(clone,"{:s} - eigen {:d} ({:s})".format(
                    type(sources[si]).__name__,ev,mode))

    def _add_pane(self,plotter,title:str):
        bind_plotter(plotter,self.controller.problem)
        pane=PlotterPane(self.paned,plotter,title,log=self._log)
        self.paned.add(pane.frame,weight=1)
        self.panes.append(pane)

    # ---------------------------------------------------------------- eigenfunction panes

    def eigen_pane_shown(self,source_index:int,eigenvector:int,eigenmode:str)->bool:
        return (source_index,eigenvector,eigenmode) in self.eigen_panes

    def toggle_eigen_pane(self,source_index:int,eigenvector:int,eigenmode:str):
        key=(source_index,eigenvector,eigenmode)
        if key in self.eigen_panes:
            self.eigen_panes.discard(key)
        else:
            self.eigen_panes.add(key)
        self._build()
        self.refresh()

    # ---------------------------------------------------------------- rendering

    def refresh(self):
        for pane in self.panes:
            pane.refresh()


from ...typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
