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

"""The tkinter/ttk user interface of the bifurcation GUI.

tkinter is used because it ships with CPython on Windows and macOS (and is a small ``python3-tk``
package on Linux), so the tool stays usable everywhere without adding a GUI toolkit to pyoomph's
dependencies. ``tk.Menu`` becomes the system menu bar on macOS and ttk widgets are natively themed
on Windows and macOS.

The diagram's matplotlib figure is embedded with ``FigureCanvasTkAgg`` and is created through the
object-oriented API, so no backend selection happens anywhere. The problem's own
:py:class:`~pyoomph.output.plotting.MatplotlibPlotter` objects are shown live beside it - see
:py:mod:`~pyoomph.utils.bifurcation_gui.panes`, which explains why those need pyplot-created figures
while this one does not.

Long solves run in the calling thread and the event loop is pumped where the old version called
``update_plot()``. The window is therefore frozen within a single Newton solve - exactly as before -
but the status label updates and the Abort button works between continuation steps. Everything the
user can trigger goes through :py:meth:`BifurcationTkApp._invoke`, which refuses to start a second
task while one is running; that guard is what makes pumping the event loop mid-solve safe.
"""

import os
import traceback

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk #type:ignore[attr-defined]

from .actions import Action, KeyMap, event_to_accelerator, format_accelerator
from .model import AXIS_PARAMETER
from .panes import PlotterPaneSet

from ...typings import *


#: Widget classes that swallow key presses, so accelerators must not fire while they have focus.
_TEXT_INPUT_CLASSES={"Entry","TEntry","Text","TSpinbox","Spinbox","TCombobox","Listbox"}


class BifurcationTkApp:
    """The main window: menu bar, toolbar, embedded plot, side panels, log and status bar."""

    def __init__(self,controller,plotter,facade=None,title:str | None=None) -> None:
        self.controller=controller
        self.plotter=plotter
        self.facade=facade if facade is not None else controller
        self.keymap=KeyMap()

        self._busy=False
        self._suspend_tree_callback=False
        self._tree_signature=None
        self._actions:dict[str,Action]={}
        self._menu_entries:dict[str,list[tuple[tk.Menu,int]]]={}
        self._toolbar_buttons:dict[str,ttk.Button]={}
        self._check_vars:dict[str,tk.BooleanVar]={}
        self._tree_items:dict[str,tuple[Any,Any]]={}
        self._tree_index:dict[int,str]={}
        self._branch_index:dict[int,str]={}
        self._eigen_vars:dict[tuple[int,int,str],Any]={}

        self.root=tk.Tk()
        self.root.title(title if title is not None else "pyoomph bifurcation diagram")
        self.root.geometry("1250x820")
        self._observable_var=tk.StringVar()
        self._xaxis_var=tk.StringVar()
        #: Display string -> axis spec, so the combo boxes and the View submenus never have to parse
        #: a label back into a (kind, name) pair.
        self._axis_choices:dict[str,Any]={}
        self._param_var=tk.StringVar()
        self._graphs_pane=None
        self._plot_panes_attached=False
        #: Re-plot the fields when the current point has moved. A mesh plot is not free, so a
        #: multistep sweep marks this once and re-plots when it finishes rather than per step.
        self._plots_dirty=True
        self.auto_update_plots=True

        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        # The status bar has to be packed before the body: pack hands out space in call order, and
        # the body expands, so a status bar added afterwards would get zero height.
        self._build_statusbar()
        self._build_body()

        self.root.bind_all("<Key>",self._on_key)
        self.root.protocol("WM_DELETE_WINDOW",self._on_close)

        self.controller.set_observer(on_changed=self.refresh,on_status=self._on_status,
                                     on_log=self.log,on_busy=self._on_busy)
        self.plotter.on_drawn=self._blit

    # ================================================================== actions

    def _add_action(self,action_id:str,label:str,callback,**kwargs)->Action:
        act=Action(id=action_id,label=label,callback=callback,**kwargs)
        self._actions[action_id]=act
        return act

    def _build_actions(self):
        c=self.controller
        A=self._add_action

        def at_bifurcation()->bool:
            return c.current_point is not None and c.current_point.eig_value_Re==0

        # --- continuation
        A("step","Step",c.step,toolbar="Step",is_solver_task=True,
          tooltip="One arclength continuation step")
        A("multistep","Multistep",c.multistep,toolbar="Multi",is_solver_task=True,
          tooltip="Keep stepping until the branch leaves the visible axes (Abort to stop)")
        A("step_shrink","Step (never grow ds)",c.step_and_shrink,is_solver_task=True)
        A("ds_increase","Increase ds",lambda: self._scale_ds(1.25),toolbar="ds +")
        A("ds_decrease","Decrease ds",lambda: self._scale_ds(1/1.25),toolbar="ds -")
        A("ds_reverse","Reverse direction",lambda: self._scale_ds(-1),toolbar="Reverse")
        A("set_ds","Set ds...",self._dialog_set_ds)
        A("continue_in_selected","Continue in the selected parameter",self._continue_in_selected_parameter,
          is_solver_task=True,enabled_when=lambda: self._selected_parameter() not in (None,c._paramname),
          tooltip="Start a new diagram continuing in the parameter selected in the Parameters tab")
        A("set_selected_parameter","Set the selected parameter's value...",self._dialog_set_parameter,
          is_solver_task=True,enabled_when=lambda: self._selected_parameter() is not None)
        A("refresh_plots","Refresh field plots",self._refresh_plots,
          enabled_when=lambda: self.plot_panes.has_any(),
          tooltip="Re-render the problem's plotters from the current solution")
        A("auto_update_plots","Update field plots automatically",self._toggle_auto_plots,kind="check",
          getter=lambda: self.auto_update_plots)
        A("show_other_slices","Show branches from other slices",self._toggle_other_slices,kind="check",
          getter=lambda: self.plotter.show_other_slices)
        A("arclength_scaling_on","Scale arclength",lambda: c.set_arclength_scaling(True))
        A("arclength_scaling_off","Do not scale arclength",lambda: c.set_arclength_scaling(False))

        # --- bifurcations
        A("locate_bifurcation","Locate bifurcation / switch branch",c.locate_bifurcation_or_switch,
          toolbar="Find bif",is_solver_task=True,
          tooltip="At a bifurcation this switches branch, otherwise it tracks one down")
        A("locate_pitchfork","Locate pitchfork",lambda: c.locate_bifurcation(pitchfork=True),is_solver_task=True)
        A("branch_switch","Switch branch",c.branch_switch,toolbar="Switch",is_solver_task=True,
          enabled_when=at_bifurcation,tooltip="Only available at a classified bifurcation")
        A("transient_leave_0","Leave branch transiently (mode 0)",lambda: c.transient_leave_branch(0),is_solver_task=True)
        A("transient_leave_1","Leave branch transiently (mode 1)",lambda: c.transient_leave_branch(1),is_solver_task=True)
        A("start_locus","Follow this bifurcation in...",self._dialog_start_locus,is_solver_task=True,
          enabled_when=lambda: at_bifurcation() and not c.on_locus(),
          tooltip="Adjust the current parameter to hold the bifurcation while continuing in another, "
                  "tracing its locus in the plane of the two")
        A("leave_locus","Leave the locus and continue in...",self._dialog_leave_locus,is_solver_task=True,
          enabled_when=lambda: c.on_locus(),
          tooltip="Step off the bifurcation onto an ordinary branch through it")
        A("eigen_settings","Eigenvalue settings...",self._dialog_eigen)

        # --- points
        A("goto_selected","Go to selected point",c.goto_selected_point,is_solver_task=True,
          enabled_when=lambda: c.selected_point is not None and c.selected_point is not c.current_point)
        A("select_prev","Select previous",lambda: c.select_relative("prev"))
        A("select_next","Select next",lambda: c.select_relative("next"))
        A("select_first","Select first",lambda: c.select_relative("first"))
        A("select_last","Select last",lambda: c.select_relative("last"))
        A("delete_point","Delete point",self._delete_point,toolbar="Delete")
        A("delete_point_alt","Delete point",self._delete_point)
        for d in range(10):
            A("tag_{:d}".format(d),"Tag {:d}".format(d),lambda d=d: c.tag_selected_point(d))

        # --- view
        A("toggle_splines","Interpolated splines",self._toggle_splines,kind="check",
          getter=lambda: c.interpolated_splines)
        A("toggle_mode","Move-point mode",self._toggle_mode,kind="check",
          getter=lambda: c.mode=="mp")
        A("cycle_observable","Next observable",self._cycle_observable)
        A("toggle_logx","Logarithmic parameter axis",lambda: self._toggle_scale("x"),kind="check",
          getter=lambda: self.plotter.get_xscale()=="log")
        A("toggle_logy","Logarithmic observable axis",lambda: self._toggle_scale("y"),kind="check",
          getter=lambda: self.plotter.get_yscale()=="log")
        A("autoscale","Autoscale to data",lambda: self.plotter.autoscale_to_data(c))
        A("reset_view","Reset view around current point",self._reset_view)
        A("parameter_range","Fix parameter range...",self._dialog_parameter_range)

        # --- file
        A("save_diagram","Save diagram",self._save_diagram)
        A("reload_diagram","Reload diagram from disk",self._reload_diagram,is_solver_task=True)
        A("export_curves","Export curves...",c.output_curves)
        A("new_branch_from_state","Start new branch from state file...",self._new_branch_from_state,is_solver_task=True)
        A("save_figure","Save figure as...",self._save_figure)
        A("toggle_demo_video","Record a frame per redraw",self._toggle_demo_video,kind="check",
          getter=lambda: c._out_demo_video)
        A("classify_bifurcations","Classify bifurcations (normal form)",self._toggle_classify,kind="check",
          getter=lambda: c.classify_bifurcations)
        A("quit","Quit",self._on_close)

        # --- misc
        A("abort","Abort running sweep",c.request_abort)
        A("keymap_dialog","Keyboard shortcuts...",self._dialog_keymap)
        A("help","Shortcut reference",self._dialog_help)

        # Custom commands installed by the user script keep working and become visible.
        for key,func in getattr(self.facade,"custom_key_functions",{}).items():
            aid="custom:"+key
            A(aid,"Custom '{:s}'".format(key),lambda func=func: func(self.facade),is_solver_task=True)
            self.keymap.set(aid,key)

    # ================================================================== menus

    def _accel(self,action_id:str)->str:
        return format_accelerator(self.keymap.get(action_id))

    def _add_menu_item(self,menu:tk.Menu,action_id:str):
        """Add one registered action to a menu, wiring label, accelerator and enabling."""
        act=self._actions[action_id]
        if act.kind=="check":
            var=self._check_vars.get(action_id)
            if var is None:
                var=tk.BooleanVar(value=bool(act.getter()) if act.getter else False)
                self._check_vars[action_id]=var
            menu.add_checkbutton(label=act.label,accelerator=self._accel(action_id),
                                 variable=var,command=lambda a=act: self._invoke(a))
        else:
            menu.add_command(label=act.label,accelerator=self._accel(action_id),
                             command=lambda a=act: self._invoke(a))
        self._menu_entries.setdefault(action_id,[]).append((menu,int(menu.index("end") or 0)))

    def _rebuild_menu_accelerators(self):
        """Refresh the accelerator column after the keymap was edited."""
        for action_id,entries in self._menu_entries.items():
            acc=self._accel(action_id)
            for menu,index in entries:
                try:
                    menu.entryconfigure(index,accelerator=acc)
                except tk.TclError:
                    pass

    def _build_menus(self):
        menubar=tk.Menu(self.root)
        self.root.config(menu=menubar)

        m=tk.Menu(menubar,tearoff=0)
        menubar.add_cascade(label="File",menu=m)
        self._add_menu_item(m,"save_diagram")
        self._add_menu_item(m,"reload_diagram")
        m.add_separator()
        self._add_menu_item(m,"new_branch_from_state")
        m.add_separator()
        self._add_menu_item(m,"export_curves")
        self._add_menu_item(m,"save_figure")
        self._add_menu_item(m,"toggle_demo_video")
        m.add_separator()
        self._add_menu_item(m,"quit")

        m=tk.Menu(menubar,tearoff=0)
        menubar.add_cascade(label="Continuation",menu=m)
        self.param_menu=tk.Menu(m,tearoff=0)
        m.add_cascade(label="Continuation parameter",menu=self.param_menu)
        m.add_separator()
        self._add_menu_item(m,"step")
        self._add_menu_item(m,"multistep")
        self._add_menu_item(m,"step_shrink")
        m.add_separator()
        self._add_menu_item(m,"ds_increase")
        self._add_menu_item(m,"ds_decrease")
        self._add_menu_item(m,"ds_reverse")
        self._add_menu_item(m,"set_ds")
        m.add_separator()
        self._add_menu_item(m,"continue_in_selected")
        self._add_menu_item(m,"set_selected_parameter")
        m.add_separator()
        self._add_menu_item(m,"arclength_scaling_on")
        self._add_menu_item(m,"arclength_scaling_off")

        m=tk.Menu(menubar,tearoff=0)
        menubar.add_cascade(label="Bifurcation",menu=m)
        self._add_menu_item(m,"locate_bifurcation")
        self._add_menu_item(m,"locate_pitchfork")
        self._add_menu_item(m,"branch_switch")
        m.add_separator()
        m.add_separator()
        self._add_menu_item(m,"start_locus")
        self._add_menu_item(m,"leave_locus")
        m.add_separator()
        self._add_menu_item(m,"transient_leave_0")
        self._add_menu_item(m,"transient_leave_1")
        m.add_separator()
        self._add_menu_item(m,"classify_bifurcations")
        self._add_menu_item(m,"eigen_settings")

        m=tk.Menu(menubar,tearoff=0)
        menubar.add_cascade(label="Points",menu=m)
        self._add_menu_item(m,"goto_selected")
        m.add_separator()
        self._add_menu_item(m,"select_prev")
        self._add_menu_item(m,"select_next")
        self._add_menu_item(m,"select_first")
        self._add_menu_item(m,"select_last")
        m.add_separator()
        self._add_menu_item(m,"delete_point")
        tagmenu=tk.Menu(m,tearoff=0)
        m.add_cascade(label="Tag",menu=tagmenu)
        for d in range(10):
            self._add_menu_item(tagmenu,"tag_{:d}".format(d))
        m.add_separator()
        self._add_menu_item(m,"toggle_mode")

        m=tk.Menu(menubar,tearoff=0)
        menubar.add_cascade(label="View",menu=m)
        self.xaxis_menu=tk.Menu(m,tearoff=0)
        m.add_cascade(label="X axis",menu=self.xaxis_menu)
        self.observable_menu=tk.Menu(m,tearoff=0)
        m.add_cascade(label="Y axis",menu=self.observable_menu)
        self._add_menu_item(m,"cycle_observable")
        m.add_separator()
        self._add_menu_item(m,"toggle_splines")
        self._add_menu_item(m,"show_other_slices")
        self._add_menu_item(m,"toggle_logx")
        self._add_menu_item(m,"toggle_logy")
        m.add_separator()
        m.add_separator()
        self.fieldplot_menu=tk.Menu(m,tearoff=0)
        m.add_cascade(label="Field plots",menu=self.fieldplot_menu)
        m.add_separator()
        self._add_menu_item(m,"autoscale")
        self._add_menu_item(m,"reset_view")
        self._add_menu_item(m,"parameter_range")

        custom=[a for a in self._actions if a.startswith("custom:")]
        if custom:
            m=tk.Menu(menubar,tearoff=0)
            menubar.add_cascade(label="Custom",menu=m)
            for aid in custom:
                self._add_menu_item(m,aid)

        m=tk.Menu(menubar,tearoff=0)
        menubar.add_cascade(label="Settings",menu=m)
        self._add_menu_item(m,"keymap_dialog")

        m=tk.Menu(menubar,tearoff=0)
        menubar.add_cascade(label="Help",menu=m)
        self._add_menu_item(m,"help")

    @staticmethod
    def _axis_display(spec)->str:
        """Unambiguous label for a menu or combo: a parameter and an observable may share a name."""
        return "{:s}  [{:s}]".format(spec[1],"param" if spec[0]==AXIS_PARAMETER else "obs")

    def _rebuild_axis_menus(self):
        """Rebuild the X/Y axis radio submenus and the choice table the combos share."""
        specs=self.controller.available_axes()
        self._axis_choices={self._axis_display(sp):sp for sp in specs}
        displays=list(self._axis_choices.keys())
        for menu,var,which in ((self.xaxis_menu,self._xaxis_var,"x"),
                               (self.observable_menu,self._observable_var,"y")):
            menu.delete(0,"end")
            for d in displays:
                menu.add_radiobutton(label=d,value=d,variable=var,
                                     command=lambda d=d,w=which: self._set_axis(w,d))
        for combo in (self.xaxis_combo,self.obs_combo):
            if list(combo["values"])!=displays:
                combo["values"]=displays
        self._rebuild_parameter_menu()
        self._sync_axis_selectors()

    def _rebuild_parameter_menu(self):
        """Radio list of every global parameter, marking the one being continued."""
        self.param_menu.delete(0,"end")
        self._param_var.set(self.controller._paramname if isinstance(self.controller._paramname,str) else "")
        for name in self.controller.all_parameter_names():
            self.param_menu.add_radiobutton(label=name,value=name,variable=self._param_var,
                                            command=lambda n=name: self._switch_continuation_parameter(n))

    def _switch_continuation_parameter(self,name:str):
        c=self.controller
        if name==c._paramname:
            return
        self.log("> Continue in "+name)
        try:
            c.run_task("Switching to "+name,lambda: c.set_continuation_parameter(name))
        except Exception as e:
            self._report_error("Switching the continuation parameter",e)
        finally:
            self._autosave()
            self.refresh()

    def _sync_axis_selectors(self):
        xd=self._axis_display(self.controller.x_axis)
        yd=self._axis_display(self.controller.y_axis)
        self._xaxis_var.set(xd)
        self._observable_var.set(yd)
        if self.xaxis_combo.get()!=xd:
            self.xaxis_combo.set(xd)
        if self.obs_combo.get()!=yd:
            self.obs_combo.set(yd)

    def _set_axis(self,which:str,display:str):
        """Put a parameter or an observable on one of the axes, rescaling that axis to its range."""
        spec=self._axis_choices.get(display)
        if spec is None:
            return
        c=self.controller
        if which=="x":
            if spec==c.x_axis:
                return
            c.set_x_axis(spec)
        else:
            if spec==c.y_axis:
                return
            c.set_y_axis(spec)
        rng=c.axis_range(spec)
        if rng is not None:
            lo,hi=rng
            pad=0.05*(hi-lo) if hi>lo else max(abs(hi),1e-4)
            if which=="x":
                self.plotter.set_xlim(lo-pad,hi+pad)
            else:
                self.plotter.set_ylim(lo-pad,hi+pad)
        self._autosave()
        self.refresh()

    # ================================================================== widgets

    def _build_toolbar(self):
        bar=ttk.Frame(self.root,padding=(4,3))
        bar.pack(side=tk.TOP,fill=tk.X)
        for aid in ["step","multistep"]:
            self._toolbar_button(bar,aid)
        ttk.Separator(bar,orient=tk.VERTICAL).pack(side=tk.LEFT,fill=tk.Y,padx=6)
        for aid in ["ds_decrease","ds_increase","ds_reverse"]:
            self._toolbar_button(bar,aid)
        ttk.Separator(bar,orient=tk.VERTICAL).pack(side=tk.LEFT,fill=tk.Y,padx=6)
        for aid in ["locate_bifurcation","branch_switch"]:
            self._toolbar_button(bar,aid)
        ttk.Separator(bar,orient=tk.VERTICAL).pack(side=tk.LEFT,fill=tk.Y,padx=6)
        self._toolbar_button(bar,"delete_point")

        self.abort_button=ttk.Button(bar,text="Abort",state="disabled",
                                     command=lambda: self._invoke(self._actions["abort"]))
        self.abort_button.pack(side=tk.RIGHT)

    def _toolbar_button(self,parent,action_id:str):
        act=self._actions[action_id]
        btn=ttk.Button(parent,text=act.toolbar or act.label,width=max(6,len(act.toolbar or act.label)+1),
                       command=lambda a=act: self._invoke(a))
        btn.pack(side=tk.LEFT,padx=2)
        if act.tooltip:
            _Tooltip(btn,act.tooltip+("  ["+self._accel(action_id)+"]" if self._accel(action_id) else ""))
        self._toolbar_buttons[action_id]=btn

    def _build_body(self):
        outer=ttk.PanedWindow(self.root,orient=tk.VERTICAL)
        outer.pack(side=tk.TOP,fill=tk.BOTH,expand=True)

        upper=ttk.PanedWindow(outer,orient=tk.HORIZONTAL)
        outer.add(upper,weight=4)

        # The diagram and the field plots share the left side through a paned window of their own, so
        # every sash is draggable and the side notebook stays where it was.
        graphs=ttk.PanedWindow(upper,orient=tk.HORIZONTAL)
        upper.add(graphs,weight=4)

        plotframe=ttk.Frame(graphs)
        graphs.add(plotframe,weight=3)
        self.canvas=FigureCanvasTkAgg(self.plotter.figure,master=plotframe)
        self.canvas.get_tk_widget().pack(side=tk.TOP,fill=tk.BOTH,expand=True)
        self.nav=NavigationToolbar2Tk(self.canvas,plotframe,pack_toolbar=False)
        self.nav.update()
        self.nav.pack(side=tk.BOTTOM,fill=tk.X)
        self.canvas.mpl_connect("button_press_event",self._on_canvas_click)

        # The problem's own plotters, rendered live. Nothing is added when the problem has none, so a
        # script without a plotter gets exactly the window it got before.
        self._graphs_pane=graphs
        self.plot_panes=PlotterPaneSet(graphs,self.controller,log=self.log)
        self._attach_plot_panes_if_needed()

        self.side=ttk.Notebook(upper,width=330)
        upper.add(self.side,weight=1)
        self._build_continuation_tab()
        self._build_parameters_tab()
        self._build_pointinfo_tab()
        self._build_branches_tab()

        logframe=ttk.Frame(outer)
        outer.add(logframe,weight=1)
        ttk.Label(logframe,text="Log",padding=(4,2)).pack(side=tk.TOP,anchor=tk.W)
        self.logtext=tk.Text(logframe,height=7,wrap=tk.NONE,state=tk.DISABLED)
        scroll=ttk.Scrollbar(logframe,orient=tk.VERTICAL,command=self.logtext.yview)
        self.logtext.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT,fill=tk.Y)
        self.logtext.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)

    def _build_continuation_tab(self):
        tab=ttk.Frame(self.side,padding=8)
        self.side.add(tab,text="Continuation")

        row=0
        ttk.Label(tab,text="Step size ds").grid(row=row,column=0,sticky=tk.W,pady=2)
        self.ds_var=tk.StringVar()
        self.ds_entry=ttk.Entry(tab,textvariable=self.ds_var,width=14)
        self.ds_entry.grid(row=row,column=1,sticky=tk.EW,pady=2)
        self.ds_entry.bind("<Return>",lambda *_: self._commit_ds())
        self.ds_entry.bind("<FocusOut>",lambda *_: self._commit_ds())
        row+=1

        frame=ttk.Frame(tab)
        frame.grid(row=row,column=0,columnspan=2,sticky=tk.EW,pady=(2,8))
        ttk.Button(frame,text="/1.25",width=6,command=lambda: self._invoke(self._actions["ds_decrease"])).pack(side=tk.LEFT)
        ttk.Button(frame,text="x1.25",width=6,command=lambda: self._invoke(self._actions["ds_increase"])).pack(side=tk.LEFT,padx=3)
        ttk.Button(frame,text="reverse",width=8,command=lambda: self._invoke(self._actions["ds_reverse"])).pack(side=tk.LEFT)
        row+=1

        ttk.Separator(tab,orient=tk.HORIZONTAL).grid(row=row,column=0,columnspan=2,sticky=tk.EW,pady=6)
        row+=1

        ttk.Label(tab,text="Eigenvalues").grid(row=row,column=0,sticky=tk.W,pady=2)
        self.neigen_var=tk.StringVar()
        neigen=ttk.Entry(tab,textvariable=self.neigen_var,width=14)
        neigen.grid(row=row,column=1,sticky=tk.EW,pady=2)
        neigen.bind("<Return>",lambda *_: self._commit_eigen())
        neigen.bind("<FocusOut>",lambda *_: self._commit_eigen())
        row+=1

        ttk.Label(tab,text="Shift").grid(row=row,column=0,sticky=tk.W,pady=2)
        self.shift_var=tk.StringVar()
        shift=ttk.Entry(tab,textvariable=self.shift_var,width=14)
        shift.grid(row=row,column=1,sticky=tk.EW,pady=2)
        shift.bind("<Return>",lambda *_: self._commit_eigen())
        shift.bind("<FocusOut>",lambda *_: self._commit_eigen())
        row+=1

        ttk.Separator(tab,orient=tk.HORIZONTAL).grid(row=row,column=0,columnspan=2,sticky=tk.EW,pady=6)
        row+=1

        # Either axis can show any parameter or any observable, which is what lets the same plot be
        # an ordinary diagram or the locus of a bifurcation in a plane of two parameters.
        ttk.Label(tab,text="x axis").grid(row=row,column=0,sticky=tk.W,pady=2)
        self.xaxis_combo=ttk.Combobox(tab,state="readonly",width=16)
        self.xaxis_combo.grid(row=row,column=1,sticky=tk.EW,pady=2)
        self.xaxis_combo.bind("<<ComboboxSelected>>",lambda *_: self._set_axis("x",self.xaxis_combo.get()))
        row+=1

        ttk.Label(tab,text="y axis").grid(row=row,column=0,sticky=tk.W,pady=2)
        self.obs_combo=ttk.Combobox(tab,state="readonly",width=16)
        self.obs_combo.grid(row=row,column=1,sticky=tk.EW,pady=2)
        self.obs_combo.bind("<<ComboboxSelected>>",lambda *_: self._set_axis("y",self.obs_combo.get()))
        row+=1

        self.scale_al_var=tk.BooleanVar(value=True)
        ttk.Checkbutton(tab,text="Scale arclength",variable=self.scale_al_var,
                        command=lambda: self.controller.set_arclength_scaling(self.scale_al_var.get())).grid(
                        row=row,column=0,columnspan=2,sticky=tk.W,pady=(6,2))
        row+=1

        self.splines_var=tk.BooleanVar()
        ttk.Checkbutton(tab,text="Interpolated splines",variable=self.splines_var,
                        command=lambda: self._invoke(self._actions["toggle_splines"])).grid(
                        row=row,column=0,columnspan=2,sticky=tk.W,pady=2)
        row+=1

        ttk.Label(tab,text="Mode").grid(row=row,column=0,sticky=tk.W,pady=(8,2))
        self.mode_var=tk.StringVar(value="al")
        modes=ttk.Frame(tab)
        modes.grid(row=row,column=1,sticky=tk.W,pady=(8,2))
        ttk.Radiobutton(modes,text="Arclength",value="al",variable=self.mode_var,
                        command=self._commit_mode).pack(anchor=tk.W)
        ttk.Radiobutton(modes,text="Move point",value="mp",variable=self.mode_var,
                        command=self._commit_mode).pack(anchor=tk.W)
        row+=1

        self.movepoint_var=tk.BooleanVar()
        self.movepoint_check=ttk.Checkbutton(tab,text="Grab selected point",variable=self.movepoint_var,
                                             command=self._commit_move_point)
        self.movepoint_check.grid(row=row,column=0,columnspan=2,sticky=tk.W,pady=2)

        tab.columnconfigure(1,weight=1)

    def _build_parameters_tab(self):
        """One row per global parameter, marking which one is being continued.

        A bifurcation diagram is a section through parameter space, so which parameter varies and
        what the others are held at is part of the result, not a setting - hence a permanent table
        rather than a dialog.
        """
        tab=ttk.Frame(self.side,padding=4)
        self.side.add(tab,text="Parameters")
        self.param_tree=ttk.Treeview(tab,columns=("value","role"),show="tree headings",selectmode="browse")
        self.param_tree.heading("#0",text="Parameter")
        self.param_tree.heading("value",text="Value")
        self.param_tree.heading("role",text="Role")
        self.param_tree.column("#0",width=110,minwidth=70)
        self.param_tree.column("value",width=110,minwidth=70)
        self.param_tree.column("role",width=90,minwidth=60)
        self.param_tree.pack(side=tk.TOP,fill=tk.BOTH,expand=True)

        buttons=ttk.Frame(tab,padding=(0,4))
        buttons.pack(side=tk.TOP,fill=tk.X)
        ttk.Button(buttons,text="Continue in this",
                   command=lambda: self._invoke(self._actions["continue_in_selected"])).pack(side=tk.LEFT)
        ttk.Button(buttons,text="Set value...",
                   command=lambda: self._invoke(self._actions["set_selected_parameter"])).pack(side=tk.LEFT,padx=4)

        self.slice_label=ttk.Label(tab,text="",wraplength=300,justify=tk.LEFT,padding=(2,6))
        self.slice_label.pack(side=tk.TOP,fill=tk.X)

    def _build_pointinfo_tab(self):
        tab=ttk.Frame(self.side,padding=8)
        self.side.add(tab,text="Points")
        self.current_info=self._info_box(tab,"Current point")
        self.selected_info=self._info_box(tab,"Selected point")
        ttk.Button(tab,text="Go to selected point",
                   command=lambda: self._invoke(self._actions["goto_selected"])).pack(side=tk.TOP,fill=tk.X,pady=(4,6))

        # The whole spectrum, not just the leading eigenvalue: watching where the others sit is how a
        # Hopf pair coming towards the axis is spotted before it crosses. Its own widget rather than
        # more lines in the boxes above, because neigen is routinely 30-50.
        frame=ttk.LabelFrame(tab,text="Eigenvalues",padding=4)
        frame.pack(side=tk.TOP,fill=tk.BOTH,expand=True)
        self.eigen_label_var=tk.StringVar(value="")
        ttk.Label(frame,textvariable=self.eigen_label_var).pack(side=tk.TOP,anchor=tk.W)
        self.eigen_tree=ttk.Treeview(frame,columns=("re","im"),show="tree headings",
                                     selectmode="none",height=8)
        self.eigen_tree.heading("#0",text="#")
        self.eigen_tree.heading("re",text="Re")
        self.eigen_tree.heading("im",text="Im")
        self.eigen_tree.column("#0",width=34,minwidth=28,stretch=False,anchor=tk.E)
        self.eigen_tree.column("re",width=110,minwidth=70,anchor=tk.E)
        self.eigen_tree.column("im",width=110,minwidth=70,anchor=tk.E)
        escroll=ttk.Scrollbar(frame,orient=tk.VERTICAL,command=self.eigen_tree.yview)
        self.eigen_tree.configure(yscrollcommand=escroll.set)
        escroll.pack(side=tk.RIGHT,fill=tk.Y)
        self.eigen_tree.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)
        # An eigenvalue with a positive real part is the whole point of looking, so it is marked.
        self.eigen_tree.tag_configure("unstable",foreground="#b00000")
        self.eigen_tree.tag_configure("critical",foreground="#804000")
        self._eigen_signature=None

    def _info_box(self,parent,title:str)->tk.Text:
        frame=ttk.LabelFrame(parent,text=title,padding=4)
        frame.pack(side=tk.TOP,fill=tk.X,pady=4)
        txt=tk.Text(frame,height=8,width=34,wrap=tk.WORD,state=tk.DISABLED,relief=tk.FLAT)
        txt.pack(fill=tk.BOTH,expand=True)
        return txt

    def _build_branches_tab(self):
        tab=ttk.Frame(self.side,padding=4)
        self.side.add(tab,text="Branches")
        self.tree=ttk.Treeview(tab,columns=("info",),show="tree headings",selectmode="browse")
        self.tree.heading("#0",text="Branch / point")
        self.tree.heading("info",text="Details")
        self.tree.column("#0",width=92,minwidth=60,stretch=False)
        self.tree.column("info",width=330,minwidth=160)
        scroll=ttk.Scrollbar(tab,orient=tk.VERTICAL,command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT,fill=tk.Y)
        self.tree.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)
        self.tree.bind("<<TreeviewSelect>>",self._on_tree_select)
        self.tree.bind("<Double-1>",lambda *_: self._invoke(self._actions["goto_selected"]))

    def _build_statusbar(self):
        bar=ttk.Frame(self.root,relief=tk.SUNKEN,padding=(6,3))
        bar.pack(side=tk.BOTTOM,fill=tk.X)
        self.status_var=tk.StringVar(value="Ready")
        self.summary_var=tk.StringVar(value="")
        ttk.Label(bar,textvariable=self.status_var,width=34,anchor=tk.W).pack(side=tk.LEFT)
        ttk.Separator(bar,orient=tk.VERTICAL).pack(side=tk.LEFT,fill=tk.Y,padx=6)
        ttk.Label(bar,textvariable=self.summary_var,anchor=tk.W).pack(side=tk.LEFT)

    # ================================================================== dispatch

    def _invoke(self,action:Action):
        """Run one command. The only path from a widget or a key to the controller."""
        if action.id=="abort":
            action.callback()
            self.status_var.set("Abort requested...")
            return
        if self._busy:
            return
        if action.enabled_when is not None and not action.enabled_when():
            return
        try:
            if action.is_solver_task:
                # The log doubles as a transcript of the session: which command was run and where
                # the solution ended up. Reconstructing that from the plot alone is guesswork.
                self.log("> "+action.label)
                self.controller.run_task(action.label,action.callback)
                if self.controller.current_point is not None:
                    self.log("  now at "+self.controller.current_point.describe(
                        self.controller._current_observable))
            else:
                action.callback()
        except Exception as e:
            self._report_error(action.label,e)
        finally:
            if action.is_solver_task:
                self._plots_dirty=True
            self.refresh()
            self._autosave()
            # Deliberately AFTER refresh/autosave and only once the task is over: a multistep sweep
            # marks the plots dirty on every step but must re-render the mesh only when it finishes.
            if self._plots_dirty and self.auto_update_plots and not self._busy:
                self._refresh_plots()

    def _autosave(self):
        """Persist after every command, as the key-driven version did after each of its handlers.

        The saved state covers the axis limits and scales too, so view-only commands are worth
        saving as well - and a session that ends in a crash then still reopens where it was.
        """
        if self.controller.current_point is None or self.controller.current_branch is None:
            return
        try:
            self.controller.save_all()
        except Exception as e:
            self.log("Could not save the diagram: "+repr(e))

    def _report_error(self,what:str,exc:Exception):
        self.log("*** "+what+" failed: "+repr(exc))
        for line in traceback.format_exc().splitlines():
            self.log("    "+line)
        messagebox.showerror(what+" failed",str(exc),parent=self.root)

    def _on_key(self,event):
        # bind_all puts this handler on the "all" bindtag, which every widget of the application
        # carries - including the ones inside dialogs. Without this check, typing in the shortcut
        # editor would also fire the commands being edited.
        if not hasattr(event.widget,"winfo_toplevel") or event.widget.winfo_toplevel() is not self.root:
            return
        cls=event.widget.winfo_class() if hasattr(event.widget,"winfo_class") else ""
        if cls in _TEXT_INPUT_CLASSES:
            return
        acc=event_to_accelerator(event)
        if acc is None:
            return
        aid=self.keymap.action_for(acc)
        if aid is None or aid not in self._actions:
            return
        self._invoke(self._actions[aid])
        return "break"

    def _on_canvas_click(self,event):
        # Clicking the plot takes the focus back from the side-panel entries, otherwise the
        # accelerators would stay swallowed by whichever field was edited last.
        self.canvas.get_tk_widget().focus_set()
        if self._busy or event.xdata is None or event.ydata is None:
            return
        if self.nav.mode:   # pan/zoom active - do not steal the click
            return
        self.controller.select_nearest_point(event.xdata,event.ydata)

    def _on_tree_select(self,*_):
        if self._suspend_tree_callback:
            return
        sel=self.tree.selection()
        if not sel:
            return
        item=sel[0]
        ref=self._tree_items.get(item)
        if ref is None:
            return
        branch,point=ref
        self.controller.select_point(branch,point)

    # ================================================================== observers

    def _blit(self):
        self.canvas.draw()

    def _on_status(self,text:str | None):
        """A long operation reports what it is doing: repaint with the overlay and pump events."""
        if text is not None:
            self.status_var.set(text)
        self.plotter.draw(self.controller,text)
        self._update_panels()
        self.pump()

    def _on_busy(self,name:str | None):
        self._busy=name is not None
        self.status_var.set((name+" ...") if name else "Ready")
        self.abort_button.configure(state="normal" if self._busy else "disabled")
        self._update_enabled_state()
        self.pump()

    def pump(self):
        """Give Tk a chance to repaint and to register an Abort click during a solve."""
        try:
            self.root.update()
        except tk.TclError:
            pass    # window closed underneath a running sweep

    def log(self,text:str):
        print(text)
        self.logtext.configure(state=tk.NORMAL)
        self.logtext.insert(tk.END,text+"\n")
        self.logtext.see(tk.END)
        self.logtext.configure(state=tk.DISABLED)

    # ================================================================== refresh

    def refresh(self,infotext:str | None=None):
        self.plotter.draw(self.controller,infotext)
        self._update_panels()
        self._update_enabled_state()

    def _update_panels(self):
        c=self.controller
        if self.ds_entry.focus_get() is not self.ds_entry:   # do not overwrite what is being typed
            self.ds_var.set("{:.6g}".format(c.ds))
        self.neigen_var.set(str(c.neigen))
        self.shift_var.set("{:g}".format(c.shift) if not isinstance(c.shift,complex) else str(c.shift))
        self.scale_al_var.set(bool(c.scale_arc_length))
        self.splines_var.set(bool(c.interpolated_splines))
        self.mode_var.set(c.mode)
        self.movepoint_var.set(bool(c.move_point_active))
        self.movepoint_check.configure(state="normal" if c.mode=="mp" else "disabled")

        if set(self._axis_choices.values())!=set(c.available_axes()):
            self._rebuild_axis_menus()
        else:
            self._sync_axis_selectors()
            if isinstance(c._paramname,str):
                self._param_var.set(c._paramname)

        for aid,var in self._check_vars.items():
            act=self._actions[aid]
            if act.getter is not None:
                var.set(bool(act.getter()))

        self._fill_info(self.current_info,c.current_point,"current")
        self._fill_info(self.selected_info,c.selected_point,"selected")
        self._update_eigen_list()
        self._update_tree()
        self._update_parameter_table()

        npts=sum(len(b) for b in c.branches)
        summary="{:d} branch{:s}, {:d} points | ds = {:.4g} | {:s}".format(
            len(c.branches),"" if len(c.branches)==1 else "es",npts,c.ds,
            "arclength" if c.mode=="al" else "move point")
        # The slice belongs in the status bar, not only in a panel: a diagram read off the screen
        # without it cannot be interpreted once there is more than one parameter.
        slice_desc=c.describe_current_slice()
        if slice_desc:
            summary+=" | fixed: "+slice_desc
        self.summary_var.set(summary)

    def _update_parameter_table(self):
        c=self.controller
        try:
            names=c.all_parameter_names()
            values=c.current_parameter_values()
        except Exception:
            return   # not initialised yet; the parameters do not exist until define_problem() ran
        branch=c.current_branch
        varying=set(branch.varying_parameters) if branch is not None else set()
        continued=branch.continuation_parameter if branch is not None else None
        tracked=branch.tracked_parameter if branch is not None else None

        existing=set(self.param_tree.get_children())
        for name in names:
            if name not in existing:
                self.param_tree.insert("","end",iid=name,text=name)
            if name==continued:
                role="continued"
            elif name==tracked:
                role="tracked"
            elif name in varying:
                role="varying"
            else:
                role="fixed"
            self.param_tree.item(name,values=("{:.10g}".format(values[name]),role))
        for stale in existing-set(names):
            self.param_tree.delete(stale)

        desc=c.describe_current_slice()
        if branch is None:
            self.slice_label.configure(text="")
        elif branch.kind=="locus":
            self.slice_label.configure(text="Locus of {:s} bifurcations in ({:s}, {:s}).\nFixed: {:s}".format(
                branch.bifurcation_type or "unclassified",str(branch.continuation_parameter),
                str(branch.tracked_parameter),desc or "-"))
        else:
            self.slice_label.configure(text="Diagram continued in {:s}.\nFixed: {:s}".format(
                str(branch.continuation_parameter),desc or "-"))

    def _fill_info(self,widget:tk.Text,point,which:str):
        c=self.controller
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0",tk.END)
        if point is None:
            widget.insert(tk.END,"(no {:s} point)".format(which))
        else:
            lines=[]
            lines.append("{:s} = {:.10g}".format(c._get_paramname_str(),point.param_value))
            obs=c._current_observable
            if obs is not None and obs in point.obs_values:
                lines.append("{:s} = {:.10g}".format(obs,point.obs_values[obs]))
            lines.append("eigenvalue = {:.6g} {:+.6g}i".format(point.eig_value_Re,point.eig_value_Im))
            if point.eig_value_Re==0:
                kind="bifurcation"
                if point.bifurcation_info is not None:
                    kind=str(point.bifurcation_info.get("type",kind))
                lines.append("--> "+kind)
            if point.tag>=0:
                lines.append("tag = {:d}".format(point.tag))
            lines.append("s = {:.6g}".format(point.scoord))
            if point.statefile:
                lines.append(os.path.basename(point.statefile))
            widget.insert(tk.END,"\n".join(lines))
        widget.configure(state=tk.DISABLED)

    def _describe_branch(self,branch)->str:
        """The branch's own summary, plus how it relates to the diagram being worked on."""
        info=branch.describe()
        if branch.kind=="locus":
            # A locus varies two parameters deliberately, so it sits in no single slice and calling it
            # "other slice" would be misleading - describe() already says what it is.
            pass
        elif not self.controller.branch_is_on_current_slice(branch):
            info+="  [other slice]"
        elif not self.controller.branch_can_be_plotted(branch):
            info+="  [not on these axes]"
        if not branch.slice_is_consistent():
            # Its supposedly fixed parameters move along it, so it is not a section of anything.
            info+="  [slice drifts!]"
        return info

    def _update_eigen_list(self):
        """Show the spectrum of the selected point, or of the current one when nothing is selected."""
        c=self.controller
        point=c.selected_point if c.selected_point is not None else c.current_point
        which="selected" if c.selected_point is not None else "current"
        if point is None:
            self.eigen_label_var.set("(no point)")
            self.eigen_tree.delete(*self.eigen_tree.get_children())
            self._eigen_signature=None
            return
        values=list(point.eig_values)
        signature=(id(point),len(values),which)
        if signature==self._eigen_signature:
            return                      # nothing to redo on every redraw
        self._eigen_signature=signature
        if not values:
            # A point from a state file written before the spectrum was recorded. Saying so beats
            # showing only the leading eigenvalue as if it were the whole spectrum.
            self.eigen_label_var.set("{:s} point: only the leading eigenvalue was recorded".format(which))
        else:
            nunstable=point.unstable_count()
            self.eigen_label_var.set("{:s} point: {:d} eigenvalue{:s}, {:d} unstable".format(
                which,len(values),"" if len(values)==1 else "s",nunstable))
        self.eigen_tree.delete(*self.eigen_tree.get_children())
        shown=values if values else [complex(point.eig_value_Re,point.eig_value_Im)]
        for i,v in enumerate(shown):
            re,im=float(v.real),float(v.imag)
            tags=("unstable",) if re>0 else (("critical",) if re==0 else ())
            self.eigen_tree.insert("","end",text=str(i),
                                   values=("{:+.6g}".format(re),"{:+.6g}".format(im)),tags=tags)

    def _update_tree(self):
        c=self.controller
        # Rebuilding the whole tree on every redraw would flicker and lose the scroll position, so
        # it is only rebuilt when the diagram's structure actually changed.
        signature=tuple((id(b),len(b)) for b in c.branches)
        if signature!=self._tree_signature:
            self._tree_signature=signature
            self._suspend_tree_callback=True
            try:
                self.tree.delete(*self.tree.get_children())
                self._tree_items={}
                self._tree_index={}
                self._branch_index={}
                for ib,b in enumerate(c.branches):
                    node=self.tree.insert("","end",text="Branch {:d}".format(ib),
                                          values=(self._describe_branch(b),),open=True)
                    self._tree_items[node]=(b,None)
                    self._branch_index[id(b)]=node
                    for ip,p in enumerate(b):
                        child=self.tree.insert(node,"end",text="  {:d}".format(ip),
                                               values=(p.describe(c.y_axis),))
                        self._tree_items[child]=(b,p)
                        self._tree_index[id(p)]=child
            finally:
                self._suspend_tree_callback=False
        else:
            # The branch rows have to be refreshed even when the structure is unchanged: switching
            # the continuation parameter or the axes changes which branches are on the current slice
            # without adding or removing a single point.
            for b in c.branches:
                node=self._branch_index.get(id(b))
                if node is not None:
                    self.tree.item(node,values=(self._describe_branch(b),))
                for p in b:
                    item=self._tree_index.get(id(p))
                    if item is not None:
                        self.tree.item(item,values=(p.describe(c.y_axis),))

        target=c.selected_point if c.selected_point is not None else c.current_point
        item=self._tree_index.get(id(target)) if target is not None else None
        if item is not None and self.tree.selection()!=(item,):
            self._suspend_tree_callback=True
            try:
                self.tree.selection_set(item)
                self.tree.see(item)
            finally:
                self._suspend_tree_callback=False

    def _update_enabled_state(self):
        for aid,act in self._actions.items():
            enabled=not self._busy
            if enabled and act.enabled_when is not None:
                try:
                    enabled=bool(act.enabled_when())
                except Exception:
                    enabled=False
            state="normal" if enabled else "disabled"
            btn=self._toolbar_buttons.get(aid)
            if btn is not None:
                btn.configure(state=state)
            for menu,index in self._menu_entries.get(aid,[]):
                try:
                    menu.entryconfigure(index,state=state)
                except tk.TclError:
                    pass

    # ================================================================== command bodies

    def _scale_ds(self,factor:float):
        self.controller.ds=self.controller.ds*factor

    def _commit_ds(self):
        try:
            self.controller.ds=float(self.ds_var.get())
        except ValueError:
            self.ds_var.set("{:.6g}".format(self.controller.ds))

    def _commit_eigen(self):
        try:
            self.controller.neigen=int(self.neigen_var.get())
        except ValueError:
            pass
        try:
            self.controller.shift=complex(self.shift_var.get()).real if "j" not in self.shift_var.get() else complex(self.shift_var.get())
        except ValueError:
            pass
        self._update_panels()

    def _commit_mode(self):
        self.controller.mode=self.mode_var.get()

    def _commit_move_point(self):
        if self.controller.move_point_active!=self.movepoint_var.get():
            self.controller.toggle_move_point()

    def _toggle_mode(self):
        self.controller.mode="mp" if self.controller.mode=="al" else "al"

    def _selected_parameter(self)->str | None:
        """The parameter highlighted in the Parameters tab, if any (its row id IS its name)."""
        sel=self.param_tree.selection() if hasattr(self,"param_tree") else ()
        return sel[0] if sel else None

    def _continue_in_selected_parameter(self):
        name=self._selected_parameter()
        if name is not None:
            self.controller.set_continuation_parameter(name)

    def _dialog_set_parameter(self):
        """Move a fixed parameter, which starts a new slice."""
        name=self._selected_parameter()
        if name is None:
            return
        if name==self.controller._paramname:
            messagebox.showinfo("Continuation parameter",
                                "'"+name+"' is the parameter being continued. Step in it, or pick a "
                                "different parameter to continue in first.",parent=self.root)
            return
        current=self.controller.current_parameter_values().get(name,0.0)
        val=simpledialog.askfloat("Set "+name,"New value for "+name+" (the diagram continues there,\n"
                                  "and a new branch is started because it is a different slice):",
                                  parent=self.root,initialvalue=current)
        if val is not None:
            self.controller.set_fixed_parameter(name,val)

    def _refresh_plots(self):
        """Re-render every field pane from the problem's current state."""
        if not self.plot_panes.has_any():
            return
        self.plot_panes.refresh()
        self._plots_dirty=False

    def _toggle_auto_plots(self):
        self.auto_update_plots=not self.auto_update_plots
        if self.auto_update_plots and self._plots_dirty:
            self._refresh_plots()

    def _rebuild_fieldplot_menu(self):
        """Field-plot entries: refresh, the auto toggle, and one eigenfunction toggle per plotter."""
        self.fieldplot_menu.delete(0,"end")
        self._menu_entries.pop("refresh_plots",None)
        self._menu_entries.pop("auto_update_plots",None)
        sources=self.plot_panes.source_plotters()
        if not sources:
            self.fieldplot_menu.add_command(label="(the problem defines no plotter)",state="disabled")
            return
        self._add_menu_item(self.fieldplot_menu,"refresh_plots")
        self._add_menu_item(self.fieldplot_menu,"auto_update_plots")
        self.fieldplot_menu.add_separator()
        # An eigenfunction plot is the same plot with eigenvector/eigenmode set, so it is derived from
        # the plotter the script already wrote rather than asked for again.
        for si,src in enumerate(sources):
            prefix=type(src).__name__+": " if len(sources)>1 else ""
            for mode in ("real","imag"):
                var=self._eigen_vars.setdefault((si,0,mode),tk.BooleanVar())
                var.set(self.plot_panes.eigen_pane_shown(si,0,mode))
                self.fieldplot_menu.add_checkbutton(
                    label=prefix+"Eigenfunction 0 ({:s} part)".format(mode),variable=var,
                    command=lambda si=si,mode=mode: self._toggle_eigen_pane(si,0,mode))

    def _toggle_eigen_pane(self,source_index:int,eigenvector:int,eigenmode:str):
        self.plot_panes.toggle_eigen_pane(source_index,eigenvector,eigenmode)
        self._attach_plot_panes_if_needed()
        self._rebuild_fieldplot_menu()

    def _attach_plot_panes_if_needed(self):
        """The pane column is only added to the layout once there is something in it."""
        if self._plot_panes_attached or self._graphs_pane is None:
            return
        if self.plot_panes.has_any():
            self._graphs_pane.add(self.plot_panes.paned,weight=2)
            self._plot_panes_attached=True

    def _toggle_other_slices(self):
        self.plotter.show_other_slices=not self.plotter.show_other_slices

    def _toggle_splines(self):
        self.controller.interpolated_splines=not self.controller.interpolated_splines

    def _toggle_classify(self):
        self.controller.classify_bifurcations=not self.controller.classify_bifurcations

    def _toggle_demo_video(self):
        self.controller._out_demo_video=not self.controller._out_demo_video

    def _toggle_scale(self,axis:str):
        if axis=="x":
            self.plotter.set_xscale("linear" if self.plotter.get_xscale()=="log" else "log")
        else:
            self.plotter.set_yscale("linear" if self.plotter.get_yscale()=="log" else "log")

    def _reset_view(self):
        cp=self.controller.current_point
        if cp is None:
            return
        x,y=cp.get_coordinate(self.controller._current_observable)
        dx=max(abs(x),1)*1e-4
        dy=max(abs(y),1)*1e-4
        self.plotter.set_xlim(x-dx,x+dx)
        self.plotter.set_ylim(y-dy,y+dy)

    def _cycle_observable(self):
        """The historical "y" command: step the vertical axis to the next observable.

        Deliberately cycles observables only, not the parameters that can now also sit on an axis -
        one keypress should not silently turn an ordinary diagram into a parameter-space plot.
        """
        obs=self.controller.available_observables
        if len(obs)<2:
            return
        current=self.controller.y_axis
        start=obs.index(current[1]) if current[0]!=AXIS_PARAMETER and current[1] in obs else -1
        self._set_observable(obs[(start+1)%len(obs)])

    def _set_observable(self,name:str):
        """Put an observable on the vertical axis (used by the "y" accelerator)."""
        from .model import observable_axis
        self._set_axis("y",self._axis_display(observable_axis(name)))

    def _delete_point(self):
        try:
            self.controller.delete_selected_point()
        except RuntimeError as e:
            messagebox.showwarning("Cannot delete",str(e),parent=self.root)

    def _save_diagram(self):
        self.controller.save_all()
        self.log("Diagram saved")

    def _reload_diagram(self):
        self.controller.load_all(apply_view=self.plotter.apply_saved_view)

    def _new_branch_from_state(self):
        outdir=self.controller.problem.get_output_directory(
            os.path.join(self.controller.data_subdir,"_states"))
        fname=filedialog.askopenfilename(parent=self.root,title="Load state file as a new branch",
                                         initialdir=outdir if os.path.isdir(outdir) else None,
                                         filetypes=[("pyoomph state dump","*.dump"),("All files","*")])
        if not fname:
            return
        self.controller.new_branch_from_state(fname)

    def _save_figure(self):
        fname=filedialog.asksaveasfilename(parent=self.root,title="Save figure",defaultextension=".pdf",
                                           filetypes=[("PDF","*.pdf"),("PNG","*.png"),("SVG","*.svg")])
        if fname:
            self.plotter.savefig(fname)
            self.log("Figure written to "+fname)

    def _dialog_set_ds(self):
        val=simpledialog.askfloat("Step size","Arclength step size ds:",parent=self.root,
                                  initialvalue=self.controller.ds)
        if val is not None:
            self.controller.ds=val

    def _choose_parameter(self,title:str,prompt:str,exclude=())->str | None:
        """Small modal list of the problem's parameters. simpledialog has no chooser of its own."""
        names=[n for n in self.controller.all_parameter_names() if n not in exclude]
        if not names:
            messagebox.showinfo(title,"No other global parameter is available.",parent=self.root)
            return None
        dlg=tk.Toplevel(self.root)
        dlg.title(title)
        ttk.Label(dlg,text=prompt,padding=8,wraplength=340,justify=tk.LEFT).pack(side=tk.TOP,anchor=tk.W)
        var=tk.StringVar(value=names[0])
        box=ttk.Combobox(dlg,state="readonly",values=names,textvariable=var,width=28)
        box.pack(side=tk.TOP,padx=8,pady=4)
        chosen:list[str]=[]
        row=ttk.Frame(dlg,padding=8)
        row.pack(side=tk.BOTTOM,fill=tk.X)
        ttk.Button(row,text="OK",command=lambda: (chosen.append(var.get()),dlg.destroy())).pack(side=tk.RIGHT)
        ttk.Button(row,text="Cancel",command=dlg.destroy).pack(side=tk.RIGHT,padx=4)
        dlg.transient(self.root)
        dlg.grab_set()
        self.root.wait_window(dlg)
        return chosen[0] if chosen else None

    def _dialog_start_locus(self):
        c=self.controller
        tracked=c._get_paramname_str()
        other=self._choose_parameter(
            "Follow this bifurcation",
            "'"+tracked+"' will be adjusted to hold the bifurcation.\nWhich parameter should be "
            "continued along it?",exclude=(tracked,))
        if other is not None:
            c.start_locus(tracked=tracked,continue_in=other)

    def _dialog_leave_locus(self):
        c=self.controller
        branch=c.current_branch
        tracked=branch.tracked_parameter if branch is not None else None
        target=self._choose_parameter(
            "Leave the locus",
            "Step off the bifurcation onto an ordinary branch at this point.\nWhich parameter should "
            "be continued afterwards?")
        if target is not None:
            c.leave_locus(continue_in=target)
        _=tracked

    def _dialog_eigen(self):
        n=simpledialog.askinteger("Eigenvalue settings","Number of eigenvalues to compute:",
                                  parent=self.root,initialvalue=self.controller.neigen,minvalue=1)
        if n is not None:
            self.controller.neigen=n
        s=simpledialog.askfloat("Eigenvalue settings","Shift of the eigensolver:",parent=self.root,
                                initialvalue=float(_real_part(self.controller.shift)))
        if s is not None:
            self.controller.shift=s

    def _dialog_parameter_range(self):
        c=self.controller
        cur=c.parameter_range if c.parameter_range else []
        lo=simpledialog.askfloat("Parameter range","Minimum (Cancel to release the fixed range):",
                                 parent=self.root,initialvalue=cur[0] if len(cur)==2 else self.plotter.get_xlim()[0])
        if lo is None:
            c.parameter_range=[]
            return
        hi=simpledialog.askfloat("Parameter range","Maximum:",parent=self.root,
                                 initialvalue=cur[1] if len(cur)==2 else self.plotter.get_xlim()[1])
        if hi is None:
            c.parameter_range=[]
            return
        c.parameter_range=[lo,hi]

    def _dialog_help(self):
        lines=["Command".ljust(40)+"Shortcut",""]
        for aid,act in self._actions.items():
            acc=self._accel(aid)
            if acc:
                lines.append(act.label.ljust(40)+acc)
        _TextDialog(self.root,"Shortcut reference","\n".join(lines))

    def _dialog_keymap(self):
        _KeyMapDialog(self)

    # ================================================================== lifecycle

    def _on_close(self):
        try:
            self.controller.save_all()
        except Exception as e:
            self.log("Could not save the diagram on exit: "+repr(e))
        self.root.destroy()

    def run(self):
        self._rebuild_axis_menus()
        self._rebuild_fieldplot_menu()
        self.refresh()
        if self.plot_panes.has_any() and self.auto_update_plots:
            self._refresh_plots()
        self.root.mainloop()


def _real_part(v):
    return v.real if isinstance(v,complex) else v


class _Tooltip:
    """Minimal hover tooltip - ttk has none of its own."""

    def __init__(self,widget,text:str) -> None:
        self.widget=widget
        self.text=text
        self.tip:tk.Toplevel | None=None
        widget.bind("<Enter>",self._show)
        widget.bind("<Leave>",self._hide)

    def _show(self,_event=None):
        if self.tip is not None:
            return
        x=self.widget.winfo_rootx()+10
        y=self.widget.winfo_rooty()+self.widget.winfo_height()+4
        self.tip=tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry("+{:d}+{:d}".format(x,y))
        tk.Label(self.tip,text=self.text,relief=tk.SOLID,borderwidth=1,padx=4,pady=2,
                 background="#ffffe0").pack()

    def _hide(self,_event=None):
        if self.tip is not None:
            self.tip.destroy()
            self.tip=None


class _TextDialog(tk.Toplevel):
    def __init__(self,parent,title:str,text:str) -> None:
        super().__init__(parent)
        self.title(title)
        box=tk.Text(self,wrap=tk.NONE,width=64,height=28)
        box.insert("1.0",text)
        box.configure(state=tk.DISABLED)
        box.pack(side=tk.TOP,fill=tk.BOTH,expand=True,padx=6,pady=6)
        ttk.Button(self,text="Close",command=self.destroy).pack(side=tk.BOTTOM,pady=6)
        self.transient(parent)


class _KeyMapDialog(tk.Toplevel):
    """Lists every command and lets its shortcut be reassigned by pressing a key."""

    def __init__(self,app:BifurcationTkApp) -> None:
        super().__init__(app.root)
        self.app=app
        self.title("Keyboard shortcuts")
        self.geometry("520x520")
        self._capturing=False

        self.tree=ttk.Treeview(self,columns=("shortcut",),show="tree headings",selectmode="browse")
        self.tree.heading("#0",text="Command")
        self.tree.heading("shortcut",text="Shortcut")
        self.tree.column("#0",width=320)
        self.tree.column("shortcut",width=140)
        self.tree.pack(side=tk.TOP,fill=tk.BOTH,expand=True,padx=6,pady=6)

        self.hint=ttk.Label(self,text="Select a command, then press \"Assign\" and hit the new key.",
                            padding=(6,2))
        self.hint.pack(side=tk.TOP,anchor=tk.W)

        buttons=ttk.Frame(self,padding=6)
        buttons.pack(side=tk.BOTTOM,fill=tk.X)
        ttk.Button(buttons,text="Assign...",command=self._start_capture).pack(side=tk.LEFT)
        ttk.Button(buttons,text="Clear",command=self._clear).pack(side=tk.LEFT,padx=4)
        ttk.Button(buttons,text="Reset all to defaults",command=self._reset).pack(side=tk.LEFT)
        ttk.Button(buttons,text="Close",command=self._close).pack(side=tk.RIGHT)

        self._fill()
        self.bind("<Key>",self._on_key)
        self.transient(app.root)
        self.grab_set()

    def _fill(self):
        self.tree.delete(*self.tree.get_children())
        self._rows={}
        for aid,act in self.app._actions.items():
            item=self.tree.insert("","end",text=act.label,
                                  values=(format_accelerator(self.app.keymap.get(aid)),))
            self._rows[item]=aid

    def _selected_action(self)->str | None:
        sel=self.tree.selection()
        return self._rows.get(sel[0]) if sel else None

    def _start_capture(self):
        if self._selected_action() is None:
            return
        self._capturing=True
        self.hint.configure(text="Press the new key now (Esc cancels)...")

    def _on_key(self,event):
        if not self._capturing:
            return
        acc=event_to_accelerator(event)
        if acc is None:
            return "break"
        self._capturing=False
        self.hint.configure(text="Select a command, then press \"Assign\" and hit the new key.")
        if acc=="escape" and self.app.keymap.get("abort")=="escape":
            return "break"   # do not let Esc silently steal the abort binding
        aid=self._selected_action()
        if aid is not None:
            previous=self.app.keymap.action_for(acc)
            self.app.keymap.set(aid,acc)
            if previous is not None and previous!=aid:
                self.app.log("Shortcut {:s} taken from \"{:s}\"".format(
                    format_accelerator(acc),self.app._actions[previous].label))
            self._fill()
        return "break"

    def _clear(self):
        aid=self._selected_action()
        if aid is not None:
            self.app.keymap.set(aid,None)
            self._fill()

    def _reset(self):
        self.app.keymap.reset_to_defaults()
        self._fill()

    def _close(self):
        self.app.keymap.save()
        self.app._rebuild_menu_accelerators()
        self.grab_release()
        self.destroy()


from ...typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
