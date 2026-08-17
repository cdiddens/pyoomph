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

"""Numerics and diagram bookkeeping behind the bifurcation GUI.

This module knows about :py:class:`~pyoomph.generic.problem.Problem` but not about matplotlib or
tkinter, so the whole workflow (continuation, bifurcation location, branch switching, saving and
reloading) can be driven headlessly from a test or a batch script. The user interface talks to it
through the observer hooks installed by :py:meth:`BifurcationController.set_observer`.
"""

from ...generic.codegen import EquationTree
from ...generic import Problem
from ... import _pyoomph_core as _pyoomph
from ...typings import *

from .model import BifurcationGUISolutionPoint, BifurcationGUISolutionBranch

from typing import Protocol
from pathlib import Path
import numpy
import os
import json
import glob
import shutil


class BifurcationViewLimits(Protocol):
    """The bit of the plot the numerics genuinely need.

    Three algorithms are defined in terms of what is currently *visible*, not of the data alone: the
    multistep sweep stops when it leaves the axes, the point-insertion heuristic measures branch
    length in axis-normalized units (so a fold looks like a fold irrespective of the units of the
    observable), and the saved state remembers the view. Rather than handing the controller a
    figure, the view supplies just these four numbers.
    """

    def get_xlim(self) -> tuple[float,float]: ...
    def get_ylim(self) -> tuple[float,float]: ...
    def get_xscale(self) -> str: ...
    def get_yscale(self) -> str: ...


class _FixedViewLimits:
    """Stand-in used when the controller runs without a plot (tests, batch scripts)."""

    def __init__(self,xlim=(0.0,1.0),ylim=(0.0,1.0)) -> None:
        self._xlim,self._ylim=tuple(xlim),tuple(ylim)

    def get_xlim(self): return self._xlim
    def get_ylim(self): return self._ylim
    def get_xscale(self): return "linear"
    def get_yscale(self): return "linear"


class InlineExecutor:
    """Runs a solver task in the calling (i.e. the GUI) thread.

    This mirrors what the tool has always done: the Newton/eigen solve happens inside the event
    callback and the window is repainted at the points that used to call ``update_plot()``. The
    window is therefore frozen *within* a single solve, but nothing about the C++/PETSc/JIT call
    path changes, and the abort flag is picked up between steps exactly as before.
    """

    #: Whether the controller may touch problem/state from the thread calling ``submit``.
    is_inline=True

    def submit(self,fn:Callable[[],Any])->Any:
        return fn()


class BifurcationController:
    """Drives a :py:class:`~pyoomph.generic.problem.Problem` along its solution branches.

    All state of a diagram lives here: the branches, which point is currently loaded into the
    problem, which one the user has selected, the continuation step size, and the eigensolver
    settings.
    """

    def __init__(self,problem:Problem,parameter:"str | _pyoomph.GiNaC_GlobalParam | None"=None) -> None:
        self.problem=problem
        self.problem._runmode="overwrite"
        self.problem.write_states=True
        self.problem.continuation_data_in_states=True
        self.data_subdir="_bifurcation_gui_data"
        self.neigen=10
        self.shift=0
        self.branches:list[BifurcationGUISolutionBranch]=[]
        self.current_branch:BifurcationGUISolutionBranch | None=None
        self.current_point:BifurcationGUISolutionPoint | None=None
        self.selected_point:BifurcationGUISolutionPoint | None=None
        self.selected_branch:BifurcationGUISolutionBranch | None=None
        self._last_ds=1
        self._tangs:dict[str,NPFloatArray]={}
        self._paramname=parameter
        self.parameter_range:list[float]=[]
        self._out_demo_video=False
        self._demo_video_step=0
        self._current_observable:str | None=None
        self._avail_observables:list[str]=[]
        self._observable_funcs:dict[str,Callable[...,float]] | None=None
        self._mode="al"
        self._move_point=False
        self.interpolated_splines=False
        self.scale_arc_length=True
        self._state_step=0
        self._abort_requested=False
        #: Write all observable values to the output files
        self.output_all_observables=False
        self._initial_view=None
        self.classify_bifurcations=False

        #: Supplies the visible axis ranges, see :py:class:`BifurcationViewLimits`.
        self.view:BifurcationViewLimits=_FixedViewLimits()
        #: Swappable so a worker-thread executor can be dropped in without touching the numerics.
        self.executor=InlineExecutor()

        self._on_changed:Callable[[],None] | None=None
        self._on_status:Callable[[str | None],None] | None=None
        self._on_log:Callable[[str],None] | None=None
        self._on_busy:Callable[[str | None],None] | None=None

    # ------------------------------------------------------------------ observers

    def set_observer(self,*,on_changed:Callable[[],None] | None=None,on_status:Callable[[str | None],None] | None=None,on_log:Callable[[str],None] | None=None,on_busy:Callable[[str | None],None] | None=None):
        """Install the callbacks through which a user interface follows this controller.

        ``on_changed`` - the diagram or the selection changed and should be redrawn.
        ``on_status``  - a long operation wants a label shown (``None`` clears it); this also
                         repaints, which is what lets the abort flag be picked up mid-sweep.
        ``on_log``     - a line of progress text.
        ``on_busy``    - a solver task starts (label) or ends (``None``).
        """
        if on_changed is not None: self._on_changed=on_changed
        if on_status is not None: self._on_status=on_status
        if on_log is not None: self._on_log=on_log
        if on_busy is not None: self._on_busy=on_busy

    def _changed(self):
        if self._on_changed is not None:
            self._on_changed()

    def _status(self,text:str | None):
        if self._on_status is not None:
            self._on_status(text)

    def log(self,*args:Any):
        text=" ".join(str(a) for a in args)
        if self._on_log is not None:
            self._on_log(text)
        else:
            print(text)

    def request_abort(self):
        """Ask a running multistep sweep to stop after the current step."""
        self._abort_requested=True

    @property
    def abort_requested(self)->bool:
        return self._abort_requested

    def clear_abort(self):
        self._abort_requested=False

    def run_task(self,name:str,fn:Callable[[],Any])->Any:
        """The single point through which every solver-touching command is dispatched.

        Everything the user interface triggers goes through here, so that disabling widgets, the
        busy indicator and (later) moving the work to a background thread are decided in exactly
        one place.
        """
        if self._on_busy is not None:
            self._on_busy(name)
        try:
            return self.executor.submit(fn)
        finally:
            if self._on_busy is not None:
                self._on_busy(None)
            self._status(None)

    # ------------------------------------------------------------------ accessors

    # The following small accessors document/enforce invariants that hold once the GUI has been
    # started via start(): current_point/current_branch are always set from then on. Using them
    # (instead of the raw Optional attributes) lets pyright narrow away the None-case at the many
    # call sites that rely on this invariant without weakening the Optional attribute types.
    def _get_current_point(self) -> "BifurcationGUISolutionPoint":
        assert self.current_point is not None
        return self.current_point

    def _get_current_branch(self) -> "BifurcationGUISolutionBranch":
        assert self.current_branch is not None
        return self.current_branch

    def _get_current_observable(self) -> str:
        assert self._current_observable is not None
        return self._current_observable

    def _get_paramname_str(self) -> str:
        assert isinstance(self._paramname,str)
        return self._paramname

    def get_bifurcation_parameter(self)->"_pyoomph.GiNaC_GlobalParam":
        if self._paramname is None:
            raise RuntimeError("This function must be customized or parameter must be passed")
        elif isinstance(self._paramname,str):
            if self._paramname not in self.problem.get_global_parameter_names():
                raise RuntimeError("Parameter "+self._paramname+" not part of the problem")
            return self.problem.get_global_parameter(self._paramname)
        else:
            return self._paramname

    def set_initial_view(self,xmin,xmax,ymin,ymax):
        self._initial_view=[xmin,xmax,ymin,ymax]

    @property
    def ds(self)->float:
        return self._last_ds

    @ds.setter
    def ds(self,value:float):
        self._last_ds=float(value)
        self._changed()

    @property
    def mode(self)->str:
        return self._mode

    @mode.setter
    def mode(self,value:str):
        if value not in ("al","mp"):
            raise ValueError("Mode must be 'al' (arclength) or 'mp' (move point)")
        self._mode=value
        self._changed()

    @property
    def available_observables(self)->list[str]:
        return list(self._avail_observables)

    @property
    def observable(self)->str:
        return self._get_current_observable()

    @observable.setter
    def observable(self,name:str):
        if name not in self._avail_observables:
            raise ValueError("Unknown observable "+str(name))
        self._current_observable=name

    def get_tangent(self,obs:str | None=None)->"NPFloatArray | None":
        return self._tangs.get(obs if obs is not None else self._get_current_observable())

    # ------------------------------------------------------------------ observables

    # By default, we allow to access all integral observables (not beginning with _) and all ODE dofs
    def evaluate_observables(self)->dict[str,float]:
        if self._observable_funcs is None:
            obs:dict[str,Callable[...,float]]={}
            def recursive_add_spatial_domains(eqtree:EquationTree):
                if eqtree._equations is not None and eqtree.get_equations()._is_ode()==False:
                    ifuncs_list=eqtree.get_mesh().list_integral_functions()
                    deps = eqtree.get_code_gen()._dependent_integral_funcs
                    ifuncs: set[str] = set(ifuncs_list)
                    ifuncs.update(deps.keys())
                    bn=eqtree.get_full_path().lstrip("/")
                    for valn in ifuncs:
                        if not valn.startswith("_"):
                            obs[bn+"/"+valn]=lambda domname=bn,valn=valn: float(self.problem.get_mesh(domname).evaluate_observable(valn))
                for child in eqtree.get_children().values():
                    recursive_add_spatial_domains(child)

            for name,eqtree in self.problem._equation_system.get_children().items():
                if eqtree.get_equations()._is_ode()==True:
                    ode=self.problem.get_ode(name)
                    _vals, inds = ode.get_element()._ode_elem_to_numpy()
                    for valn in inds.keys():
                        if not valn.startswith("_"):
                            obs[name+"/"+valn]=lambda domname=name,valn=valn: self.problem.get_ode(domname).get_value(valn,dimensional=True,as_float=True)
            recursive_add_spatial_domains(self.problem._equation_system)
            if len(obs)==0:
                raise RuntimeError("Could not identify an observable. Add ODEs or IntegralObservables to find them")
            self._observable_funcs=obs.copy()
        return {n:func() for n,func in self._observable_funcs.items()}

    # ------------------------------------------------------------------ diagram bookkeeping

    def new_branch_from_state(self,statefile):
        self.current_branch=BifurcationGUISolutionBranch()
        self.selected_branch=self.current_branch
        self.branches.append(self.current_branch)
        self.problem.load_state(statefile,ignore_continuation_data=True,ignore_eigendata=True)
        self.problem.reset_arc_length_parameters()
        self.problem.solve_eigenproblem(self.neigen,self.shift)
        self._add_current_state()
        self._update_tangents()
        self._changed()

    def _add_current_state(self):
        if self.current_branch is None:
            self.current_branch=BifurcationGUISolutionBranch()
            self.selected_branch=self.current_branch
            self.branches.append(self.current_branch)
        state_file=self.problem.get_output_directory(os.path.join(self.data_subdir,"_states","state_{:06d}.dump".format(self._state_step)))
        p=BifurcationGUISolutionPoint(self.get_bifurcation_parameter().value,self.evaluate_observables(),self.problem.get_last_eigenvalues()[0],state_file,self._state_step)
        if p.eig_value_Re==0 and self.classify_bifurcations:
            from ...generic.bifurcation_tools import NormalFormCalculator
            p.bifurcation_info=NormalFormCalculator(self.problem).get_normal_form(self.get_bifurcation_parameter().get_name())
        self.problem.save_state(state_file)
        self._state_step+=1
        self.current_point=p
        self.selected_point=None
        self.selected_branch=None
        self.current_branch.append(p)

    def load_pt(self,pt):
        self.problem.load_state(pt.statefile,ignore_outstep=True)
        self.current_point=pt
        if len(self.problem.get_arclength_dof_derivative_vector())==0:
            self.problem.reset_arc_length_parameters()
        self._update_tangents()
        self.selected_point=None

    def goto_selected_point(self):
        """Load the selected point into the problem, making it the current one."""
        if self.selected_point is None or self.selected_point is self.current_point:
            return False
        sel=self.selected_point
        if self.selected_branch is not None:
            self.current_branch=self.selected_branch
        self.run_task("Loading state",lambda: self.load_pt(sel))
        self._changed()
        return True

    def select_point(self,branch:BifurcationGUISolutionBranch | None,point:BifurcationGUISolutionPoint | None):
        self.selected_branch=branch
        self.selected_point=point
        self._changed()

    def select_nearest_point(self,x:float,y:float):
        """Pick the point closest to (x,y) in axis-normalized units, if it is close enough.

        Used by the click-on-the-plot selection; the 1e-2 cut-off is a squared normalized distance,
        i.e. clicking further than ~10% of the axes away selects nothing.
        """
        xl=self.view.get_xlim()
        yl=self.view.get_ylim()
        dx=xl[1]-xl[0]
        dy=yl[1]-yl[0]
        bestbranch,bestpoint=None,None
        bestdist=1e30
        for b in self.branches:
            for p in b:
                c=p.get_coordinate(self._current_observable)
                dist=((c[0]-x)/dx)**2+((c[1]-y)/dy)**2
                if dist<bestdist:
                    bestbranch=b
                    bestpoint=p
                    bestdist=dist
        if bestpoint and bestdist<1e-2:
            self.selected_branch=bestbranch
            self.selected_point=bestpoint
            self._changed()
            return True
        return False

    def _ensure_selection(self):
        """Fall back to the current point when nothing (or something stale) is selected."""
        cur_point=self._get_current_point()
        cur_branch=self._get_current_branch()
        if self.selected_point is None or self.selected_branch is None or self.selected_point not in self.selected_branch:
            self.selected_point=cur_point
            self.selected_branch=cur_branch
        assert self.selected_point is not None and self.selected_branch is not None
        return self.selected_point,self.selected_branch

    def select_relative(self,where:str):
        """Move the selection along its branch. ``where`` is prev/next/first/last."""
        sel_point,sel_branch=self._ensure_selection()
        index=sel_branch.index(sel_point)
        origindex=index
        if where=="prev" and index>0:
            index-=1
        elif where=="next" and index+1<len(sel_branch):
            index+=1
        elif where=="last" and sel_point is not sel_branch[-1]:
            index=len(sel_branch)-1
        elif where=="first" and sel_point is not sel_branch[0]:
            index=0
        else:
            return False

        if self._mode=="mp" and self._move_point:
            backup=sel_branch[index]
            sel_branch[index]=sel_point
            sel_branch[origindex]=backup
            self.reorder_branch_upon_point_insertion(sel_branch,None)
        else:
            self.selected_point=sel_branch[index]
        self._changed()
        return True

    def toggle_move_point(self):
        self._move_point=not self._move_point
        self._changed()

    @property
    def move_point_active(self)->bool:
        return self._move_point

    def delete_selected_point(self):
        """Remove the selected point (default: the current one) and its state file.

        Deleting the point the problem is sitting on has to reload a neighbour, and deleting the
        last point of a branch removes the branch itself.
        """
        cur_point=self._get_current_point()
        cur_branch=self._get_current_branch()
        torem,sel_branch=self._ensure_selection()
        index=sel_branch.index(torem)
        if index==0:
            if len(sel_branch)==1:
                if len(self.branches)==1:
                    raise RuntimeError("Cannot delete the last point")
                if torem.statefile:
                    self._remove_statefile(torem.statefile)

                self.branches.remove(sel_branch)
                if sel_branch==cur_branch:
                    self.current_branch=self.branches[-1]
                    new_branch=self._get_current_branch()
                    self.current_point=new_branch[-1]
                    self.load_pt(new_branch[-1])
                    self.selected_point=self.current_point
                    self.selected_branch=self.current_branch
                else:
                    self.selected_point=cur_point
                    self.selected_branch=cur_branch
                self._changed()
                return
            else:
                if torem==cur_point:
                    self.load_pt(cur_branch[1])
                    self.current_point=cur_branch[1]
                else:
                    self.selected_point=sel_branch[1]
                    self.selected_branch=sel_branch
        else:
            if torem==cur_point:
                self.load_pt(cur_branch[index-1])
                self.current_point=cur_branch[index-1]
            else:
                if index>0:
                    self.selected_point=sel_branch[index-1]
                elif index+1<len(cur_branch):
                    self.selected_point=cur_branch[index+1]
                else:
                    self.selected_point=None
        if torem.statefile:
            self._remove_statefile(torem.statefile)
        sel_branch.remove(torem)
        self._changed()

    def _remove_statefile(self,fname:str):
        # A missing dump must not abort a delete: the diagram is reloadable without it, and users
        # do clean out _states by hand between sessions.
        try:
            os.remove(fname)
        except OSError as e:
            self.log("Could not remove state file",fname,":",e)

    def toggle_point_tag(self,pt,tag):
        if pt is None:
            return
        if pt.tag==tag:
            pt.tag=-1
            return
        for b in self.branches:
            for p in b:
                if p.tag==tag:
                    p.tag=-1
        pt.tag=tag

    def tag_selected_point(self,tag:int):
        self.toggle_point_tag(self.selected_point if self.selected_point is not None else self.current_point,tag)
        self.save_all()
        self._changed()

    # ------------------------------------------------------------------ output

    def output_curves(self):
        paramname=self._get_paramname_str()
        observable=self._get_current_observable()
        ddir=self.problem.get_output_directory(self.data_subdir)
        odir=os.path.join(ddir,"output")

        Path(odir).mkdir(parents=True,exist_ok=True)
        globres=glob.glob(os.path.join(odir,"branch*","*.txt"))
        for g in globres:
            os.remove(g)
        globres=glob.glob(os.path.join(odir,"*.dump"))
        for g in globres:
            os.remove(g)

        for ib,b in enumerate(self.branches):
            bdir=os.path.join(odir,"branch{:03d}".format(ib))
            Path(bdir).mkdir(parents=True,exist_ok=True)
            if self.interpolated_splines:
                smoothedsegs,stabs=b.smooth_branch_stab_list(self._current_observable if not self.output_all_observables else None,100)
            else:
                smoothedsegs,stabs=b.to_branch_stab_list(self._current_observable if not self.output_all_observables else None)
            istab=0
            iunstab=0
            for seg,stab in zip(smoothedsegs,stabs):
                if stab:
                    fn="smoothed_stable_{:03d}.txt".format(istab)
                    istab+=1
                else:
                    fn="smoothed_unstable_{:03d}.txt".format(iunstab)
                    iunstab+=1
                numpy.savetxt(os.path.join(bdir,fn),seg[:,:-1],header=paramname+"\t"+observable+"\tReEigen\tImEigen")
            nbif=0
            for p in b:
                if p.eig_value_Re==0:
                    fn="bifurcation_{:03d}.txt".format(nbif)
                    pc=p.get_coordinate(self._current_observable,with_s=False,with_eigen=True)
                    numpy.savetxt(os.path.join(bdir,fn),numpy.array([pc],ndmin=2),header=paramname+"\t"+observable+"\tReEigen\tImEigen")
                    nbif+=1
                if p.tag>=0 and p.statefile is not None:
                    shutil.copy2(p.statefile, os.path.join(odir,"tag{:02d}.dump".format(p.tag)))
                    fn="tag_{:02d}.txt".format(p.tag)
                    pc=p.get_coordinate(self._current_observable,with_s=False)
                    numpy.savetxt(os.path.join(odir,fn),numpy.array([pc],ndmin=2),header=paramname+"\t"+observable)
        self.log("Exported curves to",odir)
        return odir

    # ------------------------------------------------------------------ tangents

    def _update_tangents(self):
        FD_eps=1e-6
        cp=self._get_current_point()
        backup,_=self.problem.get_current_dofs()
        dp=self.problem.get_arc_length_parameter_derivative()
        ddof=numpy.array(self.problem.get_arclength_dof_derivative_vector())
        if len(ddof)>0:
            self.problem.set_current_dofs(backup+FD_eps*ddof)
            po=self.evaluate_observables()
        else:
            po=cp.obs_values.copy()

        for k in self._avail_observables:
            do=(po[k]-cp.obs_values[k])/FD_eps
            self._tangs[k]=numpy.array([dp,do])
        cp._tangs=self._tangs.copy()


        if cp.bifurcation_info is not None:
            bi=cp.bifurcation_info
            cp._branch_switch_tangs=[]
            if bi["type"]=="transcritical":
                for dptr in [-dp,dp]:
                    ddof=numpy.array(bi["perturbation_predictor"](dptr))
                    self.problem.set_current_dofs(backup+FD_eps*ddof)
                    po=self.evaluate_observables()
                    btangtangs={}
                    for k in self._avail_observables:
                        do=(po[k]-cp.obs_values[k])/FD_eps
                        btangtangs[k]=numpy.array([dptr,do])
                    cp._branch_switch_tangs.append(btangtangs)

        self.problem.set_current_dofs(backup)

    # ------------------------------------------------------------------ solver commands

    def branch_switch(self):
        cp=self._get_current_point()
        if cp.eig_value_Re!=0:
            raise RuntimeError("Can only switch branches as bifurcations")
        if cp.bifurcation_info is None:
            raise RuntimeError("No bifurcation info available. Please set gui.classify_bifurcations=True")
        bi=cp.bifurcation_info
        if bi["type"]=="fold":
            self.log("Cannot switch branches at fold bifurcations")
            return

        param=self.get_bifurcation_parameter()
        curr=self.problem.get_current_dofs()[0]

        if cp._branch_switch_tangs is None or len(cp._branch_switch_tangs)==0:
            ds=0.001
            dp=bi["param_predictor"](ds)
            du=bi["perturbation_predictor"](ds)
        else:
            dp=cp._branch_switch_tangs[0][self._current_observable][0]*self._last_ds
            du=bi["perturbation_predictor"](dp)
            ds=self._last_ds

        self.log("Branch switching with dp=",dp,"dunorm",numpy.linalg.norm(du))
        self._status("BRANCH SWITCHING")

        self.problem._update_dof_vectors_for_continuation(du,curr)
        self.problem._update_param_info_for_continuation(dp,param.value)
        self.problem.arclength_continuation(param,ds)

        self.branches.append(BifurcationGUISolutionBranch())
        self.current_branch=self.branches[-1]
        self.selected_branch=self.current_branch
        self._tangs={}
        self.problem.solve_eigenproblem(self.neigen,self.shift)
        self._add_current_state()
        self._update_tangents()
        self._mode="al"

    def transient_leave_branch(self,eigenindex=0):
        self._status("LEAVING BRANCH TRANSIENTLY")
        cp=self._get_current_point()
        eig=numpy.sqrt(cp.eig_value_Re**2+cp.eig_value_Im**2)
        eig=max(1e-4,eig)
        tsnd=1/eig
        ts=self.problem.get_scaling("temporal")*tsnd

        self.problem.reset_arc_length_parameters()
        self.problem.set_current_time(0)

        self.problem.perturb_dofs(0.1*numpy.real(self.problem.get_last_eigenvectors()[eigenindex]))
        self.problem.initialise_dt(tsnd)
        self.problem.assign_initial_values_impulsive(tsnd)
        self.problem.timestepper.set_num_unsteady_steps_done(0)
        self.problem._taken_already_an_unsteady_step=False
        self.problem._last_step_was_stationary=True
        self.problem.deactivate_bifurcation_tracking()
        self.problem.run(1000*ts,startstep=ts,temporal_error=1,outstep=False,do_not_set_IC=True)
        self.problem.set_current_time(0)
        self.problem.solve(max_newton_iterations=20)
        self.problem.solve_eigenproblem(self.neigen,self.shift)
        self.branches.append(BifurcationGUISolutionBranch())
        self.current_branch=self.branches[-1]
        self.selected_branch=self.current_branch
        self._tangs={}
        self._add_current_state()
        self._update_tangents()
        self._mode="al"
        self.log("Integrated",1000*ts)


    def reorder_branch_upon_point_insertion(self,branch:BifurcationGUISolutionBranch,newp:BifurcationGUISolutionPoint | None):
        if newp is not None:
            if newp not in branch:
                branch.append(newp)

        if  len(branch)<3:
            branch[0].scoord=0
            branch[len(branch)-1].scoord=0
            if newp is not None:
                branch[branch.index(newp)].scoord=1
            elif len(branch)>1:
                branch[-1].scoord=1
            branch.sort(key=lambda p : p.scoord)
            return

        xlim=self.view.get_xlim()
        ylim=self.view.get_ylim()
        pscale=1/abs(xlim[1]-xlim[0])
        obsscale=1/abs(ylim[1]-ylim[0])

        # Renormalize s by going along the arclength. We assume it is all well ordered here
        al=0
        if newp==branch[0]:
            last=branch[1].get_coordinate(self._current_observable)
        else:
            last=branch[0].get_coordinate(self._current_observable)
        xbase=[]
        ybase=[]
        sbase=[]
        for p in branch:
            if p==newp:
                continue
            curr=p.get_coordinate(self._current_observable)
            dal=numpy.sqrt((curr[0]-last[0])**2*pscale**2+(curr[1]-last[1])**2*obsscale**2)
            al=al+dal
            p.scoord=al
            last=curr
            xbase.append(float(curr[0]*pscale))
            ybase.append(float(curr[1]*obsscale))
        # Now we have a scoord from 0 to 1
        # Note: this used to iterate over self.current_branch instead of the branch parameter,
        # which is a bug whenever this method is called with a branch other than current_branch
        # (e.g. from the point-navigation commands operating on selected_branch) -- it normalized
        # the wrong branch's scoords while leaving the just-updated `branch` unnormalized.
        for p in branch:
            if p==newp:
                continue
            p.scoord/=al
            sbase.append(p.scoord)

        if newp is not None:
            xn,yn=newp.get_coordinate(self._current_observable)
            xn,yn=float(xn*pscale),float(yn*obsscale)
            # Quite demanding, but lets give it a try: Could be improved of course
            shortest_l=1e50
            shortest_news:float=0
            # Add some additional contribution due penalized strong changes in direction
            def tangdot(x,y,index):
                tdot=(x[index]-x[index-1])*(x[index+1]-x[index])+(y[index]-y[index-1])*(y[index+1]-y[index])
                distdenom=(x[index+1]-x[index-1])**2+(y[index+1]-y[index-1])**2
                return tdot/numpy.sqrt(distdenom)
            for insert_index in range(len(xbase)+1):
                # Try to insert the new point at each index and measure the length of the branch
                # TODO: Most of these calculations can be only done once instead within this
                xnew=xbase.copy()
                ynew=ybase.copy()
                xnew.insert(insert_index,xn)
                ynew.insert(insert_index,yn)
                if insert_index==0:
                    sc=sbase[0]-0.5*(sbase[1]-sbase[0])
                    al=-tangdot(xnew,ynew,1)
                elif insert_index==len(xbase):
                    sc=sbase[-1]+0.5*(sbase[-1]-sbase[-2])
                    al=-tangdot(xnew,ynew,insert_index-1)
                else:
                    sc=0.5*(sbase[insert_index-1]+sbase[insert_index])
                    al=-tangdot(xnew,ynew,insert_index)
                lx,ly=xnew[0],ynew[0]
                for x,y in zip(xnew,ynew):
                    dal=numpy.sqrt((x-lx)**2+(y-ly)**2)
                    al+=dal
                    lx,ly=x,y

                if al<shortest_l:
                    shortest_l=al
                    shortest_news=sc
            newp.scoord=shortest_news
        branch.sort(key=lambda p : p.scoord)

    def _tangent_length(self)->float:
        """Length of the current tangent in (parameter, observable) space, 0 if there is none."""
        obs=self._current_observable
        if obs is None:
            return 0.0
        tvec=self._tangs.get(obs)
        if tvec is None:
            return 0.0
        return float(numpy.sqrt(tvec[0]*tvec[0]+tvec[1]*tvec[1]))

    def multistep(self):
        """Keep stepping until the branch leaves the visible axes or the user aborts.

        The step size is capped at the distance the first step covered, so the sweep does not run
        away once the branch turns and the continuation is free to grow ds.
        """
        assert self._current_observable is not None
        # On a brand-new diagram no continuation direction exists yet (_update_tangents has never
        # run), so the reference length would be missing or zero and every subsequent step would be
        # scaled down to nothing. One ordinary step establishes it.
        if self._tangent_length()==0.0:
            self.step()
            self.save_all()
        xlim=self.view.get_xlim()
        ylim=self.view.get_ylim()
        xscale=self.view.get_xscale()
        max_ds=self._tangent_length()*abs(self._last_ds)
        cp0=self._get_current_point().get_coordinate(self._current_observable)
        while True:
            cp=self._get_current_point().get_coordinate(self._current_observable)
            if cp[0]<xlim[0] or cp[0]>xlim[1] or cp[1]<ylim[0] or cp[1]>ylim[1]:
                break
            if self._abort_requested:
                self._abort_requested=False
                self.log("Multistep aborted")
                break
            self.step()
            self.save_all()
            tvec=self._tangs[self._current_observable]*self._last_ds
            if xscale=="log":
                xlogfactor=(cp0[0]/cp[0])**2
            else:
                xlogfactor=1
            ds=numpy.sqrt(xlogfactor*tvec[0]*tvec[0]+tvec[1]*tvec[1])
            if ds>0:
                self._last_ds*=min(1,max_ds/ds)
        self._changed()

    def step(self,ds=None):
        if ds is None:
            ds=self._last_ds
        origin=self._get_current_point()
        if self._abort_requested:
            return
        self._status("ARCLENGTH STEPPING")
        ds=self.problem.arclength_continuation(self.get_bifurcation_parameter(),ds)
        self.problem.solve_eigenproblem(self.neigen,self.shift)

        self._add_current_state()

        self.reorder_branch_upon_point_insertion(self._get_current_branch(),self._get_current_point())
        self._last_ds=ds
        self._update_tangents()
        if origin._tangs is None or len(origin._tangs)==0:
            origin._tangs=self._get_current_point()._tangs.copy()

        return ds

    def step_and_shrink(self):
        """One step, but never let ds grow - the ``*`` command of the old key interface."""
        ds_backup=self._last_ds
        self.step()
        self._last_ds=(-1 if self._last_ds<0 else 1)*min(abs(self._last_ds),abs(ds_backup))

    def locate_bifurcation(self,pitchfork:bool=False):
        self._status("BIFURCATION FINDING"+(" (PITCHFORK)" if pitchfork else ""))
        self.problem.solve_eigenproblem(self.neigen,self.shift)
        self.problem.activate_bifurcation_tracking(self._paramname,"pitchfork" if pitchfork else None)
        self.problem.solve(max_newton_iterations=20)
        self._add_current_state()
        self._update_tangents()
        self.problem.deactivate_bifurcation_tracking()
        self.reorder_branch_upon_point_insertion(self._get_current_branch(),self._get_current_point())

    def locate_bifurcation_or_switch(self):
        """What ``b`` has always done: at a bifurcation, switch branch; otherwise, go find one."""
        if self.current_point is not None and self.current_point.eig_value_Re==0:
            self.branch_switch()
        else:
            try:
                self.locate_bifurcation()
            except Exception as e:
                # Bifurcation tracking must be switched off again, otherwise the augmented system
                # stays active and every subsequent solve operates on the wrong problem.
                self.problem.deactivate_bifurcation_tracking()
                self.log("Bifurcation finding failed:",e)

    def set_arclength_scaling(self,scale:bool):
        self.scale_arc_length=scale
        self.problem.set_arc_length_parameter(scale_arc_length=scale)
        self.log("Scale arclength is set to",scale)

    # ------------------------------------------------------------------ start / persistence

    def start(self,init_ds,initial_max_newton_iterations=10):
        """Solve the initial state, discover the observables and reload an existing diagram."""
        self._last_ds=init_ds
        if not self.problem.is_initialised():
            self.problem.initialise()
        if self._paramname is None:
            avail_params=list(self.problem.get_global_parameter_names())
            if len(avail_params)!=1:
                raise RuntimeError("Please create the BifurcationGUI with a parameter name, unless you have a problem with a single global parameter only")
            self._paramname=avail_params[0]
        elif not isinstance(self._paramname,str):
            self._paramname=self._paramname.get_name()
        datadir=self.problem.get_output_directory(self.data_subdir)
        Path(datadir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(datadir,"_states")).mkdir(parents=True, exist_ok=True)

        try:
            self.problem.solve(max_newton_iterations=initial_max_newton_iterations)
        except Exception as e:
            raise RuntimeError("Make sure the problem starts where it has a stationary solution") from e
        self.problem.solve_eigenproblem(self.neigen,self.shift)
        self._avail_observables=[k for k in self.evaluate_observables().keys()]
        self._current_observable=self._avail_observables[0]
        self._add_current_state()

    def has_saved_state(self)->bool:
        outdir=self.problem.get_output_directory(self.data_subdir)
        return os.path.isfile(os.path.join(outdir,"state.json"))

    def save_all(self):
        outdir=self.problem.get_output_directory(self.data_subdir)

        cur_branch=self._get_current_branch()
        fullinfo={}
        fullinfo["branches"]=[b.to_state_dict() for b in self.branches]
        fullinfo["demo_video_step"]=self._demo_video_step
        fullinfo["xlim"]=list(self.view.get_xlim())
        fullinfo["ylim"]=list(self.view.get_ylim())
        fullinfo["xscale"]=self.view.get_xscale()
        fullinfo["yscale"]=self.view.get_yscale()

        fullinfo["statestep"]=self._state_step
        fullinfo["currentbranch"]=self.branches.index(cur_branch)
        fullinfo["currentpoint"]=cur_branch.index(self._get_current_point())
        if self.selected_branch is not None:
            fullinfo["selectedbranch"]=self.branches.index(self.selected_branch)
            if self.selected_point is not None:
                fullinfo["selectedpoint"]=self.selected_branch.index(self.selected_point)
        fullinfo["lastds"]=self._last_ds
        fullinfo["current_observable"]=self._current_observable
        fullinfo["mode"]=self._mode
        fullinfo["interpolated_splines"]=self.interpolated_splines
        with open(os.path.join(outdir,"state.json"), 'w') as f:
            json.dump(fullinfo, f, indent=4)

    def load_all(self,apply_view:Callable[[dict],None] | None=None):
        """Restore a diagram written by :py:meth:`save_all`.

        ``apply_view`` receives the raw state dict so the plot can restore limits and scales before
        anything that depends on them (the point-insertion metric) runs.
        """
        outdir=self.problem.get_output_directory(self.data_subdir)
        fname=os.path.join(outdir,"state.json")
        with open(fname) as f:
            fullinfo=json.load(f)

        self.branches=[BifurcationGUISolutionBranch.from_dict(b) for b in fullinfo["branches"]]
        if apply_view is not None:
            apply_view(fullinfo)

        if "demo_video_step" in fullinfo:
            self._demo_video_step=fullinfo["demo_video_step"]

        self._state_step=fullinfo["statestep"]
        # Note: indices are explicitly converted to int (rather than indexing with the raw Any
        # from the JSON dict) since otherwise UserList.__getitem__'s int/slice overload becomes
        # ambiguous for an Any-typed index, which makes pyright widen the assignment back to the
        # full Optional declared type instead of narrowing it to the non-None branch/point type.
        self.current_branch=self.branches[int(fullinfo["currentbranch"])]
        self.current_point=self.current_branch[int(fullinfo["currentpoint"])]
        if "selectedbranch" in fullinfo:
            self.selected_branch=self.branches[int(fullinfo["selectedbranch"])]
            if "selectedpoint" in fullinfo:
                self.selected_point=self.selected_branch[int(fullinfo["selectedpoint"])]
        self._last_ds=fullinfo["lastds"]
        self._current_observable=fullinfo["current_observable"]
        self._mode=fullinfo["mode"]
        self.interpolated_splines=fullinfo.get("interpolated_splines",self.interpolated_splines)
        self._status("LOADING")
        self.load_pt(self.current_point)
        self._update_tangents()
        self._status(None)
        self._changed()

    def observable_range(self,obs:str)->tuple[float,float] | None:
        """Min/max of an observable over the whole diagram, for autoscaling on a switch."""
        ymin=1e30
        ymax=-1e30
        found=False
        for b in self.branches:
            for p in b:
                if obs not in p.obs_values:
                    continue
                y=p.obs_values[obs]
                ymin=min(ymin,y)
                ymax=max(ymax,y)
                found=True
        return (ymin,ymax) if found else None

    def data_range(self)->tuple[tuple[float,float],tuple[float,float]] | None:
        """Bounding box of the whole diagram in (parameter, current observable)."""
        obs=self._current_observable
        if obs is None:
            return None
        xs=[];ys=[]
        for b in self.branches:
            for p in b:
                if obs not in p.obs_values:
                    continue
                xs.append(p.param_value)
                ys.append(p.obs_values[obs])
        if not xs:
            return None
        return (min(xs),max(xs)),(min(ys),max(ys))


from ...typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
