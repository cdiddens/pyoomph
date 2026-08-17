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
from ...expressions.units import unit_to_string

from .model import (eigen_settings, BifurcationGUISolutionPoint, BifurcationGUISolutionBranch, AxisSpec,
                    AXIS_OBSERVABLE, AXIS_PARAMETER, as_axis, observable_axis, parameter_axis,
                    same_parameter_value, STABILITY_EIGEN, STABILITY_INFERRED, STABILITY_UNKNOWN)

from typing import Protocol
from pathlib import Path
import numpy
import os
import json
import glob
import shutil


#: Bifurcation types that can be tracked, hence also continued as a locus. Mirrors the Literal that
#: Problem.activate_bifurcation_tracking accepts; kept as data so the GUI can offer and validate it.
TRACKABLE_BIFURCATION_TYPES=("fold","hopf","pitchfork","azimuthal","cartesian_normal_mode",
                             "real","complex","normal_mode")


def _as_tracking_type(value:str | None):
    """Narrow a recorded/user-supplied type string to what activate_bifurcation_tracking accepts."""
    if value is None or value=="":
        return None
    if value not in TRACKABLE_BIFURCATION_TYPES:
        raise ValueError("'"+str(value)+"' is not a trackable bifurcation type; expected one of "+
                         ", ".join(TRACKABLE_BIFURCATION_TYPES))
    return cast(Any,value)


#: Version stamped into state.json. 1 is implicit for files written before the slice (which
#: parameter was continued, and what the others were held at) was recorded at all.
STATE_VERSION=2


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


def _eigen_columns(unit:str)->"list[str]":
    """Header names for the two eigenvalue columns, carrying the rate unit when there is one."""
    suffix=" ["+unit+"]" if unit else ""
    return ["ReEigen"+suffix,"ImEigen"+suffix]


def _si_value(value)->float:
    """The plain number of a possibly dimensional value, in SI base units.

    ``float()`` on an expression that still carries a unit throws, and in a problem with a spatial
    scaling EVERY integral observable carries one, because the integration measure dx has length -
    which is why the GUI could not even start on a dimensional problem.
    """
    if isinstance(value,_pyoomph.Expression):
        factor,_unit,_rest,success=_pyoomph.GiNaC_collect_units(value)
        if not success:
            raise ValueError("Cannot separate the unit from "+str(value))
        return float(factor)
    return float(value)


def _unit_and_multiplier(value)->"tuple[str,float]":
    """``(unit string, SI -> that unit multiplier)`` for a dimensional value, e.g. ``("mm", 1000.0)``.

    The prefix follows the magnitude of the value handed in, which is why this is done once with a
    representative value rather than per point.
    """
    if not isinstance(value,_pyoomph.Expression):
        return "",1.0
    unit,_factor,mult=unit_to_string(value)   #type:ignore[misc]
    return str(unit),float(mult)


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
        # A separate shift for the eigensolves taken ON A LOCUS, because self.shift's default of 0 is
        # the one value that cannot work there: the tracker has put an eigenvalue exactly at 0 (fold,
        # pitchfork) or +-i*omega (Hopf, azimuthal), which is precisely where a zero shift asks the
        # shift-invert transform to factorise. Set to None to skip the locus eigensolve entirely and
        # keep the historical single synthetic value per locus point.
        self.locus_eigen_shift:float | complex | None=0.1
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
        #: Explicit axis choices. None means "follow the default": the continued parameter on x and
        #: the current observable on y, which is what an ordinary bifurcation diagram shows.
        self._x_axis:"AxisSpec | None"=None
        self._y_axis:"AxisSpec | None"=None
        self._observable_funcs:dict[str,Callable[...,float]] | None=None
        #: Memo for one observable sweep, see :py:meth:`_extremum`; keyed by (domain, name, sign).
        self._extremum_cache:dict[tuple[str,str,int],tuple[float,list[float]]]={}
        #: Observable names that came from an ExtremumObservables and already carry their own tag.
        self._extremum_axes:set[str]=set()
        #: Unit of each observable ("mm", "1/s", "" when dimensionless), and the multiplier taking the
        #: SI value into it. Resolved once, see :py:meth:`_resolve_observable_units`.
        self._observable_units:dict[str,str]={}
        self._observable_mults:dict[str,float]={}
        #: Callables returning one DIMENSIONAL value per observable, used only to read off the unit.
        self._observable_unit_probes:dict[str,Callable[[],Any]]={}
        #: Eigenvalues are stored as physical rates; see :py:meth:`_resolve_eigen_unit`.
        self._eigen_mult=1.0
        self._eigen_unit=""
        #: Normal modes to solve alongside the base state: azimuthal m (integers) or Cartesian k
        #: (reals), depending on what the problem was set up for. Empty means base state only.
        self.normal_modes:list[float]=[]
        #: Solve them at every continuation point too. Off by default because a scan of N modes is N
        #: extra eigensolves per point, and eigensolves already dominate a sweep - measured on the
        #: azimuthal probe, three modes cost 2.9x one. Off, the modes are filled in on demand by
        #: compute_spectrum()/compute_spectrum_for_branch().
        self.compute_modes_during_sweep=False
        #: Half-widths of the stripe scanned for eigenvalues: |Re| < stripe_re and |Im| < stripe_im.
        #: Bounded on purpose - the contour method integrates around the region, so an unbounded stripe
        #: cannot be asked for and the imaginary extent decides which frequencies are looked at.
        self.stripe_re=0.5
        self.stripe_im=20.0
        #: Upper bound on how many eigenvalues a scan may return. SLEPc caps a contour solve by the
        #: requested count, so passing the ordinary neigen made a scan return neigen eigenvalues and
        #: quietly miss the rest - measured: a region holding 4 returned 2 when asked for 2. When a scan
        #: comes back with exactly this many, the region probably holds more and this should be raised.
        self.stripe_max=20
        #: Merge a scan into the point's existing spectrum instead of replacing it, so a shift-invert
        #: spectrum keeps its eigenvalues and only gains the ones it could not see.
        self.stripe_merge=True
        #: When to remesh/adapt during continuation: "off" (the historical behaviour), "when_needed"
        #: (ask remesh_handler_during_continuation, which checks the problem's own remeshing_necessary()
        #: and does nothing otherwise) or "every_n" (force one every adapt_every_n steps).
        self.adapt_policy="off"
        self.adapt_every_n=5
        #: Re-solve after the remesh, i.e. remesh_handler_during_continuation(resolve=...).
        self.adapt_resolve=True
        #: Number of adaptation passes to run after a step, on top of the remesh handler. These are two
        #: different things: the handler REMESHES (it needs a remesher, e.g. RemeshWhen, and does
        #: nothing without one), while this refines/unrefines the existing mesh from the problem's
        #: SpatialErrorEstimators. Most problems have the latter and not the former.
        self.adapt_spatial=1
        #: Also refine towards an eigenfunction afterwards, via Problem.refine_eigenfunction.
        self.adapt_to_eigenfunction=False
        self.adapt_eigenindex=0
        self._steps_since_adapt=0
        #: Whether the drawn stability counts every computed mode or the base state alone. An
        #: axisymmetric state can be stable to m=0 and unstable to m=1 - a polygonal hydraulic jump is
        #: exactly that - so which of the two is being shown has to be sayable.
        self.count_normal_modes_in_stability=True
        self._mode="al"
        self._move_point=False
        self.interpolated_splines=False
        self.scale_arc_length=True
        #: D, the fraction of the arclength the continued parameter is given while the scaling is on:
        #: theta^2 is retuned after every step so that (dparameter/ds)^2 == D. oomph's own default.
        self.arclength_proportion=0.5
        self._state_step=0
        self._abort_requested=False
        #: Write all observable values to the output files
        self.output_all_observables=False
        self._initial_view=None
        #: Compute the normal form at each located bifurcation, which is what names it fold /
        #: transcritical / pitchfork and what branch switching needs. On by default: it costs about as
        #: much as locating the bifurcation did (~1 s at 7600 dofs) and runs once per bifurcation, not
        #: per continuation step. Set False on a very large problem where that matters; branch switching
        #: then computes it on demand for the one point it needs.
        self.classify_bifurcations=True
        #: Continue without solving an eigenproblem at every point, spotting bifurcations from test
        #: functions instead. See dev_docs/quick_continuation.md for what it can and cannot see.
        self.quick_mode=False
        #: "auto" watches both the determinant sign and dparameter/ds; "folds_only" watches dparam/ds
        #: alone, which needs no solver support but cannot see a pitchfork or a transcritical point.
        self.quick_mode_detector="auto"
        #: Draw stability that was propagated from a measured point rather than measured here. Turning
        #: it off shows those segments as unknown, since a Hopf crossing would make it wrong.
        self.trust_inferred_stability=True

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

    # ------------------------------------------------------------------ plot axes

    @property
    def x_axis(self)->"AxisSpec":
        """What the horizontal axis shows; by default the parameter being continued."""
        if self._x_axis is not None:
            return self._x_axis
        return parameter_axis(self._get_paramname_str())

    @property
    def y_axis(self)->"AxisSpec":
        """What the vertical axis shows; by default the selected observable."""
        if self._y_axis is not None:
            return self._y_axis
        return observable_axis(self._get_current_observable())

    def set_x_axis(self,spec:"AxisSpec | str | None"):
        self._x_axis=as_axis(spec) if spec is not None else None
        self._changed()

    def set_y_axis(self,spec:"AxisSpec | str | None"):
        spec=as_axis(spec) if spec is not None else None
        # An observable on y is also "the current observable": the tangent bookkeeping, the saved
        # state and the facade attribute all key off that name, so the two must not drift apart.
        if spec is not None and spec[0]==AXIS_OBSERVABLE:
            if spec[1] not in self._avail_observables:
                raise ValueError("Unknown observable "+str(spec[1]))
            self._current_observable=spec[1]
            self._y_axis=None
        else:
            self._y_axis=spec
        self._changed()

    def available_axes(self)->"list[AxisSpec]":
        """Everything that can go on an axis: every global parameter, then every observable."""
        res:list[AxisSpec]=[parameter_axis(n) for n in self.all_parameter_names()]
        res+=[observable_axis(n) for n in self._avail_observables]
        return res

    def axis_label(self,spec:"AxisSpec")->str:
        """Label for the figure: the name, plus the unit when the quantity has one.

        The parameter/observable distinction only has to be spelled out where a choice is being made
        (the menus and combos), not on the plot. Global parameters are plain numbers in pyoomph, so only
        observables can carry a unit.
        """
        kind,name=as_axis(spec)
        unit=self.observable_unit(name) if kind==AXIS_OBSERVABLE else ""
        return name+" ["+unit+"]" if unit else name

    def branch_can_be_plotted(self,branch:"BifurcationGUISolutionBranch")->bool:
        """False when a branch has no value to show on one of the current axes.

        Happens for a branch continued in a different parameter, and for legacy points that never
        recorded the parameter now on an axis. Such a branch is skipped rather than drawn wrong.
        """
        if len(branch)==0:
            return False
        try:
            branch[0].get_coordinate(self.y_axis,xspec=self.x_axis)
        except KeyError:
            return False
        return True

    # ------------------------------------------------------------------ observables

    # By default, we allow to access all integral observables (not beginning with _) and all ODE dofs
    def _extremum(self,domname:str,name:str,sign:int)->"tuple[float,list[float]]":
        """Value and position of one extremum, computed at most once per observable sweep.

        An ``ExtremumObservables`` entry yields up to eight axis choices (min/max times value and up to
        three coordinates), and each of those is an independent lambda in the observable table - so
        without this memo one sweep of the mesh would be paid for per *choice* instead of per extremum.
        """
        key=(domname,name,sign)
        if key not in self._extremum_cache:
            mesh=self.problem.get_mesh(domname)
            evaluate=mesh.evaluate_maximum if sign>0 else mesh.evaluate_minimum
            val,pos=evaluate(name,as_float=True,return_x=True) #type:ignore[misc]
            self._extremum_cache[key]=(float(val),[float(p) for p in pos]) #type:ignore[arg-type]
        return self._extremum_cache[key]

    def evaluate_observables(self)->dict[str,float]:
        if self._observable_funcs is None:
            obs:dict[str,Callable[...,float]]={}
            self._observable_unit_probes={}
            def recursive_add_spatial_domains(eqtree:EquationTree):
                if eqtree._equations is not None and eqtree.get_equations()._is_ode()==False:
                    ifuncs_list=eqtree.get_mesh().list_integral_functions()
                    deps = eqtree.get_code_gen()._dependent_integral_funcs
                    ifuncs: set[str] = set(ifuncs_list)
                    ifuncs.update(deps.keys())
                    bn=eqtree.get_full_path().lstrip("/")
                    # Sorted, not in set order: a set of strings iterates differently from process to
                    # process (string hashing is salted), and evaluate_observable is collective on a
                    # distributed mesh, so ranks disagreeing on the order would deadlock. It also keeps
                    # which observable the diagram opens on from changing between runs.
                    for valn in sorted(ifuncs):
                        if not valn.startswith("_"):
                            key=bn+"/"+valn
                            obs[key]=lambda domname=bn,valn=valn: _si_value(self.problem.get_mesh(domname).evaluate_observable(valn))
                            self._observable_unit_probes[key]=lambda domname=bn,valn=valn: self.problem.get_mesh(domname).evaluate_observable(valn)
                    # ExtremumObservables are not integral functions, so they need their own listing.
                    # Each becomes several axis choices: the minimum and the maximum, and where each of
                    # them sits - "h_extreme  [max, val]", "h_extreme  [max, x]", ... The position is
                    # frequently the interesting one, e.g. to watch a pattern drift as a parameter moves.
                    # The tag is part of the name (two spaces, so it lines up with the "[obs]"/"[param]"
                    # tag the axis menus add) and _extremum_axes tells those menus not to add one.
                    coords=["x","y","z"][:max(1,eqtree.get_code_gen().get_nodal_dimension())]
                    for exname in eqtree.get_code_gen()._list_extremum_functions():
                        if exname.startswith("_"):
                            continue
                        for sign,tag in ((1,"max"),(-1,"min")):
                            def add(what:str,func:Callable[...,float],unit_probe:Callable[[],Any]):
                                key=bn+"/"+exname+"  ["+tag+", "+what+"]"
                                obs[key]=func
                                self._observable_unit_probes[key]=unit_probe
                                self._extremum_axes.add(key)
                            add("val",lambda domname=bn,exname=exname,sign=sign: self._extremum(domname,exname,sign)[0],
                                lambda domname=bn,exname=exname,sign=sign:
                                    self.problem.get_mesh(domname).get_code_gen()._get_extremum_expression_unit_factor(exname)
                                    *self._extremum(domname,exname,sign)[0])
                            for i,coord in enumerate(coords):
                                add(coord,lambda domname=bn,exname=exname,sign=sign,i=i: self._extremum(domname,exname,sign)[1][i],
                                    lambda domname=bn,exname=exname,sign=sign,i=i:
                                        self.problem.get_scaling("spatial")*self._extremum(domname,exname,sign)[1][i]
                                        /_si_value(self.problem.get_scaling("spatial")))
                for child in eqtree.get_children().values():
                    recursive_add_spatial_domains(child)

            for name,eqtree in self.problem._equation_system.get_children().items():
                if eqtree.get_equations()._is_ode()==True:
                    ode=self.problem.get_ode(name)
                    _vals, inds = ode.get_element()._ode_elem_to_numpy()
                    for valn in inds.keys():
                        if not valn.startswith("_"):
                            key=name+"/"+valn
                            obs[key]=lambda domname=name,valn=valn: self.problem.get_ode(domname).get_value(valn,dimensional=True,as_float=True)
                            self._observable_unit_probes[key]=lambda domname=name,valn=valn: self.problem.get_ode(domname).get_value(valn,dimensional=True)
            recursive_add_spatial_domains(self.problem._equation_system)
            if len(obs)==0:
                raise RuntimeError("Could not identify an observable. Add ODEs or IntegralObservables to find them")
            self._observable_funcs=obs.copy()
            self._resolve_observable_units()
        self._extremum_cache={}
        mults=self._observable_mults
        return {n:func()*mults.get(n,1.0) for n,func in self._observable_funcs.items()}

    def _resolve_observable_units(self):
        """Fix each observable's unit ONCE, from the values at the starting point.

        Once, not per point: an estimated SI prefix follows the magnitude, so re-deriving it every step
        would silently switch a branch from mm to m half way along and the stored numbers would stop
        meaning the same thing. The multiplier is applied to every later evaluation instead.

        A value that happens to be exactly zero here carries no unit at all - GiNaC simplifies 0*meter
        to 0 - so such an observable stays unlabelled rather than being guessed at.
        """
        self._observable_units={}
        self._observable_mults={}
        self._extremum_cache={}
        assert self._observable_funcs is not None
        for name in self._observable_funcs:
            unit,mult="",1.0
            probe=self._observable_unit_probes.get(name)
            if probe is not None:
                try:
                    unit,mult=_unit_and_multiplier(probe())
                except Exception as e:
                    self.log("Could not determine the unit of",name+":",repr(e))
            self._observable_units[name]=unit
            self._observable_mults[name]=mult

    def _resolve_eigen_unit(self):
        """Express eigenvalues as rates in the problem's time unit rather than per time SCALE.

        An eigenvalue out of the eigensolver is nondimensional, i.e. in units of 1/(temporal scale):
        with ``set_scaling(temporal=10*second)`` a true rate of -0.5 1/s is reported as -5. Dividing by
        the scale turns it into a rate in 1/s.

        Deliberately NOT prefix-estimated from a representative value, the way observables are: the
        spectrum spans decades, so a prefix chosen from one eigenvalue would misrepresent the rest. The
        unit is whatever the temporal scale is stated in, and stays put. A problem without scalings has
        a dimensionless temporal scale of 1, so this is the identity and nothing changes.
        """
        self._eigen_mult=1.0
        self._eigen_unit=""
        try:
            ts=self.problem.get_scaling("temporal")
            ts_si=_si_value(ts)
            if ts_si!=0:
                self._eigen_mult=1.0/ts_si
            if isinstance(ts,_pyoomph.Expression):
                unit=str(unit_to_string(ts,estimate_prefix=False))
                if unit:
                    self._eigen_unit="1/"+unit
        except Exception as e:
            self.log("Could not determine the unit of the eigenvalues:",repr(e))

    def _phys_eig(self,value):
        """One eigenvalue, or a sequence of them, as a physical rate."""
        if value is None:
            return None
        if isinstance(value,(list,tuple,numpy.ndarray)):
            return [complex(v)*self._eigen_mult for v in value]
        return complex(value)*self._eigen_mult

    @property
    def eigen_unit(self)->str:
        """Unit the recorded eigenvalues are in, e.g. "1/s"; empty for a nondimensional problem."""
        return self._eigen_unit

    def observable_unit(self,name:str)->str:
        """Unit string of an observable, e.g. "mm"; empty when it is dimensionless or unknown."""
        return self._observable_units.get(name,"")

    # ------------------------------------------------------------------ diagram bookkeeping

    def all_parameter_names(self)->list[str]:
        """Every global parameter of the problem, in a stable order.

        Only complete once the problem has been initialised - the parameters are created while
        define_problem() runs.
        """
        return sorted(self.problem.get_global_parameter_names())

    def current_parameter_values(self)->dict[str,float]:
        return {n:float(self.problem.get_global_parameter(n).value) for n in self.all_parameter_names()}

    def describe_current_slice(self)->str:
        """What the current branch holds fixed, e.g. "b = 0.3" - for the status bar and the plot."""
        if self.current_branch is None:
            return ""
        return self.current_branch.describe_slice()

    def slice_groups(self)->"dict[tuple,list[BifurcationGUISolutionBranch]]":
        """Branches grouped by the slice of parameter space they live in, preserving order."""
        groups:dict[tuple,list[BifurcationGUISolutionBranch]]={}
        for b in self.branches:
            groups.setdefault(b.slice_key(),[]).append(b)
        return groups

    def _new_branch(self,*,kind:str="solution",tracked_parameter:str | None=None,
                    bifurcation_type:str | None=None)->BifurcationGUISolutionBranch:
        """Open a branch, stamped with the slice it is about to be computed in."""
        branch=BifurcationGUISolutionBranch(kind=kind,
                                            continuation_parameter=self._paramname if isinstance(self._paramname,str) else None,
                                            tracked_parameter=tracked_parameter,
                                            bifurcation_type=bifurcation_type)
        self.branches.append(branch)
        self.current_branch=branch
        self.selected_branch=branch
        return branch

    # ------------------------------------------------------------------ slices

    def branch_is_on_current_slice(self,branch:"BifurcationGUISolutionBranch")->bool:
        """Whether a branch holds the same parameters at the same values as the current one.

        A diagram is only a valid section of parameter space for one such setting, so branches that
        answer False here belong to a different physical result and must not be drawn as though they
        were part of this one. Branches whose slice was never recorded (a pre-slice state file) are
        not accused of being elsewhere.
        """
        cur=self.current_branch
        if cur is None or branch is cur:
            return True
        if not (branch.slice_is_known() and cur.slice_is_known()):
            return True
        mine=branch.fixed_parameters()
        theirs=cur.fixed_parameters()
        if set(mine)!=set(theirs):
            return False
        return all(same_parameter_value(mine[n],theirs[n]) for n in mine)

    def set_continuation_parameter(self,name:str):
        """Continue in a different parameter from here on.

        This starts a new diagram, not a continuation of the old one: the existing branches are
        sections at a fixed value of the parameter now being varied, so they stay as they are and a
        new branch is opened at the current solution.
        """
        if name not in self.all_parameter_names():
            raise ValueError("'"+str(name)+"' is not a global parameter of this problem")
        if name==self._paramname:
            return False
        old=self._paramname
        cp=self.current_point
        # A fold is a fold of the Jacobian, not of one particular parameter, so continuing in ANY
        # parameter from exactly there has no regular tangent to start from and the first step fails
        # (see dev_docs/bifurcation_loci.md). Say so up front rather than let the solver report it.
        if cp is not None and cp.eig_value_Re==0:
            self.log("Warning: the current point is a bifurcation. Continuing from exactly there "
                     "usually fails - step back onto a regular point of the branch first.")
        self._paramname=name
        # A tangent is (dparameter, dobservable) - it says nothing about a different parameter.
        self._tangs={}
        self.problem.reset_arc_length_parameters()
        self._x_axis=None    # follow the new parameter unless the user pinned an axis
        self._new_branch()
        self._add_current_state(eig_value=None if cp is None else (cp.eig_value_Re+1j*cp.eig_value_Im),
                                eig_values=None if cp is None else list(cp.eig_values))
        self.log("Now continuing in '"+name+"' instead of '"+str(old)+"'; the previous branches stay as the "+
                 str(old)+" section at "+(self.describe_current_slice() or "their own values"))
        self._changed()
        return True

    def set_fixed_parameter(self,name:str,value:float):
        """Move a parameter that is being held fixed, which starts a new slice.

        Uses go_to_param, i.e. it continues there rather than jumping, so a large move still lands on
        a solution.
        """
        if name==self._paramname:
            raise ValueError("'"+name+"' is the continuation parameter; step in it instead of setting it")
        if name not in self.all_parameter_names():
            raise ValueError("'"+str(name)+"' is not a global parameter of this problem")
        self._status("MOVING "+name)
        # The dict form, not **kwargs: go_to_param has keyword options of its own (reset_pars,
        # epsilon, max_step, ...) and a parameter that happened to share one of those names would be
        # swallowed as an option instead of being moved.
        self.problem.go_to_param({name:float(value)})
        self.problem.solve_eigenproblem(self.neigen,self.shift)
        self._tangs={}
        self.problem.reset_arc_length_parameters()
        self._new_branch()
        self._add_current_state()
        self._update_tangents()
        self.log("Moved "+name+" to",value,"- this is a new slice, so a new branch was started")
        self._changed()

    # ------------------------------------------------------------------ bifurcation loci

    #: Fraction of the parameter's own magnitude used as the first offset when stepping off a fold.
    leave_locus_offset=0.05

    def on_locus(self)->bool:
        return self.current_branch is not None and self.current_branch.kind=="locus"

    def _branch_of(self,pt)->"BifurcationGUISolutionBranch | None":
        for b in self.branches:
            if pt in b:
                return b
        return None

    def _sync_tracking_to(self,branch:"BifurcationGUISolutionBranch | None"):
        """Make the problem's bifurcation tracking agree with the branch we are on.

        This is the invariant the whole feature rests on: continuing an ordinary branch with a stale
        tracker active, or a locus with none, both silently compute something else entirely. Only
        this method and the branch-opening calls are allowed to change the tracking state.
        """
        want_locus=branch is not None and branch.kind=="locus"
        active=self.problem.get_bifurcation_tracking_mode()!=""
        if want_locus:
            assert branch is not None
            if branch.tracked_parameter is None:
                raise RuntimeError("Locus branch has no tracked parameter recorded")
            # Take the guess from the handler BEFORE deactivating, when there is one. Since a locus
            # step now also solves the base state's eigenproblem to record its spectrum,
            # get_last_eigenvectors()[0] is whatever that secondary solve returned, not the tracked
            # critical vector -- reactivating from it would restart the tracker on the wrong mode.
            guess=None
            if active:
                tracked=numpy.array(self.problem._get_bifurcation_eigenvector(),dtype=numpy.complex128) #type:ignore
                if len(tracked)>0:
                    guess=tracked
                self.problem.deactivate_bifurcation_tracking()
            if guess is None:
                evs=self.problem.get_last_eigenvectors()
                guess=evs[0] if evs is not None and len(evs)>0 else None
            self.problem.activate_bifurcation_tracking(branch.tracked_parameter,
                                                       _as_tracking_type(branch.bifurcation_type),
                                                       eigenvector=guess)
            self.problem.reset_arc_length_parameters()
        elif active:
            self.problem.deactivate_bifurcation_tracking()
            self.problem.reset_arc_length_parameters()

    def start_locus(self,tracked:str,continue_in:str,bifurcation_type:str | None=None):
        """From a bifurcation, follow it through parameter space.

        ``tracked`` is adjusted to keep the bifurcation condition satisfied while ``continue_in`` is
        the one being stepped, which traces the locus of the bifurcation in the (continue_in, tracked)
        plane - the fold curve Bo_c(V) of the hanging-droplet tutorial, for instance.
        """
        cp=self.current_point
        if cp is None or cp.eig_value_Re!=0:
            raise RuntimeError("Continuing a bifurcation has to start AT one. Locate a bifurcation first.")
        names=self.all_parameter_names()
        for n in (tracked,continue_in):
            if n not in names:
                raise ValueError("'"+str(n)+"' is not a global parameter of this problem")
        if tracked==continue_in:
            raise ValueError("The tracked and the continued parameter must differ - "
                             "one is adjusted to hold the bifurcation, the other is stepped")
        if self.problem.is_distributed():
            raise RuntimeError("Continuing a bifurcation needs history dofs, which are not available "
                               "on a distributed (--distribute) problem. Locating one does work there.")

        self._status("STARTING BIFURCATION LOCUS")
        evs=self.problem.get_last_eigenvectors()
        guess=evs[0] if evs is not None and len(evs)>0 else None
        self.problem.activate_bifurcation_tracking(tracked,_as_tracking_type(bifurcation_type),eigenvector=guess)
        mode=self.problem.get_bifurcation_tracking_mode() or (bifurcation_type or "fold")
        self.problem.reset_arc_length_parameters()

        self._paramname=continue_in
        self._tangs={}
        self._new_branch(kind="locus",tracked_parameter=tracked,bifurcation_type=mode)
        # Both parameters vary along a locus, so that plane is what it should be drawn in.
        self._x_axis=parameter_axis(continue_in)
        self._y_axis=parameter_axis(tracked)
        self._add_locus_state()
        self.log("Following the "+mode+" in "+tracked+", continuing in "+continue_in)
        self._changed()

    def leave_locus(self,continue_in:str | None=None,offset:float | None=None):
        """Drop off the locus onto an ordinary branch through the bifurcation.

        The bifurcation is exactly where the plain Jacobian is singular, so there is no regular
        tangent for a continuation step to start from - seeding one produces Ds=nan and oomph-lib then
        retries for ever. Instead the parameter is offset off the bifurcation and a normal Newton
        solve is done from a guess displaced along the critical eigenvector. Only one side of a fold
        has solutions and which one is not known in advance, so the candidates are tried in turn.
        """
        branch=self.current_branch
        if branch is None or branch.kind!="locus":
            raise RuntimeError("Not on a bifurcation locus")
        tracked=branch.tracked_parameter
        assert tracked is not None
        target=continue_in if continue_in is not None else tracked
        if target not in self.all_parameter_names():
            raise ValueError("'"+str(target)+"' is not a global parameter of this problem")

        self._status("LEAVING THE LOCUS")
        evs=self.problem.get_last_eigenvectors()
        zeta=numpy.real(numpy.asarray(evs[0])) if evs is not None and len(evs)>0 else None
        if zeta is not None:
            nrm=numpy.linalg.norm(zeta)
            zeta=zeta/nrm if nrm>0 else None

        # After deactivating: while tracking is active the dof vector is the augmented one and a copy
        # taken then cannot be written back to the plain problem.
        self.problem.deactivate_bifurcation_tracking()
        self.problem.reset_arc_length_parameters()
        param=self.problem.get_global_parameter(target)
        at_bif=float(param.value)
        dofs_at_bif=numpy.array(self.problem.get_current_dofs()[0]).copy()

        base=offset if offset is not None else self.leave_locus_offset*max(abs(at_bif),1.0)
        landed=None
        for delta in (base,base/10,base*10):
            for sign in (1.0,-1.0):
                for hsign in ((1.0,-1.0) if zeta is not None else (0.0,)):
                    param.value=at_bif+sign*delta
                    guess=dofs_at_bif.copy()
                    if zeta is not None and hsign!=0.0:
                        guess=guess+hsign*numpy.sqrt(abs(delta))*zeta
                    self.problem.set_current_dofs(guess)
                    try:
                        self.problem.solve(max_newton_iterations=15)
                    except Exception:
                        continue
                    if numpy.linalg.norm(numpy.array(self.problem.get_current_dofs()[0])-dofs_at_bif)>1e-9:
                        landed=(sign*delta,hsign)
                        break
                if landed: break
            if landed: break
        if landed is None:
            param.value=at_bif
            self.problem.set_current_dofs(dofs_at_bif)
            self._sync_tracking_to(branch)
            raise RuntimeError("Could not step off the bifurcation. Try a different offset "
                               "(gui.controller.leave_locus_offset) or leave from another locus point.")

        self.problem.solve_eigenproblem(self.neigen,self.shift)
        self._paramname=target
        self._tangs={}
        self._x_axis=None
        self._y_axis=None
        self._new_branch()
        self._add_current_state()
        self.log("Left the locus at {:s} = {:.6g} (offset {:+.3g}); continuing in {:s}".format(
            target,at_bif,landed[0],target))
        self._changed()

    def new_branch_from_state(self,statefile):
        self._new_branch()
        self.problem.load_state(statefile,ignore_continuation_data=True,ignore_eigendata=True)
        self.problem.reset_arc_length_parameters()
        self.problem.solve_eigenproblem(self.neigen,self.shift)
        self._add_current_state()
        self._update_tangents()
        self._changed()

    def _add_locus_state(self):
        """Record the current point of a bifurcation locus, spectrum included.

        The critical eigenvalue is NOT re-solved: every point of a locus IS the bifurcation, and that
        is what the synthetic 0 + i*omega the tracker reports says. Re-solving would turn the exact
        zero into a small nonzero value and the point would stop reading as a bifurcation.

        The REST of the spectrum is solved for, though, since that is the only way to see a codim-2
        point coming (a second eigenvalue reaching zero, a pair crossing) - the base state's
        eigenproblem is available while tracking, see Problem.solve_eigenproblem. It needs a nonzero
        shift; see self.locus_eigen_shift, which is also how the eigensolve is switched off.

        Non-fatal on purpose: a shift-invert factorisation that fails should cost this point its
        spectrum, not abort a two-parameter sweep that may have been running for hours.
        """
        # omega comes out of the tracker in the same nondimensional units as any eigenvalue, so it
        # needs the same conversion as everything else that reaches a point.
        crit=self._phys_eig(0+1j*self.problem._get_bifurcation_omega()) #type:ignore
        spectrum=None
        if self.locus_eigen_shift:
            try:
                evs,_=self.problem.solve_eigenproblem(self.neigen,self.locus_eigen_shift,quiet=True)
                spectrum=self._phys_eig(list(evs))
            except Exception as e:
                self.log("Could not solve the eigenproblem on the locus ("+str(e).split("\n")[0]+"); recording the critical eigenvalue only")
        self._add_current_state(eig_value=crit,eig_values=spectrum)

    def _record_mode_observables(self,point):
        """Expose each mode's leading eigenvalue as an observable, so it can go on a plot axis.

        "eigen/max Re [m=1]" and its imaginary partner. Observables rather than a third kind of axis
        because the axis menus, the labels, the CSV export, value_of() and the state file already carry
        observables; a new kind would have to be taught to every one of them.

        Points where no scan ran simply do not have the key, and branch_can_be_plotted() already hides a
        branch that cannot supply the current axis.
        """
        if point.eig_modes is None or not point.eig_values:
            return
        kind=self.normal_mode_kind() or "m"
        seen:list[float]=[]
        for m in point.eig_modes:
            if m not in seen:
                seen.append(m)
        for m in seen:
            vals=point.eigenvalues_of_mode(m)
            if not vals:
                continue
            lead=max(vals,key=lambda v: numpy.real(v))
            tag="  [{:s}={:g}]".format(kind,m)
            for what,value in (("max Re",float(numpy.real(lead))),("max Im",float(numpy.imag(lead)))):
                name="eigen/"+what+tag
                point.obs_values[name]=value
                if name not in self._avail_observables:
                    self._avail_observables.append(name)
                    self._observable_units[name]=self._eigen_unit
                    self._observable_mults[name]=1.0
                    self._extremum_axes.add(name)   # the tag replaces the "[obs]" the menus would add

    def _last_solved_modes(self,n:int)->"list[float] | None":
        """The per-eigenvalue mode array the problem still holds, when it matches the spectrum length.

        None when only the base state was solved - which is what the problem reports then, since
        get_last_eigenmodes_m() is left at None unless a mode list was passed.
        """
        for raw in (self.problem.get_last_eigenmodes_m(),self.problem.get_last_eigenmodes_k()):
            if raw is not None and len(raw)==n:
                return [float(m) for m in raw]
        return None

    def _add_current_state(self,eig_value=None,eig_values=None,det_sign=None,dparam_ds=None,
                           measured:bool=True,eig_modes=None):
        """Record the problem's current state as a new point of the current branch.

        ``eig_value`` overrides the eigenvalue that would be read from the problem, which is what
        re-recording an existing solution under a new branch needs: re-solving the eigenproblem there
        would turn an exact zero into a small nonzero value and the point would stop being a
        bifurcation. ``eig_values`` likewise overrides the recorded spectrum; when only ``eig_value``
        is given the spectrum is just that one value, since whatever the problem still holds belongs
        to a different solve.
        """
        branch=self.current_branch if self.current_branch is not None else self._new_branch()
        if eig_value is None and measured and det_sign is None and dparam_ds is None:
            spectrum=self.problem.get_last_eigenvalues()
            if spectrum is None or len(spectrum)==0:
                # The eigensolve produced nothing - a shift-invert with shift 0 near a fold is the easy
                # way to get there. The point itself is perfectly good, so it is recorded WITHOUT a
                # spectrum rather than lost along with the rest of the sweep; its stability then reads as
                # unknown, exactly like a quick-mode point, and can be filled in later with
                # compute_spectrum().
                self.log("Warning: the eigensolve returned no eigenvalues here, so this point has no "
                         "spectrum. Its stability is unknown; a nonzero shift usually helps.")
                eig_values=[]
            else:
                spectrum=self._phys_eig(spectrum)
                eig_value=spectrum[0]
                if eig_values is None:
                    eig_values=list(spectrum)
                    eig_modes=self._last_solved_modes(len(eig_values))
        elif eig_value is not None and eig_values is None:
            eig_values=[eig_value]
        state_file=self.problem.get_output_directory(os.path.join(self.data_subdir,"_states","state_{:06d}.dump".format(self._state_step)))
        p=BifurcationGUISolutionPoint(self.get_bifurcation_parameter().value,self.evaluate_observables(),eig_value,state_file,self._state_step,
                                      param_values=self.current_parameter_values(),eig_values=eig_values,
                                      det_sign=det_sign,dparam_ds=dparam_ds,eig_modes=eig_modes,
                                      eig_settings=self.current_eigen_settings() if eig_values else None)
        self._record_mode_observables(p)
        # On a locus EVERY point has a zero real part, so classifying them all would run a normal-form
        # calculation per step for an answer already known: the tracked type.
        if p.eig_value_Re==0 and self.classify_bifurcations and branch.kind!="locus":
            p.bifurcation_info=self._classify_current_point()
        if p.stability_source==STABILITY_EIGEN and p.eig_values:
            p.unstable_count=p.measured_unstable_count()
        self.problem.save_state(state_file)
        self._state_step+=1
        self.current_point=p
        self.selected_point=None
        self.selected_branch=None
        branch.append(p)

    def load_pt(self,pt):
        # The dump restores EVERY global parameter, not just the continued one, so this can silently
        # move the user off the slice they were working on. Report it rather than let them believe the
        # value they last typed is still in force.
        before=self.current_parameter_values() if self.problem.is_initialised() else {}
        branch=self._branch_of(pt)
        # A point saved during tracked continuation has an arclength direction vector of the AUGMENTED
        # size, which cannot be read into the plain problem ("Mismatching size in the dof direction
        # vector"). The direction is re-established by _sync_tracking_to below anyway.
        on_locus=branch is not None and branch.kind=="locus"
        if self.problem.get_bifurcation_tracking_mode()!="":
            self.problem.deactivate_bifurcation_tracking()
        self.problem.load_state(pt.statefile,ignore_outstep=True,ignore_continuation_data=on_locus)
        after=self.current_parameter_values()
        moved={n:(before[n],after[n]) for n in after
               if n in before and n!=self._paramname and not same_parameter_value(before[n],after[n])}
        if moved:
            self.log("Loading this point moved "+", ".join(
                "{:s} {:.6g} -> {:.6g}".format(n,a,b) for n,(a,b) in sorted(moved.items())))
        self.current_point=pt
        if branch is not None:
            self.current_branch=branch
            if branch.continuation_parameter is not None:
                self._paramname=branch.continuation_parameter
        self._sync_tracking_to(branch)
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
            # Off-slice branches are drawn faint and are deliberately not selectable: loading one of
            # their points would move the fixed parameters underneath the user.
            if not (self.branch_is_on_current_slice(b) and self.branch_can_be_plotted(b)):
                continue
            for p in b:
                c=p.get_coordinate(self.y_axis,xspec=self.x_axis)
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

    @staticmethod
    def _slice_directory_names(slices:"dict[tuple,list[BifurcationGUISolutionBranch]]")->dict[tuple,str]:
        """One directory name per slice. Numbered rather than built from the values, because
        parameter values make poor path components (signs, dots, exponents)."""
        names={}
        for i,key in enumerate(slices):
            names[key]="slice{:02d}".format(i)
        return names

    def _export_header_suffix(self,branch:"BifurcationGUISolutionBranch")->str:
        """The part of a savetxt header that says what this curve is a section through.

        An exported curve that does not record the parameters held fixed cannot be interpreted later,
        let alone captioned, so it goes into every file - including the tag files.
        """
        parts=[]
        if branch.kind=="locus":
            parts.append("locus of {:s} bifurcations".format(branch.bifurcation_type or "unclassified"))
            if branch.tracked_parameter is not None:
                parts.append("tracked in "+branch.tracked_parameter)
        if branch.continuation_parameter is not None:
            parts.append("continued in "+branch.continuation_parameter)
        desc=branch.describe_slice()
        if desc:
            parts.append("fixed: "+desc)
        return ("\n"+"; ".join(parts)) if parts else ""

    def _export_axes(self,branch:"BifurcationGUISolutionBranch",obs_cols:"list[AxisSpec]"):
        """The (x, [y...]) a branch should be written out in.

        A locus leads with its two parameters, so the first two columns of the file are the curve the
        tutorials write by hand (``V``, ``Bo_c``), with the observables after them.
        """
        xspec=parameter_axis(branch.continuation_parameter) if branch.continuation_parameter else self.x_axis
        yspecs=list(obs_cols)
        if branch.kind=="locus" and branch.tracked_parameter is not None:
            yspecs=[parameter_axis(branch.tracked_parameter)]+yspecs
        return xspec,yspecs

    def output_curves(self):
        # output_all_observables writes one column per observable. It used to pass None down as the
        # observable name, which reached obs_values[None] and raised - so the flag never worked; the
        # y axis is now simply a list of specs, which the segmentation handles natively.
        if self.output_all_observables:
            obs_cols=[observable_axis(n) for n in self._avail_observables]
        else:
            obs_cols=[self.y_axis] if self.y_axis[0]==AXIS_OBSERVABLE else [observable_axis(self._get_current_observable())]
        ddir=self.problem.get_output_directory(self.data_subdir)
        odir=os.path.join(ddir,"output")

        Path(odir).mkdir(parents=True,exist_ok=True)
        # Now nested one level deeper (slice/branch/*.txt) than before, hence the two patterns.
        for pattern in ("branch*/*.txt","slice*/branch*/*.txt","*.dump"):
            for g in glob.glob(os.path.join(odir,pattern)):
                os.remove(g)

        slices=self.slice_groups()
        # One directory per slice of parameter space, since curves from different slices are
        # different physical results and must not land in one folder to be plotted together.
        slice_dirs=self._slice_directory_names(slices)
        for ib,b in enumerate(self.branches):
            bdir=os.path.join(odir,slice_dirs[b.slice_key()],"branch{:03d}".format(ib))
            Path(bdir).mkdir(parents=True,exist_ok=True)
            header_suffix=self._export_header_suffix(b)
            # Each branch is exported in ITS OWN natural coordinates, not in whatever the window
            # happens to be showing: a locus is the curve of its two parameters (V, Bo_c in the
            # hanging-droplet tutorial), and a solution branch is its own parameter versus the
            # observables. Exporting everything in the current view produced a locus labelled with
            # the parameter it does not even vary.
            export_x,export_y=self._export_axes(b,obs_cols)
            # The exported numbers are in the observable's own unit, so the header has to say which.
            column_names=[self.axis_label(sp) for sp in export_y]
            if self.interpolated_splines:
                smoothedsegs,stabs=b.smooth_branch_stab_list(export_y,100,xspec=export_x,
                                                             trust_inferred=self.trust_inferred_stability,
                                                             include_modes=self.count_normal_modes_in_stability)
            else:
                smoothedsegs,stabs=b.to_branch_stab_list(export_y,xspec=export_x,
                                                         trust_inferred=self.trust_inferred_stability,
                                                         include_modes=self.count_normal_modes_in_stability)
            istab=0
            iunstab=0
            for seg,stab in zip(smoothedsegs,stabs):
                if stab:
                    fn="smoothed_stable_{:03d}.txt".format(istab)
                    istab+=1
                else:
                    fn="smoothed_unstable_{:03d}.txt".format(iunstab)
                    iunstab+=1
                numpy.savetxt(os.path.join(bdir,fn),seg[:,:-1],header="\t".join([self.axis_label(export_x)]+column_names+_eigen_columns(self._eigen_unit))+header_suffix)
            nbif=0
            for p in b:
                if p.eig_value_Re==0:
                    fn="bifurcation_{:03d}.txt".format(nbif)
                    pc=p.get_coordinate(export_y,with_s=False,with_eigen=True,xspec=export_x)
                    numpy.savetxt(os.path.join(bdir,fn),numpy.array([pc],ndmin=2),header="\t".join([self.axis_label(export_x)]+column_names+_eigen_columns(self._eigen_unit))+header_suffix)
                    nbif+=1
                if p.tag>=0 and p.statefile is not None:
                    shutil.copy2(p.statefile, os.path.join(odir,"tag{:02d}.dump".format(p.tag)))
                    fn="tag_{:02d}.txt".format(p.tag)
                    pc=p.get_coordinate(export_y,with_s=False,xspec=export_x)
                    numpy.savetxt(os.path.join(odir,fn),numpy.array([pc],ndmin=2),header="\t".join([export_x[1]]+column_names)+self._export_header_suffix(b))
        self.log("Exported curves to",odir)
        return odir

    def output_tagged_points(self)->int:
        """Write each tagged point's FIELDS - plots, VTUs, whatever the problem outputs.

        The curve export only copies a tagged point's state dump, which preserves the solution but shows
        nothing. This loads each one and runs the problem's own output into its own directory,
        ``output/tag01/`` and so on.

        Follows the discipline PeriodicOrbit.output_orbit uses for the same trick (problem.py:280): the
        output directory, ``write_states`` and ``_output_step`` are all put back afterwards - without
        that, the redirected outputs would keep counting from wherever the diagram had got to, and every
        loaded point would write another state dump into the diagram's own store. The point that was
        current is restored too, since loading moves the problem.

        Its own command rather than part of the curve export: this reloads and re-outputs every tagged
        point, which on a real mesh with a plotter attached is not instant.
        """
        tagged=[p for b in self.branches for p in b if p.tag>=0 and p.statefile]
        if not tagged:
            self.log("No tagged points to output. Mark one with the number keys first.")
            return 0
        odir=os.path.join(self.problem.get_output_directory(self.data_subdir),"output")
        restore=self.current_point
        olddir=self.problem.get_output_directory()
        write_states=self.problem.write_states
        outstep=self.problem._output_step
        done=0
        try:
            self.problem.write_states=False
            for i,p in enumerate(sorted(tagged,key=lambda q: q.tag)):
                if self._abort_requested:
                    self._abort_requested=False
                    self.log("Tagged-point output aborted after {:d} of {:d}".format(done,len(tagged)))
                    break
                self._status("OUTPUT TAG {:d} ({:d}/{:d})".format(p.tag,i+1,len(tagged)))
                try:
                    self.load_pt(p)
                    # _change_output_directory is the whole redirection: each file output stores the
                    # new location RELATIVE to the problem's base directory and writes there. The base
                    # directory itself must NOT be moved, or that relative path is computed against a
                    # directory that no longer exists.
                    self.problem._change_output_directory(os.path.join(odir,"tag{:02d}".format(p.tag)))
                    self.problem._output_step=0
                    self.problem.output()
                    done+=1
                except Exception as e:
                    self.log("Could not output tag {:d}: {:s}".format(p.tag,repr(e)))
        finally:
            # Each restore stands on its own: an output kind that refuses to be redirected raises here
            # too, and letting that escape would leave write_states off and the output directory
            # pointing into a tag folder for the rest of the session.
            try:
                self.problem._change_output_directory(olddir)
            except Exception as e:
                self.log("Could not restore the output directory:",repr(e))
            self.problem.write_states=write_states
            self.problem._output_step=outstep
            if restore is not None:
                try:
                    self.load_pt(restore)
                except Exception as e:
                    self.log("Could not return to the previous point: "+repr(e))
        self.log("Wrote the fields of {:d} tagged point{:s} to {:s}".format(
            done,"" if done==1 else "s",odir))
        self._changed()
        return done

    # ------------------------------------------------------------------ tangents

    def _update_tangents(self):
        # On a locus the arclength derivative belongs to the augmented system and the finite-difference
        # probe below would perturb its eigenvector/parameter entries too. The plotted direction comes
        # from axis_tangent() instead, which needs no solver internals.
        if self.on_locus():
            self._tangs={}
            return
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

    # ------------------------------------------------------------------ quick mode

    def determinant_sign_supported(self)->bool:
        """Whether the configured linear solver can report a determinant sign at all.

        Asks the CLASS, not the current value: get_determinant_sign() also returns None when no
        factorisation exists yet, which at enable time is the normal state and must not be mistaken for
        "this solver cannot do it".
        """
        from ...solvers.generic import GenericLinearSystemSolver
        try:
            solver=self.problem.get_la_solver()
        except Exception:
            return False
        return type(solver).get_determinant_sign is not GenericLinearSystemSolver.get_determinant_sign

    def set_quick_mode(self,on:bool,detector:str | None=None):
        """Turn quick mode on or off, refusing "auto" on a solver that cannot report a determinant.

        Refusing rather than quietly falling back: a mode that silently stops seeing pitchforks and
        transcritical points is worse than one that will not start, and the message names the way out.
        """
        if detector is not None:
            if detector not in ("auto","folds_only"):
                raise ValueError("quick_mode_detector must be 'auto' or 'folds_only'")
            self.quick_mode_detector=detector
        if on and self.quick_mode_detector=="auto" and not self.determinant_sign_supported():
            solver=self.problem.get_la_solver()
            raise RuntimeError(
                "Quick mode cannot watch the determinant with the '"+str(getattr(solver,"idname","?"))+
                "' linear solver: it does not report a determinant sign.\n"
                "Use problem.set_linear_solver('superlu'), or 'petsc' with use_mumps(), to get folds AND "
                "branch points.\nOr set the detector to 'folds_only', which needs no solver support but "
                "sees only folds - a pitchfork or transcritical point will pass unnoticed.")
        self.quick_mode=bool(on)
        self.log("Quick mode "+("on" if on else "off")+
                 (" (detector: "+self.quick_mode_detector+")" if on else ""))
        self._changed()

    def _determinant_sign(self)->int | None:
        """The solver's determinant sign, but only when it belongs to the PLAIN Jacobian.

        While an augmented system is active - bifurcation tracking - the factorisation is of a bordered
        matrix whose determinant does not vanish at the bifurcation, so its sign says nothing about one.
        _get_n_unaugmented_dofs()==0 is oomph's "not augmented" sentinel.
        """
        if self.quick_mode_detector!="auto":
            return None
        try:
            if self.problem._get_n_unaugmented_dofs()!=0:
                return None
            return self.problem.get_la_solver().get_determinant_sign()
        except Exception:
            return None

    def _detect_bifurcation_between(self,previous,current):
        """Set current.detected_bifurcation when a test function changed across the pair.

        A fold reverses dparameter/ds AND flips the determinant; a branch point (pitchfork,
        transcritical) flips only the determinant, which is exactly why both are watched. A Hopf moves
        neither and is invisible - that limit is the price of not eigensolving.
        """
        if previous is None or current is None:
            return
        turned=(previous.dparam_ds is not None and current.dparam_ds is not None
                and numpy.sign(previous.dparam_ds)!=numpy.sign(current.dparam_ds))
        det_flipped=(previous.det_sign is not None and current.det_sign is not None
                     and previous.det_sign!=0 and current.det_sign!=0
                     and previous.det_sign!=current.det_sign)
        if turned:
            current.detected_bifurcation="fold"
        elif det_flipped:
            current.detected_bifurcation="branch_point"
        else:
            return
        self.log("Detected a {:s} between {:s} = {:.6g} and {:.6g}{:s}".format(
            current.detected_bifurcation,self._get_paramname_str(),
            previous.param_value,current.param_value,
            "" if not (turned and det_flipped) else " (both test functions changed)"))

    def propagate_stability(self,branch=None):
        """Carry the unstable count along a branch from the points where it was measured.

        sign(det J) flips exactly when an odd number of real eigenvalues crosses zero, so the count can
        be stepped along the branch and flipped at each recorded change. Where a LATER measured point
        disagrees with the propagated value, that is a crossing the determinant could not see - a Hopf -
        and it is reported rather than hidden.
        """
        branches=[branch] if branch is not None else self.branches
        for b in branches:
            for p in b:
                if p.stability_source==STABILITY_EIGEN:
                    p.unstable_count=p.measured_unstable_count() if p.eig_values else p.unstable_count
            measured=[i for i,p in enumerate(b) if p.stability_source==STABILITY_EIGEN and p.unstable_count is not None]
            if not measured:
                continue
            def walk(indices,step):
                count=None
                last_det=None
                for i in indices:
                    p=b[i]
                    if p.stability_source==STABILITY_EIGEN and p.unstable_count is not None:
                        if count is not None and p.unstable_count!=count:
                            self.log(("Warning: the stability inferred from the determinant disagrees with "
                                      "the spectrum measured at {:s} = {:.6g} ({:d} vs {:d} unstable). A "
                                      "Hopf bifurcation was probably crossed - the determinant cannot see "
                                      "one.").format(self._get_paramname_str(),p.param_value,count,p.unstable_count))
                        count=p.unstable_count
                        last_det=p.det_sign
                        continue
                    if count is None:
                        continue
                    if p.det_sign is not None and last_det is not None and p.det_sign!=last_det:
                        count=count+1 if count==0 else count-1
                    if p.det_sign is not None:
                        last_det=p.det_sign
                    p.unstable_count=count
                    p.stability_source=STABILITY_INFERRED
            walk(range(len(b)),1)
            walk(range(len(b)-1,-1,-1),-1)

    def compute_spectrum(self,point,force:bool=False)->bool:
        """Solve the eigenproblem at a point, from its own state dump, scanning the requested modes.

        This is what makes quick mode a workflow rather than a compromise: the dumps are all there, so a
        cheap sweep can have its eigenvalues filled in afterwards without redoing the continuation - and
        it is equally how a spectrum is REDONE after the eigenvalue count or the mode list changes.
        ``force`` is only about which points a caller offers; a point handed here is always recomputed.
        """
        if point is None or not point.statefile:
            return False
        restore=self.current_point
        try:
            self.load_pt(point)
            self.problem.solve_eigenproblem(self.neigen,self.shift,**self._mode_kwargs())
            # Back-filled spectra have to land in the same units as the ones recorded during the sweep.
            spectrum=self._phys_eig(list(self.problem.get_last_eigenvalues()))
            point.eig_values=[complex(v) for v in spectrum]
            point.eig_modes=self._last_solved_modes(len(spectrum))
            point.eig_settings=self.current_eigen_settings()
            self._record_mode_observables(point)
            point.eig_value_Re=numpy.real(spectrum[0])
            point.eig_value_Im=numpy.imag(spectrum[0])
            point.stability_source=STABILITY_EIGEN
            point.unstable_count=point.measured_unstable_count()
            return True
        except Exception as e:
            self.log("Could not compute the spectrum at "+self._get_paramname_str()+
                     " = {:.6g}: {:s}".format(point.param_value,repr(e)))
            return False
        finally:
            if restore is not None and restore is not point:
                try:
                    self.load_pt(restore)
                except Exception as e:
                    self.log("Could not return to the previous point: "+repr(e))

    def compute_spectrum_for_branch(self,branch=None,force:bool=False)->int:
        """Compute the spectrum along a branch. Abortable.

        Without ``force`` this takes the points that need it: the ones with no spectrum, and the ones
        whose spectrum was computed with different settings from the current ones. That second group is
        the point of spectrum_is_stale() - raising neigen from 4 to 30 used to recompute nothing at all,
        because every point already "had" a spectrum. ``force`` redoes the whole branch, which is what
        the explicit Recompute command asks for.
        """
        b=branch if branch is not None else self._get_current_branch()
        todo=[p for p in b if force or p.stability_source!=STABILITY_EIGEN or not p.eig_values
              or self.spectrum_is_stale(p)]
        done=0
        for i,p in enumerate(todo):
            if self._abort_requested:
                self._abort_requested=False
                self.log("Spectrum back-fill aborted after {:d} of {:d} points".format(done,len(todo)))
                break
            self._status("EIGENVALUES {:d}/{:d}".format(i+1,len(todo)))
            if self.compute_spectrum(p,force=force):
                done+=1
        self.propagate_stability(b)
        self.log("Computed the spectrum at {:d} of {:d} points".format(done,len(todo)))
        self._changed()
        return done

    # ------------------------------------------------------------------ solver commands

    #: First parameter offset tried when stepping onto the other branch, relative to the parameter's
    #: own magnitude. Smaller is more faithful to the normal form but closer to the singular point.
    branch_switch_offset=0.02

    def _classify_current_point(self)->dict | None:
        """Normal form of the bifurcation the problem is sitting at, or None if it cannot be had.

        Non-fatal on purpose. This is computed while a bifurcation is being recorded, and a normal-form
        calculation that fails must not take the located point down with it - losing the bifurcation is a
        worse outcome than not knowing which kind it is.
        """
        from ...generic.bifurcation_tools import NormalFormCalculator
        try:
            if self.problem.get_bifurcation_tracking_mode()=="":
                # get_normal_form reads the critical eigenpair from the last eigensolve. While tracking
                # is active that is already the critical one AND solve_eigenproblem would refuse, so this
                # only runs when classifying after the fact.
                self.problem.solve_eigenproblem(self.neigen,self.shift)
            return NormalFormCalculator(self.problem).get_normal_form(self._get_paramname_str())
        except Exception as e:
            self.log("Could not compute the normal form of this bifurcation: "+repr(e))
            return None

    def branch_switch(self,offset:float | None=None,direction:int=1):
        """Step onto the other branch through the current bifurcation and record it as a new branch.

        The numerics live on :py:meth:`~pyoomph.generic.problem.Problem.switch_branch`, so the same
        manoeuvre is available to a plain script; what is here is the part that is about the diagram -
        which point we are at, opening a branch for the result, and the step size to carry on with.
        """
        if self.on_locus():
            raise RuntimeError("Branch switching applies to an ordinary branch, not to a bifurcation locus")
        cp=self._get_current_point()
        if cp.eig_value_Re!=0:
            raise RuntimeError("Can only switch branches at bifurcations")
        if cp.bifurcation_info is None:
            # Computed here rather than sending the user away to set a flag and redo the run: the problem
            # is sitting at the bifurcation, which is all the normal form needs. This is also what makes
            # switching work at a point loaded from a diagram recorded without classification.
            self.log("This bifurcation was not classified; computing its normal form now")
            cp.bifurcation_info=self._classify_current_point()
        if cp.bifurcation_info is None:
            raise RuntimeError("Cannot switch branches: the normal form of this bifurcation could not be "
                               "computed, so there is no prediction for where the other branch goes. The "
                               "reason is in the log above.")
        if cp.bifurcation_info.get("type")=="fold":
            self.log("A fold has only one branch through it - there is nothing to switch to")
            return False

        self._status("BRANCH SWITCHING")
        ds=self.problem.switch_branch(self.get_bifurcation_parameter(),normal_form=cp.bifurcation_info,
                                      offset=offset,direction=direction,
                                      relative_offset=self.branch_switch_offset,quiet=True)
        if ds is None:
            self.log("Could not step onto the other branch. Try a different offset "
                     "(gui.controller.branch_switch_offset) or the other direction.")
            return False
        # Continue at the scale of the jump just taken, not at whatever ds the old branch had reached:
        # just off the bifurcation dU/dparameter is badly conditioned and a larger step overshoots back
        # onto the branch we came from. arclength_continuation grows it again by itself.
        self._last_ds=ds
        self.log("Switched onto the other {:s} branch; ds set to {:.3g}".format(
            str(cp.bifurcation_info.get("type")),self._last_ds))
        self.problem.solve_eigenproblem(self.neigen,self.shift)
        self._new_branch()
        self._tangs={}
        self._add_current_state()
        self._update_tangents()
        self._mode="al"
        self._changed()
        return True

    def transient_leave_branch(self,eigenindex=0):
        self._status("LEAVING BRANCH TRANSIENTLY")
        if self.quick_mode and len(self.problem.get_last_eigenvectors())<=eigenindex:
            # This perturbs the solution along an eigenvector, so it needs one; in quick mode none was
            # computed. Solving here beats failing several frames deep in the perturbation.
            self.log("Quick mode: solving the eigenproblem, which leaving a branch transiently needs")
            self.problem.solve_eigenproblem(self.neigen,self.shift)
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
        self._new_branch()
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
            last=branch[1].get_coordinate(self.y_axis,xspec=self.x_axis)
        else:
            last=branch[0].get_coordinate(self.y_axis,xspec=self.x_axis)
        xbase=[]
        ybase=[]
        sbase=[]
        for p in branch:
            if p==newp:
                continue
            curr=p.get_coordinate(self.y_axis,xspec=self.x_axis)
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
            xn,yn=newp.get_coordinate(self.y_axis,xspec=self.x_axis)
            xn,yn=float(xn*pscale),float(yn*obsscale)
            # Quite demanding, but lets give it a try: Could be improved of course
            shortest_l=1e50
            shortest_news:float=0
            # Add some additional contribution due penalized strong changes in direction
            def tangdot(x,y,index):
                tdot=(x[index]-x[index-1])*(x[index+1]-x[index])+(y[index]-y[index-1])*(y[index+1]-y[index])
                distdenom=(x[index+1]-x[index-1])**2+(y[index+1]-y[index-1])**2
                if distdenom<=0:
                    # The neighbours either side coincide in the plotted coordinates, so there is no
                    # direction to penalize. This used to divide by zero: the resulting -inf made that
                    # insertion index win the shortest-length search below unconditionally, i.e. the
                    # new point was ordered by an arithmetic accident rather than by the metric.
                    return 0.0
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

    def axis_tangent(self)->"NPFloatArray | None":
        """Direction of travel per unit ds, measured in the CURRENT axes.

        Taken from the two most recent points rather than from the solver's arclength derivative,
        because that derivative is (dparameter, ddofs) of whatever system is active - on a locus the
        augmented one - and says nothing about a second parameter on the vertical axis. Two points in
        the plotted coordinates always do.
        """
        branch=self.current_branch
        cp=self.current_point
        if branch is None or cp is None or len(branch)<2 or cp not in branch:
            return None
        i=branch.index(cp)
        other=branch[i-1] if i>0 else branch[i+1]
        try:
            a=cp.get_coordinate(self.y_axis,xspec=self.x_axis)
            b=other.get_coordinate(self.y_axis,xspec=self.x_axis)
        except KeyError:
            return None
        ds=abs(self._last_ds) or 1.0
        return numpy.array([(a[0]-b[0])/ds,(a[1]-b[1])/ds])

    def _tangent_length(self)->float:
        """Length of the direction of travel in the current axes, 0 if there is none yet."""
        tvec=self.axis_tangent()
        if tvec is None:
            return 0.0
        return float(numpy.sqrt(tvec[0]*tvec[0]+tvec[1]*tvec[1]))

    def multistep(self):
        """Keep stepping until the branch leaves the visible axes or the user aborts.

        The step size is capped at the distance the first step covered, so the sweep does not run
        away once the branch turns and the continuation is free to grow ds.
        """
        assert self._current_observable is not None
        # On a brand-new branch there is only one point, so there is no direction yet and the
        # reference length would be zero - which would scale every subsequent ds down to nothing.
        # One ordinary step establishes it.
        if self._tangent_length()==0.0:
            self.step()
            self.save_all()
        xlim=self.view.get_xlim()
        ylim=self.view.get_ylim()
        xscale=self.view.get_xscale()
        max_ds=self._tangent_length()*abs(self._last_ds)
        cp0=self._get_current_point().get_coordinate(self.y_axis,xspec=self.x_axis)
        while True:
            cp=self._get_current_point().get_coordinate(self.y_axis,xspec=self.x_axis)
            if cp[0]<xlim[0] or cp[0]>xlim[1] or cp[1]<ylim[0] or cp[1]>ylim[1]:
                break
            if self._abort_requested:
                self._abort_requested=False
                self.log("Multistep aborted")
                break
            self.step()
            self.save_all()
            atang=self.axis_tangent()
            tvec=(atang if atang is not None else numpy.zeros(2))*self._last_ds
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
        locus=self.on_locus()
        quick=self.quick_mode and not locus
        self._status("FOLLOWING THE BIFURCATION" if locus else
                     ("QUICK STEPPING (no eigensolve)" if quick else "ARCLENGTH STEPPING"))
        theta_before=self.problem.get_arc_length_theta_sqr()
        dparam_ds_before=self.problem.get_arc_length_parameter_derivative()
        ds=self.problem.arclength_continuation(self.get_bifurcation_parameter(),ds)
        ds=self._recast_ds_after_metric_change(ds,theta_before,dparam_ds_before)
        if quick:
            # The whole point: no eigensolve. Two test functions are recorded instead, read from work
            # the step has already done - the factorisation the Newton solve produced, and the
            # continuation tangent.
            self._add_current_state(eig_value=None,det_sign=self._determinant_sign(),
                                   dparam_ds=self.problem.get_arc_length_parameter_derivative())
            self._detect_bifurcation_between(origin,self.current_point)
        elif locus:
            self._add_locus_state()
        else:
            self.problem.solve_eigenproblem(self.neigen,self.shift,
                                            **(self._mode_kwargs() if self.compute_modes_during_sweep else {}))
            self._add_current_state()

        self._adapt_after_step()
        self.reorder_branch_upon_point_insertion(self._get_current_branch(),self._get_current_point())
        self._last_ds=ds
        self._update_tangents()
        if origin._tangs is None or len(origin._tangs)==0:
            origin._tangs=self._get_current_point()._tangs.copy()

        return ds

    def scan_stripe(self,point=None)->bool:
        """Find every eigenvalue in the stripe |Re|<stripe_re, |Im|<stripe_im at one point.

        A shift-invert solve returns the eigenvalues nearest the shift, so a Hopf pair far up the
        imaginary axis is simply not in the answer. This asks the other question - everything inside a
        region - which is what the contour method is for.

        Only the SLEPc solver can do it, and only against a complex PETSc build. Merging (the default)
        keeps whatever the point already had and adds what the scan found, which is the useful
        combination: the shift-invert spectrum plus the pairs it missed.
        """
        point=point if point is not None else self.current_point
        if point is None:
            return False
        solver=self.problem.get_eigen_solver()
        if not hasattr(solver,"set_eigenvalue_region"):
            self.log("The stripe scan needs the SLEPc eigensolver (problem.set_eigen_solver('slepc')); "
                     "the current one is "+type(solver).__name__)
            return False
        restore=self.current_point
        try:
            self._status("SCANNING THE STRIPE")
            if restore is not point:
                self.load_pt(point)
            solver.set_eigenvalue_region(-abs(self.stripe_re),abs(self.stripe_re),
                                         -abs(self.stripe_im),abs(self.stripe_im))
            try:
                found,modes=self._solve_spectrum(neigen=max(1,int(self.stripe_max)))
            finally:
                # Always put the solver back, or every later eigensolve silently becomes a region scan.
                solver.eigenvalue_region=None
            self.log("Stripe |Re|<{:g}, |Im|<{:g}: {:d} eigenvalue{:s}".format(
                self.stripe_re,self.stripe_im,len(found),"" if len(found)==1 else "s"))
            if len(found)>=int(self.stripe_max):
                self.log("That is the cap ({:d}); the stripe may hold more - raise it to be sure"
                         .format(int(self.stripe_max)))
            self._store_spectrum(point,found,modes,merge=self.stripe_merge)
            return True
        except Exception as e:
            self.log("The stripe scan failed:",repr(e))
            return False
        finally:
            if restore is not None and restore is not point:
                try:
                    self.load_pt(restore)
                except Exception as e:
                    self.log("Could not return to the previous point: "+repr(e))

    def scan_stripe_for_branch(self,branch=None)->int:
        """Scan the stripe at every point of a branch. Abortable."""
        b=branch if branch is not None else self._get_current_branch()
        done=0
        for i,p in enumerate(b):
            if self._abort_requested:
                self._abort_requested=False
                self.log("Stripe scan aborted after {:d} of {:d} points".format(done,len(b)))
                break
            self._status("STRIPE {:d}/{:d}".format(i+1,len(b)))
            if self.scan_stripe(p):
                done+=1
        self.propagate_stability(b)
        self.log("Scanned the stripe at {:d} of {:d} points".format(done,len(b)))
        self._changed()
        return done

    def _store_spectrum(self,point,values,modes,merge:bool):
        """Put a computed spectrum on a point, optionally merging it with what is already there.

        Merging concatenates and drops duplicates: a region scan and a shift-invert solve overlap near
        the shift, and the same eigenvalue arriving twice would be counted twice by the unstable count.
        The result is re-sorted by descending real part, because "the leading eigenvalue" and the
        located-bifurcation test both read the first entry.
        """
        vals=[complex(v) for v in values]
        mods=list(modes) if modes is not None else None
        if merge and point.eig_values:
            old_modes=point.eig_modes if point.eig_modes is not None else [0.0]*len(point.eig_values)
            new_modes=mods if mods is not None else [0.0]*len(vals)
            allv=list(point.eig_values)+vals
            allm=list(old_modes)+list(new_modes)
            keptv:list[complex]=[]
            keptm:list[float]=[]
            for v,m in zip(allv,allm):
                scale=max(1.0,abs(v))
                if any(mm==m and abs(v-kv)<=1e-8*scale for kv,mm in zip(keptv,keptm)):
                    continue        # the same eigenvalue of the same mode, found by both methods
                keptv.append(v)
                keptm.append(m)
            vals,mods=keptv,(keptm if (mods is not None or point.eig_modes is not None) else None)
        order=sorted(range(len(vals)),key=lambda i: -numpy.real(vals[i]))
        point.eig_values=[vals[i] for i in order]
        point.eig_modes=[mods[i] for i in order] if mods is not None else None
        if point.eig_values:
            point.eig_value_Re=numpy.real(point.eig_values[0])
            point.eig_value_Im=numpy.imag(point.eig_values[0])
        point.stability_source=STABILITY_EIGEN
        point.unstable_count=point.measured_unstable_count()
        point.eig_settings=self.current_eigen_settings()
        self._record_mode_observables(point)

    def _adapt_after_step(self):
        """Remesh and/or adapt after an arclength step, per adapt_policy. Off by default.

        "when_needed" hands the decision to Problem.remesh_handler_during_continuation, which consults
        the problem's own remeshing_necessary() and returns False when there is nothing to do - so the
        RemeshWhen criteria the problem already declares stay in charge. "every_n" forces one.

        The handler carries the continuation tangent across the new mesh itself (through the history
        slots) and renormalises it, so ds keeps meaning a step length afterwards.
        """
        if self.adapt_policy=="off":
            return False
        self._steps_since_adapt+=1
        force=False
        if self.adapt_policy=="every_n":
            if self._steps_since_adapt<max(1,int(self.adapt_every_n)):
                return False
            force=True
        did=False
        try:
            self._status("REMESHING")
            did=bool(self.problem.remesh_handler_during_continuation(force=force,
                                                                    resolve=self.adapt_resolve))
        except Exception as e:
            self.log("Remeshing during continuation failed:",repr(e))
        if self.adapt_spatial>0:
            try:
                self._status("ADAPTING")
                nref=nunref=0
                for _ in range(int(self.adapt_spatial)):
                    r,u=self.problem.adapt()
                    nref+=r
                    nunref+=u
                    if r==0 and u==0:
                        break
                if nref or nunref:
                    did=True
                    # adapt() carries the continuation tangent across the new mesh through the history
                    # slots and renormalises it, so ds still means a step length afterwards.
                    if self.adapt_resolve:
                        self.problem.solve()
                    self.log("Adapted during continuation: {:d} refined, {:d} unrefined; ndof is now "
                             "{:d}".format(nref,nunref,self.problem.ndof()))
            except Exception as e:
                self.log("Adapting during continuation failed:",repr(e))
        if did:
            self._steps_since_adapt=0
            self._adapt_to_eigenfunction()
        elif force:
            self._steps_since_adapt=0
        return did

    def _adapt_to_eigenfunction(self):
        """Refine towards an eigenfunction after a remesh, if that was asked for.

        Refused up front rather than several frames into refine_eigenfunction: it renumbers, which pulls
        the augmented dof vector out from under a bifurcation tracker, and it cannot run distributed.
        """
        if not self.adapt_to_eigenfunction:
            return
        if self.problem.get_bifurcation_tracking_mode()!="":
            self.log("Not refining towards an eigenfunction: a bifurcation tracker is active, and "
                     "adapting would renumber the augmented system out from under it")
            return
        if self.problem.is_distributed():
            self.log("Not refining towards an eigenfunction: not supported on a distributed problem")
            return
        try:
            self._status("ADAPTING TO THE EIGENFUNCTION")
            self.problem.solve_eigenproblem(max(self.neigen,self.adapt_eigenindex+1),self.shift)
            self.problem.refine_eigenfunction(eigenindex=self.adapt_eigenindex,
                                              resolve_neigen=max(self.neigen,1))
            self.log("Refined towards eigenfunction {:d}; ndof is now {:d}".format(
                self.adapt_eigenindex,self.problem.ndof()))
        except Exception as e:
            self.log("Could not refine towards the eigenfunction:",repr(e))

    def step_and_shrink(self):
        """One step, but never let ds grow - the ``*`` command of the old key interface."""
        ds_backup=self._last_ds
        self.step()
        self._last_ds=(-1 if self._last_ds<0 else 1)*min(abs(self._last_ds),abs(ds_backup))

    def _mode_of_eigenvalue(self,index:int)->"tuple[str,float]":
        """(kind, mode) of one eigenvalue of the spectrum the problem still holds; mode 0 for the base."""
        kind=self.normal_mode_kind()
        if not kind:
            return "",0.0
        raw=self.problem.get_last_eigenmodes_m() if kind=="m" else self.problem.get_last_eigenmodes_k()
        if raw is None or index>=len(raw):
            return kind,0.0
        return kind,float(raw[index])

    def normal_mode_kind(self)->str:
        """"m" for azimuthal, "k" for a Cartesian normal mode, "" when the problem has neither.

        Reads Problem.is_normal_mode_stability_set_up(), i.e. whether setup_for_stability_analysis was
        called with azimuthal_stability or additional_cartesian_mode.
        """
        kind=self.problem.is_normal_mode_stability_set_up()
        if kind=="azimuthal":
            return "m"
        if kind=="cartesian":
            return "k"
        return ""

    def _mode_kwargs(self)->dict:
        """The azimuthal_m/normal_mode_k argument for solve_eigenproblem, or nothing.

        The base mode is always included, so a diagram keeps a base-state spectrum whatever else is
        scanned. Returns {} when there are no modes to scan, which leaves solve_eigenproblem in its
        ordinary single-mode form and get_last_eigenmodes_* as None.
        """
        kind=self.normal_mode_kind()
        if not kind or not self.normal_modes:
            return {}
        if self.on_locus() and self.problem.get_bifurcation_tracking_mode() not in ("azimuthal","cartesian_normal_mode"):
            # With a fold/Hopf/pitchfork tracker active the m!=0 problems would need the equations
            # renumbered, which solve_eigenproblem refuses. The base state is still available.
            return {}
        if kind=="m":
            modes=sorted({0}|{int(m) for m in self.normal_modes})
            return {"azimuthal_m":modes}
        modes=sorted({0.0}|{float(k) for k in self.normal_modes})
        return {"normal_mode_k":modes}

    def _solve_spectrum(self,shift=None,neigen=None)->"tuple[list[complex],list[float] | None]":
        """Solve the eigenproblem, scanning the requested normal modes, as physical rates.

        Returns ``(values, modes)`` with ``modes`` parallel to ``values`` - which is what the problem
        itself provides: with a list of modes, solve_eigenproblem merges the spectra, sorts them
        together by descending real part and records the mode of each eigenvalue. ``modes`` is None when
        only the base state was solved.
        """
        kw=self._mode_kwargs()
        self.problem.solve_eigenproblem(self.neigen if neigen is None else neigen,
                                        self.shift if shift is None else shift,**kw)
        values=self._phys_eig(list(self.problem.get_last_eigenvalues()))
        modes=None
        if kw:
            raw=(self.problem.get_last_eigenmodes_m() if "azimuthal_m" in kw
                 else self.problem.get_last_eigenmodes_k())
            if raw is not None and len(raw)==len(values):
                modes=[float(m) for m in raw]
        return values,modes

    def current_eigen_settings(self)->tuple:
        """What a spectrum computed right now would be computed with, for the stale check."""
        kw=self._mode_kwargs()
        modes=next(iter(kw.values())) if kw else []
        return eigen_settings(self.neigen,self.shift,modes)

    def spectrum_is_stale(self,point)->bool:
        """True when a point's spectrum was computed with different settings from the current ones.

        This is what makes raising the eigenvalue count actionable: without it the back-fill skips every
        point that already has a spectrum, so nothing is recomputed and the extra eigenvalues never
        appear. A point from before the settings were recorded reports False - it is not KNOWN to be
        stale, and treating it as stale would recompute whole diagrams on load.
        """
        if point is None or point.eig_settings is None:
            return False
        return tuple(point.eig_settings)!=self.current_eigen_settings()

    def critical_eigenindex(self)->int:
        """Index of the eigenvalue nearest the imaginary axis, i.e. the one about to cross.

        NOT the leading one, which is what bifurcation tracking picks by default. The spectrum is sorted
        by descending real part, so on a branch that is ALREADY unstable index 0 is the eigenvalue that
        crossed at some earlier bifurcation, while the one about to cross next is further down. Tracking
        the leading one there converges to the wrong thing - the Kuramoto-Sivashinsky hexdot branch past
        its fold is exactly that case: the transcritical point at gamma = 0 belongs to the SECOND
        eigenvalue.
        """
        evs=self.problem.get_last_eigenvalues()
        if evs is None or len(evs)==0:
            return 0
        return int(numpy.argmin(numpy.abs(numpy.real(numpy.asarray(evs)))))

    def locate_bifurcation(self,pitchfork:bool=False,eigenindex:int | None=None):
        """Find the nearest bifurcation and record it.

        ``eigenindex`` selects which eigenvalue is expected to cross; by default the one nearest the
        imaginary axis (:py:meth:`critical_eigenindex`). Pass it explicitly to track a particular mode -
        the eigenvalue list in the Points tab is there to read the index off.
        """
        if self.on_locus():
            raise RuntimeError("Already following a bifurcation. Leave the locus first "
                               "(Bifurcation -> Leave the locus) to work on an ordinary branch.")
        self._status("BIFURCATION FINDING"+(" (PITCHFORK)" if pitchfork else ""))
        self.problem.solve_eigenproblem(self.neigen,self.shift,**self._mode_kwargs())
        if eigenindex is None:
            eigenindex=self.critical_eigenindex()
        evs=self.problem.get_last_eigenvalues()
        if evs is not None and len(evs)>eigenindex:
            self.log("Tracking eigenvalue {:d} of {:d}: {:.4g}{:+.4g}i{:s}".format(
                eigenindex,len(evs),numpy.real(evs[eigenindex]),numpy.imag(evs[eigenindex]),
                "" if eigenindex==0 else "  (not the leading one - this branch is already unstable)"))
        # A critical eigenvalue belonging to a normal mode needs the matching tracker: the ordinary
        # fold/Hopf handlers work on the base state and would converge to something else entirely.
        kind,mode=self._mode_of_eigenvalue(eigenindex)
        track_kw:dict={}
        btype:str | None="pitchfork" if pitchfork else None
        if mode:
            btype="azimuthal" if kind=="m" else "cartesian_normal_mode"
            track_kw={"azimuthal_mode":int(mode)} if kind=="m" else {"cartesian_wavenumber_k":mode}
            self.log("That eigenvalue belongs to {:s}={:g}, so a {:s} bifurcation is tracked".format(
                kind,mode,btype))
        self.problem.activate_bifurcation_tracking(self._paramname,btype,
                                                   eigenvector=eigenindex,**track_kw)
        self.problem.solve(max_newton_iterations=20)
        self._add_current_state()
        self._update_tangents()
        self.problem.deactivate_bifurcation_tracking()
        self.reorder_branch_upon_point_insertion(self._get_current_branch(),self._get_current_point())

    def locate_bifurcation_or_switch(self,eigenindex:int | None=None):
        """What ``b`` has always done: at a bifurcation, switch branch; otherwise, go find one."""
        if self.current_point is not None and self.current_point.eig_value_Re==0:
            self.branch_switch()
        else:
            try:
                self.locate_bifurcation(eigenindex=eigenindex)
            except Exception as e:
                # Bifurcation tracking must be switched off again, otherwise the augmented system
                # stays active and every subsequent solve operates on the wrong problem.
                self.problem.deactivate_bifurcation_tracking()
                self.log("Bifurcation finding failed:",e)

    def set_arclength_scaling(self,scale:bool):
        self.scale_arc_length=scale
        self.problem.set_arc_length_parameter(scale_arc_length=scale)
        theta=self.problem.get_arc_length_theta_sqr()
        if scale:
            self.log("Scale arclength is on: theta^2 will be retuned each step so the parameter takes",
                     "{:.3g}".format(self.arclength_proportion),"of the arclength (theta^2 is now {:.4g})".format(theta))
        else:
            # Switching off FREEZES theta^2 wherever the last scaled step left it - which near a fold
            # is a very small number, since scaling drives theta^2 down as |dU/dparameter| grows. Worth
            # saying out loud, because it silently decides what ds means from here on.
            self.log("Scale arclength is off: theta^2 stays at its current value {:.4g}".format(theta))

    def set_arclength_inner_product(self,kind:"str | None"):
        """Choose the norm the arclength measures the solution in; see
        :py:meth:`~pyoomph.generic.problem.Problem.set_arclength_inner_product`.

        ``None`` restores oomph's dof-sum metric together with the scaling, which is what the
        ``Scale arclength`` checkbox controls; the other choices make that scaling redundant and switch
        it off, since both would be tuning the same scalar.
        """
        self.problem.set_arclength_inner_product(kind)
        if kind is None:
            self.set_arclength_scaling(self.scale_arc_length)
            self.log("Arclength metric: oomph's dof sum")
        else:
            self.scale_arc_length=False
            self.log("Arclength metric:",kind,
                     "- mesh-independent, so the arclength scaling is no longer needed and is off")

    def set_arclength_proportion(self,proportion:float):
        """Set D, the share of the arclength the continuation parameter is given.

        Only has an effect while the scaling is on, where it makes ``(dparameter/ds)^2 == D`` hold after
        every step. Larger D spends more of a step on the parameter and less on the solution, so the
        sweep marches through the parameter faster but resolves a rapidly changing solution less well.
        """
        if not (0.0<proportion<1.0):
            raise ValueError("The arclength proportion is a fraction and must lie strictly between 0 "
                             "and 1 (oomph divides by both D and 1-D when it retunes theta^2), got "+str(proportion))
        self.arclength_proportion=proportion
        self.problem.set_arc_length_parameter(desired_proportion_of_arc_length=proportion)
        self.log("The parameter now takes {:.3g} of the arclength".format(proportion),
                 "" if self.scale_arc_length else "(no effect until arclength scaling is switched on)")

    def _recast_ds_after_metric_change(self,ds:float,theta_before:float,dparam_ds_before:float)->float:
        """Keep ds meaning the same physical step when theta^2 changed underneath it.

        The arclength constraint is ``ds = (dp/ds)*dp + theta^2 * (dU/ds).dU``, which along the tangent
        collapses to ``ds = dp/(dp/ds)`` - oomph re-derives exactly that at problem.cc:11029. So a given
        ds buys a parameter increment of ``ds*|dp/ds|``, and since ``|dp/ds| = 1/sqrt(1+theta^2*chi)``,
        changing theta^2 changes what the same number buys. oomph compensates only on the very FIRST
        step (problem.cc:11176-11181, guarded by ``!Arc_length_step_taken``) - after that the sweep just
        changes its stride.

        Toggling the scaling is where that shows: off at a fold, |dp/ds| can be 0.05, and switching on
        pins it at sqrt(D) = 0.71, so the next step covers fourteen times as much parameter for the same
        ds. Rescaling by ``|dp/ds|_before / |dp/ds|_after`` preserves the parameter increment, which
        preserves the whole step, the tangent direction being unchanged by theta^2.

        In a settled scaled sweep theta^2 moves every step while |dp/ds| stays pinned at sqrt(D), so the
        factor is 1 and this does nothing - it only acts where the metric really shifted.
        """
        if self.problem._arclength_inner_product is not None:
            # Problem.set_arclength_inner_product() retunes theta^2 inside arclength_continuation and
            # rescales the step it passes on by the same rule as below. Applying it again here would
            # double-count the correction - and with an inner product set, oomph's own Scale_arc_length
            # is off, so nothing else can move theta^2 behind our back.
            return ds
        theta_after=self.problem.get_arc_length_theta_sqr()
        if theta_after==theta_before:
            return ds
        dparam_ds_after=self.problem.get_arc_length_parameter_derivative()
        if abs(dparam_ds_after)<1e-30 or abs(dparam_ds_before)<1e-30:
            return ds
        factor=abs(dparam_ds_before)/abs(dparam_ds_after)
        if abs(factor-1.0)>0.01:
            self.log("theta^2 changed {:.4g} -> {:.4g}, so ds is recast by {:.4g} to keep the same "
                     "parameter step".format(theta_before,theta_after,factor))
        return ds*factor

    # ------------------------------------------------------------------ start / persistence

    def prepare(self,init_ds:float | None=None):
        """Get the problem ready for a diagram without solving anything.

        Split out of :py:meth:`start` because the *order* decides whether a stored diagram survives.
        The constructor already sets ``_runmode="overwrite"``, but that only helps if it happens before
        the problem is initialised: under the default "delete" runmode, ``initialise()`` strips every
        file from every subdirectory of the output directory (the ``for s in subdirs`` loop in
        ``Problem._do_initialise``), and the diagram lives in one of those. A script that solves - and
        so initialises - before building the GUI therefore threw its diagram away on every run.

        Idempotent; :py:meth:`start` and :py:meth:`BifurcationGUI.must_init` both call it.
        """
        if init_ds is not None:
            self._last_ds=init_ds
        # Re-assert what the constructor set, in case the problem was reconfigured in between. A
        # --runmode on the command line still wins, since parse_cmd_line runs inside initialise():
        # asking for "delete" explicitly is a request to start over, not something to override.
        self.problem._runmode="overwrite"
        self.problem.write_states=True
        self.problem.continuation_data_in_states=True
        if not self.problem.is_initialised():
            self.problem.initialise()
        if self._paramname is None:
            avail_params=list(self.problem.get_global_parameter_names())
            if len(avail_params)!=1:
                raise RuntimeError("Please create the BifurcationGUI with a parameter name, unless you have a problem with a single global parameter only")
            self._paramname=avail_params[0]
        elif not isinstance(self._paramname,str):
            self._paramname=self._paramname.get_name()
        self._resolve_eigen_unit()
        datadir=self.problem.get_output_directory(self.data_subdir)
        Path(datadir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(datadir,"_states")).mkdir(parents=True, exist_ok=True)

    def start(self,init_ds,initial_max_newton_iterations=10,ignore_saved:bool=False)->bool:
        """Discover the observables and solve the first point of a fresh diagram.

        Returns whether a starting point was computed. With a stored diagram present nothing is solved:
        its state dumps are converged solutions that :py:meth:`load_all` restores anyway, and the
        problem need not have a usable initial guess at all - :py:meth:`BifurcationGUI.must_init`
        returns False and the script skips its setup in exactly that case, so solving here would fail on the raw initial
        condition rather than save any work.

        ``ignore_saved`` starts fresh even though a diagram is stored, which is what a *failed* reload
        needs: the file being there says nothing about it being readable.
        """
        self.prepare(init_ds)
        self._avail_observables=[k for k in self.evaluate_observables().keys()]
        if self._current_observable is None or self._current_observable not in self._avail_observables:
            self._current_observable=self._avail_observables[0]
        if self.has_saved_state() and not ignore_saved:
            return False

        try:
            self.problem.solve(max_newton_iterations=initial_max_newton_iterations)
        except Exception as e:
            raise RuntimeError("Make sure the problem starts where it has a stationary solution") from e
        self.problem.solve_eigenproblem(self.neigen,self.shift)
        self._add_current_state()
        return True

    def has_saved_state(self)->bool:
        outdir=self.problem.get_output_directory(self.data_subdir)
        return os.path.isfile(os.path.join(outdir,"state.json"))

    def save_all(self):
        outdir=self.problem.get_output_directory(self.data_subdir)

        cur_branch=self._get_current_branch()
        fullinfo={}
        # Note on size: every point carries its whole spectrum, so this file grows with neigen times
        # the number of points (a few MB for neigen=50 over some hundreds of points). That is the price
        # of being able to inspect the eigenvalues of a point without reloading its state dump.
        fullinfo["version"]=STATE_VERSION
        fullinfo["parameter"]=self._paramname if isinstance(self._paramname,str) else None
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
        # The values are stored IN these units, so a reloaded diagram can label itself without having to
        # re-derive them - and a file written before units existed simply has none.
        fullinfo["observable_units"]=dict(self._observable_units)
        fullinfo["eigen_unit"]=self._eigen_unit
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

        version=int(fullinfo.get("version",1))
        if version>STATE_VERSION:
            self.log("Warning: state.json was written by a newer pyoomph (format {:d} > {:d}); "
                     "unknown entries are ignored".format(version,STATE_VERSION))
        # A version-1 file records neither which parameter was continued nor what the others were
        # held at. The continued one can be recovered honestly - it can only have been this GUI's -
        # but the fixed values cannot, and are left unknown rather than invented.
        stored_param=fullinfo.get("parameter")
        if isinstance(stored_param,str) and isinstance(self._paramname,str) and stored_param!=self._paramname:
            self.log("Note: the stored diagram was continued in '"+stored_param+
                     "', this session was started with '"+self._paramname+"'")
        default_param=stored_param if isinstance(stored_param,str) else (self._paramname if isinstance(self._paramname,str) else None)
        self.branches=[BifurcationGUISolutionBranch.from_dict(b,default_continuation_parameter=default_param)
                       for b in fullinfo["branches"]]
        if version<STATE_VERSION:
            self.log("Loaded a format-{:d} diagram: the parameters held fixed were not recorded, "
                     "so the slice is reported as unknown".format(version))
        for b in self.branches:
            if not b.slice_is_consistent():
                self.log("Warning: branch",self.branches.index(b),
                         "has points at differing values of its supposedly fixed parameters")
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
        # Only for labelling: the stored values are already in these units. A file that predates units
        # has none, and the units resolved from this run's own problem are kept instead.
        stored_units=fullinfo.get("observable_units")
        if stored_units:
            self._observable_units.update({str(k):str(v) for k,v in stored_units.items()})
        self._status("LOADING")
        self.load_pt(self.current_point)
        self._update_tangents()
        self._status(None)
        self._changed()

    def axis_range(self,spec:"AxisSpec | str")->tuple[float,float] | None:
        """Min/max along one axis over the whole diagram, for autoscaling after a switch."""
        lo,hi=1e30,-1e30
        found=False
        for b in self.branches:
            for p in b:
                try:
                    v=p.value_of(spec)
                except KeyError:
                    continue    # a branch that never recorded this quantity
                lo=min(lo,v)
                hi=max(hi,v)
                found=True
        return (lo,hi) if found else None

    def observable_range(self,obs:str)->tuple[float,float] | None:
        """Backwards-compatible alias of :py:meth:`axis_range` for an observable."""
        return self.axis_range(observable_axis(obs))

    def data_range(self)->tuple[tuple[float,float],tuple[float,float]] | None:
        """Bounding box of the whole diagram on the current axes."""
        try:
            xr=self.axis_range(self.x_axis)
            yr=self.axis_range(self.y_axis)
        except AssertionError:
            return None
        if xr is None or yr is None:
            return None
        return xr,yr


from ...typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
