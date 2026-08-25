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

from ...generic.bifurcation_tools import attach_normal_form_predictors, _as_real_eigenvector
from ...generic.codegen import EquationTree
from ...generic import Problem
from ... import _pyoomph_core as _pyoomph
from ...typings import *
from ...expressions.units import unit_to_string

from .model import (eigen_settings, BifurcationGUISolutionPoint, BifurcationGUISolutionBranch, AxisSpec,
                    AXIS_OBSERVABLE, AXIS_PARAMETER, as_axis, observable_axis, parameter_axis,
                    same_parameter_value, STABILITY_EIGEN, STABILITY_INFERRED, STABILITY_UNKNOWN,
                    BRANCH_SOLUTION, BRANCH_LOCUS, BRANCH_ORBIT,
                    ORBIT_MIN_TAG, ORBIT_MAX_TAG, ORBIT_T_KEY, orbit_band_names, orbit_band_base,
                    is_orbit_band_name)

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


def _eigen_columns(unit:str,orbit:bool=False)->"list[str]":
    """Header names for the two eigenvalue columns, carrying the rate unit when there is one.

    On an orbit branch the same two columns hold the Floquet EXPONENTS, which are rates in the same
    unit but a different quantity, so they say so.
    """
    suffix=" ["+unit+"]" if unit else ""
    if orbit:
        return ["ReFloquetExponent"+suffix,"ImFloquetExponent"+suffix]
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
        # A separate shift for the eigensolves taken WHILE A TRACKER IS INSTALLED - on a locus and at
        # a freshly located bifurcation alike - because self.shift's default of 0 is the one value
        # that cannot work there: the tracker has put an eigenvalue exactly at 0 (fold, pitchfork,
        # transcritical) or +-i*omega (Hopf, azimuthal), which is precisely where a zero shift asks
        # the shift-invert transform to factorise. Set to None to skip that eigensolve entirely and
        # keep the historical single synthetic value per tracked point.
        self.tracked_eigen_shift:float | complex | None=0.1
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
        #: Observable the diagram should open on, set before start(). Only a wish at that point: the
        #: names are not known until the problem is initialised, so it is resolved in start().
        self._initial_observable:str | None=None
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
        #: Half-widths of the stripe scanned for eigenvalues: ``|Re| < stripe_re`` and
        #: ``|Im| < stripe_im``. Literal markup, because sphinx reads bars around a word as a
        #: substitution reference and reported an undefined one.
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
        #: Arclength scaling (oomph's Scale_arc_length). Off, because the metric set below is already
        #: mesh-independent and both tune the same theta^2.
        self.scale_arc_length=False
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

        # --- deflation (dev_docs/deflation.md). Deflation looks for solutions the diagram does not
        # have yet: it multiplies the residual by a factor that blows up at every solution already
        # known, so Newton cannot converge onto one of them again and has to go somewhere else.
        #: Shift of the deflation operator. Larger means the known solutions are pushed away over a
        #: shorter range, so a nearby second solution is easier to reach and a distant one harder.
        self.deflation_alpha=0.1
        #: Order of the deflation, i.e. the power of 1/||U-W||. Integer, at least 1.
        self.deflation_p=2
        #: How far the guess is moved off a known solution before the deflated solve, or None for a
        #: value read off the current solution. It carries the SCALE of the whole search: the
        #: deflation measures ||U-W|| in units of it, so alpha is dimensionless and this is the one
        #: number that has to match the problem. A fixed 0.5 was wrong by orders of magnitude for any
        #: field that is not O(1) - measured on a pitchfork PDE of amplitude 1e-3, it found 7 of 21
        #: branches where a perturbation matched to the field found 21 of 21.
        self.deflation_perturbation:float | None=None
        #: Random perturbations tried per deflated solve before giving up. A failed attempt is a
        #: Newton solve that gives up early, so these are cheap next to the ones that succeed.
        self.deflation_random_tries=4
        #: Perturb along the leading eigenvector as well. That direction is a FIELD, so unlike a random
        #: dof-index vector it means the same thing however the mesh is partitioned, and it usually
        #: points at the branch that is about to appear. Costs one eigensolve per attempt.
        self.deflation_use_eigenperturbation=True
        #: Seed of the random perturbations; None for a different sequence every run. Seeded by default
        #: so that a search can be repeated -- and, under MPI, so that every rank perturbs identically.
        self.deflation_random_seed:int | None=0
        #: Newton iteration cap for a deflated solve, or None for the problem's own. A deflated solve
        #: is asked to fail often (that is how the search terminates), so a low cap costs little.
        self.deflation_max_newton_iterations:int | None=20
        #: Number of parameter steps a deflated continuation takes from the current value. Ten,
        #: matching initial_view_ds_ahead: with the default increment below that is exactly the part
        #: of the parameter axis a fresh diagram opens on, so the default scan covers the plot and
        #: stops there rather than being reported as cut short before it starts.
        self.deflated_scan_steps=10
        #: Signed parameter increment per step, or None for ds. Deflated continuation steps the
        #: parameter where arclength steps the arclength, so ds is the step size the user has already
        #: chosen and the one the direction arrow is drawn with -- which also makes "Reverse
        #: direction" reverse the scan.
        self.deflated_scan_dparam:float | None=None
        #: Solve an eigenproblem at every point of a deflated scan. On, like an ordinary continuation
        #: step: a branch drawn without stability is half a bifurcation diagram, and finding a new
        #: branch without knowing whether it is stable rarely answers the question that led to it.
        #: Turn it off on a problem where the eigensolves dominate; the spectra can then be filled in
        #: afterwards with "Compute the eigenvalues along this branch", once it is clear which of the
        #: branches found are worth them.
        self.deflated_scan_eigensolve=True
        #: Dof vectors the deflated search is currently avoiding, and the parameter value they belong
        #: to. Accumulated across clicks, so pressing "Deflated solve" repeatedly walks through the
        #: solutions at one parameter value instead of finding the same one again.
        self._deflation_known:list[NPFloatArray]=[]
        self._deflation_at:float | None=None

        # --- periodic orbits. A Hopf sheds a periodic orbit, and until now the GUI could only leave
        # one transiently and see where the solution ended up. Switching onto the orbit itself and
        # continuing it is what these settings drive; see dev_docs/bifurcation_loci.md.
        #: Number of time steps the orbit is discretized with. Raised at use time to a multiple of the
        #: collocation order and to an even number - with a DAE and an ODD number of time intervals a
        #: spurious Floquet multiplier lands on exactly -1, which is where a period doubling would be
        #: (dev_docs/floquet_multipliers.md).
        self.orbit_NT=30
        #: Discretization of the orbit. Only "collocation" and "floquet" carry an explicit degree of
        #: freedom at the end of the period, which is what the Floquet multipliers are computed from;
        #: the others can be continued but have no stability.
        self.orbit_mode:str="collocation"
        self.orbit_order=3
        self.orbit_GL_order=-1
        self.orbit_T_constraint:str="phase"
        #: Extra factor on the amplitude of the starting guess, for a Hopf whose normal form is a poor
        #: predictor of the orbit a finite step away from it.
        self.orbit_amplitude_factor=1.0
        #: Verify that the solved orbit did not collapse back onto the stationary branch. On, because
        #: a collapsed "orbit" looks like a perfectly good branch of solutions until its period is
        #: read.
        self.orbit_check_collapse=True
        #: Parameter step off the Hopf, or None to take what the current ds buys. The parameter offset
        #: is eps**2, so this IS the epsilon the switch is made with, squared - which is why it is
        #: taken from ds by default: the same key that steers the sweep steers the step onto the orbit.
        self.orbit_eps:float | None=None
        #: Samples per period used for the minimum, average and maximum of each observable, or None
        #: for one per time step of the orbit. Each sample is a full observable sweep.
        self.orbit_observable_samples:int | None=None
        #: Store the orbit's own degrees of freedom as full state dumps, one per time point, instead
        #: of the raw dof vector. Partition- and mesh-independent, at nT times the disk; forced on a
        #: distributed problem, where the raw vector means nothing outside its own partitioning.
        self.orbit_portable=False
        #: Compute the Floquet multipliers at every continuation point, like the eigensolve on an
        #: ordinary branch. Off means the orbit branch is drawn with unknown stability until they are
        #: asked for.
        self.floquet_enabled=True
        self.floquet_method:str="condensed"
        #: How many multipliers, or None for all of them (as far as the method allows).
        self.floquet_n:int | None=None
        #: Tolerance for recognising the TRIVIAL multiplier at 1, which every orbit has by
        #: time-translation invariance. It must be removed, and not because it would look like a
        #: bifurcation: it comes back as 1+-1e-15, so its exponent is a tiny number of EITHER SIGN, and
        #: left in it flips the whole branch between stable and unstable from one point to the next.
        self.floquet_unity_tol=1e-5
        #: Deadband on ``|mu| > 1`` for counting an unstable direction.
        self.floquet_unstable_tol=1e-6
        self.floquet_dense_threshold=2000
        self.floquet_shift_invert=True
        self.floquet_sigma:complex | None=None
        #: Leave the Floquet eigenvectors on the problem instead of restoring what was there before.
        #: They are the interesting field to plot on an orbit, but everything that reads
        #: get_last_eigenvalues() would then be reading multipliers as eigenvalues.
        self.floquet_feeds_eigen_panes=False
        #: The orbit currently installed on the problem, or None. Never used as a context manager:
        #: PeriodicOrbit.__exit__ deactivates the handler and re-seeds a transient history.
        self._orbit:Any=None
        #: Observables the exact time integral can serve, i.e. the mesh integral ones. ODE values and
        #: extremum observables are not meshes' integral functions and are averaged by sampling.
        self._integral_observables:set[str]=set()
        #: Unit and multiplier of the period, resolved once like the observables' own.
        self._orbit_T_unit=""
        self._orbit_T_mult=1.0
        #: Fingerprint of the equation numbering the installed orbit's blocks belong to, taken while
        #: the plain system was still in place. See _capture_dof_fingerprint.
        self._orbit_dof_fingerprint=""
        #: How far the trivial multiplier came out from 1 in the last Floquet computation, i.e. the
        #: accuracy of the discretization there. See _orbit_floquet.
        self._orbit_trivial_multiplier_error=float("nan")
        #: Relative amplitude below which a solved orbit is taken to have collapsed onto the
        #: stationary branch. Measured as the widest band on the branch divided by the scale of its
        #: own observable, so it is dimensionless and the same number serves every problem.
        self.orbit_collapse_tolerance=1e-6

        #: Supplies the visible axis ranges, see :py:class:`BifurcationViewLimits`.
        self.view:BifurcationViewLimits=_FixedViewLimits()
        #: Swappable so a worker-thread executor can be dropped in without touching the numerics.
        self.executor=InlineExecutor()

        self._on_changed:Callable[[],None] | None=None
        self._on_status:Callable[[str | None],None] | None=None
        self._on_log:Callable[[str],None] | None=None
        self._on_busy:Callable[[str | None],None] | None=None

        # Start in the mass-matrix metric rather than oomph's dof sum: the sum has no continuum limit,
        # so the same ds buys a different step after a refinement. Set on the problem directly - the
        # controller's set_arclength_inner_product() logs, and no observer is attached yet.
        self.problem.set_arclength_inner_product("l2")

    # ------------------------------------------------------------------ observers

    def set_observer(self,*,on_changed:Callable[[],None] | None=None,on_status:Callable[[str | None],None] | None=None,on_log:Callable[[str],None] | None=None,on_busy:Callable[[str | None],None] | None=None):
        """Install the callbacks through which a user interface follows this controller.

        ``on_changed``
            the diagram or the selection changed and should be redrawn.
        ``on_status``
            a long operation wants a label shown (``None`` clears it); this also repaints, which is
            what lets the abort flag be picked up mid-sweep.
        ``on_log``
            a line of progress text.
        ``on_busy``
            a solver task starts (label) or ends (``None``).
        """
        if on_changed is not None: self._on_changed=on_changed
        if on_status is not None: self._on_status=on_status
        if on_log is not None: self._on_log=on_log
        if on_busy is not None: self._on_busy=on_busy

    def clear_observer(self):
        """Forget the user interface's callbacks, once its window is gone.

        They are bound methods of the window, so a controller that outlives it - a script keeps the
        facade after :py:meth:`~pyoomph.utils.bifurcation_gui.BifurcationGUI.start` returns - would
        otherwise keep the whole window, and everything it holds, alive with it.
        """
        self._on_changed=None
        self._on_status=None
        self._on_log=None
        self._on_busy=None

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

    def set_initial_observable(self,name:str):
        """Observable the diagram opens on, instead of the first one in alphabetical order.

        Call it before :py:meth:`start`, where the name is checked against the observables the problem
        actually has - they cannot be listed earlier, since they are read off the initialised problem.
        A stored diagram overrides this, as it remembers what was last selected.

        The double space that the ExtremumObservables names carry ("liquid/interface/nx  [min, x]")
        need not be reproduced: names are matched with their whitespace collapsed.
        """
        self._initial_observable=name

    def _resolve_initial_observable(self):
        """Apply set_initial_observable() once the observable names are known."""
        want=self._initial_observable
        if want is None:
            return
        def norm(n:str)->str:
            return " ".join(n.split())
        match=None
        for n in self._avail_observables:
            if n==want or norm(n)==norm(want):
                match=n
                break
        if match is None:
            raise ValueError("Unknown initial observable "+repr(want)+". Available are:\n  "
                             +"\n  ".join(self._avail_observables))
        self._current_observable=match

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
                if eqtree._equations and eqtree._is_ode()==False:
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
                            # These, and only these, are what PeriodicOrbit.evaluate_observable_time_integral
                            # can integrate exactly over the period: it resolves the part before the
                            # last "/" with get_mesh(). An ODE domain is an ODEStorageMesh and not a
                            # mesh at all, and an extremum observable is not an integral function, so
                            # both are averaged by sampling instead.
                            self._integral_observables.add(key)
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
                if eqtree._is_ode()==True:
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

    def _resolve_orbit_period_unit(self):
        """Fix the unit the periods are recorded in, once, from the temporal scaling.

        Same discipline as the observables (:py:meth:`_resolve_observable_units`): the prefix is
        estimated once and the multiplier applied to every later period, so a branch cannot switch
        from ms to s half way along.
        """
        self._orbit_T_unit=""
        self._orbit_T_mult=1.0
        try:
            self._orbit_T_unit,self._orbit_T_mult=_unit_and_multiplier(self.problem.get_scaling("temporal"))
        except Exception as e:
            self.log("Could not determine the unit of the period:",repr(e))

    # ------------------------------------------------------------------ periodic orbits

    def _require_orbit(self):
        """The orbit installed on the problem, or a refusal naming what to do about it."""
        if self._orbit is None or self._orbit_handler() is None:
            raise RuntimeError("There is no periodic orbit on the problem. Switch onto one at a Hopf "
                               "bifurcation, or load a point of an orbit branch.")
        return self._orbit

    def orbit_period(self,dimensional:bool=True)->float:
        """The period of the orbit currently installed, in the recorded unit."""
        orbit=self._require_orbit()
        if not dimensional:
            return float(orbit.get_T(dimensional=False))
        return _si_value(orbit.get_T())*self._orbit_T_mult

    def _register_orbit_axes(self,names:"Iterable[str]"):
        """Make the derived orbit names selectable as axes, once each.

        Registered here rather than in the observable table because they only exist once an orbit has
        been computed - exactly like the mode observables, and for the same reason. The unit is copied
        from the observable the band belongs to and the multiplier is 1: these values have already
        been through _observable_mults on their way in, and multiplying again would scale the band
        away from its own centre line.
        """
        for name in names:
            if name in self._avail_observables:
                continue
            self._avail_observables.append(name)
            base=orbit_band_base(name)
            if base is not None:
                self._observable_units[name]=self._observable_units.get(base,"")
            elif name==ORBIT_T_KEY:
                self._observable_units[name]=self._orbit_T_unit
            self._observable_mults[name]=1.0
            self._extremum_axes.add(name)

    def _evaluate_orbit_observables(self)->"tuple[dict[str,float],dict]":
        """Minimum, average and maximum of every observable over one period, plus the period.

        The average goes under the observable's OWN name, so that every axis, tangent, export and
        selection path keeps working and an orbit branch continues the stationary line straight
        through the Hopf it came from. The extremes go under the derived band names.
        """
        orbit=self._require_orbit()
        handler=self._orbit_handler()
        assert handler is not None
        N=int(self.orbit_observable_samples or 0) or handler.get_num_time_steps()
        mins:dict[str,float]={}
        maxs:dict[str,float]={}
        sums:dict[str,float]={}
        count=0
        # endpoint=False: s=0 and s=1 are the same state, so including both counts one sample twice
        # and biases the mean by 1/N. It costs nothing here and would be invisible in the result.
        for _t in orbit.iterate_over_samples(N=N,endpoint=False):
            vals=self.evaluate_observables()
            for k,v in vals.items():
                if k in sums:
                    sums[k]+=v
                    mins[k]=min(mins[k],v)
                    maxs[k]=max(maxs[k],v)
                else:
                    sums[k]=v
                    mins[k]=v
                    maxs[k]=v
            count+=1
        if count==0:
            raise RuntimeError("The orbit produced no samples to evaluate the observables on")
        avg={k:v/count for k,v in sums.items()}
        # The exact, Gauss-Legendre weighted time average where the observable supports it. AFTER the
        # sampling loop, never inside it: both back the dofs up on the handler, and the nested second
        # one throws "the dofs have already been backed up".
        exact=sorted(k for k in avg if k in self._integral_observables)
        if exact:
            try:
                res=orbit.evaluate_observable_time_integral(*exact)
                if len(exact)==1:
                    res=(res,)
                Tdim=_si_value(orbit.get_T())
                if Tdim!=0:
                    for k,val in zip(exact,res):
                        avg[k]=_si_value(val)/Tdim*self._observable_mults.get(k,1.0)
            except Exception as e:
                self.log("Falling back to the sampled mean for the integral observables:",repr(e))
                exact=[]
        out=dict(avg)
        for k in avg:
            lo,hi=orbit_band_names(k)
            out[lo]=mins[k]
            out[hi]=maxs[k]
        out[ORBIT_T_KEY]=self.orbit_period()
        self._register_orbit_axes(out.keys())
        info={"T":float(orbit.get_T(dimensional=False)),
              "nT":int(handler.get_num_time_steps()),
              "mode":str(orbit.mode),"order":int(orbit.order),"GL_order":int(orbit.GL_order),
              "T_constraint":str(orbit.T_constraint),
              "nbase":int(handler.get_base_ndof()),
              "samples":int(count),
              "exact_average":bool(exact)}
        return out,info

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

    def _new_branch(self,*,kind:str=BRANCH_SOLUTION,tracked_parameter:str | None=None,
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
        if self.on_orbit():
            raise RuntimeError("Changing the continued parameter re-solves the problem, which would have to be done on the orbit as a whole. Load a stationary point first.")
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
        if self.on_orbit():
            raise RuntimeError("Moving a parameter continues the problem to the new value, which cannot be done while an orbit is installed. Load a stationary point first.")
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

    @property
    def locus_eigen_shift(self)->"float | complex | None":
        """Former name of :py:attr:`tracked_eigen_shift`, from when only a locus took that eigensolve."""
        return self.tracked_eigen_shift

    @locus_eigen_shift.setter
    def locus_eigen_shift(self,value:"float | complex | None"):
        self.tracked_eigen_shift=value

    def on_locus(self)->bool:
        return self.current_branch is not None and self.current_branch.kind==BRANCH_LOCUS

    def on_orbit(self)->bool:
        """Whether the current branch is a branch of periodic orbits."""
        return self.current_branch is not None and self.current_branch.kind==BRANCH_ORBIT

    def _orbit_handler(self)->"_pyoomph.PeriodicOrbitHandler | None":
        """The periodic-orbit handler installed on the problem, or None."""
        handler=self.problem.assembly_handler_pt()
        return handler if isinstance(handler,_pyoomph.PeriodicOrbitHandler) else None

    def _augmented_system_active(self)->bool:
        """Whether ANY augmented system is installed: a bifurcation tracker OR a periodic orbit.

        get_bifurcation_tracking_mode() answers "" for an orbit - start_orbit_tracking only swaps the
        assembly handler and never touches the tracking mode - so every guard written against that
        string alone is blind to an orbit, and would let a state be loaded, a mesh be adapted or a
        tracker be reinstalled on top of the augmented orbit dof vector.
        """
        return self.problem.get_bifurcation_tracking_mode()!="" or self._orbit_handler() is not None

    def _deactivate_any_augmentation(self):
        """Take whatever augmented system is installed back off. Covers orbits, unlike the bare call."""
        if self._augmented_system_active():
            self.problem.deactivate_bifurcation_tracking()

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
        want_locus=branch is not None and branch.kind==BRANCH_LOCUS
        tracking=self.problem.get_bifurcation_tracking_mode()!=""
        # An orbit handler counts as active here even though it reports no tracking mode; otherwise
        # moving from an orbit branch to a solution point would leave it installed and every later
        # solve would work on the augmented orbit system. See _augmented_system_active.
        active=tracking or self._orbit_handler() is not None
        if want_locus:
            assert branch is not None
            if branch.tracked_parameter is None:
                raise RuntimeError("Locus branch has no tracked parameter recorded")
            # Take the guess from the handler BEFORE deactivating, when there is one. The handler is
            # the authority on the tracked vector: a tracked point also solves the base state's
            # eigenproblem to record its spectrum, and although _tracked_spectrum puts the tracked
            # eigenpair back afterwards, anything else that eigensolves in between would leave
            # get_last_eigenvectors()[0] pointing at a different mode, and the tracker would restart
            # on that one.
            guess=None
            if tracking:
                # Only a TRACKER has a critical eigenvector to hand over; an orbit handler has none,
                # and asking it for one would read the augmented orbit vector as an eigenfunction.
                tracked=numpy.array(self.problem._get_bifurcation_eigenvector(),dtype=numpy.complex128) #type:ignore
                if len(tracked)>0:
                    guess=tracked
            if active:
                self.problem.deactivate_bifurcation_tracking()
            if guess is None:
                evs=self.problem.get_last_eigenvectors()
                guess=evs[0] if evs is not None and len(evs)>0 else None
            self.problem.activate_bifurcation_tracking(branch.tracked_parameter,
                                                       _as_tracking_type(branch.bifurcation_type),
                                                       eigenvector=guess)
            self.problem.reset_arc_length_parameters()
        elif branch is not None and branch.kind==BRANCH_ORBIT:
            # An orbit point's dump holds one phase of the cycle; the rest of it, and the period, come
            # back out of the companion the point was written with. Without this the problem would sit
            # on a single snapshot of the orbit and every later step would continue a stationary state.
            if active:
                self.problem.deactivate_bifurcation_tracking()
            self._orbit=None
            point=self.current_point
            if point is None or point.orbit_info is None:
                self.log("This orbit point carries no record of its cycle, so only the state it was "
                         "dumped at could be restored")
            else:
                self._install_orbit_from_sidecar(point)
        elif active:
            self.problem.deactivate_bifurcation_tracking()
            self._orbit=None
            self.problem.reset_arc_length_parameters()

    def start_locus(self,tracked:str,continue_in:str,bifurcation_type:str | None=None):
        """From a bifurcation, follow it through parameter space.

        ``tracked`` is adjusted to keep the bifurcation condition satisfied while ``continue_in`` is
        the one being stepped, which traces the locus of the bifurcation in the (continue_in, tracked)
        plane - the fold curve Bo_c(V) of the hanging-droplet tutorial, for instance.
        """
        if self.on_orbit():
            raise RuntimeError("A locus is a curve of STATIONARY bifurcations; leave the orbit branch "
                               "first.")
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
        self._new_branch(kind=BRANCH_LOCUS,tracked_parameter=tracked,bifurcation_type=mode)
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
        if branch is None or branch.kind!=BRANCH_LOCUS:
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
        if self.on_orbit():
            raise RuntimeError("A new branch is started from a stationary state; an orbit is "
                               "restored by loading one of its own points, which is what "
                               "carries the rest of its cycle.")
        self._new_branch()
        self.problem.load_state(statefile,ignore_continuation_data=True,ignore_eigendata=True)
        self.problem.reset_arc_length_parameters()
        self.problem.solve_eigenproblem(self.neigen,self.shift)
        self._add_current_state()
        self._update_tangents()
        self._changed()

    def _index_of_tracked_eigenvalue(self,spectrum,evecs,crit,tracked_vec,exclude=None)->"int | None":
        """Which entry of a base-state spectrum IS the eigenvalue the tracker is holding on the axis.

        Matched by EIGENVECTOR overlap among the entries that are numerically at ``crit``, not by
        |lambda - crit| alone: at a codim-2 point two eigenvalues sit on the axis together, and a
        nearest-value rule would then drop whichever of them rounding happened to favour - which is
        the one situation where getting this wrong actually costs something, since the second
        eigenvalue reaching zero is the whole reason for solving the spectrum here. The overlap
        |<v,w>|/(|v||w|) is 1 for the tracked mode and small for any other, however close its
        eigenvalue. It also keeps a Hopf's conjugate partner at -i*omega, which is a genuinely
        different entry of the spectrum and must stay in the list.

        Falls back to the nearest value when there are no eigenvectors to compare. None when nothing
        is close enough at all, i.e. the eigensolve never saw the tracked mode - too few eigenvalues,
        or a shift too far away from it.
        """
        if not spectrum:
            return None
        vals=numpy.asarray(spectrum,dtype=complex)
        scale=float(max(float(numpy.max(numpy.abs(vals))),abs(crit),1.0))
        # Deliberately generous: this window only has to rule out eigenvalues that are plainly
        # somewhere else, the overlap below does the discriminating. Tightened to 1e-4 when there are
        # no eigenvectors and the distance is all there is to go on.
        cand=[i for i in range(len(vals)) if abs(vals[i]-crit)<=1e-2*scale and i!=exclude]
        if not cand:
            return None
        nearest=int(min(cand,key=lambda i: abs(vals[i]-crit)))
        w=None if tracked_vec is None else numpy.asarray(tracked_vec,dtype=complex).ravel()
        if w is not None and evecs is not None and len(evecs)>0 and w.size>0:
            nw=float(numpy.linalg.norm(w))
            best,best_overlap=None,-1.0
            for i in cand:
                if i>=len(evecs):
                    continue
                v=numpy.asarray(evecs[i],dtype=complex).ravel()
                nv=float(numpy.linalg.norm(v))
                if v.size!=w.size or nv<=0 or nw<=0:
                    continue
                overlap=abs(complex(numpy.vdot(v,w)))/(nv*nw)
                if overlap>best_overlap:
                    best,best_overlap=i,overlap
            if best is not None and best_overlap>=0.5:
                return best
        return nearest if abs(vals[nearest]-crit)<=1e-4*scale else None

    def _tracked_spectrum(self)->"tuple[complex,list[complex] | None,list[float] | None,int | None]":
        """The spectrum to record at a point a bifurcation tracker has converged onto.

        Returns ``(critical, spectrum, modes, tracked_index)`` as physical rates. ``critical`` is the
        synthetic value the tracker reports - 0 + i*omega, exactly zero in the real part, which is
        what makes the point read as a bifurcation everywhere the diagram tests ``eig_value_Re == 0``.
        Re-solving THAT would turn the exact zero into a small nonzero number and the point would stop
        being a bifurcation.

        The rest of the spectrum is worth having and is solved for here: on a locus a second
        eigenvalue reaching the axis is a codim-2 point, and at a freshly located bifurcation the rest
        of the spectrum is what says whether the branch was already unstable and which mode goes next.
        The base state's eigenproblem is available while a tracker is installed, see
        Problem.solve_eigenproblem; it needs a nonzero shift (self.tracked_eigen_shift, which is also
        how the whole thing is switched off), since the tracker has just put an eigenvalue exactly
        where a zero shift would factorise.

        ``critical`` REPLACES the entry of the solved spectrum it belongs to rather than being pushed
        in front of it, so the tracked mode is listed once and at its exact value; ``tracked_index``
        says which entry that is. Recording both copies would show the same mode twice, a hand-width
        apart, and the solved copy's rounding-error-sized positive real part would have the point
        counted as unstable.

        ``spectrum`` and ``tracked_index`` are None when no extra eigensolve was taken - switched off,
        or failed. The failure is non-fatal on purpose: a shift-invert factorisation that does not go
        through should cost this point its spectrum, not abort a two-parameter sweep that may have
        been running for hours.
        """
        prob=self.problem
        # What tracked continuation and a tracked solve() leave behind: the single synthetic
        # eigenvalue, in the same nondimensional units as any other, plus the mode it belongs to.
        raw=prob.get_last_eigenvalues()
        crit=(complex(raw[0]) if raw is not None and len(raw)>0
              else 0+1j*float(prob._get_bifurcation_omega())) #type:ignore
        crit_modes=self._last_solved_modes(1)
        if not self.tracked_eigen_shift:
            return self._phys_eig(crit),None,crit_modes,None
        tracked_vec=None
        try:
            vec=numpy.array(prob._get_bifurcation_eigenvector(),dtype=numpy.complex128) #type:ignore
            tracked_vec=vec if vec.size>0 else None
        except Exception:
            tracked_vec=None
        # The eigensolve overwrites the problem's record of the tracked eigenpair, and everything that
        # runs afterwards at this point - the normal-form classification above all, which reads
        # get_last_eigenvalues()[0]/get_last_eigenvectors()[0] - expects the tracked one there. The
        # handler stays the authority (see _sync_tracking_to), but nothing else has to know that if
        # what the solve found is put back the way it was.
        keep=(prob._last_eigenvalues,prob._last_eigenvectors,
              prob._last_eigenvalues_m,prob._last_eigenvalues_k)
        try:
            evs,evecs=prob.solve_eigenproblem(self.neigen,self.tracked_eigen_shift,quiet=True)
            spectrum=[complex(v) for v in evs]
            modes=self._last_solved_modes(len(spectrum))
            if modes is None:
                # With no mode argument, solve_eigenproblem assembles at whatever the azimuthal m /
                # Cartesian k parameter holds - the TRACKED mode while a normal-mode tracker is
                # installed - and reports no per-eigenvalue modes for it. Saying so keeps a located
                # azimuthal bifurcation labelled with its m, which the single-value record it used to
                # carry did.
                tracked_mode=prob._critical_normal_mode(0)
                if tracked_mode is not None:
                    modes=[tracked_mode[1]]*len(spectrum)
            idx=self._index_of_tracked_eigenvalue(spectrum,evecs,crit,tracked_vec)
            if idx is None:
                self.log("The eigensolve at this bifurcation did not return the tracked eigenvalue "
                         "itself; it is listed first, ahead of the {:d} that were found. Raise the "
                         "eigenvalue count or move the shift closer to it.".format(len(spectrum)))
                spectrum.insert(0,crit)
                if modes is not None:
                    modes.insert(0,crit_modes[0] if crit_modes else 0.0)
                idx=0
            else:
                spectrum[idx]=crit
                if numpy.imag(crit)!=0:
                    # A Hopf (or azimuthal) tracker holds the whole PAIR on the axis, so the partner
                    # at conj(crit) is exactly there too and is snapped as well. Its solved copy's
                    # real part is rounding-error-sized with an ARBITRARY SIGN, and a positive one had
                    # the located bifurcation counting itself unstable - measured on an ODE Hopf,
                    # where the partner came back at +4e-17. Only the tracked one is marked, though:
                    # it is the value the point reports, and the pair is one mode.
                    #
                    # In the branch above instead of beside it: an inserted value shifts every index
                    # past it, and evecs would no longer line up with spectrum.
                    conj=self._index_of_tracked_eigenvalue(
                        spectrum,evecs,numpy.conj(crit),
                        None if tracked_vec is None else numpy.conj(tracked_vec),exclude=idx)
                    if conj is not None:
                        spectrum[conj]=complex(numpy.conj(crit))
            return self._phys_eig(crit),self._phys_eig(spectrum),modes,idx
        except Exception as e:
            self.log("Could not solve the eigenproblem at the tracked point ("+str(e).split("\n")[0]+
                     "); recording the critical eigenvalue only")
            return self._phys_eig(crit),None,crit_modes,None
        finally:
            (prob._last_eigenvalues,prob._last_eigenvectors,
             prob._last_eigenvalues_m,prob._last_eigenvalues_k)=keep

    def _add_locus_state(self):
        """Record the current point of a bifurcation locus, spectrum included.

        Everything about which eigenvalues are recorded here, and why the critical one is not
        re-solved, is in :py:meth:`_tracked_spectrum`.
        """
        crit,spectrum,modes,tracked=self._tracked_spectrum()
        self._add_current_state(eig_value=crit,eig_values=spectrum,eig_modes=modes,
                                tracked_eigenindex=tracked)

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
                           measured:bool=True,eig_modes=None,observables=None,
                           tracked_eigenindex:int | None=None):
        """Record the problem's current state as a new point of the current branch.

        ``eig_value`` overrides the eigenvalue that would be read from the problem, which is what
        re-recording an existing solution under a new branch needs: re-solving the eigenproblem there
        would turn an exact zero into a small nonzero value and the point would stop being a
        bifurcation. ``eig_values`` likewise overrides the recorded spectrum; when only ``eig_value``
        is given the spectrum is just that one value, since whatever the problem still holds belongs
        to a different solve. ``tracked_eigenindex`` says which entry of that spectrum a bifurcation
        tracker was holding on the axis, see :py:meth:`_tracked_spectrum`.
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
        # observables are passed in for an orbit, where one value per observable is an average over a
        # whole cycle and evaluating them here would record whichever phase the mesh happens to hold.
        obs=self.evaluate_observables() if observables is None else dict(observables)
        p=BifurcationGUISolutionPoint(self.get_bifurcation_parameter().value,obs,eig_value,state_file,self._state_step,
                                      param_values=self.current_parameter_values(),eig_values=eig_values,
                                      det_sign=det_sign,dparam_ds=dparam_ds,eig_modes=eig_modes,
                                      eig_settings=self.current_eigen_settings() if eig_values else None,
                                      tracked_eigenindex=tracked_eigenindex)
        self._record_mode_observables(p)
        # On a locus EVERY point has a zero real part, so classifying them all would run a normal-form
        # calculation per step for an answer already known: the tracked type.
        # ... and on an ORBIT it would be a bifurcation of the orbit, which the normal-form
        # calculation (a stationary construction, and one that solves an eigenproblem the orbit
        # handler refuses) cannot describe.
        if p.eig_value_Re==0 and self.classify_bifurcations and branch.kind not in (BRANCH_LOCUS,BRANCH_ORBIT):
            p.bifurcation_info=self._classify_current_point(p)
        if p.stability_source==STABILITY_EIGEN and p.eig_values:
            p.unstable_count=p.measured_unstable_count()
        if branch.kind==BRANCH_ORBIT:
            # The tangent belongs to the augmented orbit system, so writing it into the dump only
            # produces a file that reports a mismatched length on every reload. It goes into the
            # orbit's own sidecar instead, where it can be applied to the augmented system again.
            keep_conti=self.problem.continuation_data_in_states
            self.problem.continuation_data_in_states=False
            try:
                self.problem.save_state(state_file)
            finally:
                self.problem.continuation_data_in_states=keep_conti
        else:
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
        # An orbit's tangent is augmented for the same reason and just as unloadable, so it is dropped
        # here too; _sync_tracking_to reinstalls the handler and its direction from the sidecar.
        on_locus=branch is not None and branch.kind in (BRANCH_LOCUS,BRANCH_ORBIT)
        # Not "is a tracker active": an orbit handler reports no tracking mode, and loading a state
        # into the augmented orbit dof vector fails on the dof count instead.
        self._deactivate_any_augmentation()
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
        """Grab or release the selected point, so the selection keys move it along its branch.

        Grabbing switches into move-point mode as well: the flag does nothing outside it, and a grab
        that silently has no effect is indistinguishable from a broken one.
        """
        self._move_point=not self._move_point
        if self._move_point and self._mode!="mp":
            self.mode="mp"
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

    def delete_branch(self,branch=None)->int:
        """Remove a whole branch and every state file it owns. Returns how many points went.

        Defaults to the SELECTED branch, falling back to the current one, which is what the tree
        offers. Deleting the branch the problem is sitting on has to leave it somewhere, so a
        neighbouring branch's last point is loaded; the last remaining branch cannot be deleted, for
        the same reason delete_selected_point() refuses the last point.

        This throws the dumps away, so it is the one command in the diagram that cannot be undone by
        reloading. The confirmation for that lives in the GUI, not here: a script driving the
        controller has already decided.
        """
        if branch is None:
            branch=self.selected_branch if self.selected_branch is not None else self._get_current_branch()
        index=self._branch_index_of(branch)
        if len(self.branches)==1:
            raise RuntimeError("Cannot delete the last branch of the diagram")
        npoints=len(branch)
        was_current=branch is self.current_branch
        for p in branch:
            if p.statefile:
                self._remove_statefile(p.statefile)
        del self.branches[index]
        self.selected_point=None
        self.selected_branch=None
        if was_current:
            # The problem is holding a solution whose point has just been thrown away. Put it on a
            # neighbour's last point rather than leave current_point dangling - every later command
            # reads it, and a step from a state with no point would record onto a branch it is not on.
            neighbour=self.branches[min(index,len(self.branches)-1)]
            self.current_branch=neighbour
            self.current_point=None
            if len(neighbour):
                self.load_pt(neighbour[-1])
            self._tangs={}
            self.problem.reset_arc_length_parameters()
        self.log("Deleted a branch of {:d} point{:s}; {:d} branch{:s} left".format(
            npoints,"" if npoints==1 else "s",len(self.branches),"" if len(self.branches)==1 else "es"))
        self._changed()
        return npoints

    def split_branch(self,point=None,branch=None)->"BifurcationGUISolutionBranch":
        """Cut a branch in two at a point, which becomes the first point of the new half.

        For the ordinary way a diagram goes wrong: a continuation step lands on a different branch and
        everything from there on belongs somewhere else. Nothing is recomputed and no state file is
        touched - the points are the same objects, they just stop being drawn as one curve. Select the
        first point that belongs to the OTHER branch; splitting at the first point of a branch would
        leave nothing behind and is refused.
        """
        if point is None or branch is None:
            point,branch=self._ensure_selection()
        if point not in branch:
            raise RuntimeError("That point is not on that branch")
        index=branch.index(point)
        if index==0:
            raise RuntimeError("Splitting here would leave the first half empty. Select the first "
                               "point that belongs to the new branch, not the first point of this one.")
        tail=BifurcationGUISolutionBranch(list(branch[index:]),kind=branch.kind,
                                          continuation_parameter=branch.continuation_parameter,
                                          tracked_parameter=branch.tracked_parameter,
                                          bifurcation_type=branch.bifurcation_type)
        del branch.data[index:]
        # By identity: a branch is a UserList, so == compares its POINTS, and two branches holding the
        # same points would have index()/remove() act on whichever comes first.
        self.branches.insert(self._branch_index_of(branch)+1,tail)
        # The problem is still sitting where it was, so whichever half now holds the current point is
        # the branch a continuation would extend.
        if self.current_point is not None and self.current_point in tail:
            self.current_branch=tail
        self.selected_branch=tail
        self.selected_point=point
        self.reorder_branch_upon_point_insertion(branch,None)
        self.reorder_branch_upon_point_insertion(tail,None)
        self.log("Split off {:d} point{:s} into a new branch, {:d} left on the old one".format(
            len(tail),"" if len(tail)==1 else "s",len(branch)))
        self._changed()
        return tail

    def merge_branches(self,first=None,second=None)->"BifurcationGUISolutionBranch":
        """Join two branches that are really one curve, ordering them by which ends meet.

        Defaults to merging the SELECTED branch into the CURRENT one, which is what the tree offers.
        The joint is chosen from the four ways two curves can meet end to end, and the gap it leaves
        is reported: nothing here checks that the two are the same solution, only the user knows that,
        but a joint of the wrong length is worth seeing.

        Refused for branches that are not sections of the same thing - a different kind of branch, a
        different continuation parameter, or a different slice of parameter space. Joining those would
        produce a curve that is not a section of anything.
        """
        if first is None:
            first=self._get_current_branch()
        if second is None:
            second=self.selected_branch if self.selected_branch is not first else None
        if second is None or first is second:
            raise RuntimeError("Merging needs two different branches: select the other one in the "
                               "Branches tab, then merge it into the one you are on")
        if first.kind!=second.kind or first.continuation_parameter!=second.continuation_parameter \
                or first.tracked_parameter!=second.tracked_parameter:
            raise RuntimeError("These branches were not computed the same way (kind or parameters "
                               "differ), so they cannot be one curve")
        if first.slice_is_known() and second.slice_is_known() and first.slice_key()!=second.slice_key():
            raise RuntimeError("These branches sit in different slices of parameter space ("+
                               first.describe_slice()+" against "+second.describe_slice()+
                               "), so they are not one curve")

        def coord(p):
            c=p.get_coordinate(self.y_axis,xspec=self.x_axis)
            return numpy.array([float(c[0]),float(c[1])])
        xlim=self.view.get_xlim(); ylim=self.view.get_ylim()
        scale=numpy.array([1/max(abs(xlim[1]-xlim[0]),1e-30),1/max(abs(ylim[1]-ylim[0]),1e-30)])
        # The four ways two curves can be laid end to end. The tail of `order` is what the merged list
        # becomes, so reversing is expressed here rather than by mutating either branch.
        options=[("first[-1] to second[0]", lambda: list(first)+list(second), coord(first[-1]),coord(second[0])),
                 ("first[-1] to second[-1]",lambda: list(first)+list(reversed(second)),coord(first[-1]),coord(second[-1])),
                 ("first[0] to second[0]",  lambda: list(reversed(second))+list(first),coord(first[0]),coord(second[0])),
                 ("first[0] to second[-1]", lambda: list(second)+list(first),coord(first[0]),coord(second[-1]))]
        label,build,a,b=min(options,key=lambda o:numpy.linalg.norm((o[2]-o[3])*scale))
        gap=float(numpy.linalg.norm((a-b)*scale))
        first.data=build()
        del self.branches[self._branch_index_of(second)]
        if self.current_branch is second:
            self.current_branch=first
        self.selected_branch=first
        if self.selected_point is not None and self.selected_point not in first:
            self.selected_point=None
        self.reorder_branch_upon_point_insertion(first,None)
        self.log("Merged {:s}; the joint spans {:.3g} of the plot, and the branch now has {:d} points"
                 .format(label,gap,len(first)))
        self._changed()
        return first

    def _branch_index_of(self,branch)->int:
        """Position of a branch in the list, by identity rather than by equality."""
        for i,b in enumerate(self.branches):
            if b is branch:
                return i
        raise RuntimeError("That branch is not part of this diagram")

    def _remove_statefile(self,fname:str):
        # A missing dump must not abort a delete: the diagram is reloadable without it, and users
        # do clean out _states by hand between sessions.
        try:
            os.remove(fname)
        except OSError as e:
            self.log("Could not remove state file",fname,":",e)
        # An orbit point owns a companion holding the rest of its cycle. Removed here rather than at
        # the three call sites, and tolerantly: most points never had one.
        npzfile,dirfile=self._orbit_sidecar_paths(fname)
        try:
            os.remove(npzfile)
        except OSError:
            pass
        shutil.rmtree(dirfile,ignore_errors=True)

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
        if branch.kind==BRANCH_LOCUS:
            parts.append("locus of {:s} bifurcations".format(branch.bifurcation_type or "unclassified"))
            if branch.tracked_parameter is not None:
                parts.append("tracked in "+branch.tracked_parameter)
        elif branch.kind==BRANCH_ORBIT:
            # Which of the two averages produced the column matters to anyone comparing the file
            # against their own post-processing, so it is stated rather than left to be guessed.
            info=next((p.orbit_info for p in branch if p.orbit_info),None) or {}
            parts.append("periodic orbits, {:d} time steps, {:s} order {:d}".format(
                int(info.get("nT",0)),str(info.get("mode","?")),int(info.get("order",0))))
            parts.append("the observable columns are the average over the period ({:s}), with its "
                         "minimum and maximum alongside".format(
                             "exact time integral" if info.get("exact_average") else "sampled"))
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
        # Only the observables this branch actually has. Not every branch carries every name: the
        # mode observables appear the first time a mode scan runs, and an orbit's min/max only on
        # orbit branches - and get_coordinate raises KeyError rather than skipping, so exporting with
        # output_all_observables used to take the whole export down with it.
        have=branch[0].obs_values if len(branch) else {}
        yspecs=[sp for sp in obs_cols if sp[0]!=AXIS_OBSERVABLE or sp[1] in have]
        if branch.kind==BRANCH_LOCUS and branch.tracked_parameter is not None:
            yspecs=[parameter_axis(branch.tracked_parameter)]+yspecs
        elif branch.kind==BRANCH_ORBIT:
            # The exported average is only half of what an orbit point knows; the extremes follow each
            # observable it has them for, and the period goes last so it is easy to cut off.
            extra:list[AxisSpec]=[]
            for sp in list(yspecs):
                if sp[0]!=AXIS_OBSERVABLE:
                    continue
                lo,hi=orbit_band_names(sp[1])
                if lo in have and hi in have:
                    extra+=[observable_axis(lo),observable_axis(hi)]
            if ORBIT_T_KEY in have:
                extra.append(observable_axis(ORBIT_T_KEY))
            yspecs=yspecs+extra
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
        for pattern in ("branch*/*.txt","slice*/branch*/*.txt","*.dump","*.orbit.npz"):
            for g in glob.glob(os.path.join(odir,pattern)):
                os.remove(g)
        for g in glob.glob(os.path.join(odir,"*_orbit")):
            shutil.rmtree(g,ignore_errors=True)

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
                numpy.savetxt(os.path.join(bdir,fn),seg[:,:-1],header="\t".join([self.axis_label(export_x)]+column_names+_eigen_columns(self._eigen_unit,b.kind==BRANCH_ORBIT))+header_suffix)
            nbif=0
            for p in b:
                if p.eig_value_Re==0:
                    fn="bifurcation_{:03d}.txt".format(nbif)
                    pc=p.get_coordinate(export_y,with_s=False,with_eigen=True,xspec=export_x)
                    numpy.savetxt(os.path.join(bdir,fn),numpy.array([pc],ndmin=2),header="\t".join([self.axis_label(export_x)]+column_names+_eigen_columns(self._eigen_unit,b.kind==BRANCH_ORBIT))+header_suffix)
                    nbif+=1
                if p.tag>=0 and p.statefile is not None:
                    shutil.copy2(p.statefile, os.path.join(odir,"tag{:02d}.dump".format(p.tag)))
                    # The dump of an orbit point is one phase of the cycle; without its companion the
                    # exported state cannot reproduce the orbit it was tagged for.
                    src_npz,src_dir=self._orbit_sidecar_paths(p.statefile)
                    dst_npz,dst_dir=self._orbit_sidecar_paths(os.path.join(odir,"tag{:02d}.dump".format(p.tag)))
                    if os.path.exists(src_npz):
                        shutil.copy2(src_npz,dst_npz)
                    if os.path.isdir(src_dir):
                        shutil.rmtree(dst_dir,ignore_errors=True)
                        shutil.copytree(src_dir,dst_dir)
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
        if self.on_orbit():
            # Same reason as the locus, and one more: the augmented orbit vector has the same length
            # as the dof vector, so the probe below would silently run - perturbing every time point
            # of the orbit and paying one full sweep of the cycle per observable to do it. The drawn
            # direction comes from the two most recent points instead, via axis_tangent().
            self._tangs={}
            cp=self._get_current_point()
            prev=self._point_before_current()
            if prev is not None and self._last_ds:
                ds=float(self._last_ds)
                self._tangs={k:numpy.array([(cp.param_value-prev.param_value)/ds,
                                            (cp.obs_values[k]-prev.obs_values[k])/ds])
                             for k in cp.obs_values if k in prev.obs_values}
            cp._tangs=self._tangs.copy()
            return
        FD_eps=1e-6
        cp=self._get_current_point()
        backup,_=self.problem.get_current_dofs()
        if len(self.problem.get_arclength_dof_derivative_vector())==0:
            self._compute_continuation_tangent(cp)
        dp=self.problem.get_arc_length_parameter_derivative()
        ddof=numpy.array(self.problem.get_arclength_dof_derivative_vector())
        if len(ddof)==len(backup) and len(ddof)>0:
            self.problem.set_current_dofs(backup+FD_eps*ddof)
            po=self.evaluate_observables()
            for k in self._avail_observables:
                # Not every available observable is on every point: a mode observable
                # ("eigen/max Re [m=0]") only exists where a mode scan ran, and the list grows the
                # first time one does. Reading it off a point recorded before that raised KeyError out
                # of here, which - once a tangent existed at a loaded point to make this loop run at
                # all - took down the spectrum back-fill that had just loaded it.
                if k not in po or k not in cp.obs_values:
                    continue
                self._tangs[k]=numpy.array([dp,(po[k]-cp.obs_values[k])/FD_eps])
            cp._tangs=self._tangs.copy()
        else:
            # There is no continuation state to read a direction from, so there is no arrow to draw.
            # This is exactly the situation AT a located bifurcation: the normal-form calculation
            # deactivates bifurcation tracking, which resets oomph's continuation vectors, so the dof
            # derivative comes back EMPTY and dparameter/ds reads back as 1. Filling that in as [1,0]
            # drew a horizontal arrow of length ds at every located bifurcation - a direction of travel
            # that is not the branch's and not the one a branch switch would take. Elsewhere the
            # tangent is computed rather than left out; see _compute_continuation_tangent.
            self._tangs={}
            cp._tangs={}
        self._update_departure_tangents(cp,backup)
        self.problem.set_current_dofs(backup)

    def _solved_critical_eigenvector(self)->NPComplexArray | None:
        """The eigenvector of the mode nearest the imaginary axis in the last eigensolve.

        Nearest by REAL part, not by modulus: at a Hopf the critical pair sits at +-i*omega and is the
        furthest from the origin of anything in the spectrum, while every stable mode it is competing
        with has a real part well below zero. None when no eigensolve is at hand, which is what a point
        just loaded from a diagram looks like - state files do not carry eigenvectors.
        """
        try:
            evecs=self.problem.get_last_eigenvectors()
            evals=self.problem.get_last_eigenvalues()
        except Exception:
            return None
        if evecs is None or evals is None or len(evecs)==0 or len(evals)==0:
            return None
        idx=int(numpy.argmin(numpy.abs(numpy.real(numpy.asarray(evals,dtype=complex)))))
        return numpy.asarray(evecs[min(idx,len(evecs)-1)])

    def _ensure_normal_form(self,cp,allow_eigensolve:bool=False)->bool:
        """Complete a normal form that came back from a saved diagram, which carries no null vector.

        state.json keeps the COEFFICIENTS of a classification but not zeta, which is one entry per
        degree of freedom (see model._normal_form_to_state). A reloaded bifurcation therefore knows what
        it IS - which is what its label and the choice of how to leave it need - but cannot say where
        the other branch goes until zeta is back. That costs one eigensolve at the point, about what a
        continuation step costs, and only ever at a bifurcation whose classification was restored.

        zeta is normalised to unit length because that is the convention its coefficients were computed
        in: b2 and b3 scale with it, so a null vector of another length would predict another amplitude.
        The remaining freedom is its SIGN, which swaps the two arms of a pitchfork - and both of them
        are offered anyway.
        """
        bi=cp.bifurcation_info
        if bi is None:
            return False
        if bi.get("zeta") is not None:
            return True
        vec=self._solved_critical_eigenvector()
        if vec is None:
            if not allow_eigensolve:
                return False
            self.log("Solving the eigenproblem to recover the eigenvector of this restored "
                     +str(bi.get("type"))+", which was not saved with the diagram")
            try:
                self.problem.solve_eigenproblem(self.neigen,self.shift)
            except Exception as e:
                self.log("... which did not work: "+str(e).split("\n")[0])
                return False
            vec=self._solved_critical_eigenvector()
            if vec is None:
                return False
        try:
            zeta=_as_real_eigenvector(numpy.asarray(vec,dtype=complex),"The restored eigenvector")
        except RuntimeError as e:
            # Same rule as the rest of this method: a normal form that cannot be completed is a
            # False, not an exception - the caller logs and recomputes.
            self.log("The eigenvector for the saved classification is not real: "+str(e).split("\n")[0])
            return False
        if zeta.size!=self.problem.ndof():
            return False
        n=float(numpy.linalg.norm(zeta))
        if n<=0 or not numpy.isfinite(n):
            return False
        bi["zeta"]=zeta/n
        attach_normal_form_predictors(bi)
        return True

    def _compute_continuation_tangent(self,cp)->bool:
        """Ask the solver for the direction of travel at a point that was not reached by a step.

        A branch switch lands on the other branch with a Newton solve at a prescribed parameter offset,
        a transient leaves one wherever it settles, and the first point of a diagram is just a solve;
        none of them is an arclength step, so oomph holds no tangent afterwards and the point drew no
        arrow at all until the step after it had been taken. But the tangent is a property of the point,
        not of how the point was reached, so it can simply be computed: one solve of J dU = -dR/dp,
        normalised onto the arclength constraint, which is what the end of every step does anyway.

        Not AT a bifurcation - that Jacobian is singular, and it is the one place where having no
        tangent is the right answer (see :py:meth:`_update_tangents`).

        The arclength scaling is switched off for the call. With it on, oomph retunes theta^2 as part of
        computing the derivatives, so that the parameter takes its desired share of the arclength - 50%
        by default - and the tangent comes back normalised in a metric that is not the one in force. It
        showed as dparameter/ds = 1/sqrt(2) at every point regardless of the branch: on a pitchfork arm,
        where the true value is 0.758, that is a visibly wrong direction. Retuning belongs at the end of
        a step, which rescales ds to match (_recast_ds_after_metric_change); a direction merely being
        asked about must not move the metric under the step size.
        """
        if cp.eig_value_Re==0 or self.on_locus():
            return False
        if self.scale_arc_length:
            self.problem.set_arc_length_parameter(scale_arc_length=False)
        try:
            self.problem._compute_arclength_tangent(self._paramname)
        except Exception as e:
            # A singular or nearly singular Jacobian is the ordinary reason, and an arrow is not worth
            # interrupting anything for - this is exactly the state the code below already handles.
            self.log("Could not compute the continuation direction here: "+str(e).split("\n")[0])
            return False
        finally:
            if self.scale_arc_length:
                self.problem.set_arc_length_parameter(scale_arc_length=True)
        return True

    def _critical_null_vector(self,cp)->NPFloatArray | None:
        """The critical eigenvector at a located bifurcation, as a real dof vector.

        Taken from the normal form when there is one, since that is the vector the classification was
        built on, and from the last eigensolve otherwise.
        """
        bi=cp.bifurcation_info
        cand=bi.get("zeta") if bi is not None else None
        if cand is None:
            cand=self._solved_critical_eigenvector()
        if cand is None:
            return None
        # A Hopf's eigenvector is complex, and its REAL part is where the transient starts from:
        # perturb_by_eigenfunction adds Re(zeta*exp(i*omega*t)), which at t = 0 is exactly that.
        vec=numpy.real(numpy.asarray(cand,dtype=complex)).astype(float)
        return vec if vec.size==self.problem.ndof() else None

    def _arclength_unit_dof_direction(self,vec)->NPFloatArray | None:
        """Scale a dof direction to unit length in the arclength metric, at dparameter/ds = 0.

        The constraint is (dparameter/ds)^2 + theta^2*|dU/ds|^2 = 1 (see
        Problem._renormalise_continuation_tangent), so with the parameter held still the dof part has
        length 1/theta - not 1. Getting that wrong does not point the tangent anywhere else, it just
        makes ds buy a different stride than it says.
        """
        if vec is None:
            return None
        theta_sqr=self.problem.get_arc_length_theta_sqr()
        if not numpy.isfinite(theta_sqr) or theta_sqr<=0:
            theta_sqr=1.0
        n=float(numpy.linalg.norm(vec))
        if n<=0 or not numpy.isfinite(n):
            return None
        return numpy.asarray(vec,dtype=float)/(n*numpy.sqrt(theta_sqr))

    def _update_departure_tangents(self,cp,base):
        """Where the ways off this bifurcation go, as arrows in the same units as the arclength one.

        Two families, drawn alike but meaning quite different things:

        * a transcritical or a pitchfork HAS a second steady branch, so the arrows are the chord to the
          point :py:meth:`branch_switch` predicts, at the offset it would use. A pitchfork gets both of
          its arms - they differ in the sign of the amplitude at the same parameter offset, so drawing
          one of them would suggest the other does not exist.
        * a fold or a Hopf has no second steady branch through it. What leaves them is a transient, and
          what it leaves along is the critical eigenvector, so the arrows are +-zeta with NO component
          along the parameter: perturbing the solution does not move the parameter. At a fold that is
          also the branch's own tangent, which is what :py:meth:`step` continues along there.

        Everything is stored divided by |ds|, because that is what the plotter multiplies it by again -
        which is also what makes every arrow here scale with the step size in use.
        """
        cp._departure_tangs=[]
        cp._departure_kind=None
        bi=cp.bifurcation_info
        btype=bi.get("type") if bi is not None else None
        if btype is not None and bi.get("zeta") is None:
            # A classification restored from a diagram: it knows its type but not its null vector, and
            # every arrow below is built from that vector. The dofs are put back first because the
            # caller may have left them displaced by its finite-difference probe.
            self.problem.set_current_dofs(base)
            self._ensure_normal_form(cp,allow_eigensolve=self.recover_restored_normal_forms)
        scale=abs(self._last_ds) if self._last_ds else 1.0
        if btype in ("transcritical","pitchfork"):
            param_predictor=bi.get("param_predictor")
            perturbation_predictor=bi.get("perturbation_predictor")
            if param_predictor is None or perturbation_predictor is None:
                return
            eps=self.branch_switch_parameter_offset()
            try:
                offsets=[(float(param_predictor(sgn*eps)),
                          numpy.asarray(perturbation_predictor(sgn*eps),dtype=float)) for sgn in (1,-1)]
            except Exception as e:
                self.log("Could not evaluate the branch-switch prediction: "+repr(e))
                return
            kind="switch"
        elif btype in ("fold","hopf"):
            zeta=self._arclength_unit_dof_direction(self._critical_null_vector(cp))
            if zeta is None:
                return
            # One step's worth of arclength along the eigenvector and nothing along the parameter,
            # which is what the arclength constraint gives once dparameter/ds is set to zero.
            offsets=[(0.0,sgn*scale*zeta) for sgn in (1,-1)]
            kind="perturb"
        else:
            return
        for dpar,du in offsets:
            if du.size!=len(base):
                cp._departure_tangs=[]
                return
            try:
                self.problem.set_current_dofs(base+du)
                po=self.evaluate_observables()
            except Exception as e:
                # The prediction can land where the observables cannot be evaluated at all - a film
                # height through zero, say. An arrow is not worth aborting anything for.
                self.log("Could not draw the direction off the bifurcation: "+repr(e))
                cp._departure_tangs=[]
                return
            cp._departure_tangs.append(
                {k:numpy.array([dpar,po[k]-cp.obs_values[k]])/scale for k in self._avail_observables
                 if k in po and k in cp.obs_values})
        cp._departure_kind=kind

    def _prime_fold_continuation_tangent(self)->bool:
        """Hand the continuation the fold's own tangent, so that a step from an exact fold can be taken.

        At a fold dU/dparameter does not exist - that IS the fold - so the tangent oomph computes when
        it holds none (by solving J dU = -dR/dparameter) is a singular solve there, and a step from a
        located fold died inside oomph instead of going anywhere. The tangent is known analytically
        instead: the parameter turns around, so dparameter/ds = 0, and the branch runs along the null
        eigenvector. Priming those two is enough, because oomph takes Dof_current and Parameter_current
        from the present state at the start of every step and only the derivatives from before it.

        The sign is chosen to carry on the way the branch was already travelling. At a fold it is the
        PARAMETER that reverses while the solution keeps going, so it is the direction in the
        observable that has to be preserved - and it is taken against ds, since a negative ds steps
        backwards along whatever tangent is stored.
        """
        cp=self._get_current_point()
        direction=self._arclength_unit_dof_direction(self._critical_null_vector(cp))
        if direction is None:
            return False
        sign=1.0
        incoming=self.axis_tangent()
        dep=cp._departure_tangs[0] if cp._departure_tangs else None
        if incoming is not None and dep is not None and self._current_observable in dep:
            if dep[self._current_observable][1]*incoming[1]<0:
                sign=-1.0
        if self._last_ds is not None and self._last_ds<0:
            sign=-sign
        # _set_dof_direction_arclength resets every arclength parameter, theta^2 among them, so it is
        # put back afterwards - the step retunes it itself, as it does for an ordinary step.
        theta_sqr=self.problem.get_arc_length_theta_sqr()
        self.problem._set_dof_direction_arclength(list(sign*direction))
        self.problem._set_arc_length_parameter_derivative(0.0)
        self.problem._set_arc_length_theta_sqr(theta_sqr)
        self.log("Continuing around the fold along its null vector (dparameter/ds = 0)")
        return True

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

        That argument needs a TEST FUNCTION at every point walked across, and propagation stops at a
        point that has none - no det_sign and no dparam_ds. Without one there is nothing saying an
        eigenvalue did not cross in between, and carrying the count on regardless would not be an
        inference, it would be an assumption painted in the same colour as a measurement. Two kinds of
        point are in that position: the ones a DEFLATED SCAN records (it steps the parameter with no
        arclength control and no test function, in exactly the regions where branches appear and
        disappear), and the ones where an eigensolve returned nothing at all. Both are left unknown,
        and "Compute the eigenvalues along this branch" is what fills them in.
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
                    if p.det_sign is None and p.dparam_ds is None:
                        # No test function here, so the walk cannot see past this point. Not merely
                        # "leave this one unknown": everything beyond it is behind the same blind spot.
                        count=None
                        last_det=None
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
        if point.eig_value_Re==0:
            # A located bifurcation's leading eigenvalue is the tracker's exact zero, and that exact
            # zero is what makes the point a bifurcation everywhere in the diagram. Re-solving it from
            # the dump - with no tracker installed, at a state whose Jacobian is singular - would
            # replace it by a rounding-error-sized number and silently demote the point to an ordinary
            # one, losing its classification and its departure arrows with it. Its spectrum is
            # recorded when the point is made, see _tracked_spectrum.
            self.log("Not recomputing the spectrum at the bifurcation at "+self._get_paramname_str()+
                     " = {:.6g}: that would overwrite the tracked exact zero".format(point.param_value))
            return False
        restore=self.current_point
        try:
            self.load_pt(point)
            if point.orbit_info is not None:
                # An orbit's stability is its Floquet multipliers - solve_eigenproblem refuses while
                # the handler is installed, and the answer it would give is not the one being asked
                # for. load_pt has just put the whole cycle back, so they can be computed here.
                exps,mults=self._orbit_floquet()
                if not exps:
                    return False
                point.eig_values=[complex(v) for v in exps]
                point.eig_modes=None
                point.eig_settings=None
                point.floquet=mults
                point.eig_value_Re=numpy.real(exps[0])
                point.eig_value_Im=numpy.imag(exps[0])
                point.stability_source=STABILITY_EIGEN
                point.unstable_count=self.orbit_unstable_count(mults)
                return True
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

    def _classify_current_point(self,point=None)->dict | None:
        """Normal form of the bifurcation the problem is sitting at, or None if it cannot be had.

        Non-fatal on purpose. This is computed while a bifurcation is being recorded, and a normal-form
        calculation that fails must not take the located point down with it - losing the bifurcation is a
        worse outcome than not knowing which kind it is.
        """
        from ...generic.bifurcation_tools import NormalFormCalculator
        # ``point`` explicitly, not self.current_point: this runs from inside _add_current_state, where
        # the point being classified has not been installed as the current one yet, and the previous
        # point's modes say nothing about this bifurcation.
        try:
            mode=self.problem._critical_normal_mode(0)
        except Exception:
            # A problem that cannot answer is not a reason to refuse, and a controller built without a
            # real one at all (the stub-based tests) has no modes by construction.
            mode=None
        if mode is None:
            mode=self._point_normal_mode(point)
        if mode is not None:
            # Said plainly rather than let the refusal below come back as "Could not compute the
            # normal form": there is nothing wrong here to fix, and the point is a perfectly good
            # bifurcation - it just has no normal form in these degrees of freedom.
            self.log("This is a bifurcation of the {:s} = {:g}, whose branch is not representable in "
                     "this problem's degrees of freedom, so no normal form is computed for it. The "
                     "bifurcation itself is located and can be continued in a second parameter."
                     .format("azimuthal mode m" if mode[0]=="m" else "Cartesian mode k",mode[1]))
            return None
        try:
            if self.problem.get_bifurcation_tracking_mode()=="":
                # get_normal_form reads the critical eigenpair from the last eigensolve. While tracking
                # is active that is already the critical one AND solve_eigenproblem would refuse, so this
                # only runs when classifying after the fact.
                self.problem.solve_eigenproblem(self.neigen,self.shift)
            return NormalFormCalculator(self.problem).get_normal_form(
                self._get_paramname_str(),assume=self._fold_ruled_out_by_the_branch())
        except Exception as e:
            self.log("Could not compute the normal form of this bifurcation: "+repr(e))
            return None

    def _fold_ruled_out_by_the_branch(self)->str | None:
        """``"branch_point"`` when the branch demonstrably ran THROUGH the parameter value we are at.

        The normal form decides fold against branch point from a single number, the projection of
        dR/dparameter on the left null vector, and near a second nearly-critical eigenvalue that number
        is the least reliable one in the calculation. The branch itself answers the same question
        without any of that: a fold is where the parameter turns around, so both points either side of
        one lie on the SAME side of its parameter value, while a branch point sits strictly between
        them. Only used when it applies - away from a fresh continuation step this returns None and the
        normal form decides alone.
        """
        located=float(self.get_bifurcation_parameter().value)
        prev=self._point_before_current()
        cur=self.current_point
        if prev is None or cur is None:
            return None
        evs=self.problem.get_last_eigenvalues()
        if evs is not None and len(evs)>0 and abs(numpy.imag(evs[0]))>1e-8:
            return None   # a Hopf also passes straight through; this test says nothing about one
        lo,hi=sorted((float(prev.param_value),float(cur.param_value)))
        if hi-lo<=0:
            return None
        margin=0.01*(hi-lo)
        if lo+margin<located<hi-margin:
            return "branch_point"
        return None

    def branch_switch_parameter_offset(self)->float:
        """The parameter offset a branch switch steps off by, taken from the step size in use.

        ds is an ARCLENGTH step, so what it buys in the parameter is ds*|dparameter/ds|. The tangent at
        a located bifurcation is gone (see :py:meth:`_update_tangents`), so the one recorded at the
        point before it is used. Where there is none to ask, this falls back to what it always was:
        :py:attr:`branch_switch_offset` times the parameter - 2% of it, which on a diagram spanning 4%
        of the parameter overshoots every branch in it, and did.
        """
        prev=self._point_before_current()
        tang=None
        if prev is not None and prev._tangs:
            tang=prev._tangs.get(self._current_observable)
            if tang is None:
                tang=next(iter(prev._tangs.values()))
        if tang is not None and self._last_ds:
            eps=abs(float(self._last_ds)*float(tang[0]))
            if eps>0:
                return eps
        return self.branch_switch_offset*max(abs(float(self.get_bifurcation_parameter().value)),1.0)

    def _point_normal_mode(self,point)->"tuple[str,float] | None":
        """``("m", value)`` / ``("k", value)`` of the mode a point's bifurcation belongs to, or None.

        Read off the POINT, not off the problem: by the time a branch switch is asked for, the tracker
        that knew the mode has been deactivated and the last eigensolve may well have been a base-mode
        one, so the problem no longer remembers. The point does - see _tracked_spectrum.
        """
        # The point's own record first, so that this needs nothing of the problem for a point that
        # carries no modes - which is every point of a problem that has none.
        if point is None or not point.eig_modes:
            return None
        kind=self.normal_mode_kind()
        if not kind:
            return None
        i=point.tracked_eigenindex if point.tracked_eigenindex is not None else 0
        if not 0<=i<len(point.eig_modes):
            return None
        mode=float(point.eig_modes[i])
        nontrivial=mode!=0 if kind=="m" else abs(mode)>1e-7
        return (kind,mode) if nontrivial else None

    def branch_switch(self,offset:float | None=None,direction:int | None=None):
        """Step onto the other branch through the current bifurcation and record it as a new branch.

        The numerics live on :py:meth:`~pyoomph.generic.problem.Problem.switch_branch`, so the same
        manoeuvre is available to a plain script; what is here is the part that is about the diagram -
        which point we are at, opening a branch for the result, and the step size to carry on with.

        Both the offset and the direction default to what the continuation is set to: the offset to the
        parameter step the current ds buys (:py:meth:`branch_switch_parameter_offset`) and the direction
        to the sign of ds. So the keys that steer a sweep steer this too - ``/`` picks which side of a
        transcritical, or which arm of a pitchfork, is tried first, and ``+``/``-`` how far off the
        bifurcation the switch aims.
        """
        if self.on_locus():
            raise RuntimeError("Branch switching applies to an ordinary branch, not to a bifurcation locus")
        cp=self._get_current_point()
        if cp.eig_value_Re!=0:
            raise RuntimeError("Can only switch branches at bifurcations")
        # Before the normal form is computed, not after: at a normal-mode bifurcation the calculation
        # would go through and hand back a plausible fold or pitchfork built from the wrong Hessian.
        found=self._point_normal_mode(cp)
        if found is not None:
            self.problem._refuse_at_normal_mode_bifurcation("Branch switching",mode=found)
        if cp.bifurcation_info is None:
            # Computed here rather than sending the user away to set a flag and redo the run: the problem
            # is sitting at the bifurcation, which is all the normal form needs. This is also what makes
            # switching work at a point loaded from a diagram recorded without classification.
            self.log("This bifurcation was not classified; computing its normal form now")
            cp.bifurcation_info=self._classify_current_point(cp)
        if cp.bifurcation_info is None:
            raise RuntimeError("Cannot switch branches: the normal form of this bifurcation could not be "
                               "computed, so there is no prediction for where the other branch goes. The "
                               "reason is in the log above.")
        # A Hopf has no second steady branch either, but it does have somewhere to go: the periodic
        # orbit it sheds. Dispatched here so that the one key that means "leave this bifurcation
        # sideways" does the right thing at every kind of bifurcation.
        if cp.bifurcation_info.get("type")=="hopf":
            refusal=self.orbit_can_be_started()
            if refusal is not None:
                self.log("A Hopf sheds a periodic orbit rather than a second steady branch, but that "
                         "cannot be started here. "+refusal)
                return False
            self.log("A Hopf has no second steady branch - switching onto the periodic orbit instead")
            return bool(self.switch_to_orbit())
        # Before any attempt to complete the normal form below: what rules a fold out is its TYPE, and
        # recovering predictors for a switch that is about to be refused would be work done to no end.
        if cp.bifurcation_info.get("type")=="fold":
            self.log("A fold has only one branch through it - there is nothing to switch to")
            return False
        if cp.bifurcation_info.get("param_predictor") is None:
            # Read back from a diagram: the coefficients are there and only the null vector is missing,
            # so one eigensolve completes it - much less than the normal form costs to compute again.
            if not self._ensure_normal_form(cp,allow_eigensolve=True):
                self.log("The saved classification could not be completed; recomputing the normal form")
                cp.bifurcation_info=self._classify_current_point(cp)
            if cp.bifurcation_info is None or cp.bifurcation_info.get("param_predictor") is None:
                raise RuntimeError("Cannot switch branches: this bifurcation's normal form has no "
                                   "prediction for where the other branch goes. The reason is in the "
                                   "log above.")

        if offset is None:
            offset=self.branch_switch_parameter_offset()
        if direction is None:
            direction=-1 if (self._last_ds is not None and self._last_ds<0) else 1
        self.log("Switching branches with a parameter offset of {:.3g} in direction {:+d}".format(
            offset,direction))
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
        # The offset is 4*ds by switch_branch's contract, and its SIGN is the useful part: for a
        # pitchfork it says which side of the parameter the two arms turned out to be on, which is the
        # sub- against supercritical question the normal form's b3 is too shaky to answer.
        self.log("Switched onto the other {:s} branch at {:s} = {:.6g} (offset {:+.3g}); ds set to {:.3g}"
                 .format(str(cp.bifurcation_info.get("type")),self._get_paramname_str(),
                         self.get_bifurcation_parameter().value,4*ds,self._last_ds))
        self.problem.solve_eigenproblem(self.neigen,self.shift)
        self._new_branch()
        self._tangs={}
        self._add_current_state()
        self._update_tangents()
        self._mode="al"
        self._changed()
        return True

    #: How many growth times of the mode the transient runs for at most, and how many steps one growth
    #: time is resolved with. The step size matters more than the total: see transient_leave_branch.
    transient_growth_times=100
    transient_steps_per_growth=20
    #: Growth of the distance from the branch, per monitoring interval, below which the unstable mode
    #: is no longer what is driving the solution. One interval is one growth time, so a mode still
    #: growing at its own rate multiplies the distance by e = 2.72; well under that means the departure
    #: has gone nonlinear and the time step no longer has to resolve the mode.
    transient_stall_growth=1.6
    #: What the cap on dt is multiplied by per monitoring interval once it has departed. The cap never
    #: passes the interval itself, so the distance keeps being sampled.
    transient_relax_factor=2.0
    #: Stop once one monitoring interval moves the solution less than this, relative to how far it has
    #: travelled: it has arrived somewhere and the remaining growth times would only confirm it.
    transient_settle_tolerance=1e-3

    def transient_leave_branch(self,eigenindex=0):
        """Perturb along an eigenfunction and integrate until the solution settles somewhere else.

        While the solution is still NEAR the branch, the time step must stay well below the growth time
        of the mode being followed. A fully implicit step much longer than 1/lambda does not amplify an
        unstable mode, it damps it - BDF2's amplification factor tends to zero as lambda*dt grows - and
        the adaptive stepper walks straight into that: the solution sits near a stationary state, so the
        temporal error is tiny, so dt is doubled every step. Measured on a thin-film branch point, dt
        reached 500 growth times and the run came back to the very solution it was told to leave, with
        the same eigenvalue to six digits.

        That argument only holds while the mode is what is growing. Once the solution is far from the
        branch the dynamics are the nonlinear approach to wherever it is going, the temporal error
        controller is a sound judge of it again, and holding dt at the old cap only makes the departure
        take a long time in small steps. So the distance from the branch is watched, and the cap is
        relaxed once it is clear the solution has left - and the run stops early when that distance
        stops changing, which is the solution having arrived. A limit cycle never stops changing, so
        that case runs to the full time as before.

        The perturbation comes from Problem.perturb_by_eigenfunction, which scales it to a residual
        rather than to a fixed multiple of an eigenvector of arbitrary norm, and fills in the history
        dofs so the first steps do not have to recover from an impulsive start.
        """
        if self.on_orbit():
            raise RuntimeError("A transient departure needs to time-step the problem, which cannot be "
                               "done while an orbit is installed. Load a stationary point first.")
        self._status("LEAVING BRANCH TRANSIENTLY")
        evecs=self.problem.get_last_eigenvectors()
        if evecs is None or len(evecs)<=eigenindex:
            # This perturbs the solution along an eigenvector, so it needs one, and there are two
            # ordinary ways not to have it: quick mode never computed one, and loading a point drops
            # the eigendata. Solving here beats failing several frames deep in the perturbation.
            self.log("Solving the eigenproblem, which leaving a branch transiently needs")
            self.problem.solve_eigenproblem(self.neigen,self.shift)
        cp=self._get_current_point()
        # AT a bifurcation the rate is zero and there is no growth time to speak of; the floor is what
        # the previous implementation used and keeps the step finite. The recorded eigenvalues are
        # PHYSICAL rates (see _phys_eig), so they go back to the problem's own time scale before being
        # turned into a time - dividing the temporal scaling by a physical rate is not a time at all
        # unless that scaling happens to be one second.
        rate=max(1e-4,numpy.sqrt(cp.eig_value_Re**2+cp.eig_value_Im**2)/(self._eigen_mult or 1.0))
        TS=self.problem.get_scaling("temporal")
        growth_time=TS/rate

        self.problem.deactivate_bifurcation_tracking()
        self.problem.reset_arc_length_parameters()
        self.problem.set_current_time(0)

        on_branch=numpy.array(self.problem.get_current_dofs()[0])
        dt=self.problem.perturb_by_eigenfunction(dt=growth_time/self.transient_steps_per_growth,
                                                 eigenmode=eigenindex)
        kick=float(numpy.linalg.norm(numpy.array(self.problem.get_current_dofs()[0])-on_branch))
        if kick<=0 or not numpy.isfinite(kick):
            # Nothing to measure a departure against. Fall back to the fixed cap for the whole run,
            # which is what this method did before the distance was watched at all.
            kick=None

        growth_nd=float(growth_time/TS)
        maxstep_nd=growth_nd/5
        previous=numpy.array(self.problem.get_current_dofs()[0])
        travelled=kick
        stalled=0
        relaxed=False
        reason="ran the full {:g} growth times".format(self.transient_growth_times)
        for interval in range(1,int(self.transient_growth_times)+1):
            try:
                self.problem.run(interval*growth_nd*TS,startstep=dt,maxstep=maxstep_nd*TS,
                                 temporal_error=1,outstep=False,do_not_set_IC=True)
            except Exception as e:
                # There is not always somewhere else to go: perturbed the other way, a fold's own
                # normal form u' = -u^2 runs off to infinity in finite time, and what comes back is the
                # time stepper failing rather than a solution. Since this is now what the default action
                # does at a fold, it has to report that and put the problem back, not throw.
                self.log("The transient did not reach anything: "+str(e).split("\n")[0])
                self.log("The solution ran away after {:.3g} growth times. Nothing was recorded; back "
                         "at the point it started from.".format(
                             self.problem.get_current_time(as_float=True,dimensional=False)/growth_nd))
                self.load_pt(cp)
                self._changed()
                return False
            dt=None  # carry on with the step the adaptive run worked its way up to
            now=numpy.array(self.problem.get_current_dofs()[0])
            moved=float(numpy.linalg.norm(now-previous))
            previous=now
            if self._abort_requested:
                self._abort_requested=False
                reason="aborted after {:d} growth times".format(interval)
                break
            if kick is None:
                continue
            before,travelled=travelled,float(numpy.linalg.norm(now-on_branch))
            if moved<self.transient_settle_tolerance*max(travelled,kick):
                reason=("came back to the branch it was told to leave after {:d} growth times"
                        if travelled<kick else "settled after {:d} growth times").format(interval)
                break
            # The distance is measured against where the growth of an undisturbed mode would put it,
            # NOT against a fixed multiple of the perturbation: perturb_by_eigenfunction scales the kick
            # to a residual, and on a small problem that can already be most of the way to the new
            # state, so "has it travelled 100 kicks yet" is a question that is never answered yes.
            if before>0 and travelled<self.transient_stall_growth*before:
                stalled+=1
            else:
                stalled=0
            # Two intervals, not one: on an oscillatory mode the distance stands still twice per period
            # while the mode is perfectly healthy, and one quiet interval would let dt go at the worst
            # possible moment.
            if stalled>=2:
                if not relaxed:
                    relaxed=True
                    self.log("The departure has gone nonlinear after {:d} growth times; the time step "
                             "no longer has to resolve the mode".format(interval))
                # Never past the monitoring interval itself, or the distance would stop being sampled.
                maxstep_nd=min(growth_nd,maxstep_nd*self.transient_relax_factor)
        self.problem.set_current_time(0)
        try:
            self.problem.solve(max_newton_iterations=20)
            self.problem.solve_eigenproblem(self.neigen,self.shift)
        except Exception as e:
            # Where the transient stopped is not always a steady state - a limit cycle is the ordinary
            # case, and the Newton solve from a point on it need not converge to anything.
            self.log("The transient arrived somewhere, but no steady state could be solved for there: "
                     +str(e).split("\n")[0])
            self.log("Back at the point it started from; try leaving with a different mode, or follow "
                     "the orbit in time instead.")
            self.load_pt(cp)
            self._changed()
            return False
        self._new_branch()
        self._tangs={}
        self._add_current_state()
        self._update_tangents()
        self._mode="al"
        self.log("Transient leaving "+reason+" (one growth time is "+str(growth_time)+")")
        return True


    def _outward_extension_scoord(self,newp,origin,pbase,xbase,ybase,sbase,xn,yn)->float | None:
        """The s of a point that continues the branch OUTWARDS from one of its ends, or None.

        A continuation step taken from an end point is the one insertion whose place is known before any
        geometry is measured: it belongs beyond that end, because that is where the continuation went.
        The search below decides instead by which insertion makes the path shortest, and a branch that
        curves back towards its own beginning can be shorter with the new point at the OTHER end - an
        isola about to close puts it there, and then the order the points were computed in is gone,
        along with the stability segments, the splines and the tangent that are read off it.

        Only outwards. Reversing ds at an end and stepping back over the branch is a legitimate thing to
        do, and the point then really does belong between two others, which is what the search is for.
        The two are told apart on the last leg of the branch: a step that does not point back along it
        is continuing the direction of travel.
        """
        if origin is None or origin is newp or len(pbase)<2:
            return None
        if origin is pbase[-1]:
            leg=(xbase[-1]-xbase[-2],ybase[-1]-ybase[-2])
            step=(xn-xbase[-1],yn-ybase[-1])
            if leg[0]*step[0]+leg[1]*step[1]>0:
                return sbase[-1]+0.5*(sbase[-1]-sbase[-2])
        elif origin is pbase[0]:
            leg=(xbase[0]-xbase[1],ybase[0]-ybase[1])
            step=(xn-xbase[0],yn-ybase[0])
            if leg[0]*step[0]+leg[1]*step[1]>0:
                return sbase[0]-0.5*(sbase[1]-sbase[0])
        return None

    def reorder_branch_upon_point_insertion(self,branch:BifurcationGUISolutionBranch,newp:BifurcationGUISolutionPoint | None,
                                            origin:BifurcationGUISolutionPoint | None=None):
        """Put ``newp`` where it belongs along ``branch`` and renormalise every s.

        ``origin`` is the point the new one was computed FROM, when there is one. It is what makes a
        step off the end of the branch extend it rather than be placed by the search - see
        :py:meth:`_outward_extension_scoord`.
        """
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
        pbase=[]
        for p in branch:
            if p==newp:
                continue
            pbase.append(p)
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
            outward=self._outward_extension_scoord(newp,origin,pbase,xbase,ybase,sbase,xn,yn)
            if outward is not None:
                newp.scoord=outward
                branch.sort(key=lambda p : p.scoord)
                return
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

    #: How far ahead of the first point a fresh diagram opens, in units of the initial ds, and how
    #: far behind it. Ahead, because that is where Step is about to go and roughly what a multistep
    #: sweep covers before it hits the border; a little behind, so that the point is not glued to the
    #: edge and a Reverse has somewhere to land.
    initial_view_ds_ahead=10.0
    initial_view_ds_behind=2.0

    def initial_view_box(self)->tuple[float,float,float,float]:
        """The window a fresh diagram opens on, as (xmin, xmax, ymin, ymax).

        Along the axis carrying the continuation parameter the range is set from ds, so that the first
        several steps and the direction arrow are visible without zooming. The other axis keeps the
        tiny box: nothing is known about the observable's scale before a step has been taken, and the
        view only ever grows as points arrive, so guessing there would only be wrong more expensively.

        An explicit initial view set on the GUI wins over all of this.
        """
        cp=self._get_current_point().get_coordinate(self.y_axis,xspec=self.x_axis)
        box=[cp[0]-1e-4,cp[0]+1e-4,cp[1]-1e-4,cp[1]+1e-4]
        ds=float(self._last_ds or 0.0)
        if ds!=0.0 and isinstance(self._paramname,str):
            pax=parameter_axis(self._paramname)
            ahead,behind=ds*self.initial_view_ds_ahead,-ds*self.initial_view_ds_behind
            if as_axis(self.x_axis)==pax:
                box[0],box[1]=min(cp[0]+ahead,cp[0]+behind),max(cp[0]+ahead,cp[0]+behind)
            elif as_axis(self.y_axis)==pax:
                box[2],box[3]=min(cp[1]+ahead,cp[1]+behind),max(cp[1]+ahead,cp[1]+behind)
        return (box[0],box[1],box[2],box[3])

    def plotted_tangent(self)->"NPFloatArray | None":
        """Direction the continuation arrow should show, per unit ds, in the current axes.

        Normally the arclength tangent recorded for the plotted observable. On a FRESH branch there is
        none - one point, no step taken - and the arrow was simply missing at the moment it is most
        wanted, which is when the user is deciding whether to press Step or Reverse first. A first
        continuation step moves the parameter and, to first order, nothing else, so a unit vector
        along whichever axis carries the continuation parameter is the honest answer: multiplied by
        ds it draws (ds, 0) on the ordinary diagram.

        The fallback is deliberately restricted to a branch with fewer than two points. At a LOCATED
        BIFURCATION `_tangs` is empty too, on purpose (see _update_tangents), and the arrows worth
        drawing there are the departure directions the plotter draws itself - one more arrow along the
        parameter axis would say something untrue about a point that has no arclength tangent.
        """
        key=self.y_axis[1] if self.y_axis[0]==AXIS_OBSERVABLE else None
        if key is not None:
            tang=self._tangs.get(key)
            if tang is not None:
                return tang
        branch=self.current_branch
        if branch is not None and len(branch)>=2:
            return None
        if not isinstance(self._paramname,str):
            return None
        pax=parameter_axis(self._paramname)
        if as_axis(self.x_axis)==pax:
            return numpy.array([1.0,0.0])
        if as_axis(self.y_axis)==pax:
            return numpy.array([0.0,1.0])
        return None

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
        orbit=self.on_orbit()
        quick=self.quick_mode and not (locus or orbit)
        self._status("FOLLOWING THE BIFURCATION" if locus else
                     ("CONTINUING THE ORBIT" if orbit else
                      ("QUICK STEPPING (no eigensolve)" if quick else "ARCLENGTH STEPPING")))
        # A step from an exact fold has no tangent to start from and cannot compute one there; see
        # _prime_fold_continuation_tangent. Only a fold: at a branch point or a Hopf the ordinary
        # restart works and continues the branch we came in on, which is a step worth having.
        if (not locus and not orbit and origin.bifurcation_info is not None
                and origin.bifurcation_info.get("type")=="fold"
                and len(self.problem.get_arclength_dof_derivative_vector())==0):
            if not self._prime_fold_continuation_tangent():
                raise RuntimeError(
                    "Cannot continue from this fold: it needs the null eigenvector to step along, and "
                    "none is available here (a point loaded from a diagram carries no eigenvectors). "
                    "Solve the eigenproblem at this point first, or leave the branch transiently.")
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
        elif orbit:
            # No eigensolve: the problem refuses one while the orbit handler is installed, and the
            # answer there is the Floquet multipliers, which _add_orbit_state computes.
            self._add_orbit_state()
        else:
            self.problem.solve_eigenproblem(self.neigen,self.shift,
                                            **(self._mode_kwargs() if self.compute_modes_during_sweep else {}))
            self._add_current_state()

        if not orbit:
            # Adapting renumbers the equations, which pulls the augmented orbit dof vector out from
            # under the handler and invalidates every set of time blocks already stored on the branch.
            self._adapt_after_step()
        # origin is where this step started: a step off the end of the branch extends it, rather than
        # being placed among the points already there.
        self.reorder_branch_upon_point_insertion(self._get_current_branch(),self._get_current_point(),
                                                 origin=origin)
        self._last_ds=ds
        self._update_tangents()
        if origin._tangs is None or len(origin._tangs)==0:
            origin._tangs=self._get_current_point()._tangs.copy()

        return ds

    # ------------------------------------------------------------------ deflation

    def deflation_perturbation_value(self)->float:
        """The perturbation amplitude to use, resolving None to a value read off the current solution.

        A tenth of the solution's RMS magnitude: far enough off the known solution for the deflation
        to be finite, small enough to stay in a basin. When the state is exactly zero - the trivial
        branch, before anything has bifurcated off it - there is no length in the problem at all and
        nothing can be derived; it falls back to 0.5, which is the historical value, and the tab says
        so. That is the one case where the number has to be set by hand, and it is worth setting: it
        is the scale the whole search is measured in.
        """
        if self.deflation_perturbation is not None:
            return abs(float(self.deflation_perturbation))
        try:
            U=numpy.asarray(self.problem.get_current_dofs()[0],dtype=float)
        except Exception:
            return 0.5
        rms=float(numpy.sqrt(numpy.mean(U*U))) if len(U) else 0.0
        return 0.1*rms if rms>0.0 else 0.5

    def eigenvector_is_essentially_real(self,index:int=0,tol:float=1e-6)->bool:
        """Whether eigenvector ``index`` has no imaginary part worth plotting.

        A real eigenvalue's eigenvector is determined only up to a global phase, and a COMPLEX PETSc
        build has no reason to return it with that phase set to zero. What comes back is exp(i*phi)*w
        for a real w, so Re v = cos(phi)*w and Im v = sin(phi)*w are the SAME FUNCTION at two
        amplitudes - and an autoscaled plot of the two is the same picture twice, which is exactly how
        this was noticed.

        Tested on the eigenVECTOR rather than on the eigenvalue: "is the imaginary part of lambda
        small" needs a scale to be small against, and near a fold both parts of lambda go to zero
        together. Whether Re v and Im v are parallel needs no such scale, and it is the question the
        plot actually asks - it is about whether there are two functions to draw or one.
        """
        evs=self.problem.get_last_eigenvectors()
        if evs is None or len(evs)<=index:
            return True     # nothing to draw either way
        v=numpy.asarray(evs[index])
        if not numpy.iscomplexobj(v):
            return True
        re,im=numpy.real(v),numpy.imag(v)
        nre,nim=float(numpy.linalg.norm(re)),float(numpy.linalg.norm(im))
        if nre<=0.0 and nim<=0.0:
            return True
        # Either part alone being negligible is the phase being 0 or pi/2; i*w is as real a mode as w.
        if nim<=tol*nre or nre<=tol*nim:
            return True
        return abs(float(numpy.dot(re,im)))/(nre*nim)>=1.0-tol

    # ------------------------------------------------------------------ orbit stability

    def _orbit_floquet(self)->"tuple[list[complex],list[complex]]":
        """(Floquet exponents as physical rates, raw multipliers) of the installed orbit.

        The exponents are ``log(mu)/T``, and they are what goes into the point's spectrum: the whole
        stability machinery is written against a real part, and ``Re(log(mu)/T) > 0`` is exactly
        ``|mu| > 1``. The multipliers are kept alongside because they, and not the exponents, say what
        KIND of bifurcation is approaching - a multiplier leaving the unit circle through -1 is a
        period doubling and a complex pair a torus, and the two have the same exponent real part.
        """
        orbit=self._require_orbit()
        if not self.floquet_enabled:
            return [],[]
        handler=self._orbit_handler()
        assert handler is not None
        if not handler.is_floquet_mode():
            self.log("No Floquet multipliers in '{:s}' mode: it carries no degree of freedom at the "
                     "end of the period. Use collocation or floquet for the stability of an orbit."
                     .format(str(orbit.mode)))
            return [],[]
        # get_floquet_multipliers OVERWRITES the problem's last eigenvalues and eigenvectors with the
        # multipliers. Everything that reads them afterwards - _add_current_state's own branch,
        # critical_eigenindex, _sync_tracking_to, the eigenfunction panes - would take multipliers for
        # eigenvalues, so what was there is put back.
        keep=(self.problem._last_eigenvalues,self.problem._last_eigenvectors,
              self.problem._last_eigenvalues_m,self.problem._last_eigenvalues_k)
        try:
            self._status("FLOQUET MULTIPLIERS")
            # ignore_periodic_unity=False on purpose: it is a TOLERANCE, and it removes every
            # multiplier within it of 1. Near the Hopf the orbit's own multiplier tends to 1 as well
            # - measured on the subcritical Lorenz orbit 1e-4 off its Hopf, the trivial one came out
            # at 1+2.1e-9 and the physical one at 1+3.9e-6, and a tolerance of 1e-5 deleted both, i.e.
            # exactly the number that answers whether the orbit is unstable. Every orbit has EXACTLY
            # ONE trivial multiplier, so the one nearest 1 is removed below and nothing else is.
            mults=self.problem.get_floquet_multipliers(
                n=self.floquet_n,method=self.floquet_method,
                ignore_periodic_unity=False,quiet=True,
                dense_threshold=self.floquet_dense_threshold,
                shift_invert=self.floquet_shift_invert,sigma=self.floquet_sigma)
        except Exception as e:
            self.log("Could not compute the Floquet multipliers here ("+str(e).split("\n")[0]+
                     "); this point's stability stays unknown")
            return [],[]
        finally:
            if not self.floquet_feeds_eigen_panes:
                (self.problem._last_eigenvalues,self.problem._last_eigenvectors,
                 self.problem._last_eigenvalues_m,self.problem._last_eigenvalues_k)=keep
        # Belt and braces on the trivial multiplier: it is 1+-1e-15, so its exponent is a tiny number
        # of either sign, and left in it flips the branch between stable and unstable from one point
        # to the next. It is not a bifurcation marker - eig_value_Re would have to be exactly zero -
        # but it does corrupt every count.
        allm=[complex(m) for m in mults]
        kept=list(allm)
        self._orbit_trivial_multiplier_error=float("nan")
        if kept:
            # The one nearest 1 is the trivial multiplier, from the orbit's time-translation
            # invariance. How far from 1 it actually came out is the accuracy of the discretization
            # here and nothing else, which is worth saying when it is poor - it bounds what can be
            # believed about every other multiplier.
            i=min(range(len(kept)),key=lambda j: abs(kept[j]-1.0))
            self._orbit_trivial_multiplier_error=abs(kept[i]-1.0)
            if self._orbit_trivial_multiplier_error>self.floquet_unity_tol:
                self.log("The trivial Floquet multiplier came out at 1{:+.3g} rather than at 1; that "
                         "is how accurate this discretization is here, so nothing closer to the unit "
                         "circle than that can be read. More time steps, or a higher order, tighten it."
                         .format(float(numpy.real(kept[i]))-1.0))
            kept.pop(i)
        T=float(orbit.get_T(dimensional=False))
        exps:list[complex]=[]
        for m in kept:
            if abs(m)<=0.0:
                # An algebraic direction annihilated by the discretization: no finite exponent, and
                # arbitrarily stable. Reported as a large negative rate rather than -inf, which
                # nothing downstream can plot or compare.
                exps.append(complex(-1e30,0.0))
            else:
                exps.append(complex(numpy.log(complex(m)))/T)
        order=sorted(range(len(exps)),key=lambda i:-exps[i].real)
        kept=[kept[i] for i in order]
        exps=[exps[i] for i in order]
        return list(self._phys_eig(exps) or []),kept

    def orbit_unstable_count(self,mults:"Sequence[complex]")->int:
        """How many multipliers are outside the unit circle, with a deadband."""
        return sum(1 for m in mults if abs(m)>1.0+self.floquet_unstable_tol)

    # ------------------------------------------------------------------ orbit state files

    def _orbit_blocks(self)->"tuple[NPFloatArray,float]":
        """The orbit's own unknowns as ``(nT, nbase)`` time blocks, plus the period.

        Read straight out of the augmented dof vector in its naive time-major order rather than by
        re-sampling the orbit: a resample interpolates, and PeriodicOrbit.change_sampling - the
        obvious thing to copy - also duplicates the first time point and drops the last.
        """
        handler=self._orbit_handler()
        assert handler is not None
        nbase=handler.get_base_ndof()
        nT=handler.get_num_time_steps()
        dofs=numpy.asarray(self.problem.get_current_dofs()[0],dtype=float)
        order=handler.get_naive_equation_order()
        if len(order):
            dofs=dofs[numpy.asarray(order,dtype=numpy.int64)]
        return dofs[:nbase*nT].reshape(nT,nbase).copy(),float(dofs[nbase*nT])

    def _capture_dof_fingerprint(self)->str:
        """What the raw dof blocks are only meaningful for: this equation numbering.

        A state dump is partition-independent because it is written per node; a raw dof vector is not,
        and neither survives a mesh adaptation. Refusing on a mismatch is the whole point - a wrong
        orbit loads perfectly happily and looks plausible.

        Taken only while the PLAIN system is installed, and remembered: get_dof_description is sized by
        ndof(), which is the augmented count under a handler, while its walk fills the base entries
        alone - asked during orbit tracking it prints "UNASSIGNED DOF IN DOFLIST" and describes the
        time-block copies by whatever the unfilled entries happen to hold.
        """
        import hashlib
        if self._augmented_system_active():
            return self._orbit_dof_fingerprint
        try:
            inds,names=self.problem.get_dof_description()
            h=hashlib.sha1(numpy.asarray(inds,dtype=numpy.int64).tobytes())
            h.update(("\0".join(str(n) for n in names)).encode("utf-8"))
            self._orbit_dof_fingerprint=h.hexdigest()
        except Exception as e:
            self.log("Could not fingerprint the degrees of freedom:",repr(e))
            self._orbit_dof_fingerprint=""
        return self._orbit_dof_fingerprint

    @staticmethod
    def _orbit_sidecar_paths(statefile:str)->"tuple[str,str]":
        """``(npz, directory)`` belonging to one state dump. Derived, so a point never loses them."""
        base=statefile[:-5] if statefile.endswith(".dump") else os.path.splitext(statefile)[0]
        return base+".orbit.npz",base+"_orbit"

    def _write_orbit_sidecar(self,point)->dict:
        """Store the orbit's unknowns next to the base state, and say where.

        A state dump holds the mesh, i.e. ONE phase of the cycle. Everything that makes it an orbit -
        the other time points and the period - lives in the handler and has to go somewhere else.
        """
        npzfile,dirfile=self._orbit_sidecar_paths(point.statefile)
        blocks,T=self._orbit_blocks()
        portable=self.orbit_portable or self.problem.is_distributed()
        info=dict(point.orbit_info or {})
        info["format"]="dumps" if portable else "dofs"
        info["fingerprint"]="" if portable else self._orbit_dof_fingerprint
        info["nbase"]=int(blocks.shape[1])
        info["nT"]=int(blocks.shape[0])
        info["T"]=float(T)
        # The tangent belongs to the AUGMENTED system, so it is not written into the dump (it would be
        # dropped on load with a note) but it is what makes a reloaded branch step on in the same
        # direction instead of guessing one.
        ddof=numpy.asarray(self.problem.get_arclength_dof_derivative_vector(),dtype=float)
        extra={"T":numpy.array([T]),
               "dof_deriv":ddof,
               "param_deriv":numpy.array([self.problem.get_arc_length_parameter_derivative()]),
               "theta_sqr":numpy.array([self.problem.get_arc_length_theta_sqr()])}
        if portable:
            Path(dirfile).mkdir(parents=True,exist_ok=True)
            handler=self._orbit_handler()
            assert handler is not None
            nT=blocks.shape[0]
            handler.backup_dofs()
            try:
                for i in range(nT):
                    handler.set_dofs_to_interpolated_values(i/(nT-1))
                    self.problem.invalidate_cached_mesh_data()
                    self.problem.save_state(os.path.join(dirfile,"block_{:03d}.dump".format(i)),quiet=True)
            finally:
                handler.restore_dofs()
                self.problem.invalidate_cached_mesh_data()
            numpy.savez_compressed(os.path.join(dirfile,"orbit.npz"),**extra)
        else:
            numpy.savez_compressed(npzfile,blocks=blocks,**extra)
        point.orbit_info=info
        return info

    def _install_orbit_from_sidecar(self,point):
        """Put the orbit of a stored point back on the problem, exactly as it was."""
        info=point.orbit_info or {}
        npzfile,dirfile=self._orbit_sidecar_paths(point.statefile)
        if info.get("format")=="dumps":
            blocks,extra=self._read_orbit_dumps(dirfile,info)
        else:
            if not os.path.exists(npzfile):
                raise RuntimeError("The orbit belonging to this point is missing ("+npzfile+"), so "
                                   "only one phase of the cycle could be restored.")
            data=numpy.load(npzfile)
            blocks=numpy.asarray(data["blocks"],dtype=float)
            extra=data
            # Captured here, with the plain system still installed: load_pt has just taken any
            # handler off, and this is the last moment the base numbering can be described.
            want=info.get("fingerprint")
            if want and want!=self._capture_dof_fingerprint():
                raise RuntimeError(
                    "This orbit was stored for a different arrangement of the degrees of freedom (a "
                    "mesh adaptation, or a different number of processes), so its stored unknowns "
                    "cannot be read back. Re-compute the branch, or store orbits in the portable "
                    "format (controller.orbit_portable=True), which is partition-independent.")
        T=float(numpy.asarray(extra["T"]).reshape(-1)[0])
        self._install_orbit(info,blocks,T)
        # The augmented tangent, so that continuing a reloaded branch goes on the way it was going.
        try:
            ddof=numpy.asarray(extra["dof_deriv"],dtype=float)
            if len(ddof)==self.problem.ndof():
                self.problem._set_dof_direction_arclength(ddof)
                self.problem._set_arc_length_parameter_derivative(float(numpy.asarray(extra["param_deriv"]).reshape(-1)[0]))
                self.problem._set_arc_length_theta_sqr(float(numpy.asarray(extra["theta_sqr"]).reshape(-1)[0]))
        except Exception as e:
            self.log("The stored orbit tangent could not be restored:",repr(e))

    def _read_orbit_dumps(self,dirfile:str,info:dict):
        """Harvest the time blocks from one state dump per time point (the portable format)."""
        nT=int(info.get("nT",0))
        blocks=[]
        for i in range(nT):
            fname=os.path.join(dirfile,"block_{:03d}.dump".format(i))
            if not os.path.exists(fname):
                raise RuntimeError("The orbit belonging to this point is incomplete ("+fname+" is missing)")
            self.problem.load_state(fname,ignore_outstep=True,ignore_continuation_data=True,
                                    ignore_eigendata=True,quiet=True)
            blocks.append(numpy.asarray(self.problem.get_current_dofs()[0],dtype=float))
        extra=numpy.load(os.path.join(dirfile,"orbit.npz"))
        return numpy.array(blocks),extra

    def _install_orbit(self,info:dict,blocks:"NPFloatArray",T_nondim:float):
        """Re-create the handler from stored time blocks and write them back exactly."""
        nT=int(blocks.shape[0])
        if nT<2:
            raise RuntimeError("A stored orbit needs at least two time points")
        mode=str(info.get("mode","collocation"))
        # In floquet/collocation mode the handler appends the end-of-period block itself, from the
        # current state; handing it over as history as well would make the orbit one block longer on
        # every reload.
        floquet_mode=mode in ("collocation","floquet")
        history=[blocks[i] for i in range(1,nT-1 if floquet_mode else nT)]
        self._deactivate_any_augmentation()
        self.problem.set_current_dofs(blocks[0])
        orbit=self.problem.activate_periodic_orbit_handler(
            T_nondim*self.problem.get_scaling("temporal"),history_dofs=history,mode=mode,
            order=int(info.get("order",3)),GL_order=int(info.get("GL_order",-1)),
            T_constraint=str(info.get("T_constraint","phase")))
        handler=self._orbit_handler()
        assert handler is not None
        if handler.get_num_time_steps()!=nT:
            self.problem.deactivate_bifurcation_tracking()
            raise RuntimeError("The restored orbit has {:d} time points instead of {:d}".format(
                handler.get_num_time_steps(),nT))
        # Write every block back, the period included. activate_periodic_orbit_handler rebuilds the
        # end-of-period block from the current state and re-nondimensionalises T, so this is what
        # makes the restore exact rather than approximately right.
        aug=numpy.concatenate([blocks.reshape(-1),[T_nondim]])
        order=handler.get_naive_equation_order()
        if len(order):
            full=numpy.empty_like(aug)
            full[numpy.asarray(order,dtype=numpy.int64)]=aug
            aug=full
        self.problem.set_current_dofs(aug)
        self._orbit=orbit
        return orbit

    def _add_orbit_state(self):
        """Record the installed orbit as a point of the current branch.

        Not _add_current_state on its own: that one reads the problem's last eigenvalues when it is
        given none, and after a Floquet computation those are MULTIPLIERS. It also evaluates the
        observables at whatever phase of the cycle the mesh happens to be holding.
        """
        obs,info=self._evaluate_orbit_observables()
        exps,mults=self._orbit_floquet()
        collapsed=self._orbit_has_collapsed(obs,mults)
        info["collapsed"]=bool(collapsed)
        self._add_current_state(eig_value=(exps[0] if exps else None),
                                eig_values=exps if exps else None,
                                measured=bool(exps),observables=obs)
        p=self.current_point
        assert p is not None
        p.orbit_info=info
        p.floquet=mults
        if mults:
            # From |mu| > 1 with a deadband, not from the count of positive exponent real parts: the
            # multiplier is the quantity with the clean threshold, and stability_indicator prefers
            # unstable_count whenever it is set.
            p.unstable_count=self.orbit_unstable_count(mults)
        self._write_orbit_sidecar(p)
        if collapsed:
            # The point is recorded first and the failure raised after: it IS a converged solution,
            # so throwing it away would be worse than leaving one the user can look at and delete.
            # Raising rather than setting the abort flag, because that flag stops the current sweep
            # and then quietly makes the NEXT Step do nothing; and because continuing from a
            # collapsed point is precisely what must not happen silently.
            raise RuntimeError(
                "This orbit has collapsed onto the stationary branch: every observable is constant "
                "over the 'cycle'. The point is recorded so it can be inspected, but continuing from "
                "it would record stationary solutions as orbits. Delete it and step again with a "
                "smaller ds.")
        return p

    def orbit_floquet_here(self)->bool:
        """Recompute the Floquet multipliers at the current orbit point."""
        if not self.on_orbit():
            raise RuntimeError("Floquet multipliers belong to a periodic orbit; this is not one")
        point=self._get_current_point()
        obs,info=self._evaluate_orbit_observables()
        exps,mults=self._orbit_floquet()
        if not exps:
            return False
        point.obs_values.update(obs)
        point.orbit_info=dict(info,**{k:v for k,v in (point.orbit_info or {}).items()
                                      if k in ("format","fingerprint")})
        point.floquet=mults
        point.eig_values=[complex(v) for v in exps]
        point.eig_value_Re=numpy.real(exps[0])
        point.eig_value_Im=numpy.imag(exps[0])
        point.stability_source=STABILITY_EIGEN
        point.unstable_count=self.orbit_unstable_count(mults)
        self.log("{:d} Floquet multiplier{:s}, {:d} outside the unit circle".format(
            len(mults),"" if len(mults)==1 else "s",point.unstable_count))
        self._changed()
        return True

    def orbit_floquet_for_branch(self,branch=None)->int:
        """Fill in the multipliers along a whole orbit branch. Abortable between points."""
        branch=branch if branch is not None else self._get_current_branch()
        if branch.kind!=BRANCH_ORBIT:
            raise RuntimeError("This is not a branch of periodic orbits")
        done=0
        restore=self.current_point
        for p in list(branch):
            if self._abort_requested:
                self._abort_requested=False
                self.log("Stopped after {:d} point{:s}".format(done,"" if done==1 else "s"))
                break
            if self.compute_spectrum(p):
                done+=1
        if restore is not None:
            try:
                self.load_pt(restore)
            except Exception as e:
                self.log("Could not return to the previous point:",repr(e))
        self._changed()
        return done

    def output_orbit_cycle(self,subdir:str | None=None)->str:
        """Write the problem's own output along the whole cycle, not just the phase it is sitting at."""
        orbit=self._require_orbit()
        name=subdir or "orbit_{:s}_{:.6g}".format(self._get_paramname_str(),
                                                  self.get_bifurcation_parameter().value)
        sub=os.path.join(self.data_subdir,"output",name)
        orbit.output_orbit(sub)
        out=self.problem.get_output_directory(sub)
        self.log("Wrote the cycle to",out)
        return out

    def _orbit_has_collapsed(self,obs:"dict[str,float]",mults:"Sequence[complex]")->bool:
        """Whether what was just solved is a stationary state wearing an orbit's clothes.

        Continuation can walk an orbit branch straight back onto the stationary one it came from -
        Newton has no reason not to, an unstable orbit least of all - and the result is a perfectly
        converged solution that is recorded as an orbit, drawn as an orbit and is not one. The
        collapse check in switch_to_hopf_orbit only guards the FIRST orbit; this guards every step.

        Measured on the amplitude alone: every observable is constant over the "cycle", so every band
        has zero width, relative to the observable's own scale. Not on the multipliers, tempting as it
        is - a collapsed orbit does lose its multiplier at 1, but so does a perfectly good one whose
        trivial multiplier the discretization has not resolved to within the tolerance, and the two
        cannot be told apart that way. Measured: a genuine Lorenz orbit gives 7e-3 here and a collapsed
        one 1e-11, so the two are three orders of magnitude clear of any threshold in between.
        """
        widest=0.0
        for name in list(obs):
            lo,hi=orbit_band_names(name)
            if lo not in obs or hi not in obs:
                continue
            scale=max(abs(obs.get(name,0.0)),abs(obs[lo]),abs(obs[hi]))
            if scale<=0.0:
                continue
            widest=max(widest,(obs[hi]-obs[lo])/scale)
        return widest<self.orbit_collapse_tolerance

    def orbit_can_be_started(self)->"str | None":
        """Why switching onto an orbit here would fail, or None when it would not.

        Answered up front because the reasons are all things the user has to go and change elsewhere,
        and because the Hessian one cannot be recovered from once the handler is installed - the throw
        comes out of the first Newton solve, with the augmented system already in place.
        """
        if self.on_locus():
            return "Switching onto an orbit applies to an ordinary branch, not to a bifurcation locus"
        if self.on_orbit():
            return "This is already a branch of periodic orbits"
        if not self.problem.are_hessian_products_calculated_analytically():
            return ("A periodic orbit needs the analytic Hessian, which has to be generated when the "
                    "problem is compiled: add problem.setup_for_stability_analysis(analytic_hessian="
                    "True) BEFORE the problem is initialised and run the script again. The stored "
                    "diagram is kept, so nothing computed so far is lost.")
        if self.problem.is_distributed():
            return ("Switching onto an orbit from a Hopf does not work on a distributed problem "
                    "(--distribute); the orbit itself and its Floquet multipliers do.")
        mode=self._point_normal_mode(self.current_point)
        if mode is not None:
            # What a Hopf of an m != 0 mode sheds is a rotating or travelling wave, not a standing
            # oscillation of these degrees of freedom. Answered here as well as at Problem level so
            # that the menu can grey the command out instead of offering a call that will refuse.
            return ("This is a Hopf of the {:s} = {:g}; the orbit it sheds is a wave running in the "
                    "{:s} direction, which this problem's degrees of freedom cannot represent."
                    .format("azimuthal mode m" if mode[0]=="m" else "Cartesian mode k",mode[1],
                            "azimuthal" if mode[0]=="m" else "extra Cartesian"))
        return None

    def _orbit_NT(self)->int:
        """The number of time steps to ask for, with the parity the multipliers need.

        With a DAE and an ODD number of time intervals the algebraic directions land on exactly -1,
        which is where a period doubling would be - a discretization artefact sitting on top of the
        one multiplier value that matters most (dev_docs/floquet_multipliers.md).
        """
        NT=max(2,int(self.orbit_NT))
        if self.orbit_mode=="collocation":
            order=max(1,int(self.orbit_order))
            step=order if order%2==0 else 2*order
        else:
            step=2
        if NT%step:
            NT=((NT//step)+1)*step
            self.log("Using {:d} time steps: a multiple of the collocation order, and even - with an "
                     "odd number of time intervals a DAE puts a spurious Floquet multiplier on "
                     "exactly -1, where a period doubling would be.".format(NT))
        return NT

    def _hopf_eigenindex(self,tol:float=1e-8)->"int | None":
        """The complex pair nearest the imaginary axis in the last eigensolve.

        Restricted to |Im| > tol, unlike critical_eigenindex: at a located Hopf the tracker has put a
        pair at exactly 0 +- i*omega, and a real eigenvalue can easily be nearer the axis than that.
        """
        evs=self.problem.get_last_eigenvalues()
        if evs is None or len(evs)==0:
            return None
        best,bestre=None,None
        for i,v in enumerate(evs):
            if abs(numpy.imag(v))<=tol:
                continue
            re=abs(float(numpy.real(v)))
            if bestre is None or re<bestre:
                best,bestre=i,re
        return best

    def switch_to_orbit(self,eps:float | None=None):
        """From a Hopf bifurcation, step onto the periodic orbit it sheds.

        The numerics are Problem.switch_to_hopf_orbit, so the same manoeuvre is available to a plain
        script; what is here is the part that is about the diagram - checking that it can work at all
        before anything is installed, choosing the step off the Hopf, opening a branch for the result
        and leaving the continuation ready to carry on.

        The step off the bifurcation is taken from ds, like a branch switch: the parameter offset a
        Hopf orbit starts at is eps**2, and branch_switch_parameter_offset() is what one ds buys in
        the parameter, so passing it as dparam makes the first orbit sit exactly one step away. Which
        SIDE of the Hopf it sits on is not ours to choose - it is where the orbits exist, and the
        first Lyapunov coefficient says which side that is.
        """
        refusal=self.orbit_can_be_started()
        if refusal is not None:
            raise RuntimeError(refusal)
        cp=self._get_current_point()
        if cp.eig_value_Re!=0:
            raise RuntimeError("Orbits are switched onto at a bifurcation; this is an ordinary point")
        if cp.bifurcation_info is None:
            self.log("This bifurcation was not classified; computing its normal form now")
            cp.bifurcation_info=self._classify_current_point(cp)
        btype=cp.bifurcation_info.get("type") if cp.bifurcation_info is not None else None
        if btype!="hopf":
            raise RuntimeError("Only a Hopf bifurcation sheds a periodic orbit; this one is "+
                               (str(btype) if btype else "not classified"))

        offset=abs(float(eps)) if eps is not None else (
            abs(float(self.orbit_eps)) if self.orbit_eps is not None
            else self.branch_switch_parameter_offset())
        if not offset>0:
            # switch_to_hopf_orbit tests dparam for truth, so a zero would silently fall through to
            # its own default eps rather than being used.
            offset=self.branch_switch_offset*max(abs(float(self.get_bifurcation_parameter().value)),1.0)

        # The tracker is off by now - locate_bifurcation deactivates it in its finally, and the
        # normal-form calculation deactivates it too - but switch_to_hopf_orbit reads the Hopf out of
        # an ACTIVE tracker. We are sitting on the solution, so this is a couple of Newton steps.
        self._status("SOLVING THE HOPF FOR THE ORBIT")
        self.problem.solve_eigenproblem(self.neigen,self.shift)
        idx=self._hopf_eigenindex()
        if idx is None:
            raise RuntimeError("No complex eigenvalue pair could be found here, so there is no Hopf "
                               "to start an orbit from. Solve the eigenproblem with more eigenvalues.")
        evs=self.problem.get_last_eigenvalues()
        self.log("Re-solving the Hopf to start the orbit from it (eigenvalue {:d}: {:.4g}{:+.4g}i)"
                 .format(idx,float(numpy.real(evs[idx])),float(numpy.imag(evs[idx]))))
        NT=self._orbit_NT()
        # The last moment the plain equation numbering can be described; the orbit's stored blocks
        # only mean anything against it.
        self._capture_dof_fingerprint()
        self.problem.activate_bifurcation_tracking(self._paramname,"hopf",eigenvector=idx)
        try:
            self.problem.solve(max_newton_iterations=20)
            self._status("SWITCHING ONTO THE ORBIT")
            self.log("Starting the orbit with a parameter step of {:.4g} (eps = {:.4g}), {:d} time "
                     "steps, {:s}".format(offset,numpy.sqrt(offset),NT,self.orbit_mode))
            orbit=self.problem.switch_to_hopf_orbit(
                dparam=offset,NT=NT,mode=self.orbit_mode,order=self.orbit_order,
                GL_order=self.orbit_GL_order,T_constraint=self.orbit_T_constraint,
                amplitude_factor=self.orbit_amplitude_factor,do_solve=True,
                check_collapse_to_stationary=self.orbit_check_collapse)
        except Exception as e:
            # switch_to_hopf_orbit switches the tracker off itself and can leave an ORBIT handler
            # installed when its own solve fails; either one left behind would make every later
            # command work on the wrong system.
            self._deactivate_any_augmentation()
            self._orbit=None
            self.problem.reset_arc_length_parameters()
            if "collapse" in str(e).lower():
                self.log("The orbit collapsed back onto the stationary branch. Step further off the "
                         "Hopf (a larger ds, or the parameter step in the Orbit tab), or raise the "
                         "amplitude factor.")
            try:
                self.load_pt(cp)
            except Exception as e2:
                self.log("Could not return to the Hopf bifurcation:",repr(e2))
            raise
        self._orbit=orbit
        lyap=orbit.emerging_info.get("lyap_coeff")
        if lyap:
            self.log("First Lyapunov coefficient {:.4g}: the Hopf is {:s}, so the orbits are "
                     "initially {:s}".format(float(lyap),
                     "supercritical" if orbit.starts_supercritically() else "subcritical",
                     "stable" if orbit.starts_supercritically() else "unstable"))
        self._new_branch(kind=BRANCH_ORBIT)
        self._tangs={}
        ds=orbit.get_init_ds()
        # get_init_ds clamps to 5e-10 when the parameter barely moved, which would make every
        # following step invisible. Its SIGN is the useful part - the side the orbits are on.
        if abs(ds)<1e-9:
            ds=numpy.sign(ds) if ds else 1.0
            ds=float(ds)*offset
        self._last_ds=float(ds)
        p=self._add_orbit_state()
        self.log("Period {:.6g}{:s} at {:s} = {:.6g}; ds set to {:.3g}".format(
            self.orbit_period()," "+self._orbit_T_unit if self._orbit_T_unit else "",
            self._get_paramname_str(),self.get_bifurcation_parameter().value,self._last_ds))
        if p.floquet:
            self.log("{:d} Floquet multiplier{:s}, {:d} outside the unit circle".format(
                len(p.floquet),"" if len(p.floquet)==1 else "s",self.orbit_unstable_count(p.floquet)))
        self._mode="al"
        self._changed()
        return True

    def _make_deflation_operator(self):
        """A deflation operator built from the current settings, attached to the problem."""
        from ...generic.bifurcation_tools import DeflationOperator
        op=DeflationOperator(alpha=self.deflation_alpha,p=int(self.deflation_p))
        # Distances in units of the perturbation, so alpha means the same thing at every scale; the
        # same rule Problem.iterate_over_multiple_solutions_by_deflation applies to its own operator.
        op.scale=max(self.deflation_perturbation_value(),1e-300)*numpy.sqrt(max(self.problem.ndof(),1))
        return op

    def deflation_known_count(self)->int:
        """How many solutions the deflated search is currently avoiding."""
        return len(self._deflation_known)

    def clear_deflation_known_solutions(self):
        """Forget them, so the next deflated solve starts the search over from here."""
        self._deflation_known=[]
        self._deflation_at=None
        self.log("Deflation: forgot the known solutions")
        self._changed()

    def _refresh_deflation_known(self):
        """Make sure the avoided set belongs to the parameter value we are actually at.

        The set is only meaningful at ONE parameter value: a solution at a different value is not a
        solution here, and deflating it would push the search away from a perfectly good root for no
        reason. Moving the parameter therefore starts a fresh search.
        """
        here=float(self.get_bifurcation_parameter().value)
        if self._deflation_at is None or not same_parameter_value(self._deflation_at,here):
            self._deflation_known=[]
            self._deflation_at=here
        current=self.problem.get_current_dofs()[0]
        if all(not self._same_solution(current,W) for W in self._deflation_known):
            self._deflation_known.append(numpy.array(current))

    @staticmethod
    def _same_solution(a,b,tol:float=1e-6)->bool:
        if len(a)!=len(b):
            return False
        scale=max(1.0,float(numpy.linalg.norm(a)),float(numpy.linalg.norm(b)))
        return float(numpy.linalg.norm(numpy.asarray(a)-numpy.asarray(b)))<=tol*scale

    def _deflation_perturbation(self,rng)->NPFloatArray:
        """The direction the guess is moved in before a deflated solve.

        The eigenvector when there is one and the user asked for it, a random dof vector otherwise.
        The eigenvector is worth preferring where it exists: it is a field rather than a list of dof
        indices, and it points along the mode that is about to go unstable, which is where a new
        branch usually is.
        """
        n=self.problem.ndof()
        if self.deflation_use_eigenperturbation:
            evs=self.problem.get_last_eigenvectors()
            if evs is not None and len(evs)>0 and len(evs[0])==n:
                v=numpy.real(numpy.array(evs[0]))
                m=float(numpy.amax(numpy.absolute(v)))
                if m>0:
                    return v/m*self.deflation_perturbation_value()
            self.log("Deflation: no usable eigenvector for the perturbation, using a random one")
        return (rng.random(n)-0.5)*self.deflation_perturbation_value()

    def deflated_solve(self)->bool:
        """One deflated solve here; a solution that is genuinely new opens a new branch.

        The current solution, and everything found by earlier clicks at this same parameter value,
        are deflated away, so pressing this repeatedly walks through the solutions at this parameter
        rather than finding the same one over and over. The parameter is not moved.
        """
        if self.on_orbit():
            raise RuntimeError("A deflated search looks for another STATIONARY solution; it cannot be run against an orbit. Load a stationary point first.")
        if self.current_point is None and self.current_branch is None:
            self.log("Deflation: solve or step once first, so there is a solution to deflate away")
            return False
        self._refresh_deflation_known()
        before=numpy.array(self.problem.get_current_dofs()[0])
        op=self._make_deflation_operator()
        rng=numpy.random.default_rng(self.deflation_random_seed)
        self.problem.set_deflation_operator(op)
        found_one=False
        try:
            for W in self._deflation_known:
                op.add_known_solution(W)
            for attempt in range(max(1,int(self.deflation_random_tries))):
                if self._abort_requested:
                    self._abort_requested=False
                    self.log("Deflated solve aborted")
                    break
                self._status("DEFLATED SOLVE {:d}/{:d} (avoiding {:d})".format(
                    attempt+1,max(1,int(self.deflation_random_tries)),len(self._deflation_known)))
                self.problem.set_current_dofs(before+self._deflation_perturbation(rng))
                try:
                    self.problem.solve(max_newton_iterations=self.deflation_max_newton_iterations)
                except Exception as e:
                    self.log("Deflated solve attempt {:d} did not converge ({:s})".format(attempt+1,type(e).__name__))
                    continue
                found=self.problem.get_current_dofs()[0]
                if any(self._same_solution(found,W) for W in self._deflation_known):
                    # Deflation stops Newton converging ONTO a known solution but not arbitrarily
                    # close to one; near a bifurcation the branches meet. Opening a branch for that
                    # would put a duplicate on top of the one already there.
                    self.log("Deflated solve attempt {:d} came back to a known solution".format(attempt+1))
                    continue
                self._deflation_known.append(numpy.array(found))
                found_one=True
                # Off before the eigensolve and before the point is recorded: everything from here on
                # -- the eigenproblem, the state dump, any later step -- must see the ordinary
                # residual, not the deflated one.
                self.problem.set_deflation_operator(None)
                self._new_branch()
                if self.quick_mode:
                    self._add_current_state(measured=False)
                else:
                    self.problem.solve_eigenproblem(self.neigen,self.shift,
                                                    **(self._mode_kwargs() if self.compute_modes_during_sweep else {}))
                    self._add_current_state()
                # The new solution is not on the branch the arclength tangent describes, so an
                # ordinary Step from here has to start a fresh one.
                self.problem.reset_arc_length_parameters()
                self._tangs={}
                self.log("Deflation found a new solution, on branch {:d} (now avoiding {:d})".format(
                    len(self.branches)-1,len(self._deflation_known)))
                self.save_all()
                self._changed()
                return True
            self.log("Deflation found nothing new here after {:d} attempt{:s}".format(
                max(1,int(self.deflation_random_tries)),"" if self.deflation_random_tries==1 else "s"))
            return False
        finally:
            self.problem.set_deflation_operator(None)
            if not found_one:
                # Every attempt perturbed the dofs and then failed or came back to a known solution,
                # so the problem is holding whichever of those it stopped on. Put the state the user
                # was looking at back; it is a converged solution already, so no solve is needed.
                self.problem.set_current_dofs(before)

    def scanned_parameter_axis_limits(self)->"tuple[float,float] | None":
        """The visible range of the axis that shows the continuation parameter, or None.

        Which axis that is depends on what the user pinned; on the usual diagram it is x, but either
        can carry either, and the parameter need not be drawn at all - in which case there is no
        range to run out of and the scan is bounded only by its step count.
        """
        if not isinstance(self._paramname,str):
            return None
        pax=parameter_axis(self._paramname)
        if as_axis(self.x_axis)==pax:
            lim=self.view.get_xlim()
        elif as_axis(self.y_axis)==pax:
            lim=self.view.get_ylim()
        else:
            return None
        lo,hi=min(lim),max(lim)
        # A view is only a bound if it is a number. Matplotlib will hand out a non-finite limit if a
        # non-finite coordinate ever reached extend_lims, and comparing against it silently answers
        # False to everything - which came out of the tab as a scan range of "nan ... nan".
        if not (numpy.isfinite(lo) and numpy.isfinite(hi)):
            return None
        return (float(lo),float(hi))

    def deflated_scan_values(self)->list[float]:
        """The parameter values a deflated continuation would visit, current value first.

        Truncated where the parameter leaves the visible axes, in the same spirit as
        :py:meth:`multistep`: a scan that marches off the plot is computing points nobody asked to
        see. Two things follow multistep rather than being stricter than it, and both were wrong the
        first time round -- the label read "in 0 steps" and the button did nothing:

        * the first value OUTSIDE the range is kept, so the scan steps and then notices it has left,
          exactly as multistep does (it steps first and tests afterwards). A scan therefore always
          takes at least one step;
        * if the starting value is not inside the range at all, nothing is clipped. The view then says
          nothing about where this scan should stop -- most often it has not been framed yet, since a
          diagram with no points has never had its limits set -- and clipping against it would leave
          the whole scan behind.
        """
        start=float(self.get_bifurcation_parameter().value)
        n=max(1,int(self.deflated_scan_steps))
        dp=self.deflated_scan_dparam
        if dp is None:
            # ds, signed: the same step the arclength continuation would take and the same length the
            # direction arrow is drawn with. A rule based on the parameter's own magnitude instead
            # (0.05*|p|, as this was) has nothing to do with either, and on a fresh diagram - whose
            # window is 10 ds wide - a single step of it could already leave the plot, so the scan
            # reported itself as cut short after one step.
            dp=float(self._last_ds) or 0.05*max(abs(start),1.0)
        dp=float(dp)
        if not (numpy.isfinite(start) and numpy.isfinite(dp)):
            return [start]
        values=[start+i*dp for i in range(n+1)]
        lim=self.scanned_parameter_axis_limits()
        # Clipped only when the view actually bounds this scan, i.e. the starting value is inside it
        # AND at least one whole step fits inside it too. Otherwise the view has nothing to say and the
        # full scan runs: after ten continuation steps the point sits exactly ON the right edge (the
        # plotter grows the limits to include it), and clipping there reported a ten-step scan as a
        # one-step one. The view is not a wall in any case - it grows as the scan draws into it - and
        # the loop's own box test is what stops a scan that has genuinely left the diagram.
        if lim is not None and lim[0]<=start<=lim[1] and lim[0]<=start+dp<=lim[1]:
            keep=[values[0]]
            for v in values[1:]:
                keep.append(v)
                if v<lim[0] or v>lim[1]:
                    break
            values=keep
        return values

    def deflated_scan_is_clipped(self)->bool:
        """Whether the visible range shortens the scan the settings ask for."""
        return len(self.deflated_scan_values())<max(1,int(self.deflated_scan_steps))+1

    def deflated_continuation(self)->int:
        """Scan the continuation parameter, looking for new solutions at every value.

        Farrell, Beentjes & Birkisson (arXiv:1603.00809), driven through
        :py:meth:`~pyoomph.generic.problem.Problem.deflated_continuation`. Unlike arclength
        continuation this cannot turn a fold -- it steps the parameter, full stop -- but it does find
        branches that are not connected to the one being followed, which arclength never can.

        Every branch it reports becomes a NEW branch of the diagram, including the continuation of
        the solution we started from: a parameter scan and an arclength branch are different objects
        and merging them would put points on a branch whose ordering they do not respect.

        Abortable between parameter steps. Returns the number of branches created.
        """
        if self.on_orbit():
            raise RuntimeError("A deflated scan looks for stationary solutions; it cannot be run against an orbit. Load a stationary point first.")
        if not isinstance(self._paramname,str):
            self.log("Deflated continuation needs a named continuation parameter")
            return 0
        values=self.deflated_scan_values()
        self.log("Deflated continuation in '{:s}' over {:g} ... {:g} in {:d} steps".format(
            self._paramname,values[0],values[-1],len(values)-1))
        by_index:dict[int,BifurcationGUISolutionBranch]={}
        xlim,ylim=self.view.get_xlim(),self.view.get_ylim()
        at_value=float(values[0])
        step_points,any_visible=0,False
        gen=self.problem.deflated_continuation(
            deflation_alpha=self.deflation_alpha,deflation_p=int(self.deflation_p),
            perturbation_amplitude=self.deflation_perturbation_value(),
            max_newton_iterations=self.deflation_max_newton_iterations,
            use_eigenperturbation=self.deflation_use_eigenperturbation,
            num_random_tries=max(1,int(self.deflation_random_tries)),
            random_seed=self.deflation_random_seed,
            **{self._paramname:values})
        try:
            for branch_index,pvalue,_dofs in gen:
                if self._abort_requested:
                    self._abort_requested=False
                    self.log("Deflated continuation aborted")
                    break
                if not same_parameter_value(pvalue,at_value):
                    # A parameter step is done. Stop if nothing it produced was on the plot: the
                    # parameter itself may still be inside the visible range (or not drawn at all)
                    # while every branch has left it through the other axis, which is the case
                    # multistep's box test catches and a range check on the parameter alone does not.
                    if step_points and not any_visible:
                        self.log("Deflated continuation stopped: every branch has left the visible axes")
                        break
                    at_value=float(pvalue)
                    step_points,any_visible=0,False
                branch=by_index.get(branch_index)
                if branch is None:
                    branch=self._new_branch()
                    by_index[branch_index]=branch
                self.current_branch=branch
                self._status("DEFLATED SCAN {:s}={:g} ({:d} branch{:s})".format(
                    str(self._paramname),float(pvalue),len(by_index),"" if len(by_index)==1 else "es"))
                if self.deflated_scan_eigensolve:
                    self.problem.solve_eigenproblem(self.neigen,self.shift,
                                                    **(self._mode_kwargs() if self.compute_modes_during_sweep else {}))
                    self._add_current_state()
                else:
                    # measured=False: no eigensolve was done here, so the point must be recorded WITHOUT
                    # a spectrum rather than with whatever the problem still holds from an earlier one.
                    # Its stability then reads as unknown and "Compute the eigenvalues along this
                    # branch" fills it in on demand.
                    self._add_current_state(measured=False)
                step_points+=1
                cp=self._get_current_point().get_coordinate(self.y_axis,xspec=self.x_axis)
                if xlim[0]<=cp[0]<=xlim[1] and ylim[0]<=cp[1]<=ylim[1]:
                    any_visible=True
        finally:
            # close() rather than letting it be collected: the generator takes the deflation operator
            # off the problem in its own finally, and an aborted scan must not leave it installed.
            gen.close()
            self.problem.set_deflation_operator(None)
            # Put the problem back on the last point that was RECORDED. A scan ends on whatever its
            # last attempt left behind, and its last attempt is by construction a failed one - the
            # hunt for new solutions stops when a deflated solve stops converging - so the dofs are a
            # diverged state that is on no branch. Everything downstream assumes current_point is what
            # the problem holds: the next command starts from it, and a second Deflated continuation
            # went straight to inf/nan residuals because its opening solve began there.
            if self.current_point is not None:
                self.load_pt(self.current_point)
            # The parameter was moved by hand, so oomph's arclength tangent describes a step that was
            # never taken. An ordinary Step from here has to start a fresh one.
            self.problem.reset_arc_length_parameters()
            self._tangs={}
            for b in by_index.values():
                self.propagate_stability(b)
            self.save_all()
            self._changed()
        self.log("Deflated continuation created {:d} branch{:s}".format(
            len(by_index),"" if len(by_index)==1 else "es"))
        return len(by_index)

    def scan_stripe(self,point=None)->bool:
        """Find every eigenvalue in the stripe ``|Re|<stripe_re``, ``|Im|<stripe_im`` at one point.

        A shift-invert solve returns the eigenvalues nearest the shift, so a Hopf pair far up the
        imaginary axis is simply not in the answer. This asks the other question - everything inside a
        region - which is what the contour method is for.

        Only the SLEPc solver can do it, and only against a complex PETSc build. Merging (the default)
        keeps whatever the point already had and adds what the scan found, which is the useful
        combination: the shift-invert spectrum plus the pairs it missed.
        """
        if self.on_orbit():
            raise RuntimeError("A stripe scan solves an eigenproblem, which the problem refuses while an orbit is installed. The stability of an orbit is its Floquet multipliers.")
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
            # str, not repr, when there is something to read: the messages the eigensolver raises here
            # explain themselves (a real-scalar PETSc cannot integrate a contour at all), and repr()
            # buried that behind "PETSc.Error(56)"-style noise.
            msg=str(e).strip()
            self.log("The stripe scan failed: "+(msg if msg else repr(e)))
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
        # A located bifurcation's tracked eigenvalue is the tracker's EXACT zero, and no eigensolve
        # taken here can produce it again: the tracker is long gone and this state is precisely where
        # the Jacobian is singular. It is carried across whatever is stored, or a stripe scan at a
        # bifurcation would replace it by a rounding-error-sized number and quietly demote the point
        # to an ordinary one, taking its classification and its departure arrows with it.
        tracked=None
        tracked_mode=0.0
        if point.eig_value_Re==0 and point.tracked_eigenindex is not None \
                and point.tracked_eigenindex<len(point.eig_values):
            tracked=point.eig_values[point.tracked_eigenindex]
            if point.eig_modes is not None and point.tracked_eigenindex<len(point.eig_modes):
                tracked_mode=point.eig_modes[point.tracked_eigenindex]
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
        if tracked is not None:
            # Drop whatever this solve found in its place - one entry per mode, and the exact one is
            # the one to keep - and put the tracked value back. Uniform over merge: with merging the
            # dedup above has already kept the exact copy, so this only reinstates it after a replace.
            scale=max(1.0,abs(tracked))
            keptv=[v for v in vals if abs(v-tracked)>1e-8*scale]
            keptm=([m for v,m in zip(vals,mods) if abs(v-tracked)>1e-8*scale]
                   if mods is not None else None)
            vals=[tracked]+keptv
            mods=None if keptm is None else [tracked_mode]+keptm
        order=sorted(range(len(vals)),key=lambda i: -numpy.real(vals[i]))
        point.eig_values=[vals[i] for i in order]
        point.eig_modes=[mods[i] for i in order] if mods is not None else None
        if point.eig_values:
            point.eig_value_Re=numpy.real(point.eig_values[0])
            point.eig_value_Im=numpy.imag(point.eig_values[0])
        # None on an ordinary point: a spectrum solved without a tracker installed has no tracked
        # entry, and a stale index would mark an unrelated eigenvalue after the re-sort.
        point.tracked_eigenindex=(None if tracked is None else
                                  next((i for i,v in enumerate(point.eig_values) if v==tracked),None))
        if tracked is not None and point.tracked_eigenindex is not None:
            point.eig_value_Re=numpy.real(tracked)
            point.eig_value_Im=numpy.imag(tracked)
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
        if self._augmented_system_active():
            self.log("Not refining towards an eigenfunction: an augmented system is active (a "
                     "bifurcation tracker or a periodic orbit), and adapting would renumber it out "
                     "from under the handler")
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
        re=numpy.real(numpy.asarray(evs))
        # Nearest the axis is the wrong pick right after a crossing. Two modes going unstable within a
        # few thousandths of each other - which is what a periodic domain does routinely - leave the
        # next STABLE eigenvalue marginally nearer zero than the two that just crossed, and the fold
        # tracker then goes looking for a bifurcation the branch has not reached, from a starting guess
        # that belongs to a different mode. It diverges, or converges to something unrelated. So when
        # the last step is what made this branch (more) unstable, take the least unstable of the modes
        # now on the wrong side: that is the one whose crossing lies between the last two points.
        prev=self._point_before_current()
        cur=self.current_point
        if (cur is not None and prev is not None and cur.unstable_count is not None
                and prev.unstable_count is not None and cur.unstable_count>prev.unstable_count):
            positive=[i for i in range(len(re)) if re[i]>0]
            if positive:
                return int(min(positive,key=lambda i:re[i]))
        return int(numpy.argmin(numpy.abs(re)))

    def _point_before_current(self):
        """The point recorded just before the current one on the current branch, if there is one."""
        branch=self.current_branch
        if branch is None or self.current_point is None or len(branch)<2:
            return None
        try:
            i=branch.index(self.current_point)
        except ValueError:
            return branch[-2]
        return branch[i-1] if i>0 else None

    def locate_bifurcation(self,pitchfork:bool=False,eigenindex:int | None=None):
        """Find the nearest bifurcation and record it.

        ``eigenindex`` selects which eigenvalue is expected to cross; by default the one nearest the
        imaginary axis (:py:meth:`critical_eigenindex`). Pass it explicitly to track a particular mode -
        the eigenvalue list in the Points tab is there to read the index off.
        """
        if self.on_locus():
            raise RuntimeError("Already following a bifurcation. Leave the locus first "
                               "(Bifurcation -> Leave the locus) to work on an ordinary branch.")
        if self.on_orbit():
            raise RuntimeError("A bifurcation OF AN ORBIT is a Floquet multiplier leaving the unit "
                               "circle, which the trackers here cannot locate. The multipliers in the "
                               "Points tab show it coming; leave the orbit branch to track a "
                               "stationary bifurcation.")
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
        # The teardown belongs HERE, with the call that switched tracking on, not in whichever caller
        # happens to catch the failure. The tracking solve is the one that can diverge, and when it did
        # the augmented system stayed installed: every later solve then worked on the wrong problem,
        # and the next attempt to locate a bifurcation reported "a non-zero shift is required" from its
        # opening eigensolve - the leftover tracker's message, with nothing about the solve that
        # actually failed. Two of the three entry points (Locate pitchfork, Locate the bifurcation of
        # the selected eigenvalue) call this directly and had no cleanup at all.
        try:
            self.problem.solve(max_newton_iterations=20)
            # The whole spectrum, not just the tracked eigenvalue the solve leaves behind: on a branch
            # that is already unstable, which modes are unstable HERE is exactly what says whether the
            # right bifurcation was found, and a second eigenvalue sitting on the axis alongside the
            # tracked one is a codim-2 point the diagram would otherwise never show. See
            # _tracked_spectrum for the shift it needs and how the tracked value is kept unique.
            crit,spectrum,modes,tracked=self._tracked_spectrum()
            if spectrum is not None:
                self.log("Spectrum at the bifurcation: {:d} eigenvalue{:s}, {:d} unstable; the tracked "
                         "one is #{:d}".format(len(spectrum),"" if len(spectrum)==1 else "s",
                         sum(1 for v in spectrum if numpy.real(v)>0),tracked if tracked is not None else 0))
            self._add_current_state(eig_value=crit,eig_values=spectrum,eig_modes=modes,
                                    tracked_eigenindex=tracked)
            self._update_tangents()
        finally:
            self.problem.deactivate_bifurcation_tracking()
        self.reorder_branch_upon_point_insertion(self._get_current_branch(),self._get_current_point())

    #: Solve the eigenproblem once when a bifurcation whose classification was READ BACK from a saved
    #: diagram is loaded, to recover the null vector that state.json does not store - which is what its
    #: arrows and its branch switching are built from. Off means a reloaded bifurcation keeps its label
    #: and its type but shows no arrows until something else solves an eigenproblem there.
    recover_restored_normal_forms=True

    def leave_bifurcation(self)->bool:
        """Take the way off the bifurcation we are sitting on that its TYPE offers.

        A transcritical or a pitchfork has a second steady branch through it, so the answer is a branch
        switch. A fold and a Hopf do not: a fold turns the one branch around and a Hopf sheds a periodic
        orbit, so nothing steady leaves them and the answer is to leave transiently and see where the
        solution ends up. Asking for a branch switch at a fold used to be the only offer, and it could
        only ever decline.

        Continuing the branch itself is unaffected and stays on ``step``: past a branch point or a Hopf
        that is an ordinary step, and around a fold it is the null-vector restart.
        """
        cp=self._get_current_point()
        mode=self._point_normal_mode(cp)
        if mode is not None:
            # Every way off a bifurcation offered here - a branch switch, an orbit, a transient
            # departure along the eigenvector - moves along the critical eigenfunction, and this one
            # is not a direction in these degrees of freedom. Said here rather than left to come out
            # as "the type could not be determined", which is not the reason.
            self.log("This is a bifurcation of the {:s} = {:g}. What leaves it breaks the symmetry "
                     "these degrees of freedom impose, so there is no way off it in this problem - "
                     "follow it in a second parameter instead (Bifurcation -> Continue the "
                     "bifurcation), or resolve the {:s} direction explicitly."
                     .format("azimuthal mode m" if mode[0]=="m" else "Cartesian mode k",mode[1],
                             "azimuthal" if mode[0]=="m" else "extra Cartesian"))
            return False
        if cp.bifurcation_info is None:
            # Same reasoning as in branch_switch: the problem is sitting at the bifurcation, which is
            # all the normal form needs, so it is cheaper to classify here than to send the user away.
            self.log("This bifurcation was not classified; computing its normal form now")
            cp.bifurcation_info=self._classify_current_point(cp)
        btype=cp.bifurcation_info.get("type") if cp.bifurcation_info is not None else None
        if btype in ("transcritical","pitchfork"):
            return bool(self.branch_switch())
        if btype=="hopf":
            # Used to leave transiently, on the grounds that nothing STEADY leaves a Hopf. What does
            # leave it is the periodic orbit, which the diagram can now follow; a transient departure
            # stays available on its own key for a Hopf whose orbit cannot be started.
            refusal=self.orbit_can_be_started()
            if refusal is None:
                self.log("A Hopf sheds a periodic orbit - switching onto it")
                return bool(self.switch_to_orbit())
            self.log(refusal+" Leaving the branch transiently instead.")
            return bool(self.transient_leave_branch())
        if btype=="fold":
            self.log("A fold has no second steady branch, so this leaves it transiently")
            return bool(self.transient_leave_branch())
        self.log("The type of this bifurcation could not be determined, so there is no way off it to "
                 "choose. Try 'Switch branch' or leaving the branch transiently by hand.")
        return False

    def locate_bifurcation_or_switch(self,eigenindex:int | None=None):
        """What ``b`` has always done: at a bifurcation, leave it; otherwise, go find one."""
        if self.current_point is not None and self.current_point.eig_value_Re==0:
            self.leave_bifurcation()
        else:
            try:
                self.locate_bifurcation(eigenindex=eigenindex)
            except Exception as e:
                # A backstop only: locate_bifurcation switches tracking off in its own finally, and
                # the other way off a bifurcation (leave_bifurcation -> branch_switch /
                # transient_leave_branch) manages its own. Left here because the cost of being wrong
                # about that is every later solve silently working on an augmented system.
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

    def arclength_metric(self)->str:
        """Which metric the arclength currently measures the solution in.

        "dofsum" (oomph's own, i.e. no inner product set), "ndof", "l2", or "custom" for a callable
        weighting installed on the problem directly. This is what the menu's radio group reads.
        """
        kind=self.problem._arclength_inner_product
        if kind is None:
            return "dofsum"
        if callable(kind):
            return "custom"
        return "l2" if kind=="mass" else str(kind)

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
        self._resolve_orbit_period_unit()
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
        self._resolve_initial_observable()
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
        kind,name=as_axis(spec)
        # An orbit shows a band, and an autoscale that only saw the average would clip it.
        band=orbit_band_names(name) if kind==AXIS_OBSERVABLE else ()
        for b in self.branches:
            for p in b:
                try:
                    v=p.value_of(spec)
                except KeyError:
                    continue    # a branch that never recorded this quantity
                lo=min(lo,v)
                hi=max(hi,v)
                found=True
                for edge in band:
                    if edge in p.obs_values:
                        lo=min(lo,p.obs_values[edge])
                        hi=max(hi,p.obs_values[edge])
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
