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

"""Data model of a bifurcation diagram: solution points and the branches they form.

Deliberately free of matplotlib and tkinter, so the diagram can be built, saved and reloaded
headlessly - see :py:mod:`~pyoomph.utils.bifurcation_gui.controller` for the numerics and
:py:mod:`~pyoomph.utils.bifurcation_gui.tkapp` for the user interface.
"""

from collections import UserList
import numbers
import numpy
from scipy.interpolate import UnivariateSpline

from ...typings import *


#: Where a point's stability came from. "eigen" = an eigenproblem was solved there; "inferred" =
#: carried along the branch from such a point and flipped at each determinant-sign change, which is
#: blind to a Hopf; "unknown" = neither.
STABILITY_EIGEN="eigen"
STABILITY_INFERRED="inferred"
STABILITY_UNKNOWN="unknown"


#: What a branch is. "solution" is an ordinary branch of stationary states; "locus" a curve of
#: bifurcation points, varying two parameters; "orbit" a branch of periodic orbits, whose points
#: carry a period and a whole cycle's worth of each observable rather than one value.
BRANCH_SOLUTION="solution"
BRANCH_LOCUS="locus"
BRANCH_ORBIT="orbit"


#: An orbit point records a whole cycle of each observable, not one value. The cycle's AVERAGE is
#: stored under the observable's own name, so that every axis, tangent, export and selection path
#: keeps working and an orbit branch continues the stationary line straight through its Hopf; the
#: extremes go under these derived names. Two spaces, like the ExtremumObservables tags, so the axis
#: menus leave them alone.
ORBIT_MIN_TAG="  [orbit min]"
ORBIT_MAX_TAG="  [orbit max]"

#: The period, offered as an ordinary observable rather than a third kind of axis: everything that
#: already carries observables - the menus, value_of, the export, state.json - then carries it too.
#: A stationary point does not have it, which is exactly why a period-vs-parameter plot shows the
#: orbit branches alone.
ORBIT_T_KEY="orbit/T"


def orbit_min_name(name:str)->str:
    return name+ORBIT_MIN_TAG


def orbit_max_name(name:str)->str:
    return name+ORBIT_MAX_TAG


def orbit_band_names(name:str)->"tuple[str,str]":
    """The (min, max) observable names belonging to one observable's band."""
    return name+ORBIT_MIN_TAG,name+ORBIT_MAX_TAG


def is_orbit_band_name(name:str)->bool:
    return name.endswith(ORBIT_MIN_TAG) or name.endswith(ORBIT_MAX_TAG)


def orbit_band_base(name:str)->"str | None":
    """The observable a band name belongs to, or None when it is not one."""
    for tag in (ORBIT_MIN_TAG,ORBIT_MAX_TAG):
        if name.endswith(tag):
            return name[:-len(tag)]
    return None


#: What a plot axis shows: ``("observable", name)`` or ``("parameter", name)``. Having parameters and
#: observables on the same footing is what lets one drawing path produce both an ordinary diagram
#: (parameter vs observable) and the locus of a bifurcation in a plane of two parameters.
AxisSpec:TypeAlias="tuple[str,str]"

AXIS_OBSERVABLE="observable"
AXIS_PARAMETER="parameter"


def observable_axis(name:str)->"AxisSpec":
    return (AXIS_OBSERVABLE,name)


def parameter_axis(name:str)->"AxisSpec":
    return (AXIS_PARAMETER,name)


def as_axis(spec)->"AxisSpec":
    """Accept a bare observable name as well as a full spec."""
    if isinstance(spec,str):
        return (AXIS_OBSERVABLE,spec)
    kind,name=spec
    if kind not in (AXIS_OBSERVABLE,AXIS_PARAMETER):
        raise ValueError("Unknown axis kind '"+str(kind)+"'")
    return (kind,name)


def eigen_settings(neigen:int,shift,modes:"Sequence[float] | None")->tuple:
    """Canonical, JSON-safe record of what a spectrum was computed with.

    Everything is reduced to plain scalars and a tuple, because this is compared for equality to decide
    whether a point is stale: a tuple written to state.json comes back as a LIST, so comparing the raw
    values would report every reloaded point as stale. The shift is split into real and imaginary parts
    for the same reason - json cannot write a complex.
    """
    sh=complex(shift) if shift is not None else complex(0.0)
    return (int(neigen),float(sh.real),float(sh.imag),
            tuple(float(m) for m in modes) if modes is not None else ())


def _normal_form_to_state(bi:dict | None)->dict | None:
    """The part of a normal form that can go into state.json: its numbers, not its vectors.

    What is dropped is zeta, which is one entry per degree of freedom, and the predictor closures built
    from it. Writing a dof-sized vector per bifurcation would change what state.json is - it is a small
    text file next to the binary dumps that hold the solutions - and the vector is recoverable at the
    point itself from one eigensolve, which is what BifurcationController._ensure_normal_form does.
    What is kept is enough to say WHAT the bifurcation is without touching the solver at all, which is
    what the diagram's labels and the choice of how to leave it need.
    """
    if not bi:
        return None
    out:dict[str,Any]={}
    for k,v in bi.items():
        if callable(v) or isinstance(v,numpy.ndarray):
            continue
        if isinstance(v,(bool,str)):
            out[k]=v
        elif isinstance(v,numbers.Integral):
            out[k]=int(v)
        elif isinstance(v,numbers.Real):
            out[k]=float(v)
        elif isinstance(v,numbers.Complex):
            # A Hopf's a and b are complex. Stored as a tagged pair rather than a bare [re, im] list,
            # so reading it back cannot confuse it with an entry that is genuinely a list of two.
            out[k]={"__complex__":[float(numpy.real(v)),float(numpy.imag(v))]}
    return out or None


def _normal_form_from_state(stored:dict | None)->dict | None:
    """Invert _normal_form_to_state. The result has no zeta and no predictors - see there."""
    if not stored:
        return None
    out:dict[str,Any]={}
    for k,v in stored.items():
        if isinstance(v,dict) and "__complex__" in v:
            re_im=v["__complex__"]
            out[k]=complex(float(re_im[0]),float(re_im[1]))
        else:
            out[k]=v
    return out


class BifurcationGUISolutionPoint:
    """One computed solution: every global parameter, all observables, the leading eigenvalue and
    the state file it was dumped to."""

    def __init__(self,param_value,obs_values,eig_value,statefile,outstep,param_values:dict[str,float] | None=None,
                 eig_values:"Sequence[complex] | None"=None,det_sign:int | None=None,
                 dparam_ds:float | None=None,eig_modes:"Sequence[float] | None"=None,
                 eig_settings:"tuple | None"=None,tracked_eigenindex:int | None=None) -> None:
        self.param_value=param_value
        #: Every global parameter at this point, not just the continued one. The state dump restores
        #: all of them, so without this the diagram could not say which slice of parameter space it
        #: is a section through - and reloading a point silently moves the others.
        self.param_values:dict[str,float]=dict(param_values) if param_values else {}
        #: False for points read from a state file written before the parameters were recorded. Such
        #: a point gets the continued parameter backfilled (that value IS known) but nothing else, so
        #: "empty param_values" cannot be used to mean "the slice was never recorded".
        self.param_values_complete=bool(param_values)
        self.obs_values=obs_values
        # The leading eigenvalue stays the primary field: a located bifurcation is flagged by its real
        # part being exactly zero. NaN when no eigenproblem was solved here (quick mode) - NOT zero,
        # which would make every such point look like a bifurcation, and comparisons against NaN are
        # False, so the existing "== 0" tests answer correctly without knowing about it.
        if eig_value is None:
            self.eig_value_Re=float("nan")
            self.eig_value_Im=float("nan")
        else:
            self.eig_value_Re=numpy.real(eig_value)
            self.eig_value_Im=numpy.imag(eig_value)
        #: The whole computed spectrum, for the eigenvalue list in the Points tab. Empty for points
        #: read from a state file written before it was recorded - which is why this cannot simply be
        #: assumed to start with the leading eigenvalue above.
        self.eig_values:list[complex]=[complex(v) for v in eig_values] if eig_values is not None else []
        #: Which normal mode each recorded eigenvalue belongs to, parallel to :py:attr:`eig_values`.
        #: None when only the base state was solved - which is also what the problem reports then, since
        #: get_last_eigenmodes_m() is None unless a mode list was passed. The kind of mode (azimuthal m
        #: or Cartesian k) is one per diagram and lives on the controller, not here.
        self.eig_modes:list[float] | None=[float(m) for m in eig_modes] if eig_modes is not None else None
        #: (neigen, shift, modes) the spectrum was computed with, so a point can be recognised as stale
        #: after the eigenvalue count is raised. None for points from before this was recorded.
        self.eig_settings:tuple | None=tuple(eig_settings) if eig_settings is not None else None
        #: Which entry of :py:attr:`eig_values` is the eigenvalue a bifurcation tracker was holding on
        #: the axis here, so the list can mark it. Set only where a tracker was installed AND the base
        #: state's own spectrum was solved for alongside it, which is what distinguishes a point whose
        #: whole spectrum is known from one carrying the tracker's synthetic value alone.
        self.tracked_eigenindex:int | None=int(tracked_eigenindex) if tracked_eigenindex is not None else None
        self.statefile=statefile
        self.outstep=outstep
        self.scoord:float=0
        self._tangs:dict[str,NPFloatArray]={}
        #: Directions off this point when it is a bifurcation, per observable and divided by |ds|:
        #: the branches a switch would reach, or the +-eigenvector a transient would leave along.
        #: Recomputed by the controller, never read from a file - see _update_departure_tangents.
        self._departure_tangs:list[Any]=[]
        #: "switch" or "perturb", saying which of the two the entries above are; None when there are none.
        self._departure_kind:str | None=None
        self.tag=-1
        self.bifurcation_info:dict | None=None
        #: Sign of the determinant of the plain Jacobian, from the factorisation the continuation step
        #: already computed. Only its CHANGES between neighbouring points mean anything - see
        #: GenericLinearSystemSolver.get_determinant_sign.
        self.det_sign=det_sign
        #: dparameter/ds of the continuation tangent. Reverses at a fold and nowhere else.
        self.dparam_ds=dparam_ds
        #: "fold" or "branch_point" when a test function changed between the previous point and this
        #: one. The bifurcation is BRACKETED by the two, not located at either.
        self.detected_bifurcation:str | None=None
        self.stability_source=STABILITY_EIGEN if eig_value is not None else STABILITY_UNKNOWN
        #: Number of eigenvalues with a positive real part: measured, or propagated in quick mode.
        self.unstable_count:int | None=None
        #: What makes this a point of a PERIODIC ORBIT rather than a stationary state: the period and
        #: the discretization it was computed with, plus where the orbit's own degrees of freedom were
        #: stored (a state dump holds only the base state, i.e. one phase of the cycle). None on a
        #: stationary point, which is also how "is this an orbit" is answered everywhere.
        self.orbit_info:dict | None=None
        #: Floquet multipliers, kept alongside the exponents in :py:attr:`eig_values`. The exponents
        #: drive the stability machinery (Re > 0 is ``|mu| > 1``); the multipliers say what KIND of
        #: bifurcation is approaching, which the exponents cannot - a multiplier leaving through -1 is
        #: a period doubling and a complex pair on the unit circle a torus, and both have the same
        #: exponent real part.
        self.floquet:list[complex]=[]

    @staticmethod
    def from_dict(res):
        inst=BifurcationGUISolutionPoint(res["param_value"],res["obs_value"],res["eig_value_Re"]+1j*res["eig_value_Im"],res["statefile"],res["outstep"])
        inst.scoord=res["scoord"]
        inst.tag=res.get("tag",-1)
        # Absent in files written before the slice was recorded. Left empty rather than filled with
        # the continued parameter alone, so "unknown slice" stays distinguishable from "no other
        # parameters exist".
        inst.param_values=dict(res.get("param_values",{}))
        inst.param_values_complete="param_values" in res
        inst.det_sign=res.get("det_sign")
        inst.dparam_ds=res.get("dparam_ds")
        inst.detected_bifurcation=res.get("detected_bifurcation")
        inst.bifurcation_info=_normal_form_from_state(res.get("bifurcation_info"))
        # A point written before quick mode existed always had a spectrum, so the default is "eigen" -
        # which is also what its NaN-free eig_value_Re says.
        inst.stability_source=res.get("stability_source",STABILITY_EIGEN)
        inst.unstable_count=res.get("unstable_count")
        re_list,im_list=res.get("eig_values_Re"),res.get("eig_values_Im")
        if re_list is not None and im_list is not None:
            inst.eig_values=[complex(r,i) for r,i in zip(re_list,im_list)]
        modes=res.get("eig_modes")
        inst.eig_modes=[float(m) for m in modes] if modes is not None else None
        tracked=res.get("tracked_eigenindex")
        inst.tracked_eigenindex=int(tracked) if tracked is not None else None
        settings=res.get("eig_settings")
        # Rebuilt through the canonicaliser, since json gives the nested modes back as a list and a
        # stale check compares these for equality.
        inst.eig_settings=(eigen_settings(settings[0],complex(settings[1],settings[2]),settings[3])
                           if settings is not None else None)
        orbit=res.get("orbit_info")
        inst.orbit_info=dict(orbit) if orbit else None
        fre,fim=res.get("floquet_Re"),res.get("floquet_Im")
        if fre is not None and fim is not None:
            inst.floquet=[complex(r,i) for r,i in zip(fre,fim)]
        for k,v in res["tangs"].items():
            inst._tangs[k]=numpy.array(v)
        return inst

    def to_state_dict(self):
        res={}
        res["param_value"]=self.param_value
        res["obs_value"]=self.obs_values
        res["eig_value_Re"]=self.eig_value_Re
        res["eig_value_Im"]=self.eig_value_Im
        res["statefile"]=self.statefile
        res["outstep"]=self.outstep
        res["scoord"]=self.scoord
        if self.tag>=0:
            res["tag"]=self.tag
        if self.param_values:
            res["param_values"]=dict(self.param_values)
        if self.det_sign is not None:
            res["det_sign"]=int(self.det_sign)
        if self.dparam_ds is not None:
            res["dparam_ds"]=float(self.dparam_ds)
        if self.detected_bifurcation is not None:
            res["detected_bifurcation"]=self.detected_bifurcation
        nf=_normal_form_to_state(self.bifurcation_info)
        if nf:
            res["bifurcation_info"]=nf
        if self.stability_source!=STABILITY_EIGEN:
            res["stability_source"]=self.stability_source
        if self.unstable_count is not None:
            res["unstable_count"]=int(self.unstable_count)
        if self.eig_values:
            # Two flat lists rather than pairs: shorter in the file and json-native. Note this is the
            # entry that makes state.json grow with neigen - see the comment in save_all.
            res["eig_values_Re"]=[float(numpy.real(v)) for v in self.eig_values]
            res["eig_values_Im"]=[float(numpy.imag(v)) for v in self.eig_values]
            if self.eig_modes is not None:
                res["eig_modes"]=[float(m) for m in self.eig_modes]
            if self.tracked_eigenindex is not None:
                res["tracked_eigenindex"]=int(self.tracked_eigenindex)
        if self.eig_settings is not None:
            n,sr,si,modes=self.eig_settings
            res["eig_settings"]=[n,sr,si,list(modes)]
        if self.orbit_info:
            res["orbit_info"]=dict(self.orbit_info)
        if self.floquet:
            # Two flat lists, like the spectrum above and for the same reason: json has no complex.
            res["floquet_Re"]=[float(numpy.real(v)) for v in self.floquet]
            res["floquet_Im"]=[float(numpy.imag(v)) for v in self.floquet]
        res["tangs"]={}
        for k,v in self._tangs.items():
            res["tangs"][k]=list(v)
        return res

    def eigenvalues_of_mode(self,mode:float | None)->"list[complex]":
        """The recorded eigenvalues belonging to one normal mode; ``None`` asks for all of them.

        With no mode scan at this point every eigenvalue IS the base state, so mode 0 returns the whole
        spectrum rather than nothing.
        """
        if mode is None or self.eig_modes is None:
            return list(self.eig_values)
        return [v for v,m in zip(self.eig_values,self.eig_modes) if m==mode]

    def measured_unstable_count(self,include_modes:bool=True)->int:
        """How many recorded eigenvalues have a positive real part.

        With ``include_modes=False`` only the base mode counts, i.e. the answer the diagram gave before
        normal modes could be computed at all. It is a real distinction and not a display detail: an
        axisymmetric state can be perfectly stable to m=0 and unstable to m=1, which is exactly what a
        polygonal hydraulic jump is.
        """
        values=self.eig_values if (include_modes or self.eig_modes is None) else self.eigenvalues_of_mode(0.0)
        return sum(1 for v in values if numpy.real(v)>0)

    def stability_indicator(self,trust_inferred:bool=True,include_modes:bool=True)->float:
        """A signed number the branch segmentation can read: <0 stable, >0 unstable, 0 a bifurcation.

        For a point where an eigenproblem was solved this IS the leading real part, so an ordinary
        diagram behaves exactly as before. For a quick-mode point it is the sign of the propagated
        unstable count, and NaN while that is unknown - which the segmentation turns into the neutral
        style it already uses for a piece straddling a change of stability.
        """
        if self.eig_value_Re==0:
            return 0.0          # a LOCATED bifurcation: the boundary itself
        if self.stability_source==STABILITY_EIGEN:
            # The count, when it is known, rather than the leading real part: they agree while the
            # spectrum is sorted by descending real part, but the count says what it means. With a mode
            # scan the stored count is the all-mode one, so restricting to the base mode has to recount.
            if not include_modes and self.eig_modes is not None:
                return 1.0 if self.measured_unstable_count(include_modes=False)>0 else -1.0
            if self.unstable_count is not None:
                return 1.0 if self.unstable_count>0 else -1.0
            return float(self.eig_value_Re)
        if trust_inferred and self.stability_source==STABILITY_INFERRED and self.unstable_count is not None:
            return 1.0 if self.unstable_count>0 else -1.0
        return float("nan")

    def fixed_parameters(self,varying:Iterable[str])->dict[str,float]:
        """The parameters this point holds fixed, i.e. all of them except the ones being varied."""
        varying=set(varying)
        return {n:v for n,v in self.param_values.items() if n not in varying}


    def value_of(self,spec)->float:
        """The value this point has along one axis.

        ``spec`` is an :py:data:`AxisSpec`: ``("observable", name)`` or ``("parameter", name)``. A
        bare string is taken as an observable, which is what every call site meant before an axis
        could be a parameter at all.
        """
        kind,name=as_axis(spec)
        if kind=="parameter":
            if name not in self.param_values:
                raise KeyError("Point has no recorded value for parameter '"+str(name)+"'")
            return self.param_values[name]
        return self.obs_values[name]

    def get_coordinate(self,yspec,with_s=False,with_eigen=False,xspec=None):
        """``[x, y..., (ReEig, ImEig), (s)]`` for this point.

        ``yspec`` may be a list of specs, in which case one column per entry is produced - that is
        how the all-observables export writes every observable next to the parameter.
        ``xspec`` defaults to the continued parameter's value, i.e. the historical x-axis.
        """
        x=self.param_value if xspec is None else self.value_of(xspec)
        if isinstance(yspec,list):
            res=[x]+[self.value_of(s) for s in yspec]
        else:
            res=[x,self.value_of(yspec)]
        if with_eigen:
            res=res+[self.eig_value_Re,self.eig_value_Im]
        if with_s:
            res=res+[self.scoord]
        return res

    def describe(self,obs=None,obs_unit:str="",eig_unit:str="")->str:
        """One-line human-readable summary, used by the point-info panel and the branch tree.

        The units are passed in rather than looked up: a point knows its numbers, the controller knows
        what they are measured in.
        """
        res="{:.6g}".format(self.param_value)
        if obs is not None:
            try:
                res+=" | {:.6g}".format(self.value_of(obs))+(" "+obs_unit if obs_unit else "")
            except KeyError:
                pass
        res+=" | eig {:.3g}{:+.3g}i".format(self.eig_value_Re,self.eig_value_Im)+(" "+eig_unit if eig_unit else "")
        if self.eig_value_Re==0:
            kind="bifurcation"
            if self.bifurcation_info is not None:
                kind=str(self.bifurcation_info.get("type",kind))
            res+=" ["+kind+"]"
        if self.tag>=0:
            res+=" #"+str(self.tag)
        return res



#: Relative tolerance for deciding whether two parameter values are "the same" fixed value.
#: The parameters that are not being continued stay bit-identical through an arclength step, so this
#: only has to absorb values the user typed in or reached via go_to_param.
SLICE_RTOL=1e-9


def same_parameter_value(a:float,b:float,rtol:float=SLICE_RTOL)->bool:
    return abs(a-b)<=rtol*max(1.0,abs(a),abs(b))


class BifurcationGUISolutionBranch(UserList[BifurcationGUISolutionPoint]):
    """A list of solution points forming one branch, ordered by their arclength coordinate.

    A branch is produced by one continuation, so it belongs to one *slice* of parameter space: one
    or two parameters vary along it and every other one is held fixed. Which is which is recorded
    here, because a diagram that does not say what was held fixed cannot be interpreted - let alone
    published - once the problem has more than one global parameter.
    """

    def __init__(self,initlist=None,*,kind:str=BRANCH_SOLUTION,continuation_parameter:str | None=None,
                 tracked_parameter:str | None=None,bifurcation_type:str | None=None) -> None:
        super().__init__(initlist or [])
        #: What this branch is. "solution" is an ordinary branch of stationary states, continued in
        #: one parameter. "locus" is a curve of bifurcation points, tracked in ``tracked_parameter``
        #: while being continued in ``continuation_parameter``, so two parameters vary along it.
        #: Written as running text rather than an aligned list, which sphinx read as an indented block.
        self.kind=kind
        self.continuation_parameter=continuation_parameter
        self.tracked_parameter=tracked_parameter
        self.bifurcation_type=bifurcation_type

    @property
    def varying_parameters(self)->list[str]:
        res=[n for n in (self.continuation_parameter,self.tracked_parameter) if n is not None]
        return res

    def fixed_parameters(self)->dict[str,float]:
        """The slice this branch sits in, derived from its points rather than assumed.

        Deriving it matters: a custom key function calling go_to_param mid-branch would otherwise
        leave the branch labelled with a slice it has since left. See :py:meth:`slice_is_consistent`.
        """
        for p in self:
            if p.param_values:
                return p.fixed_parameters(self.varying_parameters)
        return {}

    def slice_is_consistent(self)->bool:
        """False if the supposedly fixed parameters actually move along this branch."""
        ref=None
        for p in self:
            if not p.param_values:
                continue
            fixed=p.fixed_parameters(self.varying_parameters)
            if ref is None:
                ref=fixed
            elif set(fixed)!=set(ref) or any(not same_parameter_value(fixed[n],ref[n]) for n in fixed):
                return False
        return True

    def slice_key(self)->tuple:
        """Hashable identity of the slice, for grouping branches and for the export directories."""
        fixed=self.fixed_parameters()
        return (self.kind,self.continuation_parameter,self.tracked_parameter,
                tuple(sorted((n,float(v)) for n,v in fixed.items())))

    def slice_is_known(self)->bool:
        """False for branches read from a file written before parameters were recorded.

        Distinguishing this from "there are no other parameters" is the whole point: an unknown
        slice must be reported as unknown, never silently drawn as if nothing were held fixed.
        """
        return any(p.param_values_complete for p in self)

    def describe_slice(self)->str:
        """Human-readable "b = 0.3, c = 2" for the status bar, the plot and file headers."""
        if not self.slice_is_known():
            return "slice unknown"
        fixed=self.fixed_parameters()
        if not fixed:
            return "no other parameters"
        return ", ".join("{:s} = {:.6g}".format(n,v) for n,v in sorted(fixed.items()))

    def describe(self)->str:
        """One line for a branch list: what this branch is, not merely how long it is.

        "12 points" alone cannot be told apart from another diagram in a different parameter or from
        a curve of bifurcations, which is the information actually needed when several are on screen.
        """
        parts=["{:d} point{:s}".format(len(self),"" if len(self)==1 else "s")]
        if self.kind==BRANCH_LOCUS:
            parts.append("{:s} locus: {:s} tracked, continued in {:s}".format(
                self.bifurcation_type or "bifurcation",
                str(self.tracked_parameter),str(self.continuation_parameter)))
        elif self.kind==BRANCH_ORBIT:
            info=next((p.orbit_info for p in self if p.orbit_info),None) or {}
            what="periodic orbits"
            if info.get("nT"):
                what+=", {:d} time steps".format(int(info["nT"]))
            if info.get("mode"):
                what+=" ({:s}".format(str(info["mode"]))
                if info.get("order"):
                    what+=" order {:d}".format(int(info["order"]))
                what+=")"
            parts.append(what)
            if self.continuation_parameter is not None:
                parts.append("continued in "+self.continuation_parameter)
        elif self.continuation_parameter is not None:
            parts.append("continued in "+self.continuation_parameter)
        if not self.slice_is_known():
            parts.append("slice unknown")
        elif self.fixed_parameters():
            parts.append("at "+self.describe_slice())
        return " | ".join(parts)

    def to_state_dict(self):
        res={}
        res["points"]=[p.to_state_dict() for p in self]
        res["kind"]=self.kind
        if self.continuation_parameter is not None:
            res["continuation_parameter"]=self.continuation_parameter
        if self.tracked_parameter is not None:
            res["tracked_parameter"]=self.tracked_parameter
        if self.bifurcation_type is not None:
            res["bifurcation_type"]=self.bifurcation_type
        return res

    @staticmethod
    def from_dict(res,default_continuation_parameter:str | None=None):
        inst=BifurcationGUISolutionBranch(
            kind=res.get("kind",BRANCH_SOLUTION),
            # A file written before the slice was recorded has only ever been continued in the
            # parameter the GUI was constructed with, so that is the honest default here.
            continuation_parameter=res.get("continuation_parameter",default_continuation_parameter),
            tracked_parameter=res.get("tracked_parameter"),
            bifurcation_type=res.get("bifurcation_type"))
        for p in res["points"]:
            inst.append(BifurcationGUISolutionPoint.from_dict(p))
        # A legacy point knows the continued parameter's value - it is param_value - just not under
        # that name. Backfilling it lets such a diagram still be plotted against its own parameter,
        # while slice_is_known() keeps reporting the rest as unrecorded.
        if inst.continuation_parameter is not None:
            for p in inst:
                if not p.param_values_complete:
                    p.param_values.setdefault(inst.continuation_parameter,p.param_value)
        return inst


    def to_point_list(self,yspec,xspec=None):
        res=[]
        for p in self:
            res.append(p.get_coordinate(yspec,xspec=xspec))
        return numpy.array(res)

    def smooth_point_list(self,yspecs,subsampling=10,xspec=None):
        """One splined polyline over the WHOLE branch, with no stability segmentation.

        The segmented version cannot be used for the orbit min/max band: its segments overlap by one
        point at every change of stability (each transition piece repeats the point before it), and a
        shaded band built per segment therefore double-covers every join and darkens visibly there.
        The band is one polygon over the whole branch; the LINE carries the stability.
        """
        specs=yspecs if isinstance(yspecs,list) else [yspecs]
        if len(self)<=1:
            return self.to_point_list(specs,xspec=xspec)
        s=[p.scoord for p in self]
        k=min(3,len(s)-1)
        ss=numpy.linspace(s[0],s[-1],subsampling*(len(s)-1)+1)
        xs=[p.param_value if xspec is None else p.value_of(xspec) for p in self]
        cols=[UnivariateSpline(s,xs,s=0,k=k)(ss)]
        cols+=[UnivariateSpline(s,[p.value_of(sp) for p in self],s=0,k=k)(ss) for sp in specs]
        return numpy.column_stack(cols)

    def orbit_band(self,yspec,xspec=None,smooth:bool=False,subsampling=10):
        """``(x, lo, hi)`` of this branch's min/max band, or None when it has none.

        None rather than an exception for every ordinary reason a branch has no band: it is not an
        orbit branch, the vertical axis is a parameter or the period, or the branch was recorded
        before the extremes were being kept.
        """
        if self.kind!=BRANCH_ORBIT or len(self)<2:
            return None
        kind,name=as_axis(yspec)
        if kind!=AXIS_OBSERVABLE:
            return None
        lo,hi=orbit_band_names(name)
        if any(lo not in p.obs_values or hi not in p.obs_values for p in self):
            return None
        specs=[observable_axis(lo),observable_axis(hi)]
        try:
            pts=(self.smooth_point_list(specs,subsampling=subsampling,xspec=xspec) if smooth
                 else self.to_point_list(specs,xspec=xspec))
        except KeyError:
            return None
        return pts[:,0],pts[:,1],pts[:,2]

    def smooth_branch_stab_list(self,yspec,subsampling=10,xspec=None,trust_inferred:bool=True,
                                include_modes:bool=True):
        """Like :py:meth:`to_branch_stab_list`, with each segment resampled through a spline."""
        if len(self)<=1:
            return self.to_branch_stab_list(yspec,xspec=xspec,trust_inferred=trust_inferred,
                                            include_modes=include_modes)
        s=[p.scoord for p in self]
        # One spline per column, so a list of y specs (the all-observables export) works exactly as
        # a single one does. The previous version special-cased that into a list of splines it could
        # then not evaluate, and the all-observables path raised instead of running.
        yspecs=yspec if isinstance(yspec,list) else [yspec]
        x=[p.param_value if xspec is None else p.value_of(xspec) for p in self]
        order=min(3,len(s)-1)
        xi=UnivariateSpline(s,x,s=0,k=order)
        yis=[UnivariateSpline(s,[p.value_of(sp) for p in self],s=0,k=order) for sp in yspecs]
        eigRei=UnivariateSpline(s,[p.eig_value_Re for p in self],s=0,k=order)
        eigImi=UnivariateSpline(s,[p.eig_value_Im for p in self],s=0,k=order)
        segs,stabs=self.to_branch_stab_list(yspec,xspec=xspec,trust_inferred=trust_inferred,
                                            include_modes=include_modes)
        smoothsegs=[]
        for seg in segs:
            if len(seg)==1:
                smoothsegs.append(seg)
            else:
                sseg=[]
                for pi in range(len(seg)-1):
                    s0=seg[pi,-1]
                    s1=seg[pi+1,-1]
                    if pi==len(seg)-2:
                        ssamp=numpy.linspace(s0,s1,subsampling+1,endpoint=True)
                    else:
                        ssamp=numpy.linspace(s0,s1,subsampling,endpoint=False)
                    xs=xi(ssamp)
                    ys=[yf(ssamp) for yf in yis]
                    eR=eigRei(ssamp)
                    eI=eigImi(ssamp)
                    for k in range(len(ssamp)):
                        sseg.append([xs[k]]+[yv[k] for yv in ys]+[eR[k],eI[k],ssamp[k]])
                smoothsegs.append(numpy.array(sseg))
        return smoothsegs,stabs


    def to_branch_stab_list(self,yspec,xspec=None,trust_inferred:bool=True,include_modes:bool=True):
        """Split the branch into segments of constant stability.

        Returns ``(segments, stabilities)``; each segment is an array of
        ``[x, y..., ReEig, ImEig, s]`` rows and the matching stability is True (stable), False
        (unstable) or None for the short piece that straddles a change of stability.
        """
        def coord(p):
            return numpy.array(p.get_coordinate(yspec,with_s=True,with_eigen=True,xspec=xspec))
        # Read through stability_indicator() rather than eig_value_Re directly, so a quick-mode point
        # (no eigenproblem solved) can contribute its propagated stability, or NaN while that is not
        # known. For a branch where every point has a spectrum this returns exactly eig_value_Re, so the
        # behaviour of an ordinary diagram is unchanged.
        def sv(p):
            return p.stability_indicator(trust_inferred,include_modes)
        def unknown(v):
            return v!=v          # NaN
        res:list[NPFloatArray]=[]
        stabs:list[bool | None]=[]
        if len(self)==0:
            return res,[]
        if len(self)==1:
            return numpy.array([[coord(self[0])]]),[None]
        # Precomputed, because the stability AFTER a bifurcation is read off the points that follow it
        # rather than assumed to be the opposite of what came before. Flipping is wrong whenever the
        # bifurcation belongs to a second eigenvalue: on a branch that is already unstable, a further
        # eigenvalue crossing makes it MORE unstable, and flipping marked what follows as stable.
        svs=[sv(p) for p in self]

        def stability_from(index:int)->bool | None:
            """Stability of the first point at or after `index` whose own stability is known."""
            for j in range(index,len(svs)):
                v=svs[j]
                if not unknown(v) and v!=0:
                    return bool(v<0)
            return None

        currseg:list[NPFloatArray]=[]
        currstab=stability_from(0)
        for ip,(p1,p2) in enumerate(zip(self,self[1:])):
            s1,s2=svs[ip],svs[ip+1]
            if unknown(s1) or unknown(s2):
                # Nothing can be claimed across a pair whose stability is not known on both sides, so
                # it becomes the same neutral piece a change of stability produces.
                if len(currseg)>0:
                    res.append(numpy.array(currseg))
                    stabs.append(currstab)
                    currseg=[]
                res.append(numpy.array([coord(p1),coord(p2)]))
                stabs.append(None)
                if not unknown(s2) and s2!=0:
                    currstab=bool(s2<0)
            elif s2==0:
                if len(currseg)==0:
                    currseg.append(coord(p1))
                currseg.append(coord(p2))
                res.append(numpy.array(currseg))
                stabs.append(currstab)
                currseg=[]
                after=stability_from(ip+2)
                currstab=after if after is not None else (None if currstab is None else not currstab)
            elif s1*s2>=0: # Same stability
                if len(currseg)==0:
                    currseg.append(coord(p1))
                currseg.append(coord(p2))
            else: # Change in stability
                if len(currseg)>0:
                    res.append(numpy.array(currseg))
                    stabs.append(currstab)
                    currseg=[]
                res.append(numpy.array([coord(p1),coord(p2)]))
                stabs.append(None)
                currstab=bool(s2<0)
        if len(currseg)>0:
            res.append(numpy.array(currseg))
            stabs.append(currstab)
        return res,stabs


from ...typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
