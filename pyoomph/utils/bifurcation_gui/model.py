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
import numpy
from scipy.interpolate import UnivariateSpline

from ...typings import *


#: Where a point's stability came from. "eigen" = an eigenproblem was solved there; "inferred" =
#: carried along the branch from such a point and flipped at each determinant-sign change, which is
#: blind to a Hopf; "unknown" = neither.
STABILITY_EIGEN="eigen"
STABILITY_INFERRED="inferred"
STABILITY_UNKNOWN="unknown"


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


class BifurcationGUISolutionPoint:
    """One computed solution: every global parameter, all observables, the leading eigenvalue and
    the state file it was dumped to."""

    def __init__(self,param_value,obs_values,eig_value,statefile,outstep,param_values:dict[str,float] | None=None,
                 eig_values:"Sequence[complex] | None"=None,det_sign:int | None=None,
                 dparam_ds:float | None=None) -> None:
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
        self.statefile=statefile
        self.outstep=outstep
        self.scoord:float=0
        self._tangs:dict[str,NPFloatArray]={}
        self._branch_switch_tangs:list[Any]=[]
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
        # A point written before quick mode existed always had a spectrum, so the default is "eigen" -
        # which is also what its NaN-free eig_value_Re says.
        inst.stability_source=res.get("stability_source",STABILITY_EIGEN)
        inst.unstable_count=res.get("unstable_count")
        re_list,im_list=res.get("eig_values_Re"),res.get("eig_values_Im")
        if re_list is not None and im_list is not None:
            inst.eig_values=[complex(r,i) for r,i in zip(re_list,im_list)]
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
        if self.stability_source!=STABILITY_EIGEN:
            res["stability_source"]=self.stability_source
        if self.unstable_count is not None:
            res["unstable_count"]=int(self.unstable_count)
        if self.eig_values:
            # Two flat lists rather than pairs: shorter in the file and json-native. Note this is the
            # entry that makes state.json grow with neigen - see the comment in save_all.
            res["eig_values_Re"]=[float(numpy.real(v)) for v in self.eig_values]
            res["eig_values_Im"]=[float(numpy.imag(v)) for v in self.eig_values]
        res["tangs"]={}
        for k,v in self._tangs.items():
            res["tangs"][k]=list(v)
        return res

    def measured_unstable_count(self)->int:
        """How many of the recorded eigenvalues have a positive real part."""
        return sum(1 for v in self.eig_values if numpy.real(v)>0)

    def stability_indicator(self,trust_inferred:bool=True)->float:
        """A signed number the branch segmentation can read: <0 stable, >0 unstable, 0 a bifurcation.

        For a point where an eigenproblem was solved this IS the leading real part, so an ordinary
        diagram behaves exactly as before. For a quick-mode point it is the sign of the propagated
        unstable count, and NaN while that is unknown - which the segmentation turns into the neutral
        style it already uses for a piece straddling a change of stability.
        """
        if self.stability_source==STABILITY_EIGEN:
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

    def describe(self,obs=None)->str:
        """One-line human-readable summary, used by the point-info panel and the branch tree."""
        res="{:.6g}".format(self.param_value)
        if obs is not None:
            try:
                res+=" | {:.6g}".format(self.value_of(obs))
            except KeyError:
                pass
        res+=" | eig {:.3g}{:+.3g}i".format(self.eig_value_Re,self.eig_value_Im)
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

    def __init__(self,initlist=None,*,kind:str="solution",continuation_parameter:str | None=None,
                 tracked_parameter:str | None=None,bifurcation_type:str | None=None) -> None:
        super().__init__(initlist or [])
        #: "solution" - an ordinary branch of stationary states, continued in one parameter.
        #: "locus"    - a curve of bifurcation points, tracked in `tracked_parameter` while being
        #:              continued in `continuation_parameter`; two parameters vary along it.
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
        if self.kind=="locus":
            parts.append("{:s} locus: {:s} tracked, continued in {:s}".format(
                self.bifurcation_type or "bifurcation",
                str(self.tracked_parameter),str(self.continuation_parameter)))
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
            kind=res.get("kind","solution"),
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

    def smooth_branch_stab_list(self,yspec,subsampling=10,xspec=None,trust_inferred:bool=True):
        """Like :py:meth:`to_branch_stab_list`, with each segment resampled through a spline."""
        if len(self)<=1:
            return self.to_branch_stab_list(yspec,xspec=xspec,trust_inferred=trust_inferred)
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
        segs,stabs=self.to_branch_stab_list(yspec,xspec=xspec,trust_inferred=trust_inferred)
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


    def to_branch_stab_list(self,yspec,xspec=None,trust_inferred:bool=True):
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
            return p.stability_indicator(trust_inferred)
        def unknown(v):
            return v!=v          # NaN
        res=[]
        stabs=[]
        if len(self)==0:
            return res,[]
        if len(self)==1:
            return numpy.array([[coord(self[0])]]),[None]
        currseg=[]
        currstab=sv(self[0])<0
        if unknown(sv(self[0])) or sv(self[0])==0:
            currstab=sv(self[1])<0
        for p1,p2 in zip(self,self[1:]):
            s1,s2=sv(p1),sv(p2)
            if unknown(s1) or unknown(s2):
                # Nothing can be claimed across a pair whose stability is not known on both sides, so
                # it becomes the same neutral piece a change of stability produces.
                if len(currseg)>0:
                    res.append(numpy.array(currseg))
                    stabs.append(currstab)
                    currseg=[]
                res.append(numpy.array([coord(p1),coord(p2)]))
                stabs.append(None)
                currstab=(s2<0) if not unknown(s2) else currstab
            elif s2==0:
                if len(currseg)==0:
                    currseg.append(coord(p1))
                currseg.append(coord(p2))
                res.append(numpy.array(currseg))
                stabs.append(currstab)
                currseg=[]
                currstab=not currstab
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
                currstab=s2<0
        if len(currseg)>0:
            res.append(numpy.array(currseg))
            stabs.append(currstab)
        return res,stabs


from ...typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
