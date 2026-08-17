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


class BifurcationGUISolutionPoint:
    """One computed solution: every global parameter, all observables, the leading eigenvalue and
    the state file it was dumped to."""

    def __init__(self,param_value,obs_values,eig_value,statefile,outstep,param_values:dict[str,float] | None=None) -> None:
        self.param_value=param_value
        #: Every global parameter at this point, not just the continued one. The state dump restores
        #: all of them, so without this the diagram could not say which slice of parameter space it
        #: is a section through - and reloading a point silently moves the others.
        self.param_values:dict[str,float]=dict(param_values) if param_values else {}
        self.obs_values=obs_values
        self.eig_value_Re=numpy.real(eig_value)
        self.eig_value_Im=numpy.imag(eig_value)
        self.statefile=statefile
        self.outstep=outstep
        self.scoord:float=0
        self._tangs:dict[str,NPFloatArray]={}
        self._branch_switch_tangs:list[Any]=[]
        self.tag=-1
        self.bifurcation_info:dict | None=None

    @staticmethod
    def from_dict(res):
        inst=BifurcationGUISolutionPoint(res["param_value"],res["obs_value"],res["eig_value_Re"]+1j*res["eig_value_Im"],res["statefile"],res["outstep"])
        inst.scoord=res["scoord"]
        inst.tag=res.get("tag",-1)
        # Absent in files written before the slice was recorded. Left empty rather than filled with
        # the continued parameter alone, so "unknown slice" stays distinguishable from "no other
        # parameters exist".
        inst.param_values=dict(res.get("param_values",{}))
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
        res["tangs"]={}
        for k,v in self._tangs.items():
            res["tangs"][k]=list(v)
        return res

    def fixed_parameters(self,varying:Iterable[str])->dict[str,float]:
        """The parameters this point holds fixed, i.e. all of them except the ones being varied."""
        varying=set(varying)
        return {n:v for n,v in self.param_values.items() if n not in varying}


    def get_coordinate(self,obs,with_s=False,with_eigen=False):
        if with_eigen:
            if with_s:
                return [self.param_value,self.obs_values[obs],self.eig_value_Re,self.eig_value_Im, self.scoord]
            else:
                return [self.param_value,self.obs_values[obs],self.eig_value_Re,self.eig_value_Im]
        else:
            if with_s:
                return [self.param_value,self.obs_values[obs],self.scoord]
            else:
                return [self.param_value,self.obs_values[obs]]

    def describe(self,obs:str | None=None)->str:
        """One-line human-readable summary, used by the point-info panel and the branch tree."""
        res="{:.6g}".format(self.param_value)
        if obs is not None and obs in self.obs_values:
            res+=" | {:.6g}".format(self.obs_values[obs])
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
        return any(p.param_values for p in self)

    def describe_slice(self)->str:
        """Human-readable "b = 0.3, c = 2" for the status bar, the plot and file headers."""
        if not self.slice_is_known():
            return "slice unknown"
        fixed=self.fixed_parameters()
        if not fixed:
            return "no other parameters"
        return ", ".join("{:s} = {:.6g}".format(n,v) for n,v in sorted(fixed.items()))

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
        return inst


    def to_point_list(self,obs):
        res=[]
        for p in self:
            res.append(p.get_coordinate(obs))
        return numpy.array(res)

    def smooth_branch_stab_list(self,obs,subsampling=10):
        if len(self)<=1:
            return self.to_branch_stab_list(obs)
        s=[p.scoord for p in self]
        x=[p.param_value for p in self]
        # Note: obs is checked once here to build y/yi together, since the two are only ever used
        # consistently with each other (the "obs is None" branch producing an all-observables array is
        # currently unsupported further down/in to_branch_stab_list, see comment near the sampling loop)
        if obs is not None:
            y=[p.obs_values[obs] for p in self]
            yi:UnivariateSpline | list[UnivariateSpline]=UnivariateSpline(s,y,s=0,k=min(3,len(s)-1))
        else:
            y=numpy.array([[p.obs_values[k]  for p in self] for k in self[0].obs_values.keys()]).transpose()
            yi=[UnivariateSpline(s,y[:,i],s=0,k=min(3,len(s)-1)) for i in range(y.shape[1])]
        eigRe=[p.eig_value_Re for p in self]
        eigIm=[p.eig_value_Im for p in self]
        xi=UnivariateSpline(s,x,s=0,k=min(3,len(s)-1))
        eigRei=UnivariateSpline(s,eigRe,s=0,k=min(3,len(s)-1))
        eigImi=UnivariateSpline(s,eigIm,s=0,k=min(3,len(s)-1))
        segs,stabs=self.to_branch_stab_list(obs)
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
                    # Note: when obs is None (all-observables mode), yi is a list of splines and
                    # to_branch_stab_list(None) above already raises (obs_values has no None key),
                    # so this line is only ever reached with obs not None, where yi is callable.
                    for xs,ys,eR,eI,ss in zip(xi(ssamp),yi(ssamp),eigRei(ssamp),eigImi(ssamp),ssamp): #type:ignore
                        sseg.append([xs,ys,eR,eI,ss])
                smoothsegs.append(numpy.array(sseg))
        return smoothsegs,stabs


    def to_branch_stab_list(self,obs):
        res=[]
        stabs=[]
        if len(self)==0:
            return res,[]
        if len(self)==1:
            return numpy.array([[self[0].get_coordinate(obs,with_s=True,with_eigen=True)]]),[None]
        currseg=[]
        currstab=self[0].eig_value_Re<0
        if self[0].eig_value_Re==0:
            currstab=self[1].eig_value_Re<0
        for p1,p2 in zip(self,self[1:]):
            if p2.eig_value_Re==0:
                if len(currseg)==0:
                    currseg.append(numpy.array(p1.get_coordinate(obs,with_s=True,with_eigen=True)))
                currseg.append(numpy.array(p2.get_coordinate(obs,with_s=True,with_eigen=True)))
                res.append(numpy.array(currseg))
                stabs.append(currstab)
                currseg=[]
                currstab=not currstab
            elif p1.eig_value_Re*p2.eig_value_Re>=0: # Same stability
                if len(currseg)==0:
                    currseg.append(numpy.array(p1.get_coordinate(obs,with_s=True,with_eigen=True)))
                currseg.append(numpy.array(p2.get_coordinate(obs,with_s=True,with_eigen=True)))
            else: # Change in stability
                if len(currseg)>0:
                    res.append(numpy.array(currseg))
                    stabs.append(currstab)
                    currseg=[]
                res.append(numpy.array([p1.get_coordinate(obs,with_s=True,with_eigen=True),p2.get_coordinate(obs,with_s=True,with_eigen=True)]))
                stabs.append(None)
                currstab=p2.eig_value_Re<0
        if len(currseg)>0:
            res.append(numpy.array(currseg))
            stabs.append(currstab)
        return res,stabs


from ...typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
