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
    """One computed solution: the parameter value, all observables, the leading eigenvalue and the
    state file it was dumped to."""

    def __init__(self,param_value,obs_values,eig_value,statefile,outstep) -> None:
        self.param_value=param_value
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
        res["tangs"]={}
        for k,v in self._tangs.items():
            res["tangs"][k]=list(v)
        return res


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



class BifurcationGUISolutionBranch(UserList[BifurcationGUISolutionPoint]):
    """A list of solution points forming one branch, ordered by their arclength coordinate."""

    def to_state_dict(self):
        res={}
        res["points"]=[p.to_state_dict() for p in self]
        return res

    @staticmethod
    def from_dict(res):
        inst=BifurcationGUISolutionBranch()
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
