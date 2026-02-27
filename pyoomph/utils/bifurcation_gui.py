#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2025  Christian Diddens & Duarte Rocha
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
#  The authors may be contacted at c.diddens@utwente.nl and d.rocha@utwente.nl
#
# ========================================================================
 

from ..generic.codegen import EquationTree
from ..generic import Problem
from ..typings import *

import matplotlib
import os
if os.environ.get("PYOOMPH_MPLBACKEND") is not None:
    matplotlib.use(os.environ.get("PYOOMPH_MPLBACKEND"))
else:
    os.environ["PYOOMPH_MPLBACKEND"]=matplotlib.get_backend()

import matplotlib.pyplot as plt

from pathlib import Path
from collections import UserList
import numpy
from scipy.interpolate import UnivariateSpline
import os 
import json
import glob
import shutil

class BifurcationGUISolutionPoint:
    def __init__(self,param_value,obs_values,eig_value,statefile,outstep) -> None:
        self.param_value=param_value
        self.obs_values=obs_values
        self.eig_value_Re=numpy.real(eig_value)
        self.eig_value_Im=numpy.imag(eig_value)
        self.statefile=statefile
        self.outstep=outstep
        self.scoord=0
        self._tangs={}
        self._branch_switch_tangs=[]
        self.tag=-1
        self.bifurcation_info:Optional[Dict]=None

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
    


class BifurcationGUISolutionBranch(UserList[BifurcationGUISolutionPoint]):

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
        if obs is not None:
            y=[p.obs_values[obs] for p in self]
        else:
            y=numpy.array([[p.obs_values[k]  for p in self] for k in self[0].obs_values.keys()]).transpose()
            
        eigRe=[p.eig_value_Re for p in self]
        eigIm=[p.eig_value_Im for p in self]
        xi=UnivariateSpline(s,x,s=0,k=min(3,len(s)-1))
        if obs is not None:
            yi=UnivariateSpline(s,y,s=0,k=min(3,len(s)-1))
        else:
            yi=[UnivariateSpline(s,y[:,i],s=0,k=min(3,len(s)-1)) for i in range(y.shape[1])]
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
                    for xs,ys,eR,eI,ss in zip(xi(ssamp),yi(ssamp),eigRei(ssamp),eigImi(ssamp),ssamp):
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

class BifurcationGUI:
    def __init__(self,problem:Problem,parameter:Optional[str]=None) -> None:
        self.problem=problem
        self.problem._runmode="overwrite"
        self.problem.write_states=True
        self.problem.continuation_data_in_states=True
        self.data_subdir="_bifurcation_gui_data"
        self.neigen=10
        self.shift=0
        self.branches:List[BifurcationGUISolutionBranch]=[]
        self.current_branch=None
        self.current_point=None
        self.selected_point=None
        self.selected_branch=None
        self._last_ds=1
        self._fig=None
        self._tangs={}
        self._paramname=parameter
        self.parameter_range=[]
        self._out_demo_video=False
        self._demo_video_step=0
        self._current_observable=None
        self._avail_observables=[]
        self._observable_funcs=None
        self._mode="al"
        self._move_point=False
        self.interpolated_splines=False
        self._state_step=0
        self._waitstring=None
        self._modifier_keys={"shift":False}
        self._escape_pressed=False
        #: Write all observable values to the output files
        self.output_all_observables=False
        self.custom_key_functions:Dict[str,Callable[[BifurcationGUI],None]]={}
        self._initial_view=None
        self.classify_bifurcations=False
        
        
    def set_initial_view(self,xmin,xmax,ymin,ymax):
        self._initial_view=[xmin,xmax,ymin,ymax]

    def get_bifurcation_parameter(self):
        if self._paramname is None:
            raise RuntimeError("This function must be customized or parameter must be passed")
        elif isinstance(self._paramname,str):
            if self._paramname not in self.problem.get_global_parameter_names():
                raise RuntimeError("Parameter "+self._paramname+" not part of the problem")
            return self.problem.get_global_parameter(self._paramname)
    
    # By default, we allow to access all integral observables (not beginning with _) and all ODE dofs
    def evalulate_observables(self)->Dict[str,float]:
        if self._observable_funcs is None:
            obs={}
            def recursive_add_spatial_domains(eqtree:EquationTree):
                if eqtree._equations is not None and eqtree.get_equations()._is_ode()==False:
                    ifuncs=eqtree.get_mesh().list_integral_functions()                    
                    deps = eqtree._codegen._dependent_integral_funcs
                    ifuncs: Set[str] = set(ifuncs)                    
                    ifuncs.update(deps.keys())
                    bn=eqtree.get_full_path().lstrip("/")
                    for valn in ifuncs:
                        if not valn.startswith("_"):
                            obs[bn+"/"+valn]=lambda domname=bn,valn=valn: self.problem.get_mesh(domname).evaluate_observable(valn).float_value()
                for child in eqtree.get_children().values():
                    recursive_add_spatial_domains(child)

            for name,eqtree in self.problem._equation_system.get_children().items():
                if eqtree.get_equations()._is_ode()==True:
                    ode=self.problem.get_ode(name)
                    _vals, inds = ode._get_ODE("ODE").to_numpy()
                    for valn in inds.keys():
                        if not valn.startswith("_"):
                            obs[name+"/"+valn]=lambda domname=name,valn=valn: self.problem.get_ode(domname).get_value(valn,dimensional=True,as_float=True)
            recursive_add_spatial_domains(self.problem._equation_system)
            self._observable_funcs=obs.copy()
            if len(self._observable_funcs)==0:
                raise RuntimeError("Could not identify an observable. Add ODEs or IntegralObservables to find them")
        return {n:func() for n,func in self._observable_funcs.items()}        
    
    def new_branch_from_state(self,statefile):
        self.current_branch=BifurcationGUISolutionBranch()        
        self.selected_branch=self.current_branch
        self.branches.append(self.current_branch)
        self.problem.load_state(statefile,ignore_continuation_data=True,ignore_eigendata=True)
        self.problem.reset_arc_length_parameters()
        self.problem.solve_eigenproblem(self.neigen,self.shift)
        self._add_current_state()
        self._update_tangents()
        self.update_plot()
    
    def _add_current_state(self):
        if self.current_branch is None:
            self.current_branch=BifurcationGUISolutionBranch()        
            self.selected_branch=self.current_branch
            self.branches.append(self.current_branch)
        state_file=self.problem.get_output_directory(os.path.join(self.data_subdir,"_states","state_{:06d}.dump".format(self._state_step))) 
        p=BifurcationGUISolutionPoint(self.get_bifurcation_parameter().value,self.evalulate_observables(),self.problem.get_last_eigenvalues()[0],state_file,self._state_step)                                 
        if p.eig_value_Re==0 and self.classify_bifurcations:
            from pyoomph.generic.bifurcation_tools import NormalFormCalculator
            p.bifurcation_info=NormalFormCalculator(self.problem).get_normal_form(self.get_bifurcation_parameter().get_name())            
        self.problem.save_state(state_file)
        self._state_step+=1
        #if self.current_point is not None:            
        #    p.scoord=self.current_point.scoord+self.problem.get_arc_length_parameter_derivative()*self._last_ds
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

    def output_curves(self):
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
                numpy.savetxt(os.path.join(bdir,fn),seg[:,:-1],header=self._paramname+"\t"+self._current_observable+"\tReEigen\tImEigen")
            nbif=0
            for p in b:
                if p.eig_value_Re==0:
                    fn="bifurcation_{:03d}.txt".format(nbif)
                    pc=p.get_coordinate(self._current_observable,with_s=False,with_eigen=True)
                    numpy.savetxt(os.path.join(bdir,fn),numpy.array([pc],ndmin=2),header=self._paramname+"\t"+self._current_observable+"\tReEigen\tImEigen")
                    nbif+=1
                if p.tag>=0 and p.statefile is not None:                    
                    shutil.copy2(p.statefile, os.path.join(odir,"tag{:02d}.dump".format(p.tag)))
                    fn="tag_{:02d}.txt".format(p.tag)
                    pc=p.get_coordinate(self._current_observable,with_s=False)                    
                    numpy.savetxt(os.path.join(odir,fn),numpy.array([pc],ndmin=2),header=self._paramname+"\t"+self._current_observable)


    def toggle_point_tag(self,pt,tag):
        if pt.tag==tag:
            pt.tag=-1
            return
        for b in self.branches:
            for p in b:
                if p.tag==tag:
                    p.tag=-1
        pt.tag=tag

    def on_key_release(self,event):
#        print("KEY RELEASE",event.key)
        if event.key in {"shift"}:
#            print("RESELSE SHIFT")
            self._modifier_keys[event.key]=False
        elif event.key=="shift+shift":
            self._modifier_keys['shift']=False
            
    def on_key(self,event):
        if event.key == "escape":
            self._escape_pressed=True
            print("ESCAPE PRESSED!")
            return
        else:
            self._escape_pressed=False

        if event.key in self.custom_key_functions:
            self.custom_key_functions[event.key](self)
        elif event.key in ["0","1","2","3","4","5","6","7","8","9"]:
            self.toggle_point_tag(self.selected_point if self.selected_point is not None else self.current_point,int(event.key))
            self.save_all()
            self.update_plot()
        elif event.key=="o":
            self.output_curves()
        elif event.key=="a":
            self.problem.set_arc_length_parameter(scale_arc_length=not self._modifier_keys["shift"])
            print("Scale arclength is set to ",not self._modifier_keys["shift"], "use a or shift+a to switch")
        elif event.key=="A":
            self.problem.set_arc_length_parameter(scale_arc_length=False)
            print("Scale arclength is set to ",False, "use a or shift+a to switch")
        elif event.key=="i":
            self.interpolated_splines=not self.interpolated_splines
            self.update_plot()
        elif event.key in {"shift"}:
            self._modifier_keys[event.key]=True
        elif event.key==' ':
            if self._mode=="al":
                if self._modifier_keys["shift"]:
                    self.multistep()
                else:
                    self.step()
                self.save_all()
                self.update_plot()
            else:
                self._move_point=not self._move_point
                self.update_plot()

        elif event.key=="t":
            self.transient_leave_branch(0)
            self.save_all()
            self.update_plot()            
        elif event.key=="T":
            self.transient_leave_branch(1)
            self.save_all()
            self.update_plot()                        
        elif event.key=="*":  
            ds_backup=self._last_ds          
            self.step()
            self._last_ds=(-1 if self._last_ds<0 else 1)*min(abs(self._last_ds),abs(ds_backup))
            self.save_all()
            self.update_plot()
        elif event.key=="backspace" or event.key=="delete":
            if self.selected_point is None:
                self.selected_branch=self.current_branch
                self.selected_point=self.current_point
            if self.selected_point is None:
                self.selected_point=self.current_point
                self.selected_branch=self.current_branch
            if self.selected_point not in self.selected_branch:
                self.selected_branch=self.current_branch
                self.selected_point=self.current_point
            torem=self.selected_point
            index=self.selected_branch.index(torem)
            if index==0:
                if len(self.selected_branch)==1:
                    if len(self.branches)==1:                        
                        raise RuntimeError("Cannot delete the last point")
                    if torem.statefile:
                        os.remove(torem.statefile)
                    
                    self.branches.remove(self.selected_branch)                    
                    if self.selected_branch==self.current_branch:
                        self.current_branch=self.branches[-1]
                        self.current_point=self.current_branch[-1]
                        self.load_pt(self.current_branch[-1])
                        self.selected_point=self.current_point
                        self.selected_branch=self.current_branch
                    else:
                        self.selected_point=self.current_point
                        self.selected_branch=self.current_branch
                    self.save_all()
                    self.update_plot()
                    return
                else:
                    if torem==self.current_point:
                        self.load_pt(self.current_branch[1])                
                        self.current_point=self.current_branch[1]
                    else:
                        self.selected_point=self.selected_branch[1]
                        self.selected_branch=self.selected_branch
            else:
                if torem==self.current_point:
                    self.load_pt(self.current_branch[index-1])                
                    self.current_point=self.current_branch[index-1]
                else:
                    if index>0:
                        self.selected_point=self.selected_branch[index-1]                        
                    elif index+1<len(self.current_branch):
                        self.selected_point=self.current_branch[index+1]                        
                    else:
                        self.selected_point=None                        
            if torem.statefile:
                os.remove(torem.statefile)
            self.selected_branch.remove(torem)
            self.save_all()
            self.update_plot()
        elif event.key=="+":
            self._last_ds*=1.25
            self.update_plot()
        elif event.key=='-':
            self._last_ds/=1.25
            self.update_plot()
        elif event.key=='/':
            self._last_ds*=-1
            self.update_plot()
        elif event.key in {'pagedown',"pageup","home","end"}:
            if self.selected_point is None:
                self.selected_point=self.current_point
                self.selected_branch=self.current_branch
            if self.selected_branch is None:
                self.selected_branch=self.current_branch
            if self.selected_point not in self.selected_branch:
                self.selected_point=self.current_point
                self.selected_branch=self.current_branch            
            index=self.selected_branch.index(self.selected_point)
            origindex=index            
            if event.key=='pagedown' and index>0:
                index-=1
            elif event.key=="pageup" and index+1<len(self.selected_branch):
                index+=1
            elif event.key=="end" and self.selected_point is not self.selected_branch[-1]:
                index=len(self.selected_branch)-1
            elif event.key=="home" and self.selected_point is not self.selected_branch[0]:
                index=0 
            else: 
                return

            if self._mode=="mp" and self._move_point:
                backup=self.selected_branch[index]
                self.selected_branch[index]=self.selected_point
                self.selected_branch[origindex]=backup
                self.reorder_branch_upon_point_insertion(self.selected_branch,None)
                self.update_plot()
            else:
                self.selected_point=self.selected_branch[index]
            self.update_plot()
        elif event.key=="enter":
            if self.selected_point is not None:
                if self.selected_point is not self.current_point:
                    if self.selected_branch is not None:
                        self.current_branch=self.selected_branch         
                    self.load_pt(self.selected_point)                    
                    self.save_all()
                    self.update_plot()
        elif event.key=="y":
            currindex=self._avail_observables.index(self._current_observable)
            newindex=(currindex+1)%len(self._avail_observables)
            if currindex!=newindex:
                self._current_observable=self._avail_observables[newindex]
                ymin=1e30
                ymax=-1e30
                for b in self.branches:
                    for p in b:
                        y=p.get_coordinate(self._current_observable)[1]
                        ymin=min(ymin,y)
                        ymax=max(ymax,y)
                self._fig.gca().set_ylim(ymin,ymax)
                self.save_all()
                self.update_plot()
        elif event.key=="m":
            modes=["al","mp"]
            self._mode=modes[(modes.index(self._mode)+1)%len(modes)]
            self.save_all()
            self.update_plot()

        elif event.key=="b":
            if self.current_point is not None and self.current_point.eig_value_Re==0:
                self.branch_switch()
                self.save_all()
                self.update_plot()
            else:                    
                try:
                    self.locate_bifurcation()
                    self.save_all()
                except:
                    self.problem.deactivate_bifurcation_tracking()
                self.update_plot()
            
        elif event.key=="p":
            self.locate_bifurcation(pitchfork=True)
            self.save_all()
            self.update_plot()            

        print('you pressed', event.key, event.xdata, event.ydata)
        
    def on_press(self,event):
        if event.xdata is None:
            return 
        xl=self._fig.gca().get_xlim()
        yl=self._fig.gca().get_ylim()
        dx=xl[1]-xl[0]
        dy=yl[1]-yl[0]
        bestbranch,bestpoint=None,None
        bestdist=1e30
        for b in self.branches:
            for p in b:
                c=p.get_coordinate(self._current_observable)
                dist=((c[0]-event.xdata)/dx)**2+((c[1]-event.ydata)/dy)**2
                if dist<bestdist:
                    bestbranch=b
                    bestpoint=p
                    bestdist=dist
        if bestpoint and bestdist<1e-2:            
            self.selected_branch=bestbranch
            self.selected_point=bestpoint
            self.update_plot()
        
        print('you pressed', event.button, event.xdata, event.ydata)


    def _open_plot(self):
        self._fig = plt.figure()     
        cid_key_press = self._fig.canvas.mpl_connect('key_press_event', lambda event : self.on_key(event))
        cid_key_release = self._fig.canvas.mpl_connect('key_release_event', lambda event : self.on_key_release(event))
        cid_butt = self._fig.canvas.mpl_connect('button_press_event', lambda event : self.on_press(event))


    def update_plot(self,infotext:Optional[str]=None):     
        #global matplotlib
        #matplotlib = reload(matplotlib)
        #matplotlib.use('GTK3Agg', force=True)
        #import matplotlib.pyplot as plt      
        if self._fig is None:
            self._open_plot()
            cp=self.current_point.get_coordinate(self._current_observable)
            if len(self.branches)==1 and len(self.current_branch)==1:
                if self._initial_view is not None:
                    self._fig.gca().set_xlim(self._initial_view[0],self._initial_view[1])
                    self._fig.gca().set_ylim(self._initial_view[2],self._initial_view[3])
                else:
                    self._fig.gca().set_xlim(cp[0]-1e-4,cp[0]+1e-4)
                    self._fig.gca().set_ylim(cp[1]-1e-4,cp[1]+1e-4)
            self.update_plot()            
            if plt._get_backend_mod().FigureCanvas.required_interactive_framework is None:
                raise RuntimeError("You likely have imported pyoomph.output.plotting before pyoomph.utils.bifurcation_gui.\nThis could also have happened if you import a file that imports pyoomph.output.plotting.\nMake sure to first import pyoomph.utils.bifurcation_gui!")
#            print(plt._get_backend_mod())
#            print(dir(plt._get_backend_mod()))
#            print(plt._get_backend_mod().FigureCanvas.required_interactive_framework)
            plt.show()
        
        gca=self._fig.gca()
        xlim=list(gca.get_xlim())
        ylim=list(gca.get_ylim())
        xscal=gca.get_xscale()
        yscal=gca.get_yscale()

        def extend_lims(p):
            xlim[0]=min(xlim[0],p[0])
            xlim[1]=max(xlim[1],p[0])
            ylim[0]=min(ylim[0],p[1])
            ylim[1]=max(ylim[1],p[1])
        #self._fig.clear(keep_observers=True)
        #print(gca.lines)
        while len(gca.lines)>0:
            gca.lines[0].remove()
        while len(gca.artists)>0:
            gca.artists[0].remove()
        annotations = [child for child in gca.get_children() if isinstance(child, matplotlib.text.Annotation)]
        for a in annotations:
            a.remove()
        while len(gca.texts)>0:
            try:
                gca.texts[0].remove()
            except:
                del gca.texts[-1]

        for b in self.branches:
            color="red" if b == self.current_branch else "grey"
            
            if self.interpolated_splines:
                segs,stabs=b.smooth_branch_stab_list(self._current_observable)
            else:
                segs,stabs=b.to_branch_stab_list(self._current_observable)

            for seg,stab in zip(segs,stabs):
                if stab == True:
                    dt="-"
                    lw=1.5
                elif stab == False:
                    dt="dashed"
                    lw=0.75
                else:
                    dt="dotted"               
                    lw=1.0
                #print("IN",stab,len(seg),dt,seg)                 
                gca.plot(seg[:,0],seg[:,1], linestyle=dt,color=color,linewidth=lw)
            normpts=numpy.array([p.get_coordinate(self._current_observable) for p in b if p.eig_value_Re!=0],ndmin=2)
            #print("BEF",normpts)
            #if normpts.shape[0]>1:
            #    normpts=normpts.transpose()
            #print("AFT",normpts)
            if len(normpts)>0:
                gca.plot(normpts[:,0],normpts[:,1], 'o', markersize=3,color=color)
            for p in b:                
                if p.eig_value_Re==0:                                        
                    pc=p.get_coordinate(self._current_observable)
                    gca.plot([pc[0]],[pc[1]], marker='o', markersize=6,color="brown")                    
                    if p.bifurcation_info is not None:
                        shorts={"transcritical":"T","fold":"F","pitchfork":"P","Hopf":"H"}
                        if p.bifurcation_info["type"] in shorts:                            
                            gca.annotate(str(p.tag)+"," if p.tag>=0 else ""+ shorts[p.bifurcation_info["type"]],pc)
                        elif p.tag>=0:
                            gca.annotate(str(p.tag),pc)
                elif p.tag>=0:
                    pc=p.get_coordinate(self._current_observable)
                    gca.annotate(str(p.tag),pc)

            
#            self._fig.gca().plot(curve[:,0],curve[:,1], marker='o', markersize=3)
        
        if self.current_point is not None:
            extend_lims(self.current_point.get_coordinate(self._current_observable))
            pc=self.current_point.get_coordinate(self._current_observable,with_eigen=True)
            if self._mode=="al":
                self._fig.gca().plot([pc[0]],[pc[1]], marker='o', markersize=5,color="green" )
                if self._current_observable in self._tangs and self._last_ds is not None:
                    x0=numpy.array([pc[0],pc[1]])
                    dx=self._last_ds*self._tangs[self._current_observable]
                    #self._fig.gca().arrow(x0[0],x0[1],dx[0],dx[1])
                    extend_lims(x0)
                    self._fig.gca().annotate("", xy=x0+dx, xytext=x0,arrowprops=dict(arrowstyle="->"),annotation_clip=False)
                    for i,bst in enumerate(self.current_point._branch_switch_tangs):
                        dx=self._last_ds*bst[self._current_observable]
                        #self._fig.gca().arrow(x0[0],x0[1],dx[0],dx[1])
                        extend_lims(x0)
                        self._fig.gca().annotate("", xy=x0+dx, xytext=x0,arrowprops=dict(arrowstyle="->",color="brown",linewidth=1 if i==0 else 0.1),annotation_clip=False)
                eigv=pc[2]+1j*pc[3]
                pttext="({:3.3g},{:3.3g})\n".format(pc[0],pc[1])+f'{eigv:.2g}'                
            else:
                pttext=None
        else:
            pttext=None

        if self.selected_point is not None:
            pc=self.selected_point.get_coordinate(self._current_observable,with_eigen=True)
            if self._mode=="mp":
               if self._move_point:
                    gca.plot([pc[0]],[pc[1]], marker='o', markersize=5,color="blue")
               else:
                    gca.plot([pc[0]],[pc[1]], marker='x', markersize=12,color="grey")
            else:
                gca.plot([pc[0]],[pc[1]], marker='x', markersize=12,color="grey")
            eigv=pc[2]+1j*pc[3]
            seltext="({:3.3g},{:3.3g})\n".format(pc[0],pc[1])+f'{eigv:.2g}'
        else:
            seltext=None

        if self.parameter_range is not None and len(self.parameter_range)==2:
            gca.set_xlim(self.parameter_range)


        gca.set_xlabel(self.get_bifurcation_parameter().get_name())
        gca.set_ylabel(self._current_observable)
        gca.set_xlim(xlim)
        gca.set_ylim(ylim)
        gca.set_xscale(xscal)
        gca.set_yscale(yscal)
        if infotext is not None:
            gca.text(0.5, 0.5, infotext,horizontalalignment='center',verticalalignment='center',transform=gca.transAxes,bbox = dict(boxstyle="round", fc="lightgrey", ec="0.5", alpha=0.9))
        
        if pttext is not None:
            gca.text(0.1, 1.01, pttext,horizontalalignment='center',verticalalignment='bottom',transform=gca.transAxes,color="red")
        if seltext is not None:
            gca.text(0.9, 1.01, seltext,horizontalalignment='center',verticalalignment='bottom',transform=gca.transAxes,color="grey")
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        
        if self._out_demo_video:
            ddir=self.problem.get_output_directory(self.data_subdir)
            odir=os.path.join(ddir,"demo_movie")        
            Path(odir).mkdir(parents=True,exist_ok=True)
            self._fig.savefig(os.path.join(odir,"plot_{:06d}.png".format(self._demo_video_step)))
            self._demo_video_step+=1
        
    def _update_tangents(self):
        FD_eps=1e-6
        #if self.current_point._tangs is not None and len(self.current_point._tangs)>0:
        #    self._tangs=self.current_point._tangs.copy()
        backup,_=self.problem.get_current_dofs()
        dp=self.problem.get_arc_length_parameter_derivative()
        ddof=numpy.array(self.problem.get_arclength_dof_derivative_vector())
        if len(ddof)>0:
            self.problem.set_current_dofs(backup+FD_eps*ddof)
            po=self.evalulate_observables()
        else:
            po=self.current_point.obs_values.copy()

        for k in self._avail_observables:
            do=(po[k]-self.current_point.obs_values[k])/FD_eps
            self._tangs[k]=numpy.array([dp,do])    
        self.current_point._tangs=self._tangs.copy()
        
        
        if self.current_point.bifurcation_info is not None:
            bi=self.current_point.bifurcation_info
            self.current_point._branch_switch_tangs=[]
            if bi["type"]=="transcritical":
                for dptr in [-dp,dp]:
                    ddof=numpy.array(bi["perturbation_predictor"](dptr))                    
                    self.problem.set_current_dofs(backup+FD_eps*ddof)
                    po=self.evalulate_observables()
                    btangtangs={}
                    for k in self._avail_observables:
                        do=(po[k]-self.current_point.obs_values[k])/FD_eps
                        btangtangs[k]=numpy.array([dptr,do])    
                    self.current_point._branch_switch_tangs.append(btangtangs)
                    
        self.problem.set_current_dofs(backup)

    def branch_switch(self):
        if self.current_point.eig_value_Re!=0:
            raise RuntimeError("Can only switch branches as bifurcations")
        if self.current_point.bifurcation_info is None:
            raise RuntimeError("No bifurcation info available. Please set gui.classify_bifurcations=True")
        bi=self.current_point.bifurcation_info
        if bi["type"]=="fold":
            print("Cannot switch branches at fold bifurcations")
            return
        
        param=self.get_bifurcation_parameter()
        curr=self.problem.get_current_dofs()[0]       
        
        if self.current_point._branch_switch_tangs is None or len(self.current_point._branch_switch_tangs)==0:
            ds=0.001 
            dp=bi["param_predictor"](ds)
            du=bi["perturbation_predictor"](ds)      
        else:
            dp=self.current_point._branch_switch_tangs[0][self._current_observable][0]*self._last_ds
            du=bi["perturbation_predictor"](dp)                
            ds=self._last_ds
        
        print("Branch switching with dp=",dp,"dunorm",numpy.linalg.norm(du))
        self.update_plot("BRANCH SWITCHING")
        
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
        self.update_plot("LEAVING BRANCH TRANSIENTLY")
        eig=numpy.sqrt(self.current_point.eig_value_Re**2+self.current_point.eig_value_Im**2)
        eig=max(1e-4,eig)
        tsnd=1/eig
        ts=self.problem.get_scaling("temporal")*tsnd

        self.problem.reset_arc_length_parameters()
        self.problem.set_current_time(0)
  
        #self.problem.solve()
        #self.problem.solve_eigenproblem(self.neigen)
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
        print("Integrated",1000*ts)
        

    def reorder_branch_upon_point_insertion(self,branch:BifurcationGUISolutionBranch,newp:BifurcationGUISolutionPoint):
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
        
        #pscale,obsscale=1,1
        xlim=self._fig.gca().get_xlim()
        ylim=self._fig.gca().get_ylim()
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
        for p in self.current_branch:
            if p==newp:
                continue
            p.scoord/=al
            sbase.append(p.scoord)

        if newp is not None:
            xn,yn=newp.get_coordinate(self._current_observable)
            xn,yn=float(xn*pscale),float(yn*obsscale)
            # Quite demanding, but lets give it a try: Could be improved of course
            shortest_l=1e50
            shortest_news=0
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
    #            al=0
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

    def multistep(self):
        xlim=self._fig.gca().get_xlim()
        ylim=self._fig.gca().get_ylim()
        xscale=self._fig.gca().get_xscale()
        yscale=self._fig.gca().get_yscale()
        tvec=self._tangs[self._current_observable]*self._last_ds
        max_ds=numpy.sqrt(tvec[0]*tvec[0]+tvec[1]*tvec[1])
        cp0=self.current_point.get_coordinate(self._current_observable)
        while True:
            cp=self.current_point.get_coordinate(self._current_observable)
            if cp[0]<xlim[0] or cp[0]>xlim[1] or cp[1]<ylim[0] or cp[1]>ylim[1]:                
                break
            if self._escape_pressed:
                self._escape_pressed=False
                self.update_plot()
                break
                #raise RuntimeError("LEAVING")
            self.step()
            self.save_all()
            tvec=self._tangs[self._current_observable]*self._last_ds
            if xscale=="log":
                xlogfactor=(cp0[0]/cp[0])**2
            else:
                xlogfactor=1
            ds=numpy.sqrt(xlogfactor*tvec[0]*tvec[0]+tvec[1]*tvec[1])
            self._last_ds*=min(1,max_ds/ds)
        self.update_plot()

    def step(self,ds=None):
        if ds is None:
            ds=self._last_ds
        origin=self.current_point
        if self._escape_pressed:
            return
        self.update_plot("ARCLENGTH STEPPING")
        ds=self.problem.arclength_continuation(self.get_bifurcation_parameter(),ds)
        self.problem.solve_eigenproblem(self.neigen,self.shift)
       
        self._add_current_state()

        self.reorder_branch_upon_point_insertion(self.current_branch,self.current_point)
        self._last_ds=ds        
        self._update_tangents()  
        if origin._tangs is None or len(origin._tangs)==0:
            origin._tangs=self.current_point._tangs.copy()

        return ds


    def locate_bifurcation(self,pitchfork:bool=False):
        self.update_plot("BIFURCATION FINDING"+(" (PITCHFORK)" if pitchfork else ""))
        self.problem.solve_eigenproblem(self.neigen,self.shift)
        self.problem.activate_bifurcation_tracking(self._paramname,"pitchfork" if pitchfork else None)
        self.problem.solve(max_newton_iterations=20)
        self._add_current_state()
        self._update_tangents()
        self.problem.deactivate_bifurcation_tracking()
        self.reorder_branch_upon_point_insertion(self.current_branch,self.current_point)

    
    def start(self,init_ds,initial_max_newton_iterations=10):
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
        
        # TODO: If no data there, restart. Otherwise, load
        try:
            self.problem.solve(max_newton_iterations=initial_max_newton_iterations)
        except:
            raise RuntimeError("Make sure the problem starts where it has a stationary solution")
        self.problem.solve_eigenproblem(self.neigen,self.shift)
        self._avail_observables=[k for k in self.evalulate_observables().keys()]
        self._current_observable=self._avail_observables[0]
        self._add_current_state()

        outdir=self.problem.get_output_directory(self.data_subdir)
        fname=os.path.join(outdir,"state.json")
        try:
            open(fname).close()
            self.load_all()
            plt.show()
        except Exception as e:
            print(e)

        self.update_plot()
        
    
    def save_all(self):
        outdir=self.problem.get_output_directory(self.data_subdir)
        
        fullinfo={}
        fullinfo["branches"]=[b.to_state_dict() for b in self.branches]
        fullinfo["demo_video_step"]=self._demo_video_step
        fullinfo["xlim"]=self._fig.gca().get_xlim()
        fullinfo["ylim"]=self._fig.gca().get_ylim()
        fullinfo["xscale"]=self._fig.gca().get_xscale()
        fullinfo["yscale"]=self._fig.gca().get_yscale()

        fullinfo["statestep"]=self._state_step
        fullinfo["currentbranch"]=self.branches.index(self.current_branch)
        fullinfo["currentpoint"]=self.current_branch.index(self.current_point)
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

    def load_all(self):
        outdir=self.problem.get_output_directory(self.data_subdir)
        fname=os.path.join(outdir,"state.json")
        fullinfo=json.load(open(fname))

        self.branches=[BifurcationGUISolutionBranch.from_dict(b) for b in fullinfo["branches"]]
        if self._fig is None:
            self._open_plot()
        self._fig.gca().set_xlim(fullinfo["xlim"])
        self._fig.gca().set_ylim(fullinfo["ylim"])
        self._fig.gca().set_xscale(fullinfo["xscale"])
        self._fig.gca().set_yscale(fullinfo["yscale"])
        
        if "demo_video_step" in fullinfo:
            self._demo_video_step=fullinfo["demo_video_step"]

        self._state_step=fullinfo["statestep"]
        self.current_branch=self.branches[fullinfo["currentbranch"]]
        self.current_point=self.current_branch[fullinfo["currentpoint"]]
        if "selectedbranch" in fullinfo:
            self.selected_branch=self.branches[fullinfo["selectedbranch"]]
            if "selectedpoint" in fullinfo:
                self.selected_point=self.selected_branch[fullinfo["selectedpoint"]]
        self._last_ds=fullinfo["lastds"]
        self._current_observable=fullinfo["current_observable"]
        self._mode=fullinfo["mode"]
        self.interpolated_splines=fullinfo.get("interpolated_splines",self.interpolated_splines)
        self.update_plot("LOADING")
        self.load_pt(self.current_point)
        self._update_tangents()
        self.update_plot()



    