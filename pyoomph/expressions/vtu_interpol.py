#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2026  Christian Diddens & Duarte Rocha
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
 
 
from .cb import CustomMathExpression
from .generic import vector,ExpressionOrNum,var
from ..typings import *
import vtk

# Provides the class VTUInterpolatorByVTK
#
# You can first load a file by 
#   interp=VTUInterpolatorByVTK("your_vtu_file.vtu",spatial_scale=meter)  
# The latter argument, only if you have metric dimensions. Do not set the problem spatial scale here! It is mainly the spatial scale of the VTU, which is by default meters in pyoomph

# And then you can do e.g.
#   eqs+=InitialCondition(velocity_x=interp.get_field("velocity_x",scale=meter/second),velocity_y=interp.get_field("velocity_y",scale=meter/second))        
#   eqs+=InitialCondition(pressure=interp.get_field("pressure",scale=pascal))        

class _VTUInterpolatorBase:    
    def __init__(self,vtufile:str,spatial_scale:ExpressionOrNum=1,offset=vector(0),resize_internally:bool=True):
        self.vtufile=vtufile
        self.spatial_scale_factor=0+spatial_scale
        self.offset=offset
        self.resize_internally=resize_internally
        self._dim=2
        
    def _get_arg_list(self):        
        coord_suffix=["x","y","z"]
        return [(var("coordinate_"+suffix)+self.offset[index])/self.spatial_scale_factor for index,suffix in enumerate(coord_suffix) if index<self._dim]        
    
    def _auto_strip_component(self,fieldname:str):
        if fieldname.endswith("_x"):
            return fieldname[:-2],0
        elif fieldname.endswith("_y"):
            return fieldname[:-2],1
        elif fieldname.endswith("_z"):
            return fieldname[:-2],2
        else:
            return fieldname,0
    
    def get_field(self,fieldname:str,component_index:Union[int,Literal["auto"]]="auto",scale:ExpressionOrNum=1)->CustomMathExpression:
        raise RuntimeError("Please override")
        

class _VTUFieldInterpolatorByVTK(CustomMathExpression):    
        def __init__(self,vtu:"VTUInterpolatorByVTK",fieldname:str,component_index:int):
            super().__init__()
            self.vtu=vtu
            self.fieldname=fieldname
            self.component_index=component_index
            self._cache_index=0
            
        def eval(self, arg_array) -> float:
            x=list(arg_array)
            if self.vtu._use_cache:
                key=tuple(arg_array)
                if key in self.vtu._cache:
                    return self.vtu._cache[key][self._cache_index][self.component_index]
            while len(x)<3:
                x.append(0.0)            
            self.vtu.probe_pt.SetCenter(*x)
            self.vtu.probe_pt.Update()
            self.vtu.interpolator.Update()
            if self.vtu._use_cache:
                cache_res=[]
                for f in self.vtu._fields_to_cache:
                    arr=self.vtu.interpolator.GetOutput().GetPointData().GetArray(f)
                    
                    entry=[]
                    for e in range(arr.GetNumberOfComponents()):
                        res=arr.GetVariantValue(e)
                        if res.IsDouble():
                            res=float(res.ToDouble())
                        elif res.IsFloat():
                            res=float(res.ToFloat())
                        elif res.IsInt():
                            res=float(res.ToInt())
                        else:
                            raise RuntimeError("Not a float")
                        entry.append(res)
                    cache_res.append(entry)
                self.vtu._cache[key]=cache_res
                return cache_res[self.vtu._fields_to_cache.index(self.fieldname)][self.component_index]
                    
            arr=self.vtu.interpolator.GetOutput().GetPointData().GetArray(self.fieldname)
            if arr is None:
                raise RuntimeError("Point outside the interpolation or array "+self.fieldname+" not available")
            res=arr.GetVariantValue(self.component_index)            
            if res.IsDouble():
                res=float(res.ToDouble())
            elif res.IsFloat():
                res=float(res.ToFloat())
            elif res.IsInt():
                res=float(res.ToInt())
            else:
                raise RuntimeError("Not a float")
            return res
        
class VTUInterpolatorByVTK(_VTUInterpolatorBase):        
    
    # spatial_scale: Set it to [X_dest]/[X_src]
    # if e.g. the coordinates in the VTU file are written in micro meters, [X_src] is 1e6
    # By default, MeshFileOutput writes in meters, so [X_src] is usually 1
    # if e.g. the destination problem uses metric units, use [X_dest]=meter, i.e. spatial_scale=meter        
    
    # resize_internally: will internally resize the input file coordinates to unity range. This reduces interpolation errors
    
    # use_cache: If you interpolate for multiple fields, it is sometimes better to interpolate each point only once and store the results of all fields in a cache
    # This can be quite memory intensive, though
    def __init__(self, vtufile: str,spatial_scale:ExpressionOrNum=1,resize_internally:bool=True,use_cache:Union[bool,Literal["auto"]]="auto"):
        super().__init__(vtufile,spatial_scale=0+spatial_scale,resize_internally=resize_internally)
        self.use_cache=use_cache
        self._use_cache=True if self.use_cache is True else False
        self._num_interpolators=0
        self._cache={}
        self._fields_to_cache=[]
        
        
        # Load the VTU
        self.vtu_in=vtk.vtkXMLUnstructuredGridReader()
        self.vtu_in.SetFileName(self.vtufile)        
        self.vtu_in.Update()
        if self.vtu_in.GetOutput().GetNumberOfCells()==0:
            raise RuntimeError("No cells found in "+str(vtufile))                
        
        # Calculate the boundaries for dimension analysis and optional rescaling
        bounds=self.vtu_in.GetOutput().GetPoints().GetBounds()
        xscale=bounds[1]-bounds[0]
        yscale=bounds[3]-bounds[2]
        zscale=bounds[5]-bounds[4]        
        scalef=max(xscale,yscale,zscale)
        if zscale>0:
            self._dim=3            
        elif yscale>0:
            self._dim=2
        elif xscale>0:
            self._dim=1
        else:
            self._dim=0
            scalef=1        
        
        self.vector_scale=1
        # Rescale it to fit in unity range
        if self.resize_internally:
            transform=vtk.vtkTransform()
            transform.Scale(1/scalef,1/scalef,1/scalef)
            self.vtu=vtk.vtkTransformFilter()
            self.vtu.SetTransformAllInputVectors(False)
            self.vtu.SetInputData(self.vtu_in.GetOutput())
            self.vtu.SetTransform(transform)            
            self.vtu.Update()
            self.spatial_scale_factor*=scalef
        else:
            self.vtu=self.vtu_in    
        
        # The point used to probe the current positions
        


        self.probe_pt=vtk.vtkPointSource()
        self.probe_pt.SetNumberOfPoints(1)
        #eps=1e-6*(scalef if not self.resize_internally else 1)
        self.probe_pt.SetRadius(0.0)       
        # And the interpolator
        self.interpolator=vtk.vtkProbeFilter()        
        self.interpolator.SetInputData(self.probe_pt.GetOutput())
        self.interpolator.SetSourceData(self.vtu.GetOutput())         

    def get_field(self, fieldname: str,component_index:Union[int,Literal["auto"]]="auto",scale:ExpressionOrNum=1) -> CustomMathExpression:
        if component_index=="auto":
            fieldname,component_index=self._auto_strip_component(fieldname)
        self._num_interpolators+=1
        if self.use_cache=="auto" and self._num_interpolators>1:
            self._use_cache=True
        if fieldname not in self._fields_to_cache:
            self._fields_to_cache.append(fieldname)            
        interp=_VTUFieldInterpolatorByVTK(self,fieldname,component_index)
        interp._cache_index=self._fields_to_cache.index(fieldname)
        return interp(*self._get_arg_list())*scale
        
        
        