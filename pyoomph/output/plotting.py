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
 
import os,json


import matplotlib
import matplotlib.pyplot
from ..generic.codegen import Equations

from ..meshes.mesh import AnySpatialMesh
from ..expressions.generic import Expression, ExpressionNumOrNone, ExpressionOrNum
from ..expressions.units import unit_to_string
from ..meshes.meshdatacache import MeshDataCacheEntry


if os.environ.get("PYOOMPH_MPLBACKEND") is not None:
    matplotlib.use(os.environ.get("PYOOMPH_MPLBACKEND")) #type:ignore
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.collections as collections
from matplotlib.patches import Ellipse, Polygon
import  matplotlib.colorbar
import matplotlib.axes
import matplotlib.cm as cm
from scipy import interpolate #type:ignore

from ..typings import *
import numpy

import _pyoomph


import matplotlib.image as mpimg

if TYPE_CHECKING:
    from ..generic.problem import Problem
    from ..meshes.meshdatacache import MeshDataEigenModes


_MatPlotLibPartTypeVar=TypeVar("_MatPlotLibPartTypeVar",bound=Type["MatplotLibPart"])
MatPlotLibAddPlotReturns=Union["MatplotLibPart","MatplotLibOverlayBase","MatplotlibVectorFieldArrows"]

class BasePlotter:
    """
    A generic class for plotting of problems. This class is not meant to be used directly, but to be inherited by a specific plotter class.
    """
    def __init__(self,problem:"Problem",eigenvector:Optional[int]=None,eigenmode:"MeshDataEigenModes"="abs"):
        self._problem:"Problem"=problem
        # You can have more than one problem to plot, e.g. for comparison
        self._named_problems:Dict[str,"Problem"]={"":self._problem}
        self._initialised=False
        self._output_step=0 # Will be set by the problem
        self.active=True
        #: The eigenvector to plot (given by an index). If ``None``, it will plot the normal solution.
        self.eigenvector=eigenvector
        #: The mode to plot eigenvectors, e.g. ``"abs"`` for the absolute value, ``"real"`` for the real part, etc.
        self.eigenmode:"MeshDataEigenModes"=eigenmode
        
        self._eigenanimation_m:Optional[int]=None
        self._eigenanimation_lambda:Optional[complex]=None
        
    def add_additional_problem(self,problem:"Problem",problem_name:str,output_dir:Optional[str]=None):
        self._named_problems[problem_name]=problem        
        if not problem.is_initialised():
            if output_dir is None:
                raise RuntimeError("Initialize the additional problem first or pass the output_dir")
            else:
                problem._outdir=output_dir
            problem._runmode="overwrite"
            problem.ignore_command_line=True
            problem.initialise()

    def get_eigenvalue(self,problem_name:str="")->Optional[complex]:
        """
        When plotting eigenfunctions, it will return the eigenvalue of the current eigenvector. When plotting normal solutions, it is ``None``.
        """
        if self._eigenanimation_lambda is not None:
            return self._eigenanimation_lambda
        if self.eigenvector is None:
            return None
        else:
            return self.get_problem(problem_name)._last_eigenvalues[self.eigenvector] #type:ignore
        
    def get_azimuthal_eigenmode(self,problem_name:str="")->Optional[int]:
        if self._eigenanimation_m is not None:
            return self._eigenanimation_m
        if self.eigenvector is None:
            return None
        elif self.get_problem(problem_name)._last_eigenvalues_m is None or self.eigenvector>=len(self.get_problem(problem_name)._last_eigenvalues_m):
            return 0
        else:
            return self.get_problem(problem_name)._last_eigenvalues_m[self.eigenvector]

    def get_problem(self,problem_name:str=""):
        """
        Returns the problem on which we want to plot. Useful to access the properties of the problem, e.g. sizes.
        """
        return self._named_problems[problem_name]

    def initialise(self):
        pass

    def _reset_before_plot(self):
        pass

    def _after_plot(self):
        pass

    def define_plot(self):
        """
        This functions must be implemented by the derived class to specify what should be plotted.
        """
        pass

    def plot(self):
        if not self._initialised:
            self.initialise()
            self._initialised=True
        self._has_invalid_triangulation=False
        self._reset_before_plot()        
        if self.eigenvector is not None:
            if self.eigenvector >= len(self._problem._last_eigenvectors): #type:ignore
                #print("SKIPPING PLOT :"+str(self._problem._last_eigenvectors))
                return  # No output hrere
        self.define_plot()
        self._after_plot()



class PlotTransform:
    """
    Base class to transform the data to be plotted, e.g. by mirroring, rotating, shifting, etc.
    """
    def __init__(self):
        pass

    def apply(self,coordinates:NPFloatArray,values:Optional[NPFloatArray])->Tuple[NPFloatArray,NPFloatArray]:
        return numpy.array(coordinates),numpy.array(values) #type:ignore

    def get_mirror(self):
        return [False,False,False]

class PlotTransformShift(PlotTransform):
    def __init__(self,offset_x:float,offset_y:float):
        super(PlotTransform,self).__init__()
        self.offset:NPFloatArray=numpy.array([offset_x,offset_y]) #type:ignore

    def apply(self,coordinates:NPFloatArray,values:Optional[NPFloatArray])->Tuple[NPFloatArray,NPFloatArray]:
        return numpy.transpose(numpy.transpose(numpy.array(coordinates))+self.offset),numpy.array(values) #type:ignore


class PlotTransformMirror(PlotTransform):
    def __init__(self,x:bool=False,y:bool=False,z:bool=False,tensor_transform:Optional[Callable[[NPFloatArray],NPFloatArray]]=None,offset_x:Optional[float]=None,offset_y:Optional[float]=None):
        super(PlotTransformMirror, self).__init__()
        self.mirror_x=x
        self.mirror_y = y
        self.mirror_z = z
        self.offset_x,self.offset_y=offset_x,offset_y
        self.tensor_transform=tensor_transform

    def get_mirror(self):
        return [self.mirror_x,self.mirror_y,self.mirror_z]

    def apply(self,coordinates:NPFloatArray,values:Optional[NPFloatArray])->Tuple[NPFloatArray,NPFloatArray]:
        cs=coordinates.copy()
        if values is not None and len(values.shape)>1:
            vecdim=values.shape[0]
        else:
            vecdim=None
        if vecdim is not None and len(values.shape)>2:
            tensdim=values.shape[1]
        else:
            tensdim=None
        if vecdim is not None and values is not None:
            values=values.copy() #type:ignore
        if self.mirror_x:
            cs[0]=-1.0*cs[0]
            if vecdim is not None and values is not None:
                if tensdim is None:
                    values[0]=-1.0*values[0]
                elif self.tensor_transform is None:
                    raise RuntimeError("Pass tensor_transform function to set how the tensors values/components must be transformed")
                else:
                    values=self.tensor_transform(values.copy())
        if self.mirror_y:
            cs[1]=-1.0*cs[1]
            if vecdim is not None and vecdim>1 and values is not None:
                if tensdim is None:
                    values[1]=-1*values[1]
                elif self.tensor_transform is None:
                    raise RuntimeError("Pass tensor_transform function to set how the tensors values/components must be transformed")
                else:
                    values=self.tensor_transform(values.copy())
            
        if self.offset_x is not None or self.offset_y is not None:
            offset=numpy.array([self.offset_x or 0,self.offset_y or 0])
            return numpy.transpose(numpy.transpose(numpy.array(cs))+offset),numpy.array(values) #type:ignore
        else:
            return numpy.array(cs),numpy.array(values) #type:ignore

class PlotTransformRotate90(PlotTransform):
    def __init__(self, mode:int=1,mirror_x:bool=False, mirror_y:bool=False, mirror_z:bool=False):
        super().__init__()
        self.mode=mode
        self.mirror_x = mirror_x
        self.mirror_y = mirror_y
        self.mirror_z = mirror_z

    def get_mirror(self):
        return [self.mirror_x, self.mirror_y, self.mirror_z]


    def apply(self,coordinates:NPFloatArray,values:Optional[NPFloatArray])->Tuple[NPFloatArray,NPFloatArray]:
        cs=coordinates.copy()
        if values is not None and len(values.shape)>1:
            vecdim=values.shape[0]
        else:
            vecdim=None
        if vecdim is not None and len(values.shape)>2:
            tensdim=values.shape[1]
        else:
            tensdim=None            
        if vecdim is not None and values is not None:
            values=values.copy()
        if self.mode==1:
            cs[1],cs[0]=cs[0].copy(),cs[1].copy() #type:ignore
        if vecdim is not None and vecdim>1:
            values[1],values[0]=values[0].copy(),values[1].copy() #type:ignore
        if self.mirror_x:
            cs[0]=-1*cs[0]
            if vecdim is not None and values is not None:
                values[0]=-1*values[0]
                if tensdim is not None:
                    raise RuntimeError("Rotate tensors here")
        if self.mirror_y:
            cs[1]=-1*cs[1]
            if vecdim is not None and vecdim>1 and values is not None:
                values[1]=-1*values[1]
                if tensdim is not None:
                    raise RuntimeError("Rotate tensors here")
        return numpy.array(cs),numpy.array(values) #type:ignore

#class ConcatenatedTransform(PlotTransform):
#    def __init__(self,*args):
#        super(ConcatenatedTransform, self).__init__()
#        self.transforms=[*args]

#    def apply(self, coordinates, values):
#        for t in self.transforms:
#            coordinates,values=t.apply(coordinates,values)
#        return coordinates,values


class MatplotLibPart:
    mode=None
    zindex=0
    preprocess_order=0
    mode_to_class:Dict[str,Type["MatplotLibPart"]]={}
    def __init__(self,plotter:"MatplotlibPlotter"):
        self.plotter=plotter
        pass

    def pre_process(self):
        pass

    def add_to_plot(self):
        pass

    def post_process(self):
        pass

    def set_kwargs(self,kwargs:Dict[str,Any]):
        for k,v in kwargs.items():
            if v is not None:
                setattr(self,k,v)

    @classmethod
    def register(cls, *, override:bool=False):
        def decorator(subclass:_MatPlotLibPartTypeVar)->_MatPlotLibPartTypeVar:
            if issubclass(subclass, MatplotLibPart): #type:ignore
                if subclass.mode is None:
                    raise RuntimeError("Must set the value mode")
                if subclass.mode in cls.mode_to_class.keys() and not override:
                    raise RuntimeError("Object with mode "+str(subclass.mode)+" already registered, use override=True to override")
                cls.mode_to_class[subclass.mode]=subclass
            else:
                raise RuntimeError("Must be inherited from MatplotLibPart")
            return subclass
        return decorator


class MatplotLibPartWithMeshData(MatplotLibPart):
    use_lagrangian_coordinates=False
    problem_name:str=""
    def __init__(self,plotter:"MatplotlibPlotter"):
        super(MatplotLibPartWithMeshData, self).__init__(plotter)
        self.mshcache:MeshDataCacheEntry
        self.transform:Optional[PlotTransform]=None
        self.field:Union[str,List[str]]

    def set_mesh_data(self,mshdata:MeshDataCacheEntry,field:Union[str,List[str]],transform:Optional[PlotTransform]):
        self.mshcache=mshdata
        self.field=field
        self.transform=transform


class MatplotlibTriangulationBased(MatplotLibPartWithMeshData):
    anchor_x="auto"
    anchor_y = "auto"
    min_triangle_circle_ratio=None

    def __init__(self,plotter:"MatplotlibPlotter"):
        super(MatplotlibTriangulationBased, self).__init__(plotter)
        self.triang:Any = None
        self.data:NPFloatArray
        self.ptsinside:Optional[NPIntArray]=None
        self.interpdata:Optional[List[Tuple[int,int,float]]]=None
        self.bounding_box:Optional[List[float]]=None
        self.range_mask_func:Optional[Callable[[Union[float,NPFloatArray],Union[float,NPFloatArray]],bool]]=None # set to lambda x,y : Bool



    def get_sampled_points(self,density):
        if (self.plotter.xmin is None) or (self.plotter.xmax is None) or (self.plotter.ymax  is None) or (self.plotter.ymin  is None):
            raise RuntimeError("Must use set_view before plotting ellipses")
        dx=(self.plotter.xmax-self.plotter.xmin)/density
        dy = (self.plotter.ymax - self.plotter.ymin) / density
        if dx>dy:
            dy=dx
        else:
            dx=dy

        if self.anchor_x=="auto":
            if  self.transform is not None and self.transform.get_mirror()[0]:
                self.anchor_x="left"
            else:
                self.anchor_x="right"

        assert self.bounding_box is not None
        if self.anchor_x=="left":
            xls:NPFloatArray = numpy.arange(self.bounding_box[1] - 0.5 * dx, self.bounding_box[0], -dx)  #type:ignore
        elif self.anchor_x=="right":
            xls:NPFloatArray = numpy.arange(self.bounding_box[0] + 0.5 * dx, self.bounding_box[1], dx) #type:ignore
        elif self.anchor_x=="x_min":
            xls:NPFloatArray = numpy.arange(self.plotter.xmin + 0.5 * dx, self.bounding_box[1], dx) #type:ignore
        elif self.anchor_x=="x_max":
            xls:NPFloatArray = numpy.arange(self.plotter.xmax - 0.5 * dx, self.bounding_box[0], -dx) #type:ignore
        else:
            raise RuntimeError("Strange anchor_x: "+str(self.anchor_x))

        if self.anchor_y=="auto":
            if self.transform is not None and self.transform.get_mirror()[1]:
                self.anchor_y="top"
            else:
                self.anchor_y="bottom"

        if self.anchor_y=="top":
            yls:NPFloatArray = numpy.arange(self.bounding_box[3] - 0.5 * dy, self.bounding_box[2], -dy) #type:ignore
        elif self.anchor_y=="bottom":
            yls:NPFloatArray = numpy.arange(self.bounding_box[2]+0.5*dy,self.bounding_box[3], dy) #type:ignore
        elif self.anchor_y=="y_min":
            yls:NPFloatArray = numpy.arange(self.plotter.ymin+0.5*dy,self.bounding_box[3], dy) #type:ignore
        elif self.anchor_y=="y_max":
            yls:NPFloatArray = numpy.arange(self.plotter.ymax - 0.5 * dy, self.bounding_box[2], -dy) #type:ignore
        else:
            raise RuntimeError("Strange anchor_y: "+str(self.anchor_y))
        return xls,yls

    def pre_process(self):
        coordinates = self.mshcache.get_coordinates(lagrangian=self.use_lagrangian_coordinates)
        data=self.mshcache.get_data(self.field)
        assert data is not None
        self.data = data
        if not self.transform is None:
            coordinates, self.data = self.transform.apply(coordinates, self.data)
        if not (None in [self.plotter.xmin,self.plotter.xmax,self.plotter.ymin,self.plotter.ymax]): #type:ignore
            #print("REDU")
            #Reduce the triangles that are in there
            axmin:float
            axmax:float
            aymin:float
            aymax:float
            axmin, axmax = numpy.amin(coordinates[0]), numpy.amax(coordinates[0]) #type:ignore
            aymin, aymax = numpy.amin(coordinates[1]), numpy.amax(coordinates[1]) #type:ignore
            xmin:float=self.plotter.xmin #type:ignore
            xmax:float=self.plotter.xmax #type:ignore
            ymin:float=self.plotter.ymin #type:ignore
            ymax:float = self.plotter.ymax  #type:ignore
            if axmin >= xmin and axmax <= xmax and aymin >= ymin and aymax <= ymax:   # All inside 
                reducedtris = self.mshcache.elem_indices
                #print("NOND REUDCED",self.field,len(reducedtris))
            else:
                x:NPFloatArray=coordinates[0]
                y:NPFloatArray=coordinates[1]
                pntclass = numpy.zeros((len(y),), dtype=int) #type:ignore
                for i in range(len(y)):
                    pntclass[i] = ((1 if y[i] < ymin else 0) + (2 if y[i] > ymax else 0) + (4 if x[i] < xmin else 0) + (
                        8 if x[i] > xmax else 0))
                triclass = pntclass[self.mshcache.elem_indices]
                trisNotCompletelyInside = numpy.any(triclass, axis=1) #type:ignore
                trisCompletelyInside = numpy.fromfunction(lambda i: trisNotCompletelyInside[i] == 0,(len(trisNotCompletelyInside),), dtype=int) #type:ignore
                trisToRender = trisCompletelyInside[:] #type:ignore
                interpdata = []  # Storing interpolation info at the screen boundaries
                ptsinside = numpy.fromfunction(lambda i: pntclass[i] == 0, (len(pntclass),), dtype=int) #type:ignore
                # Check the tris not completely inside whether they are completely outside or partially outside
                for ti, relevant in enumerate(trisNotCompletelyInside):
                    if not relevant: continue
                    pnt = triclass[ti]
                    andres = pnt[0] & pnt[1] & pnt[2]
                    # print andres
                    if andres == 0:  # Either at least one point inside or possibly crossing by an edge
                        trisToRender[ti] = True
                        for ei in range(3):
                            p1 = self.mshcache.elem_indices[ti][ei]
                            p2 = self.mshcache.elem_indices[ti][(ei + 1) % 3]
                            x1 = x[p1]
                            y1 = y[p1]
                            x2 = x[p2]
                            y2 = y[p2]
                            for xb in [xmin, xmax]:
                                if (x1 - xb) * (x2 - xb) < 0:
                                    linter = (xb - x1) / (x2 - x1)
                                    yinter = y1 * (1.0 - linter) + y2 * linter
                                    if yinter >= ymin and yinter <= ymax:
                                        interpdata.append((p1, p2, linter)) #type:ignore
                            for yb in [ymin, ymax]:
                                if (y1 - yb) * (y2 - yb) < 0:
                                    linter = (yb - y1) / (y2 - y1)
                                    xinter = x1 * (1.0 - linter) + x2 * linter
                                    if xinter >= xmin and xinter <= xmax:
                                        interpdata.append((p1, p2, linter)) #type:ignore
                reducedtris:NPIntArray=self.mshcache.elem_indices[trisToRender] #type:ignore
                #print("REUDCED", self.field, len(reducedtris),len(self.mshcache.elem_indices))
                self.ptsinside=ptsinside
                self.interpdata=interpdata
        else:
            reducedtris:NPIntArray = self.mshcache.elem_indices
            
        
            
        if len(reducedtris)>0:
            
            self.triang = tri.Triangulation(coordinates[0], coordinates[1], reducedtris)
            if self.min_triangle_circle_ratio is None:                
                self.min_triangle_circle_ratio=self.plotter.min_triangle_circle_ratio
            if self.min_triangle_circle_ratio is not None:
                # Filter out triangles that are too small
                newmask = tri.TriAnalyzer(self.triang).get_flat_tri_mask(self.min_triangle_circle_ratio)
                self.triang.set_mask(newmask) #type:ignore
           
                
                
            # Also create the bounding box
            if self.ptsinside is not None:
                
                if numpy.any(self.ptsinside):
                    pxmin:float= numpy.amin(coordinates[0, self.ptsinside]) #type:ignore
                    pxmax:float= numpy.amax(coordinates[0, self.ptsinside]) #type:ignore
                    pymin:float = numpy.amin(coordinates[1, self.ptsinside])#type:ignore
                    pymax:float =  numpy.amax(coordinates[1, self.ptsinside]) #type:ignore
                elif len(self.interpdata)>0:
                    idata=self.interpdata[0]
                    pxmin:float= coordinates[0, idata[0]] * (1.0 - idata[2]) + coordinates[0, idata[1]] * (idata[2]) #type:ignore
                    pxmax:float= pxmin #type:ignore
                    pymin:float =coordinates[1, idata[0]] * (1.0 - idata[2]) + coordinates[1, idata[1]] * (idata[2]) #type:ignore
                    pymax:float = pymin #type:ignore
                else: 
                    raise RuntimeError("Should not happen")
                for idata in self.interpdata: #type:ignore
                    interp:float = coordinates[0, idata[0]] * (1.0 - idata[2]) + coordinates[0, idata[1]] * (idata[2]) #type:ignore
                    if interp < pxmin: pxmin = interp #type:ignore
                    if interp > pxmax: pxmax = interp #type:ignore
                    interp:float = coordinates[1, idata[0]] * (1.0 - idata[2]) + coordinates[1, idata[1]] * (idata[2]) #type:ignore
                    if interp < pymin: pymin = interp #type:ignore
                    if interp > pymax: pymax = interp #type:ignore
            else:
                pxmin:float = numpy.amin(coordinates[0]) #type:ignore
                pxmax:float= numpy.amax(coordinates[0])  #type:ignore
                pymin:float = numpy.amin(coordinates[1])  #type:ignore
                pymax:float=numpy.amax(coordinates[1]) #type:ignore
            self.bounding_box = [pxmin, pxmax, pymin, pymax] #type:ignore
            
            if self.plotter.merge_duplicate_points:
                xs,ys= self.triang.x, self.triang.y #type:ignore
                xy=numpy.stack((xs,ys), axis=1)
                
                from scipy.spatial import KDTree
                kdtree=KDTree(xy)
                same=kdtree.query_pairs(1e-8*numpy.sqrt((pxmax-pxmin)**2+(pymax-pymin)**2)) #type:ignore
                
                for pinds in same:
                    xs[pinds[1]]=xs[pinds[0]]
                    ys[pinds[1]]=ys[pinds[0]]
                    
                xy=numpy.stack((xs,ys), axis=1)
                dummy,inds,duplicarr=numpy.unique(["{} {}".format(i, j) for i,j in xy], return_index=True,return_inverse=True,)	
                xs,ys=xs[inds],ys[inds]
                self.triang.triangles=duplicarr[self.triang.triangles]+0 #type:ignore
                if len(self.data.shape)>1:
                        self.data=numpy.array([self.data[i,inds] for i in range(self.data.shape[0])])
                else:
                    self.data=self.data[inds]
                data=self.data
                #print("After processing",len(self.triang.x),len(xs),self.data.shape)
                nondegen=[]
                for t in self.triang.triangles:
                    if t[0]==t[1] or t[0]==t[2] or t[1]==t[2]:
                        pass
                    else:
                        nondegen.append(t)
                self.triang.triangles=numpy.array(nondegen) #type:ignore
                
                self.triang.x=xs+0
                self.triang.y=ys+0
                self.triang=tri.Triangulation(xs,ys,self.triang.triangles) #type:ignore
                coordinates=numpy.array([xs,ys])
                
                if self.min_triangle_circle_ratio is not None:
                    # Filter out triangles that are too small
                    newmask = tri.TriAnalyzer(self.triang).get_flat_tri_mask(self.min_triangle_circle_ratio)
                    self.triang.set_mask(newmask) #type:ignore
                
             
            if self.plotter.min_triangle_area is not None:
                rescaled_coords_x=coordinates[0]/(pxmax-pxmin) #type:ignore
                rescaled_coords_y=coordinates[1]/(pymax-pymin) #type:ignore
                triangles = self.triang.triangles
                x = rescaled_coords_x
                y = rescaled_coords_y
                # Calculate area using the shoelace formula for each triangle
                areas = 0.5 * numpy.abs(
                    (x[triangles[:,0]] * (y[triangles[:,1]] - y[triangles[:,2]]) +
                     x[triangles[:,1]] * (y[triangles[:,2]] - y[triangles[:,0]]) +
                     x[triangles[:,2]] * (y[triangles[:,0]] - y[triangles[:,1]]))
                )
                #print("MINAREA",numpy.amin(areas),self.plotter.min_triangle_area,numpy.amax(areas))
                if self.triang.mask is None:
                    self.triang.set_mask(areas <= self.plotter.min_triangle_area)                
                else:
                    self.triang.set_mask(numpy.logical_or(self.triang.mask, areas <= self.plotter.min_triangle_area))                
                
                
                 
        else:
            self.triang = None
            self.bounding_box=None



    def get_visible_data_range(self,data:Optional[NPFloatArray]=None)->Tuple[Optional[float],Optional[float]]:
        if data is None:
            data=self.data
        if self.range_mask_func is not None:
            mi:float=numpy.nanmax(data) #type:ignore
            ma:float = numpy.nanmin(data) #type:ignore
            coordinates = self.mshcache.get_coordinates(lagrangian=self.use_lagrangian_coordinates)
            d = self.mshcache.get_data(self.field)
            assert d is not None
            if not self.transform is None:
                coordinates, d = self.transform.apply(coordinates, d)

            for i,x,y in zip(range(len(coordinates[0])),coordinates[0],coordinates[1]):
                if self.range_mask_func(x,y)!=False:
                    if data[i]>ma:
                        ma=data[i]
                    if data[i]<mi:
                        mi=data[i]
            return cast(float,mi), cast(float,ma) #type:ignore


        if self.ptsinside is not None and self.interpdata is not None:
            if not numpy.any(self.ptsinside): #type:ignore
                return None,None #type:ignore
            #print(self.ptsinside)
            mi:float
            ma:float
            mi,ma=numpy.nanmin(data[self.ptsinside]),numpy.nanmax(data[self.ptsinside]) #type:ignore
            for idata in self.interpdata:
                interp:float = data[idata[0]] * (1.0 - idata[2]) + data[idata[1]] * (idata[2]) #type:ignore
                if interp < mi: mi = interp
                if interp > ma: ma = interp
            return cast(float,mi), cast(float,ma) #type:ignore
        else:
            return cast(float,numpy.nanmin(data)),cast(float,numpy.nanmax(data)) #type:ignore

@MatplotLibPart.register()
class MatplotLibTricontourf(MatplotlibTriangulationBased):
    mode="tricontourf"
    extend="both"
    datamap=None
    invisible:bool=False
    dataoffset:float=0
    check_for_finite_values:bool=False

    def __init__(self,plotter:"MatplotlibPlotter"):
        super().__init__(plotter=plotter)
        self.colorbar:Optional["MatplotLibColorbar"]=None
        self.scaled_data:Optional[NPFloatArray]=None

    def pre_process(self):
        if self.colorbar is None:
            raise RuntimeError("Please use the arg colorbar=... to plot surfaces")
        assert isinstance(self.colorbar,MatplotLibColorbar)
        super().pre_process()

        #Vector to magnitude
        if len(self.data.shape)>1:
            nd=self.data.shape[0]
            newdata:NPFloatArray=numpy.zeros((self.data.shape[1])) #type:ignore
            for i in range(nd):
                newdata+=self.data[i,:]**2 #type:ignore
            self.data=numpy.sqrt(newdata)

        if self.datamap is not None:
            self.data=self.datamap(self.data)
        self.scaled_data=cast(NPFloatArray,self.data*self.colorbar.factor+self.colorbar.offset+self.dataoffset)
        mi,ma=self.get_visible_data_range()
        if mi is not None:
            mi=mi*self.colorbar.factor+self.colorbar.offset+self.dataoffset
        if ma is not None:
            ma = ma * self.colorbar.factor + self.colorbar.offset+self.dataoffset
        if mi is not None and ma is not None:
            self.colorbar.consider_range(mi,ma)

    def add_to_plot(self):
        cb=self.colorbar
        assert cb is not None
        kwargs={}
        if cb.range is None:
            return
        if (cb.range is None) or (cb.range.vmin is None) or (cb.range.vmax is None):
            return
        delta0:float = cb.range.vmax - cb.range.vmin
        delta:float = 0
        if delta0 < 1e-13: delta = 0.5 * (cb.range.vmax + cb.range.vmin) * 1e-4
        if delta < 1e-13: delta = 1e-13
        kwargs["vmin"] = cb.range.vmin - 0.01 * delta
        kwargs["vmax"] = cb.range.vmax + 0.01 * delta
        if isinstance(cb.Ndisc,(tuple,list,numpy.ndarray)):
            kwargs["levels"] = cb.range.vmin+(cb.range.vmax-cb.range.vmin)*numpy.array(cb.Ndisc) #type:ignore
            #print(kwargs["levels"])
            #exit()
        elif cb.Ndisc is not None and cb.Ndisc > 0:
            if delta0>1e-20:
                kwargs["levels"] = numpy.linspace(cb.range.vmin , cb.range.vmax , cb.Ndisc+1 ,endpoint=True) #type:ignore
            else:
                kwargs["levels"] = numpy.linspace(cb.range.vmin - 0.1 * delta, cb.range.vmax + 0.1 * delta, cb.Ndisc, endpoint=True) #type:ignore
            #print(cb.range.vmin,cb.range.vmax)
            rang=kwargs["levels"][-1]-kwargs["levels"][0]
            if rang<1e-12:
                kwargs["levels"]=[kwargs["levels"][0]-0.01*delta,0.5*(kwargs["levels"][-1]+kwargs["levels"][0]),kwargs["levels"][-1]+0.01*delta]

        norm=cb.get_norm()
        if isinstance(norm,matplotlib.colors.LogNorm):
            del kwargs["levels"]
            #kwargs["locator"]=matplotlib.ticker.LogLocator()
            kwargs["vmin"]=norm.vmin
            kwargs["vmax"] = norm.vmax
            assert self.scaled_data is not None
            self.scaled_data=numpy.maximum(norm.vmin,self.scaled_data)

            kwargs["levels"]=numpy.power(10,numpy.linspace(numpy.log10(norm.vmin),numpy.log10(norm.vmax),num=cb.Ndisc)) #type:ignore

            #print("WKWARGS",kwargs)
            #print(self.scaled_data)

        if self.invisible!=True and (self.triang is not None):
            if self.check_for_finite_values:
                isbad=numpy.isnan(self.scaled_data)
                mask = numpy.any(numpy.where(isbad[self.triang.triangles], True, False), axis=1)
                self.triang.set_mask(mask)
            plt.tricontourf(self.triang, self.scaled_data,cmap=cb.cmap,norm=norm,extend=self.extend, **kwargs) #type:ignore


@MatplotLibPart.register()
class MatplotLibTricontour(MatplotlibTriangulationBased):
    mode="tricontour"
    levels=None
    linecolor=None
    linewidths=1
    def __init__(self,plotter:"MatplotlibPlotter"):
        super().__init__(plotter=plotter)

    def pre_process(self):
        super().pre_process()
        #Vector to magnitude
        if len(self.data.shape)>1:
            nd=self.data.shape[0]
            newdata:NPFloatArray=numpy.zeros((self.data.shape[1]),dtype=numpy.float64) #type:ignore
            for i in range(nd):
                newdata+=self.data[i,:]**2 #type:ignore
            self.data=numpy.sqrt(newdata)

        self.scaled_data=self.data
        #mi,ma=self.get_visible_data_range()

    def add_to_plot(self):
        kwargs={}
        if self.levels is not None:
            kwargs["levels"]=self.levels
        if self.linecolor is not None:
            kwargs["colors"]=self.linecolor
        plt.tricontour(self.triang, self.scaled_data,zorder=self.zindex,linewidths=self.linewidths,**kwargs) #type:ignore




@MatplotLibPart.register()
class MatplotlibVectorFieldArrows(MatplotlibTriangulationBased):
    mode = "arrows"
    zindex = 1
    scalemode="normalized"
    arrowlength=-0.025
    arrowcenter=0.5

    arrowstyle="->"
    linewidths = 2
    linecolor = "black"
    arrowdensity:float=50
    use_quiver:bool=False # Quiver plots faster, but is less customizeable

    invisible:bool=False

    def add_to_plot(self):
        if self.invisible:
            return
        vx=self.data[0]
        vy = self.data[1]
        if (self.plotter.xmin is None) or (self.plotter.xmax is None) or (self.plotter.ymax  is None) or (self.plotter.ymin  is None):
            raise RuntimeError("Must use set_view before plotting arrows")

        xls,yls=self.get_sampled_points(self.arrowdensity)       

        if len(xls)==0  or len(yls)==0:
            return
        xis:NPFloatArray
        yis:NPFloatArray
        xis, yis = numpy.meshgrid(xls, yls) #type:ignore
        
        
        

        if self.plotter.crash_on_invalid_triangulation:
            #print(len(vx),len(self.triang.x))
            interp_u = tri.LinearTriInterpolator(self.triang, vx)
            interp_v = tri.LinearTriInterpolator(self.triang, vy)
        else:
            try:
                interp_u = tri.LinearTriInterpolator(self.triang, vx)
                interp_v = tri.LinearTriInterpolator(self.triang, vy)
            except:
                if False:
                    try:
                        print("INVALID TRIANGULATION! Testing CubicTriInterpolator")
                        interp_u = tri.CubicTriInterpolator(self.triang, vx,kind="geom")
                        interp_v = tri.CubicTriInterpolator(self.triang, vy,kind="geom")                    
                    except:
                        print("INVALID TRIANGULATION, also with the cubic interpolator! Skipping plot")
                        self.plotter._has_invalid_triangulation=True
                        return 
                else:
                    print("INVALID TRIANGULATION! Skipping plot")
                    self.plotter._has_invalid_triangulation=True
                    return 

        uinter = interp_u(xis, yis)
        vinter = interp_v(xis, yis)

        ax:matplotlib.axes.Axes=plt.gca() #type:ignore
        arrl=self.arrowlength
        if arrl<0:
            arrl=-arrl*max(self.plotter.xmax-self.plotter.xmin,self.plotter.ymax-self.plotter.ymin)
        if not self.use_quiver:
            arrowprops=dict(color=self.linecolor, fc=self.linecolor,arrowstyle=self.arrowstyle, lw=self.linewidths)
            if isinstance(self.arrowstyle,dict):
                del arrowprops["arrowstyle"]
                arrowprops.update(self.arrowstyle)

            for xi in range(0, len(xls)):
                for yi in range(0, len(yls)):
                    lu = uinter[yi][xi]
                    if lu is numpy.ma.masked: continue
                    lv = vinter[yi][xi]
                    lm = numpy.sqrt(lu * lu + lv * lv)
                    if lm < 1e-50: continue
                    if self.scalemode == "normalized":
                        denom = 1.0 / lm
                    else:
                        denom = 1.0
                    adx:float = arrl * lu * denom
                    ady:float = arrl * lv * denom
                    acx:float = xls[xi]
                    acy:float = yls[yi]
                    

                    ax.annotate("", xy=(acx + self.arrowcenter * adx, acy + self.arrowcenter * ady), #type:ignore
                                    xytext=(acx - (1.0 - self.arrowcenter) * adx, acy - (1.0 - self.arrowcenter) * ady), #type:ignore
                                    arrowprops=arrowprops,
                                    zorder=self.zindex,annotation_clip=False) 
        else:
            lm = numpy.sqrt(uinter **2 + vinter **2)
            uinter=numpy.ma.masked_where(lm<1e-50,uinter)
            vinter=numpy.ma.masked_where(lm<1e-50,vinter)
            if self.scalemode == "normalized":
                uinter /= lm
                vinter /= lm
            if self.arrowcenter<0.25:
                pivot="tail"
            elif self.arrowcenter<0.75:
                pivot="mid"
            else:
                pivot="head"
            width=self.plotter.xmax-self.plotter.xmin
            ax.quiver(xls,yls,uinter,vinter,pivot=pivot,width=2*self.linewidths*(width))



@MatplotLibPart.register()
class MatplotlibVectorFieldStreams(MatplotlibTriangulationBased):
    mode = "streamlines"
    numx=200
    numy=200
    boundoffsx=0.5
    boundoffsy=0.5
    density=10
    zindex = 1
    colorbar:Optional["MatplotLibColorbar"] = None
    dataoffset=0
    colorfield=None

    linewidths = 0.5
    linecolor = "black"
    density=1
    minlength=0.1
    maxlength=4.0
    arrowstyle=None
    start_points=None

    def __init__(self,plotter:"MatplotlibPlotter"):
        super().__init__(plotter=plotter)
        self.scaled_data=None
        self.color_data=None

    def pre_process(self):
        if self.colorbar is not None and self.colorfield is not None:
            backup = self.field
            self.field = self.colorfield
            super(MatplotlibVectorFieldStreams, self).pre_process()
            if self.data is None:
                raise RuntimeError("Cannot use "+str(self.colorfield)+" to color the streamlines. Probably not a valid field?")
            self.field=backup
            self.color_data=self.data

        super(MatplotlibVectorFieldStreams, self).pre_process()
        #Vector to magnitude
        nd=self.data.shape[0]
        newdata:NPFloatArray=numpy.zeros((self.data.shape[1]),dtype=numpy.float64) #type:ignore
        for i in range(nd):
            newdata+=self.data[i,:]**2 #type:ignore
        newdata=numpy.sqrt(newdata)
        if self.colorbar is not None:
            self.scaled_data=newdata*self.colorbar.factor+self.colorbar.offset+self.dataoffset
            if self.colorfield is None:
                mi,ma=self.get_visible_data_range(newdata)
                if mi is not None:
                    mi=mi*self.colorbar.factor+self.colorbar.offset+self.dataoffset
                if ma is not None:
                    ma = ma * self.colorbar.factor + self.colorbar.offset+self.dataoffset
                if mi is not None and ma is not None:
                    self.colorbar.consider_range(mi,ma)
            else:
                mi, ma = self.get_visible_data_range(self.color_data)
                if mi is not None:
                    mi = mi * self.colorbar.factor + self.colorbar.offset + self.dataoffset
                if ma is not None:
                    ma = ma * self.colorbar.factor + self.colorbar.offset + self.dataoffset
                if mi is not None and ma is not None:
                    self.colorbar.consider_range(mi, ma)
        else:
            self.scaled_data = newdata + self.dataoffset

    def add_to_plot(self):
        vx=self.data[0]
        vy = self.data[1]
        if (self.plotter.xmin is None) or (self.plotter.xmax is None) or (self.plotter.ymax  is None) or (self.plotter.ymin  is None):
            raise RuntimeError("Must use set_view before plotting streamlines")

        assert self.bounding_box is not None
        lx = (self.bounding_box[1] - self.bounding_box[0])
        ly = (self.bounding_box[3] - self.bounding_box[2])

        xls:NPFloatArray = numpy.linspace(self.bounding_box[0] + self.boundoffsx * lx / self.numx, self.bounding_box[1] - self.boundoffsx * lx / self.numx, self.numx) #type:ignore
        yls:NPFloatArray = numpy.linspace(self.bounding_box[2] + self.boundoffsy * ly / self.numy, self.bounding_box[3] - self.boundoffsy * ly / self.numy, self.numy) #type:ignore

        if numpy.amax(self.scaled_data)<1e-20: #type:ignore
            return

        if len(xls)==0  or len(yls)==0:
            return
        xi:NPFloatArray
        yi:NPFloatArray
        xi, yi = numpy.meshgrid(xls, yls) #type:ignore
        if self.plotter.crash_on_invalid_triangulation:
            interp_u = tri.LinearTriInterpolator(self.triang, vx)
            interp_v = tri.LinearTriInterpolator(self.triang, vy)
        else:
            try:
                interp_u = tri.LinearTriInterpolator(self.triang, vx)
                interp_v = tri.LinearTriInterpolator(self.triang, vy)
            except:
                self.plotter._has_invalid_triangulation=True
                return

        uinter = interp_u(xi, yi)
        vinter = interp_v(xi, yi)
        if self.range_mask_func:
            mask=numpy.logical_not(self.range_mask_func(xi,yi))
            #print("MASK",mask)
            #print("UINTER",uinter)
            uinter.mask|=mask
            vinter.mask |=mask
            #exit()
        scal=1
        kwargs={}
        kwargs["color"]=self.linecolor
        if self.colorbar is not None:
            interp_mag = tri.LinearTriInterpolator(self.triang, self.scaled_data if self.color_data is None else self.color_data)
            inter_mag = interp_mag(xi, yi)
            kwargs["color"] = inter_mag
            kwargs["cmap"]=self.colorbar.cmap
            kwargs["norm"] = self.colorbar.get_norm()

        kwargs["linewidth"] = self.linewidths
        kwargs["density"]=self.density
        kwargs["minlength"] = self.minlength
        kwargs["maxlength"] = self.maxlength

        if self.arrowstyle is not None:
            kwargs["arrowstyle"]=self.arrowstyle

        if self.start_points is not None:
            kwargs["start_points"]=self.start_points


        plt.streamplot(xls, yls, scal * uinter, scal * vinter, **kwargs) #type:ignore







@MatplotLibPart.register()
class MatplotlibTensorFieldEllipses(MatplotlibTriangulationBased):
    mode = "ellipses"
    zindex = 10    
    linewidths = 2
    linecolor = "black"
    density=20
    size=-1/20.0
    scalemode = "normalized"
    facecolor=None


    def add_to_plot(self):
        Txx=self.data[0][0]
        Txy=self.data[0][1]
        Tyy=self.data[1][1]
        
        xls,yls=self.get_sampled_points(self.density)

        if len(xls)==0  or len(yls)==0:
            return
        xis:NPFloatArray
        yis:NPFloatArray
        xis, yis = numpy.meshgrid(xls, yls) #type:ignore
        if self.plotter.crash_on_invalid_triangulation:
            interp_Txx = tri.LinearTriInterpolator(self.triang, Txx)
            interp_Txy = tri.LinearTriInterpolator(self.triang, Txy)
            interp_Tyy = tri.LinearTriInterpolator(self.triang, Tyy)
        else:
            try:
                interp_Txx = tri.LinearTriInterpolator(self.triang, Txx)
                interp_Txy = tri.LinearTriInterpolator(self.triang, Txy)
                interp_Tyy = tri.LinearTriInterpolator(self.triang, Tyy)
            except:            
                print("INVALID TRIANGULATION! Skipping plot")
                self.plotter._has_invalid_triangulation=True
                return 

        Txx_inter = interp_Txx(xis, yis)
        Txy_inter = interp_Txy(xis, yis)
        Tyy_inter = interp_Tyy(xis, yis)


        ax:matplotlib.axes.Axes=plt.gca() #type:ignore
        arrl=self.size
        if arrl<0:
            arrl=-arrl*max(self.plotter.xmax-self.plotter.xmin,self.plotter.ymax-self.plotter.ymin)
        ellipses = []
        for xi in range(0, len(xls)):
            for yi in range(0, len(yls)):
                lxx = Txx_inter[yi][xi]
                if lxx is numpy.ma.masked: continue
                lxy = Txy_inter[yi][xi]
                lyy = Tyy_inter[yi][xi]

                #print([[Txx_inter,Txy_inter],[Txy_inter,Tyy_inter]])
                try:
                    w,v=numpy.linalg.eigh([[lxx,lxy],[lxy,lyy]])
                    #dotp=v[0][0]*v[1][0]+v[0][1]*v[1][1]
                    #print( xls[xi], yls[xi],w,v,[[lxx,lxy],[lxy,lyy]],dotp)
                    largest=abs(w[0])
                    smallest=abs(w[1])
                    if largest>=smallest:
                        theta=numpy.arctan2(v[0][1],v[0][0])
                    else:
                        smallest,largest=largest,smallest
                        theta=numpy.arctan2(v[1][1],v[1][0])
                    #theta=0
                except:
                    theta=0.0
                    largest=1.0
                    smallest=1.0

                total=numpy.sqrt(largest**2+smallest**2)/numpy.sqrt(2.0)
                if total>1e-20:
                    largest/=total
                    smallest/=total
                else:
                    largest=1
                    smallest=1

                if self.scalemode != "normalized":
                    raise RuntimeError("TODO: Non normalized mode")
                    #denom = 1.0 / lm
                else:
                    denom = 1.0
                
                adx:float = arrl * largest * denom
                ady:float = arrl * smallest * denom
                acx:float = xls[xi]
                acy:float = yls[yi]
                #print(acx,acy,theta,largest,smallest)
                ell=Ellipse((acx,acy),adx,ady,angle=theta*180/numpy.pi)
                
                ellipses.append(ell)
        #ax.add_patch(ell)
        pc=collections.PatchCollection(ellipses)
        pc.set_linewidth(self.linewidths)
        pc.set_edgecolor(self.linecolor)                
        if self.facecolor is None:
            pc.set_facecolor("none")
        else:
            pc.set_facecolor(self.facecolor)
        #if self.facecolor is None:
        #    pc.set_fill(False)
        ax.add_collection(pc)
                
                




@MatplotLibPart.register()
class MatplotLibInterfaceLine(MatplotLibPartWithMeshData):
    mode="interfaceline"
    linewidths=2
    linecolor="black"
    def __init__(self,plotter:"MatplotlibPlotter"):
        super(MatplotLibInterfaceLine, self).__init__(plotter=plotter)

    def add_to_plot(self):        
        coordinates = self.mshcache.get_coordinates(lagrangian=self.use_lagrangian_coordinates)        
        data=self.mshcache.get_data(self.field)
        #assert data is not None
        if not self.transform is None:
            coordinates,data=self.transform.apply(coordinates,data)
        lines,_=self.mshcache.get_interface_line_segments()
        for lentry in lines:
            x=[coordinates[0,lentry[i]] for i in range(len(lentry))]
            y = [coordinates[1, lentry[i]] for i in range(len(lentry))]
            plt.gca().plot(x,y,color=self.linecolor,lw=self.linewidths) #type:ignore




@MatplotLibPart.register()
class MatplotLibInterfaceCmap(MatplotLibInterfaceLine):
    mode="interfacecmap"
    linewidths = 6
    zindex = 10
    border_color=None
    border_width=3

    def __init__(self,plotter:"MatplotlibPlotter"):
        super().__init__(plotter)
        self.field=""
        self.colorbar:Optional[MatplotLibColorbar]=None
        self._coordinates=None
        self._data:Optional[NPFloatArray]=None
        self._lsegs:Optional[List[List[int]]]=None
        self._ninter=None
        self.scaled_data=None

    def get_visible_data_range(self)->Tuple[float,float]:
        assert self._data is not None
        return numpy.amin(self._data),numpy.amax(self._data) #type:ignore #TODO: Crop outside

    def pre_process(self):
        self._coordinates=self.mshcache.get_coordinates(lagrangian=self.use_lagrangian_coordinates)
        self._data=self.mshcache.get_data(self.field)
        assert self._data is not None
        assert self.colorbar is not None
        if not self.transform is None:            
            self._coordinates,self._data=self.transform.apply(self._coordinates,self._data)
        self._lsegs, self._ninter=self.mshcache.get_interface_line_segments()

        self.scaled_data = self._data * self.colorbar.factor + self.colorbar.offset
        mi, ma = self.get_visible_data_range()
        mi = mi * self.colorbar.factor + self.colorbar.offset
        ma = ma * self.colorbar.factor + self.colorbar.offset
        self.colorbar.consider_range(mi, ma)

    def add_to_plot(self):
        coordinates = self._coordinates
        data=self.scaled_data

        assert self._lsegs is not None
        assert coordinates is not None
        assert data is not None
        assert self.colorbar is not None

        if self.border_color is not None and self.border_width is not None and self.border_width>0:
            for lentry in self._lsegs:
                x = [coordinates[0, lentry[i]] for i in range(len(lentry))]
                y = [coordinates[1, lentry[i]] for i in range(len(lentry))]
                d = [data[lentry[i]] for i in range(len(lentry))]
                points = numpy.array([x, y]).T.reshape(-1, 1, 2) #type:ignore
                segments = numpy.concatenate([points[:-2], points[1:-1], points[2:]], axis=1) #type:ignore
                lc = collections.LineCollection(segments, facecolors=self.border_color, colors=self.border_color,linewidths=self.linewidths+self.border_width,zorder=self.zindex) #type:ignore
                lc.set_array(numpy.asarray(d)) #type:ignore
                plt.gca().add_collection(lc) #type:ignore


        for lentry in self._lsegs:
            x=[coordinates[0,lentry[i]] for i in range(len(lentry))]
            y = [coordinates[1, lentry[i]] for i in range(len(lentry))]
            d=[data[lentry[i]] for i in range(len(lentry))]
            points = numpy.array([x, y]).T.reshape(-1, 1, 2) #type:ignore
            segments = numpy.concatenate([points[:-2],points[1:-1], points[2:]], axis=1) #type:ignore
            lc = collections.LineCollection(segments, cmap=self.colorbar.cmap, norm=self.colorbar.get_norm(), linewidth=self.linewidths,zorder=self.zindex) #type:ignore
            lc.set_array(numpy.asarray(d)) #type:ignore
            plt.gca().add_collection(lc) #type:ignore

@MatplotLibPart.register()
class MatplotlibInterfaceArrows(MatplotLibPartWithMeshData):
    mode="interfacearrows"
    zindex = 10.0
    linewidths=None
    arrowstyle=None
    arrowprops=None
    linecolor=None
    arrowdensity=100.0
    lengthfactor=1.0
    attached_with_head:Union[bool,Literal["positive","negative"]]=False
    start_index=None
    end_index=None
    skip_index=None
    start_with_offset=True
    end_with_offset=True
    ignore_range_contribution=False
    skip_outside=False


    def __init__(self,plotter:"MatplotlibPlotter"):
        super().__init__(plotter)
        self._arrows:List[Tuple[float,float,float,float,float]]=[]
        self._coordinates:Optional[NPFloatArray]=None
        self.arrowkey:Optional[MatplotLibArrowKey] = None

    def pre_process(self):
        super().pre_process()
        #Generate the arrows
        self._coordinates = self.mshcache.get_coordinates(lagrangian=self.use_lagrangian_coordinates)
        self._data = self.mshcache.get_data(self.field)

        coordinates = self.mshcache.get_coordinates(lagrangian=self.use_lagrangian_coordinates)
        assert self._data is not None
        data = self._data
        
        nx=self.mshcache.get_data("normal_x")
        ny = self.mshcache.get_data("normal_y")
        assert nx is not None
        assert ny is not None
        
        # Scale by the aspect ratio
        asp=float(self.plotter.aspect_ratio)
        if self.plotter.aspect_ratio is not True and self.plotter.aspect_ratio!=1:            
            nxl=numpy.sqrt(nx*nx+ny*ny*(asp*asp))
            nx/=nxl
            ny/=nxl*asp
        
        dx=data*nx
        dy=data*ny
        if not self.transform is None:
            coordinates, data = self.transform.apply(coordinates, numpy.array([dx,dy])) #type:ignore
            dx=data[0,:]
            dy = data[1, :]
        lines, _ = self.mshcache.get_interface_line_segments()

        self._arrows=[]

        
        for lentry in lines:
            x:List[float] = [coordinates[0, lentry[i]] for i in range(len(lentry))]
            y:List[float] = [coordinates[1, lentry[i]] for i in range(len(lentry))]
            dxx:List[float]=[dx[lentry[i]]*asp for i in range(len(lentry))]
            dyy:List[float] = [dy[lentry[i]] for i in range(len(lentry))]
            datasegs:List[float]=[self._data[lentry[i]] for i in range(len(lentry))]

            if self.arrowdensity is not None:
                spacing:float=0.0
                if (self.plotter.xmax is not None) and (self.plotter.ymax is not None):
                    spacing=(self.plotter.xmax - self.plotter.xmin) / self.arrowdensity #type:ignore
                if (self.plotter.ymax is not None) and (self.plotter.ymax is not None): 
                    spacing=max(spacing, (self.plotter.ymax - self.plotter.ymin) / self.arrowdensity)  #type:ignore
                if spacing==0:
                    raise RuntimeError("Cannot determine interface arrow distance without set_view first")
                interflengthsL:List[float] = []
                interflength:float = 0.0
                for i in range(len(x)):
                    if i > 0:
                        dl:float = numpy.sqrt((x[i] - x[i - 1]) * (x[i] - x[i - 1]) + (y[i] - y[i - 1]) * (y[i] - y[i - 1]))
                    else:
                        dl:float = 0.0
                    interflength = interflength + dl
                    interflengthsL.append(interflength)
                interflengths:NPFloatArray = numpy.array(interflengthsL)  #type:ignore
                splineorder = 2
                if len(x) < splineorder:
                    splineorder = len(x)                
                if not numpy.all(numpy.diff(interflengths) > 0.0): #type:ignore
                    for x,y,l in zip(x,y,interflengths):  #type:ignore
                        print(x,y,l)
                    raise ValueError('x must be strictly increasing')
                xinter = interpolate.InterpolatedUnivariateSpline(interflengths, x, k=splineorder)
                yinter = interpolate.InterpolatedUnivariateSpline(interflengths, y, k=splineorder)
                dxinter = interpolate.InterpolatedUnivariateSpline(interflengths, dxx, k=splineorder)
                dyinter = interpolate.InterpolatedUnivariateSpline(interflengths, dyy, k=splineorder)
                datainter=interpolate.InterpolatedUnivariateSpline(interflengths,datasegs,k=splineorder)
                numpts = int(interflength / spacing)
                if numpts>1:
                    edgedelta=0.5*interflength/(numpts-1)
                    interpos:NPFloatArray  = numpy.linspace(0 + edgedelta*self.start_with_offset, interflength - edgedelta*self.end_with_offset, numpts - 1, endpoint=True) #type:ignore
                else:
                    interpos:NPFloatArray =numpy.array([0.5*interflength]) #type:ignore

                xA:NPFloatArray  = xinter(interpos) #type:ignore
                yA:NPFloatArray  = yinter(interpos) #type:ignore
                dxxA:NPFloatArray  = dxinter(interpos) #type:ignore
                dyyA:NPFloatArray  = dyinter(interpos) #type:ignore
                datasegsA:NPFloatArray =datainter(interpos) #type:ignore
            else:
                xA=x #type:ignore
                yA=y #type:ignore
                dxxA=dxx #type:ignore
                dyyA=dyy #type:ignore
                datasegsA=datasegs #type:ignore


            vma:float=0
            for i in range(len(xA)): #Each point
                if self.skip_outside and (xA[i] < self.plotter.xmin or xA[i] > self.plotter.xmax or yA[i] < self.plotter.ymin or yA[i] > self.plotter.ymax):
                    continue
                self._arrows.append((xA[i],yA[i],dxxA[i],dyyA[i],datasegsA[i])) #type:ignore
                vma=max(vma,numpy.sqrt(dxxA[i]*dxxA[i]+dyyA[i]*dyyA[i])) #type:ignore
            assert self.arrowkey is not None
            if not self.ignore_range_contribution:
                self.arrowkey.consider_range(vma)


    def add_to_plot(self):
        assert self.arrowkey is not None
        assert self.arrowkey.range is not None
        if (self.arrowkey.range.vmax is None):
            return
        ax:matplotlib.axes.Axes = plt.gca() #type:ignore
        arrl = self.arrowkey.arrow_length_scale
        assert arrl is not None
        lw=self.linewidths if self.linewidths is not None else self.arrowkey.linewidths
        arrs = self.arrowstyle if self.arrowstyle is not None else self.arrowkey.arrowstyle
        arrp=self.arrowprops if self.arrowprops is not None else self.arrowkey.arrowprops        
        lc=self.linecolor if self.linecolor is not None else self.arrowkey.linecolor
        if arrp is None:
            arrp=dict(color=lc, fc=lc, arrowstyle=arrs, lw=lw)
        for arr in self._arrows[self.start_index:self.end_index:self.skip_index]:
            acx = arr[0]
            acy = arr[1]
            adx = arr[2]*arrl
            ady = arr[3]*arrl
            if self.attached_with_head is True or (self.attached_with_head=="negative" and arr[4]<0) or (self.attached_with_head=="positive" and arr[4]>0):
                ax.annotate("", xytext=(acx - adx, acy - ady), xy=(acx, acy), #type:ignore
                            arrowprops=arrp, zorder=self.zindex,
                            annotation_clip=False)
            else:
                ax.annotate("", xy=(acx+adx , acy+ady),xytext=(acx , acy), #type:ignore
                            arrowprops=arrp,zorder=self.zindex,annotation_clip=False)




@MatplotLibPart.register()
class MatplotLibElementOutlines(MatplotLibPartWithMeshData):
    mode="outlines"
    linewidths=1
    linecolor="black"
    zindex=20
    def __init__(self,plotter:"MatplotlibPlotter"):
        super().__init__(plotter=plotter)

    def add_to_plot(self):
        allines:List[List[Tuple[float]]] = []
        tr=self.transform
        mesh=self.mshcache.mesh
        if isinstance(self.plotter.eigenvector,int):
            backup_dofs, backup_pinned = mesh.get_problem(self.problem_name).set_eigenfunction_as_dofs(self.plotter.eigenvector, mode=self.plotter.eigenmode) #type:ignore

        ss=mesh.get_output_scale("spatial")
        for n in range(mesh.nelement()):
            e = mesh.element_pt(n)
            outl:NPFloatArray = e.get_outline(self.use_lagrangian_coordinates)*ss
            if tr is not None:
                outl,_=tr.apply(outl,None)
            outl=numpy.transpose(outl) #type:ignore
            last = outl[-1]
            for l in outl:
                allines += [[tuple(last), tuple(l)]]
                last = l
        lc = collections.LineCollection(allines, colors=self.linecolor, linewidths=self.linewidths,zorder=self.zindex) #type:ignore
        plt.gca().add_collection(lc) #type:ignore

        if isinstance(self.plotter.eigenvector,int):
                mesh.get_problem(self.problem_name).set_all_values_at_current_time(backup_dofs, backup_pinned,False) #type:ignore


class MatplotLibOverlayBase(MatplotLibPart):
    """
    A base class for any kind of plot object like text, colorbars, etc. that is to be overlaid on a plot.
    """
    #: Index of the overlay. The higher the index, the more on top of the plot it will be.
    zindex = 10.0
    #: Order when to preprocess the overlay. The higher the number, the later it will be processed.
    preprocess_order = 10.0
    #: Position of the overlay. Can be a tuple/list of floats as coordinates, or a string like ``"top left"``, ``"bottom right"``, ``"center"``, etc.
    position:Optional[Union[Tuple[float,float],List[float],str]] = None
    #: Margin in x direction
    xmargin = 0.025
    #: Margin in y direction
    ymargin = 0.05
    weight="bold"
    #: Text color
    textcolor="black"
    #: Text size
    textsize=12.0
    horizontalalign=None
    verticalalign=None
    #: Additional shift in x-direction
    xshift=0.0
    #: Additional shift in y-direction
    yshift=0.0

    def _map_x(self,x:float)->float:
        return self.plotter.xmin + x * (self.plotter.xmax - self.plotter.xmin) #type:ignore

    def _map_y(self,y:float)->float:
        return self.plotter.ymin + y * (self.plotter.ymax - self.plotter.ymin) #type:ignore

    def __init__(self,plotter:"MatplotlibPlotter"):
        super().__init__(plotter)
        self.xpos:Optional[float] = None
        self.ypos:Optional[float] = None

    def get_overlay_size(self):
        return [0.0,0.0]

    def check_pos(self):
        if self.ypos is None:
            #Auto determine position
            if isinstance(self.position,(list,tuple)):
                self.ypos=self.position[1]
            elif self.position is not None:
                if self.position.find("lower")>=0 or self.position.find("bottom")>=0:
                    self.ypos=self.ymargin
                    if self.verticalalign is None:
                        self.verticalalign="bottom"
                elif self.position.find("upper")>=0 or self.position.find("top")>=0:
                    self.ypos = 1-self.ymargin-self.get_overlay_size()[1]
                    if self.verticalalign is None:
                        self.verticalalign="top"
        if self.xpos is None:
            if isinstance(self.position,(list,tuple)):
                self.xpos=self.position[0]
            elif self.position is not None:
                if self.position.find("left")>=0:
                    self.xpos=self.xmargin
                    if self.horizontalalign is None:
                        self.horizontalalign="left"
                elif self.position.find("right")>=0:
                    self.xpos = 1-self.xmargin-self.get_overlay_size()[0]
                    if self.horizontalalign is None:
                        self.horizontalalign="right"

        #Try once more for center
        if isinstance(self.position,str):
            if self.xpos is None:
                if self.position is not None:
                    if self.position.find("center")>=0:
                        self.xpos=0.5-self.get_overlay_size()[0]*0.5
                        if self.horizontalalign is None:
                            self.horizontalalign="center"
            if self.ypos is None:
                if self.position is not None:
                    if self.position.find("center")>=0:
                        self.ypos=0.5-self.get_overlay_size()[1]*0.5
                        if self.verticalalign is None:
                            self.verticalalign="center"

        if self.xpos is None or self.ypos is None:
            raise RuntimeError("Please either set '"+str(self.mode)+"' xpos and ypos or use position='lower left' or similar")

    def pre_process(self):
        super(MatplotLibOverlayBase, self).pre_process()
        self.check_pos()

class MatplotLibBaseRange:
    def __init__(self) -> None:
        self.vmin:Optional[float]
        self.vmax:Optional[float]

    def consider_range(self,vmin:float,vmax:float):
        pass

    def reset_range(self):
        pass

#An object which allows you to store the range of e.g. colorbars over multiple timesteps
class MatplotLibPersistentRange(MatplotLibBaseRange):
    def __init__(self,vmin:Optional[float],vmax:Optional[float],mode:Union[str,Union[Tuple[float,float],List[float]]]="current"):
        super(MatplotLibPersistentRange, self).__init__()        
        self.vmin=vmin
        self.vmax=vmax
        self.mode=mode

    def reset_range(self):
        if self.mode=="current":
            self.vmin=None
            self.vmax=None
        elif isinstance(self.mode,(tuple,list,)):
            self.vmin=self.mode[0]
            self.vmax=self.mode[1]

    def consider_range(self,vmin:float,vmax:float):
        if vmin>vmax:
            vmin,vmax=vmax,vmin
        if self.mode=="current":
            self.vmin=vmin
            self.vmax=vmax
        elif self.mode=="fixed":
            delta=vmax-vmin
            if delta>1e-24:
                if self.vmin is None:
                    self.vmin=vmin
                if self.vmax is None:
                    self.vmax=vmax




@MatplotLibPart.register()
class MatplotLibColorbar(MatplotLibOverlayBase):
    """
    A color bar to be used in plots. Create it with :py:meth:`MatplotlibPlotter.add_colorbar`.
    """
    mode="colorbar"
    #: Number of discrete colors to use. If None, use a continuous color map.
    Ndisc:Optional[int] = 20
    cmap:Union[str,matplotlib.colors.Colormap] = "coolwarm"
    #: A norm (matplotlib.norm) to use
    norm = None
    #: Length of the colorbar in nondimensional figure coordinates
    length = 0.4
    #: Thickness of the colorbar in nondimensional figure coordinates
    thickness = 0.05
    ticks=None
    #: Tick size of the colorbar
    ticsize=10
    labelpad=4
    #: Hide each second tick
    hide_some_ticks:Optional[Union[bool,Sequence[int]]]=True
    #: Hide all ticks
    hide_all_ticks:bool=False
    #: Orientation of the colorbar
    orientation="horizontal"
    #: A factor to multiply the data with. Useful to convert units
    factor:float=1.0
    #: An offset to add to the data, useful to plot e.g. temperature in kelvin
    offset:float=0.0
    rangemode="current"
    extend="auto"
    unit:ExpressionNumOrNone=None
    useOffset=True
    scientific=True
    #: Symmetrize the colorbar range. Good for e.g. vorticity fields
    symmetrize_min_max=False

    def get_norm(self):
        if self.norm is None:
            assert self.range is not None
            self.norm=matplotlib.colors.Normalize(vmin=self.range.vmin, vmax=self.range.vmax)
        return self.norm

    def get_overlay_size(self):
        return [(self.length if self.orientation=="horizontal" else self.thickness),(self.length if self.orientation=="vertical" else self.thickness)]

    def __init__(self,plotter:"MatplotlibPlotter"):
        super().__init__(plotter)
        self._vmin:Optional[float]=None
        self._vmax:Optional[float]=None
        self._clamp_min:Optional[float]=None
        self._clamp_max:Optional[float]=None
        self.cb=None
        self.title="Colorbar"
        self.range:Optional["MatplotLibBaseRange"]=None
        #: If True, the colorbar will not be plotted
        self.invisible=False

    def consider_range(self,vmin:Optional[float]=None,vmax:Optional[float]=None):
        if vmin is not None:
            if self._vmin is not None:
                self._vmin=min(self._vmin,vmin)
            else:
                self._vmin=vmin
            if self.symmetrize_min_max and self._vmin<=0:
                if self._vmax is not None:
                    self._vmax=max(self._vmax,-self._vmin)
                else:
                    self._vmax=-self._vmin
                
        if vmax is not None:
            if self._vmax is not None:
                self._vmax=max(self._vmax,vmax)
            else:
                self._vmax=vmax        
            if self.symmetrize_min_max and self._vmax>=0:
                if self._vmin is not None:
                    self._vmin=min(self._vmin,-self._vmax)
                else:
                    self._vmin=-self._vmax
                    
        if self._clamp_min is not None:
            if self._vmin is not None:
                self._vmin=max(self._vmin,self._clamp_min)
        if self._clamp_max is not None:
            if self._vmax is not None:
                self._vmax=min(self._vmax,self._clamp_max)

    def discrete_cmap(self, N:Union[int,Sequence[float]], base_cmap:Optional[str]=None)->matplotlib.colors.LinearSegmentedColormap:
        base = matplotlib.pyplot.get_cmap(base_cmap) #type:ignore
        if isinstance(N,int):
            N=numpy.linspace(0, 1, N) #type:ignore
            cmap_name = base.name + str(N) #type:ignore
        else:
            cmap_name = base.name + str(len(N)) #type:ignore
        color_list = base(N) #type:ignore

        return matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, color_list, len(N)) #type:ignore

    def set_kwargs(self,kwargs:Dict[str,Any]):
        super(MatplotLibColorbar, self).set_kwargs(kwargs)
        if self.unit is not None:
            ustr,_num,factor=unit_to_string(self.unit,estimate_prefix=True)
            if ustr!="":
                self.factor*=factor
                self.title=self.title+" ["+ustr+"]"

    def pre_process(self):
        super(MatplotLibColorbar, self).pre_process()
        if isinstance(self.cmap,str):
            if (self.Ndisc is not None) and (isinstance(self.Ndisc,(tuple,list,numpy.ndarray)) or self.Ndisc > 0):
                self.cmap = self.discrete_cmap(self.Ndisc, plt.get_cmap(self.cmap)) #type:ignore
            else:
                self.cmap=plt.get_cmap(self.cmap) #type:ignore
        if self.range is None:
            self.range=self.plotter.get_range_object(self,mode=self.rangemode)
        if self._vmin is not None and self._vmax is not None:
            self.range.consider_range(self._vmin,self._vmax)


    def add_to_plot(self):
        if self.invisible:
            return
        if self.range is None or (self.range.vmin is None and self.range.vmax is None):
            return
        current_ax = plt.gcf().gca()
        assert self.xpos is not None and self.ypos is not None
        cax = plt.gcf().add_axes([self.xpos+self.xshift, self.ypos+self.yshift, self.length, self.thickness]) #type:ignore
        scalarMap = cm.ScalarMappable(norm=self.get_norm(), cmap=self.cmap)
        
        scalarMap.set_array([self.range.vmin, self.range.vmax]) #type:ignore
        kw:Dict[str,str]={}
        if self.orientation=="horizontal":
            kw["orientation"]="horizontal"
        elif self.orientation=="vertical":
            kw["orientation"]="vertical"
        if self.extend=="auto":
            if self._vmin is not None and self._vmax is not None and self.range.vmin is not None and self.range.vmax is not None:
                mi=self._vmin<self.range.vmin
                ma=self._vmax>self.range.vmax
            else:
                mi=False
                ma=False
            if mi and ma:
                kw["extend"] = "both"
            elif mi:
                kw["extend"] = "min"
            elif ma:
                kw["extend"] = "max"
            else:
                kw["extend"] = "neither"

        else:
            kw["extend"] = self.extend

        cb:matplotlib.colorbar.Colorbar = plt.gcf().colorbar(scalarMap, cax=cax, ticks=self.ticks, **kw) #type:ignore
        if hasattr(cb.formatter,"set_useOffset"): #type:ignore
            cb.formatter.set_useOffset(False) #type:ignore
        cb.update_ticks()
        self.cb = cb
        cb.set_label(self.title, size=self.textsize, weight=self.weight,labelpad=self.labelpad, color=self.textcolor) #type:ignore
        if hasattr(cb.formatter,"set_useOffset"): #type:ignore
            cb.formatter.set_useOffset(self.useOffset) #type:ignore
        if hasattr(cb.formatter, "set_scientific"): 
            cb.formatter.set_scientific(self.scientific) #type:ignore
        cb.ax.tick_params(labelsize=self.ticsize, color=self.textcolor) #type:ignore
        cbytick_obj = plt.getp(cb.ax.axes, 'xticklabels') #type:ignore
        plt.setp(cbytick_obj, color=self.textcolor) #type:ignore
        if self.hide_some_ticks is True:
            for label in cb.ax.xaxis.get_ticklabels()[1::2]:
                label.set_visible(False)
        elif (self.hide_some_ticks is not None) and (self.hide_some_ticks!=False):
            hinfo = self.hide_some_ticks
            for label in cb.ax.xaxis.get_ticklabels():
                label.set_visible(False)
            for label in cb.ax.xaxis.get_ticklabels()[hinfo[0]::hinfo[1]]:
                label.set_visible(True)
        if self.hide_all_ticks:
            cb.ax.xaxis.set_ticklabels([]) #type:ignore
            cb.ax.xaxis.set_ticks([]) #type:ignore

        if cb.orientation == 'vertical': #type:ignore
            #long_axis, short_axis = cb.ax.yaxis, cb.ax.xaxis
            pass
        else:
            #long_axis, short_axis = cb.ax.xaxis, cb.ax.yaxis
            cb.ax.xaxis.set_label_position('top')

        plt.gcf().sca(current_ax)


@MatplotLibPart.register()
class MatplotlibText(MatplotLibOverlayBase):
    """
    A text overlay to be used in plots. Create it with :py:meth:`MatplotlibPlotter.add_text`.
    """
    mode="text"
    zindex = 10
    bbox = None
    weight = "normal"
    color="black"


    def __init__(self,plotter:"MatplotlibPlotter"):
        super(MatplotlibText, self).__init__(plotter)
        self.text=None

    def add_to_plot(self):
        plt.gcf().text(self.xpos+self.xshift,self.ypos+self.yshift,self.text,fontsize=self.textsize,fontweight=self.weight,va=self.verticalalign or "center",ha=self.horizontalalign or "center",bbox=self.bbox,color=self.color) #type:ignore



@MatplotLibPart.register()
class MatplotlibTimeLabel(MatplotlibText):
    mode="timelabel"
    format="{:04.4f}"
    unit="auto"
    bbox = dict(boxstyle='round', facecolor='wheat', alpha=1)
    textsize = 18
    useTeX=False

    def pre_process(self):
        super(MatplotlibTimeLabel, self).pre_process()
        unit=1
        unitstr=""
        if self.unit=="auto":
            ts:ExpressionOrNum=self.plotter.get_problem().get_scaling("temporal") #type:ignore
            if isinstance(ts,float) or isinstance(ts,int):
                unit=1
                unitstr=""
            else:
                try:
                    float(ts)
                    unitstr = ""
                except:

                    unitstr=" s"
        else:
            if self.unit=="s":
                unit=1
                unitstr=" s"
            elif self.unit=="ms":
                unit = 1000
                unitstr = " ms"
            elif self.unit=="us":
                unit = 1000*1000
                if self.useTeX:
                    unitstr = r" \mu s"
                else:
                    unitstr = " us"
            elif self.unit=="ns":
                unit = 1000*1000*1000
                unitstr = " ns"                
            else:
                raise RuntimeError("TODO "+str(self.unit))
        if self.useTeX:
            self.text=self.format.format(unit*self.plotter.get_problem().get_current_time(as_float=True))+r"$\ \mathrm{"+unitstr+r"}$" #type:ignore
        else:
            self.text=self.format.format(unit*self.plotter.get_problem().get_current_time(as_float=True))+unitstr #type:ignore
        

@MatplotLibPart.register()
class MatplotLibScaleBar(MatplotLibOverlayBase):
    mode="scalebar"
    minlength=0.05
    maxlength=0.1
    stoplength=0.02
    linewidths=3
    textsize = 14
    text_yoffset=0
    orientation="horizontal"
    invisible=False

    def _fit_length(self,scale:float):
        maxrange = self.maxlength / scale
        minrange = self.minlength / scale
        oom = numpy.floor(numpy.log10(0.5 * (maxrange + minrange)))  # Possible order of magnitude
        factors = [1.0, 0.5, 0.25, 0.75, 0.2, 0.4, 0.6, 0.8, 0.3, 0.9, 0.7, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75,
                   0.85, 0.95]
        reallength = numpy.power(10, oom)
        for pref in factors:
            checkme = pref * numpy.power(10, oom)
            while checkme < minrange: checkme = checkme * 10
            while checkme > maxrange: checkme = checkme / 10
            if checkme >= minrange:
                reallength = checkme
                break
        return reallength,reallength*scale

    def add_to_plot(self):
        if self.invisible:
            return
        assert self.plotter.xmin is not None and self.plotter.xmax is not None
        assert self.plotter.ymin is not None and self.plotter.ymax is not None
        reallength,figlength=self._fit_length(1/(self.plotter.xmax-self.plotter.xmin) if self.orientation!="vertical" else 1/(self.plotter.ymax-self.plotter.ymin))
        shift = 0.5
        if self.orientation=="vertical":
            nx,ny= 0,1
        else:
            nx, ny = 1, 0
        map_x:Callable[[float],float]=lambda x : self._map_x(x+self.xshift)
        map_y:Callable[[float],float] = lambda y: self._map_y(y+self.yshift)
        x,y,l=self.xpos,self.ypos,figlength
        plt.gca().annotate("", xytext=(map_x(x - (1.0 - shift) * l * nx), map_y(y - (1.0 - shift) * l * ny)), #type:ignore
                         xy=(map_x(x + shift * l * nx), map_y(y + shift * l * ny)),
                         arrowprops=dict(color=self.textcolor, fc=self.textcolor,arrowstyle="-", lw=self.linewidths),zorder=self.zindex)
        # Bars at the end
        if self.stoplength>0:
            bl = self.stoplength #/TODO: Scale by x/y
            plt.gca().annotate("", xytext=(map_x(x - (1.0 - shift) * l * nx + bl * ny), map_y(y - (1.0 - shift) * l * ny + bl * nx)), #type:ignore
                             xy=(map_x(x - (1.0 - shift) * l * nx - bl * ny), map_y(y - (1.0 - shift) * l * ny - bl * nx)),
                             arrowprops=dict(color=self.textcolor, fc=self.textcolor,arrowstyle="-", lw=self.linewidths),zorder=self.zindex)
            plt.gca().annotate("", xytext=(map_x(x + (shift) * l * nx + bl * ny), map_y(y + (shift) * l * ny + bl * nx)), #type:ignore
                             xy=(map_x(x + (shift) * l * nx - bl * ny), map_y(y + (shift) * l * ny - bl * nx)),
                             arrowprops=dict(color=self.textcolor, fc=self.textcolor,arrowstyle="-", lw=self.linewidths),zorder=self.zindex)

        lstr=str(reallength)
        ss=self.plotter.get_problem().get_scaling("spatial") #type:ignore
        if (not isinstance(ss,int)) and (not isinstance(ss,float)):
            #Assuming meter
            lstr = "{:.8g} m".format(reallength)
            if reallength < 1e-6:
                lstr = "{:.8g} nm".format(reallength * 1000 * 1000 * 1000)
            elif reallength < 1e-4:
                lstr = "{:.8g} um".format(reallength * 1000 * 1000)
            elif reallength < 1e-2:
                lstr = "{:.8g} mm".format(reallength * 1000)
            elif reallength < 1e-1:
                lstr = "{:.8g} cm".format(reallength * (100))
        plt.gca().text(map_x(x), map_y(y+self.text_yoffset),lstr, fontsize=self.textsize,color=self.textcolor,va="bottom",ha="center",zorder=self.zindex) #type:ignore





@MatplotLibPart.register()
class MatplotLibArrowKey(MatplotLibOverlayBase):
    """
    An arrow key to be used in plots. Create it with :py:meth:`MatplotlibPlotter.add_arrow_key`.
    """
    mode="arrowkey"
    minlength_key=0.1
    maxlength_key=0.15
    maxlength=0.2
    textoffset=0.02
    multi_title_offset=0.05
    factor=1
    invisible=False
    orientation="horizontal"
    preprocess_order = 1000 # Make it late so that vmax is set
    linecolor="lawngreen"
    linewidths=2
    arrowstyle="->"
    arrowprops=None
    format="{:04.4f}"
    rangemode="fixed"
    unit=None
    textsize2 = None
    weight2=None

    def __init__(self,plotter:"MatplotlibPlotter"):
        super(MatplotLibArrowKey, self).__init__(plotter)
        self.figlength:Optional[float]=None
        self.reallength:Optional[float]=None
        self.title:Optional[Union[str,Dict[str,str]]]=None
        self.range:Optional[MatplotLibBaseRange]=None
        self._vmax:float=0
        self.spatial_scale:float=0
        self.arrow_length_scale:float=0.0

    def set_kwargs(self, kwargs:Dict[str,Any]):
        super(MatplotLibArrowKey, self).set_kwargs(kwargs)
        if self.unit is not None:
            ustr, _num, factor = unit_to_string(self.unit, estimate_prefix=True)
            print("UNIT",self.unit,ustr,factor,self.factor)
            #exit()
            if ustr != "":
                #self.factor *= factor
                self.format = self.format + " [" + ustr + "]"

    def consider_range(self,vm:float):
        self._vmax=max(vm,self._vmax)


    def _fit_length(self,scale:float):
        maxrange = self.maxlength_key / scale
        minrange = self.minlength_key / scale
        oom = numpy.floor(numpy.log10(0.5 * (maxrange + minrange)))  # Possible order of magnitude
        factors = [1.0, 0.5, 0.25, 0.75, 0.2, 0.4, 0.6, 0.8, 0.3, 0.9, 0.7, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75,
                   0.85, 0.95]
        reallength = numpy.power(10, oom)
        for pref in factors:
            checkme = pref * numpy.power(10, oom)
            while checkme < minrange: checkme = checkme * 10
            while checkme > maxrange: checkme = checkme / 10
            if checkme >= minrange:
                reallength = checkme
                break
        return reallength,reallength*scale

    def set_range(self,vmax:float):
        if self.range is None:
            self.range=self.plotter.get_range_object(self,mode=self.rangemode)
        self.range.consider_range(0, vmax)
        if self.figlength is None and (self.range.vmax is not None):
            self.spatial_scale:float=min(self.plotter.xmax - self.plotter.xmin,self.plotter.ymax - self.plotter.ymin) #type:ignore
            self.arrow_length_scale=self.maxlength/self.range.vmax*self.spatial_scale
            #reallength, figlength = self._fit_length( (1 / (self.plotter.xmax - self.plotter.xmin) if self.orientation != "vertical" else 1 / (self.plotter.ymax - self.plotter.ymin))self.vmax)
            reallength, figlength = self._fit_length(self.arrow_length_scale/self.spatial_scale) 
            self.figlength=figlength
            self.reallength=reallength

    def pre_process(self):
        super(MatplotLibArrowKey, self).pre_process()
        if self.range is None:
            self.range=self.plotter.get_range_object(self,mode=self.rangemode)
        self.range.consider_range(0,self._vmax)
        if self.figlength is None and (self.range.vmax is not None):
            self.spatial_scale:float=min(self.plotter.xmax - self.plotter.xmin,self.plotter.ymax - self.plotter.ymin) #type:ignore
            if abs(self.range.vmax)<1e-20:
                self.arrow_length_scale=self.maxlength*self.spatial_scale
            else:                
                self.arrow_length_scale=self.maxlength/self.range.vmax*self.spatial_scale
            #reallength, figlength = self._fit_length( (1 / (self.plotter.xmax - self.plotter.xmin) if self.orientation != "vertical" else 1 / (self.plotter.ymax - self.plotter.ymin))self.vmax)
            reallength, figlength = self._fit_length(self.arrow_length_scale/self.spatial_scale)
            self.figlength=figlength
            self.reallength=reallength

    def add_to_plot(self):
        assert self.range is not None
        if (self.range.vmax is None):
            return
        map_x:Callable[[float],float] = lambda x: self._map_x(x)+self.xshift
        map_y:Callable[[float],float] = lambda y: self._map_y(y)+self.yshift
        shift=0.5
        nx,ny=1,0
        assert self.reallength is not None
        assert self.maxlength_key is not None
        assert self.xpos is not None and self.ypos is not None
        x, y, l = self.xpos, self.ypos, self.reallength*self.arrow_length_scale
        arrp=self.arrowprops
        if arrp is None:
            arrp=dict(color=self.linecolor, fc=self.linecolor, arrowstyle=self.arrowstyle, lw=self.linewidths)
        if self.orientation=="vertical":
            raise RuntimeError("TODO")
        else:
            if self.horizontalalign=="left":
                x+=self.maxlength_key*0.5
            elif self.horizontalalign=="right":
                x -= self.maxlength_key * 0.5
        if not self.invisible:
            if isinstance(self.title,dict):
                offs=0
                for tit,col in self.title.items():
                    plt.gca().annotate("", #type:ignore
                                       xytext=(map_x(x) - (1.0 - shift) * l * nx, map_y(y-offs*self.multi_title_offset) - (1.0 - shift) * l * ny),
                                       xy=(map_x(x) + shift * l * nx, map_y(y-offs*self.multi_title_offset) + shift * l * ny),
                                       arrowprops=dict(color=col, fc=col,
                                                       arrowstyle=self.arrowstyle, lw=self.linewidths),
                                       zorder=self.zindex)
                    plt.gca().text(map_x(x-self.textoffset)- (1.0 - shift) * l * nx, map_y(y-offs*self.multi_title_offset), tit, color=self.textcolor, ha="right", #type:ignore
                                   va="center", size=self.textsize, weight=self.weight, zorder=self.zindex)
                    offs=offs+1
                plt.gca().text(map_x(x), map_y(y - (offs-1)*self.multi_title_offset-self.textoffset), #type:ignore
                                   self.format.format(self.reallength * self.factor), color=self.textcolor, ha="center",
                                   va="top", size=self.textsize2 or self.textsize, weight=self.weight2 or self.weight,
                                   zorder=self.zindex)

            else:
                plt.gca().annotate("", xytext=(map_x(x)- (1.0 - shift) * l * nx, map_y(y) - (1.0 - shift) * l * ny), #type:ignore
                                   xy=(map_x(x) + shift * l * nx, map_y(y) + shift * l * ny),
                                   arrowprops=arrp,zorder=self.zindex)
                if self.title is not None:
                    plt.gca().text(map_x(x), map_y(y+self.textoffset), self.title,color=self.textcolor,ha="center",va="bottom",size=self.textsize, weight=self.weight,zorder=self.zindex) #type:ignore
                plt.gca().text(map_x(x), map_y(y - self.textoffset), self.format.format(self.reallength*self.factor), color=self.textcolor, ha="center", va="top",size=self.textsize2 or self.textsize, weight=self.weight2 or self.weight,zorder=self.zindex) #type:ignore


@MatplotLibPart.register()
class MatplotLibAxes(MatplotLibOverlayBase):
    mode = "axes"
    Ndisc = 20
    width = 0.4
    height = 0.3
    title:Optional[str] = None
    xlabel:Optional[str] = None
    ylabel:Optional[str] = None
    y2label:Optional[str] = None
    ylabel_color:str="black"
    y2label_color:str="black"
    xmin=None
    xmax=None
    rangemode_x="auto"
    ymin = None
    ymax = None
    y2min=None
    y2max=None
    rangemode_y = "auto"
    rangemode_y2 = "auto"
    alpha=1
    ticksize=None    
    legend_position:Optional[str]=None
    hide_y_ticks=False

    def consider_range(self,xmin:Optional[float]=None,xmax:Optional[float]=None,ymin:Optional[float]=None,ymax:Optional[float]=None,use_y2=False):
        if self.rangemode_x=="auto":
            if self.xmin is None:
                self.xmin=xmin
            elif xmin is not None:
                self.xmin=min(self.xmin,xmin)
            if self.xmax is None:
                self.xmax=xmax
            elif xmax is not None:
                self.xmax=max(self.xmax,xmax)
        if not use_y2:
            if self.rangemode_y=="auto":
                if self.ymin is None:
                    self.ymin=ymin
                elif ymin is not None:
                    self.ymin=min(self.ymin,ymin)
                if self.ymax is None:
                    self.ymax=ymax
                elif ymax is not None:
                    self.ymax=max(self.ymax,ymax)
        else:
            if self.rangemode_y2=="auto":
                if self.y2min is None:
                    self.y2min=ymin
                elif ymin is not None:
                    self.y2min=min(self.y2min,ymin)
                if self.y2max is None:
                    self.y2max=ymax
                elif ymax is not None:
                    self.y2max=max(self.y2max,ymax)

    def get_overlay_size(self):
        return [self.width, self.height]

    def __init__(self, plotter:"MatplotlibPlotter"):
        super(MatplotLibAxes, self).__init__(plotter)
        self.ax = None
        self.ax_y2=None
        self.invisible = False
        self.xfactor=1.0
        self.yfactor=1.0
        self._linelist=[]

    def add_to_plot(self):
        if self.invisible:
            return
        current_ax = plt.gcf().gca()
        self.ax = plt.gcf().add_axes([self.xpos + self.xshift, self.ypos + self.yshift, self.width, self.height]) #type:ignore

        if self.title is not None:
            self.ax.set_title(self.title)

        if self.xlabel is not None:
            self.ax.set_xlabel(self.xlabel,size=self.textsize) #type:ignore
        if self.ylabel is not None:
            self.ax.set_ylabel(self.ylabel,size=self.textsize,color=self.ylabel_color) #type:ignore
        if self.y2label is not None:
            if self.ax_y2 is None:
                self.ax_y2=self.ax.twinx()
                self.ax_y2.set_ylabel(self.y2label,size=self.textsize,color=self.y2label_color) #type:ignore

        if self.xmin is not None and self.xmax is not None:
            self.ax.set_xlim(self.xmin,self.xmax)
        if self.ymin is not None and self.ymax is not None:
            self.ax.set_ylim(self.ymin,self.ymax)
        if self.ax_y2 is not None and self.y2min is not None and self.y2max is not None:
            self.ax_y2.set_ylim(self.y2min,self.y2max)
        if self.ticksize is not None:
            self.ax.tick_params(axis='both', which='major', labelsize=self.ticksize) #type:ignore
        self.ax.patch.set_alpha(self.alpha) #type:ignore

        if self.hide_y_ticks:
            self.ax.tick_params(axis='y',which='both',left=False, right=False,  labelleft=False)

        plt.gcf().sca(current_ax)
    
    def post_process(self):
        super().post_process()
        if self.legend_position:
            lls=[]
            labs=[]
            for l in self._linelist:
                lab=l.get_label()                
                if lab and not lab.startswith("_"):
                    lls.append(l)
                    labs.append(lab)
            self.ax.legend(lls,labs, loc=self.legend_position)


@MatplotLibPart.register()
class MatplotLibLinePlot(MatplotLibPartWithMeshData):
    mode = "lineplot"
    xfactor:float=1
    yfactor:float=1
    linewidth:float=1
    linestyle:str="-"
    color:Optional[str]=None
    use_y2=False
    label:Optional[str]=None
    markersize:Optional[float]=None
    markerstyle:Optional[str]=None
    sort_by_x=True

    def __init__(self, plotter:"MatplotlibPlotter"):
        super(MatplotLibLinePlot, self).__init__(plotter)
        self.axes:Optional[MatplotLibAxes] = None
        self._coordinates=None
        self._data=None
        self._plotdata=None
        self._external_xdata:Optional[NPFloatArray]=None
        self._external_ydata:Optional[NPFloatArray]=None

    def set_external_data(self,x:NPFloatArray,y:NPFloatArray):
        self._external_xdata=x
        self._external_ydata=y

    def pre_process(self):
        assert self.axes is not None
        if self.axes.ax is None:
            self.axes.pre_process()
        if self.zindex<=self.axes.zindex:
            self.zindex=self.axes.zindex+1
        if self._external_xdata is not None and self._external_ydata is not None:
            xdata:NPFloatArray=self._external_xdata
            ydata:NPFloatArray=self._external_ydata
        else:
            self._coordinates=self.mshcache.get_coordinates(lagrangian=self.use_lagrangian_coordinates)
            self._data=self.mshcache.get_data(self.field)
            assert self._data is not None
            xdata:NPFloatArray = self._coordinates[0]
            ydata:NPFloatArray = self._data

        xdata*=self.axes.xfactor*self.xfactor
        ydata *= self.axes.yfactor*self.yfactor
        if self.sort_by_x:
            srt:NPIntArray = numpy.argsort(xdata) #type:ignore
            xdata,ydata=xdata[srt],ydata[srt]
        self._plotdata=[xdata, ydata]
        xmin:float=numpy.amin(xdata) #type:ignore
        xmax:float=numpy.amax(xdata) #type:ignore        
        self.axes.consider_range(xmin=xmin,xmax=xmax)
        if self.axes.rangemode_x=="fixed":
            inds= numpy.logical_and(self.axes.xmin <= xdata, xdata <= self.axes.xmax) #type:ignore
            self.axes.consider_range(ymin=numpy.amin(ydata[inds]), ymax=numpy.amax(ydata[inds]),use_y2=self.use_y2) #type:ignore
        else:
            self.axes.consider_range(ymin=numpy.amin(ydata),ymax=numpy.amax(ydata),use_y2=self.use_y2) #type:ignore

    def add_to_plot(self):
        kwargs={"linewidth":self.linewidth,"linestyle":self.linestyle}
        if self.color:
            kwargs["color"]=self.color        
        if self.label:
            kwargs["label"]=self.label
        if self.markerstyle:
            kwargs["marker"]=self.markerstyle
        if self.markersize:
            kwargs["markersize"]=self.markersize
        if self.use_y2:
            l=self.axes.ax_y2.plot(self._plotdata[0],self._plotdata[1],**kwargs) #type:ignore
        else:
            l=self.axes.ax.plot(self._plotdata[0],self._plotdata[1],**kwargs) #type:ignore
        self.axes._linelist+=l


@MatplotLibPart.register()
class MatplotLibPolygon(MatplotLibPart):
    mode="polygon"
    zindex=-10
    facecolor=None
    edgecolor="black"
    linewidth=1
    fill=True
    alpha=None
    def __init__(self,plotter:"MatplotlibPlotter"):
        super(MatplotLibPolygon, self).__init__(plotter)
        self.points:Sequence[float]=[]

    def add_to_plot(self):
        if len(self.points)<3:
            return
        poly = Polygon(self.points, facecolor=self.facecolor, edgecolor=self.edgecolor,linewidth=self.linewidth,fill=self.fill,alpha=self.alpha)
        poly.zorder=self.zindex
        plt.gca().add_patch(poly) #type:ignore


@MatplotLibPart.register()
class MatplotLibTracers(MatplotLibPart):
    mode="tracers"
    marker="o"
    size=4
    color:Optional[str]=None
    edgecolor="face"
    zindex=5
    invisible=False


    def __init__(self,plotter:"MatplotlibPlotter"):
        super(MatplotLibTracers, self).__init__(plotter)
        self.tracer_name:Optional[str] = None
        self.transform:Optional[PlotTransform]=None
        self.mesh=None

    def set_tracer_data(self,name:str,mesh:AnySpatialMesh,transform:Optional[PlotTransform]):
        self.tracer_name=name
        self.mesh=mesh
        self.transform=transform
        
    def pre_process(self):
        super(MatplotLibTracers, self).pre_process()



    def add_to_plot(self):
        assert self.tracer_name is not None
        assert self.mesh is not None
        col=self.mesh.get_tracers(self.tracer_name)
        assert col is not None
        pos=col.get_positions()
        ss = self.mesh.get_output_scale("spatial")
        pos*=ss
        if self.transform is not None:
            pos, _ = self.transform.apply(numpy.transpose(pos), None) #type:ignore
            pos=numpy.transpose(pos) #type:ignore


        if len(pos)==0: #type:ignore
            return
        if len(pos.shape)!=2:
            raise RuntimeError("Strange tracer position array "+str(pos)) #type:ignore
        if pos.shape[0]==0:
            return
        if pos.shape[1]!=2:
            raise RuntimeError("Can only plot tracers on 2d meshes")
        if not self.invisible:
            plt.gca().scatter(pos[:,0],pos[:,1],marker=self.marker,s=self.size,c=self.color, edgecolor=self.edgecolor,zorder=self.zindex) #type:ignore



@MatplotLibPart.register()
class MatplotLibImage(MatplotLibOverlayBase):
    mode="image"
    image:Optional[str]=None
    pixel_per_spatial_unit=None
    cmap='gray'
    verticalalign="center"
    horizontalalign = "center"


    def pre_process(self):
        self.check_pos()
        pass

    def add_to_plot(self):
        assert self.image is not None
        imgdata=mpimg.imread(self.image) #type:ignore
        if self.pixel_per_spatial_unit is not None:
            sizex=imgdata.shape[1]/self.pixel_per_spatial_unit
            sizey = imgdata.shape[0]/self.pixel_per_spatial_unit
            self.position
            cornerx=self._map_x(self.xpos)+self.xshift #type:ignore
            cornery=self._map_y(self.ypos)+self.yshift #type:ignore

            #print(self.xpos,self.ypos)
            #exit()
            if self.verticalalign=="center":
                cornery-=sizey/2
            elif self.verticalalign=="top":
                cornery -= sizey
            if self.horizontalalign=="center":
                cornerx-=sizex/2
            elif self.horizontalalign=="right":
                cornerx -= sizex

            extent=(cornerx,cornerx+sizex,cornery,cornery+sizey) #type:ignore
        else:
            extent=(self.plotter.xmin,self.plotter.xmax,self.plotter.ymin,self.plotter.ymax)
        plt.imshow(imgdata,extent=extent,cmap=self.cmap) #type:ignore



class MatplotlibPlotter(BasePlotter):
    """
    A class to invoke plotting of 2d problems using matplotlib.
    
    Args:
        problem: The problem to plot
        filetrunk: The trunk of the file name to save the plot to (without extension)
        fileext: The extension of the file to save the plot to
        eigenvector: If set, we plot eigenvector at the given index instead of the solution
        eigenmode: If eigenvector is set, this is the mode to plot ( ``"abs"``, ``"real"``, ``"imag"`` )
        add_eigen_to_mesh_positions: If eigenvector is set and we have a moving mesh, we can select whether we add the base mesh positions to the eigenvector of the mesh positions
        position_eigen_scale: If eigenvector is set and we have a moving mesh, we can scale the eigenvector to be added to the actual mesh positions by this factor
        eigenscale: If eigenvector is set, we can scale the eigenvector by this factor (this includes also the mesh positions)
        
    """
    def __init__(self,problem:Optional["Problem"]=None,filetrunk:str="plot_{:05d}",fileext:Union[str,List[str]]="png",eigenvector:Optional[int]=None,eigenmode:"MeshDataEigenModes"="abs",add_eigen_to_mesh_positions:bool=True,position_eigen_scale:float=1,eigenscale:float=1):
        super(MatplotlibPlotter, self).__init__(problem,eigenvector=eigenvector,eigenmode=eigenmode)
        self.xmin:Optional[float]=None
        self.xmax:Optional[float]=None
        self.ymin:Optional[float]=None
        self.ymax:Optional[float]=None
        self.aspect_ratio=True
        self.fullscreen=True
        #: A format string to save the plot to. The output step will be inserted into the format string
        self.file_trunk=filetrunk
        #: File extension to save the plot. Can also be a list for multiple simultaneous file formats
        self.file_ext=fileext
        #: Size of the plot in pixels
        self.image_size=[1280,720]
        #: DPI of the plot
        self.dpi:float=100
        self._added_parts:List[MatplotLibPart]=[]
        self._mode_to_class=MatplotLibPart.mode_to_class.copy()
        #: Set to change the background color of the plot
        self.background_color:Optional[str]=None
        self._range_objects:Dict[str,MatplotLibBaseRange]={}
        self.write_cb_range_files:bool=True
        self.load_cb_ranges_dir:str=""
        #: Stop the execution if an invalid triangulation is detected. Otherwise, the plot will be just skipped
        self.crash_on_invalid_triangulation:bool=True
        self.min_triangle_circle_ratio:Optional[float]=None
        self.min_triangle_area:Optional[float]=None
        #: Merge points that are multiple times in the mesh. Note that it does not average the data on these points
        self.merge_duplicate_points:bool=False
        self._has_invalid_triangulation:bool=False
        self.add_eigen_to_mesh_positions=add_eigen_to_mesh_positions
        self.position_eigen_scale=position_eigen_scale
        self.eigenscale=eigenscale
        self._output_dir="_plots"
        self._eigenfactor_right=None # Optional complex values to scale the eigenvector for the eigendynamics animation
        self._eigenfactor_left=None 
        self._eigenvector_for_animation=None

    
    def useLaTeXFont(self):
        plt.rcParams.update({ #type:ignore
            "text.usetex": True,
            "font.family": "sans-serif",
            "text.latex.preamble": r"\usepackage{amsmath} \usepackage{txfonts} \usepackage{color}",  # for the align enivironment
            "font.sans-serif": ["Helvetica"]})

    def get_range_object(self,plotobject:Any,mode:Union[str,Union[Tuple[float,float],List[float]]]="current")->MatplotLibBaseRange:
        if isinstance(plotobject,str):
            key=plotobject
        else:
            key=cast(Union[str,Dict[str,str]],plotobject.title) #type:ignore
        if isinstance(key,dict):
            key=list(key.keys())[0]

        if self.load_cb_ranges_dir!="":
            fn=os.path.join(self.load_cb_ranges_dir,"cb_ranges_{:05d}.txt".format(self._output_step))
            try:
                f=open(fn,"r")
                data=json.load(f)
                if key in data.keys():
                    print("Using colorbar ranges for '"+key+"' from '"+fn+"'")
                    return MatplotLibPersistentRange(data[key][0],data[key][1],"fixed")
            except:
                print("Cannot load "+fn)
                pass

        if key in self._range_objects.keys():
            return self._range_objects[key]
        else:
            self._range_objects[key]=MatplotLibPersistentRange(None,None,mode)
            return self._range_objects[key]

    def save(self,fname:Optional[Union[str,List[str]]]=None):
        if self._has_invalid_triangulation:
            return
        if fname is None:
            pdir=os.path.join(self._problem.get_output_directory(),self._output_dir)
            os.makedirs(pdir,exist_ok=True)
            file_exts=self.file_ext
            if not isinstance(file_exts,(list,tuple,set)):
                file_exts=[file_exts]
            fname=[os.path.join(pdir,self.file_trunk.format(self._output_step)+"."+fe) for fe in file_exts]
        if not isinstance(fname,(list,tuple,set)):
            fname=[fname]
        for fn in fname:
            if self.background_color=="transparent":
                plt.savefig(fn,dpi=self.dpi,transparent=True) #type:ignore
            else:
                plt.savefig(fn,dpi=self.dpi,facecolor=self.background_color) #type:ignore
        if self.write_cb_range_files and self.load_cb_ranges_dir=="":
            os.makedirs(os.path.join(pdir,"_cb_ranges"),exist_ok=True)
            f=open(os.path.join(pdir,"_cb_ranges","cb_ranges_{:05d}.txt".format(self._output_step)),"w")
            
            #f.write("cb_ranges={}\n")
            odict:Dict[str,Tuple[float,float]]={}
            for nam,rang in self._range_objects.items():
                odict[nam]=(rang.vmin,rang.vmax)
                #f.write('cb_ranges["'+nam+'"]=['+str(rang.vmin)+', '+str(rang.vmax)+']\n')                

            json.dump(odict,f)
                



    def _get_mesh_data(self,msh:Union[str,AnySpatialMesh],problem_name:str="",ignore_eigenfactors:bool=False,mirror_x:bool=False):
        
        if ignore_eigenfactors or (self._eigenfactor_right is None or self._eigenfactor_left is None or self._eigenvector_for_animation is None):            
            return self.get_problem(problem_name=problem_name).get_cached_mesh_data(msh,nondimensional=False,tesselate_tri=True,eigenvector=self.eigenvector,eigenmode=self.eigenmode,add_eigen_to_mesh_positions=self.add_eigen_to_mesh_positions)
        else:
            self.get_problem(problem_name=problem_name).invalidate_cached_mesh_data()
            olddofs,_=self.get_problem(problem_name=problem_name).get_current_dofs()
            ef=self._eigenfactor_left if mirror_x else self._eigenfactor_right
            self.get_problem(problem_name=problem_name).set_current_dofs(olddofs+numpy.real(ef*self._eigenvector_for_animation))
            res=self.get_problem(problem_name=problem_name).get_cached_mesh_data(msh,nondimensional=False,tesselate_tri=True,eigenvector=self.eigenvector,eigenmode=self.eigenmode,add_eigen_to_mesh_positions=self.add_eigen_to_mesh_positions)
            self.get_problem(problem_name=problem_name).set_current_dofs(olddofs)
            return res


    def _gen_transform(self,transform:Optional[Union[str,PlotTransform]]=None):
        if transform is None:
            return None
        elif transform == "mirror_x":
            return PlotTransformMirror(x=True)
        elif transform == "mirror_y":
            return PlotTransformMirror(y=True)
        elif transform == "mirror_x_y":
            return PlotTransformMirror(x=True,y=True)
        elif transform == "rotate_cw":
            return PlotTransformRotate90()
        elif transform == "rotate_ccw":
            return PlotTransformRotate90(mirror_x=True)
        elif transform == "rotate_ccw_mirror":
            return PlotTransformRotate90(mirror_x=True,mirror_y=True)
        elif isinstance(transform,str):
            raise RuntimeError("Unknown transform mode "+transform)
        else:            
            return transform

    @overload
    def defaults(self,what:Literal["arrows"])->Type[MatplotlibVectorFieldArrows]: ...

    @overload
    def defaults(self,what:Literal["tracers"])->Type[MatplotLibTracers]: ...

    @overload
    def defaults(self,what:Literal["colorbar"])->Type[MatplotLibColorbar]: ...

    def defaults(self,what:str)->Union[Type[MatplotLibPart],Type[MatplotLibColorbar],Type[MatplotLibTricontourf],Type[MatplotLibTricontour],Type[MatplotLibInterfaceLine],Type[MatplotLibElementOutlines]]:
        if not what in self._mode_to_class.keys():
            raise RuntimeError("Can only access defaults for "+str(self._mode_to_class.keys()))
        return self._mode_to_class[what]

    def _add_part(self,mode:str,**kwargs:Any):
        cls = self._mode_to_class[mode]
        part = cls(self)
        part.set_kwargs(kwargs)
        self._added_parts.append(part)
        return part

    def get_spatial_scale_as_float(self,flt:ExpressionNumOrNone=None):
        ss = flt if flt is not None else self.get_problem().get_scaling("spatial")
        if not isinstance(ss,Expression):
            ss=Expression(ss)
        factor, _unit, _rest, _success = _pyoomph.GiNaC_collect_units(ss)
        return float(factor)

    def transform_position(self,xreal:ExpressionNumOrNone=None,yreal:ExpressionNumOrNone=None):
        res:List[float]=[]
        if xreal is not None: # real position to graph position (0,1)
            ss=self.get_problem().get_scaling("spatial")
            if not isinstance(ss,Expression):
                ss=Expression(ss)
            _factor, unit, _rest, _success = _pyoomph.GiNaC_collect_units(ss)
            xreal=float(xreal/unit)
            assert self.xmin is not None and self.xmax is not None
            res.append((xreal-self.xmin)/(self.xmax-self.xmin))
        if yreal is not None: # real position to graph position (0,1)
            ss=self.get_problem().get_scaling("spatial")
            if not isinstance(ss,Expression):
                ss=Expression(ss)
            _factor, unit, _rest, _success = _pyoomph.GiNaC_collect_units(ss)
            yreal=float(yreal/unit)
            assert self.ymin is not None and self.ymax is not None
            res.append((yreal-self.ymin)/(self.ymax-self.ymin))
        if len(res)==0:
            return None
        elif len(res)==1:
            return res[0]
        else:
            return res


    def add_colorbar(self,title:Optional[str]=None,cmap:Optional[str]=None,xpos:Optional[float]=None,ypos:Optional[float]=None,position:Optional[Union[str,Tuple[float,float]]]=None,orientation:Optional[str]=None,factor:Optional[float]=1.0,unit:Union[ExpressionNumOrNone,str]=None,offset:Optional[float]=0.0,length:Optional[float]=None,thickness:Optional[float]=None,norm:Optional[Any]=None,vmin:Optional[float]=None,vmax:Optional[float]=None)->MatplotLibColorbar:
        """
        Adds a colorbar to the plot with a given title, colormap, position either by coordinates or by positional string and a lot of other options.        

        Args:
            title (Optional[str], optional): Title label of the colorbar. Defaults to None.
            cmap (Optional[str], optional): colormap (see matplotlib cmap). Defaults to None.
            xpos (Optional[float], optional): x-position. Defaults to None.
            ypos (Optional[float], optional): y-position. Defaults to None.
            position (Optional[Union[str,Tuple[float,float]]], optional): Use eg. "top center" or "bottom right" to find the correct xpos and ypos. Defaults to None.
            orientation (Optional[str], optional): Horizontal or vertical orientation. Defaults to None.
            factor (Optional[float], optional): Factor to multiply the plotted results with. Defaults to 1.0.
            unit (Union[ExpressionNumOrNone,str], optional): An optional unit. Defaults to None.
            offset (Optional[float], optional): An offset to be added to the data, can be e.g. -273.15 to plot a temperature in celsius. Defaults to 0.0.
            length (Optional[float], optional): Length of the colorbar. Defaults to None.
            thickness (Optional[float], optional): Thickness of the colorbar. Defaults to None.
            norm (Optional[Any], optional): How to normalize the data, can be e.g. matplotlib.norm.LogNorm(...). Defaults to None.
            vmin (Optional[float], optional): minimum data range. Defaults to None.
            vmax (Optional[float], optional): maximum data range. Defaults to None.

        Returns:
            MatplotLibColorbar: The colorbar object to be used in e.g. add_plot("...",colorbar=...)
        """
        allkwargs = {"title": title,"cmap":cmap,"xpos":xpos,"ypos":ypos,"position":position,"orientation":orientation,"factor":factor,"offset":offset,"length":length,"thickness":thickness,"unit":unit,"norm":norm,"_vmin":vmin,"_vmax":vmax}
        res=self._add_part("colorbar",**allkwargs)
        assert isinstance(res,MatplotLibColorbar)
        return res

    def add_axes(self,title:Optional[str]=None,xpos:Optional[float]=None,ypos:Optional[float]=None,position:Optional[Union[str,Tuple[float,float]]]=None,width:Optional[float]=None,height:Optional[float]=None,xlabel:Optional[str]=None,xfactor:Optional[float]=None,ylabel:Optional[str]=None,yfactor:Optional[float]=None)->MatplotLibAxes:
        allkwargs = {"title": title,"xpos":xpos,"ypos":ypos,"position":position,"width":width,"height":height,"xlabel":xlabel,"xfactor":xfactor,"ylabel":ylabel,"yfactor":yfactor}
        res=self._add_part("axes",**allkwargs)
        assert isinstance(res,MatplotLibAxes)
        return res

    def add_text(self,text:str,position:Optional[Union[str,Tuple[float,float]]]=None,textsize:Optional[float]=None,verticalalign:Optional[str]=None,horizontalalign:Optional[str]=None,bbox:Optional[Any]=None,zindex:Optional[float]=None,color:Optional[str]=None)->MatplotlibText:
        """
        Adds text to the plot.

        Args:
            text: The text to be added.
            position: The position of the text, either as coordinates or by e.g. ``"top left"``.
            textsize: Font size.
            verticalalign: Vertical alignment of the text.
            horizontalalign: Horizontal alignment of the text.
            bbox: Frame it with an optional box.
            zindex: Index to order the text in the plot, i.e. which text is in front of which.
            color: Text color.

        Returns:
            The added text.
        """
        allkwargs={"text":text,"position":position,"textsize":textsize,"verticalalign":verticalalign,"horizontalalign":horizontalalign,"bbox":bbox,"zindex":zindex,"color":color}
        res=self._add_part("text",**allkwargs)
        assert isinstance(res,MatplotlibText)
        return res

    def add_scale_bar(self,position:Optional[Union[str,Tuple[float,float]]]=None)->MatplotLibScaleBar:
        """
        Adds a scale bar to the plot.

        Args:
            position (Optional[Union[str,Tuple[float,float]]], optional): Either a string like "top left" or explicit coordinates. Defaults to None.

        Returns:
            MatplotLibScaleBar: The added scalebar
        """
        allkwargs = {"position": position}
        res=self._add_part("scalebar",**allkwargs)
        assert isinstance(res,MatplotLibScaleBar)
        return res

    def add_arrow_key(self,position:Optional[Union[str,Tuple[float,float]]]=None,title:Optional[str]=None,factor:Optional[float]=None,format:Optional[str]=None,unit:ExpressionNumOrNone=None,linewidths:Optional[float]=None)->MatplotLibArrowKey:
        """
        Creates an arrow key to indicate a scale of arrows added at an interface, e.g. for mass transfer.

        Args:
            position: The position of the arrow key.
            title: The title of the arrow key.
            factor: Factor to scale the data with. Useful to express the data e.g. in g/(m^2s) instead of kg/(m^2s).
            format: An optional format string.
            unit: An optional unit of the data.
            linewidths: With of the arrows.

        Returns:
            The arrow key to be used in the :py:meth:`add_plot` method.
        """
        allkwargs = {"position": position,"title":title,"factor":factor,"format":format,"unit":unit,"linewidths":linewidths}
        res=self._add_part("arrowkey",**allkwargs)
        assert isinstance(res,MatplotLibArrowKey)
        return res

    def add_time_label(self,position:Optional[Union[str,Tuple[float,float]]]=None)->MatplotlibTimeLabel:
        """
        Adds a label to show the current time.

        Args:
            position (Optional[Union[str,Tuple[float,float]]], optional): Position like "top center" or "bottom right". Defaults to None.

        Returns:
            MatplotlibTimeLabel: The time label object. Properties can be set after creation.
        """
        allkwargs={"position":position}
        res=self._add_part("timelabel",**allkwargs)
        assert isinstance(res,MatplotlibTimeLabel)
        return res

    def add_polygon(self,pointlist:List[Union[Tuple[float,float],List[float]]],edgecolor:Optional[Any]=None,facecolor:Optional[Any]=None,linewidth:Optional[float]=None,zindex:Optional[float]=None,fill:bool=True,alpha:Optional[float]=None)->MatplotLibPolygon:
        allkwargs={"points":pointlist,"edgecolor":edgecolor,"facecolor":facecolor,"linewidth":linewidth,"zindex":zindex,"fill":fill,"alpha":alpha}
        res=self._add_part("polygon",**allkwargs)
        assert isinstance(res,MatplotLibPolygon)
        return res

    def has_bulk_field(self,field:str,problem_name:str="") -> bool:
        msh = field.split("/")
        if len(msh) < 2:
            return False
        else:
            field=msh[-1]
            mshname="/".join(msh[0:-1])
        msh=self.get_problem(problem_name=problem_name).get_mesh(mshname,return_None_if_not_found=True)
        if msh is None:
            return False
        cached = self._get_mesh_data(msh,problem_name=problem_name,ignore_eigenfactors=True)
        if not field in cached.nodal_field_inds.keys():
            # Check if it is a vector field
            beqs=msh._eqtree.get_code_gen().get_equations()
            assert isinstance(beqs,Equations)
            if field in beqs._vectorfields:
                # Do the magnitude
                #field = beqs._vectorfields.get(field, None)
                return True
            else:
                return False
        return True

    def has_field(self,field:str,problem_name:str="")->bool:
        msh = field.split("/")        
        if len(msh)<=1:
            return False
        field=msh[-1]
        mshname="/".join(msh[0:-1])
        msh=self._problem.get_mesh(mshname,return_None_if_not_found=True)
        if msh is None:
            return False
        cached = self._get_mesh_data(msh,problem_name=problem_name,ignore_eigenfactors=True)        
        if not field in cached.nodal_field_inds.keys():
            # Check if it is a vector field
            beqs=msh._eqtree.get_code_gen().get_equations()
            assert isinstance(beqs,Equations)
            if field in cached.local_expr_indices.keys():
                return True
            if field in beqs._vectorfields:
                # Do the magnitude
                #field = beqs._vectorfields.get(field, None)
                return True
            else:
                return False
        return True        



    def add_plot(self,infield:str,mode:Optional[str]=None,transform:Union[List[Union[PlotTransform,None]],List[Union[str,None]],Union[str,PlotTransform,None]]=None,*,linecolor:Optional[str]=None,linewidths:Optional[float]=None,colorbar:Optional[MatplotLibColorbar]=None,arrowkey:Optional[MatplotLibArrowKey]=None,arrowdensity:Optional[float]=None,arrowstyle:Optional[str]=None,arrowlength:Optional[float]=None,levels:Optional[int]=None,datamap:Optional[Any]=None,axes:Optional[MatplotLibAxes]=None,problem_name:str="")->Union[MatPlotLibAddPlotReturns,List[MatPlotLibAddPlotReturns]]:
        """
        Adds a plot of the field infield (e.g. "domain/velocity") to the current figure.
        If you pass a colorbar, you will get a color plot of the field (potentially along the interface).
        If you pass an interface without any mode, you will get an interface line.
        If you pass and arrowkey for a field at an interface, you will get arrows along the interface, e.g. for indicating mass transfer.
        Otherwise, you have to set the mode, which can be e.g. "arrows" or "streamlines".

        Parameters:
        - infield (str): The name of the field to plot, given by the domain and the final field name, e.g. "domain/velocity".
        - mode (str, optional): The plotting mode. Defaults to None and then selects on the basis of the other arguments if possible. Otherwise, "streamlines" or "arrows" can be used.
        - transform (Union[List[Union[PlotTransform, None]], List[Union[str, None]], Union[str, PlotTransform, None]], optional): The transformation to apply to the plot. Defaults to None, but can be e.g. a list of transforms, e.g. also ["mirror_x",None] to return two plots, one mirrored at the y-axis and one without mirroring.
        - linecolor (str, optional): The color of the lines in the plot. Defaults to None.
        - linewidths (float, optional): The width of the lines in the plot. Defaults to None.
        - colorbar (MatplotLibColorbar, optional): The colorbar to consider for colormap plots. Defaults to None.
        - arrowkey (MatplotLibArrowKey, optional): The arrow key to use for interface arrow plots. Defaults to None.
        - arrowdensity (float, optional): The density of the arrows in the plot. Defaults to None.
        - arrowstyle (str, optional): The style of the arrows (see matplotlib ArrowStyle). Defaults to None.
        - arrowlength (float, optional): The length of the arrows in the plot. Defaults to None.
        - levels (int, optional): The number of levels in the plot, e.g. for contour plots . Defaults to None.
        - datamap (Any, optional): The data map to apply to the plot. Defaults to None.
        - axes (MatplotLibAxes, optional): The axes to use for the plot. Defaults to None.

        Returns:
        - Union[MatPlotLibAddPlotReturns, List[MatPlotLibAddPlotReturns]]: The added plot or a list of added plots if you use e.g. multiple transforms.
        """
   
        
        allkwargs={"linecolor":linecolor,"linewidths":linewidths,"colorbar":colorbar,"arrowkey":arrowkey,"arrowdensity":arrowdensity,"arrowstyle":arrowstyle,"levels":levels,"datamap":datamap,"arrowlength":arrowlength,"axes":axes,"problem_name":problem_name}
        if isinstance(transform,list) :
            res:List[MatPlotLibAddPlotReturns]=[]
            for t in transform:
                entry=self.add_plot(infield,transform=t,mode=mode,**allkwargs)
                assert not isinstance(entry,list)
                res.append(entry)
            return res
        #if isinstance(mode,list) or isinstance(mode,tuple):
        #    res=[]
        #    for m in mode:
        #        res.append(self.add_plot(infield,mode=m,transform=transform,**allkwargs))
        #    return res


        if mode=="image":
            pass
        else:
            msh=infield.split("/")            
            if len(msh)<2:
                if len(msh)==1:
                    # Assuming no data to be plotted on a bulk mesh
                    field=None
                    mshname="/".join(msh)
                else:
                    raise ValueError("Cannot plot the field "+str(infield))
            elif self.get_problem(problem_name=problem_name).get_mesh(infield,return_None_if_not_found=True):
                field=None
                mshname=infield
            else:
                field=msh[-1]
                mshname="/".join(msh[0:-1])
            msh=self._problem.get_mesh(mshname,return_None_if_not_found=True)
            if msh is None:
                raise ValueError("Cannot find the mesh "+mshname+" in the problem to plot "+str(field))
            dim=msh.get_dimension()
            
            cached=self._get_mesh_data(msh,problem_name=problem_name,ignore_eigenfactors=True)
            
            
            if mode is None:
                if field is None:
                    if dim==2:
                        mode="outlines"
                    else:                        
                        mode="interfaceline"
                elif dim==2:
                    mode="tricontourf"
                else:
                    if colorbar is not None:
                        mode="interfacecmap"
                    elif arrowkey is not None:
                        mode="interfacearrows"
                    elif axes is not None:
                        mode="lineplot"
                    else:
                        raise RuntimeError("Please either set 'colorbar' for a color plot of this field along the interface, 'arrowkey' for interface normal arrows or 'axes' for a typical plot in an axes object ")

            if mode is None:
                raise RuntimeError("Cannot automatically determine the plotting mode")

            if not (field is None):
                if not field in cached.nodal_field_inds.keys():
                    # Check if it is a vector field
                    beq=msh._eqtree.get_code_gen().get_equations()
                    assert isinstance(beq,Equations)
                    if field in beq._vectorfields:
                        # Do the magnitude
                        field=beq._vectorfields.get(field,None)
                    elif field in beq._tensorfields:
                        # Do the magnitude
                        field=beq._tensorfields.get(field,None)
                    elif msh.get_tracers(field,error_on_missing=False) is not None:
                        pass
                    elif cached.get_data(field) is None:
                        raise RuntimeError("TODO: Cannot find the field to plot: "+str(infield))


            if not (mode in self._mode_to_class.keys()):
                raise RuntimeError("Cannot plot in mode "+str(mode)+" since there is no class associated with this mode")


        transformG=self._gen_transform(transform)

        cls=self._mode_to_class[mode]
        part=cls(self)
        part.set_kwargs(allkwargs)
        if isinstance(part,MatplotLibPartWithMeshData):
            mirror_x=False
            if transformG is not None:
                mirror_x=transformG.get_mirror()[0]            
            part.set_mesh_data(self._get_mesh_data(msh,problem_name=problem_name,mirror_x=mirror_x),field,transformG) #type:ignore
        elif isinstance(part,MatplotLibTracers):
            part.set_tracer_data(field,msh,transformG) #type:ignore
        elif isinstance(part,MatplotLibImage):
            part.image=infield


        self._added_parts.append(part)
        return part


    def _reset_before_plot(self):
        self._added_parts=[]
        if self.aspect_ratio:
            if self.aspect_ratio is True:
                plt.gca().set_aspect('equal') #type:ignore
            else:
                #raise RuntimeError("TODO ASPECT")
                plt.gca().set_aspect(self.aspect_ratio) #type:ignore
        if self.fullscreen:
            plt.margins(0, 0) #type:ignore
            plt.gca().set_axis_off() #type:ignore
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0) #type:ignore
            plt.gca().xaxis.set_major_locator(plt.NullLocator()) #type:ignore
            plt.gca().yaxis.set_major_locator(plt.NullLocator()) #type:ignore

        for _k,r in self._range_objects.items():
            r.reset_range()


    def perform_plot(self):
        self._has_invalid_triangulation=False
        if self.background_color is not None and self.background_color!="transparent":
            plt.gcf().set_facecolor(self.background_color)
            plt.gca().set_facecolor(self.background_color) #type:ignore

        for entry in sorted(self._added_parts,key=lambda e : e.preprocess_order):            
            entry.pre_process()
            if self._has_invalid_triangulation:
                break
        if not self._has_invalid_triangulation:
            for entry in sorted(self._added_parts,key=lambda e : e.zindex):
                entry.add_to_plot()
                if self._has_invalid_triangulation:
                    break
        if not self._has_invalid_triangulation:
            for entry in sorted(self._added_parts,key=lambda e : e.zindex):
                entry.post_process()
                if self._has_invalid_triangulation:
                    break

    def _after_plot(self):
        #print("PERFORMING PLOT")
        self.perform_plot()
        #print("PLOT PERFORMED")
        if self.file_trunk is not None:
            self.save()
        #print("PLOT SAVED")
        plt.close() #type:ignore
        #print("PLOT CLOSED")

    def ensure_spatial_nondim(self,x:ExpressionOrNum) -> float:
        if isinstance(x,Expression):
            factor,_unit,rest,_success=_pyoomph.GiNaC_collect_units(x)
            return float(factor*rest)
        else:
            return float(x)

    def set_view(self,xmin:ExpressionNumOrNone=None,ymin:ExpressionNumOrNone=None,xmax:ExpressionNumOrNone=None,ymax:ExpressionNumOrNone=None,center:Optional[List[ExpressionOrNum]]=None,size:Optional[List[ExpressionOrNum]]=None):
        """
        Set the view range of the plot. Either by setting the min and max values of x and y or by setting the center and size of the view range.

        Args:
            xmin (ExpressionNumOrNone, optional): The minimum x-coordinate of the view. Defaults to None.
            ymin (ExpressionNumOrNone, optional): The minimum y-coordinate of the view. Defaults to None.
            xmax (ExpressionNumOrNone, optional): The maximum x-coordinate of the view. Defaults to None.
            ymax (ExpressionNumOrNone, optional): The maximum y-coordinate of the view. Defaults to None.
            center (Optional[List[ExpressionOrNum]], optional): The center coordinates of the view. Defaults to None.
            size (Optional[List[ExpressionOrNum]], optional): The size of the view. Defaults to None.
        """

        if center is not None and size is not None:
            self.set_view(xmin=center[0]-size[0]/2,xmax=center[0]+size[0]/2,ymin=center[1]-size[1]/2,ymax=center[1]+size[1]/2)        
        if xmin is not None:
            self.xmin=self.ensure_spatial_nondim(xmin)
            plt.xlim(left=self.xmin) #type:ignore
        if xmax is not None:
            self.xmax=self.ensure_spatial_nondim(xmax)
            plt.xlim(right=self.xmax) #type:ignore
        if ymin is not None:
            self.ymin=self.ensure_spatial_nondim(ymin)
            plt.ylim(bottom=self.ymin) #type:ignore
        if ymax is not None:
            self.ymax=self.ensure_spatial_nondim(ymax)
            plt.ylim(top=self.ymax) #type:ignore
        

        if self.aspect_ratio and self.fullscreen and (self.xmin is not None) and (self.xmax is not None) and (self.ymin is not None) and (self.ymax is not None):
            # Enforce the image size to match it
            dx=self.xmax-self.xmin
            dy=self.ymax-self.ymin
            #print("DX and DY",dx,dy,self.xmax,self.xmin,self.xmax-self.xmin)
            ar=1 if self.aspect_ratio is True else self.aspect_ratio
            RX=self.image_size[0]/dx
            RY = self.image_size[1] / dy

            if RX>RY:
                wW,hH=self.image_size[0] / self.dpi , self.image_size[0] * dy / dx / self.dpi
            else:
                wW,hH=self.image_size[1]*dx/dy / self.dpi, self.image_size[1]/ self.dpi


            if int(wW*self.dpi)%2==1:
                wW=(int(wW*self.dpi)+1.00001)/self.dpi
            if int(hH*self.dpi)%2==1:
                hH=(int(hH*self.dpi)+1.00001)/self.dpi
            #print(W, H, W * self.dpi, H * self.dpi)
            #exit()
            plt.gcf().set_size_inches(wW,hH*ar)


