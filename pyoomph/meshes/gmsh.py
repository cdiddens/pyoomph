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
 
from pathlib import Path
from ..typings import *
import numpy


from ..expressions.generic import Expression, ExpressionNumOrNone, ExpressionOrNum

from .mesh import MeshTemplate

# from .meshio import MeshioMesh2d
import _pyoomph
import pygmsh #type:ignore
from pygmsh.common.point import Point #type:ignore

from pygmsh.common.line import Line #type:ignore
from pygmsh.common.spline import Spline #type:ignore
from pygmsh.common.bspline import BSpline #type:ignore
from pygmsh.common.circle_arc import CircleArc #type:ignore 

from pygmsh.common.plane_surface import PlaneSurface #type:ignore
from pygmsh.common.surface import Surface #type:ignore

from pygmsh.common.volume import Volume #type:ignore


import gmsh #type:ignore
import os

import meshio #type:ignore
import scipy.optimize #type:ignore
import re
from scipy import interpolate #type:ignore


import scipy.spatial #type:ignore
import math

if TYPE_CHECKING:
    from ..generic.problem import Problem

class GmshSizeCallback:
    def __init__(self,default_resolution:float=1.0):
        self.gmsh:"GmshTemplate"
        self.default_resolution=default_resolution
        self._registered_handlers:List[Dict[int,Callable[[float,float,float],float]]] = [{},{},{},{}]


    def initialize(self):
        self._registered_handlers = [{},{},{},{}]
        for m in dir(self):
            if m.startswith("size_"):
                if m.startswith("size_line_"):
                    name=m[10:]
                    entities=self.gmsh._named_entities.get(name,None) #type:ignore
                    if entities is not None:
                        for entity in entities:
                            self._registered_handlers[1][entity._id]=getattr(self,m) #type:ignore
                elif m.startswith("size_surface_"):
                    name=m[13:]
                    entities=self.gmsh._named_entities.get(name,None) #type:ignore
                    if entities is not None:
                        for entity in entities:
                            self._registered_handlers[2][entity._id]=getattr(self,m) #type:ignore

    def get_points_at_line(self,name:str,merge_connected:bool=True,sort:Optional[Literal["x","y","z","rev_x","rev_y","rev_z"]]=None,circle_arc_samples:int=25):
        entities = self.gmsh._named_entities.get(name, None) #type:ignore
        if entities is None:
            raise RuntimeError("No named entity with name "+name)
        allpts:List[NPFloatArray]=[]
        for entity in entities:
            pts:List[List[float]]=[]
            if isinstance(entity,(pygmsh.geo.geometry.common.geometry.Line,pygmsh.geo.geometry.common.geometry.Spline,pygmsh.geo.geometry.common.geometry.BSpline)):
                for pt in entity.points: #type:ignore
                    pts.append(pt.x) #type:ignore
            elif isinstance(entity,pygmsh.geo.geometry.common.geometry.CircleArc):
                start=numpy.array(entity.points[0].x) #type:ignore
                center = numpy.array(entity.points[1].x) #type:ignore
                end = numpy.array(entity.points[2].x) #type:ignore
                s=start-center #type:ignore
                e=end-center #type:ignore
                r=numpy.linalg.norm(s) #type:ignore
                s,e=s/r,e/r #type:ignore
                t_samples=numpy.linspace(0,1,circle_arc_samples) #type:ignore
                t_samples=t_samples #type:ignore
                slerps=scipy.spatial.geometric_slerp(s,e,t_samples) #type:ignore
                slerps=slerps[1:-1]
                pts.append(list(start)) #type:ignore
                for s in slerps:
                    pts.append(list(r*s+center))
                pts.append(list(end)) #type:ignore
            else:
                raise RuntimeError("Implement circle (and potentially samples spline?)")
            ptsN=numpy.array(pts) #type:ignore
            allpts.append(ptsN)

        # See whether we can connect parts
        if merge_connected:
            old=allpts
            allpts=[]
            while len(old):
                start=old[-1]
                end=start
                current_seg=[start]
                old.pop()
                for o2 in old:
                    #print("S",start)
                    #print("E",end)
                    #print("O",o2)
                    #print(numpy.linalg.norm(start[0]-o2[-1]))
                    dist=1e-8
                    if numpy.linalg.norm(end[-1]-o2[0])<dist: #type:ignore
                        #print("A")
                        end=o2[1:]
                        old.remove(o2)
                        current_seg.append(end)
                    elif numpy.linalg.norm(start[0]-o2[-1])<dist: #type:ignore
                        #print("B")
                        start=o2[:-1]
                        old.remove(o2)
                        current_seg.insert(0,start)
                    elif numpy.linalg.norm(end[-1]-o2[-1])<dist: #type:ignore
                        #print("C")
                        end=o2[-2::-1]
                        old.remove(o2)
                        current_seg.append(end)
                    elif numpy.linalg.norm(start[0]-o2[0])<dist: #type:ignore
                        #print("D")
                        start=o2[:0:-1]
                        old.remove(o2)
                        current_seg.insert(0,start)
                current_seg=numpy.vstack(current_seg) #type:ignore
                allpts.append(current_seg)
        #exit()
        if (sort is not None) and len(allpts)>0:
            if sort in {"x","y","z","rev_x","rev_y","rev_z"}:
                si:int={"x":0,"rev_x":0,"y":1,"rev_y":1,"z":2,"rev_z":2}[sort]
                allpts=[pts[numpy.argsort(pts[:,si])] for pts in allpts ] #type:ignore
                allpts=list(sorted(allpts,key=lambda pts:pts[0][si]))
                if sort in {"rev_x","rev_y","rev_z"}:
                    allpts=[numpy.array(list(reversed(pt))) for pt in reversed(allpts)] #type:ignore
            else:
                raise ValueError("sort must be one of x,y,z,rev_x,rev_y,rev_z")

        return allpts


    def default_size(self,dim:int,tag:int,x:float,y:float,z:float)->float:
        return self.default_resolution

    def _cb(self,dim:int,tag:int,x:float,y:float,z:float)->float:
        if tag in self._registered_handlers[dim].keys():
            return self._registered_handlers[dim][tag](x, y, z)
        return self.default_size(dim,tag,x,y,z)

    def finalize(self):
        pass

    def _setup_for_mesh(self,gmsh:"GmshTemplate")->Callable[[int,int,float,float,float],float]:
        self.gmsh=gmsh
        self.initialize()
        return lambda dim,tag,x,y,z : self._cb(dim,tag,x,y,z)





def generate_mesh_to_file(geom:pygmsh.geo.Geometry, outdir:str, trunk:str, mesher:Optional["GmshTemplate"]=None,dim:int=2, order:Optional[int]=None, algorithm:Optional[int]=None, verbose:bool=False, recombine_algo:Optional[int]=None,
                          postgen_cb:Optional[Callable[[],None]]=None, only_geo:bool=False,mesh_mode:Optional[str]=None,mesh_size_callback:Optional[Union[GmshSizeCallback,Callable[[int,int,float,float,float],float]]]=None):
    geom.synchronize()

    for item in geom._AFTER_SYNC_QUEUE: #type:ignore
        item.exec() #type:ignore

    for item, host in geom._EMBED_QUEUE: #type:ignore
        gmsh.model.mesh.embed(item.dim, [item._id], host.dim, host._id) #type:ignore

    # set compound entities after sync
    for c in geom._COMPOUND_ENTITIES: #type:ignore
        gmsh.model.mesh.setCompound(*c) #type:ignore

    for s in geom._RECOMBINE_ENTITIES: #type:ignore
        gmsh.model.mesh.setRecombine(*s) #type:ignore

    for t in geom._TRANSFINITE_CURVE_QUEUE: #type:ignore
        gmsh.model.mesh.setTransfiniteCurve(*t) #type:ignore

    for t in geom._TRANSFINITE_SURFACE_QUEUE: #type:ignore
        gmsh.model.mesh.setTransfiniteSurface(*t) #type:ignore

    for e in geom._TRANSFINITE_VOLUME_QUEUE: #type:ignore
        gmsh.model.mesh.setTransfiniteVolume(*e) #type:ignore

    for item, size in geom._SIZE_QUEUE: #type:ignore
        gmsh.model.mesh.setSize(gmsh.model.getBoundary(item.dim_tags, False, False, True), size) #type:ignore

    for entities, label in geom._PHYSICAL_QUEUE: #type:ignore
        d = entities[0].dim #type:ignore
        assert all(e.dim == d for e in entities) #type:ignore
        tag = gmsh.model.addPhysicalGroup(d, [e._id for e in entities]) #type:ignore
        if label is not None: 
            gmsh.model.setPhysicalName(d, tag, label) #type:ignore

    for entity in geom._OUTWARD_NORMALS: #type:ignore
        gmsh.model.mesh.setOutwardOrientation(entity.id) #type:ignore

    if mesh_mode=="SV":
        if order!=1:
            raise RuntimeError("mesh_mode='SV' only works for order=1")
    if order is not None:
        gmsh.model.mesh.setOrder(order) #type:ignore

    


    if verbose:
        gmsh.option.setNumber("General.Terminal", 1) #type:ignore

    # set algorithm
    # http://gmsh.info/doc/texinfo/gmsh.html#index-Mesh_002eAlgorithm
    if recombine_algo:
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", recombine_algo) #type:ignore
    if algorithm:
        gmsh.option.setNumber("Mesh.Algorithm", algorithm) #type:ignore

    if order and order == 2:
        gmsh.option.setNumber("Mesh.ElementOrder", 2) #type:ignore
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0) #type:ignore
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 1) #type:ignore


    if mesh_mode in ["only_quads"]:
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm",1)
    if mesh_mode in ["quads","only_quads"]:
        gmsh.option.setNumber("Mesh.RecombineAll", 1) #type:ignore
        
    if mesher:
        for n,v in mesher.gmsh_options.items():
            if n=="algorithm":
                continue
            #print("SETTING",n,v,"FOR",mesher,"IN",mesher.gmsh_options,"IN",mesher.gmsh_options.items())
            gmsh.option.setNumber(n,v) #type:ignore
        
    gmsh.write(os.path.join(outdir, trunk + ".geo_unrolled")) #type:ignore

    

    if only_geo:
        return

    
    if mesh_size_callback:
        print("HAS MESH SIZE CB", mesh_size_callback, mesher)
        if isinstance(mesh_size_callback,GmshSizeCallback):
            mesh_size_callback=mesh_size_callback._setup_for_mesh(mesher) #type:ignore
        gmsh.model.mesh.setSizeCallback(None) #type:ignore
        gmsh.model.mesh.setSizeCallback(mesh_size_callback) #type:ignore
    gmsh.model.mesh.generate(dim)
    if postgen_cb is not None:
        postgen_cb()

    gmsh.write(os.path.join(outdir, trunk + ".msh")) #type:ignore
    gmsh.clear()





class GmshTemplate(MeshTemplate):
    """
    A template for creating a mesh using Gmsh as backend. Specify the geometry in an overridden :py:meth:`define_geometry` method, using the :py:meth:`point`, :py:meth:`line`, :py:meth:`spline`, :py:meth:`circle_arc`, :py:meth:`plane_surface` and other methods.
    """
    def __init__(self,loaded_from_mesh_file:Optional[str]=None):
        super(GmshTemplate, self).__init__()
        self._meshfile:Optional[str]
        self._loaded_from_mesh_file=loaded_from_mesh_file
        
        #: If True, macro elements will be used for the mesh, i.e. curved elements will be considered
        self.use_macro_elements:bool=True
        self._geom:Optional[pygmsh.geo.Geometry] = None
        self._named_entities:Dict[str,List[object]] = {}
        self._rev_names:Dict[object,str] = {}
        self._dim_tag_names:Dict[Tuple[int,int],Tuple(str,object)] = {}
        
        #: If set, the input mesh will be mirrored and copied along the given axis or axes. Useful to generate symmetric meshes for e.g. pitchfork tracking
        self.mirror_mesh:Optional[Union[Literal["mirror_x","mirror_y"],List[Literal["mirror_x","mirror_y"]]]]=None
        #: If set, the entire mesh will be extruded in the next dimension. The first entry in the tuple is the dimensional distance, the second the number of layers. 
        self.extrude_generated_mesh:Optional[Tuple[ExpressionOrNum,int]]=None 
        self.gmsh_options:Dict[str,int] = {}
        #self.gmsh_options["algorithm"] = 8
        #self.gmsh_options["recombine_algo"] = 2
        #self.gmsh_options["recombine_algo"] = None
        self._entities2d:Dict[int,Union[PlaneSurface,Surface]] = {}
        self._entities1d:Dict[int,Union[Line,Spline,BSpline,CircleArc]] = {}
        self._entities0d:Dict[int,Point] = {}
        self._pointhash:Dict[Tuple[float,float,float],Point] = {}
        self._point_size_hash:Dict[Point,float] = {}
        self._onedims_attached_to_point:Dict[Point,Set[Union[Line,Spline,BSpline,CircleArc]]]={}
        

        self._mesh_size_callback=None

        self._curved_entities0d = {}  # TODO Does this make sense at all?
        self._curved_entities1d:Dict[int,_pyoomph.MeshTemplateCurvedEntityBase] = {} 
        self._curved_entities2d:Dict[int,_pyoomph.MeshTemplateCurvedEntityBase] = {}  # TODO: Set those

        self._mesh:Any = None
        #: The default resolution for the mesh as a nondimensional typical element length scale
        self.default_resolution:Optional[float] = None
        
        #: This factor is used to scale all size arguments (including the default resolution) by the given factor. Useful to e.g. increase the mesh resolution by a factor.
        self.mesh_size_factor:float=1
        #self.default_quads = True
        #self.default_quads = False
        
        #: Selects the default element type of the mesh. Can be ``"quads"`` (try to create quads if possible), ``"tris"`` (only triangles), ``"SV"`` (Scott-Vogelius elements) or ``"only_quads"`` (only quadrilateral elements by splitting triangles)
        self.mesh_mode:Literal["quads","tris","SV","only_quads"]="quads"
        #: The default order of the elements. Can be 1 or 2. Note that if only first order (``"C1"``) elements are created, the mesh will be reduced to first order, even if the mesh is set to second order. Likewise, a first order mesh will be split to second order if second order elements (``"C2"``) are created on it.
        self.order = 2
        self._maxdim = 0

        self.consider_spatial_scale:bool=True

        if False and  self._loaded_from_mesh_file:
            self._geometry_defined = True
            super(GmshTemplate, self)._do_define_geometry(self._problem)
            self._set_problem(self._problem)
            print("Loading mesh from: "+self._loaded_from_mesh_file)
            self._load_mesh(self._loaded_from_mesh_file)
        pass



    def point(self, x:ExpressionOrNum, y:ExpressionOrNum=0.0, z:ExpressionOrNum=0.0, size:ExpressionNumOrNone=None, *,name:Optional[str]=None,consider_spatial_scale:Optional[bool]=None)->Point:
        """
        Add a point to the geometry. Coordinates must be given in the spatial unit, e.g. in meter if the problem has a metric set_scaling(spatial=...) set.
        The size controls the mesh size and will default to self.default_resolution if not given.
        By a name, the point can be identified later, e.g. for a boundary condition.

        Args:
            x: The x-coordinate of the point.
            y: The y-coordinate of the point. Defaults to 0.0.
            z: The z-coordinate of the point. Defaults to 0.0.
            size: The size of the point. Defaults to None, which means default_resolution. Negative sizes are in terms of default_resolution
            name: The name of the point. Defaults to None.
            consider_spatial_scale: Whether to consider spatial scaling. Defaults to None.

        Returns:
            Point: The created point. Can be used for e.g. creating lines, circle_arcs or splines.

        Raises:
            RuntimeError: If geometry is added outside the 'define_geometry' function.
            RuntimeError: If the mesh resolution (size argument) is not a float.

        """
        if self._geom is None:
            raise RuntimeError("Can only add geometry inside the function 'define_geometry'")
        coord = [x, y, z]
        if consider_spatial_scale is None:
            consider_spatial_scale=self.consider_spatial_scale
        for i, c in enumerate(coord):
            if consider_spatial_scale:
                c = c / self._problem.get_scaling("spatial")
                if isinstance(c,Expression):
                    c=c.float_value()
                coord[i] = c
        x, y, z = cast(List[float],coord)
        if self._pointhash.get(tuple([x, y, z])) is not None:
            return self._pointhash[tuple([x, y, z])]
        if size is None:
            size = self.default_resolution
        if size is not None:
            if not (isinstance(size, int) or isinstance(size, float)):
                try:
                    size=float(size)
                except:
                    raise RuntimeError("mesh resolution (i.e. size argument) is expected to be nondimensional, i.e. must be a float, not "+str(size))
    #            if isinstance(size, _pyoomph.Expression):
    #                size = size / self._problem.get_scaling("spatial")
    #                size = size.float_value()
        size=cast(float,size)
        if size is not None:
            if size<0:
                size=-size*self.default_resolution
            size*=self.mesh_size_factor
        if size is not None and self.mesh_mode=="only_quads":
            size*=2
        res = self._geom.add_point([x, y, z], size) #type:ignore
        self._pointhash[tuple([x, y, z])] = res
        self._point_size_hash[res]=size
        self._store_name(name, res)
        self._entities0d[res._id] = res #type:ignore
        self._maxdim = max(self._maxdim, 0)
        return res

    def points(self,*coords:List[ExpressionOrNum],size:Optional[Union[float,Sequence[float]]]=None) -> List[Point]:
        res:List[Point]=[]
        if size:
            if isinstance(size,float):
                sizeC=[size]*len(coords)
            else:
                sizeC=size
        else:
            sizeC=None
        for i,c in enumerate(coords):
            s=None
            if sizeC is not None:
                s=sizeC[i]
            x,y,z=0,0,0
            if len(c)>0:
                x=c[0]
                if len(c)>1:
                    y=c[1]
                    if len(c)>2:
                        z=c[2]
                        if len(c)>3:
                            s=c[3]
                            assert isinstance(s,float) or s is None
            res.append(self.point(x,y,z,size=s))
        return res

    def _store_name(self, name:Optional[str], obj:object):
        if name is None:
            return
        if isinstance(obj, list):
            for e in obj: #type:ignore
                self._store_name(name, e) #type:ignore
            return
        if not (name in self._named_entities.keys()):
            self._named_entities[name] = []
        self._named_entities[name].append(obj)
        self._rev_names[obj] = name
        self._dim_tag_names[obj.dim_tag] = (name,obj)

    def _resolve_name(self, typ:str, *args:Union[str,object])->List[object]:
        res:List[object] = []
        for a in args:
            if a is None:
                continue
            if isinstance(a, str):
                if not (a in self._named_entities.keys()):
                    # Test for glob
                    if '*' in a:
                        r = re.compile(a)
                        newlist = list(filter(r.match, self._named_entities.keys()))
                        if len(newlist) == 0:
                            raise ValueError("No named mesh entity matched the regex")
                        else:
                            return self._resolve_name(typ, *newlist)

                    raise ValueError("Cannot find an entity with name '" + a + "'")
                sub = self._named_entities[a]
                for b in sub:
                    res.append(b)
            else:
                res.append(a)
        return res

    def line(self, *args:Union[Sequence[ExpressionOrNum],Point], name:Optional[str]=None)->Optional[Line]:
        """
        Create a line (segment-wise) line for the mesh. When given a name, it can be used to identify the line later, e.g. for boundary conditions.

        Args:
            *args: Variable-length argument list of points or sequences of points defining the line.
            name: Name of the line entity.

        Returns:
            The created line entity, or None if the line is degenerate.

        Raises:
            RuntimeError: If the geometry is not defined inside the 'define_geometry' function.
        """
        if self._geom is None:
            raise RuntimeError("Can only add geometry inside the function 'define_geometry'")
        
        argsc:List[Point]=[]
        for _,a in enumerate(list(args)):
            if isinstance(a,(list,tuple)):
                assert len(a)<=3 or isinstance(a[3],float)
                argsc.append(self.point(*a)) #type:ignore
            else:
                assert isinstance(a,Point)
                argsc.append(a)
        
        if argsc[0]==argsc[1]:
            return None
        
        for _,other in self._entities1d.items():
            if isinstance(other,pygmsh.geo.geometry.common.geometry.Line):
                if len(other.points)==len(argsc) and ((other.points[0]==argsc[0] and other.points[1]==argsc[1]) or (other.points[0]==argsc[1] and other.points[1]==argsc[0])):
                    #TODO CHeck name
                    return other
        
        res = self._geom.add_line(*argsc) #type:ignore
        self._store_name(name, res)
        self._entities1d[res._id] = res #type:ignore
        self._maxdim = max(self._maxdim, 1)
        
        for p in argsc:
            if p not in self._onedims_attached_to_point:
                self._onedims_attached_to_point[p]=set()
            self._onedims_attached_to_point[p].add(res)
        
        return res

    def make_lines_transfinite(self,*linesIn:Union[Line,str],numnodes:Union[Literal["auto"],int]="auto",mode:str="Progression",coeff:Optional[float]=None,dry_run:bool=False) -> List[Tuple[int, float]]:
        lines=self._resolve_name("lines", *linesIn)
        res_info:List[Tuple[int,float]]=[]
        for line in lines:
            if numnodes == "auto":
                if isinstance(line,(pygmsh.geo.geometry.common.geometry.Line, pygmsh.geo.geometry.common.geometry.Spline,pygmsh.geo.geometry.common.geometry.BSpline)):
                    lastpt = line.points[0] #type:ignore
                    line_len=0
                    for p in line.points:   #type:ignore
                        dl = numpy.sqrt(sum((lastpt.x[i] - p.x[i]) ** 2 for i in range(3))) #type:ignore
                        lastpt = p  #type:ignore
                        line_len+=dl
                elif isinstance(line, pygmsh.geo.geometry.common.geometry.CircleArc):
                    start = numpy.array(line.points[0].x) #type:ignore
                    center = numpy.array(line.points[1].x) #type:ignore
                    end = numpy.array(line.points[2].x) #type:ignore
                    s = start - center  #type:ignore
                    e = end - center  #type:ignore
                    r = numpy.linalg.norm(s)  #type:ignore
                    s, e = s / r, e / r  #type:ignore
                    t_samples = numpy.linspace(0, 1, 25)  #type:ignore # TODO: Make this analytical
                    t_samples = t_samples  #type:ignore
                    slerps = scipy.spatial.geometric_slerp(s, e, t_samples)  #type:ignore
                    slerps = slerps[1:-1]  #type:ignore
                    pts = [] 
                    pts.append(list(start))  #type:ignore
                    for s in slerps:
                        pts.append(list(r * s + center))  #type:ignore
                    pts.append(list(end))  #type:ignore
                    lastpt = pts[0]  #type:ignore
                    line_len = 0
                    for p in pts:   #type:ignore
                        dl = numpy.sqrt(sum((lastpt[i] - p[i]) ** 2 for i in range(3)))  #type:ignore
                        lastpt = p  #type:ignore
                        line_len += dl

                sstart=self._point_size_hash[line.points[0]]  #type:ignore
                send=self._point_size_hash[line.points[-1]]  #type:ignore
                numnodes = math.ceil(0.5 * (line_len /sstart + line_len / send) - 1e-12)  #type:ignore
                if coeff is None:
                    coeff = ( send/sstart) ** (1.0 / ((2 if numnodes < 2 else numnodes) - 1))  #type:ignore
                    #if send < sstart:
                    #    if coeff > 1:
                    #        coeff = 1 / coeff
                    #if line._id<0:
                    #    coeff=1/coeff

            if coeff is None:
                coeff=1.0
            res_info.append((numnodes,coeff,))
            if not dry_run:
                self._geom.set_transfinite_curve(line,numnodes,mesh_type=mode,coeff=coeff) #type:ignore
        return res_info

    def make_surface_transfinite(self,*surfsIn:Union[str,PlaneSurface],corners:List[Point]=[],arrangement:str=""):
        surfs=self._resolve_name("surfaces", *surfsIn)
        for surf in surfs:
            for s in surf: #type:ignore
                if len(corners) == 0:
                    if s.num_edges!=4: #type:ignore
                        raise RuntimeError("Please set corners explicitly, surface has more than 4 corners")
                    if len(s.holes)>0: #type:ignore
                        raise RuntimeError("Please set corners explicitly, surface has holes")
                    corners=[c.points[0] for c in s.curve_loop.curves] #type:ignore
                    linfos=[self.make_lines_transfinite(l,dry_run=True)[0] for l in s.curve_loop.curves] #type:ignore
                    N1=int(numpy.ceil( (linfos[0][0]+linfos[2][0]-1e-10)/2))
                    N2 = int(numpy.ceil((linfos[1][0] + linfos[3][0] - 1e-10) / 2))
                    NS=[N1,N2,N1,N2]
                    #Progs = [N1, N2, N1, N2]
                    for i,l in enumerate(s.curve_loop.curves): #type:ignore
                        print(i,l,linfos[i][1]) #type:ignore
                        coeff=linfos[i][1] #type:ignore
                        #if l._id<0:
                        #    coeff=1/coeff
                        self.make_lines_transfinite(l,numnodes=NS[i],coeff=coeff) #type:ignore
                    #exit()
                self._geom.set_transfinite_surface(s,arrangement,corner_pts=corners) #type:ignore


    # Add lines as p1, <name>, p2, <name>, p3, <name>, p4...
    def create_lines(self, *args:Union[Point,List[ExpressionOrNum],Tuple[ExpressionOrNum],str]) -> List[Line]:
        """
        Creates multiple lines with different names based on the given arguments. 
        For line loop around a box, you can e.g. do
        
        .. code-block:: python
        
            lines=self.create_lines((0,0), "left", (0,1), "top", (1,1), "right", (1,0), "bottom", (0,0))
            self.plane_surface(*lines,name="box") # Create the surface of the box

        Args:
            *args: Variable number of arguments representing the points and names of the lines. Each argument should be in the format: p1, <name>, p2, <name>, p3, <name>, p4, ... If it stops with a name, not a point, it will close the line loop to the first point

        Returns:
            A list of Line objects representing the created lines.

        Raises:
            ValueError: If the arguments are not in the correct format.
        """
        closed_loop=False
        if len(args) % 2 != 1:
            closed_loop=True
            #raise ValueError("create line needs arguments like p1, <name>, p2, <name>, p3, <name>, p4 ,...")
        NL = len(args) // 2
        res:List[Line] = []
        for i in range(NL):
            pstart = args[2 * i]
            name = args[2 * i + 1]
            if closed_loop and 2*i+2==len(args):
                pend=args[0]
            else:
                pend = args[2 * i + 2]
            if (not isinstance(pstart, (Point,list,tuple))) or (not isinstance(pend, (Point,list,tuple))) or not (isinstance(name, str)):
                raise ValueError("create line needs arguments like p1, <name>, p2, <name>, p3, <name>, p4 ,...")
            if isinstance(pstart,(list,tuple)):
                pstart=self.point(*pstart) #type:ignore
            if isinstance(pend,(list,tuple)):
                pend=self.point(*pend) #type:ignore
            if name=="":
                name=None
            lin=self.line(pstart, pend, name=name)
            if lin is not None:
                res.append(lin)
        return res

    def bspline(self, ptlist:Sequence[Point], *, name:Optional[str]=None)->BSpline:
        if self._geom is None:
            raise RuntimeError("Can only add geometry inside the function 'define_geometry'")
        res = self._geom.add_bspline(ptlist) #type:ignore
        self._store_name(name, res)
        self._entities1d[res._id] = res #type:ignore
        self._maxdim = max(self._maxdim, 1)
        for p in ptlist:
            if p not in self._onedims_attached_to_point:
                self._onedims_attached_to_point[p]=set()
            self._onedims_attached_to_point[p].add(res)
        return res

    def spline(self, ptlistIn:Sequence[Union[Point,Sequence[ExpressionOrNum]]], *, name:Optional[str]=None,with_macro_element:bool=True)->Spline:
        """
        Create a spline curve given by a list of points. If a name is supplied, it can be used for e.g. boundary conditions.

        Args:
            ptlistIn: List of points defining the spline curve.
            name: Name of the spline curve. Default is None.
            with_macro_element: Flag indicating whether to include a macro element for the spline curve. Default is True.

        Returns:
            Spline: The created spline curve.
        """
        if self._geom is None:
            raise RuntimeError("Can only add geometry inside the function 'define_geometry'")
        ptlist=[pt for pt in ptlistIn]
        for i,p in enumerate(ptlist):
            if isinstance(p,(list,tuple)):
                ptlist[i]=self.point(*p) #type:ignore
        ptlist=cast(List[Point],ptlist)
        for _,other in self._entities1d.items():
            if isinstance(other,pygmsh.geo.geometry.common.geometry.Spline):
                if len(other.points)==len(ptlist):
                    all_the_same=True
                    for i in range(len(other.points)):
                        if other.points[i]!=ptlist[i]:
                            all_the_same=False
                            break
                    if all_the_same:
                        return other
                    all_the_same=True
                    for i in range(len(other.points)):
                        if other.points[i] != ptlist[len(ptlist)-i-1]:
                            all_the_same = False
                            break
                    if all_the_same:
                        return other
        res = self._geom.add_spline(ptlist) #type:ignore
        self._store_name(name, res)
        self._entities1d[res._id] = res #type:ignore
        if with_macro_element:
            self._curved_entities1d[res._id] = _pyoomph.CurvedEntityCatmullRomSpline(numpy.array([p.x for p in ptlist])) #type:ignore
        # self._curved_entities1d[res._id] = CurvedEntitySpline(ptlist)
        self._maxdim = max(self._maxdim, 1)
        for p in ptlist:
            if p not in self._onedims_attached_to_point:
                self._onedims_attached_to_point[p]=set()
            self._onedims_attached_to_point[p].add(res)
        return res

    def create_circle_lines(self,centre:Union[Tuple[ExpressionOrNum,...],Point],radius:ExpressionOrNum,*,mesh_size:Optional[float]=None,line_name:Optional[str]=None)->List[Line]:
        if not isinstance(centre,Point):
            centre=self.point(*centre)
        corners:List[Point]=[]
        SS=self.get_problem().get_scaling("spatial")
        for signs in [[1,0],[0,1],[-1,0],[0,-1]]:
            corners.append(self.point(centre.x[0]*SS +signs[0]*radius,centre.x[1]*SS+signs[1]*radius,size=mesh_size))
        corners.append(corners[0])
        lines:List[CircleArc]=[]
        for i in range(4):
            lines.append(self.circle_arc(corners[i],corners[i+1],center=centre,name=line_name))
        return lines
            

    def circle_arc(self, startpt:Union[Point,Sequence[ExpressionOrNum]], endpt:Union[Point,Sequence[ExpressionOrNum]], *, center:Optional[Union[Point,Sequence[ExpressionOrNum]]]=None, through_point:Optional[Union[Point,Sequence[ExpressionOrNum]]]=None, name:Optional[str]=None, with_macro_element:bool=True)->Optional[Union[Line,CircleArc]]:
        """
        Adds a circlular arc to the mesh geometry.

        Parameters:
            startpt: The starting point of the circle arc.
            endpt: The ending point of the circle arc.
            center: The center point of the circle arc. If not provided, it will be calculated based on the through_point argument.
            through_point: A point that the circle arc should pass through. If not provided, the circle arc requires a center.
            name: The name of the circle arc to identify it later e.g. as boundary.
            with_macro_element: Whether to include a macro element. In that case, spatial refinements (on moving meshes only the initial refinement) will be mapped on the circle.

        Returns:
            The created circle arc or a line if the circle arc is degenerate.

        Raises:
            RuntimeError: If the geometry is not defined inside the 'define_geometry' function.
            RuntimeError: If both center and through_point are provided.
        """
                      
        if self._geom is None:
            raise RuntimeError("Can only add geometry inside the function 'define_geometry'")
        if isinstance(startpt,(list,tuple)):
                startpt=self.point(*startpt)
        if isinstance(endpt,(list,tuple)):
                endpt=self.point(*endpt)
        if (center is None) and not (through_point is None):
            if isinstance(through_point, (list, tuple)):
                through_point = self.point(*through_point)
            x1, y1 = startpt.x[0], startpt.x[1] #type:ignore
            x2, y2 = endpt.x[0], endpt.x[1] #type:ignore
            x3, y3 = through_point.x[0], through_point.x[1] #type:ignore
            a = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2; #type:ignore
            b = (x1 * x1 + y1 * y1) * (y3 - y2) + (x2 * x2 + y2 * y2) * (y1 - y3) + (x3 * x3 + y3 * y3) * (y2 - y1); #type:ignore
            c = (x1 * x1 + y1 * y1) * (x2 - x3) + (x2 * x2 + y2 * y2) * (x3 - x1) + (x3 * x3 + y3 * y3) * (x1 - x2); #type:ignore
            if abs(a) < 1e-10: #type:ignore
                res = self.line(startpt, endpt)
                if res is not None:
                    self._store_name(name, res)
                    self._entities1d[res._id] = res #type:ignore
                return res
            else:
                xc = -b / (2 * a)  #type:ignore
                yc = -c / (2 * a)  #type:ignore
                center = self.point(xc, yc,consider_spatial_scale=False)  #type:ignore
                res = self._geom.add_circle_arc(startpt, center, endpt)  #type:ignore
        elif (through_point is None) and not (center is None):
            if isinstance(center, (list, tuple)):
                center = self.point(*center)
            res = self._geom.add_circle_arc(startpt, center, endpt)  #type:ignore
        elif (center is None):
            raise RuntimeError("Either use center=... or through_point=...")
        else:
            raise RuntimeError("Cannot use kwargs center and through_point at the same time")
        self._store_name(name, res)
        self._entities1d[res._id] = res #type:ignore
        # self._curved_entities1d[res._id] = CurvedEntityCircleArc(startpt,center, endpt)
        if with_macro_element:
            self._curved_entities1d[res._id] = _pyoomph.CurvedEntityCircleArc(center.x, startpt.x, endpt.x) #type:ignore
        self._maxdim = max(self._maxdim, 1)
        for p in [startpt, endpt]: 
            assert isinstance(p,Point)
            if p not in self._onedims_attached_to_point: 
                self._onedims_attached_to_point[p]=set()
            self._onedims_attached_to_point[p].add(res)
        return res



    def _get_boundary_corner_size_map(self)->Dict[str,Dict[Tuple[float,...],float]]:
        entlist:Dict[Union[Line,Spline,BSpline,CircleArc],Set[Point]]=dict()
        for pt,ptinfo in self._onedims_attached_to_point.items():
            for l in ptinfo:
                if l not in entlist:
                    entlist[l]=set()
                entlist[l].add(pt)
        res:Dict[str,Dict[Tuple[float],float]] = {}
        for l,pts in entlist.items():
            name=self._rev_names.get(l)
            if name is None:
                continue
            if name not in res:
                res[name]={}
            for p in pts:
                attached=self._onedims_attached_to_point[p]
                for att in attached:
                    if att is l:
                        continue
                    attname = self._rev_names.get(att)
                    if attname is None:
                        continue
                    if attname==name:
                        continue
                    res[name][tuple(p.x)]=self._point_size_hash[p] #type:ignore
        return res

    def sphere(self, origin:Point, radius:float=1, surface_name:Optional[str]=None, mesh_size:Optional[float]=None)->Any:
        if self._geom is None:
            raise RuntimeError("Can only add geometry inside the function 'define_geometry'")
        ball = self._geom.add_ball([x for x in origin.x], radius, mesh_size=mesh_size) #type:ignore
        if surface_name is not None:
            for entry in ball.surface_loop.surfaces: #type:ignore
                self._store_name(surface_name, entry) #type:ignore
                self._entities2d[entry._id] = entry #type:ignore
        self._maxdim = max(self._maxdim, 2)  ##TODO 2 or 1
        return ball

    def _sort_line_loop(self, lst:Sequence[Union[Line,Spline,BSpline,CircleArc]], name:Optional[str]=None) -> List[List[Union[Line , Spline , BSpline , CircleArc]]]:
        tocheck = [a for a in lst]
        currentelem = tocheck.pop(0)
        startpoint = currentelem.points[0] #type:ignore
        currentendpoint = currentelem.points[-1] #type:ignore
        gmshres = [currentelem]
        totalres:List[List[Union[Line,Spline,BSpline,CircleArc]]]=[]
        debug_info = [self._rev_names.get(currentelem, "<unnamed>")]
        while len(tocheck) > 0:
            found = False
            for ind, checkelem in enumerate(tocheck):
                if checkelem.points[0] == currentendpoint:
                    gmshres.append(checkelem)
                    currentendpoint = (checkelem.points[-1]) #type:ignore
                    debug_info.append(self._rev_names.get(checkelem, "<unnamed>"))
                    tocheck.pop(ind)
                    found = True
                    if  currentendpoint==startpoint and len(tocheck)>0:
                        totalres.append(gmshres) #type:ignore
                        currentelem = tocheck.pop(0)
                        startpoint = currentelem.points[0] #type:ignore
                        currentendpoint = currentelem.points[-1] #type:ignore
                        gmshres = [currentelem]
                    break
                elif checkelem.points[-1] == currentendpoint:
                    gmshres.append(-checkelem)
                    currentendpoint = (checkelem.points[0]) #type:ignore
                    debug_info.append("-" + self._rev_names.get(checkelem, "<unnamed>"))
                    tocheck.pop(ind)
                    found = True
                    if  currentendpoint==startpoint and len(tocheck)>0:
                        totalres.append(gmshres) #type:ignore
                        currentelem = tocheck.pop(0)
                        startpoint = currentelem.points[0] #type:ignore
                        currentendpoint = currentelem.points[-1] #type:ignore
                        gmshres = [currentelem]
                    break
            if not found:
                llist = map(lambda e: self._rev_names.get(e, "<unnamed>"), lst)
                mshdir = os.path.join(self._problem._outdir, "_gmsh") #type:ignore
                Path(mshdir).mkdir(parents=True, exist_ok=True)
                mshtrunk = "DEBUG"
                assert self._geom is not None
                generate_mesh_to_file(self._geom, mshdir, mshtrunk, mesher=self,dim=self._maxdim, order=self.order,
                                      algorithm=self.gmsh_options.get("algorithm",None),
                                      recombine_algo=self.gmsh_options.get("recombine_algo",None),
                                      postgen_cb=lambda: self.post_process(),mesh_mode=self.mesh_mode,mesh_size_callback=self._mesh_size_callback)
                raise RuntimeError("Cannot close line loop" + (
                    "" if name is None else " for surface " + name) + ". Cannot find the next element in the loop.\nLoop so far: " + (
                                       "\n".join(debug_info)) + "\n\nLine list:\n" + "\n".join(llist))
        if currentendpoint != startpoint:
            llist = map(lambda e: self._rev_names.get(e, "<unnamed>"), lst)
            mshdir = os.path.join(self._problem._outdir, "_gmsh") #type:ignore
            Path(mshdir).mkdir(parents=True, exist_ok=True)
            mshtrunk = "DEBUG"
            assert self._geom is not None
            generate_mesh_to_file(self._geom, mshdir, mshtrunk, mesher=self, dim=self._maxdim, order=self.order,
                                  algorithm=self.gmsh_options.get("algorithm",None),
                                  recombine_algo=self.gmsh_options.get("recombine_algo",None),
                                  postgen_cb=lambda: self.post_process(),mesh_mode=self.mesh_mode,mesh_size_callback=self._mesh_size_callback)
            raise RuntimeError("Could not close line loop" + (
                "" if name is None else " for surface " + name) + ". Start and end not matching.\nLoop so far: " + (
                                   "\n".join(debug_info)) + "\n\nLine list:\n" + "\n".join(llist))

        totalres.append(gmshres)
        return totalres

    def set_gmsh_parameter(self, n:str, v:Union[float,int]):
        gmsh.option.setNumber(n, v) #type:ignore


    def plane_surface(self, *args:Union[str,Line,Spline,BSpline,CircleArc,None], name:Optional[str]=None,holes:Optional[List[Sequence[Union[str,Line,Spline,BSpline,CircleArc]]]]=None,reversed_order:bool=False) -> List[PlaneSurface]:
        """
        Creates a planar surface in the mesh. You must supply the enclosing boundaries (either by name or the line/circle_arc/spline/bspline objects) as arguments.
        If you give it a name, it can be used to add equations to the domain.

        Args:
            *args: Variable length arguments representing the lines or curves that define the boundary of the surface, in any order.
            name: Name of the surface for e.g. adding equations later on by the name.
            holes: List of holes within the surface, where each hole is defined by a sequence of lines or curves.
            reversed_order: Flag indicating whether to reverse the order of the lines or curves.

        Returns:
            List of created plane surfaces. Can be multiple, if you have e.g. multiple disjunct domains circumscripted by the given lines/splines/circle_arcs.
        """
        holesO = []
        if holes is not None:
            holes_names:List[List[str]]=[]
            revnamemap:Dict[object,str]={}
            for k,v in self._named_entities.items():
                for vv in v:
                    revnamemap[vv]=k
            for hole in holes:
                resolved = self._resolve_name("lines", *hole)                
                srted = self._sort_line_loop(resolved) #type:ignore                
                hole_names:List[str]=[]
                for s in srted:
                    ll = self._geom.add_curve_loop(s) #type:ignore
                    holesO.append(ll) #type:ignore
                    for ss in s:
                        if ss in revnamemap.keys():
                            ssn=revnamemap[ss]
                            if not ssn in hole_names:
                                hole_names.append(ssn)
                
                if len(hole_names)>0:
                    holes_names.append(hole_names)
            if len(holes_names)>0 and self.remesher is not None and name is not None:
                self.remesher.set_holes(name,holes_names)


        resolved = self._resolve_name("lines", *args)
        resolved = [l for l in resolved if l is not None]
        
        srted = self._sort_line_loop(resolved, name=name)  #type:ignore
        allres:List[PlaneSurface]=[]
        for s in srted:
            if reversed_order:
                s = list(reversed([-x for x in s]))
            ll = self._geom.add_curve_loop(s) #type:ignore
            #print("holes",name,holesO,ll.curves)
            is_hole = False
            for h in holesO:
                if h.curves == ll.curves:
                    print("Found hole in surface",ll,holesO)
                    is_hole = True
                    break
            if is_hole:
                continue
            res = self._geom.add_plane_surface(ll,holes=holesO) #type:ignore
            self._entities2d[res._id] = res #type:ignore
            if name is not None:
                self._store_name(name, res)
            if self.mesh_mode in ["quads","only_quads"]:
                self.set_recombined_surfaces(res)
            allres.append(res)
        self._maxdim = max(self._maxdim, 2)
        
        #print("resolved",allres,holesO)
        #if name=="liquid":
         #   exit()
        
        return allres

    def ruled_surface(self, *args:Union[str,Line,Spline,BSpline,CircleArc], name:Optional[str]=None,reversed_order:bool=False) -> List[Surface]:
        resolved = self._resolve_name("lines", *args)
        srted = self._sort_line_loop(resolved, name=name) #type:ignore
        allres:List[Surface] = []
        for s in srted:
            if reversed_order:
                s = list(reversed([-x for x in s]))
            ll = self._geom.add_curve_loop(s) #type:ignore
            res = self._geom.add_surface(ll) #type:ignore
            self._entities2d[res._id] = res #type:ignore
            if name is not None:
                self._store_name(name, res)
            if self.mesh_mode in ["quads","only_quads"]:
                self.set_recombined_surfaces(res)
            self._maxdim = max(self._maxdim, 2)
            allres.append(res)
        return allres

    def volume(self, *args:Union[str,Surface,PlaneSurface], name:Optional[str]=None,reversed_order:bool=False) -> List[Volume]:
        resolved = self._resolve_name("surfaces", *args) #type:ignore
        #print(resolved)
        srted=resolved # TODO: Sort?
        #srted=[s for l in resolved for s in l] #type:ignore
        #srted=
        #print(srted)
        #exit()
        allres:List[Volume]=[]
        #srted = self._sort_line_loop(resolved, name=kwargs.get("name"))
        #if kwargs.get("reversed", False) == True:
        #    srted = list(reversed([-x for x in srted]))
        s=srted #type:ignore
        
        #if reversed_order:
        #    for entry in s:
        #        entry._id=-entry._id
        ll = self._geom.add_surface_loop(s) #type:ignore
        
        res = self._geom.add_volume(ll) #type:ignore
        #if reversed_order:
        #    for entry in s:
        #        entry._id=-entry._id
        #print(res)
        if name is not None:
            self._store_name(name, res)
#        if self.mesh_mode in ["quads"]:
#            self.set_recombined_surfaces(res)
        # TODO: Mesh mode
        self._maxdim = max(self._maxdim, 3)
        allres.append(res)
        #exit()
        return allres

    def set_recombined_surfaces(self, surfs:Union[List[Union[PlaneSurface,Surface,str]],str,PlaneSurface,Surface]):
        if isinstance(surfs, list):
            for e in surfs:
                self.set_recombined_surfaces(e)
        elif isinstance(surfs, str):
            resolve = self._resolve_name("surfaces", surfs)
            self.set_recombined_surfaces(resolve) #type:ignore
        else:
            self._geom.set_recombined_surfaces([surfs]) #type:ignore

    def define_geometry(self):
        """
        Override specifically to define the geometry of this mesh by adding points, lines, surfaces, etc.
        """
        pass

    def post_process(self):
        #gmsh.model.mesh.refine()
        #gmsh.model.mesh.recombine()
        pass



    def process_cells_for_optional_mirroring(self,cells):
        if self.mirror_mesh is not None:            
            mirrors=self.mirror_mesh
            if not isinstance(mirrors,list):
                mirrors=[mirrors]
            for i,direct in enumerate(mirrors):
                cells=numpy.r_[cells.copy(),cells+self._mirror_index_shift[i]]
        return cells
    
    # Must return process cells and potentially, if nodes might be overlapping, True
    def process_points_for_optional_mirroring(self,points):
        reindex=False
        if self.mirror_mesh is not None:
            self._mirror_index_shift=[]
            mirrors=self.mirror_mesh
            if not isinstance(mirrors,list):
                mirrors=[mirrors]
            for i,direct in enumerate(mirrors):
                reindex=True
                self._mirror_index_shift.append(len(points)) # Store how many points are in half the mesh
                mirrvec=[1,1,1]
                if direct=="mirror_x":
                    mirrvec[0]=-1
                elif direct=="mirror_y":
                    mirrvec[1]=-1
                elif direct=="mirror_z":
                    mirrvec[2]=-1
                points=numpy.r_[points.copy(),points*numpy.array([mirrvec])]
        if self.extrude_generated_mesh is not None:
            if self.mirror_mesh is not None:
                raise RuntimeError("Cannot extrude and mirror the mesh yet at the same time")
            dist=self.extrude_generated_mesh[0]
            dist=self.nondim_size(dist)
            layers=int(self.extrude_generated_mesh[1])
            if layers<1:
                raise RuntimeError("Extrusion must have at least 1 layer")
            
            if self._maxdim==2 and self._max_elem_dim==2:
                if self.order==1:
                    zs=numpy.linspace(0,dist,layers+1,endpoint=True)
                elif self.order==2:
                    zs=numpy.linspace(0,dist,layers*2+1,endpoint=True)
                numpoints=len(points)
                points=numpy.transpose(numpy.tile(numpy.transpose(points),len(zs)))
                zcoords=numpy.repeat(zs,numpoints)
                points[:,2]=zcoords
                self._maxdim=3
                self._max_elem_dim=3
            else:
                raise RuntimeError("Implement mesh extrusion for nodal dim "+str(self._maxdim)+" and element dim "+str(self._max_elem_dim))
            
                
        return points,reindex
    
    
    def _post_extrude_mesh(self):
        raise RuntimeError("TODO: Implement post extrusion...")
        # We now have to cast all the cells to one dimension higher
        newcells=[]
        newcell_set={}
        print(type(self._mesh.cell_sets))
        for name, entry in self._mesh.cell_sets.items(): #type:ignore
            if name == "gmsh:bounding_entities":
                print("SKIPPING BOUNDING ENTITIES",entry)
                continue
            print("name",name,entry)
            
        self._mesh.cells=newcells
        self._mesh.cell_sets=newcell_set
        exit()
    
    def _load_mesh(self,mshfilename:str):
        print("Loading mesh file: "+mshfilename)
        self._mesh = meshio.read(mshfilename, file_format="gmsh") #type:ignore
        curvedfile,_=os.path.splitext(mshfilename)
        curvedfile=curvedfile+".geo_unrolled"
        self.read_curved_entities(curvedfile)
        # Find the maximum element dimension. All domains of this dimension will be considered to be bulk domains, the rest are interfaces
        maxeldim = -1
        named_eldims = {"line": 1, "line3": 1, "quad": 2, "quad9": 2, "triangle": 2, "triangle6": 2, "hexahedron27": 3,
                        "hexahedron": 3, "vertex": 0, "tetra10":3,"tetra":3}
        for name, entry in self._mesh.cell_sets.items(): #type:ignore
            if name == "gmsh:bounding_entities":
                continue
            myeldim = None
            for i, idx in enumerate(entry): #type:ignore
                if len(idx): #type:ignore
                    cells = self._mesh.cells[i] #type:ignore
                    if not cells.type in named_eldims.keys(): #type:ignore
                        raise RuntimeError("Unknown cell type: " + str(cells.type) + " in physical group " + name) #type:ignore
                    maxeldim = max(maxeldim, named_eldims[cells.type]) #type:ignore
                    if myeldim is None:
                        myeldim = named_eldims[cells.type] #type:ignore
                    elif myeldim != named_eldims[cells.type]: #type:ignore
                        raise RuntimeError(
                            "The physical group " + name + " has elements of different dimensions, namely at least " + str(
                                myeldim) + " and " + str(named_eldims[cells.type])) #type:ignore

        self._max_elem_dim = maxeldim

        # Create the points
        self._nodeinds:List[int] = []
        _nodal_dim = 1
        mesh_points=self._mesh.points
        mesh_points,unique_adding=self.process_points_for_optional_mirroring(mesh_points)        
        for i, p in enumerate(mesh_points): #type:ignore
            if unique_adding:
                self._nodeinds.append(self.add_node_unique(p[0],p[1],p[2]))
            else:
                self._nodeinds.append(self.add_node(p[0], p[1], p[2])) #type:ignore
            if p[1] * p[1] > 1e-15 and _nodal_dim < 2:
                _nodal_dim = 2
            if p[2] * p[2] > 1e-15 and _nodal_dim < 3:
                _nodal_dim = 3
        self._max_nodal_dim = _nodal_dim
        self._nodeinds = numpy.array(self._nodeinds) #type:ignore
        
        if self.extrude_generated_mesh is not None:
            self._post_extrude_mesh()

        for name, entry in self._mesh.cell_sets.items(): #type:ignore
            if name == "gmsh:bounding_entities":
                continue
            mydim=None
            for i, idx in enumerate(entry): #type:ignore
                mydim = -1
                if len(idx): #type:ignore
                    cells = self._mesh.cells[i] #type:ignore
                    if not cells.type in named_eldims.keys(): #type:ignore
                        raise RuntimeError("Unknown cell type: " + str(cells.type) + " in physical group " + name) #type:ignore
                    mydim = named_eldims[cells.type] #type:ignore
                    break
            assert mydim is not None
            if mydim == self._max_elem_dim:
                if self._max_elem_dim == 2:
                    self._construct_template_domain_2d(name, entry) #type:ignore
                elif self._max_elem_dim == 1:
                    self._construct_template_domain_1d(name, entry) #type:ignore
                elif self._max_elem_dim == 3:
                    self._construct_template_domain_3d(name, entry) #type:ignore
                else:
                    raise RuntimeError("IMPLEMENT For element dimension " + str(mydim))

        self._geometry_defined = True
        if self.auto_find_opposite_interface_connections:
            self._find_opposite_interface_connections()


    def _do_define_geometry(self, problem:"Problem", filename_trunk:Optional[str]=None):
        self._problem=problem
        if not self._geometry_defined:
            mshdir = os.path.join(self._problem._outdir, "_gmsh") #type:ignore
            Path(mshdir).mkdir(parents=True, exist_ok=True)
            # Find a unique name:
            if filename_trunk is None:
                cnt:Optional[int] = None
                mshtrunk = self.__class__.__name__
                for mt in problem._meshtemplate_list: #type:ignore
                    if isinstance(mt, GmshTemplate):
                        cn = mt.__class__.__name__
                        if cn == self.__class__.__name__:
                            if mt is self:
                                if cnt is not None:
                                    mshtrunk = mshtrunk + "_" + str(cnt)
                                break
                            elif cnt is None:
                                cnt = 1
                            else:
                                assert isinstance(cnt,int)
                                cnt = cnt + 1
            else:
                mshtrunk = filename_trunk
            self._fntrunk = mshtrunk

            if self._loaded_from_mesh_file is None:
                with pygmsh.geo.Geometry(["-noenv"]) as geom:
                    self._geom = geom
                    super(GmshTemplate, self)._do_define_geometry(problem)
                    for name, objlist in self._named_entities.items():
                        geom.add_physical(objlist, label=name) #type:ignore

                    self._meshfile=os.path.join(mshdir, mshtrunk + ".msh")
                    if ((self._problem._runmode!="continue" or self._problem._continue_initialized) and self._problem._runmode!="replot") or not os.path.exists(self._meshfile): #type:ignore
                        generate_mesh_to_file(geom, mshdir, mshtrunk, mesher=self, dim=self._maxdim, order=self.order,
                                          algorithm=self.gmsh_options.get("algorithm",None),
                                          recombine_algo=self.gmsh_options.get("recombine_algo",None),
                                          postgen_cb=lambda: self.post_process(),mesh_mode=self.mesh_mode,mesh_size_callback=self._mesh_size_callback)

                        #self.write_curved_entities(os.path.join(mshdir, mshtrunk + ".curved"))
            else:
                super(GmshTemplate, self)._do_define_geometry(problem)
            if self._loaded_from_mesh_file:
                self._meshfile=self._loaded_from_mesh_file
            #print("BEFORE LOAD",self._loaded_from_mesh_file,self._meshfile)
            assert self._meshfile is not None
            self._load_mesh(self._meshfile)
            self._loaded_from_mesh_file = None


    def _construct_template_domain_3d(self, name:str, entry:Any):
        domain = self.new_domain(name)
        if self.all_nodes_as_boundary_nodes:
            domain.set_all_nodes_as_boundary_nodes()
        for i, idx in enumerate(entry):
            if len(idx):
                cells = self._mesh.cells[i]
                mycells = cells.data[idx]
                if cells.type == "tetra10":
                    #perm = [0, 1, 2,3,4,5,6,7,8,9]
                    #perm = [0, 1, 2, 3, 4,6,7,5,9,8]
                    perm = [0, 2, 1, 3, 6, 4, 7, 5, 8, 9]
                    for q in mycells:
                        domain.add_tetra_3d_C2(self._nodeinds[q[perm]]) #type:ignore
                elif cells.type == "tetra":
                    #perm = [0, 1, 2,3]
                    perm = [0, 2,1, 3]
                    for q in mycells:
                        domain.add_tetra_3d_C1(*self._nodeinds[q[perm]]) #type:ignore
                elif cells.type =="hexahedron":
                    perm=[1,2,0,3,5,6,4,7]
                    for q in mycells:
                        domain.add_brick_3d_C1(*self._nodeinds[q[perm]]) #type:ignore
                elif cells.type =="hexahedron27":
                    #perm=[1,2,0,3,5,6,4,7]
                    perm=[1,9,2,8,24,10,0,11,3,17,21,18,22,26,23,16,20,19,5,13,6,12,25,14,4,15,7]
                    for q in mycells:
                        domain.add_brick_3d_C2(self._nodeinds[q[perm]]) #type:ignore
                else:
                    raise RuntimeError("Unsupported cell type: " + cells.type)

        domain.set_nodal_dimension(self._max_nodal_dim)
        domain.set_lagrangian_dimension(self._max_nodal_dim)

        no_macro_elements = not self.use_macro_elements

        for name, cs in self._mesh.cell_sets.items():
            if name == "gmsh:bounding_entities": continue
            for i, idx in enumerate(cs):
                if len(idx) > 0:
                    cells = self._mesh.cells[i]
                    if cells.type == "triangle" or cells.type == "triangle6" or cells.type=="quad" or cells.type=="quad9":
                        mycells = cells.data[idx]
                        mygeoms = self._mesh.cell_data["gmsh:geometrical"][i][idx]
                        for li, l in enumerate(mycells): #type:ignore 
                            ninds:List[int] = self._nodeinds[l] #type:ignore 
                            if -1 in ninds:  # Do only consider lines full inside
                                continue
                            if no_macro_elements:
                                self.add_nodes_to_boundary(name, ninds)
                            else:
                                curved = self._curved_entities2d.get(mygeoms[li])
                                if curved:
                                    self._has_curved_entries = True
                                #									print("MARKIN MACRO",ninds)
                                self.add_nodes_to_boundary(name, ninds)
                                if curved:
                                    raise RuntimeError("CURVED 3d Bounds")
                                    self.add_facet_to_curve_entity(ninds[[0, 1]], curved)



    def _construct_template_domain_2d(self, name:str, entry:Any):    
        domain = self.new_domain(name)
        if self.all_nodes_as_boundary_nodes:
            domain.set_all_nodes_as_boundary_nodes()
        for i, idx in enumerate(entry): #type:ignore
            if len(idx): #type:ignore
                cells = self._mesh.cells[i]
                mycells = cells.data[idx]
                mycells=self.process_cells_for_optional_mirroring(mycells)
                if cells.type == "quad":
                    perm = [0, 1, 3, 2]
                    for q in mycells:
                        domain.add_quad_2d_C1(*self._nodeinds[q[perm]]) #type:ignore
                elif cells.type == "quad9":
                    perm = [0, 4, 1, 7, 8, 5, 3, 6, 2]

                    for q in mycells:
                        domain.add_quad_2d_C2(*self._nodeinds[q[perm]]) #type:ignore
                elif cells.type == "triangle":
                    perm = [0, 1, 2]
                    if self.mesh_mode!="SV":
                        for t in mycells:
                            domain.add_tri_2d_C1(*self._nodeinds[t[perm]]) #type:ignore
                    else:
                        for t in mycells:
                            domain.add_SV_tri_2d_C1(*self._nodeinds[t[perm]]) #type:ignore
                            #c1,c2,c3=self._nodeinds[t[perm]]                            
                            #bc_pos=(numpy.array(self.get_node_position(c1))+numpy.array(self.get_node_position(c2))+numpy.array(self.get_node_position(c3)))/3.0
                            #bc=self.add_node_unique(*bc_pos)
                            #domain.add_tri_2d_C1(c1,c2,bc) #type:ignore
                            #domain.add_tri_2d_C1(c2,c3,bc) #type:ignore
                            #domain.add_tri_2d_C1(c3,c1,bc) #type:ignore
                elif cells.type == "triangle6":                    

                    perm = [0, 1, 2,3,4,5]
                    for t in mycells:
                        domain.add_tri_2d_C2(*self._nodeinds[t[perm]]) #type:ignore
                else:
                    raise RuntimeError("Unsupported cell type: " + cells.type)

        domain.set_nodal_dimension(self._max_nodal_dim)
        domain.set_lagrangian_dimension(self._max_nodal_dim)

        no_macro_elements = not self.use_macro_elements


        #print(dir(self._mesh))
        #print(self._mesh)
        #print(self._mesh.cell_data["gmsh:physical"])
        #print(self._mesh.point_data["gmsh:dim_tags"])
        #print(self._mesh.cell_sets["gmsh:bounding_entities"])
        #exit()

        for name, cs in self._mesh.cell_sets.items():
            if name == "gmsh:bounding_entities": continue
            for i, idx in enumerate(cs):
                if len(idx) > 0:
                    cells = self._mesh.cells[i]
                    if cells.type == "line" or cells.type == "line3":
                        mycells = cells.data[idx]
                        mycells=self.process_cells_for_optional_mirroring(mycells)
                        mygeoms = self._mesh.cell_data["gmsh:geometrical"][i][idx]
                        for li, l in enumerate(mycells):
                            ninds = self._nodeinds[l] #type:ignore
                            if -1 in ninds:  #type:ignore # Do only consider lines full inside
                                continue
                            if no_macro_elements:
                                self.add_nodes_to_boundary(name, ninds) #type:ignore
                            else:
                                curved = self._curved_entities1d.get(mygeoms[li])
                                if curved:
                                    self._has_curved_entries = True
                                #									print("MARKIN MACRO",ninds)
                                self.add_nodes_to_boundary(name, ninds) #type:ignore
                                if curved: 
                                    self.add_facet_to_curve_entity(ninds[[0, 1]], curved) #type:ignore
                    #elif cells.type=="vertex":
                    #    mycells = cells.data[idx]
                    #    mygeoms = self._mesh.cell_data["gmsh:geometrical"][i][idx]
                    #    for li, l in enumerate(mycells):
                    #        ninds = self._nodeinds[l]
                    #        self.add_nodes_to_boundary(name, ninds)
                    #        print(name,ninds)
                    #        exit()


    def _construct_template_domain_1d(self, name:str, entry:Any):
        domain = self.new_domain(name)
        if self.all_nodes_as_boundary_nodes:
            domain.set_all_nodes_as_boundary_nodes()
        for i, idx in enumerate(entry): #type:ignore
            if len(idx):
                cells = self._mesh.cells[i]
                mycells = cells.data[idx]
                if cells.type == "line":
                    perm = [0, 1]
                    for q in mycells:
                        domain.add_line_1d_C1(*self._nodeinds[q[perm]]) #type:ignore
                elif cells.type == "line3":
                    perm = [0, 2, 1]
                    for q in mycells:
                        domain.add_line_1d_C2(*self._nodeinds[q[perm]]) #type:ignore
                else:
                    raise RuntimeError("Unsupported cell type: " + cells.type)

        domain.set_nodal_dimension(self._max_nodal_dim)
        domain.set_lagrangian_dimension(self._max_nodal_dim)

        no_macro_elements = False  # TODO

        for name, cs in self._mesh.cell_sets.items():
            if name == "gmsh:bounding_entities": continue
            for i, idx in enumerate(cs):
                if len(idx) > 0:
                    cells = self._mesh.cells[i]
                    if cells.type == "vertex":
                        mycells = cells.data[idx]
                        mygeoms = self._mesh.cell_data["gmsh:geometrical"][i][idx]
                        for li, l in enumerate(mycells):
                            ninds = self._nodeinds[l] #type:ignore
                            if -1 in ninds:  #type:ignore # Do only consider lines full inside
                                continue
                            if no_macro_elements:
                                self.add_nodes_to_boundary(name, ninds)  #type:ignore
                            else:
                                curved = self._curved_entities0d.get(mygeoms[li]) #type:ignore
                                if curved:
                                    self._has_curved_entries = True
                                #									print("MARKIN MACRO",ninds)
                                self.add_nodes_to_boundary(name, ninds) #type:ignore
                                if curved:
                                    self.add_facet_to_curve_entity(ninds, curved) #type:ignore


    def write_curved_entities(self,fname:str):
        fout=open(fname,"w")
        fout.write(str(len(self._curved_entities1d)) + "\n")
        for i,e in self._curved_entities1d.items():
            fout.write(str(i)+"\n")
            fout.write(str(e.__class__.__name__)+"\n")
            fout.write(e.get_information_string())
        fout.close()

    def read_curved_entities(self,fname:str):
        self._curved_entities1d = {}
        try:
            geo = Path(fname).read_text()
        except:
            return
        geo=geo.replace("\n","")
        geo = geo.replace("\r", "")
        cmds=geo.split(";")



        points={}
        #num_patt = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
        #num_patt="[-+]?\d*\.?\d+|[-+]?\d+"
        num_patt=r"[-+]?(\d+([.]\d*)?|[.]\d+)([eE][-+]?\d+)?"
        for c in cmds:
            if c.startswith("Point("):
                match=re.search(r"\s*Point\(\s*(?P<index>\d*)\s*\)\s*=\s*\{\s*(?P<x>"+num_patt+r")\s*,\s*(?P<y>"+num_patt+r")\s*,\s*(?P<z>"+num_patt+r")\s*",c)
                if not match:
                    raise RuntimeError("Cannot match Point at "+c)
                ind=int(match.group("index"))
                pos=list(map(float,match.group("x","y","z")))
                points[ind]=pos
            elif c.startswith("Circle"):
                match = re.search(r"\s*Circle\(\s*(?P<index>\d*)\s*\)\s*=\s*\{\s*(?P<P1>\d*)\s*,\s*(?P<P2>\d*)\s*,\s*(?P<P3>\d*)\s*",c)
                if not match:
                    raise RuntimeError("Cannot match Circle at " + c)
                ind = int(match.group("index"))
                PS=list(map(int,match.group("P1","P2","P3")))
                center,startpt,endpt=points[PS[1]],points[PS[0]],points[PS[2]] #type:ignore
                self._curved_entities1d[ind]=_pyoomph.CurvedEntityCircleArc(center, startpt, endpt) #type:ignore
            elif c.startswith("Spline"):
                match=re.search(r"\s*Spline\(\s*(?P<index>\d*)\s*\)\s*=\s*\{\s*(?P<list>(?:\d+,\s*)+\d+\s*)",c)
                if not match:
                    raise RuntimeError("Cannot match Spline at "+c)
                ind = int(match.group("index"))
                lst=match.group("list").replace(" ","").replace("\t","")
                lst=list(map(int,lst.split(",")))
                PTS=numpy.array([points[l] for l in lst]) #type:ignore
                self._curved_entities1d[ind]=_pyoomph.CurvedEntityCatmullRomSpline(PTS) #type:ignore



    class GmshFakeEntry:
        """ Just a fake entry to store the dimension and tag of an entity """
        def __init__(self,my_id,dim_tag):
            self._id=my_id
            self.dim_tag=dim_tag
            self.dim_tags=[dim_tag]
            self.dim=dim_tag[0]
                
            
    def extrude(self,*args,shift:List[ExpressionOrNum]=[0,0,1],recombine:bool=False,start_name=lambda s: s+"_start",end_name=lambda s: s+"_end",layers:Optional[int]=None):
        """Extrudes the given entities by the given shift. The bulk surface name will become a volume and the line surfaces will become 2d surfaces.
        Additionally, the start and end surfaces of the extrusion will be named according to the given functions.
        
        Args:
            *args: Variable length arguments representing the entities to extrude. Can be names or the entities themselves.
            shift: The shift to extrude by. Can be a list of expressions or numbers.
            recombine: Flag indicating whether to recombine the extruded entities.
            start_name: Function to generate the start name of the extrusion.
            end_name: Function to generate the end name of the extrusion.
            layers: Number of layers to extrude. If None, the extrusion will be determined by the mesh size.        
        """
        for i, c in enumerate(shift):
            c = c / self._problem.get_scaling("spatial")
            if isinstance(c,Expression):
                c=c.float_value()
            shift[i] = c
        gmsh.model.geo.synchronize()
        dimtags=[]
        newdim=0
        name_list=[]
        to_extrude=[]
        def add_name(a):
            name_list.append(self._rev_names.get(a,None))
            if a in self._rev_names:
                del self._named_entities[self._rev_names[a]]
                del self._rev_names[a]
                self._store_name(start_name(name_list[-1]),a)
        for a in args:
            if isinstance(a,list):
                for aa in a:
                    dimtags.append(aa.dim_tag)
                    to_extrude.append(aa)
                    add_name(aa)
            elif isinstance(a,str):
                raise RuntimeError("Not implemented: Supporting strings as names here")
                a=gmsh.model.getEntitiesForPhysicalGroup(2,self.get_physical_group(a))[0]
                dimtags.append(a)
                to_extrude.append(a)
                
            else:
                dimtags.append(a.dim_tag)
                to_extrude.append(a)
                add_name(a)
            newdim=max(newdim,dimtags[-1][0])

        newdim+=1 # The new dimension
        self._maxdim=max(self._maxdim,newdim)
        res=gmsh.model.geo.extrude(dimtags,shift[0],shift[1],shift[2],recombine=recombine,numElements=([layers]*len(dimtags) if layers is not None else None))        
        
        gmsh.model.geo.synchronize()
        bulk_name_index=0
        for i,entry in enumerate(res):            
            #print("ADD PHYSICAL GROUP",entry[0],[entry[1]])
            if entry[0]==newdim:
                name=name_list[bulk_name_index]
                if name is not None:
                    self._store_name(name,GmshTemplate.GmshFakeEntry(entry[1],(entry[0],entry[1])))
                    assert i>=1
                    self._store_name(end_name(name),GmshTemplate.GmshFakeEntry(res[i-1][1],(res[i-1][0],res[i-1][1])))                
                bulk_name_index+=1
            elif entry[0]==newdim-1:
                pass

        # Go over it once more, finding the missing entities
        given_names={a._id for a in self._rev_names}
        start_index=-1
        
        for entry in res:
            if entry[0]==newdim:
                start_index+=1
                sub_index=0
            if entry[0]==newdim-1:
                if entry[1] not in given_names:
                    original=to_extrude[start_index]                    
                    if isinstance(original,PlaneSurface):
                        orig_curv=original.curve_loop.curves[sub_index]
                        dim_tag=(orig_curv.dim_tag[0],abs(orig_curv.dim_tag[1]))
                        if self._dim_tag_names.get(dim_tag,None) is not None:
                            #print("Already exists",dim_tag,self._dim_tag_names.get(dim_tag,None))
                            del self._rev_names[self._dim_tag_names[dim_tag][1]]
                            del self._named_entities[self._dim_tag_names[dim_tag][0]]                                                        
                            self._store_name(self._dim_tag_names[dim_tag][0],GmshTemplate.GmshFakeEntry(entry[1],entry))
                            del self._dim_tag_names[dim_tag]
                    else:
                        raise RuntimeError("Not implemented:"+str(original  ))                        
                    
                    sub_index+=1                    
        

        return res