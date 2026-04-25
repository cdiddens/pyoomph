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
 


import math

from ..typings import *
import numpy


import _pyoomph

from .gmsh import GmshTemplate, Point, Line,Spline
from .mesh import MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d,MeshTemplate


class RemesherPointEntry:
    def __init__(self,x:float,y:float,z:float,size:float):
        self.x,self.y,self.z,self.size=x,y,z,size
        self.set_sizes:List[float]=[] # Sizes can be modified
        #self.on_bounds=set()
        self.gmsh_point:Optional[Point]=None

    def get_size(self) -> float:
        if len(self.set_sizes)==0:
            return self.size
        else:
            return sum(self.set_sizes)/len(self.set_sizes)



class RemesherLineEntry:
    def __init__(self,ptlist:List[RemesherPointEntry],mode:str,bname:str):
        self.ptlist=ptlist
        self.mode=mode
        self.gmsh_line:Optional[Union[Line,Spline]]=None
        self.bname=bname



class RemesherBase:
    def __init__(self,template:"MeshTemplate"):
        self.template=template
        self._cnt:int=0        
        #self._point_entries = {}
        self._line_entries:List[RemesherLineEntry] = []
        self._unique_pts:List[RemesherPointEntry]=[]
        self._old_meshes:Dict[str,Union[MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d]]={}
        #self._domain_points={} # access the points via domain names

    def add_point_entry(self,x:float,y:float,z:float,size:float) -> RemesherPointEntry:
        for p in self._unique_pts:
            if abs(p.x-x)<1e-9 and abs(p.y-y)<1e-9 and abs(p.z-z)<1e-9:
                return p
        else:
            res=RemesherPointEntry(x,y,z,size)
            self._unique_pts.append(res)
            return res

    def add_line_entry(self,ptlist:List[RemesherPointEntry],mode:str,bname:str):
        self._line_entries.append(RemesherLineEntry(ptlist,mode,bname))

    def _get_points_by_phys_name(self,name:str)->List[List[RemesherPointEntry]]:
        raise RuntimeError("Implement")

    def actions_after_remeshing(self):
        self._line_entries = []
        self._unique_pts = []
        #self._domain_points:Dict[str,Dict[str,List[Node]]] = {}
        pass

    def remesh(self):
        pass

    def replace_old_with_new_meshes(self):
        raise RuntimeError("Implement")

    def get_new_template(self)->"MeshTemplate":
        raise RuntimeError("Implement")


class GmshRemesher2d(GmshTemplate):
    def __init__(self,remesher:RemesherBase):
        super(GmshRemesher2d, self).__init__()
        self.remesher=remesher

    def define_geometry(self):
        assert self.remesher is not None
        assert isinstance(self.remesher,Remesher2d)
        if isinstance(self.remesher.template,GmshTemplate):
            self.mesh_mode=self.remesher.template.mesh_mode #TODO: Optionally also copy other props
            self.use_macro_elements=self.remesher.template.use_macro_elements
            self.gmsh_options=self.remesher.template.gmsh_options.copy()
        self.remesher._define_geometry() 

class Remesher2dBoundaryLineCollection:
    def __init__(self,boundname:str,remesher:"Remesher2d",point_size_func:Optional[Callable[[float,float],float]]=None):
        super(Remesher2dBoundaryLineCollection, self).__init__()
        self.name=boundname
        self.parts=[]
        self.oldnodes:Dict[Tuple[_pyoomph.Node,_pyoomph.Node],List[_pyoomph.Node]]= {} #Dict mapping from a pair of vertex nodes to the non-vertex nodes in between
        self.curves:List[List[_pyoomph.Node]]=[]
        self._node_to_bound_elems:Dict[_pyoomph.Node,Set[_pyoomph.OomphGeneralisedElement]] = {}
        self.point_size_func=point_size_func
        self.remesher=remesher


    def split_into_curves(self): #A boundary may contain more than one subcurve
        self.curves = []
        neighb_connects:Dict[_pyoomph.Node,List[_pyoomph.Node]]={} # A dict mapping to a list of node neighbors
        #print("OLDNODES",self.oldnodes)
        for n1,n2 in self.oldnodes.keys():
            neighb_connects.setdefault(n1, []).append(n2)
            neighb_connects.setdefault(n2, []).append(n1)

        while len(neighb_connects)>0:
            for n,neighs in neighb_connects.items():
                if len(neighs)==1:
                    startnode:_pyoomph.Node=n
                    break
            else:
                startnode:_pyoomph.Node=next(iter(neighb_connects.keys())) #type:ignore #Just any node. Seems to be looped

            currentcurve:List[_pyoomph.Node]=[]
            currentnode=startnode

            while len(neighb_connects)>0:
                while True:
                    #print(self.name,len(self.curves),len(neighb_connects))
                    currentcurve.append(currentnode)
                    if len(neighb_connects.get(currentnode,[]))==0:
                        for n, neighs in neighb_connects.items():
                            if len(neighs) == 1:
                                startnode = n
                                break
                        else:
                            if len(neighb_connects) == 0:
                                break
                            startnode = next(iter(neighb_connects.keys())) #type:ignore # Just any node. Seems to be looped
                        #print("ADD MODE 1",len(currentcurve))
                        self.curves.append(currentcurve)
                        currentcurve = []
                        currentnode = startnode
                        break
                    nextnode=neighb_connects[currentnode][0]
                    neighb_connects[currentnode].remove(nextnode)
                    if len(neighb_connects[currentnode])==0:
                        neighb_connects.pop(currentnode)
                    neighb_connects[nextnode].remove(currentnode)
                    if len(neighb_connects[nextnode])==0:
                        neighb_connects.pop(nextnode)
                    inbetween=self.oldnodes.get((currentnode,nextnode,),self.oldnodes.get((nextnode,currentnode,),None))
                    if inbetween is not None:
                        for i in inbetween:
                            currentcurve.append(i)
                    currentnode=nextnode
                    if currentnode==startnode:
                        currentcurve.append(startnode) # Indicate a loop
                        break
                #print("ADD MODE 3",len(currentcurve))
                if len(currentcurve)>0:
                    self.curves.append(currentcurve)
                currentcurve = []


    def get_size_at_point(self,p:_pyoomph.Node) -> float:
        if self.point_size_func is not None:
            if callable(self.point_size_func):
                return self.point_size_func(p.x(0),p.x(1))
            else:
                return self.point_size_func
        elif p in self.remesher._ptsizes.keys():
            return self.remesher._ptsizes[p]
        elif p in self._node_to_bound_elems.keys():
            avg_i=0.0
            avg_c = 0.0
            cnt=0
            for e in self._node_to_bound_elems[p]:
                lvl=e.refinement_level()
                scal:int=2**lvl
                avg_i+=math.sqrt(e.get_initial_cartesian_nondim_size())*scal
                avg_c+=math.sqrt(e.get_current_cartesian_nondim_size())*scal
                cnt+=1
            assert cnt!=0
            avg_i/=cnt
            avg_c /= cnt
            #print("AVG",avg_c,avg_i)
            return 1*(avg_i+avg_c)/2 #* 2 # Again *2 if refine is used to generate the quads
            #return math.sqrt(avg_i)
        return 1.0



    def create_entries(self):
        cmapI = self.remesher._corner_size_map #type:ignore
        if cmapI is not None:
            cmap=cmapI[self.name]
        else:
            cmap=None
        for c in self.curves:
            coords = numpy.array([[c[i].x(0), c[i].x(1), 0.0] for i in range(len(c))]) #type:ignore            
            isline = False
            if c[0] != c[-1]:
                dx = c[-1].x(0) - c[0].x(0)
                dy = c[-1].x(1) - c[0].x(1)
                d = math.sqrt(dx * dx + dy * dy)
                dx /= d
                dy /= d
                isline = numpy.allclose((coords[:, 0] - coords[0, 0]) * dy - (coords[:, 1] - coords[0, 1]) * dx, 0) #type:ignore

            sizes = None
            if self.point_size_func is None and cmap is not None:
                # Use the cmap
                # Find start and end
                mindist = 1e20
                minsize = None
                for p, ptsize in cmap.items():
                    #print("INFOx",coords[0][0],p[0])
                    #print("INFOy",coords[0][1],p[1])
                    d = (coords[0][0] - p[0]) ** 2 + (coords[0][1] - p[1]) ** 2
                    if d < mindist:
                        mindist = d
                        minsize = ptsize
                startsize = minsize
                mindist = 1e20
                minsize = None
                for p, ptsize in cmap.items():
                    d = (coords[-1][0] - p[0]) ** 2 + (coords[-1][1] - p[1]) ** 2
                    if d < mindist:
                        mindist = d
                        minsize = ptsize
                endsize = minsize
                if startsize is not None and endsize is not None:
                    arclength = numpy.zeros([len(coords)]) #type:ignore
                    last = coords[0]
                    acclength = 0.0
                    for i in range(len(arclength)):
                        dl = numpy.sqrt((last[0] - coords[i][0]) ** 2 + (last[1] - coords[i][1]) ** 2)
                        arclength[i] = acclength + dl
                        acclength += dl
                        last = coords[i]
                    arclength /= acclength
                    sizes = (1 - arclength) * startsize + arclength * endsize

            if sizes is None:
                sizes = [self.get_size_at_point(c[i]) for i in range(len(c))]

            if isline:
                # TODO: Check size variations and possibliy add multiple lines
                plst = [self.remesher.add_point_entry(c[i].x(0), c[i].x(1),0, size=sizes[i]) for i in [0, len(c) - 1]]
                self.remesher.add_line_entry(plst, "line",self.name)
            else:
                plst = [self.remesher.add_point_entry(c[i].x(0), c[i].x(1),0, size=sizes[i]) for i in range(len(c))]
                self.remesher.add_line_entry(plst, "spline",self.name)






class Remesher2d(RemesherBase):
    """
    A class to allow remeshing of 2d meshes by using Gmsh.
    You must set an instance of this class to the :py:attr:`~pyoomph.meshes.mesh.MeshTemplate.remesher` attribute of the :py:class:`~pyoomph.meshes.mesh.MeshTemplate`.
    
    Args:
        template: The mesh template to be remeshed. 
    """
    def __init__(self,template:MeshTemplate):
        super(Remesher2d, self).__init__(template)
        self._old_meshes={}
        self._boundary_nodes:Dict[str,Remesher2dBoundaryLineCollection]={}
        self.problem=None
        self.gmsh=GmshRemesher2d(self)
        self._meshbounds={}
        self._ptsizes:Dict[_pyoomph.Node,float]={}
        self._boundary_point_size_funcs:Dict[str,Callable[[float,float],float]]={}
        self.use_corner_sizes=True
        self._corner_size_map=None
        self._mesh_size_callback=None
        self._holes_info:Dict[str,List[List[str]]]={}

    def set_holes(self,domain:str,holes:List[List[str]]):
        self._holes_info[domain]=holes

    def set_boundary_point_size(self,**kwargs:Callable[[float,float],float]):
        for name,func_or_val in kwargs.items():
            self._boundary_point_size_funcs[name]=func_or_val


    def actions_after_remeshing(self):
        super(Remesher2d, self).actions_after_remeshing()
        self.gmsh = GmshRemesher2d(self) #Recreate the intenral gmsh remesher
        self._old_meshes={}
        self._meshbounds:Dict[str,List[str]]={}
        self._unique_pts = []

    def get_new_template(self)->MeshTemplate:
        return self.gmsh

    def _identify_domains(self):
        self._old_meshes={}
        assert self.problem is not None
        for k,m in self.problem._meshdict.items():
            if isinstance(m,MeshFromTemplate2d):
                if self.template.has_domain(k):
                    self._old_meshes[k]=m

    def _preprocess_domain(self,n:str):
        pass
        #mesh=self._old_meshes[n]
        #print(mesh.get_boundary_names())
        #print(dir(mesh))

    def _define_boundaries_for_domain(self,n:str):
        mesh=self._old_meshes[n]
        self._meshbounds[n]=[]
        for bn in mesh.get_boundary_names():
            ind=mesh.get_boundary_index(bn)
            nelem=mesh.nboundary_element(ind)
            if nelem==0: #TODO: These bounds could still be relevant
                continue

            self._meshbounds[n].append(bn)
            #if n not in self._domain_points.keys():
            #    self._domain_points[n] = {}
            #if bn not in self._domain_points[n].keys():
            #    self._domain_points[n][bn]={}

            if not (bn in self._boundary_nodes.keys()):
                self._boundary_nodes[bn]=Remesher2dBoundaryLineCollection(bn,self,point_size_func=self._boundary_point_size_funcs.get(bn,None))
                bnd = self._boundary_nodes[bn]
                for be,dir in mesh.boundary_elements(bn,with_directions=True):
                    internodes=[be.boundary_node_pt(dir,i) for i in range(be.nnode_1d())]
                    bnd.oldnodes[(internodes[0],internodes[-1],)]=internodes[1:-1]
                    for inode in internodes:
                        bnd._node_to_bound_elems.setdefault(inode, set()).add(be) #type:ignore
                bnd.split_into_curves()


    def _mesh_domain(self,n:str):
        mshbounds=self._meshbounds[n].copy()
        holes=self._holes_info.get(n,None)
        if holes is not None:
            for hole in holes:
                for iname in hole:
                    if iname in mshbounds:
                        mshbounds.remove(iname)
        self.gmsh.plane_surface(*mshbounds,name=n,holes=holes)


    def _define_geometry(self):
        mesh=self.gmsh
        for n in self._old_meshes.keys():
            self._define_boundaries_for_domain(n)

        for n,bn in self._boundary_nodes.items():
            bn.create_entries()


        # Here we can still modify everything
        # TODO
        assert self.problem is not None
        self.problem._equation_system._setup_remeshing_size(self,True)  # Preorder loop
        self.problem._equation_system._setup_remeshing_size(self, False)  # Post order loop


        # Add the points
        for p in self._unique_pts:
            p.gmsh_point=mesh.point(p.x,p.y,p.z,size=p.get_size(),consider_spatial_scale=False)

        # add the lines
        for l in self._line_entries:
            if l.mode=="line":
                p0=l.ptlist[0].gmsh_point
                p1 = l.ptlist[-1].gmsh_point
                assert p0 is not None and p1 is not None
                l.gmsh_line=mesh.line(p0,p1,name=l.bname)
            elif l.mode=="spline":
                pts:List[Point] = []
                for p in l.ptlist:
                    assert p.gmsh_point  is not None
                    pts.append(p.gmsh_point)
                l.gmsh_line = mesh.spline(pts, name=l.bname)
            else:
                raise RuntimeError("Strange mode "+str(l.mode))



        for n in self._old_meshes.keys():
            self._mesh_domain(n)


    def get_line_entries_by_phys_name(self,name:str):
        res=[]
        for l in self._line_entries:
            if l.bname==name:
                res.append(l)
        if len(res)==0:
            raise RuntimeError("No physical lines named '"+name+"' found ")
        return res


    def remesh(self):
        self.problem=self.template._problem 
        assert self.problem is not None
        self._identify_domains()
        self._boundary_nodes={}
        self._corner_size_map = None
        if self.use_corner_sizes:
            if isinstance(self.template,GmshTemplate):
                self._corner_size_map=self.template._get_boundary_corner_size_map() 
        for n in self._old_meshes.keys():
            self._preprocess_domain(n)
        if self.template._fntrunk is not None:
            fnformat:str=self.template._fntrunk+"_REMESH_{:06d}" 
        else:
            print(self.template)
            raise RuntimeError("TODO: Good trunk name here. Set _fntrunk of the MeshTemplate")
        self.gmsh._meshfile=None 
        self.gmsh._loaded_from_mesh_file = None 
        self.gmsh._mesh_size_callback=self._mesh_size_callback 
        if self.gmsh._mesh_size_callback is not None:
            print("SETTING MESH SIZE CALLBACK",self._mesh_size_callback)
        self.gmsh._do_define_geometry(self.problem,fnformat.format(self._cnt)) 
        self.template._meshfile=self.gmsh._meshfile 
        self.template.get_template()._meshfile=self.gmsh._meshfile 
        self._cnt+=1


    def _get_points_by_phys_name(self,name:str) -> List[List[RemesherPointEntry]]:
        splt=name.split("/")
        if len(splt)<2:
            raise RuntimeError("Cannot identify remeshed mesh points by a 2d domain")
        dn=splt[0]
        if splt[1]  not in self._meshbounds[dn]:
            raise RuntimeError("Cannot find an interface named "+splt[1]+" to remesh on domain "+splt[0]+"\n"+"Available interfaces: "+str(self._meshbounds[dn]))
        #boundline=self._boundary_nodes[splt[1]]
        if len(splt)==2:
            respts:List[List[RemesherPointEntry]]=[]
            for l in self._line_entries:
                if l.bname==splt[1]:
                    respts.append(l.ptlist)
            return respts
        elif len(splt)==3:
            if splt[1]==splt[2]:
                raise RuntimeError("Cannot find intersections between the same lines")
            pset1:Set[RemesherPointEntry]=set()
            pset2:Set[RemesherPointEntry]=set()
            for l in self._line_entries:
                if l.bname == splt[1]:
                    for p in l.ptlist:
                        pset1.add(p)
                elif l.bname==splt[2]:
                    for p in l.ptlist:
                        pset2.add(p)
            pset = pset1.intersection(pset2)
            respts=[]
            for p in pset:
                respts.append([p])
            return respts
        else:
            raise RuntimeError("TODO ?")



class RemesherViaRecreation(RemesherBase):
    def __init__(self,template:MeshTemplate):
        super().__init__(template)
        self.base_trunk=None

    def get_new_template(self):
        return self.template
        
    def remesh(self):
        if self.base_trunk is None:
            self.base_trunk=self.template._fntrunk
        fnformat:str=self.base_trunk+"_REMESH_{:06d}"
        
                
        self._old_meshes={}        
        for k,m in self.template._problem._meshdict.items():
            if isinstance(m,(MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d)):
                if self.template.has_domain(k):
                    self._old_meshes[k]=m
                    
        self.template._reset()
        self.template._do_define_geometry(self.template._problem,fnformat.format(self._cnt))                 
        self._cnt+=1        
        
        


# Can be used for a GmshTemplate, which depends only on problem parameters, e.g. a droplet mesh with a prescribed contact angle
# It will be remeshed by using the same GmshTemplate, but with the current value of the parameter
class ParametricGmshMeshRemesher2d(Remesher2d):
    def __init__(self, template: MeshTemplate):
        super().__init__(template)
        assert isinstance(template,GmshTemplate)
        self.gmsh:GmshTemplate=template
        
    def remesh(self):
        
        self.problem=self.template._problem 
        assert self.problem is not None
        self.gmsh._meshfile=None 
        self.gmsh._loaded_from_mesh_file = None 
        self.gmsh._mesh_size_callback=self._mesh_size_callback 
        if self.template._fntrunk is not None:
            fnformat:str=self.template._fntrunk+"_REMESH_{:06d}" 
        else:
            print(self.template)
            raise RuntimeError("TODO: Good trunk name here. Set _fntrunk of the MeshTemplate")
        if self.gmsh._mesh_size_callback is not None:
            print("SETTING MESH SIZE CALLBACK",self._mesh_size_callback)
        
        self._identify_domains()
        self.gmsh._geometry_defined=False
        self.gmsh._named_entities={}
        self.gmsh._pointhash={}
        self.gmsh._domains={}
        self.gmsh._geom = None        
        self.gmsh._do_define_geometry(self.problem,fnformat.format(self._cnt)) 
        self.template._meshfile=self.gmsh._meshfile 
        self.template.get_template()._meshfile=self.gmsh._meshfile 
        self._cnt+=1
