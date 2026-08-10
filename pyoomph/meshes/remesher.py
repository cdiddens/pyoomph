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
 


import math
import warnings

from ..typings import *
import numpy


from .. import _pyoomph_core as _pyoomph

from .gmsh import GmshTemplate, Point, Line,Spline
from .mesh import MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d,MeshTemplate,MeshedMeshTemplate
from ..generic.mpi import get_mpi_nproc, get_mpi_world_comm

from ..typings import *
if TYPE_CHECKING:
    from ..expressions import ExpressionOrNum
    from ..generic.problem import Problem

class RemesherPointEntry:
    def __init__(self,x:float,y:float,z:float,size:float):
        self.x,self.y,self.z,self.size=x,y,z,size
        self.set_sizes:list[float]=[] # Sizes can be modified
        #self.on_bounds=set()
        self.gmsh_point:Point | None=None

    def get_size(self) -> float:
        if len(self.set_sizes)==0:
            return self.size
        else:
            return sum(self.set_sizes)/len(self.set_sizes)



class RemesherLineEntry:
    def __init__(self,ptlist:list[RemesherPointEntry],mode:str,bname:str):
        self.ptlist=ptlist
        self.mode=mode
        self.gmsh_line:Line | Spline | None=None
        self.bname=bname



class RemesherBase:
    #: What this remesher still gets wrong on a distributed (``--distribute``) problem, quoted by the
    #: refusal in :py:meth:`~pyoomph.generic.problem.Problem.force_remesh`, or ``None`` if it works.
    #: It only names what is specific to *this* remesher; the limitations that hold for all of them
    #: are added there. See dev_docs/distributed_remeshing.md.
    distributed_limitation:str | None="it rebuilds the mesh from this rank's part of the problem only"

    def __init__(self,template:"MeshTemplate"):
        self.template=template
        self._cnt:int=0        
        #self._point_entries = {}
        self._line_entries:list[RemesherLineEntry] = []
        self._unique_pts:list[RemesherPointEntry]=[]
        self._old_meshes:dict[str,MeshFromTemplate1d | MeshFromTemplate2d | MeshFromTemplate3d]={}
        #self._domain_points={} # access the points via domain names

    @property
    def problem(self) -> "Problem":
        # Resolved live from the template, never stored: a Python-level attribute holding the
        # Problem here closed a cycle that the garbage collector cannot break, because one of
        # its edges is a nb::keep_alive record and thus invisible to gc:
        #   Problem -(keep_alive from add_sub_mesh)-> mesh -> _templatemesh -> MeshTemplate
        #   -> remesher -> Problem.
        # Nothing then made the Problem collectible, so its __del__/release() never ran and
        # every remeshing script leaked its whole Problem (meshes, nodes, elements, equations)
        # - reported at shutdown as "nanobind: leaked N instances". Problem.force_remesh() only
        # breaks this for the *superseded* meshes (_destroy_superseded_mesh(), see
        # generic/problem.py); the replacement mesh must keep its _templatemesh, so the edge has
        # to be removed on this side. get_problem() is the same non-owning C++-side lookup that
        # the meshes and code generators use for exactly this reason.
        return self.template.get_problem()

    def add_point_entry(self,x:float,y:float,z:float,size:float) -> RemesherPointEntry:
        for p in self._unique_pts:
            if abs(p.x-x)<1e-9 and abs(p.y-y)<1e-9 and abs(p.z-z)<1e-9:
                return p
        else:
            res=RemesherPointEntry(x,y,z,size)
            self._unique_pts.append(res)
            return res

    def add_line_entry(self,ptlist:list[RemesherPointEntry],mode:str,bname:str):
        self._line_entries.append(RemesherLineEntry(ptlist,mode,bname))

    def _get_points_by_phys_name(self,name:str)->list[list[RemesherPointEntry]]:
        raise RuntimeError("Implement")

    def actions_after_remeshing(self):
        self._line_entries = []
        self._unique_pts = []
        # Everything gathered from the mesh being replaced must go here, at the end of the
        # remeshing process: right after this, Problem.force_remesh() destroys the superseded
        # meshes for good (_destroy_superseded_mesh(), see generic/problem.py), so any Node,
        # element or mesh wrapper still cached here would be left pointing into freed C++
        # memory until the next remesh happens to overwrite it - and merely touching one from
        # a script (or a debugger) segfaults.
        self._old_meshes={}
        #self._domain_points:Dict[str,Dict[str,List[Node]]] = {}

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
            self.kernel=self.remesher.template.kernel
        self.remesher._define_geometry() 

class RemeshBoundaryPoint:
    """A point of a boundary of the mesh being replaced, addressed by its position.

    The remesher used to walk the ``Node`` objects of the local mesh directly. Under ``--distribute``
    each rank holds only its share of every boundary, so the curves have to be stitched together from
    all of them - and a node pointer means nothing on another rank, while its position is the same
    everywhere. The element sizes that set the local resolution are accumulated here as well, since
    the elements around a point on the partition cut are split over several ranks.
    """
    __slots__=("_x","_y","sum_initial","sum_current","count")

    def __init__(self,x:float,y:float):
        self._x,self._y=x,y
        self.sum_initial=0.0
        self.sum_current=0.0
        self.count=0.0

    def x(self,i:int)->float:
        return self._x if i==0 else self._y

    def element_size(self)->float:
        """The resolution the old mesh had here: the mean of the initial and the current element
        size, averaged over the boundary elements around the point - all of them, not just this
        rank's."""
        if self.count<=0:
            return 1.0
        return (self.sum_initial/self.count+self.sum_current/self.count)/2


class Remesher2dBoundaryLineCollection:
    def __init__(self,boundname:str,remesher:"Remesher2d",point_size_func:Callable[[float, float], float] | None=None):
        super(Remesher2dBoundaryLineCollection, self).__init__()
        self.name=boundname
        self.parts=[]
        self.oldnodes:dict[tuple[RemeshBoundaryPoint,RemeshBoundaryPoint],list[RemeshBoundaryPoint]]= {} #Dict mapping from a pair of vertex points to the non-vertex points in between
        self.curves:list[list[RemeshBoundaryPoint]]=[]
        self.point_size_func=point_size_func
        self.remesher=remesher


    def split_into_curves(self): #A boundary may contain more than one subcurve
        self.curves = []
        neighb_connects:dict[_pyoomph.Node,list[_pyoomph.Node]]={} # A dict mapping to a list of node neighbors
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

            currentcurve:list[_pyoomph.Node]=[]
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


    def get_size_at_point(self,p:RemeshBoundaryPoint) -> float:
        if self.point_size_func is not None:
            if callable(self.point_size_func):
                return self.point_size_func(p.x(0),p.x(1))
            else:
                return self.point_size_func
        # The element sizes were accumulated while the points were collected - see
        # RemeshBoundaryPoint - rather than looked up through a node-to-element map here, because the
        # elements around a point can live on several ranks.
        return p.element_size()



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
            elif c[0] == c[-1] and len(c) >= 5:
                # A closed curve must NOT go to gmsh as one spline with its first point repeated.
                # Such a spline has a seam, and the element that straddles it gets its second-order
                # mid-side node from the average of its endpoints' curve parameters - which at the
                # seam averages t~1 and t~0 to t~0.5, placing the node HALFWAY AROUND THE LOOP.
                # The result is one bulk element per closed boundary with a mid-side node at the
                # antipode of where it belongs, on the boundary and at the right radius, so nothing
                # downstream notices; it silently distorts that element and destroys any
                # arclength-based boundary parameterisation. Emitting two open splines instead
                # leaves no seam. See dev_docs/mesh_point_locator.md.
                half = len(c) // 2
                for seq in (range(0, half + 1), range(half, len(c))):
                    plst = [self.remesher.add_point_entry(c[i].x(0), c[i].x(1), 0, size=sizes[i]) for i in seq]
                    self.remesher.add_line_entry(plst, "spline", self.name)
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
    # Works distributed since stage 4 of dev_docs/distributed_remeshing.md: the boundary curves are
    # stitched together from every rank's share before the geometry is described.
    distributed_limitation=None

    def __init__(self,template:MeshTemplate):
        super(Remesher2d, self).__init__(template)
        self._old_meshes={}
        self._boundary_nodes:dict[str,Remesher2dBoundaryLineCollection]={}
        self.gmsh=GmshRemesher2d(self)
        self._meshbounds={}
        self._boundary_point_size_funcs:dict[str,Callable[[float,float],float]]={}
        self.use_corner_sizes=True
        self._corner_size_map=None
        self._mesh_size_callback=None
        self._holes_info:dict[str,list[list[str]]]={}

    def set_holes(self,domain:str,holes:list[list[str]]):
        self._holes_info[domain]=holes

    def set_boundary_point_size(self,**kwargs:Callable[[float,float],float]):
        for name,func_or_val in kwargs.items():
            self._boundary_point_size_funcs[name]=func_or_val


    def actions_after_remeshing(self):
        super(Remesher2d, self).actions_after_remeshing()
        self.gmsh = GmshRemesher2d(self) #Recreate the intenral gmsh remesher
        self._meshbounds:dict[str,list[str]]={}
        self._unique_pts = []
        # The per-boundary node/element bookkeeping of the mesh just replaced - see the base
        # class for why none of it may outlive the remeshing process. All of it is rebuilt from
        # scratch by the next remesh() anyway.
        self._boundary_nodes={}
        self._corner_size_map = None

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

    def _collect_local_boundary_edges(self,n:str)->dict[str,tuple[list[list[float]],list[tuple[int,int,list[int]]]]]:
        """This rank's share of every boundary of domain ``n``, as positions rather than as nodes.

        Per boundary: a point table of ``[x, y, sum sqrt(initial size), sum sqrt(current size),
        element count]`` and the edges of the boundary as index triples ``(first, last, in between)``.
        Both are position-based on purpose, since they are merged across the ranks afterwards.
        """
        mesh=self._old_meshes[n]
        distributed=bool(mesh.is_mesh_distributed())
        local:dict[str,tuple[list[list[float]],list[tuple[int,int,list[int]]]]]={}
        for bn in mesh.get_boundary_names():
            ind=mesh.get_boundary_index(bn)
            if mesh.nboundary_element(ind)==0: #TODO: These bounds could still be relevant
                continue
            pts:list[list[float]]=[]
            index_of:dict[_pyoomph.Node,int]={}
            edges:list[tuple[int,int,list[int]]]=[]
            for be,dir in mesh.boundary_elements(bn,with_directions=True):
                # A halo element is a copy of one that another rank owns, and the boundary lookup
                # keeps it. Letting it contribute would give its nodes the element twice, which is
                # not what the size average means.
                if distributed and be.is_halo():
                    continue
                scal:int=2**be.refinement_level()
                size_i=math.sqrt(be.get_initial_cartesian_nondim_size())*scal
                size_c=math.sqrt(be.get_current_cartesian_nondim_size())*scal
                idx:list[int]=[]
                for i in range(be.nnode_1d()):
                    nd=be.boundary_node_pt(dir,i)
                    k=index_of.get(nd)
                    if k is None:
                        k=len(pts)
                        index_of[nd]=k
                        pts.append([nd.x(0),nd.x(1),0.0,0.0,0.0])
                    pts[k][2]+=size_i
                    pts[k][3]+=size_c
                    pts[k][4]+=1.0
                    idx.append(k)
                edges.append((idx[0],idx[-1],idx[1:-1]))
            local[bn]=(pts,edges)
        return local

    def _merge_boundary_edges(self,local:dict[str,tuple[list[list[float]],list[tuple[int,int,list[int]]]]],order:list[str]):
        """Stitch every rank's share of the boundaries into the whole ones.

        Collective on a distributed mesh, and a no-op otherwise - the merging below is a fixed point
        for a single contribution, so serial and distributed take the same path.

        Every rank ends up with the same result rather than only rank 0: the geometry is described by
        every rank (only rank 0's ``.msh`` is kept, but the others have to describe *something* that
        closes), so what they describe had better be the same thing. `allgather` and a deterministic
        merge give that without a second broadcast.
        """
        contributions=[local]
        if get_mpi_nproc()>1 and any(m.is_mesh_distributed() for m in self._old_meshes.values()):
            comm=get_mpi_world_comm()
            assert comm is not None
            contributions=comm.allgather(local)
        merged:dict[str,tuple[list[RemeshBoundaryPoint],list[tuple[int,int,list[int]]]]]={}
        # In the mesh's own order of boundary names rather than in the order this rank happens to
        # hold them: it is the same list everywhere, and it is the order a serial run uses, which the
        # gmsh entity numbering - and through it the mesh gmsh produces - depends on.
        present={b for c in contributions for b in c.keys()}
        for bn in [b for b in order if b in present]:
            allpts:list[list[float]]=[]
            alledges:list[tuple[int,int,list[int]]]=[]
            for c in contributions:
                if bn not in c:
                    continue
                pts,edges=c[bn]
                off=len(allpts)
                allpts.extend(pts)
                alledges.extend((a+off,b+off,[i+off for i in inb]) for a,b,inb in edges)
            merged[bn]=self._fuse_boundary_points(allpts,alledges)
        return merged

    def _fuse_boundary_points(self,allpts:list[list[float]],alledges:list[tuple[int,int,list[int]]]):
        """Make one point out of the copies of a node that several ranks contributed.

        By position, not by index: the copies come from different ranks and carry no common
        numbering. They are the same node, so they coincide to the last bits - the tolerance is only
        there so that a difference in the last bits does not tear the curve apart at the partition
        cut, and is far below any distance between two genuinely distinct nodes.
        """
        if not allpts:
            return [],[]
        from scipy.spatial import cKDTree
        coords=numpy.array([[p[0],p[1]] for p in allpts])
        extent=max(float(numpy.amax(numpy.abs(coords))),1.0)
        tol=1e-10*extent
        tree=cKDTree(coords)
        # The lowest index within the tolerance represents the cluster. With copies of one node that
        # is unambiguous, and it makes the numbering below independent of how many ranks contributed.
        rep=numpy.arange(len(coords))
        for i,neighbours in enumerate(tree.query_ball_point(coords,r=tol)):
            rep[i]=min(neighbours)
        order={}
        points:list[RemeshBoundaryPoint]=[]
        for i,r in enumerate(rep):
            r=int(r)
            if r not in order:
                order[r]=len(points)
                points.append(RemeshBoundaryPoint(allpts[r][0],allpts[r][1]))
            p=points[order[r]]
            p.sum_initial+=allpts[i][2]
            p.sum_current+=allpts[i][3]
            p.count+=allpts[i][4]
        remap=[order[int(r)] for r in rep]
        edges=[(remap[a],remap[b],[remap[i] for i in inb]) for a,b,inb in alledges]
        # By position, so that the order does not depend on how many ranks contributed and in which
        # order they were concatenated. It is not cosmetic: split_into_curves() starts its walk at the
        # first edge it meets, which fixes the direction of the curve and with it the order the gmsh
        # points are created in - and gmsh gives a different mesh for a differently numbered geometry.
        edges.sort(key=lambda e:(points[e[0]].x(0),points[e[0]].x(1),points[e[1]].x(0),points[e[1]].x(1)))
        return points,edges

    def _define_boundaries_for_domain(self,n:str):
        merged=self._merge_boundary_edges(self._collect_local_boundary_edges(n),
                                          self._old_meshes[n].get_boundary_names())
        self._meshbounds[n]=[bn for bn in merged.keys() if merged[bn][0]]
        for bn,(points,edges) in merged.items():
            if not points or bn in self._boundary_nodes.keys():
                continue
            bnd=Remesher2dBoundaryLineCollection(bn,self,point_size_func=self._boundary_point_size_funcs.get(bn,None))
            self._boundary_nodes[bn]=bnd
            for a,b,inb in edges:
                bnd.oldnodes[(points[a],points[b],)]=[points[i] for i in inb]
            bnd.split_into_curves()


    def _mesh_domain(self,n:str):
        mshbounds=self._meshbounds[n].copy()
        holes=self._holes_info.get(n,None)
        if holes is not None:
            for hole in holes:
                for iname in hole:
                    if iname in mshbounds:
                        mshbounds.remove(iname)
        self.gmsh.plane_surface(*mshbounds,name=n,holes=holes) #type:ignore #holes is list[list[str]], which is a valid (but invariance-incompatible) special case of list[Sequence[str|Line|Spline|BSpline|CircleArc]]


    def _define_geometry(self):
        mesh=self.gmsh
        # _define_boundaries_for_domain is collective on a distributed mesh, so the ranks have to
        # enter it for the same domains in the same order. _old_meshes follows Problem._meshdict,
        # which is built identically everywhere, so its own order already is that order.
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
                newline=mesh.line(p0,p1,name=l.bname) # Called with exactly 2 points, so this can only return a Line or None, never a list[Line]
                assert newline is None or isinstance(newline,Line)
                l.gmsh_line=newline
            elif l.mode=="spline":
                pts:list[Point] = []
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


    def _get_points_by_phys_name(self,name:str) -> list[list[RemesherPointEntry]]:
        splt=name.split("/")
        if len(splt)<2:
            raise RuntimeError("Cannot identify remeshed mesh points by a 2d domain")
        dn=splt[0]
        if splt[1]  not in self._meshbounds[dn]:
            raise RuntimeError("Cannot find an interface named "+splt[1]+" to remesh on domain "+splt[0]+"\n"+"Available interfaces: "+str(self._meshbounds[dn]))
        #boundline=self._boundary_nodes[splt[1]]
        if len(splt)==2:
            respts:list[list[RemesherPointEntry]]=[]
            for l in self._line_entries:
                if l.bname==splt[1]:
                    respts.append(l.ptlist)
            return respts
        elif len(splt)==3:
            if splt[1]==splt[2]:
                raise RuntimeError("Cannot find intersections between the same lines")
            pset1:set[RemesherPointEntry]=set()
            pset2:set[RemesherPointEntry]=set()
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
    """
    Remesher of a :py:class:`~pyoomph.meshes.mesh.MeshedMeshTemplate`, which just calls the
    :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.define_geometry` method of the template again. It is attached by
    default to any :py:class:`~pyoomph.meshes.mesh.MeshedMeshTemplate`, i.e. also to any
    :py:class:`~pyoomph.meshes.gmsh.GmshTemplate`, so it usually does not have to be created manually.
    """
    # Works distributed since stage 1 of dev_docs/distributed_remeshing.md: get_boundary_coordinates()
    # merges the boundary across the ranks, so every rank describes the same geometry again.
    distributed_limitation=None

    def __init__(self,template:MeshTemplate):
        super().__init__(template)
        self.base_trunk=None

    def get_new_template(self):
        return self.template
        
    def remesh(self):
        if self.base_trunk is None:
            self.base_trunk=self.template._fntrunk
        if self.base_trunk is None:
            # A backend that writes no files of its own never sets a trunk (only the GmshTemplate does, for its
            # .msh files). The trunk is merely the base name of whatever files the recreated mesh may produce, so
            # deriving one from the class name is enough to keep the rounds apart.
            self.base_trunk=type(self.template).__name__
        fnformat:str=self.base_trunk+"_REMESH_{:06d}"


        self._old_meshes={}
        for k,m in self.template.get_problem()._meshdict.items():
            if isinstance(m,(MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d)):
                if self.template.has_domain(k):
                    self._old_meshes[k]=m

        self.template._reset()
        assert isinstance(self.template,MeshedMeshTemplate) # Only a MeshedMeshTemplate recreates its geometry, and only its _do_define_geometry accepts the extra filename_trunk argument
        self.template._do_define_geometry(self.template.get_problem(),fnformat.format(self._cnt))
        self._cnt+=1
        
        


# Can be used for a GmshTemplate, which depends only on problem parameters, e.g. a droplet mesh with a prescribed contact angle
# It will be remeshed by using the same GmshTemplate, but with the current value of the parameter
class ParametricGmshMeshRemesher2d(Remesher2d):
    def __init__(self, template: MeshTemplate):
        super().__init__(template)
        assert isinstance(template,GmshTemplate)
        self.gmsh:GmshTemplate=template
        
    def remesh(self):
        
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



class RemeshableGmshTemplate2d(GmshTemplate):
    """
    .. deprecated::
        Just use a plain :py:class:`~pyoomph.meshes.gmsh.GmshTemplate` instead. Remeshing via recreation, together with
        :py:meth:`~pyoomph.meshes.mesh.MeshedMeshTemplate.is_remeshing` and
        :py:meth:`~pyoomph.meshes.mesh.MeshedMeshTemplate.get_boundary_coordinates`, is the default of any
        :py:class:`~pyoomph.meshes.mesh.MeshedMeshTemplate` by now, so this class does not add anything anymore.
    """
    def __init__(self,loaded_from_mesh_file:str | None=None):
        super().__init__(loaded_from_mesh_file=loaded_from_mesh_file)
        warnings.warn("RemeshableGmshTemplate2d is deprecated. Use a plain GmshTemplate instead, which remeshes by recreation by default.",DeprecationWarning,stacklevel=2)


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
