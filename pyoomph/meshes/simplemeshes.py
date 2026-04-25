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
from .mesh import MeshTemplate
from .remesher import Remesher2d
from ..expressions import ExpressionOrNum
import _pyoomph
from ..typings import *
import numpy

class LineMesh(MeshTemplate):
    """
    A class representing a line mesh.

    Args:
        N: The number of elements in the mesh.
        size: The size of the mesh, i.e. the length of the line.
        minimum: The position of the left boundary, i.e. the line ranges from ``minimum`` to ``minimum + size``.
        name: The name of the domain or a function that returns the name based on the center of each element. This allows to create multiple domains in the same mesh, where the interfaces in between are automatically created and named based on the domain names separated with an underscore.
        left_name: The name of the left boundary.
        right_name: The name of the right boundary.
        nodal_dimension: The nodal dimension of the mesh, can be used to curve the mesh later on.
        periodic: Whether the mesh is periodic.
    """

    def __init__(self, N: int = 10, size: ExpressionOrNum = 1.0, minimum: ExpressionOrNum = 0.0, name: Union[str, Callable[[float], str]] = "domain", left_name: str = "left", right_name: str = "right",
                 nodal_dimension: Optional[int] = None, periodic: bool = False):
        super(LineMesh, self).__init__()
        self.N = N
        self.size = size
        self.name = name
        self.minimum = minimum
        self.left_name = left_name
        self.right_name = right_name
        self.nodal_dimension = nodal_dimension
        self.periodic = periodic

    def define_geometry(self):
        """
        Define the geometry of the line mesh.
        """
        if self.N <= 0:
            raise RuntimeError("LineMesh.N must be positive")
        domain_table: Dict[str, _pyoomph.MeshTemplateElementCollection] = {}  # If self.name is callable
        lastdom = None
        if isinstance(self.name, str):
            domain = self.new_domain(self.name, nodal_dimension=self.nodal_dimension)
        else:
            domain = None

        size = self.nondim_size(self.size)
        minimum = self.nondim_size(self.minimum)

        for i in range(self.N):
            p0 = self.add_node_unique(minimum + i * size / self.N)
            p1 = self.add_node_unique(minimum + (i + 1) * size / self.N)
            if domain is None:
                assert not isinstance(self.name, str)
                center = minimum + (i + 0.5) * size / self.N
                local_domain = self.name(center)  # Name as a function
                if local_domain not in domain_table.keys():
                    domain_table[local_domain] = self.new_domain(local_domain)
                domain_table[local_domain].add_line_1d_C1(p0, p1)
                if lastdom is not None and lastdom != local_domain:
                    self.add_nodes_to_boundary(lastdom + "_" + local_domain, [p0])
                lastdom = local_domain
            else:
                domain.add_line_1d_C1(p0, p1)

            if i == 0:
                pleft = p0

            if i == self.N - 1:
                pright = p1
        assert isinstance(pleft, int)  # type:ignore
        assert isinstance(pright, int)  # type:ignore
        self.add_nodes_to_boundary(self.left_name, [pleft])
        self.add_nodes_to_boundary(self.right_name, [pright])
        if self.periodic:
            self.add_periodic_node_pair(pleft, pright)


###################################


class RectangularQuadMesh(MeshTemplate):
    """
    A class representing a rectangular mesh consisting of quadrilateral elements by default.

    Args:
        name: The name of the domain or a function that returns the name based on the center coordinates of each element. The interfaces   in between are automatically generated and named based on the domain names separated with an underscore.
        size: The size of the mesh, either by a single value (for both directions) or by two values (for x and y directions).            
        N: The number of elements in each dimension.. Can be a single value or a list of two values for x and y dimensions respectively.
        lower_left: The coordinates of the lower-left corner of the mesh, i.e. the mesh ranges from ``lower_left[0]`` to ``lower_left[0] + size[0]`` in x-direction and ``lower_left[1]`` to ``lower_left[1] + size[1]]`` in y-direction.            
        periodic: Whether the mesh is periodic, either in both directions or in x and y directions separately.
        split_in_tris: Split the quadrilateral elements into triangles.
        split_scott_vogelius: Whether to use splitting into Scott-Vogelius elements.
        boundary_names: A dictionary mapping boundary names ``"left"``, ``"right"``, ``"top"``, ``"bottom"`` to their corresponding names. Alternatively a function, which also takes the center coordinates of each element as input, can be used to define the boundary names.            
        nodal_dimension: The nodal dimension of the mesh, can be used to curve the mesh later on.
    """
    
    def __init__(self, *, name:Union[str,Callable[[float,float],str]]="domain", size:Union[ExpressionOrNum,List[ExpressionOrNum]]=1.0, N:Union[int,List[int]]=10, lower_left:Union[ExpressionOrNum,List[ExpressionOrNum]]=[0, 0], periodic:Union[bool,List[bool]]=False, split_in_tris:Literal[False, "alternate_left", "alternate_right", "left", "right", "crossed"]=False,split_scott_vogelius:bool=False, boundary_names:Dict[str,Union[str,Callable[[float],str]]]={},nodal_dimension:Optional[int]=None):
        super().__init__()
        self.name:Union[str,Callable[[float,float],str]] = name
        self.size = size
        self.N = N
        self.lower_left = lower_left
        
        if isinstance(periodic, bool):
            self.periodic = [periodic, periodic]
        else:
            self.periodic = periodic
        self.split_in_tris = split_in_tris
        self.remesher = Remesher2d(self)
        self.boundary_names=boundary_names
        self.split_scott_vogelius=split_scott_vogelius
        self.nodal_dimension=nodal_dimension
        if self.nodal_dimension is not None:
            if not isinstance(self.nodal_dimension, int):
                raise RuntimeError("nodal_dimension must be an integer")
            if self.nodal_dimension<2:
                raise RuntimeError("nodal_dimension must be at least 2")
            elif self.nodal_dimension>3:
                raise RuntimeError("nodal_dimension must be at most 3")

    def define_geometry(self):
        dynamic_names:Dict[str,_pyoomph.MeshTemplateElementCollection]={}
        if not callable(self.name):            
            domain = self.new_domain(self.name)
            if self.nodal_dimension is not None:
                domain.set_nodal_dimension(self.nodal_dimension)
        else:
            domain=None

        size = self.nondim_size(self.size)        
        if isinstance(size, int) or isinstance(size, float):
            size = [size, size]
        elif type(size) == "list": #type:ignore
            if len(size) != 2:
                raise ValueError(
                    "Argument size must be a number or a 2d list [Size X, Size Y], but got size=" + str(size))
        if not (isinstance(size[0], int) or isinstance(size[0], float)):  #type:ignore
            raise ValueError("Argument size[0] must be a number, but got size=" + str(size))
        if not (isinstance(size[1], int) or isinstance(size[1], float)):  #type:ignore
            raise ValueError("Argument size[1] must be a number, but got size=" + str(size))

        size=cast(List[float],size)
        assert len(size)==2
        nN = self.N
        if isinstance(nN, int) or isinstance(nN, float):
            nN = [nN, nN]
        elif type(nN) == "list":  #type:ignore
            if len(nN) != 2:
                raise ValueError("Argument N must be a number or a 2d list [NX, NY]")
        if not (isinstance(nN[0], int)):
            if nN[0] is None and (isinstance(nN[1], int)):
                nN[0]=max(round(nN[1]*size[0]/size[1]),1)
            else:
                raise ValueError("Argument N[0] must be an integer")            
        if not (isinstance(nN[1], int)):
            if nN[1] is None and (isinstance(nN[0], int)):
                nN[1]=max(round(nN[0]*size[1]/size[0]),1)
            else:
                raise ValueError("Argument N[1] must be an integer")

        if nN[0] <= 0 or nN[1] <= 0:
            raise ValueError("Mesh size must be a positive integer, but got " + str(nN))

        lower_left = self.lower_left
        if isinstance(lower_left, list) or isinstance(lower_left, tuple) or isinstance(lower_left, numpy.ndarray):
            lower_left=list(lower_left)
            lower_left = [self.nondim_size(x) for x in lower_left]
        else:
            lower_left = self.nondim_size(lower_left)
            lower_left = [lower_left, lower_left]

        splt = self.split_in_tris
        if splt:
            if not splt in [False, "alternate_left", "alternate_right", "left", "right", "crossed"]:
                raise RuntimeError(
                    "kwarg split_in_tris can only be False,'left', 'right', 'alternate_left', 'alternate_right' or 'crossed'")


        def add_to_bound(bn:str,nodes:List[int],centercoord:Optional[float]):
            bnn=self.boundary_names.get(bn,bn)
            if callable(bnn):
                if centercoord is None:
                    raise RuntimeError("Domain name lambdas are not allowed")
                bn=bnn(centercoord)
            else:
                bn=bnn
            self.add_nodes_to_boundary(bn,nodes)
      
        if self.split_scott_vogelius:
            add_tri_C1=lambda n1,n2,n3: domain.add_SV_tri_2d_C1(n1,n2,n3) #type:ignore
        else:                            
            add_tri_C1=lambda n1,n2,n3: domain.add_tri_2d_C1(n1,n2,n3) #type:ignore

        alternate = False
        for ix in range(nN[0]):
            splt = self.split_in_tris
            if self.split_in_tris == "alternate_left":
                splt = "left" if ix % 2 == 0 else "right"
                alternate = True
            elif self.split_in_tris == "alternate_right":
                splt = "right" if ix % 2 == 0 else "left"
                alternate = True
            for iy in range(nN[1]):
                n00 = self.add_node_unique(ix * size[0] / nN[0] + lower_left[0], iy * size[1] / nN[1] + lower_left[1])
                n10 = self.add_node_unique((ix + 1) * size[0] / nN[0] + lower_left[0],
                                           iy * size[1] / nN[1] + lower_left[1])
                n01 = self.add_node_unique(ix * size[0] / nN[0] + lower_left[0],
                                           (iy + 1) * size[1] / nN[1] + lower_left[1])
                n11 = self.add_node_unique((ix + 1) * size[0] / nN[0] + lower_left[0],
                                           (iy + 1) * size[1] / nN[1] + lower_left[1])

                    
                if callable(self.name):
                    assert not isinstance(self.name,str)
                    dn=self.name((ix+0.5) * size[0] / nN[0] + lower_left[0], (iy+0.5) * size[1] / nN[1] + lower_left[1]) #type:ignore
                    if dn not in dynamic_names.keys():
                        dynamic_names[dn]=self.new_domain(dn)
                        if self.nodal_dimension is not None:
                            dynamic_names[dn].set_nodal_dimension(self.nodal_dimension)
                    domain=dynamic_names[dn]
                    if ix>0:
                        otherdom=self.name((ix-0.5) * size[0] / nN[0] + lower_left[0], (iy+0.5) * size[1] / nN[1] + lower_left[1])
                        if otherdom!=dn:
                            bnd = dn + "_" + otherdom if dn < otherdom else otherdom + "_" + dn
                            add_to_bound(bnd,[n00,n01],None)
                    elif ix<nN[0]-1:
                        otherdom = self.name((ix + 1.5) * size[0] / nN[0] + lower_left[0],
                                             (iy + 0.5) * size[1] / nN[1] + lower_left[1])
                        if otherdom != dn:
                            bnd = dn + "_" + otherdom if dn < otherdom else otherdom + "_" + dn
                            add_to_bound(bnd, [n10, n11],None)
                    if iy>0:
                        otherdom = self.name((ix + 0.5) * size[0] / nN[0] + lower_left[0],
                                             (iy - 0.5) * size[1] / nN[1] + lower_left[1])
                        if otherdom != dn:
                            bnd = dn + "_" + otherdom if dn < otherdom else otherdom + "_" + dn
                            add_to_bound(bnd, [n00, n10],None)
                    elif iy<nN[1]-1:
                        otherdom = self.name((ix + 0.5) * size[0] / nN[0] + lower_left[0],
                                             (iy - 1.5) * size[1] / nN[1] + lower_left[1])
                        if otherdom != dn:
                            bnd = dn + "_" + otherdom if dn < otherdom else otherdom + "_" + dn
                            add_to_bound(bnd, [n01, n11],None)
                assert isinstance(domain,_pyoomph.MeshTemplateElementCollection)
                if not splt:
                    if self.split_scott_vogelius:                        
                        raise RuntimeError("Scott-Vogelius only works if you set split_in_tris")
                    domain.add_quad_2d_C1(n00, n10, n01, n11)
                
                if splt == "crossed":
                    ncc = self.add_node_unique((ix + 0.5) * size[0] / nN[0] + lower_left[0],
                                               (iy + 0.5) * size[1] / nN[1] + lower_left[1])
                    add_tri_C1(n00, n10, ncc)
                    add_tri_C1(n10, n11, ncc)
                    add_tri_C1(n11, n01, ncc)
                    add_tri_C1(n01, n00, ncc)
                elif splt == "left":
                    add_tri_C1(n00, n10, n11)
                    add_tri_C1(n00, n11, n01)
                elif splt == "right":
                    add_tri_C1(n01, n10, n11)
                    add_tri_C1(n00, n10, n01)

                if alternate:
                    if splt == "left":
                        splt = "right"
                    else:
                        splt = "left"


                if ix == 0:  add_to_bound("left", [n00, n01],(iy + 0.5) * size[1] / nN[1] + lower_left[1])
                if ix == nN[0] - 1:
                    add_to_bound("right", [n10, n11],(iy + 0.5) * size[1] / nN[1] + lower_left[1])
                if iy == 0:  add_to_bound("bottom", [n00, n10],(ix + 0.5) * size[0] / nN[0] + lower_left[0])
                if iy == nN[1] - 1:  add_to_bound("top", [n01, n11],(ix + 0.5) * size[0] / nN[0] + lower_left[0])

        if self.periodic[0]:
            xl = lower_left[0]
            xr = size[0] + lower_left[0]
            for iy in range(nN[1] + 1):
                y = iy * size[1] / nN[1] + lower_left[1]
                nl = self.add_node_unique(xl, y)
                nr = self.add_node_unique(xr, y)
                print(y, nl, nr)
                self.add_periodic_node_pair(nl, nr)

        if self.periodic[1]:
            yb = lower_left[1]
            yt = size[1] + lower_left[1]
            for ix in range(nN[0] + 1):
                x = ix * size[0] / nN[0] + lower_left[0]
                nb = self.add_node_unique(x, yb)
                nt = self.add_node_unique(x, yt)
                self.add_periodic_node_pair(nb, nt)

        if not callable(self.name):
            self._fntrunk = "RectangularQuadMesh_" + self.name
        else:
            self._fntrunk = "RectangularQuadMesh_" + self.name(lower_left[0],lower_left[1])


#############################################


class CircularMesh(MeshTemplate):
    """
    Create a circular mesh by separting a filled circle into four segments. The segments can be specified by the ``segments`` argument. The mesh is created by connecting the center of the circle with the four corners of the segments. The inner factor specifies the radius of the inner quadratic elements relative to the circle radius. The domain name is given by ``domain_name`` and the outer interface name can be controlled by ``outer_interface``. The straight interfaces are named ``center_to_north``, ``center_to_west``, ``center_to_south``, and ``center_to_east`` by default, but can be remapped by the ``straight_interface_name`` argument. 

    Args:
        radius: The radius of the circle.
        inner_factor: The factor by which the inner radius of the quadratic elements is smaller than the circle radius.
        segments: The segments of the circle to be meshed. Can be a list of strings ``"NW"``, ``"NE"``, ``"SW"``, and ``"SE"`` or the string ``"all"``.
        domain_name: The name of the domain.
        outer_interface: The name of the outer interface.
        straight_interface_name: The name of the straight interfaces. Can be a string, a dictionary mapping the default names to new names, or a callable function that maps the default names to new names.
        with_curved_entities: Whether to create curved entities.
        internal_straight_names: The name of the internal straight interfaces, i.e. interior interfaces from the four directions to the center. Can be a string or a dictionary mapping the default names to new names.
    """
    def __init__(self, radius:ExpressionOrNum=1, inner_factor:float=0.4, segments:Union[Literal["all"],List[Literal["NW","NE","SW","SE"]]]="all", domain_name:str="domain", outer_interface:str="circumference",straight_interface_name:Optional[Union[str,Dict[str,str],Callable[[str],str]]]=None,with_curved_entities:bool=True,internal_straight_names:Optional[Union[str,Dict[str,str]]]=None):
        super(CircularMesh, self).__init__()
        self.radius = radius
        self.inner_factor = inner_factor
        self.domain_name = domain_name
        self.segments = segments
        self.outer_interface=outer_interface
        self.straight_interface_name=straight_interface_name
        self.internal_straight_names=internal_straight_names
        self._curved_entities:List[_pyoomph.MeshTemplateCurvedEntityBase]=[]
        self.with_curved_entities=with_curved_entities

    def define_geometry(self):
        router = self.nondim_size(self.radius)
        rinner = self.inner_factor * router
        rdiag = math.sqrt(0.5) * router

        norig = self.add_node_unique(0, 0)
        domain = self.new_domain(self.domain_name)

        allsegs = ["NE",  "NW","SW","SE"]
        if self.segments == "all":
            segments = allsegs
        elif not isinstance(self.segments,(list,set)):
            raise ValueError("Segements needs to be a list")
        else:
            segments:List[str]=[]
            for a in self.segments:
                if not (a in allsegs):
                    raise ValueError("Segements need to be a subset of " + str(allsegs))
                segments.append(a)

        segpresent = [True, True, True, True]
        nextboundname = ["center_to_north", "center_to_west", "center_to_south", "center_to_east"]
        for mode in range(4):
            if not (allsegs[mode] in segments):
                segpresent[mode] = False

        for mode in range(4):
            if not segpresent[mode]:
                continue

            def g(x:float, y:float, i:int):
                transform = [[x, y], [-y, x], [-x, -y], [y, -x]]
                return transform[mode][i]

            def un(x:float, y:float):
                return self.add_node_unique(g(x, y, 0), g(x, y, 1))

            ni0 = un(rinner, 0)
            n0i = un(0, rinner)
            nii = un(rinner, rinner)
            no0 = un(router, 0)
            n0o = un(0, router)
            ndd = un(rdiag, rdiag)
            domain.add_quad_2d_C1(norig, ni0, n0i, nii)
            domain.add_quad_2d_C1(n0i, nii, n0o, ndd)
            domain.add_quad_2d_C1(ni0, no0, nii, ndd)
            self.add_nodes_to_boundary(self.outer_interface, [n0o, ndd])
            self.add_nodes_to_boundary(self.outer_interface, [no0, ndd])

            if self.with_curved_entities:
                ce=_pyoomph.CurvedEntityCircleArc(self.get_node_position(norig), self.get_node_position(n0o), self.get_node_position(ndd))
                self.add_facet_to_curve_entity([n0o, ndd], ce)
                self._curved_entities.append(ce)
                ce = _pyoomph.CurvedEntityCircleArc(self.get_node_position(norig), self.get_node_position(ndd),self.get_node_position(no0))
                self.add_facet_to_curve_entity([no0, ndd], ce)
                self._curved_entities.append(ce)

            def get_straight_boundname(mode:int):
                bn = nextboundname[mode]
                if self.straight_interface_name is None:
                    return bn
                elif isinstance(self.straight_interface_name, str):
                    bn = self.straight_interface_name
                elif callable(self.straight_interface_name):
                    bn = self.straight_interface_name(bn)
                elif isinstance(self.straight_interface_name, dict): #type:ignore
                    if not bn in self.straight_interface_name.keys():
                        raise RuntimeError("straight_interface_name must have a key named "+bn)
                    bn = self.straight_interface_name[bn]
                else:
                    raise RuntimeError(
                        "straight_interface_name must be either a string to name all straight boundaries like this, or a dict or a callable to map the names")
                return bn

            if not segpresent[mode - 1]:
                self.add_nodes_to_boundary(get_straight_boundname(mode-1), [norig, ni0, no0])
            elif self.internal_straight_names is not None:
                if isinstance(self.internal_straight_names,str):
                    self.add_nodes_to_boundary(self.internal_straight_names, [norig, ni0, no0])
                elif isinstance(self.internal_straight_names,dict) and nextboundname[(mode+3)%4] in self.internal_straight_names.keys(): #type:ignore
                    self.add_nodes_to_boundary(self.internal_straight_names[nextboundname[(mode+3)%4]], [norig, ni0, no0])

            if not segpresent[(mode + 1) % 4]:
                self.add_nodes_to_boundary(get_straight_boundname(mode), [norig, n0i, n0o])
            elif self.internal_straight_names is not None:
                if isinstance(self.internal_straight_names, str):
                    self.add_nodes_to_boundary(self.internal_straight_names, [norig, n0i, n0o])
                elif isinstance(self.internal_straight_names, dict) and nextboundname[mode ] in self.internal_straight_names.keys(): #type:ignore
                    self.add_nodes_to_boundary(self.internal_straight_names[nextboundname[mode]], [norig, n0i, n0o])

        #exit()

##########################


class CuboidBrickMesh(MeshTemplate):
    def __init__(self,*, size:Union[ExpressionOrNum,List[ExpressionOrNum]]=1.0, N:Union[int,List[int]]=10, lower_left:Union[ExpressionOrNum,List[ExpressionOrNum]]=[0, 0, 0],domain_name:str="domain"):
        super().__init__()
        self.size=size
        self.N=N
        self.lower_left=lower_left
        self.periodic:bool=False
        self.domain_name=domain_name

    def define_geometry(self):
        size = self.nondim_size(self.size)
        periodic=self.periodic
        N=self.N
        lower_left=self.lower_left
        if isinstance(size, int) or isinstance(size, float):
            size = [size, size, size]
        elif type(size) == "list": #type:ignore
            if len(size) != 3:
                raise ValueError(
                    "Argument size must be a number or a 3d list [Size X, Size Y, Size Z], but got size=" + str(size))
        if not (isinstance(size[0], int) or isinstance(size[0], float)): #type:ignore
            raise ValueError("Argument size[0] must be a number, but got size=" + str(size))
        if not (isinstance(size[1], int) or isinstance(size[1], float)): #type:ignore
            raise ValueError("Argument size[1] must be a number, but got size=" + str(size))
        if not (isinstance(size[2], int) or isinstance(size[2], float)): #type:ignore
            raise ValueError("Argument size[2] must be a number, but got size=" + str(size))


        if isinstance(periodic, bool): #type:ignore
            periodic = [periodic, periodic, periodic]
        if True in periodic:
            raise RuntimeError("Periodic not implemented")

        if isinstance(N, int) :
            N = [N, N, N] #type:ignore
        elif type(N) == "list": #type:ignore
            if len(N) != 3:
                raise ValueError("Argument N must be a number or a 3d list [NX, NY, NZ]")
        if not (isinstance(N[0], int)): #type:ignore
            raise ValueError("Argument N[0] must be an integer")
        if not (isinstance(N[1], int)): #type:ignore
            raise ValueError("Argument N[1] must be an integer")
        if not (isinstance(N[2], int)): #type:ignore
            raise ValueError("Argument N[2] must be an integer")

        if isinstance(self.lower_left, list) : #type:ignore
            lower_left = [self.nondim_size(x) for x in self.lower_left]
        else:
            lower_left = self.nondim_size(self.lower_left)
            lower_left = [lower_left, lower_left, lower_left]

        # if split_in_tris:
        #         if not split_in_tris in [False, "alternate_left", "alternate_right", "left", "right", "crossed"]:
        #           raise RuntimeError(
        #               "kwarg split_in_tris can only be False,'left', 'right', 'alternate_left', 'alternate_right' or 'crossed'")

        dom=self.new_domain(self.domain_name)
        for ix in range(N[0]):
            for iy in range(N[1]):
                for iz in range(N[2]):
                    def add_node(px:int, py:int, pz:int) -> int:
                        return self.add_node_unique((ix + px) * size[0] / N[0] + lower_left[0],
                                                (iy + py) * size[1] / N[1] + lower_left[1],
                                                (iz + pz) * size[2] / N[2] + lower_left[2])

                    n000 = add_node(0, 0, 0)
                    n100 = add_node(1, 0, 0)
                    n010 = add_node(0, 1, 0)
                    n110 = add_node(1, 1, 0)

                    n001 = add_node(0, 0, 1)
                    n101 = add_node(1, 0, 1)
                    n011 = add_node(0, 1, 1)
                    n111 = add_node(1, 1, 1)

                    dom.add_brick_3d_C1(n000, n100, n010, n110, n001, n101, n011, n111)

                    if ix == 0:  self.add_nodes_to_boundary("left", [n000, n010, n001, n011])
                    if ix == N[0] - 1: self.add_nodes_to_boundary("right", [n100, n110, n101, n111])

                    if iy == 0:  self.add_nodes_to_boundary("bottom", [n000, n100, n001, n101])
                    if iy == N[1] - 1:  self.add_nodes_to_boundary("top", [n010, n110, n011, n111])

                    if iz == 0:  self.add_nodes_to_boundary("back", [n000, n100, n010, n110])
                    if iz == N[2] - 1:  self.add_nodes_to_boundary("front", [n001, n101, n011, n111])

#############################################



class SphericalOctantMesh(MeshTemplate):
    """
    Creates a coarse spherical octant.

    Args:
        radius: The radius of the sphere.
        inner_factor: The factor by which the inner radius of the quadratic elements is smaller than the sphere radius.
        domain_name: The name of the domain.
        interface_names: A dictionary mapping the interface names to their corresponding names.
    """
    def __init__(self, radius:ExpressionOrNum=1, inner_factor:float=0.4,domain_name:str="domain",interface_names:Dict[str,str]={"shell":"shell","plane_x0":"plane_x0","plane_y0":"plane_y0"}):
        super(SphericalOctantMesh, self).__init__()
        self.radius=radius
        self.inner_factor=inner_factor
        self.domain_name=domain_name
        self.interface_names=interface_names

    def define_geometry(self):
        router = self.nondim_size(self.radius)
        rinner = self.inner_factor * router
        rdiag = math.sqrt(0.5) * router
        rtriag = math.sqrt(1/3) * router

        domain = self.new_domain(self.domain_name)

        un:Callable[[float,float,float],int]= lambda x, y,z : self.add_node_unique(x,y,z)

        n000 =un(0,0,0)
        ni00 = un(rinner, 0,0)
        n0i0 = un(0, rinner,0)
        nii0 = un(rinner, rinner,0)
        n00i = un(0, 0, rinner)
        ni0i = un(rinner, 0, rinner)
        n0ii = un(0, rinner, rinner)
        niii = un(rinner, rinner, rinner)
        n00o = un(0, 0, router)
        n0o0 = un(0, router, 0)
        no00 = un(router, 0, 0)
        nd0d = un(rdiag, 0, rdiag)
        n0dd = un(0, rdiag, rdiag)
        ndd0 = un(rdiag, rdiag, 0)
        nttt = un(rtriag, rtriag, rtriag)

        domain.add_brick_3d_C1(n000,ni00,n0i0,nii0,n00i,ni0i,n0ii,niii)
        domain.add_brick_3d_C1(n00i, ni0i, n0ii, niii,n00o,nd0d,n0dd,nttt)
        domain.add_brick_3d_C1(n0i0,nii0,n0o0,ndd0,n0ii,niii,n0dd,nttt)
        domain.add_brick_3d_C1(ni00,no00,nii0,ndd0,ni0i,nd0d,niii,nttt)

        iname=self.interface_names.get("shell","shell")
        if iname is not None:
            self.add_nodes_to_boundary(iname,[no00,n0o0,n00o,nttt,ndd0,nd0d,n0dd])
        iname = self.interface_names.get("plane_x0","plane_x0")
        if iname is not None:
            self.add_nodes_to_boundary(iname, [n000,n00o,n0o0,n00i,n0i0,n0dd,n0ii])
        iname = self.interface_names.get("plane_y0","plane_y0")
        if iname is not None:
            self.add_nodes_to_boundary(iname, [n000, n00o, no00, n00i, ni00, nd0d, ni0i])
        iname = self.interface_names.get("plane_z0", "plane_z0")
        if iname is not None:
            self.add_nodes_to_boundary(iname, [n000, n0o0, no00, n0i0, ni00, ndd0, nii0])

        if False:
            # TODO: This does not work yet
            import _pyoomph
            ce = _pyoomph.CurvedEntitySpherePart(self.get_node_position(n000), self.get_node_position(n00o),[1,0,0])
            self._ce=ce
            self.add_facet_to_curve_entity([n00o, n0dd,nd0d,nttt], ce)
            self.add_facet_to_curve_entity([nd0d, no00, nttt, ndd0], ce)
            self.add_facet_to_curve_entity([ndd0, nttt, nd0d, ndd0], ce)




# TODO: Add curved entities!
class CylinderMesh(MeshTemplate):
    def __init__(self, radius:ExpressionOrNum=1, height:ExpressionOrNum=1, nsegments_h:int=1, inner_factor:float=0.4,  domain_name:str="domain", outer_interface:str="mantle",top_interface:str="top",bottom_interface:str="bottom",zshift:ExpressionOrNum=0):
        super(CylinderMesh, self).__init__()
        self.radius = radius
        self.height=height
        self.nsegments_h=nsegments_h
        self.inner_factor = inner_factor
        self.domain_name = domain_name
       
        self.outer_interface=outer_interface
        self.top_interface=top_interface
        self.bottom_interface=bottom_interface
        self.zshift=zshift

    def define_geometry(self):
        router = self.nondim_size(self.radius)
        height = self.nondim_size(self.height)
        rinner = self.inner_factor * router
        zshift=self.nondim_size(self.zshift)
        import math
        rdiag = math.sqrt(0.5) * router

        domain = self.new_domain(self.domain_name)

        for mode in range(4):
            def g(x:float, y:float,z:float, i:int)->float:
                transform = [[x, y,z], [-y, x,z], [-x, -y,z], [y, -x,z]]
                return transform[mode][i]

            def un(x:float, y:float,z:float)->int:
                return self.add_node_unique(g(x, y, z,0), g(x, y,z, 1),g(x, y,z, 2))

            hs=numpy.linspace(0,1,num=self.nsegments_h+1,endpoint=True) #type:ignore
            for ns in range(self.nsegments_h):
                zlower:float=hs[ns]*height+zshift
                zupper:float=hs[ns+1]*height+zshift
                norigl = self.add_node_unique(0, 0, zlower)
                ni0l = un(rinner, 0,zlower)
                n0il = un(0, rinner,zlower)
                niil = un(rinner, rinner,zlower)
                no0l = un(router, 0,zlower)
                n0ol = un(0, router,zlower)
                nddl = un(rdiag, rdiag,zlower)

                norigu = self.add_node_unique(0, 0, zupper)
                ni0u = un(rinner, 0,zupper)
                n0iu = un(0, rinner,zupper)
                niiu = un(rinner, rinner,zupper)
                no0u = un(router, 0,zupper)
                n0ou = un(0, router,zupper)
                nddu = un(rdiag, rdiag,zupper)

                domain.add_brick_3d_C1(norigl, ni0l, n0il, niil, norigu, ni0u, n0iu, niiu)
                domain.add_brick_3d_C1(n0il, niil, n0ol, nddl , n0iu, niiu, n0ou, nddu)
                domain.add_brick_3d_C1(ni0l, no0l, niil, nddl , ni0u, no0u, niiu, nddu)
                self.add_nodes_to_boundary(self.outer_interface, [n0ol, nddl,n0ou, nddu])
                self.add_nodes_to_boundary(self.outer_interface, [no0l, nddl , no0u, nddu])
                if ns==0:
                    self.add_nodes_to_boundary(self.bottom_interface,[norigl,norigl, ni0l, n0il, niil,nddl,no0l])
                if ns==self.nsegments_h-1:
                    self.add_nodes_to_boundary(self.top_interface,[norigu,norigu, ni0u, n0iu, niiu,nddu,no0u])


class PointMesh(MeshTemplate):
    """ 
    A mesh consisting of a single point (without any spatial coordinates). This is useful to use the normal mode expansion to a 1d problem
    """
    def __init__(self,domain_name:str="domain",nodal_dimension:int=0):
        super().__init__()
        self.domain_name=domain_name
        self.nodal_dimension=nodal_dimension
        
    def define_geometry(self) -> None:
        dom=self.new_domain(self.domain_name)        
        dom.set_nodal_dimension(self.nodal_dimension)
        dom.add_point_element(self.add_node_unique(0))