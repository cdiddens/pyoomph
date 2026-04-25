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
 

from ..generic.codegen import InterfaceEquations,EquationTree
from ..expressions import ExpressionOrNum
from .mesh import InterfaceMesh
from .meshdatacache import MeshDataCacheEntry
import numpy
from ..typings import *

if TYPE_CHECKING:
    from .interpolator import BaseMeshToMeshInterpolator


class AssignZetaCoordinatesBase(InterfaceEquations):
    def assign_zetas(self,mesh:InterfaceMesh):
        raise RuntimeError("This function must be implemented!")

    def after_mapping_on_macro_elements(self):
        self.assign_zetas(self.get_mesh())
        return super().after_mapping_on_macro_elements()    
    
    def before_mesh_to_mesh_interpolation(self, eqtree: "EquationTree", interpolator: "BaseMeshToMeshInterpolator"):
        new_mesh=eqtree._mesh # self.get_mesh()
        old_mesh=interpolator.old.get_mesh(new_mesh.get_name())
        self.assign_zetas(old_mesh)
        self.assign_zetas(new_mesh)
        return super().before_mesh_to_mesh_interpolation(eqtree,interpolator)

    def after_remeshing(self, eqtree: "EquationTree"):
        self.assign_zetas(self.get_mesh())
        return super().after_remeshing(eqtree)

class AssignZetaCoordinatesByEulerianCoordinate(AssignZetaCoordinatesBase):
    def __init__(self,direction:Union[int,Literal["x","y","z"]]):
        super().__init__()
        if isinstance(direction,str):
            if direction=="x":
                direction=0
            elif direction=="y":
                direction=1
            elif direction=="z":
                direction=2
            else:
                raise RuntimeError("unknown direction: "+direction)
        self.direction=direction
    
    def assign_zetas(self,mesh:InterfaceMesh):
        if mesh.get_dimension()!=1:
            raise RuntimeError("Currently only implemented for 1d interfaces meshes")
        bmesh=mesh.get_bulk_mesh()
        if isinstance(bmesh,InterfaceMesh):
            raise RuntimeError("Cannot do it, if the parent mesh is not a bulk mesh")
        bind=bmesh.get_boundary_index(mesh.get_name())
        minzeta=1e40
        maxzeta=-minzeta        
        nodes_set=0
        for e in mesh.elements():            
            for ni in range(e.nnode()):
                n=e.node_pt(ni)
                zeta=n.x(self.direction)
                minzeta=min(zeta,minzeta)
                maxzeta=max(zeta,maxzeta)
                n.set_coordinates_on_boundary(bind,[zeta])
                nodes_set+=1
        bmesh.boundary_coordinate_bool(bind)  
        mesh.update_zeta_in_buffer()      
        if maxzeta-minzeta<1e-10 and nodes_set>1:
            raise self.add_exception_info(RuntimeError("The assigned zeta coordinates are not meaningful. Probably align along another axis"))


class AssignZetaCoordinatesByArclength(AssignZetaCoordinatesBase):
    def __init__(self,start_near_point:Optional[Tuple[ExpressionOrNum,ExpressionOrNum]]=None,sort_along_axis:Optional[Literal["x+","x-","y+","y-"]]=None,normalized:bool=True,segment_jump_offset:float=1.0,individual_segments:bool=True):
        super().__init__()
        self.start_near_point=start_near_point
        self.sort_along_axis=sort_along_axis
        self.normalized=normalized

        self.segment_jump_offset=segment_jump_offset  # Add this offset to the arclength when a new segment is started
        self.individual_segments=individual_segments # process and potentially normalize each segment individually. The total zeta parametrization is then by concatenation

        if (start_near_point is None and sort_along_axis is None) or (start_near_point is not None and sort_along_axis is not None):
            raise RuntimeError("Please add one parameter identifying the direction of the zeta parameterization")

    def assign_zetas(self,mesh):        
        if mesh.get_dimension()!=1:
            raise RuntimeError("Currently only implemented for 1d interfaces meshes")
        bmesh=mesh.get_bulk_mesh()
        if isinstance(bmesh,InterfaceMesh):
            raise RuntimeError("Cannot do it, if the parent mesh is not a bulk mesh")
        bind=bmesh.get_boundary_index(mesh.get_name())
        cache=MeshDataCacheEntry(mesh,True,True)
                
        pts=cache.get_coordinates()
        segs,_=cache.get_interface_line_segments()        

        # Sort and reverse the segments based on the settings
        if self.sort_along_axis is not None:
            index,sign=({"x+":(0,1),"x-":(0,-1),"y+":(1,1),"y-":(1,-1)})[self.sort_along_axis]
            for i,seg in enumerate(segs):
                diff=pts[index,seg[-1]]-pts[index,seg[0]]
                if diff*sign<0:
                    segs[i]=list(reversed(seg))
            segs=sorted(segs,key=lambda s: sign*pts[index,s[0]])
        elif self.start_near_point is not None:
            stp=self.start_near_point
            for i,seg in enumerate(segs):
                d1=(pts[0,seg[0]]-stp[0])**2+(pts[1,seg[0]]-stp[1])**2
                d2=(pts[0,seg[-1]]-stp[0])**2+(pts[1,seg[-1]]-stp[1])**2
                if d2>d1:
                    segs[i]=list(reversed(seg))
            segs=sorted(segs,key=lambda s: (pts[0,s[0]]-stp[0])**2+(pts[1,s[0]]-stp[1])**2 )

        nodemap=mesh.fill_node_index_to_node_map()
        if len(nodemap)!=sum(len(seg) for seg in segs):
            print("NODEMAP",nodemap)
            print("SEGS",segs)
            raise RuntimeError("NODEMAP AND SEGMENT LENGTH MISMATCH")


        alengths:List[float]=[]
        ptinds:List[int]=[]
        aleng=0.0

        if not self.individual_segments:
            for seg in segs:
                oldx,oldy=pts[0,seg[0]],pts[1,seg[0]]
                for ptind in seg:
                    x,y=pts[0,ptind],pts[1,ptind]
                    dl=numpy.sqrt((x-oldx)**2+(y-oldy)**2)
                    aleng+=dl
                    alengths.append(aleng)
                    ptinds.append(ptind)
                    oldx,oldy=x,y
                aleng+=self.segment_jump_offset
                    
            if self.normalized:
                alengths=numpy.array(alengths)/alengths[-1]
        else:   
            aleng_segs:List[NPFloatArray]=[]
            for seg in segs:
                alength_seg:List[float]=[]
                aleng=0.0
                oldx,oldy=pts[0,seg[0]],pts[1,seg[0]]
                for ptind in seg:
                    x,y=pts[0,ptind],pts[1,ptind]
                    dl=numpy.sqrt((x-oldx)**2+(y-oldy)**2)
                    aleng+=dl
                    alength_seg.append(aleng)
                    ptinds.append(ptind)
                    oldx,oldy=x,y                                   

                if self.normalized:
                    alength_seg=numpy.array(alength_seg)/alength_seg[-1]
                else:
                    alength_seg=numpy.array(alength_seg)
                aleng_segs.append(alength_seg)
            offs=0.0
            for i in range(len(aleng_segs)):
                aleng_segs[i]+=offs
                offs=aleng_segs[i][-1]+self.segment_jump_offset
            alengths=numpy.concatenate(aleng_segs)

        for al,pti in zip(alengths,ptinds):
            n=nodemap[pti]
            n.set_coordinates_on_boundary(bind,[al])
        bmesh.boundary_coordinate_bool(bind)
        mesh.update_zeta_in_buffer()      




class DebugZetaCoordinate(InterfaceEquations):
    def get_zeta_name(self):
        master=self._get_combined_element()
        name=master._assert_codegen()._name
        return "zeta_"+str(name)
    
    def define_fields(self):
        self.define_scalar_field(self.get_zeta_name(),"C1")

    def define_residuals(self):
        self.set_Dirichlet_condition(self.get_zeta_name(),True)

    def update_zetas(self):
        mesh=self.get_mesh()
        assert isinstance(mesh,InterfaceMesh)
        bmesh=mesh.get_bulk_mesh()
        interfid = bmesh.has_interface_dof_id(self.get_zeta_name())
        bind=bmesh.get_boundary_index(mesh.get_name())
        print("UPDATEING ZETAS",interfid,bind)  
        for n in mesh.nodes():
            ind=n.additional_value_index(interfid)
            n.set_value(ind,n.get_coordinates_on_boundary(bind)[0])
            n.pin(ind)
        print("ZETAS UPDATED")

    def after_mapping_on_macro_elements(self):
        self.update_zetas()
        return super().after_mapping_on_macro_elements()

    def after_remeshing(self, eqtree: "EquationTree"):
        self.update_zetas()
        return super().after_remeshing(eqtree)