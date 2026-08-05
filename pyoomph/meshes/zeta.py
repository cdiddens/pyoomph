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
 

from ..generic.codegen import InterfaceEquations,EquationTree
from ..expressions import ExpressionOrNum
from .mesh import InterfaceMesh, ODEStorageMesh
from .meshdatacache import MeshDataCacheEntry, MeshDataCacheKey
import numpy
from ..typings import *

if TYPE_CHECKING:
    from .interpolator import BaseMeshToMeshInterpolator


def _find_closed_segments(cache:MeshDataCacheEntry,segs:list[list[int]])->list[bool]:
    """Mark which of the segments returned by ``get_interface_line_segments`` are closed loops.

    The segment walk notices the looped case (it falls back to picking an arbitrary start node when
    no endpoint has a single neighbour) but only prints about it, and it walks the loop all the way
    round so that the start node appears again at the end. That repeated node is the reliable
    signal - and also the reason a closed loop currently dies with "NODEMAP AND SEGMENT LENGTH
    MISMATCH" further down, since the segment then holds one more entry than the mesh has nodes.
    A loop whose walk happens not to repeat the start node is caught by the adjacency test instead.
    """
    endpoint_pairs:set[tuple[int,int]]=set()
    for e in cache.elem_indices:
        a,b=int(e[0]),int(e[-1])
        endpoint_pairs.add((a,b))
        endpoint_pairs.add((b,a))
    return [len(seg)>2 and (seg[0]==seg[-1] or (seg[-1],seg[0]) in endpoint_pairs) for seg in segs]


def _check_zeta_is_invertible(mesh:InterfaceMesh,bind:int,context:str)->None:
    """Verify that the zeta values just written to `mesh` actually form a chart.

    The interpolation feeds zeta to oomph's ``locate_zeta`` on the interface mesh (see
    ``Mesh::nodal_interpolate_from`` in src/mesh.cpp), which treats zeta as a *global* coordinate and
    inverts it element by element. That only works if zeta is single-valued and monotone within each
    element. When it is not - a closed loop, or an interface that folds back along the axis chosen
    for :py:class:`AssignZetaCoordinatesByEulerianCoordinate` - nothing throws: the offending element
    simply matches queries it has no business matching, and the interpolated values come from the
    wrong part of the interface. This turns that into an error instead.

    See dev_docs/mesh_point_locator.md for why the real fix is a periodic zeta space rather than a
    validity check.
    """
    ranges:list[tuple[float,float]]=[]  # (zmin, zmax) per element, from its two end nodes
    for e in mesh.elements():
        zetas=[e.node_pt(ni).get_coordinates_on_boundary(bind)[0] for ni in range(e.nnode())]
        if len(zetas)<2:
            continue
        # Only the end values order the element; intermediate C2 nodes may legitimately sit anywhere
        # between them, so a strict sort check over all nodes would be too strong.
        lo,hi=min(zetas[0],zetas[-1]),max(zetas[0],zetas[-1])
        if hi-lo<=0:
            raise RuntimeError("The zeta coordinates assigned by "+context+" are degenerate on an element of '"+mesh.get_name()+"': both ends have zeta="+str(lo)+". zeta must vary along the interface.")
        ranges.append((lo,hi))

    if len(ranges)<2:
        return

    # A valid chart tiles its range: the elements' zeta intervals abut without overlapping, so their
    # widths sum to the total span. A seam element wrapping a closed loop, or an interface folding
    # back along the chosen axis, re-covers ground the other elements already own, and the sum
    # exceeds the span. This is a much better detector than "one element is suspiciously wide",
    # which false-positives on a coarse interface where one element legitimately owns a large share.
    # Disconnected segments only ever make the sum SMALLER than the span (the jump offsets are gaps
    # no element covers), so they never trip it.
    span=max(r[1] for r in ranges)-min(r[0] for r in ranges)
    if span<=0:
        return
    covered=sum(hi-lo for lo,hi in ranges)
    if covered>1.5*span:
        raise RuntimeError("The zeta coordinates assigned by "+context+" on '"+mesh.get_name()+"' are not invertible: the "+str(len(ranges))+" elements cover a total zeta length of "+str(covered)+" while spanning only "+str(span)+", so they overlap. This is what a closed loop (the seam element wrapping from the last zeta back to the first, covering the whole range on its own) or an interface folding back along the chosen axis looks like. Interpolation through such a zeta silently takes values from the wrong part of the interface.")


class AssignZetaCoordinatesBase(InterfaceEquations):
    #: Validate after each assignment that zeta is actually invertible. Only turn this off if you
    #: know the interpolation cannot be misled by the overlap it complains about.
    validate_zetas:bool=True

    def assign_zetas(self,mesh:InterfaceMesh)->None:
        raise RuntimeError("This function must be implemented!")

    def _refuse_if_distributed(self)->None:
        """Zeta assignment walks the *local* mesh only.

        Under ``--distribute`` that means the arclength restarts on every rank, the segment
        orientation heuristics act on partial curves, halo nodes never receive a consistent value,
        and a zeta owned by another rank is not found at interpolation time at all. None of that
        announces itself, so refuse rather than produce a plausible wrong answer. See
        dev_docs/mesh_point_locator.md phase 5.
        """
        problem=self.get_problem()
        if problem is not None and problem.is_distributed():
            raise self.add_exception_info(RuntimeError("Zeta coordinates cannot be assigned on a distributed mesh (mpirun --distribute): the assignment only sees this rank's part of the interface. See dev_docs/mesh_point_locator.md."))

    def after_mapping_on_macro_elements(self):
        self.assign_zetas(self.get_mesh())
        return super().after_mapping_on_macro_elements()

    def before_mesh_to_mesh_interpolation(self, eqtree: "EquationTree", interpolator: "BaseMeshToMeshInterpolator"):
        new_mesh=eqtree.get_mesh() # self.get_mesh()
        # This is only ever attached to InterfaceEquations, so the eqtree/interpolator here
        # always belong to a spatial interface mesh, never an ODEStorageMesh.
        assert isinstance(new_mesh,InterfaceMesh)
        old_base=interpolator.old
        assert not isinstance(old_base,ODEStorageMesh)
        old_mesh=old_base.get_mesh(new_mesh.get_name())
        assert isinstance(old_mesh,InterfaceMesh)
        self.assign_zetas(old_mesh)
        self.assign_zetas(new_mesh)
        return super().before_mesh_to_mesh_interpolation(eqtree,interpolator)

    def after_remeshing(self, eqtree: "EquationTree"):
        self.assign_zetas(self.get_mesh())
        return super().after_remeshing(eqtree)

class AssignZetaCoordinatesByEulerianCoordinate(AssignZetaCoordinatesBase):
    """Assigns a zeta coordinate which coindices with the Eulerian coordinate in a given direction. 
    This is useful for example to get a direct mapping from the old to the new boundary during remeshing and improves the interpolation along boundaries considerably. 
    
    Args:
        direction: The direction along which the zeta coordinate is assigned. Can be an integer (0,1,2) or a string ("x","y","z")
    """
    def __init__(self,direction:int | Literal["x", "y", "z"]):
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
        self._refuse_if_distributed()
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
        if self.validate_zetas:
            # An interface overhanging in the chosen direction produces a perfectly non-degenerate
            # zeta that is nonetheless not invertible, which the min/max check above cannot see.
            try:
                _check_zeta_is_invertible(mesh,bind,"AssignZetaCoordinatesByEulerianCoordinate(direction="+str(self.direction)+")")
            except RuntimeError as e:
                raise self.add_exception_info(e)


class AssignZetaCoordinatesByArclength(AssignZetaCoordinatesBase):
    """Assigns a zeta coordinate which is the arclength along the interface.
    This is useful for example to get a direct mapping from the old to the new boundary during remeshing and improves the interpolation along boundaries considerably.
    
    Args:
        start_near_point: If given, the zeta coordinate is assigned starting from the point closest to this point. Either this or sort_along_axis must be given, but not both.
        sort_along_axis: If given, the zeta coordinate is assigned starting from the point with the lowest/highest coordinate in this direction. Can be "x+","x-","y+","y-" for the respective directions. Either this or start_near_point must be given, but not both.
        normalized: If True, the zeta coordinate is normalized to [0,1]. If False, the zeta coordinate is the actual arclength along the interface.
        segment_jump_offset: This offset is added to the arclength when changing to a new segment of the boundary in case of disconnected curves.
        individual_segments: Mainly concerns normalization. If True, each segment is normalized individually, otherwise all segments are normalized together. 
    """
    def __init__(self,start_near_point:tuple[ExpressionOrNum, ExpressionOrNum] | None=None,sort_along_axis:Literal["x+", "x-", "y+", "y-"] | None=None,normalized:bool=True,segment_jump_offset:float=1.0,individual_segments:bool=True):
        super().__init__()
        self.start_near_point=start_near_point
        self.sort_along_axis=sort_along_axis
        self.normalized=normalized

        self.segment_jump_offset=segment_jump_offset  # Add this offset to the arclength when a new segment is started
        self.individual_segments=individual_segments # process and potentially normalize each segment individually. The total zeta parametrization is then by concatenation

        if (start_near_point is None and sort_along_axis is None) or (start_near_point is not None and sort_along_axis is not None):
            raise RuntimeError("Please add one parameter identifying the direction of the zeta parameterization")

    def assign_zetas(self,mesh:InterfaceMesh)->None:
        if mesh.get_dimension()!=1:
            raise RuntimeError("Currently only implemented for 1d interfaces meshes")
        self._refuse_if_distributed()
        bmesh=mesh.get_bulk_mesh()
        if isinstance(bmesh,InterfaceMesh):
            raise RuntimeError("Cannot do it, if the parent mesh is not a bulk mesh")
        bind=bmesh.get_boundary_index(mesh.get_name())
        cache=MeshDataCacheEntry(mesh,MeshDataCacheKey(nondimensional=True,tesselate_tri=True))

        pts=cache.get_coordinates()
        segs,_=cache.get_interface_line_segments()

        closed=_find_closed_segments(cache,segs)
        if any(closed):
            raise self.add_exception_info(RuntimeError("The interface '"+mesh.get_name()+"' contains "+str(sum(closed))+" closed loop(s), which cannot be parameterised by a single-valued zeta: the element closing the loop runs from the last zeta back to the first and therefore spans the whole range, so it matches essentially any query during interpolation and silently returns values from the opposite side of the loop. A periodic zeta space is the fix - see dev_docs/mesh_point_locator.md."))

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


        alengths_list:list[float]=[]
        ptinds:list[int]=[]
        aleng=0.0
        alengths:NPFloatArray | list[float]

        if not self.individual_segments:
            for seg in segs:
                oldx,oldy=pts[0,seg[0]],pts[1,seg[0]]
                for ptind in seg:
                    x,y=pts[0,ptind],pts[1,ptind]
                    dl=numpy.sqrt((x-oldx)**2+(y-oldy)**2)
                    aleng+=dl
                    alengths_list.append(aleng)
                    ptinds.append(ptind)
                    oldx,oldy=x,y
                aleng+=self.segment_jump_offset

            if self.normalized:
                alengths=numpy.array(alengths_list)/alengths_list[-1]
            else:
                alengths=alengths_list
        else:
            aleng_segs:list[NPFloatArray]=[]
            for seg in segs:
                alength_seg:list[float]=[]
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
                    alength_seg_arr=numpy.array(alength_seg)/alength_seg[-1]
                else:
                    alength_seg_arr=numpy.array(alength_seg)
                aleng_segs.append(alength_seg_arr)
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
        if self.validate_zetas:
            try:
                _check_zeta_is_invertible(mesh,bind,"AssignZetaCoordinatesByArclength")
            except RuntimeError as e:
                raise self.add_exception_info(e)




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