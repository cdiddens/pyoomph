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
 
from scipy import optimize
from ..expressions.generic import ExpressionNumOrNone,var,partial_t
from ..generic.codegen import InterfaceEquations,Equations
from ..meshes.mesh import AnySpatialMesh, InterfaceMesh, MeshFromTemplate2d,MeshFromTemplateBase,Node,Element,AnyMesh
from ..meshes.meshdatacache import MeshDataCache, MeshDataCacheEntry
from ..meshes.remesher import Remesher2d,RemesherBase, RemesherPointEntry
from ..meshes.interpolator import BaseMeshToMeshInterpolator,InternalInterpolator
from scipy.interpolate import InterpolatedUnivariateSpline,UnivariateSpline #type:ignore
import numpy
from ..typings import *

if TYPE_CHECKING:
    from ..generic.problem import Problem
    from ..generic.codegen import EquationTree,FiniteElementCodeGenerator


# Basic class to handle pinch-off and coalescence. Can be customized
class BaseAxisymmetricPinchoffAndCoalescence(InterfaceEquations):
    def __init__(self) -> None:
        super(BaseAxisymmetricPinchoffAndCoalescence, self).__init__()
        self._datacache=MeshDataCache(tesselate_tri=False,nondimensional=True) # access to the interface segements of the mesh
        self._has_coalescence:bool=False
        self._has_pinch_off:bool=False
        self._interface_segment_overlap_factor:float=0
        

    # This function can be overriden to nondimensionalize any spatial control parameters
    def ensure_nondimensional_distance_parameters(self,problem:"Problem")->None:
        pass

    # This function must be overridden to remove and add line segments to complete the pinch-off and coalescence events
    def handle_pinch_off_and_coalescence_during_remeshing(self,remesher:Remesher2d)->None:
        pass

    # This function must be overridden to identify any pinch-off events. It is called after each sucessfull solve
    # If we find a pinch-off, we must return True, else False
    def check_for_pinch_offs(self,segments:List[List[int]],coords:NPFloatArray,interface_data:MeshDataCacheEntry)->bool:
        return False

    # This function must be overridden to identify any coalescence events. It is called after each sucessfull solve
    # If we find a coalescence, we must return True, else False
    def check_for_coalescence(self,segments:List[List[int]],coords:NPFloatArray,interface_data:MeshDataCacheEntry,distfactor:float=1.0)->bool:
        return False        


    def get_segments_and_coords(self,mesh:InterfaceMesh,datacache:Optional[MeshDataCache]=None)->Tuple[List[List[int]],NPFloatArray,MeshDataCacheEntry]:        
        if datacache is None:
            datacache=self._datacache
        data=datacache.get_data(mesh)
        segments,_=data.get_interface_line_segments()        
        coords=data.get_coordinates()
        both_at_axis=[] # Segments which starts and ends on the axis of symmetry
        one_at_axis=[] # Segments which either start or end on the axis of symmetry
        for s in segments:
            if coords[0,s[0]]<1e-7 and coords[0,s[-1]]<1e-7:
                if coords[1,s[0]]>coords[1,s[-1]]:
                    both_at_axis.append(list(reversed(s)))
                else:
                    both_at_axis.append(s)
            elif coords[0,s[0]]<1e-7:
                one_at_axis.append(s)
            elif  coords[0,s[-1]]<1e-7: 
                one_at_axis.append(list(reversed(s)))
            else:
                raise RuntimeError("Found a free interface segment that does not hit the axis of symmetry")
        if len(one_at_axis)>2:
            raise RuntimeError("Found more than two segments that are only once connected to the axis of symmetry")
        both_at_axis=list(sorted(both_at_axis,key=lambda l:coords[1,l[0]])) # Sort lines by ascending y start
        # Insert the missing segments
        if len(both_at_axis)>0:
            for s in one_at_axis:
                if coords[1,s[0]]-self._interface_segment_overlap_factor<coords[1,both_at_axis[0][0]]:
                    both_at_axis.insert(0,list(reversed(s)))
                elif coords[1,s[0]]+self._interface_segment_overlap_factor>coords[1,both_at_axis[-1][-1]]:
                    both_at_axis.append(s)
                else: 
                    print("#x,y of segment only one time attached to the axis:")
                    for p in s:
                        print(coords[0,p],coords[1,p])
                    print()
                    print("#x,y of segment attached two times to the axis:")
                    for p in both_at_axis[0]:
                        print(coords[0,p],coords[1,p])
                    print()
                    raise RuntimeError("Segments that are only attached to the axis once are only allowed at the top or bottom")
        else:
            both_at_axis=list(sorted(one_at_axis,key=lambda s:coords[1,s[0]]))
            if len(both_at_axis)>1:
                both_at_axis[0]=list(reversed(both_at_axis[0]))
            
        return both_at_axis,coords,data
        

    def after_newton_solve(self)->None:
        mesh=self.get_my_domain()._mesh #type:ignore
        if not isinstance(mesh,InterfaceMesh):
            raise RuntimeError("Please attach the pinch-off and coalescence handler on an interface, not on a bulk domain")

        self.ensure_nondimensional_distance_parameters(mesh.get_problem())

        if mesh.get_element_dimension()!=1:
            raise RuntimeError("Axisymmetric pinch-off and coalescence only works on an InterfaceMesh of elemental dimension 1")
        if not isinstance(mesh.get_bulk_mesh(),MeshFromTemplateBase):
            raise RuntimeError("pinch-off and coalescence only works when attached to a 2d bulk mesh")

        self._datacache.clear()
        segments,coords,data=self.get_segments_and_coords(mesh)
        

        self._has_pinch_off=self.check_for_pinch_offs(segments,coords,data)
        self._has_coalescence=self.check_for_coalescence(segments,coords,data)                

        if self._has_pinch_off or self._has_coalescence:
            parent_domain=self.get_parent_domain()
            assert parent_domain is not None
            parent_mesh=parent_domain._mesh #type:ignore
            assert isinstance(parent_mesh,MeshFromTemplate2d)
            self.get_current_code_generator().get_problem()._domains_to_remesh.add(parent_mesh._templatemesh) #type:ignore



    # Return the boundary segments (i.e. separated lines)
    # These can be sorted, i.e. segments [index 0] and the orientation of the segment (i.e. the points  [index 1]) by ascending y
    def get_boundary_line_segments(self,remesher:Remesher2d,name:str,sort_segments_by_y:bool=True,sort_orientation_by_y:bool=True) -> List[List[RemesherPointEntry]]:
        my_name=self.get_current_code_generator().get_full_name() # full name (e.g. "liquid/interface")
        full_name=my_name.split("/")[0]+"/"+name
        lns=remesher._get_points_by_phys_name(full_name)
        if sort_segments_by_y:
            lns=list(sorted(lns,key=lambda l:min(l[0].y,l[-1].y)))
        else:
            return list(lns)
        if sort_orientation_by_y:
            return [list(line) if line[0].y<line[-1].y else list(reversed(line)) for line in lns]
        else:
            return lns


    # Obtain the name of the interface (e.g "interface"), the name of the axis (e.g. "axis") of symmetry
    def get_interface_and_axisymm_name(self,remesher:Remesher2d,cg:Optional["FiniteElementCodeGenerator"]=None)->Tuple[str,str]:
        if cg is None:
            cg=self.get_current_code_generator()
        my_name=cg.get_full_name()
        splt=my_name.split("/")
        other_interfaces = set([le.bname for le in remesher._line_entries if le.bname != splt[1]])

        cornerpts:Dict[str,List[List[RemesherPointEntry]]]={} # find all corners, i.e. points where we meet other interfaces
        for on in other_interfaces:        
            cornerpts[on] = remesher._get_points_by_phys_name("/".join(splt) + "/" + on)
        
        

        # Identify the axisymmetry boundary:
        axisymm_name=None
        for k,b in cornerpts.items():
            try:
                olines = remesher._get_points_by_phys_name(splt[0]+"/"+k)
            except:
                continue
            if all([abs(p.x)<1e-7  for l in olines for p in l]): # The axis of symmetry can be found, since all points have r=0
                if axisymm_name is None:
                    axisymm_name=k
                else:
                    raise RuntimeError("Cannot identify the axis of symmetry. It could be either "+axisymm_name+" or "+k) # Multiple candidates... Not good
        if axisymm_name is None:
            raise RuntimeError("Cannot find the axisymmetry line") # No axis of symmetry could be found

        return splt[-1],axisymm_name
    
    def get_opposite_and_opposite_axisymm_name(self,remesher:Remesher2d):
        if self.get_current_code_generator()._get_opposite_interface() is None:
            return None,None
        opp=self.get_current_code_generator()._get_opposite_interface()
        return self.get_interface_and_axisymm_name(remesher,opp)



    def remove_boundary_segment(self,remesher:Remesher2d,name:str,segment:List[RemesherPointEntry]) -> None:
        remesher._line_entries=list(filter(lambda le : le.bname!=name or not ((le.ptlist[0]==segment[0] and le.ptlist[-1]==segment[-1]) or (le.ptlist[0]==segment[-1] and le.ptlist[-1]==segment[0])) ,remesher._line_entries))

    def add_boundary_segment(self,remesher:Remesher2d,name:str,segment:List[RemesherPointEntry]):
        remesher.add_line_entry(segment,"spline" if len(segment)>3 else "line",name)

    def get_arclength_along_curve(self,lx:NPFloatArray,ly:NPFloatArray)->NPFloatArray:
            # reparametrize by the arclength
            arclength:float=0.0
            arclengths:List[float]=[]
            lastx:float=lx[0]
            lasty:float=ly[0]
            for x,y in zip(lx,ly):
                arclength+=numpy.sqrt((x-lastx)**2+(y-lasty)**2)
                lastx,lasty=x,y
                arclengths.append(arclength)            
            return numpy.array(arclengths)

    def quadratic_spline_roots(self,spl:UnivariateSpline)->NPFloatArray:
        roots = []
        knots = spl.get_knots()
        for a, b in zip(knots[:-1], knots[1:]):
            u, v, w = spl(a), spl((a + b) / 2), spl(b) #type:ignore
            t = numpy.roots([u + w - 2 * v, w - u, 2 * v]) #type:ignore
            t = t[numpy.isreal(t) & (numpy.abs(t) <= 1)] #type:ignore
            roots.extend(t * (b - a) / 2 + (b + a) / 2) #type:ignore
        return numpy.array(roots) #type:ignore            

    def setup_remeshing_size(self,remesher:RemesherBase,preorder:bool)->None:
        if not isinstance(remesher, Remesher2d):
            raise RuntimeError("Only works with Remesher2d at the moment")

        if preorder:
            return
        
        self.handle_pinch_off_and_coalescence_during_remeshing(remesher)



# Marking disjunct domains by an integer D0 field
# Can be used e.g. in integral expressions or similar
class DisjunctDomainMarker(Equations):
    def __init__(self,name:str,direction:Literal["up","down"]="up") -> None:
        super().__init__()
        self.name=name
        self.direction:Literal["up","down"]=direction # Direction of increasing marker

    def define_fields(self) -> None:
        self.define_scalar_field(self.name,"D0")

    def define_residuals(self) -> None:
        self.set_Dirichlet_condition(self.name,True) # Do not solve for it. Will be set by hand

    def _update_marker(self,mesh:AnySpatialMesh):
        if mesh.nelement()==0:
            return
        marker_index=mesh.element_pt(0).get_code_instance().get_discontinuous_field_index(self.name)
        # Reset all markers
        unhandled_nodes:Set[Node]=set()
        unhandled_elems:Set[Element]=set()
        nodes2elem:Dict[Node,List[Element]]={}
        # Create the look-up tables for unhandles nodes and node->elements map
        for e in mesh.elements():
            e.internal_data_pt(marker_index).set_value(0,-1)
            unhandled_elems.add(e)
            for ni in range(e.nnode()):
                n=e.node_pt(ni)
                unhandled_nodes.add(n)
                if n not in nodes2elem.keys():
                    nodes2elem[n]=[]
                nodes2elem[n].append(e)
        
        # Start over numbering the droplets
        domain_index=0
        self._max_droplet_index=0
        while len(unhandled_nodes)>0: # We still have nodes which do not belong to any domain
            # Find the node with maximum or minimum y
            ym=1e20*(-1 if self.direction=="down" else 1)
            startnode=None
            for n in unhandled_nodes:
                if (n.x(1)<ym if self.direction=="up" else n.x(1)>ym):
                    startnode=n
                    ym=n.x(1)
            if startnode is None:
                break

            # Flood-fill like algorithm
            checknodes:Set[Node]=set([startnode]) # seed the start node
            while len(checknodes)>0:
                nn=checknodes.pop() # get one node out of the bucket
                if nn in unhandled_nodes: # only check further if the node was not handled before
                    unhandled_nodes.remove(nn)
                    for e in nodes2elem[nn]: # go over all elements the node is part of
                        if e in unhandled_elems:
                            e.internal_data_pt(marker_index).set_value(0,domain_index) # mark the element
                            self._max_droplet_index=domain_index
                            unhandled_elems.remove(e)
                            for ni in range(e.nnode()):
                                n=e.node_pt(ni)
                                if n in unhandled_nodes:
                                    checknodes.add(n)
            domain_index+=1


    def on_apply_boundary_conditions(self, mesh: "AnyMesh"):
        assert isinstance(mesh,MeshFromTemplate2d)
        self._update_marker(mesh)
        #self.ensure_nondimensional_distance_parameters(mesh.get_problem())
        return super().on_apply_boundary_conditions(mesh)
        
    

        
# This is the specific class
class AxisymmetricPinchoffAndCoalescence(BaseAxisymmetricPinchoffAndCoalescence):
    def __init__(self,rmin:ExpressionNumOrNone,distmin:ExpressionNumOrNone,arclength_pinchoff_separation_factor:float=4,coalescence_distance_factor:float=1.5,coalescence_overlap_factor:Optional[float]=0.2,check_mesh_motion_direction:bool=True,assign_zeta_coordinates:bool=True) -> None:
        super().__init__()
        self.rmin=rmin # minimum radial distance (dimensional) for pinch-off
        self._rmin_nd=None # nondimensional radial distance for pinch-off
        self.distmin=distmin # if None, no coalescence
        self._distmin_nd=None # nondimensional distance for coalescence
        self.arclength_pinchoff_separation_factor=arclength_pinchoff_separation_factor # min distance along the interface between two pinch-off points
        self.coalescence_distance_factor=coalescence_distance_factor # factor to remove points when coalescing
        if coalescence_overlap_factor is not None:
            if coalescence_overlap_factor<0 or coalescence_overlap_factor>=1:
                raise RuntimeError("coalescence_overlap_factor must be >0 and <1")
        # when two ends are approaching, we make sure that no overlap can happen. If so, we reject the time step
        # close to 0 allows to almost overlap
        # close to 1 rejects time steps very close to the coalescence distance
        self.coalescence_overlap_factor=coalescence_overlap_factor # TODO
        self.assign_zeta_coordinates:bool=assign_zeta_coordinates

        # If set, we check whether the mesh is actually moving radially inward at a potential pinch-off
        # and whether two ends are actually moving towards each other for coalescence
        self.check_mesh_motion_direction=check_mesh_motion_direction
        
        # identified pinch-off points in (normalized arclength coordinate,y coordinate) per interface segment (sorted by ascending y-start)
        self._pinch_off_points:List[List[Tuple[float,float]]]=[]
        self._coalsecence_points:List[Tuple[float,float]]=[] # List of identified coalescence points (coordinate_y(lower_segment),coordinate_y(higher_segment))



    def define_additional_functions(self):
        if self.check_mesh_motion_direction:
            self.add_local_function("_topo_mesh_v_r", partial_t(var("mesh_x")) )
            self.add_local_function("_topo_mesh_v_z", partial_t(var("mesh_y")) )
        return super().define_additional_functions()


    # Nondimensionalize the distance parameters
    def ensure_nondimensional_distance_parameters(self,problem:"Problem")->None:
        if self.rmin is None:
            self._rmin_nd=None # Do not check for pinch-offs
        elif self._rmin_nd is None:
            self._rmin_nd=float(self.rmin/problem.get_scaling("spatial")) # nondim pinch-off radius

        if self.distmin is None:
            self._distmin_nd=None # do not check for coalescence
        elif self._distmin_nd is None:                
            self._distmin_nd=float(self.distmin/problem.get_scaling("spatial")) # nondim coalescence distance

        if self._distmin_nd is not None:
            self._interface_segment_overlap_factor=self._distmin_nd



    # Find pinch-off points for all interface segments
    def check_for_pinch_offs(self,segments:List[List[int]],coords:NPFloatArray,interface_data:MeshDataCacheEntry)->bool:

        # Clear the pinch-offs
        self._pinch_off_points=[[] for _i in range(len(segments))]

        if self._rmin_nd is None: # Should not check for pinch-offs
            return False 

        has_pinch_off:bool=False
        for linenum,l in enumerate(segments):           
            # reparametrize by the arclength
            arclengths=self.get_arclength_along_curve(coords[0,l],coords[1,l])
            arclength:float=arclengths[-1]
            arclengths/=arclength # Normalized arclength            

            intr=InterpolatedUnivariateSpline(arclengths,coords[0,l],k=3) # radius as function of the normalized arclength
            inty=InterpolatedUnivariateSpline(arclengths,coords[1,l],k=3) # axial coordinate
            dinter=intr.derivative() # first and second derivative of the radial position
            ddinter=dinter.derivative()

            roots=self.quadratic_spline_roots(dinter) # Find zeros of r'(arclength) -> extrema of r(arclength)
            distf=self.arclength_pinchoff_separation_factor*self._rmin_nd/arclength # Required distance

            if self.check_mesh_motion_direction:
                mesh_velo_r=interface_data.get_data("_topo_mesh_v_r")
                assert mesh_velo_r is not None
                int_mesh_velo_r=InterpolatedUnivariateSpline(arclengths,mesh_velo_r[l])
            else:
                int_mesh_velo_r=None

            # Go over all exterma of r(arclength)
            for ndarclength in sorted(roots):
                # if r<pinch_off distance and positive curvature (i.e. r(arclength) has a minimum)
                # and we are sufficiently away from the ends of the segment
                if intr(ndarclength)<self._rmin_nd and ddinter(ndarclength)>0 and ndarclength>distf and ndarclength<1-distf:
                    # Check if we are close to a previous point
                    if len(self._pinch_off_points[linenum])==0 or ndarclength-self._pinch_off_points[linenum][-1][0]>distf:
                        if not self.check_mesh_motion_direction or int_mesh_velo_r(ndarclength)<0: # checking radial mesh velocity if desired
                            # Found a pinch-off. Add it to the list
                            self._pinch_off_points[linenum].append((ndarclength,float(inty(ndarclength))))
                            has_pinch_off=True 
        if has_pinch_off:
            print("Must remesh due to pinch-off at non-dimensional (arclength,y) values ",self._pinch_off_points)
            
        return has_pinch_off



    def before_newton_convergence_check(self,eqtree:"EquationTree") -> bool:
        # Check and prevent overlap by rejecting time steps
        if self._distmin_nd is None or self.coalescence_overlap_factor is None:
            return super().before_newton_convergence_check(eqtree)
        
        self._datacache.clear()
        mesh=eqtree.get_mesh()
        assert isinstance(mesh,InterfaceMesh)
        data=self._datacache.get_data(mesh)
        segments,_=data.get_interface_line_segments()        
        coords=data.get_coordinates()
        segments=list(sorted(segments,key=lambda l:min(coords[1,l[0]],coords[1,l[-1]]))) # Sort lines by ascending y start
        for li,seg in enumerate(segments):
            if coords[1,seg[0]]>coords[1,seg[-1]]:
                segments[li]=list(reversed(seg))
        has_coalescence=self.check_for_coalescence(segments,coords,data,self.coalescence_overlap_factor)
        if has_coalescence:
            print("This step would invoke overlap! Rejecting")            
            return False
        return super().before_newton_convergence_check(eqtree)

    # Check if we have coalescence
    def check_for_coalescence(self, segments: List[List[int]], coords: NPFloatArray, interface_data: MeshDataCacheEntry,distfactor:float=1.0) -> bool:
        self._coalsecence_points=[]
        if self._distmin_nd is None:
            return False
       
        
        if self.check_mesh_motion_direction:
            mesh_velo_z=interface_data.get_data("_topo_mesh_v_r")
            assert mesh_velo_z is not None

        has_coalescence:bool=False
        for l1,l2 in zip(segments,segments[1:]):
            e1=coords[1,l1[-1]]
            s2=coords[1,l2[0]]
            dy=s2-e1 # distance of the end of the lower segment and the start of the upper segment
            if self.check_mesh_motion_direction:
                uz_e1=mesh_velo_z[l1[-1]] # mesh z-velocity of the upper point of the lower segment
                uz_s2=mesh_velo_z[l2[0]] # mesh z-velocity of the lower point of the upper segment
                if uz_s2-uz_e1>0:
                    continue # No need to coalesce here, if they are moving appart
            if dy<self._distmin_nd*distfactor:
                has_coalescence=True
                self._coalsecence_points.append((s2,e1))
                print("Must remesh due to coalescence at non-dimensional y",0.5*(s2+e1))
        
        return has_coalescence
        


    def before_mesh_to_mesh_interpolation(self, eqtree: "EquationTree", interpolator: "BaseMeshToMeshInterpolator"):
        self.ensure_nondimensional_distance_parameters(eqtree.get_mesh().get_problem())
        assert isinstance(interpolator,InternalInterpolator)
        remesher=eqtree.get_mesh().get_bulk_mesh()._templatemesh.remesher
        assert isinstance(remesher,Remesher2d)
        interface_name,axisymm_name=self.get_interface_and_axisymm_name(remesher)
        # Tell the remesher to ignore points at the intersection of the axis of symmetry and the free interface from interpolation
        # if there is no point on the old intersection of these interfaces within the range. Will happen at pinch-offs
        # In that case, we just take the values from the closes point on the old interface or axis itself, not on the mutual point
        interpolator.boundary_max_distances[interface_name+'/'+axisymm_name]=self._rmin_nd*2
        interpolator.boundary_max_distances[axisymm_name+'/'+interface_name]=self._rmin_nd*2
    #    print("IMDIST_EE",interpolator.boundary_max_distances)
        if self.assign_zeta_coordinates:
            new_mesh=self.get_mesh()
            old_mesh=interpolator.old.get_mesh(new_mesh.get_name())
            old_data=self.assign_zetas(old_mesh)            
            if self._has_pinch_off or self._has_coalescence:
                #for d in old_data:
                #    print(d)
                print("Zeta assignment is different due to either pinchoff or coalescence:",self._has_pinch_off,self._has_coalescence)
                #print(old_data[:,0])
                x_of_zeta=InterpolatedUnivariateSpline(old_data[:,0],old_data[:,1],k=min(len(old_data),3))
                y_of_zeta=InterpolatedUnivariateSpline(old_data[:,0],old_data[:,2],k=min(len(old_data),3))
                bmesh=new_mesh.get_bulk_mesh()
                bind=bmesh.get_boundary_index(new_mesh.get_name())
                def minifunc(zeta,x,y):
                    return (x_of_zeta(zeta[0])-x)**2+(y_of_zeta(zeta[0])-y)**2
                for n in new_mesh.nodes():
                    x,y=n.x(0),n.x(1)
                    guess_zeta=0
                    guess_dist=1e20
                    for d in old_data:
                        dz=minifunc([d[0]],x,y)
                        if dz<guess_dist:
                            guess_dist=dz
                            guess_zeta=d[0]                    
                    optzeta=optimize.minimize(minifunc,[guess_zeta],args=(x,y))
                    n.set_coordinates_on_boundary(bind,[optzeta.x[0]])
                bmesh.boundary_coordinate_bool(bind)
                
            else:
                self.assign_zetas(new_mesh)            

        return super().before_mesh_to_mesh_interpolation(eqtree, interpolator)

    def assign_zetas(self,mesh:InterfaceMesh):
        bmesh=mesh.get_bulk_mesh()
        bind=bmesh.get_boundary_index(mesh.get_name())
        cache=MeshDataCache(True,True)
        segs,pts,data=self.get_segments_and_coords(mesh,cache)        

        nodemap=mesh.fill_node_index_to_node_map()
        if len(nodemap)!=sum(len(seg) for seg in segs):
            print("NODEMAP",nodemap)
            print("SEGS",segs)
            raise RuntimeError("NODEMAP AND SEGMENT LENGTH MISMATCH")


        alengths:List[float]=[]
        ptinds:List[int]=[]
        aleng=0.0

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
            alength_seg=numpy.array(alength_seg)/alength_seg[-1]            
            aleng_segs.append(alength_seg)
        offs=0.0
        for i in range(len(aleng_segs)):
            aleng_segs[i]+=offs
            offs=aleng_segs[i][-1]+1 # Here the pinch-off and coalescence dynamics has to go
        alengths=numpy.concatenate(aleng_segs)

        res:List[Tuple[float,float,float]]=[]
        for al,pti in zip(alengths,ptinds):
            n=nodemap[pti]
            n.set_coordinates_on_boundary(bind,[al])
            res.append((al,n.x(0),n.x(1),))
            
        bmesh.boundary_coordinate_bool(bind)
        return numpy.array(res)


    def after_mapping_on_macro_elements(self):
        if self.assign_zeta_coordinates:
            if not (self._has_coalescence or self._has_pinch_off):
                self.assign_zetas(self.get_mesh())
        return super().after_mapping_on_macro_elements()    
    
    def after_remeshing(self, eqtree: "EquationTree"):
        if self.assign_zeta_coordinates:
            if not (self._has_coalescence or self._has_pinch_off):
                self.assign_zetas(self.get_mesh())
        return super().after_remeshing(eqtree)

    def handle_pinch_off_and_coalescence_during_remeshing(self,remesher:Remesher2d)->None:
        if not self._has_pinch_off and not self._has_coalescence:
            return 

        interface_name,axisymm_name=self.get_interface_and_axisymm_name(remesher)
        opp_interface_name,opp_axisymm_name=self.get_opposite_and_opposite_axisymm_name(remesher)
        

        interface_segments=self.get_boundary_line_segments(remesher,interface_name)
        axisymm_lines=self.get_boundary_line_segments(remesher,axisymm_name)
        if opp_interface_name:
            opp_axisymm_lines=self.get_boundary_line_segments(remesher,opp_interface_name)

        if len(axisymm_lines)!=len(interface_segments):
            # This is required to match for the connection of the axisymmetric lines later on
            raise RuntimeError("Strange: Mismatch in axisymmetric line segments and interface line segments: "+str(len(axisymm_lines))+" vs. "+str(len(interface_segments)))

        # Freshly pinched-off points should never coalesce in the same step
        ignore_coalescence_at_points:Set[RemesherPointEntry]=set()
        axi_points:List[RemesherPointEntry]=[]

        # Do the pinch-off
        if self._has_pinch_off:
            if len(self._pinch_off_points)!=len(interface_segments):
                print("PINCH-OFF INFORMATION: ",self._pinch_off_points)
                print("LINE SEGMENTS:")
                for l in interface_segments:
                    for p in l:
                        print("",p.x,p.y)
                    print()
                raise RuntimeError("Strange: Mismatch in interface segments between pinch-off information and actual segments: "+str(len(self._pinch_off_points))+" vs. "+str(len(interface_segments)))

            assert self._rmin_nd is not None

            for lineind,l in enumerate(interface_segments): # iterate over all interface lines               

                # reparametrize by the arclength
                arclengths=self.get_arclength_along_curve(numpy.array([p.x for p in l]),numpy.array([p.y for p in l]))
                arclength:float=arclengths[-1]
                arclengths=arclengths/arclength # Normalized arclength

                # identified arclengths and y values of pinch-offs at this segment
                pinch_off_arclengths=self._pinch_off_points[lineind]

                if len(pinch_off_arclengths)==0:
                    continue # No pinch-off happening within this interface segment

                # Some distance factors to remove points
                distf0=0.5*self._rmin_nd
                distf1=0.8*self._rmin_nd
                distf_al=(distf1+2*self._rmin_nd)/arclength

                # New interface segments to create
                new_interface_segments:List[List[RemesherPointEntry]] = [[] for _i in range(len(pinch_off_arclengths)+1)]
                
                currind=0 # Current index to the handled pinch-off position
                # Iterate over all points
                
                for p,al in zip(l,arclengths):
                    if currind<len(pinch_off_arclengths):
                        if al<pinch_off_arclengths[currind][0]-distf_al:
                            new_interface_segments[currind].append(p)
                        elif al>=pinch_off_arclengths[currind][0]+distf_al:
                            currind+=1
                            new_interface_segments[currind].append(p)
                    else:
                        new_interface_segments[currind].append(p)
                
                # Complete the curves by adding new points
                for i in range(len(new_interface_segments)):
                    if i>0:
                        #Add a start interpolation point
                        new_interface_segments[i].insert(0, remesher.add_point_entry(self._rmin_nd, pinch_off_arclengths[i - 1][1] + distf1, 0,size=new_interface_segments[i][0].size))
                        new_interface_segments[i].insert(0,remesher.add_point_entry(0, pinch_off_arclengths[i - 1][1] +distf0, 0, size=new_interface_segments[i][0].size))
                        ignore_coalescence_at_points.add(new_interface_segments[i][0])
                        axi_points.append(new_interface_segments[i][0])
                    if i+1<len(new_interface_segments):
                        #adding end interpolation points
                        new_interface_segments[i].append(remesher.add_point_entry(self._rmin_nd, pinch_off_arclengths[i][1] - distf1, 0, size=new_interface_segments[i][-1].size))
                        new_interface_segments[i].append(remesher.add_point_entry(0, pinch_off_arclengths[i][1] - distf0,0,size=new_interface_segments[i][-1].size))
                        ignore_coalescence_at_points.add(new_interface_segments[i][-1])
                        axi_points.append(new_interface_segments[i][-1])


                # Remove the old part from the line entries
                self.remove_boundary_segment(remesher,interface_name,l)
                
                # Add the new line entries
                for r in new_interface_segments:
                    self.add_boundary_segment(remesher,interface_name,r)
                    


                # Patch the axisymmetric line segments
                lax=axisymm_lines[lineind]
                self.remove_boundary_segment(remesher,axisymm_name,lax)
                startpt,endpt=lax[0],lax[-1]
                for axisegind in range(len(new_interface_segments)):
                    stp=new_interface_segments[axisegind][0] if axisegind>0 else startpt
                    ept=new_interface_segments[axisegind][-1] if axisegind+1<len(new_interface_segments) else endpt
                    self.add_boundary_segment(remesher,axisymm_name,[stp,ept])
                    if opp_axisymm_name is not None:
                        if axisegind+1<len(axi_points):
                            self.add_boundary_segment(remesher,opp_axisymm_name,[axi_points[axisegind],axi_points[axisegind+1]])




        # Handle coalescence
        if self._has_coalescence and self._distmin_nd is not None:

            def update_segments() -> tuple[List[List[RemesherPointEntry]], List[List[RemesherPointEntry]]]:
                axisymm_lines=self.get_boundary_line_segments(remesher,axisymm_name)
                interface_segments=self.get_boundary_line_segments(remesher,interface_name)
                if len(axisymm_lines)!=len(interface_segments):
                    raise RuntimeError("Strange: Expected axisymmetric segments and interface segments to have the same length, but got "+str(len(axisymm_lines))+" vs. "+str(len(interface_segments)))
                return axisymm_lines,interface_segments

            # get the lines once more        
            axisymm_lines,interface_segments=update_segments()
            currind=0
            while currind+1<len(axisymm_lines):
                # Get the current and next part of the interfaces
                l1=axisymm_lines[currind]
                l2=axisymm_lines[currind+1]
                s1=interface_segments[currind]
                s2=interface_segments[currind+1]

                # Distance betweem them
                dy=l2[0].y-l1[-1].y
                if dy<self._distmin_nd and not (s2[0] in ignore_coalescence_at_points and s1[-1] in ignore_coalescence_at_points):
                    center_y=0.5*(l2[0].y+l1[-1].y)
                    print("Handle coalescence at non-dimensional y  ",center_y,"NEWPT",s2[0].y,s1[-1].y,s2[0] in ignore_coalescence_at_points ,s1[-1] in ignore_coalescence_at_points)

                    # Remove the two axisymmetric lines
                    self.remove_boundary_segment(remesher,axisymm_name,l1)
                    self.remove_boundary_segment(remesher,axisymm_name,l2)
                    # and make one out of it
                    self.add_boundary_segment(remesher,axisymm_name,[l1[0],l2[-1]])
                    
                    # Remove the two interface segments
                    self.remove_boundary_segment(remesher,interface_name,s1)
                    self.remove_boundary_segment(remesher,interface_name,s2)                   
                    
                    if self._rmin_nd is None:
                        remdist_r=2*self._distmin_nd*self.coalescence_distance_factor
                        remdist_y=2*self._distmin_nd*self.coalescence_distance_factor
                    else:
                        remdist_r=2*self._rmin_nd*self.coalescence_distance_factor
                        remdist_y=2*self._rmin_nd*self.coalescence_distance_factor

                    # Make to arclength parameterized splines. Check the distance
                    s1rev=list(reversed(s1))
                    s1rx=numpy.array([p.x for p in s1rev])
                    s1ry=numpy.array([p.y for p in s1rev])
                    s2x=numpy.array([p.x for p in s2])
                    s2y=numpy.array([p.y for p in s2])
                    als1=self.get_arclength_along_curve(s1rx,s1ry)
                    als2=self.get_arclength_along_curve(s2x,s2y)
                    maxal=min(als1[-1],als2[-1])*0.5 # arclength range to sample
                    int_s1_x=InterpolatedUnivariateSpline(als1,s1rx,k=min(3,len(als1)))
                    int_s1_y=InterpolatedUnivariateSpline(als1,s1ry,k=min(3,len(als1)))
                    int_s2_x=InterpolatedUnivariateSpline(als2,s2x,k=min(3,len(als2)))
                    int_s2_y=InterpolatedUnivariateSpline(als2,s2y,k=min(3,len(als2)))
                    numsampls=1000
                    alsampls=numpy.linspace(0,maxal,numsampls)
                    dists=numpy.sqrt((int_s1_x(alsampls)-int_s2_x(alsampls))**2+(int_s1_y(alsampls)-int_s2_y(alsampls))**2)                    
                    # Find the last index that is sufficiently away
                    maxind=len(dists)-1
                    while dists[maxind-1]>remdist_r and maxind>1:
                        maxind-=1
                    alcrop=alsampls[maxind]
                    lastind_i1=len(s1)-numpy.argwhere(als1>alcrop)[0][0]
                    lastind_i2=numpy.argwhere(als2>alcrop)[0][0]
                    
            
                    if False:
                        lastind_i1=len(s1)-1
                        while (s1[lastind_i1].x<remdist_r or s1[lastind_i1].y>center_y-remdist_y)  and lastind_i1>1:
                            lastind_i1-=1
                        lastind_i2=0
                        while (s2[lastind_i2].x<remdist_r or s2[lastind_i2].y<center_y+remdist_y) and lastind_i2+1<len(s2):
                            lastind_i2+=1

                    #print("LASTINDS VS LENG",lastind_i1,len(s1),lastind_i2,len(s2))
                    new_seg:List[RemesherPointEntry]=[]
                    for i in range(lastind_i1+1):
                        new_seg.append(s1[i])
                    for i in range(min(max(1,lastind_i2-1),len(s2)-1),len(s2)):
                        new_seg.append(s2[i])

                    print("NEWSEG",new_seg,[(p.x,p.y) for p in new_seg])                    
                    
                    self.add_boundary_segment(remesher,interface_name,new_seg)                    

                    axisymm_lines,interface_segments=update_segments() # Update the segments, but do not increase the index
                else:
                    currind=currind+1 # increase the index if there is no coalescence

