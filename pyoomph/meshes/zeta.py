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
from .ordering import SortAlongAxis, check_sorting_arguments, sort_line_segments
from ..generic.mpi import get_mpi_any, get_mpi_nproc, get_mpi_rank, get_mpi_world_comm, get_mpi_max, get_mpi_min, get_mpi_sum, mpi_share_root_failure
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


def _walk_closed_loop(cache:MeshDataCacheEntry)->list[int] | None:
    """Order the nodes of a closed interface loop by walking the element connectivity.

    ``get_interface_line_segments`` is not usable for this. Its walk is written around open curves -
    it looks for an endpoint of degree one to start from and falls back to an arbitrary node when
    there is none - and on a loop it can emit a segment whose last entries are not adjacent at all.
    Measured on a remeshed disc: the returned order jumped half way across the circle twice near the
    end, inflating the accumulated arclength by a factor 1.6 and, since zeta is that arclength,
    corrupting the whole parameterisation. Walking the elements is unambiguous, so do that.

    Returns the node indices in order around the loop, each appearing once, or None if the boundary
    is not a single closed loop.
    """
    elems=[[int(i) for i in e] for e in cache.elem_indices]
    if not elems:
        return None
    at_end:dict[int,list[int]]={}
    for ei,e in enumerate(elems):
        at_end.setdefault(e[0],[]).append(ei)
        at_end.setdefault(e[-1],[]).append(ei)
    # Every node of a closed loop is shared by exactly two elements; anything else is not one.
    if any(len(v)!=2 for v in at_end.values()):
        return None

    order:list[int]=[]
    ei=0
    node=elems[0][0]
    visited:set[int]=set()
    while ei not in visited:
        visited.add(ei)
        e=elems[ei]
        # orient this element so that it starts at `node`
        chain=e if e[0]==node else list(reversed(e))
        order.extend(chain[:-1])   # the far end is the next element's start
        node=chain[-1]
        nxt=[k for k in at_end[node] if k!=ei]
        if len(nxt)!=1:
            return None
        ei=nxt[0]
    if len(visited)!=len(elems) or len(set(order))!=len(order):
        return None  # more than one loop, or the walk did not close cleanly
    return order


def _loop_seam_anchor(loop:list[int],pts,direction:tuple[float,float])->tuple[int,float]:
    """Where to start measuring arclength around a closed loop.

    It has to be a point on the CURVE, not one of its nodes. A node-quantised seam differs between
    the old and the new mesh by up to one element, which shifts the whole parameterisation by that
    much and defeats the purpose of having one. Taking the outermost intersection of the loop with a
    ray from its centroid gives a point that is defined by the geometry alone, so two discretisations
    of the same curve agree on it to O(h^2).

    Returns (index into `loop` of the segment containing the anchor, fraction along that segment).
    """
    n=len(loop)
    cx=sum(pts[0,i] for i in loop)/n
    cy=sum(pts[1,i] for i in loop)/n
    dx,dy=direction
    best=None
    for k in range(n):
        a,b=loop[k],loop[(k+1)%n]
        ax,ay=pts[0,a]-cx,pts[1,a]-cy
        bx,by=pts[0,b]-cx,pts[1,b]-cy
        # solve  a + u*(b-a) = t*d  for u in [0,1], t > 0
        ex,ey=bx-ax,by-ay
        den=dx*ey-dy*ex
        if abs(den)<1e-300:
            continue
        # Solving a + u*e = t*d for u gives u = (dy*ax - dx*ay) / (dx*ey - dy*ex). Getting the
        # numerator's sign wrong here does not fail loudly: every real crossing comes out with
        # u in [-1,0], is rejected as "outside the edge", and the search falls through to the
        # extremal-node fallback below - which is node-quantised, so the seam then lands on a
        # different node in the old and the new mesh and the whole parameterisation is offset by
        # about one element.
        u=(dy*ax-dx*ay)/den
        if u<0.0 or u>1.0:
            continue
        px,py=ax+u*ex,ay+u*ey
        t=px*dx+py*dy
        if t<=0.0:
            continue
        if best is None or t>best[2]:
            best=(k,u,t)
    if best is None:
        # No crossing at all (the centroid is outside a strongly non-convex loop). Fall back to the
        # extremal node along the direction, which is still geometric, just only O(h) stable.
        k=max(range(n),key=lambda i: pts[0,loop[i]]*dx+pts[1,loop[i]]*dy)
        return k,0.0
    return best[0],best[1]


def _check_zeta_is_invertible(mesh:InterfaceMesh,bind:int,context:str,period:float=0.0)->None:
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
        if period>0:
            # Read each element on its own branch, exactly as the locator does: the element closing
            # a loop runs from z_last to z_first + period, not backwards across the whole range.
            ref=zetas[0]
            zetas=[z-period*round((z-ref)/period) for z in zetas]
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
    if period>0:
        span=period  # the elements tile the full period, which the min/max above cannot see
    if span<=0:
        return
    covered=sum(hi-lo for lo,hi in ranges)
    if covered>1.5*span:
        raise RuntimeError("The zeta coordinates assigned by "+context+" on '"+mesh.get_name()+"' are not invertible: the "+str(len(ranges))+" elements cover a total zeta length of "+str(covered)+" while spanning only "+str(span)+", so they overlap. This is what a closed loop (the seam element wrapping from the last zeta back to the first, covering the whole range on its own) or an interface folding back along the chosen axis looks like. Interpolation through such a zeta silently takes values from the wrong part of the interface.")


def assign_zetas_from_position_table(mesh:InterfaceMesh,bind:int,table:"NPFloatArray | Sequence[tuple[float,float,float]]",tol:float=1e-8)->None:
    """Set zeta on this rank's nodes of ``mesh`` from an ``(N,3)`` table of ``(x, y, zeta)``.

    The table is addressed by position rather than by index because whoever built it numbers the
    points of the *whole* interface, which says nothing about this rank's node order: the merged data
    of a distributed run, or - see :py:mod:`pyoomph.meshes.axisymm_topology` - the polylines a
    topological surgery was planned on. Every local node is one of those points, so the match is exact
    up to the last bits; ``tol`` is relative to the extent of the table, and anything further away
    means the table does not describe this interface and is an error rather than a nearest neighbour
    worth taking.

    Only this rank's nodes are touched, so this is MPI-safe by construction.
    """
    from scipy.spatial import cKDTree
    tab=numpy.asarray(table,dtype=float).reshape(-1,3)
    coords=tab[:,0:2]
    zetas=tab[:,2]
    nodes=[e.node_pt(ni) for e in mesh.elements() for ni in range(e.nnode())]
    if not len(coords) or not nodes:
        return
    dist,idx=cKDTree(coords).query(numpy.array([[n.x(0),n.x(1)] for n in nodes]))
    extent=max(float(numpy.amax(numpy.abs(coords))),1.0)
    if float(numpy.amax(dist))>tol*extent:
        raise RuntimeError("Assigning zeta on '"+mesh.get_name()+"' from a position table: a node of this rank is "+str(float(numpy.amax(dist)))+" away from the nearest point of the table, which should be zero. The table does not describe the same interface.")
    for n,i in zip(nodes,idx):
        n.set_coordinates_on_boundary(bind,[float(zetas[i])])


def assign_zetas_by_polyline_projection(mesh:InterfaceMesh,bind:int,chains:Sequence[tuple["NPFloatArray","NPFloatArray"]],tol_factor:float=0.5,jump_factor:float=10.0)->None:
    """Set zeta on this rank's nodes of ``mesh`` by projecting them onto zeta-carrying polylines.

    ``chains`` are ``(points (M,2), zeta (M,))`` pairs, one per connected piece. Each node is matched
    to the closest point of the closest segment over all chains and gets the zeta interpolated
    linearly along that segment.

    Unlike :py:func:`assign_zetas_from_position_table` this cannot match exactly, and must not try to:
    the polylines are a *plan* that a mesh generator then meshed, so a new node sits on the curve
    through them and sags off the chords by O(h^2 * curvature). The tolerance is therefore a fraction
    of the length of the segment that was hit - the only length scale available locally - rather than
    an absolute number.

    ``jump_factor``: a segment whose zeta span exceeds this multiple of the median is a DISCONTINUITY
    of the chart rather than a stretch of it - the offset between two formerly separate pieces of
    interface that a coalescence bridged. Interpolating across it would hand a node a zeta no source
    point has, so a node landing there takes the value of the nearer end instead.

    Only this rank's nodes are touched, so this is MPI-safe by construction.
    """
    starts:list[NPFloatArray]=[]
    ends:list[NPFloatArray]=[]
    zst:list[NPFloatArray]=[]
    zen:list[NPFloatArray]=[]
    for pts,zet in chains:
        p=numpy.asarray(pts,dtype=float).reshape(-1,2)
        z=numpy.asarray(zet,dtype=float).reshape(-1)
        if len(p)!=len(z):
            raise RuntimeError("assign_zetas_by_polyline_projection got a polyline of "+str(len(p))+" points with "+str(len(z))+" zeta values")
        if len(p)<2:
            continue
        starts.append(p[:-1])
        ends.append(p[1:])
        zst.append(z[:-1])
        zen.append(z[1:])
    if not starts:
        raise RuntimeError("assign_zetas_by_polyline_projection got no polyline with at least two points")
    A=numpy.vstack(starts)
    E=numpy.vstack(ends)-A
    za=numpy.concatenate(zst)
    zb=numpy.concatenate(zen)
    L2=numpy.einsum("ij,ij->i",E,E)
    L=numpy.sqrt(L2)
    L2safe=numpy.maximum(L2,1e-300)
    dz=numpy.abs(zb-za)
    is_jump=dz>jump_factor*max(float(numpy.median(dz)),1e-300)

    # Nodes shared by two elements are visited twice and simply written twice; deduplicating them
    # would mean identity-comparing the Python wrappers, which are not the same object for the same
    # node on every access.
    nodes=[e.node_pt(ni) for e in mesh.elements() for ni in range(e.nnode())]
    if not nodes:
        return
    P=numpy.array([[n.x(0),n.x(1)] for n in nodes],dtype=float)
    # Blocked, because the full node x segment distance matrix is the one thing here that can grow
    # quadratically on a well resolved interface.
    for beg in range(0,len(P),256):
        blk=P[beg:beg+256]
        d=blk[:,None,:]-A[None,:,:]
        t=numpy.einsum("nsj,sj->ns",d,E)/L2safe[None,:]
        numpy.clip(t,0.0,1.0,out=t)
        diff=d-t[:,:,None]*E[None,:,:]
        dist2=numpy.einsum("nsj,nsj->ns",diff,diff)
        j=numpy.argmin(dist2,axis=1)
        rows=numpy.arange(len(blk))
        dist=numpy.sqrt(dist2[rows,j])
        tt=t[rows,j]
        allowed=tol_factor*L[j]
        bad=numpy.nonzero(dist>allowed)[0]
        if len(bad):
            k=int(bad[int(numpy.argmax(dist[bad]-allowed[bad]))])
            raise RuntimeError("Assigning zeta on '"+mesh.get_name()+"' by projection: the node at ("+str(float(blk[k,0]))+", "+str(float(blk[k,1]))+") is "+str(float(dist[k]))+" away from the nearest of the given polylines, more than the "+str(float(allowed[k]))+" allowed there ("+str(tol_factor)+" of the segment it hit). The polylines do not describe this interface.")
        zz=numpy.where(is_jump[j],numpy.where(tt<0.5,za[j],zb[j]),za[j]+tt*(zb[j]-za[j]))
        for k,n in enumerate(nodes[beg:beg+256]):
            n.set_coordinates_on_boundary(bind,[float(zz[k])])


class AssignZetaCoordinatesBase(InterfaceEquations):
    #: Validate after each assignment that zeta is actually invertible. Only turn this off if you
    #: know the interpolation cannot be misled by the overlap it complains about.
    validate_zetas:bool=True

    def assign_zetas(self,mesh:InterfaceMesh)->None:
        raise RuntimeError("This function must be implemented!")

    def _needs_merged_interface(self,mesh:InterfaceMesh)->bool:
        """Whether zeta has to be built from the globally merged interface rather than the local one.

        Zeta is a chart over the WHOLE interface, so an assignment that reads only this rank's part
        of it is not a piece of the answer - the arclength restarts on every rank and the segment
        orientation heuristics act on a partial curve. An assignment that reads only node positions
        (see :py:class:`AssignZetaCoordinatesByEulerianCoordinate`) needs none of this.

        Collective, and answered identically on every rank, since it decides whether this rank enters
        the merge: the gate is Problem-wide, and the local answer is agreed inside it because an
        interface mesh a rank holds no part of does not carry the distributed flag. Same reasoning as
        MeshedMeshTemplate._resolve_mesh_for_boundary_coordinates.
        """
        problem=self.get_problem()
        if problem is None or not problem.is_distributed() or get_mpi_nproc()<=1:
            return False
        from .meshdatamerge import needs_merging
        return get_mpi_any(needs_merging(mesh))

    def after_mapping_on_macro_elements(self):
        mesh=self.get_mesh()
        # Between the interpolation hooks and the transfer itself, which is exactly the window in
        # which a topological-change handler's chart has to survive. Without this test the assigner
        # re-charted the NEW mesh here while the OLD one kept the handler's chart, and the transfer
        # then read the two through different parameterisations.
        if isinstance(mesh,InterfaceMesh) and mesh._zeta_chart_overridden:
            return super().after_mapping_on_macro_elements()
        self.assign_zetas(mesh)
        return super().after_mapping_on_macro_elements()

    def _before_mesh_to_mesh_interpolation(self, eqtree: "EquationTree", interpolator: "BaseMeshToMeshInterpolator"):
        new_mesh=eqtree.get_mesh() # self.get_mesh()
        # This is only ever attached to InterfaceEquations, so the eqtree/interpolator here
        # always belong to a spatial interface mesh, never an ODEStorageMesh.
        assert isinstance(new_mesh,InterfaceMesh)
        if new_mesh._zeta_chart_overridden or new_mesh.get_full_name() in interpolator.zeta_overridden_boundaries:
            # Somebody else - a topological-change handler - has taken this boundary's chart over,
            # because it has to span an event that the geometry alone cannot describe (the old and
            # the new interface are not the same curve there). Whoever runs second wins, so the one
            # that cannot be reconstructed has to be the one left standing. Both tests, because this
            # hook can run before the handler's (which is what the interpolator registry is for) or
            # after it (the mesh flag, which also covers after_mapping_on_macro_elements).
            return super()._before_mesh_to_mesh_interpolation(eqtree,interpolator)
        old_base=interpolator.old
        assert not isinstance(old_base,ODEStorageMesh)
        old_mesh=old_base.get_mesh(new_mesh.get_name())
        assert isinstance(old_mesh,InterfaceMesh)
        self.assign_zetas(old_mesh)
        self.assign_zetas(new_mesh)
        return super()._before_mesh_to_mesh_interpolation(eqtree,interpolator)

    def after_remeshing(self, eqtree: "EquationTree"):
        # Deliberately NOT gated on _zeta_chart_overridden: the transfer is over by now, and this is
        # where the assigner takes its own chart back. Order-independent, since it does not matter
        # whether the handler has already cleared the flag.
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
        bmesh=mesh.get_bulk_mesh()
        if isinstance(bmesh,InterfaceMesh):
            raise RuntimeError("Cannot do it, if the parent mesh is not a bulk mesh")
        bind=bmesh.get_boundary_index(mesh.get_name())
        minzeta=1e40
        maxzeta=-minzeta
        nodes_set=0
        # Nothing to merge: zeta here IS a nodal coordinate, so a rank assigning it to its own nodes
        # produces exactly the values it would in a serial run, and the chart is global by
        # construction. Only the degeneracy test below is about the interface as a whole.
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
        if self._needs_merged_interface(mesh):
            # A rank whose share of the interface is a short stretch - or none at all - would
            # otherwise report the whole boundary as degenerate.
            minzeta,maxzeta=get_mpi_min(minzeta),get_mpi_max(maxzeta)
            nodes_set=get_mpi_sum(nodes_set)
        if maxzeta-minzeta<1e-10 and nodes_set>1:
            raise self._add_exception_info(RuntimeError("The assigned zeta coordinates are not meaningful. Probably align along another axis"))
        if self.validate_zetas:
            # An interface overhanging in the chosen direction produces a perfectly non-degenerate
            # zeta that is nonetheless not invertible, which the min/max check above cannot see.
            try:
                _check_zeta_is_invertible(mesh,bind,"AssignZetaCoordinatesByEulerianCoordinate(direction="+str(self.direction)+")")
            except RuntimeError as e:
                raise self._add_exception_info(e)


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
    def __init__(self,start_near_point:tuple[ExpressionOrNum, ExpressionOrNum] | None=None,sort_along_axis:SortAlongAxis | None=None,normalized:bool=True,segment_jump_offset:float=1.0,individual_segments:bool=True):
        super().__init__()
        self.start_near_point=start_near_point
        self.sort_along_axis:SortAlongAxis | None=sort_along_axis # spelled out: an inferred attribute type would widen the literals to str
        self.normalized=normalized

        self.segment_jump_offset=segment_jump_offset  # Add this offset to the arclength when a new segment is started
        self.individual_segments=individual_segments # process and potentially normalize each segment individually. The total zeta parametrization is then by concatenation

        check_sorting_arguments(sort_along_axis,start_near_point,require_one=True,whom="AssignZetaCoordinatesByArclength")

    def _closed_loop_zetas(self,mesh:InterfaceMesh,pts,loop:list[int])->tuple[float,list[tuple[int,float]]]:
        """Parameterise a closed loop by arclength, and declare the result periodic.

        A circle has no single-valued zeta - that is a property of the circle, not of the code - so
        instead of a chart this records a period. The element closing the loop then runs from the
        last node's zeta to the first node's zeta + period rather than backwards across the whole
        range, which is what makes it invertible like any other element.

        Two things have to agree between the old and the new mesh for the parameterisation to be
        useful at all, and both are fixed by the geometry rather than by the discretisation: the
        seam, which is where arclength is measured from (see :py:func:`_loop_seam_anchor`), and the
        orientation, which is taken counter-clockwise via the polygon's signed area.
        """
        loop=list(loop)
        n=len(loop)
        if n<3:
            raise self._add_exception_info(RuntimeError("A closed interface loop needs at least three distinct nodes, got "+str(n)))

        # Orientation, so both meshes traverse the loop the same way.
        area=0.0
        for k in range(n):
            a,b=loop[k],loop[(k+1)%n]
            area+=pts[0,a]*pts[1,b]-pts[0,b]*pts[1,a]
        if area<0:
            loop=[loop[0]]+list(reversed(loop[1:]))

        if self.sort_along_axis is not None:
            direction=({"x+":(1.0,0.0),"x-":(-1.0,0.0),"y+":(0.0,1.0),"y-":(0.0,-1.0)})[str(self.sort_along_axis)]
        else:
            assert self.start_near_point is not None # check_sorting_arguments(require_one=True) in __init__ has already insisted on one of the two
            cx=sum(pts[0,i] for i in loop)/n
            cy=sum(pts[1,i] for i in loop)/n
            dx,dy=float(self.start_near_point[0])-cx,float(self.start_near_point[1])-cy
            norm=numpy.sqrt(dx*dx+dy*dy)
            direction=(dx/norm,dy/norm) if norm>0 else (1.0,0.0)

        seam_k,seam_u=_loop_seam_anchor(loop,pts,direction)

        # Cumulative arclength around the loop, node k at cum[k], closing edge included.
        edge=[0.0]*n
        for k in range(n):
            a,b=loop[k],loop[(k+1)%n]
            edge[k]=float(numpy.sqrt((pts[0,b]-pts[0,a])**2+(pts[1,b]-pts[1,a])**2))
        # A single wildly oversized step means the loop is not geometrically ordered, and since zeta
        # IS the accumulated arclength the whole parameterisation would be silently wrong. Seen for
        # real: a remeshed disc whose closing element carries its mid-side node at the ANTIPODE of
        # where it belongs (still on the circle, so radius checks pass), which inflated the loop
        # length by 1.6x. Report it rather than parameterise a broken loop.
        srt=sorted(edge)
        median=srt[len(srt)//2]
        if median>0:
            worst=max(range(n),key=lambda k:edge[k])
            if edge[worst]>5.0*median:
                a,b=loop[worst],loop[(worst+1)%n]
                raise self._add_exception_info(RuntimeError("The closed loop '"+mesh.get_name()+"' is not geometrically ordered: the step from node at ("+str(pts[0,a])+", "+str(pts[1,a])+") to ("+str(pts[0,b])+", "+str(pts[1,b])+") is "+str(edge[worst])+", more than five times the median step "+str(median)+". A node of this boundary is misplaced, so its arclength - and therefore its zeta - would be meaningless."))

        cum=[0.0]*n
        for k in range(1,n):
            cum[k]=cum[k-1]+edge[k-1]
        total=cum[-1]+edge[-1]
        if total<=0:
            raise self._add_exception_info(RuntimeError("The closed interface loop '"+mesh.get_name()+"' has zero length"))

        s_anchor=cum[seam_k]+seam_u*edge[seam_k]
        period=1.0 if self.normalized else total

        return period,[(ptind,((cum[k]-s_anchor)%total)/(total if self.normalized else 1.0))
                       for k,ptind in enumerate(loop)]

    def _compute_zetas(self,mesh:InterfaceMesh,cache:MeshDataCacheEntry,whole_interface:bool)->tuple[float,list[tuple[int,float]]]:
        """The zeta of every point of ``cache``, and the period (0 for an open boundary).

        Touches no node, so that the same computation serves the local mesh and - on rank 0 - the
        globally merged one, where the point indices do not address this rank's nodes at all.
        ``whole_interface`` says the cache covers the entire boundary, which is what lets the node
        count be compared against it."""
        pts=cache.get_coordinates()
        segs,_=cache.get_interface_line_segments()

        closed=_find_closed_segments(cache,segs)
        if any(closed):
            if len(segs)!=1:
                raise self._add_exception_info(RuntimeError("The interface '"+mesh.get_name()+"' has "+str(len(segs))+" segments of which "+str(sum(closed))+" are closed loops. A periodic zeta has one period for the whole boundary, so a closed loop has to be the only segment on it. Split the boundary, or parameterise it some other way."))
            loop=_walk_closed_loop(cache)
            if loop is None:
                raise self._add_exception_info(RuntimeError("The interface '"+mesh.get_name()+"' looks like a closed loop but its elements do not form one single cycle, so it cannot be given a periodic zeta."))
            return self._closed_loop_zetas(mesh,pts,loop)

        # Sort and reverse the segments based on the settings
        segs=sort_line_segments(pts,segs,sort_along_axis=self.sort_along_axis,start_near_point=self.start_near_point,spatial_unit=mesh.get_code_gen().get_scaling("spatial"),whom="AssignZetaCoordinatesByArclength")

        if whole_interface:
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

        return 0.0,[(pti,float(al)) for al,pti in zip(alengths,ptinds)]

    def _assign_zetas_by_position(self,mesh:InterfaceMesh,bind:int,table:list[tuple[float,float,float]])->None:
        """The shared table assignment, with this equation's context attached to any failure."""
        try:
            assign_zetas_from_position_table(mesh,bind,table)
        except RuntimeError as e:
            raise self._add_exception_info(e)

    def assign_zetas(self,mesh:InterfaceMesh)->None:
        if mesh.get_dimension()!=1:
            raise RuntimeError("Currently only implemented for 1d interfaces meshes")
        bmesh=mesh.get_bulk_mesh()
        if isinstance(bmesh,InterfaceMesh):
            raise RuntimeError("Cannot do it, if the parent mesh is not a bulk mesh")
        bind=bmesh.get_boundary_index(mesh.get_name())

        cache:"MeshDataCacheEntry | None"
        if not self._needs_merged_interface(mesh):
            cache=MeshDataCacheEntry(mesh,MeshDataCacheKey(nondimensional=True,tesselate_tri=True))
            period,zetas=self._compute_zetas(mesh,cache,whole_interface=True)
            nodemap=mesh.fill_node_index_to_node_map()
            for ptind,z in zetas:
                nodemap[ptind].set_coordinates_on_boundary(bind,[z])
        else:
            # Arclength is a property of the whole curve, so it can only be measured on the whole
            # curve: merge the interface (collective, result on rank 0), parameterise it there, and
            # hand the (position, zeta) pairs to everybody. Broadcasting the result rather than the
            # merged entry keeps the payload to the boundary itself.
            problem=self.get_problem()
            comm=get_mpi_world_comm()
            assert comm is not None
            cache=problem.get_cached_mesh_data(mesh,nondimensional=True,global_mesh=True)
            payload=None
            error:BaseException | None=None
            if get_mpi_rank()==0:
                try:
                    assert cache is not None
                    period,zetas=self._compute_zetas(mesh,cache,whole_interface=False)
                    pts=cache.get_coordinates()
                    payload=(period,[(float(pts[0,ptind]),float(pts[1,ptind]),z) for ptind,z in zetas])
                except BaseException as e:
                    error=e
            payload=comm.bcast(payload,root=0)
            mpi_share_root_failure(error,context="parameterising the merged interface '"+mesh.get_name()+"' by arclength")
            assert payload is not None
            period,table=payload
            self._assign_zetas_by_position(mesh,bind,table)

        bmesh.boundary_coordinate_bool(bind)
        bmesh.set_boundary_zeta_period(bind,period)
        mesh.update_zeta_in_buffer()
        if self.validate_zetas:
            # Local, and correct to be: the test is that the elements THIS rank holds tile their own
            # stretch of zeta without overlapping, which is exactly as meaningful on a partition as on
            # the whole interface (disconnected stretches only ever leave gaps, never overlaps).
            try:
                _check_zeta_is_invertible(mesh,bind,"AssignZetaCoordinatesByArclength"+(" (closed loop)" if period>0 else ""),period)
            except RuntimeError as e:
                raise self._add_exception_info(e)




class DebugZetaCoordinate(InterfaceEquations):
    def get_zeta_name(self):
        master=self._master()
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


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
