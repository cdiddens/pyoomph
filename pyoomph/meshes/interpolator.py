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
 
from ..typings import *

import numpy

from ..generic.problem import *
from ..meshes.mesh import ODEStorageMesh
from .. import _pyoomph_core as _pyoomph


if TYPE_CHECKING:
    from .mesh import AnyMesh,AnySpatialMesh


class BaseMeshToMeshInterpolator:
    def __init__(self, old:"AnyMesh", new:"AnyMesh"):
        self.old = old
        self.new = new

    def interpolate(self)->None:
        raise NotImplementedError()

class ProjectionInternalInterpolator(BaseMeshToMeshInterpolator):
    """Transfer by L2 projection rather than by nodal interpolation.

    Instead of asking "what was the old solution at this new node", this solves
    ``integral (u_new - u_old) * psi = 0`` over the new mesh, i.e. the new field is the L2-best
    representation of the old one. That is conservative in a way pointwise interpolation is not,
    which is why it is worth having even though :py:class:`InternalInterpolator` is cheaper.

    The residual is **linear** in the unknowns, so each solve is one Newton step whose Jacobian is a
    mass matrix. That matrix does not depend on the field or on the history level, which is where the
    remaining performance work is (see dev_docs/mesh_point_locator.md phase 4b).
    """

    #: Meshes whose integration points have been mapped, filled by every instance's constructor
    #: before any of them runs :py:meth:`interpolate`. The projection is a single global solve over
    #: all of them - solving one mesh at a time would leave the others assembling their physical
    #: equations, so the "projection" would be fighting the physics everywhere else.
    _pending:list["AnySpatialMesh"]=[]

    def __init__(self, old: "AnySpatialMesh", new: "AnySpatialMesh"):
        super().__init__(old, new)
        self.old:AnySpatialMesh=old
        self.new:AnySpatialMesh=new
        # The source is the OLD mesh. This used to pass `new`, i.e. it located the new mesh's
        # integration points in itself, which is a no-op mapping.
        old.prepare_interpolation()
        new.prepare_zeta_interpolation(old)
        ProjectionInternalInterpolator._pending.append(new)

    def interpolate(self):
        cls=ProjectionInternalInterpolator
        meshes=[m for m in cls._pending if m._has_zeta_projection_prepared()]
        if not meshes:
            return   # another instance already ran the global solve
        cls._pending=[]

        problem=self.new.get_problem()
        if problem is None:
            raise RuntimeError("Cannot run a projection interpolation without a problem")

        n_history=next(iter(meshes[0].nodes())).ntstorage()

        # The projection system is a mass matrix: structurally unrelated to the physical problem, so
        # it must not inherit the physical problem's solver configuration or its cached sparsity.
        # See dev_docs/mesh_point_locator.md phase 4b for why each of these matters.
        old_frozen=problem.use_frozen_sparsity
        problem.use_frozen_sparsity=False
        old_solver=problem._lasolver
        proj_solver=getattr(problem,"_projection_lasolver",None)
        if proj_solver is not None:
            problem.set_linear_solver(proj_solver)
        try:
            for time_index in reversed(range(n_history)):
                for m in meshes:
                    m._set_time_level_for_projection(time_index)
                    m._set_zeta_projection_enabled(True)
                try:
                    problem.steady_newton_solve()
                finally:
                    for m in meshes:
                        m._set_zeta_projection_enabled(False)
                if time_index>0:
                    # The solve writes into the current values; move them to the history level they
                    # were projected from. On the NEW mesh - the old one is the source and must not
                    # be touched.
                    for m in meshes:
                        for n in m.nodes():
                            for ni in range(n.nvalue()):
                                n.set_value_at_t(time_index,ni,n.value(ni))
                            coord=n.variable_position_pt()
                            for ci in range(coord.nvalue()):
                                coord.set_value_at_t(time_index,ci,coord.value(ci))
        finally:
            problem.use_frozen_sparsity=old_frozen
            if proj_solver is not None and old_solver is not None:
                problem.set_linear_solver(old_solver)


class InternalInterpolator(BaseMeshToMeshInterpolator):
    def __init__(self, old:"AnySpatialMesh", new:"AnySpatialMesh"):
        super(InternalInterpolator, self).__init__(old, new)
        self.old:AnySpatialMesh=old
        self.new:AnySpatialMesh=new
        self.boundary_max_distances:dict[str,float]={}
        self.try_to_use_zeta_on_boundary:bool=True
        #: On a boundary with no zeta defined, match interface nodes by projecting them onto the old
        #: interface geometry instead of blending the two nearest old nodes. Set False to get the old
        #: behaviour back. See dev_docs/mesh_point_locator.md.
        self.project_on_boundary_without_zeta:bool=True
        old.prepare_interpolation()
        # Remove the macro elements, since they are really troublesome for the locate_zeta
        for e in old.elements():
            e.set_macro_element(None, False)
            while e.get_father_element() is not None:
                e = e.get_father_element()
                e.set_macro_element(None, False)

    def interpolate(self):
        self.new.nodal_interpolate_from(self.old,-1)
        for bn in self.new.get_boundary_names():
            
            bi_new = self.new.get_boundary_index(bn)
            bi_old = self.old.get_boundary_index(bn)
            intermesh_new=self.new.get_mesh(bn,return_None_if_not_found=True)
            intermesh_old = self.old.get_mesh(bn, return_None_if_not_found=True)
            if intermesh_old is None and intermesh_new is None:
                #print("No interface mesh for boundary",bn,"in old and new mesh, skipping interpolation")                                
                self.new.nodal_interpolate_from(self.old,bi_old)
                continue # Happens e.g. on corners to another domain
            assert intermesh_old is not None, "Old interface mesh "+bn+" of "+self.old.get_name()+" not found. Index: "+str(bi_old)
            assert intermesh_new is not None, "New interface mesh "+bn+" of "+self.new.get_name()+" not found. Index: "+str(bi_new)
            boundary_interpolation_max_dist=self.boundary_max_distances.get(bn,0.0)            
            #print("INTERPOLATE BOUNDARY ",bn)
            if self.try_to_use_zeta_on_boundary and self.old.is_boundary_coordinate_defined(bi_old):
                if not self.new.is_boundary_coordinate_defined(bi_new):
                    raise RuntimeError("Boundary coordinate along "+bn+" is defined on the old, but not the new mesh")
                intermesh_new.nodal_interpolate_from(intermesh_old,bi_new)
            elif self.project_on_boundary_without_zeta and _pyoomph.Mesh.get_use_point_locator():
                # No zeta along this boundary. Rather than the nearest-node blend below - which is
                # not an interpolation and is quadratic in the mesh size - match each new interface
                # node to the closest point of the OLD interface geometry and evaluate the old
                # element's shape functions there. That needs no chart, so unlike zeta it also works
                # for a 2d interface in 3d. Anything it cannot place falls through to the blend
                # inside nodal_interpolate_from, which now reports when it does.
                intermesh_new.nodal_interpolate_from(intermesh_old,bi_new,False)
            else:
                self.new.nodal_interpolate_along_boundary(self.old, bi_new, bi_old, intermesh_new,intermesh_old,boundary_interpolation_max_dist)
            
        # Now also go over all corners etc
        for iname,imsh in self.new._interfacemeshes.items(): 
            for bn in imsh.get_boundary_names():
                codim2mesh_new=imsh._interfacemeshes.get(bn) 
                bi_new = imsh.get_boundary_index(bn)
                if codim2mesh_new is not None and imsh.nboundary_element(bi_new):
                    if imsh.nboundary_element(bi_new)>0: # Has elements on that boundary
                        imsh_old = self.old.get_mesh(iname, return_None_if_not_found=True)
                        assert imsh_old is not None
                        codim2mesh_old = imsh_old.get_mesh(bn, return_None_if_not_found=True)
                        assert codim2mesh_old is not None
                        bi_old=imsh_old.get_boundary_index(bn)
#                        print("INTER",iname,bn,codim2mesh_new,codim2mesh_old,bi_new,bi_old)
                        #print(imsh.nboundary_node(bi_new))
                        #print(imsh.nboundary_element(bi_new))
                        boundary_interpolation_max_dist=max(self.boundary_max_distances.get(iname+'/'+bn,0.0),self.boundary_max_distances.get(bn+'/'+iname,0.0))
                        #print("BMAXDIST",boundary_interpolation_max_dist,self.boundary_max_distances)
                        imsh.nodal_interpolate_along_boundary(imsh_old, bi_new, bi_old, codim2mesh_new,codim2mesh_old, boundary_interpolation_max_dist)

                    if len(codim2mesh_new._interfacemeshes) > 0:
                        raise RuntimeError("Codim 3 interpolation")
            #print(iname,msh,msh.get_boundary_names())
        #print(dir(self.new))
        #exit()


class ODEInterpolator(BaseMeshToMeshInterpolator):
    def __init__(self,old:ODEStorageMesh,new:ODEStorageMesh):
        super(ODEInterpolator, self).__init__(old,new)
        self.new:ODEStorageMesh=new
        self.old:ODEStorageMesh=old

    def interpolate(self):
        newode = self.new.get_element()
        oldode = self.old.get_element()
        oldindices=oldode._ode_elem_to_numpy()[1]
        newindices = newode._ode_elem_to_numpy()[1]
        new_to_old_indices={newi:oldindices[k] for k,newi in newindices.items() if k in oldindices.keys()}

        for newi,oldi in new_to_old_indices.items():
            for nt in range(newode.internal_data_pt(newi).ntstorage()):
                newode.internal_data_pt(newi).set_value_at_t(nt,0,oldode.internal_data_pt(oldi).value_at_t(nt,0))


_DefaultInterpolatorClass = InternalInterpolator

if False:
    from sklearn.neighbors import NearestNeighbors


    class KNNInterpolator(BaseMeshToMeshInterpolator):
        def __init__(self, old, new, nneigh=20):
            super(KNNInterpolator, self).__init__(old, new)

            self.boundaries_separate = True

            old.prepare_interpolation()
            # Remove the macro elements, since they are really troublesome for the locate_zeta
            for e in old.elements():
                e.set_macro_element(None, False)
                while e.get_father_element() is not None:
                    e = e.get_father_element()
                    e.set_macro_element(None, False)

            self.nneigh = nneigh
            self.node_to_elem = []
            pointlist = []
            for e in old.elements():
                for i in range(e.nnode()):
                    self.node_to_elem.append(e)
                    n = e.node_pt(i)
                    p = [n.x(j) for j in range(n.ndim())]
                    pointlist.append(p)
            pointlist = numpy.array(pointlist)
            self.KNN = NearestNeighbors(n_neighbors=self.nneigh, )
            self.KNN.fit(pointlist)

        def interpolate_bulk(self, also_on_bounds=False):
            xprobe = []
            destnodes = []
            for n in self.new.nodes():
                if also_on_bounds or (not n.is_on_boundary()):
                    xprobe.append([n.x(j) for j in range(n.ndim())])
                    destnodes.append(n)
            xprobe = numpy.array(xprobe)
            inds = self.KNN.kneighbors(xprobe, return_distance=False)
            for j in range(len(xprobe)):
                for i in inds[j]:
                    el = self.node_to_elem[i]
                    s = numpy.zeros((el.dim()))
                    x = xprobe[j]
                    s = el.locate_zeta(x, s, False)
                    if len(s) > 0:
                        nodalvals = el.get_interpolated_nodal_values_at_s(0, s)
                        for k, v in enumerate(nodalvals):
                            destnodes[j].set_value(k, v)
                        break
                else:
                    dists, inds = self.KNN.kneighbors([xprobe[j]], return_distance=True)
                    print("FAILED PROBE RESULT (index,dist,ind,s)")
                    locs = []
                    x = xprobe[j]
                    for i in inds[0]:
                        s = numpy.zeros((el.dim()))
                        s = el.locate_zeta(x, s, False)
                        locs.append(s)
                    for i, ind in enumerate(inds[0]):
                        print("", i, dists[0][i], ind, locs[i])
                    raise RuntimeError("Cannot locate the point at " + str(x))

        def interpolate_boundary(self):
            for bn in self.new.get_boundary_names():
                bi_new = self.new.get_boundary_index(bn)
                bi_old = self.old.get_boundary_index(bn)
                intermesh_new = self.new.get_mesh(bn, return_None_if_not_found=True)
                intermesh_old = self.old.get_mesh(bn, return_None_if_not_found=True)
                self.new.nodal_interpolate_along_boundary(self.old, bi_new, bi_old, intermesh_new, intermesh_old, 0.0)

        def interpolate(self):
            self.interpolate_bulk(also_on_bounds=not self.boundaries_separate)
            if self.boundaries_separate:
                self.interpolate_boundary()

    _DefaultInterpolatorClass = KNNInterpolator
else:
    pass
