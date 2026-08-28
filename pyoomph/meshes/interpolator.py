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
from ..meshes.mesh import ODEStorageMesh, InterfaceMesh
from .. import _pyoomph_core as _pyoomph


if TYPE_CHECKING:
    from .mesh import AnyMesh,AnySpatialMesh


class BaseMeshToMeshInterpolator:
    #: What this interpolator still gets wrong on a distributed (``--distribute``) problem, quoted by
    #: the refusal in :py:meth:`~pyoomph.generic.problem.Problem.force_remesh`, or ``None`` if it
    #: works there. The default says no, since transferring between two meshes that are partitioned
    #: differently is the hard part and every interpolator has to solve it deliberately.
    #: See dev_docs/distributed_remeshing.md.
    distributed_limitation:str | None="it transfers only what this rank's part of the old mesh covers"

    def __init__(self, old:"AnyMesh", new:"AnyMesh"):
        self.old = old
        self.new = new
        #: Every interpolator of the transfer this one belongs to, keyed by domain name, filled by
        #: :py:meth:`~pyoomph.generic.problem.Problem.force_remesh` before any
        #: ``_before_mesh_to_mesh_interpolation`` hook runs. One interpolator is created per domain
        #: and each hook is dispatched with its own, so an equation living on a boundary that two
        #: domains SHARE - a free surface between two phases - can only configure the far side's
        #: transfer through this.
        self.remesh_group:dict[str,"BaseMeshToMeshInterpolator"]={}
        #: Full names ("liquid/interface") of interface meshes whose zeta chart is written by
        #: somebody else. Read by :py:class:`~pyoomph.meshes.zeta.AssignZetaCoordinatesBase`, which
        #: then leaves them alone instead of overwriting a chart it could not have produced; the
        #: interpolation itself does not consult it.
        self.zeta_overridden_boundaries:set[str]=set()
        #: Boundary names whose nodes are transferred by locating them in the old BULK mesh instead
        #: of through the old interface mesh. See :py:meth:`InternalInterpolator.interpolate`.
        self.bulk_locate_boundaries:set[str]=set()
        #: Boundary names whose zeta chart governs the INTERFACE-ONLY dofs alone, everything else
        #: being matched geometrically. For a chart that deliberately does not follow the geometry;
        #: see :py:meth:`InternalInterpolator.interpolate`.
        self.zeta_for_interface_fields_only:set[str]=set()
        #: How far a codimension-2 match may reach, per ``"interface/subboundary"`` key, in the
        #: NONDIMENSIONAL coordinates the nodes are stored in. Anything further away is left alone
        #: rather than snapped to a corner that a topological change has moved somewhere else.
        self.boundary_max_distances:dict[str,float]={}

    def interpolate(self)->None:
        raise NotImplementedError()


def _transfer_internal_facet_fields(old:"AnySpatialMesh", new:"AnySpatialMesh")->None:
    """Carry the fields living on the interior-facet skeleton of ``old`` over to ``new``.

    The skeleton (``"_internal_facets_"``) is not a named boundary, so the boundary loops of the
    interpolators do not visit it, and its fields are discontinuous ones held in the facet elements'
    own internal data rather than on nodes - the nodal transfer could not carry them either way. On
    the C++ side each new facet reads the old skeleton at its own sample points and least-squares
    fits the result, deciding which old facet may answer by locating the new facet in the OLD BULK
    mesh; see ``InterfaceMesh::interpolate_discontinuous_data_from``.
    """
    new_skel = new.get_mesh("_internal_facets_", return_None_if_not_found=True)
    if new_skel is None:
        return
    assert isinstance(new_skel, InterfaceMesh)
    old_skel = old.get_mesh("_internal_facets_", return_None_if_not_found=True)
    if old_skel is None:
        # The skeleton is new (i.e. the equations changed along with the mesh, redefine_problem).
        # There is nothing to transfer; the new facets keep whatever their recovery expressions - or,
        # failing those, the allocated zero - gave them.
        return
    new_skel.interpolate_discontinuous_data_from(old_skel)


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

    #: The instances whose integration points have been mapped, filled by every constructor before
    #: any of them runs :py:meth:`interpolate`. The projection is a single global solve over all of
    #: them - solving one mesh at a time would leave the others assembling their physical equations,
    #: so the "projection" would be fighting the physics everywhere else. The instances rather than
    #: their mesh pairs, so that the seeding nodal transfer below can be configured per domain.
    _pending:list["ProjectionInternalInterpolator"]=[]

    #: Residual below which a history level counts as projected.
    projection_tolerance:float=1e-11
    #: Chord iterations allowed per level before giving up. One suffices whenever the position dofs
    #: are pinned, since the system is then exactly linear.
    max_chord_iterations:int=12

    def __init__(self, old: "AnySpatialMesh", new: "AnySpatialMesh"):
        super().__init__(old, new)
        self.old:AnySpatialMesh=old
        self.new:AnySpatialMesh=new
        # The source is the OLD mesh. This used to pass `new`, i.e. it located the new mesh's
        # integration points in itself, which is a no-op mapping.
        old.prepare_interpolation()
        new.prepare_zeta_interpolation(old)
        ProjectionInternalInterpolator._pending.append(self)

    def interpolate(self):
        cls=ProjectionInternalInterpolator
        instances=[i for i in cls._pending if i.new._has_zeta_projection_prepared()]
        pairs=[(i.old,i.new) for i in instances]
        meshes=[n for _,n in pairs]
        if not meshes:
            return   # another instance already ran the global solve
        cls._pending=[]

        # During remeshing neither mesh is guaranteed to have its problem pointer set - get_problem()
        # raises rather than returning None - so try the equation trees as well before giving up.
        problem=None
        for getter in (lambda: self.old.get_problem(), lambda: self.new.get_problem(),
                       lambda: self.old.get_eqtree().get_problem(),
                       lambda: self.new.get_eqtree().get_problem()):
            try:
                problem=getter()
            except Exception:
                problem=None
            if problem is not None:
                break
        if problem is None:
            raise RuntimeError("Cannot run a projection interpolation without a problem")

        n_history=next(iter(meshes[0].nodes())).ntstorage()

        # The projection system is a mass matrix: structurally unrelated to the physical problem, so
        # it must not inherit the physical problem's solver configuration or its cached sparsity.
        # See dev_docs/mesh_point_locator.md phase 4b for why each of these matters.
        # The C++ solver callback holds a weakref to "the current problem", and during remeshing it
        # is not pointing at ours - the projection solve would fail with "The problem has not been
        # set yet" before reaching the linear algebra at all.
        problem._activate_solver_callback()

        old_frozen=problem.use_frozen_sparsity
        problem.use_frozen_sparsity=False
        old_solver=problem._lasolver
        proj_solver=getattr(problem,"_projection_lasolver",None)
        if proj_solver is not None:
            problem.set_linear_solver(proj_solver)
        # The current positions are the ones the mesh generator produced for the new mesh and are
        # the answer already - nothing may move them. But a Newton solve can only solve for CURRENT
        # dofs, so projecting a history POSITION has to go through them: solve, copy the result into
        # the history level, and put the generator's positions back. History positions matter because
        # the mesh velocity on the new mesh is computed from them.
        generator_positions=[[[n.x(i) for i in range(n.ndim())] for n in m.nodes()] for m in meshes]

        def restore_positions():
            for m,saved in zip(meshes,generator_positions):
                for n,xs in zip(m.nodes(),saved):
                    for i,v in enumerate(xs):
                        n.set_x(i,v)

        # One factorisation for the whole transfer.
        #
        # The projection matrix is a mass matrix: it depends on neither the field nor the history
        # level, so factorising it once and reusing it for every level replaces N_history full
        # solves with one factorisation and N cheap back-substitutions. On a small mesh that is
        # invisible; a direct factorisation of a 200k system repeated per history level per remesh
        # is not.
        #
        # It is not quite a linear system, though, and that is why this iterates rather than taking
        # a single step: for t > 0 the position dofs are unknowns, and the integration weights
        # W = J_eulerian(s) * w depend on the geometry those dofs describe. Reusing the factorisation
        # while the residual is re-evaluated is a chord iteration - it converges quickly because the
        # geometry barely moves, and it never needs a second factorisation.
        # Start from the nodal transfer rather than from the raw generator state.
        #
        # This is not just a speed-up. With a moving mesh the position dofs are unknowns and the
        # Jacobian assembled here omits the dependence of the integration weights on them, so the
        # iteration is a fixed point rather than a Newton method: started 3% away from the answer it
        # diverged outright (residual 1.7e-3, 7.1e-4, 4.7e-2, 1.6e+47, NaN, as elements inverted).
        # The nodal interpolator already puts fields, positions and history within interpolation
        # error of the answer, so the projection begins essentially converged and only has to move
        # the fields onto their L2-best values.
        for inst in instances:
            seed=InternalInterpolator(inst.old,inst.new)
            # Configured the way its own domain's projection interpolator was: whatever told that one
            # that a boundary needs the bulk-locate path, or how far a boundary match may reach, was
            # talking about the nodal transfer, which is exactly what runs here.
            seed.bulk_locate_boundaries=set(inst.bulk_locate_boundaries)
            seed.zeta_for_interface_fields_only=set(inst.zeta_for_interface_fields_only)
            seed.boundary_max_distances=dict(inst.boundary_max_distances)
            seed.interpolate()

        import scipy.sparse.linalg
        factorisation=None

        def solve_current_level():
            nonlocal factorisation
            previous=None
            for _ in range(self.max_chord_iterations):
                # Residual only unless the matrix is actually about to be used. Every iteration used
                # to assemble the full Jacobian just to read the residual out of it, and the
                # convergence check that ends each level is the majority of those: measured on a
                # 12.6k-node, 4-field remesh, assembly was 53% of the whole remesh against 14% for
                # the single factorisation.
                J=None
                if factorisation is None:
                    res,J=problem.assemble_jacobian(with_residual=True)
                else:
                    res=numpy.array(problem.get_residuals())
                norm=float(numpy.max(numpy.absolute(res)))
                if norm<self.projection_tolerance:
                    return
                # Reuse the factorisation while it is still doing its job, and rebuild it when it is
                # not. With the positions pinned the system is exactly linear and one solve finishes
                # it, so the factorisation is built once for the whole transfer. With a moving mesh
                # the matrix genuinely changes - the integration weights depend on the geometry the
                # position dofs describe - and a chord iteration alone stalls, so it refactorises
                # rather than grinding to the iteration limit.
                if factorisation is None or (previous is not None and norm>0.5*previous):
                    if J is None:
                        res,J=problem.assemble_jacobian(with_residual=True)
                    factorisation=scipy.sparse.linalg.splu(J.tocsc())
                previous=norm
                dofs=numpy.array(problem.get_current_dofs()[0])
                problem.set_current_dofs(list(dofs-factorisation.solve(numpy.array(res))))
            raise RuntimeError("The projection did not converge to "+str(self.projection_tolerance)+" in "+str(self.max_chord_iterations)+" iterations")

        try:
            for time_index in reversed(range(n_history)):
                for m in meshes:
                    m._set_time_level_for_projection(time_index)
                    m._set_zeta_projection_enabled(True)
                try:
                    solve_current_level()
                finally:
                    for m in meshes:
                        m._set_zeta_projection_enabled(False)
                if time_index>0:
                    # The solve writes into the current values; move them to the history level they
                    # were projected from. On the NEW mesh - the old one is the source and must not
                    # be touched.
                    # Field values only. The POSITIONS at this history level are already correct -
                    # the nodal transfer that seeded this put them there - and they are frozen during
                    # the solve, so copying the current ones over would overwrite the history with
                    # the present geometry and flatten the mesh velocity to zero.
                    for m in meshes:
                        for n in m.nodes():
                            for ni in range(n.nvalue()):
                                n.set_value_at_t(time_index,ni,n.value(ni))
        finally:
            restore_positions()
            problem.use_frozen_sparsity=old_frozen
            if proj_solver is not None and old_solver is not None:
                problem.set_linear_solver(old_solver)

        # Again, after the solve. The seeding InternalInterpolator above already transferred the
        # interior-facet skeleton, but the projection solve is a solve of the whole problem: only the
        # meshes put into projection mode assemble the projection residual, every interface mesh
        # assembles its PHYSICAL one, so a facet unknown gets dragged wherever its own equations want
        # it. The transfer is idempotent and reads the old mesh, which is still alive here, so simply
        # redoing it is the cheapest way to end up with the transferred values either way.
        for old_mesh,new_mesh in pairs:
            _transfer_internal_facet_fields(old_mesh,new_mesh)


class InternalInterpolator(BaseMeshToMeshInterpolator):
    # Works distributed since stage 3 of dev_docs/distributed_remeshing.md: nodal_interpolate_from
    # pools across the ranks what each of them could place. Its nodal_interpolate_along_boundary
    # branches do not, which is why force_remesh() refuses a problem with codim-2 interfaces.
    distributed_limitation=None

    def __init__(self, old:"AnySpatialMesh", new:"AnySpatialMesh"):
        super(InternalInterpolator, self).__init__(old, new)
        self.old:AnySpatialMesh=old
        self.new:AnySpatialMesh=new
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
            if bn in self.bulk_locate_boundaries:
                # Ask the old BULK mesh, not the old boundary. After a topological change the fresh
                # nodes on this boundary - the symmetry axis inside a gap a pinch-off just opened,
                # say - lie in the old bulk of the correct phase, while the old boundary of the same
                # name either did not reach there at all or ran through material that now belongs to
                # the other phase; the interface-pair passes below would pair them with that.
                # Deliberately before get_boundary_index() on the OLD mesh, which throws for a
                # boundary that only the new mesh has; node_is_in_scope() only ever uses bi_new.
                self.new.nodal_interpolate_from(self.old,bi_new)
                continue
            bi_old = self.old.get_boundary_index(bn)
            intermesh_new=self.new.get_mesh(bn,return_None_if_not_found=True)
            intermesh_old = self.old.get_mesh(bn, return_None_if_not_found=True)
            if intermesh_old is None and intermesh_new is None:
                #print("No interface mesh for boundary",bn,"in old and new mesh, skipping interpolation")
                # bi_NEW: the index selects which of THIS (i.e. the new) mesh's nodes the pass is
                # responsible for (Mesh::node_is_in_scope), so it is evaluated on the destination.
                # Boundary indices are assigned per mesh in the order the nodes are visited, so the
                # old and the new mesh need not agree on them; passing bi_old then left the nodes of
                # this boundary without any pass of their own - they kept whatever they were built
                # with, since the bulk pass skips every boundary node.
                self.new.nodal_interpolate_from(self.old,bi_new)
                continue # Happens e.g. on corners to another domain
            assert intermesh_old is not None, "Old interface mesh "+bn+" of "+self.old.get_name()+" not found. Index: "+str(bi_old)
            assert intermesh_new is not None, "New interface mesh "+bn+" of "+self.new.get_name()+" not found. Index: "+str(bi_new)
            boundary_interpolation_max_dist=self.boundary_max_distances.get(bn,0.0)            
            #print("INTERPOLATE BOUNDARY ",bn)
            if self.try_to_use_zeta_on_boundary and self.old.is_boundary_coordinate_defined(bi_old):
                if not self.new.is_boundary_coordinate_defined(bi_new):
                    raise RuntimeError("Boundary coordinate along "+bn+" is defined on the old, but not the new mesh")
                if bn in self.zeta_for_interface_fields_only:
                    # Two passes, geometry first and the chart on top of it. zeta answers "which
                    # point of the old interface does this node CORRESPOND to", which is the same
                    # question as "where in the old mesh is it" only as long as the chart follows the
                    # geometry. Across a topological change it deliberately does not - the fresh cap
                    # of a pinch-off is charted onto the old interface near the waist, which is where
                    # its surface fields come from but not where a bulk field sampled at the cap
                    # does, nor where its position history is. So the ordinary projection pass keeps
                    # governing those and the chart governs the interface-only dofs alone.
                    intermesh_new.nodal_interpolate_from(intermesh_old,bi_new,False)
                    intermesh_new.nodal_interpolate_from(intermesh_old,bi_new,True,True)
                else:
                    intermesh_new.nodal_interpolate_from(intermesh_old,bi_new)
            elif self.project_on_boundary_without_zeta:
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
                        # Interface fields only: the per-boundary passes above have already put
                        # interpolated bulk values on these very nodes, and this call matches by
                        # nearest node instead - it would replace them by a two-node blend, which on
                        # a corner that moved is simply the old corner's value. All this pass is here
                        # for is the dofs the codim-2 mesh adds on top of them.
                        imsh.nodal_interpolate_along_boundary(imsh_old, bi_new, bi_old, codim2mesh_new,codim2mesh_old, boundary_interpolation_max_dist, only_interface_fields=True)

                    if len(codim2mesh_new._interfacemeshes) > 0:
                        raise RuntimeError("Codim 3 interpolation")
            #print(iname,msh,msh.get_boundary_names())
        #print(dir(self.new))
        #exit()

        # Last, because it reads the bulk: the recovery expressions used for facets that get nothing
        # from the old skeleton evaluate the (by now transferred) bulk solution.
        _transfer_internal_facet_fields(self.old,self.new)


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


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
