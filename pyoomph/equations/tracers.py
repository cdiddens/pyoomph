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
 
 
import _pyoomph

from ..expressions import eval_flag,evaluate_in_past,partial_t,scale_factor,mesh_velocity
from ..expressions.generic import Expression,ExpressionOrNum
from ..generic.codegen import Equations,var,InterfaceEquations

from ..meshes.mesh import assert_spatial_mesh

from ..typings import *

import numpy

if TYPE_CHECKING:
    from ..meshes.mesh import AnyMesh,AnySpatialMesh
    from ..generic.codegen import EquationTree

class TracerParticles(Equations):
    """
        This class defines the equations for tracer particles that are advected by a velocity field. The goal is to track the movement of particles in a fluid flow.
        These particles do not affect the flow field and are only used for visualization purposes.

        Args:
            distance (ExpressionOrNum): The distance between the tracer particles. If None, the particles are placed on a grid with this spacing.
            advection (Expression): The advection velocity field that advects the tracer particles. Default is var("velocity").
            tracer_name (str): The name of the tracer particles. Default is "tracers".
    """
    def __init__(self,distance:ExpressionOrNum,advection:Expression=var("velocity"),tracer_name:str="tracers"):
        super(TracerParticles, self).__init__()
        self.advection_expression=advection
        self.tracer_name=tracer_name
        self._mesh:Optional["AnySpatialMesh"]=None
        self.distance=distance


    def _update_mesh(self,mesh:"AnySpatialMesh"):
        if mesh!=self._mesh:
            if self.tracer_name in mesh._tracers.keys(): #type:ignore
                raise RuntimeError("Tracers with the name " + str(self.tracer_name) + " already added by other TracerParticles")
        self._mesh = mesh
        self._mesh._tracers[self.tracer_name] = _pyoomph.TracerCollection(self.tracer_name) #type:ignore
        self._mesh._tracers[self.tracer_name]._set_mesh(self._mesh) #type:ignore

    def get_collection(self) -> Optional[_pyoomph.TracerCollection]:
        if (self._mesh is None) or (self.tracer_name not in self._mesh._tracers.keys()): #type:ignore
            return None
        return self._mesh._tracers[self.tracer_name] #type:ignore

    def before_assigning_equations_preorder(self, mesh:"AnyMesh"):
        if self._mesh is None:
            self._update_mesh(assert_spatial_mesh(mesh))

    def _init_output(self,eqtree:"EquationTree",continue_info:Optional[Dict[str,Any]],rank:int):
        mesh=assert_spatial_mesh(eqtree._mesh)        
        self._update_mesh(mesh)
        assert self._mesh is not None
        # Create a grid of tracers in the distance
        coll = self.get_collection()
        assert coll is not None
        if self.distance is not None:
            dnd=self.distance/self._mesh.get_problem().get_scaling("spatial")
            dnd=float(dnd)
            d=self._mesh.get_dimension()
            mins=[1e60]*d
            maxs = [-1e60] * d
            for n in self._mesh.nodes():
                for i in range(d):
                    x=n.x(i)
                    mins[i]=min(mins[i],x)
                    maxs[i] = max(maxs[i], x)
            # move d/2 from the boundaries
            for i in range(d):
                mins[i]+=dnd/2
                maxs[i] -= dnd / 2
            npts=[max(1,round((maxs[i]-mins[i])/dnd)+1) for i in range(d)]
            if d==2:
                xs =numpy.linspace(mins[0],maxs[0],npts[0],endpoint=True) #type:ignore
                ys = numpy.linspace(mins[1], maxs[1], npts[1], endpoint=True) #type:ignore
                for y in ys: #type:ignore
                    for x in xs: #type:ignore
                        coll.add_tracer([x,y])
            else:
                raise RuntimeError("Implement here tracer grid generation for other dimensions")
        coll._locate_elements()


    def before_newton_solve(self):
        coll=self.get_collection()
        assert coll is not None
        coll._prepare_advection()

    def after_newton_solve(self):
        assert self._mesh is not None
        print("Advecting tracers '"+str(self.tracer_name)+"' on domain '"+self._mesh.get_full_name()+"'")
        coll=self.get_collection()
        assert coll is not None        
        coll._advect_all()

    def after_remeshing(self,eqtree:"EquationTree"):
        coll=self.get_collection()
        assert coll is not None           
        coll._locate_elements()

    def define_additional_functions(self):
        master = self._get_combined_element()
        cg=master._assert_codegen()
        adv=self.advection_expression

        TF=eval_flag("timefrac_tracer")
        adv_blend=adv*TF+evaluate_in_past(adv)*(1-TF)        
        # TODO: ALE term in past?
        ALE_term = eval_flag("moving_mesh") * mesh_velocity()
        ALE_corrected=scale_factor("temporal")/scale_factor("spatial")*(adv_blend-ALE_term)
        cg._register_tracer_advection(self.tracer_name, ALE_corrected)

    #def before_finalization(self,codegen):
        #self._update_mesh(codegen._mesh)


class TracerTransferAtInterface(InterfaceEquations):
    required_parent_type = TracerParticles
    required_opposite_parent_type = TracerParticles

    def __init__(self,vice_versa:bool=True):
        super(TracerTransferAtInterface, self).__init__()
        self.vice_versa=vice_versa

    def before_assigning_equations_preorder(self, mesh:"AnyMesh"):
        mytr=self.get_parent_equations()
        othertr=self.get_opposite_parent_equations()
        assert isinstance(mytr,TracerParticles) and isinstance(othertr,TracerParticles)
        pmesh=mytr._mesh 
        mycol=mytr.get_collection()
        othcol=othertr.get_collection()
        if pmesh and mycol and othcol:
            bind=pmesh.get_boundary_index(self.get_mesh().get_name())
            mycol._set_transfer_interface(bind,othcol)
            if self.vice_versa:
                opmesh=othertr._mesh 
                if opmesh:
                    obind=opmesh.get_boundary_index(self.get_mesh().get_name())
                    othcol._set_transfer_interface(obind,mycol)

