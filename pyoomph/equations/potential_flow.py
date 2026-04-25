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
 
 
from .. import *
from ..expressions import *
from ..meshes.mesh import AnyMesh
from ..typings import *

class PotentialFlow(Equations):
    def __init__(self,potential_name:str="phi",space:FiniteElementSpaceEnum="C2",scale=scale_factor("velocity")*scale_factor("spatial"),velo_projection:Union[bool,FiniteElementSpaceEnum]=True,velocity_name:str="velocity",mass_density:ExpressionNumOrNone=None,dynamic_viscosity:ExpressionNumOrNone=None,pressure_projection:Union[bool,FiniteElementSpaceEnum]=True,pressure_name="pressure",bulk_force_potential:ExpressionOrNum=0):
        super().__init__()
        self.potential_name=potential_name
        self.velocity_name=velocity_name
        self.pressure_name=pressure_name
        self.space=space
        self.scale=scale
        self.rho=mass_density

        # If viscosity is set, we calculate the velocity at free interfaces, even if we do not project the velocity 
        # we need it, since we cannot calculate the gradient of u, which is second order derivative of phi
        self.dynamic_viscosity=dynamic_viscosity
        self.velo_at_free_interface_space="D2" # D1 should be sufficient. But must be DG
        self.velo_at_free_interface_name="dgvelo"
        
        self.velo_projection=velo_projection # False: No output/calculate, True: LocalExpression, FiniteElementSpace: Projection
        self.pressure_projection=pressure_projection # same as above, but only calculated if mass_density is set (can't be done without rho)
        # Only used to get the pressure. We set it to a potential V that fulfills that the bulk force is f=-grad(V), requires of course that rot(f)=0
        self.bulk_force_potential=bulk_force_potential


        

    def define_fields(self):
        self.define_scalar_field(self.potential_name,self.space,scale=self.scale,testscale=scale_factor("spatial")**2/self.scale)
        if not isinstance(self.velo_projection,bool):
            self.define_vector_field(self.velocity_name,self.velo_projection,testscale=1/scale_factor(self.velocity_name))
        elif self.dynamic_viscosity is not None:
            self.define_vector_field(self.velo_at_free_interface_name,self.velo_at_free_interface_space,scale=scale_factor("velocity"),testscale=1/scale_factor("velocity"))

        if not isinstance(self.pressure_projection,bool) and self.rho is not None:
            self.define_scalar_field(self.pressure_name,self.pressure_projection,testscale=1/scale_factor(self.pressure_name))

    def define_residuals(self):
        phi,phitest=var_and_test(self.potential_name)
        self.add_weak(grad(phi),grad(phitest))
        if not isinstance(self.velo_projection,bool):
            u,utest=var_and_test(self.velocity_name)
            self.add_weak(u-grad(phi),utest,coordinate_system=cartesian)
        elif self.velo_projection:            
            self.add_local_function(self.velocity_name,grad(phi))
        if self.rho is not None and self.pressure_projection:
            pdef=-self.rho*(partial_t(phi)+dot(grad(phi),grad(phi)))+self.bulk_force_potential
            if not isinstance(self.pressure_projection,bool):
                p,ptest=var_and_test(self.pressure_name)
                self.add_weak(p-pdef,ptest)
            elif self.pressure_projection:            
                self.add_local_function(self.pressure_name,pdef)

        if self.dynamic_viscosity is not None and isinstance(self.velo_projection,bool):
            # We need the gradient of phi for u as dof anyways to add viscosities at free surfaces
            ui,uitest=var_and_test(self.velo_at_free_interface_name)
            self.add_weak(ui-grad(phi),uitest,coordinate_system=cartesian)

    def get_interface_velocity_for_viscous_at_interfaces(self,domain=None):
        if self.dynamic_viscosity is not None:
            if isinstance(self.velo_projection,bool):
                return var(self.velo_at_free_interface_name,domain=domain)
            else: 
                return var(self.velocity_name,domain=domain)
        


    def before_assigning_equations_postorder(self, mesh: AnyMesh):
        if self.dynamic_viscosity is None or not isinstance(self.velo_projection,bool):
            return super().before_assigning_equations_postorder(mesh)
        # Pin all degrees in the bulk that are not part of an element connected to one of the desired interfaces
        dirs=["x","y","z"]
        vcomps=[self.velo_at_free_interface_name+"_"+dirs[i] for i in range(self.get_nodal_dimension())]
        # Pin all
        for e in mesh.elements():
            for vcompo in vcomps:
                for ni,ind in e.get_field_data_list(vcompo,False):
                    ni.pin(ind)
                    ni.set_value(ind,0)
        # Unpin the elements that are attached to at least one interface elements
        for iname in mesh._interfacemeshes.keys():
            imesh=mesh.get_mesh(iname)
            free_inters=imesh.get_eqtree().get_equations().get_equation_of_type(_PotentialFlowFreeInterfaceBase,always_as_list=True)
            if len(free_inters)>0:
                for iel in imesh.elements():
                    e=iel.get_bulk_element()
                    for vcompo in vcomps:
                        for ni,ind in e.get_field_data_list(vcompo,False):
                            ni.unpin(ind)
                    
        return super().before_assigning_equations_postorder(mesh)


class _PotentialFlowInterfaceEquations(InterfaceEquations):
    required_parent_type=PotentialFlow
    def get_potential_flow(self)->PotentialFlow:
        potflow=self.get_parent_equations(of_type=PotentialFlow)
        assert isinstance(potflow,PotentialFlow)
        return potflow
    
    def get_phi_and_test(self,bulk:bool=False):
        potflow=self.get_potential_flow()
        phi,phi_test=var_and_test(potflow.potential_name,domain=".." if bulk else None)
        return phi,phi_test

class PotentialFlowNormalVelocity(_PotentialFlowInterfaceEquations):
    def __init__(self,unorm:ExpressionOrNum=0):
        super().__init__()
        self.unorm=unorm

    def define_residuals(self):        
        _phi,phi_test=self.get_phi_and_test()
        self.add_weak(-self.unorm,phi_test)

class PotentialFlowFarField(_PotentialFlowInterfaceEquations):
    def __init__(self,phi:ExpressionNumOrNone=0,origin:ExpressionOrNum=vector(0)):
        super().__init__()
        self.far_phi_value=phi        
        self.origin=origin

    def define_residuals(self):        
        phi,phi_test=self.get_phi_and_test()
        n=var("normal")
        d=var("coordinate")-self.origin                
        self.add_residual(weak( (phi - self.far_phi_value) * dot(n,d)/dot(d,d),phi_test))
        
        

class _PotentialFlowFreeInterfaceBase(_PotentialFlowInterfaceEquations):
    def __init__(self,*,additional_pressure:ExpressionOrNum=0,surface_tension:ExpressionNumOrNone=None,curvature_sign:int=-1,total_mass_transfer_rate:Optional[ExpressionOrNum]=None):
        super().__init__()
        self.additional_pressure=additional_pressure
        self.sigma=surface_tension
        self.curvature_sign=curvature_sign
        self.total_mass_transfer_rate=total_mass_transfer_rate

    def get_potential_flow_on_opposite_side(self):
        if self.get_opposite_parent_domain(raise_error_if_none=False) is not None:
            oppot=self.get_opposite_parent_equations(PotentialFlow).get_equation_of_type(PotentialFlow,always_as_list=True)
            if len(oppot)>0:
                if len(oppot)>1:
                    raise RuntimeError("Multiple PotentialFlow equations found in the opposite domain")
                return oppot[0]
        else:
            return None

    def define_fields(self):
        potflow=self.get_potential_flow()
        if self.sigma is not None:
            self.define_vector_field("_proj_normal",potflow.space)
            curvspace="C2"
            self.define_scalar_field("_curvature",curvspace,scale=1/scale_factor("spatial"),testscale=scale_factor("spatial"))        
        if self.get_potential_flow_on_opposite_side():
            self.define_scalar_field("_lagr_un_connect",potflow.space,testscale=1/scale_factor("velocity"),scale=1/scale_factor("temporal"))

    def define_residuals(self):
        n=var("normal")
        if self.sigma is not None:            
            pn,pn_test=var_and_test("_proj_normal")
            curv,curv_test=var_and_test("_curvature")
            self.add_weak(pn-n,pn_test)
            self.add_weak(curv-self.curvature_sign*div(pn),curv_test)
        opposite=self.get_potential_flow_on_opposite_side()
        if opposite:
            l,ltest=var_and_test("_lagr_un_connect")
            phiI,phiItest=self.get_phi_and_test(bulk=True)
            phiO,phiOtest=var_and_test(opposite.potential_name,domain="|..")
            #self.add_weak(l*scale_factor("spatial"),ltest)
            self.add_weak(dot(n,grad(phiI)-grad(phiO)),ltest)
            if self.total_mass_transfer_rate:
                pf=self.get_potential_flow()
                self.add_weak(-self.total_mass_transfer_rate*(1/pf.rho-1/opposite.rho),ltest)
            self.add_weak(l,phiItest)
            self.add_weak(-l,phiOtest)

    def get_laplace_pressure(self):
        if self.sigma is None:
            return 0
        else:
            return var("_curvature")*self.sigma
        
    def get_kinematic_boundary_condition(self,vectorial:bool=False):
        n=var("normal")
        x=var("mesh")
        potflow=self.get_potential_flow()
        u=grad(var(potflow.potential_name,domain=".."))
        j=self.total_mass_transfer_rate if self.total_mass_transfer_rate is not None else 0
        if vectorial:
            return mesh_velocity()-u+j/potflow.rho*n # (xdot-u)=0
        else:
            return dot(n,mesh_velocity()-u)+j/potflow.rho # n*(xdot-u)=0
    
    def get_dynamic_boundary_condition(self):
        potflow=self.get_potential_flow()
        phi=var(potflow.potential_name)
        phiB=var(potflow.potential_name,domain="..")        
        x=var("mesh")
        # Custom ALE here. Must be advected with bulk gradient
        #dtphi_moving=partial_t(phi,ALE=False)-dot(partial_t(x),grad(phiB))
        #inertia=dtphi_moving+1/2*dot(grad(phiB),grad(phiB)) # In total, it gives partial_t(phi,ALE=False)-dot(grad(phiB),grad(phiB))...
        inertia=partial_t(phi,ALE=False)-1/2*dot(grad(phiB),grad(phiB))
        pL=self.get_laplace_pressure()    
        # TODO: Viscosity
        traction=-(-pL+self.additional_pressure)
        if potflow.dynamic_viscosity is not None:
            n=var("normal")
            u=potflow.get_interface_velocity_for_viscous_at_interfaces(domain="..")            
            traction-=2*potflow.dynamic_viscosity*dot(n,matproduct(sym(grad(u)),n))                    
        opposite=self.get_potential_flow_on_opposite_side()
        if opposite:
            phiO=var(opposite.potential_name,domain="|..")
            inertia-=(partial_t(phiO,ALE=False)-1/2*dot(grad(phiO),grad(phiO)))*opposite.rho/potflow.rho
            if opposite.dynamic_viscosity is not None:
                n=var("normal")
                u=opposite.get_interface_velocity_for_viscous_at_interfaces(domain="|..")            
                traction+=2*opposite.dynamic_viscosity*dot(n,matproduct(sym(grad(u)),n))*(opposite.rho/potflow.rho)
        return inertia-traction/potflow.rho


#Imposing the dyn BC by shifting the mesh so that the dynBC is fulfilled
#Imposing the kin BC by setting a NeumannBC for phi to set the velocity to partial_t(x)*n
class PotentialFlowFreeInterface1(_PotentialFlowFreeInterfaceBase):
    def __init__(self, *, additional_pressure: ExpressionOrNum = 0, surface_tension: ExpressionNumOrNone = None, curvature_sign: int = -1,total_mass_transfer_rate:Optional[ExpressionOrNum]=None):
        super().__init__(additional_pressure=additional_pressure, surface_tension=surface_tension, curvature_sign=curvature_sign,total_mass_transfer_rate=total_mass_transfer_rate)
    def define_fields(self):
        potflow=self.get_potential_flow()
        self.define_scalar_field("_lagr_dynbc",potflow.space,scale=1/test_scale_factor("mesh"), testscale=scale_factor("temporal")/scale_factor(potflow.potential_name))
        return super().define_fields()
    
    def define_residuals(self):
        potflow=self.get_potential_flow()
        x,xtest=var_and_test("mesh")
        phi,phitest=var_and_test(potflow.potential_name)
        n=var("normal")
        ldyn,ldyn_test=var_and_test("_lagr_dynbc")

        dyn_bc=self.get_dynamic_boundary_condition()
        self.add_weak(dyn_bc,ldyn_test)
        self.add_weak(ldyn*n,xtest)

        self.add_weak(-dot(mesh_velocity(),n),phitest)
        return super().define_residuals()           

    def before_assigning_equations_postorder(self, mesh: AnyMesh):        
        self.pin_redundant_lagrange_multipliers(mesh,"_lagr_dynbc","mesh")



#Imposing the kin BC by moving the mesh in normal direction
#Imposing the dyn BC by adjusting phi
class PotentialFlowFreeInterface2(_PotentialFlowFreeInterfaceBase):
    def __init__(self, *, additional_pressure: ExpressionOrNum = 0, surface_tension: ExpressionNumOrNone = None, curvature_sign: int = -1,total_mass_transfer_rate:Optional[ExpressionOrNum]=None):
        super().__init__(additional_pressure=additional_pressure, surface_tension=surface_tension, curvature_sign=curvature_sign,total_mass_transfer_rate=total_mass_transfer_rate)

    def define_fields(self):
        potflow=self.get_potential_flow()
        self.define_scalar_field("_lagr_kinbc",potflow.space,scale=1/test_scale_factor("mesh"), testscale=1/scale_factor(potflow.velocity_name))
        self.define_scalar_field("_lagr_dynbc",potflow.space,scale=1/test_scale_factor(potflow.potential_name),testscale=scale_factor("temporal")/scale_factor(potflow.potential_name))
        return super().define_fields()
    
    def define_residuals(self):
        potflow=self.get_potential_flow()
        x,xtest=var_and_test("mesh")
        phi,phitest=var_and_test(potflow.potential_name)
        n=var("normal")
        ldyn,ldyn_test=var_and_test("_lagr_dynbc")
        lkin,lkin_test=var_and_test("_lagr_kinbc")

        dyn_bc=self.get_dynamic_boundary_condition()
        self.add_weak(dyn_bc,ldyn_test)
        self.add_weak(ldyn,phitest)

        kin_bc=self.get_kinematic_boundary_condition()
        self.add_weak(kin_bc,lkin_test)
        self.add_weak(n*lkin,xtest)

        return super().define_residuals()        


    def before_assigning_equations_postorder(self, mesh: AnyMesh):        
        potflow=self.get_potential_flow()
        self.pin_redundant_lagrange_multipliers(mesh,"_lagr_dynbc",potflow.potential_name)
        self.pin_redundant_lagrange_multipliers(mesh,"_lagr_kinbc","mesh")





#Imposing the kin BC by moving the mesh with the flow, normally and tangetially
#Imposing the dyn BC by adjusting phi
class PotentialFlowFreeInterface3(_PotentialFlowFreeInterfaceBase):
    def __init__(self, *, additional_pressure: ExpressionOrNum = 0, surface_tension: ExpressionNumOrNone = None, curvature_sign: int = -1,total_mass_transfer_rate:Optional[ExpressionOrNum]=None):
        super().__init__(additional_pressure=additional_pressure, surface_tension=surface_tension, curvature_sign=curvature_sign,total_mass_transfer_rate=total_mass_transfer_rate)

    def define_fields(self):
        potflow=self.get_potential_flow()
        self.define_vector_field("_lagr_kinbc",potflow.space,scale=1/test_scale_factor("mesh"), testscale=1/scale_factor(potflow.velocity_name))
        self.define_scalar_field("_lagr_dynbc",potflow.space,scale=1/test_scale_factor(potflow.potential_name),testscale=scale_factor("temporal")/scale_factor(potflow.potential_name))
        return super().define_fields()
    
    def define_residuals(self):
        potflow=self.get_potential_flow()
        x,xtest=var_and_test("mesh")
        phi,phitest=var_and_test(potflow.potential_name)
        n=var("normal")
        ldyn,ldyn_test=var_and_test("_lagr_dynbc")
        lkin,lkin_test=var_and_test("_lagr_kinbc")

        dyn_bc=self.get_dynamic_boundary_condition()
        self.add_weak(dyn_bc,ldyn_test,coordinate_system=cartesian)
        self.add_weak(ldyn,phitest)

        kin_bc=self.get_kinematic_boundary_condition(vectorial=True)
        self.add_weak(kin_bc,lkin_test,coordinate_system=cartesian)
        self.add_weak(lkin,xtest,coordinate_system=cartesian)

        return super().define_residuals()        


    def before_assigning_equations_postorder(self, mesh: AnyMesh):        
        potflow=self.get_potential_flow()
        ndim=self.get_nodal_dimension()
        dirs=["x","y","z"]
        for i in range(ndim):
            self.pin_redundant_lagrange_multipliers(mesh,"_lagr_kinbc_"+dirs[i],"mesh")
        self.pin_redundant_lagrange_multipliers(mesh,"_lagr_dynbc",potflow.potential_name)



class PotentialFlowFreeInterface(_PotentialFlowInterfaceEquations):
    def __init__(self,*,additional_pressure:ExpressionOrNum=0,surface_tension:ExpressionNumOrNone=None,total_mass_transfer_rate:Optional[ExpressionOrNum]=None):
        super().__init__()
        self.additional_pressure=additional_pressure
        self.sigma=surface_tension
        self.new_version:bool=True
        self.total_mass_transfer_rate=total_mass_transfer_rate

    def define_fields(self):
        potflow=self.get_potential_flow()
        if self.new_version:
            self.define_scalar_field("_lagr_kinbc",potflow.space,scale=1/test_scale_factor("mesh"), testscale=1/scale_factor(potflow.velocity_name))
            self.define_scalar_field("_lagr_dynbc",potflow.space,scale=1/test_scale_factor(potflow.potential_name),testscale=scale_factor("temporal")/scale_factor(potflow.potential_name))
        else:
            self.define_scalar_field("_lagr_dynbc",potflow.space,scale=1/test_scale_factor("mesh"), testscale=scale_factor("temporal")/scale_factor(potflow.potential_name))
        if self.sigma is not None:
            self.define_vector_field("_proj_normal",potflow.space)
            self.define_scalar_field("_curvature",potflow.space,scale=1/scale_factor("spatial"),testscale=scale_factor("spatial"))

    def define_residuals(self):
        crdsys_lagr=None
        phi,phi_test=self.get_phi_and_test()
        phiB,_phiB_test=self.get_phi_and_test(bulk=True)
        potflow=self.get_potential_flow()        
        if potflow.rho is None:
            raise RuntimeError("Requires mass_density to be set in the PotentialFlow")
        
                
        x,xtest=var_and_test("mesh")
        n=var("normal")

        if self.new_version:
            phi_node_dot=-dot(grad(phiB),grad(phiB))/2+self.additional_pressure/potflow.rho

            if self.sigma is not None:
                pn,pn_test=var_and_test("_proj_normal")
                curv,curv_test=var_and_test("_curvature")
                self.add_weak(pn-n,pn_test,coordinate_system=crdsys_lagr)
                self.add_weak(curv-div(pn),curv_test)
                phi_node_dot+=self.sigma*curv/potflow.rho

            if potflow.dynamic_viscosity is not None:
                u=potflow.get_interface_velocity_for_viscous_at_interfaces(domain="..")
                
                #print(self.expand_expression_for_debugging(grad(u),collect_units=False))
                #exit()
                #self.add_local_function("strain1",-2*potflow.dynamic_viscosity*dot(n,matproduct(sym(grad(u)),n)))
                #self.add_local_function("strain1",-2*potflow.dynamic_viscosity*(grad(u)[0]))
                #self.add_local_function("strain2",4*potflow.dynamic_viscosity*dot(partial_t(x),n)/dot(x,n))
                #self.add_local_function("strain1",4*potflow.dynamic_viscosity*dot(n,u)/dot(x,n))
                #self.add_local_function("strain1",dot(n,grad(phiB)))
                #self.add_local_function("strain2",dot(partial_t(x),n))
                strain=2*potflow.dynamic_viscosity*dot(n,matproduct(sym(grad(u)),n))
                #strain=-4*potflow.dynamic_viscosity*dot(partial_t(x),n)/dot(x,n)
                phi_node_dot+=strain/potflow.rho

            ldyn,ldyntest=var_and_test("_lagr_dynbc")
            self.add_weak(partial_t(phi)+phi_node_dot,ldyntest,coordinate_system=crdsys_lagr)
            self.add_weak(ldyn,phi_test,coordinate_system=crdsys_lagr)
            
            lkin,lkintest=var_and_test("_lagr_kinbc")
            kinbc=dot(n,partial_t(x,ALE=True)-grad(phiB)) # The partial_t(x,ALE=True) does not make sense. Will be 0
            if self.total_mass_transfer_rate:
                kinbc-=self.total_mass_transfer_rate/potflow.rho
            self.add_weak(kinbc,lkintest,coordinate_system=crdsys_lagr)
            self.add_weak(lkin,dot(n,xtest),coordinate_system=crdsys_lagr)

        else:

            # Don't ask me why we have to take a minus here at the grad(phi)^2 term... But otherwise, it does not agree with the Rayleigh-Plesset
            dyn_bc=-partial_t(phi)-dot(grad(phiB),grad(phiB))/2+self.additional_pressure/potflow.rho

            if self.sigma is not None:
                pn,pn_test=var_and_test("_proj_normal")
                curv,curv_test=var_and_test("_curvature")
                self.add_weak(pn-n,pn_test)
                self.add_weak(curv-div(pn),curv_test)
                dyn_bc+=self.sigma*curv/potflow.rho

            if potflow.dynamic_viscosity is not None:
                u=potflow.get_interface_velocity_for_viscous_at_interfaces(domain="..")
                strain=-2*potflow.dynamic_viscosity*dot(n,matproduct(sym(grad(u)),n))
                dyn_bc+=strain/potflow.rho

            ldyn,ldyntest=var_and_test("_lagr_dynbc")
            self.add_weak(dyn_bc,ldyntest)
            self.add_weak(ldyn,dot(n,xtest))

            self.add_weak(mesh_velocity(),n*phi_test)
        

    def before_assigning_equations_postorder(self, mesh: AnyMesh):
        
        potflow=self.get_potential_flow()
        if self.new_version:
            self.pin_redundant_lagrange_multipliers(mesh,"_lagr_kinbc",potflow.potential_name)
            self.pin_redundant_lagrange_multipliers(mesh,"_lagr_dynbc","mesh")
        else:        
            self.pin_redundant_lagrange_multipliers(mesh,"_lagr_dynbc","mesh")
        
