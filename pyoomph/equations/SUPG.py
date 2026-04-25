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
 
 
from ..meshes.mesh import AnyMesh
from ..generic import Equations
from ..expressions import *  # Import grad et al
from ..equations.navier_stokes import StokesEquations,NavierStokesEquations
from ..typings import *
if TYPE_CHECKING:
    from ..generic.codegen import FiniteElementCodeGenerator

class ElementSizeForSUPG(Equations):
    """
    Projects the size (length/area/volume) of each element to a discontinuous ``"D0"`` space. Can be used for SUPG stabilization by getting the characteristic element length scale via :py:meth:`get_element_h`.
    
    Args:
        varname: Name of the variable to store the element size (length/area/volume) 
    """
    def __init__(self,*,varname:str="_supg_elemsize"):
        super(ElementSizeForSUPG, self).__init__()
        self.varname=varname

    def define_fields(self):
        # Element size: On refinement, also divide by n_sons^1, on unrefinement, multiply by it#
        self.define_scalar_field(self.varname, "D0", discontinuous_refinement_exponent=1)

    def define_scaling(self):
        self.set_scaling(**{self.varname:self.get_scaling("spatial")**self.get_element_dimension()})
        self.set_test_scaling(**{self.varname:1/scale_factor(self.varname)})

    def define_residuals(self):
        elemsize,elemsize_test=var_and_test(self.varname)
        self.add_residual(eval_flag("moving_mesh")*(elemsize * elemsize_test * self.get_nodal_delta() - weak(1, elemsize_test, coordinate_system=cartesian,dimensional_dx=True)))

    # Size for SUPG
    def get_element_h(self,domain:Optional[Union[str,"FiniteElementCodeGenerator"]]=None) -> Expression:
        """
        Returns the characteristic element length scale by taking the d-th root of the element size (length/area/volume), where d is the dimension of the element.

        Args:
            domain: Can be used for code generation to specify the domain of the variable. If None, the domain is inferred from the context.

        Returns:
            The typical length scale of the element.
        """
        return var(self.varname,domain=domain)**rational_num(1,self.get_element_dimension())
        
    def before_assigning_equations_postorder(self, mesh: AnyMesh):
        dg_index=None
        has_moving_nodes=True
        for e in mesh.elements():
            dg_index=e.get_code_instance().get_discontinuous_field_index(self.varname)
            has_moving_nodes=e.get_code_instance().has_moving_nodes
            break
        
        if dg_index is not None:
            for e in mesh.elements():
                e.internal_data_pt(dg_index).set_value(0,e.get_current_cartesian_nondim_size())
                if has_moving_nodes:
                    e.internal_data_pt(dg_index).pin(0)



class ElementSizeFromInitialCartesianSize(ElementSizeForSUPG):
    def define_fields(self):
        pass

    def define_scaling(self):
        pass

    def define_residuals(self):
        pass

    def get_element_h(self) -> Expression:
        return 0.01
    

class GenericStabilizationMethod(Equations):
    def __init__(self):
        super().__init__()
        self.velocity_epsilon=0
        self.velocity_history=0
        self.element_size_history=0

    def get_velocity_magnitude(self):
        u=var("velocity")
        if self.velocity_history>0:
            u=evaluate_in_past(u,self.velocity_history)
        return subexpression(square_root(dot(u,u)+self.velocity_epsilon**2))

    def get_element_size(self):
        cmb_eqs=self.get_combined_equations()
        esize=cmb_eqs.get_equation_of_type(ElementSizeForSUPG)
        if not isinstance(esize,ElementSizeForSUPG):
            raise RuntimeError("Must be combined with a single ElementSizeForSUPG")
        h=esize.get_element_h()
        if self.velocity_history>0:
            h=evaluate_in_past(h,self.element_size_history)
        return h

    def get_flow_equations(self):
        cmb_eqs=self.get_combined_equations()
        ns=cmb_eqs.get_equation_of_type(StokesEquations)
        if not isinstance(ns,StokesEquations):
            raise RuntimeError("Must be combined with a single (Navier)StokesEquations")        
        return ns
    
    def get_momentum_residual(self):
        ns=self.get_flow_equations()
        u,p=var(["velocity","pressure"])
        res=ns.mass_density*material_derivative(u,u)+grad(p)
        if ns.bulkforce is not None:
            res-=ns.bulkforce
        if ns.gravity is not None:
            res-=ns.mass_density*ns.gravity
        return res



class PSPG(GenericStabilizationMethod):
    def __init__(self,tau_name:Optional[str]="_tau_PSPG",U_name:Optional[str]=None,velocity_offset=1e-5):
        super().__init__()
        self.tau_name=tau_name
        self.U_name=U_name        
        self.velocity_offset=velocity_offset

    def define_fields(self):
        if self.tau_name is not None:
            self.define_scalar_field(self.tau_name,"D0")
        if self.U_name is not None:
            self.define_scalar_field(self.U_name,"D0")


    def get_tau(self):        
        h=self.get_element_size()        
        ns=self.get_flow_equations()
        rho=ns.mass_density
        mu=ns.dynamic_viscosity
        u=var("velocity")
        if self.U_name is not None:
            U,Utest=var_and_test(self.U_name)
            self.add_residual(weak(U-subexpression(square_root(dot(u,u)+self.velocity_offset**2)),Utest))
        else:        
            U=subexpression(square_root(dot(u,u)+self.velocity_offset**2))
        Re=U*rho*h/(2*mu)        
        z=minimum(Re/3,1)
        tau_def=h/(2*U+self.velocity_offset)*z
        if self.tau_name is not None:
            tau,tau_test=var_and_test(self.tau_name)            
        else:
            tau=tau_def
        return tau

    def define_residuals(self):        
        ns=self.get_flow_equations()
        u=var("velocity")
        p,q=var_and_test("pressure")
        if self.U_name is not None:
            U,Utest=var_and_test(self.U_name)
            self.add_residual(weak(U-subexpression(square_root(dot(u,u)+self.velocity_offset**2)),Utest))
        else:        
            U=subexpression(square_root(dot(u,u)+self.velocity_offset**2))
        #Us=1
        rho=ns.mass_density
        tau=self.get_tau()
        moment_residual=self.get_momentum_residual()

        self.add_residual(weak(tau*moment_residual,grad(q)/rho))

        return super().define_residuals()
    

class ASGS(GenericStabilizationMethod):
    def __init__(self):
        super().__init__()
        self.alpha=1
        self.velocity_epsilon=1e-5*scale_factor("velocity")
        self.velocity_history=1
        self.element_size_history=1

    def get_tau1(self):
        ns=self.get_flow_equations()
        h=self.get_element_size()        
        rho,mu=ns.mass_density,ns.dynamic_viscosity
        
        tau1 =(self.alpha*timestepper_weight(1,0,"BDF1")+4*mu/(h**2*rho))**(-1)

        U=self.get_velocity_magnitude()
        tau1=(2*rho*U/h+4*mu/h**2)**(-1)
        return tau1
    
    def get_tau2(self):
        ns=self.get_flow_equations()
        h=self.get_element_size()        
        rho,mu=ns.mass_density,ns.dynamic_viscosity
        U=self.get_velocity_magnitude()
        tau2=mu/rho
        tau2=mu+rho*U*h/(2)
        return tau2

    def define_residuals(self):
        tau1=self.get_tau1()
        tau2=self.get_tau2()
        moment_residual=self.get_momentum_residual()
        q=testfunction("pressure")
        self.add_residual(weak(tau1*moment_residual,grad(q)))
        u,w=var_and_test("velocity")
        self.add_residual(weak(tau2*div(u),div(w)))
        return super().define_residuals()
