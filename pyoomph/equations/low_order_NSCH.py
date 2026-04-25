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
from ..materials import *


# Piecewise potential to prevent overshooting
# Returns f=F'
class PiecewiseLowOrderNSCHPotential(CustomMultiReturnExpression):
    def __init__(self) -> None:
        super().__init__()
    
    def get_num_returned_scalars(self, nargs: int) -> int:
        assert nargs==1
        return 1
    
    def eval(self, flag: int, arg_list: NPFloatArray, result_list: NPFloatArray, derivative_matrix: NPFloatArray) -> None:
        phi=arg_list[0]
        if phi<-1:
            result_list[0]=2*(phi+1)
        elif phi>1:
            result_list[0]=2*(phi - 1)
        else:
            result_list[0]=phi*(phi**2 - 1)
        if flag:
            if abs(phi)>1:            
                derivative_matrix[0]=2
            else:            
                derivative_matrix[0]=3*phi**2 - 1
            
    def generate_c_code(self) -> str:
        return """
        const double phi=arg_list[0];
        if (phi<-1)
        {
            result_list[0]=2*(phi+1);
        }
        else if (phi>1)
        {
            result_list[0]=2*(phi - 1);
        }
        else
        {                
            result_list[0]=phi*(phi*phi - 1.0);
        }
        if (flag)
        {
            if (phi<-1)
            {
                derivative_matrix[0]=2.0;
            }
            else if (phi>1)
            {
                derivative_matrix[0]=2.0;
            }
            else
            {
                derivative_matrix[0]=3.0*phi*phi - 1.0;
            }
        }
         
        """
            



#https://arxiv.org/abs/1911.06718
class LowOrderNSCH(Equations):
    def __init__(self,rho_a,rho_b,mu_a,mu_b,epsilon,gamma,mobility,*,W=rational_num(1,4),gravity=0,low_order=True,with_cut_off=True,with_subexpressions=True,mobility_mode:int=0,phase_boundary_level:Union[int,Literal["max"]]=0,phase_boundary_threshold:float=0.8,incompressiblity_PI:bool=False,real_pressure:bool=False,use_sym_grad_u:bool=True,swap_test_functions:bool=True,piecewise_potential:bool=True,compression_term_lambda:ExpressionOrNum=0):
        super().__init__()
        self.low_order=low_order
        self.gravity=gravity
        self.with_cut_off=with_cut_off
        self.with_subexpressions=with_subexpressions
        self.rho_a,self.rho_b=rho_a,rho_b
        self.mu_a,self.mu_b=mu_a,mu_b
        self.epsilon,self.gamma=epsilon,gamma
        self.mobility=mobility
        self.mobility_mode=mobility_mode
        self.W=W
        self.phase_boundary_level=phase_boundary_level
        self.phase_boundary_threshold=phase_boundary_threshold
        self.incompressiblity_PI=incompressiblity_PI
        self.use_sym_grad_u=use_sym_grad_u
        self.real_pressure=real_pressure
        self.swap_test_functions=swap_test_functions
        self.piecewise_potential=piecewise_potential
        self.use_normal_for_slope=False
        self.compression_term_lambda=compression_term_lambda
               
        
    def subexpression(self,expr):
        if self.with_subexpressions:
            return subexpression(expr)
        else:
            return expr
        
    def define_fields(self):
        if self.swap_test_functions:
            testscale_phi=1/scale_factor("phi")
            testscale_mu=scale_factor("temporal")
        else:
            testscale_phi=scale_factor("temporal")
            testscale_mu=1/scale_factor("phi")
            
        
        high_space="C1" if self.low_order else "C2"
        self.define_vector_field("velocity",high_space,testscale=scale_factor("spatial") / scale_factor("pressure"))
        self.define_scalar_field("phi",high_space,testscale=testscale_phi)
        self.define_scalar_field("eta",high_space,scale=scale_factor("spatial")/self.mobility*scale_factor("velocity"), testscale=testscale_mu)
        self.define_scalar_field("pressure","C1",testscale=scale_factor("spatial") / scale_factor("velocity"))
        
    def get_rho(self):
        phi=var("phi")
        if self.with_cut_off:
            phi=self.subexpression(maximum(minimum(phi,1),-1))
        rho_diff=(self.rho_a-self.rho_b)/2
        rho_avg=(self.rho_a+self.rho_b)/2        
        return self.subexpression(rho_diff*phi+rho_avg)
    
    def get_mu(self):
        phi=var("phi")
        if self.with_cut_off:
            phi=self.subexpression(maximum(minimum(phi,1),-1))
        mu_diff=(self.mu_a-self.mu_b)/2
        mu_avg=(self.mu_a+self.mu_b)/2        
        return self.subexpression(mu_diff*phi+mu_avg)
        
    def get_mobility(self):
        if self.mobility_mode==0:
            return self.mobility
        elif self.mobility_mode==2:
            return self.subexpression(self.mobility*maximum((1-var("phi")**2),0))
        elif self.mobility_mode==3:
            phi_abs=self.subexpression(minimum(absolute(var("phi")),1))
            return self.subexpression(self.mobility*(2*phi_abs**3-3*phi_abs**2+1))
        else:
            raise RuntimeError("mobility_mode may be only 0, 2 or 3")
        
            
    def define_residuals(self):
        u,w=var_and_test("velocity")
        p,q=var_and_test("pressure")
        phi,s=var_and_test("phi")
        eta,v=var_and_test("eta")
        # Swapped implementation by default
        if not self.swap_test_functions:
            s,v=v,s
        rho=self.get_rho()
        mu=self.get_mu()
        M=self.get_mobility()
        rho_diff=(self.rho_a-self.rho_b)/2
        J=-rho_diff*M*grad(eta)
        kappa=3/(4*square_root(2*self.W))*self.gamma/self.epsilon
        if not self.piecewise_potential:
            _F=self.W*(1-phi**2)**2
            f=symbolic_diff(_F,phi)
        else:
            f=4*self.W*PiecewiseLowOrderNSCHPotential()(phi) 
        adv_term=rho*(material_derivative(u,u)-self.gravity)-(matproduct(transpose(grad(u)),J))
        if not self.real_pressure:
            adv_term-=kappa*eta*grad(phi)
        else:
            self.add_weak(-kappa/2*self.epsilon*dyadic(grad(phi),grad(phi)),grad(w))
        #if self.low_order:            
        #    adv_term=self.subexpression(adv_term)
        eq1=weak(adv_term,w)

        
        if self.use_sym_grad_u:
            eq1+=weak(2*mu*(sym(grad(u))),grad(w)) 
        else:
            eq1+=weak(2*mu*(grad(u)),grad(w)) 
        
        if self.incompressiblity_PI:
            eq1+=-weak(p,div(w))
            eq2=weak(u,grad(q))
        else:
            eq1+=weak(-identity_matrix()*p,grad(w))
            eq2=weak(div(u),q)
        
        
        
        #eq3=weak(partial_t(phi)+dot(u,grad(phi)),v)+weak(M*grad(eta),grad(v))
        eq3=weak(material_derivative(phi,u),v)+weak(M*grad(eta),grad(v))
        
        # http://dx.doi.org/10.1016/j.cnsns.2015.06.012
        if not (is_zero(self.compression_term_lambda)):
            eq3+=weak(self.compression_term_lambda*M*grad(phi),grad(v))
            gradphi_mag=square_root(dot(grad(phi),grad(phi)))
            compression=self.subexpression(M/(square_root(2)*self.epsilon)* (1-phi**2)*grad(phi)/(gradphi_mag+0.01/self.epsilon))
            eq3-=weak(self.compression_term_lambda*compression,grad(v))
        
        eq4=weak(eta-f,s)
        if self.use_normal_for_slope:
            normal=self.subexpression(grad(var("phi"))/square_root(dot(grad(var("phi")),grad(var("phi")))+0.01/self.epsilon**2))
            eq4-=weak(self.epsilon**2*dot(grad(phi),normal),dot(grad(s),normal)) 
        else:
            eq4-=weak(self.epsilon**2*grad(phi),grad(s)) 
        
        
        self.add_residual(eq1)
        self.add_residual(eq2)
        self.add_residual(eq3)
        self.add_residual(eq4)
        
        if self.low_order:   
            he=var("element_length_h")
            tau_p=he**2/(4*mu)
            tau_u=self.subexpression(1/square_root((1/tau_p**2+(2*rho/he)**2*dot(u,u))))
            stab_test=tau_u*rho*matproduct(transpose(grad(w)),u)+tau_p*grad(q)
            self.add_weak(adv_term+grad(p),stab_test)
            
            #self.add_weak(partial_t(phi)+dot(u,grad(phi)),10*tau_u*rho*dot(u,grad(v)))
            

    def calculate_error_overrides(self):
        if self.phase_boundary_level is None or self.phase_boundary_level==0:
            return
        mesh=self.get_current_code_generator()._mesh 
        assert mesh is not None
        must_refine = 100 * mesh.max_permitted_error
        may_not_unrefine = 0.5 * (mesh.max_permitted_error+mesh.min_permitted_error)        

        phi_index=mesh.get_code_gen().get_code().get_nodal_field_index("phi")
        for e in mesh.elements():
            refine_this_elem=False
            for ni in range(e.nnode()):
                 ph=e.node_pt(ni).value(phi_index)
                 if abs(ph)<self.phase_boundary_threshold:
                      refine_this_elem=True
                      break
            if not refine_this_elem:
                 continue
            #e._elemental_error_max_override
            if self.phase_boundary_level!="max":
                assert isinstance(self.phase_boundary_level,int)
                blk=e
                while blk.get_bulk_element() is not None:
                    blk=blk.get_bulk_element()
                lvl=blk.refinement_level()
                if lvl>=self.phase_boundary_level: 
                    e._elemental_error_max_override = max(e._elemental_error_max_override,may_not_unrefine)
                    continue
            e._elemental_error_max_override=must_refine        
            
            
class MaterialBasedLowOrderNSCH(LowOrderNSCH):
    def __init__(self, fluidA:AnyFluidProperties, fluidB:AnyFluidProperties, epsilon, mobility, *, interface:Optional[AnyFluidFluidInterface]=None,W=rational_num(1, 4), gravity=0, low_order=True, with_cut_off=True, with_subexpressions=True, mobility_mode: int = 0, phase_boundary_level: Union[int, Literal['max']] = 0, incompressiblity_PI: bool = False, real_pressure: bool = False, use_sym_grad_u: bool = True, swap_test_functions: bool = True, piecewise_potential:bool=True,compression_term_lambda:ExpressionOrNum=0):
        if interface is None:
            interface=fluidA | fluidB
        super().__init__(fluidA.mass_density, fluidB.mass_density, fluidA.dynamic_viscosity, fluidB.dynamic_viscosity, epsilon, interface.surface_tension, mobility, W=W, gravity=gravity, low_order=low_order, with_cut_off=with_cut_off, with_subexpressions=with_subexpressions, mobility_mode=mobility_mode, phase_boundary_level=phase_boundary_level, incompressiblity_PI=incompressiblity_PI, real_pressure=real_pressure, use_sym_grad_u=use_sym_grad_u, swap_test_functions=swap_test_functions,piecewise_potential=piecewise_potential,compression_term_lambda=compression_term_lambda)
        
        
        
class LowOrderNSCHWetting(InterfaceEquations):
    required_parent_type=LowOrderNSCH
    def __init__(self,contact_angle_for_fluidA:ExpressionOrNum):
        super().__init__()
        self.theta=contact_angle_for_fluidA
        
    def define_residuals(self):
        peqs=self.get_parent_equations()
        assert isinstance(peqs,LowOrderNSCH)
        gradphi=grad(var("phi",domain=".."))
        if peqs.use_normal_for_slope:
            normal=peqs.subexpression(gradphi/square_root(dot(gradphi,gradphi)+0.01/peqs.epsilon**2))
            gphi_norm=square_root(dot(gradphi,normal)**2)
        else:
            gphi_norm=square_root(dot(gradphi,gradphi))
        if not peqs.swap_test_functions:
            s=testfunction("eta")
        else:
            s=testfunction("phi")
        self.add_weak(peqs.epsilon**2*gphi_norm*cos(self.theta),s)
