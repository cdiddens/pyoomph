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
 
 
from ..generic import Equations,InterfaceEquations
from ..expressions import *  # Import grad et al
from ..typings import *

if TYPE_CHECKING:
    from ..materials.generic import PureLiquidProperties,PureGasProperties,MixtureLiquidProperties,MixtureGasProperties


class CahnHilliardEquation(Equations):
    """
        Represents the Cahn-Hilliard equations for phase separation in a binary fluid mixture, given by the following equations:

        mu = 3 * sigma / 2**(3/2) / epsilon * (c**3 - c) + 3 * sigma / 2**(3/2) * epsilon * laplace(c)
        dc/dt = mobility * laplace(mu)

        Args:
            sigma (ExpressionOrNum): Surface tension coefficient. Default is 1.
            epsilon (ExpressionOrNum): Interface thickness. Default is 1.
            mobility (ExpressionOrNum): Mobility coefficient. Default is 1.
            space (FiniteElementSpaceEnum): Finite element space for the fields. Default is "C2".
            phase_name (str): Name of the phase field variable. Default is "c".
            potential_name (str): Name of the chemical potential variable. Default is "mu".
            temporal_error_factor (float): Mutiplicative factor for the temporal error estimator by for the phase field. Default is 0.
    """
    def __init__(self,sigma:ExpressionOrNum=1,epsilon:ExpressionOrNum=1,mobility:ExpressionOrNum=1,space:FiniteElementSpaceEnum="C2",phase_name:str="c",potential_name:str="mu",temporal_error_factor:float=0):
        super(CahnHilliardEquation, self).__init__()
        self.mobility=mobility
        self.sigma=sigma
        self.epsilon=epsilon
        self.space:FiniteElementSpaceEnum=space
        self.phase_name=phase_name
        self.potential_name=potential_name
        self.temporal_error_factor=temporal_error_factor

    def define_fields(self):
        self.define_scalar_field(self.phase_name,self.space)
        self.define_scalar_field(self.potential_name,self.space)

    def define_residuals(self):
        c,c_test=var_and_test(self.phase_name)
        mu, mu_test = var_and_test(self.potential_name)
        MU = self.get_scaling(self.potential_name)

        psiprime = c * (c ** 2 - 1)
        sigma_tilde=3/(2*2**rational_num(1,2))*self.sigma
        self.add_residual(weak(1 / MU * mu , mu_test ))
        self.add_residual(weak(1 / MU * (-sigma_tilde / self.epsilon * psiprime) , mu_test ))
        self.add_residual(weak(1 / MU * (-self.epsilon * sigma_tilde *  grad(c)), grad(mu_test)) )

        T=self.get_scaling("temporal")
        self.add_residual(weak(T * partial_t(c) ,c_test)  )
        self.add_residual(weak(T * self.mobility * grad(mu), grad(c_test)))

    def define_error_estimators(self):
        self.add_spatial_error_estimator(grad(var(self.phase_name),nondim=True))
        if self.temporal_error_factor>0:
            self.set_temporal_error_factor(self.phase_name,self.temporal_error_factor)


class CahnHilliardEquationsForNSCH(CahnHilliardEquation):
    """
        Represents the Cahn-Hilliard equations for phase separation in a binary fluid mixture, inheriting from CahnHilliardEquation and adding the advection of the phase field and the forcing of the surface tension by the velocity field. 
        The additional equations to the Cahn-Hilliard object are given by:

            TODO: Add equations


        Args:
            sigma (ExpressionOrNum): Surface tension coefficient. Default is 1.
            epsilon (ExpressionOrNum): Interface thickness. Default is 1.
            mobility (ExpressionOrNum): Mobility coefficient. Default is 1.
            space (FiniteElementSpaceEnum): Finite element space for the fields. Default is "C2".
            phase_name (str): Name of the phase field variable. Default is "c".
            potential_name (str): Name of the chemical potential variable. Default is "mu".
            velocity_name (str): Name of the velocity field variable. Default is "velocity".
            temporal_error_factor (float): Mutiplicative factor for the temporal error estimator by for the phase field. Default is 0.    
    """
    def __init__(self,sigma:ExpressionOrNum=1,epsilon:ExpressionOrNum=1,mobility:ExpressionOrNum=1,space:FiniteElementSpaceEnum="C2",phase_name:str="c",potential_name:str="mu",velocity_name:str="velocity",temporal_error_factor:float=0):
        super(CahnHilliardEquationsForNSCH, self).__init__(sigma=sigma,epsilon=epsilon,mobility=mobility,space=space,phase_name=phase_name,potential_name=potential_name,temporal_error_factor=temporal_error_factor)
        self.velocity_name=velocity_name
    def define_residuals(self):
        super(CahnHilliardEquationsForNSCH, self).define_residuals()
        u,u_test=var_and_test(self.velocity_name)
        c,c_test=var_and_test(self.phase_name)
        T=self.get_scaling("temporal")
        P = self.get_scaling("pressure")
        X = self.get_scaling("spatial")
        # Add the advection of the phase field
        self.add_residual(-T*(weak(c*u,grad(c_test))))
        #And the forcing of the surface tension
        tens=matproduct(grad(c),transpose(grad(c)))
        sigma_tilde = 3 / (2 * 2 ** rational_num(1, 2)) * self.sigma
        self.add_residual(-X/P*sigma_tilde*self.epsilon*weak(tens,grad(u_test)))



class CahnHilliardWettingInterface(InterfaceEquations):
    """
    TODO: Add description    
    """
    required_parent_type=CahnHilliardEquation

    def __init__(self,sigma_fs_plus:Optional[ExpressionOrNum]=None,sigma_fs_minus:Optional[ExpressionOrNum]=None,contact_angle_plus:Optional[ExpressionOrNum]=None,contact_angle_minus:Optional[ExpressionOrNum]=None):
        super(CahnHilliardWettingInterface, self).__init__()
        self.sigma_fs_plus=sigma_fs_plus
        self.sigma_fs_minus=sigma_fs_minus
        self.contact_angle_plus=contact_angle_plus
        self.contact_angle_minus = contact_angle_minus
        if (self.contact_angle_plus is not None) and (self.contact_angle_minus is not None):
            raise ValueError("Cannot specify both contact angles simultaneously")
        if (self.sigma_fs_plus is None) and (self.sigma_fs_minus is None):
            if (self.contact_angle_plus is None) and (self.contact_angle_minus is None):
                raise ValueError("Please set one of the following three options: sigma_fs_plus & sigma_fs_minus, contact_angle_plus or contact_angle_minus, i.e. the fluid-solid surface tensions of the phases c=1 and c=-1, the contact angle in phase c=1 or the one in the c=-1 phase")
        if (self.sigma_fs_plus is None)!=(self.sigma_fs_minus is None):
            raise ValueError(
                "Please set one of the following three options: sigma_fs_plus & sigma_fs_minus, contact_angle_plus or contact_angle_minus, i.e. the fluid-solid surface tensions of the phases c=1 and c=-1, the contact angle in phase c=1 or the one in the c=-1 phase")
        if (self.sigma_fs_plus is not None):
            if ((self.contact_angle_plus is not None) or (self.contact_angle_minus is not None)):
                raise ValueError("Please set one of the following three options: sigma_fs_plus & sigma_fs_minus, contact_angle_plus or contact_angle_minus, i.e. the fluid-solid surface tensions of the phases c=1 and c=-1, the contact angle in phase c=1 or the one in the c=-1 phase")

    def define_residuals(self):
        parent = self.get_parent_equations()
        assert isinstance(parent,CahnHilliardEquation)

        #Get missing surface tension from contact angle if necessary
        if (self.sigma_fs_plus is None):
            if self.contact_angle_plus is not None:
                delta_sigma=parent.sigma*cos(self.contact_angle_plus)
            else:
                delta_sigma=-parent.sigma * cos(self.contact_angle_minus)
        else:
            assert self.sigma_fs_minus is not None
            delta_sigma =self.sigma_fs_minus-self.sigma_fs_plus

        c,_=var_and_test(parent.phase_name)
        _,mu_test=var_and_test(parent.potential_name)
        add=1/4*(3*c**2-3)*delta_sigma
        X=self.get_scaling("spatial")
        MU=self.get_scaling(parent.potential_name)
        add = -add / (MU*X)
        self.add_residual(weak(add,mu_test))





def clamp_0_1(x:ExpressionOrNum)->Expression:
    return heaviside(x)*x-heaviside(x-1)*(x-1)

def smooth_clamp(x:ExpressionOrNum,mini:ExpressionOrNum,maxi:ExpressionOrNum)->Expression:
    x=clamp_0_1((x-mini)/(maxi-mini))
    return x*x*x*(x*(x*6-15)+10)

#See DOI:  doi:10.3390/math8081224
class SimpleNSCH(Equations):
    """
    TODO: Add description    
    """
    from ..materials.generic import AnyFluidProperties
    def __init__(self,fluid_plus:AnyFluidProperties,fluid_minus:AnyFluidProperties,sigma:ExpressionOrNum=1,epsilon:ExpressionOrNum=1,mobility:ExpressionOrNum=1,space:FiniteElementSpaceEnum="C2",phase_name:str="c",potential_name:str="mu",velocity_name:str="velocity",temporal_error_factor:float=0,dyadic_forcing:bool=False):
        super(SimpleNSCH, self).__init__()
        self.mobility=mobility
        self.sigma=sigma
        self.epsilon=epsilon
        self.space:FiniteElementSpaceEnum=space
        self.phase_name=phase_name
        self.potential_name=potential_name
        self.temporal_error_factor=temporal_error_factor
        self.velocity_name = velocity_name
        self.fluid_plus=fluid_plus
        self.fluid_minus=fluid_minus
        self.dyadic_forcing=dyadic_forcing



    def get_density_for_ns(self) -> Expression:
        phi=var(self.phase_name)
        phi = subexpression(smooth_clamp(phi,-1,1))
        return subexpression(self.fluid_plus.mass_density*(1+phi)/2+self.fluid_minus.mass_density*(1-phi)/2)


    def get_viscosity_for_ns(self,use_volume_fraction:bool=True) -> Expression:
        phi=var(self.phase_name)
        phi = subexpression(smooth_clamp(phi, -1, 1))
        return subexpression(self.fluid_plus.dynamic_viscosity*(1+phi)/2+self.fluid_minus.dynamic_viscosity*(1-phi)/2)

    def define_fields(self):
        self.define_scalar_field(self.phase_name,self.space)
        self.define_scalar_field(self.potential_name,self.space)

    def define_residuals(self):
        dx=self.get_dx(use_scaling=False)
        c,c_test=var_and_test(self.phase_name)
        mu, mu_test = var_and_test(self.potential_name)
        u, u_test = var_and_test(self.velocity_name)
        MU = self.get_scaling(self.potential_name)

        psiprime = c * (c ** 2 - 1)
        sigma_tilde=3/(2*2**rational_num(1,2))*self.sigma
        self.add_residual(1 / MU * mu * mu_test * dx)
        self.add_residual(1 / MU * (-sigma_tilde / self.epsilon * psiprime) * mu_test * dx)
        self.add_residual(1 / MU * (-self.epsilon * sigma_tilde *  dot(grad(c), grad(mu_test)) * dx))

        T=self.get_scaling("temporal")
        self.add_residual(T * partial_t(c) * c_test  * dx)
        self.add_residual(T * ( self.mobility * dot(grad(mu), grad(c_test)) * dx))


        P = self.get_scaling("pressure")
        X = self.get_scaling("spatial")
        # Add the advection of the phase field
        self.add_residual(-T * (dot(c * u, grad(c_test))) * dx)

        # And the forcing of the surface tension
        tens = matproduct(grad(c), transpose(grad(c)))
        sigma_tilde = 3 / (2 * 2 ** rational_num(1, 2)) * self.sigma
        if self.dyadic_forcing:
            self.add_residual(-X / P * self.epsilon* sigma_tilde * contract(tens, grad(u_test)) * dx)
            W=1/(4*self.epsilon**2)*(c**2-1)**2
            diagmat=(1/2*dot(grad(c),grad(c))+W)*identity_matrix(self.get_element_dimension())
            self.add_residual(X / P * self.epsilon * sigma_tilde * contract(diagmat, grad(u_test)) * dx)
        else:
            self.add_residual(-X / P  * mu*dot(grad(c),u_test) * dx)

        #10.1103/PhysRevE.67.066307
        self.add_residual(X/P*(dot(grad(c),grad(c))*dot(self.epsilon*grad(sigma_tilde),u_test)*dx))


    def define_error_estimators(self):
        self.add_spatial_error_estimator(grad(var(self.phase_name),nondim=True))
        if self.temporal_error_factor>0:
            self.set_temporal_error_factor(self.phase_name,self.temporal_error_factor)


class SimpleNSCHWettingInterface(CahnHilliardWettingInterface):
    """
    TODO: Add description    
    """
    required_parent_type=SimpleNSCH

    def __init__(self,sigma_fs_plus:Optional[ExpressionOrNum]=None,sigma_fs_minus:Optional[ExpressionOrNum]=None,contact_angle_plus:Optional[ExpressionOrNum]=None,contact_angle_minus:Optional[ExpressionOrNum]=None):
        super(SimpleNSCHWettingInterface, self).__init__(sigma_fs_plus, sigma_fs_minus, contact_angle_plus, contact_angle_minus)

