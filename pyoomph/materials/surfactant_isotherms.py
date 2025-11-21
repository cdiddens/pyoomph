#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2025  Christian Diddens & Duarte Rocha
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
 
 
#Taken from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7995737/

from ..expressions.phys_consts import gas_constant
from ..expressions.units import *
from ..expressions import *
from .generic import BaseInterfaceProperties,LiquidGasInterfaceProperties, LiquidSolidInterfaceProperties

# Default rates: Please use the ones appropriate for the surfactant you are using
# Not by modifying these parameters here, but by passing k_ads and k_des to the Isotherm constructor
# These are just examples showing the required units of these quantities

_default_k_ads= 5e-7 * meter / second # Taken from the thesis of Ruben van Gaalen
_default_k_des= 9.5e-1 / second # Taken from the thesis of Ruben van Gaalen
# Used in Frumkin desorption and in and the vdW-isotherm (set to 0 here, just to show the required units!)
_default_beta=0*meter**4*kilogram/(mol**2*second**2)


# Base class for surfactant isotherms
class SurfactantIsotherm:
    """
    Base class for a surfactant isotherm. This class should not be used directly, but should be subclassed.
    
    Args:
        surfactant_name: Name of the surfactant
        k_ads: Adsorption rate (in m/s)
        k_des: Desorption rate (in 1/s)
        K: Equilibrium constant (in m). If K is passed, k_ads and k_des are not independent, and only two of these can be passed.
    """
    def __init__(self,surfactant_name:str,k_ads:ExpressionNumOrNone=_default_k_ads,k_des:ExpressionNumOrNone=_default_k_des,K:ExpressionNumOrNone=None):
        self.surfactant_name=surfactant_name # Name of the surfactant
        if K is not None:
            if k_ads is not None and k_des is not None:
                raise ValueError("You have passed K, k_ads and k_des to the isotherm, but these are not independent. Please pass exactly two of these")
            elif k_ads is None and k_des is not None:
                k_ads=K*k_des
            elif k_des is None and k_ads is not None :
                k_des=k_ads/K
        self.k_ads:ExpressionOrNum=cast(ExpressionOrNum,k_ads) # Rates
        self.k_des:ExpressionOrNum=cast(ExpressionOrNum,k_des)

    def get_C_variable(self) -> Expression:
        return var("molarconc_"+self.surfactant_name) # Get variable C in molar concentration (mol/meter**3)

    def get_Gamma_variable(self) -> Expression:
        return var("surfconc_" + self.surfactant_name) # Get Gamma in mol/meter**2

    def get_T_variable(self) -> Expression:
        return var("temperature") # Temperature in Kelvin

    # This must be specialized to yield the surface pressure
    def get_surface_pressure(self)->ExpressionOrNum:
        raise NotImplementedError()

    # This must be specialized to yield the adsorption flux
    def get_adsorption_flux(self)->ExpressionOrNum:
        raise NotImplementedError()

    # This must be specialized to yield the desorption flux
    def get_desorption_flux(self)->ExpressionOrNum:
        raise NotImplementedError()

    # Set to a LiquidGasInterfaceProperties object
    # pure_surface_tension is the surface tension without surfactants
    # min_surface_tension can be set to prevent negative or zero surface tension.
    # In that case, it has to be positive (0 is not okay)
    def apply_on_interface(self,interface_properties:BaseInterfaceProperties,pure_surface_tension:ExpressionNumOrNone=None,min_surface_tension:ExpressionNumOrNone=None):
        """
        Apply the surfactant isotherm to interface properties.

        Args:
            interface_properties: The interface properties to apply the isotherm to
            pure_surface_tension: Surface tension without surfactants. If not set, it will be taken from the interface properties.
            min_surface_tension: Minimum surface tension. If set, the surface tension will be limited to this value.
        """
        assert isinstance(interface_properties,(LiquidGasInterfaceProperties,LiquidSolidInterfaceProperties))
        if pure_surface_tension is None:
            pure_surface_tension=interface_properties.surface_tension
        pure_surface_tension=cast(ExpressionOrNum,pure_surface_tension)
        interface_properties.surface_tension=cast(ExpressionOrNum,pure_surface_tension-self.get_surface_pressure())
        if min_surface_tension is not None:
            if is_zero(min_surface_tension):
                raise RuntimeError("min_surface_tension may not be zero")
            interface_properties.surface_tension=min_surface_tension*maximum(1,interface_properties.surface_tension/min_surface_tension)
        interface_properties.surfactant_adsorption_rate[self.surfactant_name]=self.get_adsorption_flux()-self.get_desorption_flux()


# Henry isotherm:
# Surface pressure: R*T*Gamma
# Adsorption rate: k_ads*C
# Desorption rate: k_des*Gamma
# Equilibrium: Gamma=K*C
class HenryIsotherm(SurfactantIsotherm):
    """
    Henry isotherm for surfactant adsorption. This is a simple isotherm where the surface pressure is proportional to the surfactant concentration in the bulk:
    The surface pressure reads :math:`RT\\Gamma`, the adsorption rate is :math:`k_\\mathrm{ads}C` and the desorption rate is :math:`k_\\mathrm{des}\\Gamma`. The equilibrium is given by :math:`\\Gamma=K C`.
    
    Args:
        surfactant_name: Name of the surfactant
        k_ads: Adsorption rate (in m/s)
        k_des: Desorption rate (in 1/s)
        K: Equilibrium constant (in m). If K is passed, k_ads and k_des are not independent, and only two of these can be passed.
    """
    def __init__(self,surfactant_name:str,k_ads:ExpressionNumOrNone=_default_k_ads,k_des:ExpressionNumOrNone=_default_k_des,K:ExpressionNumOrNone=None):
        super(HenryIsotherm, self).__init__(surfactant_name,k_ads,k_des,K=K)

    def get_adsorption_flux(self) -> Expression:
        C=self.get_C_variable()
        return self.k_ads*C

    def get_desorption_flux(self) -> Expression:
        Gamma=self.get_Gamma_variable()
        return self.k_des*Gamma

    def get_surface_pressure(self) -> Expression:
        T = self.get_T_variable()
        Gamma = self.get_Gamma_variable()
        return gas_constant*T*Gamma


# Langmuir isotherm:
# Surface pressure: -R*T*GammaInfty*ln(1-Gamma/GammaInfty)
# Adsorption rate: k_ads*C*(GammaInfty-Gamma)/GammaInfty
# Desorption rate: k_des*Gamma
# Equilibrium: Gamma=GammaInfty*K*C/(1+K*C)
class LangmuirIsotherm(SurfactantIsotherm):
    """
    Langmuir isotherm for surfactant adsorption.
    The surface pressure reads :math:`-RT\\Gamma_\\infty\\log(1-\\Gamma/\\Gamma_\\infty)`, the adsorption rate is :math:`k_\\mathrm{ads}C(\\Gamma-\\Gamma_\\infty)/\\Gamma_\\infty` and the desorption rate is :math:`k_\\mathrm{des}\\Gamma`. The equilibrium is given by :math:`\\Gamma=\\Gamma_\\infty KC/(1+KC)`.
    
    Args:
        surfactant_name: Name of the surfactant
        GammaInfty: Saturation surface concentration (in mol/m^2)
        k_ads: Adsorption rate (in m/s)
        k_des: Desorption rate (in 1/s)
        K: Equilibrium constant (in m). If K is passed, k_ads and k_des are not independent, and only two of these can be passed.
    """    
    def __init__(self,surfactant_name:str,GammaInfty:ExpressionOrNum,k_ads:ExpressionNumOrNone=_default_k_ads,k_des:ExpressionNumOrNone=_default_k_des,K:ExpressionNumOrNone=None):
        super(LangmuirIsotherm, self).__init__(surfactant_name,k_ads,k_des,K=K)
        self.GammaInfty=GammaInfty

    def get_surface_pressure(self) -> Expression:
        T = self.get_T_variable()
        Gamma = self.get_Gamma_variable()
        # Preventing the log to get negative numbers
        log_arg=maximum(1e-50, 1 - Gamma / self.GammaInfty)
        return -gas_constant * T * self.GammaInfty * log(log_arg)

    def get_adsorption_flux(self) -> Expression:
        C=self.get_C_variable()
        Gamma = self.get_Gamma_variable()
        # Note that k_ads is normalized here by GammaInfty (this is not always the case)
        return self.k_ads / self.GammaInfty * C * (self.GammaInfty - Gamma)

    def get_desorption_flux(self) -> Expression:
        Gamma=self.get_Gamma_variable()
        return self.k_des*Gamma

# Volmer isotherm:
# Surface pressure: R*T*GammaInfty*1/(1-Gamma/GammaInfty)
# Adsorption rate: k_ads*C*(GammaInfty-Gamma)/GammaInfty
# Desorption rate: k_des*Gamma*exp(Gamma/(GammaInfty-Gamma))
class VolmerIsotherm(SurfactantIsotherm):
    """
    Volmer isotherm for surfactant adsorption.
    The surface pressure reads :math:`RT\\Gamma_\\infty/(1-\\Gamma/\\Gamma_\\infty)`, the adsorption rate is :math:`k_\\mathrm{ads}C(\\Gamma-\\Gamma_\\infty)/\\Gamma_\\infty` and the desorption rate is :math:`k_\\mathrm{des}\\Gamma\\exp(\\Gamma/(\\Gamma_\\infty-\\Gamma))`. .
    
    Args:
        surfactant_name: Name of the surfactant
        GammaInfty: Saturation surface concentration (in mol/m^2)
        k_ads: Adsorption rate (in m/s)
        k_des: Desorption rate (in 1/s)
    """ 
    def __init__(self,surfactant_name:str,GammaInfty:ExpressionOrNum,k_ads:ExpressionNumOrNone=_default_k_ads,k_des:ExpressionNumOrNone=_default_k_des):
        super(VolmerIsotherm, self).__init__(surfactant_name,k_ads,k_des)
        self.GammaInfty=GammaInfty

    def get_surface_pressure(self) -> Expression:
        T = self.get_T_variable()
        Gamma = self.get_Gamma_variable()
        # Preventing the denom to become singular
        denom_arg=maximum(1e-50, 1 - Gamma / self.GammaInfty)
        return gas_constant * T * Gamma/denom_arg

    def get_adsorption_flux(self) -> Expression:
        C=self.get_C_variable()
        Gamma = self.get_Gamma_variable()
        # Note that k_ads is normalized here by GammaInfty (this is not always the case)
        return self.k_ads / self.GammaInfty * C * (self.GammaInfty - Gamma)

    def get_desorption_flux(self) -> Expression:
        Gamma=self.get_Gamma_variable()
        return self.k_des*Gamma*exp(Gamma/(self.GammaInfty-Gamma))


class FrumkinIsotherm(SurfactantIsotherm):
    """
    Frumkin isotherm for surfactant adsorption.
    The surface pressure reads :math:`-RT\\Gamma_\\infty\\log(1-\\Gamma/\\Gamma_\\infty)`, the adsorption rate is :math:`k_\\mathrm{ads}C(\\Gamma-\\Gamma_\\infty)/\\Gamma_\\infty` and the desorption rate is :math:`k_\\mathrm{des}\\Gamma\\exp(\\beta\\Gamma/(\\Gamma_\\infty-\\Gamma))`. .
    
    Args:
        surfactant_name: Name of the surfactant
        GammaInfty: Saturation surface concentration (in mol/m^2)
        beta: Frumkin parameter (in m^4 kg/(mol^2 s^2))
        k_ads: Adsorption rate (in m/s)
        k_des: Desorption rate (in 1/s)
    """ 
    def __init__(self,surfactant_name:str,GammaInfty:ExpressionOrNum,beta:ExpressionOrNum=_default_beta,k_ads:ExpressionOrNum=_default_k_ads,k_des:ExpressionOrNum=_default_k_des):
        super(FrumkinIsotherm, self).__init__(surfactant_name,k_ads,k_des)
        self.GammaInfty=GammaInfty
        self.beta=beta

    def get_surface_pressure(self) -> Expression:
        T = self.get_T_variable()
        Gamma = self.get_Gamma_variable()
        # Preventing the log to get negative numbers
        log_arg=maximum(1e-50, 1 - Gamma / self.GammaInfty)
        return -gas_constant * T * self.GammaInfty * log(log_arg)

    def get_adsorption_flux(self) -> Expression:
        C=self.get_C_variable()
        Gamma = self.get_Gamma_variable()
        # Note that k_ads is normalized here by GammaInfty (this is not always the case)
        return self.k_ads/self.GammaInfty*C*(self.GammaInfty-Gamma)

    def get_desorption_flux(self) -> Expression:
        Gamma=self.get_Gamma_variable()
        T=self.get_T_variable()
        return self.k_des*Gamma*exp(-self.beta*Gamma/(gas_constant*T))


class VanDerWaalsIsotherm(SurfactantIsotherm):
    """
    Van der Waals isotherm for surfactant adsorption.
    The surface pressure reads :math:`RT\\Gamma_\\infty/(1-\\Gamma/\\Gamma_\\infty)-\\beta\\Gamma^2/2`, the adsorption rate is :math:`k_\\mathrm{ads}C(\\Gamma-\\Gamma_\\infty)/\\Gamma_\\infty` and the desorption rate is :math:`k_\\mathrm{des}\\Gamma\\exp(\\Gamma/(\\Gamma_\\infty-\\Gamma)-\\beta\\Gamma/(RT))`.
    
    Args:
        surfactant_name: Name of the surfactant
        GammaInfty: Saturation surface concentration (in mol/m^2)
        beta: Van der Waals parameter (in m^4 kg/(mol^2 s^2))
        k_ads: Adsorption rate (in m/s)
        k_des: Desorption rate (in 1/s)
    """ 
    def __init__(self,surfactant_name:str,GammaInfty:ExpressionOrNum,beta:ExpressionOrNum=_default_beta,k_ads:ExpressionOrNum=_default_k_ads,k_des:ExpressionOrNum=_default_k_des):
        super(VanDerWaalsIsotherm, self).__init__(surfactant_name,k_ads,k_des)
        self.GammaInfty=GammaInfty
        self.beta=beta

    def get_surface_pressure(self) -> Expression:
        T = self.get_T_variable()
        Gamma = self.get_Gamma_variable()
        # Preventing the denom to become singular
        denom_arg = maximum(1e-50, 1 - Gamma / self.GammaInfty)
        return gas_constant * T * Gamma / denom_arg-self.beta/2*Gamma**2

    def get_adsorption_flux(self) -> Expression:
        C=self.get_C_variable()
        Gamma = self.get_Gamma_variable()
        # Note that k_ads is normalized here by GammaInfty (this is not always the case)
        return self.k_ads / self.GammaInfty * C * (self.GammaInfty - Gamma)

    def get_desorption_flux(self) -> Expression:
        Gamma=self.get_Gamma_variable()
        T=self.get_T_variable()
        return self.k_des*Gamma*exp(Gamma/(self.GammaInfty-Gamma) -self.beta*Gamma/(gas_constant*T))
