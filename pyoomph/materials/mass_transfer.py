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
 
from ..expressions import Union
from ..expressions.units import Union
from .generic import MixtureGasProperties, MixtureLiquidProperties, PureGasProperties, PureLiquidProperties
from ..typings import Union
from ..generic import InterfaceEquations
from ..expressions import * #Import grad et al
from ..expressions.units import *

from ..generic.codegen import FiniteElementCodeGenerator
from ..typings import *

if TYPE_CHECKING:
    from .generic import MaterialProperties,BaseInterfaceProperties,PureLiquidProperties,MixtureLiquidProperties,PureGasProperties,MixtureGasProperties


class MassTransferModelBase:
    """
    Base class for mass transfer models
    """
    def __init__(self):
        self._inside_domain=None
        self._outside_domain=None
        self._interface=None
        self._opposite_interface=None
        self.interface_props=None
        self._for_lubrication=False
        pass


    def get_mass_transfer_name(self,name:str) -> str:
        return "masstrans_"+name

    def _setup_for_code(self,interf:FiniteElementCodeGenerator,interface_props:"BaseInterfaceProperties",for_lubrication:Union[Literal[False],Dict[str,ExpressionOrNum]]=False):
        self._interface=interf
        self._inside_domain=interf.get_parent_domain()
        self._opposite_interface=cast(FiniteElementCodeGenerator,interf._get_opposite_interface())
        if self._opposite_interface is not None:
            self._outside_domain=self._opposite_interface.get_parent_domain()
        self.interface_props=interface_props
        self._for_lubrication=for_lubrication

    @abstractmethod
    def setup_scaling(self,ieqs:InterfaceEquations):
        pass

    @abstractmethod
    def define_fields(self,ieqs:InterfaceEquations):
        pass

    @abstractmethod
    def define_residuals(self,ieqs:InterfaceEquations):
        pass

    def _clean_up_for_code(self):
        self._interface = None
        self._inside_domain = None
        self._opposite_interface=None
        self._outside_domain = None
        self._for_lubrication=False

    @abstractmethod
    def identify_transfer_components(self)->Set[str]:
        raise NotImplementedError("identify_volatile_components")
        #return set()

    @abstractmethod
    def get_mass_transfer_rate_of(self, name:str)->Expression:
        raise NotImplementedError("get_mass_transfer_rate_of")
        #return 0

    def get_all_masstransfer_rates(self)->Dict[str,Expression]:
        vc=self.identify_transfer_components()
        return {n:self.get_mass_transfer_rate_of(n) for n in vc}

    def get_latent_heat_flux(self)->Expression:
        raise NotImplementedError("get_latent_heat_flux")
    
    def get_vapor_recoil_pressure(self)->Expression:
        return Expression(0)


# This model will project the mass transfer rates on fields
class ProjectedMassTransferModelBase(MassTransferModelBase):
    def __init__(self):
        super(ProjectedMassTransferModelBase, self).__init__()
        self.rates_as_fields:bool = True # If set, we project the transfer rates
        self.projection_space:Optional[FiniteElementSpaceEnum]=None
        self.test_scale:Optional[ExpressionOrNum]=1 / scale_factor("mass_transfer_rate")

    def get_mass_transfer_space(self,name:str,ieqs:InterfaceEquations) -> FiniteElementSpaceEnum:
        if self.projection_space is None:
            opp_pdom=ieqs.get_opposite_side_of_interface().get_parent_domain()
            pdom=ieqs.get_parent_domain()
            assert opp_pdom is not None
            space=opp_pdom.get_space_of_field("massfrac_"+name)
            if space=="":
                for c in self.props_outside.components:
                    space=opp_pdom.get_space_of_field("massfrac_"+c)
                    if space!="":
                        break
            if space=="":
                space=pdom.get_space_of_field("massfrac_"+name)
            if space=="":
                for c in self.props_inside.components:
                    space=pdom.get_space_of_field("massfrac_"+c)
                    if space!="":
                        break                    
            if space=="":
                raise RuntimeError("Cannot find a space for the field "+name+". Please set the projection_space attribute.")
            space=cast(FiniteElementSpaceEnum,space)
            return space
        else:
            return self.projection_space

    def get_masstransfer_definition(self,name:str)->Expression:
        raise NotImplementedError("Implement the definition of the mass transfer rate here")

    def get_mass_transfer_rate_of(self,name:str)->Expression:
        if self.rates_as_fields:
            return var(self.get_mass_transfer_name(name))
        else:
            return self.get_masstransfer_definition(name)
        
    

    def define_fields(self, ieqs:InterfaceEquations):
        if self.rates_as_fields:
            comps = self.identify_transfer_components()
            for ec in comps:
                ieqs.define_scalar_field(self.get_mass_transfer_name(ec), self.get_mass_transfer_space(ec,ieqs))

    def define_residuals(self, ieqs:InterfaceEquations):
        if self.rates_as_fields:
            comps = self.identify_transfer_components()
            for ec in comps:
                rhs = self.get_masstransfer_definition(ec)
                j, jtest = var_and_test(self.get_mass_transfer_name(ec))
                ieqs.add_residual(weak(j - rhs, jtest))

    def setup_scaling(self, ieqs:InterfaceEquations):
        if self.rates_as_fields:
            comps = self.identify_transfer_components()
            kwargs = {}
            tkwargs = {}
            for ec in comps:
                kwargs[self.get_mass_transfer_name(ec)] = "mass_transfer_rate"
                if self.test_scale is not None:
                    tkwargs[self.get_mass_transfer_name(ec)] = self.test_scale
            ieqs.set_scaling(**kwargs)
            ieqs.set_test_scaling(**tkwargs)




class PrescribedMassTransfer(ProjectedMassTransferModelBase):
    def __init__(self,**rates:ExpressionOrNum):
        super(PrescribedMassTransfer, self).__init__()
        self.rates=rates.copy()

    def identify_transfer_components(self) -> Set[str]:
        return set(self.rates.keys())

    def get_masstransfer_definition(self,name:str)->Expression:
        if name in self.rates.keys():
            rate=self.rates[name]
            if isinstance(rate,Expression):
                return rate
            else:
                return Expression(rate)
        else:
            return Expression(0)


class FluidPropMassTransferModel(ProjectedMassTransferModelBase):
    def __init__(self,props_inside:"MaterialProperties",props_outside:"MaterialProperties"):
        super(FluidPropMassTransferModel, self).__init__()
        self.props_inside=props_inside
        self.props_outside=props_outside
         # For terms as in Prosperetti, Plesset, Phys. Fluids 27(7), (1984)
         # 1/2*{[(u_l-u_inter)*n]**2-[(u_g-u_inter)*n]**2}=J**3*(1/rho_g**2-1/rho_l**2)/2 in the heat equation
         # And the vapor recoil pressure in the dynamic condition
        self.with_prosperetti_term=False


    def get_vapor_recoil_pressure(self)->Expression:
        res=super(FluidPropMassTransferModel, self).get_vapor_recoil_pressure()
        if self.with_prosperetti_term:
            js=self.get_all_masstransfer_rates()
            assert self.interface_props is not None            
            JTotal=Expression(0)
            for _name,j in js.items():
                JTotal+=j
            rho_inside=self.props_inside.mass_density
            rho_outside=evaluate_in_domain(self.props_outside.mass_density,domain=self._opposite_interface)
            res=JTotal**2*(1/rho_outside-1/rho_inside)                
        return res

    def identify_transfer_components(self) -> Set[str]:
        if self.props_outside.is_pure:
            couter={self.props_outside.name}
        else:
            couter=self.props_outside.components
        if self.props_inside.is_pure:
            cinner={self.props_inside.name}
        else:
            cinner=self.props_inside.components
        cboth=cinner.intersection(couter)
        return cboth

    def get_latent_heat_flux(self)->Expression:
        js=self.get_all_masstransfer_rates()
        q=Expression(0)
        assert self.interface_props is not None
        # For terms as in Prosperetti, Plesset, Phys. Fluids 27(7), (1984)
        JTotal=Expression(0)
        for name,j in js.items():
            Lambda=self.interface_props.get_latent_heat_of(name)
            q+=Lambda*j
            JTotal+=j
        if self.with_prosperetti_term:
            rho_inside=self.props_inside.mass_density
            rho_outside=evaluate_in_domain(self.props_outside.mass_density,domain=self._opposite_interface)
            prosperetti_term=JTotal**3*(1/rho_outside**2-1/rho_inside**2)/2 
            q+=prosperetti_term
        return q



class DifferenceDrivenMassTransferModel(FluidPropMassTransferModel):
    """
    A mass transfer model that is driven by the difference between the equilibrium composition and the actual composition.
    This difference is multiplied by a mass transfer coefficient to get the mass transfer rate.

    Args:
        FluidPropMassTransferModel (_type_): _description_
    """
    def __init__(self,props_inside:"MaterialProperties",props_outside:"MaterialProperties"):
        super(DifferenceDrivenMassTransferModel, self).__init__(props_inside,props_outside)
        #: The factor of proportionality between the driving force and the mass transfer rate.
        self.default_mass_flux_coefficient=100*kilogram/(meter**2*second)

    def get_mass_flux_coeff_for(self,name:str)->Expression:
        return self.default_mass_flux_coefficient

    @abstractmethod
    def get_driving_nondimensional_difference_for(self,name:str)->Expression:
        raise RuntimeError("Please override specifically")

    def get_masstransfer_definition(self,name:str)->Expression:
        return self.get_mass_flux_coeff_for(name) * self.get_driving_nondimensional_difference_for(name)



class LagrangeMultiplierMassTransferModel(FluidPropMassTransferModel):
    def __init__(self,props_inside:"MaterialProperties",props_outside:"MaterialProperties"):
        super(LagrangeMultiplierMassTransferModel, self).__init__(props_inside,props_outside)
        self.test_scale:Optional[ExpressionOrNum]=None

    def define_fields(self,ieqs:InterfaceEquations):
        evaps=self.identify_transfer_components()
        for ec in evaps:
            ieqs.define_scalar_field(self.get_mass_transfer_name(ec),self.get_mass_transfer_space(ec,ieqs))

    @abstractmethod
    def get_equilibrium_expression(self,name:str)->Expression:
        raise RuntimeError("IMPLEMENT!")
        return curr-des #Must be overriden: Returns (current-desired)

    def define_residuals(self,ieqs:InterfaceEquations):
        evaps = self.identify_transfer_components()
        for ec in evaps:
            eq=self.get_equilibrium_expression(ec)
            _,ltest=var_and_test(self.get_mass_transfer_name(ec))
            ieqs.add_residual(weak(eq,ltest)) #*(self.test_scale if self.test_scale is not None else 1)


    def get_mass_transfer_rate_of(self, name:str) -> Expression:
        return var("masstrans_"+name)

class LagrangeMultiplierMassTransferModelLiquidGas(LagrangeMultiplierMassTransferModel):
    def __init__(self,props_inside:Union["PureLiquidProperties","MixtureLiquidProperties"],props_outside:Union["PureGasProperties","MixtureGasProperties"]):
        super(LagrangeMultiplierMassTransferModelLiquidGas, self).__init__(props_inside,props_outside)        
        if props_inside.state_of_matter!="liquid":
            raise RuntimeError("This mass transfer model only works for liquids as inner phase")
        if props_outside.state_of_matter!="gas":
            raise RuntimeError("This mass transfer model only works for gases as outer phase")
        self.props_inside=cast(Union["PureLiquidProperties","MixtureLiquidProperties"],self.props_inside)
        self.props_outside=cast(Union["PureLiquidProperties","MixtureLiquidProperties"],self.props_outside)


    def get_mass_transfer_space(self, name:str,ieqs:InterfaceEquations) -> FiniteElementSpaceEnum:
        opp_pdom=ieqs.get_opposite_side_of_interface().get_parent_domain()
        assert opp_pdom is not None
        space=opp_pdom.get_space_of_field("massfrac_"+name)
        if space=="":
            for c in self.props_outside.components:
               space=opp_pdom.get_space_of_field("massfrac_"+c)
               if space!="":
                   break
        if space=="":
            # This actually happens for a case with only one component. Damn
            space=ieqs.get_opposite_side_of_interface().get_parent_domain()._coordinate_space
            if space=="":
                raise RuntimeError("What??")
        space=cast(FiniteElementSpaceEnum,space)
        return space

    def identify_transfer_components(self) -> Set[str]:
        possible=super(LagrangeMultiplierMassTransferModelLiquidGas, self).identify_transfer_components()
        for n in possible:
            if self.props_inside.get_vapor_pressure_for(n) is None:
                print("Cannot find any vapor pressure for "+n+". Hence, the component will be non-volatile.")
                possible.remove(n)
        return possible

    def get_equilibrium_expression(self,name:str) -> Expression:
        opp = self._opposite_interface
        if opp is None:
            return Expression(0)  # No mass transfer if there is no gas phase!
        psat = self.props_inside.get_vapor_pressure_for(name)
        if psat is None:
            return Expression(0)
        elif isinstance(psat,(float,int)):
            psat=Expression(psat)
        ptot = var("absolute_pressure", domain=opp)
        xVapDesired = psat / ptot
        xVapPresent = var("molefrac_" + name, domain=opp)
        #xVapPresent = var("massfrac_" + name, domain=opp)
        nddoff=(xVapPresent-xVapDesired)#*scale_factor("mass_transfer_rate")
        return nddoff


# Std model with a transfer rate
class DifferenceDrivenMassTransferModelLiquidGas(DifferenceDrivenMassTransferModel):
    """
    The standard mass transfer model for liquid-gas interface. We enforce Raoult's law by a rate-limited process.
    The driving force is the difference between the equilibrium mole fractions and the actual mole fractions in the gas phase.
    This difference is multiplier by a factor to get the mass transfer rate. If this factor is large, it behvaes almost like a Dirichlet condition, i.e. it will be diffusion-limited mass transfer.

    Args:
        props_inside: The properties of the liquid phase
        props_outside: The properties of the gas phase        
    """
    def __init__(self,props_inside:Union["PureLiquidProperties","MixtureLiquidProperties"],props_outside:Union["PureGasProperties","MixtureGasProperties"]):
        super(DifferenceDrivenMassTransferModelLiquidGas, self).__init__(props_inside,props_outside)
        if props_inside.state_of_matter!="liquid":
            raise RuntimeError("This mass transfer model only works for liquids as inner phase")
        if props_outside.state_of_matter!="gas":
            raise RuntimeError("This mass transfer model only works for gases as outer phase")
        self.props_inside=cast(Union["PureLiquidProperties","MixtureLiquidProperties"],self.props_inside)
        self.props_outside=cast(Union["PureLiquidProperties","MixtureLiquidProperties"],self.props_outside)

    def identify_transfer_components(self) -> Set[str]:
        possible=super(DifferenceDrivenMassTransferModelLiquidGas, self).identify_transfer_components()
        res:Set[str]=set()
        for n in possible:
            if self.props_inside.get_vapor_pressure_for(n) is None:
                print("Cannot find any vapor pressure for "+n+". Hence, the component will be non-volatile.")
            else:
                res.add(n)
        return res

    def get_driving_nondimensional_difference_for(self,name:str)->Expression:
        if self._for_lubrication:
            self._for_lubrication=cast(Dict[str,ExpressionOrNum],self._for_lubrication)
            gasbulk=self._inside_domain
            psat = self.props_inside.get_vapor_pressure_for(name)
            if psat is None:
                return Expression(0)
            elif isinstance(psat,(float,int)):
                psat=Expression(psat)
            ptot = var("absolute_pressure", domain=gasbulk)
            xVapDesired = psat / ptot
            xVapPresent = var("molefrac_" + name, domain=gasbulk)
            cutoff=self._for_lubrication["cutoff"]
            if cutoff is None:
                cutoff=1
            if not isinstance(cutoff,Expression):
                cutoff=Expression(cutoff)
            return (xVapDesired-xVapPresent)*cutoff
        else:
            opp=self._opposite_interface
            if opp is None:
                return Expression(0) #No mass transfer if there is no gas phase!
            psat=self.props_inside.get_vapor_pressure_for(name)
            if psat is None:
                return Expression(0)
            elif isinstance(psat,(float,int)):
                psat=Expression(psat)
            ptot=var("absolute_pressure",domain=opp)
            xVapDesired=psat/ptot
            xVapPresent=var("molefrac_"+name,domain=opp)
            return xVapDesired-xVapPresent


class HertzKnudsenSchrageMassTransferModel(DifferenceDrivenMassTransferModelLiquidGas):
    def __init__(self, props_inside: Union[PureLiquidProperties, MixtureLiquidProperties], props_outside: Union[PureGasProperties, MixtureGasProperties]):
        super().__init__(props_inside, props_outside)
        self.sticking_coefficient:Union[ExpressionOrNum,Dict[str,ExpressionOrNum]]=0.1

    def get_mass_flux_coeff_for(self,name:str)->Expression:
        from ..expressions.phys_consts import gas_constant
        pc=self.props_inside.get_pure_component(name)
        M=pc.molar_mass
        T=var("temperature")        
        R=gas_constant
        opp=self._opposite_interface
        ptot=var("absolute_pressure",domain=opp)
        kin_gas_factor=ptot*square_root(M/(2*pi*R*T))
        if isinstance(self.sticking_coefficient,(float,int,Expression,GlobalParameter)):
            return self.sticking_coefficient*kin_gas_factor
        elif isinstance(self.sticking_coefficient,dict):
            if name not in self.sticking_coefficient.keys():
                raise RuntimeError("Must set the sticking coefficient of "+str(name))
            return self.sticking_coefficient[name]*kin_gas_factor
        else:
            raise RuntimeError("sticking_coefficient must be either an expression or a dict mapping component names to expressions")

class LLEMassTransferModel(DifferenceDrivenMassTransferModel):
    def __init__(self, props_inside: MixtureLiquidProperties, props_outside: MixtureLiquidProperties,*,unifac_model:Optional[str]=None,FD_epsilon:Optional[float]=1e-9,mass_transfer_factor:Optional[ExpressionNumOrNone]=None, use_log_approach: bool = False, reference_molar_mass:ExpressionNumOrNone=None):
        super().__init__(props_inside, props_outside)
        from .activity import UNIFACMultiReturnExpression
        if self.props_inside.components != self.props_inside.components:
            raise RuntimeError("Only works for the same components inside and outside")
        if unifac_model is None:
            if self.props_inside._unifac_model is None:
                raise RuntimeError("No UNIFAC model specified in the mixture")
            unifac_model=self.props_inside._unifac_model
        if FD_epsilon is None:
            self.activity_calc=None
            assert isinstance(self.props_inside, MixtureLiquidProperties)
            self.activity_table=self.props_inside.activity_coefficients
        else:
            self.activity_calc=UNIFACMultiReturnExpression(self.props_inside,unifac_model,FD_epsilon=FD_epsilon)
        if mass_transfer_factor is not None:
            self.default_mass_flux_coefficient=mass_transfer_factor
        self.use_log_approach=use_log_approach
        self.reference_molar_mass=reference_molar_mass

    def get_driving_nondimensional_difference_for(self, name: str) -> Expression:
        if self.activity_calc is None:
            gammaI=0+self.activity_table[name]
            xI=var("molefrac_"+name)
            gammaO=evaluate_in_domain(0+self.activity_table[name],domain="|.")
            xO=var("molefrac_"+name,domain="|.")
            muI=log(xI*gammaI) if self.use_log_approach else xI*gammaI
            muO=log(xO*gammaO) if self.use_log_approach else xO*gammaO
        else:
            muI=log(var("molefrac_"+name)*self.activity_calc.get_activity_coefficient(name)) if self.use_log_approach else var("molefrac_"+name)*self.activity_calc.get_activity_coefficient(name)
            muO=log(var("molefrac_"+name,domain="|.")*self.activity_calc.get_activity_coefficient(name,domain="|.")) if self.use_log_approach else var("molefrac_"+name,domain="|.")*self.activity_calc.get_activity_coefficient(name,domain="|.")
        res= (muI-muO)
        if self.reference_molar_mass is not None:
            res*=self.reference_molar_mass/self.props_inside.get_pure_component(name).molar_mass
        return res
    
    def get_mass_flux_coeff_for(self, name: str) -> Expression:
        return super().get_mass_flux_coeff_for(name)


StandardMassTransferModelLiquidGas=DifferenceDrivenMassTransferModelLiquidGas
