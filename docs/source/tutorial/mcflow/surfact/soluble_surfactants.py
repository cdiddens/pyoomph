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


from pyoomph import *

from pyoomph.materials import *  # materials
import pyoomph.materials.default_materials  # and the default materiasl

from pyoomph.expressions.units import *  # units
from pyoomph.expressions.phys_consts import gas_constant  # and the gas constant


# Register an soluble surfactant
@MaterialProperties.register()
class MySolubleSurfactant(SurfactantProperties):  # It is automatically also a pure liquid
    name = "my_soluble_surfactant"

    def __init__(self):
        super(MySolubleSurfactant, self).__init__()
        self.molar_mass = 100 * gram / mol  # required so that we can mix it with other liquids
        self.surface_diffusivity = 0.5e-9 * meter ** 2 / second  # default surface diffusivity


# Define how the liquid mixture should behave in the bulk
@MaterialProperties.register()
class MixLiquidWaterMySolubleSurfactant(MixtureLiquidProperties):
    components = {"water", "my_soluble_surfactant"}

    def __init__(self, pure_props):
        super(MixLiquidWaterMySolubleSurfactant, self).__init__(pure_props)
        # Copy the relevant properties from the water. We assume that the surfactant concentration is small
        # so that all properties are close to these of water
        self.mass_density = self.pure_properties["water"].mass_density
        self.dynamic_viscosity = self.pure_properties["water"].dynamic_viscosity
        self.default_surface_tension["gas"] = self.pure_properties["water"].default_surface_tension["gas"]
        # However, we must set a diffusivity
        self.set_diffusion_coefficient(1e-9 * meter ** 2 / second)


# Import the isotherms
from pyoomph.materials.surfactant_isotherms import *


@MaterialProperties.register()
class InterfaceWaterMySolubleSurfactantVSGas(DefaultLiquidGasInterface):
    liquid_components = {"water", "my_soluble_surfactant"}  # Water and the surfactant are in the liquid phase
    # gas_components = {"air","water"} # do not specify any particular gas phase here: Hold for all gas mixtures
    surfactants = {"my_soluble_surfactant"}  # The soluble surfactant may also be on the interface

    def __init__(self, phaseA, phaseB, surfactants):
        super(InterfaceWaterMySolubleSurfactantVSGas, self).__init__(phaseA, phaseB, surfactants)
        # Create a LangmuirIsotherm for my_soluble_surfactant
        isotherm = LangmuirIsotherm("my_soluble_surfactant", k_ads=5e-6 * meter / second, k_des=9.5 / second,
                                    GammaInfty=5 * micro * mol / meter ** 2)
        # And apply it to this interface. This will modify self.surface_tension by substracting the surface pressure
        # and furthermore it will set self.surfactant_adsorption_rate["my_soluble_surfactant"] to the total ad-/desorption flux
        isotherm.apply_on_interface(self, pure_surface_tension=self.surface_tension,min_surface_tension=20*milli*newton/meter)


if __name__ == "__main__":

    # For soluble surfactants, we also must have it in the bulk (potentially at zero concentration)
    liquid = Mixture(get_pure_liquid("water")+0.001*get_pure_liquid("my_soluble_surfactant"))
    gas = get_pure_gas("air")
    # Dict stating the initial surface concentration
    surfactants = {"my_soluble_surfactant": 1 * micro * mol / meter ** 2}

    # Getting interface properties with surfactants.
    # For a soluble surfactant, it must be present in both the liquid phase and in the surfactants dict
    # Any of them may be present at zero concentration, but it must be specified to be present at all
    interface = get_interface_properties(liquid, gas, surfactants=surfactants)

    # Getting e.g. the surface tension
    sigma = interface.surface_tension
    print(sigma)  # Rather complicated, since it depends on T and Gamma

    # First plug in by the liquid initial composition and the temperature
    sigma1 = liquid.evaluate_at_condition(sigma, liquid.initial_condition, temperature=20 * celsius)
    print(sigma1)  # Still a function of Gamma

    # Now also evaluate at the initial surfactant concentration
    sigma2 = interface.evaluate_at_initial_surfactant_concentrations(sigma1)  # plug in initial surfactant concentration
    print(sigma2.evalf())
