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


# Register an insoluble surfactant
@MaterialProperties.register()
class MyInsolubleSurfactant(SurfactantProperties):
    name = "my_insoluble_surfactant"

    def __init__(self):
        super(MyInsolubleSurfactant, self).__init__()
        self.surface_diffusivity = 1e-9 * meter ** 2 / second  # default surface diffusivity


# Register the interface properties of a water liquid phase in contact with a gaseous phase
# with the surfactant "my_insoluble_surfactant" on the interface
# It is best to inherit from the DefaultLiquidGasInterface to setup all properties to reasonable defaults
@MaterialProperties.register()
class InterfaceWaterVsVaporAirWithMyInsolubleSurfactant(DefaultLiquidGasInterface):
    liquid_components = {"water"} # Pure water must be the liquid phase
    # If we uncomment this, it will only be used if the gas phase consist of air and water vapor
    # If not set, it is valid for arbitrary gas phases
    #       gas_components = {"air","water"}
    surfactants = {"my_insoluble_surfactant"} # This surfactant must be present

    def __init__(self, phaseA, phaseB, surfactants):
        super(InterfaceWaterVsVaporAirWithMyInsolubleSurfactant, self).__init__(phaseA, phaseB, surfactants)
        # set the surface tension sigma(Gamma)=sigma_0 - R*T*Gamma
        Gamma = var("surfconc_my_insoluble_surfactant")  # surface concentration Gamma of the surfactant "my_surfactant"
        T = var("temperature")
        self.surface_tension = self.surface_tension - gas_constant * T * Gamma
        # We could also modify the surface diffusivity for this particular interface
        # self.set_surface_diffusivity("my_surfactant",1e-10*meter**2/second)


if __name__=="__main__":

	liquid = get_pure_liquid("water")
	gas = get_pure_gas("air")
	surfactants = {"my_insoluble_surfactant": 1 * micro * mol / meter ** 2} # Dict stating the initial concentration
	# alternatively, load the surfactant:
	#       my_surfactant=get_surfactant("my_insoluble_surfactant")
	#       surfactants = {my_surfactant: 1 * micro * mol / meter ** 2} #

	# Gettting interface properties with surfactants
	interface = get_interface_properties(liquid, gas, surfactants=surfactants)

	# Getting e.g. the surface tension
	sigma=interface.surface_tension
	print(sigma) # Rather complicated, since it depends on T and Gamma

	# First plug in by the liquid initial composition and the temperature
	sigma1=liquid.evaluate_at_condition(sigma,liquid.initial_condition,temperature=20*celsius)
	print(sigma1) # Still a function of Gamma

	# Now also evaluate at the initial surfactant concentration
	sigma2=interface.evaluate_at_initial_surfactant_concentrations(sigma1) # plug in initial surfactant concentration
	print(sigma2)

