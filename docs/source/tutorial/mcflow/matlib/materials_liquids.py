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


from pyoomph.materials import *

# Define pure water
@MaterialProperties.register()
class PureLiquidWater(PureLiquidProperties):
	name="water"
	def __init__(self):
		super().__init__()
		self.molar_mass=18.01528*gram/mol # Molar mass 

		# Density and viscosity (assuming constants here)
		self.mass_density=998*kilogram/meter**3
		self.dynamic_viscosity=1*milli* pascal * second

		# Thermal properties (assuming constants here)
		self.specific_heat_capacity=4.187* kilo * joule / (kilogram * kelvin)
		self.thermal_conductivity=0.597* watt / (meter * kelvin)
		self.latent_heat_of_evaporation=2437.69081321*kilo*joule/kilogram # Liquids also have a latent heat of evaporation
		
		# Default surface tension against air as function of the temperature
		TKelvin=var("temperature")/kelvin
		self.default_surface_tension["gas"]=0.07275*(1.0-0.002*(TKelvin-291.0)) * newton/meter

		# Vapor pressure can be set by Antoine coefficients (in mmHg, C convention)
		# You can also add e.g. bar and kelvin as fourth and fifth argument to use the [bar,K] convention
		self.set_vapor_pressure_by_Antoine_coeffs(8.07131,1730.63 ,233.426)
		#Alternatively, you can set the vapor pressure by hand by setting self.vapor_pressure= ...

		#For UNIFAC calculations of activity coefficients in mixtures, we also need the UNIFAC groups
		self.set_unifac_groups({"H2O":1}) #Just one H2O group here



@MaterialProperties.register()
class PureLiquidGlycerol(PureLiquidProperties):
	name="glycerol"
	def __init__(self):
		super().__init__()
		self.molar_mass=92.09382*gram/mol
		self.dynamic_viscosity=1*pascal*second	# For simplicity, just constants here
		self.mass_density=1260*kilogram/meter**3
		self.default_surface_tension["gas"]=64*milli*newton/meter

		TCelsius=var("temperature")/kelvin-273.15
		self.thermal_conductivity=(0.289500000000009+0.000103999999999881*TCelsius)*watt/(meter*kelvin)
		self.specific_heat_capacity=2.43*kilo*joule/(kilogram*kelvin)

		#Different UNIFAC models have different decompositions of glycerol
		self.set_unifac_groups({"CH2":2,"CH1":1,"OH":3},only_for="Original")
		self.set_unifac_groups({"CH2": 2, "CH1": 1, "OH (P)": 2, "OH (S)": 1}, only_for="Dortmund")
		self.set_unifac_groups({"CH2(hydroxy)": 2, "CH(hydroxy)": 1, "OH(new)": 3}, only_for="AIOMFAC")




@MaterialProperties.register()
class MixtureLiquidGlycerolWater(MixtureLiquidProperties):
	components={"water","glycerol"}
	passive_field="water"
	def __init__(self,pure_properties):
		super().__init__(pure_properties)
		self.set_by_weighted_average("mass_density") # realistic assumption here: rho=rho_water*w_water+rho_glyc*w_glyc
		self.set_by_weighted_average("thermal_conductivity")
		self.set_by_weighted_average("specific_heat_capacity")
		
		yG=self.get_mass_fraction_field("glycerol") # will just expand to var("massfrac_glycerol")

		# Model for the dynamic viscosity
		TCelsius = subexpression(var("temperature") / kelvin-273.15)
		a=0.705 - 0.0017 * TCelsius
		b = (4.9 + 0.036 * TCelsius) * a ** 2.5
		muG=12100 * exp((-1233 + TCelsius) * TCelsius / (9900 + 70 * TCelsius))
		muW =1.790 * exp((-1230 - TCelsius) * TCelsius / (36100 + 360 * TCelsius))
		alpha = subexpression(1 - yG + a * b * yG * (1 - yG) / (a * yG + b * (1 - yG)))
		self.dynamic_viscosity= subexpression(muW* (muG/muW) ** (1-alpha)* 0.001*pascal * second)

		# Surface tension function
		self.default_surface_tension["gas"]=subexpression(72.45e-3 * ((1.0 - 0.1214690683 * yG + 0.4874796412 * yG ** 2 - 2.208295376 * yG ** 3 + 3.412242927 * yG ** 4 - 1.698619738 * yG ** 5) - (0.0001455 * (1 - yG) + 0.00008845 * yG) * (TCelsius))* newton / meter)

		# Diffusion coefficient fit
		D=1.024e-11 * (-0.721 * yG + 0.7368) / (0.49311e-2 * yG + 0.7368e-2)*meter ** 2 / second
		self.set_diffusion_coefficient(D)

		# Set activity coefficients by AIOMFAC
		self.set_activity_coefficients_by_unifac("AIOMFAC")
