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


# Pure air: see before (we took the simple constant expressions here, but it also works for expressions depending on pressure and temperature)
@MaterialProperties.register()
class PureGasAir(PureGasProperties): 
	name="air"
	def __init__(self):
		super().__init__()
		self.molar_mass = 28.9645 * gram / mol
		self.dynamic_viscosity=0.01813 *milli*pascal*second 
		self.mass_density=1.225*kilogram/meter**3
	

# Create the pure gaseous water
@MaterialProperties.register()
class PureGasWater(PureGasProperties):
	name="water" # name it "water"
	def __init__(self):
		super().__init__()
		self.molar_mass = 18.01528*gram/mol # Molar mass is important to convert to e.g. mole fractions
		
		# If only used in mixtures, we do not require the mass density and dynamic viscosity of the pure substance here



# Create mixture properties that will apply if you mix gaseous water with gaseous air
@MaterialProperties.register()
class MixtureGasWaterAir(MixtureGasProperties): # MixtureGasProperties is the base class for gas mixtures
	components={"water","air"} # This class applied when mixing "water" and "air"
	# We can specify a passive component: We have to solve n-1 advection diffusion equations for the mass fractions
	# The nth follows from 1 minus the others. We can select, which component is not explicitly solved for
	passive_field="air"  # we choose air here

	# The constructor gets the pure properties as a dict {"water":PureGasWater instance, "air":PureGasAir instance}
	def __init__(self,pure_properties): 
		super().__init__(pure_properties) # pass it to the parent constuctor
		
		self.set_mass_density_from_ideal_gas_law() # Density from ideal gas law also works for mixtures
		self.dynamic_viscosity=self.pure_properties["air"].dynamic_viscosity # Just take the dynamic viscosity from the air

		# In a binary mixture, it is sufficient to specify a single diffusion coefficent
		# This may of course also be a function fo the composition, temperature and pressure
		self.set_diffusion_coefficient(2.42e-5*meter**2/second)
		

# Get the pure properties
air=get_pure_gas("air")
water_vapor=get_pure_gas("water")

# Mix in terms of mass fraction. One quantifier (here 0.98 for air) can be omitted
mix_gas=Mixture(air+0.02*water_vapor)

# We can access the initial condition, which will result in {'massfrac_air': 0.98, 'massfrac_water': 0.02, 'temperature': None}
print(mix_gas.initial_condition)

# To evaluate e.g. the mass density at the initial condition, we can just pass the initial condition, but we also have to add information on the pressure and temperature to get a single value
print(mix_gas.evaluate_at_condition("mass_density",mix_gas.initial_condition,temperature=20*celsius,absolute_pressure=1*atm))



# Create the pure gaseous ethanol (analogous to water)
@MaterialProperties.register()
class PureGasEthanol(PureGasProperties):
	name="ethanol" 
	def __init__(self):
		super().__init__()
		self.molar_mass = 0.4607E-01*kilogram/mol
		# Again we skip any further definitions



# Defining ternary mixture properties is similar to binary mixtures:
@MaterialProperties.register()
class MixtureGasWaterAirEthanol(MixtureGasProperties):
	components={"ethanol","water","air"} # Now three components
	passive_field="air"  # we choose again air as passive field

	def __init__(self,pure_properties): 
		super().__init__(pure_properties) # Pure properties now has three entries
		
		self.set_mass_density_from_ideal_gas_law() # Again assuming ideal gas law
		
		# However, we now want to (artificially) increase the viscosity slightly with the mass fraction of ethanol:
		mu_air=self.pure_properties["air"].dynamic_viscosity # Get the viscosity of pure air
		massfrac_ethanol=var("massfrac_ethanol")  # Get the variable ethanol mass fraction
		self.dynamic_viscosity=mu_air*(1+0.2*massfrac_ethanol) # With increasing ethanol, the gas gets more viscous

		# We now have three components, so effectively have a 2x2 diffusion matrix. We only assume diagonal terms:
		self.set_diffusion_coefficient("water",2.42e-5*meter**2/second)
		self.set_diffusion_coefficient("ethanol",1.35e-5* meter**2/second)
	

# Also get the pure ethanol vapor
ethanol_vapor=get_pure_gas("ethanol")
mix_gas=Mixture(air+0.02*water_vapor+0.06*ethanol_vapor) # Mix it in terms of mass fractions


