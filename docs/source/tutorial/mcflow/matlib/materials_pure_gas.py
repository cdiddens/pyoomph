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


from pyoomph.materials import * # Import the material API

# The following line will register this material to the material library
@MaterialProperties.register()
class PureGasAir(PureGasProperties): # Inherit from the PureGasProperties, which will set the state of matter to gas
	name="air" # We must set the name here to identify the substance
	def __init__(self):
		super().__init__() # Call the parent constructor
		self.molar_mass = 28.9645 * gram / mol # Setting the molar mass
		self.dynamic_viscosity=0.01813 *milli*pascal*second # dynamic viscosity (here a constant)
		self.mass_density=1.225*kilogram/meter**3 # Mass density

			
# Since the material is registered as pure gaseous material, we can load it as follows
air=get_pure_gas("air")

print("Dynamic viscosity:",air.dynamic_viscosity) # Printing some properties

air.mass_density=2*kilogram/meter**3 # Changing the properties by hand

air2=get_pure_gas("air") # Loading another instance of air
print("Mass densities",air.mass_density,air2.mass_density) # Compare the densities


###############################################
### Consider temperature and pressure effects #
###############################################


# Load the universal gas constant
from pyoomph.expressions.phys_consts import gas_constant
        
# Redefine our gas. Since a gas with name "gas" is already registered, we must pass override=True here
@MaterialProperties.register(override=True)
class PureGasAir(PureGasProperties):
	name="air"
	def __init__(self):
		super().__init__()
		self.molar_mass = 28.9645 * gram / mol # Same as before, always a constant

		# Use ideal gas law to get the density
		self.mass_density=var("absolute_pressure")*self.molar_mass/ (gas_constant * var("temperature"))
		# Alternatively, we can just call self.set_mass_density_from_ideal_gas_law() for that

		TKelvin = var("temperature") / kelvin # bind the temperature in kelvin for the following fit expressions
		# Varying dynamic viscosity
		self.dynamic_viscosity=(0.0409424 + 0.00725803 * TKelvin - 4.12727e-06 * (TKelvin) ** 2)* 1e-5 * pascal * second 
		
		# For thermal problems, also the following must be set. If only isothermal problems are considered, it is not required
		self.thermal_conductivity=(-0.0217506 + 0.00984373 * TKelvin - 3.4318e-06 * TKelvin * TKelvin)*1e-5 * kilo * watt / (meter * kelvin)
		self.specific_heat_capacity=1.005* kilo * joule / (kilogram * kelvin)

		
air=get_pure_gas("air") # Load the new definition of "air"
print("VARIABLE DENSITY",air.mass_density) # Print the functional expression rho(p,T)

# Evaluate at a particular condition
rho_std=air.evaluate_at_condition("mass_density",temperature=18*celsius,absolute_pressure=1*atm) 
print("EVALUATED DENSITY",rho_std)
# To convert to a float (e.g. to write to a file), we just have to cancel out the desired unit and call float(...)
print("EVALUATED DENSITY in (kg/m**3)",float(rho_std/(kilogram/meter**3)))

# Loop over Celsius values
for T_in_Celsius in [10,15,20,25,30]:
	# Evaluate at 1 atm and this temperature
	rho_at_T=air.evaluate_at_condition(air.mass_density,temperature=T_in_Celsius*celsius,absolute_pressure=1*atm)
	print(f"DENSITY AT T[C]= {T_in_Celsius} is {float(rho_at_T/(kilogram/meter**3))} kg/m^3") #Print it (can also write to file)
	
	


