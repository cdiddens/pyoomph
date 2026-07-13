Properties as function of temperature and pressure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As already mentioned, using constant values for the properties is not very useful, as these constants are only valid within a small temperature and pressure range. Furthermore, any effects as e.g. natural convection or Marangoni flow require that properties change with e.g. the temperature or the pressure. For a pure substance, only the pressure and the temperature can influence the physical properties, so we will focus on these first. The (local) temperature can be accessed with ``var("temperature")``, whereas the pressure is obtained by ``var("absolute_pressure")``. As mentioned before, ``var("pressure")`` is usually used for the pressure in (Navier-)Stokes problems, which can be quite different, e.g. zero or even negative. ``var("absolute_pressure")`` is used for the thermodynamic pressure, which must be positive.

We can e.g. use these variables to utilize the ideal gas law for the mass density, :math:`\rho=pM/(RT)`, by defining our air as follows

.. code:: python

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

Obviously, we can just use the dimensional variables ``var("temperature")`` and ``var("absolute_pressure")`` to directly calculate the mass density by the ideal gas law, which is now valid for all temperatures and pressures (as long as air can be considered as an ideal gas). There is a method, :py:meth:`~pyoomph.materials.generic.PureGasProperties.set_mass_density_from_ideal_gas_law`, which does exactly this as a shortcut. For other properties, e.g. the dynamic viscosity, we use fitted data from experimental results. The fitted expression for the viscosity used here reads

.. math:: \mu[10^{-5}\:\mathrm{Pa} \cdot \mathrm{s}]=0.049424+0.00724803 T[\:\mathrm{K}]-4.12727{\times}10^{-6} T^2[\:\mathrm{K}^2]

To use the temperature as numeric value in :math:`\:\mathrm{K}`, we can just divide ``var("temperature")/kelvin``. In the same way, we just multiply the fit result by :math:`10^{-5}\:\mathrm{Pa} \cdot \mathrm{s}` to cast it into the unit required for the viscosity according to this fit.

When the temperature is solved, one usually also requires the thermal conductivity and the specific heat capacity, which are set to the properties :py:attr:`~pyoomph.materials.generic.MaterialProperties.thermal_conductivity` and :py:attr:`~pyoomph.materials.generic.MaterialProperties.specific_heat_capacity`. Note that the latter requires the heat capacity per mass, not per mole, but these can be easily converted using the molar mass.

Of course, these expressions are more complicated than a simple constant and it might cause additional computational time to consider the variations of the properties with the temperature and pressure (and fluid composition/surfactant concentration for mixtures). However, if we solve e.g. an isothermal and isobaric problem, pyoomph only evaluates these expressions at the selected temperature and pressure once, namely when generating the C code. Hence, in that case, one eventually ends up using constants again.

To evaluate a property at particular conditions, we can do the following:

.. code:: python

   air=get_pure_gas("air") # Load the new definition of "air"
   print("VARIABLE DENSITY",air.mass_density) # Print the functional expression rho(p,T)

   # Evaluate at a particular condition
   rho_std=air.evaluate_at_condition("mass_density",temperature=18*celsius,absolute_pressure=1*atm) 
   print("EVALUATED DENSITY",rho_std)
   # To convert to a float (e.g. to write to a file), we just have to cancel out the desired unit and call float(...)
   print("EVALUATED DENSITY in (kg/m**3)",float(rho_std/(kilogram/meter**3)))

Note that :py:meth:`~pyoomph.materials.generic.MaterialProperties.evaluate_at_condition` can also take the expression ``air.mass_density`` (or ``air.dynamic_viscosity``) instead of the string ``"mass_density"`` (or ``"dynamic_viscosity"``). To cast a constant dimensional value into a ``float``, you just cancel out the desired unit (e.g. ``kilogram/meter**3``) and wrap it into a ``float()`` call. Of course, if we divide by e.g. ``gram/(centi*meter)**3``, we get the numerical float value in :math:`\:\mathrm{g}/\mathrm{m}^3` instead. You cannot convert a dimensional quantity into a float without dividing by the desired unit first and likewise, you cannot cast expressions to a float that still depend on the temperature or the pressure, i.e. having terms containing ``var("...")`` in it.

However, you can easily make a table of float values by scanning ranges of the temperature or the absolute pressure, e.g. by:

.. code:: python

   # Loop over Celsius values
   for T_in_Celsius in [10,15,20,25,30]:
   	# Evaluate at 1 atm and this temperature
   	rho_at_T=air.evaluate_at_condition(air.mass_density,temperature=T_in_Celsius*celsius,absolute_pressure=1*atm)
   	print(f"DENSITY AT T[C]= {T_in_Celsius} is {float(rho_at_T/(kilogram/meter**3))} kg/m^3") #Print it (can also write to file)

Likewise, you can write this data to a file. Thereby, you can easily check whether the implemented expression indeed agrees with the experimental data. Note that we have made use of Python's *Literal String Interpolation (PEP 498)* here.

.. note::

   In general, instead of  :py:meth:`~pyoomph.materials.generic.MaterialProperties.evaluate_at_condition`, you can also use e.g. ``air.mass_density(temperature=18*celsius,absolute_pressure=1*atm)`` to evaluate properties at a given condition. This way of substitution of ``var`` terms can be applied for arbitrary expressions, i.e. also outside of the context of material properties. However, in case of property expressions of mixtures (see next page), :py:meth:`~pyoomph.materials.generic.MaterialProperties.evaluate_at_condition` automatically casts mass, mole and volume fractions accordingly and is hence preferred.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <materials_gas_mixture.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		   
