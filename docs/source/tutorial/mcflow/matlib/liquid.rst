.. _secmcflowpureliquids:

Pure liquids
~~~~~~~~~~~~

The definition of pure liquids and liquid mixtures is very similar to the definition of gases, except that pure liquids must inherit from the :py:class:`~pyoomph.materials.generic.PureLiquidProperties` and mixtures from the :py:class:`~pyoomph.materials.generic.MixtureLiquidProperties` base classes. To define e.g. the pure liquid water, we proceed as follows:

.. code:: python

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

The :py:attr:`~pyoomph.materials.generic.BaseLiquidProperties.dynamic_viscosity` and :py:attr:`~pyoomph.materials.generic.MaterialProperties.mass_density` properties are the same as in the case of gases. Also the thermal properties :py:attr:`~pyoomph.materials.generic.MaterialProperties.specific_heat_capacity` and :py:attr:`~pyoomph.materials.generic.MaterialProperties.thermal_conductivity` must be set when thermal effects should be considered in the simulation.

Pure liquids also have some additional properties, which can be set. First of all, there is the thermal property :py:attr:`~pyoomph.materials.generic.PureLiquidProperties.latent_heat_of_evaporation` (measured per mass, not per mole), which must be set when evaporative cooling should be considered. Then, pure liquids have a :py:meth:`~pyoomph.materials.generic.PureLiquidProperties.vapor_pressure` property, which can either be set by hand or by the method :py:meth:`~pyoomph.materials.generic.PureLiquidProperties.set_vapor_pressure_by_Antoine_coeffs`. The coefficients :math:`A`, :math:`B` and :math:`C` of the Antoine equation are often given in the (:math:`\:\mathrm{mmHg}`,\ :math:`\:\mathrm{^\circ C}`) convention in the literature, but you can also change the convention by supplying e.g. ``bar, kelvin`` as additional arguments, if the Antoine parameters :math:`A`, :math:`B` and :math:`C` in the literature are given in that convention. The definition of the vapor pressure is important if mass transfer should be considered, e.g. evaporation. Pure liquids without a vapor pressure will be non-volatile when the default mass transfer model is used.

Also, we can specify a default surface tension of the liquid against the gas phase. Usually, the particular composition of the gas phase does not alter the surface tension strongly, whereas the liquid composition does. Thus, irrespectively of the composition of the gas phase, we can set a typical surface tension which is always used, when a liquid-gas interface is considered with this particular liquid, unless it is explicitly overridden by a definition of particular liquid-gas interface properties (cf. :numref:`secmcflowlginterfaces`).

Finally, if you want to use UNIFAC models to calculate the activity coefficients of mixtures, you must specify the particular UNIFAC subgroups of each pure component. This can be done with the :py:meth:`~pyoomph.materials.generic.PureLiquidProperties.set_unifac_groups` method. More details on this are provided later in :numref:`secmcflowunifac`.

Loading a pure liquid from the material library works exactly as loading a pure gas, but with the routine :py:func:`~pyoomph.materials.generic.get_pure_liquid` instead of :py:func:`~pyoomph.materials.generic.get_pure_gas`.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <materials_liquids.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		   