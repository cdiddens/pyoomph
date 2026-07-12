Liquid mixtures
~~~~~~~~~~~~~~~

Properties of liquid mixtures are defined similarly to gas mixtures. Again, we define the required components for this particular mixture and can select one species as passive field, i.e. the composition field which is not explicitly solved for. When we define a pure liquid named ``"glycerol"`` analogous to the pure liquid ``"water"``, we can define the mixture properties e.g. as follows:

.. code:: python

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

Again, as in the case of gas mixtures, the :py:attr:`~pyoomph.materials.generic.MaterialProperties.components` and :py:attr:`~pyoomph.materials.generic.BaseMixedProperties.passive_field` must be set. The constructor takes again a ``dict`` of the pure properties.

If one does not know details on the particular change of the liquid properties with the composition, one always can use :py:meth:`~pyoomph.materials.generic.BaseMixedProperties.set_by_weighted_average` to calculate the average of the pure properties weighted by the local mass fractions. This makes at least sure that the properties are correct when taking the pure limits. One can also modify the optional argument ``fraction_type`` to ``"mole_fraction"`` to blend between the pure properties weighted by the mole fractions instead of the mass fractions. The local mass and mole fractions of each component can be obtained by :py:meth:`~pyoomph.materials.generic.BaseMixedProperties.get_mass_fraction_field` and :py:meth:`~pyoomph.materials.generic.BaseMixedProperties.get_mole_fraction_field`, respectively. Alternatively, one can directly use e.g. ``var("massfrac_water")`` or ``var("molefrac_glycerol")`` to bind these fields to form arbitrary expressions.

As shown in the above example for the :py:meth:`~pyoomph.materials.generic.BaseLiquidProperties.dynamic_viscosity`, one can assemble functions of the composition and temperature easily. Here, we have used a viscosity model developed by :cite:t:`Cheng2008`, while the surface tension was obtained by a fit of experimental data :cite:`Takamura2012`. The same holds true for the diffusion coefficient based on the data of :cite:t:`DErrico2004`.

We can set the activity coefficients either directly by setting the ``dict`` values of the member :py:attr:`~pyoomph.materials.generic.MixtureLiquidProperties.activity_coefficients`, e.g. ``activity_coefficients["water"]=...``. If the vapor pressure shall be calculated by Raoult's law (cf. :math:numref:`eqmcflowraoults` later on), one has to call :py:meth:`~pyoomph.materials.generic.MixtureLiquidProperties.set_vapor_pressure_by_raoults_law` afterwards. Alternatively, the vapor pressure of each component can be set directly by the ``dict`` :py:attr:`~pyoomph.materials.generic.MixtureLiquidProperties.vapor_pressure_for`, e.g. ``vapor_pressure_for["water"]=...``. One can also invoke the various UNIFAC models to calculate the activity coefficients and set the vapor pressure according to Raoult's law with these activity coefficients. To that end, a simple call of :py:meth:`~pyoomph.materials.generic.MixtureLiquidProperties.set_activity_coefficients_by_unifac` will do the trick. One has to select a particular UNIFAC model (``"Original"``, ``"Dortmund"`` or ``"AIOMFAC"``). Of course, to use these models, one has to set the group contributions in the pure liquids (cf. :numref:`secmcflowunifac` later on).

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <materials_liquids.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		   