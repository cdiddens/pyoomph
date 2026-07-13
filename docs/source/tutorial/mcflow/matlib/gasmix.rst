Definition of gaseous mixtures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to solve e.g. the evaporation of a water droplet, you must account for water vapor (i.e. gaseous water) which can diffuse through the ambient gas, typically air. Before creating a gaseous mixture of these two substances, we first must ensure that both pure gaseous substances are defined. Therefore, we first load our old script, where we defined the air and subsequently define pure water in its gaseous phase:

.. code:: python

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

Note that we skipped the definition of the mass density and the dynamic viscosity (as well as the thermal properties) for the ``PureGasWater``. When we intend to only use it within a mixture, e.g. with air, the properties of the pure substance are not required. If we want to solve, however, a flow problem of pure gaseous water, i.e. above the boiling point where pure water can exist in the gas phase, we would require the mass density and dynamic viscosity of the pure substance for the Navier-Stokes equation. The molar mass, on the other hand, is always required - for each substance. It is important to convert mass into molar fractions and it is used for e.g. Raoult's law or the calculation of the mass density of mixtures by the ideal gas law.

Next, we move on defining the properties of a mixture of air and water in the gas phase:

.. code:: python

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

The registration to the library is done - as for pure substances - with the decorator ``@MaterialProperties.register()``. Gas mixtures must inherit from the base class :py:class:`~pyoomph.materials.generic.MixtureGasProperties` and it is important that the constructor accepts a single argument, namely a ``dict`` which contains the instances of the pure properties (i.e. instances of sub-classes of the :py:class:`~pyoomph.materials.generic.PureGasProperties` class), indexed by their name. This ``dict`` of pure properties must be passed to the ``super`` constructor. Then, it is important to specify the class property :py:attr:`~pyoomph.materials.generic.MaterialProperties.components`, which is a ``set`` of the component names within this mixture. This ``set`` is used to find the correct mixture property class, when e.g. mixing the pure substances ``"air"`` and ``"water"`` in a second.

We also should set a passive field. This does not really change a lot, so you can basically pick any of the elements of :py:attr:`~pyoomph.materials.generic.MaterialProperties.components` here. It is just required to simplify the advection-diffusion equations for the mixture: When a mixture of :math:`n` components is considered, only :math:`n-1` advection-diffusion equations for the mass fractions must be solved. The last one follows from 1 minus the sum of the other :math:`n-1` mass fractions. This last field, which is not explicitly solved, is identified by the :py:attr:`~pyoomph.materials.generic.BaseMixedProperties.passive_field`.

.. warning::

   The choice of the :py:attr:`~pyoomph.materials.generic.BaseMixedProperties.passive_field` has one important consequence: If you want to set an :py:class:`~pyoomph.equations.generic.InitialCondition` or a :py:class:`~pyoomph.meshes.bcs.DirichletBC`, you cannot set the mass fraction of the component specified by the :py:attr:`~pyoomph.materials.generic.BaseMixedProperties.passive_field`. In our example here, you cannot explicitly set initial conditions or Dirichlet boundary conditions for the air, but you can set it for water vapor. You still can impose Neumann fluxes, i.e. in/outflux of the passive component, though. This is possible since the corresponding test functions are internally substituted accordingly.

The rest of the constructor follows the definition of the pure substances. You still can access the method :py:meth:`~pyoomph.materials.generic.MixtureGasProperties.set_mass_density_from_ideal_gas_law`, which now will evaluate the mass density according to the local composition. Since at room temperature, the vapor concentration is usually small, we just copy the dynamic viscosity from the pure substance air here. Finally, we have to set a diffusion coefficient. For a binary mixture, a single diffusion coefficient is sufficient, which is set by :py:meth:`~pyoomph.materials.generic.BaseMixedProperties.set_diffusion_coefficient` with the single diffusion coefficient as an argument. Also this coefficient can be a function of the composition, temperature and absolute pressure.

Let us now see how to create a specific mixture:

.. code:: python

   # Get the pure properties
   air=get_pure_gas("air")
   water_vapor=get_pure_gas("water")

   # Mix in terms of mass fraction. One quantifier (here 0.98 for air) can be omitted
   mix_gas=Mixture(air+0.02*water_vapor)

   # We can access the initial condition, which will result in {'massfrac_air': 0.98, 'massfrac_water': 0.02, 'temperature': None}
   print(mix_gas.initial_condition)

   # To evaluate e.g. the mass density at the initial condition, we can just pass the initial condition, but we also have to add information on the pressure and temperature to get a single value
   print(mix_gas.evaluate_at_condition("mass_density",mix_gas.initial_condition,temperature=20*celsius,absolute_pressure=1*atm))

To mix in terms of mass fractions, we can just add multiple pure substances and wrap them into a :py:func:`~pyoomph.materials.generic.Mixture` call. We have to specify also the initial mass fractions, e.g. here :math:`2\:\mathrm{\%}` air in terms of mass fraction. Since the corresponding mass fraction of air, :math:`98\:\mathrm{\%}`, follows from the requirement that all mass fractions have to sum to unity, the quantification of one pure substance can be omitted.

The initial condition can be accessed by the :py:attr:`~pyoomph.materials.generic.MaterialProperties.initial_condition` property, which is a ``dict`` containing the mass fractions. Again, we can evaluate properties by calling :py:meth:`~pyoomph.materials.generic.MaterialProperties.evaluate_at_condition`, but in order to evaluate at the initial condition, we have to pass :py:attr:`~pyoomph.materials.generic.MaterialProperties.initial_condition` as the first argument. Furthermore, since the ideal gas law also requires a temperature and a pressure, we have to pass these as keyword arguments to obtain a single dimensional value for the mass density at the end.

Let us now move on to a ternary gas mixture of water, ethanol and air. Again, first pure ethanol as a gaseous component is required, followed by a definition of the mixture properties:

.. code:: python

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

As apparent, things work exactly the same as for binary mixtures and also higher order mixtures are defined the same way. What we have done additionally here is explicitly defining the dynamic viscosity to be a function of the ethanol mass fraction. The used expression was chosen arbitrarily and not supported by any experimental data. The mass fraction of ethanol can be obtained by ``var("massfrac_ethanol")``, likewise ``var("massfrac_water")`` and ``var("massfrac_air")`` can be used. The latter, since air is the passive field, is implicitly replaced by ``1-var("massfrac_ethanol")-var("massfrac_water")`` when the C code is generated. Additionally, e.g. ``var("molefrac_ethanol")`` can be used for the molar fractions, but where possible, mass fractions are the better choice when e.g. fitting experimental data of some property. This is due to the fact that the composition is solved in terms of mass fractions and molar fractions must be calculated first from the former.

In a ternary or higher order mixture, also the diffusion matrix becomes more complicated. The general diffusive flux of component :math:`\alpha` reads

.. math:: \mathbf{J}_\alpha=-\rho\sum_{\beta=1}^{n} D_{\alpha\beta}\nabla w_\beta,

where :math:`w_\beta` are the mass fraction fields and :math:`D_{\alpha\beta}` are the entries of the diffusion matrix. In the binary mixture, we used :py:meth:`~pyoomph.materials.generic.BaseMixedProperties.set_diffusion_coefficient` with only a single argument, namely a diffusion coefficient. Calling that method in this way will just set the entire diagonal, i.e. *all* entries :math:`D_{ii}`, to the supplied argument. If :py:meth:`~pyoomph.materials.generic.BaseMixedProperties.set_diffusion_coefficient` is called with two parameters, we only set the diagonal diffusion coefficient :math:`D_{ii}` for a single :math:`i`, namely the index :math:`i` which corresponds to the name of the component supplied by the first argument. All unset diffusion coefficients default to zero. This means, in the above example, the diffusion fluxes are set to

.. math:: \mathbf{J}_\text{w}=-\rho D_{\text{ww}}\nabla w_\text{w}, \qquad \mathbf{J}_\text{e}=-\rho D_{\text{ee}}\nabla w_\text{e}

for water (w) and ethanol (e), respectively. The diffusion flux of air has not been defined properly, but since it is the passive component, it is not required.

For ternary and higher mixtures, one also might have to set off-diagonal coefficients, which can be done by e.g. calling ``set_diffusion_coefficient("water","ethanol",...)`` to set the coefficient :math:`D_\text{we}`. Note that off-diagonal diffusion coefficients should not be a constant, but depend on the composition. These off-diagonal entries also can be negative.

Of course, also the thermal properties :py:attr:`~pyoomph.materials.generic.MaterialProperties.thermal_conductivity` and :py:attr:`~pyoomph.materials.generic.MaterialProperties.specific_heat_capacity` must be set in the gas mixture definition class, when thermal dynamics are desired.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <materials_gas_mixture.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		   