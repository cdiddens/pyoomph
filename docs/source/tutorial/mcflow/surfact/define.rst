Defining insoluble surfactants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To register an insoluble surfactant to the material library, one has to inherit from the :py:class:`~pyoomph.materials.generic.SurfactantProperties` class and again decorate it with the ``@MaterialProperties.register()``. For an insoluble surfactant, it is sufficient to just set a default value for the :py:attr:`~pyoomph.materials.generic.SurfactantProperties.surface_diffusivity`, i.e. :math:`D_S` in :math:numref:`eqmcflowsurftransport`:

.. code:: python

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

An insoluble surfactant will reduce the surface tension according to some relation. Therefore, we must define interface properties in the material library for each combination of the liquid composition and the surfactants on the interface (and potentially also for different gas compositions). This can be done similar to the specification of liquid-gas interface properties as discussed in :numref:`secmcflowlginterfaces`. We just inherit from the class :py:class:`~pyoomph.materials.generic.DefaultLiquidGasInterface`, which automatically sets the surface tension to the liquid-gas surface tension from the :py:attr:`~pyoomph.materials.generic.BaseLiquidProperties.default_surface_tension`. We can then modify the surface tension and potentially also the surface diffusivity for this particular interface.

.. code:: python

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

When the gas phase may be an arbitrary pure gas or gaseous mixture, we can just leave out the ``gas`` specification. Then, this property will hold for all gases. The :py:attr:`~pyoomph.materials.generic.BaseInterfaceProperties.surface_tension` is now just altered by :math:`\sigma(\Gamma,T)=\sigma_0(T)-RT\Gamma`, with :math:`R` being the gas constant. This relation is the simplest effect of surfactants and belongs to the *Henry isotherm*. It just considers the thermodynamic effect of the presence of other molecules at the interface without any interaction terms.

Additionally, we can change the surface diffusivity :math:`D_S` to give a different value on each interface, i.e. overriding the default :py:attr:`~pyoomph.materials.generic.SurfactantProperties.surface_diffusivity` set in the :py:class:`~pyoomph.materials.generic.SurfactantProperties` class.

When the surfactant is defined, we can obtain the interface properties by a combination of liquid properties, gas properties and a surfactant ``dict``. We cannot use the operator ``|`` anymore, i.e. ``liquid | gas`` to get the properties, since the surfactant table must be passed as third argument. Therefore, one must call the :py:func:`~pyoomph.materials.generic.get_interface_properties` function, which finds the right interface properties based on the passed phases and the ``surfactants``. The latter is just a ``dict``, where the keys are either surfactant names (strings) or surfactant properties loaded by :py:func:`~pyoomph.materials.generic.get_surfactant`. The values of the ``dict`` ``surfactants`` are the initial concentrations:

.. code:: python

   	liquid = get_pure_liquid("water")
   	gas = get_pure_gas("air")
   	surfactants = {"my_insoluble_surfactant": 1 * micro * mol / meter ** 2} # Dict stating the initial concentration
   	# alternatively, load the surfactant:
   	#       my_surfactant=get_surfactant("my_insoluble_surfactant")
   	#       surfactants = {my_surfactant: 1 * micro * mol / meter ** 2} #

   	# Gettting interface properties with surfactants
   	interface = get_interface_properties(liquid, gas, surfactants=surfactants)

Again, we can just get the properties, as e.g. the :py:attr:`~pyoomph.materials.generic.BaseInterfaceProperties.surface_tension` directly from the properties. But usually, these are functions of the liquid composition and the temperature. If one wants to get the initial surface tension, e.g. to set a reasonable pressure scale based on the initial surface tension, one can again plug in the initial mixture composition of the liquid into the expression to evaluate the expression at the initial liquid composition and temperature:

.. code:: python

   	# Getting e.g. the surface tension
   	sigma=interface.surface_tension
   	print(sigma) # Rather complicated, since it depends on T and Gamma

   	# First plug in by the liquid initial composition and the temperature
   	sigma1=liquid.evaluate_at_condition(sigma,liquid.initial_condition,temperature=20*celsius)
   	print(sigma1) # Still a function of Gamma

   	# Now also evaluate at the initial surfactant concentration
   	sigma2=interface.evaluate_at_initial_surfactant_concentrations(sigma1) # plug in initial surfactant concentration
   	print(sigma2)

However, ``sigma1`` will still depend on the surface concentration :math:`\Gamma` of the surfactant ``"my_insoluble_surfactant"``. To plug in the initial surface concentrations, one to call the :py:meth:`~pyoomph.materials.generic.LiquidGasInterfaceProperties.evaluate_at_initial_surfactant_concentrations` method of the interface properties. Thereby, ``sigma2`` will be just a constant value, corresponding to the initial surface tension of this interface.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <insoluble_surfactant_definition.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		   