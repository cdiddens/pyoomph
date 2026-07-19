.. _secmcflowlginterfaces:

Free surfaces and evaporation
-----------------------------

The previous example had just a single liquid domain. When a liquid and a gas domain are in contact, a free surface emerges where evaporation can happen. Fluid-fluid interfaces in general (i.e. liquid-gas or liquid-liquid) are both covered by the :py:class:`~pyoomph.equations.multi_component.MultiComponentNavierStokesInterface` class in the multi-component flow framework of pyoomph. It takes the properties of the interface as a required argument. Interface properties can be obtained from the material library by using the operator ``|`` on the bulk phase properties. When we hence have e.g. multi-component flow via the :py:func:`~pyoomph.equations.multi_component.CompositionFlowEquations` in a ``"liquid"`` domain and either :py:func:`~pyoomph.equations.multi_component.CompositionFlowEquations` (or :py:func:`~pyoomph.equations.multi_component.CompositionDiffusionEquations` for pure diffusive transport) in a ``"gas"`` phase, the assembly of the equations could look like

.. code:: python

   temperature=20*celsius

   # Bulk properites (pure water liquid and air with some vapor in the gas phase)
   liquid=get_pure_liquid("water")
   gas=Mixture(get_pure_gas("air")+20*percent*get_pure_gas("water"),quantity="relative_humidity",temperature=temperature)

   # Get the interface properties
   interface_props=liquid | gas

   # Assembly of an equation system
   eqs=CompositionFlowEquations(liquid)@"liquid"
   eqs+=CompositionDiffusionEquations(gas)@"gas" # (alternatively CompositionFlowEquations)
   # free surface should be added to the liquid side of the liquid-gas interface:
   eqs+=MultiComponentNavierStokesInterface(interface_props)@"liquid/liquid_gas" 

The properties of liquid-gas interfaces can be defined by hand, but it is not necessary: If no particular properties of an interface between a liquid and a gas phase are present in the material library, default properties will be constructed. To that end the :py:attr:`~pyoomph.materials.generic.BaseLiquidProperties.default_surface_tension` dict entry ``["gas"]`` of the liquid bulk properties will be used as :py:attr:`~pyoomph.materials.generic.BaseInterfaceProperties.surface_tension`:

.. code:: python

   # Access the surface tension
   print("Surface tension function:",interface_props.surface_tension) # same as liquid.default_surface_tension["gas"]
   sigma_at_T=liquid.evaluate_at_condition(interface_props.surface_tension,liquid.initial_condition,temperature=temperature)
   print("Surface tension at temperature T",sigma_at_T)

If one, for whatever reason, wants to modify properties of a particular liquid-gas interface in the material library, one can register a particular class for that to the library by inheriting from the :py:class:`~pyoomph.materials.generic.LiquidGasInterfaceProperties` class:

.. code:: python

   # Register a particular liquid-gas interface to the library
   @MaterialProperties.register()
   class CustomInterfacePropertiesWaterVsVaporAir(LiquidGasInterfaceProperties):
       liquid_components = {"water"} # Components in the liquid phase
       gas_components = {"water","air"} # components in the gas phase
       def __init__(self,phaseA,phaseB,surfactants):
           super(CustomInterfacePropertiesWaterVsVaporAir, self).__init__(phaseA,phaseB,surfactants)
           self.surface_tension=50*milli*newton/meter # set a custom surface tension

   # Get the interface properties
   new_interface_props=liquid | gas
   print("New surface tension",new_interface_props.surface_tension)

When one tries to determine the interface properties via the ``|`` operator, these properties will be used when one has a liquid phase consisting of ``"water"`` and a gas phase with the components ``"water"`` and ``"air"``. This is identified by the sets :py:attr:`~pyoomph.materials.generic.LiquidGasInterfaceProperties.liquid_components` and :py:attr:`~pyoomph.materials.generic.LiquidGasInterfaceProperties.gas_components` of the class.

However, if one only wants to change e.g. the surface tension, it is easier to just set it directly after getting the default interface properties. I.e. without defining and registering the particular class, but just overriding the property directly:

.. code:: python

   # Override interface properties by hand
   interface_props.surface_tension=50*milli*newton/meter


.. toctree::
   :maxdepth: 5
   :hidden:
   
   freesurf/details.rst
   freesurf/evapmdl.rst
