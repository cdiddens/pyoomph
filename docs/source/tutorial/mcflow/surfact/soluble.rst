Soluble surfactants and surfactant isotherms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Soluble surfactants are allowed to move from the liquid bulk phase to the interface and vice versa. Hence, in order for a surfactant to be soluble, we must have it in the liquid phase as well as surfactant concentration at the interface. In fact, the :py:class:`~pyoomph.materials.generic.SurfactantProperties` class we have used so far to define insoluble surfactants inherits from the :py:class:`~pyoomph.materials.generic.PureLiquidProperties` class, i.e. each surfactant is automatically also a pure liquid and can hence be mixed with other liquids. However, before doing so, we must at least set the :py:attr:`~pyoomph.materials.generic.MaterialProperties.molar_mass` so that the mole fractions in the liquid mixture can be calculated. This is e.g. relevant for Raoult's law for the evaporation (cf. :math:numref:`eqmcflowraoults`).

.. code:: python

   # Register an soluble surfactant
   @MaterialProperties.register()
   class MySolubleSurfactant(SurfactantProperties):  # It is automatically also a pure liquid
       name = "my_soluble_surfactant"

       def __init__(self):
           super(MySolubleSurfactant, self).__init__()
           self.molar_mass = 100 * gram / mol  # required so that we can mix it with other liquids
           self.surface_diffusivity = 0.5e-9 * meter ** 2 / second  # default surface diffusivity

Since the surfactant is now also in the liquid phase, we must define the properties of the bulk liquid mixture we want to use. In particular, the presence of the surfactant could influence the :py:attr:`~pyoomph.materials.generic.BaseLiquidProperties.dynamic_viscosity` or :py:attr:`~pyoomph.materials.generic.MaterialProperties.mass_density`. However, for low concentrations, it is reasonable to disregard this effect and just copy the values of e.g. pure water:

.. code:: python

   # Define how the liquid mixture should behave in the bulk
   @MaterialProperties.register()
   class MixLiquidWaterMySolubleSurfactant(MixtureLiquidProperties):
       components = {"water", "my_soluble_surfactant"}

       def __init__(self, pure_props):
           super(MixLiquidWaterMySolubleSurfactant, self).__init__(pure_props)
           # Copy the relevant properties from the water. We assume that the surfactant concentration is small
           # so that all properties are close to these of water
           self.mass_density = self.pure_properties["water"].mass_density
           self.dynamic_viscosity = self.pure_properties["water"].dynamic_viscosity
           self.default_surface_tension["gas"] = self.pure_properties["water"].default_surface_tension["gas"]
           # However, we must set a diffusivity
           self.set_diffusion_coefficient(1e-9 * meter ** 2 / second)

However, we must specify the diffusivity in the bulk. This may be different from the diffusivity at the interface.

Of course, also the properties of the interface are relevant, i.e. how the surfactant influences the surface tension. For soluble surfactants, there is another relevant property to set, namely how the surfactant moves between the bulk and the interface. Therefore, the surfactant transport equation :math:numref:`eqmcflowsurftransport` is augmented by a sink/source term :math:`S_\Gamma`:

.. math:: :label: eqmcflowsurftransportsol

   \partial_t \Gamma+\nabla_S\cdot\left(\vec{u}_\text{P}\Gamma\right)=\nabla_S\cdot\left(D_S\nabla_S \Gamma\right)+S_\Gamma

:math:`S_\Gamma` is now the flux (in :math:`\:\mathrm{mol}/\mathrm{m^2} \cdot \mathrm{s}`) from the bulk to the interface. This flux is constituted by adsorption of surfactants to the interface (positive contribution to :math:`S_\Gamma`) and desorption from the interface to the bulk (negative contribution to :math:`S_\Gamma`). Of course, this transfer has to be compensated by the bulk in order to conserve the total mass of the surfactants, i.e. the sum in the liquid bulk and the interface. The molar flux :math:`S_\Gamma` can be converted to a mass flux by multiplying it with the molar mass :math:`M` of the surfactant and this flux can be applied as a Neumann condition, i.e. as a diffusive mass flux, for the corresponding compositional advection-diffusion equation :math:numref:`eqmcflowwadvdiff`. It does not contribute to the mass transfer flux rate :math:`j_\alpha`, though, since the surfactant does not cross the interface. Of course, all this is subject to a few assumptions, since a molecule requires volume in the bulk phase, but will occupy zero volume at the interface. The flux :math:`S_\Gamma` is automatically considered in the :py:class:`~pyoomph.equations.multi_component.MultiComponentNavierStokesInterface`, so there is nothing to be done.

For the adsorption/desorption rates, there are plenty of models in the literature. To that end, pyoomph offers the most common *surfactant isotherms* in the module :py:mod:`pyoomph.materials.surfactant_isotherms`. The isotherms are usually expressed in terms of the surface concentration :math:`\Gamma` and the *molar concentration* :math:`C` in the bulk, where the latter can be calculated from the bulk mass fraction :math:`w` via :math:`C=\rho w/M`. The molar concentrations can be accessed in pyoomph via the prefix ``"molarconc_"``, e.g. ``var("molarconc_my_soluble_surfactant")``. All surfactant isotherms contain expressions for the adsorption flux :math:`S_\Gamma^\text{ads}`, :math:`S_\Gamma^\text{des}` and the surface pressure :math:`\Pi`, where the latter is just the decrease of the surface tension due to the presence of the surfactant, i.e. :math:`\sigma=\sigma_0-\Pi`. The total flux is just :math:`S_\Gamma=S_\Gamma^\text{ads}-S_\Gamma^\text{des}`. The equilibrium relation between :math:`C` and :math:`\Gamma`, where the surfactants in the bulk and the interface are at equilibrium, is given by :math:`S_\Gamma=0`, i.e. :math:`S_\Gamma^\text{ads}=S_\Gamma^\text{des}`. These are listed for all predefined isotherms in :numref:`tabmcflowisotherms`.

.. table:: Predefined surfactant isotherms stating the adsorption and desorption rates and the surface pressure.
   :name: tabmcflowisotherms

   +-------------------------------------------------------------------------+-------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------+
   | isotherm                                                                | :math:`S_\Gamma^\text{ads}`                                       | :math:`S_\Gamma^\text{des}`                                                                            | :math:`\Pi`                                                               |
   +=========================================================================+===================================================================+========================================================================================================+===========================================================================+
   | :py:class:`~pyoomph.materials.surfactant_isotherms.HenryIsotherm`       | :math:`k_\text{ads}C`                                             | :math:`k_\text{des}\Gamma`                                                                             | :math:`RT\Gamma`                                                          |
   +-------------------------------------------------------------------------+-------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------+
   | :py:class:`~pyoomph.materials.surfactant_isotherms.LangmuirIsotherm`    | :math:`k_\text{ads}C\frac{\Gamma_\infty-\Gamma}{\Gamma_\infty}`   | :math:`k_\text{des}\Gamma`                                                                             | :math:`-RT\Gamma_\infty\ln\left(1-\frac{\Gamma}{\Gamma_\infty}\right)`    |
   +-------------------------------------------------------------------------+-------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------+
   | :py:class:`~pyoomph.materials.surfactant_isotherms.VolmerIsotherm`      | :math:`k_\text{ads}C\frac{\Gamma_\infty-\Gamma}{\Gamma_\infty}`   | :math:`k_\text{des}\Gamma\exp\left(\frac{\Gamma}{\Gamma_\infty-\Gamma}\right)`                         | :math:`\frac{RT\Gamma_\infty}{1-\Gamma\Gamma_\infty}`                     |
   +-------------------------------------------------------------------------+-------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------+
   | :py:class:`~pyoomph.materials.surfactant_isotherms.FrumkinIsotherm`     | :math:`k_\text{ads}C\frac{\Gamma_\infty-\Gamma}{\Gamma_\infty}`   | :math:`k_\text{des}\Gamma\exp\left(-\frac{\beta\Gamma}{RT}\right)`                                     | :math:`-RT \Gamma_\infty \ln(1 - \frac{\Gamma}{\Gamma_\infty})`           |
   +-------------------------------------------------------------------------+-------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------+
   | :py:class:`~pyoomph.materials.surfactant_isotherms.VanDerWaalsIsotherm` | :math:`k_\text{ads}C\frac{\Gamma_\infty - \Gamma}{\Gamma_\infty}` | :math:`k_\text{des}\Gamma\exp\left(\frac{\Gamma}{\Gamma_\infty-\Gamma} -\frac{\beta\Gamma}{RT}\right)` | :math:`\frac{RT\Gamma}{1 - \Gamma/\Gamma_\infty}-\frac{\beta\Gamma^2}{2}` |
   +-------------------------------------------------------------------------+-------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------+

To construct an isotherm, we just have to pass the surfactant name and the parameters ``k_ads`` and ``k_des``, as well as potential further parameters ``GammaInfty`` and ``beta`` to the constructor. Sometimes in the literature, you will find a value :math:`K`, which is just :math:`K=k_\text{ads}/k_\text{des}`. Moreover, some literature defines :math:`k_\text{ads}` as a product of :math:`k_\text{ads}\Gamma_\infty`. Here, the convention was chosen that :math:`k_\text{ads}` always has the units :math:`\:\mathrm{m}/\mathrm{s}`, whereas :math:`k_\text{des}` always has the unit :math:`1/\:\mathrm{s}`. If required for the isotherm, the infinity concentration :math:`\Gamma_\infty` has the unit :math:`\:\mathrm{mol}/\mathrm{m^2}` and the interaction parameter :math:`\beta` is associated with the units :math:`\:\mathrm{m^4}/(\mathrm{mol^2} \cdot \mathrm{s^2})`. Hence, when using values from the literature, always make sure that you cast the isotherms and parameters accordingly.

The typical time scale of the surfactant equilibration is given by both :math:`k_\text{ads}` and :math:`k_\text{des}`, whereas the ratio of these and the further parameters control the equilibrium and the surface tension reduction.

To use the isotherms on an interface, we just construct it and apply its method :py:meth:`~pyoomph.materials.surfactant_isotherms.SurfactantIsotherm.apply_on_interface`. This will set the :py:attr:`~pyoomph.materials.generic.BaseInterfaceProperties.surface_tension` of this liquid-gas interface to the passed ``pure_surface_tension`` minus the surface pressure :math:`\Pi`. Furthermore, it will set the transfer rate :math:`S_\Gamma` according to the particular isotherm. :math:`S_\Gamma` can alternatively be set by hand with the :py:attr:`~pyoomph.materials.generic.LiquidGasInterfaceProperties.surfactant_adsorption_rate` ``dict``:

.. code:: python

   @MaterialProperties.register()
   class InterfaceWaterMySolubleSurfactantVSGas(DefaultLiquidGasInterface):
       liquid_components = {"water", "my_soluble_surfactant"}  # Water and the surfactant are in the liquid phase
       # gas_components = {"air","water"} # do not specify any particular gas phase here: Hold for all gas mixtures
       surfactants = {"my_soluble_surfactant"}  # The soluble surfactant may also be on the interface

       def __init__(self, phaseA, phaseB, surfactants):
           super(InterfaceWaterMySolubleSurfactantVSGas, self).__init__(phaseA, phaseB, surfactants)
           # Create a LangmuirIsotherm for my_soluble_surfactant
           isotherm = LangmuirIsotherm("my_soluble_surfactant", k_ads=5e-6 * meter / second, k_des=9.5 / second,
                                       GammaInfty=5 * micro * mol / meter ** 2)
           # And apply it to this interface. This will modify self.surface_tension by substracting the surface pressure
           # and furthermore it will set self.surfactant_adsorption_rate["my_soluble_surfactant"] to the total ad-/desorption flux
           isotherm.apply_on_interface(self, pure_surface_tension=self.surface_tension,min_surface_tension=20*milli*newton/meter)

Since some isotherms have an unbounded surface pressure, the surface tension might become negative once the surfactant concentration exceeds the validity range of the isotherm. Therefore, you can pass a ``min_surface_tension`` to the :py:meth:`~pyoomph.materials.surfactant_isotherms.SurfactantIsotherm.apply_on_interface` call to make sure the surface tension never becomes negative. This can help to prevent crashes of the simulation, when the surfactant leaves the valid bounds.

As for the insoluble surfactants, the interface properties for an interface with soluble surfactants are obtained by :py:func:`~pyoomph.materials.generic.get_interface_properties`. However, in order for the surfactant to be indeed soluble, the surfactant must be present in both the liquid bulk properties and the interface ``surfactants``.

.. code:: python

       # For soluble surfactants, we also must have it in the bulk (potentially at zero concentration)
       liquid = Mixture(get_pure_liquid("water")+0.001*get_pure_liquid("my_soluble_surfactant"))
       gas = get_pure_gas("air")
       # Dict stating the initial surface concentration
       surfactants = {"my_soluble_surfactant": 1 * micro * mol / meter ** 2}

       # Getting interface properties with surfactants.
       # For a soluble surfactant, it must be present in both the liquid phase and in the surfactants dict
       # Any of them may be present at zero concentration, but it must be specified to be present at all
       interface = get_interface_properties(liquid, gas, surfactants=surfactants)


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <soluble_surfactants.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		   