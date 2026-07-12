Evaporation model
~~~~~~~~~~~~~~~~~

Still missing so far are the actual mass transfer rates :math:`j_\alpha`. We have elaborated how to conserve mass when it crosses the interface, i.e. what Neumann conditions for the advection-diffusion equations must be imposed and also considered for the mass transfer in the kinematic boundary condition. However, up till now, :math:`j_\alpha` can still be arbitrary.

A common way for evaporating droplets is to impose the saturated vapor at the gas side of the interface and use :math:numref:`eqmcflowjbyjsimple`, i.e. just evaluating the normal diffusive flux of the vapor at the interface in the gas phase. This works fine if the gas phase has air as host medium. A pure water droplet can hence easily evaporate into the air. However, if both the liquid and the gas phase consist only of water - i.e. no air is present as host component - one cannot impose any saturated vapor. In fact, there is no mass fraction field to be solved and thus there are no diffusive fluxes. Still, mass transfer happens if the temperature at the interface deviates from the boiling point.

To overcome this issue and provide a most general evaporation model, the default evaporation model of pyoomph does not impose any saturated vapor directly. Instead, the ideal gas law and Dalton's law are used to state that the mass transfer rate :math:`j_\alpha` is proportional to the difference of the equilibrium vapor mole fraction, which is :math:`p_\alpha^\text{sat}/p_\text{amb}`, and the actual vapor mole fraction :math:`x_\alpha^\text{g}` in the gas phase:

.. math:: j_\alpha=j_0\cdot\left(\frac{p^\text{sat}_\alpha}{p_\text{abs}}-x_\alpha^\text{g}\right)\,.

Here, :math:`j_0` is an intense rate factor, by default :math:`j_0=100\:\mathrm{kg}/(\mathrm{m^2} \cdot \mathrm{s})`, so that it quickly attains equilibrium. However, with this model also the mass transfer between pure water in both liquid and gas phases behaves correctly: the saturation pressure :math:`p^\text{sat}_\alpha` depends on the temperature :math:`T` and at the boiling point, :math:`p^\text{sat}_\alpha(T_\text{boil})=p_\text{abs}` holds. Since the mole fraction :math:`x_\alpha=1` in a single component system, it will indeed lead to the correct dynamics. In this setting, of course, the latent heat of evaporation must be considered. Thereby, the process will be limited by the thermal transport to the interface, provided :math:`j_0` is chosen sufficiently high.

The mole fractions :math:`x_\alpha^\phi` can be accessed via e.g. ``var("molefrac_water")`` and these fields are calculated based on the :py:attr:`~pyoomph.materials.generic.MaterialProperties.molar_mass` values of the pure components in the mixture from the mass fractions :math:`w_\alpha^\phi` internally. The saturation pressure :math:`p^\text{sat}_\alpha` can be accessed by e.g. ``get_vapor_pressure_for("water")`` of the liquid phase properties. For pure liquids, it is set directly with the property :py:attr:`~pyoomph.materials.generic.PureLiquidProperties.vapor_pressure` or by Antoine coefficients via :py:meth:`~pyoomph.materials.generic.PureLiquidProperties.set_vapor_pressure_by_Antoine_coeffs` (cf. :numref:`secmcflowpureliquids`). In case of a liquid mixture, Raoult's law is used:

.. math:: :label: eqmcflowraoults

   p^\text{sat}_\alpha=x_\alpha^\text{l}\gamma_\alpha p^\text{sat,pure}_\alpha

Here, :math:`x_\alpha^\text{l}` is the mole fraction of the component :math:`\alpha` in the liquid phase and :math:`p^\text{sat,pure}_\alpha` is the saturation pressure of the pure substance :math:`\alpha`. The activity coefficients :math:`\gamma_\alpha` can be either set directly or calculated by UNIFAC. If they are not set, they default to unity, i.e. the ideal Raoult's law. The activity coefficients will be discussed later in :numref:`secmcflowunifac`.

.. warning::

   Only the components that are present in both phases are allowed to evaporate! If you have e.g. a liquid phase of pure water and a gas phase of pure air, no evaporation will happen. You must add gaseous water to the gas phase. If there is initially no water vapor in the gas phase, you still must provide it in order to allow water to evaporate, e.g. by setting the gas mixture as

   .. container:: center

      ``gas_props=Mixture(get_pure_gas("air")+0*get_pure_gas("water"))``

Finally, note that the evaporation model can be customized by defining custom interface properties. One can set the mass transfer model by :py:attr:`~pyoomph.materials.generic.BaseInterfaceProperties.set_mass_transfer_model` of the :py:class:`~pyoomph.materials.generic.LiquidGasInterfaceProperties` to a custom evaporation model (cf. :py:mod:`pyoomph.materials.mass_transfer`). If one just wants to modify :math:`j_0`, one can do so by setting :py:attr:`~pyoomph.materials.mass_transfer.DifferenceDrivenMassTransferModel.default_mass_flux_coefficient` of the default mass transfer model :py:class:`~pyoomph.materials.mass_transfer.DifferenceDrivenMassTransferModelLiquidGas`, which is accessed by :py:meth:`~pyoomph.materials.generic.LiquidGasInterfaceProperties.get_mass_transfer_model` of the interface properties.
