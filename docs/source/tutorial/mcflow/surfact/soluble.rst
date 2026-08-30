Soluble surfactants and surfactant isotherms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Soluble surfactants are allowed to move from the liquid bulk phase to the interface and vice versa. Hence, in order for a surfactant to be soluble, we must have it in the liquid phase as well as surfactant concentration at the interface. In fact, the :py:class:`~pyoomph.materials.generic.SurfactantProperties` class we have used so far to define insoluble surfactants inherits from the :py:class:`~pyoomph.materials.generic.PureLiquidProperties` class, i.e. each surfactant is automatically also a pure liquid and can hence be mixed with other liquids. However, before doing so, we must at least set the :py:attr:`~pyoomph.materials.generic.MaterialProperties.molar_mass` so that the mole fractions in the liquid mixture can be calculated. This is e.g. relevant for Raoult's law for the evaporation (cf. :math:numref:`eqmcflowraoults`).

.. literalinclude:: soluble_surfactants.py
   :language: python
   :start-at: # Register an soluble surfactant
   :end-at: self.surface_diffusivity = 0.5e-9 * meter ** 2 / second  # default surface diffusivity

Since the surfactant is now also in the liquid phase, we must define the properties of the bulk liquid mixture we want to use. In particular, the presence of the surfactant could influence the :py:attr:`~pyoomph.materials.generic.BaseLiquidProperties.dynamic_viscosity` or :py:attr:`~pyoomph.materials.generic.MaterialProperties.mass_density`. However, for low concentrations, it is reasonable to disregard this effect and just copy the values of e.g. pure water:

.. literalinclude:: soluble_surfactants.py
   :language: python
   :start-at: # Define how the liquid mixture should behave in the bulk
   :end-at: self.set_diffusion_coefficient(1e-9 * meter ** 2 / second)

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

.. literalinclude:: soluble_surfactants.py
   :language: python
   :start-after: from pyoomph.materials.surfactant_isotherms import *
   :end-at: isotherm.apply_on_interface(self, pure_surface_tension=self.surface_tension,min_surface_tension=20*milli*newton/meter)

Since some isotherms have an unbounded surface pressure, the surface tension might become negative once the surfactant concentration exceeds the validity range of the isotherm. Therefore, you can pass a ``min_surface_tension`` to the :py:meth:`~pyoomph.materials.surfactant_isotherms.SurfactantIsotherm.apply_on_interface` call to make sure the surface tension never becomes negative. This can help to prevent crashes of the simulation, when the surfactant leaves the valid bounds.

As for the insoluble surfactants, the interface properties for an interface with soluble surfactants are obtained by :py:func:`~pyoomph.materials.generic.get_interface_properties`. However, in order for the surfactant to be indeed soluble, the surfactant must be present in both the liquid bulk properties and the interface ``surfactants``.

.. literalinclude:: soluble_surfactants.py
   :language: python
   :start-at: # For soluble surfactants, we also must have it in the bulk (potentially at zero concentration)
   :end-at: interface = get_interface_properties(liquid, gas, surfactants=surfactants)

In the example above, the amount of surfactant in the bulk is given as a *mass fraction*. Since both the isotherms and the literature values are expressed in terms of the molar concentration :math:`C`, it is usually more convenient to state it that way directly. Any pure liquid -- and hence any surfactant -- may be multiplied by a concentration instead of by a fraction:

.. code-block:: python

   liquid = Mixture(get_pure_liquid("water") + 1*milli*molar*get_pure_liquid("my_soluble_surfactant"),
                    temperature=20*celsius)

Unlike a dissolved salt (:numref:`secmcflowsaltsdefine`), such a component is never treated as a dilute rider: it is a genuine component of the mixture, i.e. it obtains a mass fraction, a mole fraction and a share of the density, and the registered :py:class:`~pyoomph.materials.generic.MixtureLiquidProperties` for the full set of names is required just as for the mass fraction form. The fractions of the remaining components describe the *base mixture* and are scaled by :math:`1-\sum w` to make room, so that ``water + 20*percent*glycerol + 1*milli*molar*surfactant`` retains a 20 % glycerol base. A mass concentration such as ``5*gram/litre*...`` works in the same way and needs no molar mass. A ``temperature`` must be passed, since the conversion needs the mass density as a number and essentially every density correlation depends on the temperature.

A concentration is an amount per volume, and there are two volumes involved. Which one is meant is selected by the ``concentration_basis`` argument of :py:func:`~pyoomph.materials.generic.Mixture`:

- ``"base_mixture"``, the default, is how a solution is prepared in practice: the base -- here the water, or the water-glycerol mixture -- is mixed first, its volume is measured, and the surfactant is added according to that volume. The statement is thus a mass balance and nothing is assumed about the volume the surfactant itself occupies.
- ``"solution"`` means the amount per volume of the *finished* solution, which is the quantity the simulation reports: ``var("molarconc_my_soluble_surfactant")``, i.e. :math:`C=\rho w/M` with the density :math:`\rho` of the full mixture, is then exactly the value entered at :math:`t=0`. Since :math:`\rho` depends on the very composition being calculated, this is solved by a fixed point iteration.

Both differ by the mass fraction of the surfactant itself, i.e. not at all in the dilute limit that a surfactant is usually in.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <soluble_surfactants.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		   