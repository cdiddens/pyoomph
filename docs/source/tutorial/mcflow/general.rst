Bulk equations
--------------

In multi-component flow, we have a mixture of :math:`\alpha=1,\ldots,n` components in each phase :math:`\phi`. In the bulk, the fluid properties, i.e. the density :math:`\rho`, viscosity :math:`\mu` and the diffusion coefficients :math:`D_{\alpha\beta}` may depend on the local composition. Since the Navier-Stokes equations give the mass-averaged velocity :math:`\vec{u}`, it is beneficial to express the composition in terms of mass fractions :math:`w_1,\ldots,w_n`. Each of these mass fractions ranges from 0 to 1 and furthermore, the sum :math:`\sum_\alpha w_\alpha=1` holds. Thereby, we do not have to explicitly consider :math:`w_n`, since :math:`w_n=1-\sum_{\alpha=1}^{n-1}w_i` holds.

The bulk equations for multi-component flow are hence a combination of the Navier-Stokes equations and :math:`n-1` advection-diffusion equations for the fluid composition:

.. math:: :label: eqmcflowwadvdiff

   \begin{aligned}
   \rho\left(\partial_t \vec{u}+\nabla\vec{u}\cdot\vec{u}\right)&=-\nabla p+\nabla\cdot\left[\mu\left(\nabla\vec{u}+\nabla\vec{u}^\text{t}\right)\right]+\rho\vec{g}+\vec{f}\\
   \partial_t\rho+\nabla\cdot\left(\rho\vec{u}\right)&=0\\
   \rho\left(\partial_t w_\alpha+\vec{u}\cdot\nabla w_\alpha\right)&=\nabla\cdot\left(\rho\sum_\beta D_{\alpha\beta} \nabla w_\beta\right) \quad \text{for}\quad \alpha=1,\ldots,n-1 \\
   w_n&=1-\sum_{\alpha}^{n-1}w_\alpha
   \end{aligned}

Here, :math:`\vec{g}` is a gravity vector and :math:`\vec{f}` is an additional arbitrary bulk force. Furthermore, if the thermal fluctuations are relevant, one can add a temperature equation

.. math:: :label: eqmcflowtempeq

   \rho c_p\left(\partial_t T+\vec{u}\cdot\nabla T\right)=\nabla\cdot\left(\lambda \nabla T\right)

All properties, the mass density :math:`\rho`, the dynamic viscosity :math:`\mu`, the diffusion matrix :math:`D_{\alpha\beta}`, the specific heat capacity :math:`c_p` and the thermal conductivity :math:`\lambda` may depend on the local composition :math:`w_\alpha` and the temperature :math:`T`. In particular for gases, the properties also depend on the absolute pressure :math:`p_\text{abs}`. Note that this is usually not the pressure :math:`p` from the Navier-Stokes equations, since one usually chooses the latter to be zero at stress free boundaries. For the properties of gases, it is often sufficient to approximate the absolute pressure by the ambient pressure :math:`p_\text{amb}` (e.g. :math:`p_\text{amb}=1\:\mathrm{atm}`). If pressure fluctuations are strong (i.e. :math:`p/p_\text{amb}` is considerable), one could also just set :math:`p_\text{abs}=p_\text{amb}+p`.

Pyoomph generates this entire system by multiple equations via :py:func:`~pyoomph.equations.multi_component.CompositionFlowEquations`. If the flow shall not be considered, i.e. :math:`\vec{u}=0` is a reasonable assumption (e.g. in the gas phase of an evaporating droplet), one can remove the Navier-Stokes equations by using the :py:func:`~pyoomph.equations.multi_component.CompositionDiffusionEquations` function instead. Both will require a fluid properties object as the first argument, which comprises the (potentially varying) mass density, viscosity, diffusion matrix, etc. but also the initial composition of the mixture. Before addressing these classes in detail, let us first discuss how to create properties for fluid mixtures in the next section.


