.. _secmcflowmcnsinterfacedetails:

What the MultiComponentNavierStokesInterface does
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once a :py:class:`~pyoomph.equations.multi_component.MultiComponentNavierStokesInterface` with the passed interface properties is added to the problem's equations, it will add all relevant contributions of a free surface to the system. If the gas phase has just diffusion, i.e. no flow, by a :py:func:`~pyoomph.equations.multi_component.CompositionDiffusionEquations` class, it will add both the kinematic and the dynamic boundary conditions to the system

.. math::

   \begin{aligned}
   \rho^\text{l}\left(\vec{u}^\text{l}-\vec{u}_\text{I}\right)\cdot\vec{n}&=\sum_\alpha j_\alpha \\
   \vec{n}\cdot\left[-p^\text{l}\mathbf{1}+\mu^\text{l}\left(\nabla\vec{u}^\text{l}+(\nabla\vec{u}^\text{l})^\text{t}\right)\right]&=\sigma\kappa\vec{n}+\nabla_S \sigma
   \end{aligned}

Here, the superscript :math:`\text{l}` refers to the liquid phase. For the details of the weak form implementation, please refer to :numref:`secALEfreesurfNS`. :math:`j_\alpha` are potential mass transfer rates, which will be calculated by a mass transfer model (cf. next section). In the absence of any mass transfer (:math:`j_\alpha=0` for all components :math:`\alpha`), it is again the normal kinematic and dynamic boundary conditions, where the gas phase is not considered due to the disregarded flow.

If gas flow is considered, i.e. when the gas phase has :py:func:`~pyoomph.equations.multi_component.CompositionFlowEquations` instead of :py:func:`~pyoomph.equations.multi_component.CompositionDiffusionEquations`, also the gas velocity and pressure will couple into the system and the :py:class:`~pyoomph.equations.multi_component.MultiComponentNavierStokesInterface` solves the following boundary conditions

.. math::

   \begin{aligned}
   \rho^\text{l}\left(\vec{u}^\text{l}-\vec{u}_\text{I}\right)\cdot\vec{n}&=\sum_\alpha j_\alpha \\
   \rho^\text{g}\left(\vec{u}^\text{g}-\vec{u}_\text{I}\right)\cdot\vec{n}&=\sum_\alpha j_\alpha \\
   \left(\vec{u}^\text{l}-\vec{u}^\text{g}\right)\cdot\vec{t}&=0 \\
   \vec{n}\cdot\left[-(p^\text{l}-p^\text{g})\mathbf{1}+\mu^\text{l}\left(\nabla\vec{u}^\text{l}+(\nabla\vec{u}^\text{l})^\text{t}\right)-\mu^\text{g}\left(\nabla\vec{u}^\text{g}+(\nabla\vec{u}^\text{g})^\text{t}\right)\right]&=\sigma\kappa\vec{n}+\nabla_S \sigma
   \end{aligned}

The tangential velocity is hence continuous whereas the normal velocity will have a jump (Stefan flow) when evaporation is considered and the mass densities differ. Also the stress and the pressure are now coupled. The *vapor recoil pressure* is not considered in these terms, since it is usually a tiny contribution. Such additional contributions must be added manually.

On a moving mesh, the kinematic boundary conditions are enforced by moving the mesh accordingly. If the mesh is static, i.e. no equations for the mesh motion are added, the kinematic boundary condition is enforced on the velocity instead (with interface velocity :math:`\vec{u}_\text{I}=0`). Optionally, one can also use a static interface on a moving mesh with the keyword argument ``static=True``.

Finally, if the mass transfer rates :math:`j_\alpha` are non-zero, the fluid composition on both sides changes as well. Therefore, let the species velocity of component :math:`\alpha` in phase :math:`\phi=\text{l,g}` be given by :math:`\vec{u}_\alpha^\phi`. The species velocity is not the same as the mass-averaged velocity :math:`\vec{u}^\phi` since in a mixture, different species may move in different directions (e.g. in case of diffusion). The mass-averaged velocity is connected to the species velocities via

.. math:: \vec{u}^\phi=\sum_\alpha w_\alpha \vec{u}_\alpha^\phi\,.

A transfer rate of :math:`j_\alpha` (mass per interface area and time) of component :math:`\alpha` crossing the interface means that indeed this particle flux is crossing the interface, i.e.

.. math:: j_\alpha=\rho^\phi w_\alpha^\phi\left(\vec{u}_\alpha^\phi-\vec{u}_\text{I}\right)\cdot\vec{n}

Summing over all :math:`\alpha` gives the kinematic boundary conditions due to :math:`\sum_\alpha w_\alpha=1` and the definition of the mass-averaged velocity. Since the mass has to be conserved when crossing the interface, both sides are the same, i.e. :math:`j_\alpha=j_\alpha^\text{l}=j_\alpha^\text{g}`.

As mentioned, diffusion can lead to a relative motion of species with respect to the mass-averaged velocity. In fact, the diffusive mass flux :math:`\vec{J}_\alpha` is connected to the relative motion of species :math:`\alpha` with respect to the averaged velocity by

.. math:: \vec{J}_\alpha^\phi=\rho^\phi w_\alpha^\phi\left(\vec{u}_\alpha^\phi-\vec{u}^\phi\right)\,.

Thereby, the species velocity :math:`\vec{u}_\alpha^\phi` can be replaced by the diffusive fluxes in the mass transfer rate

.. math:: j_\alpha=\rho^\phi w_\alpha^\phi\left(\vec{u}^\phi-\vec{u}_\text{I} \right)\cdot\vec{n}+ \vec{J}_\alpha^\phi\cdot\vec{n}=w_\alpha \sum_\beta j_\beta+\vec{J}_\alpha^\phi\cdot\vec{n}\,.

Thereby, the normal diffusive flux at the interface, which is exactly the Neumann term which can be imposed for the mass fraction advection diffusion equations :math:numref:`eqmcflowwadvdiff`, can be written as

.. math:: \vec{J}_\alpha^\phi\cdot \vec{n}=j_\alpha-w_\alpha^\phi\sum_\beta j_\beta\,.

If the phase :math:`\phi` considers the flow, i.e. :py:func:`~pyoomph.equations.multi_component.CompositionFlowEquations` are used, these Neumann terms are applied to the equations :math:numref:`eqmcflowwadvdiff` for :math:`\alpha=1,\ldots,n-1`, where the sum over :math:`\beta` includes also the :math:`n^\text{th}` term to account for transfer of the selected :py:attr:`~pyoomph.materials.generic.BaseMixedProperties.passive_field`, which is not explicitly solved for.

If flow is disregarded in phase :math:`\phi`, i.e. using the :py:func:`~pyoomph.equations.multi_component.CompositionDiffusionEquations`, the advection term is also disregarded since it stems from the relative velocity of the fluid to the interface. For domains with :py:func:`~pyoomph.equations.multi_component.CompositionDiffusionEquations`, the mass transfer rates are imposed directly as diffusive fluxes:

.. math:: :label: eqmcflowjbyjsimple
   
   \vec{J}_\alpha^\phi\cdot \vec{n}=j_\alpha\,.
