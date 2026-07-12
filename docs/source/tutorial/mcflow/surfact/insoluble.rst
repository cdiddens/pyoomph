Insoluble surfactant transport equation in the presence of mass transfer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In :numref:`secmultidomstokessurfact`, we already discussed the surfactant transport equation for insoluble surfactants, however, in the absence of evaporation. When there is no mass transfer, :math:numref:`eqmultidomsurftransport` holds. However, if there is mass transfer, we have to modify it to

.. math:: :label: eqmcflowsurftransport

   \partial_t \Gamma+\nabla_S\cdot\left(\vec{u}_\text{P}\Gamma\right)=\nabla_S\cdot\left(D_S\nabla_S \Gamma\right)

The only modification is the exchange of the fluid velocity :math:`\vec{u}` for the velocity :math:`\vec{u}_\text{P}`, which is the fluid velocity in tangential direction, but the interface velocity in normal direction, i.e.

.. math:: :label: eqmcflowsurftransportupdef

   \vec{u}_\text{P}=(\mathbf{1}-\vec{n}\vec{n})\vec{u}+\left(\vec{u}_\text{I}\cdot\vec{n}\right)\vec{n}\,.

In the absence of mass transfer, the kinematic boundary conditions dictate that the normal interface velocity and the normal fluid velocity are equal and thus :math:`\vec{u}_\text{P}=\vec{u}`. If there is mass transfer, this does not hold. However, the normal velocity in :math:numref:`eqmcflowsurftransport` must follow the interface velocity, not the fluid velocity. This can be understood by the example of a levitating spherical droplet evaporating in free space. In the droplet, the fluid velocity :math:`\vec{u}` will be zero, but the interface velocity will not be zero due to evaporation. When initially a homogeneous insoluble surfactant concentration :math:`\Gamma_0` is on the droplet (with initial droplet radius :math:`R_0`), we can simplify the equation to

.. math:: \partial_t \Gamma=-\nabla_S\cdot\left(\left(\vec{u}_\text{I}\cdot\vec{n}\right)\vec{n}\Gamma\right)\,,

where the diffusion term has been disregarded since the problem will remain isotropic, i.e. :math:`\Gamma` will only depend on time, but not on the location of the interface. Likewise, :math:`\vec{u}_\text{I}` will point in negative normal direction and its magnitude is constant along the interface due to isotropy. After switching to a radial-symmetric spherical coordinate system, we can carry out the surface divergence, leading to

.. math:: \partial_t \Gamma(t)=-\nabla_S\cdot\left(\vec{n}\right)\vec{u}_\text{I}\Gamma=\frac{2}{R(t)}\vec{u}_\text{I}(t)\Gamma(t)=\frac{2\dot{R}}{R}\Gamma(t)\,.

Since the surfactants are insoluble, the total moles of surfactants, i.e. the integral of :math:`\Gamma` over the droplet surface, must be conserved. This is given by :math:`\Gamma(t)=\Gamma_0R_0^2/R^2(t)`. Plugging this in the lhs indeed shows that the surfactant equation conserves the total moles of surfactant. When the fluid velocity :math:`\vec{u}=0` would have been used instead of :math:`\vec{u}_\text{P}`, the total amount of surfactants would not be conserved.

In pyoomph, the surfactant transport equation with :math:`\vec{u}_\text{P}` as advection velocity is already implemented as part of the :py:class:`~pyoomph.equations.multi_component.MultiComponentNavierStokesInterface`. The velocity :math:`\vec{u}_\text{P}` is internally calculated according to :math:numref:`eqmcflowsurftransportupdef` and the rest of the implementation is analogous to :numref:`secALEtimediff`. Furthermore, not only one surfactant may be added, but an arbitrary number.

