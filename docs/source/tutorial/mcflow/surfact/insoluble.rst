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

In pyoomph, this equation is implemented in :py:mod:`~pyoomph.equations.surfactants` and is added automatically by the :py:class:`~pyoomph.equations.multi_component.MultiComponentNavierStokesInterface` for every surfactant registered on the interface properties, so an arbitrary number of them may be present.

When writing :math:numref:`eqmcflowsurftransport` directly, i.e. as a transient term plus the surface divergence of :math:`\vec{u}_\text{P}\Gamma`, it conserves the total amount only up to the order of the time stepping: on a moving mesh the discrete rate of change of the surface metric is not the discrete :math:`\nabla_S\cdot\vec{u}_\text{I}`. Instead, the equation is assembled as the time derivative of the whole integral plus a flux,

.. math:: :label: eqmcflowsurftransportconservative

   \frac{\mathrm{d}}{\mathrm{d}t}\int_S \Gamma v\,\mathrm{d}S-\int_S \Gamma\left(\vec{u}-\vec{u}_\text{I}\right)\cdot\nabla_S v\,\mathrm{d}S+\int_S D_S\nabla_S\Gamma\cdot\nabla_S v\,\mathrm{d}S=0\,,

where :math:`v` is the test function. Note that :math:`\vec{u}_\text{P}` has disappeared: only the slip relative to the mesh advects, and :math:`\nabla_S v` is tangential to the element anyway, so the normal part of :math:`\vec{u}-\vec{u}_\text{I}` -- which is exactly the mass transfer -- cannot contribute. The  non-conservative form is still available as ``MultiComponentNavierStokesInterface(..., surfactant_transport=SurfactantTransportEquations(form="legacy"))``.

Integrating the advection by parts creates a term at the ends of the interface, e.g. at a contact line. Leaving it out is the natural boundary condition of zero total flux, which is what an insoluble surfactant needs, and it is what makes the conservation above exact. Where a nonzero flux is wanted instead, add a :py:class:`~pyoomph.equations.surfactants.SurfactantEndFlux` at that end point.

