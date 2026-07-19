.. _secALEsolid:

Nonlinear solid mechanics
-------------------------

The possibility of having a moving mesh, i.e. having the mesh coordinates as degrees of freedom, is obviously a good foundation to account for deformable solids. In this section, we cover the possibilities of pyoomph for nonlinear elasticity of solid bodies. The essential idea in pyoomph's implementation is that the Lagrangian coordinates represent the undeformed body, whereas the Eulerian coordinates reflect the current deformed configuration.

.. note::
	The implementation of nonlinear elasticity is essentially a one-to-one copy of the `implementation in oomph-lib <https://oomph-lib.github.io/oomph-lib/doc/solid/solid_theory/html/index.html>`__. Oomph-lib's documentation covers the individual aspects in considerably more detail, and with the exact co- and contravariant definitions, than this documentation here.
	
As governing equations, we have the principle of virtual displacements, which reads in pyoomph's notation

.. math::

	\left( \rho \partial_t^2 \vec{x} - \vec{f}, \Gamma \vec{\chi} \right)_\xi + \left( \left(\nabla_\xi \vec{x}\right) \boldsymbol{\sigma},\Gamma \nabla_\xi\vec{\chi}\right)_\xi-\langle \vec{T},\vec{\chi}\rangle=0


Again, :math:`\vec{x}(\vec{\xi},t)` represents the deformed configuration (with test function :math:`\vec{\chi}`) and :math:`\vec{\xi}` is the undeformed configuration. :math:`\vec{f}` is a bulk force density (in the undeformed configuration), :math:`\rho` is the mass density in undeformed configuration and :math:`\Gamma` is an isotropic growth factor, i.e. the factor the undeformed solid wants to grow in terms of the undeformed length/area/volume (depending on the dimension). 
The bulk part of the equation is available in the class :py:class:`~pyoomph.equations.solid.DeformableSolidEquations`, whereas a traction :math:`\vec{T}` can be applied on the boundary of the solid by the class :py:class:`~pyoomph.equations.solid.SolidTraction`.
For pressures, tractions in negative normal direction, the class :py:class:`~pyoomph.equations.solid.SolidNormalTraction` can be used. 

The constitutive law is entirely wrapped in the second Piola-Kirchhoff stress tensor :math:`\boldsymbol{\sigma}`. Following oomph-lib's `GeneralisedHookean class <https://oomph-lib.github.io/oomph-lib/doc/the_data_structure/html/classoomph_1_1GeneralisedHookean.html>`__, a reasonable definition is given by the following

.. math::
	\begin{aligned}
	\sigma_{ij}&=E_{ijkl}\gamma_{kl}\\
	E_{ijkl}&=\frac{E}{1+\nu}\left(\frac{\nu}{1-2\nu}G^{-1}_{ij}G^{-1}_{kl}+\frac{1}{2}\left(G^{-1}_{ik}G^{-1}_{jl}+G^{-1}_{il}G^{-1}_{jk}\right)\right)\\	
	\boldsymbol{G}&=\left(\left(\nabla_\xi \vec{x}\right)^\mathrm{t}\left(\nabla_\xi \vec{x}\right)\right)\\
	\boldsymbol{\gamma}&=\frac{1}{2}\left(\boldsymbol{G}-\boldsymbol{g}\right)\\
	\boldsymbol{g}&=\Gamma^{2/D}\left(\left(\nabla_\xi \vec{\xi}\right)^\mathrm{t}\left(\nabla_\xi \vec{\xi}\right)\right)
	\end{aligned}

Please kindly excuse the sloppy notation not respecting the co- and contravariances here correctly. This is done considerably better in `oomph-lib's documentation <https://oomph-lib.github.io/oomph-lib/doc/solid/solid_theory/html/index.html>`__. However, the way it is written here corresponds to the implementation in pyoomph, where we heavily use matrices and the vector gradients, since they automatically adjust correctly based on the considered coordinate system (e.g. :math:`\boldsymbol{g}` is the (scaled) identity matrix in Cartesian coordinates, but in axisymmetric coordinates, it is not). Entering parameters are Young's modulus :math:`E` and the Poisson ratio :math:`\nu`. For small deformations, it perfectly recovers Hooke's law. :math:`D` is the dimension of the solid, which is e.g. 3 for a 2d mesh with axisymmetric coordinate system.
This constitutive law is implemented in the class :py:class:`~pyoomph.equations.solid.GeneralizedHookeanSolidConstitutiveLaw`.

If :math:`\nu=1/2`, the solid is incompressible and the definition becomes singular. In that case, we have to use 

.. math::
	\begin{align}
	\sigma^{ij}&=-pG^{-1}_{ij}+\frac{E}{3}\left(G^{-1}_{ik}G^{-1}_{jl}+G^{-1}_{il}G^{-1}_{jk}\right)\gamma_{kl}
	\end{align}
	
instead. Here, :math:`p` is a pressure field, which enforces that 

.. math::
	\begin{align}
	\det \boldsymbol{G}-\det \boldsymbol{g}=0
	\end{align}

i.e. that the deformation locally conserves the volume. For this law, we have to use the class :py:class:`~pyoomph.equations.solid.IncompressibleHookeanSolidConstitutiveLaw`. 

.. note::

	For compressible constitutive laws, the mass density :math:`\rho` is to be understood as mass density in the undeformed configuration. The mass density in the deformed configuration can be accessed by ``var("deformed_mass_density")``, which is calculated by the conservation of mass. In the incompressible case, there is no difference and ``var("deformed_mass_density")`` will expand to :math:`\rho`.


.. warning::

	The nonlinear elasticity equations cannot be used in combination with azimuthal or Cartesian normal mode stability analysis as outlined in section :numref:`azimuthalstabana` and :numref:`cartesiannormalstabana`. For this approach, we would have to expand the mesh coordinates also by an azimuthal or additional Cartesian mode, which is not implemented (yet).


.. toctree::
   :maxdepth: 5
   :hidden:
   
   solid/cantilever.rst
   solid/compressed_disc.rst   
   solid/solid_oscillations.rst      

