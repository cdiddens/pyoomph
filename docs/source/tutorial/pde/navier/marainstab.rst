.. _secpdemarainstab:

Marangoni instability
---------------------

Another interesting instability that can arise in a system that combines the Navier-Stokes equation (velocity :math:`\vec{u}`, pressure :math:`p`) with an advection-diffusion equation (scalar field :math:`c`) is the *Marangoni instability*. This instability is driven by a combination of the non-linear advection term :math:`\vec{u}\cdot\nabla c` and a surface tension :math:`\sigma=\sigma(c)` that depends on :math:`c`. Obviously, we must impose a surface tension force, in particular a tangential traction due to potential surface tension gradient along the interface. As in the previous example, we use the predefined classes :py:class:`~pyoomph.equations.navier_stokes.NavierStokesEquations` and :py:class:`~pyoomph.equations.advection_diffusion.AdvectionDiffusionEquations`, where the velocity of the former is used to advect the field :math:`c` in the latter. The field :math:`c` just couples back via a tangential traction at the top interface. We will additionally use physical dimensions for this example:

.. literalinclude:: marangoni_instability.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: problem.run(1*second,outstep=True,startstep=0.01*second,maxstep=0.1*second,spatial_adapt=1,temporal_error=1)

Since we have a dimensional problem now, all quantities as e.g. ``rho`` and ``mu`` and also the size of the box ``W``\ :math:`\times`\ ``H`` are dimensional. To nondimensionalize, we have to set the ``spatial`` and ``temporal`` scales, as well as the ``pressure`` and ``velocity`` scale. The scale of the field ``c`` is not set, since we use a non-dimensional field for :math:`c`. Thereby, the problem gets nondimensionalized internally. As initial condition for :math:`c`, we have a linear gradient in :math:`y` direction (lower :math:`c` in the bulk as compared to the ``"top"`` interface) with a tiny tangential perturbation. The ``"left"``, ``"right"`` and ``"bottom"`` sides are just no-slip boundary conditions.

At the ``"top"`` interface, we just set the :math:`y`-velocity to zero, i.e. the liquid is not allowed to flow out of the domain. Furthermore, we define the surface tension :math:`\sigma=c\,\partial_c \sigma`, i.e. linearly dependent on the field :math:`c`. The factor ``dsigma_dc`` controls the direction, i.e. whether the surface tension increases or decreases with :math:`c`. The absolute value of the surface tension does not matter in this setting, since we only calculate tangential gradients thereof. To apply the Marangoni force, we just add a :py:class:`~pyoomph.meshes.bcs.NeumannBC` that applies the traction. The minus sign stems from the fact, that the Neumann term :math:`\langle \cdot,\cdot\rangle` in :math:numref:`eqspatialstokesweak` is negative. Since the gradient of the surface tension will only have contributions in :math:`x`-direction, it is indeed just a tangential contribution, i.e. applied on the :math:`x`-direction of the velocity. In total, this means that we apply the tangential Marangoni traction :math:`t_x=\partial_x \sigma`.
  
..  figure:: marangoni_instab.*
	:name: figpdemarangoniinstab
	:align: center
	:alt: Marangoni instability.
	:class: with-shadow
	:width: 100%

	Marangoni (in)stability. (left) When the surface tension decreases with :math:`c`, i.e. :math:`\partial_c \sigma<0`, the surface is stable. (right) when :math:`\sigma` increases with :math:`c`, we have initially a higher surface tension than the liquid in the bulk would yield. Any tiny perturbation can pull up liquid from the bulk, locally reducing the surface tension even more and enhancing the perturbation. This can easily result in chaotic dynamics.


As apparent from :numref:`figpdemarangoniinstab`, we see a drastic effect of whether :math:`\partial_c \sigma` is positive or negative. When :math:`\partial_c \sigma` is negative, any perturbation at the interface will be damped out. Any Marangoni flow will pull up liquids with higher surface tension to the positions with previously lower surface tension. Thereby, the Marangoni flow gets hampered over time. If, however, :math:`\partial_c \sigma>0`, it is vice versa: Any perturbation will pull up liquid with lower surface tension to spots where already a lower surface tension was before. Thereby, the Marangoni instability is triggered, leading to a self-enhancing Marangoni effect that eventually breaks up into chaotic flow. Of course, if the gradient of :math:`c` in bulk direction would be inverted, i.e. higher :math:`c` in the bulk as compared to the interface, also the dynamics would be unstable for :math:`\partial_c\sigma<0` and stable for :math:`\partial_c\sigma>0`.

It is remarkable that the tiny perturbation and the tiny dependence :math:`\partial_c \sigma` of the surface tension on the field :math:`c` are sufficient to trigger this chaotic dynamics. However, the *Marangoni number*, i.e. the nondimensional number to estimate the Marangoni effect, has the viscosity and in particular the diffusivity in the denominator. Both these quantities are so small that the Marangoni number is in fact large.

The Marangoni instability is the explanation why e.g. evaporating droplets consisting of ethanol and water are chaotic, whereas evaporating droplets consisting of glycerol and water show regular flow.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <marangoni_instability.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
