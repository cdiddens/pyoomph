.. _secadvstabrbconv:

Rayleigh-Benard convection in a cylindrical container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To illustrate the quite longish preface of this section by an example, let us consider a Rayleigh-Benard setting in a cylinder, which is heated from below and cooled from above, with no-slip boundary conditions at all walls. At some specific temperature difference (or better: Rayleigh number), convection will set in. We want to use the azimuthal stability framework to obtain the critical Rayleigh number :math:`\operatorname{Ra}` as function of the aspect ratio :math:`\Gamma=R/H` of the cylinder. We have to do it individually for each mode :math:`m`.

We start by the problem definition, analogous to the same case discussed in :cite:`Diddens2024`:

.. literalinclude:: rayleigh_benard_azimuthal_stability.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: self+=RectangularQuadMesh(size=[1, 1], N=20)

By setting the spatial scale of ``"coordinate_x"`` in the axisymmetric coordinate system, we effectively scale the radial coordinate :math:`r\to\Gamma r`, so that we can modify the cylinder radius without changing the mesh at all. Of course, one should not go to extreme aspect ratios (:math:`\Gamma\ll 1` or :math:`\Gamma \gg 1`) by this, since the solution won't be captured well then.

The rest starts trivially, just adding Navier-Stokes with body force given by the nondimensional temperature, which is solved by a corresponding advection-diffusion equation:

.. literalinclude:: rayleigh_benard_azimuthal_stability.py
   :language: python
   :start-at: RaPr=self.Ra*self.Pr # Shortcut for Ra*Pr
   :end-at: eqs += AdvectionDiffusionEquations(fieldnames="T",diffusivity=1, space="C1")

With ``pressure_factor`` in the :py:class:`~pyoomph.equations.navier_stokes.NavierStokesEquations`, we scale the pressure with the product of the Rayleigh and Prandtl number. This product is entering the bulk force, i.e. the buoyancy. When scaling the pressure the same way, the stationary pressure field is independent of :math:`\operatorname{Ra}\operatorname{Pr}`. Thereby, one can solve the stationary conductive solution (mainly pressure and temperature field) for any Rayleigh number and change the Rayleigh number afterwards.

Furthermore, we have to fix the null space of the pressure, originating from the fact that only no-slip boundary conditions are used. Usually, we just pin e.g. a single corner to some pressure value. However, this is problematic, since it will also pin the corresponding eigenfunction value to zero there. A typical Dirichlet condition would just remove the pressure value at this corner from the unknowns and hence also from the pressure eigenfunction. Therefore, the volume average of the pressure is enforced to be zero instead. All pressure values remain unpinned and will have a degree of freedom in the eigenvector as well. However, when considering modes :math:`m\neq 0`, the average pressure of the eigenfunction will be automatically zero, since :math:`\exp(im\phi)` averages to zero. In that case, we must deactivate this constraint to prevent overconstrainment. This is achieved by the keyword argument ``set_zero_on_normal_mode_eigensolve`` in the pressure null space removal :py:meth:`~pyoomph.equations.navier_stokes.StokesEquations.with_pressure_integral_constraint`.

The boundary conditions are straightforward:

.. literalinclude:: rayleigh_benard_azimuthal_stability.py
   :language: python
   :start-at: # Boundary conditions
   :end-at: self+=eqs@"domain"

Note that the :py:class:`~pyoomph.equations.navier_stokes.NoSlipBC` will also set the :math:`\phi`-component of the velocity to zero automatically. Also, note the :py:class:`~pyoomph.meshes.bcs.AxisymmetryBC`, which will set the correct boundary conditions for the azimuthal stability analysis, as outlined before. Also normal output is added, before the equation system is added to the problem. One last thing which has to be done when running the problem is to activate the azimuthal stability analysis. This is done by passing ``azimuthal_stability=True`` to the :py:meth:`~pyoomph.generic.problem.Problem.setup_for_stability_analysis` call.

.. code:: python

           # Activating azimuthal stability: It will perform all necessary adjustments, i.e.
           #   -expand fields and test functions with exp(i*m*phi)
           #   -consider phi-components in vector fields, i.e. here velocity
           #   -incorporate phi-derivatives in grad and div
           #   -generate the base residual, Jacobian, mass matrix and Hessian, but also
           #    the corresponding versions for the azimuthal mode m!=0
           problem.setup_for_stability_analysis(azimuthal_stability=True)


..  figure:: rb_cyl.*
	:name: figstabilityrbcyl
	:align: center
	:alt: Response of an excited drum
	:class: with-shadow
	:width: 100%

	Critical Rayleigh number for the onset of convection as function of the aspect ratio :math:`\Gamma` and the critical eigenfunction for aspect ratio :math:`\Gamma=1` and azimuthal mode :math:`m=2` and :math:`\Gamma=m=3`, respectively.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <rayleigh_benard_azimuthal_stability.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
