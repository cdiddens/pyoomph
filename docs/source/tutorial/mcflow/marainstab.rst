.. _secRicardo:

Example: Marangoni instability in a Hele-Shaw cell
--------------------------------------------------

We want to investigate the Marangoni instability (cf. :numref:`secpdemarainstab`) of an evaporating ethanol-water mixture which is confined between two plates at the top and bottom, i.e. by a *Hele-Shaw cell*. The flow in such a cell is usually three-dimensional, but when the plate distance :math:`\delta` is small compared to the flow structures, one can assume that the flow in :math:`z`-direction (i.e. the direction between both plates) is parabolic due to the no-slip boundary conditions at the top and bottom plate, i.e. at :math:`z=0` and :math:`z=\delta`. The velocity :math:`\vec{u}_\text{3d}(x,y,z,t)` is then just given by the average flow :math:`\vec{u}(x,y,t)` via :math:`\vec{u}_\text{3d}=6z(\delta-z)\vec{u}/\delta^2`. The presence of the no-slip boundary conditions modifies the Navier-Stokes equations for the projected two-dimensional flow by a factor :math:`6/5` for the nonlinear term and an additional Brinkman term of :math:`-12\mu\vec{u}/\delta^2`:

.. math:: :label: eqmcflowheleshawns

   \rho\left(\partial_t \vec{u}+\frac{6}{5}\nabla\vec{u}\cdot\vec{u}\right)=-\nabla p+\nabla\cdot\left[\mu\left(\nabla\vec{u}+\nabla\vec{u}^\text{t}\right)\right]-12\frac{\mu}{\delta^2}\vec{u}

In pyoomph, we can just pass the plate distance :math:`\delta` via the ``hele_shaw_thickness`` argument to the :py:func:`~pyoomph.equations.multi_component.CompositionFlowEquations` to automatically account for these modifications of the Navier-Stokes equations.

Experimentally, numerically and analytically, this setting has been investigated in Ref. :cite:`de2021marangoni`. Here, we will use the multi-component equations of pyoomph to reproduce the problem by simulation:

.. literalinclude:: marangoni_instability.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: self.interface_props=None # Interface properties, are determined automatically if not set

In the experiments, the evaporation happens in open space. Here, we only have a two-dimensional setting. While in 3d the ambient conditions could be imposed at infinite distances due to the far field decay of :math:`1/r` with the distance :math:`r` of the vapor concentration from the interface, in 2d it does not work: The corresponding Green's function is :math:`\ln(r)` and hence it is not bounded at infinity. Therefore, we must strongly impose the vapor concentration at a finite distance. This artificial vapor concentration can be set by the composition of the ``gas_mixture`` and must be adjusted so that the typical evaporation rates match with the experiment.

Again, in the :py:meth:`~pyoomph.generic.problem.Problem.define_problem`, we set up the ``spatial`` and ``temporal`` scale by hand and let the remaining scales be determined automatically from the properties of the ``liquid_mixture``. However, we adjust the ``velocity`` and ``pressure`` scale by hand afterwards to better match the typical orders of magnitude for this particular problem (e.g. by measurements in the experiments). Since properties may depend on the ``temperature`` and potentially the ``absolute_pressure``, we must again set these on a global (i.e. :py:class:`~pyoomph.generic.problem.Problem`-wide) level with the :py:meth:`~pyoomph.generic.problem.Problem.define_named_var`:

.. literalinclude:: marangoni_instability.py
   :language: python
   :start-at: def define_problem(self):
   :end-at: self.define_named_var(temperature=self.temperature, absolute_pressure=1 * atm)

The mesh is just a :py:class:`~pyoomph.meshes.simplemeshes.RectangularQuadMesh`, but it has to be separated into two domains. This is possible if we pass a function to the argument ``name``. Pyoomph will evaluate this function in the center of each element (in non-dimensional coordinates, i.e. measured in the ``spatial`` scale) and add these elements to the domain by this name. Here, we mark all elements that are on the left half as ``"liquid"``, whereas the elements on the right half are in the ``"gas"`` domain. If an internal facet is between two elements of different domains, it will be automatically added to the interface named by the two domains (in alphabetical order) separated by an underscore, i.e. here the liquid-gas interface will be automatically named ``"gas_liquid"``:

.. literalinclude:: marangoni_instability.py
   :language: python
   :start-at: # Mesh: All elements with center further away than 1*domain_length (measured in spatial scale) will be gas, otherwise liquid
   :end-at: self.add_mesh(mesh)

Then, the equations have to be assembled. If the user does not explicitly select the ``interface_props`` by hand, it will be determined from the material library:

.. literalinclude:: marangoni_instability.py
   :language: python
   :start-at: # We can either set the interface properties by hand, e.g. to modify the surface tension
   :end-at: liq_eqs+=RefineToLevel()@"gas_liquid" # And refine it to max_refinement_level

The liquid equations mainly consist of the :py:func:`~pyoomph.equations.multi_component.CompositionFlowEquations` with the ``liquid_mixture`` properties and the given ``hele_shaw_thickness`` along with a few boundary conditions and a static :py:class:`~pyoomph.equations.multi_component.MultiComponentNavierStokesInterface` with the ``interface_props``. As discussed in the section before, the latter will automatically impose a free surface (static here, since no equations for mesh motion are added) with the :py:attr:`~pyoomph.materials.generic.LiquidGasInterfaceProperties.surface_tension` property of the ``interface_props``. Also the evaporation model is considered and it will couple automatically to the ``"gas"`` domain. Note that we switch the space of the advection-diffusion equations for the required mass fraction fields to ``"C2"``, i.e. second order fields and also add ``spatial_errors`` for the spatial adaptivity. The free interface is always refined to the maximum level by the :py:class:`~pyoomph.equations.generic.RefineToLevel` object.

The gas equations are now just :py:func:`~pyoomph.equations.multi_component.CompositionDiffusionEquations` with a prescribed far field :py:class:`~pyoomph.meshes.bcs.DirichletBC` based on the initial composition of the ``gas_mixture``:

.. literalinclude:: marangoni_instability.py
   :language: python
   :start-at: # Gas
   :end-at: self.add_equations(liq_eqs@"liquid"+gas_eqs@"gas")

To run the simulation, we first slightly perturb the initial condition directly at the interface with random numbers. Thereby, the instability kicks in earlier, whereas otherwise, due to perfect symmetry of the mesh, it would start rather late just by the accumulation of tiny numerical errors of the Newton solver:

.. literalinclude:: marangoni_instability.py
   :language: python
   :start-at: if __name__=="__main__":
   :end-at: problem.run(10*second,startstep=0.01*second,maxstep=0.5*second,outstep=True,temporal_error=1,spatial_adapt=1)

The results are depicted in :numref:`figmcflowheleshaw` and indeed show the experimentally observed coarsening and merging arch-like patterns. For a smaller plate distance, the growing of the arches can be suppressed due to the stronger damping of the Brinkman term in :math:numref:`eqmcflowheleshawns`, whereas without the ``hele_shaw_thickness`` argument (e.g. by setting ``problem.cell_thickness=None``) for the :py:func:`~pyoomph.equations.multi_component.CompositionFlowEquations` (i.e. just the normal 2d Navier-Stokes), a violent chaotic flow would emerge.


..  figure:: heleshaw.*
	:name: figmcflowheleshaw
	:align: center
	:alt: Marangoni instability of an ethanol-water mixture evaporating in a Hele-Shaw cell.
	:class: with-shadow
	:width: 70%

	Marangoni instability of an ethanol-water mixture evaporating in a Hele-Shaw cell.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <marangoni_instability.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   	
		   
