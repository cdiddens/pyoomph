.. _secmultidomicecyl:

Melting of an ice cylinder with natural convection
--------------------------------------------------

While it has been a lot of work to develop the rather simple example of a propagating ice front, we can now harvest the fruits of our labor by re-using these equations in higher dimensions and different geometries and add even more equations to the system.

As an example system, we consider the melting of an ice cylinder in a cylindrical bath of water. This has been investigated in Ref. :cite:`Weady2022`, resulting in intriguing scalloped ice shapes due to a Kelvin-Helmholtz instability. We hence transfer the previous system into an axisymmetric variant, with an ice cylinder in the center and a liquid domain outside. In the liquid, also buoyancy driven flow will be relevant, where the density anomaly of water is important.

First of all, a mesh is required. We are lazy guys here: Although the geometry would allow us to build the elements by hand, we just use the :py:class:`~pyoomph.meshes.gmsh.GmshTemplate` to construct a mesh via gmsh (cf. :numref:`secspatialgmsh`):

.. literalinclude:: melting_ice_convection.py
   :language: python
   :start-at: from temperature_conduction_propagation import * # Get some equations from the previous example
   :end-at: self.plane_surface("liquid_bottom", "liquid_side", "interface", "liquid_top", name="liquid")

If you have read :numref:`secspatialgmsh`, nothing spectacular is happening here.

Next, we do not only solve thermal conduction, but also thermal convection in the liquid domain. We therefore must add the term :math:`\vec{u}\cdot\nabla T` to our previously developed ``ThermalConductionEquation``. This is most easily done by inheriting from the latter class:

.. literalinclude:: melting_ice_convection.py
   :language: python
   :start-at: # Augment the conduction equation by advection for the liquid phase
   :end-at: self.add_residual(weak(self.rho*self.c_p*dot(self.wind,grad(T)),Ttest)) # Just add the advection term

Next, the problem will be defined. Although the problem is entirely different from the previous one, let us re-use it to copy the physical parameters as e.g. the conductivities and the latent heat:

.. literalinclude:: melting_ice_convection.py
   :language: python
   :start-at: # We inherit from the IceFrontProblem to take over the physical parameters. The problem will be quite different:
   :end-at: self.resolution=0.05 # mesh resolution

Besides copying the parameters from the previous problem, where ``L`` is now used for the height of the cylinder, we need two radii, the viscosity of water and a fit for the density anomaly as a function of the temperature. Therefore, we normalize the actual relative temperature :math:`T-T_\text{eq}` by the unit :math:`\:\mathrm{K}` and plug this into a fit for the liquid water mass density between :math:`0\:\mathrm{^\circ C}` and :math:`20\:\mathrm{^\circ C}`.

The :py:meth:`~pyoomph.generic.problem.Problem.define_problem` method starts again by adding the mesh, but this time we have a two-dimensional mesh and switch to an ``"axisymmetric"`` coordinate system. The scales are set as in the previous problem, but we require additional scales for the ``"velocity"`` and ``"pressure"`` fields:

.. literalinclude:: melting_ice_convection.py
   :language: python
   :start-at: def define_problem(self):
   :end-at: self.set_scaling(pressure=self.mu_liq*scale_factor("velocity")/scale_factor("spatial"))

Next, the ice equations are assembled. They are essentially the same, except that we must add a few extra :py:class:`~pyoomph.meshes.bcs.DirichletBC` terms to fix the mesh at the top and bottom boundary. Furthermore, there is no :py:class:`~pyoomph.meshes.bcs.DirichletBC` for the temperature here, except the equilibrium temperature at the ``"interface"``. The ice cylinder will just warm up to :math:`0\:\mathrm{^\circ C}` over the course of the simulation:

.. literalinclude:: melting_ice_convection.py
   :language: python
   :start-at: self.add_mesh(TwoDomainMeshAxi(self.R1,self.R2,self.L,self.resolution))
   :end-at: ice_eqs += DirichletBC(T=self.T_eq)@"interface" # melting temperature at interface

The liquid equations are analogous, except that we use our new class ``ThermalAdvectionConductionEquation`` for the convection term and also add a :py:class:`~pyoomph.equations.navier_stokes.NavierStokesEquations` for the flow, together with no-slip boundary conditions at all interfaces. In reality, the density difference between ice and liquid would give rise to a non-zero normal velocity, when ice melts or solidifies, but this is not considered here, since this contribution is tiny. One could enforce this velocity jump via a Lagrange multiplier, but then, we also would allow for outflow somewhere in the domain to compensate for the gained/lost volume, i.e. to be able to satisfy the continuity equation. Since we do not allow any outflow, one pressure degree of freedom must also be fixed to remove the nullspace of the pressure (cf. :numref:`secspatialstokespuredirichlet`):

.. literalinclude:: melting_ice_convection.py
   :language: python
   :start-at: # Equations for the liquid domain
   :end-at: liq_eqs += DirichletBC(pressure=0) @ "liquid_top/liquid_side" # For pure DirichletBCs, we must fix one pressure degree

..  figure:: icecylinder.*
	:name: figmultidomicecylinder
	:align: center
	:alt: Melting of an ice cylinder with natural convection
	:class: with-shadow
	:width: 100%

	Dynamics of an ice cylinder melting in a liquid bath along with natural convection due to the density anomaly.


Optionally, we can simplify the problem: Since the front will mainly move in radial direction, we can remove all degrees of freedom associated with the :math:`y`-coordinate of the mesh:

.. literalinclude:: melting_ice_convection.py
   :language: python
   :start-at: # Since we know that the mesh mainly moves in y-direction, we can speed up the calculation by removing the motion in y-direction
   :end-at: liq_eqs += DirichletBC(mesh_y=True)

Thereby, we have fewer degrees of freedom in our system and the computation will speed up.

Finally, the interface equations are added as in the previous example and all equations are added to the problem:

.. literalinclude:: melting_ice_convection.py
   :language: python
   :start-at: # Interface: Connect the mesh position and impose the front motion
   :end-at: self.add_equations(ice_eqs@"ice"+liq_eqs@"liquid")

The code for execution is trivial:

.. literalinclude:: melting_ice_convection.py
   :language: python
   :start-at: if __name__=="__main__":
   :end-at: problem.run(400*second,outstep=True,startstep=1*second,maxstep=20*second,temporal_error=1)

The corresponding results are shown in :numref:`figmultidomicecylinder`. Obviously, the interface indeed does not recede in a straight manner, but is deformed due to the natural convection. Based on the results, one can obviously simplify the problem by neglecting the ice phase, since it becomes isothermal at :math:`T_\text{eq}=0` very quickly. This would involve the modification of the ``IceFrontSpeed``, but since this chapter is on multi-domain problems, it will not be addressed here.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <melting_ice_convection.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   	
		    
