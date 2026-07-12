.. _secmultidomstokessurfact:

Stokes' law for a falling droplet with insoluble surfactants
------------------------------------------------------------

In :numref:`secpdestokes_law`, a transient Stokes law is solved. We will now slightly modify this problem and exchange the rigid spherical object for a droplet. For simplicity, we still assume that the capillary number is small, i.e. the droplet does not deform when falling. Thereby, we do not have to consider moving meshes. The necessary modifications are first in the mesh class, so that also a droplet mesh is generated:

.. code:: python

           # Modification (besides renaming some interfaces): Add a droplet mesh
           self.line(pSnorth,pSsouth,name="droplet_axisymm")
           self.plane_surface("droplet_axisymm", "droplet_outside",name="droplet")  # droplet domain

Then, in the constructor ``__init__`` of the problem class, we add e.g. the viscosity for the droplet as well:

.. code:: python

           # Modifications: Also define viscosity of the droplet
           self.droplet_radius = 0.25 * milli * meter  # radius of the spherical object
           self.droplet_density = 1200 * kilogram / meter ** 3  # density of the sphere
           self.outer_density = 1000 * kilogram / meter ** 3  # density of the liquid
           self.droplet_viscosity = 5 * milli * pascal * second  # viscosity
           self.outer_viscosity = 1 * milli * pascal * second  # viscosity
           self.surface_tension=50*milli*newton/meter # Also define a surface tension (although it does not matter on a static mesh)

In the :py:meth:`~pyoomph.generic.problem.Problem.define_problem` method, we remove the no-slip boundary condition and add the equations of the droplet:

.. code:: python

           # Do not add a no slip
           # eqs += DirichletBC(velocity_x=0, velocity_y=0) @ "droplet_outside"  # and no-slip on the object

           # droplet equations
           inertia_correction = self.droplet_density * vector([0, 1]) * partial_t(U)
           deqs=MeshFileOutput()
           deqs+=NavierStokesEquations(dynamic_viscosity=self.droplet_viscosity,mass_density=self.droplet_density,bulkforce=inertia_correction)
           deqs+=DirichletBC(velocity_x=0)@"droplet_axisymm"
           deqs+=NavierStokesFreeSurface(surface_tension=self.surface_tension,static_interface=True)@"droplet_outside"
           deqs+=ConnectVelocityAtInterface()@"droplet_outside"
           deqs += DragContribution(U) @ "droplet_outside"  # Also add the tractions from the inside to the drag
           self.add_equations(deqs@"droplet")


   if __name__ == "__main__":
       with FallingDropletProblem() as problem:
           problem.run(0.5*second,startstep=0.05*second,outstep=True)  # solve and output


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <falling_droplet.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   	
		   

When we have static meshes (i.e. no equations for the mesh positions are added), we must add ``static_interface=True`` to the :py:class:`~pyoomph.equations.navier_stokes.NavierStokesFreeSurface`. With that, the action of the Lagrange multiplier of the kinematic boundary condition :math:numref:`eqalekinbcweak` will be added to the velocity, not the mesh motion. Since the latter is not allowed to move, only an adjustment of the velocity can guarantee the kinematic boundary condition to hold. Thereby, the kinematic boundary condition is effectively replaced by a zero normal flow condition, i.e. :math:numref:`eqspatialnofluxlagrange`.

The results of these modifications are shown in :numref:`figmultidomfallingdroplet`. The value of the surface tension does not matter here, since on static meshes, the droplet will remain perfectly spherical, i.e. corresponding to a zero capillary number. However, surface tension gradients can still drive Marangoni flow, as we will see next.

To obtain surface tension gradients, we add an insoluble surfactant to the droplet-outside interface. The corresponding transport equation for a surfactant interface concentration :math:`\Gamma` with surface diffusivity :math:`D_S` reads

.. math:: :label: eqmultidomsurftransport

   \partial_t \Gamma+\nabla_S\cdot\left(\vec{u}\Gamma\right)=\nabla_S\cdot\left(D_S\nabla_S \Gamma\right)

In pyoomph, it is again trivial to implement this:

.. code:: python

   from falling_droplet import *
   from pyoomph.expressions.phys_consts import *


   class SurfactantTransportEquation(InterfaceEquations):
       def __init__(self):
           super(SurfactantTransportEquation, self).__init__()
           self.D=1e-9*meter**2/second # diffusivity

       def define_fields(self):
           self.define_scalar_field("Gamma","C2",testscale=scale_factor("temporal")/scale_factor("Gamma"))

       def define_residuals(self):
           u=var("velocity") # velocity at the interface
           G,Gtest=var_and_test("Gamma")
           self.add_residual(weak(partial_t(G)+div(u*G),Gtest))
           self.add_residual(weak(self.D*grad(G), grad(Gtest)))

Again, if we solve it on an interface, i.e. a manifold with co-dimension 1, :py:func:`~pyoomph.expressions.div` and :py:func:`~pyoomph.expressions.generic.grad` will expand to their surface counterparts as discussed in :numref:`secspatialhelicalmesh`.

.. warning::

   This transport equation is only valid without mass transfer. When mass transfer is considered, the normal interface motion would not coincide with the normal interface velocity. Thereby, this equation must be changed.

The only thing we have to do is adding this equation to the interface, setting a suitable initial condition and scaling and modifying the surface tension so that it depends on :math:`\Gamma`. The simplest surface tension relation with surfactants is just :math:`\sigma(\Gamma)=\sigma_0-RT\Gamma` with the gas constant :math:`R` and temperature :math:`T`. Instead of modifying the problem class directly, we just add the :py:attr:`~pyoomph.generic.problem.Problem.additional_equations` and modify the ``surface_tension`` in the run script of the simulation:

.. code:: python

   if __name__ == "__main__":
       with FallingDropletProblem() as problem:
           Gamma0=1*micro*mol/meter**2
           problem.set_scaling(Gamma=Gamma0)

           add_ieqs=SurfactantTransportEquation()+InitialCondition(Gamma=Gamma0)+MeshFileOutput()
           problem.additional_equations+=add_ieqs@"droplet/droplet_outside"

           T=20*celsius
           problem.surface_tension=50*milli*newton/meter-gas_constant*T*var("Gamma")

           problem.run(0.5*second,startstep=0.05*second,outstep=True)  # solve and output

The surfactants get advected to the top of the droplet and hamper the flow in the droplet, cf. :numref:`figmultidomfallingdroplet`.


..  figure:: falling_droplet.*
	:name: figmultidomfallingdroplet
	:align: center
	:alt: Droplet falling due to gravity without and with surfactants
	:class: with-shadow
	:width: 100%

	Droplet falling due to gravity without (left) and with surfactants (right).


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <falling_droplet_with_surfactants.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   	
		   
