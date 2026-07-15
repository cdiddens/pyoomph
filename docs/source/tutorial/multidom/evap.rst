.. _secmultidomdropevap:

Evaporation of a sessile droplet
--------------------------------

Evaporation, as mass transfer between phases in general, always requires more than a single phase. We have seen how we can couple different equations on individual domains and hence we can solve evaporating sessile droplets. To that end, we have to solve the Navier-Stokes equation for the droplet, the interface dynamics and for the gas phase, the diffusion equation of the vapor field. While there is actually also advective vapor transport, it is usually negligible for temperatures below the boiling point.

First of all, we must have a mesh that comprises a sessile droplet and a surrounding gas domain:

.. literalinclude:: evaporating_water_droplet.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: self.plane_surface("gas_substrate","gas_infinity","gas_axisymm","droplet_gas",name="gas") # gas domain

Again, we use the :py:class:`~pyoomph.meshes.gmsh.GmshTemplate` for that to create a mesh. We refine it near the contact line and make it coarser in the far field of the gas domain by adding the ``size`` keyword to the :py:meth:`~pyoomph.meshes.gmsh.GmshTemplate.point` calls. The droplet-gas interface is made by a :py:meth:`~pyoomph.meshes.gmsh.GmshTemplate.circle_arc` starting at the contact line and ending at the droplet apex. Instead of passing the ``center`` of the circle, we can also pass ``through_point``, i.e. a third point which is also located on the circle (but potentially outside the segment). Here, we just use the mirrored contact line.

Next, for the problem class, we must define a few default properties:

.. literalinclude:: evaporating_water_droplet.py
   :language: python
   :start-at: class EvaporatingDroplet(Problem):
   :end-at: self.contact_angle=None # will be calculated from the height and radius if not set

In the gas phase, we will solve the vapor diffusion equation for the partial vapor mass density :math:`c` (``"c_vap"`` in python) with units :math:`\:\mathrm{kg}/\mathrm{m^3}` with saturated vapor :math:`c=c_\text{sat}` at the liquid-gas interface and ambient vapor :math:`c_\infty` far away. Then, the diffusive flux at the interface, i.e. :math:`j=-\nabla c\cdot \vec{n}` is the evaporation rate, i.e. the mass transfer per area and time, i.e. in :math:`\:\mathrm{kg}/\mathrm{m^2} \cdot \mathrm{s})`. Therefore, we bind it:

.. literalinclude:: evaporating_water_droplet.py
   :language: python
   :start-at: # Bind the evaporation rate
   :end-at: self.evap_rate = -self.vapor_diffusivity * dot(grad(c_vap), n)

It is again important to tell :py:func:`~pyoomph.expressions.generic.var` that we want to evaluate ``"c_vap"`` in the ``domain="gas"``, i.e. in the bulk domain. Otherwise, we might get the wrong gradient (i.e. the surface gradient instead of the bulk gradient).

In the :py:meth:`~pyoomph.generic.problem.Problem.define_problem`, we first set some reasonable scales for non-dimensionalization and add the mesh:

.. literalinclude:: evaporating_water_droplet.py
   :language: python
   :start-at: def define_problem(self):
   :end-at: self.contact_angle= DropletGeometry(base_radius=self.droplet_radius, apex_height=self.droplet_height).contact_angle

We also calculate the equilibrium ``contact_angle`` if not set explicitly. This is used only if ``pinned_contact_line`` is ``False``.

The droplet bulk equations are just Navier-Stokes with a moving mesh along with a free surface at the liquid-gas interface, a slip length condition at the substrate and a few :py:class:`~pyoomph.meshes.bcs.DirichletBC` terms:

.. literalinclude:: evaporating_water_droplet.py
   :language: python
   :start-at: # Droplet equations
   :end-at: d_eqs += ConnectMeshAtInterface() @ "droplet_gas" # connect the gas mesh to co-move

Note that we pass ``mass_transfer_rate=evap_rate`` to the :py:class:`~pyoomph.equations.navier_stokes.NavierStokesFreeSurface`. This augments the kinematic boundary condition :math:numref:`eqalekinbcstrong` as follows:

.. math:: :label: eqmultidomkinbcevapstrong

   \begin{aligned}
   \vec{n}\cdot\left(\vec{u}-\dot{\vec{x}}\right)=\frac{j}{\rho}\,.
   \end{aligned}

Here, :math:`j` is the ``mass_transfer_rate``, i.e. bound to :math:`j=-\nabla c\cdot \vec{n}` in this particular problem and :math:`\rho` is the density of the :py:class:`~pyoomph.equations.navier_stokes.NavierStokesEquations`, i.e. of the droplet. Thereby, the relative velocity between the liquid and the interface in normal direction is given by the lost mass.

For the contact line dynamics, we have two options, depending on the value of ``pinned_contact_line``. If it is ``False``, i.e. a free contact line, we just use the ``NavierStokesContactAngle`` as discussed in :numref:`secALEdropspread`. If the contact line is pinned, i.e. ``pinned_contact_line=True``, we have a problem: The free surface solves the augmented kinematic boundary condition :math:numref:`eqmultidomkinbcevapstrong` by adjusting the mesh positions (cf. :math:numref:`eqalekinbcweak`). However, if the contact line is pinned, the mesh should not move directly at the contact line. In order to fulfill both, the kinematic boundary condition with evaporation and the fixed position of the contact line, we adjust the radial velocity at the contact line so that the mesh does not move:

.. literalinclude:: evaporating_water_droplet.py
   :language: python
   :start-at: # Different contact line dynamics
   :end-at: d_eqs += NavierStokesContactAngle(contact_angle=self.contact_angle) @ "droplet_gas/droplet_substrate"  # and constant contact angle

With the :py:class:`~pyoomph.meshes.bcs.EnforcedBC`, the radial velocity is adjusted so that ``partial_t(var("mesh_x"),ALE=False)=mesh_velocity()[0]=0`` holds, i.e. the contact line is stationary. Intrinsically, this is again done by a Lagrange multiplier within the :py:class:`~pyoomph.meshes.bcs.EnforcedBC`. Of course, this only works with a slip length boundary condition at the substrate, not with a no-slip condition. A no-slip condition would remove the possibility to add a traction to the radial velocity here. Both contact line models, i.e. the pinned and the freely moving constant contact angle condition do essentially the same: They impose a traction at the contact line. However, the ``NavierStokesContactAngle`` adds exactly the weak term that is required to attain the prescribed contact angle (cf. :numref:`secALEdropspread`). With the :py:class:`~pyoomph.meshes.bcs.EnforcedBC`, we essentially enforce exactly that contact angle for which the contact line remains stationary.

The gas equations are just a diffusion equation, i.e. an :py:class:`~pyoomph.equations.advection_diffusion.AdvectionDiffusionEquations` without ``wind``, i.e. without any advection:

.. literalinclude:: evaporating_water_droplet.py
   :language: python
   :start-at: # Gas equations
   :end-at: g_eqs+=AdvectionDiffusionInfinity(c_vap=self.c_infty)@"gas_infinity"

Note how we impose saturated vapor strongly at the liquid-gas interface, whereas the ambient vapor is imposed by a :py:class:`~pyoomph.equations.advection_diffusion.AdvectionDiffusionInfinity` equation. This equation mimics an infinite mesh by a Robin boundary condition. This condition can be derived by knowing that in three dimensions, the diffusion equation will follow a :math:`1/r` behavior in the far field (with :math:`r` being the distance from the droplet). Far away from the droplet, the vapor field will hence read :math:`c=c_\infty+(c(R)-c_\infty)R/r` for any reasonably large distance :math:`R` and for :math:`r>R`. Deriving this with respect to :math:`r` and plugging it again into the expression gives the Robin condition

.. math:: c(R)+R\partial_r c(R)=c_\infty\,.

This condition is implemented by the :py:class:`~pyoomph.equations.advection_diffusion.AdvectionDiffusionInfinity` class as weak Neumann contribution.


Finally, we also add some remeshing options which invoke a mesh reconstruction whenever the mesh deforms too strongly and also output the volume and the interface data with evaporation to files:

.. literalinclude:: evaporating_water_droplet.py
   :language: python
   :start-at: # Control remeshing
   :end-at: self.add_equations(d_eqs@"droplet"+g_eqs@"gas")

The run script is trivial and the results are shown in :numref:`figmultidomdropevap`:

.. code:: python

   if __name__=="__main__":
       with EvaporatingDroplet() as problem:
           problem.run(500*second,startstep=10*second,outstep=True,temporal_error=1)



..  figure:: dropevap.*
	:name: figmultidomdropevap
	:align: center
	:alt: Evaporating droplet
	:class: with-shadow
	:width: 100%

	Evaporating droplet with a pinned contact line (left) and with a constant contact angle (right).


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <evaporating_water_droplet.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   	
		      
