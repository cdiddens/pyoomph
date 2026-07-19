.. _secALEstatdroplet:

Stationary solutions of a dimensional droplet with Marangoni flow and gravity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lastly, we want to check how a droplet with an equilibrium contact angle behaves under the influence of gravity and Marangoni flow. Opposed to the previous cases, this time we consider physical dimensions and calculate stationary solutions, which will reveal some pitfalls to carefully consider. We will make use of the predefined equations, which essentially do the same as e.g. the ``KinematicBC``, ``DynamicBC`` and ``SlipLength`` considered here, but allow for physical dimensions. We could implement it by hand into our versions of these equations, but this has been laid out in other examples in detail (cf. e.g. sections :numref:`secodephysdims`, :numref:`secspatialstokesdim`, :numref:`secspatialstokes_law` and :numref:`secpdemarainstab`). For simplicity, we also revert to the axisymmetric variant.

The first problem when looking for stationary solutions is the fact that the problem is non-linear, mainly due to the mesh motion. Hence, one must have a good initial guess in order for the problem to converge with a stationary solve. Since we do not know the shape *a priori*, we start with a droplet at zero gravity, zero Marangoni flow and a contact angle of :math:`90\:\mathrm{^\circ}`. For these parameters, we know that we just will have a hemisphere with vanishing velocity. We will then crank up first the gravity, then the contact angle to :math:`120\:\mathrm{^\circ}` and finally activate the Marangoni flow. To that end, we introduce three parameters to the system:

.. literalinclude:: droplet_spread_marangoni_and_gravity.py
   :language: python
   :start-at: from pyoomph import * # main pyoomph
   :end-at: self.gravity=9.81*meter/second**2 # overall gravity when param_gravity_factor.value=1

We use dimensional values for the volume and the fluid parameters. The gravity will be blended in by the parameter ``param_gravity_factor``. When this parameter reaches :math:`1`, we will get a downwards acting gravity of 9.81\ :math:`\:\mathrm{m} / \mathrm{s^2}`. The contact angle will be controlled by the global parameter ``param_contact_angle``, whose ``value`` is initialized to :math:`90\:\mathrm{^\circ}`. Finally, ``param_sigma_gradient`` will activate Marangoni flow by letting the surface tension be

.. math:: \sigma=\sigma_0\cdot\left(1+\sigma_1\left(\frac{x}{R_0}\right)^2\right)\,,

where :math:`\sigma_1` is the parameter ``param_sigma_gradient`` and :math:`R_0` is the radius of the initial hemispherical droplet.

The :py:meth:`~pyoomph.generic.problem.Problem.define_problem` method starts as follows:

.. literalinclude:: droplet_spread_marangoni_and_gravity.py
   :language: python
   :start-at: def define_problem(self):
   :end-at: eqs+=NavierStokesContactAngle(self.param_contact_angle,wall_normal=vector(0,1),wall_tangent=vector(-1,0))@"interface/substrate"

Until here, it is quite trivial if you have read the previous examples. Note how we use e.g. ``param_gravity_factor`` directly to get the symbolic parameter, i.e. not its ``value`` but its symbolic value, that will be replaced by its instantaneous value during each calculation. The refinement :py:class:`~pyoomph.equations.generic.RefineToLevel` of the initial mesh will be controlled by :py:attr:`~pyoomph.generic.problem.Problem.initial_adaption_steps`. We will later set this to a small refinement level to approximate a good guess for the initial condition of the stationary solution we are looking for on a coarse mesh. This will drastically improve the computation time. The :py:class:`~pyoomph.equations.navier_stokes.NavierStokesFreeSurface` and ``NavierStokesContactAngle`` are the predefined equivalents to the combination of ``KinematicBC`` and ``DynamicBC`` of the free surface and the ``EquilibriumContactAngle`` of the contact line developed in this chapter.

Now, we come to the most intricate problem: When solving for stationary solutions, the volume of the droplet will not be conserved at all. The reason lies in the implementation of the ``KinematicBC``. In this, we let the mesh follow the normal fluid motion. However, it involves the time derivative of the mesh coordinates, but in stationary solves, all temporal derivatives are zero. Hence, there is no guarantee at all that the volume would be conserved or even that a solution is actually found. We therefore must enforce the volume by a single Lagrange multiplier. Therefore, we will add a new ODE domain containing just the single Lagrange multiplier :math:`\lambda`, which will enforce the volume constraint by virtue of adding the Lagrange multiplier contribution

.. math:: :label: eqalevolconstr

   \lambda\left(\int_{\text{drop}} 1\:\mathrm{d}V -V_0\right)\,.

This has to be split into two contributions, namely one integral over the domain and the constant offset of the negative initial volume :math:`-V_0`. The latter part can be done as follows:

.. literalinclude:: droplet_spread_marangoni_and_gravity.py
   :language: python
   :start-at: # The volume is not conserved during stationary solving
   :end-at: self.add_equations(vol_constr_eqs@"volume_constraint") # add it to an "ODE" domain called "volume_constraint"

The :py:class:`~pyoomph.generic.codegen.GlobalLagrangeMultiplier` will just define a Lagrange multiplier with the name ``volume_lagrange`` and considering the offset :math:`-V_0` to the weak form. Since we have a dimensional problem, we also have to set the scaling of ``volume_lagrange`` and for the corresponding test function. This can be changed by adding ``Scaling`` and ``TestScaling`` objects to the ODE equation. One might wonder why we choose a scale of the pressure for the Lagrange multiplier, but this will become clear in a minute. Finally, the Lagrange multiplier is added to the ODE domain ``"volume_constraint"``.

There is still the integral portion missing. Since it involves an integral over the droplet ``"domain"``, we have to add it to the equations on ``"domain"``. To that end, we use :py:class:`~pyoomph.generic.codegen.WeakContribution`:

.. literalinclude:: droplet_spread_marangoni_and_gravity.py
   :language: python
   :start-at: # bind the volume enforcing Lagrange multiplier
   :end-at: eqs+=WeakContribution(1,testfunction(vol_constr_lagr),dimensional_dx=True)

The :py:class:`~pyoomph.generic.codegen.WeakContribution` will just add the integral :math:`\int 1\:\mathrm{d}V` to the test space of the global Lagrange multiplier :math:`\lambda`, i.e. accounting for the first term in :math:numref:`eqalevolconstr`. Note the ``dimensional_dx=True`` argument, which carries out the integral with physical dimensions, i.e. not in non-dimensional coordinates. This is required to get a dimensional volume contribution in :math:`\:\mathrm{m^3}`.

So far, so good. The volume constraint is assembled correctly. But there is no feedback yet on the droplet. We must impose the value of the Lagrange multiplier somewhere back to the system, otherwise it is just redundant. The idea is to add an additional pressure contribution to the droplet, which is proportional to the Lagrange multiplier. This can be understood easily if the initially hemispherical droplet without any gravity or flow is considered. As argued before, in absence of the additional Lagrange multiplier, the volume is not conserved and hence, there is an infinite number of solutions possible, namely for each volume :math:`V` we get a unique Laplace pressure due to the different curvature. When we add the contribution of the Lagrange multiplier to the pressure, the system is only in agreement if the volume :math:`V` is indeed the volume :math:`V_0` and the corresponding Laplace pressure in the droplet is the Laplace pressure corresponding to this volume. If the volume :math:`V` deviates from :math:`V_0`, the additional contribution to the pressure from the non-zero Lagrange multiplier will alter the pressure throughout the droplet and forces the droplet to attain the volume :math:`V_0` with the corresponding Laplace pressure during the solution procedure.

We hence add the Lagrange pressure as additional normal traction, i.e. analogous to the Laplace pressure stemming from the surface tension and curvature.

.. literalinclude:: droplet_spread_marangoni_and_gravity.py
   :language: python
   :start-at: # Finally, the Lagrange multiplier must also yield feedback to the problem
   :end-at: self.add_equations(eqs@"domain")

The problem definition ends with a :py:class:`~pyoomph.equations.generic.SpatialErrorEstimator` for the velocity, which will be relevant when Marangoni flow is activated at the end.

..  figure:: droplet_mara_grav.*
	:name: figaledropletmaragrav
	:align: center
	:alt: Droplet spreading with gravity and Marangoni flow
	:class: with-shadow
	:width: 100%

	Starting with a coarse hemispherical mesh, we first crank up the gravity, then modify the contact angle and finally solve for the Marangoni flow with refinement.


Now, we can use the solution procedure, i.e. starting with a coarse hemispherical droplet, cranking up gravity, subsequently adjusting the contact angle and finally blending in the Marangoni flow. Only at the final step, we do mesh refinement, since the other steps were just made to approach a good initial guess for the final solution:

.. literalinclude:: droplet_spread_marangoni_and_gravity.py
   :language: python
   :start-at: if __name__=="__main__":

The method :py:meth:`~pyoomph.generic.problem.Problem.go_to_param` will gradually increase/decrease the specified parameter (identified by the name passed to :py:meth:`~pyoomph.generic.problem.Problem.get_global_parameter` in the :py:class:`~pyoomph.generic.problem.Problem` constructor) to the desired value via arc length continuation. With the ``startstep`` argument, we can optionally set the first try of the parameter increment. If not set, it will try to directly reach the desired parameter value, which is not always successful and results in a lot of unsuccessful tries followed by a retry with a reduced step. The ``final_adaptive_solve`` is set to ``False`` in the first two parameter increases. This means we do not adapt after the parameter value has been reached. At the very end, we set it to ``True`` to ultimately adapt the final result to maximum accuracy, selected by :py:attr:`~pyoomph.generic.problem.Problem.max_refinement_level`. The results are depicted in :numref:`figaledropletmaragrav`. The problem would never converge if one directly tries to solve for the last step starting with the hemispherical initial mesh.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <droplet_spread_marangoni_and_gravity.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    


.. note::

   We have enforced the volume in :math:numref:`eqalevolconstr` by integrating :math:`1` over the droplet domain. While this is definitely acceptable, the volume contribution of each element depends on all nodal positions of the element. This leads to a lot of computational costs for the assembly of the Jacobian and also for the solution of the latter.

   Alternatively, we can use a trick so that only the nodal positions on the boundaries contribute. To that end, first note that :math:`\nabla\cdot \vec{x}=d`, where :math:`d` is the number of dimensions (in axisymmetric coordinates, it will be :math:`3` as well). We can hence write the contribution :math:`(1,\eta)_\text{drop}` (with :math:`\eta` being the testfunction of :math:`\lambda`) as :math:`1/d\:(\nabla\cdot\vec{x},_\eta)_\text{drop}`. Now, we can apply the divergence theorem to obtain :math:`1/d\:(\nabla\cdot\vec{x},_\eta)_\text{drop}=1/d\:\langle  \vec{x}\cdot\vec{n},\eta \rangle`, where the interface integral has to be applied to all interfaces. However, at the substrate and at the axis of symmetry, :math:`\vec{x}\cdot\vec{n}` is zero, so that only the free surface contributes.

   Instead of ``WeakContribution(1,testfunction(vol_constr_lagr),dimensional_dx=True)``, we could hence define ``x_dot_n=dot(var("coordinate"),var("normal"))`` and add ``WeakContribution(1/3*x_dot_n,testfunction(vol_constr_lagr),dimensional_dx=True)@"interface"``.

   This will speed up the assembly and the solving.
