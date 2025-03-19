From Hopf bifurcations to periodic orbits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let us consider again the Lorenz system. From the example :numref:`sectemporalbiftrack`, we know that this system exhibits Hopf bifurcations. We therefore know what periodic orbits exist, at least in the vicinity of these bifurcations. However, a linear stability analysis cannot reveal whether the Hopf bifurcation is super- or subcritical, since this requires the knowledge of the so-called first Lyapunov coefficient (don't mess it up with the Lyapunov exponent of the previous page). The first Lyapunov coefficient :math:`c_1` appears in the normal form of the Hopf bifurcation, which reads

.. math:: :label: hopfnflyapcoeff

	\begin{aligned}
	\dot z=( p + i + c_1 |z|^2)z
	\end{aligned}

Here, :math:`z` is a single complex unknown, :math:`p` is a real parameter and :math:`c_1` is, as mentioned before, the first Lyapunov coefficient. If :math:`\mathrm{Re}(c_1)<0`, stable periodic orbits appear for :math:`p>0` with an amplitude :math:`\sqrt{-p/\mathrm{Re}(c_1)}`. The Hopf bifurcation is supercritical. Otherwise, if :math:`\mathrm{Re}(c_1)>0`, unstable orbits are present for :math:`p<0` and the bifurcation is subcritical.

Close to the Hopf bifurcation, the normal form :math:numref:`hopfnflyapcoeff` and the actually considered system are equivalent. This can be utilized so calculate :math:`c_1` also for Hopf bifurcations in general systems and thereby classify the type of the Hopf bifurcation. The calculation is quite intricate, but straightforward. Details can be found here :cite:`Kuznetsov2023`, but note that pyoomph generalizes the presented approach by the presence of a mass matrix.

Additionally, we can calculate the amplitude of the orbit and thereby construct an initial guess for the orbit which can subsequently be solved and followed along the bifurcation parameter. The definition of the Lorenz system and the problem class is analogously to :numref:`sectemporalbiftrack`, so we just focus on the orbit tracking here. The code is actually really short:

.. code:: python

	with LorenzProblem() as problem:
		# To calculate c_1, we need the Hessian, so the symbolical code must be generated and compiled
		problem.setup_for_stability_analysis(analytic_hessian=True)
		# Add a non-trivial initial condition
		problem+=InitialCondition(x=1,z=24)@"lorenz"
		problem.rho.value=24 # Start close to the Hopf
		
		problem.solve() # Find a stationary solution (will be on one of the pitchfork branches)        
		problem.solve_eigenproblem(n=1) # And get some eigenvalue for the Hopf tracker
		
		problem.activate_bifurcation_tracking("rho","hopf") # Activate the Hopf tracking
		problem.solve() # Find the Hopf bifurcation by adjusting rho
		
		# Since we are on the Hopf bifurcation, we can switch to the orbit
		# We chose NT=100 time points for the orbit
		# The initial period T and the initial guess of the orbit will be calculated automatically
		with problem.switch_to_hopf_orbit(NT=100) as orbit:
		    print("Bifurcation is supercritical: "+str(orbit.starts_supercritically()))
		    print("Period at rho=",problem.rho.value, " is ",orbit.get_T())
		    # This function will write the output along the orbit to a subdirectory in the output directory
		    orbit.output_orbit("orbit_at_rho_{:.4f}".format(problem.rho.value))
		    # Perform continuation in rho
		    # We do not know in which direction we have to go (depends on the nature of the Hopf)
		    # But a good guess including direction can be obtained from the Lyapunov coefficient
		    ds=orbit.get_init_ds() 
		    while problem.rho.value>16:
		        ds=problem.arclength_continuation("rho",ds)
		        print("Period at rho=",problem.rho.value, " is ",orbit.get_T())
		        orbit.output_orbit("orbit_at_rho_{:.4f}".format(problem.rho.value))              
		        
		        
As a first step, we must the code generator to derive the Hessian and generate C code to fill the Hessian. We need it later, when we want to calculate the first Lyapunov coefficient :math:`c_1`. As detailed in Ref. :cite:`Kuznetsov2023`, the calculation of :math:`c_1` requires directional derivatives of :math:`\vec{R}_0` of second and third order around the Hopf bifurcation point. With the Hessian, we can calculate the second order directional derivatives fully symbolically, whereas the third order directional derivative is calculated by first order finite differences of the symbolically calculated second order derivatives.

We then find the Hopf bifurcation by first starting nontrivially near the Hopf and solve the problem. We will end up on one of the pitchfork branches. Then, solving the eigenproblem gives a good guess for the subsequently invoked Hopf bifurcation tracking. We will thereby locate the Hopf bifurcation and the critical parameter :math:`\rho`. 

Once this is done, we can activate the orbit tracking by :py:meth:`~pyoomph.generic.problem.Problem.switch_to_hopf_orbit`. As arguments, we can pass e.g. the number of points to use, the time interpolation mode, whether a phase or a plane constraint should be used to remove the shift invariance in the time and simultaneously constitute an equation for the unknown period :math:`T`. As long as we stay in the ``with``-statement, orbit tracking is activated. Once we leave it, it will be deactivated with suitable history conditions to perform conventional time integration via the :py:meth:`~pyoomph.generic.problem.Problem.run` command afterwards. The returned ``orbit`` object provides several methods to inspect the orbit. In particular, we can ask whether the Hopf bifurcation is supercritical (i.e. :math:`c_1<0`) by the method :py:meth:`~pyoomph.generic.problem.PeriodicOrbit.starts_supercritically`. Here, it is not, meaning that the orbits - at least close to the Hopf bifurcation - are unstable and therefore cannot be found by conventional time integration. We can obtain the period :math:`T` by the method :py:meth:`~pyoomph.generic.problem.PeriodicOrbit.get_T`. Likewise, we can output the orbit (which will be written in a subdirectory of the output directory) by the ~pyoomph.generic.problem.PeriodicOrbit.output_orbit` method. Continuation in the parameter :math:`\rho` works as before, but we do not really know any good initial step for the arclength continuation. In particular, super- and subcritical Hopf bifurcations must continue in a different direction, since orbits only exists in one direction. However, :py:meth:`~pyoomph.generic.problem.Problem.switch_to_hopf_orbit` already calculates a reasonable step, which is available via :py:meth:`~pyoomph.generic.problem.PeriodicOrbit.get_init_ds`. 

Eventually, a plot as shown in :numref:`fighopforbitslorenz` can be obtained, visualizing the orbits as function of the parameter :math:`rho`. With time integration, these orbits cannot be found due to their unstable nature.


..  figure:: switch_hopf.*
    :name: fighopforbitslorenz
    :align: center
    :alt: Orbits originating from the Hopf bifurcations in the Lorenz system
    :class: with-shadow
    :width: 100%
    
    Orbits originating from the Hopf bifurcations in the Lorenz system
    
.. only:: html

   .. container:: downloadbutton

      :download:`Download this example <hopf_switch.py>`
      
      :download:`Download all examples <../../tutorial_example_scripts.zip>`
