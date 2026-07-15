.. _secODEhopfswitch:

From Hopf bifurcations to periodic orbits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let us consider again the Lorenz system. From the example :numref:`sectemporalbiftrack`, we know that this system exhibits Hopf bifurcations. We therefore know what periodic orbits exist, at least in the vicinity of these bifurcations. However, a linear stability analysis cannot reveal whether the Hopf bifurcation is super- or subcritical, since this requires the knowledge of the so-called first Lyapunov coefficient (don't confuse it with the Lyapunov exponent of the previous page). The first Lyapunov coefficient :math:`c_1` appears in the normal form of the Hopf bifurcation, which reads

.. math:: :label: hopfnflyapcoeff

	\begin{aligned}
	\dot z=( p + i + c_1 |z|^2)z
	\end{aligned}

Here, :math:`z` is a single complex unknown, :math:`p` is a real parameter and :math:`c_1` is, as mentioned before, the first Lyapunov coefficient. If :math:`\mathrm{Re}(c_1)<0`, stable periodic orbits appear for :math:`p>0` with an amplitude :math:`\sqrt{-p/\mathrm{Re}(c_1)}`. The Hopf bifurcation is supercritical. Otherwise, if :math:`\mathrm{Re}(c_1)>0`, unstable orbits are present for :math:`p<0` and the bifurcation is subcritical.

Close to the Hopf bifurcation, the normal form :math:numref:`hopfnflyapcoeff` and the actually considered system are equivalent. This can be utilized to calculate :math:`c_1` also for Hopf bifurcations in general systems and thereby classify the type of the Hopf bifurcation. The calculation is quite intricate, but straightforward. Details can be found here :cite:`Kuznetsov2023`, but note that pyoomph generalizes the approach presented in the book :cite:`Kuznetsov2023` by the presence of an arbitrary mass matrix.

Additionally, we can calculate the amplitude of the orbit and thereby construct an initial guess for the orbit which can subsequently be solved and followed along the bifurcation parameter. The definition of the Lorenz system and the problem class is analogous to :numref:`sectemporalbiftrack`, so we just focus on the orbit tracking here. The code is actually really short:

.. literalinclude:: hopf_switch.py
   :language: python
   :start-at: with LorenzProblem() as problem:

As a first step, we must invoke the code generator to derive the Hessian and generate C code to fill the Hessian. We need it later, when we want to calculate the first Lyapunov coefficient :math:`c_1`. As detailed in Ref. :cite:`Kuznetsov2023`, the calculation of :math:`c_1` requires directional derivatives of :math:`\vec{R}_0` of second and third order around the Hopf bifurcation point. With the Hessian, we can calculate the second order directional derivatives fully symbolically, whereas the third order directional derivative is calculated by first order finite differences of the symbolically calculated second order derivatives.

We then find the Hopf bifurcation by first starting nontrivially near the Hopf and solving the problem. We will end up on one of the pitchfork branches. Then, solving the eigenproblem gives a good guess for the subsequently invoked Hopf bifurcation tracking. We will thereby locate the Hopf bifurcation and the critical parameter :math:`\rho`. 

Once this is done, we can activate the orbit tracking by :py:meth:`~pyoomph.generic.problem.Problem.switch_to_hopf_orbit`. As arguments, we can pass e.g. the number of points to use, the time interpolation mode, whether a phase or a plane constraint should be used to remove the shift invariance in time and simultaneously constitute an equation for the unknown period :math:`T`. The main interpolation modes are ``"collocation"`` (default), which performs the orthogonal collocation method as e.g. used in AUTO :cite:`Fairgrieve1991`, and ``"bspline"``, which samples the orbit by periodic B-splines. As long as we stay in the ``with``-statement, orbit tracking is activated. Once we leave it, it will be deactivated with suitable history conditions to perform conventional time integration via the :py:meth:`~pyoomph.generic.problem.Problem.run` command afterwards. The returned ``orbit`` object provides several methods to inspect the orbit. In particular, we can ask whether the Hopf bifurcation is supercritical (i.e. :math:`c_1<0`) by the method :py:meth:`~pyoomph.generic.problem.PeriodicOrbit.starts_supercritically`. Here, it is not, meaning that the orbits - at least close to the Hopf bifurcation - are unstable and therefore cannot be found by conventional time integration. We can obtain the period :math:`T` by the method :py:meth:`~pyoomph.generic.problem.PeriodicOrbit.get_T`. Likewise, we can output the orbit (which will be written in a subdirectory of the output directory) by the :py:meth:`~pyoomph.generic.problem.PeriodicOrbit.output_orbit` method. Continuation in the parameter :math:`\rho` works as before, but we do not really know any good initial step for the arclength continuation. In particular, super- and subcritical Hopf bifurcations must continue in a different direction, since orbits only exist in one direction. However, :py:meth:`~pyoomph.generic.problem.Problem.switch_to_hopf_orbit` already calculates a reasonable step, which is available via :py:meth:`~pyoomph.generic.problem.PeriodicOrbit.get_init_ds`. 

Eventually, a plot as shown in :numref:`fighopforbitslorenz` can be obtained, visualizing the orbits as a function of the parameter :math:`\rho`. With time integration, these orbits cannot be found due to their unstable nature.


..  figure:: switch_hopf.*
    :name: fighopforbitslorenz
    :align: center
    :alt: Orbits originating from the Hopf bifurcations in the Lorenz system
    :class: with-shadow
    :width: 100%
    
    Orbits originating from the Hopf bifurcations in the Lorenz system. The color-code indicates the value of :math:`\rho`. Thick lines represent the stable pitchfork branches, which become unstable in a subcritical Hopf bifurcation. By the orbit tracking, we can calculate the unstable orbits which cannot be done by time integration.
    
.. only:: html

   .. container:: downloadbutton

      :download:`Download this example <hopf_switch.py>`
      
      :download:`Download all examples <../../tutorial_example_scripts.zip>`
