.. _secODEfloquet:

Stability of orbits
~~~~~~~~~~~~~~~~~~~

Like stationary solutions, periodic orbits can exhibit different kinds of stability.
In the discussed Lorenz system, we only found unstable orbits, but how do we prove that they are indeed unstable?
As usual, stability can be defined by the linearized dynamics around the state of interest, here the orbit.

Recalling :math:numref:`orbitdefM`, a linearization around the orbit :math:`\vec{x}(t)` corresponds to the solution

.. math:: :label: orbitdeflin

	\begin{aligned}
	\mathbf{M}(\vec{x})\partial_t\vec{v}(t)+\mathbf{J}_0(\vec{x})\vec{v}&=0 \\
	\vec{v}(t+T)&=\lambda\vec{v}(t)
	\end{aligned}
	
Here, :math:`\mathbf{J}_0(\vec{x})` is the Jacobian of the time-independent residual :math:`\vec{R}_0(\vec{x})` and :math:`\vec{v}(t)` is the linear perturbation of the orbit.
After one period, we do not necessarily end up at the very same point, i.e. in general :math:`\vec{v}(t+T)\neq \vec{v}(t)`. Instead, we generically end up at another point, which is expressed by the so-called *Floquet multiplier* :math:`\lambda`. *Floquet theory* tells us that such solutions :math:`\vec{v}(t)` with corresponding values of :math:`\lambda` exist, which resemble the conventional eigenvectors and eigenvalues used for the stability of stationary solutions, although there are some differences as detailed in the following. Due to linearity, a second turn along the orbit will end up at :math:`\vec{v}(t+2T)=\lambda^2 \vec{v}(t)` and (as long as the perturbation remains small) a multiplier of :math:`\lambda^n` after :math:`n` periods. Obviously, an orbit is stable if all Floquet multipliers (which are in general complex-valued) satisfy :math:`|\lambda|<1`. However, there is always a Floquet multiplier :math:`\lambda=1` present in the system, which corresponds to the shift invariance in time. The corresponding perturbation is just :math:`\vec{v}=\partial_t\vec{x}`, i.e. if we just start our perturbation somewhere else in time, we will end up shifted as well after a period of :math:`T`.

Pyoomph can calculate Floquet multipliers, which is demonstrated in the following on the basis of the Langford ODE system. This one reads with suitable parameters :cite:`Gesla2024`:

.. math:: :label: langfordODE

	\begin{aligned}	 
         \partial_t x&=(\mu-3)x-\frac{1}{4}y+x\left(z+\frac{1}{5}(1-z^2)\right) \\
         \partial_t y&=(\mu-3)y+\frac{1}{4}x+y\left(z+\frac{1}{5}(1-z^2)\right) \\
         \partial_t z&=\mu z-\left(x^2+y^2+z^2\right)
	\end{aligned}

For :math:`\mu>1.683`, perfectly circular orbits can be found which change the stability from stable to unstable at :math:`\mu=2` :cite:`Gesla2024`.

Implementing the equation and setting up the problem is again trivial:

.. literalinclude:: langford_floquet.py
   :language: python
   :start-at: #See https://arxiv.org/abs/2407.18230v1
   :end-at: return multiplier

Note how we also provide a function to calculate the analytical nontrivial Floquet multiplier :cite:`Gesla2024`, i.e. a complex Floquet multiplier which is not the trivial one :math:`\lambda=1`. 
		        
We will then again find the Hopf bifurcation, switch to the orbit and continue in the parameter :math:`\mu`. But at each continuation step, we also calculate the Floquet multipliers and write the non-trivial one (along with the corresponding analytical solution) to the output:

.. literalinclude:: langford_floquet.py
   :language: python
   :start-at: with LangfordProblem() as problem:
   :end-at: floquet_output.add_row(problem.mu, nontrivial_floquet.real,nontrivial_floquet.imag,floq_ana.real,floq_ana.imag)

Floquet multipliers can be calculated via the method :py:meth:`~pyoomph.generic.problem.PeriodicOrbit.get_floquet_multipliers` of the :py:class:`~pyoomph.generic.problem.PeriodicOrbit` class, which by default uses the structured condensation proposed in Ref. :cite:`Fairgrieve1991`. Since the orbit is discretized by collocation in time, the Jacobian of the orbit is block bidiagonal: each element of the time discretization couples only the time points it spans. Condensing each of these elements to a single transfer matrix, and multiplying the transfer matrices along the orbit, gives the monodromy matrix, whose eigenvalues are exactly the Floquet multipliers. There are hence always as many multipliers as the system has degrees of freedom, no matter how finely the orbit is discretized in time, and none of them has to be discarded. We only have to remove the trivial multiplier :math:`\lambda=1` and select the interesting one to write it to the output. As depicted in :numref:`figfloquetslangford`, the results agree well with the analytical Floquet multiplier.

Note that the mass matrix :math:`\mathbf{M}` may be arbitrary, and in particular singular, e.g. when the system contains algebraic constraints (such as the incompressibility constraint of the Navier-Stokes equations). This is a consequence of never inverting :math:`\mathbf{M}` on its own: it only ever enters the condensation combined with :math:`\mathbf{J}_0`. However, Gauss-Legendre collocation is not stiffly accurate, so a purely algebraic direction does not decay to a zero multiplier: it comes out at exactly :math:`(-1)^{N}`, where :math:`N` is the number of time intervals of the orbit. With an odd number of intervals, that is a spurious multiplier at :math:`-1`, i.e. exactly where a period-doubling bifurcation would be located. This is a property of the time discretization rather than of the multiplier calculation, but it is worth keeping in mind when interpreting the multipliers of a system with algebraic constraints.

Both the orbit tracking and the Floquet multipliers also work under MPI, with and without ``--distribute``. Two things around them do not (yet): :py:meth:`~pyoomph.generic.problem.Problem.switch_to_hopf_orbit` on a distributed problem, since the first Lyapunov coefficient it requires is not calculated in parallel, and the automatic setup of the time history when leaving the ``with`` block of an orbit, which would continue the orbit by a transient simulation. On a distributed problem, you therefore have to construct the orbit guess yourself and pass it to :py:meth:`~pyoomph.generic.problem.Problem.activate_periodic_orbit_handler`.

For very large systems, where the monodromy matrix is too large to be assembled, only the dominant multipliers - which are the relevant ones for the stability - are calculated by applying the monodromy matrix in a matrix-free manner. Two further methods can be selected explicitly: ``method="periodic_schur"`` calculates the same multipliers by a periodic Schur decomposition, i.e. without ever multiplying the transfer matrices, which is considerably more accurate for multipliers many orders of magnitude below the dominant one (at a higher cost, and only for moderately sized systems), and ``method="eigenproblem"`` selects the older, unstructured formulation, which solves a single large eigenproblem over all time points at once and filters the result.

..  figure:: floquets.*
    :name: figfloquetslangford
    :align: center
    :alt: Floquet multipliers of the Langford ODE system
    :class: with-shadow
    :width: 50%
    
    Floquet multipliers of the Langford ODE system

.. only:: html

   .. container:: downloadbutton

      :download:`Download this example <langford_floquet.py>`
      
      :download:`Download all examples <../../tutorial_example_scripts.zip>`
      

Since the Floquet multipliers at :math:`\mu=2` cross the stability condition :math:`|\lambda|=1` by a complex-conjugated pair, this corresponds to a Neimark-Sacker bifurcation. The orbit becomes unstable to a torus. We can check this by performing time integration. The moment we leave the ``with`` statement of the ``orbit``, pyoomph will initialize the degrees of freedom to the starting point of the orbit. A trivial :py:meth:`~pyoomph.generic.problem.Problem.run` statement will then perform a time integration along the orbit. However, if we start at :math:`\mu>2` (here e.g. :math:`\mu=2.005`), it will be unstable and we can see the torus developing. We just have to replace the orbit loop (i.e. the code after solving for the Hopf bifurcation) by:

.. literalinclude:: langford_time_integration.py
   :language: python
   :start-at: with problem.switch_to_hopf_orbit(NT=50,order=3) as orbit:
   :end-at: problem.run(40*T,outstep=dt/4)

..  figure:: torus_unstable.*
    :name: figlangfordtorus
    :align: center
    :alt: Stable orbits and time integration at unstable dynamics building a torus
    :class: with-shadow
    :width: 90%
    
    Stable orbits (color-coded by :math:`\mu`) and time integration at :math:`\mu=2.005` (black) showing the unstable dynamics building a torus. Also, the path of the Floquet multipliers as a function of :math:`\mu` is shown.

    
.. only:: html

   .. container:: downloadbutton

      :download:`Download this example <langford_time_integration.py>`
      
      :download:`Download all examples <../../tutorial_example_scripts.zip>`
