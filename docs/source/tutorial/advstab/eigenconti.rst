Continuation of eigenbranches
-----------------------------------

While :py:meth:`~pyoomph.generic.problem.Problem.solve_eigenproblem` can solve for eigenvalues, it is sometimes hard to order them when varying a parameter. All eigenvalues can change, cross each other and it is then cumbersome to disentangle what is happening to each eigenbranch. Similar to bifurcation tracking, it is also possible to track a specific eigenfunction. To that end, we first have to solve for an eigenvalue/-vector pair as an initial guess. Then, we solve an augmented system for the base state and this particular eigenvalue/-vector pair. Upon continuation, we will follow this particular eigenbranch. 

We will discuss this feature on the basis of a liquid bridge with gravity. In the absence of gravity, it is well known that the system undergoes a Rayleigh-Plateau instability at :math:`L/R=2\pi`. But what happens if we add gravity in the axial direction to the system? Since the Rayleigh-Plateau instability sets in at rest, we will ignore the inertia term, i.e. going for Stokes flow. The problem class is quite simple and reads:


.. literalinclude:: eigenbranch_continuation.py
   :language: python
   :start-at: class LiquidBridgeProblem(Problem):
   :end-at: self+=eqs@"domain"

Note how we fix the top of the domain to the parameter :math:`L` via an :py:class:`~pyoomph.meshes.bcs.EnforcedDirichlet`. Unlike a conventional :py:class:`~pyoomph.meshes.bcs.DirichletBC`, the ``mesh_y`` positions are now still degrees of freedom of the system, but enforced via Lagrange multipliers to :math:`L`. This helps for continuation in :math:`L`, since the mesh positions are now considered in the arclength tangent as well. However, since the axial mesh position at the contact line is now a degree of freedom, pyoomph does not automatically pin the Lagrange multiplier for the kinematic boundary condition (see :numref:`secALEfreesurfNS`, where we pin the Lagrange multiplier of the kinematic boundary condition only if all mesh positions are pinned). Therefore, we manually have to pin it, since the system is otherwise overconstrained.

The volume is again enforced by varying the gas pressure, which is added as normal traction by enforcing the average of the kinematic boundary condition Lagrange multipliers (which are normal tractions) to the gas pressure.

Once set up, we can use this problem and solve for the stationary state at the minimum considered length :math:`L`. We store this state, so that we can load it after each branch. To scan an eigenbranch, we first load the start point, solve the eigenproblem for the initial guess and then activate eigenbranch tracking with the desired index of the eigenvalue. By continuation of the length, we can follow this particular eigensolution. At the end of the scan for :math:`\mathrm{Bo}=0`, we again store the base solution. This is then used as a start for other branches with :math:`\mathrm{Bo}\neq 0`. As you can see in the figure below, the presence of gravity leads to a fold bifurcation before the conventional Rayleigh-Plateau instability actually happens. With our approach, we find the other eigenbranches easily.

.. literalinclude:: eigenbranch_continuation.py
   :language: python
   :start-at: with LiquidBridgeProblem() as problem:
   :end-at: create_Bond_curve(0,0,"end2.dump","unstab")

..  figure:: eigenconti.*
	:name: figadvstabeigenconti
	:align: center
	:alt: Eigenbranches of a liquid bridge with gravity
	:class: with-shadow
	:width: 80%

	Eigenbranches of a liquid bridge with gravity. The original Rayleigh-Plateau instability is broken by the presence of gravity. The subcritical pitchfork bifurcation becomes imperfect when gravity is considered.

.. note::

	If you want to find the pitchfork bifurcation using the bifurcation tracking tools (cf. :numref:`sectemporalbiftrack`), you will get some issues here. Since the symmetry broken by the pitchfork bifurcation is not centered around the :math:`x`-axis, the pitchfork won't be found. To overcome this issue, you can just enforce the ``"top"`` boundary to be at ``mesh_y=self.L/2`` and do it the same way with the ``"bottom"`` to ``mesh_y=-self.L/2`` with an :py:class:`~pyoomph.meshes.bcs.EnforcedDirichlet` including the pinning of the ``_kinbc`` at ``"right/bottom"``. If the symmetry broken by the pitchfork is symmetric with respect to the :math:`x`-axis, it works fine.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <eigenbranch_continuation.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   	
		    


