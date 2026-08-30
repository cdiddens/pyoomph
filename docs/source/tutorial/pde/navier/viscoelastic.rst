.. _secpdeviscoelastic:

Viscoelastic flow: the log-conformation approach
------------------------------------------------

A polymer solution does not respond to deformation instantaneously: the dissolved chains stretch, store elastic energy and relax back over a characteristic time :math:`\lambda`. The state of the chains is described by the *conformation tensor* :math:`\mathbf{C}`, a symmetric positive definite tensor that equals the identity when the polymer is unstretched. It is carried by the flow and pulled back towards equilibrium,

.. math:: :label: eqviscoelastic

   \stackrel{\triangledown}{\mathbf{C}}\;=\;\partial_t\mathbf{C}+\left(\vec{u}\cdot\nabla\right)\mathbf{C}-\nabla\vec{u}\cdot\mathbf{C}-\mathbf{C}\cdot\left(\nabla\vec{u}\right)^\mathrm{T}\;=\;-\frac{1}{\lambda}g(\mathbf{C})

where the left hand side is the *upper-convected derivative* and the constitutive model supplies :math:`g`. For the Oldroyd-B model :cite:`Oldroyd1950` it is simply :math:`g(\mathbf{C})=\mathbf{C}-\mathbf{I}`. The polymer then contributes an extra stress

.. math::

   \boldsymbol{\tau}_\mathrm{p}=\frac{\eta_\mathrm{p}}{\lambda}\left(\mathbf{C}-\mathbf{I}\right)

which is added to the momentum equation alongside the usual solvent stress. Note that pyoomph's :py:func:`~pyoomph.expressions.generic.upper_convected_derivative` implements exactly the left hand side above, so the conformation equation can be written down almost verbatim.


Solving that equation as it stands works at low elasticity and then fails, usually well before the flow becomes physically interesting. The problem is that :math:`\mathbf{C}` grows *exponentially* in extensional regions, and a polynomial finite element basis approximates an exponential badly. Once the discrete :math:`\mathbf{C}` loses positive definiteness the computation diverges, and no amount of mesh refinement postpones it for long. This is the classical *high Weissenberg number problem*.

Fattal and Kupferman :cite:`FattalKupferman2004` observed that one should not discretise :math:`\mathbf{C}` but its matrix logarithm :math:`\boldsymbol{\Psi}=\log\mathbf{C}`. That variable grows only linearly where :math:`\mathbf{C}` grows exponentially, so a polynomial basis is well suited to it, and :math:`\mathbf{C}=\exp\boldsymbol{\Psi}` is positive definite *by construction* whatever the discrete :math:`\boldsymbol{\Psi}` does. Rewriting the equation in terms of :math:`\boldsymbol{\Psi}` gives

.. math::

   \partial_t\boldsymbol{\Psi}+\left(\vec{u}\cdot\nabla\right)\boldsymbol{\Psi}-\left(\boldsymbol{\Omega}\boldsymbol{\Psi}-\boldsymbol{\Psi}\boldsymbol{\Omega}\right)-2\mathbf{B}+\frac{1}{\lambda}\mathbf{C}^{-1}g(\mathbf{C})=0

where :math:`\boldsymbol{\Omega}` and :math:`\mathbf{B}` are the rotational and extensional parts of :math:`\nabla\vec{u}` in the eigenframe of :math:`\mathbf{C}`. All of that is done for you by :py:class:`~pyoomph.equations.viscoelastic.ViscoelasticEquations`, which is added alongside the ordinary flow equations. The viscosity handed to the latter is then the *solvent* viscosity :math:`\eta_\mathrm{s}`:

.. code:: python

   from pyoomph.equations.viscoelastic import ViscoelasticEquations, OldroydB

   eqs = NavierStokesEquations(dynamic_viscosity=eta_s, mass_density=rho)
   eqs += ViscoelasticEquations(model=OldroydB(), relaxation_time=lam, polymer_viscosity=eta_p)

Besides :py:class:`~pyoomph.equations.viscoelastic.OldroydB`, the models :py:class:`~pyoomph.equations.viscoelastic.Giesekus` :cite:`Giesekus1982`, :py:class:`~pyoomph.equations.viscoelastic.PTT`, :py:class:`~pyoomph.equations.viscoelastic.FENE_CR` and :py:class:`~pyoomph.equations.viscoelastic.FENE_P` are available, and both 2d Cartesian and axisymmetric coordinate systems are supported.


The standard benchmark for viscoelastic flow solvers is creeping flow past a cylinder placed on the centreline of a channel whose half-height is twice the cylinder radius :cite:`Alves2001,Hulsen2005,Claus2013`. Everything is nondimensionalised with the total viscosity :math:`\eta_0=\eta_\mathrm{s}+\eta_\mathrm{p}`, the cylinder radius and the mean inlet velocity, leaving the solvent fraction :math:`\beta=\eta_\mathrm{s}/\eta_0=0.59` and the Weissenberg number :math:`\mathrm{Wi}=\lambda\langle u\rangle/R`. Only the upper half is solved, with a symmetry condition on the centreline:

.. literalinclude:: viscoelastic_cylinder.py
   :language: python
   :start-at: class ConfinedCylinderProblem(Problem):
   :end-at: self += eqs @ "fluid"

Four details are worth pointing out.

The inflow condition has to prescribe not only the velocity but also the *stress* the polymer has in fully developed channel flow, otherwise the solution spends the entire upstream length relaxing towards it. That is what :py:class:`~pyoomph.equations.viscoelastic.ViscoelasticInflowBC` is for: given the same profile that the velocity condition uses, it differentiates it to obtain the local shear rate and pins the log-conformation tensor to the viscometric solution of whatever constitutive model the bulk equations were given. On the symmetry line the shear rate vanishes and :math:`\mathbf{C}` becomes isotropic, which is a degenerate case for the matrix logarithm it goes through; that case is handled. For the models whose viscometric solution is not an elementary expression -- the exponential :py:class:`~pyoomph.equations.viscoelastic.PTT` is the one here -- the same class instead enforces the constitutive equation itself on the inflow boundary, by Lagrange multipliers, which says the same thing without needing a formula for the answer.

The Weissenberg number enters as a :py:meth:`~pyoomph.generic.problem.Problem.define_global_parameter`, because the solution has to be *continued* in it. Starting a stationary solve directly at a larger :math:`\mathrm{Wi}` overshoots into a conformation tensor so stretched that :math:`\exp\boldsymbol{\Psi}` overflows on the very first Newton step. Stepping up from a converged solution at a smaller :math:`\mathrm{Wi}` costs nothing and works.

The conformation tensor needs a symmetry condition of its own. Its shear component is odd under :math:`y\to-y`, so :math:`\Psi_{xy}=0` on the centreline, and imposing that is not optional: the constitutive equation contains no diffusion, so nothing else damps an odd mode in that component.

We introduce two numbers to steer the local mesh size: a coarse ``far_resolution`` for the channel, and a much finer ``near_resolution`` attached to a single extra point on the top wall directly above the cylinder, from which gmsh grades outwards. This is combined with a O-grid, which resolves the thin polymer stress boundary layer on the cylinder.

Another catch is in :math:`eqviscoelastic` itself, since it has no diffusion whatsoever -- its only spatial operator is :math:`\left(\vec{u}\cdot\nabla\right)\boldsymbol{\Psi}` -- and just behind the rear stagnation point the polymer stress grows exponentially. Plain Galerkin answers that with a node-to-node sawtooth running the length of the wake. It barely touches the drag, which is an integral over the cylinder, but it destroys any profile taken through the wake. Passing ``stabilization="SUPG"`` to :py:class:`~pyoomph.equations.viscoelastic.ViscoelasticEquations` removes it; the reference stabilises as well, with a DEVSS-G/DG scheme.


:numref:`figpdeviscoelasticstress` reproduces Fig. 6 of :cite:`Claus2013`: the polymer stress :math:`\tau_{xx}` along a path that runs up the centreline, over the cylinder surface and away down the wake, one curve per Weissenberg number. The axes are theirs, so the two can be laid side by side.


..  figure:: viscoelastic_stress.*
	:name: figpdeviscoelasticstress
	:align: center
	:alt: Polymer stress along the centreline and cylinder surface.
	:class: with-shadow
	:width: 80%

	:math:`\tau_{xx}` along the centreline and around the cylinder, for :math:`\mathrm{Wi}=0.1` to :math:`0.9`; compare Fig. 6 of Claus and Phillips. The large peak at :math:`X=0` is the top of the cylinder, where the fluid is sheared hardest, and it grows from about 18 to about 127 over this range, matching theirs closely. The second peak just downstream of :math:`X=1` is the *birefringent strand* in the wake; it comes out higher and less resolved here than in their figure. Decreasing the mesh size factors would yield better agreement, but also longer simulation time.

Their Fig. 12 shows a different decomposition: the Cauchy stress is made traceless and then projected onto the streamline direction and its normal, giving a flow-directed shear stress :math:`S_1` and normal stress :math:`S_2`. Note that the pressure drops out of the traceless part identically, so only the solvent rate of strain and the polymer stress contribute.

Both are obtained with :py:class:`~pyoomph.equations.generic.ProjectExpression`, which solves an :math:`L^2` projection onto a real finite element field, rather than with :py:meth:`~pyoomph.generic.codegen.Equations.add_local_function`. The projection is one-way -- nothing else reads :math:`S_1` or :math:`S_2` -- therefore, we define the projection of these quantities on another residual ``"output_projection"``. During the continuation of :math:`\mathrm{Wi}`, :math:`S_1` or :math:`S_2` won't be solved. Only before output, we have to call :py:meth:`~pyoomph.generic.problem.Problem.solve_auxiliary_residual` to solve the projection alone, so that both quantities are updated and end up in the written output.

..  figure:: viscoelastic_flowstress.*
	:name: figpdeviscoelasticflowstress
	:align: center
	:alt: Flow-directed shear and normal stress around the cylinder.
	:class: with-shadow
	:width: 90%

	The flow-directed stresses :math:`S_1` and :math:`S_2`; compare Fig. 12 of Claus and Phillips.


The quantity everyone reports is the dimensionless drag on the cylinder, :math:`K=F_x/(\eta_0\langle u\rangle)`, obtained here by integrating the total traction over the cylinder surface. The script walks up in :math:`\mathrm{Wi}` with :py:meth:`~pyoomph.generic.problem.Problem.go_to_param`, which halves its step whenever Newton fails, and prints the drag next to the values of Claus and Phillips :cite:`Claus2013`:

.. code:: none

     Wi      K (pyoomph)   K (Claus & Phillips)
     0.1      130.3631         130.364
     0.2      126.6261         126.626
     0.3      123.1928         123.192
     0.4      120.5937         120.593
     0.5      118.8298         118.826
     0.6      117.7808         117.776
     0.7      117.3213         117.316
     0.8      117.3396               -
     0.9      117.7332               -

The agreement is within :math:`0.005\,\%` over the whole range for which the reference has mesh-converged values, which is comfortably inside the spread between the published values themselves -- the three independent codes they compare disagree with each other by about :math:`0.02\,\%`. The drag falls with elasticity, reaches a minimum near :math:`\mathrm{Wi}\approx0.8` and rises again, which is the behaviour the literature reports. The last two rows have no entry to compare against: at :math:`\mathrm{Wi}=0.8` and above, every column of their table is marked as diverging.

.. note::

   This benchmark is where the log-conformation approach earns its keep. Formulations that discretise :math:`\mathbf{C}` directly typically lose convergence somewhere around :math:`\mathrm{Wi}\approx0.7`, and at :math:`\mathrm{Wi}=1` every column of the comparison table in :cite:`Claus2013` is marked as diverging.

   Do not read the last two rows as evidence that this formulation is immune, though. Whether the continuation gets that far depends on the mesh, and it depends on it the wrong way round: a *finer* mesh loses convergence *earlier*, which is exactly what :cite:`Claus2013` report for every scheme they compare. The drag is also the least sensitive thing one could monitor -- it is an integral over the cylinder, and it stays smooth long after the wake has stopped converging.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <viscoelastic_cylinder.py>`

		:download:`Download all examples <../../tutorial_example_scripts.zip>`
