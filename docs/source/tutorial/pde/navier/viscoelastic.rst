.. _secpdeviscoelastic:

Viscoelastic flow: the log-conformation approach
------------------------------------------------

A polymer solution does not respond to deformation instantaneously: the dissolved chains stretch, store elastic energy and relax back over a characteristic time :math:`\lambda`. The state of the chains is described by the *conformation tensor* :math:`\mathbf{C}`, a symmetric positive definite tensor that equals the identity when the polymer is unstretched. It is carried by the flow and pulled back towards equilibrium,

.. math::

   \stackrel{\triangledown}{\mathbf{C}}\;=\;\partial_t\mathbf{C}+\left(\vec{u}\cdot\nabla\right)\mathbf{C}-\nabla\vec{u}\cdot\mathbf{C}-\mathbf{C}\cdot\left(\nabla\vec{u}\right)^\mathrm{T}\;=\;-\frac{1}{\lambda}g(\mathbf{C})

where the left hand side is the *upper-convected derivative* and the constitutive model supplies :math:`g`. For the Oldroyd-B model :cite:`Oldroyd1950` it is simply :math:`g(\mathbf{C})=\mathbf{C}-\mathbf{I}`. The polymer then contributes an extra stress

.. math::

   \boldsymbol{\tau}_\mathrm{p}=\frac{\eta_\mathrm{p}}{\lambda}\left(\mathbf{C}-\mathbf{I}\right)

which is added to the momentum equation alongside the usual solvent stress. Note that pyoomph's :py:func:`~pyoomph.expressions.generic.upper_convected_derivative` implements exactly the left hand side above, so the conformation equation can be written down almost verbatim.

Why not solve for :math:`\mathbf{C}` directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Flow past a confined cylinder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The standard benchmark for viscoelastic flow solvers is creeping flow past a cylinder placed on the centreline of a channel whose half-height is twice the cylinder radius :cite:`Alves2001,Hulsen2005,Claus2013`. Everything is nondimensionalised with the total viscosity :math:`\eta_0=\eta_\mathrm{s}+\eta_\mathrm{p}`, the cylinder radius and the mean inlet velocity, leaving the solvent fraction :math:`\beta=\eta_\mathrm{s}/\eta_0=0.59` and the Weissenberg number :math:`\mathrm{Wi}=\lambda\langle u\rangle/R`. Only the upper half is solved, with a symmetry condition on the centreline:

.. literalinclude:: viscoelastic_cylinder.py
   :language: python
   :start-at: class ConfinedCylinderProblem(Problem):
   :end-at: return float(self.get_mesh("fluid/cylinder").evaluate_observable("drag"))

Three details are worth pointing out.

The inflow condition has to prescribe not only the velocity but also the *stress* the polymer has in fully developed channel flow, otherwise the solution spends the entire upstream length relaxing towards it. The two helpers :py:func:`~pyoomph.equations.viscoelastic.oldroyd_b_shear_conformation` and :py:func:`~pyoomph.equations.viscoelastic.symmetric_2x2_matrix_log` build that condition: the first gives :math:`\mathbf{C}` for a prescribed local shear rate, the second takes its logarithm analytically. On the symmetry line the shear rate vanishes and :math:`\mathbf{C}` becomes isotropic, which is a degenerate case for a matrix logarithm; the helper handles it.

The Weissenberg number enters as a :py:meth:`~pyoomph.generic.problem.Problem.define_global_parameter`, because the solution has to be *continued* in it. Starting a stationary solve directly at a larger :math:`\mathrm{Wi}` overshoots into a conformation tensor so stretched that :math:`\exp\boldsymbol{\Psi}` overflows on the very first Newton step. Stepping up from a converged solution at a smaller :math:`\mathrm{Wi}` costs nothing and works.

Finally, the mesh around the cylinder is a structured, graded O-grid of quadrilaterals. This is not decoration: the polymer stress forms a thin boundary layer on the cylinder, and that layer is what sets the accuracy of the drag. The spatial error estimator will happily refine the far field instead, where the flow is fully developed and nothing is gained, so the layer is resolved by hand through the number of nodes across it.

..  figure:: viscoelastic_stress.*
	:name: figpdeviscoelasticstress
	:align: center
	:alt: Polymer stress around a confined cylinder.
	:class: with-shadow
	:width: 100%

	The polymer stress :math:`\tau_{xx}` at :math:`\mathrm{Wi}=0.7`, mirrored about the symmetry line. The stress is concentrated in a thin layer on the cylinder, where the flow shears the polymer strongly, and is then convected downstream into a narrow *birefringent strand* along the wake. Resolving both is what makes this benchmark demanding: the colour scale is set by the peak on the cylinder, so the strand looks faint even though it decays only slowly with distance.

The quantity everyone reports is the dimensionless drag on the cylinder, :math:`K=F_x/(\eta_0\langle u\rangle)`, obtained here by integrating the total traction over the cylinder surface. Running the script continues from :math:`\mathrm{Wi}=0.1` to :math:`0.7` and prints it next to the values of Claus and Phillips :cite:`Claus2013`:

.. code:: none

     Wi      K (pyoomph)   K (Claus & Phillips)
     0.1      130.3756         130.364
     0.2      126.6489         126.626
     0.3      123.2321         123.192
     0.4      120.6569         120.593
     0.5      118.9242         118.826
     0.6      117.9109         117.776
     0.7      117.4789         117.316

The agreement is within :math:`0.01\,\%` at low :math:`\mathrm{Wi}` and :math:`0.14\,\%` at :math:`\mathrm{Wi}=0.7`, on a mesh coarse enough to run in well under a minute. That is comfortably inside the spread between the published values themselves, which disagree with each other by about :math:`0.02\,\%`. The drag falls with elasticity, reaches a minimum near :math:`\mathrm{Wi}\approx0.7` and rises again, which is the behaviour the literature reports.

.. note::

   This benchmark is where the log-conformation approach earns its keep. Formulations that discretise :math:`\mathbf{C}` directly typically lose convergence somewhere around :math:`\mathrm{Wi}\approx0.7`, and at :math:`\mathrm{Wi}=1` every column of the comparison table in :cite:`Claus2013` is marked as diverging. Solving for :math:`\log\mathbf{C}` instead reaches a steady solution there without any stabilisation at all.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <viscoelastic_cylinder.py>`

		:download:`Download all examples <../../tutorial_example_scripts.zip>`
