.. _secspatialstokespuredirichlet:

A case with pure Dirichlet boundary conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In :numref:`secspatialpoissonpureneumann` we have learned that the Poisson equation shows some caveats when considering pure Neumann conditions. We learned that the solution is in that case not unique due to a shift-invariance with respect to the addition of an arbitrary constant. A similar issue also occurs in the Stokes equation when the velocities in normal direction are prescribed by Dirichlet boundary conditions. In that case, all potential Neumann contributions vanish and we are left with the weak form

.. math:: \left(-p\mathbf{1}+\mu\left(\nabla\vec{u}+(\nabla\vec{u})^t\right),\nabla\vec{v}\right)+\left(\nabla\cdot \vec{u},q\right)=0

To illustrate the issue, let use revert the partial integration on the pressure contribution in the first term, where the arising Neumann flux is again not present due to the pure Dirichlet boundary conditions for the velocity:

.. math:: \left(\nabla p,\vec{v}\right)+\left(\mu\left(\nabla\vec{u}+(\nabla\vec{u})^t\right),\nabla\vec{v}\right)+\left(\nabla\cdot \vec{u},q\right)=0

Since only gradients of the pressure enter the equation, it is invariant with respect to :math:`p\to p+\mathrm{const}`, i.e. a unique solution for the pressure cannot be determined. Any Neumann term would depend on the absolute value of the pressure and hence remove this ambiguity.

If one only has Dirichlet conditions, one hence should either impose a global Lagrange multiplier enforcing the average pressure, directly analogous to the procedure described in :numref:`secspatialpoissonpureneumann` for the Poisson equation with pure Neumann conditions. Alternatively, often easier, one can prescribe one single pressure degree. On a :py:class:`~pyoomph.meshes.simplemeshes.RectangularQuadMesh`, one can e.g. fix the pressure in the lower left corner by adding a ``DirichletBC(pressure=0)@"bottom/left"`` to the domain. One has to make sure that only a single degree of freedom is specified, i.e. the domain where the :py:class:`~pyoomph.meshes.bcs.DirichletBC` is applied must be a single point.
