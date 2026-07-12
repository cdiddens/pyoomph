.. _azimuthalstabana:

Azimuthal stability analysis
----------------------------

This section covers a powerful tool to investigate full three-dimensional stability analysis by considering just a two-dimensional axisymmetric mesh. The requirement is hence, that the base solution, which should be tested for stability, is perfectly axisymmetric, but then undergoes a symmetry breaking with respect to azimuthal modes. The general idea is the following: We have some stationary solution of our unknown vector :math:`\vec{U}^{(0)}`, which is axisymmetric, i.e. :math:`\vec{U}^{(0)}=\vec{U}^{(0)}(r,z)`. Components of this solution vector could be a velocity and pressure field, for instance. We want to investigate what happens, if this solution is perturbed by some azimuthal mode :math:`m` with tiny amplitude, i.e. some mode :math:`\epsilon\vec{U}^{(m)}\exp(im\phi+\lambda_m t)`. Here, the amplitude :math:`\epsilon` should be sufficiently small that only linear terms enter. :math:`\vec{U}^{(m)}(r,z)` is the perturbation vector of this mode (i.e. an eigenvector) and :math:`\lambda_m` the corresponding eigenvalue. When plugging into the system of equations and truncating at linear order in :math:`\epsilon`, the linear dynamics of this perturbation is given by the generalized eigenvalue problem:

.. math:: \lambda_m\mathbf{M}^{(m)}\vec{U}^{(m)}=\mathbf{J}^{(m)}(\vec{U}^{(0)})\vec{U}^{(m)}

Even if the original system is entirely real, the mass matrix :math:`\mathbf{M}^{(m)}` and the Jacobian :math:`\mathbf{J}^{(m)}` are usually complex-valued. The reason is that any even spatial derivatives with respect to the azimuthal coordinate, i.e. :math:`\partial_\phi`, will produce an :math:`im`.

When activated, pyoomph can derive this particular eigensystem automatically. To that end, all unknown scalar (:math:`s`) and vectorial (:math:`\vec{v}`) fields will be expanded like

.. math::

   \begin{aligned}
   s(r,z,\phi,t)&=s^{(0)}(r,z)+\epsilon s^{(m)}(r,z)\exp(i m\phi+\lambda_m t) \\
   \vec{v}(r,z,\phi,t)&=\vec{v}^{(0)}(r,z)+\epsilon \vec{v}^{(m)}(r,z)\exp(i m\phi+\lambda_m t) 
   \end{aligned}

before actually being plugged into the equations. Furthermore, all vectors :math:`\vec{v}` will automatically get a :math:`\phi`-component when defined by the :py:meth:`~pyoomph.generic.codegen.Equations.define_vector_field`, which is expanded the same way. All coordinate-system-independent spatial derivatives, i.e. :py:func:`~pyoomph.expressions.generic.grad` and :py:func:`~pyoomph.expressions.div`, will consider also the derivative with respect to :math:`\phi` and include the additional vector components :math:`v_\phi`. The test functions will be augmented by a :math:`\exp(-im\phi)` and also a test function for the :math:`v_\phi`-component is considered in all vector fields. Global, i.e. ODE quantities, will not be expanded that way.

After plugging the augmented fields and test functions into the system of weak formulations and carrying out the spatial derivatives, we obtain the residual vector :math:`\vec{R}` including the augmented mode expansions. The basic residual vector :math:`\vec{R}^{(0)}`, i.e. the residual for the axisymmetric base state, is obtained by setting :math:`\epsilon=0` and :math:`m=0`. This corresponds to the conventional residual and is real-valued. It is in fact the same residual as if just a normal axisymmetric coordinate system is used. Besides :math:`\vec{R}^{(0)}`, also :math:`\operatorname{Re}(\vec{R}^{(m)})` and :math:`\operatorname{Im}(\vec{R}^{(m)})` are determined. We just take the first order :math:`\epsilon` term here, i.e. :math:`\vec{R}^{(m)}=\partial_\epsilon \vec{R}|_{\epsilon=0}`. Thereby, azimuthal modes can appear in linear order only, but any non-linear coupling with :math:`\vec{U}^{(0)}` contributions appears. The linear contributions proportional :math:`\exp(i m\phi)` will cancel out with the corresponding :math:`\exp(-i m\phi)` of the test functions. Thereby, the complex-valued azimuthal residual :math:`\vec{R}^{(m)}` emerges, which does not depend explicitly on :math:`\phi`, but has the mode number :math:`m` in it, provided that spatial derivatives have produced these terms.

As usual, the base Jacobian :math:`\mathbf{J}^{(0)}`, mass matrix :math:`\mathbf{M}^{(0)}` and, if desired, the Hessian :math:`\mathbf{H}^{(0)}` are obtained by deriving :math:`\vec{R}^{(0)}` with respect to :math:`\vec{U}^{(0)}`. The azimuthal Jacobian and mass matrix, :math:`\mathbf{J}^{(m)}` and :math:`\mathbf{M}^{(m)}`, is obtained by deriving :math:`\vec{R}^{(m)}` with respect to :math:`\vec{U}^{(m)}`. If the Hessian :math:`\mathbf{H}^{(m)}` is desired, it is calculated by deriving :math:`\mathbf{J}^{(m)}` with respect to :math:`\vec{U}^{(0)}`, not :math:`\vec{U}^{(m)}`, as one might naively expect. The reason is that :math:`\mathbf{J}^{(m)}` is independent of :math:`\vec{U}^{(m)}`, since :math:`\vec{U}^{(m)}` enters :math:`\vec{R}^{(m)}` only in linear order. For bifurcation tracking, however, it is required to get the variations of :math:`\mathbf{J}^{(m)}` with respect to the base mode :math:`\vec{U}^{(0)}`. Therefore, the Hessian is derived that way.

Lastly, the boundary conditions at the axis of symmetry are of fundamental importance. For conventional axisymmetry, one demands that :math:`\partial_r s^{(0)}=\partial_r v_z^{(0)}=0` and :math:`v_r^{(0)}=v_\phi^{(0)}=0`, since one otherwise gets a singularity at the axis. However, for azimuthal modes, this can be different. For :math:`m=1`, we get :math:`s^{(m)}=v_z^{(m)}=0`, since otherwise these values are not well defined at :math:`r=0` when varying :math:`\phi`. However, :math:`v_r^{(m)}` and :math:`v_\phi^{(m)}` may be non-zero for :math:`m=1`, since these basis vectors of these components exactly rotate with the :math:`\phi` in exactly the same manner as the :math:`m=1`-mode. There, however, :math:`\partial_r v_r^{(m)}=\partial_r v_\phi^{(m)}=0` is required. For all other :math:`m`, all vector components and scalars have to vanish, i.e. :math:`v_r^{(m)}=v_\phi^{(m)}=v_z^{(m)}=s^{(m)}=0`.

These :math:`m`-dependent boundary conditions are automatically taken care of in the :py:class:`~pyoomph.meshes.bcs.AxisymmetryBC` object. It will impose exactly the correct boundary conditions for all vector and scalar fields.

For more details on this, we refer to our article :cite:`Diddens2024`.

.. toctree::
   :maxdepth: 5
   :hidden:
   
   azimuthal/rbconvect.rst
   azimuthal/rising_bubble.rst   
