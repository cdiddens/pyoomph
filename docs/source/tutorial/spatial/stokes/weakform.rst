Strong and weak formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Stokes equations for the velocity field :math:`\vec{u}` and the pressure :math:`p` read

.. math::

   \begin{aligned}
   -\nabla p +\nabla\cdot \left[\mu\left(\nabla\vec{u}+(\nabla\vec{u})^t\right)\right]&=0\\
   \nabla\cdot u&=0
   \end{aligned}

Here, :math:`\mu` is the dynamic viscosity of the liquid. Obviously, we have to solve two fields, one vector field for the velocity :math:`\vec{u}` and a scalar pressure field :math:`p`. The second equation, the continuity equation, is a scalar equation and reads as a constraint on the velocity field :math:`\vec{u}`, namely that its divergence vanishes. For an :math:`n`-dimensional Stokes flow problem, there are hence :math:`n` momentum equations and :math:`1` constraining equation. To solve this, we have a :math:`n`-dimensional vector field :math:`\vec{u}` and a single scalar :math:`p`, which gives rise to the generic weak form of the Stokes equations by solving the vectorial momentum equation on the test spaces of the velocity and the continuity equation on the test space of the pressure. Let :math:`\vec{v}` and :math:`q` be the corresponding test functions, the weak form thus reads

.. math:: :label: eqspatialstokesweak

   \begin{aligned}
   \left(-p\mathbf{1}+\mu\left(\nabla\vec{u}+(\nabla\vec{u})^t\right),\nabla\vec{v}\right)+\left(\nabla\cdot \vec{u},q\right)-\left\langle \vec{n}\cdot\left[-p\mathbf{1}+\mu\left(\nabla\vec{u}+(\nabla\vec{u})^t\right)\right] ,\vec{v}\right\rangle=0
   \end{aligned}

Note how the partial integration also works for tensor-valued equations by using the tensor contraction in the first term of the lhs. Furthermore, the momentum equation has been negated, since it is beneficial for the Navier-Stokes equation later on in :numref:`secpdenavstokes`.

.. warning::

   Due to the continuity equation, one can simplify the strain tensor term, i.e. do not consider the term :math:`(\nabla\vec{u})^t`. However, this gives rise to a different, usually non-physical, natural Neumann boundary condition, i.e. the :math:`\langle .,.\rangle` term. If one wants to impose tractions, i.e. Neumann conditions, it is hence important to keep the :math:`(\nabla\vec{u})^t` in the weak formulation of the Stokes equations.
