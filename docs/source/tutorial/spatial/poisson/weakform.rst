Weak formulation
~~~~~~~~~~~~~~~~

The most basic example is usually the *Poisson equation*, which is usually taken as a prime example of the *finite element method*. The Poisson equation for an unknown function :math:`u(\vec{x})` with a source function :math:`g(\vec{x})`, both defined on an :math:`n`-dimensional domain :math:`\Omega` with position vector :math:`\vec{x}` reads

.. math:: :label: eqspatialpoissonstrong

   \begin{aligned}
   -\nabla^2 u=g\,. 
   \end{aligned}

Here, :math:`\nabla^2` is the Laplace operator, reading :math:`\sum_i^n \partial_{x_i}^2` in Cartesian coordinates. Of course, this equation requires boundary conditions at the boundaries of :math:`\Omega`, which can be split into Dirichlet boundaries, :math:`u=u_\text{D}` on :math:`\Gamma_\text{D}`, and Neumann boundaries, :math:`\nabla u\cdot \vec{n}=j_\text{N}` on :math:`\Gamma_\text{N}` with outward normal :math:`\vec{n}`, so that :math:`\partial\Omega=\Gamma=\Gamma_\text{D}\cup\Gamma_\text{N}`.

The key point in the *finite element method* is to cast the strong formulation :math:numref:`eqspatialpoissonstrong` to a weak formulation. This is achieved as follows: Let :math:`v` be an arbitrary function defined on :math:`\Omega`. Upon multiplication of :math:numref:`eqspatialpoissonstrong` by :math:`v`, followed by a spatial integration over the domain :math:`\Omega` leads to:

.. math:: :label: eqspatialpoissonweak1

   \begin{aligned}
   -\int_\Omega\nabla^2 u\:v\:\mathrm{d}^n x=\int_\Omega g\:v\:\mathrm{d}^n x\,. 
   \end{aligned}

Now, we treat the term on the lhs by integration by parts to arrive at

.. math:: :label: eqspatialpoissonweak2

   \begin{aligned}
   \int_\Omega\nabla u\cdot \nabla v\:\mathrm{d}^n x-\int_\Gamma\nabla u\cdot \vec{n}\:v\:\mathrm{d}S=\int_\Omega g\:v\:\mathrm{d}^n x\,. 
   \end{aligned}

When we now demand that :math:`v=0` on the Dirichlet boundaries :math:`\Gamma_\text{D}` (which is a necessary restriction for Dirichlet boundary conditions), the Neumann boundary condition :math:`\nabla u\cdot \vec{n}=j_\text{N}` can be inserted directly in the surface integral:

.. math:: :label: eqspatialpoissonweak3

   \begin{aligned}
   \int_\Omega\nabla u\cdot \nabla v\:\mathrm{d}^n x-\int_{\Gamma_\text{N}} j_\text{N}\:v\:\mathrm{d}S=\int_\Omega g\:v\:\mathrm{d}^n x\,.
   \end{aligned}

Finally, upon introducing the shorthand integral notations

.. math:: :label: eqspatialweakshorthand

   \begin{aligned}
   \left(a,b\right)&=\int_\Omega a\: b \:\mathrm{d}^n x \qquad & \left(\vec{a},\vec{b}\right)&=\int_\Omega \vec{a}\cdot \vec{b} \:\mathrm{d}^n x \\ 
   \langle a,b \rangle&=\int_{\Gamma_\text{N}} a\: b \:\mathrm{d}S \qquad & \langle  \vec{a},\vec{b} \rangle&=\int_{\Gamma_\text{N}} \vec{a}\cdot \vec{b} \:\mathrm{d}S
   \end{aligned}

and putting all in residual form, i.e. putting all terms on one side, we arrive at

.. math:: :label: eqspatialpoissonweak

   \begin{aligned}
   \left(\nabla u,\nabla v\right)-\left(g, v\right)-\left\langle j_\text{N}, v\right\rangle=0\,. 
   \end{aligned}

It can be shown that the weak formulation :math:numref:`eqspatialpoissonweak` together with the Dirichlet conditions :math:`u|_{\Gamma_\text{D}}=u_\text{D}` is equivalent to the strong formulation :math:numref:`eqspatialpoissonstrong` together with both types of boundary conditions, as long as :math:`v` may be chosen arbitrarily (with the requirement to have :math:`v=0` on :math:`\Gamma_D`).

However, the weak formulation has several features that make it appealing: The order of the spatial derivatives has been reduced from second to first order and the implementation of the Neumann conditions is now just an interface integral. Of course, it also comes at the price that spatial integrals have to be carried out to obtain a solution and that the arbitrary test function :math:`v` is appearing in the weak formulation. However, the combination of integrals and test functions provides a neat way of solving the equations numerically on versatile geometries and in all kinds of dimensions.
