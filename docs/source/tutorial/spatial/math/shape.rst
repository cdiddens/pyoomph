Continuous basis functions, spatial discretization and solution procedure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Until now, we always have allocated the unknown function :math:`u` with ``define_scalar_field("u","C2")``. While the first argument is just the name, the second argument needs more elaboration. It defines the *finite element space* on which the function is defined. In pyoomph, there are two kinds of continuous spaces, namely ``"C1"`` and ``"C2"``. Discontinuous spaces ``"D0"`` (constant per element) and ``"DL"`` (affine linear per element, cf. :numref:`secspatialcr`) are also available, but not discussed here. To understand these two continuous spaces, let us delve slightly into the basics of the *finite element method*. In fact, the unknown function :math:`u` is spatially discretized by so-called *shape functions* :math:`\psi_l(\vec{x})`.

.. math:: :label: eqspatialbasisexpand
   
   u(\vec{x})=\sum_l u_l \psi_l(\vec{x})\,.

This linear expansion in the shape functions obviously separates the spatial dependency of the function :math:`u` into amplitudes :math:`u_l` and spatially varying basis functions :math:`\psi_l`. The second argument in :py:meth:`~pyoomph.generic.codegen.Equations.define_scalar_field` selects the particular choice of these shape functions :math:`\psi_l`, namely ``"C1"`` for linear basis functions and ``"C2"`` for basis functions of second order, i.e. quadratic ones.

In principle, the basis functions are quite arbitrary. In fact, one could select e.g. :math:`\psi_l(\vec{x})=\exp(i\vec{k}_l\cdot{x})` to obtain a Fourier decomposition. However, since the conventional finite element method is solved in the spatial domain, not in the spectral one, this choice of the basis functions is problematic. Instead, it is beneficial to choose them that the *support* only covers a few neighboring elements, i.e. they should be zero almost everywhere in the entire domain, except in the vicinity of a single position. Thereby, the degrees of freedom, i.e. :math:`u_l`, are sufficiently localized in space. This idea leads to the conclusion that the basis functions should be defined in a piece-wise manner - zero almost everywhere, but non-zero in the vicinity of a position :math:`\vec{x}_l` associated with the degree of freedom :math:`u_l`.

The simplest idea on a one-dimensional domain, discretized by positions at :math:`x_1<x_2<\ldots <x_n` is hence to use linear slopes, i.e. the :math:`l`-th basis function corresponding to the point :math:`x_l` reads

.. math:: \psi_l(x)=\left\{ \begin{array}{rcl} 0 & \text{ for } & x<x_{l-1}  \\ (x-x_{l-1})/(x_l-x_{l-1}) & \text{ for } & x_{l-1}\leq x < x_l \\ (x_{l+1}-x)/(x_{l+1}-x_l) & \text{ for } & x_{l}< x \leq x_{l+1} \\ 0 & \text{ for } & x\geq x_{l+1}  \end{array} \right.\, ,

where the missing points :math:`x_{0}` and :math:`x_{n+1}` are not required if :math:`x` is confined to the range of the domain, i.e. between :math:`x_1` and :math:`x_l`. The basis functions are shown for a 1d mesh in :numref:`figspatialshapes1d`.

..  figure:: shapes1d.*
	:name: figspatialshapes1d
	:align: center
	:alt: Shape functions in 1d
	:class: with-shadow
	:width: 70%

	Top: linear basis functions (``"C1"``). Bottom: quadratic basis functions (``"C2"``), where the dashed parabolas are the shape functions at the intermediate nodes in the center of each element.


These functions are in fact used if the space ``"C1"`` is used. Let us see how the weak form of the Poisson equation :math:numref:`eqspatialpoissonweak` in one spatial dimension reads with these basis functions:

.. math::

   \begin{aligned}
   \left(\nabla \sum_l u_l \psi_l,\nabla v\right)-\left(g, v\right)-\left\langle j_\text{N}, v\right\rangle&= \\
   \left(\sum_l u_l \partial_x \psi_l,\partial_x v\right)-\left(g, v\right)-\left\langle j_\text{N}, v\right\rangle&=0
   \end{aligned}

The derivative of :math:`u(x)` is not required, just the derivative of the shape functions :math:`\psi_l`, which can be calculated easily except on :math:`x_{l-1}`, :math:`x_l` and :math:`x_{l+1}`, where the derivative is not defined due to the piece-wise nature of the chosen basis functions. However, these points are a *null set* with respect to the spatial integration :math:`\left(.,\,.\right)` so that it is not required to consider these points within the integration.

Finally, we have not yet addressed the test function :math:`v`. As mentioned before, the above weak form has to hold for all (quite arbitrary) choices of :math:`v`. In the discretization :math:numref:`eqspatialbasisexpand`, we have used :math:`n` unknowns :math:`u_l` (for :math:`l=1,\ldots,n`). So to get a discretized system of equations, we should choose :math:`n` linearly independent test functions :math:`v_k` (with :math:`k=1,\ldots,n`). Furthermore, we have to make sure that the *mass matrix* :math:`\mathbf{M}=(M_{lk})=(\phi_l,v_k)` has a full rank. If the :math:`l`-th row of this matrix is entirely zero, it means that we have selected our :math:`n` test functions :math:`v_k` in a manner that there is no support for the degree of freedom :math:`u_l`. Thereby, we would not obtain a discretized equation for this degree of freedom.

The trivial choice of the test functions is the *Galerkin method*, where we just take the same basis functions, i.e. :math:`v_k=\psi_k`. Thereby, both requirements on the test functions hold automatically. The Poisson equation hence reads

.. math::

   \begin{aligned}
   \left(\sum_l u_l \partial_x \psi_l,\partial_x \psi_k\right)-\left(g, \psi_k\right)-\left\langle j_\text{N}, \psi_k\right\rangle=0 \quad \text{for} \quad k=1,\ldots,n\,.
   \end{aligned}

Due to the reasonable choice of our basis functions (and hence test functions), the integrands are zero almost everywhere except for the neighborhood of the corresponding point :math:`x_k`. Thus, the spatial integrals can be restricted to the support of each :math:`\psi_k`:

.. math::

   \begin{aligned}
   \int_{x_{k-1}}^{x_{k+1}} \sum_l u_l (\partial_x \psi_l) (\partial_x \psi_k) \mathrm{d}x- \int_{x_{k-1}}^{x_{k+1}} g \psi_k \mathrm{d}x&\\
   +j_\text{N}(x_1)\psi_1(x_1) \delta_{k,1}-j_\text{N}(x_n)\psi_n(x_n) \delta_{k,n}=0\quad &\text{for} \quad k=1,\ldots,n\,.
   %\left(\sum_l u_l \partial_x \psi_l,\partial_x \psi_k\right)-\left(g, \psi_k\right)-\left\langle j_\text{N}, \psi_k\rangle=0 \quad \text{for} \quad k=1,\ldots,n\,.
   \end{aligned}

The Neumann flux terms appear only at the boundaries for :math:`k=1` and :math:`k=n`, which is indicated by the *Kronecker deltas*.

The next benefit of the choice of the basis functions is that also :math:`\psi_l` are zero almost everywhere, i.e. the sum over :math:`l` can be simplified depending on :math:`k`. This eventually gives the system of equations

.. math:: :label: eqspatialdiscretizedpoissonsys

   \begin{aligned}
   u_{1}(\partial_x \psi_1,\partial_x \psi_1)+u_{2}(\partial_x \psi_2,\partial_x \psi_1)=(g,\psi_1)-j_\text{N}(x_1)\psi_1(x_1) &\quad\text{for} \quad k=1 \nonumber \\ 
   u_{k-1}(\partial_x \psi_{k-1},\partial_x \psi_k)+u_{k}(\partial_x \psi_k,\partial_x \psi_k)+u_{k+1}(\partial_x \psi_{k+1},\partial_x \psi_k)=(g,\psi_k) &\quad\text{for}\quad 2\leq k \leq n-1  \\
   u_{n-1}(\partial_x \psi_{n-1},\partial_x \psi_n)+ u_{n}(\partial_x \psi_n,\partial_x \psi_n)=(g,\psi_n)+j_\text{N}(x_n)\psi_m(x_n) &\quad\text{for}\quad k=n\,, \nonumber 
   \end{aligned}

where the integrals :math:`(.\,.)` only have to be carried out over a small section of the entire domain where both arguments are non-zero. Denoting the symmetric *stiffness matrix* :math:`\mathbf{K}=(K_{lk})=(\partial_x \psi_l,\partial_k \psi_l)`, the vector of degrees of freedom :math:`\vec{u}=(u_l)` and the vector comprising the right hand side :math:`\vec{b}`, one can rewrite this as matrix equation

.. math::

   \begin{aligned}
   \mathbf{K}\vec{u}=\vec{b}
   \end{aligned}

In principle this equation can be solved by inverting :math:`\mathbf{K}`, but due to the pure Neumann conditions and the shift-invariance :math:`u_l\to u_l+\text{const}` of the Poisson equation, :math:`\operatorname{det}(\mathbf{K})` is actually :math:`0`. We have discussed how one can overcome this issue in :numref:`secspatialpoissonpureneumann`.

Let us now see how Dirichlet boundary conditions are treated in discretized equations. When we impose a Dirichlet condition at the right side, i.e. we set :math:`u(x_n)=a`, it means that the amplitude :math:`u_n` of the expansion :math:numref:`eqspatialbasisexpand` is fixed by :math:`u_n=a`. Hence, it is not an unknown anymore. Therefore, we just remove the :math:`n`-th equation from the system :math:numref:`eqspatialdiscretizedpoissonsys`. Removing this equation will automatically remove all connection to the Neumann flux at the right side, i.e. :math:`j_\text{N}(x_n)`. This reflects the fact that one can either impose a Dirichlet or a Neumann condition at the boundaries. Imposing both simultaneously, i.e. a Cauchy condition, is not feasible (cf. :numref:`secspatialcauchybc`). However, when this equation is removed, :math:`\mathbf{K}` will become invertible, i.e. :math:`\operatorname{det}(\mathbf{K})\neq 0`. Furthermore, in the :math:`(n-1)`-th equation, the occurrence of :math:`u_n` can be replaced by :math:`a`, which connects the entire system to the value of the Dirichlet condition. A unique solution is now feasible and depends on the value :math:`a`.

This is exactly what happens internally in pyoomph when a one-dimensional Poisson equation on the space ``"C1"`` is solved. First of all, the mesh is constructed with a storage of :math:`u_l` at each node located at :math:`x_l` for :math:`l=1,\ldots,n`. Then, the Dirichlet boundary condition at :math:`x_n` is applied setting the value of :math:`u_n=a` and removing it from the system. Finally, the spatial integrals over the shape functions are carried out numerically (using *Gauss quadrature*, cf. :numref:`secmiscquadrature`) and the resulting system is solved with a linear solver back-end. We will discuss this more generally in the next section.

At the end of this section, we still have to elaborate on the space ``"C2"`` and how the shape functions are defined in higher spatial dimensions. With the second order basis functions, i.e. ``"C2"``, the approximated discretized solution is not piece-wise linear, but piece-wise quadratic. Since a parabola requires three points :math:`(x,u)` to be uniquely defined, additional points :math:`x_{l+1/2}` are introduced between each interval :math:`x_l` and :math:`x_{l+1}`, each of them associated with an own degree of freedom :math:`u_{l+1/2}`. The basis functions :math:`\phi_l` and the test functions :math:`v_k=\phi_k` are now quadratic, but the rest of the approach is essentially the same, except that also the new degrees of freedom at :math:`l+1/2` have to be added to the discretized system. The quadratic basis functions of the space ``"C2"`` for a 1d mesh are also depicted in :numref:`figspatialshapes1d`.

In higher dimensions, we limit ourselves to plot the basis functions, which are plotted for 2d elements in :numref:`figspatialshapes2dC1` and :numref:`figspatialshapes2dC2`. In three dimensions, it is generalized analogously.

..  figure:: shapes2dC1.*
	:name: figspatialshapes2dC1
	:align: center
	:alt: Shape functions in 1d
	:class: with-shadow
	:width: 70%

	First order shape functions (``"C1"``) on triangular and quadrilateral elements.



..  figure:: shapes2dC2.*
	:name: figspatialshapes2dC2
	:align: center
	:alt: Shape functions in 1d
	:class: with-shadow
	:width: 100%

	Second order shape functions (``"C2"``) on triangular and quadrilateral elements.


