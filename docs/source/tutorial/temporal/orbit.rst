Periodic orbits
---------------

Frequently, a dynamical system (or even a PDE system with constraints) shows periodic orbits. Such orbits e.g. arise at Hopf bifurcations. A periodic orbit of a solution :math:`\vec{x}(t)` is defined by 

.. math:: :label: orbitdef

	\begin{aligned}
	\vec{R}(\partial_t\vec{x}(t),\vec{x}(t))&=0 \\
	\vec{x}(t+T)&=\vec{x}(t)
	\end{aligned}

Here, the orbit period :math:`T` is chosen uniquely so that :math:`T>0` is the smallest value fulfilling the periodicity equation (obviously, multiples of :math:`T` will also fulfill it).

We furthermore demand in the following that the time derivatives are of first order (higher order time derivatives can be again realized by auxiliary degrees of freedom) and that we can write our implicit residual formulation :math:`\vec{R}(\partial_t\vec{x}(t),\vec{x}(t))=0` as

.. math:: :label: orbitdefM

	\begin{aligned}
	\mathbf{M}(\vec{x})\partial_t\vec{x}(t)+\vec{R}_0(\vec{x})&=0 \\
	\vec{x}(t+T)&=\vec{x}(t)
	\end{aligned}
	
Here, :math:`\mathbf{M}(\vec{x})` is the mass matrix, which might depend on the unknowns, but - as a restriction here - not on the time derivatives. The time-independent residual :math:`\vec{R}_0` is just the part of the residual which is recovered when substituting :math:`\partial_t \vec{x}=0`, i.e. the same one used for stationary solutions. Usually, i.e. in almost all cases, :math:numref:`orbitdef` can be rewritten in the form :math:numref:`orbitdefM`, potentially by the usage of auxiliary degrees of freedom for e.g. terms nonlinear in :math:`\partial_t \vec{x}`.

If we have a good initial guess for the orbit, we can solve for the orbit :math:`\vec{x}(t)` for :math:`t\in [0,T)`. In pyoomph, this can be done in a monolithic, fully implicit manner. To that end the time-dependency of :math:`\vec{x}(t)` is discretized in time by state vectors, :math:`\vec{x}_l` with :math:`l=1,2,\ldots,N_T`, where :math:`N_T` is the number of considered discrete representations in time. The interpolation for arbitrary times in between can be done in different ways. Pyoomph e.g. offers periodic B-splines or conventional Lagrange polynomials to perform the interpolation in time. Subsequently, the system :math:numref:`orbitdefM` can be solved. However, note that we have an additional unknown, namely the exact period :math:`T`. Luckily, we also have an additional equation stemming from the invariance of :math:`\vec{x}(t)` by a shift in time, i.e. :math:`t\to t+c` for any constant `c`. This invariance can be removed by posing a constraint, which simultaneously serves as an equation for the unknown period :math:`T`. Pyoomph can either use a plane constraint, which demands that :math:`\vec{x}(0)` is located on a plane in phase space. This plane is chosen by the initial guess of :math:`\vec{x}_0(0)` (or its value from the previous step in an arc-length continuation of an orbit in a parameter). The normal of the plane is then given by :math:`\partial_t \vec{x}_0(0)`. Alternatively, we can pose a phase constraint. Here, we use the initial guess :math:`\vec{x}_0` (or the previous continuation step) to enforce that 

.. math:: :label: orbitphaseconstraint

	\begin{aligned}
	\int_0^T \partial_t \vec{x}_0 \cdot \vec{x} \:\mathrm{d}t =0
	\end{aligned}
	
	
holds. For :math:`\vec{x}=\vec{x}_0`, this is trivially fulfilled and therefore, it automatically constitutes a good guess for actual solving or continuation.


.. toctree::
   :maxdepth: 5
   :hidden:

   orbit/hopf_switch.rst
   orbit/manual_orbit.rst   
   orbit/floquet.rst      

