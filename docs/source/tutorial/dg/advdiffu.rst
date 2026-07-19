Advection-diffusion equation with an upwind scheme
--------------------------------------------------

To illustrate how to define the necessary facet terms in the weak formulation of DG methods, we will refer back to the example of the advection-diffusion equation from :numref:`secpdecconvdiffusupg` where we discussed the SUPG stabilization.

In strong formulation, we want to solve

.. math::

   \partial_t c + \nabla\cdot (\vec{u} c) - \nabla \cdot (D \nabla c)  = 0

Upon multiplication with a test function :math:`d` and subsequent spatial integration, we get

.. math::

   \left(\partial_t c,d \right) +\left(\nabla \cdot (\vec{u} c),d\right) - \left(\nabla \cdot (D \nabla c),d\right)  = 0

Our first focus is now the advection term, :math:`(\nabla \cdot (\vec{u} c),d)`, which can be split into the contributions of each element :math:`E`, i.e.

.. math::

   \begin{aligned}
   \left(\nabla \cdot (u c),d\right)&=\sum_{E} \left(\nabla \cdot (\vec{u} c),d\right)_E\\
   &=\sum_{E} \left\{-\left(\vec{u} c,\nabla d\right)_E+\left\langle\vec{n}\cdot \vec{u} c, d\right\rangle_{\partial E} \right\}
   \end{aligned}

Here, :math:`\vec{n}` is the normal pointing outward at each point on the boundary :math:`\partial E` of each element :math:`E`.
Opposed to continuous Galerkin methods, :math:`\vec{u}` and :math:`c` may be different on both sides of the boundary in discontinuous Galerkin methods. This means that the integrations along the interior element boundaries :math:`\partial E` do not cancel out by the corresponding interior boundary contribution of the adjacent element.

Nevertheless, the boundary integrals can be rearranged into the contributions of each facet :math:`F`, shared by two elements :math:`E^+` and :math:`E^-`. Each facet is only defined once in the mesh, even if it is shared by two elements. The facet normal :math:`\vec{n}` is pointing from the element :math:`E^+` to the element :math:`E^-`. The particular order, i.e. which element is assigned as :math:`E^+` and :math:`E^-` viewed from each facet :math:`F` is irrelevant here. We can now write it as bulk integration over the full domain :math:`\Omega`, its exterior boundary :math:`\partial \Omega`, and the interior facets :math:`F`:

.. math::

   \begin{aligned}
   \left(\nabla \cdot (u c),d\right)&=-\left(\vec{u} c,\nabla d\right)_\Omega+ \left\langle\vec{n}\cdot \vec{u} c, d\right\rangle_{\partial\Omega}\\
   &+  \sum_F\left\{\left\langle\vec{n}^+\cdot \vec{u}^+ c^+, d^+\right\rangle_{F}+\left\langle\vec{n}^-\cdot \vec{u}^- c^-, d^-\right\rangle_{F}\right\}
   \end{aligned}

Note that :math:`\vec{n}^-=-\vec{n}^+` holds, since the nodal coordinates are always continuous in pyoomph (however, it might not hold on curved manifold with a co-dimension). Also, we assume a continuous given velocity field, so that :math:`\vec{u}^-=-\vec{u}^+` holds as well. In CG methods, also :math:`c^-=c^+` and :math:`d^-=d^+` holds, so that the facet terms indeed cancel out. In DG methods, however, this is not the case. In fact, the interaction of :math:`c^+` and :math:`c^-` across the facet must be explicitly implemented in a reasonable way that reflects the nature of the equation. Here, we have to make sure that whatever is transported from one element to the other is also removed from the first element. 
To that end, we apply here the *upwind scheme* by transporting :math:`c^+` if :math:`\vec{n}^+\cdot\vec{u}^+>0` and :math:`c^-` otherwise. To that end, we define an auxiliary velocity normal to the facet, which makes sure to account only for upwind transport and vanishes otherwise   

.. math::

   u_n^{\text{up},\pm}=\frac{1}{2}\left(\vec{u}^\pm\cdot\vec{n}^\pm + |\vec{u}^\pm\cdot\vec{n}^\pm|\right)
   
and rewrite the facet terms of the advection by

.. math::

   \sum_F\left\langle u_n^{\text{up},+}c^+-u_n^{\text{up},-}c^-, d^+-d^-\right\rangle_{F}

Obviously, either :math:`u_n^{\text{up},+}` or :math:`u_n^{\text{up},-}` is positive and the other one vanishes. Thereby, we either transport :math:`c^+` or :math:`c^-`, depending on the direction of the velocity, by removing this flux from one element and adding it to the other, i.e. ensure conservation. 

Since such *jump* terms, i.e. the difference of an expression on the two sides of a facet, are common in DG methods, pyoomph offers the function :py:func:`~pyoomph.expressions.generic.jump`, which lets you write the above expression in pyoomph in a shorter notation. This will be addressed later in the implementation.

For the diffusion term, similar facet terms must be derived. Doing the same splitting in bulk contribution, exterior boundary, and interior facets, we get

.. math::

   -\left(D\nabla^2 c,d\right)=\left(D\nabla c,\nabla d\right)_\Omega- \left\langle D\vec{n}\cdot\nabla c, d\right\rangle_{\partial\Omega}+  \sum_F\left\{\left\langle J_n^+, d^+\right\rangle_{F}+\left\langle J_n^-, d^-\right\rangle_{F}\right\}

where we abbreviated the diffusive fluxes through the facet by

.. math::

   \vec{J}^\pm= -D\cdot\nabla c^\pm

Opposed to the upwind scheme for advection, the diffusive fluxes should be considered in a symmetric way, i.e. we can define the average on both sides 

.. math::

   \vec{J}_n^\text{avg}= \frac{1}{2}\left(\vec{J}_n^++\vec{J}_n^-\right)

For such a term, we can use the :py:func:`~pyoomph.expressions.generic.avg` function in pyoomph, which calculates the average of the expression on both sides of the facet. This will be addressed later in the implementation as well. Similar to the transport by advection, the diffusive fluxes must be removed from one element and added to the other, i.e. we must ensure conservation, which can be done by replacing the facet terms stemming from diffusion by 

.. math::

   \left\langle J_n^+, d^+\right\rangle_{F}+\left\langle J_n^-, d^-\right\rangle_{F}=\left\langle \vec{J}_n^\text{avg}\cdot\vec{n}^+, d^+-d^-\right\rangle_{F}=-\left\langle D \operatorname{avg}(\nabla c)\cdot\vec{n}^+, \operatorname{jump}(d)\right\rangle_{F}

While this considers a conservative and reasonable diffusive transport across the facet, this form neither penalizes huge jumps in the concentration :math:`c` (which should be smoothed by diffusion) nor is it symmetric. For a symmetric, consistent and coercive formulation we can weakly enforce continuity of :math:`c` in the sense of *Nitsche's method* by adding a symmetric and a penalty term to the facet terms, i.e. we write

.. math::

   \begin{aligned}
   \left\langle J_n^+, d^+\right\rangle_{F}+\left\langle J_n^-, d^-\right\rangle_{F}=&-\left\langle D \operatorname{avg}(\nabla c)\cdot\vec{n}^+, \operatorname{jump}(d)\right\rangle_{F}\\
   &-\left\langle D \operatorname{jump}(c)\cdot\vec{n}^+, \operatorname{avg}(\nabla d)\right\rangle_{F}\\
   &+\left\langle D \frac{\alpha}{h}\operatorname{jump}(c),\operatorname{jump}(d) \right \rangle_{F}
   \end{aligned}

Here, :math:`\alpha` is a penalty parameter and :math:`h` is the average element size of both elements attached to the facet. The penalty term ensures coercivity of the problem, i.e. the existence of a unique solution. The penalty parameter :math:`\alpha` should be chosen large enough to ensure coercivity, but small enough to not dominate the solution. The average element size :math:`h` is calculated by the average of the element sizes of both elements attached to the facet. 

For the implementation, the equation starts as usual. We can pass an arbitrary finite element space, continuous or discontinuous:

.. literalinclude:: convection_diffusion.py
   :language: python
   :start-at: class ConvectionDiffusionEquation(Equations):
   :end-at: self.define_scalar_field("c", self.space)

Whenever facet terms must be added, as here, we must set the property :py:attr:`~pyoomph.generic.codegen.BaseEquations.requires_interior_facet_terms` to ``True``. This will trigger the generation of an interior skeleton mesh which includes all interior facets of the mesh. Exterior boundaries are not part of this skeleton mesh. Each facet appears only once in the interior skeleton mesh, although each interior facet is shared by two elements.
As an auxiliary method, the function :py:func:`~pyoomph.expressions.generic.is_DG_space` is used, which automatically detects if the space is discontinuous.

The definition of the weak form now also tests whether we have an continuous space or must add additional facet terms:

.. literalinclude:: convection_diffusion.py
   :language: python
   :start-at: def define_residuals(self):
   :end-at: self.add_interior_facet_residual(facet_terms)

Again, we make use of :py:func:`~pyoomph.expressions.generic.is_DG_space` to check whether the space is discontinuous. If this is the case, we define the facet terms.
For these terms, we first define the upwind convection, which is only giving a contribution if the velocity is advecting in the same direction as the facet normal, which can be bound by ``var("normal")``. The normal of the element itself, i.e. not the facet normal, can be obtained by ``var("normal",domain="..")``. This is a consequence from the expansions of the variables at the domain where you add the residual terms, which are here the facets. For the elements, you hence must go one level up in the domain hierarchy.

We furthermore define the average element length :math:`h`, which is calculated by the average :py:func:`~pyoomph.expressions.generic.avg` of both element sizes attached to the facet. The element size can be obtained by the special variable name ``var("cartesian_element_length_h")``. You can also use ``var("element_length_h")`` instead, which is identical for Cartesian coordinate systems, however, for other coordinate systems, both will differ. In fact, these lengths are calculated by integrating the elemental length/area/volume (depending on the element dimension :math:`N_\text{el}`) and subsequently taking the :math:`N_\text{el}`-th root of the result. The length/area/volume calculation is done in a Cartesian coordinate system for ``var("cartesian_element_length_h")`` and in the coordinate system of the current equations for ``var("element_length_h")``. In dimensional problems, both consider the scaling ``scale_factor("spatial")``, i.e. :math:`h` will be measured in meters if the problem is dimensional.

Also, we use :py:func:`~pyoomph.expressions.generic.jump` to calculate the jumps of the variables across the facets. By default, both :py:func:`~pyoomph.expressions.generic.avg` and :py:func:`~pyoomph.expressions.generic.jump` expands all contained variables at the bulk element domains on both sides. Both attached bulk elements can also be accessed directly via the domains ``"+"`` and ``"-"``. Here, we could e.g. replace ``avg(var("cartesian_element_length_h"))`` by ``(var("cartesian_element_length_h",domain="+"))+var("cartesian_element_length_h",domain="-"))/2``, which is exactly what :py:func:`~pyoomph.expressions.generic.avg` does.

However, in the last :py:func:`~pyoomph.expressions.generic.jump`, we want to calculate the jump of the upwind convection term. Since this term requires the facet normal, we must make sure that ``var("normal")`` is indeed evaluated at both sides of the facet, not the attached bulk elements. Therefore, we must use the ``at_facet=True`` flag to indicate that the variables should be expanded at the facet domain. It is also possible to explicitly expand on both sides of the facet domain by using the domains ``"+|"`` and ``"|-"``, respectively.

Note that one cannot know a priori which element is on the ``"+"`` and which is on the ``"-"`` side of the facet. The order of the elements is not guaranteed. However, in reasonable formulations of such facet terms, the order of the elements should not matter. 

Finally, we add the facet terms to the skeleton mesh of the facets by the :py:meth:`~pyoomph.generic.codegen.BaseEquations.add_interior_facet_residual` method.

The problem code reads as before, but we use the keyword argument ``discontinuous=True`` to obtain a better output to include the discontinuities. Thereby, each element writes its nodes individually to the output, i.e. overlapping nodes are multiple times present in the output. The same also works for the :py:class:`~pyoomph.output.meshio.MeshFileOutput` class to write VTU files in higher dimensions. The default ``discontinuous=False`` will just take the average value at overlapping nodes for the output.

.. literalinclude:: convection_diffusion.py
   :language: python
   :start-at: class OneDimAdvectionDiffusionProblem(Problem):
   :end-at: problem.run(50,outstep=1,maxstep=0.1)

The discontinuous output is plotted in :numref:`figdgconvdiffu`.

..  figure:: dgconvdiffu.*
	:name: figdgconvdiffu
	:align: center
	:alt: Discontinuous Galerkin method for the advection-diffusion equation
	:class: with-shadow
	:width: 100%

	Discontinuous Galerkin method for the advection-diffusion equation on spaces ``"D1"`` (left) and ``"D2"`` (right). The solution is shown at time steps of 10. The discontinuities are clearly visible. Also, ``"D1"`` has too few degrees of freedom, which leads to severe numerical diffusion. This is typical for underresolved upwind schemes, as also seen in the SUPG implementation, cf. :numref:`figpdesupg`.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <convection_diffusion.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   	
		   

.. note::

   It is easily possible to switch the upwind scheme by a central scheme by replacing ``un_upwind`` by
   ``un_central=dot(self.u, n)/2``
