Test functions and the residual form
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose we have a system of :math:`N` equations for :math:`N` degrees of freedom. Let us assume the degrees of freedom are stored in the :math:`N`-dimensional vector of unknowns :math:`\vec{U}`. If all equations are written into residual form, we can assemble a vectorial residual vector :math:`\vec{\mathcal{R}}`, which gives the solution of the system once

.. math::
	:label: eqgenresresult

	 \vec{\mathcal{R}}(\vec{U})=\vec{0}\,. 

In the previous example of the harmonic oscillator, the system is one-dimensional, i.e. :math:`N=1`, :math:`\vec{U}=\left[y(t)\right]` and :math:`\vec{\mathcal{R}}=\left[\partial_t^2 y + 2\delta\partial_t y +\omega^2 y - f\right]`.

One can easily separate :math:numref:`eqgenresresult` into distinct equations again, if one takes the dot product with an arbitrary vector :math:`\vec{V}`, i.e.

.. math:: :label: eqscalresresult
	
	\vec{V}\cdot\vec{\mathcal{R}}(\vec{U})=\sum_{i=0}^{N} \mathcal{R}_i(\vec{U}) V_i =0 \,.

Of course, in order to ensure that both formulations are equivalent, it is fundamental to demand that :math:`\vec{V}` is arbitrary, i.e. :math:numref:`eqscalresresult` has to be fulfilled for all choices of :math:`\vec{V}`. However, due to linearity of the projection of :math:`\vec{R}` on :math:`\vec{V}`, it is sufficient to require that :math:numref:`eqscalresresult` has to be fulfilled for :math:`N` linearly independent choices of :math:`\vec{V}`, i.e. :math:`\vec{V}^{(j)}` for :math:`j=1,\ldots,N`. The trivial choice is of course to have :math:`\vec{V}^{(j)}_i=\delta_{ij}` using the Kronecker-:math:`\delta`, i.e. the :math:`i^\text{th}` component of the :math:`j^\text{th}` vector :math:`\vec{V}^{(j)}` is :math:`1`, all other entries are :math:`0`. This is exactly how it is internally handled by pyoomph when solving ODEs. We hence have

.. math:: :label: eqscalresresultspec
	
	\sum_{i=0}^{N} \mathcal{R}_i(\vec{U}) V^{(j)}_i =0 \quad \text{for}\quad j=1,\ldots,N\,.

Obviously, :math:numref:`eqscalresresultspec` with the particular choice of :math:`V^{(j)}` being the unit vectors in the :math:`j^\text{th}` direction, is exactly the same as :math:numref:`eqgenresresult`. Later on, when it comes to spatial differential equations, it is beneficial to express it as in :math:numref:`eqscalresresultspec`, i.e. this section serves as a precursor for the weak formulation later on.

When it comes to usability, it is not very handy to assign an index :math:`i` or :math:`j` to each degree of freedom by hand. However, since we now have introduced the vectors :math:`V^{(j)}`, which are indeed the so-called *test functions*, pyoomph allows you to select the test function corresponding to a degree of freedom by the unknown itself. In the example above, we have assigned ``y`` to the degree of freedom :math:`y(t)` by using the statement ``y=var("y")``. We get the corresponding test function :math:`\vec{V}^{(j)}` not by the index :math:`j`, but instead by the variable itself, i.e. by ``testfunction(y)``.

Finally, the statement ``self.add_residual(residual*testfunction(y))`` will add the residual part, i.e. the harmonic oscillator in residual form, multiplied by the corresponding test function to the total residual of the equation. This will become more clear in the next example.
