.. _secdiffops:

Differential operators
----------------------

Spatial derivatives
^^^^^^^^^^^^^^^^^^^

.. list-table:: Spatial differential operators
    :widths: 50 50
    :header-rows: 0

    *   - :py:func:`grad(f) <pyoomph.expressions.generic.grad>`
        - Gradient :math:`\nabla f` of a scalar or vector field
    *   - :py:func:`div(f) <pyoomph.expressions.div>`
        - Divergence :math:`\nabla\cdot f` of a vector or rank-2 tensor field
    *   - :py:func:`directional_derivative(f,direction) <pyoomph.expressions.generic.directional_derivative>`
        - Advection operator :math:`(\vec{d}\cdot\nabla)f`, for a scalar, vector or rank-2 tensor :math:`f`
    *   - :py:func:`partial_x(f,[order=1]) <pyoomph.expressions.partial_x>`
        - :math:`\partial^\text{order} f/\partial x^\text{order}` with respect to the independent coordinate ``"coordinate_x"``, likewise :py:func:`~pyoomph.expressions.partial_y` and :py:func:`~pyoomph.expressions.partial_z`
    *   - :py:func:`diff(f,*vars) <pyoomph.expressions.generic.diff>`
        - Derivative with respect to one or more arbitrary variables, e.g. a field or a global parameter
    *   - :py:func:`symbolic_diff(f,x) <pyoomph.expressions.generic.symbolic_diff>`
        - The same, but held until code generation, i.e. differentiated only once all placeholders have been resolved

.. note::

    Only **first** derivatives of the basis functions are available, so a second spatial derivative of a field cannot be assembled: ``div(grad(u))`` is rejected with *Cannot handle second order derivatives of basis functions yet*. This is no loss, since in a finite element setting the Laplacian is integrated by parts anyway, i.e. written as ``weak(grad(u),grad(v))`` plus the corresponding Neumann term. If you really need a second derivative pointwise, introduce the first derivative as a field of its own.

    Note also that :py:func:`~pyoomph.expressions.generic.grad` takes a scalar or a vector, not a rank-2 tensor - the result would be rank 3, which is not implemented. :py:func:`~pyoomph.expressions.div` and :py:func:`~pyoomph.expressions.generic.directional_derivative`, on the other hand, do accept a rank-2 tensor.

All of these respect the **coordinate system** of the equations they are used in, i.e. in axisymmetry ``div(u)`` contains the :math:`u_r/r` term. A different system can be imposed per call with the ``coordsys`` argument of :py:func:`~pyoomph.expressions.generic.grad` and :py:func:`~pyoomph.expressions.div`; see :numref:`secspatialcoordsys`.

On a domain with a co-dimension, i.e. on an interface, :py:func:`~pyoomph.expressions.generic.grad` and :py:func:`~pyoomph.expressions.div` are the **surface** gradient and divergence - also for a field that lives in the bulk. To get the bulk gradient evaluated at the interface instead, take the gradient of the bulk-bound variable, i.e. ``grad(var("u",domain=".."))``.

Two further arguments are shared by both: ``lagrangian=True`` differentiates with respect to the Lagrangian instead of the Eulerian coordinates, and ``nondim=True`` with respect to the nondimensional ones, i.e. without the spatial scale factor.

.. warning::

    Mind the index order: ``grad(u)[i,j]`` is :math:`\partial u_i/\partial x_j`, the component first and the derivative direction second. Many texts write :math:`\nabla\otimes\vec{u}` the other way round, i.e. the transpose of this. As a consequence, the advection term :math:`(\vec{u}\cdot\nabla)\vec{u}` is ``matproduct(grad(u),u)``, equivalently ``grad(u) @ u`` or ``dot(grad(u),u)`` - whereas ``dot(u,grad(u))`` is :math:`\nabla(|\vec{u}|^2/2)`, which is a different thing. :py:func:`~pyoomph.expressions.div` contracts the second index accordingly, i.e. ``div(T)[i]`` is :math:`\partial_j T_{ij}`, which makes it the adjoint of the gradient and the integration-by-parts partner of ``weak(T,grad(v))``.

Temporal derivatives
^^^^^^^^^^^^^^^^^^^^

.. list-table:: Temporal differential operators
    :widths: 50 50
    :header-rows: 0

    *   - :py:func:`partial_t(f,[order=1]) <pyoomph.expressions.generic.partial_t>`
        - Time derivative :math:`\partial_t^\text{order} f`, discretized by the time stepping scheme
    *   - :py:func:`mesh_velocity() <pyoomph.expressions.generic.mesh_velocity>`
        - Mesh velocity :math:`\dot{\vec{X}}`, a shorthand for ``partial_t(var("mesh"),ALE=False)``
    *   - :py:func:`material_derivative(f,velocity) <pyoomph.expressions.generic.material_derivative>`
        - Material derivative :math:`\partial_t f + (\vec{u}\cdot\nabla) f`
    *   - :py:func:`upper_convected_derivative(A,velocity) <pyoomph.expressions.generic.upper_convected_derivative>`
        - Upper-convected derivative of a rank-2 tensor, i.e. the material derivative minus :math:`\mathbf{L}\mathbf{A}+\mathbf{A}\mathbf{L}^\mathrm{T}` with :math:`\mathbf{L}=\nabla\vec{u}`
    *   - :py:func:`convected_derivative(A,velocity,[alpha=0]) <pyoomph.expressions.generic.convected_derivative>`
        - Gordon-Schowalter derivative, interpolating between the upper- (``alpha=1``), lower- (``alpha=-1``) convected and the co-rotational Jaumann (``alpha=0``) derivative
    *   - :py:func:`time_derivative_of_integral(expr) <pyoomph.expressions.time_derivative_of_integral>`
        - :math:`\mathrm{d}/\mathrm{d}t` of an entire integral, i.e. including the change of the element size

On a **moving mesh**, a nodal time derivative is taken while the node itself moves, which is not the time derivative at a fixed position. The ``ALE`` argument controls the correction by :math:`-\dot{\vec{X}}\cdot\nabla f`: with the default ``ALE="auto"`` it is added whenever the mesh coordinates are unknowns, i.e. when :py:meth:`~pyoomph.generic.codegen.Equations.activate_coordinates_as_dofs` has been called (see :numref:`secALEtimediff`), and ``ALE=False`` gives the plain derivative of the nodal values. The latter is what you want when you are after the mesh velocity itself, which is why :py:func:`~pyoomph.expressions.generic.mesh_velocity` uses it.

For the same reason, ``weak(partial_t(u),v)`` and ``time_derivative_of_integral(weak(u,v))`` are not the same on a moving mesh: the latter also differentiates the integration measure, i.e. the element size.

The discretization of all time derivatives is set by the time stepping scheme, ``"BDF2"`` by default, which can be overridden per operator with the ``scheme`` argument and at problem level; see :numref:`secodetimestepping`.
