.. _secmiscquadrature:

Controlling the spatial integration order
-----------------------------------------

Pyoomph uses the Gauss quadrature to perform the spatial integrals of the weak formulations. These are applied per element, i.e. each element is integrated separately and the result for the residual and Jacobian of all elements is accumulated. By default, the order of the Gauss quadrature depends on the order of the element, i.e. elements with maximum space ``"C1"`` will be integrated by a quadrature formula that is accurate for at least second order polynomials, whereas if ``"C2"`` spaces are used, the formula accurate for at least fifth order polynomials will be used. If one forms linear problems, these approximations are always accurate since any combination of shape and test functions cannot exceed these orders. However, for nonlinear problems, the integrated expression might not even be a polynomial at all, so that its Taylor expansion will exceed any polynomial order and small inaccuracies will occur. Usually, it does not contribute a lot to the error, but one can modify the integration order in two ways.

Either, the default integration order can be set globally by the :py:attr:`~pyoomph.generic.problem.Problem.default_spatial_integration_order` property of the problem class. It takes a positive ``int`` value to account for the number of nodes per one-dimensional element for which it should be exact in case of a linear expression (e.g. :math:`2` for ``"C1"`` and :math:`3` for ``"C2"``, since a 1d line element has exactly this amount of nodes). The integration orders :math:`2`-:math:`5` are implemented. Any lower or higher value will select the lower or upper value.

Alternatively, you can set the integration order for a domain - and, if not overridden in the same way, also for all its interfaces - by adding a :py:class:`~pyoomph.equations.generic.SpatialIntegrationOrder` object to the equation tree, where the order :math:`2`-:math:`5` has to be passed to the constructor.

You can hence easily check whether a higher integration order gives better results. This is rarely a vast improvement and comes at the cost of increased computation time for the assembly of the residual and Jacobian, but one can (and should) give it a try if the problem is nonlinear.

