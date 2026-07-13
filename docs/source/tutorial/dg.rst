Discontinuous Galerkin methods
==============================

So far, the considered solutions were always approximated by shape functions which are continuous in space. One exception is the Crouzeix-Raviart elements in :numref:`secspatialcr`, where the pressure was allowed to discontinuously jump between two elements. However, in this particular case, the pressure is just an auxiliary field enforcing the incompressibility. 

In general, any field can be approximated by a discontinuous discretization, which can be helpful if the solution itself becomes (close to) discontinuous, e.g., in the case of shock waves. In this case, the standard finite element method (*Continuous Galerkin method*, CG) is not well suited to capture the solution accurately. The *Discontinuous Galerkin* (DG) method is a generalization which allows for such discontinuous solutions. The idea is to consider the solution in each element separately and to allow for discontinuities at the element interfaces. The solution is then approximated by a piecewise polynomial function which is continuous in each element but can have jumps at the element interfaces. 

In that case, however, it is important to incorporate these jumps into the weak formulation. This can be done by introducing so-called *numerical fluxes* at the element interfaces. These fluxes are used to enforce the continuity of the solution across the element interfaces in a weaker sense.

One can also think of the DG method as a generalization of finite volume methods. In the latter, the approximations are usually constant in each element, whereas in the DG method, the approximations can be piecewise polynomial.

For each of the continuous spaces, first (``"C1"``) and second order (``"C2"``) and the corresponding spaces ``"C1TB"`` and ``"C2TB"`` enriched by a bubble on triangular/tetrahedral elements, pyoomph provides discontinuous versions, namely ``"D1"``, ``"D2"``, ``"D1TB"`` and ``"D2TB"``. The discontinuous spaces can be used in the same way as the continuous spaces, but the weak formulation will have to be modified to incorporate the jumps at the element interfaces.

Opposed to the pure elemental discontinuous spaces ``"D0"`` and ``"DL"`` discussed so far, the values of ``"D1"``, ``"D2"``, ``"D1TB"`` and ``"D2TB"`` can be accessed directly at interfaces (i.e. without specifying ``domain=".."`` when binding them via :py:func:`~pyoomph.expressions.generic.var` or :py:func:`~pyoomph.expressions.var_and_test`). You can also set strong Dirichlet boundary conditions on these spaces, which is also not possible for ``"D0"`` and ``"DL"``.

.. toctree::
   :maxdepth: 5
   :hidden:
   
   dg/advdiffu.rst     
   dg/weakdirichlet.rst     


