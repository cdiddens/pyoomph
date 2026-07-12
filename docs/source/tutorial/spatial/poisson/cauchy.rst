.. _secspatialcauchybc:

Cauchy boundary condition
~~~~~~~~~~~~~~~~~~~~~~~~~

A Cauchy boundary condition consists of enforcing a Dirichlet value :math:`u_\text{D}` and a Neumann flux :math:`j_\text{N}` simultaneously, i.e.

.. math:: u|_\Gamma=u_\text{D} \qquad\text{and}\qquad \left(\nabla u\cdot \vec{n}\right)|_\Gamma=j_\text{N}\,.

This kind of boundary condition is hard to impose in the finite element method. Enforcing the Dirichlet condition would require the test function :math:`v` to vanish on the boundary. Then, however, any Neumann-type boundary integral contributions :math:`\langle j_\text{N}, v \rangle` cannot contribute, since :math:`v` is :math:`0`. Additionally, for e.g. the 1d Poisson equation discussed here, one can impose in total two boundary conditions. If e.g. a Cauchy boundary condition is imposed on the left side, the right side cannot have any boundary conditions. However, omitting the specification of a boundary condition in finite elements automatically leads to a zero-flux Neumann boundary condition, as discussed earlier. This would be in total three boundary conditions, more than possible to impose. In principle, one can however enforce a Cauchy boundary condition by enforcing the value strongly, i.e. with a :py:class:`~pyoomph.meshes.bcs.DirichletBC`, and adding a custom :py:class:`~pyoomph.generic.codegen.InterfaceEquations` class that monitors the slope and adjusts the opposite boundary condition accordingly, i.e. via a Lagrange multiplier that is defined in an :py:class:`~pyoomph.generic.codegen.ODEEquations` helper class. Effectively, this would be a kind of a *shooting method*. However, this is not discussed here.
