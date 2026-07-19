.. _secmultidomheatcond:

Temperature conduction through two bodies of different conductivity
-------------------------------------------------------------------

For a simple start, a static (i.e. non-moving) one-dimensional mesh consisting of two domains will be considered. The mesh should contain two interval domains ``"domainL"`` and ``"domainR"``, ranging from :math:`0` to :math:`x_\text{I}` and from :math:`x_\text{I}` to :math:`L` respectively, which are connected at a mutual interface ``"interface"`` at :math:`x_\text{I}`. We use the :py:class:`~pyoomph.meshes.mesh.MeshTemplate` class to provide such a mesh (cf. :numref:`secspatialmeshgen` for details):

.. literalinclude:: temperature_conduction.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: self.add_facet_to_boundary("right",[nodesB[-1]])

Note how we create two domains with the :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.new_domain` calls and add line elements to both of these domains. During the latter, we use ``zip`` with a shifted node list to get the nodes in pairs, i.e. (node0,node1), (node1,node2), etc., to build the elements. It is important to note, since :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.add_node_unique` will not create new nodes if a node is already existing at a point, that ``domainA[-1]`` is the very same node as ``domainB[0]``. Thereby, this node, which is marked to be the interface ``"interface"``, is part of both domains.

.. warning::

   If you want to couple the equations on different domains at mutual interfaces, all involved domains have to be generated within the very same :py:class:`~pyoomph.meshes.mesh.MeshTemplate` (or :py:class:`~pyoomph.meshes.gmsh.GmshTemplate`). It is not possible to couple e.g. two :py:class:`~pyoomph.meshes.simplemeshes.LineMesh` instances at an interface, not even if the position and the name of the interface are matching.

On the domains A and B, we want to solve the (nondimensional) temperature conduction equations, i.e. the Poisson equations

.. math::

   \begin{aligned}
   \nabla\cdot\left(k_\text{A}\nabla T_\text{A}\right)&=0&\text{on domainA}\\
   \nabla\cdot\left(k_\text{B}\nabla T_\text{B}\right)&=0&\text{on domainB}
   \end{aligned}

with the, in general different, thermal conductivities :math:`k_\text{A}` and :math:`k_\text{B}`. The solution shall be subject to the boundary conditions :math:`T_\text{A}(0)=0` and :math:`T_\text{B}(L)=1`. At the mutual interface ``"interface"`` at :math:`x_\text{I}`, we want to have a continuous temperature, i.e. :math:`T_\text{A}(x_\text{I})=T_\text{B}(x_\text{I})`. While the former boundary conditions can be realized by trivial :py:class:`~pyoomph.meshes.bcs.DirichletBC`, the latter requires some additional consideration, since it involves the temperature field on two different domains. We can write the boundary condition as a constraint with an associated Lagrange multiplier :math:`\lambda` defined on the interface ``"interface"``. As usual, the constraint can be thought of as a minimization of the Lagrange multiplier contribution

.. math:: \lambda \; \left(T_\text{A}-T_\text{B}\right)

with respect to :math:`\lambda`, :math:`T_\text{A}` and :math:`T_\text{B}`. Let the corresponding test functions be :math:`\eta`, :math:`\Theta_\text{A}` and :math:`\Theta_\text{B}`, then the corresponding weak terms read

.. math:: :label: eqmultidomcontitweak

   \left\langle T_\text{A}-T_\text{B},\eta \right\rangle+\left\langle \lambda,\Theta_\text{A} \right\rangle+\left\langle -\lambda,\Theta_\text{B} \right\rangle

In pyoomph, we can write again an :py:class:`~pyoomph.generic.codegen.InterfaceEquations` class for this:

.. literalinclude:: temperature_conduction.py
   :language: python
   :start-at: class ConnectTAtInterface(InterfaceEquations):
   :end-at: self.add_residual(weak(-lagr,opp_test)) # Lagrange Neumann contribution to the outside domain

We introduce again the Lagrange multiplier :math:`\lambda` at the interface and add the weak contributions to the residuals. Later on, the ``ConnectTAtInterface`` object will be added to the ``"interface"`` of either ``"domainA"`` or ``"domainB"``, i.e. to ``@"domainA/interface"`` or ``@"domainB/interface"``. In both domains, we will have the temperature field ``var("T")`` defined. To distinguish between the fields on the inside (i.e. the domain where the ``ConnectAtInterface`` is attached to) and the outside (i.e. the opposite domain), we must use :py:meth:`~pyoomph.generic.codegen.Equations.get_opposite_side_of_interface` for the ``domain`` to clearly state that we want to get the temperature of the opposite side of the interface, i.e. the temperature ``var("T")`` at the ``"interface"``, but evaluated at the opposite domain. Alternatively, we could have used ``var("T",domain="|.")`` as shortcut. It will also return the temperature field of the opposite side of the interface. To access the opposite bulk domain instead, use :py:meth:`~pyoomph.generic.codegen.Equations.get_opposite_parent_domain` as ``domain`` or the shortcut ``var("T",domain="|..")``.

The driver code is quite trivial

.. literalinclude:: temperature_conduction.py
   :language: python
   :start-at: class TwoDomainTemperatureConduction(Problem):
   :end-at: problem.output()

..  figure:: temp_conduction_1d.*
	:name: figmultidomtempconduction1d
	:align: center
	:alt: Temperature conduction in two different domains with different conductivity.
	:class: with-shadow
	:width: 70%

	Temperature conduction in two different domains with different conductivity, coupled at the mutual interface.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <temperature_conduction.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   	
		    


We add only one mesh, but assemble two Poisson equations, each with a different ``coefficient`` and with different :py:class:`~pyoomph.meshes.bcs.DirichletBC` terms. At the very end, the equations are restricted to ``"domainA"`` and ``"domainB"``, respectively. The ``ConnectTAtInterface`` can be either added to ``"domainA/interface"`` or ``"domainB/interface"``, but not on both simultaneously, since this would overconstrain the problem. The result is plotted in :numref:`figmultidomtempconduction1d`.

In terms of physics within this problem, we wonder, of course, whether the heat flux :math:`\vec{q}=-k\nabla T` is indeed the same across the interface. Since the normals in ``"domainA"`` and ``"domainB"`` at the ``"interface"`` obey the relation :math:`\vec{n}_\text{A}=-\vec{n}_\text{B}`, a continuous heat flux would mean that

.. math:: :label: eqmultidomcontitqflux

   \left(\vec{q}_\text{A}-\vec{q}_\text{B}\right)\cdot\vec{n}_\text{A}=-k_\text{A}\partial_xT_\text{A}+k_\text{B}\partial_xT_\text{B}=0\,.

From the results in :numref:`figmultidomtempconduction1d`, we see that :math:`\partial_xT_\text{A}=0.8` and :math:`\partial_xT_\text{B}=0.2`, and due to :math:`k_\text{A}=0.5` and :math:`k_\text{B}=2`, it is indeed fulfilled. This is not just coincidence! From the weak form :math:numref:`eqmultidomcontitweak` of the enforced continuity of :math:`T`, we see that we impose :math:`\lambda` as a Neumann flux to ``"domainA"`` and :math:`-\lambda` to ``"domainB"``. The Neumann flux is exactly the heat flux and so continuity of this relation is actually a result of the enforcing. This also works, if the temperatures have a prescribed offset :math:`\Delta T` in the enforcing, which would read

.. math:: \left\langle T_\text{A}-T_\text{B}-\Delta T,\eta \right\rangle+\left\langle \lambda,\Theta_\text{A} \right\rangle+\left\langle -\lambda,\Theta_\text{B} \right\rangle

Hence, when enforcing continuity of fields across interfaces this way, one automatically gets the correct physics, here the continuity of the transported heat across the interface. However, :math:numref:`eqmultidomcontitqflux` would be violated if the weak forms of the Poisson equations would be e.g. multiplied by a different factor in both domains, since this factor would also affect the Neumann term. Therefore, one has to pay attention.

.. warning::

   Due to the above argument, one should not be tempted to set different scalings via :py:meth:`~pyoomph.generic.problem.Problem.set_scaling` of the :py:class:`~pyoomph.generic.problem.Problem` class or by the :py:class:`~pyoomph.equations.generic.Scaling` in dimensional problems. This can easily invalidate the continuity of the Neumann flux, which can lead to unphysical behavior. If one uses the same scale for non-dimensionalization of e.g. the temperature, e.g. by setting ``set_scaling(T=1*kelvin)`` at :py:class:`~pyoomph.generic.problem.Problem` level, this issue can be circumvented.

.. note::

   It is cumbersome to write a coupling interface like the ``ConnectTAtInterface`` here for every field you want to connect at inter-domain interfaces. pyoomph already has the predefined class :py:class:`~pyoomph.equations.generic.ConnectFieldsAtInterface`, which allows enforcing continuity of scalar fields. In the current problem, we could just use ``ConnectFieldsAtInterface("T")`` instead of our custom class ``ConnectTAtInterface``.
