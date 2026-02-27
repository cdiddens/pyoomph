.. _secspatialhelicalmesh:

A helical line mesh & differential operators on manifolds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Until now the meshes have always be conforming in the number of dimensions, i.e. either one-dimensional meshes with one-dimensional elements or two-dimensional meshes with two-dimensional elements. However, you can also have a mesh with one-dimensional elements embedded in a two-dimensional or three-dimensional space. The same holds for a mesh consisting of two-dimensional elements embedded in a three-dimensional space. These meshes represent manifolds with a non-vanishing *co-dimension*. We will now create a line mesh that resembles a helical shape, i.e. has a co-dimension of 2:

.. code:: python

   from pyoomph import *
   from pyoomph.equations.poisson import *  # use the pre-defined Poisson equation


   class HelicalLineMesh(MeshTemplate):
       def __init__(self, N=100, radius=1, length=5, windings=4, domain_name="helix"):
           super(HelicalLineMesh, self).__init__()
           self.N = N
           self.radius = radius
           self.length = length
           self.windings = windings
           self.domain_name = domain_name

       def define_geometry(self):
           domain = self.new_domain(self.domain_name, 3)  # Domain, but with 3d nodes

           # function to get the node based on a parameter l from [0:1]
           def node_at_parameter(l):
               x = self.radius * cos(2 * pi * self.windings * l)
               y = self.radius * sin(2 * pi * self.windings * l)
               z = self.length * l
               return self.add_node_unique(x, y, z)

           # loop to generate the elements
           for i in range(self.N):
               n0 = node_at_parameter(i / self.N)  # constructing nodes
               n1 = node_at_parameter((i + 0.5) / self.N)
               n2 = node_at_parameter((i + 1) / self.N)
               domain.add_line_1d_C2(n0, n1, n2)  # add a second order line element
               if i == 0:  # Marking the start boundary:
                   self.add_nodes_to_boundary("start", [n0])
               elif i == self.N - 1:  # Marking the end boundary:
                   self.add_nodes_to_boundary("end", [n2])

Note how we name the domain by default ``"helix"``, so that we must add equations to the domain ``"helix"`` in the :py:meth:`~pyoomph.generic.problem.Problem.add_equations` method of the problem class to restrict them on this helix. Furthermore, in the :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.new_domain` calls, we add a second argument, ``3``, which sets the nodal dimension space of this domain to :math:`3`. The rest works analogous to the previous example, however, this time we create second order line elements instead of first order quadrilateral elements by the call of :py:meth:`~_pyoomph.MeshTemplateElementCollection.add_line_1d_C2`. Due to the second order, we must supply a third node so that we in total have a start, a center and an end node of each line element. pyoomph automatically converts first order elements to second order elements, if equations on the space ``"C2"`` are defined on this domain. Vice versa, if we only have ``"C1"`` fields defined on a domain, pyoomph will simplify all generated second order elements to first order elements.

A potential driver code could read

.. code:: python

   class MeshTestProblem(Problem):
       def define_problem(self):
           self.add_mesh(HelicalLineMesh())
           eqs = MeshFileOutput()
           x, y, z = var(["coordinate_x", "coordinate_y", "coordinate_z"])
           source = x ** 2 + 5 * y * z + z
           eqs += PoissonEquation(name="u", source=source, space="C2")
           eqs += DirichletBC(u=0) @ "start"
           eqs += DirichletBC(u=0) @ "end"
           self.add_equations(eqs @ "helix")


   if __name__ == "__main__":
       with MeshTestProblem() as problem:
           problem.solve(spatial_adapt=4)
           problem.output_at_increased_time()


..  figure:: helix.*
	:name: figspatialmeshtemplate2
	:align: center
	:alt: Poisson equation on a helical manifold
	:class: with-shadow
	:width: 70%

	Poisson equation solved on a helical manifold, where the differential operators have been implicitly replaced by their counterparts acting on manifolds.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <mesh_helical_line.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    


At this stage, one might wonder, what the :py:class:`~pyoomph.equations.poisson.PoissonEquation` actually does. It should solve

.. math:: -\nabla^2 u=g

or likewise in weak formulation

.. math:: \left(\nabla u,\nabla v\right)-\left(g,v\right)=0\,.

However, what should :math:`\nabla^2 u` or :math:`\nabla u` mean here? If the helix is unfolded, it is just a one-dimensional function. We can parameterize :math:`u` by a single parameter :math:`s`, i.e. :math:`u=u(s)`, where :math:`s` could be e.g. the arc length along the helix. However, how should we apply a gradient or a Laplacian here? The conventional definition of :math:`\nabla=(\partial_x,\partial_y,\partial_z)` cannot work, since we cannot derive :math:`u` in all these directions, but only along the helix, i.e. with respect to the parameterization :math:`s`, i.e. :math:`\partial_s u`.

Whenever pyoomph notices that we have a co-dimension, all spatial derivatives like :py:func:`~pyoomph.expressions.generic.grad` or :py:func:`~pyoomph.expressions.div` will be internally expanded in a different manner. Instead of the conventional gradient or divergence, the corresponding differential operator reasonable for the actual manifold is selected. These are defined as follows: Let :math:`n` be the dimension of the embedding space, i.e. the dimension of the nodal coordinates, and :math:`e` be the element dimension. The co-dimension is obviously :math:`k=n-e`. The manifold spanned by the elements can be parameterized (at least locally) by :math:`e` parameters :math:`\xi^\alpha` for :math:`\alpha=1,\ldots,e`. The :math:`n`-dimensional position vector hence reads :math:`\vec{R}(\xi^\alpha)`. We construct the tangents, i.e. the *covariant basis vectors* by

.. math:: \vec{g}_\alpha=\frac{\partial\vec{R}}{\partial\xi^\alpha}

And define the *covariant metric tensor* :math:`\mathbf{g}` by :math:`g_{\alpha\beta}=\vec{g}_\alpha\cdot\vec{g}_\beta`. The inverse of :math:`\mathbf{g}`, the *contravariant metric tensor*, is denoted by the components :math:`g^{\alpha\beta}`. With that, we define the scalar gradient of a field :math:`\phi` on the manifold as

.. math:: :label: eqspatialsurfacegrad

   \text{grad(phi)}=\nabla_S \phi=g^{\alpha\beta}\vec{g}_\alpha \frac{\partial \phi}{\partial \xi^\beta}

where we sum over all :math:`\alpha` and :math:`\beta`. The result is an :math:`n`-dimensional vector in the local tangent space of the manifold, pointing in the direction of the largest increase of :math:`\phi` along the surface. The choice of the particular parameterization of the manifold does not influence the result.

To illustrate that, let us consider the case of a 1d manifold embedded in a 3d space, as in our helix. Let :math:`\xi` be our only parameter :math:`\xi^1`. We get the covariant basis vector, which is just a non-normalized tangent to the helix, by

.. math:: \vec{g}=\vec{g}_1=\frac{\partial\vec{R}(\xi)}{\partial \xi}

is indeed the local normalized tangent on the helix. The metric tensor is just :math:`\mathbf{g}=[g_{11}]=[\vec{g}\cdot\vec{g}]=[(\partial_\xi\vec{R})^2]`. Hence, the contravariant metric tensor is just given by the single component :math:`g^{11}=1/g_{11}`. With that, :math:`\nabla_S\phi` reads according to the definition :math:numref:`eqspatialsurfacegrad`

.. math:: \nabla_S \phi=\frac{1}{(\partial_\xi\vec{R})^2}\left(\partial_\xi\vec{R}\right) \frac{\partial \phi}{\partial \xi}\,.

When defining the normalized tangent to the helix as :math:`\vec{t}=\partial_\xi\vec{R}/\|\partial_\xi\vec{R}\|`, we get

.. math:: \nabla_S \phi=\left(\frac{1}{\|\partial_\xi\vec{R}\|}\frac{\partial \phi}{\partial \xi}\right)\cdot \vec{t}\,.

Upon reparameterization with the arc length :math:`s=\xi\|\partial_\xi\vec{R}\|`, we arrive at

.. math:: \text{grad(phi)}=\nabla_S \phi=\frac{\partial \phi}{\partial s}\cdot \vec{t}\,,

which is indeed the slope of :math:`\phi` along the manifold pointing in tangential direction.

The divergence of a vector field :math:`\vec{\psi}` defined on a manifold reads similar to :math:numref:`eqspatialsurfacegrad`, namely

.. math:: :label: eqspatialsurfacediv

   \text{div(psi)}=\nabla_S\cdot \vec{\psi}=g^{\alpha\beta}\vec{g}_\alpha\cdot \frac{\partial \vec{\psi}}{\partial \xi^\beta}

Finally, when changing the coordinate system by :py:meth:`~pyoomph.generic.problem.Problem.set_coordinate_system`, the corresponding scale factors are also considered in the differential operators on manifolds.

As conclusion, we can just use :py:func:`~pyoomph.expressions.generic.grad` and :py:func:`~pyoomph.expressions.div` in our equations. When any equation involving these differential operators is restricted to a manifold, the only reasonable differential operator is selected automatically. This allows to use the same :py:class:`~pyoomph.equations.poisson.PoissonEquation` either in the bulk (with co-dimension 0) or at any manifold with co-dimension :math:`>0`. In the latter case, the Poisson equation reads

.. math:: -\nabla_S^2 u=g

with the *Laplace-Beltrami operator* :math:`\nabla_S^2` or likewise in weak formulation

.. math:: \left(\nabla_S u,\nabla_S v\right)-\left(g,v\right)=0\,.

.. warning::

   Spatial adaptivity does not work yet on domains with a co-dimension, but it will be implemented soon. Hence, we cannot use at :py:class:`~pyoomph.equations.generic.SpatialErrorEstimator` here yet.

.. tip::

   oomph-lib also has a tutorial on the definition and calculation of surface gradients and surface divergences at https://oomph-lib.github.io/oomph-lib/doc/navier_stokes/surface_theory/html/index.html.
