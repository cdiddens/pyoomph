.. _secspatialmesh1:

Defining nodes and elements by hand
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Until now, only trivial meshes, e.g. the 1d :py:class:`~pyoomph.meshes.simplemeshes.LineMesh` or the 2d :py:class:`~pyoomph.meshes.simplemeshes.RectangularQuadMesh` have been considered. However, in reality, these will rarely be sufficient for all kinds of problems. There are essentially two ways of creating meshes in pyoomph. The first way is to add all elements by hand. This can be a demanding task, but gives full control over the desired mesh. This approach can also be used to write custom classes to load a mesh from a file in any format.

To do so, we write a mesh class that inherits from the :py:class:`~pyoomph.meshes.mesh.MeshTemplate` class. Let us create a mesh that resembles an :math:`L`-shape:

.. code:: python

   from pyoomph import *
   from pyoomph.equations.poisson import * # use the pre-defined Poisson equation


   class LShapedMesh(MeshTemplate):
   	def __init__(self,Nx=10,Ny=5,H=1,W=1,domain_name="domain"):
   		super(LShapedMesh,self).__init__()
   		self.Nx=Nx
   		self.Ny=Ny
   		self.H=H
   		self.W=W
   		self.domain_name=domain_name

   	def define_geometry(self):
   		domain=self.new_domain(self.domain_name)
   		# row of quads in x direction
   		for ix in range(self.Nx):
   			x_l,x_u=self.W*ix,self.W*(ix+1) # lower and upper x coordinate
   			y_l, y_u = 0, self.H # lower and upper y coordinate
   			# add the corner nodes. These will be unique, i.e. no additional node will be added if it is already present
   			node_ll = self.add_node_unique(x_l,y_l)
   			node_ul = self.add_node_unique(x_u, y_l)
   			node_lu = self.add_node_unique(x_l, y_u)
   			node_uu = self.add_node_unique(x_u, y_u)
   			# add a quadrilateral element from (x_l,y_l) to (x_u,y_u)
   			domain.add_quad_2d_C1(node_ll,node_ul,node_lu,node_uu)
   			if ix==0: # Marking the left boundary:
   				self.add_facet_to_boundary("left",[node_ll,node_lu])

   		# row of quads in y direction
   		for iy in range(1,self.Ny): # we must start from 1, since the element in the corner is already present
   			x_l,x_u=self.W*(self.Nx-1),self.W*self.Nx # lower and upper x coordinate
   			y_l, y_u = self.H*iy, self.H*(iy+1) # lower and upper y coordinate
   			node_ll = self.add_node_unique(x_l,y_l)
   			node_ul = self.add_node_unique(x_u, y_l)
   			node_lu = self.add_node_unique(x_l, y_u)
   			node_uu = self.add_node_unique(x_u, y_u)
   			domain.add_quad_2d_C1(node_ll,node_ul,node_lu,node_uu)
   			if iy == self.Ny-1: # Marking the top boundary:
   				self.add_facet_to_boundary("top",[node_lu, node_uu])

Again, we pass arguments to the constructor, e.g. the number of elements ``Nx``\ :math:`\times`\ ``Ny`` in :math:`x` and :math:`y` direction. These are stored as properties of the class. The generation happens in the method :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.define_geometry`. A :py:class:`~pyoomph.meshes.mesh.MeshTemplate` consists of *nodes* and *elements*. Nodes are part of the :py:class:`~pyoomph.meshes.mesh.MeshTemplate` itself, whereas elements are stored in domains. This will allow to create multiple domains in the very same :py:class:`~pyoomph.meshes.mesh.MeshTemplate`, which is relevant for multi-domain problems in :numref:`secmultidom` later on. Therefore, before adding elements, we must create a domain to store these. This is done with the :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.new_domain` method, which takes a name of this domain as argument. The name is arbitrary, but it is relevant to identify the domain. In particular, in the :py:class:`~pyoomph.generic.problem.Problem` class, we have to restrict the equations to the very same domain name, e.g. in the call ``add_equations(eqs@"domain")``.

The, we perform a loop over the :math:`x`-direction. We calculate the corner :math:`x` and :math:`y` coordinates of each quad. Then, we add four nodes with the :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.add_node_unique` call. During the loop over ``ix``, a lot of nodes will be created multiple times. If the node is already present, :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.add_node_unique` will notice that and return the already previously added node instead of creating a new one. Thereby, adjacent elements will indeed share the very same mutual nodes. Finally in the ``ix`` loop, we add elements to the ``domain``. Here, we add first order quadrilateral elements with the method :py:meth:`~_pyoomph.MeshTemplate.add_quad_2d_C1`. It takes the four corner nodes as arguments, where it is necessary to add them in a zigzag order, i.e. starting with one corner (e.g. lower left), move to the next corner (e.g. lower right), then go diagonally (e.g. upper left) and finally the last corner (e.g. upper right). It is furthermore important to add the nodes in the order that the corner points in the argument order :math:`1`, :math:`2`, :math:`4`, :math:`3` form a counter-clockwise loop.

We do the same in :math:`y`-direction. However, opposed to nodes, it is not checked whether elements were already added. Therefore, it is important that the :math:`y`-loop starts with ``iy=1``, not ``iy=0``, to prevent the dual generation of the element in the corner of the L-shape.

Lastly, we also can add boundary markers. This is done with the :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.add_facet_to_boundary` method, that takes first an interface name and then a list of nodes defining a single facet. This function must be called for each facet on the boundary. For higher order facets, a further argument, namely a list of vertex nodes is required as additional argument.

..  figure:: meshtemplate1.*
	:name: figspatialmeshtemplate1
	:align: center
	:alt: L-shaped mesh
	:class: with-shadow
	:width: 80%

	Poisson equation on an L-shaped custom mesh without and with spatial adaptivity.


As an example how to use this mesh, we solve again a Poisson equation on this mesh, however, this time using the predefined :py:class:`~pyoomph.equations.poisson.PoissonEquation` from the module ``pyoomph.equations.poisson``:

.. code:: python

   class MeshTestProblem(Problem):
   	def define_problem(self):
   		self.add_mesh(LShapedMesh(Nx=6,Ny=4))
   		eqs=MeshFileOutput()
   		eqs+=PoissonEquation(name="u",source=0)
   		eqs+=DirichletBC(u=0)@"left"
   		eqs += DirichletBC(u=1) @ "top"
   		eqs+=SpatialErrorEstimator(u=1)
   		self.add_equations(eqs@"domain")

   if __name__=="__main__":
   	with MeshTestProblem() as problem:
   		problem.initial_adaption_steps=0
   		problem.solve(spatial_adapt=4)
   		problem.output_at_increased_time()

The predefined :py:class:`~pyoomph.equations.poisson.PoissonEquation` works as the one developed in :numref:`secspatialpoisson`. Note how we initially suppress the mesh adaption by setting :py:attr:`~pyoomph.generic.problem.Problem.initial_adaption_steps` to zero. Otherwise, we would get redundant adaption near the ``"top"`` boundary due to the non-zero :py:class:`~pyoomph.meshes.bcs.DirichletBC`. The custom mesh and the problem class in action can be seen in :numref:`figspatialmeshtemplate1`.


.. warning::
	The orientation of the elements can matter, in particular for refineable meshes. Therefore, it is advised to make sure that all elements are constructed by node indices in the same orientation. E.g. for a 2d mesh, the order of the nodes passed to :py:meth:`~_pyoomph.MeshTemplate.add_quad_2d_C1` can either lead to an element facing in positive or negative :math:`z`-direction. If elements of different orientation are connected in the very same mesh, this leads to issues upon spatial refinement. Therefore, make sure that all elements are oriented in the same direction by adjusting the order of the nodes passed to the construction of the elements. If it is wrong, pyoomph will raise an error, unless you set :py:attr:`~pyoomph.generic.problem.Problem.check_mesh_integrity` of the :py:class:`~pyoomph.generic.problem.Problem` class to ``False``. After doing so, you can easily check the mesh by *Paraview*. After outputting the mesh with :py:class:`~pyoomph.output.meshio.MeshFileOutput`, you can open it with Paraview and search for *Backface Representation* in the search box of the *Properties* box (hidden by default). Then, select *Cull Frontface* or *Cull Backface*. The entire mesh should be visible in one of these settings and entirely invisible in the other setting.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <mesh_Lshape_by_hand.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
