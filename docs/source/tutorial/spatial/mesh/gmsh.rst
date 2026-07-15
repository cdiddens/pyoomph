.. _secspatialgmsh:

Generating meshes from points and lines via Gmsh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As we have seen on the basis of the fish mesh, one can create rather complicated domains already by hand. However, it can be quite cumbersome to add all elements by hand and map facets on curved boundaries, in particular if the geometry is rather complex. One can intrinsically let the meshing tool gmsh do this job for you, which will be discussed in the following. To that end, we have to inherit our mesh from the :py:class:`~pyoomph.meshes.gmsh.GmshTemplate` class.

The constructor looks - besides the chosen class name ``GmshFishMesh`` - the same:

.. literalinclude:: mesh_gmsh_fish_mesh_modes.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: self.domain_name = domain_name

In the :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.define_geometry` method, however, there are multiple changes:

.. literalinclude:: mesh_gmsh_fish_mesh_modes.py
   :language: python
   :start-at: def define_geometry(self):
   :end-at: self.plane_surface("mouth","fin","curved",name=self.domain_name)

First of all, we do not need to call :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.nondim_size` to get nondimensional coordinates. In fact, the :py:class:`~pyoomph.meshes.gmsh.GmshTemplate` expects dimensional coordinates if a spatial dimension is set via ``set_scaling(spatial=...)`` in the problem class where the mesh is used. Next, instead of using :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.add_node_unique`, we add points via :py:meth:`~pyoomph.meshes.gmsh.GmshTemplate.point`. We then do not create any elements by hand at all. Also we do not create any domain via :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.new_domain`. Instead, we just form the outlines, which can be done with the :py:meth:`~pyoomph.meshes.gmsh.GmshTemplate.create_lines` method for a chain of straight lines. The arguments must be first a start point, then the name of the line, then the end point of the first line, which is also the start point of the next line, etc. For circular parts, we can use :py:meth:`~pyoomph.meshes.gmsh.GmshTemplate.circle_arc` using start and end point as arguments and the ``name`` and the ``center`` as keyword arguments.

Finally, we can mesh the surface, i.e. the ``"fish"`` domain, with :py:meth:`~pyoomph.meshes.gmsh.GmshTemplate.plane_surface`. Here, first the lines or the line names have to be passed as argument and the keyword ``name`` sets the name of the resulting domain.

The driver code is essentially the same as before:

.. literalinclude:: mesh_gmsh_fish_mesh_modes.py
   :language: python
   :start-at: class MeshTestProblem(Problem):
   :end-at: problem.output_at_increased_time()

The differences are that we do not allow for spatial adaptivity and introduce new parameters ``resolution`` and ``mesh_mode``, which will be passed to the mesh properties :py:attr:`~pyoomph.meshes.gmsh.GmshTemplate.default_resolution` and :py:attr:`~pyoomph.meshes.gmsh.GmshTemplate.mesh_mode`, respectively. Thereby, we can control the resolution of the mesh (the smaller, the finer) and also whether gmsh should try to create quadrilateral elements (``mesh_mode="quads"``) or should just create triangular elements (``mesh_mode="tris"``). However, there is no guarantee that ``mesh_mode="quads"`` generates only quadrilateral elements. In particular at the sharp corners of the fin, gmsh will likely produce a triangle instead, leading to a mixed mesh. Some representative generated meshes are depicted in :numref:`figspatialfishgmsh`.


..  figure:: fishgmsh.*
	:name: figspatialfishgmsh
	:align: center
	:alt: Fish mesh with gmsh
	:class: with-shadow
	:width: 100%

	Influence of :py:attr:`~pyoomph.meshes.gmsh.GmshTemplate.default_resolution` and :py:attr:`~pyoomph.meshes.gmsh.GmshTemplate.mesh_mode` on the meshes generated :py:class:`~pyoomph.meshes.gmsh.GmshTemplate`.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <mesh_gmsh_fish_mesh_modes.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    

.. warning::

   At the moment, spatial adaptivity does not work for triangular elements. The moment one triangular element is present in the mesh, spatial adaptivity is entirely deactivated. This will change in future to also allow for adaptivity of triangular and mixed meshes. As a workaround, you can set the :py:attr:`~pyoomph.meshes.gmsh.GmshTemplate.mesh_mode` to ``"only_quads"``. It will force gmsh to create only quadrilateral elements, but it will also lead to a less optimal mesh.
   
   
.. warning::
	Again, the orientation of the elements can matter, in particular for refineable meshes. *Gmsh* will select the element facing based on the order of the boundaries passed to :py:meth:`~pyoomph.meshes.gmsh.GmshTemplate.plane_surface`. When using refineable meshes, make sure that all elements in a mesh are oriented in the same direction by adjusting the order of the boundaries passed here. If it is wrong, pyoomph will raise an error, unless you set :py:attr:`~pyoomph.generic.problem.Problem.check_mesh_integrity` of the :py:class:`~pyoomph.generic.problem.Problem` class to ``False``. After doing so, you can easily check the mesh by *Paraview*. After outputting the mesh with :py:class:`~pyoomph.output.meshio.MeshFileOutput`, you can open it with Paraview and search for *Backface Representation* in the search box of the *Properties* box (hidden by default). Then, select *Cull Frontface* or *Cull Backface*. The entire mesh should be visible in one of these settings and entirely invisible in the other setting. If not, permute the order of the boundaries passed.

