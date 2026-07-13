.. _secspatialgmsh:

Generating meshes from points and lines via Gmsh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As we have seen on the basis of the fish mesh, one can create rather complicated domains already by hand. However, it can be quite cumbersome to add all elements by hand and map facets on curved boundaries, in particular if the geometry is rather complex. One can intrinsically let the meshing tool gmsh do this job for you, which will be discussed in the following. To that end, we have to inherit our mesh from the :py:class:`~pyoomph.meshes.gmsh.GmshTemplate` class.

The constructor looks - besides the chosen class name ``GmshFishMesh`` - the same:

.. code:: python

   from pyoomph import *
   from pyoomph.equations.poisson import *  # use the pre-defined Poisson equation

   from pyoomph.expressions.units import *

   class GmshFishMesh(GmshTemplate):
       def __init__(self,  size=1, mouth_angle=45*degree, fin_angle=50*degree,mouth_depth_factor=0.5,fin_length_factor=0.45,fin_height_factor=0.8,domain_name="fish"):
           super(GmshFishMesh, self).__init__()
           self.size = size # all as before
           self.mouth_angle=mouth_angle 
           self.mouth_depth_factor=mouth_depth_factor 
           self.fin_angle=fin_angle 
           self.fin_length_factor=fin_length_factor
           self.fin_height_factor = fin_height_factor
           self.domain_name = domain_name

In the :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.define_geometry` method, however, there are multiple changes:

.. code:: python

       def define_geometry(self):    	
           S=self.size # gmsh does not require to nondimensionalize the size, it will be done automatically
           # Corner nodes of the fish: instead of "add_node_unique", we use "point". 
           # We do not need p_center_body_fin and p_center_fin_end here
           p_mouth_center=self.point(-(1-self.mouth_depth_factor)*S,0)
           p_upper_jaw = self.point(-cos(self.mouth_angle / 2) * S, sin(self.mouth_angle / 2)*S)
           p_lower_jaw=self.point(-cos(self.mouth_angle/2)*S,-sin(self.mouth_angle/2)*S)
           p_upper_body_fin=self.point(cos(self.fin_angle/2)*S,sin(self.fin_angle/2)*S)
           p_lower_body_fin = self.point(cos(self.fin_angle / 2) * S, -sin(self.fin_angle / 2) * S)        
           p_upper_fin_corner=self.point((cos(self.fin_angle / 2)+self.fin_length_factor) * S, self.fin_height_factor * S)
           p_lower_fin_corner = self.point((cos(self.fin_angle / 2) + self.fin_length_factor) * S,-self.fin_height_factor * S)

           # Instead of starting with the elements, we start with the outlines
           # Create lines from lower jaw, to mouth center and to upper jaw, all named "mouth"
           self.create_lines(p_lower_jaw,"mouth",p_mouth_center,"mouth",p_upper_jaw)
           # Create the fin, also here, just a chain of straight lines, all named "fin"
           self.create_lines(p_lower_body_fin,"fin",p_lower_fin_corner,"fin",p_upper_fin_corner,"fin",p_upper_body_fin)
           # Create the body curves
           self.circle_arc(p_lower_jaw,p_lower_body_fin,center=[0,0],name="curved")
           self.circle_arc(p_upper_jaw,p_upper_body_fin,center=[0,0],name="curved")
           
           # Now, generate the surface, i.e. the domain
           self.plane_surface("mouth","fin","curved",name=self.domain_name)

First of all, we do not need to call :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.nondim_size` to get nondimensional coordinates. In fact, the :py:class:`~pyoomph.meshes.gmsh.GmshTemplate` expects dimensional coordinates if a spatial dimension is set via ``set_scaling(spatial=...)`` in the problem class where the mesh is used. Next, instead of using :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.add_node_unique`, we add points via :py:meth:`~pyoomph.meshes.gmsh.GmshTemplate.point`. We then do not create any elements by hand at all. Also we do not create any domain via :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.new_domain`. Instead, we just form the outlines, which can be done with the :py:meth:`~pyoomph.meshes.gmsh.GmshTemplate.create_lines` method for a chain of straight lines. The arguments must be first a start point, then the name of the line, then the end point of the first line, which is also the start point of the next line, etc. For circular parts, we can use :py:meth:`~pyoomph.meshes.gmsh.GmshTemplate.circle_arc` using start and end point as arguments and the ``name`` and the ``center`` as keyword arguments.

Finally, we can mesh the surface, i.e. the ``"fish"`` domain, with :py:meth:`~pyoomph.meshes.gmsh.GmshTemplate.plane_surface`. Here, first the lines or the line names have to be passed as argument and the keyword ``name`` sets the name of the resulting domain.

The driver code is essentially the same as before:

.. code:: python

   class MeshTestProblem(Problem):
       def __init__(self):
           super(MeshTestProblem, self).__init__()
           self.fish_size=0.5*meter # quite large fish, isn't it...?
           self.resolution = 0.1 # Resolution of the mesh
           self.mesh_mode="quads" # Try to use quadrilateral elements
           self.space="C2"

       def define_problem(self):
           mesh=GmshFishMesh(size=self.fish_size)
           mesh.default_resolution=self.resolution
           mesh.mesh_mode=self.mesh_mode
           self.add_mesh(mesh)        
           self.set_scaling(spatial=self.fish_size) # Nondimensionalize space by the fish size

           eqs = MeshFileOutput()
           eqs += PoissonEquation(name="u", source=1, space=self.space,coefficient=1*meter**2)

           # Boundaries all u=0
           eqs += DirichletBC(u=0)@"fin"
           eqs += DirichletBC(u=0) @ "mouth"
           eqs += DirichletBC(u=0) @ "curved"

           self.add_equations(eqs @ "fish")


   if __name__ == "__main__":
       with MeshTestProblem() as problem:
           problem.solve()
           problem.output_at_increased_time()

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

