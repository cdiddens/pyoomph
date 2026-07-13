Mesh with metric dimensions and curved boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes, we want to use physical dimensions, i.e. specify the size of the mesh in meters instead of ``float`` numbers. Furthermore, we also have frequently curved boundaries, that should always resemble the very same smooth boundary curve also upon refinement. Both aspects will be handled in the following example mesh.

We will implement a fish mesh, which was inspired by the fish mesh example of oomph-lib. The mesh definition is analogous to the L-shaped mesh from :numref:`secspatialmesh1`:

.. code:: python

   from pyoomph import *
   from pyoomph.equations.poisson import *  # use the pre-defined Poisson equation

   from pyoomph.expressions.units import *

   class FishMesh(MeshTemplate):
       def __init__(self,  size=1, mouth_angle=45*degree, fin_angle=50*degree,mouth_depth_factor=0.5,fin_length_factor=0.45,fin_height_factor=0.8,domain_name="fish"):
           super(FishMesh, self).__init__()
           self.size = size # overall size of the fish (potentially dimesional)
           self.mouth_angle=mouth_angle # angle of the mouth-opening (with respect to the body center)
           self.mouth_depth_factor=mouth_depth_factor # depth of the mouth
           self.fin_angle=fin_angle # angle of the fin-body-connection (with respect to the body center)
           self.fin_length_factor=fin_length_factor
           self.fin_height_factor = fin_height_factor
           self.domain_name = domain_name # name of the fish domain

       def define_geometry(self):
           domain = self.new_domain(self.domain_name)

           S=self.nondim_size(self.size) # Important: Nondimensionalize the potentially dimensional size

           # Corner nodes of the fish
           n_mouth_center=self.add_node_unique(-(1-self.mouth_depth_factor)*S,0)
           n_upper_jaw = self.add_node_unique(-cos(self.mouth_angle / 2) * S, sin(self.mouth_angle / 2)*S)
           n_lower_jaw=self.add_node_unique(-cos(self.mouth_angle/2)*S,-sin(self.mouth_angle/2)*S)
           n_upper_body_fin=self.add_node_unique(cos(self.fin_angle/2)*S,sin(self.fin_angle/2)*S)
           n_lower_body_fin = self.add_node_unique(cos(self.fin_angle / 2) * S, -sin(self.fin_angle / 2) * S)
           n_center_body_fin = self.add_node_unique(cos(self.fin_angle / 2) * S, 0)
           n_upper_fin_corner=self.add_node_unique((cos(self.fin_angle / 2)+self.fin_length_factor) * S, self.fin_height_factor * S)
           n_lower_fin_corner = self.add_node_unique((cos(self.fin_angle / 2) + self.fin_length_factor) * S,-self.fin_height_factor * S)
           n_center_fin_end = self.add_node_unique((cos(self.fin_angle / 2) + self.fin_length_factor) * S, 0)

           # Elements
           domain.add_quad_2d_C1(n_lower_jaw,n_lower_body_fin,n_mouth_center,n_center_body_fin) # lower body part
           domain.add_quad_2d_C1(n_mouth_center, n_center_body_fin,n_upper_jaw, n_upper_body_fin ) # upper body part
           domain.add_quad_2d_C1(n_lower_body_fin,n_lower_fin_corner,n_center_body_fin,n_center_fin_end) # lower fin part
           domain.add_quad_2d_C1(n_center_body_fin,n_center_fin_end,n_upper_body_fin,n_upper_fin_corner) # upper fin part

           # Curved entities
           upper_body_curve=self.create_curved_entity("circle_arc",n_upper_jaw,n_upper_body_fin,center=[0,0])
           lower_body_curve = self.create_curved_entity("circle_arc", n_lower_jaw, n_lower_body_fin, center=[0, 0])
           self.add_facet_to_boundary("curved",[n_upper_jaw,n_upper_body_fin],curved_entity= upper_body_curve) # top body curve
           self.add_facet_to_boundary("curved",[n_lower_body_fin, n_lower_jaw], curved_entity=lower_body_curve) # bottom body curve

           # Add nodes to boundaries
           self.add_facet_to_boundary("mouth", [ n_lower_jaw,n_mouth_center])  # nodes of the mouth
           self.add_facet_to_boundary("mouth", [ n_upper_jaw,n_mouth_center])  # nodes of the mouth
           self.add_facet_to_boundary("fin",[n_upper_body_fin,n_upper_fin_corner])
           self.add_facet_to_boundary("fin",[n_upper_fin_corner,n_center_fin_end])
           self.add_facet_to_boundary("fin",[n_lower_body_fin,n_lower_fin_corner])
           self.add_facet_to_boundary("fin",[n_lower_fin_corner,n_center_fin_end])

The first new aspect is the call of :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.nondim_size`, which will calculate a corresponding non-dimensional size of an optionally dimensional argument. The dimensional argument will just be divided by ``scale_factor("spatial")``, i.e. the ``spatial`` scale set by :py:meth:`~pyoomph.generic.problem.Problem.set_scaling` in the :py:class:`~pyoomph.generic.problem.Problem` class. Every potentially metric argument passed to the mesh should be handled that way. Thereby, the mesh will be generated in the correct non-dimensional coordinates.

..  figure:: fishmesh.*
	:name: figspatialfishmesh
	:align: center
	:alt: Fish mesh
	:class: with-shadow
	:width: 100%

	(left) Fish mesh as initially defined. (middle) mesh after converting the elements to ``"C2"`` space: The additional nodes will be mapped on the circular boundaries. (right) Final adaptive solution of the Poisson equation on the fish mesh.


The definition of the corner node looks more complicated than it is. They are just the corners of the fish mesh, but the calculation of the coordinates from the parameters is a bit longish. The basic fish mesh without any adaption can be seen in :numref:`figspatialfishmesh`. Also the elements are the same as before, but then we have to tell the ``FishMesh``, that we have facets that are located on curved boundaries. To that end, we construct the curved boundaries by the calls of :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.create_curved_entity`. The first argument ``"circle_arc"`` tells that we want to have a curved boundary in shape of a circle segment. Then we specify the start and end node and the ``center``, which can either be a node or, as here, a ``list`` of coordinates. We then still have to inform the ``FishMesh`` which facets shall be mapped onto this curve, since in principle there could be multiple facets sharing the same curved entity. This is done within the :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.add_facet_to_boundary` call, by passing it via the ``curved_entity`` kwarg.

As a driver code, we use the following with a dimensional ``fish_size``:

.. code:: python

   class MeshTestProblem(Problem):
       def __init__(self):
           super(MeshTestProblem, self).__init__()
           self.fish_size=1*meter # quite large fish, isn't it...?
           self.max_refinement_level = 5 # maximum level of refinements
           self.space="C2"

       def define_problem(self):
           self.add_mesh(FishMesh(size=self.fish_size))
           self.set_scaling(spatial=self.fish_size) # Nondimensionalize space by the fish size

           eqs = MeshFileOutput()
           # We must set a meter^2 coefficient to be consistent with the units
           eqs += PoissonEquation(name="u", source=1, space=self.space,coefficient=1*meter**2)

           # Boundaries all u=0
           eqs += DirichletBC(u=0)@"fin"
           eqs += DirichletBC(u=0) @ "mouth"
           eqs += DirichletBC(u=0) @ "curved"

           # refine the curved boundary to the highest order (i.e. max_refinement_level) during adaptive solves
           eqs += RefineToLevel("max")@"curved"
           eqs += SpatialErrorEstimator(u=1) # and adapt on all other elements based on the error

           self.add_equations(eqs @ "fish")


   if __name__ == "__main__":
       with MeshTestProblem() as problem:
           problem.solve(spatial_adapt=problem.max_refinement_level)
           problem.output_at_increased_time()

Since the ``fish_size`` is dimensional, we have to use :py:meth:`~pyoomph.generic.problem.Problem.set_scaling` to set a good spatial scale for non-dimensionalization of the coordinates. This also implies, that the coefficient of the Poisson equation has to be dimensional, since the :py:class:`~pyoomph.equations.poisson.PoissonEquation` involves a :math:`\nabla^2`, which has to be compensated for by a ``coefficient`` with the unit :math:`\:\mathrm{m}^2`. The ``coefficient`` :math:`c` enters the :py:class:`~pyoomph.equations.poisson.PoissonEquation` as :math:`-\nabla\cdot(c\nabla u)=g`.

The rest is trivial with the exception that we enforce the ``"curved"`` boundaries to be refined to maximum level. Thereby, the curvature is well resolved. The results are shown in :numref:`figspatialfishmesh`.

We started with a rather simple mesh with just four elements and the final mesh is an accurate representation of the domain including all well resolved curved boundaries and refined singularities at sharp corners.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <mesh_fish_dimensional_curved.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    

