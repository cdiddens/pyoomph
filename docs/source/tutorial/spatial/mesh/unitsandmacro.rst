Mesh with metric dimensions and curved boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes, we want to use physical dimensions, i.e. specify the size of the mesh in meters instead of ``float`` numbers. Furthermore, we also have frequently curved boundaries, that should always resemble the very same smooth boundary curve also upon refinement. Both aspects will be handled in the following example mesh.

We will implement a fish mesh, which was inspired by the fish mesh example of oomph-lib. The mesh definition is analogous to the L-shaped mesh from :numref:`secspatialmesh1`:

.. literalinclude:: mesh_fish_dimensional_curved.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: self.add_facet_to_boundary("fin",[n_lower_fin_corner,n_center_fin_end])

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

.. literalinclude:: mesh_fish_dimensional_curved.py
   :language: python
   :start-at: class MeshTestProblem(Problem):
   :end-at: problem.output_at_increased_time()

Since the ``fish_size`` is dimensional, we have to use :py:meth:`~pyoomph.generic.problem.Problem.set_scaling` to set a good spatial scale for non-dimensionalization of the coordinates. This also implies, that the coefficient of the Poisson equation has to be dimensional, since the :py:class:`~pyoomph.equations.poisson.PoissonEquation` involves a :math:`\nabla^2`, which has to be compensated for by a ``coefficient`` with the unit :math:`\:\mathrm{m}^2`. The ``coefficient`` :math:`c` enters the :py:class:`~pyoomph.equations.poisson.PoissonEquation` as :math:`-\nabla\cdot(c\nabla u)=g`.

The rest is trivial with the exception that we enforce the ``"curved"`` boundaries to be refined to maximum level. Thereby, the curvature is well resolved. The results are shown in :numref:`figspatialfishmesh`.

We started with a rather simple mesh with just four elements and the final mesh is an accurate representation of the domain including all well resolved curved boundaries and refined singularities at sharp corners.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <mesh_fish_dimensional_curved.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    

