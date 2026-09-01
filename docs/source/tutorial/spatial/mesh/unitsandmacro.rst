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
   :end-at: problem.output(increase_time_for_PVD=True)

Since the ``fish_size`` is dimensional, we have to use :py:meth:`~pyoomph.generic.problem.Problem.set_scaling` to set a good spatial scale for non-dimensionalization of the coordinates. This also implies, that the coefficient of the Poisson equation has to be dimensional, since the :py:class:`~pyoomph.equations.poisson.PoissonEquation` involves a :math:`\nabla^2`, which has to be compensated for by a ``coefficient`` with the unit :math:`\:\mathrm{m}^2`. The ``coefficient`` :math:`c` enters the :py:class:`~pyoomph.equations.poisson.PoissonEquation` as :math:`-\nabla\cdot(c\nabla u)=g`.

The rest is trivial with the exception that we enforce the ``"curved"`` boundaries to be refined to maximum level. Thereby, the curvature is well resolved. The results are shown in :numref:`figspatialfishmesh`.

We started with a rather simple mesh with just four elements and the final mesh is an accurate representation of the domain including all well resolved curved boundaries and refined singularities at sharp corners.

Curved boundaries in general
""""""""""""""""""""""""""""

The mechanism above is not restricted to quadrilaterals or to two dimensions. A facet of any element type pyoomph supports -- quadrilaterals, triangles, bricks, tetrahedra, wedges and pyramids -- can be attached to a curved entity, in 2d and in 3d, and every node introduced by refinement then lands on that entity rather than on the straight-sided interpolation between the coarse mesh's nodes. The same holds when the mesh is distributed over several processes with ``--distribute``.

Besides ``"circle_arc"``, :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.create_curved_entity` accepts ``"sphere_part"`` (one point on the sphere, plus the ``center``) and ``"cylinder_arc"`` (the arc's start and end, plus the ``center``; the axis follows from those three points). The predefined :py:class:`~pyoomph.meshes.simplemeshes.CircularMesh`, :py:class:`~pyoomph.meshes.simplemeshes.SphericalOctantMesh` and :py:class:`~pyoomph.meshes.simplemeshes.CylinderMesh` attach these to their curved boundaries automatically; each takes a ``with_curved_entities`` argument if you want the straight-sided behaviour instead.

It is worth being concrete about what this buys, because the error does not shrink when you refine. A :py:class:`~pyoomph.meshes.simplemeshes.SphericalOctantMesh` has only four elements, so it must be refined to be usable at all -- but refining a polyhedron just gives a finer polyhedron. Without a curved entity its "sphere" of radius 1 encloses a volume of 0.4013 against the exact :math:`\pi/6\approx 0.5236`, a deficit of 23% that no amount of refinement removes. With one, the boundary is exact to machine precision at every level.

For meshes generated by Gmsh, circular arcs and splines carry their curved entities automatically. Surfaces do not: :py:meth:`~pyoomph.meshes.gmsh.GmshTemplate.ruled_surface` takes an explicit ``map_to_sphere`` argument, since a ruled surface is not in general a sphere -- Gmsh's built-in kernel does not produce an exact one even when the bounding curves are great-circle arcs, so pyoomph cannot assume it. Pass ``map_to_sphere=True`` to have the sphere worked out from the bounding curves, or ``map_to_sphere=(cx,cy,cz)`` to state its centre. If the bounding curves do not determine a sphere, this raises rather than guessing.

.. warning::

   On a moving mesh, i.e. whenever the nodal positions are unknowns (:py:class:`~pyoomph.equations.ALE.LaplaceSmoothedMesh`, :py:class:`~pyoomph.equations.ALE.PseudoElasticMesh`), curved entities shape the *initial* mesh only. pyoomph discards the macro elements once the coordinates are free, so afterwards the boundary moves without being pulled back onto the curve -- which is what you want for a free surface, and something to keep in mind for a curved wall that is meant to stay put.

   If you keep them instead, with ``problem.remove_macro_elements_after_initial_adaption = False``, do not call :py:meth:`~pyoomph.generic.problem.Problem.map_nodes_on_macro_elements` once the mesh has moved: it restores the *template* geometry, undoing the motion. It also restores the original parametrisation, so it will move nodes along a boundary even when they are already exactly on it.

Writing your own curved entity
""""""""""""""""""""""""""""""

If your boundary is not a circle, sphere or cylinder, subclass :py:class:`~_pyoomph.MeshTemplateCurvedEntity` in Python. An entity is a two-way map between a *parametric coordinate* of your choosing and a Cartesian position:

.. code-block:: python

   class Parabola(MeshTemplateCurvedEntity):
       def __init__(self):
           super().__init__(1)                      # one parametric component
       def pos_to_parametric(self, t, pos, param):
           param[0] = pos[0]                        # parametrise by x
       def parametric_to_pos(self, t, param, pos):
           pos[0] = param[0]
           pos[1] = param[0]**2

The parametric coordinate is opaque to pyoomph: it need not be a length or an angle, and it may have *more* components than the boundary has dimensions. That redundancy is occasionally the point -- pyoomph's own sphere is parametrised by its outward unit normal, three numbers for a two-dimensional surface, because every two-parameter chart of a sphere degenerates at the poles.

To place a node between two facet nodes, pyoomph combines their parametric coordinates with weights and maps the result. The default combination is the weighted sum, which is right for a flat coordinate such as an angle, an arclength or the :math:`x` above. It is *not* right for a redundant one: the average of two unit normals is not a unit normal. Override ``blend(weights, params, result)`` when the default does not apply, or -- cheaper, and usually possible -- absorb the correction into ``parametric_to_pos``, which is called anyway:

.. code-block:: python

   def parametric_to_pos(self, t, param, pos):
       n = numpy.array(param[:3])
       n /= numpy.linalg.norm(n)                    # a blended normal is not a unit normal
       pos[:] = self.centre + self.radius * n

Finally, if your parametric coordinate is periodic (an angle, say), implement ``ensure_periodicity(param)`` to shift the stored values of a facet onto a common branch, so that a facet straddling the parametrisation's seam is blended the short way round.

.. only:: html

	.. container:: downloadbutton

		Full code available in the

		:download:`pyoomph example bundle <../../tutorial_example_scripts.zip>`

		``Spatial_PDEs/mesh_fish_dimensional_curved.py``
		    

