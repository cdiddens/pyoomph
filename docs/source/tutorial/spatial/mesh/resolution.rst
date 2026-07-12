.. _secspatialfishmesheye:

Splines, adding holes and locally controlling the mesh resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Actually, the previous mesh is not a fish - it does not have an eye! One could excuse this by arguing that it is a flatfish, viewed from the bottom, so that the migrated eye is not visible. However, then the mouth is wrong...

So, we must add an eye and we will introduce spline boundaries and how to control the local mesh size. To that end, only slight modifications are necessary. In the :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.define_geometry`, we now add the eye points and boundary lines:

.. code:: python

           eye_size=0.125*S
           eye_center_x,eye_center_y=-S*0.25,S*0.3
           eye_resolution=self.default_resolution*0.2 # Fine mesh near the eye
           p_eye_center=self.point(eye_center_x,eye_center_y,size=eye_resolution)
           p_eye_north=self.point(eye_center_x,eye_center_y+eye_size,size=eye_resolution)
           p_eye_west=self.point(eye_center_x-eye_size,eye_center_y,size=eye_resolution)
           p_eye_east=self.point(eye_center_x+eye_size,eye_center_y,size=eye_resolution)
           p_eye_south=self.point(eye_center_x,eye_center_y-0.25*eye_size,size=eye_resolution)
           self.circle_arc(p_eye_west,p_eye_north,center=p_eye_center,name="eye")
           self.circle_arc(p_eye_north,p_eye_east,center=p_eye_center,name="eye")
           self.spline([p_eye_west,p_eye_south,p_eye_east],name="eye")

With the keyword argument ``size`` in the :py:meth:`~pyoomph.meshes.gmsh.GmshTemplate.point` calls, the local mesh size (resolution) can be set. If it is not passed, it will default to :py:attr:`~pyoomph.meshes.gmsh.GmshTemplate.default_resolution` of the :py:class:`~pyoomph.meshes.gmsh.GmshTemplate`. The upper part of the eye is just circular segments. We have split it into two segments, since the total angle of the circular arc is 180\ :math:`\:\mathrm{^\circ}`. With a single py:meth:`~pyoomph.meshes.gmsh.GmshTemplate.circle_arc` from ``p_eye_west`` to ``p_eye_east``, it would be ambiguous, whether the arc should cover the north or the south direction. For that reason, any py:meth:`~pyoomph.meshes.gmsh.GmshTemplate.circle_arc` with an opening angle :math:`\geq180\:\mathrm{^\circ}` must be split into multiple segments. The bottom part of the eye is a :py:meth:`~pyoomph.meshes.gmsh.GmshTemplate.spline`, which takes a list of points it has to pass.

We have to tell gmsh that the eye should be removed from the fish. It is actually a hole in the mesh, so we have to pass the keyword ``holes`` to the :py:meth:`~pyoomph.meshes.gmsh.GmshTemplate.plane_surface`:

.. code:: python

           self.plane_surface("mouth","fin","curved",name=self.domain_name,holes=[["eye"]])

``holes`` must take a list of holes, which again are each lists of boundary lines or boundary line names.

..  figure:: fishgmsh_eye.*
	:name: figspatialfishgmsheye
	:align: center
	:alt: The fish has an eye now.
	:class: with-shadow
	:width: 50%

	Using a spline and local mesh size control, we add a hole to the fish mesh to create its eye.



In the problem class, a :py:class:`~pyoomph.meshes.bcs.NeumannBC` is added to the ``"eye"`` boundary

.. code:: python

           eqs += NeumannBC(u=1*meter) @ "eye"

so that the result looks as depicted in :numref:`figspatialfishgmsheye`. Since our :py:class:`~pyoomph.equations.poisson.PoissonEquation` has a ``coefficient`` in ``meter**2``, the :py:class:`~pyoomph.meshes.bcs.NeumannBC` boundary condition has to be in units of ``meter``. 

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <mesh_gmsh_fish_with_holes.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
