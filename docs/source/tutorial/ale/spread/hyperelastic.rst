.. _secALEstatdroplet2:

Hyperelastic meshes and tangentially consistent mesh motion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If we want to increase the contact angle in the previous problem case to :math:`150{}^\circ` instead of :math:`110{}^\circ` by adjusting the line

.. code::

	problem.go_to_param(contact_angle=150*degree, startstep=5*degree,final_adaptive_solve=False)
	

the solver will fail to converge. The reason is large deformation of the mesh, in particular the element at the contact line. Of course, we could use remeshing as described in :numref:`secaleremeshing`, but alternatively, we can also improve the moving mesh dynamics. Pyoomph comes with several moving mesh equations, so if we use e.g. the :py:class:`~pyoomph.equations.ALE.HyperelasticSmoothedMesh` instead of the :py:class:`~pyoomph.equations.ALE.PseudoElasticMesh`, the compilation and numerical assembly will be a bit slower, but the mesh quality is usually better under deformation. However, the main issue that prevents convergence is actually the tangential mesh node positioning at the moving boundaries. The kinematic boundary condition only prescribes the normal motion and the tangential motion is given by the bulk equations of the particular moving mesh class. However, we only add a local point force at the contact line, which leads to strong deformation of the bulk element attached to the contact line. Thereby, the element at the contact line can easily collapse to a singular element. 

To prevent this, we can make sure that also the boundary nodes close to the contact line are nicely shifted tangentially, so that the bulk element at the contact line is not collapsing. To do so, we can ensure that the nodes on the boundaries keep their relative arclength position. To that end, we add 

.. code::

	# Make sure that the interface node positions keep their relative tangential positions
	eqs+=EnforcedInterfacialLaplaceSmoothing().with_corners("substrate","axis")@"interface" # Smooth the interface
	eqs+=EnforcedInterfacialLaplaceSmoothing().with_corners("interface","axis")@"substrate" # Smooth the interface
	
to the equations. The first line ensures that the relative tangential position of all nodes on the ``"interface"`` boundary are kept, i.e. the nodes are shifted tangentially so that each node stays at its initial position when parameterized by the normalized arclength of the boundary. We must call the :py:meth:`~pyoomph.equations.ALE.EnforcedInterfacialLaplaceSmoothing.with_corners` method of the :py:class:`~pyoomph.equations.ALE.EnforcedInterfacialLaplaceSmoothing` class to tell pyoomph the considered end points of the interface. Thereby, a unique normalized arclength parameterization becomes possible. Internally, pyoomph just calculates the initial arclength parameterization and solves a Laplace equation along the surface to obtain the current arclength parameterization. At the corners, Dirichlet boundary conditions are automatically set by the :py:meth:`~pyoomph.equations.ALE.EnforcedInterfacialLaplaceSmoothing.with_corners` call, so that the arclength can be indeed recovered by solving the Laplace equation along the interface. This is of course only true if the Laplace equation is solved in a Cartesian coordinate system (despite of the axisymmetric coordinate system considered e.g. for the flow), but this is the default setting of :py:class:`~pyoomph.equations.ALE.EnforcedInterfacialLaplaceSmoothing`.

Pyoomph enforces that the difference between the initial arclength parameterization and the current parameterization found by solving the Laplace equation vanishes. The corresponding field of Lagrange multipliers enforcing this condition is acting as tangential force of the mesh along the boundary, which is arbitrary, since it does not influence the physics in this case, since e.g. the kinematic boundary condition only prescribes the motion in normal direction.

Lastly, we also make use of the class :py:meth:`~pyoomph.equations.ALE.EnforceVolumeByPressure` to do exactly what was discussed in the previous example, but as a one-liner.

The issue without shifting the tangent node positions is shown in the left part of :numref:`fighyperelastic`. Clearly, the element at the contact line becomes problematic. This aspect is solved when shifting the nodes tangentially along the entire substrate (and free surface) by keeping the relative arclength positions of the nodes fixed (middle of :numref:`fighyperelastic`). The final solution is shown on the right of :numref:`fighyperelastic`.

..  figure:: hyperelastic.*
	:name: fighyperelastic
	:align: center
	:alt: Failing of convergence without tangential shifting along the substrate
	:class: with-shadow
	:width: 100%

	(left) Without tangentially shifting the nodes along the substrate (and along the free surface), the element at the contact line collapses to a singular element when going to higher contact angles. (middle) Switching to hyperelastic mesh dynamics and in particular ensuring relative tangential arclength positioning allows for higher contact angles. (right) Final solution with the tricks discussed here.
		

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <droplet_spread_hyperelastic_tangential_shift.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   


