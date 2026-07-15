.. _secaleremeshing:

Mesh reconstruction upon large deformations
-------------------------------------------

It happens frequently that a moving mesh deforms strongly so that the elements are not well suited for solving equations on it. In that case, it might be necessary to regenerate the mesh by fitting the interfaces with splines, recreate a new mesh and interpolate all defined fields, including their history values for time stepping, from the previous mesh to the new mesh. pyoomph can do this automatically with the :py:class:`~pyoomph.equations.generic.RemeshWhen` class, which must be added to bulk domains. At the moment, this only works for two-dimensional meshes. Furthermore, it is necessary to set the property :py:attr:`~pyoomph.meshes.mesh.MeshTemplate.remesher` of the :py:class:`~pyoomph.meshes.mesh.MeshTemplate` to a :py:class:`~pyoomph.meshes.remesher.Remesher2d` instance from the module :py:mod:`pyoomph.meshes.remesher`. Thus, for an example problem, we first have to make sure to import the module and set the :py:attr:`~pyoomph.meshes.mesh.MeshTemplate.remesher` attribute to a :py:class:`~pyoomph.meshes.remesher.Remesher2d`, which requires the mesh template as first argument for the constructor:

.. literalinclude:: remeshing.py
   :language: python
   :start-at: from laplace_smoothed_mesh import *
   :end-at: mesh.remesher=Remesher2d(mesh)

The :py:class:`~pyoomph.equations.generic.RemeshingOptions` control when a mesh reconstruction shall be invoked. The parameters ``max_expansion`` and ``min_expansion`` give a threshold for which remeshing is invoked whenever an element has grown or shrunken above ``max_expansion`` or below ``min_expansion`` with respect to its initial size. This is determined based on the Cartesian area of each element. Furthermore, a ``min_quality_decrease`` can be set to the :py:class:`~pyoomph.equations.generic.RemeshingOptions`, since the area of an element can still remain quite close to its initial area, but the element becomes strongly anisotropic, e.g. by expanding in one direction while collapsing in the perpendicular direction.

..  figure:: remeshing.*
	:name: figaleremesh
	:align: center
	:alt: Laplace-smoothed mesh
	:class: with-shadow
	:width: 100%

	Without remeshing (top), the mesh can strongly deform. With remeshing (bottom), one can prevent this (at the cost of some computational time and small interpolation errors). One can furthermore control the local mesh size, e.g. to ensure a fine mesh at the sharp corner.

After that, the equations are assembled as usual. Here, we fix all boundaries except the ``"right"`` one, which will move with time:

.. literalinclude:: remeshing.py
   :language: python
   :start-at: # Add the mesh and use the Lagrange smoothed mesh
   :end-at: eqs+=DirichletBC(mesh_x=1+0.5*xi[1]*var("time"))@"right" # move the right interface with time

To automatically invoke remeshing, we just must add a :py:class:`~pyoomph.equations.generic.RemeshWhen` object to the domain, which takes :py:class:`~pyoomph.equations.generic.RemeshingOptions` as argument. When there are no further instructions added, pyoomph tries to estimate the local mesh resolution based on the previous resolution. This does not always work well and sometimes it is beneficial to instruct pyoomph to use specific mesh sizes instead. To that end, one can add :py:class:`~pyoomph.equations.generic.RemeshMeshSize` objects to interfaces and corners, by what the local mesh size (nondimensional, in terms of the spatial scale) can be set:

.. literalinclude:: remeshing.py
   :language: python
   :start-at: # Remeshing based on the options
   :end-at: self.add_equations(eqs@"domain")

To run the simulation, there is nothing specifically to be done:

.. literalinclude:: remeshing.py
   :language: python
   :start-at: if __name__=="__main__":
   :end-at: problem.run(10,outstep=True,startstep=0.5,maxstep=0.5)

As usual, the :py:class:`~pyoomph.equations.generic.RemeshingOptions` can be modified before the :py:meth:`~pyoomph.generic.problem.Problem.run` statement. In particular, one can deactivate it by setting ``active=False``. A comparison without and with remeshing is shown in :numref:`figaleremesh`.

One additional option is to add an instance of the class :py:class:`~pyoomph.equations.ALE.SetLagrangianToEulerianAfterSolve` to the equations of the moving mesh domain. This will set the Lagrangian coordinates to the Eulerian coordinates after each successful time step. Thereby, the mesh displacement in :math:numref:`eqalelaplsmooth` will be zero at the beginning of the next time step.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <remeshing.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   	
		    
