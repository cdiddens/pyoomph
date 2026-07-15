.. _secpdedoubleslit:

Double-slit
~~~~~~~~~~~

We will now solve the wave equation on a setting that resembles the famous *double-slit experiment*. To get a double-slit domain, we obviously have to create a custom mesh by interfacing Gmsh via the :py:class:`~pyoomph.meshes.gmsh.GmshTemplate` class:

.. literalinclude:: wave_eq_doubleslit.py
   :language: python
   :start-at: from wave_eq import *  # Import the wave equation from the previous example
   :end-at: self.plane_surface(*lines,name="domain",holes=[holes]) # create the domain

A lot of parameters are directly accessed from the :py:class:`~pyoomph.generic.problem.Problem` class, which will be defined soon. Therefore, the constructor of the ``DoubleSlitMesh`` gets the :py:class:`~pyoomph.generic.problem.Problem` passed, which is then accessed in the ``define_mesh`` method to get the chosen settings as e.g. ``slit_width`` or ``slit_distance``. The corner points are created and a line loop is assembled with the :py:meth:`~pyoomph.meshes.gmsh.GmshTemplate.create_lines` method. It takes alternating arguments of points and interface names for the interfaces between. The interface ``"inlet"`` will be used to impose a planar incoming wave, ``"top"`` and ``"bottom"`` are at the top and bottom of the domain before the slit. ``"wall_left"`` is the left side of the wall containing the slits, i.e. the side facing towards the incoming wave. At this side, we have to make sure to completely remove any reflections of the incoming wave to prevent an undesired interference with the incoming wave. All other interfaces of the slits and the right side of the wall are marked as ``"wall"``. There, reflection is admitted. The ``"out_top"`` and ``"out_bottom"`` are the interfaces at the top and bottom after the two slits. Also here, reflection will be prevented. Finally, at the far right side of the mesh, the ``"screen"`` interface is set where the intensity will be measured.

To measure the intensity, a new :py:class:`~pyoomph.generic.codegen.InterfaceEquations` class is defined, which will be added to the interface ``"screen"``. On that interface, a new field :math:`I` will be added, which is calculated by the accumulation of the wave intensity over time, i.e.

.. math:: I=\int u^2 \mathrm{d}t

or, in weak formulation, :math:`(\partial_t I-u^2,J)` with test function :math:`J`.

.. literalinclude:: wave_eq_doubleslit.py
   :language: python
   :start-at: # Measure the intensity of the waves at the screen
   :end-at: self.add_residual(weak(partial_t(I)-var("u")**2,J)) # I=integral of u^2 dt

Before the :py:class:`~pyoomph.generic.problem.Problem` class will be described, let us consider how any reflection can be prevented as e.g. on the screen and the wall of the double-slit facing towards the incoming wave. In the example in :numref:`secpdewaveeqoned`, we have seen that imposing a ``DirichletBC(u=0)`` reflects the wave with a change in sign. Without imposing any boundary condition, which is equivalent to imposing a zero Neumann flux, the wave gets reflected as well, but without any sign flip. So what is the correct boundary condition if we just want the wave to be absorbed without any reflection? In that case, we have to make sure that any incoming wave just passes through the interface as if the domain would just continue after the interface. To see a good solution, we factorize the differential operator in :math:numref:`eqpdewaveeq` and consider the normal direction:

.. math:: \left(\partial_t-c\vec{n}\cdot\nabla\right)\left(\partial_t+c\vec{n}\cdot\nabla\right)u=0\,.

The equation is obviously fulfilled if :math:`\partial_t u\pm c\nabla u\cdot \vec{n}=0`, reflecting the fact that the wave equation allows for traveling solutions. As a Neumann flux, however, we can impose :math:`-c^2\nabla u\cdot \vec{n}` at interfaces. Hence, when imposing :math:`c\:\partial_t u` as Neumann flux, we will not influence the wave equation due to the presence of the boundary, however, only if the wave approaches in normal direction. More sophisticated solutions are e.g. *perfectly matched layers*, as discussed in :numref:`secspatialhelmholtz`.

This finally brings us to the specification of the :py:class:`~pyoomph.generic.problem.Problem` class:

.. literalinclude:: wave_eq_doubleslit.py
   :language: python
   :start-at: class DoubleSlitProblem(Problem):
   :end-at: problem.run(1, outstep=True, startstep=0.01)

In the constructor, we have several parameters to allow for a custom wave and slit geometry. Since the problem itself is passed to the ``DoubleSlitMesh``, the latter parameters are used to construct a mesh based on these. We make use of the absorption (no reflection) boundary conditions and add the ``WaveEquationScreen`` as well as a :py:class:`~pyoomph.output.generic.TextFileOutput` to the interface ``"screen"``. Thereby, we get the results of the intensity on the screen written to a file.

In the results (cf. :numref:`figpdewavedoubleslit`) we indeed see that the incoming wave is not reflected at the wall, i.e. there is no self-interference. The same is true for the top and bottom boundary of the domain beyond the double-slit and the screen itself. The screen intensity :math:`I` shows the expected pattern, i.e. a maximum in the center of the slits with additional smaller maxima and minima off-center.

..  figure:: waveeqdoubleslit.*
	:name: figpdewavedoubleslit
	:align: center
	:alt: Wave through a double-slit
	:class: with-shadow
	:width: 100%

	Double-slit result at three different times along with the monitored intensity at the screen.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <wave_eq_doubleslit.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
