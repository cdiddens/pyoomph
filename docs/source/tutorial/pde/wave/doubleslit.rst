.. _secpdedoubleslit:

Double-slit
~~~~~~~~~~~

We will now solve the wave equation on a setting that resembles the famous *double-slit experiment*. To get a double-slit domain, we obviously have to create a custom mesh by interfacing Gmsh via the :py:class:`~pyoomph.meshes.gmsh.GmshTemplate` class:

.. code:: python

   from wave_eq import *  # Import the wave equation from the previous example


   class DoubleSlitMesh(GmshTemplate):
   	def __init__(self,problem):
   		super(DoubleSlitMesh,self).__init__()
   		self.problem=problem # store the problem to access the properties
   		self.default_resolution=self.problem.resolution
   		self.mesh_mode="tris" # use triangles here
   		
   	def define_geometry(self):		
   		p=self.problem

   		# corner points of the inlet
   		p_in_0=self.point(-p.inlet_length,0)
   		p_in_H = self.point(-p.inlet_length, p.domain_height)
   		p_wl_0=self.point(0,0)
   		p_wl_H = self.point(0, p.domain_height)

   		# slit corner points, upper slit
   		p_wl_ub = self.point(0, (p.domain_height-p.slit_width+p.slit_distance)*0.5)
   		p_wl_uu = self.point(0, (p.domain_height+p.slit_width+p.slit_distance)*0.5)
   		p_wr_ub = self.point(p.wall_thickness, (p.domain_height - p.slit_width + p.slit_distance) * 0.5)
   		p_wr_uu = self.point(p.wall_thickness, (p.domain_height + p.slit_width + p.slit_distance) * 0.5)

   		# slit corner points, bottom slit
   		p_wl_bb = self.point(0, (p.domain_height - p.slit_width - p.slit_distance) * 0.5)
   		p_wl_bu = self.point(0, (p.domain_height + p.slit_width - p.slit_distance) * 0.5)
   		p_wr_bb = self.point(p.wall_thickness, (p.domain_height - p.slit_width - p.slit_distance) * 0.5)
   		p_wr_bu = self.point(p.wall_thickness, (p.domain_height + p.slit_width - p.slit_distance) * 0.5)

   		# corner points of the domain behind the slit
   		p_wr_0 = self.point(p.wall_thickness, 0)
   		p_wr_H = self.point(p.wall_thickness, p.domain_height)
   		p_scr_0 = self.point(p.screen_distance, 0)
   		p_scr_H = self.point(p.screen_distance, p.domain_height)
   	
   		# Create a line loop of the exterior boundary
   		lines=self.create_lines(p_in_0,"inlet",p_in_H,"top",p_wl_H,"wall_left",p_wl_uu,"wall",p_wr_uu,"wall",p_wr_H,"out_top",p_scr_H,"screen",p_scr_0,"out_bottom",p_wr_0,"wall",p_wr_bb,"wall",p_wl_bb,"wall_left",p_wl_0,"bottom",p_in_0)		
   		# The center part of the wall is a hole in the mesh
   		holes=self.create_lines(p_wl_bu,"wall_left",p_wl_ub,"wall",p_wr_ub,"wall",p_wr_bu,"wall",p_wl_bu)
   		self.plane_surface(*lines,name="domain",holes=[holes]) # create the domain

A lot of parameters are directly accessed from the :py:class:`~pyoomph.generic.problem.Problem` class, which will be defined soon. Therefore, the constructor of the ``DoubleSlitMesh`` gets the :py:class:`~pyoomph.generic.problem.Problem` passed, which is then accessed in the ``define_mesh`` method to get the chosen settings as e.g. ``slit_width`` or ``slit_distance``. The corner points are created and a line loop is assembled with the :py:meth:`~pyoomph.meshes.gmsh.GmshTemplate.create_lines` method. It takes alternating arguments of points and interface names for the interfaces between. The interface ``"inlet"`` will be used to impose a planar incoming wave, ``"top"`` and ``"bottom"`` are at the top and bottom of the domain before the slit. ``"wall_left"`` is the left side of the wall containing the slits, i.e. the side facing towards the incoming wave. At this side, we have to make sure to completely remove any reflections of the incoming wave to prevent an undesired interference with the incoming wave. All other interfaces of the slits and the right side of the wall are marked as ``"wall"``. There, reflection is admitted. The ``"out_top"`` and ``"out_bottom"`` are the interfaces at the top and bottom after the two slits. Also here, reflection will be prevented. Finally, at the far right side of the mesh, the ``"screen"`` interface is set where the intensity will be measured.

To measure the intensity, a new :py:class:`~pyoomph.generic.codegen.InterfaceEquations` class is defined, which will be added to the interface ``"screen"``. On that interface, a new field :math:`I` will be added, which is calculated by the accumulation of the wave intensity over time, i.e.

.. math:: I=\int u^2 \mathrm{d}t

or, in weak formulation, :math:`(\partial_t I-u^2,J)` with test function :math:`J`.

.. code:: python

   # Measure the intensity of the waves at the screen
   class WaveEquationScreen(InterfaceEquations):
   	required_parent_type =  WaveEquation

   	def define_fields(self):
   		self.define_scalar_field("I","C2") # intensity as interface field

   	def define_residuals(self):
   		I,J=var_and_test("I")
   		self.add_residual(weak(partial_t(I)-var("u")**2,J)) # I=integral of u^2 dt

Before the :py:class:`~pyoomph.generic.problem.Problem` class will be described, let us consider how any reflection can be prevented as e.g. on the screen and the wall of the double-slit facing towards the incoming wave. In the example in :numref:`secpdewaveeqoned`, we have seen that imposing a ``DirichletBC(u=0)`` reflects the wave with a change in sign. Without imposing any boundary condition, which is equivalent to imposing a zero Neumann flux, the wave gets reflected as well, but without any sign flip. So what is the correct boundary condition if we just want the wave to be absorbed without any reflection? In that case, we have to make sure that any incoming wave just passes through the interface as if the domain would just continue after the interface. To see a good solution, we factorize the differential operator in :math:numref:`eqpdewaveeq` and consider the normal direction:

.. math:: \left(\partial_t-c\vec{n}\cdot\nabla\right)\left(\partial_t+c\vec{n}\cdot\nabla\right)u=0\,.

The equation is obviously fulfilled if :math:`\partial_t u\pm c\nabla u\cdot \vec{n}=0`, reflecting the fact that the wave equation allows for traveling solutions. As a Neumann flux, however, we can impose :math:`-c^2\nabla u\cdot \vec{n}` at interfaces. Hence, when imposing :math:`c\:\partial_t u` as Neumann flux, we will not influence the wave equation due to the presence of the boundary, however, only if the wave approaches in normal direction. More sophisticated solutions are e.g. *perfectly matched layers*, as discussed in :numref:`secspatialhelmholtz`.

This finally brings us to the specification of the :py:class:`~pyoomph.generic.problem.Problem` class:

.. code:: python

   class DoubleSlitProblem(Problem):
   	def __init__(self):
   		super(DoubleSlitProblem, self).__init__()
   		self.c = 1  # speed
   		self.omega=10 # wave frequency		
   		self.inlet_length=0.4 # length of the inlet
   		self.wall_thickness=0.1 # thickness of the wall
   		self.screen_distance=2 # distance of the screen
   		self.domain_height=4 # height of the entire domain
   		self.slit_width=0.2 # width of a slit
   		self.slit_distance=0.5 # distance of the slits
   		self.resolution=0.04 # mesh resolution, the smaller the finer

   	

   	def define_problem(self):
   		mesh=DoubleSlitMesh(self) # mesh

   		#mesh=LineMesh(size=6,N=200)
   		self.add_mesh(mesh)

   		eqs = WaveEquation(c=self.c)  # wave equation
   		eqs += MeshFileOutput()  # mesh output

   		# initial condition: incoming wave wave, exponentially damped 
   		u0=cos(self.omega*(var("coordinate_x")+self.inlet_length-self.c*var("time")))*exp(-10*(var("coordinate_x")+self.inlet_length))
   		eqs +=InitialCondition(u=u0)				
   		eqs += DirichletBC(u=u0) @ "inlet"  # incoming wave

   		# Let the waves just flow out/absorb without any reflection at some interfaces
   		u=var("u")
   		eqs += NeumannBC(u=self.c*partial_t(u)) @ "wall_left"
   		eqs += NeumannBC(u=self.c*partial_t(u)) @ "screen"
   		eqs += NeumannBC(u=self.c*partial_t(u)) @ "out_top"
   		eqs += NeumannBC(u=self.c*partial_t(u)) @ "out_bottom"

   		# Measure the intesity at the screen
   		eqs += WaveEquationScreen()@"screen"
   		eqs += TextFileOutput() @ "screen"

   		self.add_equations(eqs @ "domain")  


   if __name__ == "__main__":
   	with DoubleSlitProblem() as problem:
   		problem.c=3 # increase the wave speed
   		problem.omega=20 # and the wave frequency
   		problem.run(1, outstep=True, startstep=0.01)

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
		    
