.. _secspatialstokesdim:

Using physical dimensions and imposing a traction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If one wants to replicate an actual experimental result, one usually has to deal with dimensional quantities, i.e. with velocities given in :math:`\:\mathrm{m}/\:\mathrm{s}`, dynamic viscosities given in :math:`\:\mathrm{Pa}/\:\mathrm{s}`, pressures in :math:`\:\mathrm{Pa}` and so on. While it is often beneficial to nondimensionalize the problem by hand and identify the relevant nondimensional numbers (Reynolds, Rayleigh, Marangoni number, etc.), sometimes the problem is too complex to consider all these details. In particular for multi-component flow dynamics, each property (mass density, viscosity, diffusivity, surface tension) will change with the local composition - and usually in a nonlinear manner. Then, all these characteristic numbers are only limited in their meaning since they will vary a lot with the local composition.

For these cases, pyoomph allows for automatic nondimensionalization. In numerics, eventually everything will be nondimensional since we are dealing with floating point numbers internally. pyoomph allows us to treat the nondimensionalization in the following way: Let us again start with the weak form of the Stokes equation (cf. :math:numref:`eqspatialstokesweak`), but now with every quantity being dimensional, i.e.

.. math:: \left(-p\mathbf{1}+\mu\left(\nabla\vec{u}+(\nabla\vec{u})^t\right),\nabla\vec{v}\right)+\left(\nabla\cdot \vec{u},q\right)-\left\langle -p+\mu\left(\nabla\vec{u}+(\nabla\vec{u})^t\right) ,\vec{v}\right\rangle=0

We introduce nondimensional variables :math:`\tilde{\vec{u}}`, :math:`\tilde{p}` and the nondimensional spatial coordinate :math:`\tilde{\vec{x}}`, which are connected by typical scale factors :math:`U`, :math:`P` and :math:`X`:

.. math:: \vec{u}=U\tilde{\vec{u}}\,,\quad p=P\tilde{p}\,,\quad\vec{x}=X\tilde{\vec{x}}

These leading to the very same equation, with nondimensional variables, but still a dimensional result

.. math:: 

	\begin{split}
	\left(-P\tilde{p}\mathbf{1}+\frac{\mu U}{X}\left(\tilde\nabla\tilde{\vec{u}}+(\tilde\nabla\tilde{\vec{u}})^t\right),\frac{1}{X}\tilde\nabla\vec{v}\right)+\left(\frac{U}{X}\tilde\nabla\cdot \tilde{\vec{u}},q\right)- \\
	\left\langle -P\tilde{p}+\frac{\mu U}{X}\left(\tilde{\nabla}\tilde{\vec{u}}+(\tilde{\nabla}\tilde{\vec{u}})^t\right) ,\vec{v}\right\rangle=&0
	\end{split}

However, in numerics, eventually no dimensions shall be present anymore, i.e. the entire lhs shall just be a numeric scalar. To do so, we also introduce scales for the test functions

.. math:: \vec{v}=V\tilde{\vec{v}}\,,\quad q=Q\tilde{q}

which gives after collecting the scales in the first arguments of the weak contributions

.. math:: :label: eqspatialnondimthestokes
	
	\begin{split}
	\left(-\frac{PV}{X}\tilde{p}\mathbf{1}+\frac{\mu UV}{X^2}\left(\tilde\nabla\tilde{\vec{u}}+(\tilde\nabla\tilde{\vec{u}})^t\right),\tilde\nabla\tilde{\vec{v}}\right)+\left(\frac{UQ}{X}\tilde\nabla\cdot \tilde{\vec{u}},\tilde{q}\right)- \\
	\left\langle -\frac{PV}{X}\tilde{p}+\frac{\mu UV}{X^2}\left(\tilde{\nabla}\tilde{\vec{u}}+(\tilde{\nabla}\tilde{\vec{u}})^t\right) ,\tilde{\vec{v}}\right\rangle&=0 
	\end{split}
	
Note that in the Neumann boundary term :math:`\langle .\,,.\rangle` an additional factor :math:`1/X` arises. This one stems from the fact that the boundary integral covers one spatial dimension less than the bulk integrals :math:`(.\,,.)`. In pyoomph, these integrals are in fact nondimensional by default, i.e. :math:`\int \dots \mathrm{d}^n \tilde{x}` instead of :math:`\int \dots \mathrm{d}^n x`. This comes with the benefit that the particular nondimensionalization is independent of the number of dimensions :math:`n`.

A good choice for the pressure test scale :math:`Q` in :math:numref:`eqspatialnondimthestokes` is obviously :math:`Q=X/U`, since it will then lead to a factor of unity in the incompressibilty constraint. For :math:`V` we can choose differently, either :math:`V=X/P` or :math:`V=X^2/(U \mu)`, leading to either unity as factor for the pressure term or for the shear term in the momentum equation. The better choice depends usually on the boundary conditions desired to be imposed. If the velocity is imposed, we know a typical scale for the velocity, i.e. :math:`U`. Then, we can easily choose :math:`V=X^2/(U \mu)` and set the (typically unknown) pressure scale :math:`P=\mu U/X`. Thereby, both factors will become unity. Similarly, if one imposes pressures (or better tractions), it might be beneficial to set :math:`V=X/P` with the typical imposed traction magnitude :math:`P` and setting the a priori unknown resulting velocity scale to :math:`U=PX/\mu`. Eventually, both can be equivalent, provided both scales :math:`U` and :math:`P` are selected accordingly.

Let us now see how to implement this in pyoomph. We have to modify our Stokes equations a bit to consider the process of nondimensionalization:

.. code:: python

   from pyoomph import *
   from pyoomph.expressions import *
   from pyoomph.expressions.units import * # Import units as e.g. meter, second, etc.


   class StokesEquations(Equations):
   	# Passing the viscosity 
   	def __init__(self,mu):
   		super(StokesEquations, self).__init__()
   		self.mu=mu # Store viscosity
   		
   	def define_fields(self):
   		
   		X=scale_factor("spatial") # spatial scale (will be supplied by the Problem class)
   		U=scale_factor("velocity")
   		P=scale_factor("pressure")
   		mu=self.mu
   		
   		#Taylor-Hood pair, with testscale we can set the definition of the test scales V and Q
   		self.define_vector_field("velocity","C2",testscale=X**2/(mu*U)) 
   		self.define_scalar_field("pressure","C1",testscale=X/U)
   		
   	def define_residuals(self):
   		# Fields and test functions are dimensional here!
   		u,v=var_and_test("velocity") 
   		p,q=var_and_test("pressure")
   		stress=-p*identity_matrix()+2*self.mu*sym(grad(u))
   		self.add_residual(weak(stress,grad(v)) + weak(div(u),q)) 

Note that we have directly used the Taylor-Hood combination (``"C2"``,\ ``"C1"``). The only other difference is in the :py:meth:`~pyoomph.generic.codegen.BaseEquations.define_fields` method. Here, we pass ``testscale`` arguments, which depend on the :py:func:`~pyoomph.expressions.generic.scale_factor` of the space :math:`X`, the velocity :math:`U` and the pressure :math:`P`. These are not known at this moment and will be supplied by the :py:class:`~pyoomph.generic.problem.Problem` class. If they are not supplied by the problem, they will default to unity. Thereby, one still can use this implementation of the Stokes equation for nondimensional calculations, provided that the passed viscosity ``mu`` is nondimensional as well.

The problem class will now use dimensional units:

.. code:: python

   class DimStokesProblem(Problem):
   	def __init__(self):
   		super(DimStokesProblem, self).__init__()
   		# we are now using units for the viscosity
   		self.mu=1*milli*pascal*second
   		self.boxsize=1*milli*meter # the size of the box 
   		self.imposed_traction=1*pascal # and the imposed traction on the left

   		
   	def define_problem(self):
   		# setting the spatial scale X by the boxsize and the pressure scale P by the imposed traction
   		self.set_scaling(spatial=self.boxsize,pressure=self.imposed_traction)
   		# the velocity scale is now calculated based on these scales. scale_factor will expand to P and X, respectively
   		self.set_scaling(velocity=scale_factor("pressure")*scale_factor("spatial")/self.mu)
   		# alternatively, you can just set directly
   		# self.set_scaling(velocity=self.imposed_traction*self.boxsize/self.mu)
   		
   		self.add_mesh(RectangularQuadMesh(size=self.boxsize)) # we have to tell the mesh that it has a dimensional size now
   		eqs=StokesEquations(self.mu) # passing the dimensional viscosity to the Stokes equations
   		eqs+=MeshFileOutput() 
   		
   		# A traction is just the Neumann term
   		eqs+=NeumannBC(velocity_x=-self.imposed_traction)@"left"
   		# zero y velocity at left and right
   		eqs+=DirichletBC(velocity_y=0)@"left"
   		eqs+=DirichletBC(velocity_y=0)@"right"
   		# No slip conditions at top and bottom
   		eqs+=DirichletBC(velocity_x=0,velocity_y=0)@"bottom"
   		eqs+=DirichletBC(velocity_x=0,velocity_y=0)@"top"

   		
   		# Adding this to the default domain name "domain" of the RectangularQuadMesh above
   		self.add_equations(eqs@"domain")
   	
   		
   if __name__ == "__main__":		
   	# Create a Stokes problem with viscosity 1, quadratic velocity basis functions and linear pressure basis functions
   	with DimStokesProblem() as problem: 
   		problem.solve() # solve and output
   		problem.output()

We use :py:meth:`~pyoomph.generic.problem.Problem.set_scaling` to set the corresponding scales for :math:`X` (``spatial``), :math:`U` and :math:`P` (both identified by the name ``"velocity"`` and ``"pressure"`` we named the fields in the Stokes equation class). Furthermore, the :py:class:`~pyoomph.meshes.simplemeshes.RectangularQuadMesh` now gets a dimensional ``size`` passed, which will be canceled out internally by the spatial scale. The imposed traction is exactly the Neumann term in the Stokes equation for the momentum equation.

..  figure:: stokes_dim.*
	:name: figspatialstokesdim
	:align: center
	:alt: Stokes flow with dimensions and traction boundary condition
	:class: with-shadow
	:width: 70%

	Velocity and pressure field of the dimensional Stokes flow example.


Internally, pyoomph will now create the Stokes equations. All quantities bound by :py:func:`~pyoomph.expressions.generic.var` or :py:func:`~pyoomph.expressions.var_and_test` will be treated as dimensional quantities and successively expanded into the scale and the nondimensional quantity, e.g. :math:`p=P\tilde{p}`. In pyoomph, this means that ``var("pressure")`` will be replaced by ``scale_factor("pressure")*nondim("pressure")``. The same applies for the test functions and also the spatial differential operators :py:func:`~pyoomph.expressions.generic.grad` and :py:func:`~pyoomph.expressions.div` will be nondimensionalized by yielding :math:`1/X` (``1/scale_factor("spatial")``). When assembling the weak form, all units will cancel out and just numerical factors will survive, provided that all units and scales are selected correctly. If any unit survives this process, an error will be thrown. Thereby, one can easily identify whether the used units are in agreement. Since this happens before the C code generation, there is no additional overhead in the assembly of the system and hence in the calculation time when dimensional quantities are used.

The :py:class:`~pyoomph.output.meshio.MeshFileOutput` will write the result in dimensional units again, i.e. the velocity in :math:`\:\mathrm{m}/\:\mathrm{s}`, pressure in :math:`\:\mathrm{Pa}` and the spatial dimensions of the mesh in :math:`\:\mathrm{m}`.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <stokes_dimensional.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
