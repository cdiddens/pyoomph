.. _secspatialstokes_law:

Stokes' law - Obtaining forces by traction integrals and using global Lagrange multipliers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Stokes' law* describes the flow field around a spherical rigid object. Here, we consider a solid sphere sinking down due to buoyancy. Obviously, treating this in the lab frame would require to solve for the motion of the object directly, which can be done with a moving mesh method as described later on in :numref:`secALE`. However, we can also transform into the coordinate system co-moving with the object. In this case, a fixed mesh can be used.

Stokes' law states that the terminal velocity will be given by

.. math:: :label: eqspatialstokeslawuterm

   U=\frac{2}{9}\frac{\rho_\text{o}-\rho_\text{f}}{\mu}gR^2\,,

where :math:`\rho_\text{o}` and :math:`\rho_\text{f}` are the mass densities of the spherical object and the fluid, respectively, :math:`g` is the gravitational acceleration, :math:`\mu` is the fluid's dynamic viscosity and :math:`R` is the radius of the object. This equation can be derived by balancing the net force due to gravity

.. math:: F_g=\Delta \rho g V=\left(\rho_\text{o}-\rho_\text{f}\right) g \frac{4}{3}\pi R^3

acting on the object with the drag force :math:`F_\text{d}` into the direction of :math:`\vec{e}_z`, i.e. in direction of the gravity. The drag force can be obtained by the integration over the :math:`z`-projected traction of the object

.. math:: :label: eqspatialstokeslawtraction

   F_\text{d}=\int_\text{obj}  \vec{n}\cdot\left[-p \mathbf{1}+\mu\left(\nabla \vec{u}+(\nabla \vec{u})^\text{t} \right) \right]\cdot\vec{e}_z \:\mathrm{d}S

When solving this, the analytical radial and axial velocity (in frame of the object) reads

.. math:: :label: eqspatialstokeslawfarfield

   \begin{aligned}
   u_r&=\frac{3R^3}{4} \: \frac{rzU}{d^5}-\frac{3R}{4}\:\frac{rzU}{d^3} \nonumber \\
   u_z&=\frac{R^3}{4}\left(\frac{3Uz^2}{d^5}-\frac{U}{d^3}\right)+U-\frac{3R}{4}\left(\frac{U}{d}+\frac{Uz^2}{d^3}\right)\\
   \text{with }d&=\sqrt{r^2+z^2} \nonumber
   \end{aligned}

In our problem, we will not use the analytical velocity :math:numref:`eqspatialstokeslawuterm`, but indeed solve for it. In fact, we use the terminal velocity :math:`U` as global Lagrange multiplier (with test function :math:`V`) to enforce the force balance. Hence, we minimize with respect to the constraint

.. math:: U\cdot\left(F_\text{d}-F_g\right)=0

to determine :math:`U`. This value of :math:`U` is then used as far field condition by virtue of :math:numref:`eqspatialstokeslawfarfield`. :math:`U` is hence determined by the weak form

.. math:: :label: eqspatialstokeslawcontraint

   V\cdot\left(F_\text{d}-F_g\right)=0   

and the feedback of :math:`U` via the traction is given by the far field, which depends on :math:`U` and changes the flow to modify the value of :math:`F_\text{d}` until this constraint is met.

As a first step, we must build a mesh with a spherical object in the center. The far field boundary may be more or less arbitrary, but we chose a larger spherical shell. According to the mesh creation tutorial in :numref:`secspatialmeshgen`, this can be done e.g. by

.. literalinclude:: stokes_flow_around_object.py
   :language: python
   :start-at: from stokes_dimensional import * # Import dimensional Stokes from before
   :end-at: self.plane_surface("axisymm_lower","axisymm_upper","far_field","liquid_sphere",name="liquid") # liquid domain

We split the axis of symmetry into two parts, namely the lower and upper one. Thereby, we can later on pin a single degree of the pressure at e.g. ``"liquid_object/axisymm_lower"`` to remove the pressure nullspace.

Next, we require a possibility to calculate the drag force :math:`F_\text{d}` and add this contribution to the test space of :math:`U`, i.e. add it to the residuals with test function :math:`V`. To that end, we will later pass :math:`V` to the constructor of our new class

.. literalinclude:: stokes_flow_around_object.py
   :language: python
   :start-at: class DragContribution(InterfaceEquations):
   :end-at: self.add_residual(weak(dot(traction,self.direction),ltest,dimensional_dx=True)) # Integrate dimensionally over the traction

One important trick is here that we pass ``domain=self.get_parent_domain()`` when we bind the field ``"velocity"`` to ``"u"``. Thereby, we do not get the interfacial velocity, but the full velocity of the bulk. While the values of the bulk and interfacial velocity coincide on the interface, spatial derivatives do not! If we bound ``u=var("velocity")`` without the ``domain`` argument, :math:`\nabla\vec{u}` would take the surface gradient :math:`\nabla_S \vec{u}`, not the bulk gradient :math:`\nabla \vec{u}`. Alternatively, we could have used ``u=var("velocity",domain="..")`` as shortcut to bind the bulk velocity.

Then we add the integral :math:numref:`eqspatialstokeslawtraction` to the test space of :math:`U`, i.e on ``testfunction(U)``, which is :math:`V`. However, since :py:func:`~pyoomph.expressions.generic.weak` by default calculates integrals to the non-dimensional differential, i.e. to :math:`\mathrm{d}\tilde{S}` instead of :math:`\mathrm{d}S`, we would not get the unit of a force. Therefore, we have to tell :py:func:`~pyoomph.expressions.generic.weak` by passing ``dimensional_dx=True`` that we want to integrate dimensionally.

The :py:class:`~pyoomph.generic.problem.Problem` class uses physical dimensions and we set the default values in the constructor. Furthermore, we add a method that allows us to calculate the analytical terminal velocity according to :math:numref:`eqspatialstokeslawuterm`:

.. literalinclude:: stokes_flow_around_object.py
   :language: python
   :start-at: class StokesLawProblem(Problem):
   :end-at: return 2 / 9 * (self.sphere_density - self.fluid_density) / self.fluid_viscosity * self.gravity * self.sphere_radius ** 2

The problem definition will now use our mesh, set an axisymmetric coordinate system and introduce scalings, namely the object radius as spatial scale and the theoretical velocity as velocity scale. The pressure scale is set by the viscous pressure scale and we furthermore introduce a scale for any ``"force"``, which is initialized by the buoyancy force. This one will be used in a minute.

.. literalinclude:: stokes_flow_around_object.py
   :language: python
   :start-at: def define_problem(self):
   :end-at: self.add_mesh(StokesLawMesh())

The first part of the equations is trivial, just ``StokesEquations`` with output and a few boundary conditions:

.. literalinclude:: stokes_flow_around_object.py
   :language: python
   :start-at: eqs=StokesEquations(self.fluid_viscosity) # Stokes equation and output
   :end-at: eqs+=DirichletBC(velocity_x=0,velocity_y=0)@"liquid_sphere" # and no-slip on the object

Then, the Lagrange multiplier, i.e. the terminal velocity :math:`U`, is introduced. We use :py:class:`~pyoomph.generic.codegen.GlobalLagrangeMultiplier` for that, which will introduce a single global degree of freedom ``UStokes``. Furthermore, the constant offset of :math:`F_g` (``F_buo``) is subtracted, i.e. accounting for this term in :math:numref:`eqspatialstokeslawcontraint`. Both the definition of ``UStokes`` and the offset term are simultaneously done by passing ``UStokes=-F_buo`` to the :py:class:`~pyoomph.generic.codegen.GlobalLagrangeMultiplier`. The Lagrange multiplier equation is then augmented by a :py:class:`~pyoomph.equations.generic.Scaling` and a :py:class:`~pyoomph.equations.generic.TestScaling`, which sets the scale of ``UStokes`` to the ``"velocity"`` scale and the scale of its test function, i.e. :math:`V`, to an inverse of the ``"force"`` scale. With the latter, :math:numref:`eqspatialstokeslawcontraint` will become nondimensional, i.e. the units of force will cancel out upon the internal replacement of the variables and test functions by their non-dimensional counterparts:

.. literalinclude:: stokes_flow_around_object.py
   :language: python
   :start-at: # Define the Lagrange multiplier U
   :end-at: self.add_equations(U_eqs @ "globals") # add it to an ODE domain named "globals"

.. note::
	The :py:class:`~pyoomph.generic.problem.Problem` class has a method :py:meth:`~pyoomph.generic.problem.Problem.add_global_dof`, which simplifies the addition of a :py:class:`~pyoomph.generic.codegen.GlobalLagrangeMultiplier` with a :py:class:`~pyoomph.equations.generic.Scaling` and a :py:class:`~pyoomph.equations.generic.TestScaling` and a potential global contribution to its residual.

Since the Lagrange multiplier is global, we cannot add it to any mesh. Instead, it has to be added to an own domain, which we call ``"globals"`` here.


..  figure:: stokes_law.*
	:name: figspatialstokeslaw
	:align: center
	:alt: Velocity around objects according to Stokes law
	:class: with-shadow
	:width: 80%

	(left) Velocity around a spherical object according to Stokes law. (right) With adjustments of the mesh, one easily can replace the shape of the object.




We then bind this variable, where again the ``domain`` argument is crucial and pass it to our developed class ``DragContribution``. The ``DragContribution`` has to be attached to the ``"liquid/liquid_sphere"`` interface, since we must integrate over this interface to obtain the drag:

.. literalinclude:: stokes_flow_around_object.py
   :language: python
   :start-at: U=var("UStokes",domain="globals") # bind U from the domain "globals"
   :end-at: eqs += DragContribution(U)@"liquid_sphere" # The constraint is now fully assembled

Finally, the value of :math:`U` must be used as far field condition. To that end, we implement the analytical solution :math:numref:`eqspatialstokeslawfarfield` into pyoomph and enforce it at the far field boundary. We cannot use a :py:class:`~pyoomph.meshes.bcs.DirichletBC` here, since the analytical solution depends on :math:`U`, which is part of the unknowns, but :py:class:`~pyoomph.meshes.bcs.DirichletBC` terms should only depend on independent variables as e.g. ``"time"``:

.. literalinclude:: stokes_flow_around_object.py
   :language: python
   :start-at: # Far field condition
   :end-at: self.add_equations(eqs@"liquid")

The run code is again short, but we compare the analytical and numerical values, leading to an error of :math:`\sim 0.024\:\mathrm{\%}` for this mesh resolution:

.. literalinclude:: stokes_flow_around_object.py
   :language: python
   :start-at: if __name__ == "__main__":
   :end-at: print("NUMERICAL: ",U_num,"ANALYTICAL:",U_ana,"ERROR [%]:",abs(float((U_num-U_ana)/U_ana*100)))

The result is plotted in :numref:`figspatialstokeslaw`. We can easily change the mesh to calculate the terminal velocity around differently shaped objects. The far field solution won't be exact, but for a sufficiently large exterior mesh, the resulting error becomes small due to the convergence of :math:numref:`eqspatialstokeslawfarfield` to :math:`(u_r,u_z)=(0,U)`.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <stokes_flow_around_object.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
