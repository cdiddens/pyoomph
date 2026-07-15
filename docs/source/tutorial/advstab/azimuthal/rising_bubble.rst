.. _secadvstabrisingbubble:

Path instability of a rising bubble
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One particularly powerful feature is the possibility to tackle azimuthal instabilities of rather arbitrary problems defined on moving meshes. Such numerical approaches have been developed only quite recently to investigate e.g. the instability of a rising bubble :cite:`Bonnefis2024,Herrada2023`. Pyoomph does all the cumbersome work of deriving the azimuthal eigenproblem automatically and fully symbolically and generates corresponding C code to fill the mass and Jacobian matrices for these eigenproblems. In the following, we will reproduce the results of :cite:`Bonnefis2024` in pyoomph.

First of all, we start by defining a rectangular mesh with a circular hole in the center. This hole later represents the bubble. While we could just use the :py:class:`~pyoomph.meshes.gmsh.GmshTemplate` class to create such a mesh, we prefer to make a structured mesh by manually placing all elements of a coarse mesh, which will be refined during the solution procedure. Read :numref:`secspatialmesh1` to learn how to create meshes that way. The mesh class itself is skipped here for brevity, but is part of the example code you can download below.

As in :cite:`Bonnefis2024`, we neglect the viscosity and the mass density inside the bubble and nondimensionalize the equations in terms of a Bond number :math:`Bo=\rho g D^2/\sigma` and a Galilei number :math:`Ga=\rho\sqrt{gD^3}/\mu` (:math:`D=2R` is the droplet diameter), where we express the Galilei number by the Bond number via a Morton number :math:`Mo=g \mu^4/(\rho \sigma^3)`. The Morton number is independent of the bubble size and only depends on the liquid properties and the gravity. In particular, :math:`Mo=6.2\times 10^{-7}` holds for DMS-T05 :cite:`Bonnefis2024`, which we consider here. Keeping the Morton number fixed, we can calculate the corresponding Galilei number from the Bond number by :math:`Ga=(Bo^3/Mo)^{1/4}`. However, the latter expression is problematic if the Bond number becomes negative. By default, pyoomph's global parameters may attain positive and negative values and therefore, the 4th root will be rather problematic. We therefore inform pyoomph that the Bond number is always a positive parameter. This information will be used in the code generation later on for a good code generation of the 4th root:

.. literalinclude:: rising_bubble.py
   :language: python
   :start-at: class RisingBubbleProblem(Problem):
   :end-at: self.max_refinement_level=4 # Do not refine more than 4 times (we want to have it fast, not perfectly accurate)

The nondimensionalized equations read 

.. math:: :label: eqadvstabrisingbubble

   \begin{aligned}
   \partial_t \vec{u}+\nabla\vec{u}\cdot\vec{u}&=-\nabla p+\frac{1}{Ga}\nabla\cdot\left[\left(\nabla\vec{u}+\nabla\vec{u}^\text{t}\right)\right]-\dot U e_y\\
   \nabla\cdot\vec{u}&=0\\
   \left(\vec{u}-\dot{X}\right)\cdot \vec{n}&=0\\
   \left(-p\mathbf{1}+\frac{1}{Ga}\left(\nabla\vec{u}+\nabla\vec{u}^\text{t}\right)\right)\cdot \vec{n}&=\left(\frac{1}{Bo}\kappa +y+P\right) \vec{n}
   \end{aligned}

Here, :math:`U` is the velocity of the bubble, which is determined by enforcing that the center of mass of the bubble does not move. So we transform into the coordinate system comoving with the bubble as in :numref:`secspatialstokes_law`. Also, we have absorbed the hydrostatic pressure in the pressure field :math:`p`, which leads to the additional axial coordinate :math:`y` in the rhs of the pressure acting on the surface. The unknown :math:`P` is the bubble pressure which is determined by enforcing a constant nondimensional volume :math:`4/3\pi R^3` of the bubble (with :math:`R=0.5`, i.e. a nondimensional diameter of :math:`D=1`). For the volume constraint, we use the divergence theorem trick described in the box in :numref:`secALEstatdroplet`. In a similar fashion, the center of mass is calculated by an integration over the surface:


.. literalinclude:: rising_bubble.py
   :language: python
   :start-at: def define_problem(self):
   :end-at: eqs+=WeakContribution(-dot(var("coordinate"),var("normal"))/3,Ptest)@"interface"

We still have to add moving mesh equations and some missing boundary conditions. The :py:class:`~pyoomph.meshes.bcs.AxisymmetryBC` ensures again the toggling of the :math:`m`-dependent boundary conditions for the eigenfunction at :math:`r=0`. It automatically transfers to e.g. the intersection ``"interface/axis"``, where we have to modify e.g. the Lagrange multiplier for the kinematic boundary condition.


.. literalinclude:: rising_bubble.py
   :language: python
   :start-at: # Boundary conditions
   :end-at: self+=eqs@"domain"

Optionally, we can process all calculated eigenvectors. Here, we make sure that the average of the mesh displacement at the interface has a zero complex angle. This is possible since eigenvectors can have an arbitrary nonzero multiplicative factor. In particular, it can be complex to rotate the eigenvector with respect to real and imaginary parts. The method :py:meth:`~pyoomph.generic.problem.Problem.process_eigenvectors` is called whenever eigenvectors are calculated. Here, we just call :py:meth:`~pyoomph.generic.problem.Problem.rotate_eigenvectors` to ensure it is rotated the way mentioned above:

.. literalinclude:: rising_bubble.py
   :language: python
   :start-at: def process_eigenvectors(self, eigenvectors):
   :end-at: return self.rotate_eigenvectors(eigenvectors,"domain/interface/mesh_x",normalize_amplitude=0.2,normalize_dofs=True)

The driver code now mainly sets up the problem. In particular, we have to activate again the azimuthal stability analysis. We need a robust complex eigensolver. For that, you have to install a complex variant of the package SLEPc (see :numref:`petscslepc`).


We then start at some Bond number, relax to the initial state by some transient steps followed by a stationary solve. Then, we create an output file to write the eigenvalues and scan over the Bond number. We solve the eigenproblem using first an initial guess for the eigenvalue (using the ``shift`` and ``target`` kwargs of :py:meth:`~pyoomph.generic.problem.Problem.solve_eigenproblem`). After the first step, we just use the previously calculated eigenvalue as guess for the next Bond number. We can adapt the mesh based on the eigenfunction using :py:meth:`~pyoomph.generic.problem.Problem.refine_eigenfunction`. It will use the :py:class:`~pyoomph.equations.generic.SpatialErrorEstimator` added to the problem to refine with respect to jumps in velocity gradients across the elements. Thereby, strong changes in the eigenfunction are better captured:

.. literalinclude:: rising_bubble.py
   :language: python
   :start-at: with RisingBubbleProblem() as problem:
   :end-at: problem.go_to_param(Bo=problem.Bo.value+dBond)

Eventually, we get the eigenvalues shown below, which agree decently with the data of :cite:`Bonnefis2024`. We can do the same for other liquids and branches described in :cite:`Bonnefis2024`. Note that our mesh is quite coarse and small in terms of the far field, so one might have to take a finer mesh (using the :py:attr:`~pyoomph.generic.problem.Problem.max_refinement_level`) and a larger domain with the properties ``L_top``, ``L_bottom`` and ``W`` of our problem class. Also note that the plots of the solutions in :cite:`Bonnefis2024` apparently scale the nondimensional radius, not the diameter to unity. Therefore, the fields have different amplitudes.

..  figure:: rising_bubble.*
	:name: figrisingbubble
	:align: center
	:alt: Eigenvalues of the first :math:`m=1` instability	
	:class: with-shadow
	:width: 60%
	
	Eigenvalues of the first :math:`m=1` instability of a rising bubble with :math:`Mo=6.2\times 10^{-7}` (DMS-T05), agreeing well with the literature data. We thank Javier Sierra-Ausin and Jacques Magnaudet for providing the data of their paper.

We can also generate a movie of the instability. Please refer to :numref:`secploteigendynamics` for a tutorial on this.

.. only:: html

	.. raw:: html 

		<figure class="align-center" id="vidrisingbubble"><video autoplay="True" preload="auto" width="60%" loop=""><source src="../../../_static/rising_bubble.mp4" type="video/mp4"></video><figcaption><p><span class="caption-text">Eigendynamics at <span class="math notranslate nohighlight">\(Bo=4\)</span> </span></p></figcaption></figure>
	
	
.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <rising_bubble.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	

