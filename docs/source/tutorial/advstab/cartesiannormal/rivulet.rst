.. _secadvstabrivulets:

Rayleigh-Plateau instability in presence of a substrate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The previous example was quite simple. In particular, the calculation of the dispersion relation could also have been done analytically by pen and paper.
However, the stability analysis in an additional Cartesian direction can be used to quickly investigate considerably more intricate problems.

Here, we want to consider a long printed line of a liquid on a substrate, also called rivulet. We assume that this rivulet is infinitely long in the :math:`z`-direction and its shape is independent of :math:`z`. Alternatively, we can also think of such a rivulet confined between two plates with distance :math:`L=2\pi/k`, free slip conditions and a :math:`90^\circ` equilibrium contact angle with respect to the plate tangent.

In the absence of a substrate, i.e. for a cylindrically shaped liquid bridge, it is well-known that it undergoes a Rayleigh-Plateau instability when :math:`L/R>2\pi`. However, how does the dynamics change if we consider a rivulet on a substrate instead? Of course, there must be a specific length (or wavenumber :math:`k` in :math:`z`-direction) at which this printed line also undergoes a Rayleigh-Plateau instability, provided that we allow for some slip at the substrate. However, how does the critical wavenumber :math:`k` and the time scale of the instability depend on e.g. the contact angle with respect to the substrate and/or the slip length at the substrate?

We can easily find all the answers by expressing the shape of the base solution only in the :math:`x`-:math:`y`-plane and let pyoomph calculate the stability of such a solution with respect to perturbations in the third direction :math:`z`. We will use Stokes flow for the liquid and combine it with a moving mesh to allow for shape deformations. Moreover, we only consider the right half (i.e. :math:`x\geq 0`) and thereby only select modes which are symmetric with respect to :math:`x` in the following. Therefore, our mesh just creates half of the domain:

.. literalinclude:: rivulet.py
   :language: python
   :start-at: class RivuletMesh(GmshTemplate):
   :end-at: self.plane_surface("interface","substrate","axis",name="liquid")

Note how we use :py:class:`~pyoomph.utils.dropgeom.DropletGeometry` with the kwarg ``rivulet_instead=True`` to convert the volume and the contact angle (which we obtain from the problem defined later on) to the base radius and apex height. Using ``rivulet_instead=True`` actually converts the value shipped by ``volume`` as surface area of the circle segment. In particular, using the volume of :math:`\pi/2`, it gives a radius of curvature of unity for a contact angle :math:`\theta=90^\circ`.

For the problem class, we just define the two parameters (slip length and contact angle) and add all equations to the system.
Since we will have an effectively three-dimensional problem, it is important again to pass the ``wall_tangent`` to the :py:class:`~pyoomph.equations.navier_stokes.NavierStokesContactAngle`. This is a vector pointing inward to the bulk domain tangentially along the substrate, i.e. orthogonal to the ``wall_normal`` which is just the axially upward pointing normal. In a pure axisymmetric or 2d Cartesian case ``wall_tangent=vector(-1,0)`` is fine, but it is not true once the free surface deforms in a third direction (cf. :numref:`threedimdroplet`). If again :math:`\vec{n}` is the normal of the free surface and :math:`\vec{n}_\mathrm{w}` is the wall normal, we can use the double cross product :math:`\vec{n}_\mathrm{w}\times(\vec{n}_\mathrm{w}\times \vec{n})` to obtain such a (non-normalized) vector. We can use the known bac-cab identity along with :math:`\|\vec{n}_\mathrm{w}\|=1` to calculate it via :math:`\vec{n}_\mathrm{w}(\vec{n}_\mathrm{w}\cdot\vec{n})-\vec{n}` and subsequently normalize the result, which is valid for all contact angles :math:`0<\theta<180^\circ`. 

.. literalinclude:: rivulet.py
   :language: python
   :start-at: class RivuletProblem(Problem):
   :end-at: self+=eqs@"liquid"

This only sets up the two-dimensional problem. The eigenanalysis with the additional normal mode is activated in the driver code:

.. literalinclude:: rivulet.py
   :language: python
   :start-at: problem=RivuletProblem() # Create the problem
   :end-at: outf.add_row(k,numpy.real(evs[0]))


Again, it just takes the call of :py:meth:`~pyoomph.generic.problem.Problem.setup_for_stability_analysis` with ``additional_cartesian_mode=True`` to activate this feature and shipping ``normal_mode_k=k`` to the call of :py:meth:`~pyoomph.generic.problem.Problem.solve_eigenproblem`.

The eigenvalues are plotted in :numref:`figrivuletbranches`. It is apparent that, independently of the slip length, the critical wavenumber is at :math:`k=1` for :math:`\theta=90^\circ`, which is reasonable, since the problem can be essentially mirrored at both axes to get the conventional Rayleigh-Plateau instability (at least for high slip lengths). A smaller slip influences the magnitude of the eigenvalues, which is reasonable, since it damps the motion of the contact line. For other contact angles, it is essentially the same, but the critical wave number shifts. Due to the fixed cross-sectional area of the rivulet, a change in contact angle influences the radius of curvature; therefore the critical wave number shifts.
Some plots of the eigendynamics are shown in :numref:`figrivuletplots`, from which the influence of the slip length is clearly apparent.

..  figure:: rivuletbranches.*
	:name: figrivuletbranches
	:align: center
	:alt: Eigenvalues of the rivulet
	:class: with-shadow
	:width: 70%

	Eigenvalues of the rivulet with different contact angles and slip lengths plotted against the wave number :math:`k`.

To visualize the eigenmodes, it is beneficial to modify the problem code above by adding some operators to the :py:class:`~pyoomph.output.meshio.MeshFileOutput`:

.. code:: python
	
	from pyoomph.meshes.meshdatacache import MeshDataCombineWithEigenfunction,MeshDataCartesianExtrusion
        eqs+=MeshFileOutput(operator=MeshDataCombineWithEigenfunction(0)+MeshDataCartesianExtrusion(50))    
        
Here :py:class:`~pyoomph.meshes.meshdatacache.MeshDataCombineWithEigenfunction` will combine the base state with the eigenfunction at index 0, so that both the base solution and the eigenfunction are written to the file for Paraview. :py:class:`~pyoomph.meshes.meshdatacache.MeshDataCartesianExtrusion` will apply the extrusion in the :math:`z`-direction, respecting the oscillation of the eigenmode with :math:`\exp(ikz)`. To write this output, add the :py:func:`~pyoomph.generic.problem.Problem.output` method of the :py:class:`~pyoomph.generic.problem.Problem` to the driver code where you want to have output, however, after a :py:meth:`~pyoomph.generic.problem.Problem.solve_eigenproblem`, so that the eigensolution is available. Afterwards, you can load the files in Paraview, use the ``Calculator`` filter with an expression ``iHat*Eigen_coordinate_x+jHat*Eigen_coordinate_y`` to cast the mesh perturbation to a vector, combine it with ``Wrap by Vector`` and ``Reflect`` filters and you obtain plots as shown in :numref:`figrivuletplots`.

..  figure:: rivuletplots.*
	:name: figrivuletplots
	:align: center
	:alt: Eigenfunctions of the rivulet
	:class: with-shadow
	:width: 100%

	Eigendynamics at :math:`k=0.6` of the rivulet with (a) :math:`\theta=60^\circ, L_\mathrm{slip}=10000`, (b) :math:`\theta=90^\circ, L_\mathrm{slip}=10000` and (c) :math:`\theta=90^\circ, L_\mathrm{slip}=0.01`. Color-coded is the velocity magnitude.
	

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <rivulet.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
