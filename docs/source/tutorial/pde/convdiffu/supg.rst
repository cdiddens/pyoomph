.. _secpdecconvdiffusupg:

SUPG implementation
~~~~~~~~~~~~~~~~~~~

An analogous approach to the *upwind scheme* in finite differences is the *SUPG* stabilization in the finite element method. In the *upwind scheme*, the first order advective derivative is evaluated in the upwind direction, i.e. taking the slope in the upwind direction, which stabilizes the scheme for high *Péclet numbers*. The *SUPG* method (streamline upwind Petrov-Galerkin) does essentially the same in the finite element method. However, while it is trivial to find the degrees of freedom in the upwind direction on a regular line or 2d/3d grid in the finite difference method, for arbitrary meshes, as commonly used in the finite element method, it is not that trivial.

The one-dimensional problem is best to illustrate the idea, so we will stick to it here. The general idea is to weight the upwind residuals more, i.e. to modify the localization of the residual by the projection on the test function to be enhanced in upwind direction. This can be achieved by replacing the test function :math:`\phi` in :math:numref:`eqpdeconvdiffuweakA` by :math:`\phi+\tau\vec{u}\cdot\nabla \phi`. The choice of the parameter :math:`\tau` is crucial and will be discussed in a minute. The SUPG variant of :math:numref:`eqpdeconvdiffuweakA` (with :math:`\nabla\cdot \vec{u}=0` and for pure Dirichlet boundary conditions) hence reads (cf. e.g. :cite:`Bochev2004`):

.. math:: :label: eqpdeconvdiffuweakSUPG

   \left(\partial_t c+\vec{u}\cdot\nabla c,\phi\right)+\left(D\nabla c,\nabla \phi\right) +\left(\partial_t c+\vec{u}\cdot\nabla c, \tau\vec{u}\cdot\nabla\phi \right) =0

We have assumed a first order space (``"C1"``) here (or vanishing diffusivity :math:`D`), which removes the term :math:`D\nabla^2c` in the last weak contribution. Pyoomph cannot handle second order derivatives yet, so this approximation is also necessary at the moment for ``"C2"`` spaces, where :math:`\nabla^2c` does not vanish in each element. The selection of :math:`\tau` is important. There are several approaches, but usually it is defined as a constant per element. It must have a finite value for :math:`\vec{u}\to 0`, and must compensate the :math:`\vec{u}` term in the test function argument of the last term of :math:numref:`eqpdeconvdiffuweakSUPG` to prevent a dominance of the stabilization term for high velocities. Furthermore, it must vanish for infinitely refined meshes (so it must depend on the element size). The idea is to introduce the mesh Péclet number

.. math:: \operatorname{Pe}_h =\frac{h\|\vec{u}\|}{2D}

where :math:`h` is the size of the current element (e.g. the length of a one-dimensional element or the circumference for 2d elements). When :math:`\operatorname{Pe}_h\to 0`, we do not require any stabilization, which happens for low velocities, high diffusivities or small elements. If :math:`\operatorname{Pe}_h` becomes large (typically :math:`>3`), stabilization is necessary to prevent the spurious oscillations for advection-dominated problems. Hence, a good selection of :math:`\tau` is

.. math:: \tau_h =\frac{h}{2\|\vec{u}\|}\left(\operatorname{coth}\left(\operatorname{Pe}_h\right)-\frac{1}{\operatorname{Pe}_h}\right)

The term in the brackets indeed is :math:`0` for :math:`\operatorname{Pe}_h=0` and goes to unity for large :math:`\operatorname{Pe}_h`. The factor :math:`\frac{h}{\|\vec{u}\|}` compensates for the :math:`\nabla` and the velocity appearing in the stabilization projection on :math:`\tau\vec{u}\cdot\nabla\phi`.

To augment the advection-diffusion equations with the stabilization term, we can use the class :py:class:`~pyoomph.equations.SUPG.ElementSizeForSUPG` from :py:mod:`pyoomph.equations.SUPG`. It will calculate the Cartesian measure (i.e. length/area/volume) of each element and store it in a ``"D0"`` space. Since in moving mesh methods (cf. :numref:`secALE`) the elements can change in size, the element size becomes part of the degrees of freedom. One can access the typical element length scale by the method :py:meth:`~pyoomph.equations.SUPG.ElementSizeForSUPG.get_element_h` of the :py:class:`~pyoomph.equations.SUPG.ElementSizeForSUPG` object.

The implementation of the augmented form :math:numref:`eqpdeconvdiffuweakSUPG` reads:

.. literalinclude:: convdiffu_SUPG.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: self.add_residual(time_scheme(self.scheme,weak(radv,self.get_supg_tau() * dot(self.u, grad(ctest)))))

In the method ``get_supg_tau`` we check if the equation is combined with a single :py:class:`~pyoomph.equations.supg.ElementSizeForSUPG` object and bind the size :math:`h`. We calculate :math:`\operatorname{Pe}_h` and thereby :math:`\tau_h` according to the relations discussed above. Finally, this is used for the stabilization term, but only if ``with_SUPG`` is ``True``.

As a test class, we advect again a bump, but this time in one dimension:

.. literalinclude:: convdiffu_SUPG.py
   :language: python
   :start-at: class OneDimAdvectionDiffusionProblem(Problem):
   :end-at: self.add_equations(eqs@"domain")

It is necessary to add an :py:class:`~pyoomph.equations.SUPG.ElementSizeForSUPG` object to calculate the element size if SUPG is active. The rest is trivial, but note that we again use :py:class:`~pyoomph.meshes.bcs.DirichletBC` on both sides. Neumann conditions would have to be augmented by SUPG correction terms stemming from the consistent partial integration that leads to :math:numref:`eqpdeconvdiffuweakSUPG`.


With a simple run code, we can compare the results with and without SUPG:

.. literalinclude:: convdiffu_SUPG.py
   :language: python
   :start-at: if __name__=="__main__":
   :end-at: problem.run(50,outstep=1,maxstep=0.1)

Results are depicted in :numref:`figpdesupg`.

..  figure:: supg.*
	:name: figpdesupg
	:align: center
	:alt: Comparison of the solution with and without SUPG
	:class: with-shadow
	:width: 100%

	Without (left) and with SUPG (right). Note how spurious oscillations are suppressed by SUPG, but the bump diffuses too fast. When the number of elements is increased, both problems vanish, even without SUPG.


.. note::

    An alternative way of getting the typical element size is just using ``var("cartesian_element_size_Eulerian")`` or ``var("element_size_Eulerian")`` instead of :py:class:`~pyoomph.equations.SUPG.ElementSizeForSUPG`. See :py:func:`~pyoomph.expressions.generic.var` for more information on such keyword variables.



.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <convdiffu_SUPG.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
