.. _compresseddisc:

Compression of 2D circular disk 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
	The following case is a direct adaption of the `corresponding example in oomph-lib <https://oomph-lib.github.io/oomph-lib/doc/solid/disk_compression/html/index.html>`__.

We consider a 2d disc that is compressed by a uniform pressure. The disc initially has a radius of unity, however, we introduce an isotropic growth factor of :math:`\Gamma=1.1`, so that the disc wants to grow to a radius of :math:`\sqrt{\Gamma}` in absence of any external pressure. Opposed to the `corresponding example in oomph-lib <https://oomph-lib.github.io/oomph-lib/doc/solid/disk_compression/html/index.html>`__, we will come up with two implementations of the very same case. We either can solve the problem on a quarter circle mesh with symmetry boundary conditions in a two-dimensional Cartesian coordinate system. However, due to the coordinate-system agnostic formulation of equations, the same can be realized by a simple radial line mesh in a polar coordinate system. To that end, we introduce a flag ``polar_implementation`` in the problem class:


.. literalinclude:: compressed_disc.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: eqs+=DirichletBC(mesh_y=0)@"center_to_east"

The basis setup is analogous to the previous example, i.e. we require a constitutive law which is then used in the :py:class:`~pyoomph.equations.solid.DeformableSolidEquations`. Here, however, we impose the ``isotropic_growth_factor``, which lets the disc grow everywhere from its undeformed configuration by :math:`\Gamma` in terms of the area. Again, a :py:class:`~pyoomph.equations.solid.SolidNormalTraction` is imposed at the boundary ``circumference``. If ``polar_implementation==True``, we switch to an `axisymmetric`` coordinate system (which is a radial polar coordinate system for 1d meshes) and use a simple 1d mesh. In that case, the right boundary will be called ``circumference``, whereas the left boundary of the interval is called ``center``. At the latter, we make sure that the mesh position is fixed to the origin. If we do not solve the polar case, we do solve it on a quarter circle mesh on a 2d Cartesian coordinate system. We have to fix the mesh coordinates on the axes of symmetry in that case.

We also want to measure the current radius :math:`r` of the disc. Irrespective of the coordinate system, we can do so by integrating over the boundary ``circumference``. We calculate two integrals, namely the line length :math:`L=\int 1\:\mathrm{d}l` and the integral over the radius :math:`R=\int \|\vec{x}\|\:\mathrm{d}l`. In case of the 2d Cartesian implementation, both integrals will be only a quarter of the full disc, but this does not matter, since the (averaged) radius of the disc can be obtained by the ratio :math:`r=R/L`:

.. literalinclude:: compressed_disc.py
   :language: python
   :start-at: # To monitor the radius of the disc, we can use IntegralObservables. We integrate over the circumference of the disc to the the line length
   :end-at: self+=eqs@"domain"

In the driver code, we just iterate over the imposed pressure (starting with a negative pressure to pull the disc outwards first). To compare the actual radius :math:`r` with an analytical linearized expression (see the `oomph-lib example <https://oomph-lib.github.io/oomph-lib/doc/solid/disk_compression/html/index.html>`__ for details), we can evaluate the introduced observable and write both the numerical value and the analytical approximation to a text file in the output directory:

.. literalinclude:: compressed_disc.py
   :language: python
   :start-at: with CompressedDiscProblem() as problem:
   :end-at: problem.P.value+=delta_p

..  figure:: compressed_disc.*
	:name: figalecompresseddisc
	:align: center
	:alt: Compressing a disc
	:class: with-shadow
	:width: 50%

	Compressing a disc with an isotropic growth factor



.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <compressed_disc.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    		
