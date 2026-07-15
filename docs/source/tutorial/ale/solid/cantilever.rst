.. _cantilever:

Bending of a cantilever beam 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
	The following case is a direct adaption of the `corresponding example in oomph-lib <https://oomph-lib.github.io/oomph-lib/doc/solid/airy_cantilever/html/index.html>`__.

We consider a 2d solid cantilever of length :math:`L` and height :math:`2H` which is attached to a fixed wall on its left side. At the top of the cantilever, a pressure load :math:`P` is applied in normal direction. The cantilever hence bends downwards. The problem class is again quite short


.. literalinclude:: cantilever.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: eqs+=SolidNormalTraction(self.P)@"top" # Apply pressure on the top of the cantilever

Relevant parts are the import of the solid equations of :py:mod:`pyoomph.equations.solid`. After defining the default parameters in the problem constructor (where the pressure load :math:`P` is introduced as global parameter to vary it later on), the problem only consists of a simple :py:class:`~pyoomph.meshes.simplemeshes.RectangularQuadMesh` and :py:class:`~pyoomph.equations.solid.DeformableSolidEquations` associated with the desired boundary conditions. :py:class:`~pyoomph.equations.solid.SolidNormalTraction` will introduce the varying pressure load to the top of the cantilever in normal direction. The material properties are introduced by instances of particular constitutive law classes. Here, in particular, we use :py:class:`~pyoomph.equations.solid.GeneralizedHookeanSolidConstitutiveLaw`, which is a compressible nonlinear generalization of Hooke's law. It requires Young's modulus :math:`E` and the Poisson ratio :math:`\nu` as parameters. Note that the latter may not be :math:`\nu=1/2`, since this leads to a singularity. In this case, the material is incompressible and one has to use particular incompressible constitutive laws like e.g. the :py:class:`~pyoomph.equations.solid.IncompressibleHookeanSolidConstitutiveLaw`. Opposed to a compressible law, an incompressible constitutive law requires the introduction of a pressure field, which acts as a Lagrange multiplier enforcing the incompressibility everywhere. Since the pressure offset (reference pressure) can be chosen arbitrarily, we must remove this null space, which can be done by either fixing a single pressure degree of freedom or by demanding that e.g. the average pressure should vanish. The latter approach is easily realized by an :py:class:`~pyoomph.equations.generic.AverageConstraint`. 

Besides the constitutive law, the :py:class:`~pyoomph.equations.solid.DeformableSolidEquations` allows us to set a mass density (which is only relevant in transient problems for the inertia term) and a potential bulk force density (note that this is meant with respect to the *undeformed* solid). With ``coordinate_space="C2"``, we explicitly ask for second order elements. By default, the order of the elements will be determined by the highest order space of all other fields defined on this mesh, but since we do not define any other fields, we have to explicitly set it here. In case of a pressure field required by an incompressible constitutive law, we also can select the space of the pressure field. Here, with ``pressure_space="DL"``, we select a discontinuous pressure field, but alternatively, we could have used e.g. ``pressure_space="C1"`` for a continuous pressure space. Finally, ``with_error_estimator=True``, will introduce a functionality like the :py:class:`~pyoomph.equations.generic.SpatialErrorEstimator`, so that the mesh can be adapted based on the strains in the solid.

As in the `corresponding example in oomph-lib <https://oomph-lib.github.io/oomph-lib/doc/solid/airy_cantilever/html/index.html>`__, we also want to compare the numerically obtained stresses with an approximate analytical solution. This can be done by adding :py:class:`~pyoomph.equations.generic.LocalExpressions`, which will write the numerical and analytical stresses to the output. We therefore continue the definition of the problem by 

.. literalinclude:: cantilever.py
   :language: python
   :start-at: # To compare the numerical solution with the analytical solution, we write the numerical and analytical stress tensors
   :end-at: self+=eqs@"domain"

More details on the analytical solution can be found in `the documentation of oomph-lib <https://oomph-lib.github.io/oomph-lib/doc/solid/airy_cantilever/html/index.html>`__.

Finally, we can just run the problem by gradually increasing the pressure load, solving the problem at each load and writing the output:

.. literalinclude:: cantilever.py
   :language: python
   :start-at: with AiryCantileverProblem() as problem:
   :end-at: problem.output_at_increased_time()

..  figure:: cantilever.*
	:name: figalecantilever
	:align: center
	:alt: Bending of a cantilever beam
	:class: with-shadow
	:width: 100%

	Bending of a cantilever beam under increasing pressure load at the top



.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <cantilever.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    		
