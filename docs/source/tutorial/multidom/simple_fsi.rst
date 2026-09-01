.. _simplefsi:

Fluid-Structure Interaction
---------------------------

Having deformable solids (:numref:`secALEsolid`) and the Navier-Stokes equations on moving domains (:numref:`secALEfreesurfNS`) available, obviously, both can be combined for fluid-structure interaction scenarios.
We consider a 2d channel with two leaflets that will deform by the flow and thereby change the flow as well. The problem case is rather short, just combining the Navier-Stokes equations in the liquid domain and the solid equations in the deformable leaflets:

.. literalinclude:: simple_fsi.py
   :language: python
   :start-at: from pyoomph import *


The real main part is the :py:class:`~pyoomph.equations.solid.FSIConnection`, which must be added to the liquid side of the mutual interface. In :numref:`secconnectfluids`, we discussed how the enforcing of continuous velocity between two liquid domains via Lagrange multipliers actually ensures the balance of tractions. The same idea is used in the :py:class:`pyoomph.equations.solid.FSIConnection` interface. We enforce that the liquid velocity agrees with the solid velocity and thereby also ensure the balance of the tractions at the shared interface. Moreover, the fluid mesh is moved with the solid mesh. As discussed in :numref:`secmultidomheatcond`, it is important to use the same scale of the test function on both sides to balance the tractions. Therefore, one has to set ``scale_for_FSI=True`` in the :py:class:`~pyoomph.equations.solid.DeformableSolidEquations`.

As opposed to the :py:class:`~pyoomph.equations.ALE.ConnectMeshAtInterface` class, which moves the nodes of the meshes on both sides, the :py:class:`~pyoomph.equations.solid.FSIConnection` only moves the nodes of the liquid mesh to match those of the solid mesh. Otherwise, the particular moving mesh dynamics of the fluid domain, which does not reflect any physics, would add additional unphysical tractions to the system.

The :py:class:`~pyoomph.equations.solid.FSIConnection` constrains only those liquid nodes that lie *on* the interface. Around the tips of the leaflets, the imposed motion varies strongly along the interface, so the first layer of liquid elements has to absorb the entire mismatch between the strongly moving interface nodes and the almost stationary interior. Without further measures, these elements shear and squash so badly that some of them are inverted during roughly a quarter of the run, from :math:`t\approx13` on. With the :py:class:`~pyoomph.equations.ALE.InterfaceMeshStiffening` added on the same interface, none of them inverts. It is an interface equation, but it is tested against gradients of the *bulk* mesh test function, i.e. of ``testfunction("mesh",domain="..")``: the shape function of a node that is not on the face vanishes there, but its gradient does not, so a pure surface integral does reach the interior nodes of the attached elements. It is the surface-concentrated limit of a bulk integral over a layer of the thickness of one element, which is why its argument is simply the *relative* extra stiffness of that layer -- the deformation is thereby handed on to the elements further inside, where more of them can share it.

Note that spatial adaptivity is driven here by a :py:class:`~pyoomph.equations.generic.SpatialErrorEstimator` on the liquid velocity alone. The solid domain is given no refinement criterion of its own, yet both sides of the mutual interface must always be refined identically -- an interface element on one side has to find exactly one counterpart on the other. Since the two domains are separate meshes, oomph-lib adapts them individually, so pyoomph takes care of this itself: after every adaptation, the coarser side of a connected interface is refined until the two sides carry matching facets again. You therefore do not have to constrain the interface refinement by hand, and this works under ``mpirun --distribute`` as well, where the two domains are partitioned independently of each other.

.. only:: html

	.. raw:: html 

		<figure class="align-center" id="vidsimplefsi"><video autoplay="True" muted="" playsinline="" controls="" preload="auto" width="80%" loop=""><source src="../../_static/simple_fsi.mp4" type="video/mp4"></video><figcaption><p><span class="caption-text">Fluid-Structure Interaction</span></p></figcaption></figure>
	
	
.. only:: latex

	..  figure:: simple_fsi.*
		:name: figsimplefsi
		:align: center
		:alt: Fluid-Structure Interaction
		:class: with-shadow
		:width: 80%

		Fluid-Structure Interaction



.. only:: html

	.. container:: downloadbutton

		Full code available in the

		:download:`pyoomph example bundle <../tutorial_example_scripts.zip>`

		``Multiple_Domains/simple_fsi.py``
		    		
