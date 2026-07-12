.. _secgcl:

Evaporation from a capillary & the geometric conservation law (GCL)
-------------------------------------------------------------------

We want to use the multi-component flow equation to the evaporation of a mixture from a capillary tube, analogously to what has been done in Ref. :cite:`Raju2024`.
As in this reference, we consider the tube, filled with a mixture of glycerol and water, as a 1d system. At the bottom of the tube, water is evaporating, leaving the non-volatile glycerol behind. This implies that the total mass of glycerol inside the liquid must be conserved.

However, for simple ALE methods which just use ``partial_t(...,ALE=True)``, i.e. the ALE-corrected time derivative :py:func:`~pyoomph.expressions.generic.partial_t` (see :numref:`secALEtimediff`), we will see that it does not perfectly conserve the total mass of glycerol. Instead, we get a time-step-dependent conservation, which is a typical result when using ALE without respecting the *geometric conservation law*.

If we want to have conservation up to machine-precision, float truncation and the Newton solver threshold, we must ensure that *Reynolds transport theorem* is satisfied. In the present case, we have to e.g. solve the advection-diffusion equation for the glycerol mass fraction :math:`w_\mathrm{g}` in the liquid phase. For this binary mixture, the equation :math:numref:`eqmcflowwadvdiff` reads

.. math:: :label: eqmcflowwadvdiffnonconservative

    \begin{aligned}
    \rho\left(\partial_t w_\mathrm{g}+\vec{u}\cdot\nabla w_\mathrm{g}\right)=\nabla\cdot\left(\rho D_{\mathrm{gw}} \nabla w_\mathrm{g}\right)    
    \end{aligned}

For a test function :math:`\phi`, the weak form of this equation reads

.. math:: :label: eqweakmcflowwadvdiffnonconservative

    \begin{aligned}
    \left(\rho\left(\partial_t w_\mathrm{g}+\vec{u}\cdot\nabla w_\mathrm{g}\right),\phi\right) + \left(\rho D_{\mathrm{gw}} \nabla w_\mathrm{g},\nabla \phi\right)=\left\langle\rho D_{\mathrm{gw}} \nabla w_\mathrm{g},\mathbf{n} \phi\right\rangle
    \end{aligned}

This is how pyoomph implements the equation by default, where the time derivative considers the ALE correction on moving meshes. However, as shown below, this does not perfectly conserve the total mass of glycerol in the capillary

Alternatively, we can use the continuity equation of :math:numref:`eqmcflowwadvdiff` to rewrite :math:numref:`eqmcflowwadvdiffnonconservative` in a *conservative form* as

.. math::  :label: eqmcflowwadvdiffconservative

    \begin{aligned}
    \partial_t (\rho w_\mathrm{g})+\nabla\cdot(\rho\vec{u} w_\mathrm{g})=\nabla\cdot\left(\rho D_{\mathrm{gw}} \nabla w_\mathrm{g}\right)    
    \end{aligned}

For the corresponding weak form, we want to make sure that the change of the volume of each element is taken into account, i.e. the time derivative is applied on the entire integral in the weak formulation, considering the changing integration domain.

.. math:: :label: eqweakmcflowwadvdiffconservative

    \begin{aligned}
    \frac{\mathrm{d}}{\mathrm{d}t} \left(\rho w_\mathrm{g},\phi\right) -\left(\rho (\vec{u}-\dot{\vec{x}})w_\mathrm{g},\nabla \phi\right) + \left(\rho D_{\mathrm{gw}} \nabla w_\mathrm{g},\nabla \phi\right)=\left\langle\rho D_{\mathrm{gw}} \nabla w_\mathrm{g} - \rho (\vec{u}-\dot{\vec{x}})w_\mathrm{g},\mathbf{n} \phi\right\rangle
    \end{aligned}

The time-derivative acts now on the entire integral and the advection term was integrated by parts, giving also rise to a boundary term, i.e. Neumann conditions are now not just the diffusive fluxes as before, but also comprise advective fluxes. The ALE correction is now incorporated directly in the advective term with the mesh velocity :math:`\dot{\vec{x}}`. In pyoomph, the time derivative of a weak form integral can be implemented using the :py:func:`~pyoomph.expressions.time_derivative_of_integral` function, e.g. ``time_derivative_of_integral(weak(rho*w_g,phi))``. 

On a continuous level, both approaches are equivalent, but on a discrete level, the conservative form is more accurate in terms of conservation.

However, on the discrete level, there is another aspect to consider: *the geometric conservation law (GCL)*. In order to fulfill Reynolds transport theorem, the discrete mesh velocity must be consistent with the discrete time derivative of the element volume. This is not automatically guaranteed in ALE methods. For first order time stepping with ``"BDF1"``, the GCL is automatically satisfied, but for higher order time stepping, it is not. 

.. note::

    At the moment, pyoomph does not fully handle the GCL for higher order time stepping, but the results are still much better in conservative form as seen below.

The multi-component flow equations can activate the conservative form by setting ``GCL=True`` in the :py:class:`~pyoomph.equations.multi_component.CompositionFlowEquations`. This also augments the Navier-Stokes equation and adjusts e.g. the evaporation boundary conditions to account for the changed Neumann terms. Custom Neumann conditions, e.g. a diffusive influx on a boundary, must be adjusted accordingly. 

Let us now assess the conservation of the total mass of glycerol in the capillary tube. The code is just a quite simple 1d problem, i.e. the problem is defined as follows:

.. code:: python

    from pyoomph import *
    from pyoomph.equations.multi_component import *
    from pyoomph.equations.ALE import *
    from pyoomph.materials import *
    from pyoomph.materials.mass_transfer import *
    import pyoomph.materials.default_materials 

    class CapillaryEvaporationProblem(Problem):
        def __init__(self):
            super().__init__()
            # Filled height in the capillary
            self.L=20*milli*meter
            # Capillary radius
            self.R=0.5*milli*meter        
            # Temperature of the system
            self.temperature=20*celsius
            # Initial Liquid mixture composition (glycerol/water)
            self.mixture=Mixture(20*percent*get_pure_liquid("glycerol")+get_pure_liquid("water"))
            # Gas phase composition (air with 20% relative humidity)
            self.gas=Mixture(get_pure_gas("air")+50*percent*get_pure_gas("water"),quantity="relative_humidity",temperature=20*celsius)
            # Gravity and ambient pressure
            self.g=9.81*meter/second**2
            self.ambient_pressure=1*atm
            # Whether we use the conservative form (better GCL agreement) or not
            self.use_GCL=True    




..  figure:: gcl_capillary.*
	:name: figgclcapillary
	:align: center
	:alt: Glycerol-water mixture in a capillary tube with evaporation at the bottom.
	:class: with-shadow
	:width: 100%

	Evaporation of a glycerol-water mixture from a capillary tube. (left) the concentration of glycerol in the capillary at different times. (middle) the velocity of the top interface, receding due to evaporation at the bottom. (right) the total mass of glycerol in the capillary, which is not perfectly conserved when using ALE without GCL, but is conserved nicely when using the GCL.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <gcl_glycerol_water_capillary.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   	
		   