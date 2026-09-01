.. _secmcflowsaltcapillary:

A salt in an evaporating capillary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The capillary of :numref:`secgcl` is a good place to see what a dissolved salt does, and what the two ways of accounting for it make of the same physics. We keep the geometry, the 1d moving mesh and the conservative (GCL) form of the equations, but replace the glycerol by sodium chloride: a tube of :math:`L=20\:\mathrm{mm}` filled with :math:`1\:\mathrm{M}` brine, open at the bottom, where water evaporates into air of :math:`80\:\%` relative humidity. As before, the non-volatile solute stays behind and concentrates near the evaporating end. As opposed to glycerol, however, it is an electrolyte.

The mixture is the one from :numref:`secmcflowsaltsdefine` and AIOMFAC is asked for the activity coefficients:

.. literalinclude:: nacl_capillary_evaporation.py
   :language: python
   :start-at: def define_problem(self):
   :end-at: quantity="relative_humidity",temperature=self.temperature)

Note that the ``salt_treatment`` is passed twice, to :py:func:`~pyoomph.materials.generic.Mixture` and further down to :py:class:`~pyoomph.equations.multi_component.CompositionFlowEquations`. The former is what makes the salt a composition field of the material and hence gives it a mass fraction, a mole fraction, a molar volume and a share of the density; the latter is what makes the equations transport that field instead of a concentration. They describe the same solution and therefore have to agree.

The evaporation rate is set up exactly as in :numref:`secgcl`:

.. literalinclude:: nacl_capillary_evaporation.py
   :language: python
   :start-at: # Get the interface properties
   :end-at: j_water=4*D_vap*(c_water-c_infty)/self.R

The bulk equations are the multi-component flow equations in conservative form, i.e. with ``GCL=True``. The salt needs nothing beyond ``salt_treatment``: the transport equation and, at the evaporating interface, the condition that the salt is left behind is added on its behalf.

.. literalinclude:: nacl_capillary_evaporation.py
   :language: python
   :start-at: # Flow, composition and salt in the bulk
   :end-at: eqs+=LaplaceSmoothedMesh()

The two interfaces are set up exactly as in :numref:`secgcl`: a ``static`` one at the bottom carrying the :py:class:`~pyoomph.materials.mass_transfer.PrescribedMassTransfer` model, and a free one at the top that recedes without evaporating.

With ``salt_treatment="dilute"`` the concentration ``c_NaCl`` is a solved field and appears in the output on its own; with ``"component"`` the solved field is ``massfrac_NaCl`` and the concentration is a substituted expression, which does not end up in the output. Since we want to compare the two, we ask for it -- and for the mass density -- under names of our own:

.. literalinclude:: nacl_capillary_evaporation.py
   :language: python
   :start-at: # c_NaCl is a solved field when
   :end-at: eqs+=TextFileOutput()

Nothing else in the problem has to know which of the two descriptions is in use. The observables below read ``c_NaCl`` and the surface tension, and the water activity at the evaporating end is written as the ratio of the local vapour pressure to the pure one:

.. literalinclude:: nacl_capillary_evaporation.py
   :language: python
   :start-at: # Total amount of salt, total liquid mass and liquid volume
   :end-at: eqs+=IntegralObservableOutput("evaporating_end")@"left"

Finally we run both descriptions in turn into two output directories:

.. literalinclude:: nacl_capillary_evaporation.py
   :language: python
   :start-at: if __name__=="__main__":
   :end-at: problem.run(48*hour,outstep=True,startstep=1*second,maxstep=1*hour,temporal_error=1)

.. note::

   The first time step is :math:`1\:\mathrm{s}` rather than the :math:`1\:\mathrm{ms}` of :numref:`secgcl`. Taking small steps at low diffusivity can end up in a deadlock in the temporal adaptivity. Two nearly equal quantities divided by :math:`\Delta t` cannot be resolved at that scale below the Newton solver threshold and each rejected time step will make it worse in the next try with an even lower :math:`\Delta t`. One can fix this by selecting another temporal scale.
   

:numref:`fignaclcapillary` shows the results. The capillary does **not** dry out. Water evaporates, the salt is left behind, its concentration rises, and with it the water activity falls until it reaches the relative humidity of the surrounding air -- at which point the driving force :math:`c_\mathrm{sat}-c_\infty` vanishes and everything stops. Roughly a fifth of the liquid stays in the tube indefinitely. Of course, crystallization is not considered here.

..  figure:: nacl_capillary.*
	:name: fignaclcapillary
	:align: center
	:alt: Evaporation of a NaCl solution from a capillary, as a dilute solute and as a real component.
	:class: with-shadow
	:width: 100%

	Evaporation of a 1 M NaCl solution from a capillary into air of 80 % relative humidity, with the salt treated as a dilute solute (solid) and as a real mixture component (dashed). (left) the salt concentration along the capillary at different times, the right end of each curve being the receding upper interface. (middle) the remaining liquid volume and the mean mass density. (right) the salt concentration and the water activity at the evaporating end; evaporation stops once the activity has fallen to the ambient relative humidity.

The two descriptions agree on the thermodynamics and disagree on the bookkeeping. Both stop at the same water activity, and both do so at the same *molality*, :math:`5.17\:\mathrm{mol/kg}` -- unsurprisingly, since that is what AIOMFAC is a function of. They differ in what that molality corresponds to. The dilute treatment converts between concentration and molality with :math:`m=c/\rho`, with :math:`\rho` the density of the salt-free solvent, so a cubic metre of it holds :math:`998\:\mathrm{kg}` of water whatever the salt does. A real brine at the final :math:`23\:\%` salt by mass holds :math:`920\:\mathrm{kg}`, and the conversion is out by that :math:`8.5\:\%`. The dilute run therefore ends up reporting :math:`5.16\:\mathrm{M}` where the component run reports :math:`4.75\:\mathrm{M}` -- an :math:`8\:\%` difference in exactly the quantity a surface tension law or a crystallisation criterion would be evaluated at.

The density is the more visible of the two effects. Under the dilute treatment the liquid remains pure water throughout, :math:`998\:\mathrm{kg/m^3}`, since the salt has neither mass nor volume of its own; as a component it starts at :math:`1040\:\mathrm{kg/m^3}` and ends at :math:`1197\:\mathrm{kg/m^3}`. The middle panel of :numref:`fignaclcapillary` shows the consequence: nearly the same mass of evaporated water empties a different amount of tube, because in the dilute description the departing water takes the whole of the liquid volume with it while in the component description the salt keeps its share. About :math:`19\:\%` of the tube is left in one case and :math:`21\:\%` in the other.

Summarizing, at :math:`1\:\mathrm{mM}` of a typical buffer the two are indistinguishable and ``"dilute"`` is both cheaper and less intrusive -- it leaves the solvent mass fractions and the density correlations of the material exactly as they were. Once the salt reaches a few percent by mass, and certainly once a drying problem is going to take it there, ``"component"`` is the accurate description. 

.. only:: html

	.. container:: downloadbutton

		Full code available in the

		:download:`pyoomph example bundle <../../tutorial_example_scripts.zip>`

		``Multicomponent_Flow/nacl_capillary_evaporation.py``
