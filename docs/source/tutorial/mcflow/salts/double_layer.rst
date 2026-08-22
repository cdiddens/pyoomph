.. _secmcflowsaltdoublelayer:

The electric double layer, and how fast it relaxes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So far the salt has been electroneutral everywhere: :py:class:`~pyoomph.equations.salt_transport.SaltTransportEquations` transports a salt as an ion pair that never separates, which is the right description whenever the geometry is far larger than the Debye length. This section resolves what that description averages over. We put an electrolyte film between an electrode and a gas gap, switch on a voltage, and watch the diffuse layer build up -- with :py:func:`~pyoomph.equations.electrostatics.PoissonNernstPlanck`, i.e. one field per ion plus the potential, and no electroneutrality assumed anywhere.

The cell is one-dimensional: :math:`10\:\mathrm{\mu m}` of liquid, then :math:`10\:\mathrm{\mu m}` of air, with an electrode in the liquid at :math:`5\:\mathrm{V}` and a counter electrode beyond the gas at :math:`0\:\mathrm{V}`. The electrolyte is a symmetric 1:1 salt at :math:`1\:\mathrm{\mu M}`, so that :math:`\lambda_\mathrm{D}=305\:\mathrm{nm}` and a uniform mesh can resolve it.

.. literalinclude:: double_layer_relaxation.py
   :language: python
   :start-at: class DoubleLayerRelaxationProblem(Problem):
   :end-at: self.Ngas=40

**One scale has to be chosen deliberately.** Scales in pyoomph are cosmetic for the answer, but not for the solver, and this problem is a good illustration of the difference:

.. literalinclude:: double_layer_relaxation.py
   :language: python
   :start-at: def define_problem(self):
   :end-at: self.set_scaling(spatial=self.Lliq,temporal=self.debye_time())

The natural scale of an electrokinetic problem is the thermal voltage :math:`RT/F=25\:\mathrm{mV}`, and that is what :py:func:`~pyoomph.equations.electrostatics.set_electrostatic_scaling` uses by default. Here the applied voltage is two hundred times larger, so the nondimensional potential would run into the hundreds -- and the Newton solver then fails on the very first step, for every voltage above about :math:`0.15\:\mathrm{V}`, no matter how small the time step or how slowly the voltage is ramped. Passing ``potential=self.V`` instead fixes it outright. The rule is that the potential scale should be the largest potential the problem actually contains, not the one the physics is written in.

The cell itself needs very little. Both domains are cut out of a single ``LineMesh``, which creates the interface between them automatically:

.. literalinclude:: double_layer_relaxation.py
   :language: python
   :start-at: # One mesh, two domains
   :end-at: gas+=ElectrodeBC(0)@"counter"

Note that the electrode needs nothing but its potential. The natural boundary condition of :py:class:`~pyoomph.equations.electrostatics.NernstPlanckEquations` is already a blocking wall, i.e. an ideally polarizable electrode at which no ion is discharged, which is exactly what we want.

**What the double layer actually sees.** The liquid is a conductor and the gas is not, so the two are capacitors in series and the smaller capacitance takes the voltage. With :math:`C_\mathrm{gas}=\varepsilon_\mathrm{g}/L_\mathrm{gas}` and :math:`C_\mathrm{DL}=\varepsilon_\mathrm{l}/\lambda_\mathrm{D}` their ratio is

.. math:: \frac{C_\mathrm{gas}}{C_\mathrm{DL}}=\frac{\varepsilon_\mathrm{g}\lambda_\mathrm{D}}{\varepsilon_\mathrm{l}L_\mathrm{gas}}\approx 4\cdot 10^{-4}\,,

so the diffuse layer only ever gets that fraction of the applied voltage: :math:`5\:\mathrm{V}` across the cell leaves :math:`3.8\:\mathrm{mV}` across the layer. That is a property of an insulating gap in series with an electrolyte, not of the discretisation, and it is why this cell stays comfortably in the linear regime. It also means the interesting quantity is not the potential at the interface -- which follows the applied voltage almost exactly -- but what is left after subtracting it:

.. literalinclude:: double_layer_relaxation.py
   :language: python
   :start-at: # The potential drop across the diffuse layer
   :end-at: liq+=ifeqs@"liq_gas"

Because the layer is thin against the film, :math:`\zeta_\infty` is proportional to :math:`\lambda_\mathrm{D}` and therefore to :math:`1/\sqrt{c_0}`. The measured values, :math:`-7.60`, :math:`-3.80`, :math:`-1.90` and :math:`-0.95\:\mathrm{mV}` at :math:`0.25`, :math:`1`, :math:`4` and :math:`16\:\mathrm{\mu M}`, halve exactly as :math:`\lambda_\mathrm{D}` halves.

Three time scales, and only one of them is right
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An electrolyte between electrodes offers three candidate relaxation times: the Debye time :math:`\lambda_\mathrm{D}^2/D`, the *RC* time :math:`\lambda_\mathrm{D}L/D` of the double layer charging through the resistance of the bulk, and the bulk diffusion time :math:`L^2/D`. Here they are :math:`93\:\mathrm{\mu s}`, :math:`3.0\:\mathrm{ms}` and :math:`100\:\mathrm{s}` -- six orders of magnitude apart, so guessing is not an option and a single run cannot distinguish them either. A concentration sweep can, because the three scale differently with :math:`\lambda_\mathrm{D}`:

.. literalinclude:: double_layer_relaxation.py
   :language: python
   :start-at: # A concentration sweep
   :end-at: solve(problem,"dl_sweep_{:g}nM".format(float(c0/(nano*molar))))

:numref:`figdoublelayerrelaxation` (a) plots :math:`|\zeta-\zeta_\infty|` against :math:`t/\tau_\mathrm{D}` on a logarithmic axis. All four runs are single exponentials and all four collapse onto the same line, which is the statement that the relaxation time *is* :math:`\lambda_\mathrm{D}^2/D`. Fitting them gives :math:`0.940`, :math:`0.971`, :math:`0.985` and :math:`0.991\:\tau_\mathrm{D}` -- converging to one as the layer gets thinner against the film, which is the limit in which :math:`\tau_\mathrm{D}` is derived. Over the same 64-fold change in concentration the *RC* time moves by a factor of eight, so panel (b) leaves no room for doubt about which of the three is being measured.

That the answer is :math:`\tau_\mathrm{D}` and not the *RC* time is worth a moment. The classical result for a cell driven by *blocking electrodes on both sides* is :math:`\lambda_\mathrm{D}L/D`: the double layers are charged by a current that has to cross the bulk, so the bulk resistance is in the loop. Here it is not, because the gas capacitance is so much smaller than either double layer that almost no charge has to move at all -- the layer merely relaxes to local equilibrium with a field that is already there. Put both electrodes into the liquid and the answer changes to :math:`\lambda_\mathrm{D}L/D`.

.. note::

   The run uses fixed time steps rather than adaptive ones. An adaptive stepper is very good at stepping over a single exponential: with ``temporal_error=1`` this problem reaches equilibrium inside the first output interval and there is nothing left to look at.

Transferring charge onto the interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

So far the liquid|gas interface has only been polarized: the field pushes anions towards it and cations away, but nothing crosses it and the total charge on either side stays zero. If instead ions may *adsorb* on the interface, its charge stops being a rearrangement of the liquid and becomes a quantity of its own. :py:class:`~pyoomph.equations.electrostatics.SurfaceChargeConservation` solves for it, and its ``adsorption`` argument couples it back to the bulk:

.. literalinclude:: double_layer_relaxation.py
   :language: python
   :start-at: if self.transfer:
   :end-at: ifeqs=ElectricPotentialConnection()

The rate is given per ion and in moles, so the class knows both what it means for the charge -- :math:`z_iF` times the rate -- and what it means for the electrolyte, namely that exactly the same number of moles has to leave it. Nothing has to be written twice, and nothing can be written inconsistently.

The competition is between the charge the field induces and the charge the surface can hold, so the parameter that matters is the site density :math:`\Gamma_\mathrm{max}` measured against :math:`c_0\lambda_\mathrm{D}`, the amount of ion already sitting within a Debye length of the surface. :numref:`figdoublelayerrelaxation` (c) shows what happens as it is raised: at :math:`0.1\,c_0\lambda_\mathrm{D}` the adsorbed charge shifts :math:`\zeta` from :math:`-3.80` to :math:`-3.16\:\mathrm{mV}`, at :math:`0.5` it nearly cancels the field-driven layer, and at :math:`2` it overturns it -- the adsorbed cations now charge the surface positively and :math:`\zeta` comes out at :math:`+6.51\:\mathrm{mV}`, of the opposite sign to the applied field. The relaxation time is almost untouched throughout, between :math:`0.97` and :math:`1.00\:\tau_\mathrm{D}`: adsorption decides where the layer ends up, not how quickly it gets there.

Conservation as a diagnostic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every wall in this problem blocks, so a cation can only be dissolved or adsorbed and the sum of the two may not move at all. That is worth measuring rather than assuming, because it is exactly the quantity a wrong sign in the surface coupling would break:

.. literalinclude:: double_layer_relaxation.py
   :language: python
   :start-at: # Ion conservation: with blocking walls
   :end-at: charge=var("charge_density"))

Across all runs the sum drifts by less than :math:`10^{-15}` relative, i.e. it is conserved to the Newton tolerance rather than to the order of the time stepping. Two properties of the discretisation are responsible: the ion equations use the conservative form (see ``GCL`` in :py:class:`~pyoomph.equations.electrostatics.NernstPlanckEquations`), and the surface term the adsorption adds to the bulk is the same boundary term :py:class:`~pyoomph.equations.electrostatics.IonFluxBC` uses, with the opposite sign to the one it adds to the surface. Watching this number is the cheapest way to find out that a hand-written interface condition is wrong.

..  figure:: double_layer_relaxation.*
	:name: figdoublelayerrelaxation
	:align: center
	:alt: Charging and relaxation of an electric double layer in a liquid film facing a gas gap.
	:class: with-shadow
	:width: 100%

	An electrolyte film between an electrode and a gas gap, after switching on 5 V. (a) the approach of the diffuse-layer potential to its final value, for four bulk concentrations, against time in units of the Debye time; all four are single exponentials and all four collapse. (b) the measured relaxation time against the three candidate time scales. (c) the diffuse-layer potential with ions adsorbing on the liquid|gas interface, for three site densities; above about :math:`c_0\lambda_\mathrm{D}` the adsorbed charge overturns the field-driven layer.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <double_layer_relaxation.py>`

		:download:`Download all examples <../../tutorial_example_scripts.zip>`
