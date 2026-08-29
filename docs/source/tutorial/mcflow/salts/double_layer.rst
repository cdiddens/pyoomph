.. _secmcflowsaltdoublelayer:

The electric double layer, and how fast it relaxes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are interested in the effect of ions and salts on a large geometry, considerably larger than the Debye length, if is perfectly fine to use the :py:func:`~pyoomph.equations.multi_component.CompositionFlowEquations`. It will add :py:class:`~pyoomph.equations.salt_transport.SaltTransportEquations` for each salt to account for its transport, considering a salt as an ion pair that never separates. This section resolves what that description averages over. We put an electrolyte film between an electrode and a gas gap, switch on a voltage, and watch the diffuse layer build up -- with :py:func:`~pyoomph.equations.electrostatics.PoissonNernstPlanck`, i.e. one field per ion plus the potential, and no electroneutrality assumed anywhere.

The cell is one-dimensional: :math:`10\:\mathrm{\mu m}` of liquid, then :math:`10\:\mathrm{\mu m}` of air, with an electrode in the liquid at :math:`5\:\mathrm{V}` and a counter electrode beyond the gas at :math:`0\:\mathrm{V}`. The electrolyte is a symmetric 1:1 salt at :math:`1\:\mathrm{\mu M}`, so that :math:`\lambda_\mathrm{D}=305\:\mathrm{nm}` and a uniform mesh can resolve it.

.. literalinclude:: double_layer_relaxation.py
   :language: python
   :start-at: class DoubleLayerRelaxationProblem(Problem):
   :end-at: self.Ngas=40

While scales in pyoomph can be chosen quite arbitrarily, they might have a huge impact on the solver. For this problem, it is e.g. key to select the imposed voltage as scale for the electric potential:

.. literalinclude:: double_layer_relaxation.py
   :language: python
   :start-at: def define_problem(self):
   :end-at: self.set_scaling(spatial=self.Lliq,temporal=self.debye_time())

The natural scale of an electrokinetic problem is the thermal voltage :math:`RT/F=25\:\mathrm{mV}`, and that is what :py:func:`~pyoomph.equations.electrostatics.set_electrostatic_scaling` uses by default. Here the applied voltage is two hundred times larger, so the nondimensional potential would run into the hundreds. The Newton solver then fails on the very first step. Passing ``potential=self.V`` instead fixes this issue.

The 1d cell is trivial again: both domains are cut out of a single ``LineMesh``, which creates the interface between them automatically:

.. literalinclude:: double_layer_relaxation.py
   :language: python
   :start-at: # One mesh, two domains
   :end-at: gas+=ElectrodeBC(0)@"counter"

Note that the electrode just requires its potential. The natural boundary condition of :py:class:`~pyoomph.equations.electrostatics.NernstPlanckEquations` is already a blocking wall, i.e. an ideally polarizable electrode without ion discharge, which is exactly what we want.

Despite of the considerable voltage drop at this small scale, the electric field inside the liquid is minor. The liquid is a conductor and the gas is not, so the two are capacitors in series and the smaller capacitance takes the voltage. With :math:`C_\mathrm{gas}=\varepsilon_\mathrm{g}/L_\mathrm{gas}` and :math:`C_\mathrm{DL}=\varepsilon_\mathrm{l}/\lambda_\mathrm{D}` their ratio is

.. math:: \frac{C_\mathrm{gas}}{C_\mathrm{DL}}=\frac{\varepsilon_\mathrm{g}\lambda_\mathrm{D}}{\varepsilon_\mathrm{l}L_\mathrm{gas}}\approx 4\cdot 10^{-4}\,,

:math:`5\:\mathrm{V}` across the cell leaves :math:`3.8\:\mathrm{mV}` across the layer in the liquid. This also implies that the interesting quantity is not the potential at the interface, which follows the applied voltage almost exactly, but what is left after subtracting it:

.. literalinclude:: double_layer_relaxation.py
   :language: python
   :start-at: # The potential drop across the diffuse layer
   :end-at: liq+=ifeqs@"liq_gas"

Because the layer is thin against the film, :math:`\zeta_\infty` is proportional to :math:`\lambda_\mathrm{D}` and therefore to :math:`1/\sqrt{c_0}`. The measured values, :math:`-7.60`, :math:`-3.80`, :math:`-1.90` and :math:`-0.95\:\mathrm{mV}` at :math:`0.25`, :math:`1`, :math:`4` and :math:`16\:\mathrm{\mu M}`, halve exactly as :math:`\lambda_\mathrm{D}` halves.


An electrolyte between electrodes offers three candidate relaxation times: the Debye time :math:`\lambda_\mathrm{D}^2/D`, the *RC* time :math:`\lambda_\mathrm{D}L/D` of the double layer charging through the resistance of the bulk, and the bulk diffusion time :math:`L^2/D`. In the present case, they read :math:`93\:\mathrm{\mu s}`, :math:`3.0\:\mathrm{ms}` and :math:`100\:\mathrm{s}`, respectively, six orders of magnitude apart. To determine the actually relevant time scale, we can perform a sweep over the initial concentration:

.. literalinclude:: double_layer_relaxation.py
   :language: python
   :start-at: # A concentration sweep
   :end-at: solve(problem,"dl_sweep_{:g}nM".format(float(c0/(nano*molar))))

:numref:`figdoublelayerrelaxation` (a) plots :math:`|\zeta-\zeta_\infty|` against :math:`t/\tau_\mathrm{D}` on a logarithmic axis. All four runs are exponentials and all four collapse onto the same line. Therefore, the relaxation time is indeed given by :math:`\lambda_\mathrm{D}^2/D`, confirmed by fits, which yield :math:`0.940`, :math:`0.971`, :math:`0.985` and :math:`0.991\:\tau_\mathrm{D}`.


One might wonder what happens if ion adsorption is allowed in the present scenario. So far the liquid-gas interface has only been polarized: the field pushes anions towards it and cations away. Adsorption of a charge stops being a rearrangement of the liquid alone and it becomes a surface quantity of its own. :py:class:`~pyoomph.equations.electrostatics.SurfaceChargeConservation` solves for it, and its ``adsorption`` argument couples it back to the bulk:

.. literalinclude:: double_layer_relaxation.py
   :language: python
   :start-at: if self.transfer:
   :end-at: ifeqs=ElectricPotentialConnection()

The rate is given per ion and in moles. Here, we take a Langmuir isotherm, i.e. a competition between the charge the field induces and the charge the surface can hold. 

:numref:`figdoublelayerrelaxation` (c) shows what happens as the site density :math:`\Gamma_\mathrm{max}` is raised. At :math:`0.1\,c_0\lambda_\mathrm{D}` the adsorbed charge shifts :math:`\zeta` from :math:`-3.80` to :math:`-3.16\:\mathrm{mV}`, at :math:`0.5\,c_0\lambda_\mathrm{D}` it nearly cancels the field-driven layer, and at :math:`2\,c_0\lambda_\mathrm{D}` it overturns it. The adsorbed cations now charge the surface positively and :math:`\zeta` comes out at :math:`+6.51\:\mathrm{mV}`, of the opposite sign to the applied field. The relaxation time is almost untouched throughout, between :math:`0.97` and :math:`1.00\:\tau_\mathrm{D}`: adsorption decides where the layer ends up, not how quickly it gets there. Of course, this depends also on the particular choice of ``k_a`` and ``k_d``.


Finally, we monitor the conservation of charges/ions:

.. literalinclude:: double_layer_relaxation.py
   :language: python
   :start-at: # Ion conservation: with blocking walls
   :end-at: charge=var("charge_density"))

Across all runs the sum drifts by less than :math:`10^{-15}` relative, i.e. it is conserved to the Newton tolerance rather than to the order of the time stepping. 

..  figure:: double_layer_relaxation.*
	:name: figdoublelayerrelaxation
	:align: center
	:alt: Charging and relaxation of an electric double layer in a liquid film facing a gas gap.
	:class: with-shadow
	:width: 100%

	An electrolyte film between an electrode and a gas gap, after switching on 5 V. (a) the approach of the diffuse-layer potential to its final value, for four bulk concentrations, against time in units of the Debye time; all four are single exponentials and all four collapse. (b) the measured relaxation time against the three candidate time scales. (c) the diffuse-layer potential with ions adsorbing on the liquid-gas interface, for three different site densities; above about :math:`c_0\lambda_\mathrm{D}` the adsorbed charge overturns the field-driven layer.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <double_layer_relaxation.py>`

		:download:`Download all examples <../../tutorial_example_scripts.zip>`
