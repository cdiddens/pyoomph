Liquid mixtures
~~~~~~~~~~~~~~~

Properties of liquid mixtures are defined similarly to gas mixtures. Again, we define the required components for this particular mixture and can select one species as passive field, i.e. the composition field which is not explicitly solved for. When we define a pure liquid named ``"glycerol"`` analogous to the pure liquid ``"water"``, we can define the mixture properties e.g. as follows:

.. literalinclude:: materials_liquids.py
   :language: python
   :start-after: self.set_unifac_groups({"CH2(hydroxy)": 2, "CH(hydroxy)": 1, "OH(new)": 3}, only_for="AIOMFAC")
   :end-at: self.set_activity_coefficients_by_unifac("AIOMFAC")

Again, as in the case of gas mixtures, the :py:attr:`~pyoomph.materials.generic.MaterialProperties.components` and :py:attr:`~pyoomph.materials.generic.BaseMixedProperties.passive_field` must be set. The constructor takes again a ``dict`` of the pure properties.

If one does not know details on the particular change of the liquid properties with the composition, one always can use :py:meth:`~pyoomph.materials.generic.BaseMixedProperties.set_by_weighted_average` to calculate the average of the pure properties weighted by the local mass fractions. This makes at least sure that the properties are correct when taking the pure limits. One can also modify the optional argument ``fraction_type`` to ``"mole_fraction"`` to blend between the pure properties weighted by the mole fractions instead of the mass fractions. The local mass and mole fractions of each component can be obtained by :py:meth:`~pyoomph.materials.generic.BaseMixedProperties.get_mass_fraction_field` and :py:meth:`~pyoomph.materials.generic.BaseMixedProperties.get_mole_fraction_field`, respectively. Alternatively, one can directly use e.g. ``var("massfrac_water")`` or ``var("molefrac_glycerol")`` to bind these fields to form arbitrary expressions.

As shown in the above example for the :py:meth:`~pyoomph.materials.generic.BaseLiquidProperties.dynamic_viscosity`, one can assemble functions of the composition and temperature easily. Here, we have used a viscosity model developed by :cite:t:`Cheng2008`, while the surface tension was obtained by a fit of experimental data :cite:`Takamura2012`. The same holds true for the diffusion coefficient based on the data of :cite:t:`DErrico2004`.

We can set the activity coefficients either directly by setting the ``dict`` values of the member :py:attr:`~pyoomph.materials.generic.MixtureLiquidProperties.activity_coefficients`, e.g. ``activity_coefficients["water"]=...``. If the vapor pressure shall be calculated by Raoult's law (cf. :math:numref:`eqmcflowraoults` later on), one has to call :py:meth:`~pyoomph.materials.generic.MixtureLiquidProperties.set_vapor_pressure_by_raoults_law` afterwards. Alternatively, the vapor pressure of each component can be set directly by the ``dict`` :py:attr:`~pyoomph.materials.generic.MixtureLiquidProperties.vapor_pressure_for`, e.g. ``vapor_pressure_for["water"]=...``. One can also invoke the various UNIFAC models to calculate the activity coefficients and set the vapor pressure according to Raoult's law with these activity coefficients. To that end, a simple call of :py:meth:`~pyoomph.materials.generic.MixtureLiquidProperties.set_activity_coefficients_by_unifac` will do the trick. One has to select a particular UNIFAC model (``"Original"``, ``"Dortmund"`` or ``"AIOMFAC"``). Of course, to use these models, one has to set the group contributions in the pure liquids (cf. :numref:`secmcflowunifac` later on).

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <materials_liquids.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		   