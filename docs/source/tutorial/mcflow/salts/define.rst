.. _secmcflowsaltsdefine:

Defining and using salts and ions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The common ions and salts are registered by importing :py:mod:`pyoomph.materials.ions`, after which :py:func:`~pyoomph.materials.generic.get_ion` and :py:func:`~pyoomph.materials.generic.get_salt` fetch them by name, exactly as :py:func:`~pyoomph.materials.generic.get_pure_liquid` fetches a solvent:

.. code-block:: python

   from pyoomph.materials import *
   import pyoomph.materials.default_materials
   import pyoomph.materials.ions               # registers the ions and the salts

   Na = get_ion("Na+")                         # charge number, diffusivity, molar mass
   CaCl2 = get_salt("CaCl2")

A salt is a recipe rather than a substance here: it knows its two ions and the ratio in which they occur. That ratio is given from the charge numbers, since it is the only one that makes the solution electroneutral, so ``CaCl2.cation_stoichiometry`` is 1 and ``CaCl2.anion_stoichiometry``. Its molar mass follows from the same two ions. An ion carries its charge number and the limiting molar conductivity from which its diffusivity is obtained. A salt additionally provides the *ambipolar* diffusivity :py:meth:`~pyoomph.materials.generic.SaltProperties.get_ambipolar_diffusivity`, i.e. the rate at which the ion pair diffuses as a whole, when treating salts as sort of additional component instead of solving the detailled :py:meth:`~pyoomph.equations.electrostatics.NernstPlanckEquations` (see below).

A salt is dissolved by multiplying it by a concentration and adding it to a mixture. The unit ``molar`` (mol/litre) is available for this:

.. code-block:: python

   water, glycerol = get_pure_liquid("water"), get_pure_liquid("glycerol")
   mixture = Mixture(water + 20*percent*glycerol + 1*milli*molar*get_salt("NaCl"))

By default, salts do not sum to the mass fractions of the solvents: in the example, glycerol and water will sum up to unity, and the salt is described by its concentration instead. At 1 mM a salt is about :math:`6\cdot 10^{-5}` of the solution by mass, so this is a good description up to a few tenths molar.

The same ``<concentration>*<material>`` syntax also works for any *other* pure liquid -- a soluble surfactant, say -- but means something different there: such a component is always a genuine component of the mixture and never a dilute rider, so it has no ``salt_treatment`` analogue. Combining the two under ``salt_treatment="component"`` is consistent, but not without consequence: the salt then takes up a share of the volume, which dilutes anything given by a concentration alongside it by the factor :math:`1-c_\text{s}V_{\phi,\text{s}}`.

The ions in the material can be read back with :py:meth:`~pyoomph.materials.generic.BaseLiquidProperties.get_ions` and :py:meth:`~pyoomph.materials.generic.BaseLiquidProperties.get_bulk_concentration`.

Defining additional ions or salts works as follows:

.. code-block:: python

   new_ion("Rb+", 1, limiting_molar_conductivity=77.8*siemens*(centi*meter)**2/mol,
           molar_mass=85.468*gram/mol)

   @MaterialProperties.register()
   class SaltRubidiumChloride(SaltProperties):
       name = "RbCl"
       cation_name = "Rb+"
       anion_name = "Cl-"
       surface_tension_increment = 1.5*milli*newton/meter/molar

The stoichiometry, the molar mass and the ambipolar diffusivity of ``"RbCl"`` all follow from its two ions, so nothing else has to be supplied. Optional data such as the surface tension increment above may be given.  If absent, the salt does not contributes to Marangoni stresses.

Of the three group contribution models of :numref:`secmcflowunifac`, only AIOMFAC has been extended to electrolytes :cite:`Zuend2008,Zuend2011`, so it is the only one that can say anything about a salt solution. A *middle-range* term describes the interactions between the ions and the neutral functional groups, and between the ions themselves; a *long-range* term is the Pitzer-Debye-Hückel contribution that makes a dilute solution obey the limiting law. Both act on the solvents as well as on the ions, by what a salt lowers the water activity and therefore the vapour pressure above a brine. Nothing extra has to be requested for this, i.e. :py:meth:`~pyoomph.materials.generic.MixtureLiquidProperties.set_activity_coefficients_by_unifac` uses the electrolyte version by itself:

.. code-block:: python

   mixture.set_activity_coefficients_by_unifac("AIOMFAC")

   mixture.activity_coefficients["water"]                  # now depends on the salt
   mixture.get_ion_activity_coefficient("Na+")             # molality based, as AIOMFAC reports it
   mixture.get_mean_ionic_activity_coefficient("NaCl")     # what can actually be measured

The solvent coefficients keep their usual meaning -- multiplied by the mole fraction they give the activity, and the vapour pressures follow from Raoult's law as before -- while the ionic ones are on the molality scale with infinite dilution in pure water as reference, which is the convention AIOMFAC reports and the literature tabulates. A single ion's activity coefficient is not a measurable quantity on its own; the mean ionic one is. Note that AIOMFAC has middle-range parameters for a particular set of ions, and pyoomph's ion library is larger than that set. An ion AIOMFAC has no parameters for is refused.

Once a material carries salts, :py:func:`~pyoomph.equations.multi_component.CompositionFlowEquations` transports them automatically: each salt gets one concentration field :math:`c_s` obeying

.. math:: :label: eqmcflowsalttransport

   \partial_t c_s+\nabla\cdot\left(c_s\vec{u}-D_s\nabla c_s\right)=0

with :math:`D_s` the ambipolar diffusivity, and the ion concentrations :math:`c_i=\sum_s\nu_{i,s}c_s` follow by stoichiometry, so the solution is electroneutral by construction. There is deliberately no electric potential in this -- for that, i.e. :py:func:`~pyoomph.equations.electrostatics.PoissonNernstPlanck` must be used instead. 

As mentioned above, the dilute description is valid only in a limited concentration range. This can be modified with the ``salt_treatment`` argument. With the default ``"dilute"``, the concentration merely rides along: the solvent mass fractions still sum to unity among themselves, the mass density is that of the salt-free solvent and the salt occupies no volume. With ``salt_treatment="component"``, :py:meth:`~pyoomph.materials.generic.BaseLiquidProperties.treat_salts_as_components` upgrades the material in place, so that the salt becomes an ordinary composition field with a mass fraction of its own, a mole fraction and a share of the volume:

.. code-block:: python

   eqs = CompositionFlowEquations(mixture)                              # salt as a dilute solute
   eqs = CompositionFlowEquations(mixture, salt_treatment="component")  # ... or as a real component

The density is then volume-additive,

.. math:: :label: eqmcflowsaltdensity

   \frac{1}{\rho}=\frac{w_\mathrm{solv}}{\rho_\mathrm{solv}}+\sum_s\frac{w_s V_{\phi,s}}{M_s}\,,

with :math:`V_{\phi,s}` the apparent molar volume of the salt and :math:`\rho_\mathrm{solv}` the solvent correlation evaluated at the *renormalised*, salt-free composition. The concentration field :math:`c_s` still exists in this case, but as a substituted field computed from the mass fraction. 

.. important::

   When publishing results obtained with the electrolyte part of AIOMFAC, please cite the papers listed at https://aiomfac.lab.mcgill.ca/citation.html, as for the UNIFAC models in :numref:`secmcflowunifac`.
