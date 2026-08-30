.. _secmcflowsalts:

Salts and ions
--------------

Salts are the one solute that cannot be treated as just another mixture component: they dissociate, so what is really dissolved are ions, and both their transport and their thermodynamics have to account for that. pyoomph therefore keeps a library of ions and salts of its own, transports a salt as an electroneutral pair, and reaches for the electrolyte extension of AIOMFAC when the activity coefficients of a brine are wanted. The following describes how to obtain, dissolve and define salts and ions, then applies the two ways of accounting for a dissolved salt to an evaporating capillary, and finally resolves what the electroneutral description averages over: the electric double layer itself.

.. toctree::
   :maxdepth: 5
   :hidden:

   salts/define.rst
   salts/capillary.rst
   salts/double_layer.rst
