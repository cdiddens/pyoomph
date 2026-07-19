Definition of a pure gaseous substance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let us start by defining the parameters of air, where we consider - despite knowing better - that air is just a pure substance. This is acceptable, if we want to mix water vapor with air to solve e.g. a vapor diffusion problem. But of course, we can also mix nitrogen with oxygen and possibly argon and more pure species if we want to resolve the local composition of the air, but this will be rarely necessary. To define air as a pure gaseous substance, we do the following:

.. literalinclude:: materials_pure_gas.py
   :language: python
   :start-at: from pyoomph.materials import * # Import the material API
   :end-at: self.mass_density=1.225*kilogram/meter**3 # Mass density

As you can see, we first have to import :py:mod:`pyoomph.materials` to get access to the material library and the base classes, as e.g. :py:class:`~pyoomph.materials.generic.PureGasProperties`. With the decorator :py:meth:`~pyoomph.materials.generic.MaterialProperties.register` of the generic :py:class:`~pyoomph.materials.generic.MaterialProperties` material class, pyoomph is instructed to register this material to the library. This registration is not persistent, i.e. you have to declare the class ``PureGasAir`` with the decorator in every piece of code you want to use this material. However, you can put all your material definitions in a separate python file and ``import`` it. Some example materials are already defined in the file :py:mod:`pyoomph.materials.default_materials`.

We set the ``name`` of our substance to ``"air"``, which is used as identifier of this gas.

.. warning::

   The ``name`` of materials may not contain any spaces or symbols (except the underscore and numeric characters). So ``name="1,2-hexanediol"`` is not possible, but ``name="12_hexanediol"`` is. The reason is that the name of the materials may also occur in the generated C code and C does not allow for arbitrary variable names.

Afterwards, we set the default properties in the constructor. For a gas, the molar mass (to convert e.g. from mole to mass fractions) and the dynamic viscosity and mass density (to solve flow problems) are relevant. Here, we set the values to constants, which of course limits the applicability of our material definition. At ambient pressures or temperatures deviating from the standard room conditions, this is hence not very accurate. We will improve that very soon by allowing pressure and temperature dependence.

Once the gas is defined, we can access this as follows

.. literalinclude:: materials_pure_gas.py
   :language: python
   :start-at: # Since the material is registered as pure gaseous material, we can load it as follows
   :end-at: print("Mass densities",air.mass_density,air2.mass_density) # Compare the densities

You can hence create an instance of the pure gas ``"air"`` by calling :py:func:`~pyoomph.materials.generic.get_pure_gas`. You can just access all the properties, but you can also change them. This can be useful to e.g. investigate the influence of some parameters. Note that in the above example, the second instance of the gas, i.e. ``air2``, will not be affected by this change, i.e. the mass density will still be ``1.225*kilogram/meter**3``. Thereby, you can e.g. modify the properties by hand only for one physical domain while keeping the other occurrences of the very same substance untouched.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <materials_pure_gas.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		   